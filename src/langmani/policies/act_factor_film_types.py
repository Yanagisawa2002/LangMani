"""Portable contracts for the single M4.3b ACT-Mixed-FactorFiLM policy.

These records deliberately contain no LeRobot model objects.  They bind the
factorized oracle command, the two separated FiLM paths, one training run, and
future validation-only checkpoint selection to canonical JSON identities.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Self, cast

from langmani.datasets.identity import sha256_hex
from langmani.environments.specs import BIN_IDS, OBJECT_IDS, TaskSpec, stable_task_id
from langmani.policies.act_types import (
    ACT_ACTION_COMPONENTS,
    ACT_STATE_COMPONENTS,
    ActDataConfig,
    ActModelConfig,
    ActOptimizationConfig,
    CheckpointRecord,
    TrainingState,
)

FACTOR_FILM_ARCHITECTURE_NAME = "ACT-Mixed-FactorFiLM"
FACTOR_FILM_SCHEMA_VERSION = "langmani-m43-factor-film-v0"
FACTOR_FILM_ARCHITECTURE_SCHEMA_VERSION = "langmani-m43-factor-film-architecture-identity-v0"
FACTOR_FILM_RUN_SCHEMA_VERSION = "langmani-m43-factor-film-run-v0"
FACTOR_FILM_CHECKPOINT_SCHEMA_VERSION = "langmani-m43-factor-film-checkpoint-contract-v0"
FACTOR_FILM_DRY_RUN_SCHEMA_VERSION = "langmani-m43-factor-film-dry-run-v0"
FACTOR_FILM_SELECTION_SCHEMA_VERSION = "langmani-m43-factor-film-selection-v0"
FACTOR_FILM_MANIFEST_SCHEMA_VERSION = "langmani-m43-factor-film-training-manifest-v0"
FACTOR_FILM_VALIDATION_QUEUE_SCHEMA_VERSION = "langmani-m43-factor-film-validation-queue-v0"
TARGET_OBJECT_MAPPING_VERSION = "TargetObjectConditionV0"
DESTINATION_BIN_MAPPING_VERSION = "DestinationBinConditionV0"
TARGET_OBJECT_IDS: tuple[str, ...] = ("red_cube", "green_cube", "blue_cube")
DESTINATION_BIN_IDS: tuple[str, ...] = ("left_bin", "right_bin")
FACTOR_FILM_OBJECT_INDEX_KEY = "langmani.factor_film.target_object_index"
FACTOR_FILM_BIN_INDEX_KEY = "langmani.factor_film.destination_bin_index"
FACTOR_FILM_VISUAL_INJECTION = "resnet18_layer4_feature_map_before_act_image_projection"
FACTOR_FILM_STATE_INJECTION = "panda_state_linear_projection_before_transformer_encoder"
FACTOR_FILM_INITIALIZATION = "final_projection_weight_normal_std_1e-5_bias_zero"
SUPPORTED_LEROBOT_VERSION = "0.6.0"
FACTOR_FILM_TARGET_MODEL_CONFIG_FINGERPRINT = (
    "sha256:2d852a356e1fa6243f288267126b9703d2b2df86a201fcc4ae1160f4a3319acc"
)
FACTOR_FILM_TARGET_OPTIMIZATION_CONFIG_FINGERPRINT = (
    "sha256:5a3d8c1c01db8186df3884cef72f76de31a7bce09afd5afcc7b041b0cc0463f5"
)
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_RE = re.compile(r"^[0-9a-f]{40,64}$")

if tuple(OBJECT_IDS) != TARGET_OBJECT_IDS:  # pragma: no cover - import invariant
    raise RuntimeError("M1 target-object order changed")
if tuple(BIN_IDS) != DESTINATION_BIN_IDS:  # pragma: no cover - import invariant
    raise RuntimeError("M1 destination-bin order changed")


class FactorFiLMContractError(ValueError):
    """Raised when a portable M4.3b contract is malformed or incompatible."""


class _FrozenMapping(Mapping[str, object]):
    __slots__ = ("_items",)

    def __init__(self, value: Mapping[str, object]) -> None:
        if not all(isinstance(key, str) for key in value):
            raise TypeError("FactorFiLM JSON mappings require string keys")
        self._items = tuple((key, _freeze_json(item)) for key, item in sorted(value.items()))

    def __getitem__(self, key: str) -> object:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __deepcopy__(self, memo: object) -> _FrozenMapping:
        del memo
        return self


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return _FrozenMapping(cast(Mapping[str, object], value))
    if isinstance(value, tuple | list):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str, StrEnum)):
        return value
    raise TypeError(f"unsupported FactorFiLM JSON value {type(value).__name__}")


def _json_value(value: object) -> object:
    if isinstance(value, StrEnum):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    return value


class _JsonRecord:
    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], _json_value(asdict(self)))


def _digest(value: str, name: str) -> None:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise FactorFiLMContractError(f"{name} must be one prefixed lowercase SHA-256 digest")


def _positive_int(value: int, name: str, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < (0 if allow_zero else 1):
        raise FactorFiLMContractError(f"{name} is outside its declared bound")


def _rate(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise FactorFiLMContractError(f"{name} must be finite and lie in [0,1]")


def canonical_fingerprint(value: object) -> str:
    """Return the project canonical prefixed SHA-256 identity."""
    return f"sha256:{sha256_hex(_json_value(value))}"


@dataclass(frozen=True, slots=True)
class TargetObjectConditionMapping(_JsonRecord):
    version: str = TARGET_OBJECT_MAPPING_VERSION
    ordered_object_ids: tuple[str, ...] = TARGET_OBJECT_IDS

    def __post_init__(self) -> None:
        if self.version != TARGET_OBJECT_MAPPING_VERSION:
            raise FactorFiLMContractError("unknown target-object mapping version")
        if tuple(self.ordered_object_ids) != TARGET_OBJECT_IDS:
            raise FactorFiLMContractError("target-object order must be red, green, blue")

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = dict(value)
        ordered = payload.get("ordered_object_ids")
        if not isinstance(ordered, list | tuple):
            raise TypeError("ordered_object_ids must be an array")
        payload["ordered_object_ids"] = tuple(ordered)
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class DestinationBinConditionMapping(_JsonRecord):
    version: str = DESTINATION_BIN_MAPPING_VERSION
    ordered_bin_ids: tuple[str, ...] = DESTINATION_BIN_IDS

    def __post_init__(self) -> None:
        if self.version != DESTINATION_BIN_MAPPING_VERSION:
            raise FactorFiLMContractError("unknown destination-bin mapping version")
        if tuple(self.ordered_bin_ids) != DESTINATION_BIN_IDS:
            raise FactorFiLMContractError("destination-bin order must be left, right")

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = dict(value)
        ordered = payload.get("ordered_bin_ids")
        if not isinstance(ordered, list | tuple):
            raise TypeError("ordered_bin_ids must be an array")
        payload["ordered_bin_ids"] = tuple(ordered)
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class FactorizedTaskCondition(_JsonRecord):
    task_id: str
    target_object_id: str
    destination_bin_id: str
    target_object_index: int
    destination_bin_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise TypeError("task_id must be a non-empty stable task ID")
        if self.target_object_id not in TARGET_OBJECT_IDS:
            raise FactorFiLMContractError("unknown target-object ID")
        if self.destination_bin_id not in DESTINATION_BIN_IDS:
            raise FactorFiLMContractError("unknown destination-bin ID")
        if self.target_object_index != TARGET_OBJECT_IDS.index(self.target_object_id):
            raise FactorFiLMContractError("target-object index does not match its semantic ID")
        if self.destination_bin_index != DESTINATION_BIN_IDS.index(self.destination_bin_id):
            raise FactorFiLMContractError("destination-bin index does not match its semantic ID")
        expected_task_id = stable_task_id(
            TaskSpec(
                target_object_id=self.target_object_id,
                target_bin_id=self.destination_bin_id,
                instruction_template_id="canonical_v0",
            )
        )
        if self.task_id != expected_task_id:
            raise FactorFiLMContractError(
                "stable task ID does not match the declared target object and destination bin"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class FactorFiLMRuntimeInput(_JsonRecord):
    conditions: tuple[FactorizedTaskCondition, ...]
    dtype: str = "int64"

    def __post_init__(self) -> None:
        if not self.conditions or not all(
            isinstance(value, FactorizedTaskCondition) for value in self.conditions
        ):
            raise FactorFiLMContractError(
                "runtime input requires one or more factorized conditions"
            )
        if self.dtype != "int64":
            raise FactorFiLMContractError("FactorFiLM condition indices must use int64")

    @property
    def batch_size(self) -> int:
        return len(self.conditions)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = dict(value)
        conditions = payload.get("conditions")
        if not isinstance(conditions, list | tuple):
            raise TypeError("runtime conditions must be an array")
        payload["conditions"] = tuple(
            FactorizedTaskCondition.from_dict(cast(Mapping[str, object], item))
            for item in conditions
        )
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class FactorFiLMConfig(_JsonRecord):
    object_embedding_dimension: int = 32
    bin_embedding_dimension: int = 32
    visual_feature_channels: int = 512
    state_context_dimension: int = 512
    target_object_count: int = 3
    destination_bin_count: int = 2
    panda_state_dimension: int = ACT_STATE_COMPONENTS
    action_dimension: int = ACT_ACTION_COMPONENTS
    visual_injection_target: str = FACTOR_FILM_VISUAL_INJECTION
    state_injection_target: str = FACTOR_FILM_STATE_INJECTION
    gamma_beta_initialization: str = FACTOR_FILM_INITIALIZATION
    object_mapping: TargetObjectConditionMapping = field(
        default_factory=TargetObjectConditionMapping
    )
    bin_mapping: DestinationBinConditionMapping = field(
        default_factory=DestinationBinConditionMapping
    )
    schema_version: str = FACTOR_FILM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "object_embedding_dimension",
            "bin_embedding_dimension",
            "visual_feature_channels",
            "state_context_dimension",
        ):
            _positive_int(cast(int, getattr(self, name)), name)
        if (
            self.target_object_count,
            self.destination_bin_count,
            self.panda_state_dimension,
            self.action_dimension,
        ) != (3, 2, 9, 8):
            raise FactorFiLMContractError("FactorFiLM task/state/action dimensions changed")
        if (
            self.visual_injection_target != FACTOR_FILM_VISUAL_INJECTION
            or self.state_injection_target != FACTOR_FILM_STATE_INJECTION
        ):
            raise FactorFiLMContractError("FactorFiLM injection locations are immutable")
        if self.gamma_beta_initialization != FACTOR_FILM_INITIALIZATION:
            raise FactorFiLMContractError(
                "FactorFiLM must use the locked identity-like initialization"
            )
        if not isinstance(self.object_mapping, TargetObjectConditionMapping) or not isinstance(
            self.bin_mapping, DestinationBinConditionMapping
        ):
            raise TypeError("FactorFiLM mappings are malformed")
        if self.schema_version != FACTOR_FILM_SCHEMA_VERSION:
            raise FactorFiLMContractError("unknown FactorFiLM schema version")

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = dict(value)
        object_mapping = payload.pop("object_mapping")
        bin_mapping = payload.pop("bin_mapping")
        if not isinstance(object_mapping, Mapping) or not isinstance(bin_mapping, Mapping):
            raise TypeError("FactorFiLM mappings must be JSON objects")
        return cls(
            object_mapping=TargetObjectConditionMapping.from_dict(object_mapping),
            bin_mapping=DestinationBinConditionMapping.from_dict(bin_mapping),
            **payload,
        )


@dataclass(frozen=True, slots=True)
class FactorFiLMArchitectureIdentity(_JsonRecord):
    base_act_configuration_fingerprint: str
    factor_film_config: FactorFiLMConfig
    upstream_contract: Mapping[str, object]
    base_parameter_count: int
    factor_film_parameter_count: int
    parameter_count_increase: int
    lerobot_version: str = SUPPORTED_LEROBOT_VERSION
    architecture_name: str = FACTOR_FILM_ARCHITECTURE_NAME
    schema_version: str = FACTOR_FILM_ARCHITECTURE_SCHEMA_VERSION
    architecture_fingerprint: str = ""

    def __post_init__(self) -> None:
        _digest(self.base_act_configuration_fingerprint, "base ACT configuration fingerprint")
        if not isinstance(self.factor_film_config, FactorFiLMConfig):
            raise TypeError("factor_film_config must be a FactorFiLMConfig")
        object.__setattr__(self, "upstream_contract", _FrozenMapping(self.upstream_contract))
        for name in (
            "base_parameter_count",
            "factor_film_parameter_count",
            "parameter_count_increase",
        ):
            _positive_int(cast(int, getattr(self, name)), name)
        if (
            self.factor_film_parameter_count - self.base_parameter_count
            != self.parameter_count_increase
        ):
            raise FactorFiLMContractError("FactorFiLM parameter-count increase is inconsistent")
        if self.lerobot_version != SUPPORTED_LEROBOT_VERSION:
            raise FactorFiLMContractError("FactorFiLM supports only inspected LeRobot 0.6.0")
        if self.architecture_name != FACTOR_FILM_ARCHITECTURE_NAME:
            raise FactorFiLMContractError("unknown FactorFiLM architecture")
        if self.schema_version != FACTOR_FILM_ARCHITECTURE_SCHEMA_VERSION:
            raise FactorFiLMContractError("unknown architecture-identity schema")
        expected = canonical_fingerprint(self.identity_payload())
        if not self.architecture_fingerprint:
            object.__setattr__(self, "architecture_fingerprint", expected)
        elif self.architecture_fingerprint != expected:
            raise FactorFiLMContractError("architecture fingerprint mismatch")

    def identity_payload(self) -> dict[str, object]:
        payload = self.to_dict()
        payload.pop("architecture_fingerprint")
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = dict(value)
        config = payload.pop("factor_film_config")
        upstream = payload.get("upstream_contract")
        if not isinstance(config, Mapping) or not isinstance(upstream, Mapping):
            raise TypeError("architecture identity contains malformed nested records")
        return cls(factor_film_config=FactorFiLMConfig.from_dict(config), **payload)


class FactorFiLMTrainingMode(StrEnum):
    DRY_RUN = "dry_run"
    FIXTURE = "fixture"
    TARGET_DEVELOPMENT = "target_development"


@dataclass(frozen=True, slots=True)
class FactorFiLMTrainingConfig(_JsonRecord):
    data: ActDataConfig
    base_model: ActModelConfig
    optimization: ActOptimizationConfig
    semantic_audit_evidence_fingerprint: str
    training_seed: int = 0
    mode: FactorFiLMTrainingMode = FactorFiLMTrainingMode.TARGET_DEVELOPMENT
    execution_horizon: int = 10
    gripper_runtime_mode: str = "project"
    checkpoint_selection_source: str = "m3b_validation"
    device: str = "cuda"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", FactorFiLMTrainingMode(self.mode))
        if self.base_model.state_dimension != 9:
            raise FactorFiLMContractError("FactorFiLM keeps PandaPolicyStateV0 exactly 9D")
        if self.execution_horizon != 10 or self.gripper_runtime_mode != "project":
            raise FactorFiLMContractError("FactorFiLM must use the locked M4.2 runtime")
        if self.checkpoint_selection_source != "m3b_validation":
            raise FactorFiLMContractError("only M3B validation may select FactorFiLM checkpoints")
        if self.device not in {"cpu", "cuda"} or self.dtype != "float32":
            raise FactorFiLMContractError("unsupported FactorFiLM device/dtype")
        _positive_int(self.training_seed, "training_seed", allow_zero=True)
        _digest(self.semantic_audit_evidence_fingerprint, "semantic-audit evidence fingerprint")
        if self.mode is FactorFiLMTrainingMode.TARGET_DEVELOPMENT:
            if (
                canonical_fingerprint(self.base_model.to_dict())
                != FACTOR_FILM_TARGET_MODEL_CONFIG_FINGERPRINT
            ):
                raise FactorFiLMContractError(
                    "target development must retain the primary 9D ACT model configuration"
                )
            if (
                canonical_fingerprint(self.optimization.to_dict())
                != FACTOR_FILM_TARGET_OPTIMIZATION_CONFIG_FINGERPRINT
            ):
                raise FactorFiLMContractError(
                    "target development must retain the primary M4 optimization configuration"
                )
            if self.training_seed != 0:
                raise FactorFiLMContractError(
                    "target development must use the locked State-OneHot seed 0"
                )
            if self.device != "cuda":
                raise FactorFiLMContractError("target development requires CUDA")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = dict(value)
        data = payload.pop("data")
        model = payload.pop("base_model")
        optimization = payload.pop("optimization")
        if not all(isinstance(item, Mapping) for item in (data, model, optimization)):
            raise TypeError("FactorFiLM training nested configs must be JSON objects")
        return cls(
            data=ActDataConfig.from_dict(cast(Mapping[str, object], data)),
            base_model=ActModelConfig.from_dict(cast(Mapping[str, object], model)),
            optimization=ActOptimizationConfig.from_dict(cast(Mapping[str, object], optimization)),
            **payload,
        )


@dataclass(frozen=True, slots=True)
class FactorFiLMCheckpointContract(_JsonRecord):
    architecture_fingerprint: str
    base_act_configuration_fingerprint: str
    object_mapping_fingerprint: str
    bin_mapping_fingerprint: str
    train_statistics_fingerprint: str
    m3b_export_fingerprint: str
    m3b_split_manifest_digest: str
    execution_horizon: int = 10
    gripper_runtime_mode: str = "project"
    optimizer: str = "adamw"
    scheduler: str = "none"
    processors_persisted: bool = True
    rng_state_persisted: bool = True
    schema_version: str = FACTOR_FILM_CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "architecture_fingerprint",
            "base_act_configuration_fingerprint",
            "object_mapping_fingerprint",
            "bin_mapping_fingerprint",
            "train_statistics_fingerprint",
            "m3b_export_fingerprint",
            "m3b_split_manifest_digest",
        ):
            _digest(cast(str, getattr(self, name)), name)
        if self.optimizer != "adamw" or self.scheduler != "none":
            raise FactorFiLMContractError("FactorFiLM checkpoint optimizer contract changed")
        if self.execution_horizon != 10 or self.gripper_runtime_mode != "project":
            raise FactorFiLMContractError("FactorFiLM checkpoint runtime contract changed")
        if not self.processors_persisted or not self.rng_state_persisted:
            raise FactorFiLMContractError("FactorFiLM checkpoints must persist processors and RNG")
        if self.schema_version != FACTOR_FILM_CHECKPOINT_SCHEMA_VERSION:
            raise FactorFiLMContractError("unknown FactorFiLM checkpoint schema")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class FactorFiLMRunIdentity(_JsonRecord):
    m3b_export_fingerprint: str
    m3b_split_manifest_digest: str
    ordered_train_episode_indices: tuple[int, ...]
    ordered_validation_episode_indices: tuple[int, ...]
    architecture_identity: Mapping[str, object]
    model_config: Mapping[str, object]
    data_contract: Mapping[str, object]
    train_statistics_fingerprint: str
    optimization_config: Mapping[str, object]
    semantic_audit_evidence_fingerprint: str
    training_seed: int
    experiment_mode: FactorFiLMTrainingMode
    device: str
    dtype: str
    lerobot_version: str
    torch_version: str
    cuda_version: str | None
    git_commit: str
    git_dirty: bool
    schema_version: str = FACTOR_FILM_RUN_SCHEMA_VERSION
    run_fingerprint: str = ""

    def __post_init__(self) -> None:
        for name in (
            "m3b_export_fingerprint",
            "m3b_split_manifest_digest",
            "train_statistics_fingerprint",
            "semantic_audit_evidence_fingerprint",
        ):
            _digest(cast(str, getattr(self, name)), name)
        object.__setattr__(self, "experiment_mode", FactorFiLMTrainingMode(self.experiment_mode))
        object.__setattr__(
            self, "architecture_identity", _FrozenMapping(self.architecture_identity)
        )
        object.__setattr__(self, "model_config", _FrozenMapping(self.model_config))
        object.__setattr__(self, "data_contract", _FrozenMapping(self.data_contract))
        object.__setattr__(self, "optimization_config", _FrozenMapping(self.optimization_config))
        for name, indices in (
            ("train", self.ordered_train_episode_indices),
            ("validation", self.ordered_validation_episode_indices),
        ):
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in indices
            ):
                raise TypeError(f"FactorFiLM {name} episode indices must be nonnegative integers")
            if tuple(sorted(set(indices))) != indices:
                raise FactorFiLMContractError(
                    f"FactorFiLM {name} episode indices must be strictly increasing and unique"
                )
        if not self.ordered_train_episode_indices:
            raise FactorFiLMContractError("FactorFiLM run requires train episodes")
        if set(self.ordered_train_episode_indices) & set(self.ordered_validation_episode_indices):
            raise FactorFiLMContractError("FactorFiLM train and validation episodes overlap")
        if self.lerobot_version != SUPPORTED_LEROBOT_VERSION:
            raise FactorFiLMContractError("run uses an uninspected LeRobot version")
        if _GIT_RE.fullmatch(self.git_commit) is None:
            raise FactorFiLMContractError("run requires a full lowercase Git commit")
        if self.experiment_mode is FactorFiLMTrainingMode.TARGET_DEVELOPMENT and self.git_dirty:
            raise FactorFiLMContractError("target-development FactorFiLM requires clean Git")
        if self.experiment_mode is FactorFiLMTrainingMode.TARGET_DEVELOPMENT:
            if (
                len(self.ordered_train_episode_indices),
                len(self.ordered_validation_episode_indices),
            ) != (288, 36):
                raise FactorFiLMContractError(
                    "target-development FactorFiLM requires exactly 288 train and 36 validation episodes"
                )
            expected_episode_contract = {
                "train_episode_count": 288,
                "validation_episode_count": 36,
                "train_episode_indices_fingerprint": canonical_fingerprint(
                    self.ordered_train_episode_indices
                ),
                "validation_episode_indices_fingerprint": canonical_fingerprint(
                    self.ordered_validation_episode_indices
                ),
            }
            if any(
                self.data_contract.get(name) != expected
                for name, expected in expected_episode_contract.items()
            ):
                raise FactorFiLMContractError(
                    "target-development episode identity differs from its data contract"
                )
        if self.device not in {"cpu", "cuda"} or self.dtype != "float32":
            raise FactorFiLMContractError("run device/dtype is incompatible")
        if self.schema_version != FACTOR_FILM_RUN_SCHEMA_VERSION:
            raise FactorFiLMContractError("unknown FactorFiLM run schema")
        _positive_int(self.training_seed, "training_seed", allow_zero=True)
        forbidden = repr(self.to_dict()).lower()
        if any(value in forbidden for value in ("m42_final_v0", "fresh_seed", "m3b_test")):
            raise FactorFiLMContractError("FactorFiLM run identity contains a forbidden source")
        expected = canonical_fingerprint(self.identity_payload())
        if not self.run_fingerprint:
            object.__setattr__(self, "run_fingerprint", expected)
        elif self.run_fingerprint != expected:
            raise FactorFiLMContractError("FactorFiLM run fingerprint mismatch")

    def identity_payload(self) -> dict[str, object]:
        payload = self.to_dict()
        payload.pop("run_fingerprint")
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = dict(value)
        for name in ("ordered_train_episode_indices", "ordered_validation_episode_indices"):
            indices = payload.get(name)
            if not isinstance(indices, list | tuple):
                raise TypeError(f"{name} must be an array")
            payload[name] = tuple(indices)
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class FactorFiLMTrainingManifest(_JsonRecord):
    identity: FactorFiLMRunIdentity
    training_config: FactorFiLMTrainingConfig
    architecture_identity: FactorFiLMArchitectureIdentity
    checkpoint_contract: FactorFiLMCheckpointContract
    training_state: TrainingState
    checkpoints: tuple[CheckpointRecord, ...] = ()
    training_complete: bool = False
    schema_version: str = FACTOR_FILM_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.identity, FactorFiLMRunIdentity):
            raise TypeError("manifest identity must be FactorFiLMRunIdentity")
        if not isinstance(self.training_config, FactorFiLMTrainingConfig):
            raise TypeError("manifest training_config is malformed")
        if not isinstance(self.architecture_identity, FactorFiLMArchitectureIdentity):
            raise TypeError("manifest architecture identity is malformed")
        if not isinstance(self.checkpoint_contract, FactorFiLMCheckpointContract):
            raise TypeError("manifest checkpoint contract is malformed")
        if not isinstance(self.training_state, TrainingState):
            raise TypeError("manifest training state is malformed")
        if (
            self.identity.semantic_audit_evidence_fingerprint
            != self.training_config.semantic_audit_evidence_fingerprint
        ):
            raise FactorFiLMContractError("manifest semantic-audit fingerprint changed")
        if self.identity.training_seed != self.training_config.training_seed:
            raise FactorFiLMContractError("manifest training seed changed")
        if self.identity.experiment_mode is not self.training_config.mode:
            raise FactorFiLMContractError("manifest experiment mode changed")
        if dict(self.identity.optimization_config) != self.training_config.optimization.to_dict():
            raise FactorFiLMContractError("manifest optimization configuration changed")
        if canonical_fingerprint(self.identity.architecture_identity) != canonical_fingerprint(
            self.architecture_identity.to_dict()
        ):
            raise FactorFiLMContractError("manifest architecture identity changed")
        if self.identity.experiment_mode is FactorFiLMTrainingMode.TARGET_DEVELOPMENT:
            base_act = self.identity.model_config.get("base_act")
            expected_data_contract = {
                "checkpoint_selection_source": self.training_config.checkpoint_selection_source,
                "execution_horizon": self.training_config.execution_horizon,
                "gripper_runtime_mode": self.training_config.gripper_runtime_mode,
            }
            if canonical_fingerprint(base_act) != canonical_fingerprint(
                self.training_config.base_model.to_dict()
            ) or any(
                self.identity.data_contract.get(name) != expected
                for name, expected in expected_data_contract.items()
            ):
                raise FactorFiLMContractError("target manifest model/runtime data contract changed")
        checkpoint_contract_fields = (
            (
                self.checkpoint_contract.architecture_fingerprint,
                self.architecture_identity.architecture_fingerprint,
            ),
            (
                self.checkpoint_contract.train_statistics_fingerprint,
                self.identity.train_statistics_fingerprint,
            ),
            (
                self.checkpoint_contract.m3b_export_fingerprint,
                self.identity.m3b_export_fingerprint,
            ),
            (
                self.checkpoint_contract.m3b_split_manifest_digest,
                self.identity.m3b_split_manifest_digest,
            ),
            (
                self.checkpoint_contract.base_act_configuration_fingerprint,
                self.architecture_identity.base_act_configuration_fingerprint,
            ),
            (
                self.checkpoint_contract.object_mapping_fingerprint,
                self.architecture_identity.factor_film_config.object_mapping.fingerprint,
            ),
            (
                self.checkpoint_contract.bin_mapping_fingerprint,
                self.architecture_identity.factor_film_config.bin_mapping.fingerprint,
            ),
        )
        if any(actual != expected for actual, expected in checkpoint_contract_fields):
            raise FactorFiLMContractError("manifest checkpoint contract provenance changed")
        if any(
            value.run_fingerprint != self.identity.run_fingerprint for value in self.checkpoints
        ):
            raise FactorFiLMContractError("manifest checkpoint belongs to another run")
        expected_policy_digest = canonical_fingerprint(self.identity.model_config)
        for checkpoint in self.checkpoints:
            expected_fields = (
                (checkpoint.policy_config_digest, expected_policy_digest),
                (
                    checkpoint.statistics_fingerprint,
                    self.identity.train_statistics_fingerprint,
                ),
                (checkpoint.split_digest, self.identity.m3b_split_manifest_digest),
                (checkpoint.git_commit, self.identity.git_commit),
            )
            if not checkpoint.complete or any(
                actual != expected for actual, expected in expected_fields
            ):
                raise FactorFiLMContractError("manifest checkpoint provenance changed")
        steps = tuple(value.global_step for value in self.checkpoints)
        if steps != tuple(sorted(set(steps))):
            raise FactorFiLMContractError("manifest checkpoint steps must be ordered and unique")
        if self.checkpoints:
            if self.training_state.global_step != self.checkpoints[-1].global_step:
                raise FactorFiLMContractError(
                    "manifest state step differs from its last checkpoint"
                )
            if (
                self.training_state.last_checkpoint_fingerprint
                != self.checkpoints[-1].checkpoint_fingerprint
            ):
                raise FactorFiLMContractError(
                    "manifest state fingerprint differs from its last checkpoint"
                )
        elif (
            self.training_state.global_step != 0
            or self.training_state.last_checkpoint_fingerprint is not None
        ):
            raise FactorFiLMContractError("checkpoint-free manifest must remain at step zero")
        if self.training_complete:
            if self.training_state.global_step != self.training_config.optimization.training_steps:
                raise FactorFiLMContractError("completed manifest has the wrong final step")
            if (
                self.training_config.mode is FactorFiLMTrainingMode.TARGET_DEVELOPMENT
                and steps != tuple(range(5_000, 100_001, 5_000))
            ):
                raise FactorFiLMContractError(
                    "completed target manifest requires the exact 20-checkpoint schedule"
                )
            if not self.training_state.completed:
                raise FactorFiLMContractError("completed manifest requires completed state")
        elif self.training_state.completed:
            raise FactorFiLMContractError("incomplete manifest cannot contain completed state")
        if self.schema_version != FACTOR_FILM_MANIFEST_SCHEMA_VERSION:
            raise FactorFiLMContractError("unknown FactorFiLM training-manifest schema")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = dict(value)
        identity = payload.pop("identity")
        training_config = payload.pop("training_config")
        architecture = payload.pop("architecture_identity")
        checkpoint_contract = payload.pop("checkpoint_contract")
        training_state = payload.pop("training_state")
        checkpoints = payload.pop("checkpoints")
        nested = (identity, training_config, architecture, checkpoint_contract, training_state)
        if not all(isinstance(item, Mapping) for item in nested):
            raise TypeError("manifest nested records must be JSON objects")
        if not isinstance(checkpoints, list | tuple):
            raise TypeError("manifest checkpoints must be an array")
        return cls(
            identity=FactorFiLMRunIdentity.from_dict(cast(Mapping[str, object], identity)),
            training_config=FactorFiLMTrainingConfig.from_dict(
                cast(Mapping[str, object], training_config)
            ),
            architecture_identity=FactorFiLMArchitectureIdentity.from_dict(
                cast(Mapping[str, object], architecture)
            ),
            checkpoint_contract=FactorFiLMCheckpointContract.from_dict(
                cast(Mapping[str, object], checkpoint_contract)
            ),
            training_state=TrainingState.from_dict(cast(Mapping[str, object], training_state)),
            checkpoints=tuple(
                CheckpointRecord.from_dict(cast(Mapping[str, object], item)) for item in checkpoints
            ),
            **payload,
        )


@dataclass(frozen=True, slots=True)
class FactorFiLMValidationQueueItem(_JsonRecord):
    checkpoint_fingerprint: str
    checkpoint_step: int
    checkpoint_relative_path: str
    offline_validation_action_loss: float
    source_split: str = "m3b_validation"

    def __post_init__(self) -> None:
        _digest(self.checkpoint_fingerprint, "checkpoint_fingerprint")
        _positive_int(self.checkpoint_step, "checkpoint_step")
        if (
            isinstance(self.offline_validation_action_loss, bool)
            or not isinstance(self.offline_validation_action_loss, int | float)
            or not math.isfinite(float(self.offline_validation_action_loss))
            or float(self.offline_validation_action_loss) < 0
        ):
            raise FactorFiLMContractError(
                "queued offline validation action loss must be finite and nonnegative"
            )
        if self.source_split != "m3b_validation":
            raise FactorFiLMContractError("validation queue contains a forbidden split")
        if (
            not isinstance(self.checkpoint_relative_path, str)
            or not self.checkpoint_relative_path
            or self.checkpoint_relative_path.startswith(("/", "\\"))
            or ".." in self.checkpoint_relative_path
        ):
            raise FactorFiLMContractError("validation queue checkpoint path is unsafe")


@dataclass(frozen=True, slots=True)
class FactorFiLMValidationQueue(_JsonRecord):
    run_fingerprint: str
    validation_schedule_digest: str
    items: tuple[FactorFiLMValidationQueueItem, ...]
    expected_checkpoint_count: int = 20
    selection_source: str = "m3b_validation"
    test_accessed: bool = False
    development_accessed_for_selection: bool = False
    semantic_audit_accessed_for_selection: bool = False
    final_schedule_accessed: bool = False
    schema_version: str = FACTOR_FILM_VALIDATION_QUEUE_SCHEMA_VERSION
    queue_fingerprint: str = ""

    def __post_init__(self) -> None:
        _digest(self.run_fingerprint, "run_fingerprint")
        _digest(self.validation_schedule_digest, "validation_schedule_digest")
        _positive_int(self.expected_checkpoint_count, "expected_checkpoint_count")
        if self.selection_source != "m3b_validation":
            raise FactorFiLMContractError("FactorFiLM validation queue changed its source")
        if not all(isinstance(value, FactorFiLMValidationQueueItem) for value in self.items):
            raise TypeError("validation queue items are malformed")
        if self.expected_checkpoint_count != 20 or len(self.items) != 20:
            raise FactorFiLMContractError("validation queue requires exactly 20 checkpoints")
        if len({value.checkpoint_step for value in self.items}) != len(self.items):
            raise FactorFiLMContractError("validation queue checkpoint steps must be unique")
        if len({value.checkpoint_fingerprint for value in self.items}) != len(self.items):
            raise FactorFiLMContractError("validation queue checkpoint fingerprints must be unique")
        if tuple(value.checkpoint_step for value in self.items) != tuple(
            range(5_000, 100_001, 5_000)
        ):
            raise FactorFiLMContractError(
                "validation queue requires the exact 5k-to-100k checkpoint schedule"
            )
        if any(
            (
                self.test_accessed,
                self.development_accessed_for_selection,
                self.semantic_audit_accessed_for_selection,
                self.final_schedule_accessed,
            )
        ):
            raise FactorFiLMContractError("validation queue accessed a forbidden source")
        if self.schema_version != FACTOR_FILM_VALIDATION_QUEUE_SCHEMA_VERSION:
            raise FactorFiLMContractError("unknown FactorFiLM validation-queue schema")
        expected = canonical_fingerprint(self.identity_payload())
        if not self.queue_fingerprint:
            object.__setattr__(self, "queue_fingerprint", expected)
        elif self.queue_fingerprint != expected:
            raise FactorFiLMContractError("validation queue fingerprint mismatch")

    def identity_payload(self) -> dict[str, object]:
        payload = self.to_dict()
        payload.pop("queue_fingerprint")
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = dict(value)
        items = payload.pop("items")
        if not isinstance(items, list | tuple):
            raise TypeError("validation queue items must be an array")
        return cls(
            items=tuple(
                FactorFiLMValidationQueueItem(**cast(Mapping[str, object], item)) for item in items
            ),
            **payload,
        )


@dataclass(frozen=True, slots=True)
class FactorFiLMValidationResult(_JsonRecord):
    checkpoint_fingerprint: str
    checkpoint_step: int
    source_split: str
    schedule_digest: str
    task_success_rate: float
    wrong_object_interaction_rate: float
    wrong_object_in_target_bin_rate: float
    target_off_table_rate: float
    timeout_rate: float
    offline_validation_action_loss: float

    def __post_init__(self) -> None:
        _digest(self.checkpoint_fingerprint, "checkpoint_fingerprint")
        _digest(self.schedule_digest, "schedule_digest")
        _positive_int(self.checkpoint_step, "checkpoint_step")
        if self.source_split != "m3b_validation":
            raise FactorFiLMContractError("FactorFiLM checkpoint selection is validation-only")
        for name in (
            "task_success_rate",
            "wrong_object_interaction_rate",
            "wrong_object_in_target_bin_rate",
            "target_off_table_rate",
            "timeout_rate",
        ):
            _rate(cast(float, getattr(self, name)), name)
        if (
            not math.isfinite(self.offline_validation_action_loss)
            or self.offline_validation_action_loss < 0
        ):
            raise FactorFiLMContractError(
                "offline validation action loss must be finite and nonnegative"
            )

    @property
    def ranking_key(self) -> tuple[float, float, float, float, float, float, int]:
        return (
            -self.task_success_rate,
            self.wrong_object_interaction_rate,
            self.wrong_object_in_target_bin_rate,
            self.target_off_table_rate,
            self.timeout_rate,
            self.offline_validation_action_loss,
            self.checkpoint_step,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class FactorFiLMSelectionRecord(_JsonRecord):
    run_fingerprint: str
    validation_queue_fingerprint: str
    validation_queue: FactorFiLMValidationQueue
    validation_schedule_digest: str
    candidates: tuple[FactorFiLMValidationResult, ...]
    ranked_checkpoint_fingerprints: tuple[str, ...]
    selected_checkpoint_fingerprint: str
    selected_checkpoint_step: int
    selection_locked: bool = True
    test_accessed: bool = False
    development_accessed_for_selection: bool = False
    semantic_audit_accessed_for_selection: bool = False
    final_schedule_accessed: bool = False
    schema_version: str = FACTOR_FILM_SELECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _digest(self.run_fingerprint, "run_fingerprint")
        _digest(self.validation_queue_fingerprint, "validation_queue_fingerprint")
        _digest(self.validation_schedule_digest, "validation_schedule_digest")
        if not isinstance(self.validation_queue, FactorFiLMValidationQueue):
            raise TypeError("selection requires the published validation queue")
        if (
            self.validation_queue.run_fingerprint != self.run_fingerprint
            or self.validation_queue.queue_fingerprint != self.validation_queue_fingerprint
            or self.validation_queue.validation_schedule_digest != self.validation_schedule_digest
        ):
            raise FactorFiLMContractError("selection validation-queue identity changed")
        if not self.candidates:
            raise FactorFiLMContractError("selection requires validation candidates")
        if any(
            value.schedule_digest != self.validation_schedule_digest for value in self.candidates
        ):
            raise FactorFiLMContractError("selection candidates use different schedules")
        ranked = tuple(sorted(self.candidates, key=lambda value: value.ranking_key))
        queued = {
            (
                value.checkpoint_fingerprint,
                value.checkpoint_step,
                float(value.offline_validation_action_loss),
            )
            for value in self.validation_queue.items
        }
        observed = {
            (
                value.checkpoint_fingerprint,
                value.checkpoint_step,
                float(value.offline_validation_action_loss),
            )
            for value in self.candidates
        }
        if observed != queued:
            raise FactorFiLMContractError(
                "selection candidates differ from the published validation queue"
            )
        if self.ranked_checkpoint_fingerprints != tuple(
            value.checkpoint_fingerprint for value in ranked
        ):
            raise FactorFiLMContractError("selection ranking does not match the declared order")
        if (
            self.selected_checkpoint_fingerprint != ranked[0].checkpoint_fingerprint
            or self.selected_checkpoint_step != ranked[0].checkpoint_step
        ):
            raise FactorFiLMContractError("selected checkpoint is not the validation winner")
        if not self.selection_locked or any(
            (
                self.test_accessed,
                self.development_accessed_for_selection,
                self.semantic_audit_accessed_for_selection,
                self.final_schedule_accessed,
            )
        ):
            raise FactorFiLMContractError("FactorFiLM selection used a forbidden source")
        if self.schema_version != FACTOR_FILM_SELECTION_SCHEMA_VERSION:
            raise FactorFiLMContractError("unknown FactorFiLM selection schema")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = dict(value)
        candidates = payload.pop("candidates")
        validation_queue = payload.pop("validation_queue")
        ranked = payload.get("ranked_checkpoint_fingerprints")
        if (
            not isinstance(candidates, list | tuple)
            or not isinstance(validation_queue, Mapping)
            or not isinstance(ranked, list | tuple)
        ):
            raise TypeError("selection candidates and ranking must be arrays")
        payload["ranked_checkpoint_fingerprints"] = tuple(ranked)
        return cls(
            candidates=tuple(
                FactorFiLMValidationResult.from_dict(cast(Mapping[str, object], item))
                for item in candidates
            ),
            validation_queue=FactorFiLMValidationQueue.from_dict(validation_queue),
            **payload,
        )


@dataclass(frozen=True, slots=True)
class FactorFiLMDryRunReport(_JsonRecord):
    dataset_fingerprint: str
    split_fingerprint: str
    train_episode_count: int
    validation_episode_count: int
    feature_shapes: Mapping[str, object]
    panda_policy_state_dimension: int
    object_mapping: Mapping[str, object]
    bin_mapping: Mapping[str, object]
    visual_film_injection: str
    state_film_injection: str
    base_act_parameter_count: int
    factor_film_parameter_count: int
    parameter_count_increase: int
    train_statistics_fingerprint: str
    effective_training_configuration: Mapping[str, object]
    run_fingerprint: str
    expected_output_path: str
    expected_checkpoint_steps: tuple[int, ...]
    semantic_audit_evidence_fingerprint: str
    fixture_contract: bool
    test_accessible: bool = False
    historical_fresh_accessible: bool = False
    final_schedule_accessed: bool = False
    training_started: bool = False
    schema_version: str = FACTOR_FILM_DRY_RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "dataset_fingerprint",
            "split_fingerprint",
            "train_statistics_fingerprint",
            "run_fingerprint",
            "semantic_audit_evidence_fingerprint",
        ):
            _digest(cast(str, getattr(self, name)), name)
        if (self.train_episode_count, self.validation_episode_count) != (
            288,
            36,
        ) and not self.fixture_contract:
            raise FactorFiLMContractError("real FactorFiLM dry-run requires 288/36 episodes")
        if self.panda_policy_state_dimension != 9:
            raise FactorFiLMContractError("FactorFiLM dry-run changed PandaPolicyStateV0")
        if (
            self.parameter_count_increase
            != self.factor_film_parameter_count - self.base_act_parameter_count
        ):
            raise FactorFiLMContractError("dry-run parameter counts are inconsistent")
        if len(self.expected_checkpoint_steps) != 20 and not self.fixture_contract:
            raise FactorFiLMContractError("target FactorFiLM requires exactly 20 checkpoints")
        if any(
            (
                self.test_accessible,
                self.historical_fresh_accessible,
                self.final_schedule_accessed,
                self.training_started,
            )
        ):
            raise FactorFiLMContractError("dry-run accessed training or sealed evaluation sources")
        if self.schema_version != FACTOR_FILM_DRY_RUN_SCHEMA_VERSION:
            raise FactorFiLMContractError("unknown FactorFiLM dry-run schema")
        object.__setattr__(self, "feature_shapes", _FrozenMapping(self.feature_shapes))
        object.__setattr__(self, "object_mapping", _FrozenMapping(self.object_mapping))
        object.__setattr__(self, "bin_mapping", _FrozenMapping(self.bin_mapping))
        object.__setattr__(
            self,
            "effective_training_configuration",
            _FrozenMapping(self.effective_training_configuration),
        )


__all__ = [
    "DESTINATION_BIN_IDS",
    "DESTINATION_BIN_MAPPING_VERSION",
    "FACTOR_FILM_ARCHITECTURE_NAME",
    "FACTOR_FILM_BIN_INDEX_KEY",
    "FACTOR_FILM_INITIALIZATION",
    "FACTOR_FILM_OBJECT_INDEX_KEY",
    "FACTOR_FILM_SCHEMA_VERSION",
    "FACTOR_FILM_STATE_INJECTION",
    "FACTOR_FILM_TARGET_MODEL_CONFIG_FINGERPRINT",
    "FACTOR_FILM_TARGET_OPTIMIZATION_CONFIG_FINGERPRINT",
    "FACTOR_FILM_VISUAL_INJECTION",
    "TARGET_OBJECT_IDS",
    "TARGET_OBJECT_MAPPING_VERSION",
    "DestinationBinConditionMapping",
    "FactorFiLMArchitectureIdentity",
    "FactorFiLMCheckpointContract",
    "FactorFiLMConfig",
    "FactorFiLMContractError",
    "FactorFiLMDryRunReport",
    "FactorFiLMRunIdentity",
    "FactorFiLMRuntimeInput",
    "FactorFiLMSelectionRecord",
    "FactorFiLMTrainingConfig",
    "FactorFiLMTrainingManifest",
    "FactorFiLMTrainingMode",
    "FactorFiLMValidationQueue",
    "FactorFiLMValidationQueueItem",
    "FactorFiLMValidationResult",
    "FactorizedTaskCondition",
    "TargetObjectConditionMapping",
    "canonical_fingerprint",
]
