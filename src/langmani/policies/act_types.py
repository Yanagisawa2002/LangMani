"""Immutable contracts for the M4 ACT baselines.

The types in this module intentionally contain no LeRobot imports.  They define
the portable, canonical inputs and compact result records that bind an M4 run
to one completed M3B dataset, one Git commit, and one declared evaluation
protocol.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Self, cast

from langmani.datasets.identity import sha256_hex
from langmani.datasets.lerobot_types import (
    ACTION_FEATURE_KEY,
    IMAGE_FEATURE_KEY,
    STATE_FEATURE_KEY,
)
from langmani.environments.specs import TaskSpec, stable_task_id
from langmani.policies.act_action_bounds import (
    ActionBoundMode,
    ActionProjectionSummary,
)

M4_SCHEMA_VERSION = "langmani-m4-act-v1"
TASK_ONEHOT_MAPPING_VERSION = "CanonicalTaskOneHotV0"
SUPPORTED_LEROBOT_VERSION = "0.6.0"
ACT_STATE_COMPONENTS = 9
ACT_CONDITIONED_STATE_COMPONENTS = 15
ACT_ACTION_COMPONENTS = 8
_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")


class _FrozenMapping(Mapping[str, object]):
    """Recursively immutable, deepcopy-safe canonical mapping."""

    __slots__ = ("_items",)

    def __init__(self, value: Mapping[str, object]) -> None:
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
        if not all(isinstance(key, str) for key in value):
            raise TypeError("immutable JSON mappings require string keys")
        return _FrozenMapping(cast(Mapping[str, object], value))
    if isinstance(value, tuple | list):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str, StrEnum)):
        return value
    raise TypeError(f"unsupported immutable JSON value {type(value).__name__}")


def _frozen_mapping(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return _FrozenMapping(value)


class ActVariant(StrEnum):
    """Exactly the three scientific controls declared for M4."""

    PER_TASK = "per_task"
    MIXED_UNCONDITIONED = "mixed_unconditioned"
    MIXED_TASK_ONEHOT = "mixed_task_onehot"


class ExperimentMode(StrEnum):
    """Execution modes with intentionally different acceptance meaning."""

    DRY_RUN = "dry_run"
    TINY_OVERFIT = "tiny_overfit"
    DEVELOPMENT = "development"
    FULL = "full"


class EvaluationSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    FRESH_SEED = "fresh_seed"


class RolloutStatus(StrEnum):
    SUCCESS = "success"
    TARGET_OFF_TABLE = "target_off_table"
    INVALID_ACTION = "invalid_action"
    INFERENCE_FAILURE = "inference_failure"
    ENVIRONMENT_FAILURE = "environment_failure"
    TRUNCATED = "truncated"
    TIMEOUT = "timeout"


def _positive_int(value: int, name: str, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _finite(value: float, name: str, *, minimum: float = 0.0) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(float(value)) or float(value) < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")


def _digest(value: str, name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


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
    """Small JSON conversion mixin for frozen records."""

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], _json_value(asdict(self)))


@dataclass(frozen=True, slots=True)
class ActModelConfig(_JsonRecord):
    """Every effective ACT architecture and temporal parameter."""

    image_feature_key: str = IMAGE_FEATURE_KEY
    state_feature_key: str = STATE_FEATURE_KEY
    action_feature_key: str = ACTION_FEATURE_KEY
    image_shape_chw: tuple[int, int, int] = (3, 256, 256)
    state_dimension: int = ACT_STATE_COMPONENTS
    action_dimension: int = ACT_ACTION_COMPONENTS
    normalization_mapping: Mapping[str, str] = field(
        default_factory=lambda: {
            "VISUAL": "MEAN_STD",
            "STATE": "MEAN_STD",
            "ACTION": "MEAN_STD",
        }
    )
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = None
    replace_final_stride_with_dilation: bool = False
    chunk_size: int = 50
    n_action_steps: int = 10
    n_obs_steps: int = 1
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4
    dropout: float = 0.1
    kl_weight: float = 10.0
    pre_norm: bool = False
    temporal_ensemble_coeff: float | None = None
    push_to_hub: bool = False
    use_peft: bool = False
    repo_id: str | None = None
    private: bool | None = None
    tags: tuple[str, ...] | None = None
    license: str | None = None
    pretrained_path: str | None = None
    pretrained_revision: str | None = None
    observation_delta_indices: tuple[int, ...] | None = None
    action_delta_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "normalization_mapping",
            _frozen_mapping(self.normalization_mapping, "normalization_mapping"),
        )
        if (self.image_feature_key, self.state_feature_key, self.action_feature_key) != (
            IMAGE_FEATURE_KEY,
            STATE_FEATURE_KEY,
            ACTION_FEATURE_KEY,
        ):
            raise ValueError("ACT feature names must use the M3B policy allowlist")
        if self.image_shape_chw != (3, 256, 256):
            raise ValueError("ACT base_camera input must be CHW (3, 256, 256)")
        if self.state_dimension not in (ACT_STATE_COMPONENTS, ACT_CONDITIONED_STATE_COMPONENTS):
            raise ValueError("ACT state dimension must be 9 or 15")
        if self.action_dimension != ACT_ACTION_COMPONENTS:
            raise ValueError("ACT action dimension must be 8")
        for name in (
            "chunk_size",
            "n_action_steps",
            "n_obs_steps",
            "dim_model",
            "n_heads",
            "dim_feedforward",
            "n_encoder_layers",
            "n_decoder_layers",
            "latent_dim",
            "n_vae_encoder_layers",
        ):
            _positive_int(cast(int, getattr(self, name)), name)
        if self.n_obs_steps != 1:
            raise ValueError("installed ACT 0.6.0 supports exactly one observation step")
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps cannot exceed chunk_size")
        if self.temporal_ensemble_coeff is not None and self.n_action_steps != 1:
            raise ValueError("temporal ensemble requires n_action_steps=1")
        if self.vision_backbone != "resnet18":
            raise ValueError("M4 pins the ACT image backbone to resnet18")
        if self.pretrained_backbone_weights is not None:
            raise ValueError("M4 pins pretrained_backbone_weights=None for offline reproducibility")
        if dict(self.normalization_mapping) != {
            "VISUAL": "MEAN_STD",
            "STATE": "MEAN_STD",
            "ACTION": "MEAN_STD",
        }:
            raise ValueError("M4 pins ACT visual/state/action normalization to MEAN_STD")
        if self.replace_final_stride_with_dilation or self.feedforward_activation != "relu":
            raise ValueError("M4 pins ACT dilation off and feedforward activation to relu")
        if self.use_peft or any(
            value is not None
            for value in (
                self.repo_id,
                self.private,
                self.tags,
                self.license,
                self.pretrained_path,
                self.pretrained_revision,
            )
        ):
            raise ValueError("M4 disables PEFT and every Hub/pretrained reference field")
        if self.observation_delta_indices is not None:
            raise ValueError("M4 ACT uses only the current observation")
        expected_action_deltas = tuple(range(self.chunk_size))
        if self.action_delta_indices is None:
            object.__setattr__(self, "action_delta_indices", expected_action_deltas)
        elif self.action_delta_indices != expected_action_deltas:
            raise ValueError("M4 ACT action deltas must equal range(chunk_size)")
        _finite(self.dropout, "dropout")
        _finite(self.kl_weight, "kl_weight")
        if self.dropout >= 1.0:
            raise ValueError("dropout must be < 1")
        if self.push_to_hub:
            raise ValueError("M4 never pushes ACT artifacts to the Hub")

    @classmethod
    def for_variant(cls, variant: ActVariant, **overrides: object) -> Self:
        state_dimension = (
            ACT_CONDITIONED_STATE_COMPONENTS
            if variant is ActVariant.MIXED_TASK_ONEHOT
            else ACT_STATE_COMPONENTS
        )
        return cls(state_dimension=state_dimension, **overrides)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = dict(value)
        shape = payload.get("image_shape_chw")
        if isinstance(shape, list):
            payload["image_shape_chw"] = tuple(shape)
        for name in ("tags", "observation_delta_indices", "action_delta_indices"):
            sequence = payload.get(name)
            if isinstance(sequence, list):
                payload[name] = tuple(sequence)
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ActDataConfig(_JsonRecord):
    """Completed-M3B dataset selection and loader contract."""

    dataset_root: str
    video_backend: str = "pyav"
    download_videos: bool = False
    return_uint8: bool = True
    fps: int = 20
    expected_train_episodes: int = 288
    expected_validation_episodes: int = 36
    expected_test_episodes: int = 36

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_root, str) or not self.dataset_root:
            raise TypeError("dataset_root must be a non-empty path string")
        if self.video_backend != "pyav" or self.download_videos:
            raise ValueError("M4 requires local PyAV videos with downloads disabled")
        if self.fps != 20:
            raise ValueError("M4 datasets must use 20 FPS")
        if (
            self.expected_train_episodes,
            self.expected_validation_episodes,
            self.expected_test_episodes,
        ) != (288, 36, 36):
            raise ValueError("full M4 requires the M3B 288/36/36 episode split")

    def identity_dict(self) -> dict[str, object]:
        payload = self.to_dict()
        payload.pop("dataset_root")
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class ActOptimizationConfig(_JsonRecord):
    """Fixed primary optimization contract shared by all three variants."""

    optimizer: str = "adamw"
    learning_rate: float = 1e-5
    backbone_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    scheduler: str = "none"
    warmup_steps: int = 0
    gradient_clip_norm: float = 10.0
    mixed_precision: str = "bfloat16"
    batch_size: int = 32
    dataloader_workers: int = 4
    training_steps: int = 100_000
    checkpoint_interval: int = 5_000
    validation_interval: int = 5_000

    def __post_init__(self) -> None:
        if self.optimizer != "adamw" or self.scheduler != "none":
            raise ValueError("M4 pins AdamW without a scheduler")
        for name in (
            "learning_rate",
            "backbone_learning_rate",
            "weight_decay",
            "gradient_clip_norm",
        ):
            _finite(cast(float, getattr(self, name)), name)
        for name in (
            "warmup_steps",
            "batch_size",
            "dataloader_workers",
            "training_steps",
            "checkpoint_interval",
            "validation_interval",
        ):
            _positive_int(
                cast(int, getattr(self, name)),
                name,
                allow_zero=name in {"warmup_steps", "dataloader_workers"},
            )
        if self.mixed_precision not in {"none", "float16", "bfloat16"}:
            raise ValueError("mixed_precision must be none, float16, or bfloat16")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class ActEvaluationConfig(_JsonRecord):
    """Immutable rollout, selection, and fresh-seed protocol."""

    maximum_episode_steps: int = 200
    control_frequency_hz: int = 20
    validation_scene_groups: int = 6
    test_scene_groups: int = 6
    fresh_scene_count: int = 30
    confidence_level: float = 0.95
    fresh_seed_namespace: str = "langmani-m4-fresh-seeds-v1"
    invalid_action_policy: str = "terminate"

    def __post_init__(self) -> None:
        for name in (
            "maximum_episode_steps",
            "control_frequency_hz",
            "validation_scene_groups",
            "test_scene_groups",
            "fresh_scene_count",
        ):
            _positive_int(cast(int, getattr(self, name)), name)
        if self.control_frequency_hz != 20:
            raise ValueError("M4 rollouts must use the dataset's 20 Hz control frequency")
        if (self.validation_scene_groups, self.test_scene_groups, self.fresh_scene_count) != (
            6,
            6,
            30,
        ):
            raise ValueError("M4 fixes validation/test/fresh schedules to 6/6/30 scenes")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between zero and one")
        if self.invalid_action_policy != "terminate":
            raise ValueError("M4 terminates instead of clipping invalid actions")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class ActExperimentConfig(_JsonRecord):
    """One variant, seed, and execution-mode declaration."""

    variant: ActVariant
    data: ActDataConfig
    model: ActModelConfig
    optimization: ActOptimizationConfig = field(default_factory=ActOptimizationConfig)
    evaluation: ActEvaluationConfig = field(default_factory=ActEvaluationConfig)
    seed: int = 0
    task_id: str | None = None
    mode: ExperimentMode = ExperimentMode.FULL
    device: str = "cuda"
    dtype: str = "float32"
    allow_dirty_development: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "variant", ActVariant(self.variant))
        object.__setattr__(self, "mode", ExperimentMode(self.mode))
        _positive_int(self.seed, "seed", allow_zero=True)
        expected_state_dimension = (
            ACT_CONDITIONED_STATE_COMPONENTS
            if self.variant is ActVariant.MIXED_TASK_ONEHOT
            else ACT_STATE_COMPONENTS
        )
        if self.model.state_dimension != expected_state_dimension:
            raise ValueError("model state dimension is incompatible with ACT variant")
        if self.variant is ActVariant.PER_TASK:
            if not isinstance(self.task_id, str) or not self.task_id:
                raise ValueError("per_task runs require one stable task_id")
        elif self.task_id is not None:
            raise ValueError("mixed ACT variants must not select a single task_id")
        if self.mode is ExperimentMode.FULL and self.allow_dirty_development:
            raise ValueError(
                "dirty-tree override is development-only and cannot authorize full mode"
            )
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.dtype != "float32":
            raise ValueError("model parameters and source tensors are pinned to float32")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = dict(value)
        data = payload.pop("data")
        model = payload.pop("model")
        optimization = payload.pop("optimization")
        evaluation = payload.pop("evaluation")
        if not all(isinstance(item, Mapping) for item in (data, model, optimization, evaluation)):
            raise TypeError("experiment nested configurations must be mappings")
        return cls(
            data=ActDataConfig.from_dict(cast(Mapping[str, object], data)),
            model=ActModelConfig.from_dict(cast(Mapping[str, object], model)),
            optimization=ActOptimizationConfig.from_dict(cast(Mapping[str, object], optimization)),
            evaluation=ActEvaluationConfig.from_dict(cast(Mapping[str, object], evaluation)),
            **payload,
        )


@dataclass(frozen=True, slots=True)
class ActRunIdentity(_JsonRecord):
    """Portable semantic inputs and their canonical run fingerprint."""

    m3b_export_fingerprint: str
    m3b_split_manifest_digest: str
    ordered_train_episode_indices: tuple[int, ...]
    ordered_validation_episode_indices: tuple[int, ...]
    variant: ActVariant
    task_id: str | None
    task_onehot_mapping_version: str
    model_config: Mapping[str, object]
    data_contract: Mapping[str, object]
    train_statistics_fingerprint: str
    optimization_config: Mapping[str, object]
    training_seed: int
    experiment_mode: ExperimentMode
    device: str
    dtype: str
    lerobot_version: str
    torch_version: str
    cuda_version: str | None
    git_commit: str
    git_dirty: bool
    dirty_development_override: bool
    schema_version: str = M4_SCHEMA_VERSION
    run_fingerprint: str = ""

    def __post_init__(self) -> None:
        for name in (
            "m3b_export_fingerprint",
            "m3b_split_manifest_digest",
            "train_statistics_fingerprint",
        ):
            _digest(cast(str, getattr(self, name)), name)
        object.__setattr__(self, "variant", ActVariant(self.variant))
        object.__setattr__(self, "experiment_mode", ExperimentMode(self.experiment_mode))
        object.__setattr__(
            self,
            "model_config",
            _frozen_mapping(self.model_config, "model_config"),
        )
        object.__setattr__(
            self,
            "data_contract",
            _frozen_mapping(self.data_contract, "data_contract"),
        )
        object.__setattr__(
            self,
            "optimization_config",
            _frozen_mapping(self.optimization_config, "optimization_config"),
        )
        if not self.ordered_train_episode_indices:
            raise ValueError("run identity requires at least one train episode")
        if len(set(self.ordered_train_episode_indices)) != len(self.ordered_train_episode_indices):
            raise ValueError("train episode indices must be unique")
        if set(self.ordered_train_episode_indices) & set(self.ordered_validation_episode_indices):
            raise ValueError("train and validation episode indices must be disjoint")
        if self.variant is ActVariant.PER_TASK and not self.task_id:
            raise ValueError("per_task run identity requires task_id")
        if self.variant is not ActVariant.PER_TASK and self.task_id is not None:
            raise ValueError("mixed run identity must not contain task_id")
        if self.task_onehot_mapping_version != TASK_ONEHOT_MAPPING_VERSION:
            raise ValueError("unknown task-one-hot mapping version")
        if self.lerobot_version != SUPPORTED_LEROBOT_VERSION:
            raise ValueError("M4 supports only the inspected LeRobot 0.6.0 API")
        if not re.fullmatch(r"[0-9a-f]{40,64}", self.git_commit):
            raise ValueError("git_commit must be a full lowercase Git object ID")
        if self.schema_version != M4_SCHEMA_VERSION:
            raise ValueError("unknown M4 schema version")
        if self.device not in {"cpu", "cuda"} or self.dtype != "float32":
            raise ValueError("run identity device/dtype must match the effective M4 contract")
        if self.git_dirty and self.experiment_mode in {
            ExperimentMode.FULL,
            ExperimentMode.TINY_OVERFIT,
        }:
            raise ValueError("full and tiny-overfit identities require a clean Git tree")
        if self.git_dirty and self.experiment_mode is ExperimentMode.DEVELOPMENT:
            if not self.dirty_development_override:
                raise ValueError("dirty development identity requires the explicit override")
        elif self.dirty_development_override:
            raise ValueError("dirty development override is valid only for a dirty development run")
        expected = self.compute_fingerprint()
        if not self.run_fingerprint:
            object.__setattr__(self, "run_fingerprint", expected)
        elif self.run_fingerprint != expected:
            raise ValueError("run_fingerprint does not match semantic run identity")

    def identity_payload(self) -> dict[str, object]:
        payload = self.to_dict()
        payload.pop("run_fingerprint")
        return payload

    def compute_fingerprint(self) -> str:
        return f"sha256:{sha256_hex(self.identity_payload())}"

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = dict(value)
        for name in ("ordered_train_episode_indices", "ordered_validation_episode_indices"):
            indices = payload.get(name)
            if not isinstance(indices, list):
                raise TypeError(f"{name} must be a list")
            payload[name] = tuple(indices)
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class TrainingState(_JsonRecord):
    global_step: int
    examples_processed: int
    last_checkpoint_fingerprint: str | None = None
    completed: bool = False

    def __post_init__(self) -> None:
        _positive_int(self.global_step, "global_step", allow_zero=True)
        _positive_int(self.examples_processed, "examples_processed", allow_zero=True)
        if self.last_checkpoint_fingerprint is not None:
            _digest(self.last_checkpoint_fingerprint, "last_checkpoint_fingerprint")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class CheckpointRecord(_JsonRecord):
    checkpoint_fingerprint: str
    run_fingerprint: str
    global_step: int
    relative_path: str
    policy_config_digest: str
    statistics_fingerprint: str
    split_digest: str
    git_commit: str
    complete: bool

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_fingerprint",
            "run_fingerprint",
            "policy_config_digest",
            "statistics_fingerprint",
            "split_digest",
        ):
            _digest(cast(str, getattr(self, name)), name)
        _positive_int(self.global_step, "global_step")
        if (
            not self.relative_path
            or self.relative_path.startswith(("/", "\\"))
            or ".." in self.relative_path
        ):
            raise ValueError("checkpoint relative_path must be a safe relative path")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class ValidationResult(_JsonRecord):
    checkpoint_fingerprint: str
    checkpoint_step: int
    schedule_digest: str
    success_rate: float
    wrong_object_interaction_rate: float
    target_off_table_rate: float
    offline_validation_action_loss: float

    def __post_init__(self) -> None:
        _digest(self.checkpoint_fingerprint, "checkpoint_fingerprint")
        _digest(self.schedule_digest, "schedule_digest")
        _positive_int(self.checkpoint_step, "checkpoint_step")
        for name in (
            "success_rate",
            "wrong_object_interaction_rate",
            "target_off_table_rate",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        _finite(self.offline_validation_action_loss, "offline_validation_action_loss")


def _empty_action_projection_summary() -> Mapping[str, object]:
    """Return the legacy-compatible zero-action audit used by in-memory fixtures."""

    return ActionProjectionSummary.from_records(
        (), action_dimension=ACT_ACTION_COMPONENTS
    ).to_dict()


@dataclass(frozen=True, slots=True)
class RolloutEpisodeResult(_JsonRecord):
    evaluation_id: str
    run_fingerprint: str
    checkpoint_fingerprint: str
    schedule_digest: str
    split: EvaluationSplit
    scene_seed: int
    scene_id: str
    task_id: str
    status: RolloutStatus
    success: bool
    episode_steps: int
    final_evaluation: Mapping[str, bool]
    target_in_target_bin: bool
    target_in_wrong_bin: bool
    wrong_object_in_target_bin: bool
    target_grasped_any: bool
    wrong_object_grasped_any: bool
    target_off_table: bool
    timeout: bool
    invalid_action: bool
    inference_failure: bool
    time_to_first_target_grasp_s: float | None
    time_to_release_s: float | None
    time_to_success_s: float | None
    cumulative_action_magnitude: float
    action_min: tuple[float, ...]
    action_max: tuple[float, ...]
    inference_latency_ms: tuple[float, ...]
    environment_step_latency_ms: tuple[float, ...]
    total_episode_duration_s: float
    failure_reason: str | None = None
    runtime_fingerprint: str = ""
    action_bound_mode: ActionBoundMode = ActionBoundMode.REJECT
    task_success: bool | None = None
    strict_unprojected_success: bool | None = None
    action_projection_summary: Mapping[str, object] = field(
        default_factory=_empty_action_projection_summary
    )

    def __post_init__(self) -> None:
        if not self.runtime_fingerprint:
            object.__setattr__(self, "runtime_fingerprint", self.run_fingerprint)
        if self.task_success is None:
            object.__setattr__(self, "task_success", self.success)
        if self.strict_unprojected_success is None:
            object.__setattr__(self, "strict_unprojected_success", self.success)
        for name in (
            "run_fingerprint",
            "checkpoint_fingerprint",
            "schedule_digest",
            "runtime_fingerprint",
        ):
            _digest(cast(str, getattr(self, name)), name)
        object.__setattr__(self, "split", EvaluationSplit(self.split))
        object.__setattr__(self, "status", RolloutStatus(self.status))
        object.__setattr__(self, "action_bound_mode", ActionBoundMode(self.action_bound_mode))
        object.__setattr__(
            self,
            "final_evaluation",
            _frozen_mapping(self.final_evaluation, "final_evaluation"),
        )
        summary = ActionProjectionSummary.from_dict(self.action_projection_summary)
        object.__setattr__(
            self,
            "action_projection_summary",
            _frozen_mapping(summary.to_dict(), "action_projection_summary"),
        )
        _positive_int(self.scene_seed, "scene_seed", allow_zero=True)
        _positive_int(self.episode_steps, "episode_steps", allow_zero=True)
        if (
            len(self.action_min) != ACT_ACTION_COMPONENTS
            or len(self.action_max) != ACT_ACTION_COMPONENTS
        ):
            raise ValueError("rollout action ranges must contain eight components")
        if (
            self.success != (self.status is RolloutStatus.SUCCESS)
            or self.task_success != self.success
        ):
            raise ValueError("success must agree with rollout status")
        expected_strict = self.task_success and summary.projected_action_count == 0
        if self.strict_unprojected_success != expected_strict:
            raise ValueError(
                "strict_unprojected_success requires task success without projected actions"
            )
        for value in (
            self.cumulative_action_magnitude,
            self.total_episode_duration_s,
            *self.inference_latency_ms,
            *self.environment_step_latency_ms,
        ):
            _finite(value, "rollout metric")


@dataclass(frozen=True, slots=True)
class RolloutBenchmarkResult(_JsonRecord):
    benchmark_id: str
    run_fingerprint: str
    checkpoint_fingerprint: str
    schedule_digest: str
    split: EvaluationSplit
    episodes: tuple[RolloutEpisodeResult, ...]
    success_rate: float
    success_wilson_low: float
    success_wilson_high: float
    per_task_success_rates: Mapping[str, float]
    per_scene_group_success_rates: Mapping[str, float]
    metric_rates: Mapping[str, float]
    failure_counts: Mapping[str, int]
    inference_latency_ms: Mapping[str, float | None]
    environment_step_latency_ms: Mapping[str, float | None]
    runtime_fingerprint: str
    action_bound_mode: ActionBoundMode
    task_success_count: int
    strict_unprojected_success_count: int
    raw_action_bounds_validated: bool
    projected_action_bounds_validated: bool
    action_projection_summary: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", EvaluationSplit(self.split))
        object.__setattr__(self, "action_bound_mode", ActionBoundMode(self.action_bound_mode))
        _digest(self.runtime_fingerprint, "runtime_fingerprint")
        object.__setattr__(
            self,
            "per_task_success_rates",
            _frozen_mapping(self.per_task_success_rates, "per_task_success_rates"),
        )
        object.__setattr__(
            self,
            "failure_counts",
            _frozen_mapping(self.failure_counts, "failure_counts"),
        )
        for name in (
            "per_scene_group_success_rates",
            "metric_rates",
            "inference_latency_ms",
            "environment_step_latency_ms",
            "action_projection_summary",
        ):
            object.__setattr__(
                self,
                name,
                _frozen_mapping(cast(Mapping[str, object], getattr(self, name)), name),
            )
        if not self.episodes:
            raise ValueError("benchmark result requires episode records")
        if self.task_success_count != sum(item.task_success for item in self.episodes):
            raise ValueError("task_success_count disagrees with episode records")
        if self.strict_unprojected_success_count != sum(
            item.strict_unprojected_success for item in self.episodes
        ):
            raise ValueError("strict_unprojected_success_count disagrees with episode records")
        for value in (self.success_rate, self.success_wilson_low, self.success_wilson_high):
            if not 0.0 <= value <= 1.0:
                raise ValueError("benchmark rates must be between zero and one")


@dataclass(frozen=True, slots=True)
class CounterfactualSensitivityResult(_JsonRecord):
    variant: ActVariant
    scene_id: str
    ordered_task_ids: tuple[str, ...]
    first_actions: tuple[tuple[float, ...], ...]
    action_chunks: tuple[tuple[tuple[float, ...], ...], ...]
    pairwise_first_action_distances: Mapping[str, float]
    pairwise_chunk_distances: Mapping[str, float]
    identical_unconditioned_inputs: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "variant", ActVariant(self.variant))
        object.__setattr__(
            self,
            "pairwise_first_action_distances",
            _frozen_mapping(
                self.pairwise_first_action_distances,
                "pairwise_first_action_distances",
            ),
        )
        object.__setattr__(
            self,
            "pairwise_chunk_distances",
            _frozen_mapping(self.pairwise_chunk_distances, "pairwise_chunk_distances"),
        )
        if (
            len(self.ordered_task_ids) != 6
            or len(self.first_actions) != 6
            or len(self.action_chunks) != 6
        ):
            raise ValueError("counterfactual sensitivity requires all six tasks")
        if any(len(action) != ACT_ACTION_COMPONENTS for action in self.first_actions):
            raise ValueError("every first action must have eight components")
        horizons = {len(chunk) for chunk in self.action_chunks}
        if len(horizons) != 1 or not horizons or next(iter(horizons)) < 1:
            raise ValueError("counterfactual action chunks must share a nonempty horizon")
        if any(
            len(action) != ACT_ACTION_COMPONENTS for chunk in self.action_chunks for action in chunk
        ):
            raise ValueError("every predicted chunk action must have eight components")
        if any(
            first != chunk[0]
            for first, chunk in zip(self.first_actions, self.action_chunks, strict=True)
        ):
            raise ValueError("first actions must equal the first action in each stored chunk")


@dataclass(frozen=True, slots=True)
class ActExperimentManifest(_JsonRecord):
    identity: ActRunIdentity
    config: ActExperimentConfig
    git_dirty: bool
    dirty_development_override: bool
    train_statistics_fingerprint: str
    training_state: TrainingState
    checkpoints: tuple[CheckpointRecord, ...] = ()
    selected_checkpoint_fingerprint: str | None = None
    test_evaluation_locked: bool = True
    complete: bool = False

    def __post_init__(self) -> None:
        if self.identity.run_fingerprint == "":
            raise ValueError("manifest identity must have a run fingerprint")
        _digest(self.train_statistics_fingerprint, "train_statistics_fingerprint")
        if self.identity.variant is not self.config.variant:
            raise ValueError("manifest config variant differs from run identity")
        if self.identity.task_id != self.config.task_id:
            raise ValueError("manifest config task differs from run identity")
        if _json_value(self.identity.model_config) != self.config.model.to_dict():
            raise ValueError("manifest model config differs from run identity")
        if _json_value(self.identity.optimization_config) != self.config.optimization.to_dict():
            raise ValueError("manifest optimization config differs from run identity")
        if (
            self.identity.training_seed != self.config.seed
            or self.identity.experiment_mode is not self.config.mode
            or self.identity.device != self.config.device
            or self.identity.dtype != self.config.dtype
        ):
            raise ValueError("manifest effective execution config differs from run identity")
        if self.identity.train_statistics_fingerprint != self.train_statistics_fingerprint:
            raise ValueError("manifest statistics fingerprint differs from run identity")
        if self.git_dirty != self.identity.git_dirty:
            raise ValueError("manifest Git dirty state differs from run identity")
        if self.dirty_development_override != self.identity.dirty_development_override:
            raise ValueError("manifest dirty override differs from run identity")
        if self.git_dirty and not self.dirty_development_override:
            raise ValueError("a dirty run requires an explicit development override")
        if self.dirty_development_override and self.config.mode is not ExperimentMode.DEVELOPMENT:
            raise ValueError("dirty override is valid only in development mode")
        if self.selected_checkpoint_fingerprint is not None:
            _digest(self.selected_checkpoint_fingerprint, "selected_checkpoint_fingerprint")
        steps = tuple(item.global_step for item in self.checkpoints)
        if steps != tuple(sorted(set(steps))):
            raise ValueError("manifest checkpoints must have unique increasing steps")
        for checkpoint in self.checkpoints:
            if (
                checkpoint.run_fingerprint != self.identity.run_fingerprint
                or checkpoint.statistics_fingerprint != self.identity.train_statistics_fingerprint
                or checkpoint.split_digest != self.identity.m3b_split_manifest_digest
                or checkpoint.git_commit != self.identity.git_commit
            ):
                raise ValueError("manifest checkpoint provenance differs from run identity")
        if self.training_state.last_checkpoint_fingerprint is not None and (
            not self.checkpoints
            or self.training_state.last_checkpoint_fingerprint
            != self.checkpoints[-1].checkpoint_fingerprint
        ):
            raise ValueError("training state does not reference the latest checkpoint")
        if (
            self.selected_checkpoint_fingerprint is not None
            and self.selected_checkpoint_fingerprint
            not in {item.checkpoint_fingerprint for item in self.checkpoints}
        ):
            raise ValueError("selected checkpoint is absent from the run manifest")
        if self.complete and not self.training_state.completed:
            raise ValueError("a completed manifest requires completed training state")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = dict(value)
        identity = payload.pop("identity")
        config = payload.pop("config")
        training_state = payload.pop("training_state")
        checkpoints = payload.pop("checkpoints")
        if not isinstance(identity, Mapping) or not isinstance(config, Mapping):
            raise TypeError("manifest identity and config must be mappings")
        if not isinstance(training_state, Mapping) or not isinstance(checkpoints, list):
            raise TypeError("manifest training state/checkpoints are malformed")
        if not all(isinstance(item, Mapping) for item in checkpoints):
            raise TypeError("manifest checkpoints must be mappings")
        return cls(
            identity=ActRunIdentity.from_dict(identity),
            config=ActExperimentConfig.from_dict(config),
            training_state=TrainingState.from_dict(training_state),
            checkpoints=tuple(
                CheckpointRecord.from_dict(cast(Mapping[str, object], item)) for item in checkpoints
            ),
            **payload,
        )


@dataclass(frozen=True, slots=True)
class ActComparisonReport(_JsonRecord):
    schema_version: str
    dataset_fingerprint: str
    git_commit: str
    per_task_run_fingerprints: tuple[str, ...]
    mixed_unconditioned_run_fingerprint: str
    mixed_task_onehot_run_fingerprint: str
    interpretation: Mapping[str, str]
    full_experiment_validated: bool
    baseline_quality_validated: bool

    def __post_init__(self) -> None:
        if self.schema_version != M4_SCHEMA_VERSION:
            raise ValueError("unknown comparison schema")
        _digest(self.dataset_fingerprint, "dataset_fingerprint")
        if len(self.per_task_run_fingerprints) != 6:
            raise ValueError("comparison requires six per-task runs")
        object.__setattr__(
            self,
            "interpretation",
            _frozen_mapping(self.interpretation, "interpretation"),
        )
        for value in (
            *self.per_task_run_fingerprints,
            self.mixed_unconditioned_run_fingerprint,
            self.mixed_task_onehot_run_fingerprint,
        ):
            _digest(value, "run_fingerprint")


def task_spec_from_stable_id(task_id: str, task_specs: Sequence[TaskSpec]) -> TaskSpec:
    """Resolve a task only through stable semantic IDs, never instruction text."""
    matches = tuple(spec for spec in task_specs if stable_task_id(spec) == task_id)
    if len(matches) != 1:
        raise ValueError(f"unknown or ambiguous stable task ID {task_id!r}")
    return matches[0]


__all__ = [
    "ACT_ACTION_COMPONENTS",
    "ACT_CONDITIONED_STATE_COMPONENTS",
    "ACT_STATE_COMPONENTS",
    "M4_SCHEMA_VERSION",
    "TASK_ONEHOT_MAPPING_VERSION",
    "ActComparisonReport",
    "ActDataConfig",
    "ActEvaluationConfig",
    "ActExperimentConfig",
    "ActExperimentManifest",
    "ActModelConfig",
    "ActOptimizationConfig",
    "ActRunIdentity",
    "ActVariant",
    "CheckpointRecord",
    "CounterfactualSensitivityResult",
    "EvaluationSplit",
    "ExperimentMode",
    "RolloutBenchmarkResult",
    "RolloutEpisodeResult",
    "RolloutStatus",
    "TrainingState",
    "ValidationResult",
    "task_spec_from_stable_id",
]
