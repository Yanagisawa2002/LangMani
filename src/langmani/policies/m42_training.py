"""Provenance and orchestration contracts for M4.2 TaskToken training.

The historical M4 ``ActRunIdentity`` deliberately enumerates only its three
completed baselines.  M4.2 therefore owns a separate run identity while
reusing the existing atomic ACT checkpoint implementation through its public
field contract.  This keeps every historical M4 fingerprint unchanged.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from langmani.datasets.identity import sha256_hex
from langmani.policies.act_types import (
    ACT_CONDITIONED_STATE_COMPONENTS,
    ACT_STATE_COMPONENTS,
    ActExperimentManifest,
    ActVariant,
    CheckpointRecord,
    ExperimentMode,
    TrainingState,
)
from langmani.policies.m42_schedule import (
    M42_DEV_SCHEDULE_FINGERPRINT,
    M42_FINAL_SCHEDULE_FINGERPRINT,
)
from langmani.policies.m42_types import M42ExperimentManifest

TASK_TOKEN_RUN_SCHEMA = "langmani-m42-task-token-run-v0"
TASK_TOKEN_MANIFEST_SCHEMA = "langmani-m42-task-token-training-manifest-v0"
TASK_TOKEN_VALIDATION_QUEUE_SCHEMA = "langmani-m42-task-token-validation-queue-v0"
TASK_TOKEN_TRAINING_SUMMARY_SCHEMA = "langmani-m42-task-token-training-summary-v0"
TASK_TOKEN_TRAINING_COMPLETION_SCHEMA = "langmani-m42-task-token-training-complete-v0"
TASK_TOKEN_FAIR_COMPARISON_SCHEMA = "langmani-m42-task-token-fair-comparison-v0"
TASK_TOKEN_DATA_ORDER_SCHEMA = "langmani-m42-deterministic-data-order-v0"
RUNTIME_SELECTION_SCHEMA = "langmani-m42-runtime-selection-v0"
TASK_TOKEN_MODEL_LABEL = "act_mixed_task_token"
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_RUN_DIRECTORY_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMON_DATA_CONTRACT_FIELDS = (
    "video_backend",
    "download_videos",
    "return_uint8",
    "fps",
    "expected_train_episodes",
    "expected_validation_episodes",
    "expected_test_episodes",
    "m3b_feature_contract",
    "policy_state_schema",
)


class M42TrainingContractError(RuntimeError):
    """Raised when TaskToken training or its provenance is unsafe."""


class _FrozenMapping(Mapping[str, object]):
    __slots__ = ("_items",)

    def __init__(self, value: Mapping[str, object]) -> None:
        if not all(isinstance(key, str) for key in value):
            raise TypeError("M4.2 JSON mapping keys must be strings")
        self._items = tuple((key, _freeze_json(item)) for key, item in sorted(value.items()))

    def __getitem__(self, key: str) -> object:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return _FrozenMapping(cast(Mapping[str, object], value))
    if isinstance(value, tuple | list):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"unsupported M4.2 JSON value {type(value).__name__}")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_thaw_json(item) for item in value]
    return value


def _mapping(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return _FrozenMapping(value)


def _digest(value: str, name: str) -> None:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise M42TrainingContractError(f"{name} must be a prefixed lowercase SHA-256 digest")


def _git_commit(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40,64}", value) is None:
        raise M42TrainingContractError("git_commit must be a full lowercase Git object ID")


def canonical_fingerprint(value: Mapping[str, object]) -> str:
    """Fingerprint one JSON mapping with the repository's canonical encoder."""
    return f"sha256:{sha256_hex(cast(dict[str, object], _thaw_json(value)))}"


@dataclass(frozen=True, slots=True)
class TaskTokenFairComparisonContract:
    """Frozen M4 State-OneHot configuration that TaskToken must match.

    The only permitted architectural difference is replacing the 15D
    ``PandaPolicyStateV0 + CanonicalTaskOneHotV0`` STATE input with an
    unchanged 9D Panda STATE input and a separate 6D ENV token.  Every other
    model, optimization, split, seed, and deterministic ordering field remains
    bound to the real selected M4 run manifest.
    """

    source_run_fingerprint: str
    source_checkpoint_fingerprint: str
    m3b_export_fingerprint: str
    m3b_split_manifest_digest: str
    ordered_train_episode_indices: tuple[int, ...]
    ordered_validation_episode_indices: tuple[int, ...]
    source_model_config: Mapping[str, object]
    source_optimization_config: Mapping[str, object]
    source_common_data_contract: Mapping[str, object]
    data_order_contract: Mapping[str, object]
    source_training_seed: int
    source_device: str
    source_dtype: str
    allowed_input_difference: Mapping[str, object]
    schema_version: str = TASK_TOKEN_FAIR_COMPARISON_SCHEMA
    contract_fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != TASK_TOKEN_FAIR_COMPARISON_SCHEMA:
            raise M42TrainingContractError("unsupported TaskToken fair-comparison schema")
        for name in (
            "source_run_fingerprint",
            "source_checkpoint_fingerprint",
            "m3b_export_fingerprint",
            "m3b_split_manifest_digest",
        ):
            _digest(cast(str, getattr(self, name)), name)
        if len(self.ordered_train_episode_indices) != 288:
            raise M42TrainingContractError("fair comparison requires all 288 M3B train episodes")
        if len(self.ordered_validation_episode_indices) != 36:
            raise M42TrainingContractError(
                "fair comparison requires all 36 M3B validation episodes"
            )
        if tuple(sorted(set(self.ordered_train_episode_indices))) != (
            self.ordered_train_episode_indices
        ):
            raise M42TrainingContractError("fair-comparison train ordering is not canonical")
        if tuple(sorted(set(self.ordered_validation_episode_indices))) != (
            self.ordered_validation_episode_indices
        ):
            raise M42TrainingContractError("fair-comparison validation ordering is not canonical")
        if set(self.ordered_train_episode_indices) & set(self.ordered_validation_episode_indices):
            raise M42TrainingContractError("fair-comparison train and validation views overlap")
        if self.source_training_seed != 0:
            raise M42TrainingContractError("fair comparison requires the frozen primary M4 seed")
        if self.source_device != "cuda" or self.source_dtype != "float32":
            raise M42TrainingContractError(
                "frozen M4 fair-comparison execution must be CUDA/float32"
            )
        for name in (
            "source_model_config",
            "source_optimization_config",
            "source_common_data_contract",
            "data_order_contract",
            "allowed_input_difference",
        ):
            object.__setattr__(
                self,
                name,
                _mapping(cast(Mapping[str, object], getattr(self, name)), name),
            )
        model = self.source_model_config
        if model.get("state_dimension") != ACT_CONDITIONED_STATE_COMPONENTS:
            raise M42TrainingContractError("frozen State-OneHot model must use its real 15D state")
        if set(self.source_common_data_contract) != set(_COMMON_DATA_CONTRACT_FIELDS):
            raise M42TrainingContractError("fair-comparison common data contract is incomplete")
        expected_order = {
            "schema_version": TASK_TOKEN_DATA_ORDER_SCHEMA,
            "sampler": "DeterministicResumeBatchSampler",
            "ordered_train_episode_indices": list(self.ordered_train_episode_indices),
            "ordered_validation_episode_indices": list(self.ordered_validation_episode_indices),
            "training_seed": self.source_training_seed,
            "batch_size": self.source_optimization_config.get("batch_size"),
            "epoch_seed_rule": "(training_seed + epoch) % (2**63 - 1)",
            "resume_cursor": "completed_optimizer_step",
        }
        if _thaw_json(self.data_order_contract) != expected_order:
            raise M42TrainingContractError("fair-comparison data-order contract is inconsistent")
        expected_difference = {
            "source_state": {
                "feature_type": "STATE",
                "dimension": ACT_CONDITIONED_STATE_COMPONENTS,
                "contents": "PandaPolicyStateV0+CanonicalTaskOneHotV0",
            },
            "task_token_state": {
                "feature_type": "STATE",
                "dimension": ACT_STATE_COMPONENTS,
                "contents": "PandaPolicyStateV0",
            },
            "task_token_condition": {
                "feature_type": "ENV",
                "dimension": ACT_CONDITIONED_STATE_COMPONENTS - ACT_STATE_COMPONENTS,
                "mapping": "CanonicalTaskTokenV0",
            },
            "all_other_differences_permitted": False,
        }
        if _thaw_json(self.allowed_input_difference) != expected_difference:
            raise M42TrainingContractError("TaskToken declares an unsupported fairness difference")
        expected_fingerprint = self.compute_fingerprint()
        if not self.contract_fingerprint:
            object.__setattr__(self, "contract_fingerprint", expected_fingerprint)
        elif self.contract_fingerprint != expected_fingerprint:
            raise M42TrainingContractError("fair-comparison fingerprint is inconsistent")

    def identity_payload(self) -> dict[str, object]:
        payload = self.to_dict()
        payload.pop("contract_fingerprint")
        return payload

    def compute_fingerprint(self) -> str:
        return canonical_fingerprint(self.identity_payload())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_run_fingerprint": self.source_run_fingerprint,
            "source_checkpoint_fingerprint": self.source_checkpoint_fingerprint,
            "m3b_export_fingerprint": self.m3b_export_fingerprint,
            "m3b_split_manifest_digest": self.m3b_split_manifest_digest,
            "ordered_train_episode_indices": list(self.ordered_train_episode_indices),
            "ordered_validation_episode_indices": list(self.ordered_validation_episode_indices),
            "source_model_config": _thaw_json(self.source_model_config),
            "source_optimization_config": _thaw_json(self.source_optimization_config),
            "source_common_data_contract": _thaw_json(self.source_common_data_contract),
            "data_order_contract": _thaw_json(self.data_order_contract),
            "source_training_seed": self.source_training_seed,
            "source_device": self.source_device,
            "source_dtype": self.source_dtype,
            "allowed_input_difference": _thaw_json(self.allowed_input_difference),
            "contract_fingerprint": self.contract_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TaskTokenFairComparisonContract:
        payload = dict(value)
        for name in ("ordered_train_episode_indices", "ordered_validation_episode_indices"):
            sequence = payload.get(name)
            if not isinstance(sequence, list):
                raise M42TrainingContractError(f"{name} must be a JSON list")
            payload[name] = tuple(sequence)
        try:
            return cls(**payload)
        except (KeyError, TypeError, ValueError) as error:
            raise M42TrainingContractError(
                "TaskToken fair-comparison fields are malformed"
            ) from error


def build_task_token_fair_comparison_contract(
    manifest: ActExperimentManifest,
    *,
    selected_checkpoint_fingerprint: str,
) -> TaskTokenFairComparisonContract:
    """Project the real selected Mixed-TaskOneHot manifest into a frozen gate."""

    identity = manifest.identity
    if (
        identity.variant is not ActVariant.MIXED_TASK_ONEHOT
        or identity.task_id is not None
        or identity.experiment_mode is not ExperimentMode.FULL
        or not manifest.complete
        or not manifest.training_state.completed
        or manifest.selected_checkpoint_fingerprint != selected_checkpoint_fingerprint
    ):
        raise M42TrainingContractError(
            "fair comparison requires the completed selected Mixed-TaskOneHot run"
        )
    identity_payload = identity.to_dict()
    raw_data = identity_payload.get("data_contract")
    if not isinstance(raw_data, Mapping):
        raise M42TrainingContractError("frozen Mixed-TaskOneHot data contract is malformed")
    missing = tuple(name for name in _COMMON_DATA_CONTRACT_FIELDS if name not in raw_data)
    if missing:
        raise M42TrainingContractError(
            "frozen Mixed-TaskOneHot data contract lacks: " + ", ".join(missing)
        )
    model = cast(Mapping[str, object], identity_payload["model_config"])
    optimization = cast(Mapping[str, object], identity_payload["optimization_config"])
    return TaskTokenFairComparisonContract(
        source_run_fingerprint=identity.run_fingerprint,
        source_checkpoint_fingerprint=selected_checkpoint_fingerprint,
        m3b_export_fingerprint=identity.m3b_export_fingerprint,
        m3b_split_manifest_digest=identity.m3b_split_manifest_digest,
        ordered_train_episode_indices=identity.ordered_train_episode_indices,
        ordered_validation_episode_indices=identity.ordered_validation_episode_indices,
        source_model_config=model,
        source_optimization_config=optimization,
        source_common_data_contract={name: raw_data[name] for name in _COMMON_DATA_CONTRACT_FIELDS},
        data_order_contract={
            "schema_version": TASK_TOKEN_DATA_ORDER_SCHEMA,
            "sampler": "DeterministicResumeBatchSampler",
            "ordered_train_episode_indices": list(identity.ordered_train_episode_indices),
            "ordered_validation_episode_indices": list(identity.ordered_validation_episode_indices),
            "training_seed": identity.training_seed,
            "batch_size": optimization.get("batch_size"),
            "epoch_seed_rule": "(training_seed + epoch) % (2**63 - 1)",
            "resume_cursor": "completed_optimizer_step",
        },
        source_training_seed=identity.training_seed,
        source_device=identity.device,
        source_dtype=identity.dtype,
        allowed_input_difference={
            "source_state": {
                "feature_type": "STATE",
                "dimension": ACT_CONDITIONED_STATE_COMPONENTS,
                "contents": "PandaPolicyStateV0+CanonicalTaskOneHotV0",
            },
            "task_token_state": {
                "feature_type": "STATE",
                "dimension": ACT_STATE_COMPONENTS,
                "contents": "PandaPolicyStateV0",
            },
            "task_token_condition": {
                "feature_type": "ENV",
                "dimension": ACT_CONDITIONED_STATE_COMPONENTS - ACT_STATE_COMPONENTS,
                "mapping": "CanonicalTaskTokenV0",
            },
            "all_other_differences_permitted": False,
        },
    )


def fingerprint_owned_task_token_runs(root: Path) -> tuple[Path, ...]:
    """Return every direct 64-hex TaskToken run, complete or incomplete.

    Recovery content lives below ``.recovery`` and is intentionally outside
    this fingerprint-owned namespace.
    """

    if not root.exists():
        return ()
    if root.is_symlink() or root.is_junction() or not root.is_dir():
        raise M42TrainingContractError(f"unsafe TaskToken model root: {root}")
    resolved_root = root.resolve()
    result: list[Path] = []
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if _RUN_DIRECTORY_RE.fullmatch(child.name) is None:
            continue
        if child.is_symlink() or child.is_junction() or not child.is_dir():
            raise M42TrainingContractError(f"unsafe fingerprint-owned TaskToken run: {child}")
        resolved = child.resolve()
        if resolved.parent != resolved_root:
            raise M42TrainingContractError(f"TaskToken run escaped its model root: {child}")
        result.append(resolved)
    return tuple(result)


def _contains_final_schedule_reference(value: object) -> bool:
    if isinstance(value, str):
        return "m42_final_v0" in value
    if isinstance(value, Mapping):
        return any(
            "final_schedule" in str(key) or _contains_final_schedule_reference(item)
            for key, item in value.items()
        )
    if isinstance(value, tuple | list):
        return any(_contains_final_schedule_reference(item) for item in value)
    return False


@dataclass(frozen=True, slots=True)
class RuntimeSelectionContract:
    """The locked M4.2a runtime consumed by TaskToken training/evaluation."""

    execution_horizon: int
    gripper_mode: str
    horizon_selection_fingerprint: str
    gripper_selection_fingerprint: str
    runtime_fingerprint: str
    development_schedule_fingerprint: str
    m3b_dataset_fingerprint: str
    mixed_task_onehot_checkpoint_fingerprint: str
    representative_per_task_checkpoint_fingerprint: str
    implementation_fingerprint: str
    experiment_manifest_fingerprint: str
    evaluation_git_commit: str
    source_fingerprint: str
    selection_evidence: Mapping[str, object]
    schema_version: str = RUNTIME_SELECTION_SCHEMA
    locked: bool = True
    final_schedule_accessed: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != RUNTIME_SELECTION_SCHEMA:
            raise M42TrainingContractError("unsupported runtime-selection schema")
        if (
            isinstance(self.execution_horizon, bool)
            or not isinstance(self.execution_horizon, int)
            or self.execution_horizon not in (1, 5, 10)
        ):
            raise M42TrainingContractError("execution_horizon must be one of 1, 5, or 10")
        if self.gripper_mode not in {"project", "binary"}:
            raise M42TrainingContractError("gripper_mode must be project or binary")
        for name in (
            "horizon_selection_fingerprint",
            "gripper_selection_fingerprint",
            "runtime_fingerprint",
            "development_schedule_fingerprint",
            "m3b_dataset_fingerprint",
            "mixed_task_onehot_checkpoint_fingerprint",
            "representative_per_task_checkpoint_fingerprint",
            "implementation_fingerprint",
            "experiment_manifest_fingerprint",
            "source_fingerprint",
        ):
            _digest(cast(str, getattr(self, name)), name)
        _git_commit(self.evaluation_git_commit)
        if not isinstance(self.locked, bool) or not self.locked:
            raise M42TrainingContractError("runtime selection must be immutable and locked")
        if not isinstance(self.final_schedule_accessed, bool):
            raise M42TrainingContractError("final_schedule_accessed must be boolean")
        if self.final_schedule_accessed:
            raise M42TrainingContractError(
                "TaskToken training must not consume m42_final_v0 evidence"
            )
        object.__setattr__(
            self,
            "selection_evidence",
            _mapping(self.selection_evidence, "selection_evidence"),
        )
        if not self.selection_evidence:
            raise M42TrainingContractError("runtime selection requires evidence references")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "execution_horizon": self.execution_horizon,
            "gripper_mode": self.gripper_mode,
            "horizon_selection_fingerprint": self.horizon_selection_fingerprint,
            "gripper_selection_fingerprint": self.gripper_selection_fingerprint,
            "runtime_fingerprint": self.runtime_fingerprint,
            "development_schedule_fingerprint": self.development_schedule_fingerprint,
            "m3b_dataset_fingerprint": self.m3b_dataset_fingerprint,
            "mixed_task_onehot_checkpoint_fingerprint": (
                self.mixed_task_onehot_checkpoint_fingerprint
            ),
            "representative_per_task_checkpoint_fingerprint": (
                self.representative_per_task_checkpoint_fingerprint
            ),
            "implementation_fingerprint": self.implementation_fingerprint,
            "experiment_manifest_fingerprint": self.experiment_manifest_fingerprint,
            "evaluation_git_commit": self.evaluation_git_commit,
            "source_fingerprint": self.source_fingerprint,
            "selection_evidence": _thaw_json(self.selection_evidence),
            "locked": self.locked,
            "final_schedule_accessed": self.final_schedule_accessed,
        }


def load_runtime_selection(path: str | Path) -> RuntimeSelectionContract:
    """Load and bind a locked runtime selection without touching final seeds."""
    source = Path(path).resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise M42TrainingContractError(
            f"cannot read runtime selection {source}: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise M42TrainingContractError("runtime-selection JSON must be an object")
    required = {
        "schema_version",
        "execution_horizon",
        "gripper_mode",
        "horizon_selection_fingerprint",
        "gripper_selection_fingerprint",
        "runtime_fingerprint",
        "development_schedule_fingerprint",
        "m3b_dataset_fingerprint",
        "mixed_task_onehot_checkpoint_fingerprint",
        "representative_per_task_checkpoint_fingerprint",
        "implementation_fingerprint",
        "experiment_manifest_fingerprint",
        "evaluation_git_commit",
        "selection_evidence",
        "locked",
        "final_schedule_accessed",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise M42TrainingContractError(
            "runtime selection is missing required fields: " + ", ".join(missing)
        )
    selection_only_payload = {
        key: value for key, value in raw.items() if key != "final_schedule_accessed"
    }
    if _contains_final_schedule_reference(selection_only_payload):
        raise M42TrainingContractError(
            "runtime selection must not embed or reference m42_final_v0 evidence"
        )
    source_fingerprint = canonical_fingerprint(raw)
    evidence = raw.get("selection_evidence", {})
    if not isinstance(evidence, Mapping):
        raise M42TrainingContractError("selection_evidence must be a JSON object")
    result = RuntimeSelectionContract(
        execution_horizon=raw["execution_horizon"],
        gripper_mode=raw["gripper_mode"],
        horizon_selection_fingerprint=raw["horizon_selection_fingerprint"],
        gripper_selection_fingerprint=raw["gripper_selection_fingerprint"],
        runtime_fingerprint=raw["runtime_fingerprint"],
        development_schedule_fingerprint=raw["development_schedule_fingerprint"],
        m3b_dataset_fingerprint=raw["m3b_dataset_fingerprint"],
        mixed_task_onehot_checkpoint_fingerprint=(raw["mixed_task_onehot_checkpoint_fingerprint"]),
        representative_per_task_checkpoint_fingerprint=(
            raw["representative_per_task_checkpoint_fingerprint"]
        ),
        implementation_fingerprint=raw["implementation_fingerprint"],
        experiment_manifest_fingerprint=raw["experiment_manifest_fingerprint"],
        evaluation_git_commit=raw["evaluation_git_commit"],
        source_fingerprint=source_fingerprint,
        selection_evidence=cast(Mapping[str, object], evidence),
        schema_version=raw["schema_version"],
        locked=raw["locked"],
        final_schedule_accessed=raw["final_schedule_accessed"],
    )
    manifest_relative = evidence.get("experiment_manifest")
    if (
        not isinstance(manifest_relative, str)
        or not manifest_relative
        or Path(manifest_relative).is_absolute()
        or ".." in Path(manifest_relative).parts
    ):
        raise M42TrainingContractError(
            "runtime selection requires a safe experiment-manifest reference"
        )
    manifest_path = (source.parent / manifest_relative).resolve()
    if (
        manifest_path.parent != source.parent
        or manifest_path.is_symlink()
        or manifest_path.is_junction()
        or not manifest_path.is_file()
    ):
        raise M42TrainingContractError("runtime selection experiment manifest is unsafe")
    try:
        manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise M42TrainingContractError("cannot read runtime experiment manifest") from error
    if not isinstance(manifest_raw, Mapping):
        raise M42TrainingContractError("runtime experiment manifest must be a JSON object")
    manifest = M42ExperimentManifest.from_dict(cast(Mapping[str, object], manifest_raw))
    if (
        manifest.fingerprint != result.experiment_manifest_fingerprint
        or manifest.m3b_dataset_fingerprint != result.m3b_dataset_fingerprint
        or manifest.implementation_git_commit != result.evaluation_git_commit
        or manifest.runtime_selection_fingerprints.get("execution_horizon")
        != result.horizon_selection_fingerprint
        or manifest.runtime_selection_fingerprints.get("gripper_runtime")
        != result.gripper_selection_fingerprint
        or manifest.schedule_fingerprints.get("m42_dev_v0") != M42_DEV_SCHEDULE_FINGERPRINT
        or manifest.schedule_fingerprints.get("m42_final_v0") != M42_FINAL_SCHEDULE_FINGERPRINT
        or result.mixed_task_onehot_checkpoint_fingerprint
        not in manifest.m4_checkpoint_fingerprints
        or result.representative_per_task_checkpoint_fingerprint
        not in manifest.m4_checkpoint_fingerprints
        or manifest.artifact_paths.get("runtime_selection") != source.name
    ):
        raise M42TrainingContractError(
            "runtime selection differs from its immutable M4.2 experiment manifest"
        )
    return result


@dataclass(frozen=True, slots=True)
class TaskTokenRunIdentity:
    """Portable M4.2 identity accepted by the existing checkpoint lifecycle."""

    m3b_export_fingerprint: str
    m3b_split_manifest_digest: str
    ordered_train_episode_indices: tuple[int, ...]
    ordered_validation_episode_indices: tuple[int, ...]
    model_config: Mapping[str, object]
    data_contract: Mapping[str, object]
    train_statistics_fingerprint: str
    optimization_config: Mapping[str, object]
    runtime_selection_fingerprint: str
    runtime_selection_source_fingerprint: str
    experiment_manifest_fingerprint: str
    task_token_architecture_fingerprint: str
    training_seed: int
    device: str
    dtype: str
    lerobot_version: str
    torch_version: str
    cuda_version: str | None
    git_commit: str
    git_dirty: bool
    schema_version: str = TASK_TOKEN_RUN_SCHEMA
    model_label: str = TASK_TOKEN_MODEL_LABEL
    run_fingerprint: str = ""

    def __post_init__(self) -> None:
        if (
            self.schema_version != TASK_TOKEN_RUN_SCHEMA
            or self.model_label != TASK_TOKEN_MODEL_LABEL
        ):
            raise M42TrainingContractError("unsupported TaskToken run identity")
        for name in (
            "m3b_export_fingerprint",
            "m3b_split_manifest_digest",
            "train_statistics_fingerprint",
            "runtime_selection_fingerprint",
            "runtime_selection_source_fingerprint",
            "experiment_manifest_fingerprint",
            "task_token_architecture_fingerprint",
        ):
            _digest(cast(str, getattr(self, name)), name)
        if len(self.ordered_train_episode_indices) != 288:
            raise M42TrainingContractError("TaskToken training requires all 288 M3B train episodes")
        if len(self.ordered_validation_episode_indices) != 36:
            raise M42TrainingContractError(
                "TaskToken selection requires all 36 M3B validation episodes"
            )
        if (
            tuple(sorted(set(self.ordered_train_episode_indices)))
            != self.ordered_train_episode_indices
        ):
            raise M42TrainingContractError("train episode indices must be unique and ordered")
        if (
            tuple(sorted(set(self.ordered_validation_episode_indices)))
            != self.ordered_validation_episode_indices
        ):
            raise M42TrainingContractError("validation episode indices must be unique and ordered")
        if set(self.ordered_train_episode_indices) & set(self.ordered_validation_episode_indices):
            raise M42TrainingContractError("TaskToken train and validation views overlap")
        if self.training_seed != 0:
            raise M42TrainingContractError("TaskToken must keep the primary M4 seed policy")
        if self.device not in {"cpu", "cuda"} or self.dtype != "float32":
            raise M42TrainingContractError("TaskToken device/dtype contract is invalid")
        _git_commit(self.git_commit)
        if self.lerobot_version != "0.6.0":
            raise M42TrainingContractError("TaskToken supports only inspected LeRobot 0.6.0")
        if not isinstance(self.git_dirty, bool):
            raise M42TrainingContractError("git_dirty must be boolean")
        if self.git_dirty:
            raise M42TrainingContractError("TaskToken evidence requires a clean Git worktree")
        object.__setattr__(self, "model_config", _mapping(self.model_config, "model_config"))
        object.__setattr__(self, "data_contract", _mapping(self.data_contract, "data_contract"))
        object.__setattr__(
            self,
            "optimization_config",
            _mapping(self.optimization_config, "optimization_config"),
        )
        expected = self.compute_fingerprint()
        if not self.run_fingerprint:
            object.__setattr__(self, "run_fingerprint", expected)
        elif self.run_fingerprint != expected:
            raise M42TrainingContractError("TaskToken run fingerprint is inconsistent")

    def identity_payload(self) -> dict[str, object]:
        payload = self.to_dict()
        payload.pop("run_fingerprint")
        return payload

    def compute_fingerprint(self) -> str:
        return canonical_fingerprint(self.identity_payload())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_label": self.model_label,
            "m3b_export_fingerprint": self.m3b_export_fingerprint,
            "m3b_split_manifest_digest": self.m3b_split_manifest_digest,
            "ordered_train_episode_indices": list(self.ordered_train_episode_indices),
            "ordered_validation_episode_indices": list(self.ordered_validation_episode_indices),
            "model_config": _thaw_json(self.model_config),
            "data_contract": _thaw_json(self.data_contract),
            "train_statistics_fingerprint": self.train_statistics_fingerprint,
            "optimization_config": _thaw_json(self.optimization_config),
            "runtime_selection_fingerprint": self.runtime_selection_fingerprint,
            "runtime_selection_source_fingerprint": (self.runtime_selection_source_fingerprint),
            "experiment_manifest_fingerprint": self.experiment_manifest_fingerprint,
            "task_token_architecture_fingerprint": (self.task_token_architecture_fingerprint),
            "training_seed": self.training_seed,
            "device": self.device,
            "dtype": self.dtype,
            "lerobot_version": self.lerobot_version,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "run_fingerprint": self.run_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TaskTokenRunIdentity:
        payload = dict(value)
        for name in ("ordered_train_episode_indices", "ordered_validation_episode_indices"):
            raw = payload.get(name)
            if not isinstance(raw, list):
                raise M42TrainingContractError(f"{name} must be a JSON list")
            payload[name] = tuple(raw)
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class TaskTokenTrainingManifest:
    identity: TaskTokenRunIdentity
    task_token_experiment: Mapping[str, object]
    training_state: TrainingState
    checkpoints: tuple[CheckpointRecord, ...] = ()
    training_complete: bool = False
    validation_selection_pending: bool = True
    schema_version: str = TASK_TOKEN_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != TASK_TOKEN_MANIFEST_SCHEMA:
            raise M42TrainingContractError("unsupported TaskToken training manifest")
        if not isinstance(self.training_complete, bool) or not isinstance(
            self.validation_selection_pending, bool
        ):
            raise M42TrainingContractError("TaskToken manifest status fields must be boolean")
        if not isinstance(self.identity, TaskTokenRunIdentity):
            raise TypeError("identity must be a TaskTokenRunIdentity")
        object.__setattr__(
            self,
            "task_token_experiment",
            _mapping(self.task_token_experiment, "task_token_experiment"),
        )
        steps = tuple(record.global_step for record in self.checkpoints)
        if steps != tuple(sorted(set(steps))):
            raise M42TrainingContractError("TaskToken checkpoints must be unique and ordered")
        for record in self.checkpoints:
            if (
                record.run_fingerprint != self.identity.run_fingerprint
                or record.statistics_fingerprint != self.identity.train_statistics_fingerprint
                or record.split_digest != self.identity.m3b_split_manifest_digest
                or record.git_commit != self.identity.git_commit
            ):
                raise M42TrainingContractError("checkpoint provenance differs from TaskToken run")
        if self.training_state.last_checkpoint_fingerprint is not None and (
            not self.checkpoints
            or self.checkpoints[-1].checkpoint_fingerprint
            != self.training_state.last_checkpoint_fingerprint
        ):
            raise M42TrainingContractError("training state does not reference latest checkpoint")
        if self.training_complete and (
            not self.training_state.completed
            or self.training_state.global_step != 100_000
            or len(self.checkpoints) != 20
        ):
            raise M42TrainingContractError(
                "completed TaskToken training requires 100k steps and 20 checkpoints"
            )
        if not self.validation_selection_pending:
            raise M42TrainingContractError(
                "the training command cannot perform checkpoint selection"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_fingerprint": self.identity.run_fingerprint,
            "architecture_fingerprint": (self.identity.task_token_architecture_fingerprint),
            # Selection is a later validation-only immutable artifact.  The
            # training manifest never rewrites history to add its result.
            "selected_checkpoint_fingerprint": None,
            "identity": self.identity.to_dict(),
            "task_token_experiment": _thaw_json(self.task_token_experiment),
            "training_state": self.training_state.to_dict(),
            "checkpoints": [record.to_dict() for record in self.checkpoints],
            "training_complete": self.training_complete,
            "validation_selection_pending": self.validation_selection_pending,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TaskTokenTrainingManifest:
        raw_identity = value.get("identity")
        raw_experiment = value.get("task_token_experiment")
        raw_state = value.get("training_state")
        raw_checkpoints = value.get("checkpoints")
        if not all(
            isinstance(item, Mapping) for item in (raw_identity, raw_experiment, raw_state)
        ) or not isinstance(raw_checkpoints, list):
            raise M42TrainingContractError("TaskToken training manifest is malformed")
        if not all(isinstance(item, Mapping) for item in raw_checkpoints):
            raise M42TrainingContractError("TaskToken checkpoint records are malformed")
        result = cls(
            identity=TaskTokenRunIdentity.from_dict(cast(Mapping[str, object], raw_identity)),
            task_token_experiment=cast(Mapping[str, object], raw_experiment),
            training_state=TrainingState.from_dict(cast(Mapping[str, object], raw_state)),
            checkpoints=tuple(
                CheckpointRecord.from_dict(cast(Mapping[str, object], item))
                for item in raw_checkpoints
            ),
            training_complete=value.get("training_complete", False),
            validation_selection_pending=value.get("validation_selection_pending", True),
            schema_version=value.get("schema_version", ""),
        )
        if (
            value.get("run_fingerprint") != result.identity.run_fingerprint
            or value.get("architecture_fingerprint")
            != result.identity.task_token_architecture_fingerprint
            or value.get("selected_checkpoint_fingerprint") is not None
        ):
            raise M42TrainingContractError("TaskToken manifest projection fields disagree")
        return result


@dataclass(frozen=True, slots=True)
class TaskTokenValidationQueueItem:
    global_step: int
    checkpoint_fingerprint: str
    checkpoint_relative_path: str
    offline_validation_loss: float

    def __post_init__(self) -> None:
        if self.global_step < 1 or self.global_step % 5_000 != 0:
            raise M42TrainingContractError("queued checkpoint step must follow the 5k schedule")
        _digest(self.checkpoint_fingerprint, "checkpoint_fingerprint")
        if (
            not self.checkpoint_relative_path
            or self.checkpoint_relative_path.startswith(("/", "\\"))
            or ".." in self.checkpoint_relative_path
        ):
            raise M42TrainingContractError("checkpoint_relative_path is unsafe")
        if (
            isinstance(self.offline_validation_loss, bool)
            or not isinstance(self.offline_validation_loss, int | float)
            or not math.isfinite(float(self.offline_validation_loss))
        ):
            raise M42TrainingContractError("offline_validation_loss must be finite and numeric")

    def to_dict(self) -> dict[str, object]:
        return {
            "global_step": self.global_step,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "checkpoint_relative_path": self.checkpoint_relative_path,
            "offline_validation_loss": float(self.offline_validation_loss),
        }


@dataclass(frozen=True, slots=True)
class TaskTokenValidationQueue:
    run_fingerprint: str
    m3b_dataset_fingerprint: str
    split_digest: str
    train_statistics_fingerprint: str
    runtime_selection_fingerprint: str
    experiment_manifest_fingerprint: str
    task_token_architecture_fingerprint: str
    validation_schedule_fingerprint: str
    ordered_validation_episode_indices: tuple[int, ...]
    checkpoints: tuple[TaskTokenValidationQueueItem, ...]
    git_commit: str
    complete: bool
    schema_version: str = TASK_TOKEN_VALIDATION_QUEUE_SCHEMA
    queue_fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != TASK_TOKEN_VALIDATION_QUEUE_SCHEMA:
            raise M42TrainingContractError("unsupported TaskToken validation queue")
        if not isinstance(self.complete, bool):
            raise M42TrainingContractError("validation queue complete flag must be boolean")
        for name in (
            "run_fingerprint",
            "m3b_dataset_fingerprint",
            "split_digest",
            "train_statistics_fingerprint",
            "runtime_selection_fingerprint",
            "experiment_manifest_fingerprint",
            "task_token_architecture_fingerprint",
            "validation_schedule_fingerprint",
        ):
            _digest(cast(str, getattr(self, name)), name)
        _git_commit(self.git_commit)
        if len(self.ordered_validation_episode_indices) != 36:
            raise M42TrainingContractError("validation queue must bind all 36 validation episodes")
        if (
            tuple(sorted(set(self.ordered_validation_episode_indices)))
            != self.ordered_validation_episode_indices
        ):
            raise M42TrainingContractError(
                "validation queue episode indices must be unique and ordered"
            )
        steps = tuple(item.global_step for item in self.checkpoints)
        if steps != tuple(sorted(set(steps))):
            raise M42TrainingContractError("validation queue steps must be unique and ordered")
        if self.complete and steps != tuple(range(5_000, 100_001, 5_000)):
            raise M42TrainingContractError(
                "complete validation queue must contain all 20 checkpoints"
            )
        expected = canonical_fingerprint(self.identity_payload())
        if not self.queue_fingerprint:
            object.__setattr__(self, "queue_fingerprint", expected)
        elif self.queue_fingerprint != expected:
            raise M42TrainingContractError("validation queue fingerprint is inconsistent")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_fingerprint": self.run_fingerprint,
            "m3b_dataset_fingerprint": self.m3b_dataset_fingerprint,
            "split_digest": self.split_digest,
            "train_statistics_fingerprint": self.train_statistics_fingerprint,
            "runtime_selection_fingerprint": self.runtime_selection_fingerprint,
            "experiment_manifest_fingerprint": self.experiment_manifest_fingerprint,
            "task_token_architecture_fingerprint": (self.task_token_architecture_fingerprint),
            "validation_schedule_fingerprint": self.validation_schedule_fingerprint,
            "ordered_validation_episode_indices": list(self.ordered_validation_episode_indices),
            "checkpoints": [item.to_dict() for item in self.checkpoints],
            "git_commit": self.git_commit,
            "complete": self.complete,
            "selection_source": "m3b_validation_only",
            "forbidden_selection_sources": [
                "m3b_test",
                "m4_fresh_seed",
                "m42_dev_v0",
                "m42_final_v0",
            ],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.identity_payload(), "queue_fingerprint": self.queue_fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TaskTokenValidationQueue:
        """Restore and revalidate a persisted validation queue."""
        raw_indices = value.get("ordered_validation_episode_indices")
        raw_checkpoints = value.get("checkpoints")
        if not isinstance(raw_indices, list) or not isinstance(raw_checkpoints, list):
            raise M42TrainingContractError("TaskToken validation queue is malformed")
        if not all(isinstance(item, Mapping) for item in raw_checkpoints):
            raise M42TrainingContractError("TaskToken validation queue items are malformed")
        try:
            return cls(
                run_fingerprint=cast(str, value["run_fingerprint"]),
                m3b_dataset_fingerprint=cast(str, value["m3b_dataset_fingerprint"]),
                split_digest=cast(str, value["split_digest"]),
                train_statistics_fingerprint=cast(str, value["train_statistics_fingerprint"]),
                runtime_selection_fingerprint=cast(str, value["runtime_selection_fingerprint"]),
                experiment_manifest_fingerprint=cast(str, value["experiment_manifest_fingerprint"]),
                task_token_architecture_fingerprint=cast(
                    str, value["task_token_architecture_fingerprint"]
                ),
                validation_schedule_fingerprint=cast(str, value["validation_schedule_fingerprint"]),
                ordered_validation_episode_indices=tuple(int(item) for item in raw_indices),
                checkpoints=tuple(
                    TaskTokenValidationQueueItem(
                        global_step=int(item["global_step"]),
                        checkpoint_fingerprint=cast(str, item["checkpoint_fingerprint"]),
                        checkpoint_relative_path=cast(str, item["checkpoint_relative_path"]),
                        offline_validation_loss=float(item["offline_validation_loss"]),
                    )
                    for item in cast(list[Mapping[str, object]], raw_checkpoints)
                ),
                git_commit=cast(str, value["git_commit"]),
                complete=cast(bool, value["complete"]),
                schema_version=cast(str, value["schema_version"]),
                queue_fingerprint=cast(str, value.get("queue_fingerprint", "")),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise M42TrainingContractError(
                "TaskToken validation queue fields are malformed"
            ) from error


@dataclass(frozen=True, slots=True)
class CompletedTaskTokenRunEvidence:
    run_root: Path
    manifest: TaskTokenTrainingManifest
    queue: TaskTokenValidationQueue
    summary: Mapping[str, object]
    completion: Mapping[str, object]


def validate_completed_task_token_run(
    run_root: str | Path,
    *,
    fresh_reload: bool = True,
) -> CompletedTaskTokenRunEvidence:
    """Validate the complete 100k/20-checkpoint TaskToken evidence chain."""
    root = Path(run_root).resolve()
    if not root.is_dir() or root.is_symlink() or root.is_junction():
        raise M42TrainingContractError("completed TaskToken run root must be one real directory")

    def read(name: str) -> dict[str, object]:
        path = root / name
        if not path.is_file() or path.is_symlink() or path.is_junction():
            raise M42TrainingContractError(f"completed TaskToken artifact is unsafe: {path}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise M42TrainingContractError(
                f"cannot read completed TaskToken artifact {path}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise M42TrainingContractError(f"completed TaskToken artifact is not an object: {path}")
        return value

    manifest = TaskTokenTrainingManifest.from_dict(read("run_manifest.json"))
    queue = TaskTokenValidationQueue.from_dict(read("validation_queue.json"))
    summary = read("reports/training_summary.json")
    completion = read("training_complete.json")
    identity = manifest.identity
    final_checkpoint = manifest.checkpoints[-1] if manifest.checkpoints else None
    if (
        not manifest.training_complete
        or not queue.complete
        or final_checkpoint is None
        or queue.run_fingerprint != identity.run_fingerprint
        or queue.m3b_dataset_fingerprint != identity.m3b_export_fingerprint
        or queue.split_digest != identity.m3b_split_manifest_digest
        or queue.train_statistics_fingerprint != identity.train_statistics_fingerprint
        or queue.runtime_selection_fingerprint != identity.runtime_selection_fingerprint
        or queue.experiment_manifest_fingerprint != identity.experiment_manifest_fingerprint
        or queue.task_token_architecture_fingerprint != identity.task_token_architecture_fingerprint
        or queue.git_commit != identity.git_commit
        or tuple(item.checkpoint_fingerprint for item in queue.checkpoints)
        != tuple(item.checkpoint_fingerprint for item in manifest.checkpoints)
        or completion.get("schema_version") != TASK_TOKEN_TRAINING_COMPLETION_SCHEMA
        or completion.get("run_fingerprint") != identity.run_fingerprint
        or completion.get("experiment_manifest_fingerprint")
        != identity.experiment_manifest_fingerprint
        or completion.get("training_summary_fingerprint") != f"sha256:{sha256_hex(summary)}"
        or completion.get("validation_queue_fingerprint") != queue.queue_fingerprint
        or completion.get("final_checkpoint_fingerprint") != final_checkpoint.checkpoint_fingerprint
        or completion.get("checkpoint_selection_pending") is not True
        or summary.get("schema_version") != TASK_TOKEN_TRAINING_SUMMARY_SCHEMA
        or summary.get("passed") is not True
        or summary.get("run_fingerprint") != identity.run_fingerprint
        or summary.get("experiment_manifest_fingerprint")
        != identity.experiment_manifest_fingerprint
        or summary.get("checkpoint_count") != 20
        or summary.get("validation_queue_fingerprint") != queue.queue_fingerprint
        or summary.get("final_checkpoint_fingerprint") != final_checkpoint.checkpoint_fingerprint
        or summary.get("task_token_training_completed") is not True
        or summary.get("task_token_checkpoint_selected") is not False
        or summary.get("test_accessed") is not False
        or summary.get("development_schedule_accessed_for_selection") is not False
        or summary.get("final_schedule_accessed") is not False
    ):
        raise M42TrainingContractError(
            "completed TaskToken manifest, queue, summary, or completion marker disagree"
        )
    for checkpoint in manifest.checkpoints:
        marker = root / checkpoint.relative_path / "complete.json"
        if not marker.is_file() or marker.is_symlink() or marker.is_junction():
            raise M42TrainingContractError(
                f"completed TaskToken checkpoint marker is absent: {marker}"
            )
    if fresh_reload:
        from langmani.policies.act_checkpoint import load_act_checkpoint
        from langmani.policies.act_task_token import validate_task_token_policy

        loaded = load_act_checkpoint(
            run_root=root,
            checkpoint_relative_path=final_checkpoint.relative_path,
            expected_identity=identity,
            restore_rng=False,
        )
        if loaded.record.checkpoint_fingerprint != final_checkpoint.checkpoint_fingerprint:
            raise M42TrainingContractError("fresh TaskToken reload changed checkpoint identity")
        validate_task_token_policy(loaded.policy)
    return CompletedTaskTokenRunEvidence(
        run_root=root,
        manifest=manifest,
        queue=queue,
        summary=_FrozenMapping(summary),
        completion=_FrozenMapping(completion),
    )


__all__ = [
    "CompletedTaskTokenRunEvidence",
    "M42TrainingContractError",
    "RUNTIME_SELECTION_SCHEMA",
    "RuntimeSelectionContract",
    "TASK_TOKEN_MANIFEST_SCHEMA",
    "TASK_TOKEN_MODEL_LABEL",
    "TASK_TOKEN_RUN_SCHEMA",
    "TASK_TOKEN_TRAINING_COMPLETION_SCHEMA",
    "TASK_TOKEN_FAIR_COMPARISON_SCHEMA",
    "TASK_TOKEN_TRAINING_SUMMARY_SCHEMA",
    "TASK_TOKEN_VALIDATION_QUEUE_SCHEMA",
    "TaskTokenRunIdentity",
    "TaskTokenFairComparisonContract",
    "TaskTokenTrainingManifest",
    "TaskTokenValidationQueue",
    "TaskTokenValidationQueueItem",
    "build_task_token_fair_comparison_contract",
    "canonical_fingerprint",
    "fingerprint_owned_task_token_runs",
    "load_runtime_selection",
    "validate_completed_task_token_run",
]
