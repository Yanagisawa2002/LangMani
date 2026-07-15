"""Immutable, JSON-serializable contracts for the M4.2 oracle-control milestone.

M4.2 deliberately owns a new identity namespace.  None of the records in this
module are added to the historical M4 run identity, so loading this module
cannot change fingerprints of the eight completed M4 experiments.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from statistics import median
from typing import cast

from langmani.datasets.identity import sha256_hex
from langmani.policies.act_conditioning import CANONICAL_TASK_IDS

M42_SCHEMA_VERSION = "langmani-m42-oracle-control-v0"
M42_RUNTIME_SCHEMA_VERSION = "langmani-m42-runtime-v0"
M42_TASK_TOKEN_MAPPING_VERSION = "CanonicalTaskTokenV0"
M42_TASK_TOKEN_ARCHITECTURE_VERSION = "LangManiActEnvironmentTaskTokenV0"
M42_GRIPPER_PROCESSOR_VERSION = "BinaryGripperEnvPostprocessorV0"
M42_DEVELOPMENT_SCHEDULE_ID = "m42_dev_v0"
M42_FINAL_SCHEDULE_ID = "m42_final_v0"
M42_ACTION_COMPONENTS = 8
M42_GRIPPER_ACTION_INDEX = 7
M42_PANDA_STATE_COMPONENTS = 9
M42_TASK_COUNT = 6
_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")


class M42ContractError(ValueError):
    """Raised when an immutable M4.2 contract is malformed or inconsistent."""


class _FrozenMapping(Mapping[str, object]):
    """Recursively immutable canonical mapping that remains dataclass-copy safe."""

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
            raise TypeError("M4.2 immutable JSON mappings require string keys")
        return _FrozenMapping(cast(Mapping[str, object], value))
    if isinstance(value, tuple | list):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, bool | int | float | str | StrEnum):
        return value
    raise TypeError(f"unsupported M4.2 immutable JSON value {type(value).__name__}")


class M42Stage(StrEnum):
    """The three ordered stages that must not be merged into one sweep."""

    RUNTIME_ABLATION = "runtime_ablation"
    TASK_TOKEN_TRAINING = "task_token_training"
    FINAL_BENCHMARK = "final_benchmark"


class GripperRuntimeMode(StrEnum):
    """Explicit gripper runtimes; these do not replace reject/project bounds."""

    PROJECT = "project"
    BINARY = "binary"


class PostGraspPhase(StrEnum):
    NEVER_GRASPED_TARGET = "never_grasped_target"
    TARGET_GRASPED_NOT_LIFTED = "target_grasped_not_lifted"
    LIFTED_NOT_TRANSPORTED = "lifted_not_transported"
    TRANSPORTED_NOT_DESCENDED = "transported_not_descended"
    DESCENDED_NOT_RELEASED = "descended_not_released"
    RELEASED_OUTSIDE_SUCCESS_REGION = "released_outside_success_region"
    RELEASED_BUT_NOT_STATIC = "released_but_not_static"
    TARGET_SUCCESS_THEN_LOST = "target_success_then_lost"
    TIMED_OUT_AFTER_TARGET_GRASP = "timed_out_after_target_grasp"
    WRONG_OBJECT_INTERACTION = "wrong_object_interaction"
    ENVIRONMENT_FAILURE = "environment_failure"
    SUCCESS = "success"


class M42SelectionKind(StrEnum):
    EXECUTION_HORIZON = "execution_horizon"
    GRIPPER_RUNTIME = "gripper_runtime"
    TASK_TOKEN_CHECKPOINT = "task_token_checkpoint"


class M42Decision(StrEnum):
    GO_FOR_SMOLVLA = "go_for_smolvla"
    REMAIN_IN_ORACLE_CONTROL_LAYER = "remain_in_oracle_control_layer"


def _json_value(value: object) -> object:
    if isinstance(value, StrEnum):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("M4.2 JSON mapping keys must be strings")
            result[key] = _json_value(item)
        return result
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"unsupported M4.2 JSON value {type(value).__name__}")


class _JsonRecord:
    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], _json_value(asdict(self)))

    @property
    def fingerprint(self) -> str:
        return f"sha256:{sha256_hex(self.to_dict())}"


def _positive_int(value: int, name: str, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise M42ContractError(f"{name} must be >= {minimum}")


def _finite(value: float, name: str, *, minimum: float = 0.0) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(float(value)) or float(value) < minimum:
        raise M42ContractError(f"{name} must be finite and >= {minimum}")


def _digest(value: str, name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise M42ContractError(f"{name} must be a lowercase SHA-256 digest")


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise M42ContractError(f"{name} must be a non-empty string")


def _immutable_json_mapping(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    normalized = cast(dict[str, object], _json_value(value))
    return _FrozenMapping(normalized)


@dataclass(frozen=True, slots=True)
class ExecutionHorizonConfig(_JsonRecord):
    """Number of actions consumed from each unchanged ACT 50-action chunk."""

    actions_per_query: int
    action_chunk_size: int = 50
    version: str = "ExecutionHorizonV0"

    def __post_init__(self) -> None:
        if self.actions_per_query not in (1, 5, 10):
            raise M42ContractError("actions_per_query must be exactly one of 1, 5, or 10")
        if self.action_chunk_size != 50:
            raise M42ContractError("M4.2 keeps the frozen ACT action chunk size at 50")
        _nonempty(self.version, "version")


@dataclass(frozen=True, slots=True)
class RuntimeAblationConfig(_JsonRecord):
    """Complete frozen inputs for either horizon or gripper runtime evaluation."""

    schedule_id: str
    schedule_fingerprint: str
    m3b_dataset_fingerprint: str
    mixed_task_onehot_checkpoint_fingerprint: str
    representative_per_task_checkpoint_fingerprint: str
    representative_task_id: str
    horizons: tuple[int, ...] = (10, 5, 1)
    gripper_modes: tuple[GripperRuntimeMode, ...] = (GripperRuntimeMode.PROJECT,)
    maximum_episode_steps: int = 200
    schema_version: str = M42_RUNTIME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schedule_id != M42_DEVELOPMENT_SCHEDULE_ID:
            raise M42ContractError("runtime selection may use only m42_dev_v0")
        for name in (
            "schedule_fingerprint",
            "m3b_dataset_fingerprint",
            "mixed_task_onehot_checkpoint_fingerprint",
            "representative_per_task_checkpoint_fingerprint",
        ):
            _digest(cast(str, getattr(self, name)), name)
        if self.representative_task_id not in CANONICAL_TASK_IDS:
            raise M42ContractError("representative_task_id must use the canonical six-task mapping")
        if tuple(self.horizons) not in ((10, 5, 1), (10,), (5,), (1,)):
            raise M42ContractError("horizon ablation must preserve the declared 10, 5, 1 order")
        if not self.gripper_modes or any(
            not isinstance(mode, GripperRuntimeMode) for mode in self.gripper_modes
        ):
            raise M42ContractError("gripper_modes must contain explicit GripperRuntimeMode values")
        if len(set(self.gripper_modes)) != len(self.gripper_modes):
            raise M42ContractError("gripper_modes must not contain duplicates")
        _positive_int(self.maximum_episode_steps, "maximum_episode_steps")


@dataclass(frozen=True, slots=True)
class RuntimeAblationResult(_JsonRecord):
    """Compact aggregate used by deterministic horizon/gripper selection."""

    config_fingerprint: str
    schedule_id: str
    schedule_fingerprint: str
    model_label: str
    task_id: str | None
    execution_horizon: int
    gripper_mode: GripperRuntimeMode
    episode_count: int
    successes: int
    post_grasp_timeouts: int
    wrong_object_interactions: int
    wrong_object_grasp_count: int
    wrong_object_in_target_bin_count: int
    target_in_wrong_bin_count: int
    target_off_table_count: int
    invalid_action_count: int
    successful_episode_steps: tuple[int, ...]
    policy_query_count: int
    release_sign_transitions: int
    grasp_sign_transitions: int
    unnecessary_gripper_sign_transitions: int
    action_metrics: Mapping[str, object] = field(default_factory=dict)
    latency_metrics: Mapping[str, object] = field(default_factory=dict)
    report_fingerprint: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "gripper_mode", GripperRuntimeMode(self.gripper_mode))
        _digest(self.config_fingerprint, "config_fingerprint")
        _digest(self.schedule_fingerprint, "schedule_fingerprint")
        if self.schedule_id != M42_DEVELOPMENT_SCHEDULE_ID:
            raise M42ContractError("runtime ablation results must use m42_dev_v0")
        _nonempty(self.model_label, "model_label")
        if self.task_id is not None and self.task_id not in CANONICAL_TASK_IDS:
            raise M42ContractError("task_id must be canonical when present")
        ExecutionHorizonConfig(self.execution_horizon)
        for name in (
            "episode_count",
            "successes",
            "post_grasp_timeouts",
            "wrong_object_interactions",
            "wrong_object_grasp_count",
            "wrong_object_in_target_bin_count",
            "target_in_wrong_bin_count",
            "target_off_table_count",
            "invalid_action_count",
            "policy_query_count",
            "release_sign_transitions",
            "grasp_sign_transitions",
            "unnecessary_gripper_sign_transitions",
        ):
            _positive_int(cast(int, getattr(self, name)), name, allow_zero=True)
        if self.episode_count < 1:
            raise M42ContractError("episode_count must be positive")
        if any(
            cast(int, getattr(self, name)) > self.episode_count
            for name in (
                "successes",
                "post_grasp_timeouts",
                "wrong_object_interactions",
                "wrong_object_grasp_count",
                "wrong_object_in_target_bin_count",
                "target_in_wrong_bin_count",
                "target_off_table_count",
                "invalid_action_count",
            )
        ):
            raise M42ContractError("episode outcome counts cannot exceed episode_count")
        if not (
            max(self.wrong_object_grasp_count, self.wrong_object_in_target_bin_count)
            <= self.wrong_object_interactions
            <= self.wrong_object_grasp_count + self.wrong_object_in_target_bin_count
        ):
            raise M42ContractError(
                "wrong_object_interactions must be the episode-level union of its components"
            )
        if len(self.successful_episode_steps) != self.successes:
            raise M42ContractError("successful_episode_steps must contain one entry per success")
        for step in self.successful_episode_steps:
            _positive_int(step, "successful episode step")
        object.__setattr__(
            self, "action_metrics", _immutable_json_mapping(self.action_metrics, "action_metrics")
        )
        object.__setattr__(
            self,
            "latency_metrics",
            _immutable_json_mapping(self.latency_metrics, "latency_metrics"),
        )
        if self.report_fingerprint is not None:
            _digest(self.report_fingerprint, "report_fingerprint")

    @property
    def success_rate(self) -> float:
        return self.successes / self.episode_count

    @property
    def post_grasp_timeout_rate(self) -> float:
        return self.post_grasp_timeouts / self.episode_count

    @property
    def wrong_object_interaction_rate(self) -> float:
        return self.wrong_object_interactions / self.episode_count

    @property
    def median_successful_steps(self) -> float:
        return (
            float(median(self.successful_episode_steps))
            if self.successful_episode_steps
            else math.inf
        )


@dataclass(frozen=True, slots=True)
class PostGraspFailureRecord(_JsonRecord):
    scene_seed: int
    task_spec: Mapping[str, object]
    selected_checkpoint_fingerprint: str
    execution_horizon: int
    first_target_grasp_step: int
    maximum_target_height: float
    closest_target_to_bin_distance: float
    first_release_command_step: int | None
    final_target_position: tuple[float, float, float]
    final_target_velocity: tuple[float, float, float]
    final_environment_evaluation: Mapping[str, object]
    inferred_failure_phase: PostGraspPhase
    projection_summary: Mapping[str, object]
    rollout_video_path: str | None = None

    def __post_init__(self) -> None:
        _positive_int(self.scene_seed, "scene_seed", allow_zero=True)
        _digest(self.selected_checkpoint_fingerprint, "selected_checkpoint_fingerprint")
        ExecutionHorizonConfig(self.execution_horizon)
        _positive_int(self.first_target_grasp_step, "first_target_grasp_step", allow_zero=True)
        if self.first_release_command_step is not None:
            _positive_int(
                self.first_release_command_step, "first_release_command_step", allow_zero=True
            )
        _finite(self.maximum_target_height, "maximum_target_height")
        _finite(self.closest_target_to_bin_distance, "closest_target_to_bin_distance")
        for name, vector in (
            ("final_target_position", self.final_target_position),
            ("final_target_velocity", self.final_target_velocity),
        ):
            if len(vector) != 3:
                raise M42ContractError(f"{name} must contain exactly three values")
            for item in vector:
                if not math.isfinite(float(item)):
                    raise M42ContractError(f"{name} must contain finite values")
        object.__setattr__(self, "task_spec", _immutable_json_mapping(self.task_spec, "task_spec"))
        object.__setattr__(
            self,
            "final_environment_evaluation",
            _immutable_json_mapping(
                self.final_environment_evaluation, "final_environment_evaluation"
            ),
        )
        object.__setattr__(
            self,
            "projection_summary",
            _immutable_json_mapping(self.projection_summary, "projection_summary"),
        )
        if self.inferred_failure_phase is PostGraspPhase.SUCCESS:
            raise M42ContractError("a PostGraspFailureRecord cannot use the success phase")


@dataclass(frozen=True, slots=True)
class PostGraspFailureSummary(_JsonRecord):
    schedule_id: str
    episode_count: int
    phase_counts: Mapping[str, object]
    failed_record_count: int

    def __post_init__(self) -> None:
        _nonempty(self.schedule_id, "schedule_id")
        _positive_int(self.episode_count, "episode_count", allow_zero=True)
        _positive_int(self.failed_record_count, "failed_record_count", allow_zero=True)
        counts = _immutable_json_mapping(self.phase_counts, "phase_counts")
        allowed = {phase.value for phase in PostGraspPhase}
        if not set(counts).issubset(allowed):
            raise M42ContractError("phase_counts contains an unknown post-grasp phase")
        total = 0
        for key, value in counts.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise M42ContractError(f"phase count {key!r} must be non-negative")
            total += value
        if total != self.episode_count:
            raise M42ContractError("phase_counts must cover every episode exactly once")
        if self.failed_record_count > self.episode_count:
            raise M42ContractError("failed_record_count cannot exceed episode_count")
        object.__setattr__(self, "phase_counts", counts)


@dataclass(frozen=True, slots=True)
class TaskTokenMapping(_JsonRecord):
    mapping_version: str = M42_TASK_TOKEN_MAPPING_VERSION
    ordered_task_ids: tuple[str, ...] = CANONICAL_TASK_IDS
    dtype: str = "float32"
    shape: tuple[int, ...] = (M42_TASK_COUNT,)

    def __post_init__(self) -> None:
        if self.mapping_version != M42_TASK_TOKEN_MAPPING_VERSION:
            raise M42ContractError("unsupported task-token mapping version")
        if tuple(self.ordered_task_ids) != CANONICAL_TASK_IDS:
            raise M42ContractError("TaskToken ordering must equal CanonicalTaskOneHotV0")
        if self.dtype != "float32" or tuple(self.shape) != (M42_TASK_COUNT,):
            raise M42ContractError("TaskToken command input must be float32 with shape (6,)")


@dataclass(frozen=True, slots=True)
class TaskTokenConfig(_JsonRecord):
    """The single permitted M4.2 task-token architecture."""

    mapping: TaskTokenMapping = field(default_factory=TaskTokenMapping)
    architecture_version: str = M42_TASK_TOKEN_ARCHITECTURE_VERSION
    hidden_dimension: int = 512
    panda_state_dimension: int = M42_PANDA_STATE_COMPONENTS
    task_input_dimension: int = M42_TASK_COUNT
    task_feature_key: str = "observation.environment_state"
    injection_location: str = "act_transformer_encoder_environment_token"
    input_projection: str = "linear_6_to_hidden_dimension"
    action_dimension: int = M42_ACTION_COMPONENTS

    def __post_init__(self) -> None:
        if not isinstance(self.mapping, TaskTokenMapping):
            raise TypeError("mapping must be a TaskTokenMapping")
        _positive_int(self.hidden_dimension, "hidden_dimension")
        if self.panda_state_dimension != M42_PANDA_STATE_COMPONENTS:
            raise M42ContractError("PandaPolicyStateV0 must remain exactly 9D")
        if self.task_input_dimension != M42_TASK_COUNT:
            raise M42ContractError("CanonicalTaskTokenV0 must have six inputs")
        if self.task_feature_key != "observation.environment_state":
            raise M42ContractError("TaskToken must use LeRobot's dedicated ENV feature token")
        if self.injection_location != "act_transformer_encoder_environment_token":
            raise M42ContractError("unsupported TaskToken injection location")
        if self.action_dimension != M42_ACTION_COMPONENTS:
            raise M42ContractError("M1 action semantics require exactly eight outputs")


@dataclass(frozen=True, slots=True)
class TaskTokenExperimentConfig(_JsonRecord):
    dataset_fingerprint: str
    split_digest: str
    train_statistics_fingerprint: str
    implementation_git_commit: str
    runtime_selection_fingerprint: str
    token_config: TaskTokenConfig = field(default_factory=TaskTokenConfig)
    training_steps: int = 100_000
    checkpoint_interval: int = 5_000
    batch_size: int = 32
    seed: int = 0
    validation_only_selection: bool = True
    schema_version: str = M42_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "dataset_fingerprint",
            "split_digest",
            "train_statistics_fingerprint",
            "runtime_selection_fingerprint",
        ):
            _digest(cast(str, getattr(self, name)), name)
        if re.fullmatch(r"[0-9a-f]{40}", self.implementation_git_commit) is None:
            raise M42ContractError("implementation_git_commit must be a full 40-hex commit")
        if self.training_steps != 100_000 or self.checkpoint_interval != 5_000:
            raise M42ContractError("TaskToken must keep the primary M4 100k/5k schedule")
        if self.batch_size != 32:
            raise M42ContractError("TaskToken primary comparison batch size is fixed at 32")
        _positive_int(self.seed, "seed", allow_zero=True)
        if not self.validation_only_selection:
            raise M42ContractError("TaskToken checkpoint selection must be validation-only")


@dataclass(frozen=True, slots=True)
class M42SelectionRecord(_JsonRecord):
    """Immutable content-bound selection made before final schedule access."""

    selection_kind: M42SelectionKind
    selected_value: str
    candidate_values: tuple[str, ...]
    ranking_version: str
    evidence_fingerprints: tuple[str, ...]
    development_schedule_fingerprint: str | None
    selected_checkpoint_fingerprint: str | None = None
    validation_split_digest: str | None = None
    final_schedule_accessed: bool = False
    locked_at_utc: str = ""
    schema_version: str = M42_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.selection_kind, M42SelectionKind):
            raise TypeError("selection_kind must be M42SelectionKind")
        _nonempty(self.selected_value, "selected_value")
        if not self.candidate_values or self.selected_value not in self.candidate_values:
            raise M42ContractError("selected_value must be one of candidate_values")
        if len(set(self.candidate_values)) != len(self.candidate_values):
            raise M42ContractError("candidate_values must be unique")
        _nonempty(self.ranking_version, "ranking_version")
        if not self.evidence_fingerprints:
            raise M42ContractError("selection requires evidence fingerprints")
        for digest in self.evidence_fingerprints:
            _digest(digest, "evidence fingerprint")
        if self.development_schedule_fingerprint is not None:
            _digest(self.development_schedule_fingerprint, "development_schedule_fingerprint")
        if not isinstance(self.final_schedule_accessed, bool):
            raise TypeError("final_schedule_accessed must be boolean")
        if self.final_schedule_accessed:
            raise M42ContractError("M4.2 selection records must never access m42_final_v0")
        if self.selection_kind is M42SelectionKind.TASK_TOKEN_CHECKPOINT:
            if self.development_schedule_fingerprint is not None:
                raise M42ContractError("TaskToken checkpoint selection cannot use m42_dev_v0")
            if self.selected_checkpoint_fingerprint != self.selected_value:
                raise M42ContractError("checkpoint selection must bind selected checkpoint digest")
            if self.validation_split_digest is None:
                raise M42ContractError(
                    "checkpoint selection requires the M3B validation split digest"
                )
        else:
            if self.development_schedule_fingerprint is None:
                raise M42ContractError("runtime selection requires m42_dev_v0 evidence")
        if self.selected_checkpoint_fingerprint is not None:
            _digest(self.selected_checkpoint_fingerprint, "selected_checkpoint_fingerprint")
        if self.validation_split_digest is not None:
            _digest(self.validation_split_digest, "validation_split_digest")
        _nonempty(self.locked_at_utc, "locked_at_utc")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> M42SelectionRecord:
        expected = {
            "selection_kind",
            "selected_value",
            "candidate_values",
            "ranking_version",
            "evidence_fingerprints",
            "development_schedule_fingerprint",
            "selected_checkpoint_fingerprint",
            "validation_split_digest",
            "final_schedule_accessed",
            "locked_at_utc",
            "schema_version",
        }
        if set(value) != expected:
            raise M42ContractError("selection record fields differ from the frozen schema")
        candidates = value["candidate_values"]
        evidence = value["evidence_fingerprints"]
        if not isinstance(candidates, list | tuple) or not isinstance(evidence, list | tuple):
            raise M42ContractError("selection candidate/evidence fields must be arrays")
        try:
            return cls(
                selection_kind=M42SelectionKind(cast(str, value["selection_kind"])),
                selected_value=cast(str, value["selected_value"]),
                candidate_values=tuple(cast(Sequence[str], candidates)),
                ranking_version=cast(str, value["ranking_version"]),
                evidence_fingerprints=tuple(cast(Sequence[str], evidence)),
                development_schedule_fingerprint=cast(
                    str | None, value["development_schedule_fingerprint"]
                ),
                selected_checkpoint_fingerprint=cast(
                    str | None, value["selected_checkpoint_fingerprint"]
                ),
                validation_split_digest=cast(str | None, value["validation_split_digest"]),
                final_schedule_accessed=cast(bool, value["final_schedule_accessed"]),
                locked_at_utc=cast(str, value["locked_at_utc"]),
                schema_version=cast(str, value["schema_version"]),
            )
        except (TypeError, ValueError) as error:
            raise M42ContractError("selection record values are malformed") from error


@dataclass(frozen=True, slots=True)
class TaskTokenValidationResult(_JsonRecord):
    """One checkpoint evaluated only on the fixed 36-episode M3B validation split."""

    checkpoint_fingerprint: str
    checkpoint_step: int
    validation_schedule_fingerprint: str
    validation_split_digest: str
    episode_count: int
    success_count: int
    wrong_object_interaction_count: int
    target_off_table_count: int
    timeout_count: int
    offline_validation_action_loss: float
    development_schedule_accessed: bool = False
    final_schedule_accessed: bool = False
    test_split_accessed: bool = False

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_fingerprint",
            "validation_schedule_fingerprint",
            "validation_split_digest",
        ):
            _digest(cast(str, getattr(self, name)), name)
        _positive_int(self.checkpoint_step, "checkpoint_step")
        if self.checkpoint_step % 5_000 != 0 or self.checkpoint_step > 100_000:
            raise M42ContractError("TaskToken checkpoint step must follow the 5k/100k schedule")
        if self.episode_count != 36:
            raise M42ContractError("TaskToken selection requires all 36 validation episodes")
        for name in (
            "success_count",
            "wrong_object_interaction_count",
            "target_off_table_count",
            "timeout_count",
        ):
            _positive_int(cast(int, getattr(self, name)), name, allow_zero=True)
            if cast(int, getattr(self, name)) > self.episode_count:
                raise M42ContractError(f"{name} cannot exceed episode_count")
        _finite(self.offline_validation_action_loss, "offline_validation_action_loss")
        if any(
            (
                self.development_schedule_accessed,
                self.final_schedule_accessed,
                self.test_split_accessed,
            )
        ):
            raise M42ContractError(
                "TaskToken checkpoint selection may access only the M3B validation split"
            )

    @property
    def success_rate(self) -> float:
        return self.success_count / self.episode_count

    @property
    def wrong_object_interaction_rate(self) -> float:
        return self.wrong_object_interaction_count / self.episode_count

    @property
    def target_off_table_rate(self) -> float:
        return self.target_off_table_count / self.episode_count

    @property
    def timeout_rate(self) -> float:
        return self.timeout_count / self.episode_count


@dataclass(frozen=True, slots=True)
class M42FinalBenchmarkConfig(_JsonRecord):
    final_schedule_fingerprint: str
    horizon_selection_fingerprint: str
    gripper_selection_fingerprint: str
    task_token_selection_fingerprint: str
    final_evaluation_git_commit: str
    model_checkpoint_fingerprints: tuple[str, ...]
    expected_scene_count: int = 30
    expected_task_count: int = 6
    expected_episode_count_per_model: int = 180
    schema_version: str = M42_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "final_schedule_fingerprint",
            "horizon_selection_fingerprint",
            "gripper_selection_fingerprint",
            "task_token_selection_fingerprint",
        ):
            _digest(cast(str, getattr(self, name)), name)
        if re.fullmatch(r"[0-9a-f]{40}", self.final_evaluation_git_commit) is None:
            raise M42ContractError("final_evaluation_git_commit must be a full commit")
        if len(self.model_checkpoint_fingerprints) != 8:
            raise M42ContractError("final benchmark binds six PerTask, OneHot, and TaskToken")
        for value in self.model_checkpoint_fingerprints:
            _digest(value, "model_checkpoint_fingerprint")
        if (
            self.expected_scene_count,
            self.expected_task_count,
            self.expected_episode_count_per_model,
        ) != (
            30,
            6,
            180,
        ):
            raise M42ContractError("sealed final benchmark must be 30 scenes x 6 tasks")


@dataclass(frozen=True, slots=True)
class M42FinalBenchmarkResult(_JsonRecord):
    config_fingerprint: str
    completed: bool
    model_results: Mapping[str, object]
    paired_comparisons: Mapping[str, object]
    task_sensitivity: Mapping[str, object]
    raw_action_metrics: Mapping[str, object]
    runtime_action_metrics: Mapping[str, object]
    failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _digest(self.config_fingerprint, "config_fingerprint")
        for name in (
            "model_results",
            "paired_comparisons",
            "task_sensitivity",
            "raw_action_metrics",
            "runtime_action_metrics",
        ):
            object.__setattr__(
                self,
                name,
                _immutable_json_mapping(cast(Mapping[str, object], getattr(self, name)), name),
            )
        if self.completed and self.failures:
            raise M42ContractError(
                "a completed final benchmark cannot retain infrastructure failures"
            )


@dataclass(frozen=True, slots=True)
class M42GoNoGoMetrics(_JsonRecord):
    task_token_successes: int
    task_token_trials: int
    per_task_aggregate_successes: int
    per_task_aggregate_trials: int
    task_token_successes_by_task: tuple[int, ...]
    timeout_count: int
    wrong_object_grasp_count: int
    wrong_object_in_target_bin_count: int
    target_in_wrong_bin_count: int
    target_off_table_count: int
    arm_projection_count: int
    nan_count: int
    inf_count: int
    malformed_action_count: int
    task_sensitivity_ratio: float

    def __post_init__(self) -> None:
        for name in (
            "task_token_successes",
            "task_token_trials",
            "per_task_aggregate_successes",
            "per_task_aggregate_trials",
            "timeout_count",
            "wrong_object_grasp_count",
            "wrong_object_in_target_bin_count",
            "target_in_wrong_bin_count",
            "target_off_table_count",
            "arm_projection_count",
            "nan_count",
            "inf_count",
            "malformed_action_count",
        ):
            _positive_int(cast(int, getattr(self, name)), name, allow_zero=True)
        if self.task_token_trials != 180 or self.per_task_aggregate_trials != 180:
            raise M42ContractError("go/no-go requires the complete 180-episode paired benchmark")
        if not 0 <= self.task_token_successes <= self.task_token_trials:
            raise M42ContractError("task_token_successes is outside its trial count")
        if not 0 <= self.per_task_aggregate_successes <= self.per_task_aggregate_trials:
            raise M42ContractError("per_task_aggregate_successes is outside its trial count")
        if len(self.task_token_successes_by_task) != M42_TASK_COUNT:
            raise M42ContractError("go/no-go needs six per-task success counts")
        if any(value < 0 or value > 30 for value in self.task_token_successes_by_task):
            raise M42ContractError("per-task successes must be between 0 and 30")
        if sum(self.task_token_successes_by_task) != self.task_token_successes:
            raise M42ContractError("per-task successes must sum to overall successes")
        _finite(self.task_sensitivity_ratio, "task_sensitivity_ratio")


@dataclass(frozen=True, slots=True)
class M42GoNoGoDecision(_JsonRecord):
    metrics: M42GoNoGoMetrics
    decision: M42Decision
    thresholds_passed: Mapping[str, object]
    failed_thresholds: tuple[str, ...]
    final_benchmark_fingerprint: str
    schema_version: str = M42_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.metrics, M42GoNoGoMetrics):
            raise TypeError("metrics must be M42GoNoGoMetrics")
        _digest(self.final_benchmark_fingerprint, "final_benchmark_fingerprint")
        checks = _immutable_json_mapping(self.thresholds_passed, "thresholds_passed")
        if not checks or any(not isinstance(value, bool) for value in checks.values()):
            raise M42ContractError("thresholds_passed must contain boolean threshold outcomes")
        expected_failures = tuple(sorted(key for key, value in checks.items() if not value))
        if tuple(sorted(self.failed_thresholds)) != expected_failures:
            raise M42ContractError("failed_thresholds is inconsistent with threshold outcomes")
        expected_decision = (
            M42Decision.GO_FOR_SMOLVLA
            if not expected_failures
            else M42Decision.REMAIN_IN_ORACLE_CONTROL_LAYER
        )
        if self.decision is not expected_decision:
            raise M42ContractError("decision is inconsistent with threshold outcomes")
        object.__setattr__(self, "thresholds_passed", checks)


@dataclass(frozen=True, slots=True)
class M42ExperimentManifest(_JsonRecord):
    stage: M42Stage
    implementation_git_commit: str
    m3b_dataset_fingerprint: str
    prior_m4_verification_fingerprint: str
    schedule_fingerprints: Mapping[str, object]
    m4_checkpoint_fingerprints: tuple[str, ...]
    m41_runtime_processor_fingerprint: str
    runtime_selection_fingerprints: Mapping[str, object]
    task_token_experiment_fingerprint: str | None
    artifact_paths: Mapping[str, object]
    completed: bool
    schema_version: str = M42_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", M42Stage(self.stage))
        if self.schema_version != M42_SCHEMA_VERSION:
            raise M42ContractError("unsupported M4.2 experiment-manifest schema")
        if self.stage is not M42Stage.RUNTIME_ABLATION:
            raise M42ContractError(
                "the shared M4.2 provenance manifest is locked after runtime ablation"
            )
        if re.fullmatch(r"[0-9a-f]{40}", self.implementation_git_commit) is None:
            raise M42ContractError("implementation_git_commit must be a full commit")
        for name in (
            "m3b_dataset_fingerprint",
            "prior_m4_verification_fingerprint",
            "m41_runtime_processor_fingerprint",
        ):
            _digest(cast(str, getattr(self, name)), name)
        if len(self.m4_checkpoint_fingerprints) != 8:
            raise M42ContractError("manifest must preserve all eight M4 checkpoint fingerprints")
        if len(set(self.m4_checkpoint_fingerprints)) != 8:
            raise M42ContractError("manifest M4 checkpoint fingerprints must be unique")
        for value in self.m4_checkpoint_fingerprints:
            _digest(value, "m4_checkpoint_fingerprint")
        if self.task_token_experiment_fingerprint is not None:
            _digest(self.task_token_experiment_fingerprint, "task_token_experiment_fingerprint")
        for name in ("schedule_fingerprints", "runtime_selection_fingerprints", "artifact_paths"):
            object.__setattr__(
                self,
                name,
                _immutable_json_mapping(cast(Mapping[str, object], getattr(self, name)), name),
            )
        if set(self.schedule_fingerprints) != {
            M42_DEVELOPMENT_SCHEDULE_ID,
            M42_FINAL_SCHEDULE_ID,
        }:
            raise M42ContractError("manifest must bind both locked M4.2 schedules exactly once")
        for value in self.schedule_fingerprints.values():
            _digest(cast(str, value), "schedule_fingerprint")
        if set(self.runtime_selection_fingerprints) != {
            "execution_horizon",
            "gripper_runtime",
        }:
            raise M42ContractError(
                "manifest must bind the horizon and gripper selection records exactly"
            )
        for value in self.runtime_selection_fingerprints.values():
            _digest(cast(str, value), "runtime_selection_fingerprint")
        if not self.completed:
            raise M42ContractError("the shared M4.2 provenance manifest must be complete")
        for name, value in self.artifact_paths.items():
            if (
                not isinstance(value, str)
                or not value
                or Path(value).is_absolute()
                or ".." in Path(value).parts
            ):
                raise M42ContractError(f"artifact path {name!r} must be safe and relative")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> M42ExperimentManifest:
        """Restore and fully revalidate one persisted provenance manifest."""
        if not isinstance(value, Mapping):
            raise TypeError("M4.2 experiment manifest must be a mapping")
        payload = dict(value)
        checkpoints = payload.get("m4_checkpoint_fingerprints")
        if not isinstance(checkpoints, list):
            raise M42ContractError("manifest checkpoint fingerprints must be a JSON list")
        payload["m4_checkpoint_fingerprints"] = tuple(checkpoints)
        try:
            return cls(**payload)
        except (KeyError, TypeError, ValueError) as error:
            raise M42ContractError("M4.2 experiment manifest fields are malformed") from error


def canonical_fingerprint(value: _JsonRecord | Mapping[str, object]) -> str:
    """Return the namespaced SHA-256 used by all new M4.2 artifacts."""
    payload = value.to_dict() if isinstance(value, _JsonRecord) else dict(value)
    return f"sha256:{sha256_hex(payload)}"


def validate_exact_task_order(task_ids: Sequence[str]) -> tuple[str, ...]:
    result = tuple(task_ids)
    if result != CANONICAL_TASK_IDS:
        raise M42ContractError("task ordering must equal the canonical six-task ordering")
    return result


__all__ = [
    "ExecutionHorizonConfig",
    "GripperRuntimeMode",
    "M42_ACTION_COMPONENTS",
    "M42_DEVELOPMENT_SCHEDULE_ID",
    "M42_FINAL_SCHEDULE_ID",
    "M42_GRIPPER_ACTION_INDEX",
    "M42_GRIPPER_PROCESSOR_VERSION",
    "M42_RUNTIME_SCHEMA_VERSION",
    "M42_SCHEMA_VERSION",
    "M42_TASK_TOKEN_ARCHITECTURE_VERSION",
    "M42_TASK_TOKEN_MAPPING_VERSION",
    "M42ContractError",
    "M42Decision",
    "M42ExperimentManifest",
    "M42FinalBenchmarkConfig",
    "M42FinalBenchmarkResult",
    "M42GoNoGoDecision",
    "M42GoNoGoMetrics",
    "M42SelectionKind",
    "M42SelectionRecord",
    "M42Stage",
    "PostGraspFailureRecord",
    "PostGraspFailureSummary",
    "PostGraspPhase",
    "RuntimeAblationConfig",
    "RuntimeAblationResult",
    "TaskTokenConfig",
    "TaskTokenExperimentConfig",
    "TaskTokenMapping",
    "TaskTokenValidationResult",
    "canonical_fingerprint",
    "validate_exact_task_order",
]
