"""Immutable JSON contracts for the M4.3 semantic-alignment audit.

This module deliberately has no Torch, LeRobot, or simulator imports.  The
zero-training audit therefore remains portable and cannot accidentally step an
environment, mutate a checkpoint, or access a sealed evaluation schedule.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import cast

from langmani.datasets.identity import sha256_hex
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import BIN_IDS, OBJECT_IDS, stable_task_id

M43_SCHEMA_VERSION = "langmani-m43-shared-policy-repair-v0"
ACTION_CHUNK_DISTANCE_VERSION = "ActionChunkDistanceV0"
PRIMARY_ACTION_CHUNK_DISTANCE_METRIC = "action_range_normalized_arm_only_execution_window_l2"
ACTION_CHUNK_WINDOW_KEYS = (
    "first_action",
    "first_5_actions",
    "execution_window",
    "full_chunk",
)
M43_ACTION_COMPONENTS = 8
M43_ACTION_CHUNK_SIZE = 50
M43_ARM_ACTION_INDICES = tuple(range(7))
M43_GRIPPER_ACTION_INDICES = (7,)
M43_CANONICAL_TASK_IDS = tuple(stable_task_id(value) for value in CANONICAL_TASK_SPECS)


class M43ContractError(ValueError):
    """Raised when a portable M4.3 record violates its frozen contract."""


class _FrozenMapping(Mapping[str, object]):
    """Recursively immutable mapping with deterministic key order."""

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
            raise TypeError("M4.3 immutable JSON mappings require string keys")
        return _FrozenMapping(cast(Mapping[str, object], value))
    if isinstance(value, tuple | list):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, bool | int | float | str | StrEnum):
        return value
    raise TypeError(f"unsupported M4.3 immutable JSON value {type(value).__name__}")


def _json_value(value: object) -> object:
    if isinstance(value, StrEnum):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"unsupported M4.3 JSON value {type(value).__name__}")


class _JsonRecord:
    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], _json_value(asdict(self)))

    @property
    def fingerprint(self) -> str:
        return f"sha256:{sha256_hex(self.to_dict())}"


def _finite(value: float, name: str, *, strictly_positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(float(value)) or (strictly_positive and float(value) <= 0.0):
        qualifier = "finite and positive" if strictly_positive else "finite"
        raise M43ContractError(f"{name} must be {qualifier}")


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise M43ContractError(f"{name} must be a non-empty string")


def _frozen_mapping(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return _FrozenMapping(value)


def _validate_task_id(value: str, name: str) -> None:
    if value not in M43_CANONICAL_TASK_IDS:
        raise M43ContractError(f"{name} must use CanonicalTaskOneHotV0 ordering")


class SemanticFailureClass(StrEnum):
    """Exhaustive, stable M4.3 rollout outcome taxonomy."""

    NEVER_GRASPED_TARGET = "never_grasped_target"
    WRONG_OBJECT_INTERACTION = "wrong_object_interaction"
    TARGET_GRASPED_NOT_LIFTED = "target_grasped_not_lifted"
    LIFTED_NOT_TRANSPORTED = "lifted_not_transported"
    TRANSPORTED_NOT_DESCENDED = "transported_not_descended"
    DESCENDED_NOT_RELEASED = "descended_not_released"
    RELEASED_OUTSIDE_SUCCESS_REGION = "released_outside_success_region"
    RELEASED_BUT_NOT_STATIC = "released_but_not_static"
    TARGET_SUCCESS_THEN_LOST = "target_success_then_lost"
    TIMED_OUT_AFTER_TARGET_GRASP = "timed_out_after_target_grasp"
    ENVIRONMENT_FAILURE = "environment_failure"
    SUCCESS = "success"


class SemanticAuditConclusion(StrEnum):
    """Permitted evidence-backed interpretations of a candidate shared policy."""

    CONDITION_NOT_USED = "condition_not_used"
    CONDITION_CHANGES_OUTPUT_BUT_WRONG_SEMANTICS = "condition_changes_output_but_wrong_semantics"
    TARGET_OBJECT_CONFUSION = "target_object_confusion"
    DESTINATION_BIN_CONFUSION = "destination_bin_confusion"
    SHARED_CONTROL_DEGRADATION_AFTER_CORRECT_SELECTION = (
        "shared_control_degradation_after_correct_selection"
    )
    POST_GRASP_EXECUTION_FAILURE = "post_grasp_execution_failure"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True, slots=True)
class ActionChunkDistanceConfig(_JsonRecord):
    """All normalization and temporal choices for ActionChunkDistanceV0."""

    action_lower_bounds: tuple[float, ...]
    action_upper_bounds: tuple[float, ...]
    train_action_std: tuple[float, ...]
    locked_execution_horizon: int
    action_dimension: int = M43_ACTION_COMPONENTS
    chunk_size: int = M43_ACTION_CHUNK_SIZE
    arm_action_indices: tuple[int, ...] = M43_ARM_ACTION_INDICES
    gripper_action_indices: tuple[int, ...] = M43_GRIPPER_ACTION_INDICES
    primary_metric: str = PRIMARY_ACTION_CHUNK_DISTANCE_METRIC
    version: str = ACTION_CHUNK_DISTANCE_VERSION

    def __post_init__(self) -> None:
        if self.version != ACTION_CHUNK_DISTANCE_VERSION:
            raise M43ContractError("unknown action-chunk distance version")
        if self.primary_metric != PRIMARY_ACTION_CHUNK_DISTANCE_METRIC:
            raise M43ContractError("M4.3 primary retrieval metric is immutable")
        if (
            self.action_dimension != M43_ACTION_COMPONENTS
            or self.chunk_size != M43_ACTION_CHUNK_SIZE
        ):
            raise M43ContractError("M4.3 requires 50x8 ACT action chunks")
        if self.arm_action_indices != M43_ARM_ACTION_INDICES:
            raise M43ContractError("M4.3 arm-only distance requires action indices 0..6")
        if self.gripper_action_indices != M43_GRIPPER_ACTION_INDICES:
            raise M43ContractError("M4.3 gripper-only distance requires action index 7")
        if isinstance(self.locked_execution_horizon, bool) or not isinstance(
            self.locked_execution_horizon, int
        ):
            raise TypeError("locked_execution_horizon must be an integer")
        if not 1 <= self.locked_execution_horizon <= self.chunk_size:
            raise M43ContractError("locked_execution_horizon must be within the ACT chunk")
        for name in ("action_lower_bounds", "action_upper_bounds", "train_action_std"):
            if len(cast(tuple[float, ...], getattr(self, name))) != self.action_dimension:
                raise M43ContractError(f"{name} must contain eight components")
        for index, (low, high, std) in enumerate(
            zip(
                self.action_lower_bounds,
                self.action_upper_bounds,
                self.train_action_std,
                strict=True,
            )
        ):
            _finite(low, f"action_lower_bounds[{index}]")
            _finite(high, f"action_upper_bounds[{index}]")
            _finite(std, f"train_action_std[{index}]", strictly_positive=True)
            if high <= low:
                raise M43ContractError(f"action bounds component {index} has non-positive range")


def _validate_distance_mapping(
    value: Mapping[str, object], name: str, *, allow_none: bool = False
) -> Mapping[str, object]:
    if tuple(sorted(value)) != tuple(sorted(ACTION_CHUNK_WINDOW_KEYS)):
        raise M43ContractError(f"{name} must contain the four ActionChunkDistanceV0 windows")
    for key, item in value.items():
        if item is None and allow_none:
            continue
        _finite(cast(float, item), f"{name}.{key}")
        if cast(float, item) < 0.0:
            raise M43ContractError(f"{name}.{key} must be non-negative")
    return _frozen_mapping(value, name)


@dataclass(frozen=True, slots=True)
class ActionChunkDistanceResult(_JsonRecord):
    """Complete distance matrix for one candidate/reference ACT chunk pair."""

    raw_l2: Mapping[str, object]
    action_range_normalized_l2: Mapping[str, object]
    train_std_normalized_l2: Mapping[str, object]
    cosine_distance: Mapping[str, object]
    arm_only_raw_l2: Mapping[str, object]
    arm_only_action_range_normalized_l2: Mapping[str, object]
    arm_only_train_std_normalized_l2: Mapping[str, object]
    arm_only_cosine_distance: Mapping[str, object]
    gripper_only_raw_l2: Mapping[str, object]
    gripper_only_action_range_normalized_l2: Mapping[str, object]
    gripper_only_train_std_normalized_l2: Mapping[str, object]
    gripper_only_cosine_distance: Mapping[str, object]
    primary_metric: str
    primary_distance: float
    chunk_length: int = M43_ACTION_CHUNK_SIZE
    action_dimension: int = M43_ACTION_COMPONENTS
    version: str = ACTION_CHUNK_DISTANCE_VERSION

    def __post_init__(self) -> None:
        if self.version != ACTION_CHUNK_DISTANCE_VERSION:
            raise M43ContractError("unknown action-chunk distance result version")
        if self.primary_metric != PRIMARY_ACTION_CHUNK_DISTANCE_METRIC:
            raise M43ContractError("distance result uses the wrong primary metric")
        if (self.chunk_length, self.action_dimension) != (
            M43_ACTION_CHUNK_SIZE,
            M43_ACTION_COMPONENTS,
        ):
            raise M43ContractError("distance result must describe one 50x8 chunk")
        for name in (
            "raw_l2",
            "action_range_normalized_l2",
            "train_std_normalized_l2",
            "arm_only_raw_l2",
            "arm_only_action_range_normalized_l2",
            "arm_only_train_std_normalized_l2",
            "gripper_only_raw_l2",
            "gripper_only_action_range_normalized_l2",
            "gripper_only_train_std_normalized_l2",
        ):
            object.__setattr__(
                self,
                name,
                _validate_distance_mapping(cast(Mapping[str, object], getattr(self, name)), name),
            )
        for name in (
            "cosine_distance",
            "arm_only_cosine_distance",
            "gripper_only_cosine_distance",
        ):
            object.__setattr__(
                self,
                name,
                _validate_distance_mapping(
                    cast(Mapping[str, object], getattr(self, name)), name, allow_none=True
                ),
            )
        _finite(self.primary_distance, "primary_distance")
        if self.primary_distance < 0.0:
            raise M43ContractError("primary_distance must be non-negative")
        expected = self.arm_only_action_range_normalized_l2["execution_window"]
        if not math.isclose(self.primary_distance, cast(float, expected), abs_tol=0.0):
            raise M43ContractError("primary_distance is inconsistent with its declared metric")


@dataclass(frozen=True, slots=True)
class TaskRetrievalConfusion(_JsonRecord):
    """Rectangular requested/predicted categorical confusion matrix."""

    requested_labels: tuple[str, ...]
    predicted_labels: tuple[str, ...]
    counts: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.requested_labels or not self.predicted_labels:
            raise M43ContractError("confusion labels must not be empty")
        if len(set(self.requested_labels)) != len(self.requested_labels) or len(
            set(self.predicted_labels)
        ) != len(self.predicted_labels):
            raise M43ContractError("confusion labels must be unique")
        if len(self.counts) != len(self.requested_labels):
            raise M43ContractError("confusion row count does not match requested labels")
        for row in self.counts:
            if len(row) != len(self.predicted_labels):
                raise M43ContractError("confusion column count does not match predicted labels")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in row
            ):
                raise M43ContractError("confusion counts must be non-negative integers")

    @property
    def total(self) -> int:
        return sum(sum(row) for row in self.counts)


@dataclass(frozen=True, slots=True)
class SemanticRetrievalResult(_JsonRecord):
    """Six-way nearest-PerTask result for one fixed policy observation."""

    observation_id: str
    requested_task_id: str
    nearest_task_id: str
    ranked_task_ids: tuple[str, ...]
    primary_distances: Mapping[str, object]
    correct_rank: int
    top1_correct: bool
    top2_correct: bool
    reciprocal_rank: float
    correct_margin: float

    def __post_init__(self) -> None:
        _nonempty(self.observation_id, "observation_id")
        _validate_task_id(self.requested_task_id, "requested_task_id")
        _validate_task_id(self.nearest_task_id, "nearest_task_id")
        if self.ranked_task_ids != tuple(dict.fromkeys(self.ranked_task_ids)) or set(
            self.ranked_task_ids
        ) != set(M43_CANONICAL_TASK_IDS):
            raise M43ContractError("ranked_task_ids must be a permutation of all six tasks")
        if self.nearest_task_id != self.ranked_task_ids[0]:
            raise M43ContractError("nearest_task_id must equal the first ranked task")
        distances = _frozen_mapping(self.primary_distances, "primary_distances")
        if set(distances) != set(M43_CANONICAL_TASK_IDS):
            raise M43ContractError("primary_distances must contain all six task IDs")
        for task_id, value in distances.items():
            _finite(cast(float, value), f"primary_distances[{task_id}]")
            if cast(float, value) < 0.0:
                raise M43ContractError("primary distances must be non-negative")
        object.__setattr__(self, "primary_distances", distances)
        if not 1 <= self.correct_rank <= len(M43_CANONICAL_TASK_IDS):
            raise M43ContractError("correct_rank must be in [1, 6]")
        if self.top1_correct != (self.correct_rank == 1):
            raise M43ContractError("top1_correct is inconsistent with correct_rank")
        if self.top2_correct != (self.correct_rank <= 2):
            raise M43ContractError("top2_correct is inconsistent with correct_rank")
        if not math.isclose(self.reciprocal_rank, 1.0 / self.correct_rank):
            raise M43ContractError("reciprocal_rank is inconsistent with correct_rank")
        _finite(self.correct_margin, "correct_margin")


@dataclass(frozen=True, slots=True)
class ObjectRetrievalResult(_JsonRecord):
    observation_id: str
    requested_object_id: str
    predicted_object_id: str
    ranked_object_ids: tuple[str, ...]
    primary_distances: Mapping[str, object]
    correct: bool
    correct_margin: float

    def __post_init__(self) -> None:
        _nonempty(self.observation_id, "observation_id")
        if self.requested_object_id not in OBJECT_IDS or self.predicted_object_id not in OBJECT_IDS:
            raise M43ContractError("object retrieval IDs must use M1 semantic object IDs")
        if set(self.ranked_object_ids) != set(OBJECT_IDS) or len(self.ranked_object_ids) != len(
            OBJECT_IDS
        ):
            raise M43ContractError(
                "ranked_object_ids must contain red, green, and blue exactly once"
            )
        if self.predicted_object_id != self.ranked_object_ids[0]:
            raise M43ContractError("predicted_object_id must equal the first ranked object")
        distances = _frozen_mapping(self.primary_distances, "primary_distances")
        if set(distances) != set(OBJECT_IDS):
            raise M43ContractError("object primary distances must contain all three objects")
        object.__setattr__(self, "primary_distances", distances)
        for object_id, value in distances.items():
            _finite(cast(float, value), f"primary_distances[{object_id}]")
        if self.correct != (self.requested_object_id == self.predicted_object_id):
            raise M43ContractError("object retrieval correctness is inconsistent")
        _finite(self.correct_margin, "correct_margin")


@dataclass(frozen=True, slots=True)
class BinRetrievalResult(_JsonRecord):
    observation_id: str
    requested_bin_id: str
    predicted_bin_id: str
    ranked_bin_ids: tuple[str, ...]
    primary_distances: Mapping[str, object]
    correct: bool
    correct_margin: float
    conditional_predicted_bin_id: str
    conditional_primary_distances: Mapping[str, object]
    conditional_correct: bool
    conditional_correct_margin: float

    def __post_init__(self) -> None:
        _nonempty(self.observation_id, "observation_id")
        for value in (
            self.requested_bin_id,
            self.predicted_bin_id,
            self.conditional_predicted_bin_id,
        ):
            if value not in BIN_IDS:
                raise M43ContractError("bin retrieval IDs must use M1 semantic bin IDs")
        if set(self.ranked_bin_ids) != set(BIN_IDS) or len(self.ranked_bin_ids) != len(BIN_IDS):
            raise M43ContractError("ranked_bin_ids must contain left and right exactly once")
        if self.predicted_bin_id != self.ranked_bin_ids[0]:
            raise M43ContractError("predicted_bin_id must equal the first ranked bin")
        for name in ("primary_distances", "conditional_primary_distances"):
            distances = _frozen_mapping(cast(Mapping[str, object], getattr(self, name)), name)
            if set(distances) != set(BIN_IDS):
                raise M43ContractError(f"{name} must contain both bins")
            for bin_id, value in distances.items():
                _finite(cast(float, value), f"{name}[{bin_id}]")
            object.__setattr__(self, name, distances)
        if self.correct != (self.requested_bin_id == self.predicted_bin_id):
            raise M43ContractError("bin retrieval correctness is inconsistent")
        if self.conditional_correct != (self.requested_bin_id == self.conditional_predicted_bin_id):
            raise M43ContractError("conditional bin retrieval correctness is inconsistent")
        _finite(self.correct_margin, "correct_margin")
        _finite(self.conditional_correct_margin, "conditional_correct_margin")


@dataclass(frozen=True, slots=True)
class FirstInteractionRecord(_JsonRecord):
    """Compact semantic interaction record; no image or trajectory payloads."""

    observation_id: str
    requested_task_id: str
    requested_target_object_id: str
    first_object_grasped: str | None
    first_object_displaced: str | None
    displacement_threshold_m: float
    requested_destination_bin_id: str
    first_bin_approached: str | None
    first_bin_entered_by_any_object: str | None
    final_object_bin_relationship: Mapping[str, object]
    nearest_per_task_task_id: str
    first_grasp_matches_nearest_per_task: bool | None
    initial_semantic_error_visible: bool
    failure_class: SemanticFailureClass

    def __post_init__(self) -> None:
        _nonempty(self.observation_id, "observation_id")
        _validate_task_id(self.requested_task_id, "requested_task_id")
        _validate_task_id(self.nearest_per_task_task_id, "nearest_per_task_task_id")
        if self.requested_target_object_id not in OBJECT_IDS:
            raise M43ContractError("requested_target_object_id must be canonical")
        for name in ("first_object_grasped", "first_object_displaced"):
            value = cast(str | None, getattr(self, name))
            if value is not None and value not in OBJECT_IDS:
                raise M43ContractError(f"{name} must be a canonical object ID or null")
        if self.requested_destination_bin_id not in BIN_IDS:
            raise M43ContractError("requested_destination_bin_id must be canonical")
        for name in ("first_bin_approached", "first_bin_entered_by_any_object"):
            value = cast(str | None, getattr(self, name))
            if value is not None and value not in BIN_IDS:
                raise M43ContractError(f"{name} must be a canonical bin ID or null")
        _finite(self.displacement_threshold_m, "displacement_threshold_m", strictly_positive=True)
        relationship = _frozen_mapping(
            self.final_object_bin_relationship, "final_object_bin_relationship"
        )
        if set(relationship) != set(OBJECT_IDS) or any(
            value is not None and value not in BIN_IDS for value in relationship.values()
        ):
            raise M43ContractError(
                "final_object_bin_relationship must map every object to a bin or null"
            )
        object.__setattr__(self, "final_object_bin_relationship", relationship)
        object.__setattr__(self, "failure_class", SemanticFailureClass(self.failure_class))
        if (
            self.first_object_grasped is None
            and self.first_grasp_matches_nearest_per_task is not None
        ):
            raise M43ContractError("a missing first grasp requires a null nearest-match result")


@dataclass(frozen=True, slots=True)
class SemanticAlignmentAudit(_JsonRecord):
    """Complete compact result for one frozen candidate policy."""

    policy_label: str
    observation_count: int
    distance_config: ActionChunkDistanceConfig
    task_retrieval: tuple[SemanticRetrievalResult, ...]
    object_retrieval: tuple[ObjectRetrievalResult, ...]
    bin_retrieval: tuple[BinRetrievalResult, ...]
    task_confusion: TaskRetrievalConfusion
    object_confusion: TaskRetrievalConfusion
    bin_confusion: TaskRetrievalConfusion
    first_interactions: tuple[FirstInteractionRecord, ...]
    first_interaction_confusions: Mapping[str, object]
    failure_distribution: Mapping[str, object]
    conclusions: tuple[SemanticAuditConclusion, ...]
    complete_distance_results: Mapping[str, object] = field(default_factory=dict)
    raw_policy_action_metrics: Mapping[str, object] = field(default_factory=dict)
    runtime_action_metrics: Mapping[str, object] = field(default_factory=dict)
    deterministic_reload_validated: bool = False
    deterministic_reset_validated: bool = False
    schema_version: str = M43_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _nonempty(self.policy_label, "policy_label")
        if self.schema_version != M43_SCHEMA_VERSION:
            raise M43ContractError("unknown M4.3 semantic audit schema")
        if isinstance(self.observation_count, bool) or not isinstance(self.observation_count, int):
            raise TypeError("observation_count must be an integer")
        if self.observation_count < 1:
            raise M43ContractError("semantic audit requires at least one observation")
        if any(
            len(values) != self.observation_count
            for values in (self.task_retrieval, self.object_retrieval, self.bin_retrieval)
        ):
            raise M43ContractError("each offline retrieval must cover every audit observation")
        if len(set(self.conclusions)) != len(self.conclusions):
            raise M43ContractError("audit conclusions must be unique")
        object.__setattr__(
            self,
            "conclusions",
            tuple(SemanticAuditConclusion(value) for value in self.conclusions),
        )
        distribution = _frozen_mapping(self.failure_distribution, "failure_distribution")
        if set(distribution) != {value.value for value in SemanticFailureClass}:
            raise M43ContractError("failure_distribution must contain all twelve failure classes")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in distribution.values()
        ):
            raise M43ContractError("failure distribution counts must be non-negative integers")
        if sum(cast(int, value) for value in distribution.values()) != len(self.first_interactions):
            raise M43ContractError("failure distribution must count every interaction record once")
        object.__setattr__(self, "failure_distribution", distribution)
        for name in (
            "complete_distance_results",
            "first_interaction_confusions",
            "raw_policy_action_metrics",
            "runtime_action_metrics",
        ):
            object.__setattr__(
                self,
                name,
                _frozen_mapping(cast(Mapping[str, object], getattr(self, name)), name),
            )


__all__ = [
    "ACTION_CHUNK_DISTANCE_VERSION",
    "ACTION_CHUNK_WINDOW_KEYS",
    "M43_ACTION_CHUNK_SIZE",
    "M43_ACTION_COMPONENTS",
    "M43_ARM_ACTION_INDICES",
    "M43_CANONICAL_TASK_IDS",
    "M43_GRIPPER_ACTION_INDICES",
    "M43_SCHEMA_VERSION",
    "PRIMARY_ACTION_CHUNK_DISTANCE_METRIC",
    "ActionChunkDistanceConfig",
    "ActionChunkDistanceResult",
    "BinRetrievalResult",
    "FirstInteractionRecord",
    "M43ContractError",
    "ObjectRetrievalResult",
    "SemanticAlignmentAudit",
    "SemanticAuditConclusion",
    "SemanticFailureClass",
    "SemanticRetrievalResult",
    "TaskRetrievalConfusion",
]
