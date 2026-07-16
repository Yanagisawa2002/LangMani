"""Pure target-development contracts for the M4.3b FactorFiLM policy.

This module contains no simulator, Torch, LeRobot, or filesystem integration.
It converts already collected validation/development evidence into immutable
JSON records, derives first-interaction evidence from numeric step snapshots,
and evaluates the locked M4.3b development quality gate without rounding.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Self, cast

from langmani.environments.pick_place_by_instruction import (
    BIN_INTERIOR_HALF_SIZE,
    CUBE_HALF_SIZE,
)
from langmani.environments.specs import BIN_IDS, OBJECT_IDS, stable_scene_id
from langmani.policies.act_factor_film_types import (
    FactorFiLMContractError,
    FactorFiLMSelectionRecord,
    FactorFiLMValidationResult,
    canonical_fingerprint,
)
from langmani.policies.act_semantic_audit import (
    SemanticRolloutTrace,
    build_first_interaction_record,
    first_interaction_confusions,
)
from langmani.policies.m42_schedule import (
    M42_DEV_SCHEDULE_FINGERPRINT,
    M42_DEV_SCHEDULE_ID,
)
from langmani.policies.m43_types import (
    M43_CANONICAL_TASK_IDS,
    PRIMARY_ACTION_CHUNK_DISTANCE_METRIC,
    FirstInteractionRecord,
    SemanticFailureClass,
)

FACTOR_FILM_EVALUATION_SCHEMA_VERSION = "langmani-m43b-factor-film-evaluation-v0"
FACTOR_FILM_RELOAD_SCHEMA_VERSION = "langmani-m43b-factor-film-reload-v0"
FACTOR_FILM_FIRST_INTERACTION_SCHEMA_VERSION = "langmani-m43b-factor-film-first-interaction-v0"
FACTOR_FILM_PAIRED_IDENTITIES_SCHEMA_VERSION = "langmani-m43b-factor-film-paired-identities-v0"
FACTOR_FILM_DEVELOPMENT_GATE_SCHEMA_VERSION = "langmani-m43b-factor-film-development-gate-v0"

FACTOR_FILM_DEVELOPMENT_EPISODE_COUNT = 72
FACTOR_FILM_DEVELOPMENT_EPISODES_PER_TASK = 12
FACTOR_FILM_VALIDATION_EPISODE_COUNT = 36
FACTOR_FILM_RELOAD_ATOL = 1e-6
FACTOR_FILM_RELOAD_RTOL = 1e-6
FACTOR_FILM_DISPLACEMENT_THRESHOLD_METERS = 0.01
FACTOR_FILM_LIFT_DELTA_METERS = 0.04
FACTOR_FILM_DESTINATION_NEAR_XY_METERS = BIN_INTERIOR_HALF_SIZE + CUBE_HALF_SIZE
FACTOR_FILM_DESCENT_HEIGHT_METERS = CUBE_HALF_SIZE + 0.035

FACTOR_FILM_DEVELOPMENT_POLICY_LABELS = (
    "ACT-PerTask",
    "ACT-Mixed-TaskOneHot",
    "ACT-Mixed-FactorFiLM",
)

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class FactorFiLMEvaluationError(FactorFiLMContractError):
    """Raised when M4.3b target-development evidence is inconsistent."""


class _FrozenMapping(Mapping[str, object]):
    __slots__ = ("_items",)

    def __init__(self, value: Mapping[str, object]) -> None:
        if not all(isinstance(key, str) for key in value):
            raise TypeError("FactorFiLM evaluation mappings require string keys")
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
    raise TypeError(f"unsupported FactorFiLM evaluation JSON value {type(value).__name__}")


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
        raise FactorFiLMEvaluationError(f"{name} must be one prefixed lowercase SHA-256 digest")


def _integer(
    value: int,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise FactorFiLMEvaluationError(f"{name} is outside its declared bound")


def _finite(value: float, name: str, *, minimum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    if not math.isfinite(float(value)) or (minimum is not None and float(value) < minimum):
        raise FactorFiLMEvaluationError(f"{name} must be finite and within its declared bound")


def _rate(value: float, name: str) -> None:
    _finite(value, name, minimum=0.0)
    if float(value) > 1.0:
        raise FactorFiLMEvaluationError(f"{name} must lie in [0,1]")


def _vector(value: Sequence[object], *, length: int, name: str) -> tuple[float, ...]:
    if isinstance(value, str | bytes) or len(value) != length:
        raise FactorFiLMEvaluationError(f"{name} must contain exactly {length} components")
    result = tuple(float(item) for item in value)
    for index, item in enumerate(result):
        _finite(item, f"{name}[{index}]")
    return result


def _matrix(
    value: Sequence[object],
    *,
    rows: int,
    columns: int,
    name: str,
) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, str | bytes) or len(value) != rows:
        raise FactorFiLMEvaluationError(f"{name} must contain exactly {rows} rows")
    return tuple(
        _vector(cast(Sequence[object], row), length=columns, name=f"{name}[{index}]")
        for index, row in enumerate(value)
    )


def _bool_matrix(
    value: Sequence[object],
    *,
    rows: int,
    columns: int,
    name: str,
) -> tuple[tuple[bool, ...], ...]:
    if isinstance(value, str | bytes) or len(value) != rows:
        raise FactorFiLMEvaluationError(f"{name} must contain exactly {rows} rows")
    result: list[tuple[bool, ...]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, Sequence) or isinstance(row, str | bytes) or len(row) != columns:
            raise FactorFiLMEvaluationError(
                f"{name}[{row_index}] must contain exactly {columns} booleans"
            )
        if not all(isinstance(item, bool) for item in row):
            raise TypeError(f"{name}[{row_index}] must contain booleans")
        result.append(tuple(cast(Sequence[bool], row)))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class FactorFiLMValidationCountResult(_JsonRecord):
    """One complete 36-episode validation result before rate conversion."""

    checkpoint_fingerprint: str
    checkpoint_step: int
    schedule_digest: str
    success_count: int
    wrong_object_interaction_count: int
    wrong_object_in_target_bin_count: int
    target_off_table_count: int
    timeout_count: int
    offline_validation_action_loss: float
    episode_count: int = FACTOR_FILM_VALIDATION_EPISODE_COUNT
    source_split: str = "m3b_validation"
    development_schedule_accessed: bool = False
    test_split_accessed: bool = False
    fresh_seed_accessed: bool = False
    semantic_audit_accessed_for_selection: bool = False
    final_schedule_accessed: bool = False
    schema_version: str = FACTOR_FILM_EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _digest(self.checkpoint_fingerprint, "checkpoint_fingerprint")
        _digest(self.schedule_digest, "schedule_digest")
        _integer(self.checkpoint_step, "checkpoint_step", minimum=5_000, maximum=100_000)
        if self.checkpoint_step % 5_000:
            raise FactorFiLMEvaluationError("checkpoint_step must follow the 5k schedule")
        if self.episode_count != FACTOR_FILM_VALIDATION_EPISODE_COUNT:
            raise FactorFiLMEvaluationError("FactorFiLM selection requires all 36 episodes")
        for name in (
            "success_count",
            "wrong_object_interaction_count",
            "wrong_object_in_target_bin_count",
            "target_off_table_count",
            "timeout_count",
        ):
            _integer(
                cast(int, getattr(self, name)),
                name,
                maximum=self.episode_count,
            )
        _finite(
            self.offline_validation_action_loss,
            "offline_validation_action_loss",
            minimum=0.0,
        )
        if self.source_split != "m3b_validation":
            raise FactorFiLMEvaluationError("FactorFiLM checkpoint selection is validation-only")
        if any(
            (
                self.development_schedule_accessed,
                self.test_split_accessed,
                self.fresh_seed_accessed,
                self.semantic_audit_accessed_for_selection,
                self.final_schedule_accessed,
            )
        ):
            raise FactorFiLMEvaluationError("validation result accessed a forbidden source")
        if self.schema_version != FACTOR_FILM_EVALUATION_SCHEMA_VERSION:
            raise FactorFiLMEvaluationError("unknown FactorFiLM evaluation schema")

    @property
    def ranking_key(self) -> tuple[int, int, int, int, int, float, int]:
        """The locked seven-level count ranking, with earlier step last."""
        return (
            -self.success_count,
            self.wrong_object_interaction_count,
            self.wrong_object_in_target_bin_count,
            self.target_off_table_count,
            self.timeout_count,
            float(self.offline_validation_action_loss),
            self.checkpoint_step,
        )

    def to_selection_result(self) -> FactorFiLMValidationResult:
        """Convert counts to the pre-existing selection rate contract."""
        denominator = float(self.episode_count)
        return FactorFiLMValidationResult(
            checkpoint_fingerprint=self.checkpoint_fingerprint,
            checkpoint_step=self.checkpoint_step,
            source_split=self.source_split,
            schedule_digest=self.schedule_digest,
            task_success_rate=self.success_count / denominator,
            wrong_object_interaction_rate=self.wrong_object_interaction_count / denominator,
            wrong_object_in_target_bin_rate=(self.wrong_object_in_target_bin_count / denominator),
            target_off_table_rate=self.target_off_table_count / denominator,
            timeout_rate=self.timeout_count / denominator,
            offline_validation_action_loss=self.offline_validation_action_loss,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        return cls(**dict(value))


def factor_film_validation_count_result(
    *,
    checkpoint_fingerprint: str,
    checkpoint_step: int,
    schedule_digest: str,
    aggregate: Mapping[str, object],
    offline_validation_action_loss: float,
) -> FactorFiLMValidationCountResult:
    """Project one closed-loop benchmark aggregate into validation counts."""
    required = {
        "episode_count",
        "successes",
        "wrong_object_interaction_count",
        "wrong_object_in_target_bin_count",
        "target_off_table_count",
        "timeout_count",
    }
    missing_fields = required - set(aggregate)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise FactorFiLMEvaluationError(f"validation aggregate is missing {missing}")
    return FactorFiLMValidationCountResult(
        checkpoint_fingerprint=checkpoint_fingerprint,
        checkpoint_step=checkpoint_step,
        schedule_digest=schedule_digest,
        episode_count=cast(int, aggregate["episode_count"]),
        success_count=cast(int, aggregate["successes"]),
        wrong_object_interaction_count=cast(int, aggregate["wrong_object_interaction_count"]),
        wrong_object_in_target_bin_count=cast(int, aggregate["wrong_object_in_target_bin_count"]),
        target_off_table_count=cast(int, aggregate["target_off_table_count"]),
        timeout_count=cast(int, aggregate["timeout_count"]),
        offline_validation_action_loss=offline_validation_action_loss,
    )


def rank_factor_film_validation_count_results(
    candidates: Sequence[FactorFiLMValidationCountResult],
) -> tuple[FactorFiLMValidationCountResult, ...]:
    """Rank exactly 20 validation candidates and verify rate-ranking parity."""
    values = tuple(candidates)
    if len(values) != 20:
        raise FactorFiLMEvaluationError("FactorFiLM selection requires all 20 checkpoints")
    if not all(isinstance(value, FactorFiLMValidationCountResult) for value in values):
        raise TypeError("validation candidates must be FactorFiLMValidationCountResult values")
    if (
        len({value.checkpoint_fingerprint for value in values}) != 20
        or len({value.checkpoint_step for value in values}) != 20
    ):
        raise FactorFiLMEvaluationError("validation candidates must be unique")
    if len({value.schedule_digest for value in values}) != 1:
        raise FactorFiLMEvaluationError("validation candidates use different schedules")
    ranked = tuple(sorted(values, key=lambda value: value.ranking_key))
    ranked_rates = tuple(
        sorted(
            (value.to_selection_result() for value in values),
            key=lambda value: value.ranking_key,
        )
    )
    if tuple(value.checkpoint_fingerprint for value in ranked) != tuple(
        value.checkpoint_fingerprint for value in ranked_rates
    ):
        raise FactorFiLMEvaluationError("count and rate validation rankings disagree")
    return ranked


@dataclass(frozen=True, slots=True)
class FactorFiLMReloadValidationRecord(_JsonRecord):
    """Fresh-instance selected-checkpoint reload and deterministic inference proof."""

    run_fingerprint: str
    selection_fingerprint: str
    selected_checkpoint_fingerprint: str
    selected_checkpoint_step: int
    architecture_fingerprint: str
    preprocessor_fingerprint: str
    policy_postprocessor_fingerprint: str
    action_runtime_fingerprint: str
    validation_observation_fingerprint: str
    reference_output_fingerprint: str
    reloaded_output_fingerprint: str
    maximum_absolute_error: float
    maximum_relative_error: float
    fresh_policy_instance: bool
    preprocessor_reloaded: bool
    policy_postprocessor_reloaded: bool
    architecture_reconstructed: bool
    task_mappings_reconstructed: bool
    action_runtime_reconstructed: bool
    fingerprints_validated: bool
    deterministic_inference_validated: bool
    reload_validated: bool
    absolute_tolerance: float = FACTOR_FILM_RELOAD_ATOL
    relative_tolerance: float = FACTOR_FILM_RELOAD_RTOL
    development_schedule_accessed: bool = False
    test_split_accessed: bool = False
    fresh_seed_accessed: bool = False
    final_schedule_accessed: bool = False
    schema_version: str = FACTOR_FILM_RELOAD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "run_fingerprint",
            "selection_fingerprint",
            "selected_checkpoint_fingerprint",
            "architecture_fingerprint",
            "preprocessor_fingerprint",
            "policy_postprocessor_fingerprint",
            "action_runtime_fingerprint",
            "validation_observation_fingerprint",
            "reference_output_fingerprint",
            "reloaded_output_fingerprint",
        ):
            _digest(cast(str, getattr(self, name)), name)
        _integer(
            self.selected_checkpoint_step,
            "selected_checkpoint_step",
            minimum=5_000,
            maximum=100_000,
        )
        if self.selected_checkpoint_step % 5_000:
            raise FactorFiLMEvaluationError("selected checkpoint step must follow 5k schedule")
        for name in (
            "maximum_absolute_error",
            "maximum_relative_error",
            "absolute_tolerance",
            "relative_tolerance",
        ):
            _finite(cast(float, getattr(self, name)), name, minimum=0.0)
        if (
            self.absolute_tolerance != FACTOR_FILM_RELOAD_ATOL
            or self.relative_tolerance != FACTOR_FILM_RELOAD_RTOL
        ):
            raise FactorFiLMEvaluationError("FactorFiLM reload tolerances are immutable")
        required = (
            self.fresh_policy_instance,
            self.preprocessor_reloaded,
            self.policy_postprocessor_reloaded,
            self.architecture_reconstructed,
            self.task_mappings_reconstructed,
            self.action_runtime_reconstructed,
            self.fingerprints_validated,
            self.deterministic_inference_validated,
        )
        if self.reload_validated != all(required):
            raise FactorFiLMEvaluationError("reload_validated disagrees with required checks")
        if any(
            (
                self.development_schedule_accessed,
                self.test_split_accessed,
                self.fresh_seed_accessed,
                self.final_schedule_accessed,
            )
        ):
            raise FactorFiLMEvaluationError("reload validation accessed a forbidden source")
        if self.schema_version != FACTOR_FILM_RELOAD_SCHEMA_VERSION:
            raise FactorFiLMEvaluationError("unknown FactorFiLM reload schema")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        return cls(**dict(value))


def validate_reload_against_selection(
    record: FactorFiLMReloadValidationRecord,
    selection: FactorFiLMSelectionRecord,
) -> None:
    """Bind fresh reload evidence to the immutable validation selection."""
    if not isinstance(record, FactorFiLMReloadValidationRecord) or not isinstance(
        selection, FactorFiLMSelectionRecord
    ):
        raise TypeError("reload validation requires typed FactorFiLM records")
    if (
        record.run_fingerprint != selection.run_fingerprint
        or record.selection_fingerprint != canonical_fingerprint(selection.to_dict())
        or record.selected_checkpoint_fingerprint != selection.selected_checkpoint_fingerprint
        or record.selected_checkpoint_step != selection.selected_checkpoint_step
    ):
        raise FactorFiLMEvaluationError("reload evidence differs from checkpoint selection")
    if not record.reload_validated:
        raise FactorFiLMEvaluationError("selected checkpoint reload did not validate")


@dataclass(frozen=True, slots=True)
class FactorFiLMSemanticStepSnapshot(_JsonRecord):
    """One numeric, policy-independent simulator snapshot for pure analysis."""

    step: int
    cube_positions: tuple[tuple[float, ...], ...]
    cube_orientations: tuple[tuple[float, ...], ...]
    cube_linear_velocities: tuple[tuple[float, ...], ...]
    cube_angular_velocities: tuple[tuple[float, ...], ...]
    bin_floor_centers: tuple[tuple[float, ...], ...]
    tcp_position: tuple[float, ...]
    object_is_grasped: tuple[bool, ...]
    object_in_bin: tuple[tuple[bool, ...], ...]
    executed_action: tuple[float, ...] | None
    evaluation: Mapping[str, object]

    def __post_init__(self) -> None:
        _integer(self.step, "step")
        object.__setattr__(
            self,
            "cube_positions",
            _matrix(self.cube_positions, rows=3, columns=3, name="cube_positions"),
        )
        orientations = _matrix(self.cube_orientations, rows=3, columns=4, name="cube_orientations")
        if any(
            math.isclose(math.sqrt(sum(item * item for item in row)), 0.0) for row in orientations
        ):
            raise FactorFiLMEvaluationError("cube orientations must contain nonzero quaternions")
        object.__setattr__(self, "cube_orientations", orientations)
        object.__setattr__(
            self,
            "cube_linear_velocities",
            _matrix(
                self.cube_linear_velocities,
                rows=3,
                columns=3,
                name="cube_linear_velocities",
            ),
        )
        object.__setattr__(
            self,
            "cube_angular_velocities",
            _matrix(
                self.cube_angular_velocities,
                rows=3,
                columns=3,
                name="cube_angular_velocities",
            ),
        )
        object.__setattr__(
            self,
            "bin_floor_centers",
            _matrix(self.bin_floor_centers, rows=2, columns=3, name="bin_floor_centers"),
        )
        object.__setattr__(
            self,
            "tcp_position",
            _vector(self.tcp_position, length=3, name="tcp_position"),
        )
        if len(self.object_is_grasped) != 3 or not all(
            isinstance(item, bool) for item in self.object_is_grasped
        ):
            raise FactorFiLMEvaluationError("object_is_grasped must contain three booleans")
        object.__setattr__(self, "object_is_grasped", tuple(self.object_is_grasped))
        in_bin = _bool_matrix(self.object_in_bin, rows=3, columns=2, name="object_in_bin")
        if any(sum(row) > 1 for row in in_bin):
            raise FactorFiLMEvaluationError("one object cannot occupy both bins")
        object.__setattr__(self, "object_in_bin", in_bin)
        if self.executed_action is not None:
            object.__setattr__(
                self,
                "executed_action",
                _vector(self.executed_action, length=8, name="executed_action"),
            )
        if not isinstance(self.evaluation, Mapping) or not all(
            isinstance(key, str) and isinstance(value, bool)
            for key, value in self.evaluation.items()
        ):
            raise TypeError("evaluation must map string fields to booleans")
        object.__setattr__(self, "evaluation", _FrozenMapping(self.evaluation))

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = dict(value)
        for name in (
            "cube_positions",
            "cube_orientations",
            "cube_linear_velocities",
            "cube_angular_velocities",
            "bin_floor_centers",
            "object_in_bin",
        ):
            nested = payload.get(name)
            if not isinstance(nested, list | tuple):
                raise TypeError(f"{name} must be an array")
            payload[name] = tuple(tuple(cast(Sequence[object], row)) for row in nested)
        for name in ("tcp_position", "object_is_grasped", "executed_action"):
            item = payload.get(name)
            if item is not None:
                if not isinstance(item, list | tuple):
                    raise TypeError(f"{name} must be an array or null")
                payload[name] = tuple(item)
        return cls(**payload)


class SemanticErrorStage(StrEnum):
    NONE = "none"
    INITIAL_CHUNK = "initial_chunk"
    EMERGED_AFTER_INITIAL_CHUNK = "emerged_after_initial_chunk"
    NO_FIRST_INTERACTION = "no_first_interaction"


@dataclass(frozen=True, slots=True)
class FirstInteractionSnapshot(_JsonRecord):
    """Episode-level first-interaction and post-grasp evidence derived from steps."""

    observation_id: str
    requested_task_id: str
    requested_target_object_id: str
    requested_destination_bin_id: str
    nearest_per_task_task_id: str
    first_object_grasped: str | None
    first_object_displaced: str | None
    first_bin_approached: str | None
    first_bin_entered_by_any_object: str | None
    final_object_bin_relationship: Mapping[str, object]
    first_target_grasp_step: int | None
    first_release_command_step: int | None
    maximum_target_height: float
    closest_target_to_bin_distance: float
    final_target_pose: tuple[float, ...]
    final_target_velocity: tuple[float, ...]
    final_environment_evaluation: Mapping[str, object]
    failure_class: SemanticFailureClass
    semantic_error_stage: SemanticErrorStage
    displacement_threshold_m: float = FACTOR_FILM_DISPLACEMENT_THRESHOLD_METERS
    schema_version: str = FACTOR_FILM_FIRST_INTERACTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.observation_id, str) or not self.observation_id:
            raise FactorFiLMEvaluationError("observation_id must be non-empty")
        if self.requested_task_id not in M43_CANONICAL_TASK_IDS or (
            self.nearest_per_task_task_id not in M43_CANONICAL_TASK_IDS
        ):
            raise FactorFiLMEvaluationError("first-interaction task IDs must be canonical")
        if self.requested_target_object_id not in OBJECT_IDS:
            raise FactorFiLMEvaluationError("requested target object must be canonical")
        if self.requested_destination_bin_id not in BIN_IDS:
            raise FactorFiLMEvaluationError("requested destination bin must be canonical")
        if self.requested_target_object_id != _task_object_id(self.requested_task_id) or (
            self.requested_destination_bin_id != _task_bin_id(self.requested_task_id)
        ):
            raise FactorFiLMEvaluationError(
                "requested object/bin do not match the stable task identity"
            )
        for name in ("first_object_grasped", "first_object_displaced"):
            value = cast(str | None, getattr(self, name))
            if value is not None and value not in OBJECT_IDS:
                raise FactorFiLMEvaluationError(f"{name} must be canonical or null")
        for name in ("first_bin_approached", "first_bin_entered_by_any_object"):
            value = cast(str | None, getattr(self, name))
            if value is not None and value not in BIN_IDS:
                raise FactorFiLMEvaluationError(f"{name} must be canonical or null")
        relationship = dict(self.final_object_bin_relationship)
        if set(relationship) != set(OBJECT_IDS) or any(
            value is not None and value not in BIN_IDS for value in relationship.values()
        ):
            raise FactorFiLMEvaluationError(
                "final relationship must map every object to one bin or null"
            )
        object.__setattr__(self, "final_object_bin_relationship", _FrozenMapping(relationship))
        for name in ("first_target_grasp_step", "first_release_command_step"):
            value = cast(int | None, getattr(self, name))
            if value is not None:
                _integer(value, name)
        if self.first_release_command_step is not None and self.first_target_grasp_step is None:
            raise FactorFiLMEvaluationError("release evidence requires a target grasp")
        _finite(self.maximum_target_height, "maximum_target_height")
        _finite(
            self.closest_target_to_bin_distance,
            "closest_target_to_bin_distance",
            minimum=0.0,
        )
        object.__setattr__(
            self,
            "final_target_pose",
            _vector(self.final_target_pose, length=7, name="final_target_pose"),
        )
        object.__setattr__(
            self,
            "final_target_velocity",
            _vector(self.final_target_velocity, length=6, name="final_target_velocity"),
        )
        if not isinstance(self.final_environment_evaluation, Mapping) or not all(
            isinstance(key, str) and isinstance(value, bool)
            for key, value in self.final_environment_evaluation.items()
        ):
            raise TypeError("final_environment_evaluation must map strings to booleans")
        object.__setattr__(
            self,
            "final_environment_evaluation",
            _FrozenMapping(self.final_environment_evaluation),
        )
        object.__setattr__(self, "failure_class", SemanticFailureClass(self.failure_class))
        object.__setattr__(
            self, "semantic_error_stage", SemanticErrorStage(self.semantic_error_stage)
        )
        expected_stage = _semantic_error_stage_from_fields(
            initial_semantic_error=(self.nearest_per_task_task_id != self.requested_task_id),
            requested_object=self.requested_target_object_id,
            requested_bin=self.requested_destination_bin_id,
            first_object_grasped=self.first_object_grasped,
            first_object_displaced=self.first_object_displaced,
            first_bin_approached=self.first_bin_approached,
            first_bin_entered=self.first_bin_entered_by_any_object,
            final_relationship=self.final_object_bin_relationship,
        )
        if self.semantic_error_stage is not expected_stage:
            raise FactorFiLMEvaluationError("semantic error stage is inconsistent")
        if (self.failure_class is SemanticFailureClass.SUCCESS) != bool(
            self.final_environment_evaluation.get("success", False)
        ):
            raise FactorFiLMEvaluationError(
                "success failure class disagrees with final environment evaluation"
            )
        if self.first_object_grasped is None and self.first_target_grasp_step is not None:
            raise FactorFiLMEvaluationError("target-grasp step requires a first grasped object")
        if (
            self.first_object_grasped == self.requested_target_object_id
            and self.first_target_grasp_step is None
        ):
            raise FactorFiLMEvaluationError("target first grasp requires its simulator step")
        if self.displacement_threshold_m != FACTOR_FILM_DISPLACEMENT_THRESHOLD_METERS:
            raise FactorFiLMEvaluationError("first-interaction displacement threshold changed")
        if self.schema_version != FACTOR_FILM_FIRST_INTERACTION_SCHEMA_VERSION:
            raise FactorFiLMEvaluationError("unknown first-interaction schema")

    def to_first_interaction_record(self) -> FirstInteractionRecord:
        initial_error = self.nearest_per_task_task_id != self.requested_task_id
        record = FirstInteractionRecord(
            observation_id=self.observation_id,
            requested_task_id=self.requested_task_id,
            requested_target_object_id=self.requested_target_object_id,
            first_object_grasped=self.first_object_grasped,
            first_object_displaced=self.first_object_displaced,
            displacement_threshold_m=self.displacement_threshold_m,
            requested_destination_bin_id=self.requested_destination_bin_id,
            first_bin_approached=self.first_bin_approached,
            first_bin_entered_by_any_object=self.first_bin_entered_by_any_object,
            final_object_bin_relationship=self.final_object_bin_relationship,
            nearest_per_task_task_id=self.nearest_per_task_task_id,
            first_grasp_matches_nearest_per_task=(
                None
                if self.first_object_grasped is None
                else self.first_object_grasped == _task_object_id(self.nearest_per_task_task_id)
            ),
            initial_semantic_error_visible=initial_error,
            failure_class=self.failure_class,
        )
        expected_stage = _semantic_error_stage(record)
        if self.semantic_error_stage is not expected_stage:
            raise FactorFiLMEvaluationError("semantic error stage is inconsistent")
        return record

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = dict(value)
        for name in ("final_target_pose", "final_target_velocity"):
            item = payload.get(name)
            if not isinstance(item, list | tuple):
                raise TypeError(f"{name} must be an array")
            payload[name] = tuple(item)
        return cls(**payload)


def _task_object_id(task_id: str) -> str:
    return OBJECT_IDS[M43_CANONICAL_TASK_IDS.index(task_id) // 2]


def _task_bin_id(task_id: str) -> str:
    return BIN_IDS[M43_CANONICAL_TASK_IDS.index(task_id) % 2]


def _semantic_error_stage(record: FirstInteractionRecord) -> SemanticErrorStage:
    return _semantic_error_stage_from_fields(
        initial_semantic_error=record.initial_semantic_error_visible,
        requested_object=record.requested_target_object_id,
        requested_bin=record.requested_destination_bin_id,
        first_object_grasped=record.first_object_grasped,
        first_object_displaced=record.first_object_displaced,
        first_bin_approached=record.first_bin_approached,
        first_bin_entered=record.first_bin_entered_by_any_object,
        final_relationship=record.final_object_bin_relationship,
    )


def _semantic_error_stage_from_fields(
    *,
    initial_semantic_error: bool,
    requested_object: str,
    requested_bin: str,
    first_object_grasped: str | None,
    first_object_displaced: str | None,
    first_bin_approached: str | None,
    first_bin_entered: str | None,
    final_relationship: Mapping[str, object],
) -> SemanticErrorStage:
    if initial_semantic_error:
        return SemanticErrorStage.INITIAL_CHUNK
    interactions = (
        first_object_grasped,
        first_object_displaced,
        first_bin_approached,
        first_bin_entered,
    )
    if all(value is None for value in interactions):
        return SemanticErrorStage.NO_FIRST_INTERACTION
    wrong_object = any(
        value is not None and value != requested_object
        for value in (first_object_grasped, first_object_displaced)
    )
    wrong_bin = any(
        value is not None and value != requested_bin
        for value in (first_bin_approached, first_bin_entered)
    )
    target_final_bin = final_relationship.get(requested_object)
    wrong_final_relationship = (
        target_final_bin is not None and target_final_bin != requested_bin
    ) or any(
        object_id != requested_object and bin_id == requested_bin
        for object_id, bin_id in final_relationship.items()
    )
    if wrong_object or wrong_bin or wrong_final_relationship:
        return SemanticErrorStage.EMERGED_AFTER_INITIAL_CHUNK
    return SemanticErrorStage.NONE


def _first_unique_label(
    values: Sequence[tuple[int, tuple[bool, ...]]], labels: Sequence[str], _name: str
) -> str | None:
    for _, flags in values:
        selected = tuple(labels[index] for index, value in enumerate(flags) if value)
        if selected:
            # Discrete sampling can make multiple geometric predicates become
            # true at one step.  Resolve that tie by the declared semantic
            # order, never by mapping iteration order.
            return selected[0]
    return None


def derive_first_interaction_snapshot(
    *,
    observation_id: str,
    requested_task_id: str,
    nearest_per_task_task_id: str,
    step_snapshots: Sequence[FactorFiLMSemanticStepSnapshot | Mapping[str, object]],
    timed_out: bool,
    environment_failure: bool,
    raw_actions_by_step: Mapping[int, Sequence[object]] | None = None,
) -> FirstInteractionSnapshot:
    """Derive episode first-interaction evidence from ordered physical snapshots."""
    if requested_task_id not in M43_CANONICAL_TASK_IDS or (
        nearest_per_task_task_id not in M43_CANONICAL_TASK_IDS
    ):
        raise FactorFiLMEvaluationError("first-interaction task IDs must be canonical")
    steps = tuple(
        item
        if isinstance(item, FactorFiLMSemanticStepSnapshot)
        else FactorFiLMSemanticStepSnapshot.from_dict(item)
        for item in step_snapshots
    )
    if not steps or steps[0].step != 0:
        raise FactorFiLMEvaluationError("first-interaction snapshots require reset step zero")
    if tuple(item.step for item in steps) != tuple(sorted({item.step for item in steps})):
        raise FactorFiLMEvaluationError(
            "first-interaction snapshot steps must be unique and ordered"
        )
    if steps[0].executed_action is not None or any(
        item.executed_action is None for item in steps[1:]
    ):
        raise FactorFiLMEvaluationError("only the reset snapshot may omit its executed action")
    raw_actions: dict[int, tuple[float, ...]] | None = None
    if raw_actions_by_step is not None:
        raw_actions = {
            step: _vector(action, length=8, name=f"raw_actions_by_step[{step}]")
            for step, action in raw_actions_by_step.items()
        }
        expected_action_steps = {item.step for item in steps[1:]}
        if set(raw_actions) != expected_action_steps:
            raise FactorFiLMEvaluationError(
                "raw release-command actions must cover every executed simulator step"
            )

    target_object = _task_object_id(requested_task_id)
    target_index = OBJECT_IDS.index(target_object)
    destination_bin = _task_bin_id(requested_task_id)
    destination_index = BIN_IDS.index(destination_bin)
    initial_positions = steps[0].cube_positions

    first_grasped = _first_unique_label(
        tuple((item.step, item.object_is_grasped) for item in steps),
        OBJECT_IDS,
        "first object grasped",
    )
    displaced_by_step = tuple(
        (
            item.step,
            tuple(
                math.dist(initial_positions[index], item.cube_positions[index])
                > FACTOR_FILM_DISPLACEMENT_THRESHOLD_METERS
                for index in range(len(OBJECT_IDS))
            ),
        )
        for item in steps
    )
    first_displaced = _first_unique_label(
        displaced_by_step,
        OBJECT_IDS,
        "first object displaced",
    )
    approach_by_step = tuple(
        (
            item.step,
            tuple(
                math.dist(
                    item.tcp_position[:2],
                    item.bin_floor_centers[index][:2],
                )
                <= FACTOR_FILM_DESTINATION_NEAR_XY_METERS
                for index in range(len(BIN_IDS))
            ),
        )
        for item in steps
    )
    first_bin_approached = _first_unique_label(
        approach_by_step,
        BIN_IDS,
        "first bin approached",
    )
    entry_by_step = tuple(
        (
            item.step,
            tuple(
                any(item.object_in_bin[object_index][bin_index] for object_index in range(3))
                for bin_index in range(2)
            ),
        )
        for item in steps
    )
    first_bin_entered = _first_unique_label(
        entry_by_step,
        BIN_IDS,
        "first bin entered",
    )
    final = steps[-1]
    required_final_evaluation = {
        "success",
        "target_in_target_bin",
        "target_is_static",
        "target_off_table",
    }
    missing_final = required_final_evaluation - set(final.evaluation)
    if missing_final:
        raise FactorFiLMEvaluationError(
            "final environment evaluation is missing " + ", ".join(sorted(missing_final))
        )
    final_relationship = {
        object_id: next(
            (
                BIN_IDS[bin_index]
                for bin_index, value in enumerate(final.object_in_bin[object_index])
                if value
            ),
            None,
        )
        for object_index, object_id in enumerate(OBJECT_IDS)
    }
    first_target_grasp_step = next(
        (item.step for item in steps if item.object_is_grasped[target_index]),
        None,
    )
    first_release_step = next(
        (
            item.step
            for item in steps
            if first_target_grasp_step is not None
            and item.step >= first_target_grasp_step
            and (
                raw_actions[item.step][7]
                if raw_actions is not None
                else cast(tuple[float, ...], item.executed_action)[7]
            )
            >= 0.0
        ),
        None,
    )
    target_heights = tuple(item.cube_positions[target_index][2] for item in steps)
    target_to_destination = tuple(
        math.dist(
            item.cube_positions[target_index],
            item.bin_floor_centers[destination_index],
        )
        for item in steps
    )
    target_near_destination = tuple(
        item
        for item in steps
        if math.dist(
            item.cube_positions[target_index][:2],
            item.bin_floor_centers[destination_index][:2],
        )
        <= FACTOR_FILM_DESTINATION_NEAR_XY_METERS
    )
    ever_lifted = max(target_heights) >= target_heights[0] + FACTOR_FILM_LIFT_DELTA_METERS
    ever_descended = any(
        item.cube_positions[target_index][2]
        <= item.bin_floor_centers[destination_index][2] + FACTOR_FILM_DESCENT_HEIGHT_METERS
        for item in target_near_destination
    )
    released_inside = any(
        item.step >= (first_release_step if first_release_step is not None else 10**9)
        and bool(item.evaluation.get("target_in_target_bin", False))
        for item in steps
    )
    success_seen = any(bool(item.evaluation.get("success", False)) for item in steps)
    final_success = bool(final.evaluation.get("success", False))
    if final_success and (
        not bool(final.evaluation["target_in_target_bin"])
        or not bool(final.evaluation["target_is_static"])
        or bool(final.evaluation["target_off_table"])
    ):
        raise FactorFiLMEvaluationError("final success contradicts M1 evaluation fields")
    if not isinstance(timed_out, bool) or not isinstance(environment_failure, bool):
        raise TypeError("timed_out and environment_failure must be booleans")
    if final_success and (timed_out or environment_failure):
        raise FactorFiLMEvaluationError("successful rollout cannot time out or fail infrastructure")
    wrong_object_grasped = any(
        any(
            grasped for index, grasped in enumerate(item.object_is_grasped) if index != target_index
        )
        for item in steps
    )
    trace = SemanticRolloutTrace(
        success=final_success,
        environment_failure=environment_failure,
        wrong_object_grasped=wrong_object_grasped,
        ever_grasped_target=first_target_grasp_step is not None,
        ever_lifted_target=ever_lifted,
        ever_transported_to_destination=bool(target_near_destination),
        ever_descended_at_destination=ever_descended,
        release_command_observed=first_release_step is not None,
        released_inside_success_region=released_inside,
        final_target_static=bool(final.evaluation.get("target_is_static", False)),
        success_observed_before_final_step=success_seen and not final_success,
        timed_out=timed_out,
    )
    canonical = build_first_interaction_record(
        observation_id=observation_id,
        requested_task_id=requested_task_id,
        first_object_grasped=first_grasped,
        first_object_displaced=first_displaced,
        requested_destination_bin_id=destination_bin,
        first_bin_approached=first_bin_approached,
        first_bin_entered_by_any_object=first_bin_entered,
        final_object_bin_relationship=final_relationship,
        nearest_per_task_task_id=nearest_per_task_task_id,
        trace=trace,
        displacement_threshold_m=FACTOR_FILM_DISPLACEMENT_THRESHOLD_METERS,
    )
    result = FirstInteractionSnapshot(
        observation_id=observation_id,
        requested_task_id=requested_task_id,
        requested_target_object_id=target_object,
        requested_destination_bin_id=destination_bin,
        nearest_per_task_task_id=nearest_per_task_task_id,
        first_object_grasped=first_grasped,
        first_object_displaced=first_displaced,
        first_bin_approached=first_bin_approached,
        first_bin_entered_by_any_object=first_bin_entered,
        final_object_bin_relationship=final_relationship,
        first_target_grasp_step=first_target_grasp_step,
        first_release_command_step=first_release_step,
        maximum_target_height=max(target_heights),
        closest_target_to_bin_distance=min(target_to_destination),
        final_target_pose=(
            *final.cube_positions[target_index],
            *final.cube_orientations[target_index],
        ),
        final_target_velocity=(
            *final.cube_linear_velocities[target_index],
            *final.cube_angular_velocities[target_index],
        ),
        final_environment_evaluation=final.evaluation,
        failure_class=canonical.failure_class,
        semantic_error_stage=_semantic_error_stage(canonical),
    )
    if result.to_first_interaction_record() != canonical:
        raise FactorFiLMEvaluationError("first-interaction derivation changed M4.3 semantics")
    return result


def factor_film_first_interaction_confusions(
    snapshots: Sequence[FirstInteractionSnapshot],
) -> dict[str, object]:
    """Return the five canonical JSON confusion matrices for real new rollouts."""
    records = tuple(value.to_first_interaction_record() for value in snapshots)
    if not records:
        raise FactorFiLMEvaluationError("first-interaction analysis requires episode evidence")
    return {
        name: confusion.to_dict()
        for name, confusion in first_interaction_confusions(records).items()
    }


@dataclass(frozen=True, slots=True)
class DevelopmentEpisodeIdentity(_JsonRecord):
    episode_index: int
    scene_seed: int
    scene_id: str
    task_id: str

    def __post_init__(self) -> None:
        _integer(self.episode_index, "episode_index", maximum=71)
        _integer(self.scene_seed, "scene_seed")
        if self.scene_id != stable_scene_id(self.scene_seed):
            raise FactorFiLMEvaluationError("development scene ID does not match its seed")
        if self.task_id not in M43_CANONICAL_TASK_IDS:
            raise FactorFiLMEvaluationError("development task ID is not canonical")

    @property
    def semantic_key(self) -> tuple[str, str]:
        return (self.scene_id, self.task_id)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class PairedDevelopmentIdentityRecord(_JsonRecord):
    schedule_fingerprint: str
    ordered_episode_identities: tuple[DevelopmentEpisodeIdentity, ...]
    policy_episode_fingerprints: Mapping[str, object]
    paired_identity_fingerprint: str
    schedule_id: str = M42_DEV_SCHEDULE_ID
    episode_count: int = FACTOR_FILM_DEVELOPMENT_EPISODE_COUNT
    policy_labels: tuple[str, ...] = FACTOR_FILM_DEVELOPMENT_POLICY_LABELS
    test_split_accessed: bool = False
    fresh_seed_accessed: bool = False
    final_schedule_accessed: bool = False
    schema_version: str = FACTOR_FILM_PAIRED_IDENTITIES_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schedule_id != M42_DEV_SCHEDULE_ID or (
            self.schedule_fingerprint != M42_DEV_SCHEDULE_FINGERPRINT
        ):
            raise FactorFiLMEvaluationError("paired evidence must use immutable m42_dev_v0")
        if (
            self.episode_count != FACTOR_FILM_DEVELOPMENT_EPISODE_COUNT
            or len(self.ordered_episode_identities) != self.episode_count
        ):
            raise FactorFiLMEvaluationError("paired development evidence requires 72 episodes")
        if tuple(self.policy_labels) != FACTOR_FILM_DEVELOPMENT_POLICY_LABELS:
            raise FactorFiLMEvaluationError("paired development policy set changed")
        if not all(
            isinstance(value, DevelopmentEpisodeIdentity)
            for value in self.ordered_episode_identities
        ):
            raise TypeError("paired episode identities are malformed")
        if tuple(value.episode_index for value in self.ordered_episode_identities) != tuple(
            range(self.episode_count)
        ):
            raise FactorFiLMEvaluationError("paired episode indices must be exactly 0..71")
        semantic_keys = tuple(value.semantic_key for value in self.ordered_episode_identities)
        if len(set(semantic_keys)) != self.episode_count:
            raise FactorFiLMEvaluationError("paired development identities contain duplicates")
        if len({value.scene_id for value in self.ordered_episode_identities}) != 12:
            raise FactorFiLMEvaluationError("m42_dev_v0 must contain exactly 12 scenes")
        tasks_by_scene: dict[str, set[str]] = {}
        for value in self.ordered_episode_identities:
            tasks_by_scene.setdefault(value.scene_id, set()).add(value.task_id)
        if any(tasks != set(M43_CANONICAL_TASK_IDS) for tasks in tasks_by_scene.values()):
            raise FactorFiLMEvaluationError("every m42_dev_v0 scene must contain all six tasks")
        task_counts = Counter(value.task_id for value in self.ordered_episode_identities)
        if task_counts != Counter(
            {
                task_id: FACTOR_FILM_DEVELOPMENT_EPISODES_PER_TASK
                for task_id in M43_CANONICAL_TASK_IDS
            }
        ):
            raise FactorFiLMEvaluationError("m42_dev_v0 must contain 12 episodes per task")
        fingerprints = dict(self.policy_episode_fingerprints)
        if set(fingerprints) != set(self.policy_labels):
            raise FactorFiLMEvaluationError("paired policy fingerprints are incomplete")
        for label, value in fingerprints.items():
            _digest(cast(str, value), f"policy_episode_fingerprints[{label}]")
        object.__setattr__(self, "policy_episode_fingerprints", _FrozenMapping(fingerprints))
        _digest(self.paired_identity_fingerprint, "paired_identity_fingerprint")
        expected = canonical_fingerprint(
            {
                "schedule_id": self.schedule_id,
                "schedule_fingerprint": self.schedule_fingerprint,
                "ordered_episode_identities": [
                    value.to_dict() for value in self.ordered_episode_identities
                ],
                "policy_episode_fingerprints": fingerprints,
            }
        )
        if self.paired_identity_fingerprint != expected:
            raise FactorFiLMEvaluationError("paired identity fingerprint mismatch")
        if any((self.test_split_accessed, self.fresh_seed_accessed, self.final_schedule_accessed)):
            raise FactorFiLMEvaluationError(
                "paired development evidence accessed a forbidden source"
            )
        if self.schema_version != FACTOR_FILM_PAIRED_IDENTITIES_SCHEMA_VERSION:
            raise FactorFiLMEvaluationError("unknown paired-identity schema")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = dict(value)
        identities = payload.pop("ordered_episode_identities")
        labels = payload.get("policy_labels")
        if not isinstance(identities, list | tuple) or not isinstance(labels, list | tuple):
            raise TypeError("paired identity arrays are malformed")
        payload["policy_labels"] = tuple(labels)
        return cls(
            ordered_episode_identities=tuple(
                DevelopmentEpisodeIdentity.from_dict(cast(Mapping[str, object], item))
                for item in identities
            ),
            **payload,
        )


def validate_paired_development_identities(
    *,
    per_task: Sequence[DevelopmentEpisodeIdentity],
    state_onehot: Sequence[DevelopmentEpisodeIdentity],
    factor_film: Sequence[DevelopmentEpisodeIdentity],
    schedule_fingerprint: str,
) -> PairedDevelopmentIdentityRecord:
    """Require PerTask, State-OneHot, and FactorFiLM to cover identical 72 episodes."""
    by_policy = {
        FACTOR_FILM_DEVELOPMENT_POLICY_LABELS[0]: tuple(per_task),
        FACTOR_FILM_DEVELOPMENT_POLICY_LABELS[1]: tuple(state_onehot),
        FACTOR_FILM_DEVELOPMENT_POLICY_LABELS[2]: tuple(factor_film),
    }
    if schedule_fingerprint != M42_DEV_SCHEDULE_FINGERPRINT:
        raise FactorFiLMEvaluationError("paired evidence must use immutable m42_dev_v0")
    if not all(len(value) == FACTOR_FILM_DEVELOPMENT_EPISODE_COUNT for value in by_policy.values()):
        raise FactorFiLMEvaluationError("every paired policy must contain 72 episodes")
    ordered_by_policy = {
        label: tuple(sorted(values, key=lambda value: value.episode_index))
        for label, values in by_policy.items()
    }
    canonical = ordered_by_policy[FACTOR_FILM_DEVELOPMENT_POLICY_LABELS[0]]
    canonical_keys = tuple((value.episode_index, *value.semantic_key) for value in canonical)
    for label, values in ordered_by_policy.items():
        if tuple((value.episode_index, *value.semantic_key) for value in values) != canonical_keys:
            raise FactorFiLMEvaluationError(
                f"{label} does not cover identical ordered development identities"
            )
    policy_fingerprints = {
        label: canonical_fingerprint([value.to_dict() for value in values])
        for label, values in ordered_by_policy.items()
    }
    payload = {
        "schedule_id": M42_DEV_SCHEDULE_ID,
        "schedule_fingerprint": schedule_fingerprint,
        "ordered_episode_identities": [value.to_dict() for value in canonical],
        "policy_episode_fingerprints": policy_fingerprints,
    }
    return PairedDevelopmentIdentityRecord(
        schedule_fingerprint=schedule_fingerprint,
        ordered_episode_identities=canonical,
        policy_episode_fingerprints=policy_fingerprints,
        paired_identity_fingerprint=canonical_fingerprint(payload),
    )


@dataclass(frozen=True, slots=True)
class DevelopmentPolicyMetrics(_JsonRecord):
    """Exact FactorFiLM development counts consumed by the quality gate."""

    report_fingerprint: str
    success_count: int
    per_task_success_counts: Mapping[str, object]
    wrong_object_grasp_count: int
    wrong_object_in_target_bin_count: int
    target_in_wrong_bin_count: int
    target_off_table_count: int
    timeout_count: int
    arm_projection_count: int
    nan_count: int
    inf_count: int
    malformed_action_count: int
    episode_count: int = FACTOR_FILM_DEVELOPMENT_EPISODE_COUNT
    schedule_id: str = M42_DEV_SCHEDULE_ID
    schedule_fingerprint: str = M42_DEV_SCHEDULE_FINGERPRINT
    test_split_accessed: bool = False
    fresh_seed_accessed: bool = False
    final_schedule_accessed: bool = False

    def __post_init__(self) -> None:
        _digest(self.report_fingerprint, "report_fingerprint")
        if (
            self.episode_count != FACTOR_FILM_DEVELOPMENT_EPISODE_COUNT
            or self.schedule_id != M42_DEV_SCHEDULE_ID
            or self.schedule_fingerprint != M42_DEV_SCHEDULE_FINGERPRINT
        ):
            raise FactorFiLMEvaluationError("development metrics must use complete m42_dev_v0")
        _integer(self.success_count, "success_count", maximum=self.episode_count)
        counts = dict(self.per_task_success_counts)
        if set(counts) != set(M43_CANONICAL_TASK_IDS):
            raise FactorFiLMEvaluationError("per-task success counts must cover all six tasks")
        for task_id, value in counts.items():
            _integer(
                cast(int, value),
                f"per_task_success_counts[{task_id}]",
                maximum=FACTOR_FILM_DEVELOPMENT_EPISODES_PER_TASK,
            )
        if sum(cast(int, value) for value in counts.values()) != self.success_count:
            raise FactorFiLMEvaluationError("per-task successes disagree with overall success")
        object.__setattr__(self, "per_task_success_counts", _FrozenMapping(counts))
        for name in (
            "wrong_object_grasp_count",
            "wrong_object_in_target_bin_count",
            "target_in_wrong_bin_count",
            "target_off_table_count",
            "timeout_count",
        ):
            _integer(
                cast(int, getattr(self, name)),
                name,
                maximum=self.episode_count,
            )
        for name in (
            "arm_projection_count",
            "nan_count",
            "inf_count",
            "malformed_action_count",
        ):
            _integer(cast(int, getattr(self, name)), name)
        if any((self.test_split_accessed, self.fresh_seed_accessed, self.final_schedule_accessed)):
            raise FactorFiLMEvaluationError("development metrics accessed a forbidden source")

    @property
    def success_rate(self) -> float:
        return self.success_count / self.episode_count

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class DevelopmentSemanticMetrics(_JsonRecord):
    semantic_report_fingerprint: str
    full_task_top1_retrieval: float
    target_object_retrieval: float
    task_sensitivity_ratio_relative_to_per_task: float
    observation_count: int = FACTOR_FILM_DEVELOPMENT_EPISODE_COUNT
    schedule_id: str = M42_DEV_SCHEDULE_ID
    schedule_fingerprint: str = M42_DEV_SCHEDULE_FINGERPRINT
    primary_metric: str = PRIMARY_ACTION_CHUNK_DISTANCE_METRIC
    test_split_accessed: bool = False
    fresh_seed_accessed: bool = False
    final_schedule_accessed: bool = False

    def __post_init__(self) -> None:
        _digest(self.semantic_report_fingerprint, "semantic_report_fingerprint")
        if (
            self.observation_count != FACTOR_FILM_DEVELOPMENT_EPISODE_COUNT
            or self.schedule_id != M42_DEV_SCHEDULE_ID
            or self.schedule_fingerprint != M42_DEV_SCHEDULE_FINGERPRINT
        ):
            raise FactorFiLMEvaluationError("semantic metrics must use complete m42_dev_v0")
        if self.primary_metric != PRIMARY_ACTION_CHUNK_DISTANCE_METRIC:
            raise FactorFiLMEvaluationError("FactorFiLM primary semantic metric changed")
        _rate(self.full_task_top1_retrieval, "full_task_top1_retrieval")
        _rate(self.target_object_retrieval, "target_object_retrieval")
        _finite(
            self.task_sensitivity_ratio_relative_to_per_task,
            "task_sensitivity_ratio_relative_to_per_task",
            minimum=0.0,
        )
        if any((self.test_split_accessed, self.fresh_seed_accessed, self.final_schedule_accessed)):
            raise FactorFiLMEvaluationError("semantic metrics accessed a forbidden source")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        return cls(**dict(value))


class DevelopmentGateCriterion(StrEnum):
    OVERALL_SUCCESS_COUNT = "overall_success_count"
    OVERALL_SUCCESS_RATE = "overall_success_rate"
    SUCCESS_GAP_TO_PER_TASK = "success_gap_to_per_task"
    EVERY_TASK_SUCCESS_COUNT = "every_task_success_count"
    WRONG_OBJECT_GRASP_COUNT = "wrong_object_grasp_count"
    WRONG_OBJECT_IN_TARGET_BIN_COUNT = "wrong_object_in_target_bin_count"
    TIMEOUT_COUNT = "timeout_count"
    TARGET_IN_WRONG_BIN_COUNT = "target_in_wrong_bin_count"
    TARGET_OFF_TABLE_COUNT = "target_off_table_count"
    ARM_PROJECTION_COUNT = "arm_projection_count"
    NAN_COUNT = "nan_count"
    INF_COUNT = "inf_count"
    MALFORMED_ACTION_COUNT = "malformed_action_count"
    FULL_TASK_TOP1_RETRIEVAL = "full_task_top1_retrieval"
    TARGET_OBJECT_RETRIEVAL = "target_object_retrieval"
    TASK_SENSITIVITY_RATIO = "task_sensitivity_ratio_relative_to_per_task"


class GateComparison(StrEnum):
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    EQUAL = "equal"


@dataclass(frozen=True, slots=True)
class DevelopmentQualityGateCheck(_JsonRecord):
    criterion: DevelopmentGateCriterion
    comparison: GateComparison
    observed: int | float
    threshold: int | float
    passed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "criterion", DevelopmentGateCriterion(self.criterion))
        object.__setattr__(self, "comparison", GateComparison(self.comparison))
        _finite(self.observed, "observed")
        _finite(self.threshold, "threshold")
        expected = {
            GateComparison.GREATER_THAN_OR_EQUAL: self.observed >= self.threshold,
            GateComparison.LESS_THAN_OR_EQUAL: self.observed <= self.threshold,
            GateComparison.EQUAL: self.observed == self.threshold,
        }[self.comparison]
        if self.passed != expected:
            raise FactorFiLMEvaluationError("quality-gate check truth value is inconsistent")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class DevelopmentQualityGateResult(_JsonRecord):
    factor_film_metrics: DevelopmentPolicyMetrics
    semantic_metrics: DevelopmentSemanticMetrics
    per_task_reference_success_count: int
    checks: tuple[DevelopmentQualityGateCheck, ...]
    development_quality_gate_passed: bool
    final_benchmark_authorized: bool
    final_schedule_accessed: bool = False
    test_split_accessed: bool = False
    fresh_seed_accessed: bool = False
    smolvla_go: bool = False
    schema_version: str = FACTOR_FILM_DEVELOPMENT_GATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.factor_film_metrics, DevelopmentPolicyMetrics) or not isinstance(
            self.semantic_metrics, DevelopmentSemanticMetrics
        ):
            raise TypeError("development gate requires typed rollout and semantic metrics")
        _integer(
            self.per_task_reference_success_count,
            "per_task_reference_success_count",
            maximum=FACTOR_FILM_DEVELOPMENT_EPISODE_COUNT,
        )
        if not all(isinstance(value, DevelopmentQualityGateCheck) for value in self.checks):
            raise TypeError("development gate checks are malformed")
        if tuple(value.criterion for value in self.checks) != tuple(DevelopmentGateCriterion):
            raise FactorFiLMEvaluationError("development gate must contain all checks in order")
        passed = all(value.passed for value in self.checks)
        if (
            self.development_quality_gate_passed != passed
            or self.final_benchmark_authorized != passed
        ):
            raise FactorFiLMEvaluationError("development gate authorization is inconsistent")
        if any(
            (
                self.final_schedule_accessed,
                self.test_split_accessed,
                self.fresh_seed_accessed,
                self.smolvla_go,
            )
        ):
            raise FactorFiLMEvaluationError("development gate accessed or authorized future work")
        if self.schema_version != FACTOR_FILM_DEVELOPMENT_GATE_SCHEMA_VERSION:
            raise FactorFiLMEvaluationError("unknown development-gate schema")

    @property
    def failed_criteria(self) -> tuple[str, ...]:
        return tuple(value.criterion.value for value in self.checks if not value.passed)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = dict(value)
        metrics = payload.pop("factor_film_metrics")
        semantic = payload.pop("semantic_metrics")
        checks = payload.pop("checks")
        if (
            not isinstance(metrics, Mapping)
            or not isinstance(semantic, Mapping)
            or not isinstance(checks, list | tuple)
        ):
            raise TypeError("development gate nested records are malformed")
        return cls(
            factor_film_metrics=DevelopmentPolicyMetrics.from_dict(metrics),
            semantic_metrics=DevelopmentSemanticMetrics.from_dict(semantic),
            checks=tuple(
                DevelopmentQualityGateCheck.from_dict(cast(Mapping[str, object], item))
                for item in checks
            ),
            **payload,
        )


def _check(
    criterion: DevelopmentGateCriterion,
    comparison: GateComparison,
    observed: int | float,
    threshold: int | float,
) -> DevelopmentQualityGateCheck:
    passed = {
        GateComparison.GREATER_THAN_OR_EQUAL: observed >= threshold,
        GateComparison.LESS_THAN_OR_EQUAL: observed <= threshold,
        GateComparison.EQUAL: observed == threshold,
    }[comparison]
    return DevelopmentQualityGateCheck(
        criterion=criterion,
        comparison=comparison,
        observed=observed,
        threshold=threshold,
        passed=passed,
    )


def evaluate_development_quality_gate(
    *,
    factor_film_metrics: DevelopmentPolicyMetrics,
    semantic_metrics: DevelopmentSemanticMetrics,
    per_task_reference_success_count: int,
) -> DevelopmentQualityGateResult:
    """Evaluate all sixteen immutable M4.3b development conditions conjunctively."""
    _integer(
        per_task_reference_success_count,
        "per_task_reference_success_count",
        maximum=FACTOR_FILM_DEVELOPMENT_EPISODE_COUNT,
    )
    minimum_task_success = min(
        cast(int, value) for value in factor_film_metrics.per_task_success_counts.values()
    )
    gap = per_task_reference_success_count - factor_film_metrics.success_count
    checks = (
        _check(
            DevelopmentGateCriterion.OVERALL_SUCCESS_COUNT,
            GateComparison.GREATER_THAN_OR_EQUAL,
            factor_film_metrics.success_count,
            50,
        ),
        _check(
            DevelopmentGateCriterion.OVERALL_SUCCESS_RATE,
            GateComparison.GREATER_THAN_OR_EQUAL,
            factor_film_metrics.success_rate,
            0.6944,
        ),
        _check(
            DevelopmentGateCriterion.SUCCESS_GAP_TO_PER_TASK,
            GateComparison.LESS_THAN_OR_EQUAL,
            gap,
            8,
        ),
        _check(
            DevelopmentGateCriterion.EVERY_TASK_SUCCESS_COUNT,
            GateComparison.GREATER_THAN_OR_EQUAL,
            minimum_task_success,
            7,
        ),
        _check(
            DevelopmentGateCriterion.WRONG_OBJECT_GRASP_COUNT,
            GateComparison.LESS_THAN_OR_EQUAL,
            factor_film_metrics.wrong_object_grasp_count,
            6,
        ),
        _check(
            DevelopmentGateCriterion.WRONG_OBJECT_IN_TARGET_BIN_COUNT,
            GateComparison.LESS_THAN_OR_EQUAL,
            factor_film_metrics.wrong_object_in_target_bin_count,
            2,
        ),
        _check(
            DevelopmentGateCriterion.TIMEOUT_COUNT,
            GateComparison.LESS_THAN_OR_EQUAL,
            factor_film_metrics.timeout_count,
            22,
        ),
        _check(
            DevelopmentGateCriterion.TARGET_IN_WRONG_BIN_COUNT,
            GateComparison.EQUAL,
            factor_film_metrics.target_in_wrong_bin_count,
            0,
        ),
        _check(
            DevelopmentGateCriterion.TARGET_OFF_TABLE_COUNT,
            GateComparison.EQUAL,
            factor_film_metrics.target_off_table_count,
            0,
        ),
        _check(
            DevelopmentGateCriterion.ARM_PROJECTION_COUNT,
            GateComparison.EQUAL,
            factor_film_metrics.arm_projection_count,
            0,
        ),
        _check(
            DevelopmentGateCriterion.NAN_COUNT,
            GateComparison.EQUAL,
            factor_film_metrics.nan_count,
            0,
        ),
        _check(
            DevelopmentGateCriterion.INF_COUNT,
            GateComparison.EQUAL,
            factor_film_metrics.inf_count,
            0,
        ),
        _check(
            DevelopmentGateCriterion.MALFORMED_ACTION_COUNT,
            GateComparison.EQUAL,
            factor_film_metrics.malformed_action_count,
            0,
        ),
        _check(
            DevelopmentGateCriterion.FULL_TASK_TOP1_RETRIEVAL,
            GateComparison.GREATER_THAN_OR_EQUAL,
            semantic_metrics.full_task_top1_retrieval,
            0.70,
        ),
        _check(
            DevelopmentGateCriterion.TARGET_OBJECT_RETRIEVAL,
            GateComparison.GREATER_THAN_OR_EQUAL,
            semantic_metrics.target_object_retrieval,
            0.80,
        ),
        _check(
            DevelopmentGateCriterion.TASK_SENSITIVITY_RATIO,
            GateComparison.GREATER_THAN_OR_EQUAL,
            semantic_metrics.task_sensitivity_ratio_relative_to_per_task,
            0.75,
        ),
    )
    passed = all(value.passed for value in checks)
    return DevelopmentQualityGateResult(
        factor_film_metrics=factor_film_metrics,
        semantic_metrics=semantic_metrics,
        per_task_reference_success_count=per_task_reference_success_count,
        checks=checks,
        development_quality_gate_passed=passed,
        final_benchmark_authorized=passed,
    )


__all__ = [
    "FACTOR_FILM_DEVELOPMENT_EPISODE_COUNT",
    "FACTOR_FILM_DEVELOPMENT_POLICY_LABELS",
    "FACTOR_FILM_DESCENT_HEIGHT_METERS",
    "FACTOR_FILM_DESTINATION_NEAR_XY_METERS",
    "FACTOR_FILM_DISPLACEMENT_THRESHOLD_METERS",
    "FACTOR_FILM_LIFT_DELTA_METERS",
    "FACTOR_FILM_RELOAD_ATOL",
    "FACTOR_FILM_RELOAD_RTOL",
    "FACTOR_FILM_VALIDATION_EPISODE_COUNT",
    "DevelopmentEpisodeIdentity",
    "DevelopmentGateCriterion",
    "DevelopmentPolicyMetrics",
    "DevelopmentQualityGateCheck",
    "DevelopmentQualityGateResult",
    "DevelopmentSemanticMetrics",
    "FactorFiLMEvaluationError",
    "FactorFiLMReloadValidationRecord",
    "FactorFiLMSemanticStepSnapshot",
    "FactorFiLMValidationCountResult",
    "FirstInteractionSnapshot",
    "GateComparison",
    "PairedDevelopmentIdentityRecord",
    "SemanticErrorStage",
    "derive_first_interaction_snapshot",
    "evaluate_development_quality_gate",
    "factor_film_first_interaction_confusions",
    "factor_film_validation_count_result",
    "rank_factor_film_validation_count_results",
    "validate_paired_development_identities",
    "validate_reload_against_selection",
]
