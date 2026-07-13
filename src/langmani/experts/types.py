"""Typed, JSON-ready contracts for the M2 privileged expert.

The contracts deliberately contain only compact metadata and counters. Planner
trajectories, tensors, observations, and rendered images remain runtime values and
must not be embedded in these result objects.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any


class ExpertStatus(StrEnum):
    """Stable terminal statuses for an expert rollout or phase."""

    SUCCESS = "success"
    INVALID_TASK = "invalid_task"
    INITIALIZATION_FAILURE = "initialization_failure"
    IK_FAILURE = "ik_failure"
    PLANNING_FAILURE = "planning_failure"
    EXECUTION_FAILURE = "execution_failure"
    GRASP_FAILURE = "grasp_failure"
    TRANSPORT_FAILURE = "transport_failure"
    PLACEMENT_FAILURE = "placement_failure"
    VERIFICATION_FAILURE = "verification_failure"
    TARGET_OFF_TABLE = "target_off_table"
    TIMEOUT = "timeout"
    UNEXPECTED_EXCEPTION = "unexpected_exception"


class ExpertPhase(StrEnum):
    """Stable identifiers for the explicit M2 pick-and-place sequence."""

    INITIALIZE = "initialize"
    MOVE_TO_PREGRASP = "move_to_pregrasp"
    APPROACH_TARGET = "approach_target"
    CLOSE_GRIPPER = "close_gripper"
    VERIFY_GRASP = "verify_grasp"
    LIFT_TARGET = "lift_target"
    MOVE_ABOVE_DESTINATION = "move_above_destination"
    DESCEND_TO_PLACE = "descend_to_place"
    OPEN_GRIPPER = "open_gripper"
    SETTLE_AFTER_RELEASE = "settle_after_release"
    RETREAT = "retreat"
    VERIFY_TASK = "verify_task"


EXPERT_PHASE_SEQUENCE: tuple[ExpertPhase, ...] = tuple(ExpertPhase)


def _require_bool(value: object, *, field_name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool")


def _require_int(value: object, *, field_name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{field_name} must be {qualifier}")


def _finite_float(value: object, *, field_name: str, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    if positive and normalized <= 0:
        raise ValueError(f"{field_name} must be positive")
    if not positive and normalized < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return normalized


def _require_optional_string(value: object, *, field_name: str) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise TypeError(f"{field_name} must be None or a non-empty string")


@dataclass(frozen=True, slots=True)
class ExpertConfig:
    """Validated deterministic execution settings for the M2 Panda expert.

    ManiSkill 3.0.1's Panda motion-planning example supports ``pd_joint_pos``.
    The Linux dependency selected by ManiSkill is mplib 0.1.1. Its screw planner
    exposes neither a planner seed nor a timeout, so non-``None`` values for those
    optional capabilities are rejected instead of being silently ignored.
    """

    control_mode: str = "pd_joint_pos"
    max_episode_steps: int = 200
    max_planning_attempts_per_phase: int = 1
    pregrasp_clearance: float = 0.08
    grasp_approach_distance: float = 0.05
    lift_clearance: float = 0.10
    transport_clearance: float = 0.10
    placement_clearance: float = 0.002
    release_settling_steps: int = 15
    retreat_distance: float = 0.10
    planner_timeout_seconds: float | None = None
    planner_seed: int | None = None
    diagnostic_rendering: bool = False

    def __post_init__(self) -> None:
        if self.control_mode != "pd_joint_pos":
            raise ValueError("control_mode must be 'pd_joint_pos' for the M2 expert")
        _require_int(self.max_episode_steps, field_name="max_episode_steps", minimum=1)
        _require_int(
            self.max_planning_attempts_per_phase,
            field_name="max_planning_attempts_per_phase",
            minimum=1,
        )
        _require_int(
            self.release_settling_steps,
            field_name="release_settling_steps",
            minimum=0,
        )

        for field_name in (
            "pregrasp_clearance",
            "grasp_approach_distance",
            "lift_clearance",
            "transport_clearance",
            "placement_clearance",
            "retreat_distance",
        ):
            normalized = _finite_float(
                getattr(self, field_name),
                field_name=field_name,
                positive=True,
            )
            object.__setattr__(self, field_name, normalized)

        if self.planner_timeout_seconds is not None:
            raise ValueError(
                "planner_timeout_seconds is unsupported by the mplib 0.1.1 screw planner"
            )
        if self.planner_seed is not None:
            raise ValueError("planner_seed is unsupported by mplib 0.1.1")
        _require_bool(self.diagnostic_rendering, field_name="diagnostic_rendering")

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh JSON-serializable configuration mapping."""
        return {
            "control_mode": self.control_mode,
            "max_episode_steps": self.max_episode_steps,
            "max_planning_attempts_per_phase": self.max_planning_attempts_per_phase,
            "pregrasp_clearance": self.pregrasp_clearance,
            "grasp_approach_distance": self.grasp_approach_distance,
            "lift_clearance": self.lift_clearance,
            "transport_clearance": self.transport_clearance,
            "placement_clearance": self.placement_clearance,
            "release_settling_steps": self.release_settling_steps,
            "retreat_distance": self.retreat_distance,
            "planner_timeout_seconds": self.planner_timeout_seconds,
            "planner_seed": self.planner_seed,
            "diagnostic_rendering": self.diagnostic_rendering,
        }


@dataclass(frozen=True, slots=True)
class PhaseResult:
    """Compact outcome and accounting for one expert phase."""

    phase: ExpertPhase
    success: bool
    status: ExpertStatus
    attempts: int = 0
    environment_steps: int = 0
    planning_calls: int = 0
    replans: int = 0
    planning_duration_seconds: float = 0.0
    execution_duration_seconds: float = 0.0
    message: str = ""
    planner_status: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.phase, ExpertPhase):
            raise TypeError("phase must be an ExpertPhase")
        _require_bool(self.success, field_name="success")
        if not isinstance(self.status, ExpertStatus):
            raise TypeError("status must be an ExpertStatus")
        if self.success != (self.status is ExpertStatus.SUCCESS):
            raise ValueError("success and status are inconsistent")

        for field_name in ("attempts", "environment_steps", "planning_calls", "replans"):
            _require_int(getattr(self, field_name), field_name=field_name, minimum=0)
        if self.replans > self.planning_calls:
            raise ValueError("replans cannot exceed planning_calls")

        for field_name in ("planning_duration_seconds", "execution_duration_seconds"):
            normalized = _finite_float(
                getattr(self, field_name),
                field_name=field_name,
                positive=False,
            )
            object.__setattr__(self, field_name, normalized)
        if not isinstance(self.message, str):
            raise TypeError("message must be a string")
        _require_optional_string(self.planner_status, field_name="planner_status")

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh JSON-serializable phase mapping."""
        return {
            "phase": self.phase.value,
            "success": self.success,
            "status": self.status.value,
            "attempts": self.attempts,
            "environment_steps": self.environment_steps,
            "planning_calls": self.planning_calls,
            "replans": self.replans,
            "planning_duration_seconds": self.planning_duration_seconds,
            "execution_duration_seconds": self.execution_duration_seconds,
            "message": self.message,
            "planner_status": self.planner_status,
        }


@dataclass(frozen=True, slots=True)
class ExpertResult:
    """JSON-ready terminal result for one privileged expert rollout."""

    success: bool
    status: ExpertStatus
    scene_seed: int | None
    scene_id: str | None
    task_id: str | None
    canonical_instruction: str | None
    target_object_id: str | None
    target_bin_id: str | None
    total_environment_steps: int
    total_planning_calls: int
    total_replans: int
    completed_phases: tuple[ExpertPhase, ...]
    failed_phase: ExpertPhase | None
    final_environment_evaluation: Mapping[str, bool]
    phase_results: tuple[PhaseResult, ...]
    planning_duration_seconds: float
    execution_duration_seconds: float
    diagnostic_artifact_paths: tuple[str, ...] = ()
    exception_type: str | None = None
    exception_message: str | None = None

    def __post_init__(self) -> None:
        _require_bool(self.success, field_name="success")
        if not isinstance(self.status, ExpertStatus):
            raise TypeError("status must be an ExpertStatus")
        if self.success != (self.status is ExpertStatus.SUCCESS):
            raise ValueError("success and status are inconsistent")

        if self.scene_seed is not None:
            _require_int(self.scene_seed, field_name="scene_seed", minimum=0)
        for field_name in (
            "scene_id",
            "task_id",
            "canonical_instruction",
            "target_object_id",
            "target_bin_id",
        ):
            _require_optional_string(getattr(self, field_name), field_name=field_name)

        for field_name in (
            "total_environment_steps",
            "total_planning_calls",
            "total_replans",
        ):
            _require_int(getattr(self, field_name), field_name=field_name, minimum=0)
        if self.total_replans > self.total_planning_calls:
            raise ValueError("total_replans cannot exceed total_planning_calls")

        if not isinstance(self.completed_phases, tuple) or not all(
            isinstance(phase, ExpertPhase) for phase in self.completed_phases
        ):
            raise TypeError("completed_phases must be a tuple of ExpertPhase values")
        if self.completed_phases != EXPERT_PHASE_SEQUENCE[: len(self.completed_phases)]:
            raise ValueError("completed_phases must be a unique prefix of the phase sequence")
        if self.failed_phase is not None and not isinstance(self.failed_phase, ExpertPhase):
            raise TypeError("failed_phase must be None or an ExpertPhase")
        if self.failed_phase in self.completed_phases:
            raise ValueError("failed_phase cannot also be completed")

        if not isinstance(self.phase_results, tuple) or not all(
            isinstance(result, PhaseResult) for result in self.phase_results
        ):
            raise TypeError("phase_results must be a tuple of PhaseResult values")
        result_phases = tuple(result.phase for result in self.phase_results)
        if result_phases != EXPERT_PHASE_SEQUENCE[: len(result_phases)]:
            raise ValueError("phase_results must follow the phase sequence without gaps")
        successful_phases = tuple(result.phase for result in self.phase_results if result.success)
        if successful_phases != self.completed_phases:
            raise ValueError("completed_phases must match successful phase_results")
        failed_results = tuple(result for result in self.phase_results if not result.success)
        if len(failed_results) > 1 or (
            failed_results and failed_results[0] is not self.phase_results[-1]
        ):
            raise ValueError("at most the final phase_result may be a failure")
        if failed_results:
            if self.failed_phase is not failed_results[0].phase:
                raise ValueError("failed_phase must match the failed phase_result")
            if self.status is not failed_results[0].status:
                raise ValueError("result status must match the failed phase_result")

        if self.success:
            if self.completed_phases != EXPERT_PHASE_SEQUENCE:
                raise ValueError("a successful result must complete all expert phases")
            if self.failed_phase is not None:
                raise ValueError("a successful result cannot have failed_phase")
        elif self.failed_phase is not None and not failed_results:
            next_index = len(self.completed_phases)
            if next_index >= len(EXPERT_PHASE_SEQUENCE):
                raise ValueError("failed_phase cannot follow a complete phase sequence")
            if self.failed_phase is not EXPERT_PHASE_SEQUENCE[next_index]:
                raise ValueError("failed_phase must be the next uncompleted phase")

        if not isinstance(self.final_environment_evaluation, Mapping):
            raise TypeError("final_environment_evaluation must be a mapping")
        evaluation = dict(self.final_environment_evaluation)
        if not all(isinstance(key, str) and key for key in evaluation):
            raise TypeError("final_environment_evaluation keys must be non-empty strings")
        if not all(isinstance(value, bool) for value in evaluation.values()):
            raise TypeError("final_environment_evaluation values must be bools")
        if self.success and evaluation.get("success") is not True:
            raise ValueError("a successful result requires final evaluation success=True")
        object.__setattr__(self, "final_environment_evaluation", MappingProxyType(evaluation))

        if self.total_environment_steps < sum(
            result.environment_steps for result in self.phase_results
        ):
            raise ValueError("total_environment_steps is smaller than the per-phase total")
        if self.total_planning_calls < sum(result.planning_calls for result in self.phase_results):
            raise ValueError("total_planning_calls is smaller than the per-phase total")
        if self.total_replans < sum(result.replans for result in self.phase_results):
            raise ValueError("total_replans is smaller than the per-phase total")

        for field_name in ("planning_duration_seconds", "execution_duration_seconds"):
            normalized = _finite_float(
                getattr(self, field_name),
                field_name=field_name,
                positive=False,
            )
            object.__setattr__(self, field_name, normalized)

        if not isinstance(self.diagnostic_artifact_paths, tuple) or not all(
            isinstance(path, str) and path for path in self.diagnostic_artifact_paths
        ):
            raise TypeError("diagnostic_artifact_paths must be a tuple of non-empty strings")

        _require_optional_string(self.exception_type, field_name="exception_type")
        _require_optional_string(self.exception_message, field_name="exception_message")
        has_exception = self.exception_type is not None or self.exception_message is not None
        if has_exception and (self.exception_type is None or self.exception_message is None):
            raise ValueError("exception_type and exception_message must be provided together")
        if self.status is ExpertStatus.UNEXPECTED_EXCEPTION and not has_exception:
            raise ValueError("unexpected_exception must preserve exception type and message")
        if self.status is not ExpertStatus.UNEXPECTED_EXCEPTION and has_exception:
            raise ValueError("exception details are reserved for unexpected_exception")

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh JSON-serializable result mapping."""
        return {
            "success": self.success,
            "status": self.status.value,
            "scene_seed": self.scene_seed,
            "scene_id": self.scene_id,
            "task_id": self.task_id,
            "canonical_instruction": self.canonical_instruction,
            "target_object_id": self.target_object_id,
            "target_bin_id": self.target_bin_id,
            "total_environment_steps": self.total_environment_steps,
            "total_planning_calls": self.total_planning_calls,
            "total_replans": self.total_replans,
            "completed_phases": [phase.value for phase in self.completed_phases],
            "failed_phase": self.failed_phase.value if self.failed_phase is not None else None,
            "final_environment_evaluation": dict(self.final_environment_evaluation),
            "phase_results": [result.to_dict() for result in self.phase_results],
            "planning_duration_seconds": self.planning_duration_seconds,
            "execution_duration_seconds": self.execution_duration_seconds,
            "diagnostic_artifact_paths": list(self.diagnostic_artifact_paths),
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
        }
