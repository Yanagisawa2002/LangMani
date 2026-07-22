"""Immutable execution contracts for the Phase 2 planar-pushing expert."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any


class PushExpertStatus(StrEnum):
    SUCCESS = "success"
    INVALID_TASK = "invalid_task"
    INITIALIZATION_FAILURE = "initialization_failure"
    IK_FAILURE = "ik_failure"
    PLANNING_FAILURE = "planning_failure"
    EXECUTION_FAILURE = "execution_failure"
    CONTACT_FAILURE = "contact_failure"
    PUSH_FAILURE = "push_failure"
    CORRECTION_FAILURE = "correction_failure"
    VERIFICATION_FAILURE = "verification_failure"
    TARGET_OUTSIDE_WORKSPACE = "target_outside_workspace"
    TARGET_LIFTED = "target_lifted"
    TARGET_TOPPLED = "target_toppled"
    WRONG_OBJECT_INTERACTION = "wrong_object_interaction"
    TIMEOUT = "timeout"
    UNEXPECTED_EXCEPTION = "unexpected_exception"


class PushExpertPhase(StrEnum):
    INITIALIZE = "initialize"
    CLOSE_GRIPPER = "close_gripper"
    MOVE_TO_PRECONTACT = "move_to_precontact"
    ESTABLISH_CONTACT = "establish_contact"
    PRIMARY_PUSH = "primary_push"
    CORRECTIVE_PUSH_1 = "corrective_push_1"
    CORRECTIVE_PUSH_2 = "corrective_push_2"
    SETTLE = "settle"
    VERIFY_TASK = "verify_task"


PUSH_EXPERT_PHASE_SEQUENCE: tuple[PushExpertPhase, ...] = tuple(PushExpertPhase)


def _positive_int(value: object, label: str, *, allow_zero: bool = False) -> None:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")


def _finite(value: object, label: str, *, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0) or (not positive and result < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be finite and {qualifier}")
    return result


@dataclass(frozen=True, slots=True)
class PushExpertConfig:
    control_mode: str = "pd_joint_pos"
    maximum_episode_steps: int = 250
    maximum_planning_attempts_per_phase: int = 1
    gripper_close_steps: int = 6
    precontact_clearance: float = 0.075
    contact_offset: float = 0.045
    precontact_height: float = 0.12
    precontact_staging_height: float = 0.32
    push_height: float = 0.025
    cylinder_push_height: float = 0.015
    region_goal_margin: float = 0.035
    cylinder_region_goal_margin: float = 0.035
    primary_push_increment: float = 0.08
    minimum_primary_push_increment: float = 0.02
    maximum_primary_push_segments: int = 6
    free_space_action_stride: int = 2
    settle_steps: int = 16
    maximum_corrective_pushes: int = 2
    tcp_position_tolerance: float = 0.025
    lateral_cube_compensation_degrees: float = 15.0
    lateral_cylinder_compensation_degrees: float = 0.0
    minimum_approach_obstacle_clearance: float = 0.015
    diagnostic_rendering: bool = False

    def __post_init__(self) -> None:
        if self.control_mode != "pd_joint_pos":
            raise ValueError("push expert requires control_mode='pd_joint_pos'")
        for label in (
            "maximum_episode_steps",
            "maximum_planning_attempts_per_phase",
            "gripper_close_steps",
            "maximum_primary_push_segments",
            "free_space_action_stride",
            "settle_steps",
        ):
            _positive_int(getattr(self, label), label)
        _positive_int(
            self.maximum_corrective_pushes,
            "maximum_corrective_pushes",
            allow_zero=True,
        )
        if self.maximum_corrective_pushes != 2:
            raise ValueError("Phase 2 push expert fixes exactly two bounded corrective pushes")
        for label in (
            "precontact_clearance",
            "contact_offset",
            "precontact_height",
            "precontact_staging_height",
            "push_height",
            "cylinder_push_height",
            "region_goal_margin",
            "cylinder_region_goal_margin",
            "primary_push_increment",
            "minimum_primary_push_increment",
            "tcp_position_tolerance",
            "minimum_approach_obstacle_clearance",
        ):
            object.__setattr__(self, label, _finite(getattr(self, label), label))
        if self.precontact_staging_height <= self.precontact_height:
            raise ValueError("precontact_staging_height must exceed precontact_height")
        if self.minimum_primary_push_increment > self.primary_push_increment:
            raise ValueError(
                "minimum_primary_push_increment must not exceed primary_push_increment"
            )
        for label in (
            "lateral_cube_compensation_degrees",
            "lateral_cylinder_compensation_degrees",
        ):
            object.__setattr__(
                self,
                label,
                _finite(getattr(self, label), label, positive=False),
            )
            if getattr(self, label) > 25.0:
                raise ValueError(f"{label} must not exceed 25")
        if not isinstance(self.diagnostic_rendering, bool):
            raise TypeError("diagnostic_rendering must be a bool")

    def to_dict(self) -> dict[str, object]:
        return {
            "control_mode": self.control_mode,
            "maximum_episode_steps": self.maximum_episode_steps,
            "maximum_planning_attempts_per_phase": self.maximum_planning_attempts_per_phase,
            "gripper_close_steps": self.gripper_close_steps,
            "precontact_clearance": self.precontact_clearance,
            "contact_offset": self.contact_offset,
            "precontact_height": self.precontact_height,
            "precontact_staging_height": self.precontact_staging_height,
            "push_height": self.push_height,
            "cylinder_push_height": self.cylinder_push_height,
            "region_goal_margin": self.region_goal_margin,
            "cylinder_region_goal_margin": self.cylinder_region_goal_margin,
            "primary_push_increment": self.primary_push_increment,
            "minimum_primary_push_increment": self.minimum_primary_push_increment,
            "maximum_primary_push_segments": self.maximum_primary_push_segments,
            "free_space_action_stride": self.free_space_action_stride,
            "settle_steps": self.settle_steps,
            "maximum_corrective_pushes": self.maximum_corrective_pushes,
            "tcp_position_tolerance": self.tcp_position_tolerance,
            "lateral_cube_compensation_degrees": self.lateral_cube_compensation_degrees,
            "lateral_cylinder_compensation_degrees": self.lateral_cylinder_compensation_degrees,
            "minimum_approach_obstacle_clearance": self.minimum_approach_obstacle_clearance,
            "diagnostic_rendering": self.diagnostic_rendering,
        }


@dataclass(frozen=True, slots=True)
class PushPhaseResult:
    phase: PushExpertPhase
    success: bool
    status: PushExpertStatus
    attempts: int
    environment_steps: int
    planning_calls: int
    planning_duration_seconds: float
    execution_duration_seconds: float
    message: str
    planner_status: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.phase, PushExpertPhase):
            raise TypeError("phase must be PushExpertPhase")
        if not isinstance(self.success, bool) or self.success != (
            self.status is PushExpertStatus.SUCCESS
        ):
            raise ValueError("push phase success/status are inconsistent")
        for label in ("attempts", "environment_steps", "planning_calls"):
            _positive_int(getattr(self, label), label, allow_zero=True)
        for label in ("planning_duration_seconds", "execution_duration_seconds"):
            object.__setattr__(self, label, _finite(getattr(self, label), label, positive=False))
        if not isinstance(self.message, str):
            raise TypeError("message must be a string")
        if self.planner_status is not None and not isinstance(self.planner_status, str):
            raise TypeError("planner_status must be None or a string")

    def to_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase.value,
            "success": self.success,
            "status": self.status.value,
            "attempts": self.attempts,
            "environment_steps": self.environment_steps,
            "planning_calls": self.planning_calls,
            "planning_duration_seconds": self.planning_duration_seconds,
            "execution_duration_seconds": self.execution_duration_seconds,
            "message": self.message,
            "planner_status": self.planner_status,
        }


@dataclass(frozen=True, slots=True)
class PushExpertResult:
    success: bool
    status: PushExpertStatus
    scene_seed: int | None
    scene_id: str | None
    task_id: str | None
    canonical_instruction: str | None
    target_object_id: str | None
    target_region_id: str | None
    difficulty: str | None
    total_environment_steps: int
    total_planning_calls: int
    completed_phases: tuple[PushExpertPhase, ...]
    failed_phase: PushExpertPhase | None
    final_environment_evaluation: Mapping[str, bool | int | float]
    phase_results: tuple[PushPhaseResult, ...]
    planning_duration_seconds: float
    execution_duration_seconds: float
    exception_type: str | None = None
    exception_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool) or self.success != (
            self.status is PushExpertStatus.SUCCESS
        ):
            raise ValueError("push expert success/status are inconsistent")
        _positive_int(self.total_environment_steps, "total_environment_steps", allow_zero=True)
        _positive_int(self.total_planning_calls, "total_planning_calls", allow_zero=True)
        phases = tuple(result.phase for result in self.phase_results)
        if phases != PUSH_EXPERT_PHASE_SEQUENCE[: len(phases)]:
            raise ValueError("push phase results must be a sequence prefix")
        completed = tuple(result.phase for result in self.phase_results if result.success)
        if completed != self.completed_phases:
            raise ValueError("completed_phases disagree with phase results")
        failures = tuple(result for result in self.phase_results if not result.success)
        if len(failures) > 1 or (failures and failures[-1] is not self.phase_results[-1]):
            raise ValueError("only the final push phase may fail")
        if failures and self.failed_phase is not failures[0].phase:
            raise ValueError("failed_phase disagrees with the failed phase result")
        if self.success and completed != PUSH_EXPERT_PHASE_SEQUENCE:
            raise ValueError("successful push expert must complete every phase")
        evaluation = dict(self.final_environment_evaluation)
        if not all(isinstance(key, str) and key for key in evaluation):
            raise TypeError("final evaluation keys must be strings")
        if not all(
            isinstance(value, (bool, int, float)) and not isinstance(value, complex)
            for value in evaluation.values()
        ):
            raise TypeError("final evaluation values must be JSON scalars")
        object.__setattr__(self, "final_environment_evaluation", MappingProxyType(evaluation))
        for label in ("planning_duration_seconds", "execution_duration_seconds"):
            object.__setattr__(self, label, _finite(getattr(self, label), label, positive=False))
        exception = self.exception_type is not None or self.exception_message is not None
        if exception != (self.status is PushExpertStatus.UNEXPECTED_EXCEPTION):
            raise ValueError("exception details are reserved for unexpected_exception")
        if exception and not (
            isinstance(self.exception_type, str) and isinstance(self.exception_message, str)
        ):
            raise TypeError("exception type and message must be strings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status.value,
            "scene_seed": self.scene_seed,
            "scene_id": self.scene_id,
            "task_id": self.task_id,
            "canonical_instruction": self.canonical_instruction,
            "target_object_id": self.target_object_id,
            "target_region_id": self.target_region_id,
            "difficulty": self.difficulty,
            "total_environment_steps": self.total_environment_steps,
            "total_planning_calls": self.total_planning_calls,
            "completed_phases": [phase.value for phase in self.completed_phases],
            "failed_phase": self.failed_phase.value if self.failed_phase is not None else None,
            "final_environment_evaluation": dict(self.final_environment_evaluation),
            "phase_results": [result.to_dict() for result in self.phase_results],
            "planning_duration_seconds": self.planning_duration_seconds,
            "execution_duration_seconds": self.execution_duration_seconds,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
        }


__all__ = [
    "PUSH_EXPERT_PHASE_SEQUENCE",
    "PushExpertConfig",
    "PushExpertPhase",
    "PushExpertResult",
    "PushExpertStatus",
    "PushPhaseResult",
]
