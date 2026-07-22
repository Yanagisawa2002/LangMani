"""Immutable contracts for the Phase 2B.3 geometry-constrained Push expert."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from langmani.experts.push_types import PushExpertStatus


class GeometryPushState(StrEnum):
    """Explicit controller states authorized for Phase 2B.3."""

    PLAN_PUSH = "plan_push"
    MOVE_TO_STAGING = "move_to_staging"
    MOVE_TO_PRECONTACT = "move_to_precontact"
    ESTABLISH_CONTACT = "establish_contact"
    PUSH_CLOSED_LOOP = "push_closed_loop"
    VERIFY_PROGRESS = "verify_progress"
    RECOVER_CONTACT = "recover_contact"
    REPLAN = "replan"
    SUCCESS = "success"
    SAFE_ABORT = "safe_abort"


ALLOWED_GEOMETRY_PUSH_TRANSITIONS: Mapping[GeometryPushState, frozenset[GeometryPushState]] = (
    MappingProxyType(
        {
            GeometryPushState.PLAN_PUSH: frozenset(
                {GeometryPushState.MOVE_TO_STAGING, GeometryPushState.SAFE_ABORT}
            ),
            GeometryPushState.MOVE_TO_STAGING: frozenset(
                {
                    GeometryPushState.MOVE_TO_PRECONTACT,
                    GeometryPushState.REPLAN,
                    GeometryPushState.SAFE_ABORT,
                }
            ),
            GeometryPushState.MOVE_TO_PRECONTACT: frozenset(
                {
                    GeometryPushState.ESTABLISH_CONTACT,
                    GeometryPushState.REPLAN,
                    GeometryPushState.SAFE_ABORT,
                }
            ),
            GeometryPushState.ESTABLISH_CONTACT: frozenset(
                {
                    GeometryPushState.PUSH_CLOSED_LOOP,
                    GeometryPushState.RECOVER_CONTACT,
                    GeometryPushState.REPLAN,
                    GeometryPushState.SAFE_ABORT,
                }
            ),
            GeometryPushState.PUSH_CLOSED_LOOP: frozenset(
                {
                    GeometryPushState.VERIFY_PROGRESS,
                    GeometryPushState.RECOVER_CONTACT,
                    GeometryPushState.REPLAN,
                    GeometryPushState.SAFE_ABORT,
                }
            ),
            GeometryPushState.VERIFY_PROGRESS: frozenset(
                {
                    GeometryPushState.PUSH_CLOSED_LOOP,
                    GeometryPushState.RECOVER_CONTACT,
                    GeometryPushState.REPLAN,
                    GeometryPushState.SUCCESS,
                    GeometryPushState.SAFE_ABORT,
                }
            ),
            GeometryPushState.RECOVER_CONTACT: frozenset(
                {
                    GeometryPushState.MOVE_TO_STAGING,
                    GeometryPushState.REPLAN,
                    GeometryPushState.SAFE_ABORT,
                }
            ),
            GeometryPushState.REPLAN: frozenset(
                {GeometryPushState.MOVE_TO_STAGING, GeometryPushState.SAFE_ABORT}
            ),
            GeometryPushState.SUCCESS: frozenset(),
            GeometryPushState.SAFE_ABORT: frozenset(),
        }
    )
)


STATE_MACHINE_VERSION = "geometry_constrained_closed_loop_push_v1"


@dataclass(frozen=True, slots=True)
class GeometryPushStateContract:
    """Reviewable entry, control, progress, timeout, and failure contract for one state."""

    entry_condition: str
    control_target: str
    progress_condition: str
    timeout_rule: str
    failure_reason: str
    allowed_next: frozenset[GeometryPushState]


GEOMETRY_PUSH_STATE_CONTRACTS: Mapping[GeometryPushState, GeometryPushStateContract] = (
    MappingProxyType(
        {
            GeometryPushState.PLAN_PUSH: GeometryPushStateContract(
                "fresh reset and compatible native contracts",
                "support-aware complete push plan",
                "geometry and native planner initialized",
                "one visit",
                "invalid task, geometry, feedback, or planner contract",
                ALLOWED_GEOMETRY_PUSH_TRANSITIONS[GeometryPushState.PLAN_PUSH],
            ),
            GeometryPushState.MOVE_TO_STAGING: GeometryPushStateContract(
                "accepted complete geometry plan",
                "workspace-safe elevated staging pose",
                "TCP reaches staging tolerance",
                "four visits across all recoveries and replans",
                "native motion planning or execution failure",
                ALLOWED_GEOMETRY_PUSH_TRANSITIONS[GeometryPushState.MOVE_TO_STAGING],
            ),
            GeometryPushState.MOVE_TO_PRECONTACT: GeometryPushStateContract(
                "staging pose reached",
                "support-aware precontact pose",
                "TCP reaches precontact tolerance",
                "four visits across all recoveries and replans",
                "native motion planning or execution failure",
                ALLOWED_GEOMETRY_PUSH_TRANSITIONS[GeometryPushState.MOVE_TO_PRECONTACT],
            ),
            GeometryPushState.ESTABLISH_CONTACT: GeometryPushStateContract(
                "precontact pose reached",
                "shape-aware contact point and push height",
                "contact pose reached without a latched safety event",
                "four visits across all recoveries and replans",
                "unreachable contact or unsafe execution",
                ALLOWED_GEOMETRY_PUSH_TRANSITIONS[GeometryPushState.ESTABLISH_CONTACT],
            ),
            GeometryPushState.PUSH_CLOSED_LOOP: GeometryPushStateContract(
                "contact pose or verified prior segment",
                "next bounded segment from current object state",
                "one short segment executes or native containment is reached",
                "twelve total segments",
                "planning, execution, or safety-event failure",
                ALLOWED_GEOMETRY_PUSH_TRANSITIONS[GeometryPushState.PUSH_CLOSED_LOOP],
            ),
            GeometryPushState.VERIFY_PROGRESS: GeometryPushStateContract(
                "one bounded segment completed",
                "no action; inspect native state, contact, drift, and progress",
                "native success or sufficient distance improvement",
                "twelve verifications",
                "contact loss, rollout, drift, stagnation, or unresolved task",
                ALLOWED_GEOMETRY_PUSH_TRANSITIONS[GeometryPushState.VERIFY_PROGRESS],
            ),
            GeometryPushState.RECOVER_CONTACT: GeometryPushStateContract(
                "observed contact loss, rollout, or repeated stagnation",
                "bounded retreat followed by a fresh geometry plan",
                "retreat executes and complete plan revalidates",
                "two total contact recoveries",
                "recovery limit, planning failure, or safety event",
                ALLOWED_GEOMETRY_PUSH_TRANSITIONS[GeometryPushState.RECOVER_CONTACT],
            ),
            GeometryPushState.REPLAN: GeometryPushStateContract(
                "recoverable planner rejection or excessive lateral drift",
                "fresh plan from current simulator state",
                "complete support and workspace gates pass",
                "three total replans",
                "replan limit or infeasible geometry",
                ALLOWED_GEOMETRY_PUSH_TRANSITIONS[GeometryPushState.REPLAN],
            ),
            GeometryPushState.SUCCESS: GeometryPushStateContract(
                "native stable success is true",
                "none",
                "terminal",
                "one terminal visit",
                "not applicable",
                ALLOWED_GEOMETRY_PUSH_TRANSITIONS[GeometryPushState.SUCCESS],
            ),
            GeometryPushState.SAFE_ABORT: GeometryPushStateContract(
                "bounded failure or safety gate fired",
                "none",
                "terminal evidence preserved",
                "one terminal visit",
                "classified terminal status",
                ALLOWED_GEOMETRY_PUSH_TRANSITIONS[GeometryPushState.SAFE_ABORT],
            ),
        }
    )
)


@dataclass(frozen=True, slots=True)
class GeometryPushExpertConfig:
    """Frozen semantic configuration for the one authorized expert rebuild."""

    maximum_episode_steps: int = 250
    gripper_close_steps: int = 6
    maximum_contact_recoveries: int = 2
    maximum_replans: int = 3
    maximum_state_visits: int = 48
    maximum_push_segments: int = 12
    maximum_segment_distance: float = 0.100
    minimum_segment_distance: float = 0.012
    minimum_progress: float = 0.003
    maximum_stagnant_verifications: int = 2
    maximum_lateral_error: float = 0.045
    maximum_cylinder_angular_speed: float = 8.0
    gripper_contact_padding: float = 0.020
    precontact_clearance: float = 0.060
    precontact_height: float = 0.085
    staging_height: float = 0.180
    cube_push_height: float = 0.025
    cylinder_push_height: float = 0.025
    goal_margin: float = 0.012
    near_goal_step_scale: float = 0.60
    object_clearance: float = 0.002
    tcp_clearance: float = 0.015
    tracking_margin: float = 0.002
    contact_force_threshold: float = 0.05
    contact_distance_tolerance: float = 0.035
    retreat_distance: float = 0.045
    settle_steps: int = 8
    free_space_action_stride: int = 2

    def __post_init__(self) -> None:
        integer_fields = (
            "maximum_episode_steps",
            "gripper_close_steps",
            "maximum_state_visits",
            "maximum_push_segments",
            "maximum_stagnant_verifications",
            "settle_steps",
            "free_space_action_stride",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("maximum_contact_recoveries", "maximum_replans"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name, value in asdict(self).items():
            if name in integer_fields or name in {"maximum_contact_recoveries", "maximum_replans"}:
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.minimum_segment_distance > self.maximum_segment_distance:
            raise ValueError("minimum_segment_distance cannot exceed maximum_segment_distance")
        if not 0.0 < self.near_goal_step_scale <= 1.0:
            raise ValueError("near_goal_step_scale must be in (0, 1]")
        if self.staging_height < self.precontact_height:
            raise ValueError("staging_height must not be below precontact_height")

    def to_dict(self) -> dict[str, int | float | str]:
        """Return all semantic fields including the state-machine identity."""

        return {"state_machine_version": STATE_MACHINE_VERSION, **asdict(self)}


@dataclass(frozen=True, slots=True)
class GeometryPushTransition:
    """One checked state transition and its observable reason."""

    source: GeometryPushState
    target: GeometryPushState
    reason: str
    environment_steps: int
    target_distance: float | None
    contact_recoveries: int
    replans: int

    def __post_init__(self) -> None:
        if self.target not in ALLOWED_GEOMETRY_PUSH_TRANSITIONS[self.source]:
            raise ValueError(f"illegal geometry Push transition {self.source} -> {self.target}")
        if not self.reason:
            raise ValueError("transition reason cannot be empty")
        if min(self.environment_steps, self.contact_recoveries, self.replans) < 0:
            raise ValueError("transition counters must be non-negative")
        if self.target_distance is not None and (
            not math.isfinite(self.target_distance) or self.target_distance < 0.0
        ):
            raise ValueError("target_distance must be finite and non-negative")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible transition."""

        return {
            "source": self.source.value,
            "target": self.target.value,
            "reason": self.reason,
            "environment_steps": self.environment_steps,
            "target_distance": self.target_distance,
            "contact_recoveries": self.contact_recoveries,
            "replans": self.replans,
        }


@dataclass(frozen=True, slots=True)
class GeometryPushExpertResult:
    """Terminal result with the complete explicit state trace."""

    success: bool
    status: PushExpertStatus
    scene_seed: int | None
    task: Mapping[str, str]
    total_environment_steps: int
    total_planning_calls: int
    contact_recoveries: int
    replans: int
    push_segments: int
    transitions: tuple[GeometryPushTransition, ...]
    final_environment_evaluation: Mapping[str, bool | int | float]
    geometry_plans: tuple[Mapping[str, object], ...]
    exception_type: str | None = None
    exception_message: str | None = None

    def __post_init__(self) -> None:
        if self.success != (self.status is PushExpertStatus.SUCCESS):
            raise ValueError("geometry Push result success/status are inconsistent")
        for value in (
            self.total_environment_steps,
            self.total_planning_calls,
            self.contact_recoveries,
            self.replans,
            self.push_segments,
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("result counters must be non-negative integers")
        task = dict(self.task)
        evaluation = dict(self.final_environment_evaluation)
        plans = tuple(MappingProxyType(dict(plan)) for plan in self.geometry_plans)
        object.__setattr__(self, "task", MappingProxyType(task))
        object.__setattr__(self, "final_environment_evaluation", MappingProxyType(evaluation))
        object.__setattr__(self, "geometry_plans", plans)
        exceptional = self.exception_type is not None or self.exception_message is not None
        if exceptional != (self.status is PushExpertStatus.UNEXPECTED_EXCEPTION):
            raise ValueError("exception details are reserved for unexpected_exception")

    def to_dict(self) -> dict[str, Any]:
        """Return a complete JSON-compatible result."""

        return {
            "success": self.success,
            "status": self.status.value,
            "scene_seed": self.scene_seed,
            "task": dict(self.task),
            "total_environment_steps": self.total_environment_steps,
            "total_planning_calls": self.total_planning_calls,
            "contact_recoveries": self.contact_recoveries,
            "replans": self.replans,
            "push_segments": self.push_segments,
            "transitions": [value.to_dict() for value in self.transitions],
            "final_environment_evaluation": dict(self.final_environment_evaluation),
            "geometry_plans": [dict(value) for value in self.geometry_plans],
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
        }


__all__ = [
    "ALLOWED_GEOMETRY_PUSH_TRANSITIONS",
    "GEOMETRY_PUSH_STATE_CONTRACTS",
    "GeometryPushExpertConfig",
    "GeometryPushExpertResult",
    "GeometryPushState",
    "GeometryPushStateContract",
    "GeometryPushTransition",
    "STATE_MACHINE_VERSION",
]
