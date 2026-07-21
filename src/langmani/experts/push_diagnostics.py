"""Typed Phase 2A side-push diagnostics and bounded root-cause classification."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

type PushFailureRootCause = Literal[
    "approach_unreachable",
    "bad_contact_side",
    "premature_contact",
    "contact_lost",
    "insufficient_push_distance",
    "lateral_drift",
    "cylinder_roll_or_rotation",
    "overshoot",
    "correction_ineffective",
    "workspace_margin_violation",
    "planner_failure",
    "action_bound_violation",
    "verification_boundary_case",
    "other",
]

PUSH_FAILURE_ROOT_CAUSES: tuple[PushFailureRootCause, ...] = (
    "approach_unreachable",
    "bad_contact_side",
    "premature_contact",
    "contact_lost",
    "insufficient_push_distance",
    "lateral_drift",
    "cylinder_roll_or_rotation",
    "overshoot",
    "correction_ineffective",
    "workspace_margin_violation",
    "planner_failure",
    "action_bound_violation",
    "verification_boundary_case",
    "other",
)


def _finite_tuple(value: Sequence[float], *, length: int, label: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value)
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain {length} finite values")
    return result


@dataclass(frozen=True, slots=True)
class PushDiagnosticSnapshot:
    """Small immutable phase-boundary snapshot; never a per-step policy observation."""

    phase: str
    environment_steps: int
    target_object_pose: tuple[float, ...]
    object_poses: Mapping[str, tuple[float, ...]]
    target_center: tuple[float, ...]
    tcp_pose: tuple[float, ...]
    intended_push_direction: tuple[float, ...]
    chosen_precontact_point: tuple[float, ...] | None
    chosen_contact_point: tuple[float, ...] | None
    target_distance: float
    projected_progress: float
    lateral_error: float
    contact_proxy: bool
    contact_proxy_distance: float
    workspace_margin: float
    target_inside_region: bool
    target_is_static: bool
    stable_success_steps: int

    def __post_init__(self) -> None:
        if not self.phase:
            raise ValueError("diagnostic phase must be non-empty")
        if self.environment_steps < 0:
            raise ValueError("environment_steps must be non-negative")
        object.__setattr__(
            self,
            "target_object_pose",
            _finite_tuple(self.target_object_pose, length=7, label="target_object_pose"),
        )
        poses = {
            str(key): _finite_tuple(value, length=7, label=f"object_poses[{key!r}]")
            for key, value in self.object_poses.items()
        }
        object.__setattr__(self, "object_poses", MappingProxyType(poses))
        object.__setattr__(
            self,
            "target_center",
            _finite_tuple(self.target_center, length=3, label="target_center"),
        )
        object.__setattr__(
            self, "tcp_pose", _finite_tuple(self.tcp_pose, length=7, label="tcp_pose")
        )
        object.__setattr__(
            self,
            "intended_push_direction",
            _finite_tuple(
                self.intended_push_direction,
                length=2,
                label="intended_push_direction",
            ),
        )
        for name in ("chosen_precontact_point", "chosen_contact_point"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite_tuple(value, length=3, label=name))
        for name in (
            "target_distance",
            "projected_progress",
            "lateral_error",
            "contact_proxy_distance",
            "workspace_margin",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.stable_success_steps < 0:
            raise ValueError("stable_success_steps must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "environment_steps": self.environment_steps,
            "target_object_pose": list(self.target_object_pose),
            "object_poses": {key: list(value) for key, value in self.object_poses.items()},
            "target_center": list(self.target_center),
            "tcp_pose": list(self.tcp_pose),
            "intended_push_direction": list(self.intended_push_direction),
            "chosen_precontact_point": (
                list(self.chosen_precontact_point)
                if self.chosen_precontact_point is not None
                else None
            ),
            "chosen_contact_point": (
                list(self.chosen_contact_point) if self.chosen_contact_point is not None else None
            ),
            "target_distance": self.target_distance,
            "projected_progress": self.projected_progress,
            "lateral_error": self.lateral_error,
            "contact_proxy": self.contact_proxy,
            "contact_proxy_distance": self.contact_proxy_distance,
            "workspace_margin": self.workspace_margin,
            "target_inside_region": self.target_inside_region,
            "target_is_static": self.target_is_static,
            "stable_success_steps": self.stable_success_steps,
        }


def count_contact_losses(trace: Sequence[PushDiagnosticSnapshot]) -> int:
    """Count observable phase-boundary contact-to-no-contact transitions."""

    relevant = tuple(item.contact_proxy for item in trace if item.phase != "initialize")
    return sum(before and not after for before, after in zip(relevant, relevant[1:], strict=False))


def classify_push_failure(
    *,
    status: str,
    failed_phase: str | None,
    evaluation: Mapping[str, bool | int | float],
    trace: Sequence[PushDiagnosticSnapshot],
) -> PushFailureRootCause:
    """Assign one deterministic preceding cause instead of returning generic timeout."""

    if evaluation.get("action_out_of_bounds") is True or evaluation.get("invalid_action") is True:
        return "action_bound_violation"
    if evaluation.get("target_outside_workspace") is True:
        return "workspace_margin_violation"
    if evaluation.get("wrong_object_displaced") is True:
        return "premature_contact"
    if evaluation.get("target_overshoot") is True:
        return "overshoot"
    if status in {"planning_failure", "ik_failure"}:
        if failed_phase == "move_to_precontact":
            return "approach_unreachable"
        return "planner_failure"
    if (
        evaluation.get("target_inside_region") is True
        and evaluation.get("target_is_static") is True
        and int(evaluation.get("stable_success_steps", 0)) < 5
    ):
        return "verification_boundary_case"
    losses = count_contact_losses(trace)
    if losses:
        return "contact_lost"
    primary = next((item for item in trace if item.phase == "primary_push"), None)
    corrections = tuple(item for item in trace if item.phase.startswith("corrective_push"))
    if corrections:
        before = primary.projected_progress if primary is not None else 0.0
        if corrections[-1].projected_progress - before < 0.01:
            return "correction_ineffective"
    if primary is not None and primary.projected_progress < 0.02:
        return "insufficient_push_distance"
    if trace and trace[-1].lateral_error > 0.08:
        return "lateral_drift"
    return "other"


def root_cause_counts(causes: Sequence[PushFailureRootCause]) -> dict[str, int]:
    return dict(sorted(Counter(causes).items()))


def planar_diagnostics(
    *,
    initial_position: np.ndarray,
    current_position: np.ndarray,
    target_center: np.ndarray,
    push_direction: np.ndarray,
) -> tuple[float, float, float]:
    """Return target distance, progress along the initial line, and lateral error."""

    displacement = np.asarray(current_position[:2] - initial_position[:2], dtype=np.float64)
    direction = np.asarray(push_direction[:2], dtype=np.float64)
    projected = float(np.dot(displacement, direction))
    lateral = float(np.linalg.norm(displacement - direction * projected))
    distance = float(np.linalg.norm(target_center[:2] - current_position[:2]))
    return distance, projected, lateral


__all__ = [
    "PUSH_FAILURE_ROOT_CAUSES",
    "PushDiagnosticSnapshot",
    "PushFailureRootCause",
    "classify_push_failure",
    "count_contact_losses",
    "planar_diagnostics",
    "root_cause_counts",
]
