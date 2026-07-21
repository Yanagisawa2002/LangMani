from __future__ import annotations

import numpy as np

from langmani.experts.push_diagnostics import (
    PushDiagnosticSnapshot,
    classify_push_failure,
    count_contact_losses,
    planar_diagnostics,
)


def _snapshot(
    phase: str,
    *,
    progress: float = 0.0,
    lateral: float = 0.0,
    contact: bool = False,
) -> PushDiagnosticSnapshot:
    return PushDiagnosticSnapshot(
        phase=phase,
        environment_steps=0,
        target_object_pose=(0.0, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0),
        object_poses={"blue_cube": (0.0, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0)},
        target_center=(0.2, 0.0, 0.001),
        tcp_pose=(-0.045, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0),
        intended_push_direction=(1.0, 0.0),
        chosen_precontact_point=(-0.12, 0.0, 0.12),
        chosen_contact_point=(-0.045, 0.0, 0.025),
        target_distance=0.2 - progress,
        projected_progress=progress,
        lateral_error=lateral,
        contact_proxy=contact,
        contact_proxy_distance=0.045 if contact else 0.12,
        workspace_margin=0.2,
        target_inside_region=False,
        target_is_static=True,
        stable_success_steps=0,
    )


def test_push_phase_diagnostics_measure_progress_and_contact_loss() -> None:
    distance, progress, lateral = planar_diagnostics(
        initial_position=np.array([0.0, 0.0, 0.025]),
        current_position=np.array([0.1, 0.02, 0.025]),
        target_center=np.array([0.2, 0.0, 0.001]),
        push_direction=np.array([1.0, 0.0]),
    )
    trace = (_snapshot("establish_contact", contact=True), _snapshot("primary_push"))

    assert distance > 0.1
    assert progress == 0.1
    assert lateral == 0.02
    assert count_contact_losses(trace) == 1


def test_push_failure_classifier_prefers_preceding_causes_over_timeout() -> None:
    trace = (
        _snapshot("primary_push", progress=0.03, contact=True),
        _snapshot("corrective_push_1", progress=0.031, contact=True),
    )

    cause = classify_push_failure(
        status="timeout",
        failed_phase="corrective_push_1",
        evaluation={"target_inside_region": False},
        trace=trace,
    )

    assert cause == "correction_ineffective"


def test_push_failure_classifier_separates_boundary_and_origin_failures() -> None:
    boundary = classify_push_failure(
        status="verification_failure",
        failed_phase="verify_task",
        evaluation={
            "target_inside_region": True,
            "target_is_static": True,
            "stable_success_steps": 3,
        },
        trace=(_snapshot("verify_task"),),
    )
    action = classify_push_failure(
        status="execution_failure",
        failed_phase="move_to_precontact",
        evaluation={"action_out_of_bounds": True},
        trace=(_snapshot("move_to_precontact"),),
    )

    assert boundary == "verification_boundary_case"
    assert action == "action_bound_violation"
