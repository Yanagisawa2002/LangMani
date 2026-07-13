"""Unit tests for the compact, JSON-ready M2 expert contracts."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest
import torch

from langmani.experts.types import (
    EXPERT_PHASE_SEQUENCE,
    ExpertConfig,
    ExpertPhase,
    ExpertResult,
    ExpertStatus,
    PhaseResult,
)


def _evaluation(*, success: bool) -> dict[str, bool]:
    return {
        "target_in_target_bin": success,
        "target_in_wrong_bin": False,
        "wrong_object_in_target_bin": False,
        "target_is_grasped": False,
        "target_is_static": success,
        "target_off_table": False,
        "success": success,
        "fail": False,
    }


def _successful_phase(phase: ExpertPhase) -> PhaseResult:
    return PhaseResult(
        phase=phase,
        success=True,
        status=ExpertStatus.SUCCESS,
        attempts=1,
        environment_steps=1,
        planning_calls=0,
        replans=0,
        planning_duration_seconds=0.01,
        execution_duration_seconds=0.02,
    )


def _successful_result() -> ExpertResult:
    phase_results = tuple(_successful_phase(phase) for phase in EXPERT_PHASE_SEQUENCE)
    return ExpertResult(
        success=True,
        status=ExpertStatus.SUCCESS,
        scene_seed=123,
        scene_id="langmani-pick-place-scene-v0:123",
        task_id="langmani-pick-place-task-v0:red_cube:right_bin:canonical_v0",
        canonical_instruction="Pick up the red cube and place it in the right bin.",
        target_object_id="red_cube",
        target_bin_id="right_bin",
        total_environment_steps=len(phase_results),
        total_planning_calls=0,
        total_replans=0,
        completed_phases=EXPERT_PHASE_SEQUENCE,
        failed_phase=None,
        final_environment_evaluation=_evaluation(success=True),
        phase_results=phase_results,
        planning_duration_seconds=0.12,
        execution_duration_seconds=0.24,
        diagnostic_artifact_paths=("outputs/diagnostics/m2/final.png",),
    )


def test_status_and_phase_values_are_stable_and_complete() -> None:
    assert tuple(status.value for status in ExpertStatus) == (
        "success",
        "invalid_task",
        "initialization_failure",
        "ik_failure",
        "planning_failure",
        "execution_failure",
        "grasp_failure",
        "transport_failure",
        "placement_failure",
        "verification_failure",
        "target_off_table",
        "timeout",
        "unexpected_exception",
    )
    assert tuple(phase.value for phase in ExpertPhase) == (
        "initialize",
        "move_to_pregrasp",
        "approach_target",
        "close_gripper",
        "verify_grasp",
        "lift_target",
        "move_above_destination",
        "descend_to_place",
        "open_gripper",
        "settle_after_release",
        "retreat",
        "verify_task",
    )
    assert tuple(ExpertPhase) == EXPERT_PHASE_SEQUENCE


def test_expert_config_defaults_are_frozen_and_json_serializable() -> None:
    config = ExpertConfig()

    assert config.control_mode == "pd_joint_pos"
    assert config.max_episode_steps == 200
    assert config.max_planning_attempts_per_phase == 1
    assert config.placement_clearance == pytest.approx(0.002)
    assert config.release_settling_steps == 15
    assert config.planner_timeout_seconds is None
    assert config.planner_seed is None
    assert json.loads(json.dumps(config.to_dict())) == config.to_dict()

    with pytest.raises(FrozenInstanceError):
        config.max_episode_steps = 201  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"control_mode": "pd_joint_delta_pos"}, ValueError, "pd_joint_pos"),
        ({"max_episode_steps": 0}, ValueError, "positive"),
        ({"max_episode_steps": True}, TypeError, "integer"),
        ({"max_planning_attempts_per_phase": 0}, ValueError, "positive"),
        ({"release_settling_steps": -1}, ValueError, "non-negative"),
        ({"pregrasp_clearance": 0.0}, ValueError, "positive"),
        ({"placement_clearance": float("nan")}, ValueError, "finite"),
        ({"diagnostic_rendering": 1}, TypeError, "bool"),
        ({"planner_timeout_seconds": 1.0}, ValueError, "unsupported"),
        ({"planner_seed": 0}, ValueError, "unsupported"),
    ],
)
def test_expert_config_rejects_invalid_or_unsupported_values(
    kwargs: dict[str, object], error_type: type[Exception], message: str
) -> None:
    with pytest.raises(error_type, match=message):
        ExpertConfig(**kwargs)  # type: ignore[arg-type]


def test_phase_result_is_frozen_validated_and_json_serializable() -> None:
    result = PhaseResult(
        phase=ExpertPhase.MOVE_TO_PREGRASP,
        success=False,
        status=ExpertStatus.IK_FAILURE,
        attempts=1,
        planning_calls=1,
        replans=0,
        planning_duration_seconds=0.25,
        message="no IK solution",
        planner_status="IK Failed",
    )

    payload = result.to_dict()
    assert payload["phase"] == "move_to_pregrasp"
    assert payload["status"] == "ik_failure"
    assert json.loads(json.dumps(payload)) == payload

    with pytest.raises(FrozenInstanceError):
        result.message = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="inconsistent"):
        PhaseResult(
            phase=ExpertPhase.INITIALIZE,
            success=True,
            status=ExpertStatus.INITIALIZATION_FAILURE,
        )
    with pytest.raises(ValueError, match="replans"):
        PhaseResult(
            phase=ExpertPhase.MOVE_TO_PREGRASP,
            success=False,
            status=ExpertStatus.PLANNING_FAILURE,
            planning_calls=0,
            replans=1,
        )


def test_success_result_is_compact_immutable_and_json_serializable() -> None:
    result = _successful_result()
    payload = result.to_dict()

    assert payload["completed_phases"] == [phase.value for phase in ExpertPhase]
    assert payload["failed_phase"] is None
    assert payload["final_environment_evaluation"]["success"] is True
    assert json.loads(json.dumps(payload)) == payload
    assert "trajectory" not in payload
    assert "observation" not in payload
    assert "image" not in payload

    with pytest.raises(TypeError):
        result.final_environment_evaluation["success"] = False  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        result.status = ExpertStatus.VERIFICATION_FAILURE  # type: ignore[misc]


def test_expected_phase_failure_preserves_phase_and_accounting() -> None:
    initialize = _successful_phase(ExpertPhase.INITIALIZE)
    failed = PhaseResult(
        phase=ExpertPhase.MOVE_TO_PREGRASP,
        success=False,
        status=ExpertStatus.IK_FAILURE,
        attempts=1,
        planning_calls=1,
        planning_duration_seconds=0.2,
        planner_status="IK Failed",
    )
    result = ExpertResult(
        success=False,
        status=ExpertStatus.IK_FAILURE,
        scene_seed=7,
        scene_id="langmani-pick-place-scene-v0:7",
        task_id="langmani-pick-place-task-v0:blue_cube:left_bin:canonical_v0",
        canonical_instruction="Pick up the blue cube and place it in the left bin.",
        target_object_id="blue_cube",
        target_bin_id="left_bin",
        total_environment_steps=1,
        total_planning_calls=1,
        total_replans=0,
        completed_phases=(ExpertPhase.INITIALIZE,),
        failed_phase=ExpertPhase.MOVE_TO_PREGRASP,
        final_environment_evaluation=_evaluation(success=False),
        phase_results=(initialize, failed),
        planning_duration_seconds=0.2,
        execution_duration_seconds=0.02,
    )

    assert result.to_dict()["failed_phase"] == "move_to_pregrasp"
    assert result.to_dict()["phase_results"][-1]["planner_status"] == "IK Failed"


def test_unexpected_exception_can_represent_failure_before_identity_is_available() -> None:
    result = ExpertResult(
        success=False,
        status=ExpertStatus.UNEXPECTED_EXCEPTION,
        scene_seed=None,
        scene_id=None,
        task_id=None,
        canonical_instruction=None,
        target_object_id=None,
        target_bin_id=None,
        total_environment_steps=0,
        total_planning_calls=0,
        total_replans=0,
        completed_phases=(),
        failed_phase=ExpertPhase.INITIALIZE,
        final_environment_evaluation={},
        phase_results=(),
        planning_duration_seconds=0.0,
        execution_duration_seconds=0.0,
        exception_type="RuntimeError",
        exception_message="planner initialization crashed",
    )

    payload = result.to_dict()
    assert payload["scene_seed"] is None
    assert payload["exception_type"] == "RuntimeError"
    assert json.loads(json.dumps(payload)) == payload

    with pytest.raises(ValueError, match="preserve exception"):
        ExpertResult(
            success=False,
            status=ExpertStatus.UNEXPECTED_EXCEPTION,
            scene_seed=None,
            scene_id=None,
            task_id=None,
            canonical_instruction=None,
            target_object_id=None,
            target_bin_id=None,
            total_environment_steps=0,
            total_planning_calls=0,
            total_replans=0,
            completed_phases=(),
            failed_phase=ExpertPhase.INITIALIZE,
            final_environment_evaluation={},
            phase_results=(),
            planning_duration_seconds=0.0,
            execution_duration_seconds=0.0,
        )


def test_result_rejects_tensor_evaluation_and_inconsistent_phase_lists() -> None:
    with pytest.raises(TypeError, match="values must be bools"):
        ExpertResult(
            success=False,
            status=ExpertStatus.INITIALIZATION_FAILURE,
            scene_seed=None,
            scene_id=None,
            task_id=None,
            canonical_instruction=None,
            target_object_id=None,
            target_bin_id=None,
            total_environment_steps=0,
            total_planning_calls=0,
            total_replans=0,
            completed_phases=(),
            failed_phase=ExpertPhase.INITIALIZE,
            final_environment_evaluation={"success": torch.tensor(False)},  # type: ignore[dict-item]
            phase_results=(),
            planning_duration_seconds=0.0,
            execution_duration_seconds=0.0,
        )

    with pytest.raises(ValueError, match="prefix"):
        ExpertResult(
            success=False,
            status=ExpertStatus.EXECUTION_FAILURE,
            scene_seed=1,
            scene_id="scene",
            task_id="task",
            canonical_instruction="instruction",
            target_object_id="red_cube",
            target_bin_id="left_bin",
            total_environment_steps=0,
            total_planning_calls=0,
            total_replans=0,
            completed_phases=(ExpertPhase.MOVE_TO_PREGRASP,),
            failed_phase=ExpertPhase.INITIALIZE,
            final_environment_evaluation={},
            phase_results=(),
            planning_duration_seconds=0.0,
            execution_duration_seconds=0.0,
        )
