"""Result classification and ACT closure for Phase 2C-A.1."""

from __future__ import annotations

from typing import cast

from langmani.v2.phase2c_a import TASK_IDS
from langmani.v2.phase2c_a1_analysis import analyze_phase2c_a1_results


def _episode(index: int, *, success: bool) -> dict[str, object]:
    return {
        "success": success,
        "timeout": not success,
        "invalid_action": False,
        "simulator_error": False,
        "actions_executed": 1,
        "policy_query_count": 1,
        "outcome": "success" if success else "timeout",
        "failure_category": "other" if success else "no_interaction",
        "action_sequence": [[float(index) / 100.0] * 8],
        "inference_latency_samples_ms": [1.0],
        "action_smoothness_mean_l2": None,
        "checkpoint_identity": "sha256:" + "a" * 64,
        "execution_horizon": 4,
    }


def _scope(successes: int) -> dict[str, dict[str, list[dict[str, object]]]]:
    return {
        split: {
            task_id: [_episode(index, success=index < successes) for index in range(30)]
            for task_id in TASK_IDS
        }
        for split in ("validation", "test_unseen_reset", "test_visual_shift")
    }


def _analyze(per_successes: int, shared_successes: int) -> dict[str, object]:
    return analyze_phase2c_a1_results(
        per_task=_scope(per_successes),
        shared=_scope(shared_successes),
        training_runs_complete=True,
        checkpoints_recoverable=True,
        padding_mask_verified=True,
        closed_loop_smoke_passed=True,
    )


def test_result_a_and_permanent_closure() -> None:
    result = _analyze(10, 8)
    assert result["result"] == "RESULT_A"
    assert result["act_baselines_validated"] is True
    assert result["act_phase_closed"] is True
    assert result["further_act_architecture_authorized"] is False
    assert result["smolvla_phase_eligible"] is True
    assert result["smolvla_training_authorized"] is False
    assert result["vla_jepa_training_authorized"] is False


def test_result_b_for_valid_but_weak_per_task_policy() -> None:
    result = _analyze(2, 8)
    assert result["result"] == "RESULT_B"
    assert result["act_policy_quality_weak"] is True
    assert result["act_baselines_validated"] is True


def test_result_c_for_shared_only_failure() -> None:
    result = _analyze(8, 2)
    assert result["result"] == "RESULT_C"
    assert result["shared_act_failed"] is True
    assert result["act_baselines_validated"] is True


def test_result_d_for_invalid_consumer_action() -> None:
    per_task = _scope(8)
    first = cast(
        dict[str, object],
        per_task["test_unseen_reset"]["PickCube-v1"][0],
    )
    first["invalid_action"] = True
    result = analyze_phase2c_a1_results(
        per_task=per_task,
        shared=_scope(8),
        training_runs_complete=True,
        checkpoints_recoverable=True,
        padding_mask_verified=True,
        closed_loop_smoke_passed=True,
    )
    assert result["result"] == "RESULT_D"
    assert result["act_baselines_validated"] is False
    assert result["act_phase_closed"] is True
    assert result["smolvla_phase_eligible"] is False
