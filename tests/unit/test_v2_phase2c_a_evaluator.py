"""CPU-safe summaries and frozen selection tests for Phase 2C-A evaluation."""

from __future__ import annotations

import pytest

pytest.importorskip("lerobot")

from langmani.v2.phase2c_a import TASK_IDS  # noqa: E402
from langmani.v2.phase2c_a_evaluator import (  # noqa: E402
    select_execution_horizon,
    summarize_evaluation,
)


def _episode(*, success: bool, task_id: str = "PickCube-v1") -> dict[str, object]:
    return {
        "success": success,
        "outcome": "success" if success else "timeout",
        "failure_category": "other" if success else "timeout",
        "invalid_action": False,
        "task_id": task_id,
        "policy_id": "policy",
        "checkpoint_identity": "sha256:" + "1" * 64,
        "episode_length": 100,
        "policy_query_count": 25,
        "action_saturation_rate": 0.0,
        "action_smoothness_mean_l2": 0.02,
        "inference_latency_samples_ms": [4.0, 5.0],
        "post_contact_divergence_m": 0.001,
    }


def test_evaluation_summary_has_confidence_interval_and_latency() -> None:
    result = summarize_evaluation([_episode(success=True), _episode(success=False)])
    assert result["success_count"] == 1
    assert result["success_rate"] == 0.5
    assert len(result["success_wilson_95"]) == 2
    assert result["inference_latency_p50_ms"] == 4.5
    assert result["inference_latency_p95_ms"] == pytest.approx(4.95)


def test_horizon_selection_uses_validation_only_and_prefers_success() -> None:
    reports = []
    for horizon, successes in ((1, 3), (4, 7), (8, 5)):
        reports.append(
            {
                "execution_horizon": horizon,
                "split": "validation",
                "episode_count": 10,
                "success_count": successes,
                "invalid_action_count": 0,
                "outcomes": {"timeout": 10 - successes},
                "mean_action_smoothness_l2": 0.1,
                "mean_policy_query_count": 100 / horizon,
                "inference_latency_p95_ms": 5.0,
            }
        )
    result = select_execution_horizon(reports)
    assert result["passed"] is True
    assert result["selected_execution_horizon"] == 4
    assert result["final_test_outcomes_available"] is False


def test_invalid_horizon_candidate_is_ineligible() -> None:
    reports = []
    for horizon in (1, 4, 8):
        reports.append(
            {
                "execution_horizon": horizon,
                "split": "validation",
                "episode_count": 10,
                "success_count": 9 if horizon == 8 else 1,
                "invalid_action_count": 1 if horizon == 8 else 0,
                "outcomes": {"timeout": 0},
                "mean_action_smoothness_l2": 0.1,
                "mean_policy_query_count": 10.0,
                "inference_latency_p95_ms": 5.0,
            }
        )
    result = select_execution_horizon(reports)
    assert result["selected_execution_horizon"] in {1, 4}


def test_summary_accepts_all_three_tasks_without_language_claim() -> None:
    records = [_episode(success=True, task_id=task_id) for task_id in TASK_IDS]
    result = summarize_evaluation(records)
    assert result["task_ids"] == list(TASK_IDS)
