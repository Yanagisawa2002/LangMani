from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from langmani.v2.push_expert_recovery import (
    PushExpertRecoveryError,
    PushFailureCategory,
    build_failure_registry,
    build_seed_registry,
    classify_historical_failure,
    read_seed_bank,
    sha256_file,
)
from scripts.langmani_v2.replay_push_expert_failure import (
    _save_frame,
    _selected_cases,
    _workspace_violation_detail,
)


def _result(*, status: str, phase: str, evaluation: dict[str, object]) -> dict[str, object]:
    return {
        "success": False,
        "status": status,
        "failed_phase": phase,
        "final_environment_evaluation": evaluation,
    }


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            _result(status="planning_failure", phase="move_to_precontact", evaluation={}),
            PushFailureCategory.PREAPPROACH_PLANNING_FAILURE,
        ),
        (
            _result(status="timeout", phase="primary_push", evaluation={"no_progress_stall": True}),
            PushFailureCategory.STALLED_PROGRESS,
        ),
        (
            _result(
                status="target_outside_workspace",
                phase="settle",
                evaluation={"target_outside_workspace": True},
            ),
            PushFailureCategory.OBJECT_WORKSPACE_VIOLATION,
        ),
        (
            _result(
                status="execution_failure",
                phase="primary_push",
                evaluation={"action_out_of_bounds": True},
            ),
            PushFailureCategory.ACTION_BOUND_REJECTION,
        ),
    ],
)
def test_historical_failure_classification(
    result: dict[str, object], expected: PushFailureCategory
) -> None:
    assert classify_historical_failure(result) is expected


def test_seed_registry_is_disjoint_and_acceptance_is_sealed() -> None:
    registry = build_seed_registry()
    development = read_seed_bank(registry, "development")
    assert development == tuple(range(69_000, 69_100))
    with pytest.raises(PushExpertRecoveryError, match="sealed"):
        read_seed_bank(registry, "acceptance")
    report = {
        "passed": True,
        "source_commit": "a" * 40,
        "completed_episodes": 100,
        "successes": 95,
        "workspace_violations": 0,
        "action_bound_violations": 0,
        "nonfinite_actions": 0,
        "simulator_errors": 0,
        "false_successes": 0,
    }
    acceptance = read_seed_bank(
        registry,
        "acceptance",
        development_report=report,
        expected_source_commit="a" * 40,
    )
    assert acceptance == tuple(range(69_200, 69_300))
    assert not set(development).intersection(acceptance)


def test_acceptance_rejects_below_gate_development() -> None:
    registry = build_seed_registry()
    report = {
        "passed": True,
        "source_commit": "a" * 40,
        "completed_episodes": 100,
        "successes": 94,
        "workspace_violations": 0,
        "action_bound_violations": 0,
        "nonfinite_actions": 0,
        "simulator_errors": 0,
        "false_successes": 0,
    }
    with pytest.raises(PushExpertRecoveryError, match="below 95"):
        read_seed_bank(
            registry,
            "acceptance",
            development_report=report,
            expected_source_commit="a" * 40,
        )


def test_registry_rejects_report_digest_drift(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    report = {"runtime": {"git_commit": "a" * 40}, "results": []}
    report_path = run / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    config: dict[str, object] = {
        "runs": [
            {
                "candidate_id": "CandidateE",
                "run_id": "run",
                "source_commit": "a" * 40,
                "report": "report.json",
                "sha256": sha256_file(report_path),
            }
        ]
    }
    registry = build_failure_registry(config=config, history_root=tmp_path)
    assert registry["historical_result_count"] == 0
    changed = copy.deepcopy(config)
    runs = changed["runs"]
    assert isinstance(runs, list) and isinstance(runs[0], dict)
    runs[0]["sha256"] = "0" * 64
    with pytest.raises(PushExpertRecoveryError, match="digest mismatch"):
        build_failure_registry(config=changed, history_root=tmp_path)


def test_keyframe_writer_moves_device_tensor_to_cpu(tmp_path: Path) -> None:
    import numpy as np

    class DeviceFrame:
        detached = False

        def detach(self) -> DeviceFrame:
            self.detached = True
            return self

        def cpu(self) -> np.ndarray:
            assert self.detached
            return np.zeros((8, 8, 3), dtype=np.uint8)

    class Environment:
        def render(self) -> DeviceFrame:
            return DeviceFrame()

    output = tmp_path / "frame.png"
    _save_frame(Environment(), output)
    assert output.is_file()


def test_replay_selection_counts_only_new_category_cases() -> None:
    records = [
        {
            "candidate_id": "CandidateE" if seed < 3 else "CandidateH",
            "seed": seed,
            "episode_result": "FAILURE",
            "failure_category": "STALLED_PROGRESS",
        }
        for seed in range(5)
    ]
    selected = _selected_cases(
        {"records": records},
        {
            "per_failure_category": 5,
            "selection": {"candidate_f_per_category": 5},
        },
    )
    assert len(selected) == 5


def test_workspace_violation_binds_plan_execution_and_object_geometry() -> None:
    common = {
        "target_center": [0.2, 0.0, 0.0],
        "target_object_planar_radius": 0.025,
        "workspace_bounds_xy": [-0.43, 0.43, -0.38, 0.38],
        "target_contact": True,
    }
    trace = [
        common
        | {
            "step": 4,
            "target_object_pose": [0.39, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0],
            "tcp_position": [0.35, 0.0, 0.025],
            "evaluation": {"target_outside_workspace": False},
        },
        common
        | {
            "step": 5,
            "target_object_pose": [0.41, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0],
            "tcp_position": [0.37, 0.0, 0.025],
            "evaluation": {"target_outside_workspace": True},
        },
    ]
    planned = [0.4, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]
    detail = _workspace_violation_detail(
        trace=trace,
        planner_calls=[
            {
                "action_start_index": 4,
                "action_end_index": 6,
                "planned_pose": planned,
            }
        ],
        object_shape="cylinder",
    )
    assert detail is not None
    assert detail["planned_pose"] == planned
    assert detail["executed_pose"] == [0.37, 0.0, 0.025]
    assert detail["nearest_workspace_boundary"] == "x_max"
    assert detail["signed_workspace_margin"] == pytest.approx(-0.005)
    assert detail["object_support_extent_in_push_direction"] == pytest.approx(0.025)
