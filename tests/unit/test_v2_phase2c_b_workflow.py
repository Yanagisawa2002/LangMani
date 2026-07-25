from __future__ import annotations

import json
from pathlib import Path

from langmani.v2.phase2c_a import PACKAGE_FINGERPRINT
from langmani.v2.phase2c_b import (
    HORIZON_SELECTION_EPISODES_PER_TASK,
    build_horizon_selection_lock,
    canonical_fingerprint,
)
from scripts.diagnose_v2_phase2c_b_smolvla import _balanced_indices
from scripts.select_v2_phase2c_b_checkpoints import main as select_checkpoints
from scripts.select_v2_phase2c_b_policy import main as select_policy


def _write_fingerprinted(path: Path, semantic: dict[str, object]) -> None:
    path.write_text(
        json.dumps(
            {**semantic, "fingerprint": canonical_fingerprint(semantic)},
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_balanced_diagnostic_indices_cover_each_task_without_duplicates() -> None:
    indices = _balanced_indices(
        {"PickCube-v1": 10, "StackCube-v1": 20, "PushCube-v1": 30},
        maximum_frames_per_task=4,
    )
    assert indices == [0, 3, 6, 9, 10, 16, 23, 29, 30, 40, 49, 59]
    assert len(indices) == len(set(indices)) == 12


def test_checkpoint_and_horizon_selection_use_more_than_total_loss(
    tmp_path: Path,
    monkeypatch,
) -> None:
    diagnostics = []
    checkpoints = ["sha256:" + "1" * 64, "sha256:" + "2" * 64]
    for index, (step, checkpoint, first_error, validation_loss) in enumerate(
        (
            (5_000, checkpoints[0], 0.2, 0.1),
            (10_000, checkpoints[1], 0.1, 0.2),
        )
    ):
        path = tmp_path / f"diagnostic-{index}.json"
        _write_fingerprinted(
            path,
            {
                "schema_version": ("langmani-v2-phase2c-b-offline-checkpoint-diagnostic-v0"),
                "package_fingerprint": PACKAGE_FINGERPRINT,
                "model_kind": "pick_smolvla",
                "checkpoint": f"/checkpoint/{step}",
                "checkpoint_sha256": checkpoint,
                "optimizer_step": step,
                "split": "validation",
                "physical_first_action_mae": first_error,
                "arm_action_mae": first_error,
                "gripper_action_mae": first_error,
                "validation_flow_matching_loss": validation_loss,
                "invalid_action_count": 0,
                "passed": True,
            },
        )
        diagnostics.append(path)
    checkpoint_selection = tmp_path / "checkpoint-selection.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "select-checkpoints",
            "--diagnostic",
            str(diagnostics[0]),
            "--diagnostic",
            str(diagnostics[1]),
            "--output",
            str(checkpoint_selection),
        ],
    )
    assert select_checkpoints() == 0
    selected = json.loads(checkpoint_selection.read_text(encoding="utf-8"))
    assert selected["selected"][0]["checkpoint_sha256"] == checkpoints[1]
    assert selected["selected_on_total_loss_only"] is False

    schedule_fingerprint = "sha256:" + "3" * 64
    horizon_lock = tmp_path / "horizon-lock.json"
    _write_fingerprinted(
        horizon_lock,
        {
            "schema_version": "langmani-v2-phase2c-b-horizon-selection-lock-v0",
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "evaluation_schedule_fingerprint": schedule_fingerprint,
            "split": "validation",
            "episodes_per_task": HORIZON_SELECTION_EPISODES_PER_TASK,
            "execution_horizons": [1, 4, 8],
            "evaluation_ids": {},
            "one_common_horizon_preferred": True,
            "multiskill_practical_difference_threshold": 0.1,
            "settings_mutable_after_results": False,
            "final_test_opened": False,
        },
    )
    summaries: list[Path] = []
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        for horizon in (1, 4, 8):
            success_count = 2 if checkpoint_index == 1 and horizon == 4 else 1
            path = tmp_path / f"summary-{checkpoint_index}-{horizon}.json"
            _write_fingerprinted(
                path,
                {
                    "schema_version": "langmani-v2-phase2c-a-evaluation-summary-v0",
                    "package_fingerprint": PACKAGE_FINGERPRINT,
                    "model_kind": "pick_smolvla",
                    "task_id": "PickCube-v1",
                    "split": "validation",
                    "instruction_condition": "correct",
                    "checkpoint_identities": [checkpoint],
                    "execution_horizon": horizon,
                    "episode_count": HORIZON_SELECTION_EPISODES_PER_TASK,
                    "success_count": success_count,
                    "success_rate": success_count / HORIZON_SELECTION_EPISODES_PER_TASK,
                    "invalid_action_count": 0,
                    "simulator_error_count": 0,
                    "outcomes": {"timeout": HORIZON_SELECTION_EPISODES_PER_TASK - success_count},
                    "mean_action_smoothness_l2": 0.2,
                    "mean_policy_query_count": 10.0,
                    "inference_latency_p95_ms": 20.0,
                    "schedule_fingerprint": schedule_fingerprint,
                },
            )
            summaries.append(path)
    policy_selection = tmp_path / "policy-selection.json"
    argv = [
        "select-policy",
        "--checkpoint-selection",
        str(checkpoint_selection),
        "--horizon-lock",
        str(horizon_lock),
    ]
    for summary in summaries:
        argv.extend(("--summary", str(summary)))
    argv.extend(("--output", str(policy_selection)))
    monkeypatch.setattr("sys.argv", argv)
    assert select_policy() == 0
    policy = json.loads(policy_selection.read_text(encoding="utf-8"))
    assert policy["selected_checkpoint_sha256"] == checkpoints[1]
    assert policy["selected_execution_horizon"] == 4
    assert policy["final_test_opened"] is False


def test_horizon_lock_freezes_first_six_validation_ids() -> None:
    schedules = {
        "validation": {
            task_id: [
                {"evaluation_id": f"{task_id}:{index}"}
                for index in range(HORIZON_SELECTION_EPISODES_PER_TASK + 1)
            ]
            for task_id in ("PickCube-v1", "StackCube-v1", "PushCube-v1")
        }
    }
    semantic = {
        "schema_version": "langmani-v2-phase2c-b-evaluation-schedule-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "schedules": schedules,
    }
    schedule = {**semantic, "fingerprint": canonical_fingerprint(semantic)}
    lock = build_horizon_selection_lock(schedule)
    assert lock["episodes_per_task"] == HORIZON_SELECTION_EPISODES_PER_TASK
    assert len(lock["evaluation_ids"]["PickCube-v1"]) == 6
    assert lock["settings_mutable_after_results"] is False
