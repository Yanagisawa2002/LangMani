from __future__ import annotations

import json
from pathlib import Path

import pytest

from langmani.v2.push_expert_recovery import load_json, sha256_file, write_json
from scripts.langmani_v2 import finalize_phase2b3_rejection as finalizer


def test_rejection_finalizer_preserves_sealed_gates_and_zero_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "artifacts"
    expert_path = finalizer.PROJECT_ROOT / "configs/langmani_v2/phase_2b3/expert.json"
    diagnostic_commit = "a" * 40
    preflight_commit = "b" * 40
    result = {
        "status": "wrong_object_interaction",
        "success": False,
        "total_environment_steps": 25,
        "total_planning_calls": 1,
        "contact_recoveries": 0,
        "replans": 0,
        "transitions": [{"source": "move_to_staging", "target": "safe_abort"}],
        "final_environment_evaluation": {
            "target_distance": 0.4,
            "target_inside_region": False,
            "target_outside_workspace": False,
            "invalid_action": False,
            "action_out_of_bounds": False,
            "wrong_object_contact": False,
            "wrong_object_displaced": True,
        },
    }
    record = {
        "seed": 1,
        "task": {
            "target_object_id": "blue_cube",
            "target_region_id": "left",
            "difficulty": "hard",
        },
        "result": result,
    }
    report = {
        "source_commit": diagnostic_commit,
        "completed_episodes": 1,
        "results": [result],
        "successes": 0,
        "failures": 1,
        "success_rate": 0.0,
        "mean_episode_steps": 25.0,
        "mean_contact_recoveries": 0.0,
        "mean_replans": 0.0,
        "workspace_violations": 0,
        "action_bound_violations": 0,
        "nonfinite_actions": 0,
        "simulator_errors": 0,
        "false_successes": 0,
        "official_collection_episodes": 0,
        "optimizer_steps": 0,
        "expert_config_sha256": sha256_file(expert_path),
        "expert_semantic_digest": "c" * 64,
    }
    diagnostic_path = tmp_path / "diagnostic.json"
    ledger_path = tmp_path / "ledger.jsonl"
    remote_static_path = tmp_path / "remote-static.json"
    gpu_path = tmp_path / "gpu.txt"
    write_json(diagnostic_path, report)
    ledger_path.write_text(json.dumps(record) + "\n", encoding="utf-8", newline="\n")
    write_json(remote_static_path, {"source_commit": preflight_commit, "passed": True})
    gpu_path.write_text("test gpu", encoding="utf-8", newline="\n")
    write_json(
        output_dir / "static_geometry_validation.json",
        {"passed": True, "config_sha256": sha256_file(expert_path)},
    )
    write_json(output_dir / "replay_validation.json", {"case_count": 55})
    for name in ("failure_registry", "root_cause_summary", "seed_registry"):
        write_json(output_dir / f"{name}.json", {"schema_version": "test"})

    monkeypatch.setattr(finalizer, "_source_commit", lambda: "d" * 40)
    summary = finalizer.build_rejection_artifacts(
        diagnostic_report_path=diagnostic_path,
        diagnostic_ledger_path=ledger_path,
        remote_static_report_path=remote_static_path,
        gpu_report_path=gpu_path,
        output_dir=output_dir,
        diagnostic_run_id="diagnostic",
        preflight_run_id="preflight",
        initial_smoke_run_id="smoke",
        replay_run_id="replay",
        expected_diagnostic_source_commit=diagnostic_commit,
        expected_preflight_source_commit=preflight_commit,
    )

    assert summary["expert_package"] == "REJECTED"
    development = load_json(output_dir / "development_evaluation.json")
    acceptance = load_json(output_dir / "acceptance_evaluation.json")
    manifest = load_json(output_dir / "expert_package_manifest.json")
    assert development["status"] == "NOT_RUN_REGRESSION_GATE_FAILED"
    assert development["episodes_executed"] == 0
    assert acceptance["status"] == "NOT_RUN_DEVELOPMENT_GATE_FAILED"
    assert acceptance["episodes_executed"] == 0
    assert acceptance["sealed"] is True
    assert manifest["package_status"] == "REJECTED"
    assert manifest["official_collection_episodes"] == 0
    assert manifest["optimizer_steps"] == 0
