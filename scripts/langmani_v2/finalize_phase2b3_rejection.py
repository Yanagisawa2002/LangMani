"""Finalize compact Phase 2B.3 Result B evidence after a fail-fast diagnostic."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from langmani.v2.push_expert_recovery import load_json, sha256_file, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse compact-evidence finalization arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-report", type=Path, required=True)
    parser.add_argument("--diagnostic-ledger", type=Path, required=True)
    parser.add_argument("--remote-static-report", type=Path, required=True)
    parser.add_argument("--gpu-report", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-run-id",
        default="20260723T_phase2b3-smoke4_3e7f805",
    )
    parser.add_argument(
        "--preflight-run-id",
        default="20260723T_phase2b3-preflight_d5e310e",
    )
    parser.add_argument(
        "--initial-smoke-run-id",
        default="20260723T_phase2b3-smoke4_d5e310e",
    )
    parser.add_argument(
        "--replay-run-id",
        default="20260722T202052Z_phase2b3-replay-evidence_e63931c",
    )
    parser.add_argument(
        "--expected-diagnostic-source-commit",
        default="3e7f805f7b9a0c0a230a61c27b83eb85d2e7c591",
    )
    parser.add_argument(
        "--expected-preflight-source-commit",
        default="d5e310e2a0a9e6584188c97e1537bb574d2e73f9",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/langmani_v2/phase_2b3",
    )
    return parser.parse_args()


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def _load_ledger(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"ledger line {line_number} is not a mapping")
        records.append(value)
    return records


def _require_zero(report: dict[str, Any], key: str) -> None:
    if report.get(key) != 0:
        raise ValueError(f"diagnostic {key} must be zero, observed {report.get(key)!r}")


def _case_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for record in records:
        task = record.get("task")
        result = record.get("result")
        if not isinstance(task, dict) or not isinstance(result, dict):
            raise ValueError("diagnostic ledger record is malformed")
        evaluation = result.get("final_environment_evaluation")
        if not isinstance(evaluation, dict):
            raise ValueError("diagnostic result is missing final environment evaluation")
        transitions = result.get("transitions")
        if not isinstance(transitions, list):
            raise ValueError("diagnostic result is missing transitions")
        last_transition = transitions[-1] if transitions else {}
        summaries.append(
            {
                "seed": record.get("seed"),
                "task": task,
                "status": result.get("status"),
                "success": result.get("success"),
                "episode_steps": result.get("total_environment_steps"),
                "planning_calls": result.get("total_planning_calls"),
                "contact_recoveries": result.get("contact_recoveries"),
                "replans": result.get("replans"),
                "terminal_state": (
                    last_transition.get("target") if isinstance(last_transition, dict) else None
                ),
                "target_distance": evaluation.get("target_distance"),
                "target_inside_region": evaluation.get("target_inside_region"),
                "target_outside_workspace": evaluation.get("target_outside_workspace"),
                "invalid_action": evaluation.get("invalid_action"),
                "action_out_of_bounds": evaluation.get("action_out_of_bounds"),
                "wrong_object_contact": evaluation.get("wrong_object_contact"),
                "wrong_object_displaced": evaluation.get("wrong_object_displaced"),
            }
        )
    return summaries


def build_rejection_artifacts(
    *,
    diagnostic_report_path: Path,
    diagnostic_ledger_path: Path,
    remote_static_report_path: Path,
    gpu_report_path: Path,
    output_dir: Path,
    diagnostic_run_id: str,
    preflight_run_id: str,
    initial_smoke_run_id: str,
    replay_run_id: str,
    expected_diagnostic_source_commit: str,
    expected_preflight_source_commit: str,
) -> dict[str, Any]:
    """Validate retrieved evidence and write the compact rejected-result contract."""

    diagnostic = load_json(diagnostic_report_path)
    records = _load_ledger(diagnostic_ledger_path)
    static = load_json(output_dir / "static_geometry_validation.json")
    remote_static = load_json(remote_static_report_path)
    replay = load_json(output_dir / "replay_validation.json")
    expert_path = PROJECT_ROOT / "configs/langmani_v2/phase_2b3/expert.json"
    seed_registry_path = output_dir / "seed_registry.json"
    root_cause_path = output_dir / "root_cause_summary.json"

    if diagnostic.get("source_commit") != expected_diagnostic_source_commit:
        raise ValueError("diagnostic source commit does not match the authorized revision")
    if remote_static.get("source_commit") != expected_preflight_source_commit:
        raise ValueError("remote static preflight source commit is unexpected")
    if diagnostic.get("completed_episodes") != len(records):
        raise ValueError("diagnostic report and ledger episode counts differ")
    report_results = diagnostic.get("results")
    if not isinstance(report_results, list) or len(report_results) != len(records):
        raise ValueError("diagnostic report result inventory is incomplete")
    if any(
        record.get("result") != result
        for record, result in zip(records, report_results, strict=True)
    ):
        raise ValueError("diagnostic report results do not match the ledger")
    if diagnostic.get("official_collection_episodes") != 0:
        raise ValueError("official collection must remain zero")
    if diagnostic.get("optimizer_steps") != 0:
        raise ValueError("optimizer steps must remain zero")
    for key in (
        "workspace_violations",
        "action_bound_violations",
        "nonfinite_actions",
        "simulator_errors",
        "false_successes",
    ):
        _require_zero(diagnostic, key)
    wrong_object_interactions = sum(
        record.get("result", {}).get("status") == "wrong_object_interaction" for record in records
    )
    if wrong_object_interactions < 1:
        raise ValueError("rejection finalization requires an observed zero-tolerance failure")
    if static.get("passed") is not True or remote_static.get("passed") is not True:
        raise ValueError("local and remote static validation must both pass")
    if static.get("config_sha256") != diagnostic.get("expert_config_sha256"):
        raise ValueError("current expert config does not match the evaluated diagnostic")
    case_count = replay.get("case_count")
    if not isinstance(case_count, int) or case_count < len(records):
        raise ValueError("replay bank count is incompatible with the diagnostic")

    current_source_commit = _source_commit()
    cases = _case_summaries(records)
    status_counts = dict(sorted(Counter(str(case["status"]) for case in cases).items()))
    terminal_state_counts = dict(
        sorted(Counter(str(case["terminal_state"]) for case in cases).items())
    )
    regression = {
        "schema_version": "langmani-v2-phase2b3-regression-evaluation-v1",
        "status": "FAILED_ZERO_TOLERANCE_DIAGNOSTIC",
        "passed": False,
        "formal_gate_eligible": False,
        "evaluated_source_commit": diagnostic["source_commit"],
        "finalizer_source_commit": current_source_commit,
        "diagnostic_run_id": diagnostic_run_id,
        "full_regression_bank_episodes": case_count,
        "diagnostic_episodes_executed": len(records),
        "untouched_regression_episodes": case_count - len(records),
        "successes": diagnostic.get("successes"),
        "failures": diagnostic.get("failures"),
        "diagnostic_success_rate": diagnostic.get("success_rate"),
        "status_counts": status_counts,
        "terminal_state_counts": terminal_state_counts,
        "mean_episode_steps": diagnostic.get("mean_episode_steps"),
        "mean_contact_recoveries": diagnostic.get("mean_contact_recoveries"),
        "mean_replans": diagnostic.get("mean_replans"),
        "workspace_violations": diagnostic.get("workspace_violations"),
        "action_bound_violations": diagnostic.get("action_bound_violations"),
        "nonfinite_actions": diagnostic.get("nonfinite_actions"),
        "simulator_errors": diagnostic.get("simulator_errors"),
        "false_successes": diagnostic.get("false_successes"),
        "wrong_object_interactions": wrong_object_interactions,
        "stopped_early_reason": "wrong_object_interaction",
        "development_gate_authorized": False,
        "acceptance_bank_accessed": False,
        "official_collection_episodes": 0,
        "optimizer_steps": 0,
        "diagnostic_report_sha256": sha256_file(diagnostic_report_path),
        "diagnostic_ledger_sha256": sha256_file(diagnostic_ledger_path),
        "expert_config_sha256": sha256_file(expert_path),
        "expert_semantic_digest": diagnostic.get("expert_semantic_digest"),
        "cases": cases,
    }
    write_json(output_dir / "regression_evaluation.json", regression)

    development = {
        "schema_version": "langmani-v2-phase2b3-development-evaluation-v1",
        "status": "NOT_RUN_REGRESSION_GATE_FAILED",
        "passed": False,
        "episodes_expected": 100,
        "episodes_executed": 0,
        "successes": 0,
        "success_rate": None,
        "minimum_successes": 95,
        "reason": "A zero-tolerance wrong-object interaction failed the diagnostic regression gate.",
        "regression_evaluation_sha256": sha256_file(output_dir / "regression_evaluation.json"),
        "acceptance_bank_accessed": False,
        "official_collection_episodes": 0,
        "optimizer_steps": 0,
    }
    write_json(output_dir / "development_evaluation.json", development)

    acceptance = {
        "schema_version": "langmani-v2-phase2b3-acceptance-evaluation-v1",
        "status": "NOT_RUN_DEVELOPMENT_GATE_FAILED",
        "passed": False,
        "sealed": True,
        "episodes_expected": 100,
        "episodes_executed": 0,
        "successes": 0,
        "success_rate": None,
        "minimum_successes": 95,
        "reason": "Development was not authorized after the regression gate failed.",
        "development_evaluation_sha256": sha256_file(output_dir / "development_evaluation.json"),
        "official_collection_episodes": 0,
        "optimizer_steps": 0,
    }
    write_json(output_dir / "acceptance_evaluation.json", acceptance)

    remote_audit = {
        "schema_version": "langmani-v2-phase2b3-remote-execution-audit-v1",
        "status": "COMPLETE_REJECTED",
        "remote_tracked_source_clean": True,
        "evaluated_source_commits": {
            "replay": replay.get("runner_source_commit"),
            "static_preflight_and_initial_smoke": expected_preflight_source_commit,
            "budget_aware_diagnostic": expected_diagnostic_source_commit,
        },
        "run_ids": {
            "replay_evidence": replay_run_id,
            "static_preflight": preflight_run_id,
            "initial_four_episode_smoke": initial_smoke_run_id,
            "twelve_episode_diagnostic": diagnostic_run_id,
        },
        "runtime": {
            "python": "3.12.3",
            "torch": "2.11.0+cu130",
            "numpy": "1.26.4",
            "maniskill": "3.0.1",
            "sapien": "3.0.3",
            "mplib": "0.1.1",
            "gpu_inventory": gpu_report_path.read_text(encoding="utf-8").strip(),
        },
        "retrieved_evidence_sha256": {
            "diagnostic_report": sha256_file(diagnostic_report_path),
            "diagnostic_ledger": sha256_file(diagnostic_ledger_path),
            "remote_static_preflight": sha256_file(remote_static_report_path),
            "gpu_inventory": sha256_file(gpu_report_path),
        },
        "training_performed": False,
        "formal_data_collection_performed": False,
        "lerobot_export_performed": False,
        "smolvla_started": False,
        "phase_2c2_started": False,
        "phase_2d_started": False,
        "official_collection_episodes": 0,
        "optimizer_steps": 0,
        "server_disposition": "ONLINE_PER_USER_INSTRUCTION",
    }
    write_json(output_dir / "remote_execution_audit.json", remote_audit)

    manifest_inputs = {
        "expert_config": expert_path,
        "failure_registry": output_dir / "failure_registry.json",
        "replay_validation": output_dir / "replay_validation.json",
        "root_cause_summary": root_cause_path,
        "seed_registry": seed_registry_path,
        "static_geometry_validation": output_dir / "static_geometry_validation.json",
        "regression_evaluation": output_dir / "regression_evaluation.json",
        "development_evaluation": output_dir / "development_evaluation.json",
        "acceptance_evaluation": output_dir / "acceptance_evaluation.json",
        "remote_execution_audit": output_dir / "remote_execution_audit.json",
    }
    manifest = {
        "schema_version": "langmani-v2-phase2b3-expert-package-manifest-v1",
        "package_status": "REJECTED",
        "architecture": "GeometryConstrainedPushExpert",
        "state_machine_version": "geometry_constrained_closed_loop_push_v1",
        "evaluated_source_commit": diagnostic["source_commit"],
        "finalizer_source_commit": current_source_commit,
        "expert_config_sha256": sha256_file(expert_path),
        "expert_semantic_digest": diagnostic.get("expert_semantic_digest"),
        "rejection_reason": "ZERO_TOLERANCE_WRONG_OBJECT_INTERACTION",
        "development_passed": False,
        "acceptance_passed": False,
        "acceptance_bank_accessed": False,
        "collection_entrypoint_authorized": False,
        "official_collection_episodes": 0,
        "optimizer_steps": 0,
        "file_sha256": {name: sha256_file(path) for name, path in sorted(manifest_inputs.items())},
    }
    write_json(output_dir / "expert_package_manifest.json", manifest)
    return {
        "result": "RESULT_B_EXPERT_GATE_FAILED",
        "diagnostic_episodes": len(records),
        "wrong_object_interactions": wrong_object_interactions,
        "development_episodes": 0,
        "acceptance_episodes": 0,
        "official_collection_episodes": 0,
        "expert_package": "REJECTED",
    }


def main() -> int:
    """Run rejection finalization and print its compact result."""

    args = parse_args()
    summary = build_rejection_artifacts(
        diagnostic_report_path=args.diagnostic_report,
        diagnostic_ledger_path=args.diagnostic_ledger,
        remote_static_report_path=args.remote_static_report,
        gpu_report_path=args.gpu_report,
        output_dir=args.output_dir,
        diagnostic_run_id=args.diagnostic_run_id,
        preflight_run_id=args.preflight_run_id,
        initial_smoke_run_id=args.initial_smoke_run_id,
        replay_run_id=args.replay_run_id,
        expected_diagnostic_source_commit=args.expected_diagnostic_source_commit,
        expected_preflight_source_commit=args.expected_preflight_source_commit,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
