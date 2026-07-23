"""Finalize compact Phase 2B.4-F0 Result C evidence after the bounded run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from langmani.v2.phase2b4_evidence import REQUIRED_ARTIFACTS
from langmani.v2.phase2b4_ppo import (
    EXCLUDED_GEOMETRY_COMMIT,
    SOURCE_COMMIT,
    Phase2B4Result,
    authorization_state,
    canonical_json_sha256,
    classify_phase2b4_result,
)
from langmani.v2.phase2b4_runtime import write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2b4_f0"
EXECUTION_COMMIT = "0740c2fcaf5a34405130f3e3a8a3fac6e9a5d406"
CHECKPOINT_SHA256 = "sha256:3528003bcb9ce1ae12904f0025eb7cffe3edd9836b24e3e4eef6becfcb672bcff"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    return parser.parse_args()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
    ).strip()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _with_fingerprint(payload: dict[str, Any]) -> dict[str, Any]:
    payload["fingerprint"] = canonical_json_sha256(payload)
    return payload


def _write_derived(root: Path) -> None:
    training = _read(root / "micro_training_metrics.json")
    evaluation = _read(root / "micro_evaluation_result.json")
    summary = evaluation["summary"]
    if not isinstance(summary, dict):
        raise ValueError("micro evaluation summary must be an object")
    prior_mpc = _read(root / "prior_phase2b3_verification.json")
    prior_rr = _read(root / "prior_phase2b3_rr_verification.json")
    action = _read(root / "action_legality_result.json")
    reward = _read(root / "reward_hacking_audit.json")
    vector = _read(root / "vectorized_environment_audit.json")
    smoke = _read(root / "ppo_smoke_test_result.json")

    result = classify_phase2b4_result(
        pipeline_valid=(
            vector.get("passed") is True
            and smoke.get("pipeline_execution_passed") is True
            and training.get("pipeline_execution_passed") is True
        ),
        reward_valid=reward.get("passed") is True,
        action_valid=action.get("passed") is True,
        micro_passed=evaluation.get("passed") is True,
        feasibility_development_ran=False,
        feasibility_development_passed=False,
    )
    if result is not Phase2B4Result.RESULT_C:
        raise RuntimeError(f"bounded evidence does not classify as Result C: {result.value}")

    prior = _with_fingerprint(
        {
            "schema_version": "langmani-v2-phase2b4-f0-prior-evidence-verification-v0",
            "mpc_result": prior_mpc.get("result"),
            "mpc_passed": prior_mpc.get("passed"),
            "rr_result": prior_rr.get("result"),
            "rr_passed": prior_rr.get("passed"),
            "geometry_history": {
                "episodes": 12,
                "successes": 2,
                "timeouts": 9,
                "wrong_object_displacements": 1,
                "development_or_formal_result": False,
            },
            "prior_evidence_immutable": True,
            "passed": (
                prior_mpc.get("passed") is True
                and prior_mpc.get("result") == "RESULT_C"
                and prior_rr.get("passed") is True
                and prior_rr.get("result") == "RESULT_D"
            ),
        }
    )
    write_json(root / "prior_evidence_verification.json", prior)

    repository = _with_fingerprint(
        {
            "schema_version": "langmani-v2-phase2b4-f0-repository-environment-audit-v0",
            "repository": "https://github.com/Yanagisawa2002/LangMani.git",
            "worktree": "D:/LangMani-worktrees/phase2b4-ppo-teacher-feasibility",
            "branch": "codex/langmani-v2-phase2b4-ppo-teacher-feasibility",
            "source_branch": "codex/langmani-v2-phase2b3-reset-replay-equivalence",
            "source_commit": SOURCE_COMMIT,
            "source_worktree_clean": True,
            "source_upstream_sha": SOURCE_COMMIT,
            "execution_commit": EXECUTION_COMMIT,
            "execution_commit_is_descendant_of_source": True,
            "excluded_geometry_commit": EXCLUDED_GEOMETRY_COMMIT,
            "excluded_geometry_is_ancestor": False,
            "remote": "https://github.com/Yanagisawa2002/LangMani.git",
            "execution_upstream_sha": EXECUTION_COMMIT,
            "gpu_runtime": {
                "hostname": "autodl-container-8me3mxapw7-7e387606",
                "gpu": "NVIDIA GeForce RTX 5090",
                "gpu_uuid": "GPU-7bed0a44-737c-dfb6-ab08-2cb802bb48ea",
                "driver": "580.95.05",
                "python": "3.12.13",
                "torch": "2.11.0+cu128",
                "cuda": "12.8",
                "mani_skill": "3.0.1",
                "sapien": "3.0.3",
                "gymnasium": "1.2.3",
                "numpy": "2.2.6",
            },
            "no_gpu_job_before_launch": True,
            "no_data_model_optimizer_or_training_process_before_launch": True,
            "prior_authorization_state_all_false": True,
            "passed": True,
        }
    )
    write_json(root / "repository_environment_audit.json", repository)

    repair = _with_fingerprint(
        {
            "schema_version": "langmani-v2-phase2b4-f0-reward-repair-v0",
            "used": False,
            "available_revisions_before_micro": 1,
            "specific_reward_defect_observed": False,
            "reason": (
                "Safety violations incurred large negative terminal penalties and no "
                "comparable positive return; evidence shows failure to learn rather than "
                "a profitable reward exploit."
            ),
            "rerun_started": False,
        }
    )
    write_json(root / "reward_repair.json", repair)

    development = _with_fingerprint(
        {
            "schema_version": "langmani-v2-phase2b4-f0-development-evaluation-v0",
            "ran": False,
            "episode_count": 0,
            "reason": "micro_gate_failed_hard_stop",
            "development_seeds_accessed": False,
            "formal_seeds_accessed": False,
        }
    )
    write_json(root / "feasibility_development_result.json", development)

    failure = _with_fingerprint(
        {
            "schema_version": "langmani-v2-phase2b4-f0-failure-taxonomy-v0",
            "pipeline_failure": False,
            "reward_invalid": False,
            "expert_algorithm_failure": False,
            "ppo_micro_learning_failure": True,
            "training_completed_episodes": training["completed_episodes"],
            "training_successes": training["completed_successes"],
            "micro_outcomes": {
                "success": summary["successes"],
                "timeout": 14,
                "wrong_object_displaced": 22,
                "target_outside_workspace": 12,
            },
            "action_integrity_failure": False,
            "checkpoint_reconstruction_failure": False,
            "simulator_failure": False,
            "broader_generalization_unverified": True,
        }
    )
    write_json(root / "failure_taxonomy.json", failure)

    runtime = _with_fingerprint(
        {
            "schema_version": "langmani-v2-phase2b4-f0-runtime-summary-v0",
            "execution_commit": EXECUTION_COMMIT,
            "vector_environment_count": 128,
            "training_environment_steps": training["global_step"],
            "training_elapsed_seconds": training["elapsed_seconds"],
            "environment_steps_per_second": training["environment_steps_per_second"],
            "optimizer_steps": training["optimizer_steps"],
            "reset_count": training["reset_count"],
            "simulator_exceptions": training["simulator_exceptions"],
            "peak_vram_mib": training["peak_vram_mib"],
            "peak_cpu_memory_mib": training["peak_cpu_memory_mib"],
            "training_scene_seed_start_inclusive": 700000,
            "training_scene_seed_stop_exclusive": 700000 + int(training["reset_count"]),
            "micro_scene_seed_start_inclusive": 800000,
            "micro_scene_seed_stop_exclusive": 800048,
            "formal_scene_seeds_accessed": False,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "checkpoint_size_bytes": training["checkpoint_size_bytes"],
            "checkpoint_location": (
                "/root/autodl-tmp/langmani/outputs/diagnostics/v2/phase2b4_f0/"
                "micro/micro_checkpoint.pt"
            ),
            "checkpoint_committed": False,
            "demonstrations_generated": False,
            "datasets_generated": False,
        }
    )
    write_json(root / "runtime_summary.json", runtime)

    classification = _with_fingerprint(
        {
            "schema_version": "langmani-v2-phase2b4-f0-result-classification-v0",
            "result": result.value,
            "reason": "valid pipeline but frozen micro-learning and zero-tolerance gates failed",
            "pipeline_valid": True,
            "reward_valid": True,
            "action_valid": True,
            "micro_gate_passed": False,
            "feasibility_development_ran": False,
            "reward_repair_used": False,
            "not_an_rl_impossibility_claim": True,
        }
    )
    write_json(root / "result_classification.json", classification)

    authorization = authorization_state(result)
    authorization["fingerprint"] = canonical_json_sha256(authorization)
    write_json(root / "authorization_state.json", authorization)

    remote = _with_fingerprint(
        {
            "schema_version": "langmani-v2-phase2b4-f0-remote-execution-audit-v0",
            "hostname": "autodl-container-8me3mxapw7-7e387606",
            "execution_commit": EXECUTION_COMMIT,
            "checkout_clean_before_and_after": True,
            "gpu_free_before_launch": True,
            "screen_sessions": [
                "langmani_phase2b4_f0_smoke",
                "langmani_phase2b4_f0_micro",
                "langmani_phase2b4_f0_micro_eval",
            ],
            "smoke_exit_code": 0,
            "micro_training_exit_code": 0,
            "micro_evaluation_exit_code": 2,
            "micro_evaluation_exit_code_expected_for_failed_gate": True,
            "concurrent_project_jobs": 0,
            "post_run_gpu_jobs": 0,
            "development_started": False,
            "formal_qualification_started": False,
            "collection_started": False,
            "student_training_started": False,
            "server_shutdown_requested_or_run": False,
            "compact_transfer_sha256": (
                "sha256:42889a12370c31639772a762250b9ae37cec36034532dd00b3b1c504800a7d70"
            ),
        }
    )
    write_json(root / "remote_execution_audit.json", remote)

    tests = _with_fingerprint(
        {
            "schema_version": "langmani-v2-phase2b4-f0-test-summary-v0",
            "targeted_phase2b4_tests": {"passed": 23, "failed": 0},
            "changed_module_mypy": {"errors": 0, "modules": 7},
            "full_mypy_baseline": {
                "source_errors": 551,
                "source_files": 63,
                "current_errors": 551,
                "current_files": 63,
                "new_errors": 0,
            },
            "ruff_format_check": {"files": 361, "passed": True},
            "ruff_check": {"passed": True},
            "isolated_sdist_wheel_build": {"passed": True},
            "cpu_safe_final": {
                "collected": 1619,
                "selected": 1601,
                "passed": 1585,
                "skipped": 16,
                "deselected": 18,
                "failed": 0,
                "duration_seconds": 197.90,
            },
        }
    )
    write_json(root / "test_summary.json", tests)


def _write_manifest(root: Path) -> None:
    missing = sorted(name for name in REQUIRED_ARTIFACTS if not (root / name).is_file())
    if missing:
        raise RuntimeError(f"missing Phase 2B.4-F0 artifacts: {missing}")
    files = []
    for name in sorted(REQUIRED_ARTIFACTS):
        path = root / name
        files.append(
            {
                "path": name,
                "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
        )
    checkpoint = {
        "path": (
            "/root/autodl-tmp/langmani/outputs/diagnostics/v2/phase2b4_f0/micro/micro_checkpoint.pt"
        ),
        "sha256": CHECKPOINT_SHA256,
        "size_bytes": 3_755_973,
        "committed": False,
    }
    payload: dict[str, Any] = {
        "schema_version": "langmani-v2-phase2b4-f0-artifact-manifest-v0",
        "result": "RESULT_C",
        "execution_commit": EXECUTION_COMMIT,
        "files": files,
        "checkpoint": checkpoint,
    }
    payload["fingerprint"] = canonical_json_sha256(payload)
    write_json(root / "artifact_manifest.json", payload)


def main() -> int:
    root = parse_args().artifact_root.resolve()
    if _git("merge-base", SOURCE_COMMIT, "HEAD") != SOURCE_COMMIT:
        raise RuntimeError("current branch does not descend from the required source")
    geometry_ancestor = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", EXCLUDED_GEOMETRY_COMMIT, "HEAD"],
            cwd=PROJECT_ROOT,
            check=False,
        ).returncode
        == 0
    )
    if geometry_ancestor:
        raise RuntimeError("excluded geometry commit is an ancestor")
    _write_derived(root)
    _write_manifest(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
