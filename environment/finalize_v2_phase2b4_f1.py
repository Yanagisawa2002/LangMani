"""Finalize compact Phase 2B.4-F1 evidence after the bounded hard stop."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

from langmani.v2.phase2b4_evidence import canonical_git_text_bytes
from langmani.v2.phase2b4_f1 import (
    Phase2B4F1Result,
    authorization_state,
    canonical_json_sha256,
    classify_f1_result,
    probe_configuration,
)
from langmani.v2.phase2b4_f1_runtime import reconstruct_f1_policy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "diagnostics" / "v2" / "phase2b4_f1"
DEFAULT_PREFLIGHT_ROOT = DEFAULT_OUTPUT_ROOT / "preflight_final"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2b4_f1"
F0_ROOT = PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2b4_f0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--preflight-root", type=Path, default=DEFAULT_PREFLIGHT_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--test-summary", type=Path)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fingerprinted(payload: dict[str, object]) -> dict[str, object]:
    payload["fingerprint"] = canonical_json_sha256(payload)
    return payload


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
    ).strip()


def _copy_json(source: Path, destination: Path) -> dict[str, Any]:
    payload = _read(source)
    _write(destination, payload)
    return payload


def _write_manifest(
    *,
    root: Path,
    result: Phase2B4F1Result,
    execution_commit: str,
    checkpoint: Path,
) -> None:
    files = []
    for path in sorted(root.glob("*.json")):
        if path.name == "artifact_manifest.json":
            continue
        canonical = canonical_git_text_bytes(path)
        files.append(
            {
                "path": path.name,
                "sha256": "sha256:" + hashlib.sha256(canonical).hexdigest(),
                "size_bytes": len(canonical),
            }
        )
    checkpoint_bytes = checkpoint.read_bytes()
    payload: dict[str, object] = {
        "schema_version": "langmani-v2-phase2b4-f1-artifact-manifest-v0",
        "result": result.value,
        "execution_commit": execution_commit,
        "finalizer_commit": _git("rev-parse", "HEAD"),
        "text_byte_contract": ("UTF-8 JSON with CRLF normalized to LF, matching Git text bytes"),
        "files": files,
        "checkpoint": {
            "path": checkpoint.as_posix(),
            "sha256": "sha256:" + hashlib.sha256(checkpoint_bytes).hexdigest(),
            "size_bytes": len(checkpoint_bytes),
            "committed": False,
        },
    }
    payload["fingerprint"] = canonical_json_sha256(payload)
    _write(root / "artifact_manifest.json", payload)


def main() -> int:
    args = parse_args()
    output = args.output_root.resolve()
    preflight = args.preflight_root.resolve()
    root = args.artifact_root.resolve()
    checkpoint = output / "probe_a" / "probe_a_checkpoint.pt"
    probe_a_training = _read(output / "probe_a_training_metrics.json")
    probe_a_evaluation = _read(output / "probe_a_evaluation.json")
    exploration = _read(output / "exploration_distribution_audit.json")
    contact = _read(output / "contact_credit_audit.json")
    comparison = _read(output / "global_versus_residual_comparison.json")
    reward_decision = _read(output / "reward_decision.json")
    f0_training = _read(F0_ROOT / "micro_training_metrics.json")
    if probe_a_training.get("global_step") != 262_144:
        raise RuntimeError("Probe A did not use its exact frozen budget")
    if exploration.get("passed") is not True:
        raise RuntimeError("residual action legality did not pass")
    if comparison.get("residual_materially_safer") is not True:
        raise RuntimeError("residual physical comparison did not pass")
    if probe_a_evaluation.get("passed") is not False:
        raise RuntimeError("this finalizer expects the observed Probe A hard stop")
    if (output / "probe_b_training_metrics.json").exists():
        raise RuntimeError("Probe B artifact exists despite failed Probe A")
    result = classify_f1_result(
        diagnosis_valid=True,
        pipeline_valid=True,
        residual_exploration_safer=True,
        probe_a_passed=False,
        probe_b_ran=False,
        probe_b_passed=False,
    )
    if result is not Phase2B4F1Result.RESULT_C:
        raise RuntimeError(f"unexpected F1 result: {result.value}")

    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    preflight_files = (
        "repository_environment_audit.json",
        "prior_evidence_verification.json",
        "f0_learning_dynamics_audit.json",
        "termination_bootstrap_audit.json",
        "safe_curriculum_manifest.json",
        "seed_disjointness_audit.json",
        "probe_a_configuration.json",
        "operation_boundary.json",
    )
    for name in preflight_files:
        _copy_json(preflight / name, root / name)
    _write(root / "exploration_distribution_audit.json", exploration)
    _write(root / "contact_credit_audit.json", contact)
    _write(root / "global_versus_residual_comparison.json", comparison)
    _write(root / "reward_decision.json", reward_decision)
    _write(root / "probe_a_training_metrics.json", probe_a_training)
    _write(root / "probe_a_evaluation.json", probe_a_evaluation)
    policy, checkpoint_payload = reconstruct_f1_policy(checkpoint)
    _write(root / "residual_action_transform_manifest.json", policy.action_manifest())
    legality = exploration.get("residual_legality")
    if not isinstance(legality, dict):
        raise RuntimeError("exploration report lacks residual legality evidence")
    _write(root / "residual_action_legality_audit.json", legality)
    not_authorized = _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b4-f1-probe-b-not-authorized-v0",
            "authorized": False,
            "ran": False,
            "training_steps": 0,
            "evaluation_episodes": 0,
            "reason": "probe_a_frozen_gate_failed",
            "probe_a_gate": probe_a_evaluation["gate"],
            "configuration_if_future_authorized": probe_configuration(probe="B"),
        }
    )
    _write(root / "probe_b_not_authorized.json", not_authorized)
    comparison_payload = _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b4-f1-f0-f1-comparison-v0",
            "f0": {
                "training_steps": f0_training["global_step"],
                "training_successes": f0_training["completed_successes"],
                "evaluation_episodes": 48,
                "evaluation_successes": 0,
                "wrong_object_interactions": 22,
                "workspace_exits": 12,
                "initial_arm_target_l2_mean": exploration["global_f0_initial"][
                    "arm_joint_target_l2"
                ]["mean"],
                "initial_max_tcp_step_mean": comparison["global_f0"][
                    "maximum_tcp_step_translation"
                ]["mean"],
            },
            "f1_probe_a": {
                "training_steps": probe_a_training["global_step"],
                "training_successes": probe_a_training["completed_successes"],
                "evaluation_episodes": probe_a_evaluation["summary"]["episode_count"],
                "evaluation_successes": probe_a_evaluation["summary"]["successes"],
                "correct_contact_episodes": probe_a_evaluation["summary"][
                    "correct_contact_episodes"
                ],
                "target_progress_episodes": probe_a_evaluation["summary"][
                    "target_progress_episodes"
                ],
                "wrong_object_interactions": probe_a_evaluation["summary"][
                    "wrong_object_interactions"
                ],
                "workspace_exits": probe_a_evaluation["summary"]["workspace_exits"],
                "initial_arm_target_l2_mean": exploration["state_centered_residual_initial"][
                    "arm_joint_target_l2"
                ]["mean"],
                "initial_max_tcp_step_mean": comparison["residual_f1"][
                    "maximum_tcp_step_translation"
                ]["mean"],
            },
            "mechanism_interpretation": (
                "State-centered residual mapping materially reduced initial local motion, "
                "but actor initialization/log-std and Stage-0 task distribution also "
                "changed, so causal attribution cannot be isolated. The redesign still "
                "failed to discover correct contact and retained wrong-object interaction."
            ),
            "unequal_training_budget_warning": (
                "F0 used 1,048,576 steps and Probe A used 262,144; final task metrics "
                "are diagnostic comparisons, not equal-budget performance estimates."
            ),
        }
    )
    _write(root / "f0_f1_comparison.json", comparison_payload)
    failure = _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b4-f1-failure-taxonomy-v0",
            "result": result.value,
            "infrastructure_failure": False,
            "pipeline_invalidity": False,
            "action_integrity_failure": False,
            "residual_exploration_locality_failure": False,
            "ppo_teacher_algorithm_failure": True,
            "correct_contact_learning_failure": True,
            "target_progress_learning_failure": True,
            "probe_a_safety_failure": True,
            "probe_b_not_authorized": True,
            "formal_qualification_not_run": True,
            "collection_not_run": True,
            "student_training_not_run": True,
        }
    )
    _write(root / "failure_taxonomy.json", failure)
    runtime = _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b4-f1-runtime-summary-v0",
            "execution_commit": args.execution_commit,
            "runtime": probe_a_training["runtime"],
            "probe_a_environment_steps": probe_a_training["global_step"],
            "probe_a_elapsed_seconds": probe_a_training["elapsed_seconds"],
            "probe_a_environment_steps_per_second": probe_a_training[
                "environment_steps_per_second"
            ],
            "probe_a_optimizer_steps": probe_a_training["optimizer_steps"],
            "probe_a_completed_episodes": probe_a_training["completed_episodes"],
            "probe_a_training_successes": probe_a_training["completed_successes"],
            "probe_a_evaluation_episodes": probe_a_evaluation["summary"]["episode_count"],
            "probe_b_environment_steps": 0,
            "combined_f1_training_steps": probe_a_training["global_step"],
            "combined_budget_limit": 524_288,
            "peak_vram_mib": probe_a_training["peak_vram_mib"],
            "checkpoint_global_step": checkpoint_payload["global_step"],
            "checkpoint_sha256": probe_a_training["checkpoint_sha256"],
            "checkpoint_size_bytes": probe_a_training["checkpoint_size_bytes"],
            "formal_seeds_accessed": False,
            "demonstrations_generated": False,
            "datasets_generated": False,
        }
    )
    _write(root / "runtime_summary.json", runtime)
    classification = _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b4-f1-result-classification-v0",
            "result": result.value,
            "diagnosis_valid": True,
            "pipeline_valid": True,
            "residual_exploration_materially_safer": True,
            "probe_a_passed": False,
            "probe_b_authorized": False,
            "probe_b_ran": False,
            "hard_stop_reason": (
                "Probe A evaluation had seven wrong-object interactions, zero correct "
                "contacts, and zero target-progress episodes."
            ),
            "custom_ppo_teacher_route_frozen": True,
            "another_ppo_iteration_prohibited": True,
            "recommended_next_method": (
                "Use a standard task with an existing validated expert or official "
                "demonstrations; alternatively use separately validated teleoperation."
            ),
        }
    )
    _write(root / "result_classification.json", classification)
    _write(root / "authorization_state.json", authorization_state(result))
    remote = _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b4-f1-remote-execution-audit-v0",
            "hostname": platform.node(),
            "execution_commit": args.execution_commit,
            "checkout_clean_before_each_target_stage": True,
            "network_turbo_sourced": True,
            "vulkan_icd": "/etc/vulkan/icd.d/my_nvidia_icd.json",
            "screen_sessions": ["langmani_phase2b4_f1_probe_a"],
            "probe_a_exit_code": 0,
            "probe_a_evaluation_exit_code": 2,
            "probe_a_evaluation_exit_expected_for_failed_gate": True,
            "probe_b_started": False,
            "concurrent_project_jobs": 0,
            "post_run_gpu_jobs": 0,
            "formal_qualification_started": False,
            "collection_started": False,
            "student_training_started": False,
            "server_shutdown_requested_or_run": False,
        }
    )
    _write(root / "remote_execution_audit.json", remote)
    if args.test_summary is not None:
        _copy_json(args.test_summary.resolve(), root / "test_summary.json")
    _write_manifest(
        root=root,
        result=result,
        execution_commit=args.execution_commit,
        checkpoint=checkpoint,
    )
    print(
        json.dumps(
            {
                "artifact_root": root.as_posix(),
                "result": result.value,
                "files": len(list(root.glob("*.json"))),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
