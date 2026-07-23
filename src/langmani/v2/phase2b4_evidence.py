"""Independent compact-evidence verification for Phase 2B.4-F0."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Final

REQUIRED_ARTIFACTS: Final = frozenset(
    {
        "action_legality_result.json",
        "action_transform_manifest.json",
        "authorization_state.json",
        "failure_taxonomy.json",
        "feasibility_development_result.json",
        "micro_evaluation_result.json",
        "micro_training_configuration.json",
        "micro_training_metrics.json",
        "operation_boundary.json",
        "ppo_implementation_audit.json",
        "ppo_smoke_test_result.json",
        "prior_evidence_verification.json",
        "prior_phase2b3_rr_verification.json",
        "prior_phase2b3_verification.json",
        "remote_execution_audit.json",
        "repository_environment_audit.json",
        "result_classification.json",
        "reward_hacking_audit.json",
        "reward_manifest.json",
        "reward_repair.json",
        "runtime_summary.json",
        "seed_disjointness_audit.json",
        "state_normalization_manifest.json",
        "student_privilege_exclusion_audit.json",
        "teacher_observation_manifest.json",
        "test_summary.json",
        "vectorized_environment_audit.json",
    }
)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(value, path.name)


def _safe_artifact_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 1:
        raise ValueError(f"unsafe Phase 2B.4-F0 artifact path {relative!r}")
    path = root / pure.name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or linked Phase 2B.4-F0 artifact {relative!r}")
    return path


def verify_phase2b4_f0_artifacts(root: Path) -> dict[str, object]:
    root = root.resolve()
    manifest = _read_json(root / "artifact_manifest.json")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("artifact manifest files must be a list")
    hash_checks: dict[str, bool] = {}
    size_checks: dict[str, bool] = {}
    declared: set[str] = set()
    for raw_entry in entries:
        entry = _mapping(raw_entry, "artifact manifest entry")
        relative = str(entry.get("path"))
        if relative in declared:
            raise ValueError(f"duplicate artifact path {relative!r}")
        declared.add(relative)
        path = _safe_artifact_path(root, relative)
        digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        hash_checks[relative] = digest == entry.get("sha256")
        size_checks[relative] = path.stat().st_size == entry.get("size_bytes")

    documents = {name: _read_json(root / name) for name in REQUIRED_ARTIFACTS}
    action_audit = documents["action_legality_result.json"]
    action_manifest = documents["action_transform_manifest.json"]
    authorization = documents["authorization_state.json"]
    development = documents["feasibility_development_result.json"]
    micro_evaluation = documents["micro_evaluation_result.json"]
    micro_training = documents["micro_training_metrics.json"]
    operation_boundary = documents["operation_boundary.json"]
    prior_mpc = documents["prior_phase2b3_verification.json"]
    prior_rr = documents["prior_phase2b3_rr_verification.json"]
    repository = documents["repository_environment_audit.json"]
    result = documents["result_classification.json"]
    reward_audit = documents["reward_hacking_audit.json"]
    reward_repair = documents["reward_repair.json"]
    seed_audit = documents["seed_disjointness_audit.json"]
    smoke = documents["ppo_smoke_test_result.json"]
    student = documents["student_privilege_exclusion_audit.json"]
    teacher = documents["teacher_observation_manifest.json"]
    vector = documents["vectorized_environment_audit.json"]

    micro_summary = _mapping(micro_evaluation.get("summary"), "micro summary")
    safety_counts = _mapping(micro_summary.get("safety_counts"), "micro safety counts")
    checks = {
        "artifact_set_exact": declared == REQUIRED_ARTIFACTS,
        "all_hashes_valid": all(hash_checks.values()),
        "all_sizes_valid": all(size_checks.values()),
        "repository_isolated": repository.get("passed") is True,
        "prior_mpc_result_c_verified": (
            prior_mpc.get("passed") is True and prior_mpc.get("result") == "RESULT_C"
        ),
        "prior_rr_result_d_verified": (
            prior_rr.get("passed") is True and prior_rr.get("result") == "RESULT_D"
        ),
        "teacher_observation_frozen": (
            teacher.get("dimension") == 87 and teacher.get("current_state_only") is True
        ),
        "student_privilege_excluded": student.get("passed") is True,
        "action_transform_structurally_bounded": (
            action_manifest.get("clipping") is False
            and action_manifest.get("projection") is False
            and action_manifest.get("fallback") is False
        ),
        "action_audit_passed": (
            action_audit.get("passed") is True
            and action_audit.get("sample_count") == 100000
            and action_audit.get("nonfinite_actions") == 0
            and action_audit.get("out_of_bounds_actions") == 0
        ),
        "reward_audit_passed_without_repair": (
            reward_audit.get("passed") is True
            and reward_audit.get("reward_revision_used") == 0
            and reward_repair.get("used") is False
        ),
        "seed_audit_passed": (
            seed_audit.get("passed") is True and seed_audit.get("formal_seed_accessed") is False
        ),
        "vector_audit_passed": vector.get("passed") is True,
        "ppo_smoke_passed": (
            smoke.get("pipeline_execution_passed") is True
            and smoke.get("finite_losses") is True
            and smoke.get("finite_gradients") is True
            and smoke.get("deterministic_reload_max_abs_error") == 0.0
        ),
        "micro_budget_exact": micro_training.get("global_step") == 1_048_576,
        "micro_pipeline_completed": micro_training.get("pipeline_execution_passed") is True,
        "micro_gate_failed": micro_evaluation.get("passed") is False,
        "micro_episode_count_exact": micro_summary.get("episode_count") == 48,
        "micro_success_zero": micro_summary.get("successes") == 0,
        "micro_safety_failure_recorded": sum(int(value) for value in safety_counts.values()) > 0,
        "development_hard_stopped": development.get("ran") is False,
        "result_c_classified": result.get("result") == "RESULT_C",
        "all_authorizations_false": all(
            authorization.get(key) is False
            for key in (
                "ppo_full_training_eligible",
                "ppo_full_training_authorized",
                "expert_qualification_authorized",
                "data_collection_authorized",
                "smolvla_training_authorized",
                "training_started_for_student_policy",
                "demonstration_source_validated",
            )
        ),
        "prohibited_operations_not_started": (
            operation_boundary.get("demonstrations_generated") is False
            and operation_boundary.get("datasets_created") is False
            and operation_boundary.get("student_training_started") is False
            and operation_boundary.get("formal_qualification_started") is False
        ),
    }
    return {
        "schema_version": "langmani-v2-phase2b4-f0-independent-verification-v0",
        "result": result.get("result"),
        "checks": checks,
        "artifact_hash_checks": hash_checks,
        "artifact_size_checks": size_checks,
        "passed": all(checks.values()),
        "ppo_full_training_eligible": authorization.get("ppo_full_training_eligible"),
        "ppo_full_training_authorized": authorization.get("ppo_full_training_authorized"),
        "expert_qualification_authorized": authorization.get("expert_qualification_authorized"),
        "data_collection_authorized": authorization.get("data_collection_authorized"),
        "smolvla_training_authorized": authorization.get("smolvla_training_authorized"),
    }


__all__ = ["REQUIRED_ARTIFACTS", "verify_phase2b4_f0_artifacts"]
