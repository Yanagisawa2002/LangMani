"""Versioned exclusion and acceptance contracts for Phase 2B.6-v2.

This module contains only deterministic dataset-production logic.  It has no
model, policy, checkpoint, optimizer, backward, planner, or expert dependency.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, cast

from langmani.v2.phase2b5 import canonical_json_sha256
from langmani.v2.phase2b6 import PRIMARY_SPLITS, TASK_IDS

SOURCE_BRANCH: Final = "codex/langmani-v2-phase2b6-1-v2-forensic-replay"
SOURCE_COMMIT: Final = "18c31f8bc6839d80b465ad73894fa886fa6c8dfb"
TARGET_BRANCH: Final = "codex/langmani-v2-phase2b6-v2-instrumented-production"
PACKAGE_ID: Final = "LangManiOfficialMultiSkill-v2"
SPEC_SCHEMA: Final = "langmani-v2-phase2b6-v2-production-spec-v0"
POLICY_SCHEMA: Final = "langmani-v2-phase2b6-v2-exclusion-policy-v0"
EXPECTED_SOURCE_EPISODES: Final = 3_000
EXPECTED_SOURCE_TRANSITIONS: Final = 254_374
REQUIRED_SUB_GATES: Final = (
    "rendering_preflight_gate",
    "environment_construction_gate",
    "reset_identity_gate",
    "action_contract_gate",
    "step_execution_gate",
    "simulator_exception_gate",
    "canonical_terminal_success_gate",
    "source_replay_outcome_agreement_gate",
    "action_count_gate",
    "pre_action_frame_count_gate",
    "observation_action_alignment_gate",
    "rgb_completeness_gate",
    "state_completeness_gate",
    "timestamp_monotonicity_gate",
    "temporary_writer_gate",
    "episode_serialization_gate",
)


class Phase2B6V2ContractError(ValueError):
    """Raised when the frozen v2 contract is malformed or violated."""


class EpisodeClass(StrEnum):
    """The only per-source production classifications."""

    ACCEPTED_REPLAY = "ACCEPTED_REPLAY"
    EXCLUDED_DETERMINISTIC_PHYSICAL = "EXCLUDED_DETERMINISTIC_PHYSICAL"
    EXCLUDED_SOURCE_CONTRACT = "EXCLUDED_SOURCE_CONTRACT"
    RETRYABLE_INFRASTRUCTURE = "RETRYABLE_INFRASTRUCTURE"
    UNCLASSIFIED_FAILURE = "UNCLASSIFIED_FAILURE"


class Phase2B6V2Result(StrEnum):
    """The only terminal phase results."""

    RESULT_A = "RESULT_A"
    RESULT_B = "RESULT_B"
    RESULT_C = "RESULT_C"
    RESULT_D = "RESULT_D"


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2B6V2ContractError(f"failed to load JSON-compatible YAML {path}") from error
    if not isinstance(value, dict):
        raise Phase2B6V2ContractError(f"{path} must contain one object")
    return value


def fingerprinted(payload: Mapping[str, object]) -> dict[str, object]:
    """Return a canonical JSON fingerprinted copy."""

    result = dict(payload)
    if "fingerprint" in result:
        raise Phase2B6V2ContractError("payload is already fingerprinted")
    result["fingerprint"] = canonical_json_sha256(result)
    return result


def portable_text_sha256(path: str | Path) -> str:
    """Hash repository text after normalizing checkout line endings to LF."""

    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def load_exclusion_policy(path: str | Path) -> dict[str, Any]:
    """Load and enforce the preregistered exclusion policy."""

    policy_path = Path(path).resolve()
    policy = _load_object(policy_path)
    if policy.get("schema_version") != POLICY_SCHEMA:
        raise Phase2B6V2ContractError("exclusion policy schema changed")
    classes = policy.get("classes")
    if not isinstance(classes, Mapping) or set(classes) != {item.value for item in EpisodeClass}:
        raise Phase2B6V2ContractError("exclusion classes changed")
    thresholds = policy.get("thresholds")
    expected_thresholds = {
        "overall_accepted_source_rate_minimum": 0.997,
        "per_task_accepted_source_rate_minimum": 0.995,
        "accepted_skill_families_required": 3,
        "maximum_exclusions_per_task": 5,
        "maximum_exclusions_total": 9,
        "unclassified_failures": 0,
        "invalid_actions": 0,
        "non_finite_actions": 0,
        "accepted_episode_simulator_errors": 0,
        "accepted_episode_alignment_failures": 0,
        "accepted_episode_writer_readback_failures": 0,
    }
    if thresholds != expected_thresholds:
        raise Phase2B6V2ContractError("dataset acceptance thresholds changed")
    registered = policy.get("registered_exclusions")
    if not isinstance(registered, list) or len(registered) != 1:
        raise Phase2B6V2ContractError("registered exclusion review set changed")
    episode_938 = cast(Mapping[str, object], registered[0])
    checks = {
        "task": episode_938.get("task_id") == "StackCube-v1",
        "episode": episode_938.get("source_episode_id") == 938,
        "trajectory": episode_938.get("source_trajectory") == "traj_938",
        "seed": episode_938.get("seed") == 962,
        "action": episode_938.get("action_sha256")
        == "sha256:5fc50bc7beeb91e55b2eeccf3f016414811d314142e7a80ad6037d4f83d92eb8",
        "class": episode_938.get("eligible_class")
        == EpisodeClass.EXCLUDED_DETERMINISTIC_PHYSICAL.value,
        "gate": episode_938.get("expected_first_failed_sub_gate")
        == "canonical_terminal_success_gate",
        "normal_attempt": episode_938.get("normal_v2_attempt_required") is True,
        "not_automatic": episode_938.get("automatic_exclusion") is False,
    }
    if not all(checks.values()):
        raise Phase2B6V2ContractError("episode 938 registration changed")
    policy["_path"] = policy_path.as_posix()
    policy["_sha256"] = "sha256:" + portable_text_sha256(policy_path)
    policy["_fingerprint"] = canonical_json_sha256(
        {key: value for key, value in policy.items() if not key.startswith("_")}
    )
    return policy


def load_production_spec(path: str | Path, *, repo_root: str | Path) -> dict[str, Any]:
    """Load the v2 production specification and its bound exclusion policy."""

    spec_path = Path(path).resolve()
    root = Path(repo_root).resolve()
    spec = _load_object(spec_path)
    if spec.get("schema_version") != SPEC_SCHEMA:
        raise Phase2B6V2ContractError("production specification schema changed")
    if spec.get("derived_dataset_version") != PACKAGE_ID:
        raise Phase2B6V2ContractError("dataset package identity changed")
    authorization = spec.get("source_authorization")
    if not isinstance(authorization, Mapping):
        raise Phase2B6V2ContractError("source authorization is malformed")
    if authorization.get("source_branch") != SOURCE_BRANCH:
        raise Phase2B6V2ContractError("source branch changed")
    if authorization.get("source_commit") != SOURCE_COMMIT:
        raise Phase2B6V2ContractError("source commit changed")
    if authorization.get("target_branch") != TARGET_BRANCH:
        raise Phase2B6V2ContractError("target branch changed")
    tasks = spec.get("tasks")
    if not isinstance(tasks, Mapping) or tuple(tasks) != TASK_IDS:
        raise Phase2B6V2ContractError("selected task order changed")
    totals = spec.get("expected_totals")
    if not isinstance(totals, Mapping):
        raise Phase2B6V2ContractError("source totals are malformed")
    if totals.get("source_episodes") != EXPECTED_SOURCE_EPISODES:
        raise Phase2B6V2ContractError("source episode count changed")
    if totals.get("source_transitions") != EXPECTED_SOURCE_TRANSITIONS:
        raise Phase2B6V2ContractError("source transition count changed")
    policy_binding = spec.get("exclusion_policy")
    if not isinstance(policy_binding, Mapping):
        raise Phase2B6V2ContractError("exclusion policy binding is malformed")
    policy_path = root / str(policy_binding.get("relative_path"))
    policy = load_exclusion_policy(policy_path)
    if policy["_sha256"] != policy_binding.get("sha256"):
        raise Phase2B6V2ContractError("exclusion policy bytes changed")
    if policy.get("policy_id") != policy_binding.get("policy_id"):
        raise Phase2B6V2ContractError("exclusion policy identity changed")
    authorization_state = spec.get("authorization")
    if not isinstance(authorization_state, Mapping) or any(
        authorization_state.get(key) is not False
        for key in (
            "training_permitted",
            "optimizer_creation_permitted",
            "backward_pass_permitted",
            "student_policy_training_started",
        )
    ):
        raise Phase2B6V2ContractError("production specification opens model work")
    if authorization_state.get("optimizer_steps") != 0:
        raise Phase2B6V2ContractError("optimizer steps must remain zero")
    spec["_path"] = spec_path.as_posix()
    spec["_sha256"] = "sha256:" + portable_text_sha256(spec_path)
    spec["_policy"] = policy
    return spec


def enforce_starting_point(
    *, source_commit: str, branch: str, source_is_ancestor: bool, no_merges_since_source: bool
) -> None:
    """Enforce the exact isolated v2 Git starting point."""

    if source_commit != SOURCE_COMMIT or branch != TARGET_BRANCH:
        raise Phase2B6V2ContractError("Phase 2B.6-v2 Git identity changed")
    if not source_is_ancestor or not no_merges_since_source:
        raise Phase2B6V2ContractError("Phase 2B.6-v2 ancestry is not isolated")


def gate(status: str, **observed: object) -> dict[str, object]:
    """Create one explicit sub-gate record."""

    if status not in {"passed", "failed", "not_applicable", "not_reached"}:
        raise Phase2B6V2ContractError(f"unknown gate status {status!r}")
    return {"status": status, "observed": observed}


def first_failed_gate(sub_gates: Mapping[str, Mapping[str, object]]) -> str | None:
    """Return the first failed gate in the frozen order."""

    if set(sub_gates) != set(REQUIRED_SUB_GATES):
        raise Phase2B6V2ContractError("explicit sub-gate set changed")
    for name in REQUIRED_SUB_GATES:
        if sub_gates[name].get("status") == "failed":
            return name
    return None


def classify_attempt(
    *,
    sub_gates: Mapping[str, Mapping[str, object]],
    full_action_execution: bool,
    retryable_reason: str | None = None,
) -> EpisodeClass:
    """Classify one attempt without silently discarding it."""

    failed = first_failed_gate(sub_gates)
    if failed is None and all(
        sub_gates[name].get("status") == "passed" for name in REQUIRED_SUB_GATES
    ):
        return EpisodeClass.ACCEPTED_REPLAY
    if failed in {"reset_identity_gate", "action_contract_gate"}:
        return EpisodeClass.EXCLUDED_SOURCE_CONTRACT
    if (
        failed
        in {
            "canonical_terminal_success_gate",
            "source_replay_outcome_agreement_gate",
        }
        and full_action_execution
        and sub_gates["simulator_exception_gate"].get("status") == "passed"
    ):
        return EpisodeClass.EXCLUDED_DETERMINISTIC_PHYSICAL
    if failed in {
        "rendering_preflight_gate",
        "environment_construction_gate",
        "temporary_writer_gate",
        "episode_serialization_gate",
    } and retryable_reason in {
        "environment_construction",
        "temporary_vulkan_initialization",
        "transient_file_access",
        "temporary_writer_initialization",
        "temporary_storage_access",
    }:
        return EpisodeClass.RETRYABLE_INFRASTRUCTURE
    return EpisodeClass.UNCLASSIFIED_FAILURE


def accepted_assignments(
    assignments: Sequence[Mapping[str, object]],
    attempt_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Filter accepted policy identities without replacement sampling."""

    accepted_ids = {
        str(row["derived_episode_identity"])
        for row in attempt_rows
        if row.get("classification") == EpisodeClass.ACCEPTED_REPLAY.value
    }
    rows = [
        dict(row) for row in assignments if str(row["derived_episode_identity"]) in accepted_ids
    ]
    if len(rows) != len(accepted_ids):
        raise Phase2B6V2ContractError("accepted replay identity is absent from source assignments")
    return rows


def threshold_audit(attempt_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Apply the frozen source-accounting and exclusion thresholds."""

    terminal_rows = [row for row in attempt_rows if row.get("terminal_attempt") is True]
    if len(terminal_rows) != EXPECTED_SOURCE_EPISODES:
        raise Phase2B6V2ContractError("threshold audit requires all 3,000 terminal classifications")
    classes = Counter(str(row.get("classification")) for row in terminal_rows)
    per_task: dict[str, object] = {}
    total_accepted = 0
    total_excluded = 0
    for task_id in TASK_IDS:
        rows = [row for row in terminal_rows if row.get("task_id") == task_id]
        accepted = sum(
            row.get("classification") == EpisodeClass.ACCEPTED_REPLAY.value for row in rows
        )
        excluded = len(rows) - accepted
        total_accepted += accepted
        total_excluded += excluded
        per_task[task_id] = {
            "source_count": len(rows),
            "accepted_count": accepted,
            "excluded_count": excluded,
            "accepted_source_rate": accepted / len(rows),
            "passed": (len(rows) == 1_000 and accepted / len(rows) >= 0.995 and excluded <= 5),
        }
    checks = {
        "all_source_identities_accounted": len(terminal_rows) == EXPECTED_SOURCE_EPISODES,
        "overall_accepted_source_rate": total_accepted / EXPECTED_SOURCE_EPISODES >= 0.997,
        "per_task_accepted_source_rate": all(
            cast(Mapping[str, object], report)["passed"] is True for report in per_task.values()
        ),
        "maximum_total_exclusions": total_excluded <= 9,
        "unclassified_failures": classes[EpisodeClass.UNCLASSIFIED_FAILURE.value] == 0,
        "retryable_failures_terminal": classes[EpisodeClass.RETRYABLE_INFRASTRUCTURE.value] == 0,
    }
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-v2-threshold-audit-v0",
            "source_count": EXPECTED_SOURCE_EPISODES,
            "accepted_count": total_accepted,
            "excluded_count": total_excluded,
            "accepted_source_rate": total_accepted / EXPECTED_SOURCE_EPISODES,
            "class_counts": dict(classes),
            "per_task": per_task,
            "checks": checks,
            "passed": all(checks.values()),
        }
    )


def split_count_report(
    assignments: Sequence[Mapping[str, object]], accepted: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Report source and accepted counts per frozen split without replacement."""

    accepted_ids = {str(row["derived_episode_identity"]) for row in accepted}
    rows: dict[str, object] = {}
    for task_id in TASK_IDS:
        rows[task_id] = {
            split: {
                "source_count": sum(
                    row.get("task_id") == task_id and row.get("primary_split") == split
                    for row in assignments
                ),
                "accepted_count": sum(
                    row.get("task_id") == task_id
                    and row.get("primary_split") == split
                    and str(row.get("derived_episode_identity")) in accepted_ids
                    for row in assignments
                ),
            }
            for split in PRIMARY_SPLITS
        }
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-v2-split-counts-v0",
            "tasks": rows,
            "replacement_sampling": False,
            "excluded_sources_remain_in_source_inventory": True,
            "passed": len(accepted_ids) == len(accepted),
        }
    )


def authorization_state(*, accepted: bool) -> dict[str, object]:
    """Expose eligibility separately from immutable no-training authorization."""

    return {
        "schema_version": "langmani-v2-phase2b6-v2-authorization-state-v0",
        "accepted_multiskill_dataset_validated": accepted,
        "act_baseline_training_eligible": accepted,
        "smolvla_training_eligible": accepted,
        "vla_jepa_training_eligible": accepted,
        "eligibility_is_authorization": False,
        "act_training_authorized": False,
        "smolvla_training_authorized": False,
        "vla_jepa_training_authorized": False,
        "student_policy_training_started": False,
        "optimizer_created": False,
        "backward_passes": 0,
        "optimizer_steps": 0,
    }


def classify_result(
    *,
    integrity_valid: bool,
    threshold_valid: bool,
    archive_created: bool,
    restore_valid: bool,
) -> Phase2B6V2Result:
    """Apply Result A/B/C/D without converting eligibility into authorization."""

    if not integrity_valid:
        return Phase2B6V2Result.RESULT_C
    if not threshold_valid:
        return Phase2B6V2Result.RESULT_B
    if not archive_created or not restore_valid:
        return Phase2B6V2Result.RESULT_D
    return Phase2B6V2Result.RESULT_A


def file_sha256(path: str | Path) -> str:
    """Return a tagged SHA-256 for a regular file."""

    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise Phase2B6V2ContractError(f"expected ordinary file: {target}")
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


__all__ = [
    "EXPECTED_SOURCE_EPISODES",
    "EXPECTED_SOURCE_TRANSITIONS",
    "PACKAGE_ID",
    "POLICY_SCHEMA",
    "REQUIRED_SUB_GATES",
    "SOURCE_BRANCH",
    "SOURCE_COMMIT",
    "SPEC_SCHEMA",
    "TARGET_BRANCH",
    "EpisodeClass",
    "Phase2B6V2ContractError",
    "Phase2B6V2Result",
    "accepted_assignments",
    "authorization_state",
    "classify_attempt",
    "classify_result",
    "enforce_starting_point",
    "file_sha256",
    "fingerprinted",
    "first_failed_gate",
    "gate",
    "load_exclusion_policy",
    "load_production_spec",
    "portable_text_sha256",
    "split_count_report",
    "threshold_audit",
]
