"""Final acceptance and compact verification for Phase 2B.6-v2."""

from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from langmani.v2.phase2b5 import canonical_json_sha256, sha256_file
from langmani.v2.phase2b6_lerobot import lerobot_environment_manifest
from langmani.v2.phase2b6_v2 import (
    PACKAGE_ID,
    Phase2B6V2Result,
    authorization_state,
    classify_result,
    fingerprinted,
)


class Phase2B6V2FinalizeError(RuntimeError):
    """Raised when final evidence cannot satisfy the frozen gate."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2B6V2FinalizeError(f"failed to read {path}") from error
    if not isinstance(value, dict):
        raise Phase2B6V2FinalizeError(f"{path} must contain one object")
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.parent / f".{path.name}.partial"
    staging.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    staging.replace(path)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise Phase2B6V2FinalizeError(f"git {' '.join(args)} failed")
    return result.stdout.strip()


def finalize_acceptance(
    *,
    repo_root: Path,
    production_root: Path,
    evidence_root: Path,
    command_log: Sequence[str],
) -> dict[str, object]:
    """Create the accepted package only after every required report passes."""

    reports = {
        name: _read_json(evidence_root / filename)
        for name, filename in {
            "source": "source_integrity_audit.json",
            "source_schema": "source_schema_result.json",
            "production": "production_run_manifest.json",
            "accepted": "accepted_episode_manifest.json",
            "excluded": "excluded_episode_manifest.json",
            "thresholds": "exclusion_threshold_audit.json",
            "accounting": "source_accounting_verification.json",
            "task_roots": "task_specific_root_manifests.json",
            "unified": "unified_multi_root_manifest.json",
            "readback": "full_readback_result.json",
            "equality": "source_to_derived_verifier_result.json",
            "leakage": "leakage_audit.json",
            "padding": "padding_audit.json",
            "statistics": "dataset_statistics.json",
            "normalization": "normalization_manifest.json",
            "observation": "observation_action_contract.json",
            "visual": "visual_shift_manifest.json",
            "folds": "cross_skill_fold_manifests.json",
            "archive": "archive_manifest.json",
            "restore": "restore_validation_result.json",
        }.items()
    }
    production = reports["production"]
    thresholds = reports["thresholds"]
    accepted = reports["accepted"]
    excluded = reports["excluded"]
    readback = reports["readback"]
    gates = {
        "three_skill_families_retained": len(
            cast(Mapping[str, object], reports["task_roots"]["task_roots"])
        )
        == 3,
        "accepted_source_thresholds": thresholds["passed"] is True,
        "all_3000_sources_accounted": (
            reports["accounting"]["passed"] is True
            and reports["accounting"]["source_episode_count"] == 3_000
        ),
        "unclassified_failures_zero": cast(Mapping[str, object], thresholds["class_counts"]).get(
            "UNCLASSIFIED_FAILURE", 0
        )
        == 0,
        "accepted_replay_outcome_agreement": production["passed"] is True,
        "accepted_action_frame_alignment": all(
            row.get("action_frame_alignment") is True
            for row in _terminal_accepted_rows(production_root)
        ),
        "invalid_and_nonfinite_actions_zero": all(
            row.get("invalid_action_count") == 0 and row.get("nonfinite_value_count") == 0
            for row in _terminal_accepted_rows(production_root)
        ),
        "accepted_simulator_errors_zero": all(
            row.get("simulator_error_count") == 0
            for row in _terminal_accepted_rows(production_root)
        ),
        "privileged_fields_excluded": reports["observation"]["passed"] is True,
        "lerobot_full_readback": readback["passed"] is True,
        "source_to_derived_equality": reports["equality"]["passed"] is True,
        "primary_split_leakage_zero": (
            reports["leakage"]["passed"] is True
            and reports["leakage"]["primary_split_overlap_count"] == 0
        ),
        "train_only_normalization": (
            reports["normalization"]["passed"] is True
            and reports["normalization"]["source_view"] == "primary_train_only"
        ),
        "padding_masks_verified": reports["padding"]["passed"] is True,
        "archive_created": reports["archive"]["passed"] is True,
        "restore_validated": reports["restore"]["passed"] is True,
    }
    integrity_valid = (
        all(
            report["passed"] is True for name, report in reports.items() if name not in {"excluded"}
        )
        and reports["excluded"]["passed"] is True
    )
    result = classify_result(
        integrity_valid=integrity_valid,
        threshold_valid=thresholds["passed"] is True,
        archive_created=reports["archive"]["passed"] is True,
        restore_valid=reports["restore"]["passed"] is True,
    )
    if result is not Phase2B6V2Result.RESULT_A or not all(gates.values()):
        raise Phase2B6V2FinalizeError(f"accepted package gate failed with {result.value}: {gates}")
    package = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-v2-accepted-package-v0",
            "package_identity": PACKAGE_ID,
            "source_episode_count": 3_000,
            "accepted_episode_count": accepted["episode_count"],
            "accepted_frame_count": accepted["frame_count"],
            "excluded_episode_count": excluded["episode_count"],
            "exclusions": [
                {
                    "task_id": row["task_id"],
                    "source_episode_id": row["source_episode_id"],
                    "classification": row["classification"],
                    "first_failed_sub_gate": row["first_failed_sub_gate"],
                    "source_trajectory_identity": row["source_trajectory_identity"],
                    "source_action_sha256": row["source_action_sha256"],
                }
                for row in cast(list[dict[str, object]], excluded["episodes"])
            ],
            "tasks": reports["task_roots"]["task_roots"],
            "unified_multi_root_fingerprint": reports["unified"]["fingerprint"],
            "language_manifest": "language_manifest.json",
            "primary_source_split_manifest": "primary_split_manifest.json",
            "accepted_split_manifest": "accepted_primary_split_manifest.json",
            "visual_shift_fingerprint": reports["visual"]["fingerprint"],
            "cross_skill_fold_fingerprint": reports["folds"]["fingerprint"],
            "padding_fingerprint": reports["padding"]["fingerprint"],
            "statistics_fingerprint": reports["statistics"]["fingerprint"],
            "normalization_fingerprint": reports["normalization"]["fingerprint"],
            "readback_fingerprint": reports["readback"]["fingerprint"],
            "source_to_derived_fingerprint": reports["equality"]["fingerprint"],
            "leakage_fingerprint": reports["leakage"]["fingerprint"],
            "archive": {
                "location": reports["archive"]["content_addressed_archive_location"],
                "sha256": reports["archive"]["archive_sha256"],
                "primary_tree_digest": reports["archive"]["primary_tree_digest"],
            },
            "restore": {
                "location": reports["restore"]["restored_location"],
                "tree_digest": reports["restore"]["restored_tree_digest"],
                "fingerprint": reports["restore"]["fingerprint"],
            },
            "gates": gates,
            "contains_trained_model": False,
            "student_policy_training_started": False,
            "optimizer_steps": 0,
        }
    )
    authorization = fingerprinted(authorization_state(accepted=True))
    result_report = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-v2-result-v0",
            "result": result.value,
            "label": "Versioned multi-skill dataset accepted",
            "accepted_package_fingerprint": package["fingerprint"],
            "gates": gates,
            "accepted_multiskill_dataset_validated": True,
            "student_policy_training_started": False,
            "optimizer_created": False,
            "backward_passes": 0,
            "optimizer_steps": 0,
            "passed": True,
        }
    )
    remote = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-v2-remote-execution-v0",
            "hostname": platform.node(),
            "producer_commit": _git(repo_root, "rev-parse", "HEAD"),
            "producer_branch": _git(repo_root, "branch", "--show-current"),
            "producer_worktree_clean": not _git(repo_root, "status", "--porcelain=v1"),
            "lerobot_environment": lerobot_environment_manifest(),
            "commands": list(command_log),
            "official_recorded_actions_only": True,
            "previous_partial_records_reused": False,
            "policy_model_loaded": False,
            "optimizer_created": False,
            "backward_passes": 0,
            "optimizer_steps": 0,
            "student_policy_training_started": False,
            "passed": True,
        }
    )
    for name, document in {
        "accepted_multiskill_dataset_package.json": package,
        "result_classification.json": result_report,
        "eligibility_state.json": authorization,
        "authorization_state_final.json": authorization,
        "remote_execution_audit.json": remote,
    }.items():
        _write_json(evidence_root / name, document)
    _write_json(
        production_root / "primary" / "metadata" / "accepted_multiskill_dataset_package.json",
        package,
    )
    return result_report


def _terminal_accepted_rows(production_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    path = production_root / "work" / "production_attempts.jsonl"
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if (
                isinstance(value, dict)
                and value.get("terminal_attempt") is True
                and value.get("classification") == "ACCEPTED_REPLAY"
            ):
                rows.append(cast(dict[str, object], value))
    return rows


def write_artifact_manifest(evidence_root: Path) -> dict[str, object]:
    """Hash every compact JSON artifact and reject non-JSON evidence."""

    manifest_path = evidence_root / "artifact_manifest.json"
    if manifest_path.exists():
        raise Phase2B6V2FinalizeError("artifact manifest already exists")
    entries: list[dict[str, object]] = []
    for path in sorted(evidence_root.iterdir(), key=lambda item: item.name):
        if path.name.startswith("."):
            continue
        if path.suffix != ".json" or path.is_symlink() or not path.is_file():
            raise Phase2B6V2FinalizeError(f"unexpected compact evidence file {path}")
        entries.append(
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": "sha256:" + sha256_file(path),
            }
        )
    required = {
        "repository_environment_audit.json",
        "prior_evidence_verification.json",
        "exclusion_policy_manifest.json",
        "frozen_production_specification.json",
        "source_inventory.json",
        "production_run_manifest.json",
        "accepted_episode_manifest.json",
        "excluded_episode_manifest.json",
        "explicit_sub_gate_summary.json",
        "task_specific_root_manifests.json",
        "unified_multi_root_manifest.json",
        "language_manifest.json",
        "primary_split_manifest.json",
        "visual_shift_manifest.json",
        "cross_skill_fold_manifests.json",
        "padding_audit.json",
        "dataset_statistics.json",
        "normalization_manifest.json",
        "task_balance_manifest.json",
        "full_readback_result.json",
        "source_to_derived_verifier_result.json",
        "source_accounting_verification.json",
        "leakage_audit.json",
        "archive_manifest.json",
        "restore_validation_result.json",
        "accepted_multiskill_dataset_package.json",
        "result_classification.json",
        "eligibility_state.json",
        "authorization_state_final.json",
        "remote_execution_audit.json",
    }
    names = {str(row["path"]) for row in entries}
    missing = sorted(required - names)
    if missing:
        raise Phase2B6V2FinalizeError(f"compact evidence is incomplete: {missing}")
    manifest = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-v2-artifact-manifest-v0",
            "scope": "accepted_versioned_multiskill_production",
            "file_count": len(entries),
            "files": entries,
            "large_rgb_or_npz_committed": False,
            "student_policy_training_started": False,
            "optimizer_steps": 0,
            "passed": True,
        }
    )
    _write_json(manifest_path, manifest)
    return manifest


def verify_artifacts(evidence_root: Path) -> dict[str, object]:
    """Independently rehash the final compact evidence package."""

    manifest = _read_json(evidence_root / "artifact_manifest.json")
    unhashed = dict(manifest)
    fingerprint = unhashed.pop("fingerprint", None)
    rows = cast(list[dict[str, object]], manifest.get("files", []))
    checks = [
        {
            "path": row["path"],
            "expected_sha256": row["sha256"],
            "observed_sha256": (
                "sha256:" + sha256_file(evidence_root / str(row["path"]))
                if (evidence_root / str(row["path"])).is_file()
                else None
            ),
            "expected_size_bytes": row["size_bytes"],
            "observed_size_bytes": (
                (evidence_root / str(row["path"])).stat().st_size
                if (evidence_root / str(row["path"])).is_file()
                else None
            ),
        }
        for row in rows
    ]
    passed = (
        canonical_json_sha256(unhashed) == fingerprint
        and all(
            row["expected_sha256"] == row["observed_sha256"]
            and row["expected_size_bytes"] == row["observed_size_bytes"]
            for row in checks
        )
        and _read_json(evidence_root / "result_classification.json").get("result")
        == Phase2B6V2Result.RESULT_A.value
        and _read_json(evidence_root / "authorization_state_final.json").get(
            "act_training_authorized"
        )
        is False
    )
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-v2-verification-v0",
            "artifact_manifest_fingerprint": fingerprint,
            "file_count": len(rows),
            "files": checks,
            "passed": passed,
        }
    )


__all__ = [
    "Phase2B6V2FinalizeError",
    "finalize_acceptance",
    "verify_artifacts",
    "write_artifact_manifest",
]
