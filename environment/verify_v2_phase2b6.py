"""Independently verify compact Phase 2B.6 evidence without simulation or data bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, cast

from langmani.v2.phase2b5 import canonical_json_sha256
from langmani.v2.phase2b6 import (
    EXPECTED_EPISODES,
    EXPECTED_TRANSITIONS,
    TASK_IDS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2b6"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "diagnostics" / "v2" / "phase2b6" / "verification.json"

REQUIRED_FILES = {
    "repository_environment_audit.json",
    "phase2b5_source_package_verification.json",
    "frozen_production_specification.json",
    "source_integrity_audit.json",
    "source_schema_result.json",
    "production_run_manifest.json",
    "replay_summary_pickcube.json",
    "replay_summary_stackcube.json",
    "replay_summary_pushcube.json",
    "rejected_production_manifest.json",
    "observation_action_contract.json",
    "language_template_manifest.json",
    "task_specific_root_manifests.json",
    "unified_multi_root_manifest.json",
    "primary_split_manifest.json",
    "cross_skill_fold_manifests.json",
    "visual_shift_pilot.json",
    "visual_shift_manifest.json",
    "padding_audit.json",
    "dataset_statistics.json",
    "normalization_manifest.json",
    "task_balance_manifest.json",
    "lerobot_environment_manifest.json",
    "full_readback_result.json",
    "source_to_derived_verifier_result.json",
    "leakage_audit.json",
    "archive_manifest.json",
    "restore_validation_result.json",
    "accepted_multiskill_dataset_package.json",
    "result_classification.json",
    "authorization_state.json",
    "remote_execution_audit.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read JSON {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return value


def _safe(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 1:
        raise ValueError(f"unsafe artifact path {relative!r}")
    path = root / pure.name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or linked artifact {relative!r}")
    return path


def _fingerprint_valid(payload: Mapping[str, object]) -> bool:
    fingerprint = payload.get("fingerprint")
    if not isinstance(fingerprint, str):
        return False
    body = dict(payload)
    del body["fingerprint"]
    return canonical_json_sha256(body) == fingerprint


def main() -> int:
    args = parse_args()
    root = args.artifact_root.resolve()
    manifest = _read(root / "artifact_manifest.json")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("artifact manifest files are malformed")
    hash_checks: dict[str, bool] = {}
    size_checks: dict[str, bool] = {}
    fingerprint_checks: dict[str, bool] = {}
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("artifact manifest entry is malformed")
        relative = str(raw_entry["path"])
        path = _safe(root, relative)
        normalized = path.read_bytes().replace(b"\r\n", b"\n")
        hash_checks[relative] = (
            "sha256:" + hashlib.sha256(normalized).hexdigest() == raw_entry["sha256"]
        )
        size_checks[relative] = len(normalized) == raw_entry["size_bytes"]
        fingerprint_checks[relative] = _fingerprint_valid(_read(path))
    entry_names = set(hash_checks)
    package = _read(root / "accepted_multiskill_dataset_package.json")
    result = _read(root / "result_classification.json")
    authorization = _read(root / "authorization_state.json")
    source = _read(root / "source_schema_result.json")
    production = _read(root / "production_run_manifest.json")
    task_roots = _read(root / "task_specific_root_manifests.json")
    readback = _read(root / "full_readback_result.json")
    equality = _read(root / "source_to_derived_verifier_result.json")
    leakage = _read(root / "leakage_audit.json")
    archive = _read(root / "archive_manifest.json")
    restore = _read(root / "restore_validation_result.json")
    padding = _read(root / "padding_audit.json")
    normalization = _read(root / "normalization_manifest.json")
    checks = {
        "artifact_manifest_fingerprint": _fingerprint_valid(manifest),
        "required_artifacts_present": entry_names >= REQUIRED_FILES,
        "all_artifact_hashes": all(hash_checks.values()),
        "all_artifact_sizes": all(size_checks.values()),
        "all_artifact_fingerprints": all(fingerprint_checks.values()),
        "result_a": result.get("result") == "RESULT_A" and result.get("passed") is True,
        "accepted_package": (
            package.get("schema_version") == "langmani-v2-phase2b6-accepted-multiskill-dataset-v0"
            and package.get("episode_count") == EXPECTED_EPISODES
            and package.get("frame_count") == EXPECTED_TRANSITIONS
            and package.get("tasks") == list(TASK_IDS)
            and package.get("contains_trained_model") is False
        ),
        "source_accounting": (
            source.get("passed") is True
            and source.get("episode_count") == EXPECTED_EPISODES
            and source.get("transition_count") == EXPECTED_TRANSITIONS
        ),
        "full_replay": (
            production.get("passed") is True
            and production.get("episode_count") == EXPECTED_EPISODES
            and production.get("frame_count") == EXPECTED_TRANSITIONS
            and cast(Mapping[str, object], production["aggregate_gate"]).get(
                "categorical_outcome_agreement_rate"
            )
            == 1.0
        ),
        "task_roots": (
            task_roots.get("passed") is True
            and task_roots.get("episode_count") == EXPECTED_EPISODES
            and task_roots.get("frame_count") == EXPECTED_TRANSITIONS
        ),
        "full_readback": (
            readback.get("passed") is True
            and readback.get("episode_count") == EXPECTED_EPISODES
            and readback.get("frame_count") == EXPECTED_TRANSITIONS
            and readback.get("visual_shift_episode_count") == 150
            and readback.get("decoding_failure_count") == 0
        ),
        "source_equality": (
            equality.get("passed") is True
            and equality.get("source_episode_count") == EXPECTED_EPISODES
            and equality.get("exact_action_equality_count") == EXPECTED_EPISODES
            and equality.get("approximate_action_tolerance_used") is False
        ),
        "zero_leakage": (
            leakage.get("passed") is True and leakage.get("primary_split_overlap_count") == 0
        ),
        "padding_masks": (
            padding.get("passed") is True
            and padding.get("masked_loss_contract_verified") is True
            and set(cast(Mapping[str, object], padding["audits"]))
            == {
                "10",
                "16",
                "50",
            }
        ),
        "train_only_normalization": (
            normalization.get("passed") is True
            and normalization.get("source_view") == "primary_train_only"
            and normalization.get("validation_episode_count") == 0
            and normalization.get("test_episode_count") == 0
        ),
        "archive_restore": (
            archive.get("passed") is True
            and restore.get("passed") is True
            and restore.get("restored_tree_digest") == archive.get("primary_tree_digest")
        ),
        "eligibility_only": (
            authorization.get("accepted_multiskill_dataset_validated") is True
            and authorization.get("act_baseline_training_eligible") is True
            and authorization.get("smolvla_push_multiskill_training_eligible") is True
            and authorization.get("vla_jepa_training_eligible") is True
            and authorization.get("eligibility_is_authorization") is False
        ),
        "all_training_unauthorized": all(
            authorization.get(key) is False
            for key in (
                "act_training_authorized",
                "smolvla_training_authorized",
                "vla_jepa_training_authorized",
                "student_policy_training_started",
                "optimizer_created",
                "act_training_started",
                "smolvla_training_started",
                "vla_jepa_training_started",
            )
        )
        and authorization.get("optimizer_steps") == 0
        and authorization.get("backward_passes") == 0,
    }
    report: dict[str, object] = {
        "schema_version": "langmani-v2-phase2b6-independent-verification-v0",
        "artifact_file_count": len(entries),
        "checks": checks,
        "artifact_hash_checks": hash_checks,
        "artifact_size_checks": size_checks,
        "artifact_fingerprint_checks": fingerprint_checks,
        "result": result.get("result"),
        "accepted_package_fingerprint": package.get("fingerprint"),
        "accepted_multiskill_dataset_validated": authorization.get(
            "accepted_multiskill_dataset_validated"
        ),
        "act_training_authorized": authorization.get("act_training_authorized"),
        "smolvla_training_authorized": authorization.get("smolvla_training_authorized"),
        "vla_jepa_training_authorized": authorization.get("vla_jepa_training_authorized"),
        "student_policy_training_started": authorization.get("student_policy_training_started"),
        "optimizer_steps": authorization.get("optimizer_steps"),
        "passed": all(checks.values()),
    }
    report["fingerprint"] = canonical_json_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
