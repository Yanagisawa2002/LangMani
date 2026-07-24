"""Independently verify compact Phase 2B.6.1-v2 forensic evidence."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from langmani.v2.phase2b5 import canonical_json_sha256, sha256_file
from langmani.v2.phase2b6_1_v2 import (
    SOURCE_COMMIT,
    TARGET_BRANCH,
    authorization_state,
    classify_result,
    determinism_audit,
    eligibility_state,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2b6_1_v2"


class VerificationError(RuntimeError):
    """Raised when compact evidence is missing or malformed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"failed to read {path}: {error}") from error
    if not isinstance(value, dict):
        raise VerificationError(f"{path} must contain one JSON object")
    return value


def _fingerprint_valid(document: Mapping[str, object]) -> bool:
    expected = document.get("fingerprint")
    unhashed = dict(document)
    unhashed.pop("fingerprint", None)
    return isinstance(expected, str) and canonical_json_sha256(unhashed) == expected


def _fresh_process_identity(run: Mapping[str, object]) -> str:
    identity = cast(Mapping[str, object], run["run_identity"])
    fresh = cast(Mapping[str, object], identity.get("fresh_process", {}))
    return str(fresh.get("identity"))


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.parent / f".{path.name}.partial"
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with staging.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()


def verify(evidence_root: Path) -> dict[str, object]:
    """Recompute hashes, result classification, and closed authorization."""

    required = (
        "repository_environment_audit.json",
        "prior_evidence_verification.json",
        "validated_launcher_binding.json",
        "validated_rendering_contract_manifest.json",
        "frozen_partial_output_inventory.json",
        "source_episode_identity.json",
        "action_identity.json",
        "forensic_protocol.json",
        "producer_forensic_call_sequence_audit.json",
        "control_936_result.json",
        "control_937_result.json",
        "episode_938_mode_a_results.json",
        "episode_938_mode_b_results.json",
        "episode_938_mode_c_result.json",
        "task_state_diagnostics.json",
        "success_timing_audit.json",
        "determinism_audit.json",
        "observation_action_alignment_audit.json",
        "writer_filesystem_audit.json",
        "first_failure_report.json",
        "result_classification.json",
        "eligibility_state.json",
        "authorization_state.json",
        "authorization_state_final.json",
        "remote_execution_audit.json",
        "artifact_manifest.json",
    )
    documents = {name: _read_json(evidence_root / name) for name in required}
    manifest = documents["artifact_manifest.json"]
    rows = cast(Sequence[Mapping[str, object]], manifest["files"])
    manifest_hashes_valid = all(
        (evidence_root / str(row["path"])).is_file()
        and "sha256:" + sha256_file(evidence_root / str(row["path"])) == row["sha256"]
        for row in rows
        if row["path"] != "verification.json"
    )
    manifest_paths = {str(row["path"]) for row in rows}
    control_runs = [
        documents["control_936_result.json"],
        documents["control_937_result.json"],
    ]
    mode_a = cast(
        list[dict[str, object]],
        documents["episode_938_mode_a_results.json"]["runs"],
    )
    mode_b = cast(
        list[dict[str, object]],
        documents["episode_938_mode_b_results.json"]["runs"],
    )
    mode_c = cast(
        list[dict[str, object]],
        documents["episode_938_mode_c_result.json"]["runs"],
    )
    source = documents["source_episode_identity.json"]
    rendering = documents["validated_rendering_contract_manifest.json"]
    recomputed_result = classify_result(
        source_identity_proven=source["producer_input_identity_proven"] is True,
        rendering_preflight_passed=rendering["passed"] is True,
        control_runs=control_runs,
        mode_a_runs=mode_a,
        mode_b_runs=mode_b,
        mode_c_runs=mode_c,
    )
    result = documents["result_classification.json"]
    recomputed_determinism = determinism_audit(mode_a)
    expected_authorization = authorization_state()
    expected_eligibility = eligibility_state(result)
    repository = documents["repository_environment_audit.json"]
    action = documents["action_identity.json"]
    protocol = documents["forensic_protocol.json"]
    first_failure = documents["first_failure_report.json"]
    remote = documents["remote_execution_audit.json"]
    checks = {
        "all_required_documents_present": all(
            (evidence_root / name).is_file() for name in required
        ),
        "all_document_fingerprints_valid": all(
            _fingerprint_valid(document) for document in documents.values()
        ),
        "artifact_manifest_hashes_valid": manifest_hashes_valid,
        "artifact_manifest_covers_required_nonself_files": set(required[:-1]).issubset(
            manifest_paths
        ),
        "source_commit_exact": repository.get("source_commit") == SOURCE_COMMIT,
        "branch_exact": repository.get("branch") == TARGET_BRANCH,
        "prior_evidence_verified": documents["prior_evidence_verification.json"].get("passed")
        is True,
        "validated_launcher_bound": documents["validated_launcher_binding.json"].get("passed")
        is True,
        "rendering_preflight_passed": rendering.get("passed") is True,
        "source_identity_proven": source.get("producer_input_identity_proven") is True,
        "target_action_identity_exact": action.get("passed") is True,
        "controls_exact": len(control_runs) == 2,
        "mode_a_budget_not_exceeded": len(mode_a) <= 3,
        "mode_b_budget_not_exceeded": len(mode_b) <= 2,
        "mode_c_budget_not_exceeded": len(mode_c) <= 1,
        "fresh_process_identities_unique": len(
            {_fresh_process_identity(run) for run in (*control_runs, *mode_a, *mode_b, *mode_c)}
        )
        == len((*control_runs, *mode_a, *mode_b, *mode_c)),
        "result_recomputed_exactly": result == recomputed_result,
        "determinism_recomputed_exactly": (
            documents["determinism_audit.json"] == recomputed_determinism
        ),
        "first_failure_matches_result": (
            first_failure.get("exact_first_failed_sub_gate")
            == result.get("exact_first_failed_sub_gate")
            and first_failure.get("primary_failure_classification")
            == result.get("primary_failure_classification")
        ),
        "stable_gate_not_fabricated": (
            protocol.get("separate_stable_success_gate_present_in_producer") is False
            and protocol.get("stable_success_gate_status") == "not_applicable"
        ),
        "acceptance_policy_unchanged": protocol.get("production_acceptance_policy_changed")
        is False,
        "authorization_closed_exactly": (
            documents["authorization_state.json"] == expected_authorization
            and documents["authorization_state_final.json"] == expected_authorization
        ),
        "eligibility_recomputed_exactly": (
            documents["eligibility_state.json"] == expected_eligibility
        ),
        "frozen_inventory_unchanged": remote.get("frozen_inventory_unchanged") is True,
        "production_not_resumed": remote.get("production_resumed") is False,
        "stackcube_939_plus_not_processed": remote.get("stackcube_939_or_later_processed") is False,
        "pushcube_not_processed": remote.get("pushcube_processed") is False,
        "no_package_or_archive": (
            remote.get("dataset_package_created") is False
            and remote.get("archive_or_restore_started") is False
        ),
        "no_model_or_optimizer": (
            remote.get("model_or_optimizer_loaded") is False
            and expected_authorization["optimizer_created"] is False
            and expected_authorization["optimizer_steps"] == 0
        ),
    }
    return {
        "schema_version": "langmani-v2-phase2b6-1-v2-independent-verification-v0",
        "result": result.get("result"),
        "check_count": len(checks),
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    args = parse_args()
    evidence_root = args.evidence_root.resolve()
    report = verify(evidence_root)
    if args.report is not None:
        _write_json(args.report.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
