"""Independently verify compact Phase 2B.6.1 evidence without simulation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, cast

from langmani.v2.phase2b5 import canonical_json_sha256, sha256_file
from langmani.v2.phase2b6_1 import (
    CONTROL_EPISODE_IDS,
    GATE_ORDER,
    GateStatus,
    authorization_state,
    classify_result,
    controls_validated,
    eligibility_state,
    run_passed_for_mode,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2b6_1"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs" / "diagnostics" / "v2" / "phase2b6_1" / "verification.json"
)

REQUIRED_FILES = {
    "repository_environment_audit.json",
    "phase2b6_prior_evidence_verification.json",
    "phase2b5_source_package_verification.json",
    "frozen_partial_output_inventory.json",
    "source_episode_identity_manifest.json",
    "historical_evidence_audit.json",
    "infrastructure_failure_report.json",
    "forensic_protocol.json",
    "producer_forensic_call_sequence_audit.json",
    "control_episode_results.json",
    "episode_938_mode_a_results.json",
    "episode_938_mode_b_results.json",
    "episode_938_mode_c_result.json",
    "physical_state_comparison.json",
    "success_timing_audit.json",
    "observation_action_alignment_audit.json",
    "writer_filesystem_audit.json",
    "first_failure_report.json",
    "result_classification.json",
    "eligibility_state.json",
    "authorization_state.json",
    "authorization_state_final.json",
    "remote_execution_audit.json",
    "artifact_manifest.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return value


def _safe(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe artifact path: {relative}")
    return root.joinpath(*pure.parts)


def _fingerprint_valid(document: Mapping[str, object]) -> bool:
    expected = document.get("fingerprint")
    payload = dict(document)
    payload.pop("fingerprint", None)
    return expected == canonical_json_sha256(payload)


def verify(artifact_root: Path) -> dict[str, object]:
    """Recompute all compact identities and the terminal classification."""

    root = artifact_root.resolve()
    manifest = _read(root / "artifact_manifest.json")
    entries = cast(list[dict[str, object]], manifest.get("files", []))
    listed = {str(entry["path"]) for entry in entries}
    hash_checks: dict[str, bool] = {}
    size_checks: dict[str, bool] = {}
    fingerprints: dict[str, bool] = {}
    documents: dict[str, dict[str, Any]] = {}
    for entry in entries:
        relative = str(entry["path"])
        path = _safe(root, relative)
        hash_checks[relative] = path.is_file() and "sha256:" + sha256_file(path) == entry["sha256"]
        size_checks[relative] = path.is_file() and path.stat().st_size == entry["size_bytes"]
        if path.suffix == ".json" and path.is_file():
            document = _read(path)
            documents[relative] = document
            fingerprints[relative] = _fingerprint_valid(document)
    control_runs = cast(
        list[Mapping[str, object]], documents["control_episode_results.json"]["runs"]
    )
    mode_a = cast(list[Mapping[str, object]], documents["episode_938_mode_a_results.json"]["runs"])
    mode_b = cast(list[Mapping[str, object]], documents["episode_938_mode_b_results.json"]["runs"])
    mode_c = cast(list[Mapping[str, object]], documents["episode_938_mode_c_result.json"]["runs"])
    infrastructure = documents.get("infrastructure_failure_report.json")
    run_gate_checks = []
    for run in control_runs + mode_a + mode_b + mode_c:
        sub_gates = cast(Mapping[str, Mapping[str, object]], run["sub_gates"])
        run_gate_checks.append(
            list(sub_gates) == list(GATE_ORDER)
            and all(
                gate_document["status"] in {status.value for status in GateStatus}
                for gate_document in sub_gates.values()
            )
            and run["passed"] is run_passed_for_mode(run)
        )
    source = documents["source_episode_identity_manifest.json"]
    recomputed_classification = classify_result(
        source_identity_proven=source["producer_input_identity_proven"] is True,
        control_runs=control_runs,
        mode_a_runs=mode_a,
        mode_b_runs=mode_b,
        mode_c_runs=mode_c,
        historical_producer_final_success=False,
        infrastructure_failure=infrastructure is not None,
    )
    recorded_classification = documents["result_classification.json"]
    recomputed_eligibility = eligibility_state(recomputed_classification)
    recorded_eligibility = documents["eligibility_state.json"]
    expected_authorization = authorization_state()
    recorded_authorization = documents["authorization_state_final.json"]
    control_ids = sorted(
        int(cast(Any, cast(Mapping[str, object], run["run_identity"])["episode_id"]))
        for run in control_runs
    )
    checks = {
        "required_files": (listed | {"artifact_manifest.json"}) >= REQUIRED_FILES,
        "manifest_fingerprint": _fingerprint_valid(manifest),
        "all_hashes": bool(hash_checks) and all(hash_checks.values()),
        "all_sizes": bool(size_checks) and all(size_checks.values()),
        "all_document_fingerprints": bool(fingerprints) and all(fingerprints.values()),
        "explicit_run_gates": all(run_gate_checks),
        "controls_exact": (
            control_ids == list(CONTROL_EPISODE_IDS) and controls_validated(control_runs)
            if infrastructure is None
            else not control_runs
            and infrastructure["passed"] is True
            and infrastructure["attempt_result"]["source_action_submissions"] == 0
        ),
        "mode_a_budget_exact": len(mode_a) == 3 if infrastructure is None else not mode_a,
        "mode_b_budget_not_exceeded": len(mode_b) <= 2,
        "mode_c_budget_not_exceeded": len(mode_c) <= 1,
        "hard_stop_preserved": (
            True if infrastructure is None else not mode_a and not mode_b and not mode_c
        ),
        "classification_recomputed": recomputed_classification == recorded_classification,
        "eligibility_recomputed": recomputed_eligibility == recorded_eligibility,
        "authorization_recomputed": expected_authorization == recorded_authorization,
        "production_resume_closed": (
            recorded_authorization["phase2b6_production_resume_authorized"] is False
        ),
        "accepted_dataset_closed": (
            recorded_authorization["accepted_multiskill_dataset_validated"] is False
        ),
        "all_training_closed": all(
            recorded_authorization[key] is False
            for key in (
                "act_training_eligible",
                "smolvla_training_eligible",
                "vla_jepa_training_eligible",
                "act_training_authorized",
                "smolvla_training_authorized",
                "vla_jepa_training_authorized",
                "student_policy_training_started",
                "optimizer_created",
            )
        ),
        "optimizer_steps_zero": recorded_authorization["optimizer_steps"] == 0,
    }
    return {
        "schema_version": "langmani-v2-phase2b6-1-independent-verification-v0",
        "artifact_root": root.as_posix(),
        "result": recorded_classification["result"],
        "primary_failure_classification": recorded_classification["primary_failure_classification"],
        "checks": checks,
        "phase2b6_production_resume_authorized": False,
        "accepted_multiskill_dataset_validated": False,
        "all_training_authorized": False,
        "optimizer_steps": 0,
        "passed": all(checks.values()),
    }


def main() -> int:
    args = parse_args()
    report = verify(args.artifact_root)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
