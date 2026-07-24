"""Independently verify compact Phase 2B.6.1-R evidence without simulation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, cast

from langmani.v2.phase2b6_1r import (
    CandidateKind,
    Phase2B61RResult,
    authorization_state,
    candidate_protocol,
    canonical_json_sha256,
    classify_result,
    device_identity_matches,
    repeatability_passed,
    sha256_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2b6_1r"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "diagnostics"
    / "v2"
    / "phase2b6_1r"
    / "verification.json"
)
REQUIRED_FILES = {
    "repository_environment_audit.json",
    "phase2b6_1_evidence_verification.json",
    "rendering_stack_inventory.json",
    "candidate_icd_manifest.json",
    "icd_hashes.json",
    "frozen_preflight_protocol.json",
    "launcher_runtime_manifest.json",
    "vulkaninfo_results.json",
    "sapien_probe_results.json",
    "zero_step_stackcube_results.json",
    "fresh_process_repeatability_result.json",
    "negative_control_result.json",
    "device_binding_audit.json",
    "cleanup_audit.json",
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


def _zero_runs(document: Mapping[str, object]) -> list[Mapping[str, object]]:
    runs = document.get("runs")
    if not isinstance(runs, list):
        return []
    return [cast(Mapping[str, object], run) for run in runs if isinstance(run, dict)]


def verify(artifact_root: Path) -> dict[str, object]:
    """Rehash evidence and recompute the result and authorization boundary."""

    root = artifact_root.resolve()
    manifest = _read(root / "artifact_manifest.json")
    entries = cast(list[dict[str, object]], manifest.get("files", []))
    listed = {str(entry["path"]) for entry in entries}
    hashes = {}
    sizes = {}
    fingerprints = {}
    documents = {}
    for entry in entries:
        relative = str(entry["path"])
        path = _safe(root, relative)
        hashes[relative] = path.is_file() and sha256_file(path) == entry.get("sha256")
        sizes[relative] = path.is_file() and path.stat().st_size == entry.get("size_bytes")
        if path.is_file() and path.suffix == ".json":
            document = _read(path)
            documents[relative] = document
            fingerprints[relative] = _fingerprint_valid(document)
    required_documents = REQUIRED_FILES - {"artifact_manifest.json"}
    missing_documents = required_documents - set(documents)
    if missing_documents:
        raise ValueError(f"required documents missing from manifest: {sorted(missing_documents)}")
    candidates = documents["candidate_icd_manifest.json"]
    candidate_rows = cast(Sequence[Mapping[str, object]], candidates["candidates"])
    expected_candidates = candidate_protocol()
    candidate_contract = len(candidate_rows) == len(expected_candidates) and all(
        row["candidate_id"] == expected["candidate_id"]
        and row["vk_icd_filenames"] == expected["vk_icd_filenames"]
        and row["vk_driver_files"] == expected["vk_driver_files"]
        and row["may_run_zero_step_stackcube"] == expected["may_run_zero_step_stackcube"]
        for row, expected in zip(candidate_rows, expected_candidates, strict=True)
    )
    zero = documents["zero_step_stackcube_results.json"]
    zero_runs = _zero_runs(zero)
    flattened = []
    for index, run in enumerate(zero_runs, start=1):
        device = cast(Mapping[str, object], run.get("render_device", {}))
        flattened.append(
            {
                "pid": cast(
                    Sequence[Mapping[str, object]],
                    documents["fresh_process_repeatability_result.json"]["runs"],
                )[index - 1]["pid"],
                "passed": run.get("passed"),
                "icd_sha256": cast(
                    Sequence[Mapping[str, object]],
                    documents["fresh_process_repeatability_result.json"]["runs"],
                )[index - 1]["icd_sha256"],
                "vulkan_device_name": cast(
                    Sequence[Mapping[str, object]],
                    documents["fresh_process_repeatability_result.json"]["runs"],
                )[index - 1]["vulkan_device_name"],
                "render_device_name": device.get("name"),
                "render_device_pci": device.get("pci_string"),
                "action_shape": run.get("action_shape", []),
                "control_mode": run.get("control_mode"),
                "obs_mode": run.get("obs_mode"),
            }
        )
    zero_counts = {
        "reset": zero.get("explicit_reset_count"),
        "step": zero.get("explicit_step_count"),
        "action": zero.get("action_submission_count"),
    }
    repeatability = repeatability_passed(flattened)
    correct_device = len(zero_runs) == 3 and all(
        device_identity_matches(cast(Mapping[str, object], run.get("render_device", {})))
        for run in zero_runs
    )
    vulkan = documents["vulkaninfo_results.json"]
    sapien = documents["sapien_probe_results.json"]
    cleanup = documents["cleanup_audit.json"]
    repository = documents["repository_environment_audit.json"]
    prior = documents["phase2b6_1_evidence_verification.json"]
    remote = documents["remote_execution_audit.json"]
    recorded = documents["result_classification.json"]
    recomputed = classify_result(
        host_evidence_sufficient=repository["passed"] is True
        and prior["passed"] is True
        and remote["frozen_root_unchanged"] is True,
        vulkan_passed=vulkan["passed"] is True,
        sapien_passed=sapien["passed"] is True,
        zero_step_passed=zero["passed"] is True,
        repeatability_validated=repeatability,
        correct_device_selected=correct_device,
        cleanup_passed=cleanup["passed"] is True,
        first_failure_layer=(
            None
            if recorded["result"] == Phase2B61RResult.RESULT_A.value
            else str(recorded["reason"])
        ),
    )
    expected_authorization = authorization_state()
    authorization = documents["authorization_state_final.json"]
    eligibility = documents["eligibility_state.json"]
    result_a = recorded["result"] == Phase2B61RResult.RESULT_A.value
    checks = {
        "required_files": (listed | {"artifact_manifest.json"}) >= REQUIRED_FILES,
        "manifest_fingerprint": _fingerprint_valid(manifest),
        "all_hashes": bool(hashes) and all(hashes.values()),
        "all_sizes": bool(sizes) and all(sizes.values()),
        "all_document_fingerprints": bool(fingerprints) and all(fingerprints.values()),
        "candidate_protocol_exact": candidate_contract,
        "primary_candidate_exact": (
            candidate_rows[0]["candidate_id"] == CandidateKind.PRIMARY.value
        ),
        "zero_step_run_count": len(zero_runs) == 3 if result_a else len(zero_runs) <= 3,
        "reset_count_zero": zero_counts["reset"] == 0,
        "step_count_zero": zero_counts["step"] == 0,
        "action_count_zero": zero_counts["action"] == 0,
        "repeatability_recomputed": repeatability
        == documents["fresh_process_repeatability_result.json"]["passed"],
        "device_binding_recomputed": correct_device
        == documents["device_binding_audit.json"]["passed"],
        "classification_recomputed": recorded == recomputed,
        "eligibility_matches_result": eligibility["phase2b6_1_forensic_restart_eligible"]
        is result_a,
        "authorization_recomputed": authorization == expected_authorization,
        "forensic_restart_unauthorized": authorization[
            "phase2b6_1_forensic_restart_authorized"
        ]
        is False,
        "production_resume_unauthorized": authorization[
            "phase2b6_production_resume_authorized"
        ]
        is False,
        "accepted_dataset_closed": authorization["accepted_multiskill_dataset_validated"]
        is False,
        "training_closed": all(
            authorization[name] is False
            for name in (
                "act_training_eligible",
                "smolvla_training_eligible",
                "vla_jepa_training_eligible",
                "act_training_authorized",
                "smolvla_training_authorized",
                "vla_jepa_training_authorized",
                "student_policy_training_started",
            )
        ),
        "optimizer_zero": authorization["optimizer_steps"] == 0
        and authorization["backward_passes"] == 0
        and authorization["optimizer_created"] is False,
        "frozen_root_unchanged": remote["frozen_root_unchanged"] is True,
        "no_replay_or_production": remote["forensic_replay_started"] is False
        and remote["production_resumed"] is False
        and remote["dataset_writer_started"] is False,
    }
    return {
        "schema_version": "langmani-v2-phase2b6-1r-independent-verification-v0",
        "artifact_root": root.as_posix(),
        "checks": checks,
        "result": recorded["result"],
        "vulkan_preflight_validated": recorded["vulkan_preflight_validated"],
        "stackcube_zero_step_construction_validated": recorded[
            "stackcube_zero_step_construction_validated"
        ],
        "phase2b6_1_forensic_restart_eligible": recorded[
            "phase2b6_1_forensic_restart_eligible"
        ],
        "phase2b6_1_forensic_restart_authorized": False,
        "phase2b6_production_resume_authorized": False,
        "accepted_multiskill_dataset_validated": False,
        "all_training_authorized": False,
        "optimizer_steps": 0,
        "passed": all(checks.values()),
    }


def main() -> int:
    args = parse_args()
    report = verify(args.artifact_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
