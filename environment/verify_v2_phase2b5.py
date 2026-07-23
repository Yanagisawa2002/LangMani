"""Independently verify compact Phase 2B.5 Result A evidence without simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from langmani.v2.phase2b5 import SELECTED_TASK_IDS, canonical_json_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2b5"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "diagnostics" / "v2" / "phase2b5" / "verification.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _safe(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 1:
        raise ValueError(f"unsafe artifact path {relative!r}")
    path = root / pure.name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or linked artifact {relative!r}")
    return path


def _task_gate(report: dict[str, Any], task_id: str) -> dict[str, Any]:
    for task in report.get("task_reports", []):
        if isinstance(task, dict) and task.get("task_id") == task_id:
            gate = task.get("gate")
            if isinstance(gate, dict):
                return gate
    raise ValueError(f"replay report lacks {task_id}")


def main() -> int:
    args = parse_args()
    root = args.artifact_root.resolve()
    manifest = _read(root / "artifact_manifest.json")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("artifact manifest files must be a list")
    hash_checks: dict[str, bool] = {}
    size_checks: dict[str, bool] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("artifact entry must be an object")
        relative = str(entry["path"])
        path = _safe(root, relative)
        raw = path.read_bytes().replace(b"\r\n", b"\n")
        hash_checks[relative] = "sha256:" + hashlib.sha256(raw).hexdigest() == entry["sha256"]
        size_checks[relative] = len(raw) == entry["size_bytes"]

    authorization = _read(root / "authorization_state.json")
    bounded = _read(root / "bounded_replay_results.json")
    strong = _read(root / "strong_replay_results.json")
    classification = _read(root / "result_classification.json")
    package = _read(root / "accepted_official_demo_source_package.json")
    schema = _read(root / "source_schema_statistics.json")
    visual = _read(root / "visual_pilot_manifest.json")
    conversion = _read(root / "conversion_pilot_manifest.json")
    readback = _read(root / "lerobot_readback_result.json")
    privilege = _read(root / "privilege_exclusion_audit.json")
    split = _read(root / "split_design_manifest.json")
    closure = _read(root / "custom_push_route_closure.json")
    candidate = _read(root / "task_candidate_audit.json")
    replay_checks = {
        f"bounded_{task_id}": (
            _task_gate(bounded, task_id).get("episode_count") == 20
            and _task_gate(bounded, task_id).get("passed") is True
        )
        for task_id in SELECTED_TASK_IDS
    }
    replay_checks.update(
        {
            f"strong_{task_id}": (
                _task_gate(strong, task_id).get("episode_count") == 100
                and _task_gate(strong, task_id).get("passed") is True
            )
            for task_id in SELECTED_TASK_IDS
        }
    )
    checks = {
        "all_hashes_valid": bool(hash_checks) and all(hash_checks.values()),
        "all_sizes_valid": bool(size_checks) and all(size_checks.values()),
        "result_a": classification.get("result") == "RESULT_A",
        "three_selected_tasks": package.get("selected_task_ids") == list(SELECTED_TASK_IDS),
        "three_distinct_skills": len(set(package.get("skill_families", []))) == 3,
        "selected_source_schema_valid": schema.get("all_selected_sources_valid") is True,
        "all_replay_gates_passed": all(replay_checks.values()),
        "visual_pilot_passed": (visual.get("passed") is True and visual.get("episode_count") == 15),
        "privileged_fields_excluded": privilege.get("passed") is True,
        "lerobot_conversion_pilot_passed": (
            conversion.get("passed") is True and conversion.get("episode_count") == 15
        ),
        "lerobot_readback_passed": readback.get("passed") is True,
        "split_design_leakage_check_passed": (
            split.get("leakage_checks", {}).get("passed") is True
            and split.get("materialized") is False
        ),
        "custom_route_closed": (
            closure.get("custom_push_expert_route_active") is False
            and closure.get("custom_push_expert_reactivation_authorized") is False
        ),
        "poke_and_pull_not_forced_into_contract": (
            "delta-action" in str(candidate.get("decision", {}).get("PokeCube-v1"))
            and "delta-action" in str(candidate.get("decision", {}).get("PullCube-v1"))
        ),
        "source_validated": authorization.get("official_demo_source_validated") is True,
        "phase2b6_eligible_only": (
            authorization.get("phase2b6_dataset_production_eligible") is True
            and authorization.get("full_dataset_production_started") is False
        ),
        "all_training_unauthorized": all(
            authorization.get(key) is False
            for key in (
                "act_training_authorized",
                "smolvla_training_authorized",
                "vla_jepa_training_authorized",
                "student_policy_training_started",
            )
        ),
    }
    report: dict[str, object] = {
        "schema_version": "langmani-v2-phase2b5-independent-verification-v0",
        "result": classification.get("result"),
        "checks": checks,
        "replay_checks": replay_checks,
        "artifact_hash_checks": hash_checks,
        "artifact_size_checks": size_checks,
        "passed": all(checks.values()),
        "official_demo_source_validated": authorization.get("official_demo_source_validated"),
        "phase2b6_dataset_production_eligible": authorization.get(
            "phase2b6_dataset_production_eligible"
        ),
        "act_training_authorized": authorization.get("act_training_authorized"),
        "smolvla_training_authorized": authorization.get("smolvla_training_authorized"),
        "vla_jepa_training_authorized": authorization.get("vla_jepa_training_authorized"),
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
