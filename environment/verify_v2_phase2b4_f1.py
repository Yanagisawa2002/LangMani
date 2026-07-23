"""Independently verify compact Phase 2B.4-F1 Result C evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from langmani.v2.phase2b4_evidence import canonical_git_text_bytes
from langmani.v2.phase2b4_ppo import canonical_json_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2b4_f1"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs" / "diagnostics" / "v2" / "phase2b4_f1" / "verification.json"
)


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


def main() -> int:
    args = parse_args()
    root = args.artifact_root.resolve()
    manifest = _read(root / "artifact_manifest.json")
    hash_checks: dict[str, bool] = {}
    size_checks: dict[str, bool] = {}
    for raw_entry in manifest.get("files", []):
        if not isinstance(raw_entry, dict):
            raise ValueError("artifact manifest entry must be an object")
        name = str(raw_entry["path"])
        canonical = canonical_git_text_bytes(root / name)
        hash_checks[name] = "sha256:" + hashlib.sha256(canonical).hexdigest() == raw_entry["sha256"]
        size_checks[name] = len(canonical) == raw_entry["size_bytes"]
    result = _read(root / "result_classification.json")
    authorization = _read(root / "authorization_state.json")
    probe_a = _read(root / "probe_a_evaluation.json")
    runtime = _read(root / "runtime_summary.json")
    comparison = _read(root / "global_versus_residual_comparison.json")
    legality = _read(root / "residual_action_legality_audit.json")
    checks = {
        "all_hashes_valid": bool(hash_checks) and all(hash_checks.values()),
        "all_sizes_valid": bool(size_checks) and all(size_checks.values()),
        "result_c": result.get("result") == "RESULT_C",
        "probe_a_failed": probe_a.get("passed") is False,
        "probe_a_exact_episode_count": probe_a.get("summary", {}).get("episode_count") == 32,
        "wrong_object_hard_stop": probe_a.get("summary", {}).get("wrong_object_interactions") == 7,
        "probe_b_not_run": runtime.get("probe_b_environment_steps") == 0,
        "combined_budget_respected": runtime.get("combined_f1_training_steps") == 262_144,
        "residual_legality_passed": legality.get("passed") is True,
        "residual_physical_comparison_passed": comparison.get("residual_materially_safer") is True,
        "all_downstream_authorizations_false": all(
            authorization.get(key) is False
            for key in (
                "ppo_safe_curriculum_eligible",
                "ppo_full_training_authorized",
                "expert_qualification_authorized",
                "data_collection_authorized",
                "smolvla_training_authorized",
                "demonstration_source_validated",
                "student_policy_training_started",
            )
        ),
    }
    report: dict[str, object] = {
        "schema_version": "langmani-v2-phase2b4-f1-independent-verification-v0",
        "result": result.get("result"),
        "checks": checks,
        "artifact_hash_checks": hash_checks,
        "artifact_size_checks": size_checks,
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
