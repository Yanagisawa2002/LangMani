"""Verify the real LangMani 2.0 Phase 1 evidence before Phase 2 work."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from langmani.v2.phase1_verification import verify_phase1_preconditions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_MANIFEST = PROJECT_ROOT / "releases/langmani_v1/manifest.yaml"
DEFAULT_POLICY_CONFIG = PROJECT_ROOT / "configs/langmani_v2/policies/act_per_task_red_left_v0.json"
DEFAULT_TASK_CONFIG = PROJECT_ROOT / "configs/langmani_v2/tasks/pick_and_place_v0.json"
DEFAULT_EVIDENCE_ROOT = PROJECT_ROOT / "outputs/diagnostics/v2/phase1/act-red-left-three-seed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, default=DEFAULT_RELEASE_MANIFEST)
    parser.add_argument("--policy-config", type=Path, default=DEFAULT_POLICY_CONFIG)
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=(41001, 41002, 41003))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _write_report(path: Path, payload: dict[str, object]) -> None:
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
    if staging.exists():
        raise RuntimeError(f"verification staging path already exists: {staging}")
    staging.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    staging.replace(output)


def main() -> int:
    args = parse_args()
    result = verify_phase1_preconditions(
        project_root=PROJECT_ROOT,
        release_manifest_path=args.release_manifest.resolve(),
        policy_config_path=args.policy_config.resolve(),
        task_config_path=args.task_config.resolve(),
        evidence_root=args.evidence_root.resolve(),
        expected_seeds=args.seeds,
    )
    payload = result.to_dict()
    if args.output is not None:
        _write_report(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
