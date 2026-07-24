"""Run the bounded Phase 2B.6.1-R Vulkan and zero-step environment gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langmani.v2.phase2b6_1r_runtime import (
    child_probe,
    execute_preflight,
    prepare_audit,
    write_artifact_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAUNCHER = PROJECT_ROOT / "environment" / "run_v2_phase2b6_1r_preflight.sh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("audit", "execute", "child-probe", "artifact-manifest"),
    )
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--frozen-root", type=Path)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--conda-prefix", type=Path)
    parser.add_argument("--launcher", type=Path, default=DEFAULT_LAUNCHER)
    parser.add_argument("--candidate-id")
    parser.add_argument("--run-index", type=int)
    parser.add_argument("--run-sapien", action="store_true")
    parser.add_argument("--run-zero-step", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _required_path(value: Path | None, label: str) -> Path:
    if value is None:
        raise ValueError(f"{label} is required")
    return value.resolve()


def _required_text(value: str | None, label: str) -> str:
    if not value:
        raise ValueError(f"{label} is required")
    return value


def _required_int(value: int | None, label: str) -> int:
    if value is None or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    if args.stage == "audit":
        result = prepare_audit(
            repo_root=repo_root,
            frozen_root=_required_path(args.frozen_root, "--frozen-root"),
            evidence_root=_required_path(args.evidence_root, "--evidence-root"),
            conda_prefix=_required_path(args.conda_prefix, "--conda-prefix"),
        )
    elif args.stage == "execute":
        result = execute_preflight(
            repo_root=repo_root,
            source_root=_required_path(args.source_root, "--source-root"),
            frozen_root=_required_path(args.frozen_root, "--frozen-root"),
            evidence_root=_required_path(args.evidence_root, "--evidence-root"),
            output_root=_required_path(args.output_root, "--output-root"),
            conda_prefix=_required_path(args.conda_prefix, "--conda-prefix"),
            launcher_path=args.launcher.resolve(),
        )
    elif args.stage == "child-probe":
        result = child_probe(
            candidate_id=_required_text(args.candidate_id, "--candidate-id"),
            source_root=_required_path(args.source_root, "--source-root"),
            output=_required_path(args.output, "--output"),
            run_index=_required_int(args.run_index, "--run-index"),
            run_sapien=args.run_sapien,
            run_zero_step=args.run_zero_step,
        )
    else:
        result = write_artifact_manifest(_required_path(args.evidence_root, "--evidence-root"))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
