"""Execute strictly bounded LangMani 2.0 Phase 2B.6.1 forensic stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langmani.v2.phase2b6_1_runtime import (
    finalize_forensics,
    prepare_forensics,
    run_forensic_replay,
    write_artifact_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = PROJECT_ROOT / "configs" / "langmani_v2" / "phase2b6_dataset_production.yaml"
DEFAULT_PHASE2B5_ARTIFACTS = PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2b5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("preflight", "run", "finalize", "artifact-manifest"),
    )
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--frozen-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument(
        "--phase2b5-artifact-root",
        type=Path,
        default=DEFAULT_PHASE2B5_ARTIFACTS,
    )
    parser.add_argument("--network-turbo-sourced", action="store_true")
    parser.add_argument("--mode", choices=("A", "B", "C"))
    parser.add_argument("--episode-id", type=int)
    parser.add_argument("--repetition", type=int)
    return parser.parse_args()


def _required(value: Path | None, label: str) -> Path:
    if value is None:
        raise ValueError(f"{label} is required for this stage")
    return value.resolve()


def _required_int(value: int | None, label: str) -> int:
    if value is None:
        raise ValueError(f"{label} is required for this stage")
    return value


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    evidence_root = args.evidence_root.resolve()
    if args.stage == "preflight":
        result = prepare_forensics(
            repo_root=args.repo_root.resolve(),
            source_root=_required(args.source_root, "--source-root"),
            frozen_root=_required(args.frozen_root, "--frozen-root"),
            output_root=output_root,
            evidence_root=evidence_root,
            spec_path=args.spec.resolve(),
            phase2b5_artifact_root=args.phase2b5_artifact_root.resolve(),
            network_turbo_sourced=args.network_turbo_sourced,
        )
    elif args.stage == "run":
        if args.mode is None:
            raise ValueError("--mode is required for run")
        result = run_forensic_replay(
            source_root=_required(args.source_root, "--source-root"),
            output_root=output_root,
            spec_path=args.spec.resolve(),
            mode_value=args.mode,
            episode_id=_required_int(args.episode_id, "--episode-id"),
            repetition=_required_int(args.repetition, "--repetition"),
        )
    elif args.stage == "finalize":
        result = finalize_forensics(
            output_root=output_root,
            evidence_root=evidence_root,
        )
    else:
        result = write_artifact_manifest(evidence_root)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
