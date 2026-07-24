"""Execute bounded LangMani 2.0 Phase 2B.6.1-v2 forensic stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langmani.v2.phase2b6_1_v2_runtime import (
    finalize_forensics,
    prepare_forensics,
    run_rendering_preflight,
    run_replay,
    write_artifact_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = PROJECT_ROOT / "configs" / "langmani_v2" / "phase2b6_dataset_production.yaml"
DEFAULT_ACCEPTED_LAUNCHER = PROJECT_ROOT / "environment" / "run_v2_phase2b6_1r_preflight.sh"
DEFAULT_ADAPTER_LAUNCHER = PROJECT_ROOT / "environment" / "run_v2_phase2b6_1_v2_forensic.sh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "rendering-preflight",
            "run",
            "finalize",
            "artifact-manifest",
        ),
    )
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--frozen-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--accepted-launcher", type=Path, default=DEFAULT_ACCEPTED_LAUNCHER)
    parser.add_argument("--adapter-launcher", type=Path, default=DEFAULT_ADAPTER_LAUNCHER)
    parser.add_argument("--network-turbo-sourced", action="store_true")
    parser.add_argument("--mode", choices=("A", "B", "C"))
    parser.add_argument("--episode-id", type=int)
    parser.add_argument("--repetition", type=int)
    return parser.parse_args()


def _required_path(value: Path | None, label: str) -> Path:
    if value is None:
        raise ValueError(f"{label} is required")
    return value.resolve()


def _required_int(value: int | None, label: str) -> int:
    if value is None:
        raise ValueError(f"{label} is required")
    return value


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    evidence_root = args.evidence_root.resolve()
    if args.stage == "prepare":
        result = prepare_forensics(
            repo_root=args.repo_root.resolve(),
            source_root=_required_path(args.source_root, "--source-root"),
            frozen_root=_required_path(args.frozen_root, "--frozen-root"),
            output_root=output_root,
            evidence_root=evidence_root,
            spec_path=args.spec.resolve(),
            accepted_launcher=args.accepted_launcher.resolve(),
            adapter_launcher=args.adapter_launcher.resolve(),
            network_turbo_sourced=args.network_turbo_sourced,
        )
    elif args.stage == "rendering-preflight":
        result = run_rendering_preflight(
            source_root=_required_path(args.source_root, "--source-root"),
            output_root=output_root,
            evidence_root=evidence_root,
        )
    elif args.stage == "run":
        if args.mode is None:
            raise ValueError("--mode is required for run")
        result = run_replay(
            source_root=_required_path(args.source_root, "--source-root"),
            output_root=output_root,
            evidence_root=evidence_root,
            spec_path=args.spec.resolve(),
            mode=args.mode,
            episode_id=_required_int(args.episode_id, "--episode-id"),
            repetition=_required_int(args.repetition, "--repetition"),
        )
    elif args.stage == "finalize":
        result = finalize_forensics(
            output_root=output_root,
            evidence_root=evidence_root,
            frozen_root=_required_path(args.frozen_root, "--frozen-root"),
        )
    else:
        result = write_artifact_manifest(evidence_root)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
