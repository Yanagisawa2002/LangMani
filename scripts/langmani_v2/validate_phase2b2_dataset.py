"""Rehash raw episodes and fully scan the real Phase 2B.2 LeRobot export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langmani.v2.phase2b2 import (
    build_raw_episode_manifest,
    scan_lerobot_quality,
    write_json_once,
)
from langmani.v2.push_dataset import PushCollectionConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "langmani_v2" / "phase_2b2" / "dataset_contract.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", choices=("full", "top_up"), required=True)
    parser.add_argument("--expert-report", type=Path, required=True)
    parser.add_argument("--raw-manifest-output", type=Path, required=True)
    parser.add_argument("--dataset-manifest-output", type=Path, required=True)
    parser.add_argument("--quality-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PushCollectionConfig.load(args.config)
    expert = json.loads(args.expert_report.read_text(encoding="utf-8"))
    raw = build_raw_episode_manifest(
        config=config, source_root=args.source_root, stages=args.stages
    )
    manifest, quality = scan_lerobot_quality(
        config=config,
        source_root=args.source_root,
        export_root=args.export_root,
        stages=args.stages,
        expert_report=expert,
    )
    write_json_once(args.raw_manifest_output, raw)
    write_json_once(args.dataset_manifest_output, manifest)
    write_json_once(args.quality_output, quality)
    print(
        json.dumps(
            {
                "episodes": manifest["episode_count"],
                "frames": manifest["frame_count"],
                "passed": quality["passed"],
            },
            sort_keys=True,
        )
    )
    return 0 if quality["passed"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
