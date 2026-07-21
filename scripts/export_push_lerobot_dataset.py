"""Export independently replayed Phase 2B push episodes through LeRobot 0.6.0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langmani.v2.push_dataset import PushCollectionConfig
from langmani.v2.push_lerobot import export_push_lerobot_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "langmani_v2" / "phase2b_push_collection.yaml"
DEFAULT_SOURCE = PROJECT_ROOT / "outputs" / "datasets" / "langmani_v2" / "phase2b-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--stages", nargs="+", choices=("pilot", "full", "top_up"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = export_push_lerobot_dataset(
        config=PushCollectionConfig.load(args.config),
        source_root=args.source_root,
        stages=args.stages,
        output_root=args.output_root,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
