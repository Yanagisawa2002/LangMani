"""Independently action-replay Phase 2B pushing demonstrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langmani.v2.push_dataset import PushCollectionConfig
from langmani.v2.push_replay import replay_push_attempts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "langmani_v2" / "phase2b_push_collection.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "datasets" / "langmani_v2" / "phase2b-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stages", nargs="+", choices=("pilot", "full", "top_up"), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = replay_push_attempts(
        config=PushCollectionConfig.load(args.config),
        output_root=args.output_root,
        stages=args.stages,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
