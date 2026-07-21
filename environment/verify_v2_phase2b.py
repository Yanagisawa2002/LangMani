"""Independently verify Phase 2B pilot or full collection/export evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langmani.v2.push_audit import verify_phase2b_evidence
from langmani.v2.push_dataset import PushCollectionConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "langmani_v2" / "phase2b_push_collection.yaml"
DEFAULT_SOURCE = PROJECT_ROOT / "outputs" / "datasets" / "langmani_v2" / "phase2b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", choices=("pilot", "full", "top_up"), required=True)
    parser.add_argument("--compatibility-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = verify_phase2b_evidence(
        config=PushCollectionConfig.load(args.config),
        source_root=args.source_root,
        export_root=args.export_root,
        stages=args.stages,
        compatibility_report=args.compatibility_report,
        output=args.output,
        full=args.full,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
