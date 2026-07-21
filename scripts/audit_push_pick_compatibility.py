"""Audit and optionally index immutable Phase 2B push and historical pick data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langmani.v2.push_audit import (
    audit_pick_place_compatibility,
    build_unified_dataset_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pick-place-root", type=Path, required=True)
    parser.add_argument("--push-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unified-index", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_pick_place_compatibility(
        pick_place_root=args.pick_place_root,
        push_export_root=args.push_root,
        output=args.output,
    )
    if args.unified_index is not None and report["passed"] is True:
        build_unified_dataset_index(
            pick_place_root=args.pick_place_root,
            push_export_root=args.push_root,
            compatibility_report=report,
            output=args.unified_index,
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
