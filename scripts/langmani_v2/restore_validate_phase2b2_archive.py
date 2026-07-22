"""Restore one copied Phase 2B.2 archive and validate its complete tree and loader surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langmani.v2.phase2b2 import (
    PHASE2B2_DATASET_ID,
    restore_validate_archive,
    write_json_once,
)
from langmani.v2.push_dataset import DATASET_SPLITS, PushDatasetContractError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--prefix", choices=("raw", "lerobot"), required=True)
    parser.add_argument("--expected-tree-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _lerobot_readback(root: Path) -> dict[str, object]:
    try:
        from lerobot.datasets import LeRobotDataset  # type: ignore[import-untyped]
    except ImportError as error:
        raise PushDatasetContractError("LeRobot 0.6 is required for restored readback") from error
    counts: dict[str, object] = {}
    for split in DATASET_SPLITS:
        dataset = LeRobotDataset(
            repo_id=f"{PHASE2B2_DATASET_ID}-{split}",
            root=root / "splits" / split,
            video_backend="pyav",
            return_uint8=True,
        )
        if int(dataset.num_frames) < 1 or not isinstance(dataset[0].get("task"), str):
            raise PushDatasetContractError(f"restored LeRobot split is unreadable: {split}")
        counts[split] = {
            "episodes": int(dataset.num_episodes),
            "frames": int(dataset.num_frames),
        }
    return {"splits": counts, "passed": True}


def main() -> int:
    args = parse_args()
    destination = args.destination.resolve()
    restored = restore_validate_archive(
        archive_path=args.archive.resolve(),
        destination=destination,
        prefix=args.prefix,
        expected_tree_digest=args.expected_tree_digest,
    )
    readback = (
        _lerobot_readback(destination / args.prefix)
        if args.prefix == "lerobot"
        else {"not_applicable": True, "passed": True}
    )
    report = {
        "schema_version": "langmani-v2-phase2b2-restored-copy-v0",
        "asset": args.prefix,
        "archive_sha256": restored["archive_sha256"],
        "restored_tree_digest": restored["restored_tree_digest"],
        "file_count": restored["file_count"],
        "total_bytes": restored["total_bytes"],
        "lerobot_readback": readback,
        "passed": restored["passed"] is True and readback["passed"] is True,
    }
    write_json_once(args.output.resolve(), report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
