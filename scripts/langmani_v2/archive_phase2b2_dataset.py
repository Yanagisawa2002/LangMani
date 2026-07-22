"""Create and restore-validate the immutable Phase 2B.2 dataset archives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langmani.v2.phase2b2 import (
    PHASE2B2_DATASET_ID,
    create_deterministic_tar_gz,
    file_tree_manifest,
    restore_validate_archive,
    write_json_once,
)
from langmani.v2.push_dataset import DATASET_SPLITS, PushDatasetContractError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--restore-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _archive_one(
    *, source: Path, archive_root: Path, restore_root: Path, label: str
) -> dict[str, object]:
    tree = file_tree_manifest(source)
    digest = str(tree["tree_digest"])
    archive = archive_root / f"phase2b2-{label}-{digest.removeprefix('sha256:')[:16]}.tar.gz"
    created = create_deterministic_tar_gz(
        source_root=source,
        archive_path=archive,
        prefix=label,
    )
    restored = restore_validate_archive(
        archive_path=archive,
        destination=restore_root / label,
        prefix=label,
        expected_tree_digest=digest,
    )
    return {
        "label": label,
        "filename": archive.name,
        "size_bytes": created["size_bytes"],
        "sha256": created["sha256"],
        "source_tree_digest": digest,
        "restored_tree_digest": restored["restored_tree_digest"],
        "restored_file_count": restored["file_count"],
        "restore_passed": restored["passed"],
    }


def _lerobot_readback(root: Path) -> dict[str, object]:
    try:
        from lerobot.datasets import LeRobotDataset  # type: ignore[import-untyped]
    except ImportError as error:
        raise PushDatasetContractError("LeRobot 0.6 is required for archive readback") from error
    values: dict[str, object] = {}
    for split in DATASET_SPLITS:
        dataset = LeRobotDataset(
            repo_id=f"{PHASE2B2_DATASET_ID}-{split}",
            root=root / "splits" / split,
            video_backend="pyav",
            return_uint8=True,
        )
        if int(dataset.num_frames) < 1:
            raise PushDatasetContractError(f"restored LeRobot split is empty: {split}")
        item = dataset[0]
        if not isinstance(item, dict) or not isinstance(item.get("task"), str):
            raise PushDatasetContractError("restored LeRobot sample is malformed")
        values[split] = {
            "episodes": int(dataset.num_episodes),
            "frames": int(dataset.num_frames),
            "first_sample_loaded": True,
        }
    return {"splits": values, "passed": True}


def main() -> int:
    args = parse_args()
    archive_root = args.archive_root.resolve()
    restore_root = args.restore_root.resolve()
    archive_root.mkdir(parents=True, exist_ok=True)
    restore_root.mkdir(parents=True, exist_ok=True)
    values = {
        label: _archive_one(
            source=source.resolve(),
            archive_root=archive_root,
            restore_root=restore_root,
            label=label,
        )
        for label, source in (("raw", args.raw_root), ("lerobot", args.export_root))
    }
    readback = _lerobot_readback(restore_root / "lerobot" / "lerobot")
    report = {
        "schema_version": "langmani-v2-phase2b2-archive-report-v0",
        "archives": values,
        "restored_lerobot_readback": readback,
        "all_restores_passed": all(item["restore_passed"] is True for item in values.values()),
        "optimizer_steps": 0,
        "passed": all(item["restore_passed"] is True for item in values.values())
        and readback["passed"] is True,
    }
    write_json_once(args.output.resolve(), report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
