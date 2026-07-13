"""Inspect M3A manifests, source shards, checksums, and balanced group structure."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from langmani.collection.command_support import validated_dataset_root, write_json_atomic
from langmani.collection.inspection import inspect_raw_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_DATASET_ROOT = OUTPUT_ROOT / "datasets" / "m3a" / "langmani-pick-place-raw-v1"
DEFAULT_REPORT = OUTPUT_ROOT / "diagnostics" / "m3a" / "inspection.json"


def _dataset_root(value: str) -> Path:
    try:
        return validated_dataset_root(value, output_root=OUTPUT_ROOT)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=_dataset_root, default=DEFAULT_DATASET_ROOT)
    return parser.parse_args()


def main() -> int:
    DEFAULT_REPORT.unlink(missing_ok=True)
    args = parse_args()
    dataset_root = validated_dataset_root(args.dataset_root, output_root=OUTPUT_ROOT)
    try:
        summary = inspect_raw_dataset(dataset_root)
        report = {
            "command": "inspect_raw_demos",
            "dataset_root": str(dataset_root),
            "summary": summary.to_dict(),
            "passed": True,
        }
        exit_code = 0
    except Exception as error:  # noqa: BLE001 - inspection command boundary
        traceback.print_exc()
        report = {
            "command": "inspect_raw_demos",
            "dataset_root": str(dataset_root),
            "passed": False,
            "exception_type": type(error).__name__,
            "exception_message": str(error) or repr(error),
        }
        exit_code = 1
    report_path = write_json_atomic(DEFAULT_REPORT, report)
    print(json.dumps({"passed": report["passed"], "report": str(report_path)}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
