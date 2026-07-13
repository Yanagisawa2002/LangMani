"""Independently validate a finalized local LangMani LeRobotDataset v3."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from langmani.collection.command_support import validated_dataset_root, write_json_atomic
from langmani.datasets.lerobot_types import ExportMode
from langmani.datasets.lerobot_validation import validate_lerobot_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_SOURCE_ROOT = OUTPUT_ROOT / "datasets" / "m3a" / "langmani-pick-place-raw-v1"
DEFAULT_DATASET_ROOT = OUTPUT_ROOT / "datasets" / "m3b" / "langmani-pick-place-lerobot-v1"
DEFAULT_SOURCE_REPORT = OUTPUT_ROOT / "diagnostics" / "m3a" / "verification.json"
DEFAULT_COMMAND_REPORT = OUTPUT_ROOT / "diagnostics" / "m3b" / "validation_command.json"


def _dataset_root(value: str) -> Path:
    try:
        return validated_dataset_root(value, output_root=OUTPUT_ROOT)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=_dataset_root, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--source-root", type=_dataset_root, default=DEFAULT_SOURCE_ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument(
        "--source-verification-report",
        type=Path,
        default=DEFAULT_SOURCE_REPORT,
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_COMMAND_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = args.report.resolve()
    report_path.unlink(missing_ok=True)
    try:
        validation, summary = validate_lerobot_dataset(
            args.dataset_root,
            source_root=args.source_root,
            source_verification_report=args.source_verification_report,
            require_completion_marker=True,
            metadata_only=args.metadata_only,
        )
        requested_mode = ExportMode.FULL if args.full else ExportMode.SMOKE
        if summary.mode is not requested_mode:
            raise RuntimeError(
                f"dataset mode is {summary.mode.value!r}, requested {requested_mode.value!r}"
            )
        metadata_passed = all(
            (
                validation.dataset_load_validated,
                validation.feature_schema_validated,
                validation.parquet_validated,
                validation.split_integrity_validated,
                validation.privileged_leakage_validated,
            )
        )
        command_passed = metadata_passed if args.metadata_only else validation.passed
        result = {
            "command": "validate_lerobot_dataset",
            "metadata_only": bool(args.metadata_only),
            "full_acceptance": False if args.metadata_only else validation.passed,
            "summary": summary.to_dict(),
            "validation": validation.to_dict(),
            "passed": command_passed,
        }
        exit_code = 0 if command_passed else 1
    except Exception as error:  # noqa: BLE001 - outer command boundary preserves diagnostics
        traceback.print_exc()
        result = {
            "command": "validate_lerobot_dataset",
            "metadata_only": bool(args.metadata_only),
            "passed": False,
            "exception_type": type(error).__name__,
            "exception_message": str(error) or repr(error),
        }
        exit_code = 1
    written = write_json_atomic(report_path, result)
    print(json.dumps({"passed": result["passed"], "report": str(written)}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
