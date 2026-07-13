"""Export a validated M3A archive to one staged local LeRobotDataset v3."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from langmani.collection.command_support import validated_dataset_root, write_json_atomic
from langmani.datasets.lerobot_export import build_export_plan, export_lerobot_dataset
from langmani.datasets.lerobot_types import ExportMode, LeRobotExportConfig, SplitConfig
from langmani.datasets.lerobot_writer import inspect_lerobot_runtime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_SOURCE_ROOT = OUTPUT_ROOT / "datasets" / "m3a" / "langmani-pick-place-raw-v1"
DEFAULT_OUTPUT_ROOT = OUTPUT_ROOT / "datasets" / "m3b" / "langmani-pick-place-lerobot-v1"
DEFAULT_SOURCE_REPORT = OUTPUT_ROOT / "diagnostics" / "m3a" / "verification.json"
DEFAULT_COMMAND_REPORT = OUTPUT_ROOT / "diagnostics" / "m3b" / "export_command.json"


def _dataset_root(value: str) -> Path:
    try:
        return validated_dataset_root(value, output_root=OUTPUT_ROOT)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=_dataset_root, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=_dataset_root, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--repo-id",
        default="langmani/pick-place-by-instruction-v0",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="Export one validated six-task group.")
    mode.add_argument("--full", action="store_true", help="Export the validated 60-group archive.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--split-seed", type=_non_negative_int, default=0)
    parser.add_argument("--clean-staging", action="store_true")
    parser.add_argument(
        "--source-verification-report",
        type=Path,
        default=DEFAULT_SOURCE_REPORT,
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_COMMAND_REPORT)
    return parser.parse_args()


def _source_identity(path: Path) -> dict[str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read content-bound M3A verification report: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("M3A verification report must be a JSON object")
    required = {
        "collection_run_id": payload.get("collection_run_id"),
        "config_fingerprint": payload.get("config_fingerprint"),
        "source_archive_digest": payload.get("source_archive_digest"),
    }
    if (
        payload.get("schema_version") != "langmani-m3a-verification-v3"
        or payload.get("source_archive_identity_status") != "validated"
    ):
        raise RuntimeError("M3A verification report does not contain validated v3 source identity")
    if not all(isinstance(value, str) and value for value in required.values()):
        raise RuntimeError("M3A verification report is missing source identity fields")
    return {key: str(value) for key, value in required.items()}


def _config(args: argparse.Namespace) -> LeRobotExportConfig:
    identity = _source_identity(args.source_verification_report.resolve())
    runtime = inspect_lerobot_runtime()
    libraries = dict(runtime.av_libraries)
    mode = ExportMode.FULL if args.full else ExportMode.SMOKE
    split = (
        SplitConfig.full(split_seed=args.split_seed)
        if mode is ExportMode.FULL
        else SplitConfig.smoke(split_seed=args.split_seed)
    )
    return LeRobotExportConfig(
        source_root=str(args.source_root.resolve()),
        output_root=str(args.output_root.resolve()),
        repo_id=args.repo_id,
        expected_source_collection_run_id=identity["collection_run_id"],
        expected_source_run_fingerprint=identity["config_fingerprint"],
        expected_source_archive_digest=identity["source_archive_digest"],
        mode=mode,
        split_config=split,
        pyav_version=runtime.av,
        libavcodec_version=libraries["libavcodec"],
    )


def main() -> int:
    args = parse_args()
    report_path = args.report.resolve()
    report_path.unlink(missing_ok=True)
    try:
        config = _config(args)
        if args.dry_run:
            plan = build_export_plan(
                config,
                source_verification_report=args.source_verification_report,
            )
            result: dict[str, Any] = plan.to_dict(config)
            result.update({"command": "export_lerobot_dataset", "passed": True})
        else:
            outcome = export_lerobot_dataset(
                config,
                source_verification_report=args.source_verification_report,
                clean_staging=args.clean_staging,
            )
            result = {
                "command": "export_lerobot_dataset",
                "dry_run": False,
                "output_root": str(outcome.output_root),
                "export_fingerprint": outcome.manifest.export_fingerprint,
                "summary": outcome.summary.to_dict(),
                "validation": outcome.validation_report.to_dict(),
                "passed": True,
            }
        exit_code = 0
    except Exception as error:  # noqa: BLE001 - outer command boundary preserves diagnostics
        traceback.print_exc()
        result = {
            "command": "export_lerobot_dataset",
            "dry_run": bool(args.dry_run),
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
