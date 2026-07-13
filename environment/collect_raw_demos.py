"""Collect or resume the authoritative LangMani M3A raw demonstration archive."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from langmani.collection import RawDemonstrationCollector
from langmani.collection.command_support import validated_dataset_root, write_json_atomic
from langmani.collection.manifest import installed_runtime_versions
from langmani.datasets.types import (
    CollectionConfig,
    CollectionStatus,
    OverwritePolicy,
    ReplayValidationMode,
    ResumePolicy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_DATASET_ROOT = OUTPUT_ROOT / "datasets" / "m3a" / "langmani-pick-place-raw-v1"
COMMAND_REPORT = OUTPUT_ROOT / "diagnostics" / "m3a" / "collection_command.json"


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a real number") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _dataset_root(value: str) -> Path:
    try:
        return validated_dataset_root(value, output_root=OUTPUT_ROOT)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=_dataset_root, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--candidate-seed-start", type=_non_negative_int, default=0)
    parser.add_argument("--target-complete-scene-count", type=_positive_int, default=60)
    parser.add_argument("--maximum-candidate-scene-count", type=_positive_int, default=120)
    parser.add_argument("--maximum-expert-attempts-per-task", type=_positive_int, default=3)
    parser.add_argument("--shard-size", type=_positive_int, default=60)
    parser.add_argument(
        "--sim-backend",
        choices=("physx_cpu", "physx_cuda"),
        default="physx_cpu",
    )
    parser.add_argument(
        "--replay-validation-mode",
        choices=tuple(mode.value for mode in ReplayValidationMode),
        default=ReplayValidationMode.ACTION_AND_STATE_AUDIT.value,
    )
    parser.add_argument("--final-position-tolerance-m", type=_positive_float, default=0.005)
    parser.add_argument(
        "--final-orientation-tolerance-rad",
        type=_positive_float,
        default=0.05,
    )
    parser.add_argument("--final-joint-tolerance-rad", type=_positive_float, default=0.05)
    parser.add_argument("--retain-failed-raw-trajectories", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    COMMAND_REPORT.unlink(missing_ok=True)
    args = parse_args()
    output_root = validated_dataset_root(args.output_root, output_root=OUTPUT_ROOT)
    report: dict[str, object]
    try:
        config = CollectionConfig(
            candidate_scene_seed_start=args.candidate_seed_start,
            target_complete_scene_count=args.target_complete_scene_count,
            maximum_candidate_scene_count=args.maximum_candidate_scene_count,
            maximum_expert_attempts_per_task=args.maximum_expert_attempts_per_task,
            raw_output_root=str(output_root),
            shard_size=args.shard_size,
            sim_backend=args.sim_backend,
            retain_failed_raw_trajectories=args.retain_failed_raw_trajectories,
            replay_validation_mode=ReplayValidationMode(args.replay_validation_mode),
            final_position_tolerance_m=args.final_position_tolerance_m,
            final_orientation_tolerance_rad=args.final_orientation_tolerance_rad,
            final_joint_tolerance_rad=args.final_joint_tolerance_rad,
            overwrite_policy=(OverwritePolicy.REPLACE if args.overwrite else OverwritePolicy.ERROR),
            resume_policy=(ResumePolicy.ERROR if args.no_resume else ResumePolicy.RESUME),
            runtime_versions=installed_runtime_versions(),
        )
        summary = RawDemonstrationCollector(config).collect()
        report = {
            "command": "collect_raw_demos",
            "dataset_root": str(output_root),
            "config": config.to_dict(),
            "summary": summary.to_dict(),
            "passed": summary.status is CollectionStatus.COMPLETE,
        }
        exit_code = 0 if summary.status is CollectionStatus.COMPLETE else 1
    except Exception as error:  # noqa: BLE001 - required command boundary
        traceback.print_exc()
        report = {
            "command": "collect_raw_demos",
            "dataset_root": str(output_root),
            "passed": False,
            "exception_type": type(error).__name__,
            "exception_message": str(error) or repr(error),
        }
        exit_code = 1
    report_path = write_json_atomic(COMMAND_REPORT, report)
    print(json.dumps({"passed": report["passed"], "report": str(report_path)}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
