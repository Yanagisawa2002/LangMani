"""Re-run M3A action replay validation sequentially with one environment."""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

from langmani.collection.command_support import validated_dataset_root, write_json_atomic
from langmani.collection.manifest import installed_runtime_versions, load_manifest
from langmani.collection.replay import validate_action_replay
from langmani.datasets.types import ReplayValidationMode
from langmani.experts.command_support import create_expert_environment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_DATASET_ROOT = OUTPUT_ROOT / "datasets" / "m3a" / "langmani-pick-place-raw-v1"
DEFAULT_REPORT = OUTPUT_ROOT / "diagnostics" / "m3a" / "replay_validation.json"


def _dataset_root(value: str) -> Path:
    try:
        return validated_dataset_root(value, output_root=OUTPUT_ROOT)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("limit must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("limit must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=_dataset_root, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--raw-trajectory-id", action="append", default=[])
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in ReplayValidationMode),
        help="Override audit depth without changing the stored collection identity.",
    )
    return parser.parse_args()


def main() -> int:
    DEFAULT_REPORT.unlink(missing_ok=True)
    args = parse_args()
    dataset_root = validated_dataset_root(args.dataset_root, output_root=OUTPUT_ROOT)
    env: Any | None = None
    results: list[dict[str, Any]] = []
    command_errors: list[dict[str, str]] = []
    try:
        manifest = load_manifest(dataset_root)
        current_versions = installed_runtime_versions()
        if dict(manifest.runtime_versions) != current_versions:
            raise RuntimeError(
                "current runtime versions differ from the authoritative collection manifest"
            )
        requested_ids = tuple(args.raw_trajectory_id)
        known_ids = {record.raw_trajectory_id for record in manifest.raw_episodes}
        unknown = sorted(set(requested_ids) - known_ids)
        if unknown:
            raise ValueError(f"unknown raw trajectory IDs: {unknown}")
        records = [
            record
            for record in manifest.raw_episodes
            if not requested_ids or record.raw_trajectory_id in requested_ids
        ]
        if args.limit is not None:
            records = records[: args.limit]
        if not records:
            raise ValueError("no raw episodes selected for replay")
        config = manifest.config
        if args.mode is not None:
            config = replace(config, replay_validation_mode=ReplayValidationMode(args.mode))
        scheduled = {
            episode.scheduled_episode_id: episode for episode in manifest.schedule.episodes
        }
        env = create_expert_environment(
            diagnostic_rendering=False,
            sim_backend=config.sim_backend,
        )
        for record in records:
            try:
                result = validate_action_replay(
                    dataset_root / record.h5_path,
                    native_episode_id=record.native_episode_id,
                    scheduled_episode=scheduled[record.scheduled_episode_id],
                    config=config,
                    sim_backend=config.sim_backend,
                    environment=env,
                )
                results.append(
                    {
                        "raw_trajectory_id": record.raw_trajectory_id,
                        "source_shard_id": record.source_shard_id,
                        "native_episode_id": record.native_episode_id,
                        "validation": result.to_dict(),
                    }
                )
            except Exception as error:  # noqa: BLE001 - explicit batch replay boundary
                traceback.print_exc()
                results.append(
                    {
                        "raw_trajectory_id": record.raw_trajectory_id,
                        "source_shard_id": record.source_shard_id,
                        "native_episode_id": record.native_episode_id,
                        "validation": None,
                        "exception_type": type(error).__name__,
                        "exception_message": str(error) or repr(error),
                    }
                )
    except Exception as error:  # noqa: BLE001 - command boundary
        traceback.print_exc()
        command_errors.append({"type": type(error).__name__, "message": str(error) or repr(error)})
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as error:  # noqa: BLE001 - command boundary
                traceback.print_exc()
                command_errors.append(
                    {"type": type(error).__name__, "message": str(error) or repr(error)}
                )

    all_passed = (
        not command_errors
        and bool(results)
        and all(
            isinstance(item.get("validation"), dict) and item["validation"].get("passed") is True
            for item in results
        )
    )
    report = {
        "schema_version": "langmani-m3a-replay-report-v1",
        "dataset_root": str(dataset_root),
        "selected_episode_count": len(results),
        "passed_episode_count": sum(
            isinstance(item.get("validation"), dict) and item["validation"].get("passed") is True
            for item in results
        ),
        "command_errors": command_errors,
        "results": results,
        "passed": all_passed,
    }
    report_path = write_json_atomic(DEFAULT_REPORT, report)
    print(json.dumps({"passed": all_passed, "report": str(report_path)}))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
