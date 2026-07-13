"""Sequentially benchmark the M2 expert over all six semantic tasks."""

from __future__ import annotations

import argparse
import json
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from langmani.environments.pick_place_by_instruction import ENV_ID
from langmani.environments.specs import BIN_IDS, OBJECT_IDS, TaskSpec
from langmani.experts import ExpertConfig, PickPlaceExpert
from langmani.experts.command_support import (
    create_expert_environment,
    generic_unexpected_result,
    task_reset_options,
    validated_output_path,
    write_json,
)
from langmani.experts.types import ExpertResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "diagnostics" / "m2" / "benchmark.json"


def _parse_seeds(value: str) -> tuple[int, ...]:
    parts = value.split(",")
    if any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError("seeds must not contain empty entries")
    try:
        seeds = tuple(int(part.strip()) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not seeds or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must contain non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must not contain duplicates")
    return seeds


def _output_path(value: str) -> Path:
    try:
        return validated_output_path(value, output_root=OUTPUT_ROOT)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=_parse_seeds, default=(0,))
    parser.add_argument(
        "--sim-backend",
        choices=("physx_cpu", "physx_cuda"),
        default="physx_cpu",
    )
    parser.add_argument("--diagnostic-rendering", action="store_true")
    parser.add_argument("--output", type=_output_path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _task_specs() -> tuple[TaskSpec, ...]:
    return tuple(
        TaskSpec(
            target_object_id=object_id,
            target_bin_id=bin_id,
            instruction_template_id="canonical_v0",
        )
        for object_id in OBJECT_IDS
        for bin_id in BIN_IDS
    )


def main() -> int:
    args = parse_args()
    args.output = validated_output_path(args.output, output_root=OUTPUT_ROOT)
    config = ExpertConfig(diagnostic_rendering=args.diagnostic_rendering)
    results: list[ExpertResult] = []
    env: Any | None = None
    command_errors: list[dict[str, str]] = []
    try:
        env = create_expert_environment(
            diagnostic_rendering=config.diagnostic_rendering,
            sim_backend=args.sim_backend,
        )
        for scene_seed in args.seeds:
            for task_spec in _task_specs():
                expert: PickPlaceExpert | None = None
                try:
                    env.reset(seed=scene_seed, options=task_reset_options(task_spec))
                    frame_dir = None
                    if config.diagnostic_rendering:
                        frame_dir = (
                            args.output.parent
                            / "frames"
                            / f"seed_{scene_seed}"
                            / f"{task_spec.target_object_id}_{task_spec.target_bin_id}"
                        )
                    expert = PickPlaceExpert(
                        env,
                        config=config,
                        diagnostic_directory=frame_dir,
                    )
                    result = expert.run()
                except Exception as error:  # noqa: BLE001 - batch boundary collects failures
                    result = (
                        expert.unexpected_exception_result(error)
                        if expert is not None
                        else generic_unexpected_result(
                            error,
                            scene_seed=scene_seed,
                            task_spec=task_spec,
                        )
                    )
                    traceback.print_exc()
                results.append(result)
                print(
                    f"seed={scene_seed} task={task_spec.target_object_id}/"
                    f"{task_spec.target_bin_id} status={result.status.value}"
                )
    except Exception as error:  # noqa: BLE001 - outer benchmark boundary
        traceback.print_exc()
        command_errors.append({"type": type(error).__name__, "message": str(error)})
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as error:  # noqa: BLE001 - outer benchmark boundary
                traceback.print_exc()
                command_errors.append({"type": type(error).__name__, "message": str(error)})

    status_counts = Counter(result.status.value for result in results)
    expected_rollouts = len(args.seeds) * len(OBJECT_IDS) * len(BIN_IDS)
    payload = {
        "schema_version": "langmani-m2-benchmark-v0",
        "environment_id": ENV_ID,
        "seeds": list(args.seeds),
        "config": config.to_dict(),
        "expected_rollouts": expected_rollouts,
        "completed_rollouts": len(results),
        "successful_rollouts": sum(result.success for result in results),
        "status_counts": dict(sorted(status_counts.items())),
        "command_errors": command_errors,
        "results": [result.to_dict() for result in results],
    }
    output_path = write_json(args.output, payload, output_root=OUTPUT_ROOT)
    print(json.dumps({"status_counts": payload["status_counts"], "output": str(output_path)}))
    all_passed = (
        not command_errors
        and len(results) == expected_rollouts
        and all(result.success for result in results)
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
