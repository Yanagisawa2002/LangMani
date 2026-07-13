"""Run one privileged M2 pick-and-place diagnostic rollout."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from langmani.environments.specs import BIN_IDS, OBJECT_IDS, TaskSpec
from langmani.experts import ExpertConfig, PickPlaceExpert
from langmani.experts.command_support import (
    create_expert_environment,
    generic_unexpected_result,
    task_reset_options,
    validated_output_path,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "diagnostics" / "m2" / "single_result.json"


def _non_negative_seed(value: str) -> int:
    try:
        seed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("seed must be an integer") from error
    if seed < 0:
        raise argparse.ArgumentTypeError("seed must be non-negative")
    return seed


def _output_path(value: str) -> Path:
    try:
        return validated_output_path(value, output_root=OUTPUT_ROOT)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=_non_negative_seed, default=0)
    parser.add_argument("--object-id", choices=OBJECT_IDS, default="red_cube")
    parser.add_argument("--bin-id", choices=BIN_IDS, default="left_bin")
    parser.add_argument(
        "--sim-backend",
        choices=("physx_cpu", "physx_cuda"),
        default="physx_cpu",
    )
    parser.add_argument("--diagnostic-rendering", action="store_true")
    parser.add_argument("--output", type=_output_path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output = validated_output_path(args.output, output_root=OUTPUT_ROOT)
    task_spec = TaskSpec(
        target_object_id=args.object_id,
        target_bin_id=args.bin_id,
        instruction_template_id="canonical_v0",
    )
    config = ExpertConfig(diagnostic_rendering=args.diagnostic_rendering)
    env: Any | None = None
    expert: PickPlaceExpert | None = None
    close_error: Exception | None = None
    try:
        env = create_expert_environment(
            diagnostic_rendering=config.diagnostic_rendering,
            sim_backend=args.sim_backend,
        )
        env.reset(seed=args.seed, options=task_reset_options(task_spec))
        artifact_directory = args.output.parent / "frames" if config.diagnostic_rendering else None
        expert = PickPlaceExpert(
            env,
            config=config,
            diagnostic_directory=artifact_directory,
        )
        result = expert.run()
    except Exception as error:  # noqa: BLE001 - this is the required outer command boundary
        result = (
            expert.unexpected_exception_result(error)
            if expert is not None
            else generic_unexpected_result(error, scene_seed=args.seed, task_spec=task_spec)
        )
        traceback.print_exc()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as error:  # noqa: BLE001 - outer command boundary
                close_error = error

    if close_error is not None:
        result = (
            expert.unexpected_exception_result(close_error)
            if expert is not None
            else generic_unexpected_result(
                close_error,
                scene_seed=args.seed,
                task_spec=task_spec,
            )
        )
        traceback.print_exception(close_error)

    output_path = write_json(args.output, result.to_dict(), output_root=OUTPUT_ROOT)
    print(json.dumps({"status": result.status.value, "output": str(output_path)}))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
