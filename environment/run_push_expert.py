"""Run one deterministic privileged Phase 2 planar-pushing expert rollout."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from langmani.environments.push_specs import (
    PUSH_DIFFICULTIES,
    PUSH_OBJECT_IDS,
    TARGET_REGION_IDS,
    PushTaskSpec,
)
from langmani.environments.push_to_region import ENV_ID
from langmani.experts.command_support import validated_output_path, write_json
from langmani.experts.push import PushToRegionExpert
from langmani.experts.push_types import (
    PushExpertConfig,
    PushExpertPhase,
    PushExpertResult,
    PushExpertStatus,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_OUTPUT = OUTPUT_ROOT / "diagnostics" / "v2" / "phase2" / "push-expert.json"


def _seed(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("seed must be an integer") from error
    if result < 0:
        raise argparse.ArgumentTypeError("seed must be non-negative")
    return result


def _output(value: str) -> Path:
    try:
        return validated_output_path(value, output_root=OUTPUT_ROOT)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=_seed, default=43001)
    parser.add_argument("--object-id", choices=PUSH_OBJECT_IDS, default="blue_cube")
    parser.add_argument("--target-region", choices=TARGET_REGION_IDS, default="left")
    parser.add_argument("--difficulty", choices=PUSH_DIFFICULTIES, default="standard")
    parser.add_argument(
        "--sim-backend",
        choices=("physx_cpu", "physx_cuda"),
        default="physx_cpu",
    )
    parser.add_argument("--diagnostic-rendering", action="store_true")
    parser.add_argument("--output", type=_output, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _create_environment(*, sim_backend: str, render: bool) -> object:
    import gymnasium as gym

    import langmani.environments  # noqa: F401 - registers ENV_ID

    return gym.make(
        ENV_ID,
        num_envs=1,
        obs_mode="state_dict",
        reward_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array" if render else None,
        sim_backend=sim_backend,
    )


def _unexpected(error: BaseException, *, seed: int, task: PushTaskSpec) -> PushExpertResult:
    return PushExpertResult(
        success=False,
        status=PushExpertStatus.UNEXPECTED_EXCEPTION,
        scene_seed=seed,
        scene_id=None,
        task_id=None,
        canonical_instruction=None,
        target_object_id=task.target_object_id,
        target_region_id=task.target_region_id,
        difficulty=task.difficulty,
        total_environment_steps=0,
        total_planning_calls=0,
        completed_phases=(),
        failed_phase=PushExpertPhase.INITIALIZE,
        final_environment_evaluation={},
        phase_results=(),
        planning_duration_seconds=0.0,
        execution_duration_seconds=0.0,
        exception_type=type(error).__name__,
        exception_message=str(error) or repr(error),
    )


def main() -> int:
    args = parse_args()
    output = validated_output_path(args.output, output_root=OUTPUT_ROOT)
    task = PushTaskSpec(
        target_object_id=args.object_id,
        target_region_id=args.target_region,
        difficulty=args.difficulty,
    )
    config = PushExpertConfig(diagnostic_rendering=args.diagnostic_rendering)
    environment: Any | None = None
    expert: PushToRegionExpert | None = None
    try:
        environment = _create_environment(
            sim_backend=args.sim_backend,
            render=args.diagnostic_rendering,
        )
        environment.reset(seed=args.seed, options={"task_spec": task.to_dict()})
        expert = PushToRegionExpert(
            environment,
            config=config,
            diagnostic_directory=(output.parent / "frames"),
        )
        result = expert.run()
    except Exception as error:  # noqa: BLE001 - required outer command boundary
        traceback.print_exc()
        result = (
            expert.unexpected_exception_result(error)
            if expert is not None
            else _unexpected(error, seed=args.seed, task=task)
        )
    finally:
        if environment is not None:
            try:
                environment.close()
            except Exception:  # noqa: BLE001 - preserve the primary diagnostic result
                traceback.print_exc()

    report = {
        "schema_version": "langmani-v2-push-expert-command-v0",
        "environment_id": ENV_ID,
        "config": config.to_dict(),
        "result": result.to_dict(),
        "action_count": len(expert.action_trace) if expert is not None else 0,
    }
    path = write_json(output, report, output_root=OUTPUT_ROOT)
    print(json.dumps({"status": result.status.value, "output": str(path)}))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
