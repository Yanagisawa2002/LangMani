"""Validate the Phase 2 pushing expert on deterministic standard and hard schedules."""

from __future__ import annotations

import argparse
import json
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langmani.environments.push_specs import (
    PUSH_OBJECT_IDS,
    TARGET_REGION_IDS,
    PushTaskSpec,
)
from langmani.environments.push_to_region import ENV_ID
from langmani.experts.command_support import validated_output_path, write_json
from langmani.experts.push import PushToRegionExpert
from langmani.experts.push_types import PushExpertConfig, PushExpertResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_OUTPUT = OUTPUT_ROOT / "diagnostics" / "v2" / "phase2" / "push-expert-benchmark.json"


@dataclass(frozen=True, slots=True)
class ScheduledPushEpisode:
    seed: int
    task_spec: PushTaskSpec

    def to_dict(self) -> dict[str, object]:
        return {"seed": self.seed, "task_spec": self.task_spec.to_dict()}


def build_schedule(*, smoke: bool) -> tuple[ScheduledPushEpisode, ...]:
    """Build either all-task smoke or the fixed 50/30 quality schedule."""

    standard_count, hard_count = (8, 8) if smoke else (50, 30)
    episodes: list[ScheduledPushEpisode] = []
    for difficulty, count, seed_start in (
        ("standard", standard_count, 44_000),
        ("hard", hard_count, 45_000),
    ):
        combinations = tuple(
            (object_id, region_id)
            for object_id in PUSH_OBJECT_IDS
            for region_id in TARGET_REGION_IDS
        )
        for index in range(count):
            object_id, region_id = combinations[index % len(combinations)]
            episodes.append(
                ScheduledPushEpisode(
                    seed=seed_start + index,
                    task_spec=PushTaskSpec(object_id, region_id, difficulty),
                )
            )
    return tuple(episodes)


def _output(value: str) -> Path:
    try:
        return validated_output_path(value, output_root=OUTPUT_ROOT)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--target-validation", action="store_true")
    parser.add_argument(
        "--sim-backend",
        choices=("physx_cpu", "physx_cuda"),
        default="physx_cpu",
    )
    parser.add_argument("--output", type=_output, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _create_environment(sim_backend: str) -> object:
    import gymnasium as gym

    import langmani.environments  # noqa: F401 - registers ENV_ID

    return gym.make(
        ENV_ID,
        num_envs=1,
        obs_mode="state_dict",
        reward_mode="none",
        control_mode="pd_joint_pos",
        render_mode=None,
        sim_backend=sim_backend,
    )


def _event_counts(results: list[PushExpertResult]) -> dict[str, int]:
    names = (
        "wrong_object_contact",
        "wrong_object_displaced",
        "target_outside_workspace",
        "target_overshoot",
        "target_toppled",
        "target_lifted",
        "target_is_grasped",
        "invalid_action",
        "action_out_of_bounds",
        "no_progress_stall",
        "robot_collision",
    )
    return {
        name: sum(result.final_environment_evaluation.get(name) is True for result in results)
        for name in names
    }


def _difficulty_summary(
    results: list[PushExpertResult],
    difficulty: str,
) -> dict[str, object]:
    selected = [result for result in results if result.difficulty == difficulty]
    successes = sum(result.success for result in selected)
    return {
        "episodes": len(selected),
        "successes": successes,
        "success_rate": successes / len(selected) if selected else 0.0,
        "status_counts": dict(sorted(Counter(item.status.value for item in selected).items())),
    }


def main() -> int:
    args = parse_args()
    output = validated_output_path(args.output, output_root=OUTPUT_ROOT)
    schedule = build_schedule(smoke=args.smoke)
    expected = 16 if args.smoke else 80
    if len(schedule) != expected or len({item.seed for item in schedule}) != expected:
        raise RuntimeError("push expert schedule contract changed")

    config = PushExpertConfig()
    environment: Any | None = None
    results: list[PushExpertResult] = []
    command_errors: list[dict[str, str]] = []
    try:
        environment = _create_environment(args.sim_backend)
        for index, scheduled in enumerate(schedule):
            expert: PushToRegionExpert | None = None
            try:
                environment.reset(
                    seed=scheduled.seed,
                    options={"task_spec": scheduled.task_spec.to_dict()},
                )
                expert = PushToRegionExpert(environment, config=config)
                result = expert.run()
            except Exception as error:  # noqa: BLE001 - batch keeps classified failures
                traceback.print_exc()
                if expert is None:
                    command_errors.append(
                        {
                            "seed": str(scheduled.seed),
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    )
                    break
                result = expert.unexpected_exception_result(error)
            results.append(result)
            print(
                f"[{index + 1}/{expected}] seed={scheduled.seed} "
                f"task={result.target_object_id}/{result.target_region_id}/"
                f"{result.difficulty} status={result.status.value} "
                f"steps={result.total_environment_steps}",
                flush=True,
            )
    except Exception as error:  # noqa: BLE001 - outer benchmark boundary
        traceback.print_exc()
        command_errors.append({"type": type(error).__name__, "message": str(error)})
    finally:
        if environment is not None:
            try:
                environment.close()
            except Exception as error:  # noqa: BLE001 - outer benchmark boundary
                traceback.print_exc()
                command_errors.append({"type": type(error).__name__, "message": str(error)})

    standard = _difficulty_summary(results, "standard")
    hard = _difficulty_summary(results, "hard")
    completed = len(results) == expected and not command_errors
    standard_quality = standard["success_rate"] >= 0.90
    hard_quality = hard["success_rate"] >= 0.70
    payload = {
        "schema_version": "langmani-v2-push-expert-benchmark-v0",
        "mode": "smoke" if args.smoke else "target_validation",
        "environment_id": ENV_ID,
        "config": config.to_dict(),
        "schedule": [item.to_dict() for item in schedule],
        "expected_episodes": expected,
        "completed_episodes": len(results),
        "benchmark_completed": completed,
        "standard": standard,
        "hard": hard,
        "standard_quality_target_passed": standard_quality,
        "hard_quality_target_passed": hard_quality,
        "expert_quality_targets_passed": standard_quality and hard_quality,
        "event_counts": _event_counts(results),
        "command_errors": command_errors,
        "results": [result.to_dict() for result in results],
    }
    path = write_json(output, payload, output_root=OUTPUT_ROOT)
    print(
        json.dumps(
            {
                "benchmark_completed": completed,
                "standard": standard,
                "hard": hard,
                "expert_quality_targets_passed": standard_quality and hard_quality,
                "output": str(path),
            },
            sort_keys=True,
        )
    )
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
