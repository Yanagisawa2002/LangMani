"""Run the disjoint 100-episode Phase 2B.2 Push expert gate."""

from __future__ import annotations

import argparse
import json
import math
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from langmani.environments.push_specs import (
    PUSH_OBJECT_IDS,
    TARGET_REGION_IDS,
    PushDifficulty,
    PushTaskSpec,
)
from langmani.environments.push_to_region import ENV_ID
from langmani.experts.push import PushToRegionExpert
from langmani.experts.push_types import PushExpertConfig, PushExpertResult, PushExpertStatus
from langmani.v2.phase2b2 import write_json_once
from langmani.v2.push_collection import inspect_phase2b_runtime
from langmani.v2.push_dataset import PushCollectionConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "langmani_v2" / "phase_2b2" / "dataset_contract.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-seed-start", type=int)
    parser.add_argument("--probe-standard-episodes", type=int)
    parser.add_argument("--probe-hard-episodes", type=int)
    return parser.parse_args()


def _schedule(
    config: PushCollectionConfig,
    *,
    seed_start: int | None = None,
    standard_episodes: int | None = None,
    hard_episodes: int | None = None,
) -> tuple[tuple[int, PushTaskSpec], ...]:
    gate = config.payload["expert_evaluation"]
    assert isinstance(gate, dict)
    start = int(gate["seed_start"]) if seed_start is None else seed_start
    values: list[tuple[int, PushTaskSpec]] = []
    index = 0
    combinations = tuple(
        (object_id, region_id) for object_id in PUSH_OBJECT_IDS for region_id in TARGET_REGION_IDS
    )
    difficulty_counts: tuple[tuple[PushDifficulty, int], ...] = (
        (
            "standard",
            int(gate["standard_episodes"]) if standard_episodes is None else standard_episodes,
        ),
        ("hard", int(gate["hard_episodes"]) if hard_episodes is None else hard_episodes),
    )
    if start < 0 or any(count < 0 for _, count in difficulty_counts):
        raise ValueError("expert evaluation seed and episode counts must be non-negative")
    if sum(count for _, count in difficulty_counts) < 1:
        raise ValueError("expert evaluation schedule cannot be empty")
    for difficulty, count in difficulty_counts:
        for offset in range(count):
            object_id, region_id = combinations[offset % len(combinations)]
            values.append((start + index, PushTaskSpec(object_id, region_id, difficulty)))
            index += 1
    return tuple(values)


def _environment() -> Any:
    import gymnasium as gym

    import langmani.environments  # noqa: F401

    return gym.make(
        ENV_ID,
        num_envs=1,
        obs_mode="state_dict",
        reward_mode="none",
        control_mode="pd_joint_pos",
        render_mode=None,
        sim_backend="physx_cuda",
    )


def _probe_stop_reason(
    *,
    completed: int,
    total: int,
    successes: int,
    minimum_success_rate: float,
    zero_tolerance_failure: bool,
) -> str | None:
    if not 0 <= successes <= completed <= total or total < 1:
        raise ValueError("probe progress counters are inconsistent")
    if not 0.0 <= minimum_success_rate <= 1.0:
        raise ValueError("probe minimum_success_rate must be in [0, 1]")
    if zero_tolerance_failure:
        return "zero_tolerance_failure"
    required_successes = math.ceil(minimum_success_rate * total)
    if successes + (total - completed) < required_successes:
        return "success_ceiling_below_gate"
    return None


def _sanitized_runtime(runtime: dict[str, object]) -> dict[str, object]:
    gpu_values = runtime.get("gpu")
    gpus: list[dict[str, object]] = []
    if isinstance(gpu_values, list):
        for value in gpu_values:
            if isinstance(value, dict):
                gpus.append({key: value.get(key) for key in ("name", "driver", "memory_mib")})
    return {
        key: runtime.get(key)
        for key in (
            "python",
            "cuda_runtime",
            "torch",
            "numpy",
            "mplib",
            "mani_skill",
            "lerobot",
            "git_commit",
            "git_clean",
            "accepted_expert_commit",
        )
    } | {"gpu": gpus}


def main() -> int:
    args = parse_args()
    config = PushCollectionConfig.load(args.config)
    runtime = inspect_phase2b_runtime(config)
    probe_values = (
        args.probe_seed_start,
        args.probe_standard_episodes,
        args.probe_hard_episodes,
    )
    if any(value is not None for value in probe_values) and not all(
        value is not None for value in probe_values
    ):
        raise ValueError("diagnostic probe requires all three --probe-* arguments")
    probe_mode = all(value is not None for value in probe_values)
    schedule = _schedule(
        config,
        seed_start=args.probe_seed_start,
        standard_episodes=args.probe_standard_episodes,
        hard_episodes=args.probe_hard_episodes,
    )
    results: list[PushExpertResult] = []
    command_errors: list[dict[str, object]] = []
    probe_stopped_early = False
    probe_stop_reason: str | None = None
    gate = config.payload["expert_evaluation"]
    assert isinstance(gate, dict)
    minimum_success_rate = float(gate["minimum_success_rate"])
    for index, (seed, task) in enumerate(schedule):
        environment: Any | None = None
        expert: PushToRegionExpert | None = None
        try:
            environment = _environment()
            environment.reset(seed=seed, options={"task_spec": task.to_dict()})
            expert = PushToRegionExpert(environment, config=PushExpertConfig())
            try:
                result = expert.run()
            except Exception as error:  # noqa: BLE001 - gate preserves classified failure
                result = expert.unexpected_exception_result(error)
            results.append(result)
            print(
                f"[{index + 1}/{len(schedule)}] seed={seed} "
                f"task={task.target_object_id}/{task.target_region_id}/{task.difficulty} "
                f"status={result.status.value} steps={result.total_environment_steps}",
                flush=True,
            )
            if probe_mode:
                evaluation = result.final_environment_evaluation
                zero_tolerance_failure = bool(
                    result.status is PushExpertStatus.UNEXPECTED_EXCEPTION
                    or evaluation.get("invalid_action", False)
                    or evaluation.get("action_out_of_bounds", False)
                    or evaluation.get("target_outside_workspace", False)
                )
                probe_stop_reason = _probe_stop_reason(
                    completed=len(results),
                    total=len(schedule),
                    successes=sum(item.success for item in results),
                    minimum_success_rate=minimum_success_rate,
                    zero_tolerance_failure=zero_tolerance_failure,
                )
                if probe_stop_reason is not None:
                    probe_stopped_early = True
                    print(
                        f"diagnostic probe stopped early: {probe_stop_reason}",
                        flush=True,
                    )
                    break
        except Exception as error:  # noqa: BLE001 - simulator construction is evidence
            traceback.print_exc()
            command_errors.append(
                {"seed": seed, "type": type(error).__name__, "message": str(error)}
            )
            break
        finally:
            if environment is not None:
                environment.close()
    successes = sum(result.success for result in results)
    evaluations = [result.final_environment_evaluation for result in results]
    simulator_errors = len(command_errors) + sum(
        result.status is PushExpertStatus.UNEXPECTED_EXCEPTION for result in results
    )
    nonfinite_actions = sum(bool(value.get("invalid_action", False)) for value in evaluations)
    action_bound_violations = sum(
        bool(value.get("action_out_of_bounds", False)) for value in evaluations
    )
    workspace_violations = sum(
        bool(value.get("target_outside_workspace", False)) for value in evaluations
    )
    timeouts = sum(result.status is PushExpertStatus.TIMEOUT for result in results)
    success_rate = successes / len(schedule)
    passed = bool(
        not probe_mode
        and len(results) == len(schedule)
        and success_rate >= 0.95
        and simulator_errors == 0
        and nonfinite_actions == 0
        and action_bound_violations == 0
        and workspace_violations == 0
    )
    report = {
        "schema_version": "langmani-v2-phase2b2-expert-evaluation-v0",
        "dataset_id": "langmani/phase2b-push-v2",
        "evaluation_mode": "diagnostic_probe" if probe_mode else "formal_gate",
        "formal_gate_eligible": not probe_mode,
        "runtime": _sanitized_runtime(runtime),
        "schedule": [{"seed": seed, "task_spec": task.to_dict()} for seed, task in schedule],
        "expected_episodes": len(schedule),
        "completed_episodes": len(results),
        "successes": successes,
        "failures": len(results) - successes,
        "success_rate": success_rate,
        "status_counts": dict(sorted(Counter(result.status.value for result in results).items())),
        "simulator_errors": simulator_errors,
        "nonfinite_actions": nonfinite_actions,
        "action_bound_violations": action_bound_violations,
        "workspace_violations": workspace_violations,
        "timeouts": timeouts,
        "teleport_or_state_mutation_used": False,
        "success_flag_mutation_used": False,
        "controller_execution_only": True,
        "command_errors": command_errors,
        "results": [result.to_dict() for result in results],
        "optimizer_steps": 0,
        "passed": passed,
        "probe_completed": bool(
            probe_mode
            and len(results) == len(schedule)
            and simulator_errors == 0
            and nonfinite_actions == 0
            and action_bound_violations == 0
            and workspace_violations == 0
        ),
        "probe_stopped_early": probe_stopped_early,
        "probe_stop_reason": probe_stop_reason,
    }
    write_json_once(args.output, report)
    print(
        json.dumps(
            {key: report[key] for key in ("successes", "failures", "success_rate", "passed")},
            sort_keys=True,
        )
    )
    if probe_mode:
        return 0 if report["probe_completed"] is True else 2
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
