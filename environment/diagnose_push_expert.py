"""Replay the fixed failed lateral Phase 2 episodes with phase-boundary telemetry."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from langmani.environments.push_specs import PushTaskSpec
from langmani.environments.push_to_region import ENV_ID
from langmani.experts.command_support import validated_output_path, write_json
from langmani.experts.push import PushToRegionExpert
from langmani.experts.push_diagnostics import (
    classify_push_failure,
    count_contact_losses,
    root_cause_counts,
)
from langmani.experts.push_types import PushExpertConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_BASELINE = (
    OUTPUT_ROOT / "diagnostics" / "v2" / "phase2" / "push-expert-target-validation-500b09c.json"
)
DEFAULT_OUTPUT = OUTPUT_ROOT / "diagnostics" / "v2" / "phase2" / "side-push-failure-replay.json"
LATERAL_REGIONS = frozenset({"left", "right"})


def _output(value: str) -> Path:
    try:
        return validated_output_path(value, output_root=OUTPUT_ROOT)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=_output, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=_output, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sim-backend",
        choices=("physx_cpu", "physx_cuda"),
        default="physx_cpu",
    )
    return parser.parse_args()


def _create_environment(sim_backend: str) -> Any:
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


def _failed_lateral_schedule(payload: dict[str, Any]) -> tuple[tuple[int, PushTaskSpec], ...]:
    if payload.get("mode") != "target_validation" or payload.get("expected_episodes") != 80:
        raise ValueError("baseline report must be the fixed 50/30 target validation")
    selected: list[tuple[int, PushTaskSpec]] = []
    for raw in payload.get("results", []):
        if raw.get("success") is True or raw.get("target_region_id") not in LATERAL_REGIONS:
            continue
        task = PushTaskSpec(
            target_object_id=raw["target_object_id"],
            target_region_id=raw["target_region_id"],
            difficulty=raw["difficulty"],
        )
        selected.append((int(raw["scene_seed"]), task))
    if not selected:
        raise ValueError("baseline report contains no failed lateral episodes")
    return tuple(selected)


def _first_contact_proxy(trace: tuple[Any, ...]) -> list[float] | None:
    snapshot = next(
        (
            item
            for item in trace
            if item.phase in {"establish_contact", "primary_push"} and item.contact_proxy
        ),
        None,
    )
    return list(snapshot.tcp_pose[:3]) if snapshot is not None else None


def _push_segments(trace: tuple[Any, ...]) -> list[dict[str, object]]:
    phases = {"primary_push", "corrective_push_1", "corrective_push_2"}
    return [
        {
            "phase": item.phase,
            "environment_steps": item.environment_steps,
            "target_position": list(item.target_object_pose[:3]),
            "target_distance": item.target_distance,
            "projected_progress": item.projected_progress,
            "lateral_error": item.lateral_error,
            "contact_proxy": item.contact_proxy,
            "workspace_margin": item.workspace_margin,
        }
        for item in trace
        if item.phase in phases
    ]


def main() -> int:
    args = parse_args()
    baseline_path = validated_output_path(args.baseline_report, output_root=OUTPUT_ROOT)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if not isinstance(baseline, dict):
        raise TypeError("baseline report must contain one JSON object")
    schedule = _failed_lateral_schedule(baseline)
    episodes: list[dict[str, object]] = []
    causes: list[Any] = []
    command_errors: list[dict[str, str]] = []

    for index, (seed, task) in enumerate(schedule):
        environment: Any | None = None
        try:
            environment = _create_environment(args.sim_backend)
            environment.reset(seed=seed, options={"task_spec": task.to_dict()})
            expert = PushToRegionExpert(environment, config=PushExpertConfig())
            result = expert.run()
            trace = expert.diagnostic_trace
            evaluation = dict(result.final_environment_evaluation)
            cause = classify_push_failure(
                status=result.status.value,
                failed_phase=result.failed_phase.value if result.failed_phase is not None else None,
                evaluation=evaluation,
                trace=trace,
            )
            causes.append(cause)
            first = trace[0]
            last = trace[-1]
            episodes.append(
                {
                    "task_id": result.task_id,
                    "seed": seed,
                    "object_geometry": task.target_object_id,
                    "target_direction": task.target_region_id,
                    "difficulty": task.difficulty,
                    "initial_object_pose": list(first.target_object_pose),
                    "target_pose": list(first.target_center),
                    "chosen_approach_point": (
                        list(last.chosen_precontact_point)
                        if last.chosen_precontact_point is not None
                        else None
                    ),
                    "intended_contact_normal": list(first.intended_push_direction),
                    "actual_first_contact_point_proxy": _first_contact_proxy(trace),
                    "actual_contact_source": "tcp_center_phase_boundary_proxy",
                    "push_segments": _push_segments(trace),
                    "contact_losses": count_contact_losses(trace),
                    "correction_attempts": sum(
                        item.phase.startswith("corrective_push") and item.environment_steps > 0
                        for item in trace
                    ),
                    "planner_failures": sum(
                        not item.success and item.status.value in {"planning_failure", "ik_failure"}
                        for item in result.phase_results
                    ),
                    "action_clipping": False,
                    "action_projection": False,
                    "action_out_of_bounds": evaluation.get("action_out_of_bounds") is True,
                    "minimum_workspace_margin": min(item.workspace_margin for item in trace),
                    "final_object_to_target_distance": last.target_distance,
                    "result": result.to_dict(),
                    "trace": [item.to_dict() for item in trace],
                    "primary_root_cause": cause,
                }
            )
            print(
                f"[{index + 1}/{len(schedule)}] seed={seed} "
                f"task={task.target_object_id}/{task.target_region_id}/{task.difficulty} "
                f"status={result.status.value} root_cause={cause}",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001 - diagnostic batch preserves failures
            traceback.print_exc()
            command_errors.append(
                {"seed": str(seed), "type": type(error).__name__, "message": str(error)}
            )
        finally:
            if environment is not None:
                environment.close()

    completed = len(episodes) == len(schedule) and not command_errors
    payload = {
        "schema_version": "langmani-v2-side-push-failure-replay-v0",
        "baseline_report": str(baseline_path.relative_to(PROJECT_ROOT)),
        "baseline_completed_episodes": baseline.get("completed_episodes"),
        "expected_replays": len(schedule),
        "completed_replays": len(episodes),
        "replay_completed": completed,
        "root_cause_counts": root_cause_counts(causes),
        "command_errors": command_errors,
        "episodes": episodes,
    }
    path = write_json(args.output, payload, output_root=OUTPUT_ROOT)
    print(json.dumps({"completed": completed, "output": str(path)}, sort_keys=True))
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
