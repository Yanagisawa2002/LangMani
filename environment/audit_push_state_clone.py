"""Audit complete state restoration and isolated physical push sandboxes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

import langmani.environments  # noqa: F401 - registers the environment
from langmani.environments.push_simulation_state import (
    PushSimulationSnapshot,
    capture_push_simulation_state,
    push_simulation_configuration,
    restore_push_simulation_state,
)
from langmani.environments.push_specs import PushTaskSpec
from langmani.environments.push_to_region import CONTACT_FORCE_THRESHOLD, ENV_ID
from langmani.experts.push import PushToRegionExpert
from langmani.v2.phase2b2 import write_json_once

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_OUTPUT = OUTPUT_ROOT / "diagnostics" / "v2" / "phase2b3" / "state-clone-audit.json"
ACTION_SEQUENCE_LENGTH = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=69_000)
    return parser.parse_args()


def _environment() -> Any:
    return gym.make(
        ENV_ID,
        num_envs=1,
        obs_mode="state_dict",
        reward_mode="none",
        control_mode="pd_joint_pos",
        render_mode=None,
        sim_backend="physx_cuda",
    )


def _reset(environment: Any, seed: int, task: PushTaskSpec) -> None:
    environment.reset(seed=seed, options={"task_spec": task.to_dict()})


def _numeric(value: object) -> np.ndarray:
    candidate = value
    for method_name in ("detach", "cpu"):
        method = getattr(candidate, method_name, None)
        if callable(method):
            candidate = method()
    numpy_method = getattr(candidate, "numpy", None)
    if callable(numpy_method):
        candidate = numpy_method()
    return np.asarray(candidate)


def _flatten_numeric(value: object, *, prefix: str = "") -> dict[str, np.ndarray]:
    if isinstance(value, Mapping):
        result: dict[str, np.ndarray] = {}
        for key in sorted(value, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_numeric(value[key], prefix=child))
        return result
    array = _numeric(value)
    if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
        raise TypeError(f"trace field {prefix!r} is not numeric")
    return {prefix: array.copy()}


def _trace_frame(environment: Any, observation: object) -> dict[str, object]:
    base = environment.unwrapped
    context = base.get_push_expert_task_context()
    evaluation = base.get_push_expert_evaluation()
    target_force = _numeric(
        base._robot_contact_magnitude(context.target_object.actor, arm_only=False)
    )
    return {
        "observation": _flatten_numeric(observation),
        "joint_position": _numeric(base.agent.robot.get_qpos()).copy(),
        "joint_velocity": _numeric(base.agent.robot.get_qvel()).copy(),
        "object_pose": {
            handle.object_id: _numeric(handle.actor.pose.raw_pose).copy()
            for handle in context.objects
        },
        "object_linear_velocity": {
            handle.object_id: _numeric(handle.actor.linear_velocity).copy()
            for handle in context.objects
        },
        "object_angular_velocity": {
            handle.object_id: _numeric(handle.actor.angular_velocity).copy()
            for handle in context.objects
        },
        "evaluation": {
            key: _numeric(value).reshape(-1)[0].item() for key, value in evaluation.items()
        },
        "target_contact": bool(target_force.reshape(-1)[0] > CONTACT_FORCE_THRESHOLD),
    }


def _run_actions(environment: Any, actions: Sequence[np.ndarray]) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for action in actions:
        observation, _reward, _terminated, _truncated, _info = environment.step(action)
        values.append(_trace_frame(environment, observation))
    return values


def _target_position(environment: Any) -> np.ndarray:
    context = environment.unwrapped.get_push_expert_task_context()
    return _numeric(context.target_object.actor.pose.p)[0].astype(np.float64, copy=True)


def _contact_bearing_segment(
    *,
    seed: int,
    task: PushTaskSpec,
) -> tuple[
    PushSimulationSnapshot,
    tuple[np.ndarray, ...],
    list[dict[str, object]],
    dict[str, object],
]:
    source = _environment()
    replay = _environment()
    live = _environment()
    try:
        _reset(source, seed, task)
        expert = PushToRegionExpert(source)
        result = expert.run()
        actions = expert.action_trace
        if len(actions) < ACTION_SEQUENCE_LENGTH:
            raise RuntimeError("reference expert produced too few actions for clone audit")
        _reset(replay, seed, task)
        initial = _target_position(replay)
        first_motion_index: int | None = None
        for index, action in enumerate(actions):
            replay.step(action)
            displacement = float(np.linalg.norm(_target_position(replay)[:2] - initial[:2]))
            if first_motion_index is None and displacement >= 5e-4:
                first_motion_index = index
            if first_motion_index is not None and index >= first_motion_index + 4:
                break
        if first_motion_index is None:
            raise RuntimeError("reference action trace never established physical target motion")
        start = max(0, first_motion_index - 2)
        stop = min(len(actions), start + ACTION_SEQUENCE_LENGTH)
        selected = tuple(
            np.asarray(action, dtype=np.float32).copy() for action in actions[start:stop]
        )
        if len(selected) < 6:
            raise RuntimeError("contact-bearing clone sequence is too short")
        _reset(live, seed, task)
        for action in actions[:start]:
            live.step(action)
        live_snapshot = capture_push_simulation_state(live)
        live_trace = _run_actions(live, selected)
        return (
            live_snapshot,
            selected,
            live_trace,
            {
                "reference_status": result.status.value,
                "reference_success": result.success,
                "reference_total_actions": len(actions),
                "first_target_motion_action_index": first_motion_index,
                "selected_action_start": start,
                "selected_action_count": len(selected),
            },
        )
    finally:
        live.close()
        replay.close()
        source.close()


def _cold_restore(
    environment: Any,
    *,
    seed: int,
    task: PushTaskSpec,
    snapshot: PushSimulationSnapshot,
) -> None:
    """Clear opaque PhysX contact caches, then restore all public state."""

    _reset(environment, seed, task)
    restore_push_simulation_state(environment, snapshot)


def _comparison(
    first: Sequence[dict[str, object]],
    second: Sequence[dict[str, object]],
) -> dict[str, object]:
    if len(first) != len(second):
        return {"passed": False, "reason": "trace_length_changed"}
    maxima = {
        "observation": 0.0,
        "joint_position": 0.0,
        "joint_velocity": 0.0,
        "object_pose": 0.0,
        "object_linear_velocity": 0.0,
        "object_angular_velocity": 0.0,
    }
    categorical_total = 0
    categorical_matches = 0
    for left, right in zip(first, second, strict=True):
        for group in maxima:
            left_group = left[group]
            right_group = right[group]
            if isinstance(left_group, Mapping):
                assert isinstance(right_group, Mapping)
                keys = set(left_group) | set(right_group)
                for key in keys:
                    difference = np.max(
                        np.abs(_numeric(left_group[key]) - _numeric(right_group[key]))
                    )
                    maxima[group] = max(maxima[group], float(difference))
            else:
                difference = np.max(np.abs(_numeric(left_group) - _numeric(right_group)))
                maxima[group] = max(maxima[group], float(difference))
        left_evaluation = left["evaluation"]
        right_evaluation = right["evaluation"]
        assert isinstance(left_evaluation, Mapping) and isinstance(right_evaluation, Mapping)
        categorical_keys = {
            key for key, value in left_evaluation.items() if isinstance(value, bool)
        }
        for key in categorical_keys:
            categorical_total += 1
            categorical_matches += left_evaluation[key] == right_evaluation.get(key)
        categorical_total += 1
        categorical_matches += left["target_contact"] == right["target_contact"]
    agreement = categorical_matches / categorical_total if categorical_total else 1.0
    tolerances = {
        "observation": 2e-4,
        "joint_position": 2e-5,
        "joint_velocity": 2e-4,
        "object_pose": 2e-5,
        "object_linear_velocity": 2e-4,
        "object_angular_velocity": 2e-4,
    }
    return {
        "continuous_maximum_absolute_errors": maxima,
        "continuous_tolerances": tolerances,
        "categorical_total": categorical_total,
        "categorical_matches": categorical_matches,
        "categorical_agreement": agreement,
        "passed": agreement == 1.0 and all(maxima[key] <= tolerances[key] for key in maxima),
    }


def _trace_digest(trace: Sequence[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for frame in trace:
        for group in (
            "observation",
            "joint_position",
            "joint_velocity",
            "object_pose",
            "object_linear_velocity",
            "object_angular_velocity",
        ):
            values = frame[group]
            if isinstance(values, Mapping):
                for key in sorted(values):
                    digest.update(str(key).encode())
                    digest.update(_numeric(values[key]).tobytes())
            else:
                digest.update(_numeric(values).tobytes())
        digest.update(json.dumps(frame["evaluation"], sort_keys=True).encode())
        digest.update(str(frame["target_contact"]).encode())
    return digest.hexdigest()


def _action_digest(actions: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for action in actions:
        digest.update(np.asarray(action, dtype=np.float32).tobytes())
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    task = PushTaskSpec("blue_cube", "left", "standard")
    snapshot, actions, live_trace, reference = _contact_bearing_segment(seed=args.seed, task=task)
    main_environment = _environment()
    sandbox_a = _environment()
    sandbox_b = _environment()
    try:
        for environment in (main_environment, sandbox_a, sandbox_b):
            _cold_restore(environment, seed=args.seed, task=task, snapshot=snapshot)
        configurations = [
            push_simulation_configuration(environment)
            for environment in (main_environment, sandbox_a, sandbox_b)
        ]
        configuration_equivalent = configurations[0] == configurations[1] == configurations[2]

        main_before = capture_push_simulation_state(main_environment)
        sandbox_b_before = capture_push_simulation_state(sandbox_b)
        first = _run_actions(sandbox_a, actions)
        main_after_sandbox = capture_push_simulation_state(main_environment)
        sandbox_b_after = capture_push_simulation_state(sandbox_b)
        isolation = {
            "main_state_unchanged": main_before.state_sha256 == main_after_sandbox.state_sha256,
            "other_sandbox_state_unchanged": sandbox_b_before.state_sha256
            == sandbox_b_after.state_sha256,
            "main_elapsed_steps_unchanged": int(
                _numeric(main_before.task_tensors["_elapsed_steps"]).reshape(-1)[0]
            )
            == int(_numeric(main_after_sandbox.task_tensors["_elapsed_steps"]).reshape(-1)[0]),
        }

        _cold_restore(sandbox_a, seed=args.seed, task=task, snapshot=snapshot)
        second = _run_actions(sandbox_a, actions)
        _cold_restore(main_environment, seed=args.seed, task=task, snapshot=snapshot)
        main_trace = _run_actions(main_environment, actions)
        restore_comparison = _comparison(first, second)
        cross_environment_comparison = _comparison(first, main_trace)
        live_restore_comparison = _comparison(live_trace, first)
        low, high = main_environment.unwrapped.get_push_expert_action_bounds()
        action_legal = all(
            action.shape == (8,)
            and np.isfinite(action).all()
            and np.all(action >= low)
            and np.all(action <= high)
            for action in actions
        )
        passed = bool(
            configuration_equivalent
            and all(isolation.values())
            and action_legal
            and restore_comparison["passed"] is True
            and cross_environment_comparison["passed"] is True
            and live_restore_comparison["passed"] is True
        )
        report = {
            "schema_version": "langmani-v2-phase2b3-state-clone-audit-v0",
            "environment_id": ENV_ID,
            "scene_seed": args.seed,
            "task_spec": task.to_dict(),
            "snapshot_schema": snapshot.schema_version,
            "snapshot_sha256": snapshot.state_sha256,
            "captured_components": {
                "actor_pose_velocity": True,
                "articulation_state": True,
                "robot_joint_position_velocity": True,
                "gripper_state": True,
                "controller_state": True,
                "elapsed_episode_steps": True,
                "task_progress_and_latched_events": True,
                "main_and_episode_rng_state": True,
                "contact_solver_cache": False,
                "contact_solver_cache_note": (
                    "ManiSkill 3.0.1 exposes no public contact-cache snapshot; deterministic "
                    "cold reset plus complete public-state restoration is the accepted sandbox "
                    "semantics and contact-bearing live continuation is the acceptance test"
                ),
            },
            "reference": reference,
            "fixed_action_sequence_sha256": _action_digest(actions),
            "fixed_action_count": len(actions),
            "action_contract_validated": action_legal,
            "configuration": configurations[0],
            "sandbox_configuration_equivalent": configuration_equivalent,
            "isolation": isolation,
            "restore_replay": restore_comparison,
            "sandbox_main_replay": cross_environment_comparison,
            "live_continuation_replay": live_restore_comparison,
            "restore_mode": "fresh_isolated_environment_reset_then_complete_state_restore",
            "warm_in_place_restore_authorized": False,
            "trace_sha256": {
                "live_continuation": _trace_digest(live_trace),
                "sandbox_first": _trace_digest(first),
                "sandbox_restored": _trace_digest(second),
                "main_execution": _trace_digest(main_trace),
            },
            "categorical_agreement_required": 1.0,
            "optimizer_steps": 0,
            "collection_started": False,
            "passed": passed,
        }
        output = args.output.resolve()
        output.relative_to(OUTPUT_ROOT.resolve())
        write_json_once(output, report)
        print(json.dumps({"output": str(output), "passed": passed}, sort_keys=True))
        return 0 if passed else 2
    finally:
        sandbox_b.close()
        sandbox_a.close()
        main_environment.close()


if __name__ == "__main__":
    raise SystemExit(main())
