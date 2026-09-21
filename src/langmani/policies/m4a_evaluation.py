"""Paired ACT/expert evaluation. Privileged reset auditing never feeds the policy."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import traceback
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from langmani.policies.m4a_data import TASK, digest, numpy, write_json
from langmani.policies.m4a_training import load_checkpoint, policy_observation


def aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("cannot aggregate zero episodes")
    for ep in episodes:
        if type(ep["success"]) is not bool or type(ep["length"]) is not int or ep["length"] < 0:
            raise ValueError("episode result has invalid success/length")
    lengths = np.array([ep["length"] for ep in episodes], dtype=np.float64)
    successes = sum(ep["success"] for ep in episodes)
    return {
        "num_episodes": len(episodes),
        "successes": successes,
        "failures": len(episodes) - successes,
        "success_rate": successes / len(episodes),
        "mean_episode_length": float(lengths.mean()),
        "std_episode_length": float(lengths.std(ddof=0)),
        "termination_reason_counts": dict(Counter(ep["termination_reason"] for ep in episodes)),
        "target_off_table_count": sum(ep["target_off_table"] for ep in episodes),
    }


def evaluation_seeds(split: Mapping[str, Any], count: int, seed: int) -> list[int]:
    """Fresh reset seeds drawn once; exclude EVERY source scene, not only the chosen task."""
    if count < 1 or seed < 0:
        raise ValueError("evaluation count must be positive and seed nonnegative")
    excluded = set(split["all_source_scene_seeds"])
    rng = np.random.default_rng(seed)
    result: list[int] = []
    while len(result) < count:
        value = int(rng.integers(0, 2**31 - 1))
        if value not in excluded and value not in result:
            result.append(value)
    return result


def state_digest(state: Any) -> str:
    """Hash public simulator state for reset pairing; never return it to action selection."""
    hasher = hashlib.sha256()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value):
                hasher.update(str(key).encode())
                visit(value[key])
        else:
            array = numpy(value)
            if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
                raise ValueError("invalid physical reset audit state")
            hasher.update(str((array.dtype.str, array.shape)).encode())
            hasher.update(array.tobytes())

    visit(state)
    return hasher.hexdigest()


def bool_value(value: Any) -> bool:
    array = numpy(value)
    if array.size != 1 or array.dtype.kind != "b":
        raise ValueError("environment must return one boolean per termination/evaluation flag")
    return bool(array.item())


def run_act(
    env: Any, observation: Any, policy: Any, pre: Any, post: Any, max_steps: int = 200
) -> dict[str, Any]:
    from langmani.datasets.observation_reconstruction import extract_base_camera_rgb
    from langmani.datasets.policy_state import extract_panda_policy_state_v0

    policy.reset()
    pre.reset()
    post.reset()
    result = {
        "success": False,
        "length": 0,
        "termination_reason": "timeout",
        "target_off_table": False,
    }
    space = env.unwrapped.single_action_space
    if tuple(space.shape) != (8,):
        raise ValueError("environment action dimension changed")
    with torch.inference_mode():
        for step in range(max_steps):
            rgb = extract_base_camera_rgb(observation)
            qpos = extract_panda_policy_state_v0(env.unwrapped.agent.robot)
            batch = policy_observation(np.transpose(rgb, (2, 0, 1)).copy(), qpos)
            # select_action owns the queue: consume ten actions, then query using the new observation.
            action = numpy(post(policy.select_action(pre(batch))))
            if (
                action.shape != (1, 8)
                or action.dtype != np.float32
                or not np.isfinite(action).all()
            ):
                result["termination_reason"] = "invalid_action"
                break
            if (action[0] < space.low).any() or (action[0] > space.high).any():
                result["termination_reason"] = "action_out_of_bounds"
                break
            observation, _, terminated, truncated, info = env.step(action[0])
            result["length"] = step + 1
            result["success"] = bool_value(info["success"])
            result["target_off_table"] = bool_value(info["target_off_table"])
            if result["success"]:
                result["termination_reason"] = "success"
                break
            if bool_value(terminated) or bool_value(truncated):
                result["termination_reason"] = (
                    "target_off_table"
                    if result["target_off_table"]
                    else "timeout"
                    if bool_value(truncated)
                    else "terminated"
                )
                break
    return result


def make_environment(sim_backend: str) -> Any:
    if platform.system() != "Linux":
        raise RuntimeError("physical M4A evaluation requires native Linux ManiSkill + Vulkan")
    import gymnasium as gym

    import langmani.environments  # noqa: F401
    from langmani.environments.pick_place_by_instruction import ENV_ID

    return gym.make(
        ENV_ID,
        num_envs=1,
        obs_mode="rgb",
        reward_mode="none",
        control_mode="pd_joint_pos",
        render_mode=None,
        sim_backend=sim_backend,
        render_backend="sapien_cuda",
        max_episode_steps=200,
    )


def evaluate(
    *,
    checkpoint: Path,
    output: Path,
    split: Mapping[str, Any],
    count: int,
    seed: int,
    device: str,
    sim_backend: str,
    provenance: dict[str, Any],
    environment_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    from langmani.experts import ExpertConfig, PickPlaceExpert

    if environment_factory is None and platform.system() != "Linux":
        raise RuntimeError("closed-loop evaluation is pending a native Linux ManiSkill runtime")
    if output.exists():
        raise FileExistsError("evaluation output exists; choose a new run directory")
    seeds = evaluation_seeds(split, count, seed)
    config = {
        "schema_version": "langmani-m4a-evaluation-v1",
        "seed": seed,
        "seeds": seeds,
        "distribution": "M1 reset(seed), source scenes excluded, paired exact state hashes",
        "task": TASK.to_dict(),
        "device": device,
        "sim_backend": sim_backend,
        "max_episode_steps": 200,
        "action_bounds": "reject before env.step; no clipping",
        "checkpoint": str(checkpoint.resolve()),
        "provenance": provenance,
        "schedule_sha256": digest(seeds),
        "fixture": environment_factory is not None,
    }
    policy, pre, post = load_checkpoint(checkpoint, device)
    write_json(output / "config.json", config)
    write_json(output / "split.json", dict(split))
    rows: list[dict[str, Any]] = []
    factory = environment_factory or (lambda: make_environment(sim_backend))
    for episode_seed in seeds:
        for controller in ("expert", "act"):
            env = None
            row: dict[str, Any] = {
                "controller": controller,
                "seed": episode_seed,
                "success": False,
                "length": 0,
                "termination_reason": "infrastructure_error",
                "target_off_table": False,
                "initial_state_sha256": None,
                "error": None,
                "expert_status": None,
            }
            try:
                env = factory()
                observation, _ = env.reset(seed=episode_seed, options={"task_spec": TASK.to_dict()})
                # Audit only. get_state_dict never reaches run_act or its processors.
                row["initial_state_sha256"] = state_digest(env.unwrapped.get_state_dict())
                if controller == "expert":
                    expert = PickPlaceExpert(env, config=ExpertConfig(max_episode_steps=200)).run()
                    if expert.exception_type or expert.status.value in {
                        "initialization_failure",
                        "invalid_task",
                        "unexpected_exception",
                    }:
                        raise RuntimeError(
                            f"expert infrastructure: {expert.exception_type}: {expert.exception_message}"
                        )
                    task_success = expert.final_environment_evaluation.get("success", False)
                    row.update(
                        success=bool(task_success),
                        length=expert.total_environment_steps,
                        termination_reason="success" if task_success else expert.status.value,
                        expert_status=expert.status.value,
                        target_off_table=bool(
                            expert.final_environment_evaluation.get("target_off_table", False)
                        ),
                    )
                else:
                    row.update(run_act(env, observation, policy, pre, post))
            except Exception as error:
                traceback.print_exc()
                row["error"] = f"{type(error).__name__}: {error}"
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception as error:
                        traceback.print_exc()
                        row.update(
                            error=f"close: {error}",
                            termination_reason="infrastructure_error",
                            success=False,
                        )
            rows.append(row)
            with (output / "episodes.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(json.dumps(row), flush=True)
    summaries = {
        name: aggregate([row for row in rows if row["controller"] == name])
        for name in ("expert", "act")
    }
    paired = all(
        rows[i]["initial_state_sha256"] is not None
        and rows[i]["initial_state_sha256"] == rows[i + 1]["initial_state_sha256"]
        for i in range(0, len(rows), 2)
    )
    infrastructure_errors = sum(row["error"] is not None for row in rows)
    valid = paired and infrastructure_errors == 0
    act_steps = sum(row["length"] for row in rows if row["controller"] == "act")
    gap = summaries["expert"]["success_rate"] - summaries["act"]["success_rate"]
    metrics = {
        "schema_version": "langmani-m4a-metrics-v1",
        **summaries,
        "expert_minus_act_success_rate": gap if valid else None,
        "absolute_success_rate_gap": abs(gap) if valid else None,
        "initial_states_paired": paired,
        "infrastructure_errors": infrastructure_errors,
        "act_environment_steps": act_steps,
        "closed_loop_evaluated": valid and environment_factory is None and act_steps > 0,
        "passed": valid,
        "benchmark_kind": "fresh reset diagnostic; no checkpoint selection",
    }
    write_json(output / "metrics.json", metrics)
    return metrics
