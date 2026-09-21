"""Paired ACT/expert evaluation. Privileged reset auditing never feeds the policy."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
import traceback
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from langmani.policies.m4a_data import TASK, digest, numpy, write_json
from langmani.policies.m4a_training import load_checkpoint, policy_observation


def expert_result(env: Any) -> dict[str, Any]:
    from langmani.experts import ExpertConfig, PickPlaceExpert

    expert = PickPlaceExpert(env, config=ExpertConfig(max_episode_steps=200)).run()
    if expert.exception_type or expert.status.value in {
        "initialization_failure",
        "invalid_task",
        "unexpected_exception",
    }:
        raise RuntimeError(
            f"expert infrastructure: {expert.exception_type}: {expert.exception_message}"
        )
    success = bool(expert.final_environment_evaluation.get("success", False))
    return {
        "success": success,
        "length": expert.total_environment_steps,
        "termination_reason": "success" if success else expert.status.value,
        "expert_status": expert.status.value,
        "target_off_table": bool(
            expert.final_environment_evaluation.get("target_off_table", False)
        ),
    }


def isolated_expert(seed: int, sim_backend: str, output: Path) -> dict[str, Any]:
    """Run the paired expert in its pinned NumPy-1 interpreter, retaining worker evidence."""
    from langmani.experts.runtime import resolve_planner_python

    worker_output = output / "expert_workers" / f"seed-{seed}.json"
    worker_output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        resolve_planner_python(),
        "-m",
        __name__,
        "--seed",
        str(seed),
        "--sim-backend",
        sim_backend,
        "--output",
        str(worker_output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
    worker_output.with_suffix(".log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"expert worker exited {completed.returncode}; see {worker_output.with_suffix('.log')}"
        )
    result = json.loads(worker_output.read_text(encoding="utf-8"))
    if result.pop("seed") != seed or result.pop("sim_backend") != sim_backend:
        raise ValueError("expert worker reset identity differs")
    return result


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
        "gripper_saturation_count": sum(ep.get("gripper_saturation_count", 0) for ep in episodes),
        "gripper_max_overshoot": max(ep.get("gripper_max_overshoot", 0.0) for ep in episodes),
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
        "gripper_saturation_count": 0,
        "gripper_max_overshoot": 0.0,
    }
    space = env.unwrapped.single_action_space
    if tuple(space.shape) != (8,):
        raise ValueError("environment action dimension changed")
    if space.low[7] != -1 or space.high[7] != 1:
        raise ValueError("expected the Panda normalized gripper action in [-1, 1]")
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
            action = action[0].copy()
            if (action[:7] < space.low[:7]).any() or (action[:7] > space.high[:7]).any():
                result["termination_reason"] = "action_out_of_bounds"
                break
            # ManiSkill's normalized mimic controller clips before scaling to finger targets.
            # Preserve that native behavior; absolute arm joint commands remain unmodified.
            raw_gripper = float(action[7])
            action[7] = np.clip(action[7], -1.0, 1.0)
            overshoot = abs(raw_gripper - float(action[7]))
            if overshoot:
                result["gripper_saturation_count"] += 1
                result["gripper_max_overshoot"] = max(result["gripper_max_overshoot"], overshoot)
            observation, _, terminated, truncated, info = env.step(action)
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
    if environment_factory is None and platform.system() != "Linux":
        raise RuntimeError("closed-loop evaluation is pending a native Linux ManiSkill runtime")
    if output.exists():
        raise FileExistsError("evaluation output exists; choose a new run directory")
    seeds = evaluation_seeds(split, count, seed)
    config = {
        "schema_version": "langmani-m4a-evaluation-v2",
        "seed": seed,
        "seeds": seeds,
        "distribution": "M1 reset(seed), source scenes excluded, paired exact state hashes",
        "task": TASK.to_dict(),
        "device": device,
        "sim_backend": sim_backend,
        "max_episode_steps": 200,
        "action_bounds": (
            "reject invalid/nonfinite or out-of-bounds absolute arm joints; "
            "saturate normalized gripper to [-1, 1] exactly as the native controller; audit every saturation"
        ),
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
                "gripper_saturation_count": 0,
                "gripper_max_overshoot": 0.0,
            }
            try:
                if controller == "expert" and environment_factory is None:
                    row.update(isolated_expert(episode_seed, sim_backend, output))
                else:
                    env = factory()
                    observation, _ = env.reset(
                        seed=episode_seed, options={"task_spec": TASK.to_dict()}
                    )
                    # Audit only. get_state_dict never reaches run_act or its processors.
                    row["initial_state_sha256"] = state_digest(env.unwrapped.get_state_dict())
                    row.update(
                        expert_result(env)
                        if controller == "expert"
                        else run_act(env, observation, policy, pre, post)
                    )
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
    act_steps = sum(row["length"] for row in rows if row["controller"] == "act")
    valid = paired and infrastructure_errors == 0 and act_steps > 0
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


def expert_worker_main() -> None:
    """Internal subprocess entry point; no ACT checkpoint or policy observations cross it."""
    import argparse

    from langmani.experts.runtime import (
        planner_runtime_matches_expected,
        query_planner_runtime_versions,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--sim-backend", required=True, choices=("physx_cpu", "physx_cuda"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    versions = query_planner_runtime_versions(sys.executable)
    if not planner_runtime_matches_expected(versions):
        raise RuntimeError(f"expert runtime pins differ: {versions}")
    env = make_environment(args.sim_backend)
    try:
        env.reset(seed=args.seed, options={"task_spec": TASK.to_dict()})
        initial_state = state_digest(env.unwrapped.get_state_dict())
        result = expert_result(env)
    finally:
        env.close()
    write_json(
        args.output,
        {
            "seed": args.seed,
            "sim_backend": args.sim_backend,
            "initial_state_sha256": initial_state,
            **result,
        },
    )
    write_json(args.output.with_suffix(".runtime.json"), versions)


if __name__ == "__main__":
    expert_worker_main()
