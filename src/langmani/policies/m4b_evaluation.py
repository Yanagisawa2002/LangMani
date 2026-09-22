"""Paired language intervention on native M1 scenes, with separate privileged scoring."""

from __future__ import annotations

import csv
import json
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from langmani.policies.m4a_data import digest, file_digest, numpy, read_json, write_json
from langmani.policies.m4a_evaluation import (
    bool_value,
    expert_result,
    make_environment,
    state_digest,
)
from langmani.policies.m4b_protocol import GOALS, metrics, score_rollouts, task
from langmani.policies.m4b_training import load_checkpoint, policy_observation


def reset_audit(env: Any, observation: Any) -> dict[str, str]:
    from langmani.datasets.observation_reconstruction import extract_base_camera_rgb
    from langmani.datasets.policy_state import extract_panda_policy_state_v0

    return {
        "initial_state_sha256": state_digest(env.unwrapped.get_state_dict()),
        "initial_rgb_sha256": state_digest(extract_base_camera_rgb(observation)),
        "initial_qpos_sha256": state_digest(
            extract_panda_policy_state_v0(env.unwrapped.agent.robot)
        ),
    }


def score_state(env: Any) -> dict[str, dict[str, bool]]:
    """The unchanged M1 success function, queried for each goal outside policy input."""
    from langmani.environments import pick_place_by_instruction as m1
    from langmani.environments.specs import OBJECT_IDS
    from langmani.environments.task_logic import evaluate_task_state

    base = env.unwrapped
    positions = torch.stack([cube.pose.p for cube in base.cubes], dim=1)
    grasped = torch.stack([base.agent.is_grasping(cube) for cube in base.cubes], dim=1)
    static = torch.stack(
        [cube.is_static(lin_thresh=1e-2, ang_thresh=0.1) for cube in base.cubes], dim=1
    )
    result = {}
    for index, goal in enumerate(GOALS):
        flags = evaluate_task_state(
            cube_positions=positions,
            bin_floor_centers=base._bin_floor_centers(),
            cube_is_grasped=grasped,
            cube_is_static=static,
            target_object_index=torch.tensor(
                [OBJECT_IDS.index("red_cube")], device=positions.device
            ),
            target_bin_index=torch.tensor([index], device=positions.device),
            cube_bounding_radius=m1.CUBE_BOUNDING_RADIUS,
            cube_resting_height=m1.CUBE_HALF_SIZE,
            bin_interior_half_size=m1.BIN_INTERIOR_HALF_SIZE,
            containment_clearance=m1.CONTAINMENT_CLEARANCE,
            resting_height_tolerance=m1.RESTING_HEIGHT_TOLERANCE,
            off_table_height=m1.OFF_TABLE_HEIGHT,
        )
        result[goal] = {key: bool_value(value) for key, value in flags.items()}
    return result


def noise_schedule(seed: int) -> tuple[list[torch.Tensor], str]:
    """Generate all 20 chunks independently of rollout length and language."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    chunks = [torch.randn((1, 50, 32), generator=generator, dtype=torch.float32) for _ in range(20)]
    return chunks, state_digest(torch.stack(chunks))


def write_video(path: Path, frames: list[np.ndarray]) -> None:
    import av

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=20)
        stream.width, stream.height, stream.pix_fmt = 256, 256, "yuv420p"
        stream.options = {"crf": "18"}
        for array in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(path)) as container:
        if sum(1 for _ in container.decode(video=0)) != len(frames):
            raise RuntimeError("saved evidence video is incomplete")


def run_policy(
    env: Any,
    observation: Any,
    policy: Any,
    pre: Any,
    post: Any,
    instruction: str,
    seed: int,
    artifact: Path,
) -> dict[str, Any]:
    from langmani.datasets.observation_reconstruction import extract_base_camera_rgb
    from langmani.datasets.policy_state import extract_panda_policy_state_v0

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    policy.reset()
    pre.reset()
    post.reset()
    chunks, noise_hash = noise_schedule(seed)
    result: dict[str, Any] = {
        "achieved_goal": None,
        "length": 0,
        "termination_reason": "timeout",
        "gripper_saturation_count": 0,
        "gripper_max_overshoot": 0.0,
        "noise_schedule_sha256": noise_hash,
    }
    space = env.unwrapped.single_action_space
    if tuple(space.shape) != (8,) or space.low[7] != -1 or space.high[7] != 1:
        raise ValueError("native Panda absolute-joint/gripper contract changed")
    frames = [extract_base_camera_rgb(observation).copy()]
    actions: list[np.ndarray] = []
    raw_actions: list[np.ndarray] = []
    flags = score_state(env)
    with torch.inference_mode():
        for step in range(200):
            rgb = extract_base_camera_rgb(observation)
            qpos = extract_panda_policy_state_v0(env.unwrapped.agent.robot)
            batch = pre(policy_observation(rgb.transpose(2, 0, 1).copy(), qpos, instruction))
            action = numpy(
                post(policy.select_action(batch, noise=chunks[step // 10].to(policy.config.device)))
            )
            if (
                action.shape != (1, 8)
                or action.dtype != np.float32
                or not np.isfinite(action).all()
            ):
                result["termination_reason"] = "invalid_action"
                break
            action = action[0].copy()
            raw_actions.append(action.copy())
            if (action[:7] < space.low[:7]).any() or (action[:7] > space.high[:7]).any():
                result["termination_reason"] = "action_out_of_bounds"
                break
            raw_gripper = float(action[7])
            action[7] = np.clip(action[7], -1, 1)
            overshoot = abs(raw_gripper - float(action[7]))
            if overshoot:
                result["gripper_saturation_count"] += 1
                result["gripper_max_overshoot"] = max(result["gripper_max_overshoot"], overshoot)
            observation, _, terminated, truncated, _ = env.step(action)
            actions.append(action.copy())
            frames.append(extract_base_camera_rgb(observation).copy())
            result["length"] = step + 1
            flags = score_state(env)
            achieved = [goal for goal in GOALS if flags[goal]["success"]]
            if len(achieved) > 1:
                raise RuntimeError("incompatible destinations simultaneously successful")
            if achieved:
                result.update(achieved_goal=achieved[0], termination_reason="goal_reached")
                break
            if any(f["target_off_table"] for f in flags.values()):
                result["termination_reason"] = "target_off_table"
                break
            if bool_value(terminated) or bool_value(truncated):
                result["termination_reason"] = "timeout" if bool_value(truncated) else "terminated"
                break
    action_array = np.asarray(actions, dtype=np.float32).reshape(-1, 8)
    np.savez_compressed(
        artifact.with_suffix(".npz"),
        actions=action_array,
        raw_actions=np.asarray(raw_actions, dtype=np.float32).reshape(-1, 8),
    )
    write_video(artifact.with_suffix(".mp4"), frames)
    result.update(
        final_flags=flags,
        actions_sha256=state_digest(action_array),
        action_file_sha256=file_digest(artifact.with_suffix(".npz")),
        video_sha256=file_digest(artifact.with_suffix(".mp4")),
        video_frames=len(frames),
    )
    return result


def isolated_expert(seed: int, goal: str, sim_backend: str, output: Path) -> dict[str, Any]:
    from langmani.experts.runtime import resolve_planner_python

    path = output / "expert_workers" / f"seed-{seed}-{task(goal).target_bin_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            resolve_planner_python(),
            "-m",
            __name__,
            "--seed",
            str(seed),
            "--goal",
            goal,
            "--sim-backend",
            sim_backend,
            "--output",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    path.with_suffix(".log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"expert worker exited {completed.returncode}: {path.with_suffix('.log')}"
        )
    result = read_json(path)
    if (
        result["seed"] != seed
        or result["semantic_goal"] != goal
        or result["sim_backend"] != sim_backend
    ):
        raise ValueError("expert worker identity mismatch")
    return result


def native_gate(schedule: dict[str, Any], output: Path, sim_backend: str) -> dict[str, Any]:
    """Verify exact reset inputs and both expert goals before fitting the learned policy."""
    if output.exists():
        raise FileExistsError("native gate output already exists")
    write_json(output / "schedule.json", schedule)
    rows = []
    for scene in schedule["scenes"]:
        audits = []
        for goal in GOALS:
            row = isolated_expert(scene["seed"], goal, sim_backend, output)
            env = make_environment(sim_backend)
            try:
                observation, _ = env.reset(
                    seed=scene["seed"], options={"task_spec": task(goal).to_dict()}
                )
                audit = reset_audit(env, observation)
                if any(row[k] != audit[k] for k in audit):
                    raise RuntimeError(
                        "planner and policy runtimes produce different initial inputs"
                    )
                audits.append(audit)
            finally:
                env.close()
            rows.append(row)
            write_json(output / "episodes.json", {"episodes": rows})
            print(json.dumps(row), flush=True)
        if audits[0] != audits[1]:
            raise RuntimeError("changing only semantic goal changes native reset observation/state")
    report = {
        "passed": all(0 < r["length"] <= 200 for r in rows),
        "schedule_sha256": schedule["schedule_sha256"],
        "sim_backend": sim_backend,
        "initial_inputs_exactly_paired": True,
        "expert_episodes": len(rows),
        "expert_successes": sum(r["success"] for r in rows),
        "episodes_sha256": digest(rows),
    }
    write_json(output / "gate.json", report)
    return report


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("empty CSV")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {
                k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                for k, v in row.items()
            }
            for row in rows
        )


def evaluate(
    *,
    checkpoint: Path,
    output: Path,
    schedule: dict[str, Any],
    gate_path: Path,
    device: str,
    sim_backend: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    gate = read_json(gate_path / "gate.json")
    experts = read_json(gate_path / "episodes.json")["episodes"]
    if (
        gate["passed"] is not True
        or gate["schedule_sha256"] != schedule["schedule_sha256"]
        or gate["sim_backend"] != sim_backend
        or digest(experts) != gate["episodes_sha256"]
        or read_json(gate_path / "schedule.json") != schedule
    ):
        raise ValueError("native paired expert/observation gate is missing or mismatched")
    if output.exists():
        raise FileExistsError("evaluation output exists")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    policy, pre, post = load_checkpoint(checkpoint, device)
    config = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_digest(checkpoint / "model.safetensors"),
        "schedule": schedule,
        "native_gate": gate,
        "device": device,
        "sim_backend": sim_backend,
        "max_episode_steps": 200,
        "provenance": provenance,
        "termination": "first either-goal original M1 success; native failure; 200 steps",
        "common_noise": "CPU torch.randn seed=scene seed; 20 chunks [1,50,32]; shared across all prompts",
    }
    write_json(output / "config.json", config)
    write_json(output / "expert_episodes.json", {"episodes": experts})
    directory = output / "rollouts"
    directory.mkdir()
    rows = []
    started = time.perf_counter()
    for scene in schedule["scenes"]:
        seed = scene["seed"]
        reference = next(r for r in experts if r["seed"] == seed)
        for prompt in scene["prompts"]:
            rollout_id = f"seed-{seed}-{prompt['prompt_id']}"
            artifact = directory / rollout_id
            row: dict[str, Any] = {
                "rollout_id": rollout_id,
                "seed": seed,
                **prompt,
                "video": f"rollouts/{rollout_id}.mp4",
                "error": None,
            }
            env = None
            try:
                env = make_environment(sim_backend)
                # The scoring anchor never enters policy input; metadata is fixed for all prompts.
                observation, _ = env.reset(
                    seed=seed, options={"task_spec": task(GOALS[0]).to_dict()}
                )
                audit = reset_audit(env, observation)
                row.update(audit)
                if any(audit[k] != reference[k] for k in audit):
                    raise RuntimeError("evaluation reset differs from frozen paired gate")
                row.update(
                    run_policy(
                        env, observation, policy, pre, post, prompt["instruction"], seed, artifact
                    )
                )
            except Exception as error:
                traceback.print_exc()
                row.update(
                    error=f"{type(error).__name__}: {error}",
                    termination_reason="infrastructure_error",
                )
                write_json(artifact.with_suffix(".json"), row)
                write_json(
                    output / "failure.json",
                    {"passed": False, "closed_loop_evaluated": False, "rollout": row},
                )
                raise
            finally:
                if env is not None:
                    env.close()
            write_json(artifact.with_suffix(".json"), row)
            rows.append(row)
            write_csv(output / "rollouts.csv", rows)
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in ("rollout_id", "achieved_goal", "length", "termination_reason")
                    }
                ),
                flush=True,
            )
    scored = score_rollouts(rows)
    write_csv(output / "episodes.csv", scored)
    result = metrics(rows)
    result.update(
        evaluation_seconds=time.perf_counter() - started,
        expert_reference={
            "episodes": len(experts),
            "successes": sum(r["success"] for r in experts),
        },
        checkpoint_sha256=config["checkpoint_sha256"],
        schedule_sha256=schedule["schedule_sha256"],
    )
    representatives = {}
    for row in scored:
        category = f"{row['condition']}/{row['semantic_goal']}/{'success' if row['success'] else row['failure_category']}"
        representatives.setdefault(
            category,
            {
                k: row[k]
                for k in (
                    "rollout_id",
                    "instruction",
                    "semantic_goal",
                    "achieved_goal",
                    "video",
                    "length",
                )
            },
        )
    write_json(output / "representative_videos.json", representatives)
    write_json(output / "metrics.json", result)
    return result


def expert_worker_main() -> None:
    import argparse

    from langmani.experts.runtime import (
        planner_runtime_matches_expected,
        query_planner_runtime_versions,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--goal", choices=GOALS, required=True)
    parser.add_argument("--sim-backend", choices=("physx_cpu", "physx_cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    versions = query_planner_runtime_versions(sys.executable)
    if not planner_runtime_matches_expected(versions):
        raise RuntimeError(f"expert runtime pins differ: {versions}")
    env = make_environment(args.sim_backend)
    try:
        observation, _ = env.reset(seed=args.seed, options={"task_spec": task(args.goal).to_dict()})
        audit = reset_audit(env, observation)
        result = expert_result(env)
    finally:
        env.close()
    write_json(
        args.output,
        {
            "seed": args.seed,
            "semantic_goal": args.goal,
            "sim_backend": args.sim_backend,
            **audit,
            **result,
        },
    )
    write_json(args.output.with_suffix(".runtime.json"), versions)


if __name__ == "__main__":
    expert_worker_main()
