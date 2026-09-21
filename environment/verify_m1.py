"""Validate the LangMani M1 environment without modifying the host system.

The default command runs contract checks everywhere and a single-environment CPU
simulation check on native Linux.  ``--target`` additionally requires native Linux,
CUDA, Vulkan, six-way GPU vectorization, policy rendering, object visibility, and a
separate human-render frame.  Generated reports and frames stay under ``outputs/``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
from mani_skill.utils.structs import Pose
from PIL import Image

from langmani.environments.pick_place_by_instruction import CUBE_HALF_SIZE, ENV_ID
from langmani.environments.specs import BIN_IDS, OBJECT_IDS, EpisodeSpec, TaskSpec
from langmani.environments.task_logic import PRIVILEGED_OBSERVATION_KEYS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "diagnostics" / "m1"
REPORT_PATH = OUTPUT_DIR / "verification.json"
POLICY_FRAME_PATH = OUTPUT_DIR / "base_camera_rgb.png"
HUMAN_FRAME_PATH = OUTPUT_DIR / "render_camera_rgb.png"
EVALUATION_KEYS = (
    "target_in_target_bin",
    "target_in_wrong_bin",
    "wrong_object_in_target_bin",
    "target_is_grasped",
    "target_is_static",
    "target_off_table",
    "success",
    "fail",
)
WORKER_TIMEOUT_SECONDS = 600
WORKER_CHOICES = ("cpu", "gpu")


@dataclass
class Report:
    """Collect machine-readable checks and determine the process status."""

    checks: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        name: str,
        status: str,
        detail: str,
        *,
        required: bool = True,
    ) -> None:
        print(f"[{status.upper()}] {name}: {detail}")
        self.checks.append({"name": name, "status": status, "detail": detail, "required": required})

    def check(self, name: str, condition: bool, detail: str) -> None:
        self.record(name, "pass" if condition else "fail", detail)

    def extend(self, checks: list[dict[str, Any]]) -> None:
        """Merge validated checks emitted by one isolated PhysX worker."""
        for check in checks:
            if set(check) != {"name", "status", "detail", "required"}:
                raise ValueError(f"invalid worker check fields: {sorted(check)}")
            if not isinstance(check["name"], str) or not isinstance(check["detail"], str):
                raise TypeError("worker check name and detail must be strings")
            if check["status"] not in {"pass", "fail", "skip"}:
                raise ValueError(f"invalid worker check status: {check['status']!r}")
            if not isinstance(check["required"], bool):
                raise TypeError("worker check required flag must be boolean")
        self.checks.extend(checks)

    @property
    def failed(self) -> bool:
        return any(check["required"] and check["status"] == "fail" for check in self.checks)

    @property
    def skipped(self) -> bool:
        return any(check["status"] == "skip" for check in self.checks)

    def write(self, *, target: bool) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "environment_id": ENV_ID,
            "mode": "target" if target else "cpu-safe",
            "platform": platform.platform(),
            "python": platform.python_version(),
            "versions": {
                package: metadata.version(package)
                for package in ("mani-skill", "gymnasium", "torch")
            },
            "checks": self.checks,
            "result": (
                "failed" if self.failed else "passed_with_skips" if self.skipped else "passed"
            ),
            "physical_target_validated": target and not self.failed and not self.skipped,
        }
        REPORT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"[INFO] report: {REPORT_PATH}")

    def write_worker(self, path: Path) -> None:
        """Write compact subprocess evidence for the parent verifier."""
        path.write_text(
            json.dumps({"checks": self.checks}, indent=2) + "\n",
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        action="store_true",
        help="Require native Linux GPU simulation and Vulkan-backed rendering.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Scene seed (default: 123).")
    parser.add_argument("--verbose", action="store_true", help="Print full tracebacks.")
    parser.add_argument("--_worker", choices=WORKER_CHOICES, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def native_linux() -> bool:
    release = platform.release().lower()
    return (
        platform.system() == "Linux" and "microsoft" not in release and not os.getenv("WSL_INTEROP")
    )


def task_options(object_id: str, bin_id: str) -> dict[str, Any]:
    return {
        "task_spec": {
            "target_object_id": object_id,
            "target_bin_id": bin_id,
            "instruction_template_id": "canonical_v0",
        }
    }


def cube_poses(env: Any) -> torch.Tensor:
    return torch.stack([cube.pose.raw_pose for cube in env.cubes], dim=1)


def teleport_cube_to_bin(env: Any, *, object_index: int, bin_index: int) -> None:
    position = env._bin_floor_centers()[:, bin_index].clone()
    position[:, 2] += CUBE_HALF_SIZE
    quaternion = torch.zeros((env.num_envs, 4), device=env.device)
    quaternion[:, 0] = 1.0
    cube = env.cubes[object_index]
    cube.set_pose(Pose.create_from_pq(p=position, q=quaternion))
    cube.set_linear_velocity(torch.zeros((env.num_envs, 3), device=env.device))
    cube.set_angular_velocity(torch.zeros((env.num_envs, 3), device=env.device))


def has_string_leaf(value: Any) -> bool:
    if isinstance(value, str):
        return True
    if isinstance(value, dict):
        return any(has_string_leaf(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(has_string_leaf(child) for child in value)
    return False


def check_one_step(report: Report, env: Any, *, batch_size: int, label: str) -> None:
    result = env.step(env.action_space.sample())
    tuple_ok = isinstance(result, tuple) and len(result) == 5
    if not tuple_ok:
        report.check(f"{label} step tuple", False, f"received {type(result).__name__}")
        return

    _, reward, terminated, truncated, info = result
    shapes_ok = (
        reward.shape == (batch_size,)
        and terminated.shape == (batch_size,)
        and truncated.shape == (batch_size,)
        and all(key in info and info[key].shape == (batch_size,) for key in EVALUATION_KEYS)
    )
    report.check(
        f"{label} step contract",
        shapes_ok and not has_string_leaf(info),
        "five-tuple, batched reward/termination/evaluation, and no string metadata",
    )


def save_frame(value: torch.Tensor, path: Path) -> None:
    frame = value.detach().cpu().numpy()
    if frame.ndim == 4:
        frame = frame[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path)


def run_contract_checks(report: Report) -> None:
    from mani_skill.utils.registration import REGISTERED_ENVS

    langmani_ids = sorted(uid for uid in REGISTERED_ENVS if uid.startswith("LangMani-"))
    report.check("single registration", langmani_ids == [ENV_ID], repr(langmani_ids))
    registered = REGISTERED_ENVS[ENV_ID]
    report.check(
        "episode time limit",
        registered.max_episode_steps == 200,
        f"max_episode_steps={registered.max_episode_steps}",
    )
    report.check(
        "no downloaded assets",
        registered.asset_download_ids == [],
        f"asset_download_ids={registered.asset_download_ids}",
    )

    episodes = tuple(
        EpisodeSpec.create(
            scene_seed=0,
            task_spec=TaskSpec(
                target_object_id=object_id,
                target_bin_id=bin_id,
                instruction_template_id="canonical_v0",
            ),
        )
        for object_id in OBJECT_IDS
        for bin_id in BIN_IDS
    )
    expected_texts = (
        "Pick up the red cube and place it in the left bin.",
        "Pick up the red cube and place it in the right bin.",
        "Pick up the green cube and place it in the left bin.",
        "Pick up the green cube and place it in the right bin.",
        "Pick up the blue cube and place it in the left bin.",
        "Pick up the blue cube and place it in the right bin.",
    )
    report.check(
        "canonical language",
        tuple(episode.canonical_instruction for episode in episodes) == expected_texts,
        "six exact canonical English instructions",
    )


def run_cpu_simulation(report: Report, seed: int) -> None:
    env = gym.make(
        ENV_ID,
        num_envs=1,
        obs_mode="state_dict",
        reward_mode="sparse",
        control_mode="pd_joint_delta_pos",
        sim_backend="physx_cpu",
        render_backend="none",
    )
    try:
        observation_a, _ = env.reset(seed=seed, options=task_options("red_cube", "left_bin"))
        layout_a = cube_poses(env.unwrapped).clone()
        spec_a = env.unwrapped.get_episode_specs()[0]
        observation_b, info = env.reset(seed=seed, options=task_options("blue_cube", "right_bin"))
        layout_b = cube_poses(env.unwrapped)
        spec_b = env.unwrapped.get_episode_specs()[0]

        report.check(
            "counterfactual layout",
            torch.equal(layout_a, layout_b),
            "same scene seed with different TaskSpec produced identical cube poses",
        )
        report.check(
            "scene/task separation",
            spec_a.scene_id == spec_b.scene_id and spec_a.task_id != spec_b.task_id,
            f"scene_id={spec_b.scene_id}; task_id={spec_b.task_id}",
        )
        report.check(
            "privileged state fields",
            observation_b["extra"].keys() >= PRIVILEGED_OBSERVATION_KEYS,
            repr(sorted(observation_b["extra"])),
        )
        evaluation_shapes_ok = all(
            key in info and info[key].shape == (1,) and info[key].dtype == torch.bool
            for key in EVALUATION_KEYS
        )
        report.check(
            "batched evaluation fields",
            evaluation_shapes_ok,
            "all eight fields are bool tensors with shape (1,)",
        )
        report.check(
            "initial cube velocities",
            all(
                torch.count_nonzero(cube.linear_velocity) == 0
                and torch.count_nonzero(cube.angular_velocity) == 0
                for cube in env.unwrapped.cubes
            ),
            "linear and angular velocities are zero",
        )
        check_one_step(report, env, batch_size=1, label="CPU custom environment")

        env.reset(seed=seed, options=task_options("red_cube", "left_bin"))
        teleport_cube_to_bin(env.unwrapped, object_index=0, bin_index=0)
        zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        settled_step = 0
        for step_index in range(1, 21):
            settled_step = step_index
            _, reward, terminated, _, success_info = env.step(zero_action)
            if bool(success_info["success"][0]):
                break
        report.check(
            "physical success and sparse reward",
            bool(
                success_info["target_in_target_bin"][0]
                and success_info["target_is_static"][0]
                and not success_info["target_is_grasped"][0]
                and success_info["success"][0]
                and reward[0] == 1
                and terminated[0]
            ),
            "teleported released/static target settled in real bin geometry with reward +1 "
            f"after {settled_step} step(s)",
        )

        env.reset(seed=seed, options=task_options("red_cube", "left_bin"))
        teleport_cube_to_bin(env.unwrapped, object_index=0, bin_index=1)
        wrong_bin_info = env.unwrapped.evaluate()
        env.reset(seed=seed, options=task_options("red_cube", "left_bin"))
        teleport_cube_to_bin(env.unwrapped, object_index=1, bin_index=0)
        wrong_object_info = env.unwrapped.evaluate()
        report.check(
            "physical wrong-placement rejection",
            bool(
                wrong_bin_info["target_in_wrong_bin"][0]
                and not wrong_bin_info["success"][0]
                and wrong_object_info["wrong_object_in_target_bin"][0]
                and not wrong_object_info["success"][0]
            ),
            "target-in-wrong-bin and wrong-object-in-target-bin both rejected",
        )
        del observation_a
    finally:
        env.close()


def run_gpu_vectorization(report: Report) -> None:
    env = gym.make(
        ENV_ID,
        num_envs=6,
        obs_mode="state_dict",
        reward_mode="sparse",
        control_mode="pd_joint_delta_pos",
        sim_backend="physx_cuda",
        render_backend="none",
    )
    try:
        observation, info = env.reset(seed=[0, 1, 2, 3, 4, 5])
        report.check(
            "GPU simulation",
            env.unwrapped.gpu_sim_enabled,
            f"device={env.unwrapped.device}",
        )
        report.check(
            "six-way vectorization",
            observation["extra"]["cube_poses"].shape == (6, 21)
            and all(value.shape == (6,) for value in info.values() if torch.is_tensor(value)),
            "num_envs=6",
        )
        report.check(
            "six default tasks",
            len(set(env.unwrapped.get_task_texts())) == 6,
            repr(env.unwrapped.get_task_texts()),
        )

        layout_before = cube_poses(env.unwrapped).clone()
        qpos_before = env.unwrapped.agent.robot.get_qpos().clone()
        specs_before = env.unwrapped.get_episode_specs()
        partial_options = task_options("red_cube", "right_bin")
        partial_options["env_idx"] = [1, 4]
        env.reset(options=partial_options)
        layout_after = cube_poses(env.unwrapped)
        qpos_after = env.unwrapped.agent.robot.get_qpos()
        specs_after = env.unwrapped.get_episode_specs()
        untouched = torch.tensor([0, 2, 3, 5], device=env.unwrapped.device)
        selected = torch.tensor([1, 4], device=env.unwrapped.device)
        partial_reset_ok = (
            torch.equal(layout_before[untouched], layout_after[untouched])
            and torch.equal(qpos_before[untouched], qpos_after[untouched])
            and torch.all(torch.any(layout_before[selected] != layout_after[selected], dim=(1, 2)))
            and all(specs_before[index] == specs_after[index] for index in (0, 2, 3, 5))
            and all(
                specs_after[index].task_spec.target_object_id == "red_cube"
                and specs_after[index].task_spec.target_bin_id == "right_bin"
                for index in (1, 4)
            )
        )
        report.check(
            "GPU partial reset mask",
            bool(partial_reset_ok),
            "env_idx=[1,4] changed only selected layouts/specs and broadcast TaskSpec",
        )
        check_one_step(report, env, batch_size=6, label="GPU custom environment")

        env.reset(
            seed=[10, 11, 12, 13, 14, 15],
            options=task_options("red_cube", "right_bin"),
        )
        teleport_cube_to_bin(env.unwrapped, object_index=0, bin_index=1)
        env.unwrapped.scene._gpu_apply_all()
        zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        settled_step = 0
        for step_index in range(1, 21):
            settled_step = step_index
            _, reward, terminated, _, success_info = env.step(zero_action)
            if bool(torch.all(success_info["success"])):
                break
        report.check(
            "GPU physical success and sparse reward",
            bool(
                torch.all(success_info["target_in_target_bin"])
                and torch.all(success_info["target_is_static"])
                and not torch.any(success_info["target_is_grasped"])
                and torch.all(success_info["success"])
                and torch.all(reward == 1)
                and torch.all(terminated)
            ),
            f"all six environments settled successfully after {settled_step} step(s)",
        )
    finally:
        env.close()


def run_gpu_rendering(report: Report, seed: int) -> None:
    env = gym.make(
        ENV_ID,
        num_envs=1,
        obs_mode="rgb+segmentation",
        reward_mode="none",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        sim_backend="physx_cuda",
        render_backend="sapien_cuda",
    )
    try:
        observation, _ = env.reset(seed=seed)
        extra_keys = set(observation["extra"])
        report.check(
            "visual no-leakage",
            extra_keys == {"tcp_pose"},
            f"visual extra keys={sorted(extra_keys)}",
        )
        textures = observation["sensor_data"]["base_camera"]
        policy_rgb = textures["rgb"]
        report.check(
            "policy camera RGB",
            policy_rgb.shape == (1, 256, 256, 3)
            and policy_rgb.dtype == torch.uint8
            and torch.unique(policy_rgb).numel() > 16,
            f"shape={tuple(policy_rgb.shape)}, dtype={policy_rgb.dtype}",
        )

        actor_segmentation = textures["segmentation"][..., 0]
        task_actor_pixel_counts = {
            actor.name: int(
                torch.count_nonzero(
                    actor_segmentation == int(actor.per_scene_id.detach().cpu().tolist()[0])
                )
                .detach()
                .cpu()
                .tolist()
            )
            for actor in (*env.unwrapped.cubes, *env.unwrapped.bins)
        }
        report.check(
            "policy camera task visibility",
            all(pixel_count >= 4 for pixel_count in task_actor_pixel_counts.values()),
            f"segmentation pixels={task_actor_pixel_counts}",
        )

        links_by_name = {link.name: link for link in env.unwrapped.agent.robot.get_links()}
        hand = links_by_name["panda_hand"]
        hand_id = int(hand.per_scene_id.detach().cpu().tolist()[0])
        hand_pixels = int(
            torch.count_nonzero(actor_segmentation == hand_id).detach().cpu().tolist()
        )
        report.check(
            "policy camera Panda hand visibility",
            hand_pixels >= 4,
            f"panda_hand segmentation pixels={hand_pixels}",
        )

        human_rgb = env.render()
        report.check(
            "human render camera RGB",
            human_rgb.shape == (1, 512, 512, 3)
            and human_rgb.dtype == torch.uint8
            and torch.unique(human_rgb).numel() > 16,
            f"shape={tuple(human_rgb.shape)}, dtype={human_rgb.dtype}",
        )
        save_frame(policy_rgb, POLICY_FRAME_PATH)
        save_frame(human_rgb, HUMAN_FRAME_PATH)
        report.record("diagnostic frames", "pass", f"{POLICY_FRAME_PATH}; {HUMAN_FRAME_PATH}")
    finally:
        env.close()


def _print_worker_output(value: str | bytes | None, *, stderr: bool = False) -> None:
    if not value:
        return
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
    print(text.rstrip(), file=sys.stderr if stderr else sys.stdout)


def run_isolated_worker(
    report: Report,
    worker: str,
    *,
    seed: int,
    verbose: bool,
) -> None:
    """Run one PhysX backend in a fresh process and merge its checks."""
    if worker not in WORKER_CHOICES:
        raise ValueError(f"unsupported M1 worker: {worker!r}")
    with tempfile.TemporaryDirectory(prefix=f"langmani-m1-{worker}-") as directory:
        output_path = Path(directory) / "checks.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_worker",
            worker,
            "--_worker-output",
            str(output_path),
            "--seed",
            str(seed),
        ]
        if verbose:
            command.append("--verbose")
        try:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=WORKER_TIMEOUT_SECONDS,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        except subprocess.TimeoutExpired as error:
            _print_worker_output(error.stdout)
            _print_worker_output(error.stderr, stderr=True)
            report.check(
                f"{worker.upper()} PhysX worker process",
                False,
                f"timed out after {WORKER_TIMEOUT_SECONDS} seconds",
            )
            return

        _print_worker_output(completed.stdout)
        _print_worker_output(completed.stderr, stderr=True)
        if not output_path.is_file():
            report.check(
                f"{worker.upper()} PhysX worker process",
                False,
                f"exit code {completed.returncode}; worker result file was not written",
            )
            return

        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            checks = payload.get("checks")
            if not isinstance(checks, list) or not all(isinstance(item, dict) for item in checks):
                raise TypeError("worker result checks must be a list of mappings")
            report.extend(checks)
        except (OSError, TypeError, ValueError) as error:
            report.check(
                f"{worker.upper()} PhysX worker process",
                False,
                f"malformed worker result: {type(error).__name__}: {error}",
            )
            return
        report.check(
            f"{worker.upper()} PhysX worker process",
            completed.returncode == 0,
            f"isolated process exit code {completed.returncode}",
        )


def run_worker(args: argparse.Namespace) -> int:
    """Execute exactly one simulator backend for the parent verifier."""
    if args._worker_output is None:
        raise ValueError("an internal M1 worker requires --_worker-output")
    report = Report()
    try:
        if args._worker == "cpu":
            run_cpu_simulation(report, args.seed)
        elif args._worker == "gpu":
            run_gpu_vectorization(report)
            run_gpu_rendering(report, args.seed)
        else:
            raise ValueError(f"unsupported M1 worker: {args._worker!r}")
    except Exception as error:  # noqa: BLE001 - worker must preserve third-party diagnostics
        report.record(
            f"{str(args._worker).upper()} PhysX worker exception",
            "fail",
            f"{type(error).__name__}: {error}",
        )
        if args.verbose:
            traceback.print_exc()
    finally:
        report.write_worker(args._worker_output)
    return 1 if report.failed else 0


def main() -> int:
    args = parse_args()
    if args._worker is not None:
        return run_worker(args)
    if args._worker_output is not None:
        raise ValueError("--_worker-output is only valid for an internal M1 worker")

    report = Report()
    print(f"LangMani M1 verification: {'target' if args.target else 'cpu-safe'}")
    run_contract_checks(report)

    is_native = native_linux()
    if is_native:
        run_isolated_worker(report, "cpu", seed=args.seed, verbose=args.verbose)
    else:
        report.record(
            "native Linux CPU simulation",
            "skip",
            "not run outside the repository's native Linux simulator boundary",
            required=False,
        )

    if args.target:
        prerequisites = {
            "native Linux": is_native,
            "CUDA": torch.cuda.is_available(),
            "vulkaninfo": shutil.which("vulkaninfo") is not None,
        }
        for name, available in prerequisites.items():
            report.check(f"target prerequisite: {name}", available, str(available))
        if all(prerequisites.values()):
            run_isolated_worker(report, "gpu", seed=args.seed, verbose=args.verbose)

    report.write(target=args.target)
    return 1 if report.failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001 - command-line diagnostics expose all failures
        print(f"[FATAL] {type(error).__name__}: {error}", file=sys.stderr)
        if "--verbose" in sys.argv:
            traceback.print_exc()
        raise SystemExit(1) from error
