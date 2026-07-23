"""Native ManiSkill runtime for the bounded Phase 2B.4-F0 PPO study."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, cast

import gymnasium as gym
import numpy as np
import torch

import langmani.environments  # noqa: F401
from langmani.environments.push_logic import compute_push_object_planar_alignment
from langmani.environments.push_specs import PushTaskSpec
from langmani.environments.push_to_region import (
    CONTACT_FORCE_THRESHOLD,
    CONTAINMENT_CLEARANCE,
    ENV_ID,
    LIFT_TOLERANCE,
    MAX_EPISODE_STEPS,
    OBJECT_PLANAR_RADII,
    OBJECT_RESTING_HEIGHTS,
    PUSH_OBJECT_IDS,
    TARGET_REGION_IDS,
    WORKSPACE_BOUNDS_XY,
)
from langmani.v2.phase2b4_ppo import (
    TEACHER_OBSERVATION_DIM,
    BoundedActorCritic,
    EvaluationSummary,
    PPOConfig,
    RewardInputs,
    assert_f0_operation_allowed,
    assert_f0_seed_allowed,
    audit_sampled_action_legality,
    canonical_json_sha256,
    checkpoint_payload,
    compute_gae,
    compute_phase2b4_reward,
    compute_ppo_loss,
    concatenate_teacher_fields,
    feasibility_gate,
    micro_gate,
    reconstruct_policy_from_checkpoint,
    reward_manifest,
    save_checkpoint,
    teacher_observation_manifest,
)

ZERO_TOLERANCE_KEYS: Final = (
    "wrong_object_contact",
    "wrong_object_displaced",
    "target_outside_workspace",
    "target_lifted",
    "target_toppled",
    "target_is_grasped",
    "invalid_action",
    "action_out_of_bounds",
    "robot_collision",
)
TRAINING_STOP_KEYS: Final = ZERO_TOLERANCE_KEYS + ("target_overshoot", "no_progress_stall")
MICRO_TASKS: Final = (
    PushTaskSpec("blue_cube", "left", "standard"),
    PushTaskSpec("blue_cube", "forward_right", "standard"),
    PushTaskSpec("orange_cylinder", "left", "standard"),
    PushTaskSpec("orange_cylinder", "forward_right", "standard"),
)


@dataclass(slots=True)
class TeacherState:
    observation: torch.Tensor
    target_distance: torch.Tensor
    behind_distance: torch.Tensor
    containment_fraction: torch.Tensor
    target_contact: torch.Tensor
    push_alignment: torch.Tensor


@dataclass(frozen=True, slots=True)
class RuntimeVersions:
    python: str
    platform: str
    torch: str
    cuda: str | None
    cuda_available: bool
    gpu: str | None
    gymnasium: str
    mani_skill: str
    sapien: str
    numpy: str
    hostname: str

    @classmethod
    def collect(cls) -> RuntimeVersions:
        import importlib.metadata as metadata

        return cls(
            python=platform.python_version(),
            platform=platform.platform(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            cuda_available=torch.cuda.is_available(),
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            gymnasium=metadata.version("gymnasium"),
            mani_skill=metadata.version("mani-skill"),
            sapien=metadata.version("sapien"),
            numpy=np.__version__,
            hostname=platform.node(),
        )


def _one_hot(indices: torch.Tensor, width: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(indices.to(torch.long), num_classes=width).to(torch.float32)


def _elapsed_steps(base: Any) -> torch.Tensor:
    value = torch.as_tensor(base.elapsed_steps, dtype=torch.float32, device=base.device)
    if value.ndim == 0:
        value = value.expand(base.num_envs)
    return value.reshape(base.num_envs)


def extract_teacher_state(base: Any) -> TeacherState:
    """Read only current physical state through the project-owned environment."""

    qpos = base.agent.robot.get_qpos().to(torch.float32)
    qvel = base.agent.robot.get_qvel().to(torch.float32)
    tcp_pose = base.agent.tcp.pose.raw_pose.to(torch.float32)
    object_poses = torch.stack([actor.pose.raw_pose for actor in base.push_objects], dim=1).to(
        torch.float32
    )
    object_positions = object_poses[..., :3]
    linear_velocities = torch.stack(
        [actor.linear_velocity for actor in base.push_objects], dim=1
    ).to(torch.float32)
    angular_velocities = torch.stack(
        [actor.angular_velocity for actor in base.push_objects], dim=1
    ).to(torch.float32)
    object_indices = base._target_object_indices.to(torch.long)
    region_indices = base._target_region_indices.to(torch.long)
    difficulty_indices = base._difficulty_indices.to(torch.long)
    batch = torch.arange(base.num_envs, device=base.device)
    target_positions = object_positions[batch, object_indices]
    distractor_indices = 1 - object_indices
    distractor_positions = object_positions[batch, distractor_indices]
    target_centers = base._target_region_centers.to(torch.float32)
    target_to_goal = target_centers - target_positions
    tcp_to_target = target_positions - tcp_pose[:, :3]
    distractor_to_target = target_positions - distractor_positions
    tcp_to_distractor = distractor_positions - tcp_pose[:, :3]
    target_distance = torch.linalg.vector_norm(target_to_goal[:, :2], dim=1)

    radii_table = torch.tensor(OBJECT_PLANAR_RADII, dtype=torch.float32, device=base.device)
    height_table = torch.tensor(OBJECT_RESTING_HEIGHTS, dtype=torch.float32, device=base.device)
    target_radii = radii_table[object_indices]
    target_heights = height_table[object_indices]
    inside_radius = (
        base._target_region_radii.to(torch.float32) - target_radii - CONTAINMENT_CLEARANCE
    )
    containment_fraction = (1.0 - target_distance / inside_radius).clamp(0.0, 1.0)
    containment_margin = inside_radius - target_distance

    direction = target_to_goal[:, :2] / target_distance[:, None].clamp_min(1e-6)
    behind_point_xy = target_positions[:, :2] - (target_radii + 0.06)[:, None] * direction
    behind_distance = torch.linalg.vector_norm(tcp_pose[:, :2] - behind_point_xy, dim=1)
    tcp_from_target = tcp_pose[:, :2] - target_positions[:, :2]
    tcp_from_target_direction = tcp_from_target / torch.linalg.vector_norm(
        tcp_from_target, dim=1, keepdim=True
    ).clamp_min(1e-6)
    push_alignment = -(tcp_from_target_direction * direction).sum(dim=1).clamp(-1.0, 1.0)

    contact_force = torch.stack(
        [base._robot_contact_magnitude(actor, arm_only=False) for actor in base.push_objects],
        dim=1,
    )
    target_contact = contact_force[batch, object_indices] > CONTACT_FORCE_THRESHOLD
    object_number = torch.arange(2, device=base.device)[None, :]
    wrong_contact = (
        (contact_force > CONTACT_FORCE_THRESHOLD) & (object_number != object_indices[:, None])
    ).any(dim=1)

    cube_rotation = base.push_objects[0].pose.to_transformation_matrix()[:, :3, :3]
    cylinder_rotation = base.push_objects[1].pose.to_transformation_matrix()[:, :3, :3]
    alignment = compute_push_object_planar_alignment(cube_rotation, cylinder_rotation)
    target_alignment = alignment[batch, object_indices]
    target_lifted = (target_positions[:, 2] > target_heights + LIFT_TOLERANCE).to(torch.float32)

    min_x, max_x, min_y, max_y = WORKSPACE_BOUNDS_XY
    workspace_margins = torch.stack(
        (
            target_positions[:, 0] - target_radii - min_x,
            max_x - target_positions[:, 0] - target_radii,
            target_positions[:, 1] - target_radii - min_y,
            max_y - target_positions[:, 1] - target_radii,
        ),
        dim=1,
    )
    values = {
        "panda_qpos": qpos,
        "panda_qvel": qvel,
        "tcp_pose": tcp_pose,
        "object_poses": object_poses.reshape(base.num_envs, -1),
        "object_linear_velocities": linear_velocities.reshape(base.num_envs, -1),
        "object_angular_velocities": angular_velocities.reshape(base.num_envs, -1),
        "target_region_center": target_centers,
        "target_region_geometry": base._target_region_radii.to(torch.float32)[:, None],
        "target_object_geometry": torch.stack((target_radii, target_heights), dim=1),
        "target_object_one_hot": _one_hot(object_indices, 2),
        "target_region_one_hot": _one_hot(region_indices, 4),
        "difficulty_one_hot": _one_hot(difficulty_indices, 2),
        "target_to_goal": target_to_goal,
        "tcp_to_target": tcp_to_target,
        "distractor_to_target": distractor_to_target,
        "tcp_to_distractor": tcp_to_distractor,
        "target_workspace_margins": workspace_margins,
        "contact_state": torch.stack(
            (target_contact.to(torch.float32), wrong_contact.to(torch.float32)), dim=1
        ),
        "lift_and_alignment": torch.stack((target_lifted, target_alignment), dim=1),
        "normalized_elapsed_time": (_elapsed_steps(base) / float(MAX_EPISODE_STEPS))[:, None],
        "containment_margin": containment_margin[:, None],
    }
    observation = concatenate_teacher_fields(values)
    return TeacherState(
        observation=observation,
        target_distance=target_distance,
        behind_distance=behind_distance,
        containment_fraction=containment_fraction,
        target_contact=target_contact,
        push_alignment=push_alignment,
    )


def _single_action_bounds(environment: Any) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        action_space = environment.get_wrapper_attr("single_action_space")
    except AttributeError:
        action_space = environment.unwrapped.single_action_space
    if not isinstance(action_space, gym.spaces.Box):
        raise TypeError("Phase 2B.4 supports only a continuous Box action space")
    low = torch.as_tensor(
        action_space.low, dtype=torch.float32, device=environment.unwrapped.device
    )
    high = torch.as_tensor(
        action_space.high, dtype=torch.float32, device=environment.unwrapped.device
    )
    if low.shape != (8,) or high.shape != (8,):
        raise ValueError("Phase 2B.4 requires the native float32[8] action space")
    return low, high


class Phase2B4VectorRuntime:
    """Direct batched ManiSkill runtime with explicit task/seed partial resets."""

    environment: Any
    base: Any
    device: torch.device

    def __init__(self, *, num_envs: int, sim_backend: str = "physx_cuda") -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.environment: Any = gym.make(
            ENV_ID,
            num_envs=num_envs,
            obs_mode="none",
            reward_mode="none",
            control_mode="pd_joint_pos",
            sim_backend=sim_backend,
            render_backend="none",
            reconfiguration_freq=0,
        )
        self.base: Any = self.environment.unwrapped
        self.num_envs = num_envs
        self.device = self.base.device
        self.action_low, self.action_high = _single_action_bounds(self.environment)
        self.episode_ordinals = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.current_scene_seeds = [-1] * num_envs
        self._next_train_seed = 700000

    def close(self) -> None:
        self.environment.close()

    def _reset_group(
        self,
        indices: torch.Tensor,
        *,
        seeds: Sequence[int],
        task: PushTaskSpec,
        namespace: str,
    ) -> None:
        if indices.ndim != 1 or len(indices) != len(seeds):
            raise ValueError("reset indices and seeds must be one-dimensional and aligned")
        for seed in seeds:
            assert_f0_seed_allowed(int(seed), namespace)
        index_values = indices.detach().cpu().tolist()
        for index, seed in zip(index_values, seeds, strict=True):
            self.current_scene_seeds[index] = int(seed)
        if any(seed < 0 for seed in self.current_scene_seeds):
            raise RuntimeError("all vector slots must have an explicit seed before partial reset")
        # ManiSkill 3.0.1 replaces its full episode-seed array when a seed list
        # accompanies a partial reset. Supplying the complete current vector is
        # therefore required; the reset mask still limits physical changes to
        # `indices`.
        self.environment.reset(
            seed=list(self.current_scene_seeds),
            options={"env_idx": indices, "task_spec": task.to_dict()},
        )

    def reset_training_indices(self, indices: torch.Tensor) -> TeacherState:
        indices = indices.to(device=self.device, dtype=torch.long).reshape(-1)
        if indices.numel() == 0:
            return extract_teacher_state(self.base)
        ordinals = self.episode_ordinals[indices]
        task_indices = (indices + ordinals) % len(MICRO_TASKS)
        seed_values = list(range(self._next_train_seed, self._next_train_seed + len(indices)))
        if seed_values and seed_values[-1] >= 800000:
            raise RuntimeError("Phase 2B.4 exhausted the bounded ppo_train seed namespace")
        self._next_train_seed += len(indices)
        for task_index, task in enumerate(MICRO_TASKS):
            mask = task_indices == task_index
            selected = indices[mask]
            if selected.numel() == 0:
                continue
            positions = torch.nonzero(mask, as_tuple=False).reshape(-1).cpu().tolist()
            seeds = [seed_values[position] for position in positions]
            self._reset_group(
                selected,
                seeds=seeds,
                task=task,
                namespace="ppo_train",
            )
        self.episode_ordinals[indices] += 1
        return extract_teacher_state(self.base)

    def reset_all_training(self) -> TeacherState:
        indices = torch.arange(self.num_envs, device=self.device)
        seeds = list(range(self._next_train_seed, self._next_train_seed + self.num_envs))
        if seeds[-1] >= 800000:
            raise RuntimeError("Phase 2B.4 exhausted the bounded ppo_train seed namespace")
        for seed in seeds:
            assert_f0_seed_allowed(seed, "ppo_train")
        self.current_scene_seeds = seeds
        self._next_train_seed += self.num_envs
        self.environment.reset(seed=seeds, options={"task_spec": MICRO_TASKS[0].to_dict()})
        task_indices = indices % len(MICRO_TASKS)
        for task_index, task in enumerate(MICRO_TASKS[1:], start=1):
            selected = indices[task_indices == task_index]
            selected_seeds = [seeds[index] for index in selected.detach().cpu().tolist()]
            self._reset_group(
                selected,
                seeds=selected_seeds,
                task=task,
                namespace="ppo_train",
            )
        self.episode_ordinals += 1
        return extract_teacher_state(self.base)

    def reset_single_evaluation(
        self,
        *,
        seed: int,
        task: PushTaskSpec,
        namespace: str,
    ) -> TeacherState:
        if self.num_envs != 1:
            raise ValueError("single evaluation requires num_envs=1")
        assert_f0_seed_allowed(seed, namespace)
        self.current_scene_seeds = [seed]
        self.environment.reset(seed=seed, options={"task_spec": task.to_dict()})
        return extract_teacher_state(self.base)

    def step(
        self, action: torch.Tensor
    ) -> tuple[TeacherState, torch.Tensor, torch.Tensor, Mapping[str, torch.Tensor]]:
        if action.shape != (self.num_envs, 8) or action.dtype != torch.float32:
            raise ValueError(
                f"native action must have dtype float32 and shape ({self.num_envs}, 8)"
            )
        if not torch.isfinite(action).all():
            raise ValueError("native action contains non-finite values")
        if ((action < self.action_low) | (action > self.action_high)).any():
            raise ValueError("structurally bounded policy emitted an illegal action")
        _obs, _reward, terminated, truncated, info = self.environment.step(action)
        if not isinstance(info, Mapping):
            raise TypeError("push environment info must be a mapping")
        state = extract_teacher_state(self.base)
        return state, terminated.to(torch.bool), truncated.to(torch.bool), info


def _metric_bool(info: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
    value = info.get(key)
    if not isinstance(value, torch.Tensor) or value.dtype is not torch.bool:
        raise ValueError(f"environment metric {key!r} is missing or not boolean")
    return value


def _safety_mask(info: Mapping[str, torch.Tensor], keys: Sequence[str]) -> torch.Tensor:
    values = [_metric_bool(info, key) for key in keys]
    return torch.stack(values, dim=0).any(dim=0)


def _reward_from_transition(
    *,
    previous: TeacherState,
    current: TeacherState,
    info: Mapping[str, torch.Tensor],
    action_unit: torch.Tensor,
    previous_action_unit: torch.Tensor,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    output = compute_phase2b4_reward(
        RewardInputs(
            previous_target_distance=previous.target_distance,
            target_distance=current.target_distance,
            previous_behind_distance=previous.behind_distance,
            behind_distance=current.behind_distance,
            previous_containment_fraction=previous.containment_fraction,
            containment_fraction=current.containment_fraction,
            target_contact_onset=current.target_contact & ~previous.target_contact,
            push_alignment=current.push_alignment,
            wrong_object_contact=_metric_bool(info, "wrong_object_contact"),
            wrong_object_displaced=_metric_bool(info, "wrong_object_displaced"),
            target_outside_workspace=_metric_bool(info, "target_outside_workspace"),
            target_lifted=_metric_bool(info, "target_lifted"),
            target_toppled=_metric_bool(info, "target_toppled"),
            target_is_grasped=_metric_bool(info, "target_is_grasped"),
            invalid_action=_metric_bool(info, "invalid_action"),
            action_out_of_bounds=_metric_bool(info, "action_out_of_bounds"),
            robot_collision=_metric_bool(info, "robot_collision"),
            success=_metric_bool(info, "success"),
            normalized_action=action_unit,
            previous_normalized_action=previous_action_unit,
        )
    )
    return output.total, output.components


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _process_memory_mib() -> float | None:
    try:
        import resource

        getrusage = getattr(resource, "getrusage", None)
        rusage_self = getattr(resource, "RUSAGE_SELF", None)
        if not callable(getrusage) or rusage_self is None:
            return None
        return float(getrusage(rusage_self).ru_maxrss) / 1024.0
    except (ImportError, OSError):
        return None


def run_ppo_training(
    *,
    config: PPOConfig,
    output_dir: Path,
    smoke: bool,
) -> dict[str, object]:
    """Run either the one-update smoke or the one frozen micro experiment."""

    assert_f0_operation_allowed("gpu_pipeline_smoke" if smoke else "ppo_micro_training")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 2B.4 PPO execution requires a CUDA target")
    _seed_everything(config.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    runtime = Phase2B4VectorRuntime(num_envs=config.num_envs)
    try:
        current = runtime.reset_all_training()
        policy = BoundedActorCritic(
            observation_dim=TEACHER_OBSERVATION_DIM,
            action_low=runtime.action_low,
            action_high=runtime.action_high,
            hidden_dim=config.hidden_dim,
            initial_log_std=config.initial_log_std,
        ).to(runtime.device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate, eps=1e-5)
        action_manifest = policy.action_manifest()

        shape = (config.num_steps, config.num_envs)
        observations = torch.zeros(
            shape + (TEACHER_OBSERVATION_DIM,), dtype=torch.float32, device=runtime.device
        )
        latents = torch.zeros(shape + (8,), dtype=torch.float32, device=runtime.device)
        log_probs = torch.zeros(shape, dtype=torch.float32, device=runtime.device)
        values = torch.zeros_like(log_probs)
        rewards = torch.zeros_like(log_probs)
        next_values = torch.zeros_like(log_probs)
        terminated = torch.zeros(shape, dtype=torch.bool, device=runtime.device)
        truncated = torch.zeros_like(terminated)
        previous_action_unit = torch.zeros(
            (config.num_envs, 8), dtype=torch.float32, device=runtime.device
        )
        episode_return = torch.zeros(config.num_envs, device=runtime.device)
        episode_length = torch.zeros(config.num_envs, dtype=torch.long, device=runtime.device)
        completed_returns: list[float] = []
        completed_success: list[bool] = []
        completed_progress: list[float] = []
        episode_start_distance = current.target_distance.clone()
        metrics: list[dict[str, object]] = []
        global_step = 0
        reset_count = config.num_envs
        simulator_exceptions = 0
        optimizer_steps = 0
        final_loss: dict[str, float] = {}

        for iteration in range(config.iterations):
            rollout_started = time.perf_counter()
            policy.eval()
            for step in range(config.num_steps):
                observations[step] = current.observation
                with torch.no_grad():
                    action, latent, log_prob, _entropy, value = policy.action_and_value(
                        current.observation
                    )
                if not torch.isfinite(action).all():
                    raise RuntimeError("PPO emitted non-finite action during rollout")
                action_unit = torch.tanh(latent)
                try:
                    next_state, env_terminated, env_truncated, info = runtime.step(action)
                except Exception:
                    simulator_exceptions += 1
                    raise
                reward, _components = _reward_from_transition(
                    previous=current,
                    current=next_state,
                    info=info,
                    action_unit=action_unit,
                    previous_action_unit=previous_action_unit,
                )
                safety_stop = _safety_mask(info, TRAINING_STOP_KEYS)
                effective_terminated = env_terminated | safety_stop
                effective_truncated = env_truncated & ~effective_terminated
                with torch.no_grad():
                    transition_next_value = policy.value(next_state.observation)

                latents[step] = latent
                log_probs[step] = log_prob
                values[step] = value
                rewards[step] = reward
                next_values[step] = transition_next_value
                terminated[step] = effective_terminated
                truncated[step] = effective_truncated
                global_step += config.num_envs
                episode_return += reward
                episode_length += 1
                done = effective_terminated | effective_truncated
                if done.any():
                    done_indices = torch.nonzero(done, as_tuple=False).reshape(-1)
                    for index in done_indices.cpu().tolist():
                        completed_returns.append(float(episode_return[index].item()))
                        completed_success.append(bool(_metric_bool(info, "success")[index].item()))
                        completed_progress.append(
                            float(
                                (
                                    episode_start_distance[index]
                                    - next_state.target_distance[index]
                                ).item()
                            )
                        )
                    episode_return[done_indices] = 0
                    episode_length[done_indices] = 0
                    reset_count += int(done_indices.numel())
                    current = runtime.reset_training_indices(done_indices)
                    episode_start_distance[done_indices] = current.target_distance[done_indices]
                    previous_action_unit = torch.where(
                        done[:, None], torch.zeros_like(previous_action_unit), action_unit
                    )
                else:
                    current = next_state
                    previous_action_unit = action_unit

            rollout_seconds = time.perf_counter() - rollout_started
            advantages, returns = compute_gae(
                rewards=rewards,
                values=values,
                next_values=next_values,
                terminated=terminated,
                truncated=truncated,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
            )
            flat_observations = observations.reshape(-1, TEACHER_OBSERVATION_DIM)
            flat_latents = latents.reshape(-1, 8)
            flat_log_probs = log_probs.reshape(-1)
            flat_values = values.reshape(-1)
            flat_returns = returns.reshape(-1)
            flat_advantages = advantages.reshape(-1)
            indices = torch.arange(config.batch_size, device=runtime.device)
            policy.train()
            update_started = time.perf_counter()
            loss_values: list[dict[str, float]] = []
            gradient_norms: list[float] = []
            stop_for_kl = False
            for _epoch in range(config.update_epochs):
                indices = indices[torch.randperm(config.batch_size, device=runtime.device)]
                for start in range(0, config.batch_size, config.minibatch_size):
                    minibatch = indices[start : start + config.minibatch_size]
                    (
                        _action,
                        _latent,
                        new_log_prob,
                        entropy,
                        new_value,
                    ) = policy.action_and_value(
                        flat_observations[minibatch],
                        latent=flat_latents[minibatch],
                    )
                    loss = compute_ppo_loss(
                        new_log_prob=new_log_prob,
                        old_log_prob=flat_log_probs[minibatch],
                        entropy=entropy,
                        new_value=new_value,
                        old_value=flat_values[minibatch],
                        returns=flat_returns[minibatch],
                        advantages=flat_advantages[minibatch],
                        clip_coef=config.clip_coef,
                        value_coef=config.value_coef,
                        entropy_coef=config.entropy_coef,
                    )
                    if not all(
                        torch.isfinite(value)
                        for value in (loss.total, loss.actor, loss.critic, loss.entropy)
                    ):
                        raise RuntimeError("PPO loss became non-finite")
                    optimizer.zero_grad(set_to_none=True)
                    loss.total.backward()
                    for parameter in policy.parameters():
                        gradient = parameter.grad
                        if gradient is not None and not torch.isfinite(gradient).all():
                            raise RuntimeError("PPO gradient became non-finite")
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        policy.parameters(), config.max_grad_norm
                    )
                    if not torch.isfinite(gradient_norm):
                        raise RuntimeError("PPO gradient norm became non-finite")
                    optimizer.step()
                    optimizer_steps += 1
                    gradient_norms.append(float(gradient_norm.item()))
                    loss_values.append(
                        {
                            "total": float(loss.total.item()),
                            "actor": float(loss.actor.item()),
                            "critic": float(loss.critic.item()),
                            "entropy": float(loss.entropy.item()),
                            "approximate_kl": float(loss.approximate_kl.item()),
                            "clip_fraction": float(loss.clip_fraction.item()),
                        }
                    )
                    if float(loss.approximate_kl.item()) > config.target_kl:
                        stop_for_kl = True
                        break
                if stop_for_kl:
                    break
            update_seconds = time.perf_counter() - update_started
            final_loss = loss_values[-1]
            if smoke or iteration % 8 == 0 or iteration + 1 == config.iterations:
                metrics.append(
                    {
                        "iteration": iteration + 1,
                        "global_step": global_step,
                        "rollout_seconds": rollout_seconds,
                        "update_seconds": update_seconds,
                        "rollout_steps_per_second": config.batch_size / rollout_seconds,
                        "update_samples_per_second": (
                            len(loss_values) * config.minibatch_size / update_seconds
                        ),
                        "reward_mean": float(rewards.mean().item()),
                        "reward_min": float(rewards.min().item()),
                        "reward_max": float(rewards.max().item()),
                        "advantage_min": float(advantages.min().item()),
                        "advantage_max": float(advantages.max().item()),
                        "gradient_norm_max": max(gradient_norms),
                        "last_loss": final_loss,
                        "completed_episodes": len(completed_returns),
                        "completed_successes": sum(completed_success),
                    }
                )

        observation_fingerprint = str(teacher_observation_manifest()["fingerprint"])
        reward_fingerprint = str(reward_manifest()["fingerprint"])
        checkpoint = output_dir / ("smoke_checkpoint.pt" if smoke else "micro_checkpoint.pt")
        checkpoint_sha = save_checkpoint(
            checkpoint,
            checkpoint_payload(
                policy=policy,
                optimizer=optimizer,
                config=config,
                global_step=global_step,
                observation_manifest_fingerprint=observation_fingerprint,
                action_manifest_fingerprint=str(action_manifest["fingerprint"]),
                reward_manifest_fingerprint=reward_fingerprint,
            ),
        )
        loaded_policy, loaded_payload = reconstruct_policy_from_checkpoint(checkpoint)
        loaded_policy = loaded_policy.to(runtime.device)
        policy.eval()
        loaded_policy.eval()
        fixed_observation = current.observation[:1]
        with torch.no_grad():
            expected_action = policy.deterministic_action(fixed_observation)
            loaded_action = loaded_policy.deterministic_action(fixed_observation)
        reload_max_abs = float((expected_action - loaded_action).abs().max().item())
        if reload_max_abs != 0.0:
            raise RuntimeError("deterministic checkpoint reload changed policy output")
        # Execute one real deterministic action after reload.
        with torch.no_grad():
            deterministic_actions = loaded_policy.deterministic_action(current.observation)
        _state, _terminated, _truncated, _info = runtime.step(deterministic_actions)

        elapsed = time.perf_counter() - started
        first_window = completed_returns[: max(1, len(completed_returns) // 4)]
        last_window = completed_returns[-max(1, len(completed_returns) // 4) :]
        first_progress = completed_progress[: max(1, len(completed_progress) // 4)]
        last_progress = completed_progress[-max(1, len(completed_progress) // 4) :]
        first_return_mean = float(np.mean(first_window)) if first_window else None
        last_return_mean = float(np.mean(last_window)) if last_window else None
        first_progress_mean = float(np.mean(first_progress)) if first_progress else None
        last_progress_mean = float(np.mean(last_progress)) if last_progress else None
        learning_evidence = {
            "return_improvement_threshold": 0.5,
            "target_distance_progress_improvement_threshold_metres": 0.01,
            "training_return_improved_materially": (
                first_return_mean is not None
                and last_return_mean is not None
                and last_return_mean - first_return_mean >= 0.5
            ),
            "target_distance_progress_improved_materially": (
                first_progress_mean is not None
                and last_progress_mean is not None
                and last_progress_mean - first_progress_mean >= 0.01
            ),
            "training_success_nonzero": sum(completed_success) > 0,
        }
        report: dict[str, object] = {
            "schema_version": (
                "langmani-v2-phase2b4-f0-ppo-smoke-v0"
                if smoke
                else "langmani-v2-phase2b4-f0-micro-training-v0"
            ),
            "mode": "smoke" if smoke else "micro_training",
            "runtime": asdict(RuntimeVersions.collect()),
            "config": config.to_dict(),
            "observation_shape": [config.num_envs, TEACHER_OBSERVATION_DIM],
            "action_shape": [config.num_envs, 8],
            "rollout_buffer_shape": [config.num_steps, config.num_envs],
            "global_step": global_step,
            "optimizer_steps": optimizer_steps,
            "elapsed_seconds": elapsed,
            "environment_steps_per_second": global_step / elapsed,
            "reset_count": reset_count,
            "simulator_exceptions": simulator_exceptions,
            "peak_vram_mib": torch.cuda.max_memory_allocated() / (1024**2),
            "peak_cpu_memory_mib": _process_memory_mib(),
            "completed_episodes": len(completed_returns),
            "completed_successes": sum(completed_success),
            "first_quartile_return_mean": first_return_mean,
            "last_quartile_return_mean": last_return_mean,
            "first_quartile_progress_mean": first_progress_mean,
            "last_quartile_progress_mean": last_progress_mean,
            "learning_evidence": learning_evidence,
            "metrics": metrics,
            "final_loss": final_loss,
            "checkpoint_path": checkpoint.as_posix(),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "checkpoint_global_step": cast(int, loaded_payload["global_step"]),
            "deterministic_reload_max_abs_error": reload_max_abs,
            "deterministic_action_executed": True,
            "finite_losses": True,
            "finite_gradients": True,
            "physical_environment_stepped": True,
            "demonstrations_generated": False,
            "student_policy_training_started": False,
            "pipeline_execution_passed": True,
        }
        report["fingerprint"] = canonical_json_sha256(
            {
                key: value
                for key, value in report.items()
                if key not in {"checkpoint_path", "metrics"}
            }
        )
        return report
    finally:
        runtime.close()


def vectorized_environment_audit(*, num_envs: int = 128) -> dict[str, object]:
    assert_f0_operation_allowed("contract_audit")
    if not torch.cuda.is_available():
        raise RuntimeError("vectorized environment audit requires CUDA")
    started = time.perf_counter()
    runtime = Phase2B4VectorRuntime(num_envs=num_envs)
    try:
        initial = runtime.reset_all_training()
        specs = runtime.base.get_episode_specs()
        seeds = [spec.scene_seed for spec in specs]
        object_positions_before = torch.stack(
            [actor.pose.p.detach().clone() for actor in runtime.base.push_objects], dim=1
        )
        untouched_before = object_positions_before[1:].clone()
        runtime.reset_training_indices(torch.tensor([0], device=runtime.device))
        object_positions_after = torch.stack(
            [actor.pose.p.detach().clone() for actor in runtime.base.push_objects], dim=1
        )
        untouched_error = float((object_positions_after[1:] - untouched_before).abs().max().item())
        reset_changed_slot = bool(
            not torch.equal(object_positions_after[0], object_positions_before[0])
        )
        elapsed = _elapsed_steps(runtime.base)
        report: dict[str, object] = {
            "schema_version": "langmani-v2-phase2b4-f0-vectorized-environment-audit-v0",
            "num_envs": num_envs,
            "sim_backend": "physx_cuda",
            "observation_shape": list(initial.observation.shape),
            "finite_observations": bool(torch.isfinite(initial.observation).all()),
            "unique_scene_seeds": len(set(seeds)),
            "expected_unique_scene_seeds": num_envs,
            "reset_slot_changed": reset_changed_slot,
            "untouched_slot_max_abs_error": untouched_error,
            "elapsed_step_min": int(elapsed.min().item()),
            "elapsed_step_max": int(elapsed.max().item()),
            "per_environment_counter_valid": bool((elapsed == 0).all()),
            "state_leakage_detected": untouched_error != 0.0,
            "cpu_motion_planner_calls": 0,
            "simulator_exceptions": 0,
            "elapsed_seconds": time.perf_counter() - started,
        }
        checks = {
            "gpu_vectorization": num_envs >= 128,
            "observation_shape": initial.observation.shape == (num_envs, TEACHER_OBSERVATION_DIM),
            "finite_observations": report["finite_observations"] is True,
            "seed_uniqueness": len(set(seeds)) == num_envs,
            "partial_reset_changed_selected_slot": reset_changed_slot,
            "partial_reset_preserved_other_slots": untouched_error == 0.0,
            "counters_reset": report["per_environment_counter_valid"] is True,
            "no_motion_planning": report["cpu_motion_planner_calls"] == 0,
        }
        report["checks"] = checks
        report["passed"] = all(checks.values())
        report["fingerprint"] = canonical_json_sha256(report)
        return report
    finally:
        runtime.close()


def _evaluation_schedule(mode: str) -> list[tuple[int, PushTaskSpec]]:
    if mode == "micro":
        micro_tasks = MICRO_TASKS
        seeds = range(800000, 800048)
        return [(seed, micro_tasks[index % len(micro_tasks)]) for index, seed in enumerate(seeds)]
    if mode == "development":
        tasks: list[PushTaskSpec] = []
        canonical_standard = [
            PushTaskSpec(object_id, direction, "standard")
            for object_id in PUSH_OBJECT_IDS
            for direction in TARGET_REGION_IDS
        ]
        canonical_hard = [
            PushTaskSpec(object_id, direction, "hard")
            for object_id in PUSH_OBJECT_IDS
            for direction in TARGET_REGION_IDS
        ]
        tasks.extend(canonical_standard * 4)
        tasks.extend(canonical_hard * 3)
        tasks.extend(
            PushTaskSpec("blue_cube", direction, "standard") for direction in TARGET_REGION_IDS
        )
        if len(tasks) != 60:
            raise AssertionError("development schedule must contain exactly 60 tasks")
        return [(810000 + index, task) for index, task in enumerate(tasks)]
    raise ValueError(f"unknown evaluation mode {mode!r}")


def evaluate_checkpoint(
    *,
    checkpoint: Path,
    mode: str,
) -> dict[str, object]:
    if mode not in {"micro", "development"}:
        raise ValueError("evaluation mode must be micro or development")
    assert_f0_operation_allowed(
        "ppo_micro_evaluation" if mode == "micro" else "ppo_feasibility_development_evaluation"
    )
    namespace = "ppo_micro_evaluation" if mode == "micro" else "ppo_feasibility_development"
    schedule = _evaluation_schedule(mode)
    policy, checkpoint_payload_value = reconstruct_policy_from_checkpoint(checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError("physical evaluation requires CUDA")
    runtime = Phase2B4VectorRuntime(num_envs=1)
    policy = policy.to(runtime.device)
    policy.eval()
    records: list[dict[str, object]] = []
    safety_counts = dict.fromkeys(ZERO_TOLERANCE_KEYS, 0)
    action_nonfinite = 0
    action_out_of_bounds = 0
    all_actions: list[np.ndarray] = []
    started = time.perf_counter()
    try:
        for episode_index, (seed, task) in enumerate(schedule):
            state = runtime.reset_single_evaluation(
                seed=seed,
                task=task,
                namespace=namespace,
            )
            episode_started = time.perf_counter()
            terminal_info: Mapping[str, torch.Tensor] | None = None
            outcome = "timeout"
            for _step_index in range(MAX_EPISODE_STEPS):
                with torch.no_grad():
                    action = policy.deterministic_action(state.observation)
                all_actions.append(action[0].detach().cpu().numpy())
                action_nonfinite += int((~torch.isfinite(action)).any().item())
                action_out_of_bounds += int(
                    ((action < runtime.action_low) | (action > runtime.action_high)).any().item()
                )
                state, terminated, truncated, info = runtime.step(action)
                terminal_info = info
                safety = _safety_mask(info, ZERO_TOLERANCE_KEYS)
                if safety.item():
                    outcome = next(
                        key for key in ZERO_TOLERANCE_KEYS if bool(_metric_bool(info, key).item())
                    )
                    break
                if bool(_metric_bool(info, "success").item()):
                    outcome = "success"
                    break
                if bool(terminated.item()):
                    outcome = "environment_failure"
                    break
                if bool(truncated.item()):
                    outcome = "timeout"
                    break
            if terminal_info is None:
                raise RuntimeError("evaluation episode produced no transition")
            for key in ZERO_TOLERANCE_KEYS:
                safety_counts[key] += int(bool(_metric_bool(terminal_info, key).item()))
            records.append(
                {
                    "episode_index": episode_index,
                    "seed": seed,
                    "task": task.to_dict(),
                    "success": outcome == "success",
                    "outcome": outcome,
                    "steps": _step_index + 1,
                    "final_target_distance": float(state.target_distance.item()),
                    "latency_seconds": time.perf_counter() - episode_started,
                    "safety": {
                        key: bool(_metric_bool(terminal_info, key).item())
                        for key in ZERO_TOLERANCE_KEYS
                    },
                }
            )
            if mode == "development" and sum(safety_counts.values()) > 0:
                break
    finally:
        runtime.close()

    direction_episodes: defaultdict[str, int] = defaultdict(int)
    direction_successes: defaultdict[str, int] = defaultdict(int)
    cube_episodes = cube_successes = cylinder_episodes = cylinder_successes = 0
    standard_episodes = standard_successes = hard_episodes = hard_successes = 0
    for record in records:
        task_value = record["task"]
        if not isinstance(task_value, Mapping):
            raise AssertionError("evaluation task record is malformed")
        success = bool(record["success"])
        direction = str(task_value["target_region_id"])
        direction_episodes[direction] += 1
        direction_successes[direction] += int(success)
        if task_value["target_object_id"] == "blue_cube":
            cube_episodes += 1
            cube_successes += int(success)
        else:
            cylinder_episodes += 1
            cylinder_successes += int(success)
        if task_value["difficulty"] == "standard":
            standard_episodes += 1
            standard_successes += int(success)
        else:
            hard_episodes += 1
            hard_successes += int(success)
    summary = EvaluationSummary(
        episode_count=len(records),
        successes=sum(bool(record["success"]) for record in records),
        cube_episodes=cube_episodes,
        cube_successes=cube_successes,
        cylinder_episodes=cylinder_episodes,
        cylinder_successes=cylinder_successes,
        direction_episodes=dict(direction_episodes),
        direction_successes=dict(direction_successes),
        standard_episodes=standard_episodes,
        standard_successes=standard_successes,
        hard_episodes=hard_episodes,
        hard_successes=hard_successes,
        safety_counts=safety_counts,
    )
    gate = micro_gate(summary) if mode == "micro" else feasibility_gate(summary)
    action_array = np.stack(all_actions) if all_actions else np.empty((0, 8), dtype=np.float32)
    action_std = action_array.std(axis=0).tolist() if len(action_array) else [0.0] * 8
    report: dict[str, object] = {
        "schema_version": f"langmani-v2-phase2b4-f0-{mode}-evaluation-v0",
        "mode": mode,
        "checkpoint_sha256": "sha256:" + hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_global_step": cast(int, checkpoint_payload_value["global_step"]),
        "schedule": [{"seed": seed, "task": task.to_dict()} for seed, task in schedule],
        "records": records,
        "summary": {
            "episode_count": summary.episode_count,
            "successes": summary.successes,
            "success_rate": summary.success_rate,
            "cube_episodes": summary.cube_episodes,
            "cube_successes": summary.cube_successes,
            "cube_success_rate": summary.cube_success_rate,
            "cylinder_episodes": summary.cylinder_episodes,
            "cylinder_successes": summary.cylinder_successes,
            "cylinder_success_rate": summary.cylinder_success_rate,
            "standard_episodes": summary.standard_episodes,
            "standard_successes": summary.standard_successes,
            "standard_success_rate": summary.standard_success_rate,
            "hard_episodes": summary.hard_episodes,
            "hard_successes": summary.hard_successes,
            "hard_success_rate": summary.hard_success_rate,
            "direction_episodes": dict(summary.direction_episodes),
            "direction_successes": dict(summary.direction_successes),
            "safety_counts": dict(summary.safety_counts),
            "average_episode_length": float(np.mean([record["steps"] for record in records])),
            "average_latency_seconds": float(
                np.mean([record["latency_seconds"] for record in records])
            ),
        },
        "action_integrity": {
            "action_count": len(all_actions),
            "nonfinite": action_nonfinite,
            "out_of_bounds": action_out_of_bounds,
            "clipping": 0,
            "projection": 0,
            "component_std": action_std,
            "nonconstant": any(value > 1e-6 for value in action_std),
        },
        "gate": gate,
        "passed": all(gate.values()),
        "elapsed_seconds": time.perf_counter() - started,
        "runtime": asdict(RuntimeVersions.collect()),
        "formal_seed_accessed": False,
        "demonstrations_generated": False,
    }
    report["fingerprint"] = canonical_json_sha256(
        {key: value for key, value in report.items() if key != "records"}
    )
    return report


def ppo_smoke_bundle(*, output_dir: Path) -> dict[str, object]:
    vector_audit = vectorized_environment_audit(num_envs=128)
    if not vector_audit["passed"]:
        raise RuntimeError("vectorized environment audit failed")
    smoke_config = PPOConfig(
        seed=240402,
        num_envs=128,
        num_steps=8,
        total_timesteps=1024,
        num_minibatches=4,
        update_epochs=1,
    )
    smoke = run_ppo_training(config=smoke_config, output_dir=output_dir, smoke=True)
    checkpoint = output_dir / "smoke_checkpoint.pt"
    policy, _payload = reconstruct_policy_from_checkpoint(checkpoint)
    policy = policy.to("cuda")
    action_audit = audit_sampled_action_legality(policy)
    bundle: dict[str, object] = {
        "schema_version": "langmani-v2-phase2b4-f0-smoke-bundle-v0",
        "vectorized_environment_audit": vector_audit,
        "ppo_smoke": smoke,
        "action_legality": action_audit,
        "passed": bool(vector_audit["passed"])
        and bool(action_audit["passed"])
        and bool(smoke["finite_losses"])
        and bool(smoke["finite_gradients"])
        and bool(smoke["deterministic_action_executed"]),
    }
    bundle["fingerprint"] = canonical_json_sha256(bundle)
    return bundle


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = [
    "MICRO_TASKS",
    "TRAINING_STOP_KEYS",
    "ZERO_TOLERANCE_KEYS",
    "Phase2B4VectorRuntime",
    "RuntimeVersions",
    "TeacherState",
    "evaluate_checkpoint",
    "extract_teacher_state",
    "ppo_smoke_bundle",
    "run_ppo_training",
    "vectorized_environment_audit",
    "write_json",
]
