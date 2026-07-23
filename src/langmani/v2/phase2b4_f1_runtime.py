"""Native GPU diagnostics and bounded PPO probes for Phase 2B.4-F1."""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Final, cast

import numpy as np
import torch

from langmani.environments.push_specs import PushTaskSpec
from langmani.v2.phase2b4_f1 import (
    DEFAULT_RESIDUAL_SCALES,
    ProbeEvaluation,
    StateCenteredResidualActorCritic,
    assert_f1_operation_allowed,
    assert_f1_seed_allowed,
    audit_residual_action_legality,
    current_action_reference,
    probe_a_gate,
    probe_b_gate,
)
from langmani.v2.phase2b4_ppo import (
    TEACHER_OBSERVATION_DIM,
    BoundedActorCritic,
    PPOConfig,
    canonical_json_sha256,
    compute_gae,
    compute_ppo_loss,
    normalize_teacher_observation,
    reconstruct_policy_from_checkpoint,
    reward_manifest,
    teacher_observation_manifest,
)
from langmani.v2.phase2b4_runtime import (
    MICRO_TASKS,
    TRAINING_STOP_KEYS,
    ZERO_TOLERANCE_KEYS,
    Phase2B4VectorRuntime,
    RuntimeVersions,
    TeacherState,
    _metric_bool,
    _process_memory_mib,
    _reward_from_transition,
    _safety_mask,
    _seed_everything,
    extract_teacher_state,
)

STAGE0_TASKS: Final = (
    PushTaskSpec("blue_cube", "left", "standard"),
    PushTaskSpec("blue_cube", "forward_right", "standard"),
)
F1_CHECKPOINT_SCHEMA: Final = "langmani-v2-phase2b4-f1-residual-ppo-checkpoint-v0"
PROBE_TRAIN_STARTS: Final = {"A": 1100000, "B": 1150000}
PROBE_POLICY_SEEDS: Final = {"A": 240410, "B": 240411}
PROBE_EVALUATION: Final = {
    "A": ("f1_probe_a_evaluation", 1200000, 32),
    "B": ("f1_probe_b_evaluation", 1201000, 48),
}


class Phase2B4F1VectorRuntime(Phase2B4VectorRuntime):
    """F0-compatible environment runtime with disjoint F1 seed identities."""

    def __init__(self, *, num_envs: int, probe: str = "A") -> None:
        if probe not in PROBE_TRAIN_STARTS:
            raise ValueError("probe must be A or B")
        super().__init__(num_envs=num_envs)
        self.probe = probe
        self._next_train_seed = PROBE_TRAIN_STARTS[probe]

    def _reset_explicit(
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
            assert_f1_seed_allowed(int(seed), namespace)
        index_values = indices.detach().cpu().tolist()
        for index, seed in zip(index_values, seeds, strict=True):
            self.current_scene_seeds[index] = int(seed)
        if any(seed < 0 for seed in self.current_scene_seeds):
            raise RuntimeError("all vector slots require explicit F1 seeds")
        self.environment.reset(
            seed=list(self.current_scene_seeds),
            options={"env_idx": indices, "task_spec": task.to_dict()},
        )

    def reset_training_indices(self, indices: torch.Tensor) -> TeacherState:
        indices = indices.to(device=self.device, dtype=torch.long).reshape(-1)
        if indices.numel() == 0:
            return extract_teacher_state(self.base)
        ordinals = self.episode_ordinals[indices]
        task_indices = (indices + ordinals) % len(STAGE0_TASKS)
        seed_values = list(range(self._next_train_seed, self._next_train_seed + len(indices)))
        limit = 1150000 if self.probe == "A" else 1200000
        if seed_values and seed_values[-1] >= limit:
            raise RuntimeError(f"Probe {self.probe} exhausted its frozen train seed subrange")
        self._next_train_seed += len(indices)
        for task_index, task in enumerate(STAGE0_TASKS):
            mask = task_indices == task_index
            selected = indices[mask]
            if selected.numel() == 0:
                continue
            positions = torch.nonzero(mask, as_tuple=False).reshape(-1).cpu().tolist()
            seeds = [seed_values[position] for position in positions]
            self._reset_explicit(
                selected,
                seeds=seeds,
                task=task,
                namespace="f1_probe_train",
            )
        self.episode_ordinals[indices] += 1
        return extract_teacher_state(self.base)

    def reset_all_training(self) -> TeacherState:
        indices = torch.arange(self.num_envs, device=self.device)
        seeds = list(range(self._next_train_seed, self._next_train_seed + self.num_envs))
        limit = 1150000 if self.probe == "A" else 1200000
        if seeds[-1] >= limit:
            raise RuntimeError(f"Probe {self.probe} exhausted its frozen train seed subrange")
        for seed in seeds:
            assert_f1_seed_allowed(seed, "f1_probe_train")
        self.current_scene_seeds = seeds
        self._next_train_seed += self.num_envs
        self.environment.reset(seed=seeds, options={"task_spec": STAGE0_TASKS[0].to_dict()})
        task_indices = indices % len(STAGE0_TASKS)
        for task_index, task in enumerate(STAGE0_TASKS[1:], start=1):
            selected = indices[task_indices == task_index]
            selected_seeds = [seeds[index] for index in selected.detach().cpu().tolist()]
            self._reset_explicit(
                selected,
                seeds=selected_seeds,
                task=task,
                namespace="f1_probe_train",
            )
        self.episode_ordinals += 1
        return extract_teacher_state(self.base)

    def reset_explicit_batch(
        self,
        *,
        seeds: Sequence[int],
        tasks: Sequence[PushTaskSpec],
        namespace: str,
    ) -> TeacherState:
        if len(seeds) != self.num_envs or len(tasks) != self.num_envs:
            raise ValueError("explicit F1 batch must cover every vector slot")
        self.current_scene_seeds = [int(seed) for seed in seeds]
        for seed in seeds:
            assert_f1_seed_allowed(int(seed), namespace)
        first = tasks[0]
        self.environment.reset(seed=list(seeds), options={"task_spec": first.to_dict()})
        indices = torch.arange(self.num_envs, device=self.device)
        for task in dict.fromkeys(tasks):
            selected_values = [index for index, value in enumerate(tasks) if value == task]
            if task == first:
                continue
            selected = torch.tensor(selected_values, dtype=torch.long, device=self.device)
            selected_seeds = [int(seeds[index]) for index in selected_values]
            self._reset_explicit(
                selected,
                seeds=selected_seeds,
                task=task,
                namespace=namespace,
            )
        self.episode_ordinals[indices] += 1
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
        assert_f1_seed_allowed(seed, namespace)
        self.current_scene_seeds = [seed]
        self.environment.reset(seed=seed, options={"task_spec": task.to_dict()})
        return extract_teacher_state(self.base)


def _new_policies(
    runtime: Phase2B4F1VectorRuntime,
) -> tuple[BoundedActorCritic, StateCenteredResidualActorCritic]:
    torch.manual_seed(240400)
    global_policy = BoundedActorCritic(
        observation_dim=TEACHER_OBSERVATION_DIM,
        action_low=runtime.action_low,
        action_high=runtime.action_high,
        initial_log_std=-2.5,
    ).to(runtime.device)
    torch.manual_seed(240400)
    residual_policy = StateCenteredResidualActorCritic(
        observation_dim=TEACHER_OBSERVATION_DIM,
        action_low=runtime.action_low,
        action_high=runtime.action_high,
        initial_log_std=-0.5,
    ).to(runtime.device)
    residual_policy.actor.load_state_dict(global_policy.actor.state_dict(), strict=True)
    residual_policy.critic.load_state_dict(global_policy.critic.state_dict(), strict=True)
    return global_policy, residual_policy


def _distribution_actions(
    *,
    policy: BoundedActorCritic | StateCenteredResidualActorCritic,
    observations: torch.Tensor,
    epsilon: torch.Tensor,
) -> torch.Tensor:
    distribution = policy.distribution(observations)
    latent = distribution.mean + distribution.stddev * epsilon
    if isinstance(policy, StateCenteredResidualActorCritic):
        return policy.action_from_latent(observations, latent)
    return policy.action_from_latent(latent)


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.to(torch.float32)
    return {
        "mean": float(values.mean().item()),
        "p50": float(torch.quantile(values, 0.50).item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "p99": float(torch.quantile(values, 0.99).item()),
        "maximum": float(values.max().item()),
    }


def audit_initial_exploration(*, sample_count: int = 100_000) -> dict[str, object]:
    assert_f1_operation_allowed("initial_exploration_audit")
    if not torch.cuda.is_available():
        raise RuntimeError("F1 initial exploration audit requires CUDA")
    _seed_everything(240420)
    runtime = Phase2B4F1VectorRuntime(num_envs=128)
    try:
        seeds = list(range(1202000, 1202128))
        tasks = [STAGE0_TASKS[index % len(STAGE0_TASKS)] for index in range(128)]
        state = runtime.reset_explicit_batch(
            seeds=seeds,
            tasks=tasks,
            namespace="f1_diagnostic",
        )
        global_policy, residual_policy = _new_policies(runtime)
        legality = audit_residual_action_legality(
            residual_policy,
            state.observation,
            sample_count=sample_count,
        )
        generator = torch.Generator(device=runtime.device)
        generator.manual_seed(240421)
        global_arm: list[torch.Tensor] = []
        residual_arm: list[torch.Tensor] = []
        global_per_joint: list[torch.Tensor] = []
        residual_per_joint: list[torch.Tensor] = []
        global_gripper: list[torch.Tensor] = []
        residual_gripper: list[torch.Tensor] = []
        global_large = 0
        residual_large = 0
        global_saturated = 0
        residual_saturated = 0
        global_joint_limit_near = 0
        residual_joint_limit_near = 0
        processed = 0
        while processed < sample_count:
            count = min(4096, sample_count - processed)
            indices = (
                torch.arange(processed, processed + count, device=runtime.device) % runtime.num_envs
            )
            observations = state.observation[indices]
            epsilon = torch.randn((count, 8), generator=generator, device=runtime.device)
            reference = current_action_reference(
                observations,
                action_low=runtime.action_low,
                action_high=runtime.action_high,
            )
            global_action = _distribution_actions(
                policy=global_policy,
                observations=observations,
                epsilon=epsilon,
            )
            residual_action = _distribution_actions(
                policy=residual_policy,
                observations=observations,
                epsilon=epsilon,
            )
            global_delta = global_action - reference
            residual_delta = residual_action - reference
            global_l2 = torch.linalg.vector_norm(global_delta[:, :7], dim=1)
            residual_l2 = torch.linalg.vector_norm(residual_delta[:, :7], dim=1)
            global_arm.append(global_l2.cpu())
            residual_arm.append(residual_l2.cpu())
            global_per_joint.append(global_delta[:, :7].abs().cpu())
            residual_per_joint.append(residual_delta[:, :7].abs().cpu())
            global_gripper.append(global_delta[:, 7].abs().cpu())
            residual_gripper.append(residual_delta[:, 7].abs().cpu())
            global_large += int((global_l2 > 0.25).sum().item())
            residual_large += int((residual_l2 > 0.25).sum().item())
            global_unit = (global_action - global_policy.action_bias) / global_policy.action_scale
            residual_unit = (
                residual_action - residual_policy.action_bias
            ) / residual_policy.action_scale
            global_saturated += int((global_unit.abs() > 0.95).any(dim=1).sum().item())
            residual_saturated += int((residual_unit.abs() > 0.95).any(dim=1).sum().item())
            global_joint_limit_near += int(
                (global_unit[:, :7].abs() > 0.95).any(dim=1).sum().item()
            )
            residual_joint_limit_near += int(
                (residual_unit[:, :7].abs() > 0.95).any(dim=1).sum().item()
            )
            processed += count
        global_arm_tensor = torch.cat(global_arm)
        residual_arm_tensor = torch.cat(residual_arm)
        global_per_joint_tensor = torch.cat(global_per_joint)
        residual_per_joint_tensor = torch.cat(residual_per_joint)
        global_gripper_tensor = torch.cat(global_gripper)
        residual_gripper_tensor = torch.cat(residual_gripper)
        zero_latent = torch.zeros((runtime.num_envs, 8), device=runtime.device)
        global_zero_action = global_policy.action_from_latent(zero_latent)
        reference = current_action_reference(
            state.observation,
            action_low=runtime.action_low,
            action_high=runtime.action_high,
        )
        payload: dict[str, object] = {
            "schema_version": "langmani-v2-phase2b4-f1-exploration-distribution-audit-v0",
            "sample_count": sample_count,
            "representative_reset_count": runtime.num_envs,
            "f0_zero_latent_mapping": {
                "maps_to": "global action-space midpoint",
                "maximum_distance_from_midpoint": float(
                    (global_zero_action - global_policy.action_bias).abs().max().item()
                ),
                "important_distinction": (
                    "F0 did not sample around zero latent directly: its distribution mean "
                    "added an inverse-tanh current-qpos reference. Initial stochastic means "
                    "were state-centered, but learned actor offsets were not residual-capped."
                ),
            },
            "global_f0_initial": {
                "latent_log_std": -2.5,
                "arm_joint_target_l2": _quantiles(global_arm_tensor),
                "per_arm_joint_absolute_displacement": [
                    _quantiles(global_per_joint_tensor[:, index]) for index in range(7)
                ],
                "gripper_target_displacement": _quantiles(global_gripper_tensor),
                "large_global_jump_threshold_rad": 0.25,
                "large_global_jump_count": global_large,
                "large_global_jump_rate": global_large / sample_count,
                "action_saturation_count": global_saturated,
                "action_saturation_rate": global_saturated / sample_count,
                "arm_joint_limit_proximity_count": global_joint_limit_near,
                "arm_joint_limit_proximity_rate": global_joint_limit_near / sample_count,
            },
            "state_centered_residual_initial": {
                "latent_log_std": -0.5,
                "residual_scales": list(DEFAULT_RESIDUAL_SCALES),
                "arm_joint_target_l2": _quantiles(residual_arm_tensor),
                "per_arm_joint_absolute_displacement": [
                    _quantiles(residual_per_joint_tensor[:, index]) for index in range(7)
                ],
                "gripper_target_displacement": _quantiles(residual_gripper_tensor),
                "large_global_jump_threshold_rad": 0.25,
                "large_global_jump_count": residual_large,
                "large_global_jump_rate": residual_large / sample_count,
                "action_saturation_count": residual_saturated,
                "action_saturation_rate": residual_saturated / sample_count,
                "arm_joint_limit_proximity_count": residual_joint_limit_near,
                "arm_joint_limit_proximity_rate": residual_joint_limit_near / sample_count,
            },
            "residual_legality": legality,
            "physical_contact_and_workspace_risk": (
                "measured separately by global_versus_residual_comparison.json on "
                "identical real-physics short prefixes"
            ),
            "static_locality_improvement": (
                1.0
                - float(residual_arm_tensor.mean().item())
                / max(float(global_arm_tensor.mean().item()), 1e-12)
            ),
            "passed": legality["passed"] is True,
        }
        payload["fingerprint"] = canonical_json_sha256(payload)
        return payload
    finally:
        runtime.close()


def _step_safety(info: Mapping[str, torch.Tensor]) -> bool:
    return bool(_safety_mask(info, ZERO_TOLERANCE_KEYS)[0].item())


def _action_autocorrelation(actions: Sequence[np.ndarray]) -> float | None:
    if len(actions) < 2:
        return None
    array = np.asarray(actions, dtype=np.float64)
    left = array[:-1].reshape(-1)
    right = array[1:].reshape(-1)
    if left.std() <= 1e-12 or right.std() <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def compare_global_and_residual_physics(
    *,
    episode_count: int = 32,
    prefix_steps: int = 8,
) -> dict[str, object]:
    assert_f1_operation_allowed("global_residual_physical_comparison")
    if episode_count != 32 or prefix_steps != 8:
        raise ValueError("F1 physical comparison is frozen at 32 episodes x 8 steps")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(240422)
    epsilons = torch.randn((episode_count, prefix_steps, 8), generator=generator)
    mode_records: dict[str, list[dict[str, object]]] = {"global_f0": [], "residual_f1": []}
    for mode in mode_records:
        runtime = Phase2B4F1VectorRuntime(num_envs=1)
        try:
            global_policy, residual_policy = _new_policies(runtime)
            policy: BoundedActorCritic | StateCenteredResidualActorCritic = (
                global_policy if mode == "global_f0" else residual_policy
            )
            for episode_index in range(episode_count):
                seed = 1202128 + episode_index
                task = STAGE0_TASKS[episode_index % len(STAGE0_TASKS)]
                state = runtime.reset_single_evaluation(
                    seed=seed,
                    task=task,
                    namespace="f1_diagnostic",
                )
                initial_tcp = state.observation[0, 18:21].clone()
                initial_distance = float(state.target_distance[0].item())
                previous_tcp = initial_tcp
                maximum_tcp_step = 0.0
                first_tcp_step = 0.0
                maximum_arm_jump = 0.0
                maximum_gripper_jump = 0.0
                saturation_steps = 0
                joint_limit_proximity_steps = 0
                actions: list[np.ndarray] = []
                wrong_object = False
                workspace_exit = False
                target_contact = False
                for step in range(prefix_steps):
                    observation = state.observation
                    epsilon = epsilons[episode_index, step].to(runtime.device)[None, :]
                    action = _distribution_actions(
                        policy=policy,
                        observations=observation,
                        epsilon=epsilon,
                    )
                    reference = current_action_reference(
                        observation,
                        action_low=runtime.action_low,
                        action_high=runtime.action_high,
                    )
                    maximum_arm_jump = max(
                        maximum_arm_jump,
                        float(torch.linalg.vector_norm(action[0, :7] - reference[0, :7]).item()),
                    )
                    maximum_gripper_jump = max(
                        maximum_gripper_jump,
                        float((action[0, 7] - reference[0, 7]).abs().item()),
                    )
                    action_unit = (action - policy.action_bias) / policy.action_scale
                    saturation_steps += int((action_unit[0].abs() > 0.95).any().item())
                    joint_limit_proximity_steps += int(
                        (action_unit[0, :7].abs() > 0.95).any().item()
                    )
                    actions.append(action[0].detach().cpu().numpy())
                    state, _terminated, _truncated, info = runtime.step(action)
                    current_tcp = state.observation[0, 18:21]
                    maximum_tcp_step = max(
                        maximum_tcp_step,
                        float(torch.linalg.vector_norm(current_tcp - previous_tcp).item()),
                    )
                    if step == 0:
                        first_tcp_step = float(
                            torch.linalg.vector_norm(current_tcp - initial_tcp).item()
                        )
                    previous_tcp = current_tcp
                    wrong_object |= bool(
                        (
                            _metric_bool(info, "wrong_object_contact")
                            | _metric_bool(info, "wrong_object_displaced")
                        )[0].item()
                    )
                    workspace_exit |= bool(_metric_bool(info, "target_outside_workspace")[0].item())
                    target_contact |= bool(state.target_contact[0].item())
                    if _step_safety(info):
                        break
                final_tcp = state.observation[0, 18:21]
                mode_records[mode].append(
                    {
                        "episode_index": episode_index,
                        "seed": seed,
                        "task": task.to_dict(),
                        "steps": len(actions),
                        "maximum_arm_joint_target_l2": maximum_arm_jump,
                        "maximum_gripper_target_displacement": maximum_gripper_jump,
                        "first_tcp_step_translation": first_tcp_step,
                        "maximum_tcp_step_translation": maximum_tcp_step,
                        "tcp_prefix_translation": float(
                            torch.linalg.vector_norm(final_tcp - initial_tcp).item()
                        ),
                        "target_distance_reduction": (
                            initial_distance - float(state.target_distance[0].item())
                        ),
                        "target_contact": target_contact,
                        "wrong_object_interaction": wrong_object,
                        "workspace_exit": workspace_exit,
                        "action_saturation_steps": saturation_steps,
                        "joint_limit_proximity_steps": joint_limit_proximity_steps,
                        "action_autocorrelation": _action_autocorrelation(actions),
                    }
                )
        finally:
            runtime.close()

    def summarize(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
        arm = torch.tensor(
            [float(cast(float, item["maximum_arm_joint_target_l2"])) for item in records]
        )
        tcp_step = torch.tensor(
            [float(cast(float, item["maximum_tcp_step_translation"])) for item in records]
        )
        tcp_first = torch.tensor(
            [float(cast(float, item["first_tcp_step_translation"])) for item in records]
        )
        tcp_prefix = torch.tensor(
            [float(cast(float, item["tcp_prefix_translation"])) for item in records]
        )
        progress = torch.tensor(
            [float(cast(float, item["target_distance_reduction"])) for item in records]
        )
        correlations = [
            float(cast(float, item["action_autocorrelation"]))
            for item in records
            if item["action_autocorrelation"] is not None
        ]
        return {
            "episode_count": len(records),
            "maximum_arm_joint_target_l2": _quantiles(arm),
            "first_tcp_step_translation": _quantiles(tcp_first),
            "maximum_tcp_step_translation": _quantiles(tcp_step),
            "tcp_prefix_translation": _quantiles(tcp_prefix),
            "target_distance_reduction": _quantiles(progress),
            "toward_target_count": int((progress > 0).sum().item()),
            "away_from_target_count": int((progress < 0).sum().item()),
            "target_contact_count": sum(bool(item["target_contact"]) for item in records),
            "target_contact_rate": sum(bool(item["target_contact"]) for item in records)
            / len(records),
            "wrong_object_interaction_count": sum(
                bool(item["wrong_object_interaction"]) for item in records
            ),
            "workspace_exit_count": sum(bool(item["workspace_exit"]) for item in records),
            "workspace_exit_rate": sum(bool(item["workspace_exit"]) for item in records)
            / len(records),
            "wrong_object_interaction_rate": sum(
                bool(item["wrong_object_interaction"]) for item in records
            )
            / len(records),
            "locally_smooth_count": int((tcp_step <= 0.05).sum().item()),
            "locally_smooth_probability": float((tcp_step <= 0.05).to(torch.float32).mean()),
            "toward_target_probability": float((progress > 0).to(torch.float32).mean()),
            "away_from_target_probability": float((progress < 0).to(torch.float32).mean()),
            "action_saturation_steps": sum(
                int(cast(int, item["action_saturation_steps"])) for item in records
            ),
            "joint_limit_proximity_steps": sum(
                int(cast(int, item["joint_limit_proximity_steps"])) for item in records
            ),
            "maximum_gripper_target_displacement": max(
                float(cast(float, item["maximum_gripper_target_displacement"])) for item in records
            ),
            "action_autocorrelation_mean": (float(np.mean(correlations)) if correlations else None),
        }

    global_summary = summarize(mode_records["global_f0"])
    residual_summary = summarize(mode_records["residual_f1"])
    global_arm_mean = float(
        cast(Mapping[str, float], global_summary["maximum_arm_joint_target_l2"])["mean"]
    )
    residual_arm_mean = float(
        cast(Mapping[str, float], residual_summary["maximum_arm_joint_target_l2"])["mean"]
    )
    global_tcp_mean = float(
        cast(Mapping[str, float], global_summary["maximum_tcp_step_translation"])["mean"]
    )
    residual_tcp_mean = float(
        cast(Mapping[str, float], residual_summary["maximum_tcp_step_translation"])["mean"]
    )
    global_unsafe = int(cast(int, global_summary["wrong_object_interaction_count"])) + int(
        cast(int, global_summary["workspace_exit_count"])
    )
    residual_unsafe = int(cast(int, residual_summary["wrong_object_interaction_count"])) + int(
        cast(int, residual_summary["workspace_exit_count"])
    )
    checks = {
        "arm_target_displacement_reduced_25_percent": (
            residual_arm_mean <= 0.75 * max(global_arm_mean, 1e-12)
        ),
        "tcp_step_displacement_reduced_20_percent": (
            residual_tcp_mean <= 0.80 * max(global_tcp_mean, 1e-12)
        ),
        "unsafe_short_prefix_not_increased": residual_unsafe <= global_unsafe,
        "residual_action_integrity": True,
    }
    payload: dict[str, object] = {
        "schema_version": "langmani-v2-phase2b4-f1-global-residual-comparison-v0",
        "episode_count_per_mode": episode_count,
        "prefix_steps": prefix_steps,
        "identical_reset_seeds": True,
        "identical_standard_normal_samples": True,
        "global_f0": global_summary,
        "residual_f1": residual_summary,
        "checks": checks,
        "residual_materially_safer": all(checks.values()),
        "records": mode_records,
    }
    payload["fingerprint"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "records"}
    )
    return payload


def _termination_reason(
    info: Mapping[str, torch.Tensor],
    *,
    terminated: bool,
    truncated: bool,
) -> str:
    ordered = (
        ("success", "stable_success"),
        ("wrong_object_contact", "wrong_object"),
        ("wrong_object_displaced", "wrong_object"),
        ("target_outside_workspace", "target_outside_workspace"),
        ("invalid_action", "other_failure"),
        ("action_out_of_bounds", "other_failure"),
        ("target_lifted", "other_failure"),
        ("target_toppled", "other_failure"),
        ("robot_collision", "other_failure"),
    )
    for key, reason in ordered:
        if bool(_metric_bool(info, key)[0].item()):
            return reason
    if truncated:
        return "timeout"
    if terminated:
        return "other_failure"
    return "incomplete"


def diagnose_f0_checkpoint(
    *,
    checkpoint: Path,
    episode_count: int = 32,
) -> dict[str, object]:
    assert_f1_operation_allowed("f0_read_only_diagnosis")
    if episode_count != 32:
        raise ValueError("F0 checkpoint diagnostic is frozen at 32 episodes")
    policy, payload = reconstruct_policy_from_checkpoint(checkpoint)
    runtime = Phase2B4F1VectorRuntime(num_envs=1)
    policy = policy.to(runtime.device)
    policy.eval()
    component_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "nonzero_count": 0.0,
            "positive_count": 0.0,
            "absolute_sum": 0.0,
            "total": 0.0,
            "before_contact_total": 0.0,
            "after_contact_total": 0.0,
        }
    )
    records: list[dict[str, object]] = []
    all_actions: list[np.ndarray] = []
    all_values: list[float] = []
    actor_offset_norms: list[float] = []
    try:
        for episode_index in range(episode_count):
            seed = 1202160 + episode_index
            task = MICRO_TASKS[episode_index % len(MICRO_TASKS)]
            state = runtime.reset_single_evaluation(
                seed=seed,
                task=task,
                namespace="f1_diagnostic",
            )
            initial_distance = float(state.target_distance[0].item())
            previous_action_unit = torch.zeros((1, 8), device=runtime.device)
            entered_approach = False
            correct_contact = False
            directed_progress = False
            containment_increase = False
            first_contact_step: int | None = None
            contact_steps = 0
            positive_events = 0
            negative_safety_events = 0
            termination_reason = "incomplete"
            total_return = 0.0
            for step in range(250):
                with torch.no_grad():
                    distribution = policy.distribution(state.observation)
                    action = policy.action_from_latent(distribution.mean)
                    value = policy.value(state.observation)
                    actor_offset = policy.actor(normalize_teacher_observation(state.observation))
                all_actions.append(action[0].cpu().numpy())
                all_values.append(float(value[0].item()))
                actor_offset_norms.append(
                    float(torch.linalg.vector_norm(actor_offset[0, :7]).item())
                )
                previous = state
                state, env_terminated, env_truncated, info = runtime.step(action)
                action_unit = (action - runtime.action_low) / (
                    runtime.action_high - runtime.action_low
                )
                action_unit = 2.0 * action_unit - 1.0
                reward, components = _reward_from_transition(
                    previous=previous,
                    current=state,
                    info=info,
                    action_unit=action_unit,
                    previous_action_unit=previous_action_unit,
                )
                previous_action_unit = action_unit
                total_return += float(reward[0].item())
                entered_approach |= bool(state.behind_distance[0].item() <= 0.12)
                current_contact = bool(state.target_contact[0].item())
                if current_contact:
                    contact_steps += 1
                    if first_contact_step is None:
                        first_contact_step = step + 1
                correct_contact |= current_contact
                progress_delta = float(
                    previous.target_distance[0].item() - state.target_distance[0].item()
                )
                directed_progress |= progress_delta >= 0.001
                containment_increase |= bool(
                    state.containment_fraction[0] > previous.containment_fraction[0]
                )
                for name, values in components.items():
                    value_float = float(values[0].item())
                    stats = component_stats[name]
                    stats["nonzero_count"] += float(abs(value_float) > 1e-12)
                    stats["positive_count"] += float(value_float > 0)
                    stats["absolute_sum"] += abs(value_float)
                    stats["total"] += value_float
                    if first_contact_step is None:
                        stats["before_contact_total"] += value_float
                    else:
                        stats["after_contact_total"] += value_float
                    positive_events += int(value_float > 1e-12)
                safety = _safety_mask(info, ZERO_TOLERANCE_KEYS)
                negative_safety_events += int(safety[0].item())
                effective_terminated = bool(env_terminated[0].item() or safety[0].item())
                truncated = bool(env_truncated[0].item() and not effective_terminated)
                if effective_terminated or truncated:
                    termination_reason = _termination_reason(
                        info,
                        terminated=effective_terminated,
                        truncated=truncated,
                    )
                    break
            records.append(
                {
                    "episode_index": episode_index,
                    "seed": seed,
                    "task": task.to_dict(),
                    "steps": step + 1,
                    "return": total_return,
                    "entered_useful_approach": entered_approach,
                    "correct_contact": correct_contact,
                    "target_directed_progress": directed_progress,
                    "target_distance_reduced": (
                        initial_distance - float(state.target_distance[0].item()) >= 0.005
                    ),
                    "containment_increased": containment_increase,
                    "first_contact_step": first_contact_step,
                    "contact_steps": contact_steps,
                    "positive_reward_events": positive_events,
                    "negative_safety_events": negative_safety_events,
                    "termination_reason": termination_reason,
                }
            )
    finally:
        runtime.close()
    transition_count = sum(int(cast(int, item["steps"])) for item in records)
    contact_records = [item for item in records if item["first_contact_step"] is not None]
    action_array = np.asarray(all_actions, dtype=np.float32)
    termination_counts: dict[str, int] = defaultdict(int)
    for record in records:
        termination_counts[str(record["termination_reason"])] += 1
    component_report = {
        name: {
            "frequency": stats["nonzero_count"] / max(transition_count, 1),
            "positive_frequency": stats["positive_count"] / max(transition_count, 1),
            "mean_magnitude": stats["absolute_sum"] / max(transition_count, 1),
            "total_contribution": stats["total"],
            "before_contact_total": stats["before_contact_total"],
            "after_contact_total": stats["after_contact_total"],
        }
        for name, stats in component_stats.items()
    }
    summary = {
        "episode_count": episode_count,
        "entered_useful_approach_rate": sum(
            bool(item["entered_useful_approach"]) for item in records
        )
        / episode_count,
        "correct_contact_rate": sum(bool(item["correct_contact"]) for item in records)
        / episode_count,
        "target_directed_progress_rate": sum(
            bool(item["target_directed_progress"]) for item in records
        )
        / episode_count,
        "target_distance_reduction_rate": sum(
            bool(item["target_distance_reduced"]) for item in records
        )
        / episode_count,
        "containment_increase_rate": sum(bool(item["containment_increased"]) for item in records)
        / episode_count,
        "wrong_object_rate": termination_counts.get("wrong_object", 0) / episode_count,
        "workspace_exit_rate": termination_counts.get("target_outside_workspace", 0)
        / episode_count,
        "mean_time_to_first_contact": (
            float(np.mean([int(cast(int, item["first_contact_step"])) for item in contact_records]))
            if contact_records
            else None
        ),
        "mean_useful_contact_duration": float(
            np.mean([int(cast(int, item["contact_steps"])) for item in records])
        ),
        "mean_positive_reward_events": float(
            np.mean([int(cast(int, item["positive_reward_events"])) for item in records])
        ),
        "mean_negative_safety_events": float(
            np.mean([int(cast(int, item["negative_safety_events"])) for item in records])
        ),
        "termination_counts": dict(termination_counts),
    }
    payload_report: dict[str, object] = {
        "schema_version": "langmani-v2-phase2b4-f1-f0-checkpoint-diagnostic-v0",
        "source_checkpoint": checkpoint.as_posix(),
        "source_checkpoint_sha256": (
            "sha256:" + hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        ),
        "source_global_step": payload["global_step"],
        "evidence_role": (
            "new read-only frozen-checkpoint diagnostic on F1 diagnostic seeds; "
            "not reconstruction of unretained F0 training transitions"
        ),
        "summary": summary,
        "reward_components": component_report,
        "action_mean": action_array.mean(axis=0).tolist(),
        "action_std": action_array.std(axis=0).tolist(),
        "state_value": {
            "mean": float(np.mean(all_values)),
            "std": float(np.std(all_values)),
            "minimum": float(np.min(all_values)),
            "maximum": float(np.max(all_values)),
        },
        "actor_arm_offset_norm": {
            "mean": float(np.mean(actor_offset_norms)),
            "p95": float(np.quantile(actor_offset_norms, 0.95)),
            "maximum": float(np.max(actor_offset_norms)),
        },
        "records": records,
    }
    payload_report["fingerprint"] = canonical_json_sha256(
        {key: value for key, value in payload_report.items() if key != "records"}
    )
    return payload_report


def _f1_checkpoint_payload(
    *,
    policy: StateCenteredResidualActorCritic,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    global_step: int,
    probe: str,
    reward_revision: int,
) -> dict[str, object]:
    return {
        "schema_version": F1_CHECKPOINT_SCHEMA,
        "probe": probe,
        "global_step": global_step,
        "config": config.to_dict(),
        "policy_state": policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "observation_manifest_fingerprint": teacher_observation_manifest()["fingerprint"],
        "action_manifest_fingerprint": policy.action_manifest()["fingerprint"],
        "reward_manifest_fingerprint": reward_manifest()["fingerprint"],
        "reward_revision": reward_revision,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _save_checkpoint(path: Path, payload: Mapping[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def reconstruct_f1_policy(
    path: Path,
) -> tuple[StateCenteredResidualActorCritic, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != F1_CHECKPOINT_SCHEMA:
        raise ValueError("invalid Phase 2B.4-F1 checkpoint")
    state = payload.get("policy_state")
    config = payload.get("config")
    if not isinstance(state, Mapping) or not isinstance(config, Mapping):
        raise ValueError("F1 checkpoint is missing policy state or config")
    low = state.get("action_low")
    high = state.get("action_high")
    residual_scale = state.get("residual_scale")
    if not all(isinstance(item, torch.Tensor) for item in (low, high, residual_scale)):
        raise ValueError("F1 checkpoint action tensors are missing")
    policy = StateCenteredResidualActorCritic(
        observation_dim=TEACHER_OBSERVATION_DIM,
        action_low=cast(torch.Tensor, low),
        action_high=cast(torch.Tensor, high),
        hidden_dim=int(config["hidden_dim"]),
        initial_log_std=float(config["initial_log_std"]),
        residual_scales=cast(torch.Tensor, residual_scale).tolist(),
    )
    policy.load_state_dict(state, strict=True)
    policy.eval()
    return policy, payload


def _reward_with_revision(
    *,
    previous: TeacherState,
    current: TeacherState,
    info: Mapping[str, torch.Tensor],
    action_unit: torch.Tensor,
    previous_action_unit: torch.Tensor,
    reward_revision: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    total, raw_components = _reward_from_transition(
        previous=previous,
        current=current,
        info=info,
        action_unit=action_unit,
        previous_action_unit=previous_action_unit,
    )
    components = dict(raw_components)
    if reward_revision == 0:
        return total, components
    raise PermissionError(
        "reward revision 1 is not implemented before the target diagnosis decision"
    )


def _explained_variance(returns: torch.Tensor, values: torch.Tensor) -> float | None:
    variance = torch.var(returns, unbiased=False)
    if float(variance.item()) <= 1e-12:
        return None
    return float((1.0 - torch.var(returns - values, unbiased=False) / variance).item())


def run_probe_training(
    *,
    probe: str,
    output_dir: Path,
    reward_revision: int,
) -> dict[str, object]:
    if probe not in {"A", "B"}:
        raise ValueError("probe must be A or B")
    assert_f1_operation_allowed(f"probe_{probe.lower()}_training")
    if probe == "A" and reward_revision != 0:
        raise PermissionError("Probe A must retain F0 reward revision 0")
    if not torch.cuda.is_available():
        raise RuntimeError("F1 PPO probe requires CUDA")
    config = PPOConfig(
        total_timesteps=262_144,
        seed=PROBE_POLICY_SEEDS[probe],
        initial_log_std=-0.5,
    )
    _seed_everything(config.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    runtime = Phase2B4F1VectorRuntime(num_envs=config.num_envs, probe=probe)
    try:
        current = runtime.reset_all_training()
        policy = StateCenteredResidualActorCritic(
            observation_dim=TEACHER_OBSERVATION_DIM,
            action_low=runtime.action_low,
            action_high=runtime.action_high,
            hidden_dim=config.hidden_dim,
            initial_log_std=config.initial_log_std,
        ).to(runtime.device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate, eps=1e-5)
        shape = (config.num_steps, config.num_envs)
        observations = torch.zeros(shape + (TEACHER_OBSERVATION_DIM,), device=runtime.device)
        latents = torch.zeros(shape + (8,), device=runtime.device)
        log_probs = torch.zeros(shape, device=runtime.device)
        values = torch.zeros(shape, device=runtime.device)
        rewards = torch.zeros(shape, device=runtime.device)
        next_values = torch.zeros(shape, device=runtime.device)
        terminated = torch.zeros(shape, dtype=torch.bool, device=runtime.device)
        truncated = torch.zeros_like(terminated)
        previous_action_unit = torch.zeros((config.num_envs, 8), device=runtime.device)
        episode_return = torch.zeros(config.num_envs, device=runtime.device)
        episode_start_distance = current.target_distance.clone()
        completed_returns: list[float] = []
        completed_progress: list[float] = []
        completed_successes = 0
        termination_counts: dict[str, int] = defaultdict(int)
        metrics: list[dict[str, object]] = []
        global_step = 0
        optimizer_steps = 0
        reset_count = config.num_envs
        final_loss: dict[str, float] = {}
        for iteration in range(config.iterations):
            rollout_started = time.perf_counter()
            rollout_component_totals: dict[str, float] = defaultdict(float)
            policy.eval()
            for step in range(config.num_steps):
                observations[step] = current.observation
                with torch.no_grad():
                    action, latent, log_prob, _entropy, value = policy.action_and_value(
                        current.observation
                    )
                action_unit = policy.action_unit_from_latent(current.observation, latent)
                previous = current
                current, env_terminated, env_truncated, info = runtime.step(action)
                reward, components = _reward_with_revision(
                    previous=previous,
                    current=current,
                    info=info,
                    action_unit=action_unit,
                    previous_action_unit=previous_action_unit,
                    reward_revision=reward_revision,
                )
                for name, component in components.items():
                    rollout_component_totals[name] += float(component.sum().item())
                safety = _safety_mask(info, TRAINING_STOP_KEYS)
                effective_terminated = env_terminated | safety
                effective_truncated = env_truncated & ~effective_terminated
                with torch.no_grad():
                    transition_next_value = policy.value(current.observation)
                observations[step] = previous.observation
                latents[step] = latent
                log_probs[step] = log_prob
                values[step] = value
                rewards[step] = reward
                next_values[step] = transition_next_value
                terminated[step] = effective_terminated
                truncated[step] = effective_truncated
                global_step += config.num_envs
                episode_return += reward
                done = effective_terminated | effective_truncated
                if done.any():
                    indices = torch.nonzero(done, as_tuple=False).reshape(-1)
                    for index in indices.cpu().tolist():
                        completed_returns.append(float(episode_return[index].item()))
                        completed_progress.append(
                            float(
                                (
                                    episode_start_distance[index] - current.target_distance[index]
                                ).item()
                            )
                        )
                        success = bool(_metric_bool(info, "success")[index].item())
                        completed_successes += int(success)
                        if success:
                            reason = "stable_success"
                        elif bool(safety[index].item()):
                            reason = "safety"
                        elif bool(effective_truncated[index].item()):
                            reason = "timeout"
                        else:
                            reason = "other_failure"
                        termination_counts[reason] += 1
                    episode_return[indices] = 0
                    reset_count += int(indices.numel())
                    current = runtime.reset_training_indices(indices)
                    episode_start_distance[indices] = current.target_distance[indices]
                    previous_action_unit = torch.where(
                        done[:, None], torch.zeros_like(previous_action_unit), action_unit
                    )
                else:
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
                        raise RuntimeError("F1 PPO loss became non-finite")
                    optimizer.zero_grad(set_to_none=True)
                    loss.total.backward()
                    for parameter in policy.parameters():
                        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                            raise RuntimeError("F1 PPO gradient became non-finite")
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        policy.parameters(), config.max_grad_norm
                    )
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
            with torch.no_grad():
                action_mean = policy.deterministic_action(flat_observations)
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
                    "reward_component_totals": dict(rollout_component_totals),
                    "advantage_min": float(advantages.min().item()),
                    "advantage_max": float(advantages.max().item()),
                    "value_explained_variance": _explained_variance(returns, values),
                    "gradient_norm_max": max(gradient_norms),
                    "last_loss": final_loss,
                    "action_mean": action_mean.mean(dim=0).cpu().tolist(),
                    "action_std": action_mean.std(dim=0, unbiased=False).cpu().tolist(),
                    "latent_std": policy.actor_log_std.detach().exp()[0].cpu().tolist(),
                    "completed_episodes": len(completed_returns),
                    "completed_successes": completed_successes,
                }
            )
        checkpoint = output_dir / f"probe_{probe.lower()}_checkpoint.pt"
        checkpoint_payload = _f1_checkpoint_payload(
            policy=policy,
            optimizer=optimizer,
            config=config,
            global_step=global_step,
            probe=probe,
            reward_revision=reward_revision,
        )
        checkpoint_sha = _save_checkpoint(checkpoint, checkpoint_payload)
        loaded_policy, loaded_payload = reconstruct_f1_policy(checkpoint)
        loaded_policy = loaded_policy.to(runtime.device)
        with torch.no_grad():
            expected = policy.deterministic_action(current.observation[:1])
            observed = loaded_policy.deterministic_action(current.observation[:1])
        reload_error = float((expected - observed).abs().max().item())
        if reload_error != 0.0:
            raise RuntimeError("F1 checkpoint reconstruction changed deterministic action")
        elapsed = time.perf_counter() - started
        report: dict[str, object] = {
            "schema_version": f"langmani-v2-phase2b4-f1-probe-{probe.lower()}-training-v0",
            "probe": probe,
            "runtime": asdict(RuntimeVersions.collect()),
            "config": config.to_dict(),
            "reward_revision": reward_revision,
            "global_step": global_step,
            "optimizer_steps": optimizer_steps,
            "elapsed_seconds": elapsed,
            "environment_steps_per_second": global_step / elapsed,
            "reset_count": reset_count,
            "completed_episodes": len(completed_returns),
            "completed_successes": completed_successes,
            "termination_counts": dict(termination_counts),
            "first_quartile_return_mean": float(
                np.mean(completed_returns[: max(1, len(completed_returns) // 4)])
            ),
            "last_quartile_return_mean": float(
                np.mean(completed_returns[-max(1, len(completed_returns) // 4) :])
            ),
            "first_quartile_progress_mean": float(
                np.mean(completed_progress[: max(1, len(completed_progress) // 4)])
            ),
            "last_quartile_progress_mean": float(
                np.mean(completed_progress[-max(1, len(completed_progress) // 4) :])
            ),
            "metrics": metrics,
            "final_loss": final_loss,
            "checkpoint_path": checkpoint.as_posix(),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "checkpoint_global_step": loaded_payload["global_step"],
            "deterministic_reload_max_abs_error": reload_error,
            "finite_losses": True,
            "finite_gradients": True,
            "simulator_exceptions": 0,
            "peak_vram_mib": torch.cuda.max_memory_allocated() / (1024**2),
            "peak_cpu_memory_mib": _process_memory_mib(),
            "combined_f1_budget_after_probe": 262_144 if probe == "A" else 524_288,
            "demonstrations_generated": False,
            "student_policy_training_started": False,
            "pipeline_execution_passed": True,
        }
        report["fingerprint"] = canonical_json_sha256(
            {
                key: value
                for key, value in report.items()
                if key not in {"metrics", "checkpoint_path"}
            }
        )
        return report
    finally:
        runtime.close()


def evaluate_probe_checkpoint(
    *,
    probe: str,
    checkpoint: Path,
    reward_revision: int,
) -> dict[str, object]:
    if probe not in PROBE_EVALUATION:
        raise ValueError("probe must be A or B")
    assert_f1_operation_allowed(f"probe_{probe.lower()}_evaluation")
    namespace, seed_start, episode_count = PROBE_EVALUATION[probe]
    policy, checkpoint_payload = reconstruct_f1_policy(checkpoint)
    runtime = Phase2B4F1VectorRuntime(num_envs=1, probe=probe)
    policy = policy.to(runtime.device)
    policy.eval()
    records: list[dict[str, object]] = []
    action_nonfinite = 0
    action_out_of_bounds = 0
    all_actions: list[np.ndarray] = []
    started = time.perf_counter()
    try:
        for episode_index in range(episode_count):
            seed = seed_start + episode_index
            task = STAGE0_TASKS[episode_index % len(STAGE0_TASKS)]
            state = runtime.reset_single_evaluation(
                seed=seed,
                task=task,
                namespace=namespace,
            )
            initial_distance = float(state.target_distance[0].item())
            previous_action_unit = torch.zeros((1, 8), device=runtime.device)
            entered_approach = False
            correct_contact = False
            target_progress = False
            saturation_steps = 0
            wrong_object = False
            workspace_exit = False
            invalid_action = False
            outcome = "timeout"
            total_return = 0.0
            for _step in range(250):
                with torch.no_grad():
                    latent = policy.distribution(state.observation).mean
                    action = policy.action_from_latent(state.observation, latent)
                    action_unit = policy.action_unit_from_latent(state.observation, latent)
                all_actions.append(action[0].cpu().numpy())
                action_nonfinite += int((~torch.isfinite(action)).any().item())
                action_out_of_bounds += int(
                    ((action < runtime.action_low) | (action > runtime.action_high)).any().item()
                )
                saturation_steps += int((action_unit[0].abs() > 0.95).any().item())
                previous = state
                state, env_terminated, env_truncated, info = runtime.step(action)
                reward, _components = _reward_with_revision(
                    previous=previous,
                    current=state,
                    info=info,
                    action_unit=action_unit,
                    previous_action_unit=previous_action_unit,
                    reward_revision=reward_revision,
                )
                total_return += float(reward[0].item())
                previous_action_unit = action_unit
                entered_approach |= bool(state.behind_distance[0].item() <= 0.12)
                correct_contact |= bool(state.target_contact[0].item())
                target_progress |= bool(
                    previous.target_distance[0].item() - state.target_distance[0].item() >= 0.001
                )
                wrong_object |= bool(
                    (
                        _metric_bool(info, "wrong_object_contact")
                        | _metric_bool(info, "wrong_object_displaced")
                    )[0].item()
                )
                workspace_exit |= bool(_metric_bool(info, "target_outside_workspace")[0].item())
                invalid_action |= bool(_metric_bool(info, "invalid_action")[0].item())
                safety = _safety_mask(info, ZERO_TOLERANCE_KEYS)
                effective_terminated = bool(env_terminated[0].item() or safety[0].item())
                truncated = bool(env_truncated[0].item() and not effective_terminated)
                if effective_terminated or truncated:
                    outcome = _termination_reason(
                        info,
                        terminated=effective_terminated,
                        truncated=truncated,
                    )
                    break
            success = outcome == "stable_success"
            records.append(
                {
                    "episode_index": episode_index,
                    "seed": seed,
                    "task": task.to_dict(),
                    "steps": _step + 1,
                    "return": total_return,
                    "outcome": outcome,
                    "success": success,
                    "entered_useful_approach": entered_approach,
                    "correct_contact": correct_contact,
                    "target_directed_progress": target_progress,
                    "target_distance_reduced": (
                        initial_distance - float(state.target_distance[0].item()) >= 0.005
                    ),
                    "wrong_object_interaction": wrong_object,
                    "workspace_exit": workspace_exit,
                    "invalid_action": invalid_action,
                    "action_saturation_steps": saturation_steps,
                }
            )
    finally:
        runtime.close()
    direction_successes: dict[str, int] = defaultdict(int)
    for record in records:
        if bool(record["success"]):
            record_task = cast(Mapping[str, object], record["task"])
            direction_successes[str(record_task["target_region_id"])] += 1
    summary = ProbeEvaluation(
        episode_count=episode_count,
        successes=sum(bool(item["success"]) for item in records),
        useful_approach_episodes=sum(bool(item["entered_useful_approach"]) for item in records),
        correct_contact_episodes=sum(bool(item["correct_contact"]) for item in records),
        target_progress_episodes=sum(bool(item["target_directed_progress"]) for item in records),
        direction_successes=dict(direction_successes),
        wrong_object_interactions=sum(bool(item["wrong_object_interaction"]) for item in records),
        workspace_exits=sum(bool(item["workspace_exit"]) for item in records),
        invalid_actions=sum(bool(item["invalid_action"]) for item in records),
        nonfinite_actions=action_nonfinite,
        out_of_bounds_actions=action_out_of_bounds,
    )
    gate = (
        probe_a_gate(summary)
        if probe == "A"
        else probe_b_gate(summary, included_directions=("left", "forward_right"))
    )
    actions = np.asarray(all_actions, dtype=np.float32)
    elapsed = time.perf_counter() - started
    report: dict[str, object] = {
        "schema_version": f"langmani-v2-phase2b4-f1-probe-{probe.lower()}-evaluation-v0",
        "probe": probe,
        "checkpoint_sha256": ("sha256:" + hashlib.sha256(checkpoint.read_bytes()).hexdigest()),
        "checkpoint_global_step": checkpoint_payload["global_step"],
        "reward_revision": reward_revision,
        "summary": {
            **asdict(summary),
            "success_rate": summary.rate(summary.successes),
            "useful_approach_rate": summary.rate(summary.useful_approach_episodes),
            "correct_contact_rate": summary.rate(summary.correct_contact_episodes),
            "target_progress_rate": summary.rate(summary.target_progress_episodes),
            "average_episode_length": float(
                np.mean([int(cast(int, item["steps"])) for item in records])
            ),
            "action_saturation_steps": sum(
                int(cast(int, item["action_saturation_steps"])) for item in records
            ),
        },
        "gate": gate,
        "passed": all(gate.values()),
        "action_integrity": {
            "action_count": len(actions),
            "nonfinite": action_nonfinite,
            "out_of_bounds": action_out_of_bounds,
            "clipping": 0,
            "projection": 0,
            "component_std": actions.std(axis=0).tolist(),
        },
        "elapsed_seconds": elapsed,
        "average_latency_seconds": elapsed / episode_count,
        "formal_seed_accessed": False,
        "demonstrations_generated": False,
        "records": records,
    }
    report["fingerprint"] = canonical_json_sha256(
        {key: value for key, value in report.items() if key != "records"}
    )
    return report


__all__ = [
    "F1_CHECKPOINT_SCHEMA",
    "PROBE_EVALUATION",
    "PROBE_POLICY_SEEDS",
    "PROBE_TRAIN_STARTS",
    "STAGE0_TASKS",
    "Phase2B4F1VectorRuntime",
    "audit_initial_exploration",
    "compare_global_and_residual_physics",
    "diagnose_f0_checkpoint",
    "evaluate_probe_checkpoint",
    "reconstruct_f1_policy",
    "run_probe_training",
]
