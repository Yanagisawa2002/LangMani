"""Bounded PPO failure-diagnosis contracts for LangMani 2.0 Phase 2B.4-F1."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as functional

from langmani.v2.phase2b4_ppo import (
    PANDA_QPOS_SLICE,
    TEACHER_OBSERVATION_DIM,
    canonical_json_sha256,
    compute_gae,
    normalize_teacher_observation,
)

SOURCE_COMMIT: Final = "c5892a357da85f753d5d0c3775ccf2b94e0d8910"
EXCLUDED_GEOMETRY_COMMIT: Final = "ae6aae49c61d96c68de434e08dc1167461a29543"
FORMAL_QUALIFICATION_SEEDS: Final = frozenset(range(66300, 66400))
ACTION_SCHEMA_VERSION: Final = "langmani-v2-phase2b4-f1-state-centered-residual-v1"
SEED_SCHEMA_VERSION: Final = "langmani-v2-phase2b4-f1-seeds-v0"
CURRICULUM_SCHEMA_VERSION: Final = "langmani-v2-phase2b4-f1-safe-curriculum-v0"
PANDA_FINGER_OPEN_QPOS: Final = 0.04
SAFE_ATANH_EPSILON: Final = 1e-6
ZERO_RESIDUAL_IDENTITY_TOLERANCE: Final = 2e-6
DEFAULT_RESIDUAL_SCALES: Final = (0.08,) * 7 + (0.25,)
F1_ALLOWED_OPERATIONS: Final = frozenset(
    {
        "contract_audit",
        "f0_read_only_diagnosis",
        "initial_exploration_audit",
        "global_residual_physical_comparison",
        "probe_a_training",
        "probe_a_evaluation",
        "probe_b_training",
        "probe_b_evaluation",
    }
)
F1_PROHIBITED_OPERATIONS: Final = frozenset(
    {
        "stage_1_training",
        "stage_2_training",
        "ppo_full_training",
        "formal_qualification",
        "expert_qualification",
        "demonstration_collection",
        "lerobot_export",
        "student_training",
        "smolvla_training",
        "act_training",
        "vla_jepa_training",
        "simulator_mpc",
        "scripted_fallback",
        "motion_planning",
    }
)


@dataclass(frozen=True, slots=True)
class SeedNamespace:
    name: str
    start_inclusive: int
    stop_exclusive: int
    accessible_in_f1: bool

    def __post_init__(self) -> None:
        if self.start_inclusive < 0 or self.stop_exclusive <= self.start_inclusive:
            raise ValueError(f"invalid seed namespace {self.name!r}")

    @property
    def values(self) -> range:
        return range(self.start_inclusive, self.stop_exclusive)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "start_inclusive": self.start_inclusive,
            "stop_exclusive": self.stop_exclusive,
            "count": self.stop_exclusive - self.start_inclusive,
            "accessible_in_f1": self.accessible_in_f1,
        }


F1_SEED_NAMESPACES: Final = (
    SeedNamespace("formal_qualification", 66300, 66400, False),
    SeedNamespace("f1_probe_train", 1100000, 1200000, True),
    SeedNamespace("f1_probe_a_evaluation", 1200000, 1200032, True),
    SeedNamespace("f1_probe_b_evaluation", 1201000, 1201048, True),
    SeedNamespace("f1_diagnostic", 1202000, 1202256, True),
    SeedNamespace("future_full_curriculum", 1210000, 1310000, False),
)


def assert_f1_operation_allowed(operation: str) -> None:
    if operation not in F1_ALLOWED_OPERATIONS:
        status = "prohibited" if operation in F1_PROHIBITED_OPERATIONS else "undeclared"
        raise PermissionError(f"Phase 2B.4-F1 operation {operation!r} is {status}")


def assert_f1_seed_allowed(seed: int, namespace: str) -> None:
    if seed in FORMAL_QUALIFICATION_SEEDS:
        raise PermissionError("formal qualification seeds 66300--66399 are sealed")
    matches = [item for item in F1_SEED_NAMESPACES if item.name == namespace]
    if len(matches) != 1:
        raise ValueError(f"unknown F1 seed namespace {namespace!r}")
    selected = matches[0]
    if not selected.accessible_in_f1:
        raise PermissionError(f"F1 seed namespace {namespace!r} is sealed")
    if seed not in selected.values:
        raise ValueError(f"seed {seed} is outside namespace {namespace!r}")


def seed_disjointness_audit() -> dict[str, object]:
    overlaps: list[dict[str, object]] = []
    for index, left in enumerate(F1_SEED_NAMESPACES):
        left_values = set(left.values)
        for right in F1_SEED_NAMESPACES[index + 1 :]:
            intersection = sorted(left_values.intersection(right.values))
            if intersection:
                overlaps.append(
                    {
                        "left": left.name,
                        "right": right.name,
                        "first": intersection[0],
                        "count": len(intersection),
                    }
                )
    payload: dict[str, object] = {
        "schema_version": SEED_SCHEMA_VERSION,
        "namespaces": [item.to_dict() for item in F1_SEED_NAMESPACES],
        "overlaps": overlaps,
        "formal_seed_accessed": False,
        "f0_micro_evaluation_reused": False,
        "passed": not overlaps,
    }
    payload["fingerprint"] = canonical_json_sha256(payload)
    return payload


def safe_atanh(
    value: torch.Tensor,
    *,
    epsilon: float = SAFE_ATANH_EPSILON,
) -> torch.Tensor:
    """Map normalized current state to finite inverse-tanh coordinates."""
    if not 0.0 < epsilon < 0.1:
        raise ValueError("safe-atanh epsilon must be in (0, 0.1)")
    if not torch.isfinite(value).all():
        raise ValueError("safe-atanh input must be finite")
    return torch.atanh(value.clamp(min=-1.0 + epsilon, max=1.0 - epsilon))


def _log_tanh_derivative(pre_tanh: torch.Tensor) -> torch.Tensor:
    """Return the exact, numerically stable log derivative of tanh."""
    return 2.0 * (math.log(2.0) - pre_tanh - functional.softplus(-2.0 * pre_tanh))


def _validate_bounds(
    action_low: torch.Tensor, action_high: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    low = action_low.detach().to(dtype=torch.float32).reshape(-1)
    high = action_high.detach().to(dtype=torch.float32).reshape(-1)
    if low.shape != (8,) or high.shape != (8,):
        raise ValueError("F1 action bounds must have shape (8,)")
    if not torch.isfinite(low).all() or not torch.isfinite(high).all() or not torch.all(low < high):
        raise ValueError("F1 action bounds must be finite and ordered")
    return low, high


def current_action_reference(
    observation: torch.Tensor,
    *,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
) -> torch.Tensor:
    """Map current Panda qpos to the native eight-dimensional target action."""
    if observation.ndim != 2 or observation.shape[1] != TEACHER_OBSERVATION_DIM:
        raise ValueError(f"observation must have shape (batch, {TEACHER_OBSERVATION_DIM})")
    low, high = _validate_bounds(action_low, action_high)
    qpos = observation[:, PANDA_QPOS_SLICE]
    finger_mean = qpos[:, 7:9].mean(dim=1)
    gripper = 2.0 * finger_mean / PANDA_FINGER_OPEN_QPOS - 1.0
    reference = torch.cat((qpos[:, :7], gripper[:, None]), dim=1)
    # This is a current-state coordinate-domain guard. It never clips an emitted
    # policy action and its use is recorded in the transform manifest.
    return torch.maximum(torch.minimum(reference, high), low)


def _layer_init(layer: nn.Linear, *, std: float = math.sqrt(2.0)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class StateCenteredResidualActorCritic(nn.Module):
    """PPO policy with a bounded inverse-tanh residual around current qpos."""

    action_low: torch.Tensor
    action_high: torch.Tensor
    action_scale: torch.Tensor
    action_bias: torch.Tensor
    residual_scale: torch.Tensor
    actor: nn.Sequential
    critic: nn.Sequential
    actor_log_std: nn.Parameter

    def __init__(
        self,
        *,
        observation_dim: int,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        hidden_dim: int = 256,
        initial_log_std: float = -0.5,
        residual_scales: Sequence[float] = DEFAULT_RESIDUAL_SCALES,
    ) -> None:
        super().__init__()
        if observation_dim != TEACHER_OBSERVATION_DIM:
            raise ValueError(f"observation_dim must be {TEACHER_OBSERVATION_DIM}")
        low, high = _validate_bounds(action_low, action_high)
        scale = torch.as_tensor(residual_scales, dtype=torch.float32).reshape(-1)
        if scale.shape != (8,) or not torch.isfinite(scale).all() or not torch.all(scale > 0):
            raise ValueError("residual scales must be eight positive finite values")
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)
        self.register_buffer("residual_scale", scale)
        self.actor = nn.Sequential(
            _layer_init(nn.Linear(observation_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, 8), std=0.01 * math.sqrt(2.0)),
        )
        self.critic = nn.Sequential(
            _layer_init(nn.Linear(observation_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.actor_log_std = nn.Parameter(torch.full((1, 8), initial_log_std))

    def current_action_unit(self, observation: torch.Tensor) -> torch.Tensor:
        reference = current_action_reference(
            observation,
            action_low=self.action_low,
            action_high=self.action_high,
        )
        return (reference - self.action_bias) / self.action_scale

    def distribution(self, observation: torch.Tensor) -> Normal:
        mean = self.actor(normalize_teacher_observation(observation))
        std = torch.exp(self.actor_log_std).expand_as(mean)
        return Normal(mean, std)

    def action_unit_from_latent(
        self, observation: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        if latent.shape != (observation.shape[0], 8):
            raise ValueError("latent must align with observation and have width eight")
        current_coordinate = safe_atanh(self.current_action_unit(observation))
        bounded_residual = self.residual_scale * torch.tanh(latent)
        return torch.tanh(current_coordinate + bounded_residual)

    def action_from_latent(self, observation: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        unit = self.action_unit_from_latent(observation, latent)
        return (self.action_bias + self.action_scale * unit).to(torch.float32)

    def log_prob(
        self,
        distribution: Normal,
        observation: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        residual_unit = torch.tanh(latent)
        current_coordinate = safe_atanh(self.current_action_unit(observation))
        action_coordinate = current_coordinate + self.residual_scale * residual_unit
        log_jacobian = (
            torch.log(self.action_scale)
            + torch.log(self.residual_scale)
            + _log_tanh_derivative(action_coordinate)
            + _log_tanh_derivative(latent)
        )
        return (distribution.log_prob(latent) - log_jacobian).sum(dim=1)

    def action_and_value(
        self,
        observation: torch.Tensor,
        *,
        latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation)
        selected_latent = distribution.rsample() if latent is None else latent
        action = self.action_from_latent(observation, selected_latent)
        log_prob = self.log_prob(distribution, observation, selected_latent)
        sampled_entropy = -log_prob
        value = self.critic(normalize_teacher_observation(observation)).squeeze(1)
        return action, selected_latent, log_prob, sampled_entropy, value

    def deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        distribution = self.distribution(observation)
        return self.action_from_latent(observation, distribution.mean)

    def value(self, observation: torch.Tensor) -> torch.Tensor:
        return self.critic(normalize_teacher_observation(observation)).squeeze(1)

    def action_manifest(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": ACTION_SCHEMA_VERSION,
            "name": "state_centered_bounded_residual_v1",
            "dtype": "float32",
            "shape": [8],
            "control_mode": "pd_joint_pos",
            "bounds_source": "active environment single_action_space",
            "low": self.action_low.detach().cpu().tolist(),
            "high": self.action_high.detach().cpu().tolist(),
            "residual_scales": self.residual_scale.detach().cpu().tolist(),
            "transform": (
                "current action reference -> normalized z_current -> safe atanh; "
                "z_next=tanh(atanh_safe(z_current)+scale*tanh(policy_latent)); "
                "native action=affine(z_next)"
            ),
            "arm_reference": "current Panda qpos[0:7]",
            "gripper_reference": "current mean finger qpos mapped from [0,0.04] to [-1,1]",
            "zero_residual_identity": True,
            "current_coordinate_boundary_guard": ("clamp to [-1+1e-6,1-1e-6] before atanh"),
            "stochastic_and_deterministic_transform_shared": True,
            "exact_change_of_variables_log_probability": True,
            "clipping_emitted_action": False,
            "projection": False,
            "fallback": False,
        }
        payload["fingerprint"] = canonical_json_sha256(payload)
        return payload


def audit_residual_action_legality(
    policy: StateCenteredResidualActorCritic,
    observations: torch.Tensor,
    *,
    sample_count: int = 100_000,
    batch_size: int = 10_000,
) -> dict[str, object]:
    if sample_count < 100_000:
        raise ValueError("F1 residual action audit requires at least 100,000 samples")
    if observations.ndim != 2 or observations.shape[1] != TEACHER_OBSERVATION_DIM:
        raise ValueError("audit observations have the wrong schema")
    device = policy.action_low.device
    observations = observations.to(device)
    nonfinite = 0
    out_of_bounds = 0
    maximum_zero_identity_error = 0.0
    observed_min = torch.full((8,), torch.inf, device=device)
    observed_max = torch.full((8,), -torch.inf, device=device)
    processed = 0
    while processed < sample_count:
        count = min(batch_size, sample_count - processed)
        indices = torch.arange(processed, processed + count, device=device) % len(observations)
        batch_observations = observations[indices]
        distribution = policy.distribution(batch_observations)
        latent = distribution.sample()
        action = policy.action_from_latent(batch_observations, latent)
        zero_action = policy.action_from_latent(batch_observations, torch.zeros_like(latent))
        reference = current_action_reference(
            batch_observations,
            action_low=policy.action_low,
            action_high=policy.action_high,
        )
        maximum_zero_identity_error = max(
            maximum_zero_identity_error,
            float((zero_action - reference).abs().max().item()),
        )
        nonfinite += int((~torch.isfinite(action)).any(dim=1).sum().item())
        out_of_bounds += int(
            ((action < policy.action_low) | (action > policy.action_high)).any(dim=1).sum().item()
        )
        observed_min = torch.minimum(observed_min, action.amin(dim=0))
        observed_max = torch.maximum(observed_max, action.amax(dim=0))
        processed += count
    payload: dict[str, object] = {
        "schema_version": "langmani-v2-phase2b4-f1-residual-action-legality-v0",
        "action_schema": ACTION_SCHEMA_VERSION,
        "sample_count": sample_count,
        "nonfinite_actions": nonfinite,
        "out_of_bounds_actions": out_of_bounds,
        "clipping_events": 0,
        "projection_events": 0,
        "maximum_zero_residual_identity_error": maximum_zero_identity_error,
        "zero_residual_tolerance": ZERO_RESIDUAL_IDENTITY_TOLERANCE,
        "observed_min": observed_min.detach().cpu().tolist(),
        "observed_max": observed_max.detach().cpu().tolist(),
        "passed": (
            nonfinite == 0
            and out_of_bounds == 0
            and maximum_zero_identity_error <= ZERO_RESIDUAL_IDENTITY_TOLERANCE
        ),
    }
    payload["fingerprint"] = canonical_json_sha256(payload)
    return payload


def curriculum_manifest() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": CURRICULUM_SCHEMA_VERSION,
        "physics_equivalence": {
            "environment": "LangMani-PushToRegion-v0",
            "robot": "Panda",
            "control_mode": "pd_joint_pos",
            "action_shape": [8],
            "success_predicate": "unchanged canonical stable success",
            "episode_budget": 250,
            "object_geometry_changed": False,
            "target_geometry_changed": False,
            "mass_or_friction_changed": False,
        },
        "stages": [
            {
                "name": "stage_0_contact_acquisition",
                "runnable_in_f1": True,
                "objects": ["blue_cube"],
                "directions": ["left", "forward_right"],
                "difficulty": ["standard"],
                "distractor": "unchanged Standard distractor at its canonical safe location",
                "reset_range": "f1_probe_train",
                "advancement": {
                    "probe_a": (
                        "zero safety events, contact >=25%, target-directed progress >=20%"
                    ),
                    "probe_b": (
                        "success >=20%, per-direction cube success, contact >=50%, "
                        "progress >=40%, zero safety events"
                    ),
                },
                "rollback": "any frozen zero-tolerance event or failed gate hard-stops F1",
            },
            {
                "name": "stage_1_geometry_direction_expansion",
                "runnable_in_f1": False,
                "objects": ["blue_cube", "orange_cylinder"],
                "directions": ["left", "right", "forward_left", "forward_right"],
                "difficulty": ["standard"],
            },
            {
                "name": "stage_2_full_task_distribution",
                "runnable_in_f1": False,
                "objects": ["blue_cube", "orange_cylinder"],
                "directions": ["left", "right", "forward_left", "forward_right"],
                "difficulty": ["standard", "hard"],
            },
        ],
        "forbidden_simplifications": [
            "success relaxation",
            "episode-budget increase",
            "physics modification",
            "direct object control",
            "scripted fallback",
            "motion planning",
            "distractor deletion from the environment implementation",
        ],
    }
    payload["fingerprint"] = canonical_json_sha256(payload)
    return payload


def reconstruct_f0_learning_dynamics(report: Mapping[str, Any]) -> dict[str, object]:
    raw_metrics = report.get("metrics")
    if not isinstance(raw_metrics, list) or not raw_metrics:
        raise ValueError("F0 report has no recorded training metric snapshots")
    metrics = [item for item in raw_metrics if isinstance(item, Mapping)]
    if len(metrics) != len(raw_metrics):
        raise ValueError("F0 metric snapshot is malformed")
    first = metrics[0]
    last = metrics[-1]
    first_loss = first.get("last_loss")
    last_loss = last.get("last_loss")
    if not isinstance(first_loss, Mapping) or not isinstance(last_loss, Mapping):
        raise ValueError("F0 loss snapshots are malformed")
    clip_fractions = [
        float(item["last_loss"]["clip_fraction"])
        for item in metrics
        if isinstance(item.get("last_loss"), Mapping)
    ]
    approximate_kls = [
        float(item["last_loss"]["approximate_kl"])
        for item in metrics
        if isinstance(item.get("last_loss"), Mapping)
    ]
    entropies = [
        float(item["last_loss"]["entropy"])
        for item in metrics
        if isinstance(item.get("last_loss"), Mapping)
    ]
    reward_means = [float(item["reward_mean"]) for item in metrics]
    raw_learning_evidence = report.get("learning_evidence")
    learning_evidence = raw_learning_evidence if isinstance(raw_learning_evidence, Mapping) else {}
    result: dict[str, object] = {
        "schema_version": "langmani-v2-phase2b4-f1-f0-learning-dynamics-audit-v0",
        "source_report_schema": report.get("schema_version"),
        "source_report_fingerprint": report.get("fingerprint"),
        "recorded_snapshot_count": len(metrics),
        "global_step": report.get("global_step"),
        "completed_episodes": report.get("completed_episodes"),
        "completed_successes": report.get("completed_successes"),
        "first_quartile_return_mean": report.get("first_quartile_return_mean"),
        "last_quartile_return_mean": report.get("last_quartile_return_mean"),
        "first_quartile_progress_mean": report.get("first_quartile_progress_mean"),
        "last_quartile_progress_mean": report.get("last_quartile_progress_mean"),
        "recorded_reward_mean": {
            "first": reward_means[0],
            "last": reward_means[-1],
            "minimum": min(reward_means),
            "maximum": max(reward_means),
        },
        "recorded_actor_loss": {
            "first": float(first_loss["actor"]),
            "last": float(last_loss["actor"]),
        },
        "recorded_critic_loss": {
            "first": float(first_loss["critic"]),
            "last": float(last_loss["critic"]),
        },
        "recorded_transformed_sample_entropy": {
            "first": entropies[0],
            "last": entropies[-1],
            "note": (
                "F0 recorded -log_prob after the action transform, not an analytic "
                "Normal entropy; this trace alone cannot prove entropy collapse."
            ),
        },
        "recorded_approximate_kl": {
            "first": approximate_kls[0],
            "last": approximate_kls[-1],
            "maximum": max(approximate_kls),
            "snapshots_above_target_0_1": sum(value > 0.1 for value in approximate_kls),
        },
        "recorded_clip_fraction": {
            "first": clip_fractions[0],
            "last": clip_fractions[-1],
            "median": float(np.median(clip_fractions)),
            "maximum": max(clip_fractions),
        },
        "diagnostic_classification": {
            "no_useful_task_learning": (
                int(report.get("completed_successes", -1)) == 0
                and not bool(
                    learning_evidence.get("target_distance_progress_improved_materially", False)
                )
            ),
            "partial_approach_learning_proven": False,
            "contact_learning_without_progress_proven": False,
            "unsafe_progress_observed_in_fixed_evaluation": True,
            "critic_failure_proven": False,
            "entropy_collapse_proven": False,
            "premature_policy_collapse_proven": False,
            "reward_cancellation_proven": False,
        },
        "unavailable_from_frozen_training_report": [
            "per-episode median return",
            "per-component reward timeline",
            "value explained variance",
            "action mean and standard deviation by iteration",
            "state-value prediction distribution",
            "training termination reason per episode",
            "contact and containment event frequencies",
        ],
        "evidence_boundary": (
            "These fields were not retained by F0 and are reported unavailable rather "
            "than reconstructed from later rollouts."
        ),
    }
    result["fingerprint"] = canonical_json_sha256(result)
    return result


def termination_bootstrap_audit() -> dict[str, object]:
    rewards = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    values = torch.tensor([[0.25, 0.25]], dtype=torch.float32)
    next_values = torch.tensor([[2.0, 2.0]], dtype=torch.float32)
    terminated = torch.tensor([[True, False]])
    truncated = torch.tensor([[False, True]])
    advantages, _returns = compute_gae(
        rewards=rewards,
        values=values,
        next_values=next_values,
        terminated=terminated,
        truncated=truncated,
        gamma=0.99,
        gae_lambda=0.95,
    )
    expected_terminated = 1.0 - 0.25
    expected_truncated = 1.0 + 0.99 * 2.0 - 0.25
    payload: dict[str, object] = {
        "schema_version": "langmani-v2-phase2b4-f1-termination-bootstrap-audit-v0",
        "training_semantics": {
            "safety_mask_merged_into_terminated": True,
            "safety_reset_same_transition": True,
            "post_failure_tail_steps": 0,
            "terminated_bootstrap": False,
            "truncated_bootstrap": True,
            "terminal_observation_value_computed_then_masked_for_termination": True,
            "asynchronous_reset_uses_explicit_full_seed_vector_and_physical_mask": True,
        },
        "probe_advantages": advantages.tolist(),
        "expected_terminated_advantage": expected_terminated,
        "expected_truncated_advantage": expected_truncated,
        "passed": bool(
            torch.isclose(advantages[0, 0], torch.tensor(expected_terminated))
            and torch.isclose(advantages[0, 1], torch.tensor(expected_truncated))
        ),
        "historical_episode_reason_limitation": (
            "F0 did not retain per-training-episode termination labels; the exact "
            "21,034-episode reason histogram is unavailable."
        ),
    }
    payload["fingerprint"] = canonical_json_sha256(payload)
    return payload


@dataclass(frozen=True, slots=True)
class ProbeEvaluation:
    episode_count: int
    successes: int
    useful_approach_episodes: int
    correct_contact_episodes: int
    target_progress_episodes: int
    direction_successes: Mapping[str, int]
    wrong_object_interactions: int
    workspace_exits: int
    invalid_actions: int
    nonfinite_actions: int
    out_of_bounds_actions: int

    def __post_init__(self) -> None:
        if self.episode_count <= 0:
            raise ValueError("probe evaluation must contain episodes")

    def rate(self, count: int) -> float:
        return count / self.episode_count


def probe_a_gate(summary: ProbeEvaluation) -> dict[str, bool]:
    return {
        "episode_count_exact": summary.episode_count == 32,
        "action_integrity": (
            summary.invalid_actions == 0
            and summary.nonfinite_actions == 0
            and summary.out_of_bounds_actions == 0
        ),
        "wrong_object_zero": summary.wrong_object_interactions == 0,
        "workspace_exit_zero": summary.workspace_exits == 0,
        "correct_contact_rate": summary.rate(summary.correct_contact_episodes) >= 0.25,
        "target_progress_rate": summary.rate(summary.target_progress_episodes) >= 0.20,
    }


def probe_b_gate(
    summary: ProbeEvaluation, *, included_directions: Sequence[str]
) -> dict[str, bool]:
    return {
        "episode_count_exact": summary.episode_count == 48,
        "success_rate": summary.rate(summary.successes) >= 0.20,
        "each_direction_success": all(
            int(summary.direction_successes.get(direction, 0)) >= 1
            for direction in included_directions
        ),
        "correct_contact_rate": summary.rate(summary.correct_contact_episodes) >= 0.50,
        "target_progress_rate": summary.rate(summary.target_progress_episodes) >= 0.40,
        "wrong_object_zero": summary.wrong_object_interactions == 0,
        "workspace_exit_zero": summary.workspace_exits == 0,
        "action_integrity": (
            summary.invalid_actions == 0
            and summary.nonfinite_actions == 0
            and summary.out_of_bounds_actions == 0
        ),
    }


class Phase2B4F1Result(StrEnum):
    RESULT_A = "RESULT_A"
    RESULT_B = "RESULT_B"
    RESULT_C = "RESULT_C"
    RESULT_D = "RESULT_D"


def classify_f1_result(
    *,
    diagnosis_valid: bool,
    pipeline_valid: bool,
    residual_exploration_safer: bool,
    probe_a_passed: bool,
    probe_b_ran: bool,
    probe_b_passed: bool,
) -> Phase2B4F1Result:
    if not diagnosis_valid or not pipeline_valid:
        return Phase2B4F1Result.RESULT_D
    if not residual_exploration_safer or not probe_a_passed:
        return Phase2B4F1Result.RESULT_C
    if not probe_b_ran or not probe_b_passed:
        return Phase2B4F1Result.RESULT_B
    return Phase2B4F1Result.RESULT_A


def authorization_state(result: Phase2B4F1Result) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "langmani-v2-phase2b4-f1-authorization-v0",
        "result": result.value,
        "ppo_failure_mechanism_identified": result is not Phase2B4F1Result.RESULT_D,
        "ppo_safe_curriculum_validated": result is Phase2B4F1Result.RESULT_A,
        "ppo_safe_curriculum_eligible": result is Phase2B4F1Result.RESULT_A,
        "ppo_full_training_authorized": False,
        "expert_qualification_authorized": False,
        "data_collection_authorized": False,
        "smolvla_training_authorized": False,
        "demonstration_source_validated": False,
        "student_policy_training_started": False,
    }
    payload["fingerprint"] = canonical_json_sha256(payload)
    return payload


def probe_configuration(*, probe: str) -> dict[str, object]:
    if probe not in {"A", "B"}:
        raise ValueError("probe must be A or B")
    payload: dict[str, object] = {
        "schema_version": f"langmani-v2-phase2b4-f1-probe-{probe.lower()}-configuration-v0",
        "probe": probe,
        "frozen_before_run": True,
        "total_environment_steps": 262_144,
        "combined_f1_budget_limit": 524_288,
        "num_envs": 128,
        "num_steps": 32,
        "iterations": 64,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "update_epochs": 4,
        "num_minibatches": 8,
        "clip_coefficient": 0.2,
        "value_coefficient": 0.5,
        "entropy_coefficient": 0.001,
        "max_gradient_norm": 0.5,
        "target_kl": 0.1,
        "policy_rng_seed": 240410 if probe == "A" else 240411,
        "action_schema": ACTION_SCHEMA_VERSION,
        "reward_schema": (
            "langmani-v2-phase2b4-f0-reward-v0"
            if probe == "A"
            else "conditional_supported_revision_only"
        ),
        "curriculum_stage": "stage_0_contact_acquisition",
        "resume": False,
        "sweep": False,
        "formal_qualification": False,
        "demonstration_collection": False,
    }
    payload["fingerprint"] = canonical_json_sha256(payload)
    return payload


def repository_operation_boundary() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "langmani-v2-phase2b4-f1-operation-boundary-v0",
        "source_commit": SOURCE_COMMIT,
        "excluded_geometry_commit": EXCLUDED_GEOMETRY_COMMIT,
        "allowed": sorted(F1_ALLOWED_OPERATIONS),
        "prohibited": sorted(F1_PROHIBITED_OPERATIONS),
        "maximum_probe_count": 2,
        "maximum_combined_training_steps": 524_288,
        "formal_seeds_accessed": False,
        "demonstrations_generated": False,
        "datasets_created": False,
        "student_training_started": False,
    }
    payload["fingerprint"] = canonical_json_sha256(payload)
    return payload


__all__ = [
    "ACTION_SCHEMA_VERSION",
    "CURRICULUM_SCHEMA_VERSION",
    "DEFAULT_RESIDUAL_SCALES",
    "EXCLUDED_GEOMETRY_COMMIT",
    "F1_ALLOWED_OPERATIONS",
    "F1_PROHIBITED_OPERATIONS",
    "F1_SEED_NAMESPACES",
    "FORMAL_QUALIFICATION_SEEDS",
    "Phase2B4F1Result",
    "ProbeEvaluation",
    "SAFE_ATANH_EPSILON",
    "SOURCE_COMMIT",
    "StateCenteredResidualActorCritic",
    "ZERO_RESIDUAL_IDENTITY_TOLERANCE",
    "assert_f1_operation_allowed",
    "assert_f1_seed_allowed",
    "audit_residual_action_legality",
    "authorization_state",
    "classify_f1_result",
    "current_action_reference",
    "curriculum_manifest",
    "probe_a_gate",
    "probe_b_gate",
    "probe_configuration",
    "reconstruct_f0_learning_dynamics",
    "repository_operation_boundary",
    "safe_atanh",
    "seed_disjointness_audit",
    "termination_bootstrap_audit",
]
