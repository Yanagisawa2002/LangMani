"""Bounded privileged-state PPO contracts for LangMani 2.0 Phase 2B.4-F0.

The PPO update follows the maintained ManiSkill 3.0.1 state-PPO baseline at
tag ``v3.0.1`` (commit ``a4a4f9272ad64b1564035874b605ceb687b63ed8``), but
the policy distribution is deliberately not copied: the upstream baseline
clips an unbounded Normal sample, while this phase requires a structurally
bounded tanh-affine distribution with no clipping or projection.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

import torch
from torch import nn
from torch.distributions import Normal

SOURCE_COMMIT: Final = "3ded856bc0fba7cd749f1fb0caf738fd7fce5751"
EXCLUDED_GEOMETRY_COMMIT: Final = "ae6aae49c61d96c68de434e08dc1167461a29543"
FORMAL_QUALIFICATION_SEEDS: Final = frozenset(range(66300, 66400))
OFFICIAL_PPO_REVISION: Final = "a4a4f9272ad64b1564035874b605ceb687b63ed8"
OBSERVATION_SCHEMA_VERSION: Final = "langmani-v2-phase2b4-f0-teacher-observation-v0"
ACTION_SCHEMA_VERSION: Final = "langmani-v2-phase2b4-f0-tanh-affine-action-v0"
REWARD_SCHEMA_VERSION: Final = "langmani-v2-phase2b4-f0-reward-v0"
SEED_SCHEMA_VERSION: Final = "langmani-v2-phase2b4-f0-seeds-v0"
PPO_SCHEMA_VERSION: Final = "langmani-v2-phase2b4-f0-ppo-v0"
F0_ALLOWED_OPERATIONS: Final = frozenset(
    {
        "contract_audit",
        "gpu_pipeline_smoke",
        "ppo_micro_training",
        "ppo_micro_evaluation",
        "ppo_feasibility_development_evaluation",
    }
)
F0_PROHIBITED_OPERATIONS: Final = frozenset(
    {
        "demonstration_collection",
        "lerobot_export",
        "student_training",
        "smolvla_training",
        "act_training",
        "vla_jepa_training",
        "ppo_full_training",
        "expert_qualification",
        "formal_qualification_seed_access",
    }
)


@dataclass(frozen=True, slots=True)
class ObservationField:
    name: str
    width: int
    scale: tuple[float, ...]
    privileged: bool = True

    def __post_init__(self) -> None:
        if self.width <= 0 or len(self.scale) != self.width:
            raise ValueError(f"invalid observation field {self.name!r}")
        if any(not math.isfinite(value) or value <= 0 for value in self.scale):
            raise ValueError(f"observation scale must be positive and finite for {self.name!r}")


def _repeat(value: float, count: int) -> tuple[float, ...]:
    return (value,) * count


TEACHER_OBSERVATION_FIELDS: Final = (
    ObservationField("panda_qpos", 9, _repeat(3.2, 7) + _repeat(0.04, 2)),
    ObservationField("panda_qvel", 9, _repeat(5.0, 9)),
    ObservationField("tcp_pose", 7, _repeat(1.0, 7)),
    ObservationField("object_poses", 14, _repeat(1.0, 14)),
    ObservationField("object_linear_velocities", 6, _repeat(1.0, 6)),
    ObservationField("object_angular_velocities", 6, _repeat(10.0, 6)),
    ObservationField("target_region_center", 3, _repeat(1.0, 3)),
    ObservationField("target_region_geometry", 1, (1.0,)),
    ObservationField("target_object_geometry", 2, _repeat(1.0, 2)),
    ObservationField("target_object_one_hot", 2, _repeat(1.0, 2)),
    ObservationField("target_region_one_hot", 4, _repeat(1.0, 4)),
    ObservationField("difficulty_one_hot", 2, _repeat(1.0, 2)),
    ObservationField("target_to_goal", 3, _repeat(1.0, 3)),
    ObservationField("tcp_to_target", 3, _repeat(1.0, 3)),
    ObservationField("distractor_to_target", 3, _repeat(1.0, 3)),
    ObservationField("tcp_to_distractor", 3, _repeat(1.0, 3)),
    ObservationField("target_workspace_margins", 4, _repeat(1.0, 4)),
    ObservationField("contact_state", 2, _repeat(1.0, 2)),
    ObservationField("lift_and_alignment", 2, _repeat(1.0, 2)),
    ObservationField("normalized_elapsed_time", 1, (1.0,)),
    ObservationField("containment_margin", 1, (1.0,)),
)
TEACHER_OBSERVATION_DIM: Final = sum(field.width for field in TEACHER_OBSERVATION_FIELDS)
PANDA_QPOS_SLICE: Final = slice(0, 9)

FUTURE_STUDENT_OBSERVATION_FIELDS: Final = (
    "observation.images.base_camera",
    "observation.state.panda_qpos",
)
FUTURE_STUDENT_STATE_DIM: Final = 9
FORBIDDEN_STUDENT_PRIVILEGED_FIELDS: Final = frozenset(
    field.name for field in TEACHER_OBSERVATION_FIELDS if field.privileged
) - {"panda_qpos"}

_NORMALIZATION_SCALE: Final = torch.tensor(
    [value for field in TEACHER_OBSERVATION_FIELDS for value in field.scale],
    dtype=torch.float32,
)


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def assert_f0_operation_allowed(operation: str) -> None:
    if operation not in F0_ALLOWED_OPERATIONS:
        status = (
            "explicitly prohibited" if operation in F0_PROHIBITED_OPERATIONS else "not allowlisted"
        )
        raise ValueError(f"Phase 2B.4-F0 operation {operation!r} is {status}")


def teacher_observation_manifest() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "dimension": TEACHER_OBSERVATION_DIM,
        "dtype": "float32",
        "current_state_only": True,
        "fields": [
            {
                "name": field.name,
                "width": field.width,
                "scale": list(field.scale),
                "privileged": field.privileged,
            }
            for field in TEACHER_OBSERVATION_FIELDS
        ],
        "excluded": [
            "future_states",
            "scene_seed",
            "expert_action",
            "optimal_contact_point",
            "scripted_phase",
            "future_success",
            "qualification_outcome",
            "target_action",
            "direct_object_control",
        ],
    }
    payload["fingerprint"] = canonical_json_sha256(payload)
    return payload


def student_privilege_exclusion_audit() -> dict[str, object]:
    student_leaf_names = {
        item.rsplit(".", maxsplit=1)[-1] for item in FUTURE_STUDENT_OBSERVATION_FIELDS
    }
    leaked = sorted(student_leaf_names & FORBIDDEN_STUDENT_PRIVILEGED_FIELDS)
    payload: dict[str, object] = {
        "schema_version": "langmani-v2-phase2b4-f0-student-privilege-exclusion-v0",
        "student_fields": list(FUTURE_STUDENT_OBSERVATION_FIELDS),
        "student_state_dimension": FUTURE_STUDENT_STATE_DIM,
        "teacher_privileged_fields": sorted(FORBIDDEN_STUDENT_PRIVILEGED_FIELDS),
        "leaked_fields": leaked,
        "passed": not leaked,
        "student_normalization_statistics_created": False,
    }
    payload["fingerprint"] = canonical_json_sha256(payload)
    return payload


def concatenate_teacher_fields(values: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Validate and concatenate already-computed current-state teacher fields."""

    expected = {field.name for field in TEACHER_OBSERVATION_FIELDS}
    if set(values) != expected:
        missing = sorted(expected - set(values))
        extra = sorted(set(values) - expected)
        raise ValueError(f"teacher field mismatch: missing={missing}, extra={extra}")
    batch_size: int | None = None
    flattened: list[torch.Tensor] = []
    for field in TEACHER_OBSERVATION_FIELDS:
        value = values[field.name]
        if value.ndim != 2 or value.shape[1] != field.width:
            raise ValueError(f"teacher field {field.name!r} must have shape (batch, {field.width})")
        if batch_size is None:
            batch_size = int(value.shape[0])
        elif int(value.shape[0]) != batch_size:
            raise ValueError("teacher observation fields must share a batch size")
        flattened.append(value.to(dtype=torch.float32))
    observation = torch.cat(flattened, dim=1)
    if observation.shape[1] != TEACHER_OBSERVATION_DIM:
        raise AssertionError("teacher observation dimension drift")
    if not torch.isfinite(observation).all():
        raise ValueError("teacher observation contains non-finite values")
    return observation


def normalize_teacher_observation(observation: torch.Tensor) -> torch.Tensor:
    if observation.ndim != 2 or observation.shape[1] != TEACHER_OBSERVATION_DIM:
        raise ValueError(f"teacher observation must have shape (batch, {TEACHER_OBSERVATION_DIM})")
    scale = _NORMALIZATION_SCALE.to(device=observation.device)
    normalized = observation.to(torch.float32) / scale
    if not torch.isfinite(normalized).all():
        raise ValueError("normalized teacher observation contains non-finite values")
    return normalized


def normalization_manifest() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "langmani-v2-phase2b4-f0-state-normalization-v0",
        "observation_schema": OBSERVATION_SCHEMA_VERSION,
        "method": "fixed_componentwise_scale_without_clipping",
        "fit_from_student_data": False,
        "online_statistics": False,
        "scales": {field.name: list(field.scale) for field in TEACHER_OBSERVATION_FIELDS},
    }
    payload["fingerprint"] = canonical_json_sha256(payload)
    return payload


def assert_student_observation_has_no_privilege(fields: Sequence[str]) -> None:
    leaf_names = {field.rsplit(".", maxsplit=1)[-1] for field in fields}
    leaked = sorted(leaf_names & FORBIDDEN_STUDENT_PRIVILEGED_FIELDS)
    if leaked:
        raise ValueError(
            "privileged teacher fields leaked into student schema: " + ", ".join(leaked)
        )


@dataclass(frozen=True, slots=True)
class RewardInputs:
    previous_target_distance: torch.Tensor
    target_distance: torch.Tensor
    previous_behind_distance: torch.Tensor
    behind_distance: torch.Tensor
    previous_containment_fraction: torch.Tensor
    containment_fraction: torch.Tensor
    target_contact_onset: torch.Tensor
    push_alignment: torch.Tensor
    wrong_object_contact: torch.Tensor
    wrong_object_displaced: torch.Tensor
    target_outside_workspace: torch.Tensor
    target_lifted: torch.Tensor
    target_toppled: torch.Tensor
    target_is_grasped: torch.Tensor
    invalid_action: torch.Tensor
    action_out_of_bounds: torch.Tensor
    robot_collision: torch.Tensor
    success: torch.Tensor
    normalized_action: torch.Tensor
    previous_normalized_action: torch.Tensor


@dataclass(frozen=True, slots=True)
class RewardOutput:
    total: torch.Tensor
    components: Mapping[str, torch.Tensor]


def compute_phase2b4_reward(inputs: RewardInputs) -> RewardOutput:
    """Compute a small potential-based dense reward plus terminal/safety terms."""

    batch = inputs.target_distance.shape
    scalar_fields = (
        inputs.previous_target_distance,
        inputs.previous_behind_distance,
        inputs.behind_distance,
        inputs.previous_containment_fraction,
        inputs.containment_fraction,
        inputs.push_alignment,
    )
    boolean_fields = (
        inputs.target_contact_onset,
        inputs.wrong_object_contact,
        inputs.wrong_object_displaced,
        inputs.target_outside_workspace,
        inputs.target_lifted,
        inputs.target_toppled,
        inputs.target_is_grasped,
        inputs.invalid_action,
        inputs.action_out_of_bounds,
        inputs.robot_collision,
        inputs.success,
    )
    if any(value.shape != batch for value in scalar_fields + boolean_fields):
        raise ValueError("reward scalar fields must share shape (batch,)")
    if inputs.normalized_action.ndim != 2 or inputs.normalized_action.shape[0] != batch[0]:
        raise ValueError("normalized_action must have shape (batch, action)")
    if inputs.previous_normalized_action.shape != inputs.normalized_action.shape:
        raise ValueError("previous_normalized_action must match normalized_action")

    dtype = torch.float32
    device = inputs.target_distance.device
    step = torch.full(batch, -0.01, dtype=dtype, device=device)
    target_progress = 20.0 * (inputs.previous_target_distance - inputs.target_distance)
    behind_progress = 2.0 * (inputs.previous_behind_distance - inputs.behind_distance)
    containment_progress = 2.0 * (
        inputs.containment_fraction - inputs.previous_containment_fraction
    )
    correct_contact = (
        0.05
        * inputs.target_contact_onset.to(dtype)
        * torch.clamp(inputs.push_alignment, min=0.0, max=1.0)
    )
    wrong_side_contact = (
        -0.10
        * inputs.target_contact_onset.to(dtype)
        * torch.clamp(-inputs.push_alignment, min=0.0, max=1.0)
    )
    action_change = -0.01 * torch.mean(
        (inputs.normalized_action - inputs.previous_normalized_action).square(), dim=1
    )
    saturation = -0.02 * torch.mean(
        torch.relu(torch.abs(inputs.normalized_action) - 0.95) / 0.05, dim=1
    )
    terminal_success = 10.0 * inputs.success.to(dtype)
    safety = (
        -10.0 * inputs.wrong_object_contact.to(dtype)
        - 10.0 * inputs.wrong_object_displaced.to(dtype)
        - 12.0 * inputs.target_outside_workspace.to(dtype)
        - 12.0 * inputs.target_lifted.to(dtype)
        - 12.0 * inputs.target_toppled.to(dtype)
        - 12.0 * inputs.target_is_grasped.to(dtype)
        - 20.0 * inputs.invalid_action.to(dtype)
        - 20.0 * inputs.action_out_of_bounds.to(dtype)
        - 5.0 * inputs.robot_collision.to(dtype)
    )
    components = {
        "step": step,
        "target_progress": target_progress,
        "behind_progress": behind_progress,
        "containment_progress": containment_progress,
        "correct_contact_onset": correct_contact,
        "wrong_side_contact": wrong_side_contact,
        "action_change": action_change,
        "saturation": saturation,
        "terminal_success": terminal_success,
        "safety": safety,
    }
    total = torch.stack(tuple(components.values()), dim=0).sum(dim=0)
    if not torch.isfinite(total).all():
        raise ValueError("reward contains non-finite values")
    return RewardOutput(total=total, components=components)


def reward_manifest() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": REWARD_SCHEMA_VERSION,
        "terminal_success_predicate": "unchanged LangMani-PushToRegion-v0 stable success",
        "components": {
            "step": "-0.01",
            "target_progress": "20 * (previous_distance - current_distance)",
            "behind_progress": "2 * (previous_behind_distance - current_behind_distance)",
            "containment_progress": "2 * (current_fraction - previous_fraction)",
            "correct_contact_onset": "0.05 * onset * max(push_alignment, 0)",
            "wrong_side_contact": "-0.10 * onset * max(-push_alignment, 0)",
            "action_change": "-0.01 * mean((a_t - a_t-1)^2)",
            "saturation": "-0.02 * mean(relu(abs(a_normalized)-0.95)/0.05)",
            "terminal_success": "10 * canonical_success",
            "safety": {
                "wrong_object_contact": -10.0,
                "wrong_object_displaced": -10.0,
                "target_outside_workspace": -12.0,
                "target_lifted": -12.0,
                "target_toppled": -12.0,
                "target_is_grasped": -12.0,
                "invalid_action": -20.0,
                "action_out_of_bounds": -20.0,
                "robot_collision": -5.0,
            },
        },
        "potential_based_dense_terms": [
            "target_progress",
            "behind_progress",
            "containment_progress",
        ],
        "direct_object_control": False,
        "reward_revision_limit": 1,
        "reward_revision_used": 0,
    }
    payload["fingerprint"] = canonical_json_sha256(payload)
    return payload


def _scenario_reward(
    *,
    target_delta: float = 0.0,
    behind_delta: float = 0.0,
    containment_delta: float = 0.0,
    success: bool = False,
    wrong_object: bool = False,
    outside: bool = False,
    lifted: bool = False,
    toppled: bool = False,
    contact_onset: bool = False,
    alignment: float = 1.0,
    action: float = 0.0,
    previous_action: float = 0.0,
) -> float:
    def scalar(value: float) -> torch.Tensor:
        return torch.tensor([value], dtype=torch.float32)

    def flag(value: bool) -> torch.Tensor:
        return torch.tensor([value], dtype=torch.bool)

    result = compute_phase2b4_reward(
        RewardInputs(
            previous_target_distance=scalar(0.4),
            target_distance=scalar(0.4 - target_delta),
            previous_behind_distance=scalar(0.2),
            behind_distance=scalar(0.2 - behind_delta),
            previous_containment_fraction=scalar(0.0),
            containment_fraction=scalar(containment_delta),
            target_contact_onset=flag(contact_onset),
            push_alignment=scalar(alignment),
            wrong_object_contact=flag(wrong_object),
            wrong_object_displaced=flag(wrong_object),
            target_outside_workspace=flag(outside),
            target_lifted=flag(lifted),
            target_toppled=flag(toppled),
            target_is_grasped=flag(False),
            invalid_action=flag(False),
            action_out_of_bounds=flag(False),
            robot_collision=flag(False),
            success=flag(success),
            normalized_action=torch.full((1, 8), action),
            previous_normalized_action=torch.full((1, 8), previous_action),
        )
    )
    return float(result.total.item())


def audit_reward_hacking_scenarios() -> dict[str, object]:
    scenarios = {
        "no_movement": 20 * _scenario_reward(),
        "hovering_near_object": _scenario_reward(behind_delta=0.08) + 19 * _scenario_reward(),
        "approaching_wrong_side": _scenario_reward(
            behind_delta=-0.05, contact_onset=True, alignment=-1.0
        ),
        "repeated_tapping": _scenario_reward(contact_onset=True) + 19 * _scenario_reward(),
        "moving_away": _scenario_reward(target_delta=-0.08),
        "wrong_object_interaction": _scenario_reward(wrong_object=True),
        "outside_workspace": _scenario_reward(target_delta=0.1, outside=True),
        "lifting": _scenario_reward(lifted=True),
        "toppling": _scenario_reward(toppled=True),
        "oscillating": 10
        * (
            _scenario_reward(target_delta=0.01, action=0.8, previous_action=-0.8)
            + _scenario_reward(target_delta=-0.01, action=-0.8, previous_action=0.8)
        ),
        "saturating": 20 * _scenario_reward(action=1.0, previous_action=1.0),
        "brief_unstable_target": _scenario_reward(
            target_delta=0.3, containment_delta=0.9, success=False
        ),
        "canonical_success": _scenario_reward(
            target_delta=0.35,
            behind_delta=0.12,
            containment_delta=1.0,
            success=True,
            contact_onset=True,
            alignment=1.0,
        ),
    }
    canonical = scenarios["canonical_success"]
    failure_values = [value for key, value in scenarios.items() if key != "canonical_success"]
    checks = {
        "canonical_success_highest": canonical > max(failure_values),
        "failure_not_comparable_to_success": max(failure_values) < 0.75 * canonical,
        "wrong_object_not_profitable": scenarios["wrong_object_interaction"] < 0,
        "outside_workspace_not_profitable": scenarios["outside_workspace"] < 0,
        "hovering_below_progress": scenarios["hovering_near_object"] < canonical,
        "oscillation_not_profitable": scenarios["oscillating"] < 0,
    }
    payload: dict[str, object] = {
        "schema_version": "langmani-v2-phase2b4-f0-reward-hacking-audit-v0",
        "reward_schema": REWARD_SCHEMA_VERSION,
        "scenarios": scenarios,
        "checks": checks,
        "passed": all(checks.values()),
        "reward_revision_used": 0,
    }
    payload["fingerprint"] = canonical_json_sha256(payload)
    return payload


@dataclass(frozen=True, slots=True)
class SeedNamespace:
    name: str
    start: int
    stop: int
    accessible_in_f0: bool

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise ValueError(f"invalid seed namespace {self.name!r}")

    @property
    def values(self) -> range:
        return range(self.start, self.stop)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "start_inclusive": self.start,
            "stop_exclusive": self.stop,
            "count": self.stop - self.start,
            "accessible_in_f0": self.accessible_in_f0,
        }


SEED_NAMESPACES: Final = (
    SeedNamespace("formal_qualification", 66300, 66400, False),
    SeedNamespace("ppo_train", 700000, 800000, True),
    SeedNamespace("ppo_micro_evaluation", 800000, 800060, True),
    SeedNamespace("ppo_feasibility_development", 810000, 810060, True),
    SeedNamespace("future_ppo_full_development", 820000, 920000, False),
)


def seed_disjointness_audit() -> dict[str, object]:
    overlaps: list[dict[str, object]] = []
    for index, left in enumerate(SEED_NAMESPACES):
        for right in SEED_NAMESPACES[index + 1 :]:
            start = max(left.start, right.start)
            stop = min(left.stop, right.stop)
            if start < stop:
                overlaps.append(
                    {"left": left.name, "right": right.name, "start": start, "stop": stop}
                )
    f0_names = {"ppo_train", "ppo_micro_evaluation", "ppo_feasibility_development"}
    f0_values = {
        seed
        for namespace in SEED_NAMESPACES
        if namespace.name in f0_names
        for seed in namespace.values
    }
    formal_accessed = bool(f0_values & FORMAL_QUALIFICATION_SEEDS)
    payload: dict[str, object] = {
        "schema_version": SEED_SCHEMA_VERSION,
        "namespaces": [namespace.to_dict() for namespace in SEED_NAMESPACES],
        "overlaps": overlaps,
        "formal_seed_accessed": formal_accessed,
        "passed": not overlaps and not formal_accessed,
    }
    payload["fingerprint"] = canonical_json_sha256(payload)
    return payload


def assert_f0_seed_allowed(seed: int, namespace: str) -> None:
    if seed in FORMAL_QUALIFICATION_SEEDS:
        raise ValueError(f"formal qualification seed {seed} is sealed")
    matches = [item for item in SEED_NAMESPACES if item.name == namespace]
    if len(matches) != 1 or not matches[0].accessible_in_f0:
        raise ValueError(f"seed namespace {namespace!r} is not accessible in F0")
    if seed not in matches[0].values:
        raise ValueError(f"seed {seed} is outside namespace {namespace!r}")


@dataclass(frozen=True, slots=True)
class PPOConfig:
    seed: int = 240400
    num_envs: int = 128
    num_steps: int = 32
    total_timesteps: int = 1_048_576
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 8
    update_epochs: int = 4
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.001
    max_grad_norm: float = 0.5
    target_kl: float = 0.1
    hidden_dim: int = 256
    initial_log_std: float = -2.5

    def __post_init__(self) -> None:
        if self.num_envs < 1 or self.num_steps < 1 or self.total_timesteps < 1:
            raise ValueError("PPO rollout sizes must be positive")
        if self.total_timesteps % self.batch_size != 0:
            raise ValueError("total_timesteps must be divisible by num_envs * num_steps")
        if self.batch_size % self.num_minibatches != 0:
            raise ValueError("PPO batch size must be divisible by num_minibatches")

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def iterations(self) -> int:
        return self.total_timesteps // self.batch_size

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "batch_size": self.batch_size,
            "minibatch_size": self.minibatch_size,
            "iterations": self.iterations,
        }


def _layer_init(layer: nn.Linear, *, std: float = math.sqrt(2.0)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class BoundedActorCritic(nn.Module):
    """PPO actor/critic with one shared, structurally bounded action transform."""

    action_low: torch.Tensor
    action_high: torch.Tensor
    action_scale: torch.Tensor
    action_bias: torch.Tensor
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
        initial_log_std: float = -2.5,
    ) -> None:
        super().__init__()
        if observation_dim != TEACHER_OBSERVATION_DIM:
            raise ValueError(f"observation_dim must be {TEACHER_OBSERVATION_DIM}")
        low = action_low.detach().to(dtype=torch.float32).reshape(-1)
        high = action_high.detach().to(dtype=torch.float32).reshape(-1)
        if low.shape != (8,) or high.shape != (8,):
            raise ValueError("Phase 2B.4 action bounds must have shape (8,)")
        if (
            not torch.isfinite(low).all()
            or not torch.isfinite(high).all()
            or not torch.all(low < high)
        ):
            raise ValueError("Phase 2B.4 action bounds must be finite and ordered")
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)
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

    def _reference_latent(self, observation: torch.Tensor) -> torch.Tensor:
        qpos = observation[:, PANDA_QPOS_SLICE]
        reference = torch.cat(
            (qpos[:, :7], torch.full((qpos.shape[0], 1), -0.9, device=qpos.device)),
            dim=1,
        )
        unit = (reference - self.action_bias) / self.action_scale
        # This bounds the mean parameterization, not an emitted action. Every emitted
        # action still comes exclusively from tanh followed by the authoritative affine map.
        safe_unit = unit.clamp(min=-0.999, max=0.999)
        return torch.atanh(safe_unit)

    def distribution(self, observation: torch.Tensor) -> Normal:
        normalized = normalize_teacher_observation(observation)
        mean = self._reference_latent(observation) + self.actor(normalized)
        std = torch.exp(self.actor_log_std).expand_as(mean)
        return Normal(mean, std)

    def action_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return (self.action_bias + self.action_scale * torch.tanh(latent)).to(torch.float32)

    def log_prob(self, distribution: Normal, latent: torch.Tensor) -> torch.Tensor:
        unit = torch.tanh(latent)
        log_jacobian = torch.log(self.action_scale) + torch.log1p(-unit.square() + 1e-6)
        return (distribution.log_prob(latent) - log_jacobian).sum(dim=1)

    def action_and_value(
        self,
        observation: torch.Tensor,
        *,
        latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation)
        selected_latent = distribution.rsample() if latent is None else latent
        action = self.action_from_latent(selected_latent)
        log_prob = self.log_prob(distribution, selected_latent)
        sampled_entropy = -log_prob
        value = self.critic(normalize_teacher_observation(observation)).squeeze(1)
        return action, selected_latent, log_prob, sampled_entropy, value

    def deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        return self.action_from_latent(self.distribution(observation).mean)

    def value(self, observation: torch.Tensor) -> torch.Tensor:
        return self.critic(normalize_teacher_observation(observation)).squeeze(1)

    def action_manifest(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": ACTION_SCHEMA_VERSION,
            "dtype": "float32",
            "shape": [8],
            "control_mode": "pd_joint_pos",
            "bounds_source": "active environment single_action_space",
            "low": self.action_low.detach().cpu().tolist(),
            "high": self.action_high.detach().cpu().tolist(),
            "transform": "Normal latent -> tanh -> affine(low, high)",
            "stochastic_and_deterministic_transform_shared": True,
            "clipping": False,
            "projection": False,
            "rejection_sampling": False,
            "fallback": False,
        }
        payload["fingerprint"] = canonical_json_sha256(payload)
        return payload


def compute_gae(
    *,
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE while bootstrapping truncations and stopping at terminations."""

    if rewards.shape != values.shape or rewards.shape != terminated.shape:
        raise ValueError("GAE rollout tensors must share shape (steps, envs)")
    if truncated.shape != rewards.shape or next_values.shape != rewards.shape:
        raise ValueError("GAE next-value and truncation tensors must match rewards")
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(rewards.shape[1], device=rewards.device)
    for index in reversed(range(rewards.shape[0])):
        bootstrap_allowed = (~terminated[index]).to(rewards.dtype)
        continuation = (~(terminated[index] | truncated[index])).to(rewards.dtype)
        delta = rewards[index] + gamma * next_values[index] * bootstrap_allowed - values[index]
        last_advantage = delta + gamma * gae_lambda * continuation * last_advantage
        advantages[index] = last_advantage
    return advantages, advantages + values


@dataclass(frozen=True, slots=True)
class PPOLoss:
    total: torch.Tensor
    actor: torch.Tensor
    critic: torch.Tensor
    entropy: torch.Tensor
    approximate_kl: torch.Tensor
    clip_fraction: torch.Tensor


def compute_ppo_loss(
    *,
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    entropy: torch.Tensor,
    new_value: torch.Tensor,
    old_value: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    clip_coef: float,
    value_coef: float,
    entropy_coef: float,
) -> PPOLoss:
    normalized_advantage = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )
    log_ratio = new_log_prob - old_log_prob
    ratio = log_ratio.exp()
    actor_unclipped = -normalized_advantage * ratio
    actor_clipped = -normalized_advantage * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    actor = torch.maximum(actor_unclipped, actor_clipped).mean()
    critic = 0.5 * (new_value - returns).square().mean()
    entropy_mean = entropy.mean()
    total = actor + value_coef * critic - entropy_coef * entropy_mean
    with torch.no_grad():
        approximate_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = ((ratio - 1.0).abs() > clip_coef).to(torch.float32).mean()
    return PPOLoss(
        total=total,
        actor=actor,
        critic=critic,
        entropy=entropy_mean,
        approximate_kl=approximate_kl,
        clip_fraction=clip_fraction,
    )


def audit_sampled_action_legality(
    policy: BoundedActorCritic,
    *,
    sample_count: int = 100_000,
    batch_size: int = 4096,
    seed: int = 240401,
) -> dict[str, object]:
    if sample_count < 100_000:
        raise ValueError("the Phase 2B.4 static action audit requires at least 100,000 samples")
    device = policy.action_low.device
    generator = torch.Generator(device=device).manual_seed(seed)
    nonfinite = 0
    out_of_bounds = 0
    observed = 0
    min_action = torch.full((8,), torch.inf, device=device)
    max_action = torch.full((8,), -torch.inf, device=device)
    while observed < sample_count:
        count = min(batch_size, sample_count - observed)
        observation = torch.randn(
            (count, TEACHER_OBSERVATION_DIM), generator=generator, device=device
        )
        low = policy.action_low
        high = policy.action_high
        qpos_unit = torch.rand((count, 7), generator=generator, device=device)
        observation[:, :7] = low[:7] + qpos_unit * (high[:7] - low[:7])
        with torch.no_grad():
            distribution = policy.distribution(observation)
            noise = torch.randn(
                distribution.mean.shape,
                dtype=distribution.mean.dtype,
                device=device,
                generator=generator,
            )
            latent = distribution.mean + distribution.stddev * noise
            action = policy.action_from_latent(latent)
        nonfinite += int((~torch.isfinite(action)).any(dim=1).sum().item())
        out_of_bounds += int(((action < low) | (action > high)).any(dim=1).sum().item())
        min_action = torch.minimum(min_action, action.amin(dim=0))
        max_action = torch.maximum(max_action, action.amax(dim=0))
        observed += count
    payload: dict[str, object] = {
        "schema_version": "langmani-v2-phase2b4-f0-action-legality-v0",
        "action_schema": ACTION_SCHEMA_VERSION,
        "sample_count": observed,
        "nonfinite_actions": nonfinite,
        "out_of_bounds_actions": out_of_bounds,
        "hidden_clipping_events": 0,
        "hidden_projection_events": 0,
        "observed_min": min_action.cpu().tolist(),
        "observed_max": max_action.cpu().tolist(),
        "passed": nonfinite == 0 and out_of_bounds == 0,
    }
    payload["fingerprint"] = canonical_json_sha256(payload)
    return payload


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    episode_count: int
    successes: int
    cube_episodes: int
    cube_successes: int
    cylinder_episodes: int
    cylinder_successes: int
    direction_episodes: Mapping[str, int]
    direction_successes: Mapping[str, int]
    standard_episodes: int
    standard_successes: int
    hard_episodes: int
    hard_successes: int
    safety_counts: Mapping[str, int]

    @property
    def success_rate(self) -> float:
        return self.successes / self.episode_count if self.episode_count else 0.0

    @property
    def cube_success_rate(self) -> float:
        return self.cube_successes / self.cube_episodes if self.cube_episodes else 0.0

    @property
    def cylinder_success_rate(self) -> float:
        return self.cylinder_successes / self.cylinder_episodes if self.cylinder_episodes else 0.0

    @property
    def standard_success_rate(self) -> float:
        return self.standard_successes / self.standard_episodes if self.standard_episodes else 0.0

    @property
    def hard_success_rate(self) -> float:
        return self.hard_successes / self.hard_episodes if self.hard_episodes else 0.0

    @property
    def zero_tolerance_total(self) -> int:
        return sum(self.safety_counts.values())


def micro_gate(summary: EvaluationSummary) -> dict[str, bool]:
    return {
        "episode_count_40_to_60": 40 <= summary.episode_count <= 60,
        "overall_at_least_70_percent": summary.success_rate >= 0.70,
        "cube_at_least_70_percent": summary.cube_success_rate >= 0.70,
        "cylinder_at_least_50_percent": summary.cylinder_success_rate >= 0.50,
        "success_in_every_included_direction": all(
            summary.direction_successes.get(direction, 0) >= 1
            for direction, count in summary.direction_episodes.items()
            if count > 0
        ),
        "zero_tolerance_safety_clean": summary.zero_tolerance_total == 0,
    }


def feasibility_gate(summary: EvaluationSummary) -> dict[str, bool]:
    return {
        "episode_count_is_60": summary.episode_count == 60,
        "overall_at_least_50_percent": summary.success_rate >= 0.50,
        "cube_at_least_60_percent": summary.cube_success_rate >= 0.60,
        "cylinder_at_least_30_percent": summary.cylinder_success_rate >= 0.30,
        "success_in_every_canonical_direction": all(
            summary.direction_successes.get(direction, 0) >= 1
            for direction in ("left", "right", "forward_left", "forward_right")
        ),
        "zero_tolerance_safety_clean": summary.zero_tolerance_total == 0,
    }


class Phase2B4Result(StrEnum):
    RESULT_A = "RESULT_A"
    RESULT_B = "RESULT_B"
    RESULT_C = "RESULT_C"
    RESULT_D = "RESULT_D"


def classify_phase2b4_result(
    *,
    pipeline_valid: bool,
    reward_valid: bool,
    action_valid: bool,
    micro_passed: bool,
    feasibility_development_ran: bool,
    feasibility_development_passed: bool,
) -> Phase2B4Result:
    if not (pipeline_valid and reward_valid and action_valid):
        return Phase2B4Result.RESULT_D
    if not micro_passed:
        return Phase2B4Result.RESULT_C
    if not feasibility_development_ran or not feasibility_development_passed:
        return Phase2B4Result.RESULT_B
    return Phase2B4Result.RESULT_A


def authorization_state(result: Phase2B4Result) -> dict[str, object]:
    eligible = result is Phase2B4Result.RESULT_A
    return {
        "schema_version": "langmani-v2-phase2b4-f0-authorization-v0",
        "result": result.value,
        "ppo_teacher_feasibility_validated": True
        if eligible
        else ("partial" if result is Phase2B4Result.RESULT_B else False),
        "ppo_full_training_eligible": eligible,
        "ppo_full_training_authorized": False,
        "expert_qualification_authorized": False,
        "data_collection_authorized": False,
        "smolvla_training_authorized": False,
        "training_started_for_student_policy": False,
        "demonstration_source_validated": False,
    }


def checkpoint_payload(
    *,
    policy: BoundedActorCritic,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    global_step: int,
    observation_manifest_fingerprint: str,
    action_manifest_fingerprint: str,
    reward_manifest_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": PPO_SCHEMA_VERSION,
        "global_step": global_step,
        "config": config.to_dict(),
        "policy_state": policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "observation_manifest_fingerprint": observation_manifest_fingerprint,
        "action_manifest_fingerprint": action_manifest_fingerprint,
        "reward_manifest_fingerprint": reward_manifest_fingerprint,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def save_checkpoint(path: Path, payload: Mapping[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return "sha256:" + digest


def load_checkpoint(path: Path) -> dict[str, object]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or value.get("schema_version") != PPO_SCHEMA_VERSION:
        raise ValueError("invalid Phase 2B.4 PPO checkpoint")
    return value


def reconstruct_policy_from_checkpoint(path: Path) -> tuple[BoundedActorCritic, dict[str, object]]:
    payload = load_checkpoint(path)
    policy_state = payload.get("policy_state")
    if not isinstance(policy_state, Mapping):
        raise ValueError("checkpoint policy_state is missing")
    low = policy_state.get("action_low")
    high = policy_state.get("action_high")
    if not isinstance(low, torch.Tensor) or not isinstance(high, torch.Tensor):
        raise ValueError("checkpoint action bounds are missing")
    config_raw = payload.get("config")
    if not isinstance(config_raw, Mapping):
        raise ValueError("checkpoint PPO config is missing")
    policy = BoundedActorCritic(
        observation_dim=TEACHER_OBSERVATION_DIM,
        action_low=low,
        action_high=high,
        hidden_dim=int(config_raw["hidden_dim"]),
        initial_log_std=float(config_raw["initial_log_std"]),
    )
    policy.load_state_dict(policy_state, strict=True)
    policy.eval()
    return policy, payload


def verify_source_identity(*, parent_sha: str, geometry_is_ancestor: bool) -> None:
    if parent_sha != SOURCE_COMMIT:
        raise ValueError(f"Phase 2B.4 must descend directly from {SOURCE_COMMIT}")
    if geometry_is_ancestor:
        raise ValueError("the excluded geometry result is an ancestor of the F0 branch")


__all__ = [
    "ACTION_SCHEMA_VERSION",
    "EXCLUDED_GEOMETRY_COMMIT",
    "F0_ALLOWED_OPERATIONS",
    "F0_PROHIBITED_OPERATIONS",
    "FORMAL_QUALIFICATION_SEEDS",
    "FUTURE_STUDENT_OBSERVATION_FIELDS",
    "FUTURE_STUDENT_STATE_DIM",
    "OBSERVATION_SCHEMA_VERSION",
    "OFFICIAL_PPO_REVISION",
    "PPOConfig",
    "PPO_SCHEMA_VERSION",
    "PANDA_QPOS_SLICE",
    "Phase2B4Result",
    "RewardInputs",
    "RewardOutput",
    "SEED_NAMESPACES",
    "SOURCE_COMMIT",
    "TEACHER_OBSERVATION_DIM",
    "TEACHER_OBSERVATION_FIELDS",
    "BoundedActorCritic",
    "EvaluationSummary",
    "PPOLoss",
    "assert_f0_seed_allowed",
    "assert_f0_operation_allowed",
    "assert_student_observation_has_no_privilege",
    "audit_reward_hacking_scenarios",
    "audit_sampled_action_legality",
    "authorization_state",
    "canonical_json_sha256",
    "checkpoint_payload",
    "classify_phase2b4_result",
    "compute_gae",
    "compute_phase2b4_reward",
    "compute_ppo_loss",
    "concatenate_teacher_fields",
    "feasibility_gate",
    "load_checkpoint",
    "micro_gate",
    "normalization_manifest",
    "normalize_teacher_observation",
    "reconstruct_policy_from_checkpoint",
    "reward_manifest",
    "save_checkpoint",
    "seed_disjointness_audit",
    "student_privilege_exclusion_audit",
    "teacher_observation_manifest",
    "verify_source_identity",
]
