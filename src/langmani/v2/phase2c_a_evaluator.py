"""Policy-agnostic official-ManiSkill evaluator for Phase 2C-A.

ACT-specific loading and preprocessing remain in the adapter.  This evaluator
accepts only the generic :class:`PolicyAdapter`, extracts the deployable RGB
and Panda state features, hard-rejects invalid actions without clipping or
projection, and records simulator-only diagnostics outside the policy batch.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from langmani.datasets.identity import sha256_hex
from langmani.datasets.lerobot_types import IMAGE_FEATURE_KEY, STATE_FEATURE_KEY
from langmani.datasets.observation_reconstruction import extract_base_camera_rgb
from langmani.datasets.policy_state import extract_panda_policy_state_v0
from langmani.policies.act_training import image_to_policy_float
from langmani.v2.phase2b5 import PANDA_ACTION_HIGH, PANDA_ACTION_LOW
from langmani.v2.phase2b6 import apply_visual_shift
from langmani.v2.phase2c_a import (
    CONTROL_FREQUENCY_HZ,
    EXECUTION_HORIZONS,
    FAILURE_CATEGORIES,
    PACKAGE_FINGERPRINT,
    SKILL_FAMILIES,
    TASK_IDS,
    VISUAL_SHIFT_EXPOSURE,
    VISUAL_SHIFT_IDENTITY,
    VISUAL_SHIFT_RGB_MULTIPLIERS,
    Phase2CAContractError,
    summarize_binary_results,
)
from langmani.v2.policy import ObservationBatch, PolicyAdapter, PolicyContext


class Phase2CAEvaluationError(RuntimeError):
    """Raised when policy/simulator evaluation violates the frozen contract."""


@dataclass(frozen=True, slots=True)
class Phase2CAEpisodeResult:
    evaluation_id: str
    split: str
    task_id: str
    skill_family: str
    policy_id: str
    checkpoint_identity: str
    source_episode_id: int
    reset_identity: str
    task_condition_id: str
    visual_transform: str | None
    execution_horizon: int
    outcome: str
    success: bool
    timeout: bool
    invalid_action: bool
    simulator_error: bool
    wrong_task_behavior: bool
    failure_category: str
    failure_reason: str | None
    episode_length: int
    policy_query_count: int
    actions_executed: int
    action_saturation_rate: float
    action_smoothness_mean_l2: float | None
    inference_latency_p50_ms: float | None
    inference_latency_p95_ms: float | None
    inference_latency_samples_ms: tuple[float, ...]
    environment_step_latency_p50_ms: float | None
    environment_step_latency_p95_ms: float | None
    post_contact_divergence_m: float | None
    initial_to_final_object_motion_m: float | None
    maximum_object_motion_m: float | None
    final_evaluation: Mapping[str, object]
    action_min: tuple[float, ...] | None
    action_max: tuple[float, ...] | None
    schema_version: str = "langmani-v2-phase2c-a-evaluation-episode-v0"

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        for key in (
            "inference_latency_samples_ms",
            "action_min",
            "action_max",
        ):
            if value[key] is not None:
                value[key] = list(value[key])
        value["final_evaluation"] = dict(self.final_evaluation)
        return value


@dataclass(frozen=True, slots=True)
class EvaluatedEpisode:
    result: Phase2CAEpisodeResult
    frames: tuple[np.ndarray, ...]


def _single_bool(value: object) -> bool | None:
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    array = np.asarray(candidate)
    if array.size != 1:
        return None
    if array.dtype.kind not in {"b", "i", "u", "f"}:
        return None
    return bool(array.reshape(-1)[0])


def _single_float(value: object) -> float | None:
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    array = np.asarray(candidate)
    if array.size != 1 or array.dtype.kind not in {"b", "i", "u", "f"}:
        return None
    result = float(array.reshape(-1)[0])
    return result if math.isfinite(result) else None


def _json_diagnostics(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        boolean = _single_bool(item)
        numeric = _single_float(item)
        if isinstance(item, (bool, np.bool_)) or (
            hasattr(item, "dtype") and str(getattr(item, "dtype", "")) == "torch.bool"
        ):
            if boolean is not None:
                result[key] = boolean
        elif numeric is not None:
            result[key] = numeric
    return result


def _evaluate_environment(base: object, fallback: object) -> dict[str, object]:
    evaluate = getattr(base, "evaluate", None)
    if callable(evaluate):
        try:
            return _json_diagnostics(evaluate())
        except Exception:  # noqa: BLE001 - fallback info is still recorded
            pass
    return _json_diagnostics(fallback)


def _success(value: Mapping[str, object]) -> bool:
    return value.get("success") is True


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all() or np.any(array < 0):
        raise Phase2CAEvaluationError("latency samples must be finite and non-negative")
    return float(np.percentile(array, fraction * 100.0))


def _object_positions(base: object, task_id: str) -> dict[str, np.ndarray]:
    names = {
        "PickCube-v1": ("cube",),
        "StackCube-v1": ("cubeA", "cubeB"),
        "PushCube-v1": ("obj",),
    }[task_id]
    result: dict[str, np.ndarray] = {}
    for name in names:
        actor = getattr(base, name, None)
        pose = getattr(actor, "pose", None)
        raw_pose = getattr(pose, "raw_pose", None)
        if raw_pose is None:
            return {}
        candidate = raw_pose
        detach = getattr(candidate, "detach", None)
        if callable(detach):
            candidate = detach()
        cpu = getattr(candidate, "cpu", None)
        if callable(cpu):
            candidate = cpu()
        array = np.asarray(candidate, dtype=np.float64)
        if array.ndim == 2 and array.shape[0] == 1:
            array = array[0]
        if array.shape != (7,) or not np.isfinite(array).all():
            return {}
        result[name] = array[:3].copy()
    return result


def _maximum_translation(
    reference: Mapping[str, np.ndarray],
    observed: Mapping[str, np.ndarray],
) -> float | None:
    if not reference or set(reference) != set(observed):
        return None
    return max(float(np.linalg.norm(reference[key] - observed[key])) for key in reference)


def _extract_observation(
    observation: object,
    *,
    base: object,
    visual_shift: bool,
) -> ObservationBatch:
    rgb = extract_base_camera_rgb(observation)
    if visual_shift:
        rgb = apply_visual_shift(
            rgb,
            exposure_multiplier=VISUAL_SHIFT_EXPOSURE,
            rgb_channel_multipliers=VISUAL_SHIFT_RGB_MULTIPLIERS,
        )
    chw = torch.from_numpy(np.transpose(rgb, (2, 0, 1)).copy())
    image = image_to_policy_float(chw)
    robot = getattr(getattr(base, "agent", None), "robot", None)
    if robot is None:
        raise Phase2CAEvaluationError("environment has no Panda robot")
    state = extract_panda_policy_state_v0(robot)
    if state.shape != (9,) or state.dtype != np.dtype(np.float32):
        raise Phase2CAEvaluationError("environment did not expose PandaPolicyStateV0 float32[9]")
    return ObservationBatch(
        features={
            IMAGE_FEATURE_KEY: image,
            STATE_FEATURE_KEY: torch.from_numpy(state.copy()),
        },
        metadata={
            "camera": "base_camera",
            "state_schema": "PandaPolicyStateV0",
            "visual_transform": VISUAL_SHIFT_IDENTITY if visual_shift else "none",
        },
    )


def _classify_failure(
    *,
    task_id: str,
    outcome: str,
    wrong_task_behavior: bool,
    action_smoothness: float | None,
    maximum_object_motion: float | None,
    diagnostics: Mapping[str, object],
) -> str:
    if outcome == "invalid_policy_output":
        return "invalid_policy_output"
    if outcome == "simulator_error":
        return "simulator_error"
    if wrong_task_behavior:
        return "wrong_task_behavior"
    if (maximum_object_motion is None or maximum_object_motion < 0.001) and outcome == "timeout":
        return "no_initial_motion"
    if action_smoothness is not None and action_smoothness > 0.35:
        return "action_oscillation"
    if task_id == "PickCube-v1":
        if diagnostics.get("is_grasped") is True and outcome != "success":
            return "object_drop"
        if maximum_object_motion is not None and maximum_object_motion >= 0.001:
            return "failed_grasp"
        return "incorrect_approach"
    if task_id == "StackCube-v1":
        if diagnostics.get("is_cubeA_on_cubeB") is True:
            return "stack_collapse"
        if diagnostics.get("is_cubeA_grasped") is True:
            return "failed_stack_alignment"
        if maximum_object_motion is not None and maximum_object_motion >= 0.001:
            return "failed_grasp"
        return "incorrect_approach"
    if task_id == "PushCube-v1":
        if diagnostics.get("target_overshoot") is True:
            return "push_overshoot"
        if maximum_object_motion is not None and maximum_object_motion >= 0.001:
            return "insufficient_push"
        return "incorrect_approach"
    return "timeout" if outcome == "timeout" else "other"


class OfficialManiSkillPolicyEvaluator:
    """Run a generic policy on one already-constructed official task."""

    def __init__(self, environment: object, *, task_id: str, maximum_steps: int = 200) -> None:
        if task_id not in TASK_IDS:
            raise Phase2CAEvaluationError("official evaluator received an unknown task")
        if maximum_steps < 1:
            raise ValueError("maximum_steps must be positive")
        self.environment = environment
        self.base = getattr(environment, "unwrapped", environment)
        self.task_id = task_id
        self.maximum_steps = maximum_steps
        spec = getattr(environment, "spec", None)
        if getattr(spec, "id", None) != task_id:
            raise Phase2CAEvaluationError(
                f"official environment ID must be {task_id}, got {getattr(spec, 'id', None)!r}"
            )
        if int(getattr(self.base, "num_envs", 0)) != 1:
            raise Phase2CAEvaluationError("evaluation requires num_envs=1")
        if getattr(self.base, "control_mode", None) != "pd_joint_pos":
            raise Phase2CAEvaluationError("evaluation requires pd_joint_pos")
        if int(getattr(self.base, "control_freq", 0)) != CONTROL_FREQUENCY_HZ:
            raise Phase2CAEvaluationError("evaluation requires the frozen 20 Hz control rate")

    def run_episode(
        self,
        *,
        policy: PolicyAdapter,
        schedule: Mapping[str, object],
        task_condition_id: str | None = None,
        capture_video: bool = False,
    ) -> EvaluatedEpisode:
        evaluation_id = schedule.get("evaluation_id")
        source_episode_id = schedule.get("source_episode_id")
        reset_identity = schedule.get("reset_identity")
        reset_kwargs = schedule.get("reset_kwargs")
        split = schedule.get("split")
        visual_transform = schedule.get("visual_transform")
        if (
            not isinstance(evaluation_id, str)
            or not isinstance(source_episode_id, int)
            or not isinstance(reset_identity, str)
            or not isinstance(reset_kwargs, Mapping)
            or not isinstance(split, str)
            or schedule.get("task_id") != self.task_id
        ):
            raise Phase2CAEvaluationError("evaluation schedule entry is malformed")
        condition = task_condition_id or self.task_id
        if condition not in TASK_IDS:
            raise Phase2CAEvaluationError("task-condition intervention ID is invalid")
        if condition not in policy.identity.compatible_task_ids:
            raise Phase2CAEvaluationError("policy is incompatible with requested task condition")
        visual_shift = visual_transform == VISUAL_SHIFT_IDENTITY
        if visual_transform not in {None, VISUAL_SHIFT_IDENTITY}:
            raise Phase2CAEvaluationError("evaluation schedule uses an unknown visual transform")
        policy.reset(
            PolicyContext(
                evaluation_task=None,
                evaluation_id=evaluation_id,
                task_id=condition,
            )
        )
        try:
            reset_result = self.environment.reset(**dict(reset_kwargs))
        except Exception as error:
            raise Phase2CAEvaluationError(
                f"official environment reset failed: {type(error).__name__}: {error}"
            ) from error
        if not isinstance(reset_result, tuple) or len(reset_result) != 2:
            raise Phase2CAEvaluationError("environment reset did not return a two-tuple")
        observation, reset_info = reset_result
        initial_positions = _object_positions(self.base, self.task_id)
        last_positions = initial_positions
        maximum_motion = 0.0 if initial_positions else None
        post_contact_divergences: list[float] = []
        frames: list[np.ndarray] = []
        if capture_video:
            initial_rgb = extract_base_camera_rgb(observation)
            frames.append(
                apply_visual_shift(
                    initial_rgb,
                    exposure_multiplier=VISUAL_SHIFT_EXPOSURE,
                    rgb_channel_multipliers=VISUAL_SHIFT_RGB_MULTIPLIERS,
                )
                if visual_shift
                else initial_rgb.copy()
            )
        diagnostics = _evaluate_environment(self.base, reset_info)
        actions: list[np.ndarray] = []
        inference_latencies: list[float] = []
        environment_latencies: list[float] = []
        saturated_components = 0
        action_components = 0
        policy_queries = 0
        outcome = "timeout"
        failure_reason: str | None = None
        steps = 0
        first_object_motion_seen = False
        while steps < self.maximum_steps:
            extracted = _extract_observation(
                observation,
                base=self.base,
                visual_shift=visual_shift,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                chunk = policy.act(extracted)
            except Exception as error:  # noqa: BLE001 - policy boundary classification
                outcome = "invalid_policy_output"
                failure_reason = f"{type(error).__name__}: {error}"
                break
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency = (time.perf_counter() - started) * 1000.0
            if not math.isfinite(latency) or latency < 0:
                raise Phase2CAEvaluationError("policy latency is invalid")
            inference_latencies.append(latency)
            policy_queries += 1
            query_start_positions = _object_positions(self.base, self.task_id)
            for raw_action in chunk.iter_valid_actions():
                if steps >= self.maximum_steps:
                    break
                candidate = (
                    raw_action.detach().cpu().numpy()
                    if isinstance(raw_action, torch.Tensor)
                    else np.asarray(raw_action)
                )
                action = np.asarray(candidate, dtype=np.float32).reshape(-1)
                if action.shape != (8,) or not np.isfinite(action).all():
                    outcome = "invalid_policy_output"
                    failure_reason = "policy action must be finite float32[8]"
                    break
                below = action < PANDA_ACTION_LOW
                above = action > PANDA_ACTION_HIGH
                if bool(np.any(below | above)):
                    outcome = "invalid_policy_output"
                    failure_reason = "policy action exceeded native pd_joint_pos bounds"
                    break
                margin = np.maximum((PANDA_ACTION_HIGH - PANDA_ACTION_LOW) * 0.01, 1e-8)
                saturated_components += int(
                    np.count_nonzero(
                        (action <= PANDA_ACTION_LOW + margin)
                        | (action >= PANDA_ACTION_HIGH - margin)
                    )
                )
                action_components += action.size
                actions.append(action.astype(np.float64))
                started = time.perf_counter()
                try:
                    step_result = self.environment.step(action)
                except Exception as error:  # noqa: BLE001 - simulator boundary classification
                    outcome = "simulator_error"
                    failure_reason = f"{type(error).__name__}: {error}"
                    break
                environment_latencies.append((time.perf_counter() - started) * 1000.0)
                if not isinstance(step_result, tuple) or len(step_result) != 5:
                    outcome = "simulator_error"
                    failure_reason = "environment step did not return a five-tuple"
                    break
                observation, _, terminated, truncated, info = step_result
                steps += 1
                if capture_video:
                    rgb = extract_base_camera_rgb(observation)
                    frames.append(
                        apply_visual_shift(
                            rgb,
                            exposure_multiplier=VISUAL_SHIFT_EXPOSURE,
                            rgb_channel_multipliers=VISUAL_SHIFT_RGB_MULTIPLIERS,
                        )
                        if visual_shift
                        else rgb.copy()
                    )
                diagnostics = _evaluate_environment(self.base, info)
                positions = _object_positions(self.base, self.task_id)
                motion = _maximum_translation(initial_positions, positions)
                if motion is not None:
                    maximum_motion = max(float(maximum_motion or 0.0), motion)
                    first_object_motion_seen |= motion >= 0.001
                last_positions = positions
                if _success(diagnostics) or _single_bool(
                    info.get("success") if isinstance(info, Mapping) else None
                ):
                    outcome = "success"
                    break
                terminated_value = _single_bool(terminated)
                truncated_value = _single_bool(truncated)
                if truncated_value:
                    outcome = "timeout"
                    failure_reason = "environment time limit truncated the episode"
                    break
                if terminated_value:
                    outcome = "terminated"
                    failure_reason = "environment terminated without success"
                    break
            query_end_positions = _object_positions(self.base, self.task_id)
            if first_object_motion_seen:
                divergence = _maximum_translation(query_start_positions, query_end_positions)
                if divergence is not None:
                    post_contact_divergences.append(divergence)
            if outcome != "timeout" or failure_reason is not None:
                break
        if outcome == "timeout" and failure_reason is None:
            failure_reason = f"episode step budget {self.maximum_steps} exhausted"
        success = outcome == "success"
        wrong_task = condition != self.task_id
        smoothness = (
            float(
                np.mean(
                    [
                        np.linalg.norm(right - left)
                        for left, right in zip(actions, actions[1:], strict=False)
                    ]
                )
            )
            if len(actions) > 1
            else None
        )
        category = (
            "other"
            if success
            else _classify_failure(
                task_id=self.task_id,
                outcome=outcome,
                wrong_task_behavior=wrong_task,
                action_smoothness=smoothness,
                maximum_object_motion=maximum_motion,
                diagnostics=diagnostics,
            )
        )
        if category not in FAILURE_CATEGORIES:
            raise Phase2CAEvaluationError(f"unknown failure category {category!r}")
        initial_to_final = _maximum_translation(initial_positions, last_positions)
        action_array = np.stack(actions) if actions else None
        result = Phase2CAEpisodeResult(
            evaluation_id=evaluation_id,
            split=split,
            task_id=self.task_id,
            skill_family=SKILL_FAMILIES[self.task_id],
            policy_id=policy.identity.policy_id,
            checkpoint_identity=policy.identity.checkpoint_identity,
            source_episode_id=source_episode_id,
            reset_identity=reset_identity,
            task_condition_id=condition,
            visual_transform=visual_transform if isinstance(visual_transform, str) else None,
            execution_horizon=int(policy.runtime_manifest["execution_horizon"]),
            outcome=outcome,
            success=success,
            timeout=outcome == "timeout",
            invalid_action=outcome == "invalid_policy_output",
            simulator_error=outcome == "simulator_error",
            wrong_task_behavior=wrong_task,
            failure_category=category,
            failure_reason=failure_reason,
            episode_length=steps,
            policy_query_count=policy_queries,
            actions_executed=len(actions),
            action_saturation_rate=(
                saturated_components / action_components if action_components else 0.0
            ),
            action_smoothness_mean_l2=smoothness,
            inference_latency_p50_ms=_percentile(inference_latencies, 0.50),
            inference_latency_p95_ms=_percentile(inference_latencies, 0.95),
            inference_latency_samples_ms=tuple(inference_latencies),
            environment_step_latency_p50_ms=_percentile(environment_latencies, 0.50),
            environment_step_latency_p95_ms=_percentile(environment_latencies, 0.95),
            post_contact_divergence_m=(
                float(statistics.fmean(post_contact_divergences))
                if post_contact_divergences
                else None
            ),
            initial_to_final_object_motion_m=initial_to_final,
            maximum_object_motion_m=maximum_motion,
            final_evaluation=diagnostics,
            action_min=(
                tuple(float(value) for value in action_array.min(axis=0))
                if action_array is not None
                else None
            ),
            action_max=(
                tuple(float(value) for value in action_array.max(axis=0))
                if action_array is not None
                else None
            ),
        )
        return EvaluatedEpisode(result=result, frames=tuple(frames))


def summarize_evaluation(
    records: Sequence[Phase2CAEpisodeResult | Mapping[str, object]],
) -> dict[str, object]:
    normalized = [
        item.to_dict() if isinstance(item, Phase2CAEpisodeResult) else dict(item)
        for item in records
    ]
    base = summarize_binary_results(normalized)
    latencies = [
        float(value)
        for record in normalized
        for value in cast_sequence(record.get("inference_latency_samples_ms"))
        if isinstance(value, int | float)
    ]
    smoothness = [
        float(value)
        for record in normalized
        if isinstance((value := record.get("action_smoothness_mean_l2")), int | float)
    ]
    episode_lengths = [int(record["episode_length"]) for record in normalized]
    return {
        "schema_version": "langmani-v2-phase2c-a-evaluation-summary-v0",
        **base,
        "task_ids": [
            task_id
            for task_id in TASK_IDS
            if task_id in {str(record["task_id"]) for record in normalized}
        ],
        "policy_ids": sorted({str(record["policy_id"]) for record in normalized}),
        "checkpoint_identities": sorted(
            {str(record["checkpoint_identity"]) for record in normalized}
        ),
        "mean_episode_length": statistics.fmean(episode_lengths),
        "mean_policy_query_count": statistics.fmean(
            int(record["policy_query_count"]) for record in normalized
        ),
        "mean_action_saturation_rate": statistics.fmean(
            float(record["action_saturation_rate"]) for record in normalized
        ),
        "mean_action_smoothness_l2": statistics.fmean(smoothness) if smoothness else None,
        "inference_latency_p50_ms": _percentile(latencies, 0.50),
        "inference_latency_p95_ms": _percentile(latencies, 0.95),
        "post_contact_divergence_mean_m": _optional_mean(
            record.get("post_contact_divergence_m") for record in normalized
        ),
        "package_fingerprint": PACKAGE_FINGERPRINT,
    }


def cast_sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, list | tuple) else ()


def _optional_mean(values: Sequence[object] | Any) -> float | None:
    finite = [
        float(value)
        for value in values
        if isinstance(value, int | float) and math.isfinite(float(value))
    ]
    return statistics.fmean(finite) if finite else None


def select_execution_horizon(
    reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Choose one common H from validation evidence before final evaluation."""

    grouped: dict[int, list[Mapping[str, object]]] = {horizon: [] for horizon in EXECUTION_HORIZONS}
    for report in reports:
        horizon = report.get("execution_horizon")
        if horizon not in EXECUTION_HORIZONS or report.get("split") != "validation":
            raise Phase2CAContractError("horizon selection received non-validation evidence")
        grouped[int(horizon)].append(report)
    if any(not values for values in grouped.values()):
        raise Phase2CAContractError("horizon selection lacks one pre-registered candidate")
    candidates = []
    for horizon, values in grouped.items():
        episode_count = sum(int(value["episode_count"]) for value in values)
        success_count = sum(int(value["success_count"]) for value in values)
        invalid_count = sum(int(value["invalid_action_count"]) for value in values)
        timeout_count = sum(
            int(cast_mapping(value.get("outcomes")).get("timeout", 0)) for value in values
        )
        smoothness = _optional_mean(value.get("mean_action_smoothness_l2") for value in values)
        queries = statistics.fmean(float(value["mean_policy_query_count"]) for value in values)
        p95 = _optional_mean(value.get("inference_latency_p95_ms") for value in values)
        candidates.append(
            {
                "execution_horizon": horizon,
                "episode_count": episode_count,
                "success_count": success_count,
                "success_rate": success_count / episode_count,
                "invalid_action_count": invalid_count,
                "timeout_count": timeout_count,
                "mean_action_smoothness_l2": smoothness,
                "mean_policy_query_count": queries,
                "mean_p95_inference_latency_ms": p95,
            }
        )
    eligible = [value for value in candidates if value["invalid_action_count"] == 0]
    if not eligible:
        raise Phase2CAContractError("all horizon candidates produced invalid actions")
    eligible.sort(
        key=lambda value: (
            -float(value["success_rate"]),
            int(value["timeout_count"]),
            (
                float(value["mean_action_smoothness_l2"])
                if value["mean_action_smoothness_l2"] is not None
                else math.inf
            ),
            float(value["mean_policy_query_count"]),
            (
                float(value["mean_p95_inference_latency_ms"])
                if value["mean_p95_inference_latency_ms"] is not None
                else math.inf
            ),
            int(value["execution_horizon"]),
        )
    )
    semantic = {
        "selection_split": "validation",
        "selection_scope": "one_common_horizon_for_all_primary_act_models",
        "selection_rule": (
            "maximize aggregate success; then minimize timeout, smoothness, policy queries, "
            "p95 latency, and execution horizon; any invalid action is ineligible"
        ),
        "candidates": candidates,
        "selected_execution_horizon": eligible[0]["execution_horizon"],
        "final_test_outcomes_available": False,
    }
    return {
        "schema_version": "langmani-v2-phase2c-a-horizon-selection-v0",
        **semantic,
        "fingerprint": f"sha256:{sha256_hex(semantic)}",
        "passed": True,
    }


def cast_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def write_video(
    path: str | Path,
    frames: Sequence[np.ndarray],
    *,
    fps: int = CONTROL_FREQUENCY_HZ,
) -> dict[str, object]:
    """Encode one representative RGB trace through PyAV."""

    if not frames:
        raise Phase2CAEvaluationError("representative video requires frames")
    try:
        import av
    except (ImportError, OSError) as error:
        raise Phase2CAEvaluationError("PyAV is unavailable for representative video") from error
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise Phase2CAEvaluationError(f"refusing to overwrite video {output}")
    container = av.open(str(output), mode="w")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = 256
        stream.height = 256
        stream.pix_fmt = "yuv444p"
        for frame in frames:
            image = np.asarray(frame)
            if image.dtype != np.uint8 or image.shape != (256, 256, 3):
                raise Phase2CAEvaluationError("video frame must be uint8 RGB[256,256,3]")
            packet_frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(packet_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    digest = _sha256_file(output)
    return {
        "path": output.as_posix(),
        "frame_count": len(frames),
        "fps": fps,
        "sha256": digest,
        "size_bytes": output.stat().st_size,
    }


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def append_episode_jsonl(path: str | Path, result: Phase2CAEpisodeResult) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result.to_dict(), sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


__all__ = [
    "EvaluatedEpisode",
    "OfficialManiSkillPolicyEvaluator",
    "Phase2CAEpisodeResult",
    "Phase2CAEvaluationError",
    "append_episode_jsonl",
    "select_execution_horizon",
    "summarize_evaluation",
    "write_video",
]
