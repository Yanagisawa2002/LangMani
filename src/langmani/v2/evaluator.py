"""Policy-agnostic closed-loop evaluator for canonical LangMani tasks."""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from langmani.datasets.lerobot_types import IMAGE_FEATURE_KEY, STATE_FEATURE_KEY
from langmani.datasets.observation_reconstruction import extract_base_camera_rgb
from langmani.datasets.policy_state import extract_panda_policy_state_v0
from langmani.environments.pick_place_by_instruction import ENV_ID as PICK_ENVIRONMENT_ID
from langmani.environments.push_to_region import ENV_ID as PUSH_ENVIRONMENT_ID
from langmani.policies.act_action_bounds import (
    ActionBoundConfig,
    ActionBoundMode,
    ActionBoundProcessingError,
    BoundedActionEnvPostprocessorV0,
)
from langmani.policies.act_training import image_to_policy_float
from langmani.v2.policy import (
    ObservationBatch,
    PolicyAdapter,
    PolicyContext,
    finite_latency_ms,
)
from langmani.v2.taxonomy import EvaluationTask

EVALUATION_RESULT_SCHEMA = "langmani-v2-evaluation-result-v0"
EVALUATION_SUMMARY_SCHEMA = "langmani-v2-evaluation-summary-v0"


class EvaluationError(RuntimeError):
    """Raised when environment or policy behavior violates the v2 evaluator contract."""


class EpisodeOutcome(StrEnum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    TARGET_OFF_TABLE = "target_off_table"
    INVALID_ACTION = "invalid_action"
    POLICY_FAILURE = "policy_failure"
    ENVIRONMENT_FAILURE = "environment_failure"
    TERMINATED = "terminated"


class ObservationExtractor(Protocol):
    """Environment-specific extraction boundary, independent of policy inference."""

    def extract(self, observation: object, *, environment: object) -> ObservationBatch: ...


class M1PolicyObservationExtractor:
    """Extract only deployed-policy RGB and PandaPolicyStateV0 features.

    Both registered v2 environments deliberately share this deployable input
    contract; no privileged task or simulator state crosses this boundary.
    """

    def extract(self, observation: object, *, environment: object) -> ObservationBatch:
        base = getattr(environment, "unwrapped", environment)
        rgb_hwc = extract_base_camera_rgb(observation)
        rgb = torch.from_numpy(np.transpose(rgb_hwc, (2, 0, 1)).copy())
        image = image_to_policy_float(rgb)
        state = extract_panda_policy_state_v0(base.agent.robot)
        if tuple(image.shape) != (3, 256, 256) or image.dtype is not torch.float32:
            raise EvaluationError("base_camera extraction must produce float32[3,256,256]")
        if state.shape != (9,) or state.dtype != np.dtype(np.float32):
            raise EvaluationError("PandaPolicyStateV0 extraction must produce float32[9]")
        return ObservationBatch(
            features={
                IMAGE_FEATURE_KEY: image,
                STATE_FEATURE_KEY: torch.from_numpy(np.array(state, copy=True)),
            },
            metadata={"camera": "base_camera", "state_schema": "PandaPolicyStateV0"},
        )


class Phase2CPolicyObservationExtractor:
    """Apply one frozen outcome-independent visual domain after deployable extraction."""

    def __init__(self, *, visual_domain: str = "base") -> None:
        if visual_domain not in {"base", "photometric_shift_v0"}:
            raise EvaluationError(f"unknown Phase 2C visual domain {visual_domain!r}")
        self.visual_domain = visual_domain
        self._base = M1PolicyObservationExtractor()

    def extract(self, observation: object, *, environment: object) -> ObservationBatch:
        result = self._base.extract(observation, environment=environment)
        if self.visual_domain == "base":
            return result
        image = result.features[IMAGE_FEATURE_KEY]
        gains = torch.tensor([0.82, 0.96, 1.08], dtype=torch.float32).reshape(3, 1, 1)
        shifted = torch.clamp(torch.pow(image, 0.92) * gains + 0.025, 0.0, 1.0)
        return ObservationBatch(
            features={
                **result.features,
                IMAGE_FEATURE_KEY: shifted,
            },
            metadata={
                **result.metadata,
                "visual_domain": "photometric_shift_v0",
                "visual_transform": "gamma=0.92,gains=[0.82,0.96,1.08],offset=0.025,clamp=[0,1]",
            },
        )


def _single_bool(value: object, label: str) -> bool:
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    array = np.asarray(candidate)
    if array.size != 1:
        raise EvaluationError(f"{label} must contain one boolean")
    return bool(array.reshape(-1)[0])


def _evaluation_bools(value: object) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, bool] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        try:
            result[key] = _single_bool(item, key)
        except EvaluationError:
            continue
    return result


def _current_evaluation(environment: object, fallback: object) -> dict[str, bool]:
    base = getattr(environment, "unwrapped", environment)
    accessor = getattr(base, "get_policy_rollout_evaluation", None)
    if callable(accessor):
        return _evaluation_bools(accessor())
    return _evaluation_bools(fallback)


def _current_numeric(environment: object, fallback: object, key: str) -> float | None:
    """Read one finite scalar diagnostic without admitting it to policy inputs."""

    base = getattr(environment, "unwrapped", environment)
    accessor = getattr(base, "get_policy_rollout_evaluation", None)
    value = accessor() if callable(accessor) else fallback
    if not isinstance(value, Mapping) or key not in value:
        return None
    candidate = value[key]
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    array = np.asarray(candidate)
    if array.size != 1:
        return None
    result = float(array.reshape(-1)[0])
    return result if math.isfinite(result) else None


def _latency_summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "maximum": None}
    ordered = sorted(float(item) for item in values)

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
        return ordered[index]

    return {
        "mean": statistics.fmean(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "maximum": ordered[-1],
    }


@dataclass(frozen=True, slots=True)
class EvaluationEpisodeResult:
    """Compact machine-readable outcome from one canonical task and seed."""

    evaluation_id: str
    seed: int
    task_identity: str
    skill_family: str
    policy_identity: str
    checkpoint_identity: str
    outcome: EpisodeOutcome
    success: bool
    timeout: bool
    wrong_object_interaction: bool
    object_drop_or_loss: bool
    invalid_action: bool
    episode_length: int
    action_projection_count: int
    action_projection_component_count: int
    policy_query_count: int
    policy_inference_latency_ms: Mapping[str, float | None]
    environment_step_latency_ms: Mapping[str, float | None]
    final_evaluation: Mapping[str, bool]
    failure_reason: str | None
    schema_version: str = EVALUATION_RESULT_SCHEMA
    actions_executed: int = 0
    mean_actions_per_policy_query: float = 0.0
    final_object_to_target_distance: float | None = None
    action_smoothness_mean_l2: float | None = None
    action_saturation_rate: float = 0.0
    effective_control_throughput_hz: float | None = None
    progress_curve: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "evaluation_id": self.evaluation_id,
            "seed": self.seed,
            "task_identity": self.task_identity,
            "skill_family": self.skill_family,
            "policy_identity": self.policy_identity,
            "checkpoint_identity": self.checkpoint_identity,
            "outcome": self.outcome.value,
            "success": self.success,
            "timeout": self.timeout,
            "wrong_object_interaction": self.wrong_object_interaction,
            "object_drop_or_loss": self.object_drop_or_loss,
            "invalid_action": self.invalid_action,
            "episode_length": self.episode_length,
            "action_projection_count": self.action_projection_count,
            "action_projection_component_count": self.action_projection_component_count,
            "policy_query_count": self.policy_query_count,
            "policy_inference_latency_ms": dict(self.policy_inference_latency_ms),
            "environment_step_latency_ms": dict(self.environment_step_latency_ms),
            "final_evaluation": dict(self.final_evaluation),
            "failure_reason": self.failure_reason,
            "actions_executed": self.actions_executed,
            "mean_actions_per_policy_query": self.mean_actions_per_policy_query,
            "final_object_to_target_distance": self.final_object_to_target_distance,
            "action_smoothness_mean_l2": self.action_smoothness_mean_l2,
            "action_saturation_rate": self.action_saturation_rate,
            "effective_control_throughput_hz": self.effective_control_throughput_hz,
            "progress_curve": list(self.progress_curve),
        }


class UnifiedPolicyEvaluator:
    """Execute any compatible PolicyAdapter without language-router coupling."""

    def __init__(
        self,
        *,
        environment: object,
        observation_extractor: ObservationExtractor | None = None,
        action_bound_config: ActionBoundConfig | None = None,
    ) -> None:
        self.environment = environment
        self.base = getattr(environment, "unwrapped", environment)
        self.observation_extractor = observation_extractor or M1PolicyObservationExtractor()
        self._validate_environment()
        self.action_processor = BoundedActionEnvPostprocessorV0.from_environment(
            environment,
            action_bound_config or ActionBoundConfig(mode=ActionBoundMode.PROJECT),
            expected_action_components=8,
        )

    def _validate_environment(self) -> None:
        spec = getattr(self.environment, "spec", None)
        if getattr(spec, "id", None) not in {
            PICK_ENVIRONMENT_ID,
            PUSH_ENVIRONMENT_ID,
        }:
            raise EvaluationError("unified evaluator requires a registered LangMani v2 environment")
        if int(getattr(self.base, "num_envs", 0)) != 1:
            raise EvaluationError("unified evaluator requires num_envs=1")
        if getattr(self.base, "control_mode", None) != "pd_joint_pos":
            raise EvaluationError("unified evaluator requires pd_joint_pos")

    def run_episode(
        self,
        *,
        policy: PolicyAdapter,
        task: EvaluationTask,
        evaluation_id: str,
        language_instruction: str | None = None,
    ) -> EvaluationEpisodeResult:
        instance = task.task_instance
        if getattr(self.environment.spec, "id", None) != instance.environment_id:
            raise EvaluationError("environment identity disagrees with the requested task instance")
        if instance.skill_family_id not in policy.identity.compatible_skill_families:
            raise EvaluationError("policy is incompatible with the requested skill family")
        if instance.canonical_task_id not in policy.identity.compatible_task_ids:
            raise EvaluationError("policy is incompatible with the requested task instance")
        context = PolicyContext(
            evaluation_task=task,
            evaluation_id=evaluation_id,
            language_instruction=language_instruction,
        )
        policy.reset(context)
        reset_result = self.environment.reset(
            seed=task.scene_seed,
            options={"task_spec": instance.environment_task_spec.to_dict()},
        )
        if not isinstance(reset_result, tuple) or len(reset_result) != 2:
            raise EvaluationError("environment reset must return the Gymnasium two-tuple")
        observation, reset_info = reset_result
        specs = self.base.get_episode_specs()
        if len(specs) != 1 or specs[0].task_id != instance.canonical_task_id:
            raise EvaluationError("EpisodeSpec disagrees with the requested task")
        if specs[0].scene_seed != task.scene_seed:
            raise EvaluationError("EpisodeSpec disagrees with the requested scene seed")

        final_evaluation = _current_evaluation(self.environment, reset_info)
        initial_distance = _current_numeric(self.environment, reset_info, "target_distance")
        progress_curve: list[float] = [] if initial_distance is None else [initial_distance]
        inference_latencies: list[float] = []
        environment_latencies: list[float] = []
        projection_count = 0
        projection_components = 0
        policy_queries = 0
        executed_actions: list[np.ndarray] = []
        saturated_components = 0
        action_components = 0
        steps = 0
        wrong_object = final_evaluation.get(
            "wrong_object_is_grasped", False
        ) or final_evaluation.get("wrong_object_in_target_bin", False)
        wrong_object |= final_evaluation.get("wrong_object_contact", False)
        wrong_object |= final_evaluation.get("wrong_object_displaced", False)
        object_loss = final_evaluation.get("target_off_table", False)
        object_loss |= final_evaluation.get("target_outside_workspace", False)
        object_loss |= final_evaluation.get("target_lifted", False)
        object_loss |= final_evaluation.get("target_toppled", False)
        outcome = EpisodeOutcome.TIMEOUT
        failure_reason: str | None = None

        while steps < instance.maximum_episode_steps:
            try:
                extracted = self.observation_extractor.extract(
                    observation,
                    environment=self.environment,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                chunk = policy.act(extracted)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_latencies.append(finite_latency_ms(start, time.perf_counter()))
                policy_queries += 1
            except Exception as error:  # noqa: BLE001 - classify policy command boundary
                outcome = EpisodeOutcome.POLICY_FAILURE
                failure_reason = f"{type(error).__name__}: {error}"
                break

            for raw_action in chunk.iter_valid_actions():
                if steps >= instance.maximum_episode_steps:
                    break
                if isinstance(raw_action, torch.Tensor):
                    candidate: object = raw_action.detach().reshape(1, -1)
                else:
                    candidate = np.asarray(raw_action).reshape(1, -1)
                try:
                    bounded = self.action_processor.process(candidate, rollout_step=steps + 1)
                except ActionBoundProcessingError as error:
                    outcome = EpisodeOutcome.INVALID_ACTION
                    failure_reason = f"{type(error).__name__}[{error.kind.value}]: {error}"
                    break
                projection_count += int(bounded.audit_record.was_projected)
                projection_components += bounded.audit_record.projected_component_count
                raw = np.asarray(bounded.audit_record.raw_action, dtype=np.float64)
                low = np.asarray(bounded.audit_record.lower_bounds, dtype=np.float64)
                high = np.asarray(bounded.audit_record.upper_bounds, dtype=np.float64)
                span = high - low
                saturation_margin = np.maximum(span * 0.01, 1e-8)
                saturated_components += int(
                    np.count_nonzero(
                        (raw <= low + saturation_margin) | (raw >= high - saturation_margin)
                    )
                )
                action_components += int(raw.size)
                executed = bounded.executed_action
                if isinstance(executed, torch.Tensor):
                    action = executed.detach().cpu().numpy()[0].copy()
                else:
                    action = np.asarray(executed)[0].copy()
                executed_actions.append(np.asarray(action, dtype=np.float64).copy())
                start = time.perf_counter()
                try:
                    step_result = self.environment.step(action)
                except Exception as error:  # noqa: BLE001 - classify simulator boundary
                    outcome = EpisodeOutcome.ENVIRONMENT_FAILURE
                    failure_reason = f"{type(error).__name__}: {error}"
                    break
                environment_latencies.append(finite_latency_ms(start, time.perf_counter()))
                if not isinstance(step_result, tuple) or len(step_result) != 5:
                    outcome = EpisodeOutcome.ENVIRONMENT_FAILURE
                    failure_reason = "environment step did not return the Gymnasium five-tuple"
                    break
                observation, _, terminated, truncated, info = step_result
                steps += 1
                final_evaluation = _current_evaluation(self.environment, info)
                current_distance = _current_numeric(self.environment, info, "target_distance")
                if current_distance is not None:
                    progress_curve.append(current_distance)
                wrong_object |= final_evaluation.get(
                    "wrong_object_is_grasped", False
                ) or final_evaluation.get("wrong_object_in_target_bin", False)
                wrong_object |= final_evaluation.get("wrong_object_contact", False)
                wrong_object |= final_evaluation.get("wrong_object_displaced", False)
                object_loss |= final_evaluation.get("target_off_table", False)
                object_loss |= final_evaluation.get("target_outside_workspace", False)
                object_loss |= final_evaluation.get("target_lifted", False)
                object_loss |= final_evaluation.get("target_toppled", False)
                if final_evaluation.get("success", False):
                    outcome = EpisodeOutcome.SUCCESS
                    break
                if final_evaluation.get("target_off_table", False) or final_evaluation.get(
                    "target_outside_workspace", False
                ):
                    outcome = EpisodeOutcome.TARGET_OFF_TABLE
                    failure_reason = "environment reported target workspace loss"
                    break
                if _single_bool(truncated, "truncated"):
                    outcome = EpisodeOutcome.TIMEOUT
                    failure_reason = "M1 time limit truncated the episode"
                    break
                if _single_bool(terminated, "terminated"):
                    outcome = EpisodeOutcome.TERMINATED
                    failure_reason = "M1 terminated without success"
                    break
            if (
                outcome is not EpisodeOutcome.TIMEOUT
                or failure_reason is not None
                or steps >= instance.maximum_episode_steps
            ):
                break
        if outcome is EpisodeOutcome.TIMEOUT and failure_reason is None:
            failure_reason = f"episode step budget {instance.maximum_episode_steps} exhausted"
        smoothness = (
            float(
                np.mean(
                    [
                        np.linalg.norm(right - left)
                        for left, right in zip(executed_actions, executed_actions[1:], strict=False)
                    ]
                )
            )
            if len(executed_actions) > 1
            else None
        )
        total_latency_ms = sum(inference_latencies) + sum(environment_latencies)
        final_distance = (
            progress_curve[-1]
            if progress_curve
            else _current_numeric(
                self.environment,
                reset_info if not final_evaluation else None,
                "target_distance",
            )
        )
        return EvaluationEpisodeResult(
            evaluation_id=evaluation_id,
            seed=task.scene_seed,
            task_identity=instance.canonical_task_id,
            skill_family=instance.skill_family_id,
            policy_identity=policy.identity.policy_id,
            checkpoint_identity=policy.identity.checkpoint_identity,
            outcome=outcome,
            success=outcome is EpisodeOutcome.SUCCESS,
            timeout=outcome is EpisodeOutcome.TIMEOUT,
            wrong_object_interaction=wrong_object,
            object_drop_or_loss=object_loss,
            invalid_action=(
                outcome is EpisodeOutcome.INVALID_ACTION
                or final_evaluation.get("invalid_action", False)
                or final_evaluation.get("action_out_of_bounds", False)
            ),
            episode_length=steps,
            action_projection_count=projection_count,
            action_projection_component_count=projection_components,
            policy_query_count=policy_queries,
            policy_inference_latency_ms=_latency_summary(inference_latencies),
            environment_step_latency_ms=_latency_summary(environment_latencies),
            final_evaluation=final_evaluation,
            failure_reason=failure_reason,
            actions_executed=len(executed_actions),
            mean_actions_per_policy_query=(
                len(executed_actions) / policy_queries if policy_queries else 0.0
            ),
            final_object_to_target_distance=final_distance,
            action_smoothness_mean_l2=smoothness,
            action_saturation_rate=(
                saturated_components / action_components if action_components else 0.0
            ),
            effective_control_throughput_hz=(
                len(executed_actions) * 1000.0 / total_latency_ms if total_latency_ms > 0 else None
            ),
            progress_curve=tuple(progress_curve),
        )


def summarize_results(results: Sequence[EvaluationEpisodeResult]) -> dict[str, object]:
    if not results:
        raise EvaluationError("cannot summarize an empty evaluation")
    task_ids = sorted({item.task_identity for item in results})
    policy_ids = {item.policy_identity for item in results}
    checkpoint_ids = {item.checkpoint_identity for item in results}
    if len(policy_ids) != 1 or len(checkpoint_ids) != 1:
        raise EvaluationError("one summary cannot mix policy or checkpoint identities")
    return {
        "schema_version": EVALUATION_SUMMARY_SCHEMA,
        "episode_count": len(results),
        "success_count": sum(item.success for item in results),
        "success_rate": sum(item.success for item in results) / len(results),
        "timeout_count": sum(item.timeout for item in results),
        "wrong_object_interaction_count": sum(item.wrong_object_interaction for item in results),
        "object_drop_or_loss_count": sum(item.object_drop_or_loss for item in results),
        "invalid_action_count": sum(item.invalid_action for item in results),
        "action_projection_count": sum(item.action_projection_count for item in results),
        "task_identities": task_ids,
        "skill_families": sorted({item.skill_family for item in results}),
        "policy_identity": next(iter(policy_ids)),
        "checkpoint_identity": next(iter(checkpoint_ids)),
        "seeds": [item.seed for item in results],
        "outcomes": {
            status.value: sum(item.outcome is status for item in results)
            for status in EpisodeOutcome
        },
    }


def persist_evaluation(
    *,
    output_root: Path,
    runtime_manifest: Mapping[str, Any],
    results: Sequence[EvaluationEpisodeResult],
) -> dict[str, object]:
    """Atomically promote one simple JSONL evaluation artifact set."""

    output = output_root.resolve()
    if output.exists():
        raise EvaluationError(f"evaluation output already exists: {output}")
    staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
    if staging.exists():
        raise EvaluationError(f"evaluation staging path already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        (staging / "runtime_manifest.json").write_text(
            json.dumps(dict(runtime_manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with (staging / "episodes.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for result in results:
                stream.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
        summary = summarize_results(results)
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        complete = {
            "schema_version": "langmani-v2-evaluation-complete-v0",
            "episode_count": len(results),
            "runtime_manifest": "runtime_manifest.json",
            "episodes": "episodes.jsonl",
            "summary": "summary.json",
        }
        (staging / "complete.json").write_text(
            json.dumps(complete, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output)
    except Exception:
        if staging.exists():
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()
        raise
    return summary


__all__ = [
    "EVALUATION_RESULT_SCHEMA",
    "EVALUATION_SUMMARY_SCHEMA",
    "EpisodeOutcome",
    "EvaluationEpisodeResult",
    "EvaluationError",
    "M1PolicyObservationExtractor",
    "ObservationExtractor",
    "Phase2CPolicyObservationExtractor",
    "UnifiedPolicyEvaluator",
    "persist_evaluation",
    "summarize_results",
]
