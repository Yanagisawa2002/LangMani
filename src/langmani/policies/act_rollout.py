"""Closed-loop ACT-to-ManiSkill adapter for the single M1 environment."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol, cast

import numpy as np
import torch
from lerobot.processor import PolicyProcessorPipeline

from langmani.datasets.lerobot_types import IMAGE_FEATURE_KEY, STATE_FEATURE_KEY
from langmani.datasets.observation_reconstruction import extract_base_camera_rgb
from langmani.datasets.policy_state import extract_panda_policy_state_v0
from langmani.environments.pick_place_by_instruction import ENV_ID
from langmani.environments.specs import TaskSpec, stable_task_id
from langmani.policies.act_action_bounds import (
    ActionBoundConfig,
    ActionBoundErrorKind,
    ActionBoundProcessingError,
    ActionProjectionRecord,
    ActionProjectionSummary,
    BoundedActionEnvPostprocessorV0,
)
from langmani.policies.act_training import image_to_policy_float
from langmani.policies.act_types import (
    ACT_ACTION_COMPONENTS,
    ActEvaluationConfig,
    ActVariant,
    EvaluationSplit,
    RolloutEpisodeResult,
    RolloutStatus,
)


class RolloutContractError(RuntimeError):
    """Raised when policy inference cannot safely drive the M1 environment."""


class StatefulPolicy(Protocol):
    def reset(self) -> None: ...

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor: ...


class ProcessedObservationAugmenter(Protocol):
    """Optional post-normalization command injection used by M4.2 TaskToken."""

    def __call__(self, batch: dict[str, torch.Tensor], task_id: str) -> dict[str, torch.Tensor]: ...


class PreBoundActionTransform(Protocol):
    """Explicit runtime transform applied before the unchanged bound processor."""

    def __call__(self, action: torch.Tensor, *, rollout_step: int) -> torch.Tensor: ...


TaskConditioner = Callable[[np.ndarray, str], np.ndarray]
ProjectionRecordSink = Callable[[str, ActionProjectionRecord], None]


def _single_bool(value: object, *, label: str) -> bool:
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    array = np.asarray(candidate)
    if array.size != 1:
        raise RolloutContractError(f"{label} must contain one boolean value")
    return bool(array.reshape(-1)[0])


def _evaluation_bools(value: object) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, bool] = {}
    for key, item in value.items():
        if isinstance(key, str):
            try:
                result[key] = _single_bool(item, label=key)
            except (TypeError, ValueError, RolloutContractError):
                continue
    return result


def _current_evaluation(base_environment: Any, fallback: object) -> dict[str, bool]:
    accessor = getattr(base_environment, "get_policy_rollout_evaluation", None)
    if callable(accessor):
        return _evaluation_bools(accessor())
    return _evaluation_bools(fallback)


def _base_environment(env: object) -> Any:
    return getattr(env, "unwrapped", env)


def _reset_component(component: object, name: str) -> None:
    reset = getattr(component, "reset", None)
    if not callable(reset):
        raise RolloutContractError(f"{name} must expose reset()")
    reset()


def build_policy_observation(
    observation: object,
    *,
    base_environment: Any,
    variant: ActVariant,
    task_id: str,
    task_conditioner: TaskConditioner | None,
) -> dict[str, torch.Tensor]:
    rgb_hwc = extract_base_camera_rgb(observation)
    rgb = torch.from_numpy(np.transpose(rgb_hwc, (2, 0, 1)).copy())
    image = image_to_policy_float(rgb)
    state = extract_panda_policy_state_v0(base_environment.agent.robot)
    if variant is ActVariant.MIXED_TASK_ONEHOT:
        if task_conditioner is None:
            raise RolloutContractError("mixed_task_onehot requires the canonical conditioner")
        state = task_conditioner(state, task_id)
    elif task_conditioner is not None:
        raise RolloutContractError("standard ACT variants must not receive task conditioning")
    expected_state = 15 if variant is ActVariant.MIXED_TASK_ONEHOT else 9
    if state.shape != (expected_state,) or state.dtype != np.dtype(np.float32):
        raise RolloutContractError(
            f"raw policy state must be float32[{expected_state}], got {state.dtype}{state.shape}"
        )
    return {
        IMAGE_FEATURE_KEY: image,
        STATE_FEATURE_KEY: torch.from_numpy(np.array(state, copy=True)),
    }


def _raw_environment_action(value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise RolloutContractError("ACT postprocessor must return a torch tensor")
    tensor = value.detach()
    if tensor.dtype != torch.float32 or tuple(tensor.shape) != (1, ACT_ACTION_COMPONENTS):
        raise RolloutContractError(
            f"postprocessed action must be float32[1,8], got {tensor.dtype}{tuple(tensor.shape)}"
        )
    return tensor


class ActManiSkillRolloutAdapter:
    """Execute installed ACT queue semantics without planner or hidden state."""

    def __init__(
        self,
        *,
        env: object,
        policy: StatefulPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        variant: ActVariant,
        run_fingerprint: str,
        checkpoint_fingerprint: str,
        schedule_digest: str,
        runtime_fingerprint: str,
        action_bound_config: ActionBoundConfig,
        evaluation: ActEvaluationConfig | None = None,
        task_conditioner: TaskConditioner | None = None,
        projection_record_sink: ProjectionRecordSink | None = None,
        processed_observation_augmenter: ProcessedObservationAugmenter | None = None,
        pre_bound_action_transform: PreBoundActionTransform | None = None,
    ) -> None:
        self.env = env
        self.base = _base_environment(env)
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.variant = ActVariant(variant)
        self.run_fingerprint = run_fingerprint
        self.checkpoint_fingerprint = checkpoint_fingerprint
        self.schedule_digest = schedule_digest
        self.runtime_fingerprint = runtime_fingerprint
        self.action_bound_config = action_bound_config
        self.evaluation = evaluation or ActEvaluationConfig()
        self.task_conditioner = task_conditioner
        self.projection_record_sink = projection_record_sink
        self.processed_observation_augmenter = processed_observation_augmenter
        self.pre_bound_action_transform = pre_bound_action_transform
        self._validate_environment()
        self.action_bound_processor = BoundedActionEnvPostprocessorV0.from_environment(
            env,
            action_bound_config,
            expected_action_components=ACT_ACTION_COMPONENTS,
        )

    def _validate_environment(self) -> None:
        spec = getattr(self.env, "spec", None)
        environment_id = getattr(spec, "id", None)
        if environment_id != ENV_ID:
            raise RolloutContractError(f"rollout requires {ENV_ID}, got {environment_id!r}")
        num_envs = getattr(self.base, "num_envs", None)
        if num_envs is None or int(num_envs) != 1:
            raise RolloutContractError("M4 policy rollouts require num_envs=1")
        control_mode = getattr(self.base, "control_mode", None)
        if control_mode != "pd_joint_pos":
            raise RolloutContractError("M4 policy rollouts require pd_joint_pos")
        control_frequency = getattr(self.base, "control_freq", None)
        if (
            control_frequency is None
            or int(control_frequency) != self.evaluation.control_frequency_hz
        ):
            raise RolloutContractError(
                "M4 policy rollouts require the declared 20 Hz environment control frequency"
            )
        if self.variant is ActVariant.MIXED_TASK_ONEHOT and self.task_conditioner is None:
            raise RolloutContractError("task-conditioned ACT requires CanonicalTaskOneHotV0")
        if self.variant is not ActVariant.MIXED_TASK_ONEHOT and self.task_conditioner is not None:
            raise RolloutContractError("standard ACT must not receive a task conditioner")

    def reset_policy_state(self) -> None:
        """Drop every queued action and any processor state at episode boundaries."""
        _reset_component(self.policy, "policy")
        _reset_component(self.preprocessor, "preprocessor")
        _reset_component(self.postprocessor, "postprocessor")
        if self.pre_bound_action_transform is not None:
            _reset_component(self.pre_bound_action_transform, "pre_bound_action_transform")

    def run_episode(
        self,
        *,
        evaluation_id: str,
        split: EvaluationSplit,
        scene_seed: int,
        task_spec: TaskSpec,
    ) -> RolloutEpisodeResult:
        """Run one episode through the declared reject or audited projection boundary."""
        self.reset_policy_state()
        task_id = stable_task_id(task_spec)
        begin_task = getattr(self.policy, "begin_task", None)
        if callable(begin_task):
            begin_task(task_id)
        reset_result = self.env.reset(
            seed=scene_seed,
            options={"task_spec": task_spec.to_dict()},
        )
        if not isinstance(reset_result, tuple) or len(reset_result) != 2:
            raise RolloutContractError("M1 reset must return the Gymnasium two-tuple")
        observation, reset_info = reset_result
        specs = self.base.get_episode_specs()
        if len(specs) != 1:
            raise RolloutContractError("M1 metadata must contain one episode spec")
        episode_spec = specs[0]
        if (
            episode_spec.scene_seed != scene_seed
            or episode_spec.task_id != task_id
            or episode_spec.task_spec != task_spec
        ):
            raise RolloutContractError("M1 reset metadata disagrees with rollout schedule")

        final_evaluation = _current_evaluation(self.base, reset_info)
        inference_latencies: list[float] = []
        environment_latencies: list[float] = []
        actions: list[np.ndarray] = []
        projection_records: list[ActionProjectionRecord] = []
        total_policy_actions = 0
        any_nonfinite_action = False
        any_malformed_action = False
        target_grasped_any = final_evaluation.get("target_is_grasped", False)
        wrong_grasped_any = final_evaluation.get("wrong_object_is_grasped", False)
        target_in_target_bin = final_evaluation.get("target_in_target_bin", False)
        target_in_wrong_bin = final_evaluation.get("target_in_wrong_bin", False)
        wrong_object_in_target_bin = final_evaluation.get("wrong_object_in_target_bin", False)
        first_grasp_step: int | None = 0 if target_grasped_any else None
        release_step: int | None = None
        success_step: int | None = None
        status = RolloutStatus.TIMEOUT
        reason: str | None = None
        episode_start = time.perf_counter()
        steps = 0

        for step in range(1, self.evaluation.maximum_episode_steps + 1):
            total_policy_actions += 1
            try:
                raw_input = build_policy_observation(
                    observation,
                    base_environment=self.base,
                    variant=self.variant,
                    task_id=task_id,
                    task_conditioner=self.task_conditioner,
                )
                inference_start = time.perf_counter()
                processed = self.preprocessor(raw_input)
                if not isinstance(processed, Mapping):
                    raise RolloutContractError("preprocessor must return a tensor mapping")
                processed_batch = cast(dict[str, torch.Tensor], dict(processed))
                if self.processed_observation_augmenter is not None:
                    processed_batch = self.processed_observation_augmenter(processed_batch, task_id)
                    if not isinstance(processed_batch, dict):
                        raise RolloutContractError(
                            "processed observation augmenter must return a tensor dictionary"
                        )
                predicted = self.policy.select_action(processed_batch)
                postprocessed = self.postprocessor(predicted)
                if torch.cuda.is_available() and str(
                    getattr(predicted, "device", "cpu")
                ).startswith("cuda"):
                    torch.cuda.synchronize()
                inference_latencies.append((time.perf_counter() - inference_start) * 1000.0)
                raw_environment_action = _raw_environment_action(postprocessed)
                if self.pre_bound_action_transform is not None:
                    raw_environment_action = _raw_environment_action(
                        self.pre_bound_action_transform(
                            raw_environment_action,
                            rollout_step=step,
                        )
                    )
                bounded = self.action_bound_processor.process(
                    raw_environment_action,
                    rollout_step=step,
                )
                projection_records.append(bounded.audit_record)
                if self.projection_record_sink is not None:
                    self.projection_record_sink(evaluation_id, bounded.audit_record)
                executed = bounded.executed_action
                if not isinstance(executed, torch.Tensor):
                    raise RolloutContractError("ACT tensor action processing changed array type")
                action = executed.detach().cpu().numpy()[0].copy()
            except ActionBoundProcessingError as error:
                if error.record is not None:
                    projection_records.append(error.record)
                    if self.projection_record_sink is not None:
                        self.projection_record_sink(evaluation_id, error.record)
                any_nonfinite_action |= error.kind is ActionBoundErrorKind.NONFINITE_ACTION
                any_malformed_action |= error.kind is ActionBoundErrorKind.MALFORMED_ACTION
                status = RolloutStatus.INVALID_ACTION
                reason = f"{type(error).__name__}[{error.kind.value}]: {error}"
                break
            except Exception as error:  # noqa: BLE001 - classify inference command boundary
                status = (
                    RolloutStatus.INVALID_ACTION
                    if isinstance(error, RolloutContractError) and "action" in str(error).lower()
                    else RolloutStatus.INFERENCE_FAILURE
                )
                reason = f"{type(error).__name__}: {error}"
                break

            actions.append(action)
            environment_start = time.perf_counter()
            try:
                step_result = self.env.step(action)
            except Exception as error:  # noqa: BLE001 - classify simulator boundary
                status = RolloutStatus.ENVIRONMENT_FAILURE
                reason = f"{type(error).__name__}: {error}"
                break
            environment_latencies.append((time.perf_counter() - environment_start) * 1000.0)
            if not isinstance(step_result, tuple) or len(step_result) != 5:
                status = RolloutStatus.ENVIRONMENT_FAILURE
                reason = "environment step did not return the Gymnasium five-tuple"
                break
            observation, _, terminated, truncated, info = step_result
            steps = step
            final_evaluation = _current_evaluation(self.base, info)
            target_grasped = final_evaluation.get("target_is_grasped", False)
            if target_grasped and first_grasp_step is None:
                first_grasp_step = step
            if first_grasp_step is not None and not target_grasped and release_step is None:
                release_step = step
            target_grasped_any |= target_grasped
            wrong_grasped_any |= final_evaluation.get("wrong_object_is_grasped", False)
            target_in_target_bin |= final_evaluation.get("target_in_target_bin", False)
            target_in_wrong_bin |= final_evaluation.get("target_in_wrong_bin", False)
            wrong_object_in_target_bin |= final_evaluation.get("wrong_object_in_target_bin", False)
            if final_evaluation.get("success", False):
                success_step = step
                status = RolloutStatus.SUCCESS
                break
            if final_evaluation.get("target_off_table", False):
                status = RolloutStatus.TARGET_OFF_TABLE
                reason = "M1 evaluator reported target_off_table"
                break
            if _single_bool(truncated, label="truncated"):
                status = RolloutStatus.TRUNCATED
                reason = "M1 time limit truncated the episode"
                break
            if _single_bool(terminated, label="terminated"):
                status = RolloutStatus.ENVIRONMENT_FAILURE
                reason = "M1 terminated without success or target_off_table"
                break
        else:
            status = RolloutStatus.TIMEOUT
            reason = f"policy step budget {self.evaluation.maximum_episode_steps} exhausted"

        total_duration = time.perf_counter() - episode_start
        if actions:
            action_matrix = np.stack(actions)
            cumulative = float(np.linalg.norm(action_matrix, axis=1).sum())
            action_min = tuple(float(value) for value in action_matrix.min(axis=0))
            action_max = tuple(float(value) for value in action_matrix.max(axis=0))
        else:
            cumulative = 0.0
            action_min = (0.0,) * ACT_ACTION_COMPONENTS
            action_max = (0.0,) * ACT_ACTION_COMPONENTS
        hz = float(self.evaluation.control_frequency_hz)
        projection_summary = ActionProjectionSummary.from_records(
            projection_records,
            action_dimension=ACT_ACTION_COMPONENTS,
            any_nonfinite_action=any_nonfinite_action,
            any_malformed_action=any_malformed_action,
            total_policy_actions=total_policy_actions,
        )
        task_success = status is RolloutStatus.SUCCESS
        return RolloutEpisodeResult(
            evaluation_id=evaluation_id,
            run_fingerprint=self.run_fingerprint,
            checkpoint_fingerprint=self.checkpoint_fingerprint,
            schedule_digest=self.schedule_digest,
            split=split,
            scene_seed=scene_seed,
            scene_id=episode_spec.scene_id,
            task_id=task_id,
            status=status,
            success=task_success,
            episode_steps=steps,
            final_evaluation=final_evaluation,
            target_in_target_bin=target_in_target_bin,
            target_in_wrong_bin=target_in_wrong_bin,
            wrong_object_in_target_bin=wrong_object_in_target_bin,
            target_grasped_any=target_grasped_any,
            wrong_object_grasped_any=wrong_grasped_any,
            target_off_table=final_evaluation.get("target_off_table", False),
            timeout=status in {RolloutStatus.TIMEOUT, RolloutStatus.TRUNCATED},
            invalid_action=status is RolloutStatus.INVALID_ACTION,
            inference_failure=status is RolloutStatus.INFERENCE_FAILURE,
            time_to_first_target_grasp_s=(
                first_grasp_step / hz if first_grasp_step is not None else None
            ),
            time_to_release_s=release_step / hz if release_step is not None else None,
            time_to_success_s=success_step / hz if success_step is not None else None,
            cumulative_action_magnitude=cumulative,
            action_min=action_min,
            action_max=action_max,
            inference_latency_ms=tuple(inference_latencies),
            environment_step_latency_ms=tuple(environment_latencies),
            total_episode_duration_s=total_duration,
            runtime_fingerprint=self.runtime_fingerprint,
            action_bound_mode=self.action_bound_config.mode,
            task_success=task_success,
            strict_unprojected_success=(
                task_success and projection_summary.projected_action_count == 0
            ),
            action_projection_summary=projection_summary.to_dict(),
            failure_reason=reason,
        )


def latency_percentiles(values: tuple[float, ...]) -> dict[str, float | None]:
    """Return p50/p95/p99 without inventing values for empty samples."""
    if not values:
        return {"p50": None, "p95": None, "p99": None}
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError("latency samples must be finite and non-negative")
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


__all__ = [
    "ActManiSkillRolloutAdapter",
    "RolloutContractError",
    "ProjectionRecordSink",
    "TaskConditioner",
    "build_policy_observation",
    "latency_percentiles",
]
