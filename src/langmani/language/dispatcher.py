"""Frozen PerTask controller loading and zero-work rejection dispatch for M5A."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np

from langmani.datasets.identity import sha256_hex
from langmani.environments.specs import TaskSpec, stable_task_id
from langmani.language.controller_registry import (
    LOCKED_EXECUTION_HORIZON,
    ControllerRegistry,
    ControllerRegistryEntry,
    ControllerRegistryError,
    ControllerRegistryLocators,
    validate_active_controller_environment,
)
from langmani.language.evaluation import (
    ActionStreamEvidence,
    ControlExecutionResult,
    ControllerDispatchRecord,
    EndToEndEpisodeResult,
)
from langmani.language.failure_attribution import (
    FailureAttributionEvidence,
    attribute_end_to_end,
)
from langmani.language.router_types import RouterDecision, RouterStatus
from langmani.policies.act_action_bounds import ActionProjectionRecord
from langmani.policies.act_rollout import ActManiSkillRolloutAdapter
from langmani.policies.act_types import (
    ActEvaluationConfig,
    ActVariant,
    EvaluationSplit,
    RolloutStatus,
)
from langmani.policies.m42_evaluation import M42PolicyKind, load_m4_checkpoint_context
from langmani.policies.m42_runtime import ExecutionHorizonPolicyV0
from langmani.policies.m42_types import ExecutionHorizonConfig


class ControllerDispatchError(RuntimeError):
    """Base error for explicit M5A controller-dispatch failures."""


class ControllerLoadError(ControllerDispatchError):
    """Raised when immutable controller artifacts cannot be loaded exactly."""


@dataclass(frozen=True, slots=True)
class ControllerExecutionTrace:
    """Control result plus proof of which runtime boundaries were entered."""

    result: ControlExecutionResult
    policy_called: bool
    environment_reset_called: bool
    episode_audit: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.result, ControlExecutionResult):
            raise TypeError("result must be a ControlExecutionResult")
        if not isinstance(self.policy_called, bool) or not isinstance(
            self.environment_reset_called, bool
        ):
            raise TypeError("execution trace flags must be boolean")
        if self.episode_audit is not None and not isinstance(self.episode_audit, Mapping):
            raise TypeError("episode_audit must be a mapping when present")
        if self.episode_audit is not None:
            object.__setattr__(self, "episode_audit", dict(self.episode_audit))


class PerTaskControllerExecutor(Protocol):
    """One loaded controller bound to exactly one registry entry."""

    controller_task_id: str

    def run_episode(
        self,
        *,
        evaluation_id: str,
        scene_seed: int,
        oracle_task_spec: TaskSpec,
    ) -> ControllerExecutionTrace: ...


class PerTaskControllerLoader(Protocol):
    """Injectable loading boundary used by real and fixture dispatch."""

    def load(
        self,
        entry: ControllerRegistryEntry,
        *,
        environment: object,
    ) -> PerTaskControllerExecutor: ...


def _fingerprint(value: object) -> str:
    return f"sha256:{sha256_hex(value)}"


class _ActPerTaskExecutor:
    """Existing M4 rollout adapter bound to one strict-loaded PerTask policy."""

    def __init__(
        self,
        *,
        entry: ControllerRegistryEntry,
        environment: object,
        loaded: object,
        schedule_digest: str,
        evaluation: ActEvaluationConfig,
    ) -> None:
        self.entry = entry
        self.controller_task_id = entry.task_id
        self._projection_records: list[ActionProjectionRecord] = []
        policy = getattr(loaded, "policy", None)
        preprocessor = getattr(loaded, "preprocessor", None)
        postprocessor = getattr(loaded, "postprocessor", None)
        if policy is None or preprocessor is None or postprocessor is None:
            raise ControllerLoadError("strict checkpoint load omitted policy processors")
        horizon_policy = ExecutionHorizonPolicyV0(
            cast(Any, policy), ExecutionHorizonConfig(LOCKED_EXECUTION_HORIZON)
        )

        def retain_record(_evaluation_id: str, record: ActionProjectionRecord) -> None:
            self._projection_records.append(record)

        runtime_fingerprint = _fingerprint(
            {
                "schema_version": "langmani-m5a-per-task-runtime-v0",
                "registry_entry": entry.to_dict(),
                "schedule_digest": schedule_digest,
            }
        )
        self._adapter = ActManiSkillRolloutAdapter(
            env=environment,
            policy=horizon_policy,
            preprocessor=cast(Any, preprocessor),
            postprocessor=cast(Any, postprocessor),
            variant=ActVariant.PER_TASK,
            run_fingerprint=entry.run_fingerprint,
            checkpoint_fingerprint=entry.checkpoint_fingerprint,
            schedule_digest=schedule_digest,
            runtime_fingerprint=runtime_fingerprint,
            action_bound_config=entry.action_bound_config,
            evaluation=evaluation,
            projection_record_sink=retain_record,
        )

    def run_episode(
        self,
        *,
        evaluation_id: str,
        scene_seed: int,
        oracle_task_spec: TaskSpec,
    ) -> ControllerExecutionTrace:
        # Deliberately pass the oracle TaskSpec to the environment.  The selected
        # controller identity remains self.controller_task_id in the dispatch record.
        self._projection_records.clear()
        rollout = self._adapter.run_episode(
            evaluation_id=evaluation_id,
            split=EvaluationSplit.DEVELOPMENT,
            scene_seed=scene_seed,
            task_spec=oracle_task_spec,
        )
        summary = rollout.action_projection_summary
        malformed = int(bool(summary.get("any_malformed_action", False)))
        nonfinite = int(bool(summary.get("any_nonfinite_action", False)))
        action_evidence = ActionStreamEvidence.from_projection_records(
            self._projection_records,
            malformed_action_count=malformed,
            nonfinite_action_count=nonfinite,
        )
        return ControllerExecutionTrace(
            result=ControlExecutionResult(
                success=rollout.success,
                status=rollout.status.value,
                environment_step_count=rollout.episode_steps,
                final_evaluation=rollout.final_evaluation,
                action_evidence=action_evidence,
                inference_latency_ms=rollout.inference_latency_ms,
                environment_latency_ms=rollout.environment_step_latency_ms,
                failure_reason=rollout.failure_reason,
                controller_inference_failed=rollout.inference_failure,
                invalid_action=rollout.invalid_action,
                timeout=rollout.timeout,
                environment_failed=rollout.status is RolloutStatus.ENVIRONMENT_FAILURE,
                wrong_object_interaction=rollout.wrong_object_grasped_any,
            ),
            policy_called=True,
            environment_reset_called=True,
            episode_audit={
                "schema_version": "langmani-m5a-controller-episode-audit-v0",
                "policy_reset_called": True,
                "rollout": rollout.to_dict(),
            },
        )


class StrictPerTaskControllerLoader:
    """Strict local-only checkpoint loader with one executor cache per environment."""

    def __init__(
        self,
        *,
        registry: ControllerRegistry,
        locators: ControllerRegistryLocators,
        schedule_digest: str,
        evaluation: ActEvaluationConfig | None = None,
    ) -> None:
        if locators.registry_fingerprint != registry.registry_fingerprint:
            raise ControllerLoadError("controller locators refer to another registry")
        self.registry = registry
        self.locators = locators
        self.schedule_digest = schedule_digest
        self.evaluation = evaluation or ActEvaluationConfig()
        self._cache: dict[tuple[int, str], _ActPerTaskExecutor] = {}

    def load(
        self,
        entry: ControllerRegistryEntry,
        *,
        environment: object,
    ) -> PerTaskControllerExecutor:
        canonical = self.registry.require(entry.task_id)
        if canonical != entry:
            raise ControllerLoadError("dispatcher entry differs from immutable registry")
        key = (id(environment), entry.task_id)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        location = self.locators.require(entry.task_id)
        if not location.run_root.is_dir() or any(
            candidate.is_symlink() or candidate.is_junction()
            for candidate in (location.run_root, *location.run_root.parents)
        ):
            raise ControllerLoadError("controller run root is missing or traverses a link")
        try:
            validate_active_controller_environment(environment, self.registry)
        except ControllerRegistryError as error:
            raise ControllerLoadError(str(error)) from error
        try:
            context = load_m4_checkpoint_context(
                location.run_root,
                policy_kind=M42PolicyKind.PER_TASK,
                expected_checkpoint_fingerprint=entry.checkpoint_fingerprint,
                expected_run_fingerprint=entry.run_fingerprint,
                expected_dataset_fingerprint=entry.m3b_export_fingerprint,
                expected_task_id=entry.task_id,
            )
        except Exception as error:
            raise ControllerLoadError(
                f"strict PerTask checkpoint load failed: {type(error).__name__}: {error}"
            ) from error
        components = context.loaded.component_fingerprints
        if (
            components.model != entry.model_component_fingerprint
            or components.preprocessor != entry.preprocessor_fingerprint
            or components.postprocessor != entry.postprocessor_fingerprint
            or context.loaded.record.global_step != entry.checkpoint_step
            or context.loaded.record.relative_path != location.checkpoint_relative_path
            or context.descriptor.statistics_fingerprint != entry.train_statistics_fingerprint
            or context.descriptor.split_digest != entry.m3b_split_manifest_digest
            or context.descriptor.git_commit != entry.producer_git_commit
        ):
            raise ControllerLoadError("loaded controller differs from registry semantics")
        executor = _ActPerTaskExecutor(
            entry=entry,
            environment=environment,
            loaded=context.loaded,
            schedule_digest=self.schedule_digest,
            evaluation=self.evaluation,
        )
        self._cache[key] = executor
        return executor


class ControllerDispatcher:
    """Select exactly one frozen controller, or return before all runtime work."""

    def __init__(self, *, registry: ControllerRegistry, loader: PerTaskControllerLoader) -> None:
        self.registry = registry
        self.loader = loader
        self.last_episode_audit: Mapping[str, object] | None = None

    def dispatch(
        self,
        decision: RouterDecision,
        *,
        environment: object,
        evaluation_id: str,
        oracle_task_spec: TaskSpec | None,
        scene_seed: int | None,
    ) -> EndToEndEpisodeResult:
        self.last_episode_audit = None
        oracle_task_id = None if oracle_task_spec is None else stable_task_id(oracle_task_spec)
        if decision.status is not RouterStatus.ROUTE:
            dispatch = ControllerDispatchRecord(
                decision_fingerprint=decision.decision_fingerprint,
                registry_fingerprint=self.registry.registry_fingerprint,
                oracle_task_id=oracle_task_id,
                predicted_task_id=None,
                controller_task_id=None,
                controller_run_fingerprint=None,
                controller_checkpoint_fingerprint=None,
                dispatched=False,
                controller_loaded=False,
                policy_called=False,
                environment_reset_called=False,
                environment_step_count=0,
                safe_rejection=True,
                controller_load_failed=False,
            )
            attribution = attribute_end_to_end(
                FailureAttributionEvidence(
                    expected_task_spec=oracle_task_spec,
                    decision=decision,
                )
            )
            return EndToEndEpisodeResult(
                evaluation_id=evaluation_id,
                scene_seed=scene_seed,
                oracle_task_spec=oracle_task_spec,
                decision=decision,
                dispatch=dispatch,
                control=None,
                failure_attribution=attribution,
            )
        if decision.task_id is None or decision.task_spec is None:
            raise ControllerDispatchError("route decision lacks an executable TaskSpec")
        if oracle_task_spec is None or scene_seed is None:
            # A routed command in a rejection-only language record is a missed
            # rejection.  It is classified but never executed without an oracle
            # M1 task/scene contract.
            dispatch = ControllerDispatchRecord(
                decision_fingerprint=decision.decision_fingerprint,
                registry_fingerprint=self.registry.registry_fingerprint,
                oracle_task_id=None,
                predicted_task_id=decision.task_id,
                controller_task_id=None,
                controller_run_fingerprint=None,
                controller_checkpoint_fingerprint=None,
                dispatched=False,
                controller_loaded=False,
                policy_called=False,
                environment_reset_called=False,
                environment_step_count=0,
                safe_rejection=False,
                controller_load_failed=False,
            )
            return EndToEndEpisodeResult(
                evaluation_id=evaluation_id,
                scene_seed=scene_seed,
                oracle_task_spec=None,
                decision=decision,
                dispatch=dispatch,
                control=None,
                failure_attribution=attribute_end_to_end(
                    FailureAttributionEvidence(expected_task_spec=None, decision=decision)
                ),
            )
        try:
            entry = self.registry.require(decision.task_id)
        except ControllerRegistryError as error:
            raise ControllerDispatchError("router selected an unsupported controller") from error
        try:
            executor = self.loader.load(entry, environment=environment)
        except ControllerLoadError:
            dispatch = ControllerDispatchRecord(
                decision_fingerprint=decision.decision_fingerprint,
                registry_fingerprint=self.registry.registry_fingerprint,
                oracle_task_id=oracle_task_id,
                predicted_task_id=decision.task_id,
                controller_task_id=None,
                controller_run_fingerprint=None,
                controller_checkpoint_fingerprint=None,
                dispatched=False,
                controller_loaded=False,
                policy_called=False,
                environment_reset_called=False,
                environment_step_count=0,
                safe_rejection=False,
                controller_load_failed=True,
            )
            return EndToEndEpisodeResult(
                evaluation_id=evaluation_id,
                scene_seed=scene_seed,
                oracle_task_spec=oracle_task_spec,
                decision=decision,
                dispatch=dispatch,
                control=None,
                failure_attribution=attribute_end_to_end(
                    FailureAttributionEvidence(
                        expected_task_spec=oracle_task_spec,
                        decision=decision,
                        controller_load_failed=True,
                    )
                ),
            )
        if executor.controller_task_id != entry.task_id:
            raise ControllerDispatchError("loader returned a different PerTask controller")
        trace = executor.run_episode(
            evaluation_id=evaluation_id,
            scene_seed=scene_seed,
            oracle_task_spec=oracle_task_spec,
        )
        self.last_episode_audit = trace.episode_audit
        control = trace.result
        dispatch = ControllerDispatchRecord(
            decision_fingerprint=decision.decision_fingerprint,
            registry_fingerprint=self.registry.registry_fingerprint,
            oracle_task_id=oracle_task_id,
            predicted_task_id=decision.task_id,
            controller_task_id=entry.task_id,
            controller_run_fingerprint=entry.run_fingerprint,
            controller_checkpoint_fingerprint=entry.checkpoint_fingerprint,
            dispatched=True,
            controller_loaded=True,
            policy_called=trace.policy_called,
            environment_reset_called=trace.environment_reset_called,
            environment_step_count=control.environment_step_count,
            safe_rejection=False,
            controller_load_failed=False,
        )
        attribution = attribute_end_to_end(
            FailureAttributionEvidence(
                expected_task_spec=oracle_task_spec,
                decision=decision,
                controller_task_id=entry.task_id,
                controller_dispatched=True,
                control_success=control.success,
                controller_inference_failed=control.controller_inference_failed,
                invalid_action=control.invalid_action,
                timeout=control.timeout,
                environment_failed=control.environment_failed,
                infrastructure_failed=control.infrastructure_failed,
            )
        )
        return EndToEndEpisodeResult(
            evaluation_id=evaluation_id,
            scene_seed=scene_seed,
            oracle_task_spec=oracle_task_spec,
            decision=decision,
            dispatch=dispatch,
            control=control,
            failure_attribution=attribution,
        )


class FixturePerTaskControllerExecutor:
    """Deterministic CPU fixture proving oracle/predicted task separation."""

    def __init__(
        self,
        *,
        entry: ControllerRegistryEntry,
        environment: object,
        success: bool,
    ) -> None:
        self.entry = entry
        self.environment = environment
        self.success = success
        self.controller_task_id = entry.task_id
        self.policy_call_count = 0
        self.episode_reset_count = 0
        self.last_oracle_task_spec: TaskSpec | None = None

    def run_episode(
        self,
        *,
        evaluation_id: str,
        scene_seed: int,
        oracle_task_spec: TaskSpec,
    ) -> ControllerExecutionTrace:
        del evaluation_id
        self.episode_reset_count += 1
        self.last_oracle_task_spec = oracle_task_spec
        reset = getattr(self.environment, "reset", None)
        step = getattr(self.environment, "step", None)
        if not callable(reset) or not callable(step):
            raise ControllerDispatchError("fixture environment must expose reset() and step()")
        reset(seed=scene_seed, options={"task_spec": oracle_task_spec.to_dict()})
        self.policy_call_count += 1
        raw = tuple(float(index) / 100.0 for index in range(8))
        executed = tuple(max(-1.0, min(1.0, value)) for value in raw)
        step(np.asarray(executed, dtype=np.float32))
        result = ControlExecutionResult(
            success=self.success,
            status="success" if self.success else "timeout",
            environment_step_count=1,
            final_evaluation={"success": self.success, "target_off_table": False},
            action_evidence=ActionStreamEvidence(
                raw_actions=(raw,),
                binary_transformed_actions=(None,),
                projected_actions=(executed,),
                executed_actions=(executed,),
                raw_violation_count=0,
                projected_action_count=0,
                arm_projected_component_count=0,
                gripper_projected_component_count=0,
            ),
            timeout=not self.success,
        )
        return ControllerExecutionTrace(
            result=result,
            policy_called=True,
            environment_reset_called=True,
            episode_audit={
                "schema_version": "langmani-m5a-controller-episode-audit-v0",
                "policy_reset_called": True,
                "rollout": {
                    "scene_seed": scene_seed,
                    "task_id": stable_task_id(oracle_task_spec),
                    "success": self.success,
                    "episode_steps": 1,
                    "time_to_first_target_grasp_s": None,
                    "time_to_release_s": None,
                    "time_to_success_s": 0.05 if self.success else None,
                    "strict_unprojected_success": self.success,
                    "action_bound_mode": "project",
                    "action_projection_summary": {
                        "maximum_linf_correction": 0.0,
                        "any_nonfinite_action": False,
                        "any_malformed_action": False,
                    },
                },
            },
        )


class FixturePerTaskControllerLoader:
    """Injectable loader that records exact loads without touching checkpoints."""

    def __init__(self, *, success_by_task_id: Mapping[str, bool] | None = None) -> None:
        self.success_by_task_id = dict(success_by_task_id or {})
        self.load_count = 0
        self.executors: dict[tuple[int, str], FixturePerTaskControllerExecutor] = {}

    def load(
        self,
        entry: ControllerRegistryEntry,
        *,
        environment: object,
    ) -> PerTaskControllerExecutor:
        self.load_count += 1
        key = (id(environment), entry.task_id)
        executor = self.executors.get(key)
        if executor is None:
            executor = FixturePerTaskControllerExecutor(
                entry=entry,
                environment=environment,
                success=self.success_by_task_id.get(entry.task_id, True),
            )
            self.executors[key] = executor
        return executor


__all__ = [
    "ControllerDispatchError",
    "ControllerDispatcher",
    "ControllerExecutionTrace",
    "ControllerLoadError",
    "FixturePerTaskControllerExecutor",
    "FixturePerTaskControllerLoader",
    "PerTaskControllerExecutor",
    "PerTaskControllerLoader",
    "StrictPerTaskControllerLoader",
]
