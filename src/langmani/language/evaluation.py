"""JSON-serializable dispatch and control evidence for M5A evaluation."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from langmani.datasets.identity import sha256_hex
from langmani.environments.specs import TaskSpec, stable_task_id
from langmani.language.failure_attribution import (
    FailureAttribution,
    FailureAttributionEvidence,
    attribute_end_to_end,
)
from langmani.language.router_types import RouterDecision, RouterStatus
from langmani.policies.act_action_bounds import ActionProjectionRecord

ACTION_STREAM_EVIDENCE_SCHEMA_VERSION = "langmani-m5a-action-stream-evidence-v0"
CONTROL_EXECUTION_RESULT_SCHEMA_VERSION = "langmani-m5a-control-execution-result-v0"
CONTROLLER_DISPATCH_RECORD_SCHEMA_VERSION = "langmani-m5a-controller-dispatch-record-v0"
END_TO_END_RESULT_SCHEMA_VERSION = "langmani-m5a-end-to-end-result-v0"
FAILURE_ATTRIBUTION_SUMMARY_SCHEMA_VERSION = "langmani-m5a-failure-summary-v0"
_ACTION_DIMENSION = 8
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_DECISION_FINGERPRINT_RE = re.compile(r"^langmani-m5a-router-decision-[0-9a-f]{64}$")


class M5AEvaluationError(ValueError):
    """Raised when M5A evaluation evidence is contradictory or malformed."""


def _fingerprint(value: object) -> str:
    return f"sha256:{sha256_hex(value)}"


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise M5AEvaluationError(f"{label} must be sha256:<64 lowercase hex>")
    return value


def _action_vector(value: Sequence[float], label: str) -> tuple[float, ...]:
    vector = tuple(float(item) for item in value)
    if len(vector) != _ACTION_DIMENSION or not all(math.isfinite(item) for item in vector):
        raise M5AEvaluationError(f"{label} must contain eight finite values")
    return vector


def _array_action(value: object, label: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.size != _ACTION_DIMENSION:
        raise M5AEvaluationError(f"{label} must contain eight values")
    return _action_vector(array.reshape(-1).tolist(), label)


def _boolean_mapping(value: Mapping[str, bool], label: str) -> Mapping[str, bool]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) and isinstance(item, bool) for key, item in value.items()
    ):
        raise M5AEvaluationError(f"{label} must be a string-to-boolean mapping")
    return dict(sorted(value.items()))


@dataclass(frozen=True, slots=True)
class ActionStreamEvidence:
    """Separate raw, binary-transformed, projected, and executed streams.

    The locked M5A runtime uses ``project`` gripper handling, so every entry in
    ``binary_transformed_actions`` is explicitly ``None``.  Keeping that absent
    transform as an aligned stream prevents a later binary-gripper runtime from
    being mistaken for ordinary bounds projection.
    """

    raw_actions: tuple[tuple[float, ...], ...]
    binary_transformed_actions: tuple[tuple[float, ...] | None, ...]
    projected_actions: tuple[tuple[float, ...] | None, ...]
    executed_actions: tuple[tuple[float, ...], ...]
    raw_violation_count: int
    projected_action_count: int
    arm_projected_component_count: int
    gripper_projected_component_count: int
    malformed_action_count: int = 0
    nonfinite_action_count: int = 0
    schema_version: str = ACTION_STREAM_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ACTION_STREAM_EVIDENCE_SCHEMA_VERSION:
            raise M5AEvaluationError("unsupported action-stream evidence schema")
        object.__setattr__(
            self,
            "raw_actions",
            tuple(_action_vector(item, "raw action") for item in self.raw_actions),
        )
        object.__setattr__(
            self,
            "binary_transformed_actions",
            tuple(
                None if item is None else _action_vector(item, "binary-transformed action")
                for item in self.binary_transformed_actions
            ),
        )
        object.__setattr__(
            self,
            "projected_actions",
            tuple(
                None if item is None else _action_vector(item, "projected action")
                for item in self.projected_actions
            ),
        )
        object.__setattr__(
            self,
            "executed_actions",
            tuple(_action_vector(item, "executed action") for item in self.executed_actions),
        )
        if len(self.binary_transformed_actions) != len(self.raw_actions):
            raise M5AEvaluationError(
                "binary-transformed actions must align one-for-one with raw actions"
            )
        if len(self.projected_actions) != len(self.raw_actions):
            raise M5AEvaluationError("projected actions must align one-for-one with raw actions")
        if sum(item is not None for item in self.projected_actions) != len(self.executed_actions):
            raise M5AEvaluationError("executed actions must align with non-rejected projections")
        for name in (
            "raw_violation_count",
            "projected_action_count",
            "arm_projected_component_count",
            "gripper_projected_component_count",
            "malformed_action_count",
            "nonfinite_action_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise M5AEvaluationError(f"{name} must be a non-negative integer")
        if self.projected_action_count > len(self.raw_actions):
            raise M5AEvaluationError("projected_action_count exceeds raw action count")

    @property
    def raw_fingerprint(self) -> str:
        return _fingerprint(self.raw_actions)

    @property
    def projected_fingerprint(self) -> str:
        return _fingerprint(self.projected_actions)

    @property
    def binary_transformed_fingerprint(self) -> str:
        return _fingerprint(self.binary_transformed_actions)

    @property
    def executed_fingerprint(self) -> str:
        return _fingerprint(self.executed_actions)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "raw_actions": [list(item) for item in self.raw_actions],
            "binary_transformed_actions": [
                None if item is None else list(item) for item in self.binary_transformed_actions
            ],
            "projected_actions": [
                None if item is None else list(item) for item in self.projected_actions
            ],
            "executed_actions": [list(item) for item in self.executed_actions],
            "raw_fingerprint": self.raw_fingerprint,
            "binary_transformed_fingerprint": self.binary_transformed_fingerprint,
            "projected_fingerprint": self.projected_fingerprint,
            "executed_fingerprint": self.executed_fingerprint,
            "raw_violation_count": self.raw_violation_count,
            "projected_action_count": self.projected_action_count,
            "arm_projected_component_count": self.arm_projected_component_count,
            "gripper_projected_component_count": self.gripper_projected_component_count,
            "malformed_action_count": self.malformed_action_count,
            "nonfinite_action_count": self.nonfinite_action_count,
        }

    @classmethod
    def empty(cls) -> ActionStreamEvidence:
        return cls(
            raw_actions=(),
            binary_transformed_actions=(),
            projected_actions=(),
            executed_actions=(),
            raw_violation_count=0,
            projected_action_count=0,
            arm_projected_component_count=0,
            gripper_projected_component_count=0,
        )

    @classmethod
    def from_projection_records(
        cls,
        records: Sequence[ActionProjectionRecord],
        *,
        malformed_action_count: int = 0,
        nonfinite_action_count: int = 0,
    ) -> ActionStreamEvidence:
        raw: list[tuple[float, ...]] = []
        binary_transformed: list[tuple[float, ...] | None] = []
        projected: list[tuple[float, ...] | None] = []
        executed: list[tuple[float, ...]] = []
        raw_violations = 0
        projected_actions = 0
        arm_components = 0
        gripper_components = 0
        for record in records:
            raw_array = np.asarray(record.raw_action, dtype=np.float64)
            if raw_array.size != _ACTION_DIMENSION or not np.all(np.isfinite(raw_array)):
                if malformed_action_count == 0 and nonfinite_action_count == 0:
                    raise M5AEvaluationError("invalid raw action lacks an explicit audit count")
                continue
            raw.append(_array_action(raw_array, "raw action"))
            # The frozen PerTask controller uses project-only action handling.
            # No binary gripper transform is part of this runtime.
            binary_transformed.append(None)
            mask = np.asarray(record.violation_mask, dtype=np.bool_).reshape(-1)
            if mask.size != _ACTION_DIMENSION:
                raise M5AEvaluationError("projection violation mask must have eight components")
            raw_violations += int(bool(np.any(mask)))
            arm_components += int(np.count_nonzero(mask[:7]))
            gripper_components += int(bool(mask[7]))
            projected_actions += int(record.was_projected)
            if record.executed_action is None:
                projected.append(None)
            else:
                vector = _array_action(record.executed_action, "runtime-projected action")
                projected.append(vector)
                executed.append(vector)
        return cls(
            raw_actions=tuple(raw),
            binary_transformed_actions=tuple(binary_transformed),
            projected_actions=tuple(projected),
            executed_actions=tuple(executed),
            raw_violation_count=raw_violations,
            projected_action_count=projected_actions,
            arm_projected_component_count=arm_components,
            gripper_projected_component_count=gripper_components,
            malformed_action_count=malformed_action_count,
            nonfinite_action_count=nonfinite_action_count,
        )


@dataclass(frozen=True, slots=True)
class ControlExecutionResult:
    """Controller-only outcome, intentionally independent of router correctness."""

    success: bool
    status: str
    environment_step_count: int
    final_evaluation: Mapping[str, bool]
    action_evidence: ActionStreamEvidence
    inference_latency_ms: tuple[float, ...] = ()
    environment_latency_ms: tuple[float, ...] = ()
    failure_reason: str | None = None
    controller_inference_failed: bool = False
    invalid_action: bool = False
    timeout: bool = False
    environment_failed: bool = False
    infrastructure_failed: bool = False
    wrong_object_interaction: bool = False
    schema_version: str = CONTROL_EXECUTION_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONTROL_EXECUTION_RESULT_SCHEMA_VERSION:
            raise M5AEvaluationError("unsupported control-execution schema")
        if (
            not isinstance(self.success, bool)
            or not isinstance(self.status, str)
            or not self.status
        ):
            raise M5AEvaluationError("control success/status are malformed")
        if (
            isinstance(self.environment_step_count, bool)
            or not isinstance(self.environment_step_count, int)
            or self.environment_step_count < 0
        ):
            raise M5AEvaluationError("environment_step_count must be non-negative")
        object.__setattr__(
            self,
            "final_evaluation",
            _boolean_mapping(self.final_evaluation, "final_evaluation"),
        )
        if not isinstance(self.action_evidence, ActionStreamEvidence):
            raise M5AEvaluationError("action_evidence must be ActionStreamEvidence")
        for name in (
            "controller_inference_failed",
            "invalid_action",
            "timeout",
            "environment_failed",
            "infrastructure_failed",
            "wrong_object_interaction",
        ):
            if not isinstance(getattr(self, name), bool):
                raise M5AEvaluationError(f"{name} must be boolean")
        failure_count = sum(
            int(getattr(self, name))
            for name in (
                "controller_inference_failed",
                "invalid_action",
                "timeout",
                "environment_failed",
                "infrastructure_failed",
            )
        )
        if failure_count > 1 or (self.success and failure_count):
            raise M5AEvaluationError("control failure signals must be mutually exclusive")
        for name in ("inference_latency_ms", "environment_latency_ms"):
            values = tuple(float(value) for value in getattr(self, name))
            if not all(math.isfinite(value) and value >= 0 for value in values):
                raise M5AEvaluationError(f"{name} must contain finite non-negative values")
            object.__setattr__(self, name, values)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "success": self.success,
            "status": self.status,
            "environment_step_count": self.environment_step_count,
            "final_evaluation": dict(self.final_evaluation),
            "action_evidence": self.action_evidence.to_dict(),
            "inference_latency_ms": list(self.inference_latency_ms),
            "environment_latency_ms": list(self.environment_latency_ms),
            "failure_reason": self.failure_reason,
            "controller_inference_failed": self.controller_inference_failed,
            "invalid_action": self.invalid_action,
            "timeout": self.timeout,
            "environment_failed": self.environment_failed,
            "infrastructure_failed": self.infrastructure_failed,
            "wrong_object_interaction": self.wrong_object_interaction,
        }


@dataclass(frozen=True, slots=True)
class ControllerDispatchRecord:
    """Auditable proof that a decision selected exactly one controller or none."""

    decision_fingerprint: str
    registry_fingerprint: str
    oracle_task_id: str | None
    predicted_task_id: str | None
    controller_task_id: str | None
    controller_run_fingerprint: str | None
    controller_checkpoint_fingerprint: str | None
    dispatched: bool
    controller_loaded: bool
    policy_called: bool
    environment_reset_called: bool
    environment_step_count: int
    safe_rejection: bool
    controller_load_failed: bool = False
    schema_version: str = CONTROLLER_DISPATCH_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONTROLLER_DISPATCH_RECORD_SCHEMA_VERSION:
            raise M5AEvaluationError("unsupported controller-dispatch schema")
        for name in (
            "dispatched",
            "controller_loaded",
            "policy_called",
            "environment_reset_called",
            "safe_rejection",
            "controller_load_failed",
        ):
            if not isinstance(getattr(self, name), bool):
                raise M5AEvaluationError(f"{name} must be boolean")
        if (
            isinstance(self.environment_step_count, bool)
            or not isinstance(self.environment_step_count, int)
            or self.environment_step_count < 0
        ):
            raise M5AEvaluationError("dispatch environment_step_count must be non-negative")
        controller_fields = (
            self.controller_task_id,
            self.controller_run_fingerprint,
            self.controller_checkpoint_fingerprint,
        )
        if self.safe_rejection:
            if any(controller_fields) or any(
                (
                    self.dispatched,
                    self.controller_loaded,
                    self.policy_called,
                    self.environment_reset_called,
                    bool(self.environment_step_count),
                    self.controller_load_failed,
                )
            ):
                raise M5AEvaluationError("safe rejection must perform zero controller/env work")
            if self.predicted_task_id is not None:
                raise M5AEvaluationError("safe rejection cannot contain a predicted TaskSpec")
        if self.dispatched:
            if not all(controller_fields) or not self.controller_loaded:
                raise M5AEvaluationError("dispatch requires one fully identified loaded controller")
            if not self.policy_called or not self.environment_reset_called:
                raise M5AEvaluationError("dispatch must reset and call exactly one controller")
            if self.predicted_task_id != self.controller_task_id:
                raise M5AEvaluationError(
                    "dispatcher selected a controller different from prediction"
                )
        elif self.policy_called or self.environment_reset_called or self.environment_step_count:
            raise M5AEvaluationError("non-dispatch record cannot contain policy or env activity")
        if self.controller_load_failed and (self.controller_loaded or self.dispatched):
            raise M5AEvaluationError("failed controller load cannot be marked loaded or dispatched")
        if _DECISION_FINGERPRINT_RE.fullmatch(self.decision_fingerprint) is None:
            raise M5AEvaluationError("decision_fingerprint has an invalid M5A identity")
        _require_digest(self.registry_fingerprint, "registry_fingerprint")
        for label, value in (
            ("controller_run_fingerprint", self.controller_run_fingerprint),
            ("controller_checkpoint_fingerprint", self.controller_checkpoint_fingerprint),
        ):
            if value is not None:
                _require_digest(value, label)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "decision_fingerprint": self.decision_fingerprint,
            "registry_fingerprint": self.registry_fingerprint,
            "oracle_task_id": self.oracle_task_id,
            "predicted_task_id": self.predicted_task_id,
            "controller_task_id": self.controller_task_id,
            "controller_run_fingerprint": self.controller_run_fingerprint,
            "controller_checkpoint_fingerprint": self.controller_checkpoint_fingerprint,
            "dispatched": self.dispatched,
            "controller_loaded": self.controller_loaded,
            "policy_called": self.policy_called,
            "environment_reset_called": self.environment_reset_called,
            "environment_step_count": self.environment_step_count,
            "safe_rejection": self.safe_rejection,
            "controller_load_failed": self.controller_load_failed,
        }


@dataclass(frozen=True, slots=True)
class EndToEndEpisodeResult:
    """One routeable control episode or one non-executed safe rejection."""

    evaluation_id: str
    scene_seed: int | None
    oracle_task_spec: TaskSpec | None
    decision: RouterDecision
    dispatch: ControllerDispatchRecord
    control: ControlExecutionResult | None
    failure_attribution: FailureAttribution | None
    schema_version: str = END_TO_END_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != END_TO_END_RESULT_SCHEMA_VERSION:
            raise M5AEvaluationError("unsupported end-to-end result schema")
        if not isinstance(self.evaluation_id, str) or not self.evaluation_id:
            raise M5AEvaluationError("evaluation_id must be non-empty")
        if self.scene_seed is not None and (
            isinstance(self.scene_seed, bool)
            or not isinstance(self.scene_seed, int)
            or self.scene_seed < 0
        ):
            raise M5AEvaluationError("scene_seed must be non-negative or None")
        if self.dispatch.decision_fingerprint != self.decision.decision_fingerprint:
            raise M5AEvaluationError("dispatch record refers to another router decision")
        oracle_task_id = (
            None if self.oracle_task_spec is None else stable_task_id(self.oracle_task_spec)
        )
        if self.dispatch.oracle_task_id != oracle_task_id:
            raise M5AEvaluationError("dispatch oracle task differs from episode oracle")
        if self.dispatch.safe_rejection:
            if self.control is not None:
                raise M5AEvaluationError("safe rejection cannot contain a control result")
            if self.decision.status is RouterStatus.ROUTE:
                raise M5AEvaluationError("route decision cannot be a safe rejection")
        elif self.dispatch.dispatched:
            if self.control is None:
                raise M5AEvaluationError("dispatched episode requires a control result")
            if self.control.environment_step_count != self.dispatch.environment_step_count:
                raise M5AEvaluationError("dispatch and control step counts disagree")
        expected_attribution = attribute_end_to_end(
            FailureAttributionEvidence(
                expected_task_spec=self.oracle_task_spec,
                decision=self.decision,
                controller_task_id=self.dispatch.controller_task_id,
                controller_dispatched=self.dispatch.dispatched,
                control_success=None if self.control is None else self.control.success,
                controller_load_failed=self.dispatch.controller_load_failed,
                controller_inference_failed=(
                    False if self.control is None else self.control.controller_inference_failed
                ),
                invalid_action=False if self.control is None else self.control.invalid_action,
                timeout=False if self.control is None else self.control.timeout,
                environment_failed=(
                    False if self.control is None else self.control.environment_failed
                ),
                infrastructure_failed=(
                    False if self.control is None else self.control.infrastructure_failed
                ),
            )
        )
        if self.failure_attribution is not expected_attribution:
            raise M5AEvaluationError("failure attribution disagrees with routing/control evidence")

    @property
    def end_to_end_success(self) -> bool:
        return self.failure_attribution is FailureAttribution.ROUTING_CORRECT_CONTROL_SUCCESS

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "evaluation_id": self.evaluation_id,
            "scene_seed": self.scene_seed,
            "oracle_task_spec": (
                None if self.oracle_task_spec is None else self.oracle_task_spec.to_dict()
            ),
            "oracle_task_id": (
                None if self.oracle_task_spec is None else stable_task_id(self.oracle_task_spec)
            ),
            "decision": self.decision.to_dict(),
            "dispatch": self.dispatch.to_dict(),
            "control": None if self.control is None else self.control.to_dict(),
            "failure_attribution": (
                None if self.failure_attribution is None else self.failure_attribution.value
            ),
            "end_to_end_success": self.end_to_end_success,
        }


@dataclass(frozen=True, slots=True)
class FailureAttributionSummary:
    """Direct observed counts without multiplying independently estimated rates."""

    total_results: int
    control_episode_count: int
    safe_rejection_count: int
    attribution_counts: Mapping[str, int]
    schema_version: str = FAILURE_ATTRIBUTION_SUMMARY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FAILURE_ATTRIBUTION_SUMMARY_SCHEMA_VERSION:
            raise M5AEvaluationError("unsupported failure-summary schema")
        expected = {item.value for item in FailureAttribution}
        if set(self.attribution_counts) != expected:
            raise M5AEvaluationError("failure summary must contain all fourteen categories")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.attribution_counts.values()
        ):
            raise M5AEvaluationError("attribution counts must be non-negative integers")
        if sum(self.attribution_counts.values()) != self.control_episode_count:
            raise M5AEvaluationError("every control episode must have exactly one attribution")
        if self.total_results != self.control_episode_count + self.safe_rejection_count:
            raise M5AEvaluationError("summary result counts are inconsistent")
        object.__setattr__(
            self, "attribution_counts", dict(sorted(self.attribution_counts.items()))
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "total_results": self.total_results,
            "control_episode_count": self.control_episode_count,
            "safe_rejection_count": self.safe_rejection_count,
            "attribution_counts": dict(self.attribution_counts),
        }


def summarize_end_to_end(
    results: Sequence[EndToEndEpisodeResult],
) -> FailureAttributionSummary:
    counts: Counter[str] = Counter()
    correct_safe_rejections = 0
    for result in results:
        if result.failure_attribution is None:
            if not result.dispatch.safe_rejection:
                raise M5AEvaluationError("unattributed result is not a correct safe rejection")
            correct_safe_rejections += 1
            continue
        counts[result.failure_attribution.value] += 1
    complete_counts = {item.value: counts[item.value] for item in FailureAttribution}
    return FailureAttributionSummary(
        total_results=len(results),
        control_episode_count=len(results) - correct_safe_rejections,
        safe_rejection_count=correct_safe_rejections,
        attribution_counts=complete_counts,
    )


__all__ = [
    "ACTION_STREAM_EVIDENCE_SCHEMA_VERSION",
    "CONTROLLER_DISPATCH_RECORD_SCHEMA_VERSION",
    "CONTROL_EXECUTION_RESULT_SCHEMA_VERSION",
    "END_TO_END_RESULT_SCHEMA_VERSION",
    "ActionStreamEvidence",
    "ControlExecutionResult",
    "ControllerDispatchRecord",
    "EndToEndEpisodeResult",
    "FailureAttributionSummary",
    "M5AEvaluationError",
    "summarize_end_to_end",
]
