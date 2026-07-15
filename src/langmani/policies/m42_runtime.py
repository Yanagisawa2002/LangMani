"""Auditable M4.2 ACT execution-horizon and gripper runtime adapters.

This module is deliberately separate from the completed M4 rollout path.  It
uses LeRobot's public ``predict_action_chunk`` API, consumes an explicitly
configured number of actions from each unchanged 50-action ACT chunk, and
optionally applies the versioned binary-gripper transform before the existing
M4.1 project boundary.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, Self, cast

import numpy as np
import torch

from langmani.datasets.identity import sha256_hex
from langmani.policies.act_action_bounds import (
    ActionBoundConfig,
    ActionBoundErrorKind,
    ActionBoundMode,
    ActionBoundProcessingError,
    ActionBoundResult,
    ActionProjectionRecord,
    BoundedActionEnvPostprocessorV0,
)
from langmani.policies.m42_types import (
    M42_ACTION_COMPONENTS,
    M42_GRIPPER_ACTION_INDEX,
    M42_GRIPPER_PROCESSOR_VERSION,
    M42_RUNTIME_SCHEMA_VERSION,
    ExecutionHorizonConfig,
    GripperRuntimeMode,
)

M42_HORIZON_METRICS_VERSION = "ExecutionHorizonMetricsV0"
M42_ACTION_RUNTIME_RECORD_VERSION = "M42ActionRuntimeRecordV0"
M42_ACTION_RUNTIME_SUMMARY_VERSION = "M42ActionRuntimeSummaryV0"

type ActionArray = np.ndarray | torch.Tensor


class M42RuntimeContractError(RuntimeError):
    """Raised when a policy/runtime value violates the frozen M4.2 contract."""


class ActionChunkPolicy(Protocol):
    """Small public-policy surface used by the execution-horizon adapter."""

    def reset(self) -> None: ...

    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor: ...


def _prefixed_fingerprint(value: object) -> str:
    return f"sha256:{sha256_hex(value)}"


def _copy_action(value: ActionArray) -> ActionArray:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _action_to_json(value: ActionArray) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return np.asarray(value).tolist()


def _count_true(value: ActionArray) -> int:
    if isinstance(value, torch.Tensor):
        return int(torch.count_nonzero(value).detach().cpu())
    return int(np.count_nonzero(value))


def _component_true_count(value: ActionArray, component: int) -> int:
    if isinstance(value, torch.Tensor):
        return int(torch.count_nonzero(value[..., component]).detach().cpu())
    return int(np.count_nonzero(value[..., component]))


@dataclass(frozen=True, slots=True)
class ExecutionHorizonMetrics:
    """Compact per-episode accounting for deterministic chunk consumption."""

    config_fingerprint: str
    actions_per_query: int
    policy_query_count: int
    executed_action_count: int
    pending_action_count: int
    discarded_on_last_reset: int
    maximum_queue_depth: int
    predicted_chunk_nan_count: int
    predicted_chunk_inf_count: int
    malformed_chunk_count: int
    active_task_id: str | None
    version: str = M42_HORIZON_METRICS_VERSION

    def __post_init__(self) -> None:
        ExecutionHorizonConfig(self.actions_per_query)
        for name in (
            "policy_query_count",
            "executed_action_count",
            "pending_action_count",
            "discarded_on_last_reset",
            "maximum_queue_depth",
            "predicted_chunk_nan_count",
            "predicted_chunk_inf_count",
            "malformed_chunk_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.active_task_id is not None and (
            not isinstance(self.active_task_id, str) or not self.active_task_id
        ):
            raise ValueError("active_task_id must be a non-empty string when present")
        if self.version != M42_HORIZON_METRICS_VERSION:
            raise ValueError("unknown execution-horizon metrics version")

    @property
    def actions_executed_per_query(self) -> float:
        if self.policy_query_count == 0:
            return 0.0
        return self.executed_action_count / self.policy_query_count

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "config_fingerprint": self.config_fingerprint,
            "actions_per_query": self.actions_per_query,
            "policy_query_count": self.policy_query_count,
            "executed_action_count": self.executed_action_count,
            "actions_executed_per_query": self.actions_executed_per_query,
            "pending_action_count": self.pending_action_count,
            "discarded_on_last_reset": self.discarded_on_last_reset,
            "maximum_queue_depth": self.maximum_queue_depth,
            "predicted_chunk_nan_count": self.predicted_chunk_nan_count,
            "predicted_chunk_inf_count": self.predicted_chunk_inf_count,
            "malformed_chunk_count": self.malformed_chunk_count,
            "active_task_id": self.active_task_id,
        }

    @property
    def fingerprint(self) -> str:
        return _prefixed_fingerprint(self.to_dict())


class ExecutionHorizonPolicyV0:
    """Consume H actions from each public ``predict_action_chunk`` query.

    ``begin_task`` is the task-aware episode boundary.  It always resets the
    wrapped policy and drops any queued actions before binding the new stable
    task ID, so actions can never leak between counterfactual tasks.
    """

    def __init__(self, policy: ActionChunkPolicy, config: ExecutionHorizonConfig) -> None:
        if not isinstance(config, ExecutionHorizonConfig):
            raise TypeError("config must be an ExecutionHorizonConfig")
        if not callable(getattr(policy, "reset", None)):
            raise TypeError("policy must expose reset()")
        if not callable(getattr(policy, "predict_action_chunk", None)):
            raise TypeError("policy must expose predict_action_chunk()")
        self.policy = policy
        self.config = config
        self._queue: deque[torch.Tensor] = deque()
        self._active_task_id: str | None = None
        self._policy_query_count = 0
        self._executed_action_count = 0
        self._discarded_on_last_reset = 0
        self._maximum_queue_depth = 0
        self._predicted_chunk_nan_count = 0
        self._predicted_chunk_inf_count = 0
        self._malformed_chunk_count = 0

    @property
    def config_fingerprint(self) -> str:
        return self.config.fingerprint

    @property
    def active_task_id(self) -> str | None:
        return self._active_task_id

    def reset(self) -> None:
        """Clear queue/counters and reset the wrapped policy."""
        discarded = len(self._queue)
        self._queue.clear()
        self._active_task_id = None
        self._policy_query_count = 0
        self._executed_action_count = 0
        self._discarded_on_last_reset = discarded
        self._maximum_queue_depth = 0
        self._predicted_chunk_nan_count = 0
        self._predicted_chunk_inf_count = 0
        self._malformed_chunk_count = 0
        self.policy.reset()

    def begin_task(self, task_id: str) -> None:
        """Start one task episode after unconditionally clearing prior state."""
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task_id must be a non-empty stable task ID")
        self.reset()
        self._active_task_id = task_id

    def _query_policy(self, batch: dict[str, torch.Tensor]) -> None:
        chunk = self.policy.predict_action_chunk(batch)
        if not isinstance(chunk, torch.Tensor):
            self._malformed_chunk_count += 1
            raise ActionBoundProcessingError(
                ActionBoundErrorKind.MALFORMED_ACTION,
                "predict_action_chunk must return a Torch tensor",
            )
        if chunk.ndim != 3 or chunk.shape[0] != 1 or chunk.shape[2] != M42_ACTION_COMPONENTS:
            self._malformed_chunk_count += 1
            raise ActionBoundProcessingError(
                ActionBoundErrorKind.MALFORMED_ACTION,
                "predicted action chunk must have shape [1, chunk, 8]",
            )
        if not chunk.is_floating_point():
            self._malformed_chunk_count += 1
            raise ActionBoundProcessingError(
                ActionBoundErrorKind.MALFORMED_ACTION,
                "predicted action chunk must use a floating dtype",
            )
        if not bool(torch.isfinite(chunk).all()):
            self._predicted_chunk_nan_count += int(torch.count_nonzero(torch.isnan(chunk)))
            self._predicted_chunk_inf_count += int(torch.count_nonzero(torch.isinf(chunk)))
            raise ActionBoundProcessingError(
                ActionBoundErrorKind.NONFINITE_ACTION,
                "predicted action chunk contains NaN or infinity",
            )
        if chunk.shape[1] != self.config.action_chunk_size:
            self._malformed_chunk_count += 1
            raise ActionBoundProcessingError(
                ActionBoundErrorKind.MALFORMED_ACTION,
                "frozen ACT policy must return the declared 50-action chunk",
            )
        for index in range(self.config.actions_per_query):
            self._queue.append(chunk[:, index, :].detach().clone())
        self._policy_query_count += 1
        self._maximum_queue_depth = max(self._maximum_queue_depth, len(self._queue))

    def select_action(
        self,
        batch: dict[str, torch.Tensor],
        *,
        task_id: str | None = None,
    ) -> torch.Tensor:
        """Return one queued action, re-querying only when the H-action queue is empty."""
        if task_id is not None:
            if self._active_task_id is None:
                raise M42RuntimeContractError(
                    "begin_task(task_id) is required before task-bound action selection"
                )
            if task_id != self._active_task_id:
                raise M42RuntimeContractError("queued actions belong to a different task")
        if not isinstance(batch, dict):
            raise TypeError("policy batch must be a dict")
        if not self._queue:
            self._query_policy(batch)
        action = self._queue.popleft()
        self._executed_action_count += 1
        return action

    def metrics(self) -> ExecutionHorizonMetrics:
        return ExecutionHorizonMetrics(
            config_fingerprint=self.config_fingerprint,
            actions_per_query=self.config.actions_per_query,
            policy_query_count=self._policy_query_count,
            executed_action_count=self._executed_action_count,
            pending_action_count=len(self._queue),
            discarded_on_last_reset=self._discarded_on_last_reset,
            maximum_queue_depth=self._maximum_queue_depth,
            predicted_chunk_nan_count=self._predicted_chunk_nan_count,
            predicted_chunk_inf_count=self._predicted_chunk_inf_count,
            malformed_chunk_count=self._malformed_chunk_count,
            active_task_id=self._active_task_id,
        )


@dataclass(frozen=True, slots=True, eq=False)
class BinaryGripperChangeRecord:
    """One explicit raw-to-binary gripper transformation audit."""

    rollout_step: int
    raw_action: ActionArray
    binary_action: ActionArray
    raw_bound_audit: ActionProjectionRecord
    changed_command_count: int
    maximum_absolute_gripper_change: float
    processor_fingerprint: str
    version: str = M42_GRIPPER_PROCESSOR_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.rollout_step, bool) or self.rollout_step < 1:
            raise ValueError("rollout_step must be a positive integer")
        if self.changed_command_count < 0:
            raise ValueError("changed_command_count must be non-negative")
        if (
            not math.isfinite(self.maximum_absolute_gripper_change)
            or self.maximum_absolute_gripper_change < 0
        ):
            raise ValueError("maximum_absolute_gripper_change must be finite and non-negative")
        if self.version != M42_GRIPPER_PROCESSOR_VERSION:
            raise ValueError("unknown binary-gripper processor version")

    @property
    def changed(self) -> bool:
        return self.changed_command_count > 0

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "rollout_step": self.rollout_step,
            "raw_action": _action_to_json(self.raw_action),
            "binary_action": _action_to_json(self.binary_action),
            "raw_bound_audit": self.raw_bound_audit.to_dict(),
            "changed_command_count": self.changed_command_count,
            "changed": self.changed,
            "maximum_absolute_gripper_change": self.maximum_absolute_gripper_change,
            "processor_fingerprint": self.processor_fingerprint,
        }

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BinaryGripperChangeRecord) and self.to_dict() == other.to_dict()


@dataclass(frozen=True, slots=True)
class M42ActionRuntimeResult:
    """Raw, optional binary, and final project-bound records for one action."""

    executed_action: ActionArray
    raw_bound_audit: ActionProjectionRecord
    runtime_projection_audit: ActionProjectionRecord
    binary_gripper_audit: BinaryGripperChangeRecord | None
    gripper_mode: GripperRuntimeMode
    runtime_fingerprint: str


class BinaryGripperEnvPostprocessorV0:
    """Map only action index seven by sign, then apply the M4.1 project boundary."""

    def __init__(self, *, low: object, high: object) -> None:
        self._project = BoundedActionEnvPostprocessorV0(
            ActionBoundConfig(mode=ActionBoundMode.PROJECT),
            low=low,
            high=high,
        )
        if (
            self._project.action_dimension != M42_ACTION_COMPONENTS
            or M42_GRIPPER_ACTION_INDEX != M42_ACTION_COMPONENTS - 1
        ):
            raise M42RuntimeContractError(
                "BinaryGripperEnvPostprocessorV0 requires the exact M1 float[8] action schema"
            )
        self._records: list[BinaryGripperChangeRecord] = []

    @classmethod
    def from_environment(cls, env: object) -> Self:
        processor = BoundedActionEnvPostprocessorV0.from_environment(
            env,
            ActionBoundConfig(mode=ActionBoundMode.PROJECT),
            expected_action_components=M42_ACTION_COMPONENTS,
        )
        contract = processor.action_space_contract()
        return cls(low=contract["lower_bounds"], high=contract["upper_bounds"])

    def configuration(self) -> dict[str, object]:
        return {
            "version": M42_GRIPPER_PROCESSOR_VERSION,
            "schema_version": M42_RUNTIME_SCHEMA_VERSION,
            "gripper_mode": GripperRuntimeMode.BINARY.value,
            "action_components": M42_ACTION_COMPONENTS,
            "gripper_action_index": M42_GRIPPER_ACTION_INDEX,
            "threshold": 0.0,
            "nonnegative_output": 1.0,
            "negative_output": -1.0,
            "downstream_action_bound_processor": {
                "name": "BoundedActionEnvPostprocessorV0",
                "mode": ActionBoundMode.PROJECT.value,
                "contract": self._project.action_space_contract(),
            },
        }

    @property
    def runtime_fingerprint(self) -> str:
        return _prefixed_fingerprint(self.configuration())

    @property
    def audit_records(self) -> tuple[BinaryGripperChangeRecord, ...]:
        """Return immutable per-action audits accumulated since the latest reset."""
        return tuple(self._records)

    @property
    def last_audit_record(self) -> BinaryGripperChangeRecord | None:
        return self._records[-1] if self._records else None

    def reset(self) -> None:
        """Clear episode-local binary-change audit records."""
        self._records.clear()

    @classmethod
    def from_configuration(
        cls,
        value: object,
        *,
        low: object,
        high: object,
    ) -> Self:
        if not isinstance(value, Mapping):
            raise TypeError("binary-gripper configuration must be a mapping")
        result = cls(low=low, high=high)
        if dict(value) != result.configuration():
            raise M42RuntimeContractError("binary-gripper configuration does not match V0")
        return result

    def transform(self, raw_action: torch.Tensor, *, rollout_step: int) -> torch.Tensor:
        """Apply only the explicit gripper sign transform before M4.1 bounds.

        This is the hook consumed by ``ActManiSkillRolloutAdapter``.  It does
        not perform an environment-bound projection; the adapter's existing
        ``BoundedActionEnvPostprocessorV0(mode=project)`` remains the one and
        only final action-space boundary.
        """
        if not isinstance(raw_action, torch.Tensor):
            raise TypeError("binary-gripper transform requires a Torch tensor")
        if tuple(raw_action.shape) != (1, M42_ACTION_COMPONENTS):
            raise M42RuntimeContractError("binary-gripper action must have shape [1,8]")
        if not raw_action.is_floating_point():
            raise M42RuntimeContractError("binary-gripper action must use a floating dtype")
        if not bool(torch.isfinite(raw_action).all()):
            raise M42RuntimeContractError("binary-gripper action contains NaN or infinity")
        binary_action = raw_action.detach().clone()
        raw_bound_audit = self._project.process(raw_action, rollout_step=rollout_step).audit_record
        raw_gripper = raw_action[..., M42_GRIPPER_ACTION_INDEX]
        binary_gripper = torch.where(
            raw_gripper >= 0,
            torch.ones_like(raw_gripper),
            -torch.ones_like(raw_gripper),
        )
        binary_action[..., M42_GRIPPER_ACTION_INDEX] = binary_gripper
        difference = torch.abs(binary_gripper - raw_gripper)
        record = BinaryGripperChangeRecord(
            rollout_step=rollout_step,
            raw_action=_copy_action(raw_action),
            binary_action=_copy_action(binary_action),
            raw_bound_audit=raw_bound_audit,
            changed_command_count=int(torch.count_nonzero(difference).detach().cpu()),
            maximum_absolute_gripper_change=float(difference.max().detach().cpu()),
            processor_fingerprint=self.runtime_fingerprint,
        )
        self._records.append(record)
        return binary_action

    def __call__(self, raw_action: torch.Tensor, *, rollout_step: int) -> torch.Tensor:
        """Make the resettable processor itself usable as the rollout hook."""
        return self.transform(raw_action, rollout_step=rollout_step)

    def process(self, value: object, *, rollout_step: int) -> M42ActionRuntimeResult:
        # The first pass is audit-only: it validates shape/dtype/finiteness and
        # records raw bound violations without changing the binary transform's input.
        raw = self._project.process(value, rollout_step=rollout_step)
        raw_action = raw.audit_record.raw_action
        if isinstance(raw_action, torch.Tensor):
            binary_action: ActionArray = self.transform(raw_action, rollout_step=rollout_step)
            binary_record = cast(BinaryGripperChangeRecord, self.last_audit_record)
        else:
            binary_action = np.array(raw_action, copy=True)
            raw_gripper = raw_action[..., M42_GRIPPER_ACTION_INDEX]
            binary_gripper = np.where(raw_gripper >= 0, 1.0, -1.0).astype(
                raw_action.dtype,
                copy=False,
            )
            binary_action[..., M42_GRIPPER_ACTION_INDEX] = binary_gripper
            difference = np.abs(binary_gripper - raw_gripper)
            changed_count = int(np.count_nonzero(difference))
            maximum_change = float(np.max(difference))
            binary_record = BinaryGripperChangeRecord(
                rollout_step=rollout_step,
                raw_action=_copy_action(raw_action),
                binary_action=_copy_action(binary_action),
                raw_bound_audit=raw.audit_record,
                changed_command_count=changed_count,
                maximum_absolute_gripper_change=maximum_change,
                processor_fingerprint=self.runtime_fingerprint,
            )
            self._records.append(binary_record)
        projected = self._project.process(binary_action, rollout_step=rollout_step)
        return M42ActionRuntimeResult(
            executed_action=projected.executed_action,
            raw_bound_audit=raw.audit_record,
            runtime_projection_audit=projected.audit_record,
            binary_gripper_audit=binary_record,
            gripper_mode=GripperRuntimeMode.BINARY,
            runtime_fingerprint=self.runtime_fingerprint,
        )


class M42ActionRuntimeProcessorV0:
    """Explicit project/binary runtime selector without redefining bound modes."""

    def __init__(self, mode: GripperRuntimeMode, *, low: object, high: object) -> None:
        if not isinstance(mode, GripperRuntimeMode):
            raise TypeError("mode must be a GripperRuntimeMode")
        self.mode = mode
        self._project = BoundedActionEnvPostprocessorV0(
            ActionBoundConfig(mode=ActionBoundMode.PROJECT), low=low, high=high
        )
        if self._project.action_dimension != M42_ACTION_COMPONENTS:
            raise M42RuntimeContractError("M4.2 requires the exact M1 float[8] action schema")
        self._binary = (
            BinaryGripperEnvPostprocessorV0(low=low, high=high)
            if mode is GripperRuntimeMode.BINARY
            else None
        )

    @classmethod
    def from_environment(cls, env: object, mode: GripperRuntimeMode) -> Self:
        processor = BoundedActionEnvPostprocessorV0.from_environment(
            env,
            ActionBoundConfig(mode=ActionBoundMode.PROJECT),
            expected_action_components=M42_ACTION_COMPONENTS,
        )
        contract = processor.action_space_contract()
        return cls(mode, low=contract["lower_bounds"], high=contract["upper_bounds"])

    def configuration(self) -> dict[str, object]:
        return {
            "schema_version": M42_RUNTIME_SCHEMA_VERSION,
            "gripper_mode": self.mode.value,
            "project_processor": {
                "name": "BoundedActionEnvPostprocessorV0",
                "mode": ActionBoundMode.PROJECT.value,
                "contract": self._project.action_space_contract(),
            },
            "binary_processor": (None if self._binary is None else self._binary.configuration()),
        }

    @property
    def runtime_fingerprint(self) -> str:
        return _prefixed_fingerprint(self.configuration())

    def process(self, value: object, *, rollout_step: int) -> M42ActionRuntimeResult:
        if self._binary is not None:
            result = self._binary.process(value, rollout_step=rollout_step)
            return M42ActionRuntimeResult(
                executed_action=result.executed_action,
                raw_bound_audit=result.raw_bound_audit,
                runtime_projection_audit=result.runtime_projection_audit,
                binary_gripper_audit=result.binary_gripper_audit,
                gripper_mode=self.mode,
                runtime_fingerprint=self.runtime_fingerprint,
            )
        projected: ActionBoundResult = self._project.process(value, rollout_step=rollout_step)
        return M42ActionRuntimeResult(
            executed_action=projected.executed_action,
            raw_bound_audit=projected.audit_record,
            runtime_projection_audit=projected.audit_record,
            binary_gripper_audit=None,
            gripper_mode=self.mode,
            runtime_fingerprint=self.runtime_fingerprint,
        )


@dataclass(frozen=True, slots=True)
class M42ActionRuntimeSummary:
    """Separate raw, binary-change, arm-project, and gripper-project metrics."""

    runtime_fingerprint: str
    gripper_mode: GripperRuntimeMode
    total_action_count: int
    raw_out_of_bounds_action_count: int
    raw_out_of_bounds_component_count: int
    binary_changed_action_count: int
    binary_changed_command_count: int
    runtime_projected_action_count: int
    runtime_projected_component_count: int
    arm_projected_component_count: int
    gripper_projected_component_count: int
    maximum_raw_bound_excess: float
    maximum_binary_gripper_change: float
    maximum_runtime_projection_correction: float
    version: str = M42_ACTION_RUNTIME_SUMMARY_VERSION

    @classmethod
    def from_results(cls, results: Sequence[M42ActionRuntimeResult]) -> Self:
        if not results:
            raise ValueError("runtime action summary requires at least one result")
        runtime_fingerprints = {item.runtime_fingerprint for item in results}
        gripper_modes = {item.gripper_mode for item in results}
        if len(runtime_fingerprints) != 1 or len(gripper_modes) != 1:
            raise ValueError("runtime action summary cannot mix runtime identities")
        raw_records = tuple(item.raw_bound_audit for item in results)
        runtime_records = tuple(item.runtime_projection_audit for item in results)
        binary_records = tuple(
            cast(BinaryGripperChangeRecord, item.binary_gripper_audit)
            for item in results
            if item.binary_gripper_audit is not None
        )
        return cls(
            runtime_fingerprint=next(iter(runtime_fingerprints)),
            gripper_mode=next(iter(gripper_modes)),
            total_action_count=len(results),
            raw_out_of_bounds_action_count=sum(item.was_projected for item in raw_records),
            raw_out_of_bounds_component_count=sum(
                item.projected_component_count for item in raw_records
            ),
            binary_changed_action_count=sum(item.changed for item in binary_records),
            binary_changed_command_count=sum(item.changed_command_count for item in binary_records),
            runtime_projected_action_count=sum(item.was_projected for item in runtime_records),
            runtime_projected_component_count=sum(
                item.projected_component_count for item in runtime_records
            ),
            arm_projected_component_count=sum(
                _count_true(item.violation_mask)
                - _component_true_count(item.violation_mask, M42_GRIPPER_ACTION_INDEX)
                for item in runtime_records
            ),
            gripper_projected_component_count=sum(
                _component_true_count(item.violation_mask, M42_GRIPPER_ACTION_INDEX)
                for item in runtime_records
            ),
            maximum_raw_bound_excess=max(
                item.maximum_absolute_bound_excess for item in raw_records
            ),
            maximum_binary_gripper_change=max(
                (item.maximum_absolute_gripper_change for item in binary_records), default=0.0
            ),
            maximum_runtime_projection_correction=max(
                item.linf_correction_magnitude for item in runtime_records
            ),
        )

    @property
    def raw_action_bounds_validated(self) -> bool:
        return self.raw_out_of_bounds_component_count == 0

    @property
    def runtime_action_bounds_validated(self) -> bool:
        return True

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "runtime_fingerprint": self.runtime_fingerprint,
            "gripper_mode": self.gripper_mode.value,
            "total_action_count": self.total_action_count,
            "raw_out_of_bounds_action_count": self.raw_out_of_bounds_action_count,
            "raw_out_of_bounds_component_count": self.raw_out_of_bounds_component_count,
            "raw_action_bounds_validated": self.raw_action_bounds_validated,
            "binary_changed_action_count": self.binary_changed_action_count,
            "binary_changed_command_count": self.binary_changed_command_count,
            "runtime_projected_action_count": self.runtime_projected_action_count,
            "runtime_projected_component_count": self.runtime_projected_component_count,
            "arm_projected_component_count": self.arm_projected_component_count,
            "gripper_projected_component_count": self.gripper_projected_component_count,
            "runtime_action_bounds_validated": self.runtime_action_bounds_validated,
            "maximum_raw_bound_excess": self.maximum_raw_bound_excess,
            "maximum_binary_gripper_change": self.maximum_binary_gripper_change,
            "maximum_runtime_projection_correction": (self.maximum_runtime_projection_correction),
        }

    @property
    def fingerprint(self) -> str:
        return _prefixed_fingerprint(self.to_dict())


__all__ = [
    "ActionChunkPolicy",
    "BinaryGripperChangeRecord",
    "BinaryGripperEnvPostprocessorV0",
    "ExecutionHorizonMetrics",
    "ExecutionHorizonPolicyV0",
    "M42ActionRuntimeProcessorV0",
    "M42ActionRuntimeResult",
    "M42ActionRuntimeSummary",
    "M42RuntimeContractError",
]
