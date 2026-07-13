"""Explicit environment-action bound handling for M4.1 ACT evaluation.

The installed LeRobot policy postprocessor converts ACT predictions back to
environment action semantics.  This module owns the separate, project-level
boundary between those raw semantic actions and ``env.step``.  Reject and
projection are explicit runtime identities; projection is never an implicit
side effect of the policy processor.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Self, cast

import numpy as np
import torch

from langmani.datasets.identity import sha256_hex

ACTION_BOUND_CONFIG_SCHEMA_VERSION = "langmani-m4.1-action-bound-config-v0"
ACTION_PROJECTION_RECORD_SCHEMA_VERSION = "langmani-m4.1-action-projection-record-v0"
ACTION_PROJECTION_SUMMARY_SCHEMA_VERSION = "langmani-m4.1-action-projection-summary-v0"
EVALUATION_RUNTIME_SCHEMA_VERSION = "langmani-m4.1-evaluation-runtime-v0"
EVALUATION_RUNTIME_MANIFEST_SCHEMA_VERSION = "langmani-m4.1-evaluation-runtime-manifest-v0"
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_RE = re.compile(r"^[0-9a-f]{40,64}$")

type ActionArray = np.ndarray | torch.Tensor


class ActionBoundMode(StrEnum):
    """Exactly the two auditable M4.1 environment-action behaviors."""

    REJECT = "reject"
    PROJECT = "project"


class ActionBoundErrorKind(StrEnum):
    MALFORMED_ACTION = "malformed_action"
    NONFINITE_ACTION = "nonfinite_action"
    BOUND_VIOLATION = "bound_violation"
    UNSUPPORTED_BOUNDS = "unsupported_bounds"


def _digest(value: str, name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a prefixed lowercase SHA-256 digest")


def _finite_nonnegative(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def _json_array(value: ActionArray | None) -> object:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        raw = value.detach().cpu().tolist()
    else:
        raw = np.asarray(value).tolist()

    def safe_number(item: object) -> object:
        if isinstance(item, list):
            return [safe_number(child) for child in item]
        if isinstance(item, float) and not math.isfinite(item):
            if math.isnan(item):
                return "nan"
            return "inf" if item > 0 else "-inf"
        return item

    return safe_number(raw)


def _copy_snapshot(value: ActionArray) -> ActionArray:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class ActionBoundConfig:
    """Versioned policy for the final environment-action boundary."""

    mode: ActionBoundMode = ActionBoundMode.REJECT
    bounds_from_environment_action_space: bool = True
    comparison_tolerance: float = 1e-6
    retain_raw_and_executed_actions: bool = True
    retain_per_step_projection_records: bool = True
    nonfinite_action_policy: str = "hard_fail"
    malformed_action_policy: str = "hard_fail"
    schema_version: str = ACTION_BOUND_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", ActionBoundMode(self.mode))
        if self.schema_version != ACTION_BOUND_CONFIG_SCHEMA_VERSION:
            raise ValueError("unknown action-bound configuration schema")
        if self.bounds_from_environment_action_space is not True:
            raise ValueError("M4.1 bounds must come from the active environment action space")
        _finite_nonnegative(self.comparison_tolerance, "comparison_tolerance")
        if self.comparison_tolerance > 1e-3:
            raise ValueError("comparison_tolerance is too large for action reload auditing")
        if self.retain_raw_and_executed_actions is not True:
            raise ValueError("M4.1 requires separate raw and executed action audit values")
        if self.retain_per_step_projection_records is not True:
            raise ValueError("M4.1 requires per-step projection audit records")
        if self.nonfinite_action_policy != "hard_fail":
            raise ValueError("nonfinite actions must use the explicit hard_fail policy")
        if self.malformed_action_policy != "hard_fail":
            raise ValueError("malformed actions must use the explicit hard_fail policy")

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "bounds_from_environment_action_space": self.bounds_from_environment_action_space,
            "comparison_tolerance": self.comparison_tolerance,
            "retain_raw_and_executed_actions": self.retain_raw_and_executed_actions,
            "retain_per_step_projection_records": self.retain_per_step_projection_records,
            "nonfinite_action_policy": self.nonfinite_action_policy,
            "malformed_action_policy": self.malformed_action_policy,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if not isinstance(value, Mapping):
            raise TypeError("action-bound configuration must be a mapping")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True, eq=False)
class ActionProjectionRecord:
    """One raw-versus-executed action audit record.

    Array snapshots stay as tensors/arrays until ``to_dict`` is called after
    the environment step.  The execution hot path therefore never converts an
    action into a Python list before validation and projection complete.
    """

    rollout_step: int
    raw_action: ActionArray
    executed_action: ActionArray | None
    lower_bounds: ActionArray
    upper_bounds: ActionArray
    violation_mask: ActionArray
    projected_component_count: int
    was_projected: bool
    was_rejected: bool
    maximum_lower_excess: float
    maximum_upper_excess: float
    maximum_absolute_bound_excess: float
    l1_correction_magnitude: float
    l2_correction_magnitude: float
    linf_correction_magnitude: float
    action_dtype: str
    action_shape: tuple[int, ...]
    processing_mode: ActionBoundMode
    schema_version: str = ACTION_PROJECTION_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.rollout_step, bool) or not isinstance(self.rollout_step, int):
            raise TypeError("rollout_step must be an integer")
        if self.rollout_step < 1:
            raise ValueError("rollout_step must be positive")
        object.__setattr__(self, "processing_mode", ActionBoundMode(self.processing_mode))
        if self.schema_version != ACTION_PROJECTION_RECORD_SCHEMA_VERSION:
            raise ValueError("unknown action-projection record schema")
        if not self.action_shape or any(dimension < 1 for dimension in self.action_shape):
            raise ValueError("action_shape must contain positive dimensions")
        if self.projected_component_count < 0:
            raise ValueError("projected_component_count must be non-negative")
        for name in (
            "maximum_lower_excess",
            "maximum_upper_excess",
            "maximum_absolute_bound_excess",
            "l1_correction_magnitude",
            "l2_correction_magnitude",
            "linf_correction_magnitude",
        ):
            _finite_nonnegative(float(getattr(self, name)), name)
        if self.was_projected != (self.projected_component_count > 0):
            raise ValueError("was_projected must agree with projected_component_count")
        if self.was_projected and self.was_rejected:
            raise ValueError("a rejected action cannot also be projected for execution")
        if self.was_rejected != (self.executed_action is None):
            raise ValueError("rejected actions must have no executed action")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "rollout_step": self.rollout_step,
            "raw_action": _json_array(self.raw_action),
            "executed_action": _json_array(self.executed_action),
            "lower_bounds": _json_array(self.lower_bounds),
            "upper_bounds": _json_array(self.upper_bounds),
            "violation_mask": _json_array(self.violation_mask),
            "projected_component_count": self.projected_component_count,
            "was_projected": self.was_projected,
            "was_rejected": self.was_rejected,
            "maximum_lower_excess": self.maximum_lower_excess,
            "maximum_upper_excess": self.maximum_upper_excess,
            "maximum_absolute_bound_excess": self.maximum_absolute_bound_excess,
            "l1_correction_magnitude": self.l1_correction_magnitude,
            "l2_correction_magnitude": self.l2_correction_magnitude,
            "linf_correction_magnitude": self.linf_correction_magnitude,
            "action_dtype": self.action_dtype,
            "action_shape": list(self.action_shape),
            "processing_mode": self.processing_mode.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if not isinstance(value, Mapping):
            raise TypeError("action-projection record must be a mapping")
        payload = dict(value)
        for name, dtype in (
            ("raw_action", np.float64),
            ("lower_bounds", np.float64),
            ("upper_bounds", np.float64),
            ("violation_mask", np.bool_),
        ):
            payload[name] = np.asarray(payload[name], dtype=dtype)
        executed = payload.get("executed_action")
        payload["executed_action"] = (
            None if executed is None else np.asarray(executed, dtype=np.float64)
        )
        payload["action_shape"] = tuple(payload["action_shape"])
        return cls(**payload)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ActionProjectionRecord) and self.to_dict() == other.to_dict()


@dataclass(frozen=True, slots=True)
class ActionProjectionSummary:
    total_policy_actions: int
    projected_action_count: int
    projected_action_rate: float
    projected_component_count: int
    maximum_bound_excess: float
    mean_bound_excess_among_projected_actions: float
    maximum_linf_correction: float
    mean_l1_correction: float
    per_action_dimension_projection_counts: tuple[int, ...]
    first_projected_rollout_step: int | None
    any_nonfinite_action: bool
    any_malformed_action: bool
    schema_version: str = ACTION_PROJECTION_SUMMARY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ACTION_PROJECTION_SUMMARY_SCHEMA_VERSION:
            raise ValueError("unknown action-projection summary schema")
        if self.total_policy_actions < 0 or self.projected_action_count < 0:
            raise ValueError("projection action counts must be non-negative")
        if self.projected_action_count > self.total_policy_actions:
            raise ValueError("projected actions cannot exceed total policy actions")
        if self.projected_component_count < 0 or any(
            value < 0 for value in self.per_action_dimension_projection_counts
        ):
            raise ValueError("projection component counts must be non-negative")
        expected_rate = (
            self.projected_action_count / self.total_policy_actions
            if self.total_policy_actions
            else 0.0
        )
        if not math.isclose(self.projected_action_rate, expected_rate, rel_tol=0, abs_tol=1e-12):
            raise ValueError("projected_action_rate disagrees with action counts")
        for name in (
            "projected_action_rate",
            "maximum_bound_excess",
            "mean_bound_excess_among_projected_actions",
            "maximum_linf_correction",
            "mean_l1_correction",
        ):
            _finite_nonnegative(float(getattr(self, name)), name)
        if self.first_projected_rollout_step is not None and self.first_projected_rollout_step < 1:
            raise ValueError("first projected rollout step must be positive")
        if (self.projected_action_count == 0) != (self.first_projected_rollout_step is None):
            raise ValueError("first projected step must agree with projected action count")

    @classmethod
    def from_records(
        cls,
        records: Sequence[ActionProjectionRecord],
        *,
        action_dimension: int,
        any_nonfinite_action: bool = False,
        any_malformed_action: bool = False,
        total_policy_actions: int | None = None,
    ) -> Self:
        if action_dimension < 1:
            raise ValueError("action_dimension must be positive")
        total = len(records) if total_policy_actions is None else total_policy_actions
        if total < len(records):
            raise ValueError("total_policy_actions cannot be less than retained records")
        projected = tuple(record for record in records if record.was_projected)
        dimension_counts = np.zeros(action_dimension, dtype=np.int64)
        for record in projected:
            mask = np.asarray(_json_array(record.violation_mask), dtype=np.bool_)
            if mask.shape[-1] != action_dimension:
                raise ValueError("projection record action dimension is inconsistent")
            dimension_counts += mask.reshape(-1, action_dimension).sum(axis=0)
        projected_excesses = [record.maximum_absolute_bound_excess for record in projected]
        return cls(
            total_policy_actions=total,
            projected_action_count=len(projected),
            projected_action_rate=len(projected) / total if total else 0.0,
            projected_component_count=sum(record.projected_component_count for record in records),
            maximum_bound_excess=max(
                (record.maximum_absolute_bound_excess for record in records), default=0.0
            ),
            mean_bound_excess_among_projected_actions=(
                float(np.mean(projected_excesses)) if projected_excesses else 0.0
            ),
            maximum_linf_correction=max(
                (record.linf_correction_magnitude for record in records), default=0.0
            ),
            mean_l1_correction=(
                float(np.mean([record.l1_correction_magnitude for record in records]))
                if records
                else 0.0
            ),
            per_action_dimension_projection_counts=tuple(int(value) for value in dimension_counts),
            first_projected_rollout_step=(
                min(record.rollout_step for record in projected) if projected else None
            ),
            any_nonfinite_action=any_nonfinite_action,
            any_malformed_action=any_malformed_action,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "total_policy_actions": self.total_policy_actions,
            "projected_action_count": self.projected_action_count,
            "projected_action_rate": self.projected_action_rate,
            "projected_component_count": self.projected_component_count,
            "maximum_bound_excess": self.maximum_bound_excess,
            "mean_bound_excess_among_projected_actions": (
                self.mean_bound_excess_among_projected_actions
            ),
            "maximum_linf_correction": self.maximum_linf_correction,
            "mean_l1_correction": self.mean_l1_correction,
            "per_action_dimension_projection_counts": list(
                self.per_action_dimension_projection_counts
            ),
            "first_projected_rollout_step": self.first_projected_rollout_step,
            "any_nonfinite_action": self.any_nonfinite_action,
            "any_malformed_action": self.any_malformed_action,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if not isinstance(value, Mapping):
            raise TypeError("action-projection summary must be a mapping")
        payload = dict(value)
        payload["per_action_dimension_projection_counts"] = tuple(
            payload["per_action_dimension_projection_counts"]
        )
        return cls(**payload)


class ActionBoundProcessingError(RuntimeError):
    """Structured hard failure before an unsafe environment step."""

    def __init__(
        self,
        kind: ActionBoundErrorKind,
        message: str,
        *,
        record: ActionProjectionRecord | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = ActionBoundErrorKind(kind)
        self.record = record


@dataclass(frozen=True, slots=True)
class ActionBoundResult:
    executed_action: ActionArray
    audit_record: ActionProjectionRecord


def resolve_environment_action_bounds(
    env: object,
    *,
    expected_action_components: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve finite single-action bounds through Gymnasium/ManiSkill wrappers."""

    base = getattr(env, "unwrapped", env)
    action_space = None
    getter = getattr(env, "get_wrapper_attr", None)
    if callable(getter):
        try:
            action_space = getter("single_action_space")
        except (AttributeError, LookupError):
            action_space = None
    for owner, name in (
        (env, "single_action_space"),
        (base, "single_action_space"),
        (env, "action_space"),
        (base, "action_space"),
    ):
        if action_space is None:
            action_space = getattr(owner, name, None)
    if (
        action_space is None
        or not hasattr(action_space, "low")
        or not hasattr(action_space, "high")
    ):
        raise ActionBoundProcessingError(
            ActionBoundErrorKind.UNSUPPORTED_BOUNDS,
            "environment must expose an action space with explicit low/high bounds",
        )
    low = np.asarray(action_space.low)
    high = np.asarray(action_space.high)
    if low.ndim == 2 and low.shape[0] == 1:
        low = low[0]
    if high.ndim == 2 and high.shape[0] == 1:
        high = high[0]
    if expected_action_components is not None and low.shape != (expected_action_components,):
        raise ActionBoundProcessingError(
            ActionBoundErrorKind.UNSUPPORTED_BOUNDS,
            f"environment single-action bounds must have shape ({expected_action_components},)",
        )
    return _validated_bounds(low, high)


def _validated_bounds(low: object, high: object) -> tuple[np.ndarray, np.ndarray]:
    low_array = np.asarray(low)
    high_array = np.asarray(high)
    if (
        low_array.shape != high_array.shape
        or not low_array.shape
        or low_array.ndim != 1
        or not np.issubdtype(low_array.dtype, np.floating)
        or not np.issubdtype(high_array.dtype, np.floating)
    ):
        raise ActionBoundProcessingError(
            ActionBoundErrorKind.UNSUPPORTED_BOUNDS,
            "environment low/high bounds must be same-shape one-dimensional floating arrays",
        )
    if not np.all(np.isfinite(low_array)) or not np.all(np.isfinite(high_array)):
        raise ActionBoundProcessingError(
            ActionBoundErrorKind.UNSUPPORTED_BOUNDS,
            "environment action bounds must be finite",
        )
    if np.any(low_array > high_array):
        raise ActionBoundProcessingError(
            ActionBoundErrorKind.UNSUPPORTED_BOUNDS,
            "environment action lower bounds must not exceed upper bounds",
        )
    low_result = np.array(low_array, copy=True)
    high_result = np.array(high_array, copy=True)
    low_result.setflags(write=False)
    high_result.setflags(write=False)
    return low_result, high_result


class BoundedActionEnvPostprocessorV0:
    """Validate and explicitly reject or project raw environment actions."""

    def __init__(self, config: ActionBoundConfig, *, low: object, high: object) -> None:
        if not isinstance(config, ActionBoundConfig):
            raise TypeError("config must be an ActionBoundConfig")
        self.config = config
        self._low, self._high = _validated_bounds(low, high)

    @classmethod
    def from_environment(
        cls,
        env: object,
        config: ActionBoundConfig,
        *,
        expected_action_components: int | None = None,
    ) -> Self:
        low, high = resolve_environment_action_bounds(
            env,
            expected_action_components=expected_action_components,
        )
        return cls(config, low=low, high=high)

    @property
    def action_dimension(self) -> int:
        return int(self._low.shape[0])

    def action_space_contract(self) -> dict[str, object]:
        return {
            "bounds_source": "environment_action_space",
            "single_action_shape": list(self._low.shape),
            "lower_dtype": str(self._low.dtype),
            "upper_dtype": str(self._high.dtype),
            "lower_bounds": self._low.tolist(),
            "upper_bounds": self._high.tolist(),
        }

    def process(self, value: object, *, rollout_step: int) -> ActionBoundResult:
        if isinstance(rollout_step, bool) or not isinstance(rollout_step, int) or rollout_step < 1:
            raise ValueError("rollout_step must be a positive integer")
        if isinstance(value, torch.Tensor):
            return self._process_tensor(value, rollout_step=rollout_step)
        if isinstance(value, np.ndarray):
            return self._process_numpy(value, rollout_step=rollout_step)
        raise ActionBoundProcessingError(
            ActionBoundErrorKind.MALFORMED_ACTION,
            "environment-semantic action must be a Torch tensor or NumPy array",
        )

    def _validate_shape(self, shape: tuple[int, ...]) -> None:
        if len(shape) not in {1, 2} or shape[-1:] != self._low.shape:
            raise ActionBoundProcessingError(
                ActionBoundErrorKind.MALFORMED_ACTION,
                "action must have single or batch shape ending in the environment action shape",
            )

    def _hard_failure_record(
        self,
        action: ActionArray,
        *,
        rollout_step: int,
    ) -> ActionProjectionRecord:
        if isinstance(action, torch.Tensor):
            low: ActionArray = torch.tensor(self._low, dtype=torch.float64, device=action.device)
            high: ActionArray = torch.tensor(self._high, dtype=torch.float64, device=action.device)
            mask: ActionArray = torch.zeros_like(action, dtype=torch.bool)
        else:
            low = np.array(self._low, dtype=np.float64, copy=True)
            high = np.array(self._high, dtype=np.float64, copy=True)
            mask = np.zeros_like(action, dtype=np.bool_)
        return ActionProjectionRecord(
            rollout_step=rollout_step,
            raw_action=_copy_snapshot(action),
            executed_action=None,
            lower_bounds=_copy_snapshot(low),
            upper_bounds=_copy_snapshot(high),
            violation_mask=_copy_snapshot(mask),
            projected_component_count=0,
            was_projected=False,
            was_rejected=True,
            maximum_lower_excess=0.0,
            maximum_upper_excess=0.0,
            maximum_absolute_bound_excess=0.0,
            l1_correction_magnitude=0.0,
            l2_correction_magnitude=0.0,
            linf_correction_magnitude=0.0,
            action_dtype=str(action.dtype),
            action_shape=tuple(action.shape),
            processing_mode=self.config.mode,
        )

    def _process_tensor(self, action: torch.Tensor, *, rollout_step: int) -> ActionBoundResult:
        try:
            self._validate_shape(tuple(action.shape))
        except ActionBoundProcessingError as error:
            raise ActionBoundProcessingError(
                error.kind,
                str(error),
                record=self._hard_failure_record(action, rollout_step=rollout_step),
            ) from error
        if not action.is_floating_point():
            raise ActionBoundProcessingError(
                ActionBoundErrorKind.MALFORMED_ACTION,
                "action tensor must use a floating dtype",
                record=self._hard_failure_record(action, rollout_step=rollout_step),
            )
        if not bool(torch.isfinite(action).all()):
            raise ActionBoundProcessingError(
                ActionBoundErrorKind.NONFINITE_ACTION,
                "action tensor contains NaN or infinity",
                record=self._hard_failure_record(action, rollout_step=rollout_step),
            )
        low = torch.tensor(self._low, dtype=action.dtype, device=action.device)
        high = torch.tensor(self._high, dtype=action.dtype, device=action.device)
        lower_excess = torch.clamp(low - action, min=0)
        upper_excess = torch.clamp(action - high, min=0)
        violation = (lower_excess > 0) | (upper_excess > 0)
        violation_count = int(torch.count_nonzero(violation).detach().cpu())
        rejected = violation_count > 0 and self.config.mode is ActionBoundMode.REJECT
        executed = None
        projected_count = 0
        if not rejected:
            if violation_count:
                executed = torch.minimum(torch.maximum(action, low), high)
                projected_count = violation_count
            else:
                executed = action
            if not bool(torch.isfinite(executed).all()) or bool(
                ((executed < low) | (executed > high)).any()
            ):
                raise ActionBoundProcessingError(
                    ActionBoundErrorKind.BOUND_VIOLATION,
                    "processed tensor action is not finite and strictly within environment bounds",
                )
        correction = torch.zeros_like(action) if executed is None else executed - action
        record = self._record(
            rollout_step=rollout_step,
            raw_action=action,
            executed_action=executed,
            low=low,
            high=high,
            violation=violation,
            lower_excess=lower_excess,
            upper_excess=upper_excess,
            correction=correction,
            projected_component_count=projected_count,
            rejected=rejected,
        )
        if rejected:
            raise ActionBoundProcessingError(
                ActionBoundErrorKind.BOUND_VIOLATION,
                "raw environment-semantic action lies outside active environment bounds",
                record=record,
            )
        assert executed is not None
        return ActionBoundResult(executed_action=executed, audit_record=record)

    def _process_numpy(self, action: np.ndarray, *, rollout_step: int) -> ActionBoundResult:
        try:
            self._validate_shape(action.shape)
        except ActionBoundProcessingError as error:
            raise ActionBoundProcessingError(
                error.kind,
                str(error),
                record=self._hard_failure_record(action, rollout_step=rollout_step),
            ) from error
        if not np.issubdtype(action.dtype, np.floating):
            raise ActionBoundProcessingError(
                ActionBoundErrorKind.MALFORMED_ACTION,
                "action array must use a floating dtype",
                record=self._hard_failure_record(action, rollout_step=rollout_step),
            )
        if not np.all(np.isfinite(action)):
            raise ActionBoundProcessingError(
                ActionBoundErrorKind.NONFINITE_ACTION,
                "action array contains NaN or infinity",
                record=self._hard_failure_record(action, rollout_step=rollout_step),
            )
        low = self._low.astype(action.dtype, copy=False)
        high = self._high.astype(action.dtype, copy=False)
        lower_excess = np.maximum(low - action, 0)
        upper_excess = np.maximum(action - high, 0)
        violation = (lower_excess > 0) | (upper_excess > 0)
        violation_count = int(np.count_nonzero(violation))
        rejected = violation_count > 0 and self.config.mode is ActionBoundMode.REJECT
        executed: np.ndarray | None = None
        projected_count = 0
        if not rejected:
            if violation_count:
                executed = np.minimum(np.maximum(action, low), high)
                projected_count = violation_count
            else:
                executed = action
            if (
                not np.all(np.isfinite(executed))
                or np.any(executed < low)
                or np.any(executed > high)
            ):
                raise ActionBoundProcessingError(
                    ActionBoundErrorKind.BOUND_VIOLATION,
                    "processed array action is not finite and strictly within environment bounds",
                )
        correction = np.zeros_like(action) if executed is None else executed - action
        record = self._record(
            rollout_step=rollout_step,
            raw_action=action,
            executed_action=executed,
            low=low,
            high=high,
            violation=violation,
            lower_excess=lower_excess,
            upper_excess=upper_excess,
            correction=correction,
            projected_component_count=projected_count,
            rejected=rejected,
        )
        if rejected:
            raise ActionBoundProcessingError(
                ActionBoundErrorKind.BOUND_VIOLATION,
                "raw environment-semantic action lies outside active environment bounds",
                record=record,
            )
        assert executed is not None
        return ActionBoundResult(executed_action=executed, audit_record=record)

    def _record(
        self,
        *,
        rollout_step: int,
        raw_action: ActionArray,
        executed_action: ActionArray | None,
        low: ActionArray,
        high: ActionArray,
        violation: ActionArray,
        lower_excess: ActionArray,
        upper_excess: ActionArray,
        correction: ActionArray,
        projected_component_count: int,
        rejected: bool,
    ) -> ActionProjectionRecord:
        def scalar_max(value: ActionArray) -> float:
            if isinstance(value, torch.Tensor):
                return float(value.max().detach().cpu()) if value.numel() else 0.0
            return float(np.max(value)) if value.size else 0.0

        def norm(value: ActionArray, order: float | int) -> float:
            if isinstance(value, torch.Tensor):
                return float(torch.linalg.vector_norm(value.reshape(-1), ord=order).detach().cpu())
            return float(np.linalg.norm(value.reshape(-1), ord=order))

        maximum_lower = scalar_max(lower_excess)
        maximum_upper = scalar_max(upper_excess)
        return ActionProjectionRecord(
            rollout_step=rollout_step,
            raw_action=_copy_snapshot(raw_action),
            executed_action=(None if executed_action is None else _copy_snapshot(executed_action)),
            lower_bounds=_copy_snapshot(low),
            upper_bounds=_copy_snapshot(high),
            violation_mask=_copy_snapshot(violation),
            projected_component_count=projected_component_count,
            was_projected=projected_component_count > 0,
            was_rejected=rejected,
            maximum_lower_excess=maximum_lower,
            maximum_upper_excess=maximum_upper,
            maximum_absolute_bound_excess=max(maximum_lower, maximum_upper),
            l1_correction_magnitude=norm(correction, 1),
            l2_correction_magnitude=norm(correction, 2),
            linf_correction_magnitude=norm(correction, float("inf")),
            action_dtype=str(raw_action.dtype),
            action_shape=tuple(raw_action.shape),
            processing_mode=self.config.mode,
        )


class _ImmutableJsonMapping(Mapping[str, object]):
    """Small recursively immutable mapping for runtime identity semantics."""

    __slots__ = ("_items",)

    def __init__(self, value: Mapping[str, object]) -> None:
        self._items = tuple((key, _freeze_json(item)) for key, item in sorted(value.items()))

    def __getitem__(self, key: str) -> object:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and _thaw_json(self) == _thaw_json(other)

    def __deepcopy__(self, memo: object) -> _ImmutableJsonMapping:
        del memo
        return self


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("runtime identity mappings require string keys")
        return _ImmutableJsonMapping(cast(Mapping[str, object], value))
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"unsupported runtime identity value {type(value).__name__}")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_mapping(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    try:
        canonical = cast(
            dict[str, object],
            __import__("json").loads(__import__("json").dumps(_thaw_json(value))),
        )
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be JSON-serializable") from error
    return _ImmutableJsonMapping(canonical)


@dataclass(frozen=True, slots=True)
class EvaluationRuntimeIdentity:
    checkpoint_fingerprint: str
    policy_preprocessor_fingerprint: str
    policy_postprocessor_fingerprint: str
    action_bound_config: ActionBoundConfig
    environment_id: str
    action_space_contract: Mapping[str, object]
    task_conditioning_mapping_version: str
    rollout_config: Mapping[str, object]
    code_git_commit: str
    schema_version: str = EVALUATION_RUNTIME_SCHEMA_VERSION
    runtime_fingerprint: str = ""

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_fingerprint",
            "policy_preprocessor_fingerprint",
            "policy_postprocessor_fingerprint",
        ):
            _digest(cast(str, getattr(self, name)), name)
        if not isinstance(self.action_bound_config, ActionBoundConfig):
            raise TypeError("action_bound_config must be an ActionBoundConfig")
        if not self.environment_id or not self.task_conditioning_mapping_version:
            raise ValueError("runtime environment and task-conditioning IDs must be non-empty")
        if _GIT_RE.fullmatch(self.code_git_commit) is None:
            raise ValueError("code_git_commit must be a full lowercase Git object ID")
        if self.schema_version != EVALUATION_RUNTIME_SCHEMA_VERSION:
            raise ValueError("unknown evaluation-runtime identity schema")
        object.__setattr__(
            self,
            "action_space_contract",
            _canonical_mapping(self.action_space_contract, "action_space_contract"),
        )
        object.__setattr__(
            self,
            "rollout_config",
            _canonical_mapping(self.rollout_config, "rollout_config"),
        )
        expected = self.compute_fingerprint()
        if not self.runtime_fingerprint:
            object.__setattr__(self, "runtime_fingerprint", expected)
        elif self.runtime_fingerprint != expected:
            raise ValueError("runtime_fingerprint does not match evaluation runtime semantics")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "policy_preprocessor_fingerprint": self.policy_preprocessor_fingerprint,
            "policy_postprocessor_fingerprint": self.policy_postprocessor_fingerprint,
            "action_bound_config": self.action_bound_config.to_dict(),
            "environment_id": self.environment_id,
            "action_space_contract": _thaw_json(self.action_space_contract),
            "task_conditioning_mapping_version": self.task_conditioning_mapping_version,
            "rollout_config": _thaw_json(self.rollout_config),
            "code_git_commit": self.code_git_commit,
        }

    def compute_fingerprint(self) -> str:
        return f"sha256:{sha256_hex(self.identity_payload())}"

    def to_dict(self) -> dict[str, object]:
        return {**self.identity_payload(), "runtime_fingerprint": self.runtime_fingerprint}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if not isinstance(value, Mapping):
            raise TypeError("evaluation-runtime identity must be a mapping")
        payload = dict(value)
        payload["action_bound_config"] = ActionBoundConfig.from_dict(payload["action_bound_config"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class EvaluationRuntimeManifest:
    identity: EvaluationRuntimeIdentity
    checkpoint_model_reload_validated: bool
    policy_processor_reload_validated: bool
    action_bound_processor_reload_validated: bool
    deterministic_raw_action_matched: bool
    raw_action_match_tolerance: float
    evaluation_output_path: str | None = None
    schema_version: str = EVALUATION_RUNTIME_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.identity, EvaluationRuntimeIdentity):
            raise TypeError("identity must be an EvaluationRuntimeIdentity")
        if self.schema_version != EVALUATION_RUNTIME_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unknown evaluation-runtime manifest schema")
        _finite_nonnegative(self.raw_action_match_tolerance, "raw_action_match_tolerance")
        for name in (
            "checkpoint_model_reload_validated",
            "policy_processor_reload_validated",
            "action_bound_processor_reload_validated",
            "deterministic_raw_action_matched",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "identity": self.identity.to_dict(),
            "checkpoint_model_reload_validated": self.checkpoint_model_reload_validated,
            "policy_processor_reload_validated": self.policy_processor_reload_validated,
            "action_bound_processor_reload_validated": (
                self.action_bound_processor_reload_validated
            ),
            "deterministic_raw_action_matched": self.deterministic_raw_action_matched,
            "raw_action_match_tolerance": self.raw_action_match_tolerance,
            "evaluation_output_path": self.evaluation_output_path,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if not isinstance(value, Mapping):
            raise TypeError("evaluation-runtime manifest must be a mapping")
        payload = dict(value)
        payload["identity"] = EvaluationRuntimeIdentity.from_dict(payload["identity"])
        return cls(**payload)


__all__ = [
    "ACTION_BOUND_CONFIG_SCHEMA_VERSION",
    "ACTION_PROJECTION_RECORD_SCHEMA_VERSION",
    "ACTION_PROJECTION_SUMMARY_SCHEMA_VERSION",
    "EVALUATION_RUNTIME_MANIFEST_SCHEMA_VERSION",
    "EVALUATION_RUNTIME_SCHEMA_VERSION",
    "ActionBoundConfig",
    "ActionBoundErrorKind",
    "ActionBoundMode",
    "ActionBoundProcessingError",
    "ActionBoundResult",
    "ActionProjectionRecord",
    "ActionProjectionSummary",
    "BoundedActionEnvPostprocessorV0",
    "EvaluationRuntimeIdentity",
    "EvaluationRuntimeManifest",
    "resolve_environment_action_bounds",
]
