"""Model-agnostic policy and action-chunk interfaces for LangMani 2.0."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from langmani.datasets.identity import sha256_hex
from langmani.v2.taxonomy import EvaluationTask

POLICY_INTERFACE_SCHEMA_VERSION = "langmani-v2-policy-interface-v0"
type Array = np.ndarray | torch.Tensor


class PolicyContractError(RuntimeError):
    """Raised when a policy adapter violates the common v2 contract."""


def _json_mapping(value: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    try:
        sha256_hex(dict(value))
    except (TypeError, ValueError) as error:
        raise PolicyContractError(f"{label} must be JSON-serializable") from error
    return dict(value)


@dataclass(frozen=True, slots=True)
class PolicyIdentity:
    """Compact identity shared by every future policy implementation."""

    policy_id: str
    adapter_name: str
    implementation: str
    checkpoint_identity: str
    compatible_skill_families: tuple[str, ...]
    compatible_task_ids: tuple[str, ...]
    schema_version: str = POLICY_INTERFACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != POLICY_INTERFACE_SCHEMA_VERSION:
            raise PolicyContractError("unknown policy interface schema version")
        for value, label in (
            (self.policy_id, "policy_id"),
            (self.adapter_name, "adapter_name"),
            (self.implementation, "implementation"),
            (self.checkpoint_identity, "checkpoint_identity"),
        ):
            if not isinstance(value, str) or not value:
                raise PolicyContractError(f"{label} must be a non-empty string")
        if not self.compatible_skill_families or not self.compatible_task_ids:
            raise PolicyContractError("policy compatibility sets cannot be empty")
        if len(set(self.compatible_task_ids)) != len(self.compatible_task_ids):
            raise PolicyContractError("compatible_task_ids must be unique")

    @property
    def fingerprint(self) -> str:
        return f"sha256:{sha256_hex(self.to_dict())}"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "adapter_name": self.adapter_name,
            "implementation": self.implementation,
            "checkpoint_identity": self.checkpoint_identity,
            "compatible_skill_families": list(self.compatible_skill_families),
            "compatible_task_ids": list(self.compatible_task_ids),
        }


@dataclass(frozen=True, slots=True)
class PolicyContext:
    """Episode-local task binding supplied at every policy reset.

    Historical LangMani v2 environments pass a structured
    :class:`EvaluationTask`.  Official ManiSkill consumers such as Phase 2C-A
    have a stable environment task ID but no LangMani-specific TaskSpec.  The
    optional ``task_id`` path keeps both behind the same policy interface
    without fabricating a taxonomy object.
    """

    evaluation_task: EvaluationTask | None
    evaluation_id: str
    language_instruction: str | None = None
    task_id: str | None = None
    allow_blank_language_instruction: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation_id, str) or not self.evaluation_id:
            raise PolicyContractError("evaluation_id must be a non-empty string")
        if self.language_instruction is not None and not isinstance(self.language_instruction, str):
            raise PolicyContractError("language_instruction must be None or a string")
        if (
            isinstance(self.language_instruction, str)
            and not self.language_instruction.strip()
            and not self.allow_blank_language_instruction
        ):
            raise PolicyContractError(
                "blank language_instruction requires an explicit intervention flag"
            )
        if not isinstance(self.allow_blank_language_instruction, bool):
            raise PolicyContractError("allow_blank_language_instruction must be bool")
        if self.evaluation_task is None:
            if not isinstance(self.task_id, str) or not self.task_id:
                raise PolicyContractError(
                    "a context without EvaluationTask requires a non-empty task_id"
                )
        elif self.task_id is not None:
            raise PolicyContractError(
                "structured EvaluationTask and direct task_id are mutually exclusive"
            )

    @property
    def canonical_task_id(self) -> str:
        """Return the stable task identity without exposing task internals."""

        if self.task_id is not None:
            return self.task_id
        assert self.evaluation_task is not None
        return self.evaluation_task.task_instance.canonical_task_id


@dataclass(frozen=True, slots=True)
class ObservationBatch:
    """Environment-extracted observation features before policy preprocessing."""

    features: Mapping[str, torch.Tensor]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.features or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in self.features.items()
        ):
            raise PolicyContractError("observation features must be a non-empty tensor mapping")
        object.__setattr__(self, "features", dict(self.features))
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, "observation metadata"))


@dataclass(frozen=True, slots=True)
class ActionChunk:
    """A validated time-major action sequence with an explicit prefix mask."""

    actions: Array
    valid_mask: Array
    horizon: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int) or self.horizon < 1:
            raise PolicyContractError("action chunk horizon must be a positive integer")
        if not isinstance(self.actions, (np.ndarray, torch.Tensor)):
            raise PolicyContractError("actions must be a NumPy array or Torch tensor")
        if not isinstance(self.valid_mask, (np.ndarray, torch.Tensor)):
            raise PolicyContractError("valid_mask must be a NumPy array or Torch tensor")
        if self.actions.ndim != 2 or self.actions.shape[0] != self.horizon:
            raise PolicyContractError("actions must have time-major shape [horizon, action_dim]")
        if self.actions.shape[1] < 1:
            raise PolicyContractError("action dimension must be positive")
        if self.valid_mask.ndim != 1 or tuple(self.valid_mask.shape) != (self.horizon,):
            raise PolicyContractError("valid_mask must have shape [horizon]")
        if isinstance(self.valid_mask, torch.Tensor):
            if self.valid_mask.dtype is not torch.bool:
                raise PolicyContractError("Torch valid_mask must use bool dtype")
            mask = self.valid_mask.detach().cpu().numpy()
        else:
            if self.valid_mask.dtype != np.dtype(np.bool_):
                raise PolicyContractError("NumPy valid_mask must use bool dtype")
            mask = self.valid_mask
        valid_count = int(np.count_nonzero(mask))
        if (
            valid_count < 1
            or not bool(np.all(mask[:valid_count]))
            or bool(np.any(mask[valid_count:]))
        ):
            raise PolicyContractError("valid_mask must contain one non-empty true prefix")
        if isinstance(self.actions, torch.Tensor):
            if not self.actions.is_floating_point() or not bool(torch.isfinite(self.actions).all()):
                raise PolicyContractError("Torch actions must be finite floating values")
        elif (
            not np.issubdtype(self.actions.dtype, np.floating)
            or not np.isfinite(self.actions).all()
        ):
            raise PolicyContractError("NumPy actions must be finite floating values")
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, "action metadata"))

    @property
    def valid_action_count(self) -> int:
        if isinstance(self.valid_mask, torch.Tensor):
            return int(torch.count_nonzero(self.valid_mask).detach().cpu())
        return int(np.count_nonzero(self.valid_mask))

    def iter_valid_actions(self):
        """Yield detached per-step actions from the validated true prefix."""

        for index in range(self.valid_action_count):
            yield self.actions[index]


@runtime_checkable
class PolicyAdapter(Protocol):
    """Only the common surface consumed by the unified evaluator."""

    @property
    def identity(self) -> PolicyIdentity: ...

    @property
    def runtime_manifest(self) -> Mapping[str, Any]: ...

    def reset(self, context: PolicyContext) -> None: ...

    def act(self, observation: ObservationBatch) -> ActionChunk: ...

    def close(self) -> None: ...


PolicyFactory = Callable[..., PolicyAdapter]


class PolicyRegistry:
    """Small explicit adapter registry; future adapters opt in by name."""

    def __init__(self) -> None:
        self._factories: dict[str, PolicyFactory] = {}

    def register(self, adapter_name: str, factory: PolicyFactory) -> None:
        if not isinstance(adapter_name, str) or not adapter_name:
            raise PolicyContractError("adapter_name must be a non-empty string")
        if adapter_name in self._factories:
            raise PolicyContractError(f"policy adapter {adapter_name!r} is already registered")
        self._factories[adapter_name] = factory

    def create(self, adapter_name: str, **kwargs: Any) -> PolicyAdapter:
        try:
            factory = self._factories[adapter_name]
        except KeyError as error:
            raise PolicyContractError(f"unregistered policy adapter {adapter_name!r}") from error
        adapter = factory(**kwargs)
        if not isinstance(adapter, PolicyAdapter):
            raise PolicyContractError("policy factory did not return a PolicyAdapter")
        return adapter

    @property
    def registered_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def finite_latency_ms(start: float, end: float) -> float:
    """Validate a measured duration before serializing it."""

    value = (end - start) * 1000.0
    if not math.isfinite(value) or value < 0:
        raise PolicyContractError("latency measurement must be finite and non-negative")
    return value


__all__ = [
    "POLICY_INTERFACE_SCHEMA_VERSION",
    "ActionChunk",
    "ObservationBatch",
    "PolicyAdapter",
    "PolicyContext",
    "PolicyContractError",
    "PolicyIdentity",
    "PolicyRegistry",
    "finite_latency_ms",
]
