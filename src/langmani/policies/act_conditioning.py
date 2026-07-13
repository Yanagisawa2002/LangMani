"""Canonical discrete task conditioning for the M4 ACT oracle baseline.

The mapping is deliberately derived from the M3A schedule rather than copied
from instruction text.  Both training and inference call the same functions:
training resolves a frame's stable task ID through the M3B episode manifest,
while inference resolves the active :class:`TaskSpec` supplied by M1.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import overload

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import TaskSpec, stable_task_id
from langmani.policies.act_types import (
    ACT_CONDITIONED_STATE_COMPONENTS,
    ACT_STATE_COMPONENTS,
    TASK_ONEHOT_MAPPING_VERSION,
)

CANONICAL_TASK_IDS: tuple[str, ...] = tuple(
    stable_task_id(task_spec) for task_spec in CANONICAL_TASK_SPECS
)
CANONICAL_TASK_COUNT = len(CANONICAL_TASK_IDS)

if CANONICAL_TASK_COUNT != 6:  # pragma: no cover - import-time invariant
    raise RuntimeError("CanonicalTaskOneHotV0 requires exactly six TaskSpecs")


def canonical_task_index(task_id: str) -> int:
    """Resolve one stable task ID without parsing language or identifier text."""
    if not isinstance(task_id, str) or not task_id:
        raise TypeError("task_id must be a non-empty stable task ID")
    try:
        return CANONICAL_TASK_IDS.index(task_id)
    except ValueError as error:
        raise ValueError(f"unknown stable task ID {task_id!r}") from error


def canonical_task_onehot(task_id: str) -> NDArray[np.float32]:
    """Return a fresh float32 CanonicalTaskOneHotV0 vector."""
    result = np.zeros(CANONICAL_TASK_COUNT, dtype=np.float32)
    result[canonical_task_index(task_id)] = np.float32(1.0)
    return result


def canonical_task_onehot_from_spec(task_spec: TaskSpec) -> NDArray[np.float32]:
    """Encode the active M1 command TaskSpec for policy inference."""
    if not isinstance(task_spec, TaskSpec):
        raise TypeError("task_spec must be a TaskSpec")
    return canonical_task_onehot(stable_task_id(task_spec))


@overload
def append_canonical_task_onehot(
    state: NDArray[np.float32], task_ids: str | Sequence[str]
) -> NDArray[np.float32]: ...


@overload
def append_canonical_task_onehot(state: Tensor, task_ids: str | Sequence[str]) -> Tensor: ...


def append_canonical_task_onehot(
    state: NDArray[np.float32] | Tensor,
    task_ids: str | Sequence[str],
) -> NDArray[np.float32] | Tensor:
    """Append the same immutable one-hot mapping to one state or a batch.

    The first nine components are never reordered.  The function rejects
    non-float32 state tensors so conditioning cannot silently alter the M3B
    policy-state contract.
    """
    ids = (task_ids,) if isinstance(task_ids, str) else tuple(task_ids)
    if not ids:
        raise ValueError("task_ids must not be empty")
    vectors = np.stack([canonical_task_onehot(task_id) for task_id in ids])

    if isinstance(state, np.ndarray):
        if state.dtype != np.dtype(np.float32):
            raise ValueError("Panda policy state must have dtype float32")
        if state.ndim == 1:
            if state.shape != (ACT_STATE_COMPONENTS,) or len(ids) != 1:
                raise ValueError("one state requires shape (9,) and exactly one task ID")
            result = np.concatenate((state, vectors[0]), axis=0)
        elif state.ndim == 2:
            if state.shape != (len(ids), ACT_STATE_COMPONENTS):
                raise ValueError("batched state must have shape (len(task_ids), 9)")
            result = np.concatenate((state, vectors), axis=1)
        else:
            raise ValueError("Panda policy state must be rank one or two")
        if result.shape[-1] != ACT_CONDITIONED_STATE_COMPONENTS:
            raise RuntimeError("conditioned ACT state did not produce 15 components")
        return np.asarray(result, dtype=np.float32)

    if not isinstance(state, Tensor):
        raise TypeError("state must be a numpy array or torch Tensor")
    if state.dtype != torch.float32:
        raise ValueError("Panda policy state must have dtype torch.float32")
    condition = torch.from_numpy(vectors).to(device=state.device)
    if state.ndim == 1:
        if tuple(state.shape) != (ACT_STATE_COMPONENTS,) or len(ids) != 1:
            raise ValueError("one state requires shape (9,) and exactly one task ID")
        result_tensor = torch.cat((state, condition[0]), dim=0)
    elif state.ndim == 2:
        if tuple(state.shape) != (len(ids), ACT_STATE_COMPONENTS):
            raise ValueError("batched state must have shape (len(task_ids), 9)")
        result_tensor = torch.cat((state, condition), dim=1)
    else:
        raise ValueError("Panda policy state must be rank one or two")
    if result_tensor.shape[-1] != ACT_CONDITIONED_STATE_COMPONENTS:
        raise RuntimeError("conditioned ACT state did not produce 15 components")
    return result_tensor


def validate_canonical_task_onehot(value: object) -> None:
    """Validate the serialized/raw six-way command condition."""
    if isinstance(value, Tensor):
        array = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        array = value
    else:
        raise TypeError("task one-hot must be a numpy array or torch Tensor")
    if array.dtype != np.dtype(np.float32) or array.shape != (CANONICAL_TASK_COUNT,):
        raise ValueError("CanonicalTaskOneHotV0 must be float32 with shape (6,)")
    if not np.all((array == 0.0) | (array == 1.0)) or float(array.sum()) != 1.0:
        raise ValueError("CanonicalTaskOneHotV0 must contain exactly one active component")


def task_onehot_contract() -> dict[str, object]:
    """Return the portable mapping recorded in run and checkpoint artifacts."""
    return {
        "mapping_version": TASK_ONEHOT_MAPPING_VERSION,
        "ordered_task_ids": list(CANONICAL_TASK_IDS),
        "dtype": "float32",
        "shape": [CANONICAL_TASK_COUNT],
    }


__all__ = [
    "CANONICAL_TASK_COUNT",
    "CANONICAL_TASK_IDS",
    "append_canonical_task_onehot",
    "canonical_task_index",
    "canonical_task_onehot",
    "canonical_task_onehot_from_spec",
    "task_onehot_contract",
    "validate_canonical_task_onehot",
]
