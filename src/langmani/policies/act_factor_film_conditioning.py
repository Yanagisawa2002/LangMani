"""Shared TaskSpec-to-factor conditioning for M4.3b training and inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import TaskSpec, stable_task_id
from langmani.policies.act_factor_film_types import (
    DESTINATION_BIN_IDS,
    FACTOR_FILM_BIN_INDEX_KEY,
    FACTOR_FILM_OBJECT_INDEX_KEY,
    TARGET_OBJECT_IDS,
    FactorFiLMContractError,
    FactorFiLMRuntimeInput,
    FactorizedTaskCondition,
)
from langmani.policies.act_types import task_spec_from_stable_id


def factorized_task_condition(task: str | TaskSpec) -> FactorizedTaskCondition:
    """Decompose one stable M1 task without reading or parsing instruction text."""
    if isinstance(task, TaskSpec):
        task_spec = task
        task_id = stable_task_id(task)
    elif isinstance(task, str) and task:
        try:
            task_spec = task_spec_from_stable_id(task, CANONICAL_TASK_SPECS)
        except (TypeError, ValueError) as error:
            raise FactorFiLMContractError(f"unknown stable task ID {task!r}") from error
        task_id = task
    else:
        raise TypeError("task must be a non-empty stable task ID or TaskSpec")
    try:
        object_index = TARGET_OBJECT_IDS.index(task_spec.target_object_id)
        bin_index = DESTINATION_BIN_IDS.index(task_spec.target_bin_id)
    except ValueError as error:
        raise FactorFiLMContractError("TaskSpec contains an unsupported object/bin") from error
    if task_spec.instruction_template_id != "canonical_v0":
        raise FactorFiLMContractError("FactorFiLM supports only canonical_v0 TaskSpecs")
    return FactorizedTaskCondition(
        task_id=task_id,
        target_object_id=task_spec.target_object_id,
        destination_bin_id=task_spec.target_bin_id,
        target_object_index=object_index,
        destination_bin_index=bin_index,
    )


def factorized_runtime_input(
    tasks: str | TaskSpec | Sequence[str | TaskSpec],
) -> FactorFiLMRuntimeInput:
    """Build one immutable batch of factorized command conditions."""
    values = (tasks,) if isinstance(tasks, str | TaskSpec) else tuple(tasks)
    if not values:
        raise FactorFiLMContractError("FactorFiLM task batch must not be empty")
    return FactorFiLMRuntimeInput(
        conditions=tuple(factorized_task_condition(value) for value in values)
    )


def runtime_input_from_episode_indices(
    episode_indices: torch.Tensor,
    *,
    task_id_by_episode: Mapping[int, str],
) -> FactorFiLMRuntimeInput:
    """Resolve a training batch exclusively through immutable episode provenance."""
    if not isinstance(episode_indices, torch.Tensor):
        raise TypeError("episode_indices must be a torch Tensor")
    if episode_indices.dtype == torch.bool or episode_indices.dtype.is_floating_point:
        raise FactorFiLMContractError("episode indices must use an integer dtype")
    if episode_indices.ndim not in (1, 2):
        raise FactorFiLMContractError("episode indices must be one-dimensional")
    flattened = episode_indices.reshape(-1)
    indices = tuple(int(value) for value in flattened.detach().cpu().tolist())
    try:
        tasks = tuple(task_id_by_episode[index] for index in indices)
    except KeyError as error:
        raise FactorFiLMContractError(
            f"episode {error.args[0]} has no stable M3B task mapping"
        ) from error
    return factorized_runtime_input(tasks)


def runtime_input_tensors(
    value: FactorFiLMRuntimeInput,
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize exact long[B] indices on the requested policy device."""
    if not isinstance(value, FactorFiLMRuntimeInput):
        raise TypeError("value must be a FactorFiLMRuntimeInput")
    object_indices = torch.tensor(
        [item.target_object_index for item in value.conditions],
        dtype=torch.long,
        device=device,
    )
    bin_indices = torch.tensor(
        [item.destination_bin_index for item in value.conditions],
        dtype=torch.long,
        device=device,
    )
    if object_indices.shape != bin_indices.shape or object_indices.ndim != 1:
        raise RuntimeError("FactorFiLM runtime tensors failed to produce aligned long[B]")
    return object_indices, bin_indices


def attach_factor_film_runtime_input(
    batch: Mapping[str, object],
    runtime_input: FactorFiLMRuntimeInput,
) -> dict[str, object]:
    """Attach internal condition indices after standard ACT preprocessing."""
    if FACTOR_FILM_OBJECT_INDEX_KEY in batch or FACTOR_FILM_BIN_INDEX_KEY in batch:
        raise FactorFiLMContractError("FactorFiLM runtime indices are already present")
    state = batch.get("observation.state")
    if not isinstance(state, torch.Tensor) or state.ndim != 2 or state.shape[1] != 9:
        raise FactorFiLMContractError("FactorFiLM requires processed Panda state shaped [B,9]")
    if state.dtype != torch.float32:
        raise FactorFiLMContractError("FactorFiLM Panda state must be float32")
    if state.shape[0] != runtime_input.batch_size:
        raise FactorFiLMContractError("condition batch size differs from policy state batch size")
    object_indices, bin_indices = runtime_input_tensors(runtime_input, device=state.device)
    result = dict(batch)
    result[FACTOR_FILM_OBJECT_INDEX_KEY] = object_indices
    result[FACTOR_FILM_BIN_INDEX_KEY] = bin_indices
    return result


def extract_factor_film_runtime_tensors(
    batch: Mapping[str, object],
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    """Remove the two project-only keys before entering installed ACT."""
    object_indices = batch.get(FACTOR_FILM_OBJECT_INDEX_KEY)
    bin_indices = batch.get(FACTOR_FILM_BIN_INDEX_KEY)
    if not isinstance(object_indices, torch.Tensor) or not isinstance(bin_indices, torch.Tensor):
        raise FactorFiLMContractError("FactorFiLM policy input lacks both condition tensors")
    if object_indices.dtype != torch.long or bin_indices.dtype != torch.long:
        raise FactorFiLMContractError("FactorFiLM condition tensors must use torch.long")
    if object_indices.ndim != 1 or tuple(object_indices.shape) != tuple(bin_indices.shape):
        raise FactorFiLMContractError(
            "FactorFiLM condition tensors must be aligned rank-one batches"
        )
    if object_indices.numel() < 1:
        raise FactorFiLMContractError("FactorFiLM condition batch must not be empty")
    clean: dict[str, Any] = {}
    for key, item in batch.items():
        if key in {FACTOR_FILM_OBJECT_INDEX_KEY, FACTOR_FILM_BIN_INDEX_KEY}:
            continue
        clean[key] = item
    state = clean.get("observation.state")
    if not isinstance(state, torch.Tensor) or state.ndim != 2:
        raise FactorFiLMContractError("FactorFiLM policy input lacks batched Panda state")
    if state.shape[0] != object_indices.shape[0]:
        raise FactorFiLMContractError("condition and policy batch sizes differ")
    if object_indices.device != state.device or bin_indices.device != state.device:
        raise FactorFiLMContractError(
            "FactorFiLM conditions and policy state use different devices"
        )
    return clean, object_indices, bin_indices


__all__ = [
    "attach_factor_film_runtime_input",
    "extract_factor_film_runtime_tensors",
    "factorized_runtime_input",
    "factorized_task_condition",
    "runtime_input_from_episode_indices",
    "runtime_input_tensors",
]
