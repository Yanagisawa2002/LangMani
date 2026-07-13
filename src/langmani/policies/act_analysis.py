"""Counterfactual prediction analysis for the three M4 ACT variants."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np

from langmani.datasets.lerobot_types import IMAGE_FEATURE_KEY, STATE_FEATURE_KEY
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.policies.act_conditioning import CANONICAL_TASK_IDS
from langmani.policies.act_types import (
    ACT_ACTION_COMPONENTS,
    ActVariant,
    CounterfactualSensitivityResult,
)

_POLICY_INPUT_KEYS = frozenset({IMAGE_FEATURE_KEY, STATE_FEATURE_KEY})


class CounterfactualAnalysisError(ValueError):
    """Raised when a sensitivity comparison is not a controlled intervention."""


@dataclass(frozen=True, slots=True)
class CounterfactualDemonstration:
    task_id: str
    rgb_digest: str
    panda_state: tuple[float, ...]
    action_chunk: tuple[tuple[float, ...], ...]
    action_is_pad: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class OfflineCounterfactualAudit:
    scene_group_count: int
    identical_initial_rgb_fraction: float
    identical_initial_state_fraction: float
    one_to_many_action_fraction: float
    mean_initial_action_variance: float
    mean_pairwise_chunk_distance_by_task_pair: Mapping[str, float]
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_counterfactual_demonstrations(
    groups: Mapping[str, tuple[CounterfactualDemonstration, ...]],
    *,
    state_atol: float = 1e-6,
    action_distance_epsilon: float = 1e-8,
) -> OfflineCounterfactualAudit:
    """Prove that equal initial observations map to task-dependent expert chunks."""
    if not groups:
        raise CounterfactualAnalysisError("offline audit requires at least one scene group")
    rgb_equal = 0
    state_equal = 0
    one_to_many = 0
    variances: list[float] = []
    pair_distances: dict[str, list[float]] = {}
    for group_id in sorted(groups):
        values = groups[group_id]
        by_task = {item.task_id: item for item in values}
        if len(values) != 6 or set(by_task) != set(CANONICAL_TASK_IDS):
            raise CounterfactualAnalysisError(
                f"scene group {group_id!r} must contain the canonical six tasks"
            )
        ordered = tuple(by_task[task_id] for task_id in CANONICAL_TASK_IDS)
        if len({item.rgb_digest for item in ordered}) == 1:
            rgb_equal += 1
        states = np.asarray([item.panda_state for item in ordered], dtype=np.float64)
        if states.shape != (6, 9) or not np.all(np.isfinite(states)):
            raise CounterfactualAnalysisError("initial Panda states must have shape (6,9)")
        if np.allclose(states, states[0], atol=state_atol, rtol=0):
            state_equal += 1
        first_actions: list[np.ndarray] = []
        group_has_difference = False
        for left in range(6):
            left_chunk = np.asarray(ordered[left].action_chunk, dtype=np.float64)
            left_pad = np.asarray(ordered[left].action_is_pad, dtype=np.bool_)
            if (
                left_chunk.ndim != 2
                or left_chunk.shape[1] != 8
                or left_pad.shape != (left_chunk.shape[0],)
            ):
                raise CounterfactualAnalysisError("expert action chunks must have shape (K,8)")
            if left_pad[0]:
                raise CounterfactualAnalysisError("the first expert action cannot be padding")
            first_actions.append(left_chunk[0])
            for right in range(left + 1, 6):
                right_chunk = np.asarray(ordered[right].action_chunk, dtype=np.float64)
                right_pad = np.asarray(ordered[right].action_is_pad, dtype=np.bool_)
                if right_chunk.shape != left_chunk.shape or right_pad.shape != left_pad.shape:
                    raise CounterfactualAnalysisError("counterfactual chunks must share one shape")
                valid = ~(left_pad | right_pad)
                if not np.any(valid):
                    raise CounterfactualAnalysisError("task pair has no shared valid action step")
                distance = float(
                    np.linalg.norm(left_chunk[valid] - right_chunk[valid], axis=1).mean()
                )
                key = pairwise_key(ordered[left].task_id, ordered[right].task_id)
                pair_distances.setdefault(key, []).append(distance)
                group_has_difference |= distance > action_distance_epsilon
        variances.append(float(np.var(np.stack(first_actions), axis=0).mean()))
        one_to_many += int(group_has_difference)
    count = len(groups)
    pair_means = {
        key: float(sum(values) / len(values)) for key, values in sorted(pair_distances.items())
    }
    return OfflineCounterfactualAudit(
        scene_group_count=count,
        identical_initial_rgb_fraction=rgb_equal / count,
        identical_initial_state_fraction=state_equal / count,
        one_to_many_action_fraction=one_to_many / count,
        mean_initial_action_variance=float(sum(variances) / len(variances)),
        mean_pairwise_chunk_distance_by_task_pair=pair_means,
        passed=rgb_equal == state_equal == one_to_many == count,
    )


def audit_m3b_train_counterfactuals(
    completed: object,
    *,
    policy_config: object,
) -> OfflineCounterfactualAudit:
    """Load only M3B train first frames through the public ACT temporal reader."""
    from langmani.policies.act_data import load_lerobot_episode_view

    loaded = load_lerobot_episode_view(
        completed,
        completed.views.train,
        policy_config=policy_config,
        include_delta_timestamps=True,
        return_uint8=True,
    )
    dataset = loaded.dataset
    episode_values = dataset.hf_dataset["episode_index"]
    frame_values = dataset.hf_dataset["frame_index"]
    first_position_by_episode: dict[int, int] = {}
    for position, (episode_value, frame_value) in enumerate(
        zip(episode_values, frame_values, strict=True)
    ):
        episode_index = int(episode_value)
        if int(frame_value) == 0 and episode_index not in first_position_by_episode:
            first_position_by_episode[episode_index] = position
    if set(first_position_by_episode) != set(completed.views.train.episode_indices):
        raise CounterfactualAnalysisError("M3B train view has missing/duplicate initial frames")
    groups: dict[str, list[CounterfactualDemonstration]] = {}
    for episode_index in completed.views.train.episode_indices:
        row = dataset[first_position_by_episode[episode_index]]
        image = _numeric_array(row[IMAGE_FEATURE_KEY], "initial RGB")
        state = _numeric_array(row[STATE_FEATURE_KEY], "initial Panda state")
        actions = _numeric_array(row["action"], "initial action chunk")
        padding = _boolean_array(row["action_is_pad"], "initial action padding")
        digest = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
        group_id = completed.views.scene_group_id_by_episode[episode_index]
        groups.setdefault(group_id, []).append(
            CounterfactualDemonstration(
                task_id=completed.views.task_id_by_episode[episode_index],
                rgb_digest=digest,
                panda_state=tuple(float(value) for value in state.reshape(-1)),
                action_chunk=tuple(
                    tuple(float(component) for component in action) for action in actions
                ),
                action_is_pad=tuple(bool(value) for value in padding.reshape(-1)),
            )
        )
    return audit_counterfactual_demonstrations({key: tuple(value) for key, value in groups.items()})


def load_m3b_reference_counterfactual_group(
    completed: object,
    *,
    policy_config: object,
) -> tuple[CounterfactualDemonstration, ...]:
    """Load the first canonical train scene group through LeRobot's temporal API."""

    from langmani.policies.act_data import load_lerobot_episode_view

    train_indices = completed.views.train.episode_indices
    if not train_indices:
        raise CounterfactualAnalysisError("M3B train view is empty")
    reference_group_id = completed.views.scene_group_id_by_episode[train_indices[0]]
    reference_indices = tuple(
        index
        for index in train_indices
        if completed.views.scene_group_id_by_episode[index] == reference_group_id
    )
    if len(reference_indices) != 6:
        raise CounterfactualAnalysisError("reference train scene group must contain six episodes")
    loaded = load_lerobot_episode_view(
        completed,
        completed.views.train,
        policy_config=policy_config,
        include_delta_timestamps=True,
        return_uint8=True,
    )
    episode_values = loaded.dataset.hf_dataset["episode_index"]
    frame_values = loaded.dataset.hf_dataset["frame_index"]
    first_position_by_episode = {
        int(episode): position
        for position, (episode, frame) in enumerate(zip(episode_values, frame_values, strict=True))
        if int(frame) == 0 and int(episode) in reference_indices
    }
    if set(first_position_by_episode) != set(reference_indices):
        raise CounterfactualAnalysisError("reference group has missing initial temporal samples")
    by_task: dict[str, CounterfactualDemonstration] = {}
    for episode_index in reference_indices:
        row = loaded.dataset[first_position_by_episode[episode_index]]
        image = _numeric_array(row[IMAGE_FEATURE_KEY], "reference initial RGB")
        state = _numeric_array(row[STATE_FEATURE_KEY], "reference initial Panda state")
        actions = _numeric_array(row["action"], "reference expert action chunk")
        padding = _boolean_array(row["action_is_pad"], "reference expert action padding")
        task_id = completed.views.task_id_by_episode[episode_index]
        by_task[task_id] = CounterfactualDemonstration(
            task_id=task_id,
            rgb_digest=hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest(),
            panda_state=tuple(float(value) for value in state.reshape(-1)),
            action_chunk=tuple(
                tuple(float(component) for component in action) for action in actions
            ),
            action_is_pad=tuple(bool(value) for value in padding.reshape(-1)),
        )
    if set(by_task) != set(CANONICAL_TASK_IDS):
        raise CounterfactualAnalysisError("reference group does not cover canonical tasks")
    return tuple(by_task[task_id] for task_id in CANONICAL_TASK_IDS)


def expert_chunk_distances(
    predicted_action_chunks: object,
    demonstrations: Sequence[CounterfactualDemonstration],
) -> dict[str, float]:
    """Compare six policy chunks with their task-matched expert target chunks."""

    predicted = _numeric_array(predicted_action_chunks, "predicted action chunks")
    if predicted.ndim != 3 or predicted.shape[0] != 6 or predicted.shape[2] != 8:
        raise CounterfactualAnalysisError("predicted action chunks must have shape (6,K,8)")
    if tuple(item.task_id for item in demonstrations) != CANONICAL_TASK_IDS:
        raise CounterfactualAnalysisError("expert chunks must use canonical task ordering")
    result: dict[str, float] = {}
    for index, demonstration in enumerate(demonstrations):
        expert = np.asarray(demonstration.action_chunk, dtype=np.float64)
        padding = np.asarray(demonstration.action_is_pad, dtype=np.bool_)
        if expert.shape != predicted[index].shape or padding.shape != (expert.shape[0],):
            raise CounterfactualAnalysisError("policy and expert chunks must share shape (K,8)")
        valid = ~padding
        if not np.any(valid):
            raise CounterfactualAnalysisError("expert target chunk has no valid action")
        result[demonstration.task_id] = float(
            np.linalg.norm(predicted[index, valid] - expert[valid], axis=1).mean()
        )
    return result


def task_identity_effects(
    pairwise_chunk_distances: Mapping[str, float],
) -> dict[str, float]:
    """Separate target-object and destination-bin prediction effects."""

    target_effects: list[float] = []
    bin_effects: list[float] = []
    for left in range(6):
        for right in range(left + 1, 6):
            key = pairwise_key(CANONICAL_TASK_IDS[left], CANONICAL_TASK_IDS[right])
            if key not in pairwise_chunk_distances:
                raise CounterfactualAnalysisError(f"missing canonical task-pair distance {key!r}")
            distance = float(pairwise_chunk_distances[key])
            left_spec = CANONICAL_TASK_SPECS[left]
            right_spec = CANONICAL_TASK_SPECS[right]
            if left_spec.target_bin_id == right_spec.target_bin_id:
                target_effects.append(distance)
            if left_spec.target_object_id == right_spec.target_object_id:
                bin_effects.append(distance)
    if len(target_effects) != 6 or len(bin_effects) != 3:
        raise RuntimeError("canonical object/bin effect comparison count changed")
    return {
        "different_target_same_bin_mean_chunk_distance": float(np.mean(target_effects)),
        "same_target_different_bin_mean_chunk_distance": float(np.mean(bin_effects)),
    }


def pairwise_key(left_task_id: str, right_task_id: str) -> str:
    """Return one stable, directional-by-canonical-order pair identifier."""

    if not isinstance(left_task_id, str) or not left_task_id:
        raise TypeError("left_task_id must be a non-empty string")
    if not isinstance(right_task_id, str) or not right_task_id:
        raise TypeError("right_task_id must be a non-empty string")
    if left_task_id == right_task_id:
        raise CounterfactualAnalysisError("a pair must contain two different task IDs")
    return f"{left_task_id}|{right_task_id}"


def policy_inputs_identical(inputs: Sequence[Mapping[str, object]]) -> bool:
    """Check exact equality of six image/state policy inputs and reject hidden keys."""

    arrays = _validated_policy_inputs(inputs)
    reference = arrays[0]
    return all(
        all(
            current[key].dtype == reference[key].dtype
            and current[key].shape == reference[key].shape
            and np.array_equal(current[key], reference[key])
            for key in reference
        )
        for current in arrays[1:]
    )


def _validated_policy_inputs(
    inputs: Sequence[Mapping[str, object]],
) -> tuple[dict[str, np.ndarray], ...]:
    if len(inputs) != 6:
        raise CounterfactualAnalysisError("counterfactual input audit requires six tasks")
    arrays: list[dict[str, np.ndarray]] = []
    for index, candidate in enumerate(inputs):
        if not isinstance(candidate, Mapping):
            raise TypeError(f"policy input {index} must be a mapping")
        if set(candidate) != _POLICY_INPUT_KEYS:
            raise CounterfactualAnalysisError(
                "counterfactual model inputs must contain only base_camera and observation.state"
            )
        arrays.append(
            {
                key: _numeric_array(candidate[key], f"policy input {index} {key}")
                for key in sorted(_POLICY_INPUT_KEYS)
            }
        )
    return tuple(arrays)


def _validate_controlled_intervention(
    variant: ActVariant,
    inputs: Sequence[Mapping[str, object]],
) -> bool:
    arrays = _validated_policy_inputs(inputs)
    if variant in {ActVariant.MIXED_UNCONDITIONED, ActVariant.PER_TASK}:
        if not policy_inputs_identical(inputs):
            raise CounterfactualAnalysisError(
                f"{variant.value} comparisons did not hold the physical policy input fixed"
            )
        return True

    reference_image = arrays[0][IMAGE_FEATURE_KEY]
    states = [item[STATE_FEATURE_KEY] for item in arrays]
    if any(
        image.dtype != reference_image.dtype
        or image.shape != reference_image.shape
        or not np.array_equal(image, reference_image)
        for image in (item[IMAGE_FEATURE_KEY] for item in arrays[1:])
    ):
        raise CounterfactualAnalysisError(
            "mixed_task_onehot comparisons must hold base_camera RGB fixed"
        )
    if any(state.shape != (15,) or state.dtype != np.dtype(np.float32) for state in states):
        raise CounterfactualAnalysisError(
            "mixed_task_onehot controlled inputs require float32 state shape (15,)"
        )
    if any(not np.array_equal(state[:9], states[0][:9]) for state in states[1:]):
        raise CounterfactualAnalysisError(
            "mixed_task_onehot comparisons must hold PandaPolicyStateV0 fixed"
        )
    suffixes = np.stack([state[9:] for state in states])
    if not np.array_equal(suffixes, np.eye(6, dtype=np.float32)):
        raise CounterfactualAnalysisError(
            "mixed_task_onehot inputs must vary only through canonical one-hot basis vectors"
        )
    return False


def compute_counterfactual_sensitivity(
    *,
    variant: ActVariant,
    scene_id: str,
    ordered_task_ids: tuple[str, ...],
    first_actions: object,
    action_chunks: object,
    policy_inputs: Sequence[Mapping[str, object]] | None,
) -> CounterfactualSensitivityResult:
    """Measure all 15 task-pair distances under one fixed physical observation.

    Chunk distance is the mean per-time-step Euclidean action distance, so it
    remains in action units and is not inflated merely by a longer chunk.
    """

    variant = ActVariant(variant)
    if not isinstance(scene_id, str) or not scene_id:
        raise TypeError("scene_id must be a non-empty string")
    if ordered_task_ids != CANONICAL_TASK_IDS:
        raise CounterfactualAnalysisError(
            "ordered_task_ids must equal the canonical six stable task IDs"
        )
    actions = _numeric_array(first_actions, "first_actions")
    chunks = _numeric_array(action_chunks, "action_chunks")
    if actions.shape != (6, ACT_ACTION_COMPONENTS):
        raise CounterfactualAnalysisError("first_actions must have shape (6, 8)")
    if chunks.ndim != 3 or chunks.shape[0] != 6 or chunks.shape[2] != ACT_ACTION_COMPONENTS:
        raise CounterfactualAnalysisError("action_chunks must have shape (6, horizon, 8)")
    if chunks.shape[1] < 1:
        raise CounterfactualAnalysisError("action chunks must contain at least one time step")
    if not np.array_equal(actions, chunks[:, 0, :]):
        raise CounterfactualAnalysisError("first_actions must equal the first action in each chunk")

    if policy_inputs is None:
        raise CounterfactualAnalysisError(
            "counterfactual sensitivity requires explicit controlled model-input evidence"
        )
    identical_inputs = _validate_controlled_intervention(variant, policy_inputs)
    if variant is ActVariant.MIXED_UNCONDITIONED and (
        not np.array_equal(actions, np.repeat(actions[:1], 6, axis=0))
        or not np.array_equal(chunks, np.repeat(chunks[:1], 6, axis=0))
    ):
        raise CounterfactualAnalysisError(
            "mixed_unconditioned predictions differ despite identical deterministic inputs"
        )

    first_distances: dict[str, float] = {}
    chunk_distances: dict[str, float] = {}
    for left in range(6):
        for right in range(left + 1, 6):
            key = pairwise_key(ordered_task_ids[left], ordered_task_ids[right])
            first_distances[key] = float(np.linalg.norm(actions[left] - actions[right]))
            step_distances = np.linalg.norm(chunks[left] - chunks[right], axis=1)
            chunk_distances[key] = float(np.mean(step_distances))

    return CounterfactualSensitivityResult(
        variant=variant,
        scene_id=scene_id,
        ordered_task_ids=ordered_task_ids,
        first_actions=tuple(tuple(float(component) for component in action) for action in actions),
        action_chunks=tuple(
            tuple(tuple(float(component) for component in action) for action in chunk)
            for chunk in chunks
        ),
        pairwise_first_action_distances=first_distances,
        pairwise_chunk_distances=chunk_distances,
        identical_unconditioned_inputs=identical_inputs,
    )


def counterfactual_sensitivity_from_dict(value: object) -> CounterfactualSensitivityResult:
    """Strictly reconstruct a sensitivity result from portable JSON data."""

    if not isinstance(value, dict):
        raise TypeError("counterfactual sensitivity JSON must be an object")
    expected = {
        "variant",
        "scene_id",
        "ordered_task_ids",
        "first_actions",
        "action_chunks",
        "pairwise_first_action_distances",
        "pairwise_chunk_distances",
        "identical_unconditioned_inputs",
    }
    if set(value) != expected:
        raise CounterfactualAnalysisError("counterfactual sensitivity JSON fields are invalid")
    raw_task_ids = value["ordered_task_ids"]
    raw_actions = value["first_actions"]
    raw_chunks = value["action_chunks"]
    if (
        not isinstance(raw_task_ids, list)
        or not isinstance(raw_actions, list)
        or not isinstance(raw_chunks, list)
    ):
        raise TypeError("counterfactual task IDs, first actions, and chunks must be lists")
    if not all(isinstance(action, list) for action in raw_actions):
        raise TypeError("counterfactual first actions must be nested lists")
    first_actions = _numeric_array(raw_actions, "serialized first_actions")
    action_chunks = _numeric_array(raw_chunks, "serialized action_chunks")
    if first_actions.shape != (6, ACT_ACTION_COMPONENTS):
        raise CounterfactualAnalysisError("serialized first_actions must have shape (6, 8)")
    if (
        action_chunks.ndim != 3
        or action_chunks.shape[0] != 6
        or action_chunks.shape[1] < 1
        or action_chunks.shape[2] != ACT_ACTION_COMPONENTS
    ):
        raise CounterfactualAnalysisError("serialized action_chunks must have shape (6,K,8)")
    if not isinstance(value["identical_unconditioned_inputs"], bool):
        raise TypeError("identical_unconditioned_inputs must be a boolean")
    task_ids = tuple(raw_task_ids)
    if task_ids != CANONICAL_TASK_IDS:
        raise CounterfactualAnalysisError(
            "serialized sensitivity task IDs must use canonical stable ordering"
        )
    expected_pairs = {
        pairwise_key(task_ids[left], task_ids[right])
        for left in range(6)
        for right in range(left + 1, 6)
    }
    for field in ("pairwise_first_action_distances", "pairwise_chunk_distances"):
        distances = value[field]
        if not isinstance(distances, dict) or set(distances) != expected_pairs:
            raise CounterfactualAnalysisError(f"{field} must contain all 15 canonical task pairs")
        if any(
            isinstance(distance, bool)
            or not isinstance(distance, int | float)
            or not np.isfinite(distance)
            or distance < 0
            for distance in distances.values()
        ):
            raise CounterfactualAnalysisError(f"{field} must contain finite non-negative values")
    return CounterfactualSensitivityResult(
        variant=ActVariant(cast(str, value["variant"])),
        scene_id=cast(str, value["scene_id"]),
        ordered_task_ids=task_ids,
        first_actions=tuple(tuple(float(value) for value in action) for action in first_actions),
        action_chunks=tuple(
            tuple(tuple(float(value) for value in action) for action in chunk)
            for chunk in action_chunks
        ),
        pairwise_first_action_distances=cast(
            Mapping[str, float], value["pairwise_first_action_distances"]
        ),
        pairwise_chunk_distances=cast(Mapping[str, float], value["pairwise_chunk_distances"]),
        identical_unconditioned_inputs=cast(bool, value["identical_unconditioned_inputs"]),
    )


def _numeric_array(value: object, label: str) -> np.ndarray:
    candidate: Any = value
    for method_name in ("detach", "cpu", "numpy"):
        method = getattr(candidate, method_name, None)
        if callable(method):
            candidate = method()
    try:
        array = np.asarray(candidate)
    except (TypeError, ValueError) as error:
        raise CounterfactualAnalysisError(f"{label} must be numeric") from error
    if not np.issubdtype(array.dtype, np.number):
        raise CounterfactualAnalysisError(f"{label} must be numeric")
    if not np.all(np.isfinite(array)):
        raise CounterfactualAnalysisError(f"{label} must contain finite values")
    return array


def _boolean_array(value: object, label: str) -> np.ndarray:
    candidate: Any = value
    for method_name in ("detach", "cpu", "numpy"):
        method = getattr(candidate, method_name, None)
        if callable(method):
            candidate = method()
    array = np.asarray(candidate)
    if array.dtype != np.dtype(np.bool_):
        raise CounterfactualAnalysisError(f"{label} must use boolean dtype")
    return array


__all__ = [
    "CounterfactualDemonstration",
    "CounterfactualAnalysisError",
    "OfflineCounterfactualAudit",
    "audit_counterfactual_demonstrations",
    "audit_m3b_train_counterfactuals",
    "compute_counterfactual_sensitivity",
    "counterfactual_sensitivity_from_dict",
    "expert_chunk_distances",
    "load_m3b_reference_counterfactual_group",
    "pairwise_key",
    "policy_inputs_identical",
    "task_identity_effects",
]
