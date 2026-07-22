"""Complete expert-only simulation snapshots for isolated push rollouts."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from langmani.environments.push_to_region import (
    ARM_COLLISION_FORCE_THRESHOLD,
    CONTACT_FORCE_THRESHOLD,
    CONTAINMENT_CLEARANCE,
    ENV_ID,
    LIFT_TOLERANCE,
    MAX_EPISODE_STEPS,
    MINIMUM_PROGRESS,
    REQUIRED_STABLE_SUCCESS_STEPS,
    STALL_STEPS,
    TOPPLE_ALIGNMENT_THRESHOLD,
    WORKSPACE_BOUNDS_XY,
    WRONG_OBJECT_DISPLACEMENT_THRESHOLD,
)

PUSH_SIMULATION_SNAPSHOT_SCHEMA = "langmani-v2-push-simulation-snapshot-v0"

_TASK_TENSOR_NAMES = (
    "_elapsed_steps",
    "_scene_seeds",
    "_target_object_indices",
    "_target_region_indices",
    "_difficulty_indices",
    "_target_region_centers",
    "_target_region_radii",
    "_initial_object_positions",
    "_initial_target_positions",
    "_stable_success_count",
    "_best_target_distance",
    "_steps_without_progress",
    "_event_wrong_object_contact",
    "_event_wrong_object_displaced",
    "_event_target_outside_workspace",
    "_event_target_overshoot",
    "_event_target_toppled",
    "_event_target_lifted",
    "_event_invalid_action",
    "_event_action_out_of_bounds",
    "_event_no_progress_stall",
    "_event_robot_collision",
    "_last_wrong_object_contact",
    "_last_invalid_action",
    "_last_action_out_of_bounds",
    "_last_robot_collision",
)


class PushSimulationStateError(RuntimeError):
    """Raised when a push simulator snapshot is incomplete or incompatible."""


@dataclass(frozen=True, slots=True)
class PushSimulationSnapshot:
    """One complete, in-memory expert-only simulation checkpoint."""

    schema_version: str
    environment_id: str
    episode_spec: Mapping[str, object]
    simulation_state: Mapping[str, object]
    controller_state: object
    task_tensors: Mapping[str, torch.Tensor]
    episode_seed: np.ndarray
    main_seed: tuple[int, ...]
    episode_rng_state: object
    main_rng_state: object
    batched_episode_rng_states: tuple[object, ...]
    batched_main_rng_states: tuple[object, ...]
    state_sha256: str


def capture_push_simulation_state(environment: object) -> PushSimulationSnapshot:
    """Capture physics, controller, task, counters, and RNG state for one push env."""

    base: Any = getattr(environment, "unwrapped", environment)
    _validate_environment(base)
    simulation_state = _clone_value(base.get_state_dict())
    controller_state = _clone_value(base.agent.get_controller_state())
    task_tensors: dict[str, torch.Tensor] = {}
    for name in _TASK_TENSOR_NAMES:
        value = getattr(base, name, None)
        if not isinstance(value, torch.Tensor):
            raise PushSimulationStateError(f"push snapshot is missing tensor {name}")
        task_tensors[name] = value.detach().clone()
    episode_seed = np.asarray(base._episode_seed, dtype=np.int64).copy()
    main_seed = tuple(int(value) for value in base._main_seed)
    snapshot = PushSimulationSnapshot(
        schema_version=PUSH_SIMULATION_SNAPSHOT_SCHEMA,
        environment_id=ENV_ID,
        episode_spec=copy.deepcopy(base.get_episode_specs()[0].to_dict()),
        simulation_state=simulation_state,
        controller_state=controller_state,
        task_tensors=task_tensors,
        episode_seed=episode_seed,
        main_seed=main_seed,
        episode_rng_state=copy.deepcopy(base._episode_rng.get_state()),
        main_rng_state=copy.deepcopy(base._main_rng.get_state()),
        batched_episode_rng_states=tuple(
            copy.deepcopy(rng.get_state()) for rng in base._batched_episode_rng.rngs
        ),
        batched_main_rng_states=tuple(
            copy.deepcopy(rng.get_state()) for rng in base._batched_main_rng.rngs
        ),
        state_sha256="",
    )
    return PushSimulationSnapshot(
        **{
            field: getattr(snapshot, field)
            for field in PushSimulationSnapshot.__dataclass_fields__
            if field != "state_sha256"
        },
        state_sha256=_state_digest(snapshot),
    )


def restore_push_simulation_state(
    environment: object,
    snapshot: PushSimulationSnapshot,
) -> None:
    """Restore a complete checkpoint without resetting or consuming one env step."""

    base: Any = getattr(environment, "unwrapped", environment)
    _validate_environment(base)
    if snapshot.schema_version != PUSH_SIMULATION_SNAPSHOT_SCHEMA:
        raise PushSimulationStateError("unsupported push simulation snapshot schema")
    if snapshot.environment_id != ENV_ID:
        raise PushSimulationStateError("push simulation snapshot environment changed")
    if base.get_episode_specs()[0].to_dict() != dict(snapshot.episode_spec):
        raise PushSimulationStateError("push simulation snapshot task identity changed")
    if snapshot.state_sha256 != _state_digest(snapshot):
        raise PushSimulationStateError("push simulation snapshot contents changed")

    base.set_state_dict(_clone_value(snapshot.simulation_state))
    if snapshot.controller_state:
        base.agent.set_controller_state(_clone_value(snapshot.controller_state))
    for name, value in snapshot.task_tensors.items():
        setattr(base, name, value.detach().clone().to(base.device))
    base._episode_seed = snapshot.episode_seed.copy()
    base._main_seed = list(snapshot.main_seed)
    base._episode_rng.set_state(copy.deepcopy(snapshot.episode_rng_state))
    base._main_rng.set_state(copy.deepcopy(snapshot.main_rng_state))
    _restore_batched_rng(base._batched_episode_rng, snapshot.batched_episode_rng_states)
    _restore_batched_rng(base._batched_main_rng, snapshot.batched_main_rng_states)


def push_simulation_configuration(environment: object) -> dict[str, object]:
    """Return the physics-relevant configuration compared by the sandbox audit."""

    base: Any = getattr(environment, "unwrapped", environment)
    _validate_environment(base)
    low, high = base.get_push_expert_action_bounds()
    return {
        "environment_id": ENV_ID,
        "num_envs": int(base.num_envs),
        "robot_uid": str(base.robot_uids),
        "control_mode": str(base.control_mode),
        "sim_backend": "physx_cuda" if base.gpu_sim_enabled else "physx_cpu",
        "simulation_frequency_hz": int(base.sim_freq),
        "control_frequency_hz": int(base.control_freq),
        "maximum_episode_steps": MAX_EPISODE_STEPS,
        "action_shape": list(low.shape),
        "action_low": [float(value) for value in low],
        "action_high": [float(value) for value in high],
        "controller_type": type(base.agent.controller).__qualname__,
        "enhanced_determinism": bool(base._enhanced_determinism),
        "success_contract": {
            "required_stable_success_steps": REQUIRED_STABLE_SUCCESS_STEPS,
            "containment_clearance": CONTAINMENT_CLEARANCE,
            "lift_tolerance": LIFT_TOLERANCE,
            "topple_alignment_threshold": TOPPLE_ALIGNMENT_THRESHOLD,
        },
        "failure_contract": {
            "workspace_bounds_xy": list(WORKSPACE_BOUNDS_XY),
            "wrong_object_displacement_threshold": WRONG_OBJECT_DISPLACEMENT_THRESHOLD,
            "contact_force_threshold": CONTACT_FORCE_THRESHOLD,
            "arm_collision_force_threshold": ARM_COLLISION_FORCE_THRESHOLD,
            "minimum_progress": MINIMUM_PROGRESS,
            "stall_steps": STALL_STEPS,
        },
    }


def _validate_environment(base: object) -> None:
    if getattr(base, "num_envs", None) != 1:
        raise PushSimulationStateError("push simulation snapshots require num_envs=1")
    for name in (
        "get_state_dict",
        "set_state_dict",
        "get_episode_specs",
        "get_push_expert_action_bounds",
    ):
        if not callable(getattr(base, name, None)):
            raise PushSimulationStateError(f"push environment lacks {name}()")
    if not hasattr(base, "agent"):
        raise PushSimulationStateError("push environment agent is unavailable")


def _restore_batched_rng(value: object, states: tuple[object, ...]) -> None:
    rngs = getattr(value, "rngs", None)
    if not isinstance(rngs, list) or len(rngs) != len(states):
        raise PushSimulationStateError("batched RNG state count changed")
    for rng, state in zip(rngs, states, strict=True):
        rng.set_state(copy.deepcopy(state))


def _clone_value(value: object) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: _clone_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    return copy.deepcopy(value)


def _state_digest(snapshot: PushSimulationSnapshot) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, snapshot.environment_id)
    _update_digest(digest, snapshot.episode_spec)
    _update_digest(digest, snapshot.simulation_state)
    _update_digest(digest, snapshot.controller_state)
    _update_digest(digest, snapshot.task_tensors)
    _update_digest(digest, snapshot.episode_seed)
    _update_digest(digest, snapshot.main_seed)
    _update_digest(digest, snapshot.episode_rng_state)
    _update_digest(digest, snapshot.main_rng_state)
    _update_digest(digest, snapshot.batched_episode_rng_states)
    _update_digest(digest, snapshot.batched_main_rng_states)
    return digest.hexdigest()


def _update_digest(digest: Any, value: object) -> None:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
        return
    if isinstance(value, np.ndarray):
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
        return
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            _update_digest(digest, str(key))
            _update_digest(digest, value[key])
        return
    if isinstance(value, tuple | list):
        for item in value:
            _update_digest(digest, item)
        return
    digest.update(repr(value).encode("utf-8"))


__all__ = [
    "PUSH_SIMULATION_SNAPSHOT_SCHEMA",
    "PushSimulationSnapshot",
    "PushSimulationStateError",
    "capture_push_simulation_state",
    "push_simulation_configuration",
    "restore_push_simulation_state",
]
