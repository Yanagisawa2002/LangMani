"""Strict native-archive validation for LangMani 2.0 pushing data."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from langmani.v2.push_dataset import PushDatasetContractError, sha256_bytes

PUSH_RAW_ARCHIVE_SCHEMA = "langmani-v2-phase2b-native-archive-v0"
PUSH_ACTION_DIMENSION = 8


@dataclass(frozen=True, slots=True)
class PushNativeEpisodeValidation:
    """Validated time contract and stable hashes for one native trajectory."""

    native_episode_id: int
    h5_group: str
    action_count: int
    trajectory_sha256: str
    initial_state_sha256: str
    final_state_sha256: str
    actions_finite: bool
    observations_finite: bool
    time_contract_valid: bool
    action_contract_valid: bool
    native_final_success: bool
    native_final_fail: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": PUSH_RAW_ARCHIVE_SCHEMA,
            "native_episode_id": self.native_episode_id,
            "h5_group": self.h5_group,
            "action_count": self.action_count,
            "trajectory_sha256": self.trajectory_sha256,
            "initial_state_sha256": self.initial_state_sha256,
            "final_state_sha256": self.final_state_sha256,
            "actions_finite": self.actions_finite,
            "observations_finite": self.observations_finite,
            "time_contract_valid": self.time_contract_valid,
            "action_contract_valid": self.action_contract_valid,
            "native_final_success": self.native_final_success,
            "native_final_fail": self.native_final_fail,
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def validate_push_native_episode(
    h5_path: str | Path,
    json_path: str | Path,
    *,
    native_episode_id: int,
) -> PushNativeEpisodeValidation:
    """Validate T actions/labels plus T+1 observations and simulator states."""

    if isinstance(native_episode_id, bool) or not isinstance(native_episode_id, int):
        raise TypeError("native_episode_id must be an integer")
    if native_episode_id < 0:
        raise ValueError("native_episode_id must be non-negative")
    h5_source = Path(h5_path)
    json_source = Path(json_path)
    try:
        metadata = json.loads(json_source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PushDatasetContractError(f"cannot read native metadata: {error}") from error
    if not isinstance(metadata, Mapping):
        raise PushDatasetContractError("native metadata root must be an object")
    episodes = metadata.get("episodes")
    if not isinstance(episodes, list):
        raise PushDatasetContractError("native metadata must contain episodes")
    matching = [
        episode
        for episode in episodes
        if isinstance(episode, Mapping) and episode.get("episode_id") == native_episode_id
    ]
    if len(matching) != 1:
        raise PushDatasetContractError("native HDF5/JSON episode reference is not unique")

    group_name = f"traj_{native_episode_id}"
    try:
        with h5py.File(h5_source, "r") as archive:
            value = archive.get(group_name)
            if not isinstance(value, h5py.Group):
                raise PushDatasetContractError(f"native group {group_name!r} is missing")
            members = set(value.keys())
            required = {
                "actions",
                "terminated",
                "truncated",
                "success",
                "fail",
                "env_states",
                "obs",
            }
            if not required.issubset(members):
                raise PushDatasetContractError(
                    f"{group_name} missing native members {sorted(required - members)}"
                )
            actions = _dataset(value, "actions")
            if actions.ndim != 2:
                raise PushDatasetContractError("native actions must have rank 2")
            action_count = int(actions.shape[0])
            if action_count < 1:
                raise PushDatasetContractError("native episode must contain an action")
            label_values: dict[str, np.ndarray] = {}
            for key in ("terminated", "truncated", "success", "fail"):
                dataset = _dataset(value, key)
                if dataset.shape != (action_count,):
                    raise PushDatasetContractError(
                        f"{group_name}/{key} must have shape ({action_count},)"
                    )
                label_values[key] = np.asarray(dataset)
            actions_array = np.asarray(actions)
            actions_finite = bool(np.all(np.isfinite(actions_array)))
            action_contract_valid = bool(
                actions_array.dtype == np.dtype(np.float32)
                and actions_array.shape == (action_count, PUSH_ACTION_DIMENSION)
                and actions_finite
            )
            observations_finite = _validate_tree(
                value["obs"], expected_length=action_count + 1, context=f"{group_name}/obs"
            )
            _validate_tree(
                value["env_states"],
                expected_length=action_count + 1,
                context=f"{group_name}/env_states",
            )
            env_states = value["env_states"]
            if not isinstance(env_states, h5py.Group):
                raise PushDatasetContractError("env_states must be an HDF5 group")
            initial_hash = _state_index_sha256(env_states, 0)
            final_hash = _state_index_sha256(env_states, action_count)
            trajectory_hash = sha256_bytes(actions_array.tobytes(order="C"))
            terminal = label_values["terminated"] | label_values["truncated"]
            time_contract_valid = bool(not np.any(terminal[:-1]) and bool(terminal[-1]))
            return PushNativeEpisodeValidation(
                native_episode_id=native_episode_id,
                h5_group=group_name,
                action_count=action_count,
                trajectory_sha256=trajectory_hash,
                initial_state_sha256=initial_hash,
                final_state_sha256=final_hash,
                actions_finite=actions_finite,
                observations_finite=observations_finite,
                time_contract_valid=time_contract_valid,
                action_contract_valid=action_contract_valid,
                native_final_success=bool(label_values["success"][-1]),
                native_final_fail=bool(label_values["fail"][-1]),
            )
    except OSError as error:
        raise PushDatasetContractError(f"cannot inspect native HDF5: {error}") from error


def load_native_actions(h5_path: str | Path, *, native_episode_id: int) -> np.ndarray:
    """Load the exact stored float32 actions without conversion or clipping."""

    with h5py.File(Path(h5_path), "r") as archive:
        actions = np.asarray(archive[f"traj_{native_episode_id}"]["actions"])
    if actions.dtype != np.dtype(np.float32) or actions.ndim != 2 or actions.shape[1] != 8:
        raise PushDatasetContractError("stored actions violate PandaJointPositionActionV0")
    if not np.all(np.isfinite(actions)):
        raise PushDatasetContractError("stored actions contain NaN or infinity")
    return np.array(actions, dtype=np.float32, copy=True, order="C")


def restore_native_state(
    environment: object,
    h5_path: str | Path,
    *,
    native_episode_id: int,
    state_index: int,
) -> None:
    """Restore one recorded ManiSkill state through its public state-dict API."""

    base = getattr(environment, "unwrapped", environment)
    setter = getattr(base, "set_state_dict", None)
    if not callable(setter):
        raise PushDatasetContractError("environment does not expose set_state_dict")
    with h5py.File(Path(h5_path), "r") as archive:
        group = archive[f"traj_{native_episode_id}"]["env_states"]
        if not isinstance(group, h5py.Group):
            raise PushDatasetContractError("native env_states must be a group")
        state = _read_tree_index(group, state_index)
    setter(state)


def load_native_state(
    h5_path: str | Path,
    *,
    native_episode_id: int,
    state_index: int,
) -> dict[str, Any]:
    """Load one public state-dict-shaped native state slice."""

    with h5py.File(Path(h5_path), "r") as archive:
        group = archive[f"traj_{native_episode_id}"]["env_states"]
        if not isinstance(group, h5py.Group):
            raise PushDatasetContractError("native env_states must be a group")
        return _read_tree_index(group, state_index)


def native_state_sha256(h5_path: str | Path, *, native_episode_id: int, state_index: int) -> str:
    with h5py.File(Path(h5_path), "r") as archive:
        group = archive[f"traj_{native_episode_id}"]["env_states"]
        if not isinstance(group, h5py.Group):
            raise PushDatasetContractError("native env_states must be a group")
        return _state_index_sha256(group, state_index)


def state_mapping_sha256(value: Mapping[str, object]) -> str:
    """Hash one public ManiSkill state mapping like a native state slice."""

    digest = hashlib.sha256()

    def visit(current: Mapping[str, object], prefix: str) -> None:
        for key in sorted(current):
            child = current[key]
            path = f"{prefix}/{key}"
            if isinstance(child, Mapping):
                visit(child, path)
                continue
            candidate = child
            detach = getattr(candidate, "detach", None)
            if callable(detach):
                candidate = detach()
            cpu = getattr(candidate, "cpu", None)
            if callable(cpu):
                candidate = cpu()
            numpy = getattr(candidate, "numpy", None)
            if callable(numpy):
                candidate = numpy()
            array = np.asarray(candidate)
            digest.update(path.encode("utf-8"))
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(json.dumps(array.shape).encode("ascii"))
            digest.update(array.tobytes(order="C"))

    visit(value, "env_states")
    return "sha256:" + digest.hexdigest()


def _dataset(group: h5py.Group, key: str) -> h5py.Dataset:
    value = group.get(key)
    if not isinstance(value, h5py.Dataset):
        raise PushDatasetContractError(f"{group.name}/{key} must be a dataset")
    return value


def _validate_tree(value: object, *, expected_length: int, context: str) -> bool:
    if not isinstance(value, h5py.Group):
        raise PushDatasetContractError(f"{context} must be a group")
    leaves = 0
    finite = True
    for key in sorted(value.keys()):
        child = value[key]
        if isinstance(child, h5py.Group):
            finite &= _validate_tree(
                child, expected_length=expected_length, context=f"{context}/{key}"
            )
        elif isinstance(child, h5py.Dataset):
            leaves += 1
            if child.ndim < 1 or child.shape[0] != expected_length:
                raise PushDatasetContractError(
                    f"{context}/{key} must have first dimension {expected_length}"
                )
            if child.dtype.kind not in "biuf":
                raise PushDatasetContractError(f"{context}/{key} must be numeric")
            finite &= bool(np.all(np.isfinite(np.asarray(child))))
        else:
            raise PushDatasetContractError(f"{context}/{key} has unsupported HDF5 type")
    if leaves == 0 and not any(isinstance(value[key], h5py.Group) for key in value):
        raise PushDatasetContractError(f"{context} has no dataset leaves")
    return finite


def _state_index_sha256(group: h5py.Group, index: int) -> str:
    digest = hashlib.sha256()

    def visit(current: h5py.Group, prefix: str) -> None:
        for key in sorted(current.keys()):
            child = current[key]
            path = f"{prefix}/{key}"
            if isinstance(child, h5py.Group):
                visit(child, path)
            elif isinstance(child, h5py.Dataset):
                if not 0 <= index < child.shape[0]:
                    raise PushDatasetContractError("state index is outside native trajectory")
                value = np.asarray(child[index])
                digest.update(path.encode("utf-8"))
                digest.update(value.dtype.str.encode("ascii"))
                digest.update(json.dumps(value.shape).encode("ascii"))
                digest.update(value.tobytes(order="C"))
            else:
                raise PushDatasetContractError("env_states contains unsupported HDF5 value")

    visit(group, "env_states")
    return "sha256:" + digest.hexdigest()


def _read_tree_index(group: h5py.Group, index: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in group:
        value = group[key]
        if isinstance(value, h5py.Group):
            result[key] = _read_tree_index(value, index)
        elif isinstance(value, h5py.Dataset):
            result[key] = np.asarray(value[index])
        else:
            raise PushDatasetContractError("native state contains unsupported HDF5 value")
    return result


__all__ = [
    "PUSH_RAW_ARCHIVE_SCHEMA",
    "PushNativeEpisodeValidation",
    "load_native_actions",
    "load_native_state",
    "native_state_sha256",
    "restore_native_state",
    "sha256_file",
    "state_mapping_sha256",
    "validate_push_native_episode",
]
