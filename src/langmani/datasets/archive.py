"""Strict ManiSkill-native archive operations for M3A.

The functions in this module deliberately operate on closed ``RecordEpisode``
HDF5/JSON pairs.  They do not depend on ManiSkill's example runners or merge
script, whose warning-only conflict handling and local episode identifiers are
not strong enough for an authoritative source archive.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import h5py
import numpy as np

from langmani.datasets.identity import (
    COLLECTION_SCHEMA_VERSION,
    stable_accepted_scene_group_id,
    stable_candidate_scene_id,
    stable_raw_trajectory_id,
    stable_source_shard_id,
)
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import (
    TaskSpec,
    canonical_instruction,
    stable_scene_id,
    stable_task_id,
)

ENVIRONMENT_ID = "LangMani-PickPlaceByInstruction-v0"
CONTROL_MODE = "pd_joint_pos"
ARCHIVE_SCHEMA_VERSION = COLLECTION_SCHEMA_VERSION
PANDA_ACTION_DIMENSION = 8
EPISODE_TIME_LIMIT = 200

_REQUIRED_TRAJECTORY_MEMBERS = frozenset(
    {"actions", "terminated", "truncated", "success", "fail", "env_states", "obs"}
)
_ALLOWED_TRAJECTORY_MEMBERS = _REQUIRED_TRAJECTORY_MEMBERS
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_EVALUATION = MappingProxyType(
    {
        "target_in_target_bin": True,
        "target_in_wrong_bin": False,
        "wrong_object_in_target_bin": False,
        "target_is_grasped": False,
        "target_is_static": True,
        "target_off_table": False,
        "success": True,
        "fail": False,
    }
)


class ArchiveValidationError(ValueError):
    """Raised when a native trajectory pair violates the M3A archive contract."""


@dataclass(frozen=True, slots=True)
class EpisodeSelection:
    """Select one local RecordEpisode episode and attach project metadata."""

    native_episode_id: int
    langmani_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if isinstance(self.native_episode_id, bool) or not isinstance(self.native_episode_id, int):
            raise TypeError("native_episode_id must be an integer")
        if self.native_episode_id < 0:
            raise ValueError("native_episode_id must be non-negative")
        normalized = _json_mapping(self.langmani_metadata, name="langmani_metadata")
        object.__setattr__(self, "langmani_metadata", MappingProxyType(normalized))


@dataclass(frozen=True, slots=True)
class EpisodeLocation:
    """Stable-to-native location mapping inside one immutable pair."""

    native_episode_id: int
    h5_group: str
    elapsed_steps: int
    scene_seed: int
    scene_id: str
    task_spec: TaskSpec
    task_id: str
    canonical_instruction: str
    scheduled_episode_id: str | None = None
    attempt_id: str | None = None
    raw_trajectory_id: str | None = None
    source_shard_id: str | None = None
    collection_run_id: str | None = None
    candidate_scene_id: str | None = None
    accepted_scene_group_id: str | None = None
    expert_config_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "native_episode_id": self.native_episode_id,
            "h5_group": self.h5_group,
            "elapsed_steps": self.elapsed_steps,
            "scene_seed": self.scene_seed,
            "scene_id": self.scene_id,
            "task_spec": self.task_spec.to_dict(),
            "task_id": self.task_id,
            "canonical_instruction": self.canonical_instruction,
            "scheduled_episode_id": self.scheduled_episode_id,
            "attempt_id": self.attempt_id,
            "raw_trajectory_id": self.raw_trajectory_id,
            "source_shard_id": self.source_shard_id,
            "collection_run_id": self.collection_run_id,
            "candidate_scene_id": self.candidate_scene_id,
            "accepted_scene_group_id": self.accepted_scene_group_id,
            "expert_config_fingerprint": self.expert_config_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class ArchivePairValidation:
    """Integrity result for one closed HDF5/JSON pair."""

    h5_path: str
    json_path: str
    h5_sha256: str
    json_sha256: str
    h5_size_bytes: int
    json_size_bytes: int
    episodes: tuple[EpisodeLocation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "h5_path": self.h5_path,
            "json_path": self.json_path,
            "h5_sha256": self.h5_sha256,
            "json_sha256": self.json_sha256,
            "h5_size_bytes": self.h5_size_bytes,
            "json_size_bytes": self.json_size_bytes,
            "episodes": [episode.to_dict() for episode in self.episodes],
        }


def sha256_file(path: str | Path) -> str:
    """Return a lowercase SHA-256 digest without reading the file all at once."""
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def paired_json_path(h5_path: str | Path) -> Path:
    """Return the required JSON sibling for a ``.h5`` native archive."""
    path = Path(h5_path)
    if path.suffix.lower() != ".h5":
        raise ValueError(f"native archive path must end in .h5: {path}")
    return path.with_suffix(".json")


def validate_native_archive_pair(
    h5_path: str | Path,
    *,
    json_path: str | Path | None = None,
    expected_env_id: str = ENVIRONMENT_ID,
    expected_control_mode: str = CONTROL_MODE,
    require_accepted_success: bool = False,
    require_langmani_metadata: bool = False,
    require_canonical_group: bool = False,
) -> ArchivePairValidation:
    """Validate a closed ManiSkill 3.0.1 ``RecordEpisode`` HDF5/JSON pair.

    ``require_accepted_success`` adds final success/failure/truncation checks.
    ``require_langmani_metadata`` additionally validates project provenance,
    ExpertResult and action-replay evidence.  A counterfactual group uses both
    flags plus ``require_canonical_group``.
    """
    h5_file_path = Path(h5_path).resolve()
    json_file_path = (
        Path(json_path).resolve() if json_path is not None else paired_json_path(h5_file_path)
    )
    _require_regular_file(h5_file_path, label="HDF5")
    _require_regular_file(json_file_path, label="JSON")

    metadata = _read_json_mapping(json_file_path)
    env_info = _required_mapping(metadata, "env_info", context="JSON root")
    env_id = env_info.get("env_id")
    if env_id != expected_env_id:
        raise ArchiveValidationError(f"env_info.env_id is {env_id!r}, expected {expected_env_id!r}")
    env_kwargs = _required_mapping(env_info, "env_kwargs", context="env_info")
    recorded_control_mode = env_kwargs.get("control_mode")
    if recorded_control_mode != expected_control_mode:
        raise ArchiveValidationError(
            "env_info.env_kwargs.control_mode is "
            f"{recorded_control_mode!r}, expected {expected_control_mode!r}"
        )
    num_envs = env_kwargs.get("num_envs")
    if num_envs != 1:
        raise ArchiveValidationError("M3A native archives require num_envs=1")
    if env_kwargs.get("obs_mode") != "none":
        raise ArchiveValidationError("M3A native archives require obs_mode='none'")
    if env_kwargs.get("reward_mode") != "none":
        raise ArchiveValidationError("M3A native archives require reward_mode='none'")
    if env_kwargs.get("sim_backend") not in {"physx_cpu", "physx_cuda"}:
        raise ArchiveValidationError(
            "M3A native archives require sim_backend='physx_cpu' or 'physx_cuda'"
        )
    if env_info.get("max_episode_steps") != EPISODE_TIME_LIMIT:
        raise ArchiveValidationError(f"env_info.max_episode_steps must be {EPISODE_TIME_LIMIT}")

    episodes_value = metadata.get("episodes")
    if not isinstance(episodes_value, list):
        raise ArchiveValidationError("JSON root.episodes must be a list")
    if not episodes_value:
        raise ArchiveValidationError("native archive must contain at least one episode")

    json_episodes: dict[int, Mapping[str, Any]] = {}
    for index, episode_value in enumerate(episodes_value):
        if not isinstance(episode_value, Mapping):
            raise ArchiveValidationError(f"episodes[{index}] must be a mapping")
        episode_id = _non_negative_int(
            episode_value.get("episode_id"), name=f"episodes[{index}].episode_id"
        )
        if episode_id in json_episodes:
            raise ArchiveValidationError(f"duplicate JSON episode_id {episode_id}")
        json_episodes[episode_id] = episode_value

    locations: list[EpisodeLocation] = []
    try:
        with h5py.File(h5_file_path, "r") as archive:
            h5_groups = set(archive.keys())
            expected_groups = {f"traj_{episode_id}" for episode_id in json_episodes}
            if h5_groups != expected_groups:
                missing = sorted(expected_groups - h5_groups)
                extra = sorted(h5_groups - expected_groups)
                raise ArchiveValidationError(
                    f"HDF5/JSON episode mismatch; missing={missing}, extra={extra}"
                )
            for name in sorted(h5_groups, key=_trajectory_group_sort_key):
                link = archive.get(name, getlink=True)
                if not isinstance(link, h5py.HardLink):
                    raise ArchiveValidationError(f"{name} must be a local HDF5 hard link")
                value = archive[name]
                if not isinstance(value, h5py.Group):
                    raise ArchiveValidationError(f"{name} must be an HDF5 group")
                episode_id = int(name.removeprefix("traj_"))
                episode = json_episodes[episode_id]
                locations.append(
                    _validate_episode(
                        value,
                        episode,
                        expected_control_mode=expected_control_mode,
                        require_accepted_success=require_accepted_success,
                        require_langmani_metadata=require_langmani_metadata,
                    )
                )
            if require_canonical_group:
                _validate_canonical_counterfactual_group(archive, locations, json_episodes)
            if require_langmani_metadata:
                _validate_stable_id_uniqueness(locations)
                _validate_archive_level_metadata(
                    archive,
                    metadata,
                    locations,
                    json_episodes,
                    require_canonical_group=require_canonical_group,
                )
    except OSError as exc:
        raise ArchiveValidationError(f"cannot read HDF5 archive {h5_file_path}: {exc}") from exc

    return ArchivePairValidation(
        h5_path=str(h5_file_path),
        json_path=str(json_file_path),
        h5_sha256=sha256_file(h5_file_path),
        json_sha256=sha256_file(json_file_path),
        h5_size_bytes=h5_file_path.stat().st_size,
        json_size_bytes=json_file_path.stat().st_size,
        episodes=tuple(locations),
    )


def create_counterfactual_scene_group_bundle(
    candidate_h5_path: str | Path,
    selections: Sequence[EpisodeSelection],
    output_h5_path: str | Path,
    *,
    group_metadata: Mapping[str, Any],
    candidate_json_path: str | Path | None = None,
) -> ArchivePairValidation:
    """Copy exactly six accepted episodes into an immutable scene-group pair.

    Selection order is semantic and must equal :data:`CANONICAL_TASK_SPECS`.
    The output is written through temporary files, re-opened and fully
    validated before it is atomically promoted.  Existing outputs are never
    overwritten.
    """
    if len(selections) != len(CANONICAL_TASK_SPECS):
        raise ArchiveValidationError(
            "a CounterfactualSceneGroup requires exactly six episode selections"
        )
    if not all(isinstance(selection, EpisodeSelection) for selection in selections):
        raise TypeError("selections must contain only EpisodeSelection values")
    source_h5 = Path(candidate_h5_path).resolve()
    source_json = (
        Path(candidate_json_path).resolve()
        if candidate_json_path is not None
        else paired_json_path(source_h5)
    )
    output_h5 = Path(output_h5_path).resolve()
    output_json = paired_json_path(output_h5)
    _ensure_new_pair(output_h5, output_json)
    if output_h5 == source_h5 or output_json == source_json:
        raise ArchiveValidationError("scene-group output must differ from its candidate pair")

    source_validation = validate_native_archive_pair(source_h5, json_path=source_json)
    source_metadata = _read_json_mapping(source_json)
    source_episodes = {
        _non_negative_int(ep.get("episode_id"), name="episode_id"): ep
        for ep in source_metadata["episodes"]
    }
    selected_ids = [selection.native_episode_id for selection in selections]
    if len(set(selected_ids)) != len(selected_ids):
        raise ArchiveValidationError("episode selections must not repeat a native episode")

    normalized_group_metadata = _json_mapping(group_metadata, name="group_metadata")
    group_id = normalized_group_metadata.get(
        "accepted_scene_group_id", normalized_group_metadata.get("scene_group_id")
    )
    _non_empty_string(group_id, name="group_metadata.accepted_scene_group_id")

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    temp_h5, temp_json = _temporary_pair_paths(output_h5)
    try:
        copied_episodes: list[dict[str, Any]] = []
        source_h5_sha = source_validation.h5_sha256
        source_json_sha = source_validation.json_sha256
        with h5py.File(source_h5, "r") as source, h5py.File(temp_h5, "w") as destination:
            for local_id, (selection, expected_task) in enumerate(
                zip(selections, CANONICAL_TASK_SPECS, strict=True)
            ):
                native_id = selection.native_episode_id
                source_group = f"traj_{native_id}"
                if native_id not in source_episodes or source_group not in source:
                    raise ArchiveValidationError(
                        f"selected native episode {native_id} is absent from candidate pair"
                    )
                episode = source_episodes[native_id]
                _validate_selected_episode_for_bundle(
                    episode,
                    source[source_group],
                    selection.langmani_metadata,
                    expected_task=expected_task,
                )
                destination_group = f"traj_{local_id}"
                source.copy(source_group, destination, name=destination_group)
                copied_episode = json.loads(json.dumps(episode))
                copied_episode["episode_id"] = local_id
                langmani = dict(selection.langmani_metadata)
                langmani["archive_schema_version"] = ARCHIVE_SCHEMA_VERSION
                langmani["source_candidate"] = {
                    "h5_sha256": source_h5_sha,
                    "json_sha256": source_json_sha,
                    "native_episode_id": native_id,
                    "h5_group": source_group,
                }
                copied_episode["langmani"] = langmani
                copied_episodes.append(copied_episode)
            destination.flush()
        _require_unchanged_pair(
            source_h5,
            source_json,
            expected_h5_sha256=source_h5_sha,
            expected_json_sha256=source_json_sha,
        )

        output_metadata = {
            key: json.loads(json.dumps(value))
            for key, value in source_metadata.items()
            if key not in {"episodes", "langmani"}
        }
        output_metadata["source_type"] = "motionplanning"
        output_metadata["source_desc"] = (
            "LangMani M3A accepted deterministic counterfactual scene group"
        )
        output_metadata["episodes"] = copied_episodes
        output_metadata["langmani"] = {
            **normalized_group_metadata,
            "archive_schema_version": ARCHIVE_SCHEMA_VERSION,
            "archive_kind": "counterfactual_scene_group",
            "immutable": True,
        }
        _write_json(temp_json, output_metadata)
        validate_native_archive_pair(
            temp_h5,
            json_path=temp_json,
            require_accepted_success=True,
            require_langmani_metadata=True,
            require_canonical_group=True,
        )
        _promote_pair(temp_h5, temp_json, output_h5, output_json)
    except Exception:
        _remove_if_exists(temp_h5)
        _remove_if_exists(temp_json)
        raise
    return validate_native_archive_pair(
        output_h5,
        require_accepted_success=True,
        require_langmani_metadata=True,
        require_canonical_group=True,
    )


def build_source_shard(
    group_bundle_h5_paths: Sequence[str | Path],
    output_h5_path: str | Path,
    *,
    shard_metadata: Mapping[str, Any],
) -> ArchivePairValidation:
    """Build one immutable source shard from complete accepted group bundles."""
    if not group_bundle_h5_paths:
        raise ArchiveValidationError("a source shard requires at least one scene-group bundle")
    output_h5 = Path(output_h5_path).resolve()
    output_json = paired_json_path(output_h5)
    _ensure_new_pair(output_h5, output_json)
    normalized_shard_metadata = _json_mapping(shard_metadata, name="shard_metadata")
    shard_id = _non_empty_string(
        normalized_shard_metadata.get("source_shard_id"),
        name="shard_metadata.source_shard_id",
    )

    bundle_entries: list[tuple[int, str, Path, Path, ArchivePairValidation, dict[str, Any]]] = []
    seen_group_ids: set[str] = set()
    for raw_path in group_bundle_h5_paths:
        bundle_h5 = Path(raw_path).resolve()
        if bundle_h5 == output_h5:
            raise ArchiveValidationError("source shard cannot use itself as an input bundle")
        bundle_json = paired_json_path(bundle_h5)
        validation = validate_native_archive_pair(
            bundle_h5,
            json_path=bundle_json,
            require_accepted_success=True,
            require_langmani_metadata=True,
            require_canonical_group=True,
        )
        metadata = _read_json_mapping(bundle_json)
        group_meta = _required_mapping(metadata, "langmani", context="bundle JSON root")
        if group_meta.get("archive_kind") != "counterfactual_scene_group":
            raise ArchiveValidationError(f"{bundle_h5} is not a scene-group bundle")
        group_id = _non_empty_string(
            group_meta.get("accepted_scene_group_id", group_meta.get("scene_group_id")),
            name="bundle group id",
        )
        if group_id in seen_group_ids:
            raise ArchiveValidationError(f"duplicate scene-group bundle {group_id!r}")
        seen_group_ids.add(group_id)
        scene_seed = validation.episodes[0].scene_seed
        bundle_entries.append((scene_seed, group_id, bundle_h5, bundle_json, validation, metadata))

    # Sorting makes the shard byte layout independent of filesystem enumeration.
    bundle_entries.sort(key=lambda entry: (entry[0], entry[1]))
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    temp_h5, temp_json = _temporary_pair_paths(output_h5)
    try:
        output_episodes: list[dict[str, Any]] = []
        group_ids: list[str] = []
        reference_env_info: Any = None
        local_id = 0
        with h5py.File(temp_h5, "w") as destination:
            for _, group_id, bundle_h5, bundle_json, validation, metadata in bundle_entries:
                env_info = metadata.get("env_info")
                if reference_env_info is None:
                    reference_env_info = env_info
                elif env_info != reference_env_info:
                    raise ArchiveValidationError(
                        f"env_info conflict while adding scene group {group_id!r}"
                    )
                group_ids.append(group_id)
                episodes_by_id = {
                    _non_negative_int(ep.get("episode_id"), name="episode_id"): ep
                    for ep in metadata["episodes"]
                }
                with h5py.File(bundle_h5, "r") as source:
                    for source_location in validation.episodes:
                        source_id = source_location.native_episode_id
                        destination_name = f"traj_{local_id}"
                        source.copy(source_location.h5_group, destination, name=destination_name)
                        episode = json.loads(json.dumps(episodes_by_id[source_id]))
                        episode["episode_id"] = local_id
                        langmani = _required_mapping(
                            episode, "langmani", context=f"bundle {group_id} episode"
                        )
                        episode["langmani"] = {
                            **langmani,
                            "archive_schema_version": ARCHIVE_SCHEMA_VERSION,
                            "source_shard_id": shard_id,
                            "source_group_bundle": {
                                "accepted_scene_group_id": group_id,
                                "h5_sha256": validation.h5_sha256,
                                "json_sha256": validation.json_sha256,
                                "native_episode_id": source_id,
                                "h5_group": source_location.h5_group,
                            },
                        }
                        output_episodes.append(episode)
                        local_id += 1
                _require_unchanged_pair(
                    bundle_h5,
                    bundle_json,
                    expected_h5_sha256=validation.h5_sha256,
                    expected_json_sha256=validation.json_sha256,
                )
            destination.flush()

        first_metadata = bundle_entries[0][5]
        output_metadata = {
            key: json.loads(json.dumps(value))
            for key, value in first_metadata.items()
            if key not in {"episodes", "langmani"}
        }
        output_metadata["source_type"] = "motionplanning"
        output_metadata["source_desc"] = (
            "LangMani M3A immutable ManiSkill-native accepted source shard"
        )
        output_metadata["episodes"] = output_episodes
        output_metadata["langmani"] = {
            **normalized_shard_metadata,
            "archive_schema_version": ARCHIVE_SCHEMA_VERSION,
            "archive_kind": "source_shard",
            "immutable": True,
            "source_shard_id": shard_id,
            "episode_count": len(output_episodes),
            "scene_group_count": len(group_ids),
            "accepted_scene_group_ids": group_ids,
        }
        _write_json(temp_json, output_metadata)
        validate_native_archive_pair(
            temp_h5,
            json_path=temp_json,
            require_accepted_success=True,
            require_langmani_metadata=True,
        )
        _promote_pair(temp_h5, temp_json, output_h5, output_json)
    except Exception:
        _remove_if_exists(temp_h5)
        _remove_if_exists(temp_json)
        raise
    return validate_native_archive_pair(
        output_h5,
        require_accepted_success=True,
        require_langmani_metadata=True,
    )


def _validate_episode(
    group: h5py.Group,
    episode: Mapping[str, Any],
    *,
    expected_control_mode: str,
    require_accepted_success: bool,
    require_langmani_metadata: bool,
) -> EpisodeLocation:
    context = group.name.lstrip("/")
    members = set(group.keys())
    missing = sorted(_REQUIRED_TRAJECTORY_MEMBERS - members)
    extra = sorted(members - _ALLOWED_TRAJECTORY_MEMBERS)
    if missing or extra:
        raise ArchiveValidationError(
            f"{context} trajectory members invalid; missing={missing}, extra={extra}"
        )
    for key in members:
        link = group.get(key, getlink=True)
        if not isinstance(link, h5py.HardLink):
            raise ArchiveValidationError(f"{context}/{key} must be a local HDF5 hard link")
    actions = _required_dataset(group, "actions", context=context)
    if actions.dtype != np.dtype(np.float32):
        raise ArchiveValidationError(f"{context}/actions must have dtype float32")
    if actions.ndim != 2 or actions.shape[1] != PANDA_ACTION_DIMENSION:
        raise ArchiveValidationError(
            f"{context}/actions must have shape [T, {PANDA_ACTION_DIMENSION}]"
        )
    transition_count = actions.shape[0]
    if transition_count <= 0:
        raise ArchiveValidationError(f"{context}/actions must contain at least one transition")
    _require_finite(actions, context=f"{context}/actions")

    for key in ("terminated", "truncated", "success", "fail"):
        dataset = _required_dataset(group, key, context=context)
        if dataset.dtype != np.dtype(bool):
            raise ArchiveValidationError(f"{context}/{key} must have dtype bool")
        if dataset.shape != (transition_count,):
            raise ArchiveValidationError(f"{context}/{key} must have shape ({transition_count},)")
    terminated_values = np.asarray(group["terminated"][:], dtype=bool)
    success_values = np.asarray(group["success"][:], dtype=bool)
    fail_values = np.asarray(group["fail"][:], dtype=bool)
    if not np.array_equal(terminated_values, success_values | fail_values):
        raise ArchiveValidationError(
            f"{context}/terminated must equal success OR fail at every transition"
        )
    env_states = group["env_states"]
    if not isinstance(env_states, h5py.Group):
        raise ArchiveValidationError(f"{context}/env_states must be a recursive group")
    state_leaves = _validate_recursive_leaves(
        env_states,
        expected_length=transition_count + 1,
        allow_empty=False,
        context=f"{context}/env_states",
    )
    if not state_leaves:
        raise ArchiveValidationError(f"{context}/env_states must contain numeric leaves")
    obs = group["obs"]
    if not isinstance(obs, h5py.Group) or len(obs) != 0:
        raise ArchiveValidationError(
            f"{context}/obs must be an empty group for the fixed obs_mode='none' contract"
        )

    elapsed_steps = _non_negative_int(episode.get("elapsed_steps"), name=f"{context}.elapsed_steps")
    if elapsed_steps != transition_count:
        raise ArchiveValidationError(
            f"{context} elapsed_steps={elapsed_steps} but actions has T={transition_count}"
        )
    if episode.get("control_mode") != expected_control_mode:
        raise ArchiveValidationError(f"{context} control_mode must be {expected_control_mode!r}")
    scene_seed, task_spec = _parse_reset_contract(episode, context=context)
    scene_id = stable_scene_id(scene_seed)
    task_id = stable_task_id(task_spec)

    if require_accepted_success:
        if episode.get("success") is not True or episode.get("fail") is not False:
            raise ArchiveValidationError(
                f"{context} JSON labels must end with success=true and fail=false"
            )
        if bool(group["success"][-1]) is not True or bool(group["fail"][-1]) is not False:
            raise ArchiveValidationError(
                f"{context} HDF5 labels must end with success=true and fail=false"
            )
        if bool(group["truncated"][-1]):
            raise ArchiveValidationError(f"{context} accepted trajectory must not end truncated")
        if not bool(group["terminated"][-1]):
            raise ArchiveValidationError(f"{context} accepted trajectory must end terminated")

    langmani = episode.get("langmani")
    if require_langmani_metadata:
        if not isinstance(langmani, Mapping):
            raise ArchiveValidationError(f"{context} is missing langmani episode metadata")
        _validate_langmani_metadata(
            langmani,
            scene_seed=scene_seed,
            scene_id=scene_id,
            task_spec=task_spec,
            task_id=task_id,
            transition_count=transition_count,
            context=context,
        )
        _validate_initial_scene_against_env_state(
            langmani.get("initial_scene_state"),
            env_states,
            context=context,
        )

    return EpisodeLocation(
        native_episode_id=int(context.removeprefix("traj_")),
        h5_group=context,
        elapsed_steps=elapsed_steps,
        scene_seed=scene_seed,
        scene_id=scene_id,
        task_spec=task_spec,
        task_id=task_id,
        canonical_instruction=canonical_instruction(task_spec),
        scheduled_episode_id=_optional_metadata_string(langmani, "scheduled_episode_id"),
        attempt_id=_optional_attempt_id(langmani),
        raw_trajectory_id=_optional_metadata_string(langmani, "raw_trajectory_id"),
        source_shard_id=_optional_metadata_string(langmani, "source_shard_id"),
        collection_run_id=_optional_metadata_string(langmani, "collection_run_id"),
        candidate_scene_id=_optional_metadata_string(langmani, "candidate_scene_id"),
        accepted_scene_group_id=_optional_metadata_string(langmani, "accepted_scene_group_id"),
        expert_config_fingerprint=_optional_metadata_string(langmani, "expert_config_fingerprint"),
    )


def _validate_selected_episode_for_bundle(
    episode: Mapping[str, Any],
    group: h5py.Group,
    langmani: Mapping[str, Any],
    *,
    expected_task: TaskSpec,
) -> None:
    context = group.name.lstrip("/")
    scene_seed, recorded_task = _parse_reset_contract(episode, context=context)
    if recorded_task != expected_task:
        raise ArchiveValidationError(
            f"{context} has task {recorded_task.to_dict()}, expected {expected_task.to_dict()}"
        )
    if episode.get("success") is not True or episode.get("fail") is not False:
        raise ArchiveValidationError(f"{context} is not a successful candidate episode")
    if bool(group["success"][-1]) is not True or bool(group["fail"][-1]) is not False:
        raise ArchiveValidationError(f"{context} has non-success terminal HDF5 labels")
    if bool(group["truncated"][-1]):
        raise ArchiveValidationError(f"{context} ended by truncation")
    _validate_langmani_metadata(
        langmani,
        scene_seed=scene_seed,
        scene_id=stable_scene_id(scene_seed),
        task_spec=recorded_task,
        task_id=stable_task_id(recorded_task),
        transition_count=group["actions"].shape[0],
        context=context,
    )


def _validate_langmani_metadata(
    metadata: Mapping[str, Any],
    *,
    scene_seed: int,
    scene_id: str,
    task_spec: TaskSpec,
    task_id: str,
    transition_count: int,
    context: str,
) -> None:
    for field in (
        "scheduled_episode_id",
        "raw_trajectory_id",
        "collection_run_id",
        "candidate_scene_id",
        "accepted_scene_group_id",
        "expert_config_fingerprint",
    ):
        _non_empty_string(metadata.get(field), name=f"{context}.langmani.{field}")
    attempt_value = metadata.get("attempt_id", metadata.get("expert_attempt_id"))
    _non_empty_string(attempt_value, name=f"{context}.langmani.attempt_id")
    if metadata.get("collection_schema_version") != COLLECTION_SCHEMA_VERSION:
        raise ArchiveValidationError(f"{context} collection schema version is unsupported")
    expected_raw_id = stable_raw_trajectory_id(
        scheduled_episode_id=str(metadata["scheduled_episode_id"]),
        attempt_id=str(attempt_value),
    )
    if metadata.get("raw_trajectory_id") != expected_raw_id:
        raise ArchiveValidationError(f"{context} raw trajectory identity is inconsistent")
    if metadata.get("scene_seed") != scene_seed:
        raise ArchiveValidationError(f"{context} langmani scene_seed disagrees with reset")
    if metadata.get("scene_id") != scene_id:
        raise ArchiveValidationError(f"{context} langmani scene_id disagrees with reset")
    metadata_task_value = metadata.get("task_spec")
    if not isinstance(metadata_task_value, Mapping):
        raise ArchiveValidationError(f"{context} langmani task_spec must be a mapping")
    try:
        metadata_task = TaskSpec.from_mapping(metadata_task_value)
    except (TypeError, ValueError) as exc:
        raise ArchiveValidationError(f"{context} invalid langmani task_spec: {exc}") from exc
    if metadata_task != task_spec or metadata.get("task_id") != task_id:
        raise ArchiveValidationError(f"{context} langmani task identity disagrees with reset")
    if metadata.get("canonical_instruction") != canonical_instruction(task_spec):
        raise ArchiveValidationError(f"{context} canonical instruction disagrees with task")

    evaluation = metadata.get("final_environment_evaluation")
    if evaluation is None:
        expert = metadata.get("expert_result")
        if isinstance(expert, Mapping):
            evaluation = expert.get("final_environment_evaluation")
    _validate_success_evaluation(evaluation, context=f"{context}.langmani")

    expert_result = metadata.get("expert_result")
    if not isinstance(expert_result, Mapping):
        raise ArchiveValidationError(f"{context} langmani expert_result must be a mapping")
    if expert_result.get("success") is not True or expert_result.get("status") != "success":
        raise ArchiveValidationError(f"{context} expert_result is not successful")
    if expert_result.get("scene_seed") != scene_seed:
        raise ArchiveValidationError(f"{context} expert_result scene_seed disagrees")
    if expert_result.get("scene_id") != scene_id or expert_result.get("task_id") != task_id:
        raise ArchiveValidationError(f"{context} expert_result identity disagrees")
    if expert_result.get("target_object_id") != task_spec.target_object_id:
        raise ArchiveValidationError(f"{context} expert_result target object disagrees")
    if expert_result.get("target_bin_id") != task_spec.target_bin_id:
        raise ArchiveValidationError(f"{context} expert_result target bin disagrees")
    if expert_result.get("total_environment_steps") != transition_count:
        raise ArchiveValidationError(
            f"{context} expert_result total_environment_steps disagrees with HDF5 actions"
        )
    _validate_success_evaluation(
        expert_result.get("final_environment_evaluation"),
        context=f"{context}.expert_result",
    )

    replay = metadata.get("replay_validation")
    if not isinstance(replay, Mapping):
        raise ArchiveValidationError(f"{context} replay_validation must be a mapping")
    replay_verdicts = [
        replay.get(name) for name in ("passed", "success", "valid") if name in replay
    ]
    if not replay_verdicts or any(verdict is not True for verdict in replay_verdicts):
        raise ArchiveValidationError(f"{context} replay_validation did not pass")
    replay_requirements = {
        "recorded_success": True,
        "replay_success": True,
        "task_spec_matches": True,
        "wrong_object_in_target_bin": False,
        "target_in_wrong_bin": False,
        "target_off_table": False,
    }
    for key, expected in replay_requirements.items():
        if replay.get(key) is not expected:
            raise ArchiveValidationError(f"{context} replay_validation {key} must be {expected!r}")
    if replay.get("recorded_action_steps") != transition_count:
        raise ArchiveValidationError(
            f"{context} replay_validation recorded_action_steps disagrees with HDF5 actions"
        )
    if replay.get("replayed_action_steps") != transition_count:
        raise ArchiveValidationError(
            f"{context} replay_validation replayed_action_steps disagrees with HDF5 actions"
        )
    failure_reasons = replay.get("failure_reasons")
    if not isinstance(failure_reasons, list) or failure_reasons:
        raise ArchiveValidationError(
            f"{context} passed replay_validation must have empty failure_reasons"
        )
    failure_codes = replay.get("failure_codes")
    if not isinstance(failure_codes, list) or failure_codes:
        raise ArchiveValidationError(
            f"{context} passed replay_validation must have empty failure_codes"
        )
    replay_evaluation = replay.get("final_environment_evaluation")
    if replay_evaluation is not None:
        _validate_success_evaluation(replay_evaluation, context=f"{context}.replay_validation")
    _validated_initial_scene_state(
        metadata.get("initial_scene_state"),
        context=f"{context}.langmani.initial_scene_state",
    )
    if metadata.get("structural_valid") is not True:
        raise ArchiveValidationError(f"{context} structural_valid must be true")
    if metadata.get("accepted") is not True:
        raise ArchiveValidationError(f"{context} accepted must be true")
    archive_schema = metadata.get("archive_schema_version")
    if archive_schema is not None and archive_schema != ARCHIVE_SCHEMA_VERSION:
        raise ArchiveValidationError(f"{context} archive schema version is unsupported")


def _validate_stable_id_uniqueness(locations: Sequence[EpisodeLocation]) -> None:
    for field_name in ("scheduled_episode_id", "attempt_id", "raw_trajectory_id"):
        values = [getattr(location, field_name) for location in locations]
        if any(value is None for value in values):
            raise ArchiveValidationError(f"every episode must have {field_name}")
        if len(set(values)) != len(values):
            raise ArchiveValidationError(f"duplicate stable episode metadata field {field_name}")


def _validate_archive_level_metadata(
    archive: h5py.File,
    metadata: Mapping[str, Any],
    locations: Sequence[EpisodeLocation],
    json_episodes: Mapping[int, Mapping[str, Any]],
    *,
    require_canonical_group: bool,
) -> None:
    root = _required_mapping(metadata, "langmani", context="JSON root")
    if root.get("archive_schema_version") != ARCHIVE_SCHEMA_VERSION:
        raise ArchiveValidationError("JSON root langmani archive_schema_version is unsupported")
    if root.get("collection_schema_version") != COLLECTION_SCHEMA_VERSION:
        raise ArchiveValidationError("JSON root langmani collection_schema_version is unsupported")
    if root.get("immutable") is not True:
        raise ArchiveValidationError("accepted archive metadata must declare immutable=true")
    archive_kind = root.get("archive_kind")
    if require_canonical_group:
        if archive_kind != "counterfactual_scene_group":
            raise ArchiveValidationError("canonical group archive_kind is invalid")
        group_id = _non_empty_string(
            root.get("accepted_scene_group_id", root.get("scene_group_id")),
            name="accepted scene group ID",
        )
        scene_seed = locations[0].scene_seed
        scene_id = locations[0].scene_id
        if root.get("scene_seed") != scene_seed or root.get("scene_id") != scene_id:
            raise ArchiveValidationError("scene-group root identity disagrees with its episodes")
        collection_run_id = _non_empty_string(
            root.get("collection_run_id"), name="collection_run_id"
        )
        candidate_scene_id = _non_empty_string(
            root.get("candidate_scene_id"), name="candidate_scene_id"
        )
        expert_fingerprint = _non_empty_string(
            root.get("expert_config_fingerprint"), name="expert_config_fingerprint"
        )
        expected_candidate = stable_candidate_scene_id(
            environment_id=ENVIRONMENT_ID,
            scene_seed=scene_seed,
            scene_id=scene_id,
            expert_fingerprint=expert_fingerprint,
            control_mode=CONTROL_MODE,
            schema_version=COLLECTION_SCHEMA_VERSION,
        )
        raw_ids = tuple(
            _non_empty_string(location.raw_trajectory_id, name="raw_trajectory_id")
            for location in locations
        )
        if candidate_scene_id != expected_candidate or group_id != stable_accepted_scene_group_id(
            candidate_scene_id=candidate_scene_id,
            raw_trajectory_ids=raw_ids,
        ):
            raise ArchiveValidationError("scene-group stable identity is inconsistent")
        if any(
            location.collection_run_id != collection_run_id
            or location.candidate_scene_id != candidate_scene_id
            or location.accepted_scene_group_id != group_id
            or location.expert_config_fingerprint != expert_fingerprint
            for location in locations
        ):
            raise ArchiveValidationError("scene-group episode identity disagrees with its root")
        return
    if archive_kind != "source_shard":
        raise ArchiveValidationError("accepted multi-episode pair must be a source_shard")
    shard_id = _non_empty_string(root.get("source_shard_id"), name="source shard ID")
    collection_run_id = _non_empty_string(root.get("collection_run_id"), name="collection_run_id")
    expert_fingerprint = _non_empty_string(
        root.get("expert_config_fingerprint"), name="expert_config_fingerprint"
    )
    shard_index = _non_negative_int(root.get("shard_index"), name="shard_index")
    if shard_id != stable_source_shard_id(
        collection_run_id=collection_run_id,
        shard_index=shard_index,
    ):
        raise ArchiveValidationError("source shard stable identity is inconsistent")
    if root.get("episode_count") != len(locations):
        raise ArchiveValidationError("source shard episode_count is inconsistent")
    group_ids_value = root.get("accepted_scene_group_ids")
    if not isinstance(group_ids_value, list) or not group_ids_value:
        raise ArchiveValidationError("source shard accepted_scene_group_ids must be a list")
    group_ids = tuple(
        _non_empty_string(value, name="accepted_scene_group_ids item") for value in group_ids_value
    )
    if len(set(group_ids)) != len(group_ids):
        raise ArchiveValidationError("source shard group IDs must be unique")
    if root.get("scene_group_count") != len(group_ids):
        raise ArchiveValidationError("source shard scene_group_count is inconsistent")
    grouped: dict[str, list[tuple[EpisodeLocation, Mapping[str, Any]]]] = {
        group_id: [] for group_id in group_ids
    }
    for location in locations:
        episode = json_episodes[location.native_episode_id]
        episode_meta = _required_mapping(episode, "langmani", context=location.h5_group)
        if episode_meta.get("source_shard_id") != shard_id:
            raise ArchiveValidationError(
                f"{location.h5_group} source_shard_id disagrees with shard root"
            )
        if (
            location.collection_run_id != collection_run_id
            or location.expert_config_fingerprint != expert_fingerprint
        ):
            raise ArchiveValidationError(
                f"{location.h5_group} collection identity disagrees with shard root"
            )
        source_group = _required_mapping(
            episode_meta, "source_group_bundle", context=location.h5_group
        )
        group_id = source_group.get("accepted_scene_group_id")
        if group_id not in grouped:
            raise ArchiveValidationError(
                f"{location.h5_group} references an unknown accepted scene group"
            )
        grouped[group_id].append((location, source_group))
    for group_id in group_ids:
        entries = grouped[group_id]
        if len(entries) != len(CANONICAL_TASK_SPECS):
            raise ArchiveValidationError(
                f"source shard scene group {group_id!r} does not contain exactly six episodes"
            )
        source_ids = [
            _non_negative_int(source.get("native_episode_id"), name="source native_episode_id")
            for _, source in entries
        ]
        if source_ids != list(range(len(CANONICAL_TASK_SPECS))):
            raise ArchiveValidationError(
                f"source shard scene group {group_id!r} is not in canonical task order"
            )
        scene_seeds = {location.scene_seed for location, _ in entries}
        scene_ids = {location.scene_id for location, _ in entries}
        if len(scene_seeds) != 1 or len(scene_ids) != 1:
            raise ArchiveValidationError(
                f"source shard scene group {group_id!r} mixes physical scenes"
            )
        candidate_ids = {location.candidate_scene_id for location, _ in entries}
        accepted_group_ids = {location.accepted_scene_group_id for location, _ in entries}
        if len(candidate_ids) != 1 or None in candidate_ids or accepted_group_ids != {group_id}:
            raise ArchiveValidationError(
                f"source shard scene group {group_id!r} has inconsistent semantic identity"
            )
        candidate_scene_id = next(iter(candidate_ids))
        assert candidate_scene_id is not None
        scene_seed = next(iter(scene_seeds))
        scene_id = next(iter(scene_ids))
        if candidate_scene_id != stable_candidate_scene_id(
            environment_id=ENVIRONMENT_ID,
            scene_seed=scene_seed,
            scene_id=scene_id,
            expert_fingerprint=expert_fingerprint,
            control_mode=CONTROL_MODE,
            schema_version=COLLECTION_SCHEMA_VERSION,
        ):
            raise ArchiveValidationError(
                f"source shard scene group {group_id!r} candidate identity is inconsistent"
            )
        raw_ids = tuple(
            _non_empty_string(location.raw_trajectory_id, name="raw_trajectory_id")
            for location, _ in entries
        )
        if group_id != stable_accepted_scene_group_id(
            candidate_scene_id=candidate_scene_id,
            raw_trajectory_ids=raw_ids,
        ):
            raise ArchiveValidationError(
                f"source shard scene group {group_id!r} stable identity is inconsistent"
            )
        for (location, source), expected_task, source_id in zip(
            entries, CANONICAL_TASK_SPECS, source_ids, strict=True
        ):
            if location.task_spec != expected_task:
                raise ArchiveValidationError(
                    f"source shard scene group {group_id!r} has a non-canonical task"
                )
            if source.get("h5_group") != f"traj_{source_id}":
                raise ArchiveValidationError(
                    f"source shard scene group {group_id!r} has inconsistent source h5_group"
                )
        provenance_hashes: dict[str, set[str]] = {
            "h5_sha256": set(),
            "json_sha256": set(),
        }
        for _, source in entries:
            for field_name in provenance_hashes:
                digest = source.get(field_name)
                if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
                    raise ArchiveValidationError(
                        f"source shard scene group {group_id!r} has invalid {field_name}"
                    )
                provenance_hashes[field_name].add(digest)
        if any(len(values) != 1 for values in provenance_hashes.values()):
            raise ArchiveValidationError(
                f"source shard scene group {group_id!r} mixes bundle provenance hashes"
            )
        fingerprints = [
            _initial_state_fingerprint(archive[location.h5_group]["env_states"])
            for location, _ in entries
        ]
        explicit_fingerprints = [
            _initial_scene_metadata_fingerprint(
                json_episodes[location.native_episode_id],
                context=location.h5_group,
            )
            for location, _ in entries
        ]
        if any(fingerprint != fingerprints[0] for fingerprint in fingerprints[1:]) or any(
            fingerprint != explicit_fingerprints[0] for fingerprint in explicit_fingerprints[1:]
        ):
            raise ArchiveValidationError(
                f"source shard scene group {group_id!r} changed its initial physical state"
            )


def _validate_success_evaluation(value: object, *, context: str) -> None:
    if not isinstance(value, Mapping):
        raise ArchiveValidationError(f"{context} final_environment_evaluation must be a mapping")
    for key, expected in _REQUIRED_EVALUATION.items():
        if value.get(key) is not expected:
            raise ArchiveValidationError(
                f"{context} evaluation {key} must be {expected!r}, got {value.get(key)!r}"
            )


def _validate_canonical_counterfactual_group(
    archive: h5py.File,
    locations: Sequence[EpisodeLocation],
    json_episodes: Mapping[int, Mapping[str, Any]],
) -> None:
    if len(locations) != len(CANONICAL_TASK_SPECS):
        raise ArchiveValidationError("counterfactual group must contain exactly six episodes")
    expected_ids = tuple(range(len(CANONICAL_TASK_SPECS)))
    if tuple(location.native_episode_id for location in locations) != expected_ids:
        raise ArchiveValidationError(
            "counterfactual group native IDs must be contiguous traj_0..traj_5"
        )
    scene_seeds = {location.scene_seed for location in locations}
    scene_ids = {location.scene_id for location in locations}
    if len(scene_seeds) != 1 or len(scene_ids) != 1:
        raise ArchiveValidationError("all counterfactual episodes must share one scene seed and ID")
    for location, expected_task in zip(locations, CANONICAL_TASK_SPECS, strict=True):
        _, task_spec = _parse_reset_contract(
            json_episodes[location.native_episode_id], context=location.h5_group
        )
        if task_spec != expected_task:
            raise ArchiveValidationError(
                "counterfactual tasks must follow canonical object-major/bin-minor order"
            )
    fingerprints = [
        _initial_state_fingerprint(archive[location.h5_group]["env_states"])
        for location in locations
    ]
    explicit_fingerprints = [
        _initial_scene_metadata_fingerprint(
            json_episodes[location.native_episode_id],
            context=location.h5_group,
        )
        for location in locations
    ]
    if any(fingerprint != fingerprints[0] for fingerprint in fingerprints[1:]) or any(
        fingerprint != explicit_fingerprints[0] for fingerprint in explicit_fingerprints[1:]
    ):
        raise ArchiveValidationError(
            "counterfactual episodes do not share an identical initial physical state"
        )


def _parse_reset_contract(episode: Mapping[str, Any], *, context: str) -> tuple[int, TaskSpec]:
    reset_kwargs = _required_mapping(episode, "reset_kwargs", context=context)
    reset_seed = _non_negative_int(reset_kwargs.get("seed"), name=f"{context}.reset_kwargs.seed")
    episode_seed = _non_negative_int(episode.get("episode_seed"), name=f"{context}.episode_seed")
    if episode_seed != reset_seed:
        raise ArchiveValidationError(
            f"{context} episode_seed={episode_seed} disagrees with reset seed={reset_seed}"
        )
    options = _required_mapping(reset_kwargs, "options", context=f"{context}.reset_kwargs")
    task_value = options.get("task_spec")
    if not isinstance(task_value, Mapping):
        raise ArchiveValidationError(f"{context} reset options.task_spec must be a mapping")
    try:
        task_spec = TaskSpec.from_mapping(task_value)
    except (TypeError, ValueError) as exc:
        raise ArchiveValidationError(f"{context} invalid reset task_spec: {exc}") from exc
    return reset_seed, task_spec


def _validate_recursive_leaves(
    group: h5py.Group,
    *,
    expected_length: int,
    allow_empty: bool,
    context: str,
) -> list[str]:
    leaves: list[str] = []
    for key in sorted(group.keys()):
        link = group.get(key, getlink=True)
        if not isinstance(link, h5py.HardLink):
            raise ArchiveValidationError(f"{context}/{key} must be a local HDF5 hard link")
        value = group[key]
        child_context = f"{context}/{key}"
        if isinstance(value, h5py.Group):
            leaves.extend(
                _validate_recursive_leaves(
                    value,
                    expected_length=expected_length,
                    allow_empty=False,
                    context=child_context,
                )
            )
        elif isinstance(value, h5py.Dataset):
            _validate_leaf(value, expected_length, context=child_context)
            leaves.append(child_context)
        else:
            raise ArchiveValidationError(f"{child_context} has an unsupported HDF5 type")
    if not allow_empty and not leaves:
        raise ArchiveValidationError(f"{context} must contain at least one dataset")
    return leaves


def _validate_leaf(dataset: h5py.Dataset, expected_length: int, *, context: str) -> None:
    if dataset.ndim < 1 or dataset.shape[0] != expected_length:
        raise ArchiveValidationError(
            f"{context} first dimension must be {expected_length}, got {dataset.shape}"
        )
    if dataset.dtype.kind not in "biuf":
        raise ArchiveValidationError(f"{context} must use a bool or numeric dtype")
    _require_finite(dataset, context=context)


def _require_finite(dataset: h5py.Dataset, *, context: str) -> None:
    for start in range(0, dataset.shape[0], 1024):
        values = dataset[start : start + 1024]
        if not np.all(np.isfinite(values)):
            raise ArchiveValidationError(f"{context} contains NaN or infinity")


def _validated_initial_scene_state(value: object, *, context: str) -> dict[str, Any]:
    """Validate the explicit physical reset snapshot, including static bins."""
    if not isinstance(value, Mapping) or set(value) != {
        "object_poses",
        "bin_poses",
        "panda_qpos",
    }:
        raise ArchiveValidationError(
            f"{context} must contain object_poses, bin_poses, and panda_qpos"
        )

    def pose_mapping(
        payload: object,
        *,
        keys: tuple[str, ...],
        field_name: str,
    ) -> dict[str, list[float]]:
        if not isinstance(payload, Mapping) or set(payload) != set(keys):
            raise ArchiveValidationError(
                f"{context}.{field_name} must contain exactly {', '.join(keys)}"
            )
        return {
            key: vector(
                payload[key],
                field_name=f"{context}.{field_name}.{key}",
                length=7,
            )
            for key in keys
        }

    def vector(payload: object, *, field_name: str, length: int) -> list[float]:
        if not isinstance(payload, list) or len(payload) != length:
            raise ArchiveValidationError(
                f"{field_name} must contain exactly {length} numeric values"
            )
        result: list[float] = []
        for item in payload:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ArchiveValidationError(f"{field_name} values must be real numbers")
            normalized = float(item)
            if not np.isfinite(normalized):
                raise ArchiveValidationError(f"{field_name} values must be finite")
            result.append(normalized)
        return result

    return {
        "object_poses": pose_mapping(
            value["object_poses"],
            keys=("red_cube", "green_cube", "blue_cube"),
            field_name="object_poses",
        ),
        "bin_poses": pose_mapping(
            value["bin_poses"],
            keys=("left_bin", "right_bin"),
            field_name="bin_poses",
        ),
        "panda_qpos": vector(
            value["panda_qpos"],
            field_name=f"{context}.panda_qpos",
            length=9,
        ),
    }


def _initial_scene_metadata_fingerprint(
    episode: Mapping[str, Any],
    *,
    context: str,
) -> str:
    langmani = _required_mapping(episode, "langmani", context=context)
    state = _validated_initial_scene_state(
        langmani.get("initial_scene_state"),
        context=f"{context}.langmani.initial_scene_state",
    )
    canonical = json.dumps(
        state,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_initial_scene_against_env_state(
    value: object,
    env_states: h5py.Group,
    *,
    context: str,
) -> None:
    """Bind the explicit cube/Panda snapshot to native state zero.

    Static bin poses cannot be cross-checked here because ManiSkill 3.0.1 omits
    static actors from ``get_state_dict``; their actual poses are covered by the
    explicit snapshot and the all-six equality check.
    """
    snapshot = _validated_initial_scene_state(
        value,
        context=f"{context}.langmani.initial_scene_state",
    )
    actors = env_states.get("actors")
    if not isinstance(actors, h5py.Group):
        raise ArchiveValidationError(f"{context}/env_states/actors must be a group")
    for object_id, expected_pose in snapshot["object_poses"].items():
        actor = actors.get(object_id)
        if not isinstance(actor, h5py.Dataset) or actor.ndim != 2 or actor.shape[1] < 7:
            raise ArchiveValidationError(
                f"{context}/env_states/actors/{object_id} must contain actor state"
            )
        if not np.allclose(
            np.asarray(actor[0, :7], dtype=np.float64),
            np.asarray(expected_pose, dtype=np.float64),
            rtol=0.0,
            atol=1e-8,
        ):
            raise ArchiveValidationError(
                f"{context} initial physical state: explicit {object_id} pose "
                "disagrees with native state zero"
            )
    articulations = env_states.get("articulations")
    panda = articulations.get("panda") if isinstance(articulations, h5py.Group) else None
    if not isinstance(panda, h5py.Dataset) or panda.ndim != 2 or panda.shape[1] < 22:
        raise ArchiveValidationError(
            f"{context}/env_states/articulations/panda must contain Panda state"
        )
    if not np.allclose(
        np.asarray(panda[0, 13:22], dtype=np.float64),
        np.asarray(snapshot["panda_qpos"], dtype=np.float64),
        rtol=0.0,
        atol=1e-8,
    ):
        raise ArchiveValidationError(
            f"{context} initial physical state: explicit Panda qpos "
            "disagrees with native state zero"
        )


def _initial_state_fingerprint(group: h5py.Group) -> str:
    digest = hashlib.sha256()

    def visit(current: h5py.Group, prefix: str) -> None:
        for key in sorted(current.keys()):
            value = current[key]
            path = f"{prefix}/{key}"
            if isinstance(value, h5py.Group):
                visit(value, path)
            else:
                first = np.asarray(value[0])
                digest.update(path.encode("utf-8"))
                digest.update(first.dtype.str.encode("ascii"))
                digest.update(json.dumps(first.shape).encode("ascii"))
                digest.update(first.tobytes(order="C"))

    visit(group, "env_states")
    return digest.hexdigest()


def _required_dataset(group: h5py.Group, key: str, *, context: str) -> h5py.Dataset:
    value = group[key]
    if not isinstance(value, h5py.Dataset):
        raise ArchiveValidationError(f"{context}/{key} must be an HDF5 dataset")
    return value


def _required_mapping(parent: Mapping[str, Any], key: str, *, context: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ArchiveValidationError(f"{context}.{key} must be a mapping")
    return value


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArchiveValidationError(f"cannot read JSON metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArchiveValidationError("native archive JSON root must be an object")
    return value


def _json_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    try:
        normalized = json.loads(
            json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be JSON-serializable: {exc}") from exc
    if not isinstance(normalized, dict):
        raise TypeError(f"{name} must serialize to a JSON object")
    return normalized


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _require_unchanged_pair(
    h5_path: Path,
    json_path: Path,
    *,
    expected_h5_sha256: str,
    expected_json_sha256: str,
) -> None:
    if sha256_file(h5_path) != expected_h5_sha256:
        raise ArchiveValidationError(f"archive changed after validation: {h5_path}")
    if sha256_file(json_path) != expected_json_sha256:
        raise ArchiveValidationError(f"archive changed after validation: {json_path}")


def _temporary_pair_paths(output_h5: Path) -> tuple[Path, Path]:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{output_h5.stem}.", suffix=".h5.tmp", dir=output_h5.parent
    )
    os.close(descriptor)
    temp_h5 = Path(raw_path)
    temp_h5.unlink()
    temp_json = temp_h5.with_suffix(".json.tmp")
    return temp_h5, temp_json


def _promote_pair(temp_h5: Path, temp_json: Path, output_h5: Path, output_json: Path) -> None:
    _fsync_file(temp_h5)
    _fsync_file(temp_json)
    _reserve_pair(output_h5, output_json)
    try:
        os.replace(temp_h5, output_h5)
        os.replace(temp_json, output_json)
        _fsync_directory(output_h5.parent)
    except Exception:
        _remove_if_exists(output_h5)
        _remove_if_exists(output_json)
        _fsync_directory(output_h5.parent)
        raise


def _fsync_file(path: Path) -> None:
    # Windows requires a writable descriptor for ``_commit``/``os.fsync``.
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _reserve_pair(h5_path: Path, json_path: Path) -> None:
    """Reserve both immutable names with O_EXCL before replacing our placeholders."""
    _ensure_new_pair(h5_path, json_path)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    h5_descriptor = os.open(h5_path, flags, 0o600)
    os.close(h5_descriptor)
    try:
        json_descriptor = os.open(json_path, flags, 0o600)
        os.close(json_descriptor)
    except Exception:
        _remove_if_exists(h5_path)
        raise
    _fsync_directory(h5_path.parent)


def _fsync_directory(path: Path) -> None:
    # Windows does not expose directory fsync through Python; file fsync and
    # exclusive reservations still preserve the no-clobber contract there.
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_new_pair(h5_path: Path, json_path: Path) -> None:
    existing = [str(path) for path in (h5_path, json_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "immutable native archive output already exists: " + ", ".join(existing)
        )


def _require_regular_file(path: Path, *, label: str) -> None:
    if not path.is_file():
        raise ArchiveValidationError(f"{label} archive file does not exist: {path}")


def _trajectory_group_sort_key(name: str) -> int:
    if not name.startswith("traj_") or not name.removeprefix("traj_").isdigit():
        raise ArchiveValidationError(f"invalid native trajectory group name {name!r}")
    return int(name.removeprefix("traj_"))


def _non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArchiveValidationError(f"{name} must be an integer")
    if value < 0:
        raise ArchiveValidationError(f"{name} must be non-negative")
    return value


def _non_empty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArchiveValidationError(f"{name} must be a non-empty string")
    return value


def _optional_metadata_string(metadata: object, key: str) -> str | None:
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get(key)
    return value if isinstance(value, str) and value else None


def _optional_attempt_id(metadata: object) -> str | None:
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("attempt_id", metadata.get("expert_attempt_id"))
    return value if isinstance(value, str) and value else None


def _remove_if_exists(path: Path) -> None:
    with suppress(FileNotFoundError):
        path.unlink()
