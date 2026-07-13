"""Narrow adapter around ManiSkill 3.0.1 ``RecordEpisode``.

The upstream recorder deliberately remains the owner of the native HDF5/JSON
layout.  This module only fixes the M3A integration contract: one single-env
candidate file, explicit flush after every attempt, no videos, and no implicit
overwrite or append behavior.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mani_skill.utils.wrappers.record import RecordEpisode

from langmani.experts.command_support import create_expert_environment

NATIVE_TRAJECTORY_STEM = "attempts"


class RecordingContractError(RuntimeError):
    """Raised when the inspected upstream recorder contract is violated."""


@dataclass(frozen=True, slots=True)
class NativeEpisodeLocation:
    """Local location assigned by one closed ManiSkill trajectory pair."""

    episode_id: int
    h5_group: str
    h5_path: Path
    json_path: Path


def native_trajectory_paths(
    output_directory: str | Path,
    *,
    trajectory_stem: str = NATIVE_TRAJECTORY_STEM,
) -> tuple[Path, Path]:
    """Return the exact pair that ``RecordEpisode`` will create."""
    if not isinstance(trajectory_stem, str) or not trajectory_stem:
        raise TypeError("trajectory_stem must be a non-empty string")
    if Path(trajectory_stem).name != trajectory_stem or Path(trajectory_stem).suffix:
        raise ValueError("trajectory_stem must be a plain filename stem")
    directory = Path(output_directory)
    return directory / f"{trajectory_stem}.h5", directory / f"{trajectory_stem}.json"


def create_candidate_recorder(
    output_directory: str | Path,
    *,
    sim_backend: str,
    trajectory_stem: str = NATIVE_TRAJECTORY_STEM,
) -> RecordEpisode:
    """Create the single supported M3A collection environment and recorder.

    Existing output is rejected before constructing ``RecordEpisode`` because
    ManiSkill 3.0.1 opens its HDF5 file with mode ``"w"``.
    """
    h5_path, json_path = native_trajectory_paths(
        output_directory,
        trajectory_stem=trajectory_stem,
    )
    if h5_path.exists() or json_path.exists():
        raise FileExistsError(
            f"candidate trajectory output already exists: {h5_path} / {json_path}"
        )
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    env = create_expert_environment(
        diagnostic_rendering=False,
        sim_backend=sim_backend,
    )
    return RecordEpisode(
        env,
        output_dir=str(h5_path.parent),
        trajectory_name=trajectory_stem,
        save_trajectory=True,
        save_video=False,
        save_on_reset=False,
        clean_on_close=True,
        record_reward=False,
        record_env_state=True,
        source_type="motionplanning",
        source_desc=("LangMani M3A deterministic PickPlaceExpert raw demonstrations"),
    )


def flush_attempt(
    recorder: RecordEpisode,
    *,
    save: bool,
    output_directory: str | Path,
    trajectory_stem: str = NATIVE_TRAJECTORY_STEM,
) -> NativeEpisodeLocation | None:
    """Explicitly close the current attempt buffer and report a new local ID.

    ``RecordEpisode.flush_trajectory`` has no return value.  M3A therefore reads
    the public JSON artifact it writes and requires that exactly zero or one
    episode was added.  Zero is valid only for an empty attempt.
    """
    h5_path, json_path = native_trajectory_paths(
        output_directory,
        trajectory_stem=trajectory_stem,
    )
    before = _episode_count(json_path)
    recorder.flush_trajectory(save=save)
    if not save:
        if _episode_count(json_path) != before:
            raise RecordingContractError("save=False unexpectedly added a native episode")
        return None

    payload = _load_native_json(json_path)
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise RecordingContractError("native trajectory JSON is missing an episodes list")
    added = len(episodes) - before
    if added == 0:
        return None
    if added != 1:
        raise RecordingContractError(
            f"one flush must add at most one native episode, added {added}"
        )
    episode = episodes[-1]
    if not isinstance(episode, dict):
        raise RecordingContractError("native episode metadata must be a mapping")
    episode_id = episode.get("episode_id")
    if isinstance(episode_id, bool) or not isinstance(episode_id, int) or episode_id < 0:
        raise RecordingContractError("native episode_id must be a non-negative integer")
    return NativeEpisodeLocation(
        episode_id=episode_id,
        h5_group=f"traj_{episode_id}",
        h5_path=h5_path,
        json_path=json_path,
    )


def close_recorder(recorder: RecordEpisode) -> None:
    """Close the recorder and surface every simulator or HDF5 failure."""
    recorder.close()


def _episode_count(json_path: Path) -> int:
    if not json_path.exists():
        return 0
    payload = _load_native_json(json_path)
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise RecordingContractError("native trajectory JSON is missing an episodes list")
    return len(episodes)


def _load_native_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RecordingContractError(
            f"cannot read native trajectory JSON {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise RecordingContractError("native trajectory JSON root must be a mapping")
    return payload


__all__ = [
    "NATIVE_TRAJECTORY_STEM",
    "NativeEpisodeLocation",
    "RecordingContractError",
    "close_recorder",
    "create_candidate_recorder",
    "flush_attempt",
    "native_trajectory_paths",
]
