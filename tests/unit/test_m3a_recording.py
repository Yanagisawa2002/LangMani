"""CPU-only tests for the narrow ManiSkill RecordEpisode adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import langmani.collection.recording as recording


def test_native_trajectory_stem_and_paths_are_strict(tmp_path: Path) -> None:
    h5_path, json_path = recording.native_trajectory_paths(tmp_path)
    assert h5_path == tmp_path / "attempts.h5"
    assert json_path == tmp_path / "attempts.json"

    for malformed in ("attempts.h5", "nested/attempts", ""):
        with pytest.raises((TypeError, ValueError)):
            recording.native_trajectory_paths(tmp_path, trajectory_stem=malformed)


def test_candidate_recorder_uses_required_native_options_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeRecordEpisode:
        def __init__(self, env: object, **kwargs: Any) -> None:
            captured.update(env=env, kwargs=kwargs)

    env = object()
    monkeypatch.setattr(recording, "create_expert_environment", lambda **kwargs: env)
    monkeypatch.setattr(recording, "RecordEpisode", FakeRecordEpisode)

    result = recording.create_candidate_recorder(tmp_path, sim_backend="physx_cpu")

    assert isinstance(result, FakeRecordEpisode)
    kwargs = captured["kwargs"]
    assert kwargs["save_trajectory"] is True
    assert kwargs["save_video"] is False
    assert kwargs["save_on_reset"] is False
    assert kwargs["clean_on_close"] is True
    assert kwargs["record_reward"] is False
    assert kwargs["record_env_state"] is True
    assert kwargs["source_type"] == "motionplanning"

    (tmp_path / "attempts.h5").write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="already exists"):
        recording.create_candidate_recorder(tmp_path, sim_backend="physx_cpu")


def test_flush_attempt_maps_only_new_saved_native_episode(tmp_path: Path) -> None:
    json_path = tmp_path / "attempts.json"

    class FakeRecorder:
        def flush_trajectory(self, *, save: bool) -> None:
            if not save:
                return
            payload = (
                json.loads(json_path.read_text(encoding="utf-8"))
                if json_path.exists()
                else {"episodes": []}
            )
            episode_id = len(payload["episodes"])
            payload["episodes"].append({"episode_id": episode_id})
            json_path.write_text(json.dumps(payload), encoding="utf-8")

    recorder = FakeRecorder()
    assert (
        recording.flush_attempt(
            recorder,
            save=False,
            output_directory=tmp_path,  # type: ignore[arg-type]
        )
        is None
    )

    location = recording.flush_attempt(
        recorder,
        save=True,
        output_directory=tmp_path,  # type: ignore[arg-type]
    )
    assert location is not None
    assert location.episode_id == 0
    assert location.h5_group == "traj_0"
    assert location.h5_path == tmp_path / "attempts.h5"
