from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from langmani.environments.push_specs import (
    PUSH_DIFFICULTIES,
    PUSH_OBJECT_IDS,
    TARGET_REGION_IDS,
    PushEpisodeSpec,
    PushTaskSpec,
    canonical_push_instruction,
)


def _task(
    object_id: str = "blue_cube",
    region_id: str = "left",
    difficulty: str = "standard",
) -> PushTaskSpec:
    return PushTaskSpec.from_mapping(
        {
            "target_object_id": object_id,
            "target_region_id": region_id,
            "difficulty": difficulty,
            "instruction_template_id": "canonical_push_v0",
        }
    )


def test_push_specs_cover_two_geometries_four_regions_and_two_difficulties() -> None:
    tasks = tuple(
        _task(object_id, region_id, difficulty)
        for object_id in PUSH_OBJECT_IDS
        for region_id in TARGET_REGION_IDS
        for difficulty in PUSH_DIFFICULTIES
    )

    assert len(tasks) == 16
    assert len(set(tasks)) == 16
    assert "cube" in canonical_push_instruction(tasks[0])
    assert "without picking it up" in canonical_push_instruction(tasks[0])


def test_push_task_and_episode_round_trip_through_json() -> None:
    task = _task("orange_cylinder", "forward_right", "hard")
    episode = PushEpisodeSpec.create(scene_seed=811, task_spec=task)

    decoded = json.loads(json.dumps(episode.to_dict(), sort_keys=True))

    assert PushTaskSpec.from_mapping(decoded["task_spec"]) == task
    assert decoded == episode.to_dict()
    assert episode.scene_id == "langmani-push-scene-v0:811"
    assert episode.task_id == (
        "langmani-push-task-v0:orange_cylinder:forward_right:hard:canonical_push_v0"
    )


def test_push_scene_and_task_identity_are_independent() -> None:
    left = PushEpisodeSpec.create(scene_seed=5, task_spec=_task())
    right = PushEpisodeSpec.create(scene_seed=5, task_spec=_task(region_id="right"))
    another_scene = PushEpisodeSpec.create(scene_seed=6, task_spec=_task())

    assert left.scene_id == right.scene_id
    assert left.task_id != right.task_id
    assert left.task_id == another_scene.task_id
    assert left.scene_id != another_scene.scene_id


def test_push_specs_are_frozen_and_reject_malformed_values() -> None:
    task = _task()
    with pytest.raises(FrozenInstanceError):
        task.difficulty = "hard"  # type: ignore[misc]
    with pytest.raises(ValueError, match="missing fields"):
        PushTaskSpec.from_mapping({})
    with pytest.raises(ValueError, match="unknown target_object_id"):
        _task("sphere")
    with pytest.raises(ValueError, match="unknown target_region_id"):
        _task(region_id="middle")
    with pytest.raises(ValueError, match="unknown difficulty"):
        _task(difficulty="extreme")


@pytest.mark.parametrize("seed", [True, -1, 1.5, "7"])
def test_push_episode_rejects_invalid_scene_seed(seed: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        PushEpisodeSpec.create(scene_seed=seed, task_spec=_task())  # type: ignore[arg-type]
