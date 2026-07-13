"""Unit tests for immutable M1 task and episode metadata."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from langmani.environments.specs import (
    BIN_IDS,
    OBJECT_IDS,
    EpisodeSpec,
    TaskSpec,
    canonical_instruction,
)


def _task(object_id: str = "red_cube", bin_id: str = "left_bin") -> TaskSpec:
    return TaskSpec.from_mapping(
        {
            "target_object_id": object_id,
            "target_bin_id": bin_id,
            "instruction_template_id": "canonical_v0",
        }
    )


def test_all_six_canonical_instructions_are_exact() -> None:
    expected = (
        "Pick up the red cube and place it in the left bin.",
        "Pick up the red cube and place it in the right bin.",
        "Pick up the green cube and place it in the left bin.",
        "Pick up the green cube and place it in the right bin.",
        "Pick up the blue cube and place it in the left bin.",
        "Pick up the blue cube and place it in the right bin.",
    )
    actual = tuple(
        canonical_instruction(_task(object_id, bin_id))
        for object_id in OBJECT_IDS
        for bin_id in BIN_IDS
    )

    assert actual == expected


def test_task_and_episode_specs_round_trip_through_json() -> None:
    task_spec = _task("green_cube", "right_bin")
    episode_spec = EpisodeSpec.create(scene_seed=123, task_spec=task_spec)

    encoded = json.dumps(episode_spec.to_dict(), sort_keys=True)
    decoded = json.loads(encoded)

    assert TaskSpec.from_mapping(decoded["task_spec"]) == task_spec
    assert decoded == episode_spec.to_dict()
    assert episode_spec.scene_id == "langmani-pick-place-scene-v0:123"
    assert episode_spec.task_id == ("langmani-pick-place-task-v0:green_cube:right_bin:canonical_v0")


def test_scene_and_task_identifiers_are_independent() -> None:
    left = EpisodeSpec.create(scene_seed=7, task_spec=_task("blue_cube", "left_bin"))
    right = EpisodeSpec.create(scene_seed=7, task_spec=_task("red_cube", "right_bin"))
    different_scene = EpisodeSpec.create(scene_seed=8, task_spec=_task("blue_cube", "left_bin"))

    assert left.scene_id == right.scene_id
    assert left.task_id != right.task_id
    assert left.task_id == different_scene.task_id
    assert left.scene_id != different_scene.scene_id


def test_specs_are_frozen() -> None:
    task_spec = _task()
    episode_spec = EpisodeSpec.create(scene_seed=0, task_spec=task_spec)

    with pytest.raises(FrozenInstanceError):
        task_spec.target_bin_id = "right_bin"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        episode_spec.scene_seed = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("value", "error_type", "message"),
    [
        (None, TypeError, "task_spec must be a mapping"),
        ({}, ValueError, "missing fields"),
        (
            {
                "target_object_id": "red_cube",
                "target_bin_id": "left_bin",
                "instruction_template_id": "canonical_v0",
                "extra": "forbidden",
            },
            ValueError,
            "unexpected fields",
        ),
        (
            {
                "target_object_id": 1,
                "target_bin_id": "left_bin",
                "instruction_template_id": "canonical_v0",
            },
            TypeError,
            "target_object_id must be a string",
        ),
        (
            {
                "target_object_id": "yellow_cube",
                "target_bin_id": "left_bin",
                "instruction_template_id": "canonical_v0",
            },
            ValueError,
            "unknown target_object_id",
        ),
        (
            {
                "target_object_id": "red_cube",
                "target_bin_id": "middle_bin",
                "instruction_template_id": "canonical_v0",
            },
            ValueError,
            "unknown target_bin_id",
        ),
        (
            {
                "target_object_id": "red_cube",
                "target_bin_id": "left_bin",
                "instruction_template_id": "paraphrase_v1",
            },
            ValueError,
            "unknown instruction_template_id",
        ),
    ],
)
def test_malformed_task_specs_fail_clearly(
    value: object, error_type: type[Exception], message: str
) -> None:
    with pytest.raises(error_type, match=message):
        TaskSpec.from_mapping(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("scene_seed", [True, -1, 1.5, "1"])
def test_invalid_scene_seed_is_rejected(scene_seed: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        EpisodeSpec.create(scene_seed=scene_seed, task_spec=_task())  # type: ignore[arg-type]


def test_episode_spec_rejects_inconsistent_derived_metadata() -> None:
    task_spec = _task()

    with pytest.raises(ValueError, match="canonical_instruction"):
        EpisodeSpec(
            scene_seed=1,
            task_spec=task_spec,
            canonical_instruction="wrong",
            scene_id="langmani-pick-place-scene-v0:1",
            task_id="langmani-pick-place-task-v0:red_cube:left_bin:canonical_v0",
        )
