"""Scene-level split determinism and counterfactual-balance tests."""

from __future__ import annotations

from collections import Counter

import pytest

import langmani.datasets.splits as split_module
from langmani.datasets.lerobot_types import DatasetSplit, SplitAssignment, SplitConfig
from langmani.datasets.splits import (
    assignment_map,
    build_split_assignments,
    scene_group_split_digest,
    task_counts_by_split,
)


def _group_ids(count: int) -> tuple[str, ...]:
    return tuple(f"scene-group-{index:03d}" for index in range(count))


def _task_mapping(group_ids: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    tasks = tuple(f"task-{index}" for index in range(6))
    return {group_id: tasks for group_id in group_ids}


def test_full_split_is_deterministic_input_order_independent_and_exact() -> None:
    config = SplitConfig.full(split_seed=20260713)
    group_ids = _group_ids(60)
    first = build_split_assignments(group_ids, config)
    second = build_split_assignments(tuple(reversed(group_ids)), config)

    assert first == second
    assert tuple(item.rank for item in first) == tuple(range(60))
    assert [item.scene_group_id for item in first] == [
        item.scene_group_id
        for item in sorted(first, key=lambda item: (item.digest, item.scene_group_id))
    ]
    assert Counter(item.split for item in first) == {
        DatasetSplit.TRAIN: 48,
        DatasetSplit.VALIDATION: 6,
        DatasetSplit.TEST: 6,
    }
    assert len(assignment_map(first)) == 60


def test_full_split_is_perfectly_balanced_across_all_six_tasks() -> None:
    group_ids = _group_ids(60)
    assignments = build_split_assignments(group_ids, SplitConfig.full(split_seed=9))
    counts = task_counts_by_split(assignments, _task_mapping(group_ids))

    assert dict(counts["train"]) == {f"task-{index}": 48 for index in range(6)}
    assert dict(counts["validation"]) == {f"task-{index}": 6 for index in range(6)}
    assert dict(counts["test"]) == {f"task-{index}": 6 for index in range(6)}
    assert sum(counts["train"].values()) == 288
    assert sum(counts["validation"].values()) == 36
    assert sum(counts["test"].values()) == 36


def test_smoke_split_is_one_train_group_and_six_balanced_episodes() -> None:
    group_ids = ("one-real-counterfactual-group",)
    config = SplitConfig.smoke(split_seed=3)
    assignments = build_split_assignments(group_ids, config)
    counts = task_counts_by_split(assignments, _task_mapping(group_ids))

    assert config.to_dict() == {
        "train_scene_groups": 1,
        "validation_scene_groups": 0,
        "test_scene_groups": 0,
        "split_seed": 3,
    }
    assert assignments[0].split is DatasetSplit.TRAIN
    assert all(value == 1 for value in counts["train"].values())
    assert all(value == 0 for value in counts["validation"].values())
    assert all(value == 0 for value in counts["test"].values())


def test_digest_collision_is_resolved_by_stable_scene_group_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        split_module,
        "scene_group_split_digest",
        lambda *, split_seed, scene_group_id: "a" * 64,
    )
    group_ids = tuple(reversed(("group-c", "group-a", "group-b")))
    assignments = build_split_assignments(group_ids, SplitConfig(2, 1, 0, 0))

    assert tuple(item.scene_group_id for item in assignments) == (
        "group-a",
        "group-b",
        "group-c",
    )
    assert tuple(item.split for item in assignments) == (
        DatasetSplit.TRAIN,
        DatasetSplit.TRAIN,
        DatasetSplit.VALIDATION,
    )


def test_split_digest_is_process_stable_and_tracks_seed_and_group() -> None:
    digest = scene_group_split_digest(split_seed=1, scene_group_id="group-a")

    assert len(digest) == 64
    assert digest == scene_group_split_digest(split_seed=1, scene_group_id="group-a")
    assert digest != scene_group_split_digest(split_seed=2, scene_group_id="group-a")
    assert digest != scene_group_split_digest(split_seed=1, scene_group_id="group-b")


@pytest.mark.parametrize(
    ("group_ids", "config", "message"),
    [
        (("duplicate", "duplicate"), SplitConfig(2, 0, 0, 0), "unique"),
        (("only-one",), SplitConfig(2, 0, 0, 0), "count"),
        ((), SplitConfig.smoke(), "count"),
    ],
)
def test_split_rejects_duplicates_missing_groups_and_partial_inputs(
    group_ids: tuple[str, ...], config: SplitConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_split_assignments(group_ids, config)


def test_assignment_map_rejects_scene_leakage_and_noncontiguous_ranks() -> None:
    first = SplitAssignment("same-group", DatasetSplit.TRAIN, "a" * 64, 0)
    leaked = SplitAssignment("same-group", DatasetSplit.TEST, "b" * 64, 1)
    skipped_rank = SplitAssignment("other-group", DatasetSplit.TEST, "b" * 64, 2)

    with pytest.raises(ValueError, match="more than one split"):
        assignment_map((first, leaked))
    with pytest.raises(ValueError, match="contiguous"):
        assignment_map((first, skipped_rank))


def test_task_balance_rejects_partial_or_inconsistent_counterfactual_groups() -> None:
    group_ids = ("group-a", "group-b")
    assignments = build_split_assignments(group_ids, SplitConfig(1, 1, 0, 0))
    mapping = _task_mapping(group_ids)

    with pytest.raises(ValueError, match="cover exactly"):
        task_counts_by_split(assignments, {"group-a": mapping["group-a"]})
    with pytest.raises(ValueError, match="six distinct"):
        task_counts_by_split(assignments, {**mapping, "group-b": ("task-0",) * 6})
    with pytest.raises(ValueError, match="same canonical six"):
        task_counts_by_split(
            assignments,
            {**mapping, "group-b": tuple(f"other-{index}" for index in range(6))},
        )
