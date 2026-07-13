"""Deterministic, counterfactual-scene-level split assignment for M3B."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from langmani.datasets.identity import sha256_hex
from langmani.datasets.lerobot_types import (
    DatasetSplit,
    EpisodeExportRecord,
    SplitAssignment,
    SplitConfig,
)


def scene_group_split_digest(*, split_seed: int, scene_group_id: str) -> str:
    """Return the canonical digest used to rank one physical scene group."""
    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise TypeError("split_seed must be an integer")
    if split_seed < 0:
        raise ValueError("split_seed must be non-negative")
    if not isinstance(scene_group_id, str) or not scene_group_id:
        raise TypeError("scene_group_id must be a non-empty string")
    return sha256_hex({"scene_group_id": scene_group_id, "split_seed": split_seed})


def build_split_assignments(
    scene_group_ids: Sequence[str], config: SplitConfig
) -> tuple[SplitAssignment, ...]:
    """Assign unique scene groups by digest rank, with stable-ID collision tie-breaking."""
    if not isinstance(config, SplitConfig):
        raise TypeError("config must be a SplitConfig")
    group_ids = tuple(scene_group_ids)
    if not all(isinstance(item, str) and item for item in group_ids):
        raise TypeError("scene_group_ids must contain non-empty strings")
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("scene_group_ids must be unique")
    if len(group_ids) != config.total_scene_groups:
        raise ValueError(
            "scene_group_ids count must equal the configured scene-group total "
            f"({config.total_scene_groups})"
        )

    ranked = sorted(
        (
            scene_group_split_digest(split_seed=config.split_seed, scene_group_id=group_id),
            group_id,
        )
        for group_id in group_ids
    )
    train_stop = config.train_scene_groups
    validation_stop = train_stop + config.validation_scene_groups

    assignments: list[SplitAssignment] = []
    for rank, (digest, group_id) in enumerate(ranked):
        if rank < train_stop:
            split = DatasetSplit.TRAIN
        elif rank < validation_stop:
            split = DatasetSplit.VALIDATION
        else:
            split = DatasetSplit.TEST
        assignments.append(
            SplitAssignment(
                scene_group_id=group_id,
                split=split,
                digest=digest,
                rank=rank,
            )
        )
    return tuple(assignments)


def assignment_map(
    assignments: Sequence[SplitAssignment],
) -> Mapping[str, SplitAssignment]:
    """Return a read-only scene-group lookup after validating uniqueness and ranks."""
    values = tuple(assignments)
    if not all(isinstance(item, SplitAssignment) for item in values):
        raise TypeError("assignments must contain SplitAssignment values")
    by_id = {item.scene_group_id: item for item in values}
    if len(by_id) != len(values):
        raise ValueError("a scene group cannot appear in more than one split")
    ranks = sorted(item.rank for item in values)
    if ranks != list(range(len(values))):
        raise ValueError("assignment ranks must be contiguous from zero")
    return MappingProxyType(by_id)


def episode_indices_by_split(
    records: Sequence[EpisodeExportRecord],
) -> Mapping[str, tuple[int, ...]]:
    """Expose deterministic LeRobot episode-index lists without copying dataset files."""
    values = tuple(records)
    if not all(isinstance(item, EpisodeExportRecord) for item in values):
        raise TypeError("records must contain EpisodeExportRecord values")
    if len({item.source_episode_id for item in values}) != len(values):
        raise ValueError("source episodes must be unique")
    if len({item.lerobot_episode_index for item in values}) != len(values):
        raise ValueError("LeRobot episode indices must be unique")
    return MappingProxyType(
        {
            split.value: tuple(
                sorted(item.lerobot_episode_index for item in values if item.split is split)
            )
            for split in DatasetSplit
        }
    )


def task_counts_by_split(
    assignments: Sequence[SplitAssignment],
    task_ids_by_scene_group: Mapping[str, Sequence[str]],
) -> Mapping[str, Mapping[str, int]]:
    """Validate six-task counterfactual groups and return immutable split task counts."""
    by_group = assignment_map(assignments)
    if set(task_ids_by_scene_group) != set(by_group):
        missing = sorted(set(by_group) - set(task_ids_by_scene_group))
        extra = sorted(set(task_ids_by_scene_group) - set(by_group))
        raise ValueError(
            "task mapping must cover exactly the assigned scene groups; "
            f"missing={missing!r}, extra={extra!r}"
        )

    canonical_tasks: frozenset[str] | None = None
    counts = {split.value: Counter[str]() for split in DatasetSplit}
    for group_id, assignment in by_group.items():
        task_ids = tuple(task_ids_by_scene_group[group_id])
        if len(task_ids) != 6 or len(set(task_ids)) != 6:
            raise ValueError("each counterfactual scene group must contain six distinct tasks")
        if not all(isinstance(item, str) and item for item in task_ids):
            raise TypeError("task IDs must be non-empty strings")
        task_set = frozenset(task_ids)
        if canonical_tasks is None:
            canonical_tasks = task_set
        elif task_set != canonical_tasks:
            raise ValueError("all scene groups must contain the same canonical six tasks")
        counts[assignment.split.value].update(task_ids)

    result: dict[str, Mapping[str, int]] = {}
    all_tasks = tuple(sorted(canonical_tasks or ()))
    for split in DatasetSplit:
        expected_count = sum(1 for item in by_group.values() if item.split is split)
        split_counts = {task_id: counts[split.value][task_id] for task_id in all_tasks}
        if any(value != expected_count for value in split_counts.values()):
            raise ValueError(f"{split.value} is not perfectly balanced across the six tasks")
        result[split.value] = MappingProxyType(split_counts)
    return MappingProxyType(result)


__all__ = [
    "assignment_map",
    "build_split_assignments",
    "episode_indices_by_split",
    "scene_group_split_digest",
    "task_counts_by_split",
]
