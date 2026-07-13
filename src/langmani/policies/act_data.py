"""Completed-M3B-only data boundary for the M4 ACT baselines.

Normal M4 training never reopens M3A.  This module validates the immutable M3B
completion evidence, derives explicit scene-safe episode views, opens those
views through LeRobot's public API, and computes normalization statistics from
the selected training frames only.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from langmani.datasets.identity import sha256_hex
from langmani.datasets.lerobot_types import (
    ACTION_FEATURE_KEY,
    IMAGE_FEATURE_KEY,
    LEROBOT_MANAGED_FEATURE_KEYS,
    PANDA_ACTION_COMPONENTS,
    POLICY_FEATURE_KEYS,
    STATE_FEATURE_KEY,
    DatasetSplit,
    EpisodeExportStatus,
    ExportMode,
    LeRobotDatasetSummary,
    LeRobotExportConfig,
    LeRobotExportManifest,
    LeRobotValidationReport,
)
from langmani.datasets.lerobot_writer import COMPLETION_MARKER, SIDECAR_DIRECTORY
from langmani.datasets.policy_state import PANDA_POLICY_STATE_COMPONENTS
from langmani.datasets.splits import build_split_assignments
from langmani.policies.act_conditioning import CANONICAL_TASK_IDS
from langmani.policies.act_types import (
    ACT_ACTION_COMPONENTS,
    ACT_CONDITIONED_STATE_COMPONENTS,
    ACT_STATE_COMPONENTS,
    TASK_ONEHOT_MAPPING_VERSION,
    ActVariant,
)

TRAIN_STATS_SCHEMA_VERSION = "langmani-m4-train-stats-v1"
TRAIN_STATS_ALGORITHM = "deterministic-batch-welford-population-v1"
SPLIT_MANIFEST_SCHEMA = "langmani-m3b-splits-v1"
COMPLETION_SCHEMA = "langmani-m3b-completion-v1"
EXPECTED_LEROBOT_CODEBASE_VERSION = "v3.0"
IMAGE_COMPONENTS = ("red", "green", "blue")


class ActDataContractError(RuntimeError):
    """Raised when completed data, views, or train statistics are unsafe."""


@dataclass(frozen=True, slots=True)
class DatasetEpisodeView:
    """One explicit, ordered set of original LeRobot episode indices."""

    split: DatasetSplit
    episode_indices: tuple[int, ...]
    task_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.split, DatasetSplit):
            object.__setattr__(self, "split", DatasetSplit(self.split))
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in self.episode_indices
        ):
            raise TypeError("episode indices must be non-negative integers")
        if len(set(self.episode_indices)) != len(self.episode_indices):
            raise ValueError("an episode view cannot contain duplicate indices")
        if tuple(sorted(self.episode_indices)) != self.episode_indices:
            raise ValueError("episode views must use ascending deterministic order")
        if self.task_id is not None and self.task_id not in CANONICAL_TASK_IDS:
            raise ValueError("per-task view has an unknown stable task ID")


@dataclass(frozen=True, slots=True)
class ActDatasetViews:
    """All global and per-task scene-level M3B views needed by M4."""

    train: DatasetEpisodeView
    validation: DatasetEpisodeView
    test: DatasetEpisodeView
    per_task: Mapping[str, Mapping[str, DatasetEpisodeView]]
    task_id_by_episode: Mapping[int, str]
    scene_group_id_by_episode: Mapping[int, str]

    def __post_init__(self) -> None:
        expected_splits = (DatasetSplit.TRAIN, DatasetSplit.VALIDATION, DatasetSplit.TEST)
        if (
            tuple(view.split for view in (self.train, self.validation, self.test))
            != expected_splits
        ):
            raise ValueError("global dataset views must be train, validation, and test")
        global_sets = [
            set(self.train.episode_indices),
            set(self.validation.episode_indices),
            set(self.test.episode_indices),
        ]
        if any(
            global_sets[left] & global_sets[right]
            for left in range(3)
            for right in range(left + 1, 3)
        ):
            raise ValueError("train, validation, and test episode views must be disjoint")
        if set(self.per_task) != set(CANONICAL_TASK_IDS):
            raise ValueError("per_task views must contain exactly the canonical six task IDs")
        normalized: dict[str, Mapping[str, DatasetEpisodeView]] = {}
        global_by_split = {
            DatasetSplit.TRAIN.value: global_sets[0],
            DatasetSplit.VALIDATION.value: global_sets[1],
            DatasetSplit.TEST.value: global_sets[2],
        }
        for task_id in CANONICAL_TASK_IDS:
            values = self.per_task[task_id]
            if set(values) != set(global_by_split):
                raise ValueError("each task must expose train, validation, and test views")
            task_views: dict[str, DatasetEpisodeView] = {}
            for split_name, global_indices in global_by_split.items():
                view = values[split_name]
                if view.task_id != task_id or not set(view.episode_indices) <= global_indices:
                    raise ValueError("per-task episode views must be subsets of their global split")
                task_views[split_name] = view
            normalized[task_id] = MappingProxyType(task_views)
        object.__setattr__(self, "per_task", MappingProxyType(normalized))
        task_map = dict(self.task_id_by_episode)
        group_map = dict(self.scene_group_id_by_episode)
        all_indices = set().union(*global_sets)
        if set(task_map) != all_indices or set(group_map) != all_indices:
            raise ValueError("episode metadata mappings must cover every view exactly")
        if any(value not in CANONICAL_TASK_IDS for value in task_map.values()):
            raise ValueError("episode task mapping contains an unknown stable task ID")
        object.__setattr__(self, "task_id_by_episode", MappingProxyType(task_map))
        object.__setattr__(self, "scene_group_id_by_episode", MappingProxyType(group_map))

    def for_split(self, split: DatasetSplit, *, task_id: str | None = None) -> DatasetEpisodeView:
        split_value = DatasetSplit(split)
        if task_id is None:
            return {
                DatasetSplit.TRAIN: self.train,
                DatasetSplit.VALIDATION: self.validation,
                DatasetSplit.TEST: self.test,
            }[split_value]
        if task_id not in self.per_task:
            raise ValueError(f"unknown stable task ID {task_id!r}")
        return self.per_task[task_id][split_value.value]


@dataclass(frozen=True, slots=True)
class CompletedM3BDataset:
    """Validated local M3B authority consumed by M4 without reopening M3A."""

    root: Path
    manifest: LeRobotExportManifest
    summary: LeRobotDatasetSummary
    validation: LeRobotValidationReport
    split_manifest_digest: str
    views: ActDatasetViews

    @property
    def export_fingerprint(self) -> str:
        return self.manifest.export_fingerprint


@dataclass(frozen=True, slots=True)
class LoadedActDataset:
    """Public LeRobot loader result and its exact temporal contract."""

    dataset: Any
    episode_indices: tuple[int, ...]
    delta_timestamps: Mapping[str, tuple[float, ...]]


def validate_temporal_episode_boundaries(
    loaded: LoadedActDataset,
    *,
    chunk_size: int,
) -> dict[str, object]:
    """Audit public first/final ACT samples without recreating delta sampling."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    dataset = loaded.dataset
    episode_values = dataset.hf_dataset["episode_index"]
    frame_values = dataset.hf_dataset["frame_index"]
    bounds: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for position, (episode_value, frame_value) in enumerate(
        zip(episode_values, frame_values, strict=True)
    ):
        bounds[int(episode_value)].append((int(frame_value), position))
    if set(bounds) != set(loaded.episode_indices):
        raise ActDataContractError("temporal boundary audit found the wrong episode set")
    for episode_index in loaded.episode_indices:
        ordered = sorted(bounds[episode_index])
        if [frame for frame, _ in ordered] != list(range(len(ordered))):
            raise ActDataContractError("temporal boundary audit found skipped frame indices")
        first = dataset[ordered[0][1]]
        final = dataset[ordered[-1][1]]
        for label, row in (("first", first), ("final", final)):
            if _scalar_int(row.get("episode_index"), "episode_index") != episode_index:
                raise ActDataContractError(f"{label} temporal sample crossed an episode boundary")
            action = _as_numpy(row.get(ACTION_FEATURE_KEY))
            padding = _as_numpy(row.get("action_is_pad"))
            if action.shape != (chunk_size, ACT_ACTION_COMPONENTS):
                raise ActDataContractError("ACT action chunk has the wrong boundary shape")
            if padding.dtype != np.dtype(np.bool_) or padding.shape != (chunk_size,):
                raise ActDataContractError("ACT action padding has the wrong boundary shape/dtype")
            if bool(padding[0]):
                raise ActDataContractError("current action cannot be marked as padding")
        final_actions = _as_numpy(final.get(ACTION_FEATURE_KEY))
        final_padding = _as_numpy(final.get("action_is_pad"))
        if chunk_size > 1 and (
            not np.all(final_padding[1:])
            or not np.allclose(final_actions[1:], final_actions[0], rtol=0, atol=0)
        ):
            raise ActDataContractError(
                "final ACT sample must clamp within the episode and mark repeated tail actions"
            )
    return {
        "episode_count": len(loaded.episode_indices),
        "chunk_size": chunk_size,
        "first_sample_validated": True,
        "final_sample_validated": True,
        "episode_crossing_detected": False,
    }


@dataclass(frozen=True, slots=True)
class FeatureStatistics:
    """Portable population statistics for one policy feature."""

    feature_name: str
    components: tuple[str, ...]
    count: int
    minimum: tuple[float, ...]
    maximum: tuple[float, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.feature_name or not self.components or self.count < 1:
            raise ValueError("feature statistics require a name, components, and positive count")
        length = len(self.components)
        values = (self.minimum, self.maximum, self.mean, self.std)
        if any(len(item) != length for item in values):
            raise ValueError("feature statistic vectors must match component order")
        if not all(np.isfinite(value) for item in values for value in item):
            raise ValueError("feature statistics must be finite")
        if any(value < 0 for value in self.std):
            raise ValueError("feature standard deviations must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_name": self.feature_name,
            "components": list(self.components),
            "count": self.count,
            "minimum": list(self.minimum),
            "maximum": list(self.maximum),
            "mean": list(self.mean),
            "std": list(self.std),
        }


@dataclass(frozen=True, slots=True)
class StatisticsLeakageAudit:
    """Independent episode-set evidence for train-only normalization."""

    source_episode_indices: tuple[int, ...]
    train_episode_indices: tuple[int, ...]
    validation_episode_indices: tuple[int, ...]
    test_episode_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        values = (
            self.source_episode_indices,
            self.train_episode_indices,
            self.validation_episode_indices,
            self.test_episode_indices,
        )
        if any(tuple(sorted(item)) != item or len(set(item)) != len(item) for item in values):
            raise ValueError("statistics audit indices must be unique and sorted")
        if self.source_episode_indices != self.train_episode_indices:
            raise ValueError("statistics source episodes must equal the declared training view")
        train = set(self.train_episode_indices)
        validation = set(self.validation_episode_indices)
        test = set(self.test_episode_indices)
        if train & validation or train & test or validation & test:
            raise ValueError("statistics audit found train/validation/test episode leakage")

    @property
    def passed(self) -> bool:
        return True

    def to_dict(self) -> dict[str, object]:
        return {
            "source_episode_indices": list(self.source_episode_indices),
            "train_episode_indices": list(self.train_episode_indices),
            "validation_episode_indices": list(self.validation_episode_indices),
            "test_episode_indices": list(self.test_episode_indices),
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class TrainOnlyStatistics:
    """Train-only statistics and exact effective processor values."""

    variant: ActVariant
    task_id: str | None
    image: FeatureStatistics
    state: FeatureStatistics
    action: FeatureStatistics
    leakage_audit: StatisticsLeakageAudit
    lerobot_version: str
    torch_version: str
    algorithm: str = TRAIN_STATS_ALGORITHM
    schema_version: str = TRAIN_STATS_SCHEMA_VERSION
    task_onehot_mapping_version: str = TASK_ONEHOT_MAPPING_VERSION
    onehot_suffix_policy: str = "identity_mean_zero_std_one"
    statistics_fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "variant", ActVariant(self.variant))
        if self.variant is ActVariant.PER_TASK:
            if self.task_id not in CANONICAL_TASK_IDS:
                raise ValueError("per-task statistics require one canonical stable task ID")
        elif self.task_id is not None:
            raise ValueError("mixed statistics must not select a task ID")
        expected_state_components = (
            ACT_CONDITIONED_STATE_COMPONENTS
            if self.variant is ActVariant.MIXED_TASK_ONEHOT
            else ACT_STATE_COMPONENTS
        )
        if len(self.state.components) != expected_state_components:
            raise ValueError("state statistics dimension is incompatible with ACT variant")
        if (
            self.algorithm != TRAIN_STATS_ALGORITHM
            or self.schema_version != TRAIN_STATS_SCHEMA_VERSION
        ):
            raise ValueError("unknown train-statistics contract")
        if self.task_onehot_mapping_version != TASK_ONEHOT_MAPPING_VERSION:
            raise ValueError("unknown one-hot mapping version")
        expected = f"sha256:{sha256_hex(self.identity_payload())}"
        if not self.statistics_fingerprint:
            object.__setattr__(self, "statistics_fingerprint", expected)
        elif self.statistics_fingerprint != expected:
            raise ValueError("statistics_fingerprint disagrees with train-only values")

    def identity_payload(self) -> dict[str, object]:
        return {
            "variant": self.variant.value,
            "task_id": self.task_id,
            "image": self.image.to_dict(),
            "state": self.state.to_dict(),
            "action": self.action.to_dict(),
            "leakage_audit": self.leakage_audit.to_dict(),
            "lerobot_version": self.lerobot_version,
            "torch_version": self.torch_version,
            "algorithm": self.algorithm,
            "schema_version": self.schema_version,
            "task_onehot_mapping_version": self.task_onehot_mapping_version,
            "onehot_suffix_policy": self.onehot_suffix_policy,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.identity_payload(), "statistics_fingerprint": self.statistics_fingerprint}

    def to_processor_stats(self) -> dict[str, dict[str, torch.Tensor]]:
        """Return tensors accepted by LeRobot 0.6.0's public ACT processors."""
        result: dict[str, dict[str, torch.Tensor]] = {}
        for feature in (self.image, self.state, self.action):
            shape = (
                (len(feature.components), 1, 1)
                if feature is self.image
                else (len(feature.components),)
            )
            result[feature.feature_name] = {
                "min": torch.tensor(feature.minimum, dtype=torch.float32).reshape(shape),
                "max": torch.tensor(feature.maximum, dtype=torch.float32).reshape(shape),
                "mean": torch.tensor(feature.mean, dtype=torch.float32).reshape(shape),
                "std": torch.tensor(feature.std, dtype=torch.float32).reshape(shape),
                "count": torch.tensor([feature.count], dtype=torch.int64),
            }
        return result


def split_manifest_digest(value: Mapping[str, object]) -> str:
    """Hash parsed split semantics, independent of paths and JSON whitespace."""
    if not isinstance(value, Mapping):
        raise TypeError("split manifest must be a mapping")
    return f"sha256:{sha256_hex(dict(value))}"


def build_dataset_views(manifest: LeRobotExportManifest) -> ActDatasetViews:
    """Build all split/per-task views and independently reject scene leakage."""
    if not isinstance(manifest, LeRobotExportManifest):
        raise TypeError("manifest must be a LeRobotExportManifest")
    records = manifest.episodes
    indices = tuple(item.lerobot_episode_index for item in records)
    if indices != tuple(range(len(records))):
        raise ActDataContractError("M3B episode indices are not contiguous and ordered")

    task_ids = set(CANONICAL_TASK_IDS)
    groups: dict[str, list[Any]] = defaultdict(list)
    scene_splits: dict[str, set[DatasetSplit]] = defaultdict(set)
    for record in records:
        if record.task_id not in task_ids:
            raise ActDataContractError(f"unknown stable M3B task ID {record.task_id!r}")
        if record.status is not EpisodeExportStatus.EXPORTED:
            raise ActDataContractError("M4 cannot consume a failed M3B episode")
        groups[record.source_scene_group_id].append(record)
        scene_splits[record.scene_id].add(record.split)
    if any(len(splits) != 1 for splits in scene_splits.values()):
        raise ActDataContractError("one physical scene leaks across M3B splits")
    for group_id, group in groups.items():
        if len(group) != 6 or {item.task_id for item in group} != task_ids:
            raise ActDataContractError(f"scene group {group_id!r} is not a complete six-task group")
        if len({item.split for item in group}) != 1 or len({item.scene_id for item in group}) != 1:
            raise ActDataContractError(f"scene group {group_id!r} is physically inconsistent")
    if manifest.split_assignments != build_split_assignments(
        tuple(groups), manifest.config.split_config
    ):
        raise ActDataContractError(
            "M3B split assignments differ from canonical digest-rank reconstruction"
        )

    global_views: dict[DatasetSplit, DatasetEpisodeView] = {}
    per_task: dict[str, dict[str, DatasetEpisodeView]] = {
        task_id: {} for task_id in CANONICAL_TASK_IDS
    }
    for split in DatasetSplit:
        selected = tuple(item.lerobot_episode_index for item in records if item.split is split)
        global_views[split] = DatasetEpisodeView(split=split, episode_indices=selected)
        for task_id in CANONICAL_TASK_IDS:
            task_indices = tuple(
                item.lerobot_episode_index
                for item in records
                if item.split is split and item.task_id == task_id
            )
            per_task[task_id][split.value] = DatasetEpisodeView(
                split=split,
                episode_indices=task_indices,
                task_id=task_id,
            )

    expected_counts = (
        (6, 0, 0, 1, 0, 0) if manifest.config.mode is ExportMode.SMOKE else (288, 36, 36, 48, 6, 6)
    )
    actual_counts = (
        len(global_views[DatasetSplit.TRAIN].episode_indices),
        len(global_views[DatasetSplit.VALIDATION].episode_indices),
        len(global_views[DatasetSplit.TEST].episode_indices),
        len(per_task[CANONICAL_TASK_IDS[0]][DatasetSplit.TRAIN.value].episode_indices),
        len(per_task[CANONICAL_TASK_IDS[0]][DatasetSplit.VALIDATION.value].episode_indices),
        len(per_task[CANONICAL_TASK_IDS[0]][DatasetSplit.TEST.value].episode_indices),
    )
    if actual_counts != expected_counts:
        raise ActDataContractError(
            "M3B global or per-task split counts differ from the fixed contract"
        )
    for task_id in CANONICAL_TASK_IDS[1:]:
        if (
            tuple(len(per_task[task_id][split.value].episode_indices) for split in DatasetSplit)
            != expected_counts[3:]
        ):
            raise ActDataContractError("M3B per-task splits are not balanced")

    return ActDatasetViews(
        train=global_views[DatasetSplit.TRAIN],
        validation=global_views[DatasetSplit.VALIDATION],
        test=global_views[DatasetSplit.TEST],
        per_task=per_task,
        task_id_by_episode={item.lerobot_episode_index: item.task_id for item in records},
        scene_group_id_by_episode={
            item.lerobot_episode_index: item.source_scene_group_id for item in records
        },
    )


def load_completed_m3b_dataset(
    dataset_root: str | Path,
    *,
    require_full: bool = True,
    validate_storage: bool = True,
) -> CompletedM3BDataset:
    """Gate one completed local M3B root without opening its M3A source."""
    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise ActDataContractError(f"M3B dataset root does not exist: {root}")
    sidecar = root / SIDECAR_DIRECTORY
    required = {
        "completion": root / COMPLETION_MARKER,
        "config": sidecar / "export_config.json",
        "manifest": sidecar / "export_manifest.json",
        "mapping": sidecar / "source_episode_mapping.json",
        "split": sidecar / "split_manifest.json",
        "summary": sidecar / "dataset_summary.json",
        "validation": sidecar / "validation_report.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise ActDataContractError("completed M3B sidecars are missing: " + ", ".join(missing))
    payloads = {name: _read_json(path) for name, path in required.items()}
    try:
        config = LeRobotExportConfig.from_dict(payloads["config"])
        manifest = LeRobotExportManifest.from_dict(payloads["manifest"])
        summary = LeRobotDatasetSummary.from_dict(payloads["summary"])
        validation = LeRobotValidationReport.from_dict(payloads["validation"])
    except Exception as error:
        raise ActDataContractError(f"invalid typed M3B completion evidence: {error}") from error
    expected_completion = {
        "schema_version": COMPLETION_SCHEMA,
        "export_fingerprint": manifest.export_fingerprint,
    }
    if payloads["completion"] != expected_completion:
        raise ActDataContractError("M3B completion marker disagrees with the export manifest")
    if config.portable_dict() != manifest.config.portable_dict():
        raise ActDataContractError("M3B export config and manifest disagree")
    if not manifest.finalized or not validation.passed:
        raise ActDataContractError("M4 requires a finalized and independently passing M3B export")
    if (
        summary.export_fingerprint != manifest.export_fingerprint
        or validation.export_fingerprint != manifest.export_fingerprint
    ):
        raise ActDataContractError("M3B summary/validation fingerprints disagree with the manifest")
    if summary.mode is not manifest.config.mode or validation.mode is not manifest.config.mode:
        raise ActDataContractError("M3B summary/validation modes disagree with the manifest")
    manifest_task_counts = Counter(item.task_id for item in manifest.episodes)
    manifest_group_counts = Counter(item.split.value for item in manifest.split_assignments)
    manifest_episode_counts = Counter(item.split.value for item in manifest.episodes)
    if (
        summary.total_scene_groups != len(manifest.split_assignments)
        or summary.total_episodes != len(manifest.episodes)
        or summary.total_frames != sum(item.output_frame_count for item in manifest.episodes)
        or dict(summary.task_episode_counts) != dict(manifest_task_counts)
        or dict(summary.split_scene_group_counts)
        != {split.value: manifest_group_counts[split.value] for split in DatasetSplit}
        or dict(summary.split_episode_counts)
        != {split.value: manifest_episode_counts[split.value] for split in DatasetSplit}
    ):
        raise ActDataContractError("M3B summary counts disagree with the export manifest")
    if require_full and manifest.config.mode is not ExportMode.FULL:
        raise ActDataContractError("full M4 experiments require a full 360-episode M3B export")
    if require_full and (
        summary.total_scene_groups != 60
        or summary.total_episodes != 360
        or dict(summary.split_scene_group_counts) != {"train": 48, "validation": 6, "test": 6}
        or dict(summary.split_episode_counts) != {"train": 288, "validation": 36, "test": 36}
        or set(summary.task_episode_counts.values()) != {60}
    ):
        raise ActDataContractError("full M3B counts differ from the fixed M4 input gate")
    if (
        len(validation.source_alignments) != summary.total_episodes
        or len(validation.video_results) != summary.total_episodes
    ):
        raise ActDataContractError("M3B validation evidence does not cover every episode")
    expected_episode_indices = set(range(summary.total_episodes))
    if {
        item.lerobot_episode_index for item in validation.source_alignments
    } != expected_episode_indices or {
        item.lerobot_episode_index for item in validation.video_results
    } != expected_episode_indices:
        raise ActDataContractError("M3B validation episode indices are missing or duplicated")
    source_ids = {item.lerobot_episode_index: item.source_episode_id for item in manifest.episodes}
    if any(
        source_ids[item.lerobot_episode_index] != item.source_episode_id
        for item in validation.source_alignments
    ):
        raise ActDataContractError("M3B source-alignment IDs disagree with the manifest")
    if not all(item.passed for item in (*validation.source_alignments, *validation.video_results)):
        raise ActDataContractError("M3B contains a failed source-alignment or video result")

    expected_mapping = {
        "schema_version": "langmani-m3b-source-mapping-v1",
        "export_fingerprint": manifest.export_fingerprint,
        "episodes": [item.to_dict() for item in manifest.episodes],
    }
    if payloads["mapping"] != expected_mapping:
        raise ActDataContractError("M3B source mapping is not an exact manifest cross-reference")
    expected_split = {
        "schema_version": SPLIT_MANIFEST_SCHEMA,
        "export_fingerprint": manifest.export_fingerprint,
        "split_config": manifest.config.split_config.to_dict(),
        "scene_group_assignments": [item.to_dict() for item in manifest.split_assignments],
        "episode_indices": {
            split.value: [
                item.lerobot_episode_index for item in manifest.episodes if item.split is split
            ]
            for split in DatasetSplit
        },
    }
    if payloads["split"] != expected_split:
        raise ActDataContractError("M3B split sidecar is not an exact manifest cross-reference")
    views = build_dataset_views(manifest)
    if validate_storage:
        _validate_local_storage(root, summary)
    return CompletedM3BDataset(
        root=root,
        manifest=manifest,
        summary=summary,
        validation=validation,
        split_manifest_digest=split_manifest_digest(cast(Mapping[str, object], payloads["split"])),
        views=views,
    )


def load_lerobot_episode_view(
    completed: CompletedM3BDataset,
    view: DatasetEpisodeView,
    *,
    policy_config: object | None,
    include_delta_timestamps: bool,
    return_uint8: bool = False,
) -> LoadedActDataset:
    """Open one explicit view with LeRobot's public episode/delta APIs."""
    if not isinstance(completed, CompletedM3BDataset) or not isinstance(view, DatasetEpisodeView):
        raise TypeError("completed and view must use the project-owned M4 contracts")
    if not view.episode_indices:
        raise ActDataContractError("cannot load an empty episode view")
    if not set(view.episode_indices) <= set(completed.views.task_id_by_episode):
        raise ActDataContractError("episode view is not part of the completed M3B manifest")
    authoritative = completed.views.for_split(view.split, task_id=view.task_id)
    if not set(view.episode_indices) <= set(authoritative.episode_indices):
        raise ActDataContractError(
            "episode view indices disagree with the authoritative M3B split/task view"
        )
    if include_delta_timestamps and policy_config is None:
        raise ValueError("policy_config is required when ACT delta timestamps are enabled")

    LeRobotDataset, LeRobotDatasetMetadata, resolve_delta_timestamps = _lerobot_public_apis()
    with _offline_hub_environment():
        dataset_meta = LeRobotDatasetMetadata(
            completed.manifest.repo_id,
            completed.root,
        )
        deltas = (
            resolve_delta_timestamps(policy_config, dataset_meta)
            if include_delta_timestamps
            else None
        )
        dataset = LeRobotDataset(
            repo_id=completed.manifest.repo_id,
            root=completed.root,
            episodes=list(view.episode_indices),
            delta_timestamps=deltas,
            video_backend="pyav",
            return_uint8=return_uint8,
            download_videos=False,
        )
    if int(dataset.fps) != 20:
        raise ActDataContractError("public LeRobot loader did not retain the 20 FPS contract")
    loaded_indices = {
        _scalar_int(value, "episode_index") for value in dataset.hf_dataset.unique("episode_index")
    }
    if loaded_indices != set(view.episode_indices):
        raise ActDataContractError("public LeRobot episode filter returned the wrong episode set")
    normalized_deltas = MappingProxyType(
        {key: tuple(float(item) for item in values) for key, values in (deltas or {}).items()}
    )
    if include_delta_timestamps:
        action_indices = tuple(cast(Any, policy_config).action_delta_indices or ())
        expected_action = tuple(index / 20 for index in action_indices)
        if normalized_deltas.get(ACTION_FEATURE_KEY) != expected_action:
            raise ActDataContractError("public ACT action delta timestamps differ from the config")
        observation_indices = cast(Any, policy_config).observation_delta_indices
        observation_keys = {key for key in normalized_deltas if key.startswith("observation.")}
        if observation_indices is None and observation_keys:
            raise ActDataContractError("ACT unexpectedly requested future/past observation frames")
    return LoadedActDataset(
        dataset=dataset,
        episode_indices=view.episode_indices,
        delta_timestamps=normalized_deltas,
    )


def compute_train_only_statistics(
    dataset: Sequence[Mapping[str, object]],
    *,
    train_episode_indices: Sequence[int],
    validation_episode_indices: Sequence[int],
    test_episode_indices: Sequence[int],
    task_id_by_episode: Mapping[int, str],
    variant: ActVariant,
    task_id: str | None = None,
) -> TrainOnlyStatistics:
    """Stream current-frame train data only; a delta-expanded dataset is rejected."""
    selected_train = _ordered_indices(
        train_episode_indices, "train_episode_indices", require_nonempty=True
    )
    selected_validation = _ordered_indices(validation_episode_indices, "validation_episode_indices")
    selected_test = _ordered_indices(test_episode_indices, "test_episode_indices")
    audit = StatisticsLeakageAudit(
        source_episode_indices=selected_train,
        train_episode_indices=selected_train,
        validation_episode_indices=selected_validation,
        test_episode_indices=selected_test,
    )
    selected_variant = ActVariant(variant)
    mapping = dict(task_id_by_episode)
    if set(selected_train) - set(mapping):
        raise ActDataContractError("training episode task mapping is incomplete")
    selected_task_ids = {mapping[index] for index in selected_train}
    if not selected_task_ids <= set(CANONICAL_TASK_IDS):
        raise ActDataContractError("training view contains an unknown stable task ID")
    if selected_variant is ActVariant.PER_TASK:
        if task_id not in CANONICAL_TASK_IDS or selected_task_ids != {task_id}:
            raise ActDataContractError("per-task statistics must use only the selected TaskSpec")
    elif task_id is not None:
        raise ActDataContractError("mixed statistics must not select one TaskSpec")

    image_stats = _RunningVectorStatistics(3)
    state_stats = _RunningVectorStatistics(ACT_STATE_COMPONENTS)
    action_stats = _RunningVectorStatistics(ACT_ACTION_COMPONENTS)
    seen_episodes: set[int] = set()
    frame_indices_by_episode: dict[int, list[int]] = defaultdict(list)
    seen_frames: set[tuple[int, int]] = set()
    frame_count = 0
    for row in dataset:
        if not isinstance(row, Mapping):
            raise TypeError("LeRobot dataset rows must be mappings")
        episode_index = _scalar_int(row.get("episode_index"), "episode_index")
        if episode_index not in selected_train:
            raise ActDataContractError(
                "statistics dataset exposed a validation/test/unknown episode"
            )
        frame_index = _scalar_int(row.get("frame_index"), "frame_index")
        frame_key = (episode_index, frame_index)
        if frame_key in seen_frames:
            raise ActDataContractError("statistics dataset contains a duplicate episode/frame row")
        seen_frames.add(frame_key)
        frame_indices_by_episode[episode_index].append(frame_index)
        image = _policy_image_float01(row.get(IMAGE_FEATURE_KEY))
        state = _float32_vector(row.get(STATE_FEATURE_KEY), ACT_STATE_COMPONENTS, STATE_FEATURE_KEY)
        action = _float32_vector(
            row.get(ACTION_FEATURE_KEY), ACT_ACTION_COMPONENTS, ACTION_FEATURE_KEY
        )
        image_stats.update(np.moveaxis(image, 0, -1).reshape(-1, 3))
        state_stats.update(state.reshape(1, -1))
        action_stats.update(action.reshape(1, -1))
        seen_episodes.add(episode_index)
        frame_count += 1
    if frame_count == 0 or seen_episodes != set(selected_train):
        raise ActDataContractError("statistics rows do not cover every selected training episode")
    if any(
        sorted(indices) != list(range(len(indices)))
        for indices in frame_indices_by_episode.values()
    ):
        raise ActDataContractError("statistics rows skip or duplicate an episode frame index")

    image = image_stats.finish(IMAGE_FEATURE_KEY, IMAGE_COMPONENTS)
    state = state_stats.finish(STATE_FEATURE_KEY, PANDA_POLICY_STATE_COMPONENTS)
    if selected_variant is ActVariant.MIXED_TASK_ONEHOT:
        suffix = tuple(
            f"CanonicalTaskOneHotV0[{index}]::{task_id}"
            for index, task_id in enumerate(CANONICAL_TASK_IDS)
        )
        state = FeatureStatistics(
            feature_name=state.feature_name,
            components=(*state.components, *suffix),
            count=state.count,
            minimum=(*state.minimum, *(0.0 for _ in suffix)),
            maximum=(*state.maximum, *(1.0 for _ in suffix)),
            mean=(*state.mean, *(0.0 for _ in suffix)),
            std=(*state.std, *(1.0 for _ in suffix)),
        )
    action = action_stats.finish(ACTION_FEATURE_KEY, PANDA_ACTION_COMPONENTS)
    return TrainOnlyStatistics(
        variant=selected_variant,
        task_id=task_id,
        image=image,
        state=state,
        action=action,
        leakage_audit=audit,
        lerobot_version=metadata.version("lerobot"),
        torch_version=torch.__version__,
    )


class _RunningVectorStatistics:
    def __init__(self, dimension: int) -> None:
        self.count = 0
        self.mean = np.zeros(dimension, dtype=np.float64)
        self.m2 = np.zeros(dimension, dtype=np.float64)
        self.minimum = np.full(dimension, np.inf, dtype=np.float64)
        self.maximum = np.full(dimension, -np.inf, dtype=np.float64)

    def update(self, value: NDArray[np.floating[Any]]) -> None:
        batch = np.asarray(value, dtype=np.float64)
        if batch.ndim != 2 or batch.shape[1] != self.mean.size or batch.shape[0] < 1:
            raise ValueError("statistics updates must be nonempty two-dimensional vectors")
        if not np.all(np.isfinite(batch)):
            raise ValueError("statistics input contains non-finite values")
        batch_count = batch.shape[0]
        batch_mean = batch.mean(axis=0)
        batch_m2 = np.square(batch - batch_mean).sum(axis=0)
        if self.count == 0:
            self.mean = batch_mean
            self.m2 = batch_m2
            self.count = batch_count
        else:
            total = self.count + batch_count
            delta = batch_mean - self.mean
            self.mean += delta * batch_count / total
            self.m2 += batch_m2 + np.square(delta) * self.count * batch_count / total
            self.count = total
        self.minimum = np.minimum(self.minimum, batch.min(axis=0))
        self.maximum = np.maximum(self.maximum, batch.max(axis=0))

    def finish(self, feature_name: str, components: tuple[str, ...]) -> FeatureStatistics:
        if self.count < 1:
            raise ActDataContractError("cannot finalize empty feature statistics")
        return FeatureStatistics(
            feature_name=feature_name,
            components=components,
            count=self.count,
            minimum=tuple(float(item) for item in self.minimum),
            maximum=tuple(float(item) for item in self.maximum),
            mean=tuple(float(item) for item in self.mean),
            std=tuple(float(item) for item in np.sqrt(self.m2 / self.count)),
        )


def _validate_local_storage(root: Path, summary: LeRobotDatasetSummary) -> None:
    info_path = root / "meta" / "info.json"
    stats_path = root / "meta" / "stats.json"
    task_path = root / "meta" / "tasks.parquet"
    required = (info_path, stats_path, task_path)
    if any(not path.is_file() for path in required):
        raise ActDataContractError("completed M3B root is missing LeRobot metadata")
    info = _read_json(info_path)
    if info.get("codebase_version") != EXPECTED_LEROBOT_CODEBASE_VERSION:
        raise ActDataContractError("LeRobot dataset codebase_version must be v3.0")
    if int(cast(Any, info.get("fps", -1))) != 20:
        raise ActDataContractError("LeRobot metadata FPS differs from M3B")
    features = info.get("features")
    if not isinstance(features, Mapping):
        raise ActDataContractError("LeRobot metadata features must be a mapping")
    if set(features) != POLICY_FEATURE_KEYS | LEROBOT_MANAGED_FEATURE_KEYS:
        raise ActDataContractError("LeRobot metadata feature allowlist differs from M3B")
    parquet_files = tuple(sorted(root.rglob("*.parquet")))
    data_files = tuple(path for path in parquet_files if "data" in path.relative_to(root).parts)
    episode_files = tuple(
        path for path in parquet_files if path.relative_to(root).parts[:2] == ("meta", "episodes")
    )
    video_files = tuple(sorted(root.rglob("*.mp4")))
    if not data_files or not episode_files or not video_files:
        raise ActDataContractError("completed M3B root lacks local Parquet or video files")
    if (
        len(parquet_files) != summary.parquet_file_count
        or len(video_files) != summary.video_file_count
    ):
        raise ActDataContractError("local Parquet/video counts differ from the M3B summary")
    if any(path.is_symlink() for path in (*required, *parquet_files, *video_files)):
        raise ActDataContractError("M4 refuses symlinked M3B-owned files")
    try:
        import av
        import pyarrow.parquet as pq

        data_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in data_files)
        for path in parquet_files:
            pq.ParquetFile(path)
        decoded_frames = 0
        for path in video_files:
            with av.open(str(path), mode="r") as container:
                decoded_frames += sum(1 for _ in container.decode(video=0))
    except Exception as error:
        raise ActDataContractError(f"local M3B Parquet/video validation failed: {error}") from error
    if data_rows != summary.total_frames or decoded_frames != summary.total_frames:
        raise ActDataContractError("local Parquet/video frame totals differ from the M3B summary")


def _policy_image_float01(value: object) -> NDArray[np.float32]:
    array = _as_numpy(value)
    if array.shape != (3, 256, 256):
        raise ValueError("base_camera statistics input must have shape (3, 256, 256)")
    if array.dtype == np.dtype(np.uint8):
        return np.asarray(array, dtype=np.float32) / np.float32(255.0)
    if array.dtype != np.dtype(np.float32):
        raise ValueError("base_camera must be uint8 or decoded float32")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("decoded base_camera float values must be finite in [0, 1]")
    return np.asarray(array, dtype=np.float32)


def _float32_vector(value: object, length: int, name: str) -> NDArray[np.float32]:
    array = _as_numpy(value)
    if array.dtype != np.dtype(np.float32) or array.shape != (length,):
        raise ValueError(f"{name} must be a current-frame float32 vector with shape ({length},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return np.asarray(array, dtype=np.float32)


def _as_numpy(value: object) -> NDArray[Any]:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    raise TypeError("policy features must be numpy arrays or torch tensors")


def _scalar_int(value: object, name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar")
        value = value.detach().cpu().item()
    elif isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"{name} must be scalar")
        value = value.reshape(-1)[0].item()
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer scalar")
    return int(value)


def _ordered_indices(
    values: Sequence[int], name: str, *, require_nonempty: bool = False
) -> tuple[int, ...]:
    result = tuple(values)
    if require_nonempty and not result:
        raise ValueError(f"{name} must not be empty")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in result):
        raise TypeError(f"{name} must contain non-negative integers")
    if len(set(result)) != len(result) or tuple(sorted(result)) != result:
        raise ValueError(f"{name} must be unique and sorted")
    return result


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ActDataContractError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ActDataContractError(f"JSON root must be an object: {path}")
    return cast(Mapping[str, object], value)


def _lerobot_public_apis() -> tuple[Any, Any, Any]:
    try:
        from lerobot.datasets import (
            LeRobotDataset,
            LeRobotDatasetMetadata,
            resolve_delta_timestamps,
        )
    except (ImportError, RuntimeError, OSError) as error:
        raise ActDataContractError("LeRobot 0.6.0 dataset APIs are unavailable") from error
    if metadata.version("lerobot") != "0.6.0":
        raise ActDataContractError("M4 supports exactly LeRobot 0.6.0")
    return LeRobotDataset, LeRobotDatasetMetadata, resolve_delta_timestamps


@contextmanager
def _offline_hub_environment() -> Iterator[None]:
    names = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update({name: "1" for name in names})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


__all__ = [
    "ActDataContractError",
    "ActDatasetViews",
    "CompletedM3BDataset",
    "DatasetEpisodeView",
    "FeatureStatistics",
    "LoadedActDataset",
    "StatisticsLeakageAudit",
    "TrainOnlyStatistics",
    "build_dataset_views",
    "compute_train_only_statistics",
    "load_completed_m3b_dataset",
    "load_lerobot_episode_view",
    "validate_temporal_episode_boundaries",
    "split_manifest_digest",
]
