"""Read-only, exhaustive M4A data checks and an explicit single-task episode view."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from langmani.datasets.lerobot_types import (
    ACTION_FEATURE_KEY as ACTION,
)
from langmani.datasets.lerobot_types import (
    IMAGE_FEATURE_KEY as IMAGE,
)
from langmani.datasets.lerobot_types import (
    LEROBOT_MANAGED_FEATURE_KEYS,
    POLICY_FEATURE_KEYS,
    FeatureContract,
    LeRobotExportManifest,
)
from langmani.datasets.lerobot_types import (
    STATE_FEATURE_KEY as STATE,
)
from langmani.datasets.splits import build_split_assignments
from langmani.environments.specs import TaskSpec, canonical_instruction, stable_task_id

TASK = TaskSpec("red_cube", "left_bin", "canonical_v0")
TASK_ID = stable_task_id(TASK)
TASK_TEXT = canonical_instruction(TASK)
CANONICAL_TEXTS = {
    canonical_instruction(TaskSpec(obj, dest, "canonical_v0"))
    for obj in ("red_cube", "green_cube", "blue_cube")
    for dest in ("left_bin", "right_bin")
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def write_json(path: Path, value: object) -> None:
    """Atomically publish small artifacts, rejecting nonfinite metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_lerobot() -> None:
    if version("lerobot") != "0.6.0":
        raise RuntimeError("M4A is verified against lerobot==0.6.0; refusing API drift")


@contextmanager
def local_dataset_only() -> Any:
    """LeRobot may auto-download missing shards; consumer commands must instead reject them."""
    from huggingface_hub import constants
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

    def reject_download(*args: Any, **kwargs: Any) -> None:
        raise ValueError("incomplete local dataset; automatic download/repair is forbidden")

    with (
        patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}),
        patch.object(constants, "HF_HUB_OFFLINE", True),
        patch.object(LeRobotDataset, "_download", reject_download),
        patch.object(LeRobotDatasetMetadata, "_pull_from_repo", reject_download),
    ):
        yield


def numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def scalar(value: Any, *, integer: bool = False) -> int | float:
    array = numpy(value)
    if array.size != 1 or not np.isfinite(array).all():
        raise ValueError("expected one finite scalar")
    result = float(array.item())
    if integer and (array.dtype.kind not in "iu" or result < 0):
        raise ValueError("expected nonnegative integer metadata")
    return int(result) if integer else result


class Moments:
    """Stable population statistics; never accumulates a dataset in memory."""

    def __init__(self, dimensions: int) -> None:
        self.count = 0
        self.mean = np.zeros(dimensions, dtype=np.float64)
        self.m2 = np.zeros(dimensions, dtype=np.float64)
        self.minimum = np.full(dimensions, np.inf)
        self.maximum = np.full(dimensions, -np.inf)

    def update(self, values: np.ndarray) -> None:
        values = values.reshape(-1, len(self.mean)).astype(np.float64)
        size = len(values)
        delta = values.mean(axis=0) - self.mean
        total = self.count + size
        self.m2 += ((values - values.mean(axis=0)) ** 2).sum(axis=0)
        self.m2 += delta**2 * self.count * size / total
        self.mean += delta * size / total
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))
        self.count = total

    def report(self) -> dict[str, Any]:
        if not self.count:
            raise ValueError("empty statistics")
        return {
            "count": self.count,
            "mean": self.mean.tolist(),
            "std": np.sqrt(np.maximum(self.m2 / self.count, 0)).tolist(),
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
        }


def validate_features(features: Mapping[str, Any]) -> None:
    expected = POLICY_FEATURE_KEYS | LEROBOT_MANAGED_FEATURE_KEYS
    if set(features) != expected:
        raise ValueError(
            f"feature allowlist mismatch (possible privileged leakage): {set(features)}"
        )
    for key, contract in FeatureContract().to_lerobot_features().items():
        actual = features[key]
        for field in ("shape", "names"):
            if tuple(actual[field]) != tuple(contract[field]):
                raise ValueError(f"{key}: wrong {field}")
        if actual["dtype"] != contract["dtype"]:
            raise ValueError(f"{key}: wrong storage dtype")


def validate_rows(dataset: Any) -> dict[str, Any]:
    """Decode and inspect EVERY row, including every camera frame, without raw M3A access."""
    validate_features(dataset.features)
    if dataset.fps != 20 or dataset.meta.info.codebase_version != "v3.0":
        raise ValueError("expected LeRobot v3 at exactly 20 FPS")
    if set(dataset.meta.camera_keys) != {IMAGE}:
        raise ValueError("camera configuration must contain only base_camera")
    if set(dataset.meta.tasks.index) != CANONICAL_TEXTS:
        raise ValueError("six canonical language sentences must remain in task metadata")
    stats = {STATE: Moments(9), ACTION: Moments(8)}
    episodes: list[dict[str, Any]] = []
    previous = -1
    frame = 0
    for index in range(len(dataset)):
        row = dataset[index]
        if set(row) != POLICY_FEATURE_KEYS | LEROBOT_MANAGED_FEATURE_KEYS | {"task"}:
            raise ValueError(f"frame {index}: unexpected/missing keys or privileged leakage")
        ep = scalar(row["episode_index"], integer=True)
        if ep != previous:
            if ep != previous + 1:
                raise ValueError(f"frame {index}: noncontiguous episode boundary")
            frame = 0
            boundary = dataset.meta.episodes[ep]
            if boundary["dataset_from_index"] != index:
                raise ValueError(f"episode {ep}: start boundary mismatch")
            episodes.append({"episode_id": ep, "frames": 0, "task": row["task"]})
            previous = ep
        if scalar(row["index"], integer=True) != index:
            raise ValueError(f"frame {index}: global ordering mismatch")
        if scalar(row["frame_index"], integer=True) != frame:
            raise ValueError(f"frame {index}: frame ordering mismatch")
        if abs(scalar(row["timestamp"]) - frame / 20) > 1e-5:
            raise ValueError(f"frame {index}: timestamp/FPS mismatch")
        task_index = scalar(row["task_index"], integer=True)
        text = dataset.meta.tasks.iloc[task_index].name
        if row["task"] != text or text != episodes[-1]["task"] or text not in CANONICAL_TEXTS:
            raise ValueError(f"frame {index}: task identity mismatch")
        if list(dataset.meta.episodes[ep]["tasks"]) != [text]:
            raise ValueError(f"episode {ep}: task metadata mismatch")
        image = numpy(row[IMAGE])
        if image.shape != (3, 256, 256) or image.dtype != np.uint8:
            raise ValueError(f"frame {index}: expected uint8 CHW camera tensor [3,256,256]")
        for key, dim in ((STATE, 9), (ACTION, 8)):
            value = numpy(row[key])
            if value.dtype != np.float32 or value.shape != (dim,) or not np.isfinite(value).all():
                raise ValueError(f"frame {index}: {key} must be finite float32[{dim}]")
            stats[key].update(value)
        frame += 1
        episodes[-1]["frames"] += 1
        boundary = dataset.meta.episodes[ep]
        if index + 1 == boundary["dataset_to_index"] and frame != boundary["length"]:
            raise ValueError(f"episode {ep}: declared length mismatch")
    if (
        not episodes
        or len(dataset) != dataset.num_frames
        or len(episodes) != dataset.num_episodes
        or len(dataset) != dataset.meta.total_frames
        or len(episodes) != dataset.meta.total_episodes
    ):
        raise ValueError("empty dataset or episode/frame totals disagree")
    for ep in episodes:
        boundary = dataset.meta.episodes[ep["episode_id"]]
        if ep["frames"] != boundary["length"]:
            raise ValueError("episode length mismatch")
        if boundary["dataset_to_index"] - boundary["dataset_from_index"] != ep["frames"]:
            raise ValueError("episode end boundary mismatch")
    return {
        "schema_version": "langmani-m4a-validation-v1",
        "passed": True,
        "fps": 20,
        "camera": IMAGE,
        "image_dtype": "uint8",
        "image_shape": [3, 256, 256],
        "state_dimension": 9,
        "action_dimension": 8,
        "num_episodes": len(episodes),
        "num_frames": len(dataset),
        "episodes": episodes,
        "statistics": {key: value.report() for key, value in stats.items()},
    }


def load_manifest(root: Path) -> LeRobotExportManifest:
    manifest = LeRobotExportManifest.from_dict(read_json(root / "langmani/export_manifest.json"))
    marker = read_json(root / "langmani/complete.json")
    if not manifest.finalized or marker != {
        "schema_version": "langmani-m3b-completion-v1",
        "export_fingerprint": manifest.export_fingerprint,
    }:
        raise ValueError("dataset lacks a matching finalized M3B completion marker")
    accepted = read_json(root / "langmani/validation_report.json")
    if (
        accepted.get("passed") is not True
        or accepted.get("export_fingerprint") != manifest.export_fingerprint
    ):
        raise ValueError("dataset lacks an accepted matching M3B source-alignment report")
    for flag in (
        "source_alignment_validated",
        "video_decode_validated",
        "feature_schema_validated",
        "privileged_leakage_validated",
    ):
        if accepted.get(flag) is not True:
            raise ValueError(f"M3B acceptance report is incomplete: {flag}")
    expected = build_split_assignments(
        sorted({ep.source_scene_group_id for ep in manifest.episodes}), manifest.config.split_config
    )
    if tuple(manifest.split_assignments) != expected:
        raise ValueError("M3B scene split is not reproducible from its seed")
    by_group = {a.scene_group_id: a.split for a in expected}
    if any(ep.split != by_group[ep.source_scene_group_id] for ep in manifest.episodes):
        raise ValueError("episode crosses the M3B scene split")
    group_seeds: set[int] = set()
    for group in by_group:
        records = [ep for ep in manifest.episodes if ep.source_scene_group_id == group]
        seeds = {ep.source_scene_seed for ep in records}
        if len(records) != 6 or len(seeds) != 1 or seeds & group_seeds:
            raise ValueError(
                "scene/episode split leaks reset seeds or has an incomplete scene group"
            )
        group_seeds.update(seeds)
        if {ep.canonical_instruction for ep in records} != CANONICAL_TEXTS:
            raise ValueError("a scene group does not retain all six canonical tasks")
        for ep in records:
            task = TaskSpec(ep.target_object_id, ep.target_bin_id, ep.instruction_template_id)
            if ep.task_id != stable_task_id(
                task
            ) or ep.canonical_instruction != canonical_instruction(task):
                raise ValueError("M3B TaskSpec and language metadata disagree")
    return manifest


@local_dataset_only()
def validate_root(root: Path, repo_id: str | None = None) -> dict[str, Any]:
    require_lerobot()
    root = root.resolve(strict=True)
    manifest = load_manifest(root)
    if repo_id is not None and repo_id != manifest.repo_id:
        raise ValueError("repo id differs from the local M3B source identity")
    from lerobot.datasets import LeRobotDataset

    dataset = LeRobotDataset(manifest.repo_id, root=root, video_backend="pyav", return_uint8=True)
    report = validate_rows(dataset)
    report["videos"] = validate_videos(root, report["num_frames"])
    mappings = {ep.lerobot_episode_index: ep for ep in manifest.episodes}
    if set(mappings) != set(range(report["num_episodes"])):
        raise ValueError("M3B episode mapping differs from actual data")
    for ep in report["episodes"]:
        source = mappings[ep["episode_id"]]
        if ep["frames"] != source.output_frame_count or ep["task"] != source.canonical_instruction:
            raise ValueError("M3B frame/task provenance disagrees with decoded data")
    inventory = {
        str(p.relative_to(root).as_posix()): file_digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    report.update(
        repo_id=manifest.repo_id,
        dataset_version="v3.0",
        source_id=manifest.export_fingerprint,
        content_sha256=digest(inventory),
        files=inventory,
    )
    return report


def validate_videos(root: Path, expected_frames: int) -> dict[str, int]:
    """Inspect encoded timing too: a 20-Hz table must not mask a wrong-FPS video."""
    import av

    paths = sorted((root / "videos").rglob("*.mp4"))
    total = 0
    if not paths:
        raise ValueError("no policy camera video files")
    for path in paths:
        with av.open(str(path)) as container:
            if len(container.streams.video) != 1:
                raise ValueError(f"{path}: expected one RGB video stream")
            stream = container.streams.video[0]
            if stream.average_rate != 20 or (stream.width, stream.height) != (256, 256):
                raise ValueError(f"{path}: encoded camera FPS or dimensions disagree")
            previous: float | None = None
            file_frames = 0
            for frame in container.decode(stream):
                if frame.time is None or not np.isfinite(frame.time):
                    raise ValueError(f"{path}: invalid encoded frame timestamp")
                timestamp = float(frame.time)
                if previous is not None and abs(timestamp - previous - 1 / 20) > 1e-5:
                    raise ValueError(f"{path}: encoded frame ordering/FPS mismatch")
                previous = timestamp
                file_frames += 1
            if not file_frames:
                raise ValueError(f"{path}: empty video")
            total += file_frames
    if total != expected_frames:
        raise ValueError("encoded video frame count differs from dataset rows")
    return {"files": len(paths), "decoded_frames": total}


def make_split(manifest: LeRobotExportManifest, validation: dict[str, Any]) -> dict[str, Any]:
    """Inherit the M3B scene split; never reassign frames or touch its held-back test partition."""
    records = sorted(
        (ep for ep in manifest.episodes if ep.task_id == TASK_ID),
        key=lambda ep: ep.lerobot_episode_index,
    )
    partitions = {
        name: [ep.lerobot_episode_index for ep in records if ep.split.value == source]
        for name, source in (
            ("train", "train"),
            ("held_out", "validation"),
            ("excluded_test", "test"),
        )
    }
    if not partitions["train"] or not partitions["held_out"]:
        raise ValueError(
            "M4A needs multiple scenes and a nonempty M3B validation split; one-group smoke data is insufficient"
        )
    result = {
        "schema_version": "langmani-m4a-split-v1",
        "task_name": TASK_ID,
        "task": TASK.to_dict(),
        "seed": manifest.config.split_config.split_seed,
        "algorithm": "inherited M3B scene digest rank",
        "repo_id": manifest.repo_id,
        "dataset_version": "v3.0",
        "source_id": manifest.export_fingerprint,
        "content_sha256": validation["content_sha256"],
        "dataset_total_frames": validation["num_frames"],
        "total_frames": sum(ep.output_frame_count for ep in records),
        **{f"{key}_episode_ids": ids for key, ids in partitions.items()},
        **{
            f"{key}_frames": sum(
                ep.output_frame_count for ep in records if ep.lerobot_episode_index in ids
            )
            for key, ids in partitions.items()
        },
        "episodes": [
            {
                "episode_id": ep.lerobot_episode_index,
                "raw_episode_id": ep.source_episode_id,
                "scene_seed": ep.source_scene_seed,
                "scene_group_id": ep.source_scene_group_id,
                "split": ep.split.value,
                "frames": ep.output_frame_count,
            }
            for ep in records
        ],
        "all_source_scene_seeds": sorted({ep.source_scene_seed for ep in manifest.episodes}),
    }
    return {**result, "split_sha256": digest(result)}


def check_split(root: Path, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    report = validate_root(root)
    expected = make_split(load_manifest(root), report)
    if read_json(path) != expected:
        raise ValueError("split manifest changed or no longer matches validated source bytes")
    return expected, report
