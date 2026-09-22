"""Read-only two-goal view over the frozen, authoritative six-task M3B dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from langmani.policies.m4a_data import (
    IMAGE,
    STATE,
    digest,
    file_digest,
    load_manifest,
    local_dataset_only,
    numpy,
    read_json,
    validate_root,
)
from langmani.policies.m4b_protocol import GOALS


def make_split(manifest: Any, validation: dict[str, Any]) -> dict[str, Any]:
    if (
        validation.get("passed") is not True
        or validation["source_id"] != manifest.export_fingerprint
    ):
        raise ValueError("source validation is absent or mismatched")
    records = sorted(
        (ep for ep in manifest.episodes if f"{ep.target_object_id}:{ep.target_bin_id}" in GOALS),
        key=lambda ep: ep.lerobot_episode_index,
    )
    groups: dict[str, list[Any]] = {}
    for ep in records:
        groups.setdefault(ep.source_scene_group_id, []).append(ep)
    if not groups or any(
        len(eps) != 2
        or {f"{ep.target_object_id}:{ep.target_bin_id}" for ep in eps} != set(GOALS)
        or len({ep.split for ep in eps}) != 1
        for eps in groups.values()
    ):
        raise ValueError("incomplete or cross-split semantic goal pair")
    partitions = {
        name: [ep.lerobot_episode_index for ep in records if ep.split.value == source]
        for name, source in (
            ("train", "train"),
            ("held_out", "validation"),
            ("excluded_test", "test"),
        )
    }
    if not all(partitions.values()):
        raise ValueError("M4B requires nonempty inherited train/validation/test partitions")
    result = {
        "schema_version": "langmani-m4b-split-v1",
        "semantic_goals": list(GOALS),
        "repo_id": manifest.repo_id,
        "source_id": manifest.export_fingerprint,
        "content_sha256": validation["content_sha256"],
        "dataset_total_frames": validation["num_frames"],
        "algorithm": "inherited M3B scene digest rank",
        "seed": manifest.config.split_config.split_seed,
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
                "semantic_goal": f"{ep.target_object_id}:{ep.target_bin_id}",
                "instruction": ep.canonical_instruction,
            }
            for ep in records
        ],
        "all_source_scene_seeds": sorted({ep.source_scene_seed for ep in manifest.episodes}),
    }
    return {**result, "split_sha256": digest(result)}


@local_dataset_only()
def validate_dataset(root: Path) -> dict[str, Any]:
    from lerobot.datasets import LeRobotDataset

    report = validate_root(root)
    manifest = load_manifest(root)
    split = make_split(manifest, report)
    dataset = LeRobotDataset(manifest.repo_id, root=root, video_backend="pyav", return_uint8=True)
    pairs = []
    for group in sorted({r["scene_group_id"] for r in split["episodes"]}):
        eps = [r for r in split["episodes"] if r["scene_group_id"] == group]
        first = [
            dataset[int(dataset.meta.episodes[r["episode_id"]]["dataset_from_index"])] for r in eps
        ]
        if not np.array_equal(numpy(first[0][STATE]), numpy(first[1][STATE])):
            raise ValueError("paired goal demonstrations start with different robot qpos")
        pairs.append(
            {
                "scene_group_id": group,
                "episode_ids": [r["episode_id"] for r in eps],
                "qpos_equal": True,
                "decoded_first_frame_rgb_mae": float(
                    np.abs(
                        numpy(first[0][IMAGE]).astype(float) - numpy(first[1][IMAGE]).astype(float)
                    ).mean()
                ),
            }
        )
    report["m4b_pair_audit"] = {
        "pairs": pairs,
        "rgb_note": "separately encoded lossy videos; exact native reset RGB is checked by the evaluation gate",
    }
    return report


def check_split(
    root: Path, split_path: Path, validation_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reuse exhaustive decoded validation only after rehashing every exact source byte."""
    report = read_json(validation_path)
    actual = {
        p.relative_to(root).as_posix(): file_digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    if (
        report.get("passed") is not True
        or actual != report.get("files")
        or digest(actual) != report.get("content_sha256")
        or "m4b_pair_audit" not in report
    ):
        raise ValueError("dataset differs from exhaustive M4B validation; rerun validate")
    expected = make_split(load_manifest(root), report)
    if read_json(split_path) != expected:
        raise ValueError("M4B split differs from validated immutable dataset")
    return expected, report
