"""Completed-M3B views, public temporal loading, and train-only statistics tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import langmani.policies.act_data as act_data
from langmani.datasets.lerobot_types import (
    LEROBOT_MANAGED_FEATURE_KEYS,
    POLICY_FEATURE_KEYS,
    DatasetSplit,
    EpisodeExportRecord,
    ExportMode,
    LeRobotDatasetSummary,
    LeRobotExportConfig,
    LeRobotExportManifest,
    LeRobotValidationReport,
    SourceAlignmentResult,
    SplitConfig,
    VideoValidationResult,
    stable_export_fingerprint,
)
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.datasets.splits import assignment_map, build_split_assignments
from langmani.environments.specs import canonical_instruction, stable_task_id
from langmani.policies.act_conditioning import CANONICAL_TASK_IDS
from langmani.policies.act_data import (
    ActDataContractError,
    CompletedM3BDataset,
    LoadedActDataset,
    build_dataset_views,
    compute_train_only_statistics,
    load_completed_m3b_dataset,
    load_lerobot_episode_view,
    split_manifest_digest,
    validate_temporal_episode_boundaries,
)
from langmani.policies.act_types import ActVariant

SOURCE_FINGERPRINT = "sha256:" + "1" * 64
SOURCE_ARCHIVE_DIGEST = "sha256:" + "2" * 64


def test_temporal_first_and_final_samples_are_episode_bounded() -> None:
    class Dataset:
        hf_dataset = {
            "episode_index": [7, 7],
            "frame_index": [0, 1],
        }

        def __getitem__(self, index: int) -> dict[str, object]:
            action = np.asarray([[float(index)] * 8, [1.0] * 8, [1.0] * 8], dtype=np.float32)
            padding = np.asarray([False, index == 1, index == 1], dtype=np.bool_)
            return {
                "episode_index": 7,
                "action": action,
                "action_is_pad": padding,
            }

    loaded = LoadedActDataset(
        dataset=Dataset(),
        episode_indices=(7,),
        delta_timestamps={"action": (0.0, 0.05, 0.1)},
    )
    result = validate_temporal_episode_boundaries(loaded, chunk_size=3)
    assert result["first_sample_validated"] is True
    assert result["final_sample_validated"] is True
    assert result["episode_crossing_detected"] is False


def _manifest(mode: ExportMode = ExportMode.SMOKE) -> LeRobotExportManifest:
    group_count = 1 if mode is ExportMode.SMOKE else 60
    split_config = (
        SplitConfig.smoke(split_seed=7)
        if mode is ExportMode.SMOKE
        else SplitConfig.full(split_seed=7)
    )
    config = LeRobotExportConfig(
        source_root="source",
        output_root="output",
        repo_id="langmani/test",
        expected_source_collection_run_id="collection-run",
        expected_source_run_fingerprint=SOURCE_FINGERPRINT,
        expected_source_archive_digest=SOURCE_ARCHIVE_DIGEST,
        mode=mode,
        split_config=split_config,
    )
    group_ids = tuple(f"group-{index:03d}" for index in range(group_count))
    assignments = build_split_assignments(group_ids, split_config)
    by_group = assignment_map(assignments)
    records: list[EpisodeExportRecord] = []
    for group_index, group_id in enumerate(group_ids):
        for task_index, task_spec in enumerate(CANONICAL_TASK_SPECS):
            episode_index = group_index * 6 + task_index
            records.append(
                EpisodeExportRecord(
                    source_collection_run_id="collection-run",
                    source_run_fingerprint=SOURCE_FINGERPRINT,
                    source_archive_digest=SOURCE_ARCHIVE_DIGEST,
                    source_episode_id=f"source-{episode_index:03d}",
                    source_scene_group_id=group_id,
                    source_scene_seed=1000 + group_index,
                    scene_id=f"scene-{group_index:03d}",
                    task_id=stable_task_id(task_spec),
                    target_object_id=task_spec.target_object_id,
                    target_bin_id=task_spec.target_bin_id,
                    instruction_template_id=task_spec.instruction_template_id,
                    canonical_instruction=canonical_instruction(task_spec),
                    source_shard_id="shard-0",
                    source_shard_path="accepted/shard-0.h5",
                    source_trajectory_key=f"traj_{episode_index}",
                    source_h5_sha256="3" * 64,
                    source_json_sha256="4" * 64,
                    source_checksum="5" * 64,
                    source_frame_count=1,
                    lerobot_episode_index=episode_index,
                    split=by_group[group_id].split,
                    raw_render_digest="6" * 64,
                    output_frame_count=1,
                )
            )
    ordered = tuple(item.source_episode_id for item in records)
    return LeRobotExportManifest(
        export_schema_version="langmani-m3b-lerobot-v1",
        export_fingerprint=stable_export_fingerprint(config, ordered),
        source_collection_run_id="collection-run",
        source_run_fingerprint=SOURCE_FINGERPRINT,
        source_archive_digest=SOURCE_ARCHIVE_DIGEST,
        source_schema_version="langmani-m3a-raw-v1",
        repo_id=config.repo_id,
        config=config,
        ordered_source_episode_ids=ordered,
        split_assignments=assignments,
        episodes=tuple(records),
        runtime_versions={"lerobot": "0.6.0"},
        finalized=True,
    )


def _summary(manifest: LeRobotExportManifest) -> LeRobotDatasetSummary:
    groups = {split.value: 0 for split in DatasetSplit}
    episodes = {split.value: 0 for split in DatasetSplit}
    tasks = {task_id: 0 for task_id in CANONICAL_TASK_IDS}
    for assignment in manifest.split_assignments:
        groups[assignment.split.value] += 1
    for record in manifest.episodes:
        episodes[record.split.value] += 1
        tasks[record.task_id] += 1
    return LeRobotDatasetSummary(
        export_fingerprint=manifest.export_fingerprint,
        repo_id=manifest.repo_id,
        mode=manifest.config.mode,
        total_scene_groups=len(manifest.split_assignments),
        total_episodes=len(manifest.episodes),
        total_frames=len(manifest.episodes),
        task_episode_counts=tasks,
        split_scene_group_counts=groups,
        split_episode_counts=episodes,
        feature_keys=tuple(sorted(POLICY_FEATURE_KEYS | LEROBOT_MANAGED_FEATURE_KEYS)),
        fps=20,
        image_shape=(256, 256, 3),
        state_shape=(9,),
        action_shape=(8,),
        parquet_file_count=3,
        video_file_count=1,
    )


def _validation(manifest: LeRobotExportManifest) -> LeRobotValidationReport:
    alignments = tuple(
        SourceAlignmentResult(
            lerobot_episode_index=record.lerobot_episode_index,
            source_episode_id=record.source_episode_id,
            task_identity_matches=True,
            scene_identity_matches=True,
            frame_count_matches=True,
            actions_match=True,
            policy_states_match=True,
            raw_render_digest_matches=True,
            decoded_video_frame_count_matches=True,
            decoded_video_quality_passed=True,
            timestamps_match=True,
            passed=True,
        )
        for record in manifest.episodes
    )
    videos = tuple(
        VideoValidationResult(
            lerobot_episode_index=record.lerobot_episode_index,
            video_path="videos/base_camera.mp4",
            expected_frame_count=1,
            decoded_frame_count=1,
            decoded_height=256,
            decoded_width=256,
            sampled_frame_indices=(0,),
            mean_absolute_error=0.0,
            minimum_psnr_db=99.0,
            frames_nonempty=True,
            frames_nonuniform=True,
            passed=True,
        )
        for record in manifest.episodes
    )
    return LeRobotValidationReport(
        export_fingerprint=manifest.export_fingerprint,
        mode=manifest.config.mode,
        dataset_load_validated=True,
        feature_schema_validated=True,
        parquet_validated=True,
        video_decode_validated=True,
        source_alignment_validated=True,
        split_integrity_validated=True,
        privileged_leakage_validated=True,
        dataloader_validated=True,
        source_alignments=alignments,
        video_results=videos,
        passed=True,
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _completed_fixture(root: Path, manifest: LeRobotExportManifest) -> None:
    summary = _summary(manifest)
    validation = _validation(manifest)
    split = {
        "schema_version": "langmani-m3b-splits-v1",
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
    sidecar = root / "langmani"
    _write_json(
        sidecar / "complete.json",
        {
            "schema_version": "langmani-m3b-completion-v1",
            "export_fingerprint": manifest.export_fingerprint,
        },
    )
    _write_json(sidecar / "export_config.json", manifest.config.portable_dict())
    _write_json(sidecar / "export_manifest.json", manifest.to_dict())
    _write_json(
        sidecar / "source_episode_mapping.json",
        {
            "schema_version": "langmani-m3b-source-mapping-v1",
            "export_fingerprint": manifest.export_fingerprint,
            "episodes": [item.to_dict() for item in manifest.episodes],
        },
    )
    _write_json(sidecar / "split_manifest.json", split)
    _write_json(sidecar / "dataset_summary.json", summary.to_dict())
    _write_json(sidecar / "validation_report.json", validation.to_dict())


def test_completed_m3b_gate_uses_only_local_completion_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    root = tmp_path / "dataset"
    _completed_fixture(root, manifest)
    monkeypatch.setattr(act_data, "_validate_local_storage", lambda *_: None)

    completed = load_completed_m3b_dataset(root, require_full=False)

    assert completed.export_fingerprint == manifest.export_fingerprint
    assert completed.views.train.episode_indices == tuple(range(6))
    assert completed.views.validation.episode_indices == ()
    assert completed.split_manifest_digest.startswith("sha256:")


def test_completed_gate_rejects_missing_or_tampered_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    root = tmp_path / "dataset"
    _completed_fixture(root, manifest)
    monkeypatch.setattr(act_data, "_validate_local_storage", lambda *_: None)
    marker = root / "langmani" / "complete.json"
    marker.unlink()
    with pytest.raises(ActDataContractError, match="sidecars are missing"):
        load_completed_m3b_dataset(root, require_full=False)
    _write_json(
        marker, {"schema_version": "wrong", "export_fingerprint": manifest.export_fingerprint}
    )
    with pytest.raises(ActDataContractError, match="completion marker"):
        load_completed_m3b_dataset(root, require_full=False)


def test_full_views_are_exact_balanced_and_scene_safe() -> None:
    manifest = _manifest(ExportMode.FULL)
    views = build_dataset_views(manifest)

    assert len(views.train.episode_indices) == 288
    assert len(views.validation.episode_indices) == 36
    assert len(views.test.episode_indices) == 36
    for task_id in CANONICAL_TASK_IDS:
        assert len(views.for_split(DatasetSplit.TRAIN, task_id=task_id).episode_indices) == 48
        assert len(views.for_split(DatasetSplit.VALIDATION, task_id=task_id).episode_indices) == 6
        assert len(views.for_split(DatasetSplit.TEST, task_id=task_id).episode_indices) == 6
    assert not set(views.train.episode_indices) & set(views.test.episode_indices)


def test_views_reject_same_physical_scene_across_splits() -> None:
    manifest = _manifest(ExportMode.FULL)
    train_record = next(item for item in manifest.episodes if item.split is DatasetSplit.TRAIN)
    test_group = next(
        item.source_scene_group_id for item in manifest.episodes if item.split is DatasetSplit.TEST
    )
    changed = tuple(
        replace(item, scene_id=train_record.scene_id)
        if item.source_scene_group_id == test_group
        else item
        for item in manifest.episodes
    )
    leaking = replace(manifest, episodes=changed)
    with pytest.raises(ActDataContractError, match="physical scene leaks"):
        build_dataset_views(leaking)


def test_split_digest_is_canonical_not_whitespace_or_path_dependent() -> None:
    first = {"schema": "v1", "indices": {"train": [2, 1], "test": [3]}}
    reordered = {"indices": {"test": [3], "train": [2, 1]}, "schema": "v1"}

    assert split_manifest_digest(first) == split_manifest_digest(reordered)
    assert split_manifest_digest(first) != split_manifest_digest(
        {"schema": "v1", "indices": {"train": [1, 2], "test": [3]}}
    )


def test_public_loader_uses_episode_filter_and_resolved_act_deltas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    views = build_dataset_views(manifest)
    completed = CompletedM3BDataset(
        root=tmp_path,
        manifest=manifest,
        summary=_summary(manifest),
        validation=_validation(manifest),
        split_manifest_digest="sha256:" + "a" * 64,
        views=views,
    )
    calls: dict[str, Any] = {}

    class FakeMetadata:
        def __init__(self, repo_id: str, root: Path) -> None:
            calls["metadata"] = (repo_id, root)

    class FakeHFDataset:
        def unique(self, key: str) -> list[int]:
            assert key == "episode_index"
            return list(views.train.episode_indices)

    class FakeDataset:
        fps = 20
        hf_dataset = FakeHFDataset()

        def __init__(self, **kwargs: object) -> None:
            calls["dataset"] = kwargs

    def fake_resolve(config: object, meta: object) -> dict[str, list[float]]:
        calls["resolve"] = (config, meta)
        return {"action": [0.0, 0.05, 0.1]}

    monkeypatch.setattr(
        act_data,
        "_lerobot_public_apis",
        lambda: (FakeDataset, FakeMetadata, fake_resolve),
    )
    policy_config = SimpleNamespace(action_delta_indices=[0, 1, 2], observation_delta_indices=None)

    loaded = load_lerobot_episode_view(
        completed,
        views.train,
        policy_config=policy_config,
        include_delta_timestamps=True,
    )

    assert loaded.delta_timestamps == {"action": (0.0, 0.05, 0.1)}
    assert calls["dataset"]["episodes"] == list(range(6))
    assert calls["dataset"]["download_videos"] is False
    assert calls["dataset"]["video_backend"] == "pyav"


def test_public_loader_rejects_episode_indices_mislabeled_as_another_split() -> None:
    manifest = _manifest(ExportMode.SMOKE)
    completed = CompletedM3BDataset(
        root=Path("dataset"),
        manifest=manifest,
        summary=_summary(manifest),
        validation=_validation(manifest),
        split_manifest_digest="sha256:" + "f" * 64,
        views=build_dataset_views(manifest),
    )
    mislabeled = act_data.DatasetEpisodeView(
        split=DatasetSplit.VALIDATION,
        episode_indices=(0,),
    )
    with pytest.raises(ActDataContractError, match="authoritative M3B split"):
        load_lerobot_episode_view(
            completed,
            mislabeled,
            policy_config=None,
            include_delta_timestamps=False,
        )


def _row(
    episode_index: int, image_value: int, state_value: float, action_value: float
) -> dict[str, object]:
    return {
        "episode_index": episode_index,
        "frame_index": 0,
        "observation.images.base_camera": np.full((3, 256, 256), image_value, dtype=np.uint8),
        "observation.state": np.full(9, state_value, dtype=np.float32),
        "action": np.full(8, action_value, dtype=np.float32),
    }


def test_train_only_streaming_stats_use_float_images_and_exclude_held_out_episodes() -> None:
    task_id = CANONICAL_TASK_IDS[0]
    rows = [_row(0, 0, 0.0, 1.0), _row(1, 255, 2.0, 3.0)]
    stats = compute_train_only_statistics(
        rows,
        train_episode_indices=(0, 1),
        validation_episode_indices=(2,),
        test_episode_indices=(3,),
        task_id_by_episode={0: task_id, 1: task_id},
        variant=ActVariant.PER_TASK,
        task_id=task_id,
    )

    assert stats.image.mean == pytest.approx((0.5, 0.5, 0.5))
    assert stats.image.std == pytest.approx((0.5, 0.5, 0.5))
    assert stats.image.count == 2 * 256 * 256
    assert stats.state.mean == pytest.approx((1.0,) * 9)
    assert len(stats.state.components) == 9
    assert stats.action.mean == pytest.approx((2.0,) * 8)
    assert stats.leakage_audit.passed
    assert stats.leakage_audit.source_episode_indices == (0, 1)
    assert stats.statistics_fingerprint.startswith("sha256:")
    processor = stats.to_processor_stats()
    assert tuple(processor["observation.images.base_camera"]["mean"].shape) == (3, 1, 1)
    assert tuple(processor["action"]["mean"].shape) == (8,)


def test_task_onehot_stats_append_identity_suffix_and_are_stable() -> None:
    rows = [_row(0, 64, 0.0, 1.0), _row(1, 128, 2.0, 3.0)]
    mapping = {0: CANONICAL_TASK_IDS[0], 1: CANONICAL_TASK_IDS[1]}
    kwargs = {
        "train_episode_indices": (0, 1),
        "validation_episode_indices": (2,),
        "test_episode_indices": (3,),
        "task_id_by_episode": mapping,
        "variant": ActVariant.MIXED_TASK_ONEHOT,
    }
    first = compute_train_only_statistics(rows, **kwargs)
    second = compute_train_only_statistics(rows, **kwargs)
    changed = compute_train_only_statistics([rows[0], _row(1, 129, 2.0, 3.0)], **kwargs)

    assert len(first.state.components) == 15
    assert first.state.mean[-6:] == (0.0,) * 6
    assert first.state.std[-6:] == (1.0,) * 6
    assert first.state.minimum[-6:] == (0.0,) * 6
    assert first.state.maximum[-6:] == (1.0,) * 6
    assert first.statistics_fingerprint == second.statistics_fingerprint
    assert first.statistics_fingerprint != changed.statistics_fingerprint


def test_statistics_reject_leakage_unexpected_rows_and_delta_expansion() -> None:
    task_id = CANONICAL_TASK_IDS[0]
    base = {
        "train_episode_indices": (0,),
        "validation_episode_indices": (1,),
        "test_episode_indices": (2,),
        "task_id_by_episode": {0: task_id},
        "variant": ActVariant.PER_TASK,
        "task_id": task_id,
    }
    with pytest.raises(ValueError, match="leakage"):
        compute_train_only_statistics(
            [_row(0, 0, 0.0, 0.0)],
            **{**base, "validation_episode_indices": (0,)},
        )
    with pytest.raises(ActDataContractError, match="validation/test/unknown"):
        compute_train_only_statistics([_row(1, 0, 0.0, 0.0)], **base)
    expanded = _row(0, 0, 0.0, 0.0)
    expanded["action"] = np.zeros((2, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="current-frame"):
        compute_train_only_statistics([expanded], **base)
    with pytest.raises(ActDataContractError, match="duplicate episode/frame"):
        compute_train_only_statistics([_row(0, 0, 0.0, 0.0), _row(0, 0, 0.0, 0.0)], **base)
