"""Unit contracts for the deterministic M3B LeRobot representation."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from langmani.datasets.identity import COLLECTION_SCHEMA_VERSION
from langmani.datasets.lerobot_types import (
    LEROBOT_EXPORT_SCHEMA_VERSION,
    LEROBOT_MANAGED_FEATURE_KEYS,
    PANDA_ACTION_COMPONENTS,
    POLICY_FEATURE_KEYS,
    CameraContract,
    DatasetSplit,
    EpisodeExportRecord,
    ExportMode,
    FeatureContract,
    LeRobotDatasetSummary,
    LeRobotExportConfig,
    LeRobotExportManifest,
    LeRobotValidationReport,
    PolicyStateSchema,
    SourceAlignmentResult,
    SplitConfig,
    VideoCodecConfig,
    VideoValidationResult,
    stable_export_fingerprint,
)
from langmani.datasets.policy_state import PANDA_POLICY_STATE_COMPONENTS
from langmani.datasets.splits import build_split_assignments

SOURCE_COLLECTION_ID = "langmani-m3a-collection-run-" + "1" * 64
SOURCE_RUN_FINGERPRINT = "sha256:" + "a" * 64
SOURCE_ARCHIVE_DIGEST = "sha256:" + "b" * 64


def _config(
    *,
    mode: ExportMode = ExportMode.FULL,
    source_root: str = "D:/machine-a/raw",
    output_root: str = "D:/machine-a/derived",
    split_seed: int = 17,
) -> LeRobotExportConfig:
    split_config = (
        SplitConfig.full(split_seed=split_seed)
        if mode is ExportMode.FULL
        else SplitConfig.smoke(split_seed=split_seed)
    )
    return LeRobotExportConfig(
        source_root=source_root,
        output_root=output_root,
        repo_id="langmani/pick-place-by-instruction-v0",
        expected_source_collection_run_id=SOURCE_COLLECTION_ID,
        expected_source_run_fingerprint=SOURCE_RUN_FINGERPRINT,
        expected_source_archive_digest=SOURCE_ARCHIVE_DIGEST,
        mode=mode,
        split_config=split_config,
    )


def _episode_record(index: int, *, group_id: str, split: DatasetSplit) -> EpisodeExportRecord:
    task_index = index % 6
    objects = ("red_cube", "red_cube", "green_cube", "green_cube", "blue_cube", "blue_cube")
    bins = ("left_bin", "right_bin", "left_bin", "right_bin", "left_bin", "right_bin")
    return EpisodeExportRecord(
        source_collection_run_id=SOURCE_COLLECTION_ID,
        source_run_fingerprint=SOURCE_RUN_FINGERPRINT,
        source_archive_digest=SOURCE_ARCHIVE_DIGEST,
        source_episode_id=f"source-episode-{index:03d}",
        source_scene_group_id=group_id,
        source_scene_seed=123,
        scene_id="scene-123",
        task_id=f"task-{task_index}",
        target_object_id=objects[task_index],
        target_bin_id=bins[task_index],
        instruction_template_id="canonical_v0",
        canonical_instruction=f"canonical instruction {task_index}",
        source_shard_id="source-shard-0",
        source_shard_path="accepted/shard-00000.h5",
        source_trajectory_key=f"traj_{index}",
        source_h5_sha256="c" * 64,
        source_json_sha256="d" * 64,
        source_checksum="e" * 64,
        source_frame_count=10 + index,
        lerobot_episode_index=index,
        split=split,
        raw_render_digest="f" * 64,
        output_frame_count=10 + index,
    )


def test_feature_contract_is_exact_and_uses_semantic_component_names() -> None:
    contract = FeatureContract()

    assert contract.policy_feature_keys == POLICY_FEATURE_KEYS
    assert contract.image_shape == (256, 256, 3)
    assert contract.state_names == PANDA_POLICY_STATE_COMPONENTS
    assert contract.action_names == PANDA_ACTION_COMPONENTS
    assert len(contract.action_names) == 8
    assert contract.to_lerobot_features() == {
        "observation.images.base_camera": {
            "dtype": "video",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (9,),
            "names": list(PANDA_POLICY_STATE_COMPONENTS),
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": list(PANDA_ACTION_COMPONENTS),
        },
    }
    assert FeatureContract.from_dict(json.loads(json.dumps(contract.to_dict()))) == contract

    with pytest.raises(ValueError, match="three-feature allowlist"):
        FeatureContract(image_feature_key="observation.images.wrist")


def test_feature_contract_validates_without_converting_or_clipping() -> None:
    contract = FeatureContract()
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    state = np.arange(9, dtype=np.float32)
    action = np.arange(8, dtype=np.float32)
    rgb_before = rgb.copy()
    state_before = state.copy()
    action_before = action.copy()

    contract.validate_frame(rgb=rgb, state=state, action=action)

    np.testing.assert_array_equal(rgb, rgb_before)
    np.testing.assert_array_equal(state, state_before)
    np.testing.assert_array_equal(action, action_before)

    with pytest.raises(TypeError, match="rgb must be a numpy"):
        contract.validate_frame(rgb=rgb.tolist(), state=state, action=action)
    with pytest.raises(ValueError, match="rgb dtype"):
        contract.validate_frame(rgb=rgb.astype(np.float32), state=state, action=action)
    with pytest.raises(ValueError, match="rgb shape"):
        contract.validate_frame(rgb=rgb[:255], state=state, action=action)
    with pytest.raises(ValueError, match="state dtype"):
        contract.validate_frame(rgb=rgb, state=state.astype(np.float64), action=action)
    with pytest.raises(ValueError, match="state shape"):
        contract.validate_frame(rgb=rgb, state=state[:8], action=action)
    with pytest.raises(ValueError, match="state must contain only finite"):
        contract.validate_frame(rgb=rgb, state=np.full(9, np.nan, np.float32), action=action)
    with pytest.raises(ValueError, match="action dtype"):
        contract.validate_frame(rgb=rgb, state=state, action=action.astype(np.float64))
    with pytest.raises(ValueError, match="action shape"):
        contract.validate_frame(rgb=rgb, state=state, action=action[:7])
    with pytest.raises(ValueError, match="action must contain only finite"):
        contract.validate_frame(rgb=rgb, state=state, action=np.full(8, np.inf, np.float32))


def test_policy_camera_codec_and_state_contracts_are_fixed_and_round_trip() -> None:
    state = PolicyStateSchema()
    camera = CameraContract()
    codec = VideoCodecConfig()

    assert state.components == PANDA_POLICY_STATE_COMPONENTS
    assert camera.eye == (0.65, -0.75, 0.70)
    assert camera.target == (-0.04, 0.0, 0.08)
    assert camera.render_backend == "sapien_cuda"
    assert codec.to_dict() == {
        "codec": "h264",
        "pixel_format": "yuv444p",
        "crf": 18,
        "gop_size": 2,
        "preset": "medium",
        "fast_decode": 0,
        "backend": "pyav",
        "encoder_threads": 1,
    }
    assert PolicyStateSchema.from_dict(json.loads(json.dumps(state.to_dict()))) == state
    assert CameraContract.from_dict(json.loads(json.dumps(camera.to_dict()))) == camera
    assert VideoCodecConfig.from_dict(json.loads(json.dumps(codec.to_dict()))) == codec


def test_export_config_is_immutable_strict_and_json_ready() -> None:
    config = _config()
    payload = json.loads(json.dumps(config.to_dict()))

    assert config.fps == 20
    assert (config.image_height, config.image_width) == (256, 256)
    assert config.video_mean_absolute_error_tolerance == 5.0
    assert config.video_min_psnr_db == 30.0
    assert LeRobotExportConfig.from_dict(payload) == config
    with pytest.raises(FrozenInstanceError):
        config.fps = 10  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"repo_id": "missing-owner"}, "owner/dataset-name"),
        ({"expected_source_run_fingerprint": "not-a-digest"}, "SHA-256"),
        ({"expected_source_archive_digest": "b" * 64}, "start with"),
        ({"fps": 30}, "fps must be 20"),
        ({"image_height": 224}, "resolution"),
        ({"overwrite_policy": "replace"}, "immutable"),
        ({"staging_policy": "resume"}, "does not resume"),
        ({"action_tolerance": 1e-6}, "exact raw-action equality"),
    ],
)
def test_export_config_rejects_unsupported_contracts(
    kwargs: dict[str, object], message: str
) -> None:
    base = _config().to_dict()
    base.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=message):
        LeRobotExportConfig.from_dict(base)


def test_export_config_portable_dict_removes_machine_paths() -> None:
    config = _config(
        source_root=r"D:\private\m3a",
        output_root=r"C:\Users\name\derived",
    )
    portable = config.portable_dict()

    assert portable["source_root"] == "<authoritative-m3a-source>"
    assert portable["output_root"] == "."
    serialized = json.dumps(portable)
    assert "D:" not in serialized
    assert "C:" not in serialized


def test_stable_export_fingerprint_covers_semantics_but_not_machine_paths() -> None:
    episode_ids = tuple(f"episode-{index}" for index in range(360))
    config = _config()
    relocated = _config(source_root="/mnt/raw", output_root="/mnt/derived")
    base = stable_export_fingerprint(config, episode_ids)

    assert base == stable_export_fingerprint(relocated, episode_ids)
    assert base.startswith("sha256:") and len(base) == 71
    assert base != stable_export_fingerprint(config, tuple(reversed(episode_ids)))
    assert base != stable_export_fingerprint(
        replace(config, video_codec=replace(config.video_codec, crf=31)), episode_ids
    )
    assert base != stable_export_fingerprint(
        replace(config, split_config=SplitConfig.full(split_seed=18)), episode_ids
    )
    assert base != stable_export_fingerprint(
        replace(config, expected_source_archive_digest="sha256:" + "9" * 64), episode_ids
    )
    with pytest.raises(ValueError, match="unique"):
        stable_export_fingerprint(config, ("duplicate", "duplicate"))


def test_smoke_manifest_is_portable_and_round_trips_every_mapping() -> None:
    config = _config(mode=ExportMode.SMOKE)
    group_id = "scene-group-0"
    assignments = build_split_assignments((group_id,), config.split_config)
    episodes = tuple(
        _episode_record(index, group_id=group_id, split=DatasetSplit.TRAIN) for index in range(6)
    )
    ordered_ids = tuple(item.source_episode_id for item in episodes)
    manifest = LeRobotExportManifest(
        export_schema_version=LEROBOT_EXPORT_SCHEMA_VERSION,
        export_fingerprint=stable_export_fingerprint(config, ordered_ids),
        source_collection_run_id=SOURCE_COLLECTION_ID,
        source_run_fingerprint=SOURCE_RUN_FINGERPRINT,
        source_archive_digest=SOURCE_ARCHIVE_DIGEST,
        source_schema_version=COLLECTION_SCHEMA_VERSION,
        repo_id=config.repo_id,
        config=config,
        ordered_source_episode_ids=ordered_ids,
        split_assignments=assignments,
        episodes=episodes,
        runtime_versions={"lerobot": "0.6.0", "av": "15.1.0"},
        finalized=True,
    )
    payload = json.loads(json.dumps(manifest.to_dict()))
    restored = LeRobotExportManifest.from_dict(payload)

    assert restored.to_dict() == payload
    assert restored.export_fingerprint == manifest.export_fingerprint
    assert restored.config.source_root == "<authoritative-m3a-source>"
    assert "D:/machine-a" not in json.dumps(payload)
    assert tuple(item.lerobot_episode_index for item in restored.episodes) == tuple(range(6))


def test_episode_record_rejects_unsafe_paths_and_frame_count_drift() -> None:
    record = _episode_record(0, group_id="group", split=DatasetSplit.TRAIN)
    assert EpisodeExportRecord.from_dict(json.loads(json.dumps(record.to_dict()))) == record

    with pytest.raises(ValueError, match="relative POSIX"):
        replace(record, source_shard_path=r"D:\raw\shard.h5")
    with pytest.raises(ValueError, match="exact source frame count"):
        replace(record, output_frame_count=record.source_frame_count + 1)


def test_validation_records_and_summary_are_json_ready() -> None:
    fingerprint = "sha256:" + "1" * 64
    video = VideoValidationResult(
        lerobot_episode_index=0,
        video_path="videos/observation.images.base_camera/chunk-000/file-000.mp4",
        expected_frame_count=10,
        decoded_frame_count=10,
        decoded_height=256,
        decoded_width=256,
        sampled_frame_indices=(0, 2, 4, 7, 9),
        mean_absolute_error=1.5,
        minimum_psnr_db=40.0,
        frames_nonempty=True,
        frames_nonuniform=True,
        passed=True,
    )
    alignment = SourceAlignmentResult(
        lerobot_episode_index=0,
        source_episode_id="episode-0",
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
    summary = LeRobotDatasetSummary(
        export_fingerprint=fingerprint,
        repo_id="langmani/pick-place-by-instruction-v0",
        mode=ExportMode.SMOKE,
        total_scene_groups=1,
        total_episodes=6,
        total_frames=60,
        task_episode_counts={f"task-{index}": 1 for index in range(6)},
        split_scene_group_counts={"train": 1, "validation": 0, "test": 0},
        split_episode_counts={"train": 6, "validation": 0, "test": 0},
        feature_keys=tuple(sorted(POLICY_FEATURE_KEYS | LEROBOT_MANAGED_FEATURE_KEYS)),
        fps=20,
        image_shape=(256, 256, 3),
        state_shape=(9,),
        action_shape=(8,),
        parquet_file_count=3,
        video_file_count=1,
    )
    report = LeRobotValidationReport(
        export_fingerprint=fingerprint,
        mode=ExportMode.SMOKE,
        dataset_load_validated=True,
        feature_schema_validated=True,
        parquet_validated=True,
        video_decode_validated=True,
        source_alignment_validated=True,
        split_integrity_validated=True,
        privileged_leakage_validated=True,
        dataloader_validated=True,
        source_alignments=(alignment,),
        video_results=(video,),
        passed=True,
    )

    assert VideoValidationResult.from_dict(json.loads(json.dumps(video.to_dict()))) == video
    assert SourceAlignmentResult.from_dict(json.loads(json.dumps(alignment.to_dict()))) == alignment
    assert LeRobotDatasetSummary.from_dict(json.loads(json.dumps(summary.to_dict()))) == summary
    assert LeRobotValidationReport.from_dict(json.loads(json.dumps(report.to_dict()))) == report
