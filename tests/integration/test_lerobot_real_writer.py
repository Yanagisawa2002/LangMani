"""Real local LeRobot 0.6.0 writer/reader integration fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from langmani.datasets.frame_alignment import rgb_quality_metrics
from langmani.datasets.lerobot_types import (
    ExportMode,
    FeatureContract,
    LeRobotExportConfig,
    SplitConfig,
    VideoCodecConfig,
)
from langmani.datasets.lerobot_validation import (
    LeRobotValidationError,
    _validate_feature_schema,
    validate_lerobot_dataset,
)
from langmani.datasets.lerobot_writer import LeRobotWriterAdapter, WriterState

pytestmark = [pytest.mark.integration, pytest.mark.video, pytest.mark.fixture]


def _rgb(episode_index: int, frame_index: int) -> np.ndarray:
    y, x = np.indices((256, 256))
    return np.stack(
        (
            (x + frame_index * 11) % 256,
            (y + episode_index * 23) % 256,
            ((x + y) // 2 + frame_index * 7) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def test_real_writer_finalize_reload_video_parquet_tasks_and_dataloader(tmp_path: Path) -> None:
    root = tmp_path / "fixture-dataset"
    contract = FeatureContract()
    writer = LeRobotWriterAdapter.create(
        root=root,
        repo_id="langmani/m3b-writer-fixture",
        fps=20,
        feature_contract=contract,
        codec=VideoCodecConfig(),
        data_files_size_in_mb=100,
        video_files_size_in_mb=200,
    )
    expected_tasks = (
        "Pick up the red cube and place it in the left bin.",
        "Pick up the blue cube and place it in the right bin.",
    )
    expected_states: list[np.ndarray] = []
    expected_actions: list[np.ndarray] = []
    expected_rgb: list[np.ndarray] = []
    for episode_index, length in enumerate((3, 4)):
        for frame_index in range(length):
            state = np.arange(9, dtype=np.float32) + frame_index
            action = np.arange(8, dtype=np.float32) + episode_index
            expected_states.append(state.copy())
            expected_actions.append(action.copy())
            rgb = _rgb(episode_index, frame_index)
            expected_rgb.append(rgb.copy())
            writer.add_policy_frame(
                rgb=rgb,
                state=state,
                action=action,
                task=expected_tasks[episode_index],
            )
        writer.save_episode()
    writer.finalize_once()

    assert writer.state is WriterState.FINALIZED
    assert writer.saved_episodes == 2
    assert writer.finalize_calls == 1

    from lerobot.datasets import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id="langmani/m3b-writer-fixture",
        root=root,
        video_backend="pyav",
        return_uint8=True,
    )
    assert dataset.meta.info.codebase_version == "v3.0"
    assert dataset.num_episodes == 2
    assert dataset.num_frames == 7
    assert dataset.meta.episodes[0]["length"] == 3
    assert dataset.meta.episodes[1]["length"] == 4
    assert dataset.meta.episodes[0]["dataset_from_index"] == 0
    assert dataset.meta.episodes[1]["dataset_to_index"] == 7
    measured_quality: list[tuple[float, float]] = []
    for index in range(7):
        item = dataset[index]
        assert item[contract.image_feature_key].shape == (3, 256, 256)
        assert item[contract.image_feature_key].dtype == torch.uint8
        assert torch.equal(
            item[contract.state_feature_key], torch.from_numpy(expected_states[index])
        )
        assert torch.equal(
            item[contract.action_feature_key], torch.from_numpy(expected_actions[index])
        )
        expected_episode = 0 if index < 3 else 1
        assert item["task"] == expected_tasks[expected_episode]
        assert float(item["timestamp"]) == pytest.approx((index if index < 3 else index - 3) / 20)
        decoded = item[contract.image_feature_key].permute(1, 2, 0).numpy()
        measured_quality.append(rgb_quality_metrics(expected_rgb[index], decoded))
    assert max(mae for mae, _ in measured_quality) <= 5.0
    assert min(psnr for _, psnr in measured_quality) >= 30.0

    parquet_paths = sorted(root.rglob("*.parquet"))
    video_paths = sorted(root.rglob("*.mp4"))
    assert len(parquet_paths) == 3
    assert len(video_paths) == 1
    import pyarrow.parquet as pq

    for path in parquet_paths:
        assert pq.ParquetFile(path).metadata.num_rows > 0

    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)))
    assert batch[contract.image_feature_key].shape == (2, 3, 256, 256)
    assert batch[contract.image_feature_key].dtype == torch.uint8
    assert batch[contract.state_feature_key].shape == (2, 9)
    assert batch[contract.action_feature_key].shape == (2, 8)
    assert batch["task"] == list(expected_tasks[:1]) * 2


def _nonvideo_features(*, include_privileged: bool = False) -> dict[str, dict[str, object]]:
    features: dict[str, dict[str, object]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (9,),
            "names": [f"state_{index}" for index in range(9)],
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": [f"action_{index}" for index in range(8)],
        },
    }
    if include_privileged:
        features["target_object_index"] = {
            "dtype": "int64",
            "shape": (1,),
            "names": ["target_object_index"],
        }
    return features


def _add_nonvideo_frame(dataset: object, *, include_privileged: bool = False) -> None:
    frame = {
        "observation.state": np.zeros(9, dtype=np.float32),
        "action": np.zeros(8, dtype=np.float32),
        "task": "Pick up the red cube and place it in the left bin.",
    }
    if include_privileged:
        frame["target_object_index"] = np.zeros(1, dtype=np.int64)
    dataset.add_frame(frame)  # type: ignore[attr-defined]


def test_real_unfinalized_dataset_is_rejected_as_incomplete(tmp_path: Path) -> None:
    from lerobot.datasets import LeRobotDataset

    root = tmp_path / "unfinalized"
    dataset = LeRobotDataset.create(
        repo_id="langmani/m3b-unfinalized-fixture",
        fps=20,
        features=_nonvideo_features(),
        root=root,
        use_videos=False,
    )
    _add_nonvideo_frame(dataset)

    with pytest.raises(LeRobotValidationError, match="sidecars are missing"):
        validate_lerobot_dataset(
            root,
            source_root=tmp_path / "source",
            source_verification_report=tmp_path / "verification.json",
            require_completion_marker=True,
        )

    dataset.save_episode(parallel_encoding=False)
    dataset.finalize()


def test_real_local_dataset_with_privileged_feature_is_rejected(tmp_path: Path) -> None:
    from lerobot.datasets import LeRobotDataset

    root = tmp_path / "schema-mismatch"
    dataset = LeRobotDataset.create(
        repo_id="langmani/m3b-schema-mismatch-fixture",
        fps=20,
        features=_nonvideo_features(include_privileged=True),
        root=root,
        use_videos=False,
    )
    _add_nonvideo_frame(dataset, include_privileged=True)
    dataset.save_episode(parallel_encoding=False)
    dataset.finalize()
    reloaded = LeRobotDataset(
        repo_id="langmani/m3b-schema-mismatch-fixture",
        root=root,
        download_videos=False,
    )
    config = LeRobotExportConfig(
        source_root="source",
        output_root="output",
        repo_id="langmani/m3b-schema-mismatch-fixture",
        expected_source_collection_run_id="collection-fixture",
        expected_source_run_fingerprint="sha256:" + "1" * 64,
        expected_source_archive_digest="sha256:" + "2" * 64,
        mode=ExportMode.SMOKE,
        split_config=SplitConfig.smoke(),
    )

    schema_reasons, leakage_reasons = _validate_feature_schema(reloaded.features, config)

    assert schema_reasons
    assert any("target_object_index" in reason for reason in leakage_reasons)
