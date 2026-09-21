"""Real CPU LeRobot writer -> ACT optimization/resume/reload, using generated arrays only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from langmani.datasets.lerobot_types import FeatureContract, VideoCodecConfig
from langmani.datasets.lerobot_writer import LeRobotWriterAdapter
from langmani.policies.m4a_data import CANONICAL_TEXTS, TASK_TEXT, validate_rows, validate_videos
from langmani.policies.m4a_training import checkpoint_directory, train_official

pytestmark = [pytest.mark.integration, pytest.mark.fixture, pytest.mark.video]


def test_real_writer_official_act_train_resume_checkpoint_reload(tmp_path: Path) -> None:
    from lerobot.datasets import LeRobotDataset

    torch.set_num_threads(2)
    root = tmp_path / "generated-fixture"
    repo_id = "langmani/m4a-generated-fixture"
    writer = LeRobotWriterAdapter.create(
        root=root,
        repo_id=repo_id,
        fps=20,
        feature_contract=FeatureContract(),
        codec=VideoCodecConfig(),
        data_files_size_in_mb=100,
        video_files_size_in_mb=200,
    )
    texts = sorted(CANONICAL_TEXTS)
    rng = np.random.default_rng(17)
    for episode in range(12):
        for _ in range(3):
            writer.add_policy_frame(
                rgb=rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
                state=rng.normal(size=9).astype(np.float32),
                action=rng.normal(size=8).astype(np.float32),
                task=texts[episode % 6],
            )
        writer.save_episode()
    writer.finalize_once()
    dataset = LeRobotDataset(repo_id, root=root, video_backend="pyav", return_uint8=True)
    report = validate_rows(dataset)
    assert report["num_frames"] == 36
    assert validate_videos(root, 36)["decoded_frames"] == 36
    with pytest.raises(ValueError, match="frame count"):
        validate_videos(root, 35)
    train_id = texts.index(TASK_TEXT)
    split = {
        "repo_id": repo_id,
        "train_episode_ids": [train_id],
        "held_out_episode_ids": [train_id + 6],
        "train_frames": 3,
    }
    output = tmp_path / "fixture-training"
    first = train_official(
        root=root,
        output=output,
        split=split,
        mode="smoke",
        device="cpu",
        batch_size=2,
        steps=1,
        seed=0,
    )
    assert first["checkpoint_reloaded"] and not first["full_act_trained"]
    checkpoint = checkpoint_directory(output / first["checkpoint"])
    resumed = train_official(
        root=root,
        output=output,
        split=split,
        mode="smoke",
        device="cpu",
        batch_size=2,
        steps=2,
        seed=0,
        resume=checkpoint,
    )
    assert resumed["steps"] == 2 and resumed["checkpoint_reloaded"]
    assert resumed["checkpoint_sha256"] != first["checkpoint_sha256"]
