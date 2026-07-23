from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path

import numpy as np
import pytest

from langmani.v2.phase2b5 import (
    CANDIDATE_TASKS,
    PANDA_ACTION_HIGH,
    PANDA_ACTION_LOW,
)
from langmani.v2.phase2b6_lerobot import (
    ACTION_FEATURE,
    DERIVED_IDENTITY_FEATURE,
    IMAGE_FEATURE,
    SKILL_FEATURE,
    SOURCE_IDENTITY_FEATURE,
    SPLIT_FEATURE,
    STATE_FEATURE,
    TASK_ID_FEATURE,
    TEMPLATE_ID_FEATURE,
    _readback_one_root,
    _writer,
)

pytestmark = [pytest.mark.integration, pytest.mark.video]

try:
    LEROBOT_VERSION = metadata.version("lerobot")
except metadata.PackageNotFoundError:
    LEROBOT_VERSION = None


@pytest.mark.skipif(
    LEROBOT_VERSION != "0.6.0",
    reason="requires the isolated LeRobot 0.6 runtime",
)
def test_task_split_root_creation_and_complete_readback(tmp_path: Path) -> None:
    task_id = "PickCube-v1"
    skill = CANDIDATE_TASKS[task_id].skill_family
    instruction = "Lift the cube and place it at the goal."
    root = tmp_path / "pickcube" / "splits" / "train"
    repo_id = "langmani/test-phase2b6-pickcube-train"
    dataset = _writer(repo_id=repo_id, root=root)
    episode_index = []
    global_frame = 0
    midpoint = (PANDA_ACTION_LOW + PANDA_ACTION_HIGH) / np.float32(2.0)
    for episode_index_value, length in enumerate((2, 3)):
        source_identity = f"sha256:source-{episode_index_value}"
        derived_identity = f"sha256:derived-{episode_index_value}"
        template_identity = f"phase2b6:test:{episode_index_value}"
        actions = np.repeat(midpoint[None, :], repeats=length, axis=0)
        for frame_index in range(length):
            dataset.add_frame(
                {
                    IMAGE_FEATURE: np.full(
                        (256, 256, 3),
                        20 + episode_index_value * 40 + frame_index,
                        dtype=np.uint8,
                    ),
                    STATE_FEATURE: np.full(
                        9, episode_index_value + frame_index / 10, dtype=np.float32
                    ),
                    ACTION_FEATURE: actions[frame_index],
                    TASK_ID_FEATURE: task_id,
                    SKILL_FEATURE: skill,
                    SOURCE_IDENTITY_FEATURE: source_identity,
                    DERIVED_IDENTITY_FEATURE: derived_identity,
                    TEMPLATE_ID_FEATURE: template_identity,
                    SPLIT_FEATURE: "train",
                    "task": instruction,
                }
            )
        dataset.save_episode(parallel_encoding=False)
        episode_index.append(
            {
                "local_episode_index": episode_index_value,
                "global_frame_start": global_frame,
                "frame_count": length,
                "source_episode_id": episode_index_value,
                "source_trajectory_identity": source_identity,
                "derived_episode_identity": derived_identity,
                "action_sha256": (
                    "sha256:" + hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest()
                ),
                "instruction_template_id": template_identity,
                "instruction": instruction,
            }
        )
        global_frame += length
    dataset.finalize()
    sidecar = {
        "task_id": task_id,
        "primary_split": "train",
        "repo_id": repo_id,
        "episode_count": 2,
        "frame_count": 5,
        "episode_index": episode_index,
    }
    report, equality = _readback_one_root(
        root=root,
        sidecar=sidecar,
        visual_shift=False,
    )
    assert report["passed"] is True
    assert report["api_episode_readback_count"] == 2
    assert report["frame_level_structural_validation_count"] == 5
    assert report["video"]["decoded_frame_count"] == 5
    assert report["failures"] == {
        "state": 0,
        "action": 0,
        "metadata": 0,
        "index_or_timestamp": 0,
        "api_image": 0,
        "api_instruction": 0,
    }
    assert all(row["action_exact"] is True for row in equality)
