from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path

import numpy as np
import pytest

from langmani.v2.phase2b5 import PANDA_ACTION_HIGH, PANDA_ACTION_LOW
from langmani.v2.phase2b5_runtime import (
    convert_visual_pilot_to_lerobot,
    readback_lerobot_pilot,
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
def test_real_lerobot_060_pilot_creation_and_readback(tmp_path: Path) -> None:
    pilot_root = tmp_path / "pilot"
    output_root = tmp_path / "reports"
    dataset_root = tmp_path / "lerobot"
    tasks = []
    task_ids = ("PickCube-v1", "StackCube-v1", "PushCube-v1")
    for task_index, task_id in enumerate(task_ids):
        relative = Path(task_id) / f"episode_{task_index:06d}.npz"
        path = pilot_root / relative
        path.parent.mkdir(parents=True)
        length = task_index + 2
        np.savez_compressed(
            path,
            rgb=np.full((length, 256, 256, 3), task_index * 20, dtype=np.uint8),
            state=np.zeros((length, 9), dtype=np.float32),
            action=np.repeat(
                ((PANDA_ACTION_LOW + PANDA_ACTION_HIGH) / np.float32(2.0))[None, :],
                repeats=length,
                axis=0,
            ),
            timestamp=np.arange(length, dtype=np.float64) / 20.0,
            terminal_rgb=np.zeros((256, 256, 3), dtype=np.uint8),
        )
        tasks.append(
            {
                "task_id": task_id,
                "episodes": [
                    {
                        "episode_id": task_index,
                        "npz_relative_path": relative.as_posix(),
                    }
                ],
            }
        )
    visual_manifest = output_root / "visual_pilot_manifest.json"
    visual_manifest.parent.mkdir(parents=True)
    visual_manifest.write_text(json.dumps({"tasks": tasks}), encoding="utf-8")

    conversion = convert_visual_pilot_to_lerobot(
        pilot_root=pilot_root,
        dataset_root=dataset_root,
        output_root=output_root,
        visual_manifest_path=visual_manifest,
    )
    assert conversion["episode_count"] == 3
    assert conversion["frame_count"] == 9
    assert conversion["passed"] is False  # the real phase requires five episodes per task

    readback = readback_lerobot_pilot(
        dataset_root=dataset_root,
        output_root=output_root,
        conversion_manifest_path=output_root / "conversion_pilot_manifest.json",
    )
    assert readback["passed"] is True
    assert readback["metadata_episode_count"] == 3
    assert readback["metadata_frame_count"] == 9
    assert readback["observed_task_ids"] == sorted(task_ids)
