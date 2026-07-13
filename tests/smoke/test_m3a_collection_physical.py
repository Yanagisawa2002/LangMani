"""Native-Linux physical smoke for one complete M3A counterfactual group."""

from __future__ import annotations

import os
import platform
from pathlib import Path

import pytest

from langmani.collection import RawDemonstrationCollector, inspect_raw_dataset
from langmani.collection.manifest import installed_runtime_versions
from langmani.datasets.types import CollectionConfig, CollectionStatus


def _is_native_linux() -> bool:
    release = platform.release().lower()
    return (
        platform.system() == "Linux" and "microsoft" not in release and not os.getenv("WSL_INTEROP")
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not _is_native_linux(),
    reason="M3A expert recording and action replay require native Linux",
)
def test_collect_replay_and_inspect_one_complete_counterfactual_group(
    tmp_path: Path,
) -> None:
    config = CollectionConfig(
        target_complete_scene_count=1,
        maximum_candidate_scene_count=1,
        maximum_expert_attempts_per_task=3,
        raw_output_root=str(tmp_path / "m3a-smoke"),
        shard_size=6,
        runtime_versions=installed_runtime_versions(),
    )

    summary = RawDemonstrationCollector(config).collect()
    inspected = inspect_raw_dataset(config.raw_output_root)

    assert summary.status is CollectionStatus.COMPLETE
    assert summary.accepted_scene_group_count == 1
    assert summary.accepted_episode_count == 6
    assert inspected.to_dict() == summary.to_dict()
