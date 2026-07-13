"""Native-Linux target tests for real M3A-to-M3B rendering and export."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import pytest

from langmani.datasets.lerobot_export import export_lerobot_dataset
from langmani.datasets.lerobot_types import ExportMode, LeRobotExportConfig, SplitConfig
from langmani.datasets.lerobot_writer import inspect_lerobot_runtime

pytestmark = [
    pytest.mark.integration,
    pytest.mark.target,
    pytest.mark.gpu,
    pytest.mark.rendering,
    pytest.mark.video,
    pytest.mark.slow,
]


def _require_native_linux() -> None:
    if platform.system() != "Linux" or "microsoft" in platform.release().lower():
        pytest.skip("real M3B export requires native Linux")


def _paths(prefix: str) -> tuple[Path, Path]:
    source = os.getenv(f"LANGMANI_{prefix}_SOURCE_ROOT")
    report = os.getenv(f"LANGMANI_{prefix}_SOURCE_REPORT")
    if not source or not report:
        pytest.skip(f"set LANGMANI_{prefix}_SOURCE_ROOT and LANGMANI_{prefix}_SOURCE_REPORT")
    return Path(source).resolve(), Path(report).resolve()


def _config(
    source: Path,
    report: Path,
    output: Path,
    *,
    mode: ExportMode,
) -> LeRobotExportConfig:
    payload = json.loads(report.read_text(encoding="utf-8"))
    runtime = inspect_lerobot_runtime()
    return LeRobotExportConfig(
        source_root=str(source),
        output_root=str(output),
        repo_id="langmani/target-real-export",
        expected_source_collection_run_id=payload["collection_run_id"],
        expected_source_run_fingerprint=payload["config_fingerprint"],
        expected_source_archive_digest=payload["source_archive_digest"],
        mode=mode,
        split_config=SplitConfig.smoke() if mode is ExportMode.SMOKE else SplitConfig.full(),
        pyav_version=runtime.av,
        libavcodec_version=dict(runtime.av_libraries)["libavcodec"],
    )


def test_real_six_task_scene_group_export_and_deterministic_reexport(tmp_path: Path) -> None:
    _require_native_linux()
    source, report = _paths("M3A_SMOKE")
    first = export_lerobot_dataset(
        _config(source, report, tmp_path / "first", mode=ExportMode.SMOKE),
        source_verification_report=report,
    )
    second = export_lerobot_dataset(
        _config(source, report, tmp_path / "second", mode=ExportMode.SMOKE),
        source_verification_report=report,
    )

    assert first.validation_report.passed
    assert first.summary.total_scene_groups == 1
    assert first.summary.total_episodes == 6
    assert len(first.summary.task_episode_counts) == 6
    assert {record.source_scene_group_id for record in first.manifest.episodes} == {
        first.manifest.episodes[0].source_scene_group_id
    }
    assert len({record.task_id for record in first.manifest.episodes}) == 6
    assert first.manifest.export_fingerprint == second.manifest.export_fingerprint
    assert first.manifest.to_dict() == second.manifest.to_dict()


def test_real_full_export_has_360_source_aligned_episodes(tmp_path: Path) -> None:
    _require_native_linux()
    if os.getenv("LANGMANI_RUN_M3B_FULL") != "1":
        pytest.skip("set LANGMANI_RUN_M3B_FULL=1 to authorize the long full export")
    source, report = _paths("M3A_FULL")
    outcome = export_lerobot_dataset(
        _config(source, report, tmp_path / "full", mode=ExportMode.FULL),
        source_verification_report=report,
    )

    assert outcome.validation_report.passed
    assert outcome.summary.total_scene_groups == 60
    assert outcome.summary.total_episodes == 360
    assert set(outcome.summary.task_episode_counts.values()) == {60}
    assert outcome.summary.split_episode_counts == {
        "train": 288,
        "validation": 36,
        "test": 36,
    }
