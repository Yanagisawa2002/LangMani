"""CPU-safe lifecycle and staging tests for the M3B LeRobot adapter."""

from __future__ import annotations

import inspect
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import langmani.datasets.lerobot_writer as writer_module
from langmani.datasets.lerobot_types import FeatureContract
from langmani.datasets.lerobot_writer import (
    COMPLETION_MARKER,
    LeRobotWriterAdapter,
    LeRobotWriterError,
    WriterState,
    inspect_lerobot_runtime,
    prepare_staging_layout,
    promote_staged_dataset,
    write_completion_marker,
)

_FINGERPRINT = "sha256:" + "1" * 64


class _FakeDataset:
    def __init__(self) -> None:
        self.pending = False
        self.frames: list[dict[str, object]] = []
        self.saved = 0
        self.finalized = 0
        self.cleared = 0

    def add_frame(self, frame: dict[str, object]) -> None:
        self.frames.append(dict(frame))
        self.pending = True

    def has_pending_frames(self) -> bool:
        return self.pending

    def save_episode(self, *, parallel_encoding: bool) -> None:
        assert parallel_encoding is False
        self.saved += 1
        self.pending = False

    def clear_episode_buffer(self, *, delete_images: bool) -> None:
        assert delete_images is True
        self.pending = False
        self.cleared += 1

    def finalize(self) -> None:
        self.finalized += 1


def _frame() -> dict[str, object]:
    return {
        "rgb": np.zeros((256, 256, 3), dtype=np.uint8),
        "state": np.zeros(9, dtype=np.float32),
        "action": np.zeros(8, dtype=np.float32),
        "task": "Pick up the red cube and place it in the left bin.",
    }


def test_writer_adapter_enforces_open_save_finalize_once() -> None:
    dataset = _FakeDataset()
    adapter = LeRobotWriterAdapter(dataset, FeatureContract())
    values = _frame()

    adapter.add_policy_frame(**values)  # type: ignore[arg-type]
    adapter.save_episode()
    adapter.finalize_once()

    assert adapter.state is WriterState.FINALIZED
    assert adapter.saved_episodes == 1
    assert adapter.finalize_calls == 1
    assert dataset.finalized == 1
    assert dataset.frames[0]["task"] == values["task"]
    with pytest.raises(LeRobotWriterError, match="finalized"):
        adapter.finalize_once()
    with pytest.raises(LeRobotWriterError, match="finalized"):
        adapter.add_policy_frame(**values)  # type: ignore[arg-type]


def test_writer_rejects_empty_episode_and_unsaved_finalize() -> None:
    adapter = LeRobotWriterAdapter(_FakeDataset(), FeatureContract())
    with pytest.raises(LeRobotWriterError, match="empty"):
        adapter.save_episode()
    values = _frame()
    adapter.add_policy_frame(**values)  # type: ignore[arg-type]
    with pytest.raises(LeRobotWriterError, match="unsaved"):
        adapter.finalize_once()


def test_failed_writer_cleanup_uses_public_methods_and_stays_failed() -> None:
    dataset = _FakeDataset()
    adapter = LeRobotWriterAdapter(dataset, FeatureContract())
    adapter.add_policy_frame(**_frame())  # type: ignore[arg-type]

    errors = adapter.close_failed()

    assert errors == ()
    assert adapter.state is WriterState.FAILED
    assert dataset.cleared == 1
    assert dataset.finalized == 1
    with pytest.raises(LeRobotWriterError, match="failed"):
        adapter.save_episode()


def test_staging_requires_explicit_owned_cleanup(tmp_path: Path) -> None:
    destination = tmp_path / "final"
    first = prepare_staging_layout(destination, _FINGERPRINT)
    (first.container / "diagnostic.txt").write_text("failed", encoding="utf-8")

    with pytest.raises(LeRobotWriterError, match="clean-staging"):
        prepare_staging_layout(destination, _FINGERPRINT)

    second = prepare_staging_layout(destination, _FINGERPRINT, clean_staging=True)
    assert second == first
    marker = json.loads((second.container / "staging.json").read_text(encoding="utf-8"))
    assert marker["export_fingerprint"] == _FINGERPRINT


def test_staging_refuses_unowned_cleanup_and_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "final"
    layout = prepare_staging_layout(destination, _FINGERPRINT)
    (layout.container / "staging.json").write_text("{}", encoding="utf-8")
    with pytest.raises(LeRobotWriterError, match="unowned"):
        prepare_staging_layout(destination, _FINGERPRINT, clean_staging=True)
    shutil.rmtree(layout.container)
    destination.mkdir()
    with pytest.raises(LeRobotWriterError, match="incomplete promoted"):
        prepare_staging_layout(destination, _FINGERPRINT)
    with pytest.raises(LeRobotWriterError, match="unowned incomplete"):
        prepare_staging_layout(destination, _FINGERPRINT, clean_staging=True)


def test_explicit_cleanup_recovers_only_an_owned_promoted_incomplete_destination(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "final"
    manifest = destination / "langmani" / "export_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"export_fingerprint": _FINGERPRINT}),
        encoding="utf-8",
    )

    with pytest.raises(LeRobotWriterError, match="incomplete promoted"):
        prepare_staging_layout(destination, _FINGERPRINT)

    layout = prepare_staging_layout(destination, _FINGERPRINT, clean_staging=True)
    assert not destination.exists()
    assert layout.container.is_dir()

    shutil.rmtree(layout.container)
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"export_fingerprint": "sha256:" + "2" * 64}))
    with pytest.raises(LeRobotWriterError, match="unowned incomplete"):
        prepare_staging_layout(destination, _FINGERPRINT, clean_staging=True)


def test_atomic_promotion_then_exclusive_completion_marker(tmp_path: Path) -> None:
    destination = tmp_path / "final"
    layout = prepare_staging_layout(destination, _FINGERPRINT)
    layout.dataset_root.mkdir()
    (layout.dataset_root / "meta.json").write_text("{}", encoding="utf-8")

    promoted = promote_staged_dataset(layout)
    shutil.rmtree(layout.container)
    marker = write_completion_marker(promoted, _FINGERPRINT)

    assert promoted == destination
    assert marker == destination / COMPLETION_MARKER
    assert json.loads(marker.read_text(encoding="utf-8"))["export_fingerprint"] == _FINGERPRINT
    with pytest.raises(LeRobotWriterError, match="already exists"):
        write_completion_marker(promoted, _FINGERPRINT)
    with pytest.raises(LeRobotWriterError, match="immutable"):
        prepare_staging_layout(promoted, _FINGERPRINT, clean_staging=True)


@pytest.mark.parametrize("fingerprint", ["", "sha256:abc", "1" * 64])
def test_staging_rejects_malformed_fingerprint(tmp_path: Path, fingerprint: str) -> None:
    with pytest.raises(LeRobotWriterError, match="fingerprint"):
        prepare_staging_layout(tmp_path / "final", fingerprint)


def test_writer_create_rejects_existing_dataset_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "exists"
    root.mkdir()
    monkeypatch.setattr(
        "langmani.datasets.lerobot_writer.inspect_lerobot_runtime",
        lambda: SimpleNamespace(),
    )
    with pytest.raises(LeRobotWriterError, match="must not already exist"):
        LeRobotWriterAdapter.create(
            root=root,
            repo_id="langmani/test",
            fps=20,
            feature_contract=FeatureContract(),
            codec=SimpleNamespace(),  # type: ignore[arg-type]
            data_files_size_in_mb=1,
            video_files_size_in_mb=1,
        )


def test_installed_writer_api_drift_fails_with_a_clear_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_signature = inspect.signature

    def incompatible_create_signature(value: object) -> inspect.Signature:
        if getattr(value, "__name__", None) == "create":
            return inspect.Signature()
        return real_signature(value)

    monkeypatch.setattr(writer_module, "_distribution_version", lambda _name: "0.6.0")
    monkeypatch.setattr(writer_module.inspect, "signature", incompatible_create_signature)

    with pytest.raises(LeRobotWriterError, match="create is missing required parameters"):
        inspect_lerobot_runtime()
