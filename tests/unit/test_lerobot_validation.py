"""CPU-safe independent M3B validator tests with explicit fixture boundaries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import langmani.datasets.lerobot_validation as validation
from langmani.datasets.frame_alignment import RawRenderDigest
from langmani.datasets.identity import COLLECTION_SCHEMA_VERSION, sha256_hex
from langmani.datasets.lerobot_types import (
    ACTION_FEATURE_KEY,
    IMAGE_FEATURE_KEY,
    LEROBOT_EXPORT_SCHEMA_VERSION,
    STATE_FEATURE_KEY,
    DatasetSplit,
    EpisodeExportRecord,
    ExportMode,
    LeRobotExportConfig,
    LeRobotExportManifest,
    SplitConfig,
    stable_export_fingerprint,
)
from langmani.datasets.lerobot_writer import COMPLETION_MARKER
from langmani.datasets.splits import build_split_assignments, episode_indices_by_split

SOURCE_COLLECTION_ID = "langmani-m3a-collection-run-" + "1" * 64
SOURCE_RUN_FINGERPRINT = "sha256:" + "a" * 64
SOURCE_ARCHIVE_DIGEST = "sha256:" + "b" * 64
CANONICAL_TASKS = (
    "Pick up the red cube and place it in the left bin.",
    "Pick up the red cube and place it in the right bin.",
    "Pick up the green cube and place it in the left bin.",
    "Pick up the green cube and place it in the right bin.",
    "Pick up the blue cube and place it in the left bin.",
    "Pick up the blue cube and place it in the right bin.",
)
OBJECTS = ("red_cube", "red_cube", "green_cube", "green_cube", "blue_cube", "blue_cube")
BINS = ("left_bin", "right_bin", "left_bin", "right_bin", "left_bin", "right_bin")


@dataclass
class _Fixture:
    root: Path
    config: LeRobotExportConfig
    manifest: LeRobotExportManifest
    source: Any
    dataset: Any
    raw_by_id: dict[str, Any]
    frames: list[np.ndarray]
    open_calls: list[Path]
    reconstruct_calls: list[tuple[int, int]]


class _TasksILoc:
    def __init__(self, texts: tuple[str, ...]) -> None:
        self._texts = texts

    def __getitem__(self, index: int) -> Any:
        return SimpleNamespace(name=self._texts[index])


class _Tasks:
    def __init__(self, texts: tuple[str, ...]) -> None:
        self.index = texts
        self.iloc = _TasksILoc(texts)


class _FakeMeta:
    def __init__(self, episodes: list[dict[str, Any]]) -> None:
        self.info = SimpleNamespace(codebase_version="v3.0")
        self.episodes = episodes
        self.tasks = _Tasks(CANONICAL_TASKS)

    def get_video_file_path(self, episode_index: int, video_key: str) -> Path:
        assert 0 <= episode_index < 6
        assert video_key == IMAGE_FEATURE_KEY
        return Path("videos/observation.images.base_camera/chunk-000/file-000.mp4")


class _FakeDataset:
    def __init__(
        self,
        *,
        features: dict[str, dict[str, Any]],
        rows: list[dict[str, Any]],
        episodes: list[dict[str, Any]],
        frames: list[np.ndarray],
    ) -> None:
        self.features = features
        self.hf_dataset = rows
        self.meta = _FakeMeta(episodes)
        self.fps = 20
        self.num_episodes = 6
        self.num_frames = len(rows)
        self._rows = rows
        self._frames = frames

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._rows[index]
        task_index = int(np.asarray(row["task_index"]).reshape(-1)[0])
        return {
            **row,
            IMAGE_FEATURE_KEY: np.transpose(self._frames[index], (2, 0, 1)),
            "task": CANONICAL_TASKS[task_index],
        }


class _RawEpisode:
    def __init__(self, record: Any, actions: np.ndarray, episode_index: int) -> None:
        self.record = record
        self.actions = actions
        self._episode_index = episode_index

    @property
    def transition_count(self) -> int:
        return len(self.actions)

    def training_state_at(self, index: int) -> dict[str, Any]:
        return {"episode_index": self._episode_index, "frame_index": index}


class _FakeReader:
    def __init__(self, source: Any, raw_by_id: dict[str, _RawEpisode]) -> None:
        assert source is not None
        self._raw_by_id = raw_by_id

    def read_episode(self, raw_trajectory_id: str) -> _RawEpisode:
        return self._raw_by_id[raw_trajectory_id]

    def revalidate_source(self) -> None:
        return None


class _FakeReconstructor:
    def __init__(
        self,
        frames: list[np.ndarray],
        states: list[np.ndarray],
        calls: list[tuple[int, int]],
    ) -> None:
        self._frames = frames
        self._states = states
        self._calls = calls
        self._open = False

    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False

    def begin_episode(self, record: Any) -> None:
        assert self._open
        assert record.scene_id == "scene-123"

    def reconstruct(self, state: dict[str, Any]) -> Any:
        episode_index = int(state["episode_index"])
        frame_index = int(state["frame_index"])
        self._calls.append((episode_index, frame_index))
        global_index = episode_index * 2 + frame_index
        return SimpleNamespace(
            rgb=np.array(self._frames[global_index], copy=True),
            state=np.array(self._states[global_index], copy=True),
        )


class _FakeFrame:
    def __init__(self, value: np.ndarray) -> None:
        self._value = value

    def to_ndarray(self, *, format: str) -> np.ndarray:
        assert format == "rgb24"
        return np.array(self._value, copy=True)


class _FakeContainer:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames
        self.streams = SimpleNamespace(video=(object(),))

    def __enter__(self) -> _FakeContainer:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def decode(self, stream: object) -> Any:
        assert stream is self.streams.video[0]
        return iter(_FakeFrame(frame) for frame in self._frames)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _frame(episode_index: int, frame_index: int) -> np.ndarray:
    value = np.full((256, 256, 3), 20 + episode_index * 3 + frame_index, dtype=np.uint8)
    value[0, 0] = np.array([1, 2, 3], dtype=np.uint8)
    return value


def _state(episode_index: int, frame_index: int) -> np.ndarray:
    return (np.arange(9, dtype=np.float32) + episode_index + frame_index / 10).astype(np.float32)


def _action(episode_index: int, frame_index: int) -> np.ndarray:
    return (np.arange(8, dtype=np.float32) + episode_index / 10 + frame_index).astype(np.float32)


def _make_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Fixture:
    root = tmp_path / "derived"
    group_id = "scene-group-123"
    config = LeRobotExportConfig(
        source_root="<authoritative-m3a-source>",
        output_root=".",
        repo_id="langmani/pick-place-by-instruction-v0",
        expected_source_collection_run_id=SOURCE_COLLECTION_ID,
        expected_source_run_fingerprint=SOURCE_RUN_FINGERPRINT,
        expected_source_archive_digest=SOURCE_ARCHIVE_DIGEST,
        mode=ExportMode.SMOKE,
        split_config=SplitConfig.smoke(split_seed=17),
    )
    assignments = build_split_assignments((group_id,), config.split_config)
    frames = [_frame(ep, frame) for ep in range(6) for frame in range(2)]
    states = [_state(ep, frame) for ep in range(6) for frame in range(2)]
    actions = [_action(ep, frame) for ep in range(6) for frame in range(2)]
    raw_records: list[Any] = []
    episode_records: list[EpisodeExportRecord] = []
    raw_by_id: dict[str, _RawEpisode] = {}
    for episode_index in range(6):
        raw_id = f"raw-episode-{episode_index}"
        task_spec = SimpleNamespace(
            target_object_id=OBJECTS[episode_index],
            target_bin_id=BINS[episode_index],
            instruction_template_id="canonical_v0",
        )
        raw = SimpleNamespace(
            raw_trajectory_id=raw_id,
            source_shard_id="source-shard-0",
            h5_group=f"traj_{episode_index}",
            h5_path="accepted/shard-00000.h5",
            json_path="accepted/shard-00000.json",
            h5_sha256="c" * 64,
            json_sha256="d" * 64,
            elapsed_steps=2,
            scene_seed=123,
            scene_id="scene-123",
            task_spec=task_spec,
            task_id=f"task-{episode_index}",
            canonical_instruction=CANONICAL_TASKS[episode_index],
        )
        raw_records.append(raw)
        episode_actions = np.stack(actions[episode_index * 2 : episode_index * 2 + 2])
        raw_by_id[raw_id] = _RawEpisode(raw, episode_actions, episode_index)
        digest = RawRenderDigest()
        digest.add(0, frames[episode_index * 2])
        digest.add(1, frames[episode_index * 2 + 1])
        source_checksum = sha256_hex(
            {
                "h5_group": raw.h5_group,
                "h5_sha256": raw.h5_sha256,
                "json_sha256": raw.json_sha256,
                "raw_trajectory_id": raw.raw_trajectory_id,
            }
        )
        episode_records.append(
            EpisodeExportRecord(
                source_collection_run_id=SOURCE_COLLECTION_ID,
                source_run_fingerprint=SOURCE_RUN_FINGERPRINT,
                source_archive_digest=SOURCE_ARCHIVE_DIGEST,
                source_episode_id=raw_id,
                source_scene_group_id=group_id,
                source_scene_seed=123,
                scene_id="scene-123",
                task_id=raw.task_id,
                target_object_id=task_spec.target_object_id,
                target_bin_id=task_spec.target_bin_id,
                instruction_template_id="canonical_v0",
                canonical_instruction=raw.canonical_instruction,
                source_shard_id="source-shard-0",
                source_shard_path=raw.h5_path,
                source_trajectory_key=raw.h5_group,
                source_h5_sha256=raw.h5_sha256,
                source_json_sha256=raw.json_sha256,
                source_checksum=source_checksum,
                source_frame_count=2,
                lerobot_episode_index=episode_index,
                split=DatasetSplit.TRAIN,
                raw_render_digest=digest.hexdigest(),
                output_frame_count=2,
            )
        )
    ordered_ids = tuple(item.source_episode_id for item in episode_records)
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
        episodes=tuple(episode_records),
        runtime_versions={"lerobot": "0.6.0", "av": "15.1.0"},
        finalized=True,
    )
    source = SimpleNamespace(
        manifest=SimpleNamespace(
            collection_run_id=SOURCE_COLLECTION_ID,
            config=SimpleNamespace(sim_backend="physx_cpu"),
        ),
        config_fingerprint=SOURCE_RUN_FINGERPRINT,
        archive_digest=SOURCE_ARCHIVE_DIGEST,
        ordered_groups=(
            SimpleNamespace(
                accepted_scene_group_id=group_id,
                episodes=tuple(raw_records),
            ),
        ),
        ordered_episodes=tuple(raw_records),
    )
    features = config.feature_contract.to_lerobot_features() | {
        "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
        "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
        "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
        "index": {"dtype": "int64", "shape": (1,), "names": None},
        "task_index": {"dtype": "int64", "shape": (1,), "names": None},
    }
    features[IMAGE_FEATURE_KEY]["info"] = {
        "video.height": 256,
        "video.width": 256,
        "video.codec": config.video_codec.codec,
        "video.pix_fmt": config.video_codec.pixel_format,
        "video.fps": 20,
        "video.channels": 3,
        "has_audio": False,
        "video.g": 2,
        "video.crf": config.video_codec.crf,
        "video.preset": config.video_codec.preset,
        "video.fast_decode": 0,
        "video.video_backend": "pyav",
        "is_depth_map": False,
    }
    rows: list[dict[str, Any]] = []
    episode_metadata: list[dict[str, Any]] = []
    video_key = f"videos/{IMAGE_FEATURE_KEY}"
    for episode_index in range(6):
        start = episode_index * 2
        episode_metadata.append(
            {
                "episode_index": episode_index,
                "length": 2,
                "tasks": [CANONICAL_TASKS[episode_index]],
                "dataset_from_index": start,
                "dataset_to_index": start + 2,
                f"{video_key}/from_timestamp": start / 20,
                f"{video_key}/to_timestamp": (start + 2) / 20,
            }
        )
        for frame_index in range(2):
            global_index = start + frame_index
            rows.append(
                {
                    STATE_FEATURE_KEY: states[global_index],
                    ACTION_FEATURE_KEY: actions[global_index],
                    "timestamp": np.array([frame_index / 20], dtype=np.float32),
                    "frame_index": np.array([frame_index], dtype=np.int64),
                    "episode_index": np.array([episode_index], dtype=np.int64),
                    "index": np.array([global_index], dtype=np.int64),
                    "task_index": np.array([episode_index], dtype=np.int64),
                }
            )
    dataset = _FakeDataset(
        features=features,
        rows=rows,
        episodes=episode_metadata,
        frames=frames,
    )

    sidecar = root / "langmani"
    _write_json(sidecar / "export_config.json", config.portable_dict())
    _write_json(sidecar / "export_manifest.json", manifest.to_dict())
    _write_json(
        sidecar / "source_episode_mapping.json",
        {
            "schema_version": "langmani-m3b-source-mapping-v1",
            "export_fingerprint": manifest.export_fingerprint,
            "episodes": [item.to_dict() for item in manifest.episodes],
        },
    )
    _write_json(
        sidecar / "split_manifest.json",
        {
            "schema_version": "langmani-m3b-splits-v1",
            "export_fingerprint": manifest.export_fingerprint,
            "split_config": config.split_config.to_dict(),
            "scene_group_assignments": [item.to_dict() for item in assignments],
            "episode_indices": {
                key: list(value)
                for key, value in episode_indices_by_split(manifest.episodes).items()
            },
        },
    )
    (sidecar / "dataset_card.md").write_text("# fixture only\n", encoding="utf-8")
    _write_json(root / "meta" / "info.json", {"fixture": True})
    _write_json(root / "meta" / "stats.json", {"fixture": True})
    for path in (
        root / "meta" / "tasks.parquet",
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        root / "data" / "chunk-000" / "file-000.parquet",
        root / "videos" / IMAGE_FEATURE_KEY / "chunk-000" / "file-000.mp4",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")

    open_calls: list[Path] = []
    reconstruct_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(validation, "_validate_parquet_files", lambda paths: [])
    monkeypatch.setattr(validation, "_validate_local_owned_references", lambda root, files: [])
    monkeypatch.setattr(validation, "_load_local_dataset", lambda repo_id, root: dataset)
    monkeypatch.setattr(
        validation, "validate_m3a_source_for_export", lambda *args, **kwargs: source
    )
    monkeypatch.setattr(
        validation,
        "ValidatedRawEpisodeReader",
        lambda value: _FakeReader(value, raw_by_id),
    )
    monkeypatch.setattr(
        validation,
        "_make_reconstructor",
        lambda source, config: _FakeReconstructor(frames, states, reconstruct_calls),
    )

    def open_video(path: Path) -> _FakeContainer:
        open_calls.append(path)
        return _FakeContainer(frames)

    monkeypatch.setattr(validation, "_open_pyav_container", open_video)
    return _Fixture(
        root=root,
        config=config,
        manifest=manifest,
        source=source,
        dataset=dataset,
        raw_by_id=raw_by_id,
        frames=frames,
        open_calls=open_calls,
        reconstruct_calls=reconstruct_calls,
    )


def _validate(fixture: _Fixture, **kwargs: Any) -> tuple[Any, Any]:
    return validation.validate_lerobot_dataset(
        fixture.root,
        source_root="authoritative-source",
        source_verification_report="verification.json",
        require_completion_marker=False,
        **kwargs,
    )


@pytest.mark.fixture
def test_full_fixture_audit_passes_without_claiming_physical_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)

    report, summary = _validate(fixture)

    assert report.passed
    assert len(report.source_alignments) == 6
    assert len(report.video_results) == 6
    assert all(item.sampled_frame_indices == (0, 1) for item in report.video_results)
    assert all(item.actions_match and item.policy_states_match for item in report.source_alignments)
    assert len(fixture.open_calls) == 1  # one physical MP4 shared by all six episodes
    assert fixture.reconstruct_calls == [(ep, frame) for ep in range(6) for frame in range(2)]
    assert summary.total_episodes == 6
    assert summary.total_frames == 12
    assert dict(summary.task_episode_counts) == {f"task-{index}": 1 for index in range(6)}


def test_metadata_only_is_explicitly_non_accepting_and_skips_physical_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)

    report, summary = _validate(fixture, metadata_only=True)

    assert not report.passed
    assert not report.video_decode_validated
    assert not report.source_alignment_validated
    assert not report.dataloader_validated
    assert not report.source_alignments
    assert not report.video_results
    assert any("metadata-only" in reason for reason in report.failure_reasons)
    assert not fixture.open_calls
    assert not fixture.reconstruct_calls
    assert summary.total_episodes == 6


def test_action_drift_returns_typed_failing_source_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    fixture.dataset.hf_dataset[0][ACTION_FEATURE_KEY] = np.full(8, 99, dtype=np.float32)

    report, _ = _validate(fixture)

    assert not report.passed
    assert not report.source_alignment_validated
    assert not report.source_alignments[0].actions_match
    assert "derived actions differ" in " ".join(report.source_alignments[0].failure_reasons)
    assert all(item.actions_match for item in report.source_alignments[1:])


def test_privileged_feature_is_reported_without_hiding_other_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    fixture.dataset.features["target_object_index"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    }

    report, _ = _validate(fixture)

    assert not report.passed
    assert not report.privileged_leakage_validated
    assert not report.feature_schema_validated
    assert report.source_alignment_validated
    assert any("target_object_index" in reason for reason in report.failure_reasons)


def test_non_v3_dataset_is_rejected_even_when_other_fixture_evidence_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    fixture.dataset.meta.info.codebase_version = "v2.1"

    report, _ = _validate(fixture)

    assert not report.passed
    assert not report.dataset_load_validated
    assert "not LeRobotDataset v3.0" in " ".join(report.failure_reasons)


@pytest.mark.parametrize(
    ("field", "value"),
    (("digest", "f" * 64), ("rank", 7)),
)
def test_stored_split_mapping_must_match_independent_canonical_recomputation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    sidecar = fixture.root / "langmani"
    manifest = json.loads((sidecar / "export_manifest.json").read_text(encoding="utf-8"))
    splits = json.loads((sidecar / "split_manifest.json").read_text(encoding="utf-8"))
    manifest["split_assignments"][0][field] = value
    splits["scene_group_assignments"][0][field] = value
    _write_json(sidecar / "export_manifest.json", manifest)
    _write_json(sidecar / "split_manifest.json", splits)

    report, _ = _validate(fixture)

    assert not report.passed
    assert not report.split_integrity_validated
    assert "canonical recomputation" in " ".join(report.failure_reasons)


@pytest.mark.fixture
def test_every_actual_video_file_is_sequentially_decoded_even_when_unreferenced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    extra = fixture.root / "videos" / IMAGE_FEATURE_KEY / "chunk-000" / "unexpected-file-001.mp4"
    extra.write_bytes(b"fixture")

    report, summary = _validate(fixture)

    assert not report.passed
    assert not report.video_decode_validated
    assert len(fixture.open_calls) == 2
    assert extra.resolve() in fixture.open_calls
    assert summary.video_file_count == 2
    assert any("not referenced" in reason for reason in report.failure_reasons)


@pytest.mark.fixture
def test_promoted_dataset_requires_matching_completion_and_validation_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    staged_report, summary = _validate(fixture)
    assert staged_report.passed
    _write_json(fixture.root / "langmani" / "dataset_summary.json", summary.to_dict())
    _write_json(fixture.root / "langmani" / "validation_report.json", staged_report.to_dict())
    _write_json(
        fixture.root / COMPLETION_MARKER,
        {
            "schema_version": "langmani-m3b-completion-v1",
            "export_fingerprint": fixture.manifest.export_fingerprint,
        },
    )

    report, final_summary = validation.validate_lerobot_dataset(
        fixture.root,
        source_root="authoritative-source",
        source_verification_report="verification.json",
        require_completion_marker=True,
    )

    assert report.passed
    assert final_summary == summary

    _write_json(
        fixture.root / COMPLETION_MARKER,
        {
            "schema_version": "langmani-m3b-completion-v1",
            "export_fingerprint": "sha256:" + "9" * 64,
        },
    )
    failing, _ = validation.validate_lerobot_dataset(
        fixture.root,
        source_root="authoritative-source",
        source_verification_report="verification.json",
        require_completion_marker=True,
    )
    assert not failing.passed
    assert any("completion marker" in reason for reason in failing.failure_reasons)


def test_missing_local_sidecars_fail_before_any_loader_or_hub_fallback(tmp_path: Path) -> None:
    root = tmp_path / "incomplete"
    root.mkdir()

    with pytest.raises(validation.LeRobotValidationError, match="sidecars are missing"):
        validation.validate_lerobot_dataset(
            root,
            source_root=tmp_path / "source",
            source_verification_report=tmp_path / "verification.json",
            require_completion_marker=False,
        )


def test_source_gate_failure_is_a_clear_non_bypassable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        validation,
        "validate_m3a_source_for_export",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stale source digest")),
    )

    with pytest.raises(validation.LeRobotValidationError, match="source gate failed"):
        _validate(fixture)


def test_public_local_loader_failure_is_wrapped_clearly_without_remote_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        validation,
        "_load_local_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("local reader corrupt")),
    )

    with pytest.raises(validation.LeRobotValidationError, match="local reload failed"):
        _validate(fixture)
