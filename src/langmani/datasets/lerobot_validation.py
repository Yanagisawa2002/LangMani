"""Independent local validation of a derived LangMani LeRobotDataset v3.

The validator deliberately starts from immutable on-disk sidecars and a fresh
``LeRobotDataset`` reader.  It never accepts exporter memory as evidence and it
never falls back to the Hugging Face Hub when a local dataset is incomplete.
M3A remains authoritative: full validation re-reads every accepted raw action
and pre-action state, reconstructs every policy observation, and audits every
decoded video frame.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from langmani.datasets.frame_alignment import (
    RawRenderDigest,
    deterministic_video_sample_indices,
    rgb_quality_metrics,
)
from langmani.datasets.identity import sha256_hex
from langmani.datasets.lerobot_source import (
    ValidatedM3ASource,
    ValidatedRawEpisodeReader,
    validate_m3a_source_for_export,
)
from langmani.datasets.lerobot_types import (
    ACTION_FEATURE_KEY,
    IMAGE_FEATURE_KEY,
    LEROBOT_MANAGED_FEATURE_KEYS,
    POLICY_FEATURE_KEYS,
    STATE_FEATURE_KEY,
    DatasetSplit,
    EpisodeExportRecord,
    ExportMode,
    LeRobotDatasetSummary,
    LeRobotExportConfig,
    LeRobotExportManifest,
    LeRobotValidationReport,
    SourceAlignmentResult,
    VideoValidationResult,
)
from langmani.datasets.lerobot_writer import COMPLETION_MARKER, SIDECAR_DIRECTORY
from langmani.datasets.observation_reconstruction import ManiSkillObservationReconstructor
from langmani.datasets.splits import build_split_assignments

_COMPLETION_SCHEMA = "langmani-m3b-completion-v1"
_SOURCE_MAPPING_SCHEMA = "langmani-m3b-source-mapping-v1"
_SPLIT_MANIFEST_SCHEMA = "langmani-m3b-splits-v1"
_EXPECTED_MANAGED_FEATURES: Mapping[str, Mapping[str, object]] = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}


class LeRobotValidationError(RuntimeError):
    """Raised when independent validation cannot safely load its authorities."""


@dataclass(frozen=True, slots=True)
class _LocalFiles:
    parquet_files: tuple[Path, ...]
    video_files: tuple[Path, ...]
    info: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _EpisodeBoundary:
    episode_index: int
    start: int
    stop: int
    length: int
    video_path: str
    video_start_frame: int
    video_stop_frame: int
    timestamps_match: bool


@dataclass(frozen=True, slots=True)
class _DerivedEpisode:
    boundary: _EpisodeBoundary
    actions: np.ndarray
    states: np.ndarray


@dataclass(slots=True)
class _DecodedEpisode:
    episode_index: int
    path: str
    expected_count: int
    sample_indices: tuple[int, ...]
    samples: dict[int, np.ndarray] = field(default_factory=dict)
    decoded_count: int = 0
    frames_nonempty: bool = True
    frames_nonuniform: bool = True
    failure_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _Sidecars:
    config: LeRobotExportConfig
    manifest: LeRobotExportManifest
    source_mapping: Mapping[str, object]
    split_manifest: Mapping[str, object]
    stored_summary: LeRobotDatasetSummary | None
    stored_report: LeRobotValidationReport | None


def validate_lerobot_dataset(
    dataset_root: str | Path,
    *,
    source_root: str | Path,
    source_verification_report: str | Path,
    require_completion_marker: bool,
    metadata_only: bool = False,
) -> tuple[LeRobotValidationReport, LeRobotDatasetSummary]:
    """Validate one local M3B dataset and return typed independent evidence.

    ``require_completion_marker=False`` is exclusively for the finalized writer
    inside the exporter's staging transaction.  A promoted immutable dataset is
    validated with ``require_completion_marker=True``.  Metadata-only mode is a
    diagnostic shortcut and intentionally returns a non-passing report because
    it does not establish video, source-frame, or DataLoader evidence.
    """
    if not isinstance(require_completion_marker, bool):
        raise TypeError("require_completion_marker must be a bool")
    if not isinstance(metadata_only, bool):
        raise TypeError("metadata_only must be a bool")

    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise LeRobotValidationError(f"local LeRobot dataset root does not exist: {root}")

    sidecars = _load_sidecars(root, require_completion_marker=require_completion_marker)
    local_files = _preflight_local_files(root)
    lifecycle_reasons = _validate_local_lifecycle(
        root,
        sidecars,
        require_completion_marker=require_completion_marker,
    )
    sidecar_reasons = _validate_sidecar_cross_references(sidecars)

    mode = "smoke" if sidecars.manifest.config.mode is ExportMode.SMOKE else "full"
    try:
        source = validate_m3a_source_for_export(
            source_root,
            verification_report_path=source_verification_report,
            mode=mode,
            expected_run_fingerprint=sidecars.manifest.source_run_fingerprint,
            expected_schema_version=sidecars.manifest.source_schema_version,
        )
    except Exception as error:
        raise LeRobotValidationError(
            "authoritative M3A source gate failed: "
            f"{type(error).__name__}: {str(error) or repr(error)}"
        ) from error

    source_reasons = _validate_manifest_against_source(sidecars.manifest, source)
    parquet_reasons = _validate_parquet_files(local_files.parquet_files)
    if parquet_reasons:
        # LeRobot attempts a Hub sync when its local reader cannot activate.  A
        # corrupt local Parquet tree is therefore a hard pre-load error, not a
        # reason to let the public constructor seek a remote replacement.
        raise LeRobotValidationError(
            "local Parquet preflight failed: " + "; ".join(parquet_reasons)
        )
    reference_reasons = _validate_local_owned_references(root, local_files)
    if reference_reasons:
        raise LeRobotValidationError(
            "local metadata references missing owned files; refusing Hub fallback: "
            + "; ".join(reference_reasons)
        )

    try:
        dataset = _load_local_dataset(sidecars.manifest.repo_id, root)
    except Exception as error:
        raise LeRobotValidationError(
            "public LeRobotDataset local reload failed: "
            f"{type(error).__name__}: {str(error) or repr(error)}"
        ) from error

    feature_reasons, leakage_reasons = _validate_feature_schema(
        getattr(dataset, "features", None), sidecars.manifest.config
    )
    try:
        derived, dataset_reasons = _audit_dataset_rows(dataset, sidecars.manifest, source)
    except LeRobotValidationError:
        raise
    except Exception as error:
        raise LeRobotValidationError(
            "public LeRobotDataset metadata/Parquet audit could not run: "
            f"{type(error).__name__}: {str(error) or repr(error)}"
        ) from error
    split_reasons = _validate_split_integrity(sidecars.manifest, source)

    summary = _build_summary(
        sidecars.manifest,
        source,
        parquet_count=len(local_files.parquet_files),
        video_count=len(local_files.video_files),
    )
    if sidecars.stored_summary is not None and sidecars.stored_summary != summary:
        lifecycle_reasons.append("stored dataset_summary.json differs from independent summary")
    if sidecars.stored_report is not None and (
        sidecars.stored_report.export_fingerprint != sidecars.manifest.export_fingerprint
        or not sidecars.stored_report.passed
    ):
        lifecycle_reasons.append(
            "stored validation_report.json is failing or belongs to another export"
        )

    source_alignments: tuple[SourceAlignmentResult, ...] = ()
    video_results: tuple[VideoValidationResult, ...] = ()
    video_reasons: list[str] = []
    alignment_reasons: list[str] = []
    dataloader_reasons: list[str] = []
    if metadata_only:
        video_reasons.append("metadata-only mode did not decode and compare policy videos")
        alignment_reasons.append(
            "metadata-only mode did not align raw actions or reconstructed states"
        )
        dataloader_reasons.append("metadata-only mode did not run direct indexing or DataLoader")
    elif len(derived) != len(sidecars.manifest.episodes):
        alignment_reasons.append("derived episode boundaries are incomplete")
        video_reasons.append("video alignment requires complete episode boundaries")
        dataloader_reasons.append("DataLoader smoke requires complete episode boundaries")
    else:
        decoded, video_file_reasons = _decode_all_videos(
            root,
            tuple(item.boundary for item in derived),
            actual_video_paths=tuple(
                _relative_posix(path, root) for path in local_files.video_files
            ),
            image_height=sidecars.manifest.config.image_height,
            image_width=sidecars.manifest.config.image_width,
        )
        video_reasons.extend(video_file_reasons)
        try:
            source_alignments, video_results = _validate_source_alignment(
                source,
                sidecars.manifest,
                derived,
                decoded,
            )
        except Exception as error:
            raise LeRobotValidationError(
                "full M3A/source-observation alignment could not initialize: "
                f"{type(error).__name__}: {str(error) or repr(error)}"
            ) from error
        video_reasons.extend(
            f"episode {item.lerobot_episode_index}: {reason}"
            for item in video_results
            for reason in item.failure_reasons
        )
        alignment_reasons.extend(
            f"episode {item.lerobot_episode_index}: {reason}"
            for item in source_alignments
            for reason in item.failure_reasons
        )
        dataloader_reasons.extend(_validate_dataloader(dataset, derived, sidecars.manifest))

    dataset_load_reasons = [
        *lifecycle_reasons,
        *sidecar_reasons,
        *source_reasons,
        *dataset_reasons,
    ]
    split_reasons = [*split_reasons]
    flags = {
        "dataset_load_validated": not dataset_load_reasons,
        "feature_schema_validated": not feature_reasons,
        "parquet_validated": not parquet_reasons,
        "video_decode_validated": not video_reasons
        and bool(video_results)
        and all(item.passed for item in video_results),
        "source_alignment_validated": not alignment_reasons
        and bool(source_alignments)
        and all(item.passed for item in source_alignments),
        "split_integrity_validated": not split_reasons,
        "privileged_leakage_validated": not leakage_reasons,
        "dataloader_validated": not dataloader_reasons,
    }
    all_reasons = _deduplicate(
        [
            *dataset_load_reasons,
            *feature_reasons,
            *video_reasons,
            *alignment_reasons,
            *split_reasons,
            *leakage_reasons,
            *dataloader_reasons,
        ]
    )
    passed = (
        all(flags.values())
        and all(item.passed for item in source_alignments)
        and all(item.passed for item in video_results)
    )
    if not passed and not all_reasons:
        all_reasons = ("one or more independent validation flags failed",)
    report = LeRobotValidationReport(
        export_fingerprint=sidecars.manifest.export_fingerprint,
        mode=sidecars.manifest.config.mode,
        source_alignments=source_alignments,
        video_results=video_results,
        passed=passed,
        failure_reasons=() if passed else all_reasons,
        **flags,
    )
    return report, summary


def _load_sidecars(root: Path, *, require_completion_marker: bool) -> _Sidecars:
    sidecar = root / SIDECAR_DIRECTORY
    required = (
        "export_config.json",
        "export_manifest.json",
        "source_episode_mapping.json",
        "split_manifest.json",
        "dataset_card.md",
    )
    missing = [name for name in required if not (sidecar / name).is_file()]
    if missing:
        raise LeRobotValidationError("required local sidecars are missing: " + ", ".join(missing))
    try:
        if not (sidecar / "dataset_card.md").read_text(encoding="utf-8").strip():
            raise LeRobotValidationError("dataset_card.md must not be empty")
    except (OSError, UnicodeError) as error:
        raise LeRobotValidationError(f"cannot read dataset_card.md: {error}") from error
    try:
        config = LeRobotExportConfig.from_dict(_read_json(sidecar / "export_config.json"))
        manifest = LeRobotExportManifest.from_dict(_read_json(sidecar / "export_manifest.json"))
        source_mapping = _read_json(sidecar / "source_episode_mapping.json")
        split_manifest = _read_json(sidecar / "split_manifest.json")
    except Exception as error:
        raise LeRobotValidationError(
            f"cannot load typed M3B sidecars: {type(error).__name__}: {error}"
        ) from error

    stored_summary: LeRobotDatasetSummary | None = None
    stored_report: LeRobotValidationReport | None = None
    if require_completion_marker:
        post = (sidecar / "dataset_summary.json", sidecar / "validation_report.json")
        if any(not path.is_file() for path in post):
            raise LeRobotValidationError(
                "completed dataset is missing dataset_summary.json or validation_report.json"
            )
        try:
            stored_summary = LeRobotDatasetSummary.from_dict(_read_json(post[0]))
            stored_report = LeRobotValidationReport.from_dict(_read_json(post[1]))
        except Exception as error:
            raise LeRobotValidationError(
                f"cannot load completed validation sidecars: {type(error).__name__}: {error}"
            ) from error
    return _Sidecars(
        config=config,
        manifest=manifest,
        source_mapping=source_mapping,
        split_manifest=split_manifest,
        stored_summary=stored_summary,
        stored_report=stored_report,
    )


def _preflight_local_files(root: Path) -> _LocalFiles:
    required = (
        root / "meta" / "info.json",
        root / "meta" / "stats.json",
        root / "meta" / "tasks.parquet",
    )
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    parquet_files = tuple(sorted(root.rglob("*.parquet")))
    data_parquet = tuple(path for path in parquet_files if path.parts[-3:-2] == ("data",))
    if not data_parquet:
        data_parquet = tuple(
            path for path in parquet_files if "data" in path.relative_to(root).parts
        )
    episode_parquet = tuple(
        path for path in parquet_files if path.relative_to(root).parts[:2] == ("meta", "episodes")
    )
    video_files = tuple(sorted(root.rglob("*.mp4")))
    if not data_parquet:
        missing.append("data/**/*.parquet")
    if not episode_parquet:
        missing.append("meta/episodes/**/*.parquet")
    if not video_files:
        missing.append("videos/**/*.mp4")
    if missing:
        raise LeRobotValidationError(
            "local dataset is incomplete; refusing Hub fallback: " + ", ".join(missing)
        )
    for path in (*required, *parquet_files, *video_files):
        if path.is_symlink():
            raise LeRobotValidationError(
                f"local dataset must not contain symlinked owned files: {path}"
            )
        _relative_posix(path, root)
    info = _read_json(required[0])
    if not isinstance(info, Mapping):
        raise LeRobotValidationError("meta/info.json root must be a mapping")
    return _LocalFiles(parquet_files=parquet_files, video_files=video_files, info=info)


def _validate_local_lifecycle(
    root: Path,
    sidecars: _Sidecars,
    *,
    require_completion_marker: bool,
) -> list[str]:
    reasons: list[str] = []
    marker_path = root / COMPLETION_MARKER
    if require_completion_marker:
        if not marker_path.is_file():
            reasons.append("completion marker is missing")
        else:
            try:
                marker = _read_json(marker_path)
            except LeRobotValidationError as error:
                reasons.append(str(error))
            else:
                expected = {
                    "schema_version": _COMPLETION_SCHEMA,
                    "export_fingerprint": sidecars.manifest.export_fingerprint,
                }
                if marker != expected:
                    reasons.append("completion marker schema or export fingerprint disagrees")
    elif marker_path.exists():
        reasons.append("staged validation found a completion marker before atomic promotion")
    if any(path.name == "staging.json" for path in root.rglob("staging.json")):
        reasons.append("dataset tree contains an incomplete staging ownership marker")
    if not sidecars.manifest.finalized:
        reasons.append("export manifest does not record successful writer finalization")
    return reasons


def _validate_sidecar_cross_references(sidecars: _Sidecars) -> list[str]:
    reasons: list[str] = []
    manifest = sidecars.manifest
    if sidecars.config.portable_dict() != manifest.config.portable_dict():
        reasons.append("export_config.json differs from export_manifest.json config")
    if (
        sidecars.config.source_root != "<authoritative-m3a-source>"
        or sidecars.config.output_root != "."
        or manifest.config.source_root != "<authoritative-m3a-source>"
        or manifest.config.output_root != "."
    ):
        reasons.append("derived sidecars contain machine-specific source or output paths")
    expected_mapping = {
        "schema_version": _SOURCE_MAPPING_SCHEMA,
        "export_fingerprint": manifest.export_fingerprint,
        "episodes": [item.to_dict() for item in manifest.episodes],
    }
    if sidecars.source_mapping != expected_mapping:
        reasons.append("source_episode_mapping.json is not an exact manifest cross-reference")
    split_indices = {
        split.value: [
            item.lerobot_episode_index for item in manifest.episodes if item.split is split
        ]
        for split in DatasetSplit
    }
    expected_split = {
        "schema_version": _SPLIT_MANIFEST_SCHEMA,
        "export_fingerprint": manifest.export_fingerprint,
        "split_config": manifest.config.split_config.to_dict(),
        "scene_group_assignments": [item.to_dict() for item in manifest.split_assignments],
        "episode_indices": split_indices,
    }
    if sidecars.split_manifest != expected_split:
        reasons.append("split_manifest.json is not an exact manifest cross-reference")
    return reasons


def _validate_manifest_against_source(
    manifest: LeRobotExportManifest, source: ValidatedM3ASource
) -> list[str]:
    reasons: list[str] = []
    if manifest.source_collection_run_id != source.manifest.collection_run_id:
        reasons.append("manifest source collection run ID differs from gated M3A")
    if manifest.source_run_fingerprint != source.config_fingerprint:
        reasons.append("manifest source run fingerprint differs from gated M3A")
    if manifest.source_archive_digest != source.archive_digest:
        reasons.append("manifest source archive digest differs from gated M3A")
    raw_ids = tuple(item.raw_trajectory_id for item in source.ordered_episodes)
    if manifest.ordered_source_episode_ids != raw_ids:
        reasons.append("derived source ordering differs from canonical gated M3A ordering")
    if len(manifest.episodes) != len(source.ordered_episodes):
        reasons.append("source-to-derived mapping is not one-to-one")
        return reasons
    source_group_by_episode: dict[str, str] = {}
    for group in source.ordered_groups:
        group_id = getattr(group, "accepted_scene_group_id", None)
        episodes = getattr(group, "episodes", None)
        if not isinstance(group_id, str) or not isinstance(episodes, tuple):
            reasons.append("gated M3A scene-group provenance is unavailable")
            continue
        for raw in episodes:
            raw_id = getattr(raw, "raw_trajectory_id", None)
            if not isinstance(raw_id, str) or raw_id in source_group_by_episode:
                reasons.append("gated M3A episode-to-group provenance is not one-to-one")
                continue
            source_group_by_episode[raw_id] = group_id
    for mapping, raw in zip(manifest.episodes, source.ordered_episodes, strict=True):
        checksum = sha256_hex(
            {
                "h5_group": raw.h5_group,
                "h5_sha256": raw.h5_sha256,
                "json_sha256": raw.json_sha256,
                "raw_trajectory_id": raw.raw_trajectory_id,
            }
        )
        expected = (
            mapping.source_collection_run_id == source.manifest.collection_run_id
            and mapping.source_run_fingerprint == source.config_fingerprint
            and mapping.source_archive_digest == source.archive_digest
            and mapping.source_episode_id == raw.raw_trajectory_id
            and mapping.source_scene_group_id == source_group_by_episode.get(raw.raw_trajectory_id)
            and mapping.source_scene_seed == raw.scene_seed
            and mapping.scene_id == raw.scene_id
            and mapping.task_id == raw.task_id
            and mapping.target_object_id == raw.task_spec.target_object_id
            and mapping.target_bin_id == raw.task_spec.target_bin_id
            and mapping.instruction_template_id == raw.task_spec.instruction_template_id
            and mapping.canonical_instruction == raw.canonical_instruction
            and mapping.source_shard_id == raw.source_shard_id
            and mapping.source_shard_path == raw.h5_path
            and mapping.source_trajectory_key == raw.h5_group
            and mapping.source_h5_sha256 == raw.h5_sha256
            and mapping.source_json_sha256 == raw.json_sha256
            and mapping.source_checksum == checksum
            and mapping.source_frame_count == raw.elapsed_steps
            and mapping.output_frame_count == raw.elapsed_steps
        )
        if not expected:
            reasons.append(
                f"episode {mapping.lerobot_episode_index} provenance differs from gated M3A"
            )
    return reasons


def _validate_parquet_files(paths: Sequence[Path]) -> list[str]:
    reasons: list[str] = []
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        return [f"PyArrow Parquet reader is unavailable: {error}"]
    for path in paths:
        try:
            parquet.ParquetFile(path).read()
        except Exception as error:
            reasons.append(f"cannot read Parquet file {path}: {type(error).__name__}: {error}")
    return reasons


def _validate_local_owned_references(root: Path, local_files: _LocalFiles) -> list[str]:
    """Resolve every episode-table data/video reference before public loading."""
    reasons: list[str] = []
    data_template = local_files.info.get("data_path")
    video_template = local_files.info.get("video_path")
    if not isinstance(data_template, str) or not data_template:
        reasons.append("meta/info.json has no data_path template")
    if not isinstance(video_template, str) or not video_template:
        reasons.append("meta/info.json has no video_path template")
    if reasons:
        return reasons
    episode_files = tuple(
        path
        for path in local_files.parquet_files
        if path.relative_to(root).parts[:2] == ("meta", "episodes")
    )
    try:
        import pyarrow.parquet as parquet

        rows = [row for path in episode_files for row in parquet.read_table(path).to_pylist()]
    except Exception as error:
        return [f"cannot load local episode reference table: {type(error).__name__}: {error}"]
    if not rows:
        return ["local episode reference table is empty"]
    for row_index, row in enumerate(rows):
        try:
            data_relative = data_template.format(
                chunk_index=int(row["data/chunk_index"]),
                file_index=int(row["data/file_index"]),
            )
            video_relative = video_template.format(
                video_key=IMAGE_FEATURE_KEY,
                chunk_index=int(row[f"videos/{IMAGE_FEATURE_KEY}/chunk_index"]),
                file_index=int(row[f"videos/{IMAGE_FEATURE_KEY}/file_index"]),
            )
            data_path = _confined_path(root, data_relative)
            video_path = _confined_path(root, video_relative)
        except Exception as error:
            reasons.append(f"episode metadata row {row_index} has invalid file references: {error}")
            continue
        if not data_path.is_file():
            reasons.append(f"episode metadata row {row_index} references missing {data_relative}")
        if not video_path.is_file():
            reasons.append(f"episode metadata row {row_index} references missing {video_relative}")
    return reasons


def _load_local_dataset(repo_id: str, root: Path) -> Any:
    # The public loader documents local root operation.  Offline environment
    # flags are set before the import/constructor as a second guard after the
    # complete local-file and Parquet preflight above.
    with _offline_hub_environment():
        from lerobot.datasets import LeRobotDataset

        return LeRobotDataset(
            repo_id=repo_id,
            root=root,
            video_backend="pyav",
            return_uint8=True,
            download_videos=False,
        )


@contextmanager
def _offline_hub_environment() -> Iterator[None]:
    names = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update({name: "1" for name in names})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _validate_feature_schema(
    value: object, config: LeRobotExportConfig
) -> tuple[list[str], list[str]]:
    schema_reasons: list[str] = []
    leakage_reasons: list[str] = []
    if not isinstance(value, Mapping):
        return ["LeRobotDataset.features is not a mapping"], ["feature allowlist is unavailable"]
    actual_keys = set(value)
    allowed = POLICY_FEATURE_KEYS | LEROBOT_MANAGED_FEATURE_KEYS
    unknown = sorted(actual_keys - allowed)
    if unknown:
        leakage_reasons.append("privileged or unknown features are present: " + ", ".join(unknown))
    if actual_keys != allowed:
        schema_reasons.append(
            "feature keys differ from the exact policy plus LeRobot-managed allowlist"
        )
    expected = config.feature_contract.to_lerobot_features() | dict(_EXPECTED_MANAGED_FEATURES)
    for key, contract in expected.items():
        feature = value.get(key)
        if not isinstance(feature, Mapping):
            schema_reasons.append(f"feature {key!r} has no metadata mapping")
            continue
        if feature.get("dtype") != contract["dtype"]:
            schema_reasons.append(f"feature {key!r} dtype differs from the contract")
        try:
            shape = tuple(feature.get("shape", ()))
        except TypeError:
            shape = ()
        if shape != tuple(contract["shape"]):
            schema_reasons.append(f"feature {key!r} shape differs from the contract")
        expected_names = contract["names"]
        actual_names = feature.get("names")
        if expected_names is None:
            if actual_names is not None:
                schema_reasons.append(f"feature {key!r} names differ from the contract")
        elif tuple(actual_names or ()) != tuple(expected_names):
            schema_reasons.append(f"feature {key!r} names differ from the contract")
    image_feature = value.get(IMAGE_FEATURE_KEY)
    video_info = image_feature.get("info") if isinstance(image_feature, Mapping) else None
    if not isinstance(video_info, Mapping):
        schema_reasons.append("base_camera feature is missing finalized video stream metadata")
    else:
        expected_video_info = {
            "video.height": config.image_height,
            "video.width": config.image_width,
            "video.codec": config.video_codec.codec,
            "video.pix_fmt": config.video_codec.pixel_format,
            "video.fps": config.fps,
            "video.channels": 3,
            "has_audio": False,
            "video.g": config.video_codec.gop_size,
            "video.crf": config.video_codec.crf,
            "video.preset": config.video_codec.preset,
            "video.fast_decode": config.video_codec.fast_decode,
            "video.video_backend": config.video_codec.backend,
            "is_depth_map": False,
        }
        for name, expected_value in expected_video_info.items():
            if video_info.get(name) != expected_value:
                schema_reasons.append(f"base_camera finalized metadata {name!r} differs")
    return schema_reasons, leakage_reasons


def _audit_dataset_rows(
    dataset: Any,
    manifest: LeRobotExportManifest,
    source: ValidatedM3ASource,
) -> tuple[tuple[_DerivedEpisode, ...], list[str]]:
    reasons: list[str] = []
    meta = getattr(dataset, "meta", None)
    info = getattr(meta, "info", None)
    if getattr(info, "codebase_version", None) != "v3.0":
        reasons.append("loaded dataset is not LeRobotDataset v3.0")
    if getattr(dataset, "fps", None) != manifest.config.fps:
        reasons.append("loaded dataset FPS differs from the export contract")
    if getattr(dataset, "num_episodes", None) != len(manifest.episodes):
        reasons.append("loaded dataset episode count differs from the manifest")
    expected_frames = sum(item.source_frame_count for item in manifest.episodes)
    if getattr(dataset, "num_frames", None) != expected_frames:
        reasons.append("loaded dataset total frame count differs from source T values")
    metadata = getattr(meta, "episodes", None)
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if metadata is None or hf_dataset is None:
        reasons.append("loaded dataset lacks public episode metadata or Parquet-backed rows")
        return (), reasons
    try:
        metadata_count = len(metadata)
        frame_count = len(hf_dataset)
    except TypeError:
        reasons.append("loaded episode metadata or Parquet rows have no finite length")
        return (), reasons
    if metadata_count != len(manifest.episodes):
        reasons.append("metadata episode table count differs from the manifest")
        return (), reasons
    if frame_count != expected_frames:
        reasons.append("Parquet row count differs from the sum of source T values")
        return (), reasons

    derived: list[_DerivedEpisode] = []
    expected_start = 0
    for mapping, raw in zip(manifest.episodes, source.ordered_episodes, strict=True):
        episode_index = mapping.lerobot_episode_index
        try:
            metadata_row = metadata[episode_index]
            if not isinstance(metadata_row, Mapping):
                raise TypeError("episode metadata row is not a mapping")
            length = _scalar_int(metadata_row.get("length"), "length")
            start = _scalar_int(metadata_row.get("dataset_from_index"), "dataset_from_index")
            stop = _scalar_int(metadata_row.get("dataset_to_index"), "dataset_to_index")
            recorded_index = _scalar_int(metadata_row.get("episode_index"), "episode_index")
        except Exception as error:
            reasons.append(
                f"episode {episode_index} metadata boundary is unreadable: "
                f"{type(error).__name__}: {error}"
            )
            return (), reasons
        if (
            recorded_index != episode_index
            or start != expected_start
            or stop != start + length
            or length != mapping.source_frame_count
            or length != raw.elapsed_steps
        ):
            reasons.append(f"episode {episode_index} metadata boundary or length is invalid")
        expected_start = stop
        tasks = metadata_row.get("tasks")
        if (
            not isinstance(tasks, Sequence)
            or isinstance(tasks, str)
            or list(tasks) != [mapping.canonical_instruction]
        ):
            reasons.append(f"episode {episode_index} metadata does not contain exactly its task")
        try:
            video_path = _video_path_for_episode(dataset, episode_index)
            video_from = _scalar_float(
                metadata_row.get(f"videos/{IMAGE_FEATURE_KEY}/from_timestamp"),
                "video from_timestamp",
            )
            video_to = _scalar_float(
                metadata_row.get(f"videos/{IMAGE_FEATURE_KEY}/to_timestamp"),
                "video to_timestamp",
            )
        except Exception as error:
            reasons.append(f"episode {episode_index} video metadata is invalid: {error}")
            return (), reasons
        video_start = round(video_from * manifest.config.fps)
        video_stop = round(video_to * manifest.config.fps)
        video_timestamps_ok = (
            abs(video_from - video_start / manifest.config.fps)
            <= manifest.config.timestamp_tolerance
            and abs(video_to - (video_start + length) / manifest.config.fps)
            <= manifest.config.timestamp_tolerance
            and video_stop == video_start + length
        )
        if not video_timestamps_ok:
            reasons.append(f"episode {episode_index} video time boundaries do not span T frames")

        actions = np.empty((length, 8), dtype=np.float32)
        states = np.empty((length, 9), dtype=np.float32)
        timestamps_ok = video_timestamps_ok
        for frame_index in range(length):
            try:
                row = hf_dataset[start + frame_index]
                if not isinstance(row, Mapping):
                    raise TypeError("Parquet frame row is not a mapping")
                action = _strict_vector(row.get(ACTION_FEATURE_KEY), (8,), "action")
                state = _strict_vector(row.get(STATE_FEATURE_KEY), (9,), "observation.state")
                timestamp = _scalar_float(row.get("timestamp"), "timestamp")
                row_episode = _scalar_int(row.get("episode_index"), "episode_index")
                row_frame = _scalar_int(row.get("frame_index"), "frame_index")
                row_index = _scalar_int(row.get("index"), "index")
                task_index = _scalar_int(row.get("task_index"), "task_index")
                task_text = _task_text(dataset, task_index)
            except Exception as error:
                reasons.append(
                    f"episode {episode_index} frame {frame_index} is invalid: "
                    f"{type(error).__name__}: {error}"
                )
                return (), reasons
            actions[frame_index] = action
            states[frame_index] = state
            expected_timestamp = frame_index / manifest.config.fps
            frame_timestamp_ok = abs(timestamp - expected_timestamp) <= (
                manifest.config.timestamp_tolerance
            )
            timestamps_ok = timestamps_ok and frame_timestamp_ok
            if (
                row_episode != episode_index
                or row_frame != frame_index
                or row_index != start + frame_index
                or task_text != mapping.canonical_instruction
            ):
                reasons.append(
                    f"episode {episode_index} frame {frame_index} managed metadata is misaligned"
                )
            if not frame_timestamp_ok:
                reasons.append(
                    f"episode {episode_index} frame {frame_index} timestamp differs from t/fps"
                )
        derived.append(
            _DerivedEpisode(
                boundary=_EpisodeBoundary(
                    episode_index=episode_index,
                    start=start,
                    stop=stop,
                    length=length,
                    video_path=video_path,
                    video_start_frame=video_start,
                    video_stop_frame=video_start + length,
                    timestamps_match=timestamps_ok,
                ),
                actions=actions,
                states=states,
            )
        )
    canonical_tasks = {item.canonical_instruction for item in source.ordered_episodes}
    loaded_tasks = _all_task_texts(dataset)
    if loaded_tasks != canonical_tasks:
        reasons.append("LeRobot task metadata is not exactly the six canonical source instructions")
    return tuple(derived), reasons


def _validate_split_integrity(
    manifest: LeRobotExportManifest, source: ValidatedM3ASource
) -> list[str]:
    reasons: list[str] = []
    try:
        source_group_ids = tuple(group.accepted_scene_group_id for group in source.ordered_groups)
        expected_assignments = build_split_assignments(
            source_group_ids,
            manifest.config.split_config,
        )
    except (AttributeError, TypeError, ValueError) as error:
        reasons.append(f"canonical split assignments could not be reconstructed: {error}")
    else:
        if manifest.split_assignments != expected_assignments:
            reasons.append(
                "stored split digest/rank/assignment mapping differs from canonical recomputation"
            )
    expected_groups = 1 if manifest.config.mode is ExportMode.SMOKE else 60
    expected_group_counts = (
        {"train": 1, "validation": 0, "test": 0}
        if manifest.config.mode is ExportMode.SMOKE
        else {"train": 48, "validation": 6, "test": 6}
    )
    group_counts = Counter(item.split.value for item in manifest.split_assignments)
    if (
        len(manifest.split_assignments) != expected_groups
        or {split.value: group_counts[split.value] for split in DatasetSplit}
        != expected_group_counts
    ):
        reasons.append("scene-group split counts differ from the fixed mode contract")
    split_by_group = {item.scene_group_id: item.split for item in manifest.split_assignments}
    if len(split_by_group) != len(manifest.split_assignments):
        reasons.append("one scene group appears in multiple split assignments")
    groups_by_scene: dict[str, set[DatasetSplit]] = defaultdict(set)
    tasks_by_split: dict[DatasetSplit, Counter[str]] = {split: Counter() for split in DatasetSplit}
    for mapping in manifest.episodes:
        split = split_by_group.get(mapping.source_scene_group_id)
        if split is None or split is not mapping.split:
            reasons.append(f"episode {mapping.lerobot_episode_index} split mapping disagrees")
            continue
        groups_by_scene[mapping.scene_id].add(split)
        tasks_by_split[split][mapping.task_id] += 1
    if any(len(splits) != 1 for splits in groups_by_scene.values()):
        reasons.append("a physical scene leaks across dataset splits")
    task_ids = {item.task_id for item in source.ordered_episodes}
    expected_per_task = (
        {DatasetSplit.TRAIN: 1, DatasetSplit.VALIDATION: 0, DatasetSplit.TEST: 0}
        if manifest.config.mode is ExportMode.SMOKE
        else {DatasetSplit.TRAIN: 48, DatasetSplit.VALIDATION: 6, DatasetSplit.TEST: 6}
    )
    for split in DatasetSplit:
        expected = {task_id: expected_per_task[split] for task_id in task_ids}
        actual = {task_id: tasks_by_split[split][task_id] for task_id in task_ids}
        if actual != expected:
            reasons.append(f"{split.value} task distribution is not perfectly balanced")
    return reasons


def _decode_all_videos(
    root: Path,
    boundaries: tuple[_EpisodeBoundary, ...],
    *,
    actual_video_paths: tuple[str, ...],
    image_height: int,
    image_width: int,
) -> tuple[dict[int, _DecodedEpisode], list[str]]:
    decoded = {
        boundary.episode_index: _DecodedEpisode(
            episode_index=boundary.episode_index,
            path=boundary.video_path,
            expected_count=boundary.length,
            sample_indices=deterministic_video_sample_indices(boundary.length),
        )
        for boundary in boundaries
    }
    by_path: dict[str, list[_EpisodeBoundary]] = defaultdict(list)
    for boundary in boundaries:
        by_path[boundary.video_path].append(boundary)
    reasons: list[str] = []
    actual_path_set = set(actual_video_paths)
    referenced_path_set = set(by_path)
    if actual_path_set != referenced_path_set:
        reasons.append("actual MP4 files are not the exact set referenced by episode metadata")

    # Decode the union, not only referenced paths: an unexpected MP4 is still
    # an actual dataset file and must be structurally decoded before rejection.
    for relative_path in sorted(actual_path_set | referenced_path_set):
        segments = by_path.get(relative_path, [])
        if not segments:
            reasons.extend(
                _decode_unreferenced_video(
                    root,
                    relative_path,
                    image_height=image_height,
                    image_width=image_width,
                )
            )
            continue
        segments.sort(key=lambda item: (item.video_start_frame, item.episode_index))
        expected_cursor = 0
        for segment in segments:
            if segment.video_start_frame != expected_cursor:
                reason = "shared video episode ranges contain a gap or overlap"
                reasons.append(f"{relative_path}: {reason}")
                decoded[segment.episode_index].failure_reasons.append(reason)
            expected_cursor = segment.video_stop_frame
        path = _confined_path(root, relative_path)
        if not path.is_file():
            reason = "referenced video file is missing"
            reasons.append(f"{relative_path}: {reason}")
            for segment in segments:
                decoded[segment.episode_index].failure_reasons.append(reason)
            continue
        frame_number = 0
        segment_index = 0
        try:
            with _open_pyav_container(path) as container:
                streams = tuple(getattr(getattr(container, "streams", None), "video", ()))
                if len(streams) != 1:
                    raise LeRobotValidationError(
                        "policy video must contain exactly one video stream"
                    )
                for frame in container.decode(streams[0]):
                    while (
                        segment_index < len(segments)
                        and frame_number >= segments[segment_index].video_stop_frame
                    ):
                        segment_index += 1
                    if segment_index >= len(segments):
                        reasons.append(f"{relative_path}: decoded unreferenced trailing frame")
                        frame_number += 1
                        continue
                    segment = segments[segment_index]
                    if frame_number < segment.video_start_frame:
                        reasons.append(f"{relative_path}: decoded frame lies in an unowned gap")
                        frame_number += 1
                        continue
                    array = np.asarray(frame.to_ndarray(format="rgb24"))
                    evidence = decoded[segment.episode_index]
                    local_index = frame_number - segment.video_start_frame
                    evidence.decoded_count += 1
                    if array.dtype != np.dtype(np.uint8) or array.shape != (
                        image_height,
                        image_width,
                        3,
                    ):
                        evidence.failure_reasons.append(
                            f"decoded frame {local_index} violates uint8 HWC RGB shape"
                        )
                    else:
                        nonempty = bool(np.any(array))
                        nonuniform = bool(np.min(array) != np.max(array))
                        evidence.frames_nonempty = evidence.frames_nonempty and nonempty
                        evidence.frames_nonuniform = evidence.frames_nonuniform and nonuniform
                        if not nonempty:
                            evidence.failure_reasons.append(f"decoded frame {local_index} is empty")
                        if not nonuniform:
                            evidence.failure_reasons.append(
                                f"decoded frame {local_index} is uniform"
                            )
                        if local_index in evidence.sample_indices:
                            evidence.samples[local_index] = np.array(array, copy=True, order="C")
                    frame_number += 1
        except Exception as error:
            reason = f"PyAV sequential decode failed: {type(error).__name__}: {error}"
            reasons.append(f"{relative_path}: {reason}")
            for segment in segments:
                decoded[segment.episode_index].failure_reasons.append(reason)
        if frame_number != expected_cursor:
            reasons.append(
                f"{relative_path}: decoded {frame_number} frames but metadata covers {expected_cursor}"
            )
        for segment in segments:
            evidence = decoded[segment.episode_index]
            if evidence.decoded_count != evidence.expected_count:
                evidence.failure_reasons.append("decoded frame count differs from episode length")
            if set(evidence.samples) != set(evidence.sample_indices):
                evidence.failure_reasons.append(
                    "one or more deterministic sample frames are missing"
                )
    return decoded, reasons


def _decode_unreferenced_video(
    root: Path,
    relative_path: str,
    *,
    image_height: int,
    image_width: int,
) -> list[str]:
    reasons = [f"{relative_path}: video file is not referenced by any derived episode"]
    path = _confined_path(root, relative_path)
    decoded_count = 0
    try:
        with _open_pyav_container(path) as container:
            streams = tuple(getattr(getattr(container, "streams", None), "video", ()))
            if len(streams) != 1:
                raise LeRobotValidationError("policy video must contain exactly one video stream")
            for frame in container.decode(streams[0]):
                array = np.asarray(frame.to_ndarray(format="rgb24"))
                if array.dtype != np.dtype(np.uint8) or array.shape != (
                    image_height,
                    image_width,
                    3,
                ):
                    reasons.append(
                        f"{relative_path}: unreferenced frame {decoded_count} violates RGB schema"
                    )
                elif not np.any(array) or np.min(array) == np.max(array):
                    reasons.append(
                        f"{relative_path}: unreferenced frame {decoded_count} is empty or uniform"
                    )
                decoded_count += 1
    except Exception as error:
        reasons.append(
            f"{relative_path}: PyAV sequential decode failed: {type(error).__name__}: {error}"
        )
    if decoded_count == 0:
        reasons.append(f"{relative_path}: unreferenced video decoded zero frames")
    return reasons


def _open_pyav_container(path: Path) -> Any:
    import av

    return av.open(str(path), mode="r")


def _validate_source_alignment(
    source: ValidatedM3ASource,
    manifest: LeRobotExportManifest,
    derived: tuple[_DerivedEpisode, ...],
    decoded: Mapping[int, _DecodedEpisode],
) -> tuple[tuple[SourceAlignmentResult, ...], tuple[VideoValidationResult, ...]]:
    reader = ValidatedRawEpisodeReader(source)
    reconstructor = _make_reconstructor(source, manifest.config)
    alignments: list[SourceAlignmentResult] = []
    videos: list[VideoValidationResult] = []
    try:
        reconstructor.open()
    except Exception as error:
        reason = f"policy reconstructor could not open: {type(error).__name__}: {error}"
        for mapping, item in zip(manifest.episodes, derived, strict=True):
            video = _failed_video_result(decoded[mapping.lerobot_episode_index], reason)
            videos.append(video)
            alignments.append(_failed_alignment(mapping, item, reason))
        return tuple(alignments), tuple(videos)

    try:
        for mapping, raw_record, item in zip(
            manifest.episodes, source.ordered_episodes, derived, strict=True
        ):
            result, video = _align_one_episode(
                reader,
                reconstructor,
                mapping,
                raw_record,
                item,
                decoded[mapping.lerobot_episode_index],
                manifest.config,
            )
            alignments.append(result)
            videos.append(video)
    finally:
        reconstructor.close()
    try:
        reader.revalidate_source()
    except Exception as error:
        reason = f"M3A source changed during validation: {type(error).__name__}: {error}"
        if alignments:
            last = alignments[-1]
            flags = last.to_dict()
            flags["raw_render_digest_matches"] = False
            flags["passed"] = False
            flags["failure_reasons"] = [*last.failure_reasons, reason]
            alignments[-1] = SourceAlignmentResult.from_dict(flags)
    return tuple(alignments), tuple(videos)


def _make_reconstructor(
    source: ValidatedM3ASource, config: LeRobotExportConfig
) -> ManiSkillObservationReconstructor:
    return ManiSkillObservationReconstructor(
        sim_backend=source.manifest.config.sim_backend,
        render_backend=config.camera_contract.render_backend,
    )


def _align_one_episode(
    reader: ValidatedRawEpisodeReader,
    reconstructor: ManiSkillObservationReconstructor,
    mapping: EpisodeExportRecord,
    raw_record: Any,
    derived: _DerivedEpisode,
    decoded: _DecodedEpisode,
    config: LeRobotExportConfig,
) -> tuple[SourceAlignmentResult, VideoValidationResult]:
    reasons: list[str] = []
    task_matches = (
        mapping.task_id == raw_record.task_id
        and mapping.canonical_instruction == raw_record.canonical_instruction
        and mapping.target_object_id == raw_record.task_spec.target_object_id
        and mapping.target_bin_id == raw_record.task_spec.target_bin_id
        and mapping.instruction_template_id == raw_record.task_spec.instruction_template_id
    )
    scene_matches = (
        mapping.scene_id == raw_record.scene_id
        and mapping.source_scene_seed == raw_record.scene_seed
    )
    frame_matches = derived.boundary.length == raw_record.elapsed_steps
    actions_match = False
    states_match = False
    digest_matches = False
    references: dict[int, np.ndarray] = {}
    try:
        raw = reader.read_episode(mapping.source_episode_id)
        frame_matches = frame_matches and raw.transition_count == derived.boundary.length
        actions_match = (
            raw.actions.dtype == np.dtype(np.float32)
            and raw.actions.shape == derived.actions.shape
            and np.array_equal(raw.actions, derived.actions, equal_nan=False)
        )
        reconstructor.begin_episode(raw_record)
        digest = RawRenderDigest()
        states_match = True
        for frame_index in range(raw.transition_count):
            policy = reconstructor.reconstruct(raw.training_state_at(frame_index))
            digest.add(frame_index, policy.rgb)
            states_match = states_match and np.allclose(
                policy.state,
                derived.states[frame_index],
                rtol=0.0,
                atol=config.state_tolerance,
                equal_nan=False,
            )
            if frame_index in decoded.sample_indices:
                references[frame_index] = np.array(policy.rgb, copy=True, order="C")
        digest_matches = digest.hexdigest() == mapping.raw_render_digest
    except Exception as error:
        reasons.append(f"source reconstruction failed: {type(error).__name__}: {error}")

    checks = {
        "task identity differs from M3A": task_matches,
        "scene identity differs from M3A": scene_matches,
        "derived frame count differs from source T": frame_matches,
        "derived actions differ from exact M3A actions": actions_match,
        "derived policy states differ from reconstructed qpos": states_match,
        "raw render digest differs from reconstructed RGB sequence": digest_matches,
        "timestamps differ from t/fps": derived.boundary.timestamps_match,
    }
    reasons.extend(message for message, passed in checks.items() if not passed)
    video = _video_result(decoded, references, config)
    if not video.passed:
        reasons.append("decoded video structural or sampled quality audit failed")
    alignment_flags = (
        task_matches,
        scene_matches,
        frame_matches,
        actions_match,
        states_match,
        digest_matches,
        video.decoded_frame_count == video.expected_frame_count,
        video.passed,
        derived.boundary.timestamps_match,
    )
    passed = all(alignment_flags)
    return (
        SourceAlignmentResult(
            lerobot_episode_index=mapping.lerobot_episode_index,
            source_episode_id=mapping.source_episode_id,
            task_identity_matches=task_matches,
            scene_identity_matches=scene_matches,
            frame_count_matches=frame_matches,
            actions_match=actions_match,
            policy_states_match=states_match,
            raw_render_digest_matches=digest_matches,
            decoded_video_frame_count_matches=(
                video.decoded_frame_count == video.expected_frame_count
            ),
            decoded_video_quality_passed=video.passed,
            timestamps_match=derived.boundary.timestamps_match,
            passed=passed,
            failure_reasons=() if passed else _deduplicate(reasons),
        ),
        video,
    )


def _video_result(
    decoded: _DecodedEpisode,
    references: Mapping[int, np.ndarray],
    config: LeRobotExportConfig,
) -> VideoValidationResult:
    reasons = list(decoded.failure_reasons)
    maes: list[float] = []
    psnrs: list[float] = []
    if set(references) != set(decoded.sample_indices):
        reasons.append("reconstructed deterministic sample frames are incomplete")
    for frame_index in decoded.sample_indices:
        reference = references.get(frame_index)
        frame = decoded.samples.get(frame_index)
        if reference is None or frame is None:
            continue
        try:
            mae, psnr = rgb_quality_metrics(reference, frame)
        except Exception as error:
            reasons.append(f"sample {frame_index} quality metric failed: {error}")
            continue
        maes.append(mae)
        psnrs.append(999.0 if math.isinf(psnr) else psnr)
    mean_mae = float(np.mean(maes)) if maes else 0.0
    minimum_psnr = min(psnrs) if psnrs else 0.0
    if maes and mean_mae > config.video_mean_absolute_error_tolerance:
        reasons.append("sampled decoded-frame MAE exceeds the configured threshold")
    if psnrs and minimum_psnr < config.video_min_psnr_db:
        reasons.append("sampled decoded-frame PSNR is below the configured threshold")
    if len(maes) != len(decoded.sample_indices):
        reasons.append("not all deterministic sample frames produced quality metrics")
    passed = (
        not reasons
        and decoded.decoded_count == decoded.expected_count
        and decoded.frames_nonempty
        and decoded.frames_nonuniform
    )
    return VideoValidationResult(
        lerobot_episode_index=decoded.episode_index,
        video_path=decoded.path,
        expected_frame_count=decoded.expected_count,
        decoded_frame_count=decoded.decoded_count,
        decoded_height=config.image_height,
        decoded_width=config.image_width,
        sampled_frame_indices=decoded.sample_indices,
        mean_absolute_error=mean_mae,
        minimum_psnr_db=minimum_psnr,
        frames_nonempty=decoded.frames_nonempty,
        frames_nonuniform=decoded.frames_nonuniform,
        passed=passed,
        failure_reasons=() if passed else _deduplicate(reasons),
    )


def _failed_video_result(decoded: _DecodedEpisode, reason: str) -> VideoValidationResult:
    decoded.failure_reasons.append(reason)
    return VideoValidationResult(
        lerobot_episode_index=decoded.episode_index,
        video_path=decoded.path,
        expected_frame_count=decoded.expected_count,
        decoded_frame_count=decoded.decoded_count,
        decoded_height=256,
        decoded_width=256,
        sampled_frame_indices=decoded.sample_indices,
        mean_absolute_error=0.0,
        minimum_psnr_db=0.0,
        frames_nonempty=decoded.frames_nonempty,
        frames_nonuniform=decoded.frames_nonuniform,
        passed=False,
        failure_reasons=_deduplicate(decoded.failure_reasons),
    )


def _failed_alignment(
    mapping: EpisodeExportRecord, derived: _DerivedEpisode, reason: str
) -> SourceAlignmentResult:
    return SourceAlignmentResult(
        lerobot_episode_index=mapping.lerobot_episode_index,
        source_episode_id=mapping.source_episode_id,
        task_identity_matches=True,
        scene_identity_matches=True,
        frame_count_matches=derived.boundary.length == mapping.source_frame_count,
        actions_match=False,
        policy_states_match=False,
        raw_render_digest_matches=False,
        decoded_video_frame_count_matches=False,
        decoded_video_quality_passed=False,
        timestamps_match=derived.boundary.timestamps_match,
        passed=False,
        failure_reasons=(reason,),
    )


def _validate_dataloader(
    dataset: Any,
    derived: tuple[_DerivedEpisode, ...],
    manifest: LeRobotExportManifest,
) -> list[str]:
    reasons: list[str] = []
    frame_indices = {derived[0].boundary.start, derived[-1].boundary.stop - 1}
    for split in DatasetSplit:
        candidate = next((item for item in manifest.episodes if item.split is split), None)
        if candidate is not None:
            frame_indices.add(derived[candidate.lerobot_episode_index].boundary.start)
    for index in sorted(frame_indices):
        try:
            _validate_policy_item(dataset[index], batched=False)
        except Exception as error:
            reasons.append(
                f"direct dataset indexing failed at frame {index}: {type(error).__name__}: {error}"
            )
    try:
        from torch.utils.data import DataLoader

        loader = DataLoader(dataset, batch_size=min(2, len(dataset)), shuffle=False, num_workers=0)
        _validate_policy_item(next(iter(loader)), batched=True)
        if getattr(loader, "num_workers", None) != 0:
            reasons.append("DataLoader smoke did not retain num_workers=0")
    except Exception as error:
        reasons.append(f"DataLoader num_workers=0 smoke failed: {type(error).__name__}: {error}")
    return reasons


def _validate_policy_item(item: object, *, batched: bool) -> None:
    if not isinstance(item, Mapping):
        raise TypeError("policy item must be a mapping")
    image = _to_numpy(item.get(IMAGE_FEATURE_KEY))
    state = _to_numpy(item.get(STATE_FEATURE_KEY))
    action = _to_numpy(item.get(ACTION_FEATURE_KEY))
    if batched:
        if image.ndim != 4 or image.shape[1:] != (3, 256, 256):
            raise ValueError(f"batched base_camera must be BCHW, got {image.shape}")
        if state.ndim != 2 or state.shape[1:] != (9,):
            raise ValueError(f"batched state must have shape (B, 9), got {state.shape}")
        if action.ndim != 2 or action.shape[1:] != (8,):
            raise ValueError(f"batched action must have shape (B, 8), got {action.shape}")
    else:
        if image.shape != (3, 256, 256):
            raise ValueError(f"base_camera loader output must be CHW, got {image.shape}")
        if state.shape != (9,) or action.shape != (8,):
            raise ValueError("direct state/action shapes differ from (9,) and (8,)")
    if image.dtype != np.dtype(np.uint8):
        raise ValueError("return_uint8=True did not yield uint8 base_camera tensors")
    if state.dtype != np.dtype(np.float32) or action.dtype != np.dtype(np.float32):
        raise ValueError("state/action loader tensors must retain float32")
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
        raise ValueError("state/action loader tensors must be finite")


def _build_summary(
    manifest: LeRobotExportManifest,
    source: ValidatedM3ASource,
    *,
    parquet_count: int,
    video_count: int,
) -> LeRobotDatasetSummary:
    task_counts = Counter(item.task_id for item in source.ordered_episodes)
    group_counts = Counter(item.split.value for item in manifest.split_assignments)
    episode_counts = Counter(item.split.value for item in manifest.episodes)
    return LeRobotDatasetSummary(
        export_fingerprint=manifest.export_fingerprint,
        repo_id=manifest.repo_id,
        mode=manifest.config.mode,
        total_scene_groups=len(source.ordered_groups),
        total_episodes=len(source.ordered_episodes),
        total_frames=sum(item.elapsed_steps for item in source.ordered_episodes),
        task_episode_counts=dict(task_counts),
        split_scene_group_counts={split.value: group_counts[split.value] for split in DatasetSplit},
        split_episode_counts={split.value: episode_counts[split.value] for split in DatasetSplit},
        feature_keys=tuple(sorted(POLICY_FEATURE_KEYS | LEROBOT_MANAGED_FEATURE_KEYS)),
        fps=manifest.config.fps,
        image_shape=manifest.config.feature_contract.image_shape,
        state_shape=manifest.config.feature_contract.state_shape,
        action_shape=manifest.config.feature_contract.action_shape,
        parquet_file_count=parquet_count,
        video_file_count=video_count,
    )


def _video_path_for_episode(dataset: Any, episode_index: int) -> str:
    path = dataset.meta.get_video_file_path(episode_index, IMAGE_FEATURE_KEY)
    if not isinstance(path, (str, Path)):
        raise TypeError("LeRobot video path must be string-like")
    pure = Path(path)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError("LeRobot video metadata path must be confined and relative")
    return pure.as_posix()


def _task_text(dataset: Any, task_index: int) -> str:
    tasks = dataset.meta.tasks
    try:
        task_row = tasks.iloc[task_index]
        text = task_row.name
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise ValueError(f"cannot resolve task_index {task_index}") from error
    get_value = getattr(task_row, "get", None)
    if callable(get_value):
        recorded_index = get_value("task_index")
        if (
            recorded_index is not None
            and _scalar_int(recorded_index, "task table index") != task_index
        ):
            raise ValueError("task table row ordering and task_index column disagree")
    if not isinstance(text, str) or not text:
        raise ValueError("task metadata must resolve to a non-empty string")
    return text


def _all_task_texts(dataset: Any) -> set[str]:
    index = getattr(dataset.meta.tasks, "index", None)
    if index is None:
        raise LeRobotValidationError("LeRobot task table has no public string index")
    result = set(index)
    if not all(isinstance(item, str) and item for item in result):
        raise LeRobotValidationError("LeRobot task table contains invalid canonical text")
    return result


def _strict_vector(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = _to_numpy(value)
    if array.dtype != np.dtype(np.float32):
        raise ValueError(f"{label} dtype must be float32, got {array.dtype}")
    if array.shape != shape:
        raise ValueError(f"{label} shape must be {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    return np.array(array, dtype=np.float32, copy=True)


def _to_numpy(value: object) -> np.ndarray:
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    numpy = getattr(candidate, "numpy", None)
    if callable(numpy):
        candidate = numpy()
    try:
        return np.asarray(candidate)
    except (TypeError, ValueError) as error:
        raise TypeError("value is not numeric array data") from error


def _scalar_int(value: object, label: str) -> int:
    array = _to_numpy(value)
    if array.size != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"{label} must be one integer scalar")
    return int(array.reshape(-1)[0])


def _scalar_float(value: object, label: str) -> float:
    array = _to_numpy(value)
    if array.size != 1 or array.dtype.kind not in "fiu":
        raise ValueError(f"{label} must be one numeric scalar")
    result = float(array.reshape(-1)[0])
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LeRobotValidationError(f"cannot read JSON file {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise LeRobotValidationError(f"JSON root must be a mapping: {path}")
    return payload


def _relative_posix(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise LeRobotValidationError(f"dataset-owned path escapes local root: {path}") from error


def _confined_path(root: Path, relative: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise LeRobotValidationError("derived dataset path must be relative and confined")
    path = (root / raw).resolve()
    _relative_posix(path, root)
    return path


def _deduplicate(reasons: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(reason for reason in reasons if reason))


__all__ = ["LeRobotValidationError", "validate_lerobot_dataset"]
