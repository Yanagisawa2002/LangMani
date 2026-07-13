"""Deterministic staged export from a validated M3A archive to LeRobot v3."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from langmani.datasets.frame_alignment import RawRenderDigest
from langmani.datasets.identity import sha256_hex
from langmani.datasets.lerobot_source import (
    ValidatedM3ASource,
    ValidatedRawEpisodeReader,
    validate_m3a_source_for_export,
)
from langmani.datasets.lerobot_types import (
    DatasetSplit,
    EpisodeExportRecord,
    ExportMode,
    LeRobotDatasetSummary,
    LeRobotExportConfig,
    LeRobotExportManifest,
    LeRobotValidationReport,
    SplitAssignment,
    stable_export_fingerprint,
)
from langmani.datasets.lerobot_writer import (
    SIDECAR_DIRECTORY,
    LeRobotRuntimeInfo,
    LeRobotWriterAdapter,
    LeRobotWriterError,
    StagingLayout,
    WriterState,
    inspect_lerobot_runtime,
    prepare_staging_layout,
    promote_staged_dataset,
    write_completion_marker,
)
from langmani.datasets.observation_reconstruction import ManiSkillObservationReconstructor
from langmani.datasets.splits import (
    assignment_map,
    build_split_assignments,
    episode_indices_by_split,
    task_counts_by_split,
)


class LeRobotExportError(RuntimeError):
    """Raised when source validation, export, finalization, or promotion fails."""


@dataclass(frozen=True, slots=True)
class LeRobotExportPlan:
    """Dry-run-safe semantic plan; constructing it never creates dataset files."""

    source: ValidatedM3ASource
    export_fingerprint: str
    split_assignments: tuple[SplitAssignment, ...]
    expected_frame_count: int
    runtime_info: LeRobotRuntimeInfo

    def to_dict(self, config: LeRobotExportConfig) -> dict[str, object]:
        split_counts = Counter(item.split.value for item in self.split_assignments)
        ordered = [
            {
                "source_episode_id": episode.raw_trajectory_id,
                "source_scene_group_id": _episode_group_id(self.source, episode.raw_trajectory_id),
                "stable_task_id": episode.task_id,
                "canonical_instruction": episode.canonical_instruction,
                "source_frame_count": episode.elapsed_steps,
            }
            for episode in self.source.ordered_episodes
        ]
        return {
            "schema_version": "langmani-m3b-export-plan-v1",
            "dry_run": True,
            "source_collection_run_id": self.source.manifest.collection_run_id,
            "source_run_fingerprint": self.source.config_fingerprint,
            "source_archive_digest": self.source.archive_digest,
            "export_fingerprint": self.export_fingerprint,
            "source_scene_group_count": len(self.source.ordered_groups),
            "source_episode_count": len(self.source.ordered_episodes),
            "expected_frame_count": self.expected_frame_count,
            "split_scene_group_counts": {
                split.value: split_counts[split.value] for split in DatasetSplit
            },
            "expected_features": config.feature_contract.to_lerobot_features(),
            "expected_output_root": str(Path(config.output_root).resolve()),
            "ordered_episodes": ordered,
            "runtime_versions": self.runtime_info.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class LeRobotExportOutcome:
    """Completed export result without large frames or source states."""

    output_root: Path
    manifest: LeRobotExportManifest
    summary: LeRobotDatasetSummary
    validation_report: LeRobotValidationReport


def build_export_plan(
    config: LeRobotExportConfig,
    *,
    source_verification_report: str | Path,
) -> LeRobotExportPlan:
    """Validate M3A and derive canonical ordering, splits, and export identity."""
    if not isinstance(config, LeRobotExportConfig):
        raise TypeError("config must be a LeRobotExportConfig")
    mode = "smoke" if config.mode is ExportMode.SMOKE else "full"
    source = validate_m3a_source_for_export(
        config.source_root,
        verification_report_path=source_verification_report,
        mode=mode,
        expected_run_fingerprint=config.expected_source_run_fingerprint,
        expected_schema_version=config.expected_source_schema_version,
    )
    if source.archive_digest != config.expected_source_archive_digest:
        raise LeRobotExportError("configured source archive digest is stale")
    if source.manifest.collection_run_id != config.expected_source_collection_run_id:
        raise LeRobotExportError("configured source collection run ID is stale")
    ordered_ids = tuple(item.raw_trajectory_id for item in source.ordered_episodes)
    fingerprint = stable_export_fingerprint(config, ordered_ids)
    group_ids = tuple(_accepted_group_id(group) for group in source.ordered_groups)
    assignments = build_split_assignments(group_ids, config.split_config)
    tasks_by_group = {
        _accepted_group_id(group): tuple(episode.task_id for episode in group.episodes)
        for group in source.ordered_groups
    }
    task_counts_by_split(assignments, tasks_by_group)
    runtime = inspect_lerobot_runtime()
    if runtime.lerobot != config.lerobot_version:
        raise LeRobotExportError("installed LeRobot version differs from export config")
    if runtime.av != config.pyav_version:
        raise LeRobotExportError("installed PyAV version differs from export config")
    libraries = dict(runtime.av_libraries)
    if libraries.get("libavcodec") != config.libavcodec_version:
        raise LeRobotExportError("installed libavcodec version differs from export config")
    return LeRobotExportPlan(
        source=source,
        export_fingerprint=fingerprint,
        split_assignments=assignments,
        expected_frame_count=sum(episode.elapsed_steps for episode in source.ordered_episodes),
        runtime_info=runtime,
    )


def export_lerobot_dataset(
    config: LeRobotExportConfig,
    *,
    source_verification_report: str | Path,
    clean_staging: bool = False,
) -> LeRobotExportOutcome:
    """Export, finalize once, independently validate, and atomically promote."""
    plan = build_export_plan(config, source_verification_report=source_verification_report)
    layout = prepare_staging_layout(
        config.output_root,
        plan.export_fingerprint,
        clean_staging=clean_staging,
    )
    writer: LeRobotWriterAdapter | None = None
    try:
        writer = LeRobotWriterAdapter.create(
            root=layout.dataset_root,
            repo_id=config.repo_id,
            fps=config.fps,
            feature_contract=config.feature_contract,
            codec=config.video_codec,
            data_files_size_in_mb=config.data_files_size_in_mb,
            video_files_size_in_mb=config.video_files_size_in_mb,
        )
        records = _write_all_episodes(config, plan, writer)
        if writer.saved_episodes != len(plan.source.ordered_episodes):
            raise LeRobotExportError("writer episode count differs from the canonical source")
        writer.finalize_once()
        if writer.finalize_calls != 1 or writer.state is not WriterState.FINALIZED:
            raise LeRobotExportError("LeRobot writer did not finalize exactly once")
        manifest = _build_manifest(config, plan, records)
        _write_prevalidation_sidecars(layout.dataset_root, config, manifest)

        # Import lazily to keep type/identity utilities usable without loading video APIs.
        from langmani.datasets.lerobot_validation import validate_lerobot_dataset

        validation_report, summary = validate_lerobot_dataset(
            layout.dataset_root,
            source_root=config.source_root,
            source_verification_report=source_verification_report,
            require_completion_marker=False,
            metadata_only=False,
        )
        _write_postvalidation_sidecars(layout.dataset_root, summary, validation_report)
        if not validation_report.passed:
            raise LeRobotExportError("independent staged validation rejected the export")

        reader = ValidatedRawEpisodeReader(plan.source)
        reader.revalidate_source()
        _fsync_tree(layout.dataset_root)
        final_root = promote_staged_dataset(layout)
        # Remove the now-empty staging ownership container before the completion
        # marker, so complete.json is the final successful filesystem mutation.
        shutil.rmtree(layout.container)
        outcome = LeRobotExportOutcome(
            output_root=final_root,
            manifest=manifest,
            summary=summary,
            validation_report=validation_report,
        )
        write_completion_marker(final_root, plan.export_fingerprint)
        return outcome
    except Exception as error:
        cleanup_errors: tuple[str, ...] = ()
        if writer is not None and writer.state is not WriterState.FINALIZED:
            cleanup_errors = writer.close_failed()
        _write_failure_report(layout, error, cleanup_errors)
        if isinstance(error, LeRobotExportError | LeRobotWriterError):
            raise
        raise LeRobotExportError(f"{type(error).__name__}: {error}") from error


def _write_all_episodes(
    config: LeRobotExportConfig,
    plan: LeRobotExportPlan,
    writer: LeRobotWriterAdapter,
) -> tuple[EpisodeExportRecord, ...]:
    source_reader = ValidatedRawEpisodeReader(plan.source)
    split_by_group = assignment_map(plan.split_assignments)
    record_group = {
        episode.raw_trajectory_id: _accepted_group_id(group)
        for group in plan.source.ordered_groups
        for episode in group.episodes
    }
    records: list[EpisodeExportRecord] = []
    reconstructor = ManiSkillObservationReconstructor(
        sim_backend=plan.source.manifest.config.sim_backend,
        render_backend=config.camera_contract.render_backend,
    )
    with reconstructor:
        for episode_index, source_record in enumerate(plan.source.ordered_episodes):
            raw = source_reader.read_episode(source_record.raw_trajectory_id)
            reconstructor.begin_episode(source_record)
            digest = RawRenderDigest()
            frame_count = 0
            for frame_index, (state_dict, action) in enumerate(raw.iter_training_pairs()):
                policy_frame = reconstructor.reconstruct(state_dict)
                digest.add(frame_index, policy_frame.rgb)
                writer.add_policy_frame(
                    rgb=policy_frame.rgb,
                    state=policy_frame.state,
                    action=action,
                    task=source_record.canonical_instruction,
                )
                frame_count += 1
            if frame_count != raw.transition_count or frame_count != source_record.elapsed_steps:
                raise LeRobotExportError("T actions did not produce exactly T policy frames")
            writer.save_episode()
            group_id = record_group[source_record.raw_trajectory_id]
            source_checksum = sha256_hex(
                {
                    "h5_group": source_record.h5_group,
                    "h5_sha256": source_record.h5_sha256,
                    "json_sha256": source_record.json_sha256,
                    "raw_trajectory_id": source_record.raw_trajectory_id,
                }
            )
            records.append(
                EpisodeExportRecord(
                    source_collection_run_id=plan.source.manifest.collection_run_id,
                    source_run_fingerprint=plan.source.config_fingerprint,
                    source_archive_digest=plan.source.archive_digest,
                    source_episode_id=source_record.raw_trajectory_id,
                    source_scene_group_id=group_id,
                    source_scene_seed=source_record.scene_seed,
                    scene_id=source_record.scene_id,
                    task_id=source_record.task_id,
                    target_object_id=source_record.task_spec.target_object_id,
                    target_bin_id=source_record.task_spec.target_bin_id,
                    instruction_template_id=source_record.task_spec.instruction_template_id,
                    canonical_instruction=source_record.canonical_instruction,
                    source_shard_id=source_record.source_shard_id,
                    source_shard_path=source_record.h5_path,
                    source_trajectory_key=source_record.h5_group,
                    source_h5_sha256=source_record.h5_sha256,
                    source_json_sha256=source_record.json_sha256,
                    source_checksum=source_checksum,
                    source_frame_count=source_record.elapsed_steps,
                    lerobot_episode_index=episode_index,
                    split=split_by_group[group_id].split,
                    raw_render_digest=digest.hexdigest(),
                    output_frame_count=frame_count,
                )
            )
    source_reader.revalidate_source()
    return tuple(records)


def _build_manifest(
    config: LeRobotExportConfig,
    plan: LeRobotExportPlan,
    records: tuple[EpisodeExportRecord, ...],
) -> LeRobotExportManifest:
    # A manifest must never persist machine paths. Paths are operational only;
    # replacing them does not alter identity_dict or the export fingerprint.
    portable_config = replace(
        config,
        source_root="<authoritative-m3a-source>",
        output_root=".",
    )
    return LeRobotExportManifest(
        export_schema_version=config.output_schema_version,
        export_fingerprint=plan.export_fingerprint,
        source_collection_run_id=plan.source.manifest.collection_run_id,
        source_run_fingerprint=plan.source.config_fingerprint,
        source_archive_digest=plan.source.archive_digest,
        source_schema_version=plan.source.manifest.collection_schema_version,
        repo_id=config.repo_id,
        config=portable_config,
        ordered_source_episode_ids=tuple(
            item.raw_trajectory_id for item in plan.source.ordered_episodes
        ),
        split_assignments=plan.split_assignments,
        episodes=records,
        runtime_versions={
            key: str(value)
            for key, value in plan.runtime_info.to_dict().items()
            if key != "av_libraries"
        }
        | {f"av.{key}": value for key, value in plan.runtime_info.av_libraries},
        finalized=True,
    )


def _write_prevalidation_sidecars(
    dataset_root: Path,
    config: LeRobotExportConfig,
    manifest: LeRobotExportManifest,
) -> None:
    sidecar = dataset_root / SIDECAR_DIRECTORY
    _write_json_atomic(sidecar / "export_config.json", config.portable_dict())
    _write_json_atomic(sidecar / "export_manifest.json", manifest.to_dict())
    _write_json_atomic(
        sidecar / "source_episode_mapping.json",
        {
            "schema_version": "langmani-m3b-source-mapping-v1",
            "export_fingerprint": manifest.export_fingerprint,
            "episodes": [record.to_dict() for record in manifest.episodes],
        },
    )
    split_indices = episode_indices_by_split(manifest.episodes)
    _write_json_atomic(
        sidecar / "split_manifest.json",
        {
            "schema_version": "langmani-m3b-splits-v1",
            "export_fingerprint": manifest.export_fingerprint,
            "split_config": manifest.config.split_config.to_dict(),
            "scene_group_assignments": [item.to_dict() for item in manifest.split_assignments],
            "episode_indices": {key: list(value) for key, value in split_indices.items()},
        },
    )
    _write_text_atomic(sidecar / "dataset_card.md", _dataset_card(manifest))


def _write_postvalidation_sidecars(
    dataset_root: Path,
    summary: LeRobotDatasetSummary,
    report: LeRobotValidationReport,
) -> None:
    sidecar = dataset_root / SIDECAR_DIRECTORY
    _write_json_atomic(sidecar / "dataset_summary.json", summary.to_dict())
    _write_json_atomic(sidecar / "validation_report.json", report.to_dict())


def _dataset_card(manifest: LeRobotExportManifest) -> str:
    return (
        "# LangMani PickPlaceByInstruction LeRobotDataset v3\n\n"
        "This is a deterministic derived representation of a validated M3A raw archive. "
        "The M3A archive remains authoritative.\n\n"
        f"- Export fingerprint: `{manifest.export_fingerprint}`\n"
        f"- Source run fingerprint: `{manifest.source_run_fingerprint}`\n"
        f"- Episodes: {len(manifest.episodes)}\n"
        "- Policy features: `observation.images.base_camera`, `observation.state`, `action`\n"
        "- Temporal contract: pre-action state[t] and RGB[t] paired with exact action[t]\n"
        "- Camera: fixed 256x256 RGB base_camera\n"
        "- State: PandaPolicyStateV0 (nine joint positions)\n"
        "- Control mode: pd_joint_pos\n"
        "- Hub publication: not performed\n"
    )


def _write_failure_report(
    layout: StagingLayout,
    error: Exception,
    cleanup_errors: tuple[str, ...],
) -> None:
    payload = {
        "schema_version": "langmani-m3b-export-failure-v1",
        "export_fingerprint": layout.export_fingerprint,
        "status": "failed",
        "exception_type": type(error).__name__,
        "exception_message": str(error) or repr(error),
        "cleanup_errors": list(cleanup_errors),
    }
    report_path = layout.container / "failure_report.json"
    if not layout.container.exists() and layout.final_root.is_dir():
        report_path = layout.final_root / SIDECAR_DIRECTORY / "failure_report.json"
    with suppress(OSError):
        _write_json_atomic(report_path, payload)


def _episode_group_id(source: ValidatedM3ASource, raw_trajectory_id: str) -> str:
    for group in source.ordered_groups:
        if any(item.raw_trajectory_id == raw_trajectory_id for item in group.episodes):
            return _accepted_group_id(group)
    raise LeRobotExportError(f"source episode has no accepted scene group: {raw_trajectory_id}")


def _accepted_group_id(group: Any) -> str:
    value = getattr(group, "accepted_scene_group_id", None)
    if not isinstance(value, str) or not value:
        raise LeRobotExportError("accepted source group is missing its stable ID")
    return value


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _fsync_tree(root: Path) -> None:
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        with path.open("rb") as stream:
            os.fsync(stream.fileno())


__all__ = [
    "LeRobotExportError",
    "LeRobotExportOutcome",
    "LeRobotExportPlan",
    "build_export_plan",
    "export_lerobot_dataset",
]
