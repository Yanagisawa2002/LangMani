"""Offline schema, corruption, checksum, and provenance inspection for M3A."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from langmani.collection.manifest import load_manifest, validate_manifest_projections
from langmani.datasets.archive import validate_native_archive_pair
from langmani.datasets.identity import sha256_hex
from langmani.datasets.types import (
    AttemptOutcome,
    CollectionManifest,
    CollectionStatus,
    RawDatasetSummary,
)


class DatasetInspectionError(RuntimeError):
    """Raised when an authoritative raw archive fails strict inspection."""


def source_archive_digest(manifest: CollectionManifest) -> str:
    """Bind canonical accepted content and immutable shard bytes to one digest.

    The payload deliberately excludes filesystem paths and timestamps. Accepted
    scene groups retain their authoritative manifest order, and episodes retain
    each group's canonical six-task order. Rejected groups and failed attempts
    are diagnostics rather than exportable source content and therefore do not
    contribute to this accepted-archive identity.
    """
    if not isinstance(manifest, CollectionManifest):
        raise TypeError("manifest must be a CollectionManifest")

    accepted_groups = tuple(group for group in manifest.scene_groups if group.accepted)
    payload = {
        "collection_schema_version": manifest.collection_schema_version,
        "collection_run_id": manifest.collection_run_id,
        "config_fingerprint": manifest.schedule.config_fingerprint,
        "accepted_scene_groups": [
            {
                "candidate_scene_index": group.candidate_scene_index,
                "candidate_scene_id": group.candidate_scene_id,
                "accepted_scene_group_id": group.accepted_scene_group_id,
                "scene_seed": group.scene_seed,
                "scene_id": group.scene_id,
                "bundle_h5_sha256": group.bundle_h5_sha256,
                "bundle_json_sha256": group.bundle_json_sha256,
                "episodes": [
                    {
                        "scheduled_episode_id": episode.scheduled_episode_id,
                        "attempt_id": episode.attempt_id,
                        "raw_trajectory_id": episode.raw_trajectory_id,
                        "source_shard_id": episode.source_shard_id,
                        "native_episode_id": episode.native_episode_id,
                        "h5_group": episode.h5_group,
                        "elapsed_steps": episode.elapsed_steps,
                        "scene_seed": episode.scene_seed,
                        "scene_id": episode.scene_id,
                        "task_id": episode.task_id,
                        "canonical_instruction": episode.canonical_instruction,
                    }
                    for episode in group.episodes
                ],
            }
            for group in accepted_groups
        ],
        "source_shards": [
            {
                "source_shard_id": shard.source_shard_id,
                "shard_index": shard.shard_index,
                "h5_sha256": shard.h5_sha256,
                "json_sha256": shard.json_sha256,
                "h5_size_bytes": shard.h5_size_bytes,
                "json_size_bytes": shard.json_size_bytes,
                "raw_trajectory_ids": list(shard.raw_trajectory_ids),
            }
            for shard in manifest.source_shards
        ],
    }
    return f"sha256:{sha256_hex(payload)}"


def inspect_raw_dataset(root: str | Path) -> RawDatasetSummary:
    """Validate every accepted shard and all manifest cross-references."""
    dataset_root = Path(root).resolve()
    manifest = load_manifest(dataset_root)
    validate_manifest_projections(dataset_root, manifest)
    manifest_raw = {record.raw_trajectory_id: record for record in manifest.raw_episodes}
    attempts_by_id = {record.attempt_id: record for record in manifest.attempts}
    scheduled_by_id = {record.scheduled_episode_id: record for record in manifest.schedule.episodes}
    accepted_group_by_raw = {
        episode.raw_trajectory_id: group.accepted_scene_group_id
        for group in manifest.scene_groups
        if group.accepted
        for episode in group.episodes
    }
    accepted_group_record_by_raw = {
        episode.raw_trajectory_id: group
        for group in manifest.scene_groups
        if group.accepted
        for episode in group.episodes
    }
    seen_raw: set[str] = set()
    accepted_h5_paths: set[Path] = set()
    accepted_json_paths: set[Path] = set()

    for shard in sorted(manifest.source_shards, key=lambda item: item.shard_index):
        h5_path = _confined_path(dataset_root, shard.h5_path)
        json_path = _confined_path(dataset_root, shard.json_path)
        if json_path != h5_path.with_suffix(".json"):
            raise DatasetInspectionError(
                f"shard JSON is not paired with HDF5: {shard.source_shard_id}"
            )
        accepted_h5_paths.add(h5_path)
        accepted_json_paths.add(json_path)
        validation = validate_native_archive_pair(
            h5_path,
            json_path=json_path,
            require_accepted_success=True,
            require_langmani_metadata=True,
        )
        if validation.h5_sha256 != shard.h5_sha256:
            raise DatasetInspectionError(f"HDF5 checksum mismatch: {h5_path}")
        if validation.json_sha256 != shard.json_sha256:
            raise DatasetInspectionError(f"JSON checksum mismatch: {json_path}")
        if validation.h5_size_bytes != shard.h5_size_bytes:
            raise DatasetInspectionError(f"HDF5 size mismatch: {h5_path}")
        if validation.json_size_bytes != shard.json_size_bytes:
            raise DatasetInspectionError(f"JSON size mismatch: {json_path}")
        if len(validation.episodes) != shard.episode_count:
            raise DatasetInspectionError(f"episode count mismatch: {shard.source_shard_id}")
        if shard.episode_count > manifest.config.shard_size:
            raise DatasetInspectionError(f"shard exceeds configured size: {shard.source_shard_id}")
        location_raw_ids = tuple(location.raw_trajectory_id for location in validation.episodes)
        if None in location_raw_ids or location_raw_ids != shard.raw_trajectory_ids:
            raise DatasetInspectionError(f"raw trajectory order mismatch: {shard.source_shard_id}")
        archive_payload = json.loads(json_path.read_text(encoding="utf-8"))
        root_metadata = archive_payload.get("langmani")
        archive_episodes = archive_payload.get("episodes")
        env_info = archive_payload.get("env_info")
        env_kwargs = env_info.get("env_kwargs") if isinstance(env_info, Mapping) else None
        if (
            not isinstance(root_metadata, Mapping)
            or not isinstance(archive_episodes, list)
            or not isinstance(env_kwargs, Mapping)
        ):
            raise DatasetInspectionError(f"archive metadata is malformed: {json_path}")
        if (
            env_info.get("env_id") != manifest.config.environment_id
            or env_kwargs.get("control_mode") != manifest.config.control_mode
            or env_kwargs.get("sim_backend") != manifest.config.sim_backend
            or env_kwargs.get("num_envs") != 1
            or env_kwargs.get("obs_mode") != "none"
            or env_kwargs.get("reward_mode") != "none"
        ):
            raise DatasetInspectionError(
                f"archive environment contract differs from collection config: {shard.source_shard_id}"
            )
        if (
            root_metadata.get("collection_run_id") != manifest.collection_run_id
            or root_metadata.get("expert_config_fingerprint")
            != manifest.config.expert_config_fingerprint
            or root_metadata.get("source_shard_id") != shard.source_shard_id
            or root_metadata.get("shard_index") != shard.shard_index
        ):
            raise DatasetInspectionError(
                f"archive root provenance differs from manifest: {shard.source_shard_id}"
            )
        episode_metadata_by_id = {
            item.get("episode_id"): item
            for item in archive_episodes
            if isinstance(item, Mapping) and isinstance(item.get("episode_id"), int)
        }
        for location in validation.episodes:
            raw_id = location.raw_trajectory_id
            assert raw_id is not None
            if raw_id in seen_raw:
                raise DatasetInspectionError(f"duplicate raw trajectory in source shards: {raw_id}")
            seen_raw.add(raw_id)
            record = manifest_raw.get(raw_id)
            if record is None:
                raise DatasetInspectionError(
                    f"source shard contains unmanifested episode: {raw_id}"
                )
            if record.source_shard_id != shard.source_shard_id:
                raise DatasetInspectionError(f"source shard ID mismatch for {raw_id}")
            if record.native_episode_id != location.native_episode_id:
                raise DatasetInspectionError(f"native episode ID mismatch for {raw_id}")
            if record.h5_group != location.h5_group:
                raise DatasetInspectionError(f"HDF5 group mismatch for {raw_id}")
            if record.h5_path != shard.h5_path or record.json_path != shard.json_path:
                raise DatasetInspectionError(f"manifest path mismatch for {raw_id}")
            if record.h5_sha256 != shard.h5_sha256 or record.json_sha256 != shard.json_sha256:
                raise DatasetInspectionError(f"episode checksum reference mismatch for {raw_id}")
            attempt = attempts_by_id.get(record.attempt_id)
            scheduled = scheduled_by_id.get(record.scheduled_episode_id)
            if attempt is None:
                raise DatasetInspectionError(f"raw trajectory has no attempt record: {raw_id}")
            if scheduled is None:
                raise DatasetInspectionError(f"raw trajectory has no scheduled episode: {raw_id}")
            if (
                location.scheduled_episode_id != record.scheduled_episode_id
                or location.attempt_id != record.attempt_id
                or location.source_shard_id != record.source_shard_id
                or location.elapsed_steps != record.elapsed_steps
                or location.scene_seed != record.scene_seed
                or location.scene_id != record.scene_id
                or location.task_spec != record.task_spec
                or location.task_id != record.task_id
                or location.canonical_instruction != record.canonical_instruction
                or location.collection_run_id != manifest.collection_run_id
                or location.candidate_scene_id != scheduled.candidate_scene_id
                or location.accepted_scene_group_id != accepted_group_by_raw.get(raw_id)
                or location.expert_config_fingerprint != manifest.config.expert_config_fingerprint
            ):
                raise DatasetInspectionError(
                    f"native archive provenance differs from raw record: {raw_id}"
                )
            if (
                attempt.outcome is not AttemptOutcome.ACCEPTED
                or attempt.scheduled_episode_id != record.scheduled_episode_id
                or attempt.raw_trajectory_id != raw_id
                or not attempt.trajectory_retained
                or attempt.expert_result.to_dict() != record.expert_result.to_dict()
                or attempt.replay_validation is None
                or attempt.replay_validation.to_dict() != record.replay_validation.to_dict()
            ):
                raise DatasetInspectionError(
                    f"accepted raw record disagrees with owning attempt: {raw_id}"
                )
            if (
                scheduled.scene_seed != record.scene_seed
                or scheduled.scene_id != record.scene_id
                or scheduled.task_spec != record.task_spec
                or scheduled.task_id != record.task_id
                or scheduled.canonical_instruction != record.canonical_instruction
            ):
                raise DatasetInspectionError(
                    f"raw record provenance differs from schedule: {raw_id}"
                )
            archived_episode = episode_metadata_by_id.get(location.native_episode_id)
            archived_langmani = (
                archived_episode.get("langmani") if isinstance(archived_episode, Mapping) else None
            )
            if not isinstance(archived_langmani, Mapping):
                raise DatasetInspectionError(f"archive episode metadata is absent: {raw_id}")
            archived_source_group = archived_langmani.get("source_group_bundle")
            accepted_group = accepted_group_record_by_raw.get(raw_id)
            if (
                archived_langmani.get("collection_schema_version")
                != manifest.collection_schema_version
                or archived_langmani.get("collection_run_id") != manifest.collection_run_id
                or archived_langmani.get("candidate_scene_id") != scheduled.candidate_scene_id
                or archived_langmani.get("accepted_scene_group_id")
                != accepted_group_by_raw.get(raw_id)
                or archived_langmani.get("source_shard_id") != record.source_shard_id
                or archived_langmani.get("scheduled_episode_id") != record.scheduled_episode_id
                or archived_langmani.get("attempt_id") != record.attempt_id
                or archived_langmani.get("raw_trajectory_id") != raw_id
                or archived_langmani.get("expert_config_fingerprint")
                != manifest.config.expert_config_fingerprint
                or archived_langmani.get("expert_result") != record.expert_result.to_dict()
                or archived_langmani.get("replay_validation") != record.replay_validation.to_dict()
                or archived_langmani.get("final_environment_evaluation")
                != dict(record.final_environment_evaluation)
                or not isinstance(archived_source_group, Mapping)
                or accepted_group is None
                or archived_source_group.get("accepted_scene_group_id")
                != accepted_group.accepted_scene_group_id
                or archived_source_group.get("h5_sha256") != accepted_group.bundle_h5_sha256
                or archived_source_group.get("json_sha256") != accepted_group.bundle_json_sha256
            ):
                raise DatasetInspectionError(
                    f"archive episode evidence differs from manifest: {raw_id}"
                )
            if (
                not record.accepted
                or not record.structural_valid
                or not record.replay_validation.passed
            ):
                raise DatasetInspectionError(
                    f"accepted raw record lacks validation evidence: {raw_id}"
                )

    if seen_raw != set(manifest_raw):
        missing = sorted(set(manifest_raw) - seen_raw)
        raise DatasetInspectionError(f"manifested raw episodes are absent from shards: {missing}")

    accepted_directory = dataset_root / "accepted" / "shards"
    actual_h5 = set(accepted_directory.glob("*.h5")) if accepted_directory.exists() else set()
    actual_json = set(accepted_directory.glob("*.json")) if accepted_directory.exists() else set()
    if actual_h5 != accepted_h5_paths or actual_json != accepted_json_paths:
        extra = sorted(
            str(path)
            for path in (actual_h5 - accepted_h5_paths) | (actual_json - accepted_json_paths)
        )
        missing = sorted(
            str(path)
            for path in (accepted_h5_paths - actual_h5) | (accepted_json_paths - actual_json)
        )
        raise DatasetInspectionError(
            f"accepted shard file set differs from manifest; missing={missing}, extra={extra}"
        )

    for group in manifest.scene_groups:
        if any(attempt_id not in attempts_by_id for attempt_id in group.attempt_ids):
            raise DatasetInspectionError(
                f"scene group references an unknown attempt: {group.candidate_scene_id}"
            )
        if group.accepted:
            if len(group.episodes) != 6:
                raise DatasetInspectionError(
                    f"accepted scene group is partial: {group.candidate_scene_id}"
                )
            if any(episode.raw_trajectory_id not in seen_raw for episode in group.episodes):
                raise DatasetInspectionError(
                    f"accepted scene group references absent raw data: {group.candidate_scene_id}"
                )

    task_counts = Counter(record.task_id for record in manifest.raw_episodes)
    expected_per_task = manifest.accepted_scene_group_count
    if set(task_counts.values()) not in (set(), {expected_per_task}):
        raise DatasetInspectionError(
            f"accepted task counts are not balanced at {expected_per_task}: {dict(task_counts)}"
        )
    if manifest.accepted_scene_group_count and len(task_counts) != 6:
        raise DatasetInspectionError("accepted archive does not contain all six TaskSpecs")
    if manifest.status is CollectionStatus.COMPLETE:
        expected_episodes = 6 * manifest.config.target_complete_scene_count
        if len(manifest.raw_episodes) != expected_episodes:
            raise DatasetInspectionError(
                f"complete archive has {len(manifest.raw_episodes)} episodes, expected {expected_episodes}"
            )
        if any(
            count != manifest.config.target_complete_scene_count for count in task_counts.values()
        ):
            raise DatasetInspectionError("complete archive is not exactly balanced per TaskSpec")
    return RawDatasetSummary.from_manifest(manifest)


def _confined_path(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise DatasetInspectionError(
            f"archive path escapes dataset root: {relative_path}"
        ) from error
    return candidate


__all__ = ["DatasetInspectionError", "inspect_raw_dataset", "source_archive_digest"]
