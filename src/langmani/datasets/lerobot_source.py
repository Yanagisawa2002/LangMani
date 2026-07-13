"""Validated M3A source boundary for deterministic M3B export.

The exporter must not treat an HDF5 trajectory key as an acceptance decision.
This module first runs the complete M3A inspector, then loads the authoritative
manifest, consumes content-bound M3A v3 verification evidence, and only then
issues a gated source object accepted by :class:`ValidatedRawEpisodeReader`.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np

from langmani.collection.inspection import inspect_raw_dataset, source_archive_digest
from langmani.collection.manifest import load_manifest
from langmani.datasets.archive import sha256_file
from langmani.datasets.identity import COLLECTION_SCHEMA_VERSION
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.datasets.types import (
    CollectionManifest,
    CollectionStatus,
    RawDatasetSummary,
    RawEpisodeRecord,
    ReplayValidationMode,
    SceneGroupRecord,
    SourceShardRecord,
)
from langmani.environments.specs import stable_task_id

type SourceExportMode = Literal["smoke", "full"]

M3A_VERIFICATION_SCHEMA_VERSION = "langmani-m3a-verification-v3"
_SOURCE_GATE_TOKEN = object()


class M3ASourceValidationError(RuntimeError):
    """Raised when an M3A archive is not eligible for M3B export."""


class RawEpisodeReadError(RuntimeError):
    """Raised when gated raw data changes or violates its fixed time contract."""


@dataclass(frozen=True, slots=True, init=False)
class ValidatedM3ASource:
    """Capability object issued only after the complete M3A input gate passes."""

    root: Path
    verification_report_path: Path
    mode: SourceExportMode
    manifest: CollectionManifest
    summary: RawDatasetSummary
    config_fingerprint: str
    archive_digest: str
    ordered_groups: tuple[SceneGroupRecord, ...]
    ordered_episodes: tuple[RawEpisodeRecord, ...]
    _gate_token: object

    def __init__(
        self,
        *,
        root: Path,
        verification_report_path: Path,
        mode: SourceExportMode,
        manifest: CollectionManifest,
        summary: RawDatasetSummary,
        config_fingerprint: str,
        archive_digest: str,
        ordered_groups: tuple[SceneGroupRecord, ...],
        ordered_episodes: tuple[RawEpisodeRecord, ...],
        _gate_token: object,
    ) -> None:
        if _gate_token is not _SOURCE_GATE_TOKEN:
            raise TypeError(
                "ValidatedM3ASource can only be created by validate_m3a_source_for_export"
            )
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "verification_report_path", verification_report_path)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "config_fingerprint", config_fingerprint)
        object.__setattr__(self, "archive_digest", archive_digest)
        object.__setattr__(self, "ordered_groups", ordered_groups)
        object.__setattr__(self, "ordered_episodes", ordered_episodes)
        object.__setattr__(self, "_gate_token", _gate_token)


@dataclass(frozen=True, slots=True)
class RawEpisodeData:
    """Exact actions and complete T+1 state sequence from one accepted episode."""

    record: RawEpisodeRecord
    actions: np.ndarray
    _states: tuple[dict[str, Any], ...]

    @property
    def transition_count(self) -> int:
        return int(self.actions.shape[0])

    @property
    def state_count(self) -> int:
        return len(self._states)

    @property
    def terminal_state_index(self) -> int:
        return self.transition_count

    def state_at(self, index: int) -> dict[str, Any]:
        """Return a copy of state[index], including the terminal state at index T."""
        _require_state_index(index, state_count=self.state_count)
        return _copy_state(self._states[index])

    def training_state_at(self, index: int) -> dict[str, Any]:
        """Return a pre-action training state; state[T] is deliberately rejected."""
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("training state index must be an integer")
        if not 0 <= index < self.transition_count:
            raise IndexError("training state index must be in [0, T-1]")
        return self.state_at(index)

    def iter_training_pairs(self) -> Iterator[tuple[dict[str, Any], np.ndarray]]:
        """Yield exactly T pairs of state[t] and the unmodified action[t]."""
        for index in range(self.transition_count):
            yield self.training_state_at(index), np.array(self.actions[index], copy=True)


def validate_m3a_source_for_export(
    source_root: str | Path,
    *,
    verification_report_path: str | Path,
    mode: SourceExportMode,
    expected_run_fingerprint: str,
    expected_schema_version: str = COLLECTION_SCHEMA_VERSION,
) -> ValidatedM3ASource:
    """Run the non-bypassable M3A gate and return canonical accepted content.

    Ordering is intentional: the complete inspector runs before this function
    loads or reasons about the manifest. A failed inspector therefore cannot be
    bypassed by a parseable manifest or an existing HDF5 trajectory key.
    """
    root = Path(source_root).resolve()
    report_path = Path(verification_report_path).resolve()
    _validate_mode(mode)
    _require_sha256_fingerprint(expected_run_fingerprint, "expected_run_fingerprint")
    if expected_schema_version != COLLECTION_SCHEMA_VERSION:
        raise M3ASourceValidationError(
            f"unsupported expected M3A schema {expected_schema_version!r}"
        )

    try:
        summary = inspect_raw_dataset(root)
    except Exception as error:
        raise M3ASourceValidationError(
            f"M3A source inspection failed: {type(error).__name__}: {str(error) or repr(error)}"
        ) from error
    try:
        manifest = load_manifest(root)
    except Exception as error:
        raise M3ASourceValidationError(
            f"M3A manifest load failed after inspection: "
            f"{type(error).__name__}: {str(error) or repr(error)}"
        ) from error

    _validate_manifest_identity(
        manifest,
        summary,
        mode=mode,
        expected_run_fingerprint=expected_run_fingerprint,
        expected_schema_version=expected_schema_version,
    )
    ordered_groups, ordered_episodes = _canonical_accepted_content(manifest, summary, mode=mode)
    digest = source_archive_digest(manifest)
    verification = _read_verification_report(report_path)
    _validate_verification_report(
        verification,
        source_root=root,
        mode=mode,
        collection_run_id=manifest.collection_run_id,
        config_fingerprint=manifest.schedule.config_fingerprint,
        archive_digest=digest,
    )
    return ValidatedM3ASource(
        root=root,
        verification_report_path=report_path,
        mode=mode,
        manifest=manifest,
        summary=summary,
        config_fingerprint=manifest.schedule.config_fingerprint,
        archive_digest=digest,
        ordered_groups=ordered_groups,
        ordered_episodes=ordered_episodes,
        _gate_token=_SOURCE_GATE_TOKEN,
    )


class ValidatedRawEpisodeReader:
    """Read exact T/T+1 data only from a previously gated M3A source."""

    def __init__(self, source: ValidatedM3ASource) -> None:
        if (
            not isinstance(source, ValidatedM3ASource)
            or source._gate_token is not _SOURCE_GATE_TOKEN
        ):
            raise TypeError("source must be returned by validate_m3a_source_for_export")
        self.source = source
        self._episodes = {episode.raw_trajectory_id: episode for episode in source.ordered_episodes}
        self._shards = {shard.source_shard_id: shard for shard in source.manifest.source_shards}
        self._validated_shard_paths: dict[str, tuple[Path, Path]] = {}

    def read_episode(self, raw_trajectory_id: str) -> RawEpisodeData:
        """Read one accepted episode after rechecking its source shard pair."""
        if not isinstance(raw_trajectory_id, str) or not raw_trajectory_id:
            raise TypeError("raw_trajectory_id must be a non-empty string")
        record = self._episodes.get(raw_trajectory_id)
        if record is None:
            raise KeyError(
                f"raw trajectory is not in the gated accepted source: {raw_trajectory_id}"
            )
        h5_path, _ = self._validated_shard_pair(record)
        try:
            with h5py.File(h5_path, "r") as archive:
                group = archive.get(record.h5_group)
                if not isinstance(group, h5py.Group):
                    raise RawEpisodeReadError(f"missing accepted HDF5 group {record.h5_group}")
                if record.h5_group != f"traj_{record.native_episode_id}":
                    raise RawEpisodeReadError("manifest native episode ID and HDF5 group disagree")
                actions_node = group.get("actions")
                if not isinstance(actions_node, h5py.Dataset):
                    raise RawEpisodeReadError("accepted episode is missing actions")
                actions = np.asarray(actions_node[:])
                _validate_actions(actions, expected_steps=record.elapsed_steps)
                states_node = group.get("env_states")
                if not isinstance(states_node, h5py.Group):
                    raise RawEpisodeReadError("accepted episode is missing env_states")
                state_count = int(actions.shape[0]) + 1
                _validate_state_tree(states_node, expected_state_count=state_count)
                states = tuple(_read_state_at(states_node, index) for index in range(state_count))
        except RawEpisodeReadError:
            raise
        except OSError as error:
            raise RawEpisodeReadError(
                f"cannot read accepted HDF5 shard {h5_path}: {error}"
            ) from error

        actions.setflags(write=False)
        return RawEpisodeData(record=record, actions=actions, _states=states)

    def revalidate_source(self) -> None:
        """Re-inspect source content after export and reject any TOCTOU change."""
        try:
            inspect_raw_dataset(self.source.root)
            manifest = load_manifest(self.source.root)
            current_digest = source_archive_digest(manifest)
        except Exception as error:
            raise RawEpisodeReadError(
                "post-export M3A source inspection failed: "
                f"{type(error).__name__}: {str(error) or repr(error)}"
            ) from error
        if manifest.collection_run_id != self.source.manifest.collection_run_id:
            raise RawEpisodeReadError("M3A collection_run_id changed during export")
        if manifest.schedule.config_fingerprint != self.source.config_fingerprint:
            raise RawEpisodeReadError("M3A config fingerprint changed during export")
        if current_digest != self.source.archive_digest:
            raise RawEpisodeReadError("M3A source archive digest changed during export")

    def _validated_shard_pair(self, record: RawEpisodeRecord) -> tuple[Path, Path]:
        cached = self._validated_shard_paths.get(record.source_shard_id)
        if cached is not None:
            return cached
        shard = self._shards.get(record.source_shard_id)
        if shard is None:
            raise RawEpisodeReadError("accepted episode references an unknown source shard")
        _validate_episode_shard_reference(record, shard)
        h5_path = _confined_source_path(self.source.root, shard.h5_path)
        json_path = _confined_source_path(self.source.root, shard.json_path)
        if json_path != h5_path.with_suffix(".json"):
            raise RawEpisodeReadError("source shard HDF5 and JSON paths are not a pair")
        try:
            actual_h5_sha256 = sha256_file(h5_path)
            actual_json_sha256 = sha256_file(json_path)
        except OSError as error:
            raise RawEpisodeReadError(f"cannot checksum source shard pair: {error}") from error
        if actual_h5_sha256 != shard.h5_sha256:
            raise RawEpisodeReadError("source shard HDF5 checksum changed after the M3A gate")
        if actual_json_sha256 != shard.json_sha256:
            raise RawEpisodeReadError("source shard JSON checksum changed after the M3A gate")
        result = (h5_path, json_path)
        self._validated_shard_paths[record.source_shard_id] = result
        return result


def _validate_mode(mode: object) -> None:
    if mode not in ("smoke", "full"):
        raise M3ASourceValidationError("mode must be 'smoke' or 'full'")


def _validate_manifest_identity(
    manifest: CollectionManifest,
    summary: RawDatasetSummary,
    *,
    mode: SourceExportMode,
    expected_run_fingerprint: str,
    expected_schema_version: str,
) -> None:
    if manifest.collection_schema_version != expected_schema_version:
        raise M3ASourceValidationError("M3A manifest schema version is unsupported")
    if summary.collection_schema_version != expected_schema_version:
        raise M3ASourceValidationError("M3A inspection summary schema differs from the manifest")
    if (
        manifest.status is not CollectionStatus.COMPLETE
        or summary.status is not CollectionStatus.COMPLETE
    ):
        raise M3ASourceValidationError("M3B accepts only a complete M3A archive")
    if manifest.schedule.config_fingerprint != expected_run_fingerprint:
        raise M3ASourceValidationError(
            "M3A config fingerprint differs from the requested source run"
        )
    if manifest.config.replay_validation_mode is not ReplayValidationMode.ACTION_AND_STATE_AUDIT:
        raise M3ASourceValidationError("M3B requires M3A action replay plus state audit evidence")
    expected_groups = 1 if mode == "smoke" else 60
    if manifest.config.target_complete_scene_count != expected_groups:
        raise M3ASourceValidationError("M3A target scene-group count differs from export mode")


def _canonical_accepted_content(
    manifest: CollectionManifest,
    summary: RawDatasetSummary,
    *,
    mode: SourceExportMode,
) -> tuple[tuple[SceneGroupRecord, ...], tuple[RawEpisodeRecord, ...]]:
    expected_groups = 1 if mode == "smoke" else 60
    expected_episodes = expected_groups * len(CANONICAL_TASK_SPECS)
    accepted_groups = tuple(group for group in manifest.scene_groups if group.accepted)
    if len(accepted_groups) != expected_groups:
        raise M3ASourceValidationError(
            f"{mode} source requires exactly {expected_groups} accepted scene groups"
        )
    candidate_indices = tuple(group.candidate_scene_index for group in accepted_groups)
    if candidate_indices != tuple(sorted(candidate_indices)) or len(set(candidate_indices)) != len(
        candidate_indices
    ):
        raise M3ASourceValidationError("accepted scene groups are not in canonical candidate order")

    ordered: list[RawEpisodeRecord] = []
    for group in accepted_groups:
        if not group.complete or len(group.episodes) != len(CANONICAL_TASK_SPECS):
            raise M3ASourceValidationError("accepted M3A scene group is partial")
        if tuple(episode.task_spec for episode in group.episodes) != CANONICAL_TASK_SPECS:
            raise M3ASourceValidationError(
                "accepted scene-group episodes are not in canonical order"
            )
        if tuple(episode.scheduled_episode_id for episode in group.episodes) != (
            group.scheduled_episode_ids
        ):
            raise M3ASourceValidationError("accepted scene-group schedule mapping is inconsistent")
        ordered.extend(group.episodes)
    ordered_episodes = tuple(ordered)
    raw_ids = tuple(episode.raw_trajectory_id for episode in ordered_episodes)
    manifest_raw_ids = tuple(episode.raw_trajectory_id for episode in manifest.raw_episodes)
    if (
        len(ordered_episodes) != expected_episodes
        or len(set(raw_ids)) != expected_episodes
        or len(manifest_raw_ids) != expected_episodes
        or set(raw_ids) != set(manifest_raw_ids)
    ):
        raise M3ASourceValidationError(
            "accepted scene groups and manifest raw episodes are not an exact one-to-one set"
        )
    manifest_by_raw_id = {episode.raw_trajectory_id: episode for episode in manifest.raw_episodes}
    if any(
        manifest_by_raw_id[episode.raw_trajectory_id] != episode for episode in ordered_episodes
    ):
        raise M3ASourceValidationError(
            "accepted scene-group episode records differ from the manifest authority"
        )

    expected_per_task = expected_groups
    task_counts = Counter(episode.task_id for episode in ordered_episodes)
    expected_task_counts = {
        stable_task_id(task_spec): expected_per_task for task_spec in CANONICAL_TASK_SPECS
    }
    if task_counts != expected_task_counts:
        raise M3ASourceValidationError("accepted M3A task distribution is not exactly balanced")
    if (
        summary.accepted_scene_group_count != expected_groups
        or summary.accepted_episode_count != expected_episodes
        or dict(summary.task_episode_counts) != expected_task_counts
    ):
        raise M3ASourceValidationError("M3A inspection summary counts differ from accepted content")
    for episode in ordered_episodes:
        evaluation = episode.final_environment_evaluation
        replay = episode.replay_validation
        if not (
            episode.accepted
            and episode.structural_valid
            and episode.expert_result.success
            and episode.expert_result.total_environment_steps == episode.elapsed_steps
            and replay.passed
            and replay.mode is ReplayValidationMode.ACTION_AND_STATE_AUDIT
            and replay.recorded_success
            and replay.replay_success
            and replay.task_spec_matches
            and not replay.wrong_object_in_target_bin
            and not replay.target_in_wrong_bin
            and not replay.target_off_table
            and replay.recorded_action_steps == episode.elapsed_steps
            and replay.replayed_action_steps == episode.elapsed_steps
            and replay.state_audit_performed
            and replay.state_audit_passed is True
            and not replay.failure_codes
            and not replay.failure_reasons
            and evaluation.get("success") is True
            and evaluation.get("fail") is False
            and evaluation.get("target_in_wrong_bin", False) is False
            and evaluation.get("wrong_object_in_target_bin", False) is False
        ):
            raise M3ASourceValidationError("accepted M3A episode contains failure evidence")
    return accepted_groups, ordered_episodes


def _read_verification_report(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise M3ASourceValidationError(
            f"cannot read M3A verification report {path}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise M3ASourceValidationError("M3A verification report root must be a mapping")
    return payload


def _validate_verification_report(
    report: Mapping[str, Any],
    *,
    source_root: Path,
    mode: SourceExportMode,
    collection_run_id: str,
    config_fingerprint: str,
    archive_digest: str,
) -> None:
    expected_mode = "target_smoke" if mode == "smoke" else "target_full"
    if report.get("schema_version") != M3A_VERIFICATION_SCHEMA_VERSION:
        raise M3ASourceValidationError("M3A verification report schema is not supported v3")
    if (
        report.get("verification_mode") != expected_mode
        or report.get("validation_scope") != expected_mode
        or report.get("target_mode") is not True
    ):
        raise M3ASourceValidationError("M3A verification mode differs from the export mode")
    recorded_root = report.get("dataset_root")
    if not isinstance(recorded_root, str) or not Path(recorded_root).is_absolute():
        raise M3ASourceValidationError("M3A verification dataset_root must be absolute")
    if Path(recorded_root).resolve() != source_root:
        raise M3ASourceValidationError("M3A verification report belongs to a different source root")
    if (
        report.get("source_archive_identity_status") != "validated"
        or report.get("source_archive_identity_error") is not None
    ):
        raise M3ASourceValidationError("M3A verification report lacks validated source identity")
    if report.get("collection_run_id") != collection_run_id:
        raise M3ASourceValidationError("M3A verification collection_run_id is stale")
    if report.get("config_fingerprint") != config_fingerprint:
        raise M3ASourceValidationError("M3A verification config fingerprint is stale")
    if report.get("source_archive_digest") != archive_digest:
        raise M3ASourceValidationError("M3A verification source archive digest is stale")

    required_true = (
        "implementation_validated",
        "prior_target_gates_validated",
        "action_replay_validated",
        "physical_target_validated",
        "physical_acceptance",
        "passed",
    )
    if any(report.get(field) is not True for field in required_true):
        raise M3ASourceValidationError("M3A verification report lacks required target evidence")
    mode_flag = "expert_collection_smoke_validated" if mode == "smoke" else "full_dataset_validated"
    if report.get(mode_flag) is not True:
        raise M3ASourceValidationError(f"M3A verification report requires {mode_flag}=true")
    if mode == "smoke" and report.get("full_dataset_validated") is not False:
        raise M3ASourceValidationError(
            "M3A smoke verification must not claim full_dataset_validated=true"
        )


def _require_sha256_fingerprint(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise M3ASourceValidationError(f"{field_name} must be sha256:<64 lowercase hex>")
    digest = value.removeprefix("sha256:")
    if any(character not in "0123456789abcdef" for character in digest):
        raise M3ASourceValidationError(f"{field_name} must be sha256:<64 lowercase hex>")


def _validate_episode_shard_reference(
    record: RawEpisodeRecord,
    shard: SourceShardRecord,
) -> None:
    if (
        record.h5_path != shard.h5_path
        or record.json_path != shard.json_path
        or record.h5_sha256 != shard.h5_sha256
        or record.json_sha256 != shard.json_sha256
        or record.raw_trajectory_id not in shard.raw_trajectory_ids
    ):
        raise RawEpisodeReadError("accepted episode and source-shard provenance disagree")


def _confined_source_path(root: Path, relative_path: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise RawEpisodeReadError("source archive path must be a non-empty relative string")
    raw_path = Path(relative_path)
    if raw_path.is_absolute():
        raise RawEpisodeReadError("source archive path must be relative to the M3A root")
    candidate = (root / raw_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise RawEpisodeReadError("source archive path escapes the M3A root") from error
    return candidate


def _validate_actions(actions: np.ndarray, *, expected_steps: int) -> None:
    if actions.dtype != np.dtype(np.float32):
        raise RawEpisodeReadError("raw actions must retain dtype float32")
    if actions.shape != (expected_steps, 8):
        raise RawEpisodeReadError(
            f"raw actions must have shape ({expected_steps}, 8), got {actions.shape}"
        )
    if expected_steps < 1 or not np.all(np.isfinite(actions)):
        raise RawEpisodeReadError("raw actions must be non-empty and finite")


def _validate_state_tree(node: h5py.Group, *, expected_state_count: int) -> None:
    leaf_count = 0

    def visit(current: h5py.Group | h5py.Dataset) -> None:
        nonlocal leaf_count
        if isinstance(current, h5py.Group):
            if not current:
                raise RawEpisodeReadError(f"raw state group {current.name} must not be empty")
            for key in current:
                child = current[key]
                if not isinstance(child, (h5py.Group, h5py.Dataset)):
                    raise RawEpisodeReadError(f"raw state node {child.name} is unsupported")
                visit(child)
            return
        leaf_count += 1
        if current.ndim < 1 or current.shape[0] != expected_state_count:
            raise RawEpisodeReadError(
                f"raw state leaf {current.name} must contain exactly {expected_state_count} states"
            )
        if current.dtype.kind not in "biuf":
            raise RawEpisodeReadError(f"raw state leaf {current.name} must be numeric or boolean")
        if not np.all(np.isfinite(current[:])):
            raise RawEpisodeReadError(f"raw state leaf {current.name} contains non-finite values")

    visit(node)
    if leaf_count == 0:
        raise RawEpisodeReadError("raw env_states must contain at least one leaf")


def _read_state_at(node: h5py.Group | h5py.Dataset, index: int) -> Any:
    if isinstance(node, h5py.Group):
        return {key: _read_state_at(node[key], index) for key in node}
    value = np.array(node[index], copy=True)
    value.setflags(write=False)
    return value


def _copy_state(state: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, Mapping):
            result[key] = _copy_state(value)
        else:
            result[key] = np.array(value, copy=True)
    return result


def _require_state_index(index: object, *, state_count: int) -> None:
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("state index must be an integer")
    if not 0 <= index < state_count:
        raise IndexError(f"state index must be in [0, {state_count - 1}]")


__all__ = [
    "M3ASourceValidationError",
    "RawEpisodeData",
    "RawEpisodeReadError",
    "SourceExportMode",
    "ValidatedM3ASource",
    "ValidatedRawEpisodeReader",
    "validate_m3a_source_for_export",
]
