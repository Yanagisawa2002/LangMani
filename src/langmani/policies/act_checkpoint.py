"""Immutable, locally reloadable checkpoints for the M4 ACT baselines.

LeRobot owns serialization of the policy and processor pipelines.  LangMani
owns the semantic compatibility checks, optimizer/scheduler/RNG state, file
integrity manifest, staging directory, and atomic promotion into a run.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, cast

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from langmani.datasets.identity import sha256_hex
from langmani.policies.act_types import CheckpointRecord, TrainingState

CHECKPOINT_SCHEMA_VERSION = "langmani-m4-act-checkpoint-v1"
CHECKPOINTS_DIRECTORY = "checkpoints"
CHECKPOINT_STAGING_DIRECTORY = ".checkpoint-staging"
PRETRAINED_MODEL_DIRECTORY = "pretrained_model"
TRAINING_STATE_DIRECTORY = "training_state"
CHECKPOINT_MANIFEST = "checkpoint_manifest.json"
CHECKPOINT_COMPLETION_MARKER = "complete.json"
RUN_COMPLETION_MARKER = "complete.json"
PREPROCESSOR_CONFIG = "policy_preprocessor.json"
POSTPROCESSOR_CONFIG = "policy_postprocessor.json"
RNG_STATE_FILE = "rng_state.pt"


class ActCheckpointError(RuntimeError):
    """Raised when a checkpoint lifecycle or compatibility contract is violated."""


class _SavePretrained(Protocol):
    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        push_to_hub: bool = False,
        config_filename: str | None = None,
        **kwargs: object,
    ) -> object: ...


class _PolicyClass(Protocol):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        local_files_only: bool,
        strict: bool,
    ) -> object: ...


class ActCheckpointIdentity(Protocol):
    """Structural identity shared by historical M4 and isolated M4.2 runs."""

    run_fingerprint: str
    model_config: Mapping[str, object]
    train_statistics_fingerprint: str
    m3b_split_manifest_digest: str
    git_commit: str

    def to_dict(self) -> dict[str, object]: ...


ProcessorLoader = Callable[[object, Path], tuple[object, object]]


@dataclass(frozen=True, slots=True)
class CheckpointComponentFingerprints:
    """Content fingerprints for independently reloaded runtime components."""

    model: str
    preprocessor: str
    postprocessor: str


@dataclass(frozen=True, slots=True)
class LoadedActCheckpoint:
    """Compact result of one integrity-checked, local-only checkpoint load."""

    policy: object
    preprocessor: object
    postprocessor: object
    optimizer: Optimizer | None
    scheduler: LRScheduler | None
    training_state: TrainingState
    record: CheckpointRecord
    checkpoint_root: Path
    training_metric: Mapping[str, object] | None
    component_fingerprints: CheckpointComponentFingerprints


@dataclass(frozen=True, slots=True)
class LoadedActCheckpointManifest:
    """Integrity-checked checkpoint metadata without model deserialization.

    This binds the persisted training metric to the immutable checkpoint
    fingerprint.  It intentionally does not claim that the model artifacts
    have been reloaded; callers that need executable weights must still use
    :func:`load_act_checkpoint`.
    """

    training_state: TrainingState
    record: CheckpointRecord
    checkpoint_root: Path
    training_metric: Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class ValidatedActCheckpointArtifacts:
    """Content-validated immutable checkpoint without model deserialization."""

    training_state: TrainingState
    record: CheckpointRecord
    checkpoint_root: Path
    training_metric: Mapping[str, object] | None
    component_fingerprints: CheckpointComponentFingerprints


@dataclass(frozen=True, slots=True)
class _ValidatedCheckpointManifest:
    checkpoint_root: Path
    manifest: Mapping[str, object]
    training_state: TrainingState
    record: CheckpointRecord
    training_metric: Mapping[str, object] | None


def save_act_checkpoint(
    *,
    run_root: str | Path,
    identity: ActCheckpointIdentity,
    training_state: TrainingState,
    policy: _SavePretrained,
    preprocessor: _SavePretrained,
    postprocessor: _SavePretrained,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    training_metric: Mapping[str, object] | None = None,
) -> CheckpointRecord:
    """Save one content-bound checkpoint through staging and atomic promotion.

    A promoted checkpoint is immutable.  Its completion marker is written and
    fsynced in staging, so the single directory rename makes the complete
    checkpoint visible atomically.
    """
    if training_state.global_step < 1:
        raise ActCheckpointError("checkpoint global_step must be positive")
    root = Path(run_root).resolve()
    if (root / RUN_COMPLETION_MARKER).exists():
        raise ActCheckpointError("completed ACT runs are immutable and cannot accept checkpoints")
    checkpoints_root = root / CHECKPOINTS_DIRECTORY
    _reject_existing_completed_step(checkpoints_root, training_state.global_step)
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    staging_parent = root / CHECKPOINT_STAGING_DIRECTORY
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f"step-{training_state.global_step:08d}-", dir=staging_parent)
    ).resolve()
    _require_contained(staging_parent, staging_root)

    promoted = False
    try:
        pretrained_root = staging_root / PRETRAINED_MODEL_DIRECTORY
        pretrained_root.mkdir(parents=True, exist_ok=False)
        _save_public_artifacts(
            pretrained_root=pretrained_root,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
        )
        state_root = staging_root / TRAINING_STATE_DIRECTORY
        state_root.mkdir(parents=True, exist_ok=False)
        _save_optimizer_and_scheduler(state_root, optimizer, scheduler)
        _save_rng_state(state_root / RNG_STATE_FILE)

        artifact_records = _artifact_records(staging_root)
        artifact_digest = _prefixed_digest(artifact_records)
        policy_config_digest = _prefixed_digest(identity.model_config)
        checkpoint_fingerprint = _checkpoint_fingerprint(
            identity=identity,
            training_state=training_state,
            artifact_digest=artifact_digest,
            policy_config_digest=policy_config_digest,
            training_metric=training_metric,
        )
        relative_path = (
            PurePosixPath(CHECKPOINTS_DIRECTORY)
            / (
                f"step-{training_state.global_step:08d}-"
                f"{checkpoint_fingerprint.removeprefix('sha256:')[:12]}"
            )
        ).as_posix()
        destination = _safe_relative_path(root, relative_path)
        if destination.exists():
            raise ActCheckpointError(
                f"completed checkpoint destination already exists: {destination}"
            )
        record = CheckpointRecord(
            checkpoint_fingerprint=checkpoint_fingerprint,
            run_fingerprint=identity.run_fingerprint,
            global_step=training_state.global_step,
            relative_path=relative_path,
            policy_config_digest=policy_config_digest,
            statistics_fingerprint=identity.train_statistics_fingerprint,
            split_digest=identity.m3b_split_manifest_digest,
            git_commit=identity.git_commit,
            complete=True,
        )
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "identity": identity.to_dict(),
            "training_state": training_state.to_dict(),
            "record": record.to_dict(),
            "scheduler_saved": scheduler is not None,
            "training_metric": dict(training_metric) if training_metric is not None else None,
            "artifact_digest": artifact_digest,
            "artifacts": artifact_records,
        }
        manifest_path = staging_root / CHECKPOINT_MANIFEST
        _write_new_json(manifest_path, manifest)
        manifest_digest = _sha256_file(manifest_path)
        _write_new_json(
            staging_root / CHECKPOINT_COMPLETION_MARKER,
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "checkpoint_fingerprint": checkpoint_fingerprint,
                "manifest_sha256": manifest_digest,
            },
        )
        _fsync_tree(staging_root)

        destination.parent.mkdir(parents=True, exist_ok=True)
        if staging_root.stat().st_dev != destination.parent.stat().st_dev:
            raise ActCheckpointError(
                "checkpoint staging and destination are on different filesystems"
            )
        try:
            os.replace(staging_root, destination)
        except OSError as error:
            raise ActCheckpointError(f"atomic checkpoint promotion failed: {error}") from error
        promoted = True
        _fsync_directory(destination.parent)
        return record
    finally:
        if not promoted and staging_root.exists():
            _require_contained(staging_parent, staging_root)
            shutil.rmtree(staging_root)


def _load_validated_checkpoint_manifest(
    *,
    run_root: str | Path,
    checkpoint_relative_path: str,
    expected_identity: ActCheckpointIdentity,
) -> _ValidatedCheckpointManifest:
    root = Path(run_root).resolve()
    checkpoint_root = _safe_relative_path(root, checkpoint_relative_path)
    if not checkpoint_root.is_dir() or checkpoint_root.is_symlink():
        raise ActCheckpointError(f"checkpoint directory is missing or unsafe: {checkpoint_root}")
    marker_path = checkpoint_root / CHECKPOINT_COMPLETION_MARKER
    manifest_path = checkpoint_root / CHECKPOINT_MANIFEST
    marker = _read_object(marker_path, "checkpoint completion marker")
    manifest = _read_object(manifest_path, "checkpoint manifest")
    if marker.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ActCheckpointError("unsupported checkpoint completion schema")
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ActCheckpointError("unsupported checkpoint manifest schema")
    if marker.get("manifest_sha256") != _sha256_file(manifest_path):
        raise ActCheckpointError("checkpoint manifest checksum mismatch")

    actual_identity = manifest.get("identity")
    if actual_identity != expected_identity.to_dict():
        raise ActCheckpointError("checkpoint semantic identity does not match the requested run")
    training_state = _parse_training_state(manifest.get("training_state"))
    record = _parse_record(manifest.get("record"))
    training_metric = manifest.get("training_metric")
    if training_metric is not None and not isinstance(training_metric, dict):
        raise ActCheckpointError("checkpoint training metric is malformed")
    normalized_relative_path = _normalize_relative_path(checkpoint_relative_path)
    if record.relative_path != normalized_relative_path:
        raise ActCheckpointError("checkpoint record path does not match the requested path")
    _validate_record(record, expected_identity, training_state, manifest)
    if marker.get("checkpoint_fingerprint") != record.checkpoint_fingerprint:
        raise ActCheckpointError("completion marker checkpoint fingerprint mismatch")
    return _ValidatedCheckpointManifest(
        checkpoint_root=checkpoint_root,
        manifest=manifest,
        training_state=training_state,
        record=record,
        training_metric=training_metric,
    )


def load_act_checkpoint_manifest(
    *,
    run_root: str | Path,
    checkpoint_relative_path: str,
    expected_identity: ActCheckpointIdentity,
) -> LoadedActCheckpointManifest:
    """Load fingerprint-bound metadata without deserializing model artifacts."""
    validated = _load_validated_checkpoint_manifest(
        run_root=run_root,
        checkpoint_relative_path=checkpoint_relative_path,
        expected_identity=expected_identity,
    )
    return LoadedActCheckpointManifest(
        training_state=validated.training_state,
        record=validated.record,
        checkpoint_root=validated.checkpoint_root,
        training_metric=validated.training_metric,
    )


def validate_act_checkpoint_artifacts(
    *,
    run_root: str | Path,
    checkpoint_relative_path: str,
    expected_identity: ActCheckpointIdentity,
) -> ValidatedActCheckpointArtifacts:
    """Validate marker, manifest and every persisted artifact without loading weights."""

    validated = _load_validated_checkpoint_manifest(
        run_root=run_root,
        checkpoint_relative_path=checkpoint_relative_path,
        expected_identity=expected_identity,
    )
    _validate_artifacts(validated.checkpoint_root, validated.manifest)
    return ValidatedActCheckpointArtifacts(
        training_state=validated.training_state,
        record=validated.record,
        checkpoint_root=validated.checkpoint_root,
        training_metric=validated.training_metric,
        component_fingerprints=_component_fingerprints(validated.manifest),
    )


def load_act_checkpoint(
    *,
    run_root: str | Path,
    checkpoint_relative_path: str,
    expected_identity: ActCheckpointIdentity,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    for_resume: bool = False,
    restore_rng: bool | None = None,
    policy_class: type[_PolicyClass] | None = None,
    processor_loader: ProcessorLoader | None = None,
    existing_policy: object | None = None,
) -> LoadedActCheckpoint:
    """Integrity-check and reload a checkpoint without contacting the Hub."""
    root = Path(run_root).resolve()
    validated = _load_validated_checkpoint_manifest(
        run_root=root,
        checkpoint_relative_path=checkpoint_relative_path,
        expected_identity=expected_identity,
    )
    checkpoint_root = validated.checkpoint_root
    manifest = validated.manifest
    training_state = validated.training_state
    record = validated.record
    training_metric = validated.training_metric
    _validate_artifacts(checkpoint_root, manifest)
    component_fingerprints = _component_fingerprints(manifest)

    should_restore_rng = for_resume if restore_rng is None else restore_rng
    if for_resume:
        if (root / RUN_COMPLETION_MARKER).exists() or training_state.completed:
            raise ActCheckpointError("completed training state is immutable and cannot be resumed")
        if optimizer is None:
            raise ActCheckpointError("resume requires a compatible optimizer instance")
        saved_scheduler = manifest.get("scheduler_saved")
        if not isinstance(saved_scheduler, bool):
            raise ActCheckpointError("checkpoint scheduler flag is malformed")
        if saved_scheduler != (scheduler is not None):
            raise ActCheckpointError("resume scheduler state is incompatible with the checkpoint")

    pretrained_root = checkpoint_root / PRETRAINED_MODEL_DIRECTORY
    chosen_policy_class = policy_class or _installed_act_policy_class()
    try:
        reloaded_policy = chosen_policy_class.from_pretrained(
            pretrained_root,
            local_files_only=True,
            strict=True,
        )
        if existing_policy is None:
            policy = reloaded_policy
        else:
            load_state_dict = getattr(existing_policy, "load_state_dict", None)
            state_dict = getattr(reloaded_policy, "state_dict", None)
            if not callable(load_state_dict) or not callable(state_dict):
                raise TypeError("existing policy must support strict state_dict loading")
            load_state_dict(state_dict(), strict=True)
            policy = existing_policy
    except Exception as error:
        raise ActCheckpointError(
            f"strict local policy reload failed: {type(error).__name__}: {error}"
        ) from error
    chosen_processor_loader = processor_loader or _load_installed_processors
    try:
        loaded_preprocessor, loaded_postprocessor = chosen_processor_loader(policy, pretrained_root)
    except Exception as error:
        raise ActCheckpointError(
            f"local processor reload failed: {type(error).__name__}: {error}"
        ) from error

    loaded_optimizer = optimizer
    loaded_scheduler = scheduler
    if for_resume:
        state_root = checkpoint_root / TRAINING_STATE_DIRECTORY
        try:
            from lerobot.optim import load_optimizer_state, load_scheduler_state

            loaded_optimizer = load_optimizer_state(optimizer, state_root)
            if not isinstance(loaded_optimizer, Optimizer):
                raise TypeError("M4 ACT expects one optimizer, not an optimizer mapping")
            if loaded_scheduler is not None:
                loaded_scheduler = load_scheduler_state(loaded_scheduler, state_root)
        except Exception as error:
            raise ActCheckpointError(
                f"training state reload failed: {type(error).__name__}: {error}"
            ) from error
        if should_restore_rng:
            _restore_rng_state(state_root / RNG_STATE_FILE)
    elif should_restore_rng:
        raise ActCheckpointError("RNG restoration is permitted only for resume")

    return LoadedActCheckpoint(
        policy=policy,
        preprocessor=loaded_preprocessor,
        postprocessor=loaded_postprocessor,
        optimizer=loaded_optimizer,
        scheduler=loaded_scheduler,
        training_state=training_state,
        record=record,
        checkpoint_root=checkpoint_root,
        training_metric=training_metric,
        component_fingerprints=component_fingerprints,
    )


def assert_resume_compatible(
    expected_identity: ActCheckpointIdentity,
    checkpoint_identity: Mapping[str, object],
) -> None:
    """Reject any changed semantic input before restoring mutable training state."""
    if dict(checkpoint_identity) != expected_identity.to_dict():
        raise ActCheckpointError("resume identity mismatch")


def _save_public_artifacts(
    *,
    pretrained_root: Path,
    policy: _SavePretrained,
    preprocessor: _SavePretrained,
    postprocessor: _SavePretrained,
) -> None:
    try:
        policy.save_pretrained(pretrained_root, push_to_hub=False)
        preprocessor.save_pretrained(
            pretrained_root,
            push_to_hub=False,
            config_filename=PREPROCESSOR_CONFIG,
        )
        postprocessor.save_pretrained(
            pretrained_root,
            push_to_hub=False,
            config_filename=POSTPROCESSOR_CONFIG,
        )
    except Exception as error:
        raise ActCheckpointError(
            f"public policy/processor save failed: {type(error).__name__}: {error}"
        ) from error


def _save_optimizer_and_scheduler(
    state_root: Path,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
) -> None:
    try:
        from lerobot.optim import save_optimizer_state, save_scheduler_state

        save_optimizer_state(optimizer, state_root)
        if scheduler is not None:
            save_scheduler_state(scheduler, state_root)
    except Exception as error:
        raise ActCheckpointError(
            f"optimizer/scheduler save failed: {type(error).__name__}: {error}"
        ) from error


def _save_rng_state(path: Path) -> None:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    try:
        torch.save(state, path)
    except (OSError, RuntimeError, TypeError) as error:
        raise ActCheckpointError(f"RNG state save failed: {error}") from error


def _restore_rng_state(path: Path) -> None:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            raise TypeError("RNG state payload must be a mapping")
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        cuda_states = state["torch_cuda"]
        if cuda_states:
            if not torch.cuda.is_available():
                raise ActCheckpointError(
                    "checkpoint contains CUDA RNG state but CUDA is unavailable"
                )
            torch.cuda.set_rng_state_all(cuda_states)
    except ActCheckpointError:
        raise
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise ActCheckpointError(f"RNG state restore failed: {error}") from error


def _installed_act_policy_class() -> type[Any]:
    try:
        from lerobot.policies.act import ACTPolicy
    except (ImportError, RuntimeError, OSError) as error:
        raise ActCheckpointError("installed LeRobot ACTPolicy is unavailable") from error
    return ACTPolicy


def _load_installed_processors(policy: object, pretrained_root: Path) -> tuple[object, object]:
    try:
        from lerobot.policies import make_pre_post_processors

        config = policy.config
        return make_pre_post_processors(policy_cfg=config, pretrained_path=str(pretrained_root))
    except (AttributeError, ImportError, RuntimeError, OSError, ValueError) as error:
        raise ActCheckpointError(
            "installed LeRobot processors could not be loaded locally"
        ) from error


def _checkpoint_fingerprint(
    *,
    identity: ActCheckpointIdentity,
    training_state: TrainingState,
    artifact_digest: str,
    policy_config_digest: str,
    training_metric: Mapping[str, object] | None,
) -> str:
    return _prefixed_digest(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_fingerprint": identity.run_fingerprint,
            "training_state": training_state.to_dict(),
            "artifact_digest": artifact_digest,
            "policy_config_digest": policy_config_digest,
            "training_metric": dict(training_metric) if training_metric is not None else None,
            "statistics_fingerprint": identity.train_statistics_fingerprint,
            "split_digest": identity.m3b_split_manifest_digest,
            "git_commit": identity.git_commit,
        }
    )


def _validate_record(
    record: CheckpointRecord,
    identity: ActCheckpointIdentity,
    training_state: TrainingState,
    manifest: Mapping[str, object],
) -> None:
    expected_policy_digest = _prefixed_digest(identity.model_config)
    if not record.complete:
        raise ActCheckpointError("checkpoint record is not complete")
    if record.global_step != training_state.global_step:
        raise ActCheckpointError("checkpoint step disagrees with training state")
    training_metric = manifest.get("training_metric")
    if isinstance(training_metric, Mapping) and training_metric.get("step") != record.global_step:
        raise ActCheckpointError("checkpoint training metric step is inconsistent")
    expected_fields = (
        (record.run_fingerprint, identity.run_fingerprint, "run fingerprint"),
        (record.policy_config_digest, expected_policy_digest, "policy config digest"),
        (
            record.statistics_fingerprint,
            identity.train_statistics_fingerprint,
            "statistics fingerprint",
        ),
        (record.split_digest, identity.m3b_split_manifest_digest, "split digest"),
        (record.git_commit, identity.git_commit, "Git commit"),
    )
    for actual, expected, name in expected_fields:
        if actual != expected:
            raise ActCheckpointError(f"checkpoint {name} mismatch")
    artifact_digest = manifest.get("artifact_digest")
    if not isinstance(artifact_digest, str):
        raise ActCheckpointError("checkpoint artifact digest is malformed")
    expected_checkpoint = _checkpoint_fingerprint(
        identity=identity,
        training_state=training_state,
        artifact_digest=artifact_digest,
        policy_config_digest=expected_policy_digest,
        training_metric=(
            manifest["training_metric"]
            if isinstance(manifest.get("training_metric"), Mapping)
            else None
        ),
    )
    if record.checkpoint_fingerprint != expected_checkpoint:
        raise ActCheckpointError("checkpoint fingerprint does not bind the manifest contents")


def _parse_training_state(value: object) -> TrainingState:
    if not isinstance(value, dict):
        raise ActCheckpointError("checkpoint training state is malformed")
    try:
        return TrainingState(**value)
    except (TypeError, ValueError) as error:
        raise ActCheckpointError(f"invalid checkpoint training state: {error}") from error


def _parse_record(value: object) -> CheckpointRecord:
    if not isinstance(value, dict):
        raise ActCheckpointError("checkpoint record is malformed")
    try:
        return CheckpointRecord(**value)
    except (TypeError, ValueError) as error:
        raise ActCheckpointError(f"invalid checkpoint record: {error}") from error


def _artifact_records(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ActCheckpointError(f"checkpoint artifacts must not contain symlinks: {path}")
        if not path.is_file():
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    if not records:
        raise ActCheckpointError("checkpoint contains no serialized artifacts")
    return records


def _validate_artifacts(root: Path, manifest: Mapping[str, object]) -> None:
    raw_records = manifest.get("artifacts")
    if not isinstance(raw_records, list) or not raw_records:
        raise ActCheckpointError("checkpoint artifact list is malformed")
    normalized: list[dict[str, object]] = []
    expected_paths: set[str] = set()
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise ActCheckpointError("checkpoint artifact entry is malformed")
        relative = raw.get("path")
        digest = raw.get("sha256")
        size = raw.get("size_bytes")
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or not isinstance(size, int)
        ):
            raise ActCheckpointError("checkpoint artifact fields are malformed")
        normalized_relative = _normalize_relative_path(relative)
        if normalized_relative in expected_paths:
            raise ActCheckpointError("checkpoint artifact list contains duplicate paths")
        expected_paths.add(normalized_relative)
        path = _safe_relative_path(root, normalized_relative)
        if not path.is_file() or path.is_symlink():
            raise ActCheckpointError(f"checkpoint artifact is missing or unsafe: {relative}")
        if path.stat().st_size != size or _sha256_file(path) != digest:
            raise ActCheckpointError(f"checkpoint artifact integrity mismatch: {relative}")
        normalized.append({"path": normalized_relative, "sha256": digest, "size_bytes": size})
    if manifest.get("artifact_digest") != _prefixed_digest(normalized):
        raise ActCheckpointError("checkpoint aggregate artifact digest mismatch")
    discovered = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    allowed = expected_paths | {CHECKPOINT_MANIFEST, CHECKPOINT_COMPLETION_MARKER}
    if discovered != allowed:
        raise ActCheckpointError("checkpoint contains unmanifested or missing files")


def _component_fingerprints(
    manifest: Mapping[str, object],
) -> CheckpointComponentFingerprints:
    raw_records = manifest.get("artifacts")
    if not isinstance(raw_records, list):
        raise ActCheckpointError("checkpoint artifact list is malformed")
    groups: dict[str, list[dict[str, object]]] = {
        "model": [],
        "preprocessor": [],
        "postprocessor": [],
    }
    for raw in raw_records:
        if not isinstance(raw, dict) or not isinstance(raw.get("path"), str):
            raise ActCheckpointError("checkpoint artifact entry is malformed")
        path = cast(str, raw["path"])
        name = PurePosixPath(path).name
        if name == PREPROCESSOR_CONFIG or name.startswith("policy_preprocessor_step_"):
            group = "preprocessor"
        elif name == POSTPROCESSOR_CONFIG or name.startswith("policy_postprocessor_step_"):
            group = "postprocessor"
        elif path.startswith(f"{PRETRAINED_MODEL_DIRECTORY}/"):
            group = "model"
        else:
            continue
        groups[group].append(dict(raw))
    if any(not records for records in groups.values()):
        raise ActCheckpointError("checkpoint lacks model or policy processor artifacts")
    return CheckpointComponentFingerprints(
        model=_prefixed_digest(groups["model"]),
        preprocessor=_prefixed_digest(groups["preprocessor"]),
        postprocessor=_prefixed_digest(groups["postprocessor"]),
    )


def _reject_existing_completed_step(checkpoints_root: Path, global_step: int) -> None:
    if not checkpoints_root.is_dir():
        return
    if any(checkpoints_root.glob(f"step-{global_step:08d}-*")):
        raise ActCheckpointError(
            f"checkpoint step {global_step} already exists or has an interrupted promotion"
        )


def _safe_relative_path(root: Path, relative_path: str) -> Path:
    normalized = _normalize_relative_path(relative_path)
    result = (root / Path(*PurePosixPath(normalized).parts)).resolve()
    _require_contained(root, result)
    return result


def _normalize_relative_path(relative_path: str) -> str:
    if not isinstance(relative_path, str) or not relative_path:
        raise ActCheckpointError("checkpoint path must be a non-empty relative path")
    if "\\" in relative_path:
        raise ActCheckpointError("checkpoint path must use portable POSIX separators")
    value = PurePosixPath(relative_path)
    if value.is_absolute() or any(part in {"", ".", ".."} for part in value.parts):
        raise ActCheckpointError("checkpoint path escapes its run root")
    if any(re.fullmatch(r"[A-Za-z0-9._-]+", part) is None for part in value.parts):
        raise ActCheckpointError("checkpoint path contains a non-portable component")
    normalized = value.as_posix()
    if normalized != relative_path:
        raise ActCheckpointError("checkpoint path is not canonically normalized")
    return normalized


def _require_contained(root: Path, child: Path) -> None:
    root_resolved = root.resolve()
    child_resolved = child.resolve()
    if child_resolved == root_resolved or root_resolved not in child_resolved.parents:
        raise ActCheckpointError(f"path is outside the owned root: {child_resolved}")


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ActCheckpointError(f"could not read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ActCheckpointError(f"{label} must contain one JSON object")
    return value


def _write_new_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise ActCheckpointError(f"refusing to overwrite immutable file: {path}") from error


def _fsync_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            try:
                # Windows requires a writable handle for FlushFileBuffers, which
                # is what ``os.fsync`` delegates to there.
                with path.open("r+b") as stream:
                    os.fsync(stream.fileno())
            except OSError as error:
                raise ActCheckpointError(
                    f"failed to flush checkpoint artifact {path}: {error}"
                ) from error


def _fsync_directory(path: Path) -> None:
    """Persist the atomic rename on POSIX; Windows has no directory fsync handle."""
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ActCheckpointError(f"could not hash checkpoint file {path}: {error}") from error
    return digest.hexdigest()


def _prefixed_digest(value: object) -> str:
    return f"sha256:{sha256_hex(value)}"


__all__ = [
    "CHECKPOINT_COMPLETION_MARKER",
    "CHECKPOINT_MANIFEST",
    "CHECKPOINT_SCHEMA_VERSION",
    "ActCheckpointIdentity",
    "ActCheckpointError",
    "CheckpointComponentFingerprints",
    "LoadedActCheckpoint",
    "LoadedActCheckpointManifest",
    "ValidatedActCheckpointArtifacts",
    "assert_resume_compatible",
    "load_act_checkpoint",
    "load_act_checkpoint_manifest",
    "save_act_checkpoint",
    "validate_act_checkpoint_artifacts",
]
