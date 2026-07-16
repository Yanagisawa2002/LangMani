"""Restricted read-only input gate for the M4.3a semantic audit.

This module intentionally does not reuse ``load_prior_m4_evidence`` because
that M4.2 gate opens historical test-runtime evidence.  M4.3a is narrower: it
may read one M3B completion marker, validation-only run/selection metadata,
the TaskToken validation-only selection, and completed ``m42_dev_v0``
evidence.  Every JSON read is authorized by an explicit relative-path
allowlist and recorded in a portable checksum trace.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, cast

from langmani.datasets.identity import sha256_hex
from langmani.datasets.lerobot_writer import COMPLETION_MARKER
from langmani.environments.specs import TaskSpec, stable_task_id
from langmani.policies.act_conditioning import CANONICAL_TASK_IDS
from langmani.policies.act_data import (
    COMPLETION_SCHEMA,
    FeatureStatistics,
    StatisticsLeakageAudit,
    TrainOnlyStatistics,
)
from langmani.policies.act_evaluation import CheckpointSelectionRecord
from langmani.policies.act_types import (
    ActExperimentManifest,
    ActVariant,
    CheckpointRecord,
    ExperimentMode,
)
from langmani.policies.m42_analysis import M42AnalysisError, create_task_token_checkpoint_selection
from langmani.policies.m42_evaluation import M42PolicyKind
from langmani.policies.m42_schedule import M42_MAXIMUM_EPISODE_STEPS
from langmani.policies.m42_training import (
    RUNTIME_SELECTION_SCHEMA,
    M42TrainingContractError,
    RuntimeSelectionContract,
    TaskTokenTrainingManifest,
    TaskTokenValidationQueue,
    TaskTokenValidationQueueItem,
    canonical_fingerprint,
)
from langmani.policies.m42_types import (
    M42SelectionKind,
    M42SelectionRecord,
    TaskTokenValidationResult,
)

M43_RESTRICTED_INPUT_SCHEMA = "langmani-m43-restricted-audit-inputs-v0"
M42_DEV_SCHEDULE_ID = "m42_dev_v0"
TASK_TOKEN_SELECTION_SCHEMA = "langmani-m42-task-token-selection-artifact-v0"
DEVELOPMENT_COMPARISON_SCHEMA = "langmani-m42-development-comparison-v0"
DEVELOPMENT_COMPLETION_SCHEMA = "langmani-m42-development-complete-v0"
M42_EVALUATION_ARTIFACT_SCHEMA = "langmani-m42-evaluation-artifact-v0"
M43_AUDIT_MODES = ("validation", "m42_dev_v0", "combined")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
_M42_EVALUATION_IMPLEMENTATION_SCHEMA = "langmani-m42-final-implementation-v0"
_M42_EVALUATION_IMPLEMENTATION_FILES = (
    "src/langmani/policies/m42_evaluation.py",
    "src/langmani/policies/m42_analysis.py",
    "src/langmani/policies/m42_runtime.py",
    "src/langmani/policies/m42_schedule.py",
    "scripts/evaluate_m42.py",
)

_RUN_DIRECTORY = re.compile(r"[0-9a-f]{64}")
_ACCESS_FALSE_KEYS = frozenset(
    {
        "test_split_accessed",
        "fresh_seed_accessed",
        "fresh_seed_schedule_accessed",
        "final_schedule_accessed",
        "m42_final_schedule_accessed",
    }
)


class M43AuditInputError(RuntimeError):
    """Raised when an audit input escapes the validation/development boundary."""


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("restricted audit JSON mappings require string keys")
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"unsupported restricted audit JSON value {type(value).__name__}")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_thaw_json(item) for item in value]
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise M43AuditInputError(f"{label} must be a prefixed SHA-256 digest")
    return value


def _portable_relative(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise M43AuditInputError(f"{label} must use a portable relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise M43AuditInputError(f"{label} must be a canonical relative path")
    _reject_forbidden_path_parts(path.parts, label=label)
    return value


def _normalized(value: str) -> str:
    return value.strip().casefold().replace("-", "_")


def _forbidden_identity(value: str) -> bool:
    normalized = _normalized(value)
    return (
        normalized == "test"
        or normalized == "m3b_test"
        or "fresh_seed" in normalized
        or normalized == "final"
        or normalized.startswith("m42_final")
        or normalized.startswith("final_attempt")
    )


def _reject_forbidden_path_parts(parts: Sequence[str], *, label: str) -> None:
    if any(_forbidden_identity(part) for part in parts):
        raise M43AuditInputError(f"{label} names a prohibited test, fresh-seed, or final path")


def _assert_development_only(value: object, *, label: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise M43AuditInputError(f"{label} contains a non-string JSON key")
            key = _normalized(raw_key)
            if key in _ACCESS_FALSE_KEYS:
                if item is not False:
                    raise M43AuditInputError(f"{label} requires {raw_key}=false")
                continue
            if _forbidden_identity(raw_key):
                raise M43AuditInputError(f"{label} contains a prohibited evidence section")
            _assert_development_only(item, label=label)
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for item in value:
            _assert_development_only(item, label=label)
        return
    if isinstance(value, str) and _forbidden_identity(value):
        raise M43AuditInputError(f"{label} references a prohibited evidence identity")


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _resolved_unlinked(path: Path, *, label: str, directory: bool) -> Path:
    lexical = _lexical_absolute(path)
    for component in (lexical, *lexical.parents):
        if component.is_symlink() or component.is_junction():
            raise M43AuditInputError(f"{label} traverses a symlink or junction: {component}")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise M43AuditInputError(f"missing or inaccessible {label}: {lexical}") from error
    if (directory and not resolved.is_dir()) or (not directory and not resolved.is_file()):
        kind = "directory" if directory else "file"
        raise M43AuditInputError(f"{label} must be a real {kind}: {resolved}")
    return resolved


def _contained(path: Path, root: Path, *, label: str) -> tuple[str, ...]:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise M43AuditInputError(f"{label} escapes its declared root: {path}") from error
    if path == root:
        raise M43AuditInputError(f"{label} must be below its declared root")
    return relative.parts


@dataclass(frozen=True, slots=True)
class AuditInputAccess:
    authority: str
    relative_path: str
    sha256: str

    def __post_init__(self) -> None:
        if self.authority not in {
            "m3b_dataset",
            "m4_checkpoints",
            "task_token_checkpoints",
            "m42_checkpoint_provenance",
            "m42_development_rollout",
        }:
            raise M43AuditInputError("unknown restricted audit input authority")
        _portable_relative(self.relative_path, "access trace path")
        _digest(self.sha256, "access trace sha256")

    def to_dict(self) -> dict[str, str]:
        return {
            "authority": self.authority,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
        }


class _RestrictedReader:
    def __init__(
        self,
        *,
        dataset_root: Path,
        m4_checkpoint_root: Path,
        task_token_checkpoint_root: Path,
        m42_diagnostics_root: Path,
        runtime_selection_path: Path,
    ) -> None:
        self.dataset_root = _resolved_unlinked(
            dataset_root, label="M3B dataset root", directory=True
        )
        self.m4_checkpoint_root = _resolved_unlinked(
            m4_checkpoint_root, label="M4 checkpoint root", directory=True
        )
        self.task_token_checkpoint_root = _resolved_unlinked(
            task_token_checkpoint_root,
            label="TaskToken checkpoint root",
            directory=True,
        )
        self.m42_diagnostics_root = _resolved_unlinked(
            m42_diagnostics_root, label="M4.2 diagnostics root", directory=True
        )
        self.runtime_selection_path = _resolved_unlinked(
            runtime_selection_path, label="runtime selection", directory=False
        )
        runtime_relative = _contained(
            self.runtime_selection_path,
            self.m42_diagnostics_root,
            label="runtime selection",
        )
        if runtime_relative != ("runtime_ablation", "runtime_selection.json"):
            raise M43AuditInputError(
                "runtime selection must be the development runtime_ablation lock"
            )
        self._trace: list[AuditInputAccess] = []

    @property
    def trace(self) -> tuple[AuditInputAccess, ...]:
        return tuple(self._trace)

    def _authorize(self, path: Path) -> tuple[str, str, Path]:
        lexical = _lexical_absolute(path)
        for component in (lexical, *lexical.parents):
            if component.is_symlink() or component.is_junction():
                raise M43AuditInputError(
                    f"restricted audit input traverses a symlink or junction: {component}"
                )
        resolved = lexical.resolve(strict=True)
        authorities = (
            ("m3b_dataset", self.dataset_root),
            ("m4_checkpoints", self.m4_checkpoint_root),
            ("task_token_checkpoints", self.task_token_checkpoint_root),
            ("m42_development", self.m42_diagnostics_root),
        )
        for authority, root in authorities:
            try:
                relative = resolved.relative_to(root)
            except ValueError:
                continue
            parts = relative.parts
            _reject_forbidden_path_parts(parts, label=f"{authority} input")
            allowed = False
            if authority == "m3b_dataset":
                allowed = parts == COMPLETION_MARKER.parts
            elif authority == "m4_checkpoints":
                allowed = (
                    len(parts) == 2
                    and _RUN_DIRECTORY.fullmatch(parts[0]) is not None
                    and parts[1]
                    in {"run_manifest.json", "checkpoint_selection.json", "train_stats.json"}
                )
            elif authority == "task_token_checkpoints":
                allowed = (
                    len(parts) == 2
                    and _RUN_DIRECTORY.fullmatch(parts[0]) is not None
                    and parts[1]
                    in {"run_manifest.json", "train_stats.json", "validation_queue.json"}
                )
            else:
                allowed = parts in {
                    ("runtime_ablation", "runtime_selection.json"),
                    ("development", "task_token_checkpoint_selection.json"),
                    ("development", "development_comparison.json"),
                    ("development", "development_complete.json"),
                } or (
                    len(parts) == 4
                    and parts[:2] == ("development", "evidence")
                    and _RUN_DIRECTORY.fullmatch(parts[2]) is not None
                    and parts[3] in {"artifact.json", "complete.json"}
                )
            if not allowed:
                raise M43AuditInputError(
                    f"read is outside the M4.3a {authority} allowlist: {relative.as_posix()}"
                )
            if not resolved.is_file():
                raise M43AuditInputError(f"restricted audit input is not a file: {resolved}")
            if authority == "m42_development":
                authority = (
                    "m42_development_rollout"
                    if parts
                    in {
                        ("development", "development_comparison.json"),
                        ("development", "development_complete.json"),
                    }
                    else "m42_checkpoint_provenance"
                )
            return authority, relative.as_posix(), resolved
        raise M43AuditInputError(f"read is outside all restricted audit roots: {resolved}")

    def read_json(self, path: Path, *, label: str) -> dict[str, Any]:
        authority, relative, resolved = self._authorize(path)
        try:
            before = os.stat(resolved, follow_symlinks=False)
            flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(resolved, flags)
            try:
                opened = os.fstat(descriptor)
                with os.fdopen(descriptor, "rb", closefd=False) as stream:
                    payload = stream.read()
            finally:
                os.close(descriptor)
            after = os.stat(resolved, follow_symlinks=False)
            before_path_identity = (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            after_path_identity = (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            opened_identity = (
                opened.st_dev,
                opened.st_ino,
                opened.st_mode,
                opened.st_size,
                opened.st_mtime_ns,
            )
            if (
                before_path_identity != after_path_identity
                or before_path_identity[:-1] != opened_identity
                or not stat.S_ISREG(opened.st_mode)
            ):
                raise M43AuditInputError(
                    f"restricted audit input changed during its authorized read: {resolved}"
                )
            value = json.loads(payload.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise M43AuditInputError(f"cannot read {label}: {resolved}") from error
        if not isinstance(value, dict):
            raise M43AuditInputError(f"{label} must contain one JSON object")
        checksum = f"sha256:{hashlib.sha256(payload).hexdigest()}"
        self._trace.append(
            AuditInputAccess(authority=authority, relative_path=relative, sha256=checksum)
        )
        return value


@dataclass(frozen=True, slots=True)
class ValidationEpisodeInput:
    episode_index: int
    scene_seed: int
    scene_group_id: str
    task_id: str
    task_spec: TaskSpec

    def __post_init__(self) -> None:
        if (
            isinstance(self.episode_index, bool)
            or not isinstance(self.episode_index, int)
            or self.episode_index < 0
        ):
            raise M43AuditInputError("validation episode_index must be non-negative")
        if isinstance(self.scene_seed, bool) or not isinstance(self.scene_seed, int):
            raise M43AuditInputError("validation scene_seed must be an integer")
        if not self.scene_group_id:
            raise M43AuditInputError("validation scene_group_id must be non-empty")
        if self.task_id != stable_task_id(self.task_spec):
            raise M43AuditInputError("validation task ID differs from its TaskSpec")

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_index": self.episode_index,
            "scene_seed": self.scene_seed,
            "scene_group_id": self.scene_group_id,
            "task_id": self.task_id,
            "task_spec": self.task_spec.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RestrictedCheckpointInput:
    policy_kind: M42PolicyKind
    task_id: str | None
    run_root: Path
    run_fingerprint: str
    checkpoint_relative_path: str
    checkpoint_fingerprint: str
    selection_fingerprint: str
    statistics_fingerprint: str
    dataset_fingerprint: str
    split_digest: str
    architecture_fingerprint: str | None
    git_commit: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_kind", M42PolicyKind(self.policy_kind))
        if self.policy_kind is M42PolicyKind.PER_TASK:
            if self.task_id not in CANONICAL_TASK_IDS:
                raise M43AuditInputError("PerTask checkpoint requires a canonical task ID")
        elif self.task_id is not None:
            raise M43AuditInputError("shared checkpoints cannot bind one TaskSpec")
        for name in (
            "run_fingerprint",
            "checkpoint_fingerprint",
            "selection_fingerprint",
            "statistics_fingerprint",
            "dataset_fingerprint",
            "split_digest",
        ):
            _digest(cast(str, getattr(self, name)), name)
        if self.architecture_fingerprint is not None:
            _digest(self.architecture_fingerprint, "architecture_fingerprint")
        if re.fullmatch(r"[0-9a-f]{40}", self.git_commit) is None:
            raise M43AuditInputError("checkpoint git_commit must be a full lowercase commit")
        _portable_relative(self.checkpoint_relative_path, "checkpoint relative path")

    @property
    def checkpoint_path(self) -> Path:
        return self.run_root.joinpath(*PurePosixPath(self.checkpoint_relative_path).parts)

    def portable_dict(self) -> dict[str, object]:
        return {
            "policy_kind": self.policy_kind.value,
            "task_id": self.task_id,
            "run_fingerprint": self.run_fingerprint,
            "checkpoint_relative_path": self.checkpoint_relative_path,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "selection_fingerprint": self.selection_fingerprint,
            "statistics_fingerprint": self.statistics_fingerprint,
            "dataset_fingerprint": self.dataset_fingerprint,
            "split_digest": self.split_digest,
            "architecture_fingerprint": self.architecture_fingerprint,
            "git_commit": self.git_commit,
        }


@dataclass(frozen=True, slots=True)
class M42DevelopmentInputs:
    runtime: RuntimeSelectionContract
    task_token_selection: M42SelectionRecord
    comparison: Mapping[str, object]
    completion: Mapping[str, object]
    task_token_m42_status: str = "rejected"

    def __post_init__(self) -> None:
        if self.task_token_m42_status != "rejected":
            raise M43AuditInputError("TaskToken must remain rejected by M4.2")
        object.__setattr__(
            self, "comparison", cast(Mapping[str, object], _freeze_json(self.comparison))
        )
        object.__setattr__(
            self, "completion", cast(Mapping[str, object], _freeze_json(self.completion))
        )

    def portable_dict(self) -> dict[str, object]:
        return {
            "runtime_fingerprint": self.runtime.runtime_fingerprint,
            "runtime_source_fingerprint": self.runtime.source_fingerprint,
            "execution_horizon": self.runtime.execution_horizon,
            "gripper_mode": self.runtime.gripper_mode,
            "development_schedule_fingerprint": self.runtime.development_schedule_fingerprint,
            "task_token_selection_fingerprint": self.task_token_selection.fingerprint,
            "selected_task_token_checkpoint_fingerprint": (
                self.task_token_selection.selected_checkpoint_fingerprint
            ),
            "development_comparison_fingerprint": canonical_fingerprint(
                cast(Mapping[str, object], _thaw_json(self.comparison))
            ),
            "task_token_m42_status": self.task_token_m42_status,
        }


@dataclass(frozen=True, slots=True)
class RestrictedAuditInputs:
    audit_mode: str
    dataset_root: Path
    dataset_fingerprint: str
    split_digest: str
    validation_episodes: tuple[ValidationEpisodeInput, ...]
    per_task_checkpoints: tuple[RestrictedCheckpointInput, ...]
    state_onehot_checkpoint: RestrictedCheckpointInput
    task_token_checkpoint: RestrictedCheckpointInput
    train_action_std: tuple[float, ...]
    runtime: RuntimeSelectionContract
    task_token_selection: M42SelectionRecord
    development: M42DevelopmentInputs | None
    access_trace: tuple[AuditInputAccess, ...]
    schema_version: str = M43_RESTRICTED_INPUT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != M43_RESTRICTED_INPUT_SCHEMA:
            raise M43AuditInputError("unknown restricted audit input schema")
        if self.audit_mode not in M43_AUDIT_MODES:
            raise M43AuditInputError("unknown restricted audit mode")
        if (self.audit_mode == "validation") != (self.development is None):
            raise M43AuditInputError("development rollout evidence differs from requested mode")
        _digest(self.dataset_fingerprint, "dataset_fingerprint")
        _digest(self.split_digest, "split_digest")
        if len(self.validation_episodes) != 36:
            raise M43AuditInputError("M4.3a requires exactly 36 M3B validation episodes")
        if tuple(item.task_id for item in self.per_task_checkpoints) != CANONICAL_TASK_IDS:
            raise M43AuditInputError("PerTask checkpoints must use canonical six-task order")
        checkpoints = (
            *self.per_task_checkpoints,
            self.state_onehot_checkpoint,
            self.task_token_checkpoint,
        )
        if any(
            item.dataset_fingerprint != self.dataset_fingerprint
            or item.split_digest != self.split_digest
            for item in checkpoints
        ):
            raise M43AuditInputError("audit checkpoints do not bind one M3B dataset/split")
        if not self.access_trace:
            raise M43AuditInputError("restricted audit inputs require a non-empty read trace")
        if len(self.train_action_std) != 8 or any(value <= 0.0 for value in self.train_action_std):
            raise M43AuditInputError("mixed train_action_std must contain eight positive values")

    def portable_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "audit_mode": self.audit_mode,
            "dataset_fingerprint": self.dataset_fingerprint,
            "split_digest": self.split_digest,
            "validation_episodes": [item.to_dict() for item in self.validation_episodes],
            "per_task_checkpoints": [item.portable_dict() for item in self.per_task_checkpoints],
            "state_onehot_checkpoint": self.state_onehot_checkpoint.portable_dict(),
            "task_token_checkpoint": self.task_token_checkpoint.portable_dict(),
            "train_action_std": list(self.train_action_std),
            "checkpoint_provenance": {
                "runtime_fingerprint": self.runtime.runtime_fingerprint,
                "runtime_source_fingerprint": self.runtime.source_fingerprint,
                "task_token_selection_fingerprint": self.task_token_selection.fingerprint,
            },
            "development": (None if self.development is None else self.development.portable_dict()),
            "access_trace": [item.to_dict() for item in self.access_trace],
            "test_split_accessed": False,
            "fresh_seed_accessed": False,
            "final_schedule_accessed": False,
        }

    @property
    def fingerprint(self) -> str:
        return f"sha256:{sha256_hex(self.portable_dict())}"


def _checkpoint_directory(run_root: Path, relative: str) -> Path:
    relative = _portable_relative(relative, "selected checkpoint")
    parts = PurePosixPath(relative).parts
    if not parts or parts[0] != "checkpoints":
        raise M43AuditInputError("selected checkpoint must remain below checkpoints/")
    candidate = run_root.joinpath(*parts)
    resolved = _resolved_unlinked(candidate, label="selected checkpoint", directory=True)
    try:
        resolved.relative_to(run_root)
    except ValueError as error:
        raise M43AuditInputError("selected checkpoint escaped its run root") from error
    return resolved


def _selected_record(
    manifest: ActExperimentManifest | TaskTokenTrainingManifest,
    fingerprint: str,
) -> CheckpointRecord:
    selected = next(
        (item for item in manifest.checkpoints if item.checkpoint_fingerprint == fingerprint),
        None,
    )
    if selected is None or not selected.complete:
        raise M43AuditInputError("selected checkpoint is absent or incomplete")
    return selected


def _feature_statistics(value: object, *, label: str) -> FeatureStatistics:
    if not isinstance(value, Mapping):
        raise M43AuditInputError(f"{label} feature statistics must be a mapping")
    try:
        return FeatureStatistics(
            feature_name=cast(str, value["feature_name"]),
            components=tuple(cast(Sequence[str], value["components"])),
            count=int(value["count"]),
            minimum=tuple(float(item) for item in cast(Sequence[object], value["minimum"])),
            maximum=tuple(float(item) for item in cast(Sequence[object], value["maximum"])),
            mean=tuple(float(item) for item in cast(Sequence[object], value["mean"])),
            std=tuple(float(item) for item in cast(Sequence[object], value["std"])),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise M43AuditInputError(f"{label} feature statistics are malformed") from error


def _train_statistics(
    value: Mapping[str, object],
    *,
    expected_variant: ActVariant,
    expected_fingerprint: str,
) -> TrainOnlyStatistics:
    leakage = value.get("leakage_audit")
    if not isinstance(leakage, Mapping):
        raise M43AuditInputError("train statistics lack a leakage audit")
    try:
        audit = StatisticsLeakageAudit(
            source_episode_indices=tuple(
                int(item) for item in cast(Sequence[object], leakage["source_episode_indices"])
            ),
            train_episode_indices=tuple(
                int(item) for item in cast(Sequence[object], leakage["train_episode_indices"])
            ),
            validation_episode_indices=tuple(
                int(item) for item in cast(Sequence[object], leakage["validation_episode_indices"])
            ),
            test_episode_indices=tuple(
                int(item) for item in cast(Sequence[object], leakage["test_episode_indices"])
            ),
        )
        result = TrainOnlyStatistics(
            variant=ActVariant(cast(str, value["variant"])),
            task_id=cast(str | None, value["task_id"]),
            image=_feature_statistics(value["image"], label="image"),
            state=_feature_statistics(value["state"], label="state"),
            action=_feature_statistics(value["action"], label="action"),
            leakage_audit=audit,
            lerobot_version=cast(str, value["lerobot_version"]),
            torch_version=cast(str, value["torch_version"]),
            algorithm=cast(str, value["algorithm"]),
            schema_version=cast(str, value["schema_version"]),
            task_onehot_mapping_version=cast(str, value["task_onehot_mapping_version"]),
            onehot_suffix_policy=cast(str, value["onehot_suffix_policy"]),
            statistics_fingerprint=cast(str, value["statistics_fingerprint"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise M43AuditInputError("train-only statistics are malformed") from error
    if (
        result.variant is not expected_variant
        or result.statistics_fingerprint != expected_fingerprint
    ):
        raise M43AuditInputError("train statistics differ from the selected run identity")
    if len(result.action.std) != 8 or any(value <= 0.0 for value in result.action.std):
        raise M43AuditInputError("action train statistics require eight positive std values")
    return result


def _discover_m4_checkpoints(
    reader: _RestrictedReader,
) -> tuple[
    tuple[RestrictedCheckpointInput, ...],
    RestrictedCheckpointInput,
    tuple[float, ...],
]:
    found: dict[tuple[ActVariant, str | None], RestrictedCheckpointInput] = {}
    onehot_std: tuple[float, ...] | None = None
    for child in sorted(reader.m4_checkpoint_root.iterdir(), key=lambda item: item.name):
        if _RUN_DIRECTORY.fullmatch(child.name) is None:
            continue
        if child.is_symlink() or child.is_junction() or not child.is_dir():
            raise M43AuditInputError(f"unsafe M4 fingerprint-owned run: {child}")
        manifest_path = child / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        raw = reader.read_json(manifest_path, label="M4 run manifest")
        identity = raw.get("identity")
        if not isinstance(identity, Mapping):
            raise M43AuditInputError("M4 run manifest lacks an identity")
        raw_variant = identity.get("variant")
        if raw_variant not in {
            ActVariant.PER_TASK.value,
            ActVariant.MIXED_TASK_ONEHOT.value,
        }:
            continue
        try:
            manifest = ActExperimentManifest.from_dict(raw)
        except (TypeError, ValueError) as error:
            raise M43AuditInputError("M4 run manifest is invalid") from error
        if (
            manifest.config.mode is not ExperimentMode.FULL
            or not manifest.complete
            or not manifest.training_state.completed
        ):
            # M4 roots are append-only experiment stores and legitimately retain
            # earlier non-full and interrupted runs.  They are not audit
            # candidates; only a completed full run reaches the strict gate
            # below.  Parsing before this filter keeps malformed historical
            # manifests fail-closed instead of silently hiding ambiguity.
            continue
        if manifest.git_dirty:
            raise M43AuditInputError("M4.3a accepts only clean completed full M4 runs")
        variant = manifest.identity.variant
        task_id = manifest.identity.task_id
        key = (variant, task_id)
        if key in found:
            raise M43AuditInputError("duplicate M4 semantic run below checkpoint root")
        selection_raw = reader.read_json(
            child / "checkpoint_selection.json", label="M4 validation-only selection"
        )
        try:
            selection = CheckpointSelectionRecord.from_dict(selection_raw)
        except (TypeError, ValueError) as error:
            raise M43AuditInputError("M4 checkpoint selection is invalid") from error
        if (
            selection.run_fingerprint != manifest.identity.run_fingerprint
            or not selection.selection_locked
            or selection.test_evaluation_status != "pending"
            or manifest.selected_checkpoint_fingerprint != selection.selected_checkpoint_fingerprint
        ):
            raise M43AuditInputError("M4 selection is not an immutable validation-only lock")
        record = _selected_record(manifest, selection.selected_checkpoint_fingerprint)
        _checkpoint_directory(child.resolve(), record.relative_path)
        kind = (
            M42PolicyKind.PER_TASK if variant is ActVariant.PER_TASK else M42PolicyKind.STATE_ONEHOT
        )
        found[key] = RestrictedCheckpointInput(
            policy_kind=kind,
            task_id=task_id,
            run_root=child.resolve(),
            run_fingerprint=manifest.identity.run_fingerprint,
            checkpoint_relative_path=record.relative_path,
            checkpoint_fingerprint=record.checkpoint_fingerprint,
            selection_fingerprint=selection.selection_fingerprint,
            statistics_fingerprint=record.statistics_fingerprint,
            dataset_fingerprint=manifest.identity.m3b_export_fingerprint,
            split_digest=manifest.identity.m3b_split_manifest_digest,
            architecture_fingerprint=None,
            git_commit=manifest.identity.git_commit,
        )
        if variant is ActVariant.MIXED_TASK_ONEHOT:
            statistics = _train_statistics(
                reader.read_json(child / "train_stats.json", label="State-OneHot train statistics"),
                expected_variant=ActVariant.MIXED_TASK_ONEHOT,
                expected_fingerprint=record.statistics_fingerprint,
            )
            onehot_std = tuple(float(value) for value in statistics.action.std)
    per_task: list[RestrictedCheckpointInput] = []
    for task_id in CANONICAL_TASK_IDS:
        checkpoint = found.get((ActVariant.PER_TASK, task_id))
        if checkpoint is None:
            raise M43AuditInputError(f"missing PerTask validation selection for {task_id}")
        per_task.append(checkpoint)
    onehot = found.get((ActVariant.MIXED_TASK_ONEHOT, None))
    if onehot is None or onehot_std is None or len(found) != 7:
        raise M43AuditInputError("M4.3a requires exactly six PerTask and one State-OneHot run")
    return tuple(per_task), onehot, onehot_std


def _runtime_selection(reader: _RestrictedReader) -> RuntimeSelectionContract:
    raw = reader.read_json(reader.runtime_selection_path, label="M4.2 runtime selection")
    _assert_development_only(raw, label="runtime selection")
    expected = {
        "schema_version",
        "execution_horizon",
        "gripper_mode",
        "horizon_selection_fingerprint",
        "gripper_selection_fingerprint",
        "runtime_fingerprint",
        "development_schedule_fingerprint",
        "m3b_dataset_fingerprint",
        "mixed_task_onehot_checkpoint_fingerprint",
        "representative_per_task_checkpoint_fingerprint",
        "implementation_fingerprint",
        "experiment_manifest_fingerprint",
        "evaluation_git_commit",
        "selection_evidence",
        "locked",
        "final_schedule_accessed",
    }
    if set(raw) != expected or raw.get("schema_version") != RUNTIME_SELECTION_SCHEMA:
        raise M43AuditInputError("runtime selection fields differ from the locked schema")
    try:
        return RuntimeSelectionContract(
            execution_horizon=cast(int, raw["execution_horizon"]),
            gripper_mode=cast(str, raw["gripper_mode"]),
            horizon_selection_fingerprint=cast(str, raw["horizon_selection_fingerprint"]),
            gripper_selection_fingerprint=cast(str, raw["gripper_selection_fingerprint"]),
            runtime_fingerprint=cast(str, raw["runtime_fingerprint"]),
            development_schedule_fingerprint=cast(str, raw["development_schedule_fingerprint"]),
            m3b_dataset_fingerprint=cast(str, raw["m3b_dataset_fingerprint"]),
            mixed_task_onehot_checkpoint_fingerprint=cast(
                str, raw["mixed_task_onehot_checkpoint_fingerprint"]
            ),
            representative_per_task_checkpoint_fingerprint=cast(
                str, raw["representative_per_task_checkpoint_fingerprint"]
            ),
            implementation_fingerprint=cast(str, raw["implementation_fingerprint"]),
            experiment_manifest_fingerprint=cast(str, raw["experiment_manifest_fingerprint"]),
            evaluation_git_commit=cast(str, raw["evaluation_git_commit"]),
            source_fingerprint=canonical_fingerprint(raw),
            selection_evidence=cast(Mapping[str, object], raw["selection_evidence"]),
            schema_version=cast(str, raw["schema_version"]),
            locked=cast(bool, raw["locked"]),
            final_schedule_accessed=cast(bool, raw["final_schedule_accessed"]),
        )
    except (M42TrainingContractError, TypeError, ValueError) as error:
        raise M43AuditInputError("runtime selection is invalid") from error


def _task_token_validation_result(value: Mapping[str, object]) -> TaskTokenValidationResult:
    try:
        return TaskTokenValidationResult(
            checkpoint_fingerprint=cast(str, value["checkpoint_fingerprint"]),
            checkpoint_step=int(value["checkpoint_step"]),
            validation_schedule_fingerprint=cast(str, value["validation_schedule_fingerprint"]),
            validation_split_digest=cast(str, value["validation_split_digest"]),
            episode_count=int(value["episode_count"]),
            success_count=int(value["success_count"]),
            wrong_object_interaction_count=int(value["wrong_object_interaction_count"]),
            target_off_table_count=int(value["target_off_table_count"]),
            timeout_count=int(value["timeout_count"]),
            offline_validation_action_loss=float(value["offline_validation_action_loss"]),
            development_schedule_accessed=cast(
                bool, value.get("development_schedule_accessed", False)
            ),
            final_schedule_accessed=cast(bool, value.get("final_schedule_accessed", False)),
            test_split_accessed=cast(bool, value.get("test_split_accessed", False)),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise M43AuditInputError("TaskToken validation result is malformed") from error


def _evaluation_implementation_fingerprint(git_commit: str) -> str:
    """Reconstruct the historical evaluation closure without opening rollout evidence."""

    if re.fullmatch(r"[0-9a-f]{40}", git_commit) is None:
        raise M43AuditInputError("M4.2 evaluation Git commit must be full lowercase SHA-1")
    git = shutil.which("git")
    if git is None:
        raise M43AuditInputError(
            "Git is required to reconstruct historical TaskToken validation provenance"
        )
    digests: dict[str, str] = {}
    for relative in _M42_EVALUATION_IMPLEMENTATION_FILES:
        completed = subprocess.run(
            [git, "show", f"{git_commit}:{relative}"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise M43AuditInputError(
                "historical M4.2 evaluation commit is unavailable for validation provenance: "
                f"{git_commit}:{relative}"
            )
        digests[relative] = f"sha256:{sha256_hex(completed.stdout.hex())}"
    return canonical_fingerprint(
        {
            "schema_version": _M42_EVALUATION_IMPLEMENTATION_SCHEMA,
            "git_commit": git_commit,
            "files": digests,
        }
    )


def _validation_artifact_identity(
    *,
    runtime: RuntimeSelectionContract,
    manifest: TaskTokenTrainingManifest,
    queue: TaskTokenValidationQueue,
    item: TaskTokenValidationQueueItem,
    implementation_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": M42_EVALUATION_ARTIFACT_SCHEMA,
        "stage": "validation_selection",
        "evaluation_git_commit": runtime.evaluation_git_commit,
        "implementation_fingerprint": implementation_fingerprint,
        "experiment_manifest_fingerprint": manifest.identity.experiment_manifest_fingerprint,
        "run_fingerprint": manifest.identity.run_fingerprint,
        "checkpoint_fingerprint": item.checkpoint_fingerprint,
        "checkpoint_step": item.global_step,
        "validation_schedule_fingerprint": queue.validation_schedule_fingerprint,
        "validation_split_digest": queue.split_digest,
        "execution_horizon": runtime.execution_horizon,
        "gripper_mode": runtime.gripper_mode,
        "maximum_episode_steps": M42_MAXIMUM_EPISODE_STEPS,
        "test_split_accessed": False,
        "development_schedule_accessed": False,
        "final_schedule_accessed": False,
    }


def _task_token_validation_results(
    reader: _RestrictedReader,
    *,
    runtime: RuntimeSelectionContract,
    manifest: TaskTokenTrainingManifest,
    queue: TaskTokenValidationQueue,
) -> tuple[TaskTokenValidationResult, ...]:
    results: list[TaskTokenValidationResult] = []
    implementation_fingerprint = _evaluation_implementation_fingerprint(
        runtime.evaluation_git_commit
    )
    for item in queue.checkpoints:
        identity = _validation_artifact_identity(
            runtime=runtime,
            manifest=manifest,
            queue=queue,
            item=item,
            implementation_fingerprint=implementation_fingerprint,
        )
        identity_fingerprint = canonical_fingerprint(identity)
        artifact_root = (
            reader.m42_diagnostics_root
            / "development"
            / "evidence"
            / identity_fingerprint.removeprefix("sha256:")
        )
        artifact = reader.read_json(
            artifact_root / "artifact.json", label="TaskToken validation evidence"
        )
        completion = reader.read_json(
            artifact_root / "complete.json", label="TaskToken validation evidence completion"
        )
        _assert_development_only(artifact, label="TaskToken validation evidence")
        _assert_development_only(completion, label="TaskToken validation completion")
        payload = artifact.get("payload")
        raw_result = payload.get("validation_result") if isinstance(payload, Mapping) else None
        if not isinstance(raw_result, Mapping):
            raise M43AuditInputError("TaskToken validation evidence lacks its ranking result")
        result = _task_token_validation_result(raw_result)
        artifact_fingerprint = canonical_fingerprint(artifact)
        if (
            artifact.get("schema_version") != M42_EVALUATION_ARTIFACT_SCHEMA
            or artifact.get("identity") != identity
            or artifact.get("passed") is not True
            or payload.get("validation_result_fingerprint") != result.fingerprint
            or completion
            != {
                "schema_version": M42_EVALUATION_ARTIFACT_SCHEMA,
                "identity_fingerprint": identity_fingerprint,
                "artifact_fingerprint": artifact_fingerprint,
                "passed": True,
            }
            or result.checkpoint_fingerprint != item.checkpoint_fingerprint
            or result.checkpoint_step != item.global_step
            or result.validation_schedule_fingerprint != queue.validation_schedule_fingerprint
            or result.validation_split_digest != queue.split_digest
            or result.offline_validation_action_loss != item.offline_validation_loss
        ):
            raise M43AuditInputError(
                "TaskToken validation artifact differs from its immutable queue/identity"
            )
        results.append(result)
    return tuple(results)


def _task_token_checkpoint(
    reader: _RestrictedReader,
    *,
    runtime: RuntimeSelectionContract,
) -> tuple[
    RestrictedCheckpointInput,
    TaskTokenTrainingManifest,
    M42SelectionRecord,
    tuple[float, ...],
]:
    manifests: list[tuple[Path, TaskTokenTrainingManifest]] = []
    for child in sorted(reader.task_token_checkpoint_root.iterdir(), key=lambda item: item.name):
        if _RUN_DIRECTORY.fullmatch(child.name) is None:
            continue
        if child.is_symlink() or child.is_junction() or not child.is_dir():
            raise M43AuditInputError(f"unsafe TaskToken fingerprint-owned run: {child}")
        path = child / "run_manifest.json"
        if not path.is_file():
            continue
        raw = reader.read_json(path, label="TaskToken training manifest")
        try:
            manifest = TaskTokenTrainingManifest.from_dict(raw)
        except (M42TrainingContractError, TypeError, ValueError) as error:
            raise M43AuditInputError("TaskToken training manifest is invalid") from error
        manifests.append((child.resolve(), manifest))
    if len(manifests) != 1:
        raise M43AuditInputError("M4.3a requires exactly one TaskToken run")
    run_root, manifest = manifests[0]
    if not manifest.training_complete:
        raise M43AuditInputError("TaskToken training run is incomplete")
    try:
        queue = TaskTokenValidationQueue.from_dict(
            reader.read_json(run_root / "validation_queue.json", label="TaskToken validation queue")
        )
    except (M42TrainingContractError, TypeError, ValueError) as error:
        raise M43AuditInputError("TaskToken validation queue is invalid") from error
    identity = manifest.identity
    validation_schedule_fingerprint = identity.data_contract.get("validation_schedule_fingerprint")
    manifest_records = tuple(
        (record.global_step, record.checkpoint_fingerprint, record.relative_path)
        for record in manifest.checkpoints
    )
    queue_records = tuple(
        (item.global_step, item.checkpoint_fingerprint, item.checkpoint_relative_path)
        for item in queue.checkpoints
    )
    if (
        not queue.complete
        or queue.run_fingerprint != identity.run_fingerprint
        or queue.m3b_dataset_fingerprint != identity.m3b_export_fingerprint
        or queue.split_digest != identity.m3b_split_manifest_digest
        or queue.train_statistics_fingerprint != identity.train_statistics_fingerprint
        or queue.runtime_selection_fingerprint != identity.runtime_selection_fingerprint
        or queue.experiment_manifest_fingerprint != identity.experiment_manifest_fingerprint
        or queue.task_token_architecture_fingerprint != identity.task_token_architecture_fingerprint
        or queue.validation_schedule_fingerprint != validation_schedule_fingerprint
        or queue.ordered_validation_episode_indices != identity.ordered_validation_episode_indices
        or queue.git_commit != identity.git_commit
        or queue_records != manifest_records
    ):
        raise M43AuditInputError(
            "TaskToken validation queue differs from its complete training identity"
        )

    development_root = reader.m42_diagnostics_root / "development"
    selection_artifact = reader.read_json(
        development_root / "task_token_checkpoint_selection.json",
        label="TaskToken validation-only selection",
    )
    _assert_development_only(selection_artifact, label="TaskToken selection")
    raw_selection = selection_artifact.get("selection")
    if not isinstance(raw_selection, Mapping):
        raise M43AuditInputError("TaskToken selection record is absent")
    try:
        selection = M42SelectionRecord.from_dict(raw_selection)
    except (TypeError, ValueError) as error:
        raise M43AuditInputError("TaskToken selection record is invalid") from error
    if (
        selection_artifact.get("schema_version") != TASK_TOKEN_SELECTION_SCHEMA
        or selection_artifact.get("selection_source") != "m3b_validation_only"
        or selection_artifact.get("locked") is not True
        or selection_artifact.get("selection_fingerprint") != selection.fingerprint
        or selection.selection_kind is not M42SelectionKind.TASK_TOKEN_CHECKPOINT
        or selection.development_schedule_fingerprint is not None
        or selection.final_schedule_accessed
    ):
        raise M43AuditInputError("TaskToken checkpoint was not selected by validation only")
    validation_results = _task_token_validation_results(
        reader, runtime=runtime, manifest=manifest, queue=queue
    )
    result_fingerprints = tuple(value.fingerprint for value in validation_results)
    persisted_fingerprints = selection_artifact.get("validation_result_fingerprints")
    try:
        recomputed = create_task_token_checkpoint_selection(
            validation_results, locked_at_utc=selection.locked_at_utc
        )
    except (M42AnalysisError, TypeError, ValueError) as error:
        raise M43AuditInputError("TaskToken validation-only ranking is invalid") from error
    if (
        not isinstance(persisted_fingerprints, Sequence)
        or isinstance(persisted_fingerprints, str | bytes)
        or tuple(persisted_fingerprints) != result_fingerprints
        or selection.evidence_fingerprints != result_fingerprints
        or selection.to_dict() != recomputed.to_dict()
    ):
        raise M43AuditInputError(
            "TaskToken selection does not reproduce all 20 validation-only results"
        )
    fingerprint = selection.selected_checkpoint_fingerprint
    if fingerprint is None:
        raise M43AuditInputError("TaskToken selection does not name a checkpoint")
    record = _selected_record(manifest, fingerprint)
    _checkpoint_directory(run_root, record.relative_path)
    if (
        identity.runtime_selection_fingerprint != runtime.runtime_fingerprint
        or identity.runtime_selection_source_fingerprint != runtime.source_fingerprint
        or identity.experiment_manifest_fingerprint != runtime.experiment_manifest_fingerprint
        or selection.validation_split_digest != identity.m3b_split_manifest_digest
    ):
        raise M43AuditInputError("TaskToken run differs from runtime or validation selection")
    result = RestrictedCheckpointInput(
        policy_kind=M42PolicyKind.TASK_TOKEN,
        task_id=None,
        run_root=run_root,
        run_fingerprint=identity.run_fingerprint,
        checkpoint_relative_path=record.relative_path,
        checkpoint_fingerprint=record.checkpoint_fingerprint,
        selection_fingerprint=selection.fingerprint,
        statistics_fingerprint=record.statistics_fingerprint,
        dataset_fingerprint=identity.m3b_export_fingerprint,
        split_digest=identity.m3b_split_manifest_digest,
        architecture_fingerprint=identity.task_token_architecture_fingerprint,
        git_commit=identity.git_commit,
    )
    statistics = _train_statistics(
        reader.read_json(run_root / "train_stats.json", label="TaskToken train statistics"),
        expected_variant=ActVariant.MIXED_UNCONDITIONED,
        expected_fingerprint=record.statistics_fingerprint,
    )
    return result, manifest, selection, tuple(float(value) for value in statistics.action.std)


def _validation_episodes(manifest: TaskTokenTrainingManifest) -> tuple[ValidationEpisodeInput, ...]:
    raw_schedule = manifest.identity.data_contract.get("validation_schedule")
    if not isinstance(raw_schedule, Sequence) or isinstance(raw_schedule, str | bytes):
        raise M43AuditInputError("TaskToken run lacks its validation-only schedule")
    indices = manifest.identity.ordered_validation_episode_indices
    if len(raw_schedule) != 36 or len(indices) != 36:
        raise M43AuditInputError("validation schedule must contain exactly 36 episodes")
    result: list[ValidationEpisodeInput] = []
    for expected_index, raw in zip(indices, raw_schedule, strict=True):
        if not isinstance(raw, Mapping) or not isinstance(raw.get("task_spec"), Mapping):
            raise M43AuditInputError("validation schedule record is malformed")
        task_spec = TaskSpec.from_mapping(cast(Mapping[str, object], raw["task_spec"]))
        record = ValidationEpisodeInput(
            episode_index=int(raw.get("episode_index", -1)),
            scene_seed=int(raw.get("scene_seed", -1)),
            scene_group_id=cast(str, raw.get("scene_group_id", "")),
            task_id=cast(str, raw.get("task_id", "")),
            task_spec=task_spec,
        )
        if record.episode_index != expected_index:
            raise M43AuditInputError("validation schedule order differs from run identity")
        result.append(record)
    groups: dict[str, list[ValidationEpisodeInput]] = {}
    for item in result:
        groups.setdefault(item.scene_group_id, []).append(item)
    if len(groups) != 6:
        raise M43AuditInputError("validation view must contain six counterfactual scene groups")
    for values in groups.values():
        if (
            len(values) != 6
            or tuple(item.task_id for item in values) != CANONICAL_TASK_IDS
            or len({item.scene_seed for item in values}) != 1
        ):
            raise M43AuditInputError(
                "each validation scene group must preserve one scene and canonical six tasks"
            )
    return tuple(result)


def _benchmark_checkpoint(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not isinstance(value.get("checkpoint"), Mapping):
        raise M43AuditInputError(f"{label} benchmark lacks a checkpoint descriptor")
    if value.get("schedule_id") != M42_DEV_SCHEDULE_ID:
        raise M43AuditInputError(f"{label} benchmark is not m42_dev_v0")
    return cast(Mapping[str, object], value["checkpoint"])


def _checkpoint_descriptor(value: RestrictedCheckpointInput) -> dict[str, object]:
    return {
        "policy_kind": value.policy_kind.value,
        "run_fingerprint": value.run_fingerprint,
        "checkpoint_fingerprint": value.checkpoint_fingerprint,
        "checkpoint_relative_path": value.checkpoint_relative_path,
        "dataset_fingerprint": value.dataset_fingerprint,
        "split_digest": value.split_digest,
        "statistics_fingerprint": value.statistics_fingerprint,
        "architecture_fingerprint": value.architecture_fingerprint,
        "task_id": value.task_id,
        "git_commit": value.git_commit,
    }


def _development_inputs(
    reader: _RestrictedReader,
    *,
    runtime: RuntimeSelectionContract,
    per_task: tuple[RestrictedCheckpointInput, ...],
    onehot: RestrictedCheckpointInput,
    task_token: RestrictedCheckpointInput,
    task_manifest: TaskTokenTrainingManifest,
    selection: M42SelectionRecord,
) -> M42DevelopmentInputs:
    root = reader.m42_diagnostics_root / "development"
    comparison = reader.read_json(
        root / "development_comparison.json", label="M4.2 development comparison"
    )
    completion = reader.read_json(
        root / "development_complete.json", label="M4.2 development completion"
    )
    _assert_development_only(comparison, label="development comparison")
    _assert_development_only(completion, label="development completion")
    comparison_fingerprint = canonical_fingerprint(comparison)
    if (
        comparison.get("schema_version") != DEVELOPMENT_COMPARISON_SCHEMA
        or comparison.get("schedule_id") != M42_DEV_SCHEDULE_ID
        or comparison.get("schedule_fingerprint") != runtime.development_schedule_fingerprint
        or comparison.get("execution_horizon") != runtime.execution_horizon
        or comparison.get("gripper_mode") != runtime.gripper_mode
        or comparison.get("selected_task_token_checkpoint_fingerprint")
        != task_token.checkpoint_fingerprint
        or comparison.get("selection_fingerprint") != selection.fingerprint
        or comparison.get("selection_source") != "m3b_validation_only"
        or comparison.get("complete") is not True
    ):
        raise M43AuditInputError("development comparison differs from locked audit inputs")
    if (
        completion.get("schema_version") != DEVELOPMENT_COMPLETION_SCHEMA
        or completion.get("passed") is not True
        or completion.get("development_schedule_fingerprint")
        != runtime.development_schedule_fingerprint
        or completion.get("runtime_selection_fingerprint") != runtime.runtime_fingerprint
        or completion.get("runtime_selection_source_fingerprint") != runtime.source_fingerprint
        or completion.get("task_token_run_fingerprint") != task_token.run_fingerprint
        or completion.get("task_token_selection_fingerprint") != selection.fingerprint
        or completion.get("selected_task_token_checkpoint_fingerprint")
        != task_token.checkpoint_fingerprint
        or completion.get("development_comparison_fingerprint") != comparison_fingerprint
        or completion.get("experiment_manifest_fingerprint")
        != task_manifest.identity.experiment_manifest_fingerprint
    ):
        raise M43AuditInputError("development completion does not bind the comparison")
    models = comparison.get("models")
    if not isinstance(models, Mapping):
        raise M43AuditInputError("development comparison lacks model evidence")
    onehot_descriptor = _benchmark_checkpoint(models.get("state_onehot"), label="State-OneHot")
    token_descriptor = _benchmark_checkpoint(models.get("task_token"), label="TaskToken")
    raw_per_task = models.get("per_task")
    if not isinstance(raw_per_task, Mapping) or not isinstance(
        raw_per_task.get("benchmarks"), Sequence
    ):
        raise M43AuditInputError("development comparison lacks PerTask benchmarks")
    per_task_benchmarks = cast(Sequence[object], raw_per_task["benchmarks"])
    if len(per_task_benchmarks) != 6:
        raise M43AuditInputError("development comparison requires six PerTask benchmarks")
    per_task_descriptors = tuple(
        _benchmark_checkpoint(value, label=f"PerTask[{index}]")
        for index, value in enumerate(per_task_benchmarks)
    )
    if (
        dict(onehot_descriptor) != _checkpoint_descriptor(onehot)
        or dict(token_descriptor) != _checkpoint_descriptor(task_token)
        or tuple(dict(item) for item in per_task_descriptors)
        != tuple(_checkpoint_descriptor(item) for item in per_task)
    ):
        raise M43AuditInputError("development model evidence uses different checkpoints/tasks")
    return M42DevelopmentInputs(
        runtime=runtime,
        task_token_selection=selection,
        comparison=comparison,
        completion=completion,
    )


def load_restricted_audit_inputs(
    *,
    mode: str = "combined",
    dataset_root: str | Path,
    m4_checkpoint_root: str | Path,
    task_token_checkpoint_root: str | Path,
    m42_diagnostics_root: str | Path,
    runtime_selection_path: str | Path,
) -> RestrictedAuditInputs:
    """Load only validation-selection and ``m42_dev_v0`` audit authorities."""

    if mode not in M43_AUDIT_MODES:
        raise M43AuditInputError(f"unknown M4.3a audit mode: {mode!r}")

    reader = _RestrictedReader(
        dataset_root=Path(dataset_root),
        m4_checkpoint_root=Path(m4_checkpoint_root),
        task_token_checkpoint_root=Path(task_token_checkpoint_root),
        m42_diagnostics_root=Path(m42_diagnostics_root),
        runtime_selection_path=Path(runtime_selection_path),
    )
    completion = reader.read_json(
        reader.dataset_root / COMPLETION_MARKER, label="M3B completion marker"
    )
    if (
        set(completion) != {"schema_version", "export_fingerprint"}
        or completion.get("schema_version") != COMPLETION_SCHEMA
    ):
        raise M43AuditInputError("M3B completion marker fields are invalid")
    dataset_fingerprint = _digest(completion.get("export_fingerprint"), "M3B fingerprint")

    per_task, onehot, onehot_std = _discover_m4_checkpoints(reader)
    runtime = _runtime_selection(reader)
    task_token, task_manifest, selection, task_token_std = _task_token_checkpoint(
        reader, runtime=runtime
    )
    validation = _validation_episodes(task_manifest)
    split_digest = onehot.split_digest
    if (
        dataset_fingerprint != onehot.dataset_fingerprint
        or dataset_fingerprint != task_token.dataset_fingerprint
        or runtime.m3b_dataset_fingerprint != dataset_fingerprint
        or runtime.mixed_task_onehot_checkpoint_fingerprint != onehot.checkpoint_fingerprint
        or runtime.representative_per_task_checkpoint_fingerprint
        not in {item.checkpoint_fingerprint for item in per_task}
        or task_manifest.identity.ordered_validation_episode_indices
        != tuple(item.episode_index for item in validation)
        or any(item.dataset_fingerprint != dataset_fingerprint for item in per_task)
        or any(item.split_digest != split_digest for item in per_task)
    ):
        raise M43AuditInputError("validation inputs do not share one frozen M3B authority")
    if task_token_std != onehot_std:
        raise M43AuditInputError(
            "State-OneHot and TaskToken action train statistics must match exactly"
        )
    task_counts = Counter(item.task_id for item in validation)
    if task_counts != Counter({task_id: 6 for task_id in CANONICAL_TASK_IDS}):
        raise M43AuditInputError("validation view must contain six episodes per TaskSpec")
    development = None
    if mode in {"m42_dev_v0", "combined"}:
        development = _development_inputs(
            reader,
            runtime=runtime,
            per_task=per_task,
            onehot=onehot,
            task_token=task_token,
            task_manifest=task_manifest,
            selection=selection,
        )
    return RestrictedAuditInputs(
        audit_mode=mode,
        dataset_root=reader.dataset_root,
        dataset_fingerprint=dataset_fingerprint,
        split_digest=split_digest,
        validation_episodes=validation,
        per_task_checkpoints=per_task,
        state_onehot_checkpoint=onehot,
        task_token_checkpoint=task_token,
        train_action_std=onehot_std,
        runtime=runtime,
        task_token_selection=selection,
        development=development,
        access_trace=reader.trace,
    )


__all__ = [
    "M43_RESTRICTED_INPUT_SCHEMA",
    "M43AuditInputError",
    "AuditInputAccess",
    "M42DevelopmentInputs",
    "RestrictedAuditInputs",
    "RestrictedCheckpointInput",
    "ValidationEpisodeInput",
    "load_restricted_audit_inputs",
]
