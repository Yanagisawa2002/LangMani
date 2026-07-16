"""Immutable evaluation-evidence lifecycle for M4.3b FactorFiLM.

This module owns compact evaluation evidence only.  It never reads or writes a
training run, checkpoint, dataset, simulator state, image, or video.  Callers
provide already-computed validation, reload, development, semantic, and quality
gate records.  The records are linked through a fixed stage DAG, checksummed in
fingerprint-owned staging, independently validated, and atomically promoted.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Self, cast

from langmani.policies.act_factor_film_evaluation import (
    FactorFiLMValidationCountResult,
    rank_factor_film_validation_count_results,
)
from langmani.policies.act_factor_film_types import (
    FactorFiLMSelectionRecord,
    canonical_fingerprint,
)

FACTOR_FILM_EVIDENCE_CONFIG_SCHEMA_VERSION = (
    "langmani-m43-factor-film-evaluation-evidence-config-v0"
)
FACTOR_FILM_EVIDENCE_STAGE_SCHEMA_VERSION = "langmani-m43-factor-film-evaluation-stage-v0"
FACTOR_FILM_EVIDENCE_MANIFEST_SCHEMA_VERSION = "langmani-m43-factor-film-evaluation-manifest-v0"
FACTOR_FILM_EVIDENCE_COMPLETION_SCHEMA_VERSION = "langmani-m43-factor-film-evaluation-complete-v0"
FACTOR_FILM_EVIDENCE_OWNER_SCHEMA_VERSION = "langmani-m43-factor-film-evaluation-owner-v0"
FACTOR_FILM_VALIDATION_EVIDENCE_SCHEMA_VERSION = "langmani-m43-factor-film-validation-evidence-v0"
FACTOR_FILM_EVIDENCE_DIRECTORY = "factor-film-evaluation-evidence"
FACTOR_FILM_EVIDENCE_STAGE_DIRECTORY = "stages"
FACTOR_FILM_EVIDENCE_CONFIG_FILE = "config.json"
FACTOR_FILM_EVIDENCE_OWNER_FILE = "owner.json"
FACTOR_FILM_EVIDENCE_MANIFEST_FILE = "manifest.json"
FACTOR_FILM_EVIDENCE_COMPLETION_FILE = "complete.json"

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_RE = re.compile(r"^[0-9a-f]{40,64}$")
_PROHIBITED_VALUE_FRAGMENTS = (
    "m3b_test",
    "test_split",
    "fresh_seed",
    "fresh-seed",
    "m4_fresh",
    "m42_final_v0",
)
_PROHIBITED_FLAG_KEYS = {
    "test_accessed",
    "test_split_accessed",
    "fresh_seed_accessed",
    "historical_fresh_accessed",
    "final_schedule_accessed",
    "final_benchmark_executed",
}


class FactorFiLMEvidenceError(RuntimeError):
    """Raised when FactorFiLM evaluation evidence is unsafe or inconsistent."""


class FactorFiLMEvaluationStage(StrEnum):
    """The only authorized M4.3b target-development evidence stages."""

    VALIDATION = "validation"
    SELECTION = "selection"
    RELOAD = "reload"
    DEVELOPMENT = "development"
    SEMANTIC = "semantic"
    QUALITY_GATE = "quality_gate_completion"


FACTOR_FILM_EVALUATION_STAGE_ORDER = tuple(FactorFiLMEvaluationStage)


class _FrozenMapping(Mapping[str, object]):
    __slots__ = ("_items",)

    def __init__(self, value: Mapping[str, object]) -> None:
        if not all(isinstance(key, str) for key in value):
            raise TypeError("FactorFiLM evidence mappings require string keys")
        self._items = tuple((key, _freeze_json(item)) for key, item in sorted(value.items()))

    def __getitem__(self, key: str) -> object:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __deepcopy__(self, memo: object) -> _FrozenMapping:
        del memo
        return self


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return _FrozenMapping(cast(Mapping[str, object], value))
    if isinstance(value, tuple | list):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, bool | int | float | str | StrEnum):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _freeze_json(to_dict())
    raise TypeError(f"unsupported FactorFiLM evidence value {type(value).__name__}")


def _json_value(value: object) -> object:
    if isinstance(value, StrEnum):
        return value.value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_value(to_dict())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"unsupported FactorFiLM evidence JSON value {type(value).__name__}")


class _JsonRecord:
    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], _json_value(asdict(self)))


def _require_digest(value: str, name: str) -> None:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise FactorFiLMEvidenceError(f"{name} must be sha256:<64 lowercase hex>")


def _require_exact_keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise FactorFiLMEvidenceError(f"{name} fields differ from the declared schema")


@dataclass(frozen=True, slots=True)
class FactorFiLMEvaluationEvidenceConfig(_JsonRecord):
    """Portable identity for one separately authorized development evaluation."""

    run_fingerprint: str
    training_manifest_fingerprint: str
    training_completion_fingerprint: str
    validation_queue_fingerprint: str
    dataset_fingerprint: str
    split_fingerprint: str
    semantic_audit_evidence_fingerprint: str
    validation_schedule_digest: str
    development_schedule_fingerprint: str
    training_git_commit: str
    evaluation_git_commit: str
    selected_execution_horizon: int = 10
    action_bound_mode: str = "project"
    test_split_accessed: bool = False
    fresh_seed_accessed: bool = False
    final_schedule_accessed: bool = False
    schema_version: str = FACTOR_FILM_EVIDENCE_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "run_fingerprint",
            "training_manifest_fingerprint",
            "training_completion_fingerprint",
            "validation_queue_fingerprint",
            "dataset_fingerprint",
            "split_fingerprint",
            "semantic_audit_evidence_fingerprint",
            "validation_schedule_digest",
            "development_schedule_fingerprint",
        ):
            _require_digest(cast(str, getattr(self, name)), name)
        for name in ("training_git_commit", "evaluation_git_commit"):
            value = getattr(self, name)
            if not isinstance(value, str) or _GIT_RE.fullmatch(value) is None:
                raise FactorFiLMEvidenceError(f"{name} must be lowercase Git hex")
        if self.selected_execution_horizon != 10 or self.action_bound_mode != "project":
            raise FactorFiLMEvidenceError("FactorFiLM evidence requires locked H=10/project")
        if any(
            value is not False
            for value in (
                self.test_split_accessed,
                self.fresh_seed_accessed,
                self.final_schedule_accessed,
            )
        ):
            raise FactorFiLMEvidenceError(
                "FactorFiLM development evidence accessed a sealed source"
            )
        if self.schema_version != FACTOR_FILM_EVIDENCE_CONFIG_SCHEMA_VERSION:
            raise FactorFiLMEvidenceError("unknown FactorFiLM evidence config schema")

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        _require_exact_keys(
            value,
            {
                "run_fingerprint",
                "training_manifest_fingerprint",
                "training_completion_fingerprint",
                "validation_queue_fingerprint",
                "dataset_fingerprint",
                "split_fingerprint",
                "semantic_audit_evidence_fingerprint",
                "validation_schedule_digest",
                "development_schedule_fingerprint",
                "training_git_commit",
                "evaluation_git_commit",
                "selected_execution_horizon",
                "action_bound_mode",
                "test_split_accessed",
                "fresh_seed_accessed",
                "final_schedule_accessed",
                "schema_version",
            },
            "FactorFiLM evaluation evidence config",
        )
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class FactorFiLMEvaluationStageArtifact(_JsonRecord):
    """One compact, content-bound stage record in the fixed evaluation DAG."""

    stage: FactorFiLMEvaluationStage
    run_fingerprint: str
    input_artifact_fingerprints: Mapping[str, object]
    payload: Mapping[str, object]
    selected_checkpoint_fingerprint: str | None = None
    selected_checkpoint_step: int | None = None
    complete: bool = True
    test_split_accessed: bool = False
    fresh_seed_accessed: bool = False
    final_schedule_accessed: bool = False
    schema_version: str = FACTOR_FILM_EVIDENCE_STAGE_SCHEMA_VERSION
    artifact_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.stage, FactorFiLMEvaluationStage):
            raise TypeError("stage must be FactorFiLMEvaluationStage")
        _require_digest(self.run_fingerprint, "stage run_fingerprint")
        if not isinstance(self.input_artifact_fingerprints, Mapping):
            raise TypeError("input_artifact_fingerprints must be a mapping")
        for name, value in self.input_artifact_fingerprints.items():
            if not isinstance(name, str) or not name or not isinstance(value, str):
                raise TypeError("stage input fingerprints require string names and values")
            _require_digest(value, f"stage input {name}")
        if not isinstance(self.payload, Mapping) or not self.payload:
            raise FactorFiLMEvidenceError("stage payload must be one non-empty mapping")
        _validate_portable_json(self.payload, "stage payload")
        _reject_prohibited_sources(self.payload)
        if self.stage is FactorFiLMEvaluationStage.VALIDATION:
            if (
                self.selected_checkpoint_fingerprint is not None
                or self.selected_checkpoint_step is not None
            ):
                raise FactorFiLMEvidenceError("validation stage cannot preselect a checkpoint")
        else:
            if self.selected_checkpoint_fingerprint is None:
                raise FactorFiLMEvidenceError(
                    "post-validation stage requires a selected checkpoint"
                )
            _require_digest(
                self.selected_checkpoint_fingerprint,
                "selected_checkpoint_fingerprint",
            )
            if (
                isinstance(self.selected_checkpoint_step, bool)
                or not isinstance(self.selected_checkpoint_step, int)
                or self.selected_checkpoint_step not in range(5_000, 100_001, 5_000)
            ):
                raise FactorFiLMEvidenceError(
                    "selected checkpoint step is outside the 20-step queue"
                )
        if self.complete is not True:
            raise FactorFiLMEvidenceError("only completed stage artifacts can be promoted")
        if any(
            value is not False
            for value in (
                self.test_split_accessed,
                self.fresh_seed_accessed,
                self.final_schedule_accessed,
            )
        ):
            raise FactorFiLMEvidenceError("stage artifact accessed a sealed source")
        if self.schema_version != FACTOR_FILM_EVIDENCE_STAGE_SCHEMA_VERSION:
            raise FactorFiLMEvidenceError("unknown FactorFiLM stage schema")
        object.__setattr__(
            self,
            "input_artifact_fingerprints",
            _FrozenMapping(cast(Mapping[str, object], self.input_artifact_fingerprints)),
        )
        object.__setattr__(self, "payload", _FrozenMapping(self.payload))
        expected = canonical_fingerprint(self.identity_payload())
        if not self.artifact_fingerprint:
            object.__setattr__(self, "artifact_fingerprint", expected)
        elif self.artifact_fingerprint != expected:
            raise FactorFiLMEvidenceError("stage artifact fingerprint mismatch")

    def identity_payload(self) -> dict[str, object]:
        payload = self.to_dict()
        payload.pop("artifact_fingerprint")
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        _require_exact_keys(
            value,
            {
                "stage",
                "run_fingerprint",
                "input_artifact_fingerprints",
                "payload",
                "selected_checkpoint_fingerprint",
                "selected_checkpoint_step",
                "complete",
                "test_split_accessed",
                "fresh_seed_accessed",
                "final_schedule_accessed",
                "schema_version",
                "artifact_fingerprint",
            },
            "FactorFiLM evaluation stage artifact",
        )
        inputs = value["input_artifact_fingerprints"]
        payload = value["payload"]
        if not isinstance(inputs, Mapping) or not isinstance(payload, Mapping):
            raise TypeError("stage inputs and payload must be mappings")
        return cls(
            stage=FactorFiLMEvaluationStage(cast(str, value["stage"])),
            run_fingerprint=cast(str, value["run_fingerprint"]),
            input_artifact_fingerprints=cast(Mapping[str, object], inputs),
            payload=cast(Mapping[str, object], payload),
            selected_checkpoint_fingerprint=cast(
                str | None, value["selected_checkpoint_fingerprint"]
            ),
            selected_checkpoint_step=cast(int | None, value["selected_checkpoint_step"]),
            complete=cast(bool, value["complete"]),
            test_split_accessed=cast(bool, value["test_split_accessed"]),
            fresh_seed_accessed=cast(bool, value["fresh_seed_accessed"]),
            final_schedule_accessed=cast(bool, value["final_schedule_accessed"]),
            schema_version=cast(str, value["schema_version"]),
            artifact_fingerprint=cast(str, value["artifact_fingerprint"]),
        )


@dataclass(frozen=True, slots=True)
class CompletedFactorFiLMEvaluationEvidence:
    """Independently validated immutable FactorFiLM evaluation evidence."""

    root: Path
    config: FactorFiLMEvaluationEvidenceConfig
    stage_artifacts: Mapping[FactorFiLMEvaluationStage, FactorFiLMEvaluationStageArtifact]
    manifest: Mapping[str, object]
    completion: Mapping[str, object]

    @property
    def evidence_fingerprint(self) -> str:
        return cast(str, self.completion["evidence_fingerprint"])


def stage_and_promote_factor_film_evaluation_evidence(
    *,
    output_root: str | Path,
    training_run_root: str | Path,
    config: FactorFiLMEvaluationEvidenceConfig,
    stage_artifacts: Mapping[FactorFiLMEvaluationStage, FactorFiLMEvaluationStageArtifact],
    clean_matching_staging: bool = False,
) -> CompletedFactorFiLMEvaluationEvidence:
    """Stage, checksum, validate, and atomically promote one evidence bundle."""
    if not isinstance(config, FactorFiLMEvaluationEvidenceConfig):
        raise TypeError("config must be FactorFiLMEvaluationEvidenceConfig")
    artifacts = _validate_stage_artifact_set(config, stage_artifacts)
    evidence_root = _safe_evidence_root(Path(output_root), Path(training_run_root))
    token = config.fingerprint.removeprefix("sha256:")
    destination = _safe_child(evidence_root, token, "completed FactorFiLM evidence")
    staging = _safe_child(evidence_root, f".staging-{token}", "FactorFiLM staging")

    if destination.exists():
        completed = validate_completed_factor_film_evaluation_evidence(
            destination,
            expected_config=config,
            expected_stage_artifacts=artifacts,
            training_run_root=training_run_root,
        )
        return completed
    if staging.exists():
        _require_real_directory(staging, "FactorFiLM staging")
        owner = _read_object(staging / FACTOR_FILM_EVIDENCE_OWNER_FILE, "staging owner")
        if owner.get("config_fingerprint") != config.fingerprint:
            raise FactorFiLMEvidenceError("refusing to clean staging owned by another config")
        if not clean_matching_staging:
            raise FactorFiLMEvidenceError(
                "matching staging exists; explicit clean_matching_staging is required"
            )
        _validate_owned_staging_for_cleanup(staging, config)
        shutil.rmtree(staging)

    staging.mkdir(parents=False, exist_ok=False)
    try:
        _write_new_json(
            staging / FACTOR_FILM_EVIDENCE_OWNER_FILE,
            {
                "schema_version": FACTOR_FILM_EVIDENCE_OWNER_SCHEMA_VERSION,
                "config_fingerprint": config.fingerprint,
                "config": config.to_dict(),
            },
        )
        _write_new_json(staging / FACTOR_FILM_EVIDENCE_CONFIG_FILE, config.to_dict())
        stage_root = staging / FACTOR_FILM_EVIDENCE_STAGE_DIRECTORY
        stage_root.mkdir(exist_ok=False)
        for stage in FACTOR_FILM_EVALUATION_STAGE_ORDER:
            _write_new_json(stage_root / f"{stage.value}.json", artifacts[stage].to_dict())

        artifact_records = _artifact_records(staging)
        stage_fingerprints = {
            stage.value: artifacts[stage].artifact_fingerprint
            for stage in FACTOR_FILM_EVALUATION_STAGE_ORDER
        }
        manifest_identity = {
            "schema_version": FACTOR_FILM_EVIDENCE_MANIFEST_SCHEMA_VERSION,
            "config_fingerprint": config.fingerprint,
            "stage_artifact_fingerprints": stage_fingerprints,
            "artifacts": artifact_records,
        }
        evidence_fingerprint = canonical_fingerprint(manifest_identity)
        manifest = {**manifest_identity, "evidence_fingerprint": evidence_fingerprint}
        _write_new_json(staging / FACTOR_FILM_EVIDENCE_MANIFEST_FILE, manifest)
        completion = _completion_payload(
            config=config,
            artifacts=artifacts,
            evidence_fingerprint=evidence_fingerprint,
            manifest_sha256=_sha256_file(staging / FACTOR_FILM_EVIDENCE_MANIFEST_FILE),
        )
        _write_new_json(staging / FACTOR_FILM_EVIDENCE_COMPLETION_FILE, completion)
        _fsync_tree(staging)
        validate_completed_factor_film_evaluation_evidence(
            staging,
            expected_config=config,
            expected_stage_artifacts=artifacts,
            training_run_root=training_run_root,
        )
        if destination.exists():
            raise FactorFiLMEvidenceError("completed evidence appeared during promotion")
        if staging.stat().st_dev != evidence_root.stat().st_dev:
            raise FactorFiLMEvidenceError("staging and destination are on different filesystems")
        try:
            os.replace(staging, destination)
        except OSError as error:
            raise FactorFiLMEvidenceError(
                f"atomic FactorFiLM evidence promotion failed: {error}"
            ) from error
        _fsync_directory(evidence_root)
    except BaseException:
        if staging.exists():
            _require_real_directory(staging, "failed FactorFiLM staging")
        raise
    return validate_completed_factor_film_evaluation_evidence(
        destination,
        expected_config=config,
        expected_stage_artifacts=artifacts,
        training_run_root=training_run_root,
    )


def validate_completed_factor_film_evaluation_evidence(
    path: str | Path,
    *,
    expected_config: FactorFiLMEvaluationEvidenceConfig | None = None,
    expected_stage_artifacts: Mapping[FactorFiLMEvaluationStage, FactorFiLMEvaluationStageArtifact]
    | None = None,
    training_run_root: str | Path | None = None,
) -> CompletedFactorFiLMEvaluationEvidence:
    """Validate schemas, stage references, file checksums, and completion."""
    root = _resolved_unlinked(Path(path), label="completed FactorFiLM evidence")
    if training_run_root is not None:
        training = _resolved_unlinked(Path(training_run_root), label="training run root")
        if _overlaps(root, training):
            raise FactorFiLMEvidenceError("evaluation evidence overlaps the training run")
    _require_real_directory(root, "completed FactorFiLM evidence")
    config = FactorFiLMEvaluationEvidenceConfig.from_dict(
        _read_object(root / FACTOR_FILM_EVIDENCE_CONFIG_FILE, "evidence config")
    )
    if expected_config is not None and config.to_dict() != expected_config.to_dict():
        raise FactorFiLMEvidenceError("completed evidence config differs from expected config")
    token = config.fingerprint.removeprefix("sha256:")
    if root.name not in {token, f".staging-{token}"}:
        raise FactorFiLMEvidenceError("evidence directory is not owned by its config fingerprint")
    owner = _read_object(root / FACTOR_FILM_EVIDENCE_OWNER_FILE, "evidence owner")
    if owner != {
        "schema_version": FACTOR_FILM_EVIDENCE_OWNER_SCHEMA_VERSION,
        "config_fingerprint": config.fingerprint,
        "config": config.to_dict(),
    }:
        raise FactorFiLMEvidenceError("evidence owner is not content-bound to its config")

    stage_artifacts: dict[FactorFiLMEvaluationStage, FactorFiLMEvaluationStageArtifact] = {}
    for stage in FACTOR_FILM_EVALUATION_STAGE_ORDER:
        payload = _read_object(
            root / FACTOR_FILM_EVIDENCE_STAGE_DIRECTORY / f"{stage.value}.json",
            f"{stage.value} stage artifact",
        )
        artifact = FactorFiLMEvaluationStageArtifact.from_dict(payload)
        if artifact.stage is not stage:
            raise FactorFiLMEvidenceError("stage filename and declared stage disagree")
        stage_artifacts[stage] = artifact
    artifacts = _validate_stage_artifact_set(config, stage_artifacts)
    if expected_stage_artifacts is not None:
        expected = _validate_stage_artifact_set(config, expected_stage_artifacts)
        if {key: value.to_dict() for key, value in artifacts.items()} != {
            key: value.to_dict() for key, value in expected.items()
        }:
            raise FactorFiLMEvidenceError("completed stage artifacts differ from expected evidence")

    manifest = _read_object(root / FACTOR_FILM_EVIDENCE_MANIFEST_FILE, "evidence manifest")
    completion = _read_object(root / FACTOR_FILM_EVIDENCE_COMPLETION_FILE, "evidence completion")
    _validate_artifact_records(root, manifest)
    stage_fingerprints = {
        stage.value: artifacts[stage].artifact_fingerprint
        for stage in FACTOR_FILM_EVALUATION_STAGE_ORDER
    }
    expected_manifest_identity = {
        "schema_version": FACTOR_FILM_EVIDENCE_MANIFEST_SCHEMA_VERSION,
        "config_fingerprint": config.fingerprint,
        "stage_artifact_fingerprints": stage_fingerprints,
        "artifacts": manifest.get("artifacts"),
    }
    expected_manifest = {
        **expected_manifest_identity,
        "evidence_fingerprint": canonical_fingerprint(expected_manifest_identity),
    }
    if manifest != expected_manifest:
        raise FactorFiLMEvidenceError("evidence manifest is not content-bound")
    expected_completion = _completion_payload(
        config=config,
        artifacts=artifacts,
        evidence_fingerprint=cast(str, expected_manifest["evidence_fingerprint"]),
        manifest_sha256=_sha256_file(root / FACTOR_FILM_EVIDENCE_MANIFEST_FILE),
    )
    if completion != expected_completion:
        raise FactorFiLMEvidenceError("evidence completion is not content-bound")

    expected_files = {
        FACTOR_FILM_EVIDENCE_CONFIG_FILE,
        FACTOR_FILM_EVIDENCE_OWNER_FILE,
        FACTOR_FILM_EVIDENCE_MANIFEST_FILE,
        FACTOR_FILM_EVIDENCE_COMPLETION_FILE,
        *{
            f"{FACTOR_FILM_EVIDENCE_STAGE_DIRECTORY}/{stage.value}.json"
            for stage in FACTOR_FILM_EVALUATION_STAGE_ORDER
        },
    }
    if _relative_files(root) != expected_files:
        raise FactorFiLMEvidenceError("completed evidence has extra or missing artifacts")
    return CompletedFactorFiLMEvaluationEvidence(
        root=root,
        config=config,
        stage_artifacts=_FrozenStageMapping(stage_artifacts),
        manifest=_FrozenMapping(manifest),
        completion=_FrozenMapping(completion),
    )


class _FrozenStageMapping(Mapping[FactorFiLMEvaluationStage, FactorFiLMEvaluationStageArtifact]):
    __slots__ = ("_items",)

    def __init__(
        self,
        value: Mapping[FactorFiLMEvaluationStage, FactorFiLMEvaluationStageArtifact],
    ) -> None:
        self._items = tuple((stage, value[stage]) for stage in FACTOR_FILM_EVALUATION_STAGE_ORDER)

    def __getitem__(self, key: FactorFiLMEvaluationStage) -> FactorFiLMEvaluationStageArtifact:
        for stage, artifact in self._items:
            if stage is key:
                return artifact
        raise KeyError(key)

    def __iter__(self):
        return (stage for stage, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)


def _validate_stage_artifact_set(
    config: FactorFiLMEvaluationEvidenceConfig,
    stage_artifacts: Mapping[FactorFiLMEvaluationStage, FactorFiLMEvaluationStageArtifact],
) -> dict[FactorFiLMEvaluationStage, FactorFiLMEvaluationStageArtifact]:
    if set(stage_artifacts) != set(FACTOR_FILM_EVALUATION_STAGE_ORDER):
        raise FactorFiLMEvidenceError("evidence requires exactly six declared stages")
    artifacts = {stage: stage_artifacts[stage] for stage in FACTOR_FILM_EVALUATION_STAGE_ORDER}
    if any(
        not isinstance(value, FactorFiLMEvaluationStageArtifact) for value in artifacts.values()
    ):
        raise TypeError("stage_artifacts contains a malformed value")
    for stage, artifact in artifacts.items():
        if artifact.stage is not stage or artifact.run_fingerprint != config.run_fingerprint:
            raise FactorFiLMEvidenceError("stage identity differs from the evidence config")

    expected_inputs: dict[FactorFiLMEvaluationStage, dict[str, str]] = {
        FactorFiLMEvaluationStage.VALIDATION: {
            "validation_queue": config.validation_queue_fingerprint,
            "validation_schedule": config.validation_schedule_digest,
        },
        FactorFiLMEvaluationStage.SELECTION: {
            "validation": artifacts[FactorFiLMEvaluationStage.VALIDATION].artifact_fingerprint,
        },
        FactorFiLMEvaluationStage.RELOAD: {
            "selection": artifacts[FactorFiLMEvaluationStage.SELECTION].artifact_fingerprint,
        },
        FactorFiLMEvaluationStage.DEVELOPMENT: {
            "reload": artifacts[FactorFiLMEvaluationStage.RELOAD].artifact_fingerprint,
            "development_schedule": config.development_schedule_fingerprint,
        },
        FactorFiLMEvaluationStage.SEMANTIC: {
            "validation": artifacts[FactorFiLMEvaluationStage.VALIDATION].artifact_fingerprint,
            "development": artifacts[FactorFiLMEvaluationStage.DEVELOPMENT].artifact_fingerprint,
            "semantic_audit": config.semantic_audit_evidence_fingerprint,
        },
        FactorFiLMEvaluationStage.QUALITY_GATE: {
            "development": artifacts[FactorFiLMEvaluationStage.DEVELOPMENT].artifact_fingerprint,
            "semantic": artifacts[FactorFiLMEvaluationStage.SEMANTIC].artifact_fingerprint,
        },
    }
    for stage, expected in expected_inputs.items():
        if dict(artifacts[stage].input_artifact_fingerprints) != expected:
            raise FactorFiLMEvidenceError(f"{stage.value} stage input references are not canonical")

    validation_payload = artifacts[FactorFiLMEvaluationStage.VALIDATION].payload
    _require_exact_keys(
        validation_payload,
        {
            "schema_version",
            "validation_queue_fingerprint",
            "validation_schedule_digest",
            "checkpoint_count",
            "results",
            "benchmark_reports",
        },
        "validation stage payload",
    )
    if (
        validation_payload["schema_version"] != FACTOR_FILM_VALIDATION_EVIDENCE_SCHEMA_VERSION
        or validation_payload["validation_queue_fingerprint"] != config.validation_queue_fingerprint
        or validation_payload["validation_schedule_digest"] != config.validation_schedule_digest
        or validation_payload["checkpoint_count"] != 20
    ):
        raise FactorFiLMEvidenceError("validation stage disagrees with its fixed queue")
    raw_results = validation_payload["results"]
    if not isinstance(raw_results, tuple | list) or len(raw_results) != 20:
        raise FactorFiLMEvidenceError("validation stage requires exactly 20 results")
    validation_results = tuple(
        FactorFiLMValidationCountResult.from_dict(
            cast(Mapping[str, object], _require_mapping(item, "validation result"))
        )
        for item in raw_results
    )
    benchmark_reports = validation_payload["benchmark_reports"]
    if (
        not isinstance(benchmark_reports, tuple | list)
        or len(benchmark_reports) != 20
        or not all(isinstance(item, Mapping) for item in benchmark_reports)
    ):
        raise FactorFiLMEvidenceError(
            "validation stage requires 20 checksum-bound benchmark reports"
        )
    if any(
        value.schedule_digest != config.validation_schedule_digest for value in validation_results
    ):
        raise FactorFiLMEvidenceError("validation results use a foreign schedule")
    rank_factor_film_validation_count_results(validation_results)

    selection = FactorFiLMSelectionRecord.from_dict(
        cast(
            Mapping[str, object],
            artifacts[FactorFiLMEvaluationStage.SELECTION].payload,
        )
    )
    if (
        selection.run_fingerprint != config.run_fingerprint
        or selection.validation_queue_fingerprint != config.validation_queue_fingerprint
        or selection.validation_schedule_digest != config.validation_schedule_digest
        or tuple(value.to_dict() for value in selection.candidates)
        != tuple(value.to_selection_result().to_dict() for value in validation_results)
    ):
        raise FactorFiLMEvidenceError("selection differs from the immutable validation evidence")
    observed_queue = tuple(
        (
            value.checkpoint_fingerprint,
            value.checkpoint_step,
            float(value.offline_validation_action_loss),
        )
        for value in validation_results
    )
    expected_queue = tuple(
        (
            value.checkpoint_fingerprint,
            value.checkpoint_step,
            float(value.offline_validation_action_loss),
        )
        for value in selection.validation_queue.items
    )
    if observed_queue != expected_queue:
        raise FactorFiLMEvidenceError("validation evidence changed the published queue order")

    selected_fingerprint = selection.selected_checkpoint_fingerprint
    selected_step = selection.selected_checkpoint_step
    for stage in FACTOR_FILM_EVALUATION_STAGE_ORDER[1:]:
        artifact = artifacts[stage]
        if (
            artifact.selected_checkpoint_fingerprint != selected_fingerprint
            or artifact.selected_checkpoint_step != selected_step
        ):
            raise FactorFiLMEvidenceError("post-selection stages changed the selected checkpoint")

    reload_payload = artifacts[FactorFiLMEvaluationStage.RELOAD].payload
    if reload_payload.get("reload_validated") is not True:
        raise FactorFiLMEvidenceError("reload stage did not validate a fresh-instance reload")
    if (
        reload_payload.get("selected_checkpoint_fingerprint") != selected_fingerprint
        or reload_payload.get("selected_checkpoint_step") != selected_step
    ):
        raise FactorFiLMEvidenceError("reload stage changed the selected checkpoint")
    development_payload = artifacts[FactorFiLMEvaluationStage.DEVELOPMENT].payload
    development_reports = development_payload.get("benchmark_reports")
    if (
        development_payload.get("development_benchmark_completed") is not True
        or development_payload.get("physical_execution") is not True
        or development_payload.get("device") != "cuda"
        or development_payload.get("total_episode_count") != 216
        or not isinstance(development_reports, Mapping)
        or set(development_reports) != {"per_task", "state_onehot", "factor_film"}
        or not isinstance(development_reports.get("per_task"), tuple | list)
        or len(cast(Sequence[object], development_reports["per_task"])) != 6
        or not isinstance(development_reports.get("state_onehot"), Mapping)
        or not isinstance(development_reports.get("factor_film"), Mapping)
    ):
        raise FactorFiLMEvidenceError("development benchmark is incomplete")
    semantic_payload = artifacts[FactorFiLMEvaluationStage.SEMANTIC].payload
    required_completed_semantic_flags = (
        "semantic_evaluation_completed",
        "post_grasp_analysis_completed",
        "first_interaction_analysis_completed",
        "raw_action_metrics_validated",
        "runtime_action_metrics_validated",
    )
    if any(semantic_payload.get(name) is not True for name in required_completed_semantic_flags):
        raise FactorFiLMEvidenceError("semantic stage is incomplete")
    quality_semantic_flags = (
        "correct_task_retrieval_validated",
        "object_retrieval_validated",
        "bin_retrieval_validated",
    )
    if any(not isinstance(semantic_payload.get(name), bool) for name in quality_semantic_flags):
        raise FactorFiLMEvidenceError("semantic quality flags must be explicit booleans")
    gate_payload = artifacts[FactorFiLMEvaluationStage.QUALITY_GATE].payload
    gate_passed = gate_payload.get("development_quality_gate_passed")
    authorized = gate_payload.get("final_benchmark_authorized")
    if not isinstance(gate_passed, bool) or authorized is not gate_passed:
        raise FactorFiLMEvidenceError("quality gate and final authorization are inconsistent")
    if gate_payload.get("smolvla_go") is not False:
        raise FactorFiLMEvidenceError("M4.3b development cannot authorize SmolVLA")
    return artifacts


def _completion_payload(
    *,
    config: FactorFiLMEvaluationEvidenceConfig,
    artifacts: Mapping[FactorFiLMEvaluationStage, FactorFiLMEvaluationStageArtifact],
    evidence_fingerprint: str,
    manifest_sha256: str,
) -> dict[str, object]:
    _require_digest(evidence_fingerprint, "evidence_fingerprint")
    _require_digest(manifest_sha256, "manifest_sha256")
    gate_payload = artifacts[FactorFiLMEvaluationStage.QUALITY_GATE].payload
    gate_passed = cast(bool, gate_payload["development_quality_gate_passed"])
    return {
        "schema_version": FACTOR_FILM_EVIDENCE_COMPLETION_SCHEMA_VERSION,
        "config_fingerprint": config.fingerprint,
        "run_fingerprint": config.run_fingerprint,
        "evidence_fingerprint": evidence_fingerprint,
        "manifest_sha256": manifest_sha256,
        "stage_artifact_fingerprints": {
            stage.value: artifacts[stage].artifact_fingerprint
            for stage in FACTOR_FILM_EVALUATION_STAGE_ORDER
        },
        "validation_only_selection_validated": True,
        "factor_film_checkpoint_selected": True,
        "factor_film_reload_validated": True,
        "development_benchmark_completed": True,
        "semantic_evaluation_completed": True,
        "development_quality_gate_passed": gate_passed,
        "final_benchmark_authorized": gate_passed,
        "test_split_accessed": False,
        "fresh_seed_accessed": False,
        "final_schedule_accessed": False,
        "smolvla_go": False,
        "complete": True,
    }


def _reject_prohibited_sources(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _PROHIBITED_FLAG_KEYS and item is not False:
                raise FactorFiLMEvidenceError(f"prohibited source flag is not false: {key}")
            if key == "smolvla_go" and item is not False:
                raise FactorFiLMEvidenceError("development evidence cannot authorize SmolVLA")
            _reject_prohibited_sources(item)
        return
    if isinstance(value, tuple | list):
        for item in value:
            _reject_prohibited_sources(item)
        return
    if isinstance(value, str):
        lowered = value.lower()
        if any(fragment in lowered for fragment in _PROHIBITED_VALUE_FRAGMENTS):
            raise FactorFiLMEvidenceError("evidence contains a prohibited evaluation identity")


def _validate_portable_json(value: object, name: str) -> None:
    try:
        json.dumps(
            _json_value(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise FactorFiLMEvidenceError(f"{name} is not portable strict JSON: {error}") from error


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise FactorFiLMEvidenceError(f"{name} must be a JSON object")
    return cast(Mapping[str, object], value)


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _resolved_unlinked(path: Path, *, label: str) -> Path:
    lexical = _lexical_absolute(path)
    for component in (lexical, *lexical.parents):
        if component.is_symlink() or component.is_junction():
            raise FactorFiLMEvidenceError(f"{label} traverses a symlink or junction: {component}")
    return lexical.resolve(strict=False)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _overlaps(left: Path, right: Path) -> bool:
    return _is_within(left, right) or _is_within(right, left)


def _safe_evidence_root(output_root: Path, training_run_root: Path) -> Path:
    output = _resolved_unlinked(output_root, label="FactorFiLM evidence output root")
    training = _resolved_unlinked(training_run_root, label="FactorFiLM training run root")
    if _overlaps(output, training):
        raise FactorFiLMEvidenceError("evaluation output must not overlap the training run")
    if output.exists() and not output.is_dir():
        raise FactorFiLMEvidenceError("evaluation output root must be a real directory")
    output.mkdir(parents=True, exist_ok=True)
    evidence = _safe_child(
        output,
        FACTOR_FILM_EVIDENCE_DIRECTORY,
        "FactorFiLM evaluation evidence root",
    )
    if evidence.exists() and not evidence.is_dir():
        raise FactorFiLMEvidenceError("evaluation evidence root must be a real directory")
    evidence.mkdir(exist_ok=True)
    return evidence


def _validate_owned_staging_for_cleanup(
    staging: Path, config: FactorFiLMEvaluationEvidenceConfig
) -> None:
    owner = _read_object(staging / FACTOR_FILM_EVIDENCE_OWNER_FILE, "staging owner")
    if owner != {
        "schema_version": FACTOR_FILM_EVIDENCE_OWNER_SCHEMA_VERSION,
        "config_fingerprint": config.fingerprint,
        "config": config.to_dict(),
    }:
        raise FactorFiLMEvidenceError("refusing to clean staging with a malformed owner")
    allowed_files = {
        FACTOR_FILM_EVIDENCE_OWNER_FILE,
        FACTOR_FILM_EVIDENCE_CONFIG_FILE,
        FACTOR_FILM_EVIDENCE_MANIFEST_FILE,
        FACTOR_FILM_EVIDENCE_COMPLETION_FILE,
        *{
            f"{FACTOR_FILM_EVIDENCE_STAGE_DIRECTORY}/{stage.value}.json"
            for stage in FACTOR_FILM_EVALUATION_STAGE_ORDER
        },
    }
    if not _relative_files(staging) <= allowed_files:
        raise FactorFiLMEvidenceError("refusing to clean staging with an unowned artifact")


def _safe_child(parent: Path, name: str, label: str) -> Path:
    candidate = _resolved_unlinked(parent / name, label=label)
    if candidate.parent != parent.resolve():
        raise FactorFiLMEvidenceError(f"{label} escaped its owned root")
    return candidate


def _require_real_directory(path: Path, label: str) -> None:
    if path.is_symlink() or path.is_junction() or not path.is_dir():
        raise FactorFiLMEvidenceError(f"{label} must be one real directory: {path}")


def _write_new_json(path: Path, payload: object) -> None:
    _validate_portable_json(payload, f"artifact {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(
                _json_value(payload),
                stream,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise FactorFiLMEvidenceError(f"immutable artifact already exists: {path}") from error


def _read_object(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or path.is_junction() or not path.is_file():
        raise FactorFiLMEvidenceError(f"{label} is missing or unsafe: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: _raise_invalid_json_constant(value),
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise FactorFiLMEvidenceError(f"{label} is not valid UTF-8 JSON: {error}") from error
    if not isinstance(payload, dict):
        raise FactorFiLMEvidenceError(f"{label} must be a JSON object")
    return cast(dict[str, object], payload)


def _raise_invalid_json_constant(value: str) -> object:
    raise ValueError(f"invalid JSON constant {value}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _relative_files(root: Path) -> set[str]:
    files: set[str] = set()
    for candidate in root.rglob("*"):
        if candidate.is_symlink() or candidate.is_junction():
            raise FactorFiLMEvidenceError(f"evidence contains a linked entry: {candidate}")
        if candidate.is_file():
            files.add(PurePosixPath(candidate.relative_to(root)).as_posix())
        elif not candidate.is_dir():
            raise FactorFiLMEvidenceError(f"evidence contains an unsafe entry: {candidate}")
    return files


def _artifact_records(root: Path) -> list[dict[str, object]]:
    excluded = {
        FACTOR_FILM_EVIDENCE_MANIFEST_FILE,
        FACTOR_FILM_EVIDENCE_COMPLETION_FILE,
    }
    return [
        {
            "path": relative,
            "sha256": _sha256_file(root / PurePosixPath(relative)),
            "size_bytes": (root / PurePosixPath(relative)).stat().st_size,
        }
        for relative in sorted(_relative_files(root) - excluded)
    ]


def _validate_artifact_records(root: Path, manifest: Mapping[str, object]) -> None:
    records = manifest.get("artifacts")
    if not isinstance(records, list):
        raise FactorFiLMEvidenceError("manifest artifacts must be a JSON array")
    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in records:
        record = _require_mapping(raw, "artifact record")
        _require_exact_keys(record, {"path", "sha256", "size_bytes"}, "artifact record")
        relative = record["path"]
        digest = record["sha256"]
        size = record["size_bytes"]
        if not isinstance(relative, str) or not relative or relative in seen:
            raise FactorFiLMEvidenceError("artifact paths must be unique non-empty strings")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
            raise FactorFiLMEvidenceError("manifest contains an unsafe artifact path")
        if relative in {
            FACTOR_FILM_EVIDENCE_MANIFEST_FILE,
            FACTOR_FILM_EVIDENCE_COMPLETION_FILE,
        }:
            raise FactorFiLMEvidenceError("manifest and completion cannot checksum themselves")
        if not isinstance(digest, str):
            raise FactorFiLMEvidenceError("artifact checksum must be a string")
        _require_digest(digest, "artifact checksum")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise FactorFiLMEvidenceError("artifact size must be nonnegative")
        path = root.joinpath(*pure.parts)
        if path.is_symlink() or path.is_junction() or not path.is_file():
            raise FactorFiLMEvidenceError(f"checksummed artifact is missing: {relative}")
        if path.stat().st_size != size or _sha256_file(path) != digest:
            raise FactorFiLMEvidenceError(f"artifact checksum or size mismatch: {relative}")
        seen.add(relative)
        normalized.append({"path": relative, "sha256": digest, "size_bytes": size})
    if normalized != sorted(normalized, key=lambda item: cast(str, item["path"])):
        raise FactorFiLMEvidenceError("artifact records must use deterministic path order")
    actual = _relative_files(root) - {
        FACTOR_FILM_EVIDENCE_MANIFEST_FILE,
        FACTOR_FILM_EVIDENCE_COMPLETION_FILE,
    }
    if actual != seen:
        raise FactorFiLMEvidenceError("manifest records are incomplete or contain extras")


def _fsync_tree(root: Path) -> None:
    if os.name == "nt":
        return
    for directory, _, filenames in os.walk(root, topdown=False):
        for filename in filenames:
            with (Path(directory) / filename).open("rb") as stream:
                os.fsync(stream.fileno())
        _fsync_directory(Path(directory))


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "FACTOR_FILM_EVALUATION_STAGE_ORDER",
    "FACTOR_FILM_VALIDATION_EVIDENCE_SCHEMA_VERSION",
    "CompletedFactorFiLMEvaluationEvidence",
    "FactorFiLMEvaluationEvidenceConfig",
    "FactorFiLMEvaluationStage",
    "FactorFiLMEvaluationStageArtifact",
    "FactorFiLMEvidenceError",
    "stage_and_promote_factor_film_evaluation_evidence",
    "validate_completed_factor_film_evaluation_evidence",
]
