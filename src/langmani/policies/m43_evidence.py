"""Portable, immutable evidence lifecycle for the M4.3a semantic audit.

This module owns evidence identities and storage only.  It never loads a
dataset, checkpoint, policy, simulator, or evaluation schedule.  Callers pass
already-computed :class:`~langmani.policies.m43_types.SemanticAlignmentAudit`
records; this layer validates their scope, writes a content-bound staging
directory, verifies every artifact, and atomically promotes the directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import cast

from langmani.datasets.identity import canonical_json, sha256_hex
from langmani.environments.specs import BIN_IDS, OBJECT_IDS
from langmani.policies.act_semantic_audit import (
    summarize_bin_retrieval,
    summarize_object_retrieval,
    summarize_task_retrieval,
)
from langmani.policies.m42_types import GripperRuntimeMode
from langmani.policies.m43_types import (
    M43_CANONICAL_TASK_IDS,
    M43_SCHEMA_VERSION,
    PRIMARY_ACTION_CHUNK_DISTANCE_METRIC,
    ActionChunkDistanceConfig,
    ActionChunkDistanceResult,
    BinRetrievalResult,
    FirstInteractionRecord,
    ObjectRetrievalResult,
    SemanticAlignmentAudit,
    SemanticAuditConclusion,
    SemanticFailureClass,
    SemanticRetrievalResult,
    TaskRetrievalConfusion,
)

M43_AUDIT_CONFIG_SCHEMA_VERSION = "langmani-m43-semantic-audit-config-v0"
M43_AUDIT_SCOPE_SCHEMA_VERSION = "langmani-m43-semantic-audit-scope-v0"
M43_AUDIT_MANIFEST_SCHEMA_VERSION = "langmani-m43-semantic-audit-manifest-v0"
M43_AUDIT_COMPLETION_SCHEMA_VERSION = "langmani-m43-semantic-audit-complete-v0"
M43_AUDIT_OWNER_SCHEMA_VERSION = "langmani-m43-semantic-audit-owner-v0"
M43_VALIDATION_OBSERVATION_COUNT = 36
M43_DEVELOPMENT_OBSERVATION_COUNT = 72
M43_CANDIDATE_POLICY_LABELS = ("state_onehot", "task_token")
M43_TASK_TOKEN_M42_STATUS = "rejected"
M43_EVIDENCE_DIRECTORY = "evidence"
M43_SCOPE_DIRECTORY = "scopes"
M43_MANIFEST_FILE = "manifest.json"
M43_COMPLETION_FILE = "complete.json"
M43_CONFIG_FILE = "config.json"
M43_OWNER_FILE = "owner.json"
M43_HUMAN_SUMMARY_FILE = "summary.md"

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_FORBIDDEN_IDENTITY_FRAGMENTS = (
    "m3b_test",
    "test_split",
    "fresh_seed",
    "fresh-seed",
    "m4_fresh",
    "m42_final_v0",
)
_POLICY_SUMMARY_KEYS = {
    "full_task_top1_retrieval",
    "full_task_top2_retrieval",
    "mean_reciprocal_rank",
    "mean_correct_reference_margin",
    "target_object_retrieval",
    "destination_bin_retrieval",
    "conditional_bin_retrieval",
    "per_task_retrieval",
    "task_confusion_matrix",
    "object_confusion_matrix",
    "bin_confusion_matrix",
    "first_interaction_confusions",
    "first_interaction_evidence_available",
    "post_grasp_failure_distribution",
    "task_sensitivity_magnitude",
    "semantic_failure_classification",
    "deterministic_reload",
    "interpretation",
}


class M43EvidenceError(RuntimeError):
    """Raised when M4.3a evidence is unsafe, incomplete, or inconsistent."""


class M43AuditScope(StrEnum):
    """Only the two authorized sources and their explicit combined view."""

    M3B_VALIDATION = "m3b_validation"
    M42_DEVELOPMENT = "m42_dev_v0"
    COMBINED = "combined_validation_and_m42_dev_v0"


class M43ObservationSource(StrEnum):
    """Physical observation sources; a combined view is never a source."""

    M3B_VALIDATION = "m3b_validation"
    M42_DEVELOPMENT = "m42_dev_v0"


class _FrozenMapping(Mapping[str, object]):
    __slots__ = ("_items",)

    def __init__(self, value: Mapping[str, object]) -> None:
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


class _FrozenObjectMapping(Mapping[object, object]):
    """Immutable mapping that preserves typed values until JSON serialization."""

    __slots__ = ("_items",)

    def __init__(self, value: Mapping[object, object]) -> None:
        self._items = tuple(value.items())

    def __getitem__(self, key: object) -> object:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __deepcopy__(self, memo: object) -> _FrozenObjectMapping:
        del memo
        return self


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("M4.3 evidence mappings require string keys")
        return _FrozenMapping(cast(Mapping[str, object], value))
    if isinstance(value, tuple | list):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, bool | int | float | str | StrEnum):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _freeze_json(to_dict())
    raise TypeError(f"unsupported M4.3 evidence value {type(value).__name__}")


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
    raise TypeError(f"unsupported M4.3 evidence JSON value {type(value).__name__}")


class _JsonRecord:
    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], _json_value(asdict(self)))

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())


def _fingerprint(value: object) -> str:
    return f"sha256:{sha256_hex(value)}"


def _require_digest(value: str, name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise M43EvidenceError(f"{name} must be sha256:<64 lowercase hex>")


def _require_nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise M43EvidenceError(f"{name} must be a non-empty string")


def _require_nonnegative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise M43EvidenceError(f"{name} must be a non-negative integer")


def _require_finite(value: float, name: str, *, minimum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise M43EvidenceError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise M43EvidenceError(f"{name} must be >= {minimum}")


def _is_absolute_text(value: str) -> bool:
    return (
        value.startswith(("file://", "file:\\\\", "\\\\"))
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
    )


def _validate_portable_value(value: object, name: str) -> None:
    """Reject machine paths, sealed-source identities, NaN, and non-JSON values."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise M43EvidenceError(f"{name} contains a non-string mapping key")
            _validate_portable_value(item, f"{name}.{key}")
        return
    if isinstance(value, tuple | list):
        for index, item in enumerate(value):
            _validate_portable_value(item, f"{name}[{index}]")
        return
    if isinstance(value, StrEnum):
        value = value.value
    if isinstance(value, str):
        lowered = value.lower()
        if _is_absolute_text(value):
            raise M43EvidenceError(f"{name} contains a machine-specific absolute path")
        if any(fragment in lowered for fragment in _FORBIDDEN_IDENTITY_FRAGMENTS):
            raise M43EvidenceError(f"{name} contains a prohibited test, fresh, or final identity")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise M43EvidenceError(f"{name} contains NaN or infinity")
    if value is None or isinstance(value, bool | int | float):
        return
    raise M43EvidenceError(f"{name} contains unsupported value {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class ObservationSourceIdentity(_JsonRecord):
    """Portable identity of one fixed RGB/state policy observation."""

    source: M43ObservationSource
    observation_id: str
    scene_id: str
    scene_group_id: str | None
    scene_seed: int
    source_task_id: str
    source_episode_id: str | None
    frame_index: int
    image_fingerprint: str
    policy_state_fingerprint: str
    queried_task_id: str
    policy_state_dimension: int = 9

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", M43ObservationSource(self.source))
        for name in ("observation_id", "scene_id"):
            _require_nonempty(cast(str, getattr(self, name)), name)
        for name in ("scene_group_id", "source_episode_id"):
            value = cast(str | None, getattr(self, name))
            if value is not None:
                _require_nonempty(value, name)
        _require_nonnegative_integer(self.scene_seed, "scene_seed")
        _require_nonnegative_integer(self.frame_index, "frame_index")
        if self.source_task_id not in M43_CANONICAL_TASK_IDS:
            raise M43EvidenceError("source_task_id must use canonical TaskSpec ordering")
        if self.queried_task_id not in M43_CANONICAL_TASK_IDS:
            raise M43EvidenceError("queried_task_id must use canonical TaskSpec ordering")
        if self.policy_state_dimension != 9:
            raise M43EvidenceError("M4.3a requires the unchanged 9D PandaPolicyStateV0")
        _require_digest(self.image_fingerprint, "image_fingerprint")
        _require_digest(self.policy_state_fingerprint, "policy_state_fingerprint")
        _validate_portable_value(self.to_dict(), "observation identity")


@dataclass(frozen=True, slots=True)
class M43AuditConfig(_JsonRecord):
    """Content-only semantic identity for one complete M4.3a audit."""

    scope: M43AuditScope
    git_commit: str
    m3b_fingerprint: str
    m3b_split_manifest_digest: str
    per_task_checkpoint_fingerprints: Mapping[str, object]
    state_onehot_checkpoint_fingerprint: str
    task_token_checkpoint_fingerprint: str
    selected_execution_horizon: int
    selected_gripper_runtime: GripperRuntimeMode
    distance_config: ActionChunkDistanceConfig
    ordered_observations: tuple[ObservationSourceIdentity, ...]
    task_token_m42_status: str = M43_TASK_TOKEN_M42_STATUS
    candidate_policy_labels: tuple[str, ...] = M43_CANDIDATE_POLICY_LABELS
    schema_version: str = M43_AUDIT_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", M43AuditScope(self.scope))
        object.__setattr__(
            self, "selected_gripper_runtime", GripperRuntimeMode(self.selected_gripper_runtime)
        )
        if self.schema_version != M43_AUDIT_CONFIG_SCHEMA_VERSION:
            raise M43EvidenceError("unknown M4.3 audit-config schema")
        if _GIT_COMMIT_RE.fullmatch(self.git_commit) is None:
            raise M43EvidenceError("git_commit must be one full lowercase Git SHA-1")
        for name in (
            "m3b_fingerprint",
            "m3b_split_manifest_digest",
            "state_onehot_checkpoint_fingerprint",
            "task_token_checkpoint_fingerprint",
        ):
            _require_digest(cast(str, getattr(self, name)), name)
        checkpoint_fingerprints = dict(self.per_task_checkpoint_fingerprints)
        if set(checkpoint_fingerprints) != set(M43_CANONICAL_TASK_IDS):
            raise M43EvidenceError(
                "per_task_checkpoint_fingerprints must contain six references in canonical order"
            )
        for task_id, fingerprint in checkpoint_fingerprints.items():
            if not isinstance(fingerprint, str):
                raise M43EvidenceError(f"PerTask checkpoint for {task_id} is not a string")
            _require_digest(fingerprint, f"per_task_checkpoint_fingerprints[{task_id}]")
        object.__setattr__(
            self, "per_task_checkpoint_fingerprints", _FrozenMapping(checkpoint_fingerprints)
        )
        if self.selected_execution_horizon not in (1, 5, 10):
            raise M43EvidenceError("selected_execution_horizon must be exactly 1, 5, or 10")
        if self.distance_config.locked_execution_horizon != self.selected_execution_horizon:
            raise M43EvidenceError("distance config must use the selected M4.2 execution horizon")
        if self.distance_config.primary_metric != PRIMARY_ACTION_CHUNK_DISTANCE_METRIC:
            raise M43EvidenceError("M4.3a primary retrieval metric changed")
        if tuple(self.distance_config.arm_action_indices) != tuple(range(7)):
            raise M43EvidenceError("primary retrieval must exclude the gripper dimension")
        if self.task_token_m42_status != M43_TASK_TOKEN_M42_STATUS:
            raise M43EvidenceError("TaskToken must remain marked rejected from M4.2")
        if self.candidate_policy_labels != M43_CANDIDATE_POLICY_LABELS:
            raise M43EvidenceError("M4.3a candidate policy labels are immutable")
        _validate_observation_set(self.scope, self.ordered_observations)
        _validate_portable_value(self.to_dict(), "audit config")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> M43AuditConfig:
        _require_exact_keys(
            value,
            {
                "scope",
                "git_commit",
                "m3b_fingerprint",
                "m3b_split_manifest_digest",
                "per_task_checkpoint_fingerprints",
                "state_onehot_checkpoint_fingerprint",
                "task_token_checkpoint_fingerprint",
                "selected_execution_horizon",
                "selected_gripper_runtime",
                "distance_config",
                "ordered_observations",
                "task_token_m42_status",
                "candidate_policy_labels",
                "schema_version",
            },
            "audit config",
        )
        observations = value["ordered_observations"]
        if not isinstance(observations, list):
            raise M43EvidenceError("ordered_observations must be a JSON array")
        checkpoint_map = _require_mapping(
            value["per_task_checkpoint_fingerprints"], "PerTask checkpoint fingerprints"
        )
        return cls(
            scope=M43AuditScope(cast(str, value["scope"])),
            git_commit=cast(str, value["git_commit"]),
            m3b_fingerprint=cast(str, value["m3b_fingerprint"]),
            m3b_split_manifest_digest=cast(str, value["m3b_split_manifest_digest"]),
            per_task_checkpoint_fingerprints=checkpoint_map,
            state_onehot_checkpoint_fingerprint=cast(
                str, value["state_onehot_checkpoint_fingerprint"]
            ),
            task_token_checkpoint_fingerprint=cast(str, value["task_token_checkpoint_fingerprint"]),
            selected_execution_horizon=cast(int, value["selected_execution_horizon"]),
            selected_gripper_runtime=GripperRuntimeMode(
                cast(str, value["selected_gripper_runtime"])
            ),
            distance_config=_parse_distance_config(
                _require_mapping(value["distance_config"], "distance config")
            ),
            ordered_observations=tuple(
                _parse_observation_identity(
                    _require_mapping(item, f"ordered_observations[{index}]")
                )
                for index, item in enumerate(observations)
            ),
            task_token_m42_status=cast(str, value["task_token_m42_status"]),
            candidate_policy_labels=tuple(cast(list[str], value["candidate_policy_labels"])),
            schema_version=cast(str, value["schema_version"]),
        )

    @property
    def required_artifact_scopes(self) -> tuple[M43AuditScope, ...]:
        if self.scope is M43AuditScope.COMBINED:
            return (
                M43AuditScope.M3B_VALIDATION,
                M43AuditScope.M42_DEVELOPMENT,
                M43AuditScope.COMBINED,
            )
        return (self.scope,)

    def observations_for_scope(self, scope: M43AuditScope) -> tuple[ObservationSourceIdentity, ...]:
        if scope is M43AuditScope.COMBINED:
            return self.ordered_observations
        source = M43ObservationSource(scope.value)
        return tuple(item for item in self.ordered_observations if item.source is source)


@dataclass(frozen=True, slots=True)
class M43AuditScopeArtifact(_JsonRecord):
    """One explicitly labeled validation, development, or combined result."""

    scope: M43AuditScope
    ordered_observation_ids: tuple[str, ...]
    policy_audits: Mapping[str, SemanticAlignmentAudit]
    policy_summaries: Mapping[str, object]
    first_interaction_evidence_available: bool
    rollout_evidence: Mapping[str, object]
    observation_identity_held_fixed: bool = True
    policy_state_held_fixed: bool = True
    policy_reset_before_each_inference: bool = True
    deterministic_inference_validated: bool = True
    primary_retrieval_excludes_gripper: bool = True
    primary_retrieval_uses_locked_horizon: bool = True
    canonical_task_tie_breaking_validated: bool = True
    task_token_m42_status: str = M43_TASK_TOKEN_M42_STATUS
    test_split_accessed: bool = False
    fresh_seed_accessed: bool = False
    final_schedule_accessed: bool = False
    final_benchmark_authorized: bool = False
    schema_version: str = M43_AUDIT_SCOPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", M43AuditScope(self.scope))
        if self.schema_version != M43_AUDIT_SCOPE_SCHEMA_VERSION:
            raise M43EvidenceError("unknown M4.3 audit-scope schema")
        if not self.ordered_observation_ids or len(set(self.ordered_observation_ids)) != len(
            self.ordered_observation_ids
        ):
            raise M43EvidenceError("ordered_observation_ids must be non-empty and unique")
        for value in self.ordered_observation_ids:
            _require_nonempty(value, "ordered_observation_ids entry")
        audits = dict(self.policy_audits)
        summaries = dict(self.policy_summaries)
        if (
            tuple(audits) != M43_CANDIDATE_POLICY_LABELS
            or tuple(summaries) != M43_CANDIDATE_POLICY_LABELS
        ):
            raise M43EvidenceError(
                "scope evidence must contain State-OneHot then TaskToken exactly once"
            )
        rollout_evidence = _validate_rollout_evidence(self.scope, self.rollout_evidence)
        external_distributions = cast(
            Mapping[str, object], rollout_evidence["per_policy_post_grasp_failure_distribution"]
        )
        for label, audit in audits.items():
            if not isinstance(audit, SemanticAlignmentAudit):
                raise TypeError(f"policy_audits[{label}] must be SemanticAlignmentAudit")
            if audit.policy_label != label:
                raise M43EvidenceError("policy audit label differs from its stable mapping key")
            if audit.deterministic_reset_validated is not True:
                raise M43EvidenceError("policy audit lacks deterministic reset evidence")
            _validate_audit_observations(audit, self.ordered_observation_ids)
            summary = _require_mapping(summaries[label], f"policy_summaries[{label}]")
            external = external_distributions.get(label)
            _validate_policy_summary(
                audit,
                summary,
                external_failure_distribution=(
                    None
                    if external is None
                    else _require_mapping(external, f"rollout failure distribution {label}")
                ),
            )
        interaction_id_sets = {
            tuple(item.observation_id for item in audit.first_interactions)
            for audit in audits.values()
        }
        if len(interaction_id_sets) != 1:
            raise M43EvidenceError(
                "candidate policies must bind the same available first-interaction evidence"
            )
        interaction_ids = next(iter(interaction_id_sets))
        if self.first_interaction_evidence_available != bool(interaction_ids):
            raise M43EvidenceError(
                "first_interaction_evidence_available disagrees with recorded interactions"
            )
        object.__setattr__(self, "policy_audits", _FrozenObjectMapping(audits))
        object.__setattr__(self, "policy_summaries", _FrozenMapping(summaries))
        object.__setattr__(self, "rollout_evidence", rollout_evidence)
        required_true = (
            self.observation_identity_held_fixed,
            self.policy_state_held_fixed,
            self.policy_reset_before_each_inference,
            self.deterministic_inference_validated,
            self.primary_retrieval_excludes_gripper,
            self.primary_retrieval_uses_locked_horizon,
            self.canonical_task_tie_breaking_validated,
        )
        if not all(value is True for value in required_true):
            raise M43EvidenceError("scope evidence lacks a required deterministic audit check")
        if self.task_token_m42_status != M43_TASK_TOKEN_M42_STATUS:
            raise M43EvidenceError("TaskToken must remain marked rejected")
        if any(
            value is not False
            for value in (
                self.test_split_accessed,
                self.fresh_seed_accessed,
                self.final_schedule_accessed,
                self.final_benchmark_authorized,
            )
        ):
            raise M43EvidenceError("audit evidence cannot access or authorize sealed evaluation")
        _validate_portable_value(self.to_dict(), "scope artifact")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> M43AuditScopeArtifact:
        _require_exact_keys(
            value,
            {
                "scope",
                "ordered_observation_ids",
                "policy_audits",
                "policy_summaries",
                "first_interaction_evidence_available",
                "rollout_evidence",
                "observation_identity_held_fixed",
                "policy_state_held_fixed",
                "policy_reset_before_each_inference",
                "deterministic_inference_validated",
                "primary_retrieval_excludes_gripper",
                "primary_retrieval_uses_locked_horizon",
                "canonical_task_tie_breaking_validated",
                "task_token_m42_status",
                "test_split_accessed",
                "fresh_seed_accessed",
                "final_schedule_accessed",
                "final_benchmark_authorized",
                "schema_version",
            },
            "scope artifact",
        )
        audit_values = _require_mapping(value["policy_audits"], "policy audits")
        summary_values = _require_mapping(value["policy_summaries"], "policy summaries")
        ids = value["ordered_observation_ids"]
        if not isinstance(ids, list):
            raise M43EvidenceError("ordered_observation_ids must be a JSON array")
        return cls(
            scope=M43AuditScope(cast(str, value["scope"])),
            ordered_observation_ids=tuple(cast(list[str], ids)),
            policy_audits={
                label: _parse_semantic_alignment_audit(
                    _require_mapping(audit_values[label], f"policy audit {label}")
                )
                for label in M43_CANDIDATE_POLICY_LABELS
                if label in audit_values
            },
            policy_summaries={
                label: _require_mapping(summary_values[label], f"policy summary {label}")
                for label in M43_CANDIDATE_POLICY_LABELS
                if label in summary_values
            },
            first_interaction_evidence_available=cast(
                bool, value["first_interaction_evidence_available"]
            ),
            rollout_evidence=_require_mapping(value["rollout_evidence"], "rollout evidence"),
            observation_identity_held_fixed=cast(bool, value["observation_identity_held_fixed"]),
            policy_state_held_fixed=cast(bool, value["policy_state_held_fixed"]),
            policy_reset_before_each_inference=cast(
                bool, value["policy_reset_before_each_inference"]
            ),
            deterministic_inference_validated=cast(
                bool, value["deterministic_inference_validated"]
            ),
            primary_retrieval_excludes_gripper=cast(
                bool, value["primary_retrieval_excludes_gripper"]
            ),
            primary_retrieval_uses_locked_horizon=cast(
                bool, value["primary_retrieval_uses_locked_horizon"]
            ),
            canonical_task_tie_breaking_validated=cast(
                bool, value["canonical_task_tie_breaking_validated"]
            ),
            task_token_m42_status=cast(str, value["task_token_m42_status"]),
            test_split_accessed=cast(bool, value["test_split_accessed"]),
            fresh_seed_accessed=cast(bool, value["fresh_seed_accessed"]),
            final_schedule_accessed=cast(bool, value["final_schedule_accessed"]),
            final_benchmark_authorized=cast(bool, value["final_benchmark_authorized"]),
            schema_version=cast(str, value["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class CompletedM43AuditEvidence:
    """Integrity-checked completed evidence plus its local directory."""

    root: Path
    config: M43AuditConfig
    scope_artifacts: Mapping[M43AuditScope, M43AuditScopeArtifact]
    manifest: Mapping[str, object]
    completion: Mapping[str, object]

    @property
    def evidence_fingerprint(self) -> str:
        return cast(str, self.completion["evidence_fingerprint"])


def build_policy_semantic_summary(
    audit: SemanticAlignmentAudit,
    *,
    task_sensitivity_magnitude: float,
    external_failure_distribution: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the complete machine summary for one candidate policy."""
    _require_finite(task_sensitivity_magnitude, "task_sensitivity_magnitude", minimum=0.0)
    task = summarize_task_retrieval(audit.task_retrieval)
    object_result = summarize_object_retrieval(audit.object_retrieval)
    bin_result = summarize_bin_retrieval(audit.bin_retrieval)
    conclusions = {value.value for value in audit.conclusions}
    full_task_consistent = all(value.top1_correct for value in audit.task_retrieval)
    object_consistent = all(value.correct for value in audit.object_retrieval)
    bin_consistent = all(
        value.correct and value.conditional_correct for value in audit.bin_retrieval
    )
    correct_semantic_alignment = full_task_consistent and object_consistent and bin_consistent
    incompatible_alignment_conclusions = {
        SemanticAuditConclusion.CONDITION_NOT_USED.value,
        SemanticAuditConclusion.CONDITION_CHANGES_OUTPUT_BUT_WRONG_SEMANTICS.value,
        SemanticAuditConclusion.TARGET_OBJECT_CONFUSION.value,
        SemanticAuditConclusion.DESTINATION_BIN_CONFUSION.value,
        SemanticAuditConclusion.INSUFFICIENT_EVIDENCE.value,
    }
    if correct_semantic_alignment and conclusions & incompatible_alignment_conclusions:
        raise M43EvidenceError(
            "semantic conclusions contradict exact full-task/object/bin retrieval"
        )
    failure_distribution = (
        dict(audit.failure_distribution)
        if external_failure_distribution is None
        else _validate_failure_distribution(external_failure_distribution)
    )
    summary = {
        "full_task_top1_retrieval": task["top1_accuracy"],
        "full_task_top2_retrieval": task["top2_accuracy"],
        "mean_reciprocal_rank": task["mean_reciprocal_rank"],
        "mean_correct_reference_margin": task["mean_correct_margin"],
        "target_object_retrieval": object_result["accuracy"],
        "destination_bin_retrieval": bin_result["accuracy"],
        "conditional_bin_retrieval": bin_result["conditional_accuracy"],
        "per_task_retrieval": task["per_task"],
        "task_confusion_matrix": audit.task_confusion.to_dict(),
        "object_confusion_matrix": audit.object_confusion.to_dict(),
        "bin_confusion_matrix": audit.bin_confusion.to_dict(),
        "first_interaction_confusions": dict(audit.first_interaction_confusions),
        "first_interaction_evidence_available": bool(audit.first_interactions),
        "post_grasp_failure_distribution": failure_distribution,
        "task_sensitivity_magnitude": float(task_sensitivity_magnitude),
        "semantic_failure_classification": {
            "conclusions": [value.value for value in audit.conclusions],
            "rollout_failure_distribution": failure_distribution,
        },
        "deterministic_reload": {
            "validated": audit.deterministic_reload_validated,
            "available": audit.deterministic_reload_validated,
            "reason": (
                None
                if audit.deterministic_reload_validated
                else "fresh checkpoint instances were not reloaded by this audit"
            ),
        },
        "interpretation": {
            "output_changes_across_task_conditions": task_sensitivity_magnitude > 0.0,
            "correct_semantic_alignment": correct_semantic_alignment,
            "full_task_retrieval_consistent": full_task_consistent,
            "object_retrieval_consistent": object_consistent,
            "bin_retrieval_consistent": bin_consistent,
            "target_object_confusion": not object_consistent,
            "destination_bin_confusion": not bin_consistent,
            "post_grasp_control_failure": (
                SemanticAuditConclusion.POST_GRASP_EXECUTION_FAILURE.value in conclusions
                or SemanticAuditConclusion.SHARED_CONTROL_DEGRADATION_AFTER_CORRECT_SELECTION.value
                in conclusions
            ),
            "nonzero_action_distance_is_not_semantic_correctness": True,
        },
    }
    _validate_portable_value(summary, "policy summary")
    return summary


def unavailable_rollout_evidence() -> dict[str, object]:
    """Return the explicit no-rollout aggregate contract for offline-only scopes."""
    return {
        "available": False,
        "source_fingerprint": None,
        "per_policy_post_grasp_failure_distribution": {},
        "per_policy_metrics": {},
    }


def build_rollout_aggregate_evidence(
    *,
    source_fingerprint: str,
    per_policy_post_grasp_failure_distribution: Mapping[str, Mapping[str, object]],
    per_policy_metrics: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Bind real aggregate rollout diagnostics without inventing interaction traces."""
    _require_digest(source_fingerprint, "rollout source fingerprint")
    if tuple(per_policy_post_grasp_failure_distribution) != M43_CANDIDATE_POLICY_LABELS:
        raise M43EvidenceError("rollout failure distributions require both candidate policies")
    if tuple(per_policy_metrics) != M43_CANDIDATE_POLICY_LABELS:
        raise M43EvidenceError("rollout metrics require both candidate policies")
    payload = {
        "available": True,
        "source_fingerprint": source_fingerprint,
        "per_policy_post_grasp_failure_distribution": {
            label: _validate_failure_distribution(per_policy_post_grasp_failure_distribution[label])
            for label in M43_CANDIDATE_POLICY_LABELS
        },
        "per_policy_metrics": {
            label: dict(per_policy_metrics[label]) for label in M43_CANDIDATE_POLICY_LABELS
        },
    }
    _validate_portable_value(payload, "rollout aggregate evidence")
    return payload


def render_human_audit_summary(
    config: M43AuditConfig,
    scope_artifacts: Mapping[M43AuditScope, M43AuditScopeArtifact],
) -> str:
    """Render a concise deterministic Markdown summary without local paths."""
    lines = [
        "# LangMani M4.3a semantic-alignment audit",
        "",
        f"- Audit fingerprint: `{config.fingerprint}`",
        f"- Requested scope: `{config.scope.value}`",
        f"- Primary metric: `{config.distance_config.primary_metric}`",
        f"- Locked execution horizon: `{config.selected_execution_horizon}`",
        f"- Selected gripper runtime: `{config.selected_gripper_runtime.value}`",
        f"- TaskToken M4.2 status: `{config.task_token_m42_status}`",
        "- Sealed evaluation sources accessed: `false`",
    ]
    for scope in config.required_artifact_scopes:
        artifact = scope_artifacts[scope]
        lines.extend(("", f"## {scope.value}", ""))
        lines.append(f"Observations: {len(artifact.ordered_observation_ids)}")
        lines.append(
            "First-interaction evidence available: "
            f"{str(artifact.first_interaction_evidence_available).lower()}"
        )
        lines.append(
            "Aggregate rollout evidence available: "
            f"{str(bool(artifact.rollout_evidence['available'])).lower()}"
        )
        for policy_label in M43_CANDIDATE_POLICY_LABELS:
            summary = cast(Mapping[str, object], artifact.policy_summaries[policy_label])
            lines.extend(
                (
                    "",
                    f"### {policy_label}",
                    "",
                    f"- Full-task top-1: {float(summary['full_task_top1_retrieval']):.6f}",
                    f"- Full-task top-2: {float(summary['full_task_top2_retrieval']):.6f}",
                    f"- Mean reciprocal rank: {float(summary['mean_reciprocal_rank']):.6f}",
                    f"- Mean correct-reference margin: {float(summary['mean_correct_reference_margin']):.6f}",
                    f"- Target-object retrieval: {float(summary['target_object_retrieval']):.6f}",
                    f"- Destination-bin retrieval: {float(summary['destination_bin_retrieval']):.6f}",
                    f"- Conditional-bin retrieval: {float(summary['conditional_bin_retrieval']):.6f}",
                    f"- Task sensitivity magnitude: {float(summary['task_sensitivity_magnitude']):.6f}",
                    "- Deterministic fresh-instance reload validated: "
                    f"{str(bool(cast(Mapping[str, object], summary['deterministic_reload'])['validated'])).lower()}",
                    "- First-interaction records: unavailable unless explicitly recorded above",
                    f"- Per-TaskSpec retrieval: `{canonical_json(summary['per_task_retrieval'])}`",
                    f"- Task confusion: `{canonical_json(summary['task_confusion_matrix'])}`",
                    f"- Object confusion: `{canonical_json(summary['object_confusion_matrix'])}`",
                    f"- Bin confusion: `{canonical_json(summary['bin_confusion_matrix'])}`",
                    "- Post-grasp distribution: `"
                    f"{canonical_json(summary['post_grasp_failure_distribution'])}`",
                    "- Semantic classification: `"
                    f"{canonical_json(summary['semantic_failure_classification'])}`",
                )
            )
    text = "\n".join(lines) + "\n"
    _validate_portable_value(text, "human summary")
    return text


def stage_and_promote_audit_evidence(
    *,
    output_root: str | Path,
    config: M43AuditConfig,
    scope_artifacts: Mapping[M43AuditScope, M43AuditScopeArtifact],
    clean_matching_staging: bool = False,
) -> CompletedM43AuditEvidence:
    """Validate, stage, checksum, and atomically promote one audit bundle."""
    if not isinstance(config, M43AuditConfig):
        raise TypeError("config must be M43AuditConfig")
    artifacts = _validate_scope_artifact_set(config, scope_artifacts)
    evidence_root = _safe_evidence_root(Path(output_root))
    token = config.fingerprint.removeprefix("sha256:")
    destination = _safe_child(evidence_root, token, "completed audit evidence")
    staging = _safe_child(evidence_root, f".staging-{token}", "audit staging")

    if destination.exists():
        return validate_completed_audit_evidence(destination, expected_config=config)
    if staging.exists():
        _require_real_directory(staging, "audit staging")
        owner = _read_object(staging / M43_OWNER_FILE, "staging owner")
        if owner.get("config_fingerprint") != config.fingerprint:
            raise M43EvidenceError("refusing to clean staging owned by another audit config")
        if not clean_matching_staging:
            raise M43EvidenceError(
                "matching incomplete staging exists; explicit clean_matching_staging is required"
            )
        shutil.rmtree(staging)
    staging.mkdir(parents=False, exist_ok=False)
    _write_new_json(
        staging / M43_OWNER_FILE,
        {
            "schema_version": M43_AUDIT_OWNER_SCHEMA_VERSION,
            "config_fingerprint": config.fingerprint,
            "config": config.to_dict(),
        },
    )
    _write_new_json(staging / M43_CONFIG_FILE, config.to_dict())
    scopes_root = staging / M43_SCOPE_DIRECTORY
    scopes_root.mkdir()
    for scope in config.required_artifact_scopes:
        _write_new_json(scopes_root / f"{scope.value}.json", artifacts[scope].to_dict())
    _write_new_text(
        staging / M43_HUMAN_SUMMARY_FILE,
        render_human_audit_summary(config, artifacts),
    )

    artifact_records = _artifact_records(staging)
    scope_fingerprints = {
        scope.value: artifacts[scope].fingerprint for scope in config.required_artifact_scopes
    }
    evidence_fingerprint = _fingerprint(
        {
            "schema_version": M43_AUDIT_MANIFEST_SCHEMA_VERSION,
            "audit_schema_version": M43_SCHEMA_VERSION,
            "config_fingerprint": config.fingerprint,
            "scope_artifact_fingerprints": scope_fingerprints,
            "artifacts": artifact_records,
        }
    )
    manifest = {
        "schema_version": M43_AUDIT_MANIFEST_SCHEMA_VERSION,
        "audit_schema_version": M43_SCHEMA_VERSION,
        "config_fingerprint": config.fingerprint,
        "scope_artifact_fingerprints": scope_fingerprints,
        "artifacts": artifact_records,
        "evidence_fingerprint": evidence_fingerprint,
    }
    _write_new_json(staging / M43_MANIFEST_FILE, manifest)
    completion = _completion_payload(
        config=config,
        evidence_fingerprint=evidence_fingerprint,
        manifest_sha256=_sha256_file(staging / M43_MANIFEST_FILE),
    )
    _write_new_json(staging / M43_COMPLETION_FILE, completion)
    _fsync_tree(staging)
    validate_completed_audit_evidence(staging, expected_config=config)

    if destination.exists():
        raise M43EvidenceError("completed evidence destination appeared during promotion")
    if staging.stat().st_dev != evidence_root.stat().st_dev:
        raise M43EvidenceError("audit staging and destination are on different filesystems")
    try:
        os.replace(staging, destination)
    except OSError as error:
        raise M43EvidenceError(f"atomic audit-evidence promotion failed: {error}") from error
    _fsync_directory(evidence_root)
    return validate_completed_audit_evidence(destination, expected_config=config)


def validate_completed_audit_evidence(
    path: str | Path, *, expected_config: M43AuditConfig | None = None
) -> CompletedM43AuditEvidence:
    """Independently validate schemas, content bindings, checksums, and layout."""
    root = _resolved_unlinked(Path(path), label="completed M4.3a evidence")
    _require_real_directory(root, "completed M4.3a evidence")
    config_payload = _read_object(root / M43_CONFIG_FILE, "audit config")
    config = M43AuditConfig.from_dict(config_payload)
    if expected_config is not None and config.to_dict() != expected_config.to_dict():
        raise M43EvidenceError("completed audit config differs from the expected config")
    owner = _read_object(root / M43_OWNER_FILE, "audit owner")
    expected_owner = {
        "schema_version": M43_AUDIT_OWNER_SCHEMA_VERSION,
        "config_fingerprint": config.fingerprint,
        "config": config.to_dict(),
    }
    if owner != expected_owner:
        raise M43EvidenceError("audit owner is not content-bound to its config")
    if root.name not in {
        config.fingerprint.removeprefix("sha256:"),
        f".staging-{config.fingerprint.removeprefix('sha256:')}",
    }:
        raise M43EvidenceError("audit evidence directory is not owned by its config fingerprint")

    manifest = _read_object(root / M43_MANIFEST_FILE, "audit manifest")
    completion = _read_object(root / M43_COMPLETION_FILE, "audit completion marker")
    _validate_artifact_records(root, manifest)

    scope_artifacts: dict[M43AuditScope, M43AuditScopeArtifact] = {}
    expected_scope_files = {
        f"{M43_SCOPE_DIRECTORY}/{scope.value}.json" for scope in config.required_artifact_scopes
    }
    actual_scope_files = {
        PurePosixPath(path.relative_to(root)).as_posix()
        for path in (root / M43_SCOPE_DIRECTORY).glob("*.json")
    }
    if actual_scope_files != expected_scope_files:
        raise M43EvidenceError("validation and development scope artifacts are missing or mixed")
    for scope in config.required_artifact_scopes:
        artifact_payload = _read_object(
            root / M43_SCOPE_DIRECTORY / f"{scope.value}.json",
            f"{scope.value} scope artifact",
        )
        artifact = M43AuditScopeArtifact.from_dict(artifact_payload)
        if artifact.scope is not scope:
            raise M43EvidenceError("scope artifact filename and declared scope disagree")
        scope_artifacts[scope] = artifact
    artifacts = _validate_scope_artifact_set(config, scope_artifacts)
    expected_scope_fingerprints = {
        scope.value: artifacts[scope].fingerprint for scope in config.required_artifact_scopes
    }
    expected_manifest_keys = {
        "schema_version",
        "audit_schema_version",
        "config_fingerprint",
        "scope_artifact_fingerprints",
        "artifacts",
        "evidence_fingerprint",
    }
    _require_exact_keys(manifest, expected_manifest_keys, "audit manifest")
    expected_evidence_fingerprint = _fingerprint(
        {
            "schema_version": M43_AUDIT_MANIFEST_SCHEMA_VERSION,
            "audit_schema_version": M43_SCHEMA_VERSION,
            "config_fingerprint": config.fingerprint,
            "scope_artifact_fingerprints": expected_scope_fingerprints,
            "artifacts": manifest["artifacts"],
        }
    )
    if manifest != {
        "schema_version": M43_AUDIT_MANIFEST_SCHEMA_VERSION,
        "audit_schema_version": M43_SCHEMA_VERSION,
        "config_fingerprint": config.fingerprint,
        "scope_artifact_fingerprints": expected_scope_fingerprints,
        "artifacts": manifest["artifacts"],
        "evidence_fingerprint": expected_evidence_fingerprint,
    }:
        raise M43EvidenceError("audit manifest is not content-bound to its typed artifacts")
    expected_completion = _completion_payload(
        config=config,
        evidence_fingerprint=expected_evidence_fingerprint,
        manifest_sha256=_sha256_file(root / M43_MANIFEST_FILE),
    )
    if completion != expected_completion:
        raise M43EvidenceError("audit completion marker is not content-bound")

    expected_files = {
        M43_CONFIG_FILE,
        M43_OWNER_FILE,
        M43_HUMAN_SUMMARY_FILE,
        M43_MANIFEST_FILE,
        M43_COMPLETION_FILE,
        *expected_scope_files,
    }
    actual_files = _relative_files(root)
    if actual_files != expected_files:
        raise M43EvidenceError("completed audit evidence has extra or missing artifacts")
    human_summary = (root / M43_HUMAN_SUMMARY_FILE).read_text(encoding="utf-8")
    expected_summary = render_human_audit_summary(config, artifacts)
    if human_summary != expected_summary:
        raise M43EvidenceError("human audit summary disagrees with machine-readable evidence")
    _validate_portable_value(manifest, "audit manifest")
    _validate_portable_value(completion, "audit completion marker")
    return CompletedM43AuditEvidence(
        root=root,
        config=config,
        scope_artifacts=cast(
            Mapping[M43AuditScope, M43AuditScopeArtifact],
            _FrozenObjectMapping(cast(Mapping[object, object], scope_artifacts)),
        ),
        manifest=_FrozenMapping(manifest),
        completion=_FrozenMapping(completion),
    )


def _validate_observation_set(
    scope: M43AuditScope, observations: Sequence[ObservationSourceIdentity]
) -> None:
    if not observations:
        raise M43EvidenceError("audit config requires ordered observation identities")
    if any(not isinstance(item, ObservationSourceIdentity) for item in observations):
        raise TypeError("ordered_observations must contain ObservationSourceIdentity values")
    ids = tuple(item.observation_id for item in observations)
    if len(set(ids)) != len(ids):
        raise M43EvidenceError("ordered audit observation IDs must be unique")
    validation = tuple(
        item for item in observations if item.source is M43ObservationSource.M3B_VALIDATION
    )
    development = tuple(
        item for item in observations if item.source is M43ObservationSource.M42_DEVELOPMENT
    )
    expected_counts = {
        M43AuditScope.M3B_VALIDATION: (
            M43_VALIDATION_OBSERVATION_COUNT,
            0,
        ),
        M43AuditScope.M42_DEVELOPMENT: (
            0,
            M43_DEVELOPMENT_OBSERVATION_COUNT,
        ),
        M43AuditScope.COMBINED: (
            M43_VALIDATION_OBSERVATION_COUNT,
            M43_DEVELOPMENT_OBSERVATION_COUNT,
        ),
    }[scope]
    if (len(validation), len(development)) != expected_counts:
        raise M43EvidenceError(
            "audit observation counts must cover all 36 validation and/or 72 development episodes"
        )
    if scope is M43AuditScope.COMBINED and tuple(observations) != (*validation, *development):
        raise M43EvidenceError(
            "combined observations must keep validation then development explicitly separated"
        )
    for source, selected in (
        (M43ObservationSource.M3B_VALIDATION, validation),
        (M43ObservationSource.M42_DEVELOPMENT, development),
    ):
        if selected:
            _validate_counterfactual_groups(source, selected)


def _validate_counterfactual_groups(
    source: M43ObservationSource, observations: Sequence[ObservationSourceIdentity]
) -> None:
    groups: dict[tuple[str, int], list[ObservationSourceIdentity]] = {}
    for item in observations:
        groups.setdefault((item.scene_id, item.scene_seed), []).append(item)
    expected_group_count = (
        M43_VALIDATION_OBSERVATION_COUNT // 6
        if source is M43ObservationSource.M3B_VALIDATION
        else M43_DEVELOPMENT_OBSERVATION_COUNT // 6
    )
    if len(groups) != expected_group_count:
        raise M43EvidenceError(f"{source.value} must contain complete six-task scene groups")
    for values in groups.values():
        if tuple(item.queried_task_id for item in values) != M43_CANONICAL_TASK_IDS:
            raise M43EvidenceError(
                f"each {source.value} scene group must query canonical six-task ordering"
            )
        held_fixed = {
            (
                item.source_task_id,
                item.source_episode_id,
                item.scene_group_id,
                item.frame_index,
                item.image_fingerprint,
                item.policy_state_fingerprint,
            )
            for item in values
        }
        if len(held_fixed) != 1:
            raise M43EvidenceError(
                f"each {source.value} scene group must hold source RGB and policy state fixed"
            )


def _validate_scope_artifact_set(
    config: M43AuditConfig,
    scope_artifacts: Mapping[M43AuditScope, M43AuditScopeArtifact],
) -> dict[M43AuditScope, M43AuditScopeArtifact]:
    normalized: dict[M43AuditScope, M43AuditScopeArtifact] = {}
    for key, value in scope_artifacts.items():
        scope = M43AuditScope(key)
        if scope in normalized:
            raise M43EvidenceError("scope artifact keys must be unique")
        if not isinstance(value, M43AuditScopeArtifact):
            raise TypeError("scope_artifacts values must be M43AuditScopeArtifact")
        if value.scope is not scope:
            raise M43EvidenceError("scope artifact key and payload disagree")
        normalized[scope] = value
    if tuple(normalized) != config.required_artifact_scopes:
        raise M43EvidenceError(
            "scope artifacts must keep validation, development, and combined results separate"
        )
    for scope in config.required_artifact_scopes:
        observations = config.observations_for_scope(scope)
        expected_ids = tuple(item.observation_id for item in observations)
        if normalized[scope].ordered_observation_ids != expected_ids:
            raise M43EvidenceError("scope artifact observation order differs from audit config")
        expected_tasks = tuple(item.queried_task_id for item in observations)
        for audit in normalized[scope].policy_audits.values():
            if audit.distance_config.to_dict() != config.distance_config.to_dict():
                raise M43EvidenceError("policy audit distance config differs from the audit config")
            if tuple(item.requested_task_id for item in audit.task_retrieval) != expected_tasks:
                raise M43EvidenceError(
                    "policy audit requested tasks differ from fixed observation queries"
                )
    return normalized


def _validate_audit_observations(
    audit: SemanticAlignmentAudit,
    ordered_observation_ids: Sequence[str],
) -> None:
    if audit.observation_count != len(ordered_observation_ids):
        raise M43EvidenceError("policy audit observation count differs from its scope")
    for name, values in (
        ("task retrieval", audit.task_retrieval),
        ("object retrieval", audit.object_retrieval),
        ("bin retrieval", audit.bin_retrieval),
    ):
        ids = tuple(item.observation_id for item in values)
        if ids != tuple(ordered_observation_ids):
            raise M43EvidenceError(f"{name} does not preserve ordered fixed observations")
    interaction_ids = tuple(item.observation_id for item in audit.first_interactions)
    if not set(interaction_ids).issubset(ordered_observation_ids):
        raise M43EvidenceError("first-interaction evidence escaped the observation set")
    if len(set(interaction_ids)) != len(interaction_ids):
        raise M43EvidenceError("first-interaction evidence must not duplicate observations")
    _validate_complete_distance_results(audit, ordered_observation_ids)


def _validate_complete_distance_results(
    audit: SemanticAlignmentAudit, ordered_observation_ids: Sequence[str]
) -> None:
    records = audit.complete_distance_results
    if set(records) != set(ordered_observation_ids):
        raise M43EvidenceError(
            "complete ActionChunkDistanceV0 records must cover every ordered observation"
        )
    task_by_observation = {value.observation_id: value for value in audit.task_retrieval}
    object_by_observation = {value.observation_id: value for value in audit.object_retrieval}
    bin_by_observation = {value.observation_id: value for value in audit.bin_retrieval}
    expected_families = {
        "task_references": M43_CANONICAL_TASK_IDS,
        "object_centroids": OBJECT_IDS,
        "bin_centroids": BIN_IDS,
        "conditional_bin_references": BIN_IDS,
    }
    for observation_id in ordered_observation_ids:
        record = _require_mapping(
            records[observation_id], f"complete distances for {observation_id}"
        )
        _require_exact_keys(record, set(expected_families), "complete distance record")
        parsed: dict[str, dict[str, ActionChunkDistanceResult]] = {}
        for family, expected_labels in expected_families.items():
            values = _require_mapping(record[family], f"distance family {family}")
            if set(values) != set(expected_labels):
                raise M43EvidenceError(f"{family} must contain every declared canonical reference")
            parsed[family] = {
                label: _parse_distance_result(_require_mapping(values[label], f"{family}[{label}]"))
                for label in expected_labels
            }
        task = task_by_observation[observation_id]
        object_result = object_by_observation[observation_id]
        bin_result = bin_by_observation[observation_id]
        primary_bindings = (
            (
                parsed["task_references"],
                task.primary_distances,
                "task reference",
            ),
            (
                parsed["object_centroids"],
                object_result.primary_distances,
                "object centroid",
            ),
            (
                parsed["bin_centroids"],
                bin_result.primary_distances,
                "bin centroid",
            ),
            (
                parsed["conditional_bin_references"],
                bin_result.conditional_primary_distances,
                "conditional bin",
            ),
        )
        for full_results, primary_results, label in primary_bindings:
            if any(
                not math.isclose(
                    result.primary_distance,
                    float(primary_results[reference]),
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                for reference, result in full_results.items()
            ):
                raise M43EvidenceError(
                    f"{label} complete distances disagree with primary retrieval"
                )


def _validate_policy_summary(
    audit: SemanticAlignmentAudit,
    summary: Mapping[str, object],
    *,
    external_failure_distribution: Mapping[str, object] | None,
) -> None:
    _require_exact_keys(summary, _POLICY_SUMMARY_KEYS, "policy semantic summary")
    magnitude = summary["task_sensitivity_magnitude"]
    if isinstance(magnitude, bool) or not isinstance(magnitude, int | float):
        raise M43EvidenceError("task_sensitivity_magnitude must be numeric")
    expected = build_policy_semantic_summary(
        audit,
        task_sensitivity_magnitude=float(magnitude),
        external_failure_distribution=external_failure_distribution,
    )
    if canonical_json(summary) != canonical_json(expected):
        raise M43EvidenceError("policy semantic summary disagrees with the complete audit")


def _validate_failure_distribution(value: Mapping[str, object]) -> dict[str, int]:
    expected = {failure.value for failure in SemanticFailureClass}
    if set(value) != expected:
        raise M43EvidenceError("rollout failure distribution must contain all twelve classes")
    result: dict[str, int] = {}
    for name in sorted(expected):
        count = value[name]
        _require_nonnegative_integer(cast(int, count), f"failure distribution {name}")
        result[name] = cast(int, count)
    return result


def _validate_rollout_evidence(
    scope: M43AuditScope, value: Mapping[str, object]
) -> Mapping[str, object]:
    payload = dict(value)
    _require_exact_keys(
        payload,
        {
            "available",
            "source_fingerprint",
            "per_policy_post_grasp_failure_distribution",
            "per_policy_metrics",
        },
        "rollout aggregate evidence",
    )
    available = payload["available"]
    if not isinstance(available, bool):
        raise M43EvidenceError("rollout evidence availability must be boolean")
    distributions = _require_mapping(
        payload["per_policy_post_grasp_failure_distribution"],
        "rollout failure distributions",
    )
    metrics = _require_mapping(payload["per_policy_metrics"], "rollout metrics")
    if not available:
        if payload["source_fingerprint"] is not None or distributions or metrics:
            raise M43EvidenceError("unavailable rollout evidence must not contain aggregate data")
    else:
        source_fingerprint = payload["source_fingerprint"]
        if not isinstance(source_fingerprint, str):
            raise M43EvidenceError("available rollout evidence requires a source fingerprint")
        _require_digest(source_fingerprint, "rollout source fingerprint")
        if (
            tuple(distributions) != M43_CANDIDATE_POLICY_LABELS
            or tuple(metrics) != M43_CANDIDATE_POLICY_LABELS
        ):
            raise M43EvidenceError("rollout aggregate evidence requires both candidate policies")
        payload["per_policy_post_grasp_failure_distribution"] = {
            label: _validate_failure_distribution(
                _require_mapping(distributions[label], f"rollout distribution {label}")
            )
            for label in M43_CANDIDATE_POLICY_LABELS
        }
        payload["per_policy_metrics"] = {
            label: dict(_require_mapping(metrics[label], f"rollout metrics {label}"))
            for label in M43_CANDIDATE_POLICY_LABELS
        }
    if scope is M43AuditScope.M3B_VALIDATION and available:
        raise M43EvidenceError("validation-only audit cannot claim rollout aggregate evidence")
    _validate_portable_value(payload, "rollout aggregate evidence")
    return _FrozenMapping(payload)


def _completion_payload(
    *, config: M43AuditConfig, evidence_fingerprint: str, manifest_sha256: str
) -> dict[str, object]:
    _require_digest(evidence_fingerprint, "evidence_fingerprint")
    _require_digest(manifest_sha256, "manifest_sha256")
    return {
        "schema_version": M43_AUDIT_COMPLETION_SCHEMA_VERSION,
        "audit_schema_version": M43_SCHEMA_VERSION,
        "config_fingerprint": config.fingerprint,
        "evidence_fingerprint": evidence_fingerprint,
        "manifest_sha256": manifest_sha256,
        "semantic_audit_completed": True,
        "task_token_m42_status": M43_TASK_TOKEN_M42_STATUS,
        "test_split_accessed": False,
        "fresh_seed_accessed": False,
        "final_schedule_accessed": False,
        "final_benchmark_authorized": False,
        "smolvla_go": False,
        "complete": True,
    }


def _parse_observation_identity(value: Mapping[str, object]) -> ObservationSourceIdentity:
    _require_exact_keys(
        value,
        {
            "source",
            "observation_id",
            "scene_id",
            "scene_group_id",
            "scene_seed",
            "source_task_id",
            "source_episode_id",
            "frame_index",
            "image_fingerprint",
            "policy_state_fingerprint",
            "queried_task_id",
            "policy_state_dimension",
        },
        "observation identity",
    )
    return ObservationSourceIdentity(
        source=M43ObservationSource(cast(str, value["source"])),
        observation_id=cast(str, value["observation_id"]),
        scene_id=cast(str, value["scene_id"]),
        scene_group_id=cast(str | None, value["scene_group_id"]),
        scene_seed=cast(int, value["scene_seed"]),
        source_task_id=cast(str, value["source_task_id"]),
        source_episode_id=cast(str | None, value["source_episode_id"]),
        frame_index=cast(int, value["frame_index"]),
        image_fingerprint=cast(str, value["image_fingerprint"]),
        policy_state_fingerprint=cast(str, value["policy_state_fingerprint"]),
        queried_task_id=cast(str, value["queried_task_id"]),
        policy_state_dimension=cast(int, value["policy_state_dimension"]),
    )


def _parse_distance_config(value: Mapping[str, object]) -> ActionChunkDistanceConfig:
    _require_exact_keys(
        value,
        {
            "action_lower_bounds",
            "action_upper_bounds",
            "train_action_std",
            "locked_execution_horizon",
            "action_dimension",
            "chunk_size",
            "arm_action_indices",
            "gripper_action_indices",
            "primary_metric",
            "version",
        },
        "action distance config",
    )
    return ActionChunkDistanceConfig(
        action_lower_bounds=tuple(cast(list[float], value["action_lower_bounds"])),
        action_upper_bounds=tuple(cast(list[float], value["action_upper_bounds"])),
        train_action_std=tuple(cast(list[float], value["train_action_std"])),
        locked_execution_horizon=cast(int, value["locked_execution_horizon"]),
        action_dimension=cast(int, value["action_dimension"]),
        chunk_size=cast(int, value["chunk_size"]),
        arm_action_indices=tuple(cast(list[int], value["arm_action_indices"])),
        gripper_action_indices=tuple(cast(list[int], value["gripper_action_indices"])),
        primary_metric=cast(str, value["primary_metric"]),
        version=cast(str, value["version"]),
    )


def _parse_distance_result(value: Mapping[str, object]) -> ActionChunkDistanceResult:
    mapping_names = (
        "raw_l2",
        "action_range_normalized_l2",
        "train_std_normalized_l2",
        "cosine_distance",
        "arm_only_raw_l2",
        "arm_only_action_range_normalized_l2",
        "arm_only_train_std_normalized_l2",
        "arm_only_cosine_distance",
        "gripper_only_raw_l2",
        "gripper_only_action_range_normalized_l2",
        "gripper_only_train_std_normalized_l2",
        "gripper_only_cosine_distance",
    )
    _require_exact_keys(
        value,
        {
            *mapping_names,
            "primary_metric",
            "primary_distance",
            "chunk_length",
            "action_dimension",
            "version",
        },
        "ActionChunkDistanceV0 result",
    )
    return ActionChunkDistanceResult(
        **{
            name: _require_mapping(value[name], f"distance result {name}") for name in mapping_names
        },
        primary_metric=cast(str, value["primary_metric"]),
        primary_distance=cast(float, value["primary_distance"]),
        chunk_length=cast(int, value["chunk_length"]),
        action_dimension=cast(int, value["action_dimension"]),
        version=cast(str, value["version"]),
    )


def _parse_confusion(value: Mapping[str, object]) -> TaskRetrievalConfusion:
    _require_exact_keys(
        value,
        {"requested_labels", "predicted_labels", "counts"},
        "retrieval confusion",
    )
    return TaskRetrievalConfusion(
        requested_labels=tuple(cast(list[str], value["requested_labels"])),
        predicted_labels=tuple(cast(list[str], value["predicted_labels"])),
        counts=tuple(tuple(row) for row in cast(list[list[int]], value["counts"])),
    )


def _parse_semantic_retrieval(value: Mapping[str, object]) -> SemanticRetrievalResult:
    _require_exact_keys(
        value,
        {
            "observation_id",
            "requested_task_id",
            "nearest_task_id",
            "ranked_task_ids",
            "primary_distances",
            "correct_rank",
            "top1_correct",
            "top2_correct",
            "reciprocal_rank",
            "correct_margin",
        },
        "semantic retrieval",
    )
    return SemanticRetrievalResult(
        observation_id=cast(str, value["observation_id"]),
        requested_task_id=cast(str, value["requested_task_id"]),
        nearest_task_id=cast(str, value["nearest_task_id"]),
        ranked_task_ids=tuple(cast(list[str], value["ranked_task_ids"])),
        primary_distances=_require_mapping(value["primary_distances"], "primary distances"),
        correct_rank=cast(int, value["correct_rank"]),
        top1_correct=cast(bool, value["top1_correct"]),
        top2_correct=cast(bool, value["top2_correct"]),
        reciprocal_rank=cast(float, value["reciprocal_rank"]),
        correct_margin=cast(float, value["correct_margin"]),
    )


def _parse_object_retrieval(value: Mapping[str, object]) -> ObjectRetrievalResult:
    _require_exact_keys(
        value,
        {
            "observation_id",
            "requested_object_id",
            "predicted_object_id",
            "ranked_object_ids",
            "primary_distances",
            "correct",
            "correct_margin",
        },
        "object retrieval",
    )
    return ObjectRetrievalResult(
        observation_id=cast(str, value["observation_id"]),
        requested_object_id=cast(str, value["requested_object_id"]),
        predicted_object_id=cast(str, value["predicted_object_id"]),
        ranked_object_ids=tuple(cast(list[str], value["ranked_object_ids"])),
        primary_distances=_require_mapping(value["primary_distances"], "object distances"),
        correct=cast(bool, value["correct"]),
        correct_margin=cast(float, value["correct_margin"]),
    )


def _parse_bin_retrieval(value: Mapping[str, object]) -> BinRetrievalResult:
    _require_exact_keys(
        value,
        {
            "observation_id",
            "requested_bin_id",
            "predicted_bin_id",
            "ranked_bin_ids",
            "primary_distances",
            "correct",
            "correct_margin",
            "conditional_predicted_bin_id",
            "conditional_primary_distances",
            "conditional_correct",
            "conditional_correct_margin",
        },
        "bin retrieval",
    )
    return BinRetrievalResult(
        observation_id=cast(str, value["observation_id"]),
        requested_bin_id=cast(str, value["requested_bin_id"]),
        predicted_bin_id=cast(str, value["predicted_bin_id"]),
        ranked_bin_ids=tuple(cast(list[str], value["ranked_bin_ids"])),
        primary_distances=_require_mapping(value["primary_distances"], "bin distances"),
        correct=cast(bool, value["correct"]),
        correct_margin=cast(float, value["correct_margin"]),
        conditional_predicted_bin_id=cast(str, value["conditional_predicted_bin_id"]),
        conditional_primary_distances=_require_mapping(
            value["conditional_primary_distances"], "conditional bin distances"
        ),
        conditional_correct=cast(bool, value["conditional_correct"]),
        conditional_correct_margin=cast(float, value["conditional_correct_margin"]),
    )


def _parse_first_interaction(value: Mapping[str, object]) -> FirstInteractionRecord:
    _require_exact_keys(
        value,
        {
            "observation_id",
            "requested_task_id",
            "requested_target_object_id",
            "first_object_grasped",
            "first_object_displaced",
            "displacement_threshold_m",
            "requested_destination_bin_id",
            "first_bin_approached",
            "first_bin_entered_by_any_object",
            "final_object_bin_relationship",
            "nearest_per_task_task_id",
            "first_grasp_matches_nearest_per_task",
            "initial_semantic_error_visible",
            "failure_class",
        },
        "first interaction",
    )
    return FirstInteractionRecord(
        observation_id=cast(str, value["observation_id"]),
        requested_task_id=cast(str, value["requested_task_id"]),
        requested_target_object_id=cast(str, value["requested_target_object_id"]),
        first_object_grasped=cast(str | None, value["first_object_grasped"]),
        first_object_displaced=cast(str | None, value["first_object_displaced"]),
        displacement_threshold_m=cast(float, value["displacement_threshold_m"]),
        requested_destination_bin_id=cast(str, value["requested_destination_bin_id"]),
        first_bin_approached=cast(str | None, value["first_bin_approached"]),
        first_bin_entered_by_any_object=cast(str | None, value["first_bin_entered_by_any_object"]),
        final_object_bin_relationship=_require_mapping(
            value["final_object_bin_relationship"], "final object/bin relationship"
        ),
        nearest_per_task_task_id=cast(str, value["nearest_per_task_task_id"]),
        first_grasp_matches_nearest_per_task=cast(
            bool | None, value["first_grasp_matches_nearest_per_task"]
        ),
        initial_semantic_error_visible=cast(bool, value["initial_semantic_error_visible"]),
        failure_class=SemanticFailureClass(cast(str, value["failure_class"])),
    )


def _parse_semantic_alignment_audit(value: Mapping[str, object]) -> SemanticAlignmentAudit:
    _require_exact_keys(
        value,
        {
            "policy_label",
            "observation_count",
            "distance_config",
            "task_retrieval",
            "object_retrieval",
            "bin_retrieval",
            "task_confusion",
            "object_confusion",
            "bin_confusion",
            "first_interactions",
            "first_interaction_confusions",
            "failure_distribution",
            "conclusions",
            "complete_distance_results",
            "raw_policy_action_metrics",
            "runtime_action_metrics",
            "deterministic_reload_validated",
            "deterministic_reset_validated",
            "schema_version",
        },
        "semantic alignment audit",
    )
    return SemanticAlignmentAudit(
        policy_label=cast(str, value["policy_label"]),
        observation_count=cast(int, value["observation_count"]),
        distance_config=_parse_distance_config(
            _require_mapping(value["distance_config"], "audit distance config")
        ),
        task_retrieval=tuple(
            _parse_semantic_retrieval(_require_mapping(item, "task retrieval item"))
            for item in _require_list(value["task_retrieval"], "task retrieval")
        ),
        object_retrieval=tuple(
            _parse_object_retrieval(_require_mapping(item, "object retrieval item"))
            for item in _require_list(value["object_retrieval"], "object retrieval")
        ),
        bin_retrieval=tuple(
            _parse_bin_retrieval(_require_mapping(item, "bin retrieval item"))
            for item in _require_list(value["bin_retrieval"], "bin retrieval")
        ),
        task_confusion=_parse_confusion(
            _require_mapping(value["task_confusion"], "task confusion")
        ),
        object_confusion=_parse_confusion(
            _require_mapping(value["object_confusion"], "object confusion")
        ),
        bin_confusion=_parse_confusion(_require_mapping(value["bin_confusion"], "bin confusion")),
        first_interactions=tuple(
            _parse_first_interaction(_require_mapping(item, "first interaction item"))
            for item in _require_list(value["first_interactions"], "first interactions")
        ),
        first_interaction_confusions=_require_mapping(
            value["first_interaction_confusions"], "first interaction confusions"
        ),
        failure_distribution=_require_mapping(
            value["failure_distribution"], "failure distribution"
        ),
        conclusions=tuple(
            SemanticAuditConclusion(item)
            for item in cast(list[str], _require_list(value["conclusions"], "conclusions"))
        ),
        complete_distance_results=_require_mapping(
            value["complete_distance_results"], "complete distance results"
        ),
        raw_policy_action_metrics=_require_mapping(
            value["raw_policy_action_metrics"], "raw policy action metrics"
        ),
        runtime_action_metrics=_require_mapping(
            value["runtime_action_metrics"], "runtime action metrics"
        ),
        deterministic_reload_validated=cast(bool, value["deterministic_reload_validated"]),
        deterministic_reset_validated=cast(bool, value["deterministic_reset_validated"]),
        schema_version=cast(str, value["schema_version"]),
    )


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise M43EvidenceError(f"{name} must be a JSON object")
    return cast(Mapping[str, object], value)


def _require_list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise M43EvidenceError(f"{name} must be a JSON array")
    return value


def _require_exact_keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise M43EvidenceError(f"{name} fields differ from the declared schema")


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _resolved_unlinked(path: Path, *, label: str) -> Path:
    lexical = _lexical_absolute(path)
    for component in (lexical, *lexical.parents):
        if component.is_symlink() or component.is_junction():
            raise M43EvidenceError(f"{label} traverses a symlink or junction: {component}")
    return lexical.resolve(strict=False)


def _safe_evidence_root(output_root: Path) -> Path:
    root = _resolved_unlinked(output_root, label="M4.3 audit output root")
    if root.exists() and not root.is_dir():
        raise M43EvidenceError("M4.3 audit output root must be a real directory")
    root.mkdir(parents=True, exist_ok=True)
    evidence = _safe_child(root, M43_EVIDENCE_DIRECTORY, "M4.3 audit evidence root")
    if evidence.exists() and not evidence.is_dir():
        raise M43EvidenceError("M4.3 audit evidence root must be a real directory")
    evidence.mkdir(exist_ok=True)
    return evidence


def _safe_child(parent: Path, name: str, label: str) -> Path:
    candidate = _resolved_unlinked(parent / name, label=label)
    if candidate.parent != parent.resolve():
        raise M43EvidenceError(f"{label} escaped its owned root")
    return candidate


def _require_real_directory(path: Path, label: str) -> None:
    if path.is_symlink() or path.is_junction() or not path.is_dir():
        raise M43EvidenceError(f"{label} must be one real directory: {path}")


def _write_new_json(path: Path, payload: object) -> None:
    _validate_portable_value(payload, f"artifact {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
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
        raise M43EvidenceError(f"immutable audit artifact already exists: {path}") from error


def _write_new_text(path: Path, value: str) -> None:
    _validate_portable_value(value, f"artifact {path.name}")
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise M43EvidenceError(f"immutable audit artifact already exists: {path}") from error


def _read_object(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or path.is_junction() or not path.is_file():
        raise M43EvidenceError(f"{label} is missing or unsafe: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: _raise_invalid_json_constant(value),
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise M43EvidenceError(f"{label} is not valid UTF-8 JSON: {error}") from error
    if not isinstance(payload, dict):
        raise M43EvidenceError(f"{label} must be a JSON object")
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
            raise M43EvidenceError(f"audit evidence contains a linked entry: {candidate}")
        if candidate.is_file():
            files.add(PurePosixPath(candidate.relative_to(root)).as_posix())
        elif not candidate.is_dir():
            raise M43EvidenceError(f"audit evidence contains an unsafe entry: {candidate}")
    return files


def _artifact_records(root: Path) -> list[dict[str, object]]:
    excluded = {M43_MANIFEST_FILE, M43_COMPLETION_FILE}
    records = []
    for relative in sorted(_relative_files(root) - excluded):
        path = root / PurePosixPath(relative)
        records.append(
            {
                "path": relative,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def _validate_artifact_records(root: Path, manifest: Mapping[str, object]) -> None:
    records = manifest.get("artifacts")
    if not isinstance(records, list):
        raise M43EvidenceError("audit manifest artifacts must be a JSON array")
    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, raw in enumerate(records):
        record = _require_mapping(raw, f"artifact record {index}")
        _require_exact_keys(record, {"path", "sha256", "size_bytes"}, "artifact record")
        relative = record["path"]
        digest = record["sha256"]
        size = record["size_bytes"]
        if not isinstance(relative, str) or not relative or relative in seen:
            raise M43EvidenceError("artifact paths must be unique non-empty strings")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
            raise M43EvidenceError("artifact manifest contains an unsafe relative path")
        if relative in {M43_MANIFEST_FILE, M43_COMPLETION_FILE}:
            raise M43EvidenceError("manifest and completion cannot checksum themselves")
        if not isinstance(digest, str):
            raise M43EvidenceError("artifact checksum must be a string")
        _require_digest(digest, "artifact checksum")
        _require_nonnegative_integer(cast(int, size), "artifact size")
        path = root.joinpath(*pure.parts)
        if path.is_symlink() or path.is_junction() or not path.is_file():
            raise M43EvidenceError(f"checksummed artifact is missing or unsafe: {relative}")
        if path.stat().st_size != size or _sha256_file(path) != digest:
            raise M43EvidenceError(f"artifact checksum or size mismatch: {relative}")
        seen.add(relative)
        normalized.append({"path": relative, "sha256": digest, "size_bytes": size})
    if normalized != sorted(normalized, key=lambda item: cast(str, item["path"])):
        raise M43EvidenceError("artifact records must use deterministic path order")
    actual = _relative_files(root) - {M43_MANIFEST_FILE, M43_COMPLETION_FILE}
    if actual != seen:
        raise M43EvidenceError("manifest artifact records are incomplete or contain extras")


def _fsync_tree(root: Path) -> None:
    # Every file write above is already flushed and fsynced.  Re-opening files
    # read-only for a second fsync is not supported by the Windows CRT.
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
    "M43_AUDIT_COMPLETION_SCHEMA_VERSION",
    "M43_AUDIT_CONFIG_SCHEMA_VERSION",
    "M43_AUDIT_MANIFEST_SCHEMA_VERSION",
    "M43_AUDIT_SCOPE_SCHEMA_VERSION",
    "M43_CANDIDATE_POLICY_LABELS",
    "M43_DEVELOPMENT_OBSERVATION_COUNT",
    "M43_TASK_TOKEN_M42_STATUS",
    "M43_VALIDATION_OBSERVATION_COUNT",
    "CompletedM43AuditEvidence",
    "M43AuditConfig",
    "M43AuditScope",
    "M43AuditScopeArtifact",
    "M43EvidenceError",
    "M43ObservationSource",
    "ObservationSourceIdentity",
    "build_policy_semantic_summary",
    "build_rollout_aggregate_evidence",
    "render_human_audit_summary",
    "stage_and_promote_audit_evidence",
    "unavailable_rollout_evidence",
    "validate_completed_audit_evidence",
]
