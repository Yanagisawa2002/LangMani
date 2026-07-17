"""Immutable contracts for M5A language routing and corpus metadata."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, cast

from langmani.datasets.identity import canonical_json, sha256_hex
from langmani.environments.specs import (
    BIN_IDS,
    OBJECT_IDS,
    BinId,
    ObjectId,
    TaskSpec,
    stable_task_id,
)

M5A_ROUTER_SCHEMA_VERSION = "langmani-m5a-router-v1"
M5A_CORPUS_SCHEMA_VERSION = "langmani-m5a-language-corpus-v1"
M5A_SPLIT_SCHEMA_VERSION = "langmani-m5a-language-splits-v1"
M5A_NORMALIZATION_VERSION = "unicode-nfkc-lower-whitespace-v1"


class RouterContractError(ValueError):
    """Raised when immutable M5A routing metadata violates its schema."""


class RouterStatus(StrEnum):
    """The complete set of executable and non-executable router outcomes."""

    ROUTE = "route"
    REJECT_AMBIGUOUS = "reject_ambiguous"
    REJECT_UNSUPPORTED = "reject_unsupported"
    REJECT_MALFORMED = "reject_malformed"


class RouterRejectionReason(StrEnum):
    """Machine-readable reasons for declining to dispatch a controller."""

    MISSING_OBJECT = "missing_object"
    MISSING_DESTINATION = "missing_destination"
    CONFLICTING_OBJECTS = "conflicting_objects"
    CONFLICTING_BINS = "conflicting_bins"
    UNSUPPORTED_OBJECT = "unsupported_object"
    UNSUPPORTED_DESTINATION = "unsupported_destination"
    MULTIPLE_TASKS = "multiple_sequential_tasks"
    UNRESOLVED_CORRECTION = "unresolved_correction"
    CONTRADICTORY_NEGATION = "contradictory_negation"
    MEANINGLESS_TEXT = "meaningless_or_noise_text"
    EMPTY_TEXT = "empty_text"
    MALFORMED_CONTROL_CHARACTERS = "malformed_control_characters"
    UNSUPPORTED_ACTION = "unsupported_action"
    UNSUPPORTED_SPATIAL_REFERENCE = "unsupported_spatial_reference"
    LOW_CONFIDENCE = "low_confidence"
    CLASSIFIER_AMBIGUOUS = "classifier_predicted_ambiguous"
    CLASSIFIER_UNSUPPORTED = "classifier_predicted_unsupported"
    CLASSIFIER_MALFORMED = "classifier_predicted_malformed"
    STRUCTURED_OUTPUT_INVALID = "structured_output_invalid"
    FORMAT_REPAIR_EXHAUSTED = "format_repair_exhausted"


class LanguageSplit(StrEnum):
    """Project-owned language splits, separate from the M3B dataset split enum."""

    TRAIN = "train"
    VALIDATION = "validation"
    DEVELOPMENT = "development"
    FINAL = "final"


def _require_non_empty(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RouterContractError(f"{name} must be a non-empty string")
    return value


def _freeze_json(value: object) -> object:
    """Validate and recursively freeze a JSON-compatible value."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RouterContractError("JSON metadata floats must be finite")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RouterContractError("JSON metadata keys must be strings")
            frozen[key] = _freeze_json(item)
        return MappingProxyType(dict(sorted(frozen.items())))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item) for item in value)
    raise RouterContractError(f"unsupported JSON metadata value: {type(value).__name__}")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _fingerprint(kind: str, payload: object) -> str:
    return f"langmani-m5a-{kind}-{sha256_hex({'kind': kind, 'payload': payload})}"


@dataclass(frozen=True, slots=True)
class RouterConfidence:
    """A confidence score only when the router has a defensible definition."""

    available: bool
    score: float | None
    definition: str | None
    calibrated: bool

    def __post_init__(self) -> None:
        if not isinstance(self.available, bool) or not isinstance(self.calibrated, bool):
            raise RouterContractError("confidence availability flags must be booleans")
        if not self.available:
            if self.score is not None or self.calibrated:
                raise RouterContractError(
                    "unavailable confidence must have score=None and calibrated=False"
                )
            if self.definition is not None:
                _require_non_empty(self.definition, name="confidence definition")
            return
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise RouterContractError("available confidence requires a numeric score")
        score = float(self.score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise RouterContractError("confidence score must be finite and in [0, 1]")
        _require_non_empty(self.definition, name="confidence definition")
        object.__setattr__(self, "score", score)

    @classmethod
    def unavailable(cls, *, definition: str | None = None) -> RouterConfidence:
        """Build an explicit no-confidence record without inventing a number."""
        return cls(available=False, score=None, definition=definition, calibrated=False)

    @classmethod
    def probability(
        cls,
        score: float,
        *,
        definition: str,
        calibrated: bool,
    ) -> RouterConfidence:
        """Build a probability-like score whose meaning is recorded alongside it."""
        return cls(
            available=True,
            score=score,
            definition=definition,
            calibrated=calibrated,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "score": self.score,
            "definition": self.definition,
            "calibrated": self.calibrated,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> RouterConfidence:
        expected = {"available", "score", "definition", "calibrated"}
        if set(value) != expected:
            raise RouterContractError("confidence record has an invalid field set")
        return cls(
            available=cast(bool, value["available"]),
            score=cast(float | None, value["score"]),
            definition=cast(str | None, value["definition"]),
            calibrated=cast(bool, value["calibrated"]),
        )


@dataclass(frozen=True, slots=True)
class RouterDecision:
    """Validated router output; rejection records are intentionally non-executable."""

    status: RouterStatus
    target_object_id: ObjectId | None
    target_bin_id: BinId | None
    task_spec: TaskSpec | None
    task_id: str | None
    confidence: RouterConfidence
    router_name: str
    router_version: str
    evidence: Mapping[str, object] = field(default_factory=dict)
    rejection_reason: RouterRejectionReason | None = None
    decision_fingerprint: str = ""
    schema_version: str = M5A_ROUTER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.status, RouterStatus):
            raise RouterContractError("status must be a RouterStatus")
        if not isinstance(self.confidence, RouterConfidence):
            raise RouterContractError("confidence must be a RouterConfidence")
        _require_non_empty(self.router_name, name="router_name")
        _require_non_empty(self.router_version, name="router_version")
        if self.schema_version != M5A_ROUTER_SCHEMA_VERSION:
            raise RouterContractError("unknown router schema version")
        if not isinstance(self.evidence, Mapping):
            raise RouterContractError("evidence must be a mapping")
        frozen_evidence = cast(Mapping[str, object], _freeze_json(self.evidence))
        object.__setattr__(self, "evidence", frozen_evidence)

        if self.status is RouterStatus.ROUTE:
            if self.rejection_reason is not None:
                raise RouterContractError("route decisions cannot contain a rejection reason")
            if not isinstance(self.task_spec, TaskSpec):
                raise RouterContractError("route decisions require a TaskSpec")
            if self.task_spec.instruction_template_id != "canonical_v0":
                raise RouterContractError(
                    "M5A routes must use instruction_template_id canonical_v0"
                )
            if self.target_object_id != self.task_spec.target_object_id:
                raise RouterContractError("target_object_id is inconsistent with task_spec")
            if self.target_bin_id != self.task_spec.target_bin_id:
                raise RouterContractError("target_bin_id is inconsistent with task_spec")
            if self.task_id != stable_task_id(self.task_spec):
                raise RouterContractError("task_id is inconsistent with task_spec")
        else:
            if not isinstance(self.rejection_reason, RouterRejectionReason):
                raise RouterContractError("rejected decisions require a rejection reason")
            if any(
                value is not None
                for value in (
                    self.target_object_id,
                    self.target_bin_id,
                    self.task_spec,
                    self.task_id,
                )
            ):
                raise RouterContractError(
                    "rejected decisions cannot contain executable TaskSpec fields"
                )

        expected_fingerprint = _fingerprint("router-decision", self.identity_dict())
        if self.decision_fingerprint and self.decision_fingerprint != expected_fingerprint:
            raise RouterContractError("decision_fingerprint does not match decision contents")
        object.__setattr__(self, "decision_fingerprint", expected_fingerprint)

    @classmethod
    def route(
        cls,
        *,
        task_spec: TaskSpec,
        confidence: RouterConfidence,
        router_name: str,
        router_version: str,
        evidence: Mapping[str, object] | None = None,
    ) -> RouterDecision:
        return cls(
            status=RouterStatus.ROUTE,
            target_object_id=task_spec.target_object_id,
            target_bin_id=task_spec.target_bin_id,
            task_spec=task_spec,
            task_id=stable_task_id(task_spec),
            confidence=confidence,
            router_name=router_name,
            router_version=router_version,
            evidence={} if evidence is None else evidence,
        )

    @classmethod
    def reject(
        cls,
        *,
        status: RouterStatus,
        rejection_reason: RouterRejectionReason,
        confidence: RouterConfidence,
        router_name: str,
        router_version: str,
        evidence: Mapping[str, object] | None = None,
    ) -> RouterDecision:
        if status is RouterStatus.ROUTE:
            raise RouterContractError("RouterDecision.reject requires a rejection status")
        return cls(
            status=status,
            target_object_id=None,
            target_bin_id=None,
            task_spec=None,
            task_id=None,
            confidence=confidence,
            router_name=router_name,
            router_version=router_version,
            evidence={} if evidence is None else evidence,
            rejection_reason=rejection_reason,
        )

    def identity_dict(self) -> dict[str, object]:
        """Return the canonical decision payload, excluding its derived fingerprint."""
        return {
            "status": self.status.value,
            "target_object_id": self.target_object_id,
            "target_bin_id": self.target_bin_id,
            "task_spec": None if self.task_spec is None else self.task_spec.to_dict(),
            "task_id": self.task_id,
            "confidence": self.confidence.to_dict(),
            "router_name": self.router_name,
            "router_version": self.router_version,
            "evidence": _thaw_json(self.evidence),
            "rejection_reason": (
                None if self.rejection_reason is None else self.rejection_reason.value
            ),
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, object]:
        result = self.identity_dict()
        result["decision_fingerprint"] = self.decision_fingerprint
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> RouterDecision:
        expected = {
            "status",
            "target_object_id",
            "target_bin_id",
            "task_spec",
            "task_id",
            "confidence",
            "router_name",
            "router_version",
            "evidence",
            "rejection_reason",
            "decision_fingerprint",
            "schema_version",
        }
        if set(value) != expected:
            raise RouterContractError("router decision has an invalid field set")
        task_payload = value["task_spec"]
        confidence_payload = value["confidence"]
        evidence_payload = value["evidence"]
        if task_payload is not None and not isinstance(task_payload, Mapping):
            raise RouterContractError("task_spec must be a mapping or null")
        if not isinstance(confidence_payload, Mapping):
            raise RouterContractError("confidence must be a mapping")
        if not isinstance(evidence_payload, Mapping):
            raise RouterContractError("evidence must be a mapping")
        rejection_value = value["rejection_reason"]
        return cls(
            status=RouterStatus(cast(str, value["status"])),
            target_object_id=cast(ObjectId | None, value["target_object_id"]),
            target_bin_id=cast(BinId | None, value["target_bin_id"]),
            task_spec=(
                None
                if task_payload is None
                else TaskSpec.from_mapping(cast(Mapping[str, object], task_payload))
            ),
            task_id=cast(str | None, value["task_id"]),
            confidence=RouterConfidence.from_dict(cast(Mapping[str, object], confidence_payload)),
            router_name=cast(str, value["router_name"]),
            router_version=cast(str, value["router_version"]),
            evidence=cast(Mapping[str, object], evidence_payload),
            rejection_reason=(
                None
                if rejection_value is None
                else RouterRejectionReason(cast(str, rejection_value))
            ),
            decision_fingerprint=cast(str, value["decision_fingerprint"]),
            schema_version=cast(str, value["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class LanguageTemplateFamily:
    """One structural language family that belongs to exactly one split."""

    family_id: str
    split: LanguageSplit
    structural_signature: str
    near_duplicate_group_id: str
    category: str
    generation_provenance: Mapping[str, object]
    schema_version: str = M5A_CORPUS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "family_id",
            "structural_signature",
            "near_duplicate_group_id",
            "category",
        ):
            _require_non_empty(getattr(self, name), name=name)
        if not isinstance(self.split, LanguageSplit):
            raise RouterContractError("family split must be a LanguageSplit")
        if self.schema_version != M5A_CORPUS_SCHEMA_VERSION:
            raise RouterContractError("unknown language family schema version")
        if not isinstance(self.generation_provenance, Mapping):
            raise RouterContractError("generation_provenance must be a mapping")
        object.__setattr__(
            self,
            "generation_provenance",
            cast(Mapping[str, object], _freeze_json(self.generation_provenance)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "family_id": self.family_id,
            "split": self.split.value,
            "structural_signature": self.structural_signature,
            "near_duplicate_group_id": self.near_duplicate_group_id,
            "category": self.category,
            "generation_provenance": _thaw_json(self.generation_provenance),
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class LanguageExample:
    """Atomic corpus record with an independently verifiable persistent ID."""

    example_id: str
    raw_text: str
    normalized_text: str
    template_family_id: str
    lexical_variant_ids: tuple[str, ...]
    expected_status: RouterStatus
    expected_task_spec: TaskSpec | None
    expected_rejection_reason: RouterRejectionReason | None
    generation_provenance: Mapping[str, object]
    split: LanguageSplit
    schema_version: str = M5A_CORPUS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_non_empty(self.example_id, name="example_id")
        if not isinstance(self.raw_text, str) or not isinstance(self.normalized_text, str):
            raise RouterContractError("language example text fields must be strings")
        _require_non_empty(self.template_family_id, name="template_family_id")
        if not isinstance(self.lexical_variant_ids, tuple) or not all(
            isinstance(value, str) and value for value in self.lexical_variant_ids
        ):
            raise RouterContractError("lexical_variant_ids must be a tuple of non-empty strings")
        if len(set(self.lexical_variant_ids)) != len(self.lexical_variant_ids):
            raise RouterContractError("lexical_variant_ids must be unique")
        if not isinstance(self.expected_status, RouterStatus):
            raise RouterContractError("expected_status must be a RouterStatus")
        if not isinstance(self.split, LanguageSplit):
            raise RouterContractError("example split must be a LanguageSplit")
        if self.schema_version != M5A_CORPUS_SCHEMA_VERSION:
            raise RouterContractError("unknown language example schema version")
        if not isinstance(self.generation_provenance, Mapping):
            raise RouterContractError("generation_provenance must be a mapping")
        object.__setattr__(
            self,
            "generation_provenance",
            cast(Mapping[str, object], _freeze_json(self.generation_provenance)),
        )
        if self.expected_status is RouterStatus.ROUTE:
            if not isinstance(self.expected_task_spec, TaskSpec):
                raise RouterContractError("routeable examples require expected_task_spec")
            if self.expected_task_spec.instruction_template_id != "canonical_v0":
                raise RouterContractError("routeable examples must target canonical_v0 TaskSpecs")
            if self.expected_rejection_reason is not None:
                raise RouterContractError("routeable examples cannot have a rejection reason")
        elif self.expected_task_spec is not None or not isinstance(
            self.expected_rejection_reason, RouterRejectionReason
        ):
            raise RouterContractError(
                "rejected examples require a reason and cannot contain expected_task_spec"
            )
        expected_id = stable_language_example_id(
            raw_text=self.raw_text,
            normalized_text=self.normalized_text,
            template_family_id=self.template_family_id,
            lexical_variant_ids=self.lexical_variant_ids,
            expected_status=self.expected_status,
            expected_task_spec=self.expected_task_spec,
            expected_rejection_reason=self.expected_rejection_reason,
            generation_provenance=self.generation_provenance,
            split=self.split,
            schema_version=self.schema_version,
        )
        if self.example_id != expected_id:
            raise RouterContractError("example_id does not match language example contents")

    @property
    def task_id(self) -> str | None:
        return None if self.expected_task_spec is None else stable_task_id(self.expected_task_spec)

    def to_dict(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "template_family_id": self.template_family_id,
            "lexical_variant_ids": list(self.lexical_variant_ids),
            "expected_status": self.expected_status.value,
            "expected_task_spec": (
                None if self.expected_task_spec is None else self.expected_task_spec.to_dict()
            ),
            "expected_rejection_reason": (
                None
                if self.expected_rejection_reason is None
                else self.expected_rejection_reason.value
            ),
            "generation_provenance": _thaw_json(self.generation_provenance),
            "split": self.split.value,
            "schema_version": self.schema_version,
        }


def stable_language_example_id(
    *,
    raw_text: str,
    normalized_text: str,
    template_family_id: str,
    lexical_variant_ids: Sequence[str],
    expected_status: RouterStatus,
    expected_task_spec: TaskSpec | None,
    expected_rejection_reason: RouterRejectionReason | None,
    generation_provenance: Mapping[str, object],
    split: LanguageSplit,
    schema_version: str = M5A_CORPUS_SCHEMA_VERSION,
) -> str:
    """Create an example ID from the complete semantic and provenance payload."""
    payload = {
        "raw_text": raw_text,
        "normalized_text": normalized_text,
        "template_family_id": template_family_id,
        "lexical_variant_ids": list(lexical_variant_ids),
        "expected_status": expected_status.value,
        "expected_task_spec": (
            None if expected_task_spec is None else expected_task_spec.to_dict()
        ),
        "expected_rejection_reason": (
            None if expected_rejection_reason is None else expected_rejection_reason.value
        ),
        "generation_provenance": dict(generation_provenance),
        "split": split.value,
        "schema_version": schema_version,
    }
    return _fingerprint("language-example", payload)


@dataclass(frozen=True, slots=True)
class LanguageCorpusConfig:
    """Predeclared exact corpus sizes and generator identity."""

    routeable_per_task: Mapping[LanguageSplit, int]
    rejected_total: Mapping[LanguageSplit, int]
    generator_version: str = "deterministic-language-generator-v1"
    normalization_version: str = M5A_NORMALIZATION_VERSION
    schema_version: str = M5A_CORPUS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        expected_splits = set(LanguageSplit)
        for name, counts in (
            ("routeable_per_task", self.routeable_per_task),
            ("rejected_total", self.rejected_total),
        ):
            if not isinstance(counts, Mapping) or set(counts) != expected_splits:
                raise RouterContractError(f"{name} must define every language split exactly once")
            normalized: dict[LanguageSplit, int] = {}
            for split, count in counts.items():
                if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                    raise RouterContractError(f"{name}[{split.value}] must be a positive integer")
                normalized[split] = count
            object.__setattr__(self, name, MappingProxyType(normalized))
        _require_non_empty(self.generator_version, name="generator_version")
        if self.normalization_version != M5A_NORMALIZATION_VERSION:
            raise RouterContractError("unknown normalization version")
        if self.schema_version != M5A_CORPUS_SCHEMA_VERSION:
            raise RouterContractError("unknown corpus schema version")

    def to_dict(self) -> dict[str, object]:
        return {
            "routeable_per_task": {
                split.value: self.routeable_per_task[split] for split in LanguageSplit
            },
            "rejected_total": {split.value: self.rejected_total[split] for split in LanguageSplit},
            "generator_version": self.generator_version,
            "normalization_version": self.normalization_version,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class LanguageSplitManifest:
    """Immutable aggregate metadata for one family-isolated language split."""

    split: LanguageSplit
    example_count: int
    routeable_count_by_task_id: Mapping[str, int]
    rejection_count_by_reason: Mapping[str, int]
    family_ids: tuple[str, ...]
    example_ids: tuple[str, ...]
    content_fingerprint: str
    texts_sealed: bool
    schema_version: str = M5A_SPLIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.split, LanguageSplit):
            raise RouterContractError("manifest split must be a LanguageSplit")
        if isinstance(self.example_count, bool) or not isinstance(self.example_count, int):
            raise RouterContractError("example_count must be an integer")
        if self.example_count < 0 or self.example_count != len(self.example_ids):
            raise RouterContractError("example_count must equal len(example_ids)")
        if len(set(self.example_ids)) != len(self.example_ids):
            raise RouterContractError("example_ids must be unique")
        if len(set(self.family_ids)) != len(self.family_ids):
            raise RouterContractError("family_ids must be unique")
        if not isinstance(self.texts_sealed, bool):
            raise RouterContractError("texts_sealed must be a boolean")
        if self.texts_sealed != (self.split is LanguageSplit.FINAL):
            raise RouterContractError("only the final language split may be sealed")
        for name, counts in (
            ("routeable_count_by_task_id", self.routeable_count_by_task_id),
            ("rejection_count_by_reason", self.rejection_count_by_reason),
        ):
            if not isinstance(counts, Mapping):
                raise RouterContractError(f"{name} must be a mapping")
            normalized: dict[str, int] = {}
            for key, count in counts.items():
                if not isinstance(key, str) or not key:
                    raise RouterContractError(f"{name} keys must be non-empty strings")
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise RouterContractError(f"{name} values must be non-negative integers")
                normalized[key] = count
            object.__setattr__(self, name, MappingProxyType(dict(sorted(normalized.items()))))
        _require_non_empty(self.content_fingerprint, name="content_fingerprint")
        if self.schema_version != M5A_SPLIT_SCHEMA_VERSION:
            raise RouterContractError("unknown split manifest schema version")
        if (
            sum(self.routeable_count_by_task_id.values())
            + sum(self.rejection_count_by_reason.values())
            != self.example_count
        ):
            raise RouterContractError("split manifest class counts do not equal example_count")

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split.value,
            "example_count": self.example_count,
            "routeable_count_by_task_id": dict(self.routeable_count_by_task_id),
            "rejection_count_by_reason": dict(self.rejection_count_by_reason),
            "family_ids": list(self.family_ids),
            "example_ids": list(self.example_ids),
            "content_fingerprint": self.content_fingerprint,
            "texts_sealed": self.texts_sealed,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class LanguageCorpusManifest:
    """Content-bound identity of the complete four-way corpus."""

    corpus_fingerprint: str
    config_fingerprint: str
    split_manifests: Mapping[LanguageSplit, LanguageSplitManifest]
    family_isolation_validated: bool
    final_locked: bool
    schema_version: str = M5A_CORPUS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_non_empty(self.corpus_fingerprint, name="corpus_fingerprint")
        _require_non_empty(self.config_fingerprint, name="config_fingerprint")
        if not isinstance(self.split_manifests, Mapping) or set(self.split_manifests) != set(
            LanguageSplit
        ):
            raise RouterContractError("split_manifests must contain every language split")
        normalized: dict[LanguageSplit, LanguageSplitManifest] = {}
        for split, manifest in self.split_manifests.items():
            if not isinstance(manifest, LanguageSplitManifest) or manifest.split is not split:
                raise RouterContractError("split manifest key is inconsistent with its contents")
            normalized[split] = manifest
        object.__setattr__(self, "split_manifests", MappingProxyType(normalized))
        if self.family_isolation_validated is not True or self.final_locked is not True:
            raise RouterContractError("corpus manifest requires isolation and a locked final split")
        if self.schema_version != M5A_CORPUS_SCHEMA_VERSION:
            raise RouterContractError("unknown corpus manifest schema version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_fingerprint": self.corpus_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "split_manifests": {
                split.value: self.split_manifests[split].to_dict() for split in LanguageSplit
            },
            "family_isolation_validated": self.family_isolation_validated,
            "final_locked": self.final_locked,
            "schema_version": self.schema_version,
        }


DEFAULT_LANGUAGE_CORPUS_CONFIG = LanguageCorpusConfig(
    routeable_per_task=MappingProxyType(
        {
            LanguageSplit.TRAIN: 100,
            LanguageSplit.VALIDATION: 30,
            LanguageSplit.DEVELOPMENT: 40,
            LanguageSplit.FINAL: 60,
        }
    ),
    rejected_total=MappingProxyType(
        {
            LanguageSplit.TRAIN: 300,
            LanguageSplit.VALIDATION: 120,
            LanguageSplit.DEVELOPMENT: 180,
            LanguageSplit.FINAL: 240,
        }
    ),
)


def validate_supported_task_spec(task_spec: TaskSpec) -> TaskSpec:
    """Validate the exact six M5A task values without inventing a fallback."""
    if task_spec.target_object_id not in OBJECT_IDS or task_spec.target_bin_id not in BIN_IDS:
        raise RouterContractError("unsupported M5A TaskSpec")
    if task_spec.instruction_template_id != "canonical_v0":
        raise RouterContractError("M5A TaskSpecs must use canonical_v0")
    return task_spec


def canonical_router_json(value: object) -> str:
    """Expose the repository canonical serializer at the language contract boundary."""
    return canonical_json(value)


def _require_sha256_fingerprint(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise RouterContractError(f"{name} must be a SHA-256 fingerprint")
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise RouterContractError(f"{name} must be a lowercase SHA-256 fingerprint")
    return value


def _require_git_commit(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RouterContractError("git_commit must be a full lowercase 40-character commit")
    return value


@dataclass(frozen=True, slots=True)
class RouterRuntimeIdentity:
    """Bind a router to the frozen controller and rollout runtime it may dispatch."""

    router_fingerprint: str
    controller_registry_fingerprint: str
    environment_id: str
    environment_contract_fingerprint: str
    rollout_config_fingerprint: str
    action_runtime_fingerprint: str
    git_commit: str
    runtime_fingerprint: str = ""
    schema_version: str = "langmani-m5a-router-runtime-v1"

    def __post_init__(self) -> None:
        for name in (
            "router_fingerprint",
            "controller_registry_fingerprint",
            "environment_contract_fingerprint",
            "rollout_config_fingerprint",
            "action_runtime_fingerprint",
        ):
            _require_sha256_fingerprint(getattr(self, name), name=name)
        _require_non_empty(self.environment_id, name="environment_id")
        _require_git_commit(self.git_commit)
        if self.schema_version != "langmani-m5a-router-runtime-v1":
            raise RouterContractError("unknown router runtime schema version")
        expected = f"sha256:{sha256_hex(self.identity_dict())}"
        if self.runtime_fingerprint and self.runtime_fingerprint != expected:
            raise RouterContractError("runtime_fingerprint does not match runtime identity")
        object.__setattr__(self, "runtime_fingerprint", expected)

    def identity_dict(self) -> dict[str, object]:
        return {
            "router_fingerprint": self.router_fingerprint,
            "controller_registry_fingerprint": self.controller_registry_fingerprint,
            "environment_id": self.environment_id,
            "environment_contract_fingerprint": self.environment_contract_fingerprint,
            "rollout_config_fingerprint": self.rollout_config_fingerprint,
            "action_runtime_fingerprint": self.action_runtime_fingerprint,
            "git_commit": self.git_commit,
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.identity_dict(), "runtime_fingerprint": self.runtime_fingerprint}


@dataclass(frozen=True, slots=True)
class RouterEvaluationRecord:
    """One language-only expected-versus-predicted routing observation."""

    example_id: str
    split: LanguageSplit
    template_family_id: str
    expected_status: RouterStatus
    expected_task_id: str | None
    expected_rejection_reason: RouterRejectionReason | None
    decision: RouterDecision
    latency_ms: float
    schema_valid: bool
    repeat_index: int = 0
    record_fingerprint: str = ""
    schema_version: str = "langmani-m5a-router-evaluation-record-v1"

    def __post_init__(self) -> None:
        _require_non_empty(self.example_id, name="example_id")
        _require_non_empty(self.template_family_id, name="template_family_id")
        if not isinstance(self.split, LanguageSplit):
            raise RouterContractError("evaluation split must be a LanguageSplit")
        if not isinstance(self.expected_status, RouterStatus):
            raise RouterContractError("expected_status must be a RouterStatus")
        if self.expected_status is RouterStatus.ROUTE:
            _require_non_empty(self.expected_task_id, name="expected_task_id")
            if self.expected_rejection_reason is not None:
                raise RouterContractError("routeable evaluation records cannot expect rejection")
        elif self.expected_task_id is not None or not isinstance(
            self.expected_rejection_reason, RouterRejectionReason
        ):
            raise RouterContractError("rejected evaluation records require only a rejection reason")
        if not isinstance(self.decision, RouterDecision):
            raise RouterContractError("decision must be a RouterDecision")
        if (
            isinstance(self.latency_ms, bool)
            or not isinstance(self.latency_ms, (int, float))
            or not math.isfinite(float(self.latency_ms))
            or self.latency_ms < 0
        ):
            raise RouterContractError("latency_ms must be a finite non-negative number")
        object.__setattr__(self, "latency_ms", float(self.latency_ms))
        if not isinstance(self.schema_valid, bool):
            raise RouterContractError("schema_valid must be a boolean")
        if (
            isinstance(self.repeat_index, bool)
            or not isinstance(self.repeat_index, int)
            or self.repeat_index < 0
        ):
            raise RouterContractError("repeat_index must be a non-negative integer")
        if self.schema_version != "langmani-m5a-router-evaluation-record-v1":
            raise RouterContractError("unknown router evaluation record schema")
        expected = _fingerprint("router-evaluation-record", self.identity_dict())
        if self.record_fingerprint and self.record_fingerprint != expected:
            raise RouterContractError("record_fingerprint does not match evaluation record")
        object.__setattr__(self, "record_fingerprint", expected)

    @property
    def exact_expected_match(self) -> bool:
        if self.decision.status is not self.expected_status:
            return False
        if self.expected_status is RouterStatus.ROUTE:
            return self.decision.task_id == self.expected_task_id
        return self.decision.rejection_reason is self.expected_rejection_reason

    @classmethod
    def from_example(
        cls,
        *,
        example: LanguageExample,
        decision: RouterDecision,
        latency_ms: float,
        schema_valid: bool,
        repeat_index: int = 0,
    ) -> RouterEvaluationRecord:
        return cls(
            example_id=example.example_id,
            split=example.split,
            template_family_id=example.template_family_id,
            expected_status=example.expected_status,
            expected_task_id=example.task_id,
            expected_rejection_reason=example.expected_rejection_reason,
            decision=decision,
            latency_ms=latency_ms,
            schema_valid=schema_valid,
            repeat_index=repeat_index,
        )

    def identity_dict(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "split": self.split.value,
            "template_family_id": self.template_family_id,
            "expected_status": self.expected_status.value,
            "expected_task_id": self.expected_task_id,
            "expected_rejection_reason": (
                None
                if self.expected_rejection_reason is None
                else self.expected_rejection_reason.value
            ),
            "decision": self.decision.to_dict(),
            "latency_ms": self.latency_ms,
            "schema_valid": self.schema_valid,
            "repeat_index": self.repeat_index,
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.identity_dict(), "record_fingerprint": self.record_fingerprint}


@dataclass(frozen=True, slots=True)
class RouterEvaluationSummary:
    """Content-bound aggregate language metrics with explicit denominators."""

    router_name: str
    router_version: str
    router_fingerprint: str
    split: LanguageSplit
    example_count: int
    routeable_count: int
    rejected_count: int
    record_fingerprints: tuple[str, ...]
    metrics: Mapping[str, object]
    confusion_matrices: Mapping[str, object]
    summary_fingerprint: str = ""
    schema_version: str = "langmani-m5a-router-evaluation-summary-v1"

    def __post_init__(self) -> None:
        _require_non_empty(self.router_name, name="router_name")
        _require_non_empty(self.router_version, name="router_version")
        _require_sha256_fingerprint(self.router_fingerprint, name="router_fingerprint")
        if not isinstance(self.split, LanguageSplit):
            raise RouterContractError("summary split must be a LanguageSplit")
        for name in ("example_count", "routeable_count", "rejected_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RouterContractError(f"{name} must be a non-negative integer")
        if self.routeable_count + self.rejected_count != self.example_count:
            raise RouterContractError(
                "routeable and rejected denominators must sum to example_count"
            )
        if len(self.record_fingerprints) != self.example_count:
            raise RouterContractError("record_fingerprints must contain one value per example")
        if len(set(self.record_fingerprints)) != len(self.record_fingerprints):
            raise RouterContractError("record_fingerprints must be unique")
        for fingerprint in self.record_fingerprints:
            _require_non_empty(fingerprint, name="record_fingerprint")
        if not isinstance(self.metrics, Mapping) or not isinstance(
            self.confusion_matrices, Mapping
        ):
            raise RouterContractError("summary metrics and confusion matrices must be mappings")
        object.__setattr__(self, "metrics", cast(Mapping[str, object], _freeze_json(self.metrics)))
        object.__setattr__(
            self,
            "confusion_matrices",
            cast(Mapping[str, object], _freeze_json(self.confusion_matrices)),
        )
        if self.schema_version != "langmani-m5a-router-evaluation-summary-v1":
            raise RouterContractError("unknown router evaluation summary schema")
        expected = _fingerprint("router-evaluation-summary", self.identity_dict())
        if self.summary_fingerprint and self.summary_fingerprint != expected:
            raise RouterContractError("summary_fingerprint does not match evaluation summary")
        object.__setattr__(self, "summary_fingerprint", expected)

    def identity_dict(self) -> dict[str, object]:
        return {
            "router_name": self.router_name,
            "router_version": self.router_version,
            "router_fingerprint": self.router_fingerprint,
            "split": self.split.value,
            "example_count": self.example_count,
            "routeable_count": self.routeable_count,
            "rejected_count": self.rejected_count,
            "record_fingerprints": list(self.record_fingerprints),
            "metrics": _thaw_json(self.metrics),
            "confusion_matrices": _thaw_json(self.confusion_matrices),
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.identity_dict(), "summary_fingerprint": self.summary_fingerprint}


@dataclass(frozen=True, slots=True)
class M5ADevelopmentGate:
    """Exact predeclared language and end-to-end development authorization gate."""

    candidate_router_name: str
    candidate_router_fingerprint: str
    candidate_is_learned: bool
    full_task_spec_accuracy: float
    object_accuracy: float
    bin_accuracy: float
    false_route_rate: float
    ambiguous_rejection_recall: float
    unsupported_rejection_recall: float
    malformed_rejection_recall: float
    schema_valid_output_rate: float
    deterministic_repeatability: float
    llm_malformed_output_rate: float | None
    oracle_success_rate: float
    predicted_success_rate: float
    wrong_object_routing_count: int
    wrong_bin_routing_count: int
    false_rejection_count: int
    target_in_wrong_bin_count: int
    target_off_table_count: int
    arm_projection_count: int
    invalid_action_count: int
    dispatch_after_rejection_count: int
    m2_expert_call_count: int
    language_final_accessed: bool = False
    control_final_accessed: bool = False
    m42_final_accessed: bool = False
    test_split_accessed: bool = False
    historical_fresh_accessed: bool = False
    smolvla_go: bool = False
    gate_fingerprint: str = ""
    schema_version: str = "langmani-m5a-development-gate-v1"

    def __post_init__(self) -> None:
        _require_non_empty(self.candidate_router_name, name="candidate_router_name")
        _require_sha256_fingerprint(
            self.candidate_router_fingerprint,
            name="candidate_router_fingerprint",
        )
        if not isinstance(self.candidate_is_learned, bool):
            raise RouterContractError("candidate_is_learned must be a boolean")
        probability_fields = (
            "full_task_spec_accuracy",
            "object_accuracy",
            "bin_accuracy",
            "false_route_rate",
            "ambiguous_rejection_recall",
            "unsupported_rejection_recall",
            "malformed_rejection_recall",
            "schema_valid_output_rate",
            "deterministic_repeatability",
            "oracle_success_rate",
            "predicted_success_rate",
        )
        for name in probability_fields:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise RouterContractError(f"{name} must be finite and in [0, 1]")
            object.__setattr__(self, name, float(value))
        if self.llm_malformed_output_rate is not None:
            value = self.llm_malformed_output_rate
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise RouterContractError("llm_malformed_output_rate must be null or in [0, 1]")
            object.__setattr__(self, "llm_malformed_output_rate", float(value))
        count_fields = (
            "wrong_object_routing_count",
            "wrong_bin_routing_count",
            "false_rejection_count",
            "target_in_wrong_bin_count",
            "target_off_table_count",
            "arm_projection_count",
            "invalid_action_count",
            "dispatch_after_rejection_count",
            "m2_expert_call_count",
        )
        for name in count_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RouterContractError(f"{name} must be a non-negative integer")
        forbidden_flags = (
            "language_final_accessed",
            "control_final_accessed",
            "m42_final_accessed",
            "test_split_accessed",
            "historical_fresh_accessed",
            "smolvla_go",
        )
        if any(not isinstance(getattr(self, name), bool) for name in forbidden_flags):
            raise RouterContractError("development access flags must be booleans")
        if any(getattr(self, name) for name in forbidden_flags):
            raise RouterContractError("M5A development accessed a sealed or out-of-scope source")
        if self.schema_version != "langmani-m5a-development-gate-v1":
            raise RouterContractError("unknown M5A development gate schema")
        expected = _fingerprint("development-gate", self.identity_dict())
        if self.gate_fingerprint and self.gate_fingerprint != expected:
            raise RouterContractError("gate_fingerprint does not match development gate")
        object.__setattr__(self, "gate_fingerprint", expected)

    @property
    def language_gate_passed(self) -> bool:
        return (
            self.full_task_spec_accuracy >= 0.95
            and self.object_accuracy >= 0.97
            and self.bin_accuracy >= 0.97
            and self.false_route_rate <= 0.03
            and self.ambiguous_rejection_recall >= 0.90
            and self.unsupported_rejection_recall >= 0.95
            and self.malformed_rejection_recall >= 0.95
            and self.schema_valid_output_rate >= 0.99
            and self.deterministic_repeatability == 1.0
            and (self.llm_malformed_output_rate is None or self.llm_malformed_output_rate <= 0.01)
        )

    @property
    def end_to_end_gate_passed(self) -> bool:
        return (
            abs(self.oracle_success_rate - self.predicted_success_rate) <= 0.05
            and self.wrong_object_routing_count <= 2
            and self.wrong_bin_routing_count <= 2
            and self.false_rejection_count <= 3
            and self.target_in_wrong_bin_count == 0
            and self.target_off_table_count == 0
            and self.arm_projection_count == 0
            and self.invalid_action_count == 0
            and self.dispatch_after_rejection_count == 0
            and self.m2_expert_call_count == 0
        )

    @property
    def development_quality_gate_passed(self) -> bool:
        return (
            self.candidate_is_learned and self.language_gate_passed and self.end_to_end_gate_passed
        )

    @property
    def final_benchmark_authorized(self) -> bool:
        return self.development_quality_gate_passed

    def identity_dict(self) -> dict[str, object]:
        return {
            "candidate_router_name": self.candidate_router_name,
            "candidate_router_fingerprint": self.candidate_router_fingerprint,
            "candidate_is_learned": self.candidate_is_learned,
            "full_task_spec_accuracy": self.full_task_spec_accuracy,
            "object_accuracy": self.object_accuracy,
            "bin_accuracy": self.bin_accuracy,
            "false_route_rate": self.false_route_rate,
            "ambiguous_rejection_recall": self.ambiguous_rejection_recall,
            "unsupported_rejection_recall": self.unsupported_rejection_recall,
            "malformed_rejection_recall": self.malformed_rejection_recall,
            "schema_valid_output_rate": self.schema_valid_output_rate,
            "deterministic_repeatability": self.deterministic_repeatability,
            "llm_malformed_output_rate": self.llm_malformed_output_rate,
            "oracle_success_rate": self.oracle_success_rate,
            "predicted_success_rate": self.predicted_success_rate,
            "wrong_object_routing_count": self.wrong_object_routing_count,
            "wrong_bin_routing_count": self.wrong_bin_routing_count,
            "false_rejection_count": self.false_rejection_count,
            "target_in_wrong_bin_count": self.target_in_wrong_bin_count,
            "target_off_table_count": self.target_off_table_count,
            "arm_projection_count": self.arm_projection_count,
            "invalid_action_count": self.invalid_action_count,
            "dispatch_after_rejection_count": self.dispatch_after_rejection_count,
            "m2_expert_call_count": self.m2_expert_call_count,
            "language_gate_passed": self.language_gate_passed,
            "end_to_end_gate_passed": self.end_to_end_gate_passed,
            "development_quality_gate_passed": self.development_quality_gate_passed,
            "final_benchmark_authorized": self.final_benchmark_authorized,
            "language_final_accessed": self.language_final_accessed,
            "control_final_accessed": self.control_final_accessed,
            "m42_final_accessed": self.m42_final_accessed,
            "test_split_accessed": self.test_split_accessed,
            "historical_fresh_accessed": self.historical_fresh_accessed,
            "smolvla_go": self.smolvla_go,
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.identity_dict(), "gate_fingerprint": self.gate_fingerprint}


@dataclass(frozen=True, slots=True)
class M5AExperimentManifest:
    """Immutable identity binding corpus, schedules, routers, controllers, and code."""

    corpus_fingerprint: str
    schedule_fingerprints: Mapping[str, str]
    controller_registry_fingerprint: str
    router_runtime_identities: Mapping[str, RouterRuntimeIdentity]
    git_commit: str
    dependency_versions: Mapping[str, str]
    language_final_accessed: bool = False
    control_final_accessed: bool = False
    m42_final_accessed: bool = False
    test_split_accessed: bool = False
    historical_fresh_accessed: bool = False
    smolvla_go: bool = False
    experiment_fingerprint: str = ""
    schema_version: str = "langmani-m5a-experiment-manifest-v1"

    def __post_init__(self) -> None:
        _require_sha256_fingerprint(self.corpus_fingerprint, name="corpus_fingerprint")
        expected_schedule_ids = {
            "m5a_language_dev_v0",
            "m5a_language_final_v0",
            "m5a_control_dev_v0",
            "m5a_control_final_v0",
        }
        if (
            not isinstance(self.schedule_fingerprints, Mapping)
            or set(self.schedule_fingerprints) != expected_schedule_ids
        ):
            raise RouterContractError("experiment manifest requires all four M5A schedules")
        frozen_schedules: dict[str, str] = {}
        for schedule_id, fingerprint in self.schedule_fingerprints.items():
            frozen_schedules[schedule_id] = _require_sha256_fingerprint(
                fingerprint,
                name=f"schedule_fingerprints[{schedule_id}]",
            )
        object.__setattr__(
            self,
            "schedule_fingerprints",
            MappingProxyType(dict(sorted(frozen_schedules.items()))),
        )
        _require_sha256_fingerprint(
            self.controller_registry_fingerprint,
            name="controller_registry_fingerprint",
        )
        expected_routers = {
            "RuleRouterV0",
            "FactorizedTextClassifierV0",
            "StructuredLocalLLMRouterV0",
        }
        if (
            not isinstance(self.router_runtime_identities, Mapping)
            or set(self.router_runtime_identities) != expected_routers
        ):
            raise RouterContractError("experiment manifest requires exactly three router runtimes")
        frozen_runtimes: dict[str, RouterRuntimeIdentity] = {}
        for router_name, identity in self.router_runtime_identities.items():
            if not isinstance(identity, RouterRuntimeIdentity):
                raise RouterContractError("router runtime values must be RouterRuntimeIdentity")
            frozen_runtimes[router_name] = identity
        object.__setattr__(
            self,
            "router_runtime_identities",
            MappingProxyType(dict(sorted(frozen_runtimes.items()))),
        )
        _require_git_commit(self.git_commit)
        if not isinstance(self.dependency_versions, Mapping) or not self.dependency_versions:
            raise RouterContractError("dependency_versions must be a non-empty mapping")
        frozen_versions: dict[str, str] = {}
        for name, version in self.dependency_versions.items():
            frozen_versions[_require_non_empty(name, name="dependency name")] = _require_non_empty(
                version,
                name="dependency version",
            )
        object.__setattr__(
            self,
            "dependency_versions",
            MappingProxyType(dict(sorted(frozen_versions.items()))),
        )
        forbidden_flags = (
            "language_final_accessed",
            "control_final_accessed",
            "m42_final_accessed",
            "test_split_accessed",
            "historical_fresh_accessed",
            "smolvla_go",
        )
        if any(not isinstance(getattr(self, name), bool) for name in forbidden_flags):
            raise RouterContractError("experiment access flags must be booleans")
        if any(getattr(self, name) for name in forbidden_flags):
            raise RouterContractError("M5A development manifest accessed a forbidden source")
        if self.schema_version != "langmani-m5a-experiment-manifest-v1":
            raise RouterContractError("unknown M5A experiment manifest schema")
        expected = f"sha256:{sha256_hex(self.identity_dict())}"
        if self.experiment_fingerprint and self.experiment_fingerprint != expected:
            raise RouterContractError("experiment_fingerprint does not match manifest")
        object.__setattr__(self, "experiment_fingerprint", expected)

    def identity_dict(self) -> dict[str, object]:
        return {
            "corpus_fingerprint": self.corpus_fingerprint,
            "schedule_fingerprints": dict(self.schedule_fingerprints),
            "controller_registry_fingerprint": self.controller_registry_fingerprint,
            "router_runtime_identities": {
                name: identity.to_dict()
                for name, identity in self.router_runtime_identities.items()
            },
            "git_commit": self.git_commit,
            "dependency_versions": dict(self.dependency_versions),
            "language_final_accessed": self.language_final_accessed,
            "control_final_accessed": self.control_final_accessed,
            "m42_final_accessed": self.m42_final_accessed,
            "test_split_accessed": self.test_split_accessed,
            "historical_fresh_accessed": self.historical_fresh_accessed,
            "smolvla_go": self.smolvla_go,
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.identity_dict(), "experiment_fingerprint": self.experiment_fingerprint}
