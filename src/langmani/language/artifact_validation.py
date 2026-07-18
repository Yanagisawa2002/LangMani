"""Read-only validators for immutable M5A production evidence.

These validators intentionally do not load a model, decode a dataset, create an
environment, or mutate an artifact.  They independently walk every declared file,
recompute content digests, and return compact identities suitable for a verifier's
cross-stage provenance checks.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol, cast

from langmani.datasets.identity import canonical_json, sha256_hex
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import EpisodeSpec, TaskSpec, stable_task_id
from langmani.language.controller_registry import ControllerRegistry, ControllerRegistryError
from langmani.language.evaluation import (
    ActionStreamEvidence,
    ControlExecutionResult,
    ControllerDispatchRecord,
    EndToEndEpisodeResult,
    M5AEvaluationError,
)
from langmani.language.failure_attribution import FailureAttribution
from langmani.language.router_evaluation import RouterEvaluationError
from langmani.language.router_evaluation import (
    RouterEvaluationRecord as LanguageRouterEvaluationRecord,
)
from langmani.language.router_evaluation import (
    RouterEvaluationSummary as LanguageRouterEvaluationSummary,
)
from langmani.language.router_types import (
    LanguageExample,
    LanguageSplit,
    RouterConfidence,
    RouterContractError,
    RouterDecision,
    RouterRejectionReason,
    RouterStatus,
)
from langmani.language.schedules import ScheduledControlEpisode, StagedControlSchedule

CORPUS_ARCHIVE_SCHEMA = "langmani-m5a-language-corpus-archive-v1"
ROUTER_EVALUATION_EVIDENCE_SCHEMA = "langmani-m5a-language-router-evaluation-evidence-v1"
CONTROL_RUN_SCHEMA = "langmani-m5a-language-control-run-v0"
CONTROL_EPISODE_SCHEMA = "langmani-m5a-language-control-episode-v0"
CONTROL_COMPLETION_SCHEMA = "langmani-m5a-language-control-complete-v0"
REJECTION_NOOP_PROBE_SCHEMA = "langmani-m5a-rejection-noop-probe-v0"
CONTROL_ROUTER_ORDER = ("oracle", "classifier", "llm")
_LANGUAGE_ROUTER_NAMES = {
    "rule": "RuleRouterV0",
    "classifier": "FactorizedTextClassifierV0",
    "llm": "StructuredLocalLLMRouterV0",
}
_CONTROL_ROUTER_NAMES = {"oracle": "OracleTaskSpecRouterV0", **_LANGUAGE_ROUTER_NAMES}


class M5AArtifactValidationError(RuntimeError):
    """Raised when immutable M5A evidence is missing, unsafe, or content-inconsistent."""


class DevelopmentControlInputsLike(Protocol):
    """Structural boundary accepted from the control command's frozen input bundle."""

    schedule: StagedControlSchedule
    episodes: tuple[ScheduledControlEpisode, ...]
    examples_by_id: Mapping[str, LanguageExample]
    corpus_fingerprint: str


@dataclass(frozen=True, slots=True)
class ValidatedCorpusArchive:
    root: Path
    corpus_fingerprint: str
    archive_fingerprint: str
    artifact_count: int
    artifact_records_fingerprint: str

    def to_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "corpus_fingerprint": self.corpus_fingerprint,
            "archive_fingerprint": self.archive_fingerprint,
            "artifact_count": self.artifact_count,
            "artifact_records_fingerprint": self.artifact_records_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class ValidatedRouterEvaluationEvidence:
    root: Path
    evaluation_fingerprint: str
    artifact_fingerprint: str
    artifact_count: int
    owner_fingerprint: str
    result_fingerprint: str
    owner: Mapping[str, object]
    result: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "evaluation_fingerprint": self.evaluation_fingerprint,
            "artifact_fingerprint": self.artifact_fingerprint,
            "artifact_count": self.artifact_count,
            "owner_fingerprint": self.owner_fingerprint,
            "result_fingerprint": self.result_fingerprint,
            "owner": dict(self.owner),
            "result": dict(self.result),
        }


@dataclass(frozen=True, slots=True)
class ValidatedControlEvidence:
    root: Path
    run_fingerprint: str
    schedule_fingerprint: str
    corpus_fingerprint: str
    controller_registry_fingerprint: str
    record_count: int
    record_set_fingerprint: str
    completion_fingerprint: str
    owner_fingerprint: str
    identity: Mapping[str, object]
    summaries: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "run_fingerprint": self.run_fingerprint,
            "schedule_fingerprint": self.schedule_fingerprint,
            "corpus_fingerprint": self.corpus_fingerprint,
            "controller_registry_fingerprint": self.controller_registry_fingerprint,
            "record_count": self.record_count,
            "record_set_fingerprint": self.record_set_fingerprint,
            "completion_fingerprint": self.completion_fingerprint,
            "owner_fingerprint": self.owner_fingerprint,
            "identity": dict(self.identity),
            "summaries": dict(self.summaries),
        }


def _require_real_directory(root: str | Path, *, label: str) -> Path:
    path = Path(os.path.abspath(os.fspath(Path(root).expanduser())))
    for component in (path, *path.parents):
        if component.is_symlink() or component.is_junction():
            raise M5AArtifactValidationError(
                f"{label} traverses a symlink or junction: {component}"
            )
    if not path.is_dir() or path.is_symlink() or path.is_junction():
        raise M5AArtifactValidationError(f"{label} must be one real directory")
    return path.resolve(strict=True)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON keys instead of accepting parser last-write wins."""

    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_object(path: Path, *, label: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink() or path.is_junction():
        raise M5AArtifactValidationError(f"{label} must be one real JSON file")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {constant}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise M5AArtifactValidationError(f"could not parse {label}: {error}") from error
    if not isinstance(value, dict):
        raise M5AArtifactValidationError(f"{label} must contain one JSON object")
    return cast(dict[str, object], value)


def _read_jsonl_objects(path: Path, *, label: str) -> tuple[dict[str, object], ...]:
    if not path.is_file() or path.is_symlink() or path.is_junction():
        raise M5AArtifactValidationError(f"{label} must be one real JSONL file")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines or any(not line for line in lines):
            raise ValueError("JSONL must contain non-empty records without blank lines")
        values = tuple(
            json.loads(
                line,
                object_pairs_hook=_unique_object,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant {constant}")
                ),
            )
            for line in lines
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise M5AArtifactValidationError(f"could not parse {label}: {error}") from error
    if any(not isinstance(value, dict) for value in values):
        raise M5AArtifactValidationError(f"{label} records must be JSON objects")
    return cast(tuple[dict[str, object], ...], values)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _artifact_records(
    root: Path,
    *,
    excluded: frozenset[str] = frozenset({"artifact_manifest.json", "complete.json"}),
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for candidate in root.rglob("*"):
        if candidate.is_symlink() or candidate.is_junction():
            raise M5AArtifactValidationError(
                f"immutable evidence contains a linked entry: {candidate}"
            )
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise M5AArtifactValidationError(
                f"immutable evidence contains a special entry: {candidate}"
            )
        relative = PurePosixPath(candidate.relative_to(root)).as_posix()
        if relative in excluded:
            continue
        records.append(
            {
                "path": relative,
                "size_bytes": candidate.stat().st_size,
                "sha256": _sha256_file(candidate),
            }
        )
    return sorted(records, key=lambda value: cast(str, value["path"]))


def validate_language_corpus_archive(
    root: str | Path,
    *,
    expected_corpus_fingerprint: str | None = None,
) -> ValidatedCorpusArchive:
    """Rehash every corpus payload and validate its manifest and completion marker."""

    archive_root = _require_real_directory(root, label="language corpus archive")
    manifest = _read_object(
        archive_root / "artifact_manifest.json", label="corpus artifact manifest"
    )
    complete = _read_object(archive_root / "complete.json", label="corpus completion")
    corpus_fingerprint = manifest.get("corpus_fingerprint")
    if not isinstance(corpus_fingerprint, str):
        raise M5AArtifactValidationError("corpus fingerprint is missing")
    if (
        expected_corpus_fingerprint is not None
        and corpus_fingerprint != expected_corpus_fingerprint
    ):
        raise M5AArtifactValidationError("corpus fingerprint differs from the expected corpus")
    artifacts = _artifact_records(archive_root)
    archive_fingerprint = (
        f"sha256:{sha256_hex({'corpus_fingerprint': corpus_fingerprint, 'artifacts': artifacts})}"
    )
    if manifest != {
        "schema_version": CORPUS_ARCHIVE_SCHEMA,
        "corpus_fingerprint": corpus_fingerprint,
        "archive_fingerprint": archive_fingerprint,
        "artifacts": artifacts,
    }:
        raise M5AArtifactValidationError("corpus artifact manifest differs from disk")
    if complete != {
        "schema_version": CORPUS_ARCHIVE_SCHEMA,
        "corpus_fingerprint": corpus_fingerprint,
        "archive_fingerprint": archive_fingerprint,
        "passed": True,
    }:
        raise M5AArtifactValidationError("corpus completion marker differs from disk")
    return ValidatedCorpusArchive(
        root=archive_root,
        corpus_fingerprint=corpus_fingerprint,
        archive_fingerprint=archive_fingerprint,
        artifact_count=len(artifacts),
        artifact_records_fingerprint=f"sha256:{sha256_hex(artifacts)}",
    )


_TASK_BY_ID = {stable_task_id(task_spec): task_spec for task_spec in CANONICAL_TASK_SPECS}


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise M5AArtifactValidationError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _json_equal(left: object, right: object) -> bool:
    """Compare JSON values without Python's ``True == 1`` coercion."""

    return canonical_json(left) == canonical_json(right)


def _parse_language_record(
    payload: Mapping[str, object],
    *,
    split: LanguageSplit,
    repeat_count: int,
    router_name: str,
) -> LanguageRouterEvaluationRecord:
    try:
        decision_payload = _mapping(payload.get("decision"), label="router decision")
        decision = RouterDecision.from_dict(decision_payload)
        example_id = payload["example_id"]
        template_family_id = payload["template_family_id"]
        expected_task_id = payload["expected_task_id"]
        latency_ms = payload["latency_ms"]
        repeat_fingerprints = _sequence(
            payload["repeat_decision_fingerprints"],
            label="router repeat decision fingerprints",
        )
        if not isinstance(example_id, str) or not example_id:
            raise TypeError("example_id must be a non-empty string")
        if not isinstance(template_family_id, str) or not template_family_id:
            raise TypeError("template_family_id must be a non-empty string")
        if expected_task_id is not None and not isinstance(expected_task_id, str):
            raise TypeError("expected_task_id must be a string or null")
        if (
            isinstance(latency_ms, bool)
            or not isinstance(latency_ms, int | float)
            or not math.isfinite(float(latency_ms))
            or float(latency_ms) < 0.0
        ):
            raise TypeError("latency_ms must be finite and non-negative")
        if any(not isinstance(value, str) for value in repeat_fingerprints):
            raise TypeError("repeat decision fingerprints must be strings")
        rejection_value = payload.get("expected_rejection_reason")
        record = LanguageRouterEvaluationRecord(
            example_id=example_id,
            template_family_id=template_family_id,
            split=LanguageSplit(cast(str, payload["split"])),
            expected_status=RouterStatus(cast(str, payload["expected_status"])),
            expected_task_id=expected_task_id,
            expected_rejection_reason=(
                None
                if rejection_value is None
                else RouterRejectionReason(cast(str, rejection_value))
            ),
            decision=decision,
            latency_ms=float(latency_ms),
            repeat_decision_fingerprints=cast(tuple[str, ...], tuple(repeat_fingerprints)),
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        RouterContractError,
        RouterEvaluationError,
    ) as error:
        raise M5AArtifactValidationError(
            f"router evaluation record is malformed: {error}"
        ) from error
    if not _json_equal(record.to_dict(), dict(payload)):
        raise M5AArtifactValidationError(
            "router evaluation record fields or derived semantics differ"
        )
    if record.split is not split:
        raise M5AArtifactValidationError("router evaluation record split differs from its path")
    if record.decision.router_name != router_name:
        raise M5AArtifactValidationError("router decision identity differs from its evidence path")
    if len(record.repeat_decision_fingerprints) != repeat_count:
        raise M5AArtifactValidationError("router repeat evidence count differs from owner")
    if record.repeat_decision_fingerprints[0] != record.decision.decision_fingerprint:
        raise M5AArtifactValidationError("router primary decision is absent from repeat evidence")
    if record.expected_status is RouterStatus.ROUTE:
        if (
            record.expected_task_id not in _TASK_BY_ID
            or record.expected_rejection_reason is not None
        ):
            raise M5AArtifactValidationError(
                "routeable router evidence lacks one canonical expected TaskSpec"
            )
    elif record.expected_task_id is not None or record.expected_rejection_reason is None:
        raise M5AArtifactValidationError(
            "rejection router evidence contains an expected task or lacks a reason"
        )
    return record


def _safe_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _linear_percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise M5AArtifactValidationError("language latency percentile has no observations")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _language_ece(
    records: Sequence[LanguageRouterEvaluationRecord], *, bins: int = 10
) -> tuple[float | None, int]:
    selected = tuple(record for record in records if record.decision.confidence.available)
    if not selected:
        return None, 0
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        bucket = tuple(
            record
            for record in selected
            if record.decision.confidence.score is not None
            and lower <= record.decision.confidence.score
            and (
                record.decision.confidence.score < upper
                or (index == bins - 1 and record.decision.confidence.score <= upper)
            )
        )
        if not bucket:
            continue
        confidence = sum(cast(float, record.decision.confidence.score) for record in bucket) / len(
            bucket
        )
        accuracy = sum(record.full_task_correct for record in bucket) / len(bucket)
        error += len(bucket) / len(selected) * abs(accuracy - confidence)
    return error, len(selected)


def _status_recall(
    records: Sequence[LanguageRouterEvaluationRecord], status: RouterStatus
) -> float:
    selected = tuple(record for record in records if record.expected_status is status)
    return _safe_ratio(
        sum(record.decision.status is status for record in selected),
        len(selected),
    )


def _language_summary(
    records: Sequence[LanguageRouterEvaluationRecord], *, split: LanguageSplit
) -> dict[str, object]:
    values = tuple(records)
    routeable = tuple(record for record in values if record.expected_status is RouterStatus.ROUTE)
    rejected = tuple(
        record for record in values if record.expected_status is not RouterStatus.ROUTE
    )
    predicted_rejections = tuple(
        record for record in values if record.decision.status is not RouterStatus.ROUTE
    )
    true_rejections = sum(
        record.expected_status is not RouterStatus.ROUTE for record in predicted_rejections
    )
    rejection_precision = _safe_ratio(true_rejections, len(predicted_rejections))
    rejection_recall = _safe_ratio(true_rejections, len(rejected))
    rejection_f1 = (
        0.0
        if rejection_precision + rejection_recall == 0.0
        else 2.0 * rejection_precision * rejection_recall / (rejection_precision + rejection_recall)
    )
    routed = tuple(record for record in values if record.decision.status is RouterStatus.ROUTE)
    malformed_reasons = {
        RouterRejectionReason.STRUCTURED_OUTPUT_INVALID,
        RouterRejectionReason.FORMAT_REPAIR_EXHAUSTED,
    }
    status_confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    rejection_confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    task_groups: dict[str, list[LanguageRouterEvaluationRecord]] = defaultdict(list)
    family_groups: dict[str, list[LanguageRouterEvaluationRecord]] = defaultdict(list)
    for record in values:
        status_confusion[record.expected_status.value][record.decision.status.value] += 1
        if record.expected_status is not RouterStatus.ROUTE:
            expected_reason = cast(RouterRejectionReason, record.expected_rejection_reason).value
            predicted_reason = (
                "none"
                if record.decision.rejection_reason is None
                else record.decision.rejection_reason.value
            )
            rejection_confusion[expected_reason][predicted_reason] += 1
        if record.expected_task_id is not None:
            task_groups[record.expected_task_id].append(record)
        family_groups[record.template_family_id].append(record)
    ece, confidence_examples = _language_ece(values)
    summary = LanguageRouterEvaluationSummary(
        split=split,
        example_count=len(values),
        routeable_count=len(routeable),
        rejected_count=len(rejected),
        status_accuracy=_safe_ratio(sum(record.status_correct for record in values), len(values)),
        valid_full_task_accuracy=_safe_ratio(
            sum(record.full_task_correct for record in routeable), len(routeable)
        ),
        object_accuracy=_safe_ratio(
            sum(
                record.decision.status is RouterStatus.ROUTE
                and record.decision.target_object_id
                == _TASK_BY_ID[cast(str, record.expected_task_id)].target_object_id
                for record in routeable
            ),
            len(routeable),
        ),
        bin_accuracy=_safe_ratio(
            sum(
                record.decision.status is RouterStatus.ROUTE
                and record.decision.target_bin_id
                == _TASK_BY_ID[cast(str, record.expected_task_id)].target_bin_id
                for record in routeable
            ),
            len(routeable),
        ),
        route_recall=_safe_ratio(
            sum(record.decision.status is RouterStatus.ROUTE for record in routeable),
            len(routeable),
        ),
        false_rejection_rate=_safe_ratio(
            sum(record.decision.status is not RouterStatus.ROUTE for record in routeable),
            len(routeable),
        ),
        rejection_precision=rejection_precision,
        rejection_recall=rejection_recall,
        rejection_f1=rejection_f1,
        false_route_rate=_safe_ratio(
            sum(record.decision.status is RouterStatus.ROUTE for record in rejected),
            len(rejected),
        ),
        rejection_reason_accuracy=_safe_ratio(
            sum(
                record.decision.rejection_reason is record.expected_rejection_reason
                for record in rejected
            ),
            len(rejected),
        ),
        ambiguous_rejection_recall=_status_recall(values, RouterStatus.REJECT_AMBIGUOUS),
        unsupported_rejection_recall=_status_recall(values, RouterStatus.REJECT_UNSUPPORTED),
        malformed_rejection_recall=_status_recall(values, RouterStatus.REJECT_MALFORMED),
        schema_valid_output_rate=1.0,
        malformed_output_rate=_safe_ratio(
            sum(record.decision.rejection_reason in malformed_reasons for record in values),
            len(values),
        ),
        coverage=_safe_ratio(len(routed), len(values)),
        selective_accuracy=_safe_ratio(
            sum(record.full_task_correct for record in routed), len(routed)
        ),
        expected_calibration_error=ece,
        confidence_examples=confidence_examples,
        latency_p50_ms=_linear_percentile([record.latency_ms for record in values], 0.50),
        latency_p95_ms=_linear_percentile([record.latency_ms for record in values], 0.95),
        latency_p99_ms=_linear_percentile([record.latency_ms for record in values], 0.99),
        deterministic_repeatability=_safe_ratio(
            sum(record.deterministic for record in values), len(values)
        ),
        status_confusion={
            key: dict(sorted(value.items())) for key, value in sorted(status_confusion.items())
        },
        rejection_reason_confusion={
            key: dict(sorted(value.items())) for key, value in sorted(rejection_confusion.items())
        },
        per_task_accuracy={
            key: _safe_ratio(sum(record.full_task_correct for record in group), len(group))
            for key, group in sorted(task_groups.items())
        },
        per_family_accuracy={
            key: _safe_ratio(sum(record.exact_decision_correct for record in group), len(group))
            for key, group in sorted(family_groups.items())
        },
    )
    return summary.to_dict()


def _language_supplemental(
    records: Sequence[LanguageRouterEvaluationRecord], *, structured_llm: bool
) -> dict[str, object]:
    values = tuple(records)
    routeable = tuple(record for record in values if record.expected_status is RouterStatus.ROUTE)
    rejected = tuple(
        record for record in values if record.expected_status is not RouterStatus.ROUTE
    )
    exhausted = repaired = initial_valid = 0
    for record in values:
        if structured_llm:
            errors = record.decision.evidence.get("parse_errors")
            error_count = len(errors) if isinstance(errors, tuple) else 0
            if record.decision.rejection_reason is RouterRejectionReason.FORMAT_REPAIR_EXHAUSTED:
                exhausted += 1
            elif error_count:
                repaired += 1
            else:
                initial_valid += 1
        else:
            initial_valid += 1
    object_confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    bin_confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    object_correct = bin_correct = 0
    for record in routeable:
        expected = _TASK_BY_ID[cast(str, record.expected_task_id)]
        predicted_object = record.decision.target_object_id or "none"
        predicted_bin = record.decision.target_bin_id or "none"
        object_confusion[expected.target_object_id][predicted_object] += 1
        bin_confusion[expected.target_bin_id][predicted_bin] += 1
        object_correct += predicted_object == expected.target_object_id
        bin_correct += predicted_bin == expected.target_bin_id
    return {
        "schema_valid_output_rate": _safe_ratio(len(values) - exhausted, len(values)),
        "initial_output_schema_valid_rate": _safe_ratio(initial_valid, len(values)),
        "format_repair_success_count": repaired,
        "format_repair_exhausted_count": exhausted,
        "structured_output_malformed_rate": _safe_ratio(exhausted, len(values)),
        "routeable_denominator": len(routeable),
        "rejected_denominator": len(rejected),
        "object_accuracy": _safe_ratio(object_correct, len(routeable)),
        "bin_accuracy": _safe_ratio(bin_correct, len(routeable)),
        "route_recall": _safe_ratio(
            sum(record.decision.status is RouterStatus.ROUTE for record in routeable),
            len(routeable),
        ),
        "false_rejection_rate": _safe_ratio(
            sum(record.decision.status is not RouterStatus.ROUTE for record in routeable),
            len(routeable),
        ),
        "false_route_rate": _safe_ratio(
            sum(record.decision.status is RouterStatus.ROUTE for record in rejected),
            len(rejected),
        ),
        "ambiguous_rejection_recall": _status_recall(values, RouterStatus.REJECT_AMBIGUOUS),
        "unsupported_rejection_recall": _status_recall(values, RouterStatus.REJECT_UNSUPPORTED),
        "malformed_rejection_recall": _status_recall(values, RouterStatus.REJECT_MALFORMED),
        "object_confusion": {
            key: dict(sorted(value.items())) for key, value in sorted(object_confusion.items())
        },
        "bin_confusion": {
            key: dict(sorted(value.items())) for key, value in sorted(bin_confusion.items())
        },
    }


def _validate_language_record_sets(
    root: Path,
    *,
    owner: Mapping[str, object],
    result: Mapping[str, object],
    expected_examples_by_split: Mapping[LanguageSplit, Sequence[LanguageExample]],
) -> dict[str, object]:
    identities = _mapping(owner.get("router_identities"), label="router identities")
    if set(identities) != {"rule", "classifier", "llm"}:
        raise M5AArtifactValidationError("router evidence must contain exactly three routers")
    repeat_count = owner.get("repeat_count")
    if isinstance(repeat_count, bool) or not isinstance(repeat_count, int) or repeat_count < 2:
        raise M5AArtifactValidationError("router repeat_count must be an integer of at least two")
    local_files_only = owner.get("local_files_only")
    if not isinstance(local_files_only, bool):
        raise M5AArtifactValidationError("router local_files_only provenance must be boolean")
    if result.get("repeat_count") != repeat_count:
        raise M5AArtifactValidationError("router result repeat_count differs from its owner")
    if result.get("local_files_only") is not local_files_only:
        raise M5AArtifactValidationError(
            "router result local_files_only provenance differs from its owner"
        )
    comparison: dict[str, object] = {}
    ids_by_split: dict[LanguageSplit, tuple[str, ...]] = {}
    expected_files = {"owner.json", "result.json", "artifact_manifest.json", "complete.json"}
    for split in (LanguageSplit.VALIDATION, LanguageSplit.DEVELOPMENT):
        expected_examples = tuple(expected_examples_by_split.get(split, ()))
        if (
            not expected_examples
            or any(example.split is not split for example in expected_examples)
            or len({example.example_id for example in expected_examples}) != len(expected_examples)
        ):
            raise M5AArtifactValidationError(
                f"authoritative {split.value} language examples are missing or malformed"
            )
        expected_by_id = {example.example_id: example for example in expected_examples}
        expected_ids = tuple(sorted(expected_by_id))
        split_payload: dict[str, object] = {}
        reference_ids: tuple[str, ...] | None = None
        for router_label in ("rule", "classifier", "llm"):
            identity = _mapping(identities[router_label], label=f"{router_label} router identity")
            router_name = _LANGUAGE_ROUTER_NAMES[router_label]
            declared_router_name = identity.get("router_name")
            if declared_router_name not in {None, router_name}:
                raise M5AArtifactValidationError("router identity declares another router_name")
            router_fingerprint = identity.get("router_fingerprint")
            if not isinstance(router_fingerprint, str):
                raise M5AArtifactValidationError("router identity lacks router_fingerprint")
            record_relative = f"{split.value}/{router_label}/records.jsonl"
            summary_relative = f"{split.value}/{router_label}/summary.json"
            expected_files.update((record_relative, summary_relative))
            payloads = _read_jsonl_objects(
                root / record_relative,
                label=f"{split.value} {router_label} records",
            )
            records = tuple(
                _parse_language_record(
                    payload,
                    split=split,
                    repeat_count=repeat_count,
                    router_name=router_name,
                )
                for payload in payloads
            )
            record_ids = tuple(record.example_id for record in records)
            if record_ids != tuple(sorted(record_ids)) or len(set(record_ids)) != len(record_ids):
                raise M5AArtifactValidationError(
                    "router record IDs must be unique and in canonical order"
                )
            if record_ids != expected_ids:
                raise M5AArtifactValidationError(
                    f"{split.value} router records differ from authoritative corpus IDs"
                )
            for record in records:
                expected = expected_by_id[record.example_id]
                if (
                    record.template_family_id != expected.template_family_id
                    or record.expected_status is not expected.expected_status
                    or record.expected_task_id != expected.task_id
                    or record.expected_rejection_reason is not expected.expected_rejection_reason
                ):
                    raise M5AArtifactValidationError(
                        f"router record {record.example_id} differs from authoritative corpus"
                    )
            if reference_ids is None:
                reference_ids = record_ids
            elif record_ids != reference_ids:
                raise M5AArtifactValidationError(
                    "three routers evaluated different language example sets"
                )
            expected_summary = {
                "summary": _language_summary(records, split=split),
                "supplemental_metrics": _language_supplemental(
                    records, structured_llm=router_label == "llm"
                ),
            }
            observed_summary = _read_object(
                root / summary_relative,
                label=f"{split.value} {router_label} summary",
            )
            if not _json_equal(observed_summary, expected_summary):
                raise M5AArtifactValidationError(
                    f"{split.value} {router_label} summary differs from atomic records"
                )
            split_payload[router_label] = expected_summary
        assert reference_ids is not None
        ids_by_split[split] = reference_ids
        comparison[split.value] = split_payload
    observed_files = {
        PurePosixPath(path.relative_to(root)).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and not path.is_junction()
    }
    if observed_files != expected_files:
        raise M5AArtifactValidationError("router evaluation evidence file set differs")
    if not _json_equal(result.get("comparison"), comparison):
        raise M5AArtifactValidationError("router comparison differs from atomic records")
    validation_lock = _mapping(
        result.get("language_validation_schedule"), label="language validation schedule"
    )
    if (
        validation_lock.get("split") != "validation"
        or validation_lock.get("example_count") != len(ids_by_split[LanguageSplit.VALIDATION])
        or validation_lock.get("content_fingerprint")
        != owner.get("language_validation_fingerprint")
    ):
        raise M5AArtifactValidationError("language validation schedule differs from records")
    development_lock = _mapping(
        result.get("language_development_schedule"), label="language development schedule"
    )
    if (
        development_lock.get("split") != "development"
        or development_lock.get("ordered_example_ids")
        != list(ids_by_split[LanguageSplit.DEVELOPMENT])
        or development_lock.get("schedule_fingerprint")
        != owner.get("language_development_schedule_fingerprint")
    ):
        raise M5AArtifactValidationError("language development schedule differs from records")
    return comparison


def validate_router_evaluation_evidence(
    root: str | Path,
    *,
    expected_examples_by_split: Mapping[LanguageSplit, Sequence[LanguageExample]],
    expected_evaluation_fingerprint: str | None = None,
) -> ValidatedRouterEvaluationEvidence:
    """Rehash a promoted language-router evaluation and validate all identity links."""

    evidence_root = _require_real_directory(root, label="router evaluation evidence")
    owner = _read_object(evidence_root / "owner.json", label="router evaluation owner")
    result = _read_object(evidence_root / "result.json", label="router evaluation result")
    manifest = _read_object(
        evidence_root / "artifact_manifest.json",
        label="router evaluation artifact manifest",
    )
    complete = _read_object(evidence_root / "complete.json", label="router evaluation completion")
    evaluation_fingerprint = owner.get("evaluation_fingerprint")
    if not isinstance(evaluation_fingerprint, str):
        raise M5AArtifactValidationError("router evaluation fingerprint is missing")
    owner_payload = dict(owner)
    owner_payload.pop("evaluation_fingerprint", None)
    if evaluation_fingerprint != f"sha256:{sha256_hex(owner_payload)}":
        raise M5AArtifactValidationError("router evaluation owner fingerprint differs")
    if evidence_root.name != evaluation_fingerprint.removeprefix("sha256:"):
        raise M5AArtifactValidationError("router evaluation is not at its promoted run path")
    if (
        expected_evaluation_fingerprint is not None
        and evaluation_fingerprint != expected_evaluation_fingerprint
    ):
        raise M5AArtifactValidationError("router evaluation fingerprint differs from expected")
    artifacts = _artifact_records(evidence_root)
    artifact_fingerprint = f"sha256:{sha256_hex({'owner': owner, 'artifacts': artifacts})}"
    if manifest != {
        "schema_version": ROUTER_EVALUATION_EVIDENCE_SCHEMA,
        "artifact_fingerprint": artifact_fingerprint,
        "artifacts": artifacts,
    }:
        raise M5AArtifactValidationError("router evaluation artifact manifest differs from disk")
    if complete != {
        "schema_version": ROUTER_EVALUATION_EVIDENCE_SCHEMA,
        "evaluation_fingerprint": evaluation_fingerprint,
        "artifact_fingerprint": artifact_fingerprint,
        "passed": True,
    }:
        raise M5AArtifactValidationError("router evaluation completion differs from disk")
    if (
        owner.get("schema_version") != ROUTER_EVALUATION_EVIDENCE_SCHEMA
        or result.get("schema_version") != ROUTER_EVALUATION_EVIDENCE_SCHEMA
        or result.get("passed") is not True
        or result.get("evaluation_fingerprint") != evaluation_fingerprint
    ):
        raise M5AArtifactValidationError("router evaluation result identity differs")
    for key in (
        "mode",
        "corpus_fingerprint",
        "corpus_archive_fingerprint",
        "router_identities",
    ):
        if result.get(key) != owner.get(key):
            raise M5AArtifactValidationError(f"router evaluation {key} differs across evidence")
    if set(expected_examples_by_split) != {
        LanguageSplit.VALIDATION,
        LanguageSplit.DEVELOPMENT,
    }:
        raise M5AArtifactValidationError(
            "router evidence authority must contain validation and development only"
        )
    _validate_language_record_sets(
        evidence_root,
        owner=owner,
        result=result,
        expected_examples_by_split=expected_examples_by_split,
    )
    if owner.get("evaluation_order") != ["validation", "development"] or result.get(
        "evaluation_order"
    ) != ["validation", "development"]:
        raise M5AArtifactValidationError("router evaluation order differs")
    final_lock = _mapping(result.get("language_final_lock"), label="language final lock")
    if final_lock.get("sealed") is not True or final_lock.get("texts_materialized") is not False:
        raise M5AArtifactValidationError("language final lock was not kept sealed")
    safety_expectations = {
        "language_final_accessed": False,
        "control_final_accessed": False,
        "m42_final_accessed": False,
        "test_split_accessed": False,
        "historical_fresh_accessed": False,
        "controller_dispatched": False,
        "environment_reset_called": False,
        "environment_step_count": 0,
        "m2_expert_call_count": 0,
        "smolvla_go": False,
    }
    for key, expected in safety_expectations.items():
        if result.get(key) != expected:
            raise M5AArtifactValidationError(f"router evaluation safety field {key} differs")
    return ValidatedRouterEvaluationEvidence(
        root=evidence_root,
        evaluation_fingerprint=evaluation_fingerprint,
        artifact_fingerprint=artifact_fingerprint,
        artifact_count=len(artifacts),
        owner_fingerprint=f"sha256:{sha256_hex(owner)}",
        result_fingerprint=f"sha256:{sha256_hex(result)}",
        owner=owner,
        result=result,
    )


def _episode_identity(
    *,
    run_fingerprint: str,
    router_label: str,
    schedule: StagedControlSchedule,
    episode: ScheduledControlEpisode,
    example: LanguageExample,
) -> dict[str, object]:
    return {
        "run_fingerprint": run_fingerprint,
        "router_label": router_label,
        "schedule_id": schedule.schedule_id,
        "schedule_fingerprint": schedule.schedule_fingerprint,
        "episode_index": episode.episode_index,
        "scene_seed": episode.scene_seed,
        "scene_id": episode.scene_id,
        "oracle_task_id": episode.task_id,
        "language_example_id": example.example_id,
        "command_fingerprint": f"sha256:{sha256_hex(example.raw_text)}",
    }


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise M5AArtifactValidationError(f"{label} must be a sequence")
    return cast(Sequence[object], value)


def _parse_action_evidence(payload: Mapping[str, object]) -> ActionStreamEvidence:
    try:
        raw_actions = tuple(
            tuple(cast(Sequence[float], _sequence(value, label="raw action")))
            for value in _sequence(payload["raw_actions"], label="raw actions")
        )
        binary_transformed_actions = tuple(
            None
            if value is None
            else tuple(
                cast(
                    Sequence[float],
                    _sequence(value, label="binary-transformed action"),
                )
            )
            for value in _sequence(
                payload["binary_transformed_actions"],
                label="binary-transformed actions",
            )
        )
        projected_actions = tuple(
            None
            if value is None
            else tuple(cast(Sequence[float], _sequence(value, label="projected action")))
            for value in _sequence(payload["projected_actions"], label="projected actions")
        )
        executed_actions = tuple(
            tuple(cast(Sequence[float], _sequence(value, label="executed action")))
            for value in _sequence(payload["executed_actions"], label="executed actions")
        )
        evidence = ActionStreamEvidence(
            raw_actions=raw_actions,
            binary_transformed_actions=binary_transformed_actions,
            projected_actions=projected_actions,
            executed_actions=executed_actions,
            raw_violation_count=cast(int, payload["raw_violation_count"]),
            projected_action_count=cast(int, payload["projected_action_count"]),
            arm_projected_component_count=cast(int, payload["arm_projected_component_count"]),
            gripper_projected_component_count=cast(
                int, payload["gripper_projected_component_count"]
            ),
            malformed_action_count=cast(int, payload["malformed_action_count"]),
            nonfinite_action_count=cast(int, payload["nonfinite_action_count"]),
            schema_version=cast(str, payload["schema_version"]),
        )
    except (KeyError, TypeError, ValueError, M5AEvaluationError) as error:
        raise M5AArtifactValidationError(f"action evidence is malformed: {error}") from error
    if not _json_equal(evidence.to_dict(), dict(payload)):
        raise M5AArtifactValidationError("action evidence fields or fingerprints differ")
    return evidence


def _parse_control_result(payload: Mapping[str, object]) -> ControlExecutionResult:
    try:
        action = _parse_action_evidence(
            _mapping(payload.get("action_evidence"), label="control action evidence")
        )
        control = ControlExecutionResult(
            success=cast(bool, payload["success"]),
            status=cast(str, payload["status"]),
            environment_step_count=cast(int, payload["environment_step_count"]),
            final_evaluation=cast(
                Mapping[str, bool],
                _mapping(payload["final_evaluation"], label="final environment evaluation"),
            ),
            action_evidence=action,
            inference_latency_ms=tuple(
                cast(
                    Sequence[float],
                    _sequence(payload["inference_latency_ms"], label="inference latency"),
                )
            ),
            environment_latency_ms=tuple(
                cast(
                    Sequence[float],
                    _sequence(payload["environment_latency_ms"], label="environment latency"),
                )
            ),
            failure_reason=cast(str | None, payload["failure_reason"]),
            controller_inference_failed=cast(bool, payload["controller_inference_failed"]),
            invalid_action=cast(bool, payload["invalid_action"]),
            timeout=cast(bool, payload["timeout"]),
            environment_failed=cast(bool, payload["environment_failed"]),
            infrastructure_failed=cast(bool, payload["infrastructure_failed"]),
            wrong_object_interaction=cast(bool, payload["wrong_object_interaction"]),
            schema_version=cast(str, payload["schema_version"]),
        )
    except (KeyError, TypeError, ValueError, M5AEvaluationError) as error:
        raise M5AArtifactValidationError(f"control result is malformed: {error}") from error
    if not _json_equal(control.to_dict(), dict(payload)):
        raise M5AArtifactValidationError("control result fields differ")
    if (
        not control.invalid_action
        and len(control.action_evidence.executed_actions) != control.environment_step_count
    ):
        raise M5AArtifactValidationError(
            "executed action count differs from successful environment steps"
        )
    return control


def _parse_dispatch_record(payload: Mapping[str, object]) -> ControllerDispatchRecord:
    try:
        dispatch = ControllerDispatchRecord(
            decision_fingerprint=cast(str, payload["decision_fingerprint"]),
            registry_fingerprint=cast(str, payload["registry_fingerprint"]),
            oracle_task_id=cast(str | None, payload["oracle_task_id"]),
            predicted_task_id=cast(str | None, payload["predicted_task_id"]),
            controller_task_id=cast(str | None, payload["controller_task_id"]),
            controller_run_fingerprint=cast(str | None, payload["controller_run_fingerprint"]),
            controller_checkpoint_fingerprint=cast(
                str | None, payload["controller_checkpoint_fingerprint"]
            ),
            dispatched=cast(bool, payload["dispatched"]),
            controller_loaded=cast(bool, payload["controller_loaded"]),
            policy_called=cast(bool, payload["policy_called"]),
            environment_reset_called=cast(bool, payload["environment_reset_called"]),
            environment_step_count=cast(int, payload["environment_step_count"]),
            safe_rejection=cast(bool, payload["safe_rejection"]),
            controller_load_failed=cast(bool, payload["controller_load_failed"]),
            schema_version=cast(str, payload["schema_version"]),
        )
    except (KeyError, TypeError, ValueError, M5AEvaluationError) as error:
        raise M5AArtifactValidationError(f"dispatch record is malformed: {error}") from error
    if not _json_equal(dispatch.to_dict(), dict(payload)):
        raise M5AArtifactValidationError("dispatch record fields differ")
    return dispatch


def _parse_end_to_end_result(
    payload: Mapping[str, object],
    *,
    expected_identity: Mapping[str, object],
    registry: ControllerRegistry,
) -> EndToEndEpisodeResult:
    try:
        raw_task = payload.get("oracle_task_spec")
        if raw_task is not None and not isinstance(raw_task, Mapping):
            raise TypeError("oracle_task_spec must be a mapping or null")
        oracle_task = (
            None
            if raw_task is None
            else TaskSpec.from_mapping(cast(Mapping[str, object], raw_task))
        )
        decision = RouterDecision.from_dict(
            _mapping(payload.get("decision"), label="control router decision")
        )
        dispatch = _parse_dispatch_record(
            _mapping(payload.get("dispatch"), label="control dispatch record")
        )
        raw_control = payload.get("control")
        control = (
            None
            if raw_control is None
            else _parse_control_result(_mapping(raw_control, label="control result"))
        )
        raw_attribution = payload.get("failure_attribution")
        attribution = (
            None if raw_attribution is None else FailureAttribution(cast(str, raw_attribution))
        )
        result = EndToEndEpisodeResult(
            evaluation_id=cast(str, payload["evaluation_id"]),
            scene_seed=cast(int | None, payload["scene_seed"]),
            oracle_task_spec=oracle_task,
            decision=decision,
            dispatch=dispatch,
            control=control,
            failure_attribution=attribution,
            schema_version=cast(str, payload["schema_version"]),
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        RouterContractError,
        M5AEvaluationError,
    ) as error:
        raise M5AArtifactValidationError(
            f"nested end-to-end control result is malformed: {error}"
        ) from error
    if not _json_equal(result.to_dict(), dict(payload)):
        raise M5AArtifactValidationError(
            "nested end-to-end control fields or derived semantics differ"
        )
    if result.dispatch.predicted_task_id != result.decision.task_id:
        raise M5AArtifactValidationError("dispatch predicted task differs from the router decision")
    if result.decision.status is not RouterStatus.ROUTE and not result.dispatch.safe_rejection:
        raise M5AArtifactValidationError(
            "non-route decision did not terminate as a safe zero-work rejection"
        )
    expected_router_name = _CONTROL_ROUTER_NAMES.get(cast(str, expected_identity["router_label"]))
    if expected_router_name is None or result.decision.router_name != expected_router_name:
        raise M5AArtifactValidationError(
            "control atom router identity differs from its declared router path"
        )
    expected_evaluation_id = (
        f"{expected_identity['schedule_id']}:{expected_identity['router_label']}:"
        f"{cast(int, expected_identity['episode_index']):03d}"
    )
    if (
        result.evaluation_id != expected_evaluation_id
        or result.scene_seed != expected_identity["scene_seed"]
        or result.dispatch.registry_fingerprint != registry.registry_fingerprint
    ):
        raise M5AArtifactValidationError("end-to-end result identity differs from its atom")
    if (
        result.oracle_task_spec is None
        or stable_task_id(result.oracle_task_spec) != expected_identity["oracle_task_id"]
    ):
        raise M5AArtifactValidationError("end-to-end oracle task differs from its atom")
    if expected_identity["router_label"] == "oracle":
        expected_oracle = RouterDecision.route(
            task_spec=result.oracle_task_spec,
            confidence=RouterConfidence.unavailable(
                definition="oracle TaskSpec is a controller ceiling, not language confidence"
            ),
            router_name="OracleTaskSpecRouterV0",
            router_version="oracle-task-spec-router-v0",
            evidence={"oracle": True},
        )
        if not _json_equal(result.decision.to_dict(), expected_oracle.to_dict()):
            raise M5AArtifactValidationError(
                "oracle control atom differs from the scheduled oracle TaskSpec decision"
            )
    if result.dispatch.dispatched:
        try:
            entry = registry.require(cast(str, result.dispatch.controller_task_id))
        except ControllerRegistryError as error:
            raise M5AArtifactValidationError("dispatch selected an unknown controller") from error
        if (
            result.dispatch.controller_run_fingerprint != entry.run_fingerprint
            or result.dispatch.controller_checkpoint_fingerprint != entry.checkpoint_fingerprint
        ):
            raise M5AArtifactValidationError(
                "dispatch controller provenance differs from the frozen registry"
            )
    return result


def _validate_rejection_noop_probe(
    path: Path,
    *,
    registry: ControllerRegistry,
) -> dict[str, object]:
    probe = _read_object(path, label="M5A rejection no-op probe")
    base = dict(probe)
    fingerprint = base.pop("probe_fingerprint", None)
    if fingerprint != f"sha256:{sha256_hex(base)}":
        raise M5AArtifactValidationError("rejection no-op probe fingerprint differs")
    try:
        result_payload = _mapping(probe.get("result"), label="rejection no-op result")
        if result_payload.get("oracle_task_spec") is not None:
            raise M5AEvaluationError("rejection no-op probe cannot contain an oracle task")
        decision = RouterDecision.from_dict(
            _mapping(result_payload.get("decision"), label="rejection probe decision")
        )
        dispatch = _parse_dispatch_record(
            _mapping(result_payload.get("dispatch"), label="rejection probe dispatch")
        )
        result = EndToEndEpisodeResult(
            evaluation_id=cast(str, result_payload["evaluation_id"]),
            scene_seed=cast(int | None, result_payload["scene_seed"]),
            oracle_task_spec=None,
            decision=decision,
            dispatch=dispatch,
            control=None,
            failure_attribution=None,
            schema_version=cast(str, result_payload["schema_version"]),
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        RouterContractError,
        M5AEvaluationError,
    ) as error:
        raise M5AArtifactValidationError(
            f"rejection no-op probe result is malformed: {error}"
        ) from error
    if not _json_equal(result.to_dict(), result_payload):
        raise M5AArtifactValidationError("rejection no-op probe result fields differ")
    expected = {
        "schema_version": REJECTION_NOOP_PROBE_SCHEMA,
        "command_fingerprint": f"sha256:{sha256_hex('Open the drawer.')}",
        "decision_fingerprint": decision.decision_fingerprint,
        "controller_registry_fingerprint": registry.registry_fingerprint,
        "physical_m1_environment": True,
        "controller_lookup_count": 0,
        "environment_reset_count": 0,
        "environment_step_count": 0,
        "result": result.to_dict(),
        "passed": True,
    }
    if not _json_equal(base, expected):
        raise M5AArtifactValidationError(
            "rejection no-op probe differs from the zero-work target contract"
        )
    if (
        result.evaluation_id != "m5a-target-development-rejection-noop-probe"
        or result.scene_seed is not None
        or decision.status is RouterStatus.ROUTE
        or decision.router_name != "RuleRouterV0"
        or dispatch.registry_fingerprint != registry.registry_fingerprint
        or not dispatch.safe_rejection
    ):
        raise M5AArtifactValidationError("rejection no-op probe entered a forbidden runtime")
    return probe


def _validated_episode_record(
    path: Path,
    *,
    expected_identity: Mapping[str, object],
    registry: ControllerRegistry,
    active_episode_spec_required: bool,
) -> dict[str, object]:
    record = _read_object(path, label="M5A control episode evidence")
    base = dict(record)
    fingerprint = base.pop("record_fingerprint", None)
    if fingerprint != f"sha256:{sha256_hex(base)}":
        raise M5AArtifactValidationError("control episode fingerprint differs")
    if record.get("schema_version") != CONTROL_EPISODE_SCHEMA:
        raise M5AArtifactValidationError("control episode schema differs")
    if not _json_equal(record.get("identity"), dict(expected_identity)):
        raise M5AArtifactValidationError("control episode belongs to another run atom")
    latency = record.get("router_inference_latency_ms")
    if (
        isinstance(latency, bool)
        or not isinstance(latency, int | float)
        or not math.isfinite(float(latency))
        or float(latency) < 0.0
    ):
        raise M5AArtifactValidationError("control episode router latency is malformed")
    if not isinstance(record.get("result"), Mapping):
        raise M5AArtifactValidationError("control episode lacks an end-to-end result")
    result = _parse_end_to_end_result(
        cast(Mapping[str, object], record["result"]),
        expected_identity=expected_identity,
        registry=registry,
    )
    active = record.get("active_episode_spec")
    if not result.dispatch.dispatched:
        if active is not None:
            raise M5AArtifactValidationError("non-dispatched control atom has active EpisodeSpec")
    elif active is not None:
        expected_active = EpisodeSpec.create(
            scene_seed=cast(int, expected_identity["scene_seed"]),
            task_spec=cast(TaskSpec, result.oracle_task_spec),
        ).to_dict()
        if not _json_equal(active, expected_active):
            raise M5AArtifactValidationError(
                "active EpisodeSpec differs from the scheduled oracle task"
            )
    elif active_episode_spec_required:
        raise M5AArtifactValidationError("dispatched control atom lacks active EpisodeSpec")
    return record


def _summary(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    attributions: Counter[str] = Counter()
    dispatched = safe_rejections = successes = correct_routes = 0
    false_rejections = wrong_object_routes = wrong_bin_routes = 0
    wrong_task_routes = wrong_route_episodes = 0
    controller_successes = correct_route_control_successes = 0
    correct_route_control_failures = 0
    wrong_object_in_target_bin = target_in_wrong_bin = target_off_table = timeouts = 0
    wrong_object_interactions = invalid_actions = 0
    malformed_actions = nonfinite_actions = dispatch_after_rejection = 0
    raw_violations = arm_projection = gripper_projection = 0
    inference_latencies: list[float] = []
    environment_latencies: list[float] = []
    router_latencies: list[float] = []
    infrastructure_failures = 0
    for record in records:
        result = cast(Mapping[str, object], record["result"])
        router_latency = record.get("router_inference_latency_ms")
        if isinstance(router_latency, int | float) and not isinstance(router_latency, bool):
            router_latencies.append(float(router_latency))
        decision = cast(Mapping[str, object], result["decision"])
        dispatch = cast(Mapping[str, object], result["dispatch"])
        dispatched += int(dispatch.get("dispatched") is True)
        safe_rejections += int(dispatch.get("safe_rejection") is True)
        if decision.get("status") != "route":
            dispatch_after_rejection += int(
                dispatch.get("dispatched") is True
                or dispatch.get("controller_loaded") is True
                or dispatch.get("policy_called") is True
                or dispatch.get("environment_reset_called") is True
                or int(dispatch.get("environment_step_count", 0)) > 0
            )
        successes += int(result.get("end_to_end_success") is True)
        route_correct = decision.get("task_id") is not None and decision.get(
            "task_id"
        ) == dispatch.get("oracle_task_id")
        correct_routes += int(route_correct)
        oracle_task = result.get("oracle_task_spec")
        if isinstance(oracle_task, Mapping) and decision.get("status") == "route":
            wrong_object = decision.get("target_object_id") != oracle_task.get("target_object_id")
            wrong_bin = decision.get("target_bin_id") != oracle_task.get("target_bin_id")
            wrong_object_routes += int(wrong_object)
            wrong_bin_routes += int(wrong_bin)
            wrong_task_routes += int(wrong_object and wrong_bin)
            wrong_route_episodes += int(wrong_object or wrong_bin)
        false_rejections += int(
            isinstance(oracle_task, Mapping) and decision.get("status") != "route"
        )
        attribution = result.get("failure_attribution")
        if isinstance(attribution, str):
            attributions[attribution] += 1
        control = result.get("control")
        if not isinstance(control, Mapping):
            infrastructure_failures += int(dispatch.get("controller_load_failed") is True)
            continue
        control_success = control.get("success") is True
        controller_successes += int(control_success)
        correct_route_control_successes += int(route_correct and control_success)
        correct_route_control_failures += int(route_correct and not control_success)
        infrastructure_failures += int(control.get("infrastructure_failed") is True)
        infrastructure_failures += int(control.get("controller_inference_failed") is True)
        timeouts += int(control.get("timeout") is True)
        invalid_actions += int(control.get("invalid_action") is True)
        wrong_object_interactions += int(control.get("wrong_object_interaction") is True)
        final_evaluation = control.get("final_evaluation")
        if isinstance(final_evaluation, Mapping):
            wrong_object_in_target_bin += int(
                final_evaluation.get("wrong_object_in_target_bin") is True
            )
            target_in_wrong_bin += int(final_evaluation.get("target_in_wrong_bin") is True)
            target_off_table += int(final_evaluation.get("target_off_table") is True)
        action = control.get("action_evidence")
        if isinstance(action, Mapping):
            raw_violations += int(action.get("raw_violation_count", 0))
            arm_projection += int(action.get("arm_projected_component_count", 0))
            gripper_projection += int(action.get("gripper_projected_component_count", 0))
            malformed_actions += int(action.get("malformed_action_count", 0))
            nonfinite_actions += int(action.get("nonfinite_action_count", 0))
        latency = control.get("inference_latency_ms")
        if isinstance(latency, list):
            inference_latencies.extend(float(value) for value in latency)
        environment_latency = control.get("environment_latency_ms")
        if isinstance(environment_latency, list):
            environment_latencies.extend(float(value) for value in environment_latency)
    sorted_latency = sorted(inference_latencies)
    sorted_environment_latency = sorted(environment_latencies)
    sorted_router_latency = sorted(router_latencies)

    def percentile(values: Sequence[float], fraction: float) -> float | None:
        if not values:
            return None
        index = min(len(values) - 1, math.ceil(fraction * len(values)) - 1)
        return values[index]

    return {
        "episode_count": len(records),
        "dispatched_count": dispatched,
        "safe_rejection_count": safe_rejections,
        "routing_correct_count": correct_routes,
        "routing_false_rejection_count": false_rejections,
        "routing_wrong_object_count": wrong_object_routes,
        "routing_wrong_bin_count": wrong_bin_routes,
        "routing_wrong_task_count": wrong_task_routes,
        "routing_wrong_episode_count": wrong_route_episodes,
        "controller_task_success_count": controller_successes,
        "routing_correct_control_success_count": correct_route_control_successes,
        "routing_correct_control_failure_count": correct_route_control_failures,
        "end_to_end_success_count": successes,
        "wrong_object_in_target_bin_count": wrong_object_in_target_bin,
        "target_in_wrong_bin_count": target_in_wrong_bin,
        "target_off_table_count": target_off_table,
        "timeout_count": timeouts,
        "wrong_object_interaction_available": True,
        "wrong_object_interaction_count": wrong_object_interactions,
        "invalid_action_count": invalid_actions,
        "malformed_action_count": malformed_actions,
        "nonfinite_action_count": nonfinite_actions,
        "dispatch_after_rejection_count": dispatch_after_rejection,
        "m2_expert_call_count": 0,
        "failure_attribution_counts": {
            value.value: attributions[value.value] for value in FailureAttribution
        },
        "raw_action_violation_count": raw_violations,
        "arm_projected_component_count": arm_projection,
        "gripper_projected_component_count": gripper_projection,
        "inference_latency_ms": {
            "sample_count": len(sorted_latency),
            "p50": percentile(sorted_latency, 0.50),
            "p95": percentile(sorted_latency, 0.95),
            "p99": percentile(sorted_latency, 0.99),
            "total": sum(sorted_latency),
        },
        "environment_latency_ms": {
            "sample_count": len(sorted_environment_latency),
            "p50": percentile(sorted_environment_latency, 0.50),
            "p95": percentile(sorted_environment_latency, 0.95),
            "p99": percentile(sorted_environment_latency, 0.99),
            "total": sum(sorted_environment_latency),
        },
        "router_inference_latency_ms": {
            "sample_count": len(sorted_router_latency),
            "p50": percentile(sorted_router_latency, 0.50),
            "p95": percentile(sorted_router_latency, 0.95),
            "p99": percentile(sorted_router_latency, 0.99),
            "total": sum(sorted_router_latency),
        },
        "observed_end_to_end_runtime_ms": sum(sorted_router_latency)
        + sum(sorted_latency)
        + sum(sorted_environment_latency),
        "infrastructure_failure_count": infrastructure_failures,
    }


def validate_development_control_evidence(
    root: str | Path,
    *,
    inputs: DevelopmentControlInputsLike,
    registry: ControllerRegistry,
) -> ValidatedControlEvidence:
    """Rehash every atom in one promoted 1/3/6-scene control stage."""

    run_root = _require_real_directory(root, label="development control evidence")
    if len(inputs.episodes) not in {6, 18, 36}:
        raise M5AArtifactValidationError("staged control inputs must contain 6, 18, or 36 episodes")
    owner = _read_object(run_root / "owner.json", label="control evidence owner")
    registry_payload = _read_object(
        run_root / "controller_registry.json", label="control controller registry"
    )
    complete = _read_object(run_root / "complete.json", label="control completion")
    identity = owner.get("identity")
    run_fingerprint = owner.get("run_fingerprint")
    if not isinstance(identity, Mapping) or not isinstance(run_fingerprint, str):
        raise M5AArtifactValidationError("control owner identity is malformed")
    if owner != {
        "schema_version": CONTROL_RUN_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "identity": dict(identity),
    }:
        raise M5AArtifactValidationError("control owner fields differ")
    if run_fingerprint != f"sha256:{sha256_hex(dict(identity))}":
        raise M5AArtifactValidationError("control owner fingerprint differs")
    if run_root.name != run_fingerprint.removeprefix("sha256:"):
        raise M5AArtifactValidationError("control evidence is not at its run-owned path")
    router_order = identity.get("router_order")
    if (
        not isinstance(router_order, list)
        or not router_order
        or router_order[0] != "oracle"
        or any(value not in CONTROL_ROUTER_ORDER for value in router_order)
        or len(set(router_order)) != len(router_order)
    ):
        raise M5AArtifactValidationError("control owner router order is not Oracle plus learned")
    expected_identity_fields = {
        "schema_version": CONTROL_RUN_SCHEMA,
        "schedule_fingerprint": inputs.schedule.schedule_fingerprint,
        "corpus_fingerprint": inputs.corpus_fingerprint,
        "controller_registry_fingerprint": registry.registry_fingerprint,
        "stage": inputs.schedule.stage.value,
        "router_order": router_order,
        "episode_count_per_router": len(inputs.episodes),
    }
    for key, value in expected_identity_fields.items():
        if identity.get(key) != value:
            raise M5AArtifactValidationError(f"control owner {key} differs")
    active_episode_spec_required = identity.get("active_episode_spec_required")
    if not isinstance(active_episode_spec_required, bool):
        raise M5AArtifactValidationError(
            "control owner active_episode_spec_required must be boolean"
        )
    rejection_probe_fingerprint = identity.get("rejection_noop_probe_fingerprint")
    if rejection_probe_fingerprint is not None and not isinstance(rejection_probe_fingerprint, str):
        raise M5AArtifactValidationError(
            "control owner rejection no-op probe fingerprint is malformed"
        )
    try:
        restored_registry = ControllerRegistry.from_dict(registry_payload)
    except (ControllerRegistryError, TypeError, ValueError) as error:
        raise M5AArtifactValidationError("control controller registry is malformed") from error
    if restored_registry.to_dict() != registry.to_dict():
        raise M5AArtifactValidationError("control controller registry differs")
    rejection_probe = (
        None
        if rejection_probe_fingerprint is None
        else _validate_rejection_noop_probe(
            run_root / "rejection_noop_probe.json",
            registry=registry,
        )
    )
    if (
        rejection_probe is not None
        and rejection_probe.get("probe_fingerprint") != rejection_probe_fingerprint
    ):
        raise M5AArtifactValidationError("control owner refers to another rejection probe")
    completion_base = dict(complete)
    completion_fingerprint = completion_base.pop("completion_fingerprint", None)
    if completion_fingerprint != f"sha256:{sha256_hex(completion_base)}":
        raise M5AArtifactValidationError("control completion fingerprint differs")
    expected_atoms = len(router_order) * len(inputs.episodes)
    if (
        complete.get("schema_version") != CONTROL_COMPLETION_SCHEMA
        or complete.get("run_fingerprint") != run_fingerprint
        or complete.get("passed") is not True
        or complete.get("completed_episode_atoms") != expected_atoms
        or complete.get("expected_episode_atoms") != expected_atoms
    ):
        raise M5AArtifactValidationError("control completion identity or counts differ")
    for key, value in {
        "schedule_id": inputs.schedule.schedule_id,
        "schedule_fingerprint": inputs.schedule.schedule_fingerprint,
        "corpus_fingerprint": inputs.corpus_fingerprint,
        "controller_registry_fingerprint": registry.registry_fingerprint,
        "stage": inputs.schedule.stage.value,
        "router_order": router_order,
    }.items():
        if complete.get(key) != value:
            raise M5AArtifactValidationError(f"control completion {key} differs")
    summaries: dict[str, object] = {}
    record_fingerprints: list[str] = []
    expected_files = {"owner.json", "controller_registry.json", "complete.json"}
    if rejection_probe is not None:
        expected_files.add("rejection_noop_probe.json")
    for router_label in router_order:
        records: list[dict[str, object]] = []
        for episode in inputs.episodes:
            example = inputs.examples_by_id.get(episode.language_example_id)
            if example is None:
                raise M5AArtifactValidationError("control input lacks a scheduled example")
            relative = f"episodes/{router_label}/{episode.episode_index:03d}.json"
            expected_files.add(relative)
            record = _validated_episode_record(
                run_root / relative,
                expected_identity=_episode_identity(
                    run_fingerprint=run_fingerprint,
                    router_label=router_label,
                    schedule=inputs.schedule,
                    episode=episode,
                    example=example,
                ),
                registry=registry,
                active_episode_spec_required=active_episode_spec_required,
            )
            records.append(record)
            record_fingerprints.append(cast(str, record["record_fingerprint"]))
        try:
            summaries[router_label] = _summary(records)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise M5AArtifactValidationError(
                f"control {router_label} records are structurally malformed"
            ) from error
    observed_files = {
        PurePosixPath(path.relative_to(run_root)).as_posix()
        for path in run_root.rglob("*")
        if path.is_file() and not path.is_symlink() and not path.is_junction()
    }
    _artifact_records(run_root, excluded=frozenset())
    if observed_files != expected_files:
        raise M5AArtifactValidationError("control evidence file set differs")
    if not _json_equal(complete.get("summaries"), summaries):
        raise M5AArtifactValidationError("control completion summaries differ from episodes")
    record_set_fingerprint = f"sha256:{sha256_hex(record_fingerprints)}"
    if complete.get("record_set_fingerprint") != record_set_fingerprint:
        raise M5AArtifactValidationError("control record-set fingerprint differs")
    aggregate_expectations = {
        "wrong_object_interaction_count": sum(
            int(cast(Mapping[str, object], value)["wrong_object_interaction_count"])
            for value in summaries.values()
        ),
        "invalid_action_count": sum(
            int(cast(Mapping[str, object], value)["invalid_action_count"])
            for value in summaries.values()
        ),
        "malformed_action_count": sum(
            int(cast(Mapping[str, object], value)["malformed_action_count"])
            for value in summaries.values()
        ),
        "nonfinite_action_count": sum(
            int(cast(Mapping[str, object], value)["nonfinite_action_count"])
            for value in summaries.values()
        ),
        "dispatch_after_rejection_count": sum(
            int(cast(Mapping[str, object], value)["dispatch_after_rejection_count"])
            for value in summaries.values()
        ),
    }
    for key, value in aggregate_expectations.items():
        if complete.get(key) != value:
            raise M5AArtifactValidationError(f"control completion {key} differs")
    if any(
        int(cast(Mapping[str, object], value)["infrastructure_failure_count"]) != 0
        for value in summaries.values()
    ):
        raise M5AArtifactValidationError("control evidence contains infrastructure failures")
    observed_safe_rejections = sum(
        int(cast(Mapping[str, object], value)["safe_rejection_count"])
        for value in summaries.values()
    )
    zero_dispatch_validated = aggregate_expectations["dispatch_after_rejection_count"] == 0 and (
        rejection_probe is not None or observed_safe_rejections > 0
    )
    required_safety = {
        "wrong_object_interaction_available": True,
        "rejection_noop_probe_fingerprint": rejection_probe_fingerprint,
        "rejection_noop_probe_validated": rejection_probe is not None,
        "zero_dispatch_after_rejection_validated": zero_dispatch_validated,
        "m2_expert_call_count": 0,
        "m2_expert_free_validated": True,
        "language_final_accessed": False,
        "control_final_accessed": False,
        "m42_final_accessed": False,
        "test_split_accessed": False,
        "historical_fresh_accessed": False,
        "smolvla_go": False,
    }
    for key, value in required_safety.items():
        if complete.get(key) != value:
            raise M5AArtifactValidationError(f"control completion safety field {key} differs")
    return ValidatedControlEvidence(
        root=run_root,
        run_fingerprint=run_fingerprint,
        schedule_fingerprint=inputs.schedule.schedule_fingerprint,
        corpus_fingerprint=inputs.corpus_fingerprint,
        controller_registry_fingerprint=registry.registry_fingerprint,
        record_count=expected_atoms,
        record_set_fingerprint=record_set_fingerprint,
        completion_fingerprint=cast(str, completion_fingerprint),
        owner_fingerprint=f"sha256:{sha256_hex(owner)}",
        identity=dict(identity),
        summaries=summaries,
    )


__all__ = [
    "CONTROL_COMPLETION_SCHEMA",
    "CONTROL_EPISODE_SCHEMA",
    "CONTROL_ROUTER_ORDER",
    "CONTROL_RUN_SCHEMA",
    "CORPUS_ARCHIVE_SCHEMA",
    "DevelopmentControlInputsLike",
    "M5AArtifactValidationError",
    "ROUTER_EVALUATION_EVIDENCE_SCHEMA",
    "ValidatedControlEvidence",
    "ValidatedCorpusArchive",
    "ValidatedRouterEvaluationEvidence",
    "validate_development_control_evidence",
    "validate_language_corpus_archive",
    "validate_router_evaluation_evidence",
]
