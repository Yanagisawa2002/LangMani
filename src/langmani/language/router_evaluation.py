"""Language-only M5A router evaluation with split-safe auditable metrics."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from langmani.language.router_types import (
    LanguageExample,
    LanguageSplit,
    RouterDecision,
    RouterRejectionReason,
    RouterStatus,
)


class RouterEvaluationError(RuntimeError):
    """Raised when language-only evaluation evidence violates its contract."""


class LanguageRouter(Protocol):
    def route(self, command: str) -> RouterDecision: ...


@dataclass(frozen=True, slots=True)
class RouterEvaluationRecord:
    example_id: str
    template_family_id: str
    split: LanguageSplit
    expected_status: RouterStatus
    expected_task_id: str | None
    expected_rejection_reason: RouterRejectionReason | None
    decision: RouterDecision
    latency_ms: float
    repeat_decision_fingerprints: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.split not in {LanguageSplit.VALIDATION, LanguageSplit.DEVELOPMENT}:
            raise RouterEvaluationError("router records may use validation/development only")
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0.0:
            raise RouterEvaluationError("router latency must be finite and non-negative")
        if not self.repeat_decision_fingerprints:
            raise RouterEvaluationError("router repeatability requires at least one decision")

    @property
    def status_correct(self) -> bool:
        return self.decision.status is self.expected_status

    @property
    def full_task_correct(self) -> bool:
        return (
            self.expected_status is RouterStatus.ROUTE
            and self.decision.status is RouterStatus.ROUTE
            and self.expected_task_id is not None
            and self.decision.task_id == self.expected_task_id
        )

    @property
    def exact_decision_correct(self) -> bool:
        if self.expected_status is RouterStatus.ROUTE:
            return self.full_task_correct
        return (
            self.decision.status is self.expected_status
            and self.decision.rejection_reason is self.expected_rejection_reason
        )

    @property
    def deterministic(self) -> bool:
        return len(set(self.repeat_decision_fingerprints)) == 1

    def to_dict(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "template_family_id": self.template_family_id,
            "split": self.split.value,
            "expected_status": self.expected_status.value,
            "expected_task_id": self.expected_task_id,
            "expected_rejection_reason": (
                None
                if self.expected_rejection_reason is None
                else self.expected_rejection_reason.value
            ),
            "decision": self.decision.to_dict(),
            "latency_ms": self.latency_ms,
            "repeat_decision_fingerprints": list(self.repeat_decision_fingerprints),
            "status_correct": self.status_correct,
            "full_task_correct": self.full_task_correct,
            "exact_decision_correct": self.exact_decision_correct,
            "deterministic": self.deterministic,
        }


@dataclass(frozen=True, slots=True)
class RouterEvaluationSummary:
    split: LanguageSplit
    example_count: int
    routeable_count: int
    rejected_count: int
    status_accuracy: float
    valid_full_task_accuracy: float
    object_accuracy: float
    bin_accuracy: float
    route_recall: float
    false_rejection_rate: float
    rejection_precision: float
    rejection_recall: float
    rejection_f1: float
    false_route_rate: float
    rejection_reason_accuracy: float
    ambiguous_rejection_recall: float
    unsupported_rejection_recall: float
    malformed_rejection_recall: float
    schema_valid_output_rate: float
    malformed_output_rate: float
    coverage: float
    selective_accuracy: float
    expected_calibration_error: float | None
    confidence_examples: int
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    deterministic_repeatability: float
    status_confusion: Mapping[str, Mapping[str, int]]
    rejection_reason_confusion: Mapping[str, Mapping[str, int]]
    per_task_accuracy: Mapping[str, float]
    per_family_accuracy: Mapping[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split.value,
            "example_count": self.example_count,
            "routeable_count": self.routeable_count,
            "rejected_count": self.rejected_count,
            "status_accuracy": self.status_accuracy,
            "valid_full_task_accuracy": self.valid_full_task_accuracy,
            "object_accuracy": self.object_accuracy,
            "bin_accuracy": self.bin_accuracy,
            "route_recall": self.route_recall,
            "false_rejection_rate": self.false_rejection_rate,
            "rejection_precision": self.rejection_precision,
            "rejection_recall": self.rejection_recall,
            "rejection_f1": self.rejection_f1,
            "false_route_rate": self.false_route_rate,
            "rejection_reason_accuracy": self.rejection_reason_accuracy,
            "ambiguous_rejection_recall": self.ambiguous_rejection_recall,
            "unsupported_rejection_recall": self.unsupported_rejection_recall,
            "malformed_rejection_recall": self.malformed_rejection_recall,
            "schema_valid_output_rate": self.schema_valid_output_rate,
            "malformed_output_rate": self.malformed_output_rate,
            "coverage": self.coverage,
            "selective_accuracy": self.selective_accuracy,
            "expected_calibration_error": self.expected_calibration_error,
            "confidence_examples": self.confidence_examples,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "deterministic_repeatability": self.deterministic_repeatability,
            "status_confusion": {
                expected: dict(predicted) for expected, predicted in self.status_confusion.items()
            },
            "rejection_reason_confusion": {
                expected: dict(predicted)
                for expected, predicted in self.rejection_reason_confusion.items()
            },
            "per_task_accuracy": dict(self.per_task_accuracy),
            "per_family_accuracy": dict(self.per_family_accuracy),
        }


def _safe_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise RouterEvaluationError("latency percentile requires observations")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _expected_calibration_error(
    records: Sequence[RouterEvaluationRecord], *, bins: int
) -> tuple[float | None, int]:
    confidence_records = tuple(record for record in records if record.decision.confidence.available)
    if not confidence_records:
        return None, 0
    if bins <= 0:
        raise RouterEvaluationError("ECE bins must be positive")
    total = len(confidence_records)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = [
            record
            for record in confidence_records
            if record.decision.confidence.score is not None
            and lower <= record.decision.confidence.score
            and (
                record.decision.confidence.score < upper
                or (index == bins - 1 and record.decision.confidence.score <= upper)
            )
        ]
        if not selected:
            continue
        confidence = sum(float(record.decision.confidence.score) for record in selected) / len(
            selected
        )
        accuracy = sum(record.full_task_correct for record in selected) / len(selected)
        error += len(selected) / total * abs(accuracy - confidence)
    return error, total


def _status_recall(
    records: Sequence[RouterEvaluationRecord],
    expected_status: RouterStatus,
) -> float:
    selected = tuple(record for record in records if record.expected_status is expected_status)
    return _safe_ratio(
        sum(record.decision.status is expected_status for record in selected),
        len(selected),
    )


def evaluate_language_router(
    *,
    router: LanguageRouter,
    examples: Sequence[LanguageExample],
    split: LanguageSplit,
    repeat_count: int = 2,
    ece_bins: int = 10,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[tuple[RouterEvaluationRecord, ...], RouterEvaluationSummary]:
    """Evaluate one router without opening train/final or any controller schedule."""

    if split not in {LanguageSplit.VALIDATION, LanguageSplit.DEVELOPMENT}:
        raise RouterEvaluationError("language router evaluation allows validation/development only")
    values = tuple(examples)
    if not values or any(example.split is not split for example in values):
        raise RouterEvaluationError("all language examples must match the declared split")
    if isinstance(repeat_count, bool) or not isinstance(repeat_count, int) or repeat_count < 1:
        raise RouterEvaluationError("repeat_count must be a positive integer")
    expected_tasks = {
        example.example_id: example.expected_task_spec
        for example in values
        if example.expected_task_spec is not None
    }
    records: list[RouterEvaluationRecord] = []
    for example in values:
        decisions: list[RouterDecision] = []
        start = clock()
        decision = router.route(example.raw_text)
        elapsed = (clock() - start) * 1000.0
        if not isinstance(decision, RouterDecision):
            raise RouterEvaluationError("router must return RouterDecision")
        decisions.append(decision)
        for _ in range(repeat_count - 1):
            repeated = router.route(example.raw_text)
            if not isinstance(repeated, RouterDecision):
                raise RouterEvaluationError("router repeat must return RouterDecision")
            decisions.append(repeated)
        records.append(
            RouterEvaluationRecord(
                example_id=example.example_id,
                template_family_id=example.template_family_id,
                split=split,
                expected_status=example.expected_status,
                expected_task_id=example.task_id,
                expected_rejection_reason=example.expected_rejection_reason,
                decision=decision,
                latency_ms=elapsed,
                repeat_decision_fingerprints=tuple(
                    value.decision_fingerprint for value in decisions
                ),
            )
        )
    frozen = tuple(records)
    routeable = tuple(record for record in frozen if record.expected_status is RouterStatus.ROUTE)
    rejected = tuple(
        record for record in frozen if record.expected_status is not RouterStatus.ROUTE
    )
    predicted_rejections = tuple(
        record for record in frozen if record.decision.status is not RouterStatus.ROUTE
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
    routed = tuple(record for record in frozen if record.decision.status is RouterStatus.ROUTE)
    malformed_reasons = {
        RouterRejectionReason.STRUCTURED_OUTPUT_INVALID,
        RouterRejectionReason.FORMAT_REPAIR_EXHAUSTED,
    }
    status_confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    rejection_confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in frozen:
        status_confusion[record.expected_status.value][record.decision.status.value] += 1
        if record.expected_status is not RouterStatus.ROUTE:
            expected_reason = (
                "none"
                if record.expected_rejection_reason is None
                else record.expected_rejection_reason.value
            )
            predicted_reason = (
                "none"
                if record.decision.rejection_reason is None
                else record.decision.rejection_reason.value
            )
            rejection_confusion[expected_reason][predicted_reason] += 1
    task_groups: dict[str, list[RouterEvaluationRecord]] = defaultdict(list)
    family_groups: dict[str, list[RouterEvaluationRecord]] = defaultdict(list)
    for record in frozen:
        if record.expected_task_id is not None:
            task_groups[record.expected_task_id].append(record)
        family_groups[record.template_family_id].append(record)
    ece, confidence_examples = _expected_calibration_error(frozen, bins=ece_bins)
    summary = RouterEvaluationSummary(
        split=split,
        example_count=len(frozen),
        routeable_count=len(routeable),
        rejected_count=len(rejected),
        status_accuracy=_safe_ratio(sum(record.status_correct for record in frozen), len(frozen)),
        valid_full_task_accuracy=_safe_ratio(
            sum(record.full_task_correct for record in routeable), len(routeable)
        ),
        object_accuracy=_safe_ratio(
            sum(
                record.decision.status is RouterStatus.ROUTE
                and record.decision.target_object_id
                == expected_tasks[record.example_id].target_object_id
                for record in routeable
            ),
            len(routeable),
        ),
        bin_accuracy=_safe_ratio(
            sum(
                record.decision.status is RouterStatus.ROUTE
                and record.decision.target_bin_id == expected_tasks[record.example_id].target_bin_id
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
        ambiguous_rejection_recall=_status_recall(frozen, RouterStatus.REJECT_AMBIGUOUS),
        unsupported_rejection_recall=_status_recall(frozen, RouterStatus.REJECT_UNSUPPORTED),
        malformed_rejection_recall=_status_recall(frozen, RouterStatus.REJECT_MALFORMED),
        schema_valid_output_rate=1.0,
        malformed_output_rate=_safe_ratio(
            sum(record.decision.rejection_reason in malformed_reasons for record in frozen),
            len(frozen),
        ),
        coverage=_safe_ratio(len(routed), len(frozen)),
        selective_accuracy=_safe_ratio(
            sum(record.full_task_correct for record in routed), len(routed)
        ),
        expected_calibration_error=ece,
        confidence_examples=confidence_examples,
        latency_p50_ms=_percentile([record.latency_ms for record in frozen], 0.50),
        latency_p95_ms=_percentile([record.latency_ms for record in frozen], 0.95),
        latency_p99_ms=_percentile([record.latency_ms for record in frozen], 0.99),
        deterministic_repeatability=_safe_ratio(
            sum(record.deterministic for record in frozen), len(frozen)
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
    return frozen, summary


__all__ = [
    "LanguageRouter",
    "RouterEvaluationError",
    "RouterEvaluationRecord",
    "RouterEvaluationSummary",
    "evaluate_language_router",
]
