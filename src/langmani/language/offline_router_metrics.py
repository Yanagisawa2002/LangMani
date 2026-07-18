"""Raw-record metrics for the M5A.2 validation/development comparison."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import cast

from langmani.language.router_evaluation import RouterEvaluationRecord
from langmani.language.router_types import (
    LanguageExample,
    LanguageSplit,
    RouterDecision,
    RouterRejectionReason,
    RouterStatus,
)


class OfflineRouterMetricError(ValueError):
    """Raised when raw language records cannot support exact metric recomputation."""


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def router_record_from_dict(value: Mapping[str, object]) -> RouterEvaluationRecord:
    expected = {
        "example_id",
        "template_family_id",
        "split",
        "expected_status",
        "expected_task_id",
        "expected_rejection_reason",
        "decision",
        "latency_ms",
        "repeat_decision_fingerprints",
        "status_correct",
        "full_task_correct",
        "exact_decision_correct",
        "deterministic",
    }
    if set(value) != expected or not isinstance(value.get("decision"), Mapping):
        raise OfflineRouterMetricError("raw router record fields differ")
    repeats = value.get("repeat_decision_fingerprints")
    if not isinstance(repeats, list) or not all(isinstance(item, str) for item in repeats):
        raise OfflineRouterMetricError("raw router repeat identities are malformed")
    reason = value.get("expected_rejection_reason")
    record = RouterEvaluationRecord(
        example_id=cast(str, value["example_id"]),
        template_family_id=cast(str, value["template_family_id"]),
        split=LanguageSplit(cast(str, value["split"])),
        expected_status=RouterStatus(cast(str, value["expected_status"])),
        expected_task_id=cast(str | None, value["expected_task_id"]),
        expected_rejection_reason=(
            None if reason is None else RouterRejectionReason(cast(str, reason))
        ),
        decision=RouterDecision.from_dict(cast(Mapping[str, object], value["decision"])),
        latency_ms=float(cast(int | float, value["latency_ms"])),
        repeat_decision_fingerprints=tuple(cast(list[str], repeats)),
    )
    if any(
        value[key] is not expected_value
        for key, expected_value in (
            ("status_correct", record.status_correct),
            ("full_task_correct", record.full_task_correct),
            ("exact_decision_correct", record.exact_decision_correct),
            ("deterministic", record.deterministic),
        )
    ):
        raise OfflineRouterMetricError("raw router derived fields differ")
    return record


def _generation_counts(record: RouterEvaluationRecord) -> tuple[int, int, int, int, int, int]:
    evidence = record.decision.evidence
    repairs = evidence.get("format_repair_attempts", 0)
    if isinstance(repairs, bool) or not isinstance(repairs, int) or repairs not in {0, 1}:
        repairs = 0
    exhausted = record.decision.rejection_reason is RouterRejectionReason.FORMAT_REPAIR_EXHAUSTED
    metadata = evidence.get("generation_metadata", ())
    token_count = 0
    peak = 0
    if isinstance(metadata, Sequence) and not isinstance(metadata, str | bytes):
        for item in metadata:
            if not isinstance(item, Mapping):
                continue
            tokens = item.get("generated_token_count")
            memory = item.get("peak_gpu_memory_bytes")
            if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0:
                token_count += tokens
            if isinstance(memory, int) and not isinstance(memory, bool) and memory >= 0:
                peak = max(peak, memory)
    return (
        int(repairs == 0),
        int(repairs == 1),
        int(repairs == 1 and not exhausted),
        int(exhausted),
        token_count,
        peak,
    )


def recompute_router_metrics(
    *,
    records: Sequence[RouterEvaluationRecord],
    examples: Sequence[LanguageExample],
) -> dict[str, object]:
    frozen = tuple(records)
    expected = tuple(examples)
    if not frozen or len(frozen) != len(expected):
        raise OfflineRouterMetricError("raw record count differs from the split")
    by_id = {example.example_id: example for example in expected}
    if len(by_id) != len(expected) or set(by_id) != {record.example_id for record in frozen}:
        raise OfflineRouterMetricError("raw record identities differ from the split")
    if any(record.split is not by_id[record.example_id].split for record in frozen):
        raise OfflineRouterMetricError("raw record split differs")
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
    object_correct = 0
    bin_correct = 0
    object_confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    bin_confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_task: dict[str, list[bool]] = defaultdict(list)
    per_family: dict[str, list[bool]] = defaultdict(list)
    rejection_family_routes: dict[str, list[bool]] = defaultdict(list)
    status_confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    reason_confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    initial_valid = repairs = repair_success = exhausted = tokens = 0
    peak_memory = 0
    for record in frozen:
        example = by_id[record.example_id]
        status_confusion[record.expected_status.value][record.decision.status.value] += 1
        per_family[record.template_family_id].append(record.exact_decision_correct)
        counts = _generation_counts(record)
        initial_valid += counts[0]
        repairs += counts[1]
        repair_success += counts[2]
        exhausted += counts[3]
        tokens += counts[4]
        peak_memory = max(peak_memory, counts[5])
        if record.expected_status is RouterStatus.ROUTE:
            task = example.expected_task_spec
            if task is None or record.expected_task_id is None:
                raise OfflineRouterMetricError("routeable raw record lost its TaskSpec")
            predicted_object = record.decision.target_object_id or "none"
            predicted_bin = record.decision.target_bin_id or "none"
            object_correct += predicted_object == task.target_object_id
            bin_correct += predicted_bin == task.target_bin_id
            object_confusion[task.target_object_id][predicted_object] += 1
            bin_confusion[task.target_bin_id][predicted_bin] += 1
            per_task[record.expected_task_id].append(record.full_task_correct)
        else:
            predicted_reason = (
                "none"
                if record.decision.rejection_reason is None
                else record.decision.rejection_reason.value
            )
            expected_reason = (
                "none"
                if record.expected_rejection_reason is None
                else record.expected_rejection_reason.value
            )
            reason_confusion[expected_reason][predicted_reason] += 1
            rejection_family_routes[record.template_family_id].append(
                record.decision.status is RouterStatus.ROUTE
            )
    rejection_precision = _ratio(true_rejections, len(predicted_rejections))
    rejection_recall = _ratio(true_rejections, len(rejected))
    rejection_f1 = (
        0.0
        if rejection_precision + rejection_recall == 0
        else 2 * rejection_precision * rejection_recall / (rejection_precision + rejection_recall)
    )

    def status_recall(status: RouterStatus) -> float:
        selected = tuple(record for record in frozen if record.expected_status is status)
        return _ratio(sum(record.decision.status is status for record in selected), len(selected))

    reason_recalls = []
    for reason in sorted(
        {
            record.expected_rejection_reason
            for record in rejected
            if record.expected_rejection_reason is not None
        },
        key=lambda value: value.value,
    ):
        selected = tuple(
            record for record in rejected if record.expected_rejection_reason is reason
        )
        reason_recalls.append(
            _ratio(
                sum(record.decision.rejection_reason is reason for record in selected),
                len(selected),
            )
        )

    result: dict[str, object] = {
        "example_count": len(frozen),
        "routeable_count": len(routeable),
        "rejected_count": len(rejected),
        "status_accuracy": _ratio(sum(record.status_correct for record in frozen), len(frozen)),
        "valid_full_task_accuracy": _ratio(
            sum(record.full_task_correct for record in routeable), len(routeable)
        ),
        "object_accuracy": _ratio(object_correct, len(routeable)),
        "bin_accuracy": _ratio(bin_correct, len(routeable)),
        "route_recall": _ratio(
            sum(record.decision.status is RouterStatus.ROUTE for record in routeable),
            len(routeable),
        ),
        "false_rejection_rate": _ratio(
            sum(record.decision.status is not RouterStatus.ROUTE for record in routeable),
            len(routeable),
        ),
        "rejection_precision": rejection_precision,
        "rejection_recall": rejection_recall,
        "rejection_f1": rejection_f1,
        "rejection_accuracy": _ratio(
            sum(record.status_correct for record in rejected), len(rejected)
        ),
        "false_route_rate": _ratio(
            sum(record.decision.status is RouterStatus.ROUTE for record in rejected),
            len(rejected),
        ),
        "exact_rejection_reason_accuracy": _ratio(
            sum(
                record.decision.rejection_reason is record.expected_rejection_reason
                for record in rejected
            ),
            len(rejected),
        ),
        "rejection_reason_macro_recall": _ratio(sum(reason_recalls), len(reason_recalls)),
        "ambiguous_rejection_recall": status_recall(RouterStatus.REJECT_AMBIGUOUS),
        "unsupported_rejection_recall": status_recall(RouterStatus.REJECT_UNSUPPORTED),
        "malformed_rejection_recall": status_recall(RouterStatus.REJECT_MALFORMED),
        "initial_output_schema_valid_rate": _ratio(initial_valid, len(frozen)),
        "final_schema_valid_output_rate": _ratio(len(frozen) - exhausted, len(frozen)),
        "schema_valid_output_rate": _ratio(len(frozen) - exhausted, len(frozen)),
        "repair_attempt_rate": _ratio(repairs, len(frozen)),
        "repair_success_rate": _ratio(repair_success, repairs),
        "malformed_output_after_repair_rate": _ratio(exhausted, len(frozen)),
        "coverage": _ratio(
            sum(record.decision.status is RouterStatus.ROUTE for record in frozen), len(frozen)
        ),
        "deterministic_repeatability": _ratio(
            sum(record.deterministic for record in frozen), len(frozen)
        ),
        "latency_p50_ms": _percentile([record.latency_ms for record in frozen], 0.50),
        "latency_p95_ms": _percentile([record.latency_ms for record in frozen], 0.95),
        "latency_p99_ms": _percentile([record.latency_ms for record in frozen], 0.99),
        "generated_token_count": tokens,
        "peak_gpu_memory_bytes": peak_memory,
        "per_task_accuracy": {
            key: _ratio(sum(values), len(values)) for key, values in sorted(per_task.items())
        },
        "per_template_family_accuracy": {
            key: _ratio(sum(values), len(values)) for key, values in sorted(per_family.items())
        },
        "rejection_family_false_route_rates": {
            key: _ratio(sum(values), len(values))
            for key, values in sorted(rejection_family_routes.items())
        },
        "status_confusion_matrix": {
            key: dict(sorted(value.items())) for key, value in sorted(status_confusion.items())
        },
        "rejection_reason_confusion_matrix": {
            key: dict(sorted(value.items())) for key, value in sorted(reason_confusion.items())
        },
        "object_confusion_matrix": {
            key: dict(sorted(value.items())) for key, value in sorted(object_confusion.items())
        },
        "bin_confusion_matrix": {
            key: dict(sorted(value.items())) for key, value in sorted(bin_confusion.items())
        },
    }
    return result


__all__ = [
    "OfflineRouterMetricError",
    "recompute_router_metrics",
    "router_record_from_dict",
]
