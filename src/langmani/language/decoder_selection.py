"""Validation-only fixed-grid selection for M5A.1 classifier decoders."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from langmani.datasets.identity import sha256_hex
from langmani.language.classifier_decoders import (
    DecoderCandidate,
    DecoderConfigurationV0,
    DecoderThresholdGridV0,
    iter_decoder_configurations,
)
from langmani.language.router_types import LanguageExample, LanguageSplit, RouterStatus
from langmani.language.text_classifier import BIN_LABELS, OBJECT_LABELS, STATUS_LABELS

DECODER_METRICS_SCHEMA = "langmani-m5a-posthoc-decoder-metrics-v0"
DECODER_SELECTION_SCHEMA = "langmani-m5a-posthoc-decoder-selection-v0"


class DecoderSelectionError(RuntimeError):
    """Raised when decoder selection would violate the locked validation contract."""


@dataclass(frozen=True, slots=True)
class DecoderMetricThresholdsV0:
    minimum_full_task_accuracy: float = 0.95
    minimum_object_accuracy: float = 0.97
    minimum_bin_accuracy: float = 0.97
    maximum_false_route_rate: float = 0.03
    minimum_ambiguous_recall: float = 0.90
    minimum_unsupported_recall: float = 0.95
    minimum_malformed_recall: float = 0.95
    required_schema_valid_rate: float = 1.0

    def to_dict(self) -> dict[str, float]:
        return {
            "minimum_full_task_accuracy": self.minimum_full_task_accuracy,
            "minimum_object_accuracy": self.minimum_object_accuracy,
            "minimum_bin_accuracy": self.minimum_bin_accuracy,
            "maximum_false_route_rate": self.maximum_false_route_rate,
            "minimum_ambiguous_recall": self.minimum_ambiguous_recall,
            "minimum_unsupported_recall": self.minimum_unsupported_recall,
            "minimum_malformed_recall": self.minimum_malformed_recall,
            "required_schema_valid_rate": self.required_schema_valid_rate,
        }


DEFAULT_DECODER_METRIC_THRESHOLDS = DecoderMetricThresholdsV0()


@dataclass(frozen=True, slots=True)
class DecoderMetricsV0:
    validation_examples: int
    routeable_examples: int
    rejected_examples: int
    full_task_spec_accuracy: float
    target_object_accuracy: float
    destination_bin_accuracy: float
    binary_route_reject_accuracy: float
    route_precision: float
    route_recall: float
    rejection_precision: float
    rejection_recall: float
    false_route_rate: float
    false_rejection_rate: float
    exact_rejection_reason_accuracy: float
    ambiguous_rejection_recall: float
    unsupported_rejection_recall: float
    malformed_rejection_recall: float
    rejection_macro_recall: float
    route_coverage: float
    schema_valid_rate: float
    four_way_confusion_matrix: Mapping[str, Mapping[str, int]]
    binary_confusion_matrix: Mapping[str, Mapping[str, int]]
    eligible: bool
    quality_gate_passed: bool
    eligibility_items: Mapping[str, bool]
    quality_gate_items: Mapping[str, bool]
    schema_version: str = DECODER_METRICS_SCHEMA

    def to_dict(self) -> dict[str, object]:
        return {
            "validation_examples": self.validation_examples,
            "routeable_examples": self.routeable_examples,
            "rejected_examples": self.rejected_examples,
            "full_task_spec_accuracy": self.full_task_spec_accuracy,
            "target_object_accuracy": self.target_object_accuracy,
            "destination_bin_accuracy": self.destination_bin_accuracy,
            "binary_route_reject_accuracy": self.binary_route_reject_accuracy,
            "route_precision": self.route_precision,
            "route_recall": self.route_recall,
            "rejection_precision": self.rejection_precision,
            "rejection_recall": self.rejection_recall,
            "false_route_rate": self.false_route_rate,
            "false_rejection_rate": self.false_rejection_rate,
            "exact_rejection_reason_accuracy": self.exact_rejection_reason_accuracy,
            "ambiguous_rejection_recall": self.ambiguous_rejection_recall,
            "unsupported_rejection_recall": self.unsupported_rejection_recall,
            "malformed_rejection_recall": self.malformed_rejection_recall,
            "rejection_macro_recall": self.rejection_macro_recall,
            "route_coverage": self.route_coverage,
            "schema_valid_rate": self.schema_valid_rate,
            "four_way_confusion_matrix": {
                row: dict(columns) for row, columns in self.four_way_confusion_matrix.items()
            },
            "binary_confusion_matrix": {
                row: dict(columns) for row, columns in self.binary_confusion_matrix.items()
            },
            "eligible": self.eligible,
            "quality_gate_passed": self.quality_gate_passed,
            "eligibility_items": dict(self.eligibility_items),
            "quality_gate_items": dict(self.quality_gate_items),
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class DecoderEvaluationArraysV0:
    expected_status: np.ndarray
    expected_object: np.ndarray
    expected_bin: np.ndarray
    status_probabilities: np.ndarray
    object_probabilities: np.ndarray
    bin_probabilities: np.ndarray


@dataclass(frozen=True, slots=True)
class CandidateSelectionSummaryV0:
    candidate: DecoderCandidate
    configurations_evaluated: int
    eligible_configurations: int
    best_eligible_configuration: DecoderConfigurationV0 | None
    best_eligible_metrics: DecoderMetricsV0 | None
    diagnostic_configuration: DecoderConfigurationV0
    diagnostic_metrics: DecoderMetricsV0

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.value,
            "configurations_evaluated": self.configurations_evaluated,
            "eligible_configurations": self.eligible_configurations,
            "best_eligible_configuration": (
                None
                if self.best_eligible_configuration is None
                else self.best_eligible_configuration.to_dict()
            ),
            "best_eligible_metrics": (
                None if self.best_eligible_metrics is None else self.best_eligible_metrics.to_dict()
            ),
            "diagnostic_configuration": self.diagnostic_configuration.to_dict(),
            "diagnostic_metrics": self.diagnostic_metrics.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class DecoderSelectionResultV0:
    grid_fingerprint: str
    temperature_comparison_fingerprint: str
    baseline_configuration: DecoderConfigurationV0
    baseline_metrics: DecoderMetricsV0
    candidate_summaries: tuple[CandidateSelectionSummaryV0, ...]
    selected_configuration: DecoderConfigurationV0 | None
    selected_metrics: DecoderMetricsV0 | None
    conclusion: str
    classifier_full_quality_gate_passed: bool
    selection_fingerprint: str = ""
    schema_version: str = DECODER_SELECTION_SCHEMA

    def __post_init__(self) -> None:
        expected_conclusion = (
            "classifier_promoted_after_posthoc_calibration"
            if self.classifier_full_quality_gate_passed
            else "classifier_rejected_after_posthoc_calibration"
        )
        if self.conclusion != expected_conclusion:
            raise DecoderSelectionError("decoder conclusion differs from the exact quality gate")
        if self.classifier_full_quality_gate_passed != (
            self.selected_metrics is not None and self.selected_metrics.quality_gate_passed
        ):
            raise DecoderSelectionError("selected decoder quality flag is inconsistent")
        expected = f"sha256:{sha256_hex(self.identity_dict())}"
        if self.selection_fingerprint and self.selection_fingerprint != expected:
            raise DecoderSelectionError("decoder selection fingerprint differs")
        object.__setattr__(self, "selection_fingerprint", expected)

    def identity_dict(self) -> dict[str, object]:
        return {
            "grid_fingerprint": self.grid_fingerprint,
            "temperature_comparison_fingerprint": self.temperature_comparison_fingerprint,
            "baseline_configuration": self.baseline_configuration.to_dict(),
            "baseline_metrics": self.baseline_metrics.to_dict(),
            "candidate_summaries": [value.to_dict() for value in self.candidate_summaries],
            "selected_configuration": (
                None
                if self.selected_configuration is None
                else self.selected_configuration.to_dict()
            ),
            "selected_metrics": (
                None if self.selected_metrics is None else self.selected_metrics.to_dict()
            ),
            "conclusion": self.conclusion,
            "classifier_full_quality_gate_passed": self.classifier_full_quality_gate_passed,
            "selection_split": "validation",
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.identity_dict(), "selection_fingerprint": self.selection_fingerprint}


def _safe_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def build_decoder_evaluation_arrays(
    *,
    examples: Sequence[LanguageExample],
    status_logits: torch.Tensor,
    object_logits: torch.Tensor,
    bin_logits: torch.Tensor,
    temperature: float,
) -> DecoderEvaluationArraysV0:
    values = tuple(examples)
    count = len(values)
    if not values or any(value.split is not LanguageSplit.VALIDATION for value in values):
        raise DecoderSelectionError("decoder scoring requires validation examples only")
    if (
        tuple(status_logits.shape) != (count, len(STATUS_LABELS))
        or tuple(object_logits.shape) != (count, len(OBJECT_LABELS))
        or tuple(bin_logits.shape) != (count, len(BIN_LABELS))
        or not all(
            bool(torch.isfinite(value).all())
            for value in (status_logits, object_logits, bin_logits)
        )
    ):
        raise DecoderSelectionError("decoder logits have the wrong shape or non-finite values")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise DecoderSelectionError("decoder temperature must be finite and positive")
    status = torch.softmax(status_logits.detach().cpu().double() / temperature, dim=-1).numpy()
    objects = torch.softmax(object_logits.detach().cpu().double(), dim=-1).numpy()
    bins = torch.softmax(bin_logits.detach().cpu().double(), dim=-1).numpy()
    expected_status = np.asarray(
        [STATUS_LABELS.index(value.expected_status.value) for value in values], dtype=np.int64
    )
    expected_object = np.full(count, -1, dtype=np.int64)
    expected_bin = np.full(count, -1, dtype=np.int64)
    for index, example in enumerate(values):
        if example.expected_status is RouterStatus.ROUTE:
            task_spec = example.expected_task_spec
            if task_spec is None:
                raise DecoderSelectionError("routeable validation example lacks a TaskSpec")
            expected_object[index] = OBJECT_LABELS.index(task_spec.target_object_id)
            expected_bin[index] = BIN_LABELS.index(task_spec.target_bin_id)
    return DecoderEvaluationArraysV0(
        expected_status=expected_status,
        expected_object=expected_object,
        expected_bin=expected_bin,
        status_probabilities=status,
        object_probabilities=objects,
        bin_probabilities=bins,
    )


def _decision_arrays(
    arrays: DecoderEvaluationArraysV0,
    configuration: DecoderConfigurationV0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    status = arrays.status_probabilities
    objects = arrays.object_probabilities
    bins = arrays.bin_probabilities
    p_route = status[:, 0]
    p_reject = status[:, 1:].sum(axis=1)
    predicted_object = objects.argmax(axis=1)
    predicted_bin = bins.argmax(axis=1)
    valid_pair = (predicted_object < 3) & (predicted_bin < 2)
    if configuration.candidate is DecoderCandidate.BASELINE_FOUR_WAY_ARGMAX_V0:
        route = status.argmax(axis=1) == 0
    else:
        assert configuration.route_threshold is not None
        assert configuration.route_margin_threshold is not None
        route = (p_route >= configuration.route_threshold) & (
            p_route - p_reject >= configuration.route_margin_threshold
        )
    if configuration.candidate in {
        DecoderCandidate.HIERARCHICAL_STATUS_DECODER_V0,
        DecoderCandidate.CONSERVATIVE_ROUTE_DECODER_V0,
    }:
        assert configuration.object_confidence_threshold is not None
        assert configuration.bin_confidence_threshold is not None
        route &= objects.max(axis=1) >= configuration.object_confidence_threshold
        route &= bins.max(axis=1) >= configuration.bin_confidence_threshold
    route &= valid_pair
    rejection_reason = status[:, 1:].argmax(axis=1) + 1
    predicted_status = np.where(route, 0, rejection_reason)
    return predicted_status, predicted_object, predicted_bin


def evaluate_decoder_configuration(
    arrays: DecoderEvaluationArraysV0,
    configuration: DecoderConfigurationV0,
    *,
    thresholds: DecoderMetricThresholdsV0 = DEFAULT_DECODER_METRIC_THRESHOLDS,
) -> DecoderMetricsV0:
    predicted_status, predicted_object, predicted_bin = _decision_arrays(arrays, configuration)
    expected_status = arrays.expected_status
    routeable = expected_status == 0
    rejected = ~routeable
    predicted_route = predicted_status == 0
    predicted_reject = ~predicted_route
    routeable_count = int(routeable.sum())
    rejected_count = int(rejected.sum())
    if routeable_count == 0 or rejected_count == 0:
        raise DecoderSelectionError("selection requires routeable and rejected validation examples")
    object_correct = predicted_object == arrays.expected_object
    bin_correct = predicted_bin == arrays.expected_bin
    full_correct = predicted_route & object_correct & bin_correct
    true_route = int((routeable & predicted_route).sum())
    false_route = int((rejected & predicted_route).sum())
    true_reject = int((rejected & predicted_reject).sum())
    false_reject = int((routeable & predicted_reject).sum())
    recalls: list[float] = []
    for status_index in range(1, 4):
        selected = expected_status == status_index
        recalls.append(
            _safe_ratio(
                int((selected & (predicted_status == status_index)).sum()), int(selected.sum())
            )
        )
    exact_rejection = _safe_ratio(
        int((rejected & (predicted_status == expected_status)).sum()), rejected_count
    )
    full_task_accuracy = _safe_ratio(int((routeable & full_correct).sum()), routeable_count)
    object_accuracy = _safe_ratio(int((routeable & object_correct).sum()), routeable_count)
    bin_accuracy = _safe_ratio(int((routeable & bin_correct).sum()), routeable_count)
    false_route_rate = _safe_ratio(false_route, rejected_count)
    false_rejection_rate = _safe_ratio(false_reject, routeable_count)
    schema_valid_rate = 1.0
    eligibility = {
        "full_task_spec_accuracy_at_least_0_95": (
            full_task_accuracy >= thresholds.minimum_full_task_accuracy
        ),
        "target_object_accuracy_at_least_0_97": (
            object_accuracy >= thresholds.minimum_object_accuracy
        ),
        "destination_bin_accuracy_at_least_0_97": (bin_accuracy >= thresholds.minimum_bin_accuracy),
        "false_route_rate_at_most_0_03": (false_route_rate <= thresholds.maximum_false_route_rate),
        "schema_valid_rate_exactly_1": schema_valid_rate == thresholds.required_schema_valid_rate,
    }
    quality = {
        **eligibility,
        "ambiguous_rejection_recall_at_least_0_90": (
            recalls[0] >= thresholds.minimum_ambiguous_recall
        ),
        "unsupported_rejection_recall_at_least_0_95": (
            recalls[1] >= thresholds.minimum_unsupported_recall
        ),
        "malformed_rejection_recall_at_least_0_95": (
            recalls[2] >= thresholds.minimum_malformed_recall
        ),
    }
    four_way = {
        STATUS_LABELS[row]: {
            STATUS_LABELS[column]: int(
                ((expected_status == row) & (predicted_status == column)).sum()
            )
            for column in range(4)
        }
        for row in range(4)
    }
    binary = {
        "routeable": {"routeable": true_route, "rejectable": false_reject},
        "rejectable": {"routeable": false_route, "rejectable": true_reject},
    }
    return DecoderMetricsV0(
        validation_examples=len(expected_status),
        routeable_examples=routeable_count,
        rejected_examples=rejected_count,
        full_task_spec_accuracy=full_task_accuracy,
        target_object_accuracy=object_accuracy,
        destination_bin_accuracy=bin_accuracy,
        binary_route_reject_accuracy=_safe_ratio(
            true_route + true_reject, routeable_count + rejected_count
        ),
        route_precision=_safe_ratio(true_route, true_route + false_route),
        route_recall=_safe_ratio(true_route, routeable_count),
        rejection_precision=_safe_ratio(true_reject, true_reject + false_reject),
        rejection_recall=_safe_ratio(true_reject, rejected_count),
        false_route_rate=false_route_rate,
        false_rejection_rate=false_rejection_rate,
        exact_rejection_reason_accuracy=exact_rejection,
        ambiguous_rejection_recall=recalls[0],
        unsupported_rejection_recall=recalls[1],
        malformed_rejection_recall=recalls[2],
        rejection_macro_recall=sum(recalls) / len(recalls),
        route_coverage=_safe_ratio(true_route, routeable_count),
        schema_valid_rate=schema_valid_rate,
        four_way_confusion_matrix=four_way,
        binary_confusion_matrix=binary,
        eligible=all(eligibility.values()),
        quality_gate_passed=all(quality.values()),
        eligibility_items=eligibility,
        quality_gate_items=quality,
    )


def _selection_key(
    configuration: DecoderConfigurationV0, metrics: DecoderMetricsV0
) -> tuple[object, ...]:
    route_threshold = (
        -math.inf if configuration.route_threshold is None else configuration.route_threshold
    )
    return (
        -metrics.ambiguous_rejection_recall,
        -metrics.unsupported_rejection_recall,
        -metrics.malformed_rejection_recall,
        -metrics.rejection_macro_recall,
        metrics.false_rejection_rate,
        -metrics.route_coverage,
        -route_threshold,
        configuration.configuration_fingerprint,
    )


def _diagnostic_key(
    configuration: DecoderConfigurationV0, metrics: DecoderMetricsV0
) -> tuple[object, ...]:
    passed = sum(metrics.eligibility_items.values())
    false_route_excess = max(0.0, metrics.false_route_rate - 0.03)
    task_deficit = max(0.0, 0.95 - metrics.full_task_spec_accuracy)
    return (
        -passed,
        false_route_excess,
        task_deficit,
        *_selection_key(configuration, metrics),
    )


def select_fixed_grid_decoder(
    *,
    examples: Sequence[LanguageExample],
    status_logits: torch.Tensor,
    object_logits: torch.Tensor,
    bin_logits: torch.Tensor,
    grid: DecoderThresholdGridV0,
    temperature_values: Mapping[str, float],
    temperature_comparison_fingerprint: str,
) -> DecoderSelectionResultV0:
    """Evaluate exactly the locked grid and apply the unchanged conjunctive gate."""

    if not temperature_comparison_fingerprint.startswith("sha256:"):
        raise DecoderSelectionError("temperature comparison fingerprint is required")
    arrays_by_mode = {
        mode: build_decoder_evaluation_arrays(
            examples=examples,
            status_logits=status_logits,
            object_logits=object_logits,
            bin_logits=bin_logits,
            temperature=temperature_values[mode],
        )
        for mode in grid.temperature_modes
    }
    grouped: dict[DecoderCandidate, list[tuple[DecoderConfigurationV0, DecoderMetricsV0]]] = {
        candidate: [] for candidate in DecoderCandidate
    }
    baseline_configuration: DecoderConfigurationV0 | None = None
    baseline_metrics: DecoderMetricsV0 | None = None
    for configuration in iter_decoder_configurations(grid, temperature_values=temperature_values):
        metrics = evaluate_decoder_configuration(
            arrays_by_mode[configuration.temperature_mode], configuration
        )
        grouped[configuration.candidate].append((configuration, metrics))
        if (
            configuration.candidate is DecoderCandidate.BASELINE_FOUR_WAY_ARGMAX_V0
            and configuration.temperature_mode == "identity"
        ):
            baseline_configuration = configuration
            baseline_metrics = metrics
    if baseline_configuration is None or baseline_metrics is None:
        raise DecoderSelectionError("locked candidate set omitted the baseline")
    summaries: list[CandidateSelectionSummaryV0] = []
    all_eligible: list[tuple[DecoderConfigurationV0, DecoderMetricsV0]] = []
    for candidate in DecoderCandidate:
        values = grouped[candidate]
        if not values:
            raise DecoderSelectionError(f"locked grid omitted {candidate.value}")
        eligible = [value for value in values if value[1].eligible]
        best_eligible = (
            min(eligible, key=lambda value: _selection_key(*value)) if eligible else None
        )
        diagnostic = min(values, key=lambda value: _diagnostic_key(*value))
        all_eligible.extend(eligible)
        summaries.append(
            CandidateSelectionSummaryV0(
                candidate=candidate,
                configurations_evaluated=len(values),
                eligible_configurations=len(eligible),
                best_eligible_configuration=None if best_eligible is None else best_eligible[0],
                best_eligible_metrics=None if best_eligible is None else best_eligible[1],
                diagnostic_configuration=diagnostic[0],
                diagnostic_metrics=diagnostic[1],
            )
        )
    selected_pair = (
        min(all_eligible, key=lambda value: _selection_key(*value)) if all_eligible else None
    )
    selected_configuration = None if selected_pair is None else selected_pair[0]
    selected_metrics = None if selected_pair is None else selected_pair[1]
    quality_passed = selected_metrics is not None and selected_metrics.quality_gate_passed
    return DecoderSelectionResultV0(
        grid_fingerprint=grid.grid_fingerprint,
        temperature_comparison_fingerprint=temperature_comparison_fingerprint,
        baseline_configuration=baseline_configuration,
        baseline_metrics=baseline_metrics,
        candidate_summaries=tuple(summaries),
        selected_configuration=selected_configuration,
        selected_metrics=selected_metrics,
        conclusion=(
            "classifier_promoted_after_posthoc_calibration"
            if quality_passed
            else "classifier_rejected_after_posthoc_calibration"
        ),
        classifier_full_quality_gate_passed=quality_passed,
    )


def decoder_runtime_fingerprint(
    *,
    classifier_checkpoint_fingerprint: str,
    tokenizer_fingerprint: str,
    processor_fingerprint: str,
    selection: DecoderSelectionResultV0,
    git_commit: str,
    corpus_fingerprint: str,
    validation_split_fingerprint: str,
) -> str:
    if (
        not selection.classifier_full_quality_gate_passed
        or selection.selected_configuration is None
    ):
        raise DecoderSelectionError("a rejected decoder cannot receive a promoted runtime identity")
    payload = {
        "classifier_checkpoint_fingerprint": classifier_checkpoint_fingerprint,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "processor_fingerprint": processor_fingerprint,
        "decoder": selection.selected_configuration.to_dict(),
        "selection_fingerprint": selection.selection_fingerprint,
        "git_commit": git_commit,
        "corpus_fingerprint": corpus_fingerprint,
        "validation_split_fingerprint": validation_split_fingerprint,
        "rejection_semantics": "rejected_decisions_never_contain_executable_taskspec_v0",
        "schema_version": "langmani-m5a-classifier-routing-runtime-v0",
    }
    return f"sha256:{sha256_hex(payload)}"


__all__ = [
    "DECODER_METRICS_SCHEMA",
    "DECODER_SELECTION_SCHEMA",
    "CandidateSelectionSummaryV0",
    "DecoderEvaluationArraysV0",
    "DecoderMetricThresholdsV0",
    "DecoderMetricsV0",
    "DecoderSelectionError",
    "DecoderSelectionResultV0",
    "build_decoder_evaluation_arrays",
    "decoder_runtime_fingerprint",
    "evaluate_decoder_configuration",
    "select_fixed_grid_decoder",
]
