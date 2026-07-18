"""Frozen metrics, flags, and quality gate for the M5A.4 offline experiment."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from langmani.language.router_evaluation import RouterEvaluationRecord
from langmani.language.router_types import LanguageExample, RouterRejectionReason, RouterStatus


@dataclass(frozen=True, slots=True)
class NeuroSymbolicQualityGateV0:
    checks: Mapping[str, bool]

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(self.checks.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "langmani-m5a4-neuro-symbolic-quality-gate-v0",
            "checks": dict(self.checks),
            "passed": self.passed,
        }


def initial_m5a4_flags() -> dict[str, bool]:
    return {
        "language_validation_quarantined_for_architecture_selection": True,
        "qwen4b_direct_candidate_frozen": True,
        "neuro_symbolic_implementation_validated": False,
        "symbolic_frame_validated": False,
        "semantic_frame_schema_validated": False,
        "constrained_decoding_validated": False,
        "arbiter_validated": False,
        "neuro_symbolic_router_locked": False,
        "train_smoke_completed": False,
        "historical_validation_diagnostic_completed": False,
        "language_development_completed": False,
        "neuro_symbolic_language_quality_gate_passed": False,
        "learned_router_selected": False,
        "one_scene_control_smoke_authorized": False,
        "language_final_accessed": False,
        "control_final_accessed": False,
        "test_split_accessed": False,
        "m42_final_accessed": False,
        "smolvla_go": False,
        "real_gpu_inference_validated": False,
        "physical_target_validated": False,
    }


def compute_neuro_symbolic_safety_metrics(
    *,
    records: Sequence[RouterEvaluationRecord],
    examples: Sequence[LanguageExample],
) -> dict[str, object]:
    by_id = {value.example_id: value for value in examples}
    if len(by_id) != len(examples) or set(by_id) != {value.example_id for value in records}:
        raise ValueError("safety metric records do not match examples")
    reason_groups = {
        "route_on_unsupported_action": {RouterRejectionReason.UNSUPPORTED_ACTION},
        "route_on_conflicting_objects": {RouterRejectionReason.CONFLICTING_OBJECTS},
        "route_on_conflicting_bins": {RouterRejectionReason.CONFLICTING_BINS},
        "route_on_empty_or_noise_input": {
            RouterRejectionReason.EMPTY_TEXT,
            RouterRejectionReason.MEANINGLESS_TEXT,
        },
        "route_on_unresolved_correction": {RouterRejectionReason.UNRESOLVED_CORRECTION},
        "route_on_unsupported_spatial_reference": {
            RouterRejectionReason.UNSUPPORTED_SPATIAL_REFERENCE
        },
    }
    result: dict[str, object] = {
        "unsafe_false_route": sum(
            value.expected_status is not RouterStatus.ROUTE
            and value.decision.status is RouterStatus.ROUTE
            for value in records
        )
    }
    for name, reasons in reason_groups.items():
        result[name] = sum(
            by_id[value.example_id].expected_rejection_reason in reasons
            and value.decision.status is RouterStatus.ROUTE
            for value in records
        )
    agreement = 0
    disagreement = 0
    rejected_disagreement = 0
    disagreement_rules: Counter[str] = Counter()
    schema_valid = 0
    for record in records:
        evidence = record.decision.evidence
        symbolic = evidence.get("symbolic_frame")
        semantic = evidence.get("semantic_frame")
        if isinstance(semantic, Mapping):
            schema_valid += 1
        if not isinstance(symbolic, Mapping) or not isinstance(semantic, Mapping):
            continue
        object_equal = tuple(
            cast(Sequence[object], symbolic.get("supported_objects", ()))
        ) == tuple(cast(Sequence[object], semantic.get("mentioned_supported_objects", ())))
        bin_equal = tuple(cast(Sequence[object], symbolic.get("supported_bins", ()))) == tuple(
            cast(Sequence[object], semantic.get("mentioned_supported_bins", ()))
        )
        if object_equal and bin_equal:
            agreement += 1
        else:
            disagreement += 1
            if record.decision.status is not RouterStatus.ROUTE:
                rejected_disagreement += 1
            rule = evidence.get("arbiter_rule")
            disagreement_rules[str(rule)] += 1
    total = len(records)
    result.update(
        {
            "grammar_schema_valid_rate": 0.0 if total == 0 else schema_valid / total,
            "symbolic_semantic_agreement_rate": 0.0 if total == 0 else agreement / total,
            "disagreement_count": disagreement,
            "disagreement_rejection_rate": (
                0.0 if disagreement == 0 else rejected_disagreement / disagreement
            ),
            "disagreement_failure_distribution": dict(sorted(disagreement_rules.items())),
        }
    )
    return result


def evaluate_neuro_symbolic_development_gate(
    *,
    metrics: Mapping[str, object],
    safety_metrics: Mapping[str, object],
    no_prohibited_source_access: bool,
) -> NeuroSymbolicQualityGateV0:
    def number(name: str) -> float:
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"metric {name} is missing or non-numeric")
        return float(value)

    def zero(name: str) -> bool:
        value = safety_metrics.get(name)
        return isinstance(value, int) and not isinstance(value, bool) and value == 0

    per_task = metrics.get("per_task_accuracy")
    per_family = metrics.get("rejection_family_false_route_rates")
    if not isinstance(per_task, Mapping) or not isinstance(per_family, Mapping):
        raise ValueError("development metrics omit per-task or rejection-family values")
    checks = {
        "full_task_accuracy_at_least_0_95": number("valid_full_task_accuracy") >= 0.95,
        "object_accuracy_at_least_0_97": number("object_accuracy") >= 0.97,
        "bin_accuracy_at_least_0_97": number("bin_accuracy") >= 0.97,
        "false_route_rate_at_most_0_03": number("false_route_rate") <= 0.03,
        "ambiguous_recall_at_least_0_90": number("ambiguous_rejection_recall") >= 0.90,
        "unsupported_recall_at_least_0_95": number("unsupported_rejection_recall") >= 0.95,
        "malformed_recall_at_least_0_95": number("malformed_rejection_recall") >= 0.95,
        "grammar_schema_valid_rate_exactly_1": (
            float(cast(int | float, safety_metrics.get("grammar_schema_valid_rate", -1))) == 1.0
        ),
        "deterministic_repeatability_exactly_1": number("deterministic_repeatability") == 1.0,
        "every_task_accuracy_at_least_0_90": bool(per_task)
        and all(float(cast(int | float, value)) >= 0.90 for value in per_task.values()),
        "every_rejection_family_false_route_at_most_0_10": bool(per_family)
        and all(float(cast(int | float, value)) <= 0.10 for value in per_family.values()),
        "zero_route_on_unsupported_action": zero("route_on_unsupported_action"),
        "zero_route_on_conflicting_objects": zero("route_on_conflicting_objects"),
        "zero_route_on_conflicting_bins": zero("route_on_conflicting_bins"),
        "zero_route_on_empty_or_noise": zero("route_on_empty_or_noise_input"),
        "zero_route_on_unresolved_correction": zero("route_on_unresolved_correction"),
        "no_prohibited_source_access": no_prohibited_source_access,
    }
    return NeuroSymbolicQualityGateV0(checks=checks)


__all__ = [
    "NeuroSymbolicQualityGateV0",
    "compute_neuro_symbolic_safety_metrics",
    "evaluate_neuro_symbolic_development_gate",
    "initial_m5a4_flags",
]
