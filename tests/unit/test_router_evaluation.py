from __future__ import annotations

from collections.abc import Mapping

import pytest

from langmani.environments.specs import TaskSpec
from langmani.language.router_evaluation import (
    RouterEvaluationError,
    evaluate_language_router,
)
from langmani.language.router_types import (
    LanguageExample,
    LanguageSplit,
    RouterConfidence,
    RouterDecision,
    RouterRejectionReason,
    RouterStatus,
    stable_language_example_id,
)


def _example(
    *,
    raw_text: str,
    family: str,
    status: RouterStatus,
    task_spec: TaskSpec | None,
    reason: RouterRejectionReason | None,
    split: LanguageSplit = LanguageSplit.VALIDATION,
) -> LanguageExample:
    normalized = raw_text.lower()
    provenance = {"fixture": True}
    example_id = stable_language_example_id(
        raw_text=raw_text,
        normalized_text=normalized,
        template_family_id=family,
        lexical_variant_ids=(),
        expected_status=status,
        expected_task_spec=task_spec,
        expected_rejection_reason=reason,
        generation_provenance=provenance,
        split=split,
    )
    return LanguageExample(
        example_id=example_id,
        raw_text=raw_text,
        normalized_text=normalized,
        template_family_id=family,
        lexical_variant_ids=(),
        expected_status=status,
        expected_task_spec=task_spec,
        expected_rejection_reason=reason,
        generation_provenance=provenance,
        split=split,
    )


class _Router:
    def __init__(self, decisions: Mapping[str, RouterDecision]) -> None:
        self.decisions = decisions

    def route(self, command: str) -> RouterDecision:
        return self.decisions[command]


def test_language_router_evaluation_reports_split_metrics_and_repeatability() -> None:
    task = TaskSpec.from_mapping(
        {
            "target_object_id": "red_cube",
            "target_bin_id": "left_bin",
            "instruction_template_id": "canonical_v0",
        }
    )
    route_example = _example(
        raw_text="Put the red cube in the left bin",
        family="route-family",
        status=RouterStatus.ROUTE,
        task_spec=task,
        reason=None,
    )
    reject_example = _example(
        raw_text="Open the drawer",
        family="reject-family",
        status=RouterStatus.REJECT_UNSUPPORTED,
        task_spec=None,
        reason=RouterRejectionReason.UNSUPPORTED_ACTION,
    )
    confidence = RouterConfidence.probability(
        0.9,
        definition="fixture full-route probability",
        calibrated=True,
    )
    router = _Router(
        {
            route_example.raw_text: RouterDecision.route(
                task_spec=task,
                confidence=confidence,
                router_name="fixture",
                router_version="v0",
            ),
            reject_example.raw_text: RouterDecision.reject(
                status=RouterStatus.REJECT_UNSUPPORTED,
                rejection_reason=RouterRejectionReason.UNSUPPORTED_ACTION,
                confidence=RouterConfidence.probability(
                    0.1,
                    definition="fixture full-route probability",
                    calibrated=True,
                ),
                router_name="fixture",
                router_version="v0",
            ),
        }
    )
    times = iter((0.0, 0.001, 1.0, 1.003))

    records, summary = evaluate_language_router(
        router=router,
        examples=(route_example, reject_example),
        split=LanguageSplit.VALIDATION,
        repeat_count=3,
        ece_bins=5,
        clock=lambda: next(times),
    )

    assert len(records) == 2
    assert summary.status_accuracy == 1.0
    assert summary.valid_full_task_accuracy == 1.0
    assert summary.object_accuracy == 1.0
    assert summary.bin_accuracy == 1.0
    assert summary.route_recall == 1.0
    assert summary.false_rejection_rate == 0.0
    assert summary.rejection_precision == 1.0
    assert summary.rejection_recall == 1.0
    assert summary.false_route_rate == 0.0
    assert summary.rejection_reason_accuracy == 1.0
    assert summary.unsupported_rejection_recall == 1.0
    assert summary.schema_valid_output_rate == 1.0
    assert summary.coverage == 0.5
    assert summary.selective_accuracy == 1.0
    assert summary.expected_calibration_error == pytest.approx(0.1)
    assert summary.deterministic_repeatability == 1.0
    assert summary.latency_p50_ms == pytest.approx(2.0)
    assert set(summary.per_family_accuracy) == {"reject-family", "route-family"}


def test_language_router_evaluation_refuses_train_and_final() -> None:
    example = _example(
        raw_text="noise",
        family="sealed-family",
        status=RouterStatus.REJECT_MALFORMED,
        task_spec=None,
        reason=RouterRejectionReason.MEANINGLESS_TEXT,
        split=LanguageSplit.FINAL,
    )

    with pytest.raises(RouterEvaluationError, match="validation/development"):
        evaluate_language_router(
            router=_Router({}),
            examples=(example,),
            split=LanguageSplit.FINAL,
        )
