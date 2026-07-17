from __future__ import annotations

import pytest
import torch

from langmani.language.text_calibration import (
    RouterCalibrationError,
    RoutingThresholdExample,
    fit_validation_temperature,
    select_validation_routing_threshold,
)


def test_temperature_scaling_uses_validation_only_and_is_bounded() -> None:
    logits = torch.tensor([[8.0, 0.0], [8.0, 0.0], [0.0, 8.0]], dtype=torch.float32)
    labels = torch.tensor([0, 1, 1], dtype=torch.long)

    result = fit_validation_temperature(
        logits,
        labels,
        evidence_split="validation",
        iterations=32,
    )

    assert 0.05 <= result.temperature <= 20.0
    assert result.validation_examples == 3
    assert result.nll_after <= result.nll_before
    with pytest.raises(RouterCalibrationError, match="validation"):
        fit_validation_temperature(logits, labels, evidence_split="development")


def _example(
    *,
    expected_route: bool,
    confidence: float,
    correct: bool = True,
) -> RoutingThresholdExample:
    return RoutingThresholdExample(
        expected_route=expected_route,
        expected_object_id="red_cube" if expected_route else None,
        expected_bin_id="left_bin" if expected_route else None,
        predicted_status="route",
        predicted_object_id="red_cube" if correct else "blue_cube",
        predicted_bin_id="left_bin",
        full_route_confidence=confidence,
    )


def test_threshold_selection_enforces_false_route_constraint_then_accuracy() -> None:
    examples = (
        _example(expected_route=True, confidence=0.95),
        _example(expected_route=True, confidence=0.75),
        _example(expected_route=True, confidence=0.60, correct=False),
        _example(expected_route=False, confidence=0.70),
        RoutingThresholdExample(
            expected_route=False,
            expected_object_id=None,
            expected_bin_id=None,
            predicted_status="reject_unsupported",
            predicted_object_id=None,
            predicted_bin_id=None,
            full_route_confidence=0.99,
        ),
    )

    result = select_validation_routing_threshold(
        examples,
        evidence_split="validation",
        candidates=(0.5, 0.7, 0.8, 0.9),
        maximum_false_route_rate=0.0,
    )

    assert result.threshold == 0.9
    assert result.valid_full_task_accuracy == pytest.approx(1.0 / 3.0)
    assert result.false_route_rate == 0.0


def test_threshold_tie_break_prefers_higher_threshold() -> None:
    examples = (
        _example(expected_route=True, confidence=0.99),
        RoutingThresholdExample(
            expected_route=False,
            expected_object_id=None,
            expected_bin_id=None,
            predicted_status="reject_malformed",
            predicted_object_id=None,
            predicted_bin_id=None,
            full_route_confidence=0.99,
        ),
    )

    result = select_validation_routing_threshold(
        examples,
        evidence_split="validation",
        candidates=(0.5, 0.7, 0.9),
        maximum_false_route_rate=0.0,
    )

    assert result.threshold == 0.9


def test_threshold_selection_rejects_development_evidence() -> None:
    with pytest.raises(RouterCalibrationError, match="validation"):
        select_validation_routing_threshold(
            (
                _example(expected_route=True, confidence=0.9),
                _example(expected_route=False, confidence=0.1),
            ),
            evidence_split="development",
            candidates=(0.5,),
        )
