"""Validation-only temperature calibration and selective-routing thresholding."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.nn import functional as F


class RouterCalibrationError(RuntimeError):
    """Raised when calibration evidence violates the M5A selection contract."""


@dataclass(frozen=True, slots=True)
class TemperatureCalibrationV0:
    temperature: float
    validation_examples: int
    nll_before: float
    nll_after: float
    bounded_iterations: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        if self.validation_examples <= 0 or self.bounded_iterations <= 0:
            raise ValueError("calibration counts must be positive")
        if not all(
            math.isfinite(value) and value >= 0.0 for value in (self.nll_before, self.nll_after)
        ):
            raise ValueError("calibration losses must be finite and non-negative")

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not bool(torch.isfinite(logits).all()):
            raise RouterCalibrationError("calibration logits are non-finite")
        return logits / self.temperature


def _validated_logits_labels(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 2:
        raise RouterCalibrationError("logits must have non-empty [examples,classes] shape")
    if labels.shape != (logits.shape[0],) or labels.dtype != torch.long:
        raise RouterCalibrationError("labels must be torch.long[examples]")
    if not bool(torch.isfinite(logits).all()):
        raise RouterCalibrationError("logits contain NaN or infinity")
    if bool((labels < 0).any()) or bool((labels >= logits.shape[1]).any()):
        raise RouterCalibrationError("labels fall outside the logits classes")
    return logits.detach().to(dtype=torch.float64, device="cpu"), labels.detach().to(device="cpu")


def fit_validation_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    evidence_split: str,
    minimum: float = 0.05,
    maximum: float = 20.0,
    iterations: int = 64,
) -> TemperatureCalibrationV0:
    """Fit one scalar on validation evidence with a deterministic bounded search."""

    if evidence_split != "validation":
        raise RouterCalibrationError("temperature scaling may use validation evidence only")
    if not (0.0 < minimum < maximum) or iterations <= 0 or iterations > 256:
        raise RouterCalibrationError("temperature-search bounds are invalid")
    values, targets = _validated_logits_labels(logits, labels)

    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        return float(F.cross_entropy(values / temperature, targets).item())

    left = math.log(minimum)
    right = math.log(maximum)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    f1 = objective(x1)
    f2 = objective(x2)
    for _ in range(iterations):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = objective(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = objective(x2)
    candidates = (
        (minimum, objective(math.log(minimum))),
        (maximum, objective(math.log(maximum))),
        (math.exp(x1), f1),
        (math.exp(x2), f2),
    )
    temperature, calibrated_nll = min(candidates, key=lambda item: (item[1], item[0]))
    return TemperatureCalibrationV0(
        temperature=temperature,
        validation_examples=values.shape[0],
        nll_before=objective(0.0),
        nll_after=calibrated_nll,
        bounded_iterations=iterations,
    )


def calibrated_full_route_confidence(
    *,
    status_logits: torch.Tensor,
    object_logits: torch.Tensor,
    bin_logits: torch.Tensor,
    temperature: TemperatureCalibrationV0,
) -> torch.Tensor:
    """Return P(route) times the winning object and bin probabilities."""

    if status_logits.ndim != 2 or object_logits.ndim != 2 or bin_logits.ndim != 2:
        raise RouterCalibrationError("all confidence logits must be rank two")
    if not (status_logits.shape[0] == object_logits.shape[0] == bin_logits.shape[0]):
        raise RouterCalibrationError("confidence logits must share a batch dimension")
    status = F.softmax(temperature.apply(status_logits), dim=-1)[:, 0]
    target_object = F.softmax(object_logits, dim=-1).max(dim=-1).values
    target_bin = F.softmax(bin_logits, dim=-1).max(dim=-1).values
    confidence = status * target_object * target_bin
    if not bool(torch.isfinite(confidence).all()):
        raise RouterCalibrationError("full-route confidence is non-finite")
    return confidence


@dataclass(frozen=True, slots=True)
class RoutingThresholdExample:
    expected_route: bool
    expected_object_id: str | None
    expected_bin_id: str | None
    predicted_status: str
    predicted_object_id: str | None
    predicted_bin_id: str | None
    full_route_confidence: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.full_route_confidence)
            or not 0.0 <= self.full_route_confidence <= 1.0
        ):
            raise ValueError("full_route_confidence must lie in [0,1]")
        if self.expected_route and (
            self.expected_object_id is None or self.expected_bin_id is None
        ):
            raise ValueError("routeable evidence requires expected object and bin")
        if not self.expected_route and (
            self.expected_object_id is not None or self.expected_bin_id is not None
        ):
            raise ValueError("rejection evidence must not contain an expected task")


@dataclass(frozen=True, slots=True)
class RoutingThresholdSelectionV0:
    threshold: float
    validation_routeable_examples: int
    validation_rejected_examples: int
    valid_full_task_accuracy: float
    false_route_rate: float
    rejection_recall: float


def select_validation_routing_threshold(
    examples: Sequence[RoutingThresholdExample],
    *,
    evidence_split: str,
    candidates: Sequence[float],
    maximum_false_route_rate: float = 0.03,
) -> RoutingThresholdSelectionV0:
    """Apply the declared lexicographic validation objective exactly."""

    if evidence_split != "validation":
        raise RouterCalibrationError("routing threshold selection may use validation evidence only")
    values = tuple(examples)
    routeable = tuple(value for value in values if value.expected_route)
    rejected = tuple(value for value in values if not value.expected_route)
    if not routeable or not rejected:
        raise RouterCalibrationError(
            "threshold selection requires routeable and rejected validation examples"
        )
    thresholds = tuple(sorted(set(float(value) for value in candidates)))
    if not thresholds or any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in thresholds
    ):
        raise RouterCalibrationError("threshold candidates must be finite values in [0,1]")
    if not 0.0 <= maximum_false_route_rate <= 1.0:
        raise RouterCalibrationError("maximum_false_route_rate must lie in [0,1]")

    def dispatches(value: RoutingThresholdExample, threshold: float) -> bool:
        return (
            value.predicted_status == "route"
            and value.predicted_object_id is not None
            and value.predicted_bin_id is not None
            and value.full_route_confidence >= threshold
        )

    feasible: list[RoutingThresholdSelectionV0] = []
    for threshold in thresholds:
        correct = sum(
            dispatches(value, threshold)
            and value.predicted_object_id == value.expected_object_id
            and value.predicted_bin_id == value.expected_bin_id
            for value in routeable
        )
        false_routes = sum(dispatches(value, threshold) for value in rejected)
        accuracy = correct / len(routeable)
        false_route_rate = false_routes / len(rejected)
        rejection_recall = 1.0 - false_route_rate
        if false_route_rate <= maximum_false_route_rate:
            feasible.append(
                RoutingThresholdSelectionV0(
                    threshold=threshold,
                    validation_routeable_examples=len(routeable),
                    validation_rejected_examples=len(rejected),
                    valid_full_task_accuracy=accuracy,
                    false_route_rate=false_route_rate,
                    rejection_recall=rejection_recall,
                )
            )
    if not feasible:
        raise RouterCalibrationError("no declared threshold satisfies the false-route constraint")
    return max(
        feasible,
        key=lambda value: (
            value.valid_full_task_accuracy,
            value.rejection_recall,
            value.threshold,
        ),
    )


__all__ = [
    "RouterCalibrationError",
    "RoutingThresholdExample",
    "RoutingThresholdSelectionV0",
    "TemperatureCalibrationV0",
    "calibrated_full_route_confidence",
    "fit_validation_temperature",
    "select_validation_routing_threshold",
]
