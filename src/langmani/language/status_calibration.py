"""Validation-only status temperature diagnostics for M5A.1."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from langmani.datasets.identity import sha256_hex
from langmani.language.text_calibration import fit_validation_temperature

STATUS_CALIBRATION_SCHEMA = "langmani-m5a-status-temperature-comparison-v0"


class StatusCalibrationError(ValueError):
    """Raised when status calibration would leave the validation-only contract."""


def expected_calibration_error(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    bins: int = 15,
) -> float:
    if (
        probabilities.ndim != 2
        or probabilities.shape[0] == 0
        or labels.shape != (probabilities.shape[0],)
        or labels.dtype != torch.long
    ):
        raise StatusCalibrationError("ECE inputs must be probabilities[N,C] and long labels[N]")
    if bins <= 0 or bins > 100:
        raise StatusCalibrationError("ECE bin count must lie in [1,100]")
    values = probabilities.detach().to(dtype=torch.float64, device="cpu")
    targets = labels.detach().to(device="cpu")
    if not bool(torch.isfinite(values).all()) or bool((values < 0.0).any()):
        raise StatusCalibrationError("ECE probabilities must be finite and non-negative")
    if not bool(
        torch.allclose(
            values.sum(dim=-1),
            torch.ones(values.shape[0], dtype=torch.float64),
            atol=1e-6,
            rtol=1e-6,
        )
    ):
        raise StatusCalibrationError("ECE probabilities must sum to one")
    confidence, predicted = values.max(dim=-1)
    correct = predicted.eq(targets).to(dtype=torch.float64)
    result = torch.zeros((), dtype=torch.float64)
    boundaries = torch.linspace(0.0, 1.0, bins + 1, dtype=torch.float64)
    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        selected = (confidence > lower) & (confidence <= upper)
        if index == 0:
            selected |= confidence == 0.0
        if bool(selected.any()):
            weight = selected.to(dtype=torch.float64).mean()
            result += weight * torch.abs(correct[selected].mean() - confidence[selected].mean())
    return float(result)


@dataclass(frozen=True, slots=True)
class StatusTemperatureComparisonV0:
    validation_examples: int
    identity_temperature: float
    fitted_temperature: float
    nll_before: float
    nll_after: float
    ece_before: float
    ece_after: float
    bounded_iterations: int
    comparison_fingerprint: str = ""
    schema_version: str = STATUS_CALIBRATION_SCHEMA

    def __post_init__(self) -> None:
        if self.validation_examples <= 0 or self.bounded_iterations <= 0:
            raise StatusCalibrationError("temperature comparison counts must be positive")
        for name in (
            "identity_temperature",
            "fitted_temperature",
            "nll_before",
            "nll_after",
            "ece_before",
            "ece_after",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise StatusCalibrationError(f"{name} must be finite and non-negative")
        if self.identity_temperature != 1.0 or self.fitted_temperature <= 0.0:
            raise StatusCalibrationError("temperature comparison identities are malformed")
        if self.schema_version != STATUS_CALIBRATION_SCHEMA:
            raise StatusCalibrationError("unknown status calibration schema")
        expected = f"sha256:{sha256_hex(self.identity_dict())}"
        if self.comparison_fingerprint and self.comparison_fingerprint != expected:
            raise StatusCalibrationError("temperature comparison fingerprint differs")
        object.__setattr__(self, "comparison_fingerprint", expected)

    def identity_dict(self) -> dict[str, object]:
        return {
            "validation_examples": self.validation_examples,
            "identity_temperature": self.identity_temperature,
            "fitted_temperature": self.fitted_temperature,
            "nll_before": self.nll_before,
            "nll_after": self.nll_after,
            "ece_before": self.ece_before,
            "ece_after": self.ece_after,
            "bounded_iterations": self.bounded_iterations,
            "evidence_split": "validation",
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.identity_dict(), "comparison_fingerprint": self.comparison_fingerprint}


def fit_status_temperature_comparison(
    status_logits: torch.Tensor,
    status_labels: torch.Tensor,
    *,
    evidence_split: str,
    iterations: int = 64,
) -> StatusTemperatureComparisonV0:
    if evidence_split != "validation":
        raise StatusCalibrationError("status calibration may use validation evidence only")
    calibration = fit_validation_temperature(
        status_logits,
        status_labels,
        evidence_split=evidence_split,
        iterations=iterations,
    )
    raw = F.softmax(status_logits.detach().to(dtype=torch.float64, device="cpu"), dim=-1)
    calibrated = F.softmax(
        status_logits.detach().to(dtype=torch.float64, device="cpu") / calibration.temperature,
        dim=-1,
    )
    return StatusTemperatureComparisonV0(
        validation_examples=int(status_logits.shape[0]),
        identity_temperature=1.0,
        fitted_temperature=calibration.temperature,
        nll_before=calibration.nll_before,
        nll_after=calibration.nll_after,
        ece_before=expected_calibration_error(raw, status_labels),
        ece_after=expected_calibration_error(calibrated, status_labels),
        bounded_iterations=calibration.bounded_iterations,
    )


__all__ = [
    "STATUS_CALIBRATION_SCHEMA",
    "StatusCalibrationError",
    "StatusTemperatureComparisonV0",
    "expected_calibration_error",
    "fit_status_temperature_comparison",
]
