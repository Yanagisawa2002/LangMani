"""Deterministic offline metrics and checkpoint selection for Phase 2C."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from langmani.datasets.identity import sha256_hex
from langmani.v2.phase2c import Phase2CContractError


def _finite_array(value: object, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise Phase2CContractError(f"{label} must contain only finite values")
    return array


@dataclass(slots=True)
class OfflineMetricAccumulator:
    """Accumulate full-state action reconstruction metrics without retaining samples."""

    horizon: int = 50
    action_dimension: int = 8
    sample_count: int = 0
    valid_action_count: int = 0
    loss_weighted_sum: float = 0.0
    normalized_absolute_sum: float = 0.0
    raw_absolute_sum: float = 0.0
    raw_squared_sum: float = 0.0
    first_action_absolute_sum: float = 0.0
    per_dimension_absolute_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(8, dtype=np.float64)
    )
    per_dimension_count: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.int64))
    per_horizon_absolute_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(50, dtype=np.float64)
    )
    per_horizon_count: np.ndarray = field(default_factory=lambda: np.zeros(50, dtype=np.int64))
    prediction_sum: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float64))
    prediction_squared_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(8, dtype=np.float64)
    )

    def add(
        self,
        *,
        validation_loss: float,
        predicted_normalized: object,
        expected_normalized: object,
        predicted_raw: object,
        expected_raw: object,
        action_is_pad: object | None = None,
    ) -> None:
        """Add one batch using the official processor's normalized and raw spaces."""

        if not math.isfinite(validation_loss):
            raise Phase2CContractError("validation loss must be finite")
        predicted_norm = _finite_array(predicted_normalized, "normalized prediction")
        expected_norm = _finite_array(expected_normalized, "normalized target")
        predicted = _finite_array(predicted_raw, "inverse-normalized prediction")
        expected = _finite_array(expected_raw, "raw target")
        shape = (predicted.shape[0], self.horizon, self.action_dimension)
        if predicted.shape != shape or expected.shape != shape:
            raise Phase2CContractError(
                f"raw action chunks must have shape [B,{self.horizon},{self.action_dimension}]"
            )
        if predicted_norm.shape != shape or expected_norm.shape != shape:
            raise Phase2CContractError("normalized and raw action chunks must share shape")
        if action_is_pad is None:
            valid = np.ones(shape[:2], dtype=bool)
        else:
            padding = np.asarray(action_is_pad)
            if padding.shape != shape[:2] or padding.dtype != np.dtype(bool):
                raise Phase2CContractError("action_is_pad must be bool[B,horizon]")
            valid = ~padding
        batch_size = shape[0]
        valid_rows = int(np.count_nonzero(valid))
        if valid_rows < 1 or not bool(np.all(valid[:, 0])):
            raise Phase2CContractError("every validation sample requires a valid first action")
        normalized_error = np.abs(predicted_norm - expected_norm)
        raw_error = predicted - expected
        raw_absolute = np.abs(raw_error)
        self.sample_count += batch_size
        self.valid_action_count += valid_rows
        self.loss_weighted_sum += validation_loss * batch_size
        self.normalized_absolute_sum += float(normalized_error[valid].sum())
        self.raw_absolute_sum += float(raw_absolute[valid].sum())
        self.raw_squared_sum += float(np.square(raw_error)[valid].sum())
        self.first_action_absolute_sum += float(raw_absolute[:, 0, :].sum())
        for dimension in range(self.action_dimension):
            values = raw_absolute[:, :, dimension][valid]
            predictions = predicted[:, :, dimension][valid]
            self.per_dimension_absolute_sum[dimension] += float(values.sum())
            self.per_dimension_count[dimension] += values.size
            self.prediction_sum[dimension] += float(predictions.sum())
            self.prediction_squared_sum[dimension] += float(np.square(predictions).sum())
        for position in range(self.horizon):
            position_valid = valid[:, position]
            values = raw_absolute[:, position, :][position_valid]
            self.per_horizon_absolute_sum[position] += float(values.sum())
            self.per_horizon_count[position] += int(values.size)

    def finalize(self) -> dict[str, object]:
        """Return finite aggregate metrics used by the frozen selection rule."""

        if self.sample_count < 1 or self.valid_action_count < 1:
            raise Phase2CContractError("offline validation accumulated no samples")
        component_count = self.valid_action_count * self.action_dimension
        means = self.prediction_sum / self.per_dimension_count
        variances = np.maximum(
            self.prediction_squared_sum / self.per_dimension_count - np.square(means), 0.0
        )
        return {
            "sample_count": self.sample_count,
            "valid_action_count": self.valid_action_count,
            "validation_loss": self.loss_weighted_sum / self.sample_count,
            "normalized_action_mae": self.normalized_absolute_sum / component_count,
            "inverse_normalized_action_mae": self.raw_absolute_sum / component_count,
            "inverse_normalized_action_rmse": math.sqrt(self.raw_squared_sum / component_count),
            "inverse_normalized_first_action_mae": (
                self.first_action_absolute_sum / (self.sample_count * self.action_dimension)
            ),
            "inverse_normalized_mae_by_dimension": (
                self.per_dimension_absolute_sum / self.per_dimension_count
            ).tolist(),
            "inverse_normalized_mae_by_horizon_position": [
                (
                    self.per_horizon_absolute_sum[index] / self.per_horizon_count[index]
                    if self.per_horizon_count[index]
                    else None
                )
                for index in range(self.horizon)
            ],
            "predicted_action_variance_by_dimension": variances.tolist(),
            "constant_action_prediction": bool(np.all(variances <= 1e-12)),
        }


def select_offline_checkpoints(
    reports: Sequence[Mapping[str, Any]],
    *,
    maximum: int = 2,
) -> dict[str, object]:
    """Apply the frozen validation-only checkpoint selection rule."""

    if maximum not in {1, 2} or not reports:
        raise Phase2CContractError("offline selection requires one or two retained checkpoints")
    candidates: list[tuple[float, float, str, Mapping[str, Any]]] = []
    for report in reports:
        if report.get("completed") is not True or report.get("split") != "validation":
            raise Phase2CContractError(
                "checkpoint selection may use completed validation reports only"
            )
        digest = report.get("checkpoint_sha256")
        metrics = report.get("metrics")
        if not isinstance(digest, str) or len(digest) != 64 or not isinstance(metrics, Mapping):
            raise Phase2CContractError("offline checkpoint report identity is malformed")
        loss = float(metrics.get("validation_loss", math.nan))
        first = float(metrics.get("inverse_normalized_first_action_mae", math.nan))
        if not math.isfinite(loss) or not math.isfinite(first):
            raise Phase2CContractError("checkpoint selection metrics must be finite")
        candidates.append((loss, first, digest, report))
    candidates.sort(key=lambda item: item[:3])
    selected = [
        {
            "checkpoint_sha256": digest,
            "validation_loss": loss,
            "inverse_normalized_first_action_mae": first,
            "rank": rank,
        }
        for rank, (loss, first, digest, _) in enumerate(candidates[:maximum], start=1)
    ]
    semantic = {
        "selection_split": "validation",
        "selection_rule": (
            "lowest full validation flow-matching loss; tie-break by inverse-normalized "
            "first-action MAE; retain at most two"
        ),
        "candidate_count": len(candidates),
        "selected": selected,
        "test_outcomes_available": False,
    }
    return {
        "schema_version": "langmani-v2-phase2c-checkpoint-selection-v0",
        **semantic,
        "semantic_sha256": f"sha256:{sha256_hex(semantic)}",
        "passed": True,
    }


__all__ = ["OfflineMetricAccumulator", "select_offline_checkpoints"]
