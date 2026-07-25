"""Validation-only checkpoint screening for Phase 2C-A.1.

Contract and degeneration checks are hard filters.  Eligible checkpoints are
ranked using several validation diagnostics so selection is not based on one
scalar loss alone.  Final test identities are never inputs to this module.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Final

from langmani.datasets.identity import sha256_hex

SCREEN_SCHEMA: Final = "langmani-v2-phase2c-a1-checkpoint-screen-v0"
SELECTION_SCHEMA: Final = "langmani-v2-phase2c-a1-checkpoint-selection-v0"
MINIMUM_POLICY_QUERIES: Final = 10_000
RANK_FIELDS: Final = (
    "total_action_loss",
    "unpadded_raw_l1",
    "first_action_raw_l1",
    "gripper_raw_l1",
)


class Phase2CA1CheckpointScreenError(RuntimeError):
    """Raised when checkpoint-screen evidence is incomplete or inconsistent."""


def _finite_number(report: Mapping[str, object], key: str) -> float:
    value = report.get(key)
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise Phase2CA1CheckpointScreenError(f"checkpoint screen lacks finite {key}")
    return float(value)


def _contract_failures(report: Mapping[str, object]) -> list[str]:
    failures: list[str] = []
    if report.get("schema_version") != SCREEN_SCHEMA:
        failures.append("wrong_screen_schema")
    if report.get("split") != "validation":
        failures.append("non_validation_input")
    if report.get("bounded_action_head_v1") is not True:
        failures.append("bounded_head_not_verified")
    if report.get("checkpoint_reconstruction_passed") is not True:
        failures.append("checkpoint_reconstruction_failed")
    if report.get("processor_reconstruction_passed") is not True:
        failures.append("processor_reconstruction_failed")
    if report.get("output_finite") is not True:
        failures.append("nonfinite_output")
    if report.get("padding_excluded_from_metrics") is not True:
        failures.append("padding_mask_not_verified")
    if report.get("clipping_events") != 0:
        failures.append("clipping_detected")
    if report.get("projection_events") != 0:
        failures.append("projection_detected")
    if report.get("lower_bound_violations") != 0:
        failures.append("lower_bound_violation")
    if report.get("upper_bound_violations") != 0:
        failures.append("upper_bound_violation")
    queries = report.get("policy_query_count")
    if not isinstance(queries, int) or queries < MINIMUM_POLICY_QUERIES:
        failures.append("insufficient_policy_queries")
    checkpoint = report.get("checkpoint_fingerprint")
    if not isinstance(checkpoint, str) or not checkpoint.startswith("sha256:"):
        failures.append("invalid_checkpoint_identity")
    components = report.get("checkpoint_components")
    if not isinstance(components, Mapping) or any(
        not isinstance(components.get(key), str) or not str(components[key]).startswith("sha256:")
        for key in ("model", "preprocessor", "postprocessor")
    ):
        failures.append("invalid_checkpoint_components")
    for key in RANK_FIELDS:
        try:
            _finite_number(report, key)
        except Phase2CA1CheckpointScreenError:
            failures.append(f"invalid_{key}")
    return failures


def _degeneration_failures(report: Mapping[str, object]) -> list[str]:
    failures: list[str] = []
    if report.get("constant_action_prediction") is True:
        failures.append("constant_action_prediction")
    variance = report.get("predicted_action_variance_by_dimension")
    if (
        not isinstance(variance, list)
        or len(variance) != 8
        or not all(
            isinstance(value, int | float) and math.isfinite(float(value)) and float(value) >= 0.0
            for value in variance
        )
    ):
        failures.append("invalid_output_variance")
    return failures


def select_phase2c_a1_checkpoints(
    reports: Sequence[Mapping[str, object]],
    *,
    maximum: int = 2,
) -> dict[str, object]:
    """Hard-filter screens, then rank eligible checkpoints by four diagnostics."""

    if maximum not in {1, 2} or not reports:
        raise Phase2CA1CheckpointScreenError("selection needs reports and maximum 1 or 2")
    model_kinds = {report.get("model_kind") for report in reports}
    if len(model_kinds) != 1 or not isinstance(next(iter(model_kinds)), str):
        raise Phase2CA1CheckpointScreenError("screens must belong to one model kind")
    fingerprints = [report.get("checkpoint_fingerprint") for report in reports]
    if len(fingerprints) != len(set(fingerprints)):
        raise Phase2CA1CheckpointScreenError("checkpoint screens contain duplicate identities")

    rejected: list[dict[str, object]] = []
    eligible: list[Mapping[str, object]] = []
    for report in reports:
        contract = _contract_failures(report)
        degeneration = _degeneration_failures(report)
        if contract or degeneration or report.get("passed") is not True:
            rejected.append(
                {
                    "checkpoint_fingerprint": report.get("checkpoint_fingerprint"),
                    "contract_failures": contract,
                    "degeneration_failures": degeneration,
                }
            )
        else:
            eligible.append(report)
    if not eligible:
        raise Phase2CA1CheckpointScreenError("no checkpoint passed contract screening")

    ordinal_by_field: dict[str, dict[str, int]] = {}
    for field in RANK_FIELDS:
        ordered = sorted(
            eligible,
            key=lambda item: (
                _finite_number(item, field),
                str(item["checkpoint_fingerprint"]),
            ),
        )
        ordinal_by_field[field] = {
            str(item["checkpoint_fingerprint"]): rank for rank, item in enumerate(ordered, start=1)
        }
    ranked = sorted(
        eligible,
        key=lambda item: (
            sum(
                ordinal_by_field[field][str(item["checkpoint_fingerprint"])]
                for field in RANK_FIELDS
            ),
            _finite_number(item, "first_action_raw_l1"),
            str(item["checkpoint_fingerprint"]),
        ),
    )
    selected = []
    for rank, report in enumerate(ranked[:maximum], start=1):
        fingerprint = str(report["checkpoint_fingerprint"])
        field_ranks = {field: ordinal_by_field[field][fingerprint] for field in RANK_FIELDS}
        selected.append(
            {
                "rank": rank,
                "checkpoint_fingerprint": fingerprint,
                "run_fingerprint": report.get("run_fingerprint"),
                "checkpoint_relative_path": report.get("checkpoint_relative_path"),
                "checkpoint_step": report.get("checkpoint_step"),
                "checkpoint_components": report.get("checkpoint_components"),
                "rank_sum": sum(field_ranks.values()),
                "field_ranks": field_ranks,
                "metrics": {field: _finite_number(report, field) for field in RANK_FIELDS},
            }
        )
    semantic = {
        "schema_version": SELECTION_SCHEMA,
        "model_kind": next(iter(model_kinds)),
        "selection_split": "validation",
        "minimum_policy_queries_per_checkpoint": MINIMUM_POLICY_QUERIES,
        "candidate_count": len(reports),
        "eligible_count": len(eligible),
        "maximum_selected": maximum,
        "hard_filters": ["contract_failure", "degeneration_failure"],
        "ranking_rule": (
            "sum ordinal ranks over total loss, unpadded physical L1, "
            "first-action physical L1, and gripper physical L1; "
            "tie-break by first-action L1 then checkpoint fingerprint"
        ),
        "ranking_fields": list(RANK_FIELDS),
        "selected": selected,
        "rejected": rejected,
        "test_outcomes_available": False,
        "passed": True,
    }
    return {
        **semantic,
        "fingerprint": f"sha256:{sha256_hex(semantic)}",
    }


__all__ = [
    "MINIMUM_POLICY_QUERIES",
    "Phase2CA1CheckpointScreenError",
    "select_phase2c_a1_checkpoints",
]
