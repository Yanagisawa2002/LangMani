"""Phase 2C-A.1 validation-only checkpoint-selection contracts."""

from __future__ import annotations

import pytest

from langmani.v2.phase2c_a1_checkpoint_screen import (
    Phase2CA1CheckpointScreenError,
    select_phase2c_a1_checkpoints,
)


def _screen(
    index: int,
    *,
    loss: float,
    raw_l1: float,
    first_l1: float,
    gripper_l1: float,
) -> dict[str, object]:
    return {
        "schema_version": "langmani-v2-phase2c-a1-checkpoint-screen-v0",
        "model_kind": "pick_act",
        "split": "validation",
        "bounded_action_head_v1": True,
        "checkpoint_reconstruction_passed": True,
        "processor_reconstruction_passed": True,
        "output_finite": True,
        "padding_excluded_from_metrics": True,
        "clipping_events": 0,
        "projection_events": 0,
        "lower_bound_violations": 0,
        "upper_bound_violations": 0,
        "policy_query_count": 10_000,
        "checkpoint_fingerprint": f"sha256:{index:064x}",
        "checkpoint_relative_path": f"checkpoints/step_{index:08d}",
        "checkpoint_step": index,
        "checkpoint_components": {
            "model": "sha256:" + "a" * 64,
            "preprocessor": "sha256:" + "b" * 64,
            "postprocessor": "sha256:" + "c" * 64,
        },
        "total_action_loss": loss,
        "unpadded_raw_l1": raw_l1,
        "first_action_raw_l1": first_l1,
        "gripper_raw_l1": gripper_l1,
        "constant_action_prediction": False,
        "predicted_action_variance_by_dimension": [0.1] * 8,
        "passed": True,
    }


def test_selection_uses_multiple_validation_diagnostics() -> None:
    reports = [
        _screen(1, loss=0.1, raw_l1=0.5, first_l1=0.5, gripper_l1=0.5),
        _screen(2, loss=0.2, raw_l1=0.1, first_l1=0.1, gripper_l1=0.1),
        _screen(3, loss=0.3, raw_l1=0.2, first_l1=0.2, gripper_l1=0.2),
    ]
    selected = select_phase2c_a1_checkpoints(reports, maximum=2)
    identities = [
        item["checkpoint_fingerprint"]
        for item in selected["selected"]  # type: ignore[index]
    ]
    assert identities == [
        reports[1]["checkpoint_fingerprint"],
        reports[2]["checkpoint_fingerprint"],
    ]
    assert selected["test_outcomes_available"] is False
    assert selected["passed"] is True


def test_contract_and_degeneration_are_the_only_hard_filters() -> None:
    healthy = _screen(1, loss=0.2, raw_l1=0.2, first_l1=0.2, gripper_l1=0.2)
    weak_but_valid = _screen(2, loss=99.0, raw_l1=99.0, first_l1=99.0, gripper_l1=99.0)
    out_of_bounds = _screen(3, loss=0.1, raw_l1=0.1, first_l1=0.1, gripper_l1=0.1)
    out_of_bounds["upper_bound_violations"] = 1
    constant = _screen(4, loss=0.1, raw_l1=0.1, first_l1=0.1, gripper_l1=0.1)
    constant["constant_action_prediction"] = True
    selected = select_phase2c_a1_checkpoints(
        [healthy, weak_but_valid, out_of_bounds, constant],
        maximum=2,
    )
    assert selected["eligible_count"] == 2
    rejected = selected["rejected"]
    assert isinstance(rejected, list)
    assert rejected[0]["contract_failures"] == ["upper_bound_violation"]
    assert rejected[1]["degeneration_failures"] == ["constant_action_prediction"]


def test_selection_rejects_duplicate_or_cross_model_evidence() -> None:
    first = _screen(1, loss=0.1, raw_l1=0.1, first_l1=0.1, gripper_l1=0.1)
    with pytest.raises(Phase2CA1CheckpointScreenError, match="duplicate"):
        select_phase2c_a1_checkpoints([first, dict(first)])
    other = _screen(2, loss=0.2, raw_l1=0.2, first_l1=0.2, gripper_l1=0.2)
    other["model_kind"] = "stack_act"
    with pytest.raises(Phase2CA1CheckpointScreenError, match="one model kind"):
        select_phase2c_a1_checkpoints([first, other])
