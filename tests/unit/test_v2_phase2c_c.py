from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from langmani.v2.phase2c_b import ACTION_LOWER, ACTION_UPPER
from langmani.v2.phase2c_c import (
    FLOAT32_RECONSTRUCTION_ATOL,
    RELATIVE_ACTION_TRANSFORM_FILE,
    Phase2CCContractError,
    StateRelativeBoundedActionV0,
    build_static_relative_action_audit,
    classify_action_formulation,
    derive_frozen_scales,
    relative_training_config,
)
from langmani.v2.phase2c_c_smolvla import project_relative_policy_batch


def _state_from_reference(reference: torch.Tensor) -> torch.Tensor:
    finger = -0.01 + 0.5 * (reference[..., 7] + 1.0) * 0.05
    return torch.cat((reference[..., :7], finger[..., None].repeat_interleave(2, dim=-1)), dim=-1)


def test_relative_transform_round_trip_zero_identity_bounds_and_reload(tmp_path: Path) -> None:
    transform = StateRelativeBoundedActionV0()
    lower = torch.tensor(ACTION_LOWER, dtype=torch.float32)
    upper = torch.tensor(ACTION_UPPER, dtype=torch.float32)
    reference = 0.35 * lower + 0.65 * upper
    state = _state_from_reference(reference).unsqueeze(0)
    lower_target = reference.clone()
    lower_target[:7] -= 0.01
    lower_target[7] = -1.0
    upper_target = reference.clone()
    upper_target[:7] += 0.01
    upper_target[7] = 1.0
    targets = torch.stack((lower_target, reference, upper_target)).unsqueeze(0)
    latent = transform.encode(targets, state)
    restored = transform.decode(latent, state)

    assert torch.isfinite(latent).all()
    assert torch.max(torch.abs(restored - targets)).item() <= FLOAT32_RECONSTRUCTION_ATOL
    zero = transform.decode(torch.zeros_like(latent), state)
    assert torch.max(torch.abs(zero - reference)).item() <= FLOAT32_RECONSTRUCTION_ATOL

    extreme = transform.decode(
        torch.tensor(
            [[[float("-inf")] * 8, [float("inf")] * 8]],
            dtype=torch.float32,
        ),
        state,
    )
    assert torch.all(extreme >= lower)
    assert torch.all(extreme <= upper)

    path = transform.save(tmp_path)
    assert path.name == RELATIVE_ACTION_TRANSFORM_FILE
    assert StateRelativeBoundedActionV0.load(tmp_path) == transform
    document = json.loads(path.read_text(encoding="utf-8"))
    document["arm_residual_scale"] = 1.0
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(Phase2CCContractError, match="identity"):
        StateRelativeBoundedActionV0.load(tmp_path)


def test_physical_delta_view_uses_measured_mimic_mean_and_no_future_state() -> None:
    transform = StateRelativeBoundedActionV0()
    reference = torch.tensor(
        [[0.1, 0.2, -0.1, -1.5, 0.05, 2.0, 0.3, 0.25]],
        dtype=torch.float32,
    )
    state = _state_from_reference(reference)
    state[0, 7] -= 1e-4
    state[0, 8] += 1e-4
    targets = reference[:, None, :].repeat(1, 50, 1)
    targets[:, :, 0] += 0.01
    residual = transform.physical_residual(targets, state)
    assert residual.shape == (1, 50, 8)
    assert torch.allclose(residual[..., 0], torch.full((1, 50), 0.01), atol=1e-7)
    assert torch.max(torch.abs(residual[..., 1:])).item() < 2e-6
    restored = transform.reconstruct_from_physical_residual(residual, state)
    assert torch.allclose(restored, targets, atol=2e-7, rtol=0.0)
    assert transform.semantic_dict()["future_state_used"] is False


def test_relative_transform_fails_closed_on_ambiguous_or_out_of_scale_inputs() -> None:
    transform = StateRelativeBoundedActionV0()
    state = torch.zeros((1, 9), dtype=torch.float32)
    state[:, 7:] = 0.2
    with pytest.raises(Phase2CCContractError, match="unambiguously"):
        transform.current_action_reference(state)

    lower = torch.tensor(ACTION_LOWER, dtype=torch.float32)
    upper = torch.tensor(ACTION_UPPER, dtype=torch.float32)
    state = _state_from_reference(upper).unsqueeze(0)
    with pytest.raises(Phase2CCContractError, match="residual scale"):
        transform.encode(lower.reshape(1, 1, 8), state)


def test_train_scale_recomputation_and_training_config_are_frozen() -> None:
    lower = torch.tensor(ACTION_LOWER, dtype=torch.float32)
    upper = torch.tensor(ACTION_UPPER, dtype=torch.float32)
    reference = 0.5 * (lower + upper)
    state = _state_from_reference(reference).repeat(4, 1)
    targets = reference.repeat(4, 1)
    audit = derive_frozen_scales(targets, state)
    assert audit["exact_scale_match"] is False
    config = relative_training_config()
    assert config["total_optimizer_steps"] == 20_000
    assert config["training_seed_count"] == 1
    assert config["vla_jepa_authorized"] is False


def test_static_relative_action_audit_checks_100k_chunks() -> None:
    with pytest.raises(Phase2CCContractError, match="100,000"):
        build_static_relative_action_audit(chunk_count=99_999)
    report = build_static_relative_action_audit(chunk_count=100_000, batch_chunks=5_000)
    assert report["passed"] is True
    assert report["chunk_count"] == 100_000
    assert report["nonfinite_component_count"] == 0
    assert report["lower_bound_violation_count"] == 0
    assert report["upper_bound_violation_count"] == 0
    assert report["clipping_event_count"] == 0
    assert report["projection_event_count"] == 0


def test_relative_batch_excludes_padding_before_bounded_encoding() -> None:
    reference = torch.tensor(
        [[0.1, 0.2, -0.1, -1.5, 0.05, 2.0, 0.3, 0.25]],
        dtype=torch.float32,
    )
    state = _state_from_reference(reference)
    actions = reference[:, None, :].repeat(1, 50, 1)
    padding = torch.zeros((1, 50), dtype=torch.bool)
    padding[:, -1] = True
    actions[:, -1, :7] = torch.tensor(ACTION_LOWER[:7], dtype=torch.float32)
    actions[:, -1, 7] = -1.0
    batch = project_relative_policy_batch(
        {
            "observation.images.base_camera": torch.zeros((1, 3, 256, 256), dtype=torch.uint8),
            "observation.state": state,
            "action": actions,
            "action_is_pad": padding,
            "task": ["Pick up the cube."],
        }
    )
    latent = batch["action"]
    assert isinstance(latent, torch.Tensor)
    assert torch.max(torch.abs(latent[:, :-1])).item() <= 1e-6
    assert torch.equal(latent[:, -1], torch.zeros((1, 8), dtype=torch.float32))


@pytest.mark.parametrize(
    ("validation", "training", "pipeline_valid", "expected"),
    [
        (3, 0, True, "CASE_A"),
        (0, 1, True, "CASE_B"),
        (0, 0, True, "CASE_C"),
        (30, 30, False, "CASE_D"),
    ],
)
def test_phase2c_c_decision_matrix(
    validation: int,
    training: int,
    pipeline_valid: bool,
    expected: str,
) -> None:
    result = classify_action_formulation(
        validation_success_count=validation,
        training_reset_success_count=training,
        pipeline_valid=pipeline_valid,
    )
    assert result["case"] == expected
    assert result["passed"] is True


def test_phase2c_c_decision_matrix_rejects_unregistered_gap() -> None:
    with pytest.raises(Phase2CCContractError, match="does not classify"):
        classify_action_formulation(
            validation_success_count=1,
            training_reset_success_count=0,
            pipeline_valid=True,
        )
