from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from langmani.v2.phase2c_b import (
    ACTION_LOWER,
    ACTION_UPPER,
    BOUNDED_ACTION_TRANSFORM_FILE,
    BoundedActionLatentV1,
    ModelKind,
    Phase2CBContractError,
    SmolVLATrainingConfig,
    apply_padding_mask,
    build_evaluation_schedule,
    build_language_intervention_manifest,
    build_model_view_manifest,
    build_padding_audit,
    build_static_action_audit,
    classify_pick_gate,
    validate_model_batch,
)
from langmani.v2.phase2c_b_smolvla import BoundedSmolVLAPolicyV1
from langmani.v2.policy import PolicyContext, PolicyContractError


def test_bounded_action_transform_is_finite_and_inside_native_bounds(tmp_path: Path) -> None:
    transform = BoundedActionLatentV1()
    endpoints = torch.tensor([ACTION_LOWER, ACTION_UPPER], dtype=torch.float32)
    latent = transform.encode(endpoints)
    restored = transform.decode(latent)

    assert torch.isfinite(latent).all()
    assert torch.all(restored >= endpoints[0])
    assert torch.all(restored <= endpoints[1])
    assert torch.max(torch.abs(restored - endpoints)).item() < 4e-6
    extreme = transform.decode(
        torch.tensor([[float("-inf")] * 8, [float("inf")] * 8], dtype=torch.float32)
    )
    assert torch.equal(extreme[0], endpoints[0])
    assert torch.equal(extreme[1], endpoints[1])

    path = transform.save(tmp_path)
    assert path.name == BOUNDED_ACTION_TRANSFORM_FILE
    assert BoundedActionLatentV1.load(tmp_path) == transform
    document = json.loads(path.read_text(encoding="utf-8"))
    document["epsilon"] = 1e-5
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(Phase2CBContractError, match="fingerprint"):
        BoundedActionLatentV1.load(tmp_path)


def test_bounded_action_transform_rejects_invalid_physical_inputs() -> None:
    transform = BoundedActionLatentV1()
    invalid = torch.tensor([ACTION_UPPER], dtype=torch.float32)
    invalid[0, 0] += 1e-3
    with pytest.raises(Phase2CBContractError, match="outside"):
        transform.encode(invalid)
    with pytest.raises(Phase2CBContractError, match="without NaN"):
        transform.decode(torch.full((1, 8), float("nan")))


def test_padding_reduction_zeros_padded_and_all_padding_loss() -> None:
    losses = torch.ones((2, 50, 8), dtype=torch.float32)
    padding = torch.zeros((2, 50), dtype=torch.bool)
    padding[0, 10:] = True
    losses[0, 10:] = 10_000
    assert apply_padding_mask(losses, padding).item() == 1.0
    assert apply_padding_mask(losses, torch.ones_like(padding)).item() == 0.0
    audit = build_padding_audit()
    assert audit["passed"] is True
    assert audit["all_padding_finite"] is True


def test_static_action_audit_requires_and_checks_100k_chunks() -> None:
    with pytest.raises(Phase2CBContractError, match="100,000"):
        build_static_action_audit(chunk_count=99_999)
    report = build_static_action_audit(chunk_count=100_000, batch_chunks=10_000)
    assert report["passed"] is True
    assert report["chunk_count"] == 100_000
    assert report["nonfinite_physical_component_count"] == 0
    assert report["lower_bound_violation_count"] == 0
    assert report["upper_bound_violation_count"] == 0
    assert report["clipping_event_count"] == 0
    assert report["projection_event_count"] == 0


def test_model_views_keep_language_and_exclude_task_id_input() -> None:
    pick = build_model_view_manifest(ModelKind.PICK)
    shared = build_model_view_manifest(ModelKind.SHARED)
    assert pick["episode_count"] == 700
    assert pick["task_ids"] == ["PickCube-v1"]
    assert shared["episode_count"] == 2_099
    assert shared["uniform_task_sampling"] is True
    assert shared["task_id_model_input"] is False
    assert sum(shared["uniform_task_probabilities"].values()) == pytest.approx(1.0)


def test_evaluation_schedule_freezes_language_and_final_identities() -> None:
    root = Path(__file__).resolve().parents[2] / "artifacts" / "langmani_v2" / "phase_2b6_v2"
    split = json.loads((root / "accepted_primary_split_manifest.json").read_text())
    inventory = json.loads((root / "source_inventory.json").read_text())
    schedule = build_evaluation_schedule(split, inventory)
    schedules = schedule["schedules"]
    assert len(schedules["validation"]["PickCube-v1"]) == 30
    assert len(schedules["test_unseen_reset"]["StackCube-v1"]) == 50
    assert len(schedules["test_unseen_task_language"]["PushCube-v1"]) == 50
    assert len(schedules["test_visual_shift"]["PickCube-v1"]) == 50
    row = schedules["test_unseen_task_language"]["PickCube-v1"][0]
    assert row["language_instruction"]
    assert row["instruction_template_id"]
    assert row["language_bank"] == "held_out"
    assert schedule["final_settings_mutable_after_results"] is False
    intervention = build_language_intervention_manifest(schedule)
    assert len(intervention["records"]) == 60
    assert intervention["conditions"] == ["correct", "wrong_skill", "blank", "shuffled"]
    assert intervention["records"][0]["blank_instruction"] == ""


def test_exact_policy_batch_allowlist_requires_natural_language() -> None:
    batch = {
        "observation.images.base_camera": torch.zeros((2, 3, 256, 256)),
        "observation.state": torch.zeros((2, 9)),
        "action": torch.zeros((2, 50, 8)),
        "action_is_pad": torch.zeros((2, 50), dtype=torch.bool),
        "task": ["pick the cube", "stack the cubes"],
    }
    validate_model_batch(batch, shared=True)
    with pytest.raises(Phase2CBContractError, match="allowlist"):
        validate_model_batch({**batch, "task_id": ["PickCube-v1"]}, shared=True)
    with pytest.raises(Phase2CBContractError, match="non-empty"):
        validate_model_batch({**batch, "task": ["", "stack the cubes"]}, shared=True)


def test_blank_language_requires_explicit_intervention_flag() -> None:
    with pytest.raises(PolicyContractError, match="intervention flag"):
        PolicyContext(
            evaluation_task=None,
            evaluation_id="blank-probe",
            task_id="PickCube-v1",
            language_instruction="",
        )
    context = PolicyContext(
        evaluation_task=None,
        evaluation_id="blank-probe",
        task_id="PickCube-v1",
        language_instruction="",
        allow_blank_language_instruction=True,
    )
    assert context.language_instruction == ""


def test_primary_training_configuration_is_single_and_frozen() -> None:
    config = SmolVLATrainingConfig()
    assert config.total_steps == 20_000
    assert config.effective_batch_size == 16
    assert config.checkpoint_steps == (5_000, 10_000, 20_000)
    with pytest.raises(Phase2CBContractError, match="configuration changed"):
        SmolVLATrainingConfig(learning_rate=2e-4)


def test_pick_gate_enforces_three_of_thirty_and_result_d_stop() -> None:
    passed = classify_pick_gate(
        success_count=3,
        episode_count=30,
        invalid_action_episode_count=0,
        simulator_error_episode_count=0,
        repair_already_used=False,
    )
    assert passed["passed"] is True
    assert passed["other_full_models_authorized"] is True

    first_zero = classify_pick_gate(
        success_count=0,
        episode_count=30,
        invalid_action_episode_count=0,
        simulator_error_episode_count=0,
        repair_already_used=False,
    )
    assert first_zero["bounded_repair_permitted"] is True
    assert first_zero["result_d_stop"] is False

    final_zero = classify_pick_gate(
        success_count=0,
        episode_count=30,
        invalid_action_episode_count=0,
        simulator_error_episode_count=0,
        repair_already_used=True,
    )
    assert final_zero["result_d_stop"] is True
    assert final_zero["other_full_models_authorized"] is False


def test_bounded_policy_training_target_and_generated_action_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, torch.Tensor] = {}

    def fake_forward(
        _self: object,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[torch.Tensor, dict[str, object]]:
        del noise, time, reduction
        captured["action"] = batch["action"].clone()
        return torch.tensor(1.0), {}

    def fake_predict(
        _self: object,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del batch, noise, kwargs
        return torch.zeros((1, 50, 8), dtype=torch.float32)

    monkeypatch.setattr(
        "lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy.forward",
        fake_forward,
    )
    monkeypatch.setattr(
        "lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy.predict_action_chunk",
        fake_predict,
    )
    policy = object.__new__(BoundedSmolVLAPolicyV1)
    midpoint = 0.5 * (
        torch.tensor(ACTION_LOWER, dtype=torch.float32)
        + torch.tensor(ACTION_UPPER, dtype=torch.float32)
    )
    physical = midpoint.reshape(1, 1, 8).expand(1, 50, 8).clone()
    padding = torch.zeros((1, 50), dtype=torch.bool)
    padding[:, -3:] = True
    loss, _ = policy.forward({"action": physical, "action_is_pad": padding})
    assert loss.item() == 1.0
    assert torch.count_nonzero(captured["action"][:, -3:]) == 0
    assert torch.isfinite(captured["action"]).all()

    generated = policy.predict_action_chunk({})
    assert generated.shape == (1, 50, 8)
    assert torch.allclose(generated[0, 0], midpoint)
