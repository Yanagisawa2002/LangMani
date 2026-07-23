from __future__ import annotations

from pathlib import Path

import pytest
import torch

from langmani.v2.phase2b4_ppo import (
    FORMAL_QUALIFICATION_SEEDS,
    FUTURE_STUDENT_OBSERVATION_FIELDS,
    TEACHER_OBSERVATION_DIM,
    TEACHER_OBSERVATION_FIELDS,
    BoundedActorCritic,
    EvaluationSummary,
    Phase2B4Result,
    PPOConfig,
    RewardInputs,
    assert_f0_operation_allowed,
    assert_f0_seed_allowed,
    assert_student_observation_has_no_privilege,
    audit_reward_hacking_scenarios,
    audit_sampled_action_legality,
    authorization_state,
    checkpoint_payload,
    classify_phase2b4_result,
    compute_gae,
    compute_phase2b4_reward,
    compute_ppo_loss,
    concatenate_teacher_fields,
    feasibility_gate,
    micro_gate,
    normalization_manifest,
    normalize_teacher_observation,
    reconstruct_policy_from_checkpoint,
    reward_manifest,
    save_checkpoint,
    seed_disjointness_audit,
    student_privilege_exclusion_audit,
    teacher_observation_manifest,
    verify_source_identity,
)


def _action_bounds() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor([-2.9] * 7 + [-1.0]), torch.tensor([2.9] * 7 + [1.0])


def _policy() -> BoundedActorCritic:
    low, high = _action_bounds()
    return BoundedActorCritic(
        observation_dim=TEACHER_OBSERVATION_DIM,
        action_low=low,
        action_high=high,
        hidden_dim=32,
    )


def _summary(
    *,
    episodes: int,
    successes: int,
    cube: tuple[int, int],
    cylinder: tuple[int, int],
    directions: dict[str, tuple[int, int]],
    standard: tuple[int, int],
    hard: tuple[int, int],
    safety: int = 0,
) -> EvaluationSummary:
    return EvaluationSummary(
        episode_count=episodes,
        successes=successes,
        cube_episodes=cube[0],
        cube_successes=cube[1],
        cylinder_episodes=cylinder[0],
        cylinder_successes=cylinder[1],
        direction_episodes={key: value[0] for key, value in directions.items()},
        direction_successes={key: value[1] for key, value in directions.items()},
        standard_episodes=standard[0],
        standard_successes=standard[1],
        hard_episodes=hard[0],
        hard_successes=hard[1],
        safety_counts={"wrong_object_contact": safety},
    )


def test_source_identity_and_geometry_exclusion_are_enforced() -> None:
    verify_source_identity(
        parent_sha="3ded856bc0fba7cd749f1fb0caf738fd7fce5751",
        geometry_is_ancestor=False,
    )
    with pytest.raises(ValueError, match="must descend directly"):
        verify_source_identity(parent_sha="bad", geometry_is_ancestor=False)
    with pytest.raises(ValueError, match="excluded geometry"):
        verify_source_identity(
            parent_sha="3ded856bc0fba7cd749f1fb0caf738fd7fce5751",
            geometry_is_ancestor=True,
        )


def test_formal_seed_is_sealed_and_namespaces_are_disjoint() -> None:
    audit = seed_disjointness_audit()
    assert audit["passed"] is True
    assert audit["formal_seed_accessed"] is False
    assert frozenset(range(66300, 66400)) == FORMAL_QUALIFICATION_SEEDS
    with pytest.raises(ValueError, match="formal qualification seed"):
        assert_f0_seed_allowed(66300, "ppo_train")
    assert_f0_seed_allowed(700000, "ppo_train")
    assert_f0_seed_allowed(800000, "ppo_micro_evaluation")
    assert_f0_seed_allowed(810000, "ppo_feasibility_development")
    with pytest.raises(ValueError, match="not accessible"):
        assert_f0_seed_allowed(820000, "future_ppo_full_development")


@pytest.mark.parametrize(
    "operation",
    [
        "demonstration_collection",
        "lerobot_export",
        "student_training",
        "smolvla_training",
        "ppo_full_training",
        "expert_qualification",
    ],
)
def test_data_collection_and_later_training_stages_are_prohibited(operation: str) -> None:
    with pytest.raises(ValueError, match="explicitly prohibited"):
        assert_f0_operation_allowed(operation)
    assert_f0_operation_allowed("ppo_micro_training")


def test_teacher_and_future_student_observation_contracts_are_separate() -> None:
    manifest = teacher_observation_manifest()
    assert manifest["dimension"] == 87 == TEACHER_OBSERVATION_DIM
    assert manifest["current_state_only"] is True
    assert student_privilege_exclusion_audit()["passed"] is True
    assert FUTURE_STUDENT_OBSERVATION_FIELDS == (
        "observation.images.base_camera",
        "observation.state.panda_qpos",
    )
    assert_student_observation_has_no_privilege(FUTURE_STUDENT_OBSERVATION_FIELDS)
    with pytest.raises(ValueError, match="privileged teacher fields leaked"):
        assert_student_observation_has_no_privilege(("observation.state.target_to_goal",))


def test_teacher_observation_concatenation_and_normalization_are_finite() -> None:
    values = {
        field.name: torch.full((3, field.width), 0.25, dtype=torch.float32)
        for field in TEACHER_OBSERVATION_FIELDS
    }
    observation = concatenate_teacher_fields(values)
    normalized = normalize_teacher_observation(observation)
    assert observation.shape == normalized.shape == (3, TEACHER_OBSERVATION_DIM)
    assert torch.isfinite(normalized).all()
    assert normalization_manifest()["method"] == "fixed_componentwise_scale_without_clipping"
    values["panda_qpos"][0, 0] = torch.nan
    with pytest.raises(ValueError, match="non-finite"):
        concatenate_teacher_fields(values)


def test_bounded_transform_uses_one_path_for_stochastic_and_deterministic_actions() -> None:
    policy = _policy()
    low, high = _action_bounds()
    observation = torch.zeros((16, TEACHER_OBSERVATION_DIM))
    action, latent, log_prob, entropy, value = policy.action_and_value(observation)
    deterministic = policy.deterministic_action(observation)
    assert action.shape == deterministic.shape == (16, 8)
    assert log_prob.shape == entropy.shape == value.shape == (16,)
    assert torch.equal(action, policy.action_from_latent(latent))
    assert torch.all(action >= low) and torch.all(action <= high)
    assert torch.all(deterministic >= low) and torch.all(deterministic <= high)
    manifest = policy.action_manifest()
    assert manifest["clipping"] is False
    assert manifest["projection"] is False


def test_static_100000_action_legality_audit() -> None:
    audit = audit_sampled_action_legality(_policy(), sample_count=100_000, batch_size=10_000)
    assert audit["sample_count"] == 100_000
    assert audit["nonfinite_actions"] == 0
    assert audit["out_of_bounds_actions"] == 0
    assert audit["hidden_clipping_events"] == 0
    assert audit["hidden_projection_events"] == 0
    assert audit["passed"] is True


def _reward_inputs(*, success: bool = False, wrong_object: bool = False) -> RewardInputs:
    scalar = lambda value: torch.tensor([value], dtype=torch.float32)  # noqa: E731
    flag = lambda value: torch.tensor([value], dtype=torch.bool)  # noqa: E731
    zeros = torch.zeros((1, 8), dtype=torch.float32)
    return RewardInputs(
        previous_target_distance=scalar(0.4),
        target_distance=scalar(0.3),
        previous_behind_distance=scalar(0.2),
        behind_distance=scalar(0.15),
        previous_containment_fraction=scalar(0.0),
        containment_fraction=scalar(0.2),
        target_contact_onset=flag(True),
        push_alignment=scalar(1.0),
        wrong_object_contact=flag(wrong_object),
        wrong_object_displaced=flag(wrong_object),
        target_outside_workspace=flag(False),
        target_lifted=flag(False),
        target_toppled=flag(False),
        target_is_grasped=flag(False),
        invalid_action=flag(False),
        action_out_of_bounds=flag(False),
        robot_collision=flag(False),
        success=flag(success),
        normalized_action=zeros,
        previous_normalized_action=zeros,
    )


def test_reward_components_terminal_bonus_and_safety_penalty() -> None:
    ordinary = compute_phase2b4_reward(_reward_inputs())
    success = compute_phase2b4_reward(_reward_inputs(success=True))
    unsafe = compute_phase2b4_reward(_reward_inputs(wrong_object=True))
    assert success.total.item() == pytest.approx(ordinary.total.item() + 10.0)
    assert unsafe.total.item() == pytest.approx(ordinary.total.item() - 20.0)
    assert reward_manifest()["reward_revision_used"] == 0


def test_reward_hacking_scenarios_pass_before_training() -> None:
    audit = audit_reward_hacking_scenarios()
    assert audit["passed"] is True
    assert audit["reward_revision_used"] == 0
    assert audit["scenarios"]["canonical_success"] > audit["scenarios"]["brief_unstable_target"]
    assert audit["scenarios"]["oscillating"] < 0


def test_gae_bootstraps_truncation_but_not_termination() -> None:
    rewards = torch.tensor([[1.0, 1.0]])
    values = torch.tensor([[0.5, 0.5]])
    next_values = torch.tensor([[2.0, 2.0]])
    terminated = torch.tensor([[True, False]])
    truncated = torch.tensor([[False, True]])
    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        next_values=next_values,
        terminated=terminated,
        truncated=truncated,
        gamma=0.9,
        gae_lambda=0.95,
    )
    assert advantages.tolist()[0] == pytest.approx([0.5, 2.3])
    assert returns.tolist()[0] == pytest.approx([1.0, 2.8])


def test_ppo_loss_is_finite_and_backpropagates() -> None:
    new_log_prob = torch.tensor([0.0, -0.2], requires_grad=True)
    loss = compute_ppo_loss(
        new_log_prob=new_log_prob,
        old_log_prob=torch.tensor([-0.1, -0.1]),
        entropy=torch.tensor([1.0, 1.0]),
        new_value=torch.tensor([0.2, 0.4], requires_grad=True),
        old_value=torch.tensor([0.1, 0.3]),
        returns=torch.tensor([1.0, -0.5]),
        advantages=torch.tensor([1.0, -1.0]),
        clip_coef=0.2,
        value_coef=0.5,
        entropy_coef=0.001,
    )
    assert all(
        torch.isfinite(value) for value in (loss.total, loss.actor, loss.critic, loss.entropy)
    )
    loss.total.backward()
    assert torch.isfinite(new_log_prob.grad).all()


def test_checkpoint_round_trip_reproduces_deterministic_action(tmp_path: Path) -> None:
    policy = _policy()
    optimizer = torch.optim.Adam(policy.parameters())
    config = PPOConfig(
        num_envs=2,
        num_steps=2,
        total_timesteps=4,
        num_minibatches=1,
        update_epochs=1,
        hidden_dim=32,
    )
    checkpoint = tmp_path / "checkpoint.pt"
    digest = save_checkpoint(
        checkpoint,
        checkpoint_payload(
            policy=policy,
            optimizer=optimizer,
            config=config,
            global_step=4,
            observation_manifest_fingerprint="sha256:observation",
            action_manifest_fingerprint="sha256:action",
            reward_manifest_fingerprint="sha256:reward",
        ),
    )
    loaded, payload = reconstruct_policy_from_checkpoint(checkpoint)
    observation = torch.zeros((1, TEACHER_OBSERVATION_DIM))
    assert digest.startswith("sha256:")
    assert payload["global_step"] == 4
    assert torch.equal(
        policy.deterministic_action(observation),
        loaded.deterministic_action(observation),
    )


def test_micro_and_feasibility_gates_are_conjunctive() -> None:
    micro = _summary(
        episodes=48,
        successes=36,
        cube=(24, 18),
        cylinder=(24, 18),
        directions={"left": (24, 18), "forward_right": (24, 18)},
        standard=(48, 36),
        hard=(0, 0),
    )
    assert all(micro_gate(micro).values())
    unsafe_micro = _summary(
        episodes=48,
        successes=36,
        cube=(24, 18),
        cylinder=(24, 18),
        directions={"left": (24, 18), "forward_right": (24, 18)},
        standard=(48, 36),
        hard=(0, 0),
        safety=1,
    )
    assert micro_gate(unsafe_micro)["zero_tolerance_safety_clean"] is False
    development = _summary(
        episodes=60,
        successes=33,
        cube=(32, 21),
        cylinder=(28, 12),
        directions={
            "left": (15, 8),
            "right": (15, 8),
            "forward_left": (15, 8),
            "forward_right": (15, 9),
        },
        standard=(36, 23),
        hard=(24, 10),
    )
    assert all(feasibility_gate(development).values())


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        (
            {
                "pipeline_valid": False,
                "reward_valid": True,
                "action_valid": True,
                "micro_passed": False,
                "feasibility_development_ran": False,
                "feasibility_development_passed": False,
            },
            Phase2B4Result.RESULT_D,
        ),
        (
            {
                "pipeline_valid": True,
                "reward_valid": True,
                "action_valid": True,
                "micro_passed": False,
                "feasibility_development_ran": False,
                "feasibility_development_passed": False,
            },
            Phase2B4Result.RESULT_C,
        ),
        (
            {
                "pipeline_valid": True,
                "reward_valid": True,
                "action_valid": True,
                "micro_passed": True,
                "feasibility_development_ran": True,
                "feasibility_development_passed": False,
            },
            Phase2B4Result.RESULT_B,
        ),
        (
            {
                "pipeline_valid": True,
                "reward_valid": True,
                "action_valid": True,
                "micro_passed": True,
                "feasibility_development_ran": True,
                "feasibility_development_passed": True,
            },
            Phase2B4Result.RESULT_A,
        ),
    ],
)
def test_result_classification_and_authorization_are_fail_closed(
    inputs: dict[str, bool],
    expected: Phase2B4Result,
) -> None:
    result = classify_phase2b4_result(**inputs)
    state = authorization_state(result)
    assert result is expected
    assert state["ppo_full_training_eligible"] is (expected is Phase2B4Result.RESULT_A)
    assert state["ppo_full_training_authorized"] is False
    assert state["expert_qualification_authorized"] is False
    assert state["data_collection_authorized"] is False
    assert state["smolvla_training_authorized"] is False
    assert state["training_started_for_student_policy"] is False
    assert state["demonstration_source_validated"] is False
