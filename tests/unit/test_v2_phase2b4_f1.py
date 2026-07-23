from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from langmani.v2.phase2b4_f1 import (
    EXCLUDED_GEOMETRY_COMMIT,
    FORMAL_QUALIFICATION_SEEDS,
    Phase2B4F1Result,
    ProbeEvaluation,
    StateCenteredResidualActorCritic,
    assert_f1_operation_allowed,
    assert_f1_seed_allowed,
    audit_residual_action_legality,
    authorization_state,
    classify_f1_result,
    current_action_reference,
    curriculum_manifest,
    probe_a_gate,
    probe_b_gate,
    probe_configuration,
    reconstruct_f0_learning_dynamics,
    repository_operation_boundary,
    safe_atanh,
    seed_disjointness_audit,
    termination_bootstrap_audit,
)
from langmani.v2.phase2b4_ppo import PANDA_QPOS_SLICE, TEACHER_OBSERVATION_DIM

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _action_bounds() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor([-2.9] * 7 + [-1.0]), torch.tensor([2.9] * 7 + [1.0])


def _observations(count: int = 16) -> torch.Tensor:
    observation = torch.zeros((count, TEACHER_OBSERVATION_DIM), dtype=torch.float32)
    qpos = observation[:, PANDA_QPOS_SLICE]
    qpos[:, :7] = torch.linspace(-0.3, 0.3, 7)
    qpos[:, 7:9] = 0.02
    return observation


def _policy(*, hidden_dim: int = 32) -> StateCenteredResidualActorCritic:
    low, high = _action_bounds()
    return StateCenteredResidualActorCritic(
        observation_dim=TEACHER_OBSERVATION_DIM,
        action_low=low,
        action_high=high,
        hidden_dim=hidden_dim,
    )


def _evaluation(
    *,
    episodes: int,
    successes: int = 0,
    contact: int = 0,
    progress: int = 0,
    direction_successes: dict[str, int] | None = None,
    wrong_object: int = 0,
    workspace_exit: int = 0,
) -> ProbeEvaluation:
    return ProbeEvaluation(
        episode_count=episodes,
        successes=successes,
        useful_approach_episodes=contact,
        correct_contact_episodes=contact,
        target_progress_episodes=progress,
        direction_successes=direction_successes or {},
        wrong_object_interactions=wrong_object,
        workspace_exits=workspace_exit,
        invalid_actions=0,
        nonfinite_actions=0,
        out_of_bounds_actions=0,
    )


def test_f1_source_boundary_and_f0_evidence_remain_immutable() -> None:
    boundary = repository_operation_boundary()
    assert boundary["excluded_geometry_commit"] == EXCLUDED_GEOMETRY_COMMIT
    assert boundary["formal_seeds_accessed"] is False
    f0_root = PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2b4_f0"
    manifest = json.loads((f0_root / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["result"] == "RESULT_C"
    assert (f0_root / "micro_training_metrics.json").is_file()


def test_formal_seed_prevention_and_all_namespaces_are_disjoint() -> None:
    audit = seed_disjointness_audit()
    assert audit["passed"] is True
    assert audit["overlaps"] == []
    assert audit["formal_seed_accessed"] is False
    assert frozenset(range(66300, 66400)) == FORMAL_QUALIFICATION_SEEDS
    with pytest.raises(PermissionError, match="formal qualification seeds"):
        assert_f1_seed_allowed(66300, "f1_probe_train")
    assert_f1_seed_allowed(1100000, "f1_probe_train")
    assert_f1_seed_allowed(1200000, "f1_probe_a_evaluation")
    assert_f1_seed_allowed(1201000, "f1_probe_b_evaluation")
    with pytest.raises(PermissionError, match="sealed"):
        assert_f1_seed_allowed(1210000, "future_full_curriculum")


@pytest.mark.parametrize(
    "operation",
    [
        "ppo_full_training",
        "stage_1_training",
        "stage_2_training",
        "formal_qualification",
        "demonstration_collection",
        "lerobot_export",
        "student_training",
        "smolvla_training",
        "act_training",
        "vla_jepa_training",
        "simulator_mpc",
        "motion_planning",
    ],
)
def test_prohibited_operations_fail_closed(operation: str) -> None:
    with pytest.raises(PermissionError):
        assert_f1_operation_allowed(operation)
    assert_f1_operation_allowed("probe_a_training")


def test_current_joint_conversion_and_zero_residual_identity() -> None:
    low, high = _action_bounds()
    observations = _observations()
    reference = current_action_reference(
        observations,
        action_low=low,
        action_high=high,
    )
    assert torch.equal(reference[:, :7], observations[:, PANDA_QPOS_SLICE][:, :7])
    assert torch.equal(reference[:, 7], torch.zeros(len(observations)))
    policy = _policy()
    zero_action = policy.action_from_latent(observations, torch.zeros((len(observations), 8)))
    assert torch.allclose(zero_action, reference, atol=1e-6, rtol=0.0)


def test_safe_inverse_tanh_is_finite_and_has_a_guard() -> None:
    values = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    output = safe_atanh(values)
    assert torch.isfinite(output).all()
    assert torch.tanh(output[1:4]).tolist() == pytest.approx(values[1:4].tolist())
    with pytest.raises(ValueError, match="finite"):
        safe_atanh(torch.tensor([float("nan")]))


def test_residual_transform_and_log_probability_are_reproducible() -> None:
    policy = _policy()
    observations = _observations()
    distribution = policy.distribution(observations)
    latent = distribution.rsample()
    action, selected, log_prob, entropy, value = policy.action_and_value(
        observations,
        latent=latent,
    )
    repeated = policy.log_prob(distribution, observations, latent)
    assert torch.equal(selected, latent)
    assert torch.equal(action, policy.action_from_latent(observations, latent))
    assert torch.allclose(log_prob, repeated, atol=0.0, rtol=0.0)
    assert torch.equal(entropy, -log_prob)
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(value).all()
    loss = -(log_prob.mean() + 0.01 * value.mean())
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in policy.parameters()
    )


def test_residual_scale_configuration_and_100000_action_legality() -> None:
    policy = _policy(hidden_dim=16)
    manifest = policy.action_manifest()
    assert manifest["name"] == "state_centered_bounded_residual_v1"
    assert manifest["clipping_emitted_action"] is False
    assert manifest["projection"] is False
    assert manifest["residual_scales"] == pytest.approx([0.08] * 7 + [0.25])
    audit = audit_residual_action_legality(
        policy,
        _observations(),
        sample_count=100_000,
        batch_size=10_000,
    )
    assert audit["passed"] is True
    assert audit["nonfinite_actions"] == 0
    assert audit["out_of_bounds_actions"] == 0
    assert audit["clipping_events"] == 0
    assert audit["projection_events"] == 0


def test_f0_learning_metric_reconstruction_reports_retention_limits() -> None:
    report = json.loads(
        (
            PROJECT_ROOT
            / "artifacts"
            / "langmani_v2"
            / "phase_2b4_f0"
            / "micro_training_metrics.json"
        ).read_text(encoding="utf-8")
    )
    audit = reconstruct_f0_learning_dynamics(report)
    assert audit["recorded_snapshot_count"] == 33
    assert audit["completed_episodes"] == 21034
    classification = audit["diagnostic_classification"]
    assert isinstance(classification, dict)
    assert classification["no_useful_task_learning"] is True
    assert classification["critic_failure_proven"] is False
    unavailable = audit["unavailable_from_frozen_training_report"]
    assert "value explained variance" in unavailable
    assert "training termination reason per episode" in unavailable


def test_terminal_bootstrap_and_curriculum_physics_contracts() -> None:
    termination = termination_bootstrap_audit()
    assert termination["passed"] is True
    semantics = termination["training_semantics"]
    assert semantics["post_failure_tail_steps"] == 0
    assert semantics["terminated_bootstrap"] is False
    assert semantics["truncated_bootstrap"] is True
    curriculum = curriculum_manifest()
    physics = curriculum["physics_equivalence"]
    assert physics["environment"] == "LangMani-PushToRegion-v0"
    assert physics["success_predicate"] == "unchanged canonical stable success"
    assert physics["episode_budget"] == 250
    assert curriculum["stages"][0]["runnable_in_f1"] is True
    assert all(not stage["runnable_in_f1"] for stage in curriculum["stages"][1:])


def test_probe_gates_are_conjunctive_and_safety_is_zero_tolerance() -> None:
    probe_a = _evaluation(episodes=32, contact=8, progress=7)
    assert all(probe_a_gate(probe_a).values())
    unsafe_a = _evaluation(episodes=32, contact=8, progress=7, wrong_object=1)
    assert probe_a_gate(unsafe_a)["wrong_object_zero"] is False
    probe_b = _evaluation(
        episodes=48,
        successes=10,
        contact=24,
        progress=20,
        direction_successes={"left": 5, "forward_right": 5},
    )
    assert all(probe_b_gate(probe_b, included_directions=("left", "forward_right")).values())
    missing_direction = _evaluation(
        episodes=48,
        successes=10,
        contact=24,
        progress=20,
        direction_successes={"left": 10},
    )
    assert (
        probe_b_gate(
            missing_direction,
            included_directions=("left", "forward_right"),
        )["each_direction_success"]
        is False
    )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {
                "diagnosis_valid": False,
                "pipeline_valid": True,
                "residual_exploration_safer": True,
                "probe_a_passed": True,
                "probe_b_ran": True,
                "probe_b_passed": True,
            },
            Phase2B4F1Result.RESULT_D,
        ),
        (
            {
                "diagnosis_valid": True,
                "pipeline_valid": True,
                "residual_exploration_safer": False,
                "probe_a_passed": False,
                "probe_b_ran": False,
                "probe_b_passed": False,
            },
            Phase2B4F1Result.RESULT_C,
        ),
        (
            {
                "diagnosis_valid": True,
                "pipeline_valid": True,
                "residual_exploration_safer": True,
                "probe_a_passed": True,
                "probe_b_ran": True,
                "probe_b_passed": False,
            },
            Phase2B4F1Result.RESULT_B,
        ),
        (
            {
                "diagnosis_valid": True,
                "pipeline_valid": True,
                "residual_exploration_safer": True,
                "probe_a_passed": True,
                "probe_b_ran": True,
                "probe_b_passed": True,
            },
            Phase2B4F1Result.RESULT_A,
        ),
    ],
)
def test_result_classification_and_authorization_fail_closed(
    kwargs: dict[str, bool],
    expected: Phase2B4F1Result,
) -> None:
    result = classify_f1_result(**kwargs)
    state = authorization_state(result)
    assert result is expected
    assert state["ppo_safe_curriculum_eligible"] is (expected is Phase2B4F1Result.RESULT_A)
    assert state["ppo_full_training_authorized"] is False
    assert state["expert_qualification_authorized"] is False
    assert state["data_collection_authorized"] is False
    assert state["smolvla_training_authorized"] is False
    assert state["demonstration_source_validated"] is False
    assert state["student_policy_training_started"] is False


def test_probe_configs_freeze_total_budget_and_no_resume_or_sweep() -> None:
    configs = [probe_configuration(probe=probe) for probe in ("A", "B")]
    assert sum(int(config["total_environment_steps"]) for config in configs) == 524_288
    assert all(config["resume"] is False for config in configs)
    assert all(config["sweep"] is False for config in configs)
    assert all(config["formal_qualification"] is False for config in configs)
    assert all(config["demonstration_collection"] is False for config in configs)
