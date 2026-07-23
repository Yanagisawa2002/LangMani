from __future__ import annotations

from pathlib import Path

import pytest
import torch

from langmani.v2.phase2b4_f1 import StateCenteredResidualActorCritic
from langmani.v2.phase2b4_f1_runtime import (
    F1_CHECKPOINT_SCHEMA,
    _f1_checkpoint_payload,
    _reward_with_revision,
    _save_checkpoint,
    _termination_reason,
    reconstruct_f1_policy,
)
from langmani.v2.phase2b4_ppo import (
    TEACHER_OBSERVATION_DIM,
    PPOConfig,
)
from langmani.v2.phase2b4_runtime import (
    TeacherState,
    _reward_from_transition,
)


def _policy() -> StateCenteredResidualActorCritic:
    return StateCenteredResidualActorCritic(
        observation_dim=TEACHER_OBSERVATION_DIM,
        action_low=torch.tensor([-2.9] * 7 + [-1.0]),
        action_high=torch.tensor([2.9] * 7 + [1.0]),
        hidden_dim=32,
    )


def _state(*, contact: bool = False) -> TeacherState:
    return TeacherState(
        observation=torch.zeros((1, TEACHER_OBSERVATION_DIM)),
        target_distance=torch.tensor([0.3]),
        behind_distance=torch.tensor([0.2]),
        containment_fraction=torch.tensor([0.0]),
        target_contact=torch.tensor([contact]),
        push_alignment=torch.tensor([1.0]),
    )


def _info(**overrides: bool) -> dict[str, torch.Tensor]:
    values = {
        "success": False,
        "wrong_object_contact": False,
        "wrong_object_displaced": False,
        "target_outside_workspace": False,
        "invalid_action": False,
        "action_out_of_bounds": False,
        "target_lifted": False,
        "target_toppled": False,
        "target_is_grasped": False,
        "robot_collision": False,
    }
    values.update(overrides)
    return {key: torch.tensor([value], dtype=torch.bool) for key, value in values.items()}


def test_f1_checkpoint_round_trip_reconstructs_action_transform(tmp_path: Path) -> None:
    policy = _policy()
    optimizer = torch.optim.Adam(policy.parameters())
    config = PPOConfig(
        num_envs=2,
        num_steps=2,
        total_timesteps=4,
        num_minibatches=1,
        update_epochs=1,
        hidden_dim=32,
        initial_log_std=-0.5,
    )
    payload = _f1_checkpoint_payload(
        policy=policy,
        optimizer=optimizer,
        config=config,
        global_step=4,
        probe="A",
        reward_revision=0,
    )
    checkpoint = tmp_path / "f1.pt"
    digest = _save_checkpoint(checkpoint, payload)
    reconstructed, loaded = reconstruct_f1_policy(checkpoint)
    observation = torch.zeros((1, TEACHER_OBSERVATION_DIM))
    observation[:, 7:9] = 0.02
    assert loaded["schema_version"] == F1_CHECKPOINT_SCHEMA
    assert loaded["global_step"] == 4
    assert digest.startswith("sha256:")
    assert torch.equal(
        policy.deterministic_action(observation),
        reconstructed.deterministic_action(observation),
    )
    assert policy.action_manifest()["fingerprint"] == reconstructed.action_manifest()["fingerprint"]


def test_reward_revision_zero_is_exact_f0_accounting_and_revision_one_is_blocked() -> None:
    previous = _state()
    current = _state()
    info = _info()
    action = torch.zeros((1, 8))
    expected_total, expected_components = _reward_from_transition(
        previous=previous,
        current=current,
        info=info,
        action_unit=action,
        previous_action_unit=action,
    )
    observed_total, observed_components = _reward_with_revision(
        previous=previous,
        current=current,
        info=info,
        action_unit=action,
        previous_action_unit=action,
        reward_revision=0,
    )
    assert torch.equal(observed_total, expected_total)
    assert observed_components.keys() == expected_components.keys()
    assert all(
        torch.equal(observed_components[key], expected_components[key])
        for key in expected_components
    )
    with pytest.raises(PermissionError, match="not implemented"):
        _reward_with_revision(
            previous=previous,
            current=current,
            info=info,
            action_unit=action,
            previous_action_unit=action,
            reward_revision=1,
        )


@pytest.mark.parametrize(
    ("overrides", "terminated", "truncated", "expected"),
    [
        ({"success": True}, True, False, "stable_success"),
        ({"wrong_object_displaced": True}, True, False, "wrong_object"),
        (
            {"target_outside_workspace": True},
            True,
            False,
            "target_outside_workspace",
        ),
        ({}, False, True, "timeout"),
        ({"invalid_action": True}, True, False, "other_failure"),
        ({}, False, False, "incomplete"),
    ],
)
def test_termination_classification(
    overrides: dict[str, bool],
    terminated: bool,
    truncated: bool,
    expected: str,
) -> None:
    assert (
        _termination_reason(
            _info(**overrides),
            terminated=terminated,
            truncated=truncated,
        )
        == expected
    )
