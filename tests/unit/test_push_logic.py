from __future__ import annotations

import pytest
import torch

from langmani.environments.push_logic import (
    PUSH_PRIVILEGED_OBSERVATION_KEYS,
    build_push_observation_extra,
    compute_push_object_planar_alignment,
    evaluate_push_state,
    update_progress_state,
    update_stable_success_count,
)


def test_push_object_alignment_allows_cylinder_roll_but_rejects_upright_axis() -> None:
    cube_rotation = torch.eye(3).repeat(3, 1, 1)
    cylinder_rotation = torch.eye(3).repeat(3, 1, 1)
    # Rolling about local x preserves the cylinder's intended horizontal symmetry axis.
    cylinder_rotation[1] = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    # Rotating local x onto world z leaves the intended rolling plane.
    cylinder_rotation[2] = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

    alignment = compute_push_object_planar_alignment(cube_rotation, cylinder_rotation)

    assert torch.allclose(alignment[:, 0], torch.ones(3))
    assert torch.allclose(alignment[:, 1], torch.tensor([1.0, 1.0, 0.0]))


def test_push_object_alignment_validates_rotation_shapes() -> None:
    with pytest.raises(ValueError, match="cube_rotation"):
        compute_push_object_planar_alignment(torch.eye(3), torch.eye(3))
    with pytest.raises(ValueError, match="cylinder_rotation"):
        compute_push_object_planar_alignment(torch.eye(3)[None], torch.eye(3).repeat(2, 1, 1))


def test_push_visual_observation_has_no_privileged_state() -> None:
    extra = build_push_observation_extra(
        tcp_pose=torch.zeros((2, 7)),
        use_privileged_state=False,
        object_poses=torch.ones((2, 14)),
        target_region_center=torch.ones((2, 3)),
        target_object_index=torch.zeros(2, dtype=torch.long),
        target_region_index=torch.zeros(2, dtype=torch.long),
        difficulty_index=torch.zeros(2, dtype=torch.long),
    )

    assert set(extra) == {"tcp_pose"}
    assert PUSH_PRIVILEGED_OBSERVATION_KEYS.isdisjoint(extra)


def test_push_stable_success_requires_consecutive_steps() -> None:
    count = torch.zeros(2, dtype=torch.int32)
    count = update_stable_success_count(count, torch.tensor([True, True]))
    count = update_stable_success_count(count, torch.tensor([True, False]))
    count = update_stable_success_count(count, torch.tensor([True, True]))

    assert count.tolist() == [3, 1]


def test_progress_state_detects_bounded_no_progress() -> None:
    best = torch.tensor([0.5, 0.5])
    steps = torch.tensor([2, 2], dtype=torch.int32)

    new_best, new_steps, stalled = update_progress_state(
        distance=torch.tensor([0.4, 0.4995]),
        best_distance=best,
        steps_without_progress=steps,
        minimum_improvement=0.001,
        stall_steps=3,
    )

    assert torch.allclose(new_best, torch.tensor([0.4, 0.4995]))
    assert new_steps.tolist() == [0, 3]
    assert stalled.tolist() == [False, True]


def test_batched_push_evaluation_separates_success_and_failure_events() -> None:
    batch_size = 8
    object_positions = torch.tensor(
        [[[0.0, 0.0, 0.025], [-0.2, 0.2, 0.025]]], dtype=torch.float32
    ).repeat(batch_size, 1, 1)
    initial_positions = object_positions.clone()
    target_center = torch.tensor([[0.2, 0.0, 0.001]], dtype=torch.float32).repeat(batch_size, 1)
    target_indices = torch.zeros(batch_size, dtype=torch.long)
    up = torch.ones((batch_size, 2))
    static = torch.ones((batch_size, 2), dtype=torch.bool)
    grasped = torch.zeros((batch_size, 2), dtype=torch.bool)
    stable = torch.full((batch_size,), 5, dtype=torch.int32)
    boolean = torch.zeros(batch_size, dtype=torch.bool)

    # 0: conservative stable success.
    object_positions[0, 0] = torch.tensor([0.2, 0.0, 0.025])
    # 1: inside, but not stable for enough consecutive steps.
    object_positions[1, 0] = torch.tensor([0.2, 0.0, 0.025])
    stable[1] = 4
    # 2: target was lifted.
    object_positions[2, 0] = torch.tensor([0.2, 0.0, 0.08])
    # 3: target left the workspace.
    object_positions[3, 0] = torch.tensor([0.7, 0.0, 0.025])
    # 4: target toppled.
    object_positions[4, 0] = torch.tensor([0.2, 0.0, 0.025])
    up[4, 0] = 0.2
    # 5: wrong object was displaced.
    object_positions[5, 1] = torch.tensor([-0.1, 0.2, 0.025])
    # 6: target overshot beyond the goal.
    object_positions[6, 0] = torch.tensor([0.25, 0.0, 0.025])
    # 7: action was invalid.
    invalid = boolean.clone()
    invalid[7] = True

    result = evaluate_push_state(
        object_positions=object_positions,
        object_up_alignment=up,
        object_is_static=static,
        object_is_grasped=grasped,
        initial_object_positions=initial_positions,
        initial_target_position=initial_positions[:, 0],
        target_region_center=target_center,
        target_object_index=target_indices,
        object_planar_radii=torch.tensor([[0.025, 0.025]]).repeat(batch_size, 1),
        object_resting_heights=torch.tensor([[0.025, 0.025]]).repeat(batch_size, 1),
        target_region_radius=torch.full((batch_size,), 0.11),
        stable_success_count=stable,
        required_stable_steps=5,
        workspace_bounds_xy=(-0.45, 0.45, -0.4, 0.4),
        containment_clearance=0.005,
        lift_tolerance=0.015,
        topple_alignment_threshold=0.75,
        wrong_object_displacement_threshold=0.03,
        wrong_object_contact=boolean,
        action_out_of_bounds=boolean,
        invalid_action=invalid,
        progress_stalled=boolean,
        robot_collision=boolean,
    )

    assert result["success"].tolist() == [True, False, False, False, False, False, False, False]
    assert result["target_lifted"][2]
    assert result["target_outside_workspace"][3]
    assert result["target_toppled"][4]
    assert result["wrong_object_displaced"][5]
    assert result["target_overshoot"][6]
    assert result["invalid_action"][7]
