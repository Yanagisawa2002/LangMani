"""CPU-only tests for batched M1 evaluation and observation gating."""

from __future__ import annotations

import torch
from mani_skill.utils.common import flatten_state_dict

from langmani.environments.task_logic import (
    PRIVILEGED_OBSERVATION_KEYS,
    build_observation_extra,
    evaluate_task_state,
)


def test_visual_only_observation_extra_has_no_privileged_leakage() -> None:
    batch_size = 2
    extra = build_observation_extra(
        tcp_pose=torch.zeros((batch_size, 7)),
        use_privileged_state=False,
        cube_poses=torch.ones((batch_size, 21)),
        bin_centers=torch.ones((batch_size, 6)),
        target_object_index=torch.zeros(batch_size, dtype=torch.long),
        target_bin_index=torch.zeros(batch_size, dtype=torch.long),
    )

    assert set(extra) == {"tcp_pose"}
    assert PRIVILEGED_OBSERVATION_KEYS.isdisjoint(extra)


def test_privileged_observation_extra_has_all_numeric_task_fields() -> None:
    batch_size = 2
    extra = build_observation_extra(
        tcp_pose=torch.zeros((batch_size, 7)),
        use_privileged_state=True,
        cube_poses=torch.ones((batch_size, 21)),
        bin_centers=torch.ones((batch_size, 6)),
        target_object_index=torch.zeros(batch_size, dtype=torch.long),
        target_bin_index=torch.ones(batch_size, dtype=torch.long),
    )

    assert set(extra) == {"tcp_pose", *PRIVILEGED_OBSERVATION_KEYS}
    assert all(torch.is_tensor(value) for value in extra.values())
    assert extra["cube_poses"].shape == (batch_size, 21)
    assert extra["bin_centers"].shape == (batch_size, 6)

    flattened = flatten_state_dict(
        {"agent": {"qpos": torch.zeros((batch_size, 9))}, "extra": extra},
        use_torch=True,
    )
    assert flattened.shape == (batch_size, 9 + 7 + 21 + 6 + 1 + 1)


def test_batched_evaluation_covers_success_and_conservative_failures() -> None:
    batch_size = 9
    bin_floor_centers = torch.tensor(
        [[[0.4, 0.5, 0.0], [0.4, -0.5, 0.0]]], dtype=torch.float32
    ).expand(batch_size, -1, -1)
    cube_positions = torch.tensor(
        [[[-0.3, 0.0, 0.025], [-0.2, 0.0, 0.025], [-0.1, 0.0, 0.025]]],
        dtype=torch.float32,
    ).repeat(batch_size, 1, 1)
    cube_is_grasped = torch.zeros((batch_size, 3), dtype=torch.bool)
    cube_is_static = torch.ones((batch_size, 3), dtype=torch.bool)
    target_object_index = torch.zeros(batch_size, dtype=torch.long)
    target_bin_index = torch.zeros(batch_size, dtype=torch.long)

    # 0: conservative success.
    cube_positions[0, 0] = torch.tensor([0.4, 0.5, 0.025])
    # 1: selected target is in the wrong bin.
    cube_positions[1, 0] = torch.tensor([0.4, -0.5, 0.025])
    # 2: only a wrong object is in the target bin.
    cube_positions[2, 1] = torch.tensor([0.4, 0.5, 0.025])
    # 3: selected target is positioned correctly but still grasped.
    cube_positions[3, 0] = torch.tensor([0.4, 0.5, 0.025])
    cube_is_grasped[3, 0] = True
    # 4: selected target is positioned correctly but moving.
    cube_positions[4, 0] = torch.tensor([0.4, 0.5, 0.025])
    cube_is_static[4, 0] = False
    # 5: selected target is held/hovering over the correct bin.
    cube_positions[5, 0] = torch.tensor([0.4, 0.5, 0.031])
    cube_is_grasped[5, 0] = True
    # 6: selected target has fallen entirely below the tabletop.
    cube_positions[6, 0] = torch.tensor([-0.3, 0.0, -0.05])
    # 7: target and a wrong object both occupy the target bin.
    cube_positions[7, 0] = torch.tensor([0.4, 0.5, 0.025])
    cube_positions[7, 2] = torch.tensor([0.4, 0.5, 0.025])
    # 8: a grasped target beyond the table edge is still recoverable.
    cube_positions[8, 0] = torch.tensor([1.8, 0.0, 0.15])
    cube_is_grasped[8, 0] = True

    result = evaluate_task_state(
        cube_positions=cube_positions,
        bin_floor_centers=bin_floor_centers,
        cube_is_grasped=cube_is_grasped,
        cube_is_static=cube_is_static,
        target_object_index=target_object_index,
        target_bin_index=target_bin_index,
        cube_bounding_radius=0.044,
        cube_resting_height=0.025,
        bin_interior_half_size=0.085,
        containment_clearance=0.002,
        resting_height_tolerance=0.005,
        off_table_height=-0.044,
    )

    expected_keys = {
        "target_in_target_bin",
        "target_in_wrong_bin",
        "wrong_object_in_target_bin",
        "target_is_grasped",
        "target_is_static",
        "target_off_table",
        "success",
        "fail",
    }
    assert set(result) == expected_keys
    assert all(value.shape == (batch_size,) for value in result.values())
    assert all(value.dtype == torch.bool for value in result.values())

    assert result["success"].tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert result["target_in_wrong_bin"][1]
    assert result["wrong_object_in_target_bin"][2]
    assert result["target_is_grasped"][3]
    assert not result["target_is_static"][4]
    assert not result["target_in_target_bin"][5]
    assert result["target_off_table"][6]
    assert result["fail"].tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
    ]
    assert result["wrong_object_in_target_bin"][7]
    assert not result["target_off_table"][8]
