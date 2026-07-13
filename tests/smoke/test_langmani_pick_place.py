"""Simulator-level acceptance tests for the M1 LangMani environment."""

from __future__ import annotations

import os
import platform
import shutil
from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import pytest
import torch
from mani_skill.utils.structs import Pose

import langmani.environments  # noqa: F401 - registers the project environment
from langmani.environments.pick_place_by_instruction import (
    BIN_INTERIOR_HALF_SIZE,
    BIN_WALL_THICKNESS,
    BIN_X,
    CUBE_BOUNDING_RADIUS,
    CUBE_HALF_SIZE,
    ENV_ID,
    SOURCE_X_RANGE,
)
from langmani.environments.task_logic import PRIVILEGED_OBSERVATION_KEYS


def _is_native_linux() -> bool:
    release = platform.release().lower()
    is_wsl = "microsoft" in release or bool(os.environ.get("WSL_INTEROP"))
    return platform.system() == "Linux" and not is_wsl


def _task_spec(object_id: str, bin_id: str) -> dict[str, dict[str, str]]:
    return {
        "task_spec": {
            "target_object_id": object_id,
            "target_bin_id": bin_id,
            "instruction_template_id": "canonical_v0",
        }
    }


def _cube_poses(base_env: Any) -> torch.Tensor:
    return torch.stack([cube.pose.raw_pose for cube in base_env.cubes], dim=1)


def _teleport_cube_to_bin(base_env: Any, object_index: int, bin_index: int) -> None:
    position = base_env._bin_floor_centers()[:, bin_index].clone()
    position[:, 2] += CUBE_HALF_SIZE
    quaternion = torch.zeros((base_env.num_envs, 4), device=base_env.device)
    quaternion[:, 0] = 1.0
    cube = base_env.cubes[object_index]
    cube.set_pose(Pose.create_from_pq(p=position, q=quaternion))
    cube.set_linear_velocity(torch.zeros((base_env.num_envs, 3), device=base_env.device))
    cube.set_angular_velocity(torch.zeros((base_env.num_envs, 3), device=base_env.device))


def _all_mapping_keys(value: Any) -> set[str]:
    if not isinstance(value, Mapping):
        return set()
    keys = {str(key) for key in value}
    for child in value.values():
        keys.update(_all_mapping_keys(child))
    return keys


def _has_string_leaf(value: Any) -> bool:
    if isinstance(value, str):
        return True
    if isinstance(value, Mapping):
        return any(_has_string_leaf(child) for child in value.values())
    if isinstance(value, (tuple, list)):
        return any(_has_string_leaf(child) for child in value)
    return False


@pytest.mark.integration
@pytest.mark.skipif(
    not _is_native_linux(),
    reason="LangMani CPU simulation is validated only on the native Linux target",
)
def test_single_env_seeded_reset_counterfactual_and_privileged_state() -> None:
    env = gym.make(
        ENV_ID,
        num_envs=1,
        obs_mode="state_dict",
        reward_mode="sparse",
        control_mode="pd_joint_delta_pos",
        sim_backend="physx_cpu",
        render_backend="none",
    )
    try:
        observation_a, info_a = env.reset(
            seed=123,
            options=_task_spec("red_cube", "right_bin"),
        )
        base_env = env.unwrapped
        layout_a = _cube_poses(base_env).clone()
        qpos_a = base_env.agent.robot.get_qpos().clone()
        bin_poses_a = torch.stack(
            [bin_actor.pose.raw_pose for bin_actor in base_env.bins], dim=1
        ).clone()
        episode_a = base_env.get_episode_specs()[0]

        observation_b, info_b = env.reset(
            seed=123,
            options=_task_spec("green_cube", "left_bin"),
        )
        layout_b = _cube_poses(base_env).clone()
        qpos_b = base_env.agent.robot.get_qpos().clone()
        bin_poses_b = torch.stack(
            [bin_actor.pose.raw_pose for bin_actor in base_env.bins], dim=1
        ).clone()
        episode_b = base_env.get_episode_specs()[0]

        assert torch.equal(layout_a, layout_b)
        assert torch.equal(qpos_a, qpos_b)
        assert torch.equal(bin_poses_a, bin_poses_b)
        assert episode_a.scene_seed == episode_b.scene_seed == 123
        assert episode_a.scene_id == episode_b.scene_id
        assert episode_a.task_id != episode_b.task_id
        assert base_env.get_task_texts() == (
            "Pick up the green cube and place it in the left bin.",
        )

        extra = observation_b["extra"]
        assert set(extra) == {"tcp_pose", *PRIVILEGED_OBSERVATION_KEYS}
        assert extra["cube_poses"].shape == (1, 21)
        assert extra["bin_centers"].shape == (1, 6)
        assert extra["target_object_index"].tolist() == [1]
        assert extra["target_bin_index"].tolist() == [0]

        cube_positions = layout_b[0, :, :3]
        assert torch.all(cube_positions[:, 0] >= SOURCE_X_RANGE[0])
        assert torch.all(cube_positions[:, 0] <= SOURCE_X_RANGE[1])
        pair_distances = torch.pdist(cube_positions[:, :2])
        assert torch.all(pair_distances > 2 * CUBE_BOUNDING_RADIUS)
        assert torch.all(
            cube_positions[:, 0]
            < BIN_X - BIN_INTERIOR_HALF_SIZE - BIN_WALL_THICKNESS - CUBE_BOUNDING_RADIUS
        )
        for cube in base_env.cubes:
            assert torch.count_nonzero(cube.linear_velocity) == 0
            assert torch.count_nonzero(cube.angular_velocity) == 0

        expected_evaluation_keys = {
            "target_in_target_bin",
            "target_in_wrong_bin",
            "wrong_object_in_target_bin",
            "target_is_grasped",
            "target_is_static",
            "target_off_table",
            "success",
            "fail",
        }
        assert expected_evaluation_keys <= info_a.keys()
        assert expected_evaluation_keys <= info_b.keys()
        assert all(info_b[key].shape == (1,) for key in expected_evaluation_keys)
        assert all(info_b[key].dtype == torch.bool for key in expected_evaluation_keys)
        rollout_evaluation = base_env.get_policy_rollout_evaluation()
        assert set(rollout_evaluation) == {
            *expected_evaluation_keys,
            "wrong_object_is_grasped",
        }
        assert rollout_evaluation["wrong_object_is_grasped"].shape == (1,)

        state_before_invalid_reset = _cube_poses(base_env).clone()
        with pytest.raises(ValueError, match="unknown target_object_id"):
            env.reset(
                seed=999,
                options=_task_spec("yellow_cube", "left_bin"),
            )
        assert torch.equal(_cube_poses(base_env), state_before_invalid_reset)

        _teleport_cube_to_bin(base_env, object_index=1, bin_index=0)
        zero_action = torch.zeros(env.action_space.shape, device=base_env.device)
        for _ in range(20):
            _, reward, terminated, _, success_info = env.step(zero_action)
            if bool(success_info["success"][0]):
                break
        assert success_info["target_in_target_bin"][0]
        assert success_info["target_is_static"][0]
        assert not success_info["target_is_grasped"][0]
        assert success_info["success"][0]
        assert reward.tolist() == [1.0]
        assert terminated[0]

        env.reset(seed=123, options=_task_spec("green_cube", "left_bin"))
        _teleport_cube_to_bin(base_env, object_index=1, bin_index=1)
        wrong_bin_info = base_env.evaluate()
        assert wrong_bin_info["target_in_wrong_bin"][0]
        assert not wrong_bin_info["success"][0]

        env.reset(seed=123, options=_task_spec("green_cube", "left_bin"))
        _teleport_cube_to_bin(base_env, object_index=0, bin_index=0)
        wrong_object_info = base_env.evaluate()
        assert wrong_object_info["wrong_object_in_target_bin"][0]
        assert not wrong_object_info["success"][0]
    finally:
        env.close()


@pytest.mark.integration
@pytest.mark.skipif(
    not _is_native_linux(),
    reason="LangMani CPU simulation is validated only on the native Linux target",
)
def test_standard_flat_state_mode_constructs_and_resets() -> None:
    env = gym.make(
        ENV_ID,
        num_envs=1,
        obs_mode="state",
        reward_mode="sparse",
        control_mode="pd_joint_delta_pos",
        sim_backend="physx_cpu",
        render_backend="none",
    )
    try:
        observation, _ = env.reset(seed=123)
        assert torch.is_tensor(observation)
        assert observation.ndim == 2
        assert observation.shape[0] == 1
        assert observation.shape[1] > 35
    finally:
        env.close()


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.skipif(
    not _is_native_linux(),
    reason="LangMani batched GPU simulation requires native Linux",
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="LangMani batched GPU simulation requires CUDA",
)
def test_six_env_gpu_batch_covers_numeric_task_contract() -> None:
    env = gym.make(
        ENV_ID,
        num_envs=6,
        obs_mode="state_dict",
        reward_mode="none",
        control_mode="pd_joint_delta_pos",
        sim_backend="physx_cuda",
        render_backend="none",
    )
    try:
        observation, info = env.reset(seed=[0, 1, 2, 3, 4, 5])
        base_env = env.unwrapped

        assert base_env.gpu_sim_enabled is True
        assert observation["extra"]["cube_poses"].shape == (6, 21)
        assert observation["extra"]["bin_centers"].shape == (6, 6)
        assert observation["extra"]["target_object_index"].tolist() == [0, 0, 1, 1, 2, 2]
        assert observation["extra"]["target_bin_index"].tolist() == [0, 1, 0, 1, 0, 1]
        assert base_env.get_task_texts() == (
            "Pick up the red cube and place it in the left bin.",
            "Pick up the red cube and place it in the right bin.",
            "Pick up the green cube and place it in the left bin.",
            "Pick up the green cube and place it in the right bin.",
            "Pick up the blue cube and place it in the left bin.",
            "Pick up the blue cube and place it in the right bin.",
        )
        assert tuple(spec.scene_seed for spec in base_env.get_episode_specs()) == tuple(range(6))
        assert all(value.shape == (6,) for value in info.values() if torch.is_tensor(value))

        layout_before_partial = _cube_poses(base_env).clone()
        qpos_before_partial = base_env.agent.robot.get_qpos().clone()
        specs_before_partial = base_env.get_episode_specs()
        partial_options: dict[str, Any] = _task_spec("red_cube", "right_bin")
        partial_options["env_idx"] = [1, 4]
        env.reset(options=partial_options)
        layout_after_partial = _cube_poses(base_env)
        qpos_after_partial = base_env.agent.robot.get_qpos()
        specs_after_partial = base_env.get_episode_specs()

        untouched = torch.tensor([0, 2, 3, 5], device=base_env.device)
        selected = torch.tensor([1, 4], device=base_env.device)
        assert torch.equal(layout_before_partial[untouched], layout_after_partial[untouched])
        assert torch.equal(qpos_before_partial[untouched], qpos_after_partial[untouched])
        assert torch.all(
            torch.any(layout_before_partial[selected] != layout_after_partial[selected], dim=(1, 2))
        )
        assert all(
            specs_before_partial[index] == specs_after_partial[index] for index in (0, 2, 3, 5)
        )
        assert all(
            specs_after_partial[index].task_spec.target_object_id == "red_cube"
            and specs_after_partial[index].task_spec.target_bin_id == "right_bin"
            for index in (1, 4)
        )

        _, reward, terminated, truncated, step_info = env.step(env.action_space.sample())
        assert reward.shape == terminated.shape == truncated.shape == (6,)
        assert torch.count_nonzero(reward) == 0
        assert not _has_string_leaf(step_info)

        env.reset(
            seed=[10, 11, 12, 13, 14, 15],
            options=_task_spec("red_cube", "right_bin"),
        )
        _teleport_cube_to_bin(base_env, object_index=0, bin_index=1)
        base_env.scene._gpu_apply_all()
        zero_action = torch.zeros(env.action_space.shape, device=base_env.device)
        for _ in range(20):
            _, reward, terminated, _, success_info = env.step(zero_action)
            if bool(torch.all(success_info["success"])):
                break
        assert torch.all(success_info["target_in_target_bin"])
        assert torch.all(success_info["target_is_static"])
        assert not torch.any(success_info["target_is_grasped"])
        assert torch.all(success_info["success"])
        assert torch.all(reward == 1)
        assert torch.all(terminated)
    finally:
        env.close()


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.rendering
@pytest.mark.skipif(
    not _is_native_linux(),
    reason="LangMani camera acceptance requires native Linux",
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="LangMani camera acceptance requires CUDA",
)
@pytest.mark.skipif(
    shutil.which("vulkaninfo") is None,
    reason="LangMani camera acceptance requires Vulkan tooling",
)
def test_visual_observation_has_no_privileged_leak_and_sees_scene_objects() -> None:
    env = gym.make(
        ENV_ID,
        num_envs=1,
        obs_mode="rgb+segmentation",
        reward_mode="none",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        sim_backend="physx_cuda",
        render_backend="sapien_cuda",
    )
    try:
        observation, _ = env.reset(seed=123)
        base_env = env.unwrapped

        assert set(observation["extra"]) == {"tcp_pose"}
        assert PRIVILEGED_OBSERVATION_KEYS.isdisjoint(_all_mapping_keys(observation))
        assert not _has_string_leaf(observation)
        assert set(observation["sensor_data"]) == {"base_camera"}

        textures = observation["sensor_data"]["base_camera"]
        assert textures["rgb"].shape == (1, 256, 256, 3)
        assert torch.unique(textures["rgb"]).numel() > 16
        segmentation = textures["segmentation"]
        actor_segmentation = segmentation[..., 0]
        for actor in (*base_env.cubes, *base_env.bins):
            actor_id = int(actor.per_scene_id.detach().cpu().tolist()[0])
            assert torch.count_nonzero(actor_segmentation == actor_id) >= 4

        links_by_name = {link.name: link for link in base_env.agent.robot.get_links()}
        assert "panda_hand" in links_by_name
        hand_id = int(links_by_name["panda_hand"].per_scene_id.detach().cpu().tolist()[0])
        assert torch.count_nonzero(actor_segmentation == hand_id) >= 4

        frame = env.render()
        assert frame.shape == (1, 512, 512, 3)
        assert frame.dtype == torch.uint8
        assert torch.unique(frame).numel() > 16
    finally:
        env.close()
