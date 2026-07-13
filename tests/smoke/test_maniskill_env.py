"""ManiSkill environment smoke tests with explicit hardware boundaries."""

from __future__ import annotations

import os
import platform
import shutil
from collections.abc import Mapping
from importlib import import_module
from typing import Any

import pytest

gym = import_module("gymnasium")
import_module("mani_skill.envs")
torch = import_module("torch")


def _is_native_linux() -> bool:
    """Exclude WSL from the native Linux test boundary."""
    release = platform.release().lower()
    is_wsl = "microsoft" in release or bool(os.environ.get("WSL_INTEROP"))
    return platform.system() == "Linux" and not is_wsl


def _assert_reset_result(result: Any) -> Any:
    """Validate the Gymnasium reset contract without assuming NumPy leaves."""
    assert isinstance(result, tuple)
    assert len(result) == 2
    observation, info = result
    assert observation is not None
    assert isinstance(info, Mapping)
    return observation


def _assert_step_result(result: Any) -> None:
    """Validate the batched ManiSkill/Gymnasium five-value step contract."""
    assert isinstance(result, tuple)
    assert len(result) == 5
    observation, reward, terminated, truncated, info = result
    if isinstance(observation, Mapping):
        assert observation
    else:
        assert torch.is_tensor(observation)
        assert len(observation.shape) == 2
        assert observation.shape[0] == 1
        assert observation.shape[1] > 0
    assert torch.is_tensor(reward)
    assert tuple(reward.shape) == (1,)
    assert reward.dtype.is_floating_point
    assert torch.is_tensor(terminated)
    assert tuple(terminated.shape) == (1,)
    assert terminated.dtype == torch.bool
    assert torch.is_tensor(truncated)
    assert tuple(truncated.shape) == (1,)
    assert truncated.dtype == torch.bool
    assert isinstance(info, Mapping)


def _assert_rgb_observation(observation: Any) -> None:
    """Validate ManiSkill's batched policy-camera RGB contract."""
    assert isinstance(observation, Mapping)
    sensor_data = observation.get("sensor_data")
    assert isinstance(sensor_data, Mapping)
    assert sensor_data

    rgb_tensors = []
    for textures in sensor_data.values():
        assert isinstance(textures, Mapping)
        if "rgb" in textures:
            rgb_tensors.append(textures["rgb"])

    assert rgb_tensors
    for rgb in rgb_tensors:
        assert torch.is_tensor(rgb)
        assert len(rgb.shape) == 4
        assert rgb.shape[0] == 1
        assert rgb.shape[1] > 0 and rgb.shape[2] > 0
        assert rgb.shape[3] == 3
        assert rgb.dtype == torch.uint8
        assert rgb.numel() > 0


@pytest.mark.integration
@pytest.mark.skipif(
    not _is_native_linux(),
    reason="PickCube CPU execution is validated only on the repository's native Linux platform",
)
def test_pickcube_cpu_reset_and_step() -> None:
    """PickCube must reset and step without CUDA or rendering."""
    env = gym.make(
        "PickCube-v1",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        sim_backend="physx_cpu",
        render_backend="none",
    )
    try:
        observation = _assert_reset_result(env.reset(seed=0))
        assert torch.is_tensor(observation)
        assert len(observation.shape) == 2
        assert observation.shape[0] == 1
        assert observation.shape[1] > 0
        _assert_step_result(env.step(env.action_space.sample()))
    finally:
        env.close()


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.rendering
@pytest.mark.skipif(
    not _is_native_linux(),
    reason="GPU simulation and rendering verification requires native Linux",
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU simulation test requires torch.cuda.is_available()",
)
@pytest.mark.skipif(
    shutil.which("vulkaninfo") is None,
    reason="Rendering test requires vulkaninfo from the Vulkan tools package",
)
def test_pickcube_gpu_rgb_render_and_step() -> None:
    """The target machine must use GPU physics and produce policy and viewer RGB."""
    env = gym.make(
        "PickCube-v1",
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        sim_backend="physx_cuda",
        render_backend="sapien_cuda",
    )
    try:
        assert env.unwrapped.gpu_sim_enabled is True

        observation, info = env.reset(seed=0)
        assert isinstance(info, Mapping)
        _assert_rgb_observation(observation)

        frame = env.render()
        assert torch.is_tensor(frame)
        assert len(frame.shape) == 4
        assert frame.shape[0] == 1
        assert frame.shape[1] > 0 and frame.shape[2] > 0
        assert frame.shape[3] == 3
        assert frame.dtype == torch.uint8
        assert frame.numel() > 0

        _assert_step_result(env.step(env.action_space.sample()))
    finally:
        env.close()
