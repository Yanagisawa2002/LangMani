"""Structural tests for the registered M1 environment contract."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import pytest
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import REGISTERED_ENVS, TimeLimitWrapper

import langmani.environments  # noqa: F401 - import performs the single registration
import langmani.environments.pick_place_by_instruction as environment_module
from langmani.environments.pick_place_by_instruction import (
    BIN_INTERIOR_HALF_SIZE,
    BIN_WALL_THICKNESS,
    BIN_X,
    CUBE_BOUNDING_RADIUS,
    ENV_ID,
    HUMAN_CAMERA_RESOLUTION,
    INITIAL_CUBE_BUILD_POSITIONS,
    MAX_EPISODE_STEPS,
    POLICY_CAMERA_RESOLUTION,
    SOURCE_X_RANGE,
    SOURCE_Y_CENTERS,
    SOURCE_Y_JITTER,
    PickPlaceByInstructionEnv,
    _normalize_reset_options,
    _validated_env_indices,
)


def test_exactly_one_langmani_environment_is_registered() -> None:
    langmani_ids = sorted(uid for uid in REGISTERED_ENVS if uid.startswith("LangMani-"))

    assert langmani_ids == ["LangMani-PickPlaceByInstruction-v0"]
    spec = REGISTERED_ENVS[ENV_ID]
    assert spec.max_episode_steps == MAX_EPISODE_STEPS == 200
    assert spec.asset_download_ids == []


def test_environment_is_panda_only_and_has_no_dense_reward_mode() -> None:
    assert PickPlaceByInstructionEnv.SUPPORTED_ROBOTS == ["panda"]
    assert PickPlaceByInstructionEnv.SUPPORTED_REWARD_MODES == ("sparse", "none")

    with pytest.raises(ValueError, match="only robot_uids='panda'"):
        PickPlaceByInstructionEnv(robot_uids="fetch")


def test_sparse_and_none_reward_semantics_are_batched() -> None:
    env = object.__new__(PickPlaceByInstructionEnv)
    info = {
        "success": torch.tensor([False, True, False, True]),
        "fail": torch.tensor([False, False, True, True]),
    }

    sparse = BaseEnv.compute_sparse_reward(env, obs={}, action=torch.zeros((4, 1)), info=info)
    assert torch.equal(sparse, torch.tensor([0.0, 1.0, -1.0, 0.0]))

    env._reward_mode = "none"
    env.num_envs = 4
    env.device = torch.device("cpu")
    none_reward = BaseEnv.get_reward(env, obs={}, action=torch.zeros((4, 1)), info=info)
    assert torch.equal(none_reward, torch.zeros(4))


def test_registered_time_limit_truncates_as_batched_tensor_at_step_200() -> None:
    class StubEnv(gym.Env):
        def __init__(self) -> None:
            self.elapsed_steps = torch.tensor([198], dtype=torch.int32)

        def step(self, action: Any) -> tuple[Any, ...]:
            del action
            self.elapsed_steps += 1
            return (
                {},
                torch.zeros(1),
                torch.zeros(1, dtype=torch.bool),
                torch.zeros(1, dtype=torch.bool),
                {},
            )

    wrapper = TimeLimitWrapper(StubEnv(), max_episode_steps=MAX_EPISODE_STEPS)

    assert not wrapper.step(None)[3][0]
    truncated = wrapper.step(None)[3]
    assert truncated.shape == (1,)
    assert truncated.dtype == torch.bool
    assert truncated[0]


def test_policy_and_human_cameras_are_separate_and_documentable() -> None:
    policy = PickPlaceByInstructionEnv.policy_camera_config()
    human = PickPlaceByInstructionEnv.human_camera_config()

    assert policy.uid == "base_camera"
    assert (policy.width, policy.height) == POLICY_CAMERA_RESOLUTION == (256, 256)
    assert policy.shader_pack == "minimal"
    assert policy.near == 0.01
    assert policy.far == 10.0

    assert human.uid == "render_camera"
    assert (human.width, human.height) == HUMAN_CAMERA_RESOLUTION == (512, 512)
    assert human.width > policy.width
    assert human.height > policy.height
    assert human.uid != policy.uid


def test_dynamic_cubes_have_safe_nonoverlapping_build_poses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeTableSceneBuilder:
        def __init__(self, *, env: object, robot_init_qpos_noise: float) -> None:
            del env, robot_init_qpos_noise

        def build(self) -> None:
            return None

    def fake_build_cube(scene: object, **kwargs: object) -> str:
        del scene
        calls.append(kwargs)
        return str(kwargs["name"])

    monkeypatch.setattr(environment_module, "_PrimitiveTableSceneBuilder", FakeTableSceneBuilder)
    monkeypatch.setattr(environment_module.actors, "build_cube", fake_build_cube)
    monkeypatch.setattr(
        PickPlaceByInstructionEnv,
        "_build_bin",
        lambda self, name, position: (name, position),
    )

    env = object.__new__(PickPlaceByInstructionEnv)
    env.scene = object()
    env.robot_init_qpos_noise = 0.0
    env._load_scene({})

    build_positions = tuple(
        tuple(float(value) for value in call["initial_pose"].p)  # type: ignore[union-attr]
        for call in calls
    )
    for actual, expected in zip(build_positions, INITIAL_CUBE_BUILD_POSITIONS, strict=True):
        assert actual == pytest.approx(expected)
    assert len(set(build_positions)) == 3


def test_random_source_cells_guarantee_scene_separation_for_every_seed() -> None:
    adjacent_center_gap = min(
        right - left for left, right in zip(SOURCE_Y_CENTERS, SOURCE_Y_CENTERS[1:], strict=False)
    )
    minimum_y_gap = adjacent_center_gap - 2 * SOURCE_Y_JITTER
    bin_min_x = BIN_X - BIN_INTERIOR_HALF_SIZE - BIN_WALL_THICKNESS

    assert minimum_y_gap > 2 * CUBE_BOUNDING_RADIUS
    assert SOURCE_X_RANGE[1] + CUBE_BOUNDING_RADIUS < bin_min_x


def test_reset_option_validation_is_strict_and_preserves_public_shape() -> None:
    raw = {
        "task_spec": {
            "target_object_id": "red_cube",
            "target_bin_id": "right_bin",
            "instruction_template_id": "canonical_v0",
        }
    }

    normalized = _normalize_reset_options(raw)

    assert normalized == raw
    assert normalized is not raw
    assert normalized["task_spec"] is not raw["task_spec"]


def test_reset_env_indices_accept_official_wrapper_array_and_normalize_device() -> None:
    numpy = pytest.importorskip("numpy")
    options = _normalize_reset_options({"env_idx": numpy.array([0, 2])})

    normalized = _validated_env_indices(
        options["env_idx"],  # type: ignore[arg-type]
        num_envs=3,
        device=torch.device("cpu"),
    )

    assert torch.equal(normalized, torch.tensor([0, 2], dtype=torch.long))


@pytest.mark.parametrize(
    ("env_idx", "message"),
    [
        (torch.tensor([0, 0]), "must not contain duplicates"),
        (torch.tensor([2, 0]), "must be strictly increasing"),
        (torch.tensor([0, 3]), r"must be in \[0, 3\)"),
    ],
)
def test_reset_env_indices_reject_ambiguous_or_invalid_order(
    env_idx: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _validated_env_indices(env_idx, num_envs=3, device=torch.device("cpu"))


@pytest.mark.parametrize(
    ("options", "error_type", "message"),
    [
        ([], TypeError, "reset options must be a mapping"),
        ({"unknown": True}, ValueError, "unsupported reset option keys"),
        (
            {"reset_to_env_states": {}},
            ValueError,
            "reset_to_env_states is not supported",
        ),
        ({"reconfigure": "yes"}, TypeError, "reconfigure.*bool"),
        ({"task_spec": []}, TypeError, "task_spec.*mapping"),
        ({1: True}, TypeError, "reset option keys must be strings"),
        ({"env_idx": []}, ValueError, "env_idx.*empty"),
        ({"env_idx": [0.5]}, TypeError, "env_idx.*integer"),
    ],
)
def test_invalid_reset_options_fail_before_simulation(
    options: object, error_type: type[Exception], message: str
) -> None:
    with pytest.raises(error_type, match=message):
        _normalize_reset_options(options)  # type: ignore[arg-type]


def test_expert_context_resolves_semantic_handles_without_actor_order() -> None:
    env = object.__new__(PickPlaceByInstructionEnv)
    env.num_envs = 1
    env.device = torch.device("cpu")
    env._scene_seeds = torch.tensor([37], dtype=torch.long)
    env._target_object_indices = torch.tensor([2], dtype=torch.long)
    env._target_bin_indices = torch.tensor([0], dtype=torch.long)

    red, green, blue = object(), object(), object()
    left = SimpleNamespace(pose=SimpleNamespace(p=torch.tensor([[0.08, 0.18, 0.004]])))
    right = SimpleNamespace(pose=SimpleNamespace(p=torch.tensor([[0.08, -0.18, 0.004]])))
    env.cubes = (green, blue, red)
    env.bins = (left, right)
    env._objects_by_id = {
        "red_cube": red,
        "green_cube": green,
        "blue_cube": blue,
    }
    env._bins_by_id = {"left_bin": left, "right_bin": right}
    robot = object()
    env.agent = SimpleNamespace(robot=robot)

    context = env.get_expert_task_context()

    assert context.episode_spec.scene_seed == 37
    assert context.target_object.object_id == "blue_cube"
    assert context.target_object.actor is blue
    assert context.target_bin.bin_id == "left_bin"
    assert context.target_bin.actor is left
    assert context.robot is robot
    assert tuple(handle.object_id for handle in context.objects) == (
        "red_cube",
        "green_cube",
        "blue_cube",
    )


def test_expert_context_rejects_vectorized_access_before_materializing_metadata() -> None:
    env = object.__new__(PickPlaceByInstructionEnv)
    env.num_envs = 2

    with pytest.raises(ValueError, match="require num_envs=1"):
        env.get_expert_task_context()
