"""Structural contracts for the LangMani 2.0 planar-pushing environment."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mani_skill.utils.registration import REGISTERED_ENVS

import langmani.environments.push_to_region as environment_module
from langmani.environments.pick_place_by_instruction import (
    HUMAN_CAMERA_RESOLUTION,
    POLICY_CAMERA_RESOLUTION,
)
from langmani.environments.push_to_region import (
    ENV_ID,
    HARD_REGION_RADIUS,
    MAX_EPISODE_STEPS,
    ROBOT_INIT_QPOS_NOISE,
    STANDARD_REGION_RADIUS,
    TARGET_MARKER_QUATERNION,
    TARGET_REGION_CENTERS_XY,
    PushToRegionEnv,
    _normalize_push_reset_options,
)


def test_push_environment_registration_robot_rewards_and_time_limit() -> None:
    spec = REGISTERED_ENVS[ENV_ID]

    assert spec.max_episode_steps == MAX_EPISODE_STEPS == 250
    assert spec.asset_download_ids == []
    assert PushToRegionEnv.SUPPORTED_ROBOTS == ["panda"]
    assert PushToRegionEnv.SUPPORTED_REWARD_MODES == ("sparse", "none")
    assert pytest.approx(0.01) == ROBOT_INIT_QPOS_NOISE

    with pytest.raises(ValueError, match="only robot_uids='panda'"):
        PushToRegionEnv(robot_uids="fetch")


def test_push_cameras_reuse_the_deployable_m1_contract() -> None:
    policy = PushToRegionEnv.policy_camera_config()
    human = PushToRegionEnv.human_camera_config()

    assert policy.uid == "base_camera"
    assert (policy.width, policy.height) == POLICY_CAMERA_RESOLUTION == (256, 256)
    assert policy.shader_pack == "minimal"
    assert human.uid == "render_camera"
    assert (human.width, human.height) == HUMAN_CAMERA_RESOLUTION == (512, 512)


def test_push_scene_builds_two_dynamic_geometries_and_two_exact_goal_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects: list[dict[str, object]] = []
    targets: list[dict[str, object]] = []

    class FakeTableSceneBuilder:
        def __init__(self, *, env: object, robot_init_qpos_noise: float) -> None:
            del env
            assert robot_init_qpos_noise == ROBOT_INIT_QPOS_NOISE

        def build(self) -> None:
            return None

    def build_object(scene: object, **kwargs: object) -> str:
        del scene
        objects.append(kwargs)
        return str(kwargs["name"])

    def build_target(scene: object, **kwargs: object) -> str:
        del scene
        targets.append(kwargs)
        return str(kwargs["name"])

    monkeypatch.setattr(environment_module, "_PrimitiveTableSceneBuilder", FakeTableSceneBuilder)
    monkeypatch.setattr(environment_module.actors, "build_cube", build_object)
    monkeypatch.setattr(environment_module.actors, "build_cylinder", build_object)
    monkeypatch.setattr(environment_module.actors, "build_red_white_target", build_target)

    env = object.__new__(PushToRegionEnv)
    env.scene = object()
    env.robot_init_qpos_noise = ROBOT_INIT_QPOS_NOISE
    env._load_scene({})

    assert [item["name"] for item in objects] == ["blue_cube", "orange_cylinder"]
    assert all(item["body_type"] == "dynamic" for item in objects)
    assert [item["radius"] for item in targets] == [
        STANDARD_REGION_RADIUS,
        HARD_REGION_RADIUS,
    ]
    assert all(item["body_type"] == "kinematic" for item in targets)
    assert all(item["add_collision"] is False for item in targets)
    assert all(
        np.allclose(item["initial_pose"].q, TARGET_MARKER_QUATERNION)  # type: ignore[union-attr]
        for item in targets
    )


def test_push_target_regions_are_distinct_and_within_panda_reach() -> None:
    assert TARGET_REGION_CENTERS_XY == (
        (0.12, 0.22),
        (0.12, -0.22),
        (0.22, 0.14),
        (0.22, -0.14),
    )
    assert len(set(TARGET_REGION_CENTERS_XY)) == 4


def test_push_reset_options_are_strict_and_json_shaped() -> None:
    raw = {
        "task_spec": {
            "target_object_id": "orange_cylinder",
            "target_region_id": "forward_right",
            "difficulty": "hard",
            "instruction_template_id": "canonical_push_v0",
        }
    }

    normalized = _normalize_push_reset_options(raw)

    assert normalized == raw
    assert normalized is not raw
    assert normalized["task_spec"] is not raw["task_spec"]


@pytest.mark.parametrize(
    ("options", "error_type", "message"),
    [
        ([], TypeError, "reset options must be a mapping"),
        ({"unknown": True}, ValueError, "unsupported reset option keys"),
        ({"reconfigure": "yes"}, TypeError, "reconfigure.*bool"),
        ({"task_spec": []}, TypeError, "task_spec.*mapping"),
        ({"task_spec": {}}, ValueError, "malformed push task_spec"),
        ({"env_idx": []}, ValueError, "env_idx.*non-empty"),
        ({"env_idx": [0.5]}, TypeError, "env_idx.*integers"),
    ],
)
def test_invalid_push_reset_options_fail_before_simulation(
    options: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        _normalize_push_reset_options(options)  # type: ignore[arg-type]


def test_push_action_inspection_records_nonfinite_and_bounds_without_silent_projection() -> None:
    env = object.__new__(PushToRegionEnv)
    env.num_envs = 2
    env.device = torch.device("cpu")
    env._orig_single_action_space = SimpleNamespace(
        low=np.full(8, -1.0, dtype=np.float32),
        high=np.full(8, 1.0, dtype=np.float32),
    )
    action = torch.zeros((2, 8), dtype=torch.float32)
    action[0, 0] = torch.nan
    action[1, 1] = 1.25

    prepared, invalid, out_of_bounds = env._inspect_action(action)

    assert torch.isfinite(prepared).all()
    assert invalid.tolist() == [True, False]
    assert out_of_bounds.tolist() == [False, True]
    assert prepared[1, 1] == pytest.approx(1.25)


def test_push_per_step_paths_do_not_materialize_tensors_on_cpu() -> None:
    for function in (
        PushToRegionEnv.evaluate,
        PushToRegionEnv._raw_evaluation,
        PushToRegionEnv._get_obs_extra,
        PushToRegionEnv.step,
    ):
        source = __import__("inspect").getsource(function)
        assert ".cpu(" not in source
        assert ".numpy(" not in source
        assert ".item(" not in source
