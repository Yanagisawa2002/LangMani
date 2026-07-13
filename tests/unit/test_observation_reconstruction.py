from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from langmani.datasets.observation_reconstruction import (
    BASE_CAMERA_HEIGHT,
    BASE_CAMERA_WIDTH,
    ManiSkillObservationReconstructor,
    ObservationReconstructionError,
)
from langmani.datasets.policy_state import PANDA_POLICY_STATE_COMPONENTS
from langmani.environments.pick_place_by_instruction import ENV_ID
from langmani.environments.specs import EpisodeSpec, TaskSpec


@dataclass(frozen=True, slots=True)
class _Joint:
    name: str


class _Robot:
    def __init__(self) -> None:
        self._qpos = np.arange(9, dtype=np.float32)[None, :] / 10.0

    def get_active_joints(self) -> tuple[_Joint, ...]:
        return tuple(_Joint(name) for name in PANDA_POLICY_STATE_COMPONENTS)

    def get_qpos(self) -> np.ndarray:
        return self._qpos


class _Base:
    num_envs = 1
    obs_mode = "rgb"
    control_mode = "pd_joint_pos"
    control_freq = 20

    def __init__(self, camera: dict[str, object]) -> None:
        self.agent = SimpleNamespace(robot=_Robot())
        self.camera = camera
        self.episode_specs: tuple[EpisodeSpec, ...] = ()
        self.restored_states: list[dict[str, Any]] = []
        self.get_obs_calls = 0
        self.step_calls = 0

    def get_episode_specs(self) -> tuple[EpisodeSpec, ...]:
        return self.episode_specs

    def set_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.restored_states.append(state_dict)

    def get_obs(self) -> dict[str, object]:
        self.get_obs_calls += 1
        return {"sensor_data": {"base_camera": self.camera}}

    def step(self, action: object) -> None:
        del action
        self.step_calls += 1
        raise AssertionError("observation reconstruction must not call base.step")


class _Env:
    def __init__(self, camera: dict[str, object]) -> None:
        self.unwrapped = _Base(camera)
        self.reset_calls: list[dict[str, object]] = []
        self.step_calls = 0
        self.closed = False

    def reset(self, *, seed: int, options: dict[str, object]) -> tuple[dict[str, object], dict]:
        self.reset_calls.append({"seed": seed, "options": options})
        task = TaskSpec.from_mapping(options["task_spec"])  # type: ignore[arg-type]
        self.unwrapped.episode_specs = (EpisodeSpec.create(scene_seed=seed, task_spec=task),)
        return {}, {}

    def step(self, action: object) -> None:
        del action
        self.step_calls += 1
        raise AssertionError("observation reconstruction must not call env.step")

    def close(self) -> None:
        self.closed = True


def _rgb_batch() -> np.ndarray:
    rgb = np.zeros((1, BASE_CAMERA_HEIGHT, BASE_CAMERA_WIDTH, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(BASE_CAMERA_WIDTH, dtype=np.uint8)[None, None, :]
    rgb[..., 1] = np.arange(BASE_CAMERA_HEIGHT, dtype=np.uint8)[None, :, None]
    return rgb


def _record(*, scene_seed: int = 17) -> SimpleNamespace:
    task = TaskSpec(
        target_object_id="green_cube",
        target_bin_id="right_bin",
        instruction_template_id="canonical_v0",
    )
    episode = EpisodeSpec.create(scene_seed=scene_seed, task_spec=task)
    return SimpleNamespace(
        raw_trajectory_id="raw-fixture",
        scene_seed=scene_seed,
        scene_id=episode.scene_id,
        task_spec=task,
        task_id=episode.task_id,
        canonical_instruction=episode.canonical_instruction,
    )


def _open_fake(
    monkeypatch: pytest.MonkeyPatch,
    camera: dict[str, object],
) -> tuple[ManiSkillObservationReconstructor, _Env, list[tuple[str, dict[str, object]]]]:
    env = _Env(camera)
    make_calls: list[tuple[str, dict[str, object]]] = []

    def fake_make(environment_id: str, **kwargs: object) -> _Env:
        make_calls.append((environment_id, dict(kwargs)))
        return env

    monkeypatch.setattr(gym, "make", fake_make)
    reconstructor = ManiSkillObservationReconstructor(
        sim_backend="physx_cpu",
        render_backend="sapien_cuda",
    )
    reconstructor.open()
    return reconstructor, env, make_calls


def test_reset_restores_exact_scene_and_task_then_uses_get_obs_without_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_rgb = _rgb_batch()
    reconstructor, env, make_calls = _open_fake(monkeypatch, {"rgb": source_rgb})
    record = _record()
    state_dict = {"actors": {"red_cube": np.arange(13, dtype=np.float32)}}
    try:
        episode = reconstructor.begin_episode(record)  # type: ignore[arg-type]
        frame = reconstructor.reconstruct(state_dict)
    finally:
        reconstructor.close()

    assert make_calls == [
        (
            ENV_ID,
            {
                "num_envs": 1,
                "obs_mode": "rgb",
                "reward_mode": "none",
                "control_mode": "pd_joint_pos",
                "render_mode": None,
                "sim_backend": "physx_cpu",
                "render_backend": "sapien_cuda",
            },
        )
    ]
    assert episode == EpisodeSpec.create(scene_seed=record.scene_seed, task_spec=record.task_spec)
    assert env.reset_calls == [
        {
            "seed": record.scene_seed,
            "options": {"task_spec": record.task_spec.to_dict()},
        }
    ]
    assert env.unwrapped.restored_states == [state_dict]
    assert env.unwrapped.get_obs_calls == 1
    assert env.step_calls == 0
    assert env.unwrapped.step_calls == 0
    assert env.closed
    assert frame.rgb.shape == (BASE_CAMERA_HEIGHT, BASE_CAMERA_WIDTH, 3)
    assert frame.rgb.dtype == np.uint8
    assert frame.rgb.flags.c_contiguous
    assert frame.state.shape == (9,)
    assert frame.state.dtype == np.float32
    np.testing.assert_allclose(frame.state, np.arange(9, dtype=np.float32) / 10.0)

    frame.rgb[0, 0, 0] = 255
    assert source_rgb[0, 0, 0, 0] == 0


@pytest.mark.parametrize(
    ("camera", "message"),
    [
        (
            {"rgb": _rgb_batch(), "depth": np.zeros((1, 256, 256, 1), dtype=np.int16)},
            "RGB only",
        ),
        ({"rgb": np.zeros((256, 256, 3), dtype=np.uint8)}, "shape"),
        ({"rgb": np.zeros((1, 256, 256, 4), dtype=np.uint8)}, "shape"),
        ({"rgb": np.zeros((1, 256, 256, 3), dtype=np.float32)}, "uint8"),
    ],
)
def test_rejects_non_rgb_only_or_invalid_camera_arrays(
    monkeypatch: pytest.MonkeyPatch,
    camera: dict[str, object],
    message: str,
) -> None:
    reconstructor, _env, _make_calls = _open_fake(monkeypatch, camera)
    try:
        reconstructor.begin_episode(_record())  # type: ignore[arg-type]
        with pytest.raises(ObservationReconstructionError, match=message):
            reconstructor.reconstruct({"actors": {"state": np.zeros(13, dtype=np.float32)}})
    finally:
        reconstructor.close()


def test_rejects_reset_episode_spec_disagreement(monkeypatch: pytest.MonkeyPatch) -> None:
    reconstructor, env, _make_calls = _open_fake(monkeypatch, {"rgb": _rgb_batch()})

    def mismatched_reset(
        *, seed: int, options: dict[str, object]
    ) -> tuple[dict[str, object], dict]:
        del options
        other_task = TaskSpec("red_cube", "left_bin", "canonical_v0")
        env.unwrapped.episode_specs = (
            EpisodeSpec.create(scene_seed=seed + 1, task_spec=other_task),
        )
        return {}, {}

    env.reset = mismatched_reset  # type: ignore[method-assign]
    try:
        with pytest.raises(ObservationReconstructionError, match="EpisodeSpec differs"):
            reconstructor.begin_episode(_record())  # type: ignore[arg-type]
    finally:
        reconstructor.close()
