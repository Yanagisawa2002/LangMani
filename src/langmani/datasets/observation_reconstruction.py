"""Deterministic M3B policy-observation reconstruction from M3A states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from langmani.datasets.policy_state import extract_panda_policy_state_v0
from langmani.datasets.types import RawEpisodeRecord
from langmani.environments.pick_place_by_instruction import ENV_ID
from langmani.environments.specs import EpisodeSpec

BASE_CAMERA_NAME = "base_camera"
BASE_CAMERA_HEIGHT = 256
BASE_CAMERA_WIDTH = 256


class ObservationReconstructionError(RuntimeError):
    """Raised when a recorded state cannot produce the fixed policy observation."""


@dataclass(frozen=True, slots=True)
class ReconstructedPolicyFrame:
    """One pre-action M3B frame in writer-ready dtypes and layouts."""

    rgb: npt.NDArray[np.uint8]
    state: npt.NDArray[np.float32]


class ManiSkillObservationReconstructor:
    """Fresh single-environment renderer using only public ManiSkill APIs.

    Each episode is reset semantically before state restoration because a raw
    simulator state does not contain the active TaskSpec or static bin actors.
    ``BaseEnv.get_obs()`` performs renderer synchronization and camera capture;
    no artificial environment step is used.
    """

    def __init__(self, *, sim_backend: str, render_backend: str = "sapien_cuda") -> None:
        if sim_backend not in {"physx_cpu", "physx_cuda"}:
            raise ValueError("sim_backend must be 'physx_cpu' or 'physx_cuda'")
        if not isinstance(render_backend, str) or not render_backend:
            raise TypeError("render_backend must be a non-empty string")
        self._sim_backend = sim_backend
        self._render_backend = render_backend
        self._env: Any | None = None
        self._base: Any | None = None
        self._active_episode: EpisodeSpec | None = None

    def __enter__(self) -> ManiSkillObservationReconstructor:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()

    def open(self) -> None:
        if self._env is not None:
            raise ObservationReconstructionError("reconstructor is already open")
        try:
            import gymnasium as gym

            import langmani.environments  # noqa: F401 - performs exact Gym registration

            env = gym.make(
                ENV_ID,
                num_envs=1,
                obs_mode="rgb",
                reward_mode="none",
                control_mode="pd_joint_pos",
                render_mode=None,
                sim_backend=self._sim_backend,
                render_backend=self._render_backend,
            )
        except Exception as error:
            raise ObservationReconstructionError(
                f"failed to create M3B rendering environment: {type(error).__name__}: {error}"
            ) from error
        base = env.unwrapped
        if getattr(base, "num_envs", None) != 1:
            env.close()
            raise ObservationReconstructionError("M3B reconstruction requires num_envs=1")
        if getattr(base, "obs_mode", None) != "rgb":
            env.close()
            raise ObservationReconstructionError("M3B reconstruction requires obs_mode='rgb'")
        if getattr(base, "control_mode", None) != "pd_joint_pos":
            env.close()
            raise ObservationReconstructionError(
                "M3B reconstruction requires control_mode='pd_joint_pos'"
            )
        if getattr(base, "control_freq", None) != 20:
            env.close()
            raise ObservationReconstructionError(
                f"M3B reconstruction requires control_freq=20, got "
                f"{getattr(base, 'control_freq', None)!r}"
            )
        self._env = env
        self._base = base

    def begin_episode(self, record: RawEpisodeRecord) -> EpisodeSpec:
        """Reset the exact scene/task and verify the public EpisodeSpec accessor."""
        base = self._require_base()
        expected = EpisodeSpec.create(scene_seed=record.scene_seed, task_spec=record.task_spec)
        if (
            record.scene_id != expected.scene_id
            or record.task_id != expected.task_id
            or record.canonical_instruction != expected.canonical_instruction
        ):
            raise ObservationReconstructionError("raw episode semantic metadata is inconsistent")
        try:
            assert self._env is not None
            self._env.reset(
                seed=record.scene_seed,
                options={"task_spec": record.task_spec.to_dict()},
            )
            specs = base.get_episode_specs()
        except Exception as error:
            raise ObservationReconstructionError(
                f"failed to reset source episode {record.raw_trajectory_id}: "
                f"{type(error).__name__}: {error}"
            ) from error
        if specs != (expected,):
            raise ObservationReconstructionError(
                "reset EpisodeSpec differs from accepted M3A source metadata"
            )
        self._active_episode = expected
        return expected

    def reconstruct(self, state_dict: dict[str, Any]) -> ReconstructedPolicyFrame:
        """Restore one state[t] and return RGB HWC uint8 plus stable Panda qpos."""
        base = self._require_base()
        if self._active_episode is None:
            raise ObservationReconstructionError("begin_episode must precede reconstruction")
        if not isinstance(state_dict, dict) or not state_dict:
            raise ObservationReconstructionError("recorded environment state must be a mapping")
        try:
            base.set_state_dict(state_dict)
            observation = base.get_obs()
        except Exception as error:
            raise ObservationReconstructionError(
                f"state restoration or sensor capture failed: {type(error).__name__}: {error}"
            ) from error
        rgb = extract_base_camera_rgb(observation)
        try:
            policy_state = extract_panda_policy_state_v0(base.agent.robot)
        except (AttributeError, TypeError, ValueError) as error:
            raise ObservationReconstructionError(
                f"PandaPolicyStateV0 extraction failed: {error}"
            ) from error
        return ReconstructedPolicyFrame(rgb=rgb, state=policy_state)

    def close(self) -> None:
        env = self._env
        self._env = None
        self._base = None
        self._active_episode = None
        if env is not None:
            env.close()

    def _require_base(self) -> Any:
        if self._base is None:
            raise ObservationReconstructionError("reconstructor is not open")
        return self._base


def extract_base_camera_rgb(observation: object) -> npt.NDArray[np.uint8]:
    """Extract the deployed-policy RGB frame from a single M1 observation."""
    if not isinstance(observation, dict):
        raise ObservationReconstructionError("rgb observation must be a dictionary")
    sensor_data = observation.get("sensor_data")
    if not isinstance(sensor_data, dict):
        raise ObservationReconstructionError("rgb observation is missing sensor_data")
    camera = sensor_data.get(BASE_CAMERA_NAME)
    if not isinstance(camera, dict):
        raise ObservationReconstructionError("rgb observation is missing base_camera")
    if set(camera) != {"rgb"}:
        raise ObservationReconstructionError(
            "base_camera must expose RGB only; unexpected modalities=" + repr(sorted(camera))
        )
    array = _as_numpy(camera["rgb"])
    expected_shape = (1, BASE_CAMERA_HEIGHT, BASE_CAMERA_WIDTH, 3)
    if array.shape != expected_shape:
        raise ObservationReconstructionError(
            f"base_camera RGB must have shape {expected_shape}, got {array.shape}"
        )
    if array.dtype != np.dtype(np.uint8):
        raise ObservationReconstructionError(f"base_camera RGB must use uint8, got {array.dtype}")
    return np.array(array[0], dtype=np.uint8, copy=True, order="C")


def _as_numpy(value: object) -> np.ndarray:
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    numpy = getattr(candidate, "numpy", None)
    if callable(numpy):
        candidate = numpy()
    try:
        return np.asarray(candidate)
    except (TypeError, ValueError) as error:
        raise ObservationReconstructionError("base_camera RGB is not numeric array data") from error


__all__ = [
    "BASE_CAMERA_HEIGHT",
    "BASE_CAMERA_NAME",
    "BASE_CAMERA_WIDTH",
    "ManiSkillObservationReconstructor",
    "ObservationReconstructionError",
    "ReconstructedPolicyFrame",
    "extract_base_camera_rgb",
]
