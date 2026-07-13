"""CPU-only tests for project-owned M3A action replay validation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pytest

from langmani.collection.replay import ReplayContractError, validate_action_replay
from langmani.datasets.schedule import build_collection_schedule
from langmani.datasets.types import CollectionConfig, ReplayFailureCode
from langmani.environments.specs import EpisodeSpec, TaskSpec


class _ReplayEnv:
    def __init__(self, *, replay_final_x: float = 0.2) -> None:
        self.unwrapped = self
        self.control_mode = "pd_joint_pos"
        self.single_action_space = SimpleNamespace(
            low=np.full(8, -1.0, dtype=np.float32),
            high=np.full(8, 1.0, dtype=np.float32),
        )
        self.replay_final_x = replay_final_x
        self.target_actor = SimpleNamespace(
            pose=SimpleNamespace(raw_pose=np.array([[0.0, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]]))
        )
        self.robot = SimpleNamespace(get_qpos=lambda: self.qpos.copy())
        self.qpos = np.zeros((1, 9), dtype=np.float64)
        self.episode: EpisodeSpec | None = None
        self.steps = 0
        self.success = False
        self.closed = False
        self.received_actions: list[np.ndarray] = []

    def reset(self, *, seed: int, options: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        task = TaskSpec.from_mapping(options["task_spec"])
        self.episode = EpisodeSpec.create(scene_seed=seed, task_spec=task)
        self.steps = 0
        self.success = False
        self.target_actor.pose.raw_pose[0] = [0.0, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]
        self.qpos.fill(0.0)
        return {}, {}

    def step(self, action: object) -> tuple[None, float, bool, bool, dict[str, bool]]:
        self.received_actions.append(np.array(action, copy=True))
        self.steps += 1
        if self.steps == 2:
            self.target_actor.pose.raw_pose[0, 0] = self.replay_final_x
            self.qpos[0] = np.arange(9) / 100
            self.success = True
        return None, 0.0, self.success, False, self.get_expert_evaluation()

    def get_episode_specs(self) -> tuple[EpisodeSpec, ...]:
        assert self.episode is not None
        return (self.episode,)

    def get_expert_task_context(self) -> Any:
        return SimpleNamespace(
            target_object=SimpleNamespace(actor=self.target_actor),
            robot=self.robot,
        )

    def get_expert_evaluation(self) -> dict[str, bool]:
        return {
            "target_in_target_bin": self.success,
            "target_in_wrong_bin": False,
            "wrong_object_in_target_bin": False,
            "target_is_grasped": False,
            "target_is_static": self.success,
            "target_off_table": False,
            "success": self.success,
            "fail": False,
        }

    def set_state_dict(self, state: dict[str, Any]) -> None:
        actor = np.asarray(state["actors"]["red_cube"])
        self.target_actor.pose.raw_pose[0] = actor[:7]
        self.qpos[0] = np.asarray(state["articulations"]["panda"])
        self.success = bool(np.isclose(actor[0], 0.2))

    def get_state_dict(self) -> dict[str, Any]:
        actor = np.zeros(13, dtype=np.float64)
        actor[:7] = self.target_actor.pose.raw_pose[0]
        return {
            "actors": {"red_cube": actor},
            "articulations": {"panda": self.qpos[0].copy()},
        }

    def close(self) -> None:
        self.closed = True


def _write_replay_pair(
    path: Path,
    task_spec: TaskSpec,
    *,
    wrong_task: bool = False,
    actions: np.ndarray | None = None,
    action_dtype: str = "float32",
    extra_state: bool = False,
    short_terminated: bool = False,
) -> None:
    if actions is None:
        actions = np.zeros((2, 8), dtype=action_dtype)
    else:
        actions = np.asarray(actions, dtype=action_dtype)
    transition_count = int(actions.shape[0])
    state_count = transition_count + 1 + int(extra_state)
    with h5py.File(path, "w") as archive:
        group = archive.create_group("traj_0")
        group.create_dataset("actions", data=actions)
        terminated = np.zeros(transition_count, dtype=bool)
        terminated[-1] = True
        if short_terminated:
            terminated = terminated[:-1]
        group.create_dataset("terminated", data=terminated)
        group.create_dataset("truncated", data=np.zeros(transition_count, dtype=bool))
        success = np.zeros(transition_count, dtype=bool)
        success[-1] = True
        group.create_dataset("success", data=success)
        group.create_dataset("fail", data=np.zeros(transition_count, dtype=bool))
        states = group.create_group("env_states")
        actors = states.create_group("actors")
        red = np.zeros((state_count, 13), dtype=np.float32)
        red[:, 2] = 0.025
        red[:, 3] = 1.0
        red[:, 0] = np.linspace(0.0, 0.2, state_count)
        actors.create_dataset("red_cube", data=red)
        articulations = states.create_group("articulations")
        panda = np.zeros((state_count, 9), dtype=np.float32)
        panda[-1] = np.arange(9) / 100
        articulations.create_dataset("panda", data=panda)
    recorded_task = (
        TaskSpec(
            target_object_id="green_cube",
            target_bin_id="left_bin",
            instruction_template_id="canonical_v0",
        )
        if wrong_task
        else task_spec
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "episode_id": 0,
                        "episode_seed": 0,
                        "control_mode": "pd_joint_pos",
                        "elapsed_steps": transition_count,
                        "reset_kwargs": {
                            "seed": 0,
                            "options": {"task_spec": recorded_task.to_dict()},
                        },
                        "success": True,
                        "fail": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _scheduled(tmp_path: Path) -> tuple[CollectionConfig, Any]:
    config = CollectionConfig(
        target_complete_scene_count=1,
        maximum_candidate_scene_count=1,
        raw_output_root=str(tmp_path / "raw"),
        shard_size=6,
    )
    return config, build_collection_schedule(config).episodes[0]


def test_action_replay_and_state_audit_pass_within_final_tolerances(tmp_path: Path) -> None:
    config, scheduled = _scheduled(tmp_path)
    path = tmp_path / "episode.h5"
    _write_replay_pair(path, scheduled.task_spec)
    env = _ReplayEnv()

    result = validate_action_replay(
        path,
        native_episode_id=0,
        scheduled_episode=scheduled,
        config=config,
        sim_backend="physx_cpu",
        environment_factory=lambda _: env,
    )

    assert result.passed
    assert result.recorded_action_steps == result.replayed_action_steps == 2
    assert result.task_spec_matches
    assert result.state_audit_performed and result.state_audit_passed
    assert result.final_position_error_m == pytest.approx(0.0, abs=1e-7)
    assert result.final_joint_error_rad == pytest.approx(0.0, abs=1e-7)
    assert result.failure_codes == ()
    assert env.closed


def test_action_replay_rejects_final_pose_drift(tmp_path: Path) -> None:
    config, scheduled = _scheduled(tmp_path)
    path = tmp_path / "episode.h5"
    _write_replay_pair(path, scheduled.task_spec)

    result = validate_action_replay(
        path,
        native_episode_id=0,
        scheduled_episode=scheduled,
        config=config,
        sim_backend="physx_cpu",
        environment_factory=lambda _: _ReplayEnv(replay_final_x=0.25),
    )

    assert not result.passed
    assert result.replay_success
    assert result.final_position_error_m == pytest.approx(0.05)
    assert ReplayFailureCode.FINAL_POSITION_TOLERANCE in result.failure_codes
    assert any("position" in reason and "exceeds" in reason for reason in result.failure_reasons)


def test_replay_rejects_recorded_task_mismatch_before_simulation(tmp_path: Path) -> None:
    config, scheduled = _scheduled(tmp_path)
    path = tmp_path / "episode.h5"
    _write_replay_pair(path, scheduled.task_spec, wrong_task=True)

    with pytest.raises(ReplayContractError, match="differs from the schedule"):
        validate_action_replay(
            path,
            native_episode_id=0,
            scheduled_episode=scheduled,
            config=config,
            sim_backend="physx_cpu",
            environment_factory=lambda _: (_ for _ in ()).throw(
                AssertionError("environment must not be created")
            ),
        )


def test_replay_rejects_seeded_initial_scene_drift(tmp_path: Path) -> None:
    config, scheduled = _scheduled(tmp_path)
    path = tmp_path / "episode.h5"
    _write_replay_pair(path, scheduled.task_spec)

    class DriftedResetEnv(_ReplayEnv):
        def get_state_dict(self) -> dict[str, Any]:
            state = super().get_state_dict()
            if self.steps == 0:
                state["actors"]["red_cube"][1] = 0.02
            return state

    result = validate_action_replay(
        path,
        native_episode_id=0,
        scheduled_episode=scheduled,
        config=config,
        sim_backend="physx_cpu",
        environment_factory=lambda _: DriftedResetEnv(),
    )

    assert not result.passed
    assert ReplayFailureCode.INITIAL_STATE_MISMATCH in result.failure_codes
    assert any("initial-state" in reason for reason in result.failure_reasons)


def test_state_audit_rejects_non_round_tripping_intermediate_state(tmp_path: Path) -> None:
    config, scheduled = _scheduled(tmp_path)
    path = tmp_path / "episode.h5"
    _write_replay_pair(path, scheduled.task_spec)

    class BadRoundTripEnv(_ReplayEnv):
        def set_state_dict(self, state: dict[str, Any]) -> None:
            super().set_state_dict(state)
            actor = np.asarray(state["actors"]["red_cube"])
            if np.isclose(actor[0], 0.1):
                self.target_actor.pose.raw_pose[0, 1] = 0.01

    result = validate_action_replay(
        path,
        native_episode_id=0,
        scheduled_episode=scheduled,
        config=config,
        sim_backend="physx_cpu",
        environment_factory=lambda _: BadRoundTripEnv(),
    )

    assert not result.passed
    assert result.state_audit_performed and result.state_audit_passed is False
    assert ReplayFailureCode.STATE_AUDIT_FAILURE in result.failure_codes
    assert any("recorded state 1" in reason for reason in result.failure_reasons)


def test_replay_passes_each_recorded_action_unchanged_and_steps_exactly_t_times(
    tmp_path: Path,
) -> None:
    config, scheduled = _scheduled(tmp_path)
    path = tmp_path / "episode.h5"
    actions = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8],
        ],
        dtype=np.float32,
    )
    _write_replay_pair(path, scheduled.task_spec, actions=actions)
    env = _ReplayEnv()

    result = validate_action_replay(
        path,
        native_episode_id=0,
        scheduled_episode=scheduled,
        config=config,
        sim_backend="physx_cpu",
        environment_factory=lambda _: env,
    )

    assert result.passed
    assert len(env.received_actions) == len(actions)
    for received, recorded in zip(env.received_actions, actions, strict=True):
        assert received.dtype == np.float32
        assert received.shape == (8,)
        np.testing.assert_array_equal(received, recorded)


def test_replay_rejects_out_of_bounds_action_without_clipping_or_stepping_it(
    tmp_path: Path,
) -> None:
    config, scheduled = _scheduled(tmp_path)
    path = tmp_path / "episode.h5"
    actions = np.zeros((2, 8), dtype=np.float32)
    actions[0, 3] = 1.25
    _write_replay_pair(path, scheduled.task_spec, actions=actions)
    env = _ReplayEnv()

    result = validate_action_replay(
        path,
        native_episode_id=0,
        scheduled_episode=scheduled,
        config=config,
        sim_backend="physx_cpu",
        environment_factory=lambda _: env,
    )

    assert not result.passed
    assert result.replayed_action_steps == 0
    assert env.received_actions == []
    assert ReplayFailureCode.ACTION_OUT_OF_BOUNDS in result.failure_codes
    assert ReplayFailureCode.ACTION_COUNT_MISMATCH in result.failure_codes
    assert any("value 1.25" in reason for reason in result.failure_reasons)


@pytest.mark.parametrize(
    ("write_kwargs", "message"),
    [
        ({"action_dtype": "float64"}, "float32 without conversion"),
        ({"extra_state": True}, "exactly 3 states"),
        ({"short_terminated": True}, "terminated must have shape"),
    ],
)
def test_replay_rejects_malformed_time_contract_before_simulation(
    tmp_path: Path,
    write_kwargs: dict[str, object],
    message: str,
) -> None:
    config, scheduled = _scheduled(tmp_path)
    path = tmp_path / "episode.h5"
    _write_replay_pair(path, scheduled.task_spec, **write_kwargs)  # type: ignore[arg-type]

    with pytest.raises(ReplayContractError, match=message) as captured:
        validate_action_replay(
            path,
            native_episode_id=0,
            scheduled_episode=scheduled,
            config=config,
            sim_backend="physx_cpu",
            environment_factory=lambda _: (_ for _ in ()).throw(
                AssertionError("environment must not be created")
            ),
        )

    assert captured.value.failure_code is ReplayFailureCode.RECORDED_TRAJECTORY_INVALID
