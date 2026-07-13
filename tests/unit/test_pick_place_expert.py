from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from langmani.environments.expert_state import (
    ExpertTaskContext,
    SemanticBinHandle,
    SemanticObjectHandle,
)
from langmani.environments.pick_place_by_instruction import ENV_ID
from langmani.environments.specs import BIN_IDS, OBJECT_IDS, EpisodeSpec, TaskSpec
from langmani.experts.pick_place import (
    PHASE_CONTRACTS,
    PLACEMENT_WALL_SAFETY_MARGIN,
    TOP_GRASP_APPROACH_DIRECTION,
    TOP_GRASP_CLOSING_DIRECTION,
    TOP_GRASP_QUATERNION_WXYZ,
    PickPlaceExpert,
)
from langmani.experts.planner import PlannerFailure, PlannerPlanResult
from langmani.experts.types import (
    EXPERT_PHASE_SEQUENCE,
    ExpertConfig,
    ExpertPhase,
    ExpertStatus,
)

CUBE_HALF_EXTENT = 0.025
BIN_FLOOR_Z = 0.008


class _PoseActor:
    def __init__(self, pose7: tuple[float, ...]) -> None:
        self.pose = SimpleNamespace(raw_pose=np.asarray([pose7], dtype=np.float64))

    def set_pose(self, pose7: np.ndarray) -> None:
        self.pose.raw_pose[0] = pose7


class _FakeRobot:
    def __init__(self) -> None:
        self.qpos = np.zeros((1, 9), dtype=np.float64)

    def get_qpos(self) -> np.ndarray:
        return self.qpos.copy()


class _FakeEnv:
    object_positions = {
        "red_cube": (-0.05, -0.06, CUBE_HALF_EXTENT),
        "green_cube": (0.02, 0.00, CUBE_HALF_EXTENT),
        "blue_cube": (-0.03, 0.07, CUBE_HALF_EXTENT),
    }
    bin_floor_centers = {
        "left_bin": (0.08, 0.18, BIN_FLOOR_Z),
        "right_bin": (0.08, -0.18, BIN_FLOOR_Z),
    }

    def __init__(
        self,
        object_id: str = "red_cube",
        bin_id: str = "left_bin",
        *,
        environment_id: str = ENV_ID,
        num_envs: int = 1,
        control_mode: str = "pd_joint_pos",
        target_handle_object_id: str | None = None,
    ) -> None:
        self.unwrapped = self
        self.num_envs = num_envs
        self.control_mode = control_mode
        self.robot = _FakeRobot()
        self.tcp = _PoseActor((0.0, 0.0, 0.20, 1.0, 0.0, 0.0, 0.0))
        self.agent = SimpleNamespace(robot=self.robot, tcp=self.tcp)
        self.actions: list[np.ndarray] = []
        self.pending_target_pose: np.ndarray | None = None
        self.last_motion_index = 0
        self.target_is_grasped = False
        self.target_in_target_bin = False
        self.target_is_static = False
        self.target_off_table = False
        self.allow_grasp = True
        self.release_in_target = True
        self.force_success_false = False
        self.wrong_object_in_target_bin = False
        self.track_arm = True
        self.lose_grasp_on_motion: int | None = None
        self.off_table_on_step: int | None = None
        self.truncate_on_step: int | None = None
        self.terminate_on_step: int | None = None
        self.terminate_on_success = False
        self.corrupt_tcp_orientation = False

        self.object_actors = {
            semantic_id: _PoseActor((*position, 1.0, 0.0, 0.0, 0.0))
            for semantic_id, position in self.object_positions.items()
        }
        self.bin_actors = {
            semantic_id: _PoseActor((*center, 1.0, 0.0, 0.0, 0.0))
            for semantic_id, center in self.bin_floor_centers.items()
        }
        objects = tuple(
            SemanticObjectHandle(object_id=semantic_id, actor=self.object_actors[semantic_id])
            for semantic_id in OBJECT_IDS
        )
        bins = tuple(
            SemanticBinHandle(
                bin_id=semantic_id,
                actor=self.bin_actors[semantic_id],
                floor_center=np.asarray(self.bin_floor_centers[semantic_id], dtype=np.float64),
            )
            for semantic_id in BIN_IDS
        )
        task_spec = TaskSpec(
            target_object_id=object_id,  # type: ignore[arg-type]
            target_bin_id=bin_id,  # type: ignore[arg-type]
            instruction_template_id="canonical_v0",
        )
        selected_handle_id = target_handle_object_id or object_id
        self.context = ExpertTaskContext(
            environment_id=environment_id,
            episode_spec=EpisodeSpec.create(scene_seed=123, task_spec=task_spec),
            agent=self.agent,
            robot=self.robot,
            target_object=next(
                handle for handle in objects if handle.object_id == selected_handle_id
            ),
            target_bin=next(handle for handle in bins if handle.bin_id == bin_id),
            objects=objects,
            bins=bins,
            cube_half_extent=CUBE_HALF_EXTENT,
            bin_interior_half_size=0.085,
            bin_wall_height=0.03,
            bin_bottom_thickness=0.008,
            table_top_z=0.0,
        )

    @property
    def target_actor(self) -> _PoseActor:
        return self.context.target_object.actor  # type: ignore[return-value]

    def get_expert_task_context(self) -> ExpertTaskContext:
        return self.context

    def get_expert_evaluation(self) -> dict[str, bool]:
        success = (
            self.target_in_target_bin
            and not self.target_is_grasped
            and self.target_is_static
            and not self.wrong_object_in_target_bin
            and not self.force_success_false
        )
        return {
            "target_in_target_bin": self.target_in_target_bin,
            "target_in_wrong_bin": False,
            "wrong_object_in_target_bin": self.wrong_object_in_target_bin,
            "target_is_grasped": self.target_is_grasped,
            "target_is_static": self.target_is_static,
            "target_off_table": self.target_off_table,
            "success": success,
            "fail": self.target_off_table,
        }

    def step(self, action: object) -> tuple[None, float, bool, bool, dict[str, object]]:
        array = np.asarray(action, dtype=np.float64)
        self.actions.append(array.copy())
        if self.track_arm:
            self.robot.qpos[0, :7] = array[:7]

        if self.pending_target_pose is not None:
            target_pose = self.pending_target_pose.copy()
            if self.corrupt_tcp_orientation:
                target_pose[3:] = (1.0, 0.0, 0.0, 0.0)
            self.tcp.set_pose(target_pose)
            self.pending_target_pose = None
            if self.target_is_grasped:
                actor_pose = self.target_actor.pose.raw_pose[0].copy()
                actor_pose[:3] = target_pose[:3]
                self.target_actor.set_pose(actor_pose)

        if self.target_is_grasped and self.lose_grasp_on_motion == self.last_motion_index:
            self.target_is_grasped = False

        gripper = float(array[-1])
        if gripper == -1.0 and self.last_motion_index == 2 and self.allow_grasp:
            self.target_is_grasped = True
        elif gripper == 1.0 and self.target_is_grasped and self.last_motion_index >= 5:
            self.target_is_grasped = False
            if self.release_in_target:
                self.target_in_target_bin = True
                self.target_is_static = True

        step_number = len(self.actions)
        if self.off_table_on_step == step_number:
            self.target_off_table = True
        truncated = self.truncate_on_step == step_number
        terminated = self.terminate_on_step == step_number or (
            self.terminate_on_success and self.get_expert_evaluation()["success"]
        )
        return None, 0.0, terminated, truncated, {}

    def render(self) -> np.ndarray:
        return np.full((16, 16, 3), 96, dtype=np.uint8)


class _FakePlanner:
    def __init__(
        self,
        env: _FakeEnv,
        *,
        failures: tuple[PlannerFailure, ...] = (),
    ) -> None:
        self.env = env
        self.failures = list(failures)
        self.plan_calls: list[tuple[np.ndarray, bool]] = []
        self.successful_motion_count = 0
        self.attach_calls: list[np.ndarray] = []
        self.detach_calls = 0
        self.close_calls = 0
        self.synchronize_calls = 0

    def synchronize(self) -> None:
        self.synchronize_calls += 1

    def plan_pose(
        self,
        pose7: object,
        *,
        use_attached: bool = False,
    ) -> PlannerPlanResult:
        target = np.asarray(pose7, dtype=np.float64)
        self.plan_calls.append((target.copy(), use_attached))
        if self.failures:
            failure = self.failures.pop(0)
            status = (
                "IK Failed! Cannot find valid solution."
                if failure is PlannerFailure.IK_FAILURE
                else "screw plan failed"
            )
            return PlannerPlanResult(False, status, failure, ())

        self.successful_motion_count += 1
        self.env.last_motion_index = self.successful_motion_count
        self.env.pending_target_pose = target.copy()
        waypoint = tuple([self.successful_motion_count / 10.0] * 7)
        return PlannerPlanResult(True, "Success", None, (waypoint,), 0.01)

    def attach_box(self, relative_pose7: object) -> None:
        self.attach_calls.append(np.asarray(relative_pose7, dtype=np.float64).copy())

    def detach(self) -> None:
        self.detach_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def _run(
    env: _FakeEnv,
    *,
    failures: tuple[PlannerFailure, ...] = (),
    config: ExpertConfig | None = None,
) -> tuple[Any, _FakePlanner]:
    planner = _FakePlanner(env, failures=failures)
    result = PickPlaceExpert(
        env,
        config=config,
        planner_factory=lambda _: planner,
    ).run()
    return result, planner


@pytest.mark.parametrize(
    ("object_id", "bin_id"),
    [(object_id, bin_id) for object_id in OBJECT_IDS for bin_id in BIN_IDS],
)
def test_all_six_semantic_tasks_complete_all_twelve_phases(
    object_id: str,
    bin_id: str,
) -> None:
    env = _FakeEnv(object_id, bin_id)
    result, planner = _run(env)

    assert result.success
    assert result.status is ExpertStatus.SUCCESS
    assert result.completed_phases == EXPERT_PHASE_SEQUENCE
    assert tuple(phase.phase for phase in result.phase_results) == EXPERT_PHASE_SEQUENCE
    assert result.failed_phase is None
    assert result.target_object_id == object_id
    assert result.target_bin_id == bin_id
    assert object_id.removesuffix("_cube") in result.canonical_instruction
    assert bin_id.removesuffix("_bin") in result.canonical_instruction

    selected_object = np.asarray(env.object_positions[object_id])
    selected_bin = np.asarray(env.bin_floor_centers[bin_id])
    np.testing.assert_allclose(planner.plan_calls[0][0][:2], selected_object[:2])
    np.testing.assert_allclose(planner.plan_calls[1][0][:3], selected_object)
    np.testing.assert_allclose(planner.plan_calls[4][0][:2], selected_bin[:2])
    assert planner.plan_calls[4][0][2] == pytest.approx(selected_bin[2] + CUBE_HALF_EXTENT + 0.002)
    for target_pose, _ in planner.plan_calls:
        np.testing.assert_allclose(target_pose[3:], TOP_GRASP_QUATERNION_WXYZ)

    assert planner.attach_calls and planner.detach_calls == 1 and planner.close_calls == 1
    assert planner.synchronize_calls == 1
    assert env.actions
    assert all(action.shape == (8,) for action in env.actions)
    assert {float(action[-1]) for action in env.actions} == {-1.0, 1.0}


def test_top_grasp_and_placement_geometry_contracts() -> None:
    assert TOP_GRASP_APPROACH_DIRECTION == (0.0, 0.0, -1.0)
    assert TOP_GRASP_CLOSING_DIRECTION == (0.0, -1.0, 0.0)
    assert TOP_GRASP_QUATERNION_WXYZ == (0.0, 1.0, 0.0, 0.0)
    assert pytest.approx(0.01) == PLACEMENT_WALL_SAFETY_MARGIN
    approach = np.asarray(TOP_GRASP_APPROACH_DIRECTION)
    closing = np.asarray(TOP_GRASP_CLOSING_DIRECTION)
    grasp_rotation = np.stack((np.cross(closing, approach), closing, approach), axis=1)
    np.testing.assert_allclose(grasp_rotation, np.diag([1.0, -1.0, -1.0]))

    env = _FakeEnv()
    env.context = replace(
        env.context,
        bin_interior_half_size=CUBE_HALF_EXTENT + PLACEMENT_WALL_SAFETY_MARGIN,
    )
    result, planner = _run(env)

    assert not result.success
    assert result.status is ExpertStatus.INITIALIZATION_FAILURE
    assert result.failed_phase is ExpertPhase.INITIALIZE
    assert "wall margin" in result.phase_results[-1].message
    assert planner.plan_calls == []


@pytest.mark.parametrize(
    ("env", "expected_status", "message_fragment"),
    [
        (
            _FakeEnv(environment_id="Not-LangMani-v0"),
            ExpertStatus.INITIALIZATION_FAILURE,
            "environment",
        ),
        (_FakeEnv(num_envs=2), ExpertStatus.INITIALIZATION_FAILURE, "num_envs=1"),
        (
            _FakeEnv(control_mode="pd_joint_delta_pos"),
            ExpertStatus.INITIALIZATION_FAILURE,
            "control_mode",
        ),
        (
            _FakeEnv("red_cube", target_handle_object_id="green_cube"),
            ExpertStatus.INVALID_TASK,
            "TaskSpec",
        ),
    ],
)
def test_initialization_rejects_environment_and_semantic_contract_mismatches(
    env: _FakeEnv,
    expected_status: ExpertStatus,
    message_fragment: str,
) -> None:
    result, planner = _run(env)

    assert not result.success
    assert result.status is expected_status
    assert result.failed_phase is ExpertPhase.INITIALIZE
    assert message_fragment in result.phase_results[-1].message
    assert planner.plan_calls == []


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        (PlannerFailure.IK_FAILURE, ExpertStatus.IK_FAILURE),
        (PlannerFailure.PLANNING_FAILURE, ExpertStatus.PLANNING_FAILURE),
    ],
)
def test_planner_failure_classification(
    failure: PlannerFailure,
    expected_status: ExpertStatus,
) -> None:
    result, planner = _run(_FakeEnv(), failures=(failure,))

    assert not result.success
    assert result.status is expected_status
    assert result.failed_phase is ExpertPhase.MOVE_TO_PREGRASP
    assert result.total_planning_calls == 1
    assert planner.close_calls == 1


@pytest.mark.parametrize(
    ("setup", "config", "expected_phase", "expected_status"),
    [
        (
            lambda env: setattr(env, "allow_grasp", False),
            None,
            ExpertPhase.VERIFY_GRASP,
            ExpertStatus.GRASP_FAILURE,
        ),
        (
            lambda env: setattr(env, "lose_grasp_on_motion", 3),
            None,
            ExpertPhase.LIFT_TARGET,
            ExpertStatus.TRANSPORT_FAILURE,
        ),
        (
            lambda env: setattr(env, "lose_grasp_on_motion", 5),
            None,
            ExpertPhase.DESCEND_TO_PLACE,
            ExpertStatus.PLACEMENT_FAILURE,
        ),
        (
            lambda env: setattr(env, "release_in_target", False),
            None,
            ExpertPhase.SETTLE_AFTER_RELEASE,
            ExpertStatus.PLACEMENT_FAILURE,
        ),
        (
            lambda env: setattr(env, "wrong_object_in_target_bin", True),
            None,
            ExpertPhase.SETTLE_AFTER_RELEASE,
            ExpertStatus.PLACEMENT_FAILURE,
        ),
        (
            lambda env: setattr(env, "off_table_on_step", 1),
            None,
            ExpertPhase.MOVE_TO_PREGRASP,
            ExpertStatus.TARGET_OFF_TABLE,
        ),
        (
            lambda env: None,
            ExpertConfig(max_episode_steps=1),
            ExpertPhase.MOVE_TO_PREGRASP,
            ExpertStatus.TIMEOUT,
        ),
        (
            lambda env: setattr(env, "force_success_false", True),
            None,
            ExpertPhase.VERIFY_TASK,
            ExpertStatus.VERIFICATION_FAILURE,
        ),
        (
            lambda env: setattr(env, "track_arm", False),
            None,
            ExpertPhase.MOVE_TO_PREGRASP,
            ExpertStatus.EXECUTION_FAILURE,
        ),
        (
            lambda env: setattr(env, "terminate_on_step", 1),
            None,
            ExpertPhase.MOVE_TO_PREGRASP,
            ExpertStatus.EXECUTION_FAILURE,
        ),
        (
            lambda env: setattr(env, "corrupt_tcp_orientation", True),
            None,
            ExpertPhase.MOVE_TO_PREGRASP,
            ExpertStatus.EXECUTION_FAILURE,
        ),
    ],
)
def test_execution_failure_classification(
    setup: Any,
    config: ExpertConfig | None,
    expected_phase: ExpertPhase,
    expected_status: ExpertStatus,
) -> None:
    env = _FakeEnv()
    setup(env)
    result, _ = _run(env, config=config)

    assert not result.success
    assert result.status is expected_status
    assert result.failed_phase is expected_phase
    assert result.phase_results[-1].phase is expected_phase
    assert result.phase_results[-1].status is expected_status


def test_planning_attempt_limit_and_replan_accounting() -> None:
    config = ExpertConfig(max_planning_attempts_per_phase=3)
    result, planner = _run(
        _FakeEnv(),
        failures=(PlannerFailure.PLANNING_FAILURE, PlannerFailure.PLANNING_FAILURE),
        config=config,
    )

    assert result.success
    pregrasp = next(
        phase for phase in result.phase_results if phase.phase is ExpertPhase.MOVE_TO_PREGRASP
    )
    assert pregrasp.attempts == 3
    assert pregrasp.planning_calls == 3
    assert pregrasp.replans == 2
    assert result.total_planning_calls == 8
    assert result.total_replans == 2
    assert len(planner.plan_calls) == 8


def test_success_terminal_signal_is_explicitly_allowed_until_retreat_completes() -> None:
    env = _FakeEnv()
    env.terminate_on_success = True

    result, _ = _run(env)

    assert result.success
    assert result.completed_phases == EXPERT_PHASE_SEQUENCE


def test_phase_contracts_cover_the_complete_stable_sequence() -> None:
    assert tuple(PHASE_CONTRACTS) == EXPERT_PHASE_SEQUENCE
    assert set(PHASE_CONTRACTS) == set(ExpertPhase)
    for phase in EXPERT_PHASE_SEQUENCE:
        contract = PHASE_CONTRACTS[phase]
        assert contract.entry_condition
        assert contract.command
        assert contract.completion_criterion
        assert contract.failure_status is not ExpertStatus.SUCCESS


def test_result_payload_is_json_ready_and_contains_no_runtime_objects() -> None:
    result, _ = _run(_FakeEnv("blue_cube", "right_bin"))
    payload = result.to_dict()

    assert json.loads(json.dumps(payload)) == payload

    def assert_compact_json_value(value: object) -> None:
        assert not isinstance(value, (np.ndarray, Path))
        if isinstance(value, dict):
            for item in value.values():
                assert_compact_json_value(item)
        elif isinstance(value, list):
            for item in value:
                assert_compact_json_value(item)

    assert_compact_json_value(payload)
    assert "positions" not in json.dumps(payload)
    assert "trajectory" not in json.dumps(payload)


def test_diagnostic_rendering_writes_one_compact_png_per_completed_phase() -> None:
    frame_directory = (
        Path(__file__).resolve().parents[2] / "outputs" / "diagnostics" / "m2" / "unit_test_frames"
    )
    result = PickPlaceExpert(
        _FakeEnv(),
        config=ExpertConfig(diagnostic_rendering=True),
        planner_factory=lambda env: _FakePlanner(env),  # type: ignore[arg-type]
        diagnostic_directory=frame_directory,
    ).run()

    assert result.success
    assert len(result.diagnostic_artifact_paths) == len(EXPERT_PHASE_SEQUENCE) == 12
    for artifact_path in result.diagnostic_artifact_paths:
        path = Path(artifact_path)
        assert path.parent == frame_directory
        assert path.suffix == ".png"
        assert path.is_file() and path.stat().st_size > 0


def test_unknown_exception_escapes_core_and_outer_boundary_preserves_details() -> None:
    env = _FakeEnv()

    class ExplodingPlanner(_FakePlanner):
        def plan_pose(
            self,
            pose7: object,
            *,
            use_attached: bool = False,
        ) -> PlannerPlanResult:
            del pose7, use_attached
            raise ArithmeticError("deliberate planner defect")

    planner = ExplodingPlanner(env)
    expert = PickPlaceExpert(env, planner_factory=lambda _: planner)

    with pytest.raises(ArithmeticError, match="deliberate planner defect") as caught:
        expert.run()

    result = expert.unexpected_exception_result(caught.value)
    assert result.status is ExpertStatus.UNEXPECTED_EXCEPTION
    assert result.exception_type == "ArithmeticError"
    assert result.exception_message == "deliberate planner defect"
    assert result.completed_phases == (ExpertPhase.INITIALIZE,)
    assert result.failed_phase is ExpertPhase.MOVE_TO_PREGRASP
    assert result.total_planning_calls == 1
    assert result.planning_duration_seconds > 0.0
    assert result.execution_duration_seconds >= 0.0
    assert json.loads(json.dumps(result.to_dict())) == result.to_dict()


def test_incompatible_planner_factory_remains_a_clean_initialization_failure() -> None:
    result = PickPlaceExpert(_FakeEnv(), planner_factory=lambda _: object()).run()

    assert result.status is ExpertStatus.INITIALIZATION_FAILURE
    assert result.failed_phase is ExpertPhase.INITIALIZE
    assert "incompatible adapter" in result.phase_results[-1].message


def test_malformed_step_result_counts_the_already_executed_environment_step() -> None:
    env = _FakeEnv()

    def malformed_step(action: object) -> tuple[None]:
        env.actions.append(np.asarray(action))
        return (None,)

    env.step = malformed_step  # type: ignore[method-assign]
    planner = _FakePlanner(env)
    expert = PickPlaceExpert(env, planner_factory=lambda _: planner)

    with pytest.raises(RuntimeError, match="five-tuple") as caught:
        expert.run()

    result = expert.unexpected_exception_result(caught.value)
    assert result.total_environment_steps == 1
    assert result.total_planning_calls == 1
    assert result.failed_phase is ExpertPhase.MOVE_TO_PREGRASP
