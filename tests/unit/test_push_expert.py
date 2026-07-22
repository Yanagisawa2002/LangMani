from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from environment.benchmark_push_expert import ScheduledPushEpisode, _run_schedule, build_schedule
from langmani.environments.push_expert_state import (
    PushExpertTaskContext,
    PushObjectHandle,
)
from langmani.environments.push_specs import PushEpisodeSpec, PushTaskSpec
from langmani.environments.push_to_region import ENV_ID
from langmani.experts.planner import PlannerPlanResult
from langmani.experts.push import PushToRegionExpert
from langmani.experts.push_types import (
    PUSH_EXPERT_PHASE_SEQUENCE,
    PushExpertConfig,
    PushExpertPhase,
    PushExpertStatus,
)


class _Pose:
    def __init__(self, value: list[float]) -> None:
        self.raw_pose = torch.tensor([value], dtype=torch.float32)


class _Robot:
    def __init__(self) -> None:
        self._qpos = torch.zeros((1, 9), dtype=torch.float32)

    def get_qpos(self) -> torch.Tensor:
        return self._qpos


class _FakeEnvironment:
    def __init__(
        self,
        *,
        wrong_object_after_step: bool = False,
        inside_after_step: int = 9,
        success_after_step: int = 9,
    ) -> None:
        self.unwrapped = self
        self.num_envs = 1
        self.control_mode = "pd_joint_pos"
        self.step_count = 0
        self.wrong_object_after_step = wrong_object_after_step
        self.inside_after_step = inside_after_step
        self.success_after_step = success_after_step
        self.robot = _Robot()
        self.tcp = SimpleNamespace(pose=_Pose([0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]))
        self.agent = SimpleNamespace(robot=self.robot, tcp=self.tcp)
        self.target = SimpleNamespace(pose=_Pose([-0.2, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]))
        self.distractor = SimpleNamespace(pose=_Pose([-0.28, 0.24, 0.025, 1.0, 0.0, 0.0, 0.0]))
        episode = PushEpisodeSpec.create(
            scene_seed=77,
            task_spec=PushTaskSpec("blue_cube", "left", "standard"),
        )
        handles = (
            PushObjectHandle("blue_cube", self.target),
            PushObjectHandle("orange_cylinder", self.distractor),
        )
        self.context = PushExpertTaskContext(
            environment_id=ENV_ID,
            episode_spec=episode,
            agent=self.agent,
            robot=self.robot,
            target_object=handles[0],
            objects=handles,
            target_region_center=torch.tensor([0.2, 0.0, 0.001]),
            target_region_radius=0.11,
            target_object_planar_radius=0.035,
            target_object_resting_height=0.025,
            full_containment_center_radius=0.07,
            table_top_z=0.0,
        )
        self.action_low = np.asarray([-3.0] * 7 + [-1.0])
        self.action_high = np.asarray([3.0] * 7 + [1.0])

    def get_push_expert_task_context(self) -> PushExpertTaskContext:
        return self.context

    def get_push_expert_evaluation(self) -> dict[str, torch.Tensor]:
        success = self.step_count >= self.success_after_step
        inside = self.step_count >= self.inside_after_step
        return {
            "success": torch.tensor([success]),
            "target_inside_region": torch.tensor([inside]),
            "target_is_static": torch.tensor([True]),
            "stable_success_steps": torch.tensor([5 if success else 0]),
            "target_distance": torch.tensor([0.0 if success else 0.4]),
            "target_outside_workspace": torch.tensor([False]),
            "target_lifted": torch.tensor([False]),
            "target_toppled": torch.tensor([False]),
            "target_is_grasped": torch.tensor([False]),
            "wrong_object_displaced": torch.tensor(
                [self.wrong_object_after_step and self.step_count > 0]
            ),
            "invalid_action": torch.tensor([False]),
            "action_out_of_bounds": torch.tensor([False]),
        }

    def get_push_expert_action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.action_low.copy(), self.action_high.copy()

    def step(self, action: np.ndarray):
        assert action.shape == (8,)
        self.step_count += 1
        evaluation = self.get_push_expert_evaluation()
        return {}, 0.0, evaluation["success"], torch.tensor([False]), evaluation


class _FakePlanner:
    def __init__(self, environment: _FakeEnvironment) -> None:
        self.environment = environment
        self.plan_count = 0
        self.closed = False

    def synchronize(self) -> None:
        return None

    def plan_pose(self, pose7, *, use_attached: bool = False) -> PlannerPlanResult:
        del use_attached
        self.plan_count += 1
        self.environment.tcp.pose.raw_pose = torch.tensor([pose7], dtype=torch.float32)
        return PlannerPlanResult(
            success=True,
            status="Success",
            failure=None,
            positions=(tuple([0.0] * 7),),
        )

    def attach_box(self, relative_pose7) -> None:
        del relative_pose7

    def detach(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _ActionBoundFallbackPlanner(_FakePlanner):
    def plan_pose(self, pose7, *, use_attached: bool = False) -> PlannerPlanResult:
        del use_attached
        self.plan_count += 1
        self.environment.tcp.pose.raw_pose = torch.tensor([pose7], dtype=torch.float32)
        position = (3.5, *([0.0] * 6)) if self.plan_count == 2 else tuple([0.0] * 7)
        return PlannerPlanResult(
            success=True,
            status="Success",
            failure=None,
            positions=(position,),
        )


def test_push_expert_completes_explicit_planar_phase_sequence() -> None:
    environment = _FakeEnvironment()
    planner = _FakePlanner(environment)
    expert = PushToRegionExpert(environment, planner_factory=lambda _env: planner)

    result = expert.run()

    assert result.success
    assert result.status is PushExpertStatus.SUCCESS
    assert result.completed_phases == PUSH_EXPERT_PHASE_SEQUENCE
    assert result.failed_phase is None
    assert result.final_environment_evaluation["success"] is True
    assert result.total_environment_steps == len(expert.action_trace) == 9
    assert result.total_planning_calls == 3
    assert [item.phase for item in expert.diagnostic_trace] == [
        phase.value for phase in PUSH_EXPERT_PHASE_SEQUENCE
    ]
    assert expert.diagnostic_trace[0].projected_progress == pytest.approx(0.0)
    assert planner.closed


def test_push_expert_classifies_wrong_object_displacement() -> None:
    environment = _FakeEnvironment(wrong_object_after_step=True)
    expert = PushToRegionExpert(
        environment,
        planner_factory=lambda _env: _FakePlanner(environment),
    )

    result = expert.run()

    assert not result.success
    assert result.status is PushExpertStatus.WRONG_OBJECT_INTERACTION
    assert result.failed_phase is PushExpertPhase.CLOSE_GRIPPER
    assert result.total_environment_steps == 1


def test_push_expert_config_rejects_unbounded_correction_search() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        PushExpertConfig(maximum_corrective_pushes=3)

    with pytest.raises(ValueError, match="must not exceed 25"):
        PushExpertConfig(lateral_cube_compensation_degrees=30.0)

    with pytest.raises(ValueError, match="must not exceed 25"):
        PushExpertConfig(lateral_cylinder_compensation_degrees=30.0)

    with pytest.raises(ValueError, match="finite and non-negative"):
        PushExpertConfig(lateral_cylinder_compensation_degrees=-1.0)

    with pytest.raises(ValueError, match="staging_height must exceed"):
        PushExpertConfig(precontact_staging_height=0.1)

    with pytest.raises(ValueError, match="must not exceed primary_push_increment"):
        PushExpertConfig(primary_push_increment=0.02, minimum_primary_push_increment=0.03)


def test_push_endpoint_stops_inside_full_containment_with_geometry_margin() -> None:
    environment = _FakeEnvironment()
    expert = PushToRegionExpert(
        environment,
        config=PushExpertConfig(region_goal_margin=0.015),
        planner_factory=lambda _env: _FakePlanner(environment),
    )
    expert._context = environment.context

    pose = expert._push_endpoint_pose(np.array([-0.2, 0.0, 0.025]))

    expected_tcp_x = 0.2 - ((0.07 - 0.015) + 0.035)
    assert pose[:3] == pytest.approx([expected_tcp_x, 0.0, 0.025])
    assert expert.config.to_dict()["region_goal_margin"] == pytest.approx(0.015)


def test_push_height_is_geometry_specific() -> None:
    environment = _FakeEnvironment()
    expert = PushToRegionExpert(
        environment,
        planner_factory=lambda _env: _FakePlanner(environment),
    )
    expert._context = environment.context
    assert expert._push_height() == pytest.approx(0.025)

    environment.context = PushExpertTaskContext(
        environment_id=environment.context.environment_id,
        episode_spec=PushEpisodeSpec.create(
            scene_seed=77,
            task_spec=PushTaskSpec("orange_cylinder", "left", "standard"),
        ),
        agent=environment.context.agent,
        robot=environment.context.robot,
        target_object=environment.context.objects[1],
        objects=environment.context.objects,
        target_region_center=environment.context.target_region_center,
        target_region_radius=environment.context.target_region_radius,
        target_object_planar_radius=0.025,
        target_object_resting_height=0.025,
        full_containment_center_radius=0.08,
        table_top_z=environment.context.table_top_z,
    )
    expert._context = environment.context
    assert expert._push_height() == pytest.approx(0.015)


def test_lateral_contact_strategy_is_geometry_specific() -> None:
    environment = _FakeEnvironment()
    expert = PushToRegionExpert(
        environment,
        planner_factory=lambda _env: _FakePlanner(environment),
    )
    expert._context = environment.context
    assert expert._contact_offset() == pytest.approx(0.045)
    assert expert._lateral_compensation_degrees() == pytest.approx(15.0)

    environment.context = PushExpertTaskContext(
        environment_id=environment.context.environment_id,
        episode_spec=PushEpisodeSpec.create(
            scene_seed=77,
            task_spec=PushTaskSpec("orange_cylinder", "left", "standard"),
        ),
        agent=environment.context.agent,
        robot=environment.context.robot,
        target_object=environment.context.objects[1],
        objects=environment.context.objects,
        target_region_center=environment.context.target_region_center,
        target_region_radius=environment.context.target_region_radius,
        target_object_planar_radius=0.025,
        target_object_resting_height=0.025,
        full_containment_center_radius=0.08,
        table_top_z=environment.context.table_top_z,
    )
    expert._context = environment.context
    assert expert._contact_offset() == pytest.approx(0.045)
    assert expert._lateral_compensation_degrees() == pytest.approx(0.0)


def test_lateral_cylinder_recontact_uses_controller_actions() -> None:
    environment = _FakeEnvironment()
    environment.context = PushExpertTaskContext(
        environment_id=environment.context.environment_id,
        episode_spec=PushEpisodeSpec.create(
            scene_seed=77,
            task_spec=PushTaskSpec("orange_cylinder", "left", "standard"),
        ),
        agent=environment.context.agent,
        robot=environment.context.robot,
        target_object=environment.context.objects[1],
        objects=environment.context.objects,
        target_region_center=torch.tensor([0.12, 0.22, 0.001]),
        target_region_radius=0.11,
        target_object_planar_radius=0.025,
        target_object_resting_height=0.025,
        full_containment_center_radius=0.085,
        table_top_z=environment.context.table_top_z,
    )
    planner = _FakePlanner(environment)
    expert = PushToRegionExpert(environment, planner_factory=lambda _env: planner)
    expert._context = environment.context
    expert._planner = planner
    result = expert._corrective_push(PushExpertPhase.CORRECTIVE_PUSH_1)

    assert result.success
    assert result.environment_steps == environment.step_count
    assert result.planning_calls > 0
    assert "corrective push" in result.message


def test_lateral_approach_falls_back_before_executing_out_of_bounds_plan() -> None:
    environment = _FakeEnvironment()
    planner = _ActionBoundFallbackPlanner(environment)
    expert = PushToRegionExpert(
        environment,
        planner_factory=lambda _env: planner,
    )
    expert._context = environment.context
    expert._planner = planner
    candidates = expert._lateral_approach_candidates(
        expert._target_position(),
        expert._push_direction(expert._target_position()),
    )

    result = expert._planned_lateral_approach(candidates)

    assert result.success
    assert result.attempts == 2
    assert result.planning_calls == 4
    assert result.environment_steps == 3
    assert environment.step_count == 3


def test_free_space_stride_preserves_the_exact_final_planner_position() -> None:
    environment = _FakeEnvironment()
    planner = _FakePlanner(environment)
    expert = PushToRegionExpert(
        environment,
        planner_factory=lambda _env: planner,
    )
    expert._context = environment.context
    expert._planner = planner
    positions = tuple(tuple([float(index) / 2.0] * 7) for index in range(5))

    def plan_pose(pose7, *, use_attached: bool = False):
        del use_attached
        environment.tcp.pose.raw_pose = torch.tensor([pose7], dtype=torch.float32)
        return PlannerPlanResult(
            success=True,
            status="Success",
            failure=None,
            positions=positions,
        )

    planner.plan_pose = plan_pose  # type: ignore[method-assign]
    result = expert._planned_motion(
        PushExpertPhase.MOVE_TO_PRECONTACT,
        (expert._tcp_pose(),),
        action_stride=2,
    )

    assert result.success
    assert [float(action[0]) for action in expert.action_trace] == [0.0, 1.0, 2.0]


def test_primary_motion_brakes_at_full_containment_until_stable_success() -> None:
    environment = _FakeEnvironment(inside_after_step=2, success_after_step=5)
    planner = _FakePlanner(environment)
    expert = PushToRegionExpert(environment, planner_factory=lambda _env: planner)
    expert._context = environment.context
    expert._planner = planner
    positions = tuple(tuple([float(index) / 2.0] * 7) for index in range(5))

    def plan_pose(pose7, *, use_attached: bool = False):
        del use_attached
        environment.tcp.pose.raw_pose = torch.tensor([pose7], dtype=torch.float32)
        return PlannerPlanResult(
            success=True,
            status="Success",
            failure=None,
            positions=positions,
        )

    planner.plan_pose = plan_pose  # type: ignore[method-assign]
    result = expert._planned_motion(
        PushExpertPhase.PRIMARY_PUSH,
        (expert._tcp_pose(),),
        stop_on_target_inside_region=True,
    )

    assert result.success
    assert result.environment_steps == 5
    assert environment.step_count == 5
    assert environment.get_push_expert_evaluation()["success"].item() is True


def test_adaptive_primary_push_stops_when_the_existing_success_gate_fires() -> None:
    environment = _FakeEnvironment()
    planner = _FakePlanner(environment)
    expert = PushToRegionExpert(
        environment,
        planner_factory=lambda _env: planner,
    )
    expert._context = environment.context
    expert._planner = planner

    result = expert._adaptive_primary_push()

    assert result.success
    assert result.phase is PushExpertPhase.PRIMARY_PUSH
    assert result.environment_steps == 9
    assert environment.step_count == 9
    assert len(expert.action_trace) == 9


def test_primary_push_uses_one_direct_plan_before_bounded_fallback() -> None:
    environment = _FakeEnvironment()
    planner = _FakePlanner(environment)
    expert = PushToRegionExpert(
        environment,
        planner_factory=lambda _env: planner,
    )
    expert._context = environment.context
    expert._planner = planner

    result = expert._primary_push()

    assert result.success
    assert result.planning_calls == 1
    assert "direct primary push" in result.message


def test_forward_strategy_preserves_the_original_contact_formula() -> None:
    environment = _FakeEnvironment()
    environment.context = PushExpertTaskContext(
        environment_id=environment.context.environment_id,
        episode_spec=PushEpisodeSpec.create(
            scene_seed=77,
            task_spec=PushTaskSpec("blue_cube", "forward_left", "standard"),
        ),
        agent=environment.context.agent,
        robot=environment.context.robot,
        target_object=environment.context.target_object,
        objects=environment.context.objects,
        target_region_center=environment.context.target_region_center,
        target_region_radius=environment.context.target_region_radius,
        target_object_planar_radius=environment.context.target_object_planar_radius,
        target_object_resting_height=environment.context.target_object_resting_height,
        full_containment_center_radius=environment.context.full_containment_center_radius,
        table_top_z=environment.context.table_top_z,
    )
    expert = PushToRegionExpert(
        environment,
        planner_factory=lambda _env: _FakePlanner(environment),
    )
    expert._context = environment.context
    position = expert._target_position()

    assert not expert._is_lateral_task()
    assert expert._motion_direction(position) == pytest.approx(expert._push_direction(position))
    assert expert._contact_offset() == pytest.approx(expert.config.contact_offset)


def test_push_expert_result_is_json_serializable() -> None:
    import json

    environment = _FakeEnvironment()
    result = PushToRegionExpert(
        environment,
        planner_factory=lambda _env: _FakePlanner(environment),
    ).run()

    assert json.loads(json.dumps(result.to_dict()))["task_id"] == result.task_id


def test_push_expert_validation_schedules_are_exact_and_balanced() -> None:
    smoke = build_schedule(smoke=True)
    target = build_schedule(smoke=False)

    assert len(smoke) == 16
    assert len(target) == 80
    assert sum(item.task_spec.difficulty == "standard" for item in target) == 50
    assert sum(item.task_spec.difficulty == "hard" for item in target) == 30
    assert {item.task_spec.target_object_id for item in target} == {
        "blue_cube",
        "orange_cylinder",
    }
    assert {item.task_spec.target_region_id for item in target} == {
        "left",
        "right",
        "forward_left",
        "forward_right",
    }
    assert len({item.seed for item in target}) == 80
    lateral = tuple(item for item in target if item.task_spec.target_region_id in {"left", "right"})
    assert len(lateral) == 42
    assert sum(item.task_spec.difficulty == "standard" for item in lateral) == 26
    assert sum(item.task_spec.difficulty == "hard" for item in lateral) == 16


def test_push_expert_benchmark_isolates_every_episode(monkeypatch) -> None:
    created: list[SimpleNamespace] = []
    schedule = (
        ScheduledPushEpisode(1, PushTaskSpec("blue_cube", "left", "standard")),
        ScheduledPushEpisode(2, PushTaskSpec("blue_cube", "right", "standard")),
    )
    result = PushToRegionExpert(
        _FakeEnvironment(),
        planner_factory=lambda environment: _FakePlanner(environment),
    ).run()

    def factory() -> SimpleNamespace:
        environment = SimpleNamespace(
            reset=lambda **_kwargs: None,
            close=lambda: setattr(environment, "closed", True),
            closed=False,
        )
        created.append(environment)
        return environment

    class StubExpert:
        def __init__(self, environment, *, config) -> None:
            del config
            self.environment = environment

        def run(self):
            return result

    monkeypatch.setattr("environment.benchmark_push_expert.PushToRegionExpert", StubExpert)
    results, errors = _run_schedule(
        schedule,
        environment_factory=factory,
        config=PushExpertConfig(),
    )

    assert len(results) == 2
    assert not errors
    assert len({id(environment) for environment in created}) == 2
    assert all(environment.closed for environment in created)
