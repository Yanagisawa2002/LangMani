from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from langmani.environments.push_expert_state import PushExpertTaskContext, PushObjectHandle
from langmani.environments.push_specs import PushEpisodeSpec, PushTaskSpec
from langmani.environments.push_to_region import ENV_ID
from langmani.experts.geometry_push_expert import GeometryConstrainedPushExpert
from langmani.experts.geometry_push_types import (
    ALLOWED_GEOMETRY_PUSH_TRANSITIONS,
    GEOMETRY_PUSH_STATE_CONTRACTS,
    GeometryPushExpertConfig,
    GeometryPushState,
    GeometryPushTransition,
)
from langmani.experts.planner import PlannerPlanResult
from langmani.experts.push_feasibility import (
    PlanarPrimitive,
    PrimitiveKind,
    WorkspaceContract,
    box_support_extent,
    build_bounded_joint_action,
    build_geometry_push_plan,
    support_extent,
)
from langmani.experts.push_types import PushExpertStatus
from langmani.v2.push_expert_recovery import PushExpertRecoveryError, load_json
from scripts.langmani_v2.evaluate_geometry_push_expert import _schedule, _zero_tolerance_reason


def _cube() -> PlanarPrimitive:
    return PlanarPrimitive(
        PrimitiveKind.BOX,
        half_extents_xy=(0.02, 0.04),
        radius=math.hypot(0.02, 0.04),
        height=0.05,
    )


def _workspace() -> WorkspaceContract:
    return WorkspaceContract((-0.43, 0.43, -0.38, 0.38), 0.002, 0.015, 0.002)


def test_box_and_cylinder_support_extents_are_direction_aware() -> None:
    assert box_support_extent((0.02, 0.04), np.array([1.0, 0.0]), 0.0) == pytest.approx(0.02)
    assert box_support_extent((0.02, 0.04), np.array([1.0, 0.0]), math.pi / 2) == pytest.approx(
        0.04
    )
    cylinder = PlanarPrimitive(PrimitiveKind.CYLINDER, (0.03, 0.03), 0.03, 0.05)
    assert support_extent(cylinder, np.array([0.3, -0.7])) == pytest.approx(0.03)


def test_geometry_plan_is_segmented_and_support_workspace_safe() -> None:
    plan = build_geometry_push_plan(
        object_xy=np.array([-0.18, 0.02]),
        target_xy=np.array([0.20, 0.14]),
        primitive=_cube(),
        object_yaw=0.4,
        workspace=_workspace(),
        full_containment_radius=0.05,
        goal_margin=0.01,
        gripper_contact_padding=0.02,
        precontact_clearance=0.06,
        push_height=0.025,
        precontact_height=0.085,
        staging_height=0.18,
        maximum_segment_distance=0.065,
    )
    assert len(plan.segment_endpoints_xy) >= 5
    assert plan.object_path_margin >= 0.0
    assert plan.tcp_path_margin >= 0.0
    assert np.linalg.norm(plan.direction_xy) == pytest.approx(1.0)


def test_geometry_plan_rejects_zero_distance_and_unsafe_footprint() -> None:
    arguments = {
        "primitive": _cube(),
        "object_yaw": 0.0,
        "workspace": _workspace(),
        "full_containment_radius": 0.05,
        "goal_margin": 0.01,
        "gripper_contact_padding": 0.02,
        "precontact_clearance": 0.06,
        "push_height": 0.025,
        "precontact_height": 0.085,
        "staging_height": 0.18,
        "maximum_segment_distance": 0.065,
    }
    with pytest.raises(ValueError, match="distinct"):
        build_geometry_push_plan(
            object_xy=np.array([0.0, 0.0]),
            target_xy=np.array([0.0, 0.0]),
            **arguments,
        )
    with pytest.raises(ValueError, match="object path leaves safe workspace"):
        build_geometry_push_plan(
            object_xy=np.array([0.41, 0.0]),
            target_xy=np.array([0.2, 0.0]),
            **arguments,
        )


def test_action_builder_rejects_instead_of_clipping() -> None:
    low = np.asarray([-1.0] * 8)
    high = np.asarray([1.0] * 8)
    action = build_bounded_joint_action(np.zeros(7), action_low=low, action_high=high)
    assert action.tolist() == [0.0] * 7 + [-1.0]
    with pytest.raises(ValueError, match="exceeds"):
        build_bounded_joint_action(np.asarray([1.01] + [0.0] * 6), action_low=low, action_high=high)


def test_state_machine_rejects_illegal_transition() -> None:
    with pytest.raises(ValueError, match="illegal"):
        GeometryPushTransition(
            source=GeometryPushState.PLAN_PUSH,
            target=GeometryPushState.SUCCESS,
            reason="skip",
            environment_steps=0,
            target_distance=0.4,
            contact_recoveries=0,
            replans=0,
        )


def test_every_state_has_a_complete_contract_matching_allowed_transitions() -> None:
    assert set(GEOMETRY_PUSH_STATE_CONTRACTS) == set(GeometryPushState)
    for state, contract in GEOMETRY_PUSH_STATE_CONTRACTS.items():
        assert contract.entry_condition
        assert contract.control_target
        assert contract.progress_condition
        assert contract.timeout_rule
        assert contract.failure_reason
        assert contract.allowed_next == ALLOWED_GEOMETRY_PUSH_TRANSITIONS[state]


class _Pose:
    def __init__(self, value: list[float]) -> None:
        self.raw_pose = torch.tensor([value], dtype=torch.float32)


class _Robot:
    def __init__(self) -> None:
        self._qpos = torch.zeros((1, 9), dtype=torch.float32)

    def get_qpos(self) -> torch.Tensor:
        return self._qpos


class _ClosedLoopEnvironment:
    def __init__(self, *, success_after_step: int = 10) -> None:
        self.unwrapped = self
        self.num_envs = 1
        self.control_mode = "pd_joint_pos"
        self.step_count = 0
        self.success_after_step = success_after_step
        self.robot = _Robot()
        self.tcp = SimpleNamespace(pose=_Pose([0.0, 0.0, 0.18, 1.0, 0.0, 0.0, 0.0]))
        self.agent = SimpleNamespace(robot=self.robot, tcp=self.tcp)
        self.target = SimpleNamespace(pose=_Pose([-0.18, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]))
        self.distractor = SimpleNamespace(pose=_Pose([-0.28, 0.24, 0.025, 1.0, 0.0, 0.0, 0.0]))
        episode = PushEpisodeSpec.create(
            scene_seed=69_000,
            task_spec=PushTaskSpec("blue_cube", "forward_left", "standard"),
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
            target_region_center=torch.tensor([0.20, 0.14, 0.001]),
            target_region_radius=0.11,
            target_object_planar_radius=math.sqrt(2.0) * 0.025,
            target_object_resting_height=0.025,
            full_containment_center_radius=0.06964466094,
            table_top_z=0.0,
        )

    def get_push_expert_task_context(self) -> PushExpertTaskContext:
        return self.context

    def get_push_expert_evaluation(self) -> dict[str, torch.Tensor]:
        success = self.step_count >= self.success_after_step
        return {
            "success": torch.tensor([success]),
            "target_inside_region": torch.tensor([success]),
            "target_is_static": torch.tensor([True]),
            "stable_success_steps": torch.tensor([5 if success else 0]),
            "target_distance": torch.tensor([0.0 if success else 0.4]),
            "target_outside_workspace": torch.tensor([False]),
            "target_lifted": torch.tensor([False]),
            "target_toppled": torch.tensor([False]),
            "target_is_grasped": torch.tensor([False]),
            "wrong_object_displaced": torch.tensor([False]),
            "invalid_action": torch.tensor([False]),
            "action_out_of_bounds": torch.tensor([False]),
        }

    def get_push_expert_feedback(self) -> dict[str, torch.Tensor]:
        return {
            "target_contact_force": torch.tensor([1.0]),
            "target_linear_velocity": torch.zeros((1, 3)),
            "target_angular_velocity": torch.zeros((1, 3)),
            "target_pose": self.target.pose.raw_pose,
            "tcp_pose": self.tcp.pose.raw_pose,
        }

    def get_push_expert_action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray([-3.0] * 7 + [-1.0]), np.asarray([3.0] * 7 + [1.0])

    def step(self, action: np.ndarray):
        assert action.shape == (8,)
        self.step_count += 1
        evaluation = self.get_push_expert_evaluation()
        return {}, 0.0, evaluation["success"], torch.tensor([False]), evaluation


class _ClosedLoopPlanner:
    def __init__(self, environment: _ClosedLoopEnvironment) -> None:
        self.environment = environment
        self.closed = False

    def synchronize(self) -> None:
        return None

    def plan_pose(self, pose7, *, use_attached: bool = False) -> PlannerPlanResult:
        del use_attached
        self.environment.tcp.pose.raw_pose = torch.tensor([pose7], dtype=torch.float32)
        return PlannerPlanResult(True, "Success", None, (tuple([0.0] * 7),))

    def attach_box(self, relative_pose7) -> None:
        del relative_pose7

    def detach(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def test_geometry_expert_uses_explicit_states_and_native_success() -> None:
    environment = _ClosedLoopEnvironment()
    planner = _ClosedLoopPlanner(environment)
    expert = GeometryConstrainedPushExpert(
        environment,
        planner_factory=lambda _environment: planner,
    )

    result = expert.run()

    assert result.status is PushExpertStatus.SUCCESS
    assert result.success
    assert [value.source for value in result.transitions] == [
        GeometryPushState.PLAN_PUSH,
        GeometryPushState.MOVE_TO_STAGING,
        GeometryPushState.MOVE_TO_PRECONTACT,
        GeometryPushState.ESTABLISH_CONTACT,
        GeometryPushState.PUSH_CLOSED_LOOP,
        GeometryPushState.VERIFY_PROGRESS,
    ]
    assert result.transitions[-1].target is GeometryPushState.SUCCESS
    assert result.final_environment_evaluation["success"] is True
    assert result.contact_recoveries == 0
    assert result.replans == 0
    assert planner.closed


def test_geometry_config_has_bounded_recovery_and_replan_limits() -> None:
    config = GeometryPushExpertConfig()
    assert config.maximum_contact_recoveries == 2
    assert config.maximum_replans == 3
    assert config.maximum_push_segments == 12
    with pytest.raises(ValueError, match="non-negative"):
        GeometryPushExpertConfig(maximum_replans=-1)


def test_development_schedule_is_frozen_and_acceptance_is_sealed() -> None:
    development = load_json("configs/langmani_v2/phase_2b3/eval_development.json")
    schedule = _schedule(
        development,
        source_commit="a" * 40,
        development_report=None,
    )
    assert len(schedule) == 100
    assert [seed for seed, _ in schedule] == list(range(69_000, 69_100))
    assert sum(task.difficulty == "standard" for _, task in schedule) == 60
    assert sum(task.difficulty == "hard" for _, task in schedule) == 40

    acceptance = load_json("configs/langmani_v2/phase_2b3/eval_acceptance.json")
    with pytest.raises(PushExpertRecoveryError, match="sealed"):
        _schedule(
            acceptance,
            source_commit="a" * 40,
            development_report=None,
        )


def test_regression_schedule_reuses_only_frozen_replay_cases() -> None:
    config = load_json("configs/langmani_v2/phase_2b3/eval_regression.json")
    schedule = _schedule(config, source_commit="a" * 40, development_report=None)
    validation = load_json("artifacts/langmani_v2/phase_2b3/replay_validation.json")
    assert len(schedule) == validation["case_count"] == 55
    assert schedule[0][0] == validation["cases"][0]["seed"]  # type: ignore[index]


def test_regression_gate_stops_on_wrong_object_interaction() -> None:
    record = {
        "result": {
            "status": "wrong_object_interaction",
            "success": False,
            "final_environment_evaluation": {},
        }
    }
    assert _zero_tolerance_reason(record) == "wrong_object_interaction"
    assert (
        _zero_tolerance_reason(
            {
                "result": {
                    "status": "timeout",
                    "success": False,
                    "final_environment_evaluation": {},
                }
            }
        )
        is None
    )
