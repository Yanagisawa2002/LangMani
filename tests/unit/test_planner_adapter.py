from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from langmani.experts import planner as planner_module


class _NamedHandle:
    def __init__(self, name: str) -> None:
        self._name = name

    def get_name(self) -> str:
        return self._name


class _FakeRobot:
    def __init__(self) -> None:
        self.pose = SimpleNamespace(raw_pose=np.array([[-0.615, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]))
        self.qpos = np.arange(9, dtype=np.float64).reshape(1, 9) / 10
        self.qpos_reads = 0

    def get_links(self) -> list[_NamedHandle]:
        return [_NamedHandle("panda_link0"), _NamedHandle("panda_hand_tcp")]

    def get_active_joints(self) -> list[_NamedHandle]:
        return [_NamedHandle(f"panda_joint{index}") for index in range(1, 8)] + [
            _NamedHandle("panda_finger_joint1"),
            _NamedHandle("panda_finger_joint2"),
        ]

    def get_qpos(self) -> np.ndarray:
        self.qpos_reads += 1
        return self.qpos.copy()


class _FakeEnv:
    def __init__(self) -> None:
        self.unwrapped = self
        self.num_envs = 1
        self.control_timestep = 0.05
        self.robot = _FakeRobot()
        self.agent = SimpleNamespace(
            robot=self.robot,
            urdf_path="/mani_skill/assets/robots/panda/panda_v2.urdf",
        )
        self.step_calls = 0

    def step(self, action: object) -> None:
        del action
        self.step_calls += 1
        raise AssertionError("the planner adapter must never step the environment")


class _FakePlanningWorld:
    def __init__(self) -> None:
        self.remove_attach_calls = 0

    def remove_attach(self) -> None:
        self.remove_attach_calls += 1


class _FakePlannerRobot:
    def __init__(self) -> None:
        self.set_qpos_calls: list[tuple[np.ndarray, bool]] = []

    def set_qpos(self, qpos: np.ndarray, full: bool = False) -> None:
        self.set_qpos_calls.append((qpos.copy(), full))


class _FakePlanner:
    instances: list[_FakePlanner] = []
    next_result: object = {
        "status": "Success",
        "position": np.zeros((2, 7), dtype=np.float64),
        "duration": 0.2,
    }

    def __init__(self, **kwargs: Any) -> None:
        self.constructor_kwargs = kwargs
        self.base_poses: list[np.ndarray] = []
        self.plan_calls: list[tuple[np.ndarray, np.ndarray, dict[str, object]]] = []
        self.attach_calls: list[tuple[np.ndarray, np.ndarray]] = []
        self.planning_world = _FakePlanningWorld()
        self.robot = _FakePlannerRobot()
        type(self).instances.append(self)

    def set_base_pose(self, pose7: np.ndarray) -> None:
        self.base_poses.append(pose7.copy())

    def plan_screw(self, target_pose: np.ndarray, qpos: np.ndarray, **kwargs: object) -> object:
        self.plan_calls.append((target_pose.copy(), qpos.copy(), dict(kwargs)))
        qpos[:] = -999.0
        return type(self).next_result

    def update_attached_box(self, size: np.ndarray, pose7: np.ndarray) -> None:
        self.attach_calls.append((size.copy(), pose7.copy()))


@pytest.fixture(autouse=True)
def _reset_fake_planner() -> None:
    _FakePlanner.instances.clear()
    _FakePlanner.next_result = {
        "status": "Success",
        "position": np.zeros((2, 7), dtype=np.float64),
        "duration": 0.2,
    }


@pytest.fixture
def fake_mplib(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = ModuleType("mplib")
    module.__version__ = "0.1.1"
    module.Planner = _FakePlanner
    monkeypatch.setitem(sys.modules, "mplib", module)
    monkeypatch.setattr(planner_module.np, "__version__", "1.26.4")
    return module


def test_planner_module_import_does_not_eagerly_import_mplib() -> None:
    assert "mani_skill.examples" not in sys.modules
    assert "mplib" not in planner_module.__dict__


def test_constructs_public_panda_planner_contract(fake_mplib: ModuleType) -> None:
    del fake_mplib
    env = _FakeEnv()

    adapter = planner_module.MplibPandaPlannerAdapter(env)
    raw_planner = _FakePlanner.instances[-1]

    assert isinstance(adapter, planner_module.PlannerAdapter)
    assert raw_planner.constructor_kwargs["urdf"].endswith("panda_v2.urdf")
    assert raw_planner.constructor_kwargs["srdf"].endswith("panda_v2.srdf")
    assert raw_planner.constructor_kwargs["move_group"] == "panda_hand_tcp"
    assert raw_planner.constructor_kwargs["user_link_names"] == [
        "panda_link0",
        "panda_hand_tcp",
    ]
    assert len(raw_planner.constructor_kwargs["user_joint_names"]) == 9
    np.testing.assert_allclose(raw_planner.constructor_kwargs["joint_vel_limits"], 0.9)
    np.testing.assert_allclose(raw_planner.constructor_kwargs["joint_acc_limits"], 0.9)
    np.testing.assert_allclose(raw_planner.base_poses[0], [-0.615, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert env.step_calls == 0


def test_plan_pose_reads_fresh_full_qpos_and_only_calls_screw(
    fake_mplib: ModuleType,
) -> None:
    del fake_mplib
    env = _FakeEnv()
    adapter = planner_module.MplibPandaPlannerAdapter(env)
    raw_planner = _FakePlanner.instances[-1]
    pose7 = [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]

    first = adapter.plan_pose(pose7)
    env.robot.qpos = np.full((1, 9), 0.25)
    second = adapter.plan_pose(pose7, use_attached=True)

    assert first.success and second.success
    assert env.robot.qpos_reads == 2
    np.testing.assert_allclose(raw_planner.plan_calls[0][1], np.arange(9) / 10)
    np.testing.assert_allclose(raw_planner.plan_calls[1][1], 0.25)
    assert raw_planner.plan_calls[0][2] == {
        "time_step": 0.05,
        "use_attach": False,
        "wrt_world": True,
    }
    assert raw_planner.plan_calls[1][2]["use_attach"] is True
    assert env.step_calls == 0


def test_synchronize_updates_public_planner_robot_with_full_simulator_qpos(
    fake_mplib: ModuleType,
) -> None:
    del fake_mplib
    env = _FakeEnv()
    adapter = planner_module.MplibPandaPlannerAdapter(env)
    raw_planner = _FakePlanner.instances[-1]

    adapter.synchronize()

    qpos, full = raw_planner.robot.set_qpos_calls[-1]
    np.testing.assert_allclose(qpos, np.arange(9) / 10)
    assert full is True


def test_success_result_is_immutable_and_json_serializable(
    fake_mplib: ModuleType,
) -> None:
    del fake_mplib
    positions = np.arange(21, dtype=np.float64).reshape(3, 7) / 100
    _FakePlanner.next_result = {
        "status": "Success",
        "position": positions,
        "duration": np.float64(0.3),
    }

    result = planner_module.MplibPandaPlannerAdapter(_FakeEnv()).plan_pose(
        [0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]
    )

    assert result == planner_module.PlannerPlanResult(
        success=True,
        status="Success",
        failure=None,
        positions=tuple(tuple(float(value) for value in row) for row in positions),
        duration_seconds=0.3,
    )
    assert json.loads(json.dumps(result.to_dict()))["positions"] == positions.tolist()


@pytest.mark.parametrize(
    ("status", "expected_failure"),
    [
        ("IK Failed! Cannot find valid solution.", planner_module.PlannerFailure.IK_FAILURE),
        ("screw plan failed", planner_module.PlannerFailure.PLANNING_FAILURE),
    ],
)
def test_plan_failure_classification(
    fake_mplib: ModuleType,
    status: str,
    expected_failure: planner_module.PlannerFailure,
) -> None:
    del fake_mplib
    _FakePlanner.next_result = {"status": status}

    result = planner_module.MplibPandaPlannerAdapter(_FakeEnv()).plan_pose(
        [0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]
    )

    assert not result.success
    assert result.failure is expected_failure
    assert result.positions == ()


@pytest.mark.parametrize(
    "position",
    [
        np.zeros(7),
        np.zeros((0, 7)),
        np.zeros((2, 6)),
        np.full((2, 7), np.nan),
    ],
)
def test_rejects_malformed_success_paths(fake_mplib: ModuleType, position: np.ndarray) -> None:
    del fake_mplib
    _FakePlanner.next_result = {"status": "Success", "position": position}
    adapter = planner_module.MplibPandaPlannerAdapter(_FakeEnv())

    with pytest.raises(planner_module.PlannerContractError, match="position"):
        adapter.plan_pose([0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0])


def test_attach_detach_and_close_are_action_free(fake_mplib: ModuleType) -> None:
    del fake_mplib
    env = _FakeEnv()
    adapter = planner_module.MplibPandaPlannerAdapter(env)
    raw_planner = _FakePlanner.instances[-1]

    adapter.attach_box([0.0, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0])
    adapter.detach()
    assert adapter.close() is None

    size, pose = raw_planner.attach_calls[0]
    np.testing.assert_allclose(size, [0.05, 0.05, 0.05])
    np.testing.assert_allclose(pose, [0.0, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0])
    assert raw_planner.planning_world.remove_attach_calls == 1
    assert env.step_calls == 0


def test_missing_mplib_has_clear_lazy_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _missing(name: str) -> ModuleType:
        assert name == "mplib"
        raise ModuleNotFoundError("No module named 'mplib'")

    monkeypatch.setattr(planner_module, "import_module", _missing)

    with pytest.raises(planner_module.PlannerImportError, match="mplib==0.1.1"):
        planner_module.MplibPandaPlannerAdapter(_FakeEnv())


def test_wrong_mplib_version_fails_before_construction(
    fake_mplib: ModuleType,
) -> None:
    fake_mplib.__version__ = "0.2.1"

    with pytest.raises(planner_module.PlannerVersionError, match="got '0.2.1'"):
        planner_module.MplibPandaPlannerAdapter(_FakeEnv())

    assert not _FakePlanner.instances


def test_numpy_two_runtime_fails_before_native_planner_construction(
    fake_mplib: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del fake_mplib
    monkeypatch.setattr(planner_module.np, "__version__", "2.2.6")

    with pytest.raises(planner_module.PlannerVersionError, match="numpy==1.26.4"):
        planner_module.MplibPandaPlannerAdapter(_FakeEnv())

    assert not _FakePlanner.instances


def test_missing_public_planner_method_is_clear(fake_mplib: ModuleType) -> None:
    class _BrokenPlanner(_FakePlanner):
        plan_screw = None  # type: ignore[assignment]

    fake_mplib.Planner = _BrokenPlanner

    with pytest.raises(planner_module.PlannerContractError, match="plan_screw"):
        planner_module.MplibPandaPlannerAdapter(_FakeEnv())


def test_rejects_non_single_env_and_non_full_qpos(fake_mplib: ModuleType) -> None:
    del fake_mplib
    env = _FakeEnv()
    env.num_envs = 2
    with pytest.raises(planner_module.PlannerContractError, match="num_envs=1"):
        planner_module.MplibPandaPlannerAdapter(env)

    env = _FakeEnv()
    adapter = planner_module.MplibPandaPlannerAdapter(env)
    env.robot.qpos = np.zeros((1, 7))
    with pytest.raises(planner_module.PlannerContractError, match=r"shape \(1, 9\)"):
        adapter.plan_pose([0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0])


def test_unexpected_upstream_planning_exception_is_not_collapsed(
    fake_mplib: ModuleType,
) -> None:
    del fake_mplib
    adapter = planner_module.MplibPandaPlannerAdapter(_FakeEnv())
    raw_planner = _FakePlanner.instances[-1]

    def explode(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise ArithmeticError("upstream numerical defect")

    raw_planner.plan_screw = explode  # type: ignore[method-assign]

    with pytest.raises(ArithmeticError, match="upstream numerical defect"):
        adapter.plan_pose([0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0])


def test_unexpected_constructor_and_base_pose_exceptions_are_not_collapsed(
    fake_mplib: ModuleType,
) -> None:
    class ExplodingConstructor:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            raise ArithmeticError("constructor defect")

    fake_mplib.Planner = ExplodingConstructor
    with pytest.raises(ArithmeticError, match="constructor defect"):
        planner_module.MplibPandaPlannerAdapter(_FakeEnv())

    class ExplodingBasePose(_FakePlanner):
        def set_base_pose(self, pose7: np.ndarray) -> None:
            del pose7
            raise ArithmeticError("base pose defect")

    fake_mplib.Planner = ExplodingBasePose
    with pytest.raises(ArithmeticError, match="base pose defect"):
        planner_module.MplibPandaPlannerAdapter(_FakeEnv())
