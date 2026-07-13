"""Deterministic Panda motion-planner boundary for the M2 expert.

The adapter intentionally owns no environment execution logic.  It translates the
single-environment ManiSkill robot state to the public ``mplib==0.1.1`` API and returns
an immutable, JSON-serializable joint path for the expert to execute.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from types import ModuleType
from typing import Any, Protocol, runtime_checkable

import numpy as np

EXPECTED_MPLIB_VERSION = "0.1.1"
PANDA_MOVE_GROUP = "panda_hand_tcp"
PANDA_ARM_DOF = 7
PANDA_FULL_DOF = 9
DEFAULT_ATTACHED_BOX_SIZE = (0.05, 0.05, 0.05)
DEFAULT_JOINT_LIMIT_SCALE = 0.9


class PlannerFailure(StrEnum):
    """Failures reported by the public mplib planning call."""

    IK_FAILURE = "ik_failure"
    PLANNING_FAILURE = "planning_failure"


class PlannerImportError(ImportError):
    """Raised when the target-only mplib dependency cannot be imported."""


class PlannerVersionError(RuntimeError):
    """Raised when the installed mplib version differs from the inspected version."""


class PlannerContractError(RuntimeError):
    """Raised when an upstream or environment planning contract has changed."""


@dataclass(frozen=True, slots=True)
class PlannerPlanResult:
    """Small, immutable planning result safe to embed in JSON diagnostics."""

    success: bool
    status: str
    failure: PlannerFailure | None
    positions: tuple[tuple[float, ...], ...]
    duration_seconds: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a representation accepted by :func:`json.dumps`."""

        return {
            "success": self.success,
            "status": self.status,
            "failure": None if self.failure is None else self.failure.value,
            "positions": [list(position) for position in self.positions],
            "duration_seconds": self.duration_seconds,
        }


@runtime_checkable
class PlannerAdapter(Protocol):
    """Action-free planning operations consumed by the phase-based expert."""

    def synchronize(self) -> None:
        """Synchronize the planner's articulated model with simulator qpos."""

    def plan_pose(self, pose7: Sequence[float], *, use_attached: bool = False) -> PlannerPlanResult:
        """Plan a world-frame TCP pose without stepping the environment."""

    def attach_box(self, relative_pose7: Sequence[float]) -> None:
        """Represent the fixed-size M1 cube as attached to the TCP."""

    def detach(self) -> None:
        """Remove the attached collision object from the planning world."""

    def close(self) -> None:
        """Release adapter resources, if any."""


class MplibPandaPlannerAdapter:
    """Thin adapter over the inspected public API of ``mplib==0.1.1``.

    MPlib is imported lazily because ManiSkill installs it only on Linux.  M2 uses
    ``plan_screw`` exclusively: it is deterministic for fixed robot state and target,
    unlike the unseedable RRT interface in mplib 0.1.1.
    """

    def __init__(
        self,
        env: object,
        *,
        attached_box_size: Sequence[float] = DEFAULT_ATTACHED_BOX_SIZE,
    ) -> None:
        self._env = env
        self._base_env = getattr(env, "unwrapped", env)
        self._validate_environment()
        self._attached_box_size = _positive_vector(
            attached_box_size, expected_size=3, label="attached_box_size"
        )

        mplib_module = _load_mplib()
        planner_type = _validate_mplib_contract(mplib_module)
        self._robot = self._base_env.agent.robot
        link_names = _names_from_handles(self._robot.get_links(), label="robot links")
        joint_names = _names_from_handles(
            self._robot.get_active_joints(), label="robot active joints"
        )
        if len(joint_names) != PANDA_FULL_DOF:
            raise PlannerContractError(
                f"Panda must expose {PANDA_FULL_DOF} active joints, got {len(joint_names)}"
            )

        urdf_path = getattr(self._base_env.agent, "urdf_path", None)
        if not isinstance(urdf_path, str) or not urdf_path.endswith(".urdf"):
            raise PlannerContractError("Panda agent.urdf_path must be a .urdf string")
        srdf_path = f"{urdf_path[:-5]}.srdf"
        limits = np.full(PANDA_ARM_DOF, DEFAULT_JOINT_LIMIT_SCALE, dtype=np.float64)

        self._planner = planner_type(
            urdf=urdf_path,
            srdf=srdf_path,
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group=PANDA_MOVE_GROUP,
            joint_vel_limits=limits.copy(),
            joint_acc_limits=limits.copy(),
        )

        _require_callable(self._planner, "set_base_pose")
        _require_callable(self._planner, "plan_screw")
        _require_callable(self._planner, "update_attached_box")
        planner_robot = getattr(self._planner, "robot", None)
        if planner_robot is None:
            raise PlannerContractError("mplib.Planner is missing public robot model")
        _require_callable(planner_robot, "set_qpos")
        planning_world = getattr(self._planner, "planning_world", None)
        if planning_world is None:
            raise PlannerContractError("mplib.Planner is missing public planning_world")
        _require_callable(planning_world, "remove_attach")

        base_pose7 = _pose_from_robot(self._robot)
        self._planner.set_base_pose(base_pose7)

    def synchronize(self) -> None:
        """Synchronize mplib kinematics and self-collision state with ManiSkill."""
        self._planner.robot.set_qpos(_full_robot_qpos(self._robot).copy(), True)

    def _validate_environment(self) -> None:
        num_envs = getattr(self._base_env, "num_envs", None)
        if num_envs != 1:
            raise PlannerContractError(f"M2 planner requires num_envs=1, got {num_envs!r}")
        control_timestep = getattr(self._base_env, "control_timestep", None)
        if not isinstance(control_timestep, (int, float)) or not np.isfinite(control_timestep):
            raise PlannerContractError("environment control_timestep must be a finite number")
        if control_timestep <= 0:
            raise PlannerContractError("environment control_timestep must be positive")
        agent = getattr(self._base_env, "agent", None)
        if agent is None or getattr(agent, "robot", None) is None:
            raise PlannerContractError("environment must expose agent.robot")

    def plan_pose(self, pose7: Sequence[float], *, use_attached: bool = False) -> PlannerPlanResult:
        """Plan one deterministic screw path from the current simulator qpos."""

        target_pose = _pose7(pose7, label="target pose")
        current_qpos = _full_robot_qpos(self._robot)
        raw_result = self._planner.plan_screw(
            target_pose,
            current_qpos.copy(),
            time_step=float(self._base_env.control_timestep),
            use_attach=use_attached,
            wrt_world=True,
        )
        return _validate_plan_result(raw_result)

    def attach_box(self, relative_pose7: Sequence[float]) -> None:
        """Attach the M1 cube collision box at a TCP-relative pose."""

        relative_pose = _pose7(relative_pose7, label="attached box relative pose")
        self._planner.update_attached_box(self._attached_box_size.copy(), relative_pose)

    def detach(self) -> None:
        """Remove the attached collision object through public PlanningWorld API."""
        self._planner.planning_world.remove_attach()

    def close(self) -> None:
        """MPlib 0.1.1 exposes no explicit resource close operation."""


def _load_mplib() -> ModuleType:
    try:
        module = import_module("mplib")
    except (ImportError, OSError) as error:
        raise PlannerImportError(
            "mplib==0.1.1 is required for M2 planning and is available only in the "
            "native Linux target environment"
        ) from error
    if not isinstance(module, ModuleType):
        raise PlannerContractError("imported mplib object is not a module")
    return module


def _validate_mplib_contract(module: ModuleType) -> type[Any]:
    version = getattr(module, "__version__", None)
    if version != EXPECTED_MPLIB_VERSION:
        raise PlannerVersionError(f"M2 requires mplib=={EXPECTED_MPLIB_VERSION}, got {version!r}")
    planner_type = getattr(module, "Planner", None)
    if not isinstance(planner_type, type):
        raise PlannerContractError("mplib 0.1.1 is missing public Planner class")
    return planner_type


def _require_callable(owner: object, name: str) -> None:
    if not callable(getattr(owner, name, None)):
        raise PlannerContractError(
            f"{type(owner).__name__} is missing required public method {name}"
        )


def _names_from_handles(handles: object, *, label: str) -> list[str]:
    if not isinstance(handles, Sequence):
        raise PlannerContractError(f"{label} must be a sequence")
    names: list[str] = []
    for handle in handles:
        get_name = getattr(handle, "get_name", None)
        if not callable(get_name):
            raise PlannerContractError(f"each entry in {label} must expose get_name()")
        name = get_name()
        if not isinstance(name, str) or not name:
            raise PlannerContractError(f"each entry in {label} must have a non-empty name")
        names.append(name)
    if not names:
        raise PlannerContractError(f"{label} must not be empty")
    return names


def _to_numpy(value: object, *, label: str) -> np.ndarray:
    tensor = value
    detach = getattr(tensor, "detach", None)
    if callable(detach):
        tensor = detach()
    cpu = getattr(tensor, "cpu", None)
    if callable(cpu):
        tensor = cpu()
    numpy_method = getattr(tensor, "numpy", None)
    if callable(numpy_method):
        tensor = numpy_method()
    try:
        array = np.asarray(tensor, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise PlannerContractError(f"{label} must be numeric") from error
    if not np.all(np.isfinite(array)):
        raise PlannerContractError(f"{label} must contain only finite values")
    return array


def _positive_vector(value: object, *, expected_size: int, label: str) -> np.ndarray:
    vector = _to_numpy(value, label=label)
    if vector.shape != (expected_size,):
        raise PlannerContractError(
            f"{label} must have shape ({expected_size},), got {vector.shape}"
        )
    if np.any(vector <= 0):
        raise PlannerContractError(f"{label} entries must be positive")
    return vector


def _pose7(value: object, *, label: str) -> np.ndarray:
    pose = _to_numpy(value, label=label)
    if pose.shape != (7,):
        raise PlannerContractError(f"{label} must have shape (7,), got {pose.shape}")
    quaternion_norm = float(np.linalg.norm(pose[3:]))
    if not np.isclose(quaternion_norm, 1.0, atol=1e-5, rtol=0.0):
        raise PlannerContractError(
            f"{label} quaternion must be unit length in [qw, qx, qy, qz] order"
        )
    return pose


def _pose_from_robot(robot: object) -> np.ndarray:
    pose = getattr(robot, "pose", None)
    raw_pose = getattr(pose, "raw_pose", None)
    if raw_pose is None:
        raise PlannerContractError("robot.pose must expose batched raw_pose")
    array = _to_numpy(raw_pose, label="robot base pose")
    if array.shape != (1, 7):
        raise PlannerContractError(f"robot base pose must have shape (1, 7), got {array.shape}")
    return _pose7(array[0], label="robot base pose")


def _full_robot_qpos(robot: object) -> np.ndarray:
    get_qpos = getattr(robot, "get_qpos", None)
    if not callable(get_qpos):
        raise PlannerContractError("robot must expose get_qpos()")
    qpos = _to_numpy(get_qpos(), label="robot qpos")
    if qpos.shape != (1, PANDA_FULL_DOF):
        raise PlannerContractError(
            f"Panda qpos must have shape (1, {PANDA_FULL_DOF}), got {qpos.shape}"
        )
    return qpos[0]


def _validate_plan_result(raw_result: object) -> PlannerPlanResult:
    if not isinstance(raw_result, Mapping):
        raise PlannerContractError("mplib plan result must be a mapping")
    status = raw_result.get("status")
    if not isinstance(status, str) or not status:
        raise PlannerContractError("mplib plan result must contain a non-empty string status")
    if status != "Success":
        failure = (
            PlannerFailure.IK_FAILURE
            if status.startswith("IK Failed")
            else PlannerFailure.PLANNING_FAILURE
        )
        return PlannerPlanResult(
            success=False,
            status=status,
            failure=failure,
            positions=(),
        )

    positions = _to_numpy(raw_result.get("position"), label="mplib result position")
    if positions.ndim != 2 or positions.shape[0] < 1 or positions.shape[1] != PANDA_ARM_DOF:
        raise PlannerContractError(
            f"successful mplib result position must have shape (steps, 7), got {positions.shape}"
        )

    raw_duration = raw_result.get("duration")
    duration: float | None = None
    if raw_duration is not None:
        duration_array = _to_numpy(raw_duration, label="mplib result duration")
        if duration_array.shape != ():
            raise PlannerContractError("mplib result duration must be a scalar")
        duration = float(duration_array)
        if duration < 0:
            raise PlannerContractError("mplib result duration must be non-negative")

    return PlannerPlanResult(
        success=True,
        status=status,
        failure=None,
        positions=tuple(tuple(float(value) for value in row) for row in positions),
        duration_seconds=duration,
    )


__all__ = [
    "MplibPandaPlannerAdapter",
    "PlannerAdapter",
    "PlannerContractError",
    "PlannerFailure",
    "PlannerImportError",
    "PlannerPlanResult",
    "PlannerVersionError",
]
