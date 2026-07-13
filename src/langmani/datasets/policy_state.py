"""Project-owned Panda policy-state extraction for the M3B export boundary.

The module deliberately does not import ManiSkill.  It consumes only the two
public robot operations needed by the exporter: ``get_active_joints()`` and
``get_qpos()``.  The returned policy vector is always ordered by the stable
``PandaPolicyStateV0`` semantic component names, independently of the order in
which an upstream articulation exposes its active joints.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt

PANDA_POLICY_STATE_COMPONENTS: tuple[str, ...] = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
)
_POLICY_STATE_NAME = "PandaPolicyStateV0"
_POLICY_STATE_SIZE = len(PANDA_POLICY_STATE_COMPONENTS)

Float32PolicyState = npt.NDArray[np.float32]


class PolicyStateContractError(ValueError):
    """Raised when a robot or candidate state violates PandaPolicyStateV0."""


class NamedActiveJoint(Protocol):
    """Minimum public active-joint surface consumed by this module."""

    name: str


class PandaPolicyStateRobot(Protocol):
    """Minimum public robot surface consumed by PandaPolicyStateV0."""

    def get_active_joints(self) -> Sequence[NamedActiveJoint]: ...

    def get_qpos(self) -> object: ...


def get_active_joint_names(robot: PandaPolicyStateRobot) -> tuple[str, ...]:
    """Return validated active-joint names in the robot's qpos order."""
    get_active_joints = getattr(robot, "get_active_joints", None)
    if not callable(get_active_joints):
        raise PolicyStateContractError("Panda robot must expose get_active_joints()")
    try:
        joints = tuple(get_active_joints())
    except TypeError as error:
        raise PolicyStateContractError(
            "Panda get_active_joints() must return an iterable of named joints"
        ) from error

    names: list[str] = []
    for index, joint in enumerate(joints):
        name = getattr(joint, "name", None)
        if not isinstance(name, str) or not name:
            raise PolicyStateContractError(
                f"Panda active joint {index} must expose a non-empty string name"
            )
        names.append(name)
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise PolicyStateContractError(
            "Panda active joint names must be unique; duplicates=" + repr(duplicates)
        )
    return tuple(names)


def panda_policy_state_v0_indices(robot: PandaPolicyStateRobot) -> tuple[int, ...]:
    """Map stable PandaPolicyStateV0 components to the robot's qpos columns."""
    active_names = get_active_joint_names(robot)
    expected = PANDA_POLICY_STATE_COMPONENTS
    missing = tuple(name for name in expected if name not in active_names)
    unexpected = tuple(name for name in active_names if name not in expected)
    if missing or unexpected or len(active_names) != _POLICY_STATE_SIZE:
        raise PolicyStateContractError(
            "Panda active joints do not match PandaPolicyStateV0; "
            f"missing={missing!r}, unexpected={unexpected!r}, "
            f"count={len(active_names)}"
        )
    return tuple(active_names.index(name) for name in expected)


def validate_panda_policy_state_v0(value: object) -> Float32PolicyState:
    """Validate one semantic policy-state vector and return an independent copy."""
    array = _as_numpy(value, label=_POLICY_STATE_NAME)
    expected_shape = (_POLICY_STATE_SIZE,)
    if array.shape != expected_shape:
        raise PolicyStateContractError(
            f"{_POLICY_STATE_NAME} must have shape {expected_shape}, got {array.shape}"
        )
    if array.dtype != np.dtype(np.float32):
        raise PolicyStateContractError(f"{_POLICY_STATE_NAME} must use float32, got {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise PolicyStateContractError(f"{_POLICY_STATE_NAME} must contain only finite values")
    return np.array(array, dtype=np.float32, copy=True, order="C")


def extract_panda_policy_state_v0(robot: PandaPolicyStateRobot) -> Float32PolicyState:
    """Extract one finite float32 state in stable semantic joint order.

    ManiSkill's single-environment Panda returns qpos with shape ``(1, 9)``.
    The second dimension follows ``get_active_joints()`` order, so the explicit
    name-derived gather keeps the output schema stable if an upstream robot
    implementation changes that physical ordering.
    """
    indices = panda_policy_state_v0_indices(robot)
    get_qpos = getattr(robot, "get_qpos", None)
    if not callable(get_qpos):
        raise PolicyStateContractError("Panda robot must expose get_qpos()")
    qpos = _as_numpy(get_qpos(), label="Panda qpos")
    expected_shape = (1, _POLICY_STATE_SIZE)
    if qpos.shape != expected_shape:
        raise PolicyStateContractError(
            f"single-environment Panda qpos must have shape {expected_shape}, got {qpos.shape}"
        )
    if qpos.dtype != np.dtype(np.float32):
        raise PolicyStateContractError(f"Panda qpos must use float32, got {qpos.dtype}")
    if not np.all(np.isfinite(qpos)):
        raise PolicyStateContractError("Panda qpos must contain only finite values")
    ordered = qpos[0, np.asarray(indices, dtype=np.intp)]
    return validate_panda_policy_state_v0(ordered)


def _as_numpy(value: object, *, label: str) -> np.ndarray:
    """Materialize NumPy data, including a detached CPU torch-like tensor."""
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
        raise PolicyStateContractError(f"{label} must be numeric array data") from error


__all__ = [
    "PANDA_POLICY_STATE_COMPONENTS",
    "Float32PolicyState",
    "NamedActiveJoint",
    "PandaPolicyStateRobot",
    "PolicyStateContractError",
    "extract_panda_policy_state_v0",
    "get_active_joint_names",
    "panda_policy_state_v0_indices",
    "validate_panda_policy_state_v0",
]
