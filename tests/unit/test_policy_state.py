from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from langmani.datasets.policy_state import (
    PANDA_POLICY_STATE_COMPONENTS,
    PolicyStateContractError,
    extract_panda_policy_state_v0,
    get_active_joint_names,
    panda_policy_state_v0_indices,
    validate_panda_policy_state_v0,
)


@dataclass(frozen=True, slots=True)
class _Joint:
    name: str


class _Robot:
    def __init__(self, names: tuple[str, ...], qpos: np.ndarray) -> None:
        self._joints = tuple(_Joint(name) for name in names)
        self._qpos = qpos
        self.active_joint_reads = 0
        self.qpos_reads = 0

    def get_active_joints(self) -> tuple[_Joint, ...]:
        self.active_joint_reads += 1
        return self._joints

    def get_qpos(self) -> np.ndarray:
        self.qpos_reads += 1
        return self._qpos


def _robot(
    *,
    names: tuple[str, ...] = PANDA_POLICY_STATE_COMPONENTS,
    qpos: np.ndarray | None = None,
) -> _Robot:
    if qpos is None:
        qpos = np.arange(len(PANDA_POLICY_STATE_COMPONENTS), dtype=np.float32)[None, :]
    return _Robot(names, qpos)


def test_component_order_is_exact() -> None:
    assert PANDA_POLICY_STATE_COMPONENTS == (
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


def test_extracts_identity_order_as_fresh_float32_vector() -> None:
    robot = _robot()

    state = extract_panda_policy_state_v0(robot)

    np.testing.assert_array_equal(state, np.arange(9, dtype=np.float32))
    assert state.shape == (9,)
    assert state.dtype == np.float32
    assert state.flags.c_contiguous
    state[0] = -1.0
    assert robot.get_qpos()[0, 0] == 0.0
    assert robot.active_joint_reads == 1
    assert robot.qpos_reads == 2


def test_reorders_qpos_by_stable_semantic_names() -> None:
    names = tuple(reversed(PANDA_POLICY_STATE_COMPONENTS))
    values_by_name = {name: float(index + 1) for index, name in enumerate(names)}
    qpos = np.asarray([[values_by_name[name] for name in names]], dtype=np.float32)
    robot = _robot(names=names, qpos=qpos)

    assert panda_policy_state_v0_indices(robot) == tuple(reversed(range(9)))
    state = extract_panda_policy_state_v0(robot)

    np.testing.assert_array_equal(
        state,
        np.asarray(
            [values_by_name[name] for name in PANDA_POLICY_STATE_COMPONENTS],
            dtype=np.float32,
        ),
    )


def test_get_active_joint_names_rejects_missing_api_and_invalid_names() -> None:
    with pytest.raises(PolicyStateContractError, match="get_active_joints"):
        get_active_joint_names(object())  # type: ignore[arg-type]

    class _UnnamedRobot:
        def get_active_joints(self) -> tuple[object, ...]:
            return (object(),)

    with pytest.raises(PolicyStateContractError, match="non-empty string name"):
        get_active_joint_names(_UnnamedRobot())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("names", "message"),
    [
        (
            PANDA_POLICY_STATE_COMPONENTS[:-1],
            "do not match PandaPolicyStateV0",
        ),
        (
            (*PANDA_POLICY_STATE_COMPONENTS[:-1], "unexpected_joint"),
            "unexpected=",
        ),
        (
            (*PANDA_POLICY_STATE_COMPONENTS[:-1], "panda_joint1"),
            "must be unique",
        ),
    ],
)
def test_rejects_invalid_active_joint_contract(names: tuple[str, ...], message: str) -> None:
    with pytest.raises(PolicyStateContractError, match=message):
        panda_policy_state_v0_indices(_robot(names=names))


def test_extract_rejects_missing_qpos_api() -> None:
    class _NoQposRobot:
        def get_active_joints(self) -> tuple[_Joint, ...]:
            return tuple(_Joint(name) for name in PANDA_POLICY_STATE_COMPONENTS)

    with pytest.raises(PolicyStateContractError, match="get_qpos"):
        extract_panda_policy_state_v0(_NoQposRobot())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("qpos", "message"),
    [
        (np.zeros(9, dtype=np.float32), "shape"),
        (np.zeros((2, 9), dtype=np.float32), "shape"),
        (np.zeros((1, 8), dtype=np.float32), "shape"),
        (np.zeros((1, 9), dtype=np.float64), "float32"),
        (np.zeros((1, 9), dtype=np.int64), "float32"),
        (
            np.asarray([[0.0] * 8 + [np.nan]], dtype=np.float32),
            "finite",
        ),
        (
            np.asarray([[0.0] * 8 + [np.inf]], dtype=np.float32),
            "finite",
        ),
    ],
)
def test_extract_rejects_invalid_qpos(qpos: np.ndarray, message: str) -> None:
    with pytest.raises(PolicyStateContractError, match=message):
        extract_panda_policy_state_v0(_robot(qpos=qpos))


def test_standalone_validator_is_strict_and_returns_a_copy() -> None:
    source = np.arange(9, dtype=np.float32)

    validated = validate_panda_policy_state_v0(source)

    np.testing.assert_array_equal(validated, source)
    assert validated is not source
    validated[0] = -1.0
    assert source[0] == 0.0


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.zeros((1, 9), dtype=np.float32), "shape"),
        (np.zeros(9, dtype=np.float64), "float32"),
        (np.asarray([0.0] * 8 + [np.nan], dtype=np.float32), "finite"),
    ],
)
def test_standalone_validator_rejects_invalid_values(value: np.ndarray, message: str) -> None:
    with pytest.raises(PolicyStateContractError, match=message):
        validate_panda_policy_state_v0(value)
