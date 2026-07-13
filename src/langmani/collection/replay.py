"""Project-owned single-environment action replay validation for M3A.

The installed ManiSkill 3.0.1 replay CLI is intentionally not used as the
acceptance oracle: its parallel path drops reset ``options`` (and therefore the
LangMani ``TaskSpec``), while neither path reports the semantic and tolerance
evidence required by M3A.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from langmani.datasets.types import (
    CollectionConfig,
    ReplayFailureCode,
    ReplayValidationMode,
    ReplayValidationResult,
    ScheduledEpisode,
)
from langmani.environments.specs import TaskSpec
from langmani.experts.command_support import create_expert_environment

ReplayEnvironmentFactory = Callable[[str], Any]
STATE_ROUND_TRIP_ABS_TOLERANCE = 1e-6


class ReplayContractError(ValueError):
    """Raised before simulation when recorded replay metadata is malformed."""

    def __init__(
        self,
        message: str,
        *,
        failure_code: ReplayFailureCode = ReplayFailureCode.RECORDED_TRAJECTORY_INVALID,
    ) -> None:
        super().__init__(message)
        self.failure_code = failure_code


def _default_environment_factory(sim_backend: str) -> Any:
    return create_expert_environment(
        diagnostic_rendering=False,
        sim_backend=sim_backend,
    )


def validate_action_replay(
    h5_path: str | Path,
    *,
    native_episode_id: int,
    scheduled_episode: ScheduledEpisode,
    config: CollectionConfig,
    sim_backend: str,
    environment_factory: ReplayEnvironmentFactory | None = None,
    environment: Any | None = None,
) -> ReplayValidationResult:
    """Replay every recorded action and compare semantic and final-state evidence.

    A normal seeded reset is authoritative.  The recorded initial state is not
    injected before action replay, so a pass demonstrates that the reset contract
    plus actions reproduce success.  State restoration is a separate audit after
    action replay and never substitutes for it.
    """
    source_h5 = Path(h5_path).resolve()
    source_json = source_h5.with_suffix(".json")
    episode, actions, states, recorded_success = _load_replay_inputs(
        source_h5,
        source_json,
        native_episode_id=native_episode_id,
        scheduled_episode=scheduled_episode,
        expected_control_mode=config.control_mode,
    )
    recorded_steps = int(actions.shape[0])
    replayed_steps = 0
    final_evaluation: dict[str, bool] = {}
    task_matches = False
    replay_success = False
    wrong_object = False
    wrong_bin = False
    off_table = False
    position_error: float | None = None
    orientation_error: float | None = None
    joint_error: float | None = None
    failure_codes: list[ReplayFailureCode] = []
    failure_reasons: list[str] = []
    state_audit_performed = (
        config.replay_validation_mode is ReplayValidationMode.ACTION_AND_STATE_AUDIT
    )
    state_audit_passed: bool | None = False if state_audit_performed else None

    def record_failure(code: ReplayFailureCode, reason: str) -> None:
        failure_codes.append(code)
        failure_reasons.append(reason)

    if not recorded_success:
        record_failure(
            ReplayFailureCode.RECORDED_SUCCESS_INVALID,
            "recorded terminal success/fail evidence is not a valid success",
        )

    factory = environment_factory or _default_environment_factory
    env: Any | None = environment
    owns_environment = environment is None
    try:
        if env is None:
            env = factory(sim_backend)
        reset_kwargs = _validated_reset_kwargs(episode, scheduled_episode)
        env.reset(**reset_kwargs)
        base_env = getattr(env, "unwrapped", env)
        if getattr(base_env, "control_mode", None) != config.control_mode:
            raise ReplayContractError(
                "replay environment control mode differs from the recorded control mode"
            )
        task_matches = _active_task_matches(base_env, scheduled_episode)
        if not task_matches:
            record_failure(
                ReplayFailureCode.TASK_SPEC_MISMATCH,
                "active TaskSpec differs from the scheduled episode",
            )
        try:
            _require_state_round_trip(
                base_env,
                states[0],
                label="seeded reset initial state",
            )
        except Exception as error:
            record_failure(
                ReplayFailureCode.INITIAL_STATE_MISMATCH,
                "seeded reset initial-state comparison raised "
                f"{type(error).__name__}: {str(error) or repr(error)}",
            )

        action_exception: Exception | None = None
        try:
            action_space = _replay_action_space(env)
            for action_index, action in enumerate(actions):
                bounds_error = _action_bounds_error(action_space, action)
                if bounds_error is not None:
                    record_failure(
                        ReplayFailureCode.ACTION_OUT_OF_BOUNDS,
                        f"recorded action {action_index} is outside the replay action space: "
                        f"{bounds_error}",
                    )
                    break
                # The recorded float32 (8,) row is passed through unchanged.  M3A
                # never normalizes, clips, reshapes, or converts control modes.
                env.step(action)
                replayed_steps += 1
        except Exception as error:  # replay evidence must retain a bounded failure
            action_exception = error
            record_failure(
                ReplayFailureCode.ACTION_EXECUTION_FAILURE,
                f"action replay raised {type(error).__name__}: {str(error) or repr(error)}",
            )

        if action_exception is None and replayed_steps == recorded_steps:
            try:
                final_evaluation = _evaluation(base_env)
                replay_success = final_evaluation.get("success", False)
                wrong_object = final_evaluation.get("wrong_object_in_target_bin", False)
                wrong_bin = final_evaluation.get("target_in_wrong_bin", False)
                off_table = final_evaluation.get("target_off_table", False)
                replay_pose, replay_qpos = _target_pose_and_qpos(base_env)
                base_env.set_state_dict(states[-1])
                recorded_pose, recorded_qpos = _target_pose_and_qpos(base_env)
                position_error = float(np.linalg.norm(replay_pose[:3] - recorded_pose[:3]))
                orientation_error = _quaternion_angle(replay_pose[3:], recorded_pose[3:])
                joint_error = float(np.max(np.abs(replay_qpos - recorded_qpos)))
            except Exception as error:
                record_failure(
                    ReplayFailureCode.FINAL_STATE_COMPARISON_FAILURE,
                    "final recorded-state comparison raised "
                    f"{type(error).__name__}: {str(error) or repr(error)}",
                )

        if not replay_success:
            record_failure(
                ReplayFailureCode.FINAL_SUCCESS_FAILURE,
                "action replay did not finish with M1 success",
            )
        if wrong_object:
            record_failure(
                ReplayFailureCode.WRONG_OBJECT_IN_TARGET_BIN,
                "a wrong object occupies the target bin after replay",
            )
        if wrong_bin:
            record_failure(
                ReplayFailureCode.TARGET_IN_WRONG_BIN,
                "the target occupies the wrong bin after replay",
            )
        if off_table:
            record_failure(
                ReplayFailureCode.TARGET_OFF_TABLE,
                "the target is off the table after replay",
            )
        if replayed_steps != recorded_steps:
            record_failure(
                ReplayFailureCode.ACTION_COUNT_MISMATCH,
                f"replayed {replayed_steps} of {recorded_steps} recorded actions",
            )
        _append_tolerance_failure(
            failure_codes,
            failure_reasons,
            code=ReplayFailureCode.FINAL_POSITION_TOLERANCE,
            label="target position",
            value=position_error,
            tolerance=config.final_position_tolerance_m,
            unit="m",
        )
        _append_tolerance_failure(
            failure_codes,
            failure_reasons,
            code=ReplayFailureCode.FINAL_ORIENTATION_TOLERANCE,
            label="target orientation",
            value=orientation_error,
            tolerance=config.final_orientation_tolerance_rad,
            unit="rad",
        )
        _append_tolerance_failure(
            failure_codes,
            failure_reasons,
            code=ReplayFailureCode.FINAL_JOINT_TOLERANCE,
            label="Panda joint state",
            value=joint_error,
            tolerance=config.final_joint_tolerance_rad,
            unit="rad",
        )

        if state_audit_performed:
            try:
                env.reset(**reset_kwargs)
                if not _active_task_matches(base_env, scheduled_episode):
                    raise ReplayContractError("state-audit reset changed the active TaskSpec")
                for state_index, state in enumerate(states):
                    base_env.set_state_dict(state)
                    _require_state_round_trip(
                        base_env,
                        state,
                        label=f"recorded state {state_index}",
                    )
                state_evaluation = _evaluation(base_env)
                state_audit_passed = (
                    state_evaluation.get("success", False)
                    and not state_evaluation.get("fail", False)
                    and not state_evaluation.get("target_in_wrong_bin", False)
                    and not state_evaluation.get("wrong_object_in_target_bin", False)
                )
                if not state_audit_passed:
                    record_failure(
                        ReplayFailureCode.STATE_AUDIT_FAILURE,
                        "recorded state audit did not finish with valid success",
                    )
            except Exception as error:
                state_audit_passed = False
                record_failure(
                    ReplayFailureCode.STATE_AUDIT_FAILURE,
                    f"state audit raised {type(error).__name__}: {str(error) or repr(error)}",
                )
    except Exception as error:
        record_failure(
            ReplayFailureCode.INITIALIZATION_FAILURE,
            f"replay initialization raised {type(error).__name__}: {str(error) or repr(error)}",
        )
    finally:
        if owns_environment and env is not None:
            try:
                env.close()
            except Exception as error:
                record_failure(
                    ReplayFailureCode.ENVIRONMENT_CLOSE_FAILURE,
                    f"replay close raised {type(error).__name__}: {str(error) or repr(error)}",
                )

    # Preserve first occurrence order while preventing repeated derived messages.
    unique_codes = tuple(dict.fromkeys(failure_codes))
    unique_reasons = tuple(dict.fromkeys(failure_reasons))
    passed = (
        recorded_success
        and replay_success
        and task_matches
        and not wrong_object
        and not wrong_bin
        and not off_table
        and replayed_steps == recorded_steps
        and final_evaluation.get("success") is True
        and final_evaluation.get("fail") is False
        and state_audit_passed is not False
        and not unique_codes
        and not unique_reasons
    )
    return ReplayValidationResult(
        passed=passed,
        mode=config.replay_validation_mode,
        recorded_success=recorded_success,
        replay_success=replay_success,
        task_spec_matches=task_matches,
        wrong_object_in_target_bin=wrong_object,
        target_in_wrong_bin=wrong_bin,
        target_off_table=off_table,
        final_environment_evaluation=final_evaluation,
        recorded_action_steps=recorded_steps,
        replayed_action_steps=replayed_steps,
        final_position_error_m=position_error,
        final_orientation_error_rad=orientation_error,
        final_joint_error_rad=joint_error,
        state_audit_performed=state_audit_performed,
        state_audit_passed=state_audit_passed,
        failure_codes=unique_codes,
        failure_reasons=unique_reasons,
    )


def _load_replay_inputs(
    h5_path: Path,
    json_path: Path,
    *,
    native_episode_id: int,
    scheduled_episode: ScheduledEpisode,
    expected_control_mode: str,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]], bool]:
    if isinstance(native_episode_id, bool) or not isinstance(native_episode_id, int):
        raise TypeError("native_episode_id must be an integer")
    if native_episode_id < 0:
        raise ValueError("native_episode_id must be non-negative")
    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReplayContractError(f"cannot read replay JSON {json_path}: {error}") from error
    if not isinstance(metadata, dict) or not isinstance(metadata.get("episodes"), list):
        raise ReplayContractError("replay JSON must contain an episodes list")
    matching = [
        episode
        for episode in metadata["episodes"]
        if isinstance(episode, dict) and episode.get("episode_id") == native_episode_id
    ]
    if len(matching) != 1:
        raise ReplayContractError(
            f"expected one JSON episode {native_episode_id}, found {len(matching)}"
        )
    episode = matching[0]
    _validated_reset_kwargs(episode, scheduled_episode)
    if episode.get("control_mode") != expected_control_mode:
        raise ReplayContractError(
            "recorded control mode differs from the configured replay control mode"
        )
    group_name = f"traj_{native_episode_id}"
    try:
        with h5py.File(h5_path, "r") as archive:
            if group_name not in archive or not isinstance(archive[group_name], h5py.Group):
                raise ReplayContractError(f"missing HDF5 group {group_name}")
            group = archive[group_name]
            actions_node = group.get("actions")
            if not isinstance(actions_node, h5py.Dataset):
                raise ReplayContractError("recorded episode is missing the actions dataset")
            actions = np.asarray(actions_node)
            if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] != 8:
                raise ReplayContractError("recorded actions must have shape (T, 8) with T > 0")
            if actions.dtype != np.dtype(np.float32):
                raise ReplayContractError("recorded actions must use float32 without conversion")
            if not np.all(np.isfinite(actions)):
                raise ReplayContractError("recorded actions contain non-finite values")
            transition_count = int(actions.shape[0])
            if episode.get("elapsed_steps") != transition_count:
                raise ReplayContractError("JSON elapsed_steps must equal the recorded action count")
            _require_time_vector(group, "terminated", transition_count)
            _require_time_vector(group, "truncated", transition_count)
            success = _require_time_vector(group, "success", transition_count)
            fail = _require_time_vector(group, "fail", transition_count)
            if "env_states" not in group or not isinstance(group["env_states"], h5py.Group):
                raise ReplayContractError("recorded episode is missing env_states")
            state_count = transition_count + 1
            _require_state_time_contract(group["env_states"], state_count)
            states = [_state_at(group["env_states"], index) for index in range(state_count)]
            recorded_success = (
                episode.get("success") is True
                and episode.get("fail") is False
                and bool(success[-1])
                and not bool(fail[-1])
            )
    except OSError as error:
        raise ReplayContractError(f"cannot read replay HDF5 {h5_path}: {error}") from error
    return episode, actions, states, recorded_success


def _validated_reset_kwargs(
    episode: Mapping[str, Any], scheduled_episode: ScheduledEpisode
) -> dict[str, Any]:
    reset_kwargs = episode.get("reset_kwargs")
    if not isinstance(reset_kwargs, Mapping):
        raise ReplayContractError("episode.reset_kwargs must be a mapping")
    seed = reset_kwargs.get("seed")
    if seed != episode.get("episode_seed") or seed != scheduled_episode.scene_seed:
        raise ReplayContractError("reset seed, episode seed, and scheduled seed must agree")
    options = reset_kwargs.get("options")
    if not isinstance(options, Mapping):
        raise ReplayContractError("reset_kwargs.options must be a mapping")
    raw_task = options.get("task_spec")
    if not isinstance(raw_task, Mapping):
        raise ReplayContractError("reset_kwargs.options.task_spec must be a mapping")
    try:
        task_spec = TaskSpec.from_mapping(raw_task)
    except (TypeError, ValueError) as error:
        raise ReplayContractError(f"invalid recorded TaskSpec: {error}") from error
    if task_spec != scheduled_episode.task_spec:
        raise ReplayContractError("recorded TaskSpec differs from the schedule")
    # JSON round-trip prevents mutating objects owned by the parsed metadata.
    return json.loads(json.dumps(dict(reset_kwargs)))


def _require_time_vector(
    group: h5py.Group,
    name: str,
    transition_count: int,
) -> np.ndarray:
    node = group.get(name)
    if not isinstance(node, h5py.Dataset):
        raise ReplayContractError(f"recorded episode is missing the {name} dataset")
    values = np.asarray(node)
    if values.shape != (transition_count,):
        raise ReplayContractError(
            f"recorded {name} must have shape ({transition_count},), got {values.shape}"
        )
    if values.dtype != np.dtype(np.bool_):
        raise ReplayContractError(f"recorded {name} must use boolean dtype")
    return values


def _require_state_time_contract(node: h5py.Group, state_count: int) -> None:
    leaf_count = 0

    def visit(current: h5py.Group | h5py.Dataset) -> None:
        nonlocal leaf_count
        if isinstance(current, h5py.Group):
            if not current:
                raise ReplayContractError(f"recorded state group {current.name} must not be empty")
            for key in current:
                visit(current[key])
            return
        leaf_count += 1
        if current.ndim < 1 or current.shape[0] != state_count:
            raise ReplayContractError(
                f"state leaf {current.name} must contain exactly {state_count} states"
            )
        if not (np.issubdtype(current.dtype, np.number) or np.issubdtype(current.dtype, np.bool_)):
            raise ReplayContractError(f"state leaf {current.name} must be numeric or boolean")
        values = np.asarray(current)
        if not np.all(np.isfinite(values)):
            raise ReplayContractError(f"state leaf {current.name} contains non-finite values")

    visit(node)
    if leaf_count == 0:
        raise ReplayContractError("recorded env_states must contain at least one numeric leaf")


def _state_at(node: h5py.Group | h5py.Dataset, index: int) -> Any:
    if isinstance(node, h5py.Group):
        return {key: _state_at(node[key], index) for key in node}
    if index >= len(node):
        raise ReplayContractError(
            f"state leaf {node.name} has length {len(node)}, expected at least {index + 1}"
        )
    return np.asarray(node[index])


def _replay_action_space(env: Any) -> Any:
    getter = getattr(env, "get_wrapper_attr", None)
    if callable(getter):
        try:
            candidate = getter("single_action_space")
        except (AttributeError, LookupError):
            candidate = None
        if candidate is not None:
            return candidate
    candidate = getattr(env, "single_action_space", None)
    if candidate is not None:
        return candidate
    candidate = getattr(env, "action_space", None)
    if candidate is None:
        raise ReplayContractError("replay environment is missing an action space")
    return candidate


def _action_bounds_error(action_space: Any, action: np.ndarray) -> str | None:
    if not hasattr(action_space, "low") or not hasattr(action_space, "high"):
        raise ReplayContractError("replay action space must expose explicit low/high bounds")
    low = _numeric_array(action_space.low, label="action-space low")
    high = _numeric_array(action_space.high, label="action-space high")
    if low.shape == (1, *action.shape):
        low = low[0]
    if high.shape == (1, *action.shape):
        high = high[0]
    try:
        low = np.broadcast_to(low, action.shape)
        high = np.broadcast_to(high, action.shape)
    except ValueError as error:
        raise ReplayContractError(
            f"action-space bounds cannot be broadcast to recorded action shape {action.shape}"
        ) from error
    if np.any(low > high):
        raise ReplayContractError("replay action space has an invalid lower/upper bound")
    invalid = np.flatnonzero((action < low) | (action > high))
    if invalid.size == 0:
        return None
    index = int(invalid[0])
    return (
        f"component {index} value {float(action[index]):.9g} is outside "
        f"[{float(low[index]):.9g}, {float(high[index]):.9g}]"
    )


def _active_task_matches(base_env: Any, scheduled_episode: ScheduledEpisode) -> bool:
    specs = base_env.get_episode_specs()
    return (
        isinstance(specs, tuple)
        and len(specs) == 1
        and specs[0].scene_seed == scheduled_episode.scene_seed
        and specs[0].scene_id == scheduled_episode.scene_id
        and specs[0].task_spec == scheduled_episode.task_spec
        and specs[0].task_id == scheduled_episode.task_id
    )


def _evaluation(base_env: Any) -> dict[str, bool]:
    raw = base_env.get_expert_evaluation()
    if not isinstance(raw, Mapping):
        raise ReplayContractError("get_expert_evaluation() must return a mapping")
    return {
        str(key): _single_bool(value, label=f"evaluation[{key!r}]") for key, value in raw.items()
    }


def _target_pose_and_qpos(base_env: Any) -> tuple[np.ndarray, np.ndarray]:
    context = base_env.get_expert_task_context()
    pose = _numeric_array(context.target_object.actor.pose.raw_pose, label="target pose")
    qpos = _numeric_array(context.robot.get_qpos(), label="Panda qpos")
    if pose.shape != (1, 7):
        raise ReplayContractError(f"target pose must have shape (1, 7), got {pose.shape}")
    if qpos.shape != (1, 9):
        raise ReplayContractError(f"Panda qpos must have shape (1, 9), got {qpos.shape}")
    return pose[0].astype(np.float64, copy=True), qpos[0].astype(np.float64, copy=True)


def _require_state_round_trip(
    base_env: Any,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    get_state_dict = getattr(base_env, "get_state_dict", None)
    if not callable(get_state_dict):
        raise ReplayContractError("environment is missing get_state_dict()")
    actual = get_state_dict()
    maximum_error, maximum_path = _state_tree_error(actual, expected, path="state")
    if maximum_error > STATE_ROUND_TRIP_ABS_TOLERANCE:
        raise ReplayContractError(
            f"{label} differs at {maximum_path}: maximum absolute error "
            f"{maximum_error:.9g} exceeds {STATE_ROUND_TRIP_ABS_TOLERANCE:.9g}"
        )


def _state_tree_error(actual: Any, expected: Any, *, path: str) -> tuple[float, str]:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise ReplayContractError(f"{path} must be a mapping")
        if set(actual) != set(expected):
            raise ReplayContractError(
                f"{path} keys differ; actual={sorted(actual)}, expected={sorted(expected)}"
            )
        maximum = (0.0, path)
        for key in sorted(expected):
            candidate = _state_tree_error(
                actual[key],
                expected[key],
                path=f"{path}.{key}",
            )
            if candidate[0] > maximum[0]:
                maximum = candidate
        return maximum

    actual_array = _numeric_array(actual, label=path)
    expected_array = _numeric_array(expected, label=f"recorded {path}")
    if actual_array.shape != expected_array.shape:
        if actual_array.shape == (1, *expected_array.shape):
            actual_array = actual_array[0]
        elif expected_array.shape == (1, *actual_array.shape):
            expected_array = expected_array[0]
    if actual_array.shape != expected_array.shape:
        raise ReplayContractError(
            f"{path} shape differs; actual={actual_array.shape}, expected={expected_array.shape}"
        )
    if actual_array.size == 0:
        return 0.0, path
    if np.issubdtype(actual_array.dtype, np.bool_) or np.issubdtype(expected_array.dtype, np.bool_):
        return (0.0, path) if np.array_equal(actual_array, expected_array) else (math.inf, path)
    error = float(
        np.max(
            np.abs(
                actual_array.astype(np.float64, copy=False)
                - expected_array.astype(np.float64, copy=False)
            )
        )
    )
    return error, path


def _numeric_array(value: object, *, label: str) -> np.ndarray:
    converted = value
    for method_name in ("detach", "cpu", "numpy"):
        method = getattr(converted, method_name, None)
        if callable(method):
            converted = method()
    try:
        array = np.asarray(converted)
    except (TypeError, ValueError) as error:
        raise ReplayContractError(f"{label} must be numeric") from error
    if not np.all(np.isfinite(array)):
        raise ReplayContractError(f"{label} must contain finite values")
    return array


def _single_bool(value: object, *, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    array = _numeric_array(value, label=label)
    if array.size != 1 or not np.issubdtype(array.dtype, np.bool_):
        raise ReplayContractError(f"{label} must contain one boolean value")
    return bool(array.reshape(-1)[0])


def _quaternion_angle(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if not math.isclose(left_norm, 1.0, abs_tol=1e-5, rel_tol=0.0) or not math.isclose(
        right_norm, 1.0, abs_tol=1e-5, rel_tol=0.0
    ):
        raise ReplayContractError("recorded and replay quaternions must be unit length")
    alignment = float(abs(np.dot(left, right)))
    return 2.0 * math.acos(min(1.0, max(0.0, alignment)))


def _append_tolerance_failure(
    codes: list[ReplayFailureCode],
    reasons: list[str],
    *,
    code: ReplayFailureCode,
    label: str,
    value: float | None,
    tolerance: float,
    unit: str,
) -> None:
    if value is None:
        if not any("comparison raised" in reason for reason in reasons):
            codes.append(code)
            reasons.append(f"{label} error is unavailable")
    elif value > tolerance:
        codes.append(code)
        reasons.append(f"{label} error {value:.8f} {unit} exceeds tolerance {tolerance:.8f} {unit}")


__all__ = [
    "ReplayContractError",
    "ReplayEnvironmentFactory",
    "validate_action_replay",
]
