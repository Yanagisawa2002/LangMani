"""Deterministic privileged planar-pushing expert for LangMani 2.0."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

from langmani.environments.push_expert_state import (
    PushExpertContextError,
    PushExpertTaskContext,
)
from langmani.environments.push_specs import PUSH_OBJECT_IDS, TARGET_REGION_IDS, PushTaskSpec
from langmani.environments.push_to_region import ENV_ID
from langmani.experts.planner import (
    MplibPandaPlannerAdapter,
    PlannerAdapter,
    PlannerContractError,
    PlannerFailure,
    PlannerImportError,
    PlannerVersionError,
)
from langmani.experts.push_types import (
    PUSH_EXPERT_PHASE_SEQUENCE,
    PushExpertConfig,
    PushExpertPhase,
    PushExpertResult,
    PushExpertStatus,
    PushPhaseResult,
)

PANDA_ARM_DOF = 7
PANDA_ACTION_DOF = 8
CLOSED_GRIPPER = -1.0


class _PushAbort(RuntimeError):
    def __init__(self, status: PushExpertStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


PlannerFactory = Callable[[object], PlannerAdapter]


def _default_planner_factory(environment: object) -> PlannerAdapter:
    return MplibPandaPlannerAdapter(environment)


class PushToRegionExpert:
    """Planar contact expert with bounded, state-dependent corrective pushes.

    The gripper is closed before contact and never re-opened around the object.
    The TCP remains at the declared planar pushing height; grasp, lift, topple,
    workspace exit, and wrong-object displacement are explicit aborts.
    """

    def __init__(
        self,
        environment: object,
        *,
        config: PushExpertConfig | None = None,
        planner_factory: PlannerFactory | None = None,
        diagnostic_directory: str | Path | None = None,
    ) -> None:
        self.environment = environment
        self.base = getattr(environment, "unwrapped", environment)
        self.config = config or PushExpertConfig()
        self._planner_factory = planner_factory or _default_planner_factory
        self._diagnostic_directory = (
            Path(diagnostic_directory) if diagnostic_directory is not None else None
        )
        self._planner: PlannerAdapter | None = None
        self._context: PushExpertTaskContext | None = None
        self._phase_results: list[PushPhaseResult] = []
        self._actions: list[np.ndarray] = []
        self._environment_steps = 0
        self._planning_calls = 0
        self._planning_duration = 0.0
        self._execution_duration = 0.0
        self._terminal_success = False
        self._rollout_started_at: float | None = None

    @property
    def action_trace(self) -> tuple[np.ndarray, ...]:
        """Return copied actions for recorder-side quality and replay checks."""

        return tuple(action.copy() for action in self._actions)

    def run(self) -> PushExpertResult:
        self._rollout_started_at = time.perf_counter()
        phases = (
            self._initialize,
            self._close_gripper,
            self._move_to_precontact,
            self._establish_contact,
            self._primary_push,
            lambda: self._corrective_push(PushExpertPhase.CORRECTIVE_PUSH_1),
            lambda: self._corrective_push(PushExpertPhase.CORRECTIVE_PUSH_2),
            self._settle,
            self._verify_task,
        )
        try:
            for expected, operation in zip(PUSH_EXPERT_PHASE_SEQUENCE, phases, strict=True):
                result = operation()
                if result.phase is not expected:
                    raise RuntimeError(f"push phase returned {result.phase}, expected {expected}")
                self._phase_results.append(result)
                self._capture_phase_frame(result.phase)
                if not result.success:
                    return self._build_result(result.status)
            return self._build_result(PushExpertStatus.SUCCESS)
        finally:
            if self._planner is not None:
                self._planner.close()

    def unexpected_exception_result(self, error: BaseException) -> PushExpertResult:
        successful = tuple(item for item in self._phase_results if item.success)
        failed_phase = (
            PUSH_EXPERT_PHASE_SEQUENCE[len(successful)]
            if len(successful) < len(PUSH_EXPERT_PHASE_SEQUENCE)
            else None
        )
        context = self._context
        episode = context.episode_spec if context is not None else None
        return PushExpertResult(
            success=False,
            status=PushExpertStatus.UNEXPECTED_EXCEPTION,
            scene_seed=episode.scene_seed if episode is not None else None,
            scene_id=episode.scene_id if episode is not None else None,
            task_id=episode.task_id if episode is not None else None,
            canonical_instruction=(episode.canonical_instruction if episode is not None else None),
            target_object_id=(episode.task_spec.target_object_id if episode is not None else None),
            target_region_id=(episode.task_spec.target_region_id if episode is not None else None),
            difficulty=episode.task_spec.difficulty if episode is not None else None,
            total_environment_steps=self._environment_steps,
            total_planning_calls=self._planning_calls,
            completed_phases=tuple(item.phase for item in successful),
            failed_phase=failed_phase,
            final_environment_evaluation={},
            phase_results=successful,
            planning_duration_seconds=self._planning_duration,
            execution_duration_seconds=self._elapsed_execution(),
            exception_type=type(error).__name__,
            exception_message=str(error) or repr(error),
        )

    def _initialize(self) -> PushPhaseResult:
        started = time.perf_counter()
        if getattr(self.base, "num_envs", None) != 1:
            return self._failure(
                PushExpertPhase.INITIALIZE,
                PushExpertStatus.INITIALIZATION_FAILURE,
                "push expert requires num_envs=1",
                execution_duration=time.perf_counter() - started,
            )
        if getattr(self.base, "control_mode", None) != self.config.control_mode:
            return self._failure(
                PushExpertPhase.INITIALIZE,
                PushExpertStatus.INITIALIZATION_FAILURE,
                f"push expert requires {self.config.control_mode}",
                execution_duration=time.perf_counter() - started,
            )
        context_accessor = getattr(self.base, "get_push_expert_task_context", None)
        evaluation_accessor = getattr(self.base, "get_push_expert_evaluation", None)
        if not callable(context_accessor) or not callable(evaluation_accessor):
            return self._failure(
                PushExpertPhase.INITIALIZE,
                PushExpertStatus.INITIALIZATION_FAILURE,
                "environment lacks explicit push expert accessors",
                execution_duration=time.perf_counter() - started,
            )
        try:
            context = context_accessor()
        except PushExpertContextError as error:
            return self._failure(
                PushExpertPhase.INITIALIZE,
                PushExpertStatus.INITIALIZATION_FAILURE,
                str(error),
                execution_duration=time.perf_counter() - started,
            )
        if not isinstance(context, PushExpertTaskContext) or context.environment_id != ENV_ID:
            return self._failure(
                PushExpertPhase.INITIALIZE,
                PushExpertStatus.INITIALIZATION_FAILURE,
                "environment returned an incompatible push expert context",
                execution_duration=time.perf_counter() - started,
            )
        task = context.episode_spec.task_spec
        if not isinstance(task, PushTaskSpec):
            return self._failure(
                PushExpertPhase.INITIALIZE,
                PushExpertStatus.INVALID_TASK,
                "active episode lacks PushTaskSpec",
                execution_duration=time.perf_counter() - started,
            )
        if (
            task.target_object_id not in PUSH_OBJECT_IDS
            or task.target_region_id not in TARGET_REGION_IDS
            or context.target_object.object_id != task.target_object_id
        ):
            return self._failure(
                PushExpertPhase.INITIALIZE,
                PushExpertStatus.INVALID_TASK,
                "push semantic handles disagree with PushTaskSpec",
                execution_duration=time.perf_counter() - started,
            )
        self._context = context
        terminal = self._event_abort()
        if terminal is not None:
            return self._failure(
                PushExpertPhase.INITIALIZE,
                terminal.status,
                str(terminal),
                execution_duration=time.perf_counter() - started,
            )

        planning_started = time.perf_counter()
        try:
            planner = self._planner_factory(self.environment)
            if not isinstance(planner, PlannerAdapter):
                raise PlannerContractError("planner factory returned an incompatible adapter")
            planner.synchronize()
        except (PlannerImportError, PlannerVersionError, PlannerContractError) as error:
            duration = time.perf_counter() - planning_started
            self._planning_duration += duration
            return self._failure(
                PushExpertPhase.INITIALIZE,
                PushExpertStatus.INITIALIZATION_FAILURE,
                f"planner initialization failed: {type(error).__name__}: {error}",
                planning_duration=duration,
                execution_duration=planning_started - started,
            )
        self._planner = planner
        duration = time.perf_counter() - planning_started
        self._planning_duration += duration
        return self._success(
            PushExpertPhase.INITIALIZE,
            "validated task, Panda, evaluation, and mplib contracts",
            attempts=1,
            planning_duration=duration,
            execution_duration=planning_started - started,
        )

    def _close_gripper(self) -> PushPhaseResult:
        started = time.perf_counter()
        before = self._environment_steps
        try:
            self._hold(self.config.gripper_close_steps)
        except _PushAbort as error:
            return self._failure(
                PushExpertPhase.CLOSE_GRIPPER,
                error.status,
                str(error),
                attempts=1,
                steps=self._environment_steps - before,
                execution_duration=time.perf_counter() - started,
            )
        return self._success(
            PushExpertPhase.CLOSE_GRIPPER,
            "closed the gripper before contact without grasping the target",
            attempts=1,
            steps=self._environment_steps - before,
            execution_duration=time.perf_counter() - started,
        )

    def _move_to_precontact(self) -> PushPhaseResult:
        object_position = self._target_position()
        direction = self._push_direction(object_position)
        pose = self._tcp_pose()
        pose[:2] = object_position[:2] - direction * (
            self.config.contact_offset + self.config.precontact_clearance
        )
        pose[2] = self.config.precontact_height
        return self._planned_motion(PushExpertPhase.MOVE_TO_PRECONTACT, (pose,))

    def _establish_contact(self) -> PushPhaseResult:
        object_position = self._target_position()
        direction = self._push_direction(object_position)
        pose = self._tcp_pose()
        pose[:2] = object_position[:2] - direction * self.config.contact_offset
        pose[2] = self.config.push_height
        return self._planned_motion(PushExpertPhase.ESTABLISH_CONTACT, (pose,))

    def _primary_push(self) -> PushPhaseResult:
        object_position = self._target_position()
        direction = self._push_direction(object_position)
        pose = self._tcp_pose()
        pose[:2] = self._target_center()[:2] - direction * self.config.contact_offset
        pose[2] = self.config.push_height
        return self._planned_motion(PushExpertPhase.PRIMARY_PUSH, (pose,))

    def _corrective_push(self, phase: PushExpertPhase) -> PushPhaseResult:
        if self._terminal_success or self._evaluation().get("target_inside_region") is True:
            return self._success(phase, "target already reached the region; no correction needed")
        object_position = self._target_position()
        direction = self._push_direction(object_position)
        current = self._tcp_pose()
        lift = current.copy()
        lift[2] = self.config.precontact_height
        behind_high = current.copy()
        behind_high[:2] = object_position[:2] - direction * (
            self.config.contact_offset + self.config.precontact_clearance
        )
        behind_high[2] = self.config.precontact_height
        contact = behind_high.copy()
        contact[:2] = object_position[:2] - direction * self.config.contact_offset
        contact[2] = self.config.push_height
        push = contact.copy()
        push[:2] = self._target_center()[:2] - direction * self.config.contact_offset
        return self._planned_motion(phase, (lift, behind_high, contact, push))

    def _settle(self) -> PushPhaseResult:
        started = time.perf_counter()
        before = self._environment_steps
        try:
            self._hold(self.config.settle_steps)
        except _PushAbort as error:
            return self._failure(
                PushExpertPhase.SETTLE,
                error.status,
                str(error),
                attempts=1,
                steps=self._environment_steps - before,
                execution_duration=time.perf_counter() - started,
            )
        return self._success(
            PushExpertPhase.SETTLE,
            "held the planar pose for stable-success evaluation",
            attempts=1,
            steps=self._environment_steps - before,
            execution_duration=time.perf_counter() - started,
        )

    def _verify_task(self) -> PushPhaseResult:
        started = time.perf_counter()
        evaluation = self._evaluation()
        if evaluation.get("success") is True:
            return self._success(
                PushExpertPhase.VERIFY_TASK,
                "stable full-containment push success passed",
                attempts=1,
                execution_duration=time.perf_counter() - started,
            )
        return self._failure(
            PushExpertPhase.VERIFY_TASK,
            PushExpertStatus.VERIFICATION_FAILURE,
            "target did not satisfy stable full-containment success",
            attempts=1,
            execution_duration=time.perf_counter() - started,
        )

    def _planned_motion(
        self,
        phase: PushExpertPhase,
        targets: Sequence[np.ndarray],
    ) -> PushPhaseResult:
        started = time.perf_counter()
        before = self._environment_steps
        planning_duration = 0.0
        planning_calls = 0
        last_status: str | None = None
        try:
            for target in targets:
                if self._terminal_success:
                    break
                planner = self._require_planner()
                planner.synchronize()
                planning_started = time.perf_counter()
                plan = planner.plan_pose(target)
                duration = time.perf_counter() - planning_started
                planning_duration += duration
                self._planning_duration += duration
                self._planning_calls += 1
                planning_calls += 1
                last_status = plan.status
                if not plan.success:
                    status = (
                        PushExpertStatus.IK_FAILURE
                        if plan.failure is PlannerFailure.IK_FAILURE
                        else PushExpertStatus.PLANNING_FAILURE
                    )
                    return self._failure(
                        phase,
                        status,
                        f"planner failed: {plan.status}",
                        attempts=1,
                        steps=self._environment_steps - before,
                        planning_calls=planning_calls,
                        planning_duration=planning_duration,
                        execution_duration=time.perf_counter() - started - planning_duration,
                        planner_status=last_status,
                    )
                for arm_position in plan.positions:
                    if self._terminal_success:
                        break
                    self._step_action(np.asarray((*arm_position, CLOSED_GRIPPER)))
        except _PushAbort as error:
            return self._failure(
                phase,
                error.status,
                str(error),
                attempts=1,
                steps=self._environment_steps - before,
                planning_calls=planning_calls,
                planning_duration=planning_duration,
                execution_duration=max(0.0, time.perf_counter() - started - planning_duration),
                planner_status=last_status,
            )
        if not self._terminal_success and targets:
            position_error = float(np.linalg.norm(self._tcp_pose()[:3] - targets[-1][:3]))
            if position_error > self.config.tcp_position_tolerance:
                return self._failure(
                    phase,
                    PushExpertStatus.EXECUTION_FAILURE,
                    f"TCP position error {position_error:.4f} m exceeds tolerance",
                    attempts=1,
                    steps=self._environment_steps - before,
                    planning_calls=planning_calls,
                    planning_duration=planning_duration,
                    execution_duration=max(0.0, time.perf_counter() - started - planning_duration),
                    planner_status=last_status,
                )
        return self._success(
            phase,
            "executed deterministic planar motion",
            attempts=1,
            steps=self._environment_steps - before,
            planning_calls=planning_calls,
            planning_duration=planning_duration,
            execution_duration=max(0.0, time.perf_counter() - started - planning_duration),
            planner_status=last_status,
        )

    def _hold(self, steps: int) -> None:
        arm = self._arm_qpos()
        for _ in range(steps):
            if self._terminal_success:
                break
            self._step_action(np.asarray((*arm, CLOSED_GRIPPER)))

    def _step_action(self, action: np.ndarray) -> None:
        if self._environment_steps >= self.config.maximum_episode_steps:
            raise _PushAbort(PushExpertStatus.TIMEOUT, "expert step budget exhausted")
        if action.shape != (PANDA_ACTION_DOF,) or not np.isfinite(action).all():
            raise _PushAbort(PushExpertStatus.EXECUTION_FAILURE, "expert action is invalid")
        result = self.environment.step(action)
        self._actions.append(action.astype(np.float32, copy=True))
        self._environment_steps += 1
        if not isinstance(result, tuple) or len(result) != 5:
            raise _PushAbort(
                PushExpertStatus.EXECUTION_FAILURE,
                "environment step did not return five values",
            )
        _, _, terminated, truncated, _ = result
        evaluation = self._evaluation()
        if evaluation.get("success") is True:
            self._terminal_success = True
            return
        event_abort = self._event_abort(evaluation)
        if event_abort is not None:
            raise event_abort
        if _single_bool(truncated, "truncated"):
            raise _PushAbort(PushExpertStatus.TIMEOUT, "environment time limit reached")
        if _single_bool(terminated, "terminated"):
            raise _PushAbort(
                PushExpertStatus.EXECUTION_FAILURE,
                "environment terminated without success",
            )

    def _event_abort(
        self,
        evaluation: Mapping[str, bool | int | float] | None = None,
    ) -> _PushAbort | None:
        current = evaluation or self._evaluation()
        for key, status, message in (
            (
                "target_outside_workspace",
                PushExpertStatus.TARGET_OUTSIDE_WORKSPACE,
                "target left the valid workspace",
            ),
            ("target_lifted", PushExpertStatus.TARGET_LIFTED, "target was lifted"),
            ("target_toppled", PushExpertStatus.TARGET_TOPPLED, "target toppled"),
            (
                "target_is_grasped",
                PushExpertStatus.CONTACT_FAILURE,
                "target was grasped during planar pushing",
            ),
            (
                "wrong_object_displaced",
                PushExpertStatus.WRONG_OBJECT_INTERACTION,
                "wrong object was displaced",
            ),
            (
                "invalid_action",
                PushExpertStatus.EXECUTION_FAILURE,
                "environment rejected an invalid action",
            ),
            (
                "action_out_of_bounds",
                PushExpertStatus.EXECUTION_FAILURE,
                "expert emitted an out-of-bounds action",
            ),
        ):
            if current.get(key) is True:
                return _PushAbort(status, message)
        return None

    def _evaluation(self) -> dict[str, bool | int | float]:
        raw = self.base.get_push_expert_evaluation()
        if not isinstance(raw, Mapping):
            raise RuntimeError("get_push_expert_evaluation() must return a mapping")
        return {str(key): _json_scalar(value, str(key)) for key, value in raw.items()}

    def _target_position(self) -> np.ndarray:
        return _pose7(self._require_context().target_object.actor.pose.raw_pose)[:3]

    def _target_center(self) -> np.ndarray:
        return _numeric(self._require_context().target_region_center, "target center").reshape(3)

    def _push_direction(self, object_position: np.ndarray) -> np.ndarray:
        delta = self._target_center()[:2] - object_position[:2]
        norm = float(np.linalg.norm(delta))
        if not np.isfinite(norm) or norm < 1e-6:
            raise _PushAbort(PushExpertStatus.CONTACT_FAILURE, "push direction is undefined")
        return delta / norm

    def _arm_qpos(self) -> np.ndarray:
        qpos = _numeric(self._require_context().robot.get_qpos(), "Panda qpos")
        if qpos.shape != (1, 9):
            raise RuntimeError(f"Panda qpos must have shape (1,9), got {qpos.shape}")
        return qpos[0, :PANDA_ARM_DOF].copy()

    def _tcp_pose(self) -> np.ndarray:
        return _pose7(self._require_context().agent.tcp.pose.raw_pose)

    def _build_result(self, status: PushExpertStatus) -> PushExpertResult:
        context = self._context
        episode = context.episode_spec if context is not None else None
        completed = tuple(item.phase for item in self._phase_results if item.success)
        failed = next((item.phase for item in self._phase_results if not item.success), None)
        return PushExpertResult(
            success=status is PushExpertStatus.SUCCESS,
            status=status,
            scene_seed=episode.scene_seed if episode is not None else None,
            scene_id=episode.scene_id if episode is not None else None,
            task_id=episode.task_id if episode is not None else None,
            canonical_instruction=(episode.canonical_instruction if episode is not None else None),
            target_object_id=(episode.task_spec.target_object_id if episode is not None else None),
            target_region_id=(episode.task_spec.target_region_id if episode is not None else None),
            difficulty=episode.task_spec.difficulty if episode is not None else None,
            total_environment_steps=self._environment_steps,
            total_planning_calls=self._planning_calls,
            completed_phases=completed,
            failed_phase=failed,
            final_environment_evaluation=self._evaluation() if context is not None else {},
            phase_results=tuple(self._phase_results),
            planning_duration_seconds=self._planning_duration,
            execution_duration_seconds=self._elapsed_execution(),
        )

    def _elapsed_execution(self) -> float:
        if self._rollout_started_at is None:
            return 0.0
        return max(
            0.0,
            time.perf_counter() - self._rollout_started_at - self._planning_duration,
        )

    def _capture_phase_frame(self, phase: PushExpertPhase) -> None:
        if not self.config.diagnostic_rendering:
            return
        if self._diagnostic_directory is None:
            raise RuntimeError("diagnostic_directory is required for rendering")
        from PIL import Image

        frame = _numeric(self.environment.render(), "diagnostic render")
        if frame.ndim == 4 and frame.shape[0] == 1:
            frame = frame[0]
        if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
            raise RuntimeError(f"diagnostic render has invalid shape {frame.shape}")
        if np.issubdtype(frame.dtype, np.floating):
            frame = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8, copy=False)
        self._diagnostic_directory.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(
            self._diagnostic_directory / f"{len(self._phase_results):02d}_{phase.value}.png"
        )

    def _require_context(self) -> PushExpertTaskContext:
        if self._context is None:
            raise RuntimeError("push expert context is unavailable")
        return self._context

    def _require_planner(self) -> PlannerAdapter:
        if self._planner is None:
            raise RuntimeError("push planner is unavailable")
        return self._planner

    @staticmethod
    def _success(
        phase: PushExpertPhase,
        message: str,
        *,
        attempts: int = 0,
        steps: int = 0,
        planning_calls: int = 0,
        planning_duration: float = 0.0,
        execution_duration: float = 0.0,
        planner_status: str | None = None,
    ) -> PushPhaseResult:
        return PushPhaseResult(
            phase=phase,
            success=True,
            status=PushExpertStatus.SUCCESS,
            attempts=attempts,
            environment_steps=steps,
            planning_calls=planning_calls,
            planning_duration_seconds=planning_duration,
            execution_duration_seconds=execution_duration,
            message=message,
            planner_status=planner_status,
        )

    @staticmethod
    def _failure(
        phase: PushExpertPhase,
        status: PushExpertStatus,
        message: str,
        *,
        attempts: int = 0,
        steps: int = 0,
        planning_calls: int = 0,
        planning_duration: float = 0.0,
        execution_duration: float = 0.0,
        planner_status: str | None = None,
    ) -> PushPhaseResult:
        return PushPhaseResult(
            phase=phase,
            success=False,
            status=status,
            attempts=attempts,
            environment_steps=steps,
            planning_calls=planning_calls,
            planning_duration_seconds=planning_duration,
            execution_duration_seconds=execution_duration,
            message=message,
            planner_status=planner_status,
        )


def _numeric(value: object, label: str) -> np.ndarray:
    converted = value
    for method_name in ("detach", "cpu"):
        method = getattr(converted, method_name, None)
        if callable(method):
            converted = method()
    numpy_method = getattr(converted, "numpy", None)
    if callable(numpy_method):
        converted = numpy_method()
    try:
        result = np.asarray(converted)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} must be numeric") from error
    if not np.isfinite(result).all():
        raise RuntimeError(f"{label} must be finite")
    return result


def _pose7(value: object) -> np.ndarray:
    pose = _numeric(value, "pose")
    if pose.shape != (1, 7):
        raise RuntimeError(f"pose must have shape (1,7), got {pose.shape}")
    result = pose[0].astype(np.float64, copy=True)
    if not np.isclose(np.linalg.norm(result[3:]), 1.0, atol=1e-5, rtol=0.0):
        raise RuntimeError("pose quaternion must have unit length")
    return result


def _single_bool(value: object, label: str) -> bool:
    result = _numeric(value, label)
    if result.size != 1 or result.reshape(-1)[0] not in (True, False):
        raise RuntimeError(f"{label} must contain one boolean")
    return bool(result.reshape(-1)[0])


def _json_scalar(value: object, label: str) -> bool | int | float:
    array = _numeric(value, label)
    if array.size != 1:
        raise RuntimeError(f"{label} must contain one scalar")
    scalar = array.reshape(-1)[0]
    if isinstance(scalar, (bool, np.bool_)):
        return bool(scalar)
    if isinstance(scalar, np.integer):
        return int(scalar)
    return float(scalar)


__all__ = ["PushToRegionExpert"]
