"""Deterministic privileged planar-pushing expert for LangMani 2.0."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from langmani.environments.push_expert_state import (
    PushExpertContextError,
    PushExpertTaskContext,
)
from langmani.environments.push_specs import PUSH_OBJECT_IDS, TARGET_REGION_IDS, PushTaskSpec
from langmani.environments.push_to_region import (
    ENV_ID,
    OBJECT_PLANAR_RADII,
    WORKSPACE_BOUNDS_XY,
)
from langmani.experts.planner import (
    MplibPandaPlannerAdapter,
    PlannerAdapter,
    PlannerContractError,
    PlannerFailure,
    PlannerImportError,
    PlannerVersionError,
)
from langmani.experts.push_diagnostics import PushDiagnosticSnapshot, planar_diagnostics
from langmani.experts.push_strategy import (
    LateralApproachCandidate,
    build_lateral_approach_candidates,
    is_lateral_region,
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
        self._diagnostic_trace: list[PushDiagnosticSnapshot] = []
        self._initial_target_position: np.ndarray | None = None
        self._initial_push_direction: np.ndarray | None = None
        self._chosen_precontact_point: np.ndarray | None = None
        self._chosen_contact_point: np.ndarray | None = None
        self._active_push_direction: np.ndarray | None = None
        self._approach_candidates: tuple[LateralApproachCandidate, ...] = ()

    @property
    def action_trace(self) -> tuple[np.ndarray, ...]:
        """Return copied actions for recorder-side quality and replay checks."""

        return tuple(action.copy() for action in self._actions)

    @property
    def diagnostic_trace(self) -> tuple[PushDiagnosticSnapshot, ...]:
        """Return immutable phase-boundary privileged telemetry for offline diagnosis."""

        return tuple(self._diagnostic_trace)

    @property
    def approach_candidate_diagnostics(self) -> tuple[dict[str, object], ...]:
        return tuple(candidate.to_dict() for candidate in self._approach_candidates)

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
                self._capture_diagnostic_snapshot(result.phase.value)
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
        self._initial_target_position = self._target_position()
        self._initial_push_direction = self._push_direction(self._initial_target_position)
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
        candidates = self._lateral_approach_candidates(object_position, direction)
        return self._planned_lateral_approach(candidates)

    def _establish_contact(self) -> PushPhaseResult:
        object_position = self._target_position()
        direction = self._motion_direction(object_position)
        pose = self._tcp_pose()
        pose[:2] = object_position[:2] - direction * self._contact_offset()
        pose[2] = self._push_height()
        self._chosen_contact_point = pose[:3].copy()
        return self._planned_motion(PushExpertPhase.ESTABLISH_CONTACT, (pose,))

    def _primary_push(self) -> PushPhaseResult:
        phase = PushExpertPhase.PRIMARY_PUSH
        object_position = self._target_position()
        direct = self._planned_motion(
            phase,
            (self._push_endpoint_pose(object_position),),
            stop_on_target_inside_region=True,
        )
        if (
            direct.success
            or direct.environment_steps
            or direct.status
            not in {
                PushExpertStatus.IK_FAILURE,
                PushExpertStatus.PLANNING_FAILURE,
            }
        ):
            return replace(direct, message="executed one content-bound direct primary push")
        fallback = self._adaptive_primary_push()
        return replace(
            fallback,
            attempts=direct.attempts + fallback.attempts,
            environment_steps=direct.environment_steps + fallback.environment_steps,
            planning_calls=direct.planning_calls + fallback.planning_calls,
            planning_duration_seconds=(
                direct.planning_duration_seconds + fallback.planning_duration_seconds
            ),
            execution_duration_seconds=(
                direct.execution_duration_seconds + fallback.execution_duration_seconds
            ),
            message="direct primary planning failed before execution; " + fallback.message,
        )

    def _adaptive_primary_push(self) -> PushPhaseResult:
        """Advance in bounded state-aware segments until full containment."""

        phase = PushExpertPhase.PRIMARY_PUSH
        before = self._environment_steps
        planning_calls = 0
        planning_duration = 0.0
        execution_duration = 0.0
        attempts = 0
        last_status: str | None = None
        for _segment in range(self.config.maximum_primary_push_segments):
            if self._terminal_success or self._evaluation().get("target_inside_region") is True:
                return self._success(
                    phase,
                    "bounded incremental push reached full containment",
                    attempts=max(attempts, 1),
                    steps=self._environment_steps - before,
                    planning_calls=planning_calls,
                    planning_duration=planning_duration,
                    execution_duration=execution_duration,
                    planner_status=last_status,
                )
            object_position = self._target_position()
            direction = self._push_direction(object_position)
            center_progress = float(
                np.dot(self._target_center()[:2] - object_position[:2], direction)
            )
            if center_progress <= 0.0:
                break
            increment = max(
                self.config.minimum_primary_push_increment,
                min(self.config.primary_push_increment, center_progress),
            )
            segment_result: PushPhaseResult | None = None
            while increment + 1e-12 >= self.config.minimum_primary_push_increment:
                attempts += 1
                pose = self._tcp_pose()
                pose[:2] = (
                    object_position[:2] - direction * self._contact_offset() + direction * increment
                )
                pose[2] = self._push_height()
                result = self._planned_motion(
                    phase,
                    (pose,),
                    stop_on_target_inside_region=True,
                )
                planning_calls += result.planning_calls
                planning_duration += result.planning_duration_seconds
                execution_duration += result.execution_duration_seconds
                last_status = result.planner_status
                segment_result = result
                if result.success:
                    break
                if result.environment_steps or result.status not in {
                    PushExpertStatus.IK_FAILURE,
                    PushExpertStatus.PLANNING_FAILURE,
                }:
                    return replace(
                        result,
                        attempts=attempts,
                        environment_steps=self._environment_steps - before,
                        planning_calls=planning_calls,
                        planning_duration_seconds=planning_duration,
                        execution_duration_seconds=execution_duration,
                    )
                increment *= 0.5
            if segment_result is None or not segment_result.success:
                assert segment_result is not None
                return replace(
                    segment_result,
                    attempts=attempts,
                    environment_steps=self._environment_steps - before,
                    planning_calls=planning_calls,
                    planning_duration_seconds=planning_duration,
                    execution_duration_seconds=execution_duration,
                    message="all bounded primary-push increments failed planning",
                )
        evaluation = self._evaluation()
        if self._terminal_success or evaluation.get("target_inside_region") is True:
            return self._success(
                phase,
                "bounded incremental push reached full containment",
                attempts=max(attempts, 1),
                steps=self._environment_steps - before,
                planning_calls=planning_calls,
                planning_duration=planning_duration,
                execution_duration=execution_duration,
                planner_status=last_status,
            )
        return self._failure(
            phase,
            PushExpertStatus.CORRECTION_FAILURE,
            "bounded incremental primary push did not reach full containment",
            attempts=max(attempts, 1),
            steps=self._environment_steps - before,
            planning_calls=planning_calls,
            planning_duration=planning_duration,
            execution_duration=execution_duration,
            planner_status=last_status,
        )

    def _corrective_push(self, phase: PushExpertPhase) -> PushPhaseResult:
        if self._terminal_success or self._evaluation().get("target_inside_region") is True:
            return self._success(phase, "target already reached the region; no correction needed")
        object_position = self._target_position()
        if self._is_lateral_task():
            candidates = self._lateral_approach_candidates(
                object_position,
                self._push_direction(object_position),
            )
            safe = next((candidate for candidate in candidates if candidate.safe), None)
            if safe is None:
                return self._failure(
                    phase,
                    PushExpertStatus.CORRECTION_FAILURE,
                    "no workspace- and obstacle-safe lateral re-contact candidate",
                )
            self._active_push_direction = np.asarray(safe.contact_direction)
        direction = self._motion_direction(object_position)
        current = self._tcp_pose()
        lift = current.copy()
        lift[2] = self.config.precontact_staging_height
        behind_high = current.copy()
        behind_high[:2] = object_position[:2] - direction * (
            self._contact_offset() + self.config.precontact_clearance
        )
        behind_high[2] = self.config.precontact_staging_height
        contact = behind_high.copy()
        contact[:2] = object_position[:2] - direction * self._contact_offset()
        contact[2] = self._push_height()
        push = self._push_endpoint_pose(object_position)
        free_space = self._planned_motion(
            phase,
            (lift, behind_high),
            action_stride=self.config.free_space_action_stride,
        )
        if not free_space.success:
            return free_space
        contact_push = self._planned_motion(
            phase,
            (contact, push),
            stop_on_target_inside_region=True,
        )
        return replace(
            contact_push,
            attempts=free_space.attempts + contact_push.attempts,
            environment_steps=free_space.environment_steps + contact_push.environment_steps,
            planning_calls=free_space.planning_calls + contact_push.planning_calls,
            planning_duration_seconds=(
                free_space.planning_duration_seconds + contact_push.planning_duration_seconds
            ),
            execution_duration_seconds=(
                free_space.execution_duration_seconds + contact_push.execution_duration_seconds
            ),
            message="executed sparse free-space re-contact and dense corrective push",
        )

    def _push_endpoint_pose(self, object_position: np.ndarray) -> np.ndarray:
        """Return a conservative TCP endpoint just inside full containment.

        The target object follows ahead of the TCP by its planar support radius.
        Stopping at a margin inside the exact full-containment center radius
        shortens the push while retaining a deterministic geometric buffer.
        """

        context = self._require_context()
        direction = self._motion_direction(object_position)
        goal_margin = (
            self.config.region_goal_margin
            if context.target_object.object_id == "blue_cube"
            else self.config.cylinder_region_goal_margin
        )
        object_center_distance = max(0.0, context.full_containment_center_radius - goal_margin)
        pose = self._tcp_pose()
        pose[:2] = self._target_center()[:2] - direction * (
            object_center_distance + context.target_object_planar_radius
        )
        pose[2] = self._push_height()
        return pose

    def _push_height(self) -> float:
        object_id = self._require_context().target_object.object_id
        if object_id == "blue_cube":
            return self.config.push_height
        if object_id == "orange_cylinder":
            return self.config.cylinder_push_height
        raise _PushAbort(
            PushExpertStatus.INVALID_TASK,
            f"push height is undefined for {object_id!r}",
        )

    def _contact_offset(self) -> float:
        return self.config.contact_offset

    def _lateral_compensation_degrees(self) -> float:
        if not self._is_lateral_task():
            return 0.0
        object_id = self._require_context().target_object.object_id
        if object_id == "blue_cube":
            return self.config.lateral_cube_compensation_degrees
        if object_id == "orange_cylinder":
            return self.config.lateral_cylinder_compensation_degrees
        raise _PushAbort(
            PushExpertStatus.INVALID_TASK,
            f"lateral compensation is undefined for {object_id!r}",
        )

    def _is_lateral_task(self) -> bool:
        return is_lateral_region(self._require_context().episode_spec.task_spec.target_region_id)

    def _motion_direction(self, object_position: np.ndarray) -> np.ndarray:
        if self._active_push_direction is not None:
            return self._active_push_direction.copy()
        return self._push_direction(object_position)

    def _lateral_approach_candidates(
        self,
        object_position: np.ndarray,
        desired_direction: np.ndarray,
    ) -> tuple[LateralApproachCandidate, ...]:
        context = self._require_context()
        distractors = tuple(
            (
                _pose7(item.actor.pose.raw_pose)[:2],
                OBJECT_PLANAR_RADII[PUSH_OBJECT_IDS.index(item.object_id)],
            )
            for item in context.objects
            if item.object_id != context.target_object.object_id
        )
        candidates = build_lateral_approach_candidates(
            desired_direction=desired_direction,
            object_position=object_position,
            tcp_position=self._tcp_pose(),
            distractors=distractors,
            contact_offset=self._contact_offset(),
            precontact_clearance=self.config.precontact_clearance,
            precontact_height=self.config.precontact_height,
            push_height=self._push_height(),
            workspace_bounds_xy=WORKSPACE_BOUNDS_XY,
            compensation_degrees=self._lateral_compensation_degrees(),
            minimum_obstacle_clearance=self.config.minimum_approach_obstacle_clearance,
        )
        self._approach_candidates = candidates
        return candidates

    def _planned_lateral_approach(
        self,
        candidates: Sequence[LateralApproachCandidate],
    ) -> PushPhaseResult:
        before = self._environment_steps
        planning_calls = 0
        planning_duration = 0.0
        execution_duration = 0.0
        attempts = 0
        last_result: PushPhaseResult | None = None
        lift = self._tcp_pose()
        lift[2] = self.config.precontact_staging_height
        lift_result = self._planned_motion(
            PushExpertPhase.MOVE_TO_PRECONTACT,
            (lift,),
            action_stride=self.config.free_space_action_stride,
        )
        planning_calls += lift_result.planning_calls
        planning_duration += lift_result.planning_duration_seconds
        execution_duration += lift_result.execution_duration_seconds
        if not lift_result.success:
            return lift_result
        for candidate in candidates:
            if not candidate.safe:
                continue
            attempts += 1
            self._chosen_precontact_point = np.asarray(candidate.precontact_point)
            staging = self._pose_with_position(candidate.precontact_point)
            staging[2] = self.config.precontact_staging_height
            staging_result = self._planned_motion(
                PushExpertPhase.MOVE_TO_PRECONTACT,
                (staging,),
            )
            planning_calls += staging_result.planning_calls
            planning_duration += staging_result.planning_duration_seconds
            execution_duration += staging_result.execution_duration_seconds
            if not staging_result.success:
                last_result = staging_result
                if self._recoverable_zero_step_plan_failure(staging_result):
                    continue
                return replace(
                    staging_result,
                    attempts=attempts,
                    environment_steps=self._environment_steps - before,
                    planning_calls=planning_calls,
                    planning_duration_seconds=planning_duration,
                    execution_duration_seconds=execution_duration,
                )
            descent_result = self._planned_motion(
                PushExpertPhase.MOVE_TO_PRECONTACT,
                (self._pose_with_position(candidate.precontact_point),),
                action_stride=self.config.free_space_action_stride,
            )
            planning_calls += descent_result.planning_calls
            planning_duration += descent_result.planning_duration_seconds
            execution_duration += descent_result.execution_duration_seconds
            last_result = descent_result
            if descent_result.success:
                self._active_push_direction = np.asarray(candidate.contact_direction)
                return replace(
                    descent_result,
                    attempts=attempts,
                    environment_steps=self._environment_steps - before,
                    planning_calls=planning_calls,
                    planning_duration_seconds=planning_duration,
                    execution_duration_seconds=execution_duration,
                    message=(
                        "executed direction-aware lateral precontact at "
                        f"{candidate.angle_degrees:+.1f} degrees"
                    ),
                )
            if self._recoverable_zero_step_plan_failure(descent_result):
                continue
            return replace(
                descent_result,
                attempts=attempts,
                environment_steps=self._environment_steps - before,
                planning_calls=planning_calls,
                planning_duration_seconds=planning_duration,
                execution_duration_seconds=execution_duration,
            )
        if last_result is None:
            return self._failure(
                PushExpertPhase.MOVE_TO_PRECONTACT,
                PushExpertStatus.CONTACT_FAILURE,
                "no workspace- and obstacle-safe lateral approach candidate",
            )
        return replace(
            last_result,
            attempts=attempts,
            environment_steps=self._environment_steps - before,
            planning_calls=planning_calls,
            planning_duration_seconds=planning_duration,
            execution_duration_seconds=execution_duration,
            message="all safe lateral approach candidates failed planning",
        )

    @staticmethod
    def _recoverable_zero_step_plan_failure(result: PushPhaseResult) -> bool:
        return result.environment_steps == 0 and (
            result.status in {PushExpertStatus.IK_FAILURE, PushExpertStatus.PLANNING_FAILURE}
            or (
                result.status is PushExpertStatus.EXECUTION_FAILURE
                and result.planner_status == "ActionBoundsPrecheck"
            )
        )

    def _pose_with_position(self, position: Sequence[float]) -> np.ndarray:
        pose = self._tcp_pose()
        pose[:3] = np.asarray(position, dtype=np.float64)
        return pose

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
        *,
        action_stride: int = 1,
        stop_on_target_inside_region: bool = False,
    ) -> PushPhaseResult:
        if (
            not isinstance(action_stride, int)
            or isinstance(action_stride, bool)
            or action_stride < 1
        ):
            raise ValueError("action_stride must be a positive integer")
        started = time.perf_counter()
        before = self._environment_steps
        planning_duration = 0.0
        planning_calls = 0
        last_status: str | None = None
        target_inside_region = False
        try:
            for target in targets:
                if self._terminal_success or target_inside_region:
                    break
                planner = self._require_planner()
                planner.synchronize()
                planning_started = time.perf_counter()
                plan = planner.plan_pose(tuple(float(value) for value in target))
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
                planned_actions = tuple(
                    np.asarray((*arm_position, CLOSED_GRIPPER), dtype=np.float64)
                    for arm_position in plan.positions
                )
                if action_stride > 1 and len(planned_actions) > 1:
                    sampled = planned_actions[::action_stride]
                    if sampled[-1] is not planned_actions[-1]:
                        sampled = (*sampled, planned_actions[-1])
                    planned_actions = sampled
                low, high = self._action_bounds()
                if any(np.any(action < low) or np.any(action > high) for action in planned_actions):
                    return self._failure(
                        phase,
                        PushExpertStatus.EXECUTION_FAILURE,
                        "planned joint action exceeds controller bounds before execution",
                        attempts=1,
                        steps=self._environment_steps - before,
                        planning_calls=planning_calls,
                        planning_duration=planning_duration,
                        execution_duration=max(
                            0.0, time.perf_counter() - started - planning_duration
                        ),
                        planner_status="ActionBoundsPrecheck",
                    )
                for action in planned_actions:
                    if self._terminal_success:
                        break
                    self._step_action(action)
                    if (
                        stop_on_target_inside_region
                        and self._evaluation().get("target_inside_region") is True
                    ):
                        target_inside_region = True
                        break
            if target_inside_region and not self._terminal_success:
                self._hold(self.config.settle_steps)
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
        if not self._terminal_success and not target_inside_region and targets:
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
        result = cast(Any, self.environment).step(action)
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
        raw = cast(Any, self.base).get_push_expert_evaluation()
        if not isinstance(raw, Mapping):
            raise RuntimeError("get_push_expert_evaluation() must return a mapping")
        return {str(key): _json_scalar(value, str(key)) for key, value in raw.items()}

    def _action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        getter = getattr(self.base, "get_push_expert_action_bounds", None)
        if not callable(getter):
            raise RuntimeError("environment lacks get_push_expert_action_bounds()")
        raw = getter()
        if not isinstance(raw, tuple) or len(raw) != 2:
            raise RuntimeError("push expert action bounds must be a (low, high) tuple")
        low = np.asarray(raw[0], dtype=np.float64)
        high = np.asarray(raw[1], dtype=np.float64)
        if (
            low.shape != (PANDA_ACTION_DOF,)
            or high.shape != (PANDA_ACTION_DOF,)
            or not np.isfinite(low).all()
            or not np.isfinite(high).all()
            or np.any(low > high)
        ):
            raise RuntimeError("push expert action bounds are malformed")
        return low, high

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

        frame = _numeric(cast(Any, self.environment).render(), "diagnostic render")
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

    def _capture_diagnostic_snapshot(self, phase: str) -> None:
        context = self._context
        initial = self._initial_target_position
        direction = self._initial_push_direction
        if context is None or initial is None or direction is None:
            return
        target_pose = self._target_position_pose()
        target_position = target_pose[:3]
        center = self._target_center()
        distance, progress, lateral = planar_diagnostics(
            initial_position=initial,
            current_position=target_position,
            target_center=center,
            push_direction=direction,
        )
        tcp = self._tcp_pose()
        contact_distance = float(np.linalg.norm(tcp[:2] - target_position[:2]))
        x_min, x_max, y_min, y_max = WORKSPACE_BOUNDS_XY
        radius = context.target_object_planar_radius
        workspace_margin = min(
            target_position[0] - radius - x_min,
            x_max - target_position[0] - radius,
            target_position[1] - radius - y_min,
            y_max - target_position[1] - radius,
        )
        evaluation = self._evaluation()
        object_poses = {
            item.object_id: tuple(float(value) for value in _pose7(item.actor.pose.raw_pose))
            for item in context.objects
        }
        self._diagnostic_trace.append(
            PushDiagnosticSnapshot(
                phase=phase,
                environment_steps=self._environment_steps,
                target_object_pose=tuple(float(value) for value in target_pose),
                object_poses=object_poses,
                target_center=tuple(float(value) for value in center),
                tcp_pose=tuple(float(value) for value in tcp),
                intended_push_direction=tuple(float(value) for value in direction),
                chosen_precontact_point=(
                    tuple(float(value) for value in self._chosen_precontact_point)
                    if self._chosen_precontact_point is not None
                    else None
                ),
                chosen_contact_point=(
                    tuple(float(value) for value in self._chosen_contact_point)
                    if self._chosen_contact_point is not None
                    else None
                ),
                target_distance=distance,
                projected_progress=progress,
                lateral_error=lateral,
                contact_proxy=contact_distance <= self.config.contact_offset + 0.025,
                contact_proxy_distance=contact_distance,
                workspace_margin=float(workspace_margin),
                target_inside_region=bool(evaluation.get("target_inside_region", False)),
                target_is_static=bool(evaluation.get("target_is_static", False)),
                stable_success_steps=int(evaluation.get("stable_success_steps", 0)),
            )
        )

    def _target_position_pose(self) -> np.ndarray:
        return _pose7(self._require_context().target_object.actor.pose.raw_pose)

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
