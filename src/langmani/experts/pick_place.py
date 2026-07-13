"""Explicit phase-based privileged expert for the LangMani M1 environment."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from langmani.environments.expert_state import ExpertContextError, ExpertTaskContext
from langmani.environments.pick_place_by_instruction import ENV_ID
from langmani.environments.specs import BIN_IDS, OBJECT_IDS, TaskSpec
from langmani.experts.planner import (
    MplibPandaPlannerAdapter,
    PlannerAdapter,
    PlannerContractError,
    PlannerFailure,
    PlannerImportError,
    PlannerPlanResult,
    PlannerVersionError,
)
from langmani.experts.types import (
    EXPERT_PHASE_SEQUENCE,
    ExpertConfig,
    ExpertPhase,
    ExpertResult,
    ExpertStatus,
    PhaseResult,
)

PANDA_ARM_DOF = 7
PANDA_ACTION_DOF = 8
OPEN_GRIPPER = 1.0
CLOSED_GRIPPER = -1.0
GRIPPER_COMMAND_STEPS = 6
FINAL_WAYPOINT_REFINEMENT_STEPS = 2
TCP_POSITION_TOLERANCE = 0.01
TCP_ORIENTATION_TOLERANCE = 0.08
ARM_JOINT_TOLERANCE = 0.08
PLACEMENT_WALL_SAFETY_MARGIN = 0.01

# Panda.build_grasp_pose places [orthogonal, closing, approaching] in the
# rotation columns.  For approaching=(0, 0, -1) and closing=(0, -1, 0), this is
# a pi rotation about world X, represented in wxyz order as (0, 1, 0, 0).  The
# installed table-scene Panda starts in this same TCP orientation, avoiding the
# exact pi relative rotation that mplib 0.1.1 screw planning rejects.
TOP_GRASP_APPROACH_DIRECTION = (0.0, 0.0, -1.0)
TOP_GRASP_CLOSING_DIRECTION = (0.0, -1.0, 0.0)
TOP_GRASP_QUATERNION_WXYZ = (0.0, 1.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class PhaseContract:
    """Human-auditable entry, command, and completion contract for one phase."""

    entry_condition: str
    command: str
    completion_criterion: str
    failure_status: ExpertStatus


PHASE_CONTRACTS: Mapping[ExpertPhase, PhaseContract] = MappingProxyType(
    {
        ExpertPhase.INITIALIZE: PhaseContract(
            "M1 environment has been reset and the target is not terminal",
            "validate semantic handles and synchronize the mplib adapter",
            "environment, task, robot, control mode, and planner contracts all validate",
            ExpertStatus.INITIALIZATION_FAILURE,
        ),
        ExpertPhase.MOVE_TO_PREGRASP: PhaseContract(
            "target remains on the table",
            "plan and execute to the fixed-orientation pose above the target",
            "TCP reaches the pre-grasp position within tolerance",
            ExpertStatus.PLANNING_FAILURE,
        ),
        ExpertPhase.APPROACH_TARGET: PhaseContract(
            "pre-grasp pose was reached and target remains on the table",
            "descend along world -Z to the target center",
            "TCP reaches the deterministic grasp position within tolerance",
            ExpertStatus.PLANNING_FAILURE,
        ),
        ExpertPhase.CLOSE_GRIPPER: PhaseContract(
            "TCP is at the target grasp pose",
            "hold arm qpos and command normalized gripper -1 for six control steps",
            "all close commands execute without timeout or terminal failure",
            ExpertStatus.EXECUTION_FAILURE,
        ),
        ExpertPhase.VERIFY_GRASP: PhaseContract(
            "close-gripper phase completed",
            "query the target-specific Panda grasp oracle and attach its collision box",
            "the active semantic target is grasped",
            ExpertStatus.GRASP_FAILURE,
        ),
        ExpertPhase.LIFT_TARGET: PhaseContract(
            "active target is grasped",
            "move the TCP vertically to the configured lift clearance",
            "lift pose is reached and target remains grasped",
            ExpertStatus.TRANSPORT_FAILURE,
        ),
        ExpertPhase.MOVE_ABOVE_DESTINATION: PhaseContract(
            "active target remains grasped",
            "transport above the selected bin interior center",
            "above-bin pose is reached and target remains grasped",
            ExpertStatus.TRANSPORT_FAILURE,
        ),
        ExpertPhase.DESCEND_TO_PLACE: PhaseContract(
            "active target remains grasped above the selected bin",
            "descend to bin floor plus cube half extent and placement clearance",
            "placement TCP pose is reached while retaining the target",
            ExpertStatus.PLACEMENT_FAILURE,
        ),
        ExpertPhase.OPEN_GRIPPER: PhaseContract(
            "placement pose was reached",
            "hold arm qpos and command normalized gripper +1 for six control steps",
            "target is released and planner attachment is removed",
            ExpertStatus.PLACEMENT_FAILURE,
        ),
        ExpertPhase.SETTLE_AFTER_RELEASE: PhaseContract(
            "target is released inside the selected bin",
            "hold the open gripper for the configured settling steps",
            "target is in the selected bin, released, and static",
            ExpertStatus.PLACEMENT_FAILURE,
        ),
        ExpertPhase.RETREAT: PhaseContract(
            "released target has settled successfully",
            "move the TCP vertically away from the bin",
            "retreat pose is reached without disturbing the valid placement",
            ExpertStatus.PLACEMENT_FAILURE,
        ),
        ExpertPhase.VERIFY_TASK: PhaseContract(
            "all motion and release phases completed",
            "query the complete M1 evaluation oracle",
            "M1 conservative success is true",
            ExpertStatus.VERIFICATION_FAILURE,
        ),
    }
)


class _ExecutionAbort(RuntimeError):
    """Expected control-loop abort carrying a stable expert status."""

    def __init__(self, status: ExpertStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


PlannerFactory = Callable[[object], PlannerAdapter]


def _default_planner_factory(env: object) -> PlannerAdapter:
    return MplibPandaPlannerAdapter(env)


class PickPlaceExpert:
    """Privileged deterministic top-grasp expert for one active M1 episode.

    The environment must already have been reset.  This class deliberately does
    not catch unknown exceptions; command boundaries are responsible for recording
    ``unexpected_exception`` while preserving exception type and message.
    """

    def __init__(
        self,
        env: object,
        *,
        config: ExpertConfig | None = None,
        planner_factory: PlannerFactory | None = None,
        diagnostic_directory: str | Path | None = None,
    ) -> None:
        self.env = env
        self.base_env = getattr(env, "unwrapped", env)
        self.config = config or ExpertConfig()
        self._planner_factory = planner_factory or _default_planner_factory
        self._diagnostic_directory = (
            Path(diagnostic_directory) if diagnostic_directory is not None else None
        )

        self._planner: PlannerAdapter | None = None
        self._context: ExpertTaskContext | None = None
        self._phase_results: list[PhaseResult] = []
        self._artifact_paths: list[str] = []
        self._environment_steps = 0
        self._planning_calls = 0
        self._replans = 0
        self._rollout_started_at: float | None = None
        self._planning_duration = 0.0
        self._active_planning_started_at: float | None = None
        self._gripper_state = OPEN_GRIPPER
        self._object_relative_position = np.zeros(3, dtype=np.float64)

    def run(self) -> ExpertResult:
        """Execute all twelve phases or stop at the first classified failure."""
        if self._phase_results:
            raise RuntimeError("PickPlaceExpert instances are single-use")
        self._rollout_started_at = time.perf_counter()
        try:
            initialize_result = self._initialize()
            self._record_phase(initialize_result)
            if not initialize_result.success:
                return self._build_result(initialize_result.status)

            phase_methods: tuple[tuple[ExpertPhase, Callable[[], PhaseResult]], ...] = (
                (ExpertPhase.MOVE_TO_PREGRASP, self._move_to_pregrasp),
                (ExpertPhase.APPROACH_TARGET, self._approach_target),
                (ExpertPhase.CLOSE_GRIPPER, self._close_gripper),
                (ExpertPhase.VERIFY_GRASP, self._verify_grasp),
                (ExpertPhase.LIFT_TARGET, self._lift_target),
                (ExpertPhase.MOVE_ABOVE_DESTINATION, self._move_above_destination),
                (ExpertPhase.DESCEND_TO_PLACE, self._descend_to_place),
                (ExpertPhase.OPEN_GRIPPER, self._open_gripper),
                (ExpertPhase.SETTLE_AFTER_RELEASE, self._settle_after_release),
                (ExpertPhase.RETREAT, self._retreat),
                (ExpertPhase.VERIFY_TASK, self._verify_task),
            )
            for expected_phase, phase_method in phase_methods:
                phase_result = phase_method()
                if phase_result.phase is not expected_phase:
                    raise RuntimeError(
                        f"phase method returned {phase_result.phase}, expected {expected_phase}"
                    )
                self._record_phase(phase_result)
                if not phase_result.success:
                    return self._build_result(phase_result.status)
            return self._build_result(ExpertStatus.SUCCESS)
        finally:
            if self._planner is not None:
                self._planner.close()

    def unexpected_exception_result(self, error: BaseException) -> ExpertResult:
        """Format an exception caught by an outer command boundary.

        This method does not catch exceptions and is never called by :meth:`run`.
        It deliberately omits a fresh simulator evaluation so formatting the
        original failure cannot trigger a second simulator-side exception.
        """
        successful_results = tuple(result for result in self._phase_results if result.success)
        completed = tuple(result.phase for result in successful_results)
        failed_phase = (
            EXPERT_PHASE_SEQUENCE[len(completed)]
            if len(completed) < len(EXPERT_PHASE_SEQUENCE)
            else None
        )
        context = self._context
        episode = context.episode_spec if context is not None else None
        message = str(error) or repr(error)
        now = time.perf_counter()
        planning_duration = self._planning_duration
        if self._active_planning_started_at is not None:
            planning_duration += now - self._active_planning_started_at
        total_duration = (
            now - self._rollout_started_at if self._rollout_started_at is not None else 0.0
        )
        return ExpertResult(
            success=False,
            status=ExpertStatus.UNEXPECTED_EXCEPTION,
            scene_seed=episode.scene_seed if episode is not None else None,
            scene_id=episode.scene_id if episode is not None else None,
            task_id=episode.task_id if episode is not None else None,
            canonical_instruction=episode.canonical_instruction if episode is not None else None,
            target_object_id=(episode.task_spec.target_object_id if episode is not None else None),
            target_bin_id=episode.task_spec.target_bin_id if episode is not None else None,
            total_environment_steps=self._environment_steps,
            total_planning_calls=self._planning_calls,
            total_replans=self._replans,
            completed_phases=completed,
            failed_phase=failed_phase,
            final_environment_evaluation={},
            phase_results=successful_results,
            planning_duration_seconds=planning_duration,
            execution_duration_seconds=max(0.0, total_duration - planning_duration),
            diagnostic_artifact_paths=tuple(self._artifact_paths),
            exception_type=type(error).__name__,
            exception_message=message,
        )

    def _initialize(self) -> PhaseResult:
        start = time.perf_counter()
        message: str | None = None
        status = ExpertStatus.INITIALIZATION_FAILURE

        if getattr(self.base_env, "num_envs", None) != 1:
            message = (
                f"M2 expert requires num_envs=1, got {getattr(self.base_env, 'num_envs', None)!r}"
            )
        elif getattr(self.base_env, "control_mode", None) != self.config.control_mode:
            message = (
                f"M2 expert requires control_mode={self.config.control_mode!r}, got "
                f"{getattr(self.base_env, 'control_mode', None)!r}"
            )
        elif not callable(getattr(self.base_env, "get_expert_task_context", None)):
            message = "environment is missing get_expert_task_context()"
        elif not callable(getattr(self.base_env, "get_expert_evaluation", None)):
            message = "environment is missing get_expert_evaluation()"
        elif self.config.diagnostic_rendering and self._diagnostic_directory is None:
            message = "diagnostic_directory is required when diagnostic_rendering=True"

        if message is not None:
            return self._phase_failure(
                ExpertPhase.INITIALIZE,
                status,
                message,
                attempts=1,
                execution_duration=time.perf_counter() - start,
            )

        try:
            context = self.base_env.get_expert_task_context()
        except ExpertContextError as error:
            return self._phase_failure(
                ExpertPhase.INITIALIZE,
                status,
                str(error),
                attempts=1,
                execution_duration=time.perf_counter() - start,
            )

        if not isinstance(context, ExpertTaskContext):
            message = "get_expert_task_context() returned an incompatible context"
        elif context.environment_id != ENV_ID:
            message = f"expected environment {ENV_ID!r}, got {context.environment_id!r}"
        elif not isinstance(context.episode_spec.task_spec, TaskSpec):
            status = ExpertStatus.INVALID_TASK
            message = "active episode does not contain a project-owned TaskSpec"
        elif context.target_object.object_id not in OBJECT_IDS:
            status = ExpertStatus.INVALID_TASK
            message = f"unknown target object ID {context.target_object.object_id!r}"
        elif context.target_bin.bin_id not in BIN_IDS:
            status = ExpertStatus.INVALID_TASK
            message = f"unknown target bin ID {context.target_bin.bin_id!r}"
        elif (
            context.target_object.object_id != context.episode_spec.task_spec.target_object_id
            or context.target_bin.bin_id != context.episode_spec.task_spec.target_bin_id
        ):
            status = ExpertStatus.INVALID_TASK
            message = "semantic target handles do not match the active TaskSpec"
        elif context.agent is None or context.robot is None:
            message = "Panda agent or robot handle is unavailable"
        elif context.target_object.actor is None or context.target_bin.actor is None:
            message = "target object or bin actor handle is unavailable"
        elif not (
            context.cube_half_extent + PLACEMENT_WALL_SAFETY_MARGIN < context.bin_interior_half_size
        ):
            message = "cube plus placement wall margin does not fit inside the bin"

        if message is not None:
            return self._phase_failure(
                ExpertPhase.INITIALIZE,
                status,
                message,
                attempts=1,
                execution_duration=time.perf_counter() - start,
            )

        self._context = context

        evaluation = self._evaluation()
        if evaluation.get("target_off_table", False):
            return self._phase_failure(
                ExpertPhase.INITIALIZE,
                ExpertStatus.TARGET_OFF_TABLE,
                "target is already off the table",
                attempts=1,
                execution_duration=time.perf_counter() - start,
            )
        if evaluation.get("success", False):
            return self._phase_failure(
                ExpertPhase.INITIALIZE,
                ExpertStatus.INITIALIZATION_FAILURE,
                "episode is already in a success terminal state",
                attempts=1,
                execution_duration=time.perf_counter() - start,
            )

        planner_start = self._begin_planning_timer()
        try:
            candidate_planner = self._planner_factory(self.env)
        except (PlannerImportError, PlannerVersionError, PlannerContractError) as error:
            planning_duration = self._end_planning_timer()
            return self._phase_failure(
                ExpertPhase.INITIALIZE,
                ExpertStatus.INITIALIZATION_FAILURE,
                f"planner initialization failed: {type(error).__name__}: {error}",
                attempts=1,
                planning_duration=planning_duration,
                execution_duration=planner_start - start,
            )

        if not isinstance(candidate_planner, PlannerAdapter):
            planning_duration = self._end_planning_timer()
            return self._phase_failure(
                ExpertPhase.INITIALIZE,
                ExpertStatus.INITIALIZATION_FAILURE,
                "planner factory returned an incompatible adapter",
                attempts=1,
                planning_duration=planning_duration,
                execution_duration=planner_start - start,
            )
        self._planner = candidate_planner
        try:
            self._planner.synchronize()
        except PlannerContractError as error:
            planning_duration = self._end_planning_timer()
            return self._phase_failure(
                ExpertPhase.INITIALIZE,
                ExpertStatus.INITIALIZATION_FAILURE,
                f"planner synchronization failed: {error}",
                attempts=1,
                planning_duration=planning_duration,
                execution_duration=planner_start - start,
            )

        planning_duration = self._end_planning_timer()
        return PhaseResult(
            phase=ExpertPhase.INITIALIZE,
            success=True,
            status=ExpertStatus.SUCCESS,
            attempts=1,
            planning_duration_seconds=planning_duration,
            execution_duration_seconds=planner_start - start,
            message="validated environment, semantic task, Panda handles, and planner",
        )

    def _move_to_pregrasp(self) -> PhaseResult:
        entry_failure = self._target_entry_failure(ExpertPhase.MOVE_TO_PREGRASP)
        if entry_failure is not None:
            return entry_failure
        grasp = self._grasp_pose()
        pregrasp = grasp.copy()
        pregrasp[2] = max(
            grasp[2] + self.config.grasp_approach_distance,
            self._require_context().table_top_z + self.config.pregrasp_clearance,
        )
        return self._planned_motion(
            ExpertPhase.MOVE_TO_PREGRASP,
            pregrasp,
            use_attached=False,
            lost_grasp_status=None,
        )

    def _approach_target(self) -> PhaseResult:
        entry_failure = self._target_entry_failure(ExpertPhase.APPROACH_TARGET)
        if entry_failure is not None:
            return entry_failure
        return self._planned_motion(
            ExpertPhase.APPROACH_TARGET,
            self._grasp_pose(),
            use_attached=False,
            lost_grasp_status=None,
        )

    def _close_gripper(self) -> PhaseResult:
        return self._gripper_command_phase(
            ExpertPhase.CLOSE_GRIPPER,
            gripper_state=CLOSED_GRIPPER,
            steps=GRIPPER_COMMAND_STEPS,
            failure_status=ExpertStatus.EXECUTION_FAILURE,
        )

    def _verify_grasp(self) -> PhaseResult:
        start = time.perf_counter()
        if not self._target_is_grasped():
            return self._phase_failure(
                ExpertPhase.VERIFY_GRASP,
                ExpertStatus.GRASP_FAILURE,
                "Panda grasp oracle rejected the active semantic target",
                attempts=1,
                execution_duration=time.perf_counter() - start,
            )

        target_pose = self._actor_pose(self._require_context().target_object.actor)
        tcp_pose = self._tcp_pose()
        self._object_relative_position = _rotate_vector(
            _quaternion_conjugate(tcp_pose[3:]),
            target_pose[:3] - tcp_pose[:3],
        )
        relative_quaternion = _quaternion_multiply(
            _quaternion_conjugate(tcp_pose[3:]), target_pose[3:]
        )
        relative_quaternion /= np.linalg.norm(relative_quaternion)
        relative_pose = np.concatenate((self._object_relative_position, relative_quaternion))
        self._begin_planning_timer()
        try:
            self._require_planner().attach_box(relative_pose)
        finally:
            self._end_planning_timer()
        return PhaseResult(
            phase=ExpertPhase.VERIFY_GRASP,
            success=True,
            status=ExpertStatus.SUCCESS,
            attempts=1,
            planning_duration_seconds=time.perf_counter() - start,
            message="active semantic target is grasped and attached in the planning world",
        )

    def _lift_target(self) -> PhaseResult:
        entry_failure = self._grasp_entry_failure(
            ExpertPhase.LIFT_TARGET,
            ExpertStatus.TRANSPORT_FAILURE,
        )
        if entry_failure is not None:
            return entry_failure
        target = self._tcp_pose()
        target[2] = max(
            target[2] + self.config.lift_clearance,
            self._require_context().table_top_z + self.config.transport_clearance,
        )
        target[3:] = TOP_GRASP_QUATERNION_WXYZ
        return self._planned_motion(
            ExpertPhase.LIFT_TARGET,
            target,
            use_attached=True,
            lost_grasp_status=ExpertStatus.TRANSPORT_FAILURE,
        )

    def _move_above_destination(self) -> PhaseResult:
        entry_failure = self._grasp_entry_failure(
            ExpertPhase.MOVE_ABOVE_DESTINATION,
            ExpertStatus.TRANSPORT_FAILURE,
        )
        if entry_failure is not None:
            return entry_failure
        placement = self._placement_tcp_pose()
        bin_floor_z = _vector3(
            self._require_context().target_bin.floor_center,
            label="target bin floor center",
        )[2]
        placement[2] = max(
            placement[2] + self.config.transport_clearance,
            bin_floor_z
            + self._require_context().bin_wall_height
            + self._require_context().cube_half_extent,
        )
        return self._planned_motion(
            ExpertPhase.MOVE_ABOVE_DESTINATION,
            placement,
            use_attached=True,
            lost_grasp_status=ExpertStatus.TRANSPORT_FAILURE,
        )

    def _descend_to_place(self) -> PhaseResult:
        entry_failure = self._grasp_entry_failure(
            ExpertPhase.DESCEND_TO_PLACE,
            ExpertStatus.PLACEMENT_FAILURE,
        )
        if entry_failure is not None:
            return entry_failure
        return self._planned_motion(
            ExpertPhase.DESCEND_TO_PLACE,
            self._placement_tcp_pose(),
            use_attached=True,
            lost_grasp_status=ExpertStatus.PLACEMENT_FAILURE,
        )

    def _open_gripper(self) -> PhaseResult:
        result = self._gripper_command_phase(
            ExpertPhase.OPEN_GRIPPER,
            gripper_state=OPEN_GRIPPER,
            steps=GRIPPER_COMMAND_STEPS,
            failure_status=ExpertStatus.PLACEMENT_FAILURE,
        )
        if not result.success:
            return result
        if self._target_is_grasped():
            return self._phase_failure(
                ExpertPhase.OPEN_GRIPPER,
                ExpertStatus.PLACEMENT_FAILURE,
                "target remained grasped after the open command",
                attempts=1,
                environment_steps=result.environment_steps,
                execution_duration=result.execution_duration_seconds,
            )
        start = time.perf_counter()
        self._begin_planning_timer()
        try:
            self._require_planner().detach()
        finally:
            self._end_planning_timer()
        return PhaseResult(
            phase=ExpertPhase.OPEN_GRIPPER,
            success=True,
            status=ExpertStatus.SUCCESS,
            attempts=1,
            environment_steps=result.environment_steps,
            planning_duration_seconds=time.perf_counter() - start,
            execution_duration_seconds=result.execution_duration_seconds,
            message="target released and planner attachment removed",
        )

    def _settle_after_release(self) -> PhaseResult:
        result = self._gripper_command_phase(
            ExpertPhase.SETTLE_AFTER_RELEASE,
            gripper_state=OPEN_GRIPPER,
            steps=self.config.release_settling_steps,
            failure_status=ExpertStatus.PLACEMENT_FAILURE,
        )
        if not result.success:
            return result
        evaluation = self._evaluation()
        if evaluation.get("target_off_table", False):
            status = ExpertStatus.TARGET_OFF_TABLE
            message = "target fell off the table while settling"
        elif evaluation.get("target_is_grasped", False):
            status = ExpertStatus.PLACEMENT_FAILURE
            message = "target is still grasped after settling"
        elif not evaluation.get("target_in_target_bin", False):
            status = ExpertStatus.PLACEMENT_FAILURE
            message = "target did not settle inside the selected bin"
        elif evaluation.get("wrong_object_in_target_bin", False):
            status = ExpertStatus.PLACEMENT_FAILURE
            message = "a wrong object occupies the selected bin"
        elif not evaluation.get("target_is_static", False):
            status = ExpertStatus.PLACEMENT_FAILURE
            message = "target was not static after the configured settling steps"
        else:
            return PhaseResult(
                phase=ExpertPhase.SETTLE_AFTER_RELEASE,
                success=True,
                status=ExpertStatus.SUCCESS,
                attempts=1,
                environment_steps=result.environment_steps,
                execution_duration_seconds=result.execution_duration_seconds,
                message="released target is inside the selected bin and static",
            )
        return self._phase_failure(
            ExpertPhase.SETTLE_AFTER_RELEASE,
            status,
            message,
            attempts=1,
            environment_steps=result.environment_steps,
            execution_duration=result.execution_duration_seconds,
        )

    def _retreat(self) -> PhaseResult:
        target = self._tcp_pose()
        target[2] += self.config.retreat_distance
        target[3:] = TOP_GRASP_QUATERNION_WXYZ
        result = self._planned_motion(
            ExpertPhase.RETREAT,
            target,
            use_attached=False,
            lost_grasp_status=None,
        )
        if not result.success:
            return result
        evaluation = self._evaluation()
        if not evaluation.get("target_in_target_bin", False) or not evaluation.get(
            "target_is_static", False
        ):
            return self._phase_failure(
                ExpertPhase.RETREAT,
                ExpertStatus.PLACEMENT_FAILURE,
                "retreat disturbed the placed target",
                attempts=result.attempts,
                environment_steps=result.environment_steps,
                planning_calls=result.planning_calls,
                replans=result.replans,
                planning_duration=result.planning_duration_seconds,
                execution_duration=result.execution_duration_seconds,
                planner_status=result.planner_status,
            )
        return result

    def _verify_task(self) -> PhaseResult:
        start = time.perf_counter()
        evaluation = self._evaluation()
        if evaluation.get("success", False):
            return PhaseResult(
                phase=ExpertPhase.VERIFY_TASK,
                success=True,
                status=ExpertStatus.SUCCESS,
                attempts=1,
                execution_duration_seconds=time.perf_counter() - start,
                message="M1 conservative success evaluation passed",
            )
        if evaluation.get("target_off_table", False):
            status = ExpertStatus.TARGET_OFF_TABLE
            message = "target is off the table"
        elif evaluation.get("target_in_wrong_bin", False):
            status = ExpertStatus.PLACEMENT_FAILURE
            message = "target is in the wrong bin"
        elif evaluation.get("target_is_grasped", False):
            status = ExpertStatus.PLACEMENT_FAILURE
            message = "target is still grasped"
        elif not evaluation.get("target_in_target_bin", False):
            status = ExpertStatus.PLACEMENT_FAILURE
            message = "target is not in the selected bin"
        elif evaluation.get("wrong_object_in_target_bin", False):
            status = ExpertStatus.PLACEMENT_FAILURE
            message = "a wrong object occupies the selected bin"
        else:
            status = ExpertStatus.VERIFICATION_FAILURE
            message = "M1 conservative success evaluation rejected the final state"
        return self._phase_failure(
            ExpertPhase.VERIFY_TASK,
            status,
            message,
            attempts=1,
            execution_duration=time.perf_counter() - start,
        )

    def _planned_motion(
        self,
        phase: ExpertPhase,
        target_pose: Sequence[float],
        *,
        use_attached: bool,
        lost_grasp_status: ExpertStatus | None,
    ) -> PhaseResult:
        target = _validated_pose7(target_pose)
        planning_start = time.perf_counter()
        plan: PlannerPlanResult | None = None
        planning_calls = 0
        for _ in range(self.config.max_planning_attempts_per_phase):
            planning_calls += 1
            self._planning_calls += 1
            if planning_calls > 1:
                self._replans += 1
            self._begin_planning_timer()
            try:
                candidate = self._require_planner().plan_pose(
                    target,
                    use_attached=use_attached,
                )
            finally:
                self._end_planning_timer()
            if candidate.success:
                plan = candidate
                break
            plan = candidate

        planning_duration = time.perf_counter() - planning_start
        if plan is None or not plan.success:
            failure = plan.failure if plan is not None else PlannerFailure.PLANNING_FAILURE
            status = (
                ExpertStatus.IK_FAILURE
                if failure is PlannerFailure.IK_FAILURE
                else ExpertStatus.PLANNING_FAILURE
            )
            planner_status = plan.status if plan is not None else "no planning result"
            return self._phase_failure(
                phase,
                status,
                f"motion planning failed after {planning_calls} call(s): {planner_status}",
                attempts=planning_calls,
                planning_calls=planning_calls,
                replans=max(0, planning_calls - 1),
                planning_duration=planning_duration,
                planner_status=planner_status,
            )

        execution_start = time.perf_counter()
        steps_before = self._environment_steps
        try:
            for arm_position in plan.positions:
                self._step_action(np.asarray((*arm_position, self._gripper_state)))
                if lost_grasp_status is not None and not self._target_is_grasped():
                    raise _ExecutionAbort(
                        lost_grasp_status,
                        f"active target was lost during {phase.value}",
                    )
            final_action = np.asarray((*plan.positions[-1], self._gripper_state))
            for _ in range(FINAL_WAYPOINT_REFINEMENT_STEPS):
                self._step_action(final_action)
                if lost_grasp_status is not None and not self._target_is_grasped():
                    raise _ExecutionAbort(
                        lost_grasp_status,
                        f"active target was lost while refining {phase.value}",
                    )
        except _ExecutionAbort as error:
            return self._phase_failure(
                phase,
                error.status,
                str(error),
                attempts=planning_calls,
                environment_steps=self._environment_steps - steps_before,
                planning_calls=planning_calls,
                replans=max(0, planning_calls - 1),
                planning_duration=planning_duration,
                execution_duration=time.perf_counter() - execution_start,
                planner_status=plan.status,
            )

        actual_arm = self._arm_qpos()
        expected_arm = np.asarray(plan.positions[-1], dtype=np.float64)
        if np.max(np.abs(actual_arm - expected_arm)) > ARM_JOINT_TOLERANCE:
            return self._phase_failure(
                phase,
                ExpertStatus.EXECUTION_FAILURE,
                "Panda arm did not track the final planned joint position",
                attempts=planning_calls,
                environment_steps=self._environment_steps - steps_before,
                planning_calls=planning_calls,
                replans=max(0, planning_calls - 1),
                planning_duration=planning_duration,
                execution_duration=time.perf_counter() - execution_start,
                planner_status=plan.status,
            )
        position_error = float(np.linalg.norm(self._tcp_pose()[:3] - target[:3]))
        if position_error > TCP_POSITION_TOLERANCE:
            return self._phase_failure(
                phase,
                ExpertStatus.EXECUTION_FAILURE,
                f"TCP position error {position_error:.4f} m exceeds {TCP_POSITION_TOLERANCE:.4f} m",
                attempts=planning_calls,
                environment_steps=self._environment_steps - steps_before,
                planning_calls=planning_calls,
                replans=max(0, planning_calls - 1),
                planning_duration=planning_duration,
                execution_duration=time.perf_counter() - execution_start,
                planner_status=plan.status,
            )
        actual_quaternion = self._tcp_pose()[3:]
        quaternion_alignment = float(abs(np.dot(actual_quaternion, target[3:])))
        orientation_error = 2.0 * math.acos(min(1.0, max(0.0, quaternion_alignment)))
        if orientation_error > TCP_ORIENTATION_TOLERANCE:
            return self._phase_failure(
                phase,
                ExpertStatus.EXECUTION_FAILURE,
                f"TCP orientation error {orientation_error:.4f} rad exceeds "
                f"{TCP_ORIENTATION_TOLERANCE:.4f} rad",
                attempts=planning_calls,
                environment_steps=self._environment_steps - steps_before,
                planning_calls=planning_calls,
                replans=max(0, planning_calls - 1),
                planning_duration=planning_duration,
                execution_duration=time.perf_counter() - execution_start,
                planner_status=plan.status,
            )
        return PhaseResult(
            phase=phase,
            success=True,
            status=ExpertStatus.SUCCESS,
            attempts=planning_calls,
            environment_steps=self._environment_steps - steps_before,
            planning_calls=planning_calls,
            replans=max(0, planning_calls - 1),
            planning_duration_seconds=planning_duration,
            execution_duration_seconds=time.perf_counter() - execution_start,
            message=f"reached {phase.value} target pose",
            planner_status=plan.status,
        )

    def _gripper_command_phase(
        self,
        phase: ExpertPhase,
        *,
        gripper_state: float,
        steps: int,
        failure_status: ExpertStatus,
    ) -> PhaseResult:
        start = time.perf_counter()
        steps_before = self._environment_steps
        arm_qpos = self._arm_qpos()
        self._gripper_state = gripper_state
        try:
            for _ in range(steps):
                self._step_action(np.asarray((*arm_qpos, gripper_state)))
        except _ExecutionAbort as error:
            status = (
                error.status
                if error.status is not ExpertStatus.EXECUTION_FAILURE
                else failure_status
            )
            return self._phase_failure(
                phase,
                status,
                str(error),
                attempts=1,
                environment_steps=self._environment_steps - steps_before,
                execution_duration=time.perf_counter() - start,
            )
        return PhaseResult(
            phase=phase,
            success=True,
            status=ExpertStatus.SUCCESS,
            attempts=1,
            environment_steps=self._environment_steps - steps_before,
            execution_duration_seconds=time.perf_counter() - start,
            message=f"executed {steps} deterministic gripper control step(s)",
        )

    def _step_action(self, action: np.ndarray) -> None:
        if self._environment_steps >= self.config.max_episode_steps:
            raise _ExecutionAbort(
                ExpertStatus.TIMEOUT,
                f"expert step budget {self.config.max_episode_steps} exhausted",
            )
        if action.shape != (PANDA_ACTION_DOF,) or not np.all(np.isfinite(action)):
            raise RuntimeError("expert generated an invalid Panda pd_joint_pos action")
        step_result = self.env.step(action)
        self._environment_steps += 1
        if not isinstance(step_result, tuple) or len(step_result) != 5:
            raise RuntimeError("environment step must return the Gymnasium five-tuple")
        _, _, terminated, truncated, _ = step_result
        if _single_bool(truncated, label="truncated"):
            raise _ExecutionAbort(ExpertStatus.TIMEOUT, "environment time limit was reached")
        evaluation = self._evaluation()
        if evaluation.get("target_off_table", False):
            raise _ExecutionAbort(ExpertStatus.TARGET_OFF_TABLE, "target fell off the table")
        if _single_bool(terminated, label="terminated") and not evaluation.get("success", False):
            raise _ExecutionAbort(
                ExpertStatus.EXECUTION_FAILURE,
                "environment terminated without task success or target-off-table failure",
            )

    def _target_entry_failure(self, phase: ExpertPhase) -> PhaseResult | None:
        evaluation = self._evaluation()
        if evaluation.get("target_off_table", False):
            return self._phase_failure(
                phase,
                ExpertStatus.TARGET_OFF_TABLE,
                f"target is off the table before {phase.value}",
                attempts=1,
            )
        return None

    def _grasp_entry_failure(
        self,
        phase: ExpertPhase,
        status: ExpertStatus,
    ) -> PhaseResult | None:
        target_failure = self._target_entry_failure(phase)
        if target_failure is not None:
            return target_failure
        if not self._target_is_grasped():
            return self._phase_failure(
                phase,
                status,
                f"active target is not grasped before {phase.value}",
                attempts=1,
            )
        return None

    def _target_is_grasped(self) -> bool:
        evaluation = self._evaluation()
        return evaluation.get("target_is_grasped", False)

    def _evaluation(self) -> dict[str, bool]:
        raw = self.base_env.get_expert_evaluation()
        if not isinstance(raw, Mapping):
            raise RuntimeError("get_expert_evaluation() must return a mapping")
        return {
            str(key): _single_bool(value, label=f"evaluation[{key!r}]")
            for key, value in raw.items()
        }

    def _grasp_pose(self) -> np.ndarray:
        target_pose = self._actor_pose(self._require_context().target_object.actor)
        return np.asarray((*target_pose[:3], *TOP_GRASP_QUATERNION_WXYZ), dtype=np.float64)

    def _placement_tcp_pose(self) -> np.ndarray:
        context = self._require_context()
        floor_center = _vector3(context.target_bin.floor_center, label="target bin floor center")
        desired_object_center = floor_center.copy()
        desired_object_center[2] += context.cube_half_extent + self.config.placement_clearance
        tcp_quaternion = np.asarray(TOP_GRASP_QUATERNION_WXYZ, dtype=np.float64)
        tcp_center = desired_object_center - _rotate_vector(
            tcp_quaternion, self._object_relative_position
        )
        return np.concatenate((tcp_center, tcp_quaternion))

    def _arm_qpos(self) -> np.ndarray:
        qpos = _numeric_array(
            self._require_context().robot.get_qpos(),
            label="Panda qpos",
        )
        if qpos.shape != (1, 9):
            raise RuntimeError(f"Panda qpos must have shape (1, 9), got {qpos.shape}")
        return qpos[0, :PANDA_ARM_DOF].copy()

    def _tcp_pose(self) -> np.ndarray:
        return self._actor_pose(self._require_context().agent.tcp)

    @staticmethod
    def _actor_pose(actor: object) -> np.ndarray:
        pose = getattr(actor, "pose", None)
        raw_pose = getattr(pose, "raw_pose", None)
        array = _numeric_array(raw_pose, label="actor pose")
        if array.shape != (1, 7):
            raise RuntimeError(f"actor pose must have shape (1, 7), got {array.shape}")
        return _validated_pose7(array[0]).copy()

    def _record_phase(self, result: PhaseResult) -> None:
        self._phase_results.append(result)
        if result.success and self.config.diagnostic_rendering:
            self._capture_phase_frame(result.phase)

    def _capture_phase_frame(self, phase: ExpertPhase) -> None:
        if self._diagnostic_directory is None:
            raise RuntimeError("diagnostic_directory is required when diagnostic_rendering=True")
        from PIL import Image

        frame = _numeric_array(self.env.render(), label="diagnostic render")
        if frame.ndim == 4 and frame.shape[0] == 1:
            frame = frame[0]
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise RuntimeError(f"diagnostic render must be HxWx3/4, got {frame.shape}")
        if np.issubdtype(frame.dtype, np.floating):
            frame = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8, copy=False)
        self._diagnostic_directory.mkdir(parents=True, exist_ok=True)
        output_path = (
            self._diagnostic_directory / f"{len(self._phase_results):02d}_{phase.value}.png"
        )
        Image.fromarray(frame).save(output_path)
        self._artifact_paths.append(str(output_path))

    def _build_result(self, status: ExpertStatus) -> ExpertResult:
        context = self._context
        episode = context.episode_spec if context is not None else None
        completed = tuple(result.phase for result in self._phase_results if result.success)
        failed = next((result.phase for result in self._phase_results if not result.success), None)
        evaluation = self._evaluation() if context is not None else {}
        return ExpertResult(
            success=status is ExpertStatus.SUCCESS,
            status=status,
            scene_seed=episode.scene_seed if episode is not None else None,
            scene_id=episode.scene_id if episode is not None else None,
            task_id=episode.task_id if episode is not None else None,
            canonical_instruction=episode.canonical_instruction if episode is not None else None,
            target_object_id=(episode.task_spec.target_object_id if episode is not None else None),
            target_bin_id=episode.task_spec.target_bin_id if episode is not None else None,
            total_environment_steps=self._environment_steps,
            total_planning_calls=self._planning_calls,
            total_replans=self._replans,
            completed_phases=completed,
            failed_phase=failed,
            final_environment_evaluation=evaluation,
            phase_results=tuple(self._phase_results),
            planning_duration_seconds=sum(
                result.planning_duration_seconds for result in self._phase_results
            ),
            execution_duration_seconds=sum(
                result.execution_duration_seconds for result in self._phase_results
            ),
            diagnostic_artifact_paths=tuple(self._artifact_paths),
        )

    @staticmethod
    def _phase_failure(
        phase: ExpertPhase,
        status: ExpertStatus,
        message: str,
        *,
        attempts: int = 0,
        environment_steps: int = 0,
        planning_calls: int = 0,
        replans: int = 0,
        planning_duration: float = 0.0,
        execution_duration: float = 0.0,
        planner_status: str | None = None,
    ) -> PhaseResult:
        return PhaseResult(
            phase=phase,
            success=False,
            status=status,
            attempts=attempts,
            environment_steps=environment_steps,
            planning_calls=planning_calls,
            replans=replans,
            planning_duration_seconds=planning_duration,
            execution_duration_seconds=execution_duration,
            message=message,
            planner_status=planner_status,
        )

    def _require_context(self) -> ExpertTaskContext:
        if self._context is None:
            raise RuntimeError("expert context is unavailable before initialization")
        return self._context

    def _require_planner(self) -> PlannerAdapter:
        if self._planner is None:
            raise RuntimeError("planner is unavailable before initialization")
        return self._planner

    def _begin_planning_timer(self) -> float:
        if self._active_planning_started_at is not None:
            raise RuntimeError("nested planning timing scopes are not supported")
        started_at = time.perf_counter()
        self._active_planning_started_at = started_at
        return started_at

    def _end_planning_timer(self) -> float:
        if self._active_planning_started_at is None:
            raise RuntimeError("planning timing scope was not started")
        duration = time.perf_counter() - self._active_planning_started_at
        self._planning_duration += duration
        self._active_planning_started_at = None
        return duration


def _numeric_array(value: object, *, label: str) -> np.ndarray:
    if value is None:
        raise RuntimeError(f"{label} is unavailable")
    converted = value
    detach = getattr(converted, "detach", None)
    if callable(detach):
        converted = detach()
    cpu = getattr(converted, "cpu", None)
    if callable(cpu):
        converted = cpu()
    numpy_method = getattr(converted, "numpy", None)
    if callable(numpy_method):
        converted = numpy_method()
    try:
        result = np.asarray(converted)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} must be numeric") from error
    if not np.all(np.isfinite(result)):
        raise RuntimeError(f"{label} must contain finite values")
    return result


def _single_bool(value: object, *, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    array = _numeric_array(value, label=label)
    if array.size != 1:
        raise RuntimeError(f"{label} must contain exactly one value, got shape {array.shape}")
    scalar = array.reshape(-1)[0]
    if not isinstance(scalar, (bool, np.bool_)):
        raise RuntimeError(f"{label} must be boolean")
    return bool(scalar)


def _vector3(value: object, *, label: str) -> np.ndarray:
    result = _numeric_array(value, label=label).astype(np.float64, copy=True)
    if result.shape != (3,):
        raise RuntimeError(f"{label} must have shape (3,), got {result.shape}")
    return result


def _validated_pose7(value: object) -> np.ndarray:
    result = _numeric_array(value, label="pose").astype(np.float64, copy=True)
    if result.shape != (7,):
        raise RuntimeError(f"pose must have shape (7,), got {result.shape}")
    norm = float(np.linalg.norm(result[3:]))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise RuntimeError("pose quaternion must be unit length in wxyz order")
    return result


def _quaternion_conjugate(quaternion: Sequence[float]) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    return np.asarray((q[0], -q[1], -q[2], -q[3]), dtype=np.float64)


def _quaternion_multiply(left: Sequence[float], right: Sequence[float]) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(left, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(right, dtype=np.float64)
    return np.asarray(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dtype=np.float64,
    )


def _rotate_vector(quaternion: Sequence[float], vector: Sequence[float]) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    q = q / np.linalg.norm(q)
    v = np.asarray(vector, dtype=np.float64)
    q_vector = q[1:]
    return v + 2.0 * np.cross(q_vector, np.cross(q_vector, v) + q[0] * v)


__all__ = [
    "PHASE_CONTRACTS",
    "PickPlaceExpert",
    "TOP_GRASP_APPROACH_DIRECTION",
    "TOP_GRASP_CLOSING_DIRECTION",
    "TOP_GRASP_QUATERNION_WXYZ",
]
