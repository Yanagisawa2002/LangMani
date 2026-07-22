"""Geometry-constrained, closed-loop expert for LangMani planar pushing."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from langmani.environments.push_to_region import (
    CUBE_HALF_SIZE,
    CYLINDER_HALF_LENGTH,
    CYLINDER_RADIUS,
    WORKSPACE_BOUNDS_XY,
)
from langmani.experts.geometry_push_types import (
    GeometryPushExpertConfig,
    GeometryPushExpertResult,
    GeometryPushState,
    GeometryPushTransition,
)
from langmani.experts.push import PushToRegionExpert, _PushAbort
from langmani.experts.push_feasibility import (
    PlanarPrimitive,
    PrimitiveKind,
    PushGeometryPlan,
    WorkspaceContract,
    build_geometry_push_plan,
)
from langmani.experts.push_types import (
    PushExpertConfig,
    PushExpertPhase,
    PushExpertStatus,
    PushPhaseResult,
)

_STATE_VISIT_LIMITS: Mapping[GeometryPushState, int] = {
    GeometryPushState.PLAN_PUSH: 1,
    GeometryPushState.MOVE_TO_STAGING: 4,
    GeometryPushState.MOVE_TO_PRECONTACT: 4,
    GeometryPushState.ESTABLISH_CONTACT: 4,
    GeometryPushState.PUSH_CLOSED_LOOP: 12,
    GeometryPushState.VERIFY_PROGRESS: 12,
    GeometryPushState.RECOVER_CONTACT: 2,
    GeometryPushState.REPLAN: 3,
    GeometryPushState.SUCCESS: 1,
    GeometryPushState.SAFE_ABORT: 1,
}


class GeometryConstrainedPushExpert(PushToRegionExpert):
    """Execute one support-aware plan through an explicit closed-loop state machine."""

    def __init__(
        self,
        environment: object,
        *,
        config: GeometryPushExpertConfig | None = None,
        planner_factory: Any | None = None,
    ) -> None:
        geometry_config = config or GeometryPushExpertConfig()
        base_config = PushExpertConfig(
            maximum_episode_steps=geometry_config.maximum_episode_steps,
            gripper_close_steps=geometry_config.gripper_close_steps,
            precontact_clearance=geometry_config.precontact_clearance,
            contact_offset=CUBE_HALF_SIZE + geometry_config.gripper_contact_padding,
            precontact_height=geometry_config.precontact_height,
            precontact_staging_height=geometry_config.staging_height,
            push_height=geometry_config.cube_push_height,
            cylinder_push_height=geometry_config.cylinder_push_height,
            primary_push_increment=geometry_config.maximum_segment_distance,
            minimum_primary_push_increment=geometry_config.minimum_segment_distance,
            maximum_primary_push_segments=geometry_config.maximum_push_segments,
            free_space_action_stride=geometry_config.free_space_action_stride,
            settle_steps=geometry_config.settle_steps,
        )
        super().__init__(environment, config=base_config, planner_factory=planner_factory)
        self.geometry_config = geometry_config
        self._state = GeometryPushState.PLAN_PUSH
        self._state_visits: Counter[GeometryPushState] = Counter()
        self._transitions: list[GeometryPushTransition] = []
        self._geometry_plans: list[dict[str, object]] = []
        self._geometry_plan: PushGeometryPlan | None = None
        self._contact_recoveries = 0
        self._replans = 0
        self._push_segments = 0
        self._stagnant_verifications = 0
        self._last_verified_position: np.ndarray | None = None
        self._last_verified_distance: float | None = None
        self._terminal_status = PushExpertStatus.PUSH_FAILURE

    def run(self) -> GeometryPushExpertResult:
        """Run the bounded state machine and close the native planner on exit."""

        try:
            for _ in range(self.geometry_config.maximum_state_visits):
                self._state_visits[self._state] += 1
                if self._state_visits[self._state] > self._state_visit_limit(self._state):
                    self._terminal_status = PushExpertStatus.TIMEOUT
                    self._transition(GeometryPushState.SAFE_ABORT, "state visit limit exhausted")
                if self._state is GeometryPushState.SUCCESS:
                    return self._result(PushExpertStatus.SUCCESS)
                if self._state is GeometryPushState.SAFE_ABORT:
                    return self._result(self._terminal_status)
                self._dispatch_state()
            self._terminal_status = PushExpertStatus.TIMEOUT
            if self._state not in {GeometryPushState.SUCCESS, GeometryPushState.SAFE_ABORT}:
                self._transition(GeometryPushState.SAFE_ABORT, "global state budget exhausted")
            return self._result(self._terminal_status)
        finally:
            if self._planner is not None:
                self._planner.close()

    def unexpected_exception_result(self, error: BaseException) -> GeometryPushExpertResult:
        """Preserve unexpected simulator or integration failures as evidence."""

        return self._result(
            PushExpertStatus.UNEXPECTED_EXCEPTION,
            exception_type=type(error).__name__,
            exception_message=str(error) or repr(error),
        )

    def _dispatch_state(self) -> None:
        operations = {
            GeometryPushState.PLAN_PUSH: self._state_plan_push,
            GeometryPushState.MOVE_TO_STAGING: self._state_move_to_staging,
            GeometryPushState.MOVE_TO_PRECONTACT: self._state_move_to_precontact,
            GeometryPushState.ESTABLISH_CONTACT: self._state_establish_contact,
            GeometryPushState.PUSH_CLOSED_LOOP: self._state_push_closed_loop,
            GeometryPushState.VERIFY_PROGRESS: self._state_verify_progress,
            GeometryPushState.RECOVER_CONTACT: self._state_recover_contact,
            GeometryPushState.REPLAN: self._state_replan,
        }
        operations[self._state]()

    def _state_plan_push(self) -> None:
        initialized = self._initialize()
        if not initialized.success:
            self._abort(initialized.status, initialized.message)
            return
        feedback = getattr(self.base, "get_push_expert_feedback", None)
        if not callable(feedback):
            self._abort(PushExpertStatus.INITIALIZATION_FAILURE, "missing closed-loop feedback")
            return
        closed = self._close_gripper()
        if not closed.success:
            self._abort(closed.status, closed.message)
            return
        try:
            self._refresh_plan()
        except (RuntimeError, ValueError) as error:
            self._abort(PushExpertStatus.CONTACT_FAILURE, f"geometry plan rejected: {error}")
            return
        self._transition(GeometryPushState.MOVE_TO_STAGING, "complete geometry plan accepted")

    def _state_move_to_staging(self) -> None:
        plan = self._require_geometry_plan()
        target = self._pose_at(plan.staging_point)
        result = self._planned_with_midpoint(
            PushExpertPhase.MOVE_TO_PRECONTACT,
            target,
            action_stride=self.geometry_config.free_space_action_stride,
        )
        if result.success:
            self._transition(GeometryPushState.MOVE_TO_PRECONTACT, "staging pose reached")
        else:
            self._route_motion_failure(result, "staging motion")

    def _state_move_to_precontact(self) -> None:
        target = self._pose_at(self._require_geometry_plan().precontact_point)
        result = self._planned_with_midpoint(
            PushExpertPhase.MOVE_TO_PRECONTACT,
            target,
            action_stride=self.geometry_config.free_space_action_stride,
        )
        if result.success:
            self._transition(GeometryPushState.ESTABLISH_CONTACT, "precontact pose reached")
        else:
            self._route_motion_failure(result, "precontact motion")

    def _state_establish_contact(self) -> None:
        target = self._pose_at(self._require_geometry_plan().contact_point)
        result = self._planned_with_midpoint(PushExpertPhase.ESTABLISH_CONTACT, target)
        if result.success:
            self._last_verified_position = self._target_position()[:2]
            self._last_verified_distance = self._target_distance()
            self._transition(GeometryPushState.PUSH_CLOSED_LOOP, "contact pose reached")
        else:
            self._route_motion_failure(result, "contact motion")

    def _state_push_closed_loop(self) -> None:
        if self._push_segments >= self.geometry_config.maximum_push_segments:
            self._abort(PushExpertStatus.TIMEOUT, "closed-loop push segment limit exhausted")
            return
        try:
            self._refresh_plan()
        except (RuntimeError, ValueError) as error:
            self._abort(
                PushExpertStatus.CONTACT_FAILURE, f"updated geometry plan rejected: {error}"
            )
            return
        plan = self._require_geometry_plan()
        direction = np.asarray(plan.direction_xy, dtype=np.float64)
        endpoint = np.asarray(plan.segment_endpoints_xy[0], dtype=np.float64)
        tcp_xy = endpoint - direction * (
            plan.support_extent + self.geometry_config.gripper_contact_padding
        )
        target = self._tcp_pose()
        target[:2] = tcp_xy
        target[2] = self._push_height_for_geometry()
        result = self._planned_motion(
            PushExpertPhase.PRIMARY_PUSH,
            (target,),
            action_stride=self._closed_loop_action_stride(),
            stop_on_target_inside_region=True,
            brake_before_containment=False,
        )
        self._push_segments += 1
        if result.success:
            self._transition(GeometryPushState.VERIFY_PROGRESS, "short push segment executed")
        else:
            self._route_motion_failure(result, "closed-loop push segment")

    def _state_verify_progress(self) -> None:
        evaluation = self._evaluation()
        if evaluation.get("success") is True or self._terminal_success:
            self._transition(GeometryPushState.SUCCESS, "native stable success reached")
            return
        if evaluation.get("target_inside_region") is True:
            try:
                self._hold_while_contained(self.geometry_config.settle_steps)
            except _PushAbort as error:
                self._abort(error.status, str(error))
                return
            if self._terminal_success or self._evaluation().get("success") is True:
                self._transition(GeometryPushState.SUCCESS, "contained target stabilized")
                return

        position = self._target_position()[:2]
        distance = self._target_distance()
        previous_position = self._last_verified_position
        previous_distance = self._last_verified_distance
        progress = 0.0 if previous_distance is None else previous_distance - distance
        lateral_error = self._lateral_error(previous_position, position)
        self._last_verified_position = position.copy()
        self._last_verified_distance = distance
        contact_force, angular_speed = self._feedback_scalars()
        contact_distance = float(np.linalg.norm(self._tcp_pose()[:2] - position))
        expected_contact = (
            self._require_geometry_plan().support_extent
            + self.geometry_config.gripper_contact_padding
        )
        contact_lost = bool(
            contact_force < self.geometry_config.contact_force_threshold
            and contact_distance
            > expected_contact + self.geometry_config.contact_distance_tolerance
        )
        if contact_lost or angular_speed > self.geometry_config.maximum_cylinder_angular_speed:
            self._transition(
                GeometryPushState.RECOVER_CONTACT,
                "contact lost" if contact_lost else "cylinder rollout threshold exceeded",
            )
            return
        if lateral_error > self.geometry_config.maximum_lateral_error:
            self._transition(GeometryPushState.REPLAN, "lateral drift exceeded bound")
            return
        if progress < self.geometry_config.minimum_progress:
            self._stagnant_verifications += 1
            if self._stagnant_verifications >= self.geometry_config.maximum_stagnant_verifications:
                self._transition(GeometryPushState.RECOVER_CONTACT, "progress stagnated")
                return
        else:
            self._stagnant_verifications = 0
        self._transition(GeometryPushState.PUSH_CLOSED_LOOP, "progress verified")

    def _state_recover_contact(self) -> None:
        if self._contact_recoveries >= self.geometry_config.maximum_contact_recoveries:
            self._abort(PushExpertStatus.CONTACT_FAILURE, "contact recovery limit exhausted")
            return
        self._contact_recoveries += 1
        plan = self._require_geometry_plan()
        direction = np.asarray(plan.direction_xy, dtype=np.float64)
        retreat = self._tcp_pose()
        retreat[:2] -= direction * self.geometry_config.retreat_distance
        retreat[2] = self.geometry_config.precontact_height
        result = self._planned_with_midpoint(
            PushExpertPhase.CORRECTIVE_PUSH_1,
            retreat,
            action_stride=self.geometry_config.free_space_action_stride,
        )
        if not result.success:
            self._route_motion_failure(result, "contact retreat")
            return
        try:
            self._refresh_plan()
        except (RuntimeError, ValueError) as error:
            self._abort(PushExpertStatus.CONTACT_FAILURE, f"recovery plan rejected: {error}")
            return
        self._transition(GeometryPushState.MOVE_TO_STAGING, "bounded contact recovery replanned")

    def _state_replan(self) -> None:
        if self._replans >= self.geometry_config.maximum_replans:
            self._abort(PushExpertStatus.PLANNING_FAILURE, "replan limit exhausted")
            return
        self._replans += 1
        try:
            self._refresh_plan()
        except (RuntimeError, ValueError) as error:
            self._abort(PushExpertStatus.PLANNING_FAILURE, f"replan rejected: {error}")
            return
        self._transition(GeometryPushState.MOVE_TO_STAGING, "bounded geometry replan accepted")

    def _refresh_plan(self) -> None:
        context = self._require_context()
        object_position = self._target_position()
        target_center = self._target_center()
        primitive = self._primitive()
        distance = float(np.linalg.norm(target_center[:2] - object_position[:2]))
        maximum_segment = self.geometry_config.maximum_segment_distance
        if distance <= 2.0 * context.full_containment_center_radius:
            maximum_segment *= self.geometry_config.near_goal_step_scale
        plan = build_geometry_push_plan(
            object_xy=object_position[:2],
            target_xy=target_center[:2],
            primitive=primitive,
            object_yaw=self._object_yaw(),
            workspace=WorkspaceContract(
                bounds_xy=WORKSPACE_BOUNDS_XY,
                object_clearance=self.geometry_config.object_clearance,
                tcp_clearance=self.geometry_config.tcp_clearance,
                tracking_margin=self.geometry_config.tracking_margin,
            ),
            full_containment_radius=context.full_containment_center_radius,
            goal_margin=self.geometry_config.goal_margin,
            gripper_contact_padding=self.geometry_config.gripper_contact_padding,
            precontact_clearance=self.geometry_config.precontact_clearance,
            push_height=self._push_height_for_geometry(),
            precontact_height=self.geometry_config.precontact_height,
            staging_height=self.geometry_config.staging_height,
            maximum_segment_distance=max(
                self.geometry_config.minimum_segment_distance,
                maximum_segment,
            ),
        )
        self._geometry_plan = plan
        self._geometry_plans.append(plan.to_dict())

    def _planned_with_midpoint(
        self,
        phase: PushExpertPhase,
        target: np.ndarray,
        *,
        action_stride: int = 1,
    ) -> PushPhaseResult:
        direct = self._planned_motion(phase, (target,), action_stride=action_stride)
        if direct.success or direct.environment_steps > 0:
            return direct
        if direct.status not in {PushExpertStatus.IK_FAILURE, PushExpertStatus.PLANNING_FAILURE}:
            return direct
        midpoint = self._tcp_pose()
        midpoint[:3] = (midpoint[:3] + target[:3]) * 0.5
        midpoint[2] = max(midpoint[2], target[2], self.geometry_config.precontact_height)
        fallback = self._planned_motion(
            phase,
            (midpoint, target),
            action_stride=action_stride,
        )
        return replace(
            fallback,
            attempts=direct.attempts + fallback.attempts,
            planning_calls=direct.planning_calls + fallback.planning_calls,
            planning_duration_seconds=(
                direct.planning_duration_seconds + fallback.planning_duration_seconds
            ),
            execution_duration_seconds=(
                direct.execution_duration_seconds + fallback.execution_duration_seconds
            ),
            message="direct plan rejected; bounded midpoint fallback: " + fallback.message,
        )

    def _route_motion_failure(self, result: PushPhaseResult, label: str) -> None:
        if result.status in {PushExpertStatus.IK_FAILURE, PushExpertStatus.PLANNING_FAILURE}:
            self._transition(GeometryPushState.REPLAN, f"{label} rejected by planner")
            return
        if result.status is PushExpertStatus.CONTACT_FAILURE:
            self._transition(GeometryPushState.RECOVER_CONTACT, f"{label} lost contact")
            return
        self._abort(result.status, f"{label} failed: {result.message}")

    def _transition(self, target: GeometryPushState, reason: str) -> None:
        self._transitions.append(
            GeometryPushTransition(
                source=self._state,
                target=target,
                reason=reason,
                environment_steps=self._environment_steps,
                target_distance=(self._target_distance() if self._context is not None else None),
                contact_recoveries=self._contact_recoveries,
                replans=self._replans,
            )
        )
        self._state = target

    def _abort(self, status: PushExpertStatus, reason: str) -> None:
        self._terminal_status = status
        self._transition(GeometryPushState.SAFE_ABORT, reason)

    def _target_distance(self) -> float:
        value = self._evaluation().get("target_distance")
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise RuntimeError("target_distance feedback must be finite")
        return float(value)

    def _feedback_scalars(self) -> tuple[float, float]:
        raw = self.base.get_push_expert_feedback()
        if not isinstance(raw, Mapping):
            raise RuntimeError("push expert feedback must be a mapping")
        force = self._finite_scalar(raw.get("target_contact_force"), "target_contact_force")
        angular = self._finite_vector(raw.get("target_angular_velocity"), "target_angular_velocity")
        return force, float(np.linalg.norm(angular))

    def _closed_loop_action_stride(self) -> int:
        """Use sparse far-field execution and restore dense control near containment."""

        distance = self._target_distance()
        near_goal = 2.0 * self._require_context().full_containment_center_radius
        return 1 if distance <= near_goal else self.geometry_config.free_space_action_stride

    def _state_visit_limit(self, state: GeometryPushState) -> int:
        if state in {GeometryPushState.PUSH_CLOSED_LOOP, GeometryPushState.VERIFY_PROGRESS}:
            return self.geometry_config.maximum_push_segments
        return _STATE_VISIT_LIMITS[state]

    def _lateral_error(
        self, previous_position: np.ndarray | None, current_position: np.ndarray
    ) -> float:
        if previous_position is None:
            return 0.0
        direction = np.asarray(self._require_geometry_plan().direction_xy)
        displacement = current_position - previous_position
        return float(abs(direction[0] * displacement[1] - direction[1] * displacement[0]))

    def _primitive(self) -> PlanarPrimitive:
        object_id = self._require_context().target_object.object_id
        if object_id == "blue_cube":
            return PlanarPrimitive(
                kind=PrimitiveKind.BOX,
                half_extents_xy=(CUBE_HALF_SIZE, CUBE_HALF_SIZE),
                radius=math.sqrt(2.0) * CUBE_HALF_SIZE,
                height=2.0 * CUBE_HALF_SIZE,
            )
        if object_id == "orange_cylinder":
            return PlanarPrimitive(
                kind=PrimitiveKind.CYLINDER,
                half_extents_xy=(CYLINDER_RADIUS, CYLINDER_RADIUS),
                radius=CYLINDER_RADIUS,
                height=2.0 * CYLINDER_HALF_LENGTH,
            )
        raise RuntimeError(f"unsupported Push primitive {object_id!r}")

    def _object_yaw(self) -> float:
        quaternion = self._target_position_pose()[3:]
        w, x, y, z = quaternion
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _push_height_for_geometry(self) -> float:
        return (
            self.geometry_config.cube_push_height
            if self._require_context().target_object.object_id == "blue_cube"
            else self.geometry_config.cylinder_push_height
        )

    def _pose_at(self, point: Sequence[float]) -> np.ndarray:
        pose = self._tcp_pose()
        pose[:3] = np.asarray(point, dtype=np.float64)
        return pose

    def _require_geometry_plan(self) -> PushGeometryPlan:
        if self._geometry_plan is None:
            raise RuntimeError("geometry push plan is unavailable")
        return self._geometry_plan

    @staticmethod
    def _finite_scalar(value: object, label: str) -> float:
        array = GeometryConstrainedPushExpert._finite_vector(value, label)
        if array.size != 1:
            raise RuntimeError(f"{label} must contain one value")
        return float(array.reshape(-1)[0])

    @staticmethod
    def _finite_vector(value: object, label: str) -> np.ndarray:
        converted = value
        for method_name in ("detach", "cpu"):
            method = getattr(converted, method_name, None)
            if callable(method):
                converted = method()
        numpy_method = getattr(converted, "numpy", None)
        if callable(numpy_method):
            converted = numpy_method()
        result = np.asarray(converted, dtype=np.float64)
        if not np.isfinite(result).all():
            raise RuntimeError(f"{label} must be finite")
        return result

    def _result(
        self,
        status: PushExpertStatus,
        *,
        exception_type: str | None = None,
        exception_message: str | None = None,
    ) -> GeometryPushExpertResult:
        context = self._context
        episode = context.episode_spec if context is not None else None
        return GeometryPushExpertResult(
            success=status is PushExpertStatus.SUCCESS,
            status=status,
            scene_seed=episode.scene_seed if episode is not None else None,
            task=episode.task_spec.to_dict() if episode is not None else {},
            total_environment_steps=self._environment_steps,
            total_planning_calls=self._planning_calls,
            contact_recoveries=self._contact_recoveries,
            replans=self._replans,
            push_segments=self._push_segments,
            transitions=tuple(self._transitions),
            final_environment_evaluation=self._evaluation() if context is not None else {},
            geometry_plans=tuple(self._geometry_plans),
            exception_type=exception_type,
            exception_message=exception_message,
        )


__all__ = ["GeometryConstrainedPushExpert"]
