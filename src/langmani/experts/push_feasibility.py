"""Pure geometry and workspace contracts for closed-loop planar pushing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class PrimitiveKind(StrEnum):
    """Supported planar support primitives."""

    BOX = "box"
    CYLINDER = "cylinder"


@dataclass(frozen=True, slots=True)
class PlanarPrimitive:
    """Planar support geometry with a physical height."""

    kind: PrimitiveKind
    half_extents_xy: tuple[float, float]
    radius: float
    height: float

    def __post_init__(self) -> None:
        values = (*self.half_extents_xy, self.radius, self.height)
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("primitive dimensions must be finite and positive")
        if self.kind is PrimitiveKind.BOX and self.radius < math.hypot(*self.half_extents_xy):
            raise ValueError("box radius must enclose its planar half extents")


@dataclass(frozen=True, slots=True)
class WorkspaceContract:
    """Native planar workspace plus fixed geometric safety clearances."""

    bounds_xy: tuple[float, float, float, float]
    object_clearance: float
    tcp_clearance: float
    tracking_margin: float

    def __post_init__(self) -> None:
        x_min, x_max, y_min, y_max = self.bounds_xy
        if not all(math.isfinite(value) for value in self.bounds_xy):
            raise ValueError("workspace bounds must be finite")
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("workspace bounds must have positive area")
        for value in (self.object_clearance, self.tcp_clearance, self.tracking_margin):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("workspace clearances must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class PushGeometryPlan:
    """One finite, support-aware straight push plan."""

    direction_xy: tuple[float, float]
    support_extent: float
    contact_point: tuple[float, float, float]
    precontact_point: tuple[float, float, float]
    staging_point: tuple[float, float, float]
    object_goal_xy: tuple[float, float]
    object_path_margin: float
    tcp_path_margin: float
    segment_endpoints_xy: tuple[tuple[float, float], ...]

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible representation."""

        return {
            "direction_xy": list(self.direction_xy),
            "support_extent": self.support_extent,
            "contact_point": list(self.contact_point),
            "precontact_point": list(self.precontact_point),
            "staging_point": list(self.staging_point),
            "object_goal_xy": list(self.object_goal_xy),
            "object_path_margin": self.object_path_margin,
            "tcp_path_margin": self.tcp_path_margin,
            "segment_endpoints_xy": [list(value) for value in self.segment_endpoints_xy],
        }


def normalized_planar_direction(start_xy: np.ndarray, goal_xy: np.ndarray) -> np.ndarray:
    """Return the finite unit direction from ``start_xy`` to ``goal_xy``."""

    start = _vector(start_xy, 2, "start_xy")
    goal = _vector(goal_xy, 2, "goal_xy")
    delta = goal - start
    norm = float(np.linalg.norm(delta))
    if norm < 1e-8:
        raise ValueError("object and target centers must be distinct")
    return delta / norm


def box_support_extent(
    half_extents_xy: tuple[float, float], direction_xy: np.ndarray, yaw: float
) -> float:
    """Return the support extent of a yawed box along a planar direction."""

    direction = normalized_planar_direction(np.zeros(2), direction_xy)
    if not math.isfinite(yaw):
        raise ValueError("box yaw must be finite")
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    local_x = cosine * direction[0] + sine * direction[1]
    local_y = -sine * direction[0] + cosine * direction[1]
    extent = abs(local_x) * half_extents_xy[0] + abs(local_y) * half_extents_xy[1]
    if not math.isfinite(extent) or extent <= 0.0:
        raise ValueError("box support extent must be finite and positive")
    return float(extent)


def support_extent(
    primitive: PlanarPrimitive, direction_xy: np.ndarray, *, yaw: float = 0.0
) -> float:
    """Return the primitive support extent in ``direction_xy``."""

    if primitive.kind is PrimitiveKind.CYLINDER:
        normalized_planar_direction(np.zeros(2), direction_xy)
        return primitive.radius
    return box_support_extent(primitive.half_extents_xy, direction_xy, yaw)


def oriented_axis_extents(primitive: PlanarPrimitive, *, yaw: float = 0.0) -> tuple[float, float]:
    """Return support extents along the world x and y axes."""

    return (
        support_extent(primitive, np.array([1.0, 0.0]), yaw=yaw),
        support_extent(primitive, np.array([0.0, 1.0]), yaw=yaw),
    )


def footprint_workspace_margin(
    center_xy: np.ndarray,
    primitive: PlanarPrimitive,
    workspace: WorkspaceContract,
    *,
    yaw: float = 0.0,
) -> float:
    """Return the signed margin of the complete primitive footprint."""

    center = _vector(center_xy, 2, "center_xy")
    extent_x, extent_y = oriented_axis_extents(primitive, yaw=yaw)
    clearance = workspace.object_clearance + workspace.tracking_margin
    x_min, x_max, y_min, y_max = workspace.bounds_xy
    return float(
        min(
            center[0] - extent_x - clearance - x_min,
            x_max - center[0] - extent_x - clearance,
            center[1] - extent_y - clearance - y_min,
            y_max - center[1] - extent_y - clearance,
        )
    )


def point_workspace_margin(point_xy: np.ndarray, workspace: WorkspaceContract) -> float:
    """Return the signed workspace margin of a TCP-center point."""

    point = _vector(point_xy, 2, "point_xy")
    clearance = workspace.tcp_clearance + workspace.tracking_margin
    x_min, x_max, y_min, y_max = workspace.bounds_xy
    return float(
        min(
            point[0] - clearance - x_min,
            x_max - point[0] - clearance,
            point[1] - clearance - y_min,
            y_max - point[1] - clearance,
        )
    )


def build_geometry_push_plan(
    *,
    object_xy: np.ndarray,
    target_xy: np.ndarray,
    primitive: PlanarPrimitive,
    object_yaw: float,
    workspace: WorkspaceContract,
    full_containment_radius: float,
    goal_margin: float,
    gripper_contact_padding: float,
    precontact_clearance: float,
    push_height: float,
    precontact_height: float,
    staging_height: float,
    maximum_segment_distance: float,
) -> PushGeometryPlan:
    """Construct and reject one complete straight push before execution."""

    object_center = _vector(object_xy, 2, "object_xy")
    target_center = _vector(target_xy, 2, "target_xy")
    scalars = {
        "full_containment_radius": full_containment_radius,
        "goal_margin": goal_margin,
        "gripper_contact_padding": gripper_contact_padding,
        "precontact_clearance": precontact_clearance,
        "push_height": push_height,
        "precontact_height": precontact_height,
        "staging_height": staging_height,
        "maximum_segment_distance": maximum_segment_distance,
    }
    if not all(math.isfinite(value) and value > 0.0 for value in scalars.values()):
        raise ValueError("push-plan distances and heights must be finite and positive")
    if goal_margin >= full_containment_radius:
        raise ValueError("goal_margin must be smaller than full_containment_radius")
    if staging_height < precontact_height or precontact_height < push_height:
        raise ValueError("push heights must satisfy staging >= precontact >= push")

    direction = normalized_planar_direction(object_center, target_center)
    extent = support_extent(primitive, direction, yaw=object_yaw)
    contact_offset = extent + gripper_contact_padding
    contact_xy = object_center - direction * contact_offset
    precontact_xy = contact_xy - direction * precontact_clearance
    goal_offset = full_containment_radius - goal_margin
    object_goal = target_center - direction * goal_offset
    travel = float(np.dot(object_goal - object_center, direction))
    if travel <= 0.0:
        raise ValueError("object already lies beyond the planned goal boundary")

    object_margin = min(
        footprint_workspace_margin(object_center, primitive, workspace, yaw=object_yaw),
        footprint_workspace_margin(object_goal, primitive, workspace, yaw=object_yaw),
    )
    final_tcp = object_goal - direction * contact_offset
    tcp_margin = min(
        point_workspace_margin(precontact_xy, workspace),
        point_workspace_margin(contact_xy, workspace),
        point_workspace_margin(final_tcp, workspace),
    )
    if object_margin < 0.0:
        raise ValueError(f"predicted object path leaves safe workspace by {-object_margin:.6f} m")
    if tcp_margin < 0.0:
        raise ValueError(f"predicted TCP path leaves safe workspace by {-tcp_margin:.6f} m")

    segment_count = max(1, math.ceil(travel / maximum_segment_distance))
    endpoint_values = tuple(
        object_center + direction * travel * index / segment_count
        for index in range(1, segment_count + 1)
    )
    endpoints = tuple((float(value[0]), float(value[1])) for value in endpoint_values)
    return PushGeometryPlan(
        direction_xy=(float(direction[0]), float(direction[1])),
        support_extent=extent,
        contact_point=(float(contact_xy[0]), float(contact_xy[1]), push_height),
        precontact_point=(
            float(precontact_xy[0]),
            float(precontact_xy[1]),
            precontact_height,
        ),
        staging_point=(float(precontact_xy[0]), float(precontact_xy[1]), staging_height),
        object_goal_xy=(float(object_goal[0]), float(object_goal[1])),
        object_path_margin=object_margin,
        tcp_path_margin=tcp_margin,
        segment_endpoints_xy=endpoints,
    )


def build_bounded_joint_action(
    arm_position: np.ndarray,
    *,
    action_low: np.ndarray,
    action_high: np.ndarray,
    gripper_command: float = -1.0,
) -> np.ndarray:
    """Build one native 8-D action and reject rather than clip contract violations."""

    arm = _vector(arm_position, 7, "arm_position")
    low = _vector(action_low, 8, "action_low")
    high = _vector(action_high, 8, "action_high")
    if np.any(low > high):
        raise ValueError("action bounds are inverted")
    if not math.isfinite(gripper_command):
        raise ValueError("gripper_command must be finite")
    action = np.asarray((*arm, float(gripper_command)), dtype=np.float64)
    if np.any(action < low) or np.any(action > high):
        raise ValueError("planned action exceeds native controller bounds")
    return action


def _vector(value: np.ndarray, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain {size} finite values")
    return result


__all__ = [
    "PlanarPrimitive",
    "PrimitiveKind",
    "PushGeometryPlan",
    "WorkspaceContract",
    "box_support_extent",
    "build_bounded_joint_action",
    "build_geometry_push_plan",
    "footprint_workspace_margin",
    "normalized_planar_direction",
    "oriented_axis_extents",
    "point_workspace_margin",
    "support_extent",
]
