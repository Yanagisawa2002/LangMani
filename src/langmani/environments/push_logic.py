"""Pure batched tensor logic for planar pushing evaluation."""

from __future__ import annotations

from collections.abc import Mapping

import torch

PUSH_PRIVILEGED_OBSERVATION_KEYS = frozenset(
    {
        "object_poses",
        "target_region_center",
        "target_object_index",
        "target_region_index",
        "difficulty_index",
    }
)


def compute_push_object_planar_alignment(
    cube_rotation: torch.Tensor,
    cylinder_rotation: torch.Tensor,
) -> torch.Tensor:
    """Measure the cube's uprightness and the rolling cylinder's planarity."""

    if cube_rotation.ndim != 3 or cube_rotation.shape[-2:] != (3, 3):
        raise ValueError("cube_rotation must have shape (batch, 3, 3)")
    if cylinder_rotation.shape != cube_rotation.shape:
        raise ValueError("cylinder_rotation must match cube_rotation")
    cube_up = torch.abs(cube_rotation[:, 2, 2])
    cylinder_axis_vertical = torch.abs(cylinder_rotation[:, 2, 0]).clamp(0.0, 1.0)
    cylinder_horizontal = torch.sqrt((1.0 - cylinder_axis_vertical.square()).clamp_min(0.0))
    return torch.stack((cube_up, cylinder_horizontal), dim=1)


def build_push_observation_extra(
    *,
    tcp_pose: torch.Tensor,
    use_privileged_state: bool,
    object_poses: torch.Tensor | None = None,
    target_region_center: torch.Tensor | None = None,
    target_object_index: torch.Tensor | None = None,
    target_region_index: torch.Tensor | None = None,
    difficulty_index: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    extra = {"tcp_pose": tcp_pose}
    if not use_privileged_state:
        return extra
    privileged = {
        "object_poses": object_poses,
        "target_region_center": target_region_center,
        "target_object_index": target_object_index,
        "target_region_index": target_region_index,
        "difficulty_index": difficulty_index,
    }
    missing = [key for key, value in privileged.items() if value is None]
    if missing:
        raise ValueError("privileged push state requested without fields: " + ", ".join(missing))
    extra.update({key: value for key, value in privileged.items() if value is not None})
    return extra


def update_stable_success_count(
    previous_count: torch.Tensor,
    stable_condition: torch.Tensor,
) -> torch.Tensor:
    if previous_count.ndim != 1 or stable_condition.shape != previous_count.shape:
        raise ValueError("stable-success tensors must share shape (batch,)")
    if stable_condition.dtype is not torch.bool:
        raise TypeError("stable_condition must be boolean")
    return torch.where(stable_condition, previous_count + 1, torch.zeros_like(previous_count))


def update_progress_state(
    *,
    distance: torch.Tensor,
    best_distance: torch.Tensor,
    steps_without_progress: torch.Tensor,
    minimum_improvement: float,
    stall_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if distance.shape != best_distance.shape or distance.shape != steps_without_progress.shape:
        raise ValueError("progress tensors must share shape (batch,)")
    improved = distance < best_distance - minimum_improvement
    new_best = torch.minimum(best_distance, distance)
    new_steps = torch.where(
        improved,
        torch.zeros_like(steps_without_progress),
        steps_without_progress + 1,
    )
    return new_best, new_steps, new_steps >= stall_steps


def evaluate_push_state(
    *,
    object_positions: torch.Tensor,
    object_up_alignment: torch.Tensor,
    object_is_static: torch.Tensor,
    object_is_grasped: torch.Tensor,
    initial_object_positions: torch.Tensor,
    initial_target_position: torch.Tensor,
    target_region_center: torch.Tensor,
    target_object_index: torch.Tensor,
    object_planar_radii: torch.Tensor,
    object_resting_heights: torch.Tensor,
    target_region_radius: torch.Tensor,
    stable_success_count: torch.Tensor,
    required_stable_steps: int,
    workspace_bounds_xy: tuple[float, float, float, float],
    containment_clearance: float,
    lift_tolerance: float,
    topple_alignment_threshold: float,
    wrong_object_displacement_threshold: float,
    wrong_object_contact: torch.Tensor,
    action_out_of_bounds: torch.Tensor,
    invalid_action: torch.Tensor,
    progress_stalled: torch.Tensor,
    robot_collision: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    """Return conservative success plus independently observable push events."""

    if object_positions.ndim != 3 or object_positions.shape[-1] != 3:
        raise ValueError("object_positions must have shape (batch, objects, 3)")
    batch_size, object_count, _ = object_positions.shape
    expected_objects = (batch_size, object_count)
    for value, label in (
        (object_up_alignment, "object_up_alignment"),
        (object_is_static, "object_is_static"),
        (object_is_grasped, "object_is_grasped"),
        (object_planar_radii, "object_planar_radii"),
        (object_resting_heights, "object_resting_heights"),
    ):
        if value.shape != expected_objects:
            raise ValueError(f"{label} must have shape (batch, objects)")
    if initial_object_positions.shape != object_positions.shape:
        raise ValueError("initial_object_positions must match object_positions")
    for value, label in (
        (target_object_index, "target_object_index"),
        (target_region_radius, "target_region_radius"),
        (stable_success_count, "stable_success_count"),
        (wrong_object_contact, "wrong_object_contact"),
        (action_out_of_bounds, "action_out_of_bounds"),
        (invalid_action, "invalid_action"),
        (progress_stalled, "progress_stalled"),
        (robot_collision, "robot_collision"),
    ):
        if value.shape != (batch_size,):
            raise ValueError(f"{label} must have shape (batch,)")
    if initial_target_position.shape != (batch_size, 3):
        raise ValueError("initial_target_position must have shape (batch, 3)")
    if target_region_center.shape != (batch_size, 3):
        raise ValueError("target_region_center must have shape (batch, 3)")

    batch_index = torch.arange(batch_size, device=object_positions.device)
    target_position = object_positions[batch_index, target_object_index]
    target_radius = object_planar_radii[batch_index, target_object_index]
    target_resting_height = object_resting_heights[batch_index, target_object_index]
    target_up_alignment = object_up_alignment[batch_index, target_object_index]
    target_static = object_is_static[batch_index, target_object_index]
    target_grasped = object_is_grasped[batch_index, target_object_index]

    target_distance = torch.linalg.vector_norm(
        target_position[..., :2] - target_region_center[..., :2], dim=1
    )
    inside_margin = target_region_radius - target_radius - containment_clearance
    target_inside_region = target_distance <= inside_margin

    min_x, max_x, min_y, max_y = workspace_bounds_xy
    target_outside_workspace = (
        (target_position[:, 0] - target_radius < min_x)
        | (target_position[:, 0] + target_radius > max_x)
        | (target_position[:, 1] - target_radius < min_y)
        | (target_position[:, 1] + target_radius > max_y)
        | (target_position[:, 2] < -target_radius)
    )
    target_lifted = target_position[:, 2] > target_resting_height + lift_tolerance
    target_toppled = target_up_alignment < topple_alignment_threshold

    object_displacement = torch.linalg.vector_norm(
        object_positions[..., :2] - initial_object_positions[..., :2], dim=2
    )
    object_indices = torch.arange(object_count, device=object_positions.device)[None, :]
    wrong_mask = object_indices != target_object_index[:, None]
    wrong_object_displaced = (
        (object_displacement > wrong_object_displacement_threshold) & wrong_mask
    ).any(dim=1)

    target_path = target_region_center[..., :2] - initial_target_position[..., :2]
    target_path_direction = target_path / torch.linalg.vector_norm(
        target_path, dim=1, keepdim=True
    ).clamp_min(1e-6)
    beyond_target = target_position[..., :2] - target_region_center[..., :2]
    target_overshoot = (beyond_target * target_path_direction).sum(dim=1) > target_radius

    stable_success = stable_success_count >= required_stable_steps
    fail = (
        invalid_action | target_outside_workspace | target_lifted | target_toppled | target_grasped
    )
    success = stable_success & target_inside_region & target_static & ~target_overshoot & ~fail
    return {
        "target_inside_region": target_inside_region,
        "target_is_static": target_static,
        "stable_success_steps": stable_success_count,
        "target_distance": target_distance,
        "wrong_object_contact": wrong_object_contact,
        "wrong_object_displaced": wrong_object_displaced,
        "target_outside_workspace": target_outside_workspace,
        "target_overshoot": target_overshoot,
        "target_toppled": target_toppled,
        "target_lifted": target_lifted,
        "target_is_grasped": target_grasped,
        "invalid_action": invalid_action,
        "action_out_of_bounds": action_out_of_bounds,
        "no_progress_stall": progress_stalled,
        "robot_collision": robot_collision,
        "success": success,
        "fail": fail,
    }


__all__ = [
    "PUSH_PRIVILEGED_OBSERVATION_KEYS",
    "build_push_observation_extra",
    "compute_push_object_planar_alignment",
    "evaluate_push_state",
    "update_progress_state",
    "update_stable_success_count",
]
