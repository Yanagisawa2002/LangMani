"""Pure batched tensor logic shared by the M1 environment and unit tests."""

from __future__ import annotations

from collections.abc import Mapping

import torch

PRIVILEGED_OBSERVATION_KEYS = frozenset(
    {"cube_poses", "bin_centers", "target_object_index", "target_bin_index"}
)


def build_observation_extra(
    *,
    tcp_pose: torch.Tensor,
    use_privileged_state: bool,
    cube_poses: torch.Tensor | None = None,
    bin_centers: torch.Tensor | None = None,
    target_object_index: torch.Tensor | None = None,
    target_bin_index: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build task extras while making the visual-only no-leak gate explicit."""
    extra = {"tcp_pose": tcp_pose}
    if not use_privileged_state:
        return extra

    privileged = {
        "cube_poses": cube_poses,
        "bin_centers": bin_centers,
        "target_object_index": target_object_index,
        "target_bin_index": target_bin_index,
    }
    missing = [name for name, value in privileged.items() if value is None]
    if missing:
        raise ValueError("privileged state requested without fields: " + ", ".join(sorted(missing)))
    extra.update({name: value for name, value in privileged.items() if value is not None})
    return extra


def cube_in_bin_matrix(
    cube_positions: torch.Tensor,
    bin_floor_centers: torch.Tensor,
    *,
    cube_bounding_radius: float,
    cube_resting_height: float,
    bin_interior_half_size: float,
    containment_clearance: float,
    resting_height_tolerance: float,
) -> torch.Tensor:
    """Return ``(batch, objects, bins)`` conservative containment flags.

    The horizontal test uses a sphere enclosing the cube, so a flagged cube fits
    within every interior wall regardless of its orientation.  The vertical test
    requires the center to be near the resting height and therefore rejects a cube
    merely held above a bin.
    """
    if cube_positions.ndim != 3 or cube_positions.shape[-1] != 3:
        raise ValueError("cube_positions must have shape (batch, objects, 3)")
    if bin_floor_centers.ndim != 3 or bin_floor_centers.shape[-1] != 3:
        raise ValueError("bin_floor_centers must have shape (batch, bins, 3)")
    if cube_positions.shape[0] != bin_floor_centers.shape[0]:
        raise ValueError("cube and bin tensors must have the same batch size")

    horizontal_margin = bin_interior_half_size - cube_bounding_radius - containment_clearance
    if horizontal_margin <= 0:
        raise ValueError("bin interior is too small for the configured cube containment")

    offset = cube_positions[:, :, None, :] - bin_floor_centers[:, None, :, :]
    within_xy = (torch.abs(offset[..., :2]) <= horizontal_margin).all(dim=-1)
    at_resting_height = torch.abs(offset[..., 2] - cube_resting_height) <= resting_height_tolerance
    return within_xy & at_resting_height


def evaluate_task_state(
    *,
    cube_positions: torch.Tensor,
    bin_floor_centers: torch.Tensor,
    cube_is_grasped: torch.Tensor,
    cube_is_static: torch.Tensor,
    target_object_index: torch.Tensor,
    target_bin_index: torch.Tensor,
    cube_bounding_radius: float,
    cube_resting_height: float,
    bin_interior_half_size: float,
    containment_clearance: float,
    resting_height_tolerance: float,
    off_table_height: float,
) -> Mapping[str, torch.Tensor]:
    """Compute all M1 success/failure diagnostics with batched Torch operations."""
    batch_size, object_count, _ = cube_positions.shape
    if bin_floor_centers.shape[1] != 2:
        raise ValueError("M1 evaluation expects exactly two bins")
    if cube_is_grasped.shape != (batch_size, object_count):
        raise ValueError("cube_is_grasped has an incompatible shape")
    if cube_is_static.shape != (batch_size, object_count):
        raise ValueError("cube_is_static has an incompatible shape")
    if target_object_index.shape != (batch_size,):
        raise ValueError("target_object_index must have shape (batch,)")
    if target_bin_index.shape != (batch_size,):
        raise ValueError("target_bin_index must have shape (batch,)")

    in_bin = cube_in_bin_matrix(
        cube_positions,
        bin_floor_centers,
        cube_bounding_radius=cube_bounding_radius,
        cube_resting_height=cube_resting_height,
        bin_interior_half_size=bin_interior_half_size,
        containment_clearance=containment_clearance,
        resting_height_tolerance=resting_height_tolerance,
    )
    batch_index = torch.arange(batch_size, device=cube_positions.device)
    target_in_target_bin = in_bin[batch_index, target_object_index, target_bin_index]
    target_in_wrong_bin = in_bin[batch_index, target_object_index, 1 - target_bin_index]

    target_bin_occupancy = torch.gather(
        in_bin,
        dim=2,
        index=target_bin_index[:, None, None].expand(-1, object_count, 1),
    ).squeeze(2)
    object_index = torch.arange(object_count, device=cube_positions.device)[None, :]
    wrong_object_in_target_bin = (
        target_bin_occupancy & (object_index != target_object_index[:, None])
    ).any(dim=1)

    target_is_grasped = cube_is_grasped[batch_index, target_object_index]
    target_is_static = cube_is_static[batch_index, target_object_index]
    target_position = cube_positions[batch_index, target_object_index]

    target_off_table = target_position[:, 2] < off_table_height

    success = (
        target_in_target_bin
        & ~wrong_object_in_target_bin
        & ~target_is_grasped
        & target_is_static
        & ~target_off_table
    )
    fail = target_off_table
    return {
        "target_in_target_bin": target_in_target_bin,
        "target_in_wrong_bin": target_in_wrong_bin,
        "wrong_object_in_target_bin": wrong_object_in_target_bin,
        "target_is_grasped": target_is_grasped,
        "target_is_static": target_is_static,
        "target_off_table": target_off_table,
        "success": success,
        "fail": fail,
    }
