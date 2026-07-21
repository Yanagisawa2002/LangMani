"""Pure direction-aware strategy helpers for bounded Phase 2A side pushes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

LATERAL_REGION_IDS = frozenset({"left", "right"})
PANDA_BASE_XY = np.array([-0.615, 0.0], dtype=np.float64)
TCP_SWEEP_RADIUS = 0.04


def is_lateral_region(region_id: str) -> bool:
    return region_id in LATERAL_REGION_IDS


def rotate_planar(direction: np.ndarray, angle_degrees: float) -> np.ndarray:
    vector = np.asarray(direction, dtype=np.float64)
    if vector.shape != (2,) or not np.isfinite(vector).all():
        raise ValueError("direction must contain two finite values")
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        raise ValueError("direction norm must be positive")
    angle = math.radians(float(angle_degrees))
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )
    result = rotation @ (vector / norm)
    return result / np.linalg.norm(result)


def point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    point_xy = np.asarray(point, dtype=np.float64)[:2]
    start_xy = np.asarray(start, dtype=np.float64)[:2]
    end_xy = np.asarray(end, dtype=np.float64)[:2]
    segment = end_xy - start_xy
    denominator = float(np.dot(segment, segment))
    if denominator < 1e-12:
        return float(np.linalg.norm(point_xy - start_xy))
    fraction = float(np.clip(np.dot(point_xy - start_xy, segment) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point_xy - (start_xy + fraction * segment)))


@dataclass(frozen=True, slots=True)
class LateralApproachCandidate:
    angle_degrees: float
    contact_direction: tuple[float, float]
    precontact_point: tuple[float, float, float]
    contact_point: tuple[float, float, float]
    alignment: float
    reachability_score: float
    workspace_margin: float
    obstacle_clearance: float
    score: float
    safe: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "angle_degrees": self.angle_degrees,
            "contact_direction": list(self.contact_direction),
            "precontact_point": list(self.precontact_point),
            "contact_point": list(self.contact_point),
            "alignment": self.alignment,
            "reachability_score": self.reachability_score,
            "workspace_margin": self.workspace_margin,
            "obstacle_clearance": self.obstacle_clearance,
            "score": self.score,
            "safe": self.safe,
        }


def build_lateral_approach_candidates(
    *,
    desired_direction: np.ndarray,
    object_position: np.ndarray,
    tcp_position: np.ndarray,
    distractors: tuple[tuple[np.ndarray, float], ...],
    contact_offset: float,
    precontact_clearance: float,
    precontact_height: float,
    push_height: float,
    workspace_bounds_xy: tuple[float, float, float, float],
    compensation_degrees: float,
    minimum_obstacle_clearance: float,
) -> tuple[LateralApproachCandidate, ...]:
    """Score four deterministic normals and return safe candidates first."""

    desired = np.asarray(desired_direction, dtype=np.float64)
    side = 1.0 if desired[1] >= 0.0 else -1.0
    preferred = side * float(compensation_degrees)
    angles = tuple(dict.fromkeys((preferred, preferred * 0.5, 0.0, -preferred * 0.5)))
    x_min, x_max, y_min, y_max = workspace_bounds_xy
    candidates: list[LateralApproachCandidate] = []
    for angle in angles:
        direction = rotate_planar(desired, angle)
        precontact = np.array(
            [
                *(object_position[:2] - direction * (contact_offset + precontact_clearance)),
                precontact_height,
            ],
            dtype=np.float64,
        )
        contact = np.array(
            [
                *(object_position[:2] - direction * contact_offset),
                push_height,
            ],
            dtype=np.float64,
        )
        workspace_margin = min(
            precontact[0] - x_min,
            x_max - precontact[0],
            precontact[1] - y_min,
            y_max - precontact[1],
        )
        radial = float(np.linalg.norm(precontact[:2] - PANDA_BASE_XY))
        reachability = 1.0 - abs(radial - 0.34) / 0.34
        obstacle_clearance = math.inf
        for center, radius in distractors:
            clearance = point_to_segment_distance(center, tcp_position, precontact) - (
                radius + TCP_SWEEP_RADIUS
            )
            obstacle_clearance = min(obstacle_clearance, clearance)
        if not distractors:
            obstacle_clearance = 1.0
        alignment = float(np.dot(desired, direction))
        preferred_error = abs(angle - preferred) / max(abs(preferred), 1.0)
        score = (
            4.0 * alignment
            + 1.5 * reachability
            + 2.0 * min(workspace_margin, 0.2)
            + 3.0 * min(obstacle_clearance, 0.2)
            - 0.4 * preferred_error
        )
        safe = workspace_margin >= 0.02 and obstacle_clearance >= minimum_obstacle_clearance
        candidates.append(
            LateralApproachCandidate(
                angle_degrees=float(angle),
                contact_direction=tuple(float(value) for value in direction),
                precontact_point=tuple(float(value) for value in precontact),
                contact_point=tuple(float(value) for value in contact),
                alignment=alignment,
                reachability_score=reachability,
                workspace_margin=float(workspace_margin),
                obstacle_clearance=float(obstacle_clearance),
                score=float(score),
                safe=safe,
            )
        )
    return tuple(sorted(candidates, key=lambda item: (not item.safe, -item.score)))


__all__ = [
    "LATERAL_REGION_IDS",
    "LateralApproachCandidate",
    "build_lateral_approach_candidates",
    "is_lateral_region",
    "point_to_segment_distance",
    "rotate_planar",
]
