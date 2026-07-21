from __future__ import annotations

import numpy as np
import pytest

from langmani.experts.push_strategy import (
    build_lateral_approach_candidates,
    is_lateral_region,
    point_to_segment_distance,
)


def _candidates(direction: tuple[float, float]):
    return build_lateral_approach_candidates(
        desired_direction=np.asarray(direction),
        object_position=np.array([-0.18, 0.0, 0.025]),
        tcp_position=np.array([0.0, 0.0, 0.18]),
        distractors=((np.array([0.0, 0.3]), 0.025),),
        contact_offset=0.04,
        precontact_clearance=0.075,
        precontact_height=0.12,
        push_height=0.015,
        workspace_bounds_xy=(-0.43, 0.43, -0.38, 0.38),
        compensation_degrees=15.0,
        minimum_obstacle_clearance=0.015,
    )


def test_lateral_classification_does_not_capture_forward_targets() -> None:
    assert is_lateral_region("left")
    assert is_lateral_region("right")
    assert not is_lateral_region("forward_left")
    assert not is_lateral_region("forward_right")


def test_direction_aware_candidates_compensate_opposite_signed_lateral_drift() -> None:
    left = _candidates((0.75, 0.66))[0]
    right = _candidates((0.75, -0.66))[0]

    assert left.safe and right.safe
    assert left.angle_degrees == pytest.approx(15.0)
    assert right.angle_degrees == pytest.approx(-15.0)
    assert left.contact_direction[0] < 0.75
    assert left.contact_direction[1] > 0.66
    assert right.contact_direction[0] < 0.75
    assert right.contact_direction[1] < -0.66


def test_candidate_scoring_rejects_a_distractor_swept_path() -> None:
    candidates = build_lateral_approach_candidates(
        desired_direction=np.array([0.8, 0.6]),
        object_position=np.array([-0.18, 0.0, 0.025]),
        tcp_position=np.array([0.0, 0.0, 0.18]),
        distractors=((np.array([-0.12, -0.02]), 0.035),),
        contact_offset=0.045,
        precontact_clearance=0.075,
        precontact_height=0.12,
        push_height=0.025,
        workspace_bounds_xy=(-0.43, 0.43, -0.38, 0.38),
        compensation_degrees=15.0,
        minimum_obstacle_clearance=0.015,
    )

    assert any(not candidate.safe for candidate in candidates)
    assert all(candidate.safe for candidate in candidates[: sum(item.safe for item in candidates)])


def test_zero_compensation_emits_bounded_symmetric_fallbacks() -> None:
    candidates = build_lateral_approach_candidates(
        desired_direction=np.array([0.8, 0.6]),
        object_position=np.array([-0.18, 0.0, 0.025]),
        tcp_position=np.array([0.0, 0.0, 0.18]),
        distractors=(),
        contact_offset=0.045,
        precontact_clearance=0.075,
        precontact_height=0.12,
        push_height=0.015,
        workspace_bounds_xy=(-0.43, 0.43, -0.38, 0.38),
        compensation_degrees=0.0,
        minimum_obstacle_clearance=0.015,
    )

    assert len(candidates) == 5
    assert {candidate.angle_degrees for candidate in candidates} == {
        -15.0,
        -7.5,
        0.0,
        7.5,
        15.0,
    }
    assert candidates[0].angle_degrees == pytest.approx(0.0)
    assert candidates[0].contact_direction == pytest.approx((0.8, 0.6))


def test_point_to_segment_distance_is_bounded_to_the_swept_segment() -> None:
    assert point_to_segment_distance(
        np.array([2.0, 1.0]), np.array([0.0, 0.0]), np.array([1.0, 0.0])
    ) == pytest.approx(2**0.5)
