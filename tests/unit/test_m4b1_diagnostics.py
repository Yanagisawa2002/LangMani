"""Rule tests use synthetic states, never physical acceptance evidence."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from langmani.policies.m4b1_diagnostics import (
    classify,
    diagnose,
    first_event,
    pair_metrics,
    reference_metrics,
)
from langmani.policies.m4b_protocol import GOALS


def trace(n: int = 30) -> dict[str, np.ndarray]:
    return {
        "tcp": np.tile([-0.3, 0, 0.4], (n, 1)),
        "cubes": np.tile(
            [[-0.12, 0, 0.025], [-0.12, -0.12, 0.025], [-0.12, 0.12, 0.025]], (n, 1, 1)
        ),
        "bins": np.tile([[0.08, 0.18, 0.008], [0.08, -0.18, 0.008]], (n, 1, 1)),
        "finger_forces": np.zeros((n, 5, 2, 3)),
        "grasped": np.zeros((n, 3), dtype=bool),
    }


def test_destination_persistence_height_progress_and_empty_hand() -> None:
    t = trace()
    t["tcp"][2:4] = [0.08, 0.18, 0.1]
    assert diagnose(t)["selected_goal"] is None
    t["tcp"][4] = [0.08, 0.18, 0.1]
    d = diagnose(t)
    assert d["selected_goal"] == GOALS[0]
    assert d["events"]["first_approached_goal"]["step"] == 4
    assert d["red_grasped"] is False
    t["tcp"][2:5, 2] = 0.3
    assert diagnose(t)["selected_goal"] is None
    t["tcp"][:] = [0.08, 0.18, 0.1]
    assert diagnose(t)["selected_goal"] is None  # proximity already at reset


def test_contacts_grasps_are_object_and_destination_distinct() -> None:
    t = trace()
    t["finger_forces"][0, 0, 0, 0] = 10  # ignored initial contact
    t["finger_forces"][1, 1, 0, 0] = 0.099
    t["finger_forces"][2, 1, 0, 0] = 0.1
    t["finger_forces"][3, 0, 0, 0] = 0.6
    t["finger_forces"][6, 4, 0, 0] = 0.2
    t["grasped"][4, 0] = True
    assert diagnose(t)["first_grasp_object"] is None
    t["grasped"][5, 0] = True
    d = diagnose(t)
    assert d["first_contact_object"] == "green_cube"
    assert d["first_contact_goal"] == GOALS[1]
    assert d["first_grasp_object"] == "red_cube"
    assert d["red_contacted"] and d["red_grasped"]
    assert d["selected_goal"] is None


def test_simultaneous_first_events_are_ambiguous_not_label_biased() -> None:
    event = first_event(np.array([[False, False], [True, True]]), GOALS)
    assert event == {"value": None, "step": 1, "ambiguous": True}


def test_placement_requires_grasp_lift_descent_and_persistence() -> None:
    t = trace()
    t["cubes"][6:, 0] = [0.08, 0.18, 0.06]
    assert diagnose(t)["placement_attempt_goal"] is None
    t["grasped"][2:8, 0] = True
    t["cubes"][3:6, 0] = [0, 0.1, 0.14]
    d = diagnose(t)
    assert d["placement_attempt_goal"] == GOALS[0]
    assert d["events"]["placement_attempt_goal"]["step"] == 7
    assert d["final_goal"] == GOALS[0]
    assert not d["red_dropped_outside_bin"]
    t["cubes"][8:, 0] = [-0.12, 0, 0.025]
    assert diagnose(t)["red_dropped_outside_bin"]
    assert diagnose(t)["final_goal"] is None


def test_nan_shape_and_native_bool_rejected() -> None:
    for key in ("tcp", "cubes", "finger_forces"):
        t = trace()
        t[key].flat[0] = np.nan
        with pytest.raises(ValueError, match="invalid trace"):
            diagnose(t)
    t = trace()
    t["grasped"] = t["grasped"].astype(float)
    with pytest.raises(ValueError, match="native boolean"):
        diagnose(t)


def test_taxonomy_precedence_and_no_language_claim_for_nonselection() -> None:
    d = diagnose(trace())
    assert classify(d, GOALS[0], None, "timeout") == "oscillation_or_stall"
    d["selected_goal"] = GOALS[1]
    assert classify(d, GOALS[0], None, "timeout") == "wrong_goal_selected"
    assert classify(d, GOALS[1], None, "timeout") == "correct_goal_no_contact"
    d["red_contacted"] = True
    assert classify(d, GOALS[1], None, "timeout") == "correct_goal_contact_no_grasp"
    d["red_grasped"] = True
    d["red_dropped_outside_bin"] = True
    assert classify(d, GOALS[1], None, "timeout") == "correct_goal_grasp_then_drop"
    d["placement_attempt_goal"] = GOALS[0]
    assert classify(d, GOALS[1], None, "timeout") == "correct_goal_wrong_destination"
    assert classify(d, GOALS[1], GOALS[1], "goal_reached") == "success"
    assert classify(d, GOALS[1], GOALS[1], "infrastructure_error") == "infrastructure_error"
    assert classify(d, None, None, "timeout") == "unprompted_destination_failure"


def test_swapped_references_blank_denominator_and_cumulative_funnel() -> None:
    d = diagnose(trace())
    row = {
        **d,
        "selected_goal": GOALS[1],
        "first_approached_goal": GOALS[1],
        "prompt_goal": GOALS[1],
        "semantic_goal": GOALS[0],
        "achieved_goal": None,
        "termination_reason": "timeout",
    }
    supplied = reference_metrics([row], "prompt_goal")
    requested = reference_metrics([row], "semantic_goal")
    assert supplied["instructed_goal_approach_accuracy"]["count"] == 1
    assert requested["instructed_goal_approach_accuracy"]["count"] == 0
    assert supplied["completion_after_correct_selection"] == {
        "count": 0,
        "denominator": 1,
        "rate": 0,
    }
    assert supplied["cumulative_funnel"]["correct_destination_approach"]["count"] == 0
    row["prompt_goal"] = None
    assert reference_metrics([row], "prompt_goal")["full_success"]["rate"] is None


def test_pair_partition_keeps_same_target_and_partial_success_separate() -> None:
    pairs = [
        (GOALS[0], GOALS[1]),
        (GOALS[0], GOALS[0]),
        (GOALS[0], None),
        (None, None),
        (GOALS[1], GOALS[0]),
    ]
    rows = [
        {"seed": i, "prompt_goal": GOALS[j], "selected_goal": selected, "achieved_goal": None}
        for i, pair in enumerate(pairs)
        for j, selected in enumerate(pair)
    ]
    result = pair_metrics(rows)
    assert set(result["categories"].values()) == {1}
    assert result["paired_instruction_switch_accuracy"]["count"] == 1
    assert result["paired_full_success"]["count"] == 0
    assert result["one_corresponding_including_same_target"] == 2
    with pytest.raises(ValueError, match="incomplete"):
        pair_metrics(rows[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        pair_metrics(rows + [copy.deepcopy(rows[0])])
