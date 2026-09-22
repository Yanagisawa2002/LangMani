"""Deterministic, offline behavioral diagnostics; never used as policy observations."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from langmani.environments.specs import OBJECT_IDS
from langmani.policies.m4b_protocol import GOALS

PROTOCOL_VERSION = "langmani-m4b1-diagnostics-v1"
THRESHOLDS = {
    "object_approach_m": 0.07,
    "object_progress_m": 0.04,
    "object_samples": 2,
    "destination_xy_m": 0.12,
    "destination_height_m": 0.20,
    "destination_progress_m": 0.08,
    "destination_samples": 3,
    "contact_force_n": 0.1,
    "grasp_samples": 2,
    "placement_xy_m": 0.085,
    "placement_height_m": 0.15,
    "lift_m": 0.02,
    "descent_m": 0.02,
    "placement_samples": 2,
    "drop_samples": 3,
    "stall_samples": 20,
    "stall_radius_m": 0.01,
    "oscillation_samples": 40,
    "oscillation_path_m": 0.10,
    "oscillation_net_m": 0.015,
}


def persistent(mask: np.ndarray, samples: int) -> np.ndarray:
    """A true event at t means the predicate held for samples ending at t."""
    result = np.zeros_like(mask, dtype=bool)
    for t in range(samples, len(mask)):
        result[t] = mask[t - samples + 1 : t + 1].all(axis=0)
    return result


def first_event(mask: np.ndarray, labels: tuple[str, ...]) -> dict[str, Any]:
    indices = np.flatnonzero(mask.any(axis=1))
    if not len(indices):
        return {"value": None, "step": None, "ambiguous": False}
    t = int(indices[0])
    choices = np.flatnonzero(mask[t])
    return {
        "value": labels[int(choices[0])] if len(choices) == 1 else None,
        "step": t,
        "ambiguous": len(choices) != 1,
    }


def inside_bins(points: np.ndarray, bins: np.ndarray) -> np.ndarray:
    relative = points[:, None, :] - bins
    return (
        (np.abs(relative[..., :2]) <= THRESHOLDS["placement_xy_m"]).all(axis=-1)
        & (relative[..., 2] >= 0)
        & (relative[..., 2] <= THRESHOLDS["placement_height_m"])
    )


def diagnose(trace: dict[str, np.ndarray]) -> dict[str, Any]:
    """Reduce T+1 samples, keeping object identity distinct from destination semantics."""
    tcp = np.asarray(trace["tcp"], dtype=np.float64)
    n = len(tcp)
    shapes = {
        "tcp": (n, 3),
        "cubes": (n, 3, 3),
        "bins": (n, 2, 3),
        "finger_forces": (n, 5, 2, 3),
        "grasped": (n, 3),
    }
    if not 2 <= n <= 201:
        raise ValueError("diagnostics require 1..200 actions and initial state")
    for key, shape in shapes.items():
        if trace[key].shape != shape or not np.isfinite(trace[key]).all():
            raise ValueError(f"invalid trace: {key}")
    if trace["grasped"].dtype != np.bool_:
        raise ValueError("grasped must contain native boolean flags")
    cubes, bins = trace["cubes"], trace["bins"]
    object_distance = np.linalg.norm(tcp[:, None, :] - cubes, axis=-1)
    object_approach = (object_distance <= THRESHOLDS["object_approach_m"]) & (
        object_distance[0] - object_distance >= THRESHOLDS["object_progress_m"]
    )
    relative = tcp[:, None, :] - bins
    xy = np.linalg.norm(relative[..., :2], axis=-1)
    destination = (
        (xy <= THRESHOLDS["destination_xy_m"])
        & (xy[0] - xy >= THRESHOLDS["destination_progress_m"])
        & (relative[..., 2] >= 0)
        & (relative[..., 2] <= THRESHOLDS["destination_height_m"])
    )
    contact = (
        np.linalg.norm(trace["finger_forces"], axis=-1).max(axis=-1)
        >= THRESHOLDS["contact_force_n"]
    )
    contact[0] = False
    grasp = persistent(trace["grasped"], int(THRESHOLDS["grasp_samples"]))
    object_approach = persistent(object_approach, int(THRESHOLDS["object_samples"]))
    destination = persistent(destination, int(THRESHOLDS["destination_samples"]))
    red_grasp = grasp[:, 0]
    grasp_seen = np.maximum.accumulate(red_grasp)
    lift = cubes[:, 0, 2] >= cubes[0, 0, 2] + THRESHOLDS["lift_m"]
    lift_seen = np.maximum.accumulate(grasp_seen & lift)
    peak = np.maximum.accumulate(np.where(grasp_seen, cubes[:, 0, 2], -np.inf))
    inside = inside_bins(cubes[:, 0], bins)
    placement = (
        inside
        & (grasp_seen & lift_seen)[:, None]
        & ((peak - cubes[:, 0, 2]) >= THRESHOLDS["descent_m"])[:, None]
    )
    placement = persistent(placement, int(THRESHOLDS["placement_samples"]))
    dropped = persistent(
        (lift_seen & ~trace["grasped"][:, 0] & ~inside.any(axis=1))[:, None],
        int(THRESHOLDS["drop_samples"]),
    )
    events = {
        "first_approached_object": first_event(object_approach, OBJECT_IDS),
        "first_approached_goal": first_event(destination, GOALS),
        "first_contact_object": first_event(contact[:, :3], OBJECT_IDS),
        "first_contact_goal": first_event(contact[:, 3:], GOALS),
        "first_grasp_object": first_event(grasp, OBJECT_IDS),
        "placement_attempt_goal": first_event(placement, GOALS),
    }
    final_choices = np.flatnonzero(inside[-1])
    tail = tcp[-int(THRESHOLDS["stall_samples"]) :]
    stalled = len(tail) == THRESHOLDS["stall_samples"] and (
        np.linalg.norm(tail - tail[0], axis=-1).max() <= THRESHOLDS["stall_radius_m"]
    )
    tail = tcp[-int(THRESHOLDS["oscillation_samples"]) :]
    path = float(np.linalg.norm(np.diff(tail, axis=0), axis=-1).sum())
    net = float(np.linalg.norm(tail[-1] - tail[0]))
    oscillation = (
        len(tail) == THRESHOLDS["oscillation_samples"]
        and path >= THRESHOLDS["oscillation_path_m"]
        and net <= THRESHOLDS["oscillation_net_m"]
    )
    return {
        "schema_version": PROTOCOL_VERSION,
        **{key: value["value"] for key, value in events.items()},
        "events": events,
        "selected_goal": events["first_approached_goal"]["value"],
        "final_goal": GOALS[int(final_choices[0])] if len(final_choices) == 1 else None,
        "red_approached": bool(object_approach[:, 0].any()),
        "red_contacted": bool(contact[:, 0].any()),
        "red_grasped": bool(red_grasp.any()),
        "red_lifted_after_grasp": bool(lift_seen.any()),
        "red_dropped_outside_bin": bool(dropped.any()),
        "any_object_approached": bool(object_approach.any()),
        "any_object_contacted": bool(contact[:, :3].any()),
        "any_object_grasped": bool(grasp.any()),
        "stalled": bool(stalled),
        "oscillating": bool(oscillation),
        "tail_tcp_path_m": path,
        "tail_tcp_net_m": net,
    }


def classify(d: dict[str, Any], goal: str | None, achieved: str | None, reason: str) -> str:
    """First-match rules; no-selection failures do not imply semantic misunderstanding."""
    if reason == "infrastructure_error":
        return "infrastructure_error"
    if goal is not None and achieved == goal:
        return "success"
    selected = d["selected_goal"]
    if selected is not None:
        if goal is None:
            return (
                "unprompted_destination_reached" if achieved else "unprompted_destination_failure"
            )
        if selected != goal:
            return "wrong_goal_selected"
        if any(d[key] not in (None, goal) for key in ("placement_attempt_goal", "final_goal")):
            return "correct_goal_wrong_destination"
        if not d["red_contacted"]:
            return "correct_goal_no_contact"
        if not d["red_grasped"]:
            return "correct_goal_contact_no_grasp"
        if d["red_dropped_outside_bin"]:
            return "correct_goal_grasp_then_drop"
        return "correct_goal_execution_timeout" if reason == "timeout" else "other"
    if d["stalled"] or d["oscillating"]:
        return "oscillation_or_stall"
    if not any(
        d[key] for key in ("any_object_approached", "any_object_contacted", "any_object_grasped")
    ):
        return "no_meaningful_interaction"
    if not d["any_object_contacted"]:
        return "object_approached_no_contact"
    if not d["any_object_grasped"]:
        return "object_contact_no_grasp"
    return "object_grasp_no_destination"


def ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "count": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def reference_metrics(rows: list[dict[str, Any]], reference: str) -> dict[str, Any]:
    valid = [r for r in rows if r[reference] is not None]
    n = len(valid)
    selected = [r for r in valid if r["selected_goal"] == r[reference]]
    predicates = {
        "instructed_goal_approach_accuracy": lambda r: r["first_approached_goal"] == r[reference],
        "instructed_goal_bin_first_contact_accuracy": lambda r: (
            r["first_contact_goal"] == r[reference]
        ),
        "red_object_first_contact_accuracy": lambda r: r["first_contact_object"] == "red_cube",
        "red_object_first_grasp_accuracy": lambda r: r["first_grasp_object"] == "red_cube",
        "instructed_goal_placement_attempt_accuracy": lambda r: (
            r["placement_attempt_goal"] == r[reference]
        ),
        "full_success": lambda r: r["achieved_goal"] == r[reference],
    }
    funnel_masks = [True] * n
    funnel = {}
    for stage, predicate in (
        ("red_approach", lambda r: r["red_approached"]),
        ("red_contact", lambda r: r["red_contacted"]),
        ("red_grasp", lambda r: r["red_grasped"]),
        ("correct_destination_approach", predicates["instructed_goal_approach_accuracy"]),
        ("correct_placement_attempt", predicates["instructed_goal_placement_attempt_accuracy"]),
        ("full_success", predicates["full_success"]),
    ):
        funnel_masks = [
            keep and bool(predicate(r)) for keep, r in zip(funnel_masks, valid, strict=True)
        ]
        funnel[stage] = ratio(sum(funnel_masks), n)
    return {
        **{key: ratio(sum(bool(fn(r)) for r in valid), n) for key, fn in predicates.items()},
        "completion_after_correct_selection": ratio(
            sum(r["achieved_goal"] == r[reference] for r in selected), len(selected)
        ),
        "cumulative_funnel": funnel,
        "taxonomy": dict(
            Counter(
                classify(r, r[reference], r["achieved_goal"], r["termination_reason"])
                for r in valid
            )
        ),
    }


def pair_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[int, dict[str, dict[str, Any]]] = {}
    for r in rows:
        if r["prompt_goal"] not in GOALS:
            raise ValueError("paired switch requires two supplied goals")
        group = groups.setdefault(r["seed"], {})
        if r["prompt_goal"] in group:
            raise ValueError("duplicate goal in pair")
        group[r["prompt_goal"]] = r
    categories: Counter[str] = Counter()
    details = []
    raw_one = success = 0
    for seed, group in sorted(groups.items()):
        if set(group) != set(GOALS):
            raise ValueError("incomplete instruction pair")
        left, right = (group[g] for g in GOALS)
        a, b = left["selected_goal"], right["selected_goal"]
        count = int(a == GOALS[0]) + int(b == GOALS[1])
        raw_one += count == 1
        success += left["achieved_goal"] == GOALS[0] and right["achieved_goal"] == GOALS[1]
        category = (
            "both_corresponding"
            if count == 2
            else "same_target"
            if a is not None and a == b
            else "only_one_corresponding"
            if count == 1
            else "indeterminate"
            if a is None or b is None
            else "both_wrong_opposite"
        )
        categories[category] += 1
        details.append(
            {"seed": seed, "left_selected": a, "right_selected": b, "category": category}
        )
    return {
        "pairs": len(groups),
        "categories": dict(categories),
        "paired_instruction_switch_accuracy": ratio(categories["both_corresponding"], len(groups)),
        "paired_full_success": ratio(success, len(groups)),
        "one_corresponding_including_same_target": raw_one,
        "details": details,
    }
