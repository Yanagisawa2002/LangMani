"""Frozen two-destination language protocol and paired scoring, without model dependencies."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from langmani.environments.specs import TaskSpec, canonical_instruction
from langmani.policies.m4a_data import digest

GOALS = ("red_cube:left_bin", "red_cube:right_bin")
CONDITIONS = ("correct", "swapped", "blank", "paraphrase")
BASE_MODEL = "lerobot/smolvla_base"
BASE_REVISION = "d9f33c94a60fb382c90dea2164c96845bd955e28"
BACKBONE_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
BACKBONE_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"


def task(goal: str) -> TaskSpec:
    if goal not in GOALS:
        raise ValueError(f"unsupported M4B semantic goal: {goal}")
    obj, dest = goal.split(":")
    return TaskSpec(obj, dest, "canonical_v0")


def prompts(index: int) -> list[dict[str, Any]]:
    rows = []
    for kind in ("canonical", "paraphrase"):
        for goal in GOALS:
            side = task(goal).target_bin_id.split("_")[0]
            variants = (
                f"Place the red cube inside the {side} bin.",
                f"Move the red block into the {side} container.",
            )
            rows.append(
                {
                    "prompt_id": f"{kind}_{side}",
                    "prompt_goal": goal,
                    "instruction": canonical_instruction(task(goal))
                    if kind == "canonical"
                    else variants[index % 2],
                }
            )
    rows.insert(2, {"prompt_id": "blank", "prompt_goal": None, "instruction": ""})
    return rows


def make_schedule(
    source_seeds: list[int], m4a_seeds: list[int], count: int = 20, seed: int = 43000
) -> dict[str, Any]:
    if count < 1 or seed < 0 or len(m4a_seeds) != 20 or len(set(m4a_seeds)) != 20:
        raise ValueError("positive scene count and the exact 20 M4A evaluation seeds are required")
    excluded = set(source_seeds) | set(m4a_seeds)
    rng = np.random.default_rng(seed)
    seeds: list[int] = []
    while len(seeds) < count:
        candidate = int(rng.integers(0, 2**31 - 1))
        if candidate not in excluded and candidate not in seeds:
            seeds.append(candidate)
    payload = {
        "schema_version": "langmani-m4b-schedule-v1",
        "seed": seed,
        "excluded_source_seeds": sorted(set(source_seeds)),
        "excluded_m4a_seeds": m4a_seeds,
        "scenes": [{"seed": value, "prompts": prompts(i)} for i, value in enumerate(seeds)],
    }
    return {**payload, "schedule_sha256": digest(payload)}


def score_rollouts(rollouts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reuse identical physical inputs explicitly; these are not independent trials."""
    by_seed: dict[int, dict[str, Any]] = {}
    for row in rollouts:
        group = by_seed.setdefault(row["seed"], {})
        if row["prompt_id"] in group:
            raise ValueError("duplicate physical rollout")
        group[row["prompt_id"]] = row
    scored = []
    for group in by_seed.values():
        if set(group) != {p["prompt_id"] for p in prompts(0)}:
            raise ValueError("incomplete five-input scene")
        for i, goal in enumerate(GOALS):
            side = task(goal).target_bin_id.split("_")[0]
            other = task(GOALS[1 - i]).target_bin_id.split("_")[0]
            for condition, key in zip(
                CONDITIONS,
                (f"canonical_{side}", f"canonical_{other}", "blank", f"paraphrase_{side}"),
                strict=True,
            ):
                row = group[key]
                success = row["achieved_goal"] == goal
                scored.append(
                    {
                        **{
                            k: row[k]
                            for k in (
                                "rollout_id",
                                "seed",
                                "instruction",
                                "prompt_goal",
                                "achieved_goal",
                                "length",
                                "termination_reason",
                                "video",
                            )
                        },
                        "condition": condition,
                        "semantic_goal": goal,
                        "success": success,
                        "prompt_goal_success": row["prompt_goal"] is not None
                        and row["achieved_goal"] == row["prompt_goal"],
                        "failure_category": None
                        if success
                        else "wrong_goal"
                        if row["achieved_goal"] is not None
                        else row["termination_reason"],
                    }
                )
    return scored


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("empty metric denominator")
    successful = [r["length"] for r in rows if r["success"]]
    return {
        "scored_rows": len(rows),
        "successes": len(successful),
        "success_rate": len(successful) / len(rows),
        "mean_successful_episode_length": float(np.mean(successful)) if successful else None,
        "timeout_count": sum(r["termination_reason"] == "timeout" for r in rows),
        "termination_reason_counts": dict(Counter(r["termination_reason"] for r in rows)),
        "failure_category_counts": dict(
            Counter(r["failure_category"] for r in rows if not r["success"])
        ),
        "achieved_goal_counts": dict(Counter(r["achieved_goal"] or "none" for r in rows)),
        "prompt_goal_success_rate": sum(r["prompt_goal_success"] for r in rows) / len(rows),
    }


def metrics(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    rows = score_rollouts(rollouts)
    conditions = {c: summarize([r for r in rows if r["condition"] == c]) for c in CONDITIONS}
    groups = {
        seed: {r["prompt_id"]: r for r in rollouts if r["seed"] == seed}
        for seed in {r["seed"] for r in rollouts}
    }
    canonical_follow = paraphrase_follow = goal_changes = action_changes = agreement = 0
    paired = True
    for group in groups.values():
        for key in (
            "initial_state_sha256",
            "initial_rgb_sha256",
            "initial_qpos_sha256",
            "noise_schedule_sha256",
        ):
            values = {row[key] for row in group.values()}
            paired &= len(values) == 1 and None not in values
        left, right = group["canonical_left"], group["canonical_right"]
        canonical_follow += left["achieved_goal"] == GOALS[0] and right["achieved_goal"] == GOALS[1]
        paraphrase_follow += (
            group["paraphrase_left"]["achieved_goal"] == GOALS[0]
            and group["paraphrase_right"]["achieved_goal"] == GOALS[1]
        )
        goal_changes += (
            left["achieved_goal"] is not None
            and right["achieved_goal"] is not None
            and left["achieved_goal"] != right["achieved_goal"]
        )
        action_changes += left["actions_sha256"] != right["actions_sha256"]
        for side in ("left", "right"):
            achieved = group[f"canonical_{side}"]["achieved_goal"]
            agreement += (
                achieved is not None and achieved == group[f"paraphrase_{side}"]["achieved_goal"]
            )
    errors = sum(r.get("error") is not None for r in rollouts)
    valid = (
        paired and not errors and all(r["length"] > 0 and r.get("video_sha256") for r in rollouts)
    )
    n = len(groups)
    return {
        "schema_version": "langmani-m4b-metrics-v1",
        "passed": bool(valid),
        "closed_loop_evaluated": bool(valid),
        "physical_policy_rollouts": len(rollouts),
        "scored_rows": len(rows),
        "scene_pairs": n,
        "initial_observations_and_noise_paired": paired,
        "infrastructure_errors": errors,
        "conditions": conditions,
        "per_semantic_goal": {
            g: {
                c: summarize([r for r in rows if r["semantic_goal"] == g and r["condition"] == c])
                for c in CONDITIONS
            }
            for g in GOALS
        },
        "canonical_pair_goal_switch_successes": canonical_follow,
        "canonical_pair_goal_switch_success_rate": canonical_follow / n if valid else None,
        "paraphrase_pair_goal_switch_success_rate": paraphrase_follow / n if valid else None,
        "achieved_goal_change_rate": goal_changes / n if valid else None,
        "action_trajectory_difference_rate": action_changes / n if valid else None,
        "canonical_paraphrase_nonnull_goal_agreement_rate": agreement / (2 * n) if valid else None,
        "agreement_denominator": "all two goal prompts per scene; two failures are not agreement",
        "correct_minus_swapped_success_rate": conditions["correct"]["success_rate"]
        - conditions["swapped"]["success_rate"]
        if valid
        else None,
        "correct_minus_blank_success_rate": conditions["correct"]["success_rate"]
        - conditions["blank"]["success_rate"]
        if valid
        else None,
        "physical_timeout_count": sum(r["termination_reason"] == "timeout" for r in rollouts),
        "physical_environment_steps": sum(r["length"] for r in rollouts),
        "gripper_saturation_count": sum(r.get("gripper_saturation_count", 0) for r in rollouts),
        "gripper_max_overshoot": max(r.get("gripper_max_overshoot", 0.0) for r in rollouts),
    }
