"""Common language/scene interventions with the completed M4B.1 observer and original policy loop."""

from __future__ import annotations

import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from langmani.policies.m4a_data import digest, file_digest, read_json, write_json
from langmani.policies.m4a_evaluation import make_environment
from langmani.policies.m4b1_diagnostics import (
    PROTOCOL_VERSION,
    classify,
    diagnose,
    inside_bins,
    pair_metrics,
    ratio,
    reference_metrics,
)
from langmani.policies.m4b1_runtime import DiagnosticObserver
from langmani.policies.m4b_evaluation import reset_audit, run_policy, write_csv
from langmani.policies.m4b_protocol import GOALS, task
from langmani.policies.m4b_training import load_checkpoint

CONDITIONS = ("seen", "lexical", "syntactic", "natural", "swapped", "blank")


def diagnostics(trace: dict[str, np.ndarray]) -> dict[str, Any]:
    if len(trace["tcp"]) > 1:
        return diagnose(trace)
    # A policy may fail action validation before any step. Do not invent an extra sample.
    keys = (
        "first_approached_object",
        "first_approached_goal",
        "first_contact_object",
        "first_contact_goal",
        "first_grasp_object",
        "placement_attempt_goal",
    )
    inside = inside_bins(trace["cubes"][:, 0], trace["bins"])[0]
    choices = [g for g, valid in zip(GOALS, inside, strict=True) if valid]
    return {
        "schema_version": PROTOCOL_VERSION,
        **dict.fromkeys(keys),
        "events": {k: {"value": None, "step": None, "ambiguous": False} for k in keys},
        "selected_goal": None,
        "final_goal": choices[0] if len(choices) == 1 else None,
        **dict.fromkeys(
            (
                "red_approached",
                "red_contacted",
                "red_grasped",
                "red_lifted_after_grasp",
                "red_dropped_outside_bin",
                "any_object_approached",
                "any_object_contacted",
                "any_object_grasped",
                "stalled",
                "oscillating",
            ),
            False,
        ),
        "tail_tcp_path_m": None,
        "tail_tcp_net_m": None,
    }


def failure(row: dict[str, Any], reference: str | None) -> str:
    if row["termination_reason"] in ("invalid_action", "action_out_of_bounds"):
        return row["termination_reason"]
    return classify(row, reference, row["achieved_goal"], row["termination_reason"])


def validate_rollout(row: dict[str, Any]) -> None:
    steps, reason = row["length"], row["termination_reason"]
    if (
        not isinstance(steps, int)
        or not 0 <= steps <= 200
        or row["video_frames"] != steps + 1
        or row["diagnostic_samples"] != steps + 1
        or reason
        not in (
            "goal_reached",
            "timeout",
            "target_off_table",
            "terminated",
            "invalid_action",
            "action_out_of_bounds",
        )
        or (reason == "timeout" and steps != 200)
        or (steps == 0 and reason not in ("invalid_action", "action_out_of_bounds"))
    ):
        raise ValueError("incoherent physical trajectory length/termination/frame counts")
    achieved = [g for g in GOALS if row["final_flags"][g]["success"]]
    if (
        len(achieved) > 1
        or row["achieved_goal"] != (achieved[0] if achieved else None)
        or (reason == "goal_reached") != bool(achieved)
        or row.get("error") is not None
    ):
        raise ValueError("incoherent final success flags or infrastructure failure")


def reference_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    result = reference_metrics(rows, key)
    result["taxonomy"] = dict(Counter(failure(r, r[key]) for r in rows if r[key] is not None))
    return result


def scored_rows(rollouts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in rollouts:
        references = [(row["condition"], row["prompt_goal"])]
        if row["condition"] == "seen":
            references.append(("swapped", GOALS[1 - GOALS.index(row["prompt_goal"])]))
        elif row["condition"] == "blank":
            references = [("blank", g) for g in GOALS]
        for condition, goal in references:
            rows.append(
                {
                    **row,
                    "condition": condition,
                    "semantic_goal": goal,
                    "requested_goal": goal,
                    "intended_goal": row["prompt_goal"],
                    "target_identity": "red_cube",
                    "target_side": task(goal).target_bin_id,
                    "success": row["achieved_goal"] == goal,
                    "prompt_goal_success": row["prompt_goal"] is not None
                    and row["achieved_goal"] == row["prompt_goal"],
                    "timeout": row["termination_reason"] == "timeout",
                    "requested_failure_category": failure(row, goal),
                    "supplied_failure_category": failure(row, row["prompt_goal"]),
                }
            )
    return rows


def metrics(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    if not rollouts or len({r["rollout_id"] for r in rollouts}) != len(rollouts):
        raise ValueError("empty or duplicated physical trajectories")
    for row in rollouts:
        validate_rollout(row)
    rows = scored_rows(rollouts)
    scenes = {r["seed"] for r in rollouts}
    expected_ids = {f"{kind}_{side}" for kind in CONDITIONS[:4] for side in ("left", "right")} | {
        "blank"
    }
    for seed in scenes:
        group = [r for r in rollouts if r["seed"] == seed]
        if len(group) != 9 or {r["prompt_id"] for r in group} != expected_ids:
            raise ValueError("incomplete or duplicated nine-prompt scene")
        for field in (
            "initial_state_sha256",
            "initial_rgb_sha256",
            "initial_qpos_sha256",
            "noise_schedule_sha256",
        ):
            if len({r[field] for r in group}) != 1:
                raise ValueError(f"language interventions no longer paired: {field}")
    conditions = {}
    for condition in CONDITIONS:
        subset = [r for r in rows if r["condition"] == condition]
        unique = {r["rollout_id"]: r for r in subset}
        requested = reference_summary(subset, "semantic_goal")
        supplied = reference_summary(subset, "prompt_goal")
        conditions[condition] = {
            "scored_rows": len(subset),
            "physical_rollouts": len(unique),
            "requested_goal": requested,
            "supplied_goal": supplied,
            "timeouts": ratio(sum(r["timeout"] for r in subset), len(subset)),
            "physical_supplied_goal_taxonomy": dict(
                Counter(failure(r, r["prompt_goal"]) for r in unique.values())
            ),
            "paired_instruction_switch": pair_metrics(list(unique.values()))
            if condition != "blank"
            else None,
            "per_requested_goal": {
                g: reference_summary(
                    [r for r in subset if r["semantic_goal"] == g], "semantic_goal"
                )
                for g in GOALS
            },
        }
    return {
        "schema_version": "langmani-m4c-metrics-v1",
        "conditions": conditions,
        "physical_rollouts": len(rollouts),
        "scored_rows": len(rows),
        "scene_pairs": len(scenes),
        "physical_steps": sum(r["length"] for r in rollouts),
        "physical_timeouts": sum(r["termination_reason"] == "timeout" for r in rollouts),
        "physical_termination_counts": dict(Counter(r["termination_reason"] for r in rollouts)),
        "initial_inputs_and_noise_paired": True,
        "infrastructure_errors": sum(r.get("error") is not None for r in rollouts),
    }


def evaluate(
    *,
    checkpoint: Path,
    output: Path,
    schedule: dict[str, Any],
    gate: Path,
    provenance: dict[str, Any],
    smoke: bool = False,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("preserve prior evaluation")
    experts = read_json(gate / "episodes.json")["episodes"]
    gate_report = read_json(gate / "gate.json")
    if (
        not gate_report["passed"]
        or digest(experts) != gate_report["episodes_sha256"]
        or gate_report["schedule_sha256"] != schedule["source_m4b_schedule_sha256"]
    ):
        raise ValueError("frozen paired native reference differs")
    if {r["seed"] for r in experts} != {r["seed"] for r in schedule["scenes"]}:
        raise ValueError("M4C scenes differ from frozen native reference")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    policy, pre, post = load_checkpoint(checkpoint, "cuda")
    config = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_digest(checkpoint / "model.safetensors"),
        "schedule": schedule,
        "native_gate": gate_report,
        "provenance": provenance,
        "smoke": smoke,
        "sim_backend": gate_report["sim_backend"],
        "max_episode_steps": 200,
        "observer_protocol": PROTOCOL_VERSION,
        "termination": "unchanged M4B first either-goal success/native failure/action failure/200 steps",
    }
    write_json(output / "config.json", config)
    write_json(output / "expert_reference.json", {"episodes": experts})
    (output / "rollouts").mkdir()
    rows = []
    started = time.perf_counter()
    for scene in schedule["scenes"][:1] if smoke else schedule["scenes"]:
        seed = scene["seed"]
        reference = next(r for r in experts if r["seed"] == seed)
        for prompt in scene["prompts"]:
            identity = f"seed-{seed}-{prompt['prompt_id']}"
            artifact = output / "rollouts" / identity
            row = {
                "rollout_id": identity,
                "seed": seed,
                **prompt,
                "video": f"rollouts/{identity}.mp4",
                "error": None,
            }
            env = None
            try:
                env = make_environment(gate_report["sim_backend"])
                obs, _ = env.reset(seed=seed, options={"task_spec": task(GOALS[0]).to_dict()})
                audit = reset_audit(env, obs)
                if any(audit[k] != reference[k] for k in audit):
                    raise RuntimeError("reset differs from frozen M4B paired reference")
                row.update(audit)
                observer = DiagnosticObserver(env)
                row.update(
                    run_policy(
                        observer, obs, policy, pre, post, prompt["instruction"], seed, artifact
                    )
                )
                arrays = observer.arrays()
                trace = artifact.with_name(identity + "-trace.npz")
                np.savez_compressed(trace, **arrays)
                row.update(diagnostics(arrays))
                row.update(
                    trace_sha256=file_digest(trace),
                    state_sequence_sha256=digest(observer.state_hashes),
                    diagnostic_samples=len(arrays["tcp"]),
                )
                write_json(
                    artifact.with_name(identity + "-states.json"), {"hashes": observer.state_hashes}
                )
            except Exception as error:
                traceback.print_exc()
                row.update(
                    error=f"{type(error).__name__}: {error}",
                    termination_reason="infrastructure_error",
                )
                write_json(output / "failure.json", row)
                raise
            finally:
                if env is not None:
                    env.close()
            write_json(artifact.with_suffix(".json"), row)
            rows.append(row)
            print(
                f"{len(rows)}/{9 if smoke else 180} {identity} {row['termination_reason']}",
                flush=True,
            )
    result = metrics(rows)
    result.update(
        passed=result["infrastructure_errors"] == 0,
        closed_loop_evaluated=not smoke and len(rows) == 180,
        evaluation_seconds=time.perf_counter() - started,
        checkpoint_sha256=config["checkpoint_sha256"],
        schedule_sha256=schedule["schedule_sha256"],
        expert_reference={
            "successes": sum(r["success"] for r in experts),
            "episodes": len(experts),
        },
    )
    scored = scored_rows(rows)
    write_csv(output / "episodes.csv", scored)
    write_csv(output / "per_episode_diagnostics.csv", scored)
    write_json(output / "episodes.json", {"episodes": scored})
    write_json(output / "metrics.json", result)
    write_json(
        output / "failure_taxonomy.json",
        {c: v["physical_supplied_goal_taxonomy"] for c, v in result["conditions"].items()},
    )
    representatives = {}
    for row in sorted(rows, key=lambda r: (r["seed"], r["prompt_id"])):
        category = f"{row['condition']}/{failure(row, row['prompt_goal'])}"
        representatives.setdefault(
            category,
            {
                k: row[k]
                for k in (
                    "rollout_id",
                    "seed",
                    "instruction",
                    "prompt_goal",
                    "achieved_goal",
                    "video",
                )
            },
        )
    write_json(output / "representative_videos.json", representatives)
    return result
