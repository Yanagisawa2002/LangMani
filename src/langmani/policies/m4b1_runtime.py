"""Native read-only sensor wrapper, frozen-action replay and frozen-policy reevaluation."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from langmani.policies.m4a_data import digest, file_digest, numpy, read_json, write_json
from langmani.policies.m4a_evaluation import bool_value, make_environment, state_digest
from langmani.policies.m4b1_diagnostics import (
    PROTOCOL_VERSION,
    THRESHOLDS,
    classify,
    diagnose,
    pair_metrics,
    ratio,
    reference_metrics,
)
from langmani.policies.m4b_evaluation import (
    reset_audit,
    run_policy,
    score_state,
    write_csv,
    write_video,
)
from langmani.policies.m4b_protocol import GOALS, score_rollouts, task
from langmani.policies.m4b_training import load_checkpoint


class DiagnosticObserver:
    """Only env.step delegates mutations. Sampling is checked for public-state invariance."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.samples: list[dict[str, np.ndarray]] = []
        self.state_hashes: list[str] = []
        self.sample(initial=True)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def sample(self, *, initial: bool = False) -> None:
        base = self.env.unwrapped
        before = state_digest(base.get_state_dict())
        fingers = (base.agent.finger1_link, base.agent.finger2_link)
        forces = np.zeros((5, 2, 3), dtype=np.float32)
        grasped = np.zeros(3, dtype=bool)
        if not initial:
            for i, target in enumerate((*base.cubes, *base.bins)):
                for j, finger in enumerate(fingers):
                    forces[i, j] = numpy(base.scene.get_pairwise_contact_forces(finger, target))[0]
            grasped = np.array([bool_value(base.agent.is_grasping(c)) for c in base.cubes])
        row = {
            "tcp": numpy(base.agent.tcp.pose.p)[0].copy(),
            "cubes": np.stack([numpy(c.pose.p)[0] for c in base.cubes]),
            "bins": numpy(base._bin_floor_centers())[0].copy(),
            "finger_forces": forces,
            "grasped": grasped,
        }
        after = state_digest(base.get_state_dict())
        if before != after:
            raise RuntimeError("diagnostic sampling mutated simulator state")
        self.samples.append(row)
        self.state_hashes.append(before)

    def step(self, action: Any) -> Any:
        result = self.env.step(action)
        self.sample()
        return result

    def arrays(self) -> dict[str, np.ndarray]:
        return {key: np.stack([s[key] for s in self.samples]) for key in self.samples[0]}


def assert_source(source: Path, row: dict[str, Any]) -> np.ndarray:
    artifact = source / "rollouts" / row["rollout_id"]
    for suffix, key in ((".npz", "action_file_sha256"), (".mp4", "video_sha256")):
        if file_digest(artifact.with_suffix(suffix)) != row[key]:
            raise ValueError(f"frozen {key} differs")
    with np.load(artifact.with_suffix(".npz"), allow_pickle=False) as archive:
        actions = archive["actions"]
    if actions.shape != (row["length"], 8) or state_digest(actions) != row["actions_sha256"]:
        raise ValueError("frozen applied actions differ")
    return actions


def reset_for_row(env: Any, row: dict[str, Any]) -> Any:
    observation, _ = env.reset(seed=row["seed"], options={"task_spec": task(GOALS[0]).to_dict()})
    if any(value != row[key] for key, value in reset_audit(env, observation).items()):
        raise RuntimeError("replay reset differs from original state/RGB/qpos")
    return observation


def replay_one(
    source: Path, row: dict[str, Any], output: Path, backend: str, *, instrumented: bool = True
) -> dict[str, Any]:
    from langmani.datasets.observation_reconstruction import extract_base_camera_rgb

    actions = assert_source(source, row)
    env = make_environment(backend)
    try:
        observation = reset_for_row(env, row)
        observed = DiagnosticObserver(env) if instrumented else None
        hashes = [state_digest(env.unwrapped.get_state_dict())]
        frames = [extract_base_camera_rgb(observation).copy()]
        reason, achieved = "timeout", None
        for t, action in enumerate(actions, start=1):
            observation, _, terminated, truncated, _ = (observed or env).step(action)
            hashes.append(state_digest(env.unwrapped.get_state_dict()))
            frames.append(extract_base_camera_rgb(observation).copy())
            flags = score_state(env)
            achieved = next((g for g in GOALS if flags[g]["success"]), None)
            reason = (
                "goal_reached"
                if achieved
                else "target_off_table"
                if any(f["target_off_table"] for f in flags.values())
                else "timeout"
                if bool_value(truncated)
                else "terminated"
                if bool_value(terminated)
                else "running"
            )
            if reason != "running" and t != len(actions):
                raise RuntimeError(f"replay terminates before frozen trajectory at step {t}")
        if reason == "running":
            reason = "timeout" if len(actions) == 200 else "incomplete"
        if (
            score_state(env) != row["final_flags"]
            or reason != row["termination_reason"]
            or achieved != row["achieved_goal"]
        ):
            raise RuntimeError("replay final score/termination differs")
        output.parent.mkdir(parents=True, exist_ok=True)
        write_video(output.with_suffix(".mp4"), frames)
        if file_digest(output.with_suffix(".mp4")) != row["video_sha256"]:
            raise RuntimeError("replay rendered video differs from immutable source")
        result = {
            "rollout_id": row["rollout_id"],
            "passed": True,
            "state_sequence_sha256": digest(hashes),
            "state_hashes": hashes,
            "video_sha256": row["video_sha256"],
            "actions_sha256": row["actions_sha256"],
            "original_result_exact": True,
        }
        if observed is not None:
            if hashes != observed.state_hashes:
                raise RuntimeError("observer state sequence differs")
            arrays = observed.arrays()
            np.savez_compressed(output.with_suffix(".npz"), **arrays)
            result.update(
                diagnostics=diagnose(arrays), trace_sha256=file_digest(output.with_suffix(".npz"))
            )
        write_json(output.with_suffix(".json"), result)
        return result
    finally:
        env.close()


def online_check(
    source: Path, row: dict[str, Any], checkpoint: Path, output: Path, backend: str
) -> dict[str, Any]:
    import torch

    expected = read_json(source / "config.json")["checkpoint_sha256"]
    if file_digest(checkpoint / "model.safetensors") != expected:
        raise ValueError("frozen checkpoint model hash differs")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    policy, pre, post = load_checkpoint(checkpoint, "cuda")
    env = make_environment(backend)
    try:
        observation = reset_for_row(env, row)
        wrapper = DiagnosticObserver(env)
        output.parent.mkdir(parents=True, exist_ok=True)
        result = run_policy(
            wrapper, observation, policy, pre, post, row["instruction"], row["seed"], output
        )
        for key in (
            "actions_sha256",
            "video_sha256",
            "length",
            "final_flags",
            "termination_reason",
        ):
            if result[key] != row[key]:
                raise RuntimeError(f"instrumented frozen checkpoint reevaluation differs: {key}")
        np.savez_compressed(output.with_name(output.name + "-trace.npz"), **wrapper.arrays())
        result.update(
            passed=True,
            diagnostics=diagnose(wrapper.arrays()),
            checkpoint_sha256=expected,
            rollout_id=row["rollout_id"],
        )
        write_json(output.with_suffix(".json"), result)
        return result
    finally:
        env.close()


def representatives(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda r: (r["seed"], r["prompt_id"]))
    tests = {
        "successful_correct": lambda r: (
            r["prompt_id"].startswith("canonical") and r["achieved_goal"] == r["prompt_goal"]
        ),
        "wrong_goal": lambda r: (
            r["prompt_goal"] is not None and r["selected_goal"] not in (None, r["prompt_goal"])
        ),
        "correct_goal_execution_failure": lambda r: (
            r["prompt_goal"] is not None
            and r["selected_goal"] == r["prompt_goal"]
            and r["achieved_goal"] != r["prompt_goal"]
        ),
        "timeout_or_stall": lambda r: r["termination_reason"] == "timeout",
    }
    result: dict[str, Any] = {}
    for category, test in tests.items():
        found = next((r for r in ordered if test(r)), None)
        result[category] = (
            {key: found[key] for key in ("rollout_id", "video", "seed", "instruction")}
            if found
            else None
        )
    canonical = [r for r in ordered if r["prompt_id"].startswith("canonical")]
    pairs = pair_metrics(canonical)
    seed = next(
        (
            r["seed"]
            for r in pairs["details"]
            if r["category"] == "both_corresponding"
            and all(
                v["achieved_goal"] == v["prompt_goal"] for v in canonical if v["seed"] == r["seed"]
            )
        ),
        None,
    )
    result["successful_instruction_switch"] = (
        [
            {key: r[key] for key in ("rollout_id", "video", "seed", "instruction")}
            for r in canonical
            if r["seed"] == seed
        ]
        if seed is not None
        else None
    )
    return result


def report(rows: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    from collections import Counter

    scored = score_rollouts(rows)
    by_id = {r["rollout_id"]: r for r in rows}
    for row in scored:
        physical = by_id[row["rollout_id"]]
        row.update({key: value for key, value in physical.items() if key not in row})
        row.update(
            intended_goal=row["prompt_goal"],
            requested_goal=row["semantic_goal"],
            target_identity="red_cube",
            target_side=task(row["semantic_goal"]).target_bin_id,
            timeout=row["termination_reason"] == "timeout",
            requested_failure_category=classify(
                row, row["semantic_goal"], row["achieved_goal"], row["termination_reason"]
            ),
            supplied_failure_category=classify(
                row, row["prompt_goal"], row["achieved_goal"], row["termination_reason"]
            ),
        )
    write_csv(output / "per_episode_diagnostics.csv", scored)
    write_json(output / "per_episode_diagnostics.json", {"episodes": scored})
    conditions = {}
    for condition in ("correct", "swapped", "blank", "paraphrase"):
        subset = [r for r in scored if r["condition"] == condition]
        conditions[condition] = {
            "scored_rows": len(subset),
            "requested_goal": reference_metrics(subset, "semantic_goal"),
            "supplied_goal": reference_metrics(subset, "prompt_goal"),
            "timeouts": ratio(sum(r["timeout"] for r in subset), len(subset)),
        }
    timeouts = [r for r in rows if r["termination_reason"] == "timeout"]
    taxonomy = {
        "physical_rollouts": len(rows),
        "physical_timeouts": len(timeouts),
        "reference": "supplied goal; blank null receives neutral unprompted categories",
        "timeout_categories": dict(
            Counter(
                classify(r, r["prompt_goal"], r["achieved_goal"], r["termination_reason"])
                for r in timeouts
            )
        ),
        "conditions": {c: v["supplied_goal"]["taxonomy"] for c, v in conditions.items()},
    }
    result = {
        "schema_version": PROTOCOL_VERSION,
        "thresholds": THRESHOLDS,
        "physical_rollouts": len(rows),
        "scored_rows": len(scored),
        "conditions": conditions,
        "pairs": {
            kind: pair_metrics([r for r in rows if r["prompt_id"].startswith(kind)])
            for kind in ("canonical", "paraphrase")
        },
        "physical_timeout_taxonomy": taxonomy,
        "unavailable": {
            "destination_grasp_accuracy": "shared red cube has no graspable left/right identity"
        },
    }
    write_json(output / "failure_taxonomy.json", taxonomy)
    write_json(output / "metrics.json", result)
    write_json(output / "representative_videos.json", representatives(rows))
    return result


def run(source: Path, output: Path, checkpoint: Path | None, *, smoke: bool = False) -> None:
    if output.exists():
        raise FileExistsError("diagnostic output must be new")
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    if status.strip():
        raise ValueError("native evidence requires a clean implementation commit")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    config = read_json(source / "config.json")
    originals = [read_json(path) for path in sorted((source / "rollouts").glob("*.json"))]
    originals.sort(key=lambda r: (r["seed"], r["prompt_id"]))
    # Existing score function validates exact five-input scene structure, without changing it.
    if len(originals) != 100 or len(score_rollouts(originals)) != 160:
        raise ValueError("complete frozen M4B 100-rollout source required")
    inventory = {
        str(path.relative_to(source)): file_digest(path)
        for path in sorted(source.rglob("*"))
        if path.is_file()
    }
    write_json(
        output / "config.json",
        {
            "source": str(source),
            "source_inventory": inventory,
            "git_commit": commit,
            "source_config": config,
            "protocol": PROTOCOL_VERSION,
            "thresholds": THRESHOLDS,
            "smoke": smoke,
        },
    )
    selected = originals
    if smoke:
        # One numerically first successful scene, five prompt IDs; selection precedes diagnostics.
        seed = min(r["seed"] for r in originals if r["achieved_goal"] is not None)
        selected = [r for r in originals if r["seed"] == seed]
    rows = []
    started = time.perf_counter()
    for row in selected:
        artifact = output / "rollouts" / row["rollout_id"]
        receipt = replay_one(source, row, artifact, config["sim_backend"])
        diagnostic = receipt["diagnostics"]
        if smoke:
            control = replay_one(
                source,
                row,
                output / "controls" / row["rollout_id"],
                config["sim_backend"],
                instrumented=False,
            )
            if control["state_hashes"] != receipt["state_hashes"]:
                raise RuntimeError("instrumentation changes per-step simulation state")
        rows.append({**row, **diagnostic, "trace_sha256": receipt["trace_sha256"]})
        print(
            json.dumps(
                {
                    "rollout_id": row["rollout_id"],
                    "completed": len(rows),
                    "total": len(selected),
                    "exact_replay": True,
                }
            ),
            flush=True,
        )
    if checkpoint is not None:
        row = next(r for r in selected if r["achieved_goal"] is not None)
        online_check(
            source, row, checkpoint, output / "online" / row["rollout_id"], config["sim_backend"]
        )
    if not smoke:
        report(rows, output)
    # Source must still be byte-identical after all replay/reevaluation work.
    if any(file_digest(source / path) != value for path, value in inventory.items()):
        raise RuntimeError("immutable M4B source changed")
    write_json(
        output / "validation.json",
        {
            "passed": True,
            "smoke": smoke,
            "replayed_rollouts": len(rows),
            "physical_actions": sum(r["length"] for r in rows),
            "all_initial_inputs_and_original_results_exact": True,
            "all_videos_byte_identical": True,
            "all_sensor_reads_state_invariant": True,
            "uninstrumented_controls": len(rows) if smoke else 0,
            "online_checkpoint_exact": checkpoint is not None,
            "seconds": time.perf_counter() - started,
            "git_commit": commit,
        },
    )
