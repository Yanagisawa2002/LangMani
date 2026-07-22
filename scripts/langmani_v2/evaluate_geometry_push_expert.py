"""Evaluate the frozen Phase 2B.3 expert on regression, development, or acceptance seeds."""

from __future__ import annotations

import argparse
import json
import subprocess
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from langmani.environments.push_specs import PUSH_OBJECT_IDS, TARGET_REGION_IDS, PushTaskSpec
from langmani.environments.push_to_region import ENV_ID
from langmani.experts.geometry_push_expert import GeometryConstrainedPushExpert
from langmani.experts.geometry_push_types import GeometryPushExpertConfig
from langmani.experts.push_types import PushExpertStatus
from langmani.v2.push_expert_recovery import (
    load_json,
    read_seed_bank,
    sha256_file,
    sha256_json,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--development-report", type=Path)
    parser.add_argument("--limit-episodes", type=int)
    return parser.parse_args()


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def _task_schedule(
    seeds: tuple[int, ...], *, standard_episodes: int
) -> tuple[tuple[int, PushTaskSpec], ...]:
    combinations = tuple(
        (object_id, region_id) for object_id in PUSH_OBJECT_IDS for region_id in TARGET_REGION_IDS
    )
    return tuple(
        (
            seed,
            PushTaskSpec(
                *combinations[index % len(combinations)],
                "standard" if index < standard_episodes else "hard",
            ),
        )
        for index, seed in enumerate(seeds)
    )


def _schedule(
    config: dict[str, Any], *, source_commit: str, development_report: Path | None
) -> tuple[tuple[int, PushTaskSpec], ...]:
    mode = config.get("mode")
    if mode == "failure_regression":
        validation = load_json(PROJECT_ROOT / str(config["replay_validation"]))
        cases = validation.get("cases")
        if not isinstance(cases, list):
            raise ValueError("replay validation cases are unavailable")
        values: list[tuple[int, PushTaskSpec]] = []
        for case in cases:
            if not isinstance(case, dict) or not isinstance(case.get("task"), dict):
                raise ValueError("replay case task is malformed")
            task = case["task"]
            values.append(
                (
                    int(case["seed"]),
                    PushTaskSpec(
                        str(task["object_id"]),  # type: ignore[arg-type]
                        str(task["target_region_id"]),  # type: ignore[arg-type]
                        str(task["difficulty"]),  # type: ignore[arg-type]
                    ),
                )
            )
        return tuple(values)
    if mode not in {"development", "acceptance"}:
        raise ValueError(f"unsupported evaluation mode {mode!r}")
    registry = load_json(PROJECT_ROOT / str(config["seed_registry"]))
    report = load_json(development_report) if development_report is not None else None
    seeds = read_seed_bank(
        registry,
        str(mode),
        development_report=report,
        expected_source_commit=source_commit,
    )
    expected = int(config["standard_episodes"]) + int(config["hard_episodes"])
    if len(seeds) != expected:
        raise ValueError(f"seed bank has {len(seeds)} entries, expected {expected}")
    return _task_schedule(seeds, standard_episodes=int(config["standard_episodes"]))


def _environment() -> Any:
    import gymnasium as gym

    import langmani.environments  # noqa: F401

    return gym.make(
        ENV_ID,
        num_envs=1,
        obs_mode="state_dict",
        reward_mode="none",
        control_mode="pd_joint_pos",
        render_mode=None,
        sim_backend="physx_cuda",
    )


def _load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("ledger records must be mappings")
        records.append(value)
    return records


def _append_ledger(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()


def _zero_tolerance_reason(record: dict[str, Any]) -> str | None:
    """Return the first gate-stopping reason without reclassifying task failure."""

    if "command_error" in record:
        return "simulator_error"
    result = record.get("result")
    if not isinstance(result, dict):
        return "malformed_result"
    if result.get("status") == PushExpertStatus.UNEXPECTED_EXCEPTION.value:
        return "simulator_error"
    if result.get("status") == PushExpertStatus.WRONG_OBJECT_INTERACTION.value:
        return "wrong_object_interaction"
    evaluation = result.get("final_environment_evaluation")
    if not isinstance(evaluation, dict):
        return "malformed_evaluation"
    for key in ("invalid_action", "action_out_of_bounds", "target_outside_workspace"):
        if evaluation.get(key) is True:
            return key
    if result.get("success") is True and evaluation.get("success") is not True:
        return "false_success"
    return None


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    source_commit = _source_commit()
    expert_path = PROJECT_ROOT / str(config["expert_config"])
    expert_payload = load_json(expert_path)
    expert_values = expert_payload.get("expert")
    if not isinstance(expert_values, dict):
        raise ValueError("expert config is malformed")
    expert_config = GeometryPushExpertConfig(**expert_values)  # type: ignore[arg-type]
    schedule = _schedule(
        config,
        source_commit=source_commit,
        development_report=args.development_report,
    )
    if args.limit_episodes is not None:
        if args.limit_episodes < 1:
            raise ValueError("--limit-episodes must be positive")
        schedule = schedule[: args.limit_episodes]
    existing = _load_ledger(args.ledger)
    if len(existing) > len(schedule):
        raise ValueError("ledger contains more records than the requested schedule")
    for index, record in enumerate(existing):
        seed, task = schedule[index]
        if record.get("seed") != seed or record.get("task") != task.to_dict():
            raise ValueError("ledger does not match the frozen schedule")

    stopped_early_reason = (
        next(
            (
                reason
                for record in existing
                if (reason := _zero_tolerance_reason(record)) is not None
            ),
            None,
        )
        if args.limit_episodes is None
        else None
    )
    for index, (seed, task) in enumerate(schedule[len(existing) :], start=len(existing)):
        if stopped_early_reason is not None:
            break
        environment: Any | None = None
        try:
            environment = _environment()
            environment.reset(seed=seed, options={"task_spec": task.to_dict()})
            expert = GeometryConstrainedPushExpert(environment, config=expert_config)
            try:
                result = expert.run()
            except Exception as error:  # noqa: BLE001 - preserved as simulator evidence
                traceback.print_exc()
                result = expert.unexpected_exception_result(error)
            record = {"seed": seed, "task": task.to_dict(), "result": result.to_dict()}
        except Exception as error:  # noqa: BLE001 - construction failure is evidence
            traceback.print_exc()
            record = {
                "seed": seed,
                "task": task.to_dict(),
                "command_error": {"type": type(error).__name__, "message": str(error)},
            }
        finally:
            if environment is not None:
                environment.close()
        _append_ledger(args.ledger, record)
        existing.append(record)
        status = record.get("result", {}).get("status", "command_error")
        print(f"[{index + 1}/{len(schedule)}] seed={seed} status={status}", flush=True)
        if args.limit_episodes is None:
            stopped_early_reason = _zero_tolerance_reason(record)

    results = [value["result"] for value in existing if isinstance(value.get("result"), dict)]
    command_errors = [value for value in existing if "command_error" in value]
    successes = sum(value.get("success") is True for value in results)
    evaluations = [value.get("final_environment_evaluation", {}) for value in results]
    simulator_errors = len(command_errors) + sum(
        value.get("status") == PushExpertStatus.UNEXPECTED_EXCEPTION.value for value in results
    )
    nonfinite_actions = sum(bool(value.get("invalid_action", False)) for value in evaluations)
    action_bound_violations = sum(
        bool(value.get("action_out_of_bounds", False)) for value in evaluations
    )
    workspace_violations = sum(
        bool(value.get("target_outside_workspace", False)) for value in evaluations
    )
    wrong_object_interactions = sum(
        value.get("status") == PushExpertStatus.WRONG_OBJECT_INTERACTION.value for value in results
    )
    false_successes = sum(
        value.get("success") is True and evaluation.get("success") is not True
        for value, evaluation in zip(results, evaluations, strict=True)
    )
    completed = len(existing)
    success_rate = successes / completed if completed else 0.0
    maximum_possible_successes = successes + (len(schedule) - completed)
    minimum_success_rate = float(config["minimum_success_rate"])
    complete = len(existing) == len(schedule) and len(results) == len(schedule)
    passed = bool(
        args.limit_episodes is None
        and complete
        and success_rate >= minimum_success_rate
        and simulator_errors == 0
        and nonfinite_actions == 0
        and action_bound_violations == 0
        and workspace_violations == 0
        and wrong_object_interactions == 0
        and false_successes == 0
    )
    report = {
        "schema_version": "langmani-v2-phase2b3-evaluation-v1",
        "mode": config["mode"],
        "source_commit": source_commit,
        "config_sha256": sha256_file(args.config),
        "expert_config_sha256": sha256_file(expert_path),
        "expert_semantic_digest": sha256_json(expert_config.to_dict()),
        "expected_episodes": len(schedule),
        "completed_episodes": completed,
        "successes": successes,
        "failures": completed - successes,
        "success_rate": success_rate,
        "maximum_possible_successes": maximum_possible_successes,
        "minimum_success_rate": minimum_success_rate,
        "status_counts": dict(
            sorted(Counter(str(value.get("status")) for value in results).items())
        ),
        "shape_counts": dict(
            sorted(Counter(str(value["task"]["target_object_id"]) for value in existing).items())
        ),
        "simulator_errors": simulator_errors,
        "nonfinite_actions": nonfinite_actions,
        "action_bound_violations": action_bound_violations,
        "workspace_violations": workspace_violations,
        "wrong_object_interactions": wrong_object_interactions,
        "false_successes": false_successes,
        "mean_episode_steps": (
            sum(int(value.get("total_environment_steps", 0)) for value in results) / len(results)
            if results
            else None
        ),
        "mean_contact_recoveries": (
            sum(int(value.get("contact_recoveries", 0)) for value in results) / len(results)
            if results
            else None
        ),
        "mean_replans": (
            sum(int(value.get("replans", 0)) for value in results) / len(results)
            if results
            else None
        ),
        "teleport_or_state_mutation_used": False,
        "success_flag_mutation_used": False,
        "official_collection_episodes": int(config["official_collection_episodes"]),
        "optimizer_steps": 0,
        "limited_probe": args.limit_episodes is not None,
        "stopped_early_reason": stopped_early_reason,
        "passed": passed,
        "results": results,
        "command_errors": command_errors,
    }
    write_json(args.output, report)
    print(
        json.dumps(
            {key: report[key] for key in ("successes", "failures", "success_rate", "passed")}
        )
    )
    return 0 if passed or args.limit_episodes is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
