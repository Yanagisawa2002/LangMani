"""Run paired Candidate E expert or hold-position sanity on the sealed Phase 2C schedule."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any

from langmani.environments.push_specs import PushTaskSpec
from langmani.environments.push_to_region import ENV_ID
from langmani.experts.push import PushToRegionExpert
from langmani.experts.push_types import PushExpertConfig, PushExpertStatus
from langmani.policies.act_action_bounds import ActionBoundConfig, ActionBoundMode
from langmani.v2.evaluator import Phase2CPolicyObservationExtractor, UnifiedPolicyEvaluator
from langmani.v2.phase2c import Phase2CContractError, read_json_object, sha256_file, write_json_once
from langmani.v2.phase2c_baselines import HoldPositionPolicyAdapter
from langmani.v2.phase2c_evaluation import classify_failure, summarize_sealed_results
from langmani.v2.taxonomy import EvaluationTask, PushTaskInstanceSpec

ACCEPTED_CANDIDATE_E_COMMIT = "59ca88e9f0514187252a6286ab1b8e06c4318fb4"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=("candidate_e", "hold_position"), required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sim-backend", choices=("physx_cpu", "physx_cuda"), default="physx_cuda")
    parser.add_argument("--limit-episodes", type=int)
    return parser.parse_args(argv)


def _environment(*, sim_backend: str, rgb: bool) -> Any:
    import gymnasium as gym

    import langmani.environments  # noqa: F401 - registers ENV_ID

    return gym.make(
        ENV_ID,
        num_envs=1,
        obs_mode="rgb" if rgb else "state_dict",
        reward_mode="none",
        control_mode="pd_joint_pos",
        render_mode=None,
        sim_backend=sim_backend,
    )


def _schedule(path: Path, limit: int | None) -> list[dict[str, Any]]:
    value = read_json_object(path, "Phase 2C evaluation schedule")
    selected = value.get("sealed")
    if not isinstance(selected, list) or not all(isinstance(item, dict) for item in selected):
        raise Phase2CContractError("sealed baseline schedule is malformed")
    if limit is not None:
        if limit < 1:
            raise Phase2CContractError("limit_episodes must be positive")
        selected = selected[:limit]
    return selected


def _expert_record(item: dict[str, Any], *, sim_backend: str) -> dict[str, object]:
    environment: Any | None = None
    expert: PushToRegionExpert | None = None
    try:
        task_value = item.get("task_spec")
        if not isinstance(task_value, dict):
            raise Phase2CContractError("scheduled task_spec is malformed")
        task = PushTaskSpec.from_mapping(task_value)
        environment = _environment(sim_backend=sim_backend, rgb=False)
        environment.reset(seed=int(item["seed"]), options={"task_spec": task.to_dict()})
        expert = PushToRegionExpert(environment, config=PushExpertConfig())
        result = expert.run()
        final = dict(result.final_environment_evaluation)
        record: dict[str, object] = {
            **item,
            "baseline": "candidate_e",
            "expert_source_commit": ACCEPTED_CANDIDATE_E_COMMIT,
            "success": result.success,
            "outcome": result.status.value,
            "timeout": result.status is PushExpertStatus.TIMEOUT,
            "failure_reason": result.exception_message,
            "wrong_object_interaction": result.status is PushExpertStatus.WRONG_OBJECT_INTERACTION,
            "invalid_action": bool(final.get("invalid_action", False)),
            "invalid_policy_output": False,
            "action_bound_failure": bool(final.get("action_out_of_bounds", False)),
            "nonfinite_action": False,
            "episode_length": result.total_environment_steps,
            "actions_executed": len(expert.action_trace),
            "policy_query_count": 0,
            "final_evaluation": final,
            "action_saturation_rate": 0.0,
        }
        record["primary_failure_category"] = classify_failure(record)
        return record
    except Exception as error:  # noqa: BLE001 - preserve paired failure evidence
        traceback.print_exc()
        record = {
            **item,
            "baseline": "candidate_e",
            "expert_source_commit": ACCEPTED_CANDIDATE_E_COMMIT,
            "success": False,
            "outcome": "unexpected_exception",
            "timeout": False,
            "failure_reason": f"{type(error).__name__}: {error}",
            "wrong_object_interaction": False,
            "invalid_action": False,
            "invalid_policy_output": False,
            "action_bound_failure": False,
            "nonfinite_action": False,
            "episode_length": 0,
            "actions_executed": 0,
            "policy_query_count": 0,
            "final_evaluation": {},
            "action_saturation_rate": 0.0,
        }
        record["primary_failure_category"] = classify_failure(record)
        return record
    finally:
        if environment is not None:
            environment.close()


def _hold_record(item: dict[str, Any], *, sim_backend: str) -> dict[str, object]:
    environment: Any | None = None
    policy = HoldPositionPolicyAdapter()
    try:
        task_value = item.get("task_spec")
        if not isinstance(task_value, dict):
            raise Phase2CContractError("scheduled task_spec is malformed")
        task_spec = PushTaskSpec.from_mapping(task_value)
        task = EvaluationTask(
            task_instance=PushTaskInstanceSpec.from_task_spec(task_spec),
            scene_seed=int(item["seed"]),
        )
        environment = _environment(sim_backend=sim_backend, rgb=True)
        evaluator = UnifiedPolicyEvaluator(
            environment=environment,
            observation_extractor=Phase2CPolicyObservationExtractor(
                visual_domain=str(item.get("visual_domain", "base"))
            ),
            action_bound_config=ActionBoundConfig(mode=ActionBoundMode.REJECT),
        )
        result = evaluator.run_episode(
            policy=policy,
            task=task,
            evaluation_id=str(item["evaluation_id"]),
            language_instruction=str(item["language_instruction"]),
        )
        record = {
            **item,
            **result.to_dict(),
            "baseline": "hold_position",
            "invalid_policy_output": result.outcome.value == "policy_failure",
            "action_bound_failure": result.outcome.value == "invalid_action",
            "nonfinite_action": False,
        }
        record["primary_failure_category"] = classify_failure(record)
        return record
    finally:
        policy.close()
        if environment is not None:
            environment.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    schedule_path = args.schedule.resolve()
    selected = _schedule(schedule_path, args.limit_episodes)
    output = args.output_dir.resolve()
    if output.exists():
        raise Phase2CContractError(f"refusing to overwrite baseline output: {output}")
    output.mkdir(parents=True)
    records: list[dict[str, object]] = []
    for ordinal, item in enumerate(selected, start=1):
        record = (
            _expert_record(item, sim_backend=args.sim_backend)
            if args.baseline == "candidate_e"
            else _hold_record(item, sim_backend=args.sim_backend)
        )
        records.append(record)
        print(
            json.dumps(
                {
                    "ordinal": ordinal,
                    "total": len(selected),
                    "evaluation_id": item["evaluation_id"],
                    "baseline": args.baseline,
                    "success": record["success"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    episodes = output / "episodes.jsonl"
    descriptor = os.open(episodes, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    summary = {
        **summarize_sealed_results(records),
        "baseline": args.baseline,
        "expert_source_commit": (
            ACCEPTED_CANDIDATE_E_COMMIT if args.baseline == "candidate_e" else None
        ),
        "schedule_sha256": f"sha256:{sha256_file(schedule_path)}",
        "episode_log_sha256": f"sha256:{sha256_file(episodes)}",
        "paired_identity_count": len(records),
        "expected_identity_count": len(selected),
        "completed": len(records) == len(selected),
    }
    write_json_once(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
