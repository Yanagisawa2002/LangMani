"""Run validation-only or sealed real ManiSkill evaluation for the frozen SmolVLA policy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from langmani.environments.push_specs import PushTaskSpec
from langmani.environments.push_to_region import ENV_ID
from langmani.policies.act_action_bounds import ActionBoundConfig, ActionBoundMode
from langmani.v2.evaluator import (
    Phase2CPolicyObservationExtractor,
    UnifiedPolicyEvaluator,
)
from langmani.v2.phase2c import Phase2CContractError, read_json_object, sha256_file, write_json_once
from langmani.v2.phase2c_evaluation import (
    FrozenPolicyManifest,
    classify_failure,
    development_competence_gate,
    quality_level,
    summarize_sealed_results,
)
from langmani.v2.registry import load_policy_adapter
from langmani.v2.smolvla_adapter import SmolVLAAdapterConfig
from langmani.v2.taxonomy import EvaluationTask, PushTaskInstanceSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("development", "sealed"), required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--adapter-config", type=Path, required=True)
    parser.add_argument("--frozen-policy-manifest", type=Path)
    parser.add_argument("--development-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sim-backend", choices=("physx_cpu", "physx_cuda"), default="physx_cuda")
    parser.add_argument("--limit-episodes", type=int)
    return parser.parse_args(argv)


def _environment(sim_backend: str) -> Any:
    import gymnasium as gym

    import langmani.environments  # noqa: F401 - registers ENV_ID

    return gym.make(
        ENV_ID,
        num_envs=1,
        obs_mode="rgb",
        reward_mode="none",
        control_mode="pd_joint_pos",
        render_mode=None,
        sim_backend=sim_backend,
    )


def _selected_schedule(value: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    selected = value.get("development" if mode == "development" else "sealed")
    if not isinstance(selected, list) or not all(isinstance(item, dict) for item in selected):
        raise Phase2CContractError("evaluation schedule is malformed")
    if mode == "development" and any(item.get("split") != "validation" for item in selected):
        raise Phase2CContractError("development evaluation may access validation only")
    if mode == "sealed" and any(item.get("split") == "validation" for item in selected):
        raise Phase2CContractError("sealed schedule cannot contain development validation episodes")
    return selected


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    schedule_path = args.schedule.resolve()
    schedule = read_json_object(schedule_path, "Phase 2C evaluation schedule")
    selected = _selected_schedule(schedule, args.mode)
    if args.limit_episodes is not None:
        if args.limit_episodes < 1:
            raise Phase2CContractError("limit_episodes must be positive")
        selected = selected[: args.limit_episodes]
    if args.mode == "sealed":
        if args.frozen_policy_manifest is None or args.development_summary is None:
            raise Phase2CContractError(
                "sealed evaluation requires frozen policy and development summary manifests"
            )
        frozen = read_json_object(
            args.frozen_policy_manifest.resolve(), "frozen final policy manifest"
        )
        try:
            frozen_contract = FrozenPolicyManifest(
                checkpoint_sha256=frozen["checkpoint_sha256"],
                base_model_revision=frozen["base_model_revision"],
                training_seed=frozen["training_seed"],
                training_config_sha256=frozen["training_config_sha256"],
                train_view_sha256=frozen["train_view_sha256"],
                normalization_sha256=frozen["normalization_sha256"],
                processor_sha256=frozen["processor_sha256"],
                action_chunk_size=frozen["action_chunk_size"],
                execution_horizon=frozen["execution_horizon"],
                control_frequency_hz=frozen["control_frequency_hz"],
                inference_seed=frozen["inference_seed"],
                final_test_identity_sha256=frozen["final_test_identity_sha256"],
                schema_version=frozen["schema_version"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise Phase2CContractError(f"frozen policy manifest is invalid: {error}") from error
        if frozen != frozen_contract.to_dict():
            raise Phase2CContractError("frozen policy manifest bytes are not canonical")
        if frozen_contract.final_test_identity_sha256 != sha256_file(schedule_path):
            raise Phase2CContractError("frozen policy does not bind the final schedule bytes")
        adapter_config = SmolVLAAdapterConfig.load(args.adapter_config.resolve())
        adapter_bindings = {
            "checkpoint_sha256": adapter_config.checkpoint_sha256,
            "base_model_revision": adapter_config.base_model_revision,
            "training_seed": adapter_config.training_seed,
            "training_config_sha256": adapter_config.training_config_sha256,
            "train_view_sha256": adapter_config.train_view_sha256,
            "normalization_sha256": adapter_config.normalization_sha256,
            "processor_sha256": adapter_config.processor_sha256,
            "execution_horizon": adapter_config.execution_horizon,
            "inference_seed": adapter_config.inference_seed,
        }
        if any(frozen.get(key) != value for key, value in adapter_bindings.items()):
            raise Phase2CContractError("adapter configuration differs from the frozen final policy")
        development = read_json_object(
            args.development_summary.resolve(), "development competence summary"
        )
        if development.get("passed") is not True or development.get("mode") != "development":
            raise Phase2CContractError("sealed evaluation requires a passing development gate")
        if development.get("schedule_sha256") != f"sha256:{sha256_file(schedule_path)}":
            raise Phase2CContractError("development evidence used a different frozen schedule")
    output = args.output_dir.resolve()
    if output.exists():
        raise Phase2CContractError(f"refusing to overwrite closed-loop output: {output}")
    output.mkdir(parents=True)
    adapter = load_policy_adapter(
        config_path=args.adapter_config.resolve(),
        project_root=PROJECT_ROOT,
    )
    records: list[dict[str, object]] = []
    try:
        for ordinal, item in enumerate(selected):
            environment: Any | None = None
            try:
                task_value = item.get("task_spec")
                if not isinstance(task_value, dict):
                    raise Phase2CContractError("scheduled task_spec is malformed")
                task_spec = PushTaskSpec.from_mapping(task_value)
                task = EvaluationTask(
                    task_instance=PushTaskInstanceSpec.from_task_spec(task_spec),
                    scene_seed=int(item["seed"]),
                )
                domain = str(item.get("visual_domain", "base"))
                environment = _environment(args.sim_backend)
                evaluator = UnifiedPolicyEvaluator(
                    environment=environment,
                    observation_extractor=Phase2CPolicyObservationExtractor(visual_domain=domain),
                    action_bound_config=ActionBoundConfig(mode=ActionBoundMode.REJECT),
                )
                result = evaluator.run_episode(
                    policy=adapter,
                    task=task,
                    evaluation_id=str(item["evaluation_id"]),
                    language_instruction=str(item["language_instruction"]),
                )
                record = {
                    **item,
                    **result.to_dict(),
                    "action_bound_failure": result.outcome.value == "invalid_action",
                    "invalid_policy_output": result.outcome.value == "policy_failure",
                    "nonfinite_action": (
                        result.outcome.value == "policy_failure"
                        and "non-finite" in str(result.failure_reason).lower()
                    ),
                }
                record["primary_failure_category"] = classify_failure(record)
                records.append(record)
                print(
                    json.dumps(
                        {
                            "ordinal": ordinal + 1,
                            "total": len(selected),
                            "evaluation_id": item["evaluation_id"],
                            "split": item["split"],
                            "success": result.success,
                            "outcome": result.outcome.value,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            finally:
                if environment is not None:
                    environment.close()
    finally:
        adapter.close()
    episodes_path = output / "episodes.jsonl"
    descriptor = os.open(episodes_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    summary = (
        development_competence_gate(records)
        if args.mode == "development"
        else summarize_sealed_results(records)
    )
    quality = (
        quality_level(summary, development_gate=development) if args.mode == "sealed" else None
    )
    summary = {
        **summary,
        "mode": args.mode,
        "schedule_sha256": f"sha256:{sha256_file(schedule_path)}",
        "adapter_config_sha256": f"sha256:{sha256_file(args.adapter_config.resolve())}",
        "episode_log_sha256": f"sha256:{sha256_file(episodes_path)}",
        "runtime_manifest": dict(adapter.runtime_manifest),
        "completed_episode_count": len(records),
        "expected_episode_count": len(selected),
        "completed": len(records) == len(selected),
        "quality_decision": quality,
    }
    write_json_once(output / "summary.json", summary)
    write_json_once(
        output / "complete.json",
        {
            "schema_version": "langmani-v2-phase2c-closed-loop-complete-v0",
            "mode": args.mode,
            "summary_sha256": f"sha256:{sha256_file(output / 'summary.json')}",
            "episodes_sha256": f"sha256:{sha256_file(episodes_path)}",
            "completed": len(records) == len(selected),
        },
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
