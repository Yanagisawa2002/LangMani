"""Run a registered LangMani 2.0 policy on canonical TaskSpecs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langmani.environments.pick_place_by_instruction import ENV_ID
from langmani.v2.evaluator import UnifiedPolicyEvaluator, persist_evaluation
from langmani.v2.registry import load_policy_adapter
from langmani.v2.taxonomy import EvaluationTask, load_task_catalog

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = (
    PROJECT_ROOT / "configs" / "langmani_v2" / "policies" / "act_per_task_red_left_v0.json"
)
DEFAULT_TASKS = PROJECT_ROOT / "configs" / "langmani_v2" / "tasks" / "pick_and_place_v0.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-config", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASKS)
    parser.add_argument(
        "--task-id",
        default="langmani-pick-place-task-v0:red_cube:left_bin:canonical_v0",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(101, 102, 103))
    parser.add_argument("--sim-backend", choices=("physx_cpu", "physx_cuda"), default="physx_cpu")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _create_environment(sim_backend: str) -> object:
    import gymnasium as gym

    import langmani.environments  # noqa: F401 - registers ENV_ID

    return gym.make(
        ENV_ID,
        num_envs=1,
        obs_mode="rgb",
        reward_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend=sim_backend,
    )


def _task_instance(path: Path, task_id: str):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("task catalog must be a JSON object")
    _, tasks = load_task_catalog(value)
    try:
        return next(item for item in tasks if item.canonical_task_id == task_id)
    except StopIteration as error:
        raise ValueError(f"task catalog does not contain {task_id}") from error


def main() -> int:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise ValueError("seeds must be unique non-negative integers")
    task_instance = _task_instance(args.task_config.resolve(), args.task_id)
    environment = _create_environment(args.sim_backend)
    policy = None
    try:
        policy = load_policy_adapter(
            config_path=args.policy_config.resolve(),
            project_root=PROJECT_ROOT,
        )
        evaluator = UnifiedPolicyEvaluator(environment=environment)
        results = tuple(
            evaluator.run_episode(
                policy=policy,
                task=EvaluationTask(task_instance=task_instance, scene_seed=seed),
                evaluation_id=f"langmani-v2-phase1:{task_instance.canonical_task_id}:{seed}",
            )
            for seed in args.seeds
        )
        summary = persist_evaluation(
            output_root=args.output_root,
            runtime_manifest=policy.runtime_manifest,
            results=results,
        )
    finally:
        if policy is not None:
            policy.close()
        close = getattr(environment, "close", None)
        if callable(close):
            close()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
