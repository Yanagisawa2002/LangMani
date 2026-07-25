"""Run resumable real closed-loop evaluation for one Phase 2C-A ACT policy."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from langmani.datasets.identity import sha256_hex
from langmani.v2.phase2c_a import (
    PACKAGE_FINGERPRINT,
    TASK_IDS,
)
from langmani.v2.phase2c_a_adapter import Phase2CAActPolicyAdapter
from langmani.v2.phase2c_a_evaluator import (
    OfficialManiSkillPolicyEvaluator,
    append_episode_jsonl,
    summarize_evaluation,
    write_video,
)


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain one object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _write_new_or_equal(path: Path, value: dict[str, object]) -> None:
    if path.exists():
        if _read_object(path) != value:
            raise RuntimeError(f"existing immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _task_intervention_conditions(
    path: Path,
    *,
    role: str,
    task_id: str,
    rows: list[dict[str, object]],
) -> tuple[dict[str, str], str]:
    document = _read_object(path.resolve())
    fingerprint = document.pop("fingerprint", None)
    if (
        document.get("schema_version") != "langmani-v2-phase2c-a1-task-intervention-lock-v0"
        or document.get("package_fingerprint") != PACKAGE_FINGERPRINT
        or document.get("split") != "validation"
        or role not in {"wrong", "shuffled"}
        or fingerprint != f"sha256:{sha256_hex(document)}"
    ):
        raise RuntimeError("task-intervention lock identity is invalid")
    by_task = document.get("assignments_by_task")
    task_rows = by_task.get(task_id) if isinstance(by_task, dict) else None
    if not isinstance(task_rows, list):
        raise RuntimeError("task-intervention lock lacks the requested task")
    conditions: dict[str, str] = {}
    for record in task_rows:
        if not isinstance(record, dict):
            raise RuntimeError("task-intervention assignment is malformed")
        evaluation_id = record.get("evaluation_id")
        condition = record.get(role)
        if (
            not isinstance(evaluation_id, str)
            or not isinstance(condition, str)
            or condition not in TASK_IDS
        ):
            raise RuntimeError("task-intervention assignment has an invalid condition")
        conditions[evaluation_id] = condition
    expected = [str(row["evaluation_id"]) for row in rows]
    if list(conditions) != expected:
        raise RuntimeError("task-intervention assignments do not match the schedule prefix")
    return conditions, str(fingerprint)


def _validate_final_policy_lock(
    path: Path,
    *,
    adapter: Phase2CAActPolicyAdapter,
    schedule_fingerprint: object,
    execution_horizon: int,
    evaluation_git_commit: str,
    split: str,
    task_id: str,
    rows: list[dict[str, object]],
) -> str:
    document = _read_object(path.resolve())
    fingerprint = document.pop("fingerprint", None)
    policies = document.get("policies")
    policy = policies.get(adapter.model_kind.value) if isinstance(policies, dict) else None
    runtime = adapter.runtime_manifest
    components = runtime.get("processor_components")
    identity_counts = document.get("final_identity_counts")
    split_counts = identity_counts.get(split) if isinstance(identity_counts, dict) else None
    identity_prefixes = document.get("final_identity_prefixes")
    split_prefixes = identity_prefixes.get(split) if isinstance(identity_prefixes, dict) else None
    task_prefix = split_prefixes.get(task_id) if isinstance(split_prefixes, dict) else None
    expected_prefix = [
        {
            "evaluation_id": row.get("evaluation_id"),
            "reset_identity": row.get("reset_identity"),
        }
        for row in rows
    ]
    if (
        document.get("schema_version") != "langmani-v2-phase2c-a1-final-policy-lock-v0"
        or document.get("package_fingerprint") != PACKAGE_FINGERPRINT
        or document.get("schedule_fingerprint") != schedule_fingerprint
        or document.get("evaluation_git_commit") != evaluation_git_commit
        or document.get("execution_horizon") != execution_horizon
        or document.get("settings_mutable_after_final_results") is not False
        or document.get("final_results_available") is not False
        or fingerprint != f"sha256:{sha256_hex(document)}"
        or not isinstance(policy, dict)
        or policy.get("checkpoint_fingerprint") != runtime.get("checkpoint_identity")
        or policy.get("run_fingerprint") != runtime.get("run_fingerprint")
        or policy.get("checkpoint_components") != components
        or not isinstance(split_counts, dict)
        or split_counts.get(task_id) != len(rows)
        or task_prefix != expected_prefix
    ):
        raise RuntimeError("final policy lock does not match the requested evaluation")
    return str(fingerprint)


def _source_document(source_root: Path, task_id: str) -> dict[str, object]:
    path = source_root / "expanded" / task_id / "motionplanning" / "trajectory.json"
    document = _read_object(path)
    if not isinstance(document.get("env_info"), dict):
        raise RuntimeError(f"source document lacks env_info: {path}")
    return document


def _environment_kwargs(document: dict[str, object]) -> dict[str, object]:
    env_info = document["env_info"]
    assert isinstance(env_info, dict)
    source_kwargs = env_info.get("env_kwargs")
    if not isinstance(source_kwargs, dict):
        raise RuntimeError("source env_info lacks env_kwargs")
    kwargs = dict(source_kwargs)
    kwargs.update(
        {
            "obs_mode": "rgb",
            "reward_mode": "none",
            "render_mode": None,
            "sim_backend": "physx_cpu",
            "render_backend": "sapien_cuda",
            "sensor_configs": {"width": 256, "height": 256},
            "num_envs": 1,
        }
    )
    return kwargs


def _scheduled_rows(
    schedule: dict[str, object],
    *,
    split: str,
    task_id: str,
    limit: int | None,
) -> list[dict[str, object]]:
    schedules = schedule.get("schedules")
    if not isinstance(schedules, dict):
        raise RuntimeError("evaluation schedule lacks schedules")
    split_value = schedules.get(split)
    if not isinstance(split_value, dict):
        raise RuntimeError(f"evaluation schedule lacks split {split}")
    task_value = split_value.get(task_id)
    if not isinstance(task_value, list) or not all(isinstance(item, dict) for item in task_value):
        raise RuntimeError(f"evaluation schedule lacks {task_id}/{split}")
    rows = list(task_value)
    if limit is not None:
        if limit < 1 or limit > len(rows):
            raise ValueError("--limit lies outside the frozen schedule")
        rows = rows[:limit]
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint-relative-path", required=True)
    parser.add_argument("--execution-horizon", type=int, choices=(1, 4, 8), required=True)
    parser.add_argument("--task-id", choices=TASK_IDS, required=True)
    parser.add_argument(
        "--split",
        choices=("validation", "test_unseen_reset", "test_visual_shift"),
        required=True,
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--maximum-steps", type=int, default=200)
    condition = parser.add_mutually_exclusive_group()
    condition.add_argument("--task-condition-id", choices=TASK_IDS)
    condition.add_argument("--task-intervention-manifest", type=Path)
    parser.add_argument("--task-intervention-role", choices=("wrong", "shuffled"))
    parser.add_argument("--evaluation-git-commit", required=True)
    parser.add_argument("--final-policy-lock", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--capture-representatives", action="store_true")
    parser.add_argument("--require-bounded-action-head-v1", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import gymnasium as gym
    import mani_skill.envs  # type: ignore[import-untyped]  # noqa: F401

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    schedule_document = _read_object(args.schedule.resolve())
    if schedule_document.get("package_fingerprint") != PACKAGE_FINGERPRINT:
        raise RuntimeError("evaluation schedule uses a noncanonical package fingerprint")
    rows = _scheduled_rows(
        schedule_document,
        split=args.split,
        task_id=args.task_id,
        limit=args.limit,
    )
    if re.fullmatch(r"[0-9a-f]{40}", args.evaluation_git_commit) is None:
        raise RuntimeError("--evaluation-git-commit must be one full Git SHA")
    if (args.task_intervention_manifest is None) != (args.task_intervention_role is None):
        raise RuntimeError("task-intervention manifest and role must be provided together")
    intervention_conditions: dict[str, str] = {}
    intervention_fingerprint: str | None = None
    if args.task_intervention_manifest is not None:
        intervention_conditions, intervention_fingerprint = _task_intervention_conditions(
            args.task_intervention_manifest,
            role=args.task_intervention_role,
            task_id=args.task_id,
            rows=rows,
        )
    adapter = Phase2CAActPolicyAdapter.from_checkpoint(
        run_root=args.run_root,
        checkpoint_relative_path=args.checkpoint_relative_path,
        execution_horizon=args.execution_horizon,
        device="cuda",
    )
    if args.require_bounded_action_head_v1 and not adapter.bounded_action_head_v1:
        raise RuntimeError("evaluation requires a bounded_action_head_v1 checkpoint")
    is_final = args.split in {"test_unseen_reset", "test_visual_shift"}
    if is_final and (
        args.final_policy_lock is None
        or args.task_condition_id is not None
        or intervention_conditions
    ):
        raise RuntimeError("final evaluation requires the frozen policy lock and no intervention")
    if not is_final and args.final_policy_lock is not None:
        raise RuntimeError("final policy lock cannot be consumed by validation evaluation")
    final_policy_lock_fingerprint = (
        _validate_final_policy_lock(
            args.final_policy_lock,
            adapter=adapter,
            schedule_fingerprint=schedule_document.get("fingerprint"),
            execution_horizon=args.execution_horizon,
            evaluation_git_commit=args.evaluation_git_commit,
            split=args.split,
            task_id=args.task_id,
            rows=rows,
        )
        if args.final_policy_lock is not None
        else None
    )
    if args.task_condition_id is None and args.task_id not in adapter.identity.compatible_task_ids:
        raise RuntimeError("policy is incompatible with the scheduled environment task")
    source_document = _source_document(args.source_root.resolve(), args.task_id)
    kwargs = _environment_kwargs(source_document)
    environment: Any = gym.make(args.task_id, **kwargs)
    evaluator = OfficialManiSkillPolicyEvaluator(
        environment,
        task_id=args.task_id,
        maximum_steps=args.maximum_steps,
    )
    manifest_semantic = {
        "schema_version": "langmani-v2-phase2c-a-evaluation-run-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "schedule_fingerprint": schedule_document.get("fingerprint"),
        "source_root": args.source_root.resolve().as_posix(),
        "task_id": args.task_id,
        "split": args.split,
        "scheduled_episode_count": len(rows),
        "maximum_steps": args.maximum_steps,
        "task_condition_id": (
            None if intervention_conditions else args.task_condition_id or args.task_id
        ),
        "task_intervention_role": args.task_intervention_role,
        "task_intervention_lock_fingerprint": intervention_fingerprint,
        "evaluation_git_commit": args.evaluation_git_commit,
        "final_policy_lock_fingerprint": final_policy_lock_fingerprint,
        "policy_runtime": dict(adapter.runtime_manifest),
        "bounded_action_head_v1_required": args.require_bounded_action_head_v1,
        "environment_kwargs": kwargs,
        "action_clipping": False,
        "action_projection": False,
        "invalid_output_policy": "hard_reject",
        "visual_shift": (
            "langmani-v2-phase2b6-postrender-appearance-v0"
            if args.split == "test_visual_shift"
            else None
        ),
    }
    manifest = {
        **manifest_semantic,
        "fingerprint": f"sha256:{sha256_hex(manifest_semantic)}",
    }
    _write_new_or_equal(output_root / "runtime_manifest.json", manifest)
    episodes_path = output_root / "episodes.jsonl"
    existing = _read_jsonl(episodes_path)
    expected_ids = [str(row["evaluation_id"]) for row in rows]
    existing_ids = [str(row.get("evaluation_id")) for row in existing]
    if (
        len(existing_ids) != len(set(existing_ids))
        or existing_ids != expected_ids[: len(existing_ids)]
    ):
        raise RuntimeError("existing evaluation log is not a valid schedule prefix")
    video_registry_path = output_root / "representative_videos.json"
    video_registry = (
        _read_object(video_registry_path)
        if video_registry_path.exists()
        else {
            "schema_version": "langmani-v2-phase2c-a-representative-videos-v0",
            "videos": [],
        }
    )
    videos = video_registry.get("videos")
    if not isinstance(videos, list):
        raise RuntimeError("representative video registry is malformed")
    filled_roles = {
        str(item.get("role")) for item in videos if isinstance(item, dict) and item.get("role")
    }
    try:
        for row in rows[len(existing) :]:
            row_id = str(row["evaluation_id"])
            task_condition_id = intervention_conditions.get(
                row_id,
                args.task_condition_id or args.task_id,
            )
            capture = args.capture_representatives and (
                args.task_condition_id is not None
                or bool(intervention_conditions)
                or f"{args.task_id}:success" not in filled_roles
                or f"{args.task_id}:failure" not in filled_roles
                or (
                    args.split == "test_visual_shift"
                    and f"{args.task_id}:visual_shift" not in filled_roles
                )
            )
            evaluated = evaluator.run_episode(
                policy=adapter,
                schedule=row,
                task_condition_id=task_condition_id,
                capture_video=capture,
            )
            append_episode_jsonl(episodes_path, evaluated.result)
            existing.append(evaluated.result.to_dict())
            if capture and evaluated.frames:
                if args.task_condition_id is not None or intervention_conditions:
                    role = (
                        f"{args.task_id}:task_condition:"
                        f"{args.task_intervention_role or args.task_condition_id}"
                    )
                elif args.split == "test_visual_shift":
                    role = f"{args.task_id}:visual_shift"
                else:
                    role = (
                        f"{args.task_id}:success"
                        if evaluated.result.success
                        else f"{args.task_id}:failure"
                    )
                if role not in filled_roles:
                    video_name = role.replace(":", "__") + ".mp4"
                    record = write_video(
                        output_root / "videos" / video_name,
                        evaluated.frames,
                    )
                    videos.append(
                        {
                            **record,
                            "role": role,
                            "evaluation_id": evaluated.result.evaluation_id,
                            "outcome": evaluated.result.outcome,
                        }
                    )
                    filled_roles.add(role)
                    staging = video_registry_path.with_name(f".{video_registry_path.name}.staging")
                    staging.write_text(
                        json.dumps(video_registry, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    os.replace(staging, video_registry_path)
    finally:
        adapter.close()
        environment.close()
    if len(existing) != len(rows):
        raise RuntimeError("evaluation did not complete its frozen schedule")
    summary = {
        **summarize_evaluation(existing),
        "execution_horizon": args.execution_horizon,
        "split": args.split,
        "task_id": args.task_id,
        "task_condition_id": (
            next(iter({str(item["task_condition_id"]) for item in existing}))
            if len({str(item["task_condition_id"]) for item in existing}) == 1
            else None
        ),
        "task_condition_ids": sorted({str(item["task_condition_id"]) for item in existing}),
        "task_intervention_role": args.task_intervention_role,
        "task_intervention_lock_fingerprint": intervention_fingerprint,
        "runtime_manifest_fingerprint": manifest["fingerprint"],
        "completed": True,
    }
    _write_new_or_equal(output_root / "summary.json", summary)
    complete = {
        "schema_version": "langmani-v2-phase2c-a-evaluation-complete-v0",
        "episode_count": len(existing),
        "runtime_manifest": "runtime_manifest.json",
        "episodes": "episodes.jsonl",
        "summary": "summary.json",
        "representative_videos": (
            "representative_videos.json" if video_registry_path.exists() else None
        ),
    }
    _write_new_or_equal(output_root / "complete.json", complete)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
