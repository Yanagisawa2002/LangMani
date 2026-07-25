"""Run resumable real closed-loop evaluation for one Phase 2C-B SmolVLA policy."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from langmani.datasets.identity import sha256_hex
from langmani.v2.phase2c_a import PACKAGE_FINGERPRINT, TASK_IDS
from langmani.v2.phase2c_a_evaluator import (
    OfficialManiSkillPolicyEvaluator,
    summarize_evaluation,
    write_video,
)
from langmani.v2.phase2c_b_adapter import Phase2CBSmolVLAPolicyAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--execution-horizon", type=int, choices=(1, 4, 8), required=True)
    parser.add_argument("--task-id", choices=TASK_IDS, required=True)
    parser.add_argument(
        "--split",
        choices=(
            "validation",
            "test_unseen_reset",
            "test_unseen_task_language",
            "test_visual_shift",
        ),
        required=True,
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--maximum-steps", type=int, default=200)
    parser.add_argument(
        "--instruction-condition",
        choices=("correct", "wrong_skill", "blank", "shuffled"),
        default="correct",
    )
    parser.add_argument("--language-intervention-lock", type=Path)
    parser.add_argument("--evaluation-git-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--capture-representatives", action="store_true")
    return parser.parse_args()


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain one object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _write_new_or_equal(path: Path, value: Mapping[str, object]) -> None:
    encoded = json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"existing immutable artifact differs: {path}")
        return
    path.write_text(encoded, encoding="utf-8", newline="\n")


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(value), sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _source_document(source_root: Path, task_id: str) -> dict[str, object]:
    path = source_root / "expanded" / task_id / "motionplanning" / "trajectory.json"
    document = _read_object(path)
    if not isinstance(document.get("env_info"), dict):
        raise RuntimeError(f"source document lacks env_info: {path}")
    return document


def _environment_kwargs(document: Mapping[str, object]) -> dict[str, object]:
    env_info = document["env_info"]
    if not isinstance(env_info, Mapping):
        raise RuntimeError("source env_info is malformed")
    source_kwargs = env_info.get("env_kwargs")
    if not isinstance(source_kwargs, Mapping):
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
    schedule: Mapping[str, object],
    *,
    split: str,
    task_id: str,
    limit: int | None,
) -> list[dict[str, object]]:
    schedules = schedule.get("schedules")
    split_value = schedules.get(split) if isinstance(schedules, Mapping) else None
    task_value = split_value.get(task_id) if isinstance(split_value, Mapping) else None
    if not isinstance(task_value, list) or not all(isinstance(item, dict) for item in task_value):
        raise RuntimeError(f"evaluation schedule lacks {task_id}/{split}")
    rows = list(task_value)
    if limit is not None:
        if limit < 1 or limit > len(rows):
            raise ValueError("--limit lies outside the frozen schedule")
        rows = rows[:limit]
    return rows


def _intervention_instructions(
    path: Path,
    *,
    schedule_fingerprint: object,
    condition: str,
    rows: list[dict[str, object]],
) -> tuple[dict[str, str], str]:
    document = _read_object(path)
    fingerprint = document.pop("fingerprint", None)
    records = document.get("records")
    if (
        document.get("schema_version") != "langmani-v2-phase2c-b-language-intervention-lock-v0"
        or document.get("package_fingerprint") != PACKAGE_FINGERPRINT
        or document.get("evaluation_schedule_fingerprint") != schedule_fingerprint
        or document.get("split") != "validation"
        or fingerprint != f"sha256:{sha256_hex(document)}"
        or not isinstance(records, list)
    ):
        raise RuntimeError("language intervention lock identity is invalid")
    key = {
        "correct": "correct_instruction",
        "wrong_skill": "wrong_skill_instruction",
        "blank": "blank_instruction",
        "shuffled": "shuffled_instruction",
    }[condition]
    selected: dict[str, str] = {}
    requested = {str(row["evaluation_id"]) for row in rows}
    for record in records:
        if not isinstance(record, Mapping):
            raise RuntimeError("language intervention record is malformed")
        evaluation_id = record.get("evaluation_id")
        instruction = record.get(key)
        if evaluation_id in requested:
            if not isinstance(evaluation_id, str) or not isinstance(instruction, str):
                raise RuntimeError("language intervention instruction is malformed")
            selected[evaluation_id] = instruction
    if list(selected) != [str(row["evaluation_id"]) for row in rows]:
        raise RuntimeError("language intervention lock does not cover requested rows")
    return selected, str(fingerprint)


def main() -> int:
    args = parse_args()
    import gymnasium as gym
    import mani_skill.envs  # type: ignore[import-untyped]  # noqa: F401

    if re.fullmatch(r"[0-9a-f]{40}", args.evaluation_git_commit) is None:
        raise RuntimeError("--evaluation-git-commit must be one full Git SHA")
    schedule = _read_object(args.schedule.resolve())
    if (
        schedule.get("schema_version") != "langmani-v2-phase2c-b-evaluation-schedule-v0"
        or schedule.get("package_fingerprint") != PACKAGE_FINGERPRINT
    ):
        raise RuntimeError("evaluation schedule identity is invalid")
    rows = _scheduled_rows(
        schedule,
        split=args.split,
        task_id=args.task_id,
        limit=args.limit,
    )
    if args.instruction_condition == "correct":
        if args.language_intervention_lock is not None:
            raise RuntimeError("correct evaluation does not consume an intervention lock")
        instructions = {str(row["evaluation_id"]): str(row["language_instruction"]) for row in rows}
        intervention_fingerprint = None
    else:
        if args.split != "validation" or args.language_intervention_lock is None:
            raise RuntimeError("language interventions require validation and its frozen lock")
        instructions, intervention_fingerprint = _intervention_instructions(
            args.language_intervention_lock.resolve(),
            schedule_fingerprint=schedule.get("fingerprint"),
            condition=args.instruction_condition,
            rows=rows,
        )
    adapter = Phase2CBSmolVLAPolicyAdapter(
        checkpoint=args.checkpoint.resolve(),
        expected_checkpoint_sha256=args.checkpoint_sha256,
        execution_horizon=args.execution_horizon,
        device="cuda",
        dtype="bfloat16",
    )
    if args.task_id not in adapter.identity.compatible_task_ids:
        raise RuntimeError("checkpoint is incompatible with the scheduled task")
    if args.split != "validation" and args.instruction_condition != "correct":
        raise RuntimeError("final splits cannot use intervention instructions")
    source_document = _source_document(args.source_root.resolve(), args.task_id)
    kwargs = _environment_kwargs(source_document)
    environment: Any = gym.make(args.task_id, **cast(Any, kwargs))
    evaluator = OfficialManiSkillPolicyEvaluator(
        environment,
        task_id=args.task_id,
        maximum_steps=args.maximum_steps,
    )
    output_root = args.output_root.resolve()
    manifest_semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-b-evaluation-run-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "schedule_fingerprint": schedule.get("fingerprint"),
        "checkpoint": args.checkpoint.resolve().as_posix(),
        "checkpoint_sha256": args.checkpoint_sha256,
        "policy_runtime": dict(adapter.runtime_manifest),
        "model_kind": adapter.model_kind.value,
        "task_id": args.task_id,
        "split": args.split,
        "scheduled_episode_count": len(rows),
        "maximum_steps": args.maximum_steps,
        "execution_horizon": args.execution_horizon,
        "instruction_condition": args.instruction_condition,
        "language_intervention_lock_fingerprint": intervention_fingerprint,
        "evaluation_git_commit": args.evaluation_git_commit,
        "environment_kwargs": kwargs,
        "action_clipping": False,
        "action_projection": False,
        "invalid_output_policy": "hard_reject",
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
            "schema_version": "langmani-v2-phase2c-b-representative-videos-v0",
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
            evaluation_id = str(row["evaluation_id"])
            instruction = instructions[evaluation_id]
            capture = args.capture_representatives and (
                f"{args.task_id}:{args.instruction_condition}:success" not in filled_roles
                or f"{args.task_id}:{args.instruction_condition}:failure" not in filled_roles
            )
            evaluated = evaluator.run_episode(
                policy=adapter,
                schedule=row,
                task_condition_id=args.task_id,
                language_instruction=instruction,
                allow_blank_language_instruction=args.instruction_condition == "blank",
                capture_video=capture,
            )
            value = evaluated.result.to_dict()
            value["instruction_condition"] = args.instruction_condition
            value["instruction_sha256"] = f"sha256:{sha256_hex({'instruction': instruction})}"
            _append_jsonl(episodes_path, value)
            existing.append(value)
            if capture and evaluated.frames:
                role = (
                    f"{args.task_id}:{args.instruction_condition}:"
                    f"{'success' if evaluated.result.success else 'failure'}"
                )
                if role not in filled_roles:
                    video_name = role.replace(":", "__") + ".mp4"
                    record = write_video(output_root / "videos" / video_name, evaluated.frames)
                    videos.append(
                        {
                            **record,
                            "role": role,
                            "evaluation_id": evaluation_id,
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
        "model_kind": adapter.model_kind.value,
        "task_id": args.task_id,
        "split": args.split,
        "execution_horizon": args.execution_horizon,
        "instruction_condition": args.instruction_condition,
        "language_intervention_lock_fingerprint": intervention_fingerprint,
        "runtime_manifest_fingerprint": manifest["fingerprint"],
        "completed": True,
    }
    _write_new_or_equal(output_root / "summary.json", summary)
    _write_new_or_equal(
        output_root / "complete.json",
        {
            "schema_version": "langmani-v2-phase2c-b-evaluation-complete-v0",
            "episode_count": len(existing),
            "runtime_manifest": "runtime_manifest.json",
            "episodes": "episodes.jsonl",
            "summary": "summary.json",
            "representative_videos": (
                "representative_videos.json" if video_registry_path.exists() else None
            ),
        },
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
