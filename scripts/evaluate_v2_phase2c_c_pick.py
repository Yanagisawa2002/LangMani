"""Run compact closed-loop Pick evaluation for a frozen Phase 2C-C schedule."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from langmani.datasets.identity import sha256_hex
from langmani.v2.phase2c_a import PACKAGE_FINGERPRINT
from langmani.v2.phase2c_a_evaluator import (
    OfficialManiSkillPolicyEvaluator,
    summarize_evaluation,
)
from langmani.v2.phase2c_b_adapter import Phase2CBSmolVLAPolicyAdapter
from langmani.v2.phase2c_c import Phase2CCContractError, canonical_fingerprint
from langmani.v2.phase2c_c_adapter import Phase2CCRelativeSmolVLAAdapter

EXPECTED_COUNTS = {
    "smoke": 1,
    "training_reset": 30,
    "validation_horizon_screen": 6,
    "validation": 30,
    "test_unseen_reset": 50,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formulation", choices=("absolute", "relative"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--execution-horizon", type=int, choices=(1, 8), required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument(
        "--split-role",
        choices=tuple(EXPECTED_COUNTS),
        required=True,
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evaluation-git-commit", required=True)
    parser.add_argument("--maximum-steps", type=int, default=200)
    parser.add_argument("--validation-authorization", type=Path)
    return parser.parse_args()


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2CCContractError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise Phase2CCContractError(f"{path} must contain one object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise Phase2CCContractError(f"{path} contains a non-object row")
        rows.append(value)
    return rows


def _write_new_or_equal(path: Path, value: Mapping[str, object]) -> None:
    encoded = json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise Phase2CCContractError(f"existing immutable evaluation differs: {path}")
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
        raise Phase2CCContractError(f"source document lacks env_info: {path}")
    return document


def _environment_kwargs(document: Mapping[str, object]) -> dict[str, object]:
    env_info = document["env_info"]
    if not isinstance(env_info, Mapping):
        raise Phase2CCContractError("source env_info is malformed")
    source_kwargs = env_info.get("env_kwargs")
    if not isinstance(source_kwargs, Mapping):
        raise Phase2CCContractError("source env_info lacks env_kwargs")
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


def _numpy(value: object) -> np.ndarray:
    candidate = value
    for name in ("detach", "cpu"):
        operation = getattr(candidate, name, None)
        if callable(operation):
            candidate = operation()
    operation = getattr(candidate, "numpy", None)
    if callable(operation):
        candidate = operation()
    return np.asarray(candidate)


class _DiagnosticEnvironment:
    """Transparent environment wrapper that records non-policy task diagnostics."""

    def __init__(self, environment: Any) -> None:
        self.environment = environment
        self.initial_cube_z: float | None = None
        self.entered_grasp_region = False
        self.ever_grasped = False
        self.ever_lifted = False

    @property
    def unwrapped(self) -> Any:
        return self.environment.unwrapped

    def reset(self, *args: object, **kwargs: object) -> object:
        result = self.environment.reset(*args, **kwargs)
        cube = _numpy(self.unwrapped.cube.pose.p).reshape(-1, 3)[0]
        self.initial_cube_z = float(cube[2])
        self.entered_grasp_region = False
        self.ever_grasped = False
        self.ever_lifted = False
        info = result[1] if isinstance(result, tuple) and len(result) == 2 else {}
        self._update(info)
        return result

    def step(self, action: object) -> object:
        result = self.environment.step(action)
        info = result[4] if isinstance(result, tuple) and len(result) == 5 else {}
        self._update(info)
        return result

    def _update(self, info: object) -> None:
        cube = _numpy(self.unwrapped.cube.pose.p).reshape(-1, 3)[0]
        tcp = _numpy(self.unwrapped.agent.tcp_pose.p).reshape(-1, 3)[0]
        distance = float(np.linalg.norm(cube - tcp))
        if math.isfinite(distance) and distance <= 0.05:
            self.entered_grasp_region = True
        if isinstance(info, Mapping):
            grasped = _numpy(info.get("is_grasped", False)).reshape(-1)
            if grasped.size and bool(grasped[0]):
                self.ever_grasped = True
        if self.initial_cube_z is not None and float(cube[2]) - self.initial_cube_z >= 0.015:
            self.ever_lifted = True

    def close(self) -> None:
        self.environment.close()


def _schedule_rows(
    schedule: Mapping[str, object],
    split_role: str,
) -> list[dict[str, object]]:
    raw = schedule.get(split_role)
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise Phase2CCContractError(f"schedule lacks role {split_role}")
    rows = [dict(row) for row in raw if isinstance(row, Mapping)]
    expected = EXPECTED_COUNTS[split_role]
    if split_role == "smoke":
        training = schedule.get("training_reset")
        if not isinstance(training, Sequence) or not training:
            raise Phase2CCContractError("smoke lacks its frozen training reset")
        rows = [dict(cast(Mapping[str, object], training[0]))]
        rows[0]["evaluation_id"] = "phase2c-c:smoke:00"
        rows[0]["split"] = "smoke"
    if len(rows) != expected:
        raise Phase2CCContractError(
            f"schedule role {split_role} must contain exactly {expected} rows"
        )
    return rows


def _validate_final_authorization(
    path: Path | None,
    *,
    checkpoint_sha256: str,
    execution_horizon: int,
) -> dict[str, object]:
    if path is None:
        raise Phase2CCContractError("test_unseen_reset requires validation authorization")
    value = _read_object(path.resolve())
    semantic = dict(value)
    fingerprint = semantic.pop("fingerprint", None)
    if (
        fingerprint != canonical_fingerprint(semantic)
        or value.get("schema_version") != "langmani-v2-phase2c-c-evaluation-summary-v0"
        or value.get("split_role") != "validation"
        or value.get("episode_count") != 30
        or not isinstance(value.get("success_count"), int)
        or int(cast(int, value["success_count"])) < 3
        or value.get("checkpoint_identities") != [checkpoint_sha256]
        or value.get("execution_horizon") != execution_horizon
    ):
        raise Phase2CCContractError("relative validation did not authorize unseen-reset access")
    return {
        "path": path.resolve().as_posix(),
        "fingerprint": value["fingerprint"],
        "success_count": value["success_count"],
    }


def main() -> int:
    args = parse_args()
    import gymnasium as gym
    import mani_skill.envs  # type: ignore[import-untyped]  # noqa: F401

    if re.fullmatch(r"[0-9a-f]{40}", args.evaluation_git_commit) is None:
        raise Phase2CCContractError("--evaluation-git-commit must be one full Git SHA")
    schedule = _read_object(args.schedule.resolve())
    schedule_semantic = dict(schedule)
    schedule_fingerprint = schedule_semantic.pop("fingerprint", None)
    if (
        schedule.get("schema_version") != "langmani-v2-phase2c-c-evaluation-schedule-v0"
        or schedule.get("package_fingerprint") != PACKAGE_FINGERPRINT
        or schedule_fingerprint != canonical_fingerprint(schedule_semantic)
    ):
        raise Phase2CCContractError("evaluation schedule identity changed")
    split_role = str(args.split_role)
    rows = _schedule_rows(schedule, split_role)
    final_authorization = None
    checkpoint_sha256 = str(args.checkpoint_sha256)
    if split_role == "test_unseen_reset":
        if args.formulation != "relative":
            raise Phase2CCContractError("absolute baseline cannot open Phase 2C-C final")
        final_authorization = _validate_final_authorization(
            args.validation_authorization,
            checkpoint_sha256=checkpoint_sha256,
            execution_horizon=args.execution_horizon,
        )
    elif args.validation_authorization is not None:
        raise Phase2CCContractError("non-final evaluation cannot consume final authorization")
    adapter: Any
    if args.formulation == "absolute":
        adapter = Phase2CBSmolVLAPolicyAdapter(
            checkpoint=args.checkpoint.resolve(),
            expected_checkpoint_sha256=checkpoint_sha256,
            execution_horizon=args.execution_horizon,
            device="cuda",
            dtype="bfloat16",
        )
    else:
        adapter = Phase2CCRelativeSmolVLAAdapter(
            checkpoint=args.checkpoint.resolve(),
            expected_checkpoint_sha256=checkpoint_sha256,
            execution_horizon=args.execution_horizon,
            device="cuda",
            dtype="bfloat16",
        )
    source_document = _source_document(args.source_root.resolve(), "PickCube-v1")
    kwargs = _environment_kwargs(source_document)
    raw_environment: Any = gym.make("PickCube-v1", **cast(Any, kwargs))
    environment = _DiagnosticEnvironment(raw_environment)
    evaluator = OfficialManiSkillPolicyEvaluator(
        environment,
        task_id="PickCube-v1",
        maximum_steps=args.maximum_steps,
    )
    output = args.output_root.resolve()
    manifest_semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-c-evaluation-run-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "schedule_fingerprint": schedule_fingerprint,
        "formulation": args.formulation,
        "checkpoint": args.checkpoint.resolve().as_posix(),
        "checkpoint_sha256": checkpoint_sha256,
        "policy_runtime": dict(adapter.runtime_manifest),
        "task_id": "PickCube-v1",
        "split_role": split_role,
        "scheduled_episode_count": len(rows),
        "maximum_steps": args.maximum_steps,
        "execution_horizon": args.execution_horizon,
        "evaluation_git_commit": args.evaluation_git_commit,
        "environment_kwargs": kwargs,
        "final_authorization": final_authorization,
        "action_clipping": False,
        "action_projection": False,
        "action_replacement": False,
        "invalid_output_policy": "hard_reject",
    }
    manifest = {
        **manifest_semantic,
        "fingerprint": canonical_fingerprint(manifest_semantic),
    }
    _write_new_or_equal(output / "runtime_manifest.json", manifest)
    episodes_path = output / "episodes.jsonl"
    existing = _read_jsonl(episodes_path)
    expected_ids = [str(row["evaluation_id"]) for row in rows]
    existing_ids = [str(row.get("evaluation_id")) for row in existing]
    if (
        len(existing_ids) != len(set(existing_ids))
        or existing_ids != expected_ids[: len(existing_ids)]
    ):
        raise Phase2CCContractError("existing evaluation log is not a valid schedule prefix")
    try:
        for row in rows[len(existing) :]:
            instruction = str(
                row.get("language_instruction") or schedule.get("canonical_instruction")
            )
            evaluated = evaluator.run_episode(
                policy=adapter,
                schedule=row,
                task_condition_id="PickCube-v1",
                language_instruction=instruction,
                capture_video=False,
            )
            value = evaluated.result.to_dict()
            value.update(
                {
                    "formulation": args.formulation,
                    "split_role": split_role,
                    "instruction_sha256": (f"sha256:{sha256_hex({'instruction': instruction})}"),
                    "entered_grasp_region": environment.entered_grasp_region,
                    "ever_grasped": environment.ever_grasped,
                    "ever_lifted": environment.ever_lifted,
                }
            )
            _append_jsonl(episodes_path, value)
            existing.append(value)
    finally:
        adapter.close()
        environment.close()
    if len(existing) != len(rows):
        raise Phase2CCContractError("evaluation did not complete its frozen schedule")
    outcomes = Counter(str(row.get("outcome")) for row in existing)
    categories = Counter(str(row.get("failure_category")) for row in existing)
    base_summary = summarize_evaluation(existing)
    summary_semantic: dict[str, object] = {
        **base_summary,
        "schema_version": "langmani-v2-phase2c-c-evaluation-summary-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "formulation": args.formulation,
        "model_kind": (
            "pick_smolvla_absolute" if args.formulation == "absolute" else "pick_smolvla_relative"
        ),
        "task_id": "PickCube-v1",
        "split_role": split_role,
        "execution_horizon": args.execution_horizon,
        "entered_grasp_region_count": sum(
            bool(row.get("entered_grasp_region")) for row in existing
        ),
        "ever_grasped_count": sum(bool(row.get("ever_grasped")) for row in existing),
        "ever_lifted_count": sum(bool(row.get("ever_lifted")) for row in existing),
        "failed_grasp_count": categories["failed_grasp"],
        "no_motion_count": categories["no_initial_motion"],
        "object_drop_count": categories["object_drop"],
        "timeout_count": outcomes["timeout"],
        "invalid_action_count": sum(bool(row.get("invalid_action")) for row in existing),
        "simulator_error_count": sum(bool(row.get("simulator_error")) for row in existing),
        "runtime_manifest_fingerprint": manifest["fingerprint"],
        "schedule_fingerprint": schedule_fingerprint,
        "final_authorization": final_authorization,
        "completed": True,
    }
    summary = {
        **summary_semantic,
        "fingerprint": canonical_fingerprint(summary_semantic),
    }
    _write_new_or_equal(output / "summary.json", summary)
    completion_semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-c-evaluation-complete-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "runtime_manifest_fingerprint": manifest["fingerprint"],
        "summary_fingerprint": summary["fingerprint"],
        "episode_count": len(existing),
        "runtime_manifest": "runtime_manifest.json",
        "episodes": "episodes.jsonl",
        "summary": "summary.json",
        "passed": True,
    }
    _write_new_or_equal(
        output / "complete.json",
        {
            **completion_semantic,
            "fingerprint": canonical_fingerprint(completion_semantic),
        },
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
