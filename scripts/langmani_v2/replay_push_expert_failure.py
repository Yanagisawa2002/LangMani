"""Replay content-bound historical Push expert failures with step telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--git-repo", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--worker-case", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--keyframe-directory", type=Path)
    return parser.parse_args()


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, object], value)


def _write(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return cast(Mapping[str, object], value)


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a sequence")
    return cast(Sequence[object], value)


def _array(value: object) -> list[float]:
    import numpy as np

    raw = value
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu()
    result = np.asarray(raw, dtype=np.float64).reshape(-1)
    if not np.isfinite(result).all():
        raise ValueError("telemetry array contains nonfinite values")
    return [float(item) for item in result]


def _scalar(value: object) -> bool | int | float:
    raw = value
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu()
    if hasattr(raw, "numel") and raw.numel() != 1:
        raise ValueError("evaluation value is not scalar")
    if hasattr(raw, "item"):
        raw = raw.item()
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        return raw
    result = float(cast(float, raw))
    if not math.isfinite(result):
        raise ValueError("evaluation value is nonfinite")
    return result


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _save_frame(environment: object, path: Path) -> None:
    import numpy as np
    from PIL import Image

    raw_frame = cast(Any, environment).render()
    if hasattr(raw_frame, "detach"):
        raw_frame = raw_frame.detach().cpu()
    frame = np.asarray(raw_frame)
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        raise ValueError(f"render returned invalid shape {frame.shape}")
    if np.issubdtype(frame.dtype, np.floating):
        frame = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
    else:
        frame = frame.astype(np.uint8, copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path)


class _RecordingEnvironment:
    def __init__(self, environment: object, *, stride: int) -> None:
        import numpy as np

        self._environment = environment
        self._base = getattr(environment, "unwrapped", environment)
        self._stride = stride
        self._actions: list[list[float]] = []
        self.snapshots: list[dict[str, object]] = []
        self._low, self._high = cast(Any, self._base).get_push_expert_action_bounds()
        self._np = np

    @property
    def unwrapped(self) -> object:
        return self._base

    @property
    def actions(self) -> tuple[list[float], ...]:
        return tuple(self._actions)

    def __getattr__(self, name: str) -> object:
        return getattr(self._environment, name)

    def step(self, action: object) -> object:
        values = self._np.asarray(action, dtype=self._np.float64).reshape(-1)
        self._actions.append([float(item) for item in values])
        result = cast(Any, self._environment).step(action)
        if len(self._actions) % self._stride == 0:
            self.capture(len(self._actions), values)
        return result

    def capture(self, step: int, action: object | None = None) -> None:
        context = cast(Any, self._base).get_push_expert_task_context()
        evaluation = cast(Any, self._base).get_push_expert_evaluation()
        diagnostics = cast(Any, self._base).get_policy_rollout_diagnostics()
        target_pose = _array(context.target_object.actor.pose.raw_pose)
        target_position = target_pose[:3]
        target_center = _array(context.target_region_center)[:3]
        radius = float(context.target_object_planar_radius)
        from langmani.environments.push_to_region import WORKSPACE_BOUNDS_XY

        x_min, x_max, y_min, y_max = WORKSPACE_BOUNDS_XY
        workspace_margin = min(
            target_position[0] - radius - x_min,
            x_max - target_position[0] - radius,
            target_position[1] - radius - y_min,
            y_max - target_position[1] - radius,
        )
        tcp_position = _array(context.agent.tcp.pose.p)[:3]
        contact_distance = math.dist(tcp_position[:2], target_position[:2])
        action_values = None if action is None else self._np.asarray(action, dtype=self._np.float64)
        action_margin = (
            None
            if action_values is None
            else float(
                self._np.min(
                    self._np.minimum(action_values - self._low, self._high - action_values)
                )
            )
        )
        self.snapshots.append(
            {
                "step": step,
                "target_object_pose": target_pose,
                "object_poses": {
                    item.object_id: _array(item.actor.pose.raw_pose) for item in context.objects
                },
                "target_center": target_center,
                "robot_qpos": _array(context.robot.get_qpos()),
                "tcp_position": tcp_position,
                "target_linear_velocity": _array(context.target_object.actor.linear_velocity),
                "target_angular_velocity": _array(context.target_object.actor.angular_velocity),
                "workspace_margin": float(workspace_margin),
                "action_bound_margin": action_margin,
                "contact_proxy": contact_distance <= radius + 0.055,
                "contact_proxy_distance": contact_distance,
                "evaluation": {key: _scalar(value) for key, value in evaluation.items()},
                "policy_diagnostics_digest": _digest_json(
                    {key: _array(value) for key, value in diagnostics.items()}
                ),
            }
        )


def _digest_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _worker(args: argparse.Namespace) -> int:
    if args.worker_case is None or args.worker_output is None:
        raise ValueError("worker requires --worker-case and --worker-output")
    case = _load(args.worker_case)
    import gymnasium as gym

    import langmani.environments  # noqa: F401
    from langmani.environments.push_specs import PushTaskSpec
    from langmani.environments.push_to_region import ENV_ID
    from langmani.experts.push import PushToRegionExpert
    from langmani.experts.push_types import PushExpertConfig

    environment: object | None = None
    try:
        render_mode = "rgb_array" if args.keyframe_directory is not None else None
        environment = gym.make(
            ENV_ID,
            num_envs=1,
            obs_mode="state_dict",
            reward_mode="none",
            control_mode="pd_joint_pos",
            render_mode=render_mode,
            sim_backend="physx_cuda",
        )
        task = PushTaskSpec(
            str(case["object_id"]),
            str(case["target_region_id"]),
            str(case["difficulty"]),
        )
        cast(Any, environment).reset(
            seed=int(cast(int, case["seed"])), options={"task_spec": task.to_dict()}
        )
        recorder = _RecordingEnvironment(environment, stride=int(cast(int, case["trace_stride"])))
        recorder.capture(0)
        if args.keyframe_directory is not None:
            _save_frame(environment, args.keyframe_directory / "initial.png")
        expert = PushToRegionExpert(recorder, config=PushExpertConfig())
        try:
            result = expert.run()
        except Exception as error:  # noqa: BLE001 - exact historical error is evidence
            result = expert.unexpected_exception_result(error)
        if not recorder.snapshots or recorder.snapshots[-1]["step"] != len(recorder.actions):
            recorder.capture(len(recorder.actions))
        if args.keyframe_directory is not None:
            _save_frame(environment, args.keyframe_directory / "final.png")
        payload = {
            "schema_version": "langmani-v2-phase2b3-replay-worker-v0",
            "case": case,
            "result": result.to_dict(),
            "step_trace": recorder.snapshots,
            "expert_diagnostic_trace": [item.to_dict() for item in expert.diagnostic_trace],
            "approach_candidates": list(expert.approach_candidate_diagnostics),
            "action_count": len(recorder.actions),
            "action_digest": f"sha256:{_digest_json(recorder.actions)}",
            "teleport_or_state_mutation_used": False,
            "success_flag_mutation_used": False,
            "official_collection_episodes": 0,
            "optimizer_steps": 0,
        }
        _write(args.worker_output, payload)
        return 0
    finally:
        if environment is not None:
            cast(Any, environment).close()


def _selected_cases(
    registry: Mapping[str, object], config: Mapping[str, object]
) -> list[dict[str, object]]:
    records = [
        dict(_mapping(item, "record")) for item in _sequence(registry.get("records"), "records")
    ]
    failures = [item for item in records if item.get("episode_result") == "FAILURE"]
    selected: dict[tuple[str, int], dict[str, object]] = {}

    def add(item: dict[str, object]) -> bool:
        key = (str(item["candidate_id"]), int(cast(int, item["seed"])))
        if key in selected:
            return False
        selected[key] = item
        return True

    for item in failures:
        if item.get("candidate_id") == "CandidateE":
            add(item)
    for item in records:
        if item.get("candidate_id") == "CandidateP":
            add(item)
    selection = _mapping(config.get("selection"), "selection")
    f_limit = int(cast(int, selection.get("candidate_f_per_category", 5)))
    f_counts: Counter[str] = Counter()
    for item in failures:
        category = str(item.get("failure_category"))
        if item.get("candidate_id") == "CandidateF" and f_counts[category] < f_limit and add(item):
            f_counts[category] += 1
    target = int(cast(int, config.get("per_failure_category", 5)))
    category_counts = Counter(
        str(item.get("failure_category"))
        for item in selected.values()
        if item.get("episode_result") == "FAILURE"
    )
    for item in failures:
        category = str(item.get("failure_category"))
        if category_counts[category] < target and add(item):
            category_counts[category] += 1
    return sorted(
        selected.values(),
        key=lambda item: (str(item["candidate_id"]), int(cast(int, item["seed"]))),
    )


def _ensure_worktree(*, git_repo: Path, root: Path, commit: str) -> Path:
    worktree = root / commit[:12]
    if worktree.exists():
        observed = subprocess.run(
            ["git", "-C", str(worktree), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if observed != commit:
            raise RuntimeError(f"existing replay worktree mismatch: {worktree}")
    else:
        subprocess.run(
            ["git", "-C", str(git_repo), "worktree", "add", "--detach", str(worktree), commit],
            check=True,
        )
    dirty = subprocess.run(
        ["git", "-C", str(worktree), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"historical replay worktree is dirty: {worktree}")
    return worktree


def _target_pose_trace(raw: Mapping[str, object]) -> list[list[float]]:
    values: list[list[float]] = []
    for item in _sequence(raw.get("step_trace"), "step_trace"):
        snapshot = _mapping(item, "snapshot")
        values.append(
            [
                _number(value, "target pose")
                for value in _sequence(snapshot["target_object_pose"], "pose")
            ]
        )
    return values


def _maximum_trace_error(left: Mapping[str, object], right: Mapping[str, object]) -> float:
    left_trace = _target_pose_trace(left)
    right_trace = _target_pose_trace(right)
    if len(left_trace) != len(right_trace):
        return math.inf
    error = 0.0
    for left_pose, right_pose in zip(left_trace, right_trace, strict=True):
        if len(left_pose) != len(right_pose):
            return math.inf
        error = max(error, *(abs(a - b) for a, b in zip(left_pose, right_pose, strict=True)))
    return error


def _contact_events(trace: Sequence[object]) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    previous = False
    for raw in trace:
        item = _mapping(raw, "snapshot")
        current = item.get("contact_proxy") is True
        if current != previous:
            events.append(
                {
                    "step": item.get("step"),
                    "event": "CONTACT_ESTABLISHED" if current else "CONTACT_LOST",
                }
            )
        previous = current
    return events


def _case_summary(
    *, case: Mapping[str, object], repetitions: Sequence[Mapping[str, object]], tolerance: float
) -> dict[str, object]:
    first = repetitions[0]
    first_result = _mapping(first.get("result"), "result")
    semantics = [
        (
            _mapping(item.get("result"), "result").get("success"),
            _mapping(item.get("result"), "result").get("status"),
            _mapping(item.get("result"), "result").get("failed_phase"),
            _mapping(item.get("result"), "result").get("total_environment_steps"),
        )
        for item in repetitions
    ]
    errors = [_maximum_trace_error(first, item) for item in repetitions[1:]]
    maximum_error = max(errors, default=0.0)
    original_matches = all(
        semantic[0] == (case.get("episode_result") == "SUCCESS")
        and semantic[1] == case.get("termination_reason")
        and semantic[3] == case.get("episode_steps")
        for semantic in semantics
    )
    trace = _sequence(first.get("step_trace"), "step_trace")
    snapshots = [_mapping(item, "snapshot") for item in trace]
    margins = [float(cast(float, item["workspace_margin"])) for item in snapshots]
    action_margins = [
        float(cast(float, item["action_bound_margin"]))
        for item in snapshots
        if item.get("action_bound_margin") is not None
    ]
    initial_pose = [
        _number(value, "target pose")
        for value in _sequence(snapshots[0]["target_object_pose"], "pose")
    ]
    displacements = []
    for item in snapshots:
        pose = [
            _number(value, "target pose") for value in _sequence(item["target_object_pose"], "pose")
        ]
        displacements.append(math.dist(initial_pose[:2], pose[:2]))
    return {
        "candidate_id": case.get("candidate_id"),
        "run_id": case.get("run_id"),
        "source_commit": case.get("source_commit"),
        "seed": case.get("seed"),
        "object_shape": case.get("object_shape"),
        "task": {
            "object_id": case.get("object_id"),
            "target_region_id": case.get("target_region_id"),
            "difficulty": case.get("difficulty"),
        },
        "historical_result": case.get("episode_result"),
        "historical_termination_reason": case.get("termination_reason"),
        "replay_semantics": [list(item) for item in semantics],
        "original_outcome_reproduced": original_matches,
        "repeat_semantics_identical": len(set(semantics)) == 1,
        "maximum_target_pose_repeat_error": maximum_error,
        "pose_repeat_within_tolerance": maximum_error <= tolerance,
        "initial_object_pose": snapshots[0].get("target_object_pose"),
        "target_pose": snapshots[0].get("target_center"),
        "robot_initial_state": {
            "qpos": snapshots[0].get("robot_qpos"),
            "tcp_position": snapshots[0].get("tcp_position"),
        },
        "minimum_workspace_margin": min(margins),
        "maximum_workspace_margin": max(margins),
        "minimum_action_bound_margin": min(action_margins) if action_margins else None,
        "final_object_target_distance": _mapping(
            first_result.get("final_environment_evaluation", {}), "evaluation"
        ).get("target_distance"),
        "maximum_object_displacement": max(displacements),
        "contact_events": _contact_events(trace),
        "workspace_violation_entity": (
            "TARGET_OBJECT"
            if _mapping(first_result.get("final_environment_evaluation", {}), "evaluation").get(
                "target_outside_workspace"
            )
            is True
            else None
        ),
    }


def _main(args: argparse.Namespace) -> int:
    required = (args.registry, args.config, args.git_repo, args.output_root, args.summary)
    if any(value is None for value in required):
        raise ValueError("main replay mode requires registry/config/git-repo/output-root/summary")
    registry = _load(args.registry)
    config = _load(args.config)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cases = _selected_cases(registry, config)
    repeats = int(cast(int, config.get("repeats", 2)))
    tolerance = float(cast(float, config.get("maximum_pose_absolute_error", 1e-5)))
    trace_stride = int(cast(int, config.get("trace_stride", 1)))
    capture_keyframes = config.get("capture_keyframes") is True
    worktrees: dict[str, Path] = {}
    summaries: list[dict[str, object]] = []
    script = Path(__file__).resolve()
    for case in cases:
        commit = str(case["source_commit"])
        if commit not in worktrees:
            worktrees[commit] = _ensure_worktree(
                git_repo=args.git_repo.resolve(),
                root=output_root / "source_worktrees",
                commit=commit,
            )
        worktree = worktrees[commit]
        repetitions: list[Mapping[str, object]] = []
        for repeat in range(repeats):
            case_id = f"{case['candidate_id']}_{case['seed']}_r{repeat}"
            case_path = output_root / "case_inputs" / f"{case_id}.json"
            output_path = output_root / "raw_replays" / f"{case_id}.json"
            worker_case = dict(case) | {"repeat": repeat, "trace_stride": trace_stride}
            _write(case_path, worker_case)
            command = [
                str(args.python_executable),
                str(script),
                "--worker-case",
                str(case_path),
                "--worker-output",
                str(output_path),
            ]
            if capture_keyframes and repeat == 0:
                command.extend(["--keyframe-directory", str(output_root / "keyframes" / case_id)])
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(worktree / "src")
            completed = subprocess.run(command, env=environment, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"replay worker failed for {case_id}: {completed.returncode}")
            repetitions.append(_load(output_path))
        summaries.append(_case_summary(case=case, repetitions=repetitions, tolerance=tolerance))
    passed = all(
        item["original_outcome_reproduced"] is True
        and item["repeat_semantics_identical"] is True
        and item["pose_repeat_within_tolerance"] is True
        for item in summaries
    )
    categories = Counter(
        str(item.get("failure_category"))
        for item in cases
        if item.get("episode_result") == "FAILURE"
    )
    report = {
        "schema_version": "langmani-v2-phase2b3-replay-validation-v0",
        "registry_digest": registry.get("records_digest"),
        "config_digest": f"sha256:{_digest_json(config)}",
        "repeats": repeats,
        "case_count": len(cases),
        "worker_episode_count": len(cases) * repeats,
        "failure_category_case_counts": dict(sorted(categories.items())),
        "maximum_pose_absolute_error": tolerance,
        "passed": passed,
        "cases": summaries,
        "raw_replays_retained_outside_git": True,
        "keyframes_retained_outside_git": capture_keyframes,
        "official_collection_episodes": 0,
        "optimizer_steps": 0,
    }
    _write(args.summary, report)
    print(json.dumps({key: report[key] for key in ("case_count", "passed")}, sort_keys=True))
    return 0 if passed else 2


def main() -> int:
    args = parse_args()
    if args.worker_case is not None:
        return _worker(args)
    return _main(args)


if __name__ == "__main__":
    raise SystemExit(main())
