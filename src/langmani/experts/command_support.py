"""Small command-boundary helpers for M2 diagnostics and benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym

import langmani.environments  # noqa: F401 - explicit registration boundary
from langmani.environments.pick_place_by_instruction import ENV_ID
from langmani.environments.specs import EpisodeSpec, TaskSpec
from langmani.experts.types import ExpertPhase, ExpertResult, ExpertStatus


def validated_output_path(path: str | Path, *, output_root: str | Path) -> Path:
    """Resolve a command artifact path and require it to stay under ``outputs/``."""
    allowed_root = Path(output_root).resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(allowed_root)
    except ValueError as error:
        raise ValueError(f"diagnostic output must stay under {allowed_root}") from error
    if resolved.suffix.lower() != ".json":
        raise ValueError("diagnostic output path must use a .json suffix")
    return resolved


def task_reset_options(task_spec: TaskSpec) -> dict[str, dict[str, str]]:
    """Return the exact M1 reset option shape for one validated task."""
    return {"task_spec": task_spec.to_dict()}


def create_expert_environment(*, diagnostic_rendering: bool, sim_backend: str) -> Any:
    """Create the one supported M2 environment/control combination."""
    return gym.make(
        ENV_ID,
        num_envs=1,
        obs_mode="none",
        reward_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array" if diagnostic_rendering else None,
        sim_backend=sim_backend,
        render_backend="sapien_cuda" if diagnostic_rendering else "none",
    )


def generic_unexpected_result(
    error: Exception,
    *,
    scene_seed: int | None = None,
    task_spec: TaskSpec | None = None,
) -> ExpertResult:
    """Preserve an exception raised before a PickPlaceExpert was constructed."""
    episode = (
        EpisodeSpec.create(scene_seed=scene_seed, task_spec=task_spec)
        if scene_seed is not None and task_spec is not None
        else None
    )
    message = str(error) or repr(error)
    return ExpertResult(
        success=False,
        status=ExpertStatus.UNEXPECTED_EXCEPTION,
        scene_seed=episode.scene_seed if episode is not None else None,
        scene_id=episode.scene_id if episode is not None else None,
        task_id=episode.task_id if episode is not None else None,
        canonical_instruction=episode.canonical_instruction if episode is not None else None,
        target_object_id=episode.task_spec.target_object_id if episode is not None else None,
        target_bin_id=episode.task_spec.target_bin_id if episode is not None else None,
        total_environment_steps=0,
        total_planning_calls=0,
        total_replans=0,
        completed_phases=(),
        failed_phase=ExpertPhase.INITIALIZE,
        final_environment_evaluation={},
        phase_results=(),
        planning_duration_seconds=0.0,
        execution_duration_seconds=0.0,
        exception_type=type(error).__name__,
        exception_message=message,
    )


def write_json(
    path: str | Path,
    payload: object,
    *,
    output_root: str | Path,
) -> Path:
    """Write one UTF-8, human-readable diagnostic JSON artifact."""
    destination = validated_output_path(path, output_root=output_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


__all__ = [
    "create_expert_environment",
    "generic_unexpected_result",
    "task_reset_options",
    "validated_output_path",
    "write_json",
]
