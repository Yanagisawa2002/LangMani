"""Run and immutably lock the M4.2 development-only runtime ablations."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import traceback
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from langmani.datasets.identity import sha256_hex
from langmani.environments.pick_place_by_instruction import ENV_ID
from langmani.policies.act_runtime import atomic_write_json, inspect_git_state
from langmani.policies.m42_analysis import (
    create_gripper_selection,
    create_horizon_selection,
)
from langmani.policies.m42_evidence import PriorM4Evidence, load_prior_m4_evidence
from langmani.policies.m42_schedule import (
    M42_DEV_SCHEDULE_FINGERPRINT,
    M42_DEV_SCHEDULE_ID,
    M42_FINAL_SCHEDULE_FINGERPRINT,
    M42_MAXIMUM_EPISODE_STEPS,
    M42ScheduledEpisode,
    load_locked_schedule,
    materialize_schedule,
)
from langmani.policies.m42_training import (
    RUNTIME_SELECTION_SCHEMA,
    TaskTokenFairComparisonContract,
    build_task_token_fair_comparison_contract,
    canonical_fingerprint,
)
from langmani.policies.m42_types import (
    GripperRuntimeMode,
    M42ExperimentManifest,
    M42SelectionRecord,
    M42Stage,
    RuntimeAblationConfig,
    RuntimeAblationResult,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "outputs" / "datasets" / "m3b" / "langmani-pick-place-lerobot-v1"
)
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "models" / "act"
DEFAULT_M4_DIAGNOSTICS_ROOT = PROJECT_ROOT / "outputs" / "diagnostics" / "m4"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "diagnostics" / "m42" / "runtime_ablation"
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "diagnostics" / "m42" / "runtime_ablation_command.json"

COMMAND_SCHEMA = "langmani-m42-runtime-ablation-command-v0"
BENCHMARK_ARTIFACT_SCHEMA = "langmani-m42-runtime-benchmark-artifact-v0"
POST_GRASP_SCHEMA = "langmani-m42-post-grasp-analysis-v0"
BENCHMARK_OWNER_FILE = "owner.json"
BENCHMARK_REPORT_FILE = "benchmark.json"
BENCHMARK_COMPLETE_FILE = "complete.json"
RUNTIME_IMPLEMENTATION_SCHEMA = "langmani-m42-runtime-implementation-v0"
EXPERIMENT_MANIFEST_FILE = "experiment_manifest.json"

PROTECTED_SOURCE_ROOTS = (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "environment",
    PROJECT_ROOT / "tests",
    PROJECT_ROOT / "docs",
    PROJECT_ROOT / ".git",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--m4-diagnostics-root", type=Path, default=DEFAULT_M4_DIAGNOSTICS_ROOT)
    parser.add_argument(
        "--schedule",
        choices=(M42_DEV_SCHEDULE_ID,),
        default=M42_DEV_SCHEDULE_ID,
        help="M4.2 runtime selection is development-only; m42_final_v0 is never accepted.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or path.is_junction():
        raise RuntimeError(f"refusing linked JSON artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact must contain one object: {path}")
    return value


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _overlaps(left: Path, right: Path) -> bool:
    return _is_within(left, right) or _is_within(right, left)


def _lexical_absolute(path: Path) -> Path:
    """Return an absolute path without following a symlink or junction."""

    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _resolved_unlinked(path: Path, *, label: str) -> Path:
    """Resolve one path only after rejecting linked existing components."""

    lexical = _lexical_absolute(path)
    for component in (lexical, *lexical.parents):
        if component.is_symlink() or component.is_junction():
            raise RuntimeError(f"{label} traverses a symlink or junction: {component}")
    return lexical.resolve(strict=False)


def _protected_paths(args: argparse.Namespace) -> tuple[Path, ...]:
    return tuple(
        _resolved_unlinked(path, label="protected path")
        for path in (
            args.dataset_root,
            args.checkpoint_root,
            args.m4_diagnostics_root,
            *PROTECTED_SOURCE_ROOTS,
        )
    )


def _safe_report_path(args: argparse.Namespace) -> Path:
    output = _resolved_unlinked(args.output_root, label="runtime-ablation output")
    report = _resolved_unlinked(args.report, label="runtime-ablation command report")
    _resolved_unlinked(report.parent, label="runtime-ablation command-report parent")
    protected = _protected_paths(args)
    if _is_within(report, output):
        raise RuntimeError(
            "runtime-ablation command report must remain outside the evidence output"
        )
    if any(_is_within(report, path) for path in protected):
        raise RuntimeError(
            "runtime-ablation command report must not overwrite protected evidence or source content"
        )
    if report.exists() and not report.is_file():
        raise RuntimeError(f"runtime-ablation command report must be a real file: {report}")
    return report


def _validate_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    inputs = tuple(
        _resolved_unlinked(path, label="M4.2 immutable input")
        for path in (args.dataset_root, args.checkpoint_root, args.m4_diagnostics_root)
    )
    for path in inputs:
        if not path.is_dir():
            raise RuntimeError(f"M4.2 input must be a real directory: {path}")
    output = _resolved_unlinked(args.output_root, label="runtime-ablation output")
    report = _safe_report_path(args)
    protected = _protected_paths(args)
    if any(_overlaps(output, path) for path in protected):
        raise RuntimeError(
            "runtime-ablation output must not overlap immutable input, evidence, source, or Git content"
        )
    if output.exists() and not output.is_dir():
        raise RuntimeError("runtime-ablation output root must be a real directory")
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
    return output, report


def _write_or_validate(path: Path, payload: dict[str, object]) -> None:
    if path.is_file():
        if _read_object(path) != payload:
            raise RuntimeError(f"existing immutable M4.2 artifact differs: {path}")
        return
    atomic_write_json(path, payload, immutable=True)


def _write_command_report(path: Path, payload: dict[str, object]) -> None:
    path = _resolved_unlinked(path, label="runtime-ablation command report")
    _resolved_unlinked(path.parent, label="runtime-ablation command-report parent")
    if path.exists() and not path.is_file():
        raise RuntimeError(f"unsafe command report path: {path}")
    atomic_write_json(path, payload)


def _plan(episodes: Sequence[M42ScheduledEpisode]) -> dict[str, object]:
    representative = tuple(item for item in episodes if item.task_index == 2)
    if len(episodes) != 72 or len(representative) != 12:
        raise RuntimeError(
            "m42_dev_v0 must materialize exactly 72 mixed and 12 representative episodes"
        )
    return {
        "schedule_id": M42_DEV_SCHEDULE_ID,
        "scene_count": 12,
        "task_count": 6,
        "horizon_order": [10, 5, 1],
        "horizon_mixed_episode_count": 216,
        "horizon_representative_episode_count": 36,
        "horizon_physical_episode_count": 252,
        "gripper_modes": [GripperRuntimeMode.PROJECT.value, GripperRuntimeMode.BINARY.value],
        "gripper_mixed_logical_episode_count": 144,
        "gripper_representative_logical_episode_count": 24,
        "gripper_logical_episode_count": 168,
        "gripper_additional_physical_episode_count": 84,
        "total_logical_episode_count": 420,
        "total_unique_physical_episode_count": 336,
        "project_gripper_evidence_reuse": "content-bound selected-horizon evidence",
        "final_schedule_materialized": False,
    }


def _implementation_fingerprint(git_commit: str) -> str:
    files = (
        "src/langmani/policies/act_rollout.py",
        "src/langmani/policies/m42_evaluation.py",
        "src/langmani/policies/m42_runtime.py",
        "src/langmani/policies/m42_analysis.py",
        "scripts/run_m42_runtime_ablation.py",
    )
    return canonical_fingerprint(
        {
            "schema_version": RUNTIME_IMPLEMENTATION_SCHEMA,
            "git_commit": git_commit,
            "files": {
                name: f"sha256:{sha256_hex((PROJECT_ROOT / name).read_bytes().hex())}"
                for name in files
            },
        }
    )


def _ablation_config(
    *,
    evidence: PriorM4Evidence,
    schedule_fingerprint: str,
    horizons: tuple[int, ...],
    modes: tuple[GripperRuntimeMode, ...],
) -> RuntimeAblationConfig:
    return RuntimeAblationConfig(
        schedule_id=M42_DEV_SCHEDULE_ID,
        schedule_fingerprint=schedule_fingerprint,
        m3b_dataset_fingerprint=evidence.dataset_fingerprint,
        mixed_task_onehot_checkpoint_fingerprint=(
            evidence.mixed_task_onehot.checkpoint_fingerprint
        ),
        representative_per_task_checkpoint_fingerprint=(
            evidence.representative_per_task.checkpoint_fingerprint
        ),
        representative_task_id=cast(str, evidence.representative_per_task.task_id),
        horizons=horizons,
        gripper_modes=modes,
        maximum_episode_steps=M42_MAXIMUM_EPISODE_STEPS,
    )


def _benchmark_identity(
    *,
    checkpoint: object,
    model_label: str,
    task_id: str | None,
    schedule_fingerprint: str,
    episodes: Sequence[M42ScheduledEpisode],
    horizon: int,
    gripper_mode: GripperRuntimeMode,
    config_fingerprint: str,
) -> dict[str, object]:
    descriptor = checkpoint.descriptor
    git = inspect_git_state(PROJECT_ROOT)
    return {
        "schema_version": BENCHMARK_ARTIFACT_SCHEMA,
        "evaluation_git_commit": git.commit,
        "implementation_fingerprint": _implementation_fingerprint(git.commit),
        "schedule_id": M42_DEV_SCHEDULE_ID,
        "schedule_fingerprint": schedule_fingerprint,
        "model_label": model_label,
        "task_id": task_id,
        "run_fingerprint": descriptor.run_fingerprint,
        "checkpoint_fingerprint": descriptor.checkpoint_fingerprint,
        "execution_horizon": horizon,
        "gripper_mode": gripper_mode.value,
        "config_fingerprint": config_fingerprint,
        "maximum_episode_steps": M42_MAXIMUM_EPISODE_STEPS,
        "ordered_episode_indices": [item.episode_index for item in episodes],
        "ordered_scene_seeds": [item.scene_seed for item in episodes],
        "ordered_task_ids": [item.task_id for item in episodes],
        "final_schedule_accessed": False,
    }


def _runtime_result_from_dict(value: Mapping[str, object]) -> RuntimeAblationResult:
    return RuntimeAblationResult(
        config_fingerprint=cast(str, value["config_fingerprint"]),
        schedule_id=cast(str, value["schedule_id"]),
        schedule_fingerprint=cast(str, value["schedule_fingerprint"]),
        model_label=cast(str, value["model_label"]),
        task_id=cast(str | None, value.get("task_id")),
        execution_horizon=cast(int, value["execution_horizon"]),
        gripper_mode=GripperRuntimeMode(cast(str, value["gripper_mode"])),
        episode_count=cast(int, value["episode_count"]),
        successes=cast(int, value["successes"]),
        post_grasp_timeouts=cast(int, value["post_grasp_timeouts"]),
        wrong_object_interactions=cast(int, value["wrong_object_interactions"]),
        wrong_object_grasp_count=cast(int, value["wrong_object_grasp_count"]),
        wrong_object_in_target_bin_count=cast(int, value["wrong_object_in_target_bin_count"]),
        target_in_wrong_bin_count=cast(int, value["target_in_wrong_bin_count"]),
        target_off_table_count=cast(int, value["target_off_table_count"]),
        invalid_action_count=cast(int, value["invalid_action_count"]),
        successful_episode_steps=tuple(cast(list[int], value["successful_episode_steps"])),
        policy_query_count=cast(int, value["policy_query_count"]),
        release_sign_transitions=cast(int, value["release_sign_transitions"]),
        grasp_sign_transitions=cast(int, value["grasp_sign_transitions"]),
        unnecessary_gripper_sign_transitions=cast(
            int, value["unnecessary_gripper_sign_transitions"]
        ),
        action_metrics=cast(Mapping[str, object], value["action_metrics"]),
        latency_metrics=cast(Mapping[str, object], value["latency_metrics"]),
        report_fingerprint=cast(str | None, value.get("report_fingerprint")),
    )


def _staging_paths(output_root: Path, identity: Mapping[str, object]) -> tuple[Path, Path, str]:
    fingerprint = canonical_fingerprint(identity)
    token = fingerprint.removeprefix("sha256:")
    evidence_root = output_root / "evidence"
    if evidence_root.exists() and (
        evidence_root.is_symlink() or evidence_root.is_junction() or not evidence_root.is_dir()
    ):
        raise RuntimeError("runtime-ablation evidence root must be a real directory")
    evidence_root.mkdir(parents=True, exist_ok=True)
    return evidence_root / token, evidence_root / f".staging-{token}", fingerprint


def _reuse_completed(
    destination: Path, *, identity: Mapping[str, object], identity_fingerprint: str
) -> tuple[RuntimeAblationResult, dict[str, object]] | None:
    if not destination.exists():
        return None
    if destination.is_symlink() or destination.is_junction() or not destination.is_dir():
        raise RuntimeError(f"unsafe completed benchmark path: {destination}")
    complete = _read_object(destination / BENCHMARK_COMPLETE_FILE)
    artifact = _read_object(destination / BENCHMARK_REPORT_FILE)
    if (
        complete.get("identity_fingerprint") != identity_fingerprint
        or artifact.get("benchmark_identity") != dict(identity)
        or complete.get("artifact_fingerprint") != canonical_fingerprint(artifact)
        or complete.get("passed") is not True
    ):
        raise RuntimeError(f"completed benchmark differs from requested identity: {destination}")
    result = artifact.get("runtime_result")
    if not isinstance(result, dict):
        raise RuntimeError("completed benchmark lacks runtime_result")
    return _runtime_result_from_dict(result), artifact


def _clean_matching_staging(
    staging: Path, *, identity_fingerprint: str, evidence_root: Path
) -> None:
    if not staging.exists():
        return
    if (
        staging.is_symlink()
        or staging.is_junction()
        or not staging.is_dir()
        or staging.resolve().parent != evidence_root.resolve()
        or not staging.name.startswith(".staging-")
    ):
        raise RuntimeError(f"unsafe runtime-ablation staging path: {staging}")
    owner = _read_object(staging / BENCHMARK_OWNER_FILE)
    if owner.get("identity_fingerprint") != identity_fingerprint:
        raise RuntimeError("refusing to clean staging owned by another runtime identity")
    shutil.rmtree(staging)


def _run_or_reuse(
    *,
    env: object,
    checkpoint: object,
    episodes: Sequence[M42ScheduledEpisode],
    schedule_fingerprint: str,
    horizon: int,
    gripper_mode: GripperRuntimeMode,
    model_label: str,
    task_id: str | None,
    config_fingerprint: str,
    output_root: Path,
) -> tuple[RuntimeAblationResult, dict[str, object], str]:
    from langmani.policies.m42_evaluation import run_m42_checkpoint_benchmark

    identity = _benchmark_identity(
        checkpoint=checkpoint,
        model_label=model_label,
        task_id=task_id,
        schedule_fingerprint=schedule_fingerprint,
        episodes=episodes,
        horizon=horizon,
        gripper_mode=gripper_mode,
        config_fingerprint=config_fingerprint,
    )
    destination, staging, identity_fingerprint = _staging_paths(output_root, identity)
    reused = _reuse_completed(
        destination, identity=identity, identity_fingerprint=identity_fingerprint
    )
    if reused is not None:
        result, artifact = reused
        return result, artifact, str(destination.relative_to(output_root))
    evidence_root = destination.parent
    _clean_matching_staging(
        staging, identity_fingerprint=identity_fingerprint, evidence_root=evidence_root
    )
    staging.mkdir()
    atomic_write_json(
        staging / BENCHMARK_OWNER_FILE,
        {"identity_fingerprint": identity_fingerprint, "benchmark_identity": identity},
        immutable=True,
    )
    benchmark = run_m42_checkpoint_benchmark(
        env=env,
        checkpoint=checkpoint,
        episodes=episodes,
        schedule_fingerprint=schedule_fingerprint,
        execution_horizon=horizon,
        gripper_mode=gripper_mode,
        model_label=model_label,
        maximum_episode_steps=M42_MAXIMUM_EPISODE_STEPS,
    )
    result = benchmark.to_runtime_ablation_result(config_fingerprint)
    benchmark_payload = benchmark.to_dict()
    aggregate = benchmark_payload.get("aggregate")
    if not isinstance(aggregate, Mapping) or aggregate.get("episode_count") != len(episodes):
        raise RuntimeError("M4.2 benchmark returned an incomplete aggregate")
    artifact: dict[str, object] = {
        "schema_version": BENCHMARK_ARTIFACT_SCHEMA,
        "benchmark_identity": identity,
        "benchmark_fingerprint": benchmark.fingerprint,
        "benchmark": benchmark_payload,
        "runtime_result": result.to_dict(),
        "passed": True,
    }
    atomic_write_json(staging / BENCHMARK_REPORT_FILE, artifact, immutable=True)
    complete = {
        "schema_version": BENCHMARK_ARTIFACT_SCHEMA,
        "identity_fingerprint": identity_fingerprint,
        "artifact_fingerprint": canonical_fingerprint(artifact),
        "passed": True,
    }
    atomic_write_json(staging / BENCHMARK_COMPLETE_FILE, complete, immutable=True)
    if destination.exists():
        raise RuntimeError("benchmark destination appeared during atomic promotion")
    os.replace(staging, destination)
    return result, artifact, str(destination.relative_to(output_root))


def _selection_with_stable_lock(
    *,
    path: Path,
    factory: Any,
    mixed: tuple[RuntimeAblationResult, ...],
    representative: tuple[RuntimeAblationResult, ...],
) -> M42SelectionRecord:
    locked_at = datetime.now(UTC).isoformat()
    if path.is_file():
        existing = _read_object(path)
        value = existing.get("locked_at_utc")
        if not isinstance(value, str):
            raise RuntimeError(f"existing selection lacks locked_at_utc: {path}")
        locked_at = value
    selection = factory(
        mixed_results=mixed,
        representative_per_task_results=representative,
        locked_at_utc=locked_at,
    )
    _write_or_validate(path, selection.to_dict())
    return selection


def _metrics_present(result: RuntimeAblationResult) -> tuple[bool, bool]:
    metrics = result.action_metrics
    raw_fields = {
        "total_policy_actions",
        "raw_out_of_bounds_action_count",
        "raw_out_of_bounds_component_count",
        "maximum_raw_bound_excess",
        "raw_action_metrics_validated",
    }
    runtime_fields = {
        "arm_action_bounds_validated",
        "arm_projected_component_count",
        "runtime_projected_action_count",
        "runtime_projected_component_count",
        "maximum_runtime_projection_correction",
        "runtime_action_bounds_validated",
    }
    return raw_fields.issubset(metrics) and metrics.get(
        "raw_action_metrics_validated"
    ) is True, runtime_fields.issubset(metrics) and all(
        (
            metrics.get("runtime_action_bounds_validated") is True,
            metrics.get("arm_action_bounds_validated") is True,
            metrics.get("arm_projected_component_count") == 0,
        )
    )


def _post_grasp_payload(
    artifacts: Sequence[tuple[str, Mapping[str, object]]], *, schedule_fingerprint: str
) -> dict[str, object]:
    summaries: list[dict[str, object]] = []
    for relative_path, artifact in artifacts:
        benchmark = artifact.get("benchmark")
        if not isinstance(benchmark, Mapping):
            raise RuntimeError("benchmark artifact lacks benchmark payload")
        aggregate = benchmark.get("aggregate")
        episodes = benchmark.get("episodes")
        if not isinstance(aggregate, Mapping) or not isinstance(episodes, list):
            raise RuntimeError("benchmark lacks complete post-grasp analysis")
        phase_counts = aggregate.get("phase_counts")
        if not isinstance(phase_counts, Mapping) or sum(
            int(value) for value in phase_counts.values()
        ) != len(episodes):
            raise RuntimeError("benchmark post-grasp phases do not cover every episode")
        failure_count = sum(
            isinstance(item, Mapping) and item.get("post_grasp_failure_record") is not None
            for item in episodes
        )
        summaries.append(
            {
                "artifact": relative_path,
                "benchmark_fingerprint": artifact["benchmark_fingerprint"],
                "episode_count": len(episodes),
                "phase_counts": dict(phase_counts),
                "post_grasp_timeout_count": aggregate.get("post_grasp_timeout_count"),
                "failed_record_count": failure_count,
            }
        )
    return {
        "schema_version": POST_GRASP_SCHEMA,
        "schedule_id": M42_DEV_SCHEDULE_ID,
        "schedule_fingerprint": schedule_fingerprint,
        "unique_physical_benchmark_count": len(summaries),
        "benchmarks": summaries,
        "complete": True,
        "final_schedule_accessed": False,
    }


def _experiment_manifest(
    *,
    evidence: PriorM4Evidence,
    git_commit: str,
    horizon_selection: M42SelectionRecord,
    gripper_selection: M42SelectionRecord,
) -> M42ExperimentManifest:
    """Lock the shared provenance root consumed by every later M4.2 stage."""
    return M42ExperimentManifest(
        stage=M42Stage.RUNTIME_ABLATION,
        implementation_git_commit=git_commit,
        m3b_dataset_fingerprint=evidence.dataset_fingerprint,
        prior_m4_verification_fingerprint=evidence.verification_fingerprint,
        schedule_fingerprints={
            "m42_dev_v0": M42_DEV_SCHEDULE_FINGERPRINT,
            "m42_final_v0": M42_FINAL_SCHEDULE_FINGERPRINT,
        },
        m4_checkpoint_fingerprints=tuple(
            item.checkpoint_fingerprint for item in evidence.checkpoints
        ),
        m41_runtime_processor_fingerprint=evidence.m41_runtime_processor_fingerprint,
        runtime_selection_fingerprints={
            "execution_horizon": horizon_selection.fingerprint,
            "gripper_runtime": gripper_selection.fingerprint,
        },
        task_token_experiment_fingerprint=None,
        artifact_paths={
            "runtime_selection": "runtime_selection.json",
            "horizon_selection": "horizon_selection.json",
            "gripper_selection": "gripper_selection.json",
            "post_grasp_analysis": "post_grasp_analysis.json",
        },
        completed=True,
    )


def _task_token_fair_comparison(evidence: PriorM4Evidence) -> TaskTokenFairComparisonContract:
    """Bind TaskToken fairness to the real frozen State-OneHot run manifest."""

    source = evidence.mixed_task_onehot
    return build_task_token_fair_comparison_contract(
        source.manifest,
        selected_checkpoint_fingerprint=source.checkpoint_fingerprint,
    )


def _create_environment() -> object:
    import gymnasium as gym

    import langmani.environments  # noqa: F401 - registers ENV_ID

    return gym.make(
        ENV_ID,
        num_envs=1,
        obs_mode="rgb",
        reward_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
    )


def _load_contexts(evidence: PriorM4Evidence) -> tuple[object, object]:
    from langmani.policies.m42_evaluation import M42PolicyKind, load_m4_checkpoint_context

    mixed = evidence.mixed_task_onehot
    representative = evidence.representative_per_task
    return (
        load_m4_checkpoint_context(
            mixed.run_root,
            policy_kind=M42PolicyKind.STATE_ONEHOT,
            expected_checkpoint_fingerprint=mixed.checkpoint_fingerprint,
            expected_run_fingerprint=mixed.run_fingerprint,
            expected_dataset_fingerprint=evidence.dataset_fingerprint,
        ),
        load_m4_checkpoint_context(
            representative.run_root,
            policy_kind=M42PolicyKind.PER_TASK,
            expected_checkpoint_fingerprint=representative.checkpoint_fingerprint,
            expected_run_fingerprint=representative.run_fingerprint,
            expected_dataset_fingerprint=evidence.dataset_fingerprint,
            expected_task_id=representative.task_id,
        ),
    )


def execute(args: argparse.Namespace) -> dict[str, object]:
    if args.schedule != M42_DEV_SCHEDULE_ID:
        raise ValueError("runtime ablation accepts only m42_dev_v0")
    output_root, report_path = _validate_paths(args)
    git = inspect_git_state(PROJECT_ROOT)
    if not git.baseline_tracked or git.dirty:
        raise RuntimeError("M4.2 runtime evidence requires a clean tracked Git worktree")
    evidence = load_prior_m4_evidence(
        diagnostics_root=args.m4_diagnostics_root,
        checkpoint_root=args.checkpoint_root,
        dataset_root=args.dataset_root,
        validate_dataset_storage=not args.dry_run,
    )
    schedule = load_locked_schedule(M42_DEV_SCHEDULE_ID)
    episodes = materialize_schedule(schedule)
    plan = _plan(episodes)
    horizon_config = _ablation_config(
        evidence=evidence,
        schedule_fingerprint=schedule.schedule_fingerprint,
        horizons=(10, 5, 1),
        modes=(GripperRuntimeMode.PROJECT,),
    )
    base_report: dict[str, object] = {
        "schema_version": COMMAND_SCHEMA,
        "execution_mode": "dry_run" if args.dry_run else "physical",
        "git": git.to_dict(),
        "implementation_fingerprint": _implementation_fingerprint(git.commit),
        "schedule_id": schedule.schedule_id,
        "schedule_fingerprint": schedule.schedule_fingerprint,
        "prior_m4_evidence_fingerprint": evidence.fingerprint,
        "m41_runtime_processor_fingerprint": evidence.m41_runtime_processor_fingerprint,
        "m4_checkpoint_fingerprints": [
            item.checkpoint_fingerprint for item in evidence.checkpoints
        ],
        "m3b_dataset_fingerprint": evidence.dataset_fingerprint,
        "mixed_task_onehot_checkpoint_fingerprint": (
            evidence.mixed_task_onehot.checkpoint_fingerprint
        ),
        "representative_per_task_checkpoint_fingerprint": (
            evidence.representative_per_task.checkpoint_fingerprint
        ),
        "plan": plan,
        "horizon_config": horizon_config.to_dict(),
        "horizon_ablation_completed": False,
        "horizon_selection_locked": False,
        "gripper_ablation_completed": False,
        "gripper_selection_locked": False,
        "post_grasp_analysis_completed": False,
        "raw_action_metrics_validated": False,
        "runtime_action_metrics_validated": False,
        "final_schedule_accessed": False,
        "physical_execution": False,
        "passed": True,
    }
    if args.dry_run:
        _write_command_report(report_path, base_report)
        return base_report

    representative_episodes = tuple(
        item for item in episodes if item.task_id == evidence.representative_per_task.task_id
    )
    if len(representative_episodes) != 12:
        raise RuntimeError("representative PerTask schedule must contain exactly 12 episodes")
    mixed_context, representative_context = _load_contexts(evidence)
    env = _create_environment()
    horizon_mixed: list[RuntimeAblationResult] = []
    horizon_representative: list[RuntimeAblationResult] = []
    artifacts: list[tuple[str, Mapping[str, object]]] = []
    try:
        for horizon in (10, 5, 1):
            for context, selected_episodes, label, task_id, target in (
                (
                    mixed_context,
                    episodes,
                    "ACT-Mixed-TaskOneHot",
                    None,
                    horizon_mixed,
                ),
                (
                    representative_context,
                    representative_episodes,
                    "ACT-PerTask-Representative",
                    evidence.representative_per_task.task_id,
                    horizon_representative,
                ),
            ):
                result, artifact, relative = _run_or_reuse(
                    env=env,
                    checkpoint=context,
                    episodes=selected_episodes,
                    schedule_fingerprint=schedule.schedule_fingerprint,
                    horizon=horizon,
                    gripper_mode=GripperRuntimeMode.PROJECT,
                    model_label=label,
                    task_id=cast(str | None, task_id),
                    config_fingerprint=horizon_config.fingerprint,
                    output_root=output_root,
                )
                target.append(result)
                artifacts.append((relative, artifact))
        horizon_selection = _selection_with_stable_lock(
            path=output_root / "horizon_selection.json",
            factory=create_horizon_selection,
            mixed=tuple(horizon_mixed),
            representative=tuple(horizon_representative),
        )
        selected_horizon = int(horizon_selection.selected_value)
        project_mixed = next(
            item for item in horizon_mixed if item.execution_horizon == selected_horizon
        )
        project_representative = next(
            item for item in horizon_representative if item.execution_horizon == selected_horizon
        )
        gripper_config = _ablation_config(
            evidence=evidence,
            schedule_fingerprint=schedule.schedule_fingerprint,
            horizons=(selected_horizon,),
            modes=(GripperRuntimeMode.PROJECT, GripperRuntimeMode.BINARY),
        )
        binary_mixed, binary_mixed_artifact, binary_mixed_relative = _run_or_reuse(
            env=env,
            checkpoint=mixed_context,
            episodes=episodes,
            schedule_fingerprint=schedule.schedule_fingerprint,
            horizon=selected_horizon,
            gripper_mode=GripperRuntimeMode.BINARY,
            model_label="ACT-Mixed-TaskOneHot",
            task_id=None,
            config_fingerprint=gripper_config.fingerprint,
            output_root=output_root,
        )
        binary_representative, binary_rep_artifact, binary_rep_relative = _run_or_reuse(
            env=env,
            checkpoint=representative_context,
            episodes=representative_episodes,
            schedule_fingerprint=schedule.schedule_fingerprint,
            horizon=selected_horizon,
            gripper_mode=GripperRuntimeMode.BINARY,
            model_label="ACT-PerTask-Representative",
            task_id=cast(str, evidence.representative_per_task.task_id),
            config_fingerprint=gripper_config.fingerprint,
            output_root=output_root,
        )
        artifacts.extend(
            (
                (binary_mixed_relative, binary_mixed_artifact),
                (binary_rep_relative, binary_rep_artifact),
            )
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    gripper_mixed = (project_mixed, binary_mixed)
    gripper_representative = (project_representative, binary_representative)
    gripper_selection = _selection_with_stable_lock(
        path=output_root / "gripper_selection.json",
        factory=create_gripper_selection,
        mixed=gripper_mixed,
        representative=gripper_representative,
    )
    selected_mode = GripperRuntimeMode(gripper_selection.selected_value)
    post_grasp = _post_grasp_payload(artifacts, schedule_fingerprint=schedule.schedule_fingerprint)
    _write_or_validate(output_root / "post_grasp_analysis.json", post_grasp)
    all_results = tuple(horizon_mixed + horizon_representative) + (
        binary_mixed,
        binary_representative,
    )
    metric_checks = tuple(_metrics_present(item) for item in all_results)
    raw_metrics_valid = all(item[0] for item in metric_checks)
    runtime_metrics_valid = all(item[1] for item in metric_checks)
    fair_comparison = _task_token_fair_comparison(evidence)
    experiment_manifest = _experiment_manifest(
        evidence=evidence,
        git_commit=git.commit,
        horizon_selection=horizon_selection,
        gripper_selection=gripper_selection,
    )
    experiment_manifest_path = output_root / EXPERIMENT_MANIFEST_FILE
    _write_or_validate(experiment_manifest_path, experiment_manifest.to_dict())
    runtime_payload = {
        "schema_version": RUNTIME_SELECTION_SCHEMA,
        "execution_horizon": selected_horizon,
        "gripper_mode": selected_mode.value,
        "horizon_selection_fingerprint": horizon_selection.fingerprint,
        "gripper_selection_fingerprint": gripper_selection.fingerprint,
        "runtime_fingerprint": canonical_fingerprint(
            {
                "schema_version": "langmani-m42-selected-runtime-v0",
                "execution_horizon": selected_horizon,
                "gripper_mode": selected_mode.value,
                "horizon_config": {"actions_per_query": selected_horizon, "chunk_size": 50},
                "project_processor": "BoundedActionEnvPostprocessorV0:project",
                "binary_processor": (
                    "BinaryGripperEnvPostprocessorV0:component-7"
                    if selected_mode is GripperRuntimeMode.BINARY
                    else None
                ),
            }
        ),
        "development_schedule_fingerprint": schedule.schedule_fingerprint,
        "m3b_dataset_fingerprint": evidence.dataset_fingerprint,
        "mixed_task_onehot_checkpoint_fingerprint": (
            evidence.mixed_task_onehot.checkpoint_fingerprint
        ),
        "representative_per_task_checkpoint_fingerprint": (
            evidence.representative_per_task.checkpoint_fingerprint
        ),
        "implementation_fingerprint": _implementation_fingerprint(git.commit),
        "evaluation_git_commit": git.commit,
        "experiment_manifest_fingerprint": experiment_manifest.fingerprint,
        "selection_evidence": {
            "horizon_selection": "horizon_selection.json",
            "gripper_selection": "gripper_selection.json",
            "post_grasp_analysis": "post_grasp_analysis.json",
            "experiment_manifest": EXPERIMENT_MANIFEST_FILE,
            "task_token_fair_comparison_contract": fair_comparison.to_dict(),
            "project_evidence_reused_from_horizon": True,
        },
        "locked": True,
        "final_schedule_accessed": False,
    }
    _write_or_validate(output_root / "runtime_selection.json", runtime_payload)
    completed_report = {
        **base_report,
        "execution_mode": "physical",
        "gripper_config": gripper_config.to_dict(),
        "horizon_selection": horizon_selection.to_dict(),
        "horizon_selection_fingerprint": horizon_selection.fingerprint,
        "gripper_selection": gripper_selection.to_dict(),
        "gripper_selection_fingerprint": gripper_selection.fingerprint,
        "runtime_selection_fingerprint": canonical_fingerprint(runtime_payload),
        "experiment_manifest": EXPERIMENT_MANIFEST_FILE,
        "experiment_manifest_fingerprint": experiment_manifest.fingerprint,
        "task_token_fair_comparison_fingerprint": fair_comparison.contract_fingerprint,
        "evidence_artifacts": [path for path, _artifact in artifacts],
        "unique_physical_benchmark_count": len(artifacts),
        "unique_physical_episode_count": sum(item.episode_count for item in all_results),
        "horizon_ablation_completed": (
            sum(item.episode_count for item in horizon_mixed) == 216
            and sum(item.episode_count for item in horizon_representative) == 36
        ),
        "horizon_selection_locked": True,
        "gripper_ablation_completed": (
            project_mixed.episode_count + binary_mixed.episode_count == 144
            and project_representative.episode_count + binary_representative.episode_count == 24
        ),
        "gripper_selection_locked": True,
        "post_grasp_analysis_completed": True,
        "raw_action_metrics_validated": raw_metrics_valid,
        "runtime_action_metrics_validated": runtime_metrics_valid,
        "final_schedule_accessed": False,
        "physical_execution": True,
    }
    completed_report["passed"] = (
        all(
            completed_report[key] is True
            for key in (
                "horizon_ablation_completed",
                "horizon_selection_locked",
                "gripper_ablation_completed",
                "gripper_selection_locked",
                "post_grasp_analysis_completed",
                "raw_action_metrics_validated",
                "runtime_action_metrics_validated",
                "physical_execution",
            )
        )
        and completed_report["final_schedule_accessed"] is False
    )
    _write_command_report(report_path, cast(dict[str, object], completed_report))
    return cast(dict[str, object], completed_report)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = execute(args)
    except Exception as error:  # noqa: BLE001 - command boundary preserves the exact failure
        traceback.print_exc()
        failure = {
            "schema_version": COMMAND_SCHEMA,
            "passed": False,
            "error_type": type(error).__name__,
            "error_message": str(error) or repr(error),
            "final_schedule_accessed": False,
        }
        try:
            _write_command_report(_safe_report_path(args), failure)
        except Exception:  # noqa: BLE001 - original failure remains authoritative
            traceback.print_exc()
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
