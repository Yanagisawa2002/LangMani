"""Select and evaluate the M4.2 oracle-conditioned ACT controls.

Development is deliberately split into two causally ordered operations:

1. select the sole TaskToken checkpoint from the fixed 36-episode M3B
   validation view; and
2. compare frozen PerTask, State-OneHot, and the selected TaskToken checkpoint
   on ``m42_dev_v0``.

The sealed final schedule is accepted only by ``--stage final`` together with
an explicit authorization and a content-matching implementation fingerprint.
A final dry-run audits locks but never materializes a final episode.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from langmani.datasets.identity import sha256_hex
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.pick_place_by_instruction import ENV_ID
from langmani.policies.act_conditioning import (
    CANONICAL_TASK_IDS,
    append_canonical_task_onehot,
)
from langmani.policies.act_rollout import build_policy_observation
from langmani.policies.act_runtime import atomic_write_json, inspect_git_state
from langmani.policies.act_types import ActVariant
from langmani.policies.m42_analysis import (
    create_go_no_go_decision,
    create_task_token_checkpoint_selection,
)
from langmani.policies.m42_evaluation import (
    M42CheckpointContext,
    M42PolicyKind,
    load_m4_checkpoint_context,
    load_task_token_checkpoint_context,
    run_m42_checkpoint_benchmark,
    run_m42_validation_benchmark,
    task_action_chunk_distances,
    task_token_validation_result,
)
from langmani.policies.m42_evidence import (
    PriorM4Evidence,
    load_m42_experiment_manifest,
    load_prior_m4_evidence,
    validate_m42_experiment_manifest,
)
from langmani.policies.m42_schedule import (
    M42_DEV_SCHEDULE_ID,
    M42_FINAL_SCHEDULE_ID,
    M42_MAXIMUM_EPISODE_STEPS,
    FinalScheduleAuthorization,
    load_locked_schedule,
    materialize_schedule,
    validate_locked_schedules,
)
from langmani.policies.m42_training import (
    TaskTokenTrainingManifest,
    TaskTokenValidationQueue,
    TaskTokenValidationQueueItem,
    canonical_fingerprint,
    fingerprint_owned_task_token_runs,
    load_runtime_selection,
    validate_completed_task_token_run,
)
from langmani.policies.m42_types import (
    GripperRuntimeMode,
    M42FinalBenchmarkConfig,
    M42FinalBenchmarkResult,
    M42GoNoGoMetrics,
    M42SelectionKind,
    M42SelectionRecord,
    TaskTokenValidationResult,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "outputs" / "datasets" / "m3b" / "langmani-pick-place-lerobot-v1"
)
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "models" / "act"
DEFAULT_TASK_TOKEN_ROOT = PROJECT_ROOT / "outputs" / "models" / "act-task-token"
DEFAULT_M4_DIAGNOSTICS_ROOT = PROJECT_ROOT / "outputs" / "diagnostics" / "m4"
DEFAULT_RUNTIME_SELECTION = (
    PROJECT_ROOT / "outputs" / "diagnostics" / "m42" / "runtime_ablation" / "runtime_selection.json"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "diagnostics" / "m42" / "evaluation"
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "diagnostics" / "m42" / "evaluate_m42.json"

COMMAND_SCHEMA = "langmani-m42-evaluate-command-v0"
EVIDENCE_SCHEMA = "langmani-m42-evaluation-artifact-v0"
SELECTION_SCHEMA = "langmani-m42-task-token-selection-artifact-v0"
DEVELOPMENT_SCHEMA = "langmani-m42-development-comparison-v0"
DEVELOPMENT_COMPLETION_SCHEMA = "langmani-m42-development-complete-v0"
FINAL_SCHEMA = "langmani-m42-final-comparison-v1"
FINAL_ATTEMPT_SCHEMA = "langmani-m42-final-attempt-v1"
FINAL_ATTEMPT_COMPLETION_SCHEMA = "langmani-m42-final-attempt-complete-v0"
FINAL_ATTEMPT_INCOMPLETE_SCHEMA = "langmani-m42-final-attempt-incomplete-v0"
FINAL_IDENTITY_SCHEMA = "langmani-m42-final-execution-identity-v0"
FINAL_SENSITIVITY_SCHEMA = "langmani-m42-final-sensitivity-artifact-v0"
IMPLEMENTATION_SCHEMA = "langmani-m42-final-implementation-v0"

PROTECTED_SOURCE_ROOTS = (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "environment",
    PROJECT_ROOT / "tests",
    PROJECT_ROOT / "docs",
    PROJECT_ROOT / ".git",
)


@dataclass(slots=True)
class _CommandProgress:
    """Ephemeral facts about this process invocation, never reconstructed from disk."""

    final_schedule_accessed: bool = False
    physical_execution_started: bool = False
    completed_evaluation_artifact_count: int = 0
    final_attempt_relative_path: str | None = None


def _command_progress(args: argparse.Namespace) -> _CommandProgress:
    value = getattr(args, "_m42_command_progress", None)
    if not isinstance(value, _CommandProgress):
        value = _CommandProgress()
        args._m42_command_progress = value
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("development", "final"), required=True)
    parser.add_argument(
        "--schedule",
        choices=(M42_DEV_SCHEDULE_ID, M42_FINAL_SCHEDULE_ID),
        required=True,
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--task-token-root", type=Path, default=DEFAULT_TASK_TOKEN_ROOT)
    parser.add_argument("--m4-diagnostics-root", type=Path, default=DEFAULT_M4_DIAGNOSTICS_ROOT)
    parser.add_argument("--runtime-selection", type=Path, default=DEFAULT_RUNTIME_SELECTION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--authorize-sealed-final",
        action="store_true",
        help="Explicitly authorize physical m42_final_v0 materialization.",
    )
    parser.add_argument(
        "--expected-implementation-fingerprint",
        help="Frozen implementation digest printed by a final-stage dry-run.",
    )
    return parser.parse_args(argv)


def _read_object(path: Path, label: str = "JSON artifact") -> dict[str, Any]:
    if path.is_symlink() or path.is_junction() or not path.is_file():
        raise RuntimeError(f"unsafe or absent {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain one JSON object: {path}")
    return value


def _write_report(path: Path, value: dict[str, object]) -> None:
    path = _resolved_unlinked(path, label="M4.2 command report")
    _resolved_unlinked(path.parent, label="M4.2 command-report parent")
    if path.exists() and not path.is_file():
        raise RuntimeError(f"unsafe command report path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, value)


def _persist_development_success(
    *,
    output_root: Path,
    comparison: Mapping[str, object],
    completion: Mapping[str, object],
    passed: bool,
) -> tuple[Path | None, Path | None]:
    """Persist immutable completion evidence only after every development gate passes."""

    if not passed:
        return None, None
    if comparison.get("complete") is not True or completion.get("passed") is not True:
        raise RuntimeError("development success artifacts cannot encode an incomplete result")
    comparison_path = output_root / "development_comparison.json"
    completion_path = output_root / "development_complete.json"
    comparison_payload = dict(comparison)
    completion_payload = dict(completion)
    if comparison_path.is_file():
        if _read_object(comparison_path, "development comparison") != comparison_payload:
            raise RuntimeError("existing immutable development comparison differs")
    else:
        atomic_write_json(comparison_path, comparison_payload, immutable=True)
    if completion_path.is_file():
        if _read_object(completion_path, "development completion") != completion_payload:
            raise RuntimeError("existing immutable development completion differs")
    else:
        atomic_write_json(completion_path, completion_payload, immutable=True)
    return comparison_path, completion_path


def _safe_report_path(args: argparse.Namespace) -> Path:
    path = _resolved_unlinked(args.report, label="M4.2 command report")
    output = _resolved_unlinked(args.output_root, label="M4.2 evaluation output")
    _resolved_unlinked(path.parent, label="M4.2 command-report parent")
    protected = _protected_paths(args)
    if _is_within(path, output):
        raise RuntimeError("command report must remain outside the evaluation evidence output")
    if any(_is_within(path, root) for root in protected):
        raise RuntimeError(
            "command report path overlaps protected input, evidence, source, or Git content"
        )
    if path.exists() and not path.is_file():
        raise RuntimeError(f"command report path must be a real file: {path}")
    return path


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
            args.task_token_root,
            args.m4_diagnostics_root,
            args.runtime_selection,
            *PROTECTED_SOURCE_ROOTS,
        )
    )


def _validate_stage(stage: str, schedule: str) -> None:
    expected = M42_DEV_SCHEDULE_ID if stage == "development" else M42_FINAL_SCHEDULE_ID
    if schedule != expected:
        raise ValueError(f"{stage} stage requires --schedule {expected}")


def _validate_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    sources = tuple(
        _resolved_unlinked(path, label="M4.2 immutable input")
        for path in (
            args.dataset_root,
            args.checkpoint_root,
            args.task_token_root,
            args.m4_diagnostics_root,
        )
    )
    runtime_path = _resolved_unlinked(args.runtime_selection, label="runtime selection")
    if not args.dry_run:
        for source in sources:
            if not source.is_dir():
                raise RuntimeError(f"M4.2 input must be a real directory: {source}")
        if not runtime_path.is_file():
            raise RuntimeError(f"runtime selection must be a real file: {runtime_path}")
    output = _resolved_unlinked(args.output_root, label="M4.2 evaluation output")
    report = _safe_report_path(args)
    protected = _protected_paths(args)
    if any(_overlaps(output, path) for path in protected):
        raise RuntimeError(
            "M4.2 evaluation output must not overlap immutable input, evidence, source, or Git content"
        )
    if output.exists() and not output.is_dir():
        raise RuntimeError("M4.2 output root must be a real directory")
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
    return output, report


def _implementation_fingerprint(git_commit: str) -> str:
    files = (
        "src/langmani/policies/m42_evaluation.py",
        "src/langmani/policies/m42_analysis.py",
        "src/langmani/policies/m42_runtime.py",
        "src/langmani/policies/m42_schedule.py",
        "scripts/evaluate_m42.py",
    )
    digests = {
        name: f"sha256:{sha256_hex((PROJECT_ROOT / name).read_bytes().hex())}" for name in files
    }
    return canonical_fingerprint(
        {"schema_version": IMPLEMENTATION_SCHEMA, "git_commit": git_commit, "files": digests}
    )


def _queue_from_dict(value: Mapping[str, object]) -> TaskTokenValidationQueue:
    return TaskTokenValidationQueue.from_dict(value)


def _verify_runtime_selection_sources(runtime: object, source_path: Path) -> None:
    selection = cast(Any, runtime)
    evidence = selection.selection_evidence
    if not isinstance(evidence, Mapping):
        raise RuntimeError("runtime selection lacks immutable evidence references")
    parent = source_path.resolve().parent
    bindings = (
        (
            "horizon_selection",
            selection.horizon_selection_fingerprint,
            str(selection.execution_horizon),
        ),
        (
            "gripper_selection",
            selection.gripper_selection_fingerprint,
            selection.gripper_mode,
        ),
    )
    for key, expected_fingerprint, expected_value in bindings:
        relative = evidence.get(key)
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise RuntimeError(f"runtime selection has an unsafe {key} reference")
        path = (parent / relative).resolve()
        if path.parent != parent:
            raise RuntimeError(f"runtime selection {key} escaped its evidence directory")
        payload = _read_object(path, key)
        if canonical_fingerprint(payload) != expected_fingerprint:
            raise RuntimeError(f"runtime selection {key} fingerprint differs")
        if str(payload.get("selected_value")) != expected_value:
            raise RuntimeError(f"runtime selection {key} value differs")


def _validate_runtime_training_binding(
    *,
    evidence: object,
    runtime: object,
    experiment_manifest: object,
    manifest: object,
    queue: object,
) -> None:
    identity = cast(Any, manifest).identity
    if (
        identity.m3b_export_fingerprint != cast(Any, evidence).dataset_fingerprint
        or identity.runtime_selection_fingerprint != cast(Any, runtime).runtime_fingerprint
        or identity.runtime_selection_source_fingerprint != cast(Any, runtime).source_fingerprint
        or identity.experiment_manifest_fingerprint != cast(Any, experiment_manifest).fingerprint
        or cast(Any, queue).runtime_selection_fingerprint != cast(Any, runtime).runtime_fingerprint
        or cast(Any, queue).experiment_manifest_fingerprint
        != cast(Any, experiment_manifest).fingerprint
    ):
        raise RuntimeError("TaskToken run differs from the selected runtime or M3B dataset")


def _load_unique_task_token_run(
    root: Path,
) -> tuple[Path, TaskTokenTrainingManifest, TaskTokenValidationQueue]:
    runs = fingerprint_owned_task_token_runs(root)
    if len(runs) != 1:
        raise RuntimeError(
            "M4.2 evaluation requires exactly one fingerprint-owned TaskToken run, "
            f"complete or incomplete; found {len(runs)}"
        )
    run_root = runs[0]
    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink() or manifest_path.is_junction():
        raise RuntimeError("the sole TaskToken run lacks one real run manifest")
    manifest = TaskTokenTrainingManifest.from_dict(
        _read_object(manifest_path, "TaskToken run manifest")
    )
    if not manifest.training_complete:
        raise RuntimeError("the sole TaskToken run is incomplete and cannot be evaluated")
    completed = validate_completed_task_token_run(run_root, fresh_reload=True)
    return run_root, completed.manifest, completed.queue


def _validation_schedule(
    manifest: TaskTokenTrainingManifest,
) -> tuple[dict[str, object], ...]:
    data_contract = manifest.identity.to_dict().get("data_contract")
    if not isinstance(data_contract, Mapping):
        raise RuntimeError("TaskToken identity lacks a data contract")
    raw = data_contract.get("validation_schedule")
    if not isinstance(raw, list) or len(raw) != 36:
        raise RuntimeError("TaskToken identity must bind exactly 36 validation episodes")
    result: list[dict[str, object]] = []
    indices: list[int] = []
    for item in raw:
        if not isinstance(item, Mapping) or not isinstance(item.get("task_spec"), Mapping):
            raise RuntimeError("TaskToken validation schedule entry is malformed")
        indices.append(int(item["episode_index"]))
        # Preserve the exact canonical JSON record used for the queue's
        # schedule fingerprint; dropping provenance fields is not equivalent.
        result.append(cast(dict[str, object], dict(item)))
    if tuple(indices) != manifest.identity.ordered_validation_episode_indices:
        raise RuntimeError("TaskToken validation schedule order differs from its identity")
    return tuple(result)


def _run_validation_checkpoint(
    *,
    env: object,
    checkpoint: M42CheckpointContext,
    schedule_records: Sequence[Mapping[str, object]],
    validation_schedule_fingerprint: str,
    execution_horizon: int,
    gripper_mode: GripperRuntimeMode,
) -> object:
    """Narrow keyword-checked bridge to the validation-only rollout API."""
    return run_m42_validation_benchmark(
        env=env,
        checkpoint=checkpoint,
        schedule_records=schedule_records,
        validation_schedule_fingerprint=validation_schedule_fingerprint,
        execution_horizon=execution_horizon,
        gripper_mode=gripper_mode,
        maximum_episode_steps=M42_MAXIMUM_EPISODE_STEPS,
    )


def _task_token_context(
    run_root: Path,
    manifest: TaskTokenTrainingManifest,
    item: TaskTokenValidationQueueItem,
) -> M42CheckpointContext:
    record = next(
        (
            record
            for record in manifest.checkpoints
            if record.checkpoint_fingerprint == item.checkpoint_fingerprint
        ),
        None,
    )
    if (
        record is None
        or record.global_step != item.global_step
        or record.relative_path != item.checkpoint_relative_path
    ):
        raise RuntimeError("queued TaskToken checkpoint differs from the training manifest")
    return load_task_token_checkpoint_context(
        run_root,
        expected_identity=manifest.identity,
        checkpoint_relative_path=item.checkpoint_relative_path,
        expected_checkpoint_fingerprint=item.checkpoint_fingerprint,
        expected_architecture_fingerprint=manifest.identity.task_token_architecture_fingerprint,
    )


def _m4_contexts(
    evidence: PriorM4Evidence,
) -> tuple[tuple[M42CheckpointContext, ...], M42CheckpointContext]:
    per_task: list[M42CheckpointContext] = []
    for item in evidence.checkpoints[:6]:
        per_task.append(
            load_m4_checkpoint_context(
                item.run_root,
                policy_kind=M42PolicyKind.PER_TASK,
                expected_checkpoint_fingerprint=item.checkpoint_fingerprint,
                expected_run_fingerprint=item.run_fingerprint,
                expected_dataset_fingerprint=evidence.dataset_fingerprint,
                expected_task_id=item.task_id,
            )
        )
    onehot = evidence.mixed_task_onehot
    return tuple(per_task), load_m4_checkpoint_context(
        onehot.run_root,
        policy_kind=M42PolicyKind.STATE_ONEHOT,
        expected_checkpoint_fingerprint=onehot.checkpoint_fingerprint,
        expected_run_fingerprint=onehot.run_fingerprint,
        expected_dataset_fingerprint=evidence.dataset_fingerprint,
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


def _artifact_paths(output_root: Path, identity: Mapping[str, object]) -> tuple[Path, Path, str]:
    fingerprint = canonical_fingerprint(identity)
    token = fingerprint.removeprefix("sha256:")
    evidence_root = _resolved_unlinked(output_root / "evidence", label="M4.2 evidence root")
    if evidence_root.exists() and not evidence_root.is_dir():
        raise RuntimeError("M4.2 evidence root must be one real directory")
    evidence_root.mkdir(parents=True, exist_ok=True)
    destination = _resolved_unlinked(
        evidence_root / token, label="M4.2 completed evaluation artifact"
    )
    staging = _resolved_unlinked(
        evidence_root / f".staging-{token}", label="M4.2 evaluation staging"
    )
    if destination.parent != evidence_root or staging.parent != evidence_root:
        raise RuntimeError("M4.2 evaluation artifact escaped its evidence root")
    return destination, staging, fingerprint


def _run_or_reuse_artifact(
    *,
    output_root: Path,
    identity: Mapping[str, object],
    runner: Any,
) -> tuple[dict[str, Any], str]:
    destination, staging, identity_fingerprint = _artifact_paths(output_root, identity)
    if destination.exists():
        if destination.is_symlink() or destination.is_junction() or not destination.is_dir():
            raise RuntimeError(f"unsafe completed evaluation artifact: {destination}")
        complete = _read_object(destination / "complete.json", "evaluation completion")
        artifact = _read_object(destination / "artifact.json", "evaluation artifact")
        if (
            artifact.get("identity") != dict(identity)
            or complete.get("identity_fingerprint") != identity_fingerprint
            or complete.get("artifact_fingerprint") != canonical_fingerprint(artifact)
            or complete.get("passed") is not True
        ):
            raise RuntimeError("completed M4.2 evaluation differs from requested identity")
        return artifact, str(destination.relative_to(output_root))
    if staging.exists():
        if (
            staging.is_symlink()
            or staging.is_junction()
            or not staging.is_dir()
            or staging.resolve().parent != destination.parent.resolve()
        ):
            raise RuntimeError(f"unsafe M4.2 staging directory: {staging}")
        owner = _read_object(staging / "owner.json", "staging owner")
        if owner.get("identity_fingerprint") != identity_fingerprint:
            raise RuntimeError("refusing to clean staging for another evaluation identity")
        shutil.rmtree(staging)
    staging.mkdir()
    atomic_write_json(
        staging / "owner.json",
        {"identity_fingerprint": identity_fingerprint, "identity": dict(identity)},
        immutable=True,
    )
    payload = runner()
    if not isinstance(payload, dict):
        raise RuntimeError("M4.2 evaluation runner must return a JSON object")
    artifact: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA,
        "identity": dict(identity),
        "payload": payload,
        "passed": True,
    }
    atomic_write_json(staging / "artifact.json", artifact, immutable=True)
    atomic_write_json(
        staging / "complete.json",
        {
            "schema_version": EVIDENCE_SCHEMA,
            "identity_fingerprint": identity_fingerprint,
            "artifact_fingerprint": canonical_fingerprint(artifact),
            "passed": True,
        },
        immutable=True,
    )
    if destination.exists():
        raise RuntimeError("evaluation destination appeared during atomic promotion")
    os.replace(staging, destination)
    return artifact, str(destination.relative_to(output_root))


def _validation_result_from_dict(value: Mapping[str, object]) -> TaskTokenValidationResult:
    return TaskTokenValidationResult(
        checkpoint_fingerprint=cast(str, value["checkpoint_fingerprint"]),
        checkpoint_step=int(value["checkpoint_step"]),
        validation_schedule_fingerprint=cast(str, value["validation_schedule_fingerprint"]),
        validation_split_digest=cast(str, value["validation_split_digest"]),
        episode_count=int(value["episode_count"]),
        success_count=int(value["success_count"]),
        wrong_object_interaction_count=int(value["wrong_object_interaction_count"]),
        target_off_table_count=int(value["target_off_table_count"]),
        timeout_count=int(value["timeout_count"]),
        offline_validation_action_loss=float(value["offline_validation_action_loss"]),
        development_schedule_accessed=cast(bool, value.get("development_schedule_accessed", False)),
        final_schedule_accessed=cast(bool, value.get("final_schedule_accessed", False)),
        test_split_accessed=cast(bool, value.get("test_split_accessed", False)),
    )


def _selection_lock(
    path: Path, values: tuple[TaskTokenValidationResult, ...]
) -> M42SelectionRecord:
    locked_at = datetime.now(UTC).isoformat()
    if path.is_file():
        existing = _read_object(path, "TaskToken selection")
        record = existing.get("selection")
        if not isinstance(record, Mapping) or not isinstance(record.get("locked_at_utc"), str):
            raise RuntimeError("existing TaskToken selection is malformed")
        locked_at = cast(str, record["locked_at_utc"])
    selection = create_task_token_checkpoint_selection(values, locked_at_utc=locked_at)
    payload: dict[str, object] = {
        "schema_version": SELECTION_SCHEMA,
        "selection": selection.to_dict(),
        "selection_fingerprint": selection.fingerprint,
        "validation_result_fingerprints": [value.fingerprint for value in values],
        "selection_source": "m3b_validation_only",
        "test_split_accessed": False,
        "development_schedule_accessed": False,
        "final_schedule_accessed": False,
        "locked": True,
    }
    if path.is_file():
        if _read_object(path, "TaskToken selection") != payload:
            raise RuntimeError("existing immutable TaskToken checkpoint selection differs")
    else:
        atomic_write_json(path, payload, immutable=True)
    return selection


def _benchmark_identity(
    *,
    stage: str,
    schedule_fingerprint: str,
    checkpoint: M42CheckpointContext,
    ordered_episode_indices: Sequence[int],
    horizon: int,
    gripper: GripperRuntimeMode,
    model_label: str,
    experiment_manifest_fingerprint: str,
    runtime_selection_fingerprint: str,
    runtime_selection_source_fingerprint: str,
) -> dict[str, object]:
    git = inspect_git_state(PROJECT_ROOT)
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "evaluation_git_commit": git.commit,
        "implementation_fingerprint": _implementation_fingerprint(git.commit),
        "experiment_manifest_fingerprint": experiment_manifest_fingerprint,
        "stage": stage,
        "schedule_fingerprint": schedule_fingerprint,
        "model_label": model_label,
        "checkpoint_descriptor_fingerprint": checkpoint.descriptor.fingerprint,
        "checkpoint_fingerprint": checkpoint.descriptor.checkpoint_fingerprint,
        "ordered_episode_indices": list(ordered_episode_indices),
        "execution_horizon": horizon,
        "gripper_mode": gripper.value,
        "runtime_selection_fingerprint": runtime_selection_fingerprint,
        "runtime_selection_source_fingerprint": runtime_selection_source_fingerprint,
        "maximum_episode_steps": M42_MAXIMUM_EPISODE_STEPS,
    }


def _episodes_from_benchmark(value: Mapping[str, object]) -> list[Mapping[str, object]]:
    episodes = value.get("episodes")
    if not isinstance(episodes, list) or not all(isinstance(item, Mapping) for item in episodes):
        raise RuntimeError("M4.2 benchmark episode evidence is malformed")
    return cast(list[Mapping[str, object]], episodes)


def _benchmark_payload(artifact: Mapping[str, object]) -> Mapping[str, object]:
    payload = artifact.get("payload")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("benchmark"), Mapping):
        raise RuntimeError("M4.2 artifact lacks benchmark evidence")
    return cast(Mapping[str, object], payload["benchmark"])


def _combine_per_task(benchmarks: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(benchmarks) != 6:
        raise RuntimeError("PerTask aggregate requires exactly six benchmark reports")
    episodes = [
        episode for benchmark in benchmarks for episode in _episodes_from_benchmark(benchmark)
    ]
    if len(episodes) not in {72, 180}:
        raise RuntimeError("PerTask aggregate does not cover one complete M4.2 schedule")
    semantic: dict[tuple[str, str], bool] = {}
    projection = [0] * 8
    raw_projection = [0] * 8
    totals: dict[str, int] = {
        "successes": 0,
        "timeout_count": 0,
        "target_grasped_count": 0,
        "wrong_object_interaction_count": 0,
        "wrong_object_grasp_count": 0,
        "wrong_object_in_target_bin_count": 0,
        "target_in_wrong_bin_count": 0,
        "target_off_table_count": 0,
        "invalid_action_count": 0,
    }
    action_totals = {
        "arm_projected_component_count": 0,
        "gripper_projected_component_count": 0,
        "raw_out_of_bounds_component_count": 0,
        "binary_changed_command_count": 0,
    }
    per_task: dict[str, dict[str, object]] = {}
    successful_steps: list[int] = []
    raw_valid = True
    runtime_valid = True
    max_raw_excess = 0.0
    for benchmark in benchmarks:
        aggregate = benchmark.get("aggregate")
        if not isinstance(aggregate, Mapping):
            raise RuntimeError("PerTask benchmark lacks aggregate metrics")
        task_metrics = aggregate.get("per_task")
        actions = aggregate.get("action_metrics")
        if (
            not isinstance(task_metrics, Mapping)
            or len(task_metrics) != 1
            or not isinstance(actions, Mapping)
        ):
            raise RuntimeError("PerTask benchmark aggregate is malformed")
        per_task.update(cast(dict[str, dict[str, object]], dict(task_metrics)))
        for name in totals:
            if name == "wrong_object_grasp_count":
                continue
            totals[name] += int(aggregate.get(name, 0))
        for name in action_totals:
            action_totals[name] += int(actions.get(name, 0))
        counts = actions.get("per_action_dimension_projection_counts")
        raw_counts = actions.get("raw_per_action_dimension_violation_counts")
        if (
            not isinstance(counts, list)
            or len(counts) != 8
            or not isinstance(raw_counts, list)
            or len(raw_counts) != 8
        ):
            raise RuntimeError("PerTask action dimension metrics are malformed")
        projection = [left + int(right) for left, right in zip(projection, counts, strict=True)]
        raw_projection = [
            left + int(right) for left, right in zip(raw_projection, raw_counts, strict=True)
        ]
        raw_valid &= actions.get("raw_action_metrics_validated") is True
        runtime_valid &= actions.get("runtime_action_bounds_validated") is True
        max_raw_excess = max(max_raw_excess, float(actions.get("maximum_raw_bound_excess", 0.0)))
        successful_steps.extend(
            int(item)
            for item in cast(Sequence[object], aggregate.get("successful_episode_steps", ()))
        )
        for episode in _episodes_from_benchmark(benchmark):
            rollout = episode.get("rollout")
            if not isinstance(rollout, Mapping):
                raise RuntimeError("PerTask episode lacks rollout")
            key = (cast(str, rollout["scene_id"]), cast(str, rollout["task_id"]))
            if key in semantic:
                raise RuntimeError("PerTask aggregate has duplicate semantic episodes")
            semantic[key] = bool(rollout["success"])
            totals["wrong_object_grasp_count"] += int(bool(rollout["wrong_object_grasped_any"]))
    return {
        "episode_count": len(episodes),
        **totals,
        "success_rate": totals["successes"] / len(episodes),
        "per_task": per_task,
        "successful_episode_steps": successful_steps,
        "action_metrics": {
            **action_totals,
            "per_action_dimension_projection_counts": projection,
            "raw_per_action_dimension_violation_counts": raw_projection,
            "maximum_raw_bound_excess": max_raw_excess,
            "raw_action_metrics_validated": raw_valid,
            "runtime_action_bounds_validated": runtime_valid,
        },
        "semantic_success": [
            {"scene_id": key[0], "task_id": key[1], "success": value}
            for key, value in sorted(semantic.items())
        ],
    }


def _semantic_success(benchmark: Mapping[str, object]) -> dict[tuple[str, str], bool]:
    result: dict[tuple[str, str], bool] = {}
    for episode in _episodes_from_benchmark(benchmark):
        rollout = episode.get("rollout")
        if not isinstance(rollout, Mapping):
            raise RuntimeError("benchmark episode lacks rollout")
        key = (cast(str, rollout["scene_id"]), cast(str, rollout["task_id"]))
        if key in result:
            raise RuntimeError("benchmark contains duplicate semantic episodes")
        result[key] = bool(rollout["success"])
    return result


def _exact_paired_bootstrap_interval(
    differences: Sequence[int], *, confidence_level: float = 0.95
) -> dict[str, object]:
    """Return the exact empirical paired-bootstrap percentile interval.

    Each resample draws ``n`` observed paired differences with replacement.
    The three-point support (-1, 0, +1) lets us evaluate that bootstrap
    distribution deterministically instead of relying on a pseudo-random seed.
    """

    if not differences or any(value not in {-1, 0, 1} for value in differences):
        raise RuntimeError("paired bootstrap requires non-empty -1/0/+1 differences")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("paired bootstrap confidence level must be between zero and one")
    sample_size = len(differences)
    probabilities = {value: differences.count(value) / sample_size for value in (-1, 0, 1)}
    distribution: dict[int, float] = {0: 1.0}
    for _ in range(sample_size):
        updated: dict[int, float] = {}
        for current_sum, current_probability in distribution.items():
            for value, probability in probabilities.items():
                if probability:
                    updated[current_sum + value] = (
                        updated.get(current_sum + value, 0.0) + current_probability * probability
                    )
        distribution = updated

    def quantile(probability: float) -> float:
        cumulative = 0.0
        for value, mass in sorted(distribution.items()):
            cumulative += mass
            if cumulative >= probability:
                return value / sample_size
        return max(distribution) / sample_size

    tail = (1.0 - confidence_level) / 2.0
    return {
        "method": "exact_empirical_paired_bootstrap_v0",
        "confidence_level": confidence_level,
        "resample_size": sample_size,
        "lower": quantile(tail),
        "upper": quantile(1.0 - tail),
    }


def _paired(
    left: Mapping[tuple[str, str], bool], right: Mapping[tuple[str, str], bool]
) -> dict[str, object]:
    if left.keys() != right.keys() or not left:
        raise RuntimeError("paired comparison requires identical non-empty semantic episodes")
    keys = tuple(sorted(left))
    left_values = tuple(left[key] for key in keys)
    right_values = tuple(right[key] for key in keys)
    differences = tuple(
        int(left_value) - int(right_value)
        for left_value, right_value in zip(left_values, right_values, strict=True)
    )
    left_only = sum(a and not b for a, b in zip(left_values, right_values, strict=True))
    right_only = sum(b and not a for a, b in zip(left_values, right_values, strict=True))
    both = sum(a and b for a, b in zip(left_values, right_values, strict=True))
    return {
        "trials": len(keys),
        "left_successes": sum(left_values),
        "right_successes": sum(right_values),
        "left_only_successes": left_only,
        "right_only_successes": right_only,
        "both_successes": both,
        "neither_successes": len(keys) - left_only - right_only - both,
        "paired_success_rate_difference": (sum(left_values) - sum(right_values)) / len(keys),
        "paired_success_rate_difference_interval": _exact_paired_bootstrap_interval(differences),
    }


def _reset_components(context: M42CheckpointContext) -> None:
    for component in (
        context.loaded.policy,
        context.loaded.preprocessor,
        context.loaded.postprocessor,
    ):
        reset = getattr(component, "reset", None)
        if not callable(reset):
            raise RuntimeError("ACT sensitivity component lacks reset()")
        reset()


def _action_chunk(
    *,
    env: object,
    observation: object,
    context: M42CheckpointContext,
    task_id: str,
) -> np.ndarray:
    _reset_components(context)
    kind = context.descriptor.policy_kind
    variant = (
        ActVariant.MIXED_TASK_ONEHOT if kind is M42PolicyKind.STATE_ONEHOT else ActVariant.PER_TASK
    )
    raw = build_policy_observation(
        observation,
        base_environment=getattr(env, "unwrapped", env),
        variant=variant,
        task_id=task_id,
        task_conditioner=append_canonical_task_onehot
        if kind is M42PolicyKind.STATE_ONEHOT
        else None,
    )
    processed = context.loaded.preprocessor(raw)
    if not isinstance(processed, Mapping):
        raise RuntimeError("ACT sensitivity preprocessor returned a non-mapping")
    batch = cast(dict[str, torch.Tensor], dict(processed))
    if kind is M42PolicyKind.TASK_TOKEN:
        from langmani.policies.act_task_token import TASK_TOKEN_FEATURE_KEY, canonical_task_token

        state = batch.get("observation.state")
        if not isinstance(state, torch.Tensor):
            raise RuntimeError("TaskToken sensitivity state is absent")
        batch[TASK_TOKEN_FEATURE_KEY] = canonical_task_token(
            task_id, device=state.device
        ).unsqueeze(0)
    predicted = cast(Any, context.loaded.policy).predict_action_chunk(batch)
    action = context.loaded.postprocessor(predicted)
    if not isinstance(action, torch.Tensor):
        raise RuntimeError("ACT sensitivity postprocessor returned a non-tensor")
    result = action.detach().cpu().numpy()
    if result.shape != (1, 50, 8) or not np.all(np.isfinite(result)):
        raise RuntimeError("ACT sensitivity chunk must be finite [1,50,8]")
    return np.array(result[0], copy=True)


def _sensitivity(
    *,
    env: object,
    scene_seeds: Sequence[int],
    per_task: Sequence[M42CheckpointContext],
    onehot: M42CheckpointContext,
    task_token: M42CheckpointContext,
) -> dict[str, object]:
    if len(per_task) != 6:
        raise RuntimeError("sensitivity requires six PerTask checkpoints")
    scenes: list[dict[str, object]] = []
    for scene_seed in scene_seeds:
        chunks = {"per_task": {}, "state_onehot": {}, "task_token": {}}
        reference_state: np.ndarray | None = None
        for index, task_spec in enumerate(CANONICAL_TASK_SPECS):
            task_id = CANONICAL_TASK_IDS[index]
            observation, _ = cast(Any, env).reset(
                seed=scene_seed, options={"task_spec": task_spec.to_dict()}
            )
            state = np.asarray(
                getattr(env, "unwrapped", env).agent.robot.get_qpos().detach().cpu().numpy()[0],
                dtype=np.float32,
            )
            if reference_state is None:
                reference_state = state
            elif not np.allclose(state, reference_state, atol=1e-6, rtol=0):
                raise RuntimeError(
                    "counterfactual sensitivity resets changed the initial Panda state"
                )
            chunks["per_task"][task_id] = _action_chunk(
                env=env, observation=observation, context=per_task[index], task_id=task_id
            )
            chunks["state_onehot"][task_id] = _action_chunk(
                env=env, observation=observation, context=onehot, task_id=task_id
            )
            chunks["task_token"][task_id] = _action_chunk(
                env=env, observation=observation, context=task_token, task_id=task_id
            )
        oracle = task_action_chunk_distances(cast(Mapping[str, np.ndarray], chunks["per_task"]))
        onehot_result = task_action_chunk_distances(
            cast(Mapping[str, np.ndarray], chunks["state_onehot"])
        )
        token_result = task_action_chunk_distances(
            cast(Mapping[str, np.ndarray], chunks["task_token"])
        )
        denominator = float(oracle["mean_pair_distance"])
        if denominator <= 0:
            raise RuntimeError("PerTask sensitivity reference distance must be positive")
        scenes.append(
            {
                "scene_seed": scene_seed,
                "per_task_oracle": oracle,
                "state_onehot": onehot_result,
                "task_token": token_result,
                "state_onehot_ratio": float(onehot_result["mean_pair_distance"]) / denominator,
                "task_token_ratio": float(token_result["mean_pair_distance"]) / denominator,
            }
        )
    oracle_mean = float(
        np.mean([cast(float, item["per_task_oracle"]["mean_pair_distance"]) for item in scenes])
    )
    onehot_mean = float(
        np.mean([cast(float, item["state_onehot"]["mean_pair_distance"]) for item in scenes])
    )
    token_mean = float(
        np.mean([cast(float, item["task_token"]["mean_pair_distance"]) for item in scenes])
    )
    return {
        "scene_count": len(scenes),
        "distance_definition": "root_mean_square_over_postprocessed_50x8_action_chunk",
        "scenes": scenes,
        "per_task_oracle_mean_pair_distance": oracle_mean,
        "state_onehot_mean_pair_distance": onehot_mean,
        "task_token_mean_pair_distance": token_mean,
        "state_onehot_ratio_relative_to_per_task": onehot_mean / oracle_mean,
        "task_token_ratio_relative_to_per_task": token_mean / oracle_mean,
    }


def _metrics_valid(benchmark: Mapping[str, object]) -> tuple[bool, bool, bool]:
    aggregate = benchmark.get("aggregate")
    if not isinstance(aggregate, Mapping) or not isinstance(
        aggregate.get("action_metrics"), Mapping
    ):
        return False, False, False
    actions = cast(Mapping[str, object], aggregate["action_metrics"])
    phases = aggregate.get("phase_counts")
    post_grasp = isinstance(phases, Mapping) and sum(
        int(value) for value in phases.values()
    ) == int(aggregate.get("episode_count", -1))
    return (
        actions.get("raw_action_metrics_validated") is True,
        all(
            (
                actions.get("runtime_action_bounds_validated") is True,
                actions.get("arm_action_bounds_validated") is True,
                actions.get("arm_projected_component_count") == 0,
            )
        ),
        post_grasp,
    )


def _final_metric_evidence_complete(
    benchmark: Mapping[str, object],
) -> tuple[bool, bool, bool]:
    """Validate metric presence independently of whether a safety count is zero."""

    raw_safe, runtime_valid, post_grasp = _metrics_valid(benchmark)
    aggregate = benchmark.get("aggregate")
    if not isinstance(aggregate, Mapping) or not isinstance(
        aggregate.get("action_metrics"), Mapping
    ):
        return False, runtime_valid, post_grasp
    actions = cast(Mapping[str, object], aggregate["action_metrics"])
    count_names = (
        "nan_count",
        "inf_count",
        "malformed_action_count",
        "nonfinite_action_episode_count",
        "malformed_action_episode_count",
    )
    counts_complete = all(
        not isinstance(actions.get(name), bool)
        and isinstance(actions.get(name), int)
        and int(actions[name]) >= 0
        for name in count_names
    )
    raw_flag = actions.get("raw_action_metrics_validated")
    dimensions = actions.get("raw_per_action_dimension_violation_counts")
    raw_complete = (
        counts_complete
        and isinstance(raw_flag, bool)
        and isinstance(dimensions, list)
        and len(dimensions) == 8
        and all(not isinstance(value, bool) and int(value) >= 0 for value in dimensions)
        and np.isfinite(float(actions.get("maximum_raw_bound_excess", float("nan"))))
    )
    if raw_safe and raw_flag is not True:
        return False, runtime_valid, post_grasp
    return raw_complete, runtime_valid, post_grasp


def _classified_task_token_action_counts(
    benchmark: Mapping[str, object],
) -> tuple[int, int, int]:
    """Return exact safety counts while rejecting unclassified invalid actions."""

    aggregate = benchmark.get("aggregate")
    if not isinstance(aggregate, Mapping) or not isinstance(
        aggregate.get("action_metrics"), Mapping
    ):
        raise RuntimeError("TaskToken benchmark lacks aggregate action metrics")
    actions = cast(Mapping[str, object], aggregate["action_metrics"])

    def count(name: str) -> int:
        value = actions.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"TaskToken action metric {name} is malformed")
        return value

    nan_count = count("nan_count")
    inf_count = count("inf_count")
    malformed_count = count("malformed_action_count")
    nonfinite_episodes = count("nonfinite_action_episode_count")
    malformed_episodes = count("malformed_action_episode_count")
    classified_invalid = 0
    observed_nonfinite = 0
    observed_malformed = 0
    observed_invalid = 0
    for episode in _episodes_from_benchmark(benchmark):
        rollout = episode.get("rollout")
        if not isinstance(rollout, Mapping) or not isinstance(
            rollout.get("action_projection_summary"), Mapping
        ):
            raise RuntimeError("TaskToken episode lacks action-failure evidence")
        summary = cast(Mapping[str, object], rollout["action_projection_summary"])
        nonfinite = summary.get("any_nonfinite_action") is True
        malformed = summary.get("any_malformed_action") is True
        invalid = rollout.get("invalid_action") is True
        if nonfinite and malformed:
            raise RuntimeError("one invalid TaskToken action has two failure classifications")
        if (nonfinite or malformed) and not invalid:
            raise RuntimeError("TaskToken action-failure flag lacks invalid-action status")
        if invalid:
            observed_invalid += 1
            if not (nonfinite or malformed):
                raise RuntimeError("sealed final contains an unclassified invalid TaskToken action")
            classified_invalid += 1
        observed_nonfinite += int(nonfinite)
        observed_malformed += int(malformed)
    aggregate_invalid = aggregate.get("invalid_action_count")
    if (
        isinstance(aggregate_invalid, bool)
        or not isinstance(aggregate_invalid, int)
        or aggregate_invalid != observed_invalid
        or aggregate_invalid != classified_invalid
        or nonfinite_episodes != observed_nonfinite
        or malformed_episodes != observed_malformed
        or (nan_count + inf_count == 0) != (nonfinite_episodes == 0)
        or (malformed_count == 0) != (malformed_episodes == 0)
    ):
        raise RuntimeError("TaskToken invalid-action aggregate is inconsistent")
    expected_raw_safe = nan_count == inf_count == malformed_count == 0
    if actions.get("raw_action_metrics_validated") is not expected_raw_safe:
        raise RuntimeError("TaskToken raw-action safety flag is inconsistent with exact counts")
    return nan_count, inf_count, malformed_count


def _run_model_benchmark(
    *,
    output_root: Path,
    stage: str,
    schedule: object,
    episodes: Sequence[object],
    context: M42CheckpointContext,
    horizon: int,
    gripper: GripperRuntimeMode,
    model_label: str,
    env: object,
    final_authorization: FinalScheduleAuthorization | None,
    experiment_manifest_fingerprint: str,
    runtime_selection_fingerprint: str,
    runtime_selection_source_fingerprint: str,
) -> tuple[dict[str, Any], str]:
    identity = _benchmark_identity(
        stage=stage,
        schedule_fingerprint=cast(Any, schedule).schedule_fingerprint,
        checkpoint=context,
        ordered_episode_indices=[cast(Any, item).episode_index for item in episodes],
        horizon=horizon,
        gripper=gripper,
        model_label=model_label,
        experiment_manifest_fingerprint=experiment_manifest_fingerprint,
        runtime_selection_fingerprint=runtime_selection_fingerprint,
        runtime_selection_source_fingerprint=runtime_selection_source_fingerprint,
    )

    def run() -> dict[str, object]:
        benchmark = run_m42_checkpoint_benchmark(
            env=env,
            checkpoint=context,
            episodes=cast(Any, episodes),
            schedule_fingerprint=cast(Any, schedule).schedule_fingerprint,
            execution_horizon=horizon,
            gripper_mode=gripper,
            model_label=model_label,
            maximum_episode_steps=M42_MAXIMUM_EPISODE_STEPS,
            final_authorization=final_authorization,
        )
        return {"benchmark": benchmark.to_dict(), "benchmark_fingerprint": benchmark.fingerprint}

    return _run_or_reuse_artifact(output_root=output_root, identity=identity, runner=run)


def _base_report(
    *,
    args: argparse.Namespace,
    git: object,
    implementation_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": COMMAND_SCHEMA,
        "stage": args.stage,
        "schedule_id": args.schedule,
        "dry_run": bool(args.dry_run),
        "git": cast(Any, git).to_dict(),
        "implementation_fingerprint": implementation_fingerprint,
        "task_token_checkpoint_selected": False,
        "validation_only_selection_validated": False,
        "development_benchmark_completed": False,
        "final_benchmark_completed": False,
        "post_grasp_analysis_completed": False,
        "raw_action_metrics_validated": False,
        "runtime_action_metrics_validated": False,
        "go_no_go_decision_completed": False,
        "smolvla_go": False,
        "final_schedule_accessed": False,
        "physical_execution": False,
        "passed": True,
    }


def _require_development_without_final_access(
    *,
    schedule_id: str,
    validation_results: Sequence[TaskTokenValidationResult],
    benchmarks: Sequence[Mapping[str, object]],
) -> None:
    if (
        schedule_id != M42_DEV_SCHEDULE_ID
        or any(
            value.development_schedule_accessed
            or value.final_schedule_accessed
            or value.test_split_accessed
            for value in validation_results
        )
        or any(value.get("schedule_id") != M42_DEV_SCHEDULE_ID for value in benchmarks)
    ):
        raise RuntimeError("development acceptance detected sealed-final or locked-split access")


def _execute_development(
    *,
    args: argparse.Namespace,
    output_root: Path,
    report: dict[str, object],
    evidence: PriorM4Evidence,
    run_root: Path,
    manifest: TaskTokenTrainingManifest,
    queue: TaskTokenValidationQueue,
    horizon: int,
    gripper: GripperRuntimeMode,
) -> dict[str, object]:
    schedule = load_locked_schedule(M42_DEV_SCHEDULE_ID)
    validation_schedule = _validation_schedule(manifest)
    if (
        queue.ordered_validation_episode_indices
        != manifest.identity.ordered_validation_episode_indices
    ):
        raise RuntimeError("validation queue and TaskToken identity episode order disagree")
    env = _create_environment()
    validation_results: list[TaskTokenValidationResult] = []
    validation_artifacts: list[str] = []
    try:
        for item in queue.checkpoints:
            context = _task_token_context(run_root, manifest, item)
            identity = {
                "schema_version": EVIDENCE_SCHEMA,
                "stage": "validation_selection",
                "evaluation_git_commit": cast(Mapping[str, object], report["git"])["commit"],
                "implementation_fingerprint": report["implementation_fingerprint"],
                "experiment_manifest_fingerprint": (
                    manifest.identity.experiment_manifest_fingerprint
                ),
                "run_fingerprint": manifest.identity.run_fingerprint,
                "checkpoint_fingerprint": item.checkpoint_fingerprint,
                "checkpoint_step": item.global_step,
                "validation_schedule_fingerprint": queue.validation_schedule_fingerprint,
                "validation_split_digest": queue.split_digest,
                "execution_horizon": horizon,
                "gripper_mode": gripper.value,
                "maximum_episode_steps": M42_MAXIMUM_EPISODE_STEPS,
                "test_split_accessed": False,
                "development_schedule_accessed": False,
                "final_schedule_accessed": False,
            }

            def runner(
                context: M42CheckpointContext = context,
                item: TaskTokenValidationQueueItem = item,
            ) -> dict[str, object]:
                benchmark = cast(
                    Any,
                    _run_validation_checkpoint(
                        env=env,
                        checkpoint=context,
                        schedule_records=validation_schedule,
                        validation_schedule_fingerprint=queue.validation_schedule_fingerprint,
                        execution_horizon=horizon,
                        gripper_mode=gripper,
                    ),
                )
                result = task_token_validation_result(
                    benchmark,
                    checkpoint_step=item.global_step,
                    validation_split_digest=queue.split_digest,
                    offline_validation_action_loss=item.offline_validation_loss,
                )
                return {
                    "benchmark": benchmark.to_dict(),
                    "benchmark_fingerprint": benchmark.fingerprint,
                    "validation_result": result.to_dict(),
                    "validation_result_fingerprint": result.fingerprint,
                }

            artifact, relative = _run_or_reuse_artifact(
                output_root=output_root, identity=identity, runner=runner
            )
            payload = artifact.get("payload")
            if not isinstance(payload, Mapping) or not isinstance(
                payload.get("validation_result"), Mapping
            ):
                raise RuntimeError("validation artifact lacks ranking evidence")
            validation_results.append(
                _validation_result_from_dict(
                    cast(Mapping[str, object], payload["validation_result"])
                )
            )
            validation_artifacts.append(relative)
        values = tuple(validation_results)
        selection_path = output_root / "task_token_checkpoint_selection.json"
        selection = _selection_lock(selection_path, values)
        selected_item = next(
            item
            for item in queue.checkpoints
            if item.checkpoint_fingerprint == selection.selected_checkpoint_fingerprint
        )
        selected_context = _task_token_context(run_root, manifest, selected_item)
        # Development episodes are materialized only after the validation-only
        # checkpoint selection is immutable.
        episodes = materialize_schedule(schedule)
        per_task_contexts, onehot_context = _m4_contexts(evidence)
        onehot_artifact, onehot_path = _run_model_benchmark(
            output_root=output_root,
            stage="development",
            schedule=schedule,
            episodes=episodes,
            context=onehot_context,
            horizon=horizon,
            gripper=gripper,
            model_label="ACT-Mixed-TaskOneHot",
            env=env,
            final_authorization=None,
            experiment_manifest_fingerprint=manifest.identity.experiment_manifest_fingerprint,
            runtime_selection_fingerprint=manifest.identity.runtime_selection_fingerprint,
            runtime_selection_source_fingerprint=manifest.identity.runtime_selection_source_fingerprint,
        )
        token_artifact, token_path = _run_model_benchmark(
            output_root=output_root,
            stage="development",
            schedule=schedule,
            episodes=episodes,
            context=selected_context,
            horizon=horizon,
            gripper=gripper,
            model_label="ACT-Mixed-TaskToken",
            env=env,
            final_authorization=None,
            experiment_manifest_fingerprint=manifest.identity.experiment_manifest_fingerprint,
            runtime_selection_fingerprint=manifest.identity.runtime_selection_fingerprint,
            runtime_selection_source_fingerprint=manifest.identity.runtime_selection_source_fingerprint,
        )
        per_task_artifacts: list[dict[str, Any]] = []
        per_task_paths: list[str] = []
        for index, context in enumerate(per_task_contexts):
            selected_episodes = tuple(
                item for item in episodes if item.task_id == CANONICAL_TASK_IDS[index]
            )
            artifact, relative = _run_model_benchmark(
                output_root=output_root,
                stage="development",
                schedule=schedule,
                episodes=selected_episodes,
                context=context,
                horizon=horizon,
                gripper=gripper,
                model_label=f"ACT-PerTask:{CANONICAL_TASK_IDS[index]}",
                env=env,
                final_authorization=None,
                experiment_manifest_fingerprint=(manifest.identity.experiment_manifest_fingerprint),
                runtime_selection_fingerprint=manifest.identity.runtime_selection_fingerprint,
                runtime_selection_source_fingerprint=manifest.identity.runtime_selection_source_fingerprint,
            )
            per_task_artifacts.append(artifact)
            per_task_paths.append(relative)
        onehot_benchmark = _benchmark_payload(onehot_artifact)
        token_benchmark = _benchmark_payload(token_artifact)
        per_task_benchmarks = tuple(_benchmark_payload(item) for item in per_task_artifacts)
        per_task_aggregate = _combine_per_task(per_task_benchmarks)
        per_task_success = {
            (cast(str, item["scene_id"]), cast(str, item["task_id"])): bool(item["success"])
            for item in cast(list[Mapping[str, object]], per_task_aggregate["semantic_success"])
        }
        sensitivity = _sensitivity(
            env=env,
            scene_seeds=schedule.ordered_scene_seeds,
            per_task=per_task_contexts,
            onehot=onehot_context,
            task_token=selected_context,
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    onehot_semantic = _semantic_success(onehot_benchmark)
    token_semantic = _semantic_success(token_benchmark)
    benchmarks = (onehot_benchmark, token_benchmark, *per_task_benchmarks)
    _require_development_without_final_access(
        schedule_id=schedule.schedule_id,
        validation_results=validation_results,
        benchmarks=benchmarks,
    )
    checks = tuple(_metrics_valid(item) for item in benchmarks)
    validation_only_selection_validated = all(
        not value.development_schedule_accessed
        and not value.final_schedule_accessed
        and not value.test_split_accessed
        for value in validation_results
    )
    development_benchmark_completed = (
        int(cast(Mapping[str, object], token_benchmark["aggregate"])["episode_count"]) == 72
        and int(cast(Mapping[str, object], onehot_benchmark["aggregate"])["episode_count"]) == 72
        and int(per_task_aggregate["episode_count"]) == 72
    )
    post_grasp_analysis_completed = all(item[2] for item in checks)
    raw_action_metrics_validated = all(item[0] for item in checks)
    runtime_action_metrics_validated = all(item[1] for item in checks)
    development_passed = all(
        (
            len(validation_results) == 20,
            schedule.schedule_id == M42_DEV_SCHEDULE_ID,
            validation_only_selection_validated,
            development_benchmark_completed,
            post_grasp_analysis_completed,
            raw_action_metrics_validated,
            runtime_action_metrics_validated,
        )
    )
    comparison = {
        "schema_version": DEVELOPMENT_SCHEMA,
        "experiment_manifest_fingerprint": manifest.identity.experiment_manifest_fingerprint,
        "schedule_id": schedule.schedule_id,
        "schedule_fingerprint": schedule.schedule_fingerprint,
        "execution_horizon": horizon,
        "gripper_mode": gripper.value,
        "selected_task_token_checkpoint_fingerprint": selection.selected_checkpoint_fingerprint,
        "selection_fingerprint": selection.fingerprint,
        "models": {
            "state_onehot": onehot_benchmark,
            "task_token": token_benchmark,
            "per_task": {
                "benchmarks": list(per_task_benchmarks),
                "aggregate": per_task_aggregate,
            },
        },
        "paired_comparisons": {
            "task_token_vs_per_task": _paired(token_semantic, per_task_success),
            "task_token_vs_state_onehot": _paired(token_semantic, onehot_semantic),
            "state_onehot_vs_per_task": _paired(onehot_semantic, per_task_success),
        },
        "task_sensitivity": sensitivity,
        "selection_source": "m3b_validation_only",
        "test_split_accessed": False,
        "final_schedule_accessed": False,
        "complete": development_passed,
    }
    comparison_fingerprint = canonical_fingerprint(comparison)
    development_completion = {
        "schema_version": DEVELOPMENT_COMPLETION_SCHEMA,
        "experiment_manifest_fingerprint": manifest.identity.experiment_manifest_fingerprint,
        "development_schedule_fingerprint": schedule.schedule_fingerprint,
        "implementation_fingerprint": report["implementation_fingerprint"],
        "evaluation_git_commit": cast(Mapping[str, object], report["git"])["commit"],
        "runtime_selection_fingerprint": manifest.identity.runtime_selection_fingerprint,
        "runtime_selection_source_fingerprint": (
            manifest.identity.runtime_selection_source_fingerprint
        ),
        "task_token_run_fingerprint": manifest.identity.run_fingerprint,
        "task_token_selection_fingerprint": selection.fingerprint,
        "selected_task_token_checkpoint_fingerprint": selection.selected_checkpoint_fingerprint,
        "development_comparison_fingerprint": comparison_fingerprint,
        "final_schedule_accessed": False,
        "passed": development_passed,
    }
    comparison_path, completion_path = _persist_development_success(
        output_root=output_root,
        comparison=comparison,
        completion=development_completion,
        passed=development_passed,
    )
    return {
        **report,
        "validation_queue_fingerprint": queue.queue_fingerprint,
        "validation_checkpoint_count": len(validation_results),
        "validation_evidence_artifacts": validation_artifacts,
        "task_token_selection": selection.to_dict(),
        "task_token_selection_fingerprint": selection.fingerprint,
        "selected_task_token_checkpoint_fingerprint": selection.selected_checkpoint_fingerprint,
        "development_evidence_artifacts": {
            "state_onehot": onehot_path,
            "task_token": token_path,
            "per_task": per_task_paths,
        },
        "development_comparison": (
            str(comparison_path.relative_to(output_root)) if comparison_path is not None else None
        ),
        "development_comparison_fingerprint": comparison_fingerprint,
        "development_completion": (
            str(completion_path.relative_to(output_root)) if completion_path is not None else None
        ),
        "task_token_checkpoint_selected": True,
        "validation_only_selection_validated": validation_only_selection_validated,
        "development_benchmark_completed": development_benchmark_completed,
        "final_benchmark_completed": False,
        "post_grasp_analysis_completed": post_grasp_analysis_completed,
        "raw_action_metrics_validated": raw_action_metrics_validated,
        "runtime_action_metrics_validated": runtime_action_metrics_validated,
        "go_no_go_decision_completed": False,
        "smolvla_go": False,
        "final_schedule_accessed": False,
        "physical_execution": True,
        "passed": development_passed,
    }


def _validate_final_development_lock(
    *,
    development_root: Path,
    queue: TaskTokenValidationQueue,
    manifest: TaskTokenTrainingManifest,
    report: Mapping[str, object],
    horizon: int,
    gripper: GripperRuntimeMode,
) -> tuple[dict[str, Any], M42SelectionRecord, TaskTokenValidationQueueItem]:
    selection_artifact = _read_object(
        development_root / "task_token_checkpoint_selection.json",
        "development TaskToken selection",
    )
    raw_selection = selection_artifact.get("selection")
    if not isinstance(raw_selection, Mapping) or selection_artifact.get("locked") is not True:
        raise RuntimeError("final benchmark requires the immutable development selection")
    selection = M42SelectionRecord.from_dict(cast(Mapping[str, object], raw_selection))
    validation_fingerprints = selection_artifact.get("validation_result_fingerprints")
    if (
        selection.selection_kind is not M42SelectionKind.TASK_TOKEN_CHECKPOINT
        or selection_artifact.get("schema_version") != SELECTION_SCHEMA
        or selection_artifact.get("selection_fingerprint") != selection.fingerprint
        or validation_fingerprints != list(selection.evidence_fingerprints)
        or selection_artifact.get("selection_source") != "m3b_validation_only"
        or selection_artifact.get("test_split_accessed") is not False
        or selection_artifact.get("development_schedule_accessed") is not False
        or selection_artifact.get("final_schedule_accessed") is not False
        or selection.validation_split_digest != queue.split_digest
        or set(selection.candidate_values)
        != {item.checkpoint_fingerprint for item in queue.checkpoints}
    ):
        raise RuntimeError("TaskToken selection lock is inconsistent with validation evidence")
    selected_item = next(
        (
            item
            for item in queue.checkpoints
            if item.checkpoint_fingerprint == selection.selected_checkpoint_fingerprint
        ),
        None,
    )
    if selected_item is None:
        raise RuntimeError("selected TaskToken checkpoint is absent from the validation queue")
    development_comparison = _read_object(
        development_root / "development_comparison.json", "development comparison"
    )
    development_completion = _read_object(
        development_root / "development_complete.json", "development completion"
    )
    if (
        development_comparison.get("schema_version") != DEVELOPMENT_SCHEMA
        or development_comparison.get("experiment_manifest_fingerprint")
        != manifest.identity.experiment_manifest_fingerprint
        or development_comparison.get("complete") is not True
        or development_comparison.get("final_schedule_accessed") is not False
        or development_comparison.get("schedule_id") != M42_DEV_SCHEDULE_ID
        or development_comparison.get("selected_task_token_checkpoint_fingerprint")
        != selection.selected_checkpoint_fingerprint
        or development_comparison.get("selection_fingerprint") != selection.fingerprint
        or development_comparison.get("execution_horizon") != horizon
        or development_comparison.get("gripper_mode") != gripper.value
        or development_completion.get("schema_version") != DEVELOPMENT_COMPLETION_SCHEMA
        or development_completion.get("experiment_manifest_fingerprint")
        != manifest.identity.experiment_manifest_fingerprint
        or development_completion.get("passed") is not True
        or development_completion.get("final_schedule_accessed") is not False
        or development_completion.get("implementation_fingerprint")
        != report["implementation_fingerprint"]
        or development_completion.get("development_schedule_fingerprint")
        != development_comparison.get("schedule_fingerprint")
        or development_completion.get("runtime_selection_fingerprint")
        != manifest.identity.runtime_selection_fingerprint
        or development_completion.get("runtime_selection_source_fingerprint")
        != manifest.identity.runtime_selection_source_fingerprint
        or development_comparison.get("runtime_selection_source_fingerprint")
        != manifest.identity.runtime_selection_source_fingerprint
        or development_completion.get("task_token_run_fingerprint")
        != manifest.identity.run_fingerprint
        or development_completion.get("task_token_selection_fingerprint") != selection.fingerprint
        or development_completion.get("development_comparison_fingerprint")
        != canonical_fingerprint(development_comparison)
    ):
        raise RuntimeError("final benchmark requires a matching immutable development completion")
    return selection_artifact, selection, selected_item


def _final_execution_identity(
    *,
    schedule_fingerprint: str,
    implementation_fingerprint: str,
    git_commit: str,
    evidence: PriorM4Evidence,
    manifest: TaskTokenTrainingManifest,
    selection: M42SelectionRecord,
    selected_checkpoint_fingerprint: str,
    horizon: int,
    gripper: GripperRuntimeMode,
) -> dict[str, object]:
    return {
        "schema_version": FINAL_IDENTITY_SCHEMA,
        "experiment_manifest_fingerprint": manifest.identity.experiment_manifest_fingerprint,
        "final_schedule_fingerprint": schedule_fingerprint,
        "implementation_fingerprint": implementation_fingerprint,
        "evaluation_git_commit": git_commit,
        "prior_m4_evidence_fingerprint": evidence.fingerprint,
        "runtime_selection_fingerprint": manifest.identity.runtime_selection_fingerprint,
        "runtime_selection_source_fingerprint": (
            manifest.identity.runtime_selection_source_fingerprint
        ),
        "task_token_run_fingerprint": manifest.identity.run_fingerprint,
        "task_token_selection_fingerprint": selection.fingerprint,
        "selected_task_token_checkpoint_fingerprint": selected_checkpoint_fingerprint,
        "evaluated_model_checkpoint_fingerprints": [
            *(item.checkpoint_fingerprint for item in evidence.checkpoints[:6]),
            evidence.mixed_task_onehot.checkpoint_fingerprint,
            selected_checkpoint_fingerprint,
        ],
        "execution_horizon": horizon,
        "gripper_mode": gripper.value,
    }


def _final_semantic_identity(identity: Mapping[str, object]) -> dict[str, object]:
    """Exclude only implementation-repair fields from a sealed final identity."""

    required = {
        "schema_version",
        "experiment_manifest_fingerprint",
        "final_schedule_fingerprint",
        "implementation_fingerprint",
        "evaluation_git_commit",
        "prior_m4_evidence_fingerprint",
        "runtime_selection_fingerprint",
        "runtime_selection_source_fingerprint",
        "task_token_run_fingerprint",
        "task_token_selection_fingerprint",
        "selected_task_token_checkpoint_fingerprint",
        "evaluated_model_checkpoint_fingerprints",
        "execution_horizon",
        "gripper_mode",
    }
    if set(identity) != required:
        raise RuntimeError("sealed final identity fields differ from the declared contract")
    return {
        key: value
        for key, value in identity.items()
        if key not in {"implementation_fingerprint", "evaluation_git_commit"}
    }


def _final_attempt_directories(output_root: Path) -> tuple[Path, tuple[Path, ...]]:
    root = _resolved_unlinked(output_root / "final_attempts", label="sealed final attempts")
    if root.exists() and not root.is_dir():
        raise RuntimeError("sealed final attempts root must be one real directory")
    root.mkdir(parents=True, exist_ok=True)
    attempts: list[Path] = []
    for candidate in sorted(root.iterdir()):
        suffix = candidate.name.removeprefix("attempt-")
        if (
            not candidate.name.startswith("attempt-")
            or not suffix.isdigit()
            or candidate.is_symlink()
            or candidate.is_junction()
            or not candidate.is_dir()
        ):
            raise RuntimeError(f"unsafe or unexpected sealed final attempt entry: {candidate}")
        attempts.append(candidate)
    return root, tuple(attempts)


def _completed_attempt_artifacts(attempt_root: Path) -> list[dict[str, str]]:
    evidence_root = attempt_root / "evidence"
    if not evidence_root.exists():
        return []
    evidence_root = _resolved_unlinked(evidence_root, label="sealed final attempt evidence")
    if not evidence_root.is_dir():
        raise RuntimeError("sealed final attempt evidence must be one real directory")
    completed: list[dict[str, str]] = []
    for candidate in sorted(evidence_root.iterdir()):
        if candidate.name.startswith(".staging-"):
            continue
        if candidate.is_symlink() or candidate.is_junction() or not candidate.is_dir():
            raise RuntimeError(f"unsafe sealed final evidence entry: {candidate}")
        completion = candidate / "complete.json"
        artifact_path = candidate / "artifact.json"
        if not completion.is_file() or not artifact_path.is_file():
            continue
        artifact = _read_object(artifact_path, "sealed final partial artifact")
        completed.append(
            {
                "path": str(candidate.relative_to(attempt_root)),
                "artifact_fingerprint": canonical_fingerprint(artifact),
            }
        )
    return completed


def _record_incomplete_final_attempt(
    *,
    attempt_root: Path,
    identity_fingerprint: str,
    progress: _CommandProgress,
    error: BaseException | None,
    reason: str,
) -> None:
    path = attempt_root / "incomplete.json"
    if path.exists():
        existing = _read_object(path, "sealed final incomplete marker")
        if (
            existing.get("schema_version") != FINAL_ATTEMPT_INCOMPLETE_SCHEMA
            or existing.get("identity_fingerprint") != identity_fingerprint
            or existing.get("state") != "invalid_incomplete"
        ):
            raise RuntimeError("sealed final incomplete marker is not content-bound")
        return
    evidence_root = attempt_root / "evidence"
    staging_paths = (
        []
        if not evidence_root.is_dir()
        else [
            str(item.relative_to(attempt_root))
            for item in sorted(evidence_root.glob(".staging-*"))
            if item.is_dir() and not item.is_symlink() and not item.is_junction()
        ]
    )
    atomic_write_json(
        path,
        {
            "schema_version": FINAL_ATTEMPT_INCOMPLETE_SCHEMA,
            "identity_fingerprint": identity_fingerprint,
            "state": "invalid_incomplete",
            "reason": reason,
            "error_type": None if error is None else type(error).__name__,
            "error_message": None if error is None else str(error) or repr(error),
            "final_schedule_accessed": progress.final_schedule_accessed,
            "physical_execution_started": progress.physical_execution_started,
            "completed_artifacts": _completed_attempt_artifacts(attempt_root),
            "partial_staging_paths": staging_paths,
            "recorded_at_utc": datetime.now(UTC).isoformat(),
        },
        immutable=True,
    )


def _start_final_attempt(
    *,
    output_root: Path,
    identity: Mapping[str, object],
    progress: _CommandProgress,
) -> tuple[Path, str]:
    attempts_root, attempts = _final_attempt_directories(output_root)
    identity_fingerprint = canonical_fingerprint(identity)
    semantic_identity_fingerprint = canonical_fingerprint(_final_semantic_identity(identity))
    supersedes: list[dict[str, object]] = []
    for previous in attempts:
        owner = _read_object(previous / "attempt.json", "sealed final attempt owner")
        previous_identity = owner.get("identity")
        if not isinstance(previous_identity, Mapping):
            raise RuntimeError("sealed final attempt owner lacks its full identity")
        previous_identity_fingerprint = canonical_fingerprint(previous_identity)
        previous_semantic_fingerprint = canonical_fingerprint(
            _final_semantic_identity(previous_identity)
        )
        if (
            owner.get("identity_fingerprint") != previous_identity_fingerprint
            or owner.get("semantic_identity_fingerprint") != previous_semantic_fingerprint
        ):
            raise RuntimeError("sealed final attempt owner fingerprints are inconsistent")
        if previous_semantic_fingerprint != semantic_identity_fingerprint:
            raise RuntimeError(
                "sealed final repair changed schedule, model, selection, or runtime identity"
            )
        if not (previous / "incomplete.json").is_file():
            _record_incomplete_final_attempt(
                attempt_root=previous,
                identity_fingerprint=previous_identity_fingerprint,
                progress=_CommandProgress(),
                error=None,
                reason="superseded_by_clean_rerun_without_authoritative_completion",
            )
        incomplete = _read_object(previous / "incomplete.json", "superseded sealed final attempt")
        if (
            incomplete.get("identity_fingerprint") != previous_identity_fingerprint
            or incomplete.get("state") != "invalid_incomplete"
        ):
            raise RuntimeError("sealed final repair predecessor is not explicitly incomplete")
        supersedes.append(
            {
                "attempt": str(previous.relative_to(output_root)),
                "identity_fingerprint": previous_identity_fingerprint,
                "implementation_fingerprint": previous_identity["implementation_fingerprint"],
                "evaluation_git_commit": previous_identity["evaluation_git_commit"],
                "incomplete_marker_fingerprint": canonical_fingerprint(incomplete),
                "repair_kind": (
                    "clean_rerun"
                    if previous_identity_fingerprint == identity_fingerprint
                    else "infrastructure_implementation_repair"
                ),
            }
        )
    next_index = (
        max(int(item.name.removeprefix("attempt-")) for item in attempts) + 1 if attempts else 1
    )
    attempt_root = _resolved_unlinked(
        attempts_root / f"attempt-{next_index:04d}", label="new sealed final attempt"
    )
    if attempt_root.parent != attempts_root:
        raise RuntimeError("sealed final attempt escaped its owned root")
    attempt_root.mkdir()
    atomic_write_json(
        attempt_root / "attempt.json",
        {
            "schema_version": FINAL_ATTEMPT_SCHEMA,
            "identity": dict(identity),
            "identity_fingerprint": identity_fingerprint,
            "semantic_identity_fingerprint": semantic_identity_fingerprint,
            "supersedes": supersedes,
            "state": "in_progress",
            "complete": False,
            "final_schedule_accessed": False,
            "started_at_utc": datetime.now(UTC).isoformat(),
        },
        immutable=True,
    )
    progress.final_attempt_relative_path = str(attempt_root.relative_to(output_root))
    return attempt_root, identity_fingerprint


def _contained_final_path(output_root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise RuntimeError(f"{label} must be one relative path")
    result = _resolved_unlinked(output_root / relative, label=label)
    try:
        result.relative_to(output_root)
    except ValueError as error:
        raise RuntimeError(f"{label} escapes the sealed final output") from error
    return result


def _validate_final_artifact_reference(
    output_root: Path, reference: object, *, label: str
) -> dict[str, Any]:
    if not isinstance(reference, Mapping):
        raise RuntimeError(f"{label} reference is malformed")
    artifact_root = _contained_final_path(output_root, reference.get("path"), label=label)
    artifact = _read_object(artifact_root / "artifact.json", f"{label} artifact")
    completion = _read_object(artifact_root / "complete.json", f"{label} completion")
    fingerprint = canonical_fingerprint(artifact)
    artifact_identity = artifact.get("identity")
    if (
        not isinstance(artifact_identity, Mapping)
        or reference.get("artifact_fingerprint") != fingerprint
        or completion.get("artifact_fingerprint") != fingerprint
        or completion.get("identity_fingerprint")
        != canonical_fingerprint(cast(Mapping[str, object], artifact_identity))
        or artifact.get("passed") is not True
        or completion.get("passed") is not True
    ):
        raise RuntimeError(f"{label} content differs from sealed final completion")
    return artifact


def _load_reusable_final_completion(
    *, output_root: Path, identity: Mapping[str, object]
) -> dict[str, Any] | None:
    comparison_path = output_root / "final_comparison.json"
    if not comparison_path.exists():
        return None
    payload = _read_object(comparison_path, "sealed final comparison")
    identity_fingerprint = canonical_fingerprint(identity)
    if (
        payload.get("schema_version") != FINAL_SCHEMA
        or payload.get("complete") is not True
        or payload.get("final_identity") != dict(identity)
        or payload.get("final_identity_fingerprint") != identity_fingerprint
        or payload.get("final_schedule_accessed") is not True
    ):
        raise RuntimeError("sealed final completion differs from the requested identity")
    final_result = payload.get("final_result")
    final_config = payload.get("final_config")
    go_no_go = payload.get("go_no_go")
    if (
        not isinstance(final_result, Mapping)
        or not isinstance(final_config, Mapping)
        or not isinstance(go_no_go, Mapping)
        or payload.get("experiment_manifest_fingerprint")
        != identity["experiment_manifest_fingerprint"]
        or payload.get("final_config_fingerprint")
        != canonical_fingerprint(cast(Mapping[str, object], final_config))
        or final_result.get("config_fingerprint") != payload.get("final_config_fingerprint")
        or payload.get("final_result_fingerprint")
        != canonical_fingerprint(cast(Mapping[str, object], final_result))
        or payload.get("go_no_go_fingerprint")
        != canonical_fingerprint(cast(Mapping[str, object], go_no_go))
        or go_no_go.get("final_benchmark_fingerprint") != payload.get("final_result_fingerprint")
    ):
        raise RuntimeError("sealed final result fingerprints are inconsistent")
    attempt_completion_path = _contained_final_path(
        output_root, payload.get("attempt_completion"), label="sealed final attempt completion"
    )
    attempt_completion = _read_object(attempt_completion_path, "sealed final attempt completion")
    attempt_root = attempt_completion_path.parent
    attempt_owner = _read_object(attempt_root / "attempt.json", "sealed final attempt owner")
    attempt_owner_fingerprint = canonical_fingerprint(attempt_owner)
    if (
        attempt_completion.get("schema_version") != FINAL_ATTEMPT_COMPLETION_SCHEMA
        or attempt_completion.get("identity_fingerprint") != identity_fingerprint
        or attempt_completion.get("final_comparison_fingerprint") != canonical_fingerprint(payload)
        or attempt_completion.get("completed_artifact_count") != 9
        or attempt_completion.get("complete") is not True
        or attempt_owner.get("identity") != dict(identity)
        or attempt_owner.get("identity_fingerprint") != identity_fingerprint
        or attempt_owner.get("semantic_identity_fingerprint")
        != canonical_fingerprint(_final_semantic_identity(identity))
        or attempt_completion.get("attempt_owner_fingerprint") != attempt_owner_fingerprint
        or attempt_completion.get("sensitivity_artifact") != payload.get("sensitivity_artifact")
        or (attempt_root / "incomplete.json").exists()
    ):
        raise RuntimeError("sealed final attempt completion is not content-bound")
    references = payload.get("evidence_artifacts")
    if not isinstance(references, Mapping):
        raise RuntimeError("sealed final evidence references are malformed")
    onehot_artifact = _validate_final_artifact_reference(
        output_root, references.get("state_onehot"), label="State-OneHot final evidence"
    )
    token_artifact = _validate_final_artifact_reference(
        output_root, references.get("task_token"), label="TaskToken final evidence"
    )
    per_task_references = references.get("per_task")
    if not isinstance(per_task_references, list) or len(per_task_references) != 6:
        raise RuntimeError("sealed final PerTask evidence references are malformed")
    per_task_artifacts = [
        _validate_final_artifact_reference(
            output_root, reference, label=f"PerTask final evidence {index}"
        )
        for index, reference in enumerate(per_task_references)
    ]
    sensitivity_artifact = _validate_final_artifact_reference(
        output_root, payload.get("sensitivity_artifact"), label="sealed final sensitivity"
    )
    model_artifacts = (onehot_artifact, token_artifact, *per_task_artifacts)
    expected_checkpoint_fingerprints = cast(
        list[object], identity["evaluated_model_checkpoint_fingerprints"]
    )
    ordered_model_fingerprints = (
        expected_checkpoint_fingerprints[6],
        expected_checkpoint_fingerprints[7],
        *expected_checkpoint_fingerprints[:6],
    )
    expected_episode_counts = (180, 180, 30, 30, 30, 30, 30, 30)
    for artifact, checkpoint_fingerprint, episode_count in zip(
        model_artifacts,
        ordered_model_fingerprints,
        expected_episode_counts,
        strict=True,
    ):
        artifact_identity = cast(Mapping[str, object], artifact["identity"])
        if (
            artifact_identity.get("stage") != "final"
            or artifact_identity.get("schedule_fingerprint")
            != identity["final_schedule_fingerprint"]
            or artifact_identity.get("implementation_fingerprint")
            != identity["implementation_fingerprint"]
            or artifact_identity.get("experiment_manifest_fingerprint")
            != identity["experiment_manifest_fingerprint"]
            or artifact_identity.get("runtime_selection_fingerprint")
            != identity["runtime_selection_fingerprint"]
            or artifact_identity.get("runtime_selection_source_fingerprint")
            != identity["runtime_selection_source_fingerprint"]
            or artifact_identity.get("checkpoint_fingerprint") != checkpoint_fingerprint
            or artifact_identity.get("execution_horizon") != identity["execution_horizon"]
            or artifact_identity.get("gripper_mode") != identity["gripper_mode"]
            or not isinstance(artifact_identity.get("ordered_episode_indices"), list)
            or len(cast(list[object], artifact_identity["ordered_episode_indices"]))
            != episode_count
        ):
            raise RuntimeError("sealed final model artifact differs from final identity")
    sensitivity_identity = cast(Mapping[str, object], sensitivity_artifact["identity"])
    if (
        sensitivity_identity.get("final_identity_fingerprint") != identity_fingerprint
        or sensitivity_identity.get("checkpoint_fingerprints") != expected_checkpoint_fingerprints
    ):
        raise RuntimeError("sealed final sensitivity differs from final identity")
    model_results = final_result.get("model_results")
    sensitivity_payload = sensitivity_artifact.get("payload")
    if (
        not isinstance(model_results, Mapping)
        or not isinstance(sensitivity_payload, Mapping)
        or model_results.get("state_onehot") != _benchmark_payload(onehot_artifact)
        or model_results.get("task_token") != _benchmark_payload(token_artifact)
        or not isinstance(model_results.get("per_task"), Mapping)
        or cast(Mapping[str, object], model_results["per_task"]).get("benchmarks")
        != [_benchmark_payload(item) for item in per_task_artifacts]
        or final_result.get("task_sensitivity") != sensitivity_payload.get("sensitivity")
    ):
        raise RuntimeError("sealed final completion differs from referenced physical evidence")
    return payload


def _attempt_artifact_reference(
    *, output_root: Path, attempt_root: Path, relative_path: str, artifact: Mapping[str, object]
) -> dict[str, str]:
    path = attempt_root / relative_path
    return {
        "path": str(path.relative_to(output_root)),
        "artifact_fingerprint": canonical_fingerprint(artifact),
    }


def _run_final_sensitivity_artifact(
    *,
    output_root: Path,
    attempt_root: Path,
    final_identity_fingerprint: str,
    env: object,
    scene_seeds: Sequence[int],
    per_task: Sequence[M42CheckpointContext],
    onehot: M42CheckpointContext,
    task_token: M42CheckpointContext,
) -> tuple[dict[str, Any], dict[str, str]]:
    contexts = (*per_task, onehot, task_token)
    identity = {
        "schema_version": FINAL_SENSITIVITY_SCHEMA,
        "final_identity_fingerprint": final_identity_fingerprint,
        "ordered_scene_seeds": list(scene_seeds),
        "checkpoint_descriptor_fingerprints": [item.descriptor.fingerprint for item in contexts],
        "checkpoint_fingerprints": [item.descriptor.checkpoint_fingerprint for item in contexts],
    }

    def run() -> dict[str, object]:
        return {
            "sensitivity": _sensitivity(
                env=env,
                scene_seeds=scene_seeds,
                per_task=per_task,
                onehot=onehot,
                task_token=task_token,
            )
        }

    artifact, relative_path = _run_or_reuse_artifact(
        output_root=attempt_root, identity=identity, runner=run
    )
    return artifact, _attempt_artifact_reference(
        output_root=output_root,
        attempt_root=attempt_root,
        relative_path=relative_path,
        artifact=artifact,
    )


def _final_success_report(
    *,
    report: Mapping[str, object],
    payload: Mapping[str, object],
    selected_checkpoint_fingerprint: str,
    completion_reused: bool,
) -> dict[str, object]:
    final_result = cast(Mapping[str, object], payload["final_result"])
    model_results = cast(Mapping[str, object], final_result["model_results"])
    per_task = cast(Mapping[str, object], model_results["per_task"])
    benchmarks = (
        cast(Mapping[str, object], model_results["state_onehot"]),
        cast(Mapping[str, object], model_results["task_token"]),
        *cast(Sequence[Mapping[str, object]], per_task["benchmarks"]),
    )
    checks = tuple(_final_metric_evidence_complete(item) for item in benchmarks)
    safety_checks = tuple(_metrics_valid(item) for item in benchmarks)
    go_no_go = cast(Mapping[str, object], payload["go_no_go"])
    evidence_valid = all(raw and runtime and post_grasp for raw, runtime, post_grasp in checks)
    return {
        **report,
        "selected_task_token_checkpoint_fingerprint": selected_checkpoint_fingerprint,
        "final_comparison": "final_comparison.json",
        "final_result_fingerprint": payload["final_result_fingerprint"],
        "go_no_go": dict(go_no_go),
        "task_token_checkpoint_selected": True,
        "validation_only_selection_validated": True,
        "development_benchmark_completed": True,
        "final_benchmark_completed": True,
        "post_grasp_analysis_completed": all(item[2] for item in checks),
        "raw_action_metrics_validated": all(item[0] for item in checks),
        "raw_action_safety_passed": all(item[0] for item in safety_checks),
        "runtime_action_metrics_validated": all(item[1] for item in checks),
        "go_no_go_decision_completed": True,
        "smolvla_go": go_no_go.get("decision") == "go_for_smolvla",
        "final_schedule_accessed": True,
        "physical_execution": True,
        "final_completion_reused": completion_reused,
        "current_command_final_schedule_accessed": not completion_reused,
        "current_command_physical_execution": not completion_reused,
        "passed": evidence_valid,
    }


def _execute_final(
    *,
    args: argparse.Namespace,
    output_root: Path,
    report: dict[str, object],
    evidence: PriorM4Evidence,
    run_root: Path,
    manifest: TaskTokenTrainingManifest,
    queue: TaskTokenValidationQueue,
    horizon: int,
    gripper: GripperRuntimeMode,
    git: object,
) -> dict[str, object]:
    development_root = output_root.parent / "development"
    selection_artifact, selection, selected_item = _validate_final_development_lock(
        development_root=development_root,
        queue=queue,
        manifest=manifest,
        report=report,
        horizon=horizon,
        gripper=gripper,
    )
    selected_fingerprint = selection.selected_checkpoint_fingerprint
    if not isinstance(selected_fingerprint, str):
        raise RuntimeError("sealed final TaskToken selection lacks a checkpoint fingerprint")
    expected = args.expected_implementation_fingerprint
    actual = cast(str, report["implementation_fingerprint"])
    if not isinstance(expected, str):
        raise RuntimeError("physical final stage requires --expected-implementation-fingerprint")
    schedule = load_locked_schedule(M42_FINAL_SCHEDULE_ID)
    authorization = FinalScheduleAuthorization(
        final_schedule_fingerprint=schedule.schedule_fingerprint,
        expected_implementation_fingerprint=expected,
        actual_implementation_fingerprint=actual,
        explicit_final_authorization=bool(args.authorize_sealed_final),
        clean_git=not cast(Any, git).dirty,
        horizon_selection_locked=True,
        gripper_selection_locked=True,
        task_token_checkpoint_selection_locked=selection_artifact.get("locked") is True,
    )
    authorization.require_valid(schedule)
    final_identity = _final_execution_identity(
        schedule_fingerprint=schedule.schedule_fingerprint,
        implementation_fingerprint=actual,
        git_commit=cast(Any, git).commit,
        evidence=evidence,
        manifest=manifest,
        selection=selection,
        selected_checkpoint_fingerprint=selected_fingerprint,
        horizon=horizon,
        gripper=gripper,
    )
    reusable = _load_reusable_final_completion(output_root=output_root, identity=final_identity)
    if reusable is not None:
        return _final_success_report(
            report=report,
            payload=reusable,
            selected_checkpoint_fingerprint=selected_fingerprint,
            completion_reused=True,
        )

    runtime_selection = load_runtime_selection(args.runtime_selection)
    selected_context = _task_token_context(run_root, manifest, selected_item)
    per_task_contexts, onehot_context = _m4_contexts(evidence)
    progress = _command_progress(args)
    attempt_root, identity_fingerprint = _start_final_attempt(
        output_root=output_root, identity=final_identity, progress=progress
    )
    try:
        progress.final_schedule_accessed = True
        episodes = materialize_schedule(schedule, final_authorization=authorization)
        env = _create_environment()
        try:
            progress.physical_execution_started = True
            onehot_artifact, onehot_path = _run_model_benchmark(
                output_root=attempt_root,
                stage="final",
                schedule=schedule,
                episodes=episodes,
                context=onehot_context,
                horizon=horizon,
                gripper=gripper,
                model_label="ACT-Mixed-TaskOneHot",
                env=env,
                final_authorization=authorization,
                experiment_manifest_fingerprint=(manifest.identity.experiment_manifest_fingerprint),
                runtime_selection_fingerprint=manifest.identity.runtime_selection_fingerprint,
                runtime_selection_source_fingerprint=(
                    manifest.identity.runtime_selection_source_fingerprint
                ),
            )
            progress.completed_evaluation_artifact_count += 1
            token_artifact, token_path = _run_model_benchmark(
                output_root=attempt_root,
                stage="final",
                schedule=schedule,
                episodes=episodes,
                context=selected_context,
                horizon=horizon,
                gripper=gripper,
                model_label="ACT-Mixed-TaskToken",
                env=env,
                final_authorization=authorization,
                experiment_manifest_fingerprint=(manifest.identity.experiment_manifest_fingerprint),
                runtime_selection_fingerprint=manifest.identity.runtime_selection_fingerprint,
                runtime_selection_source_fingerprint=(
                    manifest.identity.runtime_selection_source_fingerprint
                ),
            )
            progress.completed_evaluation_artifact_count += 1
            per_task_artifacts: list[dict[str, Any]] = []
            per_task_paths: list[str] = []
            for index, context in enumerate(per_task_contexts):
                selected_episodes = tuple(
                    item for item in episodes if item.task_id == CANONICAL_TASK_IDS[index]
                )
                artifact, relative = _run_model_benchmark(
                    output_root=attempt_root,
                    stage="final",
                    schedule=schedule,
                    episodes=selected_episodes,
                    context=context,
                    horizon=horizon,
                    gripper=gripper,
                    model_label=f"ACT-PerTask:{CANONICAL_TASK_IDS[index]}",
                    env=env,
                    final_authorization=authorization,
                    experiment_manifest_fingerprint=(
                        manifest.identity.experiment_manifest_fingerprint
                    ),
                    runtime_selection_fingerprint=(manifest.identity.runtime_selection_fingerprint),
                    runtime_selection_source_fingerprint=(
                        manifest.identity.runtime_selection_source_fingerprint
                    ),
                )
                progress.completed_evaluation_artifact_count += 1
                per_task_artifacts.append(artifact)
                per_task_paths.append(relative)
            sensitivity_artifact, sensitivity_reference = _run_final_sensitivity_artifact(
                output_root=output_root,
                attempt_root=attempt_root,
                final_identity_fingerprint=identity_fingerprint,
                env=env,
                scene_seeds=schedule.ordered_scene_seeds,
                per_task=per_task_contexts,
                onehot=onehot_context,
                task_token=selected_context,
            )
            progress.completed_evaluation_artifact_count += 1
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

        onehot = _benchmark_payload(onehot_artifact)
        task_token = _benchmark_payload(token_artifact)
        per_task_benchmarks = tuple(_benchmark_payload(item) for item in per_task_artifacts)
        per_task_aggregate = _combine_per_task(per_task_benchmarks)
        per_task_semantic = {
            (cast(str, item["scene_id"]), cast(str, item["task_id"])): bool(item["success"])
            for item in cast(list[Mapping[str, object]], per_task_aggregate["semantic_success"])
        }
        sensitivity_payload = sensitivity_artifact.get("payload")
        if not isinstance(sensitivity_payload, Mapping) or not isinstance(
            sensitivity_payload.get("sensitivity"), Mapping
        ):
            raise RuntimeError("sealed final sensitivity artifact is malformed")
        sensitivity = cast(Mapping[str, object], sensitivity_payload["sensitivity"])
        token_aggregate = cast(Mapping[str, object], task_token["aggregate"])
        token_actions = cast(Mapping[str, object], token_aggregate["action_metrics"])
        token_per_task = cast(Mapping[str, Mapping[str, object]], token_aggregate["per_task"])
        nan_count, inf_count, malformed_action_count = _classified_task_token_action_counts(
            task_token
        )
        wrong_grasp_count = sum(
            bool(cast(Mapping[str, object], item["rollout"])["wrong_object_grasped_any"])
            for item in _episodes_from_benchmark(task_token)
        )
        final_config = M42FinalBenchmarkConfig(
            final_schedule_fingerprint=schedule.schedule_fingerprint,
            horizon_selection_fingerprint=runtime_selection.horizon_selection_fingerprint,
            gripper_selection_fingerprint=runtime_selection.gripper_selection_fingerprint,
            task_token_selection_fingerprint=selection.fingerprint,
            final_evaluation_git_commit=cast(Any, git).commit,
            model_checkpoint_fingerprints=tuple(
                item.checkpoint_fingerprint for item in evidence.checkpoints[:6]
            )
            + (
                evidence.mixed_task_onehot.checkpoint_fingerprint,
                selected_fingerprint,
            ),
        )
        paired = {
            "task_token_vs_per_task": _paired(_semantic_success(task_token), per_task_semantic),
            "task_token_vs_state_onehot": _paired(
                _semantic_success(task_token), _semantic_success(onehot)
            ),
            "state_onehot_vs_per_task": _paired(_semantic_success(onehot), per_task_semantic),
        }
        final_result = M42FinalBenchmarkResult(
            config_fingerprint=final_config.fingerprint,
            completed=True,
            model_results={
                "state_onehot": onehot,
                "task_token": task_token,
                "per_task": {
                    "benchmarks": list(per_task_benchmarks),
                    "aggregate": per_task_aggregate,
                },
            },
            paired_comparisons=paired,
            task_sensitivity=sensitivity,
            raw_action_metrics={
                "task_token": token_actions,
                "state_onehot": cast(Mapping[str, object], onehot["aggregate"])["action_metrics"],
                "per_task": per_task_aggregate["action_metrics"],
            },
            runtime_action_metrics={
                "execution_horizon": horizon,
                "gripper_mode": gripper.value,
                "task_token_runtime_bounds_validated": token_actions.get(
                    "runtime_action_bounds_validated"
                ),
            },
        )
        go_metrics = M42GoNoGoMetrics(
            task_token_successes=int(token_aggregate["successes"]),
            task_token_trials=int(token_aggregate["episode_count"]),
            per_task_aggregate_successes=int(per_task_aggregate["successes"]),
            per_task_aggregate_trials=int(per_task_aggregate["episode_count"]),
            task_token_successes_by_task=tuple(
                int(token_per_task[task_id]["successes"]) for task_id in CANONICAL_TASK_IDS
            ),
            timeout_count=int(token_aggregate["timeout_count"]),
            wrong_object_grasp_count=wrong_grasp_count,
            wrong_object_in_target_bin_count=int(
                token_aggregate["wrong_object_in_target_bin_count"]
            ),
            target_in_wrong_bin_count=int(token_aggregate["target_in_wrong_bin_count"]),
            target_off_table_count=int(token_aggregate["target_off_table_count"]),
            arm_projection_count=int(token_actions["arm_projected_component_count"]),
            nan_count=nan_count,
            inf_count=inf_count,
            malformed_action_count=malformed_action_count,
            task_sensitivity_ratio=float(sensitivity["task_token_ratio_relative_to_per_task"]),
        )
        decision = create_go_no_go_decision(
            metrics=go_metrics, final_benchmark_fingerprint=final_result.fingerprint
        )
        onehot_reference = _attempt_artifact_reference(
            output_root=output_root,
            attempt_root=attempt_root,
            relative_path=onehot_path,
            artifact=onehot_artifact,
        )
        token_reference = _attempt_artifact_reference(
            output_root=output_root,
            attempt_root=attempt_root,
            relative_path=token_path,
            artifact=token_artifact,
        )
        per_task_references = [
            _attempt_artifact_reference(
                output_root=output_root,
                attempt_root=attempt_root,
                relative_path=relative_path,
                artifact=artifact,
            )
            for relative_path, artifact in zip(per_task_paths, per_task_artifacts, strict=True)
        ]
        attempt_completion_path = attempt_root / "complete.json"
        payload = {
            "schema_version": FINAL_SCHEMA,
            "experiment_manifest_fingerprint": (manifest.identity.experiment_manifest_fingerprint),
            "final_identity": final_identity,
            "final_identity_fingerprint": identity_fingerprint,
            "final_config": final_config.to_dict(),
            "final_config_fingerprint": final_config.fingerprint,
            "final_result": final_result.to_dict(),
            "final_result_fingerprint": final_result.fingerprint,
            "go_no_go": decision.to_dict(),
            "go_no_go_fingerprint": decision.fingerprint,
            "attempt_completion": str(attempt_completion_path.relative_to(output_root)),
            "evidence_artifacts": {
                "state_onehot": onehot_reference,
                "task_token": token_reference,
                "per_task": per_task_references,
            },
            "sensitivity_artifact": sensitivity_reference,
            "complete": True,
            "final_schedule_accessed": True,
        }
        success_report = _final_success_report(
            report=report,
            payload=payload,
            selected_checkpoint_fingerprint=selected_fingerprint,
            completion_reused=False,
        )
        if success_report.get("passed") is not True:
            raise RuntimeError("sealed final evidence is incomplete or internally inconsistent")
        atomic_write_json(
            attempt_completion_path,
            {
                "schema_version": FINAL_ATTEMPT_COMPLETION_SCHEMA,
                "identity_fingerprint": identity_fingerprint,
                "attempt_owner_fingerprint": canonical_fingerprint(
                    _read_object(attempt_root / "attempt.json", "sealed final attempt owner")
                ),
                "final_comparison_fingerprint": canonical_fingerprint(payload),
                "completed_artifact_count": progress.completed_evaluation_artifact_count,
                "sensitivity_artifact": sensitivity_reference,
                "complete": True,
                "completed_at_utc": datetime.now(UTC).isoformat(),
            },
            immutable=True,
        )
        comparison_path = output_root / "final_comparison.json"
        if comparison_path.is_file():
            if _read_object(comparison_path, "final comparison") != payload:
                raise RuntimeError("sealed final comparison is immutable and differs")
        else:
            atomic_write_json(comparison_path, payload, immutable=True)
        return success_report
    except BaseException as error:
        try:
            _record_incomplete_final_attempt(
                attempt_root=attempt_root,
                identity_fingerprint=identity_fingerprint,
                progress=progress,
                error=error,
                reason="sealed_final_command_did_not_complete",
            )
        except Exception as marker_error:  # noqa: BLE001 - preserve the original failure
            error.add_note(f"failed to preserve incomplete final attempt: {marker_error!r}")
        raise


def execute(args: argparse.Namespace) -> dict[str, object]:
    progress = _command_progress(args)
    progress.final_schedule_accessed = False
    progress.physical_execution_started = False
    progress.completed_evaluation_artifact_count = 0
    progress.final_attempt_relative_path = None
    _validate_stage(args.stage, args.schedule)
    output_root, _ = _validate_paths(args)
    development, final = validate_locked_schedules()
    git = inspect_git_state(PROJECT_ROOT)
    implementation = _implementation_fingerprint(git.commit)
    report = _base_report(args=args, git=git, implementation_fingerprint=implementation)
    report.update(
        {
            "development_schedule_fingerprint": development.schedule_fingerprint,
            "final_schedule_fingerprint": final.schedule_fingerprint,
        }
    )
    if args.dry_run:
        # Loading both locks is permitted; materializing final episodes is not.
        report.update(
            {
                "expected_physical_schedule_episode_count": 72
                if args.stage == "development"
                else 180,
                "final_authorization_required": args.stage == "final",
                "final_schedule_accessed": False,
                "physical_execution": False,
                "passed": True,
            }
        )
        return report
    if not git.baseline_tracked or git.dirty:
        raise RuntimeError("M4.2 physical evaluation requires a clean tracked Git worktree")
    evidence = load_prior_m4_evidence(
        diagnostics_root=args.m4_diagnostics_root,
        checkpoint_root=args.checkpoint_root,
        dataset_root=args.dataset_root,
        validate_dataset_storage=True,
    )
    runtime = load_runtime_selection(args.runtime_selection)
    _verify_runtime_selection_sources(runtime, args.runtime_selection)
    if runtime.m3b_dataset_fingerprint != evidence.dataset_fingerprint:
        raise RuntimeError("runtime selection and frozen M4 evidence use different M3B data")
    manifest_relative = runtime.selection_evidence.get("experiment_manifest")
    if not isinstance(manifest_relative, str):
        raise RuntimeError("runtime selection lacks its experiment-manifest reference")
    experiment_manifest = load_m42_experiment_manifest(
        args.runtime_selection.resolve().parent / manifest_relative
    )
    validate_m42_experiment_manifest(experiment_manifest, evidence)
    if experiment_manifest.fingerprint != runtime.experiment_manifest_fingerprint:
        raise RuntimeError("runtime selection changed its M4.2 experiment manifest")
    report["experiment_manifest_fingerprint"] = experiment_manifest.fingerprint
    run_root, manifest, queue = _load_unique_task_token_run(args.task_token_root.resolve())
    _validate_runtime_training_binding(
        evidence=evidence,
        runtime=runtime,
        experiment_manifest=experiment_manifest,
        manifest=manifest,
        queue=queue,
    )
    common = {
        "args": args,
        "output_root": output_root,
        "report": report,
        "evidence": evidence,
        "run_root": run_root,
        "manifest": manifest,
        "queue": queue,
        "horizon": runtime.execution_horizon,
        "gripper": GripperRuntimeMode(runtime.gripper_mode),
    }
    if args.stage == "development":
        return _execute_development(**common)
    return _execute_final(**common, git=git)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    progress = _CommandProgress()
    args._m42_command_progress = progress
    try:
        report = execute(args)
    except Exception as error:  # noqa: BLE001 - outer command preserves exact diagnostics
        traceback.print_exc()
        report = {
            "schema_version": COMMAND_SCHEMA,
            "stage": args.stage,
            "schedule_id": args.schedule,
            "passed": False,
            "error_type": type(error).__name__,
            "error_message": str(error) or repr(error),
            "task_token_checkpoint_selected": False,
            "validation_only_selection_validated": False,
            "development_benchmark_completed": False,
            "final_benchmark_completed": False,
            "post_grasp_analysis_completed": False,
            "raw_action_metrics_validated": False,
            "runtime_action_metrics_validated": False,
            "go_no_go_decision_completed": False,
            "smolvla_go": False,
            "final_schedule_accessed": progress.final_schedule_accessed,
            "physical_execution": progress.physical_execution_started,
            "physical_execution_partial": progress.physical_execution_started,
            "completed_evaluation_artifact_count": (progress.completed_evaluation_artifact_count),
            "final_attempt": progress.final_attempt_relative_path,
        }
    try:
        _write_report(_safe_report_path(args), cast(dict[str, object], report))
    except Exception:  # noqa: BLE001 - retain original command failure on stderr
        traceback.print_exc()
        return 1
    print(
        json.dumps(
            {**report, "report": str(args.report.resolve())}, ensure_ascii=False, sort_keys=True
        )
    )
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
