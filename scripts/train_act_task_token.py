"""Train the single provenance-bound M4.2 ACT-Mixed-TaskToken model."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
import uuid
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader

from langmani.datasets.identity import sha256_hex
from langmani.datasets.lerobot_types import DatasetSplit
from langmani.policies.act_checkpoint import (
    CHECKPOINT_COMPLETION_MARKER,
    load_act_checkpoint,
    save_act_checkpoint,
)
from langmani.policies.act_data import (
    CompletedM3BDataset,
    TrainOnlyStatistics,
    compute_train_only_statistics,
    load_completed_m3b_dataset,
    load_lerobot_episode_view,
    validate_temporal_episode_boundaries,
)
from langmani.policies.act_runtime import (
    atomic_write_json,
    inspect_git_state,
    runtime_versions,
    safe_run_directory,
)
from langmani.policies.act_task_token import (
    TASK_TOKEN_FEATURE_KEY,
    build_task_token_act_config,
    task_token_architecture_fingerprint,
    task_token_contract,
    validate_task_token_policy,
)
from langmani.policies.act_training import (
    DeterministicResumeBatchSampler,
    TrainingOutcome,
    build_policy_and_processors,
    evaluate_offline_loss,
    make_optimizer,
    seed_dataloader_worker,
    seed_everything,
    train_act,
)
from langmani.policies.act_types import (
    ActDataConfig,
    ActExperimentConfig,
    ActModelConfig,
    ActOptimizationConfig,
    ActVariant,
    CheckpointRecord,
    ExperimentMode,
    TrainingState,
)
from langmani.policies.m42_source_fingerprint import (
    TASK_TOKEN_TRAINING_SOURCE_FILES,
    current_task_token_training_source_fingerprint,
)
from langmani.policies.m42_training import (
    TASK_TOKEN_TRAINING_COMPLETION_SCHEMA,
    TASK_TOKEN_TRAINING_SUMMARY_SCHEMA,
    RuntimeSelectionContract,
    TaskTokenFairComparisonContract,
    TaskTokenRunIdentity,
    TaskTokenTrainingManifest,
    TaskTokenValidationQueue,
    TaskTokenValidationQueueItem,
    canonical_fingerprint,
    load_runtime_selection,
)
from langmani.policies.m42_types import TaskTokenConfig, TaskTokenExperimentConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "outputs" / "datasets" / "m3b" / "langmani-pick-place-lerobot-v1"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "models" / "act-task-token"
DEFAULT_RUNTIME_SELECTION = (
    PROJECT_ROOT / "outputs" / "diagnostics" / "m42" / "runtime_ablation" / "runtime_selection.json"
)
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "diagnostics" / "m42" / "train_act_task_token.json"
TRAINING_SUMMARY_SCHEMA = TASK_TOKEN_TRAINING_SUMMARY_SCHEMA
TRAINING_COMPLETION_SCHEMA = TASK_TOKEN_TRAINING_COMPLETION_SCHEMA
PRECHECKPOINT_RECOVERY_SCHEMA = "langmani-m42-task-token-precheckpoint-recovery-v0"
RUNTIME_SELECTION_THAW_REPAIR_ID = "M42TaskTokenRuntimeSelectionFrozenTupleThawV0"


def _training_source_fingerprint() -> str:
    return current_task_token_training_source_fingerprint(PROJECT_ROOT)


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
    parser.add_argument(
        "--runtime-selection",
        type=Path,
        default=DEFAULT_RUNTIME_SELECTION,
        help="Locked M4.2a runtime_selection.json; m42_final_v0 evidence is rejected.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--resume-checkpoint",
        help="Exact run-relative checkpoint path; only the latest or sole atomic orphan is valid.",
    )
    return parser.parse_args(argv)


def _read_object(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return raw


def _write_or_validate(path: Path, value: dict[str, object]) -> None:
    if path.is_file():
        if _read_object(path) != value:
            raise RuntimeError(f"existing immutable TaskToken artifact differs: {path}")
        return
    atomic_write_json(path, value, immutable=True)


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
            args.runtime_selection,
            *PROTECTED_SOURCE_ROOTS,
        )
    )


def _validated_report_path(args: argparse.Namespace) -> Path:
    report = _resolved_unlinked(args.report, label="TaskToken command report")
    output = _resolved_unlinked(args.output_root, label="TaskToken output")
    _resolved_unlinked(report.parent, label="TaskToken command-report parent")
    if _is_within(report, output) or any(
        _is_within(report, protected) for protected in _protected_paths(args)
    ):
        raise RuntimeError(
            "TaskToken command report must not overlap an immutable input, run, source, or Git content"
        )
    if report.exists() and not report.is_file():
        raise RuntimeError("TaskToken command report path is unsafe")
    return report


def _validate_paths(args: argparse.Namespace) -> None:
    _resolved_unlinked(args.dataset_root, label="M3B dataset")
    runtime = _resolved_unlinked(args.runtime_selection, label="runtime selection")
    output = _resolved_unlinked(args.output_root, label="TaskToken output")
    _validated_report_path(args)
    if not runtime.is_file():
        raise RuntimeError("TaskToken runtime selection must be one real JSON file")
    if output.exists() and not output.is_dir():
        raise RuntimeError("TaskToken output root must be one real directory")
    protected = _protected_paths(args)
    if any(_overlaps(output, path) for path in protected):
        raise RuntimeError(
            "TaskToken output must not overlap M3B, runtime-selection, source, or Git content"
        )


def _enforce_single_task_token_run(output_root: Path, expected_run_root: Path) -> None:
    """Refuse to create a second fingerprint-owned TaskToken model run."""

    output = _resolved_unlinked(output_root, label="TaskToken output")
    expected = _resolved_unlinked(expected_run_root, label="expected TaskToken run")
    if expected.parent != output:
        raise RuntimeError("expected TaskToken run escaped its output root")
    if not output.exists():
        return
    for child in output.iterdir():
        if len(child.name) != 64 or any(
            character not in "0123456789abcdef" for character in child.name
        ):
            continue
        if child.is_symlink() or child.is_junction() or not child.is_dir():
            raise RuntimeError(f"unsafe fingerprint-owned TaskToken run path: {child}")
        if child.resolve() != expected:
            raise RuntimeError(
                "exactly one ACT-Mixed-TaskToken model is permitted; "
                f"a different run identity already exists: {child.name}"
            )


def _internal_experiment(dataset_root: Path, *, device: str) -> ActExperimentConfig:
    """Build an execution object without adding a fourth historical M4 variant."""
    return ActExperimentConfig(
        variant=ActVariant.MIXED_UNCONDITIONED,
        data=ActDataConfig(dataset_root=str(dataset_root.resolve())),
        model=ActModelConfig(state_dimension=9),
        optimization=ActOptimizationConfig(),
        seed=0,
        mode=ExperimentMode.FULL,
        device=device,
    )


def _load_data_and_statistics(
    dataset_root: Path,
) -> tuple[CompletedM3BDataset, TrainOnlyStatistics]:
    completed = load_completed_m3b_dataset(
        dataset_root,
        require_full=True,
        validate_storage=True,
    )
    train = completed.views.train
    validation = completed.views.validation
    test = completed.views.test
    if tuple(
        map(len, (train.episode_indices, validation.episode_indices, test.episode_indices))
    ) != (
        288,
        36,
        36,
    ):
        raise RuntimeError("TaskToken requires the exact M3B 288/36/36 split")
    frames = load_lerobot_episode_view(
        completed,
        train,
        policy_config=None,
        include_delta_timestamps=False,
        return_uint8=True,
    )
    statistics = compute_train_only_statistics(
        frames.dataset,
        train_episode_indices=train.episode_indices,
        validation_episode_indices=validation.episode_indices,
        test_episode_indices=test.episode_indices,
        task_id_by_episode=completed.views.task_id_by_episode,
        variant=ActVariant.MIXED_UNCONDITIONED,
        task_id=None,
    )
    if (
        statistics.leakage_audit.source_episode_indices != train.episode_indices
        or not statistics.leakage_audit.passed
    ):
        raise RuntimeError("TaskToken train-only statistics leakage audit failed")
    return completed, statistics


def _validation_schedule(completed: CompletedM3BDataset) -> tuple[dict[str, object], ...]:
    episodes = tuple(
        record for record in completed.manifest.episodes if record.split is DatasetSplit.VALIDATION
    )
    if len(episodes) != 36:
        raise RuntimeError("TaskToken validation schedule must contain exactly 36 episodes")
    return tuple(
        {
            "episode_index": record.lerobot_episode_index,
            "scene_seed": record.source_scene_seed,
            "scene_group_id": record.source_scene_group_id,
            "task_id": record.task_id,
            "task_spec": {
                "target_object_id": record.target_object_id,
                "target_bin_id": record.target_bin_id,
                "instruction_template_id": record.instruction_template_id,
            },
        }
        for record in episodes
    )


def _effective_act_config_dict(config: object) -> dict[str, object]:
    act_config = cast(Any, config)
    return {
        "input_features": {
            key: {"type": value.type.value, "shape": list(value.shape)}
            for key, value in act_config.input_features.items()
        },
        "output_features": {
            key: {"type": value.type.value, "shape": list(value.shape)}
            for key, value in act_config.output_features.items()
        },
        "normalization_mapping": {
            (key.value if hasattr(key, "value") else str(key)): (
                value.value if hasattr(value, "value") else str(value)
            )
            for key, value in act_config.normalization_mapping.items()
        },
        "device": str(act_config.device),
        "use_amp": bool(act_config.use_amp),
        "chunk_size": int(act_config.chunk_size),
        "n_action_steps": int(act_config.n_action_steps),
        "dim_model": int(act_config.dim_model),
    }


def _thaw_runtime_selection_json(value: object) -> object:
    """Thaw only the immutable JSON containers owned by RuntimeSelectionContract."""

    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise RuntimeError("runtime selection JSON keys must be strings")
        return {key: _thaw_runtime_selection_json(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_thaw_runtime_selection_json(item) for item in value]
    if value is None or isinstance(value, bool | int | float | str):
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError("runtime selection JSON numbers must be finite")
        return value
    raise RuntimeError(f"runtime selection contains a non-JSON value: {type(value).__name__}")


def _fair_comparison_contract(runtime: RuntimeSelectionContract) -> TaskTokenFairComparisonContract:
    raw = runtime.selection_evidence.get("task_token_fair_comparison_contract")
    if not isinstance(raw, Mapping):
        raise RuntimeError(
            "runtime selection lacks the frozen Mixed-TaskOneHot fair-comparison contract"
        )
    thawed = _thaw_runtime_selection_json(raw)
    if not isinstance(thawed, Mapping):
        raise RuntimeError("thawed runtime fair-comparison contract is not a JSON object")
    return TaskTokenFairComparisonContract.from_dict(cast(Mapping[str, object], thawed))


def _validate_fair_comparison(
    *,
    contract: TaskTokenFairComparisonContract,
    experiment: ActExperimentConfig,
    completed: CompletedM3BDataset,
    effective_act_config: object,
    dry_run: bool,
) -> str:
    """Require exact M4 parity except for the declared STATE-to-ENV split."""

    if (
        contract.m3b_export_fingerprint != completed.export_fingerprint
        or contract.m3b_split_manifest_digest != completed.split_manifest_digest
        or contract.ordered_train_episode_indices != completed.views.train.episode_indices
        or contract.ordered_validation_episode_indices != completed.views.validation.episode_indices
    ):
        raise RuntimeError("TaskToken dataset/split/order differs from frozen Mixed-TaskOneHot")
    source_model = dict(contract.source_model_config)
    target_model = experiment.model.to_dict()
    if source_model.pop("state_dimension", None) != 15:
        raise RuntimeError("frozen Mixed-TaskOneHot source no longer records a 15D state")
    if target_model.pop("state_dimension", None) != 9:
        raise RuntimeError("TaskToken must retain the 9D Panda policy state")
    if canonical_fingerprint(source_model) != canonical_fingerprint(target_model):
        raise RuntimeError(
            "TaskToken model differs from Mixed-TaskOneHot outside the allowed input split"
        )
    if canonical_fingerprint(contract.source_optimization_config) != canonical_fingerprint(
        experiment.optimization.to_dict()
    ):
        raise RuntimeError("TaskToken optimization differs from frozen Mixed-TaskOneHot")
    if contract.source_training_seed != experiment.seed:
        raise RuntimeError("TaskToken seed policy differs from frozen Mixed-TaskOneHot")
    common_data = {
        **experiment.data.identity_dict(),
        "m3b_feature_contract": completed.manifest.config.feature_contract.to_dict(),
        "policy_state_schema": completed.manifest.config.policy_state_schema.to_dict(),
    }
    if canonical_fingerprint(contract.source_common_data_contract) != canonical_fingerprint(
        common_data
    ):
        raise RuntimeError("TaskToken common data contract differs from frozen Mixed-TaskOneHot")
    expected_order = {
        "schema_version": "langmani-m42-deterministic-data-order-v0",
        "sampler": "DeterministicResumeBatchSampler",
        "ordered_train_episode_indices": list(completed.views.train.episode_indices),
        "ordered_validation_episode_indices": list(completed.views.validation.episode_indices),
        "training_seed": experiment.seed,
        "batch_size": experiment.optimization.batch_size,
        "epoch_seed_rule": "(training_seed + epoch) % (2**63 - 1)",
        "resume_cursor": "completed_optimizer_step",
    }
    if canonical_fingerprint(contract.data_order_contract) != canonical_fingerprint(expected_order):
        raise RuntimeError("TaskToken deterministic data ordering differs from Mixed-TaskOneHot")
    effective = _effective_act_config_dict(effective_act_config)
    input_features = effective["input_features"]
    output_features = effective["output_features"]
    if not isinstance(input_features, Mapping) or not isinstance(output_features, Mapping):
        raise RuntimeError("TaskToken effective ACT feature contract is malformed")
    expected_inputs = {
        experiment.model.image_feature_key: {
            "type": "VISUAL",
            "shape": list(experiment.model.image_shape_chw),
        },
        experiment.model.state_feature_key: {"type": "STATE", "shape": [9]},
        TASK_TOKEN_FEATURE_KEY: {"type": "ENV", "shape": [6]},
    }
    expected_outputs = {experiment.model.action_feature_key: {"type": "ACTION", "shape": [8]}}
    if dict(input_features) != expected_inputs or dict(output_features) != expected_outputs:
        raise RuntimeError(
            "TaskToken effective inputs differ beyond the declared 9D STATE + 6D ENV"
        )
    normalization = effective.get("normalization_mapping")
    if not isinstance(normalization, Mapping) or dict(normalization) != {
        "VISUAL": "MEAN_STD",
        "STATE": "MEAN_STD",
        "ACTION": "MEAN_STD",
        "ENV": "IDENTITY",
    }:
        raise RuntimeError("TaskToken normalization differs beyond the dedicated ENV identity rule")
    if not dry_run and (
        experiment.device != contract.source_device
        or experiment.dtype != contract.source_dtype
        or effective.get("use_amp") is not True
    ):
        raise RuntimeError("full TaskToken execution differs from the frozen CUDA/bfloat16 policy")
    return contract.contract_fingerprint


def _identity(
    *,
    experiment: ActExperimentConfig,
    completed: CompletedM3BDataset,
    statistics: TrainOnlyStatistics,
    runtime: RuntimeSelectionContract,
    task_token: TaskTokenConfig,
    effective_act_config: object,
    fair_comparison_fingerprint: str,
    git_commit: str,
    git_dirty: bool,
) -> tuple[TaskTokenRunIdentity, TaskTokenExperimentConfig, str]:
    if runtime.m3b_dataset_fingerprint != completed.export_fingerprint:
        raise RuntimeError("runtime selection refers to a different M3B dataset")
    versions = runtime_versions()
    architecture_fingerprint = task_token_architecture_fingerprint(task_token)
    schedule = _validation_schedule(completed)
    schedule_fingerprint = canonical_fingerprint(
        {
            "schema_version": "langmani-m42-m3b-validation-schedule-v0",
            "m3b_dataset_fingerprint": completed.export_fingerprint,
            "split_digest": completed.split_manifest_digest,
            "episodes": list(schedule),
        }
    )
    task_experiment = TaskTokenExperimentConfig(
        dataset_fingerprint=completed.export_fingerprint,
        split_digest=completed.split_manifest_digest,
        train_statistics_fingerprint=statistics.statistics_fingerprint,
        implementation_git_commit=git_commit,
        runtime_selection_fingerprint=runtime.runtime_fingerprint,
        token_config=task_token,
    )
    data_contract = {
        **experiment.data.identity_dict(),
        "m3b_feature_contract": completed.manifest.config.feature_contract.to_dict(),
        "policy_state_schema": completed.manifest.config.policy_state_schema.to_dict(),
        "train_only_statistics_audit": statistics.leakage_audit.to_dict(),
        "task_token": task_token_contract(task_token),
        "runtime_selection": runtime.to_dict(),
        "fair_comparison_fingerprint": fair_comparison_fingerprint,
        "validation_schedule_fingerprint": schedule_fingerprint,
        "validation_schedule": list(schedule),
        "checkpoint_selection_source": "m3b_validation_only",
        "selection_forbidden_sources": [
            "m3b_test",
            "m4_fresh_seed",
            "m42_dev_v0",
            "m42_final_v0",
        ],
        "test_episode_indices_accessed_during_training_or_selection": [],
    }
    model_contract = {
        "base_m4_model": experiment.model.to_dict(),
        "effective_act_config": _effective_act_config_dict(effective_act_config),
        "task_token": task_token.to_dict(),
        "task_token_architecture_fingerprint": architecture_fingerprint,
        "fair_comparison_fingerprint": fair_comparison_fingerprint,
    }
    identity = TaskTokenRunIdentity(
        m3b_export_fingerprint=completed.export_fingerprint,
        m3b_split_manifest_digest=completed.split_manifest_digest,
        ordered_train_episode_indices=completed.views.train.episode_indices,
        ordered_validation_episode_indices=completed.views.validation.episode_indices,
        model_config=model_contract,
        data_contract=data_contract,
        train_statistics_fingerprint=statistics.statistics_fingerprint,
        optimization_config=experiment.optimization.to_dict(),
        runtime_selection_fingerprint=runtime.runtime_fingerprint,
        runtime_selection_source_fingerprint=runtime.source_fingerprint,
        experiment_manifest_fingerprint=runtime.experiment_manifest_fingerprint,
        task_token_architecture_fingerprint=architecture_fingerprint,
        training_seed=experiment.seed,
        device=experiment.device,
        dtype=experiment.dtype,
        lerobot_version=str(versions["lerobot"]),
        torch_version=str(versions["torch"]),
        cuda_version=versions["cuda"],
        git_commit=git_commit,
        git_dirty=git_dirty,
    )
    return identity, task_experiment, schedule_fingerprint


def _loader(
    dataset: Any,
    experiment: ActExperimentConfig,
    *,
    shuffle: bool,
    start_step: int = 0,
) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(experiment.seed)
    common = {
        "dataset": dataset,
        "num_workers": experiment.optimization.dataloader_workers,
        "pin_memory": experiment.device == "cuda",
        "persistent_workers": experiment.optimization.dataloader_workers > 0,
        "worker_init_fn": seed_dataloader_worker,
        "generator": generator,
    }
    if shuffle:
        return DataLoader(
            **common,
            batch_sampler=DeterministicResumeBatchSampler(
                dataset_size=len(dataset),
                batch_size=experiment.optimization.batch_size,
                seed=experiment.seed,
                start_step=start_step,
            ),
        )
    return DataLoader(
        **common,
        batch_size=experiment.optimization.batch_size,
        shuffle=False,
        drop_last=False,
    )


def _manifest(
    *,
    identity: TaskTokenRunIdentity,
    task_experiment: TaskTokenExperimentConfig,
    state: TrainingState,
    checkpoints: tuple[CheckpointRecord, ...],
    complete: bool,
) -> TaskTokenTrainingManifest:
    return TaskTokenTrainingManifest(
        identity=identity,
        task_token_experiment=task_experiment.to_dict(),
        training_state=state,
        checkpoints=checkpoints,
        training_complete=complete,
    )


def _metrics(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    values: list[dict[str, object]] = []
    previous = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict) or not isinstance(value.get("step"), int):
            raise RuntimeError("TaskToken metrics.jsonl contains a malformed record")
        step = cast(int, value["step"])
        if step <= previous:
            raise RuntimeError("TaskToken metric steps are not strictly increasing")
        previous = step
        values.append(value)
    return values


def _replace_metrics(path: Path, records: list[dict[str, object]]) -> None:
    staging = path.with_name(f".{path.name}.resume-staging")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with staging.open("x", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def _repair_metrics_for_resume(
    path: Path,
    *,
    checkpoint: CheckpointRecord,
    checkpoint_metric: dict[str, object] | None,
) -> list[dict[str, object]]:
    retained = [
        record for record in _metrics(path) if cast(int, record["step"]) <= checkpoint.global_step
    ]
    matches = [record for record in retained if record["step"] == checkpoint.global_step]
    expected = (
        {**checkpoint_metric, "checkpoint_path": checkpoint.relative_path}
        if checkpoint_metric is not None
        else None
    )
    if len(matches) > 1:
        raise RuntimeError("resume checkpoint metric is duplicated")
    if matches:
        actual = matches[0]
        if actual.get("checkpoint_path") is None:
            actual["checkpoint_path"] = checkpoint.relative_path
        if expected is not None and actual != expected:
            raise RuntimeError("checkpoint-bound metric differs from checkpoint evidence")
    else:
        if expected is None:
            raise RuntimeError("resume checkpoint lacks recoverable metric evidence")
        if retained and cast(int, retained[-1]["step"]) >= checkpoint.global_step:
            raise RuntimeError("resume metric repair would violate ordering")
        retained.append(expected)
    _replace_metrics(path, retained)
    return retained


def _completed_checkpoint_paths(run_root: Path) -> set[str]:
    return {
        marker.parent.relative_to(run_root).as_posix()
        for marker in (run_root / "checkpoints").glob(f"*/{CHECKPOINT_COMPLETION_MARKER}")
    }


def _resume_kind(
    run_root: Path,
    *,
    requested: str,
    manifest: TaskTokenTrainingManifest,
) -> bool:
    """Return True only for the sole promoted checkpoint orphan."""
    declared = {record.relative_path for record in manifest.checkpoints}
    completed = _completed_checkpoint_paths(run_root)
    orphans = completed - declared
    if manifest.checkpoints and requested == manifest.checkpoints[-1].relative_path:
        if orphans:
            raise RuntimeError("a newer promoted checkpoint orphan must be resumed first")
        return False
    if requested in orphans and orphans == {requested}:
        return True
    raise RuntimeError("resume requires the latest declared checkpoint or sole atomic orphan")


def _latest_safe_resume_checkpoint(run_root: Path, manifest: TaskTokenTrainingManifest) -> str:
    """Choose the sole newest content-promoted checkpoint for verifier resume."""
    declared = {record.relative_path for record in manifest.checkpoints}
    completed = _completed_checkpoint_paths(run_root)
    orphans = completed - declared
    if len(orphans) > 1:
        raise RuntimeError("multiple promoted checkpoint orphans make automatic resume ambiguous")
    if orphans:
        return next(iter(orphans))
    if manifest.checkpoints:
        latest = manifest.checkpoints[-1].relative_path
        if latest not in completed:
            raise RuntimeError("latest declared TaskToken checkpoint is not atomically complete")
        return latest
    raise RuntimeError(
        "TaskToken run stopped before its first checkpoint; no exact automatic resume exists"
    )


def _validate_precheckpoint_attempt(
    run_root: Path,
    *,
    identity: TaskTokenRunIdentity,
    task_experiment: TaskTokenExperimentConfig,
    allow_recovery_marker: bool = False,
) -> tuple[TaskTokenTrainingManifest, list[dict[str, object]]]:
    """Validate the only incomplete layout that may be rebuilt from step zero."""

    root = _resolved_unlinked(run_root, label="precheckpoint TaskToken run")
    if not root.is_dir() or root.is_symlink() or root.is_junction():
        raise RuntimeError("precheckpoint TaskToken run must be one real directory")
    for artifact in root.rglob("*"):
        if artifact.is_symlink() or artifact.is_junction():
            raise RuntimeError(
                f"precheckpoint TaskToken run contains a linked artifact: {artifact}"
            )
        if not artifact.is_file() and not artifact.is_dir():
            raise RuntimeError(
                f"precheckpoint TaskToken run contains a special artifact: {artifact}"
            )

    allowed_top_level = {
        ".checkpoint-staging",
        "checkpoints",
        "dataset_contract.json",
        "metrics.jsonl",
        "reports",
        "run_manifest.json",
        "runtime_selection.json",
        "task_token_contract.json",
        "task_token_experiment.json",
        "train_stats.json",
    }
    if allow_recovery_marker:
        allowed_top_level.add("precheckpoint_recovery.json")
    required_top_level = {
        "dataset_contract.json",
        "reports",
        "run_manifest.json",
        "runtime_selection.json",
        "task_token_contract.json",
        "task_token_experiment.json",
        "train_stats.json",
    }
    actual_top_level = {artifact.name for artifact in root.iterdir()}
    if not required_top_level.issubset(actual_top_level) or not actual_top_level.issubset(
        allowed_top_level
    ):
        raise RuntimeError("precheckpoint TaskToken run has an unsafe artifact layout")

    manifest_path = root / "run_manifest.json"
    manifest = TaskTokenTrainingManifest.from_dict(_read_object(manifest_path))
    if manifest.identity != identity:
        raise RuntimeError("precheckpoint TaskToken run identity differs; refusing recovery")
    if canonical_fingerprint(manifest.task_token_experiment) != canonical_fingerprint(
        task_experiment.to_dict()
    ):
        raise RuntimeError("precheckpoint TaskToken experiment contract differs")
    state = manifest.training_state
    if (
        manifest.training_complete
        or state.completed
        or state.global_step != 0
        or state.examples_processed != 0
        or state.last_checkpoint_fingerprint is not None
        or manifest.checkpoints
        or _completed_checkpoint_paths(root)
    ):
        raise RuntimeError("TaskToken attempt is not a safe precheckpoint run")

    checkpoints_root = root / "checkpoints"
    if checkpoints_root.exists() and any(checkpoints_root.iterdir()):
        raise RuntimeError("precheckpoint TaskToken checkpoints directory is not empty")
    reports_root = root / "reports"
    if not reports_root.is_dir() or {path.name for path in reports_root.iterdir()} != {
        "initial_validation.json"
    }:
        raise RuntimeError("precheckpoint TaskToken reports layout is unsafe")
    initial = _read_object(reports_root / "initial_validation.json")
    initial_loss = initial.get("loss")
    if (
        initial.get("run_fingerprint") != identity.run_fingerprint
        or isinstance(initial_loss, bool)
        or not isinstance(initial_loss, int | float)
        or not math.isfinite(float(initial_loss))
        or initial.get("test_accessed") is not False
        or initial.get("development_schedule_accessed") is not False
        or initial.get("final_schedule_accessed") is not False
    ):
        raise RuntimeError("precheckpoint TaskToken initial validation evidence is unsafe")
    if _read_object(root / "task_token_experiment.json") != task_experiment.to_dict():
        raise RuntimeError("precheckpoint TaskToken experiment artifact differs")
    expected_data_contract = cast(dict[str, object], identity.to_dict()["data_contract"])
    if _read_object(root / "dataset_contract.json") != expected_data_contract:
        raise RuntimeError("precheckpoint TaskToken dataset contract differs")

    records = _metrics(root / "metrics.jsonl")
    steps = [cast(int, record["step"]) for record in records]
    if steps != list(range(1, len(steps) + 1)) or (steps and steps[-1] >= 5_000):
        raise RuntimeError("precheckpoint TaskToken metrics are outside the exact <5k prefix")
    return manifest, records


def _validated_pending_precheckpoint_archive(
    *,
    output_root: Path,
    run_root: Path,
    recovery_parent: Path,
    identity: TaskTokenRunIdentity,
    task_experiment: TaskTokenExperimentConfig,
    records: list[dict[str, object]],
) -> tuple[Path, dict[str, object]] | None:
    """Validate and resume the marker-written half of an atomic archival."""

    marker = run_root / "precheckpoint_recovery.json"
    if not marker.exists():
        return None
    if not marker.is_file() or marker.is_symlink() or marker.is_junction():
        raise RuntimeError("TaskToken precheckpoint recovery marker is unsafe")
    record = _read_object(marker)
    expected_keys = {
        "schema_version",
        "run_fingerprint",
        "experiment_manifest_fingerprint",
        "task_token_experiment_fingerprint",
        "identity_git_commit",
        "reason",
        "metric_record_count",
        "last_completed_metric_step",
        "first_checkpoint_step",
        "declared_checkpoint_count",
        "promoted_checkpoint_count",
        "original_run_directory",
        "archive_directory",
        "archived_at_unix_ns",
    }
    last_step = cast(int, records[-1]["step"]) if records else 0
    expected_values: dict[str, object] = {
        "schema_version": PRECHECKPOINT_RECOVERY_SCHEMA,
        "run_fingerprint": identity.run_fingerprint,
        "experiment_manifest_fingerprint": identity.experiment_manifest_fingerprint,
        "task_token_experiment_fingerprint": task_experiment.fingerprint,
        "identity_git_commit": identity.git_commit,
        "reason": "interrupted_before_first_5000_step_checkpoint",
        "metric_record_count": len(records),
        "last_completed_metric_step": last_step,
        "first_checkpoint_step": 5_000,
        "declared_checkpoint_count": 0,
        "promoted_checkpoint_count": 0,
        "original_run_directory": str(run_root),
    }
    if set(record) != expected_keys or any(
        record.get(key) != value for key, value in expected_values.items()
    ):
        raise RuntimeError("TaskToken precheckpoint recovery marker contract differs")
    archived_at = record.get("archived_at_unix_ns")
    relative_value = record.get("archive_directory")
    if (
        isinstance(archived_at, bool)
        or not isinstance(archived_at, int)
        or archived_at <= 0
        or not isinstance(relative_value, str)
        or not relative_value
    ):
        raise RuntimeError("TaskToken precheckpoint recovery marker fields are malformed")
    relative_archive = Path(relative_value)
    if (
        relative_archive.is_absolute()
        or relative_archive.as_posix() != relative_value
        or ".." in relative_archive.parts
    ):
        raise RuntimeError("TaskToken precheckpoint recovery archive path is not canonical")
    archive = _resolved_unlinked(
        output_root / relative_archive,
        label="TaskToken pending precheckpoint recovery archive",
    )
    try:
        relative_name = archive.relative_to(recovery_parent)
    except ValueError as error:
        raise RuntimeError(
            "TaskToken precheckpoint recovery archive escapes its fingerprint-owned root"
        ) from error
    name = relative_name.as_posix()
    parts = name.split("-")
    if (
        len(relative_name.parts) != 1
        or len(parts) != 3
        or parts[0] != "attempt"
        or not parts[1].isdigit()
        or len(parts[2]) != 32
        or any(character not in "0123456789abcdef" for character in parts[2])
    ):
        raise RuntimeError("TaskToken precheckpoint recovery archive name is unsafe")
    if archive.exists():
        raise RuntimeError("TaskToken pending precheckpoint recovery destination already exists")
    return archive, record


def _archive_precheckpoint_attempt(
    *,
    output_root: Path,
    run_root: Path,
    identity: TaskTokenRunIdentity,
    task_experiment: TaskTokenExperimentConfig,
) -> Path:
    """Atomically retain an exact-identity precheckpoint attempt before rebuilding."""

    output = _resolved_unlinked(output_root, label="TaskToken output")
    root = _resolved_unlinked(run_root, label="precheckpoint TaskToken run")
    if root != safe_run_directory(output, identity.run_fingerprint):
        raise RuntimeError("precheckpoint TaskToken run is not fingerprint-owned")
    _manifest_value, records = _validate_precheckpoint_attempt(
        root,
        identity=identity,
        task_experiment=task_experiment,
        allow_recovery_marker=True,
    )
    recovery_parent = _resolved_unlinked(
        output / ".recovery" / root.name,
        label="TaskToken precheckpoint recovery root",
    )
    recovery_parent.mkdir(parents=True, exist_ok=True)
    if recovery_parent.stat().st_dev != root.stat().st_dev:
        raise RuntimeError("TaskToken precheckpoint recovery must stay on one filesystem")
    pending = _validated_pending_precheckpoint_archive(
        output_root=output,
        run_root=root,
        recovery_parent=recovery_parent,
        identity=identity,
        task_experiment=task_experiment,
        records=records,
    )
    if pending is None:
        archive = recovery_parent / f"attempt-{time.time_ns()}-{uuid.uuid4().hex}"
        if archive.exists():
            raise RuntimeError("TaskToken precheckpoint recovery destination already exists")
        relative_archive = archive.relative_to(output).as_posix()
        last_step = cast(int, records[-1]["step"]) if records else 0
        recovery_record = {
            "schema_version": PRECHECKPOINT_RECOVERY_SCHEMA,
            "run_fingerprint": identity.run_fingerprint,
            "experiment_manifest_fingerprint": identity.experiment_manifest_fingerprint,
            "task_token_experiment_fingerprint": task_experiment.fingerprint,
            "identity_git_commit": identity.git_commit,
            "reason": "interrupted_before_first_5000_step_checkpoint",
            "metric_record_count": len(records),
            "last_completed_metric_step": last_step,
            "first_checkpoint_step": 5_000,
            "declared_checkpoint_count": 0,
            "promoted_checkpoint_count": 0,
            "original_run_directory": str(root),
            "archive_directory": relative_archive,
            "archived_at_unix_ns": time.time_ns(),
        }
        atomic_write_json(
            root / "precheckpoint_recovery.json",
            recovery_record,
            immutable=True,
        )
    else:
        archive, recovery_record = pending
    try:
        os.rename(root, archive)
    except OSError as error:
        raise RuntimeError("atomic TaskToken precheckpoint evidence archival failed") from error
    if root.exists() or not archive.is_dir():
        raise RuntimeError("TaskToken precheckpoint evidence archival was not atomic")
    if _read_object(archive / "precheckpoint_recovery.json") != recovery_record:
        raise RuntimeError("TaskToken precheckpoint recovery audit record changed")
    return archive


def _recover_incomplete_attempt(
    *,
    output_root: Path,
    run_root: Path,
    identity: TaskTokenRunIdentity,
    task_experiment: TaskTokenExperimentConfig,
) -> tuple[str | None, Path | None]:
    """Select exact checkpoint resume or audited step-zero reconstruction."""

    manifest = TaskTokenTrainingManifest.from_dict(_read_object(run_root / "run_manifest.json"))
    if manifest.identity != identity or canonical_fingerprint(
        manifest.task_token_experiment
    ) != canonical_fingerprint(task_experiment.to_dict()):
        raise RuntimeError("incomplete TaskToken run identity differs; refusing a second run")
    if manifest.checkpoints or _completed_checkpoint_paths(run_root):
        return _latest_safe_resume_checkpoint(run_root, manifest), None
    archive = _archive_precheckpoint_attempt(
        output_root=output_root,
        run_root=run_root,
        identity=identity,
        task_experiment=task_experiment,
    )
    return None, archive


def _queue(
    *,
    identity: TaskTokenRunIdentity,
    checkpoints: list[CheckpointRecord],
    losses: dict[int, float],
    validation_schedule_fingerprint: str,
    complete: bool,
) -> TaskTokenValidationQueue:
    items = tuple(
        TaskTokenValidationQueueItem(
            global_step=record.global_step,
            checkpoint_fingerprint=record.checkpoint_fingerprint,
            checkpoint_relative_path=record.relative_path,
            offline_validation_loss=losses[record.global_step],
        )
        for record in checkpoints
    )
    return TaskTokenValidationQueue(
        run_fingerprint=identity.run_fingerprint,
        m3b_dataset_fingerprint=identity.m3b_export_fingerprint,
        split_digest=identity.m3b_split_manifest_digest,
        train_statistics_fingerprint=identity.train_statistics_fingerprint,
        runtime_selection_fingerprint=identity.runtime_selection_fingerprint,
        experiment_manifest_fingerprint=identity.experiment_manifest_fingerprint,
        task_token_architecture_fingerprint=identity.task_token_architecture_fingerprint,
        validation_schedule_fingerprint=validation_schedule_fingerprint,
        ordered_validation_episode_indices=identity.ordered_validation_episode_indices,
        checkpoints=items,
        git_commit=identity.git_commit,
        complete=complete,
    )


def _training_summary(
    *,
    identity: TaskTokenRunIdentity,
    metrics_path: Path,
    parameter_count: int,
    duration_s: float | None,
    initial_validation_loss: float,
    final_checkpoint: CheckpointRecord,
    reloaded_components: object,
    validation_queue: TaskTokenValidationQueue,
) -> dict[str, object]:
    records = _metrics(metrics_path)
    if len(records) != 100_000 or records[-1]["step"] != 100_000:
        raise RuntimeError("TaskToken training summary requires all 100,000 metrics")
    losses = [
        float(record["validation_loss"])
        for record in records
        if record.get("validation_loss") is not None
    ]
    if len(losses) != 20 or not all(math.isfinite(value) for value in losses):
        raise RuntimeError("TaskToken training requires 20 finite offline validation losses")
    components = cast(Any, reloaded_components)
    return {
        "schema_version": TRAINING_SUMMARY_SCHEMA,
        "passed": True,
        "model_label": "act_mixed_task_token",
        "oracle_conditioned": True,
        "language_conditioned": False,
        "run_fingerprint": identity.run_fingerprint,
        "experiment_manifest_fingerprint": identity.experiment_manifest_fingerprint,
        "parameter_count": parameter_count,
        "training_duration_s": duration_s,
        "training_duration_complete": duration_s is not None,
        "initial_validation_loss": initial_validation_loss,
        "final_validation_loss": losses[-1],
        "checkpoint_count": 20,
        "final_checkpoint_fingerprint": final_checkpoint.checkpoint_fingerprint,
        "fresh_reload_component_fingerprints": {
            "model": components.model,
            "preprocessor": components.preprocessor,
            "postprocessor": components.postprocessor,
        },
        "validation_queue_fingerprint": validation_queue.queue_fingerprint,
        "checkpoint_selection_pending": True,
        "checkpoint_selection_source": "m3b_validation_only",
        "task_token_training_completed": True,
        "task_token_checkpoint_selected": False,
        "validation_only_selection_validated": False,
        "test_accessed": False,
        "development_schedule_accessed_for_selection": False,
        "final_schedule_accessed": False,
    }


def _repair_missing_completion_marker(
    *,
    run_root: Path,
    identity: TaskTokenRunIdentity,
    task_experiment: TaskTokenExperimentConfig,
) -> None:
    """Finish only the last atomic marker after a summary-write interruption."""
    manifest = TaskTokenTrainingManifest.from_dict(_read_object(run_root / "run_manifest.json"))
    queue = TaskTokenValidationQueue.from_dict(_read_object(run_root / "validation_queue.json"))
    summary = _read_object(run_root / "reports" / "training_summary.json")
    completion_path = run_root / "training_complete.json"
    if completion_path.exists():
        raise RuntimeError("completion-marker repair refuses to overwrite an existing path")
    if (
        manifest.identity != identity
        or canonical_fingerprint(manifest.task_token_experiment)
        != canonical_fingerprint(task_experiment.to_dict())
        or not manifest.training_complete
        or not queue.complete
        or queue.run_fingerprint != identity.run_fingerprint
        or queue.queue_fingerprint != summary.get("validation_queue_fingerprint")
        or summary.get("schema_version") != TRAINING_SUMMARY_SCHEMA
        or summary.get("passed") is not True
        or summary.get("run_fingerprint") != identity.run_fingerprint
        or summary.get("experiment_manifest_fingerprint")
        != identity.experiment_manifest_fingerprint
        or summary.get("final_checkpoint_fingerprint")
        != manifest.checkpoints[-1].checkpoint_fingerprint
        or summary.get("task_token_training_completed") is not True
    ):
        raise RuntimeError("cannot repair a mismatched TaskToken completion marker")
    atomic_write_json(
        completion_path,
        {
            "schema_version": TRAINING_COMPLETION_SCHEMA,
            "run_fingerprint": identity.run_fingerprint,
            "experiment_manifest_fingerprint": identity.experiment_manifest_fingerprint,
            "training_summary_fingerprint": f"sha256:{sha256_hex(summary)}",
            "validation_queue_fingerprint": queue.queue_fingerprint,
            "final_checkpoint_fingerprint": manifest.checkpoints[-1].checkpoint_fingerprint,
            "checkpoint_selection_pending": True,
        },
        immutable=True,
    )


def _reuse_completed_training(
    *,
    run_root: Path,
    identity: TaskTokenRunIdentity,
    task_experiment: TaskTokenExperimentConfig,
    report: dict[str, object],
    task_token: TaskTokenConfig,
) -> dict[str, object]:
    """Integrity-check and reuse one immutable, identity-equal completed run."""
    manifest_path = run_root / "run_manifest.json"
    queue_path = run_root / "validation_queue.json"
    summary_path = run_root / "reports" / "training_summary.json"
    completion_path = run_root / "training_complete.json"
    manifest = TaskTokenTrainingManifest.from_dict(_read_object(manifest_path))
    queue = TaskTokenValidationQueue.from_dict(_read_object(queue_path))
    summary = _read_object(summary_path)
    completion = _read_object(completion_path)
    if manifest.identity != identity or canonical_fingerprint(
        manifest.task_token_experiment
    ) != canonical_fingerprint(task_experiment.to_dict()):
        raise RuntimeError("completed TaskToken run differs from the requested identity")
    if not manifest.training_complete or not queue.complete:
        raise RuntimeError("completed TaskToken reuse requires complete manifest and queue")
    if (
        queue.run_fingerprint != identity.run_fingerprint
        or queue.m3b_dataset_fingerprint != identity.m3b_export_fingerprint
        or queue.split_digest != identity.m3b_split_manifest_digest
        or queue.train_statistics_fingerprint != identity.train_statistics_fingerprint
        or queue.runtime_selection_fingerprint != identity.runtime_selection_fingerprint
        or queue.experiment_manifest_fingerprint != identity.experiment_manifest_fingerprint
        or queue.task_token_architecture_fingerprint != identity.task_token_architecture_fingerprint
        or tuple(item.checkpoint_fingerprint for item in queue.checkpoints)
        != tuple(item.checkpoint_fingerprint for item in manifest.checkpoints)
    ):
        raise RuntimeError("completed TaskToken queue differs from its immutable manifest")
    final_checkpoint = manifest.checkpoints[-1]
    if (
        completion.get("schema_version") != TRAINING_COMPLETION_SCHEMA
        or completion.get("run_fingerprint") != identity.run_fingerprint
        or completion.get("experiment_manifest_fingerprint")
        != identity.experiment_manifest_fingerprint
        or completion.get("training_summary_fingerprint") != f"sha256:{sha256_hex(summary)}"
        or completion.get("validation_queue_fingerprint") != queue.queue_fingerprint
        or completion.get("final_checkpoint_fingerprint") != final_checkpoint.checkpoint_fingerprint
        or completion.get("checkpoint_selection_pending") is not True
        or summary.get("schema_version") != TRAINING_SUMMARY_SCHEMA
        or summary.get("passed") is not True
        or summary.get("run_fingerprint") != identity.run_fingerprint
        or summary.get("experiment_manifest_fingerprint")
        != identity.experiment_manifest_fingerprint
        or summary.get("validation_queue_fingerprint") != queue.queue_fingerprint
        or summary.get("final_checkpoint_fingerprint") != final_checkpoint.checkpoint_fingerprint
        or summary.get("task_token_training_completed") is not True
    ):
        raise RuntimeError("completed TaskToken summary/completion evidence is inconsistent")
    for checkpoint in manifest.checkpoints:
        marker = run_root / checkpoint.relative_path / CHECKPOINT_COMPLETION_MARKER
        if not marker.is_file() or marker.is_symlink():
            raise RuntimeError(f"completed TaskToken checkpoint marker is absent: {marker}")
    reloaded = load_act_checkpoint(
        run_root=run_root,
        checkpoint_relative_path=final_checkpoint.relative_path,
        expected_identity=identity,
    )
    validate_task_token_policy(cast(Any, reloaded.policy), token_config=task_token)
    if reloaded.record.checkpoint_fingerprint != final_checkpoint.checkpoint_fingerprint:
        raise RuntimeError("fresh TaskToken reuse reload changed the final checkpoint")
    return {
        **report,
        "passed": True,
        "training_started": False,
        "training_reused": True,
        "final_step": 100_000,
        "checkpoint_count": 20,
        "run_manifest": str(manifest_path),
        "training_summary": str(summary_path),
        "validation_queue": str(queue_path),
        "validation_queue_fingerprint": queue.queue_fingerprint,
        "final_checkpoint": final_checkpoint.to_dict(),
        "checkpoint_selection_pending": True,
        "task_token_training_completed": True,
        "task_token_checkpoint_selected": False,
        "validation_only_selection_validated": False,
    }


def _finalization_only_outcome(state: TrainingState) -> TrainingOutcome | None:
    """Represent a promoted 100k checkpoint that only needs final evidence writes."""
    if state.global_step < 100_000:
        return None
    if state.global_step != 100_000:
        raise RuntimeError("TaskToken resume state exceeds the frozen 100k training bound")
    return TrainingOutcome(
        final_step=state.global_step,
        examples_processed=state.examples_processed,
        metrics=(),
    )


def execute(args: argparse.Namespace) -> dict[str, object]:
    if args.resume_checkpoint is not None and args.dry_run:
        raise ValueError("--resume-checkpoint is invalid with --dry-run")
    if args.full and args.device != "cuda":
        raise ValueError("full TaskToken training requires explicit --device cuda")
    _validate_paths(args)
    git = inspect_git_state(PROJECT_ROOT)
    if not git.baseline_tracked or git.dirty:
        raise RuntimeError("TaskToken dry/full identities require a clean tracked Git worktree")
    runtime = load_runtime_selection(args.runtime_selection)
    experiment = _internal_experiment(args.dataset_root, device=args.device)
    completed, statistics = _load_data_and_statistics(args.dataset_root)
    task_token = TaskTokenConfig(hidden_dimension=experiment.model.dim_model)
    act_config = build_task_token_act_config(
        experiment.model,
        device=experiment.device,
        # A CPU dry-run validates shapes/identity without pretending to be the
        # bfloat16 target run.  Full mode above requires CUDA, where AMP is on.
        use_amp=(experiment.device == "cuda" and experiment.optimization.mixed_precision != "none"),
        optimization=experiment.optimization,
        token_config=task_token,
    )
    fair_comparison = _fair_comparison_contract(runtime)
    fair_comparison_fingerprint = _validate_fair_comparison(
        contract=fair_comparison,
        experiment=experiment,
        completed=completed,
        effective_act_config=act_config,
        dry_run=bool(args.dry_run),
    )
    identity, task_experiment, validation_schedule_fingerprint = _identity(
        experiment=experiment,
        completed=completed,
        statistics=statistics,
        runtime=runtime,
        task_token=task_token,
        effective_act_config=act_config,
        fair_comparison_fingerprint=fair_comparison_fingerprint,
        git_commit=git.commit,
        git_dirty=git.dirty,
    )
    run_root = safe_run_directory(args.output_root, identity.run_fingerprint)
    report = {
        "schema_version": "langmani-m42-task-token-train-command-v0",
        "model_label": "act_mixed_task_token",
        "oracle_conditioned": True,
        "language_conditioned": False,
        "dataset_fingerprint": completed.export_fingerprint,
        "split_digest": completed.split_manifest_digest,
        "train_episode_count": len(completed.views.train.episode_indices),
        "validation_episode_count": len(completed.views.validation.episode_indices),
        "test_episode_count_used_for_training_or_selection": 0,
        "train_statistics_fingerprint": statistics.statistics_fingerprint,
        "runtime_selection_fingerprint": runtime.runtime_fingerprint,
        "runtime_selection_source_fingerprint": runtime.source_fingerprint,
        "experiment_manifest_fingerprint": runtime.experiment_manifest_fingerprint,
        "task_token_architecture_fingerprint": identity.task_token_architecture_fingerprint,
        "task_token_fair_comparison_fingerprint": fair_comparison_fingerprint,
        "task_token_fair_comparison_validated": True,
        "task_token_fair_comparison_source_run": fair_comparison.source_run_fingerprint,
        "task_token_fair_comparison_source_checkpoint": (
            fair_comparison.source_checkpoint_fingerprint
        ),
        "task_token_experiment_fingerprint": task_experiment.fingerprint,
        "task_token_experiment_content_fingerprint": canonical_fingerprint(
            {
                key: value
                for key, value in task_experiment.to_dict().items()
                if key != "implementation_git_commit"
            }
        ),
        "effective_act_config": _effective_act_config_dict(act_config),
        "model_contract_fingerprint": canonical_fingerprint(
            cast(Mapping[str, object], identity.to_dict()["model_config"])
        ),
        "data_contract_fingerprint": canonical_fingerprint(
            cast(Mapping[str, object], identity.to_dict()["data_contract"])
        ),
        "task_token_training_source_fingerprint": _training_source_fingerprint(),
        "task_token_training_source_files": list(TASK_TOKEN_TRAINING_SOURCE_FILES),
        "validation_schedule_fingerprint": validation_schedule_fingerprint,
        "input_shapes": {
            experiment.model.image_feature_key: list(experiment.model.image_shape_chw),
            experiment.model.state_feature_key: [9],
            TASK_TOKEN_FEATURE_KEY: [6],
        },
        "output_shape": [8],
        "optimization": experiment.optimization.to_dict(),
        "run_fingerprint": identity.run_fingerprint,
        "expected_output_directory": str(run_root),
        "checkpoint_schedule": list(range(5_000, 100_001, 5_000)),
        "checkpoint_selection_source": "m3b_validation_only",
        "forbidden_selection_sources": [
            "m3b_test",
            "m4_fresh_seed",
            "m42_dev_v0",
            "m42_final_v0",
        ],
        "git": git.to_dict(),
        "dry_run": bool(args.dry_run),
        "task_token_training_completed": False,
        "task_token_checkpoint_selected": False,
        "validation_only_selection_validated": False,
        "precheckpoint_recovery_archived": False,
        "precheckpoint_recovery_archive": None,
    }
    if args.dry_run:
        return {**report, "passed": True, "training_started": False}

    _enforce_single_task_token_run(args.output_root, run_root)
    resuming = args.resume_checkpoint is not None
    manifest_path = run_root / "run_manifest.json"
    metrics_path = run_root / "metrics.jsonl"
    queue_path = run_root / "validation_queue.json"
    summary_path = run_root / "reports" / "training_summary.json"
    completion_path = run_root / "training_complete.json"
    if run_root.exists() and not resuming:
        if completion_path.is_file() and summary_path.is_file():
            return _reuse_completed_training(
                run_root=run_root,
                identity=identity,
                task_experiment=task_experiment,
                report=report,
                task_token=task_token,
            )
        if summary_path.is_file() and not completion_path.exists():
            _repair_missing_completion_marker(
                run_root=run_root,
                identity=identity,
                task_experiment=task_experiment,
            )
            return _reuse_completed_training(
                run_root=run_root,
                identity=identity,
                task_experiment=task_experiment,
                report=report,
                task_token=task_token,
            )
        if completion_path.exists() or not manifest_path.is_file():
            raise RuntimeError("incomplete TaskToken run has an unsafe finalization layout")
        resume_checkpoint, archive = _recover_incomplete_attempt(
            output_root=args.output_root,
            run_root=run_root,
            identity=identity,
            task_experiment=task_experiment,
        )
        if resume_checkpoint is not None:
            args.resume_checkpoint = resume_checkpoint
            resuming = True
        else:
            if archive is None:
                raise RuntimeError("TaskToken incomplete recovery produced no auditable outcome")
            report["precheckpoint_recovery_archived"] = True
            report["precheckpoint_recovery_archive"] = str(archive)
    if resuming and not run_root.is_dir():
        raise FileNotFoundError("resume requires the fingerprint-owned TaskToken run directory")
    run_root.mkdir(parents=True, exist_ok=True)
    if completion_path.exists() or summary_path.exists():
        raise RuntimeError("completed TaskToken training is immutable and cannot be resumed")

    existing_manifest: TaskTokenTrainingManifest | None = None
    orphan = False
    if resuming:
        existing_manifest = TaskTokenTrainingManifest.from_dict(_read_object(manifest_path))
        if existing_manifest.identity != identity:
            raise RuntimeError("resume identity differs from the requested TaskToken run")
        if canonical_fingerprint(existing_manifest.task_token_experiment) != canonical_fingerprint(
            task_experiment.to_dict()
        ):
            raise RuntimeError("resume TaskToken experiment contract changed")
        orphan = _resume_kind(
            run_root,
            requested=args.resume_checkpoint,
            manifest=existing_manifest,
        )

    _write_or_validate(run_root / "task_token_experiment.json", task_experiment.to_dict())
    _write_or_validate(
        run_root / "dataset_contract.json",
        cast(dict[str, object], identity.to_dict()["data_contract"]),
    )
    _write_or_validate(run_root / "train_stats.json", statistics.to_dict())
    _write_or_validate(run_root / "runtime_selection.json", runtime.to_dict())
    _write_or_validate(run_root / "task_token_contract.json", task_token_contract(task_token))

    temporal_train = load_lerobot_episode_view(
        completed,
        completed.views.train,
        policy_config=act_config,
        include_delta_timestamps=True,
        return_uint8=True,
    )
    temporal_validation = load_lerobot_episode_view(
        completed,
        completed.views.validation,
        policy_config=act_config,
        include_delta_timestamps=True,
        return_uint8=True,
    )
    validate_temporal_episode_boundaries(temporal_train, chunk_size=50)
    validate_temporal_episode_boundaries(temporal_validation, chunk_size=50)
    seed_everything(experiment.seed)
    policy, preprocessor, postprocessor = build_policy_and_processors(
        act_config,
        statistics.to_processor_stats(),
    )
    validate_task_token_policy(policy, token_config=task_token)
    optimizer = make_optimizer(policy, experiment)
    start_state = TrainingState(global_step=0, examples_processed=0)
    checkpoints = list(existing_manifest.checkpoints if existing_manifest is not None else ())
    checkpoint_losses: dict[int, float] = {
        cast(int, record["step"]): float(record["validation_loss"])
        for record in _metrics(metrics_path)
        if record.get("validation_loss") is not None
    }
    if resuming:
        loaded = load_act_checkpoint(
            run_root=run_root,
            checkpoint_relative_path=args.resume_checkpoint,
            expected_identity=identity,
            optimizer=optimizer,
            for_resume=True,
            existing_policy=policy,
        )
        policy = cast(Any, loaded.policy)
        preprocessor = cast(Any, loaded.preprocessor)
        postprocessor = cast(Any, loaded.postprocessor)
        optimizer = cast(Any, loaded.optimizer)
        if optimizer is None:
            raise RuntimeError("TaskToken resume failed to restore optimizer state")
        validate_task_token_policy(policy, token_config=task_token)
        start_state = loaded.training_state
        if orphan:
            expected_step = (checkpoints[-1].global_step if checkpoints else 0) + 5_000
            if loaded.record.global_step != expected_step:
                raise RuntimeError("atomic orphan is not the next configured checkpoint")
            checkpoints.append(loaded.record)
        repaired = _repair_metrics_for_resume(
            metrics_path,
            checkpoint=loaded.record,
            checkpoint_metric=(
                dict(loaded.training_metric) if loaded.training_metric is not None else None
            ),
        )
        checkpoint_losses = {
            cast(int, record["step"]): float(record["validation_loss"])
            for record in repaired
            if record.get("validation_loss") is not None
        }
        if orphan:
            recovered_state = replace(
                start_state,
                last_checkpoint_fingerprint=loaded.record.checkpoint_fingerprint,
            )
            atomic_write_json(
                manifest_path,
                _manifest(
                    identity=identity,
                    task_experiment=task_experiment,
                    state=recovered_state,
                    checkpoints=tuple(checkpoints),
                    complete=False,
                ).to_dict(),
            )

    validation_loader = _loader(
        temporal_validation.dataset,
        experiment,
        shuffle=False,
    )

    def validation_callback(step: int, current_policy: object) -> float:
        devices = list(range(torch.cuda.device_count())) if experiment.device == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            validation_seed = experiment.seed + 1_000_003 + step
            torch.manual_seed(validation_seed)
            if experiment.device == "cuda":
                torch.cuda.manual_seed_all(validation_seed)
            return evaluate_offline_loss(
                experiment=experiment,
                policy=cast(Any, current_policy),
                preprocessor=preprocessor,
                validation_batches=validation_loader,
                task_id_by_episode=completed.views.task_id_by_episode,
                task_token_enabled=True,
            )

    initial_validation_path = run_root / "reports" / "initial_validation.json"
    if resuming:
        initial_payload = _read_object(initial_validation_path)
        if initial_payload.get("run_fingerprint") != identity.run_fingerprint or not isinstance(
            initial_payload.get("loss"), int | float
        ):
            raise RuntimeError("initial TaskToken validation evidence is mismatched")
        initial_validation_loss = float(initial_payload["loss"])
    else:
        initial_validation_loss = validation_callback(0, policy)
        atomic_write_json(
            initial_validation_path,
            {
                "schema_version": "langmani-m42-task-token-initial-validation-v0",
                "run_fingerprint": identity.run_fingerprint,
                "split_digest": completed.split_manifest_digest,
                "loss": initial_validation_loss,
                "test_accessed": False,
                "development_schedule_accessed": False,
                "final_schedule_accessed": False,
            },
            immutable=True,
        )
        atomic_write_json(
            manifest_path,
            _manifest(
                identity=identity,
                task_experiment=task_experiment,
                state=start_state,
                checkpoints=(),
                complete=False,
            ).to_dict(),
            immutable=True,
        )

    def save_checkpoint_callback(**values: object) -> str:
        step = int(values["step"])
        metric = cast(dict[str, object], values["metric"])
        validation_loss = metric.get("validation_loss")
        if not isinstance(validation_loss, int | float) or not math.isfinite(validation_loss):
            raise RuntimeError("every TaskToken checkpoint requires finite offline validation loss")
        checkpoint_state = TrainingState(
            global_step=step,
            examples_processed=int(values["examples_processed"]),
        )
        record = save_act_checkpoint(
            run_root=run_root,
            identity=identity,
            training_state=checkpoint_state,
            policy=cast(Any, values["policy"]),
            preprocessor=cast(Any, values["preprocessor"]),
            postprocessor=cast(Any, values["postprocessor"]),
            optimizer=cast(Any, values["optimizer"]),
            training_metric=metric,
        )
        checkpoints.append(record)
        checkpoint_losses[step] = float(validation_loss)
        manifest_state = replace(
            checkpoint_state,
            last_checkpoint_fingerprint=record.checkpoint_fingerprint,
        )
        atomic_write_json(
            manifest_path,
            _manifest(
                identity=identity,
                task_experiment=task_experiment,
                state=manifest_state,
                checkpoints=tuple(checkpoints),
                complete=False,
            ).to_dict(),
        )
        atomic_write_json(
            queue_path,
            _queue(
                identity=identity,
                checkpoints=checkpoints,
                losses=checkpoint_losses,
                validation_schedule_fingerprint=validation_schedule_fingerprint,
                complete=False,
            ).to_dict(),
        )
        return record.relative_path

    finalized_resume = _finalization_only_outcome(start_state)
    finalization_only = finalized_resume is not None
    if finalization_only:
        outcome = cast(TrainingOutcome, finalized_resume)
        duration_s = None
    else:
        training_started = time.perf_counter()
        train_loader = _loader(
            temporal_train.dataset,
            experiment,
            shuffle=True,
            start_step=start_state.global_step,
        )
        outcome = train_act(
            experiment=experiment,
            policy=policy,
            preprocessor=preprocessor,
            optimizer=optimizer,
            train_batches=train_loader,
            start_step=start_state.global_step,
            maximum_steps=100_000,
            metrics_path=metrics_path,
            checkpoint_callback=save_checkpoint_callback,
            postprocessor=postprocessor,
            validation_callback=validation_callback,
            task_id_by_episode=completed.views.task_id_by_episode,
            task_token_enabled=True,
            seed_at_start=not resuming,
            initial_examples_processed=start_state.examples_processed,
        )
        duration_s = time.perf_counter() - training_started if not resuming else None
    if outcome.final_step != 100_000 or len(checkpoints) != 20:
        raise RuntimeError("TaskToken training did not produce the exact 100k/20-checkpoint result")
    final_state = TrainingState(
        global_step=outcome.final_step,
        examples_processed=outcome.examples_processed,
        last_checkpoint_fingerprint=checkpoints[-1].checkpoint_fingerprint,
        completed=True,
    )
    final_manifest = _manifest(
        identity=identity,
        task_experiment=task_experiment,
        state=final_state,
        checkpoints=tuple(checkpoints),
        complete=True,
    )
    final_queue = _queue(
        identity=identity,
        checkpoints=checkpoints,
        losses=checkpoint_losses,
        validation_schedule_fingerprint=validation_schedule_fingerprint,
        complete=True,
    )
    atomic_write_json(manifest_path, final_manifest.to_dict())
    atomic_write_json(queue_path, final_queue.to_dict())
    reloaded = load_act_checkpoint(
        run_root=run_root,
        checkpoint_relative_path=checkpoints[-1].relative_path,
        expected_identity=identity,
    )
    validate_task_token_policy(cast(Any, reloaded.policy), token_config=task_token)
    summary = _training_summary(
        identity=identity,
        metrics_path=metrics_path,
        parameter_count=sum(parameter.numel() for parameter in policy.parameters()),
        duration_s=duration_s,
        initial_validation_loss=initial_validation_loss,
        final_checkpoint=checkpoints[-1],
        reloaded_components=reloaded.component_fingerprints,
        validation_queue=final_queue,
    )
    atomic_write_json(summary_path, summary, immutable=True)
    atomic_write_json(
        completion_path,
        {
            "schema_version": TRAINING_COMPLETION_SCHEMA,
            "run_fingerprint": identity.run_fingerprint,
            "experiment_manifest_fingerprint": identity.experiment_manifest_fingerprint,
            "training_summary_fingerprint": f"sha256:{sha256_hex(summary)}",
            "validation_queue_fingerprint": final_queue.queue_fingerprint,
            "final_checkpoint_fingerprint": checkpoints[-1].checkpoint_fingerprint,
            "checkpoint_selection_pending": True,
        },
        immutable=True,
    )
    return {
        **report,
        "passed": True,
        "training_started": not finalization_only,
        "training_reused": False,
        "finalization_only": finalization_only,
        "final_step": 100_000,
        "checkpoint_count": 20,
        "run_manifest": str(manifest_path),
        "training_summary": str(summary_path),
        "validation_queue": str(queue_path),
        "validation_queue_fingerprint": final_queue.queue_fingerprint,
        "final_checkpoint": checkpoints[-1].to_dict(),
        "checkpoint_selection_pending": True,
        "task_token_training_completed": True,
        "task_token_checkpoint_selected": False,
        "validation_only_selection_validated": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = execute(args)
    except Exception as error:  # noqa: BLE001 - CLI boundary preserves diagnostics
        traceback.print_exc()
        report = {
            "schema_version": "langmani-m42-task-token-train-command-v0",
            "passed": False,
            "error_type": type(error).__name__,
            "error_message": str(error) or repr(error),
            "task_token_training_completed": False,
            "task_token_checkpoint_selected": False,
            "validation_only_selection_validated": False,
        }
    try:
        report_path = _validated_report_path(args)
        atomic_write_json(report_path, report)
    except Exception:  # noqa: BLE001 - preserve the original command failure
        if report.get("passed") is True:
            raise
        traceback.print_exc()
    print(json.dumps({**report, "report": str(args.report.resolve())}, sort_keys=True))
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
