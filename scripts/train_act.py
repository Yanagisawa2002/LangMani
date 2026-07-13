"""Train one provenance-bound LangMani ACT baseline on a completed M3B dataset."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from langmani.datasets.lerobot_types import DatasetSplit
from langmani.policies.act_analysis import audit_m3b_train_counterfactuals
from langmani.policies.act_checkpoint import (
    CHECKPOINT_COMPLETION_MARKER,
    load_act_checkpoint,
    save_act_checkpoint,
)
from langmani.policies.act_conditioning import CANONICAL_TASK_IDS, task_onehot_contract
from langmani.policies.act_data import (
    CompletedM3BDataset,
    DatasetEpisodeView,
    TrainOnlyStatistics,
    compute_train_only_statistics,
    load_completed_m3b_dataset,
    load_lerobot_episode_view,
    validate_temporal_episode_boundaries,
)
from langmani.policies.act_evaluation import (
    build_fresh_seed_schedule,
    rollout_schedule_digest,
)
from langmani.policies.act_runtime import (
    atomic_write_json,
    inspect_git_state,
    runtime_versions,
    safe_run_directory,
    validate_git_for_run,
)
from langmani.policies.act_training import (
    DeterministicResumeBatchSampler,
    build_act_config,
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
    ActExperimentManifest,
    ActModelConfig,
    ActOptimizationConfig,
    ActRunIdentity,
    ActVariant,
    CheckpointRecord,
    ExperimentMode,
    TrainingState,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "outputs" / "datasets" / "m3b" / "langmani-pick-place-lerobot-v1"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "models" / "act"
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "diagnostics" / "m4" / "train_act.json"


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected one JSON object in {path}")
    return value


def _write_or_validate_json(path: Path, payload: dict[str, object]) -> None:
    if path.is_file():
        if _read_object(path) != payload:
            raise RuntimeError(f"existing run artifact differs from requested identity: {path}")
        return
    atomic_write_json(path, payload, immutable=True)


def _resume_checkpoint_is_recoverable_orphan(
    run_root: Path,
    *,
    requested_relative_path: str,
    declared_relative_paths: tuple[str, ...],
) -> bool:
    """Allow exactly one promoted checkpoint from the manifest-update crash window."""

    completed = {
        marker.parent.relative_to(run_root).as_posix()
        for marker in (run_root / "checkpoints").glob(f"*/{CHECKPOINT_COMPLETION_MARKER}")
    }
    declared = set(declared_relative_paths)
    orphans = completed - declared
    if declared_relative_paths and requested_relative_path == declared_relative_paths[-1]:
        if orphans:
            raise RuntimeError(
                "a newer promoted checkpoint is absent from the run manifest; resume that orphan"
            )
        return False
    if requested_relative_path in orphans and orphans == {requested_relative_path}:
        return True
    raise RuntimeError(
        "resume must select the latest manifest checkpoint or the sole recoverable promoted orphan"
    )


def _validate_recoverable_orphan_step(
    orphan_step: int,
    *,
    declared_steps: tuple[int, ...],
    checkpoint_interval: int,
    training_steps: int,
) -> None:
    """Require an orphan to be the next scheduled checkpoint or final step."""

    previous_step = declared_steps[-1] if declared_steps else 0
    expected_next_step = min(previous_step + checkpoint_interval, training_steps)
    if orphan_step != expected_next_step:
        raise RuntimeError("recoverable orphan step is not the next declared checkpoint/final step")


def _truncate_metrics_to_checkpoint(
    path: Path,
    checkpoint_step: int,
    *,
    recovery_record: dict[str, object] | None,
) -> None:
    retained: list[str] = []
    previous = 0
    found_checkpoint = False
    checkpoint_record: dict[str, object] | None = None
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else ()
    for line in lines:
        value = json.loads(line)
        if not isinstance(value, dict) or not isinstance(value.get("step"), int):
            raise RuntimeError("metrics.jsonl contains a malformed record")
        step = int(value["step"])
        if step <= previous:
            raise RuntimeError("metrics.jsonl steps must be strictly increasing")
        previous = step
        if step <= checkpoint_step:
            retained.append(json.dumps(value, sort_keys=True))
            found_checkpoint |= step == checkpoint_step
            if step == checkpoint_step:
                checkpoint_record = value
    if found_checkpoint and recovery_record is not None and checkpoint_record != recovery_record:
        unbound_recovery = {**recovery_record, "checkpoint_path": None}
        if checkpoint_record == unbound_recovery:
            retained[-1] = json.dumps(recovery_record, sort_keys=True)
        else:
            raise RuntimeError(
                "metrics.jsonl checkpoint record differs from checkpoint-bound evidence"
            )
    if not found_checkpoint:
        if recovery_record is None or recovery_record.get("step") != checkpoint_step:
            raise RuntimeError(
                "metrics.jsonl lacks the resumed checkpoint step and checkpoint recovery evidence"
            )
        if previous >= checkpoint_step:
            raise RuntimeError("checkpoint metric recovery would violate strict step ordering")
        retained.append(json.dumps(recovery_record, sort_keys=True))
    staging = path.with_name(f".{path.name}.resume-staging")
    with staging.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(retained) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staging, path)


def _bind_metric_checkpoint_path(path: Path, *, step: int, relative_path: str) -> None:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    matches = [value for value in records if isinstance(value, dict) and value.get("step") == step]
    if len(matches) != 1 or matches[0].get("checkpoint_path") is not None:
        raise RuntimeError("final-step metric cannot be bound to its checkpoint path")
    matches[0]["checkpoint_path"] = relative_path
    staging = path.with_name(f".{path.name}.checkpoint-path-staging")
    with staging.open("w", encoding="utf-8", newline="\n") as stream:
        for value in records:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staging, path)


def _training_summary_from_metrics(
    *,
    metrics_path: Path,
    identity: ActRunIdentity,
    config: ActExperimentConfig,
    policy: object,
    training_wall_duration_s: float | None,
    initial_validation_loss: float,
) -> dict[str, object]:
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    if not records or not all(isinstance(item, dict) for item in records):
        raise RuntimeError("training summary requires nonempty structured metrics")
    validation_losses = [
        float(item["validation_loss"])
        for item in records
        if item.get("validation_loss") is not None
    ]
    return {
        "schema_version": "langmani-m4-training-summary-v1",
        "passed": True,
        "run_fingerprint": identity.run_fingerprint,
        "variant": config.variant.value,
        "task_id": config.task_id,
        "parameter_count": sum(parameter.numel() for parameter in policy.parameters()),
        "training_duration_s": training_wall_duration_s,
        "training_duration_complete": training_wall_duration_s is not None,
        "measured_dataloader_and_optimizer_duration_s": sum(
            float(item["dataloader_time_s"]) + float(item["step_time_s"]) for item in records
        ),
        "initial_train_loss": float(records[0]["total_loss"]),
        "final_train_loss": float(records[-1]["total_loss"]),
        "initial_validation_loss": initial_validation_loss,
        "final_validation_loss": validation_losses[-1] if validation_losses else None,
        "peak_gpu_allocated_bytes": max(int(item["gpu_allocated_bytes"]) for item in records),
        "peak_gpu_reserved_bytes": max(int(item["gpu_reserved_bytes"]) for item in records),
        "mean_throughput_examples_per_s": sum(
            float(item["throughput_examples_per_s"]) for item in records
        )
        / len(records),
        "effective_model_config": config.model.to_dict(),
        "effective_optimization_config": config.optimization.to_dict(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--variant", choices=[item.value for item in ActVariant], required=True)
    parser.add_argument("--task-id", help="Stable task ID or red_cube__left_bin style alias")
    parser.add_argument("--seed", type=_non_negative_int, default=0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--tiny-overfit", action="store_true")
    modes.add_argument("--development", action="store_true")
    modes.add_argument("--full", action="store_true")
    parser.add_argument("--allow-dirty-development", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=_positive_int)
    parser.add_argument("--steps", type=_positive_int)
    parser.add_argument("--dataloader-workers", type=_non_negative_int)
    parser.add_argument("--chunk-size", type=_positive_int)
    parser.add_argument("--n-action-steps", type=_positive_int)
    parser.add_argument("--resume-checkpoint")
    return parser.parse_args()


def _mode(args: argparse.Namespace) -> ExperimentMode:
    if args.dry_run:
        return ExperimentMode.DRY_RUN
    if args.tiny_overfit:
        return ExperimentMode.TINY_OVERFIT
    if args.development:
        return ExperimentMode.DEVELOPMENT
    return ExperimentMode.FULL


def _task_id(value: str | None) -> str | None:
    if value is None:
        return None
    if value in CANONICAL_TASK_IDS:
        return value
    aliases = {"__".join(task_id.split(":")[-3:-1]): task_id for task_id in CANONICAL_TASK_IDS}
    try:
        return aliases[value]
    except KeyError as error:
        raise ValueError(
            f"unknown task ID {value!r}; expected one canonical stable ID or {sorted(aliases)}"
        ) from error


def _tiny_view(
    completed: CompletedM3BDataset,
    *,
    variant: ActVariant,
    task_id: str | None,
) -> DatasetEpisodeView:
    if variant is ActVariant.MIXED_UNCONDITIONED:
        raise ValueError("mixed_unconditioned intentionally has no six-task tiny-overfit gate")
    if variant is ActVariant.PER_TASK:
        assert task_id is not None
        indices = completed.views.for_split(DatasetSplit.TRAIN, task_id=task_id).episode_indices[:2]
    else:
        first_index = completed.views.train.episode_indices[0]
        first_group = completed.views.scene_group_id_by_episode[first_index]
        indices = tuple(
            index
            for index in completed.views.train.episode_indices
            if completed.views.scene_group_id_by_episode[index] == first_group
        )
        if len(indices) != 6:
            raise RuntimeError("tiny task-one-hot view is not one complete counterfactual group")
    return DatasetEpisodeView(
        split=DatasetSplit.TRAIN, episode_indices=tuple(indices), task_id=task_id
    )


def _configs(args: argparse.Namespace) -> tuple[ActExperimentConfig, str | None]:
    variant = ActVariant(args.variant)
    selected_task = _task_id(args.task_id)
    model_values: dict[str, object] = {}
    if args.chunk_size is not None:
        model_values["chunk_size"] = args.chunk_size
    if args.n_action_steps is not None:
        model_values["n_action_steps"] = args.n_action_steps
    model = ActModelConfig.for_variant(variant, **model_values)
    optimization = ActOptimizationConfig()
    overrides: dict[str, object] = {}
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.steps is not None:
        overrides["training_steps"] = args.steps
        if args.steps < optimization.checkpoint_interval:
            overrides["checkpoint_interval"] = args.steps
        if args.steps < optimization.validation_interval:
            overrides["validation_interval"] = args.steps
    if args.dataloader_workers is not None:
        overrides["dataloader_workers"] = args.dataloader_workers
    if args.device == "cpu":
        overrides["mixed_precision"] = "none"
    optimization = replace(optimization, **overrides)
    config = ActExperimentConfig(
        variant=variant,
        task_id=selected_task,
        data=ActDataConfig(dataset_root=str(args.dataset_root.resolve())),
        model=model,
        optimization=optimization,
        seed=args.seed,
        mode=_mode(args),
        device=args.device,
        allow_dirty_development=args.allow_dirty_development,
    )
    if config.mode is ExperimentMode.FULL:
        if config.model != ActModelConfig.for_variant(variant):
            raise ValueError("full M4 runs must use the fixed primary ACT model configuration")
        if config.optimization != ActOptimizationConfig():
            raise ValueError("full M4 runs must use the fixed primary optimization configuration")
    return config, selected_task


def _data_and_statistics(
    config: ActExperimentConfig,
) -> tuple[CompletedM3BDataset, DatasetEpisodeView, DatasetEpisodeView, TrainOnlyStatistics]:
    require_full = config.mode is ExperimentMode.FULL
    completed = load_completed_m3b_dataset(
        config.data.dataset_root,
        require_full=require_full,
        validate_storage=True,
    )
    if config.mode is ExperimentMode.TINY_OVERFIT:
        train_view = _tiny_view(
            completed,
            variant=config.variant,
            task_id=config.task_id,
        )
        validation_view = train_view
    else:
        train_view = completed.views.for_split(DatasetSplit.TRAIN, task_id=config.task_id)
        validation_view = completed.views.for_split(DatasetSplit.VALIDATION, task_id=config.task_id)
        if not validation_view.episode_indices and config.mode is ExperimentMode.DEVELOPMENT:
            validation_view = train_view
    current_frames = load_lerobot_episode_view(
        completed,
        train_view,
        policy_config=None,
        include_delta_timestamps=False,
        return_uint8=True,
    )
    statistics = compute_train_only_statistics(
        current_frames.dataset,
        train_episode_indices=train_view.episode_indices,
        validation_episode_indices=(
            completed.views.validation.episode_indices
            if config.mode is not ExperimentMode.TINY_OVERFIT
            else ()
        ),
        test_episode_indices=(
            completed.views.test.episode_indices
            if config.mode is not ExperimentMode.TINY_OVERFIT
            else ()
        ),
        task_id_by_episode=completed.views.task_id_by_episode,
        variant=config.variant,
        task_id=config.task_id,
    )
    return completed, train_view, validation_view, statistics


def _identity(
    *,
    config: ActExperimentConfig,
    completed: CompletedM3BDataset,
    train_view: DatasetEpisodeView,
    validation_view: DatasetEpisodeView,
    statistics: TrainOnlyStatistics,
    git_commit: str,
    git_dirty: bool,
) -> ActRunIdentity:
    versions = runtime_versions()
    data_contract = {
        **config.data.identity_dict(),
        "m3b_feature_contract": completed.manifest.config.feature_contract.to_dict(),
        "policy_state_schema": completed.manifest.config.policy_state_schema.to_dict(),
        "task_onehot": task_onehot_contract(),
        "evaluation_config": config.evaluation.to_dict(),
        "evaluation_schedules": _evaluation_schedule_contract(completed, config),
    }
    return ActRunIdentity(
        m3b_export_fingerprint=completed.export_fingerprint,
        m3b_split_manifest_digest=completed.split_manifest_digest,
        ordered_train_episode_indices=train_view.episode_indices,
        ordered_validation_episode_indices=validation_view.episode_indices,
        variant=config.variant,
        task_id=config.task_id,
        task_onehot_mapping_version="CanonicalTaskOneHotV0",
        model_config=config.model.to_dict(),
        data_contract=data_contract,
        train_statistics_fingerprint=statistics.statistics_fingerprint,
        optimization_config=config.optimization.to_dict(),
        training_seed=config.seed,
        experiment_mode=config.mode,
        device=config.device,
        dtype=config.dtype,
        lerobot_version=str(versions["lerobot"]),
        torch_version=str(versions["torch"]),
        cuda_version=versions["cuda"],
        git_commit=git_commit,
        git_dirty=git_dirty,
        dirty_development_override=config.allow_dirty_development,
    )


def _loader(
    dataset: Any,
    config: ActExperimentConfig,
    *,
    shuffle: bool,
    start_step: int = 0,
) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(config.seed)
    common = {
        "dataset": dataset,
        "num_workers": config.optimization.dataloader_workers,
        "pin_memory": config.device == "cuda",
        "persistent_workers": config.optimization.dataloader_workers > 0,
        "worker_init_fn": seed_dataloader_worker,
        "generator": generator,
    }
    if shuffle:
        return DataLoader(
            **common,
            batch_sampler=DeterministicResumeBatchSampler(
                dataset_size=len(dataset),
                batch_size=config.optimization.batch_size,
                seed=config.seed,
                start_step=start_step,
            ),
        )
    return DataLoader(
        **common,
        batch_size=config.optimization.batch_size,
        shuffle=False,
        drop_last=False,
    )


def _evaluation_schedule_contract(
    completed: CompletedM3BDataset,
    config: ActExperimentConfig,
) -> dict[str, object]:
    records = completed.manifest.episodes
    first_group_id = records[0].source_scene_group_id
    first_group = {
        item.task_id: item for item in records if item.source_scene_group_id == first_group_id
    }
    if set(first_group) != set(CANONICAL_TASK_IDS):
        raise RuntimeError("M3B schedule authority lacks one canonical six-task group")

    def payload_for_records(selected: tuple[object, ...]) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "scene_seed": item.source_scene_seed,
                "task_spec": {
                    "target_object_id": item.target_object_id,
                    "target_bin_id": item.target_bin_id,
                    "instruction_template_id": item.instruction_template_id,
                },
            }
            for item in selected
        )

    schedules: dict[str, tuple[dict[str, object], ...]] = {}
    for split in DatasetSplit:
        selected = tuple(item for item in records if item.split is split)
        if config.variant is ActVariant.PER_TASK:
            selected = tuple(item for item in selected if item.task_id == config.task_id)
        schedules[split.value] = payload_for_records(selected)
    fresh = build_fresh_seed_schedule(
        m3b_export_fingerprint=completed.export_fingerprint,
        accepted_source_seeds=tuple(sorted({item.source_scene_seed for item in records})),
        rejected_source_seeds=None,
        namespace=config.evaluation.fresh_seed_namespace,
    )
    selected_tasks = (
        (first_group[config.task_id],)
        if config.variant is ActVariant.PER_TASK
        else tuple(first_group[task_id] for task_id in CANONICAL_TASK_IDS)
    )
    schedules["fresh_seed"] = tuple(
        {
            "scene_seed": seed,
            "task_spec": {
                "target_object_id": task.target_object_id,
                "target_bin_id": task.target_bin_id,
                "instruction_template_id": task.instruction_template_id,
            },
        }
        for seed in fresh.ordered_fresh_seeds
        for task in selected_tasks
    )
    digests = {
        split: rollout_schedule_digest(
            m3b_export_fingerprint=completed.export_fingerprint,
            variant=config.variant.value,
            task_id=config.task_id,
            split=split,
            episodes=episodes,
        )
        for split, episodes in schedules.items()
        if episodes
    }
    return {
        "schema_version": "langmani-m4-evaluation-schedules-v1",
        "digests": digests,
        "episode_counts": {key: len(value) for key, value in schedules.items()},
        "fresh_seed_schedule": fresh.to_dict(),
    }


def _manifest(
    identity: ActRunIdentity,
    config: ActExperimentConfig,
    *,
    git_dirty: bool,
    statistics: TrainOnlyStatistics,
    state: TrainingState,
    checkpoints: tuple[CheckpointRecord, ...],
) -> ActExperimentManifest:
    return ActExperimentManifest(
        identity=identity,
        config=config,
        git_dirty=git_dirty,
        dirty_development_override=config.allow_dirty_development,
        train_statistics_fingerprint=statistics.statistics_fingerprint,
        training_state=state,
        checkpoints=checkpoints,
    )


def execute(args: argparse.Namespace) -> dict[str, object]:
    config, _ = _configs(args)
    git = inspect_git_state(PROJECT_ROOT)
    validate_git_for_run(
        git,
        mode=config.mode,
        allow_dirty_development=config.allow_dirty_development,
    )
    completed, train_view, validation_view, statistics = _data_and_statistics(config)
    identity = _identity(
        config=config,
        completed=completed,
        train_view=train_view,
        validation_view=validation_view,
        statistics=statistics,
        git_commit=git.commit,
        git_dirty=git.dirty,
    )
    run_root = safe_run_directory(args.output_root, identity.run_fingerprint)
    act_config = build_act_config(
        config.model,
        device=config.device,
        use_amp=config.optimization.mixed_precision != "none",
        optimization=config.optimization,
    )
    dry_report = {
        "schema_version": "langmani-m4-train-command-v1",
        "dataset_fingerprint": completed.export_fingerprint,
        "split_digest": completed.split_manifest_digest,
        "variant": config.variant.value,
        "task_id": config.task_id,
        "input_shapes": {
            config.model.image_feature_key: list(config.model.image_shape_chw),
            config.model.state_feature_key: [config.model.state_dimension],
        },
        "output_shape": [config.model.action_dimension],
        "train_episode_count": len(train_view.episode_indices),
        "validation_episode_count": len(validation_view.episode_indices),
        "train_statistics_fingerprint": statistics.statistics_fingerprint,
        "act_config": {
            **config.model.to_dict(),
            "device": str(act_config.device),
            "use_amp": bool(act_config.use_amp),
            "optimizer_lr": float(act_config.optimizer_lr),
            "optimizer_weight_decay": float(act_config.optimizer_weight_decay),
            "optimizer_lr_backbone": float(act_config.optimizer_lr_backbone),
            "input_features": {
                key: {"type": value.type.value, "shape": list(value.shape)}
                for key, value in act_config.input_features.items()
            },
            "output_features": {
                key: {"type": value.type.value, "shape": list(value.shape)}
                for key, value in act_config.output_features.items()
            },
        },
        "optimization": config.optimization.to_dict(),
        "delta_timestamps": {
            "observation": act_config.observation_delta_indices,
            "action": [index / config.data.fps for index in act_config.action_delta_indices],
        },
        "run_fingerprint": identity.run_fingerprint,
        "expected_output_directory": str(run_root),
        "checkpoint_schedule": list(
            range(
                config.optimization.checkpoint_interval,
                config.optimization.training_steps + 1,
                config.optimization.checkpoint_interval,
            )
        ),
        "evaluation_schedule": list(
            range(
                config.optimization.validation_interval,
                config.optimization.training_steps + 1,
                config.optimization.validation_interval,
            )
        ),
        "git": git.to_dict(),
        "fixture_evidence": config.mode is ExperimentMode.DRY_RUN,
    }
    if config.mode is ExperimentMode.DRY_RUN:
        return {**dry_report, "passed": True, "training_started": False}
    if config.mode is ExperimentMode.FULL and config.device != "cuda":
        raise RuntimeError("full M4 training requires explicit CUDA execution")
    resuming = args.resume_checkpoint is not None
    if run_root.exists() and not resuming:
        raise FileExistsError(f"run fingerprint already exists and is immutable: {run_root}")
    if resuming and not run_root.is_dir():
        raise FileNotFoundError("resume requires the existing fingerprint-owned run directory")
    run_root.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    manifest_path = run_root / "run_manifest.json"
    existing_manifest: ActExperimentManifest | None = None
    recover_orphan = False
    if resuming:
        if (run_root / "reports" / "training_summary.json").exists():
            raise RuntimeError("a successfully completed training loop cannot be resumed")
        existing_manifest = ActExperimentManifest.from_dict(_read_object(manifest_path))
        if existing_manifest.identity != identity or existing_manifest.config != config:
            raise RuntimeError("resume run manifest differs from the requested semantic identity")
        recover_orphan = _resume_checkpoint_is_recoverable_orphan(
            run_root,
            requested_relative_path=args.resume_checkpoint,
            declared_relative_paths=tuple(
                checkpoint.relative_path for checkpoint in existing_manifest.checkpoints
            ),
        )
    if config.mode is ExperimentMode.FULL:
        audit_path = run_root / "reports" / "offline_counterfactual_audit.json"
        if resuming:
            audit_payload = _read_object(audit_path)
            if audit_payload.get("passed") is not True:
                raise RuntimeError("existing full-run counterfactual audit is not passing")
        else:
            ambiguity_audit = audit_m3b_train_counterfactuals(
                completed,
                policy_config=act_config,
            )
            atomic_write_json(
                audit_path,
                ambiguity_audit.to_dict(),
                immutable=True,
            )
            if not ambiguity_audit.passed:
                raise RuntimeError("M3B train counterfactual ambiguity audit failed")
    temporal_train = load_lerobot_episode_view(
        completed,
        train_view,
        policy_config=act_config,
        include_delta_timestamps=True,
        return_uint8=True,
    )
    temporal_validation = load_lerobot_episode_view(
        completed,
        validation_view,
        policy_config=act_config,
        include_delta_timestamps=True,
        return_uint8=True,
    )
    validate_temporal_episode_boundaries(
        temporal_train,
        chunk_size=config.model.chunk_size,
    )
    validate_temporal_episode_boundaries(
        temporal_validation,
        chunk_size=config.model.chunk_size,
    )
    policy, preprocessor, postprocessor = build_policy_and_processors(
        act_config, statistics.to_processor_stats()
    )
    optimizer = make_optimizer(policy, config)
    start_state = TrainingState(global_step=0, examples_processed=0)
    if resuming:
        loaded = load_act_checkpoint(
            run_root=run_root,
            checkpoint_relative_path=args.resume_checkpoint,
            expected_identity=identity,
            optimizer=optimizer,
            for_resume=True,
            existing_policy=policy,
        )
        policy = loaded.policy
        preprocessor = loaded.preprocessor
        postprocessor = loaded.postprocessor
        optimizer = loaded.optimizer
        if optimizer is None:
            raise RuntimeError("resume did not restore the required optimizer")
        start_state = loaded.training_state
        if recover_orphan:
            assert existing_manifest is not None
            _validate_recoverable_orphan_step(
                loaded.record.global_step,
                declared_steps=tuple(
                    checkpoint.global_step for checkpoint in existing_manifest.checkpoints
                ),
                checkpoint_interval=config.optimization.checkpoint_interval,
                training_steps=config.optimization.training_steps,
            )
        recovered_metric = (
            {**loaded.training_metric, "checkpoint_path": loaded.record.relative_path}
            if loaded.training_metric is not None
            else None
        )
        _truncate_metrics_to_checkpoint(
            run_root / "metrics.jsonl",
            start_state.global_step,
            recovery_record=recovered_metric,
        )
    checkpoints: list[CheckpointRecord] = list(
        existing_manifest.checkpoints if existing_manifest is not None else ()
    )
    if resuming and recover_orphan:
        checkpoints.append(loaded.record)
        recovered_state = replace(
            start_state,
            last_checkpoint_fingerprint=loaded.record.checkpoint_fingerprint,
        )
        atomic_write_json(
            manifest_path,
            _manifest(
                identity,
                config,
                git_dirty=git.dirty,
                statistics=statistics,
                state=recovered_state,
                checkpoints=tuple(checkpoints),
            ).to_dict(),
        )
    _write_or_validate_json(run_root / "config.json", config.to_dict())
    _write_or_validate_json(
        run_root / "dataset_contract.json",
        identity.to_dict()["data_contract"],
    )
    _write_or_validate_json(run_root / "train_stats.json", statistics.to_dict())
    initial_validation_path = run_root / "reports" / "initial_validation.json"
    initial_validation_loss: float | None = None
    if resuming:
        initial_validation_payload = _read_object(initial_validation_path)
        if initial_validation_payload.get(
            "run_fingerprint"
        ) != identity.run_fingerprint or not isinstance(
            initial_validation_payload.get("loss"), int | float
        ):
            raise RuntimeError("initial validation evidence is missing or mismatched")
        initial_validation_loss = float(initial_validation_payload["loss"])
    if resuming and start_state.global_step == config.optimization.training_steps:
        final_state = replace(
            start_state,
            last_checkpoint_fingerprint=checkpoints[-1].checkpoint_fingerprint,
        )
        atomic_write_json(
            manifest_path,
            _manifest(
                identity,
                config,
                git_dirty=git.dirty,
                statistics=statistics,
                state=final_state,
                checkpoints=tuple(checkpoints),
            ).to_dict(),
        )
        training_summary_path = run_root / "reports" / "training_summary.json"
        atomic_write_json(
            training_summary_path,
            _training_summary_from_metrics(
                metrics_path=run_root / "metrics.jsonl",
                identity=identity,
                config=config,
                policy=policy,
                training_wall_duration_s=None,
                initial_validation_loss=initial_validation_loss,
            ),
            immutable=True,
        )
        return {
            **dry_report,
            "passed": True,
            "training_started": False,
            "resumed_finalization_only": True,
            "final_step": final_state.global_step,
            "examples_processed": final_state.examples_processed,
            "checkpoint_count": len(checkpoints),
            "last_checkpoint": checkpoints[-1].to_dict(),
            "run_manifest": str(manifest_path),
            "training_summary": str(training_summary_path),
        }
    if start_state.global_step > config.optimization.training_steps:
        raise RuntimeError("resume checkpoint step exceeds the requested training horizon")
    training_wall_started = time.perf_counter()
    train_loader = _loader(
        temporal_train.dataset,
        config,
        shuffle=True,
        start_step=start_state.global_step,
    )
    validation_loader = _loader(temporal_validation.dataset, config, shuffle=False)

    def save_checkpoint_callback(**values: object) -> str:
        step = int(values["step"])
        checkpoint_state = TrainingState(
            global_step=step,
            examples_processed=int(values["examples_processed"]),
        )
        record = save_act_checkpoint(
            run_root=run_root,
            identity=identity,
            training_state=checkpoint_state,
            policy=values["policy"],
            preprocessor=values["preprocessor"],
            postprocessor=values["postprocessor"],
            optimizer=values["optimizer"],
            training_metric=values["metric"],
        )
        checkpoints.append(record)
        manifest_state = replace(
            checkpoint_state,
            last_checkpoint_fingerprint=record.checkpoint_fingerprint,
        )
        atomic_write_json(
            manifest_path,
            _manifest(
                identity,
                config,
                git_dirty=git.dirty,
                statistics=statistics,
                state=manifest_state,
                checkpoints=tuple(checkpoints),
            ).to_dict(),
        )
        return record.relative_path

    def validation_callback(_step: int, current_policy: object) -> float:
        cuda_devices = list(range(torch.cuda.device_count())) if config.device == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            validation_seed = config.seed + 1_000_003 + _step
            torch.manual_seed(validation_seed)
            if config.device == "cuda":
                torch.cuda.manual_seed_all(validation_seed)
            return evaluate_offline_loss(
                experiment=config,
                policy=current_policy,
                preprocessor=preprocessor,
                validation_batches=validation_loader,
                task_id_by_episode=(
                    completed.views.task_id_by_episode
                    if config.variant is ActVariant.MIXED_TASK_ONEHOT
                    else None
                ),
            )

    if initial_validation_loss is None:
        initial_validation_loss = validation_callback(0, policy)
        atomic_write_json(
            initial_validation_path,
            {
                "schema_version": "langmani-m4-initial-validation-v1",
                "run_fingerprint": identity.run_fingerprint,
                "loss": initial_validation_loss,
            },
            immutable=True,
        )

    outcome = train_act(
        experiment=config,
        policy=policy,
        preprocessor=preprocessor,
        optimizer=optimizer,
        train_batches=train_loader,
        start_step=start_state.global_step,
        maximum_steps=config.optimization.training_steps,
        metrics_path=run_root / "metrics.jsonl",
        checkpoint_callback=save_checkpoint_callback,
        postprocessor=postprocessor,
        validation_callback=validation_callback,
        task_id_by_episode=(
            completed.views.task_id_by_episode
            if config.variant is ActVariant.MIXED_TASK_ONEHOT
            else None
        ),
        seed_at_start=not resuming,
        initial_examples_processed=start_state.examples_processed,
    )
    if not checkpoints or checkpoints[-1].global_step != outcome.final_step:
        final_checkpoint_path = save_checkpoint_callback(
            step=outcome.final_step,
            examples_processed=outcome.examples_processed,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            optimizer=optimizer,
            metric=outcome.metrics[-1].to_dict(),
        )
        _bind_metric_checkpoint_path(
            run_root / "metrics.jsonl",
            step=outcome.final_step,
            relative_path=final_checkpoint_path,
        )
    final_state = TrainingState(
        global_step=outcome.final_step,
        examples_processed=outcome.examples_processed,
        last_checkpoint_fingerprint=checkpoints[-1].checkpoint_fingerprint,
    )
    atomic_write_json(
        manifest_path,
        _manifest(
            identity,
            config,
            git_dirty=git.dirty,
            statistics=statistics,
            state=final_state,
            checkpoints=tuple(checkpoints),
        ).to_dict(),
    )
    training_summary_path = run_root / "reports" / "training_summary.json"
    training_wall_duration = time.perf_counter() - training_wall_started
    atomic_write_json(
        training_summary_path,
        _training_summary_from_metrics(
            metrics_path=run_root / "metrics.jsonl",
            identity=identity,
            config=config,
            policy=policy,
            training_wall_duration_s=(training_wall_duration if not resuming else None),
            initial_validation_loss=initial_validation_loss,
        ),
        immutable=True,
    )
    return {
        **dry_report,
        "passed": True,
        "training_started": True,
        "final_step": outcome.final_step,
        "examples_processed": final_state.examples_processed,
        "checkpoint_count": len(checkpoints),
        "last_checkpoint": checkpoints[-1].to_dict(),
        "run_manifest": str(manifest_path),
        "training_summary": str(training_summary_path),
    }


def main() -> int:
    args = parse_args()
    try:
        report = execute(args)
    except Exception as error:  # noqa: BLE001 - command boundary preserves diagnostics
        traceback.print_exc()
        report = {
            "schema_version": "langmani-m4-train-command-v1",
            "passed": False,
            "error_type": type(error).__name__,
            "error_message": str(error) or repr(error),
        }
    atomic_write_json(args.report, report)
    print(json.dumps({**report, "report": str(args.report.resolve())}, sort_keys=True))
    return 0 if report["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
