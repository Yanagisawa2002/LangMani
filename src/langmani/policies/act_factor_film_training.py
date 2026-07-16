"""M4.3b training, checkpoint, fixture, and selection integration.

The real target path reuses the completed-M3B gate, train-only statistics,
standard ACT processors, deterministic sampler/training loop, and atomic M4
checkpoint lifecycle.  This module adds only the factorized conditioning
inputs and independent M4.3b identities.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from lerobot.policies.act import ACTConfig
from torch.utils.data import DataLoader

from langmani.datasets.lerobot_types import DatasetSplit
from langmani.policies.act_checkpoint import load_act_checkpoint, save_act_checkpoint
from langmani.policies.act_conditioning import CANONICAL_TASK_IDS
from langmani.policies.act_data import (
    CompletedM3BDataset,
    DatasetEpisodeView,
    TrainOnlyStatistics,
    compute_train_only_statistics,
    load_completed_m3b_dataset,
    load_lerobot_episode_view,
    validate_temporal_episode_boundaries,
)
from langmani.policies.act_factor_film_adapter import (
    FactorFiLMACTPolicy,
    build_factor_film_architecture_identity,
    validate_factor_film_policy_structure,
)
from langmani.policies.act_factor_film_conditioning import (
    attach_factor_film_runtime_input,
    runtime_input_from_episode_indices,
)
from langmani.policies.act_factor_film_types import (
    FactorFiLMArchitectureIdentity,
    FactorFiLMCheckpointContract,
    FactorFiLMConfig,
    FactorFiLMContractError,
    FactorFiLMDryRunReport,
    FactorFiLMRunIdentity,
    FactorFiLMSelectionRecord,
    FactorFiLMTrainingConfig,
    FactorFiLMTrainingManifest,
    FactorFiLMTrainingMode,
    FactorFiLMValidationQueue,
    FactorFiLMValidationQueueItem,
    FactorFiLMValidationResult,
    canonical_fingerprint,
)
from langmani.policies.act_runtime import GitState, runtime_versions
from langmani.policies.act_training import (
    DeterministicResumeBatchSampler,
    build_act_config,
    build_policy_and_processors,
    make_optimizer,
    prepare_raw_batch,
    seed_dataloader_worker,
    seed_everything,
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
from langmani.policies.m42_types import GripperRuntimeMode
from langmani.policies.m43_evidence import (
    CompletedM43AuditEvidence,
    M43AuditScope,
    validate_completed_audit_evidence,
)

AUTHORIZED_M43_AUDIT_EVIDENCE_FINGERPRINT = (
    "sha256:6342bdf4b019df203e6021947cbb39deacac2ea78cc585b5062091ed1d228671"
)
AUTHORIZED_M3B_FINGERPRINT = (
    "sha256:3f4d81471ac7c3ecc034206bc207524c4cbfbd1f7874b25eca894a5b607acfb4"
)
AUTHORIZED_M3B_SPLIT_FINGERPRINT = (
    "sha256:d86d29374ef7956a4ad8a1d9a111ed0924387b5b456d86e8c9ad5a31752f65e5"
)
FACTOR_FILM_RELOAD_ATOL = 1e-6
FACTOR_FILM_RELOAD_RTOL = 1e-6
FACTOR_FILM_IDENTITY_INITIALIZATION_ATOL = 1e-3


@dataclass(frozen=True, slots=True)
class FactorFiLMDataBundle:
    completed: CompletedM3BDataset
    train_view: DatasetEpisodeView
    validation_view: DatasetEpisodeView
    statistics: TrainOnlyStatistics
    validation_schedule_digest: str


class FactorFiLMProcessedBatchAugmenter:
    """The single shared post-processor conditioning boundary."""

    def __init__(self, task_id_by_episode: Mapping[int, str]) -> None:
        self._task_id_by_episode = dict(task_id_by_episode)

    def __call__(
        self,
        processed_batch: Mapping[str, object],
        raw_batch: Mapping[str, object],
    ) -> Mapping[str, object]:
        episode_indices = raw_batch.get("episode_index")
        if not isinstance(episode_indices, torch.Tensor):
            raise FactorFiLMContractError(
                "FactorFiLM training batches require numeric episode_index"
            )
        runtime_input = runtime_input_from_episode_indices(
            episode_indices,
            task_id_by_episode=self._task_id_by_episode,
        )
        return attach_factor_film_runtime_input(processed_batch, runtime_input)


def validate_authorizing_semantic_audit(
    evidence_root: str | Path,
) -> CompletedM43AuditEvidence:
    """Validate the one real audit that authorized M4.3b implementation."""
    completed = validate_completed_audit_evidence(Path(evidence_root).resolve())
    if completed.evidence_fingerprint != AUTHORIZED_M43_AUDIT_EVIDENCE_FINGERPRINT:
        raise FactorFiLMContractError("semantic-audit evidence fingerprint is not authorized")
    if completed.config.scope is not M43AuditScope.COMBINED:
        raise FactorFiLMContractError(
            "FactorFiLM requires the combined validation/development audit"
        )
    if completed.config.m3b_fingerprint != AUTHORIZED_M3B_FINGERPRINT:
        raise FactorFiLMContractError("semantic audit is bound to another M3B dataset")
    if completed.config.m3b_split_manifest_digest != AUTHORIZED_M3B_SPLIT_FINGERPRINT:
        raise FactorFiLMContractError("semantic audit is bound to another M3B split")
    if (
        completed.config.selected_execution_horizon != 10
        or completed.config.selected_gripper_runtime is not GripperRuntimeMode.PROJECT
    ):
        raise FactorFiLMContractError("semantic audit does not use locked H=10/project runtime")
    completion = completed.completion
    if completion.get("semantic_audit_completed") is not True:
        raise FactorFiLMContractError("semantic audit is not complete")
    if any(
        completion.get(name) is not False
        for name in (
            "test_split_accessed",
            "fresh_seed_accessed",
            "final_schedule_accessed",
            "final_benchmark_authorized",
            "smolvla_go",
        )
    ):
        raise FactorFiLMContractError("semantic audit accessed a prohibited source or decision")
    return completed


def factor_film_validation_schedule_records(
    completed: CompletedM3BDataset,
    validation_view: DatasetEpisodeView | None = None,
) -> tuple[dict[str, object], ...]:
    """Return the exact canonical M3B-validation episode identity records."""

    view = validation_view or completed.views.for_split(DatasetSplit.VALIDATION)
    records = {value.lerobot_episode_index: value for value in completed.manifest.episodes}
    episodes: list[dict[str, object]] = []
    for index in view.episode_indices:
        value = records[index]
        episodes.append(
            {
                "episode_index": index,
                "scene_group_id": value.source_scene_group_id,
                "scene_seed": value.source_scene_seed,
                "task_id": value.task_id,
            }
        )
    if len(episodes) != 36 or len({value["episode_index"] for value in episodes}) != 36:
        raise FactorFiLMContractError(
            "FactorFiLM validation schedule requires exactly 36 unique episodes"
        )
    return tuple(episodes)


def factor_film_validation_schedule_digest(
    completed: CompletedM3BDataset,
    validation_view: DatasetEpisodeView | None = None,
) -> str:
    """Fingerprint the exact validation schedule without loading policy frames."""

    episodes = factor_film_validation_schedule_records(completed, validation_view)
    return canonical_fingerprint(
        {
            "schema_version": "langmani-m43-factor-film-validation-schedule-v0",
            "m3b_export_fingerprint": completed.export_fingerprint,
            "split_manifest_digest": completed.split_manifest_digest,
            "source": "m3b_validation",
            "episodes": episodes,
        }
    )


def load_factor_film_data(dataset_root: str | Path) -> FactorFiLMDataBundle:
    """Load exact 288/36 mixed views and recompute 9D train-only statistics."""
    completed = load_completed_m3b_dataset(
        dataset_root,
        require_full=True,
        validate_storage=True,
    )
    if completed.export_fingerprint != AUTHORIZED_M3B_FINGERPRINT:
        raise FactorFiLMContractError("FactorFiLM dataset fingerprint differs from M4.3a")
    if completed.split_manifest_digest != AUTHORIZED_M3B_SPLIT_FINGERPRINT:
        raise FactorFiLMContractError("FactorFiLM split fingerprint differs from M4.3a")
    train_view = completed.views.for_split(DatasetSplit.TRAIN)
    validation_view = completed.views.for_split(DatasetSplit.VALIDATION)
    if (len(train_view.episode_indices), len(validation_view.episode_indices)) != (288, 36):
        raise FactorFiLMContractError("FactorFiLM requires exact 288/36 mixed episode views")
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
        validation_episode_indices=completed.views.validation.episode_indices,
        test_episode_indices=completed.views.test.episode_indices,
        task_id_by_episode=completed.views.task_id_by_episode,
        variant=ActVariant.MIXED_UNCONDITIONED,
        task_id=None,
    )
    if statistics.state.mean and len(statistics.state.mean) != 9:
        raise FactorFiLMContractError("FactorFiLM train-only state statistics are not 9D")
    return FactorFiLMDataBundle(
        completed=completed,
        train_view=train_view,
        validation_view=validation_view,
        statistics=statistics,
        validation_schedule_digest=factor_film_validation_schedule_digest(
            completed, validation_view
        ),
    )


def internal_act_experiment(config: FactorFiLMTrainingConfig) -> ActExperimentConfig:
    """Reuse the existing ACT loop without adding a fourth historical ActVariant."""
    mode = (
        ExperimentMode.DEVELOPMENT
        if config.mode is FactorFiLMTrainingMode.TARGET_DEVELOPMENT
        else ExperimentMode.DRY_RUN
    )
    return ActExperimentConfig(
        variant=ActVariant.MIXED_UNCONDITIONED,
        data=config.data,
        model=config.base_model,
        optimization=config.optimization,
        seed=config.training_seed,
        mode=mode,
        device=config.device,
    )


def build_factor_film_policy_and_processors(
    *,
    training_config: FactorFiLMTrainingConfig,
    statistics: TrainOnlyStatistics | Mapping[str, Mapping[str, torch.Tensor]],
    factor_film_config: FactorFiLMConfig | None = None,
) -> tuple[
    ACTConfig,
    FactorFiLMACTPolicy,
    Any,
    Any,
    FactorFiLMArchitectureIdentity,
]:
    """Construct one policy and the unchanged 9D ACT processors."""
    act_config = build_act_config(
        training_config.base_model,
        device=training_config.device,
        use_amp=(
            training_config.device == "cuda"
            and training_config.optimization.mixed_precision != "none"
        ),
        optimization=training_config.optimization,
    )
    effective_film_config = factor_film_config or FactorFiLMConfig(
        visual_feature_channels=512,
        state_context_dimension=training_config.base_model.dim_model,
    )
    processor_stats = (
        statistics.to_processor_stats()
        if isinstance(statistics, TrainOnlyStatistics)
        else statistics
    )
    policy, preprocessor, postprocessor = build_policy_and_processors(
        act_config,
        processor_stats,
        policy_factory=lambda value: FactorFiLMACTPolicy(
            value,
            factor_film_config=effective_film_config,
        ),
    )
    if not isinstance(policy, FactorFiLMACTPolicy):
        raise RuntimeError("FactorFiLM policy factory returned the wrong type")
    architecture = build_factor_film_architecture_identity(
        policy,
        base_act_configuration=training_config.base_model.to_dict(),
    )
    return act_config, policy, preprocessor, postprocessor, architecture


def build_factor_film_run_identity(
    *,
    training_config: FactorFiLMTrainingConfig,
    data: FactorFiLMDataBundle,
    architecture: FactorFiLMArchitectureIdentity,
    git: GitState,
) -> FactorFiLMRunIdentity:
    versions = runtime_versions()
    return FactorFiLMRunIdentity(
        m3b_export_fingerprint=data.completed.export_fingerprint,
        m3b_split_manifest_digest=data.completed.split_manifest_digest,
        ordered_train_episode_indices=data.train_view.episode_indices,
        ordered_validation_episode_indices=data.validation_view.episode_indices,
        architecture_identity=architecture.to_dict(),
        model_config={
            "base_act": training_config.base_model.to_dict(),
            "factor_film": architecture.factor_film_config.to_dict(),
        },
        data_contract={
            "policy_features": {
                "observation.images.base_camera": [3, 256, 256],
                "observation.state": [9],
                "action": [8],
            },
            "condition_source": "stable_m3b_task_metadata",
            "object_mapping": architecture.factor_film_config.object_mapping.to_dict(),
            "bin_mapping": architecture.factor_film_config.bin_mapping.to_dict(),
            "checkpoint_selection_source": "m3b_validation",
            "validation_schedule_digest": data.validation_schedule_digest,
            "train_episode_count": len(data.train_view.episode_indices),
            "validation_episode_count": len(data.validation_view.episode_indices),
            "train_episode_indices_fingerprint": canonical_fingerprint(
                data.train_view.episode_indices
            ),
            "validation_episode_indices_fingerprint": canonical_fingerprint(
                data.validation_view.episode_indices
            ),
            "execution_horizon": training_config.execution_horizon,
            "gripper_runtime_mode": training_config.gripper_runtime_mode,
            "test_accessible": False,
            "historical_unseen_schedule_accessible": False,
            "sealed_schedule_accessed": False,
        },
        train_statistics_fingerprint=data.statistics.statistics_fingerprint,
        optimization_config=training_config.optimization.to_dict(),
        semantic_audit_evidence_fingerprint=(training_config.semantic_audit_evidence_fingerprint),
        training_seed=training_config.training_seed,
        experiment_mode=training_config.mode,
        device=training_config.device,
        dtype=training_config.dtype,
        lerobot_version=str(versions["lerobot"]),
        torch_version=str(versions["torch"]),
        cuda_version=cast(str | None, versions["cuda"]),
        git_commit=git.commit,
        git_dirty=git.dirty,
    )


def build_factor_film_checkpoint_contract(
    *,
    identity: FactorFiLMRunIdentity,
    architecture: FactorFiLMArchitectureIdentity,
) -> FactorFiLMCheckpointContract:
    config = architecture.factor_film_config
    return FactorFiLMCheckpointContract(
        architecture_fingerprint=architecture.architecture_fingerprint,
        base_act_configuration_fingerprint=(architecture.base_act_configuration_fingerprint),
        object_mapping_fingerprint=config.object_mapping.fingerprint,
        bin_mapping_fingerprint=config.bin_mapping.fingerprint,
        train_statistics_fingerprint=identity.train_statistics_fingerprint,
        m3b_export_fingerprint=identity.m3b_export_fingerprint,
        m3b_split_manifest_digest=identity.m3b_split_manifest_digest,
        execution_horizon=10,
        gripper_runtime_mode="project",
    )


def build_factor_film_training_manifest(
    *,
    identity: FactorFiLMRunIdentity,
    training_config: FactorFiLMTrainingConfig,
    architecture: FactorFiLMArchitectureIdentity,
    checkpoint_contract: FactorFiLMCheckpointContract,
    training_state: TrainingState,
    checkpoints: tuple[CheckpointRecord, ...],
    training_complete: bool,
) -> FactorFiLMTrainingManifest:
    """Build one immutable, independently parsed FactorFiLM run manifest."""
    return FactorFiLMTrainingManifest(
        identity=identity,
        training_config=training_config,
        architecture_identity=architecture,
        checkpoint_contract=checkpoint_contract,
        training_state=training_state,
        checkpoints=checkpoints,
        training_complete=training_complete,
    )


def build_factor_film_validation_queue(
    *,
    identity: FactorFiLMRunIdentity,
    validation_schedule_digest: str,
    checkpoints: tuple[CheckpointRecord, ...],
    offline_validation_action_losses: Mapping[str, float],
) -> FactorFiLMValidationQueue:
    """Publish the exact 20 checkpoints and their fingerprint-bound losses."""
    expected_fingerprints = {value.checkpoint_fingerprint for value in checkpoints}
    if set(offline_validation_action_losses) != expected_fingerprints:
        raise FactorFiLMContractError(
            "FactorFiLM validation losses must cover exactly the published checkpoints"
        )
    items = tuple(
        FactorFiLMValidationQueueItem(
            checkpoint_fingerprint=value.checkpoint_fingerprint,
            checkpoint_step=value.global_step,
            checkpoint_relative_path=value.relative_path,
            offline_validation_action_loss=float(
                offline_validation_action_losses[value.checkpoint_fingerprint]
            ),
        )
        for value in checkpoints
    )
    expected_steps = tuple(range(5_000, 100_001, 5_000))
    if tuple(value.checkpoint_step for value in items) != expected_steps:
        raise FactorFiLMContractError(
            "FactorFiLM validation queue requires the exact 20-checkpoint schedule"
        )
    return FactorFiLMValidationQueue(
        run_fingerprint=identity.run_fingerprint,
        validation_schedule_digest=validation_schedule_digest,
        items=items,
    )


def build_factor_film_dry_run_report(
    *,
    training_config: FactorFiLMTrainingConfig,
    data: FactorFiLMDataBundle,
    architecture: FactorFiLMArchitectureIdentity,
    identity: FactorFiLMRunIdentity,
    output_root: str | Path,
) -> FactorFiLMDryRunReport:
    """Describe a real-data run identity without starting optimization."""
    model = training_config.base_model
    expected_steps = tuple(
        range(
            training_config.optimization.checkpoint_interval,
            training_config.optimization.training_steps + 1,
            training_config.optimization.checkpoint_interval,
        )
    )
    return FactorFiLMDryRunReport(
        dataset_fingerprint=data.completed.export_fingerprint,
        split_fingerprint=data.completed.split_manifest_digest,
        train_episode_count=len(data.train_view.episode_indices),
        validation_episode_count=len(data.validation_view.episode_indices),
        feature_shapes={
            model.image_feature_key: list(model.image_shape_chw),
            model.state_feature_key: [model.state_dimension],
            model.action_feature_key: [model.chunk_size, model.action_dimension],
        },
        panda_policy_state_dimension=model.state_dimension,
        object_mapping=architecture.factor_film_config.object_mapping.to_dict(),
        bin_mapping=architecture.factor_film_config.bin_mapping.to_dict(),
        visual_film_injection=architecture.factor_film_config.visual_injection_target,
        state_film_injection=architecture.factor_film_config.state_injection_target,
        base_act_parameter_count=architecture.base_parameter_count,
        factor_film_parameter_count=architecture.factor_film_parameter_count,
        parameter_count_increase=architecture.parameter_count_increase,
        train_statistics_fingerprint=data.statistics.statistics_fingerprint,
        effective_training_configuration=training_config.to_dict(),
        run_fingerprint=identity.run_fingerprint,
        expected_output_path=str(
            Path(output_root).resolve() / identity.run_fingerprint.removeprefix("sha256:")
        ),
        expected_checkpoint_steps=expected_steps,
        semantic_audit_evidence_fingerprint=(training_config.semantic_audit_evidence_fingerprint),
        fixture_contract=False,
    )


def factor_film_dataloader(
    dataset: Any,
    config: FactorFiLMTrainingConfig,
    *,
    shuffle: bool,
    start_step: int = 0,
) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(config.training_seed)
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
                seed=config.training_seed,
                start_step=start_step,
            ),
        )
    return DataLoader(
        **common,
        batch_size=config.optimization.batch_size,
        shuffle=False,
        drop_last=False,
    )


def load_factor_film_temporal_views(
    data: FactorFiLMDataBundle,
    act_config: ACTConfig,
    training_config: FactorFiLMTrainingConfig,
) -> tuple[Any, Any]:
    train = load_lerobot_episode_view(
        data.completed,
        data.train_view,
        policy_config=act_config,
        include_delta_timestamps=True,
        return_uint8=True,
    )
    validation = load_lerobot_episode_view(
        data.completed,
        data.validation_view,
        policy_config=act_config,
        include_delta_timestamps=True,
        return_uint8=True,
    )
    validate_temporal_episode_boundaries(train, chunk_size=training_config.base_model.chunk_size)
    validate_temporal_episode_boundaries(
        validation,
        chunk_size=training_config.base_model.chunk_size,
    )
    return train.dataset, validation.dataset


def rank_factor_film_validation_results(
    candidates: tuple[FactorFiLMValidationResult, ...],
) -> tuple[FactorFiLMValidationResult, ...]:
    if len(candidates) != 20:
        raise FactorFiLMContractError("FactorFiLM selection requires all 20 checkpoints")
    if len({value.checkpoint_fingerprint for value in candidates}) != 20:
        raise FactorFiLMContractError("FactorFiLM validation checkpoints must be unique")
    if len({value.checkpoint_step for value in candidates}) != 20:
        raise FactorFiLMContractError("FactorFiLM validation steps must be unique")
    if len({value.schedule_digest for value in candidates}) != 1:
        raise FactorFiLMContractError("FactorFiLM candidates use different validation schedules")
    expected_steps = tuple(range(5_000, 100_001, 5_000))
    if tuple(sorted(value.checkpoint_step for value in candidates)) != expected_steps:
        raise FactorFiLMContractError("FactorFiLM candidates do not cover the 20-step schedule")
    return tuple(sorted(candidates, key=lambda value: value.ranking_key))


def create_factor_film_selection(
    *,
    validation_queue: FactorFiLMValidationQueue,
    candidates: tuple[FactorFiLMValidationResult, ...],
) -> FactorFiLMSelectionRecord:
    ranked = rank_factor_film_validation_results(candidates)
    if validation_queue.validation_schedule_digest != ranked[0].schedule_digest:
        raise FactorFiLMContractError("selection results differ from the published schedule")
    queued = {
        (value.checkpoint_fingerprint, value.checkpoint_step) for value in validation_queue.items
    }
    observed = {(value.checkpoint_fingerprint, value.checkpoint_step) for value in candidates}
    if observed != queued:
        raise FactorFiLMContractError(
            "selection results do not match the published validation queue"
        )
    return FactorFiLMSelectionRecord(
        run_fingerprint=validation_queue.run_fingerprint,
        validation_queue_fingerprint=validation_queue.queue_fingerprint,
        validation_queue=validation_queue,
        validation_schedule_digest=ranked[0].schedule_digest,
        candidates=candidates,
        ranked_checkpoint_fingerprints=tuple(value.checkpoint_fingerprint for value in ranked),
        selected_checkpoint_fingerprint=ranked[0].checkpoint_fingerprint,
        selected_checkpoint_step=ranked[0].checkpoint_step,
    )


def _fixture_model() -> ActModelConfig:
    return ActModelConfig(
        state_dimension=9,
        chunk_size=4,
        n_action_steps=2,
        dim_model=32,
        n_heads=2,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        latent_dim=8,
        n_vae_encoder_layers=1,
        dropout=0.0,
    )


def _fixture_stats() -> dict[str, dict[str, torch.Tensor]]:
    return {
        "observation.images.base_camera": {
            "mean": torch.zeros(3),
            "std": torch.ones(3),
            "min": torch.zeros(3),
            "max": torch.ones(3),
        },
        "observation.state": {
            "mean": torch.zeros(9),
            "std": torch.ones(9),
            "min": -torch.ones(9),
            "max": torch.ones(9),
        },
        "action": {
            "mean": torch.zeros(8),
            "std": torch.ones(8),
            "min": -torch.ones(8),
            "max": torch.ones(8),
        },
    }


def _fixture_raw_batch() -> tuple[dict[str, torch.Tensor], Mapping[int, str]]:
    generator = torch.Generator().manual_seed(43)
    batch_size = 6
    return (
        {
            "observation.images.base_camera": torch.randint(
                0,
                256,
                (batch_size, 3, 256, 256),
                generator=generator,
                dtype=torch.uint8,
            ),
            "observation.state": torch.randn(batch_size, 9, generator=generator),
            "action": torch.randn(batch_size, 4, 8, generator=generator).clamp(-1, 1),
            "action_is_pad": torch.zeros(batch_size, 4, dtype=torch.bool),
            "episode_index": torch.arange(batch_size, dtype=torch.int64),
        },
        {index: task_id for index, task_id in enumerate(CANONICAL_TASK_IDS)},
    )


def run_factor_film_fixture(git_commit: str) -> dict[str, object]:
    """Run real CPU forward/backward/optimizer/checkpoint/processor reload."""
    seed_everything(43)
    model = _fixture_model()
    optimization = ActOptimizationConfig(
        mixed_precision="none",
        batch_size=6,
        dataloader_workers=0,
        training_steps=1,
        checkpoint_interval=1,
        validation_interval=1,
    )
    training_config = FactorFiLMTrainingConfig(
        data=ActDataConfig(dataset_root="fixture-only"),
        base_model=model,
        optimization=optimization,
        semantic_audit_evidence_fingerprint=(AUTHORIZED_M43_AUDIT_EVIDENCE_FINGERPRINT),
        training_seed=43,
        mode=FactorFiLMTrainingMode.FIXTURE,
        device="cpu",
    )
    act_config, policy, preprocessor, postprocessor, architecture = (
        build_factor_film_policy_and_processors(
            training_config=training_config,
            statistics=_fixture_stats(),
            factor_film_config=FactorFiLMConfig(state_context_dimension=model.dim_model),
        )
    )
    raw, task_map = _fixture_raw_batch()
    augmenter = FactorFiLMProcessedBatchAugmenter(task_map)
    processed = preprocessor(prepare_raw_batch(raw))
    if not isinstance(processed, Mapping):
        raise FactorFiLMContractError("fixture preprocessor did not return a mapping")
    conditioned = augmenter(processed, raw)

    visual = torch.randn(6, 512, 8, 8, generator=torch.Generator().manual_seed(1))
    state = torch.randn(6, model.dim_model, generator=torch.Generator().manual_seed(2))
    object_a = torch.zeros(6, dtype=torch.long)
    object_b = torch.full((6,), 2, dtype=torch.long)
    bin_a = torch.zeros(6, dtype=torch.long)
    bin_b = torch.ones(6, dtype=torch.long)
    visual_a = policy.target_object_condition(visual, object_a)
    visual_b = policy.target_object_condition(visual, object_b)
    state_a = policy.destination_bin_condition(state, bin_a)
    state_b = policy.destination_bin_condition(state, bin_b)
    identity_error = max(
        float((visual_a - visual).abs().max().detach()),
        float((state_a - state).abs().max().detach()),
    )
    if identity_error > FACTOR_FILM_IDENTITY_INITIALIZATION_ATOL:
        raise FactorFiLMContractError("FactorFiLM initialization is not identity-like")
    if torch.equal(visual_a, visual_b) or torch.equal(state_a, state_b):
        raise FactorFiLMContractError("factorized conditions do not change their own paths")

    policy.train()
    optimizer = make_optimizer(policy, internal_act_experiment(training_config))
    optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(7)
    loss, loss_dict = policy.forward(cast(dict[str, torch.Tensor], conditioned))
    if not torch.isfinite(loss):
        raise FactorFiLMContractError("FactorFiLM fixture loss is nonfinite")
    loss.backward()

    gradient_groups = {
        "base_act": tuple(
            value.grad for name, value in policy.named_parameters() if name.startswith("model.")
        ),
        "object_embedding": (policy.target_object_condition.embedding.weight.grad,),
        "bin_embedding": (policy.destination_bin_condition.embedding.weight.grad,),
        "object_film_projection": tuple(
            value.grad for value in policy.target_object_condition.film_projection.parameters()
        ),
        "bin_film_projection": tuple(
            value.grad for value in policy.destination_bin_condition.film_projection.parameters()
        ),
    }
    gradient_evidence: dict[str, bool] = {}
    for name, values in gradient_groups.items():
        present = tuple(value for value in values if isinstance(value, torch.Tensor))
        gradient_evidence[name] = (
            bool(present)
            and all(bool(torch.isfinite(value).all()) for value in present)
            and any(bool(torch.any(value != 0)) for value in present)
        )
    if not all(gradient_evidence.values()):
        raise FactorFiLMContractError("FactorFiLM fixture lacks finite nonzero gradients")
    optimizer.step()

    dummy = "sha256:" + "4" * 64
    identity = FactorFiLMRunIdentity(
        m3b_export_fingerprint=AUTHORIZED_M3B_FINGERPRINT,
        m3b_split_manifest_digest=AUTHORIZED_M3B_SPLIT_FINGERPRINT,
        ordered_train_episode_indices=tuple(range(6)),
        ordered_validation_episode_indices=(),
        architecture_identity=architecture.to_dict(),
        model_config={
            "base_act": model.to_dict(),
            "factor_film": policy.factor_film_config.to_dict(),
        },
        data_contract={
            "fixture": True,
            "checkpoint_selection_source": "m3b_validation",
            "test_accessible": False,
            "sealed_schedule_accessed": False,
        },
        train_statistics_fingerprint=dummy,
        optimization_config=optimization.to_dict(),
        semantic_audit_evidence_fingerprint=AUTHORIZED_M43_AUDIT_EVIDENCE_FINGERPRINT,
        training_seed=43,
        experiment_mode=FactorFiLMTrainingMode.FIXTURE,
        device="cpu",
        dtype="float32",
        lerobot_version="0.6.0",
        torch_version=torch.__version__,
        cuda_version=None,
        git_commit=git_commit,
        git_dirty=False,
    )
    checkpoint_contract = build_factor_film_checkpoint_contract(
        identity=identity,
        architecture=architecture,
    )
    policy.eval()
    torch.manual_seed(11)
    expected = policy.predict_action_chunk(cast(dict[str, torch.Tensor], conditioned))
    with tempfile.TemporaryDirectory(prefix="langmani-m43-factor-film-fixture-") as temporary:
        run_root = Path(temporary) / "run"
        run_root.mkdir()
        record = save_act_checkpoint(
            run_root=run_root,
            identity=identity,
            training_state=TrainingState(global_step=1, examples_processed=6),
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            optimizer=optimizer,
            training_metric={"step": 1, "total_loss": float(loss.detach())},
        )
        loaded = load_act_checkpoint(
            run_root=run_root,
            checkpoint_relative_path=record.relative_path,
            expected_identity=identity,
            policy_class=FactorFiLMACTPolicy,
        )
        if not isinstance(loaded.policy, FactorFiLMACTPolicy):
            raise FactorFiLMContractError("fixture reload returned a non-FactorFiLM policy")
        validate_factor_film_policy_structure(loaded.policy)
        reprocessed = loaded.preprocessor(prepare_raw_batch(raw))
        if not isinstance(reprocessed, Mapping):
            raise FactorFiLMContractError("reloaded processor returned a non-mapping")
        reconditioned = augmenter(reprocessed, raw)
        actual = loaded.policy.predict_action_chunk(cast(dict[str, torch.Tensor], reconditioned))
        reload_match = torch.allclose(
            expected,
            actual,
            atol=FACTOR_FILM_RELOAD_ATOL,
            rtol=FACTOR_FILM_RELOAD_RTOL,
        )
        if not reload_match:
            raise FactorFiLMContractError("FactorFiLM fresh reload changed deterministic output")

    return {
        "passed": True,
        "fixture_training_only": True,
        "fixture_seed": 43,
        "finite_loss": True,
        "loss": float(loss.detach()),
        "loss_components": dict(loss_dict),
        "gradient_evidence": gradient_evidence,
        "optimizer_step_completed": True,
        "processor_reload_validated": True,
        "checkpoint_reload_validated": True,
        "deterministic_inference_reload_validated": True,
        "deterministic_inference_atol": FACTOR_FILM_RELOAD_ATOL,
        "deterministic_inference_rtol": FACTOR_FILM_RELOAD_RTOL,
        "identity_initialization_max_abs_error": identity_error,
        "object_condition_changes_visual_path": True,
        "bin_condition_changes_state_path": True,
        "object_condition_changes_state_path": False,
        "bin_condition_changes_visual_path": False,
        "base_act_parameter_count": architecture.base_parameter_count,
        "factor_film_parameter_count": architecture.factor_film_parameter_count,
        "parameter_count_increase": architecture.parameter_count_increase,
        "architecture_identity": architecture.to_dict(),
        "checkpoint_contract": checkpoint_contract.to_dict(),
        "factor_film_training_completed": False,
        "factor_film_checkpoints_complete": False,
        "factor_film_checkpoint_selected": False,
        "development_benchmark_completed": False,
        "final_schedule_accessed": False,
        "smolvla_go": False,
        "physical_target_validated": False,
    }


def fixture_dry_run_report(
    *,
    output_root: str | Path,
    git_commit: str,
) -> FactorFiLMDryRunReport:
    """Produce a clearly labeled no-dataset structural dry-run contract."""
    del git_commit
    seed_everything(43)
    model = ActModelConfig()
    config = FactorFiLMTrainingConfig(
        data=ActDataConfig(dataset_root="fixture-only"),
        base_model=model,
        optimization=ActOptimizationConfig(),
        semantic_audit_evidence_fingerprint=AUTHORIZED_M43_AUDIT_EVIDENCE_FINGERPRINT,
        mode=FactorFiLMTrainingMode.DRY_RUN,
        device="cpu",
    )
    _, policy, _, _, architecture = build_factor_film_policy_and_processors(
        training_config=config,
        statistics=_fixture_stats(),
        factor_film_config=FactorFiLMConfig(state_context_dimension=model.dim_model),
    )
    run_fingerprint = canonical_fingerprint(
        {
            "fixture": True,
            "architecture": architecture.to_dict(),
            "config": config.to_dict(),
        }
    )
    return FactorFiLMDryRunReport(
        dataset_fingerprint="sha256:" + "1" * 64,
        split_fingerprint="sha256:" + "2" * 64,
        train_episode_count=6,
        validation_episode_count=0,
        feature_shapes={
            "observation.images.base_camera": [3, 256, 256],
            "observation.state": [9],
            "action": [model.chunk_size, model.action_dimension],
        },
        panda_policy_state_dimension=9,
        object_mapping=policy.factor_film_config.object_mapping.to_dict(),
        bin_mapping=policy.factor_film_config.bin_mapping.to_dict(),
        visual_film_injection=policy.factor_film_config.visual_injection_target,
        state_film_injection=policy.factor_film_config.state_injection_target,
        base_act_parameter_count=architecture.base_parameter_count,
        factor_film_parameter_count=architecture.factor_film_parameter_count,
        parameter_count_increase=architecture.parameter_count_increase,
        train_statistics_fingerprint="sha256:" + "3" * 64,
        effective_training_configuration=config.to_dict(),
        run_fingerprint=run_fingerprint,
        expected_output_path=str(
            Path(output_root).resolve() / run_fingerprint.removeprefix("sha256:")
        ),
        expected_checkpoint_steps=tuple(range(5_000, 100_001, 5_000)),
        semantic_audit_evidence_fingerprint=AUTHORIZED_M43_AUDIT_EVIDENCE_FINGERPRINT,
        fixture_contract=True,
    )


__all__ = [
    "AUTHORIZED_M3B_FINGERPRINT",
    "AUTHORIZED_M3B_SPLIT_FINGERPRINT",
    "AUTHORIZED_M43_AUDIT_EVIDENCE_FINGERPRINT",
    "FACTOR_FILM_IDENTITY_INITIALIZATION_ATOL",
    "FACTOR_FILM_RELOAD_ATOL",
    "FACTOR_FILM_RELOAD_RTOL",
    "FactorFiLMDataBundle",
    "FactorFiLMProcessedBatchAugmenter",
    "build_factor_film_checkpoint_contract",
    "build_factor_film_dry_run_report",
    "build_factor_film_policy_and_processors",
    "build_factor_film_run_identity",
    "build_factor_film_training_manifest",
    "build_factor_film_validation_queue",
    "create_factor_film_selection",
    "factor_film_dataloader",
    "factor_film_validation_schedule_digest",
    "factor_film_validation_schedule_records",
    "fixture_dry_run_report",
    "internal_act_experiment",
    "load_factor_film_data",
    "load_factor_film_temporal_views",
    "rank_factor_film_validation_results",
    "run_factor_film_fixture",
    "validate_authorizing_semantic_audit",
]
