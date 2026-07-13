"""LeRobot 0.6.0 ACT construction and bounded project-owned training loop."""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act import ACTConfig, ACTPolicy, make_act_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline

from langmani.datasets.lerobot_types import (
    ACTION_FEATURE_KEY,
    IMAGE_FEATURE_KEY,
    STATE_FEATURE_KEY,
)
from langmani.policies.act_conditioning import append_canonical_task_onehot
from langmani.policies.act_types import (
    ActExperimentConfig,
    ActModelConfig,
    ActOptimizationConfig,
)


class TrainingContractError(RuntimeError):
    """Raised when a batch, device, or numerical invariant is violated."""


class CheckpointCallback(Protocol):
    def __call__(
        self,
        *,
        step: int,
        examples_processed: int,
        policy: ACTPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        optimizer: torch.optim.Optimizer,
        metric: Mapping[str, object],
    ) -> str | None: ...


class DeterministicResumeBatchSampler:
    """Infinite epoch permutations addressable by completed optimizer step."""

    def __init__(
        self,
        *,
        dataset_size: int,
        batch_size: int,
        seed: int,
        start_step: int = 0,
    ) -> None:
        if dataset_size < 1 or batch_size < 1 or seed < 0 or start_step < 0:
            raise ValueError(
                "sampler sizes, seed, and start_step must be valid non-negative values"
            )
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.start_step = start_step
        self.batches_per_epoch = (dataset_size + batch_size - 1) // batch_size

    def __iter__(self):
        epoch, first_batch = divmod(self.start_step, self.batches_per_epoch)
        while True:
            generator = torch.Generator().manual_seed((self.seed + epoch) % (2**63 - 1))
            permutation = torch.randperm(self.dataset_size, generator=generator).tolist()
            for batch_in_epoch in range(first_batch, self.batches_per_epoch):
                start = batch_in_epoch * self.batch_size
                yield permutation[start : start + self.batch_size]
            epoch += 1
            first_batch = 0


@dataclass(frozen=True, slots=True)
class TrainingMetric:
    """One compact, truthful optimization event."""

    step: int
    examples_processed: int
    total_loss: float
    action_loss: float | None
    kl_loss: float | None
    learning_rate: float
    gradient_norm: float
    dataloader_time_s: float
    step_time_s: float
    throughput_examples_per_s: float
    gpu_allocated_bytes: int
    gpu_reserved_bytes: int
    validation_loss: float | None = None
    checkpoint_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    final_step: int
    examples_processed: int
    metrics: tuple[TrainingMetric, ...]


def seed_everything(seed: int, *, deterministic: bool = True) -> dict[str, object]:
    """Seed all M4 RNGs and return the settings recorded in run provenance."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    return {
        "seed": seed,
        "python_hash_seed": str(seed),
        "torch_deterministic_algorithms": deterministic,
        "cudnn_deterministic": deterministic,
        "cudnn_benchmark": not deterministic,
        "cuda_matmul_allow_tf32": False,
    }


def seed_dataloader_worker(worker_id: int) -> None:
    """Derive NumPy/Python worker seeds from Torch's deterministic worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_act_config(
    model: ActModelConfig,
    *,
    device: str,
    use_amp: bool,
    optimization: ActOptimizationConfig | None = None,
) -> ACTConfig:
    """Translate the portable model contract to the inspected ACTConfig API."""
    effective_optimization = optimization or ActOptimizationConfig()
    config = ACTConfig(
        input_features={
            IMAGE_FEATURE_KEY: PolicyFeature(FeatureType.VISUAL, model.image_shape_chw),
            STATE_FEATURE_KEY: PolicyFeature(FeatureType.STATE, (model.state_dimension,)),
        },
        output_features={
            ACTION_FEATURE_KEY: PolicyFeature(FeatureType.ACTION, (model.action_dimension,))
        },
        device=device,
        use_amp=use_amp,
        use_peft=model.use_peft,
        push_to_hub=model.push_to_hub,
        repo_id=model.repo_id,
        private=model.private,
        tags=list(model.tags) if model.tags is not None else None,
        license=model.license,
        pretrained_path=model.pretrained_path,
        pretrained_revision=model.pretrained_revision,
        chunk_size=model.chunk_size,
        n_action_steps=model.n_action_steps,
        n_obs_steps=model.n_obs_steps,
        normalization_mapping={
            key: NormalizationMode(value) for key, value in model.normalization_mapping.items()
        },
        vision_backbone=model.vision_backbone,
        pretrained_backbone_weights=model.pretrained_backbone_weights,
        replace_final_stride_with_dilation=model.replace_final_stride_with_dilation,
        pre_norm=model.pre_norm,
        dim_model=model.dim_model,
        n_heads=model.n_heads,
        dim_feedforward=model.dim_feedforward,
        feedforward_activation=model.feedforward_activation,
        n_encoder_layers=model.n_encoder_layers,
        n_decoder_layers=model.n_decoder_layers,
        use_vae=model.use_vae,
        latent_dim=model.latent_dim,
        n_vae_encoder_layers=model.n_vae_encoder_layers,
        temporal_ensemble_coeff=model.temporal_ensemble_coeff,
        dropout=model.dropout,
        kl_weight=model.kl_weight,
        optimizer_lr=effective_optimization.learning_rate,
        optimizer_weight_decay=effective_optimization.weight_decay,
        optimizer_lr_backbone=effective_optimization.backbone_learning_rate,
    )
    # PreTrainedConfig warns and silently changes unavailable devices/AMP.  That
    # behavior is unsafe for a provenance-bound M4 run, so reject the mutation.
    if str(config.device) != device:
        raise TrainingContractError(
            f"ACTConfig silently changed requested device {device!r} to {config.device!r}"
        )
    if bool(config.use_amp) != use_amp:
        raise TrainingContractError("ACTConfig silently changed the requested AMP setting")
    if tuple(config.action_delta_indices or ()) != model.action_delta_indices:
        raise TrainingContractError("ACTConfig action delta indices differ from the pinned model")
    if config.observation_delta_indices != model.observation_delta_indices:
        raise TrainingContractError(
            "ACTConfig observation delta indices differ from the pinned model"
        )
    return config


def build_fixture_act_config(*, state_dimension: int = 9, device: str = "cpu") -> ACTConfig:
    """Small API fixture; it is never a quality or physical-validation model."""
    model = ActModelConfig(
        state_dimension=state_dimension,
        image_shape_chw=(3, 256, 256),
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
    return build_act_config(model, device=device, use_amp=False)


def build_policy_and_processors(
    config: ACTConfig,
    train_statistics: Mapping[str, Mapping[str, torch.Tensor]],
) -> tuple[ACTPolicy, PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Construct ACT and processors only from public installed interfaces."""
    stats = {
        feature: {name: tensor.detach().clone() for name, tensor in values.items()}
        for feature, values in train_statistics.items()
    }
    policy = ACTPolicy(config)
    preprocessor, postprocessor = make_act_pre_post_processors(config, dataset_stats=stats)
    return policy, preprocessor, postprocessor


def image_to_policy_float(value: torch.Tensor) -> torch.Tensor:
    """Apply the conversion missing from the ACT processor: uint8 -> float [0,1]."""
    if not isinstance(value, torch.Tensor):
        raise TypeError("base_camera image must be a torch.Tensor")
    if value.dtype == torch.uint8:
        result = value.to(dtype=torch.float32).div_(255.0)
    elif value.dtype == torch.float32:
        result = value
    else:
        raise TrainingContractError(
            f"base_camera dtype must be uint8 or float32, got {value.dtype}"
        )
    if result.ndim not in (3, 4) or result.shape[-3:] != (3, 256, 256):
        raise TrainingContractError(
            "base_camera tensor must have shape (3,256,256) or (B,3,256,256)"
        )
    if not torch.isfinite(result).all() or result.min() < 0 or result.max() > 1:
        raise TrainingContractError("base_camera values must be finite and lie in [0,1]")
    return result


def prepare_raw_batch(batch: Mapping[str, object]) -> dict[str, object]:
    """Project only the ACT allowlist and perform the explicit image conversion."""
    required = {IMAGE_FEATURE_KEY, STATE_FEATURE_KEY}
    missing = sorted(required - set(batch))
    if missing:
        raise TrainingContractError("ACT batch is missing: " + ", ".join(missing))
    image = batch[IMAGE_FEATURE_KEY]
    state = batch[STATE_FEATURE_KEY]
    if not isinstance(image, torch.Tensor) or not isinstance(state, torch.Tensor):
        raise TrainingContractError("ACT observations must be torch tensors")
    projected: dict[str, object] = {
        IMAGE_FEATURE_KEY: image_to_policy_float(image),
        STATE_FEATURE_KEY: state.to(dtype=torch.float32),
    }
    if ACTION_FEATURE_KEY in batch:
        action = batch[ACTION_FEATURE_KEY]
        if not isinstance(action, torch.Tensor):
            raise TrainingContractError("action must be a torch tensor")
        projected[ACTION_FEATURE_KEY] = action.to(dtype=torch.float32)
    if "action_is_pad" in batch:
        padding = batch["action_is_pad"]
        if not isinstance(padding, torch.Tensor):
            raise TrainingContractError("action_is_pad must be a torch tensor")
        projected["action_is_pad"] = padding.to(dtype=torch.bool)
    # task text, IDs, scene IDs, timestamps and success labels are intentionally
    # absent from the tensor batch supplied to standard ACT.
    return projected


def append_task_condition_to_batch(
    batch: Mapping[str, object],
    *,
    task_id_by_episode: Mapping[int, str],
) -> dict[str, object]:
    """Append CanonicalTaskOneHotV0 using stable episode provenance only."""
    state = batch.get(STATE_FEATURE_KEY)
    episode_index = batch.get("episode_index")
    if not isinstance(state, torch.Tensor) or not isinstance(episode_index, torch.Tensor):
        raise TrainingContractError(
            "task-conditioned batches require state and numeric episode_index tensors"
        )
    if state.dtype != torch.float32 or state.ndim != 2 or state.shape[1] != 9:
        raise TrainingContractError("task-conditioned raw state must be float32[B,9]")
    indices = tuple(int(value) for value in episode_index.detach().cpu().reshape(-1).tolist())
    try:
        task_ids = tuple(task_id_by_episode[index] for index in indices)
    except KeyError as error:
        raise TrainingContractError(
            f"batch episode {error.args[0]} has no stable M3B task mapping"
        ) from error
    augmented = append_canonical_task_onehot(state, task_ids)
    if not isinstance(augmented, torch.Tensor) or tuple(augmented.shape) != (
        state.shape[0],
        15,
    ):
        raise TrainingContractError("CanonicalTaskOneHotV0 failed to produce state[B,15]")
    result = dict(batch)
    result[STATE_FEATURE_KEY] = augmented
    return result


def validate_processed_training_batch(
    batch: Mapping[str, object],
    *,
    state_dimension: int,
    chunk_size: int,
) -> None:
    expected = {
        IMAGE_FEATURE_KEY: (3, 256, 256),
        STATE_FEATURE_KEY: (state_dimension,),
        ACTION_FEATURE_KEY: (chunk_size, 8),
        "action_is_pad": (chunk_size,),
    }
    for key, trailing_shape in expected.items():
        value = batch.get(key)
        if not isinstance(value, torch.Tensor):
            raise TrainingContractError(f"processed batch is missing tensor {key!r}")
        if tuple(value.shape[1:]) != trailing_shape:
            raise TrainingContractError(
                f"processed {key} trailing shape must be {trailing_shape}, got {tuple(value.shape)}"
            )
        if value.dtype != torch.bool and not torch.isfinite(value).all():
            raise TrainingContractError(f"processed {key} contains nonfinite values")


def make_optimizer(policy: ACTPolicy, config: ActExperimentConfig) -> torch.optim.AdamW:
    """Use ACT's public parameter groups with the pinned AdamW preset values."""
    groups = policy.get_optim_params()
    if len(groups) != 2:
        raise TrainingContractError("installed ACT must expose backbone/non-backbone groups")
    groups[1]["lr"] = config.optimization.backbone_learning_rate
    return torch.optim.AdamW(
        groups,
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )


def _autocast_context(config: ActExperimentConfig) -> Any:
    precision = config.optimization.mixed_precision
    if precision == "none":
        return nullcontext()
    if config.device != "cuda":
        raise TrainingContractError("mixed precision is supported only for declared CUDA runs")
    dtype = torch.float16 if precision == "float16" else torch.bfloat16
    if precision == "float16":
        raise TrainingContractError(
            "M4 primary training uses bfloat16 autocast; float16 would require persisted GradScaler state"
        )
    if not torch.cuda.is_bf16_supported():
        raise TrainingContractError("requested CUDA device does not support bfloat16 autocast")
    return torch.autocast(device_type="cuda", dtype=dtype)


def _finite_loss_dict(loss_dict: Mapping[str, float]) -> None:
    for key, value in loss_dict.items():
        if not math.isfinite(float(value)):
            raise TrainingContractError(f"nonfinite ACT loss component {key!r}: {value}")


def _gpu_memory(device: str) -> tuple[int, int]:
    if device != "cuda":
        return 0, 0
    return int(torch.cuda.memory_allocated()), int(torch.cuda.memory_reserved())


def train_act(
    *,
    experiment: ActExperimentConfig,
    policy: ACTPolicy,
    preprocessor: PolicyProcessorPipeline,
    optimizer: torch.optim.Optimizer,
    train_batches: Iterable[Mapping[str, object]],
    start_step: int = 0,
    maximum_steps: int | None = None,
    metrics_path: Path | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
    postprocessor: PolicyProcessorPipeline | None = None,
    validation_callback: Callable[[int, ACTPolicy], float] | None = None,
    task_id_by_episode: Mapping[int, str] | None = None,
    seed_at_start: bool = True,
    initial_examples_processed: int = 0,
) -> TrainingOutcome:
    """Run a deterministic, bounded ACT optimization loop.

    The caller owns the exact episode-scoped DataLoader and train-only processor.
    This loop never receives validation or test datasets and never selects a
    checkpoint from test metrics.
    """
    target_steps = maximum_steps or experiment.optimization.training_steps
    if target_steps <= start_step:
        raise ValueError("maximum_steps must exceed start_step")
    if checkpoint_callback is not None and postprocessor is None:
        raise ValueError("checkpoint_callback requires the saved postprocessor")
    if seed_at_start:
        seed_everything(experiment.seed)
    policy.train()
    metrics: list[TrainingMetric] = []
    if initial_examples_processed < 0:
        raise ValueError("initial_examples_processed must be non-negative")
    examples_processed = initial_examples_processed
    iterator = iter(train_batches)
    last_batch_end = time.perf_counter()

    metrics_file = None
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_path.open("a", encoding="utf-8", newline="\n")
    try:
        for step in range(start_step + 1, target_steps + 1):
            batch_wait_start = last_batch_end
            try:
                raw_batch = next(iterator)
            except StopIteration:
                iterator = iter(train_batches)
                try:
                    raw_batch = next(iterator)
                except StopIteration as error:
                    raise TrainingContractError("training DataLoader is empty") from error
            batch_ready = time.perf_counter()
            conditioned_batch = (
                append_task_condition_to_batch(
                    raw_batch,
                    task_id_by_episode=task_id_by_episode,
                )
                if experiment.model.state_dimension == 15
                else dict(raw_batch)
            )
            projected = prepare_raw_batch(conditioned_batch)
            processed = preprocessor(projected)
            if not isinstance(processed, Mapping):
                raise TrainingContractError("ACT preprocessor must return a mapping")
            validate_processed_training_batch(
                processed,
                state_dimension=experiment.model.state_dimension,
                chunk_size=experiment.model.chunk_size,
            )
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(experiment):
                loss, loss_dict = policy.forward(cast(dict[str, torch.Tensor], processed))
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise TrainingContractError(f"ACT returned invalid total loss {loss!r}")
            _finite_loss_dict(loss_dict)
            loss.backward()
            gradients = tuple(
                parameter.grad for parameter in policy.parameters() if parameter.grad is not None
            )
            if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
                raise TrainingContractError("ACT backward produced missing or nonfinite gradients")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), experiment.optimization.gradient_clip_norm
            )
            if not torch.isfinite(gradient_norm):
                raise TrainingContractError("gradient norm is nonfinite")
            optimizer.step()
            if experiment.device == "cuda":
                torch.cuda.synchronize()
            step_end = time.perf_counter()
            state = cast(torch.Tensor, processed[STATE_FEATURE_KEY])
            batch_size = int(state.shape[0])
            examples_processed += batch_size
            validation_loss = None
            if (
                validation_callback is not None
                and step % experiment.optimization.validation_interval == 0
            ):
                validation_loss = float(validation_callback(step, policy))
                if not math.isfinite(validation_loss):
                    raise TrainingContractError("validation callback returned nonfinite loss")
                policy.train()
            allocated, reserved = _gpu_memory(experiment.device)
            duration = step_end - step_start
            metric = TrainingMetric(
                step=step,
                examples_processed=examples_processed,
                total_loss=float(loss.detach().cpu()),
                action_loss=(float(loss_dict["l1_loss"]) if "l1_loss" in loss_dict else None),
                kl_loss=(float(loss_dict["kld_loss"]) if "kld_loss" in loss_dict else None),
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                gradient_norm=float(gradient_norm.detach().cpu()),
                dataloader_time_s=batch_ready - batch_wait_start,
                step_time_s=duration,
                throughput_examples_per_s=batch_size / max(duration, 1e-12),
                gpu_allocated_bytes=allocated,
                gpu_reserved_bytes=reserved,
                validation_loss=validation_loss,
            )
            checkpoint_path = None
            if (
                checkpoint_callback is not None
                and step % experiment.optimization.checkpoint_interval == 0
            ):
                checkpoint_path = checkpoint_callback(
                    step=step,
                    examples_processed=examples_processed,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=cast(PolicyProcessorPipeline, postprocessor),
                    optimizer=optimizer,
                    metric=metric.to_dict(),
                )
                if checkpoint_path is not None and (
                    not isinstance(checkpoint_path, str) or not checkpoint_path
                ):
                    raise TrainingContractError(
                        "checkpoint callback must return a non-empty relative path or None"
                    )
                metric = replace(metric, checkpoint_path=checkpoint_path)
            metrics.append(metric)
            if metrics_file is not None:
                metrics_file.write(json.dumps(metric.to_dict(), sort_keys=True) + "\n")
                metrics_file.flush()
                os.fsync(metrics_file.fileno())
            last_batch_end = time.perf_counter()
    finally:
        if metrics_file is not None:
            metrics_file.close()
    return TrainingOutcome(
        final_step=target_steps,
        examples_processed=examples_processed,
        metrics=tuple(metrics),
    )


@torch.no_grad()
def evaluate_offline_loss(
    *,
    experiment: ActExperimentConfig,
    policy: ACTPolicy,
    preprocessor: PolicyProcessorPipeline,
    validation_batches: Iterable[Mapping[str, object]],
    task_id_by_episode: Mapping[int, str] | None = None,
    maximum_batches: int | None = None,
) -> float:
    """Compute validation loss only; this function cannot receive test results."""
    policy.eval()
    losses: list[float] = []
    for batch_index, raw_batch in enumerate(validation_batches):
        if maximum_batches is not None and batch_index >= maximum_batches:
            break
        conditioned_batch = (
            append_task_condition_to_batch(
                raw_batch,
                task_id_by_episode=task_id_by_episode,
            )
            if experiment.model.state_dimension == 15
            else dict(raw_batch)
        )
        processed = preprocessor(prepare_raw_batch(conditioned_batch))
        if not isinstance(processed, Mapping):
            raise TrainingContractError("ACT preprocessor must return a mapping")
        validate_processed_training_batch(
            processed,
            state_dimension=experiment.model.state_dimension,
            chunk_size=experiment.model.chunk_size,
        )
        loss, loss_dict = policy.forward(cast(dict[str, torch.Tensor], processed))
        if not torch.isfinite(loss):
            raise TrainingContractError("offline validation produced nonfinite loss")
        _finite_loss_dict(loss_dict)
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise TrainingContractError("validation DataLoader is empty")
    return float(sum(losses) / len(losses))


__all__ = [
    "TrainingContractError",
    "DeterministicResumeBatchSampler",
    "TrainingMetric",
    "TrainingOutcome",
    "append_task_condition_to_batch",
    "build_act_config",
    "build_fixture_act_config",
    "build_policy_and_processors",
    "image_to_policy_float",
    "make_optimizer",
    "prepare_raw_batch",
    "evaluate_offline_loss",
    "seed_dataloader_worker",
    "seed_everything",
    "train_act",
    "validate_processed_training_batch",
]
