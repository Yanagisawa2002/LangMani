"""Bounded training steps and immutable classifier artifact promotion for M5A."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from langmani.datasets.identity import sha256_hex
from langmani.language.stage_protocol import (
    CheckpointRetentionPolicy,
    ClassifierStageMetrics,
    ClassifierStageTrainingConfig,
)
from langmani.language.text_calibration import (
    RoutingThresholdExample,
    RoutingThresholdSelectionV0,
    TemperatureCalibrationV0,
    calibrated_full_route_confidence,
    fit_validation_temperature,
    select_validation_routing_threshold,
)
from langmani.language.text_classifier import (
    BIN_LABELS,
    OBJECT_LABELS,
    ROUTE_STATUS_INDEX,
    STATUS_LABELS,
    FactorizedTextClassifierOutput,
    FactorizedTextClassifierV0,
    FactorizedTextLoss,
    compute_factorized_text_loss,
)

TEXT_TRAINING_ARTIFACT_SCHEMA = "langmani-m5a-text-training-artifact-v1"


class TextTrainingError(RuntimeError):
    """Raised when bounded optimization or immutable artifact handling fails."""


@dataclass(frozen=True, slots=True)
class FactorizedTextBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    status_labels: torch.Tensor
    object_labels: torch.Tensor
    bin_labels: torch.Tensor
    split: str


@dataclass(frozen=True, slots=True)
class TextTrainingStepResult:
    total_loss: float
    status_loss: float
    object_loss: float
    bin_loss: float
    routed_examples: int
    gradient_norm: float
    encoder_gradient_norm: float
    status_head_gradient_norm: float
    object_head_gradient_norm: float
    bin_head_gradient_norm: float


@dataclass(frozen=True, slots=True)
class TextTrainingTelemetry:
    """One finite optimizer-step record from the authoritative staged run."""

    step: int
    total_loss: float
    status_loss: float
    object_loss: float
    bin_loss: float
    learning_rate: float
    gradient_norm: float
    encoder_gradient_norm: float
    status_head_gradient_norm: float
    object_head_gradient_norm: float
    bin_head_gradient_norm: float
    routed_examples: int
    batch_examples: int
    examples_processed: int
    epoch_progress: float
    throughput_examples_per_second: float
    data_loader_latency_seconds: float
    step_latency_seconds: float
    gpu_allocated_bytes: int
    gpu_reserved_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "total_loss": self.total_loss,
            "status_loss": self.status_loss,
            "object_loss": self.object_loss,
            "bin_loss": self.bin_loss,
            "learning_rate": self.learning_rate,
            "gradient_norm": self.gradient_norm,
            "encoder_gradient_norm": self.encoder_gradient_norm,
            "status_head_gradient_norm": self.status_head_gradient_norm,
            "object_head_gradient_norm": self.object_head_gradient_norm,
            "bin_head_gradient_norm": self.bin_head_gradient_norm,
            "routed_examples": self.routed_examples,
            "batch_examples": self.batch_examples,
            "examples_processed": self.examples_processed,
            "epoch_progress": self.epoch_progress,
            "throughput_examples_per_second": self.throughput_examples_per_second,
            "data_loader_latency_seconds": self.data_loader_latency_seconds,
            "step_latency_seconds": self.step_latency_seconds,
            "gpu_allocated_bytes": self.gpu_allocated_bytes,
            "gpu_reserved_bytes": self.gpu_reserved_bytes,
        }


@dataclass(frozen=True, slots=True)
class StagedTextCheckpointAudit:
    """Read-only proof that a staged checkpoint can continue without restarting."""

    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    restored_step: int
    next_global_step: int
    next_learning_rate: float
    optimizer_state_restored: bool
    scheduler_state_restored: bool
    processor_state_restored: bool
    rng_state_restored: bool
    data_progress_restored: bool
    deterministic_logits_match: bool
    maximum_absolute_logit_error: float
    absolute_tolerance: float
    relative_tolerance: float
    validation_fixture_examples: int
    validation_records_restored: int
    training_records_restored: int
    pilot_complete: bool
    full_training_complete: bool
    resumable: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
            "restored_step": self.restored_step,
            "next_global_step": self.next_global_step,
            "next_learning_rate": self.next_learning_rate,
            "optimizer_state_restored": self.optimizer_state_restored,
            "scheduler_state_restored": self.scheduler_state_restored,
            "processor_state_restored": self.processor_state_restored,
            "rng_state_restored": self.rng_state_restored,
            "data_progress_restored": self.data_progress_restored,
            "deterministic_logits_match": self.deterministic_logits_match,
            "maximum_absolute_logit_error": self.maximum_absolute_logit_error,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "validation_fixture_examples": self.validation_fixture_examples,
            "validation_records_restored": self.validation_records_restored,
            "training_records_restored": self.training_records_restored,
            "pilot_complete": self.pilot_complete,
            "full_training_complete": self.full_training_complete,
            "resumable": self.resumable,
        }


@dataclass(frozen=True, slots=True)
class TextTrainingConfig:
    """Predeclared bounded optimization and validation-checkpoint schedule."""

    maximum_steps: int
    checkpoint_steps: tuple[int, ...]
    learning_rate: float
    seed: int
    weight_decay: float = 0.01
    maximum_gradient_norm: float = 1.0
    early_stopping_patience: int | None = None
    encoder_trainable: bool = True
    optimizer: str = "adamw"
    scheduler: str = "constant"

    def __post_init__(self) -> None:
        if (
            isinstance(self.maximum_steps, bool)
            or not isinstance(self.maximum_steps, int)
            or self.maximum_steps <= 0
        ):
            raise ValueError("maximum_steps must be a positive integer")
        if (
            not isinstance(self.checkpoint_steps, tuple)
            or not self.checkpoint_steps
            or tuple(sorted(set(self.checkpoint_steps))) != self.checkpoint_steps
            or any(
                isinstance(step, bool)
                or not isinstance(step, int)
                or not 1 <= step <= self.maximum_steps
                for step in self.checkpoint_steps
            )
        ):
            raise ValueError("checkpoint_steps must be sorted unique steps inside the bound")
        if self.checkpoint_steps[-1] != self.maximum_steps:
            raise ValueError("checkpoint_steps must include maximum_steps")
        for name in ("learning_rate", "maximum_gradient_norm"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(self.weight_decay, bool)
            or not isinstance(self.weight_decay, int | float)
            or not math.isfinite(float(self.weight_decay))
            or float(self.weight_decay) < 0.0
        ):
            raise ValueError("weight_decay must be finite and non-negative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.early_stopping_patience is not None and (
            isinstance(self.early_stopping_patience, bool)
            or not isinstance(self.early_stopping_patience, int)
            or self.early_stopping_patience <= 0
        ):
            raise ValueError("early_stopping_patience must be positive or null")
        if not isinstance(self.encoder_trainable, bool):
            raise ValueError("encoder_trainable must be a boolean")
        if self.optimizer != "adamw" or self.scheduler != "constant":
            raise ValueError("M5A supports the declared AdamW/constant configuration only")

    def to_dict(self) -> dict[str, object]:
        return {
            "maximum_steps": self.maximum_steps,
            "checkpoint_steps": list(self.checkpoint_steps),
            "learning_rate": float(self.learning_rate),
            "seed": self.seed,
            "weight_decay": float(self.weight_decay),
            "maximum_gradient_norm": float(self.maximum_gradient_norm),
            "early_stopping_patience": self.early_stopping_patience,
            "encoder_trainable": self.encoder_trainable,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
        }


@dataclass(frozen=True, slots=True)
class TextCheckpointValidation:
    checkpoint_id: str
    step: int
    validation_examples: int
    total_loss: float
    status_loss: float
    object_loss: float
    bin_loss: float
    evidence_split: str = "validation"


@dataclass(frozen=True, slots=True)
class BoundedTextTrainingResult:
    completed_steps: int
    stopped_early: bool
    parameter_count: int
    selected_checkpoint_id: str
    selected_step: int
    checkpoint_validations: tuple[TextCheckpointValidation, ...]
    config_fingerprint: str

    def to_dict(self) -> dict[str, object]:
        return {
            "completed_steps": self.completed_steps,
            "stopped_early": self.stopped_early,
            "parameter_count": self.parameter_count,
            "selected_checkpoint_id": self.selected_checkpoint_id,
            "selected_step": self.selected_step,
            "checkpoint_validations": [
                {
                    "checkpoint_id": value.checkpoint_id,
                    "step": value.step,
                    "validation_examples": value.validation_examples,
                    "total_loss": value.total_loss,
                    "status_loss": value.status_loss,
                    "object_loss": value.object_loss,
                    "bin_loss": value.bin_loss,
                    "evidence_split": value.evidence_split,
                }
                for value in self.checkpoint_validations
            ],
            "config_fingerprint": self.config_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class FactorizedValidationOutputs:
    status_logits: torch.Tensor
    object_logits: torch.Tensor
    bin_logits: torch.Tensor
    status_labels: torch.Tensor
    object_labels: torch.Tensor
    bin_labels: torch.Tensor
    evidence_split: str = "validation"
    batch_latency_seconds: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class TextRouterCalibrationSelection:
    temperature: TemperatureCalibrationV0
    threshold: RoutingThresholdSelectionV0

    def to_dict(self) -> dict[str, object]:
        return {
            "temperature": {
                "temperature": self.temperature.temperature,
                "validation_examples": self.temperature.validation_examples,
                "nll_before": self.temperature.nll_before,
                "nll_after": self.temperature.nll_after,
                "bounded_iterations": self.temperature.bounded_iterations,
            },
            "threshold": {
                "threshold": self.threshold.threshold,
                "validation_routeable_examples": (self.threshold.validation_routeable_examples),
                "validation_rejected_examples": (self.threshold.validation_rejected_examples),
                "valid_full_task_accuracy": self.threshold.valid_full_task_accuracy,
                "false_route_rate": self.threshold.false_route_rate,
                "rejection_recall": self.threshold.rejection_recall,
            },
        }


@dataclass(frozen=True, slots=True)
class StagedTextTrainingResult:
    """Pause/resume result from the one authoritative classifier run."""

    run_fingerprint: str
    phase: str
    completed_steps: int
    completed_epochs: int
    resumed_from_step: int
    pilot_checkpoint: str
    latest_checkpoint: str
    best_checkpoint: str
    selected_step: int | None
    stopped_early: bool
    pilot_metrics: ClassifierStageMetrics | None
    final_metrics: ClassifierStageMetrics | None
    validation_records: tuple[TextCheckpointValidation, ...]
    training_records: tuple[TextTrainingTelemetry, ...]
    pilot_outputs: FactorizedValidationOutputs | None
    classifier_training_seeds: int = 1
    robustness_across_training_seeds_not_evaluated: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "run_fingerprint": self.run_fingerprint,
            "phase": self.phase,
            "completed_steps": self.completed_steps,
            "completed_epochs": self.completed_epochs,
            "resumed_from_step": self.resumed_from_step,
            "pilot_checkpoint": self.pilot_checkpoint,
            "latest_checkpoint": self.latest_checkpoint,
            "best_checkpoint": self.best_checkpoint,
            "selected_step": self.selected_step,
            "stopped_early": self.stopped_early,
            "pilot_metrics": None if self.pilot_metrics is None else self.pilot_metrics.to_dict(),
            "final_metrics": None if self.final_metrics is None else self.final_metrics.to_dict(),
            "validation_records": [
                {
                    "checkpoint_id": value.checkpoint_id,
                    "step": value.step,
                    "validation_examples": value.validation_examples,
                    "total_loss": value.total_loss,
                    "status_loss": value.status_loss,
                    "object_loss": value.object_loss,
                    "bin_loss": value.bin_loss,
                    "evidence_split": value.evidence_split,
                }
                for value in self.validation_records
            ],
            "training_records": [value.to_dict() for value in self.training_records],
            "classifier_training_seeds": self.classifier_training_seeds,
            "robustness_across_training_seeds_not_evaluated": (
                self.robustness_across_training_seeds_not_evaluated
            ),
        }


def _module_gradient_norm(module: nn.Module) -> torch.Tensor:
    squares: list[torch.Tensor] = []
    for parameter in module.parameters():
        if parameter.grad is not None:
            squares.append(parameter.grad.detach().float().pow(2).sum())
    if not squares:
        return torch.zeros((), dtype=torch.float32)
    return torch.sqrt(torch.stack(squares).sum())


def train_factorized_text_step(
    *,
    model: FactorizedTextClassifierV0,
    optimizer: torch.optim.Optimizer,
    batch: FactorizedTextBatch,
    maximum_gradient_norm: float = 1.0,
) -> TextTrainingStepResult:
    """Run one explicit train-only gradient step."""

    if batch.split != "train":
        raise TextTrainingError("classifier gradients may use the train split only")
    if maximum_gradient_norm <= 0.0:
        raise TextTrainingError("maximum_gradient_norm must be positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
    loss: FactorizedTextLoss = compute_factorized_text_loss(
        output,
        status_labels=batch.status_labels,
        object_labels=batch.object_labels,
        bin_labels=batch.bin_labels,
    )
    losses = (loss.total, loss.status, loss.target_object, loss.target_bin)
    if not all(bool(torch.isfinite(value)) for value in losses):
        raise TextTrainingError("classifier loss is non-finite")
    loss.total.backward()
    component_norms = {
        "encoder": _module_gradient_norm(model.encoder),
        "status": _module_gradient_norm(model.status_head),
        "object": _module_gradient_norm(model.object_head),
        "bin": _module_gradient_norm(model.bin_head),
    }
    if not all(bool(torch.isfinite(value)) for value in component_norms.values()):
        raise TextTrainingError("classifier component gradient norm is non-finite")
    gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), maximum_gradient_norm)
    if not bool(torch.isfinite(gradient_norm)):
        raise TextTrainingError("classifier gradient norm is non-finite")
    optimizer.step()
    return TextTrainingStepResult(
        total_loss=float(loss.total.detach()),
        status_loss=float(loss.status.detach()),
        object_loss=float(loss.target_object.detach()),
        bin_loss=float(loss.target_bin.detach()),
        routed_examples=loss.routed_examples,
        gradient_norm=float(gradient_norm.detach()),
        encoder_gradient_norm=float(component_norms["encoder"]),
        status_head_gradient_norm=float(component_norms["status"]),
        object_head_gradient_norm=float(component_norms["object"]),
        bin_head_gradient_norm=float(component_norms["bin"]),
    )


def _batch_on_device(batch: FactorizedTextBatch, device: torch.device) -> FactorizedTextBatch:
    return FactorizedTextBatch(
        input_ids=batch.input_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
        status_labels=batch.status_labels.to(device),
        object_labels=batch.object_labels.to(device),
        bin_labels=batch.bin_labels.to(device),
        split=batch.split,
    )


def collect_factorized_validation_outputs(
    *,
    model: FactorizedTextClassifierV0,
    batches: Sequence[FactorizedTextBatch],
) -> FactorizedValidationOutputs:
    """Collect logits and labels from validation only, never development/final."""

    values = tuple(batches)
    if not values or any(batch.split != "validation" for batch in values):
        raise TextTrainingError("classifier validation evidence must use validation split only")
    try:
        device = next(model.parameters()).device
    except StopIteration as error:
        raise TextTrainingError("classifier has no trainable or frozen parameters") from error
    model.eval()
    status_logits: list[torch.Tensor] = []
    object_logits: list[torch.Tensor] = []
    bin_logits: list[torch.Tensor] = []
    status_labels: list[torch.Tensor] = []
    object_labels: list[torch.Tensor] = []
    bin_labels: list[torch.Tensor] = []
    latency_seconds: list[float] = []
    with torch.inference_mode():
        for original in values:
            batch = _batch_on_device(original, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            output = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latency_seconds.append(time.perf_counter() - started)
            status_logits.append(output.status_logits.detach().cpu())
            object_logits.append(output.object_logits.detach().cpu())
            bin_logits.append(output.bin_logits.detach().cpu())
            status_labels.append(batch.status_labels.detach().cpu())
            object_labels.append(batch.object_labels.detach().cpu())
            bin_labels.append(batch.bin_labels.detach().cpu())
    return FactorizedValidationOutputs(
        status_logits=torch.cat(status_logits),
        object_logits=torch.cat(object_logits),
        bin_logits=torch.cat(bin_logits),
        status_labels=torch.cat(status_labels),
        object_labels=torch.cat(object_labels),
        bin_labels=torch.cat(bin_labels),
        batch_latency_seconds=tuple(latency_seconds),
    )


def collect_factorized_tiny_train_outputs(
    *,
    model: FactorizedTextClassifierV0,
    batches: Sequence[FactorizedTextBatch],
) -> FactorizedValidationOutputs:
    """Collect tiny-train logits without mislabelling them as held-out validation."""

    values = tuple(batches)
    if not values or any(batch.split != "train" for batch in values):
        raise TextTrainingError("tiny-overfit evidence must come from train-only batches")
    try:
        device = next(model.parameters()).device
    except StopIteration as error:
        raise TextTrainingError("classifier has no parameters") from error
    model.eval()
    output_parts: list[FactorizedTextClassifierOutput] = []
    status_labels: list[torch.Tensor] = []
    object_labels: list[torch.Tensor] = []
    bin_labels: list[torch.Tensor] = []
    with torch.inference_mode():
        for original in values:
            batch = _batch_on_device(original, device)
            output = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
            output_parts.append(
                FactorizedTextClassifierOutput(
                    status_logits=output.status_logits.detach().cpu(),
                    object_logits=output.object_logits.detach().cpu(),
                    bin_logits=output.bin_logits.detach().cpu(),
                )
            )
            status_labels.append(batch.status_labels.detach().cpu())
            object_labels.append(batch.object_labels.detach().cpu())
            bin_labels.append(batch.bin_labels.detach().cpu())
    return FactorizedValidationOutputs(
        status_logits=torch.cat([value.status_logits for value in output_parts]),
        object_logits=torch.cat([value.object_logits for value in output_parts]),
        bin_logits=torch.cat([value.bin_logits for value in output_parts]),
        status_labels=torch.cat(status_labels),
        object_labels=torch.cat(object_labels),
        bin_labels=torch.cat(bin_labels),
        evidence_split="tiny_train",
    )


def _validation_record(
    *,
    outputs: FactorizedValidationOutputs,
    step: int,
) -> TextCheckpointValidation:
    loss = compute_factorized_text_loss(
        FactorizedTextClassifierOutput(
            status_logits=outputs.status_logits,
            object_logits=outputs.object_logits,
            bin_logits=outputs.bin_logits,
        ),
        status_labels=outputs.status_labels,
        object_labels=outputs.object_labels,
        bin_labels=outputs.bin_labels,
    )
    return TextCheckpointValidation(
        checkpoint_id=f"step_{step:08d}",
        step=step,
        validation_examples=outputs.status_labels.shape[0],
        total_loss=float(loss.total),
        status_loss=float(loss.status),
        object_loss=float(loss.target_object),
        bin_loss=float(loss.target_bin),
    )


def select_validation_checkpoint(
    records: Sequence[TextCheckpointValidation],
    *,
    evidence_split: str,
) -> TextCheckpointValidation:
    """Select minimum validation total loss, breaking exact ties by earlier step."""

    values = tuple(records)
    if evidence_split != "validation" or any(
        value.evidence_split != "validation" for value in values
    ):
        raise TextTrainingError("checkpoint selection may use validation evidence only")
    if not values:
        raise TextTrainingError("checkpoint selection requires at least one checkpoint")
    if len({value.checkpoint_id for value in values}) != len(values):
        raise TextTrainingError("checkpoint selection records must have unique IDs")
    if any(
        not math.isfinite(value.total_loss) or value.validation_examples <= 0 or value.step <= 0
        for value in values
    ):
        raise TextTrainingError("checkpoint validation records are malformed")
    return min(values, key=lambda value: (value.total_loss, value.step, value.checkpoint_id))


def run_bounded_factorized_text_training(
    *,
    model: FactorizedTextClassifierV0,
    train_batches: Sequence[FactorizedTextBatch],
    validation_batches: Sequence[FactorizedTextBatch],
    config: TextTrainingConfig,
    checkpoint_callback: Callable[[str, FactorizedTextClassifierV0, TextCheckpointValidation], None]
    | None = None,
) -> BoundedTextTrainingResult:
    """Run one deterministic bounded train/validation loop with no split leakage."""

    training = tuple(train_batches)
    validation = tuple(validation_batches)
    if not training or any(batch.split != "train" for batch in training):
        raise TextTrainingError("classifier gradients require non-empty train-only batches")
    if not validation or any(batch.split != "validation" for batch in validation):
        raise TextTrainingError("checkpoint selection requires validation-only batches")
    if not isinstance(config, TextTrainingConfig):
        raise TypeError("config must be TextTrainingConfig")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(config.encoder_trainable)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    try:
        device = next(model.parameters()).device
    except StopIteration as error:
        raise TextTrainingError("classifier has no parameters") from error
    records: list[TextCheckpointValidation] = []
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    non_improving = 0
    stopped_early = False
    completed_steps = 0
    schedule = set(config.checkpoint_steps)
    for step in range(1, config.maximum_steps + 1):
        train_factorized_text_step(
            model=model,
            optimizer=optimizer,
            batch=_batch_on_device(training[(step - 1) % len(training)], device),
            maximum_gradient_norm=float(config.maximum_gradient_norm),
        )
        completed_steps = step
        if step not in schedule:
            continue
        validation_outputs = collect_factorized_validation_outputs(
            model=model,
            batches=validation,
        )
        record = _validation_record(outputs=validation_outputs, step=step)
        records.append(record)
        if checkpoint_callback is not None:
            checkpoint_callback(record.checkpoint_id, model, record)
        if record.total_loss < best_loss - 1e-12:
            best_loss = record.total_loss
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            non_improving = 0
        else:
            non_improving += 1
        if (
            config.early_stopping_patience is not None
            and non_improving >= config.early_stopping_patience
        ):
            stopped_early = True
            break
    selected = select_validation_checkpoint(records, evidence_split="validation")
    if best_state is None:
        raise TextTrainingError("bounded training produced no selected checkpoint state")
    model.load_state_dict(best_state, strict=True)
    return BoundedTextTrainingResult(
        completed_steps=completed_steps,
        stopped_early=stopped_early,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        selected_checkpoint_id=selected.checkpoint_id,
        selected_step=selected.step,
        checkpoint_validations=tuple(records),
        config_fingerprint=f"sha256:{sha256_hex(config.to_dict())}",
    )


def classifier_stage_metrics(outputs: FactorizedValidationOutputs) -> ClassifierStageMetrics:
    """Compute every explicit pilot/training gate from validation or tiny outputs."""

    if outputs.evidence_split not in {"validation", "tiny_train"}:
        raise TextTrainingError("classifier stage metrics accept validation or tiny-train only")
    tensors = (
        outputs.status_logits,
        outputs.object_logits,
        outputs.bin_logits,
    )
    finite = all(bool(torch.isfinite(value).all()) for value in tensors)
    status = outputs.status_logits.argmax(dim=-1)
    objects = outputs.object_logits.argmax(dim=-1)
    bins = outputs.bin_logits.argmax(dim=-1)
    routeable = outputs.status_labels == ROUTE_STATUS_INDEX
    rejected = ~routeable

    def mean(mask: torch.Tensor) -> float:
        return float(mask.float().mean()) if mask.numel() else 0.0

    routed_count = int(routeable.sum())
    rejected_count = int(rejected.sum())
    if routed_count <= 0 or rejected_count <= 0:
        raise TextTrainingError("classifier stage metrics require routeable and rejected examples")
    routed_status = status[routeable] == ROUTE_STATUS_INDEX
    object_correct = objects[routeable] == outputs.object_labels[routeable]
    bin_correct = bins[routeable] == outputs.bin_labels[routeable]
    full_correct = routed_status & object_correct & bin_correct
    false_route = status[rejected] == ROUTE_STATUS_INDEX

    def recall(label: str) -> float:
        index = STATUS_LABELS.index(label)
        mask = outputs.status_labels == index
        if int(mask.sum()) <= 0:
            return 0.0
        return mean(status[mask] == index)

    return ClassifierStageMetrics(
        full_task_spec_accuracy=mean(full_correct),
        object_accuracy=mean(object_correct),
        bin_accuracy=mean(bin_correct),
        false_route_rate=mean(false_route),
        schema_valid_rate=1.0,
        ambiguous_rejection_recall=recall("reject_ambiguous"),
        unsupported_rejection_recall=recall("reject_unsupported"),
        malformed_rejection_recall=recall("reject_malformed"),
        status_accuracy=mean(status == outputs.status_labels),
        finite=finite,
    )


def _checkpoint_validation_from_dict(payload: Mapping[str, object]) -> TextCheckpointValidation:
    try:
        return TextCheckpointValidation(
            checkpoint_id=cast(str, payload["checkpoint_id"]),
            step=cast(int, payload["step"]),
            validation_examples=cast(int, payload["validation_examples"]),
            total_loss=float(cast(float, payload["total_loss"])),
            status_loss=float(cast(float, payload["status_loss"])),
            object_loss=float(cast(float, payload["object_loss"])),
            bin_loss=float(cast(float, payload["bin_loss"])),
            evidence_split=cast(str, payload["evidence_split"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise TextTrainingError("resumable checkpoint validation record is malformed") from error


def _training_telemetry_from_dict(payload: Mapping[str, object]) -> TextTrainingTelemetry:
    try:
        return TextTrainingTelemetry(
            step=cast(int, payload["step"]),
            total_loss=float(cast(float, payload["total_loss"])),
            status_loss=float(cast(float, payload["status_loss"])),
            object_loss=float(cast(float, payload["object_loss"])),
            bin_loss=float(cast(float, payload["bin_loss"])),
            learning_rate=float(cast(float, payload["learning_rate"])),
            gradient_norm=float(cast(float, payload["gradient_norm"])),
            encoder_gradient_norm=float(cast(float, payload["encoder_gradient_norm"])),
            status_head_gradient_norm=float(cast(float, payload["status_head_gradient_norm"])),
            object_head_gradient_norm=float(cast(float, payload["object_head_gradient_norm"])),
            bin_head_gradient_norm=float(cast(float, payload["bin_head_gradient_norm"])),
            routed_examples=cast(int, payload["routed_examples"]),
            batch_examples=cast(int, payload["batch_examples"]),
            examples_processed=cast(int, payload["examples_processed"]),
            epoch_progress=float(cast(float, payload["epoch_progress"])),
            throughput_examples_per_second=float(
                cast(float, payload["throughput_examples_per_second"])
            ),
            data_loader_latency_seconds=float(cast(float, payload["data_loader_latency_seconds"])),
            step_latency_seconds=float(cast(float, payload["step_latency_seconds"])),
            gpu_allocated_bytes=cast(int, payload["gpu_allocated_bytes"]),
            gpu_reserved_bytes=cast(int, payload["gpu_reserved_bytes"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise TextTrainingError("resumable checkpoint training telemetry is malformed") from error


def _rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(payload: Mapping[str, object]) -> None:
    try:
        random.setstate(cast(tuple[object, ...], payload["python"]))
        np.random.set_state(cast(tuple[Any, ...], payload["numpy"]))
        torch.set_rng_state(cast(torch.Tensor, payload["torch_cpu"]))
        cuda = payload["torch_cuda"]
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cast(list[torch.Tensor], cuda))
    except (KeyError, TypeError, RuntimeError, ValueError) as error:
        raise TextTrainingError("resumable checkpoint RNG state is malformed") from error


def _atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _save_staged_checkpoint(
    *,
    path: Path,
    role: str,
    run_fingerprint: str,
    config: ClassifierStageTrainingConfig,
    model: FactorizedTextClassifierV0,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    processor_state: Mapping[str, object],
    step: int,
    records: Sequence[TextCheckpointValidation],
    training_records: Sequence[TextTrainingTelemetry],
    best_loss: float,
    best_step: int | None,
    best_state: Mapping[str, torch.Tensor] | None,
    non_improving: int,
) -> None:
    if role not in CheckpointRetentionPolicy().retained_roles:
        raise TextTrainingError("checkpoint role is outside the bounded retention policy")
    payload = {
        "schema_version": "langmani-m5a-resumable-text-checkpoint-v0",
        "role": role,
        "run_fingerprint": run_fingerprint,
        "training_config": config.to_dict(),
        "training_config_fingerprint": config.fingerprint,
        "processor_state": dict(processor_state),
        "step": step,
        "next_global_step": step + 1,
        "pilot_complete": step >= config.pilot_step,
        "full_training_complete": False,
        "resumable": True,
        "data_progress": {
            "completed_optimizer_steps": step,
            "completed_epochs": step // config.steps_per_epoch,
            "next_batch_index": step % config.steps_per_epoch,
            "steps_per_epoch": config.steps_per_epoch,
            "reconstruction": "seeded-permutation-v0-modulo-step",
        },
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "rng_state": _rng_state(),
        "validation_records": [
            {
                "checkpoint_id": value.checkpoint_id,
                "step": value.step,
                "validation_examples": value.validation_examples,
                "total_loss": value.total_loss,
                "status_loss": value.status_loss,
                "object_loss": value.object_loss,
                "bin_loss": value.bin_loss,
                "evidence_split": value.evidence_split,
            }
            for value in records
        ],
        "training_records": [value.to_dict() for value in training_records],
        "best_loss": best_loss,
        "best_step": best_step,
        "best_state": None if best_state is None else dict(best_state),
        "non_improving": non_improving,
    }
    _atomic_torch_save(path, payload)


def _load_staged_checkpoint(
    *,
    path: Path,
    run_fingerprint: str,
    config: ClassifierStageTrainingConfig,
    processor_state: Mapping[str, object],
    model: FactorizedTextClassifierV0,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> tuple[
    int,
    list[TextCheckpointValidation],
    list[TextTrainingTelemetry],
    float,
    int | None,
    dict[str, torch.Tensor] | None,
    int,
]:
    if not path.is_file() or path.is_symlink():
        raise TextTrainingError("requested resumable classifier checkpoint is missing or linked")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise TextTrainingError("could not load resumable classifier checkpoint") from error
    if not isinstance(payload, Mapping):
        raise TextTrainingError("resumable classifier checkpoint must contain one mapping")
    if (
        payload.get("schema_version") != "langmani-m5a-resumable-text-checkpoint-v0"
        or payload.get("role") not in CheckpointRetentionPolicy().retained_roles
        or payload.get("run_fingerprint") != run_fingerprint
        or payload.get("training_config_fingerprint") != config.fingerprint
        or payload.get("training_config") != config.to_dict()
        or payload.get("processor_state") != dict(processor_state)
        or payload.get("resumable") is not True
        or payload.get("full_training_complete") is not False
    ):
        raise TextTrainingError("resumable classifier checkpoint identity differs")
    try:
        model.load_state_dict(cast(Mapping[str, torch.Tensor], payload["model_state"]), strict=True)
        optimizer.load_state_dict(cast(dict[str, object], payload["optimizer_state"]))
        scheduler.load_state_dict(cast(dict[str, object], payload["scheduler_state"]))
        _restore_rng_state(cast(Mapping[str, object], payload["rng_state"]))
        step = int(cast(int, payload["step"]))
        raw_records = cast(Sequence[Mapping[str, object]], payload["validation_records"])
        records = [_checkpoint_validation_from_dict(value) for value in raw_records]
        raw_training_records = cast(Sequence[Mapping[str, object]], payload["training_records"])
        training_records = [_training_telemetry_from_dict(value) for value in raw_training_records]
        best_loss = float(cast(float, payload["best_loss"]))
        best_step_raw = payload["best_step"]
        best_step = None if best_step_raw is None else int(cast(int, best_step_raw))
        state_raw = payload["best_state"]
        best_state = (
            None
            if state_raw is None
            else {
                name: value for name, value in cast(Mapping[str, torch.Tensor], state_raw).items()
            }
        )
        non_improving = int(cast(int, payload["non_improving"]))
    except (KeyError, TypeError, RuntimeError, ValueError) as error:
        raise TextTrainingError("resumable classifier checkpoint state is malformed") from error
    if not 0 <= step <= config.maximum_steps:
        raise TextTrainingError("resumable classifier step lies outside the declared budget")
    progress = payload.get("data_progress")
    if (
        payload.get("next_global_step") != step + 1
        or payload.get("pilot_complete") != (step >= config.pilot_step)
        or not isinstance(progress, Mapping)
        or progress.get("completed_optimizer_steps") != step
        or progress.get("completed_epochs") != step // config.steps_per_epoch
        or progress.get("next_batch_index") != step % config.steps_per_epoch
        or progress.get("steps_per_epoch") != config.steps_per_epoch
        or progress.get("reconstruction") != "seeded-permutation-v0-modulo-step"
    ):
        raise TextTrainingError("resumable classifier data progression differs")
    if len(training_records) != step or any(
        record.step != expected for expected, record in enumerate(training_records, start=1)
    ):
        raise TextTrainingError("resumable classifier training telemetry is incomplete")
    return (
        step,
        records,
        training_records,
        best_loss,
        best_step,
        best_state,
        non_improving,
    )


def factorized_validation_fixture_payload(
    outputs: FactorizedValidationOutputs,
) -> dict[str, object]:
    """Serialize one small validation fixture for cross-process reload comparison."""

    if outputs.evidence_split != "validation" or outputs.status_labels.numel() <= 0:
        raise TextTrainingError("reload fixture must contain validation examples")
    payload: dict[str, object] = {
        "schema_version": "langmani-m5a-validation-logit-fixture-v0",
        "validation_examples": int(outputs.status_labels.shape[0]),
        "status_logits": outputs.status_logits.detach().cpu().tolist(),
        "object_logits": outputs.object_logits.detach().cpu().tolist(),
        "bin_logits": outputs.bin_logits.detach().cpu().tolist(),
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 1e-6,
    }
    payload["fixture_fingerprint"] = f"sha256:{sha256_hex(payload)}"
    return payload


def _fixture_tensor(
    payload: Mapping[str, object],
    *,
    key: str,
    expected_shape: tuple[int, int],
) -> torch.Tensor:
    try:
        value = torch.tensor(payload[key], dtype=torch.float32)
    except (KeyError, TypeError, ValueError) as error:
        raise TextTrainingError(f"reload fixture {key} is malformed") from error
    if tuple(value.shape) != expected_shape or not bool(torch.isfinite(value).all()):
        raise TextTrainingError(f"reload fixture {key} shape or values differ")
    return value


def audit_staged_factorized_text_checkpoint(
    *,
    model: FactorizedTextClassifierV0,
    validation_batches: Sequence[FactorizedTextBatch],
    config: ClassifierStageTrainingConfig,
    run_fingerprint: str,
    checkpoint_path: Path,
    processor_state: Mapping[str, object],
    expected_fixture: Mapping[str, object],
    expected_step: int,
) -> StagedTextCheckpointAudit:
    """Restore model/optimizer/scheduler/RNG and compare held-out logits without stepping."""

    validation = tuple(validation_batches)
    if not validation or any(batch.split != "validation" for batch in validation):
        raise TextTrainingError("checkpoint audit requires validation-only batches")
    if expected_step != config.pilot_step:
        raise TextTrainingError("checkpoint audit step differs from the declared pilot boundary")
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    (
        step,
        validation_records,
        training_records,
        _best_loss,
        _best_step,
        _best_state,
        _non_improving,
    ) = _load_staged_checkpoint(
        path=checkpoint_path,
        run_fingerprint=run_fingerprint,
        config=config,
        processor_state=processor_state,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    if step != expected_step:
        raise TextTrainingError("fresh checkpoint reload restored the wrong optimizer step")
    if scheduler.last_epoch != step:
        raise TextTrainingError("fresh checkpoint reload restored the wrong scheduler step")
    actual = collect_factorized_validation_outputs(model=model, batches=(validation[0],))
    example_count = int(actual.status_labels.shape[0])
    expected_count = expected_fixture.get("validation_examples")
    if expected_count != example_count:
        raise TextTrainingError("reload validation fixture example count differs")
    expected_fingerprint = expected_fixture.get("fixture_fingerprint")
    content = {
        key: value for key, value in expected_fixture.items() if key != "fixture_fingerprint"
    }
    if expected_fingerprint != f"sha256:{sha256_hex(content)}":
        raise TextTrainingError("reload validation fixture fingerprint differs")
    absolute_tolerance = float(cast(float, expected_fixture.get("absolute_tolerance")))
    relative_tolerance = float(cast(float, expected_fixture.get("relative_tolerance")))
    pairs = (
        (
            actual.status_logits.detach().cpu(),
            _fixture_tensor(
                expected_fixture,
                key="status_logits",
                expected_shape=tuple(actual.status_logits.shape),
            ),
        ),
        (
            actual.object_logits.detach().cpu(),
            _fixture_tensor(
                expected_fixture,
                key="object_logits",
                expected_shape=tuple(actual.object_logits.shape),
            ),
        ),
        (
            actual.bin_logits.detach().cpu(),
            _fixture_tensor(
                expected_fixture,
                key="bin_logits",
                expected_shape=tuple(actual.bin_logits.shape),
            ),
        ),
    )
    maximum_error = max(float(torch.max(torch.abs(left - right))) for left, right in pairs)
    logits_match = all(
        torch.allclose(left, right, atol=absolute_tolerance, rtol=relative_tolerance)
        for left, right in pairs
    )
    if not logits_match:
        raise TextTrainingError("fresh checkpoint reload changed deterministic validation logits")
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TextTrainingError("checkpoint audit payload must be a mapping")
    optimizer_state = optimizer.state_dict()
    return StagedTextCheckpointAudit(
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=f"sha256:{digest.hexdigest()}",
        checkpoint_size_bytes=checkpoint_path.stat().st_size,
        restored_step=step,
        next_global_step=step + 1,
        next_learning_rate=float(optimizer.param_groups[0]["lr"]),
        optimizer_state_restored=bool(optimizer_state.get("state")),
        scheduler_state_restored=scheduler.last_epoch == step,
        processor_state_restored=payload.get("processor_state") == dict(processor_state),
        rng_state_restored=isinstance(payload.get("rng_state"), Mapping),
        data_progress_restored=isinstance(payload.get("data_progress"), Mapping),
        deterministic_logits_match=logits_match,
        maximum_absolute_logit_error=maximum_error,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        validation_fixture_examples=example_count,
        validation_records_restored=len(validation_records),
        training_records_restored=len(training_records),
        pilot_complete=payload.get("pilot_complete") is True,
        full_training_complete=payload.get("full_training_complete") is True,
        resumable=payload.get("resumable") is True,
    )


def run_staged_factorized_text_training(
    *,
    model: FactorizedTextClassifierV0,
    train_batches: Sequence[FactorizedTextBatch],
    validation_batches: Sequence[FactorizedTextBatch],
    config: ClassifierStageTrainingConfig,
    run_fingerprint: str,
    checkpoint_root: Path,
    processor_state: Mapping[str, object],
    stop_after_pilot: bool,
    resume: bool,
) -> StagedTextTrainingResult:
    """Pause once at pilot, then resume the same optimizer/scheduler/RNG state."""

    training = tuple(train_batches)
    validation = tuple(validation_batches)
    if not training or any(value.split != "train" for value in training):
        raise TextTrainingError("staged classifier gradients require train-only batches")
    if not validation or any(value.split != "validation" for value in validation):
        raise TextTrainingError("staged classifier selection requires validation-only batches")
    digest = run_fingerprint.removeprefix("sha256:")
    if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
        raise TextTrainingError("staged classifier run requires a full SHA-256 identity")
    root = _resolved_unlinked(checkpoint_root, label="staged classifier checkpoint root")
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or root.is_junction():
        raise TextTrainingError("staged classifier checkpoint root cannot be linked")
    paths = {role: root / f"{role}.pt" for role in CheckpointRetentionPolicy().retained_roles}
    if not resume and any(path.exists() for path in paths.values()):
        raise TextTrainingError("authoritative run already has checkpoints; use explicit resume")
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    try:
        device = next(model.parameters()).device
    except StopIteration as error:
        raise TextTrainingError("classifier has no parameters") from error
    records: list[TextCheckpointValidation] = []
    training_records: list[TextTrainingTelemetry] = []
    best_loss = math.inf
    best_step: int | None = None
    best_state: dict[str, torch.Tensor] | None = None
    non_improving = 0
    resumed_from_step = 0
    if resume:
        (
            resumed_from_step,
            records,
            training_records,
            best_loss,
            best_step,
            best_state,
            non_improving,
        ) = _load_staged_checkpoint(
            path=paths["latest"],
            run_fingerprint=run_fingerprint,
            config=config,
            processor_state=processor_state,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if resumed_from_step < config.pilot_step:
            raise TextTrainingError("full training cannot resume before the completed pilot")
    else:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
    validation_steps = set(config.validation_steps)
    stopped_early = False
    completed_step = resumed_from_step
    pilot_metrics: ClassifierStageMetrics | None = None
    final_metrics: ClassifierStageMetrics | None = None
    examples_processed = sum(value.batch_examples for value in training_records)
    for step in range(resumed_from_step + 1, config.maximum_steps + 1):
        original_batch = training[(step - 1) % len(training)]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_started = time.perf_counter()
        loader_started = step_started
        batch = _batch_on_device(original_batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        loader_finished = time.perf_counter()
        train_result = train_factorized_text_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            maximum_gradient_norm=float(config.maximum_gradient_norm),
        )
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_finished = time.perf_counter()
        batch_examples = int(original_batch.status_labels.shape[0])
        examples_processed += batch_examples
        step_latency = step_finished - step_started
        telemetry = TextTrainingTelemetry(
            step=step,
            total_loss=train_result.total_loss,
            status_loss=train_result.status_loss,
            object_loss=train_result.object_loss,
            bin_loss=train_result.bin_loss,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            gradient_norm=train_result.gradient_norm,
            encoder_gradient_norm=train_result.encoder_gradient_norm,
            status_head_gradient_norm=train_result.status_head_gradient_norm,
            object_head_gradient_norm=train_result.object_head_gradient_norm,
            bin_head_gradient_norm=train_result.bin_head_gradient_norm,
            routed_examples=train_result.routed_examples,
            batch_examples=batch_examples,
            examples_processed=examples_processed,
            epoch_progress=step / config.steps_per_epoch,
            throughput_examples_per_second=batch_examples / max(step_latency, 1e-12),
            data_loader_latency_seconds=loader_finished - loader_started,
            step_latency_seconds=step_latency,
            gpu_allocated_bytes=(
                int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
            ),
            gpu_reserved_bytes=(
                int(torch.cuda.memory_reserved(device)) if device.type == "cuda" else 0
            ),
        )
        finite_values = (
            telemetry.total_loss,
            telemetry.status_loss,
            telemetry.object_loss,
            telemetry.bin_loss,
            telemetry.learning_rate,
            telemetry.gradient_norm,
            telemetry.encoder_gradient_norm,
            telemetry.status_head_gradient_norm,
            telemetry.object_head_gradient_norm,
            telemetry.bin_head_gradient_norm,
            telemetry.epoch_progress,
            telemetry.throughput_examples_per_second,
            telemetry.data_loader_latency_seconds,
            telemetry.step_latency_seconds,
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise TextTrainingError("staged classifier telemetry is non-finite")
        training_records.append(telemetry)
        completed_step = step
        if step not in validation_steps:
            continue
        outputs = collect_factorized_validation_outputs(model=model, batches=validation)
        metrics = classifier_stage_metrics(outputs)
        record = _validation_record(outputs=outputs, step=step)
        records.append(record)
        improved = record.total_loss < best_loss - 1e-12
        if improved:
            best_loss = record.total_loss
            best_step = step
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            non_improving = 0
        else:
            non_improving += 1
        _save_staged_checkpoint(
            path=paths["latest"],
            role="latest",
            run_fingerprint=run_fingerprint,
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            processor_state=processor_state,
            step=step,
            records=records,
            training_records=training_records,
            best_loss=best_loss,
            best_step=best_step,
            best_state=best_state,
            non_improving=non_improving,
        )
        if improved:
            _save_staged_checkpoint(
                path=paths["validation_best"],
                role="validation_best",
                run_fingerprint=run_fingerprint,
                config=config,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                processor_state=processor_state,
                step=step,
                records=records,
                training_records=training_records,
                best_loss=best_loss,
                best_step=best_step,
                best_state=best_state,
                non_improving=non_improving,
            )
        if step == config.pilot_step:
            pilot_metrics = metrics
            _save_staged_checkpoint(
                path=paths["pilot"],
                role="pilot",
                run_fingerprint=run_fingerprint,
                config=config,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                processor_state=processor_state,
                step=step,
                records=records,
                training_records=training_records,
                best_loss=best_loss,
                best_step=best_step,
                best_state=best_state,
                non_improving=non_improving,
            )
            if stop_after_pilot:
                return StagedTextTrainingResult(
                    run_fingerprint=run_fingerprint,
                    phase="pilot_paused",
                    completed_steps=step,
                    completed_epochs=step // config.steps_per_epoch,
                    resumed_from_step=0,
                    pilot_checkpoint=str(paths["pilot"]),
                    latest_checkpoint=str(paths["latest"]),
                    best_checkpoint=str(paths["validation_best"]),
                    selected_step=best_step,
                    stopped_early=False,
                    pilot_metrics=pilot_metrics,
                    final_metrics=None,
                    validation_records=tuple(records),
                    training_records=tuple(training_records),
                    pilot_outputs=outputs,
                )
        final_metrics = metrics
        if non_improving >= config.early_stopping_patience:
            stopped_early = True
            break
    if stop_after_pilot:
        raise TextTrainingError("pilot stage did not stop at its declared checkpoint")
    if best_state is None or best_step is None or final_metrics is None:
        raise TextTrainingError("resumed classifier training produced no selected validation state")
    model.load_state_dict(best_state, strict=True)
    selected_outputs = collect_factorized_validation_outputs(model=model, batches=validation)
    final_metrics = classifier_stage_metrics(selected_outputs)
    return StagedTextTrainingResult(
        run_fingerprint=run_fingerprint,
        phase="training_complete",
        completed_steps=completed_step,
        completed_epochs=math.ceil(completed_step / config.steps_per_epoch),
        resumed_from_step=resumed_from_step,
        pilot_checkpoint=str(paths["pilot"]),
        latest_checkpoint=str(paths["latest"]),
        best_checkpoint=str(paths["validation_best"]),
        selected_step=best_step,
        stopped_early=stopped_early,
        pilot_metrics=None,
        final_metrics=final_metrics,
        validation_records=tuple(records),
        training_records=tuple(training_records),
        pilot_outputs=None,
    )


def calibrate_and_select_text_router(
    *,
    outputs: FactorizedValidationOutputs,
    threshold_candidates: Sequence[float],
    maximum_false_route_rate: float = 0.03,
    temperature_iterations: int = 64,
) -> TextRouterCalibrationSelection:
    """Fit temperature and the selective threshold from validation only."""

    if outputs.evidence_split != "validation":
        raise TextTrainingError("calibration and threshold selection require validation only")
    calibration = fit_validation_temperature(
        outputs.status_logits,
        outputs.status_labels,
        evidence_split="validation",
        iterations=temperature_iterations,
    )
    confidence = calibrated_full_route_confidence(
        status_logits=outputs.status_logits,
        object_logits=outputs.object_logits,
        bin_logits=outputs.bin_logits,
        temperature=calibration,
    )
    predicted_status = calibration.apply(outputs.status_logits).argmax(dim=-1)
    predicted_object = outputs.object_logits.argmax(dim=-1)
    predicted_bin = outputs.bin_logits.argmax(dim=-1)
    examples: list[RoutingThresholdExample] = []
    for index in range(outputs.status_labels.shape[0]):
        expected_status_index = int(outputs.status_labels[index])
        expected_route = expected_status_index == ROUTE_STATUS_INDEX
        expected_object_index = int(outputs.object_labels[index])
        expected_bin_index = int(outputs.bin_labels[index])
        if expected_route and not (0 <= expected_object_index < 3 and 0 <= expected_bin_index < 2):
            raise TextTrainingError("routeable validation labels must contain a valid TaskSpec")
        status_index = int(predicted_status[index])
        object_index = int(predicted_object[index])
        bin_index = int(predicted_bin[index])
        examples.append(
            RoutingThresholdExample(
                expected_route=expected_route,
                expected_object_id=(
                    OBJECT_LABELS[expected_object_index] if expected_route else None
                ),
                expected_bin_id=BIN_LABELS[expected_bin_index] if expected_route else None,
                predicted_status=STATUS_LABELS[status_index],
                predicted_object_id=(
                    OBJECT_LABELS[object_index] if OBJECT_LABELS[object_index] != "none" else None
                ),
                predicted_bin_id=(
                    BIN_LABELS[bin_index] if BIN_LABELS[bin_index] != "none" else None
                ),
                full_route_confidence=float(confidence[index]),
            )
        )
    threshold = select_validation_routing_threshold(
        examples,
        evidence_split="validation",
        candidates=threshold_candidates,
        maximum_false_route_rate=maximum_false_route_rate,
    )
    return TextRouterCalibrationSelection(temperature=calibration, threshold=threshold)


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _resolved_unlinked(path: Path, *, label: str) -> Path:
    lexical = _lexical_absolute(path)
    for component in (lexical, *lexical.parents):
        if component.is_symlink() or component.is_junction():
            raise TextTrainingError(f"{label} traverses a symlink or junction: {component}")
    return lexical.resolve(strict=False)


def _write_new_json(path: Path, payload: object) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise TextTrainingError(f"immutable classifier artifact already exists: {path}") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _artifact_records(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for candidate in root.rglob("*"):
        if candidate.is_symlink() or candidate.is_junction():
            raise TextTrainingError(f"classifier artifact contains a linked entry: {candidate}")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise TextTrainingError(f"classifier artifact contains a special entry: {candidate}")
        relative = PurePosixPath(candidate.relative_to(root)).as_posix()
        if relative in {"artifact_manifest.json", "complete.json"}:
            continue
        records.append(
            {
                "path": relative,
                "size_bytes": candidate.stat().st_size,
                "sha256": _sha256_file(candidate),
            }
        )
    return sorted(records, key=lambda value: cast(str, value["path"]))


def validate_completed_text_classifier_artifact(
    root: Path,
    expected_owner: Mapping[str, object] | None = None,
    expected_run_evidence: Mapping[str, object] | None = None,
    *,
    require_promoted: bool = True,
) -> Path:
    """Rehash and validate one immutable classifier artifact without loading weights.

    ``expected_owner`` and ``expected_run_evidence`` are optional so an independent
    verifier can validate an already promoted artifact directly from disk.  The
    producer supplies both mappings while staging, with ``require_promoted=False``,
    and therefore exercises the same checksum contract before and after promotion.
    """

    root = _resolved_unlinked(root, label="completed classifier artifact")
    if not root.is_dir() or root.is_symlink() or root.is_junction():
        raise TextTrainingError("completed classifier artifact must be one real directory")
    for name in (
        "owner.json",
        "run_evidence.json",
        "artifact_manifest.json",
        "complete.json",
    ):
        path = root / name
        if not path.is_file() or path.is_symlink() or path.is_junction():
            raise TextTrainingError(f"completed classifier artifact is missing safe {name}")
    try:
        owner = json.loads((root / "owner.json").read_text(encoding="utf-8"))
        run_evidence = json.loads((root / "run_evidence.json").read_text(encoding="utf-8"))
        manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
        complete = json.loads((root / "complete.json").read_text(encoding="utf-8"))
        classifier_config = json.loads(
            (root / "classifier_config.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TextTrainingError("could not parse completed classifier metadata") from error
    if not isinstance(owner, dict) or not isinstance(run_evidence, dict):
        raise TextTrainingError("completed classifier owner/run evidence must be objects")
    if expected_owner is not None and owner != dict(expected_owner):
        raise TextTrainingError("completed classifier artifact owner differs")
    if expected_run_evidence is not None and run_evidence != dict(expected_run_evidence):
        raise TextTrainingError("completed classifier run evidence differs")
    run_fingerprint = owner.get("run_fingerprint")
    if (
        not isinstance(run_fingerprint, str)
        or not run_fingerprint.startswith("sha256:")
        or len(run_fingerprint) != 71
    ):
        raise TextTrainingError("completed classifier run fingerprint is malformed")
    if require_promoted and root.name != run_fingerprint.removeprefix("sha256:"):
        raise TextTrainingError("classifier artifact is not at its promoted run path")
    if not isinstance(classifier_config, dict):
        raise TextTrainingError("completed classifier config must be one object")
    expected_owner_from_disk = {
        "schema_version": TEXT_TRAINING_ARTIFACT_SCHEMA,
        "run_fingerprint": text_classifier_run_fingerprint(
            classifier_manifest=classifier_config,
            run_evidence=run_evidence,
        ),
        "classifier_manifest_fingerprint": f"sha256:{sha256_hex(classifier_config)}",
        "run_evidence_fingerprint": f"sha256:{sha256_hex(run_evidence)}",
    }
    if owner != expected_owner_from_disk:
        raise TextTrainingError("completed classifier owner fingerprints differ from disk")
    if (
        not isinstance(manifest, dict)
        or not isinstance(complete, dict)
        or complete.get("passed") is not True
    ):
        raise TextTrainingError("classifier artifact manifest/completion is malformed")
    actual = _artifact_records(root)
    if manifest.get("artifacts") != actual:
        raise TextTrainingError("classifier artifact checksum manifest differs")
    expected_fingerprint = f"sha256:{sha256_hex({'owner': owner, 'artifacts': actual})}"
    if manifest != {
        "schema_version": TEXT_TRAINING_ARTIFACT_SCHEMA,
        "artifact_fingerprint": expected_fingerprint,
        "artifacts": actual,
    }:
        raise TextTrainingError("classifier artifact fingerprint differs")
    if complete != {
        "schema_version": TEXT_TRAINING_ARTIFACT_SCHEMA,
        "artifact_fingerprint": expected_fingerprint,
        "passed": True,
    }:
        raise TextTrainingError("classifier completion fingerprint differs")
    return root


def _validate_completed_artifact(
    root: Path,
    expected_owner: Mapping[str, object],
    expected_run_evidence: Mapping[str, object],
) -> Path:
    """Producer compatibility wrapper for validation before atomic promotion."""

    return validate_completed_text_classifier_artifact(
        root,
        expected_owner,
        expected_run_evidence,
        require_promoted=False,
    )


def text_classifier_run_fingerprint(
    *,
    classifier_manifest: Mapping[str, object],
    run_evidence: Mapping[str, object],
) -> str:
    """Bind model architecture and all declared training/runtime evidence."""

    return f"sha256:{sha256_hex({'schema_version': TEXT_TRAINING_ARTIFACT_SCHEMA, 'classifier_manifest': dict(classifier_manifest), 'run_evidence': dict(run_evidence)})}"


def stage_and_promote_text_classifier(
    *,
    output_root: str | Path,
    run_fingerprint: str,
    model: FactorizedTextClassifierV0,
    tokenizer: object,
    classifier_manifest: Mapping[str, object],
    run_evidence: Mapping[str, object] | None = None,
    clean_matching_staging: bool = False,
) -> Path:
    """Save through owned staging, checksum, validate, and atomically promote."""

    if not run_fingerprint.startswith("sha256:") or len(run_fingerprint) != 71:
        raise TextTrainingError("run_fingerprint must be a prefixed full SHA-256")
    root = _resolved_unlinked(Path(output_root), label="text classifier output root")
    if root.exists() and (not root.is_dir() or root.is_symlink() or root.is_junction()):
        raise TextTrainingError("text classifier output root must be a real directory")
    root.mkdir(parents=True, exist_ok=True)
    token = run_fingerprint.removeprefix("sha256:")
    destination = _resolved_unlinked(root / token, label="text classifier destination")
    staging = _resolved_unlinked(root / f".staging-{token}", label="text classifier staging")
    if destination.parent != root or staging.parent != root:
        raise TextTrainingError("classifier artifact escaped its owned output root")
    evidence = {} if run_evidence is None else dict(run_evidence)
    expected_run_fingerprint = text_classifier_run_fingerprint(
        classifier_manifest=classifier_manifest,
        run_evidence=evidence,
    )
    if run_fingerprint != expected_run_fingerprint:
        raise TextTrainingError(
            "run_fingerprint does not bind classifier manifest and run evidence"
        )
    owner = {
        "schema_version": TEXT_TRAINING_ARTIFACT_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "classifier_manifest_fingerprint": f"sha256:{sha256_hex(dict(classifier_manifest))}",
        "run_evidence_fingerprint": f"sha256:{sha256_hex(evidence)}",
    }
    if destination.exists():
        return validate_completed_text_classifier_artifact(
            destination,
            owner,
            evidence,
        )
    if staging.exists():
        if not staging.is_dir() or staging.is_symlink() or staging.is_junction():
            raise TextTrainingError("classifier staging is unsafe")
        owner_path = staging / "owner.json"
        existing_owner = (
            json.loads(owner_path.read_text(encoding="utf-8")) if owner_path.is_file() else None
        )
        if existing_owner != owner:
            raise TextTrainingError("refusing to clean classifier staging owned by another run")
        if not clean_matching_staging:
            raise TextTrainingError("matching staging exists; explicit cleanup is required")
        shutil.rmtree(staging)
    staging.mkdir(exist_ok=False)
    _write_new_json(staging / "owner.json", owner)
    _write_new_json(staging / "classifier_config.json", dict(classifier_manifest))
    _write_new_json(staging / "run_evidence.json", evidence)
    encoder_root = staging / "encoder"
    save_encoder = getattr(model.encoder, "save_pretrained", None)
    save_tokenizer = getattr(tokenizer, "save_pretrained", None)
    if not callable(save_encoder) or not callable(save_tokenizer):
        raise TextTrainingError(
            "real classifier artifacts require public encoder/tokenizer serializers"
        )
    save_encoder(encoder_root, push_to_hub=False)
    save_tokenizer(staging / "tokenizer", push_to_hub=False)
    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise TextTrainingError("safetensors is required for classifier heads") from error
    save_file(model.heads_state_dict(), staging / "router_heads.safetensors")
    records = _artifact_records(staging)
    artifact_fingerprint = f"sha256:{sha256_hex({'owner': owner, 'artifacts': records})}"
    _write_new_json(
        staging / "artifact_manifest.json",
        {
            "schema_version": TEXT_TRAINING_ARTIFACT_SCHEMA,
            "artifact_fingerprint": artifact_fingerprint,
            "artifacts": records,
        },
    )
    _write_new_json(
        staging / "complete.json",
        {
            "schema_version": TEXT_TRAINING_ARTIFACT_SCHEMA,
            "artifact_fingerprint": artifact_fingerprint,
            "passed": True,
        },
    )
    _validate_completed_artifact(staging, owner, evidence)
    if staging.stat().st_dev != root.stat().st_dev:
        raise TextTrainingError("classifier staging and destination are on different filesystems")
    try:
        os.replace(staging, destination)
    except OSError as error:
        raise TextTrainingError(f"atomic classifier artifact promotion failed: {error}") from error
    return validate_completed_text_classifier_artifact(
        destination,
        owner,
        evidence,
    )


def read_text_classifier_run_evidence(artifact_root: str | Path) -> dict[str, object]:
    """Read content-bound run evidence from an already completed artifact."""

    root = _resolved_unlinked(Path(artifact_root), label="text classifier artifact root")
    owner_path = root / "owner.json"
    evidence_path = root / "run_evidence.json"
    complete_path = root / "complete.json"
    if not all(
        path.is_file() and not path.is_symlink()
        for path in (owner_path, evidence_path, complete_path)
    ):
        raise TextTrainingError("classifier run evidence requires a completed real artifact")
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if not isinstance(owner, dict) or not isinstance(evidence, dict):
        raise TextTrainingError("classifier owner/run evidence must be JSON objects")
    if not isinstance(complete, dict) or complete.get("passed") is not True:
        raise TextTrainingError("classifier artifact is not complete")
    if owner.get("run_evidence_fingerprint") != f"sha256:{sha256_hex(evidence)}":
        raise TextTrainingError("classifier run evidence fingerprint differs")
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    actual = _artifact_records(root)
    artifact_fingerprint = f"sha256:{sha256_hex({'owner': owner, 'artifacts': actual})}"
    if not isinstance(manifest, dict) or manifest.get("artifacts") != actual:
        raise TextTrainingError("classifier artifact checksum manifest differs")
    if manifest.get("artifact_fingerprint") != artifact_fingerprint:
        raise TextTrainingError("classifier artifact fingerprint differs")
    if complete.get("artifact_fingerprint") != artifact_fingerprint:
        raise TextTrainingError("classifier completion fingerprint differs")
    return cast(dict[str, object], evidence)


__all__ = [
    "BoundedTextTrainingResult",
    "FactorizedTextBatch",
    "FactorizedValidationOutputs",
    "TextCheckpointValidation",
    "TextRouterCalibrationSelection",
    "TextTrainingConfig",
    "TextTrainingError",
    "TextTrainingStepResult",
    "StagedTextTrainingResult",
    "StagedTextCheckpointAudit",
    "TextTrainingTelemetry",
    "audit_staged_factorized_text_checkpoint",
    "calibrate_and_select_text_router",
    "classifier_stage_metrics",
    "collect_factorized_validation_outputs",
    "collect_factorized_tiny_train_outputs",
    "factorized_validation_fixture_payload",
    "read_text_classifier_run_evidence",
    "run_bounded_factorized_text_training",
    "run_staged_factorized_text_training",
    "select_validation_checkpoint",
    "stage_and_promote_text_classifier",
    "text_classifier_run_fingerprint",
    "train_factorized_text_step",
    "validate_completed_text_classifier_artifact",
]
