"""Bounded training steps and immutable classifier artifact promotion for M5A."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

import torch
from torch import nn

from langmani.datasets.identity import sha256_hex
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
    loss.total.backward()
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
    with torch.inference_mode():
        for original in values:
            batch = _batch_on_device(original, device)
            output = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
            )
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
    "calibrate_and_select_text_router",
    "collect_factorized_validation_outputs",
    "read_text_classifier_run_evidence",
    "run_bounded_factorized_text_training",
    "select_validation_checkpoint",
    "stage_and_promote_text_classifier",
    "text_classifier_run_fingerprint",
    "train_factorized_text_step",
    "validate_completed_text_classifier_artifact",
]
