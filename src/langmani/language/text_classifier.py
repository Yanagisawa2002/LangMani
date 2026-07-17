"""Factorized DistilBERT-compatible text router for M5A.

Transformers is imported only at explicit construction or reload boundaries so
that the project-owned contracts and generated fixtures remain lightweight.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch
from torch import nn
from torch.nn import functional as F

from langmani.datasets.identity import sha256_hex
from langmani.environments.specs import TaskSpec
from langmani.language.router_types import (
    RouterConfidence,
    RouterDecision,
    RouterRejectionReason,
    RouterStatus,
)
from langmani.language.text_calibration import (
    TemperatureCalibrationV0,
    calibrated_full_route_confidence,
)

STATUS_LABELS = ("route", "reject_ambiguous", "reject_unsupported", "reject_malformed")
OBJECT_LABELS = ("red_cube", "green_cube", "blue_cube", "none")
BIN_LABELS = ("left_bin", "right_bin", "none")
ROUTE_STATUS_INDEX = 0
TEXT_CLASSIFIER_ARTIFACT_SCHEMA = "langmani-m5a-text-classifier-v1"
TEXT_CLASSIFIER_ROUTER_NAME = "FactorizedTextClassifierV0"
TEXT_CLASSIFIER_ROUTER_VERSION = "factorized-text-classifier-router-v0"
TEXT_CLASSIFIER_MODEL_ID = "distilbert/distilbert-base-uncased"
TEXT_CLASSIFIER_MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
TEXT_CLASSIFIER_TOKENIZER_REVISION = TEXT_CLASSIFIER_MODEL_REVISION


class TextClassifierError(RuntimeError):
    """Raised when the factorized classifier contract is violated."""


class _EncoderOutput(Protocol):
    last_hidden_state: torch.Tensor


class TextTokenizer(Protocol):
    """The minimal public tokenizer surface used by classifier inference."""

    def __call__(
        self,
        text: str,
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class FactorizedTextClassifierOutput:
    status_logits: torch.Tensor
    object_logits: torch.Tensor
    bin_logits: torch.Tensor


@dataclass(frozen=True, slots=True)
class FactorizedTextLoss:
    total: torch.Tensor
    status: torch.Tensor
    target_object: torch.Tensor
    target_bin: torch.Tensor
    routed_examples: int


@dataclass(frozen=True, slots=True)
class FactorizedTextRouterConfig:
    """Frozen inference settings selected before development evaluation."""

    maximum_sequence_length: int
    routing_threshold: float
    device: str = "cpu"

    def __post_init__(self) -> None:
        if (
            isinstance(self.maximum_sequence_length, bool)
            or not isinstance(self.maximum_sequence_length, int)
            or not 1 <= self.maximum_sequence_length <= 512
        ):
            raise ValueError("maximum_sequence_length must lie in [1,512]")
        if (
            isinstance(self.routing_threshold, bool)
            or not isinstance(self.routing_threshold, int | float)
            or not math.isfinite(float(self.routing_threshold))
            or not 0.0 <= float(self.routing_threshold) <= 1.0
        ):
            raise ValueError("routing_threshold must be finite and lie in [0,1]")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        object.__setattr__(self, "routing_threshold", float(self.routing_threshold))

    def to_dict(self) -> dict[str, object]:
        return {
            "maximum_sequence_length": self.maximum_sequence_length,
            "routing_threshold": self.routing_threshold,
            "device": self.device,
        }


def _encoder_hidden_size(encoder: nn.Module) -> int:
    config = getattr(encoder, "config", None)
    for name in ("dim", "hidden_size", "d_model"):
        value = getattr(config, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    raise TextClassifierError("encoder config does not expose a positive hidden dimension")


class FactorizedTextClassifierV0(nn.Module):
    """One shared encoder with status, object, and destination heads."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        hidden_size: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not isinstance(encoder, nn.Module):
            raise TypeError("encoder must be a torch.nn.Module")
        resolved_hidden = _encoder_hidden_size(encoder) if hidden_size is None else hidden_size
        if isinstance(resolved_hidden, bool) or not isinstance(resolved_hidden, int):
            raise TypeError("hidden_size must be an integer")
        if resolved_hidden <= 0:
            raise ValueError("hidden_size must be positive")
        if isinstance(dropout, bool) or not isinstance(dropout, int | float):
            raise TypeError("dropout must be numeric")
        if not math.isfinite(float(dropout)) or not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be finite and lie in [0,1)")
        self.encoder = encoder
        self.hidden_size = resolved_hidden
        self.dropout_probability = float(dropout)
        self.dropout = nn.Dropout(self.dropout_probability)
        self.status_head = nn.Linear(resolved_hidden, len(STATUS_LABELS))
        self.object_head = nn.Linear(resolved_hidden, len(OBJECT_LABELS))
        self.bin_head = nn.Linear(resolved_hidden, len(BIN_LABELS))

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> FactorizedTextClassifierOutput:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise TextClassifierError(
                "input_ids and attention_mask must have equal [batch,seq] shape"
            )
        encoded = cast(
            _EncoderOutput, self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        )
        hidden = encoded.last_hidden_state
        if hidden.ndim != 3 or hidden.shape[:2] != input_ids.shape:
            raise TextClassifierError(
                "encoder last_hidden_state must have [batch,seq,hidden] shape"
            )
        if hidden.shape[-1] != self.hidden_size:
            raise TextClassifierError("encoder hidden dimension changed after construction")
        pooled = self.dropout(hidden[:, 0, :])
        return FactorizedTextClassifierOutput(
            status_logits=self.status_head(pooled),
            object_logits=self.object_head(pooled),
            bin_logits=self.bin_head(pooled),
        )

    @classmethod
    def from_pretrained_encoder(
        cls,
        *,
        model_id: str,
        revision: str,
        dropout: float = 0.1,
        local_files_only: bool = False,
    ) -> FactorizedTextClassifierV0:
        if not model_id or not revision:
            raise ValueError("model_id and exact revision are required")
        try:
            from transformers import AutoModel
        except ImportError as error:
            raise TextClassifierError(
                "installed Transformers is required for the real encoder"
            ) from error
        encoder = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        if not isinstance(encoder, nn.Module):
            raise TextClassifierError("AutoModel did not return a torch module")
        return cls(encoder, dropout=dropout)

    def heads_state_dict(self) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for prefix, layer in (
            ("status_head", self.status_head),
            ("object_head", self.object_head),
            ("bin_head", self.bin_head),
        ):
            for name, value in layer.state_dict().items():
                state[f"{prefix}.{name}"] = value.detach().cpu().contiguous()
        return state

    def load_heads_state_dict(self, value: dict[str, torch.Tensor]) -> None:
        expected = set(self.heads_state_dict())
        if set(value) != expected:
            raise TextClassifierError("classifier head state fields differ from the exact schema")
        for prefix, layer in (
            ("status_head", self.status_head),
            ("object_head", self.object_head),
            ("bin_head", self.bin_head),
        ):
            layer.load_state_dict(
                {
                    key.removeprefix(prefix + "."): tensor
                    for key, tensor in value.items()
                    if key.startswith(prefix + ".")
                },
                strict=True,
            )


_CLASSIFIER_REJECTION_BY_STATUS = {
    RouterStatus.REJECT_AMBIGUOUS: RouterRejectionReason.CLASSIFIER_AMBIGUOUS,
    RouterStatus.REJECT_UNSUPPORTED: RouterRejectionReason.CLASSIFIER_UNSUPPORTED,
    RouterStatus.REJECT_MALFORMED: RouterRejectionReason.CLASSIFIER_MALFORMED,
}


class FactorizedTextRouterV0:
    """Turn tokenizer/model outputs into one validated selective RouterDecision."""

    def __init__(
        self,
        *,
        model: FactorizedTextClassifierV0,
        tokenizer: TextTokenizer,
        calibration: TemperatureCalibrationV0,
        config: FactorizedTextRouterConfig,
    ) -> None:
        if not isinstance(model, FactorizedTextClassifierV0):
            raise TypeError("model must be FactorizedTextClassifierV0")
        if not isinstance(calibration, TemperatureCalibrationV0):
            raise TypeError("calibration must be TemperatureCalibrationV0")
        if not isinstance(config, FactorizedTextRouterConfig):
            raise TypeError("config must be FactorizedTextRouterConfig")
        if not callable(tokenizer):
            raise TypeError("tokenizer must be callable")
        if config.device == "cuda" and not torch.cuda.is_available():
            raise TextClassifierError("CUDA classifier routing was requested but unavailable")
        self.model = model.to(config.device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.calibration = calibration
        self.config = config

    def route(self, command: str) -> RouterDecision:
        if not isinstance(command, str):
            raise TypeError("command must be text")
        if not command.strip():
            return RouterDecision.reject(
                status=RouterStatus.REJECT_MALFORMED,
                rejection_reason=RouterRejectionReason.EMPTY_TEXT,
                confidence=RouterConfidence.unavailable(
                    definition="classifier inference is not run for empty text"
                ),
                router_name=TEXT_CLASSIFIER_ROUTER_NAME,
                router_version=TEXT_CLASSIFIER_ROUTER_VERSION,
                evidence={"precheck": "empty_text"},
            )
        encoded = self.tokenizer(
            command,
            padding=False,
            truncation=True,
            max_length=self.config.maximum_sequence_length,
            return_tensors="pt",
        )
        if not isinstance(encoded, Mapping):
            raise TextClassifierError("tokenizer output must be a mapping")
        input_ids = encoded.get("input_ids")
        attention_mask = encoded.get("attention_mask")
        if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
            raise TextClassifierError("tokenizer must return input_ids and attention_mask tensors")
        if (
            input_ids.ndim != 2
            or input_ids.shape[0] != 1
            or attention_mask.shape != input_ids.shape
        ):
            raise TextClassifierError("tokenizer tensors must have equal [1,sequence] shape")
        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids.to(self.config.device),
                attention_mask=attention_mask.to(self.config.device),
            )
            status_probabilities = F.softmax(self.calibration.apply(output.status_logits), dim=-1)
            object_probabilities = F.softmax(output.object_logits, dim=-1)
            bin_probabilities = F.softmax(output.bin_logits, dim=-1)
            confidence_tensor = calibrated_full_route_confidence(
                status_logits=output.status_logits,
                object_logits=output.object_logits,
                bin_logits=output.bin_logits,
                temperature=self.calibration,
            )
        status_index = int(status_probabilities.argmax(dim=-1)[0])
        object_index = int(object_probabilities.argmax(dim=-1)[0])
        bin_index = int(bin_probabilities.argmax(dim=-1)[0])
        full_route_confidence = float(confidence_tensor[0])
        status = RouterStatus(STATUS_LABELS[status_index])
        object_id = OBJECT_LABELS[object_index]
        bin_id = BIN_LABELS[bin_index]
        confidence = RouterConfidence.probability(
            full_route_confidence,
            definition=(
                "temperature-scaled P(route) multiplied by winning object and bin probabilities"
            ),
            calibrated=True,
        )
        evidence = {
            "maximum_sequence_length": self.config.maximum_sequence_length,
            "routing_threshold": self.config.routing_threshold,
            "temperature": self.calibration.temperature,
            "predicted_status": status.value,
            "predicted_object_id": object_id,
            "predicted_bin_id": bin_id,
            "status_probabilities": status_probabilities[0].detach().cpu().tolist(),
            "object_probabilities": object_probabilities[0].detach().cpu().tolist(),
            "bin_probabilities": bin_probabilities[0].detach().cpu().tolist(),
        }
        if (
            status is RouterStatus.ROUTE
            and object_id != "none"
            and bin_id != "none"
            and full_route_confidence >= self.config.routing_threshold
        ):
            return RouterDecision.route(
                task_spec=TaskSpec.from_mapping(
                    {
                        "target_object_id": object_id,
                        "target_bin_id": bin_id,
                        "instruction_template_id": "canonical_v0",
                    }
                ),
                confidence=confidence,
                router_name=TEXT_CLASSIFIER_ROUTER_NAME,
                router_version=TEXT_CLASSIFIER_ROUTER_VERSION,
                evidence=evidence,
            )
        if status is not RouterStatus.ROUTE:
            reason = _CLASSIFIER_REJECTION_BY_STATUS[status]
            rejection_status = status
        elif object_id == "none":
            reason = RouterRejectionReason.MISSING_OBJECT
            rejection_status = RouterStatus.REJECT_AMBIGUOUS
        elif bin_id == "none":
            reason = RouterRejectionReason.MISSING_DESTINATION
            rejection_status = RouterStatus.REJECT_AMBIGUOUS
        else:
            reason = RouterRejectionReason.LOW_CONFIDENCE
            rejection_status = RouterStatus.REJECT_AMBIGUOUS
        return RouterDecision.reject(
            status=rejection_status,
            rejection_reason=reason,
            confidence=confidence,
            router_name=TEXT_CLASSIFIER_ROUTER_NAME,
            router_version=TEXT_CLASSIFIER_ROUTER_VERSION,
            evidence=evidence,
        )


def compute_factorized_text_loss(
    output: FactorizedTextClassifierOutput,
    *,
    status_labels: torch.Tensor,
    object_labels: torch.Tensor,
    bin_labels: torch.Tensor,
    object_loss_weight: float = 1.0,
    bin_loss_weight: float = 1.0,
) -> FactorizedTextLoss:
    """Mask object/bin losses for every rejected training example."""

    batch = output.status_logits.shape[0]
    if output.status_logits.shape != (batch, len(STATUS_LABELS)):
        raise TextClassifierError("status logits have the wrong shape")
    if output.object_logits.shape != (batch, len(OBJECT_LABELS)):
        raise TextClassifierError("object logits have the wrong shape")
    if output.bin_logits.shape != (batch, len(BIN_LABELS)):
        raise TextClassifierError("bin logits have the wrong shape")
    for name, labels in (
        ("status_labels", status_labels),
        ("object_labels", object_labels),
        ("bin_labels", bin_labels),
    ):
        if labels.shape != (batch,) or labels.dtype != torch.long:
            raise TextClassifierError(f"{name} must be torch.long[batch]")
    status_loss = F.cross_entropy(output.status_logits, status_labels)
    route_mask = status_labels == ROUTE_STATUS_INDEX
    if bool(route_mask.any()):
        selected_objects = object_labels[route_mask]
        selected_bins = bin_labels[route_mask]
        if bool((selected_objects < 0).any()) or bool((selected_objects >= 3).any()):
            raise TextClassifierError("routeable object labels must be red/green/blue")
        if bool((selected_bins < 0).any()) or bool((selected_bins >= 2).any()):
            raise TextClassifierError("routeable bin labels must be left/right")
        object_loss = F.cross_entropy(output.object_logits[route_mask], selected_objects)
        bin_loss = F.cross_entropy(output.bin_logits[route_mask], selected_bins)
    else:
        object_loss = output.object_logits.sum() * 0.0
        bin_loss = output.bin_logits.sum() * 0.0
    total = (
        status_loss + float(object_loss_weight) * object_loss + float(bin_loss_weight) * bin_loss
    )
    if not bool(torch.isfinite(total)):
        raise TextClassifierError("factorized classifier loss is non-finite")
    return FactorizedTextLoss(
        total=total,
        status=status_loss,
        target_object=object_loss,
        target_bin=bin_loss,
        routed_examples=int(route_mask.sum().detach()),
    )


def classifier_manifest(
    *,
    model_id: str,
    model_revision: str,
    tokenizer_revision: str,
    hidden_size: int,
    dropout: float,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": TEXT_CLASSIFIER_ARTIFACT_SCHEMA,
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "hidden_size": hidden_size,
        "dropout": dropout,
        "status_labels": list(STATUS_LABELS),
        "object_labels": list(OBJECT_LABELS),
        "bin_labels": list(BIN_LABELS),
        "pooling": "first_token_v0",
    }
    payload["architecture_fingerprint"] = f"sha256:{sha256_hex(payload)}"
    return payload


def read_classifier_manifest(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TextClassifierError("classifier manifest must be a JSON object")
    return cast(dict[str, object], value)


def load_factorized_text_classifier(
    artifact_root: str | Path,
) -> tuple[FactorizedTextClassifierV0, object, dict[str, object]]:
    """Reload encoder, heads, and tokenizer from one local-only artifact."""

    root = Path(artifact_root).resolve()
    if not root.is_dir() or root.is_symlink() or root.is_junction():
        raise TextClassifierError("classifier artifact root must be one real directory")
    manifest = read_classifier_manifest(root / "classifier_config.json")
    required = {
        "schema_version",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "hidden_size",
        "dropout",
        "status_labels",
        "object_labels",
        "bin_labels",
        "pooling",
        "architecture_fingerprint",
    }
    if set(manifest) != required:
        raise TextClassifierError("classifier config fields differ from the exact schema")
    if manifest["schema_version"] != TEXT_CLASSIFIER_ARTIFACT_SCHEMA:
        raise TextClassifierError("classifier artifact schema version differs")
    if manifest["status_labels"] != list(STATUS_LABELS):
        raise TextClassifierError("classifier status label order differs")
    if manifest["object_labels"] != list(OBJECT_LABELS):
        raise TextClassifierError("classifier object label order differs")
    if manifest["bin_labels"] != list(BIN_LABELS):
        raise TextClassifierError("classifier bin label order differs")
    if manifest["pooling"] != "first_token_v0":
        raise TextClassifierError("classifier pooling contract differs")
    fingerprint = manifest.pop("architecture_fingerprint")
    expected_fingerprint = f"sha256:{sha256_hex(manifest)}"
    manifest["architecture_fingerprint"] = fingerprint
    if fingerprint != expected_fingerprint:
        raise TextClassifierError("classifier architecture fingerprint differs")
    hidden_size = manifest["hidden_size"]
    dropout = manifest["dropout"]
    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int):
        raise TextClassifierError("classifier hidden_size is malformed")
    if isinstance(dropout, bool) or not isinstance(dropout, int | float):
        raise TextClassifierError("classifier dropout is malformed")
    try:
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise TextClassifierError("installed Transformers and safetensors are required") from error
    try:
        encoder = AutoModel.from_pretrained(
            root / "encoder",
            local_files_only=True,
            trust_remote_code=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            root / "tokenizer",
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise TextClassifierError(
            f"strict local encoder/tokenizer reload failed: {error}"
        ) from error
    if not isinstance(encoder, nn.Module):
        raise TextClassifierError("local AutoModel reload did not return a torch module")
    model = FactorizedTextClassifierV0(
        encoder,
        hidden_size=hidden_size,
        dropout=float(dropout),
    )
    try:
        head_state = load_file(root / "router_heads.safetensors", device="cpu")
    except (OSError, RuntimeError, ValueError) as error:
        raise TextClassifierError(f"strict local classifier-head reload failed: {error}") from error
    model.load_heads_state_dict(head_state)
    return model, tokenizer, manifest


__all__ = [
    "BIN_LABELS",
    "OBJECT_LABELS",
    "ROUTE_STATUS_INDEX",
    "STATUS_LABELS",
    "TEXT_CLASSIFIER_ROUTER_NAME",
    "TEXT_CLASSIFIER_ROUTER_VERSION",
    "TEXT_CLASSIFIER_MODEL_ID",
    "TEXT_CLASSIFIER_MODEL_REVISION",
    "TEXT_CLASSIFIER_TOKENIZER_REVISION",
    "FactorizedTextClassifierOutput",
    "FactorizedTextRouterConfig",
    "FactorizedTextRouterV0",
    "FactorizedTextClassifierV0",
    "FactorizedTextLoss",
    "TextTokenizer",
    "TextClassifierError",
    "classifier_manifest",
    "compute_factorized_text_loss",
    "load_factorized_text_classifier",
    "read_classifier_manifest",
]
