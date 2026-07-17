from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from langmani.language.router_types import RouterRejectionReason, RouterStatus
from langmani.language.text_calibration import TemperatureCalibrationV0
from langmani.language.text_classifier import (
    BIN_LABELS,
    OBJECT_LABELS,
    STATUS_LABELS,
    FactorizedTextClassifierV0,
    FactorizedTextRouterConfig,
    FactorizedTextRouterV0,
    TextClassifierError,
    compute_factorized_text_loss,
)


class _TinyEncoder(nn.Module):
    def __init__(self, hidden_size: int = 12) -> None:
        super().__init__()
        self.config = SimpleNamespace(dim=hidden_size)
        self.embedding = nn.Embedding(32, hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.projection(self.embedding(input_ids)))


def _model() -> FactorizedTextClassifierV0:
    return FactorizedTextClassifierV0(_TinyEncoder(), dropout=0.0)


class _Tokenizer:
    def __call__(
        self,
        text: str,
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        del text, padding, truncation, max_length, return_tensors
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }


def _router(*, status: int, target_object: int, target_bin: int, threshold: float = 0.0):
    model = _model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.status_head.bias[status] = 10.0
        model.object_head.bias[target_object] = 10.0
        model.bin_head.bias[target_bin] = 10.0
    return FactorizedTextRouterV0(
        model=model,
        tokenizer=_Tokenizer(),
        calibration=TemperatureCalibrationV0(
            temperature=1.0,
            validation_examples=1,
            nll_before=0.1,
            nll_after=0.1,
            bounded_iterations=1,
        ),
        config=FactorizedTextRouterConfig(
            maximum_sequence_length=32,
            routing_threshold=threshold,
        ),
    )


def test_factorized_classifier_has_exact_head_shapes() -> None:
    model = _model()
    output = model(
        input_ids=torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.long),
        attention_mask=torch.ones((2, 3), dtype=torch.long),
    )

    assert output.status_logits.shape == (2, len(STATUS_LABELS))
    assert output.object_logits.shape == (2, len(OBJECT_LABELS))
    assert output.bin_logits.shape == (2, len(BIN_LABELS))


def test_rejected_examples_mask_object_and_bin_losses() -> None:
    output = _model()(
        input_ids=torch.tensor([[1, 2], [2, 3]], dtype=torch.long),
        attention_mask=torch.ones((2, 2), dtype=torch.long),
    )
    loss = compute_factorized_text_loss(
        output,
        status_labels=torch.tensor([1, 2], dtype=torch.long),
        object_labels=torch.tensor([999, -999], dtype=torch.long),
        bin_labels=torch.tensor([999, -999], dtype=torch.long),
    )

    assert loss.routed_examples == 0
    assert loss.target_object.item() == 0.0
    assert loss.target_bin.item() == 0.0
    assert loss.total.item() == pytest.approx(loss.status.item())


def test_routeable_examples_require_semantic_object_and_bin_labels() -> None:
    output = _model()(
        input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
    )

    with pytest.raises(TextClassifierError, match="routeable object labels"):
        compute_factorized_text_loss(
            output,
            status_labels=torch.tensor([0], dtype=torch.long),
            object_labels=torch.tensor([3], dtype=torch.long),
            bin_labels=torch.tensor([0], dtype=torch.long),
        )


def test_factorized_loss_backpropagates_through_all_three_heads() -> None:
    model = _model()
    output = model(
        input_ids=torch.tensor([[1, 2], [2, 3]], dtype=torch.long),
        attention_mask=torch.ones((2, 2), dtype=torch.long),
    )
    loss = compute_factorized_text_loss(
        output,
        status_labels=torch.tensor([0, 1], dtype=torch.long),
        object_labels=torch.tensor([2, -1], dtype=torch.long),
        bin_labels=torch.tensor([1, -1], dtype=torch.long),
    )
    loss.total.backward()

    assert model.status_head.weight.grad is not None
    assert model.object_head.weight.grad is not None
    assert model.bin_head.weight.grad is not None


def test_classifier_rejects_mismatched_token_shapes() -> None:
    with pytest.raises(TextClassifierError, match="equal"):
        _model()(
            input_ids=torch.ones((2, 3), dtype=torch.long),
            attention_mask=torch.ones((2, 2), dtype=torch.long),
        )


def test_classifier_router_builds_task_spec_from_three_heads() -> None:
    router = _router(status=0, target_object=1, target_bin=1)

    decision = router.route("Put the green cube in the right bin")

    assert decision.status is RouterStatus.ROUTE
    assert decision.task_spec is not None
    assert decision.task_spec.target_object_id == "green_cube"
    assert decision.task_spec.target_bin_id == "right_bin"
    assert decision.confidence.available is True
    assert decision.confidence.calibrated is True
    assert router.route("Put the green cube in the right bin").decision_fingerprint == (
        decision.decision_fingerprint
    )


def test_classifier_router_rejects_predicted_status_without_task_fields() -> None:
    decision = _router(status=2, target_object=0, target_bin=0).route("Open the drawer")

    assert decision.status is RouterStatus.REJECT_UNSUPPORTED
    assert decision.rejection_reason is RouterRejectionReason.CLASSIFIER_UNSUPPORTED
    assert decision.task_spec is None


def test_classifier_router_rejects_none_head_and_low_confidence() -> None:
    missing = _router(status=0, target_object=3, target_bin=0).route("Put it left")
    low = _router(status=0, target_object=0, target_bin=0, threshold=1.0).route(
        "Put the red cube in the left bin"
    )

    assert missing.rejection_reason is RouterRejectionReason.MISSING_OBJECT
    assert low.rejection_reason is RouterRejectionReason.LOW_CONFIDENCE
    assert missing.task_spec is None
    assert low.task_spec is None


def test_classifier_router_rejects_empty_text_before_inference() -> None:
    decision = _router(status=0, target_object=0, target_bin=0).route("  ")

    assert decision.status is RouterStatus.REJECT_MALFORMED
    assert decision.rejection_reason is RouterRejectionReason.EMPTY_TEXT
    assert decision.confidence.available is False
