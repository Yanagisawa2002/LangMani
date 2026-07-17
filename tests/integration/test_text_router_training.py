from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from langmani.language.text_classifier import (
    FactorizedTextClassifierV0,
    classifier_manifest,
    load_factorized_text_classifier,
)
from langmani.language.text_training import (
    FactorizedTextBatch,
    FactorizedValidationOutputs,
    TextCheckpointValidation,
    TextTrainingConfig,
    TextTrainingError,
    calibrate_and_select_text_router,
    read_text_classifier_run_evidence,
    run_bounded_factorized_text_training,
    select_validation_checkpoint,
    stage_and_promote_text_classifier,
    text_classifier_run_fingerprint,
    train_factorized_text_step,
)


class _TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(dim=8)
        self.embedding = nn.Embedding(16, 8)

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def _batch(split: str) -> FactorizedTextBatch:
    return FactorizedTextBatch(
        input_ids=torch.tensor([[2, 3], [4, 5]], dtype=torch.long),
        attention_mask=torch.ones((2, 2), dtype=torch.long),
        status_labels=torch.tensor([0, 2], dtype=torch.long),
        object_labels=torch.tensor([0, -1], dtype=torch.long),
        bin_labels=torch.tensor([1, -1], dtype=torch.long),
        split=split,
    )


@pytest.mark.integration
@pytest.mark.fixture
def test_tiny_offline_distilbert_step_save_and_reload(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.DistilBertConfig(
        vocab_size=32,
        max_position_embeddings=32,
        n_layers=1,
        n_heads=2,
        dim=16,
        hidden_dim=32,
        dropout=0.0,
        attention_dropout=0.0,
    )
    encoder = transformers.DistilBertModel(config)
    model = FactorizedTextClassifierV0(encoder, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = FactorizedTextBatch(
        input_ids=torch.tensor([[2, 5, 6, 3], [2, 7, 8, 3]], dtype=torch.long),
        attention_mask=torch.ones((2, 4), dtype=torch.long),
        status_labels=torch.tensor([0, 2], dtype=torch.long),
        object_labels=torch.tensor([0, -1], dtype=torch.long),
        bin_labels=torch.tensor([1, -1], dtype=torch.long),
        split="train",
    )

    result = train_factorized_text_step(model=model, optimizer=optimizer, batch=batch)

    assert result.total_loss > 0.0
    vocab = tmp_path / "vocab.txt"
    vocab.write_text(
        "\n".join(
            ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "pick", "red", "cube", "right", "bin"]
        )
        + "\n",
        encoding="utf-8",
    )
    tokenizer = transformers.DistilBertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)
    manifest = classifier_manifest(
        model_id="fixture-distilbert",
        model_revision="fixture-revision",
        tokenizer_revision="fixture-revision",
        hidden_size=16,
        dropout=0.0,
    )
    run_evidence = {
        "corpus_fingerprint": "sha256:" + "1" * 64,
        "validation_selection": {"checkpoint_id": "fixture-step"},
        "calibration": {"temperature": 1.0, "threshold": 0.5},
        "git_commit": "a" * 40,
        "dependency_versions": {"torch": torch.__version__},
    }
    fingerprint = text_classifier_run_fingerprint(
        classifier_manifest=manifest,
        run_evidence=run_evidence,
    )
    artifact = stage_and_promote_text_classifier(
        output_root=tmp_path / "models",
        run_fingerprint=fingerprint,
        model=model,
        tokenizer=tokenizer,
        classifier_manifest=manifest,
        run_evidence=run_evidence,
    )

    reloaded, reloaded_tokenizer, reloaded_manifest = load_factorized_text_classifier(artifact)
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        expected = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
        actual = reloaded(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
    assert torch.equal(expected.status_logits, actual.status_logits)
    assert torch.equal(expected.object_logits, actual.object_logits)
    assert torch.equal(expected.bin_logits, actual.bin_logits)
    assert reloaded_manifest == manifest
    assert reloaded_tokenizer("pick red cube") == tokenizer("pick red cube")
    assert read_text_classifier_run_evidence(artifact)["git_commit"] == "a" * 40


@pytest.mark.integration
def test_training_step_rejects_validation_gradient_use() -> None:
    transformers = pytest.importorskip("transformers")
    encoder = transformers.DistilBertModel(
        transformers.DistilBertConfig(
            vocab_size=16,
            max_position_embeddings=8,
            n_layers=1,
            n_heads=2,
            dim=8,
            hidden_dim=16,
        )
    )
    model = FactorizedTextClassifierV0(encoder, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = FactorizedTextBatch(
        input_ids=torch.tensor([[2, 3]], dtype=torch.long),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        status_labels=torch.tensor([0], dtype=torch.long),
        object_labels=torch.tensor([0], dtype=torch.long),
        bin_labels=torch.tensor([0], dtype=torch.long),
        split="validation",
    )
    with pytest.raises(RuntimeError, match="train split only"):
        train_factorized_text_step(model=model, optimizer=optimizer, batch=batch)


@pytest.mark.integration
def test_bounded_training_selects_only_validation_checkpoints() -> None:
    torch.manual_seed(3)
    model = FactorizedTextClassifierV0(_TinyEncoder(), dropout=0.0)
    saved: list[str] = []

    result = run_bounded_factorized_text_training(
        model=model,
        train_batches=(_batch("train"),),
        validation_batches=(_batch("validation"),),
        config=TextTrainingConfig(
            maximum_steps=3,
            checkpoint_steps=(1, 2, 3),
            learning_rate=1e-3,
            seed=3,
        ),
        checkpoint_callback=lambda checkpoint_id, _model, _record: saved.append(checkpoint_id),
    )

    assert result.completed_steps == 3
    assert len(result.checkpoint_validations) == 3
    assert saved == ["step_00000001", "step_00000002", "step_00000003"]
    assert result.selected_checkpoint_id in saved
    assert result.config_fingerprint.startswith("sha256:")
    with pytest.raises(TextTrainingError, match="validation"):
        select_validation_checkpoint(
            (
                TextCheckpointValidation(
                    checkpoint_id="development-checkpoint",
                    step=1,
                    validation_examples=1,
                    total_loss=1.0,
                    status_loss=1.0,
                    object_loss=1.0,
                    bin_loss=1.0,
                    evidence_split="development",
                ),
            ),
            evidence_split="development",
        )


@pytest.mark.integration
def test_validation_only_calibration_and_threshold_entry() -> None:
    outputs = FactorizedValidationOutputs(
        status_logits=torch.tensor(
            [[6.0, 0.0, 0.0, 0.0], [6.0, 0.0, 0.0, 0.0], [0.0, 6.0, 0.0, 0.0], [0.0, 0.0, 6.0, 0.0]]
        ),
        object_logits=torch.tensor(
            [[6.0, 0.0, 0.0, 0.0], [0.0, 6.0, 0.0, 0.0], [6.0, 0.0, 0.0, 0.0], [6.0, 0.0, 0.0, 0.0]]
        ),
        bin_logits=torch.tensor(
            [[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [6.0, 0.0, 0.0], [6.0, 0.0, 0.0]]
        ),
        status_labels=torch.tensor([0, 0, 1, 2], dtype=torch.long),
        object_labels=torch.tensor([0, 1, -1, -1], dtype=torch.long),
        bin_labels=torch.tensor([0, 1, -1, -1], dtype=torch.long),
    )

    result = calibrate_and_select_text_router(
        outputs=outputs,
        threshold_candidates=(0.1, 0.5, 0.9),
        temperature_iterations=16,
    )

    assert result.temperature.validation_examples == 4
    assert result.threshold.threshold in {0.1, 0.5, 0.9}
    assert result.threshold.false_route_rate == 0.0
    invalid = FactorizedValidationOutputs(
        status_logits=outputs.status_logits,
        object_logits=outputs.object_logits,
        bin_logits=outputs.bin_logits,
        status_labels=outputs.status_labels,
        object_labels=outputs.object_labels,
        bin_labels=outputs.bin_labels,
        evidence_split="development",
    )
    with pytest.raises(TextTrainingError, match="validation"):
        calibrate_and_select_text_router(
            outputs=invalid,
            threshold_candidates=(0.5,),
        )
