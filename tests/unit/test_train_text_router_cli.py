from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from langmani.language.corpus import build_language_corpus
from langmani.language.router_types import LanguageSplit
from langmani.language.text_calibration import (
    RoutingThresholdSelectionV0,
    TemperatureCalibrationV0,
)
from langmani.language.text_training import (
    TextRouterCalibrationSelection,
    TextTrainingConfig,
)
from langmani.policies.act_runtime import GitState
from scripts import train_text_router as cli


class _Tokenizer:
    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        assert padding is True
        assert truncation is True
        assert max_length == 16
        assert return_tensors == "pt"
        return {
            "input_ids": torch.ones((len(texts), 3), dtype=torch.long),
            "attention_mask": torch.ones((len(texts), 3), dtype=torch.long),
        }


def _selection() -> TextRouterCalibrationSelection:
    return TextRouterCalibrationSelection(
        temperature=TemperatureCalibrationV0(
            temperature=1.25,
            validation_examples=300,
            nll_before=0.5,
            nll_after=0.4,
            bounded_iterations=64,
        ),
        threshold=RoutingThresholdSelectionV0(
            threshold=0.75,
            validation_routeable_examples=180,
            validation_rejected_examples=120,
            valid_full_task_accuracy=0.95,
            false_route_rate=0.025,
            rejection_recall=0.975,
        ),
    )


def test_target_identity_and_training_schedule_are_single_and_bounded() -> None:
    assert cli.MODEL_ID == "distilbert/distilbert-base-uncased"
    assert cli.MODEL_REVISION == "12040accade4e8a0f71eabdb258fecc2e7e948be"
    assert cli.TOKENIZER_REVISION == cli.MODEL_REVISION
    assert cli.TARGET_TRAINING_CONFIG.maximum_steps == 145
    assert cli.TARGET_TRAINING_CONFIG.checkpoint_steps == (29, 58, 87, 116, 145)
    assert cli.TARGET_STAGE_TRAINING_CONFIG.maximum_epochs == 5
    assert cli.TARGET_STAGE_TRAINING_CONFIG.pilot_step == 29
    assert cli.TARGET_STAGE_TRAINING_CONFIG.early_stopping_patience == 1
    assert cli.TARGET_TRAINING_CONFIG.encoder_trainable is True
    assert cli.TARGET_BATCH_SIZE == 32
    assert cli.TARGET_MAXIMUM_SEQUENCE_LENGTH == 64


def test_authoritative_checkpoint_contract_binds_resume_semantics(tmp_path: Path) -> None:
    corpus = build_language_corpus()
    git_state = GitState(
        commit="a" * 40,
        dirty=False,
        changed_paths=(),
        baseline_tracked=True,
    )
    run_fingerprint = f"sha256:{'b' * 64}"

    contract = cli._authoritative_checkpoint_processor_state(
        corpus_manifest=corpus.manifest.to_dict(),
        git_state=git_state,
        run_fingerprint=run_fingerprint,
        run_root=tmp_path / "run",
    )

    assert contract["run_fingerprint"] == run_fingerprint
    assert contract["corpus_fingerprint"] == corpus.manifest.corpus_fingerprint
    assert contract["split_fingerprints"] == cli._split_fingerprints(corpus.manifest.to_dict())
    assert contract["training_seed"] == 0
    assert contract["label_mappings"] == {
        "status": list(cli.STATUS_LABELS),
        "object": list(cli.OBJECT_LABELS),
        "bin": list(cli.BIN_LABELS),
        "rejected_object_bin_loss_mask": -1,
    }
    assert contract["pilot_boundary"] == {
        "definition": "min(one_complete_train_pass,ceil(0.20*maximum_steps))",
        "steps_per_epoch": 29,
        "maximum_steps": 145,
        "pilot_step": 29,
    }
    assert contract["git_state"] == git_state.to_dict()


def test_factorized_batches_only_materialize_train_or_validation() -> None:
    corpus = build_language_corpus()
    train = corpus.examples_for_split(LanguageSplit.TRAIN)[:5]
    first = cli._factorized_batches(
        tokenizer=_Tokenizer(),
        examples=train,
        split=LanguageSplit.TRAIN,
        batch_size=3,
        maximum_sequence_length=16,
        seed=0,
    )

    assert len(first) == 2
    assert all(batch.split == "train" for batch in first)
    assert sum(batch.input_ids.shape[0] for batch in first) == 5
    assert all(batch.object_labels.shape == batch.status_labels.shape for batch in first)
    with pytest.raises(cli.TextRouterCommandError, match="train or validation"):
        cli._factorized_batches(
            tokenizer=_Tokenizer(),
            examples=corpus.examples_for_split(LanguageSplit.DEVELOPMENT)[:1],
            split=LanguageSplit.DEVELOPMENT,
            batch_size=1,
            maximum_sequence_length=16,
            seed=0,
        )


def test_run_evidence_has_exact_reload_schema_and_device_independent_router() -> None:
    corpus = build_language_corpus()
    config = TextTrainingConfig(
        maximum_steps=1,
        checkpoint_steps=(1,),
        learning_rate=1e-3,
        seed=0,
    )
    evidence = cli._run_evidence(
        mode="target_development",
        corpus_manifest=corpus.manifest.to_dict(),
        git_state=GitState(
            commit="a" * 40,
            dirty=False,
            changed_paths=(),
            baseline_tracked=True,
        ),
        dependencies={"torch": "fixture"},
        training_config=config,
        batch_size=32,
        maximum_sequence_length=64,
        training_result={"selected_checkpoint_id": "step_00000001"},
        selection=_selection(),
        model_id=cli.MODEL_ID,
        model_revision=cli.MODEL_REVISION,
        tokenizer_revision=cli.TOKENIZER_REVISION,
        training_device="cuda",
        train_examples=900,
        validation_examples=300,
    )

    assert set(evidence) == {
        "schema_version",
        "mode",
        "corpus_manifest",
        "split_fingerprints",
        "git_commit",
        "git_state",
        "dependencies",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "training_config",
        "training_result",
        "calibration_config",
        "calibration_selection",
        "router_config",
        "data_usage",
        "training_runtime",
        "random_seeds",
    }
    assert evidence["router_config"] == {
        "maximum_sequence_length": 64,
        "routing_threshold": 0.75,
        "device_independent": True,
    }
    assert evidence["training_runtime"] == {
        "device": "cuda",
        "dtype": "float32",
        "cuda_available": torch.cuda.is_available(),
    }
    data_usage = evidence["data_usage"]
    assert isinstance(data_usage, dict)
    assert data_usage["gradient_split"] == "train"
    assert data_usage["checkpoint_selection_split"] == "validation"
    assert data_usage["development_examples_materialized"] is False
    assert data_usage["final_examples_materialized"] is False
    calibration = evidence["calibration_selection"]
    assert isinstance(calibration, dict)
    assert set(calibration) == {"temperature", "threshold"}


def test_target_prerequisites_reject_dirty_git_before_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    dirty = GitState(
        commit="a" * 40,
        dirty=True,
        changed_paths=(" M source.py",),
        baseline_tracked=True,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with pytest.raises(cli.TextRouterCommandError, match="clean tracked Git"):
        cli._validate_target_prerequisites(dirty)


def test_target_prerequisites_reject_missing_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    clean = GitState(
        commit="a" * 40,
        dirty=False,
        changed_paths=(),
        baseline_tracked=True,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(cli.TextRouterCommandError, match="requires CUDA"):
        cli._validate_target_prerequisites(clean)


def test_output_path_rejects_final_identity_and_source_overlap(tmp_path: Path) -> None:
    with pytest.raises(cli.TextRouterCommandError, match="prohibited"):
        cli._safe_paths(
            tmp_path / "m5a_language_final_v0",
            tmp_path / "report.json",
        )
    with pytest.raises(cli.TextRouterCommandError, match="cannot overlap"):
        cli._safe_paths(
            cli.PROJECT_ROOT / "src" / "bad-output",
            tmp_path / "report.json",
        )


def test_dry_run_is_truthful_and_does_not_train(tmp_path: Path) -> None:
    report = tmp_path / "dry-run.json"

    return_code = cli.main(
        [
            "--dry-run",
            "--output-root",
            str(tmp_path / "models"),
            "--report",
            str(report),
        ]
    )

    assert return_code == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["mode"] == "dry_run"
    assert payload["classifier_training_completed"] is False
    assert payload["classifier_checkpoint_selected"] is False
    assert payload["physical_target_validated"] is False
    assert payload["development_accessed"] is False
    assert payload["language_final_accessed"] is False
    assert payload["plan"]["model_revision"] == cli.MODEL_REVISION


def test_cli_modes_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args([])
    with pytest.raises(SystemExit):
        cli.parse_args(["--dry-run", "--fixture"])
    assert cli.parse_args(["--target-development"]).target_development is True
    assert cli.parse_args(["--target-pilot"]).target_pilot is True
    assert cli.parse_args(["--target-resume"]).target_resume is True
    assert cli.parse_args(["--tiny-overfit"]).tiny_overfit is True


def test_run_evidence_rejects_non_serializable_git_contract() -> None:
    corpus = build_language_corpus()
    with pytest.raises(cli.TextRouterCommandError, match="Git state"):
        cli._run_evidence(
            mode="fixture",
            corpus_manifest=corpus.manifest.to_dict(),
            git_state=SimpleNamespace(),
            dependencies={},
            training_config=cli.TARGET_TRAINING_CONFIG,
            batch_size=1,
            maximum_sequence_length=16,
            training_result={},
            selection=_selection(),
            model_id="fixture",
            model_revision="fixture",
            tokenizer_revision="fixture",
            training_device="cpu",
            train_examples=1,
            validation_examples=1,
        )
