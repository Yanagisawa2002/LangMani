from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from langmani.language.stage_protocol import (
    M5A_PHYSICAL_STAGE_BUDGETS,
    CandidatePromotionRecord,
    CandidatePromotionState,
    CheckpointRetentionPolicy,
    ClassifierRecoveryAmendment,
    ClassifierStageMetrics,
    ClassifierStageTrainingConfig,
    ClassifierValidationRanking,
    M5AStage,
    M5AStageProtocolError,
    maximum_pre_final_physical_episodes,
    promoted_candidates_for_stage,
    stage_protocol_manifest,
)
from langmani.language.text_classifier import FactorizedTextClassifierV0
from langmani.language.text_training import (
    FactorizedTextBatch,
    TextTrainingError,
    audit_staged_factorized_text_checkpoint,
    factorized_validation_fixture_payload,
    run_staged_factorized_text_training,
)


def _fingerprint(value: str) -> str:
    return f"sha256:{value * 64}"


def _promotion(
    candidate: str,
    previous: CandidatePromotionState,
    state: CandidatePromotionState,
    index: int,
) -> CandidatePromotionRecord:
    return CandidatePromotionRecord(
        candidate_id=candidate,
        previous_state=previous,
        state=state,
        gate_passed=True,
        evidence_fingerprint=f"sha256:{index:064x}",
        reason="fixture promotion",
    )


def test_single_seed_epoch_budget_and_pilot_are_immutable() -> None:
    config = ClassifierStageTrainingConfig(train_example_count=900)

    assert config.seed == 0
    assert config.steps_per_epoch == 29
    assert config.maximum_steps == 145
    assert config.pilot_step == 29
    assert config.validation_steps == (29, 58, 87, 116, 145)
    assert config.early_stopping_patience == 1
    assert config.to_dict()["classifier_training_seeds"] == 1
    assert config.to_dict()["robustness_across_training_seeds_not_evaluated"] is True
    assert CheckpointRetentionPolicy().maximum_retained_checkpoints == 3

    with pytest.raises(M5AStageProtocolError, match="exactly seed 0"):
        ClassifierStageTrainingConfig(train_example_count=900, seed=1)
    with pytest.raises(M5AStageProtocolError, match="capped at five epochs"):
        ClassifierStageTrainingConfig(train_example_count=900, maximum_epochs=6)


def test_classifier_gates_are_distinct_and_do_not_imply_each_other() -> None:
    pilot = ClassifierStageMetrics(
        full_task_spec_accuracy=0.86,
        object_accuracy=0.91,
        bin_accuracy=0.91,
        false_route_rate=0.09,
        schema_valid_rate=1.0,
        ambiguous_rejection_recall=0.1,
        unsupported_rejection_recall=0.1,
        malformed_rejection_recall=0.1,
        status_accuracy=0.9,
    )
    assert pilot.pilot_gate_passed
    assert not pilot.training_gate_passed
    assert not pilot.tiny_overfit_gate_passed

    tiny = ClassifierStageMetrics(
        full_task_spec_accuracy=0.99,
        object_accuracy=0.99,
        bin_accuracy=0.99,
        false_route_rate=0.0,
        schema_valid_rate=1.0,
        ambiguous_rejection_recall=1.0,
        unsupported_rejection_recall=1.0,
        malformed_rejection_recall=1.0,
        status_accuracy=0.99,
    )
    assert tiny.tiny_overfit_gate_passed
    assert tiny.pilot_gate_passed
    assert tiny.training_gate_passed


def test_recovery_amendment_and_seven_key_ranking_are_immutable() -> None:
    config = ClassifierStageTrainingConfig(train_example_count=900)
    amendment = ClassifierRecoveryAmendment(
        pilot_run_fingerprint=f"sha256:{'a' * 64}",
        pilot_checkpoint_fingerprint=f"sha256:{'b' * 64}",
        implementation_git_commit="c" * 40,
        unchanged_training_config=config.to_dict(),
        maximum_total_epochs=5,
        early_stopping_patience=1,
        validation_schedule=config.validation_steps,
        checkpoint_retention_policy=CheckpointRetentionPolicy().to_dict(),
    )
    assert amendment.to_dict()["new_run_identity"] is False
    assert amendment.to_dict()["original_pilot_decision"] == "rejected_after_one_epoch"
    assert amendment.amendment_fingerprint.startswith("sha256:")

    baseline = ClassifierValidationRanking(
        step=29,
        epoch=1,
        metrics=ClassifierStageMetrics(
            full_task_spec_accuracy=0.85,
            object_accuracy=0.85,
            bin_accuracy=1.0,
            false_route_rate=0.25,
            schema_valid_rate=1.0,
            ambiguous_rejection_recall=0.74,
            unsupported_rejection_recall=0.03,
            malformed_rejection_recall=0.0,
        ),
        total_validation_loss=2.5,
    )
    higher_task_accuracy = ClassifierValidationRanking(
        step=58,
        epoch=2,
        metrics=ClassifierStageMetrics(
            full_task_spec_accuracy=0.86,
            object_accuracy=0.80,
            bin_accuracy=0.80,
            false_route_rate=0.50,
            schema_valid_rate=1.0,
            ambiguous_rejection_recall=0.0,
            unsupported_rejection_recall=0.0,
            malformed_rejection_recall=0.0,
        ),
        total_validation_loss=3.0,
    )
    assert higher_task_accuracy.better_than(baseline)
    assert not baseline.better_than(higher_task_accuracy)


def test_failed_promotion_is_terminal_and_cannot_reach_physical_stage() -> None:
    rejected = CandidatePromotionRecord(
        candidate_id="classifier",
        previous_state=CandidatePromotionState.PILOT_PASSED,
        state=CandidatePromotionState.REJECTED,
        gate_passed=False,
        evidence_fingerprint=_fingerprint("a"),
        reason="pilot gate failed",
    )
    assert promoted_candidates_for_stage((rejected,), stage=M5AStage.ONE_SCENE_CONTROL_SMOKE) == ()
    with pytest.raises(M5AStageProtocolError, match="failed gate"):
        CandidatePromotionRecord(
            candidate_id="classifier",
            previous_state=CandidatePromotionState.PILOT_PASSED,
            state=CandidatePromotionState.TRAINING_PASSED,
            gate_passed=False,
            evidence_fingerprint=_fingerprint("b"),
            reason="invalid",
        )


def test_only_promoted_candidates_consume_later_physical_budgets() -> None:
    classifier = [
        _promotion(
            "classifier",
            CandidatePromotionState.DECLARED,
            CandidatePromotionState.FIXTURE_PASSED,
            1,
        ),
        _promotion(
            "classifier",
            CandidatePromotionState.FIXTURE_PASSED,
            CandidatePromotionState.TINY_OVERFIT_PASSED,
            2,
        ),
        _promotion(
            "classifier",
            CandidatePromotionState.TINY_OVERFIT_PASSED,
            CandidatePromotionState.PILOT_PASSED,
            3,
        ),
        _promotion(
            "classifier",
            CandidatePromotionState.PILOT_PASSED,
            CandidatePromotionState.TRAINING_PASSED,
            4,
        ),
        _promotion(
            "classifier",
            CandidatePromotionState.TRAINING_PASSED,
            CandidatePromotionState.LANGUAGE_PROMOTED,
            5,
        ),
        _promotion(
            "classifier",
            CandidatePromotionState.LANGUAGE_PROMOTED,
            CandidatePromotionState.ONE_SCENE_PROMOTED,
            6,
        ),
        _promotion(
            "classifier",
            CandidatePromotionState.ONE_SCENE_PROMOTED,
            CandidatePromotionState.THREE_SCENE_PROMOTED,
            7,
        ),
        _promotion(
            "classifier",
            CandidatePromotionState.THREE_SCENE_PROMOTED,
            CandidatePromotionState.SELECTED_FOR_FULL_DEVELOPMENT,
            8,
        ),
    ]
    assert promoted_candidates_for_stage(
        classifier[:5], stage=M5AStage.ONE_SCENE_CONTROL_SMOKE
    ) == ("classifier",)
    assert promoted_candidates_for_stage(
        classifier[:6], stage=M5AStage.THREE_SCENE_CONTROL_SCREEN
    ) == ("classifier",)
    assert promoted_candidates_for_stage(classifier, stage=M5AStage.FULL_CONTROL_DEVELOPMENT) == (
        "classifier",
    )


def test_physical_costs_match_progressive_protocol_and_keep_final_sealed() -> None:
    assert M5A_PHYSICAL_STAGE_BUDGETS[M5AStage.ONE_SCENE_CONTROL_SMOKE].maximum_episode_count == 18
    assert (
        M5A_PHYSICAL_STAGE_BUDGETS[M5AStage.THREE_SCENE_CONTROL_SCREEN].maximum_episode_count == 54
    )
    assert M5A_PHYSICAL_STAGE_BUDGETS[M5AStage.FULL_CONTROL_DEVELOPMENT].maximum_episode_count == 72
    assert M5A_PHYSICAL_STAGE_BUDGETS[M5AStage.SEALED_FINAL].maximum_episode_count == 144
    assert maximum_pre_final_physical_episodes() == 144
    manifest = stage_protocol_manifest()
    assert manifest["classifier_training_seeds"] == 1
    assert manifest["final_automatically_executed"] is False
    assert isinstance(manifest["protocol_fingerprint"], str)


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(dim=12)
        self.embedding = nn.Embedding(32, 12)

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def _batch(split: str) -> FactorizedTextBatch:
    return FactorizedTextBatch(
        input_ids=torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.long),
        attention_mask=torch.ones((4, 2), dtype=torch.long),
        status_labels=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        object_labels=torch.tensor([0, -1, -1, -1], dtype=torch.long),
        bin_labels=torch.tensor([0, -1, -1, -1], dtype=torch.long),
        split=split,
    )


def test_authoritative_pilot_checkpoint_resumes_optimizer_scheduler_and_rng(tmp_path) -> None:
    config = ClassifierStageTrainingConfig(
        train_example_count=8,
        batch_size=4,
        maximum_epochs=2,
        original_maximum_steps=4,
    )
    fingerprint = f"sha256:{'1' * 64}"
    root = tmp_path / "checkpoints"
    processor = {"tokenizer_revision": "fixture-v0", "maximum_sequence_length": 8}
    model = FactorizedTextClassifierV0(_Encoder(), dropout=0.0)
    pilot = run_staged_factorized_text_training(
        model=model,
        train_batches=(_batch("train"),),
        validation_batches=(_batch("validation"),),
        config=config,
        run_fingerprint=fingerprint,
        checkpoint_root=root,
        processor_state=processor,
        stop_after_pilot=True,
        resume=False,
    )

    assert pilot.phase == "pilot_paused"
    assert pilot.completed_steps == config.pilot_step
    assert len(pilot.training_records) == config.pilot_step
    assert [record.step for record in pilot.training_records] == list(
        range(1, config.pilot_step + 1)
    )
    assert all(record.total_loss > 0.0 for record in pilot.training_records)
    assert all(record.gradient_norm > 0.0 for record in pilot.training_records)
    assert all(record.status_head_gradient_norm > 0.0 for record in pilot.training_records)
    assert all(record.object_head_gradient_norm > 0.0 for record in pilot.training_records)
    assert all(record.bin_head_gradient_norm > 0.0 for record in pilot.training_records)
    assert {path.name for path in root.glob("*.pt")} == {
        "pilot.pt",
        "latest.pt",
        "validation_best.pt",
    }

    assert pilot.pilot_outputs is not None
    fixture = factorized_validation_fixture_payload(pilot.pilot_outputs)
    audited = audit_staged_factorized_text_checkpoint(
        model=FactorizedTextClassifierV0(_Encoder(), dropout=0.0),
        validation_batches=(_batch("validation"),),
        config=config,
        run_fingerprint=fingerprint,
        checkpoint_path=root / "latest.pt",
        processor_state=processor,
        expected_fixture=fixture,
        expected_step=config.pilot_step,
    )
    assert audited.restored_step == config.pilot_step
    assert audited.next_global_step == config.pilot_step + 1
    assert audited.optimizer_state_restored
    assert audited.scheduler_state_restored
    assert audited.processor_state_restored
    assert audited.rng_state_restored
    assert audited.data_progress_restored
    assert audited.deterministic_logits_match
    assert audited.maximum_absolute_logit_error == 0.0
    assert audited.training_records_restored == config.pilot_step
    assert audited.pilot_complete
    assert not audited.full_training_complete
    assert audited.resumable

    reloaded = FactorizedTextClassifierV0(_Encoder(), dropout=0.0)
    completed = run_staged_factorized_text_training(
        model=reloaded,
        train_batches=(_batch("train"),),
        validation_batches=(_batch("validation"),),
        config=config,
        run_fingerprint=fingerprint,
        checkpoint_root=root,
        processor_state=processor,
        stop_after_pilot=False,
        resume=True,
    )
    assert completed.phase == "training_complete"
    assert completed.resumed_from_step == config.pilot_step
    assert completed.completed_steps > completed.resumed_from_step
    assert len(tuple(root.glob("*.pt"))) == 3

    with pytest.raises(TextTrainingError, match="use explicit resume"):
        run_staged_factorized_text_training(
            model=FactorizedTextClassifierV0(_Encoder(), dropout=0.0),
            train_batches=(_batch("train"),),
            validation_batches=(_batch("validation"),),
            config=config,
            run_fingerprint=fingerprint,
            checkpoint_root=root,
            processor_state=processor,
            stop_after_pilot=True,
            resume=False,
        )


def test_rejected_pilot_can_use_only_explicit_recovery_ranking(tmp_path) -> None:
    config = ClassifierStageTrainingConfig(
        train_example_count=8,
        batch_size=4,
        maximum_epochs=2,
        original_maximum_steps=4,
    )
    fingerprint = f"sha256:{'2' * 64}"
    amendment = f"sha256:{'3' * 64}"
    root = tmp_path / "recovery"
    processor = {"tokenizer_revision": "fixture-v0", "maximum_sequence_length": 8}
    pilot = run_staged_factorized_text_training(
        model=FactorizedTextClassifierV0(_Encoder(), dropout=0.0),
        train_batches=(_batch("train"),),
        validation_batches=(_batch("validation"),),
        config=config,
        run_fingerprint=fingerprint,
        checkpoint_root=root,
        processor_state=processor,
        stop_after_pilot=True,
        resume=False,
    )
    assert pilot.pilot_metrics is not None
    recovered = run_staged_factorized_text_training(
        model=FactorizedTextClassifierV0(_Encoder(), dropout=0.0),
        train_batches=(_batch("train"),),
        validation_batches=(_batch("validation"),),
        config=config,
        run_fingerprint=fingerprint,
        checkpoint_root=root,
        processor_state=processor,
        stop_after_pilot=False,
        resume=True,
        recovery_ranking=True,
        recovery_amendment_fingerprint=amendment,
        resume_checkpoint_role="pilot",
        recovery_baseline_metrics=pilot.pilot_metrics,
    )
    assert recovered.resumed_from_step == config.pilot_step
    assert recovered.validation_rankings[0].step == config.pilot_step
    assert (
        recovered.selected_step
        == min(recovered.validation_rankings, key=lambda value: value.key).step
    )
    assert {path.name for path in root.glob("*.pt")} == {
        "pilot.pt",
        "latest.pt",
        "validation_best.pt",
    }
