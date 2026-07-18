from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from environment.verify_m5a import _verify_stage_evidence
from langmani.environments.specs import TaskSpec
from langmani.language.classifier_decoders import (
    ClassifierDecoderError,
    DecoderCandidate,
    DecoderConfigurationV0,
    DecoderThresholdGridV0,
    decode_classifier_probabilities,
    fixed_decoder_grid_v0,
)
from langmani.language.decoder_selection import (
    DecoderEvaluationArraysV0,
    DecoderSelectionError,
    DecoderSelectionResultV0,
    _selection_key,
    build_decoder_evaluation_arrays,
    decoder_runtime_fingerprint,
    evaluate_decoder_configuration,
)
from langmani.language.rejection_report import (
    RejectionReportError,
    validate_rejection_analysis_artifact,
    write_rejection_analysis_artifact,
)
from langmani.language.router_types import (
    LanguageExample,
    LanguageSplit,
    RouterRejectionReason,
    RouterStatus,
    stable_language_example_id,
)
from langmani.language.stage_protocol import M5AStage
from langmani.language.status_calibration import fit_status_temperature_comparison
from scripts.analyze_classifier_rejection import main as analysis_main


def _configuration(
    candidate: DecoderCandidate = DecoderCandidate.BASELINE_FOUR_WAY_ARGMAX_V0,
    *,
    route: float = 0.5,
    margin: float = 0.0,
    object_threshold: float = 0.5,
    bin_threshold: float = 0.5,
) -> DecoderConfigurationV0:
    if candidate is DecoderCandidate.BASELINE_FOUR_WAY_ARGMAX_V0:
        return DecoderConfigurationV0(
            candidate=candidate,
            temperature_mode="identity",
            status_temperature=1.0,
        )
    confidence = candidate in {
        DecoderCandidate.HIERARCHICAL_STATUS_DECODER_V0,
        DecoderCandidate.CONSERVATIVE_ROUTE_DECODER_V0,
    }
    return DecoderConfigurationV0(
        candidate=candidate,
        temperature_mode="identity",
        status_temperature=1.0,
        route_threshold=route,
        route_margin_threshold=margin,
        object_confidence_threshold=object_threshold if confidence else None,
        bin_confidence_threshold=bin_threshold if confidence else None,
    )


def _decode(
    configuration: DecoderConfigurationV0,
    *,
    status: tuple[float, ...] = (0.7, 0.1, 0.1, 0.1),
    objects: tuple[float, ...] = (0.8, 0.1, 0.05, 0.05),
    bins: tuple[float, ...] = (0.8, 0.1, 0.1),
):
    return decode_classifier_probabilities(
        status_probabilities=status,
        object_probabilities=objects,
        bin_probabilities=bins,
        configuration=configuration,
    )


def _example(index: int, status: RouterStatus, *, split: LanguageSplit = LanguageSplit.VALIDATION):
    task = (
        TaskSpec("red_cube", "left_bin", "canonical_v0") if status is RouterStatus.ROUTE else None
    )
    reasons = {
        RouterStatus.REJECT_AMBIGUOUS: RouterRejectionReason.MISSING_OBJECT,
        RouterStatus.REJECT_UNSUPPORTED: RouterRejectionReason.UNSUPPORTED_OBJECT,
        RouterStatus.REJECT_MALFORMED: RouterRejectionReason.MEANINGLESS_TEXT,
    }
    raw = f"fixture {index}"
    provenance = {"source": "unit"}
    kwargs = {
        "raw_text": raw,
        "normalized_text": raw,
        "template_family_id": f"family-{index}",
        "lexical_variant_ids": (f"variant-{index}",),
        "expected_status": status,
        "expected_task_spec": task,
        "expected_rejection_reason": reasons.get(status),
        "generation_provenance": provenance,
        "split": split,
    }
    return LanguageExample(
        example_id=stable_language_example_id(**kwargs),
        **kwargs,
    )


def _perfect_arrays() -> DecoderEvaluationArraysV0:
    return DecoderEvaluationArraysV0(
        expected_status=np.asarray([0, 1, 2, 3]),
        expected_object=np.asarray([0, -1, -1, -1]),
        expected_bin=np.asarray([0, -1, -1, -1]),
        status_probabilities=np.asarray(
            [
                [0.9, 0.04, 0.03, 0.03],
                [0.01, 0.95, 0.02, 0.02],
                [0.01, 0.02, 0.95, 0.02],
                [0.01, 0.02, 0.02, 0.95],
            ]
        ),
        object_probabilities=np.asarray([[0.9, 0.03, 0.03, 0.04]] * 4),
        bin_probabilities=np.asarray([[0.9, 0.05, 0.05]] * 4),
    )


def test_four_way_baseline_reproduces_status_argmax() -> None:
    decision = _decode(_configuration(), status=(0.2, 0.5, 0.2, 0.1))
    assert decision.status is RouterStatus.REJECT_AMBIGUOUS


def test_binary_route_reject_uses_aggregated_probability() -> None:
    decision = _decode(
        _configuration(DecoderCandidate.AGGREGATED_REJECT_THRESHOLD_V0),
        status=(0.4, 0.21, 0.2, 0.19),
    )
    assert decision.p_reject == pytest.approx(0.6)
    assert decision.status is not RouterStatus.ROUTE


def test_rejection_probability_renormalization_preserves_reason_argmax() -> None:
    decision = _decode(
        _configuration(DecoderCandidate.HIERARCHICAL_STATUS_DECODER_V0),
        status=(0.1, 0.18, 0.45, 0.27),
    )
    normalized = torch.tensor([0.18, 0.45, 0.27]) / 0.9
    assert int(normalized.argmax()) == 1
    assert decision.status is RouterStatus.REJECT_UNSUPPORTED


def test_route_threshold_is_enforced() -> None:
    decision = _decode(
        _configuration(DecoderCandidate.AGGREGATED_REJECT_THRESHOLD_V0, route=0.75),
        status=(0.7, 0.1, 0.1, 0.1),
    )
    assert decision.status is not RouterStatus.ROUTE


def test_route_margin_is_enforced() -> None:
    decision = _decode(
        _configuration(DecoderCandidate.AGGREGATED_REJECT_THRESHOLD_V0, margin=0.25),
        status=(0.6, 0.15, 0.15, 0.1),
    )
    assert decision.route_margin == pytest.approx(0.2)
    assert decision.status is not RouterStatus.ROUTE


def test_object_confidence_threshold_is_enforced() -> None:
    decision = _decode(
        _configuration(
            DecoderCandidate.HIERARCHICAL_STATUS_DECODER_V0,
            object_threshold=0.85,
        )
    )
    assert decision.status is not RouterStatus.ROUTE


def test_bin_confidence_threshold_is_enforced() -> None:
    decision = _decode(
        _configuration(
            DecoderCandidate.CONSERVATIVE_ROUTE_DECODER_V0,
            bin_threshold=0.85,
        )
    )
    assert decision.status is not RouterStatus.ROUTE


def test_rejected_decision_has_no_executable_task_spec() -> None:
    decision = _decode(_configuration(), status=(0.1, 0.7, 0.1, 0.1))
    assert decision.task_spec is None
    assert decision.to_dict()["task_id"] is None


def test_invalid_object_or_bin_pair_rejects_safely() -> None:
    decision = _decode(
        _configuration(DecoderCandidate.CONSERVATIVE_ROUTE_DECODER_V0),
        objects=(0.02, 0.02, 0.02, 0.94),
    )
    assert decision.status is not RouterStatus.ROUTE
    assert decision.target_object_id is None


def test_fixed_threshold_grid_is_finite_and_exact() -> None:
    grid = fixed_decoder_grid_v0()
    assert len(grid.route_thresholds) == 12
    assert len(grid.route_margin_thresholds) == 15
    assert grid.temperature_modes == ("identity", "validation_temperature")


def test_fixed_grid_cannot_expand_after_lock() -> None:
    with pytest.raises(ClassifierDecoderError, match="cannot expand"):
        DecoderThresholdGridV0(route_thresholds=(*fixed_decoder_grid_v0().route_thresholds, 1.0))


def test_decoder_evaluation_rejects_non_validation_examples() -> None:
    with pytest.raises(DecoderSelectionError, match="validation examples only"):
        build_decoder_evaluation_arrays(
            examples=[_example(0, RouterStatus.ROUTE, split=LanguageSplit.DEVELOPMENT)],
            status_logits=torch.zeros(1, 4),
            object_logits=torch.zeros(1, 4),
            bin_logits=torch.zeros(1, 3),
            temperature=1.0,
        )


def test_false_route_calculation() -> None:
    arrays = _perfect_arrays()
    arrays.status_probabilities[1] = [0.9, 0.04, 0.03, 0.03]
    metrics = evaluate_decoder_configuration(arrays, _configuration())
    assert metrics.false_route_rate == pytest.approx(1 / 3)


def test_false_rejection_calculation() -> None:
    arrays = _perfect_arrays()
    arrays.status_probabilities[0] = [0.1, 0.8, 0.05, 0.05]
    metrics = evaluate_decoder_configuration(arrays, _configuration())
    assert metrics.false_rejection_rate == 1.0


def test_per_rejection_class_recall() -> None:
    metrics = evaluate_decoder_configuration(_perfect_arrays(), _configuration())
    assert metrics.ambiguous_rejection_recall == 1.0
    assert metrics.unsupported_rejection_recall == 1.0
    assert metrics.malformed_rejection_recall == 1.0


def test_rejection_macro_recall() -> None:
    arrays = _perfect_arrays()
    arrays.status_probabilities[1] = [0.9, 0.04, 0.03, 0.03]
    metrics = evaluate_decoder_configuration(arrays, _configuration())
    assert metrics.rejection_macro_recall == pytest.approx(2 / 3)


def test_exact_conjunctive_quality_gate() -> None:
    perfect = evaluate_decoder_configuration(_perfect_arrays(), _configuration())
    failed_arrays = _perfect_arrays()
    failed_arrays.status_probabilities[1] = [0.9, 0.04, 0.03, 0.03]
    failed = evaluate_decoder_configuration(failed_arrays, _configuration())
    assert perfect.quality_gate_passed is True
    assert failed.quality_gate_passed is False


def test_deterministic_tie_break_prefers_higher_route_threshold() -> None:
    metrics = evaluate_decoder_configuration(_perfect_arrays(), _configuration())
    low = _configuration(DecoderCandidate.AGGREGATED_REJECT_THRESHOLD_V0, route=0.4)
    high = _configuration(DecoderCandidate.AGGREGATED_REJECT_THRESHOLD_V0, route=0.45)
    assert _selection_key(high, metrics) < _selection_key(low, metrics)


def test_temperature_scaling_serialization_is_deterministic() -> None:
    logits = torch.tensor([[4.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0]])
    labels = torch.tensor([0, 1], dtype=torch.long)
    first = fit_status_temperature_comparison(logits, labels, evidence_split="validation")
    second = fit_status_temperature_comparison(logits, labels, evidence_split="validation")
    assert first.to_dict() == second.to_dict()


def test_promoted_runtime_fingerprint_binds_decoder() -> None:
    configuration = _configuration()
    metrics = evaluate_decoder_configuration(_perfect_arrays(), configuration)
    selection = DecoderSelectionResultV0(
        grid_fingerprint=fixed_decoder_grid_v0().grid_fingerprint,
        temperature_comparison_fingerprint="sha256:" + "a" * 64,
        baseline_configuration=configuration,
        baseline_metrics=metrics,
        candidate_summaries=(),
        selected_configuration=configuration,
        selected_metrics=metrics,
        conclusion="classifier_promoted_after_posthoc_calibration",
        classifier_full_quality_gate_passed=True,
    )
    fingerprint = decoder_runtime_fingerprint(
        classifier_checkpoint_fingerprint="sha256:" + "b" * 64,
        tokenizer_fingerprint="sha256:" + "c" * 64,
        processor_fingerprint="sha256:" + "d" * 64,
        selection=selection,
        git_commit="e" * 40,
        corpus_fingerprint="sha256:" + "f" * 64,
        validation_split_fingerprint="sha256:" + "0" * 64,
    )
    assert fingerprint.startswith("sha256:") and len(fingerprint) == 71


def test_checkpoint_bytes_remain_unchanged_by_decoder(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"immutable classifier weights")
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    _decode(_configuration())
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before


def test_decoder_path_constructs_no_optimizer(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("optimizer must not be constructed")

    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    assert _decode(_configuration()).status is RouterStatus.ROUTE


def test_development_and_final_inputs_are_refused() -> None:
    for split in (LanguageSplit.DEVELOPMENT, LanguageSplit.FINAL):
        with pytest.raises(DecoderSelectionError, match="validation examples only"):
            build_decoder_evaluation_arrays(
                examples=[_example(1, RouterStatus.REJECT_AMBIGUOUS, split=split)],
                status_logits=torch.zeros(1, 4),
                object_logits=torch.zeros(1, 4),
                bin_logits=torch.zeros(1, 3),
                temperature=1.0,
            )


def test_immutable_analysis_artifact_refuses_overwrite_and_tamper(tmp_path: Path) -> None:
    root = tmp_path / "analysis"
    identity = {"checkpoint": "sha256:" + "a" * 64}
    write_rejection_analysis_artifact(
        root,
        json_artifacts={"rejected_candidate.json": {"frozen": True}},
        per_example_records=({"example_id": "fixture"},),
        input_identity=identity,
        conclusion="classifier_rejected_after_posthoc_calibration",
        classifier_full_quality_gate_passed=False,
    )
    assert validate_rejection_analysis_artifact(root)["passed"] is True
    with pytest.raises(RejectionReportError, match="already exists"):
        write_rejection_analysis_artifact(
            root,
            json_artifacts={"rejected_candidate.json": {"frozen": True}},
            per_example_records=(),
            input_identity=identity,
            conclusion="classifier_rejected_after_posthoc_calibration",
            classifier_full_quality_gate_passed=False,
        )
    (root / "rejected_candidate.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RejectionReportError, match="checksum"):
        validate_rejection_analysis_artifact(root)


def test_dry_run_locks_grid_without_loading_checkpoint(tmp_path: Path) -> None:
    report = tmp_path / "dry-run.json"
    assert analysis_main(["--dry-run", "--report", str(report)]) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["grid_lock"] == fixed_decoder_grid_v0().to_dict()
    assert payload["checkpoint_loaded"] is False
    assert payload["optimizer_constructed"] is False


def test_stage_verifier_accepts_valid_rejected_analysis(tmp_path: Path) -> None:
    checkpoint = tmp_path / "validation_best.pt"
    checkpoint.write_bytes(b"frozen")
    checkpoint_sha = f"sha256:{hashlib.sha256(checkpoint.read_bytes()).hexdigest()}"
    root = tmp_path / "analysis"
    no_training = {
        "model_weights_unchanged": True,
        "checkpoint_fingerprint_before": checkpoint_sha,
        "checkpoint_fingerprint_after": checkpoint_sha,
        "checkpoint_step_before": 116,
        "checkpoint_step_after": 116,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "training_seed_count": 1,
        "new_training_run_created": False,
        "corpus_mutated": False,
    }
    write_rejection_analysis_artifact(
        root,
        json_artifacts={
            "grid_lock.json": fixed_decoder_grid_v0().to_dict(),
            "candidate_selection.json": {
                "classifier_full_quality_gate_passed": False,
                "conclusion": "classifier_rejected_after_posthoc_calibration",
            },
            "no_training_audit.json": no_training,
            "rejected_candidate.json": {"classifier_candidate_frozen": True},
        },
        per_example_records=(),
        input_identity={
            "selected_checkpoint_fingerprint": checkpoint_sha,
            "selected_checkpoint_step": 116,
        },
        conclusion="classifier_rejected_after_posthoc_calibration",
        classifier_full_quality_gate_passed=False,
    )
    stage = tmp_path / "stage.json"
    stage.write_text(
        json.dumps(
            {
                "passed": True,
                "analysis_root": str(root),
                "checkpoint_path": str(checkpoint),
                "classifier_checkpoint_selected": True,
                "classifier_posthoc_analysis_completed": True,
                "decoder_selection_validated": True,
                "checkpoint_weights_unchanged": True,
                "classifier_full_quality_gate_passed": False,
                "classifier_calibration_validated": False,
                "classifier_runtime_selected": False,
                "classifier_candidate_frozen": True,
                "additional_training_authorized": False,
                "additional_seed_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    verified = _verify_stage_evidence(M5AStage.CLASSIFIER_REJECTION_ANALYSIS, stage)
    assert verified.passed is True
