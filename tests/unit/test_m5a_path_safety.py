from __future__ import annotations

from pathlib import Path

import pytest

from environment import verify_m5a
from scripts import build_language_corpus as corpus_cli
from scripts import evaluate_language_routers as evaluate_cli
from scripts import run_language_control as control_cli
from scripts import train_text_router as train_cli


def test_corpus_writer_rejects_historical_generated_roots(tmp_path: Path) -> None:
    with pytest.raises(corpus_cli.CorpusCommandError, match="historical"):
        corpus_cli._safe_paths(
            corpus_cli.PROJECT_ROOT / "outputs" / "datasets" / "m3b" / "unsafe",
            tmp_path / "corpus-report.json",
        )


def test_classifier_writer_rejects_historical_evidence_root(tmp_path: Path) -> None:
    with pytest.raises(train_cli.TextRouterCommandError, match="cannot overlap"):
        train_cli._safe_paths(
            train_cli.PROJECT_ROOT / "outputs" / "diagnostics" / "m4" / "unsafe",
            tmp_path / "classifier-report.json",
        )


def _evaluation_args(tmp_path: Path):  # type: ignore[no-untyped-def]
    corpus = tmp_path / "corpus"
    classifier = tmp_path / "classifier"
    corpus.mkdir()
    classifier.mkdir()
    return evaluate_cli.parse_args(
        [
            "--fixture",
            "--corpus-root",
            str(corpus),
            "--classifier-artifact",
            str(classifier),
            "--llm-model-id",
            "fixture/local",
            "--llm-model-revision",
            "a" * 40,
            "--llm-tokenizer-revision",
            "b" * 40,
            "--llm-license",
            "fixture",
            "--device",
            "cpu",
            "--output-root",
            str(tmp_path / "language-evidence"),
            "--report",
            str(tmp_path / "language-report.json"),
        ]
    )


@pytest.mark.parametrize("input_name", ["corpus_root", "classifier_artifact"])
def test_language_evaluation_output_rejects_each_immutable_input(
    tmp_path: Path,
    input_name: str,
) -> None:
    args = _evaluation_args(tmp_path)
    args.output_root = getattr(args, input_name) / "unsafe-output"

    with pytest.raises(evaluate_cli.LanguageRouterEvaluationCommandError, match="cannot overlap"):
        evaluate_cli._safe_paths(args)


def test_language_evaluation_rejects_overlapping_inputs(tmp_path: Path) -> None:
    args = _evaluation_args(tmp_path)
    nested = args.corpus_root / "classifier"
    nested.mkdir()
    args.classifier_artifact = nested

    with pytest.raises(evaluate_cli.LanguageRouterEvaluationCommandError, match="inputs cannot"):
        evaluate_cli._safe_paths(args)


def test_language_evaluation_rejects_unconfigured_historical_root(tmp_path: Path) -> None:
    args = _evaluation_args(tmp_path)
    args.output_root = (
        evaluate_cli.PROJECT_ROOT / "outputs" / "models" / "act-factor-film" / "unsafe"
    )

    with pytest.raises(evaluate_cli.LanguageRouterEvaluationCommandError, match="cannot overlap"):
        evaluate_cli._safe_paths(args)


def _control_args(tmp_path: Path):  # type: ignore[no-untyped-def]
    return control_cli.parse_args(
        [
            "--control-schedule",
            str(tmp_path / "inputs" / "schedule.json"),
            "--classifier-artifact-root",
            str(tmp_path / "inputs" / "classifier"),
            "--router-evaluation-evidence-root",
            str(tmp_path / "inputs" / "router-evidence"),
            "--llm-model-id",
            "fixture/local",
            "--llm-model-revision",
            "a" * 40,
            "--llm-tokenizer-revision",
            "b" * 40,
            "--llm-license",
            "fixture",
            "--device",
            "cpu",
            "--dataset-root",
            str(tmp_path / "inputs" / "dataset"),
            "--checkpoint-root",
            str(tmp_path / "inputs" / "checkpoints"),
            "--runtime-selection",
            str(tmp_path / "inputs" / "runtime-selection.json"),
            "--m4-diagnostics-root",
            str(tmp_path / "inputs" / "legacy-m4"),
            "--output-root",
            str(tmp_path / "control"),
            "--report",
            str(tmp_path / "control-report.json"),
            "--stage",
            "one_scene_control_smoke",
            "--dry-run",
        ]
    )


@pytest.mark.parametrize(
    "input_name",
    [
        "control_schedule",
        "classifier_artifact_root",
        "router_evaluation_evidence_root",
        "dataset_root",
        "checkpoint_root",
        "runtime_selection",
        "m4_diagnostics_root",
    ],
)
def test_control_output_rejects_each_immutable_input(
    tmp_path: Path,
    input_name: str,
) -> None:
    args = _control_args(tmp_path)
    args.output_root = getattr(args, input_name) / "unsafe-output"

    with pytest.raises(control_cli.LanguageControlCommandError, match="cannot overlap"):
        control_cli._safe_paths(args)


def test_control_rejects_overlapping_immutable_inputs(tmp_path: Path) -> None:
    args = _control_args(tmp_path)
    args.router_evaluation_evidence_root = args.classifier_artifact_root / "router-evidence"

    with pytest.raises(control_cli.LanguageControlCommandError, match="inputs cannot overlap"):
        control_cli._safe_paths(args)


def test_control_rejects_unconfigured_historical_root(tmp_path: Path) -> None:
    args = _control_args(tmp_path)
    args.output_root = control_cli.PROJECT_ROOT / "outputs" / "datasets" / "m3a" / "unsafe"

    with pytest.raises(control_cli.LanguageControlCommandError, match="cannot overlap"):
        control_cli._safe_paths(args)


def _verifier_args(tmp_path: Path):  # type: ignore[no-untyped-def]
    verifier_root = tmp_path / "verification"
    return verify_m5a.parse_args(
        [
            "--target-development",
            "--output-root",
            str(verifier_root),
            "--m43-independent-verification",
            str(tmp_path / "inputs" / "m43" / "verification.json"),
            "--llm-model-id",
            "fixture/local",
            "--llm-model-revision",
            "a" * 40,
            "--llm-tokenizer-revision",
            "b" * 40,
            "--llm-license",
            "fixture",
            "--llm-license-reviewed",
            "--llm-model-card-reviewed",
            "--local-files-only",
            "--corpus-root",
            str(tmp_path / "corpus"),
            "--classifier-output-root",
            str(tmp_path / "classifier"),
            "--language-evidence-root",
            str(verifier_root / "language"),
            "--control-evidence-root",
            str(verifier_root / "control"),
            "--dataset-root",
            str(tmp_path / "inputs" / "dataset"),
            "--checkpoint-root",
            str(tmp_path / "inputs" / "checkpoints"),
            "--runtime-selection",
            str(tmp_path / "inputs" / "runtime-selection.json"),
            "--m4-diagnostics-root",
            str(tmp_path / "inputs" / "legacy-m4"),
        ]
    )


def test_verifier_preserves_its_managed_child_layout(tmp_path: Path) -> None:
    args = _verifier_args(tmp_path)

    layout = verify_m5a._validate_target_paths(args)

    assert layout.language_evidence_root.parent == layout.verifier_output_root
    assert layout.control_evidence_root.parent == layout.verifier_output_root


@pytest.mark.parametrize(
    "input_name",
    [
        "m43_independent_verification",
        "dataset_root",
        "checkpoint_root",
        "runtime_selection",
        "m4_diagnostics_root",
    ],
)
def test_verifier_managed_output_rejects_each_immutable_input(
    tmp_path: Path,
    input_name: str,
) -> None:
    args = _verifier_args(tmp_path)
    args.corpus_root = getattr(args, input_name) / "unsafe-corpus-output"

    with pytest.raises(verify_m5a.M5ATargetVerificationError, match="cannot overlap"):
        verify_m5a._validate_target_paths(args)


def test_verifier_rejects_overlapping_managed_outputs(tmp_path: Path) -> None:
    args = _verifier_args(tmp_path)
    args.control_evidence_root = args.language_evidence_root / "control"

    with pytest.raises(verify_m5a.M5ATargetVerificationError, match="output roots cannot overlap"):
        verify_m5a._validate_target_paths(args)


@pytest.mark.parametrize(
    "output_name",
    [
        "corpus_root",
        "classifier_output_root",
        "language_evidence_root",
        "control_evidence_root",
    ],
)
def test_verifier_managed_outputs_reject_repository_content(
    tmp_path: Path,
    output_name: str,
) -> None:
    args = _verifier_args(tmp_path)
    setattr(args, output_name, verify_m5a.PROJECT_ROOT / "src" / f"unsafe-{output_name}")

    with pytest.raises(
        verify_m5a.M5ATargetVerificationError,
        match="repository or historical",
    ):
        verify_m5a._validate_target_paths(args)


def test_verifier_rejects_source_overlap_before_reusing_stage_report(tmp_path: Path) -> None:
    args = _verifier_args(tmp_path)
    args.corpus_root = verify_m5a.PROJECT_ROOT / "src" / "unsafe-reused-corpus"
    stage_report = args.output_root / "stages" / "corpus.json"
    stage_report.parent.mkdir(parents=True)
    stage_report.write_text('{"passed":true}\n', encoding="utf-8")
    runner_calls: list[object] = []

    with pytest.raises(
        verify_m5a.M5ATargetVerificationError,
        match="repository or historical",
    ):
        verify_m5a._execute_target_development(
            args,
            report=verify_m5a.TargetDevelopmentReport(),
            runner=lambda command: runner_calls.append(command) or 0,
        )

    assert runner_calls == []
    assert stage_report.read_text(encoding="utf-8") == '{"passed":true}\n'


def test_verifier_rejects_overlapping_immutable_inputs(tmp_path: Path) -> None:
    args = _verifier_args(tmp_path)
    args.checkpoint_root = args.dataset_root / "checkpoints"

    with pytest.raises(verify_m5a.M5ATargetVerificationError, match="inputs cannot overlap"):
        verify_m5a._validate_target_paths(args)


def test_verifier_output_rejects_historical_dataset() -> None:
    with pytest.raises(RuntimeError, match="historical artifacts"):
        verify_m5a._validate_output_root(
            verify_m5a.PROJECT_ROOT / "outputs" / "datasets" / "m3b" / "unsafe"
        )
