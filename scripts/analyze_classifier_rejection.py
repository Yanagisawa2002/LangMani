#!/usr/bin/env python3
"""Run or independently verify the bounded M5A.1 validation-only analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch

from langmani.language.classifier_decoders import fixed_decoder_grid_v0
from langmani.language.decoder_selection import (
    decoder_runtime_fingerprint,
    select_fixed_grid_decoder,
)
from langmani.language.rejection_diagnostics import (
    audit_validation_data_contract,
    build_error_inventory,
    build_per_example_diagnostics,
    build_per_template_family_findings,
    collect_read_only_validation_logits,
    load_selected_checkpoint_read_only,
    load_validation_corpus_inputs,
)
from langmani.language.rejection_report import (
    validate_rejection_analysis_artifact,
    write_rejection_analysis_artifact,
)
from langmani.language.status_calibration import fit_status_temperature_comparison
from langmani.language.text_classifier import STATUS_LABELS

DEFAULT_CHECKPOINT = Path(
    "outputs/models/text-router/authoritative-runs/"
    "9e3ac659fa2b695c843650df35e3779741d94b3dd70b2aec52a429bc4b2edf49/"
    "checkpoints/validation_best.pt"
)
DEFAULT_CORPUS_ROOT = Path("outputs/datasets/m5a/langmani-language-corpus-v1")
DEFAULT_RECOVERY_REPORT = Path("outputs/diagnostics/m5a/stages/classifier-recovery-resume.json")
DEFAULT_OUTPUT_ROOT = Path("outputs/diagnostics/m5a/rejection-analysis")
DEFAULT_REPORT = Path("outputs/diagnostics/m5a/stages/classifier-rejection-analysis.json")
REPORT_SCHEMA = "langmani-m5a-classifier-rejection-analysis-command-v0"


class RejectionAnalysisCommandError(RuntimeError):
    """Raised when the M5A.1 command violates its read-only analysis contract."""


def _read_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RejectionAnalysisCommandError(f"could not read {label}: {error}") from error
    if not isinstance(value, dict):
        raise RejectionAnalysisCommandError(f"{label} must contain one JSON object")
    return cast(dict[str, object], value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _git_identity() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, not bool(status.strip())


def _write_report(path: Path, payload: Mapping[str, object]) -> None:
    path = path.resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _access_flags() -> dict[str, bool]:
    return {
        "development_accessed": False,
        "language_development_completed": False,
        "language_final_accessed": False,
        "control_evaluation_started": False,
        "control_final_accessed": False,
        "m3b_test_accessed": False,
        "historical_fresh_accessed": False,
        "m42_final_accessed": False,
        "smolvla_go": False,
    }


def _recovery_identity(
    recovery_report_path: Path,
) -> tuple[dict[str, object], str, str, int]:
    recovery = _read_object(recovery_report_path, label="classifier recovery report")
    fingerprint = recovery.get("selected_checkpoint_fingerprint")
    run_fingerprint = recovery.get("run_fingerprint")
    step = recovery.get("selected_step")
    if (
        recovery.get("passed") is not True
        or recovery.get("classifier_checkpoint_selected") is not True
        or not isinstance(fingerprint, str)
        or not fingerprint.startswith("sha256:")
        or not isinstance(run_fingerprint, str)
        or not run_fingerprint.startswith("sha256:")
        or isinstance(step, bool)
        or not isinstance(step, int)
        or step <= 0
        or recovery.get("classifier_training_seeds") != 1
    ):
        raise RejectionAnalysisCommandError("authoritative recovery identity is incomplete")
    return recovery, fingerprint, run_fingerprint, step


def _dry_run_payload() -> dict[str, object]:
    grid = fixed_decoder_grid_v0()
    return {
        "schema_version": REPORT_SCHEMA,
        "mode": "dry_run",
        "passed": True,
        "split": "validation",
        "grid_lock": grid.to_dict(),
        "checkpoint_loaded": False,
        "model_constructed": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "classifier_training_seeds": 1,
        "additional_training_authorized": False,
        "additional_seed_authorized": False,
        **_access_flags(),
    }


def _validate_analysis_artifact(
    *,
    output_root: Path,
    checkpoint: Path,
) -> dict[str, object]:
    validated = validate_rejection_analysis_artifact(output_root)
    root = Path(cast(str, validated["root"]))
    grid_payload = _read_object(root / "grid_lock.json", label="decoder grid lock")
    selection = _read_object(root / "candidate_selection.json", label="candidate selection")
    no_training = _read_object(root / "no_training_audit.json", label="no-training audit")
    input_identity = cast(Mapping[str, object], validated["input_identity"])
    expected_grid = fixed_decoder_grid_v0().to_dict()
    quality = selection.get("classifier_full_quality_gate_passed") is True
    conclusion = selection.get("conclusion")
    runtime_exists = (root / "classifier_routing_runtime.json").is_file()
    rejected_exists = (root / "rejected_candidate.json").is_file()
    checks = {
        "artifact_checksums_valid": validated.get("passed") is True,
        "fixed_grid_exact": grid_payload == expected_grid,
        "checkpoint_sha_unchanged": (
            input_identity.get("selected_checkpoint_fingerprint") == _sha256_file(checkpoint)
        ),
        "weights_unchanged": no_training.get("model_weights_unchanged") is True,
        "global_step_unchanged": (
            no_training.get("checkpoint_step_before")
            == no_training.get("checkpoint_step_after")
            == input_identity.get("selected_checkpoint_step")
        ),
        "optimizer_never_constructed": no_training.get("optimizer_constructed") is False,
        "optimizer_not_stepped": no_training.get("optimizer_steps") == 0,
        "single_seed_preserved": no_training.get("training_seed_count") == 1,
        "no_new_training_run": no_training.get("new_training_run_created") is False,
        "quality_conclusion_consistent": conclusion
        == (
            "classifier_promoted_after_posthoc_calibration"
            if quality
            else "classifier_rejected_after_posthoc_calibration"
        ),
        "runtime_artifact_conditional": runtime_exists is quality,
        "rejected_artifact_conditional": rejected_exists is (not quality),
        "forbidden_sources_sealed": all(no_training.get(name) is False for name in _access_flags()),
    }
    if not all(checks.values()):
        failures = [name for name, value in checks.items() if not value]
        raise RejectionAnalysisCommandError(
            f"independent analysis verification failed: {', '.join(failures)}"
        )
    return {
        "schema_version": REPORT_SCHEMA,
        "mode": "independent_verify",
        "passed": True,
        "analysis_root": str(root),
        "analysis_fingerprint": validated["analysis_fingerprint"],
        "checkpoint_path": str(checkpoint.resolve(strict=False)),
        "conclusion": conclusion,
        "classifier_full_quality_gate_passed": quality,
        "classifier_calibration_validated": quality,
        "classifier_runtime_selected": quality,
        "classifier_candidate_frozen": not quality,
        "classifier_checkpoint_selected": True,
        "classifier_posthoc_analysis_completed": True,
        "checkpoint_weights_unchanged": True,
        "decoder_selection_validated": True,
        "additional_training_authorized": False,
        "additional_seed_authorized": False,
        "physical_target_validated": False,
        "checks": checks,
        **_access_flags(),
    }


def _run_validation_analysis(args: argparse.Namespace) -> dict[str, object]:
    if args.split != "validation":
        raise RejectionAnalysisCommandError("M5A.1 permits the validation split only")
    git_commit, clean = _git_identity()
    if not clean:
        raise RejectionAnalysisCommandError("validation evidence requires a clean Git worktree")
    recovery, checkpoint_fingerprint, run_fingerprint, selected_step = _recovery_identity(
        args.recovery_report
    )
    checkpoint_before = _sha256_file(args.checkpoint)
    if checkpoint_before != checkpoint_fingerprint:
        raise RejectionAnalysisCommandError("selected checkpoint differs from recovery evidence")
    corpus = load_validation_corpus_inputs(args.corpus_root)
    run_root = args.checkpoint.parent.parent
    pre_resume_audit = _read_object(
        run_root / "recovery_pre_resume_audit.json",
        label="recovery pre-resume audit",
    )
    bundle = load_selected_checkpoint_read_only(
        args.checkpoint,
        expected_checkpoint_fingerprint=checkpoint_fingerprint,
        expected_run_fingerprint=run_fingerprint,
        expected_step=selected_step,
        local_files_only=args.local_files_only,
    )
    data_audit = audit_validation_data_contract(
        corpus,
        checkpoint_processor_state=bundle.processor_state,
        recovery_pre_resume_audit=pre_resume_audit,
    )
    if data_audit.get("passed") is not True:
        raise RejectionAnalysisCommandError("validation data-contract audit failed")
    evidence, no_training = collect_read_only_validation_logits(
        bundle,
        examples=corpus.validation_examples,
        batch_size=args.batch_size,
    )
    status_labels = torch.tensor(
        [STATUS_LABELS.index(example.expected_status.value) for example in evidence.examples],
        dtype=torch.long,
    )
    temperature = fit_status_temperature_comparison(
        evidence.status_logits,
        status_labels,
        evidence_split="validation",
    )
    grid = fixed_decoder_grid_v0()
    selection = select_fixed_grid_decoder(
        examples=evidence.examples,
        status_logits=evidence.status_logits,
        object_logits=evidence.object_logits,
        bin_logits=evidence.bin_logits,
        grid=grid,
        temperature_values={
            "identity": 1.0,
            "validation_temperature": temperature.fitted_temperature,
        },
        temperature_comparison_fingerprint=temperature.comparison_fingerprint,
    )
    per_example = build_per_example_diagnostics(
        evidence,
        baseline_configuration=selection.baseline_configuration,
        selected_configuration=selection.selected_configuration,
    )
    no_training.update(
        {
            "checkpoint_fingerprint_before": checkpoint_before,
            "checkpoint_fingerprint_after": _sha256_file(args.checkpoint),
            "checkpoint_sha_unchanged": _sha256_file(args.checkpoint) == checkpoint_before,
            "corpus_mutated": False,
            "additional_training_authorized": False,
            "additional_seed_authorized": False,
            **_access_flags(),
        }
    )
    if no_training["checkpoint_sha_unchanged"] is not True:
        raise RejectionAnalysisCommandError("selected checkpoint changed during analysis")
    runtime: dict[str, object] | None = None
    if selection.classifier_full_quality_gate_passed:
        runtime_fingerprint = decoder_runtime_fingerprint(
            classifier_checkpoint_fingerprint=checkpoint_fingerprint,
            tokenizer_fingerprint=bundle.tokenizer_fingerprint,
            processor_fingerprint=bundle.processor_fingerprint,
            selection=selection,
            git_commit=git_commit,
            corpus_fingerprint=corpus.corpus_fingerprint,
            validation_split_fingerprint=corpus.validation_split_fingerprint,
        )
        runtime = {
            "schema_version": "langmani-m5a-classifier-routing-runtime-v0",
            "runtime_fingerprint": runtime_fingerprint,
            "selected_classifier_checkpoint_fingerprint": checkpoint_fingerprint,
            "tokenizer_fingerprint": bundle.tokenizer_fingerprint,
            "processor_fingerprint": bundle.processor_fingerprint,
            "decoder": cast(object, selection.selected_configuration.to_dict()),
            "validation_metrics": cast(object, selection.selected_metrics.to_dict()),
            "rejection_semantics": "rejected_decisions_never_contain_executable_taskspec_v0",
            "git_commit": git_commit,
            "corpus_fingerprint": corpus.corpus_fingerprint,
            "validation_split_fingerprint": corpus.validation_split_fingerprint,
        }
    rejected = {
        "schema_version": "langmani-m5a-rejected-classifier-baseline-v0",
        "conclusion": selection.conclusion,
        "classifier_candidate_frozen": True,
        "additional_training_authorized": False,
        "additional_seed_authorized": False,
        "selected_classifier_checkpoint_fingerprint": checkpoint_fingerprint,
        "selection_fingerprint": selection.selection_fingerprint,
    }
    json_artifacts: dict[str, Mapping[str, object]] = {
        "grid_lock.json": grid.to_dict(),
        "temperature_calibration.json": temperature.to_dict(),
        "data_contract_audit.json": data_audit,
        "baseline_analysis.json": selection.baseline_metrics.to_dict(),
        "error_inventory.json": build_error_inventory(per_example),
        "per_template_family.json": build_per_template_family_findings(per_example),
        "candidate_selection.json": selection.to_dict(),
        "no_training_audit.json": no_training,
        "source_recovery_identity.json": {
            "run_fingerprint": run_fingerprint,
            "selected_epoch": recovery.get("selected_epoch"),
            "selected_step": selected_step,
            "selected_checkpoint_fingerprint": checkpoint_fingerprint,
            "selected_checkpoint_reload_maximum_absolute_error": recovery.get(
                "selected_checkpoint_reload_maximum_absolute_error"
            ),
            "training_seed": recovery.get("training_seed"),
            "classifier_training_seeds": recovery.get("classifier_training_seeds"),
        },
    }
    if runtime is None:
        json_artifacts["rejected_candidate.json"] = rejected
    else:
        json_artifacts["classifier_routing_runtime.json"] = runtime
    input_identity = {
        "selected_checkpoint_fingerprint": checkpoint_fingerprint,
        "checkpoint_path": str(args.checkpoint.resolve(strict=False)),
        "selected_checkpoint_step": selected_step,
        "classifier_run_fingerprint": run_fingerprint,
        "corpus_fingerprint": corpus.corpus_fingerprint,
        "archive_fingerprint": corpus.archive_fingerprint,
        "train_split_fingerprint": corpus.train_split_fingerprint,
        "validation_split_fingerprint": corpus.validation_split_fingerprint,
        "grid_fingerprint": grid.grid_fingerprint,
        "git_commit": git_commit,
        "analysis_split": "validation",
    }
    artifact = write_rejection_analysis_artifact(
        args.output_root,
        json_artifacts=json_artifacts,
        per_example_records=per_example,
        input_identity=input_identity,
        conclusion=selection.conclusion,
        classifier_full_quality_gate_passed=selection.classifier_full_quality_gate_passed,
    )
    return {
        "schema_version": REPORT_SCHEMA,
        "mode": "validation_only",
        "passed": True,
        "split": "validation",
        "analysis_root": artifact["root"],
        "analysis_fingerprint": artifact["analysis_fingerprint"],
        "conclusion": selection.conclusion,
        "classifier_posthoc_analysis_completed": True,
        "classifier_full_quality_gate_passed": selection.classifier_full_quality_gate_passed,
        "classifier_calibration_validated": selection.classifier_full_quality_gate_passed,
        "classifier_runtime_selected": runtime is not None,
        "classifier_candidate_frozen": runtime is None,
        "classifier_checkpoint_selected": True,
        "checkpoint_weights_unchanged": no_training["model_weights_unchanged"],
        "decoder_selection_validated": True,
        "additional_training_authorized": False,
        "additional_seed_authorized": False,
        "physical_target_validated": False,
        "selected_checkpoint_fingerprint": checkpoint_fingerprint,
        "selected_checkpoint_step": selected_step,
        "classifier_run_fingerprint": run_fingerprint,
        "grid_fingerprint": grid.grid_fingerprint,
        "selection_fingerprint": selection.selection_fingerprint,
        "runtime_fingerprint": None if runtime is None else runtime["runtime_fingerprint"],
        **_access_flags(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--validation-only", action="store_true")
    mode.add_argument("--independent-verify", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--recovery-report", type=Path, default=DEFAULT_RECOVERY_REPORT)
    parser.add_argument("--split", choices=("validation",), default="validation")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.dry_run:
            report = _dry_run_payload()
        elif args.independent_verify:
            report = _validate_analysis_artifact(
                output_root=args.output_root,
                checkpoint=args.checkpoint,
            )
        else:
            report = _run_validation_analysis(args)
    except Exception as error:  # noqa: BLE001 - outer boundary preserves exact failure
        traceback.print_exc()
        report = {
            "schema_version": REPORT_SCHEMA,
            "mode": (
                "dry_run"
                if args.dry_run
                else "independent_verify"
                if args.independent_verify
                else "validation_only"
            ),
            "passed": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "classifier_posthoc_analysis_completed": False,
            "additional_training_authorized": False,
            "additional_seed_authorized": False,
            **_access_flags(),
        }
    _write_report(args.report, report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    sys.exit(main())
