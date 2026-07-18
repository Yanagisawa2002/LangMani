from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from environment import verify_m5a

_GIT_COMMIT = "f" * 40
_DEPENDENCIES = {
    "python": "3.12.10",
    "torch": "2.11.0",
    "cuda_runtime": "12.8",
    "transformers": "5.4.0",
    "tokenizers": "0.22.2",
    "huggingface_hub": "1.4.1",
    "safetensors": "0.7.0",
}


def _digest(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode()).hexdigest()}"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _m43_report(*, passed: bool = True) -> dict[str, object]:
    required_true = {
        "implementation_validated",
        "semantic_audit_completed",
        "factor_film_implementation_validated",
        "factor_film_fixture_training_validated",
        "factor_film_training_completed",
        "factor_film_checkpoints_complete",
        "factor_film_checkpoint_selected",
        "validation_only_selection_validated",
        "factor_film_reload_validated",
        "development_benchmark_completed",
        "post_grasp_analysis_completed",
        "first_interaction_analysis_completed",
        "raw_action_metrics_validated",
        "runtime_action_metrics_validated",
        "physical_target_validated",
    }
    return {
        "schema_version": verify_m5a.M43B_INDEPENDENT_SCHEMA,
        "passed": passed,
        **{key: True for key in required_true},
        "development_quality_gate_passed": False,
        "final_benchmark_authorized": False,
        "final_schedule_accessed": False,
        "test_split_accessed": False,
        "fresh_seed_accessed": False,
        "smolvla_go": False,
        "evaluation_evidence_fingerprint": _digest("m43-evidence"),
    }


def _language_metric_payload(
    *,
    split: str,
    full_accuracy: float = 1.0,
    structured_malformed_rate: float = 0.0,
) -> dict[str, object]:
    counts = {"validation": (300, 180, 120), "development": (420, 240, 180)}[split]
    shared = {
        "object_accuracy": 1.0,
        "bin_accuracy": 1.0,
        "false_route_rate": 0.0,
        "ambiguous_rejection_recall": 1.0,
        "unsupported_rejection_recall": 1.0,
        "malformed_rejection_recall": 1.0,
    }
    return {
        "summary": {
            "split": split,
            "example_count": counts[0],
            "routeable_count": counts[1],
            "rejected_count": counts[2],
            "valid_full_task_accuracy": full_accuracy,
            "deterministic_repeatability": 1.0,
            **shared,
        },
        "supplemental_metrics": {
            "schema_valid_output_rate": 1.0 - structured_malformed_rate,
            "structured_output_malformed_rate": structured_malformed_rate,
            "routeable_denominator": counts[1],
            "rejected_denominator": counts[2],
            **shared,
        },
    }


def _failure_counts(*, successes: int) -> dict[str, int]:
    result = {value.value: 0 for value in verify_m5a.FailureAttribution}
    result["routing_correct_control_success"] = successes
    result["routing_correct_control_failure"] = 72 - successes
    return result


def _control_summary(*, successes: int) -> dict[str, object]:
    return {
        "episode_count": 72,
        "dispatched_count": 72,
        "safe_rejection_count": 0,
        "routing_correct_count": 72,
        "routing_false_rejection_count": 0,
        "routing_wrong_object_count": 0,
        "routing_wrong_bin_count": 0,
        "controller_task_success_count": successes,
        "routing_correct_control_success_count": successes,
        "routing_correct_control_failure_count": 72 - successes,
        "end_to_end_success_count": successes,
        "wrong_object_in_target_bin_count": 0,
        "target_in_wrong_bin_count": 0,
        "target_off_table_count": 0,
        "timeout_count": 72 - successes,
        "wrong_object_interaction_available": True,
        "wrong_object_interaction_count": 0,
        "invalid_action_count": 0,
        "malformed_action_count": 0,
        "nonfinite_action_count": 0,
        "dispatch_after_rejection_count": 0,
        "m2_expert_call_count": 0,
        "failure_attribution_counts": _failure_counts(successes=successes),
        "raw_action_violation_count": 0,
        "arm_projected_component_count": 0,
        "gripper_projected_component_count": 0,
        "infrastructure_failure_count": 0,
    }


def _language_report(
    *,
    corpus_fingerprint: str,
    artifact_fingerprint: str,
    language_development: dict[str, object],
    language_final: dict[str, object],
    evidence_root: Path,
    llm_model_id: str,
    model_revision: str,
    tokenizer_revision: str,
    llm_license: str,
    maximum_new_tokens: int = 128,
    repeat_count: int = 2,
    local_files_only: bool = True,
    classifier_accuracy: float = 1.0,
) -> dict[str, object]:
    router_identities = {
        "rule": {"router_name": "RuleRouterV0", "router_fingerprint": _digest("rule")},
        "classifier": {
            "router_name": "FactorizedTextClassifierV0",
            "artifact_fingerprint": artifact_fingerprint,
            "router_fingerprint": _digest("classifier-router"),
        },
        "llm": {
            "router_name": "StructuredLocalLLMRouterV0",
            "router_fingerprint": _digest("llm-router"),
            "declared_license": llm_license,
            "official_model_card_license_reviewed": True,
            "parameter_count": 500_000_000,
            "generator": "TransformersLocalTextGenerator",
            "prompt_example_split": "train",
            "prompt_fingerprint": _digest("prompt"),
            "config": {
                "model_id": llm_model_id,
                "model_revision": model_revision,
                "tokenizer_revision": tokenizer_revision,
                "dtype": "bfloat16",
                "quantization": "none",
                "maximum_new_tokens": maximum_new_tokens,
                "maximum_format_repair_attempts": 1,
            },
        },
    }
    comparison = {
        split: {
            "rule": _language_metric_payload(split=split),
            "classifier": _language_metric_payload(split=split, full_accuracy=classifier_accuracy),
            "llm": _language_metric_payload(split=split),
        }
        for split in ("validation", "development")
    }
    return {
        "schema_version": verify_m5a.LANGUAGE_COMMAND_SCHEMA,
        "passed": True,
        "mode": "target_development",
        "evaluation_fingerprint": _digest("language-evaluation"),
        "artifact_fingerprint": _digest("language-artifact"),
        "evidence_root": str(evidence_root),
        "corpus_fingerprint": corpus_fingerprint,
        "router_identities": router_identities,
        "repeat_count": repeat_count,
        "local_files_only": local_files_only,
        "language_development_schedule": language_development,
        "language_final_lock": {
            "schedule_id": language_final["schedule_id"],
            "schedule_fingerprint": language_final["schedule_fingerprint"],
            "sealed": True,
            "texts_materialized": False,
        },
        "comparison": comparison,
        "evaluation_order": ["validation", "development"],
        "validation_completed_before_development": True,
        "classifier_checkpoint_selected": True,
        "classifier_calibration_validated": True,
        "llm_router_loaded": True,
        "llm_router_fixture_loaded": False,
        "llm_prompt_locked": True,
        "language_validation_completed": True,
        "language_development_completed": True,
        "language_fixture_completed": False,
        "target_development_executed": True,
        "physical_target_validated": False,
        "final_benchmark_authorized": False,
        "language_final_accessed": False,
        "control_final_accessed": False,
        "m42_final_accessed": False,
        "test_split_accessed": False,
        "historical_fresh_accessed": False,
        "controller_dispatched": False,
        "environment_reset_called": False,
        "environment_step_count": 0,
        "m2_expert_call_count": 0,
        "smolvla_go": False,
        "git_state": {
            "commit": _GIT_COMMIT,
            "baseline_tracked": True,
            "dirty": False,
        },
        "dependencies": _DEPENDENCIES,
    }


def _control_report(
    *,
    corpus_fingerprint: str,
    control_schedule_fingerprint: str,
    artifact_fingerprint: str,
    llm_model_id: str,
    model_revision: str,
    tokenizer_revision: str,
    maximum_new_tokens: int = 128,
    router_evaluation_evidence_root: Path,
    control_evidence_root: Path | None = None,
) -> dict[str, object]:
    registry = verify_m5a.build_fixture_controller_registry()
    summaries = {
        "oracle": _control_summary(successes=72),
        "rule": _control_summary(successes=68),
        "classifier": _control_summary(successes=70),
        "llm": _control_summary(successes=70),
    }
    return {
        "schema_version": verify_m5a.CONTROL_COMPLETION_SCHEMA,
        "passed": True,
        "physical_execution": True,
        "dry_run": False,
        "schedule_fingerprint": control_schedule_fingerprint,
        "corpus_fingerprint": corpus_fingerprint,
        "controller_registry": registry.to_dict(),
        "controller_registry_fingerprint": registry.registry_fingerprint,
        "classifier_identity": {"artifact_fingerprint": artifact_fingerprint},
        "llm_config": {
            "model_id": llm_model_id,
            "model_revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "dtype": "bfloat16",
            "maximum_new_tokens": maximum_new_tokens,
        },
        "llm_prompt_fingerprint": _digest("prompt"),
        "router_evaluation_identity": {
            "evaluation_fingerprint": _digest("language-evaluation"),
            "artifact_fingerprint": _digest("language-artifact"),
            "evidence_root": str(router_evaluation_evidence_root.resolve()),
        },
        "router_order": ["oracle", "rule", "classifier", "llm"],
        "run_fingerprint": _digest("control-run"),
        "record_set_fingerprint": _digest("control-record-set"),
        "completion_fingerprint": _digest("control-completion"),
        "evidence_root": (
            str(control_evidence_root.resolve()) if control_evidence_root is not None else None
        ),
        "completed_episode_atoms": 288,
        "expected_episode_atoms": 288,
        "summaries": summaries,
        "wrong_object_interaction_available": True,
        "wrong_object_interaction_count": 0,
        "invalid_action_count": 0,
        "malformed_action_count": 0,
        "nonfinite_action_count": 0,
        "dispatch_after_rejection_count": 0,
        "rejection_noop_probe_validated": True,
        "zero_dispatch_after_rejection_validated": True,
        "m2_expert_call_count": 0,
        "m2_expert_free_validated": True,
        "language_final_accessed": False,
        "control_final_accessed": False,
        "m42_final_accessed": False,
        "test_split_accessed": False,
        "historical_fresh_accessed": False,
        "smolvla_go": False,
    }


def test_non_target_verifier_reports_truthful_fixture_flags(tmp_path) -> None:
    output_root = tmp_path / "m5a-verification"

    assert verify_m5a.main(["--output-root", str(output_root)]) == 0

    payload = json.loads((output_root / "verification.json").read_text(encoding="utf-8"))
    expected_true = {
        "implementation_validated",
        "corpus_validated",
        "split_isolation_validated",
        "rule_router_validated",
        "classifier_fixture_validated",
        "llm_router_fixture_validated",
        "controller_registry_validated",
        "final_schedules_locked",
        "passed",
    }
    expected_false = {
        "classifier_training_completed",
        "classifier_checkpoint_selected",
        "classifier_calibration_validated",
        "llm_router_loaded",
        "llm_prompt_locked",
        "language_development_completed",
        "oracle_control_development_completed",
        "predicted_control_development_completed",
        "control_development_completed",
        "rejection_noop_probe_validated",
        "failure_attribution_validated",
        "development_quality_gate_passed",
        "final_benchmark_authorized",
        "language_final_accessed",
        "control_final_accessed",
        "m42_final_accessed",
        "test_split_accessed",
        "historical_fresh_accessed",
        "smolvla_go",
        "physical_target_validated",
    }
    assert payload["schema_version"] == verify_m5a.REPORT_SCHEMA_VERSION
    assert payload["verification_mode"] == "non_target_structural_fixture"
    assert all(payload[name] is True for name in expected_true)
    assert all(payload[name] is False for name in expected_false)
    assert len(payload["schedule_fingerprints"]) == 4
    assert payload["corpus_fingerprint"].startswith("sha256:")
    assert payload["controller_registry_fingerprint"].startswith("sha256:")
    assert all(check["status"] == "pass" for check in payload["checks"])


def test_verifier_script_imports_project_commands_from_direct_entrypoint() -> None:
    completed = subprocess.run(
        [sys.executable, str(Path(verify_m5a.__file__)), "--help"],
        cwd=verify_m5a.PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--verify-stage" in completed.stdout


def test_non_target_report_cannot_claim_target_completion() -> None:
    report = verify_m5a.Report(
        implementation_validated=True,
        corpus_validated=True,
        split_isolation_validated=True,
        rule_router_validated=True,
        classifier_fixture_validated=True,
        llm_router_fixture_validated=True,
        controller_registry_validated=True,
        final_schedules_locked=True,
        classifier_training_completed=True,
    )

    assert report.passed is False


def test_verifier_output_cannot_overlap_source_tree() -> None:
    with pytest.raises(RuntimeError, match="must not overlap source or Git"):
        verify_m5a._validate_output_root(verify_m5a.PROJECT_ROOT / "tests" / "unsafe-m5a")


def _argument(command: tuple[str, ...], name: str) -> str:
    return command[command.index(name) + 1]


def _target_arguments(tmp_path: Path, *, m43_path: Path) -> list[str]:
    return [
        "--target-development",
        "--output-root",
        str(tmp_path / "verification"),
        "--m43-independent-verification",
        str(m43_path),
        "--llm-model-id",
        "fixture/local-instruct",
        "--llm-model-revision",
        "a" * 40,
        "--llm-tokenizer-revision",
        "b" * 40,
        "--llm-license",
        "Apache-2.0",
        "--llm-license-reviewed",
        "--llm-model-card-reviewed",
        "--local-files-only",
        "--corpus-root",
        str(tmp_path / "corpus"),
        "--classifier-output-root",
        str(tmp_path / "classifier"),
        "--language-evidence-root",
        str(tmp_path / "language-evidence"),
        "--control-evidence-root",
        str(tmp_path / "control-evidence"),
        "--dataset-root",
        str(tmp_path / "dataset"),
        "--checkpoint-root",
        str(tmp_path / "checkpoints"),
        "--m4-diagnostics-root",
        str(tmp_path / "m4"),
        "--runtime-selection",
        str(tmp_path / "runtime-selection.json"),
    ]


class _TargetStageRunner:
    def __init__(self, *, verifier_root: Path) -> None:
        self.verifier_root = verifier_root
        self.commands: list[tuple[str, ...]] = []
        self.artifact_fingerprint = _digest("classifier-artifact")

    def __call__(self, values: object) -> int:
        command = tuple(cast(tuple[str, ...], values))
        self.commands.append(command)
        script = Path(command[1]).name
        report_path = Path(_argument(command, "--report"))
        schedule = json.loads(
            (self.verifier_root / "target-development-schedules.json").read_text(encoding="utf-8")
        )
        corpus = verify_m5a.build_language_corpus()
        if script == "build_language_corpus.py":
            output_root = Path(_argument(command, "--output-root"))
            output_root.mkdir(parents=True, exist_ok=True)
            _write_json(
                report_path,
                {
                    "schema_version": verify_m5a.CORPUS_COMMAND_SCHEMA,
                    "passed": True,
                    "dry_run": False,
                    "corpus_fingerprint": corpus.manifest.corpus_fingerprint,
                    "corpus_manifest": corpus.manifest.to_dict(),
                    "language_development_schedule": schedule["language_development"],
                    "language_final_schedule": schedule["language_final"],
                    "corpus_validated": True,
                    "split_isolation_validated": True,
                    "final_schedule_locked": True,
                    "language_final_accessed": False,
                    "archive_fingerprint": _digest("corpus-archive"),
                    "output_root": str(output_root.resolve()),
                },
            )
            return 0
        if script == "train_text_router.py":
            output_root = Path(_argument(command, "--output-root"))
            artifact_root = output_root / ("c" * 64)
            artifact_root.mkdir(parents=True, exist_ok=True)
            _write_json(
                artifact_root / "complete.json",
                {"passed": True, "artifact_fingerprint": self.artifact_fingerprint},
            )
            _write_json(
                report_path,
                {
                    "schema_version": verify_m5a.CLASSIFIER_COMMAND_SCHEMA,
                    "passed": True,
                    "mode": "target_development",
                    "classifier_training_completed": True,
                    "classifier_fixture_completed": False,
                    "classifier_checkpoint_selected": True,
                    "classifier_calibration_validated": True,
                    "artifact_reload_validated": True,
                    "cuda_training_validated": True,
                    "development_accessed": False,
                    "language_final_accessed": False,
                    "control_final_accessed": False,
                    "m42_final_accessed": False,
                    "test_split_accessed": False,
                    "historical_fresh_accessed": False,
                    "smolvla_go": False,
                    "physical_target_validated": False,
                    "run_fingerprint": _digest("classifier-run"),
                    "artifact_root": str(artifact_root.resolve()),
                    "artifact_fingerprint": self.artifact_fingerprint,
                    "run_evidence": {
                        "schema_version": verify_m5a.CLASSIFIER_RUN_EVIDENCE_SCHEMA,
                        "mode": "target_development",
                        "corpus_manifest": corpus.manifest.to_dict(),
                        "git_commit": _GIT_COMMIT,
                        "git_state": {
                            "commit": _GIT_COMMIT,
                            "baseline_tracked": True,
                            "dirty": False,
                        },
                        "dependencies": _DEPENDENCIES,
                        "model_id": verify_m5a.TEXT_CLASSIFIER_MODEL_ID,
                        "model_revision": verify_m5a.TEXT_CLASSIFIER_MODEL_REVISION,
                        "tokenizer_revision": verify_m5a.TEXT_CLASSIFIER_TOKENIZER_REVISION,
                    },
                    "calibration_selection": {
                        "temperature": {"value": 1.0},
                        "threshold": {"threshold": 0.9},
                    },
                },
            )
            return 0
        if script == "evaluate_language_routers.py":
            output_root = Path(_argument(command, "--output-root"))
            evidence_root = output_root / ("d" * 64)
            evidence_root.mkdir(parents=True, exist_ok=True)
            language_report = _language_report(
                corpus_fingerprint=corpus.manifest.corpus_fingerprint,
                artifact_fingerprint=self.artifact_fingerprint,
                language_development=schedule["language_development"],
                language_final=schedule["language_final"],
                evidence_root=evidence_root,
                llm_model_id=_argument(command, "--llm-model-id"),
                model_revision=_argument(command, "--llm-model-revision"),
                tokenizer_revision=_argument(command, "--llm-tokenizer-revision"),
                llm_license=_argument(command, "--llm-license"),
                maximum_new_tokens=int(_argument(command, "--llm-maximum-new-tokens")),
                repeat_count=int(_argument(command, "--repeat-count")),
                local_files_only="--local-files-only" in command,
            )
            _write_json(report_path, language_report)
            immutable_result = dict(language_report)
            immutable_result["schema_version"] = (
                "langmani-m5a-language-router-evaluation-evidence-v1"
            )
            immutable_result.pop("artifact_fingerprint")
            immutable_result.pop("evidence_root")
            _write_json(evidence_root / "fixture-result.json", immutable_result)
            return 0
        if script == "run_language_control.py":
            output_root = Path(_argument(command, "--output-root"))
            control_evidence_root = output_root / "evidence" / ("e" * 64)
            control_evidence_root.mkdir(parents=True, exist_ok=True)
            control_report = _control_report(
                corpus_fingerprint=corpus.manifest.corpus_fingerprint,
                control_schedule_fingerprint=schedule["control_development"][
                    "schedule_fingerprint"
                ],
                artifact_fingerprint=self.artifact_fingerprint,
                llm_model_id=_argument(command, "--llm-model-id"),
                model_revision=_argument(command, "--llm-model-revision"),
                tokenizer_revision=_argument(command, "--llm-tokenizer-revision"),
                maximum_new_tokens=int(_argument(command, "--llm-maximum-new-tokens")),
                router_evaluation_evidence_root=Path(
                    _argument(command, "--router-evaluation-evidence-root")
                ),
                control_evidence_root=control_evidence_root,
            )
            _write_json(report_path, control_report)
            _write_json(
                control_evidence_root / "fixture-evidence.json",
                {
                    "identity": {
                        "git_commit": _GIT_COMMIT,
                        "router_evaluation": control_report["router_evaluation_identity"],
                        "active_episode_spec_required": True,
                    },
                    "report": control_report,
                },
            )
            return 0
        raise AssertionError(f"unexpected command: {command}")


def _install_target_boundary_fixtures(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = verify_m5a.build_fixture_controller_registry()
    monkeypatch.setattr(verify_m5a, "_rebuild_controller_registry", lambda _args: registry)
    monkeypatch.setattr(
        verify_m5a,
        "validate_language_corpus_archive",
        lambda *_args, **_kwargs: SimpleNamespace(archive_fingerprint=_digest("corpus-archive")),
    )
    monkeypatch.setattr(
        verify_m5a,
        "validate_completed_text_classifier_artifact",
        lambda *_args, **_kwargs: None,
    )

    def validate_language(root: Path, **_kwargs: object) -> object:
        result = json.loads((root / "fixture-result.json").read_text(encoding="utf-8"))
        return verify_m5a.ValidatedRouterEvaluationEvidence(
            root=root.resolve(),
            evaluation_fingerprint=cast(str, result["evaluation_fingerprint"]),
            artifact_fingerprint=_digest("language-artifact"),
            artifact_count=1,
            owner_fingerprint=_digest("language-owner"),
            result_fingerprint=_digest("language-result"),
            owner={},
            result=result,
        )

    def validate_control(root: Path, **_kwargs: object) -> object:
        fixture = json.loads((root / "fixture-evidence.json").read_text(encoding="utf-8"))
        report = cast(dict[str, object], fixture["report"])
        return verify_m5a.ValidatedControlEvidence(
            root=root.resolve(),
            run_fingerprint=cast(str, report["run_fingerprint"]),
            schedule_fingerprint=cast(str, report["schedule_fingerprint"]),
            corpus_fingerprint=cast(str, report["corpus_fingerprint"]),
            controller_registry_fingerprint=cast(str, report["controller_registry_fingerprint"]),
            record_count=288,
            record_set_fingerprint=cast(str, report["record_set_fingerprint"]),
            completion_fingerprint=cast(str, report["completion_fingerprint"]),
            owner_fingerprint=_digest("control-owner"),
            identity=cast(dict[str, object], fixture["identity"]),
            summaries=cast(dict[str, object], report["summaries"]),
        )

    monkeypatch.setattr(verify_m5a, "validate_router_evaluation_evidence", validate_language)
    monkeypatch.setattr(verify_m5a, "validate_development_control_evidence", validate_control)


def test_target_development_requires_passed_m43_before_subprocess(tmp_path) -> None:
    m43_path = tmp_path / "m43.json"
    _write_json(m43_path, _m43_report(passed=False))
    calls: list[tuple[str, ...]] = []

    def runner(command: object) -> int:
        calls.append(tuple(cast(tuple[str, ...], command)))
        return 0

    assert (
        verify_m5a.main(_target_arguments(tmp_path, m43_path=m43_path), command_runner=runner) == 1
    )
    assert calls == []
    payload = json.loads(
        (tmp_path / "verification" / "verification.json").read_text(encoding="utf-8")
    )
    assert payload["prior_m43b_target_validated"] is False
    assert payload["classifier_training_completed"] is False
    assert payload["passed"] is False


def _stage_flags(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "passed": True,
        "language_final_accessed": False,
        "control_final_accessed": False,
        "test_split_accessed": False,
        "smolvla_go": False,
    }
    payload.update(updates)
    return payload


def test_stage_verifier_separates_pilot_completion_from_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "pilot.json"
    _write_json(
        evidence,
        _stage_flags(classifier_pilot_completed=True, classifier_pilot_promoted=False),
    )
    output = tmp_path / "verification"

    def verify_pilot(*, payload: object, report: verify_m5a.StageVerificationReport) -> None:
        assert isinstance(payload, dict)
        report.classifier_fixture_validated = True
        report.classifier_tiny_overfit_validated = True
        report.classifier_pilot_completed = payload["classifier_pilot_completed"] is True
        report.classifier_pilot_promoted = payload["classifier_pilot_promoted"] is True
        report.artifact_reload_validated = True
        report.cuda_training_validated = True
        report.physical_target_validated = True

    monkeypatch.setattr(verify_m5a, "_verify_classifier_pilot_payload", verify_pilot)

    assert (
        verify_m5a.main(
            [
                "--verify-stage",
                "classifier_pilot",
                "--stage-report",
                str(evidence),
                "--output-root",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads((output / "verification.json").read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["classifier_pilot_completed"] is True
    assert payload["classifier_pilot_promoted"] is False


def test_stage_verifier_rejects_boolean_only_pilot_claim(tmp_path: Path) -> None:
    evidence = tmp_path / "pilot.json"
    _write_json(
        evidence,
        _stage_flags(classifier_pilot_completed=True, classifier_pilot_promoted=True),
    )
    output = tmp_path / "verification"

    assert (
        verify_m5a.main(
            [
                "--verify-stage",
                "classifier_pilot",
                "--stage-report",
                str(evidence),
                "--output-root",
                str(output),
            ]
        )
        == 1
    )
    payload = json.loads((output / "verification.json").read_text(encoding="utf-8"))
    assert payload["classifier_pilot_completed"] is True
    assert payload["classifier_pilot_promoted"] is True
    assert payload["artifact_reload_validated"] is False
    assert payload["passed"] is False


def test_stage_verifier_rejects_boolean_only_recovery_claim(tmp_path: Path) -> None:
    evidence = tmp_path / "recovery.json"
    _write_json(
        evidence,
        _stage_flags(
            classifier_tiny_overfit_validated=True,
            classifier_pilot_completed=True,
            classifier_pilot_promoted=False,
            classifier_recovery_resume_authorized=True,
            classifier_recovery_resume_completed=True,
            classifier_training_completed=True,
            classifier_checkpoint_selected=True,
            cuda_training_validated=True,
            physical_target_validated=True,
        ),
    )
    output = tmp_path / "verification"
    assert (
        verify_m5a.main(
            [
                "--verify-stage",
                "classifier_recovery_training",
                "--stage-report",
                str(evidence),
                "--output-root",
                str(output),
            ]
        )
        == 1
    )
    payload = json.loads((output / "verification.json").read_text(encoding="utf-8"))
    assert payload["classifier_recovery_resume_completed"] is True
    assert payload["implementation_validated"] is False
    assert payload["passed"] is False


def test_stage_verifier_requires_real_cuda_tiny_overfit_evidence(tmp_path: Path) -> None:
    evidence = tmp_path / "tiny.json"
    _write_json(
        evidence,
        _stage_flags(
            classifier_tiny_overfit_validated=True,
            artifact_reload_validated=True,
            cuda_training_validated=False,
        ),
    )
    output = tmp_path / "verification"
    assert (
        verify_m5a.main(
            [
                "--verify-stage",
                "classifier_tiny_overfit",
                "--stage-report",
                str(evidence),
                "--output-root",
                str(output),
            ]
        )
        == 1
    )
    payload = json.loads((output / "verification.json").read_text(encoding="utf-8"))
    assert payload["classifier_tiny_overfit_validated"] is False
    assert payload["passed"] is False


def test_stage_verifier_screen_can_complete_without_selecting_a_router(tmp_path: Path) -> None:
    evidence = tmp_path / "screen.json"
    _write_json(
        evidence,
        _stage_flags(
            stage="three_scene_control_screen",
            physical_execution=True,
            summaries={"oracle": {}, "classifier": {}},
            selected_router=None,
        ),
    )
    output = tmp_path / "verification"
    assert (
        verify_m5a.main(
            [
                "--verify-stage",
                "three_scene_control_screen",
                "--stage-report",
                str(evidence),
                "--output-root",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads((output / "verification.json").read_text(encoding="utf-8"))
    assert payload["three_scene_control_screen_completed"] is True
    assert payload["selected_router_locked"] is False
    assert payload["passed"] is True


def test_stage_verifier_full_quality_is_separate_from_completion(tmp_path: Path) -> None:
    evidence = tmp_path / "full.json"
    _write_json(
        evidence,
        _stage_flags(
            stage="full_control_development",
            physical_execution=True,
            summaries={"oracle": {}, "classifier": {}},
            development_quality_gate_passed=False,
            final_benchmark_authorized=False,
        ),
    )
    output = tmp_path / "verification"
    assert (
        verify_m5a.main(
            [
                "--verify-stage",
                "full_control_development",
                "--stage-report",
                str(evidence),
                "--output-root",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads((output / "verification.json").read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["development_quality_gate_passed"] is False
    assert payload["final_benchmark_authorized"] is False


def test_stage_verifier_rejects_any_final_access(tmp_path: Path) -> None:
    evidence = tmp_path / "language.json"
    _write_json(
        evidence,
        _stage_flags(
            llm_router_loaded=True,
            llm_prompt_locked=True,
            language_development_completed=True,
            language_final_accessed=True,
        ),
    )
    assert (
        verify_m5a.main(
            [
                "--verify-stage",
                "language_development",
                "--stage-report",
                str(evidence),
                "--output-root",
                str(tmp_path / "verification"),
            ]
        )
        == 1
    )


def test_development_gate_quality_is_separate_from_experiment_completion(tmp_path) -> None:
    corpus = verify_m5a.build_language_corpus()
    evidence_root = tmp_path / "language-evidence"
    evidence_root.mkdir()
    language = _language_report(
        corpus_fingerprint=corpus.manifest.corpus_fingerprint,
        artifact_fingerprint=_digest("artifact"),
        language_development={"schedule_fingerprint": _digest("dev")},
        language_final={
            "schedule_id": "m5a_language_final_v0",
            "schedule_fingerprint": _digest("final"),
        },
        evidence_root=evidence_root,
        llm_model_id="fixture/local",
        model_revision="a" * 40,
        tokenizer_revision="b" * 40,
        llm_license="Apache-2.0",
        classifier_accuracy=0.80,
    )
    control = _control_report(
        corpus_fingerprint=corpus.manifest.corpus_fingerprint,
        control_schedule_fingerprint=_digest("control"),
        artifact_fingerprint=_digest("artifact"),
        llm_model_id="fixture/local",
        model_revision="a" * 40,
        tokenizer_revision="b" * 40,
        router_evaluation_evidence_root=evidence_root,
    )

    gate = verify_m5a._build_development_gate(
        router_label="classifier", language_report=language, control_report=control
    )
    assert gate.language_gate_passed is False
    assert gate.development_quality_gate_passed is False

    report = verify_m5a.TargetDevelopmentReport(
        llm_model_id="fixture/local",
        llm_model_revision="a" * 40,
        llm_tokenizer_revision="b" * 40,
        llm_license="Apache-2.0",
        llm_dtype="bfloat16",
        llm_license_reviewed=True,
        llm_model_card_reviewed=True,
        implementation_validated=True,
        prior_m43b_target_validated=True,
        corpus_validated=True,
        split_isolation_validated=True,
        rule_router_validated=True,
        controller_registry_validated=True,
        experiment_manifest_validated=True,
        final_schedules_locked=True,
        classifier_training_completed=True,
        classifier_checkpoint_selected=True,
        classifier_calibration_validated=True,
        llm_router_loaded=True,
        llm_prompt_locked=True,
        language_validation_completed=True,
        language_development_completed=True,
        oracle_control_development_completed=True,
        predicted_control_development_completed=True,
        control_development_completed=True,
        rejection_noop_probe_validated=True,
        failure_attribution_validated=True,
        physical_target_validated=True,
        development_quality_gate_passed=False,
        final_benchmark_authorized=False,
    )
    assert report.passed is True
