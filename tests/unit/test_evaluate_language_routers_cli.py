from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from langmani.language.corpus import build_language_corpus
from langmani.language.router_types import LanguageSplit, RouterStatus
from langmani.language.rule_router import RuleRouterV0
from langmani.policies.act_runtime import GitState
from scripts import build_language_corpus as corpus_cli
from scripts import evaluate_language_routers as cli


def _valid_classifier_evidence() -> tuple[dict[str, object], dict[str, object]]:
    corpus = build_language_corpus()
    revision = "a" * 40
    manifest = {
        "model_id": "distilbert/distilbert-base-uncased",
        "model_revision": revision,
        "tokenizer_revision": revision,
    }
    evidence: dict[str, object] = {
        "schema_version": cli.CLASSIFIER_RUN_EVIDENCE_SCHEMA,
        "mode": "target_training_complete",
        "authoritative_run_fingerprint": f"sha256:{'c' * 64}",
        "classifier_training_seeds": 1,
        "robustness_across_training_seeds_not_evaluated": True,
        "classifier_training_promoted": True,
        "classifier_training_metrics": {
            "full_task_spec_accuracy": 0.95,
            "object_accuracy": 0.97,
            "bin_accuracy": 0.97,
            "false_route_rate": 0.03,
            "ambiguous_rejection_recall": 0.90,
            "unsupported_rejection_recall": 0.95,
            "malformed_rejection_recall": 0.95,
            "schema_valid_rate": 1.0,
        },
        "corpus_manifest": corpus.manifest.to_dict(),
        "split_fingerprints": {
            split.value: corpus.manifest.split_manifests[split].content_fingerprint
            for split in LanguageSplit
        },
        "git_commit": "b" * 40,
        "git_state": {
            "commit": "b" * 40,
            "dirty": False,
            "changed_paths": [],
            "baseline_tracked": True,
        },
        "dependencies": {
            "torch": "fixture",
            "transformers": "5.4.0",
            "tokenizers": "0.22.2",
        },
        "model_id": manifest["model_id"],
        "model_revision": revision,
        "tokenizer_revision": revision,
        "training_config": {"maximum_steps": 2},
        "training_result": {
            "selected_checkpoint_id": "step_00000002",
            "selected_step": 2,
            "checkpoint_validations": [
                {
                    "checkpoint_id": "step_00000002",
                    "step": 2,
                    "validation_examples": 300,
                    "evidence_split": "validation",
                }
            ],
        },
        "calibration_config": {
            "evidence_split": "validation",
            "maximum_false_route_rate": 0.03,
        },
        "calibration_selection": {
            "temperature": {
                "temperature": 1.25,
                "validation_examples": 300,
                "nll_before": 0.5,
                "nll_after": 0.4,
                "bounded_iterations": 64,
            },
            "threshold": {
                "threshold": 0.75,
                "validation_routeable_examples": 180,
                "validation_rejected_examples": 120,
                "valid_full_task_accuracy": 0.95,
                "false_route_rate": 0.02,
                "rejection_recall": 0.95,
            },
        },
        "router_config": {
            "maximum_sequence_length": 64,
            "routing_threshold": 0.75,
            "device_independent": True,
        },
        "data_usage": {
            "gradient_split": "train",
            "checkpoint_selection_split": "validation",
            "temperature_calibration_split": "validation",
            "threshold_selection_split": "validation",
            "train_examples": 900,
            "validation_examples": 300,
            "development_examples_materialized": False,
            "development_used_for_training_or_selection": False,
            "final_examples_materialized": False,
            "final_used_for_training_or_selection": False,
        },
        "training_runtime": {
            "device": "cuda",
            "dtype": "float32",
            "cuda_available": True,
        },
        "random_seeds": {"torch": 0, "cuda": 0, "data_order": 0},
    }
    return evidence, manifest


def test_classifier_evidence_requires_exact_validation_only_contract() -> None:
    corpus = build_language_corpus()
    evidence, manifest = _valid_classifier_evidence()

    calibration, config = cli._validate_classifier_evidence(
        evidence,
        corpus=corpus,
        classifier_manifest=manifest,
        target_development=True,
    )

    assert calibration.temperature == 1.25
    assert config.maximum_sequence_length == 64
    assert config.routing_threshold == 0.75
    invalid = json.loads(json.dumps(evidence))
    cast(dict[str, object], invalid["data_usage"])["development_used_for_training_or_selection"] = (
        True
    )
    with pytest.raises(cli.LanguageRouterEvaluationCommandError, match="train-validation"):
        cli._validate_classifier_evidence(
            invalid,
            corpus=corpus,
            classifier_manifest=manifest,
            target_development=True,
        )
    missing_seed_identity = dict(evidence)
    missing_seed_identity.pop("random_seeds")
    with pytest.raises(cli.LanguageRouterEvaluationCommandError, match="fields differ"):
        cli._validate_classifier_evidence(
            missing_seed_identity,
            corpus=corpus,
            classifier_manifest=manifest,
            target_development=True,
        )


def test_prompt_selection_is_stable_balanced_and_train_only() -> None:
    corpus = build_language_corpus()

    first = cli._select_prompt_examples(corpus)
    second = cli._select_prompt_examples(corpus)

    assert [example.example_id for example in first] == [example.example_id for example in second]
    assert all(example.split is LanguageSplit.TRAIN for example in first)
    assert len({example.task_id for example in first if example.task_id is not None}) == 6
    expected_reasons = {
        example.expected_rejection_reason
        for example in corpus.examples_for_split(LanguageSplit.TRAIN)
        if example.expected_rejection_reason is not None
    }
    selected_reasons = {
        example.expected_rejection_reason
        for example in first
        if example.expected_rejection_reason is not None
    }
    assert selected_reasons == expected_reasons - {
        cli.RouterRejectionReason.EMPTY_TEXT,
        cli.RouterRejectionReason.MALFORMED_CONTROL_CHARACTERS,
    }


def _args(tmp_path: Path, *, target: bool = False):
    mode = "--target-development" if target else "--fixture"
    return cli.parse_args(
        [
            mode,
            "--corpus-root",
            str(tmp_path / "corpus"),
            "--classifier-checkpoint",
            str(tmp_path / "classifier.pt"),
            "--classifier-rejection-evidence",
            str(tmp_path / "classifier-rejection"),
            "--device",
            "cpu",
            "--output-root",
            str(tmp_path / "evidence"),
            "--report",
            str(tmp_path / "report.json"),
        ]
    )


def test_injected_generator_cannot_masquerade_as_target(tmp_path: Path) -> None:
    args = _args(tmp_path, target=True)
    args.corpus_root.mkdir()
    args.classifier_checkpoint.write_bytes(b"fixture")
    args.classifier_rejection_evidence.mkdir()

    with pytest.raises(cli.LanguageRouterEvaluationCommandError, match="fixture-only"):
        cli.execute(args, generator_factory=lambda *_args: object())


class _RuleBackedGenerator:
    def __init__(self) -> None:
        self.router = RuleRouterV0()

    def generate(self, prompt: str, *, max_new_tokens: int) -> str:
        assert max_new_tokens == 128
        command_line = prompt.rsplit("Command: ", maxsplit=1)[1].splitlines()[0]
        command = json.loads(command_line)
        decision = self.router.route(command)
        if decision.status is RouterStatus.ROUTE:
            payload = {
                "status": "route",
                "target_object_id": decision.target_object_id,
                "target_bin_id": decision.target_bin_id,
                "reason": "explicit_object_and_destination",
            }
        else:
            payload = {
                "status": decision.status.value,
                "target_object_id": None,
                "target_bin_id": None,
                "reason": {
                    "conflicting_bins": "conflicting_destinations",
                    "meaningless_or_noise_text": "meaningless_or_noise",
                    "empty_text": "malformed_input",
                    "malformed_control_characters": "malformed_input",
                }.get(
                    cast(object, decision.rejection_reason).value,
                    cast(object, decision.rejection_reason).value,
                ),
            }
        return json.dumps(payload, sort_keys=True)


def test_fixture_evaluation_is_immutable_and_never_claims_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    corpus_cli.build_archive(args.corpus_root)
    args.classifier_checkpoint.write_bytes(b"fixture")
    args.classifier_rejection_evidence.mkdir()
    monkeypatch.setattr(
        cli,
        "validate_rejection_analysis_artifact",
        lambda *_args, **_kwargs: {"analysis_fingerprint": "sha256:" + "1" * 64},
    )
    monkeypatch.setattr(
        cli,
        "inspect_git_state",
        lambda _root: GitState(
            commit="c" * 40,
            dirty=True,
            changed_paths=("?? fixture",),
            baseline_tracked=True,
        ),
    )

    result = cli.execute(
        args,
        generator_factory=lambda *_args: _RuleBackedGenerator(),
    )

    assert result["passed"] is True
    assert result["mode"] == "fixture"
    assert result["real_gpu_inference_validated"] is False
    assert result["llm_validation_completed"] is False
    assert result["language_development_completed"] is False
    assert result["one_scene_control_smoke_authorized"] is False
    assert result["physical_target_validated"] is False
    assert result["language_final_accessed"] is False
    assert not (args.corpus_root / "final.jsonl").exists()
    evidence_root = Path(cast(str, result["evidence_root"]))
    first_result = (evidence_root / "result.json").read_bytes()
    reused = cli.execute(
        args,
        generator_factory=lambda *_args: _RuleBackedGenerator(),
    )
    assert reused["evidence_reused"] is True
    assert (evidence_root / "result.json").read_bytes() == first_result

    (evidence_root / "result.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksums differ"):
        cli.execute(
            args,
            generator_factory=lambda *_args: _RuleBackedGenerator(),
        )
