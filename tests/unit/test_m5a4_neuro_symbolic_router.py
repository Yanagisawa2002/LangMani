from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path

import pytest
from pydantic import ValidationError

from langmani.language import neuro_symbolic_verifier
from langmani.language.corpus import build_language_corpus
from langmani.language.llm_router import StructuredLLMRouterError
from langmani.language.neuro_symbolic_evidence import (
    NeuroSymbolicEvidenceError,
    validate_neuro_symbolic_evidence,
    write_development_access_lock,
    write_neuro_symbolic_evidence,
)
from langmani.language.neuro_symbolic_experiment import (
    evaluate_neuro_symbolic_development_gate,
    initial_m5a4_flags,
)
from langmani.language.neuro_symbolic_router import (
    ARBITER_VERSION,
    OUTLINES_VERSION,
    DeterministicSafetyArbiterV0,
    NeuroSymbolicRouterV0,
    QwenSemanticFrameV0,
    SemanticRequestedAction,
    SymbolicLexicalParserV0,
    build_semantic_frame_prompt,
    expected_semantic_frame_payload,
    select_semantic_prompt_examples,
    semantic_frame_schema,
    semantic_schema_fingerprint,
)
from langmani.language.router_types import LanguageSplit, RouterRejectionReason, RouterStatus
from scripts.evaluate_neuro_symbolic_router import _run_train_smoke


def semantic(
    *,
    objects: tuple[str, ...] = ("red_cube",),
    bins: tuple[str, ...] = ("left_bin",),
    action: SemanticRequestedAction = SemanticRequestedAction.PICK_AND_PLACE,
    **overrides: bool,
) -> QwenSemanticFrameV0:
    payload: dict[str, object] = {
        "mentioned_supported_objects": objects,
        "mentions_unsupported_object": False,
        "mentioned_supported_bins": bins,
        "mentions_unsupported_destination": False,
        "requested_action": action,
        "has_multiple_tasks": False,
        "has_conflicting_objects": False,
        "has_conflicting_destinations": False,
        "has_unresolved_correction": False,
        "has_contradictory_negation": False,
        "is_meaningless_or_noise": False,
        "is_malformed_input": False,
    }
    payload.update(overrides)
    return QwenSemanticFrameV0.model_validate(payload)


@dataclass
class FakeExtractor:
    frame: QwenSemanticFrameV0
    compilation_seconds: float = 0.1
    generation_metadata: list[dict[str, object]] = field(default_factory=list)

    def extract(self, command: str) -> QwenSemanticFrameV0:
        self.generation_metadata.append({"command_length": len(command)})
        return self.frame


@dataclass
class InvalidFrameExtractor:
    compilation_seconds: float = 0.1
    generation_metadata: list[dict[str, object]] = field(default_factory=list)

    def extract(self, command: str) -> QwenSemanticFrameV0:
        raise StructuredLLMRouterError("mandatory semantic validation failed")


@dataclass
class MappingExtractor:
    frames: dict[str, QwenSemanticFrameV0]
    compilation_seconds: float = 0.1
    generation_metadata: list[dict[str, object]] = field(default_factory=list)

    def extract(self, command: str) -> QwenSemanticFrameV0:
        self.generation_metadata.append({"command_length": len(command)})
        return self.frames[command]


def decision(command: str, frame: QwenSemanticFrameV0):
    return DeterministicSafetyArbiterV0().arbitrate(SymbolicLexicalParserV0().parse(command), frame)


def test_symbolic_frame_direct_routeable_command() -> None:
    frame = SymbolicLexicalParserV0().parse("Move the red cube into the left bin.")
    assert frame.supported_objects == ("red_cube",)
    assert frame.supported_bins == ("left_bin",)
    assert frame.supported_action_mentions == ("move",)
    assert frame.matched_spans


def test_symbolic_missing_object() -> None:
    frame = SymbolicLexicalParserV0().parse("Move it into the left bin.")
    assert frame.supported_objects == () and frame.supported_bins == ("left_bin",)


def test_symbolic_missing_destination() -> None:
    frame = SymbolicLexicalParserV0().parse("Move the blue cube.")
    assert frame.supported_objects == ("blue_cube",) and frame.supported_bins == ()


def test_symbolic_conflicting_objects() -> None:
    frame = SymbolicLexicalParserV0().parse("Move the red cube or blue cube left.")
    assert frame.supported_objects == ("red_cube", "blue_cube")


def test_symbolic_conflicting_bins() -> None:
    frame = SymbolicLexicalParserV0().parse("Move the red cube to the left bin or right bin.")
    assert frame.supported_bins == ("left_bin", "right_bin")


def test_symbolic_unsupported_object() -> None:
    assert SymbolicLexicalParserV0().parse(
        "Move the purple cube left."
    ).unsupported_object_mentions == ("purple cube",)


def test_symbolic_unsupported_destination() -> None:
    assert SymbolicLexicalParserV0().parse(
        "Move the red cube to the shelf."
    ).unsupported_destination_mentions == ("shelf",)


def test_symbolic_unsupported_action() -> None:
    assert SymbolicLexicalParserV0().parse(
        "Push the red cube toward the left bin."
    ).unsupported_action_mentions == ("push",)


def test_symbolic_unsupported_spatial_reference() -> None:
    assert SymbolicLexicalParserV0().parse(
        "Move the red cube over there."
    ).spatial_reference_mentions == ("over there",)


def test_symbolic_unresolved_correction() -> None:
    frame = SymbolicLexicalParserV0().parse("Actually never mind, move the red cube left.")
    assert frame.correction_present and not frame.correction_resolved


def test_symbolic_resolved_correction() -> None:
    frame = SymbolicLexicalParserV0().parse(
        "Correction to the wording: move only the red cube into only the left bin."
    )
    assert frame.correction_present and frame.correction_resolved


def test_symbolic_contradictory_negation() -> None:
    assert (
        SymbolicLexicalParserV0()
        .parse("Do not move the red cube to the left bin.")
        .contradictory_negation
    )


def test_symbolic_empty_input() -> None:
    assert SymbolicLexicalParserV0().parse("  ").empty_input


def test_symbolic_malformed_control_input() -> None:
    assert SymbolicLexicalParserV0().parse("Move\x02 red cube left.").malformed_control_characters


def test_symbolic_noise_input() -> None:
    assert SymbolicLexicalParserV0().parse("trazzle fim wob").meaningless_or_noise


def test_semantic_frame_schema_is_strict() -> None:
    schema = semantic_frame_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(QwenSemanticFrameV0.model_fields)


def test_semantic_frame_rejects_unknown_enum() -> None:
    with pytest.raises(ValidationError):
        semantic(objects=("yellow_cube",))


def test_semantic_frame_has_no_final_router_status() -> None:
    assert "status" not in QwenSemanticFrameV0.model_fields


def test_semantic_frame_has_no_final_task_spec() -> None:
    assert "task_spec" not in QwenSemanticFrameV0.model_fields


def test_outlines_json_schema_compiles() -> None:
    from outlines.types import JsonSchema

    assert version("outlines") == OUTLINES_VERSION
    assert JsonSchema(json.dumps(semantic_frame_schema())).schema


def test_arbiter_precedence_malformed_before_unsupported() -> None:
    result = decision(
        "\x02 push the purple cube to the shelf",
        semantic(action=SemanticRequestedAction.UNSUPPORTED_ACTION),
    )
    assert result.status is RouterStatus.REJECT_MALFORMED
    assert result.rejection_reason is RouterRejectionReason.MALFORMED_CONTROL_CHARACTERS


def test_arbiter_exact_reason_mapping() -> None:
    result = decision(
        "Push the red cube into the left bin.",
        semantic(action=SemanticRequestedAction.UNSUPPORTED_ACTION),
    )
    assert result.rejection_reason is RouterRejectionReason.UNSUPPORTED_ACTION
    assert result.evidence["arbiter_rule"] == "u01"


def test_arbiter_agreement_routes() -> None:
    result = decision("Move the red cube into the left bin.", semantic())
    assert result.status is RouterStatus.ROUTE and result.task_spec is not None


def test_arbiter_object_disagreement_rejects() -> None:
    result = decision("Move the red cube into the left bin.", semantic(objects=("blue_cube",)))
    assert result.rejection_reason is RouterRejectionReason.CONFLICTING_OBJECTS


def test_arbiter_bin_disagreement_rejects() -> None:
    result = decision("Move the red cube into the left bin.", semantic(bins=("right_bin",)))
    assert result.rejection_reason is RouterRejectionReason.CONFLICTING_BINS


def test_arbiter_action_disagreement_rejects() -> None:
    result = decision(
        "Move the red cube into the left bin.",
        semantic(action=SemanticRequestedAction.UNCLEAR),
    )
    assert result.status is RouterStatus.REJECT_AMBIGUOUS


def test_rejected_decision_has_no_task_spec() -> None:
    result = decision("Move the red cube.", semantic(bins=()))
    assert result.task_spec is None and result.task_id is None


def test_invalid_constrained_frame_is_malformed_not_executable() -> None:
    router = NeuroSymbolicRouterV0(
        parser=SymbolicLexicalParserV0(),
        semantic_extractor=InvalidFrameExtractor(),
        arbiter=DeterministicSafetyArbiterV0(),
    )
    result = router.route("Move the red cube into the left bin.")
    assert result.status is RouterStatus.REJECT_MALFORMED
    assert result.rejection_reason is RouterRejectionReason.STRUCTURED_OUTPUT_INVALID
    assert result.task_spec is None


def test_router_output_is_deterministic() -> None:
    router = NeuroSymbolicRouterV0(
        parser=SymbolicLexicalParserV0(),
        semantic_extractor=FakeExtractor(semantic()),
        arbiter=DeterministicSafetyArbiterV0(),
    )
    first = router.route("Move the red cube into the left bin.")
    second = router.route("Move the red cube into the left bin.")
    assert first.decision_fingerprint == second.decision_fingerprint


def test_train_only_prompt_and_stable_ids() -> None:
    corpus = build_language_corpus()
    selected = select_semantic_prompt_examples(corpus.examples_for_split(LanguageSplit.TRAIN))
    prompt, fingerprint, ids = build_semantic_frame_prompt(
        examples=selected, parser=SymbolicLexicalParserV0()
    )
    assert len(ids) == 20 and all(value.example_id in prompt for value in selected)
    assert fingerprint.startswith("sha256:")


def test_train_smoke_compares_json_normalized_semantic_frames() -> None:
    corpus = build_language_corpus()
    parser = SymbolicLexicalParserV0()
    selected = select_semantic_prompt_examples(corpus.examples_for_split(LanguageSplit.TRAIN))
    extractor = MappingExtractor(
        {
            example.raw_text: QwenSemanticFrameV0.model_validate(
                expected_semantic_frame_payload(example, parser.parse(example.raw_text))
            )
            for example in selected
        }
    )
    router = NeuroSymbolicRouterV0(
        parser=parser,
        semantic_extractor=extractor,
        arbiter=DeterministicSafetyArbiterV0(),
    )
    result = _run_train_smoke(router, parser, selected)
    checks = result["checks"]
    assert isinstance(checks, dict)
    assert checks["field_semantics_match_train_labels"] is True


def test_historical_diagnostic_cannot_change_configuration() -> None:
    parser = SymbolicLexicalParserV0()
    arbiter = DeterministicSafetyArbiterV0()
    before = (parser.contract_fingerprint, arbiter.contract_fingerprint)
    _ = decision("Move the red cube into the left bin.", semantic())
    assert before == (parser.contract_fingerprint, arbiter.contract_fingerprint)


def test_router_lock_is_immutable(tmp_path: Path) -> None:
    fingerprint = "sha256:" + "1" * 64
    payload = {"prompt_fingerprint": "sha256:" + "2" * 64}
    first = write_development_access_lock(
        tmp_path, runtime_fingerprint=fingerprint, prompt_lock=payload
    )
    second = write_development_access_lock(
        tmp_path, runtime_fingerprint=fingerprint, prompt_lock=payload
    )
    assert first == second and first["neuro_symbolic_router_locked"] is True
    with pytest.raises(NeuroSymbolicEvidenceError):
        write_development_access_lock(
            tmp_path,
            runtime_fingerprint=fingerprint,
            prompt_lock={"prompt_fingerprint": "sha256:" + "3" * 64},
        )


def _passing_metrics() -> dict[str, object]:
    return {
        "valid_full_task_accuracy": 0.95,
        "object_accuracy": 0.97,
        "bin_accuracy": 0.97,
        "false_route_rate": 0.03,
        "ambiguous_rejection_recall": 0.90,
        "unsupported_rejection_recall": 0.95,
        "malformed_rejection_recall": 0.95,
        "deterministic_repeatability": 1.0,
        "per_task_accuracy": {"task": 0.90},
        "rejection_family_false_route_rates": {"family": 0.10},
    }


def _passing_safety() -> dict[str, object]:
    return {
        "grammar_schema_valid_rate": 1.0,
        "route_on_unsupported_action": 0,
        "route_on_conflicting_objects": 0,
        "route_on_conflicting_bins": 0,
        "route_on_empty_or_noise_input": 0,
        "route_on_unresolved_correction": 0,
    }


def test_development_gate_is_conjunctive() -> None:
    passed = evaluate_neuro_symbolic_development_gate(
        metrics=_passing_metrics(),
        safety_metrics=_passing_safety(),
        no_prohibited_source_access=True,
    )
    assert passed.passed
    failed_metrics = _passing_metrics()
    failed_metrics["object_accuracy"] = 0.969999
    assert not evaluate_neuro_symbolic_development_gate(
        metrics=failed_metrics,
        safety_metrics=_passing_safety(),
        no_prohibited_source_access=True,
    ).passed


@pytest.mark.parametrize(
    "flag",
    [
        "language_final_accessed",
        "control_final_accessed",
        "test_split_accessed",
        "m42_final_accessed",
        "smolvla_go",
    ],
)
def test_forbidden_access_flags_default_false(flag: str) -> None:
    assert initial_m5a4_flags()[flag] is False


def test_no_optimizer_or_training_contract() -> None:
    contract = DeterministicSafetyArbiterV0().contract_dict()
    assert contract["arbiter_version"] == ARBITER_VERSION
    assert "optimizer" not in json.dumps(contract).lower()


def test_evidence_atomicity_and_completion_last(tmp_path: Path) -> None:
    owner = {"runtime_fingerprint": "sha256:" + "a" * 64, "mode": "fixture"}
    result = write_neuro_symbolic_evidence(
        tmp_path,
        owner=owner,
        artifacts={"prompt.json": {"train_only": True}, "summary.md": "# Fixture\n"},
        flags=initial_m5a4_flags(),
    )
    root = Path(str(result["root"]))
    validated = validate_neuro_symbolic_evidence(root, expected_owner=owner)
    assert validated["passed"] is True
    assert (root / "complete.json").stat().st_mtime_ns >= (
        root / "manifest.json"
    ).stat().st_mtime_ns


def test_evidence_output_path_safety(tmp_path: Path) -> None:
    owner = {"runtime_fingerprint": "sha256:" + "b" * 64}
    with pytest.raises(NeuroSymbolicEvidenceError):
        write_neuro_symbolic_evidence(
            tmp_path,
            owner=owner,
            artifacts={"../escape.json": {}},
            flags={},
        )


def test_failed_train_smoke_is_a_verified_terminal_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    model_file = snapshot / "tiny.bin"
    model_file.write_bytes(b"tiny")
    digest = hashlib.sha256(b"tiny").hexdigest()
    monkeypatch.setattr(
        neuro_symbolic_verifier,
        "QWEN3_4B_INSTRUCT_FILE_IDENTITIES",
        {"tiny.bin": (4, digest)},
    )
    corpus = build_language_corpus()
    parser = SymbolicLexicalParserV0()
    arbiter = DeterministicSafetyArbiterV0()
    prompt_examples = select_semantic_prompt_examples(
        corpus.examples_for_split(LanguageSplit.TRAIN)
    )
    prompt, prompt_fingerprint, prompt_ids = build_semantic_frame_prompt(
        examples=prompt_examples, parser=parser
    )
    smoke_records: list[dict[str, object]] = []
    for index, example in enumerate(prompt_examples):
        expected = expected_semantic_frame_payload(example, parser.parse(example.raw_text))
        observed = dict(expected)
        if index == 0:
            observed["is_meaningless_or_noise"] = not bool(observed["is_meaningless_or_noise"])
        smoke_records.append(
            {
                "example_id": example.example_id,
                "semantic_frame": observed,
                "expected_semantic_frame": expected,
                "semantic_fields_exact": observed == expected,
                "deterministic": True,
            }
        )
    flags = initial_m5a4_flags()
    flags.update(
        {
            "neuro_symbolic_implementation_validated": True,
            "symbolic_frame_validated": True,
            "semantic_frame_schema_validated": True,
            "constrained_decoding_validated": True,
            "arbiter_validated": True,
            "train_smoke_completed": True,
            "real_gpu_inference_validated": True,
        }
    )
    runtime_fingerprint = "sha256:" + "c" * 64
    owner = {"runtime_fingerprint": runtime_fingerprint, "mode": "target_development"}
    artifacts = {
        "symbolic_contract.json": parser.contract_dict(),
        "semantic_schema.json": {
            "schema": semantic_frame_schema(),
            "schema_fingerprint": semantic_schema_fingerprint(),
        },
        "constrained_decoder_identity.json": {
            "distribution": "outlines",
            "version": OUTLINES_VERSION,
            "license": "Apache-2.0",
            "free_form_fallback": False,
            "grammar_cached_for_session": True,
        },
        "model_identity.json": {
            "model_id": neuro_symbolic_verifier.QWEN3_4B_INSTRUCT_MODEL_ID,
            "model_revision": neuro_symbolic_verifier.QWEN3_4B_INSTRUCT_REVISION,
            "tokenizer_revision": neuro_symbolic_verifier.QWEN3_4B_INSTRUCT_REVISION,
            "dtype": "bfloat16",
            "quantization": "none",
            "weights_unchanged": True,
            "snapshot_path": str(snapshot),
            "file_identities": {"tiny.bin": {"size_bytes": 4, "sha256": f"sha256:{digest}"}},
        },
        "prompt.json": {
            "train_only": True,
            "prompt_sweep": False,
            "prompt_content": prompt,
            "prompt_fingerprint": prompt_fingerprint,
            "few_shot_example_ids": list(prompt_ids),
        },
        "arbiter_contract.json": arbiter.contract_dict(),
        "runtime_identity.json": {
            "optimizer_constructed": False,
            "training_performed": False,
            "robot_environment_created": False,
            "controller_loaded": False,
            "visible_gpu_count": 1,
        },
        "train_smoke.json": {
            "example_count": 20,
            "example_ids": list(prompt_ids),
            "checks": {"field_semantics_match_train_labels": False},
            "gate_passed": False,
            "records": smoke_records,
        },
        "candidate_selection.json": {
            "stopped_at_train_smoke": True,
            "rejection_classification": "train_only_semantic_smoke_failure",
        },
    }
    written = write_neuro_symbolic_evidence(
        tmp_path / "evidence", owner=owner, artifacts=artifacts, flags=flags
    )
    verified = neuro_symbolic_verifier.verify_neuro_symbolic_evidence(written["root"])
    assert verified["passed"] is True
    assert verified["stopped_at_train_smoke"] is True
    assert verified["prompt_lock_validated"] is False
