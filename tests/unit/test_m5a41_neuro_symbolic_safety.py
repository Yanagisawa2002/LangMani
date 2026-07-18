from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from langmani.environments.specs import TaskSpec
from langmani.language.corpus import build_language_corpus
from langmani.language.neuro_symbolic_router import (
    DeterministicSafetyArbiterV0,
    QwenSemanticFrameV0,
    SymbolicLexicalParserV0,
    expected_semantic_frame_payload,
    select_semantic_prompt_examples,
)
from langmani.language.neuro_symbolic_safety import (
    M5A41ContractAmendment,
    RoutingSafetyClass,
    evaluate_development_safety_gate,
    evaluate_train_smoke_contracts,
    evaluate_train_smoke_gate,
    initial_m5a41_flags,
    rejection_taxonomy_contract,
    safety_contract_fingerprint,
    taxonomy_audit,
    taxonomy_contract_fingerprint,
)
from langmani.language.neuro_symbolic_safety_evidence import (
    M5A41EvidenceError,
    validate_m5a41_evidence,
    write_m5a41_evidence,
    write_router_lock,
)
from langmani.language.router_types import (
    LanguageExample,
    LanguageSplit,
    RouterConfidence,
    RouterDecision,
    RouterRejectionReason,
    RouterStatus,
)


def _smoke_records(*, reproduce_historical_errors: bool) -> tuple[dict[str, object], ...]:
    corpus = build_language_corpus()
    parser = SymbolicLexicalParserV0()
    arbiter = DeterministicSafetyArbiterV0()
    examples = select_semantic_prompt_examples(corpus.examples_for_split(LanguageSplit.TRAIN))
    historical = {
        "langmani-m5a-language-example-00e9d09434060c3c4a9a6c2098ea60c6d03c78b6d598e828807149b0d83ec7e1": (
            "mentions_unsupported_destination"
        ),
        "langmani-m5a-language-example-0a173aa2dcc34aad8d604950f45c51acda63350a6bceabd29ba08acc067bb4ba": (
            "mentions_unsupported_object"
        ),
        "langmani-m5a-language-example-070f3b207453a205034f41321e8c6ee44fb849079e057dac2fa3c876a5b13aaf": (
            "mentions_unsupported_destination"
        ),
    }
    records: list[dict[str, object]] = []
    for example in examples:
        symbolic = parser.parse(example.raw_text)
        expected = expected_semantic_frame_payload(example, symbolic)
        semantic = dict(expected)
        if reproduce_historical_errors and example.example_id in historical:
            semantic[historical[example.example_id]] = True
        frame = QwenSemanticFrameV0.model_validate(semantic)
        decision = arbiter.arbitrate(symbolic, frame)
        records.append(
            {
                "example_id": example.example_id,
                "expected_status": example.expected_status.value,
                "expected_task_id": example.task_id,
                "decision": decision.to_dict(),
                "semantic_frame": frame.to_dict(),
                "expected_semantic_frame": expected,
                "semantic_fields_exact": semantic == expected,
                "deterministic": True,
            }
        )
    return tuple(records)


def _smoke_examples() -> tuple[LanguageExample, ...]:
    corpus = build_language_corpus()
    return select_semantic_prompt_examples(corpus.examples_for_split(LanguageSplit.TRAIN))


def test_exact_safety_class_contract() -> None:
    assert tuple(RoutingSafetyClass) == (
        RoutingSafetyClass.EXECUTABLE_ROUTE,
        RoutingSafetyClass.SAFE_REJECTION,
        RoutingSafetyClass.UNSAFE_FALSE_ROUTE,
        RoutingSafetyClass.FALSE_REJECTION,
        RoutingSafetyClass.MALFORMED_DECISION,
    )
    assert safety_contract_fingerprint().startswith("sha256:")


def test_historical_safe_rejections_pass_safety_but_fail_taxonomy() -> None:
    safety, taxonomy = evaluate_train_smoke_contracts(
        records=_smoke_records(reproduce_historical_errors=True),
        examples=_smoke_examples(),
    )
    gate = evaluate_train_smoke_gate(safety=safety, no_prohibited_source_access=True)
    assert gate.passed
    assert safety.routeable_full_task_accuracy == 1.0
    assert safety.safe_rejection_recall == 1.0
    assert safety.unsafe_false_route_count == 0
    assert safety.false_rejection_count == 0
    assert safety.malformed_decision_count == 0
    assert taxonomy.exact_four_way_status_accuracy == 17 / 20
    assert (
        sum(value.expected_status is not value.actual_status for value in taxonomy.mismatches) == 3
    )
    assert len(taxonomy.mismatches) >= 3
    assert not taxonomy.exact_quality_passed


def test_unsafe_false_route_remains_a_hard_smoke_failure() -> None:
    records = list(_smoke_records(reproduce_historical_errors=False))
    examples = _smoke_examples()
    index = next(
        index
        for index, example in enumerate(examples)
        if example.expected_status is not RouterStatus.ROUTE
    )
    records[index]["decision"] = RouterDecision.route(
        task_spec=TaskSpec.from_mapping(
            {
                "target_object_id": "red_cube",
                "target_bin_id": "left_bin",
                "instruction_template_id": "canonical_v0",
            }
        ),
        confidence=RouterConfidence.unavailable(),
        router_name="fixture",
        router_version="fixture-v0",
    ).to_dict()
    safety, _ = evaluate_train_smoke_contracts(records=records, examples=examples)
    assert safety.unsafe_false_route_count == 1
    assert not evaluate_train_smoke_gate(safety=safety, no_prohibited_source_access=True).passed


def test_false_rejection_remains_a_hard_smoke_failure() -> None:
    records = list(_smoke_records(reproduce_historical_errors=False))
    examples = _smoke_examples()
    index = next(
        index
        for index, example in enumerate(examples)
        if example.expected_status is RouterStatus.ROUTE
    )
    records[index]["decision"] = RouterDecision.reject(
        status=RouterStatus.REJECT_AMBIGUOUS,
        rejection_reason=RouterRejectionReason.LOW_CONFIDENCE,
        confidence=RouterConfidence.unavailable(),
        router_name="fixture",
        router_version="fixture-v0",
    ).to_dict()
    safety, _ = evaluate_train_smoke_contracts(records=records, examples=examples)
    assert safety.false_rejection_count == 1
    assert not evaluate_train_smoke_gate(safety=safety, no_prohibited_source_access=True).passed


def test_taxonomy_audit_classifies_exact_three_model_semantic_errors() -> None:
    corpus = build_language_corpus()
    audit = taxonomy_audit(
        corpus=corpus,
        source_smoke={"records": list(_smoke_records(reproduce_historical_errors=True))},
    )
    assert audit["taxonomy_contract_consistent"] is True
    assert audit["source_smoke_outputs_unchanged"] is True
    assert audit["smoke_mismatch_count"] == 3
    mismatches = cast(list[Mapping[str, object]], audit["smoke_mismatch_audits"])
    assert all(value["classification"] == "model_semantic_error" for value in mismatches)
    assert all(value["final_decision_executable"] is False for value in mismatches)
    assert all(value["controller_dispatch_possible"] is False for value in mismatches)
    assert sum(bool(value["overlapping_taxonomy"]) for value in mismatches) == 2
    assert audit["development_command_text_accessed_before_lock"] is False


def test_taxonomy_contract_covers_fourteen_corpus_reasons() -> None:
    contract = rejection_taxonomy_contract()
    assert contract["diagnostic_only"] is True
    assert contract["controls_physical_behavior"] is False
    rules = cast(Mapping[str, object], contract["rules"])
    assert len(rules) == 14
    assert taxonomy_contract_fingerprint().startswith("sha256:")


def _passing_development_metrics() -> dict[str, object]:
    return {
        "valid_full_task_accuracy": 0.95,
        "object_accuracy": 0.97,
        "bin_accuracy": 0.97,
        "per_task_accuracy": {"task": 0.90},
    }


def _passing_development_safety():
    safety, _ = evaluate_train_smoke_contracts(
        records=_smoke_records(reproduce_historical_errors=True),
        examples=_smoke_examples(),
    )
    return safety


def _zero_special_counts() -> dict[str, int]:
    return {
        "route_on_unsupported_action": 0,
        "route_on_conflicting_objects": 0,
        "route_on_conflicting_bins": 0,
        "route_on_empty_or_noise_input": 0,
        "route_on_unresolved_correction": 0,
        "route_on_unsupported_spatial_reference": 0,
        "route_on_contradictory_negation": 0,
    }


def test_development_gate_uses_safety_not_exact_taxonomy() -> None:
    gate = evaluate_development_safety_gate(
        metrics=_passing_development_metrics(),
        safety=_passing_development_safety(),
        special_route_counts=_zero_special_counts(),
        rejection_family_unsafe_false_route_rates={"family": 0.0},
        no_prohibited_source_access=True,
    )
    assert gate.passed
    assert not any("taxonomy" in key for key in gate.checks)


@pytest.mark.parametrize(
    "name",
    [
        "route_on_unsupported_action",
        "route_on_conflicting_objects",
        "route_on_conflicting_bins",
        "route_on_empty_or_noise_input",
        "route_on_unresolved_correction",
        "route_on_unsupported_spatial_reference",
        "route_on_contradictory_negation",
    ],
)
def test_each_special_safety_family_is_a_hard_gate(name: str) -> None:
    counts = _zero_special_counts()
    counts[name] = 1
    gate = evaluate_development_safety_gate(
        metrics=_passing_development_metrics(),
        safety=_passing_development_safety(),
        special_route_counts=counts,
        rejection_family_unsafe_false_route_rates={"family": 0.0},
        no_prohibited_source_access=True,
    )
    assert not gate.passed


def test_amendment_is_content_fingerprinted() -> None:
    fingerprint = "sha256:" + "a" * 64
    amendment = M5A41ContractAmendment(
        implementation_git="b" * 40,
        source_router_runtime_fingerprint=fingerprint,
        source_train_smoke_artifact_fingerprint=fingerprint,
        corpus_fingerprint=fingerprint,
        split_fingerprints={"train": fingerprint},
        taxonomy_contract_fingerprint=fingerprint,
        safety_contract_fingerprint=fingerprint,
        model_fingerprint=fingerprint,
        prompt_fingerprint=fingerprint,
        semantic_schema_fingerprint=fingerprint,
        symbolic_contract_fingerprint=fingerprint,
        arbiter_fingerprint=fingerprint,
    )
    assert amendment.to_dict()["amendment_fingerprint"] == amendment.fingerprint


def test_router_lock_is_immutable_and_declares_taxonomy_limit(tmp_path: Path) -> None:
    runtime = "sha256:" + "c" * 64
    first = write_router_lock(
        tmp_path,
        runtime_fingerprint=runtime,
        lock_payload={"prompt_fingerprint": "sha256:" + "d" * 64},
    )
    assert first["neuro_symbolic_router_locked"] is True
    assert first["exact_rejection_taxonomy_not_guaranteed"] is True
    assert first == write_router_lock(
        tmp_path,
        runtime_fingerprint=runtime,
        lock_payload={"prompt_fingerprint": "sha256:" + "d" * 64},
    )
    with pytest.raises(M5A41EvidenceError):
        write_router_lock(
            tmp_path,
            runtime_fingerprint=runtime,
            lock_payload={"prompt_fingerprint": "sha256:" + "e" * 64},
        )


def test_m5a41_evidence_is_atomic_and_completion_is_last(tmp_path: Path) -> None:
    owner = {"runtime_fingerprint": "sha256:" + "f" * 64, "mode": "fixture"}
    written = write_m5a41_evidence(
        tmp_path,
        owner=owner,
        artifacts={"protocol_amendment.json": {"fixture": True}, "summary.md": "# Test\n"},
        flags=initial_m5a41_flags(),
    )
    root = Path(cast(str, written["root"]))
    validated = validate_m5a41_evidence(root, expected_owner=owner)
    assert validated["passed"] is True
    assert (root / "complete.json").stat().st_mtime_ns >= (
        root / "manifest.json"
    ).stat().st_mtime_ns


def test_m5a41_evidence_rejects_path_escape(tmp_path: Path) -> None:
    with pytest.raises(M5A41EvidenceError):
        write_m5a41_evidence(
            tmp_path,
            owner={"runtime_fingerprint": "sha256:" + "1" * 64},
            artifacts={"../escape.json": {}},
            flags={},
        )


@pytest.mark.parametrize(
    "flag",
    [
        "language_final_accessed",
        "control_final_accessed",
        "test_split_accessed",
        "m42_final_accessed",
        "smolvla_go",
        "physical_target_validated",
    ],
)
def test_forbidden_and_physical_flags_default_false(flag: str) -> None:
    assert initial_m5a41_flags()[flag] is False
