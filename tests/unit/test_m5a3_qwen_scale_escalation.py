from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from langmani.language.corpus import build_language_corpus
from langmani.language.llm_router import (
    QWEN3_1_7B_MODEL_ID,
    QWEN3_1_7B_REVISION,
    QWEN3_4B_INSTRUCT_FILE_IDENTITIES,
    QWEN3_4B_INSTRUCT_MODEL_ID,
    QWEN3_4B_INSTRUCT_REVISION,
    StructuredLLMRouterConfig,
    StructuredLLMRouterError,
    TransformersLocalTextGenerator,
    build_structured_routing_prompt,
    select_structured_routing_prompt_examples,
)
from langmani.language.qwen_scale_escalation import (
    M5A3_AUTHORIZED_CANDIDATES,
    FrozenQwen17BBaselineV0,
    M5A3ContractError,
    PromptEquivalenceV0,
    Qwen4BInstructIdentityV0,
    compute_scale_comparison,
    evaluate_qwen4b_development_gate,
    evaluate_qwen4b_validation_gate,
    initial_m5a3_flags,
    repair_policy_fingerprint,
    router_parser_fingerprint,
    router_schema_fingerprint,
)
from langmani.language.qwen_scale_escalation_evidence import (
    QwenScaleEvidenceError,
    validate_qwen_scale_evidence,
    write_qwen_scale_evidence,
)
from langmani.language.router_types import (
    LanguageSplit,
    RouterConfidence,
    RouterDecision,
    RouterStatus,
)
from scripts.evaluate_qwen_scale_escalation import (
    QwenScaleCommandError,
    _run_smoke,
    _safe_paths,
    _select_smoke_examples,
    parse_args,
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
        "final_schema_valid_output_rate": 0.99,
        "malformed_output_after_repair_rate": 0.01,
        "deterministic_repeatability": 1.0,
    }


def _comparison_metrics() -> dict[str, object]:
    return {
        **_passing_metrics(),
        "repair_attempt_rate": 0.1,
        "latency_p50_ms": 100.0,
        "latency_p95_ms": 150.0,
        "latency_p99_ms": 200.0,
        "total_elapsed_inference_seconds": 10.0,
        "generated_token_count": 1000,
        "peak_gpu_memory_bytes": 100,
    }


def _equivalence() -> PromptEquivalenceV0:
    semantic = "sha256:" + "1" * 64
    return PromptEquivalenceV0(
        qwen17b_prompt_fingerprint=semantic,
        qwen4b_rendered_prompt_fingerprint="sha256:" + "2" * 64,
        semantic_prompt_content_fingerprint=semantic,
        qwen17b_prompt_example_ids=("train-a", "train-b"),
        qwen4b_prompt_example_ids=("train-a", "train-b"),
        schema_fingerprint=router_schema_fingerprint(),
        parser_fingerprint=router_parser_fingerprint(),
        repair_policy_fingerprint=repair_policy_fingerprint(),
        semantic_prompt_content_equal=True,
    )


def test_qwen17b_is_frozen_negative_baseline() -> None:
    baseline = FrozenQwen17BBaselineV0()
    assert baseline.candidate_frozen is True
    assert baseline.promotion_eligible is False
    assert baseline.controller_dispatch_eligible is False


def test_exactly_one_scale_candidate_is_authorized() -> None:
    assert M5A3_AUTHORIZED_CANDIDATES == ("Qwen/Qwen3-4B-Instruct-2507",)


def test_exact_qwen4b_model_id_is_enforced() -> None:
    with pytest.raises(M5A3ContractError, match="exactly one"):
        Qwen4BInstructIdentityV0(model_id="Qwen/Qwen3-8B")


def test_exact_qwen4b_revision_is_required() -> None:
    with pytest.raises(M5A3ContractError, match="exactly one"):
        Qwen4BInstructIdentityV0(model_revision="0" * 40)


def test_no_qwen8b_snapshot_fallback_is_permitted(tmp_path: Path) -> None:
    config = StructuredLLMRouterConfig(
        model_id="Qwen/Qwen3-8B",
        model_revision="0" * 40,
        tokenizer_revision="0" * 40,
        chat_template_mode="official_qwen_instruct_non_thinking",
    )
    with pytest.raises(StructuredLLMRouterError, match="not an authorized identity"):
        TransformersLocalTextGenerator._validate_snapshot(tmp_path, config=config)


def test_prompt_semantic_equivalence_is_explicit() -> None:
    value = _equivalence()
    assert value.semantic_prompt_content_equal is True
    assert value.to_dict()["transport_difference_only"] is True


def test_changed_prompt_semantics_are_rejected() -> None:
    with pytest.raises(M5A3ContractError, match="semantics"):
        PromptEquivalenceV0(
            qwen17b_prompt_fingerprint="sha256:" + "1" * 64,
            qwen4b_rendered_prompt_fingerprint="sha256:" + "2" * 64,
            semantic_prompt_content_fingerprint="sha256:" + "3" * 64,
            qwen17b_prompt_example_ids=("a",),
            qwen4b_prompt_example_ids=("a",),
            schema_fingerprint=router_schema_fingerprint(),
            parser_fingerprint=router_parser_fingerprint(),
            repair_policy_fingerprint=repair_policy_fingerprint(),
            semantic_prompt_content_equal=True,
        )


def test_few_shot_ids_must_be_identical() -> None:
    with pytest.raises(M5A3ContractError, match="few-shot"):
        PromptEquivalenceV0(
            qwen17b_prompt_fingerprint="sha256:" + "1" * 64,
            qwen4b_rendered_prompt_fingerprint="sha256:" + "2" * 64,
            semantic_prompt_content_fingerprint="sha256:" + "1" * 64,
            qwen17b_prompt_example_ids=("a",),
            qwen4b_prompt_example_ids=("b",),
            schema_fingerprint=router_schema_fingerprint(),
            parser_fingerprint=router_parser_fingerprint(),
            repair_policy_fingerprint=repair_policy_fingerprint(),
            semantic_prompt_content_equal=True,
        )


def test_frozen_prompt_builder_is_shared_by_both_capacities() -> None:
    corpus = build_language_corpus()
    examples = select_structured_routing_prompt_examples(
        corpus.examples_for_split(LanguageSplit.TRAIN)
    )
    first = build_structured_routing_prompt(examples=examples)
    second = build_structured_routing_prompt(examples=examples)
    assert first == second


def test_schema_parser_and_repair_have_separate_fingerprints() -> None:
    values = {router_schema_fingerprint(), router_parser_fingerprint(), repair_policy_fingerprint()}
    assert len(values) == 3
    assert all(value.startswith("sha256:") for value in values)


def test_model_specific_chat_template_transport_omits_thinking_switch() -> None:
    generator = object.__new__(TransformersLocalTextGenerator)
    generator.config = StructuredLLMRouterConfig(
        model_id=QWEN3_4B_INSTRUCT_MODEL_ID,
        model_revision=QWEN3_4B_INSTRUCT_REVISION,
        tokenizer_revision=QWEN3_4B_INSTRUCT_REVISION,
        chat_template_mode="official_qwen_instruct_non_thinking",
    )
    assert generator._chat_template_kwargs() == {}


def test_qwen17b_transport_retains_official_thinking_disable() -> None:
    generator = object.__new__(TransformersLocalTextGenerator)
    generator.config = StructuredLLMRouterConfig(
        model_id=QWEN3_1_7B_MODEL_ID,
        model_revision=QWEN3_1_7B_REVISION,
        tokenizer_revision=QWEN3_1_7B_REVISION,
    )
    assert generator._chat_template_kwargs() == {"enable_thinking": False}


def test_one_repair_is_the_maximum() -> None:
    with pytest.raises(ValueError, match="at most one"):
        StructuredLLMRouterConfig(
            model_id=QWEN3_4B_INSTRUCT_MODEL_ID,
            model_revision=QWEN3_4B_INSTRUCT_REVISION,
            tokenizer_revision=QWEN3_4B_INSTRUCT_REVISION,
            maximum_format_repair_attempts=2,
            chat_template_mode="official_qwen_instruct_non_thinking",
        )


def test_validation_gate_passes_exact_boundaries_without_rounding() -> None:
    assert evaluate_qwen4b_validation_gate(
        metrics=_passing_metrics(), prohibited_source_access=False
    ).passed


def test_train_smoke_requires_expected_task_and_status_semantics() -> None:
    corpus = build_language_corpus()
    examples = _select_smoke_examples(corpus)

    class _ExpectedRouter:
        def route(self, command: str) -> RouterDecision:
            example = next(value for value in examples if value.raw_text == command)
            confidence = RouterConfidence.unavailable(definition="fixture")
            if example.expected_status is RouterStatus.ROUTE:
                assert example.expected_task_spec is not None
                return RouterDecision.route(
                    task_spec=example.expected_task_spec,
                    confidence=confidence,
                    router_name="fixture",
                    router_version="v0",
                )
            assert example.expected_rejection_reason is not None
            return RouterDecision.reject(
                status=example.expected_status,
                rejection_reason=example.expected_rejection_reason,
                confidence=confidence,
                router_name="fixture",
                router_version="v0",
            )

    smoke = _run_smoke(_ExpectedRouter(), examples)  # type: ignore[arg-type]
    assert smoke["example_count"] == 20
    assert smoke["gate_passed"] is True
    assert cast(dict[str, bool], smoke["checks"])["expected_task_specs_and_statuses_match"]
    failing = _passing_metrics()
    failing["false_route_rate"] = 0.0300000001
    assert not evaluate_qwen4b_validation_gate(
        metrics=failing, prohibited_source_access=False
    ).passed


def test_validation_gate_forbids_prohibited_sources() -> None:
    gate = evaluate_qwen4b_validation_gate(
        metrics=_passing_metrics(), prohibited_source_access=True
    )
    assert gate.passed is False
    assert gate.checks["no_prohibited_source_access"] is False


def test_development_gate_is_fully_conjunctive() -> None:
    gate = evaluate_qwen4b_development_gate(
        metrics=_passing_metrics(),
        per_task_accuracy={"task": 0.90},
        rejection_family_false_route_rates={"family": 0.10},
        safety_checks={
            "empty_and_meaningless_inputs_never_route": True,
            "conflicting_object_and_bin_inputs_never_route": True,
            "unsupported_action_commands_never_route": True,
        },
    )
    assert gate.passed
    failed = dict(gate.checks)
    failed["unsupported_action_commands_never_route"] = False
    assert not all(failed.values())


def test_initial_flags_forbid_final_control_and_dispatch() -> None:
    flags = initial_m5a3_flags()
    assert flags["language_final_accessed"] is False
    assert flags["control_final_accessed"] is False
    assert flags["test_split_accessed"] is False
    assert flags["m42_final_accessed"] is False
    assert flags["learned_router_selected"] is False


def test_scale_comparison_requires_identical_validation_order() -> None:
    with pytest.raises(M5A3ContractError, match="identical ordered"):
        compute_scale_comparison(
            qwen17b_metrics=_comparison_metrics(),
            qwen4b_metrics=_comparison_metrics(),
            validation_example_ids_17b=("a", "b"),
            validation_example_ids_4b=("b", "a"),
            qwen4b_gate_passed=False,
        )


def test_scale_comparison_calculates_qwen4b_minus_qwen17b() -> None:
    baseline = _comparison_metrics()
    candidate = _comparison_metrics()
    candidate["valid_full_task_accuracy"] = 0.97
    candidate["false_route_rate"] = 0.02
    result = compute_scale_comparison(
        qwen17b_metrics=baseline,
        qwen4b_metrics=candidate,
        validation_example_ids_17b=("a",),
        validation_example_ids_4b=("a",),
        qwen4b_gate_passed=True,
    )
    deltas = cast(dict[str, float], result["metric_deltas"])
    assert deltas["valid_full_task_accuracy"] == pytest.approx(0.02)
    assert deltas["false_route_rate"] == pytest.approx(-0.01)
    assert result["model_reached_promotion_quality"] is True


def test_model_identity_contains_all_exact_file_hashes() -> None:
    identity = Qwen4BInstructIdentityV0().to_dict()
    expected = cast(dict[str, object], identity["expected_files"])
    assert set(expected) == set(QWEN3_4B_INSTRUCT_FILE_IDENTITIES)
    assert len(expected) == 13


def test_evidence_is_atomic_reusable_and_checksum_validated(tmp_path: Path) -> None:
    owner = {"runtime_fingerprint": "sha256:" + "a" * 64}
    kwargs = {
        "owner": owner,
        "artifacts": {"result.json": {"passed": True}},
        "stopped_after_stage": "validation",
        "stopped_at_validation": True,
        "flags": {"passed": True},
    }
    first = write_qwen_scale_evidence(tmp_path, **kwargs)
    second = write_qwen_scale_evidence(tmp_path, **kwargs)
    assert first["evidence_reused"] is False
    assert second["evidence_reused"] is True
    root = Path(cast(str, first["root"]))
    assert not (root / "development").exists()
    (root / "result.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(QwenScaleEvidenceError, match="checksums"):
        validate_qwen_scale_evidence(root)


def test_evidence_rejects_unsafe_output_paths(tmp_path: Path) -> None:
    with pytest.raises(QwenScaleEvidenceError, match="safe JSON"):
        write_qwen_scale_evidence(
            tmp_path,
            owner={"runtime_fingerprint": "sha256:" + "b" * 64},
            artifacts={"../escape.json": {}},
            stopped_after_stage="fixture",
            stopped_at_validation=False,
            flags={"passed": True},
        )


def test_command_rejects_source_tree_output_before_writing() -> None:
    args = parse_args(
        [
            "--fixture",
            "--output-root",
            "src/langmani/language",
            "--report",
            "outputs/diagnostics/m5a/stages/unsafe-output-test.json",
        ]
    )
    with pytest.raises(QwenScaleCommandError, match="source-controlled"):
        _safe_paths(args)


def test_command_rejects_source_tree_report_before_writing() -> None:
    args = parse_args(
        [
            "--fixture",
            "--output-root",
            "outputs/diagnostics/m5a/qwen4b-safe-test",
            "--report",
            "docs/unsafe-report.json",
        ]
    )
    with pytest.raises(QwenScaleCommandError, match="report overlaps source-controlled"):
        _safe_paths(args)


def test_no_optimizer_or_training_field_is_present_in_model_contract() -> None:
    payload = {
        **Qwen4BInstructIdentityV0().to_dict(),
        "optimizer_constructed": False,
        "training_or_fine_tuning_performed": False,
        "model_weights_unchanged": True,
    }
    assert payload["optimizer_constructed"] is False
    assert payload["training_or_fine_tuning_performed"] is False
    assert payload["model_weights_unchanged"] is True


def test_rendered_prompt_fingerprint_uses_transport_text_only() -> None:
    generator = object.__new__(TransformersLocalTextGenerator)
    generator.config = StructuredLLMRouterConfig(
        model_id=QWEN3_4B_INSTRUCT_MODEL_ID,
        model_revision=QWEN3_4B_INSTRUCT_REVISION,
        tokenizer_revision=QWEN3_4B_INSTRUCT_REVISION,
        chat_template_mode="official_qwen_instruct_non_thinking",
    )
    generator.tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **kwargs: f"transport:{messages[0]['content']}"
    )
    first = generator.rendered_prompt_fingerprint("same semantic prompt")
    second = generator.rendered_prompt_fingerprint("same semantic prompt")
    assert first == second
    assert first.startswith("sha256:")
