from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import nn

from langmani.datasets.identity import sha256_hex
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import TaskSpec, stable_task_id
from langmani.language.classifier_decoders import DecoderCandidate, DecoderConfigurationV0
from langmani.language.corpus import GeneratedLanguageCorpus, build_language_corpus
from langmani.language.language_development_evidence import (
    LanguageDevelopmentEvidenceError,
    validate_language_development_evidence,
    write_language_development_evidence,
)
from langmani.language.language_development_verifier import (
    LanguageDevelopmentVerificationError,
    verify_language_development_evidence,
)
from langmani.language.llm_router import (
    QWEN3_1_7B_MODEL_ID,
    QWEN3_1_7B_REVISION,
    StructuredLLMRouterConfig,
    StructuredLocalLLMRouterV0,
    build_structured_routing_prompt,
    select_structured_routing_prompt_examples,
)
from langmani.language.offline_language_development import (
    CLASSIFIER_NEGATIVE_ELIGIBILITY,
    LOCAL_LLM_ELIGIBILITY,
    RULE_ROUTER_ELIGIBILITY,
    CandidateEligibilityV0,
    FrozenClassifierNegativeBaselineV0,
    LanguageCandidateRole,
    LanguageRouterSelectionV0,
    M5A2ContractError,
    PromptLockV0,
    QwenModelIdentityV0,
    evaluate_llm_development_gate,
    evaluate_llm_validation_gate,
)
from langmani.language.offline_router_metrics import recompute_router_metrics
from langmani.language.rejection_diagnostics import ReadOnlyClassifierBundleV0
from langmani.language.router_evaluation import RouterEvaluationRecord, evaluate_language_router
from langmani.language.router_types import (
    LanguageSplit,
    RouterConfidence,
    RouterDecision,
    RouterStatus,
)
from langmani.language.rule_router import RuleRouterV0
from langmani.language.schema_validation import RouterSchemaError, parse_strict_router_json


class _QueuedGenerator:
    def __init__(self, *outputs: str) -> None:
        self.outputs = deque(outputs)

    def generate(self, prompt: str, *, max_new_tokens: int) -> str:
        assert prompt and max_new_tokens == 64
        return self.outputs.popleft()


def _config() -> StructuredLLMRouterConfig:
    return StructuredLLMRouterConfig(
        model_id=QWEN3_1_7B_MODEL_ID,
        model_revision=QWEN3_1_7B_REVISION,
        tokenizer_revision=QWEN3_1_7B_REVISION,
        dtype="bfloat16",
        maximum_new_tokens=64,
    )


def _route_text() -> str:
    return json.dumps(
        {
            "status": "route",
            "target_object_id": "red_cube",
            "target_bin_id": "left_bin",
            "reason": "explicit_object_and_destination",
        },
        sort_keys=True,
    )


def _reject_text() -> str:
    return json.dumps(
        {
            "status": "reject_ambiguous",
            "target_object_id": None,
            "target_bin_id": None,
            "reason": "missing_object",
        },
        sort_keys=True,
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


def test_classifier_is_frozen_negative_baseline() -> None:
    assert CLASSIFIER_NEGATIVE_ELIGIBILITY.offline_evaluation_eligible is True
    assert CLASSIFIER_NEGATIVE_ELIGIBILITY.promotion_eligible is False
    assert CLASSIFIER_NEGATIVE_ELIGIBILITY.final_selection_eligible is False


def test_classifier_dispatch_is_prohibited() -> None:
    assert CLASSIFIER_NEGATIVE_ELIGIBILITY.controller_dispatch_eligible is False
    with pytest.raises(M5A2ContractError, match="never dispatch"):
        CandidateEligibilityV0(
            candidate_id="FactorizedTextClassifierNegativeBaselineV0",
            role=LanguageCandidateRole.CLASSIFIER_NEGATIVE_BASELINE,
            offline_evaluation_eligible=True,
            controller_dispatch_eligible=True,
            promotion_eligible=False,
            final_selection_eligible=False,
            deterministic_baseline=False,
        )


def test_rule_router_is_offline_deterministic_baseline() -> None:
    assert RULE_ROUTER_ELIGIBILITY.deterministic_baseline is True
    assert RULE_ROUTER_ELIGIBILITY.offline_evaluation_eligible is True
    assert RULE_ROUTER_ELIGIBILITY.promotion_eligible is False


def test_qwen_model_identity_serializes_pinned_files() -> None:
    payload = QwenModelIdentityV0().to_dict()
    assert payload["model_id"] == QWEN3_1_7B_MODEL_ID
    assert payload["model_revision"] == QWEN3_1_7B_REVISION
    assert payload["tokenizer_revision"] == QWEN3_1_7B_REVISION
    assert cast(dict[str, object], payload["expected_files"])["config.json"]


def test_qwen_exact_revision_is_required() -> None:
    with pytest.raises(M5A2ContractError, match="exactly one pinned"):
        QwenModelIdentityV0(model_revision="a" * 40)


def test_prompt_examples_are_train_only() -> None:
    corpus = build_language_corpus()
    examples = select_structured_routing_prompt_examples(
        corpus.examples_for_split(LanguageSplit.TRAIN)
    )
    assert all(example.split is LanguageSplit.TRAIN for example in examples)
    assert len({example.task_id for example in examples if example.task_id is not None}) == 6


def test_prompt_lock_rejects_changed_prompt() -> None:
    corpus = build_language_corpus()
    examples = select_structured_routing_prompt_examples(
        corpus.examples_for_split(LanguageSplit.TRAIN)
    )
    prompt, fingerprint, ids = build_structured_routing_prompt(examples=examples)
    with pytest.raises(M5A2ContractError, match="fingerprint differs"):
        PromptLockV0(
            prompt_text=prompt + " changed",
            prompt_fingerprint=fingerprint,
            prompt_example_ids=ids,
            train_smoke_example_ids=ids,
            corpus_fingerprint=corpus.manifest.corpus_fingerprint,
            locked_before_validation=True,
        )


def test_structured_schema_parses_exact_object() -> None:
    assert parse_strict_router_json(_route_text()).status == "route"


def test_route_output_constructs_supported_task() -> None:
    decision = StructuredLocalLLMRouterV0(
        config=_config(), generator=_QueuedGenerator(_route_text())
    ).route("command")
    assert decision.task_spec == TaskSpec("red_cube", "left_bin", "canonical_v0")


def test_rejected_output_has_no_executable_task() -> None:
    decision = StructuredLocalLLMRouterV0(
        config=_config(), generator=_QueuedGenerator(_reject_text())
    ).route("command")
    assert decision.status is RouterStatus.REJECT_AMBIGUOUS
    assert decision.task_spec is None


def test_unknown_object_is_rejected_by_schema() -> None:
    with pytest.raises(RouterSchemaError, match="target_object_id"):
        parse_strict_router_json(_route_text().replace("red_cube", "yellow_cube"))


def test_unknown_bin_is_rejected_by_schema() -> None:
    with pytest.raises(RouterSchemaError, match="target_bin_id"):
        parse_strict_router_json(_route_text().replace("left_bin", "middle_bin"))


def test_multiple_json_objects_are_rejected() -> None:
    with pytest.raises(RouterSchemaError, match="one strict JSON"):
        parse_strict_router_json(_route_text() + _route_text())


def test_free_form_trailing_text_is_rejected() -> None:
    with pytest.raises(RouterSchemaError, match="one strict JSON"):
        parse_strict_router_json(_route_text() + " execute this now")


def test_router_uses_one_bounded_repair() -> None:
    router = StructuredLocalLLMRouterV0(
        config=_config(), generator=_QueuedGenerator("not json", _route_text())
    )
    decision = router.route("command")
    assert decision.status is RouterStatus.ROUTE
    assert decision.evidence["format_repair_attempts"] == 1


def test_router_never_uses_a_second_repair() -> None:
    generator = _QueuedGenerator("not json", "still not json", _route_text())
    decision = StructuredLocalLLMRouterV0(config=_config(), generator=generator).route("command")
    assert decision.status is RouterStatus.REJECT_MALFORMED
    assert len(generator.outputs) == 1


def test_generation_contract_is_deterministic() -> None:
    payload = _config().to_dict()
    assert payload["do_sample"] is False
    assert payload["num_beams"] == 1
    assert payload["thinking_enabled"] is False
    assert payload["eos_behavior"] == "tokenizer_eos_token_id"


def test_chain_of_thought_text_is_not_persisted() -> None:
    private = "private chain of thought"
    decision = StructuredLocalLLMRouterV0(
        config=_config(), generator=_QueuedGenerator(private, _reject_text())
    ).route("command")
    assert decision.evidence["chain_of_thought_persisted"] is False
    assert private not in repr(decision.evidence)
    assert decision.evidence["generation_outputs"][0]["stored"] is False


def test_validation_gate_uses_exact_unrounded_boundaries() -> None:
    assert evaluate_llm_validation_gate(
        metrics=_passing_metrics(), prohibited_source_access=False
    ).passed
    failing = _passing_metrics()
    failing["valid_full_task_accuracy"] = 0.949999999
    assert not evaluate_llm_validation_gate(metrics=failing, prohibited_source_access=False).passed


def test_development_gate_is_fully_conjunctive() -> None:
    passing = evaluate_llm_development_gate(
        metrics=_passing_metrics(),
        per_task_accuracy={"task": 0.90},
        rejection_family_false_route_rates={"family": 0.10},
        safety_checks={
            "empty_and_meaningless_inputs_never_route": True,
            "conflicting_object_and_bin_inputs_never_route": True,
            "unsupported_action_commands_never_route": True,
        },
    )
    assert passing.passed
    assert not evaluate_llm_development_gate(
        metrics=_passing_metrics(),
        per_task_accuracy={"task": 0.899999},
        rejection_family_false_route_rates={"family": 0.10},
        safety_checks={
            "empty_and_meaningless_inputs_never_route": True,
            "conflicting_object_and_bin_inputs_never_route": True,
            "unsupported_action_commands_never_route": True,
        },
    ).passed


def test_prompt_cannot_change_after_lock() -> None:
    corpus = build_language_corpus()
    examples = select_structured_routing_prompt_examples(
        corpus.examples_for_split(LanguageSplit.TRAIN)
    )
    prompt, fingerprint, ids = build_structured_routing_prompt(examples=examples)
    lock = PromptLockV0(
        prompt_text=prompt,
        prompt_fingerprint=fingerprint,
        prompt_example_ids=ids,
        train_smoke_example_ids=ids,
        corpus_fingerprint=corpus.manifest.corpus_fingerprint,
        locked_before_validation=True,
    )
    assert lock.to_dict()["prompt_sweep_count"] == 1
    assert lock.to_dict()["locked_before_validation"] is True


def test_model_switching_is_not_permitted() -> None:
    with pytest.raises(M5A2ContractError):
        QwenModelIdentityV0(model_id="Qwen/Qwen3-4B")


def _stopped_evidence_artifacts(corpus: GeneratedLanguageCorpus) -> tuple[dict[str, object], str]:
    examples = select_structured_routing_prompt_examples(
        corpus.examples_for_split(LanguageSplit.TRAIN)
    )
    prompt, fingerprint, ids = build_structured_routing_prompt(examples=examples)
    config = _config()
    flags = {
        "classifier_candidate_frozen": True,
        "classifier_offline_baseline_validated": False,
        "classifier_dispatch_prohibited": True,
        "rule_router_validation_completed": False,
        "llm_model_identity_validated": False,
        "llm_prompt_locked": False,
        "llm_train_smoke_completed": True,
        "llm_validation_completed": False,
        "llm_validation_gate_passed": False,
        "language_development_completed": False,
        "llm_language_quality_gate_passed": False,
        "learned_router_selected": False,
        "one_scene_control_smoke_authorized": False,
        "real_gpu_inference_validated": False,
    }
    model = QwenModelIdentityV0()
    artifacts: dict[str, object] = {
        "model_identity.json": {
            **model.to_dict(),
            "identity_fingerprint": model.fingerprint,
            "actual_files": {},
            "parameter_count": None,
            "model_loaded_once": True,
            "license_reviewed": False,
            "model_card_reviewed": False,
        },
        "prompt.json": {
            "prompt_text": prompt,
            "prompt_fingerprint": fingerprint,
            "prompt_example_ids": list(ids),
            "prompt_example_split": "train",
        },
        "generation_config.json": config.to_dict(),
        "controller_registry_contract.json": {
            "schema_version": "langmani-m5a2-controller-registry-metadata-audit-v0",
            "metadata_only": True,
            "canonical_task_ids": [stable_task_id(value) for value in CANONICAL_TASK_SPECS],
            "controller_registry_loaded": False,
            "controller_checkpoint_loaded": False,
            "controller_dispatched": False,
            "environment_created": False,
            "environment_step_count": 0,
        },
        "train_smoke.json": {
            "gate_passed": False,
            "example_ids": [value.example_id for value in examples],
        },
        "classifier_negative_baseline.json": {
            **CLASSIFIER_NEGATIVE_ELIGIBILITY.to_dict(),
            "classifier_candidate_frozen": True,
            "classifier_runtime_selected": False,
            "classifier_full_quality_gate_passed": False,
            "additional_training_authorized": False,
            "additional_seed_authorized": False,
            "controller_dispatch_eligible": False,
            "promotion_eligible": False,
            "identity": None,
        },
        "candidate_selection.json": {
            "selected_learned_router": None,
            "one_scene_control_smoke_authorized": False,
            "classifier": CLASSIFIER_NEGATIVE_ELIGIBILITY.to_dict(),
            "rule_router": RULE_ROUTER_ELIGIBILITY.to_dict(),
            "local_llm": LOCAL_LLM_ELIGIBILITY.to_dict(),
        },
        "result.json": {
            "flags": flags,
            "final_access_prohibitions": {
                "language_final_accessed": False,
                "control_final_accessed": False,
                "test_split_accessed": False,
                "historical_fresh_accessed": False,
                "m42_final_accessed": False,
                "smolvla_go": False,
                "controller_loaded": False,
                "environment_created": False,
                "environment_step_count": 0,
            },
        },
    }
    return artifacts, fingerprint


def test_independent_verifier_accepts_correctly_stopped_smoke(tmp_path: Path) -> None:
    corpus = build_language_corpus()
    artifacts, fingerprint = _stopped_evidence_artifacts(corpus)
    owner = {"mode": "fixture", "prompt_fingerprint": fingerprint}
    owner["runtime_fingerprint"] = f"sha256:{sha256_hex(owner)}"
    flags = cast(dict[str, object], artifacts["result.json"])["flags"]
    written = write_language_development_evidence(
        tmp_path / "evidence",
        owner=owner,
        artifacts=artifacts,
        stopped_after_stage="train_smoke",
        flags=cast(dict[str, bool], flags),
    )
    checkpoint = tmp_path / "classifier.pt"
    checkpoint.write_bytes(b"fixture")

    verified = verify_language_development_evidence(
        cast(str, written["root"]), corpus=corpus, classifier_checkpoint=checkpoint
    )

    assert verified["passed"] is True
    assert verified["llm_validation_completed"] is False
    assert verified["one_scene_control_smoke_authorized"] is False


def test_validation_failure_forbids_development_files(tmp_path: Path) -> None:
    corpus = build_language_corpus()
    artifacts, fingerprint = _stopped_evidence_artifacts(corpus)
    artifacts["development/illegal.json"] = {"opened": True}
    owner = {"mode": "fixture", "prompt_fingerprint": fingerprint}
    owner["runtime_fingerprint"] = f"sha256:{sha256_hex(owner)}"
    written = write_language_development_evidence(
        tmp_path / "evidence",
        owner=owner,
        artifacts=artifacts,
        stopped_after_stage="train_smoke",
        flags=cast(dict[str, bool], artifacts["result.json"])["flags"],
    )
    checkpoint = tmp_path / "classifier.pt"
    checkpoint.write_bytes(b"fixture")
    with pytest.raises(LanguageDevelopmentVerificationError, match="development files"):
        verify_language_development_evidence(
            cast(str, written["root"]), corpus=corpus, classifier_checkpoint=checkpoint
        )


def test_final_access_remains_false_in_stopped_evidence() -> None:
    corpus = build_language_corpus()
    artifacts, _ = _stopped_evidence_artifacts(corpus)
    result = cast(dict[str, object], artifacts["result.json"])
    prohibited = cast(dict[str, object], result["final_access_prohibitions"])
    assert prohibited["language_final_accessed"] is False
    assert prohibited["control_final_accessed"] is False
    assert prohibited["test_split_accessed"] is False


def test_only_llm_can_be_locked_as_learned_selection() -> None:
    digest = "sha256:" + "a" * 64
    selection = LanguageRouterSelectionV0(
        runtime_fingerprint=digest,
        model_identity_fingerprint=digest,
        prompt_fingerprint=digest,
        parser_fingerprint=digest,
        generation_config_fingerprint=digest,
        validation_evidence_fingerprint=digest,
        development_evidence_fingerprint=digest,
        git_commit="b" * 40,
        corpus_fingerprint=digest,
        validation_split_fingerprint=digest,
        development_split_fingerprint=digest,
    )
    assert selection.to_dict()["selected_router"] == "StructuredLocalLLMRouterV0"
    assert LOCAL_LLM_ELIGIBILITY.promotion_eligible is True


def test_rule_router_cannot_be_relabeled_as_llm() -> None:
    with pytest.raises(M5A2ContractError, match="relabeled"):
        CandidateEligibilityV0(
            candidate_id="StructuredLocalLLMRouterV0",
            role=LanguageCandidateRole.RULE_BASELINE,
            offline_evaluation_eligible=True,
            controller_dispatch_eligible=False,
            promotion_eligible=False,
            final_selection_eligible=False,
            deterministic_baseline=True,
        )


def test_evidence_is_atomically_promoted_reused_and_checksum_checked(tmp_path: Path) -> None:
    owner = {"runtime_fingerprint": "sha256:" + "c" * 64}
    kwargs = {
        "owner": owner,
        "artifacts": {"result.json": {"value": 1}},
        "stopped_after_stage": "train_smoke",
        "flags": {"passed": True},
    }
    first = write_language_development_evidence(tmp_path, **kwargs)
    second = write_language_development_evidence(tmp_path, **kwargs)
    assert first["evidence_reused"] is False
    assert second["evidence_reused"] is True
    result = Path(cast(str, first["root"])) / "result.json"
    result.write_text("{}\n", encoding="utf-8")
    with pytest.raises(LanguageDevelopmentEvidenceError, match="checksums"):
        validate_language_development_evidence(Path(cast(str, first["root"])))


def test_metrics_recompute_from_raw_record() -> None:
    corpus = build_language_corpus()
    example = next(
        value
        for value in corpus.examples_for_split(LanguageSplit.VALIDATION)
        if value.expected_status is RouterStatus.ROUTE
    )
    decision = RuleRouterV0().route(example.raw_text)
    record = RouterEvaluationRecord(
        example_id=example.example_id,
        template_family_id=example.template_family_id,
        split=LanguageSplit.VALIDATION,
        expected_status=example.expected_status,
        expected_task_id=example.task_id,
        expected_rejection_reason=None,
        decision=decision,
        latency_ms=1.0,
        repeat_decision_fingerprints=(decision.decision_fingerprint,) * 2,
    )
    metrics = recompute_router_metrics(records=(record,), examples=(example,))
    assert metrics["example_count"] == 1
    assert metrics["deterministic_repeatability"] == 1.0


class _TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> object:
        del input_ids, attention_mask
        return SimpleNamespace(
            status_logits=torch.tensor([[10.0, 0.0, 0.0, 0.0]]),
            object_logits=torch.tensor([[10.0, 0.0, 0.0, 0.0]]),
            bin_logits=torch.tensor([[10.0, 0.0, 0.0]]),
        )


class _TinyTokenizer:
    def __call__(self, _text: str, **_kwargs: object) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor([[1]], dtype=torch.long),
            "attention_mask": torch.tensor([[1]], dtype=torch.long),
        }


def test_classifier_inference_constructs_no_optimizer_and_preserves_weights() -> None:
    model = _TinyClassifier()
    bundle = ReadOnlyClassifierBundleV0(
        model=cast(object, model),
        tokenizer=_TinyTokenizer(),
        processor_state={},
        run_fingerprint="sha256:" + "1" * 64,
        checkpoint_fingerprint="sha256:" + "2" * 64,
        checkpoint_step=116,
        checkpoint_size_bytes=1,
        model_state_fingerprint="sha256:" + "3" * 64,
        tokenizer_fingerprint="sha256:" + "4" * 64,
        processor_fingerprint="sha256:" + "5" * 64,
    )
    configuration = DecoderConfigurationV0(
        candidate=DecoderCandidate.CONSERVATIVE_ROUTE_DECODER_V0,
        temperature_mode="identity",
        status_temperature=1.0,
        route_threshold=0.4,
        route_margin_threshold=-0.2,
        object_confidence_threshold=0.4,
        bin_confidence_threshold=0.4,
    )
    router = FrozenClassifierNegativeBaselineV0(
        bundle=bundle,
        configuration=configuration,
        maximum_sequence_length=8,
    )
    router.route("Pick the red cube and place it in the left bin.")
    assert router.verify_unchanged() is True
    assert not hasattr(router, "optimizer")


def test_evidence_rejects_unsafe_output_paths(tmp_path: Path) -> None:
    with pytest.raises(LanguageDevelopmentEvidenceError, match="safe JSON/Markdown"):
        write_language_development_evidence(
            tmp_path,
            owner={"runtime_fingerprint": "sha256:" + "d" * 64},
            artifacts={"../escape.json": {}},
            stopped_after_stage="fixture",
            flags={"passed": True},
        )


def test_repeatability_uses_semantics_not_variable_runtime_telemetry() -> None:
    corpus = build_language_corpus()
    example = next(
        value
        for value in corpus.examples_for_split(LanguageSplit.VALIDATION)
        if value.expected_status is RouterStatus.ROUTE
    )

    class _VariableEvidenceRouter:
        calls = 0

        def route(self, _command: str) -> RouterDecision:
            self.calls += 1
            assert example.expected_task_spec is not None
            return RouterDecision.route(
                task_spec=example.expected_task_spec,
                confidence=RouterConfidence.unavailable(definition="fixture"),
                router_name="fixture",
                router_version="v0",
                evidence={"variable_latency": self.calls},
            )

    records, _ = evaluate_language_router(
        router=_VariableEvidenceRouter(),
        examples=(example,),
        split=LanguageSplit.VALIDATION,
        repeat_count=2,
        repeat_fingerprint=lambda value: f"{value.status.value}:{value.task_id}",
    )
    assert records[0].decision.decision_fingerprint != records[0].repeat_decision_fingerprints[0]
    assert records[0].deterministic is True
