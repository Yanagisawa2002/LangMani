"""Independent read-only verifier for M5A.4.1 safety-gate evidence."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

from langmani.datasets.identity import sha256_hex
from langmani.language.corpus import build_language_corpus
from langmani.language.neuro_symbolic_router import (
    OUTLINES_VERSION,
    DeterministicSafetyArbiterV0,
    SymbolicLexicalParserV0,
    build_semantic_frame_prompt,
    select_semantic_prompt_examples,
    semantic_frame_schema,
    semantic_schema_fingerprint,
)
from langmani.language.neuro_symbolic_safety import (
    evaluate_development_safety_gate,
    evaluate_router_contracts,
    evaluate_train_smoke_contracts,
    evaluate_train_smoke_gate,
    rejection_taxonomy_contract,
    safety_contract,
    special_safety_route_counts,
    taxonomy_audit,
)
from langmani.language.neuro_symbolic_safety_evidence import (
    read_object,
    validate_m5a41_evidence,
)
from langmani.language.neuro_symbolic_verifier import verify_neuro_symbolic_evidence
from langmani.language.offline_router_metrics import (
    recompute_router_metrics,
    router_record_from_dict,
)
from langmani.language.router_evaluation import RouterEvaluationRecord
from langmani.language.router_types import LanguageSplit


class M5A41VerificationError(RuntimeError):
    """Raised when immutable evidence cannot support an M5A.4.1 claim."""


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise M5A41VerificationError(f"{label} must be an object")
    return cast(Mapping[str, object], value)


def _router_result(
    path: Path, *, examples: tuple
) -> tuple[dict[str, object], tuple[RouterEvaluationRecord, ...]]:
    payload = read_object(path)
    raw = payload.get("records")
    metrics = payload.get("metrics")
    if not isinstance(raw, list) or not isinstance(metrics, Mapping):
        raise M5A41VerificationError(f"router result is malformed: {path}")
    records = tuple(
        router_record_from_dict(cast(Mapping[str, object], value))
        for value in raw
        if isinstance(value, Mapping)
    )
    if len(records) != len(raw):
        raise M5A41VerificationError(f"router record is malformed: {path}")
    recomputed = recompute_router_metrics(records=records, examples=examples)
    if dict(metrics) != recomputed:
        raise M5A41VerificationError(f"router metrics do not recompute: {path}")
    safety, taxonomy = evaluate_router_contracts(records=records, examples=examples)
    if payload.get("safety_evaluation") != safety.to_dict():
        raise M5A41VerificationError(f"safety metrics do not recompute: {path}")
    if payload.get("taxonomy_evaluation") != taxonomy.to_dict():
        raise M5A41VerificationError(f"taxonomy metrics do not recompute: {path}")
    return payload, records


def verify_m5a41_evidence(evidence_root: str | Path) -> dict[str, object]:
    """Recompute contracts, raw metrics, gates, locks, and forbidden-access flags."""

    validated = validate_m5a41_evidence(evidence_root)
    root = Path(cast(str, validated["root"]))
    owner = _mapping(validated["owner"], label="owner")
    complete = _mapping(validated["complete"], label="completion")
    flags = _mapping(complete.get("flags"), label="completion flags")
    required_true = (
        "taxonomy_audit_completed",
        "taxonomy_contract_consistent",
        "safety_routing_contract_validated",
        "rejection_taxonomy_contract_validated",
        "train_smoke_safety_gate_passed",
        "neuro_symbolic_router_locked",
        "historical_validation_diagnostic_completed",
        "language_development_completed",
        "real_gpu_inference_validated",
    )
    if any(flags.get(name) is not True for name in required_true):
        raise M5A41VerificationError("required M5A.4.1 completion flag is false")
    forbidden = (
        "language_final_accessed",
        "control_final_accessed",
        "test_split_accessed",
        "m42_final_accessed",
        "smolvla_go",
        "physical_target_validated",
    )
    if any(flags.get(name) is not False for name in forbidden):
        raise M5A41VerificationError("a forbidden source, stage, or physical claim was accessed")

    safety_payload = read_object(root / "safety_contract.json")
    taxonomy_payload = read_object(root / "taxonomy_contract.json")
    if safety_payload != safety_contract() or taxonomy_payload != rejection_taxonomy_contract():
        raise M5A41VerificationError("safety/taxonomy contract payload differs")

    reassessment = read_object(root / "train_smoke_reassessment.json")
    source_root_value = reassessment.get("source_evidence_root")
    if not isinstance(source_root_value, str):
        raise M5A41VerificationError("source M5A.4 evidence root is missing")
    source_root = Path(source_root_value)
    source_verified = verify_neuro_symbolic_evidence(source_root)
    if (
        source_verified.get("stopped_at_train_smoke") is not True
        or reassessment.get("source_runtime_fingerprint")
        != source_verified.get("runtime_fingerprint")
        or reassessment.get("source_artifact_fingerprint")
        != source_verified.get("artifact_fingerprint")
        or reassessment.get("source_output_records_reused") is not True
        or reassessment.get("model_inference_rerun") is not False
    ):
        raise M5A41VerificationError("source M5A.4 evidence binding differs")
    source_smoke = read_object(source_root / "train_smoke.json")
    source_owner = read_object(source_root / "owner.json")
    source_prompt = read_object(source_root / "prompt.json")
    source_model = read_object(source_root / "model_identity.json")
    source_symbolic = read_object(source_root / "symbolic_contract.json")
    source_arbiter = read_object(source_root / "arbiter_contract.json")
    source_schema = read_object(source_root / "semantic_schema.json")

    corpus = build_language_corpus()
    parser = SymbolicLexicalParserV0()
    arbiter = DeterministicSafetyArbiterV0()
    prompt_examples = select_semantic_prompt_examples(
        corpus.examples_for_split(LanguageSplit.TRAIN)
    )
    prompt_content, prompt_fingerprint, prompt_ids = build_semantic_frame_prompt(
        examples=prompt_examples, parser=parser
    )
    if (
        source_prompt.get("prompt_content") != prompt_content
        or source_prompt.get("prompt_fingerprint") != prompt_fingerprint
        or source_prompt.get("few_shot_example_ids") != list(prompt_ids)
        or source_symbolic != parser.contract_dict()
        or source_arbiter != arbiter.contract_dict()
        or source_schema.get("schema") != semantic_frame_schema()
        or source_schema.get("schema_fingerprint") != semantic_schema_fingerprint()
    ):
        raise M5A41VerificationError("frozen router inputs differ from source M5A.4")

    audit = taxonomy_audit(corpus=corpus, source_smoke=source_smoke)
    if read_object(root / "taxonomy_audit.json") != audit:
        raise M5A41VerificationError("taxonomy audit does not recompute")
    if (
        audit.get("taxonomy_contract_consistent") is not True
        or audit.get("source_smoke_outputs_unchanged") is not True
        or audit.get("smoke_mismatch_count") != 3
    ):
        raise M5A41VerificationError("taxonomy audit conclusion differs")

    raw_smoke = source_smoke.get("records")
    if not isinstance(raw_smoke, list):
        raise M5A41VerificationError("source smoke records are missing")
    smoke_safety, smoke_taxonomy = evaluate_train_smoke_contracts(
        records=tuple(cast(Mapping[str, object], value) for value in raw_smoke),
        examples=prompt_examples,
    )
    smoke_gate = evaluate_train_smoke_gate(safety=smoke_safety, no_prohibited_source_access=True)
    if (
        reassessment.get("safety_evaluation") != smoke_safety.to_dict()
        or reassessment.get("taxonomy_evaluation") != smoke_taxonomy.to_dict()
        or reassessment.get("safety_gate") != smoke_gate.to_dict()
        or reassessment.get("taxonomy_gate_passed") is not smoke_taxonomy.exact_quality_passed
        or flags.get("train_smoke_safety_gate_passed") is not smoke_gate.passed
        or flags.get("train_smoke_taxonomy_gate_passed") is not smoke_taxonomy.exact_quality_passed
    ):
        raise M5A41VerificationError("train-smoke reassessment does not recompute")

    amendment = read_object(root / "protocol_amendment.json")
    amendment_identity = dict(amendment)
    amendment_fingerprint = amendment_identity.pop("amendment_fingerprint", None)
    if amendment_fingerprint != f"sha256:{sha256_hex(amendment_identity)}":
        raise M5A41VerificationError("protocol amendment fingerprint differs")
    expected_frozen = (
        "original_gate_required_exact_router_status",
        "observed_failures_all_safe_rejections",
        "amended_before_development_or_final_access",
        "model_behavior_unchanged",
        "prompt_behavior_unchanged",
        "corpus_labels_unchanged",
        "development_thresholds_unchanged_except_gate_separation",
        "final_remains_sealed",
    )
    if (
        any(amendment.get(name) is not True for name in expected_frozen)
        or amendment.get("source_router_runtime_fingerprint")
        != source_owner.get("runtime_fingerprint")
        or amendment.get("source_train_smoke_artifact_fingerprint")
        != f"sha256:{sha256_hex(source_smoke)}"
    ):
        raise M5A41VerificationError("protocol amendment source or frozen boundary differs")

    router_lock = read_object(root / "router_lock.json")
    side_lock = root.parent / ".locks" / root.name / "router_lock.json"
    if not side_lock.is_file() or read_object(side_lock) != router_lock:
        raise M5A41VerificationError("pre-access router lock differs")
    lock_identity = dict(router_lock)
    lock_fingerprint = lock_identity.pop("lock_fingerprint", None)
    if (
        lock_fingerprint != f"sha256:{sha256_hex(lock_identity)}"
        or router_lock.get("neuro_symbolic_router_locked") is not True
        or router_lock.get("exact_rejection_taxonomy_not_guaranteed") is not True
        or router_lock.get("protocol_amendment_fingerprint") != amendment_fingerprint
        or router_lock.get("prompt_fingerprint") != prompt_fingerprint
        or router_lock.get("semantic_schema_fingerprint") != semantic_schema_fingerprint()
        or router_lock.get("symbolic_contract_fingerprint") != parser.contract_fingerprint
        or router_lock.get("arbiter_fingerprint") != arbiter.contract_fingerprint
        or router_lock.get("outlines_version") != OUTLINES_VERSION
        or router_lock.get("model_file_identities") != source_model.get("file_identities")
    ):
        raise M5A41VerificationError("immutable router lock differs")

    runtime = read_object(root / "runtime_identity.json")
    if (
        runtime.get("optimizer_constructed") is not False
        or runtime.get("training_performed") is not False
        or runtime.get("weights_modified") is not False
        or runtime.get("prompt_modified") is not False
        or runtime.get("few_shot_examples_modified") is not False
        or runtime.get("semantic_schema_modified") is not False
        or runtime.get("symbolic_parser_modified") is not False
        or runtime.get("arbiter_modified") is not False
        or runtime.get("robot_environment_created") is not False
        or runtime.get("controller_loaded") is not False
        or runtime.get("env_step_called") is not False
        or runtime.get("development_access_after_lock") is not True
        or runtime.get("historical_validation_access_after_lock") is not True
        or runtime.get("visible_gpu_count") != 1
    ):
        raise M5A41VerificationError("offline immutable runtime contract differs")
    immutability = read_object(root / "immutability_identity.json")
    if (
        immutability.get("source_model_identity") != source_model
        or immutability.get("loaded_model_file_identities") != source_model.get("file_identities")
        or immutability.get("source_prompt_fingerprint") != prompt_fingerprint
        or immutability.get("source_symbolic_contract_fingerprint") != parser.contract_fingerprint
        or immutability.get("source_semantic_schema_fingerprint") != semantic_schema_fingerprint()
        or immutability.get("source_arbiter_fingerprint") != arbiter.contract_fingerprint
    ):
        raise M5A41VerificationError("model/prompt/schema/parser/arbiter immutability differs")

    validation_examples = corpus.examples_for_split(LanguageSplit.VALIDATION)
    historical, _ = _router_result(
        root / "historical_validation_diagnostic.json", examples=validation_examples
    )
    if (
        len(cast(list[object], historical["records"])) != 300
        or historical.get("post_selection_diagnostic_only") is not True
        or historical.get("configuration_change_authorized") is not False
        or historical.get("configuration_fingerprints_before")
        != historical.get("configuration_fingerprints_after")
        or historical.get("started_after_router_lock") is not True
    ):
        raise M5A41VerificationError("historical validation diagnostic contract differs")

    development_examples = corpus.examples_for_split(LanguageSplit.DEVELOPMENT)
    if len(development_examples) != 420:
        raise M5A41VerificationError("development split count differs")
    expected_names = (
        "rule_router",
        "classifier_negative_baseline",
        "qwen17b_negative_baseline",
        "qwen4b_negative_baseline",
        "neuro_symbolic_candidate",
    )
    candidate_records: tuple[RouterEvaluationRecord, ...] | None = None
    candidate_payload: Mapping[str, object] | None = None
    for name in expected_names:
        payload, records = _router_result(
            root / "development" / f"{name}.json", examples=development_examples
        )
        eligibility = _mapping(payload.get("eligibility"), label=f"{name} eligibility")
        promotable = name == "neuro_symbolic_candidate"
        if eligibility.get("promotion_eligible") is not promotable:
            raise M5A41VerificationError("baseline promotion eligibility differs")
        if promotable:
            candidate_records = records
            candidate_payload = payload
    assert candidate_records is not None and candidate_payload is not None
    metrics = _mapping(candidate_payload["metrics"], label="candidate metrics")
    safety, taxonomy = evaluate_router_contracts(
        records=candidate_records, examples=development_examples
    )
    special_counts = special_safety_route_counts(
        records=candidate_records, examples=development_examples
    )
    family_rates = cast(Mapping[str, float], metrics["rejection_family_false_route_rates"])
    gate = evaluate_development_safety_gate(
        metrics=metrics,
        safety=safety,
        special_route_counts=special_counts,
        rejection_family_unsafe_false_route_rates=family_rates,
        no_prohibited_source_access=True,
    )
    if read_object(root / "development" / "neuro_symbolic_safety_gate.json") != gate.to_dict():
        raise M5A41VerificationError("development safety gate does not recompute")
    selection = read_object(root / "candidate_selection.json")
    if (
        selection.get("learned_router_selected") is not gate.passed
        or selection.get("one_scene_control_smoke_authorized") is not gate.passed
        or flags.get("neuro_symbolic_language_safety_gate_passed") is not gate.passed
        or flags.get("learned_router_selected") is not gate.passed
        or flags.get("one_scene_control_smoke_authorized") is not gate.passed
        or flags.get("exact_rejection_taxonomy_quality_passed") is not taxonomy.exact_quality_passed
    ):
        raise M5A41VerificationError("selection flags differ from independently recomputed gates")
    limitations = read_object(root / "limitations.json")
    if (
        limitations.get("reliable_exact_rejection_reason_classification_claimed") is not False
        or limitations.get("calibrated_rejection_taxonomy_claimed") is not False
        or limitations.get("robot_control_executed") is not False
    ):
        raise M5A41VerificationError("taxonomy or control limitations were overstated")

    return {
        "schema_version": "langmani-m5a41-independent-verification-v1",
        "passed": True,
        "runtime_fingerprint": owner["runtime_fingerprint"],
        "artifact_fingerprint": validated["artifact_fingerprint"],
        "source_m5a4_artifact_fingerprint": source_verified["artifact_fingerprint"],
        "taxonomy_audit_recomputed": True,
        "safety_taxonomy_separation_validated": True,
        "source_smoke_outputs_unchanged": True,
        "prompt_schema_parser_arbiter_unchanged": True,
        "model_weights_unchanged": True,
        "router_lock_precedes_validation_and_development": True,
        "historical_validation_diagnostic_only": True,
        "historical_validation_example_count": 300,
        "development_example_count": 420,
        "development_metrics_recomputed": True,
        "development_safety_gate": gate.to_dict(),
        "development_taxonomy_quality_passed": taxonomy.exact_quality_passed,
        "forbidden_access_validated": True,
        "no_training_or_controller_validated": True,
        "flags": dict(flags),
    }


__all__ = ["M5A41VerificationError", "verify_m5a41_evidence"]
