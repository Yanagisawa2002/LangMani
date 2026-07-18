"""Independent read-only verifier for immutable M5A.4 evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from langmani.datasets.identity import canonical_json, sha256_hex
from langmani.language.corpus import build_language_corpus
from langmani.language.llm_router import (
    QWEN3_4B_INSTRUCT_FILE_IDENTITIES,
    QWEN3_4B_INSTRUCT_MODEL_ID,
    QWEN3_4B_INSTRUCT_REVISION,
)
from langmani.language.neuro_symbolic_evidence import validate_neuro_symbolic_evidence
from langmani.language.neuro_symbolic_experiment import (
    compute_neuro_symbolic_safety_metrics,
    evaluate_neuro_symbolic_development_gate,
)
from langmani.language.neuro_symbolic_router import (
    OUTLINES_LICENSE,
    OUTLINES_VERSION,
    DeterministicSafetyArbiterV0,
    SymbolicLexicalParserV0,
    build_semantic_frame_prompt,
    select_semantic_prompt_examples,
    semantic_schema_fingerprint,
)
from langmani.language.offline_router_metrics import (
    recompute_router_metrics,
    router_record_from_dict,
)
from langmani.language.router_types import LanguageSplit


class NeuroSymbolicVerificationError(RuntimeError):
    """Raised when evidence cannot support the claimed offline result."""


def _object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NeuroSymbolicVerificationError(f"could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise NeuroSymbolicVerificationError(f"{path} must contain one object")
    return cast(dict[str, object], value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _router_result(
    path: Path,
    *,
    examples: tuple,
) -> tuple[dict[str, object], tuple]:
    payload = _object(path)
    raw = payload.get("records")
    metrics = payload.get("metrics")
    if not isinstance(raw, list) or not isinstance(metrics, Mapping):
        raise NeuroSymbolicVerificationError(f"router result is malformed: {path}")
    records = tuple(
        router_record_from_dict(cast(Mapping[str, object], value))
        for value in raw
        if isinstance(value, Mapping)
    )
    if len(records) != len(raw):
        raise NeuroSymbolicVerificationError(f"router record is malformed: {path}")
    recomputed = recompute_router_metrics(records=records, examples=examples)
    if dict(metrics) != recomputed:
        raise NeuroSymbolicVerificationError(f"router metrics do not recompute: {path}")
    return payload, records


def verify_neuro_symbolic_evidence(evidence_root: str | Path) -> dict[str, object]:
    """Verify source identity, locks, raw records, gates, and forbidden access."""

    validated = validate_neuro_symbolic_evidence(evidence_root)
    root = Path(cast(str, validated["root"]))
    owner = cast(Mapping[str, object], validated["owner"])
    complete = cast(Mapping[str, object], validated["complete"])
    flags = complete.get("flags")
    if not isinstance(flags, Mapping):
        raise NeuroSymbolicVerificationError("completion flags are missing")
    model = _object(root / "model_identity.json")
    decoder = _object(root / "constrained_decoder_identity.json")
    prompt = _object(root / "prompt.json")
    runtime = _object(root / "runtime_identity.json")
    arbiter = _object(root / "arbiter_contract.json")
    semantic_schema = _object(root / "semantic_schema.json")
    symbolic = _object(root / "symbolic_contract.json")
    smoke = _object(root / "train_smoke.json")
    selection = _object(root / "candidate_selection.json")
    if (
        model.get("model_id") != QWEN3_4B_INSTRUCT_MODEL_ID
        or model.get("model_revision") != QWEN3_4B_INSTRUCT_REVISION
        or model.get("tokenizer_revision") != QWEN3_4B_INSTRUCT_REVISION
        or model.get("dtype") != "bfloat16"
        or model.get("quantization") != "none"
        or model.get("weights_unchanged") is not True
    ):
        raise NeuroSymbolicVerificationError("frozen Qwen3-4B identity differs")
    snapshot_value = model.get("snapshot_path")
    identities = model.get("file_identities")
    if not isinstance(snapshot_value, str) or not isinstance(identities, Mapping):
        raise NeuroSymbolicVerificationError("model file identity records are missing")
    snapshot = Path(snapshot_value)
    for name, (expected_size, expected_sha) in QWEN3_4B_INSTRUCT_FILE_IDENTITIES.items():
        record = identities.get(name)
        path = snapshot / name
        if (
            not isinstance(record, Mapping)
            or record.get("size_bytes") != expected_size
            or record.get("sha256") != f"sha256:{expected_sha}"
            or not path.is_file()
            or path.stat().st_size != expected_size
            or _sha256_file(path) != f"sha256:{expected_sha}"
        ):
            raise NeuroSymbolicVerificationError(f"model file identity differs: {name}")
    if (
        decoder.get("distribution") != "outlines"
        or decoder.get("version") != OUTLINES_VERSION
        or decoder.get("license") != OUTLINES_LICENSE
        or decoder.get("free_form_fallback") is not False
        or decoder.get("grammar_cached_for_session") is not True
    ):
        raise NeuroSymbolicVerificationError("constrained decoder identity differs")
    if prompt.get("train_only") is not True or prompt.get("prompt_sweep") is not False:
        raise NeuroSymbolicVerificationError("train-only prompt contract differs")
    if (
        runtime.get("optimizer_constructed") is not False
        or runtime.get("training_performed") is not False
        or runtime.get("robot_environment_created") is not False
        or runtime.get("controller_loaded") is not False
        or symbolic.get("controller_dispatch") is not False
        or arbiter.get("arbiter_version") is None
        or semantic_schema.get("schema") is None
        or smoke.get("example_count") != 20
        or runtime.get("visible_gpu_count") != 1
    ):
        raise NeuroSymbolicVerificationError("offline-only runtime contract differs")
    forbidden = (
        "language_final_accessed",
        "control_final_accessed",
        "test_split_accessed",
        "m42_final_accessed",
        "smolvla_go",
    )
    if any(flags.get(name) is not False for name in forbidden):
        raise NeuroSymbolicVerificationError("a prohibited source or stage was accessed")
    required_true = (
        "language_validation_quarantined_for_architecture_selection",
        "qwen4b_direct_candidate_frozen",
        "neuro_symbolic_implementation_validated",
        "symbolic_frame_validated",
        "semantic_frame_schema_validated",
        "constrained_decoding_validated",
        "arbiter_validated",
        "train_smoke_completed",
        "real_gpu_inference_validated",
    )
    if any(flags.get(name) is not True for name in required_true):
        raise NeuroSymbolicVerificationError("required implementation/GPU flags are false")

    corpus = build_language_corpus()
    parser = SymbolicLexicalParserV0()
    prompt_examples = select_semantic_prompt_examples(
        corpus.examples_for_split(LanguageSplit.TRAIN)
    )
    expected_prompt, expected_prompt_fingerprint, expected_prompt_ids = build_semantic_frame_prompt(
        examples=prompt_examples, parser=parser
    )
    if (
        prompt.get("prompt_content") != expected_prompt
        or prompt.get("prompt_fingerprint") != expected_prompt_fingerprint
        or prompt.get("few_shot_example_ids") != list(expected_prompt_ids)
    ):
        raise NeuroSymbolicVerificationError("train-only prompt differs")
    smoke_checks = smoke.get("checks")
    smoke_records = smoke.get("records")
    smoke_gate_passed = smoke.get("gate_passed") is True
    if (
        not isinstance(smoke_checks, Mapping)
        or not isinstance(smoke_records, list)
        or len(smoke_records) != 20
        or smoke.get("example_ids") != list(expected_prompt_ids)
        or smoke_gate_passed != all(value is True for value in smoke_checks.values())
    ):
        raise NeuroSymbolicVerificationError("train-only smoke evidence is malformed")
    for record in smoke_records:
        if not isinstance(record, Mapping):
            raise NeuroSymbolicVerificationError("train-only smoke record is malformed")
        semantic = record.get("semantic_frame")
        expected_semantic = record.get("expected_semantic_frame")
        if (
            not isinstance(semantic, Mapping)
            or not isinstance(expected_semantic, Mapping)
            or record.get("semantic_fields_exact")
            != (canonical_json(dict(semantic)) == canonical_json(dict(expected_semantic)))
        ):
            raise NeuroSymbolicVerificationError("train-only semantic comparison differs")
    if [record.get("example_id") for record in smoke_records] != list(expected_prompt_ids):
        raise NeuroSymbolicVerificationError("train-only smoke record ordering differs")
    if not smoke_gate_passed:
        optional_true = (
            "neuro_symbolic_router_locked",
            "historical_validation_diagnostic_completed",
            "language_development_completed",
            "neuro_symbolic_language_quality_gate_passed",
            "learned_router_selected",
            "one_scene_control_smoke_authorized",
        )
        forbidden_artifacts = (
            root / "prompt_lock.json",
            root / "historical_validation_diagnostic.json",
            root / "development",
            root.parent / ".locks" / root.name,
        )
        if (
            any(flags.get(name) is not False for name in optional_true)
            or any(path.exists() for path in forbidden_artifacts)
            or selection.get("stopped_at_train_smoke") is not True
            or selection.get("rejection_classification") != "train_only_semantic_smoke_failure"
        ):
            raise NeuroSymbolicVerificationError("train-smoke rejection boundary differs")
        return {
            "schema_version": "langmani-m5a4-independent-verification-v0",
            "passed": True,
            "artifact_fingerprint": validated["artifact_fingerprint"],
            "runtime_fingerprint": owner["runtime_fingerprint"],
            "exact_model_identity_validated": True,
            "constrained_decoding_validated": True,
            "prompt_lock_validated": False,
            "historical_diagnostic_non_selecting": False,
            "stopped_at_train_smoke": True,
            "development_metrics_recomputed": False,
            "development_gate": None,
            "forbidden_access_validated": True,
            "no_training_or_controller_validated": True,
            "flags": dict(flags),
        }

    prompt_lock = _object(root / "prompt_lock.json")
    historical = _object(root / "historical_validation_diagnostic.json")
    if (
        prompt_lock.get("neuro_symbolic_router_locked") is not True
        or prompt_lock.get("runtime_fingerprint") != owner.get("runtime_fingerprint")
        or prompt_lock.get("prompt_fingerprint") != prompt.get("prompt_fingerprint")
        or prompt_lock.get("symbolic_contract_fingerprint") != parser.contract_fingerprint
        or prompt_lock.get("arbiter_fingerprint")
        != DeterministicSafetyArbiterV0().contract_fingerprint
        or prompt_lock.get("semantic_schema_fingerprint") != semantic_schema_fingerprint()
    ):
        raise NeuroSymbolicVerificationError("prompt lock differs")
    side_lock = root.parent / ".locks" / root.name / "prompt_lock.json"
    if not side_lock.is_file() or _object(side_lock) != prompt_lock:
        raise NeuroSymbolicVerificationError("pre-development side lock differs")
    lock_identity = dict(prompt_lock)
    observed_lock_fingerprint = lock_identity.pop("lock_fingerprint", None)
    if observed_lock_fingerprint != f"sha256:{sha256_hex(lock_identity)}":
        raise NeuroSymbolicVerificationError("development lock fingerprint differs")
    if (
        historical.get("post_selection_diagnostic_only") is not True
        or historical.get("configuration_change_authorized") is not False
        or historical.get("configuration_fingerprints_before")
        != historical.get("configuration_fingerprints_after")
        or flags.get("neuro_symbolic_router_locked") is not True
        or flags.get("historical_validation_diagnostic_completed") is not True
    ):
        raise NeuroSymbolicVerificationError("historical diagnostic changed configuration")
    validation_examples = corpus.examples_for_split(LanguageSplit.VALIDATION)
    historical_raw = historical.get("records")
    historical_metrics = historical.get("metrics")
    if not isinstance(historical_raw, list) or not isinstance(historical_metrics, Mapping):
        raise NeuroSymbolicVerificationError("historical raw records are missing")
    historical_records = tuple(
        router_record_from_dict(cast(Mapping[str, object], value))
        for value in historical_raw
        if isinstance(value, Mapping)
    )
    if len(historical_records) != 300 or recompute_router_metrics(
        records=historical_records, examples=validation_examples
    ) != dict(historical_metrics):
        raise NeuroSymbolicVerificationError("historical diagnostic metrics do not recompute")

    development_root = root / "development"
    development_completed = flags.get("language_development_completed") is True
    recomputed_gate: dict[str, object] | None = None
    if development_completed:
        if prompt_lock.get("neuro_symbolic_router_locked") is not True:
            raise NeuroSymbolicVerificationError("development opened without the lock")
        examples = corpus.examples_for_split(LanguageSplit.DEVELOPMENT)
        if len(examples) != 420:
            raise NeuroSymbolicVerificationError("development example count differs")
        expected_names = (
            "rule_router",
            "classifier_negative_baseline",
            "qwen17b_negative_baseline",
            "qwen4b_direct_negative_baseline",
            "neuro_symbolic_candidate",
        )
        candidate_records = None
        for name in expected_names:
            payload, records = _router_result(development_root / f"{name}.json", examples=examples)
            eligibility = payload.get("eligibility")
            if not isinstance(eligibility, Mapping):
                raise NeuroSymbolicVerificationError("router eligibility is missing")
            promotable = name == "neuro_symbolic_candidate"
            if eligibility.get("promotion_eligible") is not promotable:
                raise NeuroSymbolicVerificationError("baseline promotion eligibility differs")
            if promotable:
                candidate_records = records
        assert candidate_records is not None
        candidate = _object(development_root / "neuro_symbolic_candidate.json")
        metrics = cast(Mapping[str, object], candidate["metrics"])
        safety = compute_neuro_symbolic_safety_metrics(records=candidate_records, examples=examples)
        recorded_safety = _object(development_root / "safety_metrics.json")
        if safety != recorded_safety:
            raise NeuroSymbolicVerificationError("safety metrics do not recompute")
        gate = evaluate_neuro_symbolic_development_gate(
            metrics=metrics,
            safety_metrics=safety,
            no_prohibited_source_access=True,
        )
        recorded_gate = _object(development_root / "quality_gate.json")
        if gate.to_dict() != recorded_gate:
            raise NeuroSymbolicVerificationError("development quality gate differs")
        if any(
            flags.get(name) is not gate.passed
            for name in (
                "neuro_symbolic_language_quality_gate_passed",
                "learned_router_selected",
                "one_scene_control_smoke_authorized",
            )
        ):
            raise NeuroSymbolicVerificationError("selection flags differ from the gate")
        recomputed_gate = gate.to_dict()
    elif development_root.exists():
        raise NeuroSymbolicVerificationError("development evidence exists without completion")

    return {
        "schema_version": "langmani-m5a4-independent-verification-v0",
        "passed": True,
        "artifact_fingerprint": validated["artifact_fingerprint"],
        "runtime_fingerprint": owner["runtime_fingerprint"],
        "exact_model_identity_validated": True,
        "constrained_decoding_validated": True,
        "prompt_lock_validated": True,
        "historical_diagnostic_non_selecting": True,
        "development_metrics_recomputed": development_completed,
        "development_gate": recomputed_gate,
        "forbidden_access_validated": True,
        "no_training_or_controller_validated": True,
        "flags": dict(flags),
    }


__all__ = ["NeuroSymbolicVerificationError", "verify_neuro_symbolic_evidence"]
