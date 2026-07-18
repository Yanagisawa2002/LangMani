"""Independent read-only verifier for one completed M5A.2 evidence root."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from langmani.datasets.identity import sha256_hex
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import stable_task_id
from langmani.language.corpus import GeneratedLanguageCorpus
from langmani.language.language_development_evidence import (
    validate_language_development_evidence,
)
from langmani.language.llm_router import QWEN3_1_7B_FILE_IDENTITIES
from langmani.language.offline_language_development import (
    CLASSIFIER_NEGATIVE_ELIGIBILITY,
    LOCAL_LLM_ELIGIBILITY,
    RULE_ROUTER_ELIGIBILITY,
    LanguageRouterSelectionV0,
    QwenModelIdentityV0,
    evaluate_llm_development_gate,
    evaluate_llm_validation_gate,
)
from langmani.language.offline_router_metrics import (
    recompute_router_metrics,
    router_record_from_dict,
)
from langmani.language.router_types import LanguageSplit


class LanguageDevelopmentVerificationError(RuntimeError):
    """Raised when independently recomputed M5A.2 evidence differs."""


def _read(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LanguageDevelopmentVerificationError(f"could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise LanguageDevelopmentVerificationError(f"expected one JSON object: {path}")
    return cast(dict[str, object], value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _router_payload_metrics(
    *, root: Path, relative: str, corpus: GeneratedLanguageCorpus, split: LanguageSplit
) -> tuple[dict[str, object], dict[str, object]]:
    payload = _read(root / relative)
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not all(
        isinstance(value, Mapping) for value in raw_records
    ):
        raise LanguageDevelopmentVerificationError(f"router raw records malformed: {relative}")
    records = tuple(
        router_record_from_dict(cast(Mapping[str, object], value)) for value in raw_records
    )
    recomputed = recompute_router_metrics(
        records=records, examples=corpus.examples_for_split(split)
    )
    if payload.get("metrics") != recomputed:
        raise LanguageDevelopmentVerificationError(f"router metrics differ: {relative}")
    return payload, recomputed


def verify_language_development_evidence(
    evidence_root: str | Path,
    *,
    corpus: GeneratedLanguageCorpus,
    classifier_checkpoint: str | Path,
) -> dict[str, object]:
    validated = validate_language_development_evidence(evidence_root)
    root = Path(cast(str, validated["root"]))
    owner = cast(Mapping[str, object], validated["owner"])
    complete = cast(Mapping[str, object], validated["complete"])
    result = _read(root / "result.json")
    flags = result.get("flags")
    complete_flags = complete.get("flags")
    if not isinstance(flags, Mapping) or complete_flags != flags:
        raise LanguageDevelopmentVerificationError("result/completion flags differ")
    model = _read(root / "model_identity.json")
    expected_model = QwenModelIdentityV0()
    if model.get("identity_fingerprint") != expected_model.fingerprint or any(
        model.get(key) != value for key, value in expected_model.to_dict().items()
    ):
        raise LanguageDevelopmentVerificationError("pinned Qwen model identity differs")
    if owner.get("mode") == "target_development":
        actual_files = model.get("actual_files")
        expected_files = {
            name: {"size_bytes": size, "sha256": f"sha256:{digest}"}
            for name, (size, digest) in QWEN3_1_7B_FILE_IDENTITIES.items()
        }
        if actual_files != expected_files or flags.get("real_gpu_inference_validated") is not True:
            raise LanguageDevelopmentVerificationError("real pinned GPU model files differ")
    prompt = _read(root / "prompt.json")
    smoke = _read(root / "train_smoke.json")
    train_ids = {value.example_id for value in corpus.examples_for_split(LanguageSplit.TRAIN)}
    prompt_ids = prompt.get("prompt_example_ids")
    smoke_ids = smoke.get("example_ids")
    if (
        not isinstance(prompt_ids, list)
        or not isinstance(smoke_ids, list)
        or not set(prompt_ids).issubset(train_ids)
        or not set(smoke_ids).issubset(train_ids)
        or prompt.get("prompt_example_split") != "train"
    ):
        raise LanguageDevelopmentVerificationError("prompt/smoke examples are not train-only")
    prompt_fingerprint = f"sha256:{sha256_hex({'prompt_template': prompt['prompt_text'], 'example_ids': tuple(prompt_ids)})}"
    if prompt_fingerprint != prompt.get("prompt_fingerprint") or prompt_fingerprint != owner.get(
        "prompt_fingerprint"
    ):
        raise LanguageDevelopmentVerificationError("prompt fingerprint differs")
    smoke_passed = smoke.get("gate_passed") is True
    prompt_lock_path = root / "prompt_lock.json"
    if smoke_passed:
        lock = _read(prompt_lock_path)
        if (
            lock.get("prompt_text") != prompt.get("prompt_text")
            or lock.get("prompt_fingerprint") != prompt_fingerprint
            or lock.get("prompt_example_ids") != prompt_ids
            or lock.get("locked_before_validation") is not True
            or lock.get("prompt_sweep_count") != 1
        ):
            raise LanguageDevelopmentVerificationError("prompt lock changed after smoke")
    elif prompt_lock_path.exists() or (root / "validation").exists():
        raise LanguageDevelopmentVerificationError("failed train smoke opened validation")
    generation = _read(root / "generation_config.json")
    if (
        generation.get("do_sample") is not False
        or generation.get("num_beams") != 1
        or generation.get("thinking_enabled") is not False
        or generation.get("maximum_format_repair_attempts") != 1
        or generation.get("chain_of_thought_requested") is not False
    ):
        raise LanguageDevelopmentVerificationError("generation/repair contract differs")
    controller_contract = _read(root / "controller_registry_contract.json")
    if controller_contract != {
        "schema_version": "langmani-m5a2-controller-registry-metadata-audit-v0",
        "metadata_only": True,
        "canonical_task_ids": [stable_task_id(value) for value in CANONICAL_TASK_SPECS],
        "controller_registry_loaded": False,
        "controller_checkpoint_loaded": False,
        "controller_dispatched": False,
        "environment_created": False,
        "environment_step_count": 0,
    }:
        raise LanguageDevelopmentVerificationError("controller boundary audit differs")
    negative = _read(root / "classifier_negative_baseline.json")
    if any(
        (
            negative.get("classifier_candidate_frozen") is not True,
            negative.get("classifier_runtime_selected") is not False,
            negative.get("classifier_full_quality_gate_passed") is not False,
            negative.get("additional_training_authorized") is not False,
            negative.get("additional_seed_authorized") is not False,
            negative.get("controller_dispatch_eligible") is not False,
            negative.get("promotion_eligible") is not False,
        )
    ):
        raise LanguageDevelopmentVerificationError("classifier negative-baseline role changed")
    checkpoint = Path(classifier_checkpoint).resolve(strict=False)
    identity = negative.get("identity")
    if identity is not None and (
        not isinstance(identity, Mapping)
        or identity.get("checkpoint_fingerprint") != _sha256_file(checkpoint)
        or identity.get("model_weights_unchanged") is not True
        or identity.get("model_state_fingerprint_before")
        != identity.get("model_state_fingerprint_after")
        or identity.get("optimizer_constructed") is not False
    ):
        raise LanguageDevelopmentVerificationError("classifier read-only audit differs")
    validation_completed = flags.get("llm_validation_completed") is True
    validation_gate_passed = False
    if validation_completed:
        for name in ("rule_router", "classifier_negative_baseline", "local_llm"):
            _router_payload_metrics(
                root=root,
                relative=f"validation/{name}.json",
                corpus=corpus,
                split=LanguageSplit.VALIDATION,
            )
        _, llm_validation = _router_payload_metrics(
            root=root,
            relative="validation/local_llm.json",
            corpus=corpus,
            split=LanguageSplit.VALIDATION,
        )
        validation_gate = evaluate_llm_validation_gate(
            metrics=llm_validation, prohibited_source_access=False
        )
        if _read(root / "validation" / "gate.json") != validation_gate.to_dict():
            raise LanguageDevelopmentVerificationError("LLM validation gate differs")
        validation_gate_passed = validation_gate.passed
        if flags.get("llm_validation_gate_passed") is not validation_gate_passed:
            raise LanguageDevelopmentVerificationError("LLM validation flag differs")
    development_completed = flags.get("language_development_completed") is True
    development_gate_passed = False
    if development_completed:
        if not validation_gate_passed:
            raise LanguageDevelopmentVerificationError("development bypassed validation")
        for name in ("rule_router", "classifier_negative_baseline", "local_llm"):
            _router_payload_metrics(
                root=root,
                relative=f"development/{name}.json",
                corpus=corpus,
                split=LanguageSplit.DEVELOPMENT,
            )
        _, llm_development = _router_payload_metrics(
            root=root,
            relative="development/local_llm.json",
            corpus=corpus,
            split=LanguageSplit.DEVELOPMENT,
        )
        safety = _read(root / "development" / "safety_probes.json")
        safety_checks = safety.get("checks")
        if not isinstance(safety_checks, Mapping):
            raise LanguageDevelopmentVerificationError("development safety probes malformed")
        development_gate = evaluate_llm_development_gate(
            metrics=llm_development,
            per_task_accuracy=cast(Mapping[str, object], llm_development["per_task_accuracy"]),
            rejection_family_false_route_rates=cast(
                Mapping[str, object], llm_development["rejection_family_false_route_rates"]
            ),
            safety_checks=cast(Mapping[str, bool], safety_checks),
        )
        if _read(root / "development" / "gate.json") != development_gate.to_dict():
            raise LanguageDevelopmentVerificationError("LLM development gate differs")
        development_gate_passed = development_gate.passed
    elif (root / "development").exists():
        raise LanguageDevelopmentVerificationError("stopped evidence contains development files")
    if (
        flags.get("llm_language_quality_gate_passed") is not development_gate_passed
        or flags.get("learned_router_selected") is not development_gate_passed
        or flags.get("one_scene_control_smoke_authorized") is not development_gate_passed
    ):
        raise LanguageDevelopmentVerificationError("selection/development gate flags differ")
    selection_payload = _read(root / "candidate_selection.json")
    if (
        selection_payload.get("classifier") != CLASSIFIER_NEGATIVE_ELIGIBILITY.to_dict()
        or selection_payload.get("rule_router") != RULE_ROUTER_ELIGIBILITY.to_dict()
        or selection_payload.get("local_llm") != LOCAL_LLM_ELIGIBILITY.to_dict()
        or selection_payload.get("one_scene_control_smoke_authorized")
        is not development_gate_passed
    ):
        raise LanguageDevelopmentVerificationError("candidate eligibility/authorization differs")
    selected_router = selection_payload.get("selected_learned_router")
    if development_gate_passed:
        git_commit = owner.get("git_commit")
        if not isinstance(git_commit, str):
            raise LanguageDevelopmentVerificationError("selection omitted producer Git identity")
        validation_payload = _read(root / "validation" / "local_llm.json")
        development_payload = _read(root / "development" / "local_llm.json")
        parser_fingerprint = f"sha256:{sha256_hex('strict-router-json-reason-v1')}"
        validation_fingerprint = f"sha256:{sha256_hex(validation_payload)}"
        development_fingerprint = f"sha256:{sha256_hex(development_payload)}"
        generation_fingerprint = f"sha256:{sha256_hex(generation)}"
        selection_identity = {
            "model": expected_model.fingerprint,
            "prompt": prompt_fingerprint,
            "parser": parser_fingerprint,
            "generation": generation_fingerprint,
            "validation": validation_fingerprint,
            "development": development_fingerprint,
            "git_commit": git_commit,
            "corpus": corpus.manifest.corpus_fingerprint,
        }
        expected_selection = LanguageRouterSelectionV0(
            runtime_fingerprint=f"sha256:{sha256_hex(selection_identity)}",
            model_identity_fingerprint=expected_model.fingerprint,
            prompt_fingerprint=prompt_fingerprint,
            parser_fingerprint=parser_fingerprint,
            generation_config_fingerprint=generation_fingerprint,
            validation_evidence_fingerprint=validation_fingerprint,
            development_evidence_fingerprint=development_fingerprint,
            git_commit=git_commit,
            corpus_fingerprint=corpus.manifest.corpus_fingerprint,
            validation_split_fingerprint=corpus.manifest.split_manifests[
                LanguageSplit.VALIDATION
            ].content_fingerprint,
            development_split_fingerprint=corpus.manifest.split_manifests[
                LanguageSplit.DEVELOPMENT
            ].content_fingerprint,
        )
        if selected_router != expected_selection.to_dict():
            raise LanguageDevelopmentVerificationError("learned-router selection differs")
    elif selected_router is not None:
        raise LanguageDevelopmentVerificationError("failed LLM quality gate selected a router")
    prohibited = cast(Mapping[str, object], result.get("final_access_prohibitions", {}))
    if any(
        prohibited.get(name) not in {False}
        for name in (
            "language_final_accessed",
            "control_final_accessed",
            "test_split_accessed",
            "historical_fresh_accessed",
            "m42_final_accessed",
            "smolvla_go",
            "controller_loaded",
            "environment_created",
            "environment_step_count",
        )
    ):
        raise LanguageDevelopmentVerificationError("a prohibited source or runtime was accessed")
    return {
        "schema_version": "langmani-m5a2-independent-language-development-verification-v0",
        "passed": True,
        "implementation_validated": True,
        "classifier_candidate_frozen": True,
        "classifier_offline_baseline_validated": flags.get("classifier_offline_baseline_validated")
        is True,
        "classifier_dispatch_prohibited": True,
        "rule_router_validation_completed": flags.get("rule_router_validation_completed") is True,
        "llm_model_identity_validated": flags.get("llm_model_identity_validated") is True,
        "llm_prompt_locked": flags.get("llm_prompt_locked") is True,
        "llm_train_smoke_completed": flags.get("llm_train_smoke_completed") is True,
        "llm_validation_completed": validation_completed,
        "llm_validation_gate_passed": validation_gate_passed,
        "language_development_completed": development_completed,
        "llm_language_quality_gate_passed": development_gate_passed,
        "learned_router_selected": development_gate_passed,
        "one_scene_control_smoke_authorized": development_gate_passed,
        "one_scene_control_smoke_completed": False,
        "three_scene_control_screen_completed": False,
        "predicted_control_development_completed": False,
        "language_final_accessed": False,
        "control_final_accessed": False,
        "test_split_accessed": False,
        "m42_final_accessed": False,
        "smolvla_go": False,
        "real_gpu_inference_validated": flags.get("real_gpu_inference_validated") is True,
        "physical_target_validated": False,
        "runtime_fingerprint": owner.get("runtime_fingerprint"),
        "artifact_fingerprint": validated["artifact_fingerprint"],
        "evidence_root": str(root),
    }


__all__ = [
    "LanguageDevelopmentVerificationError",
    "verify_language_development_evidence",
]
