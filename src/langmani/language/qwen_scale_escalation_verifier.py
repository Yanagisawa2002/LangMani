"""Independent read-only verifier for one completed M5A.3 evidence root."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch

from langmani.datasets.identity import sha256_hex
from langmani.language.corpus import GeneratedLanguageCorpus
from langmani.language.language_development_verifier import (
    verify_language_development_evidence,
)
from langmani.language.llm_router import QWEN3_4B_INSTRUCT_FILE_IDENTITIES
from langmani.language.offline_router_metrics import (
    recompute_router_metrics,
    router_record_from_dict,
)
from langmani.language.qwen_scale_escalation import (
    FrozenQwen17BBaselineV0,
    PromptEquivalenceV0,
    Qwen4BInstructIdentityV0,
    compute_scale_comparison,
    evaluate_qwen4b_development_gate,
    evaluate_qwen4b_validation_gate,
    repair_policy_fingerprint,
    router_parser_fingerprint,
    router_schema_fingerprint,
)
from langmani.language.qwen_scale_escalation_evidence import validate_qwen_scale_evidence
from langmani.language.router_types import LanguageSplit


class QwenScaleVerificationError(RuntimeError):
    """Raised when independently recomputed M5A.3 evidence differs."""


def _read(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QwenScaleVerificationError(f"could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise QwenScaleVerificationError(f"expected one JSON object: {path}")
    return cast(dict[str, object], value)


def _router_metrics(
    *, root: Path, relative: str, corpus: GeneratedLanguageCorpus, split: LanguageSplit
) -> tuple[dict[str, object], dict[str, object], tuple[str, ...]]:
    payload = _read(root / relative)
    raw = payload.get("records")
    if not isinstance(raw, list) or not all(isinstance(value, Mapping) for value in raw):
        raise QwenScaleVerificationError(f"router records malformed: {relative}")
    records = tuple(router_record_from_dict(cast(Mapping[str, object], value)) for value in raw)
    recomputed = recompute_router_metrics(
        records=records, examples=corpus.examples_for_split(split)
    )
    if payload.get("metrics") != recomputed:
        raise QwenScaleVerificationError(f"router metrics differ: {relative}")
    return payload, recomputed, tuple(record.example_id for record in records)


def verify_qwen_scale_evidence(
    evidence_root: str | Path,
    *,
    corpus: GeneratedLanguageCorpus,
    classifier_checkpoint: str | Path,
    qwen17b_evidence_root: str | Path,
) -> dict[str, object]:
    validated = validate_qwen_scale_evidence(evidence_root)
    root = Path(cast(str, validated["root"]))
    owner = cast(Mapping[str, object], validated["owner"])
    complete = cast(Mapping[str, object], validated["complete"])
    result = _read(root / "result.json")
    flags = result.get("flags")
    if not isinstance(flags, Mapping) or complete.get("flags") != flags:
        raise QwenScaleVerificationError("result/completion flags differ")
    model = _read(root / "model_identity.json")
    expected_model = Qwen4BInstructIdentityV0()
    if model.get("identity_fingerprint") != expected_model.fingerprint or any(
        model.get(key) != value for key, value in expected_model.to_dict().items()
    ):
        raise QwenScaleVerificationError("pinned Qwen3-4B identity differs")
    target = owner.get("mode") == "target_development"
    expected_files = {
        name: {"size_bytes": size, "sha256": f"sha256:{digest}"}
        for name, (size, digest) in QWEN3_4B_INSTRUCT_FILE_IDENTITIES.items()
    }
    expected_model_files = {
        name: value
        for name, value in expected_files.items()
        if name.startswith("model") or name in {"config.json", "generation_config.json"}
    }
    expected_tokenizer_files = {
        name: value
        for name, value in expected_files.items()
        if name in {"merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"}
    }
    if target and (
        model.get("actual_files") != expected_files
        or model.get("model_files") != expected_model_files
        or model.get("tokenizer_files") != expected_tokenizer_files
        or model.get("license") != "apache-2.0"
        or model.get("instruct_checkpoint") is not True
        or model.get("dtype") != "bfloat16"
        or model.get("device") != "cuda"
        or model.get("quantization") != "none"
        or model.get("model_loaded_once") is not True
        or model.get("visible_gpu_count") != 1
    ):
        raise QwenScaleVerificationError("real Qwen3-4B model/runtime identity differs")
    if (
        model.get("optimizer_constructed") is not False
        or model.get("training_or_fine_tuning_performed") is not False
    ):
        raise QwenScaleVerificationError("M5A.3 evidence performed model training")
    prompt = _read(root / "prompt_equivalence.json")
    try:
        equivalence = PromptEquivalenceV0(
            qwen17b_prompt_fingerprint=cast(str, prompt["qwen17b_prompt_fingerprint"]),
            qwen4b_rendered_prompt_fingerprint=cast(
                str, prompt["qwen4b_rendered_prompt_fingerprint"]
            ),
            semantic_prompt_content_fingerprint=cast(
                str, prompt["semantic_prompt_content_fingerprint"]
            ),
            qwen17b_prompt_example_ids=tuple(cast(list[str], prompt["qwen17b_prompt_example_ids"])),
            qwen4b_prompt_example_ids=tuple(cast(list[str], prompt["qwen4b_prompt_example_ids"])),
            schema_fingerprint=cast(str, prompt["schema_fingerprint"]),
            parser_fingerprint=cast(str, prompt["parser_fingerprint"]),
            repair_policy_fingerprint=cast(str, prompt["repair_policy_fingerprint"]),
            semantic_prompt_content_equal=cast(bool, prompt["semantic_prompt_content_equal"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise QwenScaleVerificationError(f"prompt equivalence is malformed: {error}") from error
    if (
        equivalence.to_dict() != prompt
        or equivalence.schema_fingerprint != router_schema_fingerprint()
        or equivalence.parser_fingerprint != router_parser_fingerprint()
        or equivalence.repair_policy_fingerprint != repair_policy_fingerprint()
        or owner.get("prompt_equivalence_fingerprint") != equivalence.fingerprint
    ):
        raise QwenScaleVerificationError("prompt/schema/parser/repair identities differ")
    generation = _read(root / "generation_config.json")
    if (
        generation.get("do_sample") is not False
        or generation.get("num_beams") != 1
        or generation.get("maximum_format_repair_attempts") != 1
        or generation.get("chat_template_mode") != "official_qwen_instruct_non_thinking"
        or generation.get("chain_of_thought_requested") is not False
        or generation.get("thinking_enabled") is not False
    ):
        raise QwenScaleVerificationError("deterministic generation/repair contract differs")
    baseline = _read(root / "qwen17b_negative_baseline.json")
    frozen = FrozenQwen17BBaselineV0()
    if any(baseline.get(key) != value for key, value in frozen.to_dict().items()):
        raise QwenScaleVerificationError("Qwen3-1.7B negative-baseline status changed")
    source_identity = baseline.get("source_evidence")
    if target:
        source_baseline = verify_language_development_evidence(
            qwen17b_evidence_root,
            corpus=corpus,
            classifier_checkpoint=classifier_checkpoint,
        )
        if (
            not isinstance(source_identity, Mapping)
            or source_identity.get("artifact_fingerprint")
            != source_baseline.get("artifact_fingerprint")
            or source_baseline.get("llm_validation_completed") is not True
            or source_baseline.get("llm_validation_gate_passed") is not False
            or source_baseline.get("language_development_completed") is not False
        ):
            raise QwenScaleVerificationError("Qwen3-1.7B source evidence is not frozen rejection")
    elif source_identity is not None:
        raise QwenScaleVerificationError("fixture evidence must not claim a real 1.7B source")
    access = _read(root / "access_audit.json")
    expected_access = {
        "language_final_accessed": False,
        "control_development_accessed": False,
        "control_final_accessed": False,
        "test_split_accessed": False,
        "m42_final_accessed": False,
        "act_controller_loaded": False,
        "robot_environment_created": False,
        "environment_step_count": 0,
        "smolvla_go": False,
    }
    if access != expected_access or result.get("access_audit") != access:
        raise QwenScaleVerificationError("a prohibited source or runtime was accessed")
    no_training = result.get("no_training_audit")
    if no_training != {
        "optimizer_constructed": False,
        "training_or_fine_tuning_performed": False,
        "lora_used": False,
        "model_weights_unchanged": True,
    }:
        raise QwenScaleVerificationError("no-training/model-weight audit differs")
    smoke = _read(root / "train_smoke.json")
    smoke_ids = smoke.get("example_ids")
    train_ids = {value.example_id for value in corpus.examples_for_split(LanguageSplit.TRAIN)}
    if (
        not isinstance(smoke_ids, list)
        or len(smoke_ids) != 20
        or not set(smoke_ids).issubset(train_ids)
    ):
        raise QwenScaleVerificationError("train-only smoke IDs differ")
    if target:
        smoke_checks = smoke.get("checks")
        if not isinstance(smoke_checks, Mapping) or smoke.get("gate_passed") is not all(
            value is True for value in smoke_checks.values()
        ):
            raise QwenScaleVerificationError("train smoke gate/checks differ")
    validation_completed = flags.get("qwen4b_validation_completed") is True
    validation_gate_passed = False
    if validation_completed:
        if smoke.get("gate_passed") is not True:
            raise QwenScaleVerificationError("validation bypassed failed train smoke")
        for name in (
            "rule_router",
            "classifier_negative_baseline",
            "qwen17b_negative_baseline",
            "qwen4b_candidate",
        ):
            _router_metrics(
                root=root,
                relative=f"validation/{name}.json",
                corpus=corpus,
                split=LanguageSplit.VALIDATION,
            )
        _, metrics4, ids4 = _router_metrics(
            root=root,
            relative="validation/qwen4b_candidate.json",
            corpus=corpus,
            split=LanguageSplit.VALIDATION,
        )
        _, metrics17, ids17 = _router_metrics(
            root=root,
            relative="validation/qwen17b_negative_baseline.json",
            corpus=corpus,
            split=LanguageSplit.VALIDATION,
        )
        source_root = Path(qwen17b_evidence_root)
        for target_name, source_name in (
            ("rule_router", "rule_router"),
            ("classifier_negative_baseline", "classifier_negative_baseline"),
            ("qwen17b_negative_baseline", "local_llm"),
        ):
            copied = _read(root / f"validation/{target_name}.json")
            source = _read(source_root / f"validation/{source_name}.json")
            if target_name == "qwen17b_negative_baseline":
                if (
                    copied.get("records") != source.get("records")
                    or copied.get("metrics") != source.get("metrics")
                    or copied.get("split") != source.get("split")
                    or copied.get("source_m5a2_eligibility") != source.get("eligibility")
                    or not isinstance(copied.get("eligibility"), Mapping)
                    or cast(Mapping[str, object], copied["eligibility"]).get(
                        "frozen_negative_baseline"
                    )
                    is not True
                    or cast(Mapping[str, object], copied["eligibility"]).get("promotion_eligible")
                    is not False
                ):
                    raise QwenScaleVerificationError("copied Qwen3-1.7B negative baseline differs")
            elif copied != source:
                raise QwenScaleVerificationError("copied frozen validation baseline differs")
        metrics4 = {
            **metrics4,
            "total_elapsed_inference_seconds": sum(
                float(value["latency_ms"])
                for value in cast(
                    list[Mapping[str, object]],
                    _read(root / "validation/qwen4b_candidate.json")["records"],
                )
            )
            / 1000.0,
        }
        metrics17 = {
            **metrics17,
            "total_elapsed_inference_seconds": sum(
                float(value["latency_ms"])
                for value in cast(
                    list[Mapping[str, object]],
                    _read(root / "validation/qwen17b_negative_baseline.json")["records"],
                )
            )
            / 1000.0,
        }
        gate = evaluate_qwen4b_validation_gate(metrics=metrics4, prohibited_source_access=False)
        if _read(root / "validation/gate.json") != gate.to_dict():
            raise QwenScaleVerificationError("Qwen3-4B validation gate differs")
        validation_gate_passed = gate.passed
        runtime_lock_path = root / "validation/runtime_lock.json"
        if validation_gate_passed:
            lock = _read(runtime_lock_path)
            lock_fingerprint = lock.pop("runtime_lock_fingerprint", None)
            if (
                lock.get("model_identity_fingerprint") != expected_model.fingerprint
                or lock.get("prompt_equivalence_fingerprint") != equivalence.fingerprint
                or lock.get("schema_fingerprint") != equivalence.schema_fingerprint
                or lock.get("parser_fingerprint") != equivalence.parser_fingerprint
                or lock.get("repair_policy_fingerprint") != equivalence.repair_policy_fingerprint
                or lock.get("generation_config_fingerprint")
                != owner.get("generation_config_fingerprint")
                or lock.get("locked_before_development") is not True
                or lock_fingerprint != f"sha256:{sha256_hex(lock)}"
            ):
                raise QwenScaleVerificationError("validation runtime lock differs")
        elif runtime_lock_path.exists():
            raise QwenScaleVerificationError("failed validation created a runtime lock")
        model_download = model.get("model_download_seconds")
        expected_comparison = compute_scale_comparison(
            qwen17b_metrics=metrics17,
            qwen4b_metrics=metrics4,
            validation_example_ids_17b=ids17,
            validation_example_ids_4b=ids4,
            qwen4b_gate_passed=gate.passed,
            model_download_seconds_4b=(
                float(model_download) if isinstance(model_download, int | float) else None
            ),
        )
        if _read(root / "scale_comparison.json") != expected_comparison:
            raise QwenScaleVerificationError("Qwen capacity comparison differs")
    elif (root / "validation").exists() or (root / "scale_comparison.json").exists():
        raise QwenScaleVerificationError("failed smoke opened validation")
    development_completed = flags.get("qwen4b_language_development_completed") is True
    development_gate_passed = False
    if development_completed:
        if not validation_gate_passed:
            raise QwenScaleVerificationError("development bypassed validation promotion")
        for name in (
            "rule_router",
            "classifier_negative_baseline",
            "qwen17b_negative_baseline",
            "qwen4b_candidate",
        ):
            _router_metrics(
                root=root,
                relative=f"development/{name}.json",
                corpus=corpus,
                split=LanguageSplit.DEVELOPMENT,
            )
        _, metrics, _ = _router_metrics(
            root=root,
            relative="development/qwen4b_candidate.json",
            corpus=corpus,
            split=LanguageSplit.DEVELOPMENT,
        )
        safety = _read(root / "development/safety_probes.json")
        checks = safety.get("checks")
        if not isinstance(checks, Mapping):
            raise QwenScaleVerificationError("development safety checks are malformed")
        gate = evaluate_qwen4b_development_gate(
            metrics=metrics,
            per_task_accuracy=cast(Mapping[str, object], metrics["per_task_accuracy"]),
            rejection_family_false_route_rates=cast(
                Mapping[str, object], metrics["rejection_family_false_route_rates"]
            ),
            safety_checks=cast(Mapping[str, bool], checks),
        )
        if _read(root / "development/gate.json") != gate.to_dict():
            raise QwenScaleVerificationError("Qwen3-4B development gate differs")
        development_gate_passed = gate.passed
    elif (root / "development").exists():
        raise QwenScaleVerificationError("stopped evidence contains development files")
    expected_flag_values = {
        "qwen17b_candidate_frozen": True,
        "qwen17b_offline_baseline_validated": target,
        "qwen4b_model_identity_validated": target,
        "qwen4b_prompt_equivalence_validated": target,
        "qwen4b_train_smoke_completed": target,
        "qwen4b_validation_completed": validation_completed,
        "qwen4b_validation_gate_passed": validation_gate_passed,
        "qwen4b_language_development_completed": development_completed,
        "qwen4b_language_quality_gate_passed": development_gate_passed,
        "learned_router_selected": development_gate_passed,
        "one_scene_control_smoke_authorized": development_gate_passed,
        "language_final_accessed": False,
        "control_final_accessed": False,
        "test_split_accessed": False,
        "m42_final_accessed": False,
        "smolvla_go": False,
        "real_gpu_inference_validated": target,
        "physical_target_validated": False,
        "passed": True,
    }
    if any(flags.get(key) is not value for key, value in expected_flag_values.items()):
        raise QwenScaleVerificationError("M5A.3 stage flags differ from recomputed evidence")
    stopped_at_validation = complete.get("stopped_at_validation")
    if stopped_at_validation is not (validation_completed and not validation_gate_passed):
        raise QwenScaleVerificationError("validation stopping marker differs")
    gpu_identity_valid = not target or (
        os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
        and torch.cuda.is_available()
        and torch.cuda.device_count() == 1
    )
    if not gpu_identity_valid:
        raise QwenScaleVerificationError("independent verifier does not see one locked GPU")
    return {
        "schema_version": "langmani-m5a3-independent-qwen-scale-verification-v0",
        **expected_flag_values,
        "implementation_validated": True,
        "evidence_checksums_validated": True,
        "model_files_validated": not target or model.get("actual_files") == expected_files,
        "one_visible_gpu_validated": target and gpu_identity_valid,
        "prompt_semantic_equivalence_validated": equivalence.semantic_prompt_content_equal,
        "no_training_validated": True,
        "weights_unchanged_validated": True,
        "stopped_at_validation": stopped_at_validation,
        "runtime_fingerprint": owner.get("runtime_fingerprint"),
        "artifact_fingerprint": validated["artifact_fingerprint"],
        "evidence_root": str(root),
    }


__all__ = [
    "QwenScaleVerificationError",
    "verify_qwen_scale_evidence",
]
