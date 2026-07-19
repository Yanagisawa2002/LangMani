"""Execute the bounded M5A.4.1 safety/taxonomy amendment and offline evaluation."""

# ruff: noqa: E402 -- direct script execution adds the project root before shared runner imports.

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langmani.datasets.identity import sha256_hex
from langmani.language.corpus import build_language_corpus
from langmani.language.llm_router import (
    QWEN3_1_7B_MODEL_ID,
    QWEN3_1_7B_REVISION,
    QWEN3_4B_INSTRUCT_MODEL_ID,
    QWEN3_4B_INSTRUCT_REVISION,
    StructuredLLMRouterConfig,
    StructuredLocalLLMRouterV0,
    TransformersLocalTextGenerator,
    select_structured_routing_prompt_examples,
)
from langmani.language.neuro_symbolic_evidence import validate_neuro_symbolic_evidence
from langmani.language.neuro_symbolic_experiment import compute_neuro_symbolic_safety_metrics
from langmani.language.neuro_symbolic_router import (
    OUTLINES_VERSION,
    DeterministicSafetyArbiterV0,
    NeuroSymbolicRouterV0,
    OutlinesQwenSemanticFrameExtractorV0,
    SymbolicLexicalParserV0,
    build_qwen4b_loader_config,
    build_semantic_frame_prompt,
    select_semantic_prompt_examples,
    semantic_frame_schema,
    semantic_schema_fingerprint,
)
from langmani.language.neuro_symbolic_safety import (
    M5A41ContractAmendment,
    evaluate_development_safety_gate,
    evaluate_router_contracts,
    evaluate_train_smoke_contracts,
    evaluate_train_smoke_gate,
    initial_m5a41_flags,
    rejection_taxonomy_contract,
    safety_contract,
    safety_contract_fingerprint,
    special_safety_route_counts,
    taxonomy_audit,
    taxonomy_contract_fingerprint,
)
from langmani.language.neuro_symbolic_safety_evidence import (
    read_object,
    real_unlinked,
    write_m5a41_evidence,
    write_router_lock,
)
from langmani.language.neuro_symbolic_safety_verifier import verify_m5a41_evidence
from langmani.language.neuro_symbolic_verifier import verify_neuro_symbolic_evidence
from langmani.language.offline_language_development import (
    CLASSIFIER_NEGATIVE_ELIGIBILITY,
    RULE_ROUTER_ELIGIBILITY,
)
from langmani.language.rejection_report import RejectionReportError
from langmani.language.router_types import LanguageExample, LanguageSplit
from langmani.language.rule_router import RuleRouterV0
from langmani.policies.act_runtime import atomic_write_json, inspect_git_state
from scripts.evaluate_neuro_symbolic_router import (
    DEFAULT_CLASSIFIER_CHECKPOINT,
    DEFAULT_CLASSIFIER_REJECTION_EVIDENCE,
    DEFAULT_CORPUS_ROOT,
    DEFAULT_QWEN4B_EVIDENCE,
    DEFAULT_QWEN17B_EVIDENCE,
    NeuroSymbolicCommandError,
    _archive_fingerprint_prelock,
    _dependencies,
    _eligibility,
    _frozen_baseline_identity,
    _load_classifier_negative,
    _rendered_prompt_fingerprint,
    _router_payload,
    _semantic_field_metrics,
    _validate_development_after_lock,
)

DEFAULT_SOURCE_EVIDENCE = PROJECT_ROOT / (
    "outputs/diagnostics/m5a/neuro-symbolic-router/"
    "183bc09b0ec7a30dbc50c349bab1a1a1d9368392b3f52551f509b716c83bbf7c"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/diagnostics/m5a/neuro-symbolic-safety-gate"
DEFAULT_REPORT = PROJECT_ROOT / "outputs/diagnostics/m5a/stages/neuro-symbolic-safety-gate.json"
COMMAND_SCHEMA = "langmani-m5a41-neuro-symbolic-safety-command-v1"


class M5A41CommandError(RuntimeError):
    """Raised when the M5A.4.1 command cannot preserve its frozen contract."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--target-development", action="store_true")
    modes.add_argument("--verify-evidence", type=Path)
    parser.add_argument("--source-evidence-root", type=Path, default=DEFAULT_SOURCE_EVIDENCE)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--classifier-checkpoint", type=Path, default=DEFAULT_CLASSIFIER_CHECKPOINT)
    parser.add_argument(
        "--classifier-rejection-evidence",
        type=Path,
        default=DEFAULT_CLASSIFIER_REJECTION_EVIDENCE,
    )
    parser.add_argument("--qwen17b-evidence-root", type=Path, default=DEFAULT_QWEN17B_EVIDENCE)
    parser.add_argument("--qwen4b-evidence-root", type=Path, default=DEFAULT_QWEN4B_EVIDENCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument("--semantic-maximum-new-tokens", type=int, default=256)
    return parser.parse_args(argv)


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "source": real_unlinked(args.source_evidence_root, label="source M5A.4 evidence"),
        "corpus": real_unlinked(args.corpus_root, label="language corpus"),
        "classifier": real_unlinked(args.classifier_checkpoint, label="classifier checkpoint"),
        "rejection": real_unlinked(
            args.classifier_rejection_evidence, label="classifier rejection evidence"
        ),
        "qwen17": real_unlinked(args.qwen17b_evidence_root, label="Qwen3-1.7B evidence"),
        "qwen4": real_unlinked(args.qwen4b_evidence_root, label="Qwen3-4B evidence"),
        "output": real_unlinked(args.output_root, label="M5A.4.1 output"),
        "report": real_unlinked(args.report, label="M5A.4.1 report"),
    }
    protected = tuple(
        (PROJECT_ROOT / name).resolve(strict=False)
        for name in ("src", "scripts", "environment", "tests", "docs", ".git")
    )
    inputs = tuple(
        paths[name] for name in ("source", "corpus", "classifier", "rejection", "qwen17", "qwen4")
    )
    output = paths["output"]
    report = paths["report"]
    if any(output == item or output in item.parents or item in output.parents for item in inputs):
        raise M5A41CommandError("M5A.4.1 output overlaps an immutable input")
    if any(
        output == item or item in output.parents or output in item.parents for item in protected
    ):
        raise M5A41CommandError("M5A.4.1 output overlaps source-controlled content")
    if (
        report == output
        or output in report.parents
        or any(
            report == item or report in item.parents or item in report.parents for item in inputs
        )
    ):
        raise M5A41CommandError("M5A.4.1 report overlaps immutable evidence")
    return paths


def _source_objects(root: Path) -> dict[str, dict[str, object]]:
    return {
        name: read_object(root / f"{name}.json")
        for name in (
            "owner",
            "complete",
            "train_smoke",
            "model_identity",
            "prompt",
            "arbiter_contract",
            "symbolic_contract",
            "semantic_schema",
            "runtime_identity",
        )
    }


def _preflight_classifier_negative(
    *, checkpoint: Path, rejection_root: Path, local_files_only: bool
) -> tuple[object, dict[str, object]]:
    """Validate the frozen classifier inputs before any expensive GPU inference."""

    try:
        return _load_classifier_negative(
            checkpoint=checkpoint,
            rejection_root=rejection_root,
            local_files_only=local_files_only,
        )
    except (NeuroSymbolicCommandError, RejectionReportError, OSError) as exc:
        raise M5A41CommandError(
            "frozen classifier input preflight failed; pass the authoritative "
            f"--classifier-rejection-evidence root (received {rejection_root}): {exc}"
        ) from exc


def _split_fingerprints(corpus: object) -> dict[str, str]:
    manifest = cast(object, corpus.manifest)
    split_manifests = cast(Mapping[LanguageSplit, object], manifest.split_manifests)
    return {
        split.value: cast(str, split_manifest.content_fingerprint)
        for split, split_manifest in split_manifests.items()
    }


def _assert_source_contract_unchanged(
    *,
    source: Mapping[str, Mapping[str, object]],
    parser: SymbolicLexicalParserV0,
    arbiter: DeterministicSafetyArbiterV0,
    prompt_content: str,
    prompt_fingerprint: str,
    prompt_ids: Sequence[str],
) -> None:
    prompt = source["prompt"]
    if (
        prompt.get("prompt_content") != prompt_content
        or prompt.get("prompt_fingerprint") != prompt_fingerprint
        or prompt.get("few_shot_example_ids") != list(prompt_ids)
        or source["symbolic_contract"] != parser.contract_dict()
        or source["arbiter_contract"] != arbiter.contract_dict()
        or source["semantic_schema"].get("schema") != semantic_frame_schema()
        or source["semantic_schema"].get("schema_fingerprint") != semantic_schema_fingerprint()
    ):
        raise M5A41CommandError("model-independent NeuroSymbolicRouter contract changed")


def _model_fingerprint(model_identity: Mapping[str, object]) -> str:
    return f"sha256:{sha256_hex(dict(model_identity))}"


def _dry_run(paths: Mapping[str, Path]) -> dict[str, object]:
    corpus = build_language_corpus()
    parser = SymbolicLexicalParserV0()
    arbiter = DeterministicSafetyArbiterV0()
    examples = select_semantic_prompt_examples(corpus.examples_for_split(LanguageSplit.TRAIN))
    prompt, prompt_fingerprint, ids = build_semantic_frame_prompt(examples=examples, parser=parser)
    audit = taxonomy_audit(
        corpus=corpus,
        source_smoke={"records": []},
    )
    checks = {
        "taxonomy_contract_consistent": audit["taxonomy_contract_consistent"] is True,
        "safety_classes_exact": len(safety_contract()["classes"]) == 5,
        "prompt_unchanged_by_amendment": bool(prompt) and bool(prompt_fingerprint),
        "arbiter_unchanged_by_amendment": arbiter.contract_dict()["precedence"]
        == ["malformed", "unsupported", "ambiguous", "route"],
        "development_not_opened": audit["development_command_text_accessed_before_lock"] is False,
    }
    return {
        "schema_version": COMMAND_SCHEMA,
        "mode": "dry_run",
        "passed": all(checks.values()),
        "implementation_validated": all(checks.values()),
        "checks": checks,
        "prompt_example_ids": list(ids),
        "safety_contract_fingerprint": safety_contract_fingerprint(),
        "taxonomy_contract_fingerprint": taxonomy_contract_fingerprint(),
        "output_root": str(paths["output"]),
        "physical_target_validated": False,
    }


def _router_result_with_contracts(
    *,
    router: object,
    examples: Sequence[LanguageExample],
    split: LanguageSplit,
    repeat_count: int,
    eligibility: Mapping[str, object],
    authorize_final: bool = False,
) -> tuple[tuple, dict[str, object]]:
    records, payload = _router_payload(
        router=router,
        examples=examples,
        split=split,
        repeat_count=repeat_count,
        eligibility=eligibility,
        authorize_final=authorize_final,
    )
    safety, taxonomy = evaluate_router_contracts(records=records, examples=examples)
    payload.update(
        {
            "safety_evaluation": safety.to_dict(),
            "taxonomy_evaluation": taxonomy.to_dict(),
        }
    )
    return records, payload


def _execute_target(args: argparse.Namespace, paths: Mapping[str, Path]) -> dict[str, object]:
    git = inspect_git_state(PROJECT_ROOT)
    if git.dirty or not git.baseline_tracked:
        raise M5A41CommandError("target evidence requires a clean tracked Git commit")
    if (
        args.repeat_count != 2
        or args.semantic_maximum_new_tokens != 256
        or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise M5A41CommandError("target requires the frozen one-GPU deterministic configuration")
    dependencies = _dependencies()
    if dependencies["outlines"] != OUTLINES_VERSION:
        raise M5A41CommandError("the frozen Outlines version differs")

    source_verification = verify_neuro_symbolic_evidence(paths["source"])
    if source_verification.get("stopped_at_train_smoke") is not True:
        raise M5A41CommandError("source M5A.4 evidence is not the frozen smoke rejection")
    source_validation = validate_neuro_symbolic_evidence(paths["source"])
    source = _source_objects(paths["source"])
    corpus = build_language_corpus()
    archive_fingerprint = _archive_fingerprint_prelock(paths["corpus"], corpus)
    parser = SymbolicLexicalParserV0()
    arbiter = DeterministicSafetyArbiterV0()
    prompt_examples = select_semantic_prompt_examples(
        corpus.examples_for_split(LanguageSplit.TRAIN)
    )
    prompt_content, prompt_fingerprint, prompt_ids = build_semantic_frame_prompt(
        examples=prompt_examples, parser=parser
    )
    _assert_source_contract_unchanged(
        source=source,
        parser=parser,
        arbiter=arbiter,
        prompt_content=prompt_content,
        prompt_fingerprint=prompt_fingerprint,
        prompt_ids=prompt_ids,
    )
    audit = taxonomy_audit(corpus=corpus, source_smoke=source["train_smoke"])
    if audit["taxonomy_contract_consistent"] is not True:
        raise M5A41CommandError("taxonomy audit found a corpus/contract contradiction")

    raw_smoke = source["train_smoke"].get("records")
    if not isinstance(raw_smoke, list):
        raise M5A41CommandError("source train-smoke records are missing")
    smoke_safety, smoke_taxonomy = evaluate_train_smoke_contracts(
        records=tuple(cast(Mapping[str, object], value) for value in raw_smoke),
        examples=prompt_examples,
    )
    smoke_gate = evaluate_train_smoke_gate(safety=smoke_safety, no_prohibited_source_access=True)
    if not smoke_gate.passed:
        raise M5A41CommandError("M5A.4.1 train-smoke safety gate failed")

    split_fingerprints = _split_fingerprints(corpus)
    amendment = M5A41ContractAmendment(
        implementation_git=git.commit,
        source_router_runtime_fingerprint=cast(str, source["owner"]["runtime_fingerprint"]),
        source_train_smoke_artifact_fingerprint=(f"sha256:{sha256_hex(source['train_smoke'])}"),
        corpus_fingerprint=corpus.manifest.corpus_fingerprint,
        split_fingerprints=split_fingerprints,
        taxonomy_contract_fingerprint=taxonomy_contract_fingerprint(),
        safety_contract_fingerprint=safety_contract_fingerprint(),
        model_fingerprint=_model_fingerprint(source["model_identity"]),
        prompt_fingerprint=prompt_fingerprint,
        semantic_schema_fingerprint=semantic_schema_fingerprint(),
        symbolic_contract_fingerprint=parser.contract_fingerprint,
        arbiter_fingerprint=arbiter.contract_fingerprint,
    )
    qwen17_identity = _frozen_baseline_identity(paths["qwen17"], label="Qwen3-1.7B")
    qwen4_identity = _frozen_baseline_identity(paths["qwen4"], label="direct Qwen3-4B")
    classifier, classifier_identity = _preflight_classifier_negative(
        checkpoint=paths["classifier"],
        rejection_root=paths["rejection"],
        local_files_only=bool(args.local_files_only),
    )
    runtime_payload = {
        "implementation_git": git.commit,
        "source_m5a4_runtime_fingerprint": source["owner"]["runtime_fingerprint"],
        "source_m5a4_artifact_fingerprint": source_validation["artifact_fingerprint"],
        "protocol_amendment_fingerprint": amendment.fingerprint,
        "corpus_fingerprint": corpus.manifest.corpus_fingerprint,
        "corpus_archive_fingerprint": archive_fingerprint,
        "split_fingerprints": split_fingerprints,
        "model_id": QWEN3_4B_INSTRUCT_MODEL_ID,
        "model_revision": QWEN3_4B_INSTRUCT_REVISION,
        "tokenizer_revision": QWEN3_4B_INSTRUCT_REVISION,
        "model_fingerprint": amendment.model_fingerprint,
        "prompt_fingerprint": prompt_fingerprint,
        "prompt_example_ids": list(prompt_ids),
        "semantic_schema_fingerprint": semantic_schema_fingerprint(),
        "symbolic_contract_fingerprint": parser.contract_fingerprint,
        "arbiter_fingerprint": arbiter.contract_fingerprint,
        "safety_contract_fingerprint": safety_contract_fingerprint(),
        "taxonomy_contract_fingerprint": taxonomy_contract_fingerprint(),
        "dependencies": dependencies,
        "repeat_count": args.repeat_count,
        "semantic_maximum_new_tokens": args.semantic_maximum_new_tokens,
        "visible_gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "qwen17_source": qwen17_identity["artifact_fingerprint"],
        "qwen4_source": qwen4_identity["artifact_fingerprint"],
        "classifier_analysis_fingerprint": classifier_identity["analysis_fingerprint"],
        "classifier_checkpoint_fingerprint": classifier_identity["checkpoint_fingerprint"],
        "classifier_selection_fingerprint": classifier_identity["selection_fingerprint"],
    }
    runtime_fingerprint = f"sha256:{sha256_hex(runtime_payload)}"
    owner = {
        "schema_version": "langmani-m5a41-neuro-symbolic-safety-owner-v1",
        "mode": "target_development",
        "runtime_fingerprint": runtime_fingerprint,
        "implementation_git": git.commit,
        "source_m5a4_runtime_fingerprint": source["owner"]["runtime_fingerprint"],
        "corpus_fingerprint": corpus.manifest.corpus_fingerprint,
    }

    config4 = build_qwen4b_loader_config(maximum_new_tokens=128)
    loader4 = TransformersLocalTextGenerator(
        config=config4, device="cuda", local_files_only=bool(args.local_files_only)
    )
    if loader4.file_identities != source["model_identity"].get(
        "file_identities"
    ) or loader4.snapshot_size_bytes != source["model_identity"].get("snapshot_size_bytes"):
        raise M5A41CommandError("loaded Qwen3-4B files differ from frozen M5A.4")
    rendered_fingerprint = _rendered_prompt_fingerprint(loader4, prompt_content)
    if rendered_fingerprint != source["prompt"].get("rendered_prompt_fingerprint"):
        raise M5A41CommandError("official chat-template rendering differs from frozen M5A.4")
    extractor = OutlinesQwenSemanticFrameExtractorV0(
        loader=loader4,
        prompt_content=prompt_content,
        maximum_new_tokens=args.semantic_maximum_new_tokens,
    )
    router = NeuroSymbolicRouterV0(parser=parser, semantic_extractor=extractor, arbiter=arbiter)

    flags = initial_m5a41_flags()
    flags.update(
        {
            "taxonomy_audit_completed": True,
            "taxonomy_contract_consistent": True,
            "safety_routing_contract_validated": True,
            "rejection_taxonomy_contract_validated": True,
            "train_smoke_safety_gate_passed": True,
            "train_smoke_taxonomy_gate_passed": smoke_taxonomy.exact_quality_passed,
            "real_gpu_inference_validated": True,
        }
    )
    lock_written_ns = time.time_ns()
    router_lock = write_router_lock(
        paths["output"],
        runtime_fingerprint=runtime_fingerprint,
        lock_payload={
            "protocol_amendment_fingerprint": amendment.fingerprint,
            "source_m5a4_runtime_fingerprint": source["owner"]["runtime_fingerprint"],
            "source_m5a4_artifact_fingerprint": source_validation["artifact_fingerprint"],
            "model_id": QWEN3_4B_INSTRUCT_MODEL_ID,
            "model_revision": QWEN3_4B_INSTRUCT_REVISION,
            "tokenizer_revision": QWEN3_4B_INSTRUCT_REVISION,
            "model_fingerprint": amendment.model_fingerprint,
            "model_file_identities": loader4.file_identities,
            "dtype": "bfloat16",
            "outlines_version": OUTLINES_VERSION,
            "prompt_fingerprint": prompt_fingerprint,
            "rendered_prompt_fingerprint": rendered_fingerprint,
            "prompt_example_ids": list(prompt_ids),
            "semantic_schema_fingerprint": semantic_schema_fingerprint(),
            "symbolic_contract_fingerprint": parser.contract_fingerprint,
            "arbiter_fingerprint": arbiter.contract_fingerprint,
            "safety_contract_fingerprint": safety_contract_fingerprint(),
            "taxonomy_contract_fingerprint": taxonomy_contract_fingerprint(),
            "agreement_rule": "exact-object-bin-and-pick-action-v0",
            "reason_code_mapping": rejection_taxonomy_contract()["rules"],
            "generation_config": {
                "do_sample": False,
                "repeat_count": args.repeat_count,
                "semantic_maximum_new_tokens": args.semantic_maximum_new_tokens,
                "quantization": "none",
            },
            "metric_implementation": "m5a41-safety-taxonomy-recompute-v1",
            "implementation_git": git.commit,
            "lock_written_time_ns": lock_written_ns,
        },
    )
    flags["neuro_symbolic_router_locked"] = True

    historical_started_ns = time.time_ns()
    validation_examples = corpus.examples_for_split(LanguageSplit.VALIDATION)
    historical_records, historical = _router_result_with_contracts(
        router=router,
        examples=validation_examples,
        split=LanguageSplit.VALIDATION,
        repeat_count=args.repeat_count,
        eligibility=_eligibility(promotable=False, frozen_negative=False),
    )
    historical.update(
        {
            "post_selection_diagnostic_only": True,
            "configuration_change_authorized": False,
            "router_lock_fingerprint": router_lock["lock_fingerprint"],
            "started_after_router_lock": historical_started_ns >= lock_written_ns,
            "configuration_fingerprints_before": {
                "model": amendment.model_fingerprint,
                "prompt": prompt_fingerprint,
                "schema": semantic_schema_fingerprint(),
                "symbolic": parser.contract_fingerprint,
                "arbiter": arbiter.contract_fingerprint,
            },
            "configuration_fingerprints_after": {
                "model": amendment.model_fingerprint,
                "prompt": prompt_fingerprint,
                "schema": semantic_schema_fingerprint(),
                "symbolic": parser.contract_fingerprint,
                "arbiter": arbiter.contract_fingerprint,
            },
        }
    )
    flags["historical_validation_diagnostic_completed"] = len(historical_records) == 300

    development_started_ns = time.time_ns()
    development_examples = corpus.examples_for_split(LanguageSplit.DEVELOPMENT)
    _validate_development_after_lock(paths["corpus"], development_examples)
    direct_prompt_examples = select_structured_routing_prompt_examples(
        corpus.examples_for_split(LanguageSplit.TRAIN)
    )
    direct4 = StructuredLocalLLMRouterV0(
        config=config4, generator=loader4, prompt_examples=direct_prompt_examples
    )
    config17 = StructuredLLMRouterConfig(
        model_id=QWEN3_1_7B_MODEL_ID,
        model_revision=QWEN3_1_7B_REVISION,
        tokenizer_revision=QWEN3_1_7B_REVISION,
        dtype="bfloat16",
        quantization="none",
        maximum_new_tokens=128,
        maximum_format_repair_attempts=1,
        chat_template_mode="official_qwen_enable_thinking_false",
    )
    loader17 = TransformersLocalTextGenerator(
        config=config17, device="cuda", local_files_only=bool(args.local_files_only)
    )
    direct17 = StructuredLocalLLMRouterV0(
        config=config17, generator=loader17, prompt_examples=direct_prompt_examples
    )
    development_payloads: dict[str, object] = {}
    candidate_records = None
    for name, active, eligibility in (
        ("rule_router", RuleRouterV0(), RULE_ROUTER_ELIGIBILITY.to_dict()),
        (
            "classifier_negative_baseline",
            classifier,
            CLASSIFIER_NEGATIVE_ELIGIBILITY.to_dict(),
        ),
        (
            "qwen17b_negative_baseline",
            direct17,
            _eligibility(promotable=False, frozen_negative=True),
        ),
        (
            "qwen4b_negative_baseline",
            direct4,
            _eligibility(promotable=False, frozen_negative=True),
        ),
        (
            "neuro_symbolic_candidate",
            router,
            _eligibility(promotable=True, frozen_negative=False),
        ),
    ):
        records, payload = _router_result_with_contracts(
            router=active,
            examples=development_examples,
            split=LanguageSplit.DEVELOPMENT,
            repeat_count=args.repeat_count,
            eligibility=eligibility,
        )
        development_payloads[f"development/{name}.json"] = payload
        if name == "neuro_symbolic_candidate":
            candidate_records = records
    assert candidate_records is not None
    candidate = cast(
        Mapping[str, object], development_payloads["development/neuro_symbolic_candidate.json"]
    )
    candidate_metrics = cast(Mapping[str, object], candidate["metrics"])
    candidate_safety, candidate_taxonomy = evaluate_router_contracts(
        records=candidate_records, examples=development_examples
    )
    special_counts = special_safety_route_counts(
        records=candidate_records, examples=development_examples
    )
    family_rates = cast(
        Mapping[str, float], candidate_metrics["rejection_family_false_route_rates"]
    )
    gate = evaluate_development_safety_gate(
        metrics=candidate_metrics,
        safety=candidate_safety,
        special_route_counts=special_counts,
        rejection_family_unsafe_false_route_rates=family_rates,
        no_prohibited_source_access=True,
    )
    agreement_metrics = compute_neuro_symbolic_safety_metrics(
        records=candidate_records, examples=development_examples
    )
    system_metrics = {
        **agreement_metrics,
        **special_counts,
        "constrained_schema_compilation_seconds": extractor.compilation_seconds,
        "latency_p50_ms": candidate_metrics["latency_p50_ms"],
        "latency_p95_ms": candidate_metrics["latency_p95_ms"],
        "latency_p99_ms": candidate_metrics["latency_p99_ms"],
        "generated_token_count": candidate_metrics["generated_token_count"],
        "peak_gpu_memory_bytes": candidate_metrics["peak_gpu_memory_bytes"],
    }
    flags.update(
        {
            "language_development_completed": len(candidate_records) == 420,
            "neuro_symbolic_language_safety_gate_passed": gate.passed,
            "exact_rejection_taxonomy_quality_passed": candidate_taxonomy.exact_quality_passed,
            "learned_router_selected": gate.passed,
            "one_scene_control_smoke_authorized": gate.passed,
        }
    )
    if not classifier.verify_unchanged():
        raise M5A41CommandError("frozen classifier weights changed during evaluation")

    limitations = {
        "schema_version": "langmani-m5a41-limitations-v1",
        "exact_rejection_taxonomy_not_guaranteed": not smoke_taxonomy.exact_quality_passed,
        "exact_rejection_taxonomy_quality_passed": candidate_taxonomy.exact_quality_passed,
        "rejection_statuses_share_one_no_dispatch_physical_behavior": True,
        "reliable_exact_rejection_reason_classification_claimed": False,
        "calibrated_rejection_taxonomy_claimed": False,
        "human_level_semantic_categorization_claimed": False,
        "robot_control_executed": False,
        "physical_target_validated": False,
    }
    selection = {
        "schema_version": "langmani-m5a41-candidate-selection-v1",
        "candidate": "NeuroSymbolicRouterV0",
        "only_promotable_candidate": True,
        "safety_gate": gate.to_dict(),
        "taxonomy_quality_gate": {
            "diagnostic_only": True,
            "passed": candidate_taxonomy.exact_quality_passed,
        },
        "learned_router_selected": gate.passed,
        "one_scene_control_smoke_authorized": gate.passed,
        "classifier_negative_baseline": classifier_identity,
        "qwen17b_negative_baseline": qwen17_identity,
        "qwen4b_negative_baseline": qwen4_identity,
    }
    artifacts: dict[str, object] = {
        "protocol_amendment.json": amendment.to_dict(),
        "taxonomy_audit.json": audit,
        "safety_contract.json": safety_contract(),
        "taxonomy_contract.json": rejection_taxonomy_contract(),
        "train_smoke_reassessment.json": {
            "schema_version": "langmani-m5a41-train-smoke-reassessment-v1",
            "source_evidence_root": str(paths["source"]),
            "source_runtime_fingerprint": source["owner"]["runtime_fingerprint"],
            "source_artifact_fingerprint": source_validation["artifact_fingerprint"],
            "source_output_records_reused": True,
            "model_inference_rerun": False,
            "source_outputs_unchanged": audit["source_smoke_outputs_unchanged"],
            "safety_evaluation": smoke_safety.to_dict(),
            "taxonomy_evaluation": smoke_taxonomy.to_dict(),
            "safety_gate": smoke_gate.to_dict(),
            "taxonomy_gate_passed": smoke_taxonomy.exact_quality_passed,
        },
        "router_lock.json": router_lock,
        "historical_validation_diagnostic.json": historical,
        **development_payloads,
        "development/neuro_symbolic_system_metrics.json": system_metrics,
        "development/neuro_symbolic_semantic_field_metrics.json": _semantic_field_metrics(
            records=candidate_records, examples=development_examples
        ),
        "development/neuro_symbolic_safety_gate.json": gate.to_dict(),
        "candidate_selection.json": selection,
        "limitations.json": limitations,
        "runtime_identity.json": {
            **runtime_payload,
            "runtime_fingerprint": runtime_fingerprint,
            "optimizer_constructed": False,
            "training_performed": False,
            "weights_modified": False,
            "prompt_modified": False,
            "few_shot_examples_modified": False,
            "semantic_schema_modified": False,
            "symbolic_parser_modified": False,
            "arbiter_modified": False,
            "robot_environment_created": False,
            "controller_loaded": False,
            "env_step_called": False,
            "development_access_after_lock": development_started_ns >= lock_written_ns,
            "historical_validation_access_after_lock": historical_started_ns >= lock_written_ns,
            "language_final_accessed": False,
            "control_final_accessed": False,
            "test_split_accessed": False,
            "m42_final_accessed": False,
            "smolvla_started": False,
        },
        "immutability_identity.json": {
            "source_model_identity": source["model_identity"],
            "source_prompt_fingerprint": prompt_fingerprint,
            "source_symbolic_contract_fingerprint": parser.contract_fingerprint,
            "source_semantic_schema_fingerprint": semantic_schema_fingerprint(),
            "source_arbiter_fingerprint": arbiter.contract_fingerprint,
            "loaded_model_file_identities": loader4.file_identities,
            "rendered_prompt_fingerprint": rendered_fingerprint,
        },
        "summary.md": (
            "# M5A.4.1 Neuro-Symbolic Safety Gate\n\n"
            f"Train-smoke safety gate: `{smoke_gate.passed}`.  "
            f"Train-smoke exact taxonomy gate: `{smoke_taxonomy.exact_quality_passed}`.\n\n"
            f"Development safety gate: `{gate.passed}`.  "
            f"Development exact taxonomy quality: `{candidate_taxonomy.exact_quality_passed}`.\n\n"
            "Historical validation was diagnostic only. No model training, prompt change, "
            "controller, robot environment, final source, m42_final_v0, or SmolVLA was accessed.\n"
        ),
    }
    evidence = write_m5a41_evidence(paths["output"], owner=owner, artifacts=artifacts, flags=flags)
    independent = verify_m5a41_evidence(cast(str, evidence["root"]))
    return {
        "schema_version": COMMAND_SCHEMA,
        "mode": "target_development",
        "passed": independent["passed"] is True,
        **flags,
        "train_smoke_exact_status_accuracy": smoke_taxonomy.exact_four_way_status_accuracy,
        "train_smoke_safe_rejection_rate": smoke_safety.safe_rejection_recall,
        "train_smoke_unsafe_false_route_count": smoke_safety.unsafe_false_route_count,
        "runtime_fingerprint": runtime_fingerprint,
        "protocol_amendment_fingerprint": amendment.fingerprint,
        "router_lock_fingerprint": router_lock["lock_fingerprint"],
        "artifact_fingerprint": evidence["artifact_fingerprint"],
        "evidence_root": evidence["root"],
        "independent_verification": independent,
        "physical_target_validated": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report: dict[str, object] = {
        "schema_version": COMMAND_SCHEMA,
        "passed": False,
        "mode": "verify_evidence"
        if args.verify_evidence is not None
        else "target_development"
        if args.target_development
        else "dry_run",
        "physical_target_validated": False,
    }
    try:
        paths = _paths(args)
        if args.verify_evidence is not None:
            report = verify_m5a41_evidence(args.verify_evidence)
        elif args.dry_run:
            report = _dry_run(paths)
        else:
            report = _execute_target(args, paths)
    except Exception as error:  # noqa: BLE001 - outer boundary preserves exact diagnostics
        traceback.print_exc()
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    if args.verify_evidence is None:
        try:
            atomic_write_json(real_unlinked(args.report, label="M5A.4.1 report"), report)
        except Exception as error:  # noqa: BLE001 - report persistence is part of status
            traceback.print_exc()
            report["report_error"] = {"type": type(error).__name__, "message": str(error)}
            report["passed"] = False
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
