"""Execute the one authorized M5A sealed language-and-control final benchmark."""

# ruff: noqa: E402 -- direct execution adds the project root before shared script imports.

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from langmani.datasets.identity import sha256_hex
from langmani.language.controller_registry import ControllerRegistry
from langmani.language.corpus import (
    GeneratedLanguageCorpus,
    build_language_corpus,
    materialize_final_language_examples,
    materialize_final_language_slot_map,
)
from langmani.language.dispatcher import ControllerDispatcher, StrictPerTaskControllerLoader
from langmani.language.full_control_evidence import validate_full_control_evidence
from langmani.language.llm_router import (
    QWEN3_1_7B_MODEL_ID,
    QWEN3_1_7B_REVISION,
    StructuredLLMRouterConfig,
    StructuredLocalLLMRouterV0,
    TransformersLocalTextGenerator,
    select_structured_routing_prompt_examples,
)
from langmani.language.neuro_symbolic_dispatch import (
    NEURO_SYMBOLIC_DISPATCH_ROUTER_LABEL,
    SelectedNeuroSymbolicDispatchSource,
    bind_selected_router_to_controller_registry,
    validate_selected_neuro_symbolic_dispatch_source,
)
from langmani.language.neuro_symbolic_experiment import compute_neuro_symbolic_safety_metrics
from langmani.language.neuro_symbolic_router import (
    DeterministicSafetyArbiterV0,
    NeuroSymbolicRouterV0,
    OutlinesQwenSemanticFrameExtractorV0,
    SymbolicLexicalParserV0,
    build_qwen4b_loader_config,
    build_semantic_frame_prompt,
    select_semantic_prompt_examples,
    semantic_frame_schema,
    semantic_frame_transport_fingerprint,
    semantic_schema_fingerprint,
)
from langmani.language.neuro_symbolic_safety import (
    evaluate_router_contracts,
    special_safety_route_counts,
)
from langmani.language.offline_language_development import (
    CLASSIFIER_NEGATIVE_ELIGIBILITY,
    RULE_ROUTER_ELIGIBILITY,
)
from langmani.language.offline_router_metrics import recompute_router_metrics
from langmani.language.router_types import LanguageExample, LanguageSplit
from langmani.language.rule_router import RuleRouterV0
from langmani.language.schedules import (
    build_language_schedule_locks,
    build_staged_control_schedule,
)
from langmani.language.sealed_final import (
    FINAL_REJECTION_CASES,
    SEALED_FINAL_EVIDENCE_SCHEMA,
    M5AFinalRunIdentityV0,
    analyze_final_control_records,
    build_final_rejection_probe_set,
    build_final_result,
    evaluate_final_language_quality,
    final_metric_contract,
    validate_final_authorization,
)
from langmani.language.sealed_final_evidence import (
    close_final_attempt,
    final_access_count,
    open_final_attempt,
    read_object,
    real_unlinked,
    write_sealed_final_evidence,
)
from langmani.language.stage_protocol import M5AStage
from langmani.policies.act_runtime import atomic_write_json, inspect_git_state
from scripts.evaluate_neuro_symbolic_router import (
    DEFAULT_CLASSIFIER_CHECKPOINT,
    DEFAULT_CLASSIFIER_REJECTION_EVIDENCE,
    DEFAULT_CORPUS_ROOT,
    DEFAULT_QWEN4B_EVIDENCE,
    DEFAULT_QWEN17B_EVIDENCE,
    _archive_fingerprint_prelock,
    _eligibility,
    _frozen_baseline_identity,
    _load_classifier_negative,
    _rendered_prompt_fingerprint,
    _semantic_field_metrics,
)
from scripts.evaluate_neuro_symbolic_safety_gate import _router_result_with_contracts
from scripts.run_language_control import (
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_RUNTIME_SELECTION,
    DevelopmentControlInputs,
    _create_environment,
    _load_frozen_controller_registry,
    _oracle_provider,
    _provider,
    _read_object,
    load_authoritative_final_schedule,
    load_sealed_final_authority,
    run_development_control,
    run_rejection_noop_probe,
)

COMMAND_SCHEMA = "langmani-m5a-sealed-final-command-v0"
EXPECTED_FINAL_AUTHORIZATION_FINGERPRINT = (
    "sha256:afb4d59a3549d42ac6e0e2cb010e8261cb39a47792af583fe5dd263645dd19b5"
)
EXPECTED_ROUTER_LOCK_FINGERPRINT = (
    "sha256:4142a3a9e61df5a0d2299ed582131907f401b67c88162888a26cac1fa675d9e7"
)
EXPECTED_CONTROLLER_REGISTRY_FINGERPRINT = (
    "sha256:01c52fbcc24b17e9989bfafb041937586dcdc5550f9cc98624e944a1b403bd88"
)
EXPECTED_RUNTIME_FINGERPRINT = (
    "sha256:fd090c3273ad209be47f8c295370da357c0ecfecc9c26b41dea4284f93321434"
)
EXPECTED_LANGUAGE_FINAL_SCHEDULE_FINGERPRINT = (
    "sha256:045005ba544cfde9bbef3064311c15ad1a2964c14270c0e5549e167c48ae19cc"
)
EXPECTED_CONTROL_FINAL_SCHEDULE_FINGERPRINT = (
    "sha256:e3e163616ec64ca853ebc0201d45561c53fdb31221513dd0ed13dfab5b19cb23"
)
DEFAULT_SCHEDULE = PROJECT_ROOT / "outputs/diagnostics/m5a/target-development-schedules.json"
DEFAULT_NEURO_EVIDENCE = PROJECT_ROOT / (
    "outputs/diagnostics/m5a/neuro-symbolic-safety-gate/"
    "3d4129571fef3b459d2cdc718d99052ff98a8c011cd4d12ea7ba36ce69e9e058"
)
DEFAULT_FULL_EVIDENCE = PROJECT_ROOT / (
    "outputs/diagnostics/m5a/control/neuro-symbolic-full-f8c583a/"
    "full-control-development/"
    "1e01efc61d3dbe45b43287b534745b0f78f5acf2345597c06ed278d4fcb9fd19"
)
DEFAULT_FULL_VERIFICATION = PROJECT_ROOT / (
    "outputs/diagnostics/m5a/stages/"
    "neuro-symbolic-full-control-independent-verification-f8c583a.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/diagnostics/m5a/sealed-final"
DEFAULT_REPORT = PROJECT_ROOT / "outputs/diagnostics/m5a/stages/sealed-final.json"


class SealedFinalCommandError(RuntimeError):
    """Raised when the single final attempt cannot preserve its frozen contract."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--target-final", action="store_true")
    parser.add_argument("--control-schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--neuro-symbolic-evidence-root", type=Path, default=DEFAULT_NEURO_EVIDENCE)
    parser.add_argument(
        "--full-development-evidence-root", type=Path, default=DEFAULT_FULL_EVIDENCE
    )
    parser.add_argument(
        "--full-development-verification", type=Path, default=DEFAULT_FULL_VERIFICATION
    )
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--classifier-checkpoint", type=Path, default=DEFAULT_CLASSIFIER_CHECKPOINT)
    parser.add_argument(
        "--classifier-rejection-evidence", type=Path, default=DEFAULT_CLASSIFIER_REJECTION_EVIDENCE
    )
    parser.add_argument("--qwen17b-evidence-root", type=Path, default=DEFAULT_QWEN17B_EVIDENCE)
    parser.add_argument("--qwen4b-evidence-root", type=Path, default=DEFAULT_QWEN4B_EVIDENCE)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--runtime-selection", type=Path, default=DEFAULT_RUNTIME_SELECTION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument("--semantic-maximum-new-tokens", type=int, default=256)
    return parser.parse_args(argv)


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "schedule": real_unlinked(args.control_schedule, label="final schedule authority"),
        "neuro": real_unlinked(args.neuro_symbolic_evidence_root, label="router evidence"),
        "full": real_unlinked(args.full_development_evidence_root, label="full development"),
        "full_verify": real_unlinked(
            args.full_development_verification, label="full-development verification"
        ),
        "corpus": real_unlinked(args.corpus_root, label="language corpus"),
        "classifier": real_unlinked(args.classifier_checkpoint, label="classifier checkpoint"),
        "rejection": real_unlinked(
            args.classifier_rejection_evidence, label="classifier rejection evidence"
        ),
        "qwen17": real_unlinked(args.qwen17b_evidence_root, label="Qwen3-1.7B evidence"),
        "qwen4": real_unlinked(args.qwen4b_evidence_root, label="Qwen3-4B evidence"),
        "dataset": real_unlinked(args.dataset_root, label="M3B dataset"),
        "checkpoints": real_unlinked(args.checkpoint_root, label="ACT checkpoints"),
        "runtime": real_unlinked(args.runtime_selection, label="action runtime selection"),
        "output": real_unlinked(args.output_root, label="sealed-final output"),
        "report": real_unlinked(args.report, label="sealed-final report"),
    }
    immutable = tuple(
        paths[key]
        for key in (
            "schedule",
            "neuro",
            "full",
            "full_verify",
            "corpus",
            "classifier",
            "rejection",
            "qwen17",
            "qwen4",
            "dataset",
            "checkpoints",
            "runtime",
        )
    )
    output, report = paths["output"], paths["report"]
    protected = tuple(
        (PROJECT_ROOT / value).resolve(strict=False)
        for value in ("src", "scripts", "environment", "tests", "docs", ".git")
    )
    if any(
        output == item or output in item.parents or item in output.parents for item in immutable
    ):
        raise SealedFinalCommandError("sealed-final output overlaps an immutable input")
    if any(
        output == item or output in item.parents or item in output.parents for item in protected
    ):
        raise SealedFinalCommandError("sealed-final output overlaps source-controlled content")
    if (
        report == output
        or output in report.parents
        or any(
            report == item or report in item.parents or item in report.parents for item in immutable
        )
    ):
        raise SealedFinalCommandError("sealed-final report overlaps immutable evidence")
    return paths


def _sha(value: object) -> str:
    return f"sha256:{sha256_hex(value)}"


def _baseline_identity_fingerprint(source: SelectedNeuroSymbolicDispatchSource) -> str:
    return _sha(
        {
            "model_id": source.model_id,
            "model_revision": source.model_revision,
            "tokenizer_revision": source.tokenizer_revision,
            "model_fingerprint": source.model_fingerprint,
            "model_file_identities": dict(source.model_file_identities),
            "snapshot_size_bytes": source.snapshot_size_bytes,
            "dtype": source.dtype,
        }
    )


def _validate_parent_authority(
    *,
    paths: Mapping[str, Path],
    corpus: GeneratedLanguageCorpus,
    registry: ControllerRegistry,
    source: SelectedNeuroSymbolicDispatchSource,
    final_language_fingerprint: str,
    final_control_fingerprint: str,
) -> tuple[dict[str, object], dict[str, object]]:
    archive = validate_full_control_evidence(paths["full"])
    root = Path(cast(str, archive["root"]))
    verification = read_object(paths["full_verify"], label="full-development verification")
    if not (
        verification.get("passed") is True
        and verification.get("physical_target_validated") is True
        and verification.get("final_benchmark_authorized") is True
        and verification.get("final_authorization_fingerprint")
        == EXPECTED_FINAL_AUTHORIZATION_FINGERPRINT
        and verification.get("artifact_fingerprint") == archive["artifact_fingerprint"]
    ):
        raise SealedFinalCommandError("full-development independent authority differs")
    router = read_object(root / "router_identity.json", label="full router identity")
    controller = read_object(root / "controller_registry.json", label="full controller registry")
    runtime = read_object(root / "runtime_identity.json", label="full runtime identity")
    if (
        router.get("router_lock_fingerprint") != source.router_lock_fingerprint
        or controller != registry.to_dict()
        or runtime.get("runtime_contract_fingerprint") != EXPECTED_RUNTIME_FINGERPRINT
    ):
        raise SealedFinalCommandError("full-development router/controller/runtime identity differs")
    authorization = validate_final_authorization(
        read_object(root / "final_authorization.json", label="final authorization"),
        expected_fingerprint=EXPECTED_FINAL_AUTHORIZATION_FINGERPRINT,
        router_fingerprint=source.router_lock_fingerprint,
        controller_registry_fingerprint=registry.registry_fingerprint,
        runtime_fingerprint=cast(str, runtime["runtime_contract_fingerprint"]),
        language_schedule_fingerprint=final_language_fingerprint,
        control_schedule_fingerprint=final_control_fingerprint,
    )
    return authorization, runtime


def _dry_run(args: argparse.Namespace, paths: Mapping[str, Path]) -> dict[str, object]:
    corpus = build_language_corpus()
    language_development, language_final = build_language_schedule_locks(corpus)
    authoritative_bundle_checked = paths["schedule"].is_file()
    final_control = (
        load_authoritative_final_schedule(paths["schedule"], corpus=corpus)
        if authoritative_bundle_checked
        else None
    )
    authority = (
        load_sealed_final_authority(paths["schedule"], corpus=corpus)
        if authoritative_bundle_checked
        else None
    )
    checks = {
        "language_final_remains_sealed": language_final.sealed,
        "control_final_remains_sealed": (
            final_control.sealed if final_control is not None else True
        ),
        "final_examples_not_materialized": True,
        "final_episodes_not_materialized": True,
        "language_development_unchanged": not language_development.sealed,
        "authorization_fingerprint_declared": (
            EXPECTED_FINAL_AUTHORIZATION_FINGERPRINT.startswith("sha256:")
        ),
        "final_language_lock_matches": (
            authority["final_language_schedule_fingerprint"]
            == EXPECTED_LANGUAGE_FINAL_SCHEDULE_FINGERPRINT
            if authority is not None
            else language_final.schedule_fingerprint == EXPECTED_LANGUAGE_FINAL_SCHEDULE_FINGERPRINT
        ),
        "final_control_lock_matches": (
            authority["final_control_schedule_fingerprint"]
            == EXPECTED_CONTROL_FINAL_SCHEDULE_FINGERPRINT
            if authority is not None
            else EXPECTED_CONTROL_FINAL_SCHEDULE_FINGERPRINT.startswith("sha256:")
            and len(EXPECTED_CONTROL_FINAL_SCHEDULE_FINGERPRINT) == 71
        ),
        "access_count_zero": final_access_count(paths["output"]) == 0,
    }
    return {
        "schema_version": COMMAND_SCHEMA,
        "mode": "dry_run",
        "passed": all(checks.values()),
        "checks": checks,
        "authoritative_schedule_bundle_checked": authoritative_bundle_checked,
        "final_example_count": len(language_final.ordered_example_ids),
        "final_scene_count": (
            min(12, len(final_control.ordered_scene_seeds)) if final_control is not None else 12
        ),
        "final_pair_count": 72,
        "final_commands_revealed": False,
        "final_scene_seeds_revealed": False,
        "physical_target_validated": False,
    }


def _evaluate_final_language(
    *,
    corpus: GeneratedLanguageCorpus,
    examples: Sequence[LanguageExample],
    paths: Mapping[str, Path],
    source: SelectedNeuroSymbolicDispatchSource,
    repeat_count: int,
    semantic_maximum_new_tokens: int,
) -> tuple[dict[str, object], NeuroSymbolicRouterV0, dict[str, object]]:
    if len(examples) != 600:
        raise SealedFinalCommandError("sealed language final must contain exactly 600 examples")
    classifier, classifier_identity = _load_classifier_negative(
        checkpoint=paths["classifier"],
        rejection_root=paths["rejection"],
        local_files_only=True,
    )
    qwen17_identity = _frozen_baseline_identity(paths["qwen17"], label="Qwen3-1.7B")
    qwen4_identity = _frozen_baseline_identity(paths["qwen4"], label="direct Qwen3-4B")
    parser = SymbolicLexicalParserV0()
    arbiter = DeterministicSafetyArbiterV0()
    prompt_examples = select_semantic_prompt_examples(
        corpus.examples_for_split(LanguageSplit.TRAIN)
    )
    prompt_content, prompt_fingerprint, prompt_ids = build_semantic_frame_prompt(
        examples=prompt_examples, parser=parser
    )
    if (
        prompt_fingerprint != source.prompt_fingerprint
        or prompt_ids != source.prompt_example_ids
        or parser.contract_fingerprint != source.symbolic_contract_fingerprint
        or arbiter.contract_fingerprint != source.arbiter_fingerprint
        or semantic_schema_fingerprint() != source.semantic_schema_fingerprint
    ):
        raise SealedFinalCommandError("final NeuroSymbolic contracts differ from the router lock")
    config4 = build_qwen4b_loader_config(maximum_new_tokens=128)
    loader4 = TransformersLocalTextGenerator(config=config4, device="cuda", local_files_only=True)
    if (
        loader4.file_identities != dict(source.model_file_identities)
        or loader4.snapshot_size_bytes != source.snapshot_size_bytes
        or _rendered_prompt_fingerprint(loader4, prompt_content)
        != source.rendered_prompt_fingerprint
        or semantic_frame_transport_fingerprint(loader4, prompt_content)
        != source.rendered_prompt_fingerprint
    ):
        raise SealedFinalCommandError("final Qwen3-4B files or chat transport differ")
    extractor = OutlinesQwenSemanticFrameExtractorV0(
        loader=loader4,
        prompt_content=prompt_content,
        maximum_new_tokens=semantic_maximum_new_tokens,
    )
    neuro = NeuroSymbolicRouterV0(parser=parser, semantic_extractor=extractor, arbiter=arbiter)
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
    loader17 = TransformersLocalTextGenerator(config=config17, device="cuda", local_files_only=True)
    direct17 = StructuredLocalLLMRouterV0(
        config=config17, generator=loader17, prompt_examples=direct_prompt_examples
    )
    payloads: dict[str, object] = {}
    candidate_records = None
    ordered = (
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
            "neuro_symbolic_final",
            neuro,
            _eligibility(promotable=True, frozen_negative=False),
        ),
    )
    for name, router, eligibility in ordered:
        records, payload = _router_result_with_contracts(
            router=router,
            examples=examples,
            split=LanguageSplit.FINAL,
            repeat_count=repeat_count,
            eligibility=eligibility,
        )
        payloads[name] = payload
        if name == "neuro_symbolic_final":
            candidate_records = records
    assert candidate_records is not None
    metrics = recompute_router_metrics(records=candidate_records, examples=examples)
    safety, taxonomy = evaluate_router_contracts(records=candidate_records, examples=examples)
    special = special_safety_route_counts(records=candidate_records, examples=examples)
    family_rates = cast(Mapping[str, float], metrics["rejection_family_false_route_rates"])
    gate = evaluate_final_language_quality(
        metrics=metrics,
        safety=safety,
        taxonomy=taxonomy,
        special_route_counts=special,
        rejection_family_unsafe_false_route_rates=family_rates,
        no_prohibited_source_access=True,
    )
    system_metrics = {
        **compute_neuro_symbolic_safety_metrics(records=candidate_records, examples=examples),
        **special,
        "constrained_schema_compilation_seconds": extractor.compilation_seconds,
        "semantic_field_metrics": _semantic_field_metrics(
            records=candidate_records, examples=examples
        ),
    }
    if not classifier.verify_unchanged():
        raise SealedFinalCommandError("frozen classifier weights changed during final evaluation")
    identities = {
        "classifier_negative_baseline": classifier_identity,
        "qwen17b_negative_baseline": qwen17_identity,
        "qwen4b_negative_baseline": qwen4_identity,
        "model_file_identities": loader4.file_identities,
        "snapshot_size_bytes": loader4.snapshot_size_bytes,
        "prompt_fingerprint": prompt_fingerprint,
        "rendered_prompt_fingerprint": source.rendered_prompt_fingerprint,
        "semantic_schema": semantic_frame_schema(),
        "semantic_schema_fingerprint": semantic_schema_fingerprint(),
        "symbolic_contract_fingerprint": parser.contract_fingerprint,
        "arbiter_fingerprint": arbiter.contract_fingerprint,
    }
    del direct17, loader17
    gc.collect()
    torch.cuda.empty_cache()
    return (
        {
            "payloads": payloads,
            "language_quality_gate": gate,
            "system_metrics": system_metrics,
            "identities": identities,
        },
        neuro,
        identities,
    )


def _run_final_rejection_probes(
    *,
    registry: ControllerRegistry,
    loader: StrictPerTaskControllerLoader,
    environment: object,
    router: NeuroSymbolicRouterV0,
) -> dict[str, object]:
    items: list[dict[str, object]] = []
    for case in FINAL_REJECTION_CASES:
        probe = run_rejection_noop_probe(
            registry=registry,
            loader=loader,
            environment=environment,
            router=router,
            command=case.command,
            evaluation_id=f"m5a-sealed-final-rejection:{case.probe_id}",
        )
        items.append({**case.to_dict(), "policy_reset_count": 0, "probe": probe})
    return build_final_rejection_probe_set(items)


def _execute_target(args: argparse.Namespace, paths: Mapping[str, Path]) -> dict[str, object]:
    git = inspect_git_state(PROJECT_ROOT)
    if git.dirty or not git.baseline_tracked:
        raise SealedFinalCommandError("sealed final requires a clean tracked implementation Git")
    if (
        not args.local_files_only
        or args.repeat_count != 2
        or args.semantic_maximum_new_tokens != 256
        or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise SealedFinalCommandError("sealed final requires the frozen one-GPU local-only runtime")
    if final_access_count(paths["output"]) != 0:
        raise SealedFinalCommandError("final access count is not zero")

    corpus = build_language_corpus()
    _language_development, language_final = build_language_schedule_locks(corpus)
    final_control_lock = load_authoritative_final_schedule(paths["schedule"], corpus=corpus)
    sealed_authority = load_sealed_final_authority(paths["schedule"], corpus=corpus)
    if (
        language_final.schedule_fingerprint != EXPECTED_LANGUAGE_FINAL_SCHEDULE_FINGERPRINT
        or final_control_lock.schedule_fingerprint != EXPECTED_CONTROL_FINAL_SCHEDULE_FINGERPRINT
        or sealed_authority["final_language_schedule_fingerprint"]
        != language_final.schedule_fingerprint
        or sealed_authority["final_control_schedule_fingerprint"]
        != final_control_lock.schedule_fingerprint
    ):
        raise SealedFinalCommandError("sealed final schedule fingerprints differ")
    source = validate_selected_neuro_symbolic_dispatch_source(paths["neuro"], corpus=corpus)
    if source.router_lock_fingerprint != EXPECTED_ROUTER_LOCK_FINGERPRINT:
        raise SealedFinalCommandError("selected router lock differs from final authorization")
    registry, locators = _load_frozen_controller_registry(
        checkpoint_root=paths["checkpoints"],
        dataset_root=paths["dataset"],
        runtime_selection_path=paths["runtime"],
    )
    if registry.registry_fingerprint != EXPECTED_CONTROLLER_REGISTRY_FINGERPRINT:
        raise SealedFinalCommandError("controller registry differs from final authorization")
    binding = bind_selected_router_to_controller_registry(source, registry)
    authorization, parent_runtime = _validate_parent_authority(
        paths=paths,
        corpus=corpus,
        registry=registry,
        source=source,
        final_language_fingerprint=language_final.schedule_fingerprint,
        final_control_fingerprint=final_control_lock.schedule_fingerprint,
    )
    archive_fingerprint = _archive_fingerprint_prelock(paths["corpus"], corpus)
    split_fingerprints = {
        split.value: corpus.manifest.split_manifests[split].content_fingerprint
        for split in LanguageSplit
    }
    metric_contract = final_metric_contract()
    identity = M5AFinalRunIdentityV0(
        implementation_git=git.commit,
        final_authorization_fingerprint=EXPECTED_FINAL_AUTHORIZATION_FINGERPRINT,
        router_lock_fingerprint=source.router_lock_fingerprint,
        controller_registry_fingerprint=registry.registry_fingerprint,
        runtime_fingerprint=cast(str, parent_runtime["runtime_contract_fingerprint"]),
        language_schedule_fingerprint=language_final.schedule_fingerprint,
        control_schedule_fingerprint=final_control_lock.schedule_fingerprint,
        corpus_fingerprint=corpus.manifest.corpus_fingerprint,
        split_fingerprints=split_fingerprints,
        model_tokenizer_identity_fingerprint=_baseline_identity_fingerprint(source),
        prompt_fingerprint=source.prompt_fingerprint,
        semantic_schema_fingerprint=source.semantic_schema_fingerprint,
        symbolic_parser_fingerprint=source.symbolic_contract_fingerprint,
        safety_arbiter_fingerprint=source.arbiter_fingerprint,
        metric_contract_fingerprint=cast(str, metric_contract["contract_fingerprint"]),
        evidence_schema_fingerprint=_sha({"schema_version": SEALED_FINAL_EVIDENCE_SCHEMA}),
    )
    attempt = open_final_attempt(paths["output"], run_identity=identity.to_dict())
    attempt_root = cast(str, attempt["attempt_root"])
    try:
        final_examples = materialize_final_language_examples(corpus, authorize_final=True)
        language_result, neuro, model_identities = _evaluate_final_language(
            corpus=corpus,
            examples=final_examples,
            paths=paths,
            source=source,
            repeat_count=args.repeat_count,
            semantic_maximum_new_tokens=args.semantic_maximum_new_tokens,
        )
        final_slots = materialize_final_language_slot_map(corpus, authorize_final=True)
        staged = build_staged_control_schedule(
            final_control_lock,
            stage=M5AStage.SEALED_FINAL,
            authorize_final=True,
        )
        inputs = DevelopmentControlInputs(
            schedule=staged,
            episodes=staged.episodes,
            examples_by_id=final_slots,
            corpus_fingerprint=corpus.manifest.corpus_fingerprint,
        )
        environment = _create_environment()
        try:
            loader = StrictPerTaskControllerLoader(
                registry=registry,
                locators=locators,
                schedule_digest=staged.schedule_fingerprint,
            )
            probes = _run_final_rejection_probes(
                registry=registry,
                loader=loader,
                environment=environment,
                router=neuro,
            )
            dispatcher = ControllerDispatcher(registry=registry, loader=loader)
            control = run_development_control(
                inputs=inputs,
                registry=registry,
                dispatcher=dispatcher,
                environment=environment,
                providers={
                    "oracle": _oracle_provider,
                    NEURO_SYMBOLIC_DISPATCH_ROUTER_LABEL: _provider(neuro),
                },
                output_root=paths["output"] / "raw-control" / identity.run_fingerprint[7:15],
                run_identity={
                    "git_commit": git.commit,
                    "final_run_fingerprint": identity.run_fingerprint,
                    "final_authorization_fingerprint": EXPECTED_FINAL_AUTHORIZATION_FINGERPRINT,
                    "neuro_symbolic_dispatch_source": source.identity_dict(),
                    "neuro_symbolic_controller_binding": binding.to_dict(),
                    "runtime_selection_fingerprint": (
                        registry.entries[0].runtime_selection_fingerprint
                    ),
                },
                require_active_episode_spec=True,
                rejection_noop_probe=None,
            )
        finally:
            close = getattr(environment, "close", None)
            if callable(close):
                close()
        atom_root = Path(cast(str, control["evidence_root"]))
        oracle_records = [
            _read_object(atom_root / "episodes" / "oracle" / f"{index:03d}.json", label="Oracle")
            for index in range(72)
        ]
        learned_records = [
            _read_object(
                atom_root / "episodes" / NEURO_SYMBOLIC_DISPATCH_ROUTER_LABEL / f"{index:03d}.json",
                label="NeuroSymbolic",
            )
            for index in range(72)
        ]
        summaries = control.get("summaries")
        if not isinstance(summaries, Mapping):
            raise SealedFinalCommandError("final control omitted router summaries")
        control_analysis = analyze_final_control_records(
            oracle_records=oracle_records,
            neuro_symbolic_records=learned_records,
            rejection_probe_set=probes,
            examples_by_id=final_slots,
            router_summaries=summaries,
        )
        language_gate = cast(Mapping[str, object], language_result["language_quality_gate"])
        final_result = build_final_result(
            language_gate=language_gate,
            control_analysis=control_analysis,
            independent_evidence_valid=True,
        )
        owner = {
            "schema_version": "langmani-m5a-sealed-final-owner-v0",
            "run_fingerprint": identity.run_fingerprint,
            "implementation_git": git.commit,
            "final_authorization_fingerprint": EXPECTED_FINAL_AUTHORIZATION_FINGERPRINT,
            "final_access_attempt_fingerprint": attempt["attempt_fingerprint"],
            "language_evaluation_fingerprint": _sha(language_result),
            "control_analysis_fingerprint": control_analysis["analysis_fingerprint"],
            "control_run_fingerprint": control["run_fingerprint"],
            "control_record_set_fingerprint": control["record_set_fingerprint"],
        }
        flags = {
            "final_authorization_validated": True,
            "sealed_language_final_accessed": True,
            "sealed_control_final_accessed": True,
            "final_language_evaluation_completed": True,
            "final_language_safety_quality_passed": language_gate[
                "final_language_safety_quality_passed"
            ]
            is True,
            "final_rejection_taxonomy_quality_passed": language_gate[
                "final_rejection_taxonomy_quality_passed"
            ]
            is True,
            "final_control_schedule_validated": True,
            "final_oracle_control_completed": True,
            "final_neuro_symbolic_control_completed": True,
            "final_paired_states_validated": control_analysis["paired_initial_state_count"] == 72,
            "final_routing_results_validated": True,
            "final_rejection_no_dispatch_validated": True,
            "final_failure_attribution_validated": True,
            "final_action_runtime_validated": True,
            "final_control_quality_passed": control_analysis["final_control_quality_passed"]
            is True,
            "final_pipeline_validated": True,
            "final_quality_gate_passed": final_result["final_quality_gate_passed"] is True,
            "physical_target_validated": True,
            "test_split_accessed": False,
            "historical_fresh_accessed": False,
            "m42_final_accessed": False,
            "smolvla_go": False,
        }
        final_language_payloads = cast(Mapping[str, object], language_result["payloads"])
        artifacts: dict[str, object] = {
            "final_authorization.json": authorization,
            "final_run_identity.json": identity.to_dict(),
            "router_identity.json": source.to_dict(),
            "controller_registry.json": registry.to_dict(),
            "runtime_identity.json": {
                "controller_binding": binding.to_dict(),
                "control_mode": "pd_joint_pos",
                "execution_horizon": 10,
                "action_bound_mode": "project",
                "policy_reset_per_episode": True,
                "parent_runtime": parent_runtime,
                "control_atom_evidence_root": str(atom_root),
                "control_run_fingerprint": control["run_fingerprint"],
                "control_record_set_fingerprint": control["record_set_fingerprint"],
                "m2_expert_call_count": 0,
            },
            "language_schedule.json": {
                **language_final.to_dict(),
                "examples": [value.to_dict() for value in final_examples],
                "corpus_archive_fingerprint": archive_fingerprint,
            },
            "language/rule_router.json": final_language_payloads["rule_router"],
            "language/classifier_negative_baseline.json": final_language_payloads[
                "classifier_negative_baseline"
            ],
            "language/qwen17b_negative_baseline.json": final_language_payloads[
                "qwen17b_negative_baseline"
            ],
            "language/qwen4b_negative_baseline.json": final_language_payloads[
                "qwen4b_negative_baseline"
            ],
            "language/neuro_symbolic_final.json": final_language_payloads["neuro_symbolic_final"],
            "language/system_metrics.json": language_result["system_metrics"],
            "language/model_identities.json": model_identities,
            "control_schedule.json": {
                **staged.to_dict(),
                "commands": [
                    {
                        "episode_index": episode.episode_index,
                        "language_example_id": episode.language_example_id,
                        "command": final_slots[episode.language_example_id].raw_text,
                    }
                    for episode in staged.episodes
                ],
            },
            "oracle_episodes.json": {"episodes": oracle_records},
            "neuro_symbolic_episodes.json": {"episodes": learned_records},
            "paired_results.json": {
                "pairs": control_analysis["paired_results"],
                "paired_outcome_counts": control_analysis["paired_outcome_counts"],
            },
            "rejection_probes.json": probes,
            "failure_attribution.json": {
                "counts": control_analysis["failure_attribution_counts"],
                "one_primary_category_per_neuro_symbolic_episode": True,
            },
            "language_quality_gate.json": language_gate,
            "control_quality_gate.json": {
                "schema_version": "langmani-m5a-final-control-quality-gate-v0",
                "gate_items": control_analysis["gate_items"],
                "final_control_quality_passed": control_analysis["final_control_quality_passed"],
            },
            "final_result.json": final_result,
            "summary.md": (
                "# LangMani M5A Sealed Final\n\n"
                f"Language safety quality: `{flags['final_language_safety_quality_passed']}`.\n\n"
                f"Exact rejection taxonomy quality: "
                f"`{flags['final_rejection_taxonomy_quality_passed']}`.\n\n"
                f"Oracle control: {control_analysis['oracle_success_count']}/72.\n\n"
                f"NeuroSymbolic control: {control_analysis['neuro_symbolic_success_count']}/72.\n\n"
                f"Final control quality: `{flags['final_control_quality_passed']}`.\n\n"
                "No M3B test, historical fresh, m42_final_v0, M2 expert, or SmolVLA was accessed.\n"
            ),
        }
        evidence = write_sealed_final_evidence(
            paths["output"], owner=owner, artifacts=artifacts, flags=flags
        )
        close_final_attempt(
            attempt_root,
            completed=True,
            evidence_fingerprint=cast(str, evidence["artifact_fingerprint"]),
        )
        return {
            "schema_version": COMMAND_SCHEMA,
            "mode": "target_final",
            "passed": True,
            **flags,
            "implementation_git": git.commit,
            "final_run_fingerprint": identity.run_fingerprint,
            "evidence_root": evidence["root"],
            "artifact_fingerprint": evidence["artifact_fingerprint"],
            "completion_fingerprint": cast(Mapping[str, object], evidence["complete"])[
                "completion_fingerprint"
            ],
            "language_metrics": {
                name: cast(Mapping[str, object], payload)["metrics"]
                for name, payload in final_language_payloads.items()
            },
            "language_quality_gate": language_gate,
            "control_analysis": control_analysis,
            "final_result": final_result,
            "physical_target_validated": True,
        }
    except Exception as error:
        try:
            close_final_attempt(
                attempt_root,
                completed=False,
                error={
                    "error_type": type(error).__name__,
                    "error_message": str(error) or repr(error),
                },
            )
        except Exception:
            traceback.print_exc()
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report: dict[str, object] = {
        "schema_version": COMMAND_SCHEMA,
        "passed": False,
        "mode": "target_final" if args.target_final else "dry_run",
        "physical_target_validated": False,
        "smolvla_go": False,
    }
    report_path: Path | None = None
    try:
        paths = _paths(args)
        report_path = paths["report"]
        report = _execute_target(args, paths) if args.target_final else _dry_run(args, paths)
    except Exception as error:  # noqa: BLE001 - command boundary preserves exact failure
        traceback.print_exc()
        report.update(
            {
                "error_type": type(error).__name__,
                "error_message": str(error) or repr(error),
                "test_split_accessed": False,
                "historical_fresh_accessed": False,
                "m42_final_accessed": False,
                "smolvla_go": False,
            }
        )
    if report_path is not None:
        try:
            atomic_write_json(report_path, report)
        except Exception:
            traceback.print_exc()
            return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
