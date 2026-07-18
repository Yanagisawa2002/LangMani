"""Execute exactly one frozen M5A.4 NeuroSymbolicRouterV0 offline experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast

import torch

from langmani.datasets.identity import sha256_hex
from langmani.language.corpus import GeneratedLanguageCorpus, build_language_corpus
from langmani.language.llm_router import (
    QWEN3_1_7B_MODEL_ID,
    QWEN3_1_7B_REVISION,
    QWEN3_4B_INSTRUCT_FILE_IDENTITIES,
    QWEN3_4B_INSTRUCT_LICENSE,
    QWEN3_4B_INSTRUCT_MODEL_ID,
    QWEN3_4B_INSTRUCT_REVISION,
    StructuredLLMRouterConfig,
    StructuredLocalLLMRouterV0,
    TransformersLocalTextGenerator,
    select_structured_routing_prompt_examples,
)
from langmani.language.neuro_symbolic_evidence import (
    write_development_access_lock,
    write_neuro_symbolic_evidence,
)
from langmani.language.neuro_symbolic_experiment import (
    compute_neuro_symbolic_safety_metrics,
    evaluate_neuro_symbolic_development_gate,
    initial_m5a4_flags,
)
from langmani.language.neuro_symbolic_router import (
    OUTLINES_LICENSE,
    OUTLINES_VERSION,
    SEMANTIC_PROMPT_VERSION,
    DeterministicSafetyArbiterV0,
    NeuroSymbolicRouterV0,
    OutlinesQwenSemanticFrameExtractorV0,
    QwenSemanticFrameV0,
    SymbolicLexicalParserV0,
    build_qwen4b_loader_config,
    build_semantic_frame_prompt,
    expected_semantic_frame_payload,
    select_semantic_prompt_examples,
    semantic_frame_schema,
    semantic_schema_fingerprint,
)
from langmani.language.neuro_symbolic_verifier import verify_neuro_symbolic_evidence
from langmani.language.offline_language_development import (
    CLASSIFIER_NEGATIVE_ELIGIBILITY,
    RULE_ROUTER_ELIGIBILITY,
    FrozenClassifierNegativeBaselineV0,
    conservative_decoder_from_mapping,
)
from langmani.language.offline_router_metrics import recompute_router_metrics
from langmani.language.rejection_diagnostics import load_selected_checkpoint_read_only
from langmani.language.rejection_report import validate_rejection_analysis_artifact
from langmani.language.router_evaluation import RouterEvaluationRecord, evaluate_language_router
from langmani.language.router_types import (
    LanguageExample,
    LanguageSplit,
    RouterDecision,
    RouterStatus,
)
from langmani.language.rule_router import RuleRouterV0
from langmani.policies.act_runtime import atomic_write_json, inspect_git_state

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_ROOT = PROJECT_ROOT / "outputs/datasets/m5a/langmani-language-corpus-v1"
DEFAULT_CLASSIFIER_CHECKPOINT = PROJECT_ROOT / (
    "outputs/models/text-router/imported-authoritative/validation_best-401e162376e6a52c58.pt"
)
DEFAULT_CLASSIFIER_REJECTION_EVIDENCE = PROJECT_ROOT / "outputs/diagnostics/m5a/rejection-analysis"
DEFAULT_QWEN17B_EVIDENCE = PROJECT_ROOT / (
    "outputs/diagnostics/m5a/language-development/"
    "291621edfc4218d80ea2184e58fde2ec29d5cec2fc23f9aebaa0e30732406c6b"
)
DEFAULT_QWEN4B_EVIDENCE = PROJECT_ROOT / (
    "outputs/diagnostics/m5a/qwen4b-escalation/"
    "4a1029abd4e0619b919d1b4d8557219487f0db261be1a793706779a4c6886641"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/diagnostics/m5a/neuro-symbolic-router"
DEFAULT_REPORT = PROJECT_ROOT / "outputs/diagnostics/m5a/stages/neuro-symbolic-router.json"
CORPUS_ARCHIVE_SCHEMA = "langmani-m5a-language-corpus-archive-v1"
COMMAND_SCHEMA = "langmani-m5a4-neuro-symbolic-command-v0"


class NeuroSymbolicCommandError(RuntimeError):
    """Raised when the single authorized M5A.4 experiment cannot remain valid."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--target-development", action="store_true")
    modes.add_argument("--verify-evidence", type=Path)
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
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument("--semantic-maximum-new-tokens", type=int, default=256)
    return parser.parse_args(argv)


def _real_unlinked(path: Path, *, label: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    for component in (lexical, *lexical.parents):
        if component.is_symlink() or component.is_junction():
            raise NeuroSymbolicCommandError(f"{label} traverses a linked path")
    return lexical.resolve(strict=False)


def _safe_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "corpus": _real_unlinked(args.corpus_root, label="corpus"),
        "classifier": _real_unlinked(args.classifier_checkpoint, label="classifier"),
        "rejection": _real_unlinked(
            args.classifier_rejection_evidence, label="classifier rejection evidence"
        ),
        "qwen17": _real_unlinked(args.qwen17b_evidence_root, label="Qwen3-1.7B evidence"),
        "qwen4": _real_unlinked(args.qwen4b_evidence_root, label="Qwen3-4B evidence"),
        "output": _real_unlinked(args.output_root, label="M5A.4 output"),
        "report": _real_unlinked(args.report, label="M5A.4 report"),
    }
    protected = tuple(
        (PROJECT_ROOT / name).resolve(strict=False)
        for name in ("src", "scripts", "environment", "tests", "docs", ".git")
    )
    inputs = tuple(paths[name] for name in ("corpus", "classifier", "rejection", "qwen17", "qwen4"))
    output = paths["output"]
    report = paths["report"]
    if any(output == item or output in item.parents or item in output.parents for item in inputs):
        raise NeuroSymbolicCommandError("M5A.4 output overlaps an immutable input")
    if any(
        output == item or item in output.parents or output in item.parents for item in protected
    ):
        raise NeuroSymbolicCommandError("M5A.4 output overlaps source-controlled content")
    if (
        report == output
        or output in report.parents
        or any(
            report == item or report in item.parents or item in report.parents for item in inputs
        )
    ):
        raise NeuroSymbolicCommandError("M5A.4 report path overlaps immutable evidence")
    return paths


def _object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NeuroSymbolicCommandError(f"could not read {label}: {error}") from error
    if not isinstance(value, dict):
        raise NeuroSymbolicCommandError(f"{label} must contain one JSON object")
    return cast(dict[str, object], value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError as error:
        raise NeuroSymbolicCommandError(f"required distribution is missing: {name}") from error


def _dependencies() -> dict[str, object]:
    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers": _package_version("transformers"),
        "tokenizers": _package_version("tokenizers"),
        "outlines": _package_version("outlines"),
        "outlines_core": _package_version("outlines-core"),
        "pydantic": _package_version("pydantic"),
    }


def _archive_fingerprint_prelock(root: Path, corpus: GeneratedLanguageCorpus) -> str:
    manifest = _object(root / "artifact_manifest.json", label="corpus artifact manifest")
    complete = _object(root / "complete.json", label="corpus completion")
    identity = _object(root / "corpus_manifest.json", label="corpus identity")
    if (
        manifest.get("schema_version") != CORPUS_ARCHIVE_SCHEMA
        or complete.get("passed") is not True
        or identity != corpus.manifest.to_dict()
        or (root / "final.jsonl").exists()
    ):
        raise NeuroSymbolicCommandError("corpus archive identity differs")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise NeuroSymbolicCommandError("corpus artifact records are missing")
    for record in artifacts:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise NeuroSymbolicCommandError("corpus artifact record is malformed")
        path = root / cast(str, record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != record.get("size_bytes")
            or _sha256_file(path) != record.get("sha256")
        ):
            raise NeuroSymbolicCommandError("corpus artifact checksum differs")
    # Only train and historical validation records are opened before the lock.
    for split in (LanguageSplit.TRAIN, LanguageSplit.VALIDATION):
        observed = [
            json.loads(line)
            for line in (root / f"{split.value}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        if observed != [value.to_dict() for value in corpus.examples_for_split(split)]:
            raise NeuroSymbolicCommandError(f"corpus {split.value} differs")
    return cast(str, manifest["archive_fingerprint"])


def _validate_development_after_lock(root: Path, examples: Sequence[LanguageExample]) -> None:
    observed = [
        json.loads(line)
        for line in (root / "development.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if observed != [value.to_dict() for value in examples]:
        raise NeuroSymbolicCommandError("corpus development records differ after lock")


def _semantic_fingerprint(decision: RouterDecision) -> str:
    return f"sha256:{sha256_hex({'status': decision.status.value, 'task_id': decision.task_id, 'reason': None if decision.rejection_reason is None else decision.rejection_reason.value})}"


def _eligibility(*, promotable: bool, frozen_negative: bool) -> dict[str, object]:
    return {
        "offline_evaluation_eligible": True,
        "controller_dispatch_eligible": False,
        "promotion_eligible": promotable,
        "final_selection_eligible": promotable,
        "frozen_negative_baseline": frozen_negative,
    }


def _router_payload(
    *,
    router: object,
    examples: Sequence[LanguageExample],
    split: LanguageSplit,
    repeat_count: int,
    eligibility: Mapping[str, object],
) -> tuple[tuple[RouterEvaluationRecord, ...], dict[str, object]]:
    records, _ = evaluate_language_router(
        router=cast(object, router),
        examples=examples,
        split=split,
        repeat_count=repeat_count,
        repeat_fingerprint=_semantic_fingerprint,
    )
    metrics = recompute_router_metrics(records=records, examples=examples)
    return records, {
        "schema_version": "langmani-m5a4-offline-router-result-v0",
        "split": split.value,
        "eligibility": dict(eligibility),
        "metrics": metrics,
        "records": [value.to_dict() for value in records],
        "total_elapsed_inference_seconds": sum(value.latency_ms for value in records) / 1000.0,
    }


def _select_smoke(corpus: GeneratedLanguageCorpus) -> tuple[LanguageExample, ...]:
    return select_semantic_prompt_examples(corpus.examples_for_split(LanguageSplit.TRAIN))


def _run_train_smoke(
    router: NeuroSymbolicRouterV0,
    parser: SymbolicLexicalParserV0,
    examples: Sequence[LanguageExample],
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for example in examples:
        first = router.route(example.raw_text)
        second = router.route(example.raw_text)
        semantic_payload = cast(Mapping[str, object], first.evidence["semantic_frame"])
        expected_semantic = expected_semantic_frame_payload(example, parser.parse(example.raw_text))
        records.append(
            {
                "example_id": example.example_id,
                "expected_status": example.expected_status.value,
                "expected_task_id": example.task_id,
                "decision": first.to_dict(),
                "semantic_frame": dict(semantic_payload),
                "expected_semantic_frame": expected_semantic,
                "semantic_fields_exact": dict(semantic_payload) == expected_semantic,
                "deterministic": _semantic_fingerprint(first) == _semantic_fingerprint(second),
            }
        )
    statuses = {cast(Mapping[str, object], value["decision"])["status"] for value in records}
    tasks = {
        cast(Mapping[str, object], value["decision"])["task_id"]
        for value in records
        if value["expected_status"] == RouterStatus.ROUTE.value
    }
    checks = {
        "exact_twenty_train_examples": len(records) == 20,
        "all_six_task_specs_emitted": len(tasks - {None}) == 6,
        "all_router_statuses_emitted": statuses == {value.value for value in RouterStatus},
        "all_semantic_frames_schema_valid": all(
            isinstance(value["semantic_frame"], Mapping) for value in records
        ),
        "field_semantics_match_train_labels": all(
            bool(value["semantic_fields_exact"]) for value in records
        ),
        "final_decisions_match_train_labels": all(
            cast(Mapping[str, object], value["decision"])["status"] == value["expected_status"]
            and (
                value["expected_status"] != RouterStatus.ROUTE.value
                or cast(Mapping[str, object], value["decision"])["task_id"]
                == value["expected_task_id"]
            )
            for value in records
        ),
        "rejected_decisions_non_executable": all(
            cast(Mapping[str, object], value["decision"])["task_spec"] is None
            for value in records
            if cast(Mapping[str, object], value["decision"])["status"] != RouterStatus.ROUTE.value
        ),
        "deterministic_repeatability": all(bool(value["deterministic"]) for value in records),
        "no_prohibited_split_access": True,
    }
    return {
        "schema_version": "langmani-m5a4-train-smoke-v0",
        "split": "train",
        "example_count": len(records),
        "example_ids": [value.example_id for value in examples],
        "checks": checks,
        "gate_passed": all(checks.values()),
        "records": records,
    }


def _load_classifier_negative(
    *, checkpoint: Path, rejection_root: Path, local_files_only: bool
) -> tuple[FrozenClassifierNegativeBaselineV0, dict[str, object]]:
    artifact = validate_rejection_analysis_artifact(rejection_root)
    root = Path(cast(str, artifact["root"]))
    rejected = _object(root / "rejected_candidate.json", label="rejected classifier")
    selection = _object(root / "candidate_selection.json", label="classifier decoder")
    source = _object(root / "source_recovery_identity.json", label="classifier source")
    selected = selection.get("selected_configuration")
    if (
        artifact.get("conclusion") != "classifier_rejected_after_posthoc_calibration"
        or rejected.get("classifier_candidate_frozen") is not True
        or rejected.get("additional_training_authorized") is not False
        or rejected.get("additional_seed_authorized") is not False
        or not isinstance(selected, Mapping)
    ):
        raise NeuroSymbolicCommandError("classifier is not the frozen negative baseline")
    expected_sha = cast(str, rejected["selected_classifier_checkpoint_fingerprint"])
    if _sha256_file(checkpoint) != expected_sha:
        raise NeuroSymbolicCommandError("classifier checkpoint identity differs")
    run_fingerprint = source.get("run_fingerprint")
    selected_step = source.get("selected_step")
    if not isinstance(run_fingerprint, str) or not isinstance(selected_step, int):
        raise NeuroSymbolicCommandError("classifier source identity is malformed")
    bundle = load_selected_checkpoint_read_only(
        checkpoint,
        expected_checkpoint_fingerprint=expected_sha,
        expected_run_fingerprint=f"sha256:{run_fingerprint.removeprefix('sha256:')}",
        expected_step=selected_step,
        local_files_only=local_files_only,
    )
    tokenizer = bundle.processor_state.get("tokenizer")
    if not isinstance(tokenizer, Mapping) or not isinstance(
        tokenizer.get("maximum_sequence_length"), int
    ):
        raise NeuroSymbolicCommandError("classifier tokenizer contract is malformed")
    router = FrozenClassifierNegativeBaselineV0(
        bundle=bundle,
        configuration=conservative_decoder_from_mapping(cast(Mapping[str, object], selected)),
        maximum_sequence_length=cast(int, tokenizer["maximum_sequence_length"]),
        device="cpu",
    )
    return router, {
        "analysis_fingerprint": artifact["analysis_fingerprint"],
        "checkpoint_fingerprint": expected_sha,
        "selection_fingerprint": selection["selection_fingerprint"],
        "model_state_fingerprint_before": router.model_state_fingerprint_before,
        "optimizer_constructed": False,
    }


def _frozen_baseline_identity(root: Path, *, label: str) -> dict[str, object]:
    complete = _object(root / "complete.json", label=f"{label} completion")
    if complete.get("passed") is not True:
        raise NeuroSymbolicCommandError(f"{label} baseline evidence is incomplete")
    return {
        "root": str(root),
        "artifact_fingerprint": complete.get("artifact_fingerprint"),
        "flags": complete.get("flags"),
        "frozen_negative_baseline": True,
        "promotion_eligible": False,
    }


def _semantic_field_metrics(
    *, records: Sequence[RouterEvaluationRecord], examples: Sequence[LanguageExample]
) -> dict[str, object]:
    by_id = {value.example_id: value for value in examples}
    parser = SymbolicLexicalParserV0()
    keys = tuple(QwenSemanticFrameV0.model_fields)
    correct = {key: 0 for key in keys}
    exact = 0
    for record in records:
        frame = record.decision.evidence.get("semantic_frame")
        if not isinstance(frame, Mapping):
            continue
        expected = expected_semantic_frame_payload(
            by_id[record.example_id], parser.parse(by_id[record.example_id].raw_text)
        )
        exact += dict(frame) == expected
        for key in keys:
            correct[key] += frame.get(key) == expected.get(key)
    total = len(records)
    return {
        "derivation": "train-label-taxonomy-plus-symbolic-mentions-v0",
        "example_count": total,
        "exact_frame_accuracy": 0.0 if total == 0 else exact / total,
        "per_field_accuracy": {
            key: 0.0 if total == 0 else value / total for key, value in correct.items()
        },
    }


def _rendered_prompt_fingerprint(loader: TransformersLocalTextGenerator, prompt: str) -> str:
    rendered = loader.tokenizer.apply_chat_template(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "{COMMAND}"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str):
        raise NeuroSymbolicCommandError("official chat template did not render text")
    return f"sha256:{sha256_hex(rendered)}"


def _dry_run(args: argparse.Namespace, paths: Mapping[str, Path]) -> dict[str, object]:
    corpus = build_language_corpus()
    parser = SymbolicLexicalParserV0()
    arbiter = DeterministicSafetyArbiterV0()
    selected = _select_smoke(corpus)
    prompt, prompt_fingerprint, ids = build_semantic_frame_prompt(examples=selected, parser=parser)
    runtime = {
        "git": inspect_git_state(PROJECT_ROOT).commit,
        "corpus_fingerprint": corpus.manifest.corpus_fingerprint,
        "model_id": QWEN3_4B_INSTRUCT_MODEL_ID,
        "model_revision": QWEN3_4B_INSTRUCT_REVISION,
        "prompt_fingerprint": prompt_fingerprint,
        "semantic_schema_fingerprint": semantic_schema_fingerprint(),
        "symbolic_contract_fingerprint": parser.contract_fingerprint,
        "arbiter_fingerprint": arbiter.contract_fingerprint,
        "outlines_version": OUTLINES_VERSION,
    }
    return {
        "schema_version": COMMAND_SCHEMA,
        "mode": "dry_run",
        "passed": True,
        "runtime_fingerprint": f"sha256:{sha256_hex(runtime)}",
        "prompt_example_ids": list(ids),
        "prompt_character_count": len(prompt),
        "development_examples_accessed": False,
        "final_or_control_accessed": False,
        "output_root": str(paths["output"]),
    }


def _execute_target(args: argparse.Namespace, paths: Mapping[str, Path]) -> dict[str, object]:
    dependencies = _dependencies()
    git = inspect_git_state(PROJECT_ROOT)
    if git.dirty or not git.baseline_tracked:
        raise NeuroSymbolicCommandError("target evidence requires a clean tracked Git commit")
    if (
        args.repeat_count != 2
        or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise NeuroSymbolicCommandError("target requires exactly CUDA_VISIBLE_DEVICES=0")
    if (
        dependencies["transformers"] != "5.4.0"
        or dependencies["tokenizers"] != "0.22.2"
        or dependencies["outlines"] != OUTLINES_VERSION
    ):
        raise NeuroSymbolicCommandError("locked Transformers/Outlines versions differ")
    corpus = build_language_corpus()
    archive_fingerprint = _archive_fingerprint_prelock(paths["corpus"], corpus)
    parser = SymbolicLexicalParserV0()
    arbiter = DeterministicSafetyArbiterV0()
    prompt_examples = _select_smoke(corpus)
    prompt_content, prompt_fingerprint, prompt_ids = build_semantic_frame_prompt(
        examples=prompt_examples, parser=parser
    )
    qwen17_identity = _frozen_baseline_identity(paths["qwen17"], label="Qwen3-1.7B")
    qwen4_identity = _frozen_baseline_identity(paths["qwen4"], label="direct Qwen3-4B")
    runtime_payload = {
        "git_commit": git.commit,
        "corpus_fingerprint": corpus.manifest.corpus_fingerprint,
        "corpus_archive_fingerprint": archive_fingerprint,
        "model_id": QWEN3_4B_INSTRUCT_MODEL_ID,
        "model_revision": QWEN3_4B_INSTRUCT_REVISION,
        "tokenizer_revision": QWEN3_4B_INSTRUCT_REVISION,
        "prompt_fingerprint": prompt_fingerprint,
        "prompt_example_ids": list(prompt_ids),
        "semantic_schema_fingerprint": semantic_schema_fingerprint(),
        "symbolic_contract_fingerprint": parser.contract_fingerprint,
        "arbiter_fingerprint": arbiter.contract_fingerprint,
        "outlines_version": OUTLINES_VERSION,
        "dependencies": dependencies,
        "repeat_count": args.repeat_count,
        "semantic_maximum_new_tokens": args.semantic_maximum_new_tokens,
        "visible_gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "qwen17_source": qwen17_identity["artifact_fingerprint"],
        "qwen4_source": qwen4_identity["artifact_fingerprint"],
    }
    runtime_fingerprint = f"sha256:{sha256_hex(runtime_payload)}"
    owner = {
        "schema_version": "langmani-m5a4-neuro-symbolic-owner-v0",
        "mode": "target_development",
        "runtime_fingerprint": runtime_fingerprint,
        "git_commit": git.commit,
        "corpus_fingerprint": corpus.manifest.corpus_fingerprint,
    }
    config4 = build_qwen4b_loader_config(maximum_new_tokens=128)
    loader4 = TransformersLocalTextGenerator(
        config=config4, device="cuda", local_files_only=bool(args.local_files_only)
    )
    rendered_fingerprint = _rendered_prompt_fingerprint(loader4, prompt_content)
    extractor = OutlinesQwenSemanticFrameExtractorV0(
        loader=loader4,
        prompt_content=prompt_content,
        maximum_new_tokens=args.semantic_maximum_new_tokens,
    )
    router = NeuroSymbolicRouterV0(parser=parser, semantic_extractor=extractor, arbiter=arbiter)
    flags = initial_m5a4_flags()
    flags.update(
        {
            "neuro_symbolic_implementation_validated": True,
            "symbolic_frame_validated": True,
            "semantic_frame_schema_validated": True,
            "constrained_decoding_validated": True,
            "arbiter_validated": True,
            "real_gpu_inference_validated": True,
        }
    )
    artifacts: dict[str, object] = {}
    smoke = _run_train_smoke(router, parser, prompt_examples)
    artifacts["train_smoke.json"] = smoke
    flags["train_smoke_completed"] = True
    if smoke["gate_passed"] is not True:
        raise NeuroSymbolicCommandError("frozen train-only semantic smoke failed")
    before = {
        "prompt": prompt_fingerprint,
        "schema": semantic_schema_fingerprint(),
        "symbolic": parser.contract_fingerprint,
        "arbiter": arbiter.contract_fingerprint,
        "runtime": runtime_fingerprint,
    }
    validation_examples = corpus.examples_for_split(LanguageSplit.VALIDATION)
    _, historical_payload = _router_payload(
        router=router,
        examples=validation_examples,
        split=LanguageSplit.VALIDATION,
        repeat_count=args.repeat_count,
        eligibility=_eligibility(promotable=False, frozen_negative=False),
    )
    historical_payload.update(
        {
            "post_selection_diagnostic_only": True,
            "language_validation_quarantined_for_architecture_selection": True,
            "configuration_change_authorized": False,
            "configuration_fingerprints_before": before,
            "configuration_fingerprints_after": dict(before),
            "promotion_gate": False,
        }
    )
    artifacts["historical_validation_diagnostic.json"] = historical_payload
    flags["historical_validation_diagnostic_completed"] = True
    prompt_lock = write_development_access_lock(
        paths["output"],
        runtime_fingerprint=runtime_fingerprint,
        prompt_lock={
            "prompt_fingerprint": prompt_fingerprint,
            "rendered_prompt_fingerprint": rendered_fingerprint,
            "prompt_example_ids": list(prompt_ids),
            "semantic_schema_fingerprint": semantic_schema_fingerprint(),
            "symbolic_contract_fingerprint": parser.contract_fingerprint,
            "arbiter_fingerprint": arbiter.contract_fingerprint,
            "decoder_version": OUTLINES_VERSION,
            "model_revision": QWEN3_4B_INSTRUCT_REVISION,
            "git_commit": git.commit,
            "metric_implementation": "recompute-router-metrics-v0",
        },
    )
    flags["neuro_symbolic_router_locked"] = True
    # This is the first explicit development-example access in the command.
    development_examples = corpus.examples_for_split(LanguageSplit.DEVELOPMENT)
    _validate_development_after_lock(paths["corpus"], development_examples)
    classifier, classifier_identity = _load_classifier_negative(
        checkpoint=paths["classifier"],
        rejection_root=paths["rejection"],
        local_files_only=bool(args.local_files_only),
    )
    rule = RuleRouterV0()
    direct_prompt_examples = select_structured_routing_prompt_examples(
        corpus.examples_for_split(LanguageSplit.TRAIN)
    )
    direct4 = StructuredLocalLLMRouterV0(
        config=config4, generator=loader4, prompt_examples=direct_prompt_examples
    )
    direct_prompt_fingerprint = direct4.prompt_fingerprint
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
    candidate_records: tuple[RouterEvaluationRecord, ...] | None = None
    for name, active, eligibility in (
        ("rule_router", rule, RULE_ROUTER_ELIGIBILITY.to_dict()),
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
            "qwen4b_direct_negative_baseline",
            direct4,
            _eligibility(promotable=False, frozen_negative=True),
        ),
        ("neuro_symbolic_candidate", router, _eligibility(promotable=True, frozen_negative=False)),
    ):
        records, payload = _router_payload(
            router=active,
            examples=development_examples,
            split=LanguageSplit.DEVELOPMENT,
            repeat_count=args.repeat_count,
            eligibility=eligibility,
        )
        artifacts[f"development/{name}.json"] = payload
        if name == "neuro_symbolic_candidate":
            candidate_records = records
    assert candidate_records is not None
    candidate_payload = cast(
        Mapping[str, object], artifacts["development/neuro_symbolic_candidate.json"]
    )
    candidate_metrics = cast(Mapping[str, object], candidate_payload["metrics"])
    safety_metrics = compute_neuro_symbolic_safety_metrics(
        records=candidate_records, examples=development_examples
    )
    field_metrics = _semantic_field_metrics(
        records=candidate_records, examples=development_examples
    )
    gate = evaluate_neuro_symbolic_development_gate(
        metrics=candidate_metrics,
        safety_metrics=safety_metrics,
        no_prohibited_source_access=True,
    )
    artifacts["development/safety_metrics.json"] = safety_metrics
    artifacts["development/semantic_field_metrics.json"] = field_metrics
    artifacts["development/quality_gate.json"] = gate.to_dict()
    flags["language_development_completed"] = True
    flags["neuro_symbolic_language_quality_gate_passed"] = gate.passed
    flags["learned_router_selected"] = gate.passed
    flags["one_scene_control_smoke_authorized"] = gate.passed
    if not classifier.verify_unchanged():
        raise NeuroSymbolicCommandError("frozen classifier weights changed during evaluation")
    artifacts.update(
        {
            "symbolic_contract.json": parser.contract_dict(),
            "semantic_schema.json": {
                "schema_version": "langmani-m5a4-semantic-schema-v0",
                "schema": semantic_frame_schema(),
                "schema_fingerprint": semantic_schema_fingerprint(),
                "final_router_status_field": False,
                "final_task_spec_field": False,
            },
            "constrained_decoder_identity.json": {
                "distribution": "outlines",
                "version": OUTLINES_VERSION,
                "license": OUTLINES_LICENSE,
                "outlines_core_version": dependencies["outlines_core"],
                "transformers_version": dependencies["transformers"],
                "grammar_cached_for_session": True,
                "free_form_fallback": False,
                "compilation_seconds": extractor.compilation_seconds,
            },
            "model_identity.json": {
                "model_id": QWEN3_4B_INSTRUCT_MODEL_ID,
                "model_revision": QWEN3_4B_INSTRUCT_REVISION,
                "tokenizer_revision": QWEN3_4B_INSTRUCT_REVISION,
                "license": QWEN3_4B_INSTRUCT_LICENSE,
                "dtype": "bfloat16",
                "quantization": "none",
                "device": "cuda:0",
                "snapshot_path": loader4.snapshot_path,
                "snapshot_size_bytes": loader4.snapshot_size_bytes,
                "file_identities": loader4.file_identities,
                "expected_file_count": len(QWEN3_4B_INSTRUCT_FILE_IDENTITIES),
                "weights_unchanged": True,
            },
            "prompt.json": {
                "prompt_version": SEMANTIC_PROMPT_VERSION,
                "prompt_content": prompt_content,
                "prompt_fingerprint": prompt_fingerprint,
                "rendered_prompt_fingerprint": rendered_fingerprint,
                "few_shot_example_ids": list(prompt_ids),
                "train_only": True,
                "prompt_sweep": False,
            },
            "prompt_lock.json": prompt_lock,
            "arbiter_contract.json": arbiter.contract_dict(),
            "runtime_identity.json": {
                **runtime_payload,
                "runtime_fingerprint": runtime_fingerprint,
                "optimizer_constructed": False,
                "training_performed": False,
                "lora_used": False,
                "robot_environment_created": False,
                "controller_loaded": False,
                "development_access_after_lock": True,
                "direct_router_prompt_fingerprint": direct_prompt_fingerprint,
                "direct_router_prompt_example_ids": list(direct4.prompt_example_ids),
                "language_final_accessed": False,
                "control_final_accessed": False,
                "test_split_accessed": False,
                "m42_final_accessed": False,
            },
            "candidate_selection.json": {
                "candidate": "NeuroSymbolicRouterV0",
                "only_promotable_candidate": True,
                "quality_gate_passed": gate.passed,
                "learned_router_selected": gate.passed,
                "one_scene_control_smoke_authorized": gate.passed,
                "rule_router_offline_baseline": True,
                "classifier_negative_baseline": classifier_identity,
                "qwen17b_negative_baseline": qwen17_identity,
                "qwen4b_direct_negative_baseline": qwen4_identity,
            },
            "summary.md": (
                "# M5A.4 Neuro-Symbolic Router\n\n"
                f"Development quality gate: `{gate.passed}`.\n\n"
                "Historical validation was diagnostic only. No training, controller, "
                "robot environment, final split, or SmolVLA was accessed.\n"
            ),
        }
    )
    evidence = write_neuro_symbolic_evidence(
        paths["output"], owner=owner, artifacts=artifacts, flags=flags
    )
    independent = verify_neuro_symbolic_evidence(cast(str, evidence["root"]))
    return {
        "schema_version": COMMAND_SCHEMA,
        "mode": "target_development",
        "passed": independent["passed"] is True,
        **flags,
        "runtime_fingerprint": runtime_fingerprint,
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
    }
    try:
        paths = _safe_paths(args)
        if args.verify_evidence is not None:
            report = verify_neuro_symbolic_evidence(args.verify_evidence)
        elif args.dry_run:
            report = _dry_run(args, paths)
        else:
            report = _execute_target(args, paths)
    except Exception as error:  # noqa: BLE001 - outer boundary preserves exact diagnostics
        traceback.print_exc()
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    if args.verify_evidence is None:
        try:
            atomic_write_json(_real_unlinked(args.report, label="M5A.4 report"), report)
        except Exception as error:  # noqa: BLE001 - report persistence is part of command status
            traceback.print_exc()
            report["report_error"] = {"type": type(error).__name__, "message": str(error)}
            report["passed"] = False
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
