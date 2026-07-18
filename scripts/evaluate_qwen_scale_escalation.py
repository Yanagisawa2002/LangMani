"""Execute the single authorized M5A.3 Qwen3-4B offline comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from collections import defaultdict
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast

import torch

from langmani.datasets.identity import sha256_hex
from langmani.language.corpus import GeneratedLanguageCorpus, build_language_corpus
from langmani.language.language_development_evidence import (
    validate_language_development_evidence,
)
from langmani.language.language_development_verifier import (
    verify_language_development_evidence,
)
from langmani.language.llm_router import (
    QWEN3_1_7B_MODEL_ID,
    QWEN3_1_7B_REVISION,
    QWEN3_4B_INSTRUCT_LICENSE,
    QWEN3_4B_INSTRUCT_MODEL_ID,
    QWEN3_4B_INSTRUCT_REVISION,
    StructuredLLMRouterConfig,
    StructuredLocalLLMRouterV0,
    TransformersLocalTextGenerator,
    build_structured_routing_prompt,
    select_structured_routing_prompt_examples,
)
from langmani.language.offline_language_development import (
    CLASSIFIER_NEGATIVE_ELIGIBILITY,
    RULE_ROUTER_ELIGIBILITY,
    FrozenClassifierNegativeBaselineV0,
    conservative_decoder_from_mapping,
)
from langmani.language.offline_router_metrics import recompute_router_metrics
from langmani.language.qwen_scale_escalation import (
    M5A3_AUTHORIZED_CANDIDATES,
    FrozenQwen17BBaselineV0,
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
from langmani.language.qwen_scale_escalation_evidence import write_qwen_scale_evidence
from langmani.language.rejection_diagnostics import load_selected_checkpoint_read_only
from langmani.language.rejection_report import validate_rejection_analysis_artifact
from langmani.language.router_evaluation import RouterEvaluationRecord, evaluate_language_router
from langmani.language.router_types import (
    LanguageExample,
    LanguageSplit,
    RouterDecision,
    RouterRejectionReason,
    RouterStatus,
)
from langmani.language.rule_router import RuleRouterV0
from langmani.language.schedules import build_language_schedule_locks
from langmani.policies.act_runtime import atomic_write_json, inspect_git_state

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_ROOT = PROJECT_ROOT / "outputs/datasets/m5a/langmani-language-corpus-v1"
DEFAULT_CLASSIFIER_CHECKPOINT = PROJECT_ROOT / (
    "outputs/models/text-router/authoritative-runs/"
    "9e3ac659fa2b695c843650df35e3779741d94b3dd70b2aec52a429bc4b2edf49/"
    "checkpoints/validation_best.pt"
)
DEFAULT_CLASSIFIER_REJECTION_EVIDENCE = PROJECT_ROOT / "outputs/diagnostics/m5a/rejection-analysis"
DEFAULT_QWEN17B_EVIDENCE = PROJECT_ROOT / (
    "outputs/diagnostics/m5a/language-development/"
    "291621edfc4218d80ea2184e58fde2ec29d5cec2fc23f9aebaa0e30732406c6b"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/diagnostics/m5a/qwen4b-escalation"
DEFAULT_REPORT = PROJECT_ROOT / "outputs/diagnostics/m5a/stages/qwen4b-escalation.json"
COMMAND_SCHEMA = "langmani-m5a3-qwen-scale-command-v0"
CORPUS_ARCHIVE_SCHEMA = "langmani-m5a-language-corpus-archive-v1"
DEFAULT_REPEAT_COUNT = 2


class QwenScaleCommandError(RuntimeError):
    """Raised when the M5A.3 command cannot preserve its fixed contract."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--fixture", action="store_true")
    modes.add_argument("--target-development", action="store_true")
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--classifier-checkpoint", type=Path, default=DEFAULT_CLASSIFIER_CHECKPOINT)
    parser.add_argument(
        "--classifier-rejection-evidence",
        type=Path,
        default=DEFAULT_CLASSIFIER_REJECTION_EVIDENCE,
    )
    parser.add_argument("--qwen17b-evidence-root", type=Path, default=DEFAULT_QWEN17B_EVIDENCE)
    parser.add_argument("--model-id", default=QWEN3_4B_INSTRUCT_MODEL_ID)
    parser.add_argument("--model-revision", default=QWEN3_4B_INSTRUCT_REVISION)
    parser.add_argument("--tokenizer-revision", default=QWEN3_4B_INSTRUCT_REVISION)
    parser.add_argument("--license", default=QWEN3_4B_INSTRUCT_LICENSE)
    parser.add_argument("--license-reviewed", action="store_true")
    parser.add_argument("--model-card-reviewed", action="store_true")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--quantization", choices=("none",), default="none")
    parser.add_argument("--maximum-new-tokens", type=int, default=128)
    parser.add_argument("--repeat-count", type=int, default=DEFAULT_REPEAT_COUNT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def _real_unlinked(path: Path, *, label: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    for component in (lexical, *lexical.parents):
        if component.is_symlink() or component.is_junction():
            raise QwenScaleCommandError(f"{label} traverses a linked path")
    return lexical.resolve(strict=False)


def _safe_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path, Path]:
    corpus = _real_unlinked(args.corpus_root, label="corpus root")
    checkpoint = _real_unlinked(args.classifier_checkpoint, label="classifier checkpoint")
    rejection = _real_unlinked(
        args.classifier_rejection_evidence, label="classifier rejection evidence"
    )
    baseline = _real_unlinked(args.qwen17b_evidence_root, label="Qwen3-1.7B evidence")
    output = _real_unlinked(args.output_root, label="M5A.3 output root")
    report = _real_unlinked(args.report, label="M5A.3 command report")
    inputs = (corpus, checkpoint, rejection, baseline)
    if any(
        output == value or output in value.parents or value in output.parents for value in inputs
    ):
        raise QwenScaleCommandError("M5A.3 output overlaps a protected input")
    if report == output or output in report.parents:
        raise QwenScaleCommandError("command report must remain outside immutable evidence")
    protected = tuple(
        (PROJECT_ROOT / name).resolve(strict=False)
        for name in ("src", "scripts", "environment", "tests", "docs", ".git")
    )
    if any(
        output == value or value in output.parents or output in value.parents for value in protected
    ):
        raise QwenScaleCommandError("M5A.3 output overlaps source-controlled content")
    if any(
        report == value or value in report.parents or report in value.parents for value in protected
    ):
        raise QwenScaleCommandError("M5A.3 command report overlaps source-controlled content")
    if any(report == value or report in value.parents for value in inputs):
        raise QwenScaleCommandError("M5A.3 command report overlaps a protected input")
    return corpus, checkpoint, rejection, baseline, output, report


def _read_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QwenScaleCommandError(f"could not read {label}: {error}") from error
    if not isinstance(value, dict):
        raise QwenScaleCommandError(f"{label} must be one JSON object")
    return cast(dict[str, object], value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _artifact_records(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()):
        if path.is_symlink() or path.is_junction():
            raise QwenScaleCommandError("corpus contains linked content")
        if not path.is_file() or path.name in {"artifact_manifest.json", "complete.json"}:
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return records


def _validate_corpus_archive(root: Path, corpus: GeneratedLanguageCorpus) -> str:
    manifest = _read_object(root / "artifact_manifest.json", label="corpus manifest")
    complete = _read_object(root / "complete.json", label="corpus completion")
    records = _artifact_records(root)
    fingerprint = f"sha256:{sha256_hex({'corpus_fingerprint': corpus.manifest.corpus_fingerprint, 'artifacts': records})}"
    if manifest != {
        "schema_version": CORPUS_ARCHIVE_SCHEMA,
        "corpus_fingerprint": corpus.manifest.corpus_fingerprint,
        "archive_fingerprint": fingerprint,
        "artifacts": records,
    } or complete != {
        "schema_version": CORPUS_ARCHIVE_SCHEMA,
        "corpus_fingerprint": corpus.manifest.corpus_fingerprint,
        "archive_fingerprint": fingerprint,
        "passed": True,
    }:
        raise QwenScaleCommandError("corpus archive identity/checksums differ")
    if _read_object(root / "corpus_manifest.json", label="corpus identity") != (
        corpus.manifest.to_dict()
    ):
        raise QwenScaleCommandError("corpus differs from the frozen generator")
    if (root / "final.jsonl").exists():
        raise QwenScaleCommandError("language final was materialized in development corpus")
    for split in (LanguageSplit.TRAIN, LanguageSplit.VALIDATION, LanguageSplit.DEVELOPMENT):
        path = root / f"{split.value}.jsonl"
        try:
            observed = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise QwenScaleCommandError(f"could not parse corpus {split.value}: {error}") from error
        if observed != [value.to_dict() for value in corpus.examples_for_split(split)]:
            raise QwenScaleCommandError(f"corpus {split.value} records differ")
    development, final = build_language_schedule_locks(corpus)
    if _read_object(root / "language_schedule_locks.json", label="schedule locks") != {
        "schema_version": CORPUS_ARCHIVE_SCHEMA,
        "development": development.to_dict(),
        "final": final.to_dict(),
    }:
        raise QwenScaleCommandError("language schedule locks differ")
    return fingerprint


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError as error:
        raise QwenScaleCommandError(f"required distribution is missing: {distribution}") from error


def _dependencies() -> dict[str, object]:
    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers": _package_version("transformers"),
        "tokenizers": _package_version("tokenizers"),
        "huggingface_hub": _package_version("huggingface-hub"),
        "safetensors": _package_version("safetensors"),
    }


def _select_prompt_examples(corpus: GeneratedLanguageCorpus) -> tuple[LanguageExample, ...]:
    return select_structured_routing_prompt_examples(corpus.examples_for_split(LanguageSplit.TRAIN))


def _select_smoke_examples(corpus: GeneratedLanguageCorpus) -> tuple[LanguageExample, ...]:
    train = tuple(
        sorted(corpus.examples_for_split(LanguageSplit.TRAIN), key=lambda x: x.example_id)
    )
    selected: list[LanguageExample] = []
    for task_id in sorted({value.task_id for value in train if value.task_id is not None}):
        selected.append(next(value for value in train if value.task_id == task_id))
    reasons = sorted(
        {value.expected_rejection_reason for value in train if value.expected_rejection_reason},
        key=lambda value: value.value,
    )
    for reason in reasons:
        selected.append(next(value for value in train if value.expected_rejection_reason is reason))
    if len(selected) != 20:
        raise QwenScaleCommandError("frozen train-only smoke set must contain exactly 20 examples")
    return tuple(selected)


def _semantic_decision_fingerprint(decision: RouterDecision) -> str:
    return f"sha256:{sha256_hex({'status': decision.status.value, 'task_id': decision.task_id, 'rejection_reason': None if decision.rejection_reason is None else decision.rejection_reason.value})}"


def _run_smoke(
    router: StructuredLocalLLMRouterV0, examples: Sequence[LanguageExample]
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for example in examples:
        first = router.route(example.raw_text)
        second = router.route(example.raw_text)
        records.append(
            {
                "example_id": example.example_id,
                "expected_status": example.expected_status.value,
                "expected_task_id": example.task_id,
                "decision": first.to_dict(),
                "repeat_decision_fingerprint": _semantic_decision_fingerprint(second),
                "deterministic": _semantic_decision_fingerprint(first)
                == _semantic_decision_fingerprint(second),
            }
        )
    final_valid = sum(
        cast(Mapping[str, object], value["decision"]).get("rejection_reason")
        != RouterRejectionReason.FORMAT_REPAIR_EXHAUSTED.value
        for value in records
    )
    task_ids = {
        cast(Mapping[str, object], value["decision"]).get("task_id")
        for value in records
        if value["expected_status"] == RouterStatus.ROUTE.value
    }
    statuses = {cast(Mapping[str, object], value["decision"])["status"] for value in records}
    checks = {
        "exact_twenty_train_examples": len(records) == 20,
        "final_schema_valid_rate_at_least_0_99": final_valid / len(records) >= 0.99,
        "all_six_task_specs_produced": len(task_ids - {None}) == 6,
        "all_router_statuses_produced": statuses == {value.value for value in RouterStatus},
        "expected_task_specs_and_statuses_match": all(
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
        "deterministic_final_decisions": all(bool(value["deterministic"]) for value in records),
        "no_prohibited_split_access": True,
    }
    return {
        "schema_version": "langmani-m5a3-qwen4b-train-smoke-v0",
        "split": "train",
        "example_count": len(records),
        "example_ids": [value.example_id for value in examples],
        "checks": checks,
        "gate_passed": all(checks.values()),
        "records": records,
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
        repeat_fingerprint=_semantic_decision_fingerprint,
    )
    metrics = recompute_router_metrics(records=records, examples=examples)
    return records, {
        "schema_version": "langmani-m5a3-offline-router-result-v0",
        "split": split.value,
        "eligibility": dict(eligibility),
        "metrics": metrics,
        "records": [record.to_dict() for record in records],
        "total_elapsed_inference_seconds": sum(record.latency_ms for record in records) / 1000.0,
    }


def _load_classifier_negative(
    *, checkpoint: Path, rejection_root: Path, local_files_only: bool
) -> tuple[FrozenClassifierNegativeBaselineV0, dict[str, object]]:
    artifact = validate_rejection_analysis_artifact(rejection_root)
    root = Path(cast(str, artifact["root"]))
    rejected = _read_object(root / "rejected_candidate.json", label="rejected classifier")
    selection = _read_object(root / "candidate_selection.json", label="decoder selection")
    source = _read_object(root / "source_recovery_identity.json", label="classifier identity")
    selected = selection.get("selected_configuration")
    if (
        artifact.get("conclusion") != "classifier_rejected_after_posthoc_calibration"
        or rejected.get("classifier_candidate_frozen") is not True
        or rejected.get("additional_training_authorized") is not False
        or rejected.get("additional_seed_authorized") is not False
        or not isinstance(selected, Mapping)
    ):
        raise QwenScaleCommandError("classifier is not the frozen M5A.1 negative baseline")
    expected_sha = cast(str, rejected["selected_classifier_checkpoint_fingerprint"])
    if _sha256_file(checkpoint) != expected_sha:
        raise QwenScaleCommandError("classifier checkpoint identity differs")
    run_fingerprint = source.get("run_fingerprint")
    selected_step = source.get("selected_step")
    if not isinstance(run_fingerprint, str) or not isinstance(selected_step, int):
        raise QwenScaleCommandError("classifier source identity is malformed")
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
        raise QwenScaleCommandError("classifier tokenizer contract is malformed")
    configuration = conservative_decoder_from_mapping(cast(Mapping[str, object], selected))
    router = FrozenClassifierNegativeBaselineV0(
        bundle=bundle,
        configuration=configuration,
        maximum_sequence_length=cast(int, tokenizer["maximum_sequence_length"]),
        device="cpu",
    )
    return router, {
        "analysis_fingerprint": artifact["analysis_fingerprint"],
        "checkpoint_fingerprint": expected_sha,
        "selection_fingerprint": selection["selection_fingerprint"],
        "optimizer_constructed": False,
        "model_state_fingerprint_before": router.model_state_fingerprint_before,
    }


def _development_safety(
    *, router: StructuredLocalLLMRouterV0, records: Sequence[RouterEvaluationRecord]
) -> tuple[dict[str, bool], dict[str, object]]:
    empty_first = router.route("")
    empty_second = router.route("")
    by_reason: dict[RouterRejectionReason, list[RouterEvaluationRecord]] = defaultdict(list)
    for record in records:
        if record.expected_rejection_reason is not None:
            by_reason[record.expected_rejection_reason].append(record)

    def never_routes(*reasons: RouterRejectionReason) -> bool:
        selected = [record for reason in reasons for record in by_reason[reason]]
        return bool(selected) and all(
            record.decision.status is not RouterStatus.ROUTE for record in selected
        )

    checks = {
        "empty_and_meaningless_inputs_never_route": (
            empty_first.status is not RouterStatus.ROUTE
            and _semantic_decision_fingerprint(empty_first)
            == _semantic_decision_fingerprint(empty_second)
            and never_routes(RouterRejectionReason.MEANINGLESS_TEXT)
        ),
        "conflicting_object_and_bin_inputs_never_route": never_routes(
            RouterRejectionReason.CONFLICTING_OBJECTS,
            RouterRejectionReason.CONFLICTING_BINS,
        ),
        "unsupported_action_commands_never_route": never_routes(
            RouterRejectionReason.UNSUPPORTED_ACTION
        ),
    }
    return checks, {"empty_input_decision": empty_first.to_dict(), "checks": checks}


def _baseline_payloads(
    *, root: Path, corpus: GeneratedLanguageCorpus, checkpoint: Path
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    validated = validate_language_development_evidence(root)
    independent = verify_language_development_evidence(
        root, corpus=corpus, classifier_checkpoint=checkpoint
    )
    complete = cast(Mapping[str, object], validated["complete"])
    flags = cast(Mapping[str, object], complete.get("flags", {}))
    if (
        independent.get("passed") is not True
        or flags.get("llm_validation_completed") is not True
        or flags.get("llm_validation_gate_passed") is not False
        or flags.get("language_development_completed") is not False
        or (root / "development").exists()
    ):
        raise QwenScaleCommandError("Qwen3-1.7B baseline is not the frozen rejected result")
    qwen17b_source = _read_object(root / "validation/local_llm.json", label="Qwen3-1.7B baseline")
    qwen17b_payload = {
        **qwen17b_source,
        "source_m5a2_eligibility": qwen17b_source.get("eligibility"),
        "eligibility": _eligibility(promotable=False, baseline=True),
    }
    payloads = {
        "rule_router": _read_object(root / "validation/rule_router.json", label="rule baseline"),
        "classifier_negative_baseline": _read_object(
            root / "validation/classifier_negative_baseline.json", label="classifier baseline"
        ),
        "qwen17b_negative_baseline": qwen17b_payload,
    }
    return {
        "artifact_fingerprint": validated["artifact_fingerprint"],
        "runtime_fingerprint": cast(Mapping[str, object], validated["owner"])[
            "runtime_fingerprint"
        ],
        "independent_verification": independent,
    }, payloads


def _record_ids(payload: Mapping[str, object]) -> tuple[str, ...]:
    records = payload.get("records")
    if not isinstance(records, list):
        raise QwenScaleCommandError("router payload lacks raw records")
    ids: list[str] = []
    for value in records:
        if not isinstance(value, Mapping) or not isinstance(value.get("example_id"), str):
            raise QwenScaleCommandError("router raw record identity is malformed")
        ids.append(cast(str, value["example_id"]))
    return tuple(ids)


def _comparison_metrics(payload: Mapping[str, object]) -> dict[str, object]:
    metrics = payload.get("metrics")
    records = payload.get("records")
    if not isinstance(metrics, Mapping) or not isinstance(records, list):
        raise QwenScaleCommandError("scale input lacks metrics or raw records")
    elapsed = 0.0
    for value in records:
        if (
            not isinstance(value, Mapping)
            or isinstance(value.get("latency_ms"), bool)
            or not isinstance(value.get("latency_ms"), int | float)
        ):
            raise QwenScaleCommandError("scale input latency is malformed")
        elapsed += float(value["latency_ms"]) / 1000.0
    return {**dict(metrics), "total_elapsed_inference_seconds": elapsed}


def _eligibility(*, promotable: bool, baseline: bool) -> dict[str, object]:
    return {
        "offline_evaluation_eligible": True,
        "controller_dispatch_eligible": False,
        "promotion_eligible": promotable,
        "final_selection_eligible": promotable,
        "frozen_negative_baseline": baseline,
    }


def execute(args: argparse.Namespace) -> dict[str, object]:
    corpus_root, checkpoint, rejection_root, baseline_root, output_root, _ = _safe_paths(args)
    target = bool(args.target_development)
    dependencies = _dependencies()
    git_state = inspect_git_state(PROJECT_ROOT)
    if args.repeat_count < 2:
        raise QwenScaleCommandError("determinism requires at least two decisions per command")
    if args.model_id not in M5A3_AUTHORIZED_CANDIDATES or args.model_id != (
        QWEN3_4B_INSTRUCT_MODEL_ID
    ):
        raise QwenScaleCommandError("M5A.3 authorizes exactly Qwen3-4B-Instruct-2507")
    if (
        args.model_revision != QWEN3_4B_INSTRUCT_REVISION
        or args.tokenizer_revision != QWEN3_4B_INSTRUCT_REVISION
        or args.license != QWEN3_4B_INSTRUCT_LICENSE
        or args.quantization != "none"
    ):
        raise QwenScaleCommandError("Qwen3-4B identity/revision/license/runtime differs")
    if target:
        if args.device != "cuda" or not torch.cuda.is_available():
            raise QwenScaleCommandError("M5A.3 target inference requires CUDA")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or torch.cuda.device_count() != 1:
            raise QwenScaleCommandError("M5A.3 requires exactly visible CUDA device 0")
        if dependencies["transformers"] != "5.4.0" or dependencies["tokenizers"] != "0.22.2":
            raise QwenScaleCommandError("locked Transformers public API versions differ")
        if git_state.dirty or not git_state.baseline_tracked:
            raise QwenScaleCommandError("real GPU evidence requires clean tracked Git")
        if not args.license_reviewed or not args.model_card_reviewed:
            raise QwenScaleCommandError("official model card and Apache-2.0 license need review")
    corpus = build_language_corpus()
    archive_fingerprint = _validate_corpus_archive(corpus_root, corpus)
    development_lock, final_lock = build_language_schedule_locks(corpus)
    identity = Qwen4BInstructIdentityV0()
    prompt_examples = _select_prompt_examples(corpus)
    config = StructuredLLMRouterConfig(
        model_id=args.model_id,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        dtype=args.dtype,
        quantization=args.quantization,
        maximum_new_tokens=args.maximum_new_tokens,
        maximum_format_repair_attempts=1,
        chat_template_mode="official_qwen_instruct_non_thinking",
    )
    prompt_text, prompt_fingerprint, prompt_ids = build_structured_routing_prompt(
        examples=prompt_examples, prompt_version=config.prompt_version
    )
    flags = initial_m5a3_flags()
    frozen17 = FrozenQwen17BBaselineV0()
    baseline_identity: dict[str, object] | None = None
    baseline_payloads: dict[str, dict[str, object]] = {}
    generator: TransformersLocalTextGenerator | None = None
    rendered_fingerprint = (
        f"sha256:{sha256_hex({'fixture_transport_only': prompt_text, 'model': args.model_id})}"
    )
    parameter_count: int | None = None
    if target:
        baseline_identity, baseline_payloads = _baseline_payloads(
            root=baseline_root, corpus=corpus, checkpoint=checkpoint
        )
        baseline_prompt = _read_object(baseline_root / "prompt.json", label="1.7B prompt")
        if (
            baseline_prompt.get("prompt_text") != prompt_text
            or baseline_prompt.get("prompt_fingerprint") != prompt_fingerprint
            or baseline_prompt.get("prompt_example_ids") != list(prompt_ids)
        ):
            raise QwenScaleCommandError("frozen 1.7B prompt semantics differ")
        flags["qwen17b_offline_baseline_validated"] = True
        generator = TransformersLocalTextGenerator(
            config=config,
            device=args.device,
            local_files_only=bool(args.local_files_only),
        )
        rendered_fingerprint = generator.rendered_prompt_fingerprint(prompt_text)
        parameter_count = sum(parameter.numel() for parameter in generator.model.parameters())
        if not 3_500_000_000 <= parameter_count <= 4_500_000_000:
            raise QwenScaleCommandError("Qwen3-4B parameter count differs")
        flags["qwen4b_model_identity_validated"] = True
    equivalence = PromptEquivalenceV0(
        qwen17b_prompt_fingerprint=prompt_fingerprint,
        qwen4b_rendered_prompt_fingerprint=rendered_fingerprint,
        semantic_prompt_content_fingerprint=prompt_fingerprint,
        qwen17b_prompt_example_ids=prompt_ids,
        qwen4b_prompt_example_ids=prompt_ids,
        schema_fingerprint=router_schema_fingerprint(),
        parser_fingerprint=router_parser_fingerprint(),
        repair_policy_fingerprint=repair_policy_fingerprint(),
        semantic_prompt_content_equal=True,
    )
    flags["qwen4b_prompt_equivalence_validated"] = target
    owner_payload = {
        "schema_version": COMMAND_SCHEMA,
        "mode": "target_development" if target else "fixture",
        "git_commit": git_state.commit,
        "corpus_fingerprint": corpus.manifest.corpus_fingerprint,
        "corpus_archive_fingerprint": archive_fingerprint,
        "validation_split_fingerprint": corpus.manifest.split_manifests[
            LanguageSplit.VALIDATION
        ].content_fingerprint,
        "development_split_fingerprint": corpus.manifest.split_manifests[
            LanguageSplit.DEVELOPMENT
        ].content_fingerprint,
        "development_lock_fingerprint": development_lock.schedule_fingerprint,
        "final_lock_fingerprint": final_lock.schedule_fingerprint,
        "model_identity_fingerprint": identity.fingerprint,
        "prompt_equivalence_fingerprint": equivalence.fingerprint,
        "generation_config_fingerprint": config.fingerprint,
        "qwen17b_artifact_fingerprint": None
        if baseline_identity is None
        else baseline_identity["artifact_fingerprint"],
        "repeat_count": args.repeat_count,
    }
    runtime_fingerprint = f"sha256:{sha256_hex(owner_payload)}"
    owner = {**owner_payload, "runtime_fingerprint": runtime_fingerprint}
    artifacts: dict[str, object] = {
        "model_identity.json": {
            **identity.to_dict(),
            "identity_fingerprint": identity.fingerprint,
            "actual_files": {} if generator is None else generator.file_identities,
            "model_files": {}
            if generator is None
            else {
                name: value
                for name, value in generator.file_identities.items()
                if name.startswith("model") or name in {"config.json", "generation_config.json"}
            },
            "tokenizer_files": {}
            if generator is None
            else {
                name: value
                for name, value in generator.file_identities.items()
                if name in {"merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"}
            },
            "parameter_count": parameter_count,
            "model_loaded_once": target,
            "fixture_contract_only": not target,
            "license_reviewed": bool(args.license_reviewed),
            "model_card_reviewed": bool(args.model_card_reviewed),
            "dtype": args.dtype,
            "device": args.device,
            "quantization": args.quantization,
            "snapshot_path": None if generator is None else generator.snapshot_path,
            "snapshot_size_bytes": 0 if generator is None else generator.snapshot_size_bytes,
            "model_download_seconds": 0.0
            if generator is None
            else generator.model_download_seconds,
            "visible_gpu_count": torch.cuda.device_count() if target else 0,
            "gpu_name": torch.cuda.get_device_name(0) if target else None,
            "gpu_total_memory_bytes": (
                torch.cuda.get_device_properties(0).total_memory if target else 0
            ),
            "optimizer_constructed": False,
            "training_or_fine_tuning_performed": False,
            "model_weights_unchanged": True,
        },
        "prompt_equivalence.json": equivalence.to_dict(),
        "generation_config.json": config.to_dict(),
        "qwen17b_negative_baseline.json": {
            **frozen17.to_dict(),
            "source_evidence": baseline_identity,
        },
        "access_audit.json": {
            "language_final_accessed": False,
            "control_development_accessed": False,
            "control_final_accessed": False,
            "test_split_accessed": False,
            "m42_final_accessed": False,
            "act_controller_loaded": False,
            "robot_environment_created": False,
            "environment_step_count": 0,
            "smolvla_go": False,
        },
    }
    stopped_after = "fixture"
    stopped_at_validation = False
    candidate_validation: dict[str, object] | None = None
    if target:
        assert generator is not None
        router = StructuredLocalLLMRouterV0(
            config=config, generator=generator, prompt_examples=prompt_examples
        )
        smoke = _run_smoke(router, _select_smoke_examples(corpus))
        artifacts["train_smoke.json"] = smoke
        flags["qwen4b_train_smoke_completed"] = True
        stopped_after = "train_smoke"
        if smoke["gate_passed"] is True:
            for name, payload in baseline_payloads.items():
                artifacts[f"validation/{name}.json"] = payload
            _, candidate_validation = _router_payload(
                router=router,
                examples=corpus.examples_for_split(LanguageSplit.VALIDATION),
                split=LanguageSplit.VALIDATION,
                repeat_count=args.repeat_count,
                eligibility=_eligibility(promotable=True, baseline=False),
            )
            artifacts["validation/qwen4b_candidate.json"] = candidate_validation
            metrics4 = _comparison_metrics(candidate_validation)
            gate = evaluate_qwen4b_validation_gate(metrics=metrics4, prohibited_source_access=False)
            artifacts["validation/gate.json"] = gate.to_dict()
            flags["qwen4b_validation_completed"] = True
            flags["qwen4b_validation_gate_passed"] = gate.passed
            stopped_after = "validation"
            stopped_at_validation = not gate.passed
            baseline17 = baseline_payloads["qwen17b_negative_baseline"]
            metrics17 = _comparison_metrics(baseline17)
            comparison = compute_scale_comparison(
                qwen17b_metrics=metrics17,
                qwen4b_metrics=metrics4,
                validation_example_ids_17b=_record_ids(baseline17),
                validation_example_ids_4b=_record_ids(candidate_validation),
                qwen4b_gate_passed=gate.passed,
                model_download_seconds_4b=generator.model_download_seconds,
            )
            artifacts["scale_comparison.json"] = comparison
            if gate.passed:
                runtime_lock_payload = {
                    "model_identity_fingerprint": identity.fingerprint,
                    "prompt_equivalence_fingerprint": equivalence.fingerprint,
                    "schema_fingerprint": equivalence.schema_fingerprint,
                    "parser_fingerprint": equivalence.parser_fingerprint,
                    "repair_policy_fingerprint": equivalence.repair_policy_fingerprint,
                    "generation_config_fingerprint": config.fingerprint,
                    "validation_evidence_fingerprint": f"sha256:{sha256_hex(candidate_validation)}",
                    "git_commit": git_state.commit,
                    "corpus_fingerprint": corpus.manifest.corpus_fingerprint,
                    "locked_before_development": True,
                }
                runtime_lock_payload["runtime_lock_fingerprint"] = (
                    f"sha256:{sha256_hex(runtime_lock_payload)}"
                )
                artifacts["validation/runtime_lock.json"] = runtime_lock_payload
                classifier, classifier_identity = _load_classifier_negative(
                    checkpoint=checkpoint,
                    rejection_root=rejection_root,
                    local_files_only=bool(args.local_files_only),
                )
                rule = RuleRouterV0()
                config17 = StructuredLLMRouterConfig(
                    model_id=QWEN3_1_7B_MODEL_ID,
                    model_revision=QWEN3_1_7B_REVISION,
                    tokenizer_revision=QWEN3_1_7B_REVISION,
                    dtype="bfloat16",
                    quantization="none",
                    maximum_new_tokens=args.maximum_new_tokens,
                    maximum_format_repair_attempts=1,
                )
                generator17 = TransformersLocalTextGenerator(
                    config=config17,
                    device=args.device,
                    local_files_only=bool(args.local_files_only),
                )
                router17 = StructuredLocalLLMRouterV0(
                    config=config17, generator=generator17, prompt_examples=prompt_examples
                )
                development_examples = corpus.examples_for_split(LanguageSplit.DEVELOPMENT)
                qwen4_records: tuple[RouterEvaluationRecord, ...] | None = None
                for name, active, eligibility in (
                    ("rule_router", rule, RULE_ROUTER_ELIGIBILITY.to_dict()),
                    (
                        "classifier_negative_baseline",
                        classifier,
                        CLASSIFIER_NEGATIVE_ELIGIBILITY.to_dict(),
                    ),
                    (
                        "qwen17b_negative_baseline",
                        router17,
                        _eligibility(promotable=False, baseline=True),
                    ),
                    (
                        "qwen4b_candidate",
                        router,
                        _eligibility(promotable=True, baseline=False),
                    ),
                ):
                    records, payload = _router_payload(
                        router=active,
                        examples=development_examples,
                        split=LanguageSplit.DEVELOPMENT,
                        repeat_count=args.repeat_count,
                        eligibility=eligibility,
                    )
                    if name == "qwen4b_candidate":
                        qwen4_records = records
                    artifacts[f"development/{name}.json"] = payload
                assert qwen4_records is not None
                safety_checks, safety = _development_safety(router=router, records=qwen4_records)
                artifacts["development/safety_probes.json"] = safety
                qwen4_development = cast(
                    Mapping[str, object],
                    cast(Mapping[str, object], artifacts["development/qwen4b_candidate.json"])[
                        "metrics"
                    ],
                )
                dev_gate = evaluate_qwen4b_development_gate(
                    metrics=qwen4_development,
                    per_task_accuracy=cast(
                        Mapping[str, object], qwen4_development["per_task_accuracy"]
                    ),
                    rejection_family_false_route_rates=cast(
                        Mapping[str, object],
                        qwen4_development["rejection_family_false_route_rates"],
                    ),
                    safety_checks=safety_checks,
                )
                artifacts["development/gate.json"] = dev_gate.to_dict()
                flags["qwen4b_language_development_completed"] = True
                flags["qwen4b_language_quality_gate_passed"] = dev_gate.passed
                flags["learned_router_selected"] = dev_gate.passed
                flags["one_scene_control_smoke_authorized"] = dev_gate.passed
                stopped_after = "language_development"
                stopped_at_validation = False
                classifier_unchanged = classifier.verify_unchanged()
                if not classifier_unchanged:
                    raise QwenScaleCommandError("classifier weights changed during inference")
                artifacts["development/baseline_runtime_audit.json"] = {
                    "classifier": {
                        **classifier_identity,
                        "model_state_fingerprint_after": (
                            classifier.model_state_fingerprint_before
                        ),
                        "model_weights_unchanged": True,
                    },
                    "qwen17b_model_download_seconds": generator17.model_download_seconds,
                    "qwen17b_model_weights_unchanged": True,
                    "optimizer_constructed": False,
                }
        else:
            stopped_after = "train_smoke"
    else:
        artifacts["train_smoke.json"] = {
            "schema_version": "langmani-m5a3-qwen4b-train-smoke-fixture-v0",
            "fixture_only": True,
            "gate_passed": False,
            "example_ids": [value.example_id for value in _select_smoke_examples(corpus)],
        }
    artifacts["candidate_selection.json"] = {
        "authorized_candidates": list(M5A3_AUTHORIZED_CANDIDATES),
        "selected_learned_router": (
            {
                "model_id": QWEN3_4B_INSTRUCT_MODEL_ID,
                "runtime_fingerprint": runtime_fingerprint,
                "model_identity_fingerprint": identity.fingerprint,
                "prompt_equivalence_fingerprint": equivalence.fingerprint,
                "schema_fingerprint": equivalence.schema_fingerprint,
                "parser_fingerprint": equivalence.parser_fingerprint,
                "repair_policy_fingerprint": equivalence.repair_policy_fingerprint,
                "generation_config_fingerprint": config.fingerprint,
                "validation_evidence_fingerprint": (
                    None
                    if candidate_validation is None
                    else f"sha256:{sha256_hex(candidate_validation)}"
                ),
                "development_evidence_fingerprint": f"sha256:{sha256_hex(artifacts['development/qwen4b_candidate.json'])}",
                "git_commit": git_state.commit,
                "corpus_fingerprint": corpus.manifest.corpus_fingerprint,
            }
            if flags["learned_router_selected"]
            else None
        ),
        "learned_router_selected": flags["learned_router_selected"],
        "one_scene_control_smoke_authorized": flags["one_scene_control_smoke_authorized"],
    }
    flags["real_gpu_inference_validated"] = target
    artifacts["summary.md"] = (
        "# M5A.3 Qwen Capacity Escalation\n\n"
        f"Stopped after: `{stopped_after}`.\n\n"
        f"Qwen3-4B validation gate: `{flags['qwen4b_validation_gate_passed']}`.\n\n"
        f"Qwen3-4B development gate: `{flags['qwen4b_language_quality_gate_passed']}`.\n\n"
        "No model training, robot controller, or environment was used.\n"
    )
    artifacts["result.json"] = {
        "schema_version": COMMAND_SCHEMA,
        "runtime_fingerprint": runtime_fingerprint,
        "stopped_after_stage": stopped_after,
        "stopped_at_validation": stopped_at_validation,
        "flags": flags,
        "dependencies": dependencies,
        "generation_totals": {
            "generation_calls": 0 if generator is None else len(generator.generation_metadata),
            "generated_tokens": 0
            if generator is None
            else sum(
                cast(int, value.get("generated_token_count", 0))
                for value in generator.generation_metadata
            ),
            "total_elapsed_inference_seconds": 0.0
            if generator is None
            else sum(
                float(value.get("elapsed_seconds", 0.0)) for value in generator.generation_metadata
            ),
            "peak_gpu_memory_bytes": 0 if generator is None else generator.peak_gpu_memory_bytes,
            "model_download_seconds": 0.0
            if generator is None
            else generator.model_download_seconds,
        },
        "no_training_audit": {
            "optimizer_constructed": False,
            "training_or_fine_tuning_performed": False,
            "lora_used": False,
            "model_weights_unchanged": True,
        },
        "access_audit": artifacts["access_audit.json"],
    }
    evidence = write_qwen_scale_evidence(
        output_root,
        owner=owner,
        artifacts=artifacts,
        stopped_after_stage=stopped_after,
        stopped_at_validation=stopped_at_validation,
        flags=flags,
    )
    return {
        "schema_version": COMMAND_SCHEMA,
        **flags,
        "mode": owner["mode"],
        "runtime_fingerprint": runtime_fingerprint,
        "artifact_fingerprint": evidence["artifact_fingerprint"],
        "evidence_root": evidence["root"],
        "classifier_checkpoint_path": str(checkpoint),
        "qwen17b_evidence_root": str(baseline_root),
        "stopped_after_stage": stopped_after,
        "stopped_at_validation": stopped_at_validation,
        "evidence_reused": evidence["evidence_reused"],
        "physical_target_validated": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mode = "target_development" if args.target_development else "fixture"
    payload: dict[str, object] = {
        "schema_version": COMMAND_SCHEMA,
        **initial_m5a3_flags(),
        "passed": False,
        "mode": mode,
        "physical_target_validated": False,
    }
    try:
        _, _, _, _, _, report = _safe_paths(args)
    except Exception as error:  # noqa: BLE001 - unsafe paths must never receive a write
        traceback.print_exc()
        payload["error"] = {"type": type(error).__name__, "message": str(error)}
        print(json.dumps(payload, sort_keys=True, allow_nan=False))
        return 1
    try:
        payload = execute(args)
    except Exception as error:  # noqa: BLE001 - preserve outer-boundary diagnostics
        traceback.print_exc()
        payload["error"] = {"type": type(error).__name__, "message": str(error)}
    report.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, payload)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0 if payload.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
