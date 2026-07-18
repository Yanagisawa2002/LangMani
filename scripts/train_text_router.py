"""Run one explicit stage of the single-seed M5A text-router pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast

import torch

from langmani.datasets.identity import sha256_hex
from langmani.language.corpus import build_language_corpus
from langmani.language.router_types import LanguageExample, LanguageSplit, RouterStatus
from langmani.language.stage_protocol import (
    M5A_CLASSIFIER_TRAINING_SEEDS,
    ClassifierStageTrainingConfig,
    stage_protocol_manifest,
)
from langmani.language.text_classifier import (
    BIN_LABELS,
    OBJECT_LABELS,
    STATUS_LABELS,
    TEXT_CLASSIFIER_MODEL_ID,
    TEXT_CLASSIFIER_MODEL_REVISION,
    TEXT_CLASSIFIER_TOKENIZER_REVISION,
    FactorizedTextClassifierV0,
    classifier_manifest,
    load_factorized_text_classifier,
)
from langmani.language.text_training import (
    FactorizedTextBatch,
    TextRouterCalibrationSelection,
    TextTrainingConfig,
    calibrate_and_select_text_router,
    classifier_stage_metrics,
    collect_factorized_tiny_train_outputs,
    collect_factorized_validation_outputs,
    run_bounded_factorized_text_training,
    run_staged_factorized_text_training,
    stage_and_promote_text_classifier,
    text_classifier_run_fingerprint,
    train_factorized_text_step,
)
from langmani.policies.act_runtime import atomic_write_json, inspect_git_state

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "models" / "text-router"
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "diagnostics" / "m5a" / "train_text_router.json"
MODEL_ID = TEXT_CLASSIFIER_MODEL_ID
MODEL_REVISION = TEXT_CLASSIFIER_MODEL_REVISION
TOKENIZER_REVISION = TEXT_CLASSIFIER_TOKENIZER_REVISION
TARGET_TRAINING_CONFIG = TextTrainingConfig(
    maximum_steps=145,
    checkpoint_steps=(29, 58, 87, 116, 145),
    learning_rate=2e-5,
    seed=0,
    weight_decay=0.01,
    maximum_gradient_norm=1.0,
    early_stopping_patience=1,
    encoder_trainable=True,
)
TARGET_STAGE_TRAINING_CONFIG = ClassifierStageTrainingConfig(train_example_count=900)
TINY_OVERFIT_MAXIMUM_STEPS = 200
TINY_ROUTEABLE_EXAMPLES_PER_TASK = 4
TINY_REJECTED_EXAMPLES_PER_STATUS = 8
TARGET_BATCH_SIZE = 32
TARGET_MAXIMUM_SEQUENCE_LENGTH = 64
TARGET_DROPOUT = 0.1
THRESHOLD_CANDIDATES = tuple(index / 100.0 for index in range(0, 101, 5))
MAXIMUM_FALSE_ROUTE_RATE = 0.03
TEMPERATURE_ITERATIONS = 64
COMMAND_SCHEMA = "langmani-m5a-train-text-router-command-v2"
RUN_EVIDENCE_SCHEMA = "langmani-m5a-text-router-run-evidence-v2"

_PROTECTED_SOURCE_ROOTS = tuple(
    PROJECT_ROOT / name for name in ("src", "scripts", "environment", "tests", "docs", ".git")
)
_PROTECTED_REPOSITORY_FILES = tuple(
    PROJECT_ROOT / name
    for name in (".gitignore", "AGENTS.md", "PLAN.md", "README.md", "pyproject.toml")
)
_PROTECTED_GENERATED_ROOTS = (
    PROJECT_ROOT / "outputs" / "datasets",
    PROJECT_ROOT / "outputs" / "models" / "act",
    PROJECT_ROOT / "outputs" / "models" / "act-task-token",
    PROJECT_ROOT / "outputs" / "models" / "act-factor-film",
    *(
        PROJECT_ROOT / "outputs" / "diagnostics" / name
        for name in ("m0", "m1", "m2", "m3a", "m3b", "m4", "m42", "m43")
    ),
)
_FORBIDDEN_PATH_IDENTITIES = (
    "m3b_test",
    "fresh_seed",
    "m42_final_v0",
    "m5a_language_final",
    "m5a_control_final",
    "smolvla",
)


class TextRouterCommandError(RuntimeError):
    """Raised when the M5A classifier command cannot preserve its contract."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--fixture", action="store_true")
    modes.add_argument("--tiny-overfit", action="store_true")
    modes.add_argument("--target-pilot", action="store_true")
    modes.add_argument("--target-resume", action="store_true")
    modes.add_argument(
        "--target-development",
        action="store_true",
        help="Retired compatibility flag; use --target-pilot then --target-resume.",
    )
    parser.add_argument(
        "--clean-staging",
        action="store_true",
        help="Clean only a matching incomplete classifier staging directory.",
    )
    return parser.parse_args(argv)


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _resolved_unlinked(path: Path, *, label: str) -> Path:
    lexical = _lexical_absolute(path)
    for component in (lexical, *lexical.parents):
        if component.is_symlink() or component.is_junction():
            raise TextRouterCommandError(f"{label} traverses a symlink or junction: {component}")
    return lexical.resolve(strict=False)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _overlaps(left: Path, right: Path) -> bool:
    return _is_within(left, right) or _is_within(right, left)


def _reject_forbidden_path_identity(path: Path, *, label: str) -> None:
    lowered = tuple(part.lower() for part in path.parts)
    if any(fragment in part for part in lowered for fragment in _FORBIDDEN_PATH_IDENTITIES):
        raise TextRouterCommandError(f"{label} names a prohibited test, final, or SmolVLA source")


def _safe_paths(output_root: Path, report: Path) -> tuple[Path, Path]:
    output = _resolved_unlinked(output_root, label="text-router output")
    report_path = _resolved_unlinked(report, label="text-router command report")
    _reject_forbidden_path_identity(output, label="text-router output")
    _reject_forbidden_path_identity(report_path, label="text-router command report")
    protected = tuple(
        _resolved_unlinked(path, label="protected repository content")
        for path in (
            *_PROTECTED_SOURCE_ROOTS,
            *_PROTECTED_REPOSITORY_FILES,
            *_PROTECTED_GENERATED_ROOTS,
        )
    )
    if any(_overlaps(output, path) for path in protected):
        raise TextRouterCommandError(
            "text-router output cannot overlap source, Git, datasets, or historical models"
        )
    if any(_overlaps(report_path, path) for path in (*protected, output)):
        raise TextRouterCommandError(
            "text-router report cannot overlap source, Git, protected evidence, or model output"
        )
    if output.exists() and (not output.is_dir() or output.is_symlink() or output.is_junction()):
        raise TextRouterCommandError("text-router output must be a real directory when present")
    if report_path.exists() and (
        not report_path.is_file() or report_path.is_symlink() or report_path.is_junction()
    ):
        raise TextRouterCommandError("text-router report must be a real file when present")
    return output, report_path


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError as error:
        raise TextRouterCommandError(f"required distribution is missing: {distribution}") from error


def _dependency_versions() -> dict[str, str | None]:
    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers": _package_version("transformers"),
        "tokenizers": _package_version("tokenizers"),
        "huggingface_hub": _package_version("huggingface-hub"),
        "safetensors": _package_version("safetensors"),
    }


def _split_fingerprints(corpus_manifest: Mapping[str, object]) -> dict[str, str]:
    raw = corpus_manifest.get("split_manifests")
    if not isinstance(raw, Mapping):
        raise TextRouterCommandError("corpus manifest does not contain split manifests")
    result: dict[str, str] = {}
    for split in LanguageSplit:
        value = raw.get(split.value)
        if not isinstance(value, Mapping):
            raise TextRouterCommandError(f"corpus manifest is missing {split.value} split")
        fingerprint = value.get("content_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint.startswith("sha256:"):
            raise TextRouterCommandError(f"{split.value} split fingerprint is malformed")
        result[split.value] = fingerprint
    return result


def _label_indices(example: LanguageExample) -> tuple[int, int, int]:
    try:
        status = STATUS_LABELS.index(example.expected_status.value)
    except ValueError as error:
        raise TextRouterCommandError("language example has an unsupported status label") from error
    if example.expected_status is not RouterStatus.ROUTE:
        if example.expected_task_spec is not None:
            raise TextRouterCommandError("rejected training example contains an executable task")
        return status, -1, -1
    task_spec = example.expected_task_spec
    if task_spec is None:
        raise TextRouterCommandError("routeable training example is missing its TaskSpec")
    try:
        object_index = OBJECT_LABELS.index(task_spec.target_object_id)
        bin_index = BIN_LABELS.index(task_spec.target_bin_id)
    except ValueError as error:
        raise TextRouterCommandError(
            "routeable example contains an unsupported TaskSpec"
        ) from error
    if object_index >= 3 or bin_index >= 2:
        raise TextRouterCommandError("routeable example cannot use the none label")
    return status, object_index, bin_index


def _tokenize_batch(
    *,
    tokenizer: object,
    examples: Sequence[LanguageExample],
    split: LanguageSplit,
    maximum_sequence_length: int,
) -> FactorizedTextBatch:
    if split not in {LanguageSplit.TRAIN, LanguageSplit.VALIDATION}:
        raise TextRouterCommandError("classifier batches may materialize train or validation only")
    if not examples or any(example.split is not split for example in examples):
        raise TextRouterCommandError("classifier batch examples do not match their declared split")
    if not callable(tokenizer):
        raise TextRouterCommandError("classifier tokenizer is not callable")
    encoded = tokenizer(
        [example.raw_text for example in examples],
        padding=True,
        truncation=True,
        max_length=maximum_sequence_length,
        return_tensors="pt",
    )
    if not isinstance(encoded, Mapping):
        raise TextRouterCommandError("classifier tokenizer output must be a mapping")
    input_ids = encoded.get("input_ids")
    attention_mask = encoded.get("attention_mask")
    if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
        raise TextRouterCommandError("tokenizer must return input_ids and attention_mask tensors")
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise TextRouterCommandError("tokenizer tensors must have equal [batch,sequence] shape")
    labels = tuple(_label_indices(example) for example in examples)
    return FactorizedTextBatch(
        input_ids=input_ids.to(dtype=torch.long),
        attention_mask=attention_mask.to(dtype=torch.long),
        status_labels=torch.tensor([value[0] for value in labels], dtype=torch.long),
        object_labels=torch.tensor([value[1] for value in labels], dtype=torch.long),
        bin_labels=torch.tensor([value[2] for value in labels], dtype=torch.long),
        split=split.value,
    )


def _factorized_batches(
    *,
    tokenizer: object,
    examples: Sequence[LanguageExample],
    split: LanguageSplit,
    batch_size: int,
    maximum_sequence_length: int,
    seed: int,
) -> tuple[FactorizedTextBatch, ...]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise TextRouterCommandError("batch_size must be a positive integer")
    values = tuple(examples)
    if split is LanguageSplit.TRAIN:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        order = torch.randperm(len(values), generator=generator).tolist()
        values = tuple(values[index] for index in order)
    elif split is not LanguageSplit.VALIDATION:
        raise TextRouterCommandError("classifier data loader may use train or validation only")
    return tuple(
        _tokenize_batch(
            tokenizer=tokenizer,
            examples=values[start : start + batch_size],
            split=split,
            maximum_sequence_length=maximum_sequence_length,
        )
        for start in range(0, len(values), batch_size)
    )


def _model_state_fingerprint(model: FactorizedTextClassifierV0) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        header = json.dumps(
            {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(value.numpy().tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def _target_model_and_tokenizer() -> tuple[FactorizedTextClassifierV0, object]:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise TextRouterCommandError("Transformers 5.4.0 is required") from error
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=TOKENIZER_REVISION,
        trust_remote_code=False,
        use_fast=True,
    )
    model = FactorizedTextClassifierV0.from_pretrained_encoder(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        dropout=TARGET_DROPOUT,
    )
    return model, tokenizer


def _fixture_model_and_tokenizer() -> tuple[
    FactorizedTextClassifierV0, object, tempfile.TemporaryDirectory[str]
]:
    try:
        from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizerFast
    except ImportError as error:
        raise TextRouterCommandError(
            "Transformers is required for the classifier fixture"
        ) from error
    temporary = tempfile.TemporaryDirectory(prefix="langmani-m5a-tokenizer-")
    vocabulary = Path(temporary.name) / "vocab.txt"
    vocabulary.write_text(
        "\n".join(
            (
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
                "pick",
                "move",
                "place",
                "red",
                "green",
                "blue",
                "cube",
                "left",
                "right",
                "bin",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    tokenizer = DistilBertTokenizerFast(vocab_file=str(vocabulary), do_lower_case=True)
    encoder = DistilBertModel(
        DistilBertConfig(
            vocab_size=len(tokenizer),
            max_position_embeddings=64,
            n_layers=1,
            n_heads=2,
            dim=16,
            hidden_dim=32,
            dropout=0.0,
            attention_dropout=0.0,
        )
    )
    return FactorizedTextClassifierV0(encoder, dropout=0.0), tokenizer, temporary


def _calibration_payload(selection: TextRouterCalibrationSelection) -> dict[str, object]:
    payload = selection.to_dict()
    if set(payload) != {"temperature", "threshold"}:
        raise TextRouterCommandError("calibration selection fields differ from the exact schema")
    return payload


def _run_evidence(
    *,
    mode: str,
    corpus_manifest: Mapping[str, object],
    git_state: object,
    dependencies: Mapping[str, str | None],
    training_config: TextTrainingConfig,
    batch_size: int,
    maximum_sequence_length: int,
    training_result: Mapping[str, object],
    selection: TextRouterCalibrationSelection,
    model_id: str,
    model_revision: str,
    tokenizer_revision: str,
    training_device: str,
    train_examples: int,
    validation_examples: int,
) -> dict[str, object]:
    git_to_dict = getattr(git_state, "to_dict", None)
    if not callable(git_to_dict):
        raise TextRouterCommandError("Git state does not support serialization")
    git_payload = cast(dict[str, object], git_to_dict())
    commit = git_payload.get("commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise TextRouterCommandError("text-router evidence requires a full Git commit")
    calibration = _calibration_payload(selection)
    threshold = cast(Mapping[str, object], calibration["threshold"])["threshold"]
    if not isinstance(threshold, int | float) or isinstance(threshold, bool):
        raise TextRouterCommandError("selected routing threshold is malformed")
    return {
        "schema_version": RUN_EVIDENCE_SCHEMA,
        "mode": mode,
        "corpus_manifest": dict(corpus_manifest),
        "split_fingerprints": _split_fingerprints(corpus_manifest),
        "git_commit": commit,
        "git_state": git_payload,
        "dependencies": dict(dependencies),
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "training_config": {
            "optimization": training_config.to_dict(),
            "batch_size": batch_size,
            "maximum_sequence_length": maximum_sequence_length,
            "dtype": "float32",
            "mixed_precision": "none",
            "class_weighting": "none",
            "train_order": "seeded-permutation-v0",
            "frozen_layers": [],
            "unfrozen_layers": [
                "encoder:*",
                "status_head",
                "object_head",
                "bin_head",
            ],
            "train_only_statistics": "not_applicable_no_numeric_normalization",
        },
        "training_result": dict(training_result),
        "calibration_config": {
            "evidence_split": "validation",
            "temperature_iterations": TEMPERATURE_ITERATIONS,
            "threshold_candidates": list(THRESHOLD_CANDIDATES),
            "maximum_false_route_rate": MAXIMUM_FALSE_ROUTE_RATE,
            "objective_order": [
                "maximum_valid_full_task_accuracy",
                "false_route_rate_lte_0.03",
                "higher_rejection_recall",
                "higher_threshold",
            ],
        },
        "calibration_selection": calibration,
        "router_config": {
            "maximum_sequence_length": maximum_sequence_length,
            "routing_threshold": float(threshold),
            "device_independent": True,
        },
        "data_usage": {
            "gradient_split": "train",
            "checkpoint_selection_split": "validation",
            "temperature_calibration_split": "validation",
            "threshold_selection_split": "validation",
            "train_examples": train_examples,
            "validation_examples": validation_examples,
            "development_examples_materialized": False,
            "development_used_for_training_or_selection": False,
            "final_examples_materialized": False,
            "final_used_for_training_or_selection": False,
        },
        "training_runtime": {
            "device": training_device,
            "dtype": "float32",
            "cuda_available": torch.cuda.is_available(),
        },
        "random_seeds": {
            "torch": training_config.seed,
            "cuda": training_config.seed,
            "data_order": training_config.seed,
        },
    }


def _validate_target_prerequisites(git_state: object) -> None:
    baseline = getattr(git_state, "baseline_tracked", None)
    dirty = getattr(git_state, "dirty", None)
    if baseline is not True or dirty is not False:
        raise TextRouterCommandError(
            "target-development classifier training requires a clean tracked Git baseline"
        )
    if not torch.cuda.is_available():
        raise TextRouterCommandError("target-development classifier training requires CUDA")
    if _package_version("transformers") != "5.4.0":
        raise TextRouterCommandError("target-development requires transformers==5.4.0")
    if _package_version("tokenizers") != "0.22.2":
        raise TextRouterCommandError("target-development requires tokenizers==0.22.2")


def _train_and_promote(
    *,
    args: argparse.Namespace,
    output_root: Path,
    git_state: object,
    mode: str,
) -> dict[str, object]:
    corpus = build_language_corpus()
    train = corpus.examples_for_split(LanguageSplit.TRAIN)
    validation = corpus.examples_for_split(LanguageSplit.VALIDATION)
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if mode == "target_development":
        _validate_target_prerequisites(git_state)
        config = TARGET_TRAINING_CONFIG
        batch_size = TARGET_BATCH_SIZE
        maximum_sequence_length = TARGET_MAXIMUM_SEQUENCE_LENGTH
        model_id = MODEL_ID
        model_revision = MODEL_REVISION
        tokenizer_revision = TOKENIZER_REVISION
        device = "cuda"
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        model, tokenizer = _target_model_and_tokenizer()
    elif mode == "fixture":
        config = TextTrainingConfig(
            maximum_steps=2,
            checkpoint_steps=(1, 2),
            learning_rate=1e-3,
            seed=0,
            weight_decay=0.0,
            maximum_gradient_norm=1.0,
            encoder_trainable=True,
        )
        train = train[:12]
        validation = validation[:12]
        batch_size = 4
        maximum_sequence_length = 32
        model_id = "fixture-distilbert"
        model_revision = "fixture-local-v1"
        tokenizer_revision = "fixture-local-v1"
        device = "cpu"
        torch.manual_seed(config.seed)
        model, tokenizer, temporary = _fixture_model_and_tokenizer()
    else:
        raise TextRouterCommandError("training mode must be fixture or target_development")
    try:
        torch.manual_seed(config.seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        model.to(device)
        train_batches = _factorized_batches(
            tokenizer=tokenizer,
            examples=train,
            split=LanguageSplit.TRAIN,
            batch_size=batch_size,
            maximum_sequence_length=maximum_sequence_length,
            seed=config.seed,
        )
        validation_batches = _factorized_batches(
            tokenizer=tokenizer,
            examples=validation,
            split=LanguageSplit.VALIDATION,
            batch_size=batch_size,
            maximum_sequence_length=maximum_sequence_length,
            seed=config.seed,
        )
        result = run_bounded_factorized_text_training(
            model=model,
            train_batches=train_batches,
            validation_batches=validation_batches,
            config=config,
        )
        selected_state_fingerprint = _model_state_fingerprint(model)
        outputs = collect_factorized_validation_outputs(
            model=model,
            batches=validation_batches,
        )
        selection = calibrate_and_select_text_router(
            outputs=outputs,
            threshold_candidates=THRESHOLD_CANDIDATES,
            maximum_false_route_rate=MAXIMUM_FALSE_ROUTE_RATE,
            temperature_iterations=TEMPERATURE_ITERATIONS,
        )
        result_payload = {
            **result.to_dict(),
            "selected_state_fingerprint": selected_state_fingerprint,
        }
        corpus_manifest = corpus.manifest.to_dict()
        manifest = classifier_manifest(
            model_id=model_id,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            hidden_size=model.hidden_size,
            dropout=model.dropout_probability,
        )
        evidence = _run_evidence(
            mode=mode,
            corpus_manifest=corpus_manifest,
            git_state=git_state,
            dependencies=_dependency_versions(),
            training_config=config,
            batch_size=batch_size,
            maximum_sequence_length=maximum_sequence_length,
            training_result=result_payload,
            selection=selection,
            model_id=model_id,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            training_device=device,
            train_examples=len(train),
            validation_examples=len(validation),
        )
        run_fingerprint = text_classifier_run_fingerprint(
            classifier_manifest=manifest,
            run_evidence=evidence,
        )
        artifact_root = stage_and_promote_text_classifier(
            output_root=output_root,
            run_fingerprint=run_fingerprint,
            model=model,
            tokenizer=tokenizer,
            classifier_manifest=manifest,
            run_evidence=evidence,
            clean_matching_staging=bool(args.clean_staging),
        )
        reloaded, _, reloaded_manifest = load_factorized_text_classifier(artifact_root)
        if reloaded_manifest != manifest:
            raise TextRouterCommandError("fresh local classifier reload changed its manifest")
        if _model_state_fingerprint(reloaded) != selected_state_fingerprint:
            raise TextRouterCommandError("fresh local classifier reload changed selected weights")
        complete = json.loads((artifact_root / "complete.json").read_text(encoding="utf-8"))
        if not isinstance(complete, dict) or complete.get("passed") is not True:
            raise TextRouterCommandError("promoted classifier is missing its completion evidence")
        return {
            "run_fingerprint": run_fingerprint,
            "artifact_root": str(artifact_root),
            "artifact_fingerprint": complete.get("artifact_fingerprint"),
            "classifier_manifest": manifest,
            "run_evidence": evidence,
            "training_result": result_payload,
            "calibration_selection": selection.to_dict(),
            "artifact_reload_validated": True,
        }
    finally:
        if temporary is not None:
            temporary.cleanup()


def _dry_run_payload(git_state: object) -> dict[str, object]:
    corpus = build_language_corpus()
    git_to_dict = getattr(git_state, "to_dict", None)
    if not callable(git_to_dict):
        raise TextRouterCommandError("Git state does not support serialization")
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "corpus_fingerprint": corpus.manifest.corpus_fingerprint,
        "split_fingerprints": _split_fingerprints(corpus.manifest.to_dict()),
        "training_config": {
            "optimization": TARGET_STAGE_TRAINING_CONFIG.to_dict(),
            "batch_size": TARGET_BATCH_SIZE,
            "maximum_sequence_length": TARGET_MAXIMUM_SEQUENCE_LENGTH,
            "dropout": TARGET_DROPOUT,
            "device": "cuda",
        },
        "calibration_config": {
            "evidence_split": "validation",
            "temperature_iterations": TEMPERATURE_ITERATIONS,
            "threshold_candidates": list(THRESHOLD_CANDIDATES),
            "maximum_false_route_rate": MAXIMUM_FALSE_ROUTE_RATE,
        },
        "git_state": git_to_dict(),
        "dependencies": _dependency_versions(),
        "stage_protocol": stage_protocol_manifest(),
        "first_target_stage": "classifier_fixture",
        "target_development_monolith_enabled": False,
        "development_examples_materialized": False,
        "final_examples_materialized": False,
    }


def _balanced_tiny_examples(examples: Sequence[LanguageExample]) -> tuple[LanguageExample, ...]:
    """Select a stable train-only subset spanning every task and rejection status."""

    by_task: dict[str, list[LanguageExample]] = {}
    by_status: dict[RouterStatus, list[LanguageExample]] = {
        status: [] for status in RouterStatus if status is not RouterStatus.ROUTE
    }
    for example in examples:
        if example.split is not LanguageSplit.TRAIN:
            raise TextRouterCommandError("tiny-overfit subset may use train examples only")
        if example.expected_status is RouterStatus.ROUTE:
            if example.task_id is None:
                raise TextRouterCommandError("tiny routeable example is missing a task ID")
            by_task.setdefault(example.task_id, []).append(example)
        else:
            by_status[example.expected_status].append(example)
    if len(by_task) != 6:
        raise TextRouterCommandError("tiny-overfit subset must contain all six TaskSpecs")
    selected: list[LanguageExample] = []
    for task_id in sorted(by_task):
        candidates = sorted(by_task[task_id], key=lambda value: value.example_id)
        if len(candidates) < TINY_ROUTEABLE_EXAMPLES_PER_TASK:
            raise TextRouterCommandError("tiny-overfit task does not have enough examples")
        selected.extend(candidates[:TINY_ROUTEABLE_EXAMPLES_PER_TASK])
    for status in sorted(by_status, key=lambda value: value.value):
        candidates = sorted(by_status[status], key=lambda value: value.example_id)
        if len(candidates) < TINY_REJECTED_EXAMPLES_PER_STATUS:
            raise TextRouterCommandError("tiny-overfit rejection class lacks enough examples")
        selected.extend(candidates[:TINY_REJECTED_EXAMPLES_PER_STATUS])
    if len({value.example_id for value in selected}) != len(selected):
        raise TextRouterCommandError("tiny-overfit subset contains duplicate examples")
    return tuple(selected)


def _authoritative_run_identity(
    *, corpus_manifest: Mapping[str, object], git_state: object
) -> tuple[dict[str, object], str]:
    git_to_dict = getattr(git_state, "to_dict", None)
    if not callable(git_to_dict):
        raise TextRouterCommandError("Git state does not support serialization")
    identity = {
        "schema_version": "langmani-m5a-authoritative-classifier-run-v0",
        "corpus_fingerprint": corpus_manifest["corpus_fingerprint"],
        "split_fingerprints": _split_fingerprints(corpus_manifest),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "training_config": TARGET_STAGE_TRAINING_CONFIG.to_dict(),
        "batch_size": TARGET_BATCH_SIZE,
        "maximum_sequence_length": TARGET_MAXIMUM_SEQUENCE_LENGTH,
        "dropout": TARGET_DROPOUT,
        "git_state": git_to_dict(),
        "dependencies": _dependency_versions(),
        "classifier_training_seeds": M5A_CLASSIFIER_TRAINING_SEEDS,
        "robustness_across_training_seeds_not_evaluated": True,
        "gradient_split": "train",
        "selection_split": "validation",
        "development_examples_materialized": False,
        "final_examples_materialized": False,
    }
    return identity, f"sha256:{sha256_hex(identity)}"


def _write_or_validate_stage(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise TextRouterCommandError("existing stage evidence cannot be parsed") from error
        if existing != dict(payload):
            raise TextRouterCommandError("immutable stage evidence differs from this run")
        return
    atomic_write_json(path, dict(payload))


def _run_tiny_overfit(
    *, args: argparse.Namespace, output_root: Path, git_state: object
) -> dict[str, object]:
    _validate_target_prerequisites(git_state)
    corpus = build_language_corpus()
    examples = _balanced_tiny_examples(corpus.examples_for_split(LanguageSplit.TRAIN))
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model, tokenizer = _target_model_and_tokenizer()
    model.to("cuda")
    batches = _factorized_batches(
        tokenizer=tokenizer,
        examples=examples,
        split=LanguageSplit.TRAIN,
        batch_size=TARGET_BATCH_SIZE,
        maximum_sequence_length=TARGET_MAXIMUM_SEQUENCE_LENGTH,
        seed=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    for step in range(TINY_OVERFIT_MAXIMUM_STEPS):
        train_factorized_text_step(
            model=model,
            optimizer=optimizer,
            batch=replace(
                batches[step % len(batches)],
                input_ids=batches[step % len(batches)].input_ids.to("cuda"),
                attention_mask=batches[step % len(batches)].attention_mask.to("cuda"),
                status_labels=batches[step % len(batches)].status_labels.to("cuda"),
                object_labels=batches[step % len(batches)].object_labels.to("cuda"),
                bin_labels=batches[step % len(batches)].bin_labels.to("cuda"),
            ),
        )
    metrics = classifier_stage_metrics(
        collect_factorized_tiny_train_outputs(model=model, batches=batches)
    )
    manifest = classifier_manifest(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
        hidden_size=model.hidden_size,
        dropout=model.dropout_probability,
    )
    evidence = {
        "schema_version": "langmani-m5a-classifier-tiny-overfit-v0",
        "corpus_fingerprint": corpus.manifest.corpus_fingerprint,
        "tiny_example_ids": [value.example_id for value in examples],
        "tiny_example_count": len(examples),
        "optimization_steps": TINY_OVERFIT_MAXIMUM_STEPS,
        "seed": 0,
        "classifier_training_seeds": 1,
        "gradient_split": "train",
        "validation_examples_materialized": False,
        "development_examples_materialized": False,
        "final_examples_materialized": False,
        "metrics": metrics.to_dict(),
        "git_state": git_state.to_dict(),
    }
    run_fingerprint = text_classifier_run_fingerprint(
        classifier_manifest=manifest, run_evidence=evidence
    )
    artifact = stage_and_promote_text_classifier(
        output_root=output_root / "tiny-overfit",
        run_fingerprint=run_fingerprint,
        model=model,
        tokenizer=tokenizer,
        classifier_manifest=manifest,
        run_evidence=evidence,
        clean_matching_staging=bool(args.clean_staging),
    )
    reloaded, reloaded_tokenizer, reloaded_manifest = load_factorized_text_classifier(artifact)
    reloaded.to("cuda")
    if reloaded_manifest != manifest:
        raise TextRouterCommandError("tiny-overfit checkpoint reload changed its manifest")
    reloaded_batches = _factorized_batches(
        tokenizer=reloaded_tokenizer,
        examples=examples,
        split=LanguageSplit.TRAIN,
        batch_size=TARGET_BATCH_SIZE,
        maximum_sequence_length=TARGET_MAXIMUM_SEQUENCE_LENGTH,
        seed=0,
    )
    original_outputs = collect_factorized_tiny_train_outputs(model=model, batches=batches)
    reloaded_outputs = collect_factorized_tiny_train_outputs(
        model=reloaded, batches=reloaded_batches
    )
    reload_matches = all(
        torch.equal(left, right)
        for left, right in (
            (original_outputs.status_logits, reloaded_outputs.status_logits),
            (original_outputs.object_logits, reloaded_outputs.object_logits),
            (original_outputs.bin_logits, reloaded_outputs.bin_logits),
        )
    )
    if not reload_matches:
        raise TextRouterCommandError("tiny-overfit checkpoint reload changed deterministic logits")
    return {
        "run_fingerprint": run_fingerprint,
        "artifact_root": str(artifact),
        "tiny_subset": {
            "example_count": len(examples),
            "routeable_count": 6 * TINY_ROUTEABLE_EXAMPLES_PER_TASK,
            "rejected_count": 3 * TINY_REJECTED_EXAMPLES_PER_STATUS,
            "all_six_tasks_present": True,
            "all_router_statuses_present": True,
        },
        "tiny_metrics": metrics.to_dict(),
        "classifier_tiny_overfit_validated": metrics.tiny_overfit_gate_passed,
        "artifact_reload_validated": True,
        "cuda_training_validated": True,
    }


def _run_authoritative_stage(
    *,
    args: argparse.Namespace,
    output_root: Path,
    git_state: object,
    resume: bool,
) -> dict[str, object]:
    _validate_target_prerequisites(git_state)
    corpus = build_language_corpus()
    train = corpus.examples_for_split(LanguageSplit.TRAIN)
    validation = corpus.examples_for_split(LanguageSplit.VALIDATION)
    if len(train) != TARGET_STAGE_TRAINING_CONFIG.train_example_count:
        raise TextRouterCommandError("authoritative train count differs from the frozen config")
    identity, run_fingerprint = _authoritative_run_identity(
        corpus_manifest=corpus.manifest.to_dict(), git_state=git_state
    )
    run_root = output_root / "authoritative-runs" / run_fingerprint.removeprefix("sha256:")
    run_root.mkdir(parents=True, exist_ok=True)
    _write_or_validate_stage(run_root / "owner.json", identity)
    pilot_path = run_root / "pilot.json"
    training_path = run_root / "training_complete.json"
    if not resume and pilot_path.exists():
        payload = json.loads(pilot_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("run_fingerprint") != run_fingerprint:
            raise TextRouterCommandError("existing pilot evidence belongs to another run")
        return {**payload, "stage_reused": True}
    if resume:
        if not pilot_path.is_file():
            raise TextRouterCommandError("target resume requires completed pilot evidence")
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        if (
            not isinstance(pilot, dict)
            or pilot.get("run_fingerprint") != run_fingerprint
            or pilot.get("classifier_pilot_completed") is not True
            or pilot.get("classifier_pilot_promoted") is not True
        ):
            raise TextRouterCommandError("classifier pilot did not authorize resume")
        if training_path.exists():
            payload = json.loads(training_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("run_fingerprint") != run_fingerprint:
                raise TextRouterCommandError("existing training evidence belongs to another run")
            return {**payload, "stage_reused": True}
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model, tokenizer = _target_model_and_tokenizer()
    model.to("cuda")
    train_batches = _factorized_batches(
        tokenizer=tokenizer,
        examples=train,
        split=LanguageSplit.TRAIN,
        batch_size=TARGET_BATCH_SIZE,
        maximum_sequence_length=TARGET_MAXIMUM_SEQUENCE_LENGTH,
        seed=0,
    )
    validation_batches = _factorized_batches(
        tokenizer=tokenizer,
        examples=validation,
        split=LanguageSplit.VALIDATION,
        batch_size=TARGET_BATCH_SIZE,
        maximum_sequence_length=TARGET_MAXIMUM_SEQUENCE_LENGTH,
        seed=0,
    )
    result = run_staged_factorized_text_training(
        model=model,
        train_batches=train_batches,
        validation_batches=validation_batches,
        config=TARGET_STAGE_TRAINING_CONFIG,
        run_fingerprint=run_fingerprint,
        checkpoint_root=run_root / "checkpoints",
        processor_state={
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "tokenizer_revision": TOKENIZER_REVISION,
            "maximum_sequence_length": TARGET_MAXIMUM_SEQUENCE_LENGTH,
        },
        stop_after_pilot=not resume,
        resume=resume,
    )
    if not resume:
        metrics = result.pilot_metrics
        if metrics is None:
            raise TextRouterCommandError("pilot stage did not produce validation metrics")
        payload = {
            "schema_version": "langmani-m5a-classifier-pilot-v0",
            "run_fingerprint": run_fingerprint,
            "classifier_pilot_completed": True,
            "classifier_pilot_promoted": metrics.pilot_gate_passed,
            "pilot_step": TARGET_STAGE_TRAINING_CONFIG.pilot_step,
            "same_authoritative_run_required_for_resume": True,
            "metrics": metrics.to_dict(),
            "training_result": result.to_dict(),
            "classifier_training_seeds": 1,
            "robustness_across_training_seeds_not_evaluated": True,
            "passed": True,
        }
        _write_or_validate_stage(pilot_path, payload)
        return payload
    metrics = result.final_metrics
    if metrics is None or result.selected_step is None:
        raise TextRouterCommandError("resumed training did not produce selected validation metrics")
    outputs = collect_factorized_validation_outputs(model=model, batches=validation_batches)
    selection = calibrate_and_select_text_router(
        outputs=outputs,
        threshold_candidates=THRESHOLD_CANDIDATES,
        maximum_false_route_rate=MAXIMUM_FALSE_ROUTE_RATE,
        temperature_iterations=TEMPERATURE_ITERATIONS,
    )
    manifest = classifier_manifest(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
        hidden_size=model.hidden_size,
        dropout=model.dropout_probability,
    )
    legacy_result = {
        "completed_steps": result.completed_steps,
        "stopped_early": result.stopped_early,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "selected_checkpoint_id": f"step_{result.selected_step:08d}",
        "selected_step": result.selected_step,
        "checkpoint_validations": result.to_dict()["validation_records"],
        "config_fingerprint": TARGET_STAGE_TRAINING_CONFIG.fingerprint,
        "resumed_from_step": result.resumed_from_step,
    }
    evidence = _run_evidence(
        mode="target_training_complete",
        corpus_manifest=corpus.manifest.to_dict(),
        git_state=git_state,
        dependencies=_dependency_versions(),
        training_config=TARGET_TRAINING_CONFIG,
        batch_size=TARGET_BATCH_SIZE,
        maximum_sequence_length=TARGET_MAXIMUM_SEQUENCE_LENGTH,
        training_result=legacy_result,
        selection=selection,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
        training_device="cuda",
        train_examples=len(train),
        validation_examples=len(validation),
    )
    evidence.update(
        {
            "authoritative_run_fingerprint": run_fingerprint,
            "classifier_training_seeds": 1,
            "robustness_across_training_seeds_not_evaluated": True,
            "classifier_training_promoted": metrics.training_gate_passed,
            "classifier_training_metrics": metrics.to_dict(),
        }
    )
    artifact_fingerprint = text_classifier_run_fingerprint(
        classifier_manifest=manifest, run_evidence=evidence
    )
    artifact = stage_and_promote_text_classifier(
        output_root=output_root / "promoted",
        run_fingerprint=artifact_fingerprint,
        model=model,
        tokenizer=tokenizer,
        classifier_manifest=manifest,
        run_evidence=evidence,
        clean_matching_staging=bool(args.clean_staging),
    )
    payload = {
        "schema_version": "langmani-m5a-classifier-training-complete-v0",
        "run_fingerprint": run_fingerprint,
        "artifact_run_fingerprint": artifact_fingerprint,
        "artifact_root": str(artifact),
        "classifier_training_completed": True,
        "classifier_checkpoint_selected": True,
        "classifier_calibration_validated": True,
        "classifier_training_promoted": metrics.training_gate_passed,
        "metrics": metrics.to_dict(),
        "calibration_selection": selection.to_dict(),
        "training_result": result.to_dict(),
        "resumed_same_authoritative_run": result.resumed_from_step
        == TARGET_STAGE_TRAINING_CONFIG.pilot_step,
        "classifier_training_seeds": 1,
        "robustness_across_training_seeds_not_evaluated": True,
        "passed": True,
    }
    _write_or_validate_stage(training_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mode = (
        "target_development_retired"
        if args.target_development
        else "target_resume"
        if args.target_resume
        else "target_pilot"
        if args.target_pilot
        else "tiny_overfit"
        if args.tiny_overfit
        else "fixture"
        if args.fixture
        else "dry_run"
    )
    payload: dict[str, object] = {
        "schema_version": COMMAND_SCHEMA,
        "passed": False,
        "mode": mode,
        "classifier_training_completed": False,
        "classifier_fixture_validated": False,
        "classifier_tiny_overfit_validated": False,
        "classifier_pilot_completed": False,
        "classifier_pilot_promoted": False,
        "classifier_checkpoint_selected": False,
        "classifier_calibration_validated": False,
        "artifact_reload_validated": False,
        "cuda_training_validated": False,
        "development_accessed": False,
        "language_final_accessed": False,
        "control_final_accessed": False,
        "m42_final_accessed": False,
        "test_split_accessed": False,
        "historical_fresh_accessed": False,
        "smolvla_go": False,
        "physical_target_validated": False,
    }
    report_path: Path | None = None
    try:
        output_root, report_path = _safe_paths(args.output_root, args.report)
        git_state = inspect_git_state(PROJECT_ROOT)
        if args.target_development:
            raise TextRouterCommandError(
                "--target-development was retired by the staged M5A protocol; "
                "run --target-pilot and then separately authorize --target-resume"
            )
        if args.dry_run:
            payload["plan"] = _dry_run_payload(git_state)
        elif args.tiny_overfit:
            payload.update(
                _run_tiny_overfit(args=args, output_root=output_root, git_state=git_state)
            )
        elif args.target_pilot:
            payload.update(
                _run_authoritative_stage(
                    args=args,
                    output_root=output_root,
                    git_state=git_state,
                    resume=False,
                )
            )
            payload["cuda_training_validated"] = True
        elif args.target_resume:
            payload.update(
                _run_authoritative_stage(
                    args=args,
                    output_root=output_root,
                    git_state=git_state,
                    resume=True,
                )
            )
            payload["artifact_reload_validated"] = True
            payload["cuda_training_validated"] = True
        else:
            result = _train_and_promote(
                args=args,
                output_root=output_root,
                git_state=git_state,
                mode=mode,
            )
            payload.update(result)
            payload.update(
                {
                    "classifier_training_completed": False,
                    "classifier_fixture_validated": bool(args.fixture),
                    "classifier_checkpoint_selected": False,
                    "classifier_calibration_validated": False,
                    "artifact_reload_validated": True,
                    "cuda_training_validated": False,
                }
            )
        payload["passed"] = True
    except Exception as error:  # noqa: BLE001 - outer command preserves exact diagnostics
        traceback.print_exc()
        payload.update(
            {
                "error_type": type(error).__name__,
                "error_message": str(error) or repr(error),
            }
        )
    if report_path is not None:
        try:
            atomic_write_json(report_path, payload)
        except Exception:
            traceback.print_exc()
            return 1
    console = {
        key: payload[key]
        for key in (
            "schema_version",
            "passed",
            "mode",
            "classifier_training_completed",
            "classifier_fixture_validated",
            "classifier_tiny_overfit_validated",
            "classifier_pilot_completed",
            "classifier_pilot_promoted",
            "classifier_checkpoint_selected",
            "classifier_calibration_validated",
            "artifact_reload_validated",
            "cuda_training_validated",
            "physical_target_validated",
            "run_fingerprint",
            "artifact_root",
            "error_type",
            "error_message",
        )
        if key in payload
    }
    print(json.dumps(console, sort_keys=True, allow_nan=False))
    return 0 if payload["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
