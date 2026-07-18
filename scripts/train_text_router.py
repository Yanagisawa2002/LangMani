"""Run one explicit stage of the single-seed M5A text-router pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import traceback
from collections import Counter
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
    CheckpointRetentionPolicy,
    ClassifierRecoveryAmendment,
    ClassifierStageMetrics,
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
    audit_staged_factorized_text_checkpoint,
    calibrate_and_select_text_router,
    classifier_stage_metrics,
    collect_factorized_tiny_train_outputs,
    collect_factorized_validation_outputs,
    factorized_validation_fixture_payload,
    run_bounded_factorized_text_training,
    run_staged_factorized_text_training,
    stage_and_promote_text_classifier,
    text_classifier_run_fingerprint,
    train_factorized_text_step,
    validate_completed_text_classifier_artifact,
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
AUTHORIZED_RECOVERY_RUN_FINGERPRINT = (
    "sha256:9e3ac659fa2b695c843650df35e3779741d94b3dd70b2aec52a429bc4b2edf49"
)
AUTHORIZED_RECOVERY_PILOT_CHECKPOINT_SHA256 = (
    "sha256:a892c2b73c87884b2b2acf843d22ed9318e6b661640eea6a4964ff8d739b5758"
)
AUTHORIZED_RECOVERY_PILOT_GIT_COMMIT = "689918fc3e8736f9d9981591d5904df0afad7031"

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
_FIXTURE_STAGE_REPORT = (
    PROJECT_ROOT / "outputs" / "diagnostics" / "m5a" / "stages" / "classifier-fixture.json"
)
_TINY_STAGE_REPORT = (
    PROJECT_ROOT / "outputs" / "diagnostics" / "m5a" / "stages" / "classifier-tiny-overfit.json"
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
    modes.add_argument("--target-pilot-recovery-resume", action="store_true")
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
    parser.add_argument("--recovery-run-fingerprint")
    parser.add_argument("--recovery-pilot-checkpoint-sha256")
    parser.add_argument("--recovery-maximum-total-epochs", type=int)
    parser.add_argument("--recovery-early-stopping-patience", type=int)
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


def _target_model_and_tokenizer(
    *, local_files_only: bool = False
) -> tuple[FactorizedTextClassifierV0, object]:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise TextRouterCommandError("Transformers 5.4.0 is required") from error
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=TOKENIZER_REVISION,
        trust_remote_code=False,
        use_fast=True,
        local_files_only=local_files_only,
    )
    model = FactorizedTextClassifierV0.from_pretrained_encoder(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        dropout=TARGET_DROPOUT,
        local_files_only=local_files_only,
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _prior_classifier_stage_evidence() -> dict[str, object]:
    results: dict[str, object] = {}
    for name, path, requirements in (
        (
            "fixture",
            _FIXTURE_STAGE_REPORT,
            ("classifier_fixture_validated", "artifact_reload_validated"),
        ),
        (
            "tiny_overfit",
            _TINY_STAGE_REPORT,
            (
                "classifier_tiny_overfit_validated",
                "artifact_reload_validated",
                "cuda_training_validated",
            ),
        ),
    ):
        if not path.is_file() or path.is_symlink():
            raise TextRouterCommandError(f"required {name} stage report is missing or linked")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise TextRouterCommandError(f"required {name} stage report is invalid") from error
        if not isinstance(payload, Mapping) or payload.get("passed") is not True:
            raise TextRouterCommandError(f"required {name} stage did not pass")
        if any(payload.get(key) is not True for key in requirements):
            raise TextRouterCommandError(f"required {name} stage lacks its validation flags")
        for forbidden in (
            "development_accessed",
            "language_final_accessed",
            "control_final_accessed",
            "m42_final_accessed",
            "test_split_accessed",
            "historical_fresh_accessed",
            "smolvla_go",
        ):
            if payload.get(forbidden) is not False:
                raise TextRouterCommandError(
                    f"required {name} stage does not prove {forbidden}=false"
                )
        results[name] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "requirements": list(requirements),
        }
    return results


def _example_counts(examples: Sequence[LanguageExample]) -> dict[str, object]:
    task_counts = Counter(example.task_id or "none" for example in examples)
    status_counts = Counter(example.expected_status.value for example in examples)
    reason_counts = Counter(
        (
            "none"
            if example.expected_rejection_reason is None
            else example.expected_rejection_reason.value
        )
        for example in examples
    )
    return {
        "examples": len(examples),
        "by_task_id": dict(sorted(task_counts.items())),
        "by_router_status": dict(sorted(status_counts.items())),
        "by_rejection_reason": dict(sorted(reason_counts.items())),
    }


def _preflight_payload(
    *,
    corpus_manifest: Mapping[str, object],
    train: Sequence[LanguageExample],
    validation: Sequence[LanguageExample],
    git_state: object,
    run_fingerprint: str,
    run_root: Path,
    prior_evidence: Mapping[str, object],
    checkpoint_processor_state: Mapping[str, object],
) -> dict[str, object]:
    git_to_dict = getattr(git_state, "to_dict", None)
    if not callable(git_to_dict):
        raise TextRouterCommandError("Git state does not support serialization")
    device_name = torch.cuda.get_device_name(0)
    return {
        "schema_version": "langmani-m5a-classifier-pilot-preflight-v0",
        "git_state": git_to_dict(),
        "runtime": {
            "gpu": device_name,
            "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "transformers": _package_version("transformers"),
            "tokenizers": _package_version("tokenizers"),
        },
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "corpus_fingerprint": corpus_manifest["corpus_fingerprint"],
        "split_fingerprints": _split_fingerprints(corpus_manifest),
        "train_counts": _example_counts(train),
        "validation_counts": _example_counts(validation),
        "training_seed": TARGET_STAGE_TRAINING_CONFIG.seed,
        "classifier_training_seeds": M5A_CLASSIFIER_TRAINING_SEEDS,
        "robustness_across_training_seeds_not_evaluated": True,
        "optimization": TARGET_STAGE_TRAINING_CONFIG.to_dict(),
        "batch_size": TARGET_BATCH_SIZE,
        "maximum_sequence_length": TARGET_MAXIMUM_SEQUENCE_LENGTH,
        "pilot_boundary": {
            "definition": "min(one_complete_train_pass,ceil(0.20*maximum_steps))",
            "steps_per_epoch": TARGET_STAGE_TRAINING_CONFIG.steps_per_epoch,
            "maximum_steps": TARGET_STAGE_TRAINING_CONFIG.maximum_steps,
            "pilot_step": TARGET_STAGE_TRAINING_CONFIG.pilot_step,
        },
        "run_fingerprint": run_fingerprint,
        "run_root": str(run_root),
        "expected_pilot_checkpoint": str(run_root / "checkpoints" / "pilot.pt"),
        "checkpoint_processor_state": dict(checkpoint_processor_state),
        "gradient_split": "train",
        "selection_split": "validation",
        "prior_stage_evidence": dict(prior_evidence),
        "development_examples_materialized": False,
        "final_examples_materialized": False,
        "control_development_accessed": False,
        "control_final_accessed": False,
    }


def _authoritative_checkpoint_processor_state(
    *,
    corpus_manifest: Mapping[str, object],
    git_state: object,
    run_fingerprint: str,
    run_root: Path,
) -> dict[str, object]:
    git_to_dict = getattr(git_state, "to_dict", None)
    if not callable(git_to_dict):
        raise TextRouterCommandError("Git state does not support serialization")
    return {
        "schema_version": "langmani-m5a-classifier-checkpoint-contract-v0",
        "model": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dropout": TARGET_DROPOUT,
            "architecture": "FactorizedTextClassifierV0",
        },
        "tokenizer": {
            "model_id": MODEL_ID,
            "tokenizer_revision": TOKENIZER_REVISION,
            "maximum_sequence_length": TARGET_MAXIMUM_SEQUENCE_LENGTH,
        },
        "label_mappings": {
            "status": list(STATUS_LABELS),
            "object": list(OBJECT_LABELS),
            "bin": list(BIN_LABELS),
            "rejected_object_bin_loss_mask": -1,
        },
        "training_seed": TARGET_STAGE_TRAINING_CONFIG.seed,
        "corpus_fingerprint": corpus_manifest["corpus_fingerprint"],
        "split_fingerprints": _split_fingerprints(corpus_manifest),
        "run_fingerprint": run_fingerprint,
        "git_state": git_to_dict(),
        "dependencies": _dependency_versions(),
        "pilot_boundary": {
            "definition": "min(one_complete_train_pass,ceil(0.20*maximum_steps))",
            "steps_per_epoch": TARGET_STAGE_TRAINING_CONFIG.steps_per_epoch,
            "maximum_steps": TARGET_STAGE_TRAINING_CONFIG.maximum_steps,
            "pilot_step": TARGET_STAGE_TRAINING_CONFIG.pilot_step,
        },
        "run_owner_path": str(run_root / "owner.json"),
        "preflight_path": str(run_root / "preflight.json"),
    }


def _mean(values: torch.Tensor) -> float:
    return 0.0 if values.numel() == 0 else float(values.float().mean())


def _confusion_matrix(
    *, labels: torch.Tensor, predictions: torch.Tensor, names: Sequence[str]
) -> dict[str, dict[str, int]]:
    return {
        expected: {
            predicted: int(((labels == row) & (predictions == column)).sum())
            for column, predicted in enumerate(names)
        }
        for row, expected in enumerate(names)
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = int(round((len(ordered) - 1) * quantile))
    return ordered[index]


def _detailed_validation_metrics(
    *,
    outputs: object,
    examples: Sequence[LanguageExample],
) -> dict[str, object]:
    if not hasattr(outputs, "status_logits"):
        raise TextRouterCommandError("validation outputs are malformed")
    status_logits = cast(torch.Tensor, outputs.status_logits)
    object_logits = cast(torch.Tensor, outputs.object_logits)
    bin_logits = cast(torch.Tensor, outputs.bin_logits)
    status_labels = cast(torch.Tensor, outputs.status_labels)
    object_labels = cast(torch.Tensor, outputs.object_labels)
    bin_labels = cast(torch.Tensor, outputs.bin_labels)
    values = tuple(examples)
    if len(values) != int(status_labels.shape[0]):
        raise TextRouterCommandError("validation examples and outputs are misaligned")
    status = status_logits.argmax(dim=-1)
    objects = object_logits.argmax(dim=-1)
    bins = bin_logits.argmax(dim=-1)
    route_index = STATUS_LABELS.index("route")
    routeable = status_labels == route_index
    rejected = ~routeable
    full_route = (status == route_index) & (objects == object_labels) & (bins == bin_labels)
    decision_correct = torch.where(routeable, full_route, status == status_labels)
    predicted_reject = status != route_index
    true_reject = rejected
    rejection_true_positive = int((predicted_reject & true_reject).sum())
    rejection_predicted_positive = int(predicted_reject.sum())
    rejection_actual_positive = int(true_reject.sum())

    by_task: dict[str, object] = {}
    for task_id in sorted({example.task_id for example in values if example.task_id is not None}):
        indices = torch.tensor(
            [index for index, example in enumerate(values) if example.task_id == task_id],
            dtype=torch.long,
        )
        by_task[cast(str, task_id)] = {
            "examples": int(indices.numel()),
            "full_task_spec_accuracy": _mean(full_route[indices]),
        }

    by_family: dict[str, object] = {}
    routeable_by_family: dict[str, object] = {}
    for family in sorted({example.template_family_id for example in values}):
        indices = torch.tensor(
            [index for index, example in enumerate(values) if example.template_family_id == family],
            dtype=torch.long,
        )
        by_family[family] = {
            "examples": int(indices.numel()),
            "decision_accuracy": _mean(decision_correct[indices]),
        }
        routeable_indices = torch.tensor(
            [index for index in indices.tolist() if bool(routeable[index])],
            dtype=torch.long,
        )
        if int(routeable_indices.numel()) > 0:
            routeable_by_family[family] = {
                "examples": int(routeable_indices.numel()),
                "full_task_spec_accuracy": _mean(full_route[routeable_indices]),
            }

    rejection_reason_confusion: dict[str, dict[str, int]] = {}
    for reason in sorted(
        {
            example.expected_rejection_reason.value
            for example in values
            if example.expected_rejection_reason is not None
        }
    ):
        indices = [
            index
            for index, example in enumerate(values)
            if example.expected_rejection_reason is not None
            and example.expected_rejection_reason.value == reason
        ]
        rejection_reason_confusion[reason] = {
            predicted: sum(int(status[index]) == column for index in indices)
            for column, predicted in enumerate(STATUS_LABELS)
        }

    latency_seconds = cast(tuple[float, ...], outputs.batch_latency_seconds)
    latency_ms = [value * 1000.0 for value in latency_seconds]
    return {
        "evidence_split": "validation",
        "routeable": {
            "examples": int(routeable.sum()),
            "full_task_spec_accuracy": _mean(full_route[routeable]),
            "target_object_accuracy": _mean((objects == object_labels)[routeable]),
            "destination_bin_accuracy": _mean((bins == bin_labels)[routeable]),
            "routing_status_accuracy": _mean((status == status_labels)[routeable]),
            "route_recall": _mean((status == route_index)[routeable]),
            "false_rejection_rate": _mean((status != route_index)[routeable]),
            "by_task_spec": by_task,
            "by_template_family": routeable_by_family,
            "object_confusion_matrix": _confusion_matrix(
                labels=object_labels[routeable],
                predictions=objects[routeable],
                names=OBJECT_LABELS[:3],
            ),
            "bin_confusion_matrix": _confusion_matrix(
                labels=bin_labels[routeable], predictions=bins[routeable], names=BIN_LABELS[:2]
            ),
        },
        "rejected": {
            "examples": int(rejected.sum()),
            "overall_rejection_accuracy": _mean((status == status_labels)[rejected]),
            "false_route_rate": _mean((status == route_index)[rejected]),
            "rejection_precision": (
                0.0
                if rejection_predicted_positive == 0
                else rejection_true_positive / rejection_predicted_positive
            ),
            "rejection_recall": (
                0.0
                if rejection_actual_positive == 0
                else rejection_true_positive / rejection_actual_positive
            ),
            "ambiguous_recall": _mean(
                (status == STATUS_LABELS.index("reject_ambiguous"))[
                    status_labels == STATUS_LABELS.index("reject_ambiguous")
                ]
            ),
            "unsupported_recall": _mean(
                (status == STATUS_LABELS.index("reject_unsupported"))[
                    status_labels == STATUS_LABELS.index("reject_unsupported")
                ]
            ),
            "malformed_recall": _mean(
                (status == STATUS_LABELS.index("reject_malformed"))[
                    status_labels == STATUS_LABELS.index("reject_malformed")
                ]
            ),
            "rejection_macro_recall": (
                _mean(
                    (status == STATUS_LABELS.index("reject_ambiguous"))[
                        status_labels == STATUS_LABELS.index("reject_ambiguous")
                    ]
                )
                + _mean(
                    (status == STATUS_LABELS.index("reject_unsupported"))[
                        status_labels == STATUS_LABELS.index("reject_unsupported")
                    ]
                )
                + _mean(
                    (status == STATUS_LABELS.index("reject_malformed"))[
                        status_labels == STATUS_LABELS.index("reject_malformed")
                    ]
                )
            )
            / 3.0,
            "rejection_reason_confusion_matrix": rejection_reason_confusion,
        },
        "all": {
            "examples": len(values),
            "schema_valid_decision_rate": 1.0,
            "malformed_decision_count": 0,
            "coverage": 1.0,
            "routing_status_accuracy": _mean(status == status_labels),
            "by_template_family": by_family,
            "latency_unit": "milliseconds_per_validation_batch",
            "latency_p50": _percentile(latency_ms, 0.50),
            "latency_p95": _percentile(latency_ms, 0.95),
            "latency_p99": _percentile(latency_ms, 0.99),
        },
    }


def _classifier_stage_metrics_from_mapping(
    value: Mapping[str, object],
) -> ClassifierStageMetrics:
    try:
        return ClassifierStageMetrics(
            full_task_spec_accuracy=float(value["full_task_spec_accuracy"]),
            object_accuracy=float(value["object_accuracy"]),
            bin_accuracy=float(value["bin_accuracy"]),
            false_route_rate=float(value["false_route_rate"]),
            schema_valid_rate=float(value["schema_valid_rate"]),
            ambiguous_rejection_recall=float(value["ambiguous_rejection_recall"]),
            unsupported_rejection_recall=float(value["unsupported_rejection_recall"]),
            malformed_rejection_recall=float(value["malformed_rejection_recall"]),
            status_accuracy=float(value.get("status_accuracy", 0.0)),
            finite=cast(bool, value.get("finite", True)),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise TextRouterCommandError("classifier validation metrics are malformed") from error


def _recovery_amendment(*, git_state: object) -> ClassifierRecoveryAmendment:
    git_to_dict = getattr(git_state, "to_dict", None)
    if not callable(git_to_dict):
        raise TextRouterCommandError("Git state does not support recovery authorization")
    commit = git_to_dict().get("commit")
    if not isinstance(commit, str):
        raise TextRouterCommandError("recovery implementation Git commit is missing")
    return ClassifierRecoveryAmendment(
        pilot_run_fingerprint=AUTHORIZED_RECOVERY_RUN_FINGERPRINT,
        pilot_checkpoint_fingerprint=AUTHORIZED_RECOVERY_PILOT_CHECKPOINT_SHA256,
        implementation_git_commit=commit,
        unchanged_training_config=TARGET_STAGE_TRAINING_CONFIG.to_dict(),
        maximum_total_epochs=TARGET_STAGE_TRAINING_CONFIG.maximum_epochs,
        early_stopping_patience=TARGET_STAGE_TRAINING_CONFIG.early_stopping_patience,
        validation_schedule=TARGET_STAGE_TRAINING_CONFIG.validation_steps,
        checkpoint_retention_policy=CheckpointRetentionPolicy().to_dict(),
    )


def _validate_recovery_arguments(args: argparse.Namespace) -> None:
    required = {
        "--recovery-run-fingerprint": (
            args.recovery_run_fingerprint,
            AUTHORIZED_RECOVERY_RUN_FINGERPRINT,
        ),
        "--recovery-pilot-checkpoint-sha256": (
            args.recovery_pilot_checkpoint_sha256,
            AUTHORIZED_RECOVERY_PILOT_CHECKPOINT_SHA256,
        ),
        "--recovery-maximum-total-epochs": (
            args.recovery_maximum_total_epochs,
            TARGET_STAGE_TRAINING_CONFIG.maximum_epochs,
        ),
        "--recovery-early-stopping-patience": (
            args.recovery_early_stopping_patience,
            TARGET_STAGE_TRAINING_CONFIG.early_stopping_patience,
        ),
    }
    for name, (actual, expected) in required.items():
        if actual is None:
            raise TextRouterCommandError(f"{name} is required for recovery continuation")
        if actual != expected:
            raise TextRouterCommandError(f"{name} does not match the approved amendment")


def _read_json_mapping(path: Path, *, label: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink() or path.is_junction():
        raise TextRouterCommandError(f"{label} is missing, linked, or not a real file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TextRouterCommandError(f"{label} cannot be parsed") from error
    if not isinstance(value, dict):
        raise TextRouterCommandError(f"{label} must contain one JSON object")
    return cast(dict[str, object], value)


def _recovery_pre_resume_audit(
    *,
    train: Sequence[LanguageExample],
    validation: Sequence[LanguageExample],
    train_batches: Sequence[FactorizedTextBatch],
    pilot: Mapping[str, object],
) -> dict[str, object]:
    train_values = tuple(train)
    validation_values = tuple(validation)
    status_counts = Counter(value.expected_status.value for value in train_values)
    validation_status_counts = Counter(value.expected_status.value for value in validation_values)
    rejection_counts = Counter(
        "none" if value.expected_rejection_reason is None else value.expected_rejection_reason.value
        for value in train_values
    )
    validation_rejection_counts = Counter(
        "none" if value.expected_rejection_reason is None else value.expected_rejection_reason.value
        for value in validation_values
    )
    object_counts = Counter(
        "none" if value.expected_task_spec is None else value.expected_task_spec.target_object_id
        for value in train_values
    )
    bin_counts = Counter(
        "none" if value.expected_task_spec is None else value.expected_task_spec.target_bin_id
        for value in train_values
    )
    task_counts = Counter(value.task_id or "none" for value in train_values)
    labels = tuple(_label_indices(value) for value in train_values)
    rejected_masked = all(
        object_index == -1 and bin_index == -1
        for value, (_status, object_index, bin_index) in zip(train_values, labels, strict=True)
        if value.expected_status is not RouterStatus.ROUTE
    )
    label_mapping_valid = all(
        status_index == STATUS_LABELS.index(value.expected_status.value)
        and (
            (object_index, bin_index) == (-1, -1)
            if value.expected_task_spec is None
            else (
                object_index == OBJECT_LABELS.index(value.expected_task_spec.target_object_id)
                and bin_index == BIN_LABELS.index(value.expected_task_spec.target_bin_id)
            )
        )
        for value, (status_index, object_index, bin_index) in zip(train_values, labels, strict=True)
    )
    train_families = {value.template_family_id for value in train_values}
    validation_families = {value.template_family_id for value in validation_values}
    unsupported = tuple(
        value for value in train_values if value.expected_status is RouterStatus.REJECT_UNSUPPORTED
    )
    malformed = tuple(
        value for value in train_values if value.expected_status is RouterStatus.REJECT_MALFORMED
    )
    rejected_labels_correct = all(
        value.expected_task_spec is None and value.expected_rejection_reason is not None
        for value in (*unsupported, *malformed)
    )
    flattened_status = torch.cat(tuple(batch.status_labels for batch in train_batches)).tolist()
    sampler_status_counts = Counter(STATUS_LABELS[index] for index in flattened_status)
    training_integrity = pilot.get("training_integrity")
    if not isinstance(training_integrity, Mapping):
        raise TextRouterCommandError("pilot training integrity evidence is missing")
    head_gradients = training_integrity.get("head_maximum_gradient_norms")
    status_head_nonzero = (
        isinstance(head_gradients, Mapping)
        and isinstance(head_gradients.get("status_head"), int | float)
        and float(cast(float, head_gradients["status_head"])) > 0.0
    )
    checks = {
        "train_validation_status_counts_reported": set(status_counts) == set(STATUS_LABELS)
        and set(validation_status_counts) == set(STATUS_LABELS),
        "rejection_reason_counts_reported": "none" in rejection_counts
        and "none" in validation_rejection_counts,
        "target_object_counts_reported": set(object_counts) == {*OBJECT_LABELS[:3], "none"},
        "destination_bin_counts_reported": set(bin_counts) == {*BIN_LABELS[:2], "none"},
        "complete_task_spec_counts_reported": len(task_counts) == 7 and task_counts["none"] > 0,
        "rejected_object_bin_loss_masking": rejected_masked,
        "routing_status_class_indices": STATUS_LABELS
        == ("route", "reject_ambiguous", "reject_unsupported", "reject_malformed"),
        "object_bin_none_class_indices": OBJECT_LABELS.index("none") == 3
        and BIN_LABELS.index("none") == 2,
        "no_label_remapping_mismatch": label_mapping_valid,
        "no_template_family_leakage": train_families.isdisjoint(validation_families),
        "unsupported_and_malformed_train_examples_present": bool(unsupported) and bool(malformed),
        "unsupported_and_malformed_train_labels_correct": rejected_labels_correct,
        "pilot_status_head_gradient_nonzero": status_head_nonzero,
        "no_sampler_class_exclusion": sampler_status_counts == status_counts,
        "all_900_examples_seen_in_first_epoch": len(train_values) == 900
        and len(flattened_status) == 900
        and training_integrity.get("examples_processed") == 900
        and training_integrity.get("no_silently_skipped_examples") is True,
    }
    payload: dict[str, object] = {
        "schema_version": "langmani-m5a-classifier-recovery-data-loss-audit-v0",
        "read_only": True,
        "gradient_split": "train",
        "selection_split": "validation",
        "train_counts": {
            "examples": len(train_values),
            "by_router_status": dict(sorted(status_counts.items())),
            "by_rejection_reason": dict(sorted(rejection_counts.items())),
            "by_target_object": dict(sorted(object_counts.items())),
            "by_destination_bin": dict(sorted(bin_counts.items())),
            "by_complete_task_spec": dict(sorted(task_counts.items())),
        },
        "validation_counts": {
            "examples": len(validation_values),
            "by_router_status": dict(sorted(validation_status_counts.items())),
            "by_rejection_reason": dict(sorted(validation_rejection_counts.items())),
        },
        "label_contract": {
            "status_indices": {name: index for index, name in enumerate(STATUS_LABELS)},
            "object_indices": {name: index for index, name in enumerate(OBJECT_LABELS)},
            "bin_indices": {name: index for index, name in enumerate(BIN_LABELS)},
            "rejected_object_bin_loss_mask": -1,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    if payload["passed"] is not True:
        failed = [name for name, passed in checks.items() if not passed]
        raise TextRouterCommandError(f"pre-resume data/loss audit failed: {failed}")
    return payload


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


def _run_authorized_recovery_stage(
    *,
    args: argparse.Namespace,
    output_root: Path,
    git_state: object,
) -> dict[str, object]:
    stage_started = time.perf_counter()
    _validate_target_prerequisites(git_state)
    _validate_recovery_arguments(args)
    corpus = build_language_corpus()
    train = corpus.examples_for_split(LanguageSplit.TRAIN)
    validation = corpus.examples_for_split(LanguageSplit.VALIDATION)
    run_fingerprint = AUTHORIZED_RECOVERY_RUN_FINGERPRINT
    run_root = output_root / "authoritative-runs" / run_fingerprint.removeprefix("sha256:")
    owner = _read_json_mapping(run_root / "owner.json", label="recovery run owner")
    preflight = _read_json_mapping(run_root / "preflight.json", label="pilot preflight")
    pilot = _read_json_mapping(run_root / "pilot.json", label="rejected pilot evidence")
    pilot_checkpoint = run_root / "checkpoints" / "pilot.pt"
    pilot_checkpoint_sha256 = _sha256_file(pilot_checkpoint)
    owner_git = owner.get("git_state")
    authorized_identity = (
        f"sha256:{sha256_hex(owner)}" == run_fingerprint
        and owner.get("schema_version") == "langmani-m5a-authoritative-classifier-run-v0"
        and isinstance(owner_git, Mapping)
        and owner_git.get("commit") == AUTHORIZED_RECOVERY_PILOT_GIT_COMMIT
        and owner_git.get("dirty") is False
        and owner.get("model_id") == MODEL_ID
        and owner.get("model_revision") == MODEL_REVISION
        and owner.get("tokenizer_revision") == TOKENIZER_REVISION
        and owner.get("training_config") == TARGET_STAGE_TRAINING_CONFIG.to_dict()
        and owner.get("corpus_fingerprint") == corpus.manifest.corpus_fingerprint
        and owner.get("split_fingerprints") == _split_fingerprints(corpus.manifest.to_dict())
    )
    if not authorized_identity:
        raise TextRouterCommandError("existing rejected pilot run identity is not authorized")
    pilot_metrics_value = pilot.get("metrics")
    if not isinstance(pilot_metrics_value, Mapping):
        raise TextRouterCommandError("rejected pilot metrics are missing")
    pilot_metrics = _classifier_stage_metrics_from_mapping(pilot_metrics_value)
    pilot_forbidden_sealed = all(
        pilot.get(name) is False
        for name in (
            "development_accessed",
            "language_final_accessed",
            "control_final_accessed",
            "m42_final_accessed",
            "test_split_accessed",
            "historical_fresh_accessed",
            "smolvla_go",
        )
    )
    if not (
        pilot.get("run_fingerprint") == run_fingerprint
        and pilot.get("classifier_pilot_completed") is True
        and pilot.get("classifier_pilot_promoted") is False
        and pilot.get("classifier_training_completed") is False
        and pilot_forbidden_sealed
        and not pilot_metrics.pilot_gate_passed
        and pilot_checkpoint_sha256 == AUTHORIZED_RECOVERY_PILOT_CHECKPOINT_SHA256
    ):
        raise TextRouterCommandError("recovery applies only to the exact verified rejected pilot")
    checkpoint_processor_state = preflight.get("checkpoint_processor_state")
    if not isinstance(checkpoint_processor_state, Mapping):
        raise TextRouterCommandError("pilot checkpoint processor contract is missing")
    if (
        preflight.get("run_fingerprint") != run_fingerprint
        or preflight.get("model_revision") != MODEL_REVISION
        or preflight.get("tokenizer_revision") != TOKENIZER_REVISION
        or preflight.get("training_seed") != 0
        or preflight.get("gradient_split") != "train"
        or preflight.get("selection_split") != "validation"
        or preflight.get("development_examples_materialized") is not False
        or preflight.get("final_examples_materialized") is not False
    ):
        raise TextRouterCommandError("pilot preflight contract differs from the recovery amendment")
    amendment = _recovery_amendment(git_state=git_state)
    amendment_path = run_root / "recovery_amendment.json"
    _write_or_validate_stage(amendment_path, amendment.to_dict())
    persisted_amendment = _read_json_mapping(amendment_path, label="classifier recovery amendment")
    if persisted_amendment != amendment.to_dict():
        raise TextRouterCommandError("persisted recovery amendment fingerprint differs")
    completion_path = run_root / "recovery_completion.json"
    if completion_path.exists():
        completion = _read_json_mapping(completion_path, label="recovery completion")
        if (
            completion.get("run_fingerprint") != run_fingerprint
            or completion.get("amendment_fingerprint") != amendment.amendment_fingerprint
            or completion.get("passed") is not True
        ):
            raise TextRouterCommandError("existing recovery completion belongs to another contract")
        return {**completion, "stage_reused": True}

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model, tokenizer = _target_model_and_tokenizer(local_files_only=True)
    model.to("cuda")
    training_model = model
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
    audit_payload = _recovery_pre_resume_audit(
        train=train,
        validation=validation,
        train_batches=train_batches,
        pilot=pilot,
    )
    audit_payload.update(
        {
            "run_fingerprint": run_fingerprint,
            "pilot_checkpoint_fingerprint": pilot_checkpoint_sha256,
            "amendment_fingerprint": amendment.amendment_fingerprint,
        }
    )
    audit_path = run_root / "recovery_pre_resume_audit.json"
    _write_or_validate_stage(audit_path, audit_payload)

    reload_fixture = pilot.get("reload_fixture")
    if not isinstance(reload_fixture, Mapping):
        raise TextRouterCommandError("pilot reload fixture is missing")
    pre_resume_checkpoint_audit = audit_staged_factorized_text_checkpoint(
        model=model,
        validation_batches=validation_batches,
        config=TARGET_STAGE_TRAINING_CONFIG,
        run_fingerprint=run_fingerprint,
        checkpoint_path=pilot_checkpoint,
        processor_state=checkpoint_processor_state,
        expected_fixture=reload_fixture,
        expected_step=TARGET_STAGE_TRAINING_CONFIG.pilot_step,
    )
    if not all(
        (
            pre_resume_checkpoint_audit.optimizer_state_restored,
            pre_resume_checkpoint_audit.scheduler_state_restored,
            pre_resume_checkpoint_audit.processor_state_restored,
            pre_resume_checkpoint_audit.rng_state_restored,
            pre_resume_checkpoint_audit.data_progress_restored,
            pre_resume_checkpoint_audit.deterministic_logits_match,
            pre_resume_checkpoint_audit.restored_step == 29,
            pre_resume_checkpoint_audit.next_global_step == 30,
        )
    ):
        raise TextRouterCommandError("exact pilot training state did not pass pre-resume audit")

    pilot_training_result = pilot.get("training_result")
    if not isinstance(pilot_training_result, Mapping):
        raise TextRouterCommandError("pilot training result is missing")
    pilot_validation_records = pilot_training_result.get("validation_records")
    if not isinstance(pilot_validation_records, list) or len(pilot_validation_records) != 1:
        raise TextRouterCommandError("pilot validation-loss record is missing")
    pilot_validation_record = pilot_validation_records[0]
    if not isinstance(pilot_validation_record, Mapping):
        raise TextRouterCommandError("pilot validation-loss record is malformed")
    pilot_validation_loss = float(cast(float, pilot_validation_record["total_loss"]))
    epoch_metrics_path = run_root / "recovery_epoch_metrics.json"
    if epoch_metrics_path.exists():
        epoch_metrics_payload = _read_json_mapping(
            epoch_metrics_path, label="recovery epoch metrics"
        )
        entries_value = epoch_metrics_payload.get("epochs")
        if epoch_metrics_payload.get(
            "amendment_fingerprint"
        ) != amendment.amendment_fingerprint or not isinstance(entries_value, list):
            raise TextRouterCommandError("recovery epoch evidence belongs to another amendment")
        epoch_entries: list[dict[str, object]] = [
            cast(dict[str, object], value) for value in entries_value if isinstance(value, dict)
        ]
        if len(epoch_entries) != len(entries_value):
            raise TextRouterCommandError("recovery epoch evidence is malformed")
    else:
        detailed_pilot = pilot.get("detailed_validation_metrics")
        if not isinstance(detailed_pilot, Mapping):
            raise TextRouterCommandError("pilot detailed validation evidence is missing")
        epoch_entries = [
            {
                "epoch": 1,
                "step": 29,
                "metrics": pilot_metrics.to_dict(),
                "rejection_macro_recall": pilot_metrics.rejection_macro_recall,
                "total_validation_loss": pilot_validation_loss,
                "detailed_validation_metrics": dict(detailed_pilot),
                "deterministic_repeatability": pilot.get("deterministic_repeatability") is True,
                "selected_after_interval": True,
                "source": "original_rejected_pilot",
            }
        ]
        atomic_write_json(
            epoch_metrics_path,
            {
                "schema_version": "langmani-m5a-classifier-recovery-epoch-metrics-v0",
                "run_fingerprint": run_fingerprint,
                "amendment_fingerprint": amendment.amendment_fingerprint,
                "selection_split": "validation",
                "epochs": epoch_entries,
            },
        )

    def observe_validation(outputs: object, record: object, ranking: object) -> None:
        if not hasattr(record, "step") or not hasattr(ranking, "to_dict"):
            raise TextRouterCommandError("recovery validation observer received malformed state")
        repeated = collect_factorized_validation_outputs(
            model=training_model, batches=validation_batches
        )
        deterministic = all(
            torch.equal(left, right)
            for left, right in (
                (outputs.status_logits, repeated.status_logits),
                (outputs.object_logits, repeated.object_logits),
                (outputs.bin_logits, repeated.bin_logits),
            )
        )
        if not deterministic:
            raise TextRouterCommandError("epoch validation was not deterministically repeatable")
        ranking_payload = ranking.to_dict()
        entry = {
            "epoch": ranking_payload["epoch"],
            "step": ranking_payload["step"],
            "metrics": ranking_payload["metrics"],
            "rejection_macro_recall": ranking_payload["rejection_macro_recall"],
            "total_validation_loss": ranking_payload["total_validation_loss"],
            "ranking_key": ranking_payload["ranking_key"],
            "detailed_validation_metrics": _detailed_validation_metrics(
                outputs=outputs, examples=validation
            ),
            "deterministic_repeatability": True,
            "source": "authorized_recovery_continuation",
        }
        existing = next(
            (value for value in epoch_entries if value.get("step") == ranking_payload["step"]),
            None,
        )
        if existing is None:
            epoch_entries.append(entry)
        elif (
            existing.get("metrics") != entry["metrics"]
            or existing.get("total_validation_loss") != entry["total_validation_loss"]
        ):
            raise TextRouterCommandError("persisted epoch validation evidence differs")
        atomic_write_json(
            epoch_metrics_path,
            {
                "schema_version": "langmani-m5a-classifier-recovery-epoch-metrics-v0",
                "run_fingerprint": run_fingerprint,
                "amendment_fingerprint": amendment.amendment_fingerprint,
                "selection_split": "validation",
                "epochs": sorted(epoch_entries, key=lambda value: cast(int, value["step"])),
            },
        )

    latest_checkpoint = run_root / "checkpoints" / "latest.pt"
    resume_role = "pilot"
    if latest_checkpoint.is_file():
        latest_header = torch.load(latest_checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(latest_header, Mapping):
            raise TextRouterCommandError("latest checkpoint payload is malformed")
        latest_step = latest_header.get("step")
        if isinstance(latest_step, int) and latest_step > TARGET_STAGE_TRAINING_CONFIG.pilot_step:
            if (
                latest_header.get("run_fingerprint") != run_fingerprint
                or latest_header.get("selection_policy")
                != "classifier_recovery_lexicographic_validation_v0"
                or latest_header.get("recovery_amendment_fingerprint")
                != amendment.amendment_fingerprint
            ):
                raise TextRouterCommandError("partial latest checkpoint is not this recovery run")
            resume_role = "latest"
        del latest_header
    result = run_staged_factorized_text_training(
        model=model,
        train_batches=train_batches,
        validation_batches=validation_batches,
        config=TARGET_STAGE_TRAINING_CONFIG,
        run_fingerprint=run_fingerprint,
        checkpoint_root=run_root / "checkpoints",
        processor_state=checkpoint_processor_state,
        stop_after_pilot=False,
        resume=True,
        recovery_ranking=True,
        recovery_amendment_fingerprint=amendment.amendment_fingerprint,
        resume_checkpoint_role=resume_role,
        recovery_baseline_metrics=pilot_metrics,
        validation_observer=observe_validation,
    )
    metrics = result.final_metrics
    selected_step = result.selected_step
    if metrics is None or selected_step is None:
        raise TextRouterCommandError("recovery training did not select a validation checkpoint")
    selected_checkpoint = (
        pilot_checkpoint
        if selected_step == TARGET_STAGE_TRAINING_CONFIG.pilot_step
        else Path(result.best_checkpoint)
    )
    selected_checkpoint_sha256 = _sha256_file(selected_checkpoint)
    selected_outputs = collect_factorized_validation_outputs(
        model=model, batches=validation_batches
    )
    selected_metrics = classifier_stage_metrics(selected_outputs)
    if selected_metrics.to_dict() != metrics.to_dict():
        raise TextRouterCommandError("selected in-memory validation metrics differ")
    selected_details = _detailed_validation_metrics(outputs=selected_outputs, examples=validation)
    selected_fixture = factorized_validation_fixture_payload(
        collect_factorized_validation_outputs(model=model, batches=(validation_batches[0],))
    )
    del model
    torch.cuda.empty_cache()
    fresh_model, _fresh_tokenizer = _target_model_and_tokenizer(local_files_only=True)
    fresh_model.to("cuda")
    selected_audit = audit_staged_factorized_text_checkpoint(
        model=fresh_model,
        validation_batches=validation_batches,
        config=TARGET_STAGE_TRAINING_CONFIG,
        run_fingerprint=run_fingerprint,
        checkpoint_path=selected_checkpoint,
        processor_state=checkpoint_processor_state,
        expected_fixture=selected_fixture,
        expected_step=selected_step,
        selection_policy=(
            "validation_loss_v0"
            if selected_step == TARGET_STAGE_TRAINING_CONFIG.pilot_step
            else "classifier_recovery_lexicographic_validation_v0"
        ),
        recovery_amendment_fingerprint=(
            None
            if selected_step == TARGET_STAGE_TRAINING_CONFIG.pilot_step
            else amendment.amendment_fingerprint
        ),
    )
    recomputed_outputs = collect_factorized_validation_outputs(
        model=fresh_model, batches=validation_batches
    )
    recomputed_metrics = classifier_stage_metrics(recomputed_outputs)
    reload_validated = (
        selected_audit.deterministic_logits_match
        and selected_audit.restored_step == selected_step
        and selected_audit.maximum_absolute_logit_error == 0.0
        and recomputed_metrics.to_dict() == metrics.to_dict()
    )
    if not reload_validated:
        raise TextRouterCommandError(
            "selected checkpoint reload or full validation recompute failed"
        )
    continuation_records = tuple(
        value
        for value in result.training_records
        if value.step > TARGET_STAGE_TRAINING_CONFIG.pilot_step
    )
    finite_training = all(
        math.isfinite(value)
        for record in continuation_records
        for value in (
            record.total_loss,
            record.status_loss,
            record.object_loss,
            record.bin_loss,
            record.learning_rate,
            record.gradient_norm,
            record.encoder_gradient_norm,
            record.status_head_gradient_norm,
            record.object_head_gradient_norm,
            record.bin_head_gradient_norm,
            record.throughput_examples_per_second,
            record.data_loader_latency_seconds,
            record.step_latency_seconds,
        )
    )
    all_heads_received_gradients = bool(continuation_records) and all(
        max(getattr(record, name) for record in continuation_records) > 0.0
        for name in (
            "status_head_gradient_norm",
            "object_head_gradient_norm",
            "bin_head_gradient_norm",
        )
    )
    training_integrity = {
        "finite_losses_and_gradients": finite_training,
        "all_three_heads_received_gradients": all_heads_received_gradients,
        "continuous_global_steps": [record.step for record in result.training_records]
        == list(range(1, result.completed_steps + 1)),
        "completed_steps": result.completed_steps,
        "completed_epochs": result.completed_epochs,
        "maximum_total_epochs": TARGET_STAGE_TRAINING_CONFIG.maximum_epochs,
        "examples_processed": result.training_records[-1].examples_processed,
        "expected_examples_processed": result.completed_epochs * len(train),
        "no_skipped_completed_epoch_examples": result.training_records[-1].examples_processed
        == result.completed_epochs * len(train),
        "resume_checkpoint_role": resume_role,
        "resumed_from_step": result.resumed_from_step,
        "next_step_at_initial_authorization": 30,
    }
    if not all(
        (
            finite_training,
            all_heads_received_gradients,
            training_integrity["continuous_global_steps"],
            training_integrity["no_skipped_completed_epoch_examples"],
            result.completed_epochs <= TARGET_STAGE_TRAINING_CONFIG.maximum_epochs,
        )
    ):
        raise TextRouterCommandError("recovery training integrity gate failed")
    quality_items = {
        "full_task_spec_accuracy_at_least_0_95": metrics.full_task_spec_accuracy >= 0.95,
        "target_object_accuracy_at_least_0_97": metrics.object_accuracy >= 0.97,
        "destination_bin_accuracy_at_least_0_97": metrics.bin_accuracy >= 0.97,
        "false_route_rate_at_most_0_03": metrics.false_route_rate <= 0.03,
        "ambiguous_rejection_recall_at_least_0_90": metrics.ambiguous_rejection_recall >= 0.90,
        "unsupported_rejection_recall_at_least_0_95": metrics.unsupported_rejection_recall >= 0.95,
        "malformed_rejection_recall_at_least_0_95": metrics.malformed_rejection_recall >= 0.95,
        "schema_valid_rate_exactly_1": metrics.schema_valid_rate == 1.0,
        "finite": metrics.finite and finite_training,
        "selected_checkpoint_reload": reload_validated,
        "development_and_final_unaccessed": pilot_forbidden_sealed,
    }
    full_quality_gate_passed = all(quality_items.values())
    selection: TextRouterCalibrationSelection | None = None
    artifact: Path | None = None
    artifact_fingerprint: str | None = None
    artifact_reload_validated = False
    artifact_reload_maximum_error: float | None = None
    if full_quality_gate_passed:
        selection = calibrate_and_select_text_router(
            outputs=recomputed_outputs,
            threshold_candidates=THRESHOLD_CANDIDATES,
            maximum_false_route_rate=MAXIMUM_FALSE_ROUTE_RATE,
            temperature_iterations=TEMPERATURE_ITERATIONS,
        )
        manifest = classifier_manifest(
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            tokenizer_revision=TOKENIZER_REVISION,
            hidden_size=fresh_model.hidden_size,
            dropout=fresh_model.dropout_probability,
        )
        legacy_result = {
            "completed_steps": result.completed_steps,
            "completed_epochs": result.completed_epochs,
            "stopped_early": result.stopped_early,
            "selected_checkpoint_id": f"step_{selected_step:08d}",
            "selected_step": selected_step,
            "checkpoint_validations": result.to_dict()["validation_records"],
            "validation_rankings": result.to_dict()["validation_rankings"],
            "config_fingerprint": TARGET_STAGE_TRAINING_CONFIG.fingerprint,
            "resumed_from_step": result.resumed_from_step,
        }
        evidence = _run_evidence(
            mode="target_pilot_recovery_complete",
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
                "recovery_amendment": amendment.to_dict(),
                "classifier_training_metrics": metrics.to_dict(),
                "classifier_full_quality_gate_passed": True,
            }
        )
        artifact_fingerprint = text_classifier_run_fingerprint(
            classifier_manifest=manifest, run_evidence=evidence
        )
        artifact = stage_and_promote_text_classifier(
            output_root=output_root / "promoted",
            run_fingerprint=artifact_fingerprint,
            model=fresh_model,
            tokenizer=tokenizer,
            classifier_manifest=manifest,
            run_evidence=evidence,
            clean_matching_staging=bool(args.clean_staging),
        )
        validate_completed_text_classifier_artifact(artifact)
        reloaded_model, reloaded_tokenizer, _manifest = load_factorized_text_classifier(artifact)
        reloaded_model.to("cuda")
        reload_batches = _factorized_batches(
            tokenizer=reloaded_tokenizer,
            examples=validation[:TARGET_BATCH_SIZE],
            split=LanguageSplit.VALIDATION,
            batch_size=TARGET_BATCH_SIZE,
            maximum_sequence_length=TARGET_MAXIMUM_SEQUENCE_LENGTH,
            seed=0,
        )
        runtime_outputs = collect_factorized_validation_outputs(
            model=reloaded_model, batches=reload_batches
        )
        reference_outputs = collect_factorized_validation_outputs(
            model=fresh_model, batches=(validation_batches[0],)
        )
        artifact_reload_maximum_error = max(
            float((left - right).abs().max())
            for left, right in (
                (runtime_outputs.status_logits, reference_outputs.status_logits),
                (runtime_outputs.object_logits, reference_outputs.object_logits),
                (runtime_outputs.bin_logits, reference_outputs.bin_logits),
            )
        )
        artifact_reload_validated = artifact_reload_maximum_error <= 1e-6
        if not artifact_reload_validated:
            raise TextRouterCommandError("promoted runtime artifact reload logits differ")
        del reloaded_model
    del fresh_model
    torch.cuda.empty_cache()
    checkpoint_inventory = {
        role: {
            "path": str(run_root / "checkpoints" / f"{role}.pt"),
            "sha256": _sha256_file(run_root / "checkpoints" / f"{role}.pt"),
            "size_bytes": (run_root / "checkpoints" / f"{role}.pt").stat().st_size,
        }
        for role in CheckpointRetentionPolicy().retained_roles
    }
    if (
        checkpoint_inventory["pilot"]["sha256"] != AUTHORIZED_RECOVERY_PILOT_CHECKPOINT_SHA256
        or any((run_root / "checkpoints").glob(".*.tmp"))
        or len(tuple((run_root / "checkpoints").glob("*.pt"))) != 3
    ):
        raise TextRouterCommandError("bounded checkpoint retention or pilot immutability failed")
    epoch_metrics = _read_json_mapping(epoch_metrics_path, label="recovery epoch metrics")
    payload: dict[str, object] = {
        "schema_version": "langmani-m5a-classifier-pilot-recovery-complete-v0",
        "run_fingerprint": run_fingerprint,
        "run_root": str(run_root),
        "amendment_path": str(amendment_path),
        "amendment_fingerprint": amendment.amendment_fingerprint,
        "protocol_amendment": amendment.to_dict(),
        "pre_resume_audit_path": str(audit_path),
        "pre_resume_audit": audit_payload,
        "pre_resume_checkpoint_audit": pre_resume_checkpoint_audit.to_dict(),
        "classifier_tiny_overfit_validated": True,
        "classifier_pilot_completed": True,
        "classifier_pilot_promoted": False,
        "classifier_recovery_resume_authorized": True,
        "classifier_recovery_resume_completed": True,
        "classifier_training_completed": True,
        "classifier_checkpoint_selected": True,
        "classifier_full_quality_gate_passed": full_quality_gate_passed,
        "classifier_calibration_validated": selection is not None and artifact_reload_validated,
        "artifact_reload_validated": artifact_reload_validated,
        "cuda_training_validated": True,
        "physical_target_validated": True,
        "training_seed": 0,
        "classifier_training_seeds": 1,
        "robustness_across_training_seeds_not_evaluated": True,
        "resume_checkpoint_fingerprint": AUTHORIZED_RECOVERY_PILOT_CHECKPOINT_SHA256,
        "initial_resume_step": 29,
        "initial_next_global_step": 30,
        "training_result": result.to_dict(),
        "training_integrity": training_integrity,
        "epoch_validation_evidence_path": str(epoch_metrics_path),
        "epoch_validation_evidence": epoch_metrics,
        "selected_epoch": selected_step // TARGET_STAGE_TRAINING_CONFIG.steps_per_epoch,
        "selected_step": selected_step,
        "selected_checkpoint_path": str(selected_checkpoint),
        "selected_checkpoint_fingerprint": selected_checkpoint_sha256,
        "selected_checkpoint_reload_audit": selected_audit.to_dict(),
        "selected_checkpoint_reload_maximum_absolute_error": (
            selected_audit.maximum_absolute_logit_error
        ),
        "selected_validation_metrics": metrics.to_dict(),
        "selected_detailed_validation_metrics": selected_details,
        "quality_gate_items": quality_items,
        "calibration_selection": None if selection is None else selection.to_dict(),
        "artifact_run_fingerprint": artifact_fingerprint,
        "artifact_root": None if artifact is None else str(artifact),
        "artifact_reload_maximum_absolute_error": artifact_reload_maximum_error,
        "checkpoint_inventory": checkpoint_inventory,
        "checkpoint_retention_validated": True,
        "elapsed_seconds": time.perf_counter() - stage_started,
        "language_development_completed": False,
        "one_scene_control_smoke_completed": False,
        "three_scene_control_screen_completed": False,
        "predicted_control_development_completed": False,
        "final_benchmark_authorized": False,
        "development_accessed": False,
        "language_final_accessed": False,
        "control_final_accessed": False,
        "m42_final_accessed": False,
        "test_split_accessed": False,
        "historical_fresh_accessed": False,
        "smolvla_go": False,
        "passed": True,
    }
    _write_or_validate_stage(completion_path, payload)
    return payload


def _run_authoritative_stage(
    *,
    args: argparse.Namespace,
    output_root: Path,
    git_state: object,
    resume: bool,
) -> dict[str, object]:
    stage_started = time.perf_counter()
    _validate_target_prerequisites(git_state)
    prior_evidence = _prior_classifier_stage_evidence()
    corpus = build_language_corpus()
    train = corpus.examples_for_split(LanguageSplit.TRAIN)
    validation = corpus.examples_for_split(LanguageSplit.VALIDATION)
    if len(train) != TARGET_STAGE_TRAINING_CONFIG.train_example_count:
        raise TextRouterCommandError("authoritative train count differs from the frozen config")
    identity, run_fingerprint = _authoritative_run_identity(
        corpus_manifest=corpus.manifest.to_dict(), git_state=git_state
    )
    run_root = output_root / "authoritative-runs" / run_fingerprint.removeprefix("sha256:")
    checkpoint_processor_state = _authoritative_checkpoint_processor_state(
        corpus_manifest=corpus.manifest.to_dict(),
        git_state=git_state,
        run_fingerprint=run_fingerprint,
        run_root=run_root,
    )
    preflight = _preflight_payload(
        corpus_manifest=corpus.manifest.to_dict(),
        train=train,
        validation=validation,
        git_state=git_state,
        run_fingerprint=run_fingerprint,
        run_root=run_root,
        prior_evidence=prior_evidence,
        checkpoint_processor_state=checkpoint_processor_state,
    )
    run_root.mkdir(parents=True, exist_ok=True)
    _write_or_validate_stage(run_root / "owner.json", identity)
    _write_or_validate_stage(run_root / "preflight.json", preflight)
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
        processor_state=checkpoint_processor_state,
        stop_after_pilot=not resume,
        resume=resume,
    )
    if not resume:
        metrics = result.pilot_metrics
        pilot_outputs = result.pilot_outputs
        if metrics is None or pilot_outputs is None:
            raise TextRouterCommandError("pilot stage did not produce validation metrics")
        detailed_validation = _detailed_validation_metrics(
            outputs=pilot_outputs,
            examples=validation,
        )
        first_fixture = collect_factorized_validation_outputs(
            model=model, batches=(validation_batches[0],)
        )
        repeated_fixture = collect_factorized_validation_outputs(
            model=model, batches=(validation_batches[0],)
        )
        deterministic_repeatability = all(
            torch.equal(left, right)
            for left, right in (
                (first_fixture.status_logits, repeated_fixture.status_logits),
                (first_fixture.object_logits, repeated_fixture.object_logits),
                (first_fixture.bin_logits, repeated_fixture.bin_logits),
            )
        )
        if not deterministic_repeatability:
            raise TextRouterCommandError(
                "pilot validation fixture is not deterministically repeatable"
            )
        reload_fixture = factorized_validation_fixture_payload(first_fixture)
        training_records = result.training_records
        finite_training = all(
            math.isfinite(value)
            for record in training_records
            for value in (
                record.total_loss,
                record.status_loss,
                record.object_loss,
                record.bin_loss,
                record.gradient_norm,
                record.encoder_gradient_norm,
                record.status_head_gradient_norm,
                record.object_head_gradient_norm,
                record.bin_head_gradient_norm,
                record.learning_rate,
                record.step_latency_seconds,
            )
        )
        head_gradient_evidence = {
            "encoder": max(record.encoder_gradient_norm for record in training_records),
            "status_head": max(record.status_head_gradient_norm for record in training_records),
            "object_head": max(record.object_head_gradient_norm for record in training_records),
            "bin_head": max(record.bin_head_gradient_norm for record in training_records),
        }
        all_heads_received_gradients = all(value > 0.0 for value in head_gradient_evidence.values())
        examples_processed = training_records[-1].examples_processed
        no_silently_skipped_examples = len(
            training_records
        ) == TARGET_STAGE_TRAINING_CONFIG.pilot_step and examples_processed == len(train)
        training_integrity = {
            "finite_losses_and_gradients": finite_training,
            "all_heads_received_gradients": all_heads_received_gradients,
            "head_maximum_gradient_norms": head_gradient_evidence,
            "optimizer_steps": len(training_records),
            "examples_processed": examples_processed,
            "expected_examples_processed": len(train),
            "no_silently_skipped_examples": no_silently_skipped_examples,
            "peak_gpu_allocated_bytes": max(
                record.gpu_allocated_bytes for record in training_records
            ),
            "peak_gpu_reserved_bytes": max(
                record.gpu_reserved_bytes for record in training_records
            ),
            "total_step_seconds": sum(record.step_latency_seconds for record in training_records),
        }
        if not (finite_training and all_heads_received_gradients and no_silently_skipped_examples):
            raise TextRouterCommandError("pilot training integrity gate failed")
        del model
        torch.cuda.empty_cache()
        fresh_model, fresh_tokenizer = _target_model_and_tokenizer(local_files_only=True)
        fresh_model.to("cuda")
        fresh_validation_batches = _factorized_batches(
            tokenizer=fresh_tokenizer,
            examples=validation,
            split=LanguageSplit.VALIDATION,
            batch_size=TARGET_BATCH_SIZE,
            maximum_sequence_length=TARGET_MAXIMUM_SEQUENCE_LENGTH,
            seed=0,
        )
        checkpoint_audit = audit_staged_factorized_text_checkpoint(
            model=fresh_model,
            validation_batches=fresh_validation_batches,
            config=TARGET_STAGE_TRAINING_CONFIG,
            run_fingerprint=run_fingerprint,
            checkpoint_path=Path(result.latest_checkpoint),
            processor_state=checkpoint_processor_state,
            expected_fixture=reload_fixture,
            expected_step=TARGET_STAGE_TRAINING_CONFIG.pilot_step,
        )
        del fresh_model
        torch.cuda.empty_cache()
        checkpoint_audit_passed = all(
            (
                checkpoint_audit.optimizer_state_restored,
                checkpoint_audit.scheduler_state_restored,
                checkpoint_audit.processor_state_restored,
                checkpoint_audit.rng_state_restored,
                checkpoint_audit.data_progress_restored,
                checkpoint_audit.deterministic_logits_match,
                checkpoint_audit.pilot_complete,
                not checkpoint_audit.full_training_complete,
                checkpoint_audit.resumable,
                checkpoint_audit.restored_step == TARGET_STAGE_TRAINING_CONFIG.pilot_step,
                checkpoint_audit.next_global_step == TARGET_STAGE_TRAINING_CONFIG.pilot_step + 1,
            )
        )
        if not checkpoint_audit_passed:
            raise TextRouterCommandError("pilot checkpoint did not pass the resumability audit")
        checkpoint_paths = {
            "pilot": Path(result.pilot_checkpoint),
            "latest": Path(result.latest_checkpoint),
            "validation_best": Path(result.best_checkpoint),
        }
        checkpoint_inventory = {
            role: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "real_file": path.is_file() and not path.is_symlink(),
                "sha256": _sha256_file(path) if role == "pilot" else None,
            }
            for role, path in checkpoint_paths.items()
        }
        checkpoint_atomicity_validated = all(
            cast(Mapping[str, object], value)["real_file"] is True
            for value in checkpoint_inventory.values()
        ) and not any((run_root / "checkpoints").glob(".*.tmp"))
        promoted = (
            metrics.pilot_gate_passed
            and checkpoint_audit_passed
            and deterministic_repeatability
            and finite_training
            and all_heads_received_gradients
            and no_silently_skipped_examples
            and checkpoint_atomicity_validated
        )
        payload = {
            "schema_version": "langmani-m5a-classifier-pilot-v0",
            "run_fingerprint": run_fingerprint,
            "run_root": str(run_root),
            "preflight_path": str(run_root / "preflight.json"),
            "preflight": preflight,
            "classifier_pilot_completed": True,
            "classifier_pilot_promoted": promoted,
            "classifier_fixture_validated": True,
            "classifier_tiny_overfit_validated": True,
            "classifier_training_completed": False,
            "classifier_checkpoint_selected": False,
            "classifier_calibration_validated": False,
            "artifact_reload_validated": checkpoint_audit_passed,
            "cuda_training_validated": True,
            "physical_target_validated": True,
            "pilot_step": TARGET_STAGE_TRAINING_CONFIG.pilot_step,
            "pilot_complete": True,
            "full_training_complete": False,
            "resumable": checkpoint_audit.resumable,
            "same_authoritative_run_required_for_resume": True,
            "metrics": metrics.to_dict(),
            "detailed_validation_metrics": detailed_validation,
            "training_result": result.to_dict(),
            "training_integrity": training_integrity,
            "checkpoint_inventory": checkpoint_inventory,
            "checkpoint_atomicity_validated": checkpoint_atomicity_validated,
            "reload_fixture": reload_fixture,
            "checkpoint_reload_audit": checkpoint_audit.to_dict(),
            "deterministic_repeatability": deterministic_repeatability,
            "elapsed_seconds": time.perf_counter() - stage_started,
            "classifier_training_seeds": 1,
            "robustness_across_training_seeds_not_evaluated": True,
            "development_accessed": False,
            "language_final_accessed": False,
            "control_final_accessed": False,
            "m42_final_accessed": False,
            "test_split_accessed": False,
            "historical_fresh_accessed": False,
            "smolvla_go": False,
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
        else "target_pilot_recovery_resume"
        if args.target_pilot_recovery_resume
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
        "classifier_recovery_resume_authorized": False,
        "classifier_recovery_resume_completed": False,
        "classifier_full_quality_gate_passed": False,
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
        recovery_arguments_present = any(
            value is not None
            for value in (
                args.recovery_run_fingerprint,
                args.recovery_pilot_checkpoint_sha256,
                args.recovery_maximum_total_epochs,
                args.recovery_early_stopping_patience,
            )
        )
        if recovery_arguments_present and not args.target_pilot_recovery_resume:
            raise TextRouterCommandError(
                "recovery continuation arguments require --target-pilot-recovery-resume"
            )
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
        elif args.target_pilot_recovery_resume:
            payload.update(
                _run_authorized_recovery_stage(
                    args=args,
                    output_root=output_root,
                    git_state=git_state,
                )
            )
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
            "classifier_recovery_resume_authorized",
            "classifier_recovery_resume_completed",
            "classifier_full_quality_gate_passed",
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
