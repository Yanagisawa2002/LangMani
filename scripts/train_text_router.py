"""Train the single pinned M5A factorized DistilBERT text router.

The real ``--target-development`` path uses corpus train examples for gradients
and corpus validation examples for checkpoint selection, temperature scaling,
and routing-threshold selection.  Development and sealed-final command text are
never materialized by this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import traceback
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast

import torch

from langmani.language.corpus import build_language_corpus
from langmani.language.router_types import LanguageExample, LanguageSplit, RouterStatus
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
    collect_factorized_validation_outputs,
    run_bounded_factorized_text_training,
    stage_and_promote_text_classifier,
    text_classifier_run_fingerprint,
)
from langmani.policies.act_runtime import atomic_write_json, inspect_git_state

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "models" / "text-router"
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "diagnostics" / "m5a" / "train_text_router.json"
MODEL_ID = TEXT_CLASSIFIER_MODEL_ID
MODEL_REVISION = TEXT_CLASSIFIER_MODEL_REVISION
TOKENIZER_REVISION = TEXT_CLASSIFIER_TOKENIZER_REVISION
TARGET_TRAINING_CONFIG = TextTrainingConfig(
    maximum_steps=600,
    checkpoint_steps=(100, 200, 300, 400, 500, 600),
    learning_rate=2e-5,
    seed=0,
    weight_decay=0.01,
    maximum_gradient_norm=1.0,
    early_stopping_patience=3,
    encoder_trainable=True,
)
TARGET_BATCH_SIZE = 32
TARGET_MAXIMUM_SEQUENCE_LENGTH = 64
TARGET_DROPOUT = 0.1
THRESHOLD_CANDIDATES = tuple(index / 100.0 for index in range(0, 101, 5))
MAXIMUM_FALSE_ROUTE_RATE = 0.03
TEMPERATURE_ITERATIONS = 64
COMMAND_SCHEMA = "langmani-m5a-train-text-router-command-v1"
RUN_EVIDENCE_SCHEMA = "langmani-m5a-text-router-run-evidence-v1"

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
    modes.add_argument("--target-development", action="store_true")
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
            "optimization": TARGET_TRAINING_CONFIG.to_dict(),
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
        "development_examples_materialized": False,
        "final_examples_materialized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mode = (
        "target_development"
        if args.target_development
        else "fixture"
        if args.fixture
        else "dry_run"
    )
    payload: dict[str, object] = {
        "schema_version": COMMAND_SCHEMA,
        "passed": False,
        "mode": mode,
        "classifier_training_completed": False,
        "classifier_fixture_completed": False,
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
        if args.dry_run:
            payload["plan"] = _dry_run_payload(git_state)
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
                    "classifier_training_completed": bool(args.target_development),
                    "classifier_fixture_completed": bool(args.fixture),
                    "classifier_checkpoint_selected": True,
                    "classifier_calibration_validated": True,
                    "artifact_reload_validated": True,
                    "cuda_training_validated": bool(args.target_development),
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
            "classifier_fixture_completed",
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
