"""Validation-only corpus, logit, confusion, and error diagnostics for M5A.1."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import nn

from langmani.environments.specs import TaskSpec
from langmani.language.classifier_decoders import (
    DecoderConfigurationV0,
    decode_classifier_probabilities,
)
from langmani.language.router_types import (
    LanguageExample,
    LanguageSplit,
    LanguageTemplateFamily,
    RouterRejectionReason,
    RouterStatus,
)
from langmani.language.splits import (
    build_language_split_manifest,
    validate_family_split_isolation,
)
from langmani.language.text_classifier import (
    BIN_LABELS,
    OBJECT_LABELS,
    STATUS_LABELS,
    FactorizedTextClassifierV0,
)
from langmani.language.text_training import (
    FactorizedTextBatch,
    collect_factorized_validation_outputs,
)

REJECTION_DIAGNOSTICS_SCHEMA = "langmani-m5a-rejection-diagnostics-v0"
DATA_CONTRACT_AUDIT_SCHEMA = "langmani-m5a-rejection-data-contract-audit-v0"


class RejectionDiagnosticsError(RuntimeError):
    """Raised when validation diagnostics encounter malformed or prohibited evidence."""


@dataclass(frozen=True, slots=True)
class ValidationCorpusInputsV0:
    train_examples: tuple[LanguageExample, ...]
    validation_examples: tuple[LanguageExample, ...]
    visible_families: tuple[LanguageTemplateFamily, ...]
    corpus_fingerprint: str
    train_split_fingerprint: str
    validation_split_fingerprint: str
    archive_fingerprint: str
    corpus_root: str


@dataclass(frozen=True, slots=True)
class ValidationLogitEvidenceV0:
    examples: tuple[LanguageExample, ...]
    status_logits: torch.Tensor
    object_logits: torch.Tensor
    bin_logits: torch.Tensor
    checkpoint_fingerprint: str
    checkpoint_step: int
    repeat_maximum_absolute_error: float
    batch_latency_seconds: tuple[float, ...]

    def __post_init__(self) -> None:
        count = len(self.examples)
        if not self.examples or any(
            example.split is not LanguageSplit.VALIDATION for example in self.examples
        ):
            raise RejectionDiagnosticsError("logit evidence requires validation examples only")
        if (
            tuple(self.status_logits.shape) != (count, len(STATUS_LABELS))
            or tuple(self.object_logits.shape) != (count, len(OBJECT_LABELS))
            or tuple(self.bin_logits.shape) != (count, len(BIN_LABELS))
            or not all(
                bool(torch.isfinite(value).all())
                for value in (self.status_logits, self.object_logits, self.bin_logits)
            )
        ):
            raise RejectionDiagnosticsError("validation logits have the wrong shape or values")
        if not self.checkpoint_fingerprint.startswith("sha256:") or self.checkpoint_step <= 0:
            raise RejectionDiagnosticsError("validation logits require a checkpoint identity")
        if (
            not math.isfinite(self.repeat_maximum_absolute_error)
            or self.repeat_maximum_absolute_error < 0.0
        ):
            raise RejectionDiagnosticsError("repeatability error must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ReadOnlyClassifierBundleV0:
    model: FactorizedTextClassifierV0
    tokenizer: object
    processor_state: Mapping[str, object]
    run_fingerprint: str
    checkpoint_fingerprint: str
    checkpoint_step: int
    checkpoint_size_bytes: int
    model_state_fingerprint: str
    tokenizer_fingerprint: str
    processor_fingerprint: str


def _real_path(path: str | Path, *, label: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    for component in (lexical, *lexical.parents):
        if component.is_symlink() or component.is_junction():
            raise RejectionDiagnosticsError(f"{label} traverses a linked path: {component}")
    return lexical.resolve(strict=False)


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RejectionDiagnosticsError(f"could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise RejectionDiagnosticsError(f"{path} must contain one JSON object")
    return cast(dict[str, object], value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _model_state_fingerprint(model: nn.Module) -> str:
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


def load_selected_checkpoint_read_only(
    checkpoint_path: str | Path,
    *,
    expected_checkpoint_fingerprint: str,
    expected_run_fingerprint: str,
    expected_step: int,
    local_files_only: bool,
) -> ReadOnlyClassifierBundleV0:
    """Load model state directly without constructing an optimizer or scheduler."""

    path = _real_path(checkpoint_path, label="selected classifier checkpoint")
    if not path.is_file():
        raise RejectionDiagnosticsError("selected classifier checkpoint is missing")
    actual_fingerprint = _sha256_file(path)
    if actual_fingerprint != expected_checkpoint_fingerprint:
        raise RejectionDiagnosticsError("selected classifier checkpoint SHA-256 differs")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise RejectionDiagnosticsError("could not load selected classifier checkpoint") from error
    if not isinstance(payload, Mapping):
        raise RejectionDiagnosticsError("selected classifier checkpoint must contain one mapping")
    processor_state = payload.get("processor_state")
    if (
        payload.get("schema_version") != "langmani-m5a-resumable-text-checkpoint-v0"
        or payload.get("role") != "validation_best"
        or payload.get("run_fingerprint") != expected_run_fingerprint
        or payload.get("step") != expected_step
        or not isinstance(processor_state, Mapping)
    ):
        raise RejectionDiagnosticsError("selected classifier checkpoint identity differs")
    model_contract = processor_state.get("model")
    tokenizer_contract = processor_state.get("tokenizer")
    if not isinstance(model_contract, Mapping) or not isinstance(tokenizer_contract, Mapping):
        raise RejectionDiagnosticsError("checkpoint model/tokenizer contract is malformed")
    model_id = model_contract.get("model_id")
    model_revision = model_contract.get("model_revision")
    tokenizer_revision = tokenizer_contract.get("tokenizer_revision")
    dropout = model_contract.get("dropout")
    if (
        model_contract.get("architecture") != "FactorizedTextClassifierV0"
        or not isinstance(model_id, str)
        or not isinstance(model_revision, str)
        or not isinstance(tokenizer_revision, str)
        or isinstance(dropout, bool)
        or not isinstance(dropout, int | float)
    ):
        raise RejectionDiagnosticsError("checkpoint classifier architecture contract differs")
    try:
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        encoder_config = AutoConfig.from_pretrained(
            model_id,
            revision=model_revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        encoder = AutoModel.from_config(encoder_config, trust_remote_code=False)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=tokenizer_revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
            use_fast=True,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise RejectionDiagnosticsError(
            f"could not reconstruct pinned classifier metadata: {error}"
        ) from error
    if not isinstance(encoder, nn.Module):
        raise RejectionDiagnosticsError("pinned AutoModel configuration returned no module")
    model = FactorizedTextClassifierV0(encoder, dropout=float(dropout))
    model_state = payload.get("model_state")
    if not isinstance(model_state, Mapping):
        raise RejectionDiagnosticsError("selected checkpoint model state is malformed")
    try:
        model.load_state_dict(cast(Mapping[str, torch.Tensor], model_state), strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise RejectionDiagnosticsError("selected classifier model state differs") from error
    del payload
    processor_payload = dict(processor_state)
    model_fingerprint = _model_state_fingerprint(model)
    return ReadOnlyClassifierBundleV0(
        model=model,
        tokenizer=tokenizer,
        processor_state=processor_payload,
        run_fingerprint=expected_run_fingerprint,
        checkpoint_fingerprint=actual_fingerprint,
        checkpoint_step=expected_step,
        checkpoint_size_bytes=path.stat().st_size,
        model_state_fingerprint=model_fingerprint,
        tokenizer_fingerprint=f"sha256:{hashlib.sha256(json.dumps(dict(tokenizer_contract), sort_keys=True, separators=(',', ':')).encode()).hexdigest()}",
        processor_fingerprint=f"sha256:{hashlib.sha256(json.dumps(processor_payload, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()}",
    )


def _label_indices(example: LanguageExample) -> tuple[int, int, int]:
    status_index = STATUS_LABELS.index(example.expected_status.value)
    if example.expected_status is not RouterStatus.ROUTE:
        return status_index, -1, -1
    task_spec = example.expected_task_spec
    if task_spec is None:
        raise RejectionDiagnosticsError("routeable validation example is missing a TaskSpec")
    return (
        status_index,
        OBJECT_LABELS.index(task_spec.target_object_id),
        BIN_LABELS.index(task_spec.target_bin_id),
    )


def _validation_batches(
    *,
    tokenizer: object,
    examples: Sequence[LanguageExample],
    batch_size: int,
    maximum_sequence_length: int,
) -> tuple[FactorizedTextBatch, ...]:
    if not callable(tokenizer) or batch_size <= 0 or maximum_sequence_length <= 0:
        raise RejectionDiagnosticsError("validation tokenizer configuration is malformed")
    values = tuple(examples)
    batches: list[FactorizedTextBatch] = []
    for start in range(0, len(values), batch_size):
        selected = values[start : start + batch_size]
        encoded = tokenizer(
            [example.raw_text for example in selected],
            padding=True,
            truncation=True,
            max_length=maximum_sequence_length,
            return_tensors="pt",
        )
        if not isinstance(encoded, Mapping):
            raise RejectionDiagnosticsError("tokenizer output must be a mapping")
        input_ids = encoded.get("input_ids")
        attention_mask = encoded.get("attention_mask")
        if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
            raise RejectionDiagnosticsError("tokenizer omitted input_ids or attention_mask")
        labels = tuple(_label_indices(example) for example in selected)
        batches.append(
            FactorizedTextBatch(
                input_ids=input_ids.to(dtype=torch.long),
                attention_mask=attention_mask.to(dtype=torch.long),
                status_labels=torch.tensor([value[0] for value in labels], dtype=torch.long),
                object_labels=torch.tensor([value[1] for value in labels], dtype=torch.long),
                bin_labels=torch.tensor([value[2] for value in labels], dtype=torch.long),
                split="validation",
            )
        )
    return tuple(batches)


def collect_read_only_validation_logits(
    bundle: ReadOnlyClassifierBundleV0,
    *,
    examples: Sequence[LanguageExample],
    batch_size: int = 32,
) -> tuple[ValidationLogitEvidenceV0, dict[str, object]]:
    tokenizer_contract = bundle.processor_state.get("tokenizer")
    if not isinstance(tokenizer_contract, Mapping):
        raise RejectionDiagnosticsError("checkpoint tokenizer contract is missing")
    maximum_length = tokenizer_contract.get("maximum_sequence_length")
    if isinstance(maximum_length, bool) or not isinstance(maximum_length, int):
        raise RejectionDiagnosticsError("checkpoint sequence length is malformed")
    batches = _validation_batches(
        tokenizer=bundle.tokenizer,
        examples=examples,
        batch_size=batch_size,
        maximum_sequence_length=maximum_length,
    )
    model_before = bundle.model_state_fingerprint
    started = time.perf_counter()
    first = collect_factorized_validation_outputs(model=bundle.model, batches=batches)
    first_elapsed = time.perf_counter() - started
    second = collect_factorized_validation_outputs(model=bundle.model, batches=batches)
    errors = tuple(
        float(torch.max(torch.abs(left - right)))
        for left, right in (
            (first.status_logits, second.status_logits),
            (first.object_logits, second.object_logits),
            (first.bin_logits, second.bin_logits),
        )
    )
    maximum_error = max(errors)
    if maximum_error != 0.0:
        raise RejectionDiagnosticsError("selected-checkpoint validation logits are not exact")
    model_after = _model_state_fingerprint(bundle.model)
    if model_before != model_after:
        raise RejectionDiagnosticsError("classifier weights changed during read-only inference")
    return (
        ValidationLogitEvidenceV0(
            examples=tuple(examples),
            status_logits=first.status_logits,
            object_logits=first.object_logits,
            bin_logits=first.bin_logits,
            checkpoint_fingerprint=bundle.checkpoint_fingerprint,
            checkpoint_step=bundle.checkpoint_step,
            repeat_maximum_absolute_error=maximum_error,
            batch_latency_seconds=first.batch_latency_seconds,
        ),
        {
            "model_state_fingerprint_before": model_before,
            "model_state_fingerprint_after": model_after,
            "model_weights_unchanged": model_before == model_after,
            "checkpoint_step_before": bundle.checkpoint_step,
            "checkpoint_step_after": bundle.checkpoint_step,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "training_seed_count": 1,
            "new_training_run_created": False,
            "validation_examples": len(examples),
            "first_pass_elapsed_seconds": first_elapsed,
            "repeat_maximum_absolute_error": maximum_error,
        },
    )


def _parse_example(value: Mapping[str, object]) -> LanguageExample:
    expected_task = value.get("expected_task_spec")
    expected_reason = value.get("expected_rejection_reason")
    provenance = value.get("generation_provenance")
    lexical = value.get("lexical_variant_ids")
    if not isinstance(provenance, Mapping) or not isinstance(lexical, list):
        raise RejectionDiagnosticsError("language example provenance or lexical IDs are malformed")
    try:
        return LanguageExample(
            example_id=cast(str, value["example_id"]),
            raw_text=cast(str, value["raw_text"]),
            normalized_text=cast(str, value["normalized_text"]),
            template_family_id=cast(str, value["template_family_id"]),
            lexical_variant_ids=tuple(cast(list[str], lexical)),
            expected_status=RouterStatus(cast(str, value["expected_status"])),
            expected_task_spec=(
                None
                if expected_task is None
                else TaskSpec.from_mapping(cast(Mapping[str, object], expected_task))
            ),
            expected_rejection_reason=(
                None
                if expected_reason is None
                else RouterRejectionReason(cast(str, expected_reason))
            ),
            generation_provenance=provenance,
            split=LanguageSplit(cast(str, value["split"])),
            schema_version=cast(str, value["schema_version"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RejectionDiagnosticsError("language example schema validation failed") from error


def _read_examples(path: Path, *, split: LanguageSplit) -> tuple[LanguageExample, ...]:
    examples: list[LanguageExample] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            for line in stream:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RejectionDiagnosticsError("language JSONL row must be an object")
                example = _parse_example(value)
                if example.split is not split:
                    raise RejectionDiagnosticsError("language JSONL row crosses its declared split")
                examples.append(example)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RejectionDiagnosticsError(f"could not read {split.value} JSONL: {error}") from error
    return tuple(examples)


def _parse_family(value: Mapping[str, object]) -> LanguageTemplateFamily:
    provenance = value.get("generation_provenance")
    if not isinstance(provenance, Mapping):
        raise RejectionDiagnosticsError("language family provenance is malformed")
    try:
        return LanguageTemplateFamily(
            family_id=cast(str, value["family_id"]),
            split=LanguageSplit(cast(str, value["split"])),
            structural_signature=cast(str, value["structural_signature"]),
            near_duplicate_group_id=cast(str, value["near_duplicate_group_id"]),
            category=cast(str, value["category"]),
            generation_provenance=provenance,
            schema_version=cast(str, value["schema_version"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RejectionDiagnosticsError("language family schema validation failed") from error


def load_validation_corpus_inputs(corpus_root: str | Path) -> ValidationCorpusInputsV0:
    """Load train/validation content without opening development or final JSONL."""

    root = _real_path(corpus_root, label="M5A language corpus")
    if not root.is_dir():
        raise RejectionDiagnosticsError("M5A language corpus root is missing")
    artifact = _read_object(root / "artifact_manifest.json")
    complete = _read_object(root / "complete.json")
    corpus_manifest = _read_object(root / "corpus_manifest.json")
    family_manifest = _read_object(root / "family_manifest.json")
    records = artifact.get("artifacts")
    if not isinstance(records, list):
        raise RejectionDiagnosticsError("corpus artifact manifest is malformed")
    record_by_path = {
        cast(str, record["path"]): record
        for record in records
        if isinstance(record, Mapping) and isinstance(record.get("path"), str)
    }
    for name in ("corpus_manifest.json", "family_manifest.json", "train.jsonl", "validation.jsonl"):
        record = record_by_path.get(name)
        path = root / name
        if (
            not isinstance(record, Mapping)
            or record.get("size_bytes") != path.stat().st_size
            or record.get("sha256") != _sha256_file(path)
        ):
            raise RejectionDiagnosticsError(f"corpus artifact checksum differs: {name}")
    train = _read_examples(root / "train.jsonl", split=LanguageSplit.TRAIN)
    validation = _read_examples(root / "validation.jsonl", split=LanguageSplit.VALIDATION)
    raw_families = family_manifest.get("families")
    if not isinstance(raw_families, list):
        raise RejectionDiagnosticsError("corpus family manifest is malformed")
    visible_families = tuple(
        family
        for family in (
            _parse_family(cast(Mapping[str, object], value))
            for value in raw_families
            if isinstance(value, Mapping)
        )
        if family.split in {LanguageSplit.TRAIN, LanguageSplit.VALIDATION}
    )
    isolation = validate_family_split_isolation(visible_families, (*train, *validation))
    if not isolation.passed:
        raise RejectionDiagnosticsError("train/validation family isolation failed")
    split_manifests = corpus_manifest.get("split_manifests")
    if not isinstance(split_manifests, Mapping):
        raise RejectionDiagnosticsError("corpus split manifests are missing")
    fingerprints: dict[LanguageSplit, str] = {}
    for split, examples in (
        (LanguageSplit.TRAIN, train),
        (LanguageSplit.VALIDATION, validation),
    ):
        rebuilt = build_language_split_manifest(
            split=split,
            families=visible_families,
            examples=examples,
        ).to_dict()
        stored = split_manifests.get(split.value)
        if stored != rebuilt:
            raise RejectionDiagnosticsError(f"{split.value} split manifest differs")
        fingerprints[split] = cast(str, rebuilt["content_fingerprint"])
    corpus_fingerprint = corpus_manifest.get("corpus_fingerprint")
    archive_fingerprint = artifact.get("archive_fingerprint")
    if (
        not isinstance(corpus_fingerprint, str)
        or not corpus_fingerprint.startswith("sha256:")
        or not isinstance(archive_fingerprint, str)
        or not archive_fingerprint.startswith("sha256:")
        or complete.get("passed") is not True
        or complete.get("corpus_fingerprint") != corpus_fingerprint
        or complete.get("archive_fingerprint") != archive_fingerprint
    ):
        raise RejectionDiagnosticsError("corpus completion identity differs")
    return ValidationCorpusInputsV0(
        train_examples=train,
        validation_examples=validation,
        visible_families=visible_families,
        corpus_fingerprint=corpus_fingerprint,
        train_split_fingerprint=fingerprints[LanguageSplit.TRAIN],
        validation_split_fingerprint=fingerprints[LanguageSplit.VALIDATION],
        archive_fingerprint=archive_fingerprint,
        corpus_root=str(root),
    )


_EXPECTED_STATUS_BY_REASON = {
    RouterRejectionReason.MISSING_OBJECT: RouterStatus.REJECT_AMBIGUOUS,
    RouterRejectionReason.MISSING_DESTINATION: RouterStatus.REJECT_AMBIGUOUS,
    RouterRejectionReason.CONFLICTING_OBJECTS: RouterStatus.REJECT_AMBIGUOUS,
    RouterRejectionReason.CONFLICTING_BINS: RouterStatus.REJECT_AMBIGUOUS,
    RouterRejectionReason.MULTIPLE_TASKS: RouterStatus.REJECT_AMBIGUOUS,
    RouterRejectionReason.UNRESOLVED_CORRECTION: RouterStatus.REJECT_AMBIGUOUS,
    RouterRejectionReason.CONTRADICTORY_NEGATION: RouterStatus.REJECT_AMBIGUOUS,
    RouterRejectionReason.UNSUPPORTED_OBJECT: RouterStatus.REJECT_UNSUPPORTED,
    RouterRejectionReason.UNSUPPORTED_DESTINATION: RouterStatus.REJECT_UNSUPPORTED,
    RouterRejectionReason.UNSUPPORTED_ACTION: RouterStatus.REJECT_UNSUPPORTED,
    RouterRejectionReason.UNSUPPORTED_SPATIAL_REFERENCE: RouterStatus.REJECT_UNSUPPORTED,
    RouterRejectionReason.MEANINGLESS_TEXT: RouterStatus.REJECT_MALFORMED,
    RouterRejectionReason.EMPTY_TEXT: RouterStatus.REJECT_MALFORMED,
    RouterRejectionReason.MALFORMED_CONTROL_CHARACTERS: RouterStatus.REJECT_MALFORMED,
}


def audit_validation_data_contract(
    corpus: ValidationCorpusInputsV0,
    *,
    checkpoint_processor_state: Mapping[str, object],
    recovery_pre_resume_audit: Mapping[str, object],
) -> dict[str, object]:
    train = corpus.train_examples
    validation = corpus.validation_examples
    train_ids = {example.example_id for example in train}
    validation_ids = {example.example_id for example in validation}
    train_text = {example.normalized_text for example in train}
    validation_text = {example.normalized_text for example in validation}
    status_counts = Counter(example.expected_status.value for example in train)
    unsupported_present = status_counts[RouterStatus.REJECT_UNSUPPORTED.value] > 0
    malformed_present = status_counts[RouterStatus.REJECT_MALFORMED.value] > 0
    rejection_labels_correct = all(
        example.expected_rejection_reason in _EXPECTED_STATUS_BY_REASON
        and _EXPECTED_STATUS_BY_REASON[
            cast(RouterRejectionReason, example.expected_rejection_reason)
        ]
        is example.expected_status
        for example in (*train, *validation)
        if example.expected_status is not RouterStatus.ROUTE
    )
    label_mappings = checkpoint_processor_state.get("label_mappings")
    expected_mappings = {
        "status": list(STATUS_LABELS),
        "object": list(OBJECT_LABELS),
        "bin": list(BIN_LABELS),
        "rejected_object_bin_loss_mask": -1,
    }
    labels_match = isinstance(label_mappings, Mapping) and all(
        label_mappings.get(key) == value for key, value in expected_mappings.items()
    )
    prior_checks = recovery_pre_resume_audit.get("checks")
    sampler_complete = (
        isinstance(prior_checks, Mapping) and prior_checks.get("no_sampler_class_exclusion") is True
    )
    checks = {
        "validation_rejection_labels_correct": rejection_labels_correct,
        "unsupported_train_examples_present": unsupported_present,
        "malformed_train_examples_present": malformed_present,
        "template_family_split_isolation": True,
        "near_duplicate_leakage_absent": True,
        "training_history_contains_every_status": sampler_complete,
        "class_indices_match": labels_match,
        "status_label_remapping_absent": labels_match,
        "rejected_object_bin_loss_mask_matches": labels_match,
        "validation_example_ids_disjoint_from_train": train_ids.isdisjoint(validation_ids),
        "validation_normalized_text_disjoint_from_train": train_text.isdisjoint(validation_text),
        "train_count_exact": len(train) == 900,
        "validation_count_exact": len(validation) == 300,
    }
    return {
        "schema_version": DATA_CONTRACT_AUDIT_SCHEMA,
        "passed": all(checks.values()),
        "read_only": True,
        "gradient_split": "train_historical_only",
        "selection_split": "validation",
        "checks": checks,
        "train_counts_by_status": dict(sorted(status_counts.items())),
        "validation_counts_by_status": dict(
            sorted(Counter(example.expected_status.value for example in validation).items())
        ),
        "corpus_fingerprint": corpus.corpus_fingerprint,
        "train_split_fingerprint": corpus.train_split_fingerprint,
        "validation_split_fingerprint": corpus.validation_split_fingerprint,
        "development_content_accessed": False,
        "final_content_accessed": False,
    }


def _entropy(probabilities: Sequence[float]) -> float:
    return -sum(value * math.log(value) for value in probabilities if value > 0.0)


def _top_margin(probabilities: Sequence[float]) -> float:
    ordered = sorted(probabilities, reverse=True)
    return ordered[0] - ordered[1]


def _softmax_rows(logits: torch.Tensor, *, temperature: float) -> list[list[float]]:
    return torch.softmax(logits.detach().cpu().double() / temperature, dim=-1).tolist()


def build_per_example_diagnostics(
    evidence: ValidationLogitEvidenceV0,
    *,
    baseline_configuration: DecoderConfigurationV0,
    selected_configuration: DecoderConfigurationV0 | None,
) -> tuple[dict[str, object], ...]:
    selected = selected_configuration or baseline_configuration
    status_raw = _softmax_rows(evidence.status_logits, temperature=1.0)
    status_selected = _softmax_rows(evidence.status_logits, temperature=selected.status_temperature)
    objects = _softmax_rows(evidence.object_logits, temperature=1.0)
    bins = _softmax_rows(evidence.bin_logits, temperature=1.0)
    records: list[dict[str, object]] = []
    for index, example in enumerate(evidence.examples):
        baseline = decode_classifier_probabilities(
            status_probabilities=status_raw[index],
            object_probabilities=objects[index],
            bin_probabilities=bins[index],
            configuration=baseline_configuration,
        )
        decision = decode_classifier_probabilities(
            status_probabilities=status_selected[index],
            object_probabilities=objects[index],
            bin_probabilities=bins[index],
            configuration=selected,
        )
        predicted_object = OBJECT_LABELS[max(range(4), key=lambda i: objects[index][i])]
        predicted_bin = BIN_LABELS[max(range(3), key=lambda i: bins[index][i])]
        expected_task = example.expected_task_spec
        route_correct = (
            example.expected_status is RouterStatus.ROUTE
            and baseline.status is RouterStatus.ROUTE
            and expected_task is not None
            and baseline.target_object_id == expected_task.target_object_id
            and baseline.target_bin_id == expected_task.target_bin_id
        )
        rejection_normalizer = sum(status_raw[index][1:])
        rejection_normalized = [value / rejection_normalizer for value in status_raw[index][1:]]
        records.append(
            {
                "schema_version": REJECTION_DIAGNOSTICS_SCHEMA,
                "example_id": example.example_id,
                "raw_text": example.raw_text,
                "template_family_id": example.template_family_id,
                "expected_status": example.expected_status.value,
                "expected_rejection_reason": (
                    None
                    if example.expected_rejection_reason is None
                    else example.expected_rejection_reason.value
                ),
                "expected_task_spec": (None if expected_task is None else expected_task.to_dict()),
                "status_logits": evidence.status_logits[index].tolist(),
                "status_probabilities": status_raw[index],
                "selected_temperature_status_probabilities": status_selected[index],
                "normalized_rejection_probabilities": rejection_normalized,
                "object_logits": evidence.object_logits[index].tolist(),
                "object_probabilities": objects[index],
                "bin_logits": evidence.bin_logits[index].tolist(),
                "bin_probabilities": bins[index],
                "baseline_predicted_status": baseline.status.value,
                "baseline_predicted_object": predicted_object,
                "baseline_predicted_bin": predicted_bin,
                "baseline_routed": baseline.status is RouterStatus.ROUTE,
                "baseline_route_correct": route_correct,
                "baseline_predicted_rejection_reason": (
                    None if baseline.status is RouterStatus.ROUTE else baseline.status.value
                ),
                "confidence": {
                    "route_probability": status_raw[index][0],
                    "rejection_probability": rejection_normalizer,
                    "route_reject_margin": status_raw[index][0] - rejection_normalizer,
                    "status_top_two_margin": _top_margin(status_raw[index]),
                    "object_top_two_margin": _top_margin(objects[index]),
                    "bin_top_two_margin": _top_margin(bins[index]),
                    "status_entropy": _entropy(status_raw[index]),
                    "object_entropy": _entropy(objects[index]),
                    "bin_entropy": _entropy(bins[index]),
                },
                "decoder_decision": decision.to_dict(),
            }
        )
    return tuple(records)


_ERROR_GROUPS = (
    "ambiguous_to_route",
    "ambiguous_to_unsupported",
    "ambiguous_to_malformed",
    "unsupported_to_route",
    "unsupported_to_ambiguous",
    "unsupported_to_malformed",
    "malformed_to_route",
    "malformed_to_ambiguous",
    "malformed_to_unsupported",
    "routeable_to_rejection",
)


def _error_group(record: Mapping[str, object]) -> str | None:
    expected = record["expected_status"]
    predicted = record["baseline_predicted_status"]
    if expected == RouterStatus.ROUTE.value and predicted != RouterStatus.ROUTE.value:
        return "routeable_to_rejection"
    names = {
        RouterStatus.REJECT_AMBIGUOUS.value: "ambiguous",
        RouterStatus.REJECT_UNSUPPORTED.value: "unsupported",
        RouterStatus.REJECT_MALFORMED.value: "malformed",
        RouterStatus.ROUTE.value: "route",
    }
    if expected != predicted and expected in names and predicted in names:
        return f"{names[cast(str, expected)]}_to_{names[cast(str, predicted)]}"
    return None


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"minimum": None, "mean": None, "median": None, "maximum": None}
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
    return {
        "minimum": ordered[0],
        "mean": sum(ordered) / len(ordered),
        "median": median,
        "maximum": ordered[-1],
    }


def build_error_inventory(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        group = _error_group(record)
        if group is not None:
            grouped[group].append(record)
    payload: dict[str, object] = {}
    for name in _ERROR_GROUPS:
        values = sorted(grouped[name], key=lambda value: cast(str, value["example_id"]))
        confidences = [cast(Mapping[str, float], value["confidence"]) for value in values]
        payload[name] = {
            "count": len(values),
            "representative_example_ids": [value["example_id"] for value in values[:5]],
            "template_families": sorted(
                {cast(str, value["template_family_id"]) for value in values}
            ),
            "route_probability": _distribution(
                [value["route_probability"] for value in confidences]
            ),
            "rejection_probability": _distribution(
                [value["rejection_probability"] for value in confidences]
            ),
            "object_confidence": _distribution(
                [max(cast(Sequence[float], value["object_probabilities"])) for value in values]
            ),
            "bin_confidence": _distribution(
                [max(cast(Sequence[float], value["bin_probabilities"])) for value in values]
            ),
        }
    rejected = [value for value in records if value["expected_status"] != RouterStatus.ROUTE.value]
    routeable = [value for value in records if value["expected_status"] == RouterStatus.ROUTE.value]
    categories = {
        "rejected_incorrectly_routed": sum(
            value["baseline_predicted_status"] == RouterStatus.ROUTE.value for value in rejected
        ),
        "rejected_with_wrong_rejection_category": sum(
            value["baseline_predicted_status"]
            not in {
                RouterStatus.ROUTE.value,
                value["expected_status"],
            }
            for value in rejected
        ),
        "routeable_incorrectly_rejected": sum(
            value["baseline_predicted_status"] != RouterStatus.ROUTE.value for value in routeable
        ),
        "routeable_routed_with_wrong_task_spec": sum(
            value["baseline_predicted_status"] == RouterStatus.ROUTE.value
            and value["baseline_route_correct"] is not True
            for value in routeable
        ),
    }
    return {
        "schema_version": REJECTION_DIAGNOSTICS_SCHEMA,
        "groups": payload,
        "separated_error_categories": categories,
    }


def build_per_template_family_findings(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        grouped[cast(str, record["template_family_id"])].append(record)
    result: dict[str, object] = {}
    for family, values in sorted(grouped.items()):
        routeable = [
            value for value in values if value["expected_status"] == RouterStatus.ROUTE.value
        ]
        rejected = [
            value for value in values if value["expected_status"] != RouterStatus.ROUTE.value
        ]
        result[family] = {
            "examples": len(values),
            "baseline_exact_status_accuracy": _safe_fraction(
                sum(
                    value["baseline_predicted_status"] == value["expected_status"]
                    for value in values
                ),
                len(values),
            ),
            "baseline_routeable_full_task_accuracy": _safe_fraction(
                sum(value["baseline_route_correct"] is True for value in routeable), len(routeable)
            ),
            "baseline_false_route_rate": _safe_fraction(
                sum(
                    value["baseline_predicted_status"] == RouterStatus.ROUTE.value
                    for value in rejected
                ),
                len(rejected),
            ),
        }
    return result


def _safe_fraction(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


__all__ = [
    "DATA_CONTRACT_AUDIT_SCHEMA",
    "REJECTION_DIAGNOSTICS_SCHEMA",
    "RejectionDiagnosticsError",
    "ReadOnlyClassifierBundleV0",
    "ValidationCorpusInputsV0",
    "ValidationLogitEvidenceV0",
    "audit_validation_data_contract",
    "build_error_inventory",
    "build_per_example_diagnostics",
    "build_per_template_family_findings",
    "load_validation_corpus_inputs",
    "load_selected_checkpoint_read_only",
    "collect_read_only_validation_logits",
]
