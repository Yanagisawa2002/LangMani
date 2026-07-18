"""Evaluate the three locked M5A language routers on validation and development.

This command never opens the language final split, a controller schedule, or a
robot environment.  Target-development uses one immutable classifier artifact
and exactly one explicitly pinned local instruct model.  Test fixtures may
inject a local generator only when ``--fixture`` is selected, and fixture
reports never claim target or physical validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import traceback
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Protocol, cast

import torch

from langmani.datasets.identity import sha256_hex
from langmani.language.corpus import GeneratedLanguageCorpus, build_language_corpus
from langmani.language.llm_router import (
    LocalTextGenerator,
    StructuredLLMRouterConfig,
    StructuredLocalLLMRouterV0,
    TransformersLocalTextGenerator,
    build_structured_routing_prompt,
    select_structured_routing_prompt_examples,
)
from langmani.language.router_evaluation import (
    RouterEvaluationRecord,
    evaluate_language_router,
)
from langmani.language.router_types import (
    LanguageExample,
    LanguageSplit,
    RouterRejectionReason,
    RouterStatus,
)
from langmani.language.rule_router import RuleRouterConfig, RuleRouterV0
from langmani.language.schedules import build_language_schedule_locks
from langmani.language.text_calibration import TemperatureCalibrationV0
from langmani.language.text_classifier import (
    FactorizedTextRouterConfig,
    FactorizedTextRouterV0,
    load_factorized_text_classifier,
)
from langmani.language.text_training import read_text_classifier_run_evidence
from langmani.policies.act_runtime import atomic_write_json, inspect_git_state

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_ROOT = PROJECT_ROOT / "outputs" / "datasets" / "m5a" / "langmani-language-corpus-v1"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "diagnostics" / "m5a" / "language-routing"
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "diagnostics" / "m5a" / "language-routing.json"

COMMAND_SCHEMA = "langmani-m5a-language-router-evaluation-command-v1"
EVIDENCE_SCHEMA = "langmani-m5a-language-router-evaluation-evidence-v1"
CLASSIFIER_RUN_EVIDENCE_SCHEMA = "langmani-m5a-text-router-run-evidence-v2"
CORPUS_ARCHIVE_SCHEMA = "langmani-m5a-language-corpus-archive-v1"
DEFAULT_REPEAT_COUNT = 2

_CLASSIFIER_RUN_EVIDENCE_FIELDS = {
    "schema_version",
    "mode",
    "corpus_manifest",
    "split_fingerprints",
    "git_commit",
    "git_state",
    "dependencies",
    "model_id",
    "model_revision",
    "tokenizer_revision",
    "training_config",
    "training_result",
    "calibration_config",
    "calibration_selection",
    "router_config",
    "data_usage",
    "training_runtime",
    "random_seeds",
}
_STAGED_CLASSIFIER_RUN_EVIDENCE_FIELDS = _CLASSIFIER_RUN_EVIDENCE_FIELDS | {
    "authoritative_run_fingerprint",
    "classifier_training_seeds",
    "robustness_across_training_seeds_not_evaluated",
    "classifier_training_promoted",
    "classifier_training_metrics",
}
_PROTECTED_ROOTS = tuple(
    PROJECT_ROOT / name for name in ("src", "scripts", "environment", "tests", "docs", ".git")
)
_PROTECTED_FILES = tuple(
    PROJECT_ROOT / name
    for name in ("AGENTS.md", "PLAN.md", "README.md", "pyproject.toml", ".gitignore")
)
_PROTECTED_GENERATED_ROOTS = (
    PROJECT_ROOT / "outputs" / "datasets",
    PROJECT_ROOT / "outputs" / "models",
    *(
        PROJECT_ROOT / "outputs" / "diagnostics" / name
        for name in ("m0", "m1", "m2", "m3a", "m3b", "m4", "m42", "m43")
    ),
)
_FORBIDDEN_INPUT_IDENTITIES = (
    "m3b_test",
    "fresh_seed",
    "m42_final_v0",
    "m5a_language_final_v0",
    "m5a_control_final_v0",
    "smolvla",
)


class LanguageRouterEvaluationCommandError(RuntimeError):
    """Raised when target language evaluation cannot preserve its evidence contract."""


class GeneratorFactory(Protocol):
    def __call__(
        self,
        config: StructuredLLMRouterConfig,
        device: str,
        local_files_only: bool,
    ) -> LocalTextGenerator: ...


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--fixture", action="store_true")
    modes.add_argument("--target-development", action="store_true")
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--classifier-artifact", type=Path, required=True)
    parser.add_argument("--llm-model-id", required=True)
    parser.add_argument("--llm-model-revision", required=True)
    parser.add_argument("--llm-tokenizer-revision", required=True)
    parser.add_argument("--llm-license", required=True)
    parser.add_argument(
        "--llm-license-reviewed",
        action="store_true",
        help="Confirm that the pinned official model card and declared license were reviewed.",
    )
    parser.add_argument(
        "--llm-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--llm-quantization", choices=("none",), default="none")
    parser.add_argument("--llm-maximum-new-tokens", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--repeat-count", type=int, default=DEFAULT_REPEAT_COUNT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _resolved_unlinked(path: Path, *, label: str) -> Path:
    lexical = _lexical_absolute(path)
    for component in (lexical, *lexical.parents):
        if component.is_symlink() or component.is_junction():
            raise LanguageRouterEvaluationCommandError(
                f"{label} traverses a symlink or junction: {component}"
            )
    return lexical.resolve(strict=False)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _overlaps(left: Path, right: Path) -> bool:
    return _is_within(left, right) or _is_within(right, left)


def _reject_forbidden_input_identity(path: Path, *, label: str) -> None:
    parts = tuple(part.lower() for part in path.parts)
    if any(fragment in part for part in parts for fragment in _FORBIDDEN_INPUT_IDENTITIES):
        raise LanguageRouterEvaluationCommandError(
            f"{label} names a forbidden test, historical-fresh, final, or SmolVLA input"
        )


def _safe_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    corpus = _resolved_unlinked(args.corpus_root, label="language corpus input")
    classifier = _resolved_unlinked(args.classifier_artifact, label="classifier artifact input")
    output = _resolved_unlinked(args.output_root, label="language-router evidence output")
    report = _resolved_unlinked(args.report, label="language-router command report")
    for label, path in (("corpus", corpus), ("classifier", classifier)):
        _reject_forbidden_input_identity(path, label=label)
        if not path.is_dir() or path.is_symlink() or path.is_junction():
            raise LanguageRouterEvaluationCommandError(f"{label} input must be one real directory")
    if _overlaps(corpus, classifier):
        raise LanguageRouterEvaluationCommandError(
            "corpus and classifier immutable inputs cannot overlap"
        )
    protected = tuple(
        _resolved_unlinked(path, label="protected repository content")
        for path in (*_PROTECTED_ROOTS, *_PROTECTED_FILES, *_PROTECTED_GENERATED_ROOTS)
    )
    if any(_overlaps(output, path) for path in (*protected, corpus, classifier)):
        raise LanguageRouterEvaluationCommandError(
            "evaluation output cannot overlap source, Git, historical artifacts, corpus, or "
            "classifier artifacts"
        )
    if any(_overlaps(report, path) for path in (*protected, corpus, classifier, output)):
        raise LanguageRouterEvaluationCommandError(
            "command report cannot overlap source, Git, immutable inputs, or evaluation output"
        )
    if output.exists() and (not output.is_dir() or output.is_symlink() or output.is_junction()):
        raise LanguageRouterEvaluationCommandError(
            "evaluation output root must be a real directory"
        )
    if report.exists() and (not report.is_file() or report.is_symlink() or report.is_junction()):
        raise LanguageRouterEvaluationCommandError("command report must be a real file")
    return corpus, classifier, output, report


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink() or path.is_junction():
        raise LanguageRouterEvaluationCommandError(f"{label} must be one real JSON file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LanguageRouterEvaluationCommandError(f"could not read {label}: {error}") from error
    if not isinstance(value, dict):
        raise LanguageRouterEvaluationCommandError(f"{label} must contain one JSON object")
    return cast(dict[str, object], value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _artifact_records(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for candidate in root.rglob("*"):
        if candidate.is_symlink() or candidate.is_junction():
            raise LanguageRouterEvaluationCommandError(
                f"immutable artifact contains a linked entry: {candidate}"
            )
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise LanguageRouterEvaluationCommandError(
                f"immutable artifact contains a special entry: {candidate}"
            )
        relative = PurePosixPath(candidate.relative_to(root)).as_posix()
        if relative in {"artifact_manifest.json", "complete.json"}:
            continue
        records.append(
            {
                "path": relative,
                "size_bytes": candidate.stat().st_size,
                "sha256": _sha256_file(candidate),
            }
        )
    return sorted(records, key=lambda value: cast(str, value["path"]))


def _validate_corpus_archive(root: Path, corpus: GeneratedLanguageCorpus) -> str:
    manifest = _read_json_object(root / "artifact_manifest.json", label="corpus manifest")
    complete = _read_json_object(root / "complete.json", label="corpus completion")
    records = _artifact_records(root)
    expected_archive_fingerprint = f"sha256:{sha256_hex({'corpus_fingerprint': corpus.manifest.corpus_fingerprint, 'artifacts': records})}"
    if set(manifest) != {
        "schema_version",
        "corpus_fingerprint",
        "archive_fingerprint",
        "artifacts",
    }:
        raise LanguageRouterEvaluationCommandError("corpus artifact manifest fields differ")
    if manifest != {
        "schema_version": CORPUS_ARCHIVE_SCHEMA,
        "corpus_fingerprint": corpus.manifest.corpus_fingerprint,
        "archive_fingerprint": expected_archive_fingerprint,
        "artifacts": records,
    }:
        raise LanguageRouterEvaluationCommandError("corpus checksum manifest differs")
    if complete != {
        "schema_version": CORPUS_ARCHIVE_SCHEMA,
        "corpus_fingerprint": corpus.manifest.corpus_fingerprint,
        "archive_fingerprint": expected_archive_fingerprint,
        "passed": True,
    }:
        raise LanguageRouterEvaluationCommandError("corpus completion marker differs")
    if _read_json_object(root / "corpus_manifest.json", label="corpus identity") != (
        corpus.manifest.to_dict()
    ):
        raise LanguageRouterEvaluationCommandError("corpus identity differs from locked generator")
    if (root / "final.jsonl").exists():
        raise LanguageRouterEvaluationCommandError(
            "target-development corpus archive must not materialize final command text"
        )
    for split in (LanguageSplit.TRAIN, LanguageSplit.VALIDATION, LanguageSplit.DEVELOPMENT):
        path = root / f"{split.value}.jsonl"
        if not path.is_file() or path.is_symlink() or path.is_junction():
            raise LanguageRouterEvaluationCommandError(
                f"corpus is missing safe {split.value}.jsonl"
            )
        try:
            observed = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise LanguageRouterEvaluationCommandError(
                f"could not parse corpus {split.value} examples: {error}"
            ) from error
        expected = [example.to_dict() for example in corpus.examples_for_split(split)]
        if observed != expected:
            raise LanguageRouterEvaluationCommandError(
                f"corpus {split.value} examples differ from the locked generator"
            )
    language_development, language_final = build_language_schedule_locks(corpus)
    schedule_locks = _read_json_object(
        root / "language_schedule_locks.json", label="language schedule locks"
    )
    if schedule_locks != {
        "schema_version": CORPUS_ARCHIVE_SCHEMA,
        "development": language_development.to_dict(),
        "final": language_final.to_dict(),
    }:
        raise LanguageRouterEvaluationCommandError("language schedule locks differ")
    return expected_archive_fingerprint


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise LanguageRouterEvaluationCommandError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise LanguageRouterEvaluationCommandError(f"{label} keys must be strings")
    return cast(Mapping[str, object], value)


def _require_full_revision(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LanguageRouterEvaluationCommandError(
            f"{label} must be an exact lowercase 40-character revision"
        )
    return value


def _require_integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LanguageRouterEvaluationCommandError(
            f"{label} must be an integer greater than or equal to {minimum}"
        )
    return value


def _require_finite_number(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise LanguageRouterEvaluationCommandError(f"{label} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise LanguageRouterEvaluationCommandError(f"{label} must be finite")
    if minimum is not None and resolved < minimum:
        raise LanguageRouterEvaluationCommandError(f"{label} lies below its lower bound")
    if maximum is not None and resolved > maximum:
        raise LanguageRouterEvaluationCommandError(f"{label} lies above its upper bound")
    return resolved


def _split_fingerprints(corpus: GeneratedLanguageCorpus) -> dict[str, str]:
    return {
        split.value: corpus.manifest.split_manifests[split].content_fingerprint
        for split in LanguageSplit
    }


def _validate_classifier_evidence(
    evidence: Mapping[str, object],
    *,
    corpus: GeneratedLanguageCorpus,
    classifier_manifest: Mapping[str, object],
    target_development: bool,
) -> tuple[TemperatureCalibrationV0, FactorizedTextRouterConfig]:
    expected_fields = (
        _STAGED_CLASSIFIER_RUN_EVIDENCE_FIELDS
        if target_development
        else _CLASSIFIER_RUN_EVIDENCE_FIELDS
    )
    if set(evidence) != expected_fields:
        raise LanguageRouterEvaluationCommandError(
            "classifier run_evidence fields differ from the exact M5A schema"
        )
    if evidence.get("schema_version") != CLASSIFIER_RUN_EVIDENCE_SCHEMA:
        raise LanguageRouterEvaluationCommandError("classifier run_evidence schema differs")
    if target_development and evidence.get("mode") != "target_training_complete":
        raise LanguageRouterEvaluationCommandError(
            "target evaluation requires the promoted staged classifier artifact"
        )
    if target_development:
        authoritative = evidence.get("authoritative_run_fingerprint")
        if (
            not isinstance(authoritative, str)
            or not authoritative.startswith("sha256:")
            or len(authoritative) != 71
            or evidence.get("classifier_training_seeds") != 1
            or evidence.get("robustness_across_training_seeds_not_evaluated") is not True
            or evidence.get("classifier_training_promoted") is not True
        ):
            raise LanguageRouterEvaluationCommandError(
                "classifier evidence does not bind the single authoritative seed-0 run"
            )
        training_metrics = _require_mapping(
            evidence.get("classifier_training_metrics"),
            label="classifier training-promotion metrics",
        )
        required_training_gate = {
            "full_task_spec_accuracy": (">=", 0.95),
            "object_accuracy": (">=", 0.97),
            "bin_accuracy": (">=", 0.97),
            "false_route_rate": ("<=", 0.03),
            "ambiguous_rejection_recall": (">=", 0.90),
            "unsupported_rejection_recall": (">=", 0.95),
            "malformed_rejection_recall": (">=", 0.95),
            "schema_valid_rate": ("==", 1.0),
        }
        for key, (operator, bound) in required_training_gate.items():
            value = _require_finite_number(
                training_metrics.get(key), label=f"classifier training metric {key}"
            )
            valid = (
                value >= bound
                if operator == ">="
                else value <= bound
                if operator == "<="
                else value == bound
            )
            if not valid:
                raise LanguageRouterEvaluationCommandError(
                    "classifier artifact did not pass its validation promotion gate"
                )
    if evidence.get("corpus_manifest") != corpus.manifest.to_dict():
        raise LanguageRouterEvaluationCommandError("classifier corpus identity differs")
    if evidence.get("split_fingerprints") != _split_fingerprints(corpus):
        raise LanguageRouterEvaluationCommandError("classifier split fingerprints differ")
    if evidence.get("model_id") != classifier_manifest.get("model_id"):
        raise LanguageRouterEvaluationCommandError("classifier model ID differs")
    for key in ("model_revision", "tokenizer_revision"):
        if evidence.get(key) != classifier_manifest.get(key):
            raise LanguageRouterEvaluationCommandError(f"classifier {key} differs")
        if target_development:
            _require_full_revision(evidence.get(key), label=f"classifier {key}")
    git_state = _require_mapping(evidence.get("git_state"), label="classifier Git state")
    if evidence.get("git_commit") != git_state.get("commit"):
        raise LanguageRouterEvaluationCommandError("classifier Git identities differ")
    if target_development and (
        git_state.get("dirty") is not False or git_state.get("baseline_tracked") is not True
    ):
        raise LanguageRouterEvaluationCommandError(
            "target classifier artifact requires a clean tracked producer"
        )
    classifier_dependencies = _require_mapping(
        evidence.get("dependencies"), label="classifier dependencies"
    )
    if target_development and (
        classifier_dependencies.get("transformers") != "5.4.0"
        or classifier_dependencies.get("tokenizers") != "0.22.2"
    ):
        raise LanguageRouterEvaluationCommandError(
            "classifier artifact was not produced with the locked text dependencies"
        )
    random_seeds = _require_mapping(evidence.get("random_seeds"), label="classifier random seeds")
    if set(random_seeds) != {"torch", "cuda", "data_order"} or any(
        _require_integer(random_seeds.get(name), label=f"classifier {name} seed") != 0
        for name in ("torch", "cuda", "data_order")
    ):
        raise LanguageRouterEvaluationCommandError(
            "classifier random seeds differ from the predeclared seed-0 run"
        )
    training_runtime = _require_mapping(
        evidence.get("training_runtime"), label="classifier training runtime"
    )
    if target_development and (
        training_runtime.get("device") != "cuda"
        or training_runtime.get("dtype") != "float32"
        or training_runtime.get("cuda_available") is not True
    ):
        raise LanguageRouterEvaluationCommandError(
            "target classifier training runtime differs from the declared CUDA float32 run"
        )
    data_usage = _require_mapping(evidence.get("data_usage"), label="classifier data usage")
    required_data_usage = {
        "gradient_split": "train",
        "checkpoint_selection_split": "validation",
        "temperature_calibration_split": "validation",
        "threshold_selection_split": "validation",
        "development_examples_materialized": False,
        "development_used_for_training_or_selection": False,
        "final_examples_materialized": False,
        "final_used_for_training_or_selection": False,
    }
    if any(data_usage.get(key) != expected for key, expected in required_data_usage.items()):
        raise LanguageRouterEvaluationCommandError(
            "classifier training/selection data usage violates the train-validation boundary"
        )
    observed_train_count = _require_integer(
        data_usage.get("train_examples"), label="classifier train example count", minimum=1
    )
    observed_validation_count = _require_integer(
        data_usage.get("validation_examples"),
        label="classifier validation example count",
        minimum=1,
    )
    if target_development and (
        observed_train_count != corpus.manifest.split_manifests[LanguageSplit.TRAIN].example_count
        or observed_validation_count
        != corpus.manifest.split_manifests[LanguageSplit.VALIDATION].example_count
    ):
        raise LanguageRouterEvaluationCommandError(
            "classifier train/validation example counts differ from the locked corpus"
        )
    calibration_config = _require_mapping(
        evidence.get("calibration_config"), label="classifier calibration config"
    )
    if (
        calibration_config.get("evidence_split") != "validation"
        or calibration_config.get("maximum_false_route_rate") != 0.03
    ):
        raise LanguageRouterEvaluationCommandError(
            "classifier calibration objective differs from the validation-only lock"
        )
    training_result = _require_mapping(
        evidence.get("training_result"), label="classifier training result"
    )
    selected_checkpoint_id = training_result.get("selected_checkpoint_id")
    selected_step = training_result.get("selected_step")
    if not isinstance(selected_checkpoint_id, str) or not selected_checkpoint_id:
        raise LanguageRouterEvaluationCommandError("classifier selected checkpoint is malformed")
    _require_integer(selected_step, label="classifier selected step", minimum=1)
    checkpoint_validations = training_result.get("checkpoint_validations")
    if not isinstance(checkpoint_validations, list) or not checkpoint_validations:
        raise LanguageRouterEvaluationCommandError(
            "classifier checkpoint validations must be a non-empty JSON array"
        )
    selected_matches = 0
    for index, raw_validation in enumerate(checkpoint_validations):
        validation = _require_mapping(
            raw_validation, label=f"classifier checkpoint validation {index}"
        )
        if validation.get("evidence_split") != "validation":
            raise LanguageRouterEvaluationCommandError(
                "classifier checkpoint selection contains non-validation evidence"
            )
        if validation.get("validation_examples") != observed_validation_count:
            raise LanguageRouterEvaluationCommandError(
                "classifier checkpoint validation denominator differs from the locked split"
            )
        if (
            validation.get("checkpoint_id") == selected_checkpoint_id
            and validation.get("step") == selected_step
        ):
            selected_matches += 1
    if selected_matches != 1:
        raise LanguageRouterEvaluationCommandError(
            "classifier selected checkpoint is not uniquely present in validation evidence"
        )
    calibration_selection = _require_mapping(
        evidence.get("calibration_selection"), label="classifier calibration selection"
    )
    if set(calibration_selection) != {"temperature", "threshold"}:
        raise LanguageRouterEvaluationCommandError("calibration selection fields differ")
    temperature_payload = _require_mapping(
        calibration_selection.get("temperature"), label="temperature calibration"
    )
    threshold_payload = _require_mapping(
        calibration_selection.get("threshold"), label="routing threshold"
    )
    if set(temperature_payload) != {
        "temperature",
        "validation_examples",
        "nll_before",
        "nll_after",
        "bounded_iterations",
    }:
        raise LanguageRouterEvaluationCommandError("temperature calibration fields differ")
    if set(threshold_payload) != {
        "threshold",
        "validation_routeable_examples",
        "validation_rejected_examples",
        "valid_full_task_accuracy",
        "false_route_rate",
        "rejection_recall",
    }:
        raise LanguageRouterEvaluationCommandError("routing threshold fields differ")
    try:
        calibration = TemperatureCalibrationV0(
            temperature=_require_finite_number(
                temperature_payload["temperature"],
                label="classifier temperature",
                minimum=sys.float_info.min,
            ),
            validation_examples=_require_integer(
                temperature_payload["validation_examples"],
                label="classifier calibration examples",
                minimum=1,
            ),
            nll_before=_require_finite_number(
                temperature_payload["nll_before"],
                label="classifier pre-calibration NLL",
                minimum=0.0,
            ),
            nll_after=_require_finite_number(
                temperature_payload["nll_after"],
                label="classifier post-calibration NLL",
                minimum=0.0,
            ),
            bounded_iterations=_require_integer(
                temperature_payload["bounded_iterations"],
                label="classifier temperature iterations",
                minimum=1,
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise LanguageRouterEvaluationCommandError(
            f"classifier temperature calibration is malformed: {error}"
        ) from error
    if calibration.validation_examples != observed_validation_count:
        raise LanguageRouterEvaluationCommandError(
            "classifier calibration denominator differs from language validation"
        )
    if calibration.nll_after > calibration.nll_before + 1e-12:
        raise LanguageRouterEvaluationCommandError(
            "classifier validation temperature increased the locked calibration objective"
        )
    threshold_routeable_count = _require_integer(
        threshold_payload.get("validation_routeable_examples"),
        label="classifier threshold routeable denominator",
    )
    threshold_rejected_count = _require_integer(
        threshold_payload.get("validation_rejected_examples"),
        label="classifier threshold rejected denominator",
    )
    if threshold_routeable_count + threshold_rejected_count != observed_validation_count:
        raise LanguageRouterEvaluationCommandError(
            "classifier threshold-selection denominators differ from validation"
        )
    if target_development and (
        threshold_routeable_count
        != sum(
            corpus.manifest.split_manifests[
                LanguageSplit.VALIDATION
            ].routeable_count_by_task_id.values()
        )
        or threshold_rejected_count
        != sum(
            corpus.manifest.split_manifests[
                LanguageSplit.VALIDATION
            ].rejection_count_by_reason.values()
        )
    ):
        raise LanguageRouterEvaluationCommandError(
            "target classifier threshold denominators differ from locked validation"
        )
    router_payload = _require_mapping(
        evidence.get("router_config"), label="classifier router config"
    )
    if set(router_payload) != {
        "maximum_sequence_length",
        "routing_threshold",
        "device_independent",
    }:
        raise LanguageRouterEvaluationCommandError("classifier router config fields differ")
    if router_payload.get("device_independent") is not True:
        raise LanguageRouterEvaluationCommandError("classifier router config is not portable")
    threshold = _require_finite_number(
        threshold_payload.get("threshold"),
        label="classifier routing threshold",
        minimum=0.0,
        maximum=1.0,
    )
    false_route_rate = _require_finite_number(
        threshold_payload.get("false_route_rate"),
        label="classifier validation false-route rate",
        minimum=0.0,
        maximum=1.0,
    )
    _require_finite_number(
        threshold_payload.get("valid_full_task_accuracy"),
        label="classifier validation full-task accuracy",
        minimum=0.0,
        maximum=1.0,
    )
    _require_finite_number(
        threshold_payload.get("rejection_recall"),
        label="classifier validation rejection recall",
        minimum=0.0,
        maximum=1.0,
    )
    if false_route_rate > 0.03:
        raise LanguageRouterEvaluationCommandError(
            "classifier selected threshold violates the predeclared false-route constraint"
        )
    if router_payload.get("routing_threshold") != threshold:
        raise LanguageRouterEvaluationCommandError(
            "classifier router threshold differs from validation selection"
        )
    try:
        config = FactorizedTextRouterConfig(
            maximum_sequence_length=_require_integer(
                router_payload["maximum_sequence_length"],
                label="classifier maximum sequence length",
                minimum=1,
            ),
            routing_threshold=threshold,
            device="cpu",
        )
    except (KeyError, TypeError, ValueError) as error:
        raise LanguageRouterEvaluationCommandError(
            f"classifier router config is malformed: {error}"
        ) from error
    return calibration, config


def _load_classifier_router(
    artifact_root: Path,
    *,
    corpus: GeneratedLanguageCorpus,
    device: str,
    target_development: bool,
) -> tuple[FactorizedTextRouterV0, dict[str, object]]:
    evidence = read_text_classifier_run_evidence(artifact_root)
    model, tokenizer, manifest = load_factorized_text_classifier(artifact_root)
    calibration, portable_config = _validate_classifier_evidence(
        evidence,
        corpus=corpus,
        classifier_manifest=manifest,
        target_development=target_development,
    )
    config = FactorizedTextRouterConfig(
        maximum_sequence_length=portable_config.maximum_sequence_length,
        routing_threshold=portable_config.routing_threshold,
        device=device,
    )
    owner = _read_json_object(artifact_root / "owner.json", label="classifier owner")
    completion = _read_json_object(artifact_root / "complete.json", label="classifier completion")
    if completion.get("passed") is not True:
        raise LanguageRouterEvaluationCommandError("classifier artifact is incomplete")
    identity = {
        "run_fingerprint": owner.get("run_fingerprint"),
        "artifact_fingerprint": completion.get("artifact_fingerprint"),
        "classifier_manifest": manifest,
        "run_evidence_fingerprint": owner.get("run_evidence_fingerprint"),
        "model_id": evidence["model_id"],
        "model_revision": evidence["model_revision"],
        "tokenizer_revision": evidence["tokenizer_revision"],
        "calibration_selection": evidence["calibration_selection"],
        "router_config": config.to_dict(),
        "training_git_commit": evidence["git_commit"],
    }
    identity["router_fingerprint"] = f"sha256:{sha256_hex(identity)}"
    return (
        FactorizedTextRouterV0(
            model=model,
            tokenizer=cast(object, tokenizer),
            calibration=calibration,
            config=config,
        ),
        identity,
    )


def _select_prompt_examples(corpus: GeneratedLanguageCorpus) -> tuple[LanguageExample, ...]:
    return select_structured_routing_prompt_examples(corpus.examples_for_split(LanguageSplit.TRAIN))


def _default_generator_factory(
    config: StructuredLLMRouterConfig,
    device: str,
    local_files_only: bool,
) -> LocalTextGenerator:
    return TransformersLocalTextGenerator(
        config=config,
        device=device,
        local_files_only=local_files_only,
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _supplemental_metrics(
    records: Sequence[RouterEvaluationRecord],
    *,
    structured_llm: bool,
) -> dict[str, object]:
    routeable = [record for record in records if record.expected_status is RouterStatus.ROUTE]
    rejected = [record for record in records if record.expected_status is not RouterStatus.ROUTE]
    exhausted = 0
    repaired = 0
    initial_valid = 0
    for record in records:
        if structured_llm:
            errors = record.decision.evidence.get("parse_errors")
            error_count = len(errors) if isinstance(errors, tuple) else 0
            if record.decision.rejection_reason is RouterRejectionReason.FORMAT_REPAIR_EXHAUSTED:
                exhausted += 1
            elif error_count:
                repaired += 1
            else:
                initial_valid += 1
        else:
            initial_valid += 1
    return {
        "schema_valid_output_rate": _safe_ratio(len(records) - exhausted, len(records)),
        "initial_output_schema_valid_rate": _safe_ratio(initial_valid, len(records)),
        "format_repair_success_count": repaired,
        "format_repair_exhausted_count": exhausted,
        "structured_output_malformed_rate": _safe_ratio(exhausted, len(records)),
        "routeable_denominator": len(routeable),
        "rejected_denominator": len(rejected),
    }


def _complete_supplemental_metrics(
    records: Sequence[RouterEvaluationRecord],
    examples: Sequence[LanguageExample],
    *,
    structured_llm: bool,
) -> dict[str, object]:
    metrics = _supplemental_metrics(records, structured_llm=structured_llm)
    by_id = {example.example_id: example for example in examples}
    routeable = [record for record in records if record.expected_status is RouterStatus.ROUTE]
    object_confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    bin_confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    object_correct = 0
    bin_correct = 0
    for record in routeable:
        task_spec = by_id[record.example_id].expected_task_spec
        if task_spec is None:
            raise LanguageRouterEvaluationCommandError(
                "routeable record lost its expected TaskSpec"
            )
        predicted_object = record.decision.target_object_id or "none"
        predicted_bin = record.decision.target_bin_id or "none"
        object_confusion[task_spec.target_object_id][predicted_object] += 1
        bin_confusion[task_spec.target_bin_id][predicted_bin] += 1
        object_correct += predicted_object == task_spec.target_object_id
        bin_correct += predicted_bin == task_spec.target_bin_id
    metrics.update(
        {
            "object_accuracy": _safe_ratio(object_correct, len(routeable)),
            "bin_accuracy": _safe_ratio(bin_correct, len(routeable)),
            "route_recall": _safe_ratio(
                sum(record.decision.status is RouterStatus.ROUTE for record in routeable),
                len(routeable),
            ),
            "false_rejection_rate": _safe_ratio(
                sum(record.decision.status is not RouterStatus.ROUTE for record in routeable),
                len(routeable),
            ),
            "false_route_rate": _safe_ratio(
                sum(
                    record.decision.status is RouterStatus.ROUTE
                    for record in records
                    if record.expected_status is not RouterStatus.ROUTE
                ),
                sum(record.expected_status is not RouterStatus.ROUTE for record in records),
            ),
            "ambiguous_rejection_recall": _status_recall(records, RouterStatus.REJECT_AMBIGUOUS),
            "unsupported_rejection_recall": _status_recall(
                records, RouterStatus.REJECT_UNSUPPORTED
            ),
            "malformed_rejection_recall": _status_recall(records, RouterStatus.REJECT_MALFORMED),
            "object_confusion": {
                key: dict(sorted(value.items())) for key, value in sorted(object_confusion.items())
            },
            "bin_confusion": {
                key: dict(sorted(value.items())) for key, value in sorted(bin_confusion.items())
            },
        }
    )
    return metrics


def _status_recall(
    records: Sequence[RouterEvaluationRecord], expected_status: RouterStatus
) -> float:
    selected = [record for record in records if record.expected_status is expected_status]
    return _safe_ratio(
        sum(record.decision.status is expected_status for record in selected),
        len(selected),
    )


def _evaluation_payload(
    *,
    router: object,
    examples: Sequence[LanguageExample],
    split: LanguageSplit,
    repeat_count: int,
    structured_llm: bool,
) -> tuple[tuple[RouterEvaluationRecord, ...], dict[str, object]]:
    records, summary = evaluate_language_router(
        router=cast(object, router),
        examples=examples,
        split=split,
        repeat_count=repeat_count,
    )
    return records, {
        "summary": summary.to_dict(),
        "supplemental_metrics": _complete_supplemental_metrics(
            records,
            examples,
            structured_llm=structured_llm,
        ),
    }


def _language_candidate_promotions(
    comparison: Mapping[str, object],
) -> tuple[dict[str, dict[str, object]], str | None]:
    """Apply the fixed language gate and rank only learned routers that passed it."""

    development = comparison.get(LanguageSplit.DEVELOPMENT.value)
    if not isinstance(development, Mapping):
        raise LanguageRouterEvaluationCommandError("language development comparison is missing")
    promotions: dict[str, dict[str, object]] = {}
    ranking: list[tuple[tuple[float, float, float, float, float], str]] = []
    for candidate in ("classifier", "llm"):
        raw = development.get(candidate)
        if not isinstance(raw, Mapping):
            raise LanguageRouterEvaluationCommandError(
                f"language development lacks {candidate} evidence"
            )
        summary_raw = raw.get("summary")
        supplemental_raw = raw.get("supplemental_metrics")
        if not isinstance(summary_raw, Mapping) or not isinstance(supplemental_raw, Mapping):
            raise LanguageRouterEvaluationCommandError(
                f"language development {candidate} metrics are malformed"
            )
        metrics = summary_raw
        rejection_macro = (
            sum(
                float(metrics[key])
                for key in (
                    "ambiguous_rejection_recall",
                    "unsupported_rejection_recall",
                    "malformed_rejection_recall",
                )
            )
            / 3.0
        )
        malformed_rate = (
            float(supplemental_raw.get("structured_output_malformed_rate", 0.0))
            if candidate == "llm"
            else 0.0
        )
        gate_checks = {
            "full_task_spec_accuracy": float(metrics["valid_full_task_accuracy"]) >= 0.95,
            "object_accuracy": float(metrics["object_accuracy"]) >= 0.97,
            "bin_accuracy": float(metrics["bin_accuracy"]) >= 0.97,
            "false_route_rate": float(metrics["false_route_rate"]) <= 0.03,
            "ambiguous_rejection_recall": (float(metrics["ambiguous_rejection_recall"]) >= 0.90),
            "unsupported_rejection_recall": (
                float(metrics["unsupported_rejection_recall"]) >= 0.95
            ),
            "malformed_rejection_recall": (float(metrics["malformed_rejection_recall"]) >= 0.95),
            "schema_valid_rate": float(metrics["schema_valid_output_rate"]) >= 0.99,
            "deterministic_repeatability": (float(metrics["deterministic_repeatability"]) == 1.0),
            "llm_malformed_output_rate": candidate != "llm" or malformed_rate <= 0.01,
        }
        promoted = all(gate_checks.values())
        p95 = float(metrics["latency_p95_ms"])
        promotions[candidate] = {
            "promoted": promoted,
            "gate_checks": gate_checks,
            "rejection_macro_recall": rejection_macro,
            "malformed_output_rate": malformed_rate,
            "latency_p95_ms": p95,
        }
        if promoted:
            ranking.append(
                (
                    (
                        -float(metrics["valid_full_task_accuracy"]),
                        float(metrics["false_route_rate"]),
                        -rejection_macro,
                        malformed_rate,
                        p95,
                    ),
                    candidate,
                )
            )
    ranking.sort()
    return promotions, None if not ranking else ranking[0][1]


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")
        )
        + "\n"
    ).encode("utf-8")


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise LanguageRouterEvaluationCommandError(
            f"refusing to overwrite immutable router evidence: {path}"
        ) from error


def _records_jsonl(records: Sequence[RouterEvaluationRecord]) -> bytes:
    return b"".join(_json_bytes(record.to_dict()) for record in records)


def _validate_completed_evidence(root: Path, *, owner: Mapping[str, object]) -> dict[str, object]:
    if not root.is_dir() or root.is_symlink() or root.is_junction():
        raise LanguageRouterEvaluationCommandError(
            "completed evaluation must be one real directory"
        )
    if _read_json_object(root / "owner.json", label="evaluation owner") != dict(owner):
        raise LanguageRouterEvaluationCommandError("completed evaluation owner differs")
    manifest = _read_json_object(root / "artifact_manifest.json", label="evaluation manifest")
    complete = _read_json_object(root / "complete.json", label="evaluation completion")
    records = _artifact_records(root)
    expected = f"sha256:{sha256_hex({'owner': dict(owner), 'artifacts': records})}"
    if manifest != {
        "schema_version": EVIDENCE_SCHEMA,
        "artifact_fingerprint": expected,
        "artifacts": records,
    }:
        raise LanguageRouterEvaluationCommandError("evaluation checksum manifest differs")
    if complete != {
        "schema_version": EVIDENCE_SCHEMA,
        "evaluation_fingerprint": owner["evaluation_fingerprint"],
        "artifact_fingerprint": expected,
        "passed": True,
    }:
        raise LanguageRouterEvaluationCommandError("evaluation completion marker differs")
    result = _read_json_object(root / "result.json", label="evaluation result")
    if result.get("passed") is not True or result.get("evaluation_fingerprint") != owner.get(
        "evaluation_fingerprint"
    ):
        raise LanguageRouterEvaluationCommandError("evaluation result identity differs")
    return {**result, "artifact_fingerprint": expected, "evidence_root": str(root)}


def _promote_evidence(
    *,
    output_root: Path,
    owner: Mapping[str, object],
    results: Mapping[
        LanguageSplit, Mapping[str, tuple[tuple[RouterEvaluationRecord, ...], dict[str, object]]]
    ],
    result_payload: Mapping[str, object],
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    evaluation_fingerprint = cast(str, owner["evaluation_fingerprint"])
    token = evaluation_fingerprint.removeprefix("sha256:")
    destination = (output_root / token).resolve(strict=False)
    if destination.parent != output_root.resolve(strict=False):
        raise LanguageRouterEvaluationCommandError("evaluation run escaped its owned output root")
    if destination.exists():
        return {
            **_validate_completed_evidence(destination, owner=owner),
            "evidence_reused": True,
        }
    staging = output_root / f".{token}.{uuid.uuid4().hex}.staging"
    staging.mkdir(exist_ok=False)
    try:
        _write_new(staging / "owner.json", _json_bytes(dict(owner)))
        for split, by_router in results.items():
            for router_name, (records, summary) in by_router.items():
                root = staging / split.value / router_name
                _write_new(root / "records.jsonl", _records_jsonl(records))
                _write_new(root / "summary.json", _json_bytes(summary))
        _write_new(staging / "result.json", _json_bytes(dict(result_payload)))
        artifact_records = _artifact_records(staging)
        artifact_fingerprint = (
            f"sha256:{sha256_hex({'owner': dict(owner), 'artifacts': artifact_records})}"
        )
        _write_new(
            staging / "artifact_manifest.json",
            _json_bytes(
                {
                    "schema_version": EVIDENCE_SCHEMA,
                    "artifact_fingerprint": artifact_fingerprint,
                    "artifacts": artifact_records,
                }
            ),
        )
        _write_new(
            staging / "complete.json",
            _json_bytes(
                {
                    "schema_version": EVIDENCE_SCHEMA,
                    "evaluation_fingerprint": evaluation_fingerprint,
                    "artifact_fingerprint": artifact_fingerprint,
                    "passed": True,
                }
            ),
        )
        _validate_completed_evidence(staging, owner=owner)
        if staging.stat().st_dev != output_root.stat().st_dev:
            raise LanguageRouterEvaluationCommandError(
                "evaluation staging and destination are on different filesystems"
            )
        os.replace(staging, destination)
        return {
            **_validate_completed_evidence(destination, owner=owner),
            "evidence_reused": False,
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError as error:
        raise LanguageRouterEvaluationCommandError(
            f"required distribution is missing: {distribution}"
        ) from error


def _dependency_versions() -> dict[str, object]:
    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers": _package_version("transformers"),
        "tokenizers": _package_version("tokenizers"),
        "huggingface_hub": _package_version("huggingface-hub"),
        "safetensors": _package_version("safetensors"),
    }


def execute(
    args: argparse.Namespace,
    *,
    generator_factory: GeneratorFactory | None = None,
) -> dict[str, object]:
    corpus_root, classifier_root, output_root, _ = _safe_paths(args)
    target_development = bool(args.target_development)
    if generator_factory is not None and target_development:
        raise LanguageRouterEvaluationCommandError(
            "injected generators are fixture-only and cannot claim target development"
        )
    if args.repeat_count < 2:
        raise LanguageRouterEvaluationCommandError(
            "router evaluation requires at least two decisions for repeatability"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise LanguageRouterEvaluationCommandError("CUDA evaluation was requested but unavailable")
    dependencies = _dependency_versions()
    if target_development:
        if dependencies["transformers"] != "5.4.0" or dependencies["tokenizers"] != "0.22.2":
            raise LanguageRouterEvaluationCommandError(
                "target router evaluation requires the locked Transformers/tokenizers versions"
            )
        _require_full_revision(args.llm_model_revision, label="LLM model revision")
        _require_full_revision(args.llm_tokenizer_revision, label="LLM tokenizer revision")
        if (
            not isinstance(args.llm_model_id, str)
            or args.llm_model_id.count("/") != 1
            or "://" in args.llm_model_id
            or Path(args.llm_model_id).exists()
        ):
            raise LanguageRouterEvaluationCommandError(
                "target LLM model ID must be one explicit non-local repository ID"
            )
        if not isinstance(args.llm_license, str) or not args.llm_license.strip():
            raise LanguageRouterEvaluationCommandError("target LLM license must be explicit")
        if args.llm_license_reviewed is not True:
            raise LanguageRouterEvaluationCommandError(
                "target LLM requires an explicit official-model-card license review"
            )
    git_state = inspect_git_state(PROJECT_ROOT)
    if target_development and (git_state.dirty or not git_state.baseline_tracked):
        raise LanguageRouterEvaluationCommandError(
            "target-development router evaluation requires one clean tracked Git commit"
        )
    corpus = build_language_corpus()
    corpus_archive_fingerprint = _validate_corpus_archive(corpus_root, corpus)
    language_development, language_final = build_language_schedule_locks(corpus)
    classifier_router, classifier_identity = _load_classifier_router(
        classifier_root,
        corpus=corpus,
        device=args.device,
        target_development=target_development,
    )
    rule_config = RuleRouterConfig()
    rule_router = RuleRouterV0(rule_config)
    rule_identity = {
        "router_name": "RuleRouterV0",
        "config": rule_config.to_dict(),
        "router_fingerprint": rule_config.fingerprint,
    }
    llm_config = StructuredLLMRouterConfig(
        model_id=args.llm_model_id,
        model_revision=args.llm_model_revision,
        tokenizer_revision=args.llm_tokenizer_revision,
        dtype=args.llm_dtype,
        quantization=args.llm_quantization,
        maximum_new_tokens=args.llm_maximum_new_tokens,
        maximum_format_repair_attempts=1,
    )
    prompt_examples = _select_prompt_examples(corpus)
    prompt_template, prompt_fingerprint, prompt_example_ids = build_structured_routing_prompt(
        examples=prompt_examples,
        prompt_version=llm_config.prompt_version,
    )
    factory = _default_generator_factory if generator_factory is None else generator_factory
    llm_generator = factory(llm_config, args.device, bool(args.local_files_only))
    llm_parameter_count: int | None = None
    if target_development:
        model = getattr(llm_generator, "model", None)
        parameters = getattr(model, "parameters", None)
        if not callable(parameters):
            raise LanguageRouterEvaluationCommandError(
                "target local LLM does not expose parameters for the 0.5B-3B gate"
            )
        llm_parameter_count = sum(parameter.numel() for parameter in parameters())
        if not 500_000_000 <= llm_parameter_count <= 3_000_000_000:
            raise LanguageRouterEvaluationCommandError(
                "target local LLM parameter count lies outside the locked 0.5B-3B range"
            )
    llm_router = StructuredLocalLLMRouterV0(
        config=llm_config,
        generator=llm_generator,
        prompt_examples=prompt_examples,
    )
    if llm_router.prompt_fingerprint != prompt_fingerprint:
        raise LanguageRouterEvaluationCommandError("LLM prompt changed during construction")
    llm_identity: dict[str, object] = {
        "router_name": "StructuredLocalLLMRouterV0",
        "config": llm_config.to_dict(),
        "declared_license": args.llm_license,
        "official_model_card_license_reviewed": bool(args.llm_license_reviewed),
        "parameter_count": llm_parameter_count,
        "prompt_template": prompt_template,
        "prompt_fingerprint": prompt_fingerprint,
        "prompt_example_ids": list(prompt_example_ids),
        "prompt_example_split": "train",
        "generator": (
            "TransformersLocalTextGenerator"
            if generator_factory is None
            else "injected_fixture_generator"
        ),
    }
    llm_identity["router_fingerprint"] = f"sha256:{sha256_hex(llm_identity)}"
    identities = {
        "rule": rule_identity,
        "classifier": classifier_identity,
        "llm": llm_identity,
    }
    owner_payload = {
        "schema_version": EVIDENCE_SCHEMA,
        "mode": "target_development" if target_development else "fixture",
        "corpus_fingerprint": corpus.manifest.corpus_fingerprint,
        "corpus_archive_fingerprint": corpus_archive_fingerprint,
        "language_validation_fingerprint": corpus.manifest.split_manifests[
            LanguageSplit.VALIDATION
        ].content_fingerprint,
        "language_development_schedule_fingerprint": language_development.schedule_fingerprint,
        "language_final_lock_fingerprint": language_final.schedule_fingerprint,
        "classifier_artifact_fingerprint": classifier_identity["artifact_fingerprint"],
        "router_identities": identities,
        "repeat_count": args.repeat_count,
        "local_files_only": bool(args.local_files_only),
        "evaluation_order": ["validation", "development"],
        "device": args.device,
        "git_commit": git_state.commit,
        "dependencies": dependencies,
    }
    evaluation_fingerprint = f"sha256:{sha256_hex(owner_payload)}"
    owner = {**owner_payload, "evaluation_fingerprint": evaluation_fingerprint}
    destination = output_root / evaluation_fingerprint.removeprefix("sha256:")
    if destination.exists():
        return {
            **_validate_completed_evidence(destination, owner=owner),
            "evidence_reused": True,
        }

    routers = {
        "rule": (rule_router, False),
        "classifier": (classifier_router, False),
        "llm": (llm_router, True),
    }
    results: dict[
        LanguageSplit,
        dict[str, tuple[tuple[RouterEvaluationRecord, ...], dict[str, object]]],
    ] = {}
    # All validation runs finish before any development command is evaluated.
    for split in (LanguageSplit.VALIDATION, LanguageSplit.DEVELOPMENT):
        examples = corpus.examples_for_split(split)
        split_results: dict[str, tuple[tuple[RouterEvaluationRecord, ...], dict[str, object]]] = {}
        for router_name, (router, structured_llm) in routers.items():
            split_results[router_name] = _evaluation_payload(
                router=router,
                examples=examples,
                split=split,
                repeat_count=args.repeat_count,
                structured_llm=structured_llm,
            )
        results[split] = split_results

    comparison = {
        split.value: {router_name: payload[1] for router_name, payload in results[split].items()}
        for split in (LanguageSplit.VALIDATION, LanguageSplit.DEVELOPMENT)
    }
    candidate_promotions, primary_learned_router = _language_candidate_promotions(comparison)
    result_payload = {
        "schema_version": EVIDENCE_SCHEMA,
        "passed": True,
        "mode": owner["mode"],
        "evaluation_fingerprint": evaluation_fingerprint,
        "corpus_fingerprint": corpus.manifest.corpus_fingerprint,
        "corpus_archive_fingerprint": corpus_archive_fingerprint,
        "router_identities": identities,
        "repeat_count": args.repeat_count,
        "local_files_only": bool(args.local_files_only),
        "language_validation_schedule": {
            "split": "validation",
            "content_fingerprint": corpus.manifest.split_manifests[
                LanguageSplit.VALIDATION
            ].content_fingerprint,
            "example_count": len(corpus.examples_for_split(LanguageSplit.VALIDATION)),
        },
        "language_development_schedule": language_development.to_dict(),
        "language_final_lock": {
            "schedule_id": language_final.schedule_id,
            "schedule_fingerprint": language_final.schedule_fingerprint,
            "sealed": True,
            "texts_materialized": False,
        },
        "comparison": comparison,
        "candidate_promotions": candidate_promotions,
        "promoted_learned_routers": [
            candidate
            for candidate in ("classifier", "llm")
            if candidate_promotions[candidate]["promoted"] is True
        ],
        "primary_learned_router": primary_learned_router,
        "rule_router_offline_baseline_only": True,
        "evaluation_order": ["validation", "development"],
        "validation_completed_before_development": True,
        "classifier_checkpoint_selected": True,
        "classifier_calibration_validated": True,
        "llm_router_loaded": target_development,
        "llm_router_fixture_loaded": not target_development,
        "llm_prompt_locked": target_development,
        "language_validation_completed": target_development,
        "language_development_completed": target_development,
        "language_fixture_completed": not target_development,
        "target_development_executed": target_development,
        "physical_target_validated": False,
        "final_benchmark_authorized": False,
        "language_final_accessed": False,
        "control_final_accessed": False,
        "m42_final_accessed": False,
        "test_split_accessed": False,
        "historical_fresh_accessed": False,
        "controller_dispatched": False,
        "environment_reset_called": False,
        "environment_step_count": 0,
        "m2_expert_call_count": 0,
        "smolvla_go": False,
        "git_state": git_state.to_dict(),
        "dependencies": dependencies,
    }
    return _promote_evidence(
        output_root=output_root,
        owner=owner,
        results=results,
        result_payload=result_payload,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mode = "target_development" if args.target_development else "fixture"
    payload: dict[str, object] = {
        "schema_version": COMMAND_SCHEMA,
        "passed": False,
        "mode": mode,
        "target_development_executed": False,
        "physical_target_validated": False,
        "language_final_accessed": False,
        "control_final_accessed": False,
        "m42_final_accessed": False,
        "test_split_accessed": False,
        "historical_fresh_accessed": False,
        "controller_dispatched": False,
        "environment_reset_called": False,
        "environment_step_count": 0,
        "m2_expert_call_count": 0,
        "smolvla_go": False,
    }
    report_path: Path | None = None
    try:
        *_, report_path = _safe_paths(args)
        payload.update(execute(args))
        payload["schema_version"] = COMMAND_SCHEMA
    except Exception as error:  # noqa: BLE001 - outer command preserves exact diagnostics
        traceback.print_exc()
        payload.update(
            {
                "error_type": type(error).__name__,
                "error_message": str(error) or repr(error),
                "traceback": traceback.format_exc(),
            }
        )
    if report_path is not None:
        try:
            atomic_write_json(report_path, payload)
        except Exception:
            traceback.print_exc()
            return 1
    console = {
        "schema_version": COMMAND_SCHEMA,
        "passed": payload.get("passed"),
        "mode": mode,
        "evaluation_fingerprint": payload.get("evaluation_fingerprint"),
        "artifact_fingerprint": payload.get("artifact_fingerprint"),
        "evidence_root": payload.get("evidence_root"),
        "language_development_completed": payload.get("language_development_completed", False),
        "language_final_accessed": payload.get("language_final_accessed", False),
        "physical_target_validated": payload.get("physical_target_validated", False),
        "report": None if report_path is None else str(report_path),
    }
    print(json.dumps(console, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if payload.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
