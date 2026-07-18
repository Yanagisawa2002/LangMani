"""Run M5A.2 train-smoke, validation, and conditionally language development.

The rejected classifier remains an offline negative baseline.  This command
loads no ACT checkpoint, constructs no robot environment, and never opens the
language-final or control schedules.
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
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import stable_task_id
from langmani.language.corpus import GeneratedLanguageCorpus, build_language_corpus
from langmani.language.language_development_evidence import (
    write_language_development_evidence,
)
from langmani.language.llm_router import (
    QWEN3_1_7B_LICENSE,
    QWEN3_1_7B_MODEL_ID,
    QWEN3_1_7B_REVISION,
    LocalTextGenerator,
    StructuredLLMRouterConfig,
    StructuredLocalLLMRouterV0,
    TransformersLocalTextGenerator,
    build_structured_routing_prompt,
    select_structured_routing_prompt_examples,
)
from langmani.language.offline_language_development import (
    CLASSIFIER_NEGATIVE_ELIGIBILITY,
    LOCAL_LLM_ELIGIBILITY,
    RULE_ROUTER_ELIGIBILITY,
    FrozenClassifierNegativeBaselineV0,
    LanguageRouterSelectionV0,
    PromptLockV0,
    QwenModelIdentityV0,
    conservative_decoder_from_mapping,
    evaluate_llm_development_gate,
    evaluate_llm_validation_gate,
)
from langmani.language.offline_router_metrics import recompute_router_metrics
from langmani.language.rejection_diagnostics import load_selected_checkpoint_read_only
from langmani.language.rejection_report import validate_rejection_analysis_artifact
from langmani.language.router_evaluation import (
    RouterEvaluationRecord,
    evaluate_language_router,
)
from langmani.language.router_types import (
    LanguageExample,
    LanguageSplit,
    RouterDecision,
    RouterRejectionReason,
    RouterStatus,
)
from langmani.language.rule_router import RuleRouterV0
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
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "diagnostics" / "m5a" / "language-development"
DEFAULT_REPORT = (
    PROJECT_ROOT / "outputs" / "diagnostics" / "m5a" / "stages" / "language-development.json"
)
DEFAULT_CLASSIFIER_CHECKPOINT = PROJECT_ROOT / (
    "outputs/models/text-router/authoritative-runs/"
    "9e3ac659fa2b695c843650df35e3779741d94b3dd70b2aec52a429bc4b2edf49/"
    "checkpoints/validation_best.pt"
)
DEFAULT_CLASSIFIER_REJECTION_EVIDENCE = (
    PROJECT_ROOT / "outputs" / "diagnostics" / "m5a" / "rejection-analysis"
)

COMMAND_SCHEMA = "langmani-m5a2-language-router-evaluation-command-v0"
EVIDENCE_SCHEMA = "langmani-m5a2-language-router-evaluation-evidence-v0"
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


class _StructuralFixtureGenerator:
    """Schema/evidence fixture; it is never reported as local-model inference."""

    def __init__(self) -> None:
        self.router = RuleRouterV0()
        self.file_identities: dict[str, object] = {}
        self.generation_metadata: list[dict[str, object]] = []
        self.peak_gpu_memory_bytes = 0

    def generate(self, prompt: str, *, max_new_tokens: int) -> str:
        if max_new_tokens <= 0:
            raise LanguageRouterEvaluationCommandError("fixture output-token bound is invalid")
        try:
            command_line = prompt.rsplit("Command: ", maxsplit=1)[1].splitlines()[0]
            command = json.loads(command_line)
        except (IndexError, json.JSONDecodeError) as error:
            raise LanguageRouterEvaluationCommandError(
                "fixture prompt omitted one JSON command"
            ) from error
        if not isinstance(command, str):
            raise LanguageRouterEvaluationCommandError("fixture command must be text")
        decision = self.router.route(command)
        if decision.status is RouterStatus.ROUTE:
            payload: dict[str, object] = {
                "status": "route",
                "target_object_id": decision.target_object_id,
                "target_bin_id": decision.target_bin_id,
                "reason": "explicit_object_and_destination",
            }
        else:
            assert decision.rejection_reason is not None
            reason = {
                RouterRejectionReason.CONFLICTING_BINS: "conflicting_destinations",
                RouterRejectionReason.MULTIPLE_TASKS: "multiple_sequential_tasks",
                RouterRejectionReason.MEANINGLESS_TEXT: "meaningless_or_noise",
                RouterRejectionReason.EMPTY_TEXT: "malformed_input",
                RouterRejectionReason.MALFORMED_CONTROL_CHARACTERS: "malformed_input",
            }.get(decision.rejection_reason, decision.rejection_reason.value)
            payload = {
                "status": decision.status.value,
                "target_object_id": None,
                "target_bin_id": None,
                "reason": reason,
            }
        self.generation_metadata.append(
            {
                "generated_token_count": 0,
                "elapsed_seconds": 0.0,
                "peak_gpu_memory_bytes": 0,
                "do_sample": False,
                "enable_thinking": False,
                "fixture_only": True,
            }
        )
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


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
    parser.add_argument("--llm-model-id", default=QWEN3_1_7B_MODEL_ID)
    parser.add_argument("--llm-model-revision", default=QWEN3_1_7B_REVISION)
    parser.add_argument("--llm-tokenizer-revision", default=QWEN3_1_7B_REVISION)
    parser.add_argument("--llm-license", default=QWEN3_1_7B_LICENSE)
    parser.add_argument(
        "--llm-license-reviewed",
        action="store_true",
        help="Confirm that the pinned official model card and declared license were reviewed.",
    )
    parser.add_argument(
        "--llm-model-card-reviewed",
        action="store_true",
        help="Confirm review of the pinned official Qwen model card and chat template.",
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


def _safe_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    corpus = _resolved_unlinked(args.corpus_root, label="language corpus input")
    classifier = _resolved_unlinked(args.classifier_checkpoint, label="classifier checkpoint input")
    rejection = _resolved_unlinked(
        args.classifier_rejection_evidence,
        label="classifier rejection-evidence input",
    )
    output = _resolved_unlinked(args.output_root, label="language-router evidence output")
    report = _resolved_unlinked(args.report, label="language-router command report")
    for label, path in (("corpus", corpus), ("classifier rejection", rejection)):
        _reject_forbidden_input_identity(path, label=label)
        if not path.is_dir() or path.is_symlink() or path.is_junction():
            raise LanguageRouterEvaluationCommandError(f"{label} input must be one real directory")
    if not classifier.is_file() or classifier.is_symlink():
        raise LanguageRouterEvaluationCommandError(
            "classifier checkpoint input must be one real file"
        )
    if _overlaps(corpus, rejection) or _overlaps(corpus, classifier):
        raise LanguageRouterEvaluationCommandError(
            "corpus and classifier immutable inputs cannot overlap"
        )
    protected = tuple(
        _resolved_unlinked(path, label="protected repository content")
        for path in (*_PROTECTED_ROOTS, *_PROTECTED_FILES, *_PROTECTED_GENERATED_ROOTS)
    )
    if any(_overlaps(output, path) for path in (*protected, corpus, classifier, rejection)):
        raise LanguageRouterEvaluationCommandError(
            "evaluation output cannot overlap source, Git, historical artifacts, corpus, or "
            "classifier artifacts"
        )
    if any(_overlaps(report, path) for path in (*protected, corpus, classifier, rejection, output)):
        raise LanguageRouterEvaluationCommandError(
            "command report cannot overlap source, Git, immutable inputs, or evaluation output"
        )
    if output.exists() and (not output.is_dir() or output.is_symlink() or output.is_junction()):
        raise LanguageRouterEvaluationCommandError(
            "evaluation output root must be a real directory"
        )
    if report.exists() and (not report.is_file() or report.is_symlink() or report.is_junction()):
        raise LanguageRouterEvaluationCommandError("command report must be a real file")
    return corpus, classifier, rejection, output, report


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


def _selected_train_smoke_examples(
    corpus: GeneratedLanguageCorpus,
) -> tuple[LanguageExample, ...]:
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
    return tuple(selected)


def _semantic_decision_fingerprint(decision: RouterDecision) -> str:
    """Fingerprint executable/rejection semantics without variable runtime telemetry."""

    return f"sha256:{sha256_hex({'status': decision.status.value, 'task_id': decision.task_id, 'rejection_reason': None if decision.rejection_reason is None else decision.rejection_reason.value})}"


def _run_train_smoke(
    router: StructuredLocalLLMRouterV0,
    examples: Sequence[LanguageExample],
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
    routed_task_ids = {
        cast(Mapping[str, object], value["decision"]).get("task_id")
        for value in records
        if value["expected_status"] == RouterStatus.ROUTE.value
    }
    produced_statuses = {
        cast(Mapping[str, object], value["decision"])["status"] for value in records
    }
    rejected_safe = all(
        cast(Mapping[str, object], value["decision"])["task_spec"] is None
        for value in records
        if cast(Mapping[str, object], value["decision"])["status"] != RouterStatus.ROUTE.value
    )
    checks = {
        "final_schema_valid_rate_at_least_0_99": final_valid / len(records) >= 0.99,
        "all_six_task_specs_produced": len(routed_task_ids - {None}) == 6,
        "all_router_statuses_produced": produced_statuses
        == {value.value for value in RouterStatus},
        "rejected_decisions_non_executable": rejected_safe,
        "deterministic_final_decisions": all(bool(value["deterministic"]) for value in records),
    }
    return {
        "schema_version": "langmani-m5a2-llm-train-smoke-v0",
        "split": "train",
        "example_count": len(records),
        "example_ids": [value.example_id for value in examples],
        "smoke_fingerprint": f"sha256:{sha256_hex([value.example_id for value in examples])}",
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
        "schema_version": "langmani-m5a2-offline-router-result-v0",
        "split": split.value,
        "eligibility": dict(eligibility),
        "metrics": metrics,
        "records": [record.to_dict() for record in records],
    }


def _load_negative_baseline(
    *,
    checkpoint: Path,
    rejection_root: Path,
    local_files_only: bool,
) -> tuple[FrozenClassifierNegativeBaselineV0, dict[str, object]]:
    artifact = validate_rejection_analysis_artifact(rejection_root)
    root = Path(cast(str, artifact["root"]))
    rejected = _read_json_object(root / "rejected_candidate.json", label="rejected classifier")
    selection = _read_json_object(root / "candidate_selection.json", label="decoder selection")
    source = _read_json_object(root / "source_recovery_identity.json", label="classifier identity")
    selected = selection.get("selected_configuration")
    if (
        artifact.get("classifier_full_quality_gate_passed") is not False
        or artifact.get("conclusion") != "classifier_rejected_after_posthoc_calibration"
        or rejected.get("classifier_candidate_frozen") is not True
        or rejected.get("additional_training_authorized") is not False
        or rejected.get("additional_seed_authorized") is not False
        or not isinstance(selected, Mapping)
    ):
        raise LanguageRouterEvaluationCommandError(
            "classifier rejection evidence does not preserve the M5A.1 decision"
        )
    expected_sha = cast(str, rejected["selected_classifier_checkpoint_fingerprint"])
    if _sha256_file(checkpoint) != expected_sha:
        raise LanguageRouterEvaluationCommandError("classifier checkpoint SHA differs")
    run_fingerprint = source.get("run_fingerprint")
    selected_step = source.get("selected_step")
    if (
        not isinstance(run_fingerprint, str)
        or not run_fingerprint.startswith("sha256:")
        or isinstance(selected_step, bool)
        or not isinstance(selected_step, int)
        or selected_step <= 0
    ):
        raise LanguageRouterEvaluationCommandError("classifier source identity is malformed")
    bundle = load_selected_checkpoint_read_only(
        checkpoint,
        expected_checkpoint_fingerprint=expected_sha,
        expected_run_fingerprint=f"sha256:{run_fingerprint.removeprefix('sha256:')}",
        expected_step=selected_step,
        local_files_only=local_files_only,
    )
    tokenizer_contract = bundle.processor_state.get("tokenizer")
    if not isinstance(tokenizer_contract, Mapping) or not isinstance(
        tokenizer_contract.get("maximum_sequence_length"), int
    ):
        raise LanguageRouterEvaluationCommandError("classifier tokenizer contract is malformed")
    configuration = conservative_decoder_from_mapping(cast(Mapping[str, object], selected))
    router = FrozenClassifierNegativeBaselineV0(
        bundle=bundle,
        configuration=configuration,
        maximum_sequence_length=cast(int, tokenizer_contract["maximum_sequence_length"]),
        device="cpu",
    )
    identity = {
        "analysis_fingerprint": artifact["analysis_fingerprint"],
        "checkpoint_fingerprint": expected_sha,
        "selection_fingerprint": selection["selection_fingerprint"],
        "decoder": configuration.to_dict(),
        "classifier_candidate_frozen": True,
        "offline_negative_baseline": True,
        "eligible_for_promotion": False,
        "eligible_for_dispatch": False,
        "classifier_runtime_selected": False,
        "classifier_full_quality_gate_passed": False,
        "additional_training_authorized": False,
        "additional_seed_authorized": False,
        "optimizer_constructed": False,
        "model_state_fingerprint_before": router.model_state_fingerprint_before,
    }
    return router, identity


def _development_safety_checks(
    *,
    router: StructuredLocalLLMRouterV0,
    records: Sequence[RouterEvaluationRecord],
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
    return checks, {
        "empty_input_decision": empty_first.to_dict(),
        "empty_repeat_decision_fingerprint": empty_second.decision_fingerprint,
        "checks": checks,
    }


def execute(
    args: argparse.Namespace,
    *,
    generator_factory: GeneratorFactory | None = None,
) -> dict[str, object]:
    corpus_root, checkpoint, rejection_root, output_root, _ = _safe_paths(args)
    target = bool(args.target_development)
    if generator_factory is not None and target:
        raise LanguageRouterEvaluationCommandError(
            "injected generators are fixture-only and cannot claim real GPU inference"
        )
    if args.repeat_count < 2:
        raise LanguageRouterEvaluationCommandError("repeat_count must be at least two")
    dependencies = _dependency_versions()
    git_state = inspect_git_state(PROJECT_ROOT)
    if target:
        if args.device != "cuda" or not torch.cuda.is_available():
            raise LanguageRouterEvaluationCommandError("M5A.2 target inference requires CUDA")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            raise LanguageRouterEvaluationCommandError("M5A.2 requires CUDA_VISIBLE_DEVICES=0")
        if dependencies["transformers"] != "5.4.0" or dependencies["tokenizers"] != "0.22.2":
            raise LanguageRouterEvaluationCommandError("locked text dependency versions differ")
        if git_state.dirty or not git_state.baseline_tracked:
            raise LanguageRouterEvaluationCommandError("target evidence requires clean tracked Git")
        if (
            args.llm_model_id != QWEN3_1_7B_MODEL_ID
            or args.llm_model_revision != QWEN3_1_7B_REVISION
            or args.llm_tokenizer_revision != QWEN3_1_7B_REVISION
            or args.llm_license != QWEN3_1_7B_LICENSE
            or args.llm_license_reviewed is not True
            or args.llm_model_card_reviewed is not True
            or args.llm_dtype != "bfloat16"
            or args.llm_quantization != "none"
        ):
            raise LanguageRouterEvaluationCommandError("pinned Qwen M5A.2 identity differs")
    corpus = build_language_corpus()
    archive_fingerprint = _validate_corpus_archive(corpus_root, corpus)
    language_development, language_final = build_language_schedule_locks(corpus)
    model_identity = QwenModelIdentityV0()
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
    prompt_text, prompt_fingerprint, prompt_example_ids = build_structured_routing_prompt(
        examples=prompt_examples,
        prompt_version=llm_config.prompt_version,
    )
    if generator_factory is not None:
        generator = generator_factory(llm_config, args.device, bool(args.local_files_only))
    elif target:
        generator = _default_generator_factory(llm_config, args.device, bool(args.local_files_only))
    else:
        generator = _StructuralFixtureGenerator()
    llm_router = StructuredLocalLLMRouterV0(
        config=llm_config, generator=generator, prompt_examples=prompt_examples
    )
    if llm_router.prompt_fingerprint != prompt_fingerprint:
        raise LanguageRouterEvaluationCommandError("prompt changed during router construction")
    parameter_count: int | None = None
    if target:
        model = getattr(generator, "model", None)
        if model is None:
            raise LanguageRouterEvaluationCommandError("target generator omitted its local model")
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if not 1_600_000_000 <= parameter_count <= 1_900_000_000:
            raise LanguageRouterEvaluationCommandError("Qwen parameter count differs")
    train_smoke_examples = _selected_train_smoke_examples(corpus)
    owner_payload = {
        "schema_version": EVIDENCE_SCHEMA,
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
        "language_development_schedule_fingerprint": language_development.schedule_fingerprint,
        "language_final_lock_fingerprint": language_final.schedule_fingerprint,
        "model_identity_fingerprint": model_identity.fingerprint,
        "prompt_fingerprint": prompt_fingerprint,
        "generation_config_fingerprint": llm_config.fingerprint,
        "classifier_rejection_analysis_fingerprint": validate_rejection_analysis_artifact(
            rejection_root
        )["analysis_fingerprint"],
        "repeat_count": args.repeat_count,
    }
    runtime_fingerprint = f"sha256:{sha256_hex(owner_payload)}"
    owner = {**owner_payload, "runtime_fingerprint": runtime_fingerprint}
    artifacts: dict[str, object] = {
        "model_identity.json": {
            **model_identity.to_dict(),
            "identity_fingerprint": model_identity.fingerprint,
            "actual_files": dict(getattr(generator, "file_identities", {})),
            "parameter_count": parameter_count,
            "model_loaded_once": target,
            "fixture_contract_only": not target,
            "license_reviewed": bool(args.llm_license_reviewed),
            "model_card_reviewed": bool(args.llm_model_card_reviewed),
        },
        "prompt.json": {
            "prompt_text": prompt_text,
            "prompt_fingerprint": prompt_fingerprint,
            "prompt_example_ids": list(prompt_example_ids),
            "prompt_example_split": "train",
        },
        "generation_config.json": llm_config.to_dict(),
        "controller_registry_contract.json": {
            "schema_version": "langmani-m5a2-controller-registry-metadata-audit-v0",
            "metadata_only": True,
            "canonical_task_ids": [stable_task_id(value) for value in CANONICAL_TASK_SPECS],
            "controller_registry_loaded": False,
            "controller_checkpoint_loaded": False,
            "controller_dispatched": False,
            "environment_created": False,
            "environment_step_count": 0,
        },
    }
    smoke = _run_train_smoke(llm_router, train_smoke_examples)
    artifacts["train_smoke.json"] = smoke
    flags: dict[str, bool] = {
        "classifier_candidate_frozen": True,
        "classifier_offline_baseline_validated": False,
        "classifier_dispatch_prohibited": True,
        "rule_router_validation_completed": False,
        "llm_model_identity_validated": target,
        "llm_prompt_locked": False,
        "llm_train_smoke_completed": target,
        "llm_validation_completed": False,
        "llm_validation_gate_passed": False,
        "language_development_completed": False,
        "llm_language_quality_gate_passed": False,
        "learned_router_selected": False,
        "one_scene_control_smoke_authorized": False,
        "one_scene_control_smoke_completed": False,
        "three_scene_control_screen_completed": False,
        "predicted_control_development_completed": False,
        "language_final_accessed": False,
        "control_final_accessed": False,
        "test_split_accessed": False,
        "m42_final_accessed": False,
        "smolvla_go": False,
        "real_gpu_inference_validated": target,
        "physical_target_validated": False,
        "passed": True,
    }
    stopped_after = "train_smoke" if target else "fixture"
    prompt_lock: PromptLockV0 | None = None
    validation_results: dict[str, object] = {}
    development_results: dict[str, object] = {}
    selection_payload: dict[str, object] = {
        "selected_learned_router": None,
        "one_scene_control_smoke_authorized": False,
        "classifier": CLASSIFIER_NEGATIVE_ELIGIBILITY.to_dict(),
        "rule_router": RULE_ROUTER_ELIGIBILITY.to_dict(),
        "local_llm": LOCAL_LLM_ELIGIBILITY.to_dict(),
    }
    classifier_identity: dict[str, object] | None = None
    classifier_router: FrozenClassifierNegativeBaselineV0 | None = None
    # Fixture generation exercises serialization and evidence only.  It must
    # never fabricate validation/development metrics or authorize a runtime.
    if smoke["gate_passed"] is True and target:
        prompt_lock = PromptLockV0(
            prompt_text=prompt_text,
            prompt_fingerprint=prompt_fingerprint,
            prompt_example_ids=prompt_example_ids,
            train_smoke_example_ids=tuple(value.example_id for value in train_smoke_examples),
            corpus_fingerprint=corpus.manifest.corpus_fingerprint,
            locked_before_validation=True,
        )
        artifacts["prompt_lock.json"] = prompt_lock.to_dict()
        flags["llm_prompt_locked"] = True
        classifier_router, classifier_identity = _load_negative_baseline(
            checkpoint=checkpoint,
            rejection_root=rejection_root,
            local_files_only=bool(args.local_files_only),
        )
        rule_router = RuleRouterV0()
        validation_examples = corpus.examples_for_split(LanguageSplit.VALIDATION)
        for name, router, eligibility in (
            ("rule_router", rule_router, RULE_ROUTER_ELIGIBILITY.to_dict()),
            (
                "classifier_negative_baseline",
                classifier_router,
                CLASSIFIER_NEGATIVE_ELIGIBILITY.to_dict(),
            ),
            ("local_llm", llm_router, LOCAL_LLM_ELIGIBILITY.to_dict()),
        ):
            _, payload = _router_payload(
                router=router,
                examples=validation_examples,
                split=LanguageSplit.VALIDATION,
                repeat_count=args.repeat_count,
                eligibility=eligibility,
            )
            validation_results[name] = payload
            artifacts[f"validation/{name}.json"] = payload
        flags["classifier_offline_baseline_validated"] = True
        flags["rule_router_validation_completed"] = True
        flags["llm_validation_completed"] = True
        llm_validation_metrics = cast(
            Mapping[str, object],
            cast(Mapping[str, object], validation_results["local_llm"])["metrics"],
        )
        validation_gate = evaluate_llm_validation_gate(
            metrics=llm_validation_metrics, prohibited_source_access=False
        )
        flags["llm_validation_gate_passed"] = validation_gate.passed
        artifacts["validation/gate.json"] = validation_gate.to_dict()
        stopped_after = "validation"
        if validation_gate.passed:
            development_examples = corpus.examples_for_split(LanguageSplit.DEVELOPMENT)
            llm_development_records: tuple[RouterEvaluationRecord, ...] | None = None
            for name, router, eligibility in (
                ("rule_router", rule_router, RULE_ROUTER_ELIGIBILITY.to_dict()),
                (
                    "classifier_negative_baseline",
                    classifier_router,
                    CLASSIFIER_NEGATIVE_ELIGIBILITY.to_dict(),
                ),
                ("local_llm", llm_router, LOCAL_LLM_ELIGIBILITY.to_dict()),
            ):
                records, payload = _router_payload(
                    router=router,
                    examples=development_examples,
                    split=LanguageSplit.DEVELOPMENT,
                    repeat_count=args.repeat_count,
                    eligibility=eligibility,
                )
                if name == "local_llm":
                    llm_development_records = records
                development_results[name] = payload
                artifacts[f"development/{name}.json"] = payload
            assert llm_development_records is not None
            safety_checks, safety_evidence = _development_safety_checks(
                router=llm_router, records=llm_development_records
            )
            artifacts["development/safety_probes.json"] = safety_evidence
            llm_development_metrics = cast(
                Mapping[str, object],
                cast(Mapping[str, object], development_results["local_llm"])["metrics"],
            )
            development_gate = evaluate_llm_development_gate(
                metrics=llm_development_metrics,
                per_task_accuracy=cast(
                    Mapping[str, object], llm_development_metrics["per_task_accuracy"]
                ),
                rejection_family_false_route_rates=cast(
                    Mapping[str, object],
                    llm_development_metrics["rejection_family_false_route_rates"],
                ),
                safety_checks=safety_checks,
            )
            artifacts["development/gate.json"] = development_gate.to_dict()
            flags["language_development_completed"] = True
            flags["llm_language_quality_gate_passed"] = development_gate.passed
            flags["one_scene_control_smoke_authorized"] = development_gate.passed
            stopped_after = "language_development"
            if development_gate.passed:
                parser_fingerprint = f"sha256:{sha256_hex('strict-router-json-reason-v1')}"
                validation_fingerprint = f"sha256:{sha256_hex(validation_results['local_llm'])}"
                development_fingerprint = f"sha256:{sha256_hex(development_results['local_llm'])}"
                selection_identity = {
                    "model": model_identity.fingerprint,
                    "prompt": prompt_fingerprint,
                    "parser": parser_fingerprint,
                    "generation": llm_config.fingerprint,
                    "validation": validation_fingerprint,
                    "development": development_fingerprint,
                    "git_commit": git_state.commit,
                    "corpus": corpus.manifest.corpus_fingerprint,
                }
                selection = LanguageRouterSelectionV0(
                    runtime_fingerprint=f"sha256:{sha256_hex(selection_identity)}",
                    model_identity_fingerprint=model_identity.fingerprint,
                    prompt_fingerprint=prompt_fingerprint,
                    parser_fingerprint=parser_fingerprint,
                    generation_config_fingerprint=llm_config.fingerprint,
                    validation_evidence_fingerprint=validation_fingerprint,
                    development_evidence_fingerprint=development_fingerprint,
                    git_commit=git_state.commit,
                    corpus_fingerprint=corpus.manifest.corpus_fingerprint,
                    validation_split_fingerprint=corpus.manifest.split_manifests[
                        LanguageSplit.VALIDATION
                    ].content_fingerprint,
                    development_split_fingerprint=corpus.manifest.split_manifests[
                        LanguageSplit.DEVELOPMENT
                    ].content_fingerprint,
                )
                selection_payload["selected_learned_router"] = selection.to_dict()
                selection_payload["one_scene_control_smoke_authorized"] = True
                flags["learned_router_selected"] = True
    if classifier_router is not None:
        unchanged = classifier_router.verify_unchanged()
        assert classifier_identity is not None
        classifier_identity["model_state_fingerprint_after"] = (
            classifier_router.model_state_fingerprint_before if unchanged else "changed"
        )
        classifier_identity["model_weights_unchanged"] = unchanged
        if not unchanged:
            raise LanguageRouterEvaluationCommandError(
                "classifier weights changed during inference"
            )
    artifacts["classifier_negative_baseline.json"] = {
        **CLASSIFIER_NEGATIVE_ELIGIBILITY.to_dict(),
        "classifier_candidate_frozen": True,
        "classifier_runtime_selected": False,
        "classifier_full_quality_gate_passed": False,
        "additional_training_authorized": False,
        "additional_seed_authorized": False,
        "identity": classifier_identity,
    }
    artifacts["candidate_selection.json"] = selection_payload
    artifacts["summary.md"] = (
        "# M5A.2 Offline Language Development\n\n"
        f"Stopped after: `{stopped_after}`.\n\n"
        f"LLM validation gate: `{flags['llm_validation_gate_passed']}`.\n\n"
        f"LLM development gate: `{flags['llm_language_quality_gate_passed']}`.\n\n"
        "No robot controller or environment was loaded.\n"
    )
    artifacts["result.json"] = {
        "schema_version": EVIDENCE_SCHEMA,
        "runtime_fingerprint": runtime_fingerprint,
        "stopped_after_stage": stopped_after,
        "flags": flags,
        "classifier_negative_baseline": classifier_identity,
        "generation_totals": {
            "generation_calls": len(getattr(generator, "generation_metadata", ())),
            "generated_tokens": sum(
                cast(int, value.get("generated_token_count", 0))
                for value in getattr(generator, "generation_metadata", ())
            ),
            "peak_gpu_memory_bytes": int(getattr(generator, "peak_gpu_memory_bytes", 0)),
        },
        "final_access_prohibitions": {
            "language_final_accessed": False,
            "control_final_accessed": False,
            "test_split_accessed": False,
            "historical_fresh_accessed": False,
            "m42_final_accessed": False,
            "smolvla_go": False,
            "controller_loaded": False,
            "environment_created": False,
            "environment_step_count": 0,
        },
    }
    evidence = write_language_development_evidence(
        output_root,
        owner=owner,
        artifacts=artifacts,
        stopped_after_stage=stopped_after,
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
        "stopped_after_stage": stopped_after,
        "evidence_reused": evidence["evidence_reused"],
        "physical_target_validated": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mode = "target_development" if args.target_development else "fixture"
    payload: dict[str, object] = {
        "schema_version": COMMAND_SCHEMA,
        "passed": False,
        "mode": mode,
        "classifier_candidate_frozen": True,
        "classifier_offline_baseline_validated": False,
        "classifier_dispatch_prohibited": True,
        "rule_router_validation_completed": False,
        "llm_model_identity_validated": False,
        "llm_prompt_locked": False,
        "llm_train_smoke_completed": False,
        "llm_validation_completed": False,
        "llm_validation_gate_passed": False,
        "language_development_completed": False,
        "llm_language_quality_gate_passed": False,
        "learned_router_selected": False,
        "one_scene_control_smoke_authorized": False,
        "one_scene_control_smoke_completed": False,
        "three_scene_control_screen_completed": False,
        "predicted_control_development_completed": False,
        "real_gpu_inference_validated": False,
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
        "stopped_after_stage": payload.get("stopped_after_stage"),
        "llm_validation_gate_passed": payload.get("llm_validation_gate_passed", False),
        "language_development_completed": payload.get("language_development_completed", False),
        "one_scene_control_smoke_authorized": payload.get(
            "one_scene_control_smoke_authorized", False
        ),
        "language_final_accessed": payload.get("language_final_accessed", False),
        "physical_target_validated": payload.get("physical_target_validated", False),
        "report": None if report_path is None else str(report_path),
    }
    print(json.dumps(console, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if payload.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
