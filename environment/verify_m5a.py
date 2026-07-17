"""Verify portable M5A contracts or orchestrate sealed target-development.

The default mode exercises deterministic CPU fixtures without claiming target
execution. ``--target-development`` first validates the completed M4.3b gate,
atomically locks all four M5A schedules, then resumably runs corpus creation,
one classifier training, three-router language evaluation, and oracle/predicted
control development. It independently aggregates correctness and quality while
leaving every final/test/fresh/SmolVLA source sealed and unexecuted.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from langmani.datasets.schedule import CANONICAL_TASK_SPECS  # noqa: E402
from langmani.environments.specs import TaskSpec  # noqa: E402
from langmani.language.artifact_validation import (  # noqa: E402
    DevelopmentControlInputsLike,
    ValidatedControlEvidence,
    ValidatedRouterEvaluationEvidence,
    validate_development_control_evidence,
    validate_language_corpus_archive,
    validate_router_evaluation_evidence,
)
from langmani.language.controller_registry import (  # noqa: E402
    ControllerRegistry,
    build_fixture_controller_registry,
    load_controller_registry_metadata,
)
from langmani.language.corpus import (  # noqa: E402
    FinalLanguageAccessError,
    GeneratedLanguageCorpus,
    build_language_corpus,
    language_corpus_counts,
)
from langmani.language.dispatcher import (  # noqa: E402
    ControllerDispatcher,
    FixturePerTaskControllerLoader,
)
from langmani.language.failure_attribution import (  # noqa: E402
    FailureAttribution,
    FailureAttributionEvidence,
    attribute_end_to_end,
)
from langmani.language.llm_router import (  # noqa: E402
    StructuredLLMRouterConfig,
    StructuredLocalLLMRouterV0,
)
from langmani.language.router_types import (  # noqa: E402
    LanguageSplit,
    M5ADevelopmentGate,
    M5AExperimentManifest,
    RouterConfidence,
    RouterDecision,
    RouterRuntimeIdentity,
    RouterStatus,
)
from langmani.language.rule_router import RuleRouterV0  # noqa: E402
from langmani.language.schedules import (  # noqa: E402
    M5A_CONTROL_DEV_SCHEDULE_ID,
    M5A_CONTROL_FINAL_SCHEDULE_ID,
    M5A_LANGUAGE_DEV_SCHEDULE_ID,
    M5A_LANGUAGE_FINAL_SCHEDULE_ID,
    M5A_TARGET_DEVELOPMENT_SCHEDULE_LOCK_SCHEMA,
    REQUIRED_EXCLUSION_SOURCE_IDS,
    ControlScheduleConfig,
    FinalControlScheduleAccessError,
    M5AScheduleBundle,
    SeedExclusionSource,
    build_authoritative_seed_exclusions,
    build_m5a_schedule_bundle,
    materialize_control_schedule,
)
from langmani.language.schema_validation import parse_strict_router_json  # noqa: E402
from langmani.language.text_calibration import (  # noqa: E402
    RoutingThresholdExample,
    fit_validation_temperature,
    select_validation_routing_threshold,
)
from langmani.language.text_classifier import (  # noqa: E402
    TEXT_CLASSIFIER_MODEL_ID,
    TEXT_CLASSIFIER_MODEL_REVISION,
    TEXT_CLASSIFIER_TOKENIZER_REVISION,
    FactorizedTextClassifierV0,
    compute_factorized_text_loss,
)
from langmani.language.text_training import (  # noqa: E402
    validate_completed_text_classifier_artifact,
)
from langmani.policies.act_runtime import atomic_write_json  # noqa: E402
from langmani.policies.m42_schedule import validate_locked_schedules  # noqa: E402

REPORT_SCHEMA_VERSION = "langmani-m5a-verification-v1"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "diagnostics" / "m5a"
DEFAULT_CORPUS_ROOT = PROJECT_ROOT / "outputs" / "datasets" / "m5a" / "langmani-language-corpus-v1"
DEFAULT_CLASSIFIER_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "models" / "text-router"
DEFAULT_LANGUAGE_EVIDENCE_ROOT = DEFAULT_OUTPUT_ROOT / "language-routing"
DEFAULT_CONTROL_EVIDENCE_ROOT = DEFAULT_OUTPUT_ROOT / "control"
DEFAULT_M3B_DATASET_ROOT = (
    PROJECT_ROOT / "outputs" / "datasets" / "m3b" / "langmani-pick-place-lerobot-v1"
)
DEFAULT_ACT_CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "models" / "act"
DEFAULT_RUNTIME_SELECTION = (
    PROJECT_ROOT / "outputs" / "diagnostics" / "m42" / "runtime_ablation" / "runtime_selection.json"
)
TARGET_SCHEDULE_LOCK_SCHEMA = M5A_TARGET_DEVELOPMENT_SCHEDULE_LOCK_SCHEMA
CORPUS_COMMAND_SCHEMA = "langmani-m5a-build-language-corpus-command-v1"
CLASSIFIER_COMMAND_SCHEMA = "langmani-m5a-train-text-router-command-v1"
CLASSIFIER_RUN_EVIDENCE_SCHEMA = "langmani-m5a-text-router-run-evidence-v1"
LANGUAGE_COMMAND_SCHEMA = "langmani-m5a-language-router-evaluation-command-v1"
CONTROL_COMPLETION_SCHEMA = "langmani-m5a-language-control-complete-v0"
M43B_INDEPENDENT_SCHEMA = "langmani-m43b-independent-target-development-verification-v0"
TARGET_STAGE_NAMES = ("corpus", "classifier", "language", "control")
PROTECTED_ROOTS = (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "environment",
    PROJECT_ROOT / "tests",
    PROJECT_ROOT / "docs",
    PROJECT_ROOT / ".git",
)
PROTECTED_GENERATED_INPUT_ROOTS = (
    PROJECT_ROOT / "outputs" / "datasets" / "m3a",
    PROJECT_ROOT / "outputs" / "datasets" / "m3b",
    PROJECT_ROOT / "outputs" / "models" / "act",
    PROJECT_ROOT / "outputs" / "models" / "act-task-token",
    PROJECT_ROOT / "outputs" / "models" / "act-factor-film",
    *(
        PROJECT_ROOT / "outputs" / "diagnostics" / name
        for name in ("m0", "m1", "m2", "m3a", "m3b", "m4", "m42", "m43")
    ),
)


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _resolved_unlinked(path: Path, *, label: str) -> Path:
    lexical = _lexical_absolute(path)
    for component in (lexical, *lexical.parents):
        if component.is_symlink() or component.is_junction():
            raise RuntimeError(f"{label} traverses a symlink or junction: {component}")
    return lexical.resolve(strict=False)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _overlaps(left: Path, right: Path) -> bool:
    return _is_within(left, right) or _is_within(right, left)


def _validate_output_root(path: Path) -> Path:
    output = _resolved_unlinked(path, label="M5A verifier output")
    protected = tuple(
        _resolved_unlinked(candidate, label="protected repository or historical artifact")
        for candidate in (*PROTECTED_ROOTS, *PROTECTED_GENERATED_INPUT_ROOTS)
    )
    if any(_overlaps(output, candidate) for candidate in protected):
        raise RuntimeError(
            "M5A verifier output must not overlap source or Git content, or historical artifacts"
        )
    if output.exists() and not output.is_dir():
        raise RuntimeError("M5A verifier output must be a real directory when present")
    report_path = _resolved_unlinked(output / "verification.json", label="M5A verifier report")
    if report_path.exists() and not report_path.is_file():
        raise RuntimeError("M5A verifier report must be a real file when present")
    return output


@dataclass(frozen=True, slots=True)
class _TargetPathLayout:
    verifier_output_root: Path
    corpus_root: Path
    classifier_output_root: Path
    language_evidence_root: Path
    control_evidence_root: Path
    m43_independent_verification: Path
    dataset_root: Path
    checkpoint_root: Path
    runtime_selection: Path
    m4_diagnostics_root: Path | None


def _validate_target_paths(args: argparse.Namespace) -> _TargetPathLayout:
    if args.m43_independent_verification is None:
        raise M5ATargetVerificationError(
            "--m43-independent-verification is required for --target-development"
        )
    layout = _TargetPathLayout(
        verifier_output_root=_validate_output_root(args.output_root),
        corpus_root=_resolved_unlinked(args.corpus_root, label="language corpus output"),
        classifier_output_root=_resolved_unlinked(
            args.classifier_output_root, label="classifier output"
        ),
        language_evidence_root=_resolved_unlinked(
            args.language_evidence_root, label="language evaluation output"
        ),
        control_evidence_root=_resolved_unlinked(
            args.control_evidence_root, label="language control output"
        ),
        m43_independent_verification=_resolved_unlinked(
            args.m43_independent_verification,
            label="M4.3b independent verification input",
        ),
        dataset_root=_resolved_unlinked(args.dataset_root, label="M3B dataset input"),
        checkpoint_root=_resolved_unlinked(args.checkpoint_root, label="ACT checkpoint input"),
        runtime_selection=_resolved_unlinked(
            args.runtime_selection, label="M4.2 runtime selection input"
        ),
        m4_diagnostics_root=(
            None
            if args.m4_diagnostics_root is None
            else _resolved_unlinked(
                args.m4_diagnostics_root,
                label="legacy M4 diagnostics input",
            )
        ),
    )
    immutable_inputs = {
        "M4.3b verification": layout.m43_independent_verification,
        "M3B dataset": layout.dataset_root,
        "ACT checkpoint": layout.checkpoint_root,
        "M4.2 runtime selection": layout.runtime_selection,
    }
    if layout.m4_diagnostics_root is not None:
        immutable_inputs["legacy M4 diagnostics"] = layout.m4_diagnostics_root
    input_items = tuple(immutable_inputs.items())
    for index, (left_label, left_path) in enumerate(input_items):
        for right_label, right_path in input_items[index + 1 :]:
            if _overlaps(left_path, right_path):
                raise M5ATargetVerificationError(
                    f"immutable {left_label} and {right_label} inputs cannot overlap"
                )

    managed_outputs = {
        "language corpus": layout.corpus_root,
        "classifier": layout.classifier_output_root,
        "language evaluation": layout.language_evidence_root,
        "language control": layout.control_evidence_root,
    }
    output_items = tuple(managed_outputs.items())
    for index, (left_label, left_path) in enumerate(output_items):
        for right_label, right_path in output_items[index + 1 :]:
            if _overlaps(left_path, right_path):
                raise M5ATargetVerificationError(
                    f"{left_label} and {right_label} output roots cannot overlap"
                )
    protected_outputs = tuple(
        _resolved_unlinked(path, label="protected repository or historical artifact")
        for path in (*PROTECTED_ROOTS, *PROTECTED_GENERATED_INPUT_ROOTS)
    )
    for output_label, output_path in managed_outputs.items():
        if any(_overlaps(output_path, path) for path in protected_outputs):
            raise M5ATargetVerificationError(
                f"{output_label} output cannot overlap repository or historical artifacts"
            )
        for input_label, input_path in input_items:
            if _overlaps(output_path, input_path):
                raise M5ATargetVerificationError(
                    f"{output_label} output cannot overlap immutable {input_label} input"
                )

    for output_label, output_path in (
        ("language corpus", layout.corpus_root),
        ("classifier", layout.classifier_output_root),
    ):
        if _overlaps(layout.verifier_output_root, output_path):
            raise M5ATargetVerificationError(
                f"verifier output cannot overlap {output_label} output"
            )
    for output_label, output_path in (
        ("language evaluation", layout.language_evidence_root),
        ("language control", layout.control_evidence_root),
    ):
        if _overlaps(layout.verifier_output_root, output_path) and not (
            output_path != layout.verifier_output_root
            and _is_within(output_path, layout.verifier_output_root)
        ):
            raise M5ATargetVerificationError(
                f"{output_label} may overlap verifier output only as a strict managed child"
            )
    for input_label, input_path in input_items:
        if _overlaps(layout.verifier_output_root, input_path):
            raise M5ATargetVerificationError(
                f"verifier output cannot overlap immutable {input_label} input"
            )
    return layout


@dataclass(slots=True)
class Report:
    """Truthful non-target flags plus small portable evidence."""

    checks: list[dict[str, object]] = field(default_factory=list)
    corpus_fingerprint: str | None = None
    split_isolation_fingerprint: str | None = None
    controller_registry_fingerprint: str | None = None
    schedule_fingerprints: dict[str, str] = field(default_factory=dict)
    implementation_validated: bool = False
    corpus_validated: bool = False
    split_isolation_validated: bool = False
    rule_router_validated: bool = False
    classifier_fixture_validated: bool = False
    llm_router_fixture_validated: bool = False
    controller_registry_validated: bool = False
    final_schedules_locked: bool = False
    classifier_training_completed: bool = False
    classifier_checkpoint_selected: bool = False
    classifier_calibration_validated: bool = False
    llm_router_loaded: bool = False
    llm_prompt_locked: bool = False
    language_development_completed: bool = False
    oracle_control_development_completed: bool = False
    predicted_control_development_completed: bool = False
    control_development_completed: bool = False
    rejection_noop_probe_validated: bool = False
    failure_attribution_validated: bool = False
    development_quality_gate_passed: bool = False
    final_benchmark_authorized: bool = False
    language_final_accessed: bool = False
    control_final_accessed: bool = False
    m42_final_accessed: bool = False
    test_split_accessed: bool = False
    historical_fresh_accessed: bool = False
    smolvla_go: bool = False
    physical_target_validated: bool = False

    def check(self, name: str, condition: bool, detail: str) -> None:
        status = "pass" if condition else "fail"
        print(f"[{status.upper()}] {name}: {detail}")
        self.checks.append({"name": name, "status": status, "detail": detail})

    @property
    def failed(self) -> bool:
        return any(check["status"] == "fail" for check in self.checks)

    @property
    def passed(self) -> bool:
        portable_claims = (
            self.implementation_validated,
            self.corpus_validated,
            self.split_isolation_validated,
            self.rule_router_validated,
            self.classifier_fixture_validated,
            self.llm_router_fixture_validated,
            self.controller_registry_validated,
            self.final_schedules_locked,
        )
        target_or_forbidden_claims = (
            self.classifier_training_completed,
            self.classifier_checkpoint_selected,
            self.classifier_calibration_validated,
            self.llm_router_loaded,
            self.llm_prompt_locked,
            self.language_development_completed,
            self.oracle_control_development_completed,
            self.predicted_control_development_completed,
            self.control_development_completed,
            self.rejection_noop_probe_validated,
            self.failure_attribution_validated,
            self.development_quality_gate_passed,
            self.final_benchmark_authorized,
            self.language_final_accessed,
            self.control_final_accessed,
            self.m42_final_accessed,
            self.test_split_accessed,
            self.historical_fresh_accessed,
            self.smolvla_go,
            self.physical_target_validated,
        )
        return not self.failed and all(portable_claims) and not any(target_or_forbidden_claims)

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "verification_mode": "non_target_structural_fixture",
            **{key: value for key, value in asdict(self).items() if key != "checks"},
            "checks": self.checks,
            "passed": self.passed,
        }


class M5ATargetVerificationError(RuntimeError):
    """Raised when target-development evidence violates an immutable contract."""


CommandRunner = Callable[[Sequence[str]], int]


@dataclass(slots=True)
class TargetDevelopmentReport:
    """Independent target-development aggregation, separate from model quality."""

    checks: list[dict[str, object]] = field(default_factory=list)
    m43b_evaluation_evidence_fingerprint: str | None = None
    corpus_fingerprint: str | None = None
    split_isolation_fingerprint: str | None = None
    classifier_run_fingerprint: str | None = None
    classifier_artifact_fingerprint: str | None = None
    language_evaluation_fingerprint: str | None = None
    language_artifact_fingerprint: str | None = None
    controller_registry_fingerprint: str | None = None
    controller_checkpoint_fingerprints: list[str] = field(default_factory=list)
    experiment_manifest_path: str | None = None
    experiment_fingerprint: str | None = None
    router_runtime_fingerprints: dict[str, str] = field(default_factory=dict)
    llm_model_id: str | None = None
    llm_model_revision: str | None = None
    llm_tokenizer_revision: str | None = None
    llm_license: str | None = None
    llm_dtype: str | None = None
    llm_maximum_new_tokens: int | None = None
    llm_repeat_count: int | None = None
    llm_license_reviewed: bool = False
    llm_model_card_reviewed: bool = False
    llm_local_files_only: bool = False
    schedule_lock_path: str | None = None
    schedule_fingerprints: dict[str, str] = field(default_factory=dict)
    exclusion_sources: dict[str, str] = field(default_factory=dict)
    stage_reports: dict[str, str] = field(default_factory=dict)
    stage_reused: dict[str, bool] = field(default_factory=dict)
    development_gates: dict[str, dict[str, object]] = field(default_factory=dict)
    implementation_validated: bool = False
    prior_m43b_target_validated: bool = False
    corpus_validated: bool = False
    split_isolation_validated: bool = False
    rule_router_validated: bool = False
    classifier_fixture_validated: bool = False
    llm_router_fixture_validated: bool = False
    controller_registry_validated: bool = False
    experiment_manifest_validated: bool = False
    final_schedules_locked: bool = False
    classifier_training_completed: bool = False
    classifier_checkpoint_selected: bool = False
    classifier_calibration_validated: bool = False
    llm_router_loaded: bool = False
    llm_prompt_locked: bool = False
    language_validation_completed: bool = False
    language_development_completed: bool = False
    oracle_control_development_completed: bool = False
    predicted_control_development_completed: bool = False
    control_development_completed: bool = False
    rejection_noop_probe_validated: bool = False
    failure_attribution_validated: bool = False
    development_quality_gate_passed: bool = False
    final_benchmark_authorized: bool = False
    final_benchmark_completed: bool = False
    language_final_accessed: bool = False
    control_final_accessed: bool = False
    m42_final_accessed: bool = False
    test_split_accessed: bool = False
    historical_fresh_accessed: bool = False
    smolvla_go: bool = False
    physical_target_validated: bool = False

    def check(self, name: str, condition: bool, detail: str) -> None:
        status = "pass" if condition else "fail"
        print(f"[{status.upper()}] {name}: {detail}")
        self.checks.append({"name": name, "status": status, "detail": detail})

    @property
    def failed(self) -> bool:
        return any(check["status"] == "fail" for check in self.checks)

    @property
    def passed(self) -> bool:
        required = (
            self.implementation_validated,
            self.prior_m43b_target_validated,
            self.corpus_validated,
            self.split_isolation_validated,
            self.rule_router_validated,
            self.controller_registry_validated,
            self.experiment_manifest_validated,
            self.final_schedules_locked,
            self.classifier_training_completed,
            self.classifier_checkpoint_selected,
            self.classifier_calibration_validated,
            self.llm_router_loaded,
            self.llm_prompt_locked,
            self.llm_license_reviewed,
            self.llm_model_card_reviewed,
            self.language_validation_completed,
            self.language_development_completed,
            self.oracle_control_development_completed,
            self.predicted_control_development_completed,
            self.control_development_completed,
            self.rejection_noop_probe_validated,
            self.failure_attribution_validated,
            self.physical_target_validated,
        )
        forbidden = (
            self.final_benchmark_completed,
            self.language_final_accessed,
            self.control_final_accessed,
            self.m42_final_accessed,
            self.test_split_accessed,
            self.historical_fresh_accessed,
            self.smolvla_go,
        )
        return (
            not self.failed
            and all(required)
            and not any(forbidden)
            and self.final_benchmark_authorized == self.development_quality_gate_passed
            and all(
                isinstance(value, str) and bool(value)
                for value in (
                    self.llm_model_id,
                    self.llm_model_revision,
                    self.llm_tokenizer_revision,
                    self.llm_license,
                    self.llm_dtype,
                )
            )
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "verification_mode": "target_development",
            **{key: value for key, value in asdict(self).items() if key != "checks"},
            "checks": self.checks,
            "passed": self.passed,
        }


class _FixtureEncoder(nn.Module):
    def __init__(self, *, vocabulary_size: int = 32, hidden_size: int = 12) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class _FixtureGenerator:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.call_count = 0

    def generate(self, prompt: str, *, max_new_tokens: int) -> str:
        if not prompt or max_new_tokens <= 0:
            raise RuntimeError("fixture generation received an invalid request")
        self.call_count += 1
        return self.outputs.pop(0)


class _FixtureEnvironment:
    def __init__(self) -> None:
        self.reset_count = 0
        self.step_count = 0
        self.last_reset: dict[str, object] | None = None

    def reset(self, *, seed: int, options: dict[str, object]) -> None:
        self.reset_count += 1
        self.last_reset = {"seed": seed, "options": options}

    def step(self, action: np.ndarray) -> None:
        if action.shape != (8,):
            raise RuntimeError("fixture action must preserve the eight-component contract")
        self.step_count += 1


def _fixture_exclusion_config() -> ControlScheduleConfig:
    sources = tuple(
        SeedExclusionSource(
            source_id=source_id,
            source_fingerprint=f"sha256:{index + 1:064x}",
            scene_seeds=(10_000 + index,),
        )
        for index, source_id in enumerate(REQUIRED_EXCLUSION_SOURCE_IDS)
    )
    return ControlScheduleConfig(exclusion_sources=sources)


def _verify_corpus_and_schedules(report: Report) -> None:
    first = build_language_corpus()
    second = build_language_corpus()
    counts = language_corpus_counts(first)
    expected_counts = {
        LanguageSplit.TRAIN: {"routeable": 600, "rejected": 300, "total": 900},
        LanguageSplit.VALIDATION: {"routeable": 180, "rejected": 120, "total": 300},
        LanguageSplit.DEVELOPMENT: {"routeable": 240, "rejected": 180, "total": 420},
        LanguageSplit.FINAL: {"routeable": 360, "rejected": 240, "total": 600},
    }
    stable = first.manifest.corpus_fingerprint == second.manifest.corpus_fingerprint
    exact_counts = all(
        dict(counts[split]) == expected for split, expected in expected_counts.items()
    )
    valid_visible_labels = all(
        (example.expected_task_spec is not None and example.expected_rejection_reason is None)
        if example.expected_status is RouterStatus.ROUTE
        else (example.expected_task_spec is None and example.expected_rejection_reason is not None)
        for example in first.examples
    )
    report.check(
        "deterministic exact-count corpus",
        stable and exact_counts and valid_visible_labels,
        f"fingerprint={first.manifest.corpus_fingerprint}; counts={expected_counts}",
    )
    report.corpus_fingerprint = first.manifest.corpus_fingerprint
    report.corpus_validated = stable and exact_counts and valid_visible_labels

    isolation = first.isolation_report
    isolated = isolation.passed and first.manifest.family_isolation_validated
    report.check(
        "family and near-duplicate split isolation",
        isolated,
        f"isolation_fingerprint={isolation.report_fingerprint}",
    )
    report.split_isolation_fingerprint = isolation.report_fingerprint
    report.split_isolation_validated = isolated

    bundle = build_m5a_schedule_bundle(corpus=first, config=_fixture_exclusion_config())
    locks = {
        bundle.language_development.schedule_id: bundle.language_development.schedule_fingerprint,
        bundle.language_final.schedule_id: bundle.language_final.schedule_fingerprint,
        bundle.control_development.schedule_id: bundle.control_development.schedule_fingerprint,
        bundle.control_final.schedule_id: bundle.control_final.schedule_fingerprint,
    }
    expected_ids = {
        M5A_LANGUAGE_DEV_SCHEDULE_ID,
        M5A_LANGUAGE_FINAL_SCHEDULE_ID,
        M5A_CONTROL_DEV_SCHEDULE_ID,
        M5A_CONTROL_FINAL_SCHEDULE_ID,
    }
    language_blocked = False
    control_blocked = False
    try:
        first.examples_for_split(LanguageSplit.FINAL)
    except FinalLanguageAccessError:
        language_blocked = True
    try:
        materialize_control_schedule(bundle.control_final)
    except FinalControlScheduleAccessError:
        control_blocked = True
    locked = (
        set(locks) == expected_ids
        and bundle.language_final.sealed
        and bundle.control_final.sealed
        and len(bundle.development_episodes) == 72
        and bundle.control_final.episode_count == 180
        and language_blocked
        and control_blocked
    )
    report.check(
        "development/final schedule locks",
        locked,
        "four locks exist; final text/control materialization was denied",
    )
    report.schedule_fingerprints = locks
    report.final_schedules_locked = locked


def _verify_rule_router(report: Report) -> RuleRouterV0:
    router = RuleRouterV0()
    direct = router.route("Pick up the red cube and place it in the left bin.")
    synonym = router.route("Please move the azure cube into the right container.")
    conflict = router.route("Move the red and blue cubes into the right bin.")
    unsupported = router.route("Place the yellow cube in the left bin.")
    malformed = router.route("")
    valid = (
        direct.status is RouterStatus.ROUTE
        and direct.target_object_id == "red_cube"
        and direct.target_bin_id == "left_bin"
        and synonym.status is RouterStatus.ROUTE
        and synonym.target_object_id == "blue_cube"
        and conflict.status is RouterStatus.REJECT_AMBIGUOUS
        and unsupported.status is RouterStatus.REJECT_UNSUPPORTED
        and malformed.status is RouterStatus.REJECT_MALFORMED
        and RouterDecision.from_dict(direct.to_dict()) == direct
    )
    report.check("RuleRouterV0 fixture", valid, "route, synonym, conflict, unsupported, malformed")
    report.rule_router_validated = valid
    return router


def _verify_classifier_fixture(report: Report) -> None:
    torch.manual_seed(0)
    model = FactorizedTextClassifierV0(_FixtureEncoder(), dropout=0.0)
    model.train()
    input_ids = torch.tensor(
        [[1, 2, 3, 0], [4, 5, 0, 0], [6, 7, 8, 9], [10, 11, 0, 0]],
        dtype=torch.long,
    )
    attention_mask = (input_ids != 0).to(dtype=torch.long)
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    loss = compute_factorized_text_loss(
        output,
        status_labels=torch.tensor([0, 1, 0, 3], dtype=torch.long),
        object_labels=torch.tensor([0, 3, 2, 3], dtype=torch.long),
        bin_labels=torch.tensor([0, 2, 1, 2], dtype=torch.long),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss.total.backward()
    finite_gradients = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    optimizer.step()

    calibration = fit_validation_temperature(
        torch.tensor(
            [[5.0, 0.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0], [0.0, 0.0, 5.0, 0.0]],
            dtype=torch.float32,
        ),
        torch.tensor([0, 1, 2], dtype=torch.long),
        evidence_split="validation",
        iterations=16,
    )
    threshold = select_validation_routing_threshold(
        (
            RoutingThresholdExample(
                True, "red_cube", "left_bin", "route", "red_cube", "left_bin", 0.95
            ),
            RoutingThresholdExample(
                True, "blue_cube", "right_bin", "route", "blue_cube", "right_bin", 0.92
            ),
            RoutingThresholdExample(False, None, None, "route", "red_cube", "left_bin", 0.20),
            RoutingThresholdExample(False, None, None, "reject_unsupported", None, None, 0.80),
        ),
        evidence_split="validation",
        candidates=(0.5, 0.9),
    )
    valid = (
        output.status_logits.shape == (4, 4)
        and output.object_logits.shape == (4, 4)
        and output.bin_logits.shape == (4, 3)
        and loss.routed_examples == 2
        and bool(torch.isfinite(loss.total))
        and finite_gradients
        and calibration.validation_examples == 3
        and threshold.threshold == 0.9
        and threshold.false_route_rate == 0.0
    )
    report.check(
        "classifier forward/backward and calibration fixtures",
        valid,
        f"loss={float(loss.total.detach()):.6f}; temperature={calibration.temperature:.6f}; threshold={threshold.threshold}",
    )
    report.classifier_fixture_validated = valid


def _llm_config() -> StructuredLLMRouterConfig:
    return StructuredLLMRouterConfig(
        model_id="fixture/local-instruct",
        model_revision="fixture-model-revision",
        tokenizer_revision="fixture-tokenizer-revision",
        dtype="float32",
        maximum_format_repair_attempts=1,
    )


def _verify_llm_fixture(report: Report) -> None:
    parsed = parse_strict_router_json(
        '{"status":"route","target_object_id":"green_cube",'
        '"target_bin_id":"right_bin","reason":"route"}'
    )
    route_generator = _FixtureGenerator(
        [
            '{"status":"route","target_object_id":"green_cube",'
            '"target_bin_id":"right_bin","reason":"route"}'
        ]
    )
    route = StructuredLocalLLMRouterV0(config=_llm_config(), generator=route_generator).route(
        "Move the green cube to the right bin."
    )
    repair_generator = _FixtureGenerator(
        [
            "not-json",
            '{"status":"reject_unsupported","target_object_id":null,'
            '"target_bin_id":null,"reason":"unsupported_action"}',
        ]
    )
    repaired = StructuredLocalLLMRouterV0(config=_llm_config(), generator=repair_generator).route(
        "Open the drawer."
    )
    exhausted_generator = _FixtureGenerator(["bad", "still bad"])
    exhausted = StructuredLocalLLMRouterV0(
        config=_llm_config(), generator=exhausted_generator
    ).route("???")
    valid = (
        parsed.target_object_id == "green_cube"
        and route.status is RouterStatus.ROUTE
        and route.confidence.available is False
        and repaired.status is RouterStatus.REJECT_UNSUPPORTED
        and repair_generator.call_count == 2
        and repaired.evidence["format_repair_attempts"] == 1
        and exhausted.status is RouterStatus.REJECT_MALFORMED
        and exhausted.task_spec is None
        and exhausted_generator.call_count == 2
    )
    report.check(
        "strict local-LLM schema and bounded repair fixtures",
        valid,
        "strict route parsed; one repair succeeded; two malformed outputs safely rejected",
    )
    report.llm_router_fixture_validated = valid


def _verify_registry_dispatch_and_attribution(report: Report, router: RuleRouterV0) -> None:
    registry = build_fixture_controller_registry()
    roundtrip = ControllerRegistry.from_dict(registry.to_dict())
    registry_valid = (
        len(registry.entries) == 6
        and roundtrip.registry_fingerprint == registry.registry_fingerprint
        and tuple(entry.task_spec for entry in registry.entries) == CANONICAL_TASK_SPECS
    )
    report.check(
        "six-controller fixture registry",
        registry_valid,
        f"registry_fingerprint={registry.registry_fingerprint}",
    )
    report.controller_registry_fingerprint = registry.registry_fingerprint

    environment = _FixtureEnvironment()
    loader = FixturePerTaskControllerLoader()
    dispatcher = ControllerDispatcher(registry=registry, loader=loader)
    rejected = dispatcher.dispatch(
        router.route("Open the drawer."),
        environment=environment,
        evaluation_id="fixture-rejection",
        oracle_task_spec=CANONICAL_TASK_SPECS[0],
        scene_seed=123,
    )
    zero_dispatch = (
        rejected.dispatch.safe_rejection
        and not rejected.dispatch.dispatched
        and not rejected.dispatch.policy_called
        and not rejected.dispatch.environment_reset_called
        and rejected.dispatch.environment_step_count == 0
        and loader.load_count == 0
        and environment.reset_count == 0
        and environment.step_count == 0
    )

    decision = router.route("Pick up the red cube and place it in the left bin.")
    routed = dispatcher.dispatch(
        decision,
        environment=environment,
        evaluation_id="fixture-route",
        oracle_task_spec=CANONICAL_TASK_SPECS[0],
        scene_seed=123,
    )
    fixture_dispatch = (
        routed.control is not None
        and routed.control.success
        and routed.dispatch.dispatched
        and routed.dispatch.controller_task_id == decision.task_id
        and environment.reset_count == 1
        and environment.step_count == 1
        and routed.control.action_evidence.raw_actions
        == routed.control.action_evidence.executed_actions
    )

    wrong = RouterDecision.route(
        task_spec=TaskSpec(
            target_object_id="red_cube",
            target_bin_id="right_bin",
            instruction_template_id="canonical_v0",
        ),
        confidence=RouterConfidence.unavailable(),
        router_name="FixtureWrongRouter",
        router_version="v0",
    )
    wrong_attribution = attribute_end_to_end(
        FailureAttributionEvidence(
            expected_task_spec=CANONICAL_TASK_SPECS[0],
            decision=wrong,
        )
    )
    attribution_valid = (
        routed.failure_attribution is FailureAttribution.ROUTING_CORRECT_CONTROL_SUCCESS
        and rejected.failure_attribution is FailureAttribution.ROUTING_FALSE_REJECTION
        and wrong_attribution is FailureAttribution.ROUTING_WRONG_BIN
    )
    report.check(
        "zero-dispatch rejection and fixture controller dispatch",
        zero_dispatch and fixture_dispatch,
        "rejection performed no load/reset/step; route selected one controller and one step",
    )
    report.check(
        "routing/control failure attribution fixture",
        attribution_valid,
        "correct success, false rejection, and wrong-bin routing remain distinct",
    )
    report.controller_registry_validated = (
        registry_valid and zero_dispatch and fixture_dispatch and attribution_valid
    )


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise M5ATargetVerificationError(f"{label} must be a JSON object")
    return cast(Mapping[str, object], value)


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    resolved = _resolved_unlinked(path, label=label)
    if not resolved.is_file():
        raise M5ATargetVerificationError(f"{label} is missing: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise M5ATargetVerificationError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise M5ATargetVerificationError(f"{label} must be a JSON object")
    return cast(dict[str, object], payload)


def _require_flag(payload: Mapping[str, object], key: str, expected: bool) -> None:
    value = payload.get(key)
    if value is not expected:
        raise M5ATargetVerificationError(
            f"{key} must be {str(expected).lower()}, observed {value!r}"
        )


def _require_exact(value: object, expected: object, *, label: str) -> None:
    if value != expected:
        raise M5ATargetVerificationError(f"{label} differs: {value!r} != {expected!r}")


def _require_nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise M5ATargetVerificationError(f"{label} must be a non-negative integer")
    return value


def _require_probability(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise M5ATargetVerificationError(f"{label} must be a probability")
    converted = float(value)
    if not np.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise M5ATargetVerificationError(f"{label} must be finite and in [0, 1]")
    return converted


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise M5ATargetVerificationError(f"{label} must be a SHA-256 fingerprint")
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise M5ATargetVerificationError(f"{label} must be a lowercase SHA-256 fingerprint")
    return value


def _validate_m43b_verification(path: Path) -> dict[str, object]:
    payload = _read_json_object(path, label="M4.3b independent target verification")
    _require_exact(
        payload.get("schema_version"),
        M43B_INDEPENDENT_SCHEMA,
        label="M4.3b independent verification schema",
    )
    _require_flag(payload, "passed", True)
    required_true = (
        "implementation_validated",
        "semantic_audit_completed",
        "factor_film_implementation_validated",
        "factor_film_fixture_training_validated",
        "factor_film_training_completed",
        "factor_film_checkpoints_complete",
        "factor_film_checkpoint_selected",
        "validation_only_selection_validated",
        "factor_film_reload_validated",
        "development_benchmark_completed",
        "post_grasp_analysis_completed",
        "first_interaction_analysis_completed",
        "raw_action_metrics_validated",
        "runtime_action_metrics_validated",
        "physical_target_validated",
    )
    for key in required_true:
        _require_flag(payload, key, True)
    for key in (
        "final_schedule_accessed",
        "test_split_accessed",
        "fresh_seed_accessed",
        "smolvla_go",
    ):
        _require_flag(payload, key, False)
    quality = payload.get("development_quality_gate_passed")
    authorization = payload.get("final_benchmark_authorized")
    if not isinstance(quality, bool) or authorization is not quality:
        raise M5ATargetVerificationError(
            "M4.3b final authorization must exactly match its development quality gate"
        )
    _require_sha256(
        payload.get("evaluation_evidence_fingerprint"),
        label="M4.3b evaluation evidence fingerprint",
    )
    return payload


def _sealed_language_lock(bundle: M5AScheduleBundle) -> dict[str, object]:
    language_final = bundle.language_final
    return {
        "schedule_id": language_final.schedule_id,
        "schedule_fingerprint": language_final.schedule_fingerprint,
        "corpus_fingerprint": language_final.corpus_fingerprint,
        "split_fingerprint": language_final.split_fingerprint,
        "example_count": len(language_final.ordered_example_ids),
        "sealed": True,
        "texts_materialized": False,
    }


def _sealed_control_lock(bundle: M5AScheduleBundle) -> dict[str, object]:
    control_final = bundle.control_final
    return {
        "schedule_id": control_final.schedule_id,
        "schedule_fingerprint": control_final.schedule_fingerprint,
        "ordered_scene_seeds": list(control_final.ordered_scene_seeds),
        "scene_count": len(control_final.ordered_scene_seeds),
        "episode_count": control_final.episode_count,
        "exclusion_digest": control_final.exclusion_digest,
        "language_schedule_fingerprint": control_final.language_schedule_fingerprint,
        "sealed": True,
        "language_example_ids_materialized": False,
        "episodes_materialized": False,
    }


def _build_target_schedule_payload(
    *,
    m43b_verification: Mapping[str, object],
) -> tuple[dict[str, object], M5AScheduleBundle, GeneratedLanguageCorpus]:
    m42_development, _m42_final = validate_locked_schedules()
    m43_fingerprint = _require_sha256(
        m43b_verification.get("evaluation_evidence_fingerprint"),
        label="M4.3b evaluation evidence fingerprint",
    )
    exclusions = build_authoritative_seed_exclusions(
        m43_development_fingerprint=m43_fingerprint,
        m43_development_scene_seeds=m42_development.ordered_scene_seeds,
    )
    config = ControlScheduleConfig(exclusion_sources=exclusions)
    corpus = build_language_corpus()
    bundle = build_m5a_schedule_bundle(corpus=corpus, config=config)
    payload = {
        "schema_version": TARGET_SCHEDULE_LOCK_SCHEMA,
        "corpus_fingerprint": corpus.manifest.corpus_fingerprint,
        "m43b_evaluation_evidence_fingerprint": m43_fingerprint,
        "control_schedule_config": config.to_dict(),
        "language_development": bundle.language_development.to_dict(),
        "language_final": _sealed_language_lock(bundle),
        "control_development": bundle.control_development.to_dict(),
        "control_final": _sealed_control_lock(bundle),
        "all_four_schedules_locked_before_training": True,
        "language_final_accessed": False,
        "control_final_accessed": False,
        "m42_final_accessed": False,
        "test_split_accessed": False,
        "historical_fresh_accessed": False,
        "smolvla_go": False,
        "final_texts_materialized": False,
        "final_episodes_materialized": False,
    }
    return payload, bundle, corpus


def _lock_target_schedules(path: Path, payload: Mapping[str, object]) -> bool:
    if path.exists():
        existing = _read_json_object(path, label="M5A target schedule lock")
        if existing != dict(payload):
            raise M5ATargetVerificationError(
                "existing M5A target schedule lock differs from authoritative inputs"
            )
        return True
    try:
        atomic_write_json(path, dict(payload), immutable=True)
    except FileExistsError as error:
        existing = _read_json_object(path, label="M5A target schedule lock")
        if existing != dict(payload):
            raise M5ATargetVerificationError(
                "concurrently created M5A target schedule lock differs"
            ) from error
        return True
    return False


def _same_path(value: object, expected: Path, *, label: str) -> None:
    if not isinstance(value, str):
        raise M5ATargetVerificationError(f"{label} must be a path string")
    observed = _resolved_unlinked(Path(value), label=label)
    target = _resolved_unlinked(expected, label=f"expected {label}")
    if observed != target:
        raise M5ATargetVerificationError(f"{label} differs from the configured path")


def _validate_corpus_report(
    payload: Mapping[str, object],
    *,
    corpus_root: Path,
    corpus_fingerprint: str,
    language_development_fingerprint: str,
    language_final_fingerprint: str,
) -> None:
    _require_exact(payload.get("schema_version"), CORPUS_COMMAND_SCHEMA, label="corpus schema")
    for key in ("passed", "corpus_validated", "split_isolation_validated", "final_schedule_locked"):
        _require_flag(payload, key, True)
    _require_flag(payload, "dry_run", False)
    _require_flag(payload, "language_final_accessed", False)
    _require_exact(
        payload.get("corpus_fingerprint"), corpus_fingerprint, label="corpus fingerprint"
    )
    _require_sha256(payload.get("archive_fingerprint"), label="corpus archive fingerprint")
    _same_path(payload.get("output_root"), corpus_root, label="corpus output root")
    manifest = _require_mapping(payload.get("corpus_manifest"), label="corpus manifest")
    _require_exact(
        manifest.get("corpus_fingerprint"), corpus_fingerprint, label="corpus manifest fingerprint"
    )
    _require_flag(manifest, "family_isolation_validated", True)
    _require_flag(manifest, "final_locked", True)
    language_development = _require_mapping(
        payload.get("language_development_schedule"), label="language development schedule"
    )
    language_final = _require_mapping(
        payload.get("language_final_schedule"), label="language final schedule"
    )
    _require_exact(
        language_development.get("schedule_fingerprint"),
        language_development_fingerprint,
        label="language development schedule fingerprint",
    )
    _require_exact(
        language_final.get("schedule_fingerprint"),
        language_final_fingerprint,
        label="language final schedule fingerprint",
    )
    _require_flag(language_final, "sealed", True)


def _validate_classifier_report(
    payload: Mapping[str, object],
    *,
    classifier_output_root: Path,
    corpus_fingerprint: str,
) -> Path:
    _require_exact(
        payload.get("schema_version"), CLASSIFIER_COMMAND_SCHEMA, label="classifier command schema"
    )
    _require_exact(payload.get("mode"), "target_development", label="classifier mode")
    for key in (
        "passed",
        "classifier_training_completed",
        "classifier_checkpoint_selected",
        "classifier_calibration_validated",
        "artifact_reload_validated",
        "cuda_training_validated",
    ):
        _require_flag(payload, key, True)
    for key in (
        "classifier_fixture_completed",
        "development_accessed",
        "language_final_accessed",
        "control_final_accessed",
        "m42_final_accessed",
        "test_split_accessed",
        "historical_fresh_accessed",
        "smolvla_go",
        "physical_target_validated",
    ):
        _require_flag(payload, key, False)
    _require_sha256(payload.get("run_fingerprint"), label="classifier run fingerprint")
    artifact_fingerprint = _require_sha256(
        payload.get("artifact_fingerprint"), label="classifier artifact fingerprint"
    )
    artifact_value = payload.get("artifact_root")
    if not isinstance(artifact_value, str):
        raise M5ATargetVerificationError("classifier artifact_root must be a path string")
    artifact_root = _resolved_unlinked(Path(artifact_value), label="classifier artifact root")
    expected_parent = _resolved_unlinked(classifier_output_root, label="classifier output root")
    if artifact_root.parent != expected_parent or not artifact_root.is_dir():
        raise M5ATargetVerificationError(
            "classifier artifact must be one completed child of the configured output root"
        )
    complete = _read_json_object(artifact_root / "complete.json", label="classifier completion")
    _require_flag(complete, "passed", True)
    _require_exact(
        complete.get("artifact_fingerprint"),
        artifact_fingerprint,
        label="classifier completion artifact fingerprint",
    )
    evidence = _require_mapping(payload.get("run_evidence"), label="classifier run evidence")
    _require_exact(
        evidence.get("schema_version"),
        CLASSIFIER_RUN_EVIDENCE_SCHEMA,
        label="classifier run evidence schema",
    )
    _require_exact(evidence.get("mode"), "target_development", label="classifier evidence mode")
    for key, expected in (
        ("model_id", TEXT_CLASSIFIER_MODEL_ID),
        ("model_revision", TEXT_CLASSIFIER_MODEL_REVISION),
        ("tokenizer_revision", TEXT_CLASSIFIER_TOKENIZER_REVISION),
    ):
        _require_exact(
            evidence.get(key),
            expected,
            label=f"classifier pinned {key}",
        )
    corpus_manifest = _require_mapping(
        evidence.get("corpus_manifest"), label="classifier corpus manifest"
    )
    _require_exact(
        corpus_manifest.get("corpus_fingerprint"),
        corpus_fingerprint,
        label="classifier corpus fingerprint",
    )
    git_state = _require_mapping(evidence.get("git_state"), label="classifier Git state")
    _require_flag(git_state, "dirty", False)
    calibration = _require_mapping(
        payload.get("calibration_selection"), label="classifier calibration selection"
    )
    if set(calibration) != {"temperature", "threshold"}:
        raise M5ATargetVerificationError(
            "classifier calibration selection must contain temperature and threshold only"
        )
    return artifact_root


def _language_router_payload(
    payload: Mapping[str, object], router_label: str
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    comparison = _require_mapping(payload.get("comparison"), label="language comparison")
    development = _require_mapping(
        comparison.get("development"), label="language development comparison"
    )
    router = _require_mapping(
        development.get(router_label), label=f"{router_label} language development result"
    )
    summary = _require_mapping(router.get("summary"), label=f"{router_label} language summary")
    supplemental = _require_mapping(
        router.get("supplemental_metrics"),
        label=f"{router_label} supplemental language metrics",
    )
    return summary, supplemental


def _validate_language_metric_pair(
    summary: Mapping[str, object], supplemental: Mapping[str, object], key: str
) -> None:
    left = _require_probability(summary.get(key), label=f"language summary {key}")
    right = _require_probability(supplemental.get(key), label=f"supplemental language {key}")
    if left != right:
        raise M5ATargetVerificationError(f"language {key} differs between independent aggregates")


def _validate_language_report(
    payload: Mapping[str, object],
    *,
    corpus_fingerprint: str,
    artifact_fingerprint: str,
    language_development_fingerprint: str,
    language_final_fingerprint: str,
    llm_model_id: str,
    llm_model_revision: str,
    llm_tokenizer_revision: str,
    llm_license: str,
    llm_dtype: str,
    llm_maximum_new_tokens: int,
    repeat_count: int,
    local_files_only: bool,
) -> None:
    _require_exact(
        payload.get("schema_version"), LANGUAGE_COMMAND_SCHEMA, label="language command schema"
    )
    _require_exact(payload.get("mode"), "target_development", label="language evaluation mode")
    for key in (
        "passed",
        "target_development_executed",
        "validation_completed_before_development",
        "classifier_checkpoint_selected",
        "classifier_calibration_validated",
        "llm_router_loaded",
        "llm_prompt_locked",
        "language_validation_completed",
        "language_development_completed",
    ):
        _require_flag(payload, key, True)
    for key in (
        "llm_router_fixture_loaded",
        "language_fixture_completed",
        "physical_target_validated",
        "final_benchmark_authorized",
        "language_final_accessed",
        "control_final_accessed",
        "m42_final_accessed",
        "test_split_accessed",
        "historical_fresh_accessed",
        "controller_dispatched",
        "environment_reset_called",
        "smolvla_go",
    ):
        _require_flag(payload, key, False)
    _require_exact(
        payload.get("environment_step_count"), 0, label="language environment step count"
    )
    _require_exact(payload.get("m2_expert_call_count"), 0, label="language M2 expert call count")
    _require_exact(payload.get("repeat_count"), repeat_count, label="language repeat count")
    _require_exact(
        payload.get("local_files_only"),
        local_files_only,
        label="language local-files-only provenance",
    )
    _require_exact(
        payload.get("evaluation_order"),
        ["validation", "development"],
        label="language evaluation order",
    )
    _require_exact(
        payload.get("corpus_fingerprint"), corpus_fingerprint, label="language corpus fingerprint"
    )
    development_lock = _require_mapping(
        payload.get("language_development_schedule"), label="language development schedule"
    )
    _require_exact(
        development_lock.get("schedule_fingerprint"),
        language_development_fingerprint,
        label="language development lock fingerprint",
    )
    final_lock = _require_mapping(payload.get("language_final_lock"), label="language final lock")
    _require_exact(
        final_lock.get("schedule_fingerprint"),
        language_final_fingerprint,
        label="language final lock fingerprint",
    )
    _require_flag(final_lock, "sealed", True)
    _require_flag(final_lock, "texts_materialized", False)
    identities = _require_mapping(payload.get("router_identities"), label="router identities")
    if set(identities) != {"rule", "classifier", "llm"}:
        raise M5ATargetVerificationError("language evaluation must contain exactly three routers")
    classifier = _require_mapping(identities.get("classifier"), label="classifier identity")
    _require_exact(
        classifier.get("artifact_fingerprint"),
        artifact_fingerprint,
        label="language classifier artifact fingerprint",
    )
    _require_sha256(classifier.get("router_fingerprint"), label="classifier router fingerprint")
    llm = _require_mapping(identities.get("llm"), label="LLM identity")
    _require_exact(llm.get("router_name"), "StructuredLocalLLMRouterV0", label="LLM router name")
    _require_exact(llm.get("declared_license"), llm_license, label="LLM declared license")
    _require_flag(llm, "official_model_card_license_reviewed", True)
    parameter_count = _require_nonnegative_int(
        llm.get("parameter_count"), label="LLM parameter count"
    )
    if not 500_000_000 <= parameter_count <= 3_000_000_000:
        raise M5ATargetVerificationError("LLM parameter count lies outside 0.5B-3B")
    _require_exact(llm.get("generator"), "TransformersLocalTextGenerator", label="LLM generator")
    _require_exact(llm.get("prompt_example_split"), "train", label="LLM prompt example split")
    _require_sha256(llm.get("prompt_fingerprint"), label="LLM prompt fingerprint")
    _require_sha256(llm.get("router_fingerprint"), label="LLM router fingerprint")
    llm_config = _require_mapping(llm.get("config"), label="LLM config")
    for key, expected in (
        ("model_id", llm_model_id),
        ("model_revision", llm_model_revision),
        ("tokenizer_revision", llm_tokenizer_revision),
        ("dtype", llm_dtype),
        ("quantization", "none"),
        ("maximum_new_tokens", llm_maximum_new_tokens),
        ("maximum_format_repair_attempts", 1),
    ):
        _require_exact(llm_config.get(key), expected, label=f"LLM config {key}")
    comparison = _require_mapping(payload.get("comparison"), label="language comparison")
    if set(comparison) != {"validation", "development"}:
        raise M5ATargetVerificationError("language comparison must contain validation/development")
    expected_counts = {
        "validation": (300, 180, 120),
        "development": (420, 240, 180),
    }
    metric_pairs = (
        "object_accuracy",
        "bin_accuracy",
        "false_route_rate",
        "ambiguous_rejection_recall",
        "unsupported_rejection_recall",
        "malformed_rejection_recall",
    )
    for split, counts in expected_counts.items():
        split_payload = _require_mapping(comparison.get(split), label=f"language {split}")
        if set(split_payload) != {"rule", "classifier", "llm"}:
            raise M5ATargetVerificationError(f"language {split} must contain all three routers")
        for router_label in ("rule", "classifier", "llm"):
            router_payload = _require_mapping(
                split_payload.get(router_label), label=f"{split} {router_label} result"
            )
            summary = _require_mapping(
                router_payload.get("summary"), label=f"{split} {router_label} summary"
            )
            supplemental = _require_mapping(
                router_payload.get("supplemental_metrics"),
                label=f"{split} {router_label} supplemental metrics",
            )
            for key, expected in zip(
                ("example_count", "routeable_count", "rejected_count"), counts, strict=True
            ):
                _require_exact(summary.get(key), expected, label=f"{split} {router_label} {key}")
            for key in metric_pairs:
                _validate_language_metric_pair(summary, supplemental, key)
            _require_probability(
                supplemental.get("schema_valid_output_rate"),
                label=f"{split} {router_label} schema-valid output rate",
            )
            _require_probability(
                summary.get("deterministic_repeatability"),
                label=f"{split} {router_label} repeatability",
            )


def _validate_control_summary(summary: Mapping[str, object], *, router_label: str) -> None:
    _require_exact(summary.get("episode_count"), 72, label=f"{router_label} episode count")
    _require_flag(summary, "wrong_object_interaction_available", True)
    _require_exact(
        summary.get("infrastructure_failure_count"),
        0,
        label=f"{router_label} infrastructure failure count",
    )
    for key in (
        "end_to_end_success_count",
        "routing_correct_control_success_count",
        "routing_correct_control_failure_count",
        "routing_wrong_object_count",
        "routing_wrong_bin_count",
        "routing_false_rejection_count",
        "target_in_wrong_bin_count",
        "target_off_table_count",
        "arm_projected_component_count",
        "invalid_action_count",
        "malformed_action_count",
        "nonfinite_action_count",
        "dispatch_after_rejection_count",
        "m2_expert_call_count",
    ):
        _require_nonnegative_int(summary.get(key), label=f"{router_label} {key}")
    attribution = _require_mapping(
        summary.get("failure_attribution_counts"),
        label=f"{router_label} failure attribution counts",
    )
    expected_attributions = {value.value for value in FailureAttribution}
    if set(attribution) != expected_attributions:
        raise M5ATargetVerificationError(
            f"{router_label} failure attribution does not contain every declared category"
        )
    attribution_total = sum(
        _require_nonnegative_int(value, label=f"{router_label} failure attribution")
        for value in attribution.values()
    )
    _require_exact(attribution_total, 72, label=f"{router_label} failure attribution total")
    _require_exact(
        summary.get("end_to_end_success_count"),
        attribution.get("routing_correct_control_success"),
        label=f"{router_label} success attribution",
    )
    _require_exact(
        summary.get("routing_correct_control_failure_count"),
        attribution.get("routing_correct_control_failure"),
        label=f"{router_label} control-failure attribution",
    )
    _require_exact(
        summary.get("routing_false_rejection_count"),
        attribution.get("routing_false_rejection"),
        label=f"{router_label} false-rejection attribution",
    )


def _validate_control_report(
    payload: Mapping[str, object],
    *,
    corpus_fingerprint: str,
    control_schedule_fingerprint: str,
    classifier_artifact_fingerprint: str,
    llm_model_id: str,
    llm_model_revision: str,
    llm_tokenizer_revision: str,
    llm_dtype: str,
    llm_maximum_new_tokens: int,
    language_evaluation_fingerprint: str,
    language_artifact_fingerprint: str,
    language_prompt_fingerprint: str,
    router_evaluation_evidence_root: Path,
) -> ControllerRegistry:
    _require_exact(
        payload.get("schema_version"), CONTROL_COMPLETION_SCHEMA, label="control completion schema"
    )
    for key in (
        "passed",
        "physical_execution",
        "rejection_noop_probe_validated",
        "zero_dispatch_after_rejection_validated",
        "m2_expert_free_validated",
    ):
        _require_flag(payload, key, True)
    for key in (
        "dry_run",
        "language_final_accessed",
        "control_final_accessed",
        "m42_final_accessed",
        "test_split_accessed",
        "historical_fresh_accessed",
        "smolvla_go",
    ):
        _require_flag(payload, key, False)
    _require_exact(payload.get("completed_episode_atoms"), 288, label="completed control atoms")
    _require_exact(payload.get("expected_episode_atoms"), 288, label="expected control atoms")
    _require_exact(
        payload.get("router_order"),
        ["oracle", "rule", "classifier", "llm"],
        label="control router order",
    )
    _require_exact(
        payload.get("corpus_fingerprint"), corpus_fingerprint, label="control corpus fingerprint"
    )
    _require_exact(
        payload.get("schedule_fingerprint"),
        control_schedule_fingerprint,
        label="control schedule fingerprint",
    )
    _require_exact(
        payload.get("dispatch_after_rejection_count"), 0, label="dispatch after rejection"
    )
    _require_exact(payload.get("m2_expert_call_count"), 0, label="control M2 expert calls")
    for key in (
        "invalid_action_count",
        "malformed_action_count",
        "nonfinite_action_count",
        "wrong_object_interaction_count",
    ):
        _require_nonnegative_int(payload.get(key), label=f"control {key}")
    summaries = _require_mapping(payload.get("summaries"), label="control summaries")
    if set(summaries) != {"oracle", "rule", "classifier", "llm"}:
        raise M5ATargetVerificationError("control summaries must contain four ordered baselines")
    for router_label in ("oracle", "rule", "classifier", "llm"):
        _validate_control_summary(
            _require_mapping(summaries.get(router_label), label=f"{router_label} summary"),
            router_label=router_label,
        )
    _require_flag(payload, "wrong_object_interaction_available", True)
    oracle_summary = _require_mapping(summaries.get("oracle"), label="oracle summary")
    for key in (
        "routing_wrong_object_count",
        "routing_wrong_bin_count",
        "routing_false_rejection_count",
        "dispatch_after_rejection_count",
        "m2_expert_call_count",
    ):
        _require_exact(oracle_summary.get(key), 0, label=f"oracle {key}")
    registry_payload = _require_mapping(
        payload.get("controller_registry"), label="controller registry"
    )
    registry = ControllerRegistry.from_dict(registry_payload)
    if (
        len(registry.entries) != 6
        or tuple(entry.task_spec for entry in registry.entries) != CANONICAL_TASK_SPECS
    ):
        raise M5ATargetVerificationError(
            "controller registry is not the canonical six-task library"
        )
    _require_exact(
        payload.get("controller_registry_fingerprint"),
        registry.registry_fingerprint,
        label="control registry fingerprint",
    )
    classifier_identity = _require_mapping(
        payload.get("classifier_identity"), label="control classifier identity"
    )
    _require_exact(
        classifier_identity.get("artifact_fingerprint"),
        classifier_artifact_fingerprint,
        label="control classifier artifact fingerprint",
    )
    llm_config = _require_mapping(payload.get("llm_config"), label="control LLM config")
    for key, expected in (
        ("model_id", llm_model_id),
        ("model_revision", llm_model_revision),
        ("tokenizer_revision", llm_tokenizer_revision),
        ("dtype", llm_dtype),
        ("maximum_new_tokens", llm_maximum_new_tokens),
    ):
        _require_exact(llm_config.get(key), expected, label=f"control LLM config {key}")
    _require_exact(
        payload.get("llm_prompt_fingerprint"),
        language_prompt_fingerprint,
        label="control LLM prompt fingerprint",
    )
    router_evaluation = _require_mapping(
        payload.get("router_evaluation_identity"),
        label="control router-evaluation identity",
    )
    _require_exact(
        router_evaluation.get("evaluation_fingerprint"),
        language_evaluation_fingerprint,
        label="control language-evaluation fingerprint",
    )
    _require_exact(
        router_evaluation.get("artifact_fingerprint"),
        language_artifact_fingerprint,
        label="control language artifact fingerprint",
    )
    _same_path(
        router_evaluation.get("evidence_root"),
        router_evaluation_evidence_root,
        label="control router-evaluation evidence root",
    )
    return registry


def _build_development_gate(
    *,
    router_label: str,
    language_report: Mapping[str, object],
    control_report: Mapping[str, object],
) -> M5ADevelopmentGate:
    if router_label not in {"classifier", "llm"}:
        raise M5ATargetVerificationError("development gates are only defined for learned routers")
    identities = _require_mapping(
        language_report.get("router_identities"), label="router identities"
    )
    identity = _require_mapping(identities.get(router_label), label=f"{router_label} identity")
    summary, supplemental = _language_router_payload(language_report, router_label)
    summaries = _require_mapping(control_report.get("summaries"), label="control summaries")
    oracle = _require_mapping(summaries.get("oracle"), label="oracle control summary")
    candidate = _require_mapping(
        summaries.get(router_label), label=f"{router_label} control summary"
    )
    oracle_denominator = _require_nonnegative_int(
        oracle.get("episode_count"), label="oracle episode count"
    )
    candidate_denominator = _require_nonnegative_int(
        candidate.get("episode_count"), label=f"{router_label} episode count"
    )
    if oracle_denominator != 72 or candidate_denominator != 72:
        raise M5ATargetVerificationError("development gate denominators must each equal 72")
    malformed = _require_nonnegative_int(
        candidate.get("malformed_action_count"), label=f"{router_label} malformed actions"
    )
    nonfinite = _require_nonnegative_int(
        candidate.get("nonfinite_action_count"), label=f"{router_label} nonfinite actions"
    )
    invalid = _require_nonnegative_int(
        candidate.get("invalid_action_count"), label=f"{router_label} invalid actions"
    )
    return M5ADevelopmentGate(
        candidate_router_name=(
            "FactorizedTextClassifierV0"
            if router_label == "classifier"
            else "StructuredLocalLLMRouterV0"
        ),
        candidate_router_fingerprint=_require_sha256(
            identity.get("router_fingerprint"), label=f"{router_label} router fingerprint"
        ),
        candidate_is_learned=True,
        full_task_spec_accuracy=_require_probability(
            summary.get("valid_full_task_accuracy"), label=f"{router_label} full TaskSpec accuracy"
        ),
        object_accuracy=_require_probability(
            summary.get("object_accuracy"), label=f"{router_label} object accuracy"
        ),
        bin_accuracy=_require_probability(
            summary.get("bin_accuracy"), label=f"{router_label} bin accuracy"
        ),
        false_route_rate=_require_probability(
            summary.get("false_route_rate"), label=f"{router_label} false-route rate"
        ),
        ambiguous_rejection_recall=_require_probability(
            summary.get("ambiguous_rejection_recall"),
            label=f"{router_label} ambiguous rejection recall",
        ),
        unsupported_rejection_recall=_require_probability(
            summary.get("unsupported_rejection_recall"),
            label=f"{router_label} unsupported rejection recall",
        ),
        malformed_rejection_recall=_require_probability(
            summary.get("malformed_rejection_recall"),
            label=f"{router_label} malformed rejection recall",
        ),
        schema_valid_output_rate=_require_probability(
            supplemental.get("schema_valid_output_rate"),
            label=f"{router_label} schema-valid output rate",
        ),
        deterministic_repeatability=_require_probability(
            summary.get("deterministic_repeatability"),
            label=f"{router_label} deterministic repeatability",
        ),
        llm_malformed_output_rate=(
            None
            if router_label == "classifier"
            else _require_probability(
                supplemental.get("structured_output_malformed_rate"),
                label="LLM structured-output malformed rate",
            )
        ),
        oracle_success_rate=_require_nonnegative_int(
            oracle.get("end_to_end_success_count"), label="oracle success count"
        )
        / oracle_denominator,
        predicted_success_rate=_require_nonnegative_int(
            candidate.get("end_to_end_success_count"),
            label=f"{router_label} success count",
        )
        / candidate_denominator,
        wrong_object_routing_count=_require_nonnegative_int(
            candidate.get("routing_wrong_object_count"),
            label=f"{router_label} wrong-object routes",
        ),
        wrong_bin_routing_count=_require_nonnegative_int(
            candidate.get("routing_wrong_bin_count"), label=f"{router_label} wrong-bin routes"
        ),
        false_rejection_count=_require_nonnegative_int(
            candidate.get("routing_false_rejection_count"),
            label=f"{router_label} false rejections",
        ),
        target_in_wrong_bin_count=_require_nonnegative_int(
            candidate.get("target_in_wrong_bin_count"),
            label=f"{router_label} target-in-wrong-bin count",
        ),
        target_off_table_count=_require_nonnegative_int(
            candidate.get("target_off_table_count"),
            label=f"{router_label} target-off-table count",
        ),
        arm_projection_count=_require_nonnegative_int(
            candidate.get("arm_projected_component_count"),
            label=f"{router_label} arm projections",
        ),
        invalid_action_count=invalid + malformed + nonfinite,
        dispatch_after_rejection_count=_require_nonnegative_int(
            candidate.get("dispatch_after_rejection_count"),
            label=f"{router_label} dispatch-after-rejection count",
        ),
        m2_expert_call_count=_require_nonnegative_int(
            candidate.get("m2_expert_call_count"), label=f"{router_label} M2 expert calls"
        ),
    )


def _default_command_runner(command: Sequence[str]) -> int:
    completed = subprocess.run(tuple(command), cwd=PROJECT_ROOT, check=False)
    return completed.returncode


def _run_or_reuse_stage(
    *,
    name: str,
    report_path: Path,
    command: Sequence[str],
    validator: Callable[[Mapping[str, object]], None],
    runner: CommandRunner,
) -> tuple[dict[str, object], bool]:
    if report_path.exists():
        existing = _read_json_object(report_path, label=f"{name} stage report")
        if existing.get("passed") is True:
            validator(existing)
            return existing, True
    return_code = runner(tuple(command))
    if return_code != 0:
        detail = "stage did not produce a report"
        if report_path.exists():
            failed = _read_json_object(report_path, label=f"failed {name} stage report")
            detail = f"{failed.get('error_type')}: {failed.get('error_message')}"
        raise M5ATargetVerificationError(f"{name} stage exited with code {return_code}: {detail}")
    payload = _read_json_object(report_path, label=f"{name} stage report")
    validator(payload)
    return payload, False


def _require_target_configuration(args: argparse.Namespace) -> None:
    if args.m43_independent_verification is None:
        raise M5ATargetVerificationError(
            "--m43-independent-verification is required for --target-development"
        )
    for name in (
        "llm_model_id",
        "llm_model_revision",
        "llm_tokenizer_revision",
        "llm_license",
    ):
        value = getattr(args, name)
        if not isinstance(value, str) or not value.strip():
            raise M5ATargetVerificationError(
                f"--{name.replace('_', '-')} is required for --target-development"
            )
    for name in ("llm_model_revision", "llm_tokenizer_revision"):
        revision = getattr(args, name)
        if len(revision) != 40 or any(
            character not in "0123456789abcdef" for character in revision
        ):
            raise M5ATargetVerificationError(
                f"--{name.replace('_', '-')} must be an exact lowercase 40-character revision"
            )
    if args.llm_license_reviewed is not True or args.llm_model_card_reviewed is not True:
        raise M5ATargetVerificationError(
            "target-development requires explicit LLM license and official model-card review"
        )
    if (
        isinstance(args.llm_maximum_new_tokens, bool)
        or not isinstance(args.llm_maximum_new_tokens, int)
        or not 1 <= args.llm_maximum_new_tokens <= 512
    ):
        raise M5ATargetVerificationError("--llm-maximum-new-tokens must lie in [1, 512]")
    if (
        isinstance(args.llm_repeat_count, bool)
        or not isinstance(args.llm_repeat_count, int)
        or args.llm_repeat_count < 2
    ):
        raise M5ATargetVerificationError("--llm-repeat-count must be an integer of at least two")


def _stage_report_paths(output_root: Path) -> dict[str, Path]:
    stage_root = output_root / "stages"
    return {name: stage_root / f"{name}.json" for name in TARGET_STAGE_NAMES}


def _command(*values: object) -> tuple[str, ...]:
    return tuple(os.fspath(value) if isinstance(value, Path) else str(value) for value in values)


def _rebuild_controller_registry(args: argparse.Namespace) -> ControllerRegistry:
    """Rebuild six validation-selected controllers without test/fresh content."""

    registry, _locators = load_controller_registry_metadata(
        checkpoint_root=args.checkpoint_root,
        dataset_root=args.dataset_root,
        runtime_selection_path=args.runtime_selection,
    )
    return registry


def _clean_git_commit(value: object, *, label: str) -> str:
    state = _require_mapping(value, label=f"{label} Git state")
    _require_flag(state, "baseline_tracked", True)
    _require_flag(state, "dirty", False)
    commit = state.get("commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise M5ATargetVerificationError(f"{label} Git commit must be a full lowercase commit")
    return commit


def _dependency_version_strings(value: object) -> dict[str, str]:
    dependencies = _require_mapping(value, label="M5A dependency versions")
    result: dict[str, str] = {}
    for name, version in dependencies.items():
        if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
            raise M5ATargetVerificationError("target dependency versions must be non-empty strings")
        result[name] = version
    return result


def _lock_experiment_manifest(
    *,
    output_root: Path,
    corpus_fingerprint: str,
    schedule_fingerprints: Mapping[str, str],
    registry: ControllerRegistry,
    language_report: Mapping[str, object],
    classifier_report: Mapping[str, object],
    control_identity: Mapping[str, object],
) -> tuple[Path, M5AExperimentManifest]:
    """Create the immutable cross-stage M5A experiment/runtime identity."""

    language_commit = _clean_git_commit(
        language_report.get("git_state"), label="language evaluation"
    )
    classifier_evidence = _require_mapping(
        classifier_report.get("run_evidence"), label="classifier run evidence"
    )
    classifier_commit = _clean_git_commit(classifier_evidence.get("git_state"), label="classifier")
    _require_exact(
        classifier_evidence.get("git_commit"),
        classifier_commit,
        label="classifier Git commit projection",
    )
    _require_exact(classifier_commit, language_commit, label="classifier/language Git commit")
    _require_exact(
        control_identity.get("git_commit"),
        language_commit,
        label="control/language Git commit",
    )
    language_dependencies = _dependency_version_strings(language_report.get("dependencies"))
    classifier_dependencies = _dependency_version_strings(classifier_evidence.get("dependencies"))
    _require_exact(
        classifier_dependencies,
        language_dependencies,
        label="classifier/language dependency versions",
    )
    identities = _require_mapping(
        language_report.get("router_identities"), label="language router identities"
    )
    expected = {
        "rule": "RuleRouterV0",
        "classifier": "FactorizedTextClassifierV0",
        "llm": "StructuredLocalLLMRouterV0",
    }
    action_runtime_fingerprints = {entry.action_runtime_fingerprint for entry in registry.entries}
    if len(action_runtime_fingerprints) != 1:
        raise M5ATargetVerificationError("controller action-runtime fingerprints differ")
    environment_ids = {entry.environment_id for entry in registry.entries}
    if len(environment_ids) != 1:
        raise M5ATargetVerificationError("controller environment IDs differ")
    runtimes: dict[str, RouterRuntimeIdentity] = {}
    for key, router_name in expected.items():
        identity = _require_mapping(identities.get(key), label=f"{key} router identity")
        _require_exact(identity.get("router_name"), router_name, label=f"{key} router name")
        runtimes[router_name] = RouterRuntimeIdentity(
            router_fingerprint=_require_sha256(
                identity.get("router_fingerprint"), label=f"{key} router fingerprint"
            ),
            controller_registry_fingerprint=registry.registry_fingerprint,
            environment_id=next(iter(environment_ids)),
            environment_contract_fingerprint=registry.environment_contract_fingerprint,
            rollout_config_fingerprint=registry.rollout_config_fingerprint,
            action_runtime_fingerprint=next(iter(action_runtime_fingerprints)),
            git_commit=language_commit,
        )
    manifest = M5AExperimentManifest(
        corpus_fingerprint=corpus_fingerprint,
        schedule_fingerprints=dict(schedule_fingerprints),
        controller_registry_fingerprint=registry.registry_fingerprint,
        router_runtime_identities=runtimes,
        git_commit=language_commit,
        dependency_versions=language_dependencies,
    )
    path = output_root / "experiment-manifest.json"
    payload = manifest.to_dict()
    if path.exists():
        if _read_json_object(path, label="M5A experiment manifest") != payload:
            raise M5ATargetVerificationError(
                "existing M5A experiment manifest differs from immutable target inputs"
            )
    else:
        try:
            atomic_write_json(path, payload, immutable=True)
        except FileExistsError as error:
            if _read_json_object(path, label="M5A experiment manifest") != payload:
                raise M5ATargetVerificationError(
                    "concurrently created M5A experiment manifest differs"
                ) from error
    return path, manifest


def _execute_target_development(
    args: argparse.Namespace,
    *,
    report: TargetDevelopmentReport,
    runner: CommandRunner,
) -> None:
    _require_target_configuration(args)
    paths = _validate_target_paths(args)
    report.llm_model_id = args.llm_model_id
    report.llm_model_revision = args.llm_model_revision
    report.llm_tokenizer_revision = args.llm_tokenizer_revision
    report.llm_license = args.llm_license
    report.llm_dtype = args.llm_dtype
    report.llm_maximum_new_tokens = args.llm_maximum_new_tokens
    report.llm_repeat_count = args.llm_repeat_count
    report.llm_license_reviewed = bool(args.llm_license_reviewed)
    report.llm_model_card_reviewed = bool(args.llm_model_card_reviewed)
    report.llm_local_files_only = bool(args.local_files_only)
    output_root = paths.verifier_output_root
    stage_reports = _stage_report_paths(output_root)
    report.stage_reports = {name: str(path) for name, path in stage_reports.items()}

    m43_path = paths.m43_independent_verification
    m43b = _validate_m43b_verification(m43_path)
    report.prior_m43b_target_validated = True
    report.m43b_evaluation_evidence_fingerprint = cast(str, m43b["evaluation_evidence_fingerprint"])
    report.check(
        "M4.3b independent target prerequisite",
        True,
        f"evaluation={report.m43b_evaluation_evidence_fingerprint}",
    )

    # The controller library is the second target-development prerequisite.
    # It is rebuilt before any language schedule or corpus is materialized and
    # reads validation-only M4 metadata, never comparison/test/fresh results.
    authoritative_registry = _rebuild_controller_registry(args)
    report.controller_registry_validated = True
    report.controller_registry_fingerprint = authoritative_registry.registry_fingerprint
    report.controller_checkpoint_fingerprints = [
        entry.checkpoint_fingerprint for entry in authoritative_registry.entries
    ]
    report.check(
        "frozen validation-selected PerTask controller registry",
        True,
        f"registry={authoritative_registry.registry_fingerprint}",
    )

    schedule_payload, bundle, corpus = _build_target_schedule_payload(m43b_verification=m43b)
    schedule_lock_path = output_root / "target-development-schedules.json"
    schedule_reused = _lock_target_schedules(schedule_lock_path, schedule_payload)
    report.schedule_lock_path = str(schedule_lock_path)
    report.schedule_fingerprints = {
        bundle.language_development.schedule_id: bundle.language_development.schedule_fingerprint,
        bundle.language_final.schedule_id: bundle.language_final.schedule_fingerprint,
        bundle.control_development.schedule_id: bundle.control_development.schedule_fingerprint,
        bundle.control_final.schedule_id: bundle.control_final.schedule_fingerprint,
    }
    config_payload = _require_mapping(
        schedule_payload["control_schedule_config"], label="control schedule config"
    )
    exclusion_values = config_payload.get("exclusion_sources")
    if not isinstance(exclusion_values, list):
        raise M5ATargetVerificationError("control schedule exclusions must be a list")
    report.exclusion_sources = {
        cast(str, source["source_id"]): cast(str, source["source_fingerprint"])
        for source in exclusion_values
        if isinstance(source, Mapping)
    }
    if set(report.exclusion_sources) != set(REQUIRED_EXCLUSION_SOURCE_IDS):
        raise M5ATargetVerificationError("target schedule lock lacks an authoritative exclusion")
    report.final_schedules_locked = True
    report.stage_reused["schedules"] = schedule_reused
    report.check(
        "all four schedules locked before training",
        True,
        f"lock={schedule_lock_path}; reused={schedule_reused}",
    )

    corpus_root = paths.corpus_root
    classifier_output_root = paths.classifier_output_root
    language_evidence_root = paths.language_evidence_root
    control_evidence_root = paths.control_evidence_root
    corpus_command = _command(
        sys.executable,
        PROJECT_ROOT / "scripts" / "build_language_corpus.py",
        "--output-root",
        corpus_root,
        "--report",
        stage_reports["corpus"],
    )

    def validate_corpus(payload: Mapping[str, object]) -> None:
        _validate_corpus_report(
            payload,
            corpus_root=corpus_root,
            corpus_fingerprint=corpus.manifest.corpus_fingerprint,
            language_development_fingerprint=bundle.language_development.schedule_fingerprint,
            language_final_fingerprint=bundle.language_final.schedule_fingerprint,
        )
        archive = validate_language_corpus_archive(
            corpus_root,
            expected_corpus_fingerprint=corpus.manifest.corpus_fingerprint,
        )
        _require_exact(
            payload.get("archive_fingerprint"),
            archive.archive_fingerprint,
            label="corpus immutable archive fingerprint",
        )

    corpus_report, reused = _run_or_reuse_stage(
        name="corpus",
        report_path=stage_reports["corpus"],
        command=corpus_command,
        validator=validate_corpus,
        runner=runner,
    )
    del corpus_report
    report.stage_reused["corpus"] = reused
    report.corpus_fingerprint = corpus.manifest.corpus_fingerprint
    report.split_isolation_fingerprint = corpus.isolation_report.report_fingerprint
    report.corpus_validated = True
    report.split_isolation_validated = True
    report.check("authoritative language corpus", True, f"reused={reused}")

    classifier_command_values: list[object] = [
        sys.executable,
        PROJECT_ROOT / "scripts" / "train_text_router.py",
        "--target-development",
        "--output-root",
        classifier_output_root,
        "--report",
        stage_reports["classifier"],
    ]
    if args.clean_staging:
        classifier_command_values.append("--clean-staging")
    classifier_command = _command(*classifier_command_values)

    def validate_classifier(payload: Mapping[str, object]) -> None:
        candidate_root = _validate_classifier_report(
            payload,
            classifier_output_root=classifier_output_root,
            corpus_fingerprint=corpus.manifest.corpus_fingerprint,
        )
        run_evidence = _require_mapping(
            payload.get("run_evidence"), label="classifier run evidence"
        )
        validate_completed_text_classifier_artifact(
            candidate_root,
            expected_run_evidence=run_evidence,
        )

    classifier_report, reused = _run_or_reuse_stage(
        name="classifier",
        report_path=stage_reports["classifier"],
        command=classifier_command,
        validator=validate_classifier,
        runner=runner,
    )
    report.stage_reused["classifier"] = reused
    artifact_root = _validate_classifier_report(
        classifier_report,
        classifier_output_root=classifier_output_root,
        corpus_fingerprint=corpus.manifest.corpus_fingerprint,
    )
    artifact_fingerprint = _require_sha256(
        classifier_report.get("artifact_fingerprint"), label="classifier artifact fingerprint"
    )
    report.classifier_run_fingerprint = _require_sha256(
        classifier_report.get("run_fingerprint"), label="classifier run fingerprint"
    )
    report.classifier_artifact_fingerprint = artifact_fingerprint
    report.classifier_training_completed = True
    report.classifier_checkpoint_selected = True
    report.classifier_calibration_validated = True
    report.check("classifier train/selection/calibration", True, f"reused={reused}")

    language_command_values: list[object] = [
        sys.executable,
        PROJECT_ROOT / "scripts" / "evaluate_language_routers.py",
        "--target-development",
        "--corpus-root",
        corpus_root,
        "--classifier-artifact",
        artifact_root,
        "--llm-model-id",
        args.llm_model_id,
        "--llm-model-revision",
        args.llm_model_revision,
        "--llm-tokenizer-revision",
        args.llm_tokenizer_revision,
        "--llm-license",
        args.llm_license,
        "--llm-license-reviewed",
        "--llm-dtype",
        args.llm_dtype,
        "--llm-quantization",
        "none",
        "--llm-maximum-new-tokens",
        args.llm_maximum_new_tokens,
        "--repeat-count",
        args.llm_repeat_count,
        "--device",
        args.device,
        "--output-root",
        language_evidence_root,
        "--report",
        stage_reports["language"],
    ]
    if args.local_files_only:
        language_command_values.append("--local-files-only")
    language_command = _command(*language_command_values)

    validated_language_evidence: ValidatedRouterEvaluationEvidence | None = None

    def validate_language(payload: Mapping[str, object]) -> None:
        nonlocal validated_language_evidence
        _validate_language_report(
            payload,
            corpus_fingerprint=corpus.manifest.corpus_fingerprint,
            artifact_fingerprint=artifact_fingerprint,
            language_development_fingerprint=bundle.language_development.schedule_fingerprint,
            language_final_fingerprint=bundle.language_final.schedule_fingerprint,
            llm_model_id=args.llm_model_id,
            llm_model_revision=args.llm_model_revision,
            llm_tokenizer_revision=args.llm_tokenizer_revision,
            llm_license=args.llm_license,
            llm_dtype=args.llm_dtype,
            llm_maximum_new_tokens=args.llm_maximum_new_tokens,
            repeat_count=args.llm_repeat_count,
            local_files_only=bool(args.local_files_only),
        )
        evidence_value = payload.get("evidence_root")
        if not isinstance(evidence_value, str):
            raise M5ATargetVerificationError("language report lacks its immutable evidence root")
        evidence_root = _resolved_unlinked(
            Path(evidence_value), label="frozen router evaluation evidence"
        )
        if evidence_root.parent != language_evidence_root:
            raise M5ATargetVerificationError(
                "router evaluation evidence is outside the configured evidence root"
            )
        expected_evaluation = _require_sha256(
            payload.get("evaluation_fingerprint"),
            label="language evaluation fingerprint",
        )
        validated = validate_router_evaluation_evidence(
            evidence_root,
            expected_examples_by_split={
                split: corpus.examples_for_split(split)
                for split in (LanguageSplit.VALIDATION, LanguageSplit.DEVELOPMENT)
            },
            expected_evaluation_fingerprint=expected_evaluation,
        )
        _require_exact(
            payload.get("artifact_fingerprint"),
            validated.artifact_fingerprint,
            label="language immutable artifact fingerprint",
        )
        for key, value in validated.result.items():
            if key != "schema_version":
                _require_exact(payload.get(key), value, label=f"language immutable result {key}")
        immutable_command = {
            **dict(validated.result),
            "schema_version": LANGUAGE_COMMAND_SCHEMA,
            "artifact_fingerprint": validated.artifact_fingerprint,
            "evidence_root": str(validated.root),
        }
        _validate_language_report(
            immutable_command,
            corpus_fingerprint=corpus.manifest.corpus_fingerprint,
            artifact_fingerprint=artifact_fingerprint,
            language_development_fingerprint=bundle.language_development.schedule_fingerprint,
            language_final_fingerprint=bundle.language_final.schedule_fingerprint,
            llm_model_id=args.llm_model_id,
            llm_model_revision=args.llm_model_revision,
            llm_tokenizer_revision=args.llm_tokenizer_revision,
            llm_license=args.llm_license,
            llm_dtype=args.llm_dtype,
            llm_maximum_new_tokens=args.llm_maximum_new_tokens,
            repeat_count=args.llm_repeat_count,
            local_files_only=bool(args.local_files_only),
        )
        validated_language_evidence = validated

    language_report, reused = _run_or_reuse_stage(
        name="language",
        report_path=stage_reports["language"],
        command=language_command,
        validator=validate_language,
        runner=runner,
    )
    report.stage_reused["language"] = reused
    if validated_language_evidence is None:
        raise M5ATargetVerificationError("language immutable evidence was not validated")
    frozen_router_evidence = validated_language_evidence.root
    language_authority = validated_language_evidence.result
    language_evaluation_fingerprint = validated_language_evidence.evaluation_fingerprint
    language_artifact_fingerprint = validated_language_evidence.artifact_fingerprint
    report.language_evaluation_fingerprint = language_evaluation_fingerprint
    report.language_artifact_fingerprint = language_artifact_fingerprint
    language_identities = _require_mapping(
        language_authority.get("router_identities"), label="language router identities"
    )
    language_llm_identity = _require_mapping(
        language_identities.get("llm"), label="language LLM identity"
    )
    language_prompt_fingerprint = _require_sha256(
        language_llm_identity.get("prompt_fingerprint"),
        label="language LLM prompt fingerprint",
    )
    report.llm_router_loaded = True
    report.llm_prompt_locked = True
    report.language_validation_completed = True
    report.language_development_completed = True
    report.rule_router_validated = True
    report.check("three-router language validation/development", True, f"reused={reused}")

    control_command_values: list[object] = [
        sys.executable,
        PROJECT_ROOT / "scripts" / "run_language_control.py",
        "--control-schedule",
        schedule_lock_path,
        "--classifier-artifact-root",
        artifact_root,
        "--router-evaluation-evidence-root",
        frozen_router_evidence,
        "--llm-model-id",
        args.llm_model_id,
        "--llm-model-revision",
        args.llm_model_revision,
        "--llm-tokenizer-revision",
        args.llm_tokenizer_revision,
        "--llm-license",
        args.llm_license,
        "--llm-license-reviewed",
        "--llm-dtype",
        args.llm_dtype,
        "--llm-maximum-new-tokens",
        args.llm_maximum_new_tokens,
        "--device",
        args.device,
        "--dataset-root",
        paths.dataset_root,
        "--checkpoint-root",
        paths.checkpoint_root,
        "--runtime-selection",
        paths.runtime_selection,
        "--output-root",
        control_evidence_root,
        "--report",
        stage_reports["control"],
    ]
    if not args.local_files_only:
        control_command_values.append("--allow-model-download")
    control_command = _command(*control_command_values)

    development_examples = {
        example.example_id: example
        for example in corpus.examples_for_split(LanguageSplit.DEVELOPMENT)
    }
    development_inputs = cast(
        DevelopmentControlInputsLike,
        SimpleNamespace(
            schedule=bundle.control_development,
            episodes=bundle.development_episodes,
            examples_by_id=development_examples,
            corpus_fingerprint=corpus.manifest.corpus_fingerprint,
        ),
    )
    validated_control_evidence: ValidatedControlEvidence | None = None

    def validate_control(payload: Mapping[str, object]) -> None:
        nonlocal validated_control_evidence
        reported_registry = _validate_control_report(
            payload,
            corpus_fingerprint=corpus.manifest.corpus_fingerprint,
            control_schedule_fingerprint=bundle.control_development.schedule_fingerprint,
            classifier_artifact_fingerprint=artifact_fingerprint,
            llm_model_id=args.llm_model_id,
            llm_model_revision=args.llm_model_revision,
            llm_tokenizer_revision=args.llm_tokenizer_revision,
            llm_dtype=args.llm_dtype,
            llm_maximum_new_tokens=args.llm_maximum_new_tokens,
            language_evaluation_fingerprint=language_evaluation_fingerprint,
            language_artifact_fingerprint=language_artifact_fingerprint,
            language_prompt_fingerprint=language_prompt_fingerprint,
            router_evaluation_evidence_root=frozen_router_evidence,
        )
        _require_exact(
            reported_registry.to_dict(),
            authoritative_registry.to_dict(),
            label="reported/independently rebuilt controller registry",
        )
        evidence_value = payload.get("evidence_root")
        if not isinstance(evidence_value, str):
            raise M5ATargetVerificationError("control report lacks its immutable evidence root")
        evidence_root = _resolved_unlinked(
            Path(evidence_value), label="development control evidence"
        )
        expected_parent = _resolved_unlinked(
            control_evidence_root / "evidence", label="configured control evidence root"
        )
        if evidence_root.parent != expected_parent:
            raise M5ATargetVerificationError(
                "control evidence is outside the configured evidence root"
            )
        validated = validate_development_control_evidence(
            evidence_root,
            inputs=development_inputs,
            registry=authoritative_registry,
        )
        for key, expected in (
            ("run_fingerprint", validated.run_fingerprint),
            ("schedule_fingerprint", validated.schedule_fingerprint),
            ("corpus_fingerprint", validated.corpus_fingerprint),
            (
                "controller_registry_fingerprint",
                validated.controller_registry_fingerprint,
            ),
            ("record_set_fingerprint", validated.record_set_fingerprint),
            ("completion_fingerprint", validated.completion_fingerprint),
        ):
            _require_exact(payload.get(key), expected, label=f"control immutable {key}")
        _require_exact(
            payload.get("summaries"),
            dict(validated.summaries),
            label="control immutable summaries",
        )
        _require_exact(
            validated.identity.get("router_evaluation"),
            payload.get("router_evaluation_identity"),
            label="control owner router-evaluation identity",
        )
        _require_exact(
            validated.identity.get("active_episode_spec_required"),
            True,
            label="control active EpisodeSpec requirement",
        )
        validated_control_evidence = validated

    control_report, reused = _run_or_reuse_stage(
        name="control",
        report_path=stage_reports["control"],
        command=control_command,
        validator=validate_control,
        runner=runner,
    )
    report.stage_reused["control"] = reused
    if validated_control_evidence is None:
        raise M5ATargetVerificationError("control immutable evidence was not validated")
    registry = authoritative_registry
    report.controller_registry_fingerprint = registry.registry_fingerprint
    report.controller_registry_validated = True
    report.oracle_control_development_completed = True
    report.predicted_control_development_completed = True
    report.control_development_completed = True
    report.rejection_noop_probe_validated = True
    report.failure_attribution_validated = True
    report.physical_target_validated = True
    report.check("oracle plus three-router physical control development", True, f"reused={reused}")

    manifest_path, manifest = _lock_experiment_manifest(
        output_root=output_root,
        corpus_fingerprint=corpus.manifest.corpus_fingerprint,
        schedule_fingerprints=report.schedule_fingerprints,
        registry=registry,
        language_report=language_authority,
        classifier_report=classifier_report,
        control_identity=validated_control_evidence.identity,
    )
    report.experiment_manifest_path = str(manifest_path)
    report.experiment_fingerprint = manifest.experiment_fingerprint
    report.router_runtime_fingerprints = {
        name: identity.runtime_fingerprint
        for name, identity in manifest.router_runtime_identities.items()
    }
    report.experiment_manifest_validated = True
    report.check(
        "immutable M5A experiment/runtime manifest",
        True,
        f"fingerprint={manifest.experiment_fingerprint}",
    )

    gates = {
        router_label: _build_development_gate(
            router_label=router_label,
            language_report=language_authority,
            control_report=control_report,
        )
        for router_label in ("classifier", "llm")
    }
    report.development_gates = {name: gate.to_dict() for name, gate in gates.items()}
    report.development_quality_gate_passed = any(
        gate.development_quality_gate_passed for gate in gates.values()
    )
    report.final_benchmark_authorized = report.development_quality_gate_passed
    report.final_benchmark_completed = False
    report.check(
        "development quality gates calculated",
        True,
        "classifier/LLM authorization=" + str(report.development_quality_gate_passed).lower(),
    )

    report.implementation_validated = (
        report.prior_m43b_target_validated
        and report.corpus_validated
        and report.split_isolation_validated
        and report.rule_router_validated
        and report.controller_registry_validated
        and report.experiment_manifest_validated
        and report.final_schedules_locked
        and report.classifier_training_completed
        and report.classifier_checkpoint_selected
        and report.classifier_calibration_validated
        and report.llm_router_loaded
        and report.llm_prompt_locked
        and report.language_validation_completed
        and report.language_development_completed
        and report.control_development_completed
        and report.rejection_noop_probe_validated
        and report.failure_attribution_validated
        and report.physical_target_validated
        and not report.failed
    )


def _write_report(output_root: Path, report: Report | TargetDevelopmentReport) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "verification.json", report.payload())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-development", action="store_true")
    parser.add_argument("--m43-independent-verification", type=Path)
    parser.add_argument("--llm-model-id")
    parser.add_argument("--llm-model-revision")
    parser.add_argument("--llm-tokenizer-revision")
    parser.add_argument("--llm-license")
    parser.add_argument("--llm-license-reviewed", action="store_true")
    parser.add_argument("--llm-model-card-reviewed", action="store_true")
    parser.add_argument(
        "--llm-dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--llm-maximum-new-tokens", type=int, default=128)
    parser.add_argument("--llm-repeat-count", type=int, default=2)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--clean-staging", action="store_true")
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument(
        "--classifier-output-root", type=Path, default=DEFAULT_CLASSIFIER_OUTPUT_ROOT
    )
    parser.add_argument(
        "--language-evidence-root", type=Path, default=DEFAULT_LANGUAGE_EVIDENCE_ROOT
    )
    parser.add_argument("--control-evidence-root", type=Path, default=DEFAULT_CONTROL_EVIDENCE_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_M3B_DATASET_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_ACT_CHECKPOINT_ROOT)
    parser.add_argument("--m4-diagnostics-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--runtime-selection", type=Path, default=DEFAULT_RUNTIME_SELECTION)
    return parser.parse_args(argv)


def _execute_non_target(report: Report) -> None:
    statuses = {status.value for status in RouterStatus}
    typed_contracts_valid = statuses == {
        "route",
        "reject_ambiguous",
        "reject_unsupported",
        "reject_malformed",
    }
    report.check(
        "imports and typed contracts",
        typed_contracts_valid,
        "RouterStatus has exactly four declared values",
    )
    _verify_corpus_and_schedules(report)
    rule_router = _verify_rule_router(report)
    _verify_classifier_fixture(report)
    _verify_llm_fixture(report)
    _verify_registry_dispatch_and_attribution(report, rule_router)
    report.implementation_validated = (
        typed_contracts_valid
        and report.corpus_validated
        and report.split_isolation_validated
        and report.rule_router_validated
        and report.classifier_fixture_validated
        and report.llm_router_fixture_validated
        and report.controller_registry_validated
        and report.final_schedules_locked
        and not report.failed
    )


def main(
    argv: list[str] | None = None,
    *,
    command_runner: CommandRunner | None = None,
) -> int:
    args = parse_args(argv)
    output_root = _validate_output_root(args.output_root)
    if args.target_development:
        target_report = TargetDevelopmentReport()
        try:
            _execute_target_development(
                args,
                report=target_report,
                runner=_default_command_runner if command_runner is None else command_runner,
            )
        except Exception as error:  # noqa: BLE001 - command boundary preserves diagnostics
            traceback.print_exc()
            target_report.check(
                "target-development verifier exception",
                False,
                f"{type(error).__name__}: {error}",
            )
        _write_report(output_root, target_report)
        print(json.dumps(target_report.payload(), sort_keys=True, allow_nan=False))
        return 0 if target_report.passed else 1

    report = Report()
    try:
        _execute_non_target(report)
    except Exception as error:  # noqa: BLE001 - command boundary preserves diagnostics
        traceback.print_exc()
        report.check(
            "unexpected verifier exception",
            False,
            f"{type(error).__name__}: {error}",
        )
    _write_report(output_root, report)
    print(json.dumps(report.payload(), sort_keys=True, allow_nan=False))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
