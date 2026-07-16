"""Execute the authorized M4.3b FactorFiLM target-development experiment.

The immutable training run is an input.  All validation, selection, reload,
development and semantic evidence is written below a separate diagnostics
root.  This module intentionally has no test, historical-fresh or final
schedule entry point.
"""

from __future__ import annotations

import gc
import json
import math
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from langmani.environments.pick_place_by_instruction import ENV_ID
from langmani.policies.act_action_bounds import (
    ActionBoundConfig,
    ActionBoundMode,
    BoundedActionEnvPostprocessorV0,
)
from langmani.policies.act_checkpoint import validate_act_checkpoint_artifacts
from langmani.policies.act_data import load_completed_m3b_dataset
from langmani.policies.act_factor_film_evaluation import (
    DevelopmentEpisodeIdentity,
    DevelopmentPolicyMetrics,
    DevelopmentSemanticMetrics,
    FactorFiLMReloadValidationRecord,
    FactorFiLMValidationCountResult,
    FirstInteractionSnapshot,
    SemanticErrorStage,
    derive_first_interaction_snapshot,
    evaluate_development_quality_gate,
    factor_film_first_interaction_confusions,
    factor_film_validation_count_result,
    rank_factor_film_validation_count_results,
    validate_paired_development_identities,
    validate_reload_against_selection,
)
from langmani.policies.act_factor_film_evidence import (
    FACTOR_FILM_VALIDATION_EVIDENCE_SCHEMA_VERSION,
    FactorFiLMEvaluationEvidenceConfig,
    FactorFiLMEvaluationStage,
    FactorFiLMEvaluationStageArtifact,
    stage_and_promote_factor_film_evaluation_evidence,
)
from langmani.policies.act_factor_film_training import (
    AUTHORIZED_M3B_FINGERPRINT,
    AUTHORIZED_M3B_SPLIT_FINGERPRINT,
    AUTHORIZED_M43_AUDIT_EVIDENCE_FINGERPRINT,
    factor_film_validation_schedule_digest,
    factor_film_validation_schedule_records,
    validate_authorizing_semantic_audit,
)
from langmani.policies.act_factor_film_types import (
    FactorFiLMSelectionRecord,
    FactorFiLMTrainingManifest,
    FactorFiLMTrainingMode,
    FactorFiLMValidationQueue,
    canonical_fingerprint,
)
from langmani.policies.act_runtime import atomic_write_json, inspect_git_state
from langmani.policies.m42_evaluation import (
    M42BenchmarkReport,
    M42CheckpointContext,
    load_factor_film_training_checkpoint_context,
    run_m42_checkpoint_benchmark,
    run_m42_validation_benchmark,
)
from langmani.policies.m42_schedule import (
    M42_DEV_SCHEDULE_FINGERPRINT,
    M42_DEV_SCHEDULE_ID,
    M42_MAXIMUM_EPISODE_STEPS,
    load_locked_schedule,
    materialize_schedule,
)
from langmani.policies.m42_types import GripperRuntimeMode
from langmani.policies.m43_audit_execution import (
    create_m43_audit_environment,
    load_m43_development_observations,
    load_m43_reference_policy_contexts,
    load_m43_validation_observations,
    m43_action_chunk_distance_config,
)
from langmani.policies.m43_audit_inputs import load_restricted_audit_inputs
from langmani.policies.m43_audit_runtime import (
    FixedPolicyObservation,
    compute_named_fixed_observation_audits,
    compute_per_task_reference_sensitivity,
    predict_postprocessed_action_chunk,
)
from langmani.policies.m43_types import M43_CANONICAL_TASK_IDS

FACTOR_FILM_EXECUTION_SCHEMA_VERSION = "langmani-m43b-factor-film-execution-v0"
FACTOR_FILM_WORK_SCHEMA_VERSION = "langmani-m43b-factor-film-work-v0"
FACTOR_FILM_VALIDATION_REPORT_SCHEMA_VERSION = "langmani-m43b-factor-film-validation-report-v0"
FACTOR_FILM_DEVELOPMENT_REPORT_SCHEMA_VERSION = "langmani-m43b-factor-film-development-report-v0"
FACTOR_FILM_REFERENCE_OUTPUT_SCHEMA_VERSION = "langmani-m43b-factor-film-reload-reference-v0"
FACTOR_FILM_TRAINING_COMPLETION_SCHEMA_VERSION = "langmani-m43-factor-film-training-complete-v0"
_POLICY_LABEL_PER_TASK = "ACT-PerTask"
_POLICY_LABEL_ONEHOT = "ACT-Mixed-TaskOneHot"
_POLICY_LABEL_FACTOR = "ACT-Mixed-FactorFiLM"
_EVIDENCE_POLICY_LABEL_FACTOR = "factor_film"
_EVIDENCE_POLICY_LABEL_ONEHOT = "state_onehot"


class FactorFiLMExecutionError(RuntimeError):
    """Raised when target-development cannot preserve its declared contract."""


@dataclass(frozen=True, slots=True)
class FactorFiLMExecutionPaths:
    """Explicit inputs and separate generated-output roots for one evaluation."""

    dataset_root: Path
    factor_film_model_root: Path
    semantic_audit_evidence_root: Path
    m4_checkpoint_root: Path
    task_token_checkpoint_root: Path
    m42_diagnostics_root: Path
    runtime_selection_path: Path
    output_root: Path
    run_root: Path | None = None


@dataclass(frozen=True, slots=True)
class FactorFiLMExecutionConfig:
    paths: FactorFiLMExecutionPaths
    evaluation_git_commit: str
    expected_training_git_commit: str
    device: str = "cuda"
    clean_matching_staging: bool = False

    def __post_init__(self) -> None:
        if self.device != "cuda":
            raise FactorFiLMExecutionError("FactorFiLM target-development evaluation requires CUDA")
        if re.fullmatch(r"[0-9a-f]{40}", self.evaluation_git_commit) is None:
            raise FactorFiLMExecutionError("evaluation_git_commit must be full lowercase Git hex")
        if re.fullmatch(r"[0-9a-f]{40}", self.expected_training_git_commit) is None:
            raise FactorFiLMExecutionError(
                "expected_training_git_commit must be full lowercase Git hex"
            )


@dataclass(frozen=True, slots=True)
class _TrainingAuthority:
    run_root: Path
    manifest: FactorFiLMTrainingManifest
    completion: Mapping[str, object]
    queue: FactorFiLMValidationQueue
    evidence_config: FactorFiLMEvaluationEvidenceConfig
    validation_schedule_records: tuple[Mapping[str, object], ...]
    validation_schedule_digest: str


def _read_object(path: Path, label: str) -> dict[str, object]:
    path = _resolved_unlinked(path, label)
    if path.is_symlink() or path.is_junction() or not path.is_file():
        raise FactorFiLMExecutionError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda item: _invalid_json_constant(item),
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise FactorFiLMExecutionError(f"{label} is not valid UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise FactorFiLMExecutionError(f"{label} must contain one JSON object")
    return cast(dict[str, object], value)


def _invalid_json_constant(value: str) -> object:
    raise ValueError(f"invalid JSON constant {value}")


def _resolved_unlinked(path: Path, label: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    for component in (lexical, *lexical.parents):
        if component.is_symlink() or component.is_junction():
            raise FactorFiLMExecutionError(f"{label} traverses a link: {component}")
    return lexical.resolve(strict=False)


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _write_or_validate(path: Path, payload: Mapping[str, object]) -> None:
    path = _resolved_unlinked(path, "generated artifact")
    parent = _resolved_unlinked(path.parent, "generated artifact parent")
    if not parent.is_dir():
        parent.mkdir(parents=True, exist_ok=True)
        parent = _resolved_unlinked(parent, "generated artifact parent")
    if not parent.is_dir():
        raise FactorFiLMExecutionError("generated artifact parent is not a real directory")
    if path.exists():
        if not path.is_file():
            raise FactorFiLMExecutionError(f"generated artifact is not a real file: {path}")
        if _read_object(path, path.name) != dict(payload):
            raise FactorFiLMExecutionError(f"immutable generated artifact changed: {path}")
        return
    atomic_write_json(path, payload, immutable=True)


def _real_directory(path: Path, label: str) -> Path:
    path = _resolved_unlinked(path, label)
    if path.exists():
        if not path.is_dir():
            raise FactorFiLMExecutionError(f"{label} must be a real directory: {path}")
    else:
        path.mkdir(parents=False, exist_ok=False)
    path = _resolved_unlinked(path, label)
    if not path.is_dir():
        raise FactorFiLMExecutionError(f"{label} must be a real directory: {path}")
    return path


def _safe_child_directory(parent: Path, name: str, label: str) -> Path:
    parent = _real_directory(parent, f"{label} parent")
    child = _resolved_unlinked(parent / name, label)
    try:
        child.relative_to(parent)
    except ValueError as error:
        raise FactorFiLMExecutionError(f"{label} escapes its declared parent") from error
    return _real_directory(child, label)


def _discover_run_root(model_root: Path) -> Path:
    root = _resolved_unlinked(model_root, "FactorFiLM model root")
    if not root.is_dir():
        raise FactorFiLMExecutionError("FactorFiLM model root does not exist")
    candidates = tuple(
        item
        for item in root.iterdir()
        if item.is_dir() and not item.is_symlink() and (item / "complete.json").is_file()
    )
    if len(candidates) != 1:
        raise FactorFiLMExecutionError(
            "FactorFiLM model root must contain exactly one completed target run; "
            "pass --run-root when it does not"
        )
    return candidates[0].resolve()


def _load_training_authority(config: FactorFiLMExecutionConfig) -> _TrainingAuthority:
    if not torch.cuda.is_available():
        raise FactorFiLMExecutionError("FactorFiLM target-development requires available CUDA")
    paths = config.paths
    run_root = (
        _discover_run_root(paths.factor_film_model_root)
        if paths.run_root is None
        else _resolved_unlinked(paths.run_root, "FactorFiLM training run")
    )
    output_root = _resolved_unlinked(paths.output_root, "FactorFiLM evaluation output")
    project_root = Path(__file__).resolve().parents[3]
    protected = (
        run_root,
        _resolved_unlinked(paths.dataset_root, "M3B dataset root"),
        _resolved_unlinked(paths.m4_checkpoint_root, "M4 checkpoint root"),
        _resolved_unlinked(paths.task_token_checkpoint_root, "TaskToken checkpoint root"),
        _resolved_unlinked(paths.m42_diagnostics_root, "M4.2 diagnostics root"),
        _resolved_unlinked(paths.semantic_audit_evidence_root, "M4.3a evidence root"),
        *(
            project_root / name
            for name in ("src", "scripts", "environment", "tests", "docs", ".git")
        ),
    )
    if any(
        _contains(root.resolve(strict=False), output_root)
        or _contains(output_root, root.resolve(strict=False))
        for root in protected
    ):
        raise FactorFiLMExecutionError(
            "evaluation output overlaps immutable input, source, evidence, or Git content"
        )
    manifest_payload = _read_object(run_root / "run_manifest.json", "training manifest")
    manifest = FactorFiLMTrainingManifest.from_dict(manifest_payload)
    completion = _read_object(run_root / "complete.json", "training completion")
    queue = FactorFiLMValidationQueue.from_dict(
        _read_object(run_root / "validation_queue.json", "validation queue")
    )
    identity = manifest.identity
    if (
        identity.git_commit != config.expected_training_git_commit
        or identity.git_dirty
        or identity.experiment_mode is not FactorFiLMTrainingMode.TARGET_DEVELOPMENT
        or identity.device != "cuda"
        or manifest.training_config.mode is not FactorFiLMTrainingMode.TARGET_DEVELOPMENT
        or manifest.training_config.device != "cuda"
        or identity.m3b_export_fingerprint != AUTHORIZED_M3B_FINGERPRINT
        or identity.m3b_split_manifest_digest != AUTHORIZED_M3B_SPLIT_FINGERPRINT
        or identity.semantic_audit_evidence_fingerprint != AUTHORIZED_M43_AUDIT_EVIDENCE_FINGERPRINT
        or manifest.training_config.semantic_audit_evidence_fingerprint
        != AUTHORIZED_M43_AUDIT_EVIDENCE_FINGERPRINT
        or manifest.checkpoint_contract.execution_horizon != 10
        or manifest.checkpoint_contract.gripper_runtime_mode != "project"
        or manifest.training_config.execution_horizon != 10
        or manifest.training_config.gripper_runtime_mode != "project"
    ):
        raise FactorFiLMExecutionError("FactorFiLM target training authority changed")
    if not manifest.training_complete or len(manifest.checkpoints) != 20:
        raise FactorFiLMExecutionError("FactorFiLM training is not complete with 20 checkpoints")
    expected_steps = tuple(range(5_000, 100_001, 5_000))
    if tuple(item.global_step for item in manifest.checkpoints) != expected_steps:
        raise FactorFiLMExecutionError("FactorFiLM training checkpoint schedule changed")
    if set(completion) != {
        "schema_version",
        "passed",
        "run_fingerprint",
        "global_step",
        "checkpoint_count",
        "manifest_fingerprint",
        "validation_queue_fingerprint",
        "factor_film_training_completed",
        "factor_film_checkpoint_selected",
        "development_benchmark_completed",
        "final_schedule_accessed",
        "smolvla_go",
    }:
        raise FactorFiLMExecutionError("FactorFiLM training completion schema changed")
    if (
        completion["schema_version"] != FACTOR_FILM_TRAINING_COMPLETION_SCHEMA_VERSION
        or completion["passed"] is not True
        or completion["run_fingerprint"] != manifest.identity.run_fingerprint
        or completion["global_step"] != 100_000
        or completion["checkpoint_count"] != 20
        or completion["manifest_fingerprint"] != canonical_fingerprint(manifest_payload)
        or completion["validation_queue_fingerprint"] != queue.queue_fingerprint
        or completion["factor_film_training_completed"] is not True
        or completion["factor_film_checkpoint_selected"] is not False
        or completion["development_benchmark_completed"] is not False
        or completion["final_schedule_accessed"] is not False
        or completion["smolvla_go"] is not False
        or queue.run_fingerprint != manifest.identity.run_fingerprint
    ):
        raise FactorFiLMExecutionError("FactorFiLM training completion is inconsistent")

    audit = validate_authorizing_semantic_audit(paths.semantic_audit_evidence_root)
    if audit.evidence_fingerprint != manifest.training_config.semantic_audit_evidence_fingerprint:
        raise FactorFiLMExecutionError("semantic-audit evidence differs from training identity")
    completed = load_completed_m3b_dataset(
        paths.dataset_root,
        require_full=True,
        validate_storage=True,
    )
    records = factor_film_validation_schedule_records(completed, completed.views.validation)
    schedule_digest = factor_film_validation_schedule_digest(completed, completed.views.validation)
    if (
        completed.export_fingerprint != manifest.identity.m3b_export_fingerprint
        or completed.split_manifest_digest != manifest.identity.m3b_split_manifest_digest
        or schedule_digest != queue.validation_schedule_digest
        or len(records) != 36
    ):
        raise FactorFiLMExecutionError("FactorFiLM evaluation data authority changed")
    for checkpoint, queued in zip(manifest.checkpoints, queue.items, strict=True):
        if (
            checkpoint.checkpoint_fingerprint != queued.checkpoint_fingerprint
            or checkpoint.global_step != queued.checkpoint_step
            or checkpoint.relative_path != queued.checkpoint_relative_path
        ):
            raise FactorFiLMExecutionError("validation queue differs from training manifest")
        validated = validate_act_checkpoint_artifacts(
            run_root=run_root,
            checkpoint_relative_path=checkpoint.relative_path,
            expected_identity=manifest.identity,
        )
        if validated.record != checkpoint:
            raise FactorFiLMExecutionError("checkpoint artifacts differ from training manifest")

    evidence_config = FactorFiLMEvaluationEvidenceConfig(
        run_fingerprint=manifest.identity.run_fingerprint,
        training_manifest_fingerprint=canonical_fingerprint(manifest_payload),
        training_completion_fingerprint=canonical_fingerprint(completion),
        validation_queue_fingerprint=queue.queue_fingerprint,
        dataset_fingerprint=manifest.identity.m3b_export_fingerprint,
        split_fingerprint=manifest.identity.m3b_split_manifest_digest,
        semantic_audit_evidence_fingerprint=audit.evidence_fingerprint,
        validation_schedule_digest=queue.validation_schedule_digest,
        development_schedule_fingerprint=M42_DEV_SCHEDULE_FINGERPRINT,
        training_git_commit=manifest.identity.git_commit,
        evaluation_git_commit=config.evaluation_git_commit,
    )
    return _TrainingAuthority(
        run_root=run_root,
        manifest=manifest,
        completion=completion,
        queue=queue,
        evidence_config=evidence_config,
        validation_schedule_records=records,
        validation_schedule_digest=schedule_digest,
    )


def _work_root(config: FactorFiLMExecutionConfig, authority: _TrainingAuthority) -> Path:
    output = _resolved_unlinked(config.paths.output_root, "FactorFiLM evaluation output")
    if not output.exists():
        output.mkdir(parents=True, exist_ok=False)
    output = _real_directory(output, "FactorFiLM evaluation output")
    token = authority.evidence_config.fingerprint.removeprefix("sha256:")
    work = _safe_child_directory(output, "work", "FactorFiLM work parent")
    root = _safe_child_directory(work, token, "FactorFiLM fingerprint-owned work root")
    owner = {
        "schema_version": FACTOR_FILM_WORK_SCHEMA_VERSION,
        "config_fingerprint": authority.evidence_config.fingerprint,
        "config": authority.evidence_config.to_dict(),
    }
    _write_or_validate(root / "owner.json", owner)
    return root


def _retarget_context(context: M42CheckpointContext, device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        raise FactorFiLMExecutionError("target-development requested unavailable CUDA")
    target = torch.device(device)
    policy = cast(Any, context.loaded.policy)
    config = getattr(policy, "config", None)
    if config is not None and hasattr(config, "device"):
        config.device = device
    policy.to(target)
    policy.eval()
    for pipeline in (context.loaded.preprocessor, context.loaded.postprocessor):
        steps = getattr(pipeline, "steps", None)
        if not isinstance(steps, Sequence):
            raise FactorFiLMExecutionError("loaded ACT processor lacks public steps")
        for step in steps:
            if hasattr(step, "device"):
                step.device = target
            if hasattr(step, "stats") and hasattr(step, "_tensor_stats"):
                initialize = getattr(step, "__post_init__", None)
                if not callable(initialize):
                    raise FactorFiLMExecutionError("normalization processor cannot retarget")
                initialize()


def _load_factor_context(
    authority: _TrainingAuthority, checkpoint_fingerprint: str, device: str
) -> M42CheckpointContext:
    context = load_factor_film_training_checkpoint_context(
        authority.run_root,
        expected_checkpoint_fingerprint=checkpoint_fingerprint,
        expected_run_fingerprint=authority.manifest.identity.run_fingerprint,
        expected_dataset_fingerprint=authority.manifest.identity.m3b_export_fingerprint,
        expected_architecture_fingerprint=(
            authority.manifest.architecture_identity.architecture_fingerprint
        ),
    )
    _retarget_context(context, device)
    return context


def _collect_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _benchmark_wrapper(
    *, authority: _TrainingAuthority, report: M42BenchmarkReport, kind: str
) -> dict[str, object]:
    return {
        "schema_version": (
            FACTOR_FILM_VALIDATION_REPORT_SCHEMA_VERSION
            if kind == "validation"
            else FACTOR_FILM_DEVELOPMENT_REPORT_SCHEMA_VERSION
        ),
        "run_fingerprint": authority.manifest.identity.run_fingerprint,
        "validation_queue_fingerprint": authority.queue.queue_fingerprint,
        "benchmark_fingerprint": report.fingerprint,
        "benchmark": report.to_dict(),
        "test_split_accessed": False,
        "fresh_seed_accessed": False,
        "final_schedule_accessed": False,
    }


def _load_benchmark_wrapper(
    path: Path, *, authority: _TrainingAuthority, kind: str
) -> dict[str, object] | None:
    if not path.exists():
        return None
    value = _read_object(path, f"{kind} benchmark")
    expected_schema = (
        FACTOR_FILM_VALIDATION_REPORT_SCHEMA_VERSION
        if kind == "validation"
        else FACTOR_FILM_DEVELOPMENT_REPORT_SCHEMA_VERSION
    )
    benchmark = value.get("benchmark")
    if (
        value.get("schema_version") != expected_schema
        or value.get("run_fingerprint") != authority.manifest.identity.run_fingerprint
        or value.get("validation_queue_fingerprint") != authority.queue.queue_fingerprint
        or not isinstance(benchmark, Mapping)
        or value.get("benchmark_fingerprint") != canonical_fingerprint(benchmark)
        or any(
            value.get(name) is not False
            for name in (
                "test_split_accessed",
                "fresh_seed_accessed",
                "final_schedule_accessed",
            )
        )
    ):
        raise FactorFiLMExecutionError(f"stored {kind} benchmark is inconsistent")
    return value


def _runtime_fingerprint(env: object) -> str:
    processor = BoundedActionEnvPostprocessorV0.from_environment(
        env,
        ActionBoundConfig(mode=ActionBoundMode.PROJECT),
        expected_action_components=8,
    )
    return canonical_fingerprint(
        {
            "component": "BoundedActionEnvPostprocessorV0",
            "config": processor.config.to_dict(),
            "action_space_contract": processor.action_space_contract(),
        }
    )


def _validation_inputs(config: FactorFiLMExecutionConfig) -> object:
    return load_restricted_audit_inputs(
        mode="validation",
        dataset_root=config.paths.dataset_root,
        m4_checkpoint_root=config.paths.m4_checkpoint_root,
        task_token_checkpoint_root=config.paths.task_token_checkpoint_root,
        m42_diagnostics_root=config.paths.m42_diagnostics_root,
        runtime_selection_path=config.paths.runtime_selection_path,
    )


def _observation_fingerprint(observation: FixedPolicyObservation) -> str:
    return canonical_fingerprint(
        {
            "source_scope": observation.source_scope,
            "scene_key": observation.scene_key,
            "observation_id": observation.observation_id,
            "requested_task_id": observation.requested_task_id,
            "source_identity": dict(observation.source_identity),
            "image_digest": observation.image_digest,
            "state_digest": observation.state_digest,
        }
    )


def execute_validation_stage(config: FactorFiLMExecutionConfig) -> dict[str, object]:
    """Run or reuse all 20x36 validation reports, then lock selection."""
    authority = _load_training_authority(config)
    work = _work_root(config, authority)
    schedule_records = authority.validation_schedule_records
    schedule_digest = authority.validation_schedule_digest
    results: list[FactorFiLMValidationCountResult] = []
    validation_root = _safe_child_directory(work, "validation", "validation work root")
    env = create_m43_audit_environment(rgb=True)
    try:
        for item in authority.queue.items:
            path = validation_root / f"step-{item.checkpoint_step:08d}.json"
            stored = _load_benchmark_wrapper(path, authority=authority, kind="validation")
            if stored is None:
                context = _load_factor_context(
                    authority, item.checkpoint_fingerprint, config.device
                )
                report = run_m42_validation_benchmark(
                    env=env,
                    checkpoint=context,
                    schedule_records=schedule_records,
                    validation_schedule_fingerprint=schedule_digest,
                    execution_horizon=10,
                    gripper_mode=GripperRuntimeMode.PROJECT,
                    maximum_episode_steps=M42_MAXIMUM_EPISODE_STEPS,
                )
                stored = _benchmark_wrapper(authority=authority, report=report, kind="validation")
                _write_or_validate(path, stored)
                del context
                _collect_cuda()
            benchmark = cast(Mapping[str, object], stored["benchmark"])
            aggregate = benchmark.get("aggregate")
            checkpoint = benchmark.get("checkpoint")
            if not isinstance(aggregate, Mapping) or not isinstance(checkpoint, Mapping):
                raise FactorFiLMExecutionError("validation benchmark is malformed")
            if (
                checkpoint.get("checkpoint_fingerprint") != item.checkpoint_fingerprint
                or benchmark.get("schedule_fingerprint") != schedule_digest
            ):
                raise FactorFiLMExecutionError("validation benchmark identity changed")
            result = factor_film_validation_count_result(
                checkpoint_fingerprint=item.checkpoint_fingerprint,
                checkpoint_step=item.checkpoint_step,
                schedule_digest=schedule_digest,
                aggregate=aggregate,
                offline_validation_action_loss=item.offline_validation_action_loss,
            )
            results.append(result)
        ranked = rank_factor_film_validation_count_results(results)
        selection = FactorFiLMSelectionRecord(
            run_fingerprint=authority.queue.run_fingerprint,
            validation_queue_fingerprint=authority.queue.queue_fingerprint,
            validation_queue=authority.queue,
            validation_schedule_digest=schedule_digest,
            candidates=tuple(value.to_selection_result() for value in results),
            ranked_checkpoint_fingerprints=tuple(value.checkpoint_fingerprint for value in ranked),
            selected_checkpoint_fingerprint=ranked[0].checkpoint_fingerprint,
            selected_checkpoint_step=ranked[0].checkpoint_step,
        )
        _write_or_validate(work / "selection.json", selection.to_dict())

        inputs = cast(Any, _validation_inputs(config))
        observations, _ = load_m43_validation_observations(inputs)
        observation = observations[0]
        selected = _load_factor_context(
            authority, selection.selected_checkpoint_fingerprint, config.device
        )
        output = predict_postprocessed_action_chunk(
            context=selected,
            observation=observation,
            task_id=observation.requested_task_id,
        )
        array = np.asarray(output, dtype=np.float64)
        if array.shape != (50, 8) or not np.all(np.isfinite(array)):
            raise FactorFiLMExecutionError("reload reference must be finite float[50,8]")
        component = selected.loaded.component_fingerprints
        reference = {
            "schema_version": FACTOR_FILM_REFERENCE_OUTPUT_SCHEMA_VERSION,
            "run_fingerprint": authority.manifest.identity.run_fingerprint,
            "selection_fingerprint": canonical_fingerprint(selection.to_dict()),
            "selected_checkpoint_fingerprint": selection.selected_checkpoint_fingerprint,
            "selected_checkpoint_step": selection.selected_checkpoint_step,
            "validation_observation_fingerprint": _observation_fingerprint(observation),
            "reference_output": array.tolist(),
            "reference_output_fingerprint": canonical_fingerprint(array.tolist()),
            "architecture_fingerprint": (
                authority.manifest.architecture_identity.architecture_fingerprint
            ),
            "preprocessor_fingerprint": component.preprocessor,
            "policy_postprocessor_fingerprint": component.postprocessor,
            "action_runtime_fingerprint": _runtime_fingerprint(env),
            "test_split_accessed": False,
            "fresh_seed_accessed": False,
            "final_schedule_accessed": False,
        }
        _write_or_validate(work / "reload_reference.json", reference)
        del selected
        _collect_cuda()
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    return {
        "stage": "validation",
        "validation_checkpoint_count": 20,
        "validation_episode_count": 720,
        "selection": selection.to_dict(),
        "work_root": str(work),
        "passed": True,
    }


def execute_reload_stage(config: FactorFiLMExecutionConfig) -> dict[str, object]:
    """Fresh-process selected checkpoint/processors/runtime reload validation."""
    authority = _load_training_authority(config)
    work = _work_root(config, authority)
    selection = FactorFiLMSelectionRecord.from_dict(
        _read_object(work / "selection.json", "FactorFiLM selection")
    )
    reference = _read_object(work / "reload_reference.json", "reload reference")
    inputs = cast(Any, _validation_inputs(config))
    observations, _ = load_m43_validation_observations(inputs)
    observation = observations[0]
    if reference.get("validation_observation_fingerprint") != _observation_fingerprint(observation):
        raise FactorFiLMExecutionError("reload validation observation changed")
    raw_reference = reference.get("reference_output")
    if not isinstance(raw_reference, list):
        raise FactorFiLMExecutionError("reload reference output is malformed")
    expected = np.asarray(raw_reference, dtype=np.float64)
    context = _load_factor_context(
        authority, selection.selected_checkpoint_fingerprint, config.device
    )
    observed = np.asarray(
        predict_postprocessed_action_chunk(
            context=context,
            observation=observation,
            task_id=observation.requested_task_id,
        ),
        dtype=np.float64,
    )
    if expected.shape != (50, 8) or observed.shape != (50, 8):
        raise FactorFiLMExecutionError("fresh reload output must be [50,8]")
    absolute = np.abs(expected - observed)
    relative = absolute / np.maximum(np.abs(expected), 1e-12)
    component = context.loaded.component_fingerprints
    env = create_m43_audit_environment(rgb=False)
    try:
        runtime_fingerprint = _runtime_fingerprint(env)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    fingerprints_validated = (
        reference.get("architecture_fingerprint")
        == authority.manifest.architecture_identity.architecture_fingerprint
        and reference.get("preprocessor_fingerprint") == component.preprocessor
        and reference.get("policy_postprocessor_fingerprint") == component.postprocessor
        and reference.get("action_runtime_fingerprint") == runtime_fingerprint
    )
    deterministic = bool(np.allclose(expected, observed, atol=1e-6, rtol=1e-6))
    record = FactorFiLMReloadValidationRecord(
        run_fingerprint=authority.manifest.identity.run_fingerprint,
        selection_fingerprint=canonical_fingerprint(selection.to_dict()),
        selected_checkpoint_fingerprint=selection.selected_checkpoint_fingerprint,
        selected_checkpoint_step=selection.selected_checkpoint_step,
        architecture_fingerprint=(
            authority.manifest.architecture_identity.architecture_fingerprint
        ),
        preprocessor_fingerprint=component.preprocessor,
        policy_postprocessor_fingerprint=component.postprocessor,
        action_runtime_fingerprint=runtime_fingerprint,
        validation_observation_fingerprint=_observation_fingerprint(observation),
        reference_output_fingerprint=canonical_fingerprint(expected.tolist()),
        reloaded_output_fingerprint=canonical_fingerprint(observed.tolist()),
        maximum_absolute_error=float(absolute.max(initial=0.0)),
        maximum_relative_error=float(relative.max(initial=0.0)),
        fresh_policy_instance=True,
        preprocessor_reloaded=True,
        policy_postprocessor_reloaded=True,
        architecture_reconstructed=True,
        task_mappings_reconstructed=True,
        action_runtime_reconstructed=True,
        fingerprints_validated=fingerprints_validated,
        deterministic_inference_validated=deterministic,
        reload_validated=fingerprints_validated and deterministic,
    )
    validate_reload_against_selection(record, selection)
    _write_or_validate(work / "reload.json", record.to_dict())
    del context
    _collect_cuda()
    return {"stage": "reload", "reload": record.to_dict(), "passed": True}


def _report_identity(value: Mapping[str, object]) -> DevelopmentEpisodeIdentity:
    rollout = value.get("rollout")
    if not isinstance(rollout, Mapping):
        raise FactorFiLMExecutionError("development episode lacks rollout identity")
    return DevelopmentEpisodeIdentity(
        episode_index=cast(int, value["episode_index"]),
        scene_seed=cast(int, rollout["scene_seed"]),
        scene_id=cast(str, rollout["scene_id"]),
        task_id=cast(str, rollout["task_id"]),
    )


def _benchmark_episodes(wrapper: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    benchmark = wrapper.get("benchmark")
    if not isinstance(benchmark, Mapping):
        raise FactorFiLMExecutionError("development wrapper lacks benchmark")
    episodes = benchmark.get("episodes")
    if not isinstance(episodes, list) or not all(isinstance(item, Mapping) for item in episodes):
        raise FactorFiLMExecutionError("development benchmark episode records are malformed")
    return tuple(cast(Mapping[str, object], item) for item in episodes)


def _benchmark_aggregate(wrapper: Mapping[str, object]) -> Mapping[str, object]:
    benchmark = wrapper.get("benchmark")
    if not isinstance(benchmark, Mapping) or not isinstance(benchmark.get("aggregate"), Mapping):
        raise FactorFiLMExecutionError("development benchmark aggregate is malformed")
    return cast(Mapping[str, object], benchmark["aggregate"])


def _run_or_load_development(
    *,
    path: Path,
    authority: _TrainingAuthority,
    env: object,
    context: M42CheckpointContext,
    episodes: Sequence[object],
    label: str,
) -> dict[str, object]:
    stored = _load_benchmark_wrapper(path, authority=authority, kind="development")
    if stored is None:
        report = run_m42_checkpoint_benchmark(
            env=env,
            checkpoint=context,
            episodes=cast(Any, episodes),
            schedule_fingerprint=M42_DEV_SCHEDULE_FINGERPRINT,
            execution_horizon=10,
            gripper_mode=GripperRuntimeMode.PROJECT,
            model_label=label,
            maximum_episode_steps=M42_MAXIMUM_EPISODE_STEPS,
        )
        stored = _benchmark_wrapper(authority=authority, report=report, kind="development")
        _write_or_validate(path, stored)
    benchmark = cast(Mapping[str, object], stored["benchmark"])
    checkpoint = benchmark.get("checkpoint")
    stored_episodes = benchmark.get("episodes")
    if (
        not isinstance(checkpoint, Mapping)
        or not isinstance(stored_episodes, list)
        or benchmark.get("schedule_id") != M42_DEV_SCHEDULE_ID
        or benchmark.get("schedule_fingerprint") != M42_DEV_SCHEDULE_FINGERPRINT
        or benchmark.get("execution_horizon") != 10
        or benchmark.get("gripper_mode") != GripperRuntimeMode.PROJECT.value
        or benchmark.get("model_label") != label
        or checkpoint.get("checkpoint_fingerprint") != context.descriptor.checkpoint_fingerprint
        or len(stored_episodes) != len(episodes)
    ):
        raise FactorFiLMExecutionError("stored development benchmark identity changed")
    return stored


def _policy_metrics(factor_wrapper: Mapping[str, object]) -> DevelopmentPolicyMetrics:
    aggregate = _benchmark_aggregate(factor_wrapper)
    action = aggregate.get("action_metrics")
    per_task = aggregate.get("per_task")
    benchmark = cast(Mapping[str, object], factor_wrapper["benchmark"])
    if not isinstance(action, Mapping) or not isinstance(per_task, Mapping):
        raise FactorFiLMExecutionError("FactorFiLM development aggregate is incomplete")
    counts: dict[str, int] = {}
    for task_id in M43_CANONICAL_TASK_IDS:
        item = per_task.get(task_id)
        if not isinstance(item, Mapping):
            raise FactorFiLMExecutionError("FactorFiLM per-task aggregate is incomplete")
        counts[task_id] = int(cast(int, item["successes"]))
    return DevelopmentPolicyMetrics(
        report_fingerprint=canonical_fingerprint(benchmark),
        success_count=int(cast(int, aggregate["successes"])),
        per_task_success_counts=counts,
        wrong_object_grasp_count=int(cast(int, aggregate["wrong_object_grasp_count"])),
        wrong_object_in_target_bin_count=int(
            cast(int, aggregate["wrong_object_in_target_bin_count"])
        ),
        target_in_wrong_bin_count=int(cast(int, aggregate["target_in_wrong_bin_count"])),
        target_off_table_count=int(cast(int, aggregate["target_off_table_count"])),
        timeout_count=int(cast(int, aggregate["timeout_count"])),
        arm_projection_count=int(cast(int, action["arm_projected_component_count"])),
        nan_count=int(cast(int, action["nan_count"])),
        inf_count=int(cast(int, action["inf_count"])),
        malformed_action_count=int(cast(int, action["malformed_action_count"])),
    )


def _raw_actions_by_step(episode: Mapping[str, object]) -> dict[int, Sequence[object]]:
    audits = episode.get("runtime_projection_audits")
    if not isinstance(audits, list):
        raise FactorFiLMExecutionError("FactorFiLM episode lacks raw projection audits")
    result: dict[int, Sequence[object]] = {}
    for raw in audits:
        if not isinstance(raw, Mapping):
            raise FactorFiLMExecutionError("projection audit is malformed")
        step = raw.get("rollout_step")
        action = raw.get("raw_action")
        if isinstance(step, bool) or not isinstance(step, int) or not isinstance(action, list):
            raise FactorFiLMExecutionError("projection audit raw action is malformed")
        if step in result:
            raise FactorFiLMExecutionError("projection audit contains a duplicate step")
        flattened = np.asarray(action, dtype=np.float64).reshape(-1)
        if flattened.shape != (8,) or not np.all(np.isfinite(flattened)):
            raise FactorFiLMExecutionError("projection raw action must be finite float[8]")
        result[step] = flattened.tolist()
    return result


def _first_interactions(
    *,
    factor_wrapper: Mapping[str, object],
    development_observations: Sequence[FixedPolicyObservation],
    nearest_task_ids: Sequence[str],
) -> tuple[FirstInteractionSnapshot, ...]:
    episodes = _benchmark_episodes(factor_wrapper)
    if len(episodes) != 72 or len(development_observations) != 72 or len(nearest_task_ids) != 72:
        raise FactorFiLMExecutionError("first-interaction inputs must cover 72 episodes")
    results: list[FirstInteractionSnapshot] = []
    for episode, observation, nearest in zip(
        episodes, development_observations, nearest_task_ids, strict=True
    ):
        rollout = episode.get("rollout")
        snapshots = episode.get("semantic_interaction_snapshots")
        if not isinstance(rollout, Mapping) or not isinstance(snapshots, list):
            raise FactorFiLMExecutionError("FactorFiLM episode lacks semantic step snapshots")
        if rollout.get(
            "task_id"
        ) != observation.requested_task_id or observation.source_identity.get(
            "scene_id"
        ) != rollout.get("scene_id"):
            raise FactorFiLMExecutionError("semantic observation and rollout identity differ")
        raw_actions = _raw_actions_by_step(episode)
        snapshot_steps = {
            int(cast(Mapping[str, object], item)["step"])
            for item in snapshots[1:]
            if isinstance(item, Mapping)
        }
        if not snapshot_steps <= set(raw_actions):
            raise FactorFiLMExecutionError(
                "raw action release evidence does not align with simulator snapshots"
            )
        executed_raw_actions = {step: raw_actions[step] for step in snapshot_steps}
        status = cast(str, rollout.get("status"))
        results.append(
            derive_first_interaction_snapshot(
                observation_id=observation.observation_id,
                requested_task_id=observation.requested_task_id,
                nearest_per_task_task_id=nearest,
                step_snapshots=cast(Any, snapshots),
                timed_out=bool(rollout.get("timeout")),
                environment_failure=(
                    bool(rollout.get("inference_failure"))
                    or bool(rollout.get("invalid_action"))
                    or status in {"environment_failure", "truncated"}
                ),
                raw_actions_by_step=executed_raw_actions,
            )
        )
    return tuple(results)


def _semantic_metrics(
    *,
    validation: object,
    development: object,
    reference_sensitivity: Mapping[str, object],
) -> DevelopmentSemanticMetrics:
    dev = cast(Any, development)
    audit = dev.policy_audits[_EVIDENCE_POLICY_LABEL_FACTOR]
    top1 = sum(item.top1_correct for item in audit.task_retrieval) / audit.observation_count
    object_accuracy = sum(item.correct for item in audit.object_retrieval) / audit.observation_count
    candidate_sensitivity = cast(
        Mapping[str, object],
        dev.policy_runtime_summaries[_EVIDENCE_POLICY_LABEL_FACTOR]["task_sensitivity"],
    )
    candidate_mean = float(cast(float, candidate_sensitivity["mean_full_chunk_rms"]))
    reference_mean = float(cast(float, reference_sensitivity["mean_full_chunk_rms"]))
    if not math.isfinite(reference_mean) or reference_mean <= 0:
        raise FactorFiLMExecutionError("PerTask reference sensitivity must be positive")
    report_fingerprint = canonical_fingerprint(
        {
            "validation": cast(Any, validation).to_dict(),
            "development": dev.to_dict(),
            "per_task_reference_sensitivity": dict(reference_sensitivity),
        }
    )
    return DevelopmentSemanticMetrics(
        semantic_report_fingerprint=report_fingerprint,
        full_task_top1_retrieval=top1,
        target_object_retrieval=object_accuracy,
        task_sensitivity_ratio_relative_to_per_task=candidate_mean / reference_mean,
    )


def _stage_artifacts(
    *,
    authority: _TrainingAuthority,
    results: Sequence[FactorFiLMValidationCountResult],
    validation_benchmark_reports: Sequence[Mapping[str, object]],
    selection: FactorFiLMSelectionRecord,
    reload_record: FactorFiLMReloadValidationRecord,
    development_payload: Mapping[str, object],
    semantic_payload: Mapping[str, object],
    quality_payload: Mapping[str, object],
) -> dict[FactorFiLMEvaluationStage, FactorFiLMEvaluationStageArtifact]:
    config = authority.evidence_config
    validation = FactorFiLMEvaluationStageArtifact(
        stage=FactorFiLMEvaluationStage.VALIDATION,
        run_fingerprint=config.run_fingerprint,
        input_artifact_fingerprints={
            "validation_queue": config.validation_queue_fingerprint,
            "validation_schedule": config.validation_schedule_digest,
        },
        payload={
            "schema_version": FACTOR_FILM_VALIDATION_EVIDENCE_SCHEMA_VERSION,
            "validation_queue_fingerprint": config.validation_queue_fingerprint,
            "validation_schedule_digest": config.validation_schedule_digest,
            "checkpoint_count": 20,
            "results": [item.to_dict() for item in results],
            "benchmark_reports": [dict(item) for item in validation_benchmark_reports],
        },
    )
    selected = {
        "selected_checkpoint_fingerprint": selection.selected_checkpoint_fingerprint,
        "selected_checkpoint_step": selection.selected_checkpoint_step,
    }
    selection_stage = FactorFiLMEvaluationStageArtifact(
        stage=FactorFiLMEvaluationStage.SELECTION,
        run_fingerprint=config.run_fingerprint,
        input_artifact_fingerprints={"validation": validation.artifact_fingerprint},
        payload=selection.to_dict(),
        **selected,
    )
    reload_stage = FactorFiLMEvaluationStageArtifact(
        stage=FactorFiLMEvaluationStage.RELOAD,
        run_fingerprint=config.run_fingerprint,
        input_artifact_fingerprints={"selection": selection_stage.artifact_fingerprint},
        payload=reload_record.to_dict(),
        **selected,
    )
    development_stage = FactorFiLMEvaluationStageArtifact(
        stage=FactorFiLMEvaluationStage.DEVELOPMENT,
        run_fingerprint=config.run_fingerprint,
        input_artifact_fingerprints={
            "reload": reload_stage.artifact_fingerprint,
            "development_schedule": config.development_schedule_fingerprint,
        },
        payload=development_payload,
        **selected,
    )
    semantic_stage = FactorFiLMEvaluationStageArtifact(
        stage=FactorFiLMEvaluationStage.SEMANTIC,
        run_fingerprint=config.run_fingerprint,
        input_artifact_fingerprints={
            "validation": validation.artifact_fingerprint,
            "development": development_stage.artifact_fingerprint,
            "semantic_audit": config.semantic_audit_evidence_fingerprint,
        },
        payload=semantic_payload,
        **selected,
    )
    quality_stage = FactorFiLMEvaluationStageArtifact(
        stage=FactorFiLMEvaluationStage.QUALITY_GATE,
        run_fingerprint=config.run_fingerprint,
        input_artifact_fingerprints={
            "development": development_stage.artifact_fingerprint,
            "semantic": semantic_stage.artifact_fingerprint,
        },
        payload=quality_payload,
        **selected,
    )
    return {
        FactorFiLMEvaluationStage.VALIDATION: validation,
        FactorFiLMEvaluationStage.SELECTION: selection_stage,
        FactorFiLMEvaluationStage.RELOAD: reload_stage,
        FactorFiLMEvaluationStage.DEVELOPMENT: development_stage,
        FactorFiLMEvaluationStage.SEMANTIC: semantic_stage,
        FactorFiLMEvaluationStage.QUALITY_GATE: quality_stage,
    }


def _validation_results_from_work(
    *, work: Path, authority: _TrainingAuthority
) -> tuple[
    tuple[FactorFiLMValidationCountResult, ...],
    tuple[Mapping[str, object], ...],
]:
    results: list[FactorFiLMValidationCountResult] = []
    reports: list[Mapping[str, object]] = []
    for item in authority.queue.items:
        stored = _load_benchmark_wrapper(
            work / "validation" / f"step-{item.checkpoint_step:08d}.json",
            authority=authority,
            kind="validation",
        )
        if stored is None:
            raise FactorFiLMExecutionError("development requires all validation reports")
        benchmark = stored.get("benchmark")
        if not isinstance(benchmark, Mapping) or not isinstance(
            benchmark.get("aggregate"), Mapping
        ):
            raise FactorFiLMExecutionError("stored validation aggregate is malformed")
        reports.append(benchmark)
        results.append(
            factor_film_validation_count_result(
                checkpoint_fingerprint=item.checkpoint_fingerprint,
                checkpoint_step=item.checkpoint_step,
                schedule_digest=authority.queue.validation_schedule_digest,
                aggregate=cast(Mapping[str, object], benchmark["aggregate"]),
                offline_validation_action_loss=item.offline_validation_action_loss,
            )
        )
    return tuple(results), tuple(reports)


def execute_development_stage(config: FactorFiLMExecutionConfig) -> dict[str, object]:
    """Run paired 3x72 development, semantic analysis and immutable promotion."""
    authority = _load_training_authority(config)
    work = _work_root(config, authority)
    selection = FactorFiLMSelectionRecord.from_dict(
        _read_object(work / "selection.json", "FactorFiLM selection")
    )
    reload_record = FactorFiLMReloadValidationRecord.from_dict(
        _read_object(work / "reload.json", "FactorFiLM reload")
    )
    validate_reload_against_selection(reload_record, selection)

    inputs = load_restricted_audit_inputs(
        mode="combined",
        dataset_root=config.paths.dataset_root,
        m4_checkpoint_root=config.paths.m4_checkpoint_root,
        task_token_checkpoint_root=config.paths.task_token_checkpoint_root,
        m42_diagnostics_root=config.paths.m42_diagnostics_root,
        runtime_selection_path=config.paths.runtime_selection_path,
    )
    if (
        inputs.runtime.execution_horizon != 10
        or inputs.runtime.gripper_mode != GripperRuntimeMode.PROJECT.value
        or inputs.runtime.development_schedule_fingerprint != M42_DEV_SCHEDULE_FINGERPRINT
    ):
        raise FactorFiLMExecutionError("M4.2 runtime selection differs from H10/project/dev")
    references = load_m43_reference_policy_contexts(inputs, device=config.device)
    factor = _load_factor_context(
        authority, selection.selected_checkpoint_fingerprint, config.device
    )
    schedule = load_locked_schedule(M42_DEV_SCHEDULE_ID)
    episodes = materialize_schedule(schedule)
    if len(episodes) != 72:
        raise FactorFiLMExecutionError("m42_dev_v0 must materialize exactly 72 episodes")
    env = create_m43_audit_environment(rgb=True)
    development_root = _safe_child_directory(work, "development", "development work root")
    try:
        per_task_wrappers: list[dict[str, object]] = []
        for task_id in M43_CANONICAL_TASK_IDS:
            context = references.per_task[task_id]
            selected_episodes = tuple(item for item in episodes if item.task_id == task_id)
            per_task_wrappers.append(
                _run_or_load_development(
                    path=development_root / f"per-task-{task_id}.json",
                    authority=authority,
                    env=env,
                    context=context,
                    episodes=selected_episodes,
                    label=f"{_POLICY_LABEL_PER_TASK}/{task_id}",
                )
            )
        onehot_wrapper = _run_or_load_development(
            path=development_root / "state-onehot.json",
            authority=authority,
            env=env,
            context=references.state_onehot,
            episodes=episodes,
            label=_POLICY_LABEL_ONEHOT,
        )
        factor_wrapper = _run_or_load_development(
            path=development_root / "factor-film.json",
            authority=authority,
            env=env,
            context=factor,
            episodes=episodes,
            label=_POLICY_LABEL_FACTOR,
        )

        per_task_episodes = tuple(
            item for wrapper in per_task_wrappers for item in _benchmark_episodes(wrapper)
        )
        onehot_episodes = _benchmark_episodes(onehot_wrapper)
        factor_episodes = _benchmark_episodes(factor_wrapper)
        paired = validate_paired_development_identities(
            per_task=tuple(_report_identity(item) for item in per_task_episodes),
            state_onehot=tuple(_report_identity(item) for item in onehot_episodes),
            factor_film=tuple(_report_identity(item) for item in factor_episodes),
            schedule_fingerprint=M42_DEV_SCHEDULE_FINGERPRINT,
        )
        factor_metrics = _policy_metrics(factor_wrapper)

        distance_config = m43_action_chunk_distance_config(
            env=env, policies=references, inputs=inputs
        )
        validation_observations, _ = load_m43_validation_observations(inputs)
        validation_semantic = compute_named_fixed_observation_audits(
            observations=validation_observations,
            per_task=references.per_task,
            candidates={
                _EVIDENCE_POLICY_LABEL_ONEHOT: references.state_onehot,
                _EVIDENCE_POLICY_LABEL_FACTOR: factor,
            },
            distance_config=distance_config,
        )
        development_observations, _ = load_m43_development_observations(env)
        development_semantic = compute_named_fixed_observation_audits(
            observations=development_observations,
            per_task=references.per_task,
            candidates={
                _EVIDENCE_POLICY_LABEL_ONEHOT: references.state_onehot,
                _EVIDENCE_POLICY_LABEL_FACTOR: factor,
            },
            distance_config=distance_config,
        )
        reference_sensitivity = compute_per_task_reference_sensitivity(
            observations=development_observations,
            per_task=references.per_task,
        )
        semantic_metrics = _semantic_metrics(
            validation=validation_semantic,
            development=development_semantic,
            reference_sensitivity=reference_sensitivity,
        )
        nearest = tuple(
            item.nearest_task_id
            for item in development_semantic.policy_audits[
                _EVIDENCE_POLICY_LABEL_FACTOR
            ].task_retrieval
        )
        interactions = _first_interactions(
            factor_wrapper=factor_wrapper,
            development_observations=development_observations,
            nearest_task_ids=nearest,
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    confusions = factor_film_first_interaction_confusions(interactions)
    timing = Counter(
        {
            "visible_in_initial_chunk": 0,
            "emerged_later": 0,
            "no_semantic_error": 0,
            "no_first_interaction": 0,
        }
    )
    for interaction in interactions:
        key = {
            SemanticErrorStage.INITIAL_CHUNK: "visible_in_initial_chunk",
            SemanticErrorStage.EMERGED_AFTER_INITIAL_CHUNK: "emerged_later",
            SemanticErrorStage.NONE: "no_semantic_error",
            SemanticErrorStage.NO_FIRST_INTERACTION: "no_first_interaction",
        }[interaction.semantic_error_stage]
        timing[key] += 1
    per_task_success = sum(
        int(cast(int, _benchmark_aggregate(wrapper)["successes"])) for wrapper in per_task_wrappers
    )
    gate = evaluate_development_quality_gate(
        factor_film_metrics=factor_metrics,
        semantic_metrics=semantic_metrics,
        per_task_reference_success_count=per_task_success,
    )
    factor_audit = development_semantic.policy_audits[_EVIDENCE_POLICY_LABEL_FACTOR]
    bin_correct = all(
        item.correct and item.conditional_correct for item in factor_audit.bin_retrieval
    )
    action_metrics = _benchmark_aggregate(factor_wrapper)["action_metrics"]
    if not isinstance(action_metrics, Mapping):
        raise FactorFiLMExecutionError("FactorFiLM runtime action metrics are malformed")

    development_payload = {
        "schema_version": "langmani-m43b-factor-film-development-stage-v0",
        "development_benchmark_completed": True,
        "physical_execution": config.device == "cuda",
        "device": config.device,
        "environment_id": ENV_ID,
        "schedule_id": M42_DEV_SCHEDULE_ID,
        "schedule_fingerprint": M42_DEV_SCHEDULE_FINGERPRINT,
        "execution_horizon": 10,
        "gripper_mode": GripperRuntimeMode.PROJECT.value,
        "m2_expert_invoked": False,
        "total_episode_count": 216,
        "per_policy_episode_counts": {
            _POLICY_LABEL_PER_TASK: 72,
            _POLICY_LABEL_ONEHOT: 72,
            _POLICY_LABEL_FACTOR: 72,
        },
        "paired_episode_identities_validated": True,
        "paired_identities": paired.to_dict(),
        "factor_film_metrics": factor_metrics.to_dict(),
        "benchmark_reports": {
            "per_task": [
                cast(Mapping[str, object], item["benchmark"]) for item in per_task_wrappers
            ],
            "state_onehot": cast(Mapping[str, object], onehot_wrapper["benchmark"]),
            "factor_film": cast(Mapping[str, object], factor_wrapper["benchmark"]),
        },
        "test_split_accessed": False,
        "fresh_seed_accessed": False,
        "final_schedule_accessed": False,
    }
    semantic_payload = {
        "schema_version": "langmani-m43b-factor-film-semantic-stage-v0",
        "semantic_evaluation_completed": True,
        "post_grasp_analysis_completed": True,
        "first_interaction_analysis_completed": True,
        "raw_action_metrics_validated": action_metrics.get("raw_action_metrics_validated") is True,
        "runtime_action_metrics_validated": (
            action_metrics.get("runtime_action_bounds_validated") is True
        ),
        "validation_observation_count": 36,
        "development_observation_count": 72,
        "validation_semantic": validation_semantic.to_dict(),
        "development_semantic": development_semantic.to_dict(),
        "per_task_reference_sensitivity": dict(reference_sensitivity),
        "first_interaction_count": len(interactions),
        "first_interactions": [item.to_dict() for item in interactions],
        "first_interaction_confusions": confusions,
        "semantic_error_timing": dict(timing),
        "post_grasp_failure_distribution": dict(
            cast(Mapping[str, object], _benchmark_aggregate(factor_wrapper)["phase_counts"])
        ),
        "semantic_metrics": semantic_metrics.to_dict(),
        "correct_task_retrieval_validated": semantic_metrics.full_task_top1_retrieval >= 0.70,
        "object_retrieval_validated": semantic_metrics.target_object_retrieval >= 0.80,
        "bin_retrieval_validated": bin_correct,
        "test_split_accessed": False,
        "fresh_seed_accessed": False,
        "final_schedule_accessed": False,
    }
    quality_payload = {
        "schema_version": "langmani-m43b-factor-film-quality-stage-v0",
        "quality_gate": gate.to_dict(),
        "development_quality_gate_passed": gate.development_quality_gate_passed,
        "final_benchmark_authorized": gate.final_benchmark_authorized,
        "test_split_accessed": False,
        "fresh_seed_accessed": False,
        "final_schedule_accessed": False,
        "smolvla_go": False,
    }

    results, validation_benchmark_reports = _validation_results_from_work(
        work=work, authority=authority
    )
    artifacts = _stage_artifacts(
        authority=authority,
        results=results,
        validation_benchmark_reports=validation_benchmark_reports,
        selection=selection,
        reload_record=reload_record,
        development_payload=development_payload,
        semantic_payload=semantic_payload,
        quality_payload=quality_payload,
    )
    completed = stage_and_promote_factor_film_evaluation_evidence(
        output_root=config.paths.output_root,
        training_run_root=authority.run_root,
        config=authority.evidence_config,
        stage_artifacts=artifacts,
        clean_matching_staging=config.clean_matching_staging,
    )
    del factor, references
    _collect_cuda()
    return {
        "stage": "development",
        "passed": True,
        "implementation_validated": True,
        "semantic_audit_completed": True,
        "factor_film_implementation_validated": True,
        "factor_film_fixture_training_validated": False,
        "physical_target_validated": config.device == "cuda",
        "factor_film_training_completed": True,
        "factor_film_checkpoints_complete": True,
        "factor_film_checkpoint_selected": True,
        "validation_only_selection_validated": True,
        "factor_film_reload_validated": True,
        "development_benchmark_completed": True,
        "post_grasp_analysis_completed": True,
        "first_interaction_analysis_completed": True,
        "correct_task_retrieval_validated": semantic_payload["correct_task_retrieval_validated"],
        "object_retrieval_validated": semantic_payload["object_retrieval_validated"],
        "bin_retrieval_validated": semantic_payload["bin_retrieval_validated"],
        "raw_action_metrics_validated": semantic_payload["raw_action_metrics_validated"],
        "runtime_action_metrics_validated": semantic_payload["runtime_action_metrics_validated"],
        "development_quality_gate_passed": gate.development_quality_gate_passed,
        "final_benchmark_authorized": gate.final_benchmark_authorized,
        "test_split_accessed": False,
        "fresh_seed_accessed": False,
        "final_schedule_accessed": False,
        "smolvla_go": False,
        "selected_checkpoint_fingerprint": selection.selected_checkpoint_fingerprint,
        "selected_checkpoint_step": selection.selected_checkpoint_step,
        "evidence_root": str(completed.root),
        "evidence_fingerprint": completed.evidence_fingerprint,
        "quality_gate": gate.to_dict(),
    }


def execute_factor_film_stage(config: FactorFiLMExecutionConfig, stage: str) -> dict[str, object]:
    """Execute one process-local stage; ``all`` is owned by the CLI boundary."""
    if stage == "validation":
        return execute_validation_stage(config)
    if stage == "reload":
        return execute_reload_stage(config)
    if stage == "development":
        return execute_development_stage(config)
    raise FactorFiLMExecutionError("process-local stage must be validation, reload or development")


def validate_evaluator_git(project_root: Path) -> dict[str, object]:
    """Require a clean evaluator commit without changing the frozen training identity."""
    git = inspect_git_state(project_root)
    if not git.baseline_tracked or git.dirty:
        raise FactorFiLMExecutionError("target-development evaluator requires clean tracked Git")
    return git.to_dict()


__all__ = [
    "FACTOR_FILM_EXECUTION_SCHEMA_VERSION",
    "FactorFiLMExecutionConfig",
    "FactorFiLMExecutionError",
    "FactorFiLMExecutionPaths",
    "execute_development_stage",
    "execute_factor_film_stage",
    "execute_reload_stage",
    "execute_validation_stage",
    "validate_evaluator_git",
]
