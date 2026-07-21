"""Independent Phase 1 runtime-evidence gate for LangMani 2.0."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from langmani.v2.act_adapter import (
    ACT_ADAPTER_NAME,
    ACT_RUNTIME_MANIFEST_SCHEMA,
    ActAdapterConfig,
    ActAdapterError,
)
from langmani.v2.evaluator import EVALUATION_RESULT_SCHEMA, EVALUATION_SUMMARY_SCHEMA
from langmani.v2.release import V1ReleaseError, validate_v1_release
from langmani.v2.taxonomy import PICK_AND_PLACE_SKILL_ID, load_task_catalog

PHASE1_PRECONDITION_SCHEMA = "langmani-v2-phase1-precondition-verification-v0"
EVALUATION_COMPLETE_SCHEMA = "langmani-v2-evaluation-complete-v0"
EXPECTED_EVALUATION_FILES = frozenset(
    {"runtime_manifest.json", "episodes.jsonl", "summary.json", "complete.json"}
)
DISALLOWED_OUTCOMES = frozenset({"policy_failure", "environment_failure", "invalid_action"})


class Phase1VerificationError(RuntimeError):
    """Raised when claimed Phase 1 runtime evidence violates the frozen contract."""


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase1VerificationError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise Phase1VerificationError(f"{label} must be a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise Phase1VerificationError(f"cannot hash runtime artifact {path}: {error}") from error
    return digest.hexdigest()


def _read_episode_records(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise Phase1VerificationError(f"cannot read episode records {path}: {error}") from error
    if not lines or any(not line.strip() for line in lines):
        raise Phase1VerificationError("episodes.jsonl must contain non-empty JSON lines")
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise Phase1VerificationError(
                f"episode record {index} is not valid JSON: {error}"
            ) from error
        if not isinstance(value, dict):
            raise Phase1VerificationError(f"episode record {index} must be a JSON object")
        records.append(value)
    return tuple(records)


def _nonnegative_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise Phase1VerificationError(f"{label} must be a non-negative number")
    return float(value)


def _latency_observed(value: object, label: str) -> None:
    if not isinstance(value, Mapping):
        raise Phase1VerificationError(f"{label} must be an object")
    for key in ("mean", "p50", "p95", "maximum"):
        if key not in value or value[key] is None:
            raise Phase1VerificationError(f"{label}.{key} must be observed")
        _nonnegative_number(value[key], f"{label}.{key}")


@dataclass(frozen=True, slots=True)
class Phase1RuntimeValidation:
    """Verified facts from one completed three-seed Phase 1 evaluation."""

    evidence_root: str
    seeds: tuple[int, ...]
    episode_count: int
    policy_query_count: int
    environment_step_count: int
    success_count: int
    policy_identity: str
    checkpoint_identity: str
    task_identity: str

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_root": self.evidence_root,
            "seeds": list(self.seeds),
            "episode_count": self.episode_count,
            "policy_query_count": self.policy_query_count,
            "environment_step_count": self.environment_step_count,
            "success_count": self.success_count,
            "policy_identity": self.policy_identity,
            "checkpoint_identity": self.checkpoint_identity,
            "task_identity": self.task_identity,
        }


@dataclass(frozen=True, slots=True)
class Phase1PreconditionVerification:
    """Fail-closed Phase 2 precondition report."""

    v1_release_validated: bool
    phase1_runtime_evidence_validated: bool
    three_seed_schedule_validated: bool
    real_policy_inference_validated: bool
    real_environment_steps_validated: bool
    phase2_authorized: bool
    errors: tuple[str, ...]
    runtime: Phase1RuntimeValidation | None
    schema_version: str = PHASE1_PRECONDITION_SCHEMA

    @property
    def passed(self) -> bool:
        return self.phase2_authorized

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "v1_release_validated": self.v1_release_validated,
            "phase1_runtime_evidence_validated": self.phase1_runtime_evidence_validated,
            "three_seed_schedule_validated": self.three_seed_schedule_validated,
            "real_policy_inference_validated": self.real_policy_inference_validated,
            "real_environment_steps_validated": self.real_environment_steps_validated,
            "phase2_authorized": self.phase2_authorized,
            "passed": self.passed,
            "errors": list(self.errors),
            "runtime": None if self.runtime is None else self.runtime.to_dict(),
        }


def validate_phase1_runtime_evidence(
    *,
    project_root: Path,
    policy_config_path: Path,
    task_config_path: Path,
    evidence_root: Path,
    expected_seeds: Sequence[int],
) -> Phase1RuntimeValidation:
    """Recompute the Phase 1 acceptance contract without loading a policy or simulator."""

    root = project_root.resolve()
    evidence = evidence_root.resolve()
    try:
        evidence_relative = evidence.relative_to(root / "outputs").as_posix()
    except ValueError as error:
        raise Phase1VerificationError("Phase 1 evidence must be below outputs/") from error
    expected_seed_tuple = tuple(expected_seeds)
    if len(expected_seed_tuple) < 3 or len(set(expected_seed_tuple)) != len(expected_seed_tuple):
        raise Phase1VerificationError("Phase 1 requires at least three unique deterministic seeds")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in expected_seed_tuple
    ):
        raise Phase1VerificationError("expected seeds must be non-negative integers")
    if not evidence.is_dir():
        raise Phase1VerificationError(f"Phase 1 evaluation evidence is missing: {evidence}")
    names = {item.name for item in evidence.iterdir()}
    if names != EXPECTED_EVALUATION_FILES:
        raise Phase1VerificationError("Phase 1 evaluation must contain exactly the four run files")

    config = ActAdapterConfig.load(policy_config_path.resolve())
    config_relative = policy_config_path.resolve().relative_to(root).as_posix()
    run_root = (root / PurePosixPath(config.run_root)).resolve()
    checkpoint_root = run_root / PurePosixPath(config.selected_checkpoint_relative_path)
    for relative, expected in config.artifact_sha256.items():
        path = checkpoint_root / PurePosixPath(relative)
        if not path.is_file() or _sha256_file(path) != expected:
            raise Phase1VerificationError(
                f"canonical ACT artifact is missing or changed: {relative}"
            )

    task_value = _read_json_object(task_config_path.resolve(), "task catalog")
    _, tasks = load_task_catalog(task_value)
    if config.expected_task_id not in {task.canonical_task_id for task in tasks}:
        raise Phase1VerificationError("canonical ACT task is absent from the task catalog")

    runtime = _read_json_object(evidence / "runtime_manifest.json", "runtime manifest")
    summary = _read_json_object(evidence / "summary.json", "evaluation summary")
    complete = _read_json_object(evidence / "complete.json", "completion marker")
    records = _read_episode_records(evidence / "episodes.jsonl")
    identity = runtime.get("policy_identity")
    if runtime.get("schema_version") != ACT_RUNTIME_MANIFEST_SCHEMA or not isinstance(
        identity, Mapping
    ):
        raise Phase1VerificationError("runtime manifest schema or policy identity is invalid")
    if identity.get("adapter_name") != ACT_ADAPTER_NAME:
        raise Phase1VerificationError("runtime manifest did not use the canonical ACT adapter")
    if identity.get("policy_id") != config.policy_id:
        raise Phase1VerificationError("runtime policy identity disagrees with its config")
    if identity.get("checkpoint_identity") != config.expected_checkpoint_fingerprint:
        raise Phase1VerificationError("runtime checkpoint identity disagrees with its config")
    if runtime.get("repository_config") != config_relative:
        raise Phase1VerificationError("runtime manifest references a different policy config")
    if runtime.get("execution_horizon") != config.execution_horizon:
        raise Phase1VerificationError("runtime execution horizon changed")
    if runtime.get("action_shape") != [8]:
        raise Phase1VerificationError("runtime action shape changed")
    if runtime.get("task_compatibility") != [config.expected_task_id]:
        raise Phase1VerificationError("runtime task compatibility changed")

    if complete != {
        "schema_version": EVALUATION_COMPLETE_SCHEMA,
        "episode_count": len(records),
        "runtime_manifest": "runtime_manifest.json",
        "episodes": "episodes.jsonl",
        "summary": "summary.json",
    }:
        raise Phase1VerificationError("completion marker disagrees with the evaluation files")
    if summary.get("schema_version") != EVALUATION_SUMMARY_SCHEMA:
        raise Phase1VerificationError("evaluation summary schema changed")
    if len(records) != len(expected_seed_tuple) or summary.get("episode_count") != len(records):
        raise Phase1VerificationError("evaluation episode count disagrees with the seed schedule")
    if summary.get("seeds") != list(expected_seed_tuple):
        raise Phase1VerificationError("evaluation summary seed order changed")
    if summary.get("policy_identity") != config.policy_id:
        raise Phase1VerificationError("evaluation summary policy identity changed")
    if summary.get("checkpoint_identity") != config.expected_checkpoint_fingerprint:
        raise Phase1VerificationError("evaluation summary checkpoint identity changed")
    if summary.get("task_identities") != [config.expected_task_id]:
        raise Phase1VerificationError("evaluation summary task identity changed")
    if summary.get("skill_families") != [PICK_AND_PLACE_SKILL_ID]:
        raise Phase1VerificationError("evaluation summary skill family changed")

    total_queries = 0
    total_steps = 0
    successes = 0
    for index, (record, seed) in enumerate(zip(records, expected_seed_tuple, strict=True)):
        if record.get("schema_version") != EVALUATION_RESULT_SCHEMA:
            raise Phase1VerificationError(f"episode {index} result schema changed")
        expected_fields = {
            "seed": seed,
            "task_identity": config.expected_task_id,
            "skill_family": PICK_AND_PLACE_SKILL_ID,
            "policy_identity": config.policy_id,
            "checkpoint_identity": config.expected_checkpoint_fingerprint,
        }
        for field, expected in expected_fields.items():
            if record.get(field) != expected:
                raise Phase1VerificationError(f"episode {index} {field} changed")
        outcome = record.get("outcome")
        if not isinstance(outcome, str) or outcome in DISALLOWED_OUTCOMES:
            raise Phase1VerificationError(f"episode {index} did not complete a valid rollout")
        queries = record.get("policy_query_count")
        steps = record.get("episode_length")
        if isinstance(queries, bool) or not isinstance(queries, int) or queries < 1:
            raise Phase1VerificationError(f"episode {index} has no real policy query")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise Phase1VerificationError(f"episode {index} has no real environment step")
        _latency_observed(record.get("policy_inference_latency_ms"), "policy latency")
        _latency_observed(record.get("environment_step_latency_ms"), "environment latency")
        total_queries += queries
        total_steps += steps
        successes += int(record.get("success") is True)
    if summary.get("success_count") != successes:
        raise Phase1VerificationError("evaluation summary success count changed")

    return Phase1RuntimeValidation(
        evidence_root=f"outputs/{evidence_relative}",
        seeds=expected_seed_tuple,
        episode_count=len(records),
        policy_query_count=total_queries,
        environment_step_count=total_steps,
        success_count=successes,
        policy_identity=config.policy_id,
        checkpoint_identity=config.expected_checkpoint_fingerprint,
        task_identity=config.expected_task_id,
    )


def verify_phase1_preconditions(
    *,
    project_root: Path,
    release_manifest_path: Path,
    policy_config_path: Path,
    task_config_path: Path,
    evidence_root: Path,
    expected_seeds: Sequence[int],
) -> Phase1PreconditionVerification:
    """Verify release and runtime evidence as the sole Phase 2 entry gate."""

    errors: list[str] = []
    release_valid = False
    runtime: Phase1RuntimeValidation | None = None
    try:
        release_valid = validate_v1_release(
            project_root=project_root,
            manifest_path=release_manifest_path,
        ).passed
    except V1ReleaseError as error:
        errors.append(f"v1 release: {error}")
    try:
        runtime = validate_phase1_runtime_evidence(
            project_root=project_root,
            policy_config_path=policy_config_path,
            task_config_path=task_config_path,
            evidence_root=evidence_root,
            expected_seeds=expected_seeds,
        )
    except (ActAdapterError, Phase1VerificationError, ValueError) as error:
        errors.append(f"Phase 1 runtime: {error}")
    runtime_valid = runtime is not None
    three_seed_valid = runtime_valid and runtime.episode_count >= 3
    policy_valid = runtime_valid and runtime.policy_query_count >= runtime.episode_count
    steps_valid = runtime_valid and runtime.environment_step_count >= runtime.episode_count
    authorized = bool(
        release_valid and runtime_valid and three_seed_valid and policy_valid and steps_valid
    )
    return Phase1PreconditionVerification(
        v1_release_validated=release_valid,
        phase1_runtime_evidence_validated=runtime_valid,
        three_seed_schedule_validated=three_seed_valid,
        real_policy_inference_validated=policy_valid,
        real_environment_steps_validated=steps_valid,
        phase2_authorized=authorized,
        errors=tuple(errors),
        runtime=runtime,
    )


__all__ = [
    "PHASE1_PRECONDITION_SCHEMA",
    "Phase1PreconditionVerification",
    "Phase1RuntimeValidation",
    "Phase1VerificationError",
    "validate_phase1_runtime_evidence",
    "verify_phase1_preconditions",
]
