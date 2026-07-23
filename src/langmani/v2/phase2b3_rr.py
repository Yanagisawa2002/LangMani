"""Fail-closed contracts for Phase 2B.3.1 reset-replay feasibility.

This module deliberately contains no simulator runner, expert, collection, or
training entrypoint.  It validates the historical transcript prerequisite and
provides pure comparison/reporting helpers for a separately authorized replay
study.  The current evidence closes before simulator execution because the
required original native action prefixes are unavailable.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

PHASE2B3_RR_SOURCE_COMMIT = "cc31ced97e66220626ef39374b3e3acf84ccd463"
PHASE2B3_RR_GEOMETRY_COMMIT = "ae6aae49c61d96c68de434e08dc1167461a29543"
PHASE2B3_RR_MERGE_BASE = "8476a72b3ce7504e7fef0a08114c6c567da77ba7"
PHASE2B3_RR_BRANCH = "codex/langmani-v2-phase2b3-reset-replay-equivalence"
PHASE2B3_RR_ACTION_DTYPE = "float32"
PHASE2B3_RR_ACTION_SHAPE = (8,)

Phase2B3RRResult = Literal["RESULT_A", "RESULT_B", "RESULT_C", "RESULT_D"]
Phase2B3RROperation = Literal[
    "transcript_audit",
    "reset_replay",
    "mpc_pilot",
    "expert_qualification",
    "collection",
    "training",
]

_REQUIRED_TRANSCRIPT_FIELDS = (
    "reset_seed",
    "reset_options",
    "task_identity",
    "object_geometry",
    "target_direction",
    "difficulty",
    "distractor_configuration",
    "controller_mode",
    "control_frequency_hz",
    "physics_substeps",
    "actions",
    "action_dtype",
    "action_shape",
    "episode_step_indices",
    "termination_by_step",
    "truncation_by_step",
    "call_sequence",
    "boundary_states",
    "action_prefix_sha256",
)

_REQUIRED_STATE_FIELDS = (
    "robot_qpos",
    "robot_qvel",
    "gripper_position",
    "target_position",
    "target_orientation",
    "target_linear_velocity",
    "target_angular_velocity",
    "distractors",
    "elapsed_steps",
    "success",
    "failure_flags",
    "termination",
    "truncation",
    "contacts",
)

_FORBIDDEN_OPERATIONS = frozenset(
    {
        "mpc_pilot",
        "expert_qualification",
        "collection",
        "training",
    }
)


class Phase2B3RRContractError(ValueError):
    """Raised when the bounded reset-replay contract would be violated."""


@dataclass(frozen=True, slots=True)
class TranscriptAudit:
    """Validation result for one immutable historical execution transcript."""

    complete: bool
    missing_fields: tuple[str, ...]
    errors: tuple[str, ...]
    action_count: int
    computed_action_prefix_sha256: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "complete": self.complete,
            "missing_fields": list(self.missing_fields),
            "errors": list(self.errors),
            "action_count": self.action_count,
            "computed_action_prefix_sha256": self.computed_action_prefix_sha256,
        }


@dataclass(frozen=True, slots=True)
class NumericFieldComparison:
    """Maximum absolute error for one continuous state field."""

    field: str
    maximum_absolute_error: float
    tolerance: float
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "field": self.field,
            "maximum_absolute_error": self.maximum_absolute_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class RuntimeInputs:
    """Measured inputs used only for an explicitly labeled MPC cost projection."""

    environment_construction_seconds: float
    reset_seconds: float
    replay_step_seconds: float
    probe_12_step_seconds: float
    p95_multiplier: float = 1.25

    def __post_init__(self) -> None:
        for field in (
            "environment_construction_seconds",
            "reset_seconds",
            "replay_step_seconds",
            "probe_12_step_seconds",
            "p95_multiplier",
        ):
            value = float(getattr(self, field))
            if not math.isfinite(value) or value < 0.0:
                raise Phase2B3RRContractError(f"{field} must be finite and non-negative")
        if self.p95_multiplier < 1.0:
            raise Phase2B3RRContractError("p95_multiplier cannot be below 1")


def validate_repository_isolation(audit: Mapping[str, object]) -> None:
    """Require the exact source lineage and explicit geometry exclusion."""

    expected = {
        "branch": PHASE2B3_RR_BRANCH,
        "head_at_creation": PHASE2B3_RR_SOURCE_COMMIT,
        "parent_sha": PHASE2B3_RR_SOURCE_COMMIT,
        "source_upstream_sha": PHASE2B3_RR_SOURCE_COMMIT,
        "geometry_branch_sha": PHASE2B3_RR_GEOMETRY_COMMIT,
        "merge_base": PHASE2B3_RR_MERGE_BASE,
    }
    mismatches = [field for field, value in expected.items() if audit.get(field) != value]
    required_false = (
        "source_is_ancestor_of_geometry",
        "geometry_is_ancestor_of_source",
        "geometry_is_ancestor_of_new_branch",
    )
    mismatches.extend(field for field in required_false if audit.get(field) is not False)
    if audit.get("source_worktree_clean_before_creation") is not True:
        mismatches.append("source_worktree_clean_before_creation")
    if audit.get("new_worktree_clean_after_creation") is not True:
        mismatches.append("new_worktree_clean_after_creation")
    if audit.get("geometry_branch_excluded") is not True:
        mismatches.append("geometry_branch_excluded")
    if mismatches:
        raise Phase2B3RRContractError(
            "repository isolation mismatch: " + ", ".join(sorted(set(mismatches)))
        )


def hash_action_prefix(actions: Sequence[Sequence[float]]) -> str:
    """Hash exact row-major ``float32[N, 8]`` native action content and order."""

    array = _native_action_array(actions)
    header = b"langmani-v2-pd-joint-pos-float32-8-v0\0"
    count = array.shape[0].to_bytes(8, byteorder="little", signed=False)
    digest = hashlib.sha256(header + count + array.tobytes(order="C")).hexdigest()
    return f"sha256:{digest}"


def audit_historical_transcript(transcript: Mapping[str, object]) -> TranscriptAudit:
    """Check whether one transcript can legally support reset-replay execution."""

    missing = tuple(field for field in _REQUIRED_TRANSCRIPT_FIELDS if field not in transcript)
    errors: list[str] = []
    actions_value = transcript.get("actions")
    action_count = 0
    computed_hash: str | None = None

    if actions_value is not None:
        try:
            actions = _sequence_of_sequences(actions_value, "actions")
            action_count = len(actions)
            computed_hash = hash_action_prefix(actions)
        except Phase2B3RRContractError as error:
            errors.append(str(error))

    if transcript.get("action_dtype") != PHASE2B3_RR_ACTION_DTYPE:
        errors.append("action_dtype must be exact float32")
    if transcript.get("action_shape") != list(PHASE2B3_RR_ACTION_SHAPE):
        errors.append("action_shape must be exact [8]")
    if transcript.get("controller_mode") != "pd_joint_pos":
        errors.append("controller_mode must be pd_joint_pos")

    reset_seed = transcript.get("reset_seed")
    if isinstance(reset_seed, bool) or not isinstance(reset_seed, int):
        errors.append("reset_seed must be an integer")
    if not isinstance(transcript.get("reset_options"), Mapping):
        errors.append("reset_options must be an object")

    frequency = transcript.get("control_frequency_hz")
    if isinstance(frequency, bool) or not isinstance(frequency, (int, float)):
        errors.append("control_frequency_hz must be numeric")
    elif not math.isfinite(float(frequency)) or float(frequency) <= 0.0:
        errors.append("control_frequency_hz must be finite and positive")

    substeps = transcript.get("physics_substeps")
    if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
        errors.append("physics_substeps must be a positive integer")

    if action_count:
        expected_indices = list(range(action_count))
        if transcript.get("episode_step_indices") != expected_indices:
            errors.append("episode_step_indices must preserve exact action order")
        for field in ("termination_by_step", "truncation_by_step"):
            values = transcript.get(field)
            if not isinstance(values, list) or len(values) != action_count:
                errors.append(f"{field} must contain one value per action")
        if computed_hash is not None and transcript.get("action_prefix_sha256") != computed_hash:
            errors.append("action_prefix_sha256 does not match exact float32 action bytes")
    elif actions_value is not None:
        errors.append("actions must contain at least one executed native action")

    call_sequence = transcript.get("call_sequence")
    if call_sequence is not None:
        try:
            validate_call_sequence(call_sequence, expected_action_count=action_count)
        except Phase2B3RRContractError as error:
            errors.append(str(error))

    boundary_states = transcript.get("boundary_states")
    if boundary_states is not None:
        if not isinstance(boundary_states, list) or not boundary_states:
            errors.append("boundary_states must be a non-empty list")
        elif any(not _state_fields_complete(value) for value in boundary_states):
            errors.append("each boundary state must contain every required comparison field")

    return TranscriptAudit(
        complete=not missing and not errors,
        missing_fields=missing,
        errors=tuple(errors),
        action_count=action_count,
        computed_action_prefix_sha256=computed_hash,
    )


def validate_call_sequence(value: object, *, expected_action_count: int) -> str:
    """Validate and fingerprint an ordered environment-call transcript."""

    if not isinstance(value, list) or not value:
        raise Phase2B3RRContractError("call_sequence must be a non-empty list")
    operations: list[str] = []
    step_indices: list[int] = []
    for index, record in enumerate(value):
        if not isinstance(record, Mapping):
            raise Phase2B3RRContractError(f"call_sequence[{index}] must be an object")
        operation = record.get("operation")
        if not isinstance(operation, str) or not operation:
            raise Phase2B3RRContractError(
                f"call_sequence[{index}].operation must be a non-empty string"
            )
        operations.append(operation)
        if operation == "env.step":
            step = record.get("episode_step_index")
            if isinstance(step, bool) or not isinstance(step, int):
                raise Phase2B3RRContractError(
                    f"call_sequence[{index}] env.step requires an integer episode_step_index"
                )
            step_indices.append(step)
    if "env.reset" not in operations:
        raise Phase2B3RRContractError("call_sequence must include env.reset")
    if "controller.initialize" not in operations:
        raise Phase2B3RRContractError("call_sequence must include controller.initialize")
    if operations.index("env.reset") > operations.index("controller.initialize"):
        raise Phase2B3RRContractError("env.reset must precede controller.initialize")
    if step_indices != list(range(expected_action_count)):
        raise Phase2B3RRContractError(
            "call_sequence env.step records must match every action in exact order"
        )
    return canonical_content_hash(value)


def canonical_content_hash(value: object) -> str:
    """Return a stable hash for finite JSON-compatible evidence."""

    normalized = _jsonable(value)
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def serialize_state_fields(state: Mapping[str, object]) -> dict[str, object]:
    """Serialize the public comparison surface while rejecting incomplete state."""

    missing = [field for field in _REQUIRED_STATE_FIELDS if field not in state]
    if missing:
        raise Phase2B3RRContractError("state fields are incomplete: " + ", ".join(missing))
    return {key: _jsonable(value) for key, value in sorted(state.items())}


def quaternion_geodesic_error(reference: Sequence[float], candidate: Sequence[float]) -> float:
    """Return sign-invariant quaternion geodesic error in radians."""

    first = np.asarray(reference, dtype=np.float64)
    second = np.asarray(candidate, dtype=np.float64)
    if first.shape != (4,) or second.shape != (4,):
        raise Phase2B3RRContractError("orientation quaternions must each have shape [4]")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise Phase2B3RRContractError("orientation quaternions must be finite")
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 0.0 or second_norm <= 0.0:
        raise Phase2B3RRContractError("orientation quaternions must have non-zero norm")
    dot = abs(float(np.dot(first / first_norm, second / second_norm)))
    return 2.0 * math.acos(float(np.clip(dot, 0.0, 1.0)))


def compare_numeric_fields(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
    tolerances: Mapping[str, float],
) -> tuple[NumericFieldComparison, ...]:
    """Compare continuous fields without averaging away a critical failure."""

    comparisons: list[NumericFieldComparison] = []
    for field, tolerance_value in tolerances.items():
        tolerance = float(tolerance_value)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise Phase2B3RRContractError(f"{field} tolerance must be finite and non-negative")
        if field not in reference or field not in candidate:
            raise Phase2B3RRContractError(f"numeric field {field!r} is missing")
        first = np.asarray(reference[field], dtype=np.float64)
        second = np.asarray(candidate[field], dtype=np.float64)
        if first.shape != second.shape:
            raise Phase2B3RRContractError(f"numeric field {field!r} shape differs")
        if not np.isfinite(first).all() or not np.isfinite(second).all():
            raise Phase2B3RRContractError(f"numeric field {field!r} contains non-finite data")
        error = float(np.max(np.abs(first - second))) if first.size else 0.0
        comparisons.append(
            NumericFieldComparison(
                field=field,
                maximum_absolute_error=error,
                tolerance=tolerance,
                passed=error <= tolerance,
            )
        )
    return tuple(comparisons)


def compare_categorical_fields(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
    fields: Sequence[str],
) -> tuple[float, tuple[str, ...]]:
    """Require exact equality for contacts, flags, and all other categorical state."""

    if not fields:
        raise Phase2B3RRContractError("categorical field list cannot be empty")
    mismatches = tuple(
        field
        for field in fields
        if field not in reference or field not in candidate or reference[field] != candidate[field]
    )
    return (len(fields) - len(mismatches)) / len(fields), mismatches


def first_divergence(records: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
    """Return the earliest failed stepwise comparison, if any."""

    indexed: list[tuple[int, Mapping[str, object]]] = []
    for record in records:
        step = record.get("step")
        if isinstance(step, bool) or not isinstance(step, int):
            raise Phase2B3RRContractError("stepwise comparisons require integer step indices")
        indexed.append((step, record))
    for _, record in sorted(indexed, key=lambda item: item[0]):
        if record.get("numeric_passed") is not True or record.get("categorical_passed") is not True:
            return dict(record)
    return None


def sandbox_isolation_preserved(before: object, after: object) -> bool:
    """Compare immutable content identities for a non-participating environment."""

    return canonical_content_hash(before) == canonical_content_hash(after)


def project_mpc_runtime(
    measured: RuntimeInputs,
    *,
    candidates_per_decision: int = 6,
    candidate_horizon_steps: int = 12,
    executed_prefix_steps: int = 3,
    episode_step_budget: int = 250,
    decision_prefix_steps: int = 100,
) -> dict[str, float | int]:
    """Project, but never label as measured, a bounded future MPC workload."""

    for label, value in (
        ("candidates_per_decision", candidates_per_decision),
        ("candidate_horizon_steps", candidate_horizon_steps),
        ("executed_prefix_steps", executed_prefix_steps),
        ("episode_step_budget", episode_step_budget),
        ("decision_prefix_steps", decision_prefix_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise Phase2B3RRContractError(f"{label} must be a positive integer")
    decisions = math.ceil(episode_step_budget / executed_prefix_steps)
    one_decision = candidates_per_decision * (
        measured.environment_construction_seconds
        + measured.reset_seconds
        + measured.replay_step_seconds * decision_prefix_steps
        + measured.probe_12_step_seconds
    )
    replayed_prefix_steps = sum(
        min(index * executed_prefix_steps, episode_step_budget) for index in range(decisions)
    )
    one_episode = candidates_per_decision * (
        measured.environment_construction_seconds
        + decisions * (measured.reset_seconds + measured.probe_12_step_seconds)
        + measured.replay_step_seconds * replayed_prefix_steps
    )
    p95_episode = one_episode * measured.p95_multiplier
    return {
        "projection_only": 1,
        "decisions_per_episode": decisions,
        "decision_prefix_steps": decision_prefix_steps,
        "replayed_prefix_steps_per_candidate_episode": replayed_prefix_steps,
        "projected_decision_seconds": one_decision,
        "projected_episode_seconds": one_episode,
        "projected_p95_episode_seconds": p95_episode,
        "projected_100_episode_hours": one_episode * 100 / 3600.0,
        "projected_400_episode_hours": one_episode * 400 / 3600.0,
    }


def runtime_viability(
    projection: Mapping[str, float | int],
    *,
    maximum_p95_episode_seconds: float = 600.0,
    maximum_400_episode_hours: float = 72.0,
) -> bool:
    """Apply the predeclared offline-production runtime limits conjunctively."""

    return (
        float(projection["projected_p95_episode_seconds"]) <= maximum_p95_episode_seconds
        and float(projection["projected_400_episode_hours"]) <= maximum_400_episode_hours
    )


def classify_phase2b3_rr_result(
    *,
    historical_transcripts_complete: bool,
    transition_equivalence_validated: bool,
    runtime_viability_validated: bool,
) -> Phase2B3RRResult:
    """Classify the feasibility study without interpreting unavailable data physically."""

    if not historical_transcripts_complete:
        return "RESULT_D"
    if not transition_equivalence_validated:
        return "RESULT_C"
    if not runtime_viability_validated:
        return "RESULT_B"
    return "RESULT_A"


def phase2b3_rr_authorization(result: Phase2B3RRResult) -> dict[str, bool]:
    """Return the mandatory fail-closed authorization surface."""

    return {
        "mpc_pilot_eligible": result == "RESULT_A",
        "mpc_pilot_authorized": False,
        "expert_qualification_authorized": False,
        "data_collection_authorized": False,
        "smolvla_training_authorized": False,
        "training_started": False,
        "demonstration_source_validated": False,
    }


def require_operation_allowed(
    operation: Phase2B3RROperation,
    *,
    historical_transcripts_complete: bool,
) -> None:
    """Prevent scope expansion and simulator entry after a transcript hard stop."""

    if operation in _FORBIDDEN_OPERATIONS:
        raise Phase2B3RRContractError(f"{operation} is unauthorized in Phase 2B.3.1-RR")
    if operation == "reset_replay" and not historical_transcripts_complete:
        raise Phase2B3RRContractError(
            "reset_replay is blocked: complete original historical action prefixes are unavailable"
        )


def verify_phase2b3_rr_result_artifacts(root: Path) -> dict[str, object]:
    """Rehash and validate the compact Result D package without simulator execution."""

    root = root.resolve()
    manifest = _load_json_object(root / "artifact_manifest.json")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise Phase2B3RRContractError("artifact manifest files must be a list")

    hash_checks: dict[str, bool] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise Phase2B3RRContractError("artifact manifest entries must be objects")
        relative = entry.get("path")
        expected = entry.get("sha256")
        if not isinstance(relative, str) or not relative or Path(relative).name != relative:
            raise Phase2B3RRContractError("artifact manifest paths must be safe basenames")
        if not isinstance(expected, str) or len(expected) != 64:
            raise Phase2B3RRContractError(f"invalid SHA-256 for artifact {relative!r}")
        path = (root / relative).resolve()
        if path.parent != root:
            raise Phase2B3RRContractError("artifact path escapes Result D root")
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
        hash_checks[relative] = actual == expected

    isolation = _load_json_object(root / "repository_isolation_audit.json")
    isolation_valid = True
    try:
        validate_repository_isolation(isolation)
    except Phase2B3RRContractError:
        isolation_valid = False

    prior = _load_json_object(root / "prior_evidence_verification.json")
    transcript = _load_json_object(root / "transcript_availability_audit.json")
    classification = _load_json_object(root / "result_classification.json")
    authorization = _load_json_object(root / "authorization_state.json")
    selected = _load_json_object(root / "selected_episodes_and_boundaries.json")
    reset = _load_json_object(root / "reset_identity_results.json")
    replay = _load_json_object(root / "prefix_replay_stepwise_comparisons.json")
    probe = _load_json_object(root / "fixed_probe_results.json")
    runtime = _load_json_object(root / "runtime_measurements.json")
    remote = _load_json_object(root / "remote_execution_audit.json")

    expected_authorization = phase2b3_rr_authorization("RESULT_D")
    checks = {
        "manifest_schema_valid": manifest.get("schema_version")
        == "langmani-v2-phase2b3-rr-artifact-manifest-v0",
        "artifact_count_matches": manifest.get("file_count") == len(entries),
        "all_artifact_hashes_valid": bool(hash_checks) and all(hash_checks.values()),
        "repository_isolation_valid": isolation_valid,
        "mpc_result_c_preserved": prior.get("mpc_result") == "RESULT_C"
        and prior.get("mpc_git_blob_hashes_passed") is True,
        "geometry_result_preserved": prior.get("geometry_result") == "COMPLETE_REJECTED"
        and prior.get("geometry_git_blob_hashes_passed") is True,
        "historical_transcript_unavailable": transcript.get("complete_original_transcripts_found")
        is False
        and transcript.get("actual_native_action_arrays_found") == 0,
        "selection_not_run": selected.get("status") == "not_run_historical_prefix_unavailable",
        "reset_identity_not_run": reset.get("status") == "not_run_historical_prefix_unavailable",
        "prefix_replay_not_run": replay.get("status") == "not_run_historical_prefix_unavailable",
        "fixed_probe_not_run": probe.get("status") == "not_run_historical_prefix_unavailable",
        "runtime_not_measured": runtime.get("status")
        == "not_measured_historical_prefix_unavailable",
        "result_d_classified": classification.get("result") == "RESULT_D"
        and classification.get("label") == "HISTORICAL_PREFIX_UNAVAILABLE"
        and classification.get("physical_nondeterminism_conclusion") is False,
        "authorization_fail_closed": all(
            authorization.get(key) is value for key, value in expected_authorization.items()
        ),
        "no_remote_execution": remote.get("simulator_started") is False
        and remote.get("gpu_workload_started") is False,
    }
    return {
        "schema_version": "langmani-v2-phase2b3-rr-independent-verification-v0",
        "passed": all(checks.values()),
        "result": "RESULT_D",
        "checks": checks,
        "artifact_hash_checks": hash_checks,
        **expected_authorization,
    }


def _native_action_array(actions: Sequence[Sequence[float]]) -> np.ndarray:
    rows: list[list[float]] = []
    for index, action in enumerate(actions):
        if isinstance(action, (str, bytes)) or not isinstance(action, Sequence):
            raise Phase2B3RRContractError(f"actions[{index}] must be a numeric sequence")
        if len(action) != PHASE2B3_RR_ACTION_SHAPE[0]:
            raise Phase2B3RRContractError(f"actions[{index}] must have exact shape [8]")
        row: list[float] = []
        for value in action:
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise Phase2B3RRContractError(f"actions[{index}] contains a non-numeric value")
            row.append(float(value))
        rows.append(row)
    array = np.asarray(rows, dtype=np.float32)
    if array.ndim != 2 or array.shape[1:] != PHASE2B3_RR_ACTION_SHAPE:
        raise Phase2B3RRContractError("actions must have exact shape [N, 8]")
    if not np.isfinite(array).all():
        raise Phase2B3RRContractError("actions contain non-finite values")
    return np.ascontiguousarray(array)


def _sequence_of_sequences(value: object, label: str) -> Sequence[Sequence[float]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise Phase2B3RRContractError(f"{label} must be a sequence")
    return value  # type: ignore[return-value]


def _state_fields_complete(value: object) -> bool:
    return isinstance(value, Mapping) and all(field in value for field in _REQUIRED_STATE_FIELDS)


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Phase2B3RRContractError("evidence values must be finite")
        return value
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        if not np.isfinite(value).all():
            raise Phase2B3RRContractError("evidence arrays must be finite")
        return value.tolist()
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise Phase2B3RRContractError("evidence mapping keys must be strings")
            result[key] = _jsonable(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_jsonable(item) for item in value]
    raise Phase2B3RRContractError(f"unsupported evidence value type: {type(value).__name__}")


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Phase2B3RRContractError(f"{path} must contain one JSON object")
    return payload


__all__ = [
    "NumericFieldComparison",
    "PHASE2B3_RR_ACTION_DTYPE",
    "PHASE2B3_RR_ACTION_SHAPE",
    "PHASE2B3_RR_BRANCH",
    "PHASE2B3_RR_GEOMETRY_COMMIT",
    "PHASE2B3_RR_MERGE_BASE",
    "PHASE2B3_RR_SOURCE_COMMIT",
    "Phase2B3RRContractError",
    "Phase2B3RROperation",
    "Phase2B3RRResult",
    "RuntimeInputs",
    "TranscriptAudit",
    "audit_historical_transcript",
    "canonical_content_hash",
    "classify_phase2b3_rr_result",
    "compare_categorical_fields",
    "compare_numeric_fields",
    "first_divergence",
    "hash_action_prefix",
    "phase2b3_rr_authorization",
    "project_mpc_runtime",
    "quaternion_geodesic_error",
    "require_operation_allowed",
    "runtime_viability",
    "sandbox_isolation_preserved",
    "serialize_state_fields",
    "validate_call_sequence",
    "validate_repository_isolation",
    "verify_phase2b3_rr_result_artifacts",
]
