"""Fail-closed contracts for the bounded Phase 2B.6.1 replay forensics.

This module is deliberately policy- and simulator-independent.  It defines the
single-trajectory protocol, explicit replay sub-gates, result classification,
and the mandatory closed authorization state.  Native execution lives in
``phase2b6_1_runtime`` and is restricted to StackCube episodes 936--938.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt

from langmani.v2.phase2b5 import canonical_json_sha256, sha256_file

SOURCE_BRANCH: Final = "codex/langmani-v2-phase2b6-dataset-production"
SOURCE_COMMIT: Final = "617e222b706ac8ba40410f817b71fd4b014bf41f"
TARGET_BRANCH: Final = "codex/langmani-v2-phase2b6-1-replay-failure-forensics"
PRODUCER_COMMIT: Final = "73b2cd1517e7c3d2bfc08bd2898e46a3195eb1c9"
TASK_ID: Final = "StackCube-v1"
TARGET_EPISODE_ID: Final = 938
CONTROL_EPISODE_IDS: Final = (936, 937)
ALLOWED_EPISODE_IDS: Final = (*CONTROL_EPISODE_IDS, TARGET_EPISODE_ID)
CONTROL_MODE: Final = "pd_joint_pos"
CONTROL_FREQUENCY_HZ: Final = 20
SOURCE_REVISION: Final = "d674485bbffdd533914e52d272fdda34c0515608"
MANISKILL_SOURCE_REVISION: Final = "a4a4f9272ad64b1564035874b605ceb687b63ed8"
SOURCE_ZIP_SHA256: Final = "sha256:f9b7d34b9aa418a04aa8e4322d4dea5aa27e8ae81757f60b210c1ffc54bf9c1b"
PHASE2B5_PACKAGE_SHA256: Final = (
    "sha256:22e93d609fc4564d0bdb3e5fc7571cf8971c8a86b3cf3bdaa910108621bcbcc2"
)
PHASE2B6_ARTIFACT_MANIFEST_FINGERPRINT: Final = (
    "sha256:a4b4ace15b3d5bd5dfb2215507abd14fcbe1a0a21dce0f875d6d91dcc676f8a0"
)
TARGET_ACTION_COUNT: Final = 105
TARGET_ACTION_SHA256: Final = (
    "sha256:5fc50bc7beeb91e55b2eeccf3f016414811d314142e7a80ad6037d4f83d92eb8"
)
TARGET_RESET_IDENTITY: Final = (
    "sha256:e396b1aea2fe017ae24ab29115f5e3702fa1a37d6d34f0f9d0629f188f44410b"
)
TARGET_SOURCE_TRAJECTORY_IDENTITY: Final = (
    "sha256:f5c25c7b4f1f7f5ed0cd8b3641f7a451f758c1a47eb9aa31bf96b0081848f7a3"
)
TARGET_DERIVED_EPISODE_IDENTITY: Final = (
    "sha256:1eefa0f33e0763a63960047ef7ee5b343fd4b84f355382d486b5918953de38d1"
)


class Phase2B61ContractError(ValueError):
    """Raised when a forensic input would widen or violate the frozen scope."""


class ForensicMode(StrEnum):
    """The only permitted progressively richer replay modes."""

    PHYSICS = "A"
    OBSERVATION = "B"
    WRITER = "C"


class GateStatus(StrEnum):
    """Machine-readable status for every explicit forensic sub-gate."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"
    NOT_REACHED = "not_reached"


class Phase2B61Result(StrEnum):
    """The four terminal classifications authorized by the frozen protocol."""

    RESULT_A = "RESULT_A"
    RESULT_B = "RESULT_B"
    RESULT_C = "RESULT_C"
    RESULT_D = "RESULT_D"


class FailureClassification(StrEnum):
    """Allowed primary and secondary failure labels."""

    SOURCE_IDENTITY_MISMATCH = "source_identity_mismatch"
    RESET_FAILURE = "reset_failure"
    ACTION_CONTRACT_FAILURE = "action_contract_failure"
    SIMULATOR_EXCEPTION = "simulator_exception"
    PHYSICAL_TRAJECTORY_DIVERGENCE = "physical_trajectory_divergence"
    CANONICAL_SUCCESS_FAILURE = "canonical_success_failure"
    STABLE_SUCCESS_FAILURE = "stable_success_failure"
    SOURCE_OUTCOME_DISAGREEMENT = "source_outcome_disagreement"
    OBSERVATION_GENERATION_FAILURE = "observation_generation_failure"
    FRAME_ACTION_ALIGNMENT_FAILURE = "frame_action_alignment_failure"
    STATE_GENERATION_FAILURE = "state_generation_failure"
    TIMESTAMP_FAILURE = "timestamp_failure"
    SERIALIZATION_FAILURE = "serialization_failure"
    WRITER_FAILURE = "writer_failure"
    INTERMITTENT_PHYSICAL_NONDETERMINISM = "intermittent_physical_nondeterminism"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


GATE_ORDER: Final = (
    "reset_identity_gate",
    "action_contract_gate",
    "step_execution_gate",
    "simulator_exception_gate",
    "canonical_terminal_success_gate",
    "stable_success_gate",
    "source_replay_outcome_agreement_gate",
    "action_count_gate",
    "pre_action_frame_count_gate",
    "observation_action_alignment_gate",
    "rgb_completeness_gate",
    "state_completeness_gate",
    "timestamp_monotonicity_gate",
    "temporary_writer_gate",
    "episode_serialization_gate",
)

PHYSICAL_GATES: Final = (
    "reset_identity_gate",
    "action_contract_gate",
    "step_execution_gate",
    "simulator_exception_gate",
    "canonical_terminal_success_gate",
    "source_replay_outcome_agreement_gate",
    "action_count_gate",
)
OBSERVATION_GATES: Final = (
    "pre_action_frame_count_gate",
    "observation_action_alignment_gate",
    "rgb_completeness_gate",
    "state_completeness_gate",
    "timestamp_monotonicity_gate",
)
WRITER_GATES: Final = ("temporary_writer_gate", "episode_serialization_gate")

MODE_REPETITION_BUDGETS: Final = {
    ForensicMode.PHYSICS.value: {
        936: 1,
        937: 1,
        938: 3,
    },
    ForensicMode.OBSERVATION.value: {938: 2},
    ForensicMode.WRITER.value: {938: 1},
}


def fingerprinted(payload: Mapping[str, object]) -> dict[str, object]:
    """Return a JSON-ready document with a content-derived fingerprint."""

    result = dict(payload)
    result.pop("fingerprint", None)
    result["fingerprint"] = canonical_json_sha256(result)
    return result


def array_sha256(value: npt.NDArray[np.generic], *, dtype: np.dtype[Any] | None = None) -> str:
    """Hash one C-contiguous array under an optional canonical dtype."""

    array = np.asarray(value, dtype=dtype)
    return "sha256:" + hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def enforce_starting_point(*, source_commit: str, branch: str) -> None:
    """Reject any Phase 2B.6.1 implementation started from another identity."""

    if source_commit != SOURCE_COMMIT:
        raise Phase2B61ContractError(
            f"Phase 2B.6.1 must start at {SOURCE_COMMIT}, got {source_commit}"
        )
    if branch != TARGET_BRANCH:
        raise Phase2B61ContractError(f"Phase 2B.6.1 must run on {TARGET_BRANCH}, got {branch}")


def validate_episode_scope(*, mode: str, episode_id: int, repetition: int) -> None:
    """Enforce the exact task, episode, mode, and repetition budget."""

    try:
        parsed_mode = ForensicMode(mode)
    except ValueError as error:
        raise Phase2B61ContractError(f"unsupported forensic mode: {mode}") from error
    if episode_id not in ALLOWED_EPISODE_IDS:
        raise Phase2B61ContractError(
            f"forensic execution is limited to StackCube episodes {ALLOWED_EPISODE_IDS}"
        )
    budget = MODE_REPETITION_BUDGETS.get(parsed_mode.value, {}).get(episode_id)
    if budget is None:
        raise Phase2B61ContractError(
            f"mode {parsed_mode.value} is not authorized for episode {episode_id}"
        )
    if isinstance(repetition, bool) or repetition < 1 or repetition > budget:
        raise Phase2B61ContractError(
            f"mode {parsed_mode.value} episode {episode_id} repetition must be 1..{budget}"
        )


def gate(
    *,
    status: GateStatus,
    detail: str,
    first_failure_step: int | None = None,
    evidence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build one explicit sub-gate without collapsing unavailable evidence."""

    if first_failure_step is not None and first_failure_step < 0:
        raise Phase2B61ContractError("first failure step cannot be negative")
    return {
        "status": status.value,
        "passed": (
            True if status is GateStatus.PASSED else False if status is GateStatus.FAILED else None
        ),
        "detail": detail,
        "first_failure_step": first_failure_step,
        "evidence": dict(evidence or {}),
    }


def required_gates_for_mode(mode: str) -> tuple[str, ...]:
    """Return only gates that can fail in the selected progressive mode."""

    parsed = ForensicMode(mode)
    if parsed is ForensicMode.PHYSICS:
        return PHYSICAL_GATES
    if parsed is ForensicMode.OBSERVATION:
        return (*PHYSICAL_GATES, *OBSERVATION_GATES)
    return (*PHYSICAL_GATES, *OBSERVATION_GATES, *WRITER_GATES)


def first_failed_gate(sub_gates: Mapping[str, Mapping[str, object]]) -> str | None:
    """Return the earliest failed sub-gate under the frozen gate order."""

    missing = [name for name in GATE_ORDER if name not in sub_gates]
    if missing:
        raise Phase2B61ContractError(f"run report is missing sub-gates: {missing}")
    for name in GATE_ORDER:
        if sub_gates[name].get("status") == GateStatus.FAILED.value:
            return name
    return None


def success_timing(success_values: Sequence[bool]) -> dict[str, object]:
    """Summarize canonical success timing without changing the success rule."""

    values = tuple(bool(value) for value in success_values)
    first = next((index for index, value in enumerate(values) if value), None)
    longest = 0
    trailing = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    for value in reversed(values):
        if not value:
            break
        trailing += 1
    return {
        "step_count": len(values),
        "first_success_action_index": first,
        "first_success_step_1_based": first + 1 if first is not None else None,
        "successful_step_count": sum(values),
        "longest_consecutive_success_steps": longest,
        "trailing_consecutive_success_steps": trailing,
        "final_step_success": values[-1] if values else False,
        "producer_success_rule": "final_step_canonical_success",
        "separate_stable_success_gate_present_in_producer": False,
    }


def alignment_audit(
    *,
    source_action_count: int,
    generated_policy_frame_count: int,
    action_indices: Sequence[int],
    frame_indices: Sequence[int],
    timestamps: Sequence[float],
    terminal_diagnostic_frame_count: int,
) -> dict[str, object]:
    """Audit pre-action alignment and explicit terminal-frame exclusion."""

    expected_indices = list(range(source_action_count))
    actions = list(action_indices)
    frames = list(frame_indices)
    times = list(timestamps)
    finite_timestamps = bool(np.all(np.isfinite(np.asarray(times, dtype=np.float64))))
    monotonic = all(right > left for left, right in zip(times, times[1:], strict=False))
    exact_period = all(
        np.isclose(value, index / CONTROL_FREQUENCY_HZ, rtol=0.0, atol=1e-12)
        for index, value in enumerate(times)
    )
    return {
        "source_action_count": source_action_count,
        "generated_policy_frame_count": generated_policy_frame_count,
        "action_indices_exact": actions == expected_indices,
        "frame_indices_exact": frames == expected_indices,
        "action_frame_indices_equal": actions == frames,
        "first_observation_semantics": "observation[0] immediately precedes action[0]",
        "last_action_bearing_frame_index": frames[-1] if frames else None,
        "terminal_diagnostic_frame_count": terminal_diagnostic_frame_count,
        "terminal_frame_in_policy_data": False,
        "finite_timestamps": finite_timestamps,
        "timestamps_strictly_monotonic": monotonic if len(times) > 1 else finite_timestamps,
        "timestamps_exact_20hz": exact_period,
        "missing_frame_count": max(0, source_action_count - len(frames)),
        "extra_frame_count": max(0, len(frames) - source_action_count),
        "passed": (
            generated_policy_frame_count == source_action_count
            and actions == expected_indices
            and frames == expected_indices
            and actions == frames
            and len(times) == source_action_count
            and finite_timestamps
            and (monotonic if len(times) > 1 else True)
            and exact_period
            and terminal_diagnostic_frame_count == 1
        ),
    }


def fresh_process_identity(
    *, pid: int, process_start_time_ns: int, run_nonce: str
) -> dict[str, object]:
    """Create a non-reusable identity for one process-isolated replay."""

    if pid <= 0 or process_start_time_ns <= 0 or not run_nonce:
        raise Phase2B61ContractError("fresh-process identity fields must be populated")
    payload = {
        "pid": pid,
        "process_start_time_ns": process_start_time_ns,
        "run_nonce": run_nonce,
    }
    return {
        **payload,
        "identity": canonical_json_sha256(payload),
    }


def _gate_passed(run: Mapping[str, object], name: str) -> bool:
    sub_gates = cast(Mapping[str, Mapping[str, object]], run.get("sub_gates", {}))
    return sub_gates.get(name, {}).get("status") == GateStatus.PASSED.value


def run_passed_for_mode(run: Mapping[str, object]) -> bool:
    """Recompute one run's mode-specific conjunction."""

    mode = str(cast(Mapping[str, object], run.get("run_identity", {})).get("mode"))
    return all(_gate_passed(run, name) for name in required_gates_for_mode(mode))


def controls_validated(control_runs: Sequence[Mapping[str, object]]) -> bool:
    """Require exactly one passing Mode-A run for both accepted controls."""

    keys = [
        (
            int(cast(Any, cast(Mapping[str, object], run["run_identity"])["episode_id"])),
            str(cast(Mapping[str, object], run["run_identity"])["mode"]),
        )
        for run in control_runs
    ]
    return sorted(keys) == [
        (936, ForensicMode.PHYSICS.value),
        (937, ForensicMode.PHYSICS.value),
    ] and all(run_passed_for_mode(run) for run in control_runs)


def _physical_outcome(run: Mapping[str, object]) -> tuple[bool, str | None]:
    timing = cast(Mapping[str, object], run.get("success_timing", {}))
    return bool(timing.get("final_step_success")), first_failed_gate(
        cast(Mapping[str, Mapping[str, object]], run["sub_gates"])
    )


def _classification_for_gate(gate_name: str | None) -> FailureClassification:
    mapping: dict[str, FailureClassification] = {
        "reset_identity_gate": FailureClassification.RESET_FAILURE,
        "action_contract_gate": FailureClassification.ACTION_CONTRACT_FAILURE,
        "step_execution_gate": FailureClassification.PHYSICAL_TRAJECTORY_DIVERGENCE,
        "simulator_exception_gate": FailureClassification.SIMULATOR_EXCEPTION,
        "canonical_terminal_success_gate": FailureClassification.CANONICAL_SUCCESS_FAILURE,
        "stable_success_gate": FailureClassification.STABLE_SUCCESS_FAILURE,
        "source_replay_outcome_agreement_gate": FailureClassification.SOURCE_OUTCOME_DISAGREEMENT,
        "action_count_gate": FailureClassification.PHYSICAL_TRAJECTORY_DIVERGENCE,
        "pre_action_frame_count_gate": FailureClassification.FRAME_ACTION_ALIGNMENT_FAILURE,
        "observation_action_alignment_gate": FailureClassification.FRAME_ACTION_ALIGNMENT_FAILURE,
        "rgb_completeness_gate": FailureClassification.OBSERVATION_GENERATION_FAILURE,
        "state_completeness_gate": FailureClassification.STATE_GENERATION_FAILURE,
        "timestamp_monotonicity_gate": FailureClassification.TIMESTAMP_FAILURE,
        "temporary_writer_gate": FailureClassification.WRITER_FAILURE,
        "episode_serialization_gate": FailureClassification.SERIALIZATION_FAILURE,
    }
    if gate_name is None:
        return FailureClassification.INSUFFICIENT_EVIDENCE
    return mapping.get(gate_name, FailureClassification.INSUFFICIENT_EVIDENCE)


def classify_result(
    *,
    source_identity_proven: bool,
    control_runs: Sequence[Mapping[str, object]],
    mode_a_runs: Sequence[Mapping[str, object]],
    mode_b_runs: Sequence[Mapping[str, object]],
    mode_c_runs: Sequence[Mapping[str, object]],
    historical_producer_final_success: bool | None,
    infrastructure_failure: bool = False,
) -> dict[str, object]:
    """Classify the bounded evidence without authorizing production or training."""

    controls_pass = controls_validated(control_runs)
    if not source_identity_proven:
        result = Phase2B61Result.RESULT_D
        primary = FailureClassification.SOURCE_IDENTITY_MISMATCH
        secondary: list[str] = []
        physical_valid: bool | None = None
    elif not controls_pass:
        result = Phase2B61Result.RESULT_D
        primary = FailureClassification.INSUFFICIENT_EVIDENCE
        secondary = [
            (
                "environment_construction_failure"
                if infrastructure_failure
                else "accepted_control_failure"
            )
        ]
        physical_valid = None
    elif len(mode_a_runs) != 3:
        result = Phase2B61Result.RESULT_D
        primary = FailureClassification.INSUFFICIENT_EVIDENCE
        secondary = ["mode_a_repetition_budget_incomplete"]
        physical_valid = None
    else:
        mode_a_passes = [run_passed_for_mode(run) for run in mode_a_runs]
        outcomes = [_physical_outcome(run)[0] for run in mode_a_runs]
        failed_gates = [_physical_outcome(run)[1] for run in mode_a_runs]
        if len(set(outcomes)) > 1 or len(set(mode_a_passes)) > 1:
            result = Phase2B61Result.RESULT_C
            primary = FailureClassification.INTERMITTENT_PHYSICAL_NONDETERMINISM
            secondary = [str(value) for value in failed_gates if value is not None]
            physical_valid = None
        elif not all(mode_a_passes):
            first = failed_gates[0]
            if any(gate_name != first for gate_name in failed_gates):
                result = Phase2B61Result.RESULT_C
                primary = FailureClassification.INTERMITTENT_PHYSICAL_NONDETERMINISM
                secondary = [str(value) for value in failed_gates if value is not None]
                physical_valid = None
            else:
                result = Phase2B61Result.RESULT_B
                primary = _classification_for_gate(first)
                secondary = (
                    [FailureClassification.SOURCE_OUTCOME_DISAGREEMENT.value]
                    if primary is FailureClassification.CANONICAL_SUCCESS_FAILURE
                    else []
                )
                physical_valid = False
        elif mode_b_runs and not all(run_passed_for_mode(run) for run in mode_b_runs):
            failed_mode_b = [
                first_failed_gate(cast(Mapping[str, Mapping[str, object]], run["sub_gates"]))
                for run in mode_b_runs
                if not run_passed_for_mode(run)
            ]
            result = Phase2B61Result.RESULT_A
            primary = _classification_for_gate(failed_mode_b[0] if failed_mode_b else None)
            secondary = [str(value) for value in failed_mode_b[1:] if value is not None]
            physical_valid = True
        elif mode_c_runs and not all(run_passed_for_mode(run) for run in mode_c_runs):
            failed_mode_c = first_failed_gate(
                cast(Mapping[str, Mapping[str, object]], mode_c_runs[0]["sub_gates"])
            )
            result = Phase2B61Result.RESULT_A
            primary = _classification_for_gate(failed_mode_c)
            secondary = []
            physical_valid = True
        elif (
            len(mode_b_runs) == 2
            and len(mode_c_runs) == 1
            and all(run_passed_for_mode(run) for run in (*mode_a_runs, *mode_b_runs, *mode_c_runs))
            and historical_producer_final_success is False
        ):
            result = Phase2B61Result.RESULT_C
            primary = FailureClassification.INTERMITTENT_PHYSICAL_NONDETERMINISM
            secondary = ["historical_producer_path_failed_but_all_fresh_process_runs_passed"]
            physical_valid = None
        else:
            result = Phase2B61Result.RESULT_D
            primary = FailureClassification.INSUFFICIENT_EVIDENCE
            secondary = ["progressive_mode_evidence_incomplete"]
            physical_valid = None
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-result-classification-v0",
            "result": result.value,
            "primary_failure_classification": primary.value,
            "secondary_contributing_factors": sorted(set(secondary)),
            "source_identity_proven": source_identity_proven,
            "accepted_controls_passed": controls_pass,
            "official_source_episode_938_physically_valid": physical_valid,
            "historical_producer_final_success": historical_producer_final_success,
            "phase2b6_result_unchanged": "RESULT_C",
            "classification_complete": result is not Phase2B61Result.RESULT_D
            or primary is not FailureClassification.INSUFFICIENT_EVIDENCE,
            "passed": True,
        }
    )


def eligibility_state(classification: Mapping[str, object]) -> dict[str, object]:
    """Set only diagnostic eligibility fields permitted by the result."""

    result = Phase2B61Result(str(classification["result"]))
    primary = FailureClassification(str(classification["primary_failure_classification"]))
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-eligibility-state-v0",
            "result": result.value,
            "phase2b6_reproduction_run_eligible": result is Phase2B61Result.RESULT_A,
            "phase2b6_protocol_revision_eligible": result is Phase2B61Result.RESULT_A,
            "source_episode_exclusion_review_eligible": result is Phase2B61Result.RESULT_B,
            "source_replay_determinism_validated": (
                result is not Phase2B61Result.RESULT_D
                and not (
                    result is Phase2B61Result.RESULT_C
                    and primary is FailureClassification.INTERMITTENT_PHYSICAL_NONDETERMINISM
                )
            ),
            "eligibility_is_authorization": False,
            "passed": True,
        }
    )


def authorization_state() -> dict[str, object]:
    """Return the mandatory closed state independent of forensic outcome."""

    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-authorization-state-v0",
            "phase2b6_production_resume_authorized": False,
            "accepted_multiskill_dataset_validated": False,
            "act_training_eligible": False,
            "smolvla_training_eligible": False,
            "vla_jepa_training_eligible": False,
            "act_training_authorized": False,
            "smolvla_training_authorized": False,
            "vla_jepa_training_authorized": False,
            "student_policy_training_started": False,
            "optimizer_created": False,
            "backward_passes": 0,
            "optimizer_steps": 0,
            "accepted_dataset_package_created": False,
            "partial_phase2b6_output_promoted": False,
            "passed": True,
        }
    )


def verify_phase2b6_artifacts(*, repository_root: Path) -> dict[str, object]:
    """Rehash immutable Phase 2B.6 artifacts without executing a simulator."""

    artifact_root = repository_root / "artifacts" / "langmani_v2" / "phase_2b6"
    manifest_path = artifact_root / "artifact_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2B61ContractError("failed to read Phase 2B.6 artifact manifest") from error
    if not isinstance(manifest, dict):
        raise Phase2B61ContractError("Phase 2B.6 artifact manifest must be an object")
    if manifest.get("fingerprint") != PHASE2B6_ARTIFACT_MANIFEST_FINGERPRINT:
        raise Phase2B61ContractError("Phase 2B.6 artifact manifest fingerprint changed")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise Phase2B61ContractError("Phase 2B.6 artifact file list is malformed")
    checks: dict[str, bool] = {}
    for raw in files:
        if not isinstance(raw, dict):
            raise Phase2B61ContractError("Phase 2B.6 artifact entry is malformed")
        relative = str(raw.get("path"))
        path = artifact_root / relative
        checks[relative] = (
            path.is_file()
            and "sha256:" + sha256_file(path) == raw.get("sha256")
            and path.stat().st_size == raw.get("size_bytes")
        )
    if not checks or not all(checks.values()):
        raise Phase2B61ContractError("Phase 2B.6 immutable artifact verification failed")
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-prior-evidence-verification-v0",
            "phase2b6_result": "RESULT_C",
            "artifact_manifest_fingerprint": manifest["fingerprint"],
            "artifact_file_count": len(checks),
            "artifact_checks": checks,
            "phase2b5_package_sha256": PHASE2B5_PACKAGE_SHA256,
            "accepted_multiskill_dataset_validated": False,
            "optimizer_steps": 0,
            "passed": True,
        }
    )


__all__ = [
    "ALLOWED_EPISODE_IDS",
    "CONTROL_EPISODE_IDS",
    "FailureClassification",
    "ForensicMode",
    "GATE_ORDER",
    "GateStatus",
    "MANISKILL_SOURCE_REVISION",
    "MODE_REPETITION_BUDGETS",
    "PHASE2B5_PACKAGE_SHA256",
    "PHASE2B6_ARTIFACT_MANIFEST_FINGERPRINT",
    "PRODUCER_COMMIT",
    "Phase2B61ContractError",
    "Phase2B61Result",
    "SOURCE_BRANCH",
    "SOURCE_COMMIT",
    "SOURCE_REVISION",
    "SOURCE_ZIP_SHA256",
    "TARGET_ACTION_COUNT",
    "TARGET_ACTION_SHA256",
    "TARGET_BRANCH",
    "TARGET_DERIVED_EPISODE_IDENTITY",
    "TARGET_EPISODE_ID",
    "TARGET_RESET_IDENTITY",
    "TARGET_SOURCE_TRAJECTORY_IDENTITY",
    "TASK_ID",
    "alignment_audit",
    "array_sha256",
    "authorization_state",
    "classify_result",
    "controls_validated",
    "eligibility_state",
    "enforce_starting_point",
    "fingerprinted",
    "first_failed_gate",
    "fresh_process_identity",
    "gate",
    "required_gates_for_mode",
    "run_passed_for_mode",
    "success_timing",
    "validate_episode_scope",
    "verify_phase2b6_artifacts",
]
