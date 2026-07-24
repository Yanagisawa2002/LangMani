"""Fail-closed contracts for Phase 2B.6.1-v2 StackCube replay forensics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Final, cast

import numpy as np

from langmani.v2.phase2b5 import canonical_json_sha256
from langmani.v2.phase2b6_1 import (
    CONTROL_EPISODE_IDS,
    MODE_REPETITION_BUDGETS,
    TARGET_ACTION_COUNT,
    TARGET_ACTION_SHA256,
    TARGET_EPISODE_ID,
    TARGET_RESET_IDENTITY,
    TARGET_SOURCE_TRAJECTORY_IDENTITY,
    ForensicMode,
    GateStatus,
    alignment_audit,
    gate,
    success_timing,
    validate_episode_scope,
)

SOURCE_BRANCH: Final = "codex/langmani-v2-phase2b6-1r-vulkan-recovery"
SOURCE_COMMIT: Final = "799e62cc9f7d5590116d236242c4328af1812e1c"
TARGET_BRANCH: Final = "codex/langmani-v2-phase2b6-1-v2-forensic-replay"
PRODUCER_COMMIT: Final = "73b2cd1517e7c3d2bfc08bd2898e46a3195eb1c9"
ACCEPTED_LAUNCHER_SHA256: Final = (
    "sha256:a5a3349d26f096a2a8800a6de40a7a127db8b254009d67735e968637b4f7d112"
)
PHASE2B6_ARTIFACT_FINGERPRINT: Final = (
    "sha256:a4b4ace15b3d5bd5dfb2215507abd14fcbe1a0a21dce0f875d6d91dcc676f8a0"
)
PHASE2B61_ARTIFACT_FINGERPRINT: Final = (
    "sha256:17427bdf2dda9ad4dcebd92e678b695f89b7f2fe147725a1daf5d6cf818ca826"
)
PHASE2B61R_ARTIFACT_FINGERPRINT: Final = (
    "sha256:23406ce54be296e47d0b8186fe1d5f10a714df50d800c429b43e6c78ebfba6c9"
)


class Phase2B61V2ContractError(ValueError):
    """Raised when the bounded v2 forensic protocol would be widened."""


class Phase2B61V2Result(StrEnum):
    """The only terminal classifications."""

    RESULT_A = "RESULT_A"
    RESULT_B = "RESULT_B"
    RESULT_C = "RESULT_C"
    RESULT_D = "RESULT_D"


class DeterminismClassification(StrEnum):
    """Fresh-process Mode A outcome classifications."""

    DETERMINISTIC_PASS = "deterministic_pass"
    DETERMINISTIC_FAILURE = "deterministic_failure"
    INTERMITTENT_CATEGORICAL_OUTCOME = "intermittent_categorical_outcome"
    CONTINUOUS_DRIFT_ONLY = "continuous_drift_only"


FAILURE_CATEGORIES: Final = (
    "source_identity_mismatch",
    "rendering_preflight_failure",
    "environment_construction_failure",
    "reset_failure",
    "action_contract_failure",
    "simulator_exception",
    "physical_trajectory_divergence",
    "canonical_success_failure",
    "stable_success_failure",
    "source_outcome_disagreement",
    "intermittent_physical_nondeterminism",
    "observation_generation_failure",
    "frame_action_alignment_failure",
    "state_generation_failure",
    "timestamp_failure",
    "serialization_failure",
    "writer_failure",
    "insufficient_evidence",
)

GATE_ORDER: Final = (
    "rendering_preflight_gate",
    "environment_construction_gate",
    "environment_configuration_gate",
    "action_space_contract_gate",
    "observation_configuration_gate",
    "reset_success_gate",
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

BASE_REQUIRED_GATES: Final = (
    "rendering_preflight_gate",
    "environment_construction_gate",
    "environment_configuration_gate",
    "action_space_contract_gate",
    "observation_configuration_gate",
    "reset_success_gate",
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

GATE_FAILURE_CATEGORY: Final = {
    "rendering_preflight_gate": "rendering_preflight_failure",
    "environment_construction_gate": "environment_construction_failure",
    "environment_configuration_gate": "environment_construction_failure",
    "action_space_contract_gate": "action_contract_failure",
    "observation_configuration_gate": "observation_generation_failure",
    "reset_success_gate": "reset_failure",
    "reset_identity_gate": "reset_failure",
    "action_contract_gate": "action_contract_failure",
    "step_execution_gate": "physical_trajectory_divergence",
    "simulator_exception_gate": "simulator_exception",
    "canonical_terminal_success_gate": "canonical_success_failure",
    "stable_success_gate": "stable_success_failure",
    "source_replay_outcome_agreement_gate": "source_outcome_disagreement",
    "action_count_gate": "physical_trajectory_divergence",
    "pre_action_frame_count_gate": "frame_action_alignment_failure",
    "observation_action_alignment_gate": "frame_action_alignment_failure",
    "rgb_completeness_gate": "observation_generation_failure",
    "state_completeness_gate": "state_generation_failure",
    "timestamp_monotonicity_gate": "timestamp_failure",
    "temporary_writer_gate": "writer_failure",
    "episode_serialization_gate": "serialization_failure",
}


def fingerprinted(payload: Mapping[str, object]) -> dict[str, object]:
    """Return a canonical fingerprinted JSON document."""

    result = dict(payload)
    result.pop("fingerprint", None)
    result["fingerprint"] = canonical_json_sha256(result)
    return result


def enforce_starting_point(*, branch: str, source_commit: str, source_is_ancestor: bool) -> None:
    """Require the exact source and target branch identities."""

    if branch != TARGET_BRANCH:
        raise Phase2B61V2ContractError(f"expected branch {TARGET_BRANCH}, got {branch}")
    if source_commit != SOURCE_COMMIT:
        raise Phase2B61V2ContractError(f"expected source commit {SOURCE_COMMIT}")
    if not source_is_ancestor:
        raise Phase2B61V2ContractError("source commit is not an ancestor of HEAD")


def required_gates_for_mode(mode: str) -> tuple[str, ...]:
    """Return the frozen conjunction for one replay mode."""

    parsed = ForensicMode(mode)
    if parsed is ForensicMode.PHYSICS:
        return BASE_REQUIRED_GATES
    if parsed is ForensicMode.OBSERVATION:
        return (*BASE_REQUIRED_GATES, *OBSERVATION_GATES)
    return (*BASE_REQUIRED_GATES, *OBSERVATION_GATES, *WRITER_GATES)


def first_failed_gate(sub_gates: Mapping[str, Mapping[str, object]]) -> str | None:
    """Return the earliest explicit failed gate."""

    missing = [name for name in GATE_ORDER if name not in sub_gates]
    if missing:
        raise Phase2B61V2ContractError(f"missing explicit gates: {missing}")
    return next(
        (name for name in GATE_ORDER if sub_gates[name].get("status") == GateStatus.FAILED.value),
        None,
    )


def run_passed(run: Mapping[str, object]) -> bool:
    """Recompute a run's required gate conjunction."""

    identity = cast(Mapping[str, object], run["run_identity"])
    sub_gates = cast(Mapping[str, Mapping[str, object]], run["sub_gates"])
    return all(
        sub_gates[name].get("status") == GateStatus.PASSED.value
        for name in required_gates_for_mode(str(identity["mode"]))
    )


def controls_validated(control_runs: Sequence[Mapping[str, object]]) -> bool:
    """Require one passing fresh-process Mode A run for both controls."""

    keys = sorted(
        (
            int(cast(int, cast(Mapping[str, object], run["run_identity"])["episode_id"])),
            str(cast(Mapping[str, object], run["run_identity"])["mode"]),
            int(cast(int, cast(Mapping[str, object], run["run_identity"])["repetition"])),
        )
        for run in control_runs
    )
    return keys == [(936, "A", 1), (937, "A", 1)] and all(run_passed(run) for run in control_runs)


def _terminal_pose_vector(run: Mapping[str, object]) -> np.ndarray | None:
    events = cast(Mapping[str, object], run.get("physical_event_snapshots", {}))
    final = events.get("final_step")
    if not isinstance(final, dict):
        return None
    upper = final.get("upper_cube_pose")
    lower = final.get("lower_cube_pose")
    if not isinstance(upper, list) or not isinstance(lower, list):
        return None
    return np.asarray([*upper, *lower], dtype=np.float64)


def determinism_audit(mode_a_runs: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Compare exactly three fresh-process Mode A repetitions without averaging."""

    if len(mode_a_runs) != 3:
        return fingerprinted(
            {
                "schema_version": "langmani-v2-phase2b6-1-v2-determinism-v0",
                "classification": None,
                "passed": False,
                "reason": "exactly_three_mode_a_runs_required",
                "run_count": len(mode_a_runs),
            }
        )
    outcomes = [run_passed(run) for run in mode_a_runs]
    timings = [cast(Mapping[str, object], run.get("success_timing", {})) for run in mode_a_runs]
    categorical = [bool(timing.get("final_step_success", False)) for timing in timings]
    signatures = [
        [
            (
                bool(row.get("canonical_success", False)),
                bool(row.get("terminated", False)),
                bool(row.get("truncated", False)),
            )
            for row in cast(Sequence[Mapping[str, object]], run.get("step_reports", []))
        ]
        for run in mode_a_runs
    ]
    first_divergence: int | None = None
    longest = max((len(value) for value in signatures), default=0)
    for index in range(longest):
        values = [signature[index] if index < len(signature) else None for signature in signatures]
        if len(set(values)) > 1:
            first_divergence = index
            break
    poses = [_terminal_pose_vector(run) for run in mode_a_runs]
    pose_drift = 0.0
    if all(pose is not None for pose in poses):
        arrays = cast(list[np.ndarray], poses)
        pose_drift = max(
            float(np.max(np.abs(left - right)))
            for offset, left in enumerate(arrays)
            for right in arrays[offset + 1 :]
        )
    if len(set(categorical)) > 1 or len(set(outcomes)) > 1:
        classification = DeterminismClassification.INTERMITTENT_CATEGORICAL_OUTCOME
    elif all(outcomes) and (first_divergence is not None or pose_drift > 1e-9):
        classification = DeterminismClassification.CONTINUOUS_DRIFT_ONLY
    elif all(outcomes):
        classification = DeterminismClassification.DETERMINISTIC_PASS
    else:
        classification = DeterminismClassification.DETERMINISTIC_FAILURE
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-v2-determinism-v0",
            "classification": classification.value,
            "categorical_outcomes": categorical,
            "run_gate_outcomes": outcomes,
            "first_success_steps_1_based": [
                timing.get("first_success_step_1_based") for timing in timings
            ],
            "trailing_success_durations": [
                timing.get("trailing_consecutive_success_steps") for timing in timings
            ],
            "action_counts": [len(signature) for signature in signatures],
            "first_transition_divergence_action_index": first_divergence,
            "maximum_terminal_pose_absolute_drift": pose_drift,
            "passed": classification
            in {
                DeterminismClassification.DETERMINISTIC_PASS,
                DeterminismClassification.CONTINUOUS_DRIFT_ONLY,
            },
        }
    )


def classify_result(
    *,
    source_identity_proven: bool,
    rendering_preflight_passed: bool,
    control_runs: Sequence[Mapping[str, object]],
    mode_a_runs: Sequence[Mapping[str, object]],
    mode_b_runs: Sequence[Mapping[str, object]],
    mode_c_runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Classify only the bounded evidence and keep authorization closed."""

    controls_pass = controls_validated(control_runs)
    determinism = determinism_audit(mode_a_runs)
    primary = "insufficient_evidence"
    secondary: list[str] = []
    physical_valid: bool | str | None = None
    exact_gate: str | None = None
    if not source_identity_proven:
        result = Phase2B61V2Result.RESULT_D
        primary = "source_identity_mismatch"
    elif not rendering_preflight_passed:
        result = Phase2B61V2Result.RESULT_D
        primary = "rendering_preflight_failure"
    elif not controls_pass:
        result = Phase2B61V2Result.RESULT_D
        for run in control_runs:
            if not run_passed(run):
                exact_gate = first_failed_gate(
                    cast(Mapping[str, Mapping[str, object]], run["sub_gates"])
                )
                break
        secondary = ["accepted_control_failure"]
    elif len(mode_a_runs) != 3:
        result = Phase2B61V2Result.RESULT_D
        secondary = ["mode_a_repetition_budget_incomplete"]
    elif (
        determinism["classification"]
        == DeterminismClassification.INTERMITTENT_CATEGORICAL_OUTCOME.value
    ):
        result = Phase2B61V2Result.RESULT_C
        primary = "intermittent_physical_nondeterminism"
        physical_valid = "uncertain"
    elif not all(run_passed(run) for run in mode_a_runs):
        failed = [
            first_failed_gate(cast(Mapping[str, Mapping[str, object]], run["sub_gates"]))
            for run in mode_a_runs
        ]
        if len(set(failed)) == 1:
            result = Phase2B61V2Result.RESULT_B
            exact_gate = failed[0]
            primary = GATE_FAILURE_CATEGORY.get(str(exact_gate), "insufficient_evidence")
            physical_valid = False
        else:
            result = Phase2B61V2Result.RESULT_C
            primary = "intermittent_physical_nondeterminism"
            secondary = [str(value) for value in failed if value is not None]
            physical_valid = "uncertain"
    else:
        physical_valid = True
        failed_richer = [run for run in (*mode_b_runs, *mode_c_runs) if not run_passed(run)]
        if failed_richer:
            exact_gate = first_failed_gate(
                cast(
                    Mapping[str, Mapping[str, object]],
                    failed_richer[0]["sub_gates"],
                )
            )
            primary = GATE_FAILURE_CATEGORY.get(str(exact_gate), "insufficient_evidence")
            result = Phase2B61V2Result.RESULT_A
        elif len(mode_b_runs) == 2 and len(mode_c_runs) == 1:
            result = Phase2B61V2Result.RESULT_A
            primary = "insufficient_evidence"
            secondary = ["historical_opaque_wrapper_rejection_not_reproduced_by_explicit_path"]
        else:
            result = Phase2B61V2Result.RESULT_D
            secondary = ["richer_mode_prerequisites_incomplete"]
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-v2-result-v0",
            "result": result.value,
            "primary_failure_classification": primary,
            "secondary_contributing_factors": secondary,
            "exact_first_failed_sub_gate": exact_gate,
            "official_source_episode_938_physically_valid": physical_valid,
            "source_replay_determinism_validated": determinism.get("passed", False),
            "controls_validated": controls_pass,
            "phase2b6_protocol_revision_eligible": result is Phase2B61V2Result.RESULT_A,
            "phase2b6_clean_reproduction_eligible": result is Phase2B61V2Result.RESULT_A,
            "source_episode_exclusion_review_eligible": result is Phase2B61V2Result.RESULT_B,
            "passed": result is not Phase2B61V2Result.RESULT_D,
        }
    )


def authorization_state() -> dict[str, object]:
    """Return the mandatory closed production, package, and model state."""

    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-v2-authorization-v0",
            "phase2b6_production_resume_authorized": False,
            "accepted_multiskill_dataset_validated": False,
            "accepted_dataset_package_created": False,
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
        }
    )


def eligibility_state(result: Mapping[str, object]) -> dict[str, object]:
    """Expose result-specific review eligibility separately from authorization."""

    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-v2-eligibility-v0",
            "phase2b6_protocol_revision_eligible": bool(
                result.get("phase2b6_protocol_revision_eligible", False)
            ),
            "phase2b6_clean_reproduction_eligible": bool(
                result.get("phase2b6_clean_reproduction_eligible", False)
            ),
            "source_episode_exclusion_review_eligible": bool(
                result.get("source_episode_exclusion_review_eligible", False)
            ),
            "eligibility_is_authorization": False,
            "all_model_training_eligible": False,
            "all_model_training_authorized": False,
        }
    )


def forensic_protocol() -> dict[str, object]:
    """Return the immutable execution and hard-stop protocol."""

    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-v2-protocol-v0",
            "source_branch": SOURCE_BRANCH,
            "source_commit": SOURCE_COMMIT,
            "target_branch": TARGET_BRANCH,
            "producer_commit": PRODUCER_COMMIT,
            "control_episodes": list(CONTROL_EPISODE_IDS),
            "target_episode": TARGET_EPISODE_ID,
            "target_action_count": TARGET_ACTION_COUNT,
            "target_action_sha256": TARGET_ACTION_SHA256,
            "target_reset_identity": TARGET_RESET_IDENTITY,
            "target_source_trajectory_identity": TARGET_SOURCE_TRAJECTORY_IDENTITY,
            "repetition_budgets": MODE_REPETITION_BUDGETS,
            "required_gates": {
                mode: list(required_gates_for_mode(mode)) for mode in ("A", "B", "C")
            },
            "gate_order": list(GATE_ORDER),
            "producer_success_rule": "final_step_canonical_success",
            "separate_stable_success_gate_present_in_producer": False,
            "stable_success_gate_status": "not_applicable",
            "observation_action_order": "observation[t] -> action[t]",
            "fresh_process_per_run": True,
            "accepted_launcher_sha256": ACCEPTED_LAUNCHER_SHA256,
            "default_glx_negative_control_rerun": False,
            "production_acceptance_policy_changed": False,
            "production_resume_authorized": False,
            "authorization": authorization_state(),
        }
    )


__all__ = [
    "ACCEPTED_LAUNCHER_SHA256",
    "FAILURE_CATEGORIES",
    "GATE_ORDER",
    "PHASE2B61R_ARTIFACT_FINGERPRINT",
    "PHASE2B61_ARTIFACT_FINGERPRINT",
    "PHASE2B6_ARTIFACT_FINGERPRINT",
    "SOURCE_BRANCH",
    "SOURCE_COMMIT",
    "TARGET_BRANCH",
    "DeterminismClassification",
    "Phase2B61V2ContractError",
    "Phase2B61V2Result",
    "alignment_audit",
    "authorization_state",
    "classify_result",
    "controls_validated",
    "determinism_audit",
    "eligibility_state",
    "enforce_starting_point",
    "fingerprinted",
    "first_failed_gate",
    "forensic_protocol",
    "gate",
    "required_gates_for_mode",
    "run_passed",
    "success_timing",
    "validate_episode_scope",
]
