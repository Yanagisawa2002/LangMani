from __future__ import annotations

import copy
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from langmani.v2.phase2b6_1 import (
    CONTROL_EPISODE_IDS,
    GATE_ORDER,
    PHASE2B6_ARTIFACT_MANIFEST_FINGERPRINT,
    SOURCE_COMMIT,
    TARGET_ACTION_COUNT,
    TARGET_ACTION_SHA256,
    TARGET_BRANCH,
    FailureClassification,
    GateStatus,
    Phase2B61ContractError,
    Phase2B61Result,
    alignment_audit,
    array_sha256,
    authorization_state,
    classify_result,
    controls_validated,
    enforce_starting_point,
    first_failed_gate,
    fresh_process_identity,
    gate,
    required_gates_for_mode,
    run_passed_for_mode,
    success_timing,
    validate_episode_scope,
    verify_phase2b6_artifacts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run(
    *,
    episode_id: int,
    mode: str = "A",
    failed_gate: str | None = None,
    final_success: bool = True,
) -> dict[str, object]:
    sub_gates = {
        name: gate(
            status=GateStatus.NOT_APPLICABLE,
            detail="outside selected mode",
        )
        for name in GATE_ORDER
    }
    for name in required_gates_for_mode(mode):
        sub_gates[name] = gate(status=GateStatus.PASSED, detail="fixture pass")
    sub_gates["stable_success_gate"] = gate(
        status=GateStatus.NOT_APPLICABLE,
        detail="producer has no separate stable gate",
    )
    if failed_gate is not None:
        sub_gates[failed_gate] = gate(
            status=GateStatus.FAILED,
            detail="fixture failure",
            first_failure_step=3,
        )
    return {
        "run_identity": {
            "episode_id": episode_id,
            "mode": mode,
            "repetition": 1,
        },
        "success_timing": {"final_step_success": final_success},
        "sub_gates": sub_gates,
        "passed": failed_gate is None,
    }


def test_source_commit_and_branch_enforcement() -> None:
    enforce_starting_point(source_commit=SOURCE_COMMIT, branch=TARGET_BRANCH)
    with pytest.raises(Phase2B61ContractError, match="must start"):
        enforce_starting_point(source_commit="0" * 40, branch=TARGET_BRANCH)
    with pytest.raises(Phase2B61ContractError, match="must run"):
        enforce_starting_point(source_commit=SOURCE_COMMIT, branch="codex/wrong")


def test_phase2b6_artifact_manifest_is_immutable() -> None:
    report = verify_phase2b6_artifacts(repository_root=PROJECT_ROOT)
    assert report["artifact_manifest_fingerprint"] == PHASE2B6_ARTIFACT_MANIFEST_FINGERPRINT
    assert report["artifact_file_count"] == 21
    assert report["phase2b6_result"] == "RESULT_C"
    assert report["passed"] is True


def test_source_episode_action_identity_is_exact() -> None:
    actions = np.arange(TARGET_ACTION_COUNT * 8, dtype=np.float32).reshape(TARGET_ACTION_COUNT, 8)
    assert array_sha256(actions).startswith("sha256:")
    assert len(TARGET_ACTION_SHA256) == 71
    changed = actions.copy()
    changed[-1, -1] += 1.0
    assert array_sha256(changed) != array_sha256(actions)


def test_scope_and_mode_separation_are_bounded() -> None:
    validate_episode_scope(mode="A", episode_id=936, repetition=1)
    validate_episode_scope(mode="A", episode_id=938, repetition=3)
    validate_episode_scope(mode="B", episode_id=938, repetition=2)
    validate_episode_scope(mode="C", episode_id=938, repetition=1)
    with pytest.raises(Phase2B61ContractError, match="limited"):
        validate_episode_scope(mode="A", episode_id=939, repetition=1)
    with pytest.raises(Phase2B61ContractError, match="not authorized"):
        validate_episode_scope(mode="B", episode_id=936, repetition=1)
    with pytest.raises(Phase2B61ContractError, match="1..3"):
        validate_episode_scope(mode="A", episode_id=938, repetition=4)
    assert "rgb_completeness_gate" not in required_gates_for_mode("A")
    assert "rgb_completeness_gate" in required_gates_for_mode("B")
    assert "episode_serialization_gate" not in required_gates_for_mode("B")
    assert "episode_serialization_gate" in required_gates_for_mode("C")


def test_success_timing_preserves_final_step_rule_and_stable_diagnostics() -> None:
    timing = success_timing([False, True, True, False, True, True, True])
    assert timing["first_success_action_index"] == 1
    assert timing["first_success_step_1_based"] == 2
    assert timing["successful_step_count"] == 5
    assert timing["longest_consecutive_success_steps"] == 3
    assert timing["trailing_consecutive_success_steps"] == 3
    assert timing["final_step_success"] is True
    assert timing["producer_success_rule"] == "final_step_canonical_success"
    assert timing["separate_stable_success_gate_present_in_producer"] is False


def test_observation_precedes_action_and_terminal_frame_is_excluded() -> None:
    report = alignment_audit(
        source_action_count=3,
        generated_policy_frame_count=3,
        action_indices=[0, 1, 2],
        frame_indices=[0, 1, 2],
        timestamps=[0.0, 0.05, 0.1],
        terminal_diagnostic_frame_count=1,
    )
    assert report["passed"] is True
    assert report["first_observation_semantics"].startswith("observation[0]")
    assert report["terminal_frame_in_policy_data"] is False
    broken = alignment_audit(
        source_action_count=3,
        generated_policy_frame_count=2,
        action_indices=[0, 1, 2],
        frame_indices=[0, 1],
        timestamps=[0.0, 0.05],
        terminal_diagnostic_frame_count=1,
    )
    assert broken["passed"] is False
    assert broken["missing_frame_count"] == 1


def test_fresh_process_identity_is_content_bound() -> None:
    first = fresh_process_identity(pid=10, process_start_time_ns=20, run_nonce="first")
    second = fresh_process_identity(pid=10, process_start_time_ns=20, run_nonce="second")
    assert first["identity"] != second["identity"]
    with pytest.raises(Phase2B61ContractError, match="must be populated"):
        fresh_process_identity(pid=0, process_start_time_ns=20, run_nonce="bad")


def test_explicit_gates_and_first_failure_are_not_collapsed() -> None:
    run = _run(
        episode_id=938,
        failed_gate="canonical_terminal_success_gate",
        final_success=False,
    )
    gates = cast(dict[str, dict[str, object]], run["sub_gates"])
    assert list(gates) == list(GATE_ORDER)
    assert first_failed_gate(gates) == "canonical_terminal_success_gate"
    assert run_passed_for_mode(run) is False


def test_controls_require_exactly_936_and_937() -> None:
    controls = [_run(episode_id=936), _run(episode_id=937)]
    assert controls_validated(controls) is True
    assert controls_validated([controls[0]]) is False
    duplicated = [controls[0], copy.deepcopy(controls[0])]
    assert controls_validated(duplicated) is False
    assert CONTROL_EPISODE_IDS == (936, 937)


def test_result_b_is_same_repeatable_physical_failure() -> None:
    controls = [_run(episode_id=936), _run(episode_id=937)]
    target = [
        _run(
            episode_id=938,
            failed_gate="canonical_terminal_success_gate",
            final_success=False,
        )
        for _ in range(3)
    ]
    report = classify_result(
        source_identity_proven=True,
        control_runs=controls,
        mode_a_runs=target,
        mode_b_runs=[],
        mode_c_runs=[],
        historical_producer_final_success=False,
    )
    assert report["result"] == Phase2B61Result.RESULT_B.value
    assert (
        report["primary_failure_classification"]
        == FailureClassification.CANONICAL_SUCCESS_FAILURE.value
    )
    assert report["official_source_episode_938_physically_valid"] is False


def test_result_c_is_mixed_physical_outcome() -> None:
    controls = [_run(episode_id=936), _run(episode_id=937)]
    target = [
        _run(episode_id=938),
        _run(
            episode_id=938,
            failed_gate="canonical_terminal_success_gate",
            final_success=False,
        ),
        _run(episode_id=938),
    ]
    report = classify_result(
        source_identity_proven=True,
        control_runs=controls,
        mode_a_runs=target,
        mode_b_runs=[],
        mode_c_runs=[],
        historical_producer_final_success=False,
    )
    assert report["result"] == Phase2B61Result.RESULT_C.value
    assert (
        report["primary_failure_classification"]
        == FailureClassification.INTERMITTENT_PHYSICAL_NONDETERMINISM.value
    )


def test_result_d_preserves_infrastructure_hard_stop() -> None:
    report = classify_result(
        source_identity_proven=True,
        control_runs=[],
        mode_a_runs=[],
        mode_b_runs=[],
        mode_c_runs=[],
        historical_producer_final_success=False,
        infrastructure_failure=True,
    )
    assert report["result"] == Phase2B61Result.RESULT_D.value
    assert (
        report["primary_failure_classification"]
        == FailureClassification.INSUFFICIENT_EVIDENCE.value
    )
    assert report["secondary_contributing_factors"] == ["environment_construction_failure"]


def test_result_a_localizes_observation_or_writer_layer() -> None:
    controls = [_run(episode_id=936), _run(episode_id=937)]
    mode_a = [_run(episode_id=938) for _ in range(3)]
    mode_b = [
        _run(
            episode_id=938,
            mode="B",
            failed_gate="rgb_completeness_gate",
        ),
        _run(episode_id=938, mode="B"),
    ]
    report = classify_result(
        source_identity_proven=True,
        control_runs=controls,
        mode_a_runs=mode_a,
        mode_b_runs=mode_b,
        mode_c_runs=[],
        historical_producer_final_success=False,
    )
    assert report["result"] == Phase2B61Result.RESULT_A.value
    assert (
        report["primary_failure_classification"]
        == FailureClassification.OBSERVATION_GENERATION_FAILURE.value
    )


def test_authorization_is_closed_for_every_result() -> None:
    state = authorization_state()
    assert state["phase2b6_production_resume_authorized"] is False
    assert state["accepted_multiskill_dataset_validated"] is False
    assert state["accepted_dataset_package_created"] is False
    assert state["act_training_eligible"] is False
    assert state["smolvla_training_eligible"] is False
    assert state["vla_jepa_training_eligible"] is False
    assert state["act_training_authorized"] is False
    assert state["smolvla_training_authorized"] is False
    assert state["vla_jepa_training_authorized"] is False
    assert state["student_policy_training_started"] is False
    assert state["optimizer_created"] is False
    assert state["backward_passes"] == 0
    assert state["optimizer_steps"] == 0
