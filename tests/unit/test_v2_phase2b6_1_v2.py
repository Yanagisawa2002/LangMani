from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import cast

import pytest

from langmani.v2.phase2b5 import canonical_json_sha256
from langmani.v2.phase2b6_1 import GateStatus
from langmani.v2.phase2b6_1_v2 import (
    ACCEPTED_LAUNCHER_SHA256,
    GATE_ORDER,
    PHASE2B6_ARTIFACT_FINGERPRINT,
    PHASE2B61_ARTIFACT_FINGERPRINT,
    PHASE2B61R_ARTIFACT_FINGERPRINT,
    SOURCE_COMMIT,
    TARGET_BRANCH,
    DeterminismClassification,
    Phase2B61V2ContractError,
    Phase2B61V2Result,
    alignment_audit,
    authorization_state,
    classify_result,
    controls_validated,
    determinism_audit,
    eligibility_state,
    enforce_starting_point,
    first_failed_gate,
    forensic_protocol,
    gate,
    required_gates_for_mode,
    run_passed,
    validate_episode_scope,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run(
    *,
    episode_id: int,
    mode: str = "A",
    repetition: int = 1,
    failed_gate: str | None = None,
    final_success: bool = True,
    terminal_offset: float = 0.0,
) -> dict[str, object]:
    sub_gates = {
        name: gate(status=GateStatus.NOT_APPLICABLE, detail="outside selected mode")
        for name in GATE_ORDER
    }
    for name in required_gates_for_mode(mode):
        sub_gates[name] = gate(status=GateStatus.PASSED, detail="fixture pass")
    sub_gates["stable_success_gate"] = gate(
        status=GateStatus.NOT_APPLICABLE,
        detail="producer has no separate stable-success gate",
    )
    if failed_gate is not None:
        sub_gates[failed_gate] = gate(
            status=GateStatus.FAILED,
            detail="fixture failure",
            first_failure_step=2,
        )
    successes = [False, True, final_success]
    return {
        "run_identity": {
            "episode_id": episode_id,
            "mode": mode,
            "repetition": repetition,
            "fresh_process": {"identity": f"{episode_id}-{mode}-{repetition}"},
        },
        "step_reports": [
            {
                "step_index": index,
                "canonical_success": success,
                "terminated": False,
                "truncated": False,
            }
            for index, success in enumerate(successes)
        ],
        "success_timing": {
            "first_success_step_1_based": 2,
            "trailing_consecutive_success_steps": 1 if final_success else 0,
            "final_step_success": final_success,
        },
        "physical_event_snapshots": {
            "final_step": {
                "upper_cube_pose": [0.0, 0.0, 0.04 + terminal_offset, 1, 0, 0, 0],
                "lower_cube_pose": [0.0, 0.0, 0.02, 1, 0, 0, 0],
            }
        },
        "sub_gates": sub_gates,
        "passed": failed_gate is None,
    }


def _controls() -> list[dict[str, object]]:
    return [_run(episode_id=936), _run(episode_id=937)]


def test_source_commit_and_branch_are_exact() -> None:
    enforce_starting_point(
        branch=TARGET_BRANCH,
        source_commit=SOURCE_COMMIT,
        source_is_ancestor=True,
    )
    with pytest.raises(Phase2B61V2ContractError, match="expected source"):
        enforce_starting_point(
            branch=TARGET_BRANCH,
            source_commit="0" * 40,
            source_is_ancestor=True,
        )
    with pytest.raises(Phase2B61V2ContractError, match="expected branch"):
        enforce_starting_point(
            branch="codex/wrong",
            source_commit=SOURCE_COMMIT,
            source_is_ancestor=True,
        )


@pytest.mark.parametrize(
    ("relative", "fingerprint"),
    [
        ("phase_2b6", PHASE2B6_ARTIFACT_FINGERPRINT),
        ("phase_2b6_1", PHASE2B61_ARTIFACT_FINGERPRINT),
        ("phase_2b6_1r", PHASE2B61R_ARTIFACT_FINGERPRINT),
    ],
)
def test_prior_artifact_manifest_fingerprint_is_immutable(relative: str, fingerprint: str) -> None:
    path = PROJECT_ROOT / "artifacts" / "langmani_v2" / relative / "artifact_manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    observed = document.pop("fingerprint")
    assert observed == fingerprint
    assert canonical_json_sha256(document) == fingerprint


def test_validated_launcher_is_required_and_negative_control_is_frozen() -> None:
    protocol = forensic_protocol()
    assert protocol["accepted_launcher_sha256"] == ACCEPTED_LAUNCHER_SHA256
    assert protocol["default_glx_negative_control_rerun"] is False
    launcher = (PROJECT_ROOT / "environment" / "run_v2_phase2b6_1_v2_forensic.sh").read_text(
        encoding="utf-8"
    )
    assert "env -i" in launcher
    assert "VK_ICD_FILENAMES=/etc/vulkan/icd.d/my_nvidia_icd.json" in launcher
    assert "CUDA_VISIBLE_DEVICES=0" in launcher
    assert "DISPLAY=" not in launcher


def test_explicit_gate_order_and_mode_separation() -> None:
    assert GATE_ORDER[:3] == (
        "rendering_preflight_gate",
        "environment_construction_gate",
        "environment_configuration_gate",
    )
    assert "rgb_completeness_gate" not in required_gates_for_mode("A")
    assert "rgb_completeness_gate" in required_gates_for_mode("B")
    assert "episode_serialization_gate" not in required_gates_for_mode("B")
    assert "episode_serialization_gate" in required_gates_for_mode("C")
    validate_episode_scope(mode="A", episode_id=938, repetition=3)
    with pytest.raises(ValueError, match="1..3"):
        validate_episode_scope(mode="A", episode_id=938, repetition=4)


def test_target_action_identity_is_full_and_exact() -> None:
    protocol = forensic_protocol()
    assert protocol["target_action_count"] == 105
    assert protocol["target_action_sha256"] == (
        "sha256:5fc50bc7beeb91e55b2eeccf3f016414811d314142e7a80ad6037d4f83d92eb8"
    )
    assert protocol["target_reset_identity"] == (
        "sha256:e396b1aea2fe017ae24ab29115f5e3702fa1a37d6d34f0f9d0629f188f44410b"
    )


def test_observation_precedes_action_and_terminal_frame_is_excluded() -> None:
    exact = alignment_audit(
        source_action_count=3,
        generated_policy_frame_count=3,
        action_indices=[0, 1, 2],
        frame_indices=[0, 1, 2],
        timestamps=[0.0, 0.05, 0.1],
        terminal_diagnostic_frame_count=1,
    )
    assert exact["passed"] is True
    assert exact["terminal_frame_in_policy_data"] is False
    broken = alignment_audit(
        source_action_count=3,
        generated_policy_frame_count=2,
        action_indices=[0, 1],
        frame_indices=[0, 1],
        timestamps=[0.0, 0.05],
        terminal_diagnostic_frame_count=1,
    )
    assert broken["passed"] is False
    assert broken["missing_frame_count"] == 1


def test_first_failure_is_not_collapsed() -> None:
    run = _run(
        episode_id=938,
        failed_gate="canonical_terminal_success_gate",
        final_success=False,
    )
    gates = cast(dict[str, dict[str, object]], run["sub_gates"])
    assert first_failed_gate(gates) == "canonical_terminal_success_gate"
    assert run_passed(run) is False
    missing = dict(gates)
    del missing["reset_success_gate"]
    with pytest.raises(Phase2B61V2ContractError, match="missing explicit gates"):
        first_failed_gate(missing)


def test_controls_require_exact_936_and_937_once() -> None:
    controls = _controls()
    assert controls_validated(controls) is True
    assert controls_validated(controls[:1]) is False
    duplicate = [controls[0], copy.deepcopy(controls[0])]
    assert controls_validated(duplicate) is False
    failed = copy.deepcopy(controls)
    cast(dict[str, dict[str, object]], failed[1]["sub_gates"])[
        "canonical_terminal_success_gate"
    ] = gate(status=GateStatus.FAILED, detail="fixture")
    assert controls_validated(failed) is False


def test_determinism_classifies_pass_drift_failure_and_intermittency() -> None:
    passed = [_run(episode_id=938, repetition=index) for index in (1, 2, 3)]
    assert (
        determinism_audit(passed)["classification"]
        == DeterminismClassification.DETERMINISTIC_PASS.value
    )
    drift = copy.deepcopy(passed)
    drift[2] = _run(
        episode_id=938,
        repetition=3,
        terminal_offset=1e-6,
    )
    assert (
        determinism_audit(drift)["classification"]
        == DeterminismClassification.CONTINUOUS_DRIFT_ONLY.value
    )
    failures = [
        _run(
            episode_id=938,
            repetition=index,
            failed_gate="canonical_terminal_success_gate",
            final_success=False,
        )
        for index in (1, 2, 3)
    ]
    assert (
        determinism_audit(failures)["classification"]
        == DeterminismClassification.DETERMINISTIC_FAILURE.value
    )
    mixed = [passed[0], failures[1], passed[2]]
    assert (
        determinism_audit(mixed)["classification"]
        == DeterminismClassification.INTERMITTENT_CATEGORICAL_OUTCOME.value
    )


def test_result_b_requires_same_reproducible_physical_gate() -> None:
    mode_a = [
        _run(
            episode_id=938,
            repetition=index,
            failed_gate="canonical_terminal_success_gate",
            final_success=False,
        )
        for index in (1, 2, 3)
    ]
    result = classify_result(
        source_identity_proven=True,
        rendering_preflight_passed=True,
        control_runs=_controls(),
        mode_a_runs=mode_a,
        mode_b_runs=[],
        mode_c_runs=[],
    )
    assert result["result"] == Phase2B61V2Result.RESULT_B.value
    assert result["exact_first_failed_sub_gate"] == "canonical_terminal_success_gate"
    assert result["primary_failure_classification"] == "canonical_success_failure"
    assert result["source_episode_exclusion_review_eligible"] is True


def test_result_c_never_averages_mixed_outcomes() -> None:
    mode_a = [
        _run(episode_id=938, repetition=1),
        _run(
            episode_id=938,
            repetition=2,
            failed_gate="canonical_terminal_success_gate",
            final_success=False,
        ),
        _run(episode_id=938, repetition=3),
    ]
    result = classify_result(
        source_identity_proven=True,
        rendering_preflight_passed=True,
        control_runs=_controls(),
        mode_a_runs=mode_a,
        mode_b_runs=[],
        mode_c_runs=[],
    )
    assert result["result"] == Phase2B61V2Result.RESULT_C.value
    assert result["primary_failure_classification"] == ("intermittent_physical_nondeterminism")
    assert result["official_source_episode_938_physically_valid"] == "uncertain"


def test_result_a_localizes_observation_writer_or_opaque_wrapper() -> None:
    mode_a = [_run(episode_id=938, repetition=index) for index in (1, 2, 3)]
    failed_b = [
        _run(
            episode_id=938,
            mode="B",
            repetition=1,
            failed_gate="rgb_completeness_gate",
        )
    ]
    observed = classify_result(
        source_identity_proven=True,
        rendering_preflight_passed=True,
        control_runs=_controls(),
        mode_a_runs=mode_a,
        mode_b_runs=failed_b,
        mode_c_runs=[],
    )
    assert observed["result"] == Phase2B61V2Result.RESULT_A.value
    assert observed["primary_failure_classification"] == ("observation_generation_failure")
    complete = classify_result(
        source_identity_proven=True,
        rendering_preflight_passed=True,
        control_runs=_controls(),
        mode_a_runs=mode_a,
        mode_b_runs=[
            _run(episode_id=938, mode="B", repetition=1),
            _run(episode_id=938, mode="B", repetition=2),
        ],
        mode_c_runs=[_run(episode_id=938, mode="C", repetition=1)],
    )
    assert complete["result"] == Phase2B61V2Result.RESULT_A.value
    assert complete["primary_failure_classification"] == "insufficient_evidence"
    assert "opaque_wrapper" in str(complete["secondary_contributing_factors"])
    writer = classify_result(
        source_identity_proven=True,
        rendering_preflight_passed=True,
        control_runs=_controls(),
        mode_a_runs=mode_a,
        mode_b_runs=[
            _run(episode_id=938, mode="B", repetition=1),
            _run(episode_id=938, mode="B", repetition=2),
        ],
        mode_c_runs=[
            _run(
                episode_id=938,
                mode="C",
                repetition=1,
                failed_gate="temporary_writer_gate",
            )
        ],
    )
    assert writer["result"] == "RESULT_A"
    assert writer["primary_failure_classification"] == "writer_failure"


def test_result_d_for_source_rendering_and_control_failures() -> None:
    source = classify_result(
        source_identity_proven=False,
        rendering_preflight_passed=True,
        control_runs=[],
        mode_a_runs=[],
        mode_b_runs=[],
        mode_c_runs=[],
    )
    assert source["result"] == "RESULT_D"
    assert source["primary_failure_classification"] == "source_identity_mismatch"
    rendering = classify_result(
        source_identity_proven=True,
        rendering_preflight_passed=False,
        control_runs=[],
        mode_a_runs=[],
        mode_b_runs=[],
        mode_c_runs=[],
    )
    assert rendering["primary_failure_classification"] == ("rendering_preflight_failure")
    controls = classify_result(
        source_identity_proven=True,
        rendering_preflight_passed=True,
        control_runs=[],
        mode_a_runs=[],
        mode_b_runs=[],
        mode_c_runs=[],
    )
    assert controls["secondary_contributing_factors"] == ["accepted_control_failure"]


def test_stable_success_is_diagnostic_not_a_new_acceptance_gate() -> None:
    protocol = forensic_protocol()
    assert protocol["producer_success_rule"] == "final_step_canonical_success"
    assert protocol["separate_stable_success_gate_present_in_producer"] is False
    assert protocol["stable_success_gate_status"] == "not_applicable"
    assert "stable_success_gate" not in required_gates_for_mode("A")


def test_authorization_and_model_activity_remain_closed() -> None:
    state = authorization_state()
    assert state["phase2b6_production_resume_authorized"] is False
    assert state["accepted_multiskill_dataset_validated"] is False
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
    eligible = eligibility_state(
        {
            "phase2b6_protocol_revision_eligible": True,
            "phase2b6_clean_reproduction_eligible": True,
            "source_episode_exclusion_review_eligible": False,
        }
    )
    assert eligible["phase2b6_clean_reproduction_eligible"] is True
    assert eligible["all_model_training_eligible"] is False
    assert eligible["all_model_training_authorized"] is False
