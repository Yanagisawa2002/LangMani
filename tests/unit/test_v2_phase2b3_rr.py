from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from langmani.v2.phase2b3_rr import (
    PHASE2B3_RR_BRANCH,
    PHASE2B3_RR_GEOMETRY_COMMIT,
    PHASE2B3_RR_MERGE_BASE,
    PHASE2B3_RR_SOURCE_COMMIT,
    Phase2B3RRContractError,
    RuntimeInputs,
    audit_historical_transcript,
    canonical_content_hash,
    classify_phase2b3_rr_result,
    compare_categorical_fields,
    compare_numeric_fields,
    first_divergence,
    hash_action_prefix,
    phase2b3_rr_authorization,
    project_mpc_runtime,
    quaternion_geodesic_error,
    require_operation_allowed,
    runtime_viability,
    sandbox_isolation_preserved,
    serialize_state_fields,
    validate_call_sequence,
    validate_repository_isolation,
    verify_phase2b3_rr_result_artifacts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _isolation() -> dict[str, object]:
    return {
        "branch": PHASE2B3_RR_BRANCH,
        "head_at_creation": PHASE2B3_RR_SOURCE_COMMIT,
        "parent_sha": PHASE2B3_RR_SOURCE_COMMIT,
        "source_upstream_sha": PHASE2B3_RR_SOURCE_COMMIT,
        "geometry_branch_sha": PHASE2B3_RR_GEOMETRY_COMMIT,
        "merge_base": PHASE2B3_RR_MERGE_BASE,
        "source_is_ancestor_of_geometry": False,
        "geometry_is_ancestor_of_source": False,
        "geometry_is_ancestor_of_new_branch": False,
        "source_worktree_clean_before_creation": True,
        "new_worktree_clean_after_creation": True,
        "geometry_branch_excluded": True,
    }


def _actions() -> list[list[float]]:
    return [[float(step + joint / 10.0) for joint in range(8)] for step in range(3)]


def _state() -> dict[str, object]:
    return {
        "robot_qpos": [0.0] * 9,
        "robot_qvel": [0.0] * 9,
        "gripper_position": [0.0, 0.0],
        "target_position": [0.1, 0.2, 0.025],
        "target_orientation": [1.0, 0.0, 0.0, 0.0],
        "target_linear_velocity": [0.0, 0.0, 0.0],
        "target_angular_velocity": [0.0, 0.0, 0.0],
        "distractors": [],
        "elapsed_steps": 0,
        "success": False,
        "failure_flags": [],
        "termination": False,
        "truncation": False,
        "contacts": [],
    }


def _transcript() -> dict[str, object]:
    actions = _actions()
    return {
        "reset_seed": 69_000,
        "reset_options": {"task": "blue_cube_left_standard"},
        "task_identity": "blue_cube_left_standard",
        "object_geometry": "cube",
        "target_direction": "left",
        "difficulty": "standard",
        "distractor_configuration": {"object": "orange_cylinder"},
        "controller_mode": "pd_joint_pos",
        "control_frequency_hz": 20,
        "physics_substeps": 25,
        "actions": actions,
        "action_dtype": "float32",
        "action_shape": [8],
        "episode_step_indices": [0, 1, 2],
        "termination_by_step": [False, False, False],
        "truncation_by_step": [False, False, False],
        "call_sequence": [
            {"operation": "env.reset"},
            {"operation": "controller.initialize"},
            {"operation": "env.step", "episode_step_index": 0},
            {"operation": "env.step", "episode_step_index": 1},
            {"operation": "env.step", "episode_step_index": 2},
        ],
        "boundary_states": [_state()],
        "action_prefix_sha256": hash_action_prefix(actions),
    }


def test_source_commit_and_geometry_exclusion_are_enforced() -> None:
    validate_repository_isolation(_isolation())

    wrong_source = {**_isolation(), "parent_sha": PHASE2B3_RR_MERGE_BASE}
    with pytest.raises(Phase2B3RRContractError, match="parent_sha"):
        validate_repository_isolation(wrong_source)

    geometry_merged = {**_isolation(), "geometry_is_ancestor_of_new_branch": True}
    with pytest.raises(Phase2B3RRContractError, match="geometry_is_ancestor"):
        validate_repository_isolation(geometry_merged)


def test_transcript_completeness_requires_native_actions_and_reset_identity() -> None:
    complete = audit_historical_transcript(_transcript())
    assert complete.complete
    assert complete.action_count == 3

    incomplete = _transcript()
    del incomplete["actions"]
    del incomplete["reset_options"]
    result = audit_historical_transcript(incomplete)
    assert not result.complete
    assert {"actions", "reset_options"} <= set(result.missing_fields)


def test_action_dtype_shape_hash_and_order_are_strict() -> None:
    actions = _actions()
    assert hash_action_prefix(actions) != hash_action_prefix(list(reversed(actions)))

    wrong = _transcript()
    wrong["action_dtype"] = "float64"
    wrong["actions"] = [[0.0] * 7]
    result = audit_historical_transcript(wrong)
    assert not result.complete
    assert any("float32" in error for error in result.errors)
    assert any("shape [8]" in error for error in result.errors)


def test_action_prefix_hash_detects_value_changes_after_float32_conversion() -> None:
    first = _actions()
    second = _actions()
    second[1][4] += 0.25
    assert hash_action_prefix(first) != hash_action_prefix(second)


def test_reset_options_are_content_bound() -> None:
    first = canonical_content_hash({"seed": 1, "options": {"difficulty": "standard"}})
    reordered = canonical_content_hash({"options": {"difficulty": "standard"}, "seed": 1})
    changed = canonical_content_hash({"seed": 1, "options": {"difficulty": "hard"}})
    assert first == reordered
    assert first != changed


def test_call_sequence_represents_every_action_in_order() -> None:
    sequence = _transcript()["call_sequence"]
    assert validate_call_sequence(sequence, expected_action_count=3).startswith("sha256:")

    reordered = list(sequence)  # type: ignore[arg-type]
    reordered[-1], reordered[-2] = reordered[-2], reordered[-1]
    with pytest.raises(Phase2B3RRContractError, match="exact order"):
        validate_call_sequence(reordered, expected_action_count=3)


def test_state_field_serialization_is_complete_finite_and_stable() -> None:
    state = _state()
    serialized = serialize_state_fields(state)
    assert serialized["target_position"] == [0.1, 0.2, 0.025]
    assert canonical_content_hash(serialized) == canonical_content_hash(serialized)

    del state["contacts"]
    with pytest.raises(Phase2B3RRContractError, match="contacts"):
        serialize_state_fields(state)

    invalid = _state()
    invalid["target_position"] = np.asarray([np.nan, 0.0, 0.0])
    with pytest.raises(Phase2B3RRContractError, match="finite"):
        serialize_state_fields(invalid)


def test_orientation_error_is_sign_invariant() -> None:
    identity = [1.0, 0.0, 0.0, 0.0]
    assert quaternion_geodesic_error(identity, [-1.0, 0.0, 0.0, 0.0]) == 0.0
    half_turn = quaternion_geodesic_error(identity, [0.0, 1.0, 0.0, 0.0])
    assert half_turn == pytest.approx(np.pi)


def test_numeric_tolerances_are_fieldwise_and_conjunctive() -> None:
    comparisons = compare_numeric_fields(
        {"target_position": [0.0, 0.0, 0.0], "qpos": [0.0, 0.0]},
        {"target_position": [0.00001, 0.0, 0.0], "qpos": [0.0, 0.01]},
        {"target_position": 0.00002, "qpos": 0.001},
    )
    assert comparisons[0].passed
    assert not comparisons[1].passed


def test_categorical_comparison_requires_100_percent() -> None:
    agreement, mismatches = compare_categorical_fields(
        {"contact": ["robot", "target"], "success": False},
        {"contact": ["robot", "distractor"], "success": False},
        ("contact", "success"),
    )
    assert agreement == 0.5
    assert mismatches == ("contact",)


def test_stepwise_first_divergence_uses_earliest_step() -> None:
    records = [
        {"step": 2, "numeric_passed": False, "categorical_passed": True, "field": "qpos"},
        {"step": 0, "numeric_passed": True, "categorical_passed": True},
        {
            "step": 1,
            "numeric_passed": True,
            "categorical_passed": False,
            "field": "contact",
        },
    ]
    divergence = first_divergence(records)
    assert divergence is not None
    assert divergence["step"] == 1
    assert first_divergence(records[:1] + [records[1]]) is not None


def test_sandbox_isolation_uses_content_not_identity() -> None:
    before = {"elapsed_steps": 4, "target": [0.1, 0.2, 0.025]}
    assert sandbox_isolation_preserved(before, dict(before))
    assert not sandbox_isolation_preserved(before, {**before, "elapsed_steps": 5})


def test_runtime_projection_and_viability_are_separate() -> None:
    projection = project_mpc_runtime(
        RuntimeInputs(
            environment_construction_seconds=0.05,
            reset_seconds=0.05,
            replay_step_seconds=0.001,
            probe_12_step_seconds=0.012,
        )
    )
    assert projection["decisions_per_episode"] == 84
    assert projection["replayed_prefix_steps_per_candidate_episode"] == 10_458
    assert projection["decision_prefix_steps"] == 100
    assert runtime_viability(projection)

    slow = project_mpc_runtime(
        RuntimeInputs(
            environment_construction_seconds=2.0,
            reset_seconds=1.0,
            replay_step_seconds=0.5,
            probe_12_step_seconds=6.0,
        )
    )
    assert not runtime_viability(slow)


@pytest.mark.parametrize(
    ("transcripts", "transition", "runtime", "expected"),
    [
        (False, False, False, "RESULT_D"),
        (True, False, False, "RESULT_C"),
        (True, True, False, "RESULT_B"),
        (True, True, True, "RESULT_A"),
    ],
)
def test_result_classification_is_exhaustive_and_fail_closed(
    transcripts: bool,
    transition: bool,
    runtime: bool,
    expected: str,
) -> None:
    assert (
        classify_phase2b3_rr_result(
            historical_transcripts_complete=transcripts,
            transition_equivalence_validated=transition,
            runtime_viability_validated=runtime,
        )
        == expected
    )


def test_authorization_is_false_even_when_pilot_is_eligible() -> None:
    accepted = phase2b3_rr_authorization("RESULT_A")
    assert accepted["mpc_pilot_eligible"] is True
    assert accepted["mpc_pilot_authorized"] is False
    assert accepted["expert_qualification_authorized"] is False
    assert accepted["data_collection_authorized"] is False
    assert accepted["smolvla_training_authorized"] is False
    assert accepted["training_started"] is False
    assert accepted["demonstration_source_validated"] is False

    rejected = phase2b3_rr_authorization("RESULT_D")
    assert not any(rejected.values())


def test_development_collection_and_training_execution_are_prevented() -> None:
    require_operation_allowed(
        "transcript_audit",
        historical_transcripts_complete=False,
    )
    with pytest.raises(Phase2B3RRContractError, match="complete original"):
        require_operation_allowed(
            "reset_replay",
            historical_transcripts_complete=False,
        )
    for operation in ("mpc_pilot", "expert_qualification", "collection", "training"):
        with pytest.raises(Phase2B3RRContractError, match="unauthorized"):
            require_operation_allowed(  # type: ignore[arg-type]
                operation,
                historical_transcripts_complete=True,
            )


def test_result_d_artifacts_verify_and_detect_tampering(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2b3_rr"
    copied = tmp_path / "phase_2b3_rr"
    shutil.copytree(source, copied)

    report = verify_phase2b3_rr_result_artifacts(copied)
    assert report["passed"] is True
    assert report["result"] == "RESULT_D"
    assert report["mpc_pilot_authorized"] is False

    manifest = json.loads((copied / "artifact_manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = copied / entry["path"]
        contents = path.read_bytes().replace(b"\r\n", b"\n")
        path.write_bytes(contents.replace(b"\n", b"\r\n"))
    translated = verify_phase2b3_rr_result_artifacts(copied)
    assert translated["passed"] is True
    assert all(translated["crlf_normalized_hash_checks"].values())

    classification = copied / "result_classification.json"
    classification.write_bytes(classification.read_bytes() + b" ")
    tampered = verify_phase2b3_rr_result_artifacts(copied)
    assert tampered["passed"] is False
    assert tampered["artifact_hash_checks"]["result_classification.json"] is False
