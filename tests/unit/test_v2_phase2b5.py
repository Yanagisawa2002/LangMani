from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from langmani.v2.phase2b5 import (
    CANDIDATE_TASKS,
    PINNED_SOURCE_FILES,
    SOURCE_COMMIT,
    TARGET_BRANCH,
    Phase2B5ContractError,
    Phase2B5Result,
    accepted_source_package,
    aggregate_replay_gate,
    authorization_state,
    classify_result,
    compare_replay_episode,
    custom_route_closure,
    enforce_source_commit,
    padding_audit,
    select_stratified_episode_ids,
    sha256_file,
    validate_action_array,
    validate_metadata_document,
    validate_policy_frame,
    validate_split_disjointness,
    validate_transition_lengths,
)


def _metadata() -> dict[str, object]:
    return {
        "env_info": {
            "env_id": "PickCube-v1",
            "env_kwargs": {
                "obs_mode": "none",
                "control_mode": "pd_joint_pos",
                "sim_backend": "auto",
            },
            "max_episode_steps": 50,
        },
        "episodes": [
            {
                "episode_id": 0,
                "episode_seed": 4,
                "control_mode": "pd_joint_pos",
                "elapsed_steps": 3,
                "reset_kwargs": {"seed": 4, "options": {}},
                "success": True,
            }
        ],
    }


def _frame() -> dict[str, object]:
    return {
        "rgb": np.zeros((256, 256, 3), dtype=np.uint8),
        "state": np.zeros(9, dtype=np.float32),
        "action": np.zeros(8, dtype=np.float32),
        "task_instruction": CANDIDATE_TASKS["PickCube-v1"].canonical_instruction,
        "task_id": "PickCube-v1",
        "timestamp": 0.0,
    }


def test_source_commit_enforcement() -> None:
    enforce_source_commit(SOURCE_COMMIT, TARGET_BRANCH)
    with pytest.raises(Phase2B5ContractError, match="must start"):
        enforce_source_commit("0" * 40, TARGET_BRANCH)
    with pytest.raises(Phase2B5ContractError, match="branch"):
        enforce_source_commit(SOURCE_COMMIT, "codex/wrong")


def test_prior_route_closure_is_fail_closed() -> None:
    closure = custom_route_closure()
    assert closure["custom_push_expert_route_active"] is False
    assert closure["custom_push_expert_reactivation_authorized"] is False
    assert closure["f2_authorized"] is False
    assert closure["probe_c_authorized"] is False
    assert closure["ppo_f1_probe_a"]["successes"] == 0
    assert closure["custom_v2_dataset"] == {
        "attempted_episodes": 0,
        "accepted_episodes": 0,
        "frames": 0,
        "bytes": 0,
    }


def test_official_source_identity_and_file_hashing(tmp_path: Path) -> None:
    identity = PINNED_SOURCE_FILES["PickCube-v1"]
    assert identity.repository == "haosulab/ManiSkill_Demonstrations"
    assert identity.revision == "d674485bbffdd533914e52d272fdda34c0515608"
    assert identity.license == "Apache-2.0"
    source = tmp_path / "source.bin"
    source.write_bytes(b"official")
    assert sha256_file(source) == (
        "6896191a14f6c66534bac457f50996b9330cd702cb6dbaae4c08d1d213e93d98"
    )
    linked = tmp_path / "linked.bin"
    try:
        linked.symlink_to(source)
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(Phase2B5ContractError, match="ordinary file"):
        sha256_file(linked)


def test_metadata_and_reset_schema() -> None:
    episodes = validate_metadata_document(_metadata(), expected_task_id="PickCube-v1")
    assert episodes[0]["reset_kwargs"] == {"seed": 4, "options": {}}
    duplicate = _metadata()
    duplicate["episodes"] = [duplicate["episodes"][0], duplicate["episodes"][0]]
    with pytest.raises(Phase2B5ContractError, match="duplicate"):
        validate_metadata_document(duplicate, expected_task_id="PickCube-v1")
    wrong_mode = _metadata()
    wrong_mode["env_info"]["env_kwargs"]["control_mode"] = "pd_joint_delta_pos"
    with pytest.raises(Phase2B5ContractError, match="pd_joint_pos"):
        validate_metadata_document(wrong_mode, expected_task_id="PickCube-v1")


def test_action_dtype_shape_finiteness_and_bounds() -> None:
    actions = np.zeros((3, 8), dtype=np.float32)
    report = validate_action_array(actions)
    assert report["passed"] is True
    assert report["finite_value_rate"] == 1.0
    wrong_dtype = actions.astype(np.float64)
    with pytest.raises(Phase2B5ContractError, match="float32"):
        validate_action_array(wrong_dtype)
    with pytest.raises(Phase2B5ContractError, match="shape"):
        validate_action_array(np.zeros((3, 7), dtype=np.float32))
    invalid = actions.copy()
    invalid[0, 0] = np.nan
    report = validate_action_array(invalid)
    assert report["nonfinite_values"] == 1
    assert report["passed"] is False
    out_of_bounds = actions.copy()
    out_of_bounds[0, 3] = 0.0
    report = validate_action_array(out_of_bounds)
    assert report["bound_violation_values"] == 1
    assert report["passed"] is False


def test_hdf5_transition_length_contract() -> None:
    validate_transition_lengths(
        action_count=3,
        transition_lengths={
            "success": 3,
            "terminated": 3,
            "truncated": 3,
            "rewards": 3,
        },
        env_state_lengths=(4, 4, 4),
    )
    with pytest.raises(Phase2B5ContractError, match="transition/action"):
        validate_transition_lengths(
            action_count=3,
            transition_lengths={"success": 2},
            env_state_lengths=(4,),
        )
    with pytest.raises(Phase2B5ContractError, match=r"T\+1"):
        validate_transition_lengths(
            action_count=3,
            transition_lengths={"success": 3},
            env_state_lengths=(3,),
        )


def test_replay_outcome_comparison_and_frozen_gate() -> None:
    row = compare_replay_episode(
        source_success=True,
        replay_success=True,
        action_count=4,
        replayed_action_count=4,
        generated_frame_count=5,
        invalid_action_count=0,
        simulator_error_count=0,
    )
    assert row["passed"] is True
    gate = aggregate_replay_gate([row] * 100)
    assert gate["categorical_outcome_agreement_rate"] == 1.0
    assert gate["passed"] is True
    disagreement = dict(row, categorical_outcome_agreement=False, passed=False)
    assert aggregate_replay_gate([disagreement] + [row] * 99)["passed"] is True
    assert aggregate_replay_gate([disagreement] * 2 + [row] * 98)["passed"] is False
    misaligned = dict(row, action_frame_alignment=False, passed=False)
    assert aggregate_replay_gate([misaligned] + [row] * 99)["passed"] is False


def test_stratified_replay_selection_spans_lengths_and_resets() -> None:
    rows = [
        {
            "episode_id": index,
            "elapsed_steps": 10 + index,
            "reset_kwargs": {"seed": index, "options": {}},
        }
        for index in range(100)
    ]
    selected = select_stratified_episode_ids(rows, count=20)
    assert len(selected) == 20
    assert selected[0] == 0
    assert selected[-1] == 99


def test_camera_state_action_language_and_privilege_allowlist() -> None:
    validate_policy_frame(_frame())
    privileged = _frame()
    privileged["object_pose"] = np.zeros(7, dtype=np.float32)
    with pytest.raises(Phase2B5ContractError, match="allowlist"):
        validate_policy_frame(privileged)
    wrong_state = _frame()
    wrong_state["state"] = np.zeros(10, dtype=np.float32)
    with pytest.raises(Phase2B5ContractError, match=r"float32\[9\]"):
        validate_policy_frame(wrong_state)
    wrong_language = _frame()
    wrong_language["task_instruction"] = "Do something else."
    with pytest.raises(Phase2B5ContractError, match="language contract"):
        validate_policy_frame(wrong_language)


def test_action_padding_mask_statistics_and_historical_act_classification() -> None:
    audit = padding_audit([3], chunk_lengths=(2, 4))
    horizon_two = audit["audits"]["2"]
    assert horizon_two["total_chunks"] == 3
    assert horizon_two["padded_chunks"] == 1
    assert horizon_two["padded_timesteps"] == 1
    assert horizon_two["required_mask_field"] == "action_is_pad"
    assert audit["historical_act_comparison"] == "historical_act_retrain_required"


def test_split_disjointness_rejects_overlap_and_frame_level_rows() -> None:
    splits = {
        "train": [{"episode_identity": "train:0"}],
        "validation": [{"episode_identity": "validation:0"}],
        "test_unseen_reset": [{"episode_identity": "reset:0"}],
        "test_unseen_task_language": [{"episode_identity": "language:0"}],
        "test_cross_skill": [{"episode_identity": "skill:0"}],
        "test_visual_shift": [{"episode_identity": "visual:0"}],
    }
    assert validate_split_disjointness(splits)["passed"] is True
    splits["validation"] = [{"episode_identity": "train:0"}]
    assert validate_split_disjointness(splits)["passed"] is False
    splits["validation"] = [{"episode_identity": "validation:0", "frame_index": 3}]
    assert validate_split_disjointness(splits)["passed"] is False


def test_result_classification_and_accepted_package_gate() -> None:
    common = {
        "source_identity_valid": True,
        "replay_valid": True,
        "common_contract_valid": True,
        "observation_valid": True,
        "lerobot_readback_valid": True,
        "privilege_exclusion_valid": True,
        "split_design_valid": True,
    }
    assert (
        classify_result(
            **common,
            accepted_skill_families={"pick_and_place", "stacking", "planar_pushing"},
        )
        is Phase2B5Result.RESULT_A
    )
    assert (
        classify_result(
            **dict(common, replay_valid=False),
            accepted_skill_families={"pick_and_place", "stacking", "planar_pushing"},
        )
        is Phase2B5Result.RESULT_C
    )
    assert (
        classify_result(
            **common,
            accepted_skill_families={"pick_and_place", "stacking"},
        )
        is Phase2B5Result.RESULT_D
    )
    gates = {
        "source_identity": True,
        "source_schema": True,
        "common_action_contract": True,
        "replay": True,
        "observation_generation": True,
        "privilege_exclusion": True,
        "lerobot_conversion": True,
        "lerobot_readback": True,
        "split_design": True,
    }
    package = accepted_source_package(
        selected_task_ids=("PickCube-v1", "StackCube-v1", "PushCube-v1"),
        gates=gates,
        package_fields={"known_limitations": ["pilot_only"]},
    )
    assert package["contains_trained_model"] is False
    with pytest.raises(Phase2B5ContractError, match="trained-model"):
        accepted_source_package(
            selected_task_ids=("PickCube-v1", "StackCube-v1", "PushCube-v1"),
            gates=gates,
            package_fields={"checkpoint": "prohibited"},
        )


def test_training_authorization_is_always_false() -> None:
    state = authorization_state(
        official_demo_source_validated=True,
        phase2b6_dataset_production_eligible=True,
    )
    assert state["official_demo_source_validated"] is True
    assert state["phase2b6_dataset_production_eligible"] is True
    assert state["act_training_authorized"] is False
    assert state["smolvla_training_authorized"] is False
    assert state["vla_jepa_training_authorized"] is False
    assert state["student_policy_training_started"] is False
    assert state["custom_push_expert_route_active"] is False
    assert state["full_dataset_production_started"] is False
    assert json.dumps(state, sort_keys=True)
