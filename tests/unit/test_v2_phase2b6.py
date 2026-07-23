from __future__ import annotations

import copy
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from langmani.v2.phase2b5 import sha256_file
from langmani.v2.phase2b6 import (
    ACTION_HORIZONS,
    EXPECTED_EPISODES,
    EXPECTED_TRANSITIONS,
    SOURCE_COMMIT,
    TARGET_BRANCH,
    TASK_IDS,
    Phase2B6ContractError,
    Phase2B6Result,
    apply_visual_shift,
    audit_primary_split_leakage,
    authorization_state,
    build_cross_skill_folds,
    build_language_template_manifest,
    build_padding_audit,
    build_primary_split_manifest,
    build_task_balance_manifest,
    classify_result,
    create_accepted_dataset_package,
    enforce_starting_point,
    load_production_spec,
    make_action_chunk,
    masked_mean_squared_error,
    source_reset_identity,
    source_trajectory_identity,
    summarize_replay_gate,
    validate_student_feature_names,
    verify_phase2b5_package,
)
from langmani.v2.phase2b6_runtime import (
    Phase2B6RuntimeError,
    freeze_production_spec,
    verify_source_bytes,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = PROJECT_ROOT / "configs" / "langmani_v2" / "phase2b6_dataset_production.yaml"
PHASE2B5_ARTIFACTS = PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2b5"


def _spec() -> dict[str, object]:
    return load_production_spec(SPEC_PATH)


def _source_rows() -> list[dict[str, object]]:
    spec = _spec()
    rows: list[dict[str, object]] = []
    for task_index, task_id in enumerate(TASK_IDS):
        for episode_id in range(1000):
            action_sha = f"sha256:{task_index + 1:02x}{episode_id:062x}"
            reset_sha = f"sha256:{task_index + 11:02x}{episode_id:062x}"
            initial_sha = f"sha256:{task_index + 21:02x}{episode_id:062x}"
            rows.append(
                {
                    "task_id": task_id,
                    "source_episode_id": episode_id,
                    "reset_identity": reset_sha,
                    "action_sha256": action_sha,
                    "initial_state_sha256": initial_sha,
                    "transition_count": 20 + task_index + episode_id % 7,
                    "source_trajectory_identity": source_trajectory_identity(
                        spec=spec,
                        task_id=task_id,
                        source_episode_id=episode_id,
                        reset_identity=reset_sha,
                        action_sha256=action_sha,
                    ),
                }
            )
    return rows


def _accepted_gates() -> dict[str, bool]:
    return {
        "official_source_hashes": True,
        "source_trajectory_accounting": True,
        "replay_outcome_agreement": True,
        "action_integrity": True,
        "finite_values": True,
        "lerobot_readback": True,
        "primary_split_leakage": True,
        "privileged_field_exclusion": True,
        "train_only_normalization": True,
        "action_padding_masks": True,
        "archive": True,
        "restore": True,
    }


def test_frozen_spec_and_source_commit_enforcement() -> None:
    spec = _spec()
    assert spec["schema_version"] == "langmani-v2-phase2b6-production-spec-v0"
    assert spec["expected_totals"] == {
        "episodes": EXPECTED_EPISODES,
        "transitions": EXPECTED_TRANSITIONS,
        "task_count": 3,
        "skill_family_count": 3,
    }
    assert tuple(spec["padding"]["candidate_horizons"]) == ACTION_HORIZONS
    enforce_starting_point(parent_commit=SOURCE_COMMIT, branch=TARGET_BRANCH)
    with pytest.raises(Phase2B6ContractError, match="must start"):
        enforce_starting_point(parent_commit="0" * 40, branch=TARGET_BRANCH)
    with pytest.raises(Phase2B6ContractError, match="must run"):
        enforce_starting_point(parent_commit=SOURCE_COMMIT, branch="codex/wrong")


def test_phase2b5_accepted_package_is_rehashed() -> None:
    report = verify_phase2b5_package(artifact_root=PHASE2B5_ARTIFACTS, spec=_spec())
    assert report["artifact_file_count"] == 27
    assert report["passed"] is True
    assert all(report["checks"].values())


def test_production_spec_lock_is_immutable(tmp_path: Path) -> None:
    root = tmp_path / "production"
    first = freeze_production_spec(spec_path=SPEC_PATH, production_root=root)
    second = freeze_production_spec(spec_path=SPEC_PATH, production_root=root)
    assert first == second
    frozen = root / "run" / "frozen_production_specification.json"
    payload = json.loads(frozen.read_text(encoding="utf-8"))
    payload["production_run_id"] = "changed"
    frozen.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Phase2B6RuntimeError, match="lock differs"):
        freeze_production_spec(spec_path=SPEC_PATH, production_root=root)


def test_selected_source_zip_hash_and_extraction_equality(tmp_path: Path) -> None:
    spec = _spec()
    source_root = tmp_path / "source"
    for task_id in TASK_IDS:
        source_h5 = b"small-hdf5-fixture"
        source_json = b'{"fixture":true}'
        archive = source_root / "sources" / f"{task_id}.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr(f"{task_id}/motionplanning/trajectory.h5", source_h5)
            bundle.writestr(f"{task_id}/motionplanning/trajectory.json", source_json)
        expanded = source_root / "expanded" / task_id / "motionplanning"
        expanded.mkdir(parents=True)
        (expanded / "trajectory.h5").write_bytes(source_h5)
        (expanded / "trajectory.json").write_bytes(source_json)
        task = spec["tasks"][task_id]
        task["source_zip_relative_path"] = f"sources/{task_id}.zip"
        task["source_zip_size_bytes"] = archive.stat().st_size
        task["source_zip_sha256"] = "sha256:" + sha256_file(archive)
    report = verify_source_bytes(source_root=source_root, spec=spec)
    assert report["passed"] is True
    assert all(task["passed"] is True for task in report["tasks"])
    changed = source_root / "expanded" / "PickCube-v1" / "motionplanning" / "trajectory.json"
    changed.write_bytes(b"changed")
    report = verify_source_bytes(source_root=source_root, spec=spec)
    assert report["passed"] is False
    assert report["tasks"][0]["checks"]["reproducible_extraction"] is False


def test_language_templates_are_disjoint_and_bounded() -> None:
    report = build_language_template_manifest(_spec())
    assert report["passed"] is True
    for task_id in TASK_IDS:
        assert report["template_counts"][task_id] == {
            "train": 6,
            "validation": 2,
            "held_out": 4,
        }
    train_texts = {row["text"] for task in report["tasks"].values() for row in task["train"]}
    held_out_texts = {row["text"] for task in report["tasks"].values() for row in task["held_out"]}
    assert train_texts.isdisjoint(held_out_texts)


def test_episode_identity_and_primary_split_materialization_are_deterministic() -> None:
    rows = _source_rows()
    first = build_primary_split_manifest(spec=_spec(), episodes=rows)
    second = build_primary_split_manifest(spec=_spec(), episodes=list(reversed(rows)))
    assert first["fingerprint"] == second["fingerprint"]
    assert first["global_counts"] == {
        "train": 2100,
        "validation": 300,
        "test_unseen_reset": 300,
        "test_unseen_task_language": 150,
        "test_visual_shift": 150,
    }
    for task_id in TASK_IDS:
        assert first["per_task_counts"][task_id] == {
            "train": 700,
            "validation": 100,
            "test_unseen_reset": 100,
            "test_unseen_task_language": 50,
            "test_visual_shift": 50,
        }
    assert first["leakage_summary"]["passed"] is True
    assert len(first["assignments"]) == EXPECTED_EPISODES
    language_banks = {row["primary_split"]: row["language_bank"] for row in first["assignments"]}
    assert language_banks["train"] == "train"
    assert language_banks["test_unseen_task_language"] == "held_out"


def test_reset_identity_is_task_local_when_official_kwargs_repeat() -> None:
    raw_identity = "sha256:" + "7" * 64
    pick = source_reset_identity(
        task_id="PickCube-v1",
        source_episode_id=17,
        reset_kwargs_sha256=raw_identity,
    )
    stack = source_reset_identity(
        task_id="StackCube-v1",
        source_episode_id=17,
        reset_kwargs_sha256=raw_identity,
    )
    assert pick != stack
    assert pick == source_reset_identity(
        task_id="PickCube-v1",
        source_episode_id=17,
        reset_kwargs_sha256=raw_identity,
    )


def test_primary_split_leakage_detects_source_and_media_overlap() -> None:
    manifest = build_primary_split_manifest(spec=_spec(), episodes=_source_rows())
    rows = copy.deepcopy(manifest["assignments"])
    train = next(row for row in rows if row["primary_split"] == "train")
    validation = next(row for row in rows if row["primary_split"] == "validation")
    validation["source_trajectory_identity"] = train["source_trajectory_identity"]
    report = audit_primary_split_leakage(rows)
    assert report["passed"] is False
    assert report["overlap_matrix"]["train__validation"]["source_trajectory_identity"] == 1
    validation["source_trajectory_identity"] = "sha256:unique"
    validation["media_namespace"] = train["media_namespace"]
    report = audit_primary_split_leakage(rows)
    assert report["overlap_matrix"]["train__validation"]["media_namespace"] == 1


def test_visual_shift_is_rgb_only_bounded_and_deterministic() -> None:
    rgb = np.full((2, 256, 256, 3), 100, dtype=np.uint8)
    shifted = apply_visual_shift(
        rgb,
        exposure_multiplier=0.9,
        rgb_channel_multipliers=(1.03, 1.0, 0.94),
    )
    assert shifted.dtype == np.uint8
    assert shifted.shape == rgb.shape
    assert not np.array_equal(shifted, rgb)
    assert np.array_equal(
        shifted,
        apply_visual_shift(
            rgb,
            exposure_multiplier=0.9,
            rgb_channel_multipliers=(1.03, 1.0, 0.94),
        ),
    )
    with pytest.raises(Phase2B6ContractError, match="uint8 RGB"):
        apply_visual_shift(
            rgb.astype(np.float32),
            exposure_multiplier=0.9,
            rgb_channel_multipliers=(1.03, 1.0, 0.94),
        )


def test_cross_skill_folds_are_metadata_only() -> None:
    splits = build_primary_split_manifest(spec=_spec(), episodes=_source_rows())
    folds = build_cross_skill_folds(spec=_spec(), split_manifest=splits)
    assert folds["passed"] is True
    assert folds["media_duplicated"] is False
    assert set(folds["folds"]) == {
        "fold_holdout_pick",
        "fold_holdout_stack",
        "fold_holdout_push",
    }
    assert folds["folds"]["fold_holdout_pick"]["train_episode_count"] == 1400
    assert folds["folds"]["fold_holdout_pick"]["test_episode_count"] == 300


def test_action_padding_and_masked_loss_exclude_padding() -> None:
    actions = np.arange(24, dtype=np.float32).reshape(3, 8)
    chunk = make_action_chunk(actions, start=1, horizon=4)
    assert chunk.action.shape == (4, 8)
    assert chunk.action_is_pad.tolist() == [False, False, True, True]
    assert np.array_equal(chunk.action[:2], actions[1:])
    assert np.count_nonzero(chunk.action[2:]) == 0
    prediction = chunk.action.copy()
    prediction[2:] = 10_000
    assert (
        masked_mean_squared_error(
            prediction[None, :], chunk.action[None, :], chunk.action_is_pad[None, :]
        )
        == 0.0
    )
    prediction[0, 0] += 2
    expected = 4.0 / (2 * 8)
    assert (
        masked_mean_squared_error(
            prediction[None, :], chunk.action[None, :], chunk.action_is_pad[None, :]
        )
        == expected
    )


def test_padding_statistics_and_task_balance_cover_all_horizons() -> None:
    splits = build_primary_split_manifest(spec=_spec(), episodes=_source_rows())
    padding = build_padding_audit(splits["assignments"])
    assert padding["passed"] is True
    assert set(padding["audits"]) == {"10", "16", "50"}
    for report in padding["audits"].values():
        assert report["padded_timesteps"] > 0
        assert report["padded_count_by_chunk_position"][0] == 0
    balance = build_task_balance_manifest(splits["assignments"])
    assert balance["passed"] is True
    assert balance["episode_counts"] == {task_id: 700 for task_id in TASK_IDS}
    natural = balance["future_sampling_recommendations"]["natural_frame_frequency"]
    assert natural["task_probabilities"]["StackCube-v1"] > 0
    assert balance["optimizer_steps"] == 0


def test_replay_gate_requires_one_hundred_percent() -> None:
    row = {
        "source_success": True,
        "replay_success": True,
        "categorical_outcome_agreement": True,
        "action_frame_alignment": True,
        "invalid_action_count": 0,
        "nonfinite_value_count": 0,
        "simulator_error_count": 0,
    }
    assert summarize_replay_gate([row] * 100)["passed"] is True
    failed = dict(row, replay_success=False, categorical_outcome_agreement=False)
    report = summarize_replay_gate([failed] + [row] * 99)
    assert report["categorical_outcome_agreement_rate"] == 0.99
    assert report["passed"] is False


def test_student_feature_allowlist_rejects_privilege() -> None:
    features = {
        "observation.images.base_camera",
        "observation.state",
        "action",
        "task",
        "task_id",
        "skill_family",
        "source_episode_identity",
        "derived_episode_identity",
        "instruction_template_id",
        "primary_split",
        "episode_index",
        "frame_index",
        "timestamp",
    }
    validate_student_feature_names(features)
    with pytest.raises(Phase2B6ContractError, match="privileged"):
        validate_student_feature_names(features | {"object_pose"})


def test_result_acceptance_and_training_authorization_are_separate() -> None:
    assert (
        classify_result(
            source_integrity_valid=True,
            replay_and_derived_integrity_valid=True,
            accepted_skill_count=3,
            archive_created=True,
            restore_valid=True,
        )
        is Phase2B6Result.RESULT_A
    )
    assert (
        classify_result(
            source_integrity_valid=True,
            replay_and_derived_integrity_valid=True,
            accepted_skill_count=3,
            archive_created=False,
            restore_valid=False,
        )
        is Phase2B6Result.RESULT_D
    )
    package = create_accepted_dataset_package(
        gates=_accepted_gates(),
        package_fields={"known_limitations": ["no policy training in Phase 2B.6"]},
    )
    assert package["episode_count"] == EXPECTED_EPISODES
    assert package["frame_count"] == EXPECTED_TRANSITIONS
    state = authorization_state(accepted=True)
    assert state["accepted_multiskill_dataset_validated"] is True
    assert state["act_baseline_training_eligible"] is True
    assert state["smolvla_push_multiskill_training_eligible"] is True
    assert state["vla_jepa_training_eligible"] is True
    assert state["eligibility_is_authorization"] is False
    assert state["act_training_authorized"] is False
    assert state["smolvla_training_authorized"] is False
    assert state["vla_jepa_training_authorized"] is False
    assert state["student_policy_training_started"] is False
    assert state["optimizer_created"] is False
    assert state["optimizer_steps"] == 0
    assert state["backward_passes"] == 0
