"""CPU-safe Phase 2B data identity, archive, acceptance, and leakage tests."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pytest

from langmani.datasets.lerobot_types import FeatureContract
from langmani.v2.push_archive import (
    load_native_actions,
    state_mapping_sha256,
    validate_push_native_episode,
)
from langmani.v2.push_dataset import (
    PushCollectionConfig,
    PushDatasetContractError,
    accepted_counts,
    audit_split_leakage,
    build_collection_schedule,
    build_split_manifest,
    generation_acceptance_failures,
    quota_deficits,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs" / "langmani_v2" / "phase2b_push_collection.yaml"
FULL_SCHEDULE = build_collection_schedule(PushCollectionConfig.load(CONFIG), "full")


def _accepted_record(index: int) -> dict[str, object]:
    task = FULL_SCHEDULE[index]
    return {
        **task.to_dict(),
        "accepted": True,
        "trajectory_sha256": f"sha256:{index + 1:064x}",
        "initial_state_sha256": f"sha256:{index + 1000:064x}",
        "raw_h5_path": f"attempts/shard-{index // 16:04d}/attempts.h5",
    }


def _write_native_pair(tmp_path: Path, *, terminal: bool = True) -> tuple[Path, Path]:
    h5_path = tmp_path / "attempts.h5"
    json_path = tmp_path / "attempts.json"
    with h5py.File(h5_path, "w") as archive:
        group = archive.create_group("traj_0")
        group.create_dataset("actions", data=np.zeros((3, 8), dtype=np.float32))
        group.create_dataset("terminated", data=np.array([False, False, terminal]))
        group.create_dataset("truncated", data=np.zeros(3, dtype=np.bool_))
        group.create_dataset("success", data=np.array([False, False, terminal]))
        group.create_dataset("fail", data=np.zeros(3, dtype=np.bool_))
        observations = group.create_group("obs")
        observations.create_dataset("agent", data=np.zeros((4, 9), dtype=np.float32))
        states = group.create_group("env_states")
        actors = states.create_group("actors")
        actors.create_dataset("blue_cube", data=np.zeros((4, 13), dtype=np.float32))
        articulations = states.create_group("articulations")
        articulations.create_dataset("panda", data=np.zeros((4, 31), dtype=np.float32))
    json_path.write_text(json.dumps({"episodes": [{"episode_id": 0}]}), encoding="utf-8")
    return h5_path, json_path


def test_collection_config_and_schedule_are_frozen_and_unique() -> None:
    config = PushCollectionConfig.load(CONFIG)
    assert config.payload["accepted_expert_commit"] == ("59ca88e9f0514187252a6286ab1b8e06c4318fb4")
    pilot = build_collection_schedule(config, "pilot")
    full = build_collection_schedule(config, "full")
    assert len(pilot) == 32
    assert len(full) == 420
    assert len({item.episode_id for item in (*pilot, *full)}) == 452
    assert len({item.seed for item in (*pilot, *full)}) == 452
    pilot_grid = {
        (
            item.task_spec.target_object_id,
            item.task_spec.target_region_id,
            item.task_spec.difficulty,
        )
        for item in pilot
    }
    assert len(pilot_grid) == 16


def test_collection_config_rejects_wrong_expert(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["accepted_expert_commit"] = "rejected"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PushDatasetContractError, match="Candidate E"):
        PushCollectionConfig.load(path)


def test_generation_acceptance_is_conjunctive() -> None:
    record = {
        "expert_success": True,
        "final_evaluation": {"success": True, "fail": False},
        "actions_finite": True,
        "observations_finite": True,
        "time_contract_valid": True,
        "action_contract_valid": True,
        "task_metadata_valid": True,
        "instruction_valid": True,
        "trajectory_sha256": "sha256:" + "0" * 64,
        "initial_state_sha256": "sha256:" + "1" * 64,
    }
    assert generation_acceptance_failures(record) == ()
    unsafe = deepcopy(record)
    unsafe["final_evaluation"]["wrong_object_displaced"] = True  # type: ignore[index]
    assert generation_acceptance_failures(unsafe) == ("wrong_object_displaced",)


def test_native_archive_enforces_t_plus_one_and_float32_action_contract(
    tmp_path: Path,
) -> None:
    h5_path, json_path = _write_native_pair(tmp_path)
    result = validate_push_native_episode(h5_path, json_path, native_episode_id=0)
    assert result.action_count == 3
    assert result.time_contract_valid
    assert result.action_contract_valid
    assert result.native_final_success
    assert load_native_actions(h5_path, native_episode_id=0).shape == (3, 8)


def test_native_archive_reports_nonterminal_flush_as_time_contract_failure(
    tmp_path: Path,
) -> None:
    h5_path, json_path = _write_native_pair(tmp_path, terminal=False)
    result = validate_push_native_episode(h5_path, json_path, native_episode_id=0)
    assert not result.time_contract_valid
    assert not result.native_final_success


def test_native_archive_rejects_frame_alignment_error(tmp_path: Path) -> None:
    h5_path, json_path = _write_native_pair(tmp_path)
    with h5py.File(h5_path, "r+") as archive:
        del archive["traj_0/obs/agent"]
        archive["traj_0/obs"].create_dataset("agent", data=np.zeros((3, 9), dtype=np.float32))
    with pytest.raises(PushDatasetContractError, match="first dimension 4"):
        validate_push_native_episode(h5_path, json_path, native_episode_id=0)


def test_public_state_mapping_hash_is_deterministic() -> None:
    left = {
        "actors": {"blue_cube": np.zeros((1, 13), dtype=np.float32)},
        "articulations": {"panda": np.ones((1, 31), dtype=np.float32)},
    }
    right = {"articulations": left["articulations"], "actors": left["actors"]}
    assert state_mapping_sha256(left) == state_mapping_sha256(right)


def test_splits_are_episode_level_and_leakage_free() -> None:
    records = [_accepted_record(index) for index in range(420)]
    manifest = build_split_manifest(records)
    assert set(manifest["counts"]) == {
        "train",
        "validation",
        "test_unseen_scene",
        "test_unseen_language",
        "test_hard",
        "test_visual_shift",
    }
    audit = audit_split_leakage(records, manifest)
    assert audit["passed"] is True
    assert audit["exported_frame_file_overlap_count"] == 0


@pytest.mark.parametrize("field", ["trajectory_sha256", "initial_state_sha256"])
def test_leakage_audit_detects_duplicate_content(field: str) -> None:
    records = [_accepted_record(index) for index in range(420)]
    records[1][field] = records[0][field]
    audit = audit_split_leakage(records, build_split_manifest(records))
    assert audit["passed"] is False
    assert any(field.removesuffix("_sha256") in item for item in audit["errors"])


def test_quota_accounting_uses_only_final_accepted_records() -> None:
    config = PushCollectionConfig.load(CONFIG)
    records = [_accepted_record(index) for index in range(160)]
    records[0]["accepted"] = False
    counts = accepted_counts(records)
    assert counts["total"] == 159
    deficits = quota_deficits(config, records)
    assert deficits["standard"] > 0
    assert deficits["hard"] > 0


def test_policy_feature_schema_excludes_privileged_simulator_state() -> None:
    contract = FeatureContract()
    assert contract.policy_feature_keys == {
        "observation.images.base_camera",
        "observation.state",
        "action",
    }
    assert not any(
        token in key
        for key in contract.policy_feature_keys
        for token in ("target", "object_pose", "region", "task_id")
    )
