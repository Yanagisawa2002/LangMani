"""CPU-only tests for strict M3A native archive construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

import langmani.datasets.archive as archive_module
from langmani.datasets.archive import (
    ARCHIVE_SCHEMA_VERSION,
    CANONICAL_TASK_SPECS,
    ArchiveValidationError,
    EpisodeSelection,
    build_source_shard,
    create_counterfactual_scene_group_bundle,
    sha256_file,
    validate_native_archive_pair,
)
from langmani.datasets.identity import (
    COLLECTION_SCHEMA_VERSION,
    expert_config_fingerprint,
    stable_accepted_scene_group_id,
    stable_attempt_id,
    stable_candidate_scene_id,
    stable_collection_run_id,
    stable_raw_trajectory_id,
    stable_scheduled_episode_id,
    stable_source_shard_id,
)
from langmani.environments.specs import (
    TaskSpec,
    canonical_instruction,
    stable_scene_id,
    stable_task_id,
)
from langmani.experts.types import ExpertConfig

_SUCCESS_EVALUATION = {
    "target_in_target_bin": True,
    "target_in_wrong_bin": False,
    "wrong_object_in_target_bin": False,
    "target_is_grasped": False,
    "target_is_static": True,
    "target_off_table": False,
    "success": True,
    "fail": False,
}
_EXPERT_FINGERPRINT = expert_config_fingerprint(ExpertConfig())
_COLLECTION_RUN_ID = stable_collection_run_id({"archive_unit_test": "v1"})


def _initial_scene_state() -> dict[str, Any]:
    return {
        "object_poses": {
            "red_cube": [-0.10, -0.10, 0.02, 1.0, 0.0, 0.0, 0.0],
            "green_cube": [-0.09, 0.00, 0.02, 1.0, 0.0, 0.0, 0.0],
            "blue_cube": [-0.08, 0.10, 0.02, 1.0, 0.0, 0.0, 0.0],
        },
        "bin_poses": {
            "left_bin": [0.08, 0.18, 0.004, 1.0, 0.0, 0.0, 0.0],
            "right_bin": [0.08, -0.18, 0.004, 1.0, 0.0, 0.0, 0.0],
        },
        "panda_qpos": [0.0] * 9,
    }


def _candidate_scene_id(scene_seed: int) -> str:
    return stable_candidate_scene_id(
        environment_id="LangMani-PickPlaceByInstruction-v0",
        scene_seed=scene_seed,
        scene_id=stable_scene_id(scene_seed),
        expert_fingerprint=_EXPERT_FINGERPRINT,
        control_mode="pd_joint_pos",
    )


def _episode_ids(scene_seed: int, task_spec: TaskSpec) -> tuple[str, str, str]:
    scheduled_id = stable_scheduled_episode_id(
        environment_id="LangMani-PickPlaceByInstruction-v0",
        scene_seed=scene_seed,
        scene_id=stable_scene_id(scene_seed),
        task_spec=task_spec,
        task_id=stable_task_id(task_spec),
        expert_fingerprint=_EXPERT_FINGERPRINT,
        control_mode="pd_joint_pos",
    )
    attempt_id = stable_attempt_id(scheduled_episode_id=scheduled_id, attempt_index=1)
    return (
        scheduled_id,
        attempt_id,
        stable_raw_trajectory_id(
            scheduled_episode_id=scheduled_id,
            attempt_id=attempt_id,
        ),
    )


def _accepted_group_id(scene_seed: int) -> str:
    return stable_accepted_scene_group_id(
        candidate_scene_id=_candidate_scene_id(scene_seed),
        raw_trajectory_ids=tuple(
            _episode_ids(scene_seed, task_spec)[2] for task_spec in CANONICAL_TASK_SPECS
        ),
    )


def _group_metadata(scene_seed: int) -> dict[str, Any]:
    return {
        "collection_schema_version": COLLECTION_SCHEMA_VERSION,
        "collection_run_id": _COLLECTION_RUN_ID,
        "candidate_scene_id": _candidate_scene_id(scene_seed),
        "accepted_scene_group_id": _accepted_group_id(scene_seed),
        "scene_seed": scene_seed,
        "scene_id": stable_scene_id(scene_seed),
        "expert_config_fingerprint": _EXPERT_FINGERPRINT,
    }


def _shard_metadata(shard_index: int = 0) -> dict[str, Any]:
    return {
        "collection_schema_version": COLLECTION_SCHEMA_VERSION,
        "collection_run_id": _COLLECTION_RUN_ID,
        "source_shard_id": stable_source_shard_id(
            collection_run_id=_COLLECTION_RUN_ID,
            shard_index=shard_index,
        ),
        "shard_index": shard_index,
        "expert_config_fingerprint": _EXPERT_FINGERPRINT,
    }


def _write_candidate(
    path: Path,
    *,
    scene_seed: int,
    tasks: tuple[TaskSpec, ...] = CANONICAL_TASK_SPECS,
    changed_initial_episode: int | None = None,
    corruption: str | None = None,
) -> None:
    episodes: list[dict[str, Any]] = []
    with h5py.File(path, "w") as archive:
        for episode_id, task_spec in enumerate(tasks):
            transition_count = 2
            group = archive.create_group(f"traj_{episode_id}")
            action_dtype = np.float64 if corruption == "action_dtype" else np.float32
            actions = np.zeros((transition_count, 8), dtype=action_dtype)
            if corruption == "nonfinite" and episode_id == 0:
                actions[0, 0] = np.nan
            group.create_dataset("actions", data=actions)
            group.create_dataset("terminated", data=np.array([False, True], dtype=bool))
            group.create_dataset("truncated", data=np.array([False, False], dtype=bool))
            group.create_dataset("success", data=np.array([False, True], dtype=bool))
            group.create_dataset("fail", data=np.array([False, False], dtype=bool))
            group.create_group("obs")  # obs_mode="none" is a legal empty group.
            states = group.create_group("env_states")
            actors = states.create_group("actors")
            state_length = (
                transition_count if corruption == "state_length" else transition_count + 1
            )
            for object_index, object_id in enumerate(("red_cube", "green_cube", "blue_cube")):
                actor_state = np.zeros((state_length, 13), dtype=np.float32)
                actor_state[:, 0] = -0.1 + 0.01 * object_index
                actor_state[:, 1] = 0.1 * (object_index - 1)
                actor_state[:, 2] = 0.02
                actor_state[:, 3] = 1.0
                actor_state[-1, 0] += 0.1
                if changed_initial_episode == episode_id and object_id == "red_cube":
                    actor_state[0, 0] += 0.001
                actors.create_dataset(object_id, data=actor_state)
            articulations = states.create_group("articulations")
            panda_state = np.zeros((state_length, 40), dtype=np.float32)
            panda_state[:, 3] = 1.0
            articulations.create_dataset("panda", data=panda_state)

            elapsed_steps = transition_count + (1 if corruption == "elapsed_steps" else 0)
            episodes.append(
                {
                    "episode_id": episode_id,
                    "episode_seed": scene_seed,
                    "control_mode": "pd_joint_pos",
                    "elapsed_steps": elapsed_steps,
                    "reset_kwargs": {
                        "seed": scene_seed,
                        "options": {"task_spec": task_spec.to_dict()},
                    },
                    "success": True,
                    "fail": False,
                }
            )
        if corruption == "dangling_group":
            archive.create_group("traj_999")
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "env_info": {
                    "env_id": "LangMani-PickPlaceByInstruction-v0",
                    "env_kwargs": {
                        "num_envs": 1,
                        "obs_mode": "none",
                        "reward_mode": "none",
                        "control_mode": "pd_joint_pos",
                        "sim_backend": "physx_cpu",
                    },
                    "max_episode_steps": 200,
                },
                "commit_info": None,
                "source_type": "motionplanning",
                "episodes": episodes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _langmani_metadata(scene_seed: int, task_spec: TaskSpec, index: int) -> dict[str, Any]:
    del index
    scene_id = stable_scene_id(scene_seed)
    task_id = stable_task_id(task_spec)
    scheduled_id, attempt_id, raw_id = _episode_ids(scene_seed, task_spec)
    expert_result = {
        "success": True,
        "status": "success",
        "scene_seed": scene_seed,
        "scene_id": scene_id,
        "task_id": task_id,
        "canonical_instruction": canonical_instruction(task_spec),
        "target_object_id": task_spec.target_object_id,
        "target_bin_id": task_spec.target_bin_id,
        "total_environment_steps": 2,
        "final_environment_evaluation": dict(_SUCCESS_EVALUATION),
    }
    return {
        "collection_schema_version": COLLECTION_SCHEMA_VERSION,
        "collection_run_id": _COLLECTION_RUN_ID,
        "candidate_scene_id": _candidate_scene_id(scene_seed),
        "accepted_scene_group_id": _accepted_group_id(scene_seed),
        "expert_config_fingerprint": _EXPERT_FINGERPRINT,
        "scheduled_episode_id": scheduled_id,
        "attempt_id": attempt_id,
        "raw_trajectory_id": raw_id,
        "scene_seed": scene_seed,
        "scene_id": scene_id,
        "task_spec": task_spec.to_dict(),
        "task_id": task_id,
        "canonical_instruction": canonical_instruction(task_spec),
        "expert_result": expert_result,
        "replay_validation": {
            "passed": True,
            "mode": "action",
            "recorded_success": True,
            "replay_success": True,
            "task_spec_matches": True,
            "wrong_object_in_target_bin": False,
            "target_in_wrong_bin": False,
            "target_off_table": False,
            "final_environment_evaluation": dict(_SUCCESS_EVALUATION),
            "recorded_action_steps": 2,
            "replayed_action_steps": 2,
            "final_position_error_m": 0.0,
            "final_orientation_error_rad": 0.0,
            "final_joint_error_rad": 0.0,
            "state_audit_performed": False,
            "state_audit_passed": None,
            "failure_codes": [],
            "failure_reasons": [],
        },
        "final_environment_evaluation": dict(_SUCCESS_EVALUATION),
        "initial_scene_state": _initial_scene_state(),
        "structural_valid": True,
        "accepted": True,
    }


def _selections(scene_seed: int) -> tuple[EpisodeSelection, ...]:
    return tuple(
        EpisodeSelection(
            native_episode_id=index,
            langmani_metadata=_langmani_metadata(scene_seed, task_spec, index),
        )
        for index, task_spec in enumerate(CANONICAL_TASK_SPECS)
    )


def _create_bundle(tmp_path: Path, *, scene_seed: int) -> Path:
    candidate = tmp_path / f"candidate_{scene_seed}.h5"
    _write_candidate(candidate, scene_seed=scene_seed)
    bundle = tmp_path / f"group_{scene_seed}.h5"
    create_counterfactual_scene_group_bundle(
        candidate,
        _selections(scene_seed),
        bundle,
        group_metadata=_group_metadata(scene_seed),
    )
    return bundle


def test_native_pair_validation_checks_hashes_and_recursive_state(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.h5"
    _write_candidate(candidate, scene_seed=41, tasks=(CANONICAL_TASK_SPECS[0],))

    result = validate_native_archive_pair(candidate)

    assert result.h5_sha256 == sha256_file(candidate)
    assert result.json_sha256 == sha256_file(candidate.with_suffix(".json"))
    assert result.h5_size_bytes > 0
    assert result.json_size_bytes > 0
    assert len(result.episodes) == 1
    assert result.episodes[0].scene_seed == 41
    assert result.episodes[0].task_id == stable_task_id(CANONICAL_TASK_SPECS[0])


@pytest.mark.parametrize("corruption", ["reward", "observation"])
def test_native_pair_rejects_data_outside_fixed_none_observation_contract(
    tmp_path: Path,
    corruption: str,
) -> None:
    candidate = tmp_path / f"{corruption}.h5"
    _write_candidate(candidate, scene_seed=42, tasks=(CANONICAL_TASK_SPECS[0],))
    with h5py.File(candidate, "a") as archive:
        group = archive["traj_0"]
        if corruption == "reward":
            group.create_dataset("rewards", data=np.zeros(2, dtype=np.float32))
        else:
            group["obs"].create_dataset("leaked", data=np.zeros((3, 1), dtype=np.float32))

    with pytest.raises(ArchiveValidationError, match="members invalid|empty group"):
        validate_native_archive_pair(candidate)


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        ("action_dtype", "dtype float32"),
        ("nonfinite", "NaN or infinity"),
        ("state_length", "first dimension must be 3"),
        ("elapsed_steps", "elapsed_steps=3"),
        ("dangling_group", "episode mismatch"),
    ],
)
def test_native_pair_validation_rejects_corruption(
    tmp_path: Path, corruption: str, match: str
) -> None:
    candidate = tmp_path / f"candidate_{corruption}.h5"
    _write_candidate(
        candidate,
        scene_seed=3,
        tasks=(CANONICAL_TASK_SPECS[0],),
        corruption=corruption,
    )

    with pytest.raises(ArchiveValidationError, match=match):
        validate_native_archive_pair(candidate)


def test_native_pair_validation_rejects_malformed_or_missing_pair(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.h5"
    _write_candidate(candidate, scene_seed=4, tasks=(CANONICAL_TASK_SPECS[0],))
    candidate.with_suffix(".json").write_text("{", encoding="utf-8")

    with pytest.raises(ArchiveValidationError, match="cannot read JSON metadata"):
        validate_native_archive_pair(candidate)

    candidate.with_suffix(".json").unlink()
    with pytest.raises(ArchiveValidationError, match="does not exist"):
        validate_native_archive_pair(candidate)


def test_native_pair_validation_rejects_external_links_and_bad_termination(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.h5"
    _write_candidate(candidate, scene_seed=5, tasks=(CANONICAL_TASK_SPECS[0],))
    external = tmp_path / "external.h5"
    with h5py.File(external, "w") as archive:
        archive.create_dataset("actions", data=np.zeros((2, 8), dtype=np.float32))
    with h5py.File(candidate, "r+") as archive:
        del archive["traj_0/actions"]
        archive["traj_0/actions"] = h5py.ExternalLink(external.name, "/actions")
    with pytest.raises(ArchiveValidationError, match="local HDF5 hard link"):
        validate_native_archive_pair(candidate)

    _write_candidate(candidate, scene_seed=5, tasks=(CANONICAL_TASK_SPECS[0],))
    with h5py.File(candidate, "r+") as archive:
        archive["traj_0/terminated"][:] = False
    with pytest.raises(ArchiveValidationError, match="must equal success OR fail"):
        validate_native_archive_pair(candidate)


@pytest.mark.parametrize("missing_field", ["control_mode", "num_envs"])
def test_native_pair_requires_reconstructable_environment_kwargs(
    tmp_path: Path, missing_field: str
) -> None:
    candidate = tmp_path / f"candidate_{missing_field}.h5"
    _write_candidate(candidate, scene_seed=6, tasks=(CANONICAL_TASK_SPECS[0],))
    json_path = candidate.with_suffix(".json")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    del metadata["env_info"]["env_kwargs"][missing_field]
    json_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ArchiveValidationError):
        validate_native_archive_pair(candidate)


def test_group_bundle_rejects_partial_group_and_wrong_task(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.h5"
    _write_candidate(candidate, scene_seed=8)
    selections = _selections(8)

    with pytest.raises(ArchiveValidationError, match="exactly six"):
        create_counterfactual_scene_group_bundle(
            candidate,
            selections[:-1],
            tmp_path / "partial.h5",
            group_metadata={"accepted_scene_group_id": "partial"},
        )

    wrong_metadata = _langmani_metadata(8, CANONICAL_TASK_SPECS[1], 0)
    wrong = (EpisodeSelection(0, wrong_metadata), *selections[1:])
    with pytest.raises(ArchiveValidationError, match="task identity disagrees with reset"):
        create_counterfactual_scene_group_bundle(
            candidate,
            wrong,
            tmp_path / "wrong.h5",
            group_metadata=_group_metadata(8),
        )


def test_group_bundle_copies_six_tasks_and_preserves_stable_provenance(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.h5"
    _write_candidate(candidate, scene_seed=12)
    bundle = tmp_path / "accepted_group.h5"

    result = create_counterfactual_scene_group_bundle(
        candidate,
        _selections(12),
        bundle,
        group_metadata=_group_metadata(12),
    )

    assert [episode.h5_group for episode in result.episodes] == [
        f"traj_{index}" for index in range(6)
    ]
    assert [episode.task_id for episode in result.episodes] == [
        stable_task_id(task_spec) for task_spec in CANONICAL_TASK_SPECS
    ]
    metadata = json.loads(bundle.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["langmani"]["archive_schema_version"] == ARCHIVE_SCHEMA_VERSION
    assert metadata["langmani"]["immutable"] is True
    assert [episode["episode_id"] for episode in metadata["episodes"]] == list(range(6))
    assert metadata["episodes"][0]["langmani"]["source_candidate"] == {
        "h5_sha256": sha256_file(candidate),
        "json_sha256": sha256_file(candidate.with_suffix(".json")),
        "native_episode_id": 0,
        "h5_group": "traj_0",
    }
    with pytest.raises(FileExistsError, match="already exists"):
        create_counterfactual_scene_group_bundle(
            candidate,
            _selections(12),
            bundle,
            group_metadata=_group_metadata(12),
        )


def test_group_bundle_rejects_nonidentical_initial_physical_state(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.h5"
    _write_candidate(candidate, scene_seed=13, changed_initial_episode=5)
    output = tmp_path / "rejected_group.h5"

    with pytest.raises(ArchiveValidationError, match="initial physical state"):
        create_counterfactual_scene_group_bundle(
            candidate,
            _selections(13),
            output,
            group_metadata=_group_metadata(13),
        )
    assert not output.exists()
    assert not output.with_suffix(".json").exists()


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("object_poses", "red_cube"),
        ("object_poses", "green_cube"),
        ("object_poses", "blue_cube"),
        ("bin_poses", "left_bin"),
        ("bin_poses", "right_bin"),
        ("panda_qpos", None),
    ],
)
def test_group_bundle_compares_every_explicit_initial_scene_component(
    tmp_path: Path,
    section: str,
    key: str | None,
) -> None:
    candidate = tmp_path / f"candidate-{section}-{key}.h5"
    _write_candidate(candidate, scene_seed=113)
    selections = list(_selections(113))
    changed = dict(selections[-1].langmani_metadata)
    snapshot = json.loads(json.dumps(changed["initial_scene_state"]))
    if key is None:
        snapshot[section][0] += 0.001
    else:
        snapshot[section][key][0] += 0.001
    changed["initial_scene_state"] = snapshot
    selections[-1] = EpisodeSelection(
        native_episode_id=selections[-1].native_episode_id,
        langmani_metadata=changed,
    )

    with pytest.raises(ArchiveValidationError, match="initial physical state"):
        create_counterfactual_scene_group_bundle(
            candidate,
            tuple(selections),
            tmp_path / f"rejected-{section}-{key}.h5",
            group_metadata=_group_metadata(113),
        )


def test_group_bundle_publish_failure_leaves_no_half_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate.h5"
    _write_candidate(candidate, scene_seed=14)
    output = tmp_path / "group.h5"
    real_replace = archive_module.os.replace
    replace_calls = 0

    def fail_second_replace(source: str | Path, destination: str | Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("injected JSON promotion failure")
        real_replace(source, destination)

    monkeypatch.setattr(archive_module.os, "replace", fail_second_replace)
    with pytest.raises(OSError, match="injected JSON promotion failure"):
        create_counterfactual_scene_group_bundle(
            candidate,
            _selections(14),
            output,
            group_metadata=_group_metadata(14),
        )
    assert not output.exists()
    assert not output.with_suffix(".json").exists()


def test_source_shard_is_deterministic_local_traj_sequence_and_immutable(tmp_path: Path) -> None:
    late_bundle = _create_bundle(tmp_path, scene_seed=22)
    early_bundle = _create_bundle(tmp_path, scene_seed=21)
    shard = tmp_path / "source_shard_000.h5"

    result = build_source_shard(
        [late_bundle, early_bundle],
        shard,
        shard_metadata=_shard_metadata(),
    )

    assert len(result.episodes) == 12
    assert [episode.h5_group for episode in result.episodes] == [
        f"traj_{index}" for index in range(12)
    ]
    assert [episode.scene_seed for episode in result.episodes] == [21] * 6 + [22] * 6
    metadata = json.loads(shard.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["langmani"]["archive_kind"] == "source_shard"
    assert metadata["langmani"]["accepted_scene_group_ids"] == [
        _accepted_group_id(21),
        _accepted_group_id(22),
    ]
    assert (
        metadata["episodes"][0]["langmani"]["source_shard_id"]
        == _shard_metadata()["source_shard_id"]
    )
    assert metadata["episodes"][0]["langmani"]["source_group_bundle"][
        "accepted_scene_group_id"
    ] == _accepted_group_id(21)
    with pytest.raises(FileExistsError, match="already exists"):
        build_source_shard(
            [early_bundle],
            shard,
            shard_metadata=_shard_metadata(),
        )


def test_source_shard_rejects_a_partial_or_unvalidated_bundle(tmp_path: Path) -> None:
    candidate = tmp_path / "partial.h5"
    _write_candidate(candidate, scene_seed=30, tasks=CANONICAL_TASK_SPECS[:5])

    with pytest.raises(ArchiveValidationError, match="missing langmani episode metadata"):
        build_source_shard(
            [candidate],
            tmp_path / "source_shard.h5",
            shard_metadata={"source_shard_id": "source-shard-partial"},
        )


def test_source_shard_validation_rechecks_canonical_task_semantics(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path, scene_seed=31)
    shard = tmp_path / "source_shard.h5"
    build_source_shard(
        [bundle],
        shard,
        shard_metadata=_shard_metadata(),
    )
    json_path = shard.with_suffix(".json")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    first = metadata["episodes"][0]
    second = metadata["episodes"][1]
    first_payload = {key: value for key, value in first.items() if key != "episode_id"}
    second_payload = {key: value for key, value in second.items() if key != "episode_id"}
    first.update(second_payload)
    second.update(first_payload)
    first["episode_id"] = 0
    second["episode_id"] = 1
    for native_id, episode in enumerate((first, second)):
        source = episode["langmani"]["source_group_bundle"]
        source["native_episode_id"] = native_id
        source["h5_group"] = f"traj_{native_id}"
    json_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ArchiveValidationError, match="stable identity|non-canonical task"):
        validate_native_archive_pair(
            shard,
            require_accepted_success=True,
            require_langmani_metadata=True,
        )
