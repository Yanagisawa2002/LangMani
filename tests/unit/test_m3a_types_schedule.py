"""M3A metadata, stable-identity, and deterministic-schedule contracts."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from langmani.collection.manifest import installed_runtime_versions
from langmani.datasets import (
    CANONICAL_TASK_SPECS,
    AttemptOutcome,
    AttemptRecord,
    CollectionConfig,
    CollectionManifest,
    CollectionSchedule,
    CollectionStatus,
    RawDatasetSummary,
    RawEpisodeRecord,
    ReplayFailureCode,
    ReplayValidationMode,
    ReplayValidationResult,
    SceneGroupRecord,
    SourceShardRecord,
    build_collection_schedule,
    canonical_json,
    scheduled_scene_group,
    sha256_hex,
    stable_accepted_scene_group_id,
    stable_attempt_id,
    stable_raw_trajectory_id,
    stable_scheduled_episode_id,
    stable_source_shard_id,
)
from langmani.environments.specs import EpisodeSpec, TaskSpec
from langmani.experts.types import (
    EXPERT_PHASE_SEQUENCE,
    ExpertConfig,
    ExpertPhase,
    ExpertResult,
    ExpertStatus,
    PhaseResult,
)


def _evaluation() -> dict[str, bool]:
    return {
        "target_in_target_bin": True,
        "target_in_wrong_bin": False,
        "wrong_object_in_target_bin": False,
        "target_is_grasped": False,
        "target_is_static": True,
        "target_off_table": False,
        "success": True,
        "fail": False,
    }


def test_default_collection_config_captures_installed_runtime_versions() -> None:
    assert dict(CollectionConfig().runtime_versions) == installed_runtime_versions()


def _expert_result(scene_seed: int, task_spec: TaskSpec) -> ExpertResult:
    episode = EpisodeSpec.create(scene_seed=scene_seed, task_spec=task_spec)
    phase_results = tuple(
        PhaseResult(phase=phase, success=True, status=ExpertStatus.SUCCESS)
        for phase in EXPERT_PHASE_SEQUENCE
    )
    return ExpertResult(
        success=True,
        status=ExpertStatus.SUCCESS,
        scene_seed=scene_seed,
        scene_id=episode.scene_id,
        task_id=episode.task_id,
        canonical_instruction=episode.canonical_instruction,
        target_object_id=task_spec.target_object_id,
        target_bin_id=task_spec.target_bin_id,
        total_environment_steps=20,
        total_planning_calls=0,
        total_replans=0,
        completed_phases=EXPERT_PHASE_SEQUENCE,
        failed_phase=None,
        final_environment_evaluation=_evaluation(),
        phase_results=phase_results,
        planning_duration_seconds=0.0,
        execution_duration_seconds=0.0,
    )


def _replay_result() -> ReplayValidationResult:
    return ReplayValidationResult(
        passed=True,
        mode=ReplayValidationMode.ACTION,
        recorded_success=True,
        replay_success=True,
        task_spec_matches=True,
        wrong_object_in_target_bin=False,
        target_in_wrong_bin=False,
        target_off_table=False,
        final_environment_evaluation=_evaluation(),
        recorded_action_steps=20,
        replayed_action_steps=20,
        final_position_error_m=0.0,
        final_orientation_error_rad=0.0,
        final_joint_error_rad=0.0,
        state_audit_performed=False,
        state_audit_passed=None,
    )


def _complete_manifest() -> CollectionManifest:
    config = CollectionConfig(
        target_complete_scene_count=1,
        maximum_candidate_scene_count=1,
        shard_size=6,
        replay_validation_mode=ReplayValidationMode.ACTION,
    )
    schedule = build_collection_schedule(config)
    shard_id = stable_source_shard_id(
        collection_run_id=schedule.collection_run_id,
        shard_index=0,
    )
    attempts: list[AttemptRecord] = []
    raw_episodes: list[RawEpisodeRecord] = []
    for native_episode_id, scheduled in enumerate(schedule.episodes):
        attempt_id = stable_attempt_id(
            scheduled_episode_id=scheduled.scheduled_episode_id,
            attempt_index=1,
        )
        raw_id = stable_raw_trajectory_id(
            scheduled_episode_id=scheduled.scheduled_episode_id,
            attempt_id=attempt_id,
        )
        expert_result = _expert_result(scheduled.scene_seed, scheduled.task_spec)
        replay_result = _replay_result()
        attempts.append(
            AttemptRecord(
                attempt_id=attempt_id,
                scheduled_episode_id=scheduled.scheduled_episode_id,
                attempt_index=1,
                scene_seed=scheduled.scene_seed,
                scene_id=scheduled.scene_id,
                task_spec=scheduled.task_spec,
                task_id=scheduled.task_id,
                outcome=AttemptOutcome.ACCEPTED,
                expert_result=expert_result,
                raw_trajectory_id=raw_id,
                replay_validation=replay_result,
                trajectory_retained=True,
            )
        )
        raw_episodes.append(
            RawEpisodeRecord(
                scheduled_episode_id=scheduled.scheduled_episode_id,
                attempt_id=attempt_id,
                raw_trajectory_id=raw_id,
                source_shard_id=shard_id,
                native_episode_id=native_episode_id,
                h5_group=f"traj_{native_episode_id}",
                h5_path="accepted/shard-00000.h5",
                json_path="accepted/shard-00000.json",
                h5_sha256="a" * 64,
                json_sha256="b" * 64,
                elapsed_steps=20,
                scene_seed=scheduled.scene_seed,
                scene_id=scheduled.scene_id,
                task_spec=scheduled.task_spec,
                task_id=scheduled.task_id,
                canonical_instruction=scheduled.canonical_instruction,
                expert_result=expert_result,
                replay_validation=replay_result,
                final_environment_evaluation=_evaluation(),
                structural_valid=True,
                accepted=True,
            )
        )

    raw_ids = tuple(episode.raw_trajectory_id for episode in raw_episodes)
    group = SceneGroupRecord(
        candidate_scene_index=0,
        candidate_scene_id=schedule.episodes[0].candidate_scene_id,
        accepted_scene_group_id=stable_accepted_scene_group_id(
            candidate_scene_id=schedule.episodes[0].candidate_scene_id,
            raw_trajectory_ids=raw_ids,
        ),
        scene_seed=0,
        scene_id=schedule.episodes[0].scene_id,
        scheduled_episode_ids=tuple(
            scheduled.scheduled_episode_id for scheduled in schedule.episodes
        ),
        attempt_ids=tuple(attempt.attempt_id for attempt in attempts),
        episodes=tuple(raw_episodes),
        complete=True,
        accepted=True,
        bundle_h5_sha256="c" * 64,
        bundle_json_sha256="d" * 64,
    )
    shard = SourceShardRecord(
        source_shard_id=shard_id,
        shard_index=0,
        h5_path="accepted/shard-00000.h5",
        json_path="accepted/shard-00000.json",
        h5_sha256="a" * 64,
        json_sha256="b" * 64,
        h5_size_bytes=100,
        json_size_bytes=100,
        episode_count=6,
        raw_trajectory_ids=raw_ids,
    )
    return CollectionManifest(
        collection_schema_version=config.collection_schema_version,
        collection_run_id=schedule.collection_run_id,
        config=config,
        schedule=schedule,
        status=CollectionStatus.COMPLETE,
        runtime_versions=config.runtime_versions,
        next_candidate_scene_index=1,
        attempts=tuple(attempts),
        raw_episodes=tuple(raw_episodes),
        scene_groups=(group,),
        source_shards=(shard,),
    )


def test_canonical_json_and_sha256_are_order_independent_and_strict() -> None:
    first = {"unicode": "立方体", "nested": {"b": 2, "a": 1}}
    second = {"nested": {"a": 1, "b": 2}, "unicode": "立方体"}

    assert canonical_json(first) == canonical_json(second)
    assert sha256_hex(first) == sha256_hex(second)
    assert len(sha256_hex(first)) == 64
    with pytest.raises(ValueError):
        canonical_json({"invalid": float("nan")})
    with pytest.raises(TypeError):
        canonical_json({1: "non-string key"})


def test_scheduled_episode_id_covers_every_required_semantic_input() -> None:
    task_spec = TaskSpec("red_cube", "left_bin", "canonical_v0")
    episode = EpisodeSpec.create(scene_seed=7, task_spec=task_spec)
    base = stable_scheduled_episode_id(
        environment_id="LangMani-PickPlaceByInstruction-v0",
        scene_seed=episode.scene_seed,
        scene_id=episode.scene_id,
        task_spec=task_spec,
        task_id=episode.task_id,
        expert_fingerprint="sha256:" + "a" * 64,
        control_mode="pd_joint_pos",
        schema_version="langmani-m3a-raw-v1",
    )
    changed_ids = {
        stable_scheduled_episode_id(
            environment_id="LangMani-PickPlaceByInstruction-v1",
            scene_seed=episode.scene_seed,
            scene_id=episode.scene_id,
            task_spec=task_spec,
            task_id=episode.task_id,
            expert_fingerprint="sha256:" + "a" * 64,
            control_mode="pd_joint_pos",
            schema_version="langmani-m3a-raw-v1",
        ),
        stable_scheduled_episode_id(
            environment_id="LangMani-PickPlaceByInstruction-v0",
            scene_seed=episode.scene_seed + 1,
            scene_id=episode.scene_id,
            task_spec=task_spec,
            task_id=episode.task_id,
            expert_fingerprint="sha256:" + "a" * 64,
            control_mode="pd_joint_pos",
            schema_version="langmani-m3a-raw-v1",
        ),
        stable_scheduled_episode_id(
            environment_id="LangMani-PickPlaceByInstruction-v0",
            scene_seed=episode.scene_seed,
            scene_id=episode.scene_id + "-changed",
            task_spec=task_spec,
            task_id=episode.task_id,
            expert_fingerprint="sha256:" + "a" * 64,
            control_mode="pd_joint_pos",
            schema_version="langmani-m3a-raw-v1",
        ),
        stable_scheduled_episode_id(
            environment_id="LangMani-PickPlaceByInstruction-v0",
            scene_seed=episode.scene_seed,
            scene_id=episode.scene_id,
            task_spec=TaskSpec("red_cube", "right_bin", "canonical_v0"),
            task_id=episode.task_id,
            expert_fingerprint="sha256:" + "a" * 64,
            control_mode="pd_joint_pos",
            schema_version="langmani-m3a-raw-v1",
        ),
        stable_scheduled_episode_id(
            environment_id="LangMani-PickPlaceByInstruction-v0",
            scene_seed=episode.scene_seed,
            scene_id=episode.scene_id,
            task_spec=task_spec,
            task_id=episode.task_id + "-changed",
            expert_fingerprint="sha256:" + "a" * 64,
            control_mode="pd_joint_pos",
            schema_version="langmani-m3a-raw-v1",
        ),
        stable_scheduled_episode_id(
            environment_id="LangMani-PickPlaceByInstruction-v0",
            scene_seed=episode.scene_seed,
            scene_id=episode.scene_id,
            task_spec=task_spec,
            task_id=episode.task_id,
            expert_fingerprint="sha256:" + "b" * 64,
            control_mode="pd_joint_pos",
            schema_version="langmani-m3a-raw-v1",
        ),
        stable_scheduled_episode_id(
            environment_id="LangMani-PickPlaceByInstruction-v0",
            scene_seed=episode.scene_seed,
            scene_id=episode.scene_id,
            task_spec=task_spec,
            task_id=episode.task_id,
            expert_fingerprint="sha256:" + "a" * 64,
            control_mode="pd_joint_delta_pos",
            schema_version="langmani-m3a-raw-v1",
        ),
        stable_scheduled_episode_id(
            environment_id="LangMani-PickPlaceByInstruction-v0",
            scene_seed=episode.scene_seed,
            scene_id=episode.scene_id,
            task_spec=task_spec,
            task_id=episode.task_id,
            expert_fingerprint="sha256:" + "a" * 64,
            control_mode="pd_joint_pos",
            schema_version="langmani-m3a-raw-v2",
        ),
    }

    assert base not in changed_ids
    assert len(changed_ids) == 8


def test_default_config_has_explicit_m3a_bounds_and_round_trips() -> None:
    config = CollectionConfig()
    payload = config.to_dict()

    assert config.target_complete_scene_count == 60
    assert config.maximum_candidate_scene_count == 120
    assert config.maximum_expert_attempts_per_task == 3
    assert config.shard_size == 60
    assert config.final_joint_tolerance_rad == pytest.approx(0.05)
    assert set(config.runtime_versions) >= {"mani_skill", "h5py", "sapien", "torch", "mplib"}
    assert payload["expert_config_fingerprint"].startswith("sha256:")
    assert CollectionConfig.from_dict(json.loads(json.dumps(payload))) == config

    with pytest.raises(FrozenInstanceError):
        config.shard_size = 6  # type: ignore[misc]
    with pytest.raises(TypeError):
        config.runtime_versions["torch"] = "changed"  # type: ignore[index]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_complete_scene_count": 0}, "positive"),
        (
            {"target_complete_scene_count": 2, "maximum_candidate_scene_count": 1},
            "cannot be smaller",
        ),
        ({"maximum_expert_attempts_per_task": 0}, "positive"),
        ({"shard_size": 7}, "multiple of six"),
        ({"final_position_tolerance_m": 0}, "positive"),
        ({"control_mode": "pd_joint_delta_pos"}, "pd_joint_pos"),
    ],
)
def test_config_rejects_invalid_bounds(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        CollectionConfig(**kwargs)  # type: ignore[arg-type]


def test_schedule_is_seed_major_object_major_and_process_stable() -> None:
    config = CollectionConfig(
        candidate_scene_seed_start=40,
        target_complete_scene_count=2,
        maximum_candidate_scene_count=3,
        shard_size=6,
    )
    first = build_collection_schedule(config)
    second = build_collection_schedule(CollectionConfig.from_dict(config.to_dict()))

    assert first == second
    assert first.candidate_scene_seeds == (40, 41, 42)
    assert len(first.episodes) == 18
    assert tuple(episode.task_spec for episode in first.episodes[:6]) == CANONICAL_TASK_SPECS
    assert tuple((task.target_object_id, task.target_bin_id) for task in CANONICAL_TASK_SPECS) == (
        ("red_cube", "left_bin"),
        ("red_cube", "right_bin"),
        ("green_cube", "left_bin"),
        ("green_cube", "right_bin"),
        ("blue_cube", "left_bin"),
        ("blue_cube", "right_bin"),
    )
    assert len({episode.candidate_scene_id for episode in first.episodes[:6]}) == 1
    assert len({episode.scheduled_episode_id for episode in first.episodes}) == 18
    assert CollectionSchedule.from_dict(json.loads(json.dumps(first.to_dict()))) == first
    assert scheduled_scene_group(first, 1) == first.episodes[6:12]


def test_schedule_identity_ignores_storage_policy_but_tracks_expert_config() -> None:
    base = CollectionConfig(
        target_complete_scene_count=1,
        maximum_candidate_scene_count=1,
        shard_size=6,
    )
    relocated = replace(base, raw_output_root="another/output/root")
    changed_expert = replace(base, expert_config=ExpertConfig(max_episode_steps=199))

    assert (
        build_collection_schedule(base).collection_run_id
        == build_collection_schedule(relocated).collection_run_id
    )
    assert (
        build_collection_schedule(base).collection_run_id
        != build_collection_schedule(changed_expert).collection_run_id
    )
    assert (
        build_collection_schedule(base).episodes[0].scheduled_episode_id
        != build_collection_schedule(changed_expert).episodes[0].scheduled_episode_id
    )


def test_replay_result_enforces_success_evidence_and_round_trips() -> None:
    result = _replay_result()
    assert ReplayValidationResult.from_dict(json.loads(json.dumps(result.to_dict()))) == result
    failure = replace(
        result,
        passed=False,
        replay_success=False,
        final_environment_evaluation={"success": False, "fail": False},
        failure_codes=(ReplayFailureCode.FINAL_SUCCESS_FAILURE,),
        failure_reasons=("action replay did not finish with M1 success",),
    )
    assert ReplayValidationResult.from_dict(json.loads(json.dumps(failure.to_dict()))) == failure
    with pytest.raises(TypeError):
        result.final_environment_evaluation["success"] = False  # type: ignore[index]
    with pytest.raises(ValueError, match="inconsistent"):
        replace(
            result,
            passed=True,
            replay_success=False,
            failure_codes=(ReplayFailureCode.FINAL_SUCCESS_FAILURE,),
            failure_reasons=("replay failed",),
        )
    with pytest.raises(ValueError, match="requires a state audit"):
        replace(result, mode=ReplayValidationMode.ACTION_AND_STATE_AUDIT)
    with pytest.raises(ValueError, match="both be empty or both be non-empty"):
        replace(
            result,
            passed=False,
            replay_success=False,
            final_environment_evaluation={"success": False, "fail": False},
            failure_codes=(ReplayFailureCode.FINAL_SUCCESS_FAILURE,),
        )


def test_complete_manifest_and_summary_round_trip_without_losing_provenance() -> None:
    manifest = _complete_manifest()
    payload = manifest.to_dict()
    decoded = CollectionManifest.from_dict(json.loads(json.dumps(payload)))
    summary = RawDatasetSummary.from_manifest(decoded)

    assert decoded == manifest
    assert decoded.accepted_scene_group_count == 1
    assert decoded.accepted_episode_count == 6
    assert len(decoded.attempts) == 6
    assert all(attempt.expert_result.success for attempt in decoded.attempts)
    assert summary.accepted_episode_count == 6
    assert summary.complete_group_acceptance_rate == pytest.approx(1.0)
    assert sum(summary.task_episode_counts.values()) == 6
    assert RawDatasetSummary.from_dict(json.loads(json.dumps(summary.to_dict()))) == summary
    assert "trajectory" not in payload["attempts"][0]["expert_result"]


def test_failed_scene_group_requires_reason_and_cannot_claim_acceptance() -> None:
    schedule = build_collection_schedule(
        CollectionConfig(
            target_complete_scene_count=1,
            maximum_candidate_scene_count=1,
            shard_size=6,
        )
    )
    kwargs = {
        "candidate_scene_index": 0,
        "candidate_scene_id": schedule.episodes[0].candidate_scene_id,
        "accepted_scene_group_id": None,
        "scene_seed": 0,
        "scene_id": schedule.episodes[0].scene_id,
        "scheduled_episode_ids": tuple(
            episode.scheduled_episode_id for episode in schedule.episodes
        ),
        "attempt_ids": (),
        "episodes": (),
        "complete": False,
        "accepted": False,
    }
    with pytest.raises(ValueError, match="rejection_reasons"):
        SceneGroupRecord(**kwargs)  # type: ignore[arg-type]

    group = SceneGroupRecord(**kwargs, rejection_reasons=("red_cube:left_bin replay failed",))  # type: ignore[arg-type]
    assert group.scene_group_id == group.candidate_scene_id
    assert SceneGroupRecord.from_dict(json.loads(json.dumps(group.to_dict()))) == group


def test_persistent_id_helpers_reject_invalid_indices() -> None:
    with pytest.raises(ValueError, match="positive"):
        stable_attempt_id(scheduled_episode_id="episode", attempt_index=0)
    with pytest.raises(ValueError, match="non-negative"):
        stable_source_shard_id(collection_run_id="run", shard_index=-1)
    with pytest.raises(ValueError, match="exactly six"):
        stable_accepted_scene_group_id(
            candidate_scene_id="candidate",
            raw_trajectory_ids=("only-one",),
        )


def test_tampered_scheduled_episode_id_is_rejected_on_resume_parse() -> None:
    schedule = build_collection_schedule(
        CollectionConfig(
            target_complete_scene_count=1,
            maximum_candidate_scene_count=1,
            shard_size=6,
        )
    )
    payload = schedule.to_dict()
    payload["episodes"][0]["scheduled_episode_id"] = "tampered"
    with pytest.raises(ValueError, match="scheduled_episode_id"):
        CollectionSchedule.from_dict(payload)


def test_phase_enum_remains_usable_in_expert_result_fixture() -> None:
    assert EXPERT_PHASE_SEQUENCE[0] is ExpertPhase.INITIALIZE
