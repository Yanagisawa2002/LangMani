"""CPU-only end-to-end tests for M3A collection orchestration."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from langmani.collection.collector import CollectionError, RawDemonstrationCollector
from langmani.collection.inspection import (
    DatasetInspectionError,
    inspect_raw_dataset,
    source_archive_digest,
)
from langmani.collection.manifest import ManifestError, initialize_or_resume_manifest, load_manifest
from langmani.datasets.identity import stable_attempt_id, stable_raw_trajectory_id
from langmani.datasets.types import (
    CollectionConfig,
    CollectionStatus,
    OverwritePolicy,
    ReplayValidationMode,
    ReplayValidationResult,
)
from langmani.environments.specs import EpisodeSpec, TaskSpec
from langmani.experts.types import (
    EXPERT_PHASE_SEQUENCE,
    ExpertPhase,
    ExpertResult,
    ExpertStatus,
    PhaseResult,
)

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


class _FakeRecorder:
    def __init__(self, directory: Path, sim_backend: str) -> None:
        assert sim_backend == "physx_cpu"
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.h5_path = directory / "attempts.h5"
        self.json_path = directory / "attempts.json"
        self.seed = 0
        self.task_spec: TaskSpec | None = None
        self.actions: list[np.ndarray] = []

    def reset(self, *, seed: int, options: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        self.seed = seed
        self.task_spec = TaskSpec.from_mapping(options["task_spec"])
        self.actions = []
        return {}, {"reconfigure": False}

    def step(self, action: object) -> tuple[None, float, bool, bool, dict[str, bool]]:
        self.actions.append(np.asarray(action, dtype=np.float32))
        success = len(self.actions) >= 2
        return (
            None,
            0.0,
            success,
            False,
            {
                "success": success,
                "fail": False,
            },
        )

    def get_episode_specs(self) -> tuple[EpisodeSpec, ...]:
        task_spec = self.task_spec
        assert task_spec is not None
        return (EpisodeSpec.create(scene_seed=self.seed, task_spec=task_spec),)

    def get_expert_initial_scene_state(self) -> dict[str, Any]:
        return {
            "object_poses": {
                "red_cube": [-0.10, -0.10, 0.025, 1.0, 0.0, 0.0, 0.0],
                "green_cube": [-0.09, 0.00, 0.025, 1.0, 0.0, 0.0, 0.0],
                "blue_cube": [-0.08, 0.10, 0.025, 1.0, 0.0, 0.0, 0.0],
            },
            "bin_poses": {
                "left_bin": [0.08, 0.18, 0.004, 1.0, 0.0, 0.0, 0.0],
                "right_bin": [0.08, -0.18, 0.004, 1.0, 0.0, 0.0, 0.0],
            },
            "panda_qpos": [0.0] * 9,
        }

    def get_expert_evaluation(self) -> dict[str, bool]:
        if len(self.actions) >= 2:
            return dict(_SUCCESS_EVALUATION)
        return {
            **_SUCCESS_EVALUATION,
            "target_in_target_bin": False,
            "target_is_static": False,
            "success": False,
        }

    def flush_trajectory(self, *, save: bool) -> None:
        if not save or not self.actions:
            return
        payload = self._payload()
        episode_id = len(payload["episodes"])
        task_spec = self.task_spec
        assert task_spec is not None
        actions = np.stack(self.actions).astype(np.float32)
        transition_count = len(actions)
        with h5py.File(self.h5_path, "a") as archive:
            group = archive.create_group(f"traj_{episode_id}")
            group.create_dataset("actions", data=actions)
            success = np.zeros(transition_count, dtype=bool)
            success[-1] = True
            group.create_dataset("terminated", data=success)
            group.create_dataset("truncated", data=np.zeros(transition_count, dtype=bool))
            group.create_dataset("success", data=success)
            group.create_dataset("fail", data=np.zeros(transition_count, dtype=bool))
            group.create_group("obs")
            states = group.create_group("env_states")
            actors = states.create_group("actors")
            for object_index, object_id in enumerate(("red_cube", "green_cube", "blue_cube")):
                actor_state = np.zeros((transition_count + 1, 13), dtype=np.float32)
                actor_state[:, 0] = -0.1 + object_index * 0.01
                actor_state[:, 1] = -0.1 + object_index * 0.1
                actor_state[:, 2] = 0.025
                actor_state[:, 3] = 1.0
                actor_state[-1, 0] += 0.2
                actors.create_dataset(object_id, data=actor_state)
            articulations = states.create_group("articulations")
            panda = np.zeros((transition_count + 1, 40), dtype=np.float32)
            panda[:, 3] = 1.0
            articulations.create_dataset("panda", data=panda)
        payload["episodes"].append(
            {
                "episode_id": episode_id,
                "episode_seed": self.seed,
                "control_mode": "pd_joint_pos",
                "elapsed_steps": transition_count,
                "reset_kwargs": {
                    "seed": self.seed,
                    "options": {"task_spec": task_spec.to_dict()},
                },
                "success": True,
                "fail": False,
            }
        )
        self.json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def close(self) -> None:
        return None

    def _payload(self) -> dict[str, Any]:
        if self.json_path.exists():
            return json.loads(self.json_path.read_text(encoding="utf-8"))
        return {
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
            "episodes": [],
        }


class _FakeExpert:
    def __init__(self, recorder: _FakeRecorder, config: CollectionConfig) -> None:
        self.recorder = recorder
        self.config = config

    def run(self) -> ExpertResult:
        self.recorder.step(np.zeros(8, dtype=np.float32))
        self.recorder.step(np.ones(8, dtype=np.float32) * 0.1)
        task_spec = self.recorder.task_spec
        assert task_spec is not None
        episode = EpisodeSpec.create(scene_seed=self.recorder.seed, task_spec=task_spec)
        phases = tuple(_successful_phase(phase) for phase in EXPERT_PHASE_SEQUENCE)
        return ExpertResult(
            success=True,
            status=ExpertStatus.SUCCESS,
            scene_seed=episode.scene_seed,
            scene_id=episode.scene_id,
            task_id=episode.task_id,
            canonical_instruction=episode.canonical_instruction,
            target_object_id=task_spec.target_object_id,
            target_bin_id=task_spec.target_bin_id,
            total_environment_steps=2,
            total_planning_calls=0,
            total_replans=0,
            completed_phases=EXPERT_PHASE_SEQUENCE,
            failed_phase=None,
            final_environment_evaluation=_SUCCESS_EVALUATION,
            phase_results=phases,
            planning_duration_seconds=0.0,
            execution_duration_seconds=0.0,
        )


def _successful_phase(phase: ExpertPhase) -> PhaseResult:
    return PhaseResult(
        phase=phase,
        success=True,
        status=ExpertStatus.SUCCESS,
        attempts=1,
    )


def _failed_result(recorder: _FakeRecorder) -> ExpertResult:
    task_spec = recorder.task_spec
    assert task_spec is not None
    episode = EpisodeSpec.create(scene_seed=recorder.seed, task_spec=task_spec)
    initialize = _successful_phase(ExpertPhase.INITIALIZE)
    failed = PhaseResult(
        phase=ExpertPhase.MOVE_TO_PREGRASP,
        success=False,
        status=ExpertStatus.PLANNING_FAILURE,
        attempts=1,
        planning_calls=1,
        message="synthetic planning failure",
        planner_status="screw plan failed",
    )
    return ExpertResult(
        success=False,
        status=ExpertStatus.PLANNING_FAILURE,
        scene_seed=episode.scene_seed,
        scene_id=episode.scene_id,
        task_id=episode.task_id,
        canonical_instruction=episode.canonical_instruction,
        target_object_id=task_spec.target_object_id,
        target_bin_id=task_spec.target_bin_id,
        total_environment_steps=0,
        total_planning_calls=1,
        total_replans=0,
        completed_phases=(ExpertPhase.INITIALIZE,),
        failed_phase=ExpertPhase.MOVE_TO_PREGRASP,
        final_environment_evaluation={**_SUCCESS_EVALUATION, "success": False},
        phase_results=(initialize, failed),
        planning_duration_seconds=0.0,
        execution_duration_seconds=0.0,
    )


def _passed_replay(*args: object, **kwargs: Any) -> ReplayValidationResult:
    del args
    config: CollectionConfig = kwargs["config"]
    state_audit = config.replay_validation_mode is ReplayValidationMode.ACTION_AND_STATE_AUDIT
    return ReplayValidationResult(
        passed=True,
        mode=config.replay_validation_mode,
        recorded_success=True,
        replay_success=True,
        task_spec_matches=True,
        wrong_object_in_target_bin=False,
        target_in_wrong_bin=False,
        target_off_table=False,
        final_environment_evaluation=_SUCCESS_EVALUATION,
        recorded_action_steps=2,
        replayed_action_steps=2,
        final_position_error_m=0.0,
        final_orientation_error_rad=0.0,
        final_joint_error_rad=0.0,
        state_audit_performed=state_audit,
        state_audit_passed=True if state_audit else None,
    )


def _config(root: Path) -> CollectionConfig:
    return CollectionConfig(
        target_complete_scene_count=1,
        maximum_candidate_scene_count=1,
        maximum_expert_attempts_per_task=1,
        raw_output_root=str(root),
        shard_size=6,
    )


def _collector(config: CollectionConfig, recorder_factory: Any) -> RawDemonstrationCollector:
    return RawDemonstrationCollector(
        config,
        recorder_factory=recorder_factory,
        expert_factory=lambda env, cfg: _FakeExpert(env, cfg),
        replay_validator=_passed_replay,
    )


def test_complete_group_is_atomically_sharded_inspectable_and_resumable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "raw"
    config = _config(root)
    recorder_calls = 0

    def recorder_factory(directory: Path, sim_backend: str) -> _FakeRecorder:
        nonlocal recorder_calls
        recorder_calls += 1
        return _FakeRecorder(directory, sim_backend)

    collector = _collector(config, recorder_factory)
    summary = collector.collect()

    assert summary.status is CollectionStatus.COMPLETE
    assert summary.accepted_scene_group_count == 1
    assert summary.accepted_episode_count == 6
    assert summary.task_episode_counts and set(summary.task_episode_counts.values()) == {1}
    assert recorder_calls == 1
    inspected = inspect_raw_dataset(root)
    assert inspected.to_dict() == summary.to_dict()

    manifest = load_manifest(root)
    assert len(manifest.attempts) == len(manifest.raw_episodes) == 6
    assert len(manifest.scene_groups) == len(manifest.source_shards) == 1
    assert manifest.scene_groups[0].accepted
    assert all(attempt.outcome.value == "accepted" for attempt in manifest.attempts)
    assert not (root / "journal").exists()
    digest = source_archive_digest(manifest)
    assert digest.startswith("sha256:") and len(digest) == len("sha256:") + 64
    assert source_archive_digest(load_manifest(root)) == digest

    relocated = replace(
        manifest,
        config=replace(manifest.config, raw_output_root=str(tmp_path / "relocated")),
    )
    assert source_archive_digest(relocated) == digest

    changed_h5_sha256 = "0" * 64
    changed_episodes = tuple(
        replace(episode, h5_sha256=changed_h5_sha256) for episode in manifest.raw_episodes
    )
    changed_by_id = {episode.raw_trajectory_id: episode for episode in changed_episodes}
    changed_groups = tuple(
        replace(
            group,
            episodes=tuple(changed_by_id[episode.raw_trajectory_id] for episode in group.episodes),
        )
        for group in manifest.scene_groups
    )
    changed_shards = tuple(
        replace(shard, h5_sha256=changed_h5_sha256) for shard in manifest.source_shards
    )
    changed_manifest = replace(
        manifest,
        raw_episodes=changed_episodes,
        scene_groups=changed_groups,
        source_shards=changed_shards,
    )
    assert source_archive_digest(changed_manifest) != digest

    resumed = RawDemonstrationCollector(
        config,
        recorder_factory=lambda *_: (_ for _ in ()).throw(
            AssertionError("completed collection must not create another recorder")
        ),
        replay_validator=_passed_replay,
    ).collect()
    assert resumed.to_dict() == summary.to_dict()


def test_resume_repairs_partial_shard_and_does_not_repeat_committed_scene(
    tmp_path: Path,
) -> None:
    root = tmp_path / "resumable"
    config = CollectionConfig(
        target_complete_scene_count=2,
        maximum_candidate_scene_count=2,
        maximum_expert_attempts_per_task=1,
        raw_output_root=str(root),
        shard_size=12,
    )
    created_seeds: list[int] = []

    def recorder_factory(directory: Path, sim_backend: str) -> _FakeRecorder:
        recorder = _FakeRecorder(directory, sim_backend)
        original_reset = recorder.reset

        def reset(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            if not created_seeds or created_seeds[-1] != kwargs["seed"]:
                created_seeds.append(kwargs["seed"])
            return original_reset(**kwargs)

        recorder.reset = reset  # type: ignore[method-assign]
        return recorder

    interrupted = _collector(config, recorder_factory)
    original_collect_candidate = interrupted._collect_candidate

    def stop_after_first(manifest: Any, candidate_index: int) -> Any:
        if candidate_index == 1:
            raise RuntimeError("simulated process interruption")
        return original_collect_candidate(manifest, candidate_index)

    interrupted._collect_candidate = stop_after_first  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="simulated process interruption"):
        interrupted.collect()

    partial = load_manifest(root)
    assert partial.status is CollectionStatus.IN_PROGRESS
    assert partial.next_candidate_scene_index == 1
    assert partial.accepted_scene_group_count == 1
    partial_h5 = root / partial.source_shards[0].h5_path
    with partial_h5.open("ab") as stream:
        stream.write(b"simulated-unmanifested-tail")

    completed = _collector(config, recorder_factory).collect()
    manifest = load_manifest(root)

    assert completed.status is CollectionStatus.COMPLETE
    assert manifest.accepted_scene_group_count == 2
    assert len(manifest.raw_episodes) == len(manifest.attempts) == 12
    assert created_seeds == [0, 1]
    assert inspect_raw_dataset(root).accepted_episode_count == 12
    assert not (root / "journal").exists()


def test_one_failed_task_rejects_the_entire_counterfactual_group(tmp_path: Path) -> None:
    root = tmp_path / "rejected"
    config = _config(root)

    class SelectiveExpert:
        def __init__(self, recorder: _FakeRecorder) -> None:
            self.recorder = recorder

        def run(self) -> ExpertResult:
            task = self.recorder.task_spec
            assert task is not None
            if task.target_object_id == "green_cube" and task.target_bin_id == "right_bin":
                return _failed_result(self.recorder)
            return _FakeExpert(self.recorder, config).run()

    summary = RawDemonstrationCollector(
        config,
        recorder_factory=lambda directory, backend: _FakeRecorder(directory, backend),
        expert_factory=lambda env, cfg: SelectiveExpert(env),
        replay_validator=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("partial groups must not reach replay")
        ),
    ).collect()
    manifest = load_manifest(root)

    assert summary.status is CollectionStatus.FAILED
    assert summary.accepted_scene_group_count == summary.accepted_episode_count == 0
    assert len(manifest.scene_groups) == 1 and not manifest.scene_groups[0].accepted
    assert manifest.raw_episodes == manifest.source_shards == ()
    assert len(manifest.attempts) == 6
    assert sum(attempt.outcome.value == "expert_failure" for attempt in manifest.attempts) == 1
    assert sum(attempt.outcome.value == "group_rejected" for attempt in manifest.attempts) == 5
    assert not (root / "accepted").exists()


@pytest.mark.parametrize(
    ("replacement", "reason"),
    [
        ({"scene_id": "scene-v1-wrong"}, "scene_id differs"),
        ({"task_id": "task-v1-wrong"}, "task_id differs"),
        ({"target_object_id": "blue_cube"}, "target object differs"),
        ({"target_bin_id": "right_bin"}, "target bin differs"),
        ({"total_environment_steps": 201}, "max_episode_steps"),
    ],
)
def test_successful_expert_contract_drift_never_reaches_action_replay(
    tmp_path: Path,
    replacement: dict[str, Any],
    reason: str,
) -> None:
    root = tmp_path / ("expert-contract-" + next(iter(replacement)))
    config = _config(root)
    replay_called = False

    class DriftedExpert(_FakeExpert):
        def run(self) -> ExpertResult:
            result = super().run()
            task_spec = self.recorder.task_spec
            assert task_spec is not None
            if task_spec.target_object_id == "red_cube" and task_spec.target_bin_id == "left_bin":
                return replace(result, **replacement)
            return result

    def replay_validator(*args: object, **kwargs: Any) -> ReplayValidationResult:
        nonlocal replay_called
        replay_called = True
        return _passed_replay(*args, **kwargs)

    summary = RawDemonstrationCollector(
        config,
        recorder_factory=lambda directory, backend: _FakeRecorder(directory, backend),
        expert_factory=lambda env, cfg: DriftedExpert(env, cfg),
        replay_validator=replay_validator,
    ).collect()
    manifest = load_manifest(root)

    assert summary.status is CollectionStatus.FAILED
    assert replay_called is False
    rejected = manifest.attempts[0]
    assert rejected.outcome.value == "expert_contract_failure"
    assert rejected.expert_result.success is True
    assert any(reason in item for item in rejected.failure_reasons)
    assert manifest.raw_episodes == ()


def test_expert_success_must_agree_with_fresh_environment_evaluation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "expert-environment-disagreement"
    config = _config(root)

    class DisagreeingRecorder(_FakeRecorder):
        def get_expert_evaluation(self) -> dict[str, bool]:
            evaluation = super().get_expert_evaluation()
            task_spec = self.task_spec
            assert task_spec is not None
            if task_spec.target_object_id == "red_cube" and task_spec.target_bin_id == "left_bin":
                evaluation["success"] = False
                evaluation["target_in_target_bin"] = False
            return evaluation

    summary = RawDemonstrationCollector(
        config,
        recorder_factory=lambda directory, backend: DisagreeingRecorder(directory, backend),
        expert_factory=lambda env, cfg: _FakeExpert(env, cfg),
        replay_validator=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("environment disagreement must not reach action replay")
        ),
    ).collect()
    manifest = load_manifest(root)

    assert summary.status is CollectionStatus.FAILED
    assert manifest.attempts[0].outcome.value == "expert_contract_failure"
    assert any(
        "disagrees with environment field success" in reason
        for reason in manifest.attempts[0].failure_reasons
    )


@pytest.mark.parametrize(
    "false_success_field",
    ["wrong_object_in_target_bin", "target_in_wrong_bin"],
)
def test_wrong_object_or_bin_false_success_never_reaches_replay(
    tmp_path: Path,
    false_success_field: str,
) -> None:
    root = tmp_path / f"false-success-{false_success_field}"
    config = _config(root)

    class FalseSuccessRecorder(_FakeRecorder):
        def get_expert_evaluation(self) -> dict[str, bool]:
            evaluation = super().get_expert_evaluation()
            task_spec = self.task_spec
            assert task_spec is not None
            if task_spec.target_object_id == "red_cube" and task_spec.target_bin_id == "left_bin":
                evaluation[false_success_field] = True
            return evaluation

    class FalseSuccessExpert(_FakeExpert):
        def run(self) -> ExpertResult:
            result = super().run()
            task_spec = self.recorder.task_spec
            assert task_spec is not None
            if task_spec.target_object_id == "red_cube" and task_spec.target_bin_id == "left_bin":
                evaluation = dict(result.final_environment_evaluation)
                evaluation[false_success_field] = True
                return replace(result, final_environment_evaluation=evaluation)
            return result

    summary = RawDemonstrationCollector(
        config,
        recorder_factory=lambda directory, backend: FalseSuccessRecorder(directory, backend),
        expert_factory=lambda env, cfg: FalseSuccessExpert(env, cfg),
        replay_validator=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("wrong-target false success must not reach action replay")
        ),
    ).collect()
    manifest = load_manifest(root)

    assert summary.status is CollectionStatus.FAILED
    assert manifest.attempts[0].outcome.value == "expert_contract_failure"
    assert any(
        f"{false_success_field} must be False" in reason
        for reason in manifest.attempts[0].failure_reasons
    )


def test_resume_recovers_crash_after_full_shard_publish_before_manifest_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "publish-crash"
    config = CollectionConfig(
        target_complete_scene_count=2,
        maximum_candidate_scene_count=3,
        maximum_expert_attempts_per_task=1,
        raw_output_root=str(root),
        shard_size=12,
    )
    import langmani.collection.collector as collector_module

    real_persist = collector_module.persist_manifest

    def crash_before_second_commit(path: Path, manifest: Any) -> None:
        if manifest.next_candidate_scene_index == 2:
            raise RuntimeError("simulated post-publish interruption")
        real_persist(path, manifest)

    monkeypatch.setattr(collector_module, "persist_manifest", crash_before_second_commit)
    with pytest.raises(RuntimeError, match="post-publish interruption"):
        _collector(
            config,
            lambda directory, backend: _FakeRecorder(directory, backend),
        ).collect()

    before_resume = load_manifest(root)
    assert before_resume.next_candidate_scene_index == 1
    assert (root / "journal").exists()

    monkeypatch.setattr(collector_module, "persist_manifest", real_persist)
    summary = _collector(
        config,
        lambda directory, backend: _FakeRecorder(directory, backend),
    ).collect()
    manifest = load_manifest(root)

    assert summary.status is CollectionStatus.COMPLETE
    assert [group.candidate_scene_index for group in manifest.scene_groups] == [0, 1, 2]
    assert [group.accepted for group in manifest.scene_groups] == [True, False, True]
    assert len(manifest.attempts) == 18
    assert len({attempt.attempt_id for attempt in manifest.attempts}) == 18
    assert inspect_raw_dataset(root).accepted_episode_count == 12
    assert not (root / "journal").exists()


def test_resume_records_active_attempt_interruption_without_reusing_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "attempt-crash"
    config = CollectionConfig(
        target_complete_scene_count=1,
        maximum_candidate_scene_count=2,
        maximum_expert_attempts_per_task=1,
        raw_output_root=str(root),
        shard_size=6,
    )

    class InterruptingExpert:
        def run(self) -> ExpertResult:
            raise KeyboardInterrupt("simulated abrupt interruption")

    interrupted = RawDemonstrationCollector(
        config,
        recorder_factory=lambda directory, backend: _FakeRecorder(directory, backend),
        expert_factory=lambda env, cfg: InterruptingExpert(),
        replay_validator=_passed_replay,
    )
    with pytest.raises(KeyboardInterrupt, match="abrupt interruption"):
        interrupted.collect()

    assert (root / "journal" / "inflight" / "candidate-000000.json").is_file()
    summary = _collector(
        config,
        lambda directory, backend: _FakeRecorder(directory, backend),
    ).collect()
    manifest = load_manifest(root)
    interrupted_scheduled_id = manifest.schedule.episodes[0].scheduled_episode_id
    interrupted_attempt_id = stable_attempt_id(
        scheduled_episode_id=interrupted_scheduled_id,
        attempt_index=1,
    )

    assert summary.status is CollectionStatus.COMPLETE
    assert [group.accepted for group in manifest.scene_groups] == [False, True]
    assert manifest.attempts[0].attempt_id == interrupted_attempt_id
    assert manifest.attempts[0].outcome.value == "unexpected_exception"
    assert len(manifest.attempts) == 7
    assert len({attempt.attempt_id for attempt in manifest.attempts}) == 7
    assert not (root / "journal").exists()


def test_completed_resume_rebuilds_stale_manifest_projections(tmp_path: Path) -> None:
    root = tmp_path / "projection-repair"
    config = _config(root)
    summary = _collector(
        config,
        lambda directory, backend: _FakeRecorder(directory, backend),
    ).collect()
    attempts_path = root / "manifests" / "attempts.jsonl"
    attempts_path.write_text("", encoding="utf-8")

    with pytest.raises(Exception, match="projection differs"):
        inspect_raw_dataset(root)

    resumed = RawDemonstrationCollector(
        config,
        recorder_factory=lambda *_: (_ for _ in ()).throw(
            AssertionError("projection repair must not recollect")
        ),
        replay_validator=_passed_replay,
    ).collect()
    assert resumed.to_dict() == summary.to_dict()
    assert inspect_raw_dataset(root).to_dict() == summary.to_dict()


def test_failed_expert_flush_error_stops_reusing_recorder(tmp_path: Path) -> None:
    root = tmp_path / "flush-failure"
    config = _config(root)
    resets = 0

    class BrokenFlushRecorder(_FakeRecorder):
        def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            nonlocal resets
            resets += 1
            return super().reset(**kwargs)

        def flush_trajectory(self, *, save: bool) -> None:
            del save
            raise OSError("synthetic HDF5 flush failure")

    summary = RawDemonstrationCollector(
        config,
        recorder_factory=lambda directory, backend: BrokenFlushRecorder(directory, backend),
        expert_factory=lambda env, cfg: SimpleFailedExpert(env),
        replay_validator=_passed_replay,
    ).collect()
    manifest = load_manifest(root)

    assert summary.status is CollectionStatus.FAILED
    assert resets == 1
    assert len(manifest.attempts) == 1
    assert manifest.attempts[0].outcome.value == "recording_failure"


def test_interrupted_initial_manifest_commit_is_safely_reinitialized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "initialization-crash"
    config = _config(root)
    import langmani.collection.manifest as manifest_module

    real_persist = manifest_module.persist_manifest
    monkeypatch.setattr(
        manifest_module,
        "persist_manifest",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("simulated initial manifest interruption")
        ),
    )
    with pytest.raises(RuntimeError, match="initial manifest interruption"):
        initialize_or_resume_manifest(config)
    assert (root / ".langmani-m3a-root").is_file()
    assert not (root / "manifests" / "collection_manifest.json").exists()

    monkeypatch.setattr(manifest_module, "persist_manifest", real_persist)
    manifest = initialize_or_resume_manifest(config)
    assert manifest.next_candidate_scene_index == 0
    assert load_manifest(root).to_dict() == manifest.to_dict()


def test_resume_finalizes_retained_failures_after_post_manifest_crash(
    tmp_path: Path,
) -> None:
    root = tmp_path / "post-manifest-retention"
    config = CollectionConfig(
        target_complete_scene_count=1,
        maximum_candidate_scene_count=1,
        maximum_expert_attempts_per_task=2,
        raw_output_root=str(root),
        shard_size=6,
        retain_failed_raw_trajectories=True,
    )
    calls: dict[str, int] = {}

    class RetryExpert:
        def __init__(self, recorder: _FakeRecorder) -> None:
            self.recorder = recorder

        def run(self) -> ExpertResult:
            task = self.recorder.task_spec
            assert task is not None
            task_id = f"{task.target_object_id}:{task.target_bin_id}"
            calls[task_id] = calls.get(task_id, 0) + 1
            if calls[task_id] == 1:
                self.recorder.step(np.zeros(8, dtype=np.float32))
                return _failed_result(self.recorder)
            return _FakeExpert(self.recorder, config).run()

    interrupted = RawDemonstrationCollector(
        config,
        recorder_factory=lambda directory, backend: _FakeRecorder(directory, backend),
        expert_factory=lambda env, cfg: RetryExpert(env),
        replay_validator=_passed_replay,
    )
    interrupted._dispose_candidate = lambda *args, **kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        RuntimeError("simulated post-manifest disposal interruption")
    )
    with pytest.raises(RuntimeError, match="post-manifest disposal interruption"):
        interrupted.collect()

    committed = load_manifest(root)
    assert committed.status is CollectionStatus.COMPLETE
    assert len(committed.attempts) == 12
    assert (root / ".staging").exists()
    assert (root / "journal" / "inflight" / "candidate-000000.json").is_file()

    resumed = RawDemonstrationCollector(
        config,
        recorder_factory=lambda *_: (_ for _ in ()).throw(
            AssertionError("committed candidate must not be recollected")
        ),
        replay_validator=_passed_replay,
    ).collect()
    failed_directories = list((root / "failed").glob("candidate-*"))

    assert resumed.status is CollectionStatus.COMPLETE
    assert len(failed_directories) == 1
    failure_index = json.loads(
        (failed_directories[0] / "failure_index.json").read_text(encoding="utf-8")
    )
    assert failure_index["accepted_scene_group"] is True
    assert not (root / ".staging").exists()
    assert not (root / "journal").exists()
    assert inspect_raw_dataset(root).accepted_episode_count == 6


def test_counterfactual_initial_state_mismatch_rejects_candidate_and_continues(
    tmp_path: Path,
) -> None:
    root = tmp_path / "counterfactual-mismatch"
    config = CollectionConfig(
        target_complete_scene_count=1,
        maximum_candidate_scene_count=2,
        maximum_expert_attempts_per_task=1,
        raw_output_root=str(root),
        shard_size=6,
    )
    recorder_count = 0

    class DriftedCounterfactualRecorder(_FakeRecorder):
        def flush_trajectory(self, *, save: bool) -> None:
            super().flush_trajectory(save=save)
            if (
                save
                and self.task_spec is not None
                and self.task_spec.target_object_id == "blue_cube"
            ):
                payload = self._payload()
                episode_id = len(payload["episodes"]) - 1
                with h5py.File(self.h5_path, "a") as archive:
                    archive[f"traj_{episode_id}/env_states/actors/red_cube"][0, 0] += 0.01

    def recorder_factory(directory: Path, backend: str) -> _FakeRecorder:
        nonlocal recorder_count
        recorder_count += 1
        if recorder_count == 1:
            return DriftedCounterfactualRecorder(directory, backend)
        return _FakeRecorder(directory, backend)

    summary = _collector(config, recorder_factory).collect()
    manifest = load_manifest(root)

    assert summary.status is CollectionStatus.COMPLETE
    assert [group.accepted for group in manifest.scene_groups] == [False, True]
    assert any(
        "initial physical state" in reason for reason in manifest.scene_groups[0].rejection_reasons
    )
    assert len(manifest.attempts) == 12
    assert inspect_raw_dataset(root).accepted_episode_count == 6


class SimpleFailedExpert:
    def __init__(self, recorder: _FakeRecorder) -> None:
        self.recorder = recorder

    def run(self) -> ExpertResult:
        return _failed_result(self.recorder)


@pytest.mark.parametrize("marker", ["forged\n", "langmani-m3a-raw-v1\n"])
def test_overwrite_refuses_forged_or_unparseable_collection_root(
    tmp_path: Path,
    marker: str,
) -> None:
    root = tmp_path / "forged-root"
    (root / "manifests").mkdir(parents=True)
    (root / ".langmani-m3a-root").write_text(marker, encoding="utf-8")
    (root / "manifests" / "collection_manifest.json").write_text("{}\n", encoding="utf-8")
    sentinel = root / "do-not-delete.txt"
    sentinel.write_text("owned by user\n", encoding="utf-8")
    config = CollectionConfig(
        target_complete_scene_count=1,
        maximum_candidate_scene_count=1,
        raw_output_root=str(root),
        shard_size=6,
        overwrite_policy=OverwritePolicy.REPLACE,
    )

    with pytest.raises(ManifestError):
        initialize_or_resume_manifest(config)
    assert sentinel.read_text(encoding="utf-8") == "owned by user\n"


def test_manifest_rejects_accepted_attempt_without_raw_episode(tmp_path: Path) -> None:
    root = tmp_path / "accepted-attempt-closure"
    config = CollectionConfig(
        target_complete_scene_count=1,
        maximum_candidate_scene_count=1,
        maximum_expert_attempts_per_task=2,
        raw_output_root=str(root),
        shard_size=6,
    )
    _collector(
        config,
        lambda directory, backend: _FakeRecorder(directory, backend),
    ).collect()
    manifest = load_manifest(root)
    original = manifest.attempts[0]
    extra_attempt_id = stable_attempt_id(
        scheduled_episode_id=original.scheduled_episode_id,
        attempt_index=2,
    )
    extra = replace(
        original,
        attempt_id=extra_attempt_id,
        attempt_index=2,
        raw_trajectory_id=stable_raw_trajectory_id(
            scheduled_episode_id=original.scheduled_episode_id,
            attempt_id=extra_attempt_id,
        ),
    )
    group = manifest.scene_groups[0]

    with pytest.raises(ValueError, match="exact one-to-one"):
        replace(
            manifest,
            attempts=manifest.attempts + (extra,),
            scene_groups=(replace(group, attempt_ids=group.attempt_ids + (extra_attempt_id,)),),
        )


def test_inspection_rejects_orphan_accepted_json_file(tmp_path: Path) -> None:
    root = tmp_path / "orphan-json"
    config = _config(root)
    _collector(
        config,
        lambda directory, backend: _FakeRecorder(directory, backend),
    ).collect()
    orphan = root / "accepted" / "shards" / "orphan.json"
    orphan.write_text("{}\n", encoding="utf-8")

    with pytest.raises(DatasetInspectionError, match="file set differs"):
        inspect_raw_dataset(root)


def test_partial_shard_repair_refuses_changed_journal_bundle(tmp_path: Path) -> None:
    root = tmp_path / "untrusted-repair"
    config = CollectionConfig(
        target_complete_scene_count=2,
        maximum_candidate_scene_count=2,
        maximum_expert_attempts_per_task=1,
        raw_output_root=str(root),
        shard_size=12,
    )
    interrupted = _collector(
        config,
        lambda directory, backend: _FakeRecorder(directory, backend),
    )
    original_collect_candidate = interrupted._collect_candidate

    def stop_before_second(manifest: Any, candidate_index: int) -> Any:
        if candidate_index == 1:
            raise RuntimeError("simulated pause after first partial-shard group")
        return original_collect_candidate(manifest, candidate_index)

    interrupted._collect_candidate = stop_before_second  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="pause after first"):
        interrupted.collect()

    manifest = load_manifest(root)
    bundle = next((root / "journal").rglob("group-*.h5"))
    with h5py.File(bundle, "a") as archive:
        archive["traj_0/actions"][0, 0] = np.float32(0.42)
    partial_shard = root / manifest.source_shards[0].h5_path
    with partial_shard.open("ab") as stream:
        stream.write(b"force-trusted-repair-path")

    with pytest.raises(CollectionError, match="untrusted or changed group bundle"):
        _collector(
            config,
            lambda directory, backend: _FakeRecorder(directory, backend),
        ).collect()


def test_normal_partial_shard_continuation_refuses_changed_prior_bundle(
    tmp_path: Path,
) -> None:
    root = tmp_path / "untrusted-continuation"
    config = CollectionConfig(
        target_complete_scene_count=2,
        maximum_candidate_scene_count=2,
        maximum_expert_attempts_per_task=1,
        raw_output_root=str(root),
        shard_size=12,
    )
    interrupted = _collector(
        config,
        lambda directory, backend: _FakeRecorder(directory, backend),
    )
    original_collect_candidate = interrupted._collect_candidate

    def stop_before_second(manifest: Any, candidate_index: int) -> Any:
        if candidate_index == 1:
            raise RuntimeError("simulated pause before continuation")
        return original_collect_candidate(manifest, candidate_index)

    interrupted._collect_candidate = stop_before_second  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="pause before continuation"):
        interrupted.collect()

    before = load_manifest(root)
    bundle = next((root / "journal").rglob("group-*.h5"))
    with h5py.File(bundle, "a") as archive:
        archive["traj_0/actions"][0, 0] = np.float32(0.42)

    with pytest.raises(CollectionError, match="untrusted or changed group bundle"):
        _collector(
            config,
            lambda directory, backend: _FakeRecorder(directory, backend),
        ).collect()
    after = load_manifest(root)

    assert after.to_dict() == before.to_dict()
    with h5py.File(root / after.source_shards[0].h5_path, "r") as archive:
        assert float(archive["traj_0/actions"][0, 0]) == pytest.approx(0.0)
