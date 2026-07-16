"""Unit contracts for M4.2 checkpoint evaluation and paired diagnostics."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from langmani.datasets.lerobot_types import STATE_FEATURE_KEY
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import stable_task_id
from langmani.policies.act_action_bounds import (
    ActionBoundConfig,
    ActionBoundMode,
    ActionProjectionSummary,
)
from langmani.policies.act_conditioning import CANONICAL_TASK_IDS
from langmani.policies.act_factor_film_types import (
    FACTOR_FILM_BIN_INDEX_KEY,
    FACTOR_FILM_OBJECT_INDEX_KEY,
)
from langmani.policies.act_types import (
    EvaluationSplit,
    RolloutEpisodeResult,
    RolloutStatus,
)
from langmani.policies.m42_evaluation import (
    M42BenchmarkReport,
    M42CheckpointDescriptor,
    M42EpisodeReport,
    M42EvaluationError,
    M42PolicyKind,
    _DiagnosticEnvironmentProxy,
    _FactorFiLMBatchAugmenter,
    _fingerprint,
    _runtime_projection_audit_count_valid,
    _validate_episode_schedule,
    load_factor_film_training_checkpoint_context,
    paired_episode_comparison,
    run_m42_checkpoint_benchmark,
    run_m42_validation_benchmark,
    task_action_chunk_distances,
    task_sensitivity_report,
    task_token_validation_result,
)
from langmani.policies.m42_schedule import (
    M42_DEV_SCHEDULE_ID,
    M42_FINAL_SCHEDULE_ID,
    load_locked_schedule,
    materialize_locked_schedule,
)
from langmani.policies.m42_types import GripperRuntimeMode

_DIGEST = "sha256:" + "1" * 64
_OTHER_DIGEST = "sha256:" + "2" * 64


def _descriptor(kind: M42PolicyKind = M42PolicyKind.TASK_TOKEN) -> M42CheckpointDescriptor:
    return M42CheckpointDescriptor(
        policy_kind=kind,
        run_fingerprint=_DIGEST,
        checkpoint_fingerprint=_OTHER_DIGEST,
        checkpoint_relative_path="checkpoints/step-005000",
        dataset_fingerprint=_DIGEST,
        split_digest=_DIGEST,
        statistics_fingerprint=_DIGEST,
        architecture_fingerprint=(
            _DIGEST if kind in {M42PolicyKind.TASK_TOKEN, M42PolicyKind.FACTOR_FILM} else None
        ),
        task_id=CANONICAL_TASK_IDS[0] if kind is M42PolicyKind.PER_TASK else None,
        git_commit="a" * 40,
    )


def _rollout(*, success: bool, scene_id: str = "scene-1") -> RolloutEpisodeResult:
    summary = ActionProjectionSummary.from_records((), action_dimension=8, total_policy_actions=1)
    return RolloutEpisodeResult(
        evaluation_id="evaluation",
        run_fingerprint=_DIGEST,
        checkpoint_fingerprint=_OTHER_DIGEST,
        schedule_digest=_DIGEST,
        split=EvaluationSplit.FRESH_SEED,
        scene_seed=101,
        scene_id=scene_id,
        task_id=CANONICAL_TASK_IDS[0],
        status=RolloutStatus.SUCCESS if success else RolloutStatus.TIMEOUT,
        success=success,
        episode_steps=10,
        final_evaluation={"success": success, "target_is_static": success},
        target_in_target_bin=success,
        target_in_wrong_bin=False,
        wrong_object_in_target_bin=False,
        target_grasped_any=True,
        wrong_object_grasped_any=False,
        target_off_table=False,
        timeout=not success,
        invalid_action=False,
        inference_failure=False,
        time_to_first_target_grasp_s=0.1,
        time_to_release_s=0.3,
        time_to_success_s=0.5 if success else None,
        cumulative_action_magnitude=1.0,
        action_min=(0.0,) * 8,
        action_max=(0.0,) * 8,
        inference_latency_ms=(1.0,),
        environment_step_latency_ms=(2.0,),
        total_episode_duration_s=0.5,
        runtime_fingerprint=_DIGEST,
        action_bound_mode=ActionBoundMode.PROJECT,
        task_success=success,
        strict_unprojected_success=success,
        action_projection_summary=summary.to_dict(),
    )


def _episode(*, success: bool, raw_gripper_violation: bool = False) -> M42EpisodeReport:
    raw_audit = {
        "violation_mask": [[False] * 7 + [raw_gripper_violation]],
        "was_projected": raw_gripper_violation,
        "projected_component_count": int(raw_gripper_violation),
        "maximum_absolute_bound_excess": 0.2 if raw_gripper_violation else 0.0,
    }
    binary_audits = (
        {
            "raw_bound_audit": raw_audit,
            "changed": True,
            "changed_command_count": 1,
            "maximum_absolute_gripper_change": 0.3,
        },
    )
    return M42EpisodeReport(
        scheduled_episode_id="episode-1",
        schedule_id="m3b_validation_v0",
        episode_index=0,
        model_label="task_token",
        rollout=_rollout(success=success),
        execution_horizon_metrics={"policy_query_count": 2},
        post_grasp_diagnostics={
            "phase": "success" if success else "timed_out_after_target_grasp",
            "release_sign_transitions": 1,
            "grasp_sign_transitions": 1,
        },
        binary_gripper_audits=binary_audits,
        post_grasp_failure_record=None,
        runtime_projection_audits=(
            {
                "raw_action": [[0.0] * 7 + [1.0]],
                "executed_action": [[0.0] * 7 + [1.0]],
                "was_projected": False,
                "projected_component_count": 0,
            },
        ),
    )


def _report(*, success: bool, model_label: str = "task_token") -> M42BenchmarkReport:
    episode = _episode(success=success)
    aggregate = {
        "episode_count": 36,
        "successes": 36 if success else 0,
        "wrong_object_interaction_count": 0,
        "target_off_table_count": 0,
        "timeout_count": 0 if success else 36,
    }
    return M42BenchmarkReport(
        schema_version="langmani-m42-evaluation-v0",
        schedule_id="m3b_validation_v0",
        schedule_fingerprint=_DIGEST,
        model_label=model_label,
        checkpoint=_descriptor(),
        execution_horizon=5,
        gripper_mode=GripperRuntimeMode.BINARY,
        runtime_fingerprint=_DIGEST,
        episodes=(episode,) * 36,
        aggregate=aggregate,
    )


def test_checkpoint_descriptor_separates_policy_kinds_and_task_binding() -> None:
    assert _descriptor(M42PolicyKind.PER_TASK).task_id == CANONICAL_TASK_IDS[0]
    with pytest.raises(M42EvaluationError, match="PerTask"):
        replace(_descriptor(), policy_kind=M42PolicyKind.PER_TASK)
    with pytest.raises(M42EvaluationError, match="mixed"):
        replace(_descriptor(), task_id=CANONICAL_TASK_IDS[0])


def test_task_action_chunk_distance_is_deterministic_and_six_task_complete() -> None:
    chunks = {
        task_id: torch.full((1, 50, 8), float(index), dtype=torch.float32)
        for index, task_id in enumerate(CANONICAL_TASK_IDS)
    }
    report = task_action_chunk_distances(chunks)
    assert report["pair_count"] == 15
    assert report["maximum_pair_distance"] == pytest.approx(5.0)
    assert task_action_chunk_distances(chunks) == report
    with pytest.raises(M42EvaluationError, match="exactly"):
        task_action_chunk_distances(dict(list(chunks.items())[:-1]))


def test_task_sensitivity_ratio_uses_per_task_oracle_distance() -> None:
    oracle = {
        task_id: np.full((50, 8), float(index), dtype=np.float32)
        for index, task_id in enumerate(CANONICAL_TASK_IDS)
    }
    conditioned = {task_id: value * 0.5 for task_id, value in oracle.items()}
    report = task_sensitivity_report(
        conditioned_chunks_by_task=conditioned,
        per_task_oracle_chunks_by_task=oracle,
    )
    assert report["task_sensitivity_ratio"] == pytest.approx(0.5)


def test_binary_raw_action_metrics_remain_separate_from_executed_projection() -> None:
    from langmani.policies.m42_evaluation import _aggregate

    aggregate = _aggregate((_episode(success=False, raw_gripper_violation=True),))
    action = aggregate["action_metrics"]
    assert action["raw_out_of_bounds_component_count"] == 1
    assert action["raw_per_action_dimension_violation_counts"] == [0] * 7 + [1]
    assert action["raw_action_bounds_validated"] is False
    assert action["raw_action_metrics_validated"] is True
    assert action["runtime_action_bounds_validated"] is True
    assert action["projected_component_count"] == 0
    assert action["runtime_projected_action_count"] == action["projected_action_count"]
    assert action["runtime_projected_component_count"] == action["projected_component_count"]
    assert aggregate["unnecessary_gripper_sign_transitions"] == 0

    successful = _aggregate((_episode(success=True, raw_gripper_violation=True),))
    assert successful["strict_unprojected_success_count"] == 0
    assert successful["strict_runtime_unprojected_success_count"] == 1
    assert successful["strict_raw_unmodified_success_count"] == 0
    serialized = _episode(success=True).to_dict()
    assert serialized["binary_gripper_audits"][0]["raw_bound_audit"] == {
        "violation_mask": [[False] * 8],
        "was_projected": False,
        "projected_component_count": 0,
        "maximum_absolute_bound_excess": 0.0,
    }
    assert serialized["runtime_projection_audits"][0]["executed_action"] == [[0.0] * 7 + [1.0]]


def test_task_token_validation_projection_is_validation_only() -> None:
    result = task_token_validation_result(
        _report(success=True),
        checkpoint_step=5_000,
        validation_split_digest=_OTHER_DIGEST,
        offline_validation_action_loss=0.25,
    )
    assert result.episode_count == 36
    assert result.success_count == 36
    assert result.development_schedule_accessed is False
    assert result.final_schedule_accessed is False
    assert result.test_split_accessed is False


def test_paired_comparison_rejects_different_schedule_and_tracks_discordance() -> None:
    left = _report(success=True, model_label="left")
    right = _report(success=False, model_label="right")
    paired = paired_episode_comparison(left, right)
    assert paired["left_only_successes"] == 1
    assert paired["right_only_successes"] == 0
    with pytest.raises(M42EvaluationError, match="schedule"):
        paired_episode_comparison(
            left,
            replace(right, schedule_fingerprint=_OTHER_DIGEST),
        )


def test_runtime_audit_may_be_one_short_only_for_classified_pre_bound_failure() -> None:
    summary = ActionProjectionSummary.from_records(
        (),
        action_dimension=8,
        total_policy_actions=3,
        any_nonfinite_action=True,
    )
    classified = replace(
        _rollout(success=False),
        status=RolloutStatus.INVALID_ACTION,
        timeout=False,
        invalid_action=True,
        failure_reason="classified nonfinite action",
        action_projection_summary=summary.to_dict(),
    )
    assert _runtime_projection_audit_count_valid(classified, 3)
    assert _runtime_projection_audit_count_valid(classified, 2)
    assert not _runtime_projection_audit_count_valid(classified, 1)

    unclassified = replace(
        classified,
        action_projection_summary=ActionProjectionSummary.from_records(
            (), action_dimension=8, total_policy_actions=3
        ).to_dict(),
    )
    assert not _runtime_projection_audit_count_valid(unclassified, 2)


def test_action_bound_configuration_is_explicit_project_not_hidden_clip() -> None:
    assert ActionBoundConfig(mode=ActionBoundMode.PROJECT).mode is ActionBoundMode.PROJECT


def test_factor_film_batch_augmenter_injects_separate_semantic_indices() -> None:
    batch = {STATE_FEATURE_KEY: torch.zeros((1, 9), dtype=torch.float32)}
    augmented = _FactorFiLMBatchAugmenter()(batch, CANONICAL_TASK_IDS[-1])
    assert augmented[FACTOR_FILM_OBJECT_INDEX_KEY].tolist() == [2]
    assert augmented[FACTOR_FILM_BIN_INDEX_KEY].tolist() == [1]
    assert tuple(augmented[STATE_FEATURE_KEY].shape) == (1, 9)
    assert FACTOR_FILM_OBJECT_INDEX_KEY not in batch

    with pytest.raises(M42EvaluationError, match="invalid FactorFiLM rollout task"):
        _FactorFiLMBatchAugmenter()(batch, "not-a-stable-task")


def test_diagnostic_proxy_captures_complete_semantic_interaction_snapshots() -> None:
    class _FakeBase:
        def get_policy_rollout_diagnostics(self) -> dict[str, torch.Tensor]:
            return {
                "target_position": torch.tensor([[0.1, 0.2, 0.3]]),
                "target_linear_velocity": torch.zeros((1, 3)),
                "target_bin_floor_center": torch.tensor([[0.4, 0.5, 0.0]]),
                "target_to_bin_center_distance": torch.tensor([0.5]),
                "object_is_grasped": torch.tensor([[False, True, False]]),
                "cube_positions": torch.arange(9, dtype=torch.float32).reshape(1, 3, 3),
                "cube_orientations": torch.tensor(
                    [[[1.0, 0.0, 0.0, 0.0]] * 3], dtype=torch.float32
                ),
                "cube_linear_velocities": torch.zeros((1, 3, 3)),
                "cube_angular_velocities": torch.ones((1, 3, 3)),
                "bin_floor_centers": torch.zeros((1, 2, 3)),
                "tcp_position": torch.tensor([[0.2, 0.1, 0.4]]),
                "object_in_bin": torch.tensor([[[False, False]] * 3]),
            }

        def get_policy_rollout_evaluation(self) -> dict[str, torch.Tensor]:
            return {"success": torch.tensor([False])}

    class _FakeEnv:
        def __init__(self) -> None:
            self.unwrapped = _FakeBase()

        def reset(
            self, *args: object, **kwargs: object
        ) -> tuple[dict[str, object], dict[str, object]]:
            del args, kwargs
            return {}, {}

        def step(
            self, action: object
        ) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
            del action
            return {}, 0.0, False, False, {}

    proxy = _DiagnosticEnvironmentProxy(_FakeEnv(), capture_semantic_interactions=True)
    proxy.reset(seed=7)
    proxy.step(np.zeros(8, dtype=np.float32))
    assert len(proxy.snapshots) == 2
    reset_snapshot = proxy.snapshots[0].semantic_interaction_dict()
    assert reset_snapshot["cube_positions"] == [
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
    ]
    assert reset_snapshot["cube_orientations"] == [[1.0, 0.0, 0.0, 0.0]] * 3
    assert reset_snapshot["cube_angular_velocities"] == [[1.0, 1.0, 1.0]] * 3
    assert reset_snapshot["executed_action"] is None
    assert proxy.snapshots[1].semantic_interaction_dict()["executed_action"] == [0.0] * 8

    uncaptured = _DiagnosticEnvironmentProxy(_FakeEnv())
    uncaptured.reset(seed=7)
    with pytest.raises(M42EvaluationError, match="not authorized"):
        uncaptured.snapshots[0].semantic_interaction_dict()


def test_factor_film_validation_uses_its_immutable_schedule_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records: list[dict[str, object]] = []
    for scene_index in range(6):
        for task_spec in CANONICAL_TASK_SPECS:
            records.append(
                {
                    "episode_index": len(records) + 100,
                    "scene_group_id": f"group-{scene_index}",
                    "scene_seed": 1_000 + scene_index,
                    "task_id": stable_task_id(task_spec),
                }
            )
    descriptor = _descriptor(M42PolicyKind.FACTOR_FILM)
    checkpoint = SimpleNamespace(descriptor=descriptor)
    schedule_fingerprint = _fingerprint(
        {
            "schema_version": "langmani-m43-factor-film-validation-schedule-v0",
            "m3b_export_fingerprint": descriptor.dataset_fingerprint,
            "split_manifest_digest": descriptor.split_digest,
            "source": "m3b_validation",
            "episodes": records,
        }
    )
    captured: dict[str, object] = {}

    def _fake_benchmark(**kwargs: object) -> object:
        captured.update(kwargs)
        return "factor-validation-report"

    monkeypatch.setattr(
        "langmani.policies.m42_evaluation._run_checkpoint_benchmark",
        _fake_benchmark,
    )
    result = run_m42_validation_benchmark(
        env=object(),
        checkpoint=cast(Any, checkpoint),
        schedule_records=records,
        validation_schedule_fingerprint=schedule_fingerprint,
        execution_horizon=10,
        gripper_mode=GripperRuntimeMode.PROJECT,
    )
    assert result == "factor-validation-report"
    episodes = cast(tuple[object, ...], captured["episodes"])
    assert len(episodes) == 36
    assert captured["model_label"] == M42PolicyKind.FACTOR_FILM.value

    malformed = [dict(item) for item in records]
    malformed[0]["task_spec"] = CANONICAL_TASK_SPECS[0].to_dict()
    malformed_fingerprint = _fingerprint(
        {
            "schema_version": "langmani-m43-factor-film-validation-schedule-v0",
            "m3b_export_fingerprint": descriptor.dataset_fingerprint,
            "split_manifest_digest": descriptor.split_digest,
            "source": "m3b_validation",
            "episodes": malformed,
        }
    )
    with pytest.raises(M42EvaluationError, match="exact schedule fields"):
        run_m42_validation_benchmark(
            env=object(),
            checkpoint=cast(Any, checkpoint),
            schedule_records=malformed,
            validation_schedule_fingerprint=malformed_fingerprint,
            execution_horizon=10,
            gripper_mode=GripperRuntimeMode.PROJECT,
        )


def test_factor_film_is_a_complete_mixed_72_episode_development_policy() -> None:
    schedule = load_locked_schedule(M42_DEV_SCHEDULE_ID)
    episodes = materialize_locked_schedule(M42_DEV_SCHEDULE_ID)
    assert len(episodes) == 72
    assert (
        _validate_episode_schedule(
            episodes,
            schedule_fingerprint=schedule.schedule_fingerprint,
            policy_kind=M42PolicyKind.FACTOR_FILM,
            task_id=None,
            final_authorization=None,
        )
        == M42_DEV_SCHEDULE_ID
    )
    prohibited = tuple(replace(item, schedule_id=M42_FINAL_SCHEDULE_ID) for item in episodes)
    with pytest.raises(M42EvaluationError, match="only the locked m42_dev_v0"):
        _validate_episode_schedule(
            prohibited,
            schedule_fingerprint=_DIGEST,
            policy_kind=M42PolicyKind.FACTOR_FILM,
            task_id=None,
            final_authorization=cast(Any, object()),
        )


def test_development_split_is_not_mislabeled_as_historical_fresh_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("langmani.policies.m42_evaluation._run_checkpoint_benchmark", fake_run)
    schedule = load_locked_schedule(M42_DEV_SCHEDULE_ID)
    episodes = materialize_locked_schedule(M42_DEV_SCHEDULE_ID)
    checkpoint = SimpleNamespace(
        descriptor=SimpleNamespace(policy_kind=M42PolicyKind.FACTOR_FILM, task_id=None)
    )
    run_m42_checkpoint_benchmark(
        env=object(),
        checkpoint=cast(Any, checkpoint),
        episodes=episodes,
        schedule_fingerprint=schedule.schedule_fingerprint,
        execution_horizon=10,
        gripper_mode=GripperRuntimeMode.PROJECT,
        model_label="factor_film",
    )
    assert captured["split"] is EvaluationSplit.DEVELOPMENT


def test_factor_film_loader_uses_manifest_identity_and_strict_policy_class(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import langmani.policies.act_factor_film_adapter as adapter_module
    import langmani.policies.act_factor_film_types as types_module

    class _FakeFactorFiLMPolicy:
        pass

    identity = SimpleNamespace(
        run_fingerprint=_DIGEST,
        m3b_export_fingerprint=_OTHER_DIGEST,
        m3b_split_manifest_digest=_DIGEST,
        train_statistics_fingerprint=_OTHER_DIGEST,
        git_commit="a" * 40,
    )
    record = SimpleNamespace(
        checkpoint_fingerprint=_OTHER_DIGEST,
        relative_path="checkpoints/step-100000",
    )
    manifest = SimpleNamespace(
        training_complete=True,
        identity=identity,
        architecture_identity=SimpleNamespace(architecture_fingerprint=_DIGEST),
        checkpoints=(record,),
    )
    monkeypatch.setattr(
        types_module.FactorFiLMTrainingManifest,
        "from_dict",
        classmethod(lambda cls, value: manifest),
    )
    monkeypatch.setattr(adapter_module, "FactorFiLMACTPolicy", _FakeFactorFiLMPolicy)
    validated: list[object] = []
    monkeypatch.setattr(
        adapter_module,
        "validate_factor_film_policy_structure",
        lambda policy: validated.append(policy),
    )
    load_arguments: dict[str, object] = {}
    policy = _FakeFactorFiLMPolicy()

    def _fake_load_act_checkpoint(**kwargs: object) -> object:
        load_arguments.update(kwargs)
        return SimpleNamespace(
            record=record,
            policy=policy,
            preprocessor=object(),
            postprocessor=object(),
        )

    monkeypatch.setattr(
        "langmani.policies.m42_evaluation.load_act_checkpoint",
        _fake_load_act_checkpoint,
    )
    (tmp_path / "run_manifest.json").write_text("{}\n", encoding="utf-8")
    context = load_factor_film_training_checkpoint_context(
        tmp_path,
        expected_checkpoint_fingerprint=_OTHER_DIGEST,
        expected_run_fingerprint=_DIGEST,
        expected_dataset_fingerprint=_OTHER_DIGEST,
        expected_architecture_fingerprint=_DIGEST,
    )
    assert context.descriptor.policy_kind is M42PolicyKind.FACTOR_FILM
    assert context.descriptor.checkpoint_relative_path == "checkpoints/step-100000"
    assert load_arguments["policy_class"] is _FakeFactorFiLMPolicy
    assert load_arguments["restore_rng"] is False
    assert validated == [policy]
