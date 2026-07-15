"""Unit contracts for M4.2 checkpoint evaluation and paired diagnostics."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from langmani.policies.act_action_bounds import (
    ActionBoundConfig,
    ActionBoundMode,
    ActionProjectionSummary,
)
from langmani.policies.act_conditioning import CANONICAL_TASK_IDS
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
    _runtime_projection_audit_count_valid,
    paired_episode_comparison,
    task_action_chunk_distances,
    task_sensitivity_report,
    task_token_validation_result,
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
        architecture_fingerprint=_DIGEST if kind is M42PolicyKind.TASK_TOKEN else None,
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
