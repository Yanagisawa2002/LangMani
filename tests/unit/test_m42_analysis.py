from __future__ import annotations

from dataclasses import replace

import pytest

from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import stable_task_id
from langmani.policies.m42_analysis import (
    PostGraspTrace,
    binary_gripper_qualifies,
    classify_post_grasp_phase,
    create_go_no_go_decision,
    create_task_token_checkpoint_selection,
    paired_success_metrics,
    rank_execution_horizons,
    select_gripper_runtime,
    task_sensitivity_ratio,
)
from langmani.policies.m42_types import (
    GripperRuntimeMode,
    M42Decision,
    M42GoNoGoMetrics,
    PostGraspPhase,
    RuntimeAblationResult,
    TaskTokenValidationResult,
)

DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64
REPRESENTATIVE_TASK = stable_task_id(CANONICAL_TASK_SPECS[2])


def _result(
    *,
    horizon: int,
    mode: GripperRuntimeMode = GripperRuntimeMode.PROJECT,
    successes: int = 5,
    timeouts: int = 2,
    wrong: int = 0,
    per_task: bool = False,
    queries: int = 20,
    transitions: int = 1,
) -> RuntimeAblationResult:
    episodes = 12 if per_task else 72
    return RuntimeAblationResult(
        config_fingerprint=DIGEST_A,
        schedule_id="m42_dev_v0",
        schedule_fingerprint=DIGEST_B,
        model_label="representative_per_task" if per_task else "mixed_task_onehot",
        task_id=REPRESENTATIVE_TASK if per_task else None,
        execution_horizon=horizon,
        gripper_mode=mode,
        episode_count=episodes,
        successes=successes,
        post_grasp_timeouts=timeouts,
        wrong_object_interactions=wrong,
        wrong_object_grasp_count=wrong,
        wrong_object_in_target_bin_count=0,
        target_in_wrong_bin_count=0,
        target_off_table_count=0,
        invalid_action_count=0,
        successful_episode_steps=tuple(range(1, successes + 1)),
        policy_query_count=queries,
        release_sign_transitions=transitions,
        grasp_sign_transitions=transitions,
        unnecessary_gripper_sign_transitions=max(0, transitions - 1) * 2,
    )


def test_horizon_ranking_uses_declared_order_and_larger_final_tie_break() -> None:
    mixed = tuple(_result(horizon=value) for value in (10, 5, 1))
    per_task = tuple(_result(horizon=value, per_task=True) for value in (10, 5, 1))
    assert rank_execution_horizons(
        mixed_results=mixed, representative_per_task_results=per_task
    ) == (10, 5, 1)

    improved = replace(mixed[1], successes=6, successful_episode_steps=(1, 2, 3, 4, 5, 6))
    assert (
        rank_execution_horizons(
            mixed_results=(mixed[0], improved, mixed[2]),
            representative_per_task_results=per_task,
        )[0]
        == 5
    )


def test_horizon_selection_rejects_binary_or_incomplete_evidence() -> None:
    mixed = tuple(_result(horizon=value) for value in (10, 5, 1))
    per_task = tuple(_result(horizon=value, per_task=True) for value in (10, 5, 1))
    with pytest.raises(ValueError, match="exactly horizons"):
        rank_execution_horizons(mixed_results=mixed[:2], representative_per_task_results=per_task)
    with pytest.raises(ValueError, match="project runtime"):
        rank_execution_horizons(
            mixed_results=(replace(mixed[0], gripper_mode=GripperRuntimeMode.BINARY),) + mixed[1:],
            representative_per_task_results=per_task,
        )


def test_binary_gripper_requires_declared_gain_without_safety_regression() -> None:
    project = _result(horizon=5, successes=20, timeouts=20)
    binary = _result(horizon=5, mode=GripperRuntimeMode.BINARY, successes=24, timeouts=14)
    per_project = _result(horizon=5, successes=5, per_task=True)
    per_binary = _result(horizon=5, mode=GripperRuntimeMode.BINARY, successes=5, per_task=True)
    assert binary_gripper_qualifies(
        mixed_project=project,
        mixed_binary=binary,
        per_task_project=per_project,
        per_task_binary=per_binary,
    )
    assert (
        select_gripper_runtime(
            mixed_results=(project, binary),
            representative_per_task_results=(per_project, per_binary),
        )
        is GripperRuntimeMode.BINARY
    )
    unsafe = replace(binary, wrong_object_interactions=1, wrong_object_grasp_count=1)
    assert not binary_gripper_qualifies(
        mixed_project=project,
        mixed_binary=unsafe,
        per_task_project=per_project,
        per_task_binary=per_binary,
    )
    assert (
        select_gripper_runtime(
            mixed_results=(project, unsafe),
            representative_per_task_results=(per_project, per_binary),
        )
        is GripperRuntimeMode.PROJECT
    )
    wrong_bin = replace(binary, target_in_wrong_bin_count=1)
    assert not binary_gripper_qualifies(
        mixed_project=project,
        mixed_binary=wrong_bin,
        per_task_project=per_project,
        per_task_binary=per_binary,
    )


def test_gripper_ranking_counts_only_transitions_beyond_one_pair_per_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import langmani.policies.m42_analysis as analysis

    project = _result(horizon=5, successes=20, timeouts=20)
    binary = _result(
        horizon=5,
        mode=GripperRuntimeMode.BINARY,
        successes=24,
        timeouts=14,
    )
    per_project = _result(horizon=5, successes=5, per_task=True)
    per_binary = _result(
        horizon=5,
        mode=GripperRuntimeMode.BINARY,
        successes=5,
        per_task=True,
    )
    assert binary.unnecessary_gripper_sign_transitions == 0
    noisy_binary = replace(binary, unnecessary_gripper_sign_transitions=1)
    tied_project = replace(
        project,
        successes=24,
        post_grasp_timeouts=14,
        successful_episode_steps=tuple(range(1, 25)),
    )
    monkeypatch.setattr(analysis, "binary_gripper_qualifies", lambda **_kwargs: True)
    assert (
        select_gripper_runtime(
            mixed_results=(tied_project, noisy_binary),
            representative_per_task_results=(per_project, per_binary),
        )
        is GripperRuntimeMode.PROJECT
    )


@pytest.mark.parametrize(
    ("changes", "expected"),
    (
        ({"success": True}, PostGraspPhase.SUCCESS),
        ({"environment_failure": True}, PostGraspPhase.ENVIRONMENT_FAILURE),
        ({"wrong_object_interaction": True}, PostGraspPhase.WRONG_OBJECT_INTERACTION),
        (
            {"success_observed_before_final_step": True},
            PostGraspPhase.TARGET_SUCCESS_THEN_LOST,
        ),
        ({}, PostGraspPhase.NEVER_GRASPED_TARGET),
        (
            {"ever_grasped_target": True, "timed_out": True},
            PostGraspPhase.TIMED_OUT_AFTER_TARGET_GRASP,
        ),
        (
            {"ever_grasped_target": True},
            PostGraspPhase.TARGET_GRASPED_NOT_LIFTED,
        ),
        (
            {"ever_grasped_target": True, "ever_lifted_target": True},
            PostGraspPhase.LIFTED_NOT_TRANSPORTED,
        ),
        (
            {
                "ever_grasped_target": True,
                "ever_lifted_target": True,
                "ever_transported_to_destination": True,
            },
            PostGraspPhase.TRANSPORTED_NOT_DESCENDED,
        ),
        (
            {
                "ever_grasped_target": True,
                "ever_lifted_target": True,
                "ever_transported_to_destination": True,
                "ever_descended_at_destination": True,
            },
            PostGraspPhase.DESCENDED_NOT_RELEASED,
        ),
        (
            {
                "ever_grasped_target": True,
                "ever_lifted_target": True,
                "ever_transported_to_destination": True,
                "ever_descended_at_destination": True,
                "release_command_observed": True,
            },
            PostGraspPhase.RELEASED_OUTSIDE_SUCCESS_REGION,
        ),
        (
            {
                "ever_grasped_target": True,
                "ever_lifted_target": True,
                "ever_transported_to_destination": True,
                "ever_descended_at_destination": True,
                "release_command_observed": True,
                "released_inside_success_region": True,
            },
            PostGraspPhase.RELEASED_BUT_NOT_STATIC,
        ),
    ),
)
def test_post_grasp_phase_classification(
    changes: dict[str, bool], expected: PostGraspPhase
) -> None:
    trace = PostGraspTrace(
        success=False,
        environment_failure=False,
        wrong_object_interaction=False,
        ever_grasped_target=False,
        ever_lifted_target=False,
        ever_transported_to_destination=False,
        ever_descended_at_destination=False,
        release_command_observed=False,
        released_inside_success_region=False,
        final_target_static=False,
        success_observed_before_final_step=False,
        timed_out=False,
    )
    assert classify_post_grasp_phase(replace(trace, **changes)) is expected


def test_paired_metrics_and_task_sensitivity_ratio() -> None:
    metrics = paired_success_metrics((True, True, False), (True, False, True))
    assert metrics["both_successes"] == 1
    assert metrics["left_only_successes"] == 1
    assert metrics["right_only_successes"] == 1
    assert task_sensitivity_ratio(policy_distance=0.3, per_task_oracle_distance=0.5) == 0.6
    with pytest.raises(ValueError, match="positive"):
        task_sensitivity_ratio(policy_distance=0.0, per_task_oracle_distance=0.0)


def _gate_metrics(*, successes: int = 126, arm_projection_count: int = 0) -> M42GoNoGoMetrics:
    return M42GoNoGoMetrics(
        task_token_successes=successes,
        task_token_trials=180,
        per_task_aggregate_successes=143,
        per_task_aggregate_trials=180,
        task_token_successes_by_task=(21, 21, 21, 21, 21, 21),
        timeout_count=54,
        wrong_object_grasp_count=14,
        wrong_object_in_target_bin_count=5,
        target_in_wrong_bin_count=0,
        target_off_table_count=0,
        arm_projection_count=arm_projection_count,
        nan_count=0,
        inf_count=0,
        malformed_action_count=0,
        task_sensitivity_ratio=0.75,
    )


def test_go_no_go_exact_boundary_and_one_failed_threshold() -> None:
    passed = create_go_no_go_decision(metrics=_gate_metrics(), final_benchmark_fingerprint=DIGEST_A)
    assert passed.decision is M42Decision.GO_FOR_SMOLVLA
    assert passed.failed_thresholds == ()

    failed = create_go_no_go_decision(
        metrics=_gate_metrics(arm_projection_count=1), final_benchmark_fingerprint=DIGEST_A
    )
    assert failed.decision is M42Decision.REMAIN_IN_ORACLE_CONTROL_LAYER
    assert failed.failed_thresholds == ("arm_projection_zero",)


def test_task_token_checkpoint_selection_uses_validation_only_ranking() -> None:
    values = tuple(
        TaskTokenValidationResult(
            checkpoint_fingerprint="sha256:" + f"{index:064x}",
            checkpoint_step=index * 5_000,
            validation_schedule_fingerprint=DIGEST_A,
            validation_split_digest=DIGEST_B,
            episode_count=36,
            success_count=10 + (1 if index == 7 else 0),
            wrong_object_interaction_count=0,
            target_off_table_count=0,
            timeout_count=26 - (1 if index == 7 else 0),
            offline_validation_action_loss=1.0 / index,
        )
        for index in range(1, 21)
    )
    selection = create_task_token_checkpoint_selection(values, locked_at_utc="2026-07-15T00:00:00Z")
    assert selection.selected_value == "sha256:" + f"{7:064x}"
    assert selection.development_schedule_fingerprint is None
    assert selection.validation_split_digest == DIGEST_B

    with pytest.raises(ValueError, match="validation split"):
        replace(values[0], development_schedule_accessed=True)
