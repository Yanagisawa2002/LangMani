"""Pure M4.2 ranking, post-grasp analysis, and final gate calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median

from langmani.datasets.identity import sha256_hex
from langmani.policies.m42_types import (
    GripperRuntimeMode,
    M42Decision,
    M42GoNoGoDecision,
    M42GoNoGoMetrics,
    M42SelectionKind,
    M42SelectionRecord,
    PostGraspPhase,
    RuntimeAblationResult,
    TaskTokenValidationResult,
)

HORIZON_RANKING_VERSION = "M42ExecutionHorizonRankingV0"
GRIPPER_RANKING_VERSION = "M42GripperRuntimeRankingV0"
GO_NO_GO_VERSION = "M42SmolVLAGoNoGoV0"
TASK_TOKEN_CHECKPOINT_RANKING_VERSION = "M42TaskTokenValidationRankingV0"


class M42AnalysisError(ValueError):
    """Raised when comparison evidence is incomplete or not comparable."""


@dataclass(frozen=True, slots=True)
class PostGraspTrace:
    """Compact privileged diagnostic trace; never a policy observation."""

    success: bool
    environment_failure: bool
    wrong_object_interaction: bool
    ever_grasped_target: bool
    ever_lifted_target: bool
    ever_transported_to_destination: bool
    ever_descended_at_destination: bool
    release_command_observed: bool
    released_inside_success_region: bool
    final_target_static: bool
    success_observed_before_final_step: bool
    timed_out: bool


def classify_post_grasp_phase(trace: PostGraspTrace) -> PostGraspPhase:
    """Classify one rollout with a fixed precedence that preserves safety events."""
    if not isinstance(trace, PostGraspTrace):
        raise TypeError("trace must be a PostGraspTrace")
    if trace.success:
        return PostGraspPhase.SUCCESS
    if trace.environment_failure:
        return PostGraspPhase.ENVIRONMENT_FAILURE
    if trace.wrong_object_interaction:
        return PostGraspPhase.WRONG_OBJECT_INTERACTION
    if trace.success_observed_before_final_step:
        return PostGraspPhase.TARGET_SUCCESS_THEN_LOST
    if not trace.ever_grasped_target:
        return PostGraspPhase.NEVER_GRASPED_TARGET
    if trace.timed_out:
        return PostGraspPhase.TIMED_OUT_AFTER_TARGET_GRASP
    if not trace.ever_lifted_target:
        return PostGraspPhase.TARGET_GRASPED_NOT_LIFTED
    if not trace.ever_transported_to_destination:
        return PostGraspPhase.LIFTED_NOT_TRANSPORTED
    if not trace.ever_descended_at_destination:
        return PostGraspPhase.TRANSPORTED_NOT_DESCENDED
    if not trace.release_command_observed:
        return PostGraspPhase.DESCENDED_NOT_RELEASED
    if not trace.released_inside_success_region:
        return PostGraspPhase.RELEASED_OUTSIDE_SUCCESS_REGION
    if not trace.final_target_static:
        return PostGraspPhase.RELEASED_BUT_NOT_STATIC
    return PostGraspPhase.RELEASED_OUTSIDE_SUCCESS_REGION


def _evidence_digest(result: RuntimeAblationResult) -> str:
    return result.report_fingerprint or f"sha256:{sha256_hex(result.to_dict())}"


def _by_horizon(
    values: tuple[RuntimeAblationResult, ...], *, label: str
) -> dict[int, RuntimeAblationResult]:
    result: dict[int, RuntimeAblationResult] = {}
    for value in values:
        if value.execution_horizon in result:
            raise M42AnalysisError(f"{label} contains duplicate horizon evidence")
        if value.gripper_mode is not GripperRuntimeMode.PROJECT:
            raise M42AnalysisError("horizon selection must use project runtime only")
        result[value.execution_horizon] = value
    if set(result) != {1, 5, 10}:
        raise M42AnalysisError(f"{label} must contain exactly horizons 1, 5, and 10")
    return result


def rank_execution_horizons(
    *,
    mixed_results: tuple[RuntimeAblationResult, ...],
    representative_per_task_results: tuple[RuntimeAblationResult, ...],
) -> tuple[int, ...]:
    """Rank the three horizons using only the predeclared development ordering."""
    mixed = _by_horizon(mixed_results, label="mixed_results")
    per_task = _by_horizon(representative_per_task_results, label="per_task_results")
    for horizon in (1, 5, 10):
        left, right = mixed[horizon], per_task[horizon]
        if (
            left.schedule_fingerprint != right.schedule_fingerprint
            or left.config_fingerprint != right.config_fingerprint
        ):
            raise M42AnalysisError("horizon evidence does not share one development contract")

    def key(horizon: int) -> tuple[float, float, float, float, float, int, int]:
        primary = mixed[horizon]
        reference = per_task[horizon]
        return (
            -primary.success_rate,
            primary.post_grasp_timeout_rate,
            primary.wrong_object_interaction_rate,
            -reference.success_rate,
            primary.median_successful_steps,
            primary.policy_query_count,
            -horizon,
        )

    return tuple(sorted((1, 5, 10), key=key))


def create_horizon_selection(
    *,
    mixed_results: tuple[RuntimeAblationResult, ...],
    representative_per_task_results: tuple[RuntimeAblationResult, ...],
    locked_at_utc: str,
) -> M42SelectionRecord:
    ranking = rank_execution_horizons(
        mixed_results=mixed_results,
        representative_per_task_results=representative_per_task_results,
    )
    all_results = mixed_results + representative_per_task_results
    return M42SelectionRecord(
        selection_kind=M42SelectionKind.EXECUTION_HORIZON,
        selected_value=str(ranking[0]),
        candidate_values=tuple(str(value) for value in ranking),
        ranking_version=HORIZON_RANKING_VERSION,
        evidence_fingerprints=tuple(_evidence_digest(value) for value in all_results),
        development_schedule_fingerprint=all_results[0].schedule_fingerprint,
        locked_at_utc=locked_at_utc,
    )


def _by_gripper(
    values: tuple[RuntimeAblationResult, ...], *, label: str
) -> dict[GripperRuntimeMode, RuntimeAblationResult]:
    result: dict[GripperRuntimeMode, RuntimeAblationResult] = {}
    for value in values:
        if value.gripper_mode in result:
            raise M42AnalysisError(f"{label} contains duplicate gripper evidence")
        result[value.gripper_mode] = value
    if set(result) != {GripperRuntimeMode.PROJECT, GripperRuntimeMode.BINARY}:
        raise M42AnalysisError(f"{label} must compare exactly project and binary")
    if len({value.execution_horizon for value in result.values()}) != 1:
        raise M42AnalysisError("gripper evidence must use one selected execution horizon")
    return result


def binary_gripper_qualifies(
    *,
    mixed_project: RuntimeAblationResult,
    mixed_binary: RuntimeAblationResult,
    per_task_project: RuntimeAblationResult,
    per_task_binary: RuntimeAblationResult,
) -> bool:
    """Apply the declared minimum-improvement gate and explicit safety guard."""
    success_gain = mixed_binary.success_rate - mixed_project.success_rate >= 0.05
    if mixed_project.post_grasp_timeouts == 0:
        timeout_gain = False
    else:
        timeout_gain = (
            mixed_project.post_grasp_timeouts - mixed_binary.post_grasp_timeouts
        ) / mixed_project.post_grasp_timeouts >= 0.25
    representative_gain = per_task_binary.successes - per_task_project.successes >= 2
    safety_not_worse = (
        mixed_binary.wrong_object_interactions <= mixed_project.wrong_object_interactions
        and mixed_binary.wrong_object_grasp_count <= mixed_project.wrong_object_grasp_count
        and mixed_binary.wrong_object_in_target_bin_count
        <= mixed_project.wrong_object_in_target_bin_count
        and mixed_binary.target_in_wrong_bin_count <= mixed_project.target_in_wrong_bin_count
        and mixed_binary.target_off_table_count <= mixed_project.target_off_table_count
        and mixed_binary.invalid_action_count <= mixed_project.invalid_action_count
        and per_task_binary.wrong_object_interactions <= per_task_project.wrong_object_interactions
        and per_task_binary.wrong_object_grasp_count <= per_task_project.wrong_object_grasp_count
        and per_task_binary.wrong_object_in_target_bin_count
        <= per_task_project.wrong_object_in_target_bin_count
        and per_task_binary.target_in_wrong_bin_count <= per_task_project.target_in_wrong_bin_count
        and per_task_binary.target_off_table_count <= per_task_project.target_off_table_count
        and per_task_binary.invalid_action_count <= per_task_project.invalid_action_count
    )
    return safety_not_worse and (success_gain or timeout_gain or representative_gain)


def select_gripper_runtime(
    *,
    mixed_results: tuple[RuntimeAblationResult, ...],
    representative_per_task_results: tuple[RuntimeAblationResult, ...],
) -> GripperRuntimeMode:
    """Rank project/binary, then enforce the stronger binary qualification gate."""
    mixed = _by_gripper(mixed_results, label="mixed_results")
    per_task = _by_gripper(representative_per_task_results, label="per_task_results")
    for mode in GripperRuntimeMode:
        left, right = mixed[mode], per_task[mode]
        if (
            left.schedule_fingerprint != right.schedule_fingerprint
            or left.execution_horizon != right.execution_horizon
        ):
            raise M42AnalysisError("gripper evidence does not share one development contract")
    project = mixed[GripperRuntimeMode.PROJECT]
    binary = mixed[GripperRuntimeMode.BINARY]
    per_project = per_task[GripperRuntimeMode.PROJECT]
    per_binary = per_task[GripperRuntimeMode.BINARY]
    if not binary_gripper_qualifies(
        mixed_project=project,
        mixed_binary=binary,
        per_task_project=per_project,
        per_task_binary=per_binary,
    ):
        return GripperRuntimeMode.PROJECT

    def key(mode: GripperRuntimeMode) -> tuple[float, float, float, float, int, int]:
        primary = mixed[mode]
        reference = per_task[mode]
        return (
            -primary.success_rate,
            primary.post_grasp_timeout_rate,
            primary.wrong_object_interaction_rate,
            -reference.success_rate,
            primary.unnecessary_gripper_sign_transitions,
            0 if mode is GripperRuntimeMode.PROJECT else 1,
        )

    return min(GripperRuntimeMode, key=key)


def create_gripper_selection(
    *,
    mixed_results: tuple[RuntimeAblationResult, ...],
    representative_per_task_results: tuple[RuntimeAblationResult, ...],
    locked_at_utc: str,
) -> M42SelectionRecord:
    selected = select_gripper_runtime(
        mixed_results=mixed_results,
        representative_per_task_results=representative_per_task_results,
    )
    all_results = mixed_results + representative_per_task_results
    return M42SelectionRecord(
        selection_kind=M42SelectionKind.GRIPPER_RUNTIME,
        selected_value=selected.value,
        candidate_values=(GripperRuntimeMode.PROJECT.value, GripperRuntimeMode.BINARY.value),
        ranking_version=GRIPPER_RANKING_VERSION,
        evidence_fingerprints=tuple(_evidence_digest(value) for value in all_results),
        development_schedule_fingerprint=all_results[0].schedule_fingerprint,
        locked_at_utc=locked_at_utc,
    )


def rank_task_token_validation_results(
    values: tuple[TaskTokenValidationResult, ...],
) -> tuple[TaskTokenValidationResult, ...]:
    """Rank all 20 checkpoints without test, development, or final evidence."""
    if len(values) != 20:
        raise M42AnalysisError("TaskToken selection requires all 20 checkpoint results")
    if tuple(sorted(value.checkpoint_step for value in values)) != tuple(
        range(5_000, 100_001, 5_000)
    ):
        raise M42AnalysisError("TaskToken validation results must cover the exact 5k schedule")
    if len({value.checkpoint_fingerprint for value in values}) != 20:
        raise M42AnalysisError("TaskToken validation checkpoint fingerprints must be unique")
    if len({value.validation_schedule_fingerprint for value in values}) != 1:
        raise M42AnalysisError("TaskToken checkpoints must share one validation schedule")
    if len({value.validation_split_digest for value in values}) != 1:
        raise M42AnalysisError("TaskToken checkpoints must share one validation split")
    return tuple(
        sorted(
            values,
            key=lambda value: (
                -value.success_rate,
                value.wrong_object_interaction_rate,
                value.target_off_table_rate,
                value.timeout_rate,
                value.offline_validation_action_loss,
                value.checkpoint_step,
            ),
        )
    )


def create_task_token_checkpoint_selection(
    values: tuple[TaskTokenValidationResult, ...], *, locked_at_utc: str
) -> M42SelectionRecord:
    ranked = rank_task_token_validation_results(values)
    selected = ranked[0]
    return M42SelectionRecord(
        selection_kind=M42SelectionKind.TASK_TOKEN_CHECKPOINT,
        selected_value=selected.checkpoint_fingerprint,
        candidate_values=tuple(value.checkpoint_fingerprint for value in ranked),
        ranking_version=TASK_TOKEN_CHECKPOINT_RANKING_VERSION,
        evidence_fingerprints=tuple(f"sha256:{sha256_hex(value.to_dict())}" for value in values),
        development_schedule_fingerprint=None,
        selected_checkpoint_fingerprint=selected.checkpoint_fingerprint,
        validation_split_digest=selected.validation_split_digest,
        locked_at_utc=locked_at_utc,
    )


def paired_success_metrics(
    left: tuple[bool, ...], right: tuple[bool, ...]
) -> dict[str, int | float]:
    """Return paired episode outcomes without treating paired runs as independent."""
    if not left or len(left) != len(right):
        raise M42AnalysisError("paired success vectors must be non-empty and equal length")
    left_only = sum(a and not b for a, b in zip(left, right, strict=True))
    right_only = sum(b and not a for a, b in zip(left, right, strict=True))
    both = sum(a and b for a, b in zip(left, right, strict=True))
    neither = len(left) - left_only - right_only - both
    return {
        "trials": len(left),
        "left_successes": sum(left),
        "right_successes": sum(right),
        "left_only_successes": left_only,
        "right_only_successes": right_only,
        "both_successes": both,
        "neither_successes": neither,
        "paired_success_rate_difference": (sum(left) - sum(right)) / len(left),
    }


def task_sensitivity_ratio(*, policy_distance: float, per_task_oracle_distance: float) -> float:
    for value, name in (
        (policy_distance, "policy_distance"),
        (per_task_oracle_distance, "per_task_oracle_distance"),
    ):
        if not math.isfinite(value) or value < 0:
            raise M42AnalysisError(f"{name} must be finite and non-negative")
    if per_task_oracle_distance == 0:
        raise M42AnalysisError("PerTask oracle distance must be positive")
    return policy_distance / per_task_oracle_distance


def go_no_go_thresholds(metrics: M42GoNoGoMetrics) -> dict[str, bool]:
    """Compute every sealed M4.2-final threshold exactly once."""
    success_rate = metrics.task_token_successes / metrics.task_token_trials
    per_task_rate = metrics.per_task_aggregate_successes / metrics.per_task_aggregate_trials
    timeout_rate = metrics.timeout_count / metrics.task_token_trials
    wrong_grasp_rate = metrics.wrong_object_grasp_count / metrics.task_token_trials
    wrong_bin_rate = metrics.wrong_object_in_target_bin_count / metrics.task_token_trials
    return {
        "overall_success_at_least_70_percent": success_rate >= 0.70,
        "overall_successes_at_least_126": metrics.task_token_successes >= 126,
        "paired_gap_to_per_task_at_most_12_points": per_task_rate - success_rate <= 0.12,
        "every_task_success_at_least_60_percent": all(
            value / 30 >= 0.60 for value in metrics.task_token_successes_by_task
        ),
        "every_task_successes_at_least_18": all(
            value >= 18 for value in metrics.task_token_successes_by_task
        ),
        "timeout_rate_at_most_30_percent": timeout_rate <= 0.30,
        "wrong_object_grasp_rate_at_most_8_percent": wrong_grasp_rate <= 0.08,
        "wrong_object_in_target_bin_rate_at_most_3_percent": wrong_bin_rate <= 0.03,
        "target_in_wrong_bin_zero": metrics.target_in_wrong_bin_count == 0,
        "target_off_table_zero": metrics.target_off_table_count == 0,
        "arm_projection_zero": metrics.arm_projection_count == 0,
        "nan_zero": metrics.nan_count == 0,
        "inf_zero": metrics.inf_count == 0,
        "malformed_action_zero": metrics.malformed_action_count == 0,
        "task_sensitivity_ratio_at_least_075": metrics.task_sensitivity_ratio >= 0.75,
    }


def create_go_no_go_decision(
    *, metrics: M42GoNoGoMetrics, final_benchmark_fingerprint: str
) -> M42GoNoGoDecision:
    checks = go_no_go_thresholds(metrics)
    failed = tuple(sorted(name for name, passed in checks.items() if not passed))
    decision = (
        M42Decision.GO_FOR_SMOLVLA if not failed else M42Decision.REMAIN_IN_ORACLE_CONTROL_LAYER
    )
    return M42GoNoGoDecision(
        metrics=metrics,
        decision=decision,
        thresholds_passed=checks,
        failed_thresholds=failed,
        final_benchmark_fingerprint=final_benchmark_fingerprint,
    )


def median_success_steps(values: tuple[int, ...]) -> float | None:
    """Small report helper with an explicit empty-success representation."""
    return float(median(values)) if values else None


__all__ = [
    "GO_NO_GO_VERSION",
    "GRIPPER_RANKING_VERSION",
    "HORIZON_RANKING_VERSION",
    "TASK_TOKEN_CHECKPOINT_RANKING_VERSION",
    "M42AnalysisError",
    "PostGraspTrace",
    "binary_gripper_qualifies",
    "classify_post_grasp_phase",
    "create_go_no_go_decision",
    "create_gripper_selection",
    "create_horizon_selection",
    "create_task_token_checkpoint_selection",
    "go_no_go_thresholds",
    "median_success_steps",
    "paired_success_metrics",
    "rank_execution_horizons",
    "rank_task_token_validation_results",
    "select_gripper_runtime",
    "task_sensitivity_ratio",
]
