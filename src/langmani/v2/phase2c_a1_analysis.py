"""Final ACT comparison and closure analysis for LangMani 2.0 Phase 2C-A.1."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Final, TypedDict

import numpy as np

from langmani.datasets.identity import sha256_hex
from langmani.v2.phase2c_a import PACKAGE_FINGERPRINT, TASK_IDS, wilson_interval

FINAL_SPLITS: Final = ("test_unseen_reset", "test_visual_shift")
WEAK_SUCCESS_RATE_MAXIMUM: Final = 0.10


class Phase2CA1AnalysisError(RuntimeError):
    """Raised when final ACT evidence is incomplete or identity inconsistent."""


class GroupSummary(TypedDict):
    episode_count: int
    success_count: int
    success_rate: float
    success_wilson_95: list[float]
    timeout_count: int
    invalid_action_count: int
    simulator_error_count: int
    actions_executed: int
    policy_queries: int
    outcomes: dict[str, int]
    failure_categories: dict[str, int]
    action_variance_by_dimension: list[float] | None
    mean_action_smoothness_l2: float | None
    inference_latency_p50_ms: float | None
    inference_latency_p95_ms: float | None
    checkpoint_identities: list[str]
    execution_horizons: list[int]


def _records(
    source: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
    split: str,
    task_id: str,
) -> Sequence[Mapping[str, object]]:
    by_task = source.get(split)
    values = by_task.get(task_id) if isinstance(by_task, Mapping) else None
    if not isinstance(values, Sequence) or isinstance(values, str | bytes) or not values:
        raise Phase2CA1AnalysisError(f"missing {split}/{task_id} episode records")
    return values


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _integer(record: Mapping[str, object], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int):
        raise Phase2CA1AnalysisError(f"episode lacks integer {key}")
    return value


def _group_summary(records: Sequence[Mapping[str, object]]) -> GroupSummary:
    successes = sum(record.get("success") is True for record in records)
    lower, upper = wilson_interval(successes, len(records))
    actions: list[np.ndarray] = []
    latencies: list[float] = []
    smoothness: list[float] = []
    for record in records:
        raw_actions = np.asarray(record.get("action_sequence"), dtype=np.float32)
        if raw_actions.size == 0:
            raw_actions = np.empty((0, 8), dtype=np.float32)
        elif raw_actions.ndim != 2 or raw_actions.shape[1:] != (8,):
            raise Phase2CA1AnalysisError("episode lacks float32[N,8] executed actions")
        if raw_actions.size and not np.isfinite(raw_actions).all():
            raise Phase2CA1AnalysisError("episode action trace contains nonfinite values")
        if len(raw_actions):
            actions.append(raw_actions)
        raw_latencies = record.get("inference_latency_samples_ms")
        if isinstance(raw_latencies, list | tuple):
            latencies.extend(
                float(value)
                for value in raw_latencies
                if isinstance(value, int | float) and math.isfinite(float(value))
            )
        value = record.get("action_smoothness_mean_l2")
        if isinstance(value, int | float) and math.isfinite(float(value)):
            smoothness.append(float(value))
    action_array = np.concatenate(actions, axis=0) if actions else np.empty((0, 8))
    return {
        "episode_count": len(records),
        "success_count": successes,
        "success_rate": successes / len(records),
        "success_wilson_95": [lower, upper],
        "timeout_count": sum(record.get("timeout") is True for record in records),
        "invalid_action_count": sum(record.get("invalid_action") is True for record in records),
        "simulator_error_count": sum(record.get("simulator_error") is True for record in records),
        "actions_executed": sum(_integer(record, "actions_executed") for record in records),
        "policy_queries": sum(_integer(record, "policy_query_count") for record in records),
        "outcomes": dict(sorted(Counter(str(record.get("outcome")) for record in records).items())),
        "failure_categories": dict(
            sorted(Counter(str(record.get("failure_category")) for record in records).items())
        ),
        "action_variance_by_dimension": (
            np.var(action_array, axis=0).tolist() if len(action_array) else None
        ),
        "mean_action_smoothness_l2": (statistics.fmean(smoothness) if smoothness else None),
        "inference_latency_p50_ms": _percentile(latencies, 0.50),
        "inference_latency_p95_ms": _percentile(latencies, 0.95),
        "checkpoint_identities": sorted(
            {str(record.get("checkpoint_identity")) for record in records}
        ),
        "execution_horizons": sorted({_integer(record, "execution_horizon") for record in records}),
    }


def analyze_phase2c_a1_results(
    *,
    per_task: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
    shared: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
    training_runs_complete: bool,
    checkpoints_recoverable: bool,
    padding_mask_verified: bool,
    closed_loop_smoke_passed: bool,
) -> dict[str, object]:
    """Analyze frozen development/final records and permanently close ACT."""

    summaries: dict[str, dict[str, dict[str, GroupSummary]]] = {
        "per_task": {},
        "shared": {},
    }
    for scope, source in (("per_task", per_task), ("shared", shared)):
        for split in ("validation", *FINAL_SPLITS):
            summaries[scope][split] = {
                task_id: _group_summary(_records(source, split, task_id)) for task_id in TASK_IDS
            }
    all_summaries = [
        summary
        for scope in summaries.values()
        for split in scope.values()
        for summary in split.values()
    ]
    expected_counts_valid = all(
        int(summaries[scope]["validation"][task_id]["episode_count"]) == 30
        and all(
            int(summaries[scope][split][task_id]["episode_count"]) == 30 for split in FINAL_SPLITS
        )
        for scope in ("per_task", "shared")
        for task_id in TASK_IDS
    )
    invalid_actions = sum(item["invalid_action_count"] for item in all_summaries)
    simulator_errors = sum(item["simulator_error_count"] for item in all_summaries)
    every_group_executed = all(item["actions_executed"] > 0 for item in all_summaries)
    pipeline_valid = (
        training_runs_complete
        and checkpoints_recoverable
        and padding_mask_verified
        and closed_loop_smoke_passed
        and expected_counts_valid
        and invalid_actions == 0
        and simulator_errors == 0
        and every_group_executed
    )
    combined_final: dict[str, dict[str, GroupSummary]] = {
        "per_task": {},
        "shared": {},
    }
    for scope, source in (("per_task", per_task), ("shared", shared)):
        combined_final[scope] = {
            task_id: _group_summary(
                [
                    *list(_records(source, FINAL_SPLITS[0], task_id)),
                    *list(_records(source, FINAL_SPLITS[1], task_id)),
                ]
            )
            for task_id in TASK_IDS
        }
    per_rates = {
        task_id: combined_final["per_task"][task_id]["success_rate"] for task_id in TASK_IDS
    }
    shared_rates = {
        task_id: combined_final["shared"][task_id]["success_rate"] for task_id in TASK_IDS
    }
    if not pipeline_valid:
        result = "RESULT_D"
    elif all(rate > WEAK_SUCCESS_RATE_MAXIMUM for rate in per_rates.values()) and any(
        rate <= WEAK_SUCCESS_RATE_MAXIMUM for rate in shared_rates.values()
    ):
        result = "RESULT_C"
    elif any(
        rate <= WEAK_SUCCESS_RATE_MAXIMUM for rate in (*per_rates.values(), *shared_rates.values())
    ):
        result = "RESULT_B"
    else:
        result = "RESULT_A"
    interference: dict[str, dict[str, float]] = {
        split: {
            task_id: (
                summaries["shared"][split][task_id]["success_rate"]
                - summaries["per_task"][split][task_id]["success_rate"]
            )
            for task_id in TASK_IDS
        }
        for split in FINAL_SPLITS
    }
    interference["combined_final"] = {
        task_id: shared_rates[task_id] - per_rates[task_id] for task_id in TASK_IDS
    }
    interference_average = {
        split: statistics.fmean(values.values()) for split, values in interference.items()
    }
    comparison_groups = {
        **{
            split: {
                "per_task": summaries["per_task"][split],
                "shared": summaries["shared"][split],
            }
            for split in ("validation", *FINAL_SPLITS)
        },
        "combined_final": combined_final,
    }
    timeout_rate_difference = {
        split: {
            task_id: (
                groups["shared"][task_id]["timeout_count"]
                / groups["shared"][task_id]["episode_count"]
                - groups["per_task"][task_id]["timeout_count"]
                / groups["per_task"][task_id]["episode_count"]
            )
            for task_id in TASK_IDS
        }
        for split, groups in comparison_groups.items()
    }
    action_variance_difference: dict[str, dict[str, list[float] | None]] = {}
    failure_category_comparison: dict[str, dict[str, dict[str, object]]] = {}
    for split, groups in comparison_groups.items():
        action_variance_difference[split] = {}
        failure_category_comparison[split] = {}
        for task_id in TASK_IDS:
            per_summary = groups["per_task"][task_id]
            shared_summary = groups["shared"][task_id]
            per_variance = per_summary["action_variance_by_dimension"]
            shared_variance = shared_summary["action_variance_by_dimension"]
            action_variance_difference[split][task_id] = (
                (
                    np.asarray(shared_variance, dtype=np.float64)
                    - np.asarray(per_variance, dtype=np.float64)
                ).tolist()
                if per_variance is not None and shared_variance is not None
                else None
            )
            per_failures = per_summary["failure_categories"]
            shared_failures = shared_summary["failure_categories"]
            categories = sorted({*per_failures, *shared_failures})
            failure_category_comparison[split][task_id] = {
                "per_task": per_failures,
                "shared": shared_failures,
                "shared_minus_per_task": {
                    category: shared_failures.get(category, 0) - per_failures.get(category, 0)
                    for category in categories
                },
            }
    visual_shift_degradation = {
        scope: {
            task_id: (
                summaries[scope]["test_visual_shift"][task_id]["success_rate"]
                - summaries[scope]["test_unseen_reset"][task_id]["success_rate"]
            )
            for task_id in TASK_IDS
        }
        for scope in ("per_task", "shared")
    }
    semantic = {
        "schema_version": "langmani-v2-phase2c-a1-result-analysis-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "weak_or_near_zero_success_rate_maximum": WEAK_SUCCESS_RATE_MAXIMUM,
        "classification_rule": (
            "D if the consumer pipeline is invalid; C if every per-task combined-final "
            "rate exceeds 0.10 and any shared rate does not; B if any remaining policy-task "
            "rate is at most 0.10; A otherwise"
        ),
        "summaries": summaries,
        "combined_final": combined_final,
        "multi_skill_interference_success_rate_difference": interference,
        "multi_skill_interference_average": interference_average,
        "multi_skill_timeout_rate_difference": timeout_rate_difference,
        "multi_skill_action_variance_difference_by_dimension": action_variance_difference,
        "multi_skill_failure_category_comparison": failure_category_comparison,
        "visual_shift_success_rate_difference": visual_shift_degradation,
        "pipeline_checks": {
            "training_runs_complete": training_runs_complete,
            "checkpoints_recoverable": checkpoints_recoverable,
            "padding_mask_verified": padding_mask_verified,
            "closed_loop_smoke_passed": closed_loop_smoke_passed,
            "expected_episode_counts_valid": expected_counts_valid,
            "invalid_action_episode_count": invalid_actions,
            "simulator_error_episode_count": simulator_errors,
            "every_evaluation_group_executed_actions": every_group_executed,
            "pipeline_valid": pipeline_valid,
        },
        "result": result,
        "act_baselines_validated": result in {"RESULT_A", "RESULT_B", "RESULT_C"},
        "act_policy_quality_weak": result == "RESULT_B",
        "shared_act_failed": result == "RESULT_C",
        "act_phase_closed": True,
        "further_act_architecture_authorized": False,
        "smolvla_phase_eligible": result in {"RESULT_A", "RESULT_B", "RESULT_C"},
        "smolvla_training_authorized": False,
        "vla_jepa_training_authorized": False,
    }
    return {
        **semantic,
        "fingerprint": f"sha256:{sha256_hex(semantic)}",
    }


__all__ = [
    "Phase2CA1AnalysisError",
    "WEAK_SUCCESS_RATE_MAXIMUM",
    "analyze_phase2c_a1_results",
]
