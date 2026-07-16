"""Pure NumPy zero-training semantic audit for frozen ACT action chunks.

The functions in this module compare already postprocessed, environment-
semantic policy actions.  They never apply runtime projection/binary gripper
logic and never step an environment.  Raw-policy and executed-runtime evidence
therefore remain separate contracts.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import BIN_IDS, OBJECT_IDS, stable_task_id
from langmani.policies.m43_types import (
    M43_CANONICAL_TASK_IDS,
    PRIMARY_ACTION_CHUNK_DISTANCE_METRIC,
    ActionChunkDistanceConfig,
    ActionChunkDistanceResult,
    BinRetrievalResult,
    FirstInteractionRecord,
    ObjectRetrievalResult,
    SemanticAlignmentAudit,
    SemanticAuditConclusion,
    SemanticFailureClass,
    SemanticRetrievalResult,
    TaskRetrievalConfusion,
)

type FloatArray = NDArray[np.float64]

_TASK_SPEC_BY_ID = {stable_task_id(task_spec): task_spec for task_spec in CANONICAL_TASK_SPECS}
_TASK_ID_BY_SEMANTICS = {
    (task_spec.target_object_id, task_spec.target_bin_id): stable_task_id(task_spec)
    for task_spec in CANONICAL_TASK_SPECS
}
_MISSING_LABEL = "none"


class SemanticAuditError(ValueError):
    """Raised when zero-training audit evidence is malformed or incomplete."""


@dataclass(frozen=True, slots=True)
class SemanticRolloutTrace:
    """Minimal booleans required by the exhaustive twelve-class taxonomy."""

    success: bool = False
    environment_failure: bool = False
    wrong_object_grasped: bool = False
    ever_grasped_target: bool = False
    ever_lifted_target: bool = False
    ever_transported_to_destination: bool = False
    ever_descended_at_destination: bool = False
    release_command_observed: bool = False
    released_inside_success_region: bool = False
    final_target_static: bool = False
    success_observed_before_final_step: bool = False
    timed_out: bool = False


def _action_chunk(value: object, config: ActionChunkDistanceConfig, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    expected = (config.chunk_size, config.action_dimension)
    if array.shape != expected:
        raise SemanticAuditError(f"{name} must have shape {expected}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise SemanticAuditError(f"{name} must contain only finite actions")
    return array


def _window_slices(config: ActionChunkDistanceConfig) -> dict[str, slice]:
    return {
        "first_action": slice(0, 1),
        "first_5_actions": slice(0, 5),
        "execution_window": slice(0, config.locked_execution_horizon),
        "full_chunk": slice(0, config.chunk_size),
    }


def _l2_by_window(
    candidate: FloatArray,
    reference: FloatArray,
    *,
    indices: tuple[int, ...],
    scale: FloatArray,
    config: ActionChunkDistanceConfig,
) -> dict[str, float]:
    difference = (candidate[:, indices] - reference[:, indices]) / scale[list(indices)]
    return {
        name: float(np.linalg.norm(difference[window]))
        for name, window in _window_slices(config).items()
    }


def _cosine_by_window(
    candidate: FloatArray,
    reference: FloatArray,
    *,
    indices: tuple[int, ...],
    config: ActionChunkDistanceConfig,
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for name, window in _window_slices(config).items():
        left = candidate[window][:, indices].reshape(-1)
        right = reference[window][:, indices].reshape(-1)
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if not math.isfinite(denominator) or denominator <= np.finfo(np.float64).eps:
            result[name] = None
            continue
        similarity = float(np.dot(left, right) / denominator)
        result[name] = 1.0 - float(np.clip(similarity, -1.0, 1.0))
    return result


def compute_action_chunk_distance(
    candidate_chunk: object,
    reference_chunk: object,
    config: ActionChunkDistanceConfig,
) -> ActionChunkDistanceResult:
    """Compute every ActionChunkDistanceV0 metric for one ordered pair."""
    if not isinstance(config, ActionChunkDistanceConfig):
        raise TypeError("config must be an ActionChunkDistanceConfig")
    candidate = _action_chunk(candidate_chunk, config, "candidate_chunk")
    reference = _action_chunk(reference_chunk, config, "reference_chunk")
    unit_scale = np.ones(config.action_dimension, dtype=np.float64)
    range_scale = np.asarray(config.action_upper_bounds, dtype=np.float64) - np.asarray(
        config.action_lower_bounds, dtype=np.float64
    )
    std_scale = np.asarray(config.train_action_std, dtype=np.float64)
    all_indices = tuple(range(config.action_dimension))

    def family(
        indices: tuple[int, ...],
    ) -> tuple[
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float | None],
    ]:
        return (
            _l2_by_window(candidate, reference, indices=indices, scale=unit_scale, config=config),
            _l2_by_window(candidate, reference, indices=indices, scale=range_scale, config=config),
            _l2_by_window(candidate, reference, indices=indices, scale=std_scale, config=config),
            _cosine_by_window(candidate, reference, indices=indices, config=config),
        )

    all_raw, all_range, all_std, all_cosine = family(all_indices)
    arm_raw, arm_range, arm_std, arm_cosine = family(config.arm_action_indices)
    gripper_raw, gripper_range, gripper_std, gripper_cosine = family(config.gripper_action_indices)
    return ActionChunkDistanceResult(
        raw_l2=all_raw,
        action_range_normalized_l2=all_range,
        train_std_normalized_l2=all_std,
        cosine_distance=all_cosine,
        arm_only_raw_l2=arm_raw,
        arm_only_action_range_normalized_l2=arm_range,
        arm_only_train_std_normalized_l2=arm_std,
        arm_only_cosine_distance=arm_cosine,
        gripper_only_raw_l2=gripper_raw,
        gripper_only_action_range_normalized_l2=gripper_range,
        gripper_only_train_std_normalized_l2=gripper_std,
        gripper_only_cosine_distance=gripper_cosine,
        primary_metric=PRIMARY_ACTION_CHUNK_DISTANCE_METRIC,
        primary_distance=arm_range["execution_window"],
    )


def _reference_chunks(
    per_task_chunks: Mapping[str, object], config: ActionChunkDistanceConfig
) -> dict[str, FloatArray]:
    if not isinstance(per_task_chunks, Mapping):
        raise TypeError("per_task_chunks must be a mapping")
    if set(per_task_chunks) != set(M43_CANONICAL_TASK_IDS):
        raise SemanticAuditError("per_task_chunks must contain exactly the six canonical tasks")
    return {
        task_id: _action_chunk(per_task_chunks[task_id], config, f"per_task_chunks[{task_id}]")
        for task_id in M43_CANONICAL_TASK_IDS
    }


def _rank_distances(
    distances: Mapping[str, float], canonical_labels: Sequence[str]
) -> tuple[str, ...]:
    order = {label: index for index, label in enumerate(canonical_labels)}
    return tuple(sorted(canonical_labels, key=lambda label: (distances[label], order[label])))


def _correct_margin(
    distances: Mapping[str, float], correct_label: str, labels: Sequence[str]
) -> float:
    nearest_incorrect = min(distances[label] for label in labels if label != correct_label)
    return nearest_incorrect - distances[correct_label]


def retrieve_task(
    *,
    observation_id: str,
    requested_task_id: str,
    candidate_chunk: object,
    per_task_chunks: Mapping[str, object],
    config: ActionChunkDistanceConfig,
) -> SemanticRetrievalResult:
    """Rank all six PerTask references with deterministic canonical tie-breaking."""
    if requested_task_id not in M43_CANONICAL_TASK_IDS:
        raise SemanticAuditError("requested_task_id must be one of the six canonical tasks")
    candidate = _action_chunk(candidate_chunk, config, "candidate_chunk")
    references = _reference_chunks(per_task_chunks, config)
    distances = {
        task_id: compute_action_chunk_distance(candidate, reference, config).primary_distance
        for task_id, reference in references.items()
    }
    ranked = _rank_distances(distances, M43_CANONICAL_TASK_IDS)
    rank = ranked.index(requested_task_id) + 1
    return SemanticRetrievalResult(
        observation_id=observation_id,
        requested_task_id=requested_task_id,
        nearest_task_id=ranked[0],
        ranked_task_ids=ranked,
        primary_distances=distances,
        correct_rank=rank,
        top1_correct=rank == 1,
        top2_correct=rank <= 2,
        reciprocal_rank=1.0 / rank,
        correct_margin=_correct_margin(distances, requested_task_id, M43_CANONICAL_TASK_IDS),
    )


def _centroid(references: Mapping[str, FloatArray], task_ids: Sequence[str]) -> FloatArray:
    return np.mean(np.stack([references[task_id] for task_id in task_ids]), axis=0)


def retrieve_object(
    *,
    observation_id: str,
    requested_task_id: str,
    candidate_chunk: object,
    per_task_chunks: Mapping[str, object],
    config: ActionChunkDistanceConfig,
) -> ObjectRetrievalResult:
    """Retrieve red/green/blue against two-PerTask action centroids."""
    if requested_task_id not in M43_CANONICAL_TASK_IDS:
        raise SemanticAuditError("requested_task_id must be canonical")
    candidate = _action_chunk(candidate_chunk, config, "candidate_chunk")
    references = _reference_chunks(per_task_chunks, config)
    requested_object = _TASK_SPEC_BY_ID[requested_task_id].target_object_id
    distances: dict[str, float] = {}
    for object_id in OBJECT_IDS:
        task_ids = tuple(
            stable_task_id(task_spec)
            for task_spec in CANONICAL_TASK_SPECS
            if task_spec.target_object_id == object_id
        )
        distances[object_id] = compute_action_chunk_distance(
            candidate, _centroid(references, task_ids), config
        ).primary_distance
    ranked = _rank_distances(distances, OBJECT_IDS)
    return ObjectRetrievalResult(
        observation_id=observation_id,
        requested_object_id=requested_object,
        predicted_object_id=ranked[0],
        ranked_object_ids=ranked,
        primary_distances=distances,
        correct=ranked[0] == requested_object,
        correct_margin=_correct_margin(distances, requested_object, OBJECT_IDS),
    )


def retrieve_bin(
    *,
    observation_id: str,
    requested_task_id: str,
    candidate_chunk: object,
    per_task_chunks: Mapping[str, object],
    config: ActionChunkDistanceConfig,
) -> BinRetrievalResult:
    """Retrieve global bin centroids and object-conditional left/right references."""
    if requested_task_id not in M43_CANONICAL_TASK_IDS:
        raise SemanticAuditError("requested_task_id must be canonical")
    candidate = _action_chunk(candidate_chunk, config, "candidate_chunk")
    references = _reference_chunks(per_task_chunks, config)
    requested_spec = _TASK_SPEC_BY_ID[requested_task_id]
    distances: dict[str, float] = {}
    for bin_id in BIN_IDS:
        task_ids = tuple(
            stable_task_id(task_spec)
            for task_spec in CANONICAL_TASK_SPECS
            if task_spec.target_bin_id == bin_id
        )
        distances[bin_id] = compute_action_chunk_distance(
            candidate, _centroid(references, task_ids), config
        ).primary_distance
    ranked = _rank_distances(distances, BIN_IDS)

    conditional_distances = {
        bin_id: compute_action_chunk_distance(
            candidate,
            references[_TASK_ID_BY_SEMANTICS[(requested_spec.target_object_id, bin_id)]],
            config,
        ).primary_distance
        for bin_id in BIN_IDS
    }
    conditional_ranked = _rank_distances(conditional_distances, BIN_IDS)
    return BinRetrievalResult(
        observation_id=observation_id,
        requested_bin_id=requested_spec.target_bin_id,
        predicted_bin_id=ranked[0],
        ranked_bin_ids=ranked,
        primary_distances=distances,
        correct=ranked[0] == requested_spec.target_bin_id,
        correct_margin=_correct_margin(distances, requested_spec.target_bin_id, BIN_IDS),
        conditional_predicted_bin_id=conditional_ranked[0],
        conditional_primary_distances=conditional_distances,
        conditional_correct=conditional_ranked[0] == requested_spec.target_bin_id,
        conditional_correct_margin=_correct_margin(
            conditional_distances, requested_spec.target_bin_id, BIN_IDS
        ),
    )


def categorical_confusion(
    pairs: Sequence[tuple[str, str]],
    *,
    requested_labels: Sequence[str],
    predicted_labels: Sequence[str],
) -> TaskRetrievalConfusion:
    """Count ordered pairs into a deterministic rectangular confusion matrix."""
    requested = tuple(requested_labels)
    predicted = tuple(predicted_labels)
    if not requested or not predicted:
        raise SemanticAuditError("confusion label sets must not be empty")
    row_index = {label: index for index, label in enumerate(requested)}
    column_index = {label: index for index, label in enumerate(predicted)}
    counts = [[0 for _ in predicted] for _ in requested]
    for actual, inferred in pairs:
        if actual not in row_index or inferred not in column_index:
            raise SemanticAuditError("confusion pair contains an undeclared label")
        counts[row_index[actual]][column_index[inferred]] += 1
    return TaskRetrievalConfusion(
        requested_labels=requested,
        predicted_labels=predicted,
        counts=tuple(tuple(row) for row in counts),
    )


def task_retrieval_confusion(
    results: Sequence[SemanticRetrievalResult],
) -> TaskRetrievalConfusion:
    return categorical_confusion(
        tuple((value.requested_task_id, value.nearest_task_id) for value in results),
        requested_labels=M43_CANONICAL_TASK_IDS,
        predicted_labels=M43_CANONICAL_TASK_IDS,
    )


def object_retrieval_confusion(
    results: Sequence[ObjectRetrievalResult],
) -> TaskRetrievalConfusion:
    return categorical_confusion(
        tuple((value.requested_object_id, value.predicted_object_id) for value in results),
        requested_labels=OBJECT_IDS,
        predicted_labels=OBJECT_IDS,
    )


def bin_retrieval_confusion(results: Sequence[BinRetrievalResult]) -> TaskRetrievalConfusion:
    return categorical_confusion(
        tuple((value.requested_bin_id, value.predicted_bin_id) for value in results),
        requested_labels=BIN_IDS,
        predicted_labels=BIN_IDS,
    )


def precision_recall(confusion: TaskRetrievalConfusion) -> dict[str, dict[str, float]]:
    """Return per-label precision/recall for a square semantic confusion matrix."""
    if confusion.requested_labels != confusion.predicted_labels:
        raise SemanticAuditError("precision/recall requires identical requested/predicted labels")
    result: dict[str, dict[str, float]] = {}
    for index, label in enumerate(confusion.requested_labels):
        true_positive = confusion.counts[index][index]
        predicted_total = sum(row[index] for row in confusion.counts)
        requested_total = sum(confusion.counts[index])
        result[label] = {
            "precision": true_positive / predicted_total if predicted_total else 0.0,
            "recall": true_positive / requested_total if requested_total else 0.0,
        }
    return result


def summarize_task_retrieval(
    results: Sequence[SemanticRetrievalResult],
) -> dict[str, object]:
    if not results:
        raise SemanticAuditError("task retrieval summary requires at least one result")
    confusion = task_retrieval_confusion(results)
    per_task: dict[str, dict[str, float | int]] = {}
    for task_id in M43_CANONICAL_TASK_IDS:
        selected = tuple(value for value in results if value.requested_task_id == task_id)
        per_task[task_id] = {
            "count": len(selected),
            "top1_accuracy": sum(value.top1_correct for value in selected) / len(selected)
            if selected
            else 0.0,
        }
    count = len(results)
    return {
        "count": count,
        "top1_accuracy": sum(value.top1_correct for value in results) / count,
        "top2_accuracy": sum(value.top2_correct for value in results) / count,
        "mean_reciprocal_rank": sum(value.reciprocal_rank for value in results) / count,
        "mean_correct_margin": sum(value.correct_margin for value in results) / count,
        "confusion": confusion.to_dict(),
        "per_task": per_task,
    }


def summarize_object_retrieval(results: Sequence[ObjectRetrievalResult]) -> dict[str, object]:
    if not results:
        raise SemanticAuditError("object retrieval summary requires at least one result")
    confusion = object_retrieval_confusion(results)
    return {
        "count": len(results),
        "accuracy": sum(value.correct for value in results) / len(results),
        "mean_correct_margin": sum(value.correct_margin for value in results) / len(results),
        "confusion": confusion.to_dict(),
        "precision_recall": precision_recall(confusion),
    }


def summarize_bin_retrieval(results: Sequence[BinRetrievalResult]) -> dict[str, object]:
    if not results:
        raise SemanticAuditError("bin retrieval summary requires at least one result")
    confusion = bin_retrieval_confusion(results)
    conditional_confusion = categorical_confusion(
        tuple((value.requested_bin_id, value.conditional_predicted_bin_id) for value in results),
        requested_labels=BIN_IDS,
        predicted_labels=BIN_IDS,
    )
    return {
        "count": len(results),
        "accuracy": sum(value.correct for value in results) / len(results),
        "conditional_accuracy": sum(value.conditional_correct for value in results) / len(results),
        "mean_correct_margin": sum(value.correct_margin for value in results) / len(results),
        "mean_conditional_correct_margin": sum(
            value.conditional_correct_margin for value in results
        )
        / len(results),
        "confusion": confusion.to_dict(),
        "conditional_confusion": conditional_confusion.to_dict(),
        "precision_recall": precision_recall(confusion),
    }


def classify_semantic_failure(trace: SemanticRolloutTrace) -> SemanticFailureClass:
    """Classify one rollout into exactly one of the predeclared twelve outcomes."""
    if not isinstance(trace, SemanticRolloutTrace):
        raise TypeError("trace must be a SemanticRolloutTrace")
    if trace.environment_failure:
        return SemanticFailureClass.ENVIRONMENT_FAILURE
    if trace.success:
        return SemanticFailureClass.SUCCESS
    if trace.wrong_object_grasped:
        return SemanticFailureClass.WRONG_OBJECT_INTERACTION
    if trace.success_observed_before_final_step:
        return SemanticFailureClass.TARGET_SUCCESS_THEN_LOST
    if not trace.ever_grasped_target:
        return SemanticFailureClass.NEVER_GRASPED_TARGET
    if trace.timed_out:
        return SemanticFailureClass.TIMED_OUT_AFTER_TARGET_GRASP
    if not trace.ever_lifted_target:
        return SemanticFailureClass.TARGET_GRASPED_NOT_LIFTED
    if not trace.ever_transported_to_destination:
        return SemanticFailureClass.LIFTED_NOT_TRANSPORTED
    if not trace.ever_descended_at_destination:
        return SemanticFailureClass.TRANSPORTED_NOT_DESCENDED
    if not trace.release_command_observed:
        return SemanticFailureClass.DESCENDED_NOT_RELEASED
    if not trace.released_inside_success_region:
        return SemanticFailureClass.RELEASED_OUTSIDE_SUCCESS_REGION
    if not trace.final_target_static:
        return SemanticFailureClass.RELEASED_BUT_NOT_STATIC
    return SemanticFailureClass.ENVIRONMENT_FAILURE


def build_first_interaction_record(
    *,
    observation_id: str,
    requested_task_id: str,
    first_object_grasped: str | None,
    first_object_displaced: str | None,
    requested_destination_bin_id: str,
    first_bin_approached: str | None,
    first_bin_entered_by_any_object: str | None,
    final_object_bin_relationship: Mapping[str, object],
    nearest_per_task_task_id: str,
    trace: SemanticRolloutTrace,
    displacement_threshold_m: float = 0.01,
) -> FirstInteractionRecord:
    """Construct derived first-interaction semantics without trajectory payloads."""
    if (
        requested_task_id not in _TASK_SPEC_BY_ID
        or nearest_per_task_task_id not in _TASK_SPEC_BY_ID
    ):
        raise SemanticAuditError("first-interaction task IDs must be canonical")
    requested_object = _TASK_SPEC_BY_ID[requested_task_id].target_object_id
    nearest_object = _TASK_SPEC_BY_ID[nearest_per_task_task_id].target_object_id
    return FirstInteractionRecord(
        observation_id=observation_id,
        requested_task_id=requested_task_id,
        requested_target_object_id=requested_object,
        first_object_grasped=first_object_grasped,
        first_object_displaced=first_object_displaced,
        displacement_threshold_m=displacement_threshold_m,
        requested_destination_bin_id=requested_destination_bin_id,
        first_bin_approached=first_bin_approached,
        first_bin_entered_by_any_object=first_bin_entered_by_any_object,
        final_object_bin_relationship=final_object_bin_relationship,
        nearest_per_task_task_id=nearest_per_task_task_id,
        first_grasp_matches_nearest_per_task=(
            None if first_object_grasped is None else first_object_grasped == nearest_object
        ),
        initial_semantic_error_visible=nearest_per_task_task_id != requested_task_id,
        failure_class=classify_semantic_failure(trace),
    )


def first_interaction_confusions(
    records: Sequence[FirstInteractionRecord],
) -> dict[str, TaskRetrievalConfusion]:
    """Build all five predeclared first-interaction confusion matrices."""
    object_predictions = (*OBJECT_IDS, _MISSING_LABEL)
    bin_predictions = (*BIN_IDS, _MISSING_LABEL)
    return {
        "requested_object_to_first_grasped_object": categorical_confusion(
            tuple(
                (
                    value.requested_target_object_id,
                    value.first_object_grasped or _MISSING_LABEL,
                )
                for value in records
            ),
            requested_labels=OBJECT_IDS,
            predicted_labels=object_predictions,
        ),
        "requested_task_to_nearest_per_task_chunk": categorical_confusion(
            tuple((value.requested_task_id, value.nearest_per_task_task_id) for value in records),
            requested_labels=M43_CANONICAL_TASK_IDS,
            predicted_labels=M43_CANONICAL_TASK_IDS,
        ),
        "nearest_per_task_chunk_to_actual_first_grasp": categorical_confusion(
            tuple(
                (
                    value.nearest_per_task_task_id,
                    value.first_object_grasped or _MISSING_LABEL,
                )
                for value in records
            ),
            requested_labels=M43_CANONICAL_TASK_IDS,
            predicted_labels=object_predictions,
        ),
        "requested_bin_to_first_approached_bin": categorical_confusion(
            tuple(
                (
                    value.requested_destination_bin_id,
                    value.first_bin_approached or _MISSING_LABEL,
                )
                for value in records
            ),
            requested_labels=BIN_IDS,
            predicted_labels=bin_predictions,
        ),
        "requested_bin_to_object_entry_bin": categorical_confusion(
            tuple(
                (
                    value.requested_destination_bin_id,
                    value.first_bin_entered_by_any_object or _MISSING_LABEL,
                )
                for value in records
            ),
            requested_labels=BIN_IDS,
            predicted_labels=bin_predictions,
        ),
    }


def classify_semantic_conclusions(
    *,
    sufficient_evidence: bool,
    condition_changes_output: bool,
    correct_task_retrieval: bool,
    correct_object_retrieval: bool,
    correct_bin_retrieval: bool,
    shared_control_degraded_after_correct_selection: bool,
    post_grasp_execution_failed: bool,
) -> tuple[SemanticAuditConclusion, ...]:
    """Translate explicit audit decisions into stable conclusion labels.

    The caller must derive these booleans from declared thresholds.  In
    particular, a merely non-zero action distance cannot imply correct semantic
    conditioning.
    """
    if not sufficient_evidence:
        return (SemanticAuditConclusion.INSUFFICIENT_EVIDENCE,)
    conclusions: list[SemanticAuditConclusion] = []
    if not condition_changes_output:
        conclusions.append(SemanticAuditConclusion.CONDITION_NOT_USED)
    elif not correct_task_retrieval:
        conclusions.append(SemanticAuditConclusion.CONDITION_CHANGES_OUTPUT_BUT_WRONG_SEMANTICS)
    if not correct_object_retrieval:
        conclusions.append(SemanticAuditConclusion.TARGET_OBJECT_CONFUSION)
    if not correct_bin_retrieval:
        conclusions.append(SemanticAuditConclusion.DESTINATION_BIN_CONFUSION)
    if shared_control_degraded_after_correct_selection:
        conclusions.append(
            SemanticAuditConclusion.SHARED_CONTROL_DEGRADATION_AFTER_CORRECT_SELECTION
        )
    if post_grasp_execution_failed:
        conclusions.append(SemanticAuditConclusion.POST_GRASP_EXECUTION_FAILURE)
    return tuple(conclusions)


def build_semantic_alignment_audit(
    *,
    policy_label: str,
    distance_config: ActionChunkDistanceConfig,
    task_results: Sequence[SemanticRetrievalResult],
    object_results: Sequence[ObjectRetrievalResult],
    bin_results: Sequence[BinRetrievalResult],
    first_interactions: Sequence[FirstInteractionRecord],
    conclusions: Sequence[SemanticAuditConclusion],
    raw_policy_action_metrics: Mapping[str, object],
    runtime_action_metrics: Mapping[str, object],
    deterministic_reload_validated: bool,
    deterministic_reset_validated: bool,
    complete_distance_results: Mapping[str, object] | None = None,
) -> SemanticAlignmentAudit:
    """Aggregate one candidate policy while preserving raw/runtime separation."""
    if not task_results:
        raise SemanticAuditError("semantic alignment audit requires offline observations")
    task_tuple = tuple(task_results)
    object_tuple = tuple(object_results)
    bin_tuple = tuple(bin_results)
    interaction_tuple = tuple(first_interactions)
    confusions = first_interaction_confusions(interaction_tuple)
    failure_counts = Counter(value.failure_class.value for value in interaction_tuple)
    return SemanticAlignmentAudit(
        policy_label=policy_label,
        observation_count=len(task_tuple),
        distance_config=distance_config,
        task_retrieval=task_tuple,
        object_retrieval=object_tuple,
        bin_retrieval=bin_tuple,
        task_confusion=task_retrieval_confusion(task_tuple),
        object_confusion=object_retrieval_confusion(object_tuple),
        bin_confusion=bin_retrieval_confusion(bin_tuple),
        first_interactions=interaction_tuple,
        first_interaction_confusions={name: value.to_dict() for name, value in confusions.items()},
        failure_distribution={
            failure.value: failure_counts.get(failure.value, 0) for failure in SemanticFailureClass
        },
        conclusions=tuple(conclusions),
        complete_distance_results=(
            {} if complete_distance_results is None else complete_distance_results
        ),
        raw_policy_action_metrics=raw_policy_action_metrics,
        runtime_action_metrics=runtime_action_metrics,
        deterministic_reload_validated=deterministic_reload_validated,
        deterministic_reset_validated=deterministic_reset_validated,
    )


__all__ = [
    "SemanticAuditError",
    "SemanticRolloutTrace",
    "bin_retrieval_confusion",
    "build_first_interaction_record",
    "build_semantic_alignment_audit",
    "categorical_confusion",
    "classify_semantic_conclusions",
    "classify_semantic_failure",
    "compute_action_chunk_distance",
    "first_interaction_confusions",
    "object_retrieval_confusion",
    "precision_recall",
    "retrieve_bin",
    "retrieve_object",
    "retrieve_task",
    "summarize_bin_retrieval",
    "summarize_object_retrieval",
    "summarize_task_retrieval",
    "task_retrieval_confusion",
]
