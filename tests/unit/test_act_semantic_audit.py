from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from typing import cast

import numpy as np
import pytest

from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import BIN_IDS, OBJECT_IDS, stable_task_id
from langmani.policies.act_semantic_audit import (
    SemanticAuditError,
    SemanticRolloutTrace,
    build_first_interaction_record,
    build_semantic_alignment_audit,
    classify_semantic_conclusions,
    classify_semantic_failure,
    compute_action_chunk_distance,
    first_interaction_confusions,
    precision_recall,
    retrieve_bin,
    retrieve_object,
    retrieve_task,
    summarize_bin_retrieval,
    summarize_object_retrieval,
    summarize_task_retrieval,
    task_retrieval_confusion,
)
from langmani.policies.m43_types import (
    ACTION_CHUNK_DISTANCE_VERSION,
    M43_CANONICAL_TASK_IDS,
    PRIMARY_ACTION_CHUNK_DISTANCE_METRIC,
    ActionChunkDistanceConfig,
    SemanticAuditConclusion,
    SemanticFailureClass,
)


def _config(*, horizon: int = 5) -> ActionChunkDistanceConfig:
    return ActionChunkDistanceConfig(
        action_lower_bounds=(-2.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -4.0),
        action_upper_bounds=(2.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 4.0),
        train_action_std=(2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0),
        locked_execution_horizon=horizon,
    )


def _chunk(*, object_value: float = 0.0, bin_value: float = 0.0) -> np.ndarray:
    value = np.zeros((50, 8), dtype=np.float64)
    value[:, 0] = object_value
    value[:, 1] = bin_value
    value[:, 2] = 0.25
    return value


def _references() -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for task_spec in CANONICAL_TASK_SPECS:
        result[stable_task_id(task_spec)] = _chunk(
            object_value=float(OBJECT_IDS.index(task_spec.target_object_id)),
            bin_value=-1.0 if task_spec.target_bin_id == "left_bin" else 1.0,
        )
    return result


def _task_id(object_id: str, bin_id: str) -> str:
    return next(
        stable_task_id(task_spec)
        for task_spec in CANONICAL_TASK_SPECS
        if task_spec.target_object_id == object_id and task_spec.target_bin_id == bin_id
    )


def test_action_chunk_config_is_frozen_json_serializable_and_versioned() -> None:
    config = _config()
    assert config.version == ACTION_CHUNK_DISTANCE_VERSION
    assert config.primary_metric == PRIMARY_ACTION_CHUNK_DISTANCE_METRIC
    assert config.fingerprint.startswith("sha256:")
    json.dumps(config.to_dict())
    with pytest.raises(FrozenInstanceError):
        config.locked_execution_horizon = 10  # type: ignore[misc]
    with pytest.raises(ValueError, match="non-positive range"):
        replace(config, action_upper_bounds=(-2.0,) + config.action_upper_bounds[1:])
    with pytest.raises(ValueError, match="positive"):
        replace(config, train_action_std=(0.0,) + config.train_action_std[1:])


def test_action_chunk_distance_reports_every_window_scope_and_normalization() -> None:
    config = _config(horizon=5)
    candidate = np.zeros((50, 8), dtype=np.float64)
    reference = np.zeros_like(candidate)
    candidate[:10, 0] = 2.0
    candidate[:10, 7] = 4.0

    result = compute_action_chunk_distance(candidate, reference, config)
    assert result.primary_metric == PRIMARY_ACTION_CHUNK_DISTANCE_METRIC
    assert result.primary_distance == pytest.approx(np.sqrt(5 * 0.5**2))
    assert result.raw_l2["first_action"] == pytest.approx(np.sqrt(2.0**2 + 4.0**2))
    assert result.arm_only_raw_l2["full_chunk"] == pytest.approx(2.0 * np.sqrt(10))
    assert result.gripper_only_raw_l2["full_chunk"] == pytest.approx(4.0 * np.sqrt(10))
    assert result.action_range_normalized_l2["execution_window"] == pytest.approx(
        np.sqrt(5 * (0.5**2 + 0.5**2))
    )
    assert result.train_std_normalized_l2["first_5_actions"] == pytest.approx(
        np.sqrt(5 * (1.0**2 + 1.0**2))
    )
    assert result.cosine_distance["full_chunk"] is None
    assert set(result.raw_l2) == {
        "first_action",
        "first_5_actions",
        "execution_window",
        "full_chunk",
    }
    json.dumps(result.to_dict())


def test_action_chunk_cosine_is_zero_for_identical_nonzero_chunks() -> None:
    chunk = _chunk(object_value=1.0, bin_value=-1.0)
    result = compute_action_chunk_distance(chunk, chunk.copy(), _config())
    assert result.primary_distance == 0.0
    assert result.cosine_distance["first_action"] == pytest.approx(0.0)
    assert result.arm_only_cosine_distance["full_chunk"] == pytest.approx(0.0)
    assert result.gripper_only_cosine_distance["full_chunk"] is None


def test_action_chunk_distance_rejects_bad_shape_and_nonfinite_actions() -> None:
    config = _config()
    with pytest.raises(SemanticAuditError, match="shape"):
        compute_action_chunk_distance(np.zeros((49, 8)), np.zeros((50, 8)), config)
    invalid = np.zeros((50, 8))
    invalid[0, 0] = np.nan
    with pytest.raises(SemanticAuditError, match="finite"):
        compute_action_chunk_distance(invalid, np.zeros((50, 8)), config)


def test_task_retrieval_uses_primary_metric_and_canonical_tie_breaking() -> None:
    references = _references()
    requested = M43_CANONICAL_TASK_IDS[-1]
    exact = retrieve_task(
        observation_id="observation-0",
        requested_task_id=requested,
        candidate_chunk=references[requested],
        per_task_chunks=references,
        config=_config(),
    )
    assert exact.nearest_task_id == requested
    assert exact.correct_rank == 1
    assert exact.top1_correct and exact.top2_correct
    assert exact.correct_margin > 0.0

    tied_references = {task_id: _chunk() for task_id in M43_CANONICAL_TASK_IDS}
    tied = retrieve_task(
        observation_id="observation-tie",
        requested_task_id=requested,
        candidate_chunk=_chunk(),
        per_task_chunks=tied_references,
        config=_config(),
    )
    assert tied.ranked_task_ids == M43_CANONICAL_TASK_IDS
    assert tied.correct_rank == 6
    assert tied.reciprocal_rank == pytest.approx(1 / 6)
    assert tied.correct_margin == 0.0


def test_task_summary_reports_topk_mrr_margin_per_task_and_confusion() -> None:
    references = _references()
    results = tuple(
        retrieve_task(
            observation_id=f"observation-{index}",
            requested_task_id=task_id,
            candidate_chunk=references[task_id],
            per_task_chunks=references,
            config=_config(),
        )
        for index, task_id in enumerate(M43_CANONICAL_TASK_IDS)
    )
    summary = summarize_task_retrieval(results)
    assert summary["top1_accuracy"] == 1.0
    assert summary["top2_accuracy"] == 1.0
    assert summary["mean_reciprocal_rank"] == 1.0
    assert len(cast(dict[str, object], summary["per_task"])) == 6
    confusion = task_retrieval_confusion(results)
    assert confusion.total == 6
    assert all(confusion.counts[index][index] == 1 for index in range(6))


def test_object_retrieval_uses_pair_centroids_and_reports_precision_recall() -> None:
    references = _references()
    results = []
    for object_id in OBJECT_IDS:
        task_id = _task_id(object_id, "right_bin")
        results.append(
            retrieve_object(
                observation_id=f"object-{object_id}",
                requested_task_id=task_id,
                candidate_chunk=references[task_id],
                per_task_chunks=references,
                config=_config(),
            )
        )
    assert all(result.correct for result in results)
    summary = summarize_object_retrieval(results)
    assert summary["accuracy"] == 1.0
    metrics = cast(dict[str, dict[str, float]], summary["precision_recall"])
    assert all(value == {"precision": 1.0, "recall": 1.0} for value in metrics.values())


def test_bin_retrieval_reports_global_and_requested_object_conditional_results() -> None:
    references = _references()
    results = []
    for bin_id in BIN_IDS:
        task_id = _task_id("blue_cube", bin_id)
        results.append(
            retrieve_bin(
                observation_id=f"bin-{bin_id}",
                requested_task_id=task_id,
                candidate_chunk=references[task_id],
                per_task_chunks=references,
                config=_config(),
            )
        )
    assert all(result.correct and result.conditional_correct for result in results)
    summary = summarize_bin_retrieval(results)
    assert summary["accuracy"] == 1.0
    assert summary["conditional_accuracy"] == 1.0
    assert "conditional_confusion" in summary


def test_precision_recall_rejects_rectangular_first_interaction_confusion() -> None:
    task_id = M43_CANONICAL_TASK_IDS[0]
    record = build_first_interaction_record(
        observation_id="observation",
        requested_task_id=task_id,
        first_object_grasped=None,
        first_object_displaced=None,
        requested_destination_bin_id="left_bin",
        first_bin_approached=None,
        first_bin_entered_by_any_object=None,
        final_object_bin_relationship={object_id: None for object_id in OBJECT_IDS},
        nearest_per_task_task_id=task_id,
        trace=SemanticRolloutTrace(),
    )
    confusion = first_interaction_confusions((record,))["requested_object_to_first_grasped_object"]
    assert confusion.predicted_labels[-1] == "none"
    with pytest.raises(SemanticAuditError, match="identical"):
        precision_recall(confusion)


@pytest.mark.parametrize(
    ("changes", "expected"),
    (
        ({}, SemanticFailureClass.NEVER_GRASPED_TARGET),
        ({"wrong_object_grasped": True}, SemanticFailureClass.WRONG_OBJECT_INTERACTION),
        (
            {"ever_grasped_target": True},
            SemanticFailureClass.TARGET_GRASPED_NOT_LIFTED,
        ),
        (
            {"ever_grasped_target": True, "ever_lifted_target": True},
            SemanticFailureClass.LIFTED_NOT_TRANSPORTED,
        ),
        (
            {
                "ever_grasped_target": True,
                "ever_lifted_target": True,
                "ever_transported_to_destination": True,
            },
            SemanticFailureClass.TRANSPORTED_NOT_DESCENDED,
        ),
        (
            {
                "ever_grasped_target": True,
                "ever_lifted_target": True,
                "ever_transported_to_destination": True,
                "ever_descended_at_destination": True,
            },
            SemanticFailureClass.DESCENDED_NOT_RELEASED,
        ),
        (
            {
                "ever_grasped_target": True,
                "ever_lifted_target": True,
                "ever_transported_to_destination": True,
                "ever_descended_at_destination": True,
                "release_command_observed": True,
            },
            SemanticFailureClass.RELEASED_OUTSIDE_SUCCESS_REGION,
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
            SemanticFailureClass.RELEASED_BUT_NOT_STATIC,
        ),
        (
            {"success_observed_before_final_step": True},
            SemanticFailureClass.TARGET_SUCCESS_THEN_LOST,
        ),
        (
            {"ever_grasped_target": True, "timed_out": True},
            SemanticFailureClass.TIMED_OUT_AFTER_TARGET_GRASP,
        ),
        ({"environment_failure": True}, SemanticFailureClass.ENVIRONMENT_FAILURE),
        ({"success": True}, SemanticFailureClass.SUCCESS),
    ),
)
def test_failure_taxonomy_covers_exactly_twelve_classes(
    changes: dict[str, bool], expected: SemanticFailureClass
) -> None:
    assert classify_semantic_failure(replace(SemanticRolloutTrace(), **changes)) is expected


def test_first_interaction_record_derives_semantic_matches_and_all_confusions() -> None:
    requested = _task_id("red_cube", "left_bin")
    nearest = _task_id("green_cube", "right_bin")
    record = build_first_interaction_record(
        observation_id="scene-1/task-red-left",
        requested_task_id=requested,
        first_object_grasped="green_cube",
        first_object_displaced="green_cube",
        requested_destination_bin_id="left_bin",
        first_bin_approached="right_bin",
        first_bin_entered_by_any_object="right_bin",
        final_object_bin_relationship={
            "red_cube": None,
            "green_cube": "right_bin",
            "blue_cube": None,
        },
        nearest_per_task_task_id=nearest,
        trace=SemanticRolloutTrace(wrong_object_grasped=True),
    )
    assert record.first_grasp_matches_nearest_per_task is True
    assert record.initial_semantic_error_visible is True
    assert record.failure_class is SemanticFailureClass.WRONG_OBJECT_INTERACTION
    confusions = first_interaction_confusions((record,))
    assert set(confusions) == {
        "requested_object_to_first_grasped_object",
        "requested_task_to_nearest_per_task_chunk",
        "nearest_per_task_chunk_to_actual_first_grasp",
        "requested_bin_to_first_approached_bin",
        "requested_bin_to_object_entry_bin",
    }
    assert all(value.total == 1 for value in confusions.values())


def test_semantic_conclusions_require_explicit_semantic_evidence() -> None:
    insufficient = classify_semantic_conclusions(
        sufficient_evidence=False,
        condition_changes_output=True,
        correct_task_retrieval=True,
        correct_object_retrieval=True,
        correct_bin_retrieval=True,
        shared_control_degraded_after_correct_selection=False,
        post_grasp_execution_failed=False,
    )
    assert insufficient == (SemanticAuditConclusion.INSUFFICIENT_EVIDENCE,)

    conclusions = classify_semantic_conclusions(
        sufficient_evidence=True,
        condition_changes_output=True,
        correct_task_retrieval=False,
        correct_object_retrieval=False,
        correct_bin_retrieval=False,
        shared_control_degraded_after_correct_selection=True,
        post_grasp_execution_failed=True,
    )
    assert SemanticAuditConclusion.CONDITION_NOT_USED not in conclusions
    assert SemanticAuditConclusion.CONDITION_CHANGES_OUTPUT_BUT_WRONG_SEMANTICS in conclusions
    assert SemanticAuditConclusion.TARGET_OBJECT_CONFUSION in conclusions
    assert SemanticAuditConclusion.DESTINATION_BIN_CONFUSION in conclusions
    assert SemanticAuditConclusion.SHARED_CONTROL_DEGRADATION_AFTER_CORRECT_SELECTION in conclusions
    assert SemanticAuditConclusion.POST_GRASP_EXECUTION_FAILURE in conclusions

    aligned = classify_semantic_conclusions(
        sufficient_evidence=True,
        condition_changes_output=True,
        correct_task_retrieval=True,
        correct_object_retrieval=True,
        correct_bin_retrieval=True,
        shared_control_degraded_after_correct_selection=False,
        post_grasp_execution_failed=False,
    )
    assert aligned == ()


def test_semantic_alignment_audit_is_compact_immutable_json_and_keeps_action_layers_separate() -> (
    None
):
    references = _references()
    requested = M43_CANONICAL_TASK_IDS[0]
    task = retrieve_task(
        observation_id="observation",
        requested_task_id=requested,
        candidate_chunk=references[requested],
        per_task_chunks=references,
        config=_config(),
    )
    object_result = retrieve_object(
        observation_id="observation",
        requested_task_id=requested,
        candidate_chunk=references[requested],
        per_task_chunks=references,
        config=_config(),
    )
    bin_result = retrieve_bin(
        observation_id="observation",
        requested_task_id=requested,
        candidate_chunk=references[requested],
        per_task_chunks=references,
        config=_config(),
    )
    interaction = build_first_interaction_record(
        observation_id="observation",
        requested_task_id=requested,
        first_object_grasped=None,
        first_object_displaced=None,
        requested_destination_bin_id="left_bin",
        first_bin_approached=None,
        first_bin_entered_by_any_object=None,
        final_object_bin_relationship={object_id: None for object_id in OBJECT_IDS},
        nearest_per_task_task_id=requested,
        trace=SemanticRolloutTrace(),
    )
    audit = build_semantic_alignment_audit(
        policy_label="state_onehot",
        distance_config=_config(),
        task_results=(task,),
        object_results=(object_result,),
        bin_results=(bin_result,),
        first_interactions=(interaction,),
        conclusions=(SemanticAuditConclusion.POST_GRASP_EXECUTION_FAILURE,),
        raw_policy_action_metrics={"out_of_bounds_rate": 0.1},
        runtime_action_metrics={"projected_rate": 0.1},
        deterministic_reload_validated=True,
        deterministic_reset_validated=True,
    )
    assert audit.failure_distribution[SemanticFailureClass.NEVER_GRASPED_TARGET.value] == 1
    assert audit.raw_policy_action_metrics["out_of_bounds_rate"] == 0.1
    assert audit.runtime_action_metrics["projected_rate"] == 0.1
    payload = audit.to_dict()
    assert "task_retrieval" in payload and "first_interactions" in payload
    assert "images" not in payload and "trajectories" not in payload
    json.dumps(payload)
    with pytest.raises(TypeError):
        audit.raw_policy_action_metrics["mutate"] = True  # type: ignore[index]


def test_retrieval_rejects_partial_reference_set() -> None:
    references = _references()
    references.pop(M43_CANONICAL_TASK_IDS[-1])
    with pytest.raises(SemanticAuditError, match="exactly the six"):
        retrieve_task(
            observation_id="observation",
            requested_task_id=M43_CANONICAL_TASK_IDS[0],
            candidate_chunk=_chunk(),
            per_task_chunks=references,
            config=_config(),
        )
