"""Pure contracts for M4.3b validation, reload, and development evidence."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import stable_scene_id, stable_task_id
from langmani.policies.act_factor_film_evaluation import (
    FACTOR_FILM_DEVELOPMENT_POLICY_LABELS,
    DevelopmentEpisodeIdentity,
    DevelopmentGateCriterion,
    DevelopmentPolicyMetrics,
    DevelopmentQualityGateResult,
    DevelopmentSemanticMetrics,
    FactorFiLMEvaluationError,
    FactorFiLMReloadValidationRecord,
    FactorFiLMSemanticStepSnapshot,
    FactorFiLMValidationCountResult,
    SemanticErrorStage,
    derive_first_interaction_snapshot,
    evaluate_development_quality_gate,
    factor_film_first_interaction_confusions,
    factor_film_validation_count_result,
    rank_factor_film_validation_count_results,
    validate_paired_development_identities,
    validate_reload_against_selection,
)
from langmani.policies.act_factor_film_training import create_factor_film_selection
from langmani.policies.act_factor_film_types import (
    FactorFiLMSelectionRecord,
    FactorFiLMValidationQueue,
    FactorFiLMValidationQueueItem,
    canonical_fingerprint,
)
from langmani.policies.m42_schedule import M42_DEV_SCHEDULE_FINGERPRINT
from langmani.policies.m43_types import SemanticFailureClass

_DIGEST_A = "sha256:" + "a" * 64
_DIGEST_B = "sha256:" + "b" * 64
_DIGEST_C = "sha256:" + "c" * 64
_DIGEST_D = "sha256:" + "d" * 64
_DIGEST_E = "sha256:" + "e" * 64
_SCHEDULE = "sha256:" + "f" * 64
_TASK_IDS = tuple(stable_task_id(task) for task in CANONICAL_TASK_SPECS)


def _validation_counts() -> tuple[FactorFiLMValidationCountResult, ...]:
    values = tuple(
        FactorFiLMValidationCountResult(
            checkpoint_fingerprint=f"sha256:{index:064x}",
            checkpoint_step=step,
            schedule_digest=_SCHEDULE,
            success_count=18,
            wrong_object_interaction_count=4,
            wrong_object_in_target_bin_count=2,
            target_off_table_count=1,
            timeout_count=10,
            offline_validation_action_loss=0.5,
        )
        for index, step in enumerate(range(5_000, 100_001, 5_000), start=1)
    )
    values = (
        values[0],
        replace(values[1], success_count=30, wrong_object_interaction_count=1),
        replace(values[2], success_count=30, wrong_object_interaction_count=2),
        *values[3:],
    )
    return values


def _selection() -> FactorFiLMSelectionRecord:
    counts = _validation_counts()
    queue = FactorFiLMValidationQueue(
        run_fingerprint=_DIGEST_A,
        validation_schedule_digest=_SCHEDULE,
        items=tuple(
            FactorFiLMValidationQueueItem(
                checkpoint_fingerprint=value.checkpoint_fingerprint,
                checkpoint_step=value.checkpoint_step,
                checkpoint_relative_path=f"checkpoints/step-{value.checkpoint_step:08d}",
                offline_validation_action_loss=value.offline_validation_action_loss,
            )
            for value in counts
        ),
    )
    return create_factor_film_selection(
        validation_queue=queue,
        candidates=tuple(value.to_selection_result() for value in counts),
    )


def test_validation_counts_roundtrip_conversion_and_seven_level_ranking() -> None:
    candidates = _validation_counts()
    ranked = rank_factor_film_validation_count_results(tuple(reversed(candidates)))
    ranked_rates = tuple(
        sorted(
            (value.to_selection_result() for value in candidates),
            key=lambda value: value.ranking_key,
        )
    )

    assert ranked[0].checkpoint_step == 10_000
    assert ranked[1].checkpoint_step == 15_000
    assert tuple(value.checkpoint_fingerprint for value in ranked) == tuple(
        value.checkpoint_fingerprint for value in ranked_rates
    )
    assert FactorFiLMValidationCountResult.from_dict(ranked[0].to_dict()) == ranked[0]
    json.dumps(ranked[0].to_dict(), sort_keys=True)

    aggregate = {
        "episode_count": 36,
        "successes": 30,
        "wrong_object_interaction_count": 1,
        "wrong_object_in_target_bin_count": 0,
        "target_off_table_count": 0,
        "timeout_count": 6,
    }
    projected = factor_film_validation_count_result(
        checkpoint_fingerprint=_DIGEST_A,
        checkpoint_step=5_000,
        schedule_digest=_SCHEDULE,
        aggregate=aggregate,
        offline_validation_action_loss=0.25,
    )
    assert projected.success_count == 30
    assert projected.to_selection_result().task_success_rate == pytest.approx(30 / 36)
    assert projected.development_schedule_accessed is False
    assert projected.test_split_accessed is False
    assert projected.fresh_seed_accessed is False
    assert projected.final_schedule_accessed is False


def test_validation_counts_reject_partial_or_forbidden_evidence() -> None:
    with pytest.raises(FactorFiLMEvaluationError, match="all 20"):
        rank_factor_film_validation_count_results(_validation_counts()[:-1])
    with pytest.raises(FactorFiLMEvaluationError, match="forbidden"):
        replace(_validation_counts()[0], development_schedule_accessed=True)
    with pytest.raises(FactorFiLMEvaluationError, match="missing timeout_count"):
        factor_film_validation_count_result(
            checkpoint_fingerprint=_DIGEST_A,
            checkpoint_step=5_000,
            schedule_digest=_SCHEDULE,
            aggregate={
                "episode_count": 36,
                "successes": 1,
                "wrong_object_interaction_count": 0,
                "wrong_object_in_target_bin_count": 0,
                "target_off_table_count": 0,
            },
            offline_validation_action_loss=0.1,
        )


def _reload_record(**overrides: object) -> FactorFiLMReloadValidationRecord:
    selection = _selection()
    values: dict[str, object] = {
        "run_fingerprint": selection.run_fingerprint,
        "selection_fingerprint": canonical_fingerprint(selection.to_dict()),
        "selected_checkpoint_fingerprint": selection.selected_checkpoint_fingerprint,
        "selected_checkpoint_step": selection.selected_checkpoint_step,
        "architecture_fingerprint": _DIGEST_B,
        "preprocessor_fingerprint": _DIGEST_C,
        "policy_postprocessor_fingerprint": _DIGEST_D,
        "action_runtime_fingerprint": _DIGEST_E,
        "validation_observation_fingerprint": _DIGEST_B,
        "reference_output_fingerprint": _DIGEST_C,
        "reloaded_output_fingerprint": _DIGEST_D,
        "maximum_absolute_error": 1e-7,
        "maximum_relative_error": 1e-7,
        "fresh_policy_instance": True,
        "preprocessor_reloaded": True,
        "policy_postprocessor_reloaded": True,
        "architecture_reconstructed": True,
        "task_mappings_reconstructed": True,
        "action_runtime_reconstructed": True,
        "fingerprints_validated": True,
        "deterministic_inference_validated": True,
        "reload_validated": True,
    }
    values.update(overrides)
    return FactorFiLMReloadValidationRecord(**values)  # type: ignore[arg-type]


def test_reload_record_binds_selection_and_roundtrips() -> None:
    selection = _selection()
    record = _reload_record()
    validate_reload_against_selection(record, selection)
    assert FactorFiLMReloadValidationRecord.from_dict(record.to_dict()) == record
    json.dumps(record.to_dict(), sort_keys=True)

    with pytest.raises(FactorFiLMEvaluationError, match="differs"):
        validate_reload_against_selection(
            replace(record, selected_checkpoint_fingerprint=_DIGEST_E), selection
        )
    with pytest.raises(FactorFiLMEvaluationError, match="forbidden"):
        replace(record, final_schedule_accessed=True)
    with pytest.raises(FactorFiLMEvaluationError, match="disagrees"):
        replace(record, preprocessor_reloaded=False)


def _step(
    step: int,
    *,
    red_position: tuple[float, float, float],
    red_grasped: bool,
    red_in_left: bool,
    action_gripper: float | None,
    success: bool = False,
) -> FactorFiLMSemanticStepSnapshot:
    return FactorFiLMSemanticStepSnapshot(
        step=step,
        cube_positions=(red_position, (0.0, 0.3, 0.02), (0.0, -0.3, 0.02)),
        cube_orientations=((0.0, 0.0, 0.0, 1.0),) * 3,
        cube_linear_velocities=((0.0, 0.0, 0.0),) * 3,
        cube_angular_velocities=((0.0, 0.0, 0.0),) * 3,
        bin_floor_centers=((0.2, 0.0, 0.0), (0.4, 0.0, 0.0)),
        tcp_position=red_position,
        object_is_grasped=(red_grasped, False, False),
        object_in_bin=((red_in_left, False), (False, False), (False, False)),
        executed_action=(None if action_gripper is None else (0.0,) * 7 + (action_gripper,)),
        evaluation={
            "success": success,
            "target_in_target_bin": red_in_left,
            "target_is_static": success,
            "target_off_table": False,
        },
    )


def test_first_interaction_is_purely_derived_and_uses_existing_confusions() -> None:
    requested = _TASK_IDS[0]
    snapshots = (
        _step(
            0,
            red_position=(0.0, 0.0, 0.02),
            red_grasped=False,
            red_in_left=False,
            action_gripper=None,
        ),
        _step(
            1,
            red_position=(0.0, 0.0, 0.02),
            red_grasped=True,
            red_in_left=False,
            action_gripper=-1.0,
        ),
        _step(
            2,
            red_position=(0.2, 0.0, 0.10),
            red_grasped=True,
            red_in_left=False,
            action_gripper=-1.0,
        ),
        _step(
            3,
            red_position=(0.2, 0.0, 0.02),
            red_grasped=False,
            red_in_left=True,
            action_gripper=1.0,
            success=True,
        ),
    )
    result = derive_first_interaction_snapshot(
        observation_id="scene-red-left",
        requested_task_id=requested,
        nearest_per_task_task_id=requested,
        step_snapshots=tuple(value.to_dict() for value in snapshots),
        timed_out=False,
        environment_failure=False,
        raw_actions_by_step={
            1: (0.0,) * 7 + (-1.0,),
            2: (0.0,) * 7 + (-1.0,),
            3: (0.0,) * 7 + (1.0,),
        },
    )

    assert result.first_object_grasped == "red_cube"
    assert result.first_object_displaced == "red_cube"
    assert result.first_bin_approached == "left_bin"
    assert result.first_bin_entered_by_any_object == "left_bin"
    assert result.first_target_grasp_step == 1
    assert result.first_release_command_step == 3
    assert result.failure_class is SemanticFailureClass.SUCCESS
    assert result.semantic_error_stage is SemanticErrorStage.NONE
    assert result.final_object_bin_relationship["red_cube"] == "left_bin"
    assert len(result.final_target_pose) == 7
    assert len(result.final_target_velocity) == 6
    assert result.to_first_interaction_record().first_grasp_matches_nearest_per_task is True
    assert result == type(result).from_dict(result.to_dict())
    confusions = factor_film_first_interaction_confusions((result,))
    assert set(confusions) == {
        "requested_object_to_first_grasped_object",
        "requested_task_to_nearest_per_task_chunk",
        "nearest_per_task_chunk_to_actual_first_grasp",
        "requested_bin_to_first_approached_bin",
        "requested_bin_to_object_entry_bin",
    }
    assert all(sum(sum(row) for row in value["counts"]) == 1 for value in confusions.values())
    json.dumps(result.to_dict(), sort_keys=True)

    projected_release = derive_first_interaction_snapshot(
        observation_id="scene-red-left-projected",
        requested_task_id=requested,
        nearest_per_task_task_id=requested,
        step_snapshots=tuple(value.to_dict() for value in snapshots),
        timed_out=False,
        environment_failure=False,
        raw_actions_by_step={
            1: (0.0,) * 7 + (-1.0,),
            2: (0.0,) * 7 + (1.0,),
            3: (0.0,) * 7 + (1.0,),
        },
    )
    assert projected_release.first_release_command_step == 2


def test_first_interaction_rejects_incomplete_and_canonically_breaks_ties() -> None:
    reset = _step(
        0,
        red_position=(0.0, 0.0, 0.02),
        red_grasped=False,
        red_in_left=False,
        action_gripper=None,
    )
    with pytest.raises(FactorFiLMEvaluationError, match="reset step zero"):
        derive_first_interaction_snapshot(
            observation_id="bad",
            requested_task_id=_TASK_IDS[0],
            nearest_per_task_task_id=_TASK_IDS[0],
            step_snapshots=(),
            timed_out=False,
            environment_failure=False,
        )
    simultaneous = replace(
        reset, step=1, object_is_grasped=(True, True, False), executed_action=(0.0,) * 8
    )
    tied = derive_first_interaction_snapshot(
        observation_id="canonical-tie",
        requested_task_id=_TASK_IDS[0],
        nearest_per_task_task_id=_TASK_IDS[0],
        step_snapshots=(reset, simultaneous),
        timed_out=True,
        environment_failure=False,
    )
    assert tied.first_object_grasped == "red_cube"

    wrong_bin = derive_first_interaction_snapshot(
        observation_id="wrong-bin-emerged",
        requested_task_id=_TASK_IDS[0],
        nearest_per_task_task_id=_TASK_IDS[0],
        step_snapshots=(
            reset,
            _step(
                1,
                red_position=(0.4, 0.0, 0.08),
                red_grasped=True,
                red_in_left=False,
                action_gripper=-1.0,
            ),
        ),
        timed_out=True,
        environment_failure=False,
    )
    assert wrong_bin.first_bin_approached == "right_bin"
    assert wrong_bin.semantic_error_stage is SemanticErrorStage.EMERGED_AFTER_INITIAL_CHUNK


def _episode_identities() -> tuple[DevelopmentEpisodeIdentity, ...]:
    return tuple(
        DevelopmentEpisodeIdentity(
            episode_index=scene_index * 6 + task_index,
            scene_seed=10_000 + scene_index,
            scene_id=stable_scene_id(10_000 + scene_index),
            task_id=task_id,
        )
        for scene_index in range(12)
        for task_index, task_id in enumerate(_TASK_IDS)
    )


def test_paired_development_identity_requires_same_ordered_72() -> None:
    episodes = _episode_identities()
    paired = validate_paired_development_identities(
        per_task=episodes,
        state_onehot=episodes,
        factor_film=episodes,
        schedule_fingerprint=M42_DEV_SCHEDULE_FINGERPRINT,
    )
    assert paired.episode_count == 72
    assert paired.policy_labels == FACTOR_FILM_DEVELOPMENT_POLICY_LABELS
    assert len(paired.policy_episode_fingerprints) == 3
    assert paired.test_split_accessed is False
    assert paired.fresh_seed_accessed is False
    assert paired.final_schedule_accessed is False
    assert type(paired).from_dict(paired.to_dict()) == paired
    json.dumps(paired.to_dict(), sort_keys=True)

    changed = (*episodes[:-1], replace(episodes[-1], scene_seed=99, scene_id=stable_scene_id(99)))
    with pytest.raises(FactorFiLMEvaluationError, match="identical ordered"):
        validate_paired_development_identities(
            per_task=episodes,
            state_onehot=episodes,
            factor_film=changed,
            schedule_fingerprint=M42_DEV_SCHEDULE_FINGERPRINT,
        )


def _development_metrics(**overrides: object) -> DevelopmentPolicyMetrics:
    values: dict[str, object] = {
        "report_fingerprint": _DIGEST_A,
        "success_count": 50,
        "per_task_success_counts": dict(zip(_TASK_IDS, (9, 9, 8, 8, 8, 8), strict=True)),
        "wrong_object_grasp_count": 6,
        "wrong_object_in_target_bin_count": 2,
        "target_in_wrong_bin_count": 0,
        "target_off_table_count": 0,
        "timeout_count": 22,
        "arm_projection_count": 0,
        "nan_count": 0,
        "inf_count": 0,
        "malformed_action_count": 0,
    }
    values.update(overrides)
    return DevelopmentPolicyMetrics(**values)  # type: ignore[arg-type]


def _semantic_metrics(**overrides: object) -> DevelopmentSemanticMetrics:
    values: dict[str, object] = {
        "semantic_report_fingerprint": _DIGEST_B,
        "full_task_top1_retrieval": 0.70,
        "target_object_retrieval": 0.80,
        "task_sensitivity_ratio_relative_to_per_task": 0.75,
    }
    values.update(overrides)
    return DevelopmentSemanticMetrics(**values)  # type: ignore[arg-type]


def test_development_quality_gate_passes_every_exact_inclusive_boundary() -> None:
    result = evaluate_development_quality_gate(
        factor_film_metrics=_development_metrics(),
        semantic_metrics=_semantic_metrics(),
        per_task_reference_success_count=58,
    )
    assert result.development_quality_gate_passed is True
    assert result.final_benchmark_authorized is True
    assert len(result.checks) == 16
    assert tuple(value.criterion for value in result.checks) == tuple(DevelopmentGateCriterion)
    assert result.failed_criteria == ()
    assert result.final_schedule_accessed is False
    assert result.test_split_accessed is False
    assert result.fresh_seed_accessed is False
    assert result.smolvla_go is False
    assert DevelopmentQualityGateResult.from_dict(result.to_dict()) == result
    json.dumps(result.to_dict(), sort_keys=True)


@pytest.mark.parametrize(
    ("metrics", "semantic", "per_task_success", "failed"),
    (
        (
            _development_metrics(
                success_count=49,
                per_task_success_counts=dict(zip(_TASK_IDS, (9, 8, 8, 8, 8, 8), strict=True)),
            ),
            _semantic_metrics(),
            57,
            "overall_success_count",
        ),
        (_development_metrics(), _semantic_metrics(), 59, "success_gap_to_per_task"),
        (
            _development_metrics(
                per_task_success_counts=dict(zip(_TASK_IDS, (12, 9, 8, 8, 7, 6), strict=True))
            ),
            _semantic_metrics(),
            58,
            "every_task_success_count",
        ),
        (
            _development_metrics(wrong_object_grasp_count=7),
            _semantic_metrics(),
            58,
            "wrong_object_grasp_count",
        ),
        (
            _development_metrics(wrong_object_in_target_bin_count=3),
            _semantic_metrics(),
            58,
            "wrong_object_in_target_bin_count",
        ),
        (_development_metrics(timeout_count=23), _semantic_metrics(), 58, "timeout_count"),
        (
            _development_metrics(target_in_wrong_bin_count=1),
            _semantic_metrics(),
            58,
            "target_in_wrong_bin_count",
        ),
        (
            _development_metrics(target_off_table_count=1),
            _semantic_metrics(),
            58,
            "target_off_table_count",
        ),
        (
            _development_metrics(arm_projection_count=1),
            _semantic_metrics(),
            58,
            "arm_projection_count",
        ),
        (_development_metrics(nan_count=1), _semantic_metrics(), 58, "nan_count"),
        (_development_metrics(inf_count=1), _semantic_metrics(), 58, "inf_count"),
        (
            _development_metrics(malformed_action_count=1),
            _semantic_metrics(),
            58,
            "malformed_action_count",
        ),
        (
            _development_metrics(),
            _semantic_metrics(full_task_top1_retrieval=0.699999),
            58,
            "full_task_top1_retrieval",
        ),
        (
            _development_metrics(),
            _semantic_metrics(target_object_retrieval=0.799999),
            58,
            "target_object_retrieval",
        ),
        (
            _development_metrics(),
            _semantic_metrics(task_sensitivity_ratio_relative_to_per_task=0.749999),
            58,
            "task_sensitivity_ratio_relative_to_per_task",
        ),
    ),
)
def test_development_quality_gate_does_not_round_or_weaken_failures(
    metrics: DevelopmentPolicyMetrics,
    semantic: DevelopmentSemanticMetrics,
    per_task_success: int,
    failed: str,
) -> None:
    result = evaluate_development_quality_gate(
        factor_film_metrics=metrics,
        semantic_metrics=semantic,
        per_task_reference_success_count=per_task_success,
    )
    assert result.development_quality_gate_passed is False
    assert result.final_benchmark_authorized is False
    assert failed in result.failed_criteria
    assert result.final_schedule_accessed is False
    assert result.smolvla_go is False


def test_development_records_reject_test_fresh_final_and_smolvla_claims() -> None:
    with pytest.raises(FactorFiLMEvaluationError, match="forbidden"):
        replace(_development_metrics(), test_split_accessed=True)
    with pytest.raises(FactorFiLMEvaluationError, match="forbidden"):
        replace(_semantic_metrics(), fresh_seed_accessed=True)
    valid = evaluate_development_quality_gate(
        factor_film_metrics=_development_metrics(),
        semantic_metrics=_semantic_metrics(),
        per_task_reference_success_count=58,
    )
    with pytest.raises(FactorFiLMEvaluationError, match="future work"):
        replace(valid, final_schedule_accessed=True)
    with pytest.raises(FactorFiLMEvaluationError, match="future work"):
        replace(valid, smolvla_go=True)
