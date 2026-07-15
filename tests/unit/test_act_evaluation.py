from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from langmani.datasets.lerobot_types import IMAGE_FEATURE_KEY, STATE_FEATURE_KEY
from langmani.policies.act_action_bounds import (
    ActionBoundMode,
    ActionProjectionSummary,
)
from langmani.policies.act_analysis import (
    CounterfactualAnalysisError,
    CounterfactualDemonstration,
    audit_counterfactual_demonstrations,
    compute_counterfactual_sensitivity,
    counterfactual_sensitivity_from_dict,
    expert_chunk_distances,
    policy_inputs_identical,
    task_identity_effects,
)
from langmani.policies.act_conditioning import CANONICAL_TASK_IDS
from langmani.policies.act_evaluation import (
    EvaluationContractError,
    FreshSeedSchedule,
    SelectionLockError,
    authorize_test_evaluation,
    build_fresh_seed_schedule,
    create_checkpoint_selection,
    load_checkpoint_selection,
    rank_validation_results,
    summarize_rollout_benchmark,
    wilson_interval,
    write_checkpoint_selection_atomic,
)
from langmani.policies.act_evaluation import (
    TestEvaluationAuthorization as EvaluationAuthorization,
)
from langmani.policies.act_types import (
    TASK_ONEHOT_MAPPING_VERSION,
    ActRunIdentity,
    ActVariant,
    EvaluationSplit,
    ExperimentMode,
    RolloutEpisodeResult,
    RolloutStatus,
    ValidationResult,
)


def _digest(character: str) -> str:
    return f"sha256:{character * 64}"


def test_offline_counterfactual_audit_uses_pad_intersection_and_equal_observations() -> None:
    demonstrations = tuple(
        CounterfactualDemonstration(
            task_id=task_id,
            rgb_digest="same-rgb",
            panda_state=(0.0,) * 9,
            action_chunk=((float(index),) * 8, (99.0,) * 8),
            action_is_pad=(False, index % 2 == 0),
        )
        for index, task_id in enumerate(CANONICAL_TASK_IDS)
    )
    report = audit_counterfactual_demonstrations({"group": demonstrations})
    assert report.passed
    assert report.identical_initial_rgb_fraction == 1.0
    assert report.codec_equivalent_initial_rgb_fraction == 1.0
    assert report.identical_initial_state_fraction == 1.0
    assert report.one_to_many_action_fraction == 1.0
    assert report.maximum_pairwise_initial_rgb_mae == 0.0
    assert report.minimum_pairwise_initial_rgb_psnr_db == 999.0
    assert len(report.mean_pairwise_chunk_distance_by_task_pair) == 15


def test_offline_counterfactual_audit_accepts_only_bounded_codec_differences() -> None:
    demonstrations = tuple(
        CounterfactualDemonstration(
            task_id=task_id,
            rgb_digest=f"rgb-{index}",
            panda_state=(0.0,) * 9,
            action_chunk=((float(index),) * 8,),
            action_is_pad=(False,),
            initial_rgb=np.full((3, 4, 4), index, dtype=np.uint8),
        )
        for index, task_id in enumerate(CANONICAL_TASK_IDS)
    )
    report = audit_counterfactual_demonstrations({"group": demonstrations})
    assert report.passed
    assert report.identical_initial_rgb_fraction == 0.0
    assert report.codec_equivalent_initial_rgb_fraction == 1.0
    assert report.maximum_pairwise_initial_rgb_mae == 5.0
    assert report.minimum_pairwise_initial_rgb_psnr_db > 30.0

    drifted = list(demonstrations)
    drifted[-1] = replace(
        drifted[-1],
        initial_rgb=np.full((3, 4, 4), 255, dtype=np.uint8),
    )
    failed = audit_counterfactual_demonstrations({"group": tuple(drifted)})
    assert not failed.passed
    assert failed.codec_equivalent_initial_rgb_fraction == 0.0
    assert failed.maximum_pairwise_initial_rgb_mae == 255.0
    assert failed.minimum_pairwise_initial_rgb_psnr_db == 0.0


def _validation(
    character: str,
    step: int,
    *,
    success: float,
    wrong: float = 0.0,
    off_table: float = 0.0,
    loss: float = 1.0,
    schedule: str = "b",
) -> ValidationResult:
    return ValidationResult(
        checkpoint_fingerprint=_digest(character),
        checkpoint_step=step,
        schedule_digest=_digest(schedule),
        success_rate=success,
        wrong_object_interaction_rate=wrong,
        target_off_table_rate=off_table,
        offline_validation_action_loss=loss,
    )


def _selection_candidates() -> tuple[ValidationResult, ...]:
    return (
        _validation("1", 100, success=0.8, wrong=0.2, off_table=0.1, loss=0.3),
        _validation("2", 200, success=0.9, wrong=0.2, off_table=0.1, loss=0.3),
        _validation("3", 300, success=0.9, wrong=0.1, off_table=0.1, loss=0.3),
        _validation("4", 400, success=0.9, wrong=0.1, off_table=0.0, loss=0.3),
        _validation("5", 500, success=0.9, wrong=0.1, off_table=0.0, loss=0.2),
        _validation("6", 450, success=0.9, wrong=0.1, off_table=0.0, loss=0.2),
    )


def test_checkpoint_ranking_uses_predeclared_validation_order() -> None:
    ranked = rank_validation_results(_selection_candidates())
    assert [item.checkpoint_fingerprint for item in ranked] == [
        _digest("6"),
        _digest("5"),
        _digest("4"),
        _digest("3"),
        _digest("2"),
        _digest("1"),
    ]


def test_checkpoint_ranking_rejects_schedule_or_checkpoint_reuse() -> None:
    first = _validation("1", 100, success=0.5)
    with pytest.raises(EvaluationContractError, match="one validation schedule"):
        rank_validation_results((first, _validation("2", 200, success=0.6, schedule="c")))
    with pytest.raises(EvaluationContractError, match="fingerprints must be unique"):
        rank_validation_results((first, replace(first, checkpoint_step=200)))
    with pytest.raises(EvaluationContractError, match="steps must be unique"):
        rank_validation_results((first, _validation("2", first.checkpoint_step, success=0.6)))


def test_checkpoint_selection_is_serializable_and_timestamp_not_semantic() -> None:
    left = create_checkpoint_selection(
        run_fingerprint=_digest("a"),
        candidates=_selection_candidates(),
        selection_timestamp_utc="2026-07-13T01:00:00Z",
    )
    right = create_checkpoint_selection(
        run_fingerprint=_digest("a"),
        candidates=_selection_candidates(),
        selection_timestamp_utc="2026-07-13T02:00:00Z",
    )
    assert left.selected_checkpoint_fingerprint == _digest("6")
    assert left.selected_checkpoint_step == 450
    assert left.selection_fingerprint == right.selection_fingerprint
    restored = type(left).from_dict(json.loads(json.dumps(left.to_dict())))
    assert restored == left


def test_checkpoint_selection_canonicalizes_candidate_enumeration_order() -> None:
    forward = create_checkpoint_selection(
        run_fingerprint=_digest("a"),
        candidates=_selection_candidates(),
        selection_timestamp_utc="2026-07-13T01:00:00Z",
    )
    reverse = create_checkpoint_selection(
        run_fingerprint=_digest("a"),
        candidates=tuple(reversed(_selection_candidates())),
        selection_timestamp_utc="2026-07-13T02:00:00Z",
    )
    assert forward.candidates == reverse.candidates
    assert forward.selection_fingerprint == reverse.selection_fingerprint


def test_checkpoint_selection_rejects_tampered_ranking() -> None:
    record = create_checkpoint_selection(
        run_fingerprint=_digest("a"),
        candidates=_selection_candidates(),
        selection_timestamp_utc="2026-07-13T01:00:00Z",
    )
    with pytest.raises(EvaluationContractError, match="stored checkpoint ranking"):
        replace(
            record,
            ranked_checkpoint_fingerprints=tuple(reversed(record.ranked_checkpoint_fingerprints)),
        )


def test_checkpoint_selection_atomic_publication_is_immutable(tmp_path: Path) -> None:
    path = tmp_path / "selection" / "checkpoint_selection.json"
    record = create_checkpoint_selection(
        run_fingerprint=_digest("a"),
        candidates=_selection_candidates(),
        selection_timestamp_utc="2026-07-13T01:00:00Z",
    )
    assert write_checkpoint_selection_atomic(path, record) == record
    assert write_checkpoint_selection_atomic(path, record) == record
    assert load_checkpoint_selection(path) == record
    different = create_checkpoint_selection(
        run_fingerprint=_digest("f"),
        candidates=_selection_candidates(),
        selection_timestamp_utc="2026-07-13T01:00:00Z",
    )
    with pytest.raises(SelectionLockError, match="immutable"):
        write_checkpoint_selection_atomic(path, different)
    assert load_checkpoint_selection(path) == record
    assert not tuple(path.parent.glob("*.staging"))


def test_full_test_lock_requires_selected_checkpoint_and_exact_schedule() -> None:
    selection = create_checkpoint_selection(
        run_fingerprint=_digest("a"),
        candidates=_selection_candidates(),
        selection_timestamp_utc="2026-07-13T01:00:00Z",
    )
    authorization = authorize_test_evaluation(
        run_fingerprint=_digest("a"),
        checkpoint_fingerprint=selection.selected_checkpoint_fingerprint,
        actual_schedule_digest=_digest("d"),
        expected_schedule_digest=_digest("d"),
        mode=ExperimentMode.FULL,
        selection=selection,
    )
    assert authorization.authorized
    assert authorization.final_eligible
    assert not authorization.development_override

    arbitrary = authorize_test_evaluation(
        run_fingerprint=_digest("a"),
        checkpoint_fingerprint=_digest("5"),
        actual_schedule_digest=_digest("d"),
        expected_schedule_digest=_digest("d"),
        mode=ExperimentMode.FULL,
        selection=selection,
    )
    assert not arbitrary.authorized
    assert not arbitrary.final_eligible
    assert "not the validation-selected checkpoint" in arbitrary.reason

    changed_schedule = authorize_test_evaluation(
        run_fingerprint=_digest("a"),
        checkpoint_fingerprint=selection.selected_checkpoint_fingerprint,
        actual_schedule_digest=_digest("e"),
        expected_schedule_digest=_digest("d"),
        mode=ExperimentMode.FULL,
        selection=selection,
    )
    assert not changed_schedule.authorized
    assert "schedule differs" in changed_schedule.reason

    changed_run = authorize_test_evaluation(
        run_fingerprint=_digest("f"),
        checkpoint_fingerprint=selection.selected_checkpoint_fingerprint,
        actual_schedule_digest=_digest("d"),
        expected_schedule_digest=_digest("d"),
        mode=ExperimentMode.FULL,
        selection=selection,
    )
    assert not changed_run.authorized
    assert "selection run" in changed_run.reason


def test_missing_selection_requires_explicit_development_override_and_stays_nonfinal() -> None:
    denied = authorize_test_evaluation(
        run_fingerprint=_digest("a"),
        checkpoint_fingerprint=_digest("1"),
        actual_schedule_digest=_digest("d"),
        expected_schedule_digest=_digest("d"),
        mode=ExperimentMode.DEVELOPMENT,
        selection=None,
    )
    assert not denied.authorized

    allowed = authorize_test_evaluation(
        run_fingerprint=_digest("a"),
        checkpoint_fingerprint=_digest("1"),
        actual_schedule_digest=_digest("d"),
        expected_schedule_digest=_digest("d"),
        mode=ExperimentMode.DEVELOPMENT,
        selection=None,
        development_override_reason="exercise test command wiring",
    )
    assert allowed.authorized
    assert allowed.development_override
    assert not allowed.final_eligible
    restored = EvaluationAuthorization.from_dict(json.loads(json.dumps(allowed.to_dict())))
    assert restored == allowed


def test_full_mode_override_is_recorded_but_never_authorizes() -> None:
    authorization = authorize_test_evaluation(
        run_fingerprint=_digest("a"),
        checkpoint_fingerprint=_digest("1"),
        actual_schedule_digest=_digest("d"),
        expected_schedule_digest=_digest("d"),
        mode=ExperimentMode.FULL,
        selection=None,
        development_override_reason="not permitted",
    )
    assert not authorization.authorized
    assert not authorization.final_eligible
    assert "cannot authorize full mode" in authorization.reason


def test_dry_run_cannot_access_test_even_with_a_development_reason() -> None:
    selection = create_checkpoint_selection(
        run_fingerprint=_digest("a"),
        candidates=_selection_candidates(),
        selection_timestamp_utc="2026-07-13T01:00:00Z",
    )
    authorization = authorize_test_evaluation(
        run_fingerprint=_digest("a"),
        checkpoint_fingerprint=selection.selected_checkpoint_fingerprint,
        actual_schedule_digest=_digest("d"),
        expected_schedule_digest=_digest("d"),
        mode=ExperimentMode.DRY_RUN,
        selection=selection,
        development_override_reason="not a development run",
    )
    assert not authorization.authorized
    assert not authorization.development_override
    assert not authorization.final_eligible


def test_fresh_seed_schedule_is_deterministic_and_excludes_all_available_source_seeds() -> None:
    kwargs = {
        "m3b_export_fingerprint": _digest("a"),
        "accepted_source_seeds": (9, 1, 5, 1),
        "rejected_source_seeds": (8, 2, 8),
        "namespace": "langmani-m4-fresh-seeds-v1",
    }
    first = build_fresh_seed_schedule(**kwargs)
    second = build_fresh_seed_schedule(**kwargs)
    assert first == second
    assert len(first.ordered_fresh_seeds) == 30
    assert len(set(first.ordered_fresh_seeds)) == 30
    assert not set(first.ordered_fresh_seeds) & {1, 2, 5, 8, 9}
    assert first.ordered_accepted_source_seeds == (1, 5, 9)
    assert first.ordered_rejected_source_seeds == (2, 8)
    assert first.rejected_source_seeds_available
    assert FreshSeedSchedule.from_dict(json.loads(json.dumps(first.to_dict()))) == first


def test_fresh_seed_schedule_accepts_frozen_run_identity_contract() -> None:
    schedule = build_fresh_seed_schedule(
        m3b_export_fingerprint=_digest("a"),
        accepted_source_seeds=(1, 2),
        rejected_source_seeds=None,
        namespace="langmani-m4-fresh-seeds-v1",
    )
    identity = ActRunIdentity(
        m3b_export_fingerprint=_digest("a"),
        m3b_split_manifest_digest=_digest("b"),
        ordered_train_episode_indices=(0,),
        ordered_validation_episode_indices=(1,),
        variant=ActVariant.MIXED_UNCONDITIONED,
        task_id=None,
        task_onehot_mapping_version=TASK_ONEHOT_MAPPING_VERSION,
        model_config={"policy": "act"},
        data_contract={"evaluation_schedules": {"fresh_seed_schedule": schedule.to_dict()}},
        train_statistics_fingerprint=_digest("c"),
        optimization_config={"optimizer": "adamw"},
        training_seed=0,
        experiment_mode=ExperimentMode.DRY_RUN,
        device="cpu",
        dtype="float32",
        lerobot_version="0.6.0",
        torch_version="2.11.0+cpu",
        cuda_version=None,
        git_commit="1" * 40,
        git_dirty=False,
        dirty_development_override=False,
    )
    schedules = identity.data_contract["evaluation_schedules"]
    assert isinstance(schedules, Mapping)
    stored = schedules["fresh_seed_schedule"]
    assert isinstance(stored, Mapping)
    assert not isinstance(stored, dict)
    assert FreshSeedSchedule.from_dict(stored) == schedule


def test_fresh_seed_schedule_records_unavailable_rejected_candidates() -> None:
    unavailable = build_fresh_seed_schedule(
        m3b_export_fingerprint=_digest("a"),
        accepted_source_seeds=(1, 2),
        rejected_source_seeds=None,
        namespace="langmani-m4-fresh-seeds-v1",
    )
    available_empty = build_fresh_seed_schedule(
        m3b_export_fingerprint=_digest("a"),
        accepted_source_seeds=(1, 2),
        rejected_source_seeds=(),
        namespace="langmani-m4-fresh-seeds-v1",
    )
    assert not unavailable.rejected_source_seeds_available
    assert available_empty.rejected_source_seeds_available
    assert unavailable.exclusion_digest != available_empty.exclusion_digest
    assert unavailable.schedule_digest != available_empty.schedule_digest


def test_fresh_schedule_changes_with_semantic_namespace_or_dataset() -> None:
    base = build_fresh_seed_schedule(
        m3b_export_fingerprint=_digest("a"),
        accepted_source_seeds=(1,),
        rejected_source_seeds=None,
        namespace="langmani-m4-fresh-seeds-v1",
    )
    changed_namespace = build_fresh_seed_schedule(
        m3b_export_fingerprint=_digest("a"),
        accepted_source_seeds=(1,),
        rejected_source_seeds=None,
        namespace="another-predeclared-benchmark",
    )
    changed_dataset = build_fresh_seed_schedule(
        m3b_export_fingerprint=_digest("b"),
        accepted_source_seeds=(1,),
        rejected_source_seeds=None,
        namespace="langmani-m4-fresh-seeds-v1",
    )
    assert (
        len(
            {
                base.schedule_digest,
                changed_namespace.schedule_digest,
                changed_dataset.schedule_digest,
            }
        )
        == 3
    )


def test_wilson_interval_matches_known_binomial_values() -> None:
    middle = wilson_interval(5, 10)
    assert middle.lower == pytest.approx(0.236593, abs=1e-6)
    assert middle.upper == pytest.approx(0.763407, abs=1e-6)
    zero = wilson_interval(0, 10)
    assert zero.lower == pytest.approx(0.0)
    assert zero.upper == pytest.approx(0.277533, abs=1e-6)
    with pytest.raises(EvaluationContractError):
        wilson_interval(1, 0)


def _episode(
    evaluation_id: str,
    *,
    task_id: str,
    status: RolloutStatus,
    schedule: str = "c",
    split: EvaluationSplit = EvaluationSplit.VALIDATION,
) -> RolloutEpisodeResult:
    success = status is RolloutStatus.SUCCESS
    return RolloutEpisodeResult(
        evaluation_id=evaluation_id,
        run_fingerprint=_digest("a"),
        checkpoint_fingerprint=_digest("b"),
        schedule_digest=_digest(schedule),
        split=split,
        scene_seed=1,
        scene_id=f"scene-{evaluation_id}",
        task_id=task_id,
        status=status,
        success=success,
        episode_steps=10,
        final_evaluation={"success": success, "target_off_table": False},
        target_in_target_bin=success,
        target_in_wrong_bin=False,
        wrong_object_in_target_bin=False,
        target_grasped_any=success,
        wrong_object_grasped_any=False,
        target_off_table=status is RolloutStatus.TARGET_OFF_TABLE,
        timeout=status is RolloutStatus.TIMEOUT,
        invalid_action=status is RolloutStatus.INVALID_ACTION,
        inference_failure=status is RolloutStatus.INFERENCE_FAILURE,
        time_to_first_target_grasp_s=0.2 if success else None,
        time_to_release_s=0.4 if success else None,
        time_to_success_s=0.5 if success else None,
        cumulative_action_magnitude=1.0,
        action_min=(0.0,) * 8,
        action_max=(1.0,) * 8,
        inference_latency_ms=(1.0, 2.0),
        environment_step_latency_ms=(3.0, 4.0),
        total_episode_duration_s=0.5,
        runtime_fingerprint=_digest("d"),
        action_bound_mode=ActionBoundMode.REJECT,
        task_success=success,
        strict_unprojected_success=success,
        action_projection_summary=ActionProjectionSummary.from_records(
            (),
            action_dimension=8,
            total_policy_actions=10,
        ).to_dict(),
        failure_reason=None if success else status.value,
    )


def test_rollout_benchmark_summary_reports_task_rates_failures_and_wilson() -> None:
    episodes = (
        _episode("one", task_id="red_left", status=RolloutStatus.SUCCESS),
        _episode("two", task_id="red_left", status=RolloutStatus.TIMEOUT),
        _episode("three", task_id="blue_right", status=RolloutStatus.SUCCESS),
    )
    result = summarize_rollout_benchmark(episodes)
    assert result.success_rate == pytest.approx(2 / 3)
    assert result.success_wilson_low < result.success_rate < result.success_wilson_high
    assert result.per_task_success_rates == {"blue_right": 1.0, "red_left": 0.5}
    assert result.failure_counts == {"timeout": 1}
    assert json.loads(json.dumps(result.to_dict()))["episodes"][0]["evaluation_id"] == "one"


def test_rollout_benchmark_rejects_mixed_schedule_or_duplicate_episode() -> None:
    first = _episode("same", task_id="red_left", status=RolloutStatus.SUCCESS)
    with pytest.raises(EvaluationContractError, match="share run"):
        summarize_rollout_benchmark(
            (
                first,
                _episode("other", task_id="red_left", status=RolloutStatus.SUCCESS, schedule="d"),
            )
        )
    with pytest.raises(EvaluationContractError, match="IDs must be unique"):
        summarize_rollout_benchmark((first, first))


def _policy_inputs(*, conditioned: bool = False) -> tuple[dict[str, np.ndarray], ...]:
    result = []
    for task_index in range(6):
        state = np.zeros(15 if conditioned else 9, dtype=np.float32)
        if conditioned:
            state[9 + task_index] = 1.0
        result.append(
            {
                IMAGE_FEATURE_KEY: np.zeros((3, 8, 8), dtype=np.float32),
                STATE_FEATURE_KEY: state,
            }
        )
    return tuple(result)


def test_unconditioned_counterfactual_requires_exact_identical_hidden_free_inputs() -> None:
    identical = _policy_inputs()
    assert policy_inputs_identical(identical)
    actions = np.zeros((6, 8), dtype=np.float32)
    chunks = np.zeros((6, 4, 8), dtype=np.float32)
    result = compute_counterfactual_sensitivity(
        variant=ActVariant.MIXED_UNCONDITIONED,
        scene_id="scene-0",
        ordered_task_ids=CANONICAL_TASK_IDS,
        first_actions=actions,
        action_chunks=chunks,
        policy_inputs=identical,
    )
    assert result.identical_unconditioned_inputs
    assert len(result.pairwise_first_action_distances) == 15
    assert set(result.pairwise_first_action_distances.values()) == {0.0}
    assert set(result.pairwise_chunk_distances.values()) == {0.0}

    changed = list(identical)
    changed[-1] = dict(changed[-1])
    changed[-1][STATE_FEATURE_KEY] = np.ones(9, dtype=np.float32)
    with pytest.raises(CounterfactualAnalysisError, match="physical policy input fixed"):
        compute_counterfactual_sensitivity(
            variant=ActVariant.MIXED_UNCONDITIONED,
            scene_id="scene-0",
            ordered_task_ids=CANONICAL_TASK_IDS,
            first_actions=actions,
            action_chunks=chunks,
            policy_inputs=tuple(changed),
        )

    hidden = list(identical)
    hidden[0] = {**hidden[0], "task": np.asarray([0])}
    with pytest.raises(CounterfactualAnalysisError, match="only base_camera"):
        policy_inputs_identical(tuple(hidden))

    differing_predictions = actions.copy()
    differing_predictions[-1, 0] = 0.1
    differing_chunks = chunks.copy()
    differing_chunks[-1, :, 0] = 0.1
    with pytest.raises(CounterfactualAnalysisError, match="predictions differ"):
        compute_counterfactual_sensitivity(
            variant=ActVariant.MIXED_UNCONDITIONED,
            scene_id="scene-0",
            ordered_task_ids=CANONICAL_TASK_IDS,
            first_actions=differing_predictions,
            action_chunks=differing_chunks,
            policy_inputs=identical,
        )


def test_conditioned_pairwise_sensitivity_has_stable_distances_and_round_trip() -> None:
    task_ids = CANONICAL_TASK_IDS
    actions = np.stack([np.full(8, float(index), dtype=np.float32) for index in range(6)])
    chunks = np.repeat(actions[:, None, :], 3, axis=1)
    result = compute_counterfactual_sensitivity(
        variant=ActVariant.MIXED_TASK_ONEHOT,
        scene_id="scene-onehot",
        ordered_task_ids=task_ids,
        first_actions=actions,
        action_chunks=chunks,
        policy_inputs=_policy_inputs(conditioned=True),
    )
    assert not result.identical_unconditioned_inputs
    pair = f"{task_ids[0]}|{task_ids[1]}"
    assert result.pairwise_first_action_distances[pair] == pytest.approx(np.sqrt(8.0))
    assert result.pairwise_chunk_distances[pair] == pytest.approx(np.sqrt(8.0))
    restored = counterfactual_sensitivity_from_dict(json.loads(json.dumps(result.to_dict())))
    assert restored == result


def test_conditioned_sensitivity_rejects_changes_beyond_canonical_onehot() -> None:
    task_ids = CANONICAL_TASK_IDS
    actions = np.zeros((6, 8), dtype=np.float32)
    chunks = np.zeros((6, 2, 8), dtype=np.float32)
    changed_qpos = list(_policy_inputs(conditioned=True))
    changed_qpos[-1] = dict(changed_qpos[-1])
    changed_qpos[-1][STATE_FEATURE_KEY] = changed_qpos[-1][STATE_FEATURE_KEY].copy()
    changed_qpos[-1][STATE_FEATURE_KEY][0] = 0.1
    with pytest.raises(CounterfactualAnalysisError, match="PandaPolicyStateV0 fixed"):
        compute_counterfactual_sensitivity(
            variant=ActVariant.MIXED_TASK_ONEHOT,
            scene_id="scene",
            ordered_task_ids=task_ids,
            first_actions=actions,
            action_chunks=chunks,
            policy_inputs=tuple(changed_qpos),
        )

    invalid_suffix = list(_policy_inputs(conditioned=True))
    invalid_suffix[-1] = dict(invalid_suffix[-1])
    invalid_suffix[-1][STATE_FEATURE_KEY] = invalid_suffix[-1][STATE_FEATURE_KEY].copy()
    invalid_suffix[-1][STATE_FEATURE_KEY][-1] = 0.0
    with pytest.raises(CounterfactualAnalysisError, match="canonical one-hot"):
        compute_counterfactual_sensitivity(
            variant=ActVariant.MIXED_TASK_ONEHOT,
            scene_id="scene",
            ordered_task_ids=task_ids,
            first_actions=actions,
            action_chunks=chunks,
            policy_inputs=tuple(invalid_suffix),
        )


def test_counterfactual_analysis_rejects_misaligned_or_nonfinite_chunks() -> None:
    task_ids = CANONICAL_TASK_IDS
    actions = np.zeros((6, 8), dtype=np.float32)
    chunks = np.zeros((6, 2, 8), dtype=np.float32)
    chunks[0, 0, 0] = 1.0
    with pytest.raises(CounterfactualAnalysisError, match="first_actions must equal"):
        compute_counterfactual_sensitivity(
            variant=ActVariant.PER_TASK,
            scene_id="scene",
            ordered_task_ids=task_ids,
            first_actions=actions,
            action_chunks=chunks,
            policy_inputs=_policy_inputs(),
        )
    chunks[0, 0, 0] = 0.0
    chunks[0, 1, 0] = np.nan
    with pytest.raises(CounterfactualAnalysisError, match="finite"):
        compute_counterfactual_sensitivity(
            variant=ActVariant.PER_TASK,
            scene_id="scene",
            ordered_task_ids=task_ids,
            first_actions=actions,
            action_chunks=chunks,
            policy_inputs=_policy_inputs(),
        )


def test_counterfactual_analysis_requires_canonical_tasks_and_expert_comparison() -> None:
    actions = np.stack([np.full(8, float(index), dtype=np.float32) for index in range(6)])
    chunks = np.repeat(actions[:, None, :], 3, axis=1)
    with pytest.raises(CounterfactualAnalysisError, match="canonical six stable"):
        compute_counterfactual_sensitivity(
            variant=ActVariant.PER_TASK,
            scene_id="scene",
            ordered_task_ids=tuple(f"task-{index}" for index in range(6)),
            first_actions=actions,
            action_chunks=chunks,
            policy_inputs=_policy_inputs(),
        )

    demonstrations = tuple(
        CounterfactualDemonstration(
            task_id=task_id,
            rgb_digest="a" * 64,
            panda_state=(0.0,) * 9,
            action_chunk=tuple(tuple(0.0 for _ in range(8)) for _ in range(3)),
            action_is_pad=(False, False, True),
        )
        for task_id in CANONICAL_TASK_IDS
    )
    distances = expert_chunk_distances(chunks, demonstrations)
    assert tuple(distances) == CANONICAL_TASK_IDS
    assert distances[CANONICAL_TASK_IDS[0]] == 0.0
    assert distances[CANONICAL_TASK_IDS[1]] == pytest.approx(np.sqrt(8.0))

    sensitivity = compute_counterfactual_sensitivity(
        variant=ActVariant.PER_TASK,
        scene_id="scene",
        ordered_task_ids=CANONICAL_TASK_IDS,
        first_actions=actions,
        action_chunks=chunks,
        policy_inputs=_policy_inputs(),
    )
    effects = task_identity_effects(sensitivity.pairwise_chunk_distances)
    assert effects["different_target_same_bin_mean_chunk_distance"] > 0
    assert effects["same_target_different_bin_mean_chunk_distance"] > 0
