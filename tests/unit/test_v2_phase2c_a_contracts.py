"""CPU-safe tests for Phase 2C-A immutable experiment contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from langmani.v2.phase2c_a import (
    EXCLUDED_EPISODES,
    INVALID_TRANSCRIBED_PACKAGE_FINGERPRINT,
    PACKAGE_FINGERPRINT,
    TASK_IDS,
    TRAIN_EPISODES,
    TRAIN_FRAMES,
    ModelKind,
    Phase2CAContractError,
    Phase2CAResult,
    UniformTaskBatchSampler,
    build_evaluation_schedule,
    build_training_view_manifest,
    classify_result,
    primary_optimization_config,
    task_condition_intervention,
    task_onehot,
    validate_consumed_training_samples,
    validate_package_document,
)
from langmani.v2.policy import PolicyContext, PolicyContractError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2b6_v2"


def _read(name: str) -> dict[str, object]:
    value = json.loads((EVIDENCE_ROOT / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_accepted_package_uses_canonical_fingerprint_and_exclusions() -> None:
    result = validate_package_document(_read("accepted_multiskill_dataset_package.json"))
    assert result["passed"] is True
    assert result["package_fingerprint"] == PACKAGE_FINGERPRINT
    assert len(PACKAGE_FINGERPRINT.removeprefix("sha256:")) == 64
    assert len(INVALID_TRANSCRIBED_PACKAGE_FINGERPRINT.removeprefix("sha256:")) == 65
    assert {
        (item["task_id"], item["source_episode_id"]) for item in result["excluded_episodes"]
    } == EXCLUDED_EPISODES


def test_package_verification_rejects_transcribed_fingerprint() -> None:
    package = _read("accepted_multiskill_dataset_package.json")
    package["fingerprint"] = INVALID_TRANSCRIBED_PACKAGE_FINGERPRINT
    with pytest.raises(Phase2CAContractError, match="identity changed"):
        validate_package_document(package)


def test_training_view_lists_exact_accepted_train_episodes() -> None:
    result = build_training_view_manifest(_read("accepted_primary_split_manifest.json"))
    assert result["passed"] is True
    assert result["episode_count"] == 2_099
    assert result["frame_count"] == 177_718
    assert result["episodes_by_task"] == TRAIN_EPISODES
    assert result["frames_by_task"] == TRAIN_FRAMES
    identities = {(record["task_id"], record["source_episode_id"]) for record in result["records"]}
    assert not identities & EXCLUDED_EPISODES


def test_training_view_rejects_excluded_episode_in_train() -> None:
    manifest = _read("accepted_primary_split_manifest.json")
    manifest["assignments"].append(
        {
            "task_id": "StackCube-v1",
            "source_episode_id": 938,
            "primary_split": "train",
            "frame_count": 1,
        }
    )
    with pytest.raises(Phase2CAContractError, match="excluded"):
        build_training_view_manifest(manifest)


def test_uniform_task_sampler_is_exact_over_three_batches() -> None:
    sampler = UniformTaskBatchSampler(
        {task_id: TRAIN_FRAMES[task_id] for task_id in TASK_IDS},
        batch_size=128,
        seed=7,
    )
    audit = sampler.audit(3)
    assert audit["passed"] is True
    assert audit["sample_counts"] == {task_id: 128 for task_id in TASK_IDS}
    batches = iter(sampler)
    first = next(batches)
    second = next(batches)
    replay = iter(
        UniformTaskBatchSampler(
            {task_id: TRAIN_FRAMES[task_id] for task_id in TASK_IDS},
            batch_size=128,
            seed=7,
        )
    )
    assert first == next(replay)
    assert second == next(replay)


def test_shared_budget_preserves_uniform_sampling_and_records_actual_passes() -> None:
    config = primary_optimization_config(ModelKind.SHARED, batch_size=408)
    assert len(set(config.target_samples_by_task.values())) == 1
    assert config.effective_passes_by_task["StackCube-v1"] < 20
    assert config.effective_passes_by_task["PushCube-v1"] > 20
    assert config.training_steps == 8_712
    assert config.target_samples_by_task == {task_id: 1_184_832 for task_id in TASK_IDS}
    assert config.checkpoint_steps[-1] == config.training_steps


def test_task_specific_steps_follow_each_accepted_frame_budget() -> None:
    for kind in (ModelKind.PICK, ModelKind.STACK, ModelKind.PUSH):
        config = primary_optimization_config(kind, batch_size=128)
        task_id = str(kind.task_id)
        assert config.target_samples_by_task == {task_id: TRAIN_FRAMES[task_id] * 20}
        assert config.effective_passes_by_task == {task_id: 20.0}
        assert config.training_steps == math.ceil(TRAIN_FRAMES[task_id] / 128) * 20


def test_shared_primary_config_rejects_nondivisible_batch() -> None:
    with pytest.raises(Phase2CAContractError, match="divisible"):
        primary_optimization_config(ModelKind.SHARED, batch_size=128)


def test_completed_training_requires_exact_consumed_sample_budget() -> None:
    config = primary_optimization_config(ModelKind.PICK, batch_size=408)
    expected = {"PickCube-v1": TRAIN_FRAMES["PickCube-v1"] * 20}
    assert validate_consumed_training_samples(config, expected)["PickCube-v1"] == 1_092_160
    with pytest.raises(Phase2CAContractError, match="sample budget mismatch"):
        validate_consumed_training_samples(config, {"PickCube-v1": 1_091_000})


def test_three_way_task_id_is_explicit_and_state_independent() -> None:
    assert task_onehot("PickCube-v1") == (1.0, 0.0, 0.0)
    assert task_onehot("StackCube-v1") == (0.0, 1.0, 0.0)
    assert task_onehot("PushCube-v1") == (0.0, 0.0, 1.0)
    with pytest.raises(Phase2CAContractError):
        task_onehot("PickCube-v2")


def test_direct_policy_context_does_not_fabricate_evaluation_task() -> None:
    context = PolicyContext(
        evaluation_task=None,
        evaluation_id="phase2c-a:validation:pickcube:000",
        task_id="PickCube-v1",
    )
    assert context.canonical_task_id == "PickCube-v1"
    with pytest.raises(PolicyContractError):
        PolicyContext(evaluation_task=None, evaluation_id="bad")


def test_evaluation_schedule_is_frozen_before_results() -> None:
    result = build_evaluation_schedule(
        _read("accepted_primary_split_manifest.json"),
        _read("source_inventory.json"),
    )
    assert result["passed"] is True
    assert result["final_settings_mutable_after_results"] is False
    assert result["execution_horizons"] == [1, 4, 8]
    for task_id in TASK_IDS:
        assert len(result["schedules"]["validation"][task_id]) == 30
        assert len(result["schedules"]["test_unseen_reset"][task_id]) == 50
        assert len(result["schedules"]["test_visual_shift"][task_id]) == 50
        assert all(
            item["visual_transform"] == "langmani-v2-phase2b6-postrender-appearance-v0"
            for item in result["schedules"]["test_visual_shift"][task_id]
        )


def test_task_condition_intervention_has_no_language_claim() -> None:
    result = task_condition_intervention(TASK_IDS * 2, seed=3)
    assert result["natural_language_grounding_claim"] is False
    assert all(
        record["incorrect_task_id"] != record["environment_task_id"] for record in result["records"]
    )


@pytest.mark.parametrize(
    ("per_task", "shared", "pipeline", "expected"),
    [
        ((1, 1, 1), (1, 1, 1), True, Phase2CAResult.RESULT_A),
        ((1, 1, 1), (1, 0, 1), True, Phase2CAResult.RESULT_B),
        ((1, 0, 1), (1, 1, 1), True, Phase2CAResult.RESULT_C),
        ((1, 1, 1), (1, 1, 1), False, Phase2CAResult.RESULT_D),
    ],
)
def test_result_classification_and_training_authorization_boundary(
    per_task: tuple[int, int, int],
    shared: tuple[int, int, int],
    pipeline: bool,
    expected: Phase2CAResult,
) -> None:
    result = classify_result(
        pipeline_valid=pipeline,
        training_runs_complete=True,
        checkpoints_recoverable=True,
        padding_mask_verified=True,
        closed_loop_complete=True,
        per_task_success_count=dict(zip(TASK_IDS, per_task, strict=True)),
        shared_success_count=dict(zip(TASK_IDS, shared, strict=True)),
        invalid_action_episode_count=0,
    )
    assert result["result"] == expected.value
    assert result["smolvla_training_authorized"] is False
    assert result["vla_jepa_training_authorized"] is False


def test_result_d_when_any_policy_action_is_invalid() -> None:
    counts = {task_id: 1 for task_id in TASK_IDS}
    result = classify_result(
        pipeline_valid=True,
        training_runs_complete=True,
        checkpoints_recoverable=True,
        padding_mask_verified=True,
        closed_loop_complete=True,
        per_task_success_count=counts,
        shared_success_count=counts,
        invalid_action_episode_count=1,
    )
    assert result["result"] == Phase2CAResult.RESULT_D.value


def test_sampler_resume_matches_uninterrupted_batches() -> None:
    lengths = {task_id: 31 + index for index, task_id in enumerate(TASK_IDS)}
    uninterrupted = iter(UniformTaskBatchSampler(lengths, batch_size=11, seed=4))
    expected = [next(uninterrupted) for _ in range(8)]
    resumed = iter(UniformTaskBatchSampler(lengths, batch_size=11, seed=4, start_batch=5))
    assert [next(resumed) for _ in range(3)] == expected[5:]


def test_policy_context_structured_path_still_works_with_mock() -> None:
    class Instance:
        canonical_task_id = "legacy-task"

    class Evaluation:
        task_instance = Instance()

    context = PolicyContext(
        evaluation_task=Evaluation(),  # type: ignore[arg-type]
        evaluation_id="legacy",
    )
    assert context.canonical_task_id == "legacy-task"
    with pytest.raises(PolicyContractError):
        PolicyContext(
            evaluation_task=Evaluation(),  # type: ignore[arg-type]
            evaluation_id="ambiguous",
            task_id="PickCube-v1",
        )


def test_uniform_sampler_indices_stay_within_concat_dataset() -> None:
    lengths = {task_id: 5 for task_id in TASK_IDS}
    batch = next(iter(UniformTaskBatchSampler(lengths, batch_size=12, seed=0)))
    assert len(batch) == 12
    assert all(0 <= index < 15 for index in batch)
    assert isinstance(torch.tensor(batch), torch.Tensor)
