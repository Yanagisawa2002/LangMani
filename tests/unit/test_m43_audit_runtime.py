from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest
import torch

from langmani.policies.act_factor_film_types import (
    FACTOR_FILM_BIN_INDEX_KEY,
    FACTOR_FILM_OBJECT_INDEX_KEY,
)
from langmani.policies.act_task_token import TASK_TOKEN_FEATURE_KEY
from langmani.policies.m42_evaluation import (
    M42CheckpointContext,
    M42CheckpointDescriptor,
    M42PolicyKind,
)
from langmani.policies.m43_audit_runtime import (
    FixedPolicyObservation,
    M43AuditRuntimeError,
    SemanticPolicyContexts,
    _candidate_audit,
    audit_fixed_observations,
    combine_fixed_audit_computations,
    compute_fixed_observation_audits,
    compute_named_fixed_observation_audits,
    compute_per_task_reference_sensitivity,
    predict_postprocessed_action_chunk,
    validate_fixed_observation_set,
)
from langmani.policies.m43_types import M43_CANONICAL_TASK_IDS, ActionChunkDistanceConfig

_DIGEST = "sha256:" + "0" * 64


class _Resettable:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


class _Preprocessor(_Resettable):
    def __call__(self, value: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: tensor.unsqueeze(0) for key, tensor in value.items()}


class _Postprocessor(_Resettable):
    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        return value


class _Policy(_Resettable):
    def __init__(self, *, kind: M42PolicyKind, task_index: int | None = None) -> None:
        super().__init__()
        self.kind = kind
        self.task_index = task_index

    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.kind is M42PolicyKind.PER_TASK:
            assert self.task_index is not None
            index = self.task_index
        elif self.kind is M42PolicyKind.STATE_ONEHOT:
            index = int(torch.argmax(batch["observation.state"][0, 9:]).item())
        elif self.kind is M42PolicyKind.TASK_TOKEN:
            index = int(torch.argmax(batch[TASK_TOKEN_FEATURE_KEY][0]).item())
        else:
            object_index = int(batch[FACTOR_FILM_OBJECT_INDEX_KEY][0].item())
            bin_index = int(batch[FACTOR_FILM_BIN_INDEX_KEY][0].item())
            index = object_index * 2 + bin_index
        object_index, bin_index = divmod(index, 2)
        chunk = torch.zeros((1, 50, 8), dtype=torch.float32)
        chunk[:, :, 0] = float(object_index)
        chunk[:, :, 1] = float(bin_index)
        return chunk


@dataclass
class _Loaded:
    policy: _Policy
    preprocessor: _Preprocessor
    postprocessor: _Postprocessor


def _context(kind: M42PolicyKind, *, task_index: int | None = None) -> M42CheckpointContext:
    task_id = None if task_index is None else M43_CANONICAL_TASK_IDS[task_index]
    descriptor = M42CheckpointDescriptor(
        policy_kind=kind,
        run_fingerprint=_DIGEST,
        checkpoint_fingerprint=_DIGEST,
        checkpoint_relative_path="checkpoints/step-000100000",
        dataset_fingerprint=_DIGEST,
        split_digest=_DIGEST,
        statistics_fingerprint=_DIGEST,
        architecture_fingerprint=(
            _DIGEST if kind in {M42PolicyKind.TASK_TOKEN, M42PolicyKind.FACTOR_FILM} else None
        ),
        task_id=task_id,
        git_commit="0" * 40,
    )
    loaded = _Loaded(
        policy=_Policy(kind=kind, task_index=task_index),
        preprocessor=_Preprocessor(),
        postprocessor=_Postprocessor(),
    )
    return M42CheckpointContext(descriptor=descriptor, loaded=loaded)  # type: ignore[arg-type]


def _policies() -> SemanticPolicyContexts:
    per_task = {
        task_id: _context(M42PolicyKind.PER_TASK, task_index=index)
        for index, task_id in enumerate(M43_CANONICAL_TASK_IDS)
    }
    return SemanticPolicyContexts(
        per_task=per_task,
        state_onehot=_context(M42PolicyKind.STATE_ONEHOT),
        task_token=_context(M42PolicyKind.TASK_TOKEN),
    )


def _observations(*, scope: str = "m3b_validation") -> tuple[FixedPolicyObservation, ...]:
    image = torch.full((3, 256, 256), 0.25, dtype=torch.float32)
    state = torch.arange(9, dtype=torch.float32)
    return tuple(
        FixedPolicyObservation(
            source_scope=scope,
            scene_key="scene-000",
            observation_id=f"{scope}/scene-000/{task_id}",
            requested_task_id=task_id,
            source_identity={
                "source_scope": scope,
                "scene_key": "scene-000",
                "representative_episode_index": 12,
            },
            image=image.clone(),
            state=state.clone(),
        )
        for task_id in M43_CANONICAL_TASK_IDS
    )


def _distance_config() -> ActionChunkDistanceConfig:
    return ActionChunkDistanceConfig(
        action_lower_bounds=(-4.0,) * 8,
        action_upper_bounds=(4.0,) * 8,
        train_action_std=(1.0,) * 8,
        locked_execution_horizon=10,
    )


def test_fixed_observation_set_holds_rgb_and_state_exactly_constant() -> None:
    groups = validate_fixed_observation_set(_observations())
    assert len(groups) == 1
    assert tuple(value.requested_task_id for value in groups[0]) == M43_CANONICAL_TASK_IDS
    assert len({value.image_digest for value in groups[0]}) == 1
    assert len({value.state_digest for value in groups[0]}) == 1


def test_fixed_observation_set_rejects_hidden_physical_or_scope_changes() -> None:
    values = list(_observations())
    changed = values[-1]
    values[-1] = FixedPolicyObservation(
        source_scope=changed.source_scope,
        scene_key=changed.scene_key,
        observation_id=changed.observation_id,
        requested_task_id=changed.requested_task_id,
        source_identity=changed.source_identity,
        image=changed.image + 0.01,
        state=changed.state,
    )
    with pytest.raises(M43AuditRuntimeError, match="remain fixed"):
        validate_fixed_observation_set(values)

    mixed = (*_observations(), *_observations(scope="m42_dev_v0"))
    with pytest.raises(M43AuditRuntimeError, match="cannot mix silently"):
        validate_fixed_observation_set(mixed)


def test_predict_chunk_resets_all_components_and_keeps_postprocessor_boundary() -> None:
    context = _context(M42PolicyKind.STATE_ONEHOT)
    chunk = predict_postprocessed_action_chunk(
        context=context,
        observation=_observations()[3],
        task_id=M43_CANONICAL_TASK_IDS[3],
    )
    assert chunk.shape == (50, 8)
    assert chunk[0, :2].tolist() == [1.0, 1.0]
    loaded = context.loaded
    assert loaded.policy.reset_count == 1
    assert loaded.preprocessor.reset_count == 1
    assert loaded.postprocessor.reset_count == 1


def test_complete_fixed_input_audit_retrieves_both_candidates_semantically() -> None:
    result = audit_fixed_observations(
        observations=_observations(), policies=_policies(), distance_config=_distance_config()
    )
    assert result["source_scope"] == "m3b_validation"
    assert result["scene_count"] == 1
    assert result["observation_count"] == 6
    assert result["fixed_input_validated"] is True
    assert result["deterministic_reset_validated"] is True
    assert result["primary_gripper_excluded"] is True
    assert result["locked_execution_horizon"] == 10
    assert result["test_split_accessed"] is False
    assert result["final_schedule_accessed"] is False
    summaries = result["summaries"]
    assert isinstance(summaries, dict)
    for label in ("state_onehot", "task_token"):
        summary = summaries[label]
        assert summary["task_retrieval"]["top1_accuracy"] == 1.0
        assert summary["task_retrieval"]["top2_accuracy"] == 1.0
        assert summary["task_retrieval"]["mean_reciprocal_rank"] == 1.0
        assert summary["object_retrieval"]["accuracy"] == 1.0
        assert summary["bin_retrieval"]["accuracy"] == 1.0
        assert summary["bin_retrieval"]["conditional_accuracy"] == 1.0
        assert summary["semantic_interpretation"]["nonzero_distance_is_not_semantic_correctness"]
        assert summary["semantic_interpretation"]["correct_semantic_alignment"] is True
        assert summary["deterministic_reload"]["validated"] is False
        assert summary["first_interaction"]["available"] is False
        audit = result["audits"][label]
        assert audit["deterministic_reload_validated"] is False
        distances = audit["complete_distance_results"]
        assert set(distances) == {value.observation_id for value in _observations()}
        first = distances[_observations()[0].observation_id]["task_references"]
        assert set(first) == set(M43_CANONICAL_TASK_IDS)
        assert set(first[M43_CANONICAL_TASK_IDS[0]]) >= {
            "raw_l2",
            "action_range_normalized_l2",
            "train_std_normalized_l2",
            "cosine_distance",
            "arm_only_raw_l2",
            "gripper_only_raw_l2",
        }
        complete = first[M43_CANONICAL_TASK_IDS[0]]
        for family in (
            "raw_l2",
            "action_range_normalized_l2",
            "train_std_normalized_l2",
            "cosine_distance",
            "arm_only_raw_l2",
            "arm_only_action_range_normalized_l2",
            "arm_only_train_std_normalized_l2",
            "arm_only_cosine_distance",
            "gripper_only_raw_l2",
            "gripper_only_action_range_normalized_l2",
            "gripper_only_train_std_normalized_l2",
            "gripper_only_cosine_distance",
        ):
            assert set(complete[family]) == {
                "first_action",
                "first_5_actions",
                "execution_window",
                "full_chunk",
            }


def test_semantic_correctness_requires_exact_consistent_retrieval() -> None:
    computation = compute_fixed_observation_audits(
        observations=_observations(), policies=_policies(), distance_config=_distance_config()
    )
    source = computation.policy_audits["state_onehot"]
    first = source.task_retrieval[0]
    wrong_rank = (first.ranked_task_ids[1], first.ranked_task_ids[0], *first.ranked_task_ids[2:])
    task_results = (
        replace(
            first,
            nearest_task_id=wrong_rank[0],
            ranked_task_ids=wrong_rank,
            correct_rank=2,
            top1_correct=False,
            top2_correct=True,
            reciprocal_rank=0.5,
        ),
        *source.task_retrieval[1:],
    )
    chunks = {
        task_id: np.full((50, 8), index, dtype=np.float64)
        for index, task_id in enumerate(M43_CANONICAL_TASK_IDS)
    }
    _, summary = _candidate_audit(
        policy_label="state_onehot",
        task_results=task_results,
        object_results=source.object_retrieval,
        bin_results=source.bin_retrieval,
        chunks_by_scene=(chunks,),
        complete_distance_results=source.complete_distance_results,
        distance_config=_distance_config(),
        deterministic_reset_validated=True,
    )
    assert summary["task_retrieval"]["top1_accuracy"] > 0.5
    assert summary["semantic_interpretation"]["correct_semantic_alignment"] is False
    assert summary["semantic_interpretation"]["full_task_retrieval_consistent"] is False


def test_named_candidate_audit_preserves_legacy_m43a_output_exactly() -> None:
    policies = _policies()
    legacy = compute_fixed_observation_audits(
        observations=_observations(), policies=policies, distance_config=_distance_config()
    )
    generic = compute_named_fixed_observation_audits(
        observations=_observations(),
        per_task=policies.per_task,
        candidates={
            "state_onehot": policies.state_onehot,
            "task_token": policies.task_token,
        },
        distance_config=_distance_config(),
    )

    assert generic.to_dict() == legacy.to_dict()


def test_named_candidate_audit_supports_factor_film_conditions() -> None:
    policies = _policies()
    factor = _context(M42PolicyKind.FACTOR_FILM)
    result = compute_named_fixed_observation_audits(
        observations=_observations(),
        per_task=policies.per_task,
        candidates={"factor_film": factor},
        distance_config=_distance_config(),
    )

    assert tuple(result.policy_audits) == ("factor_film",)
    assert result.policy_runtime_summaries["factor_film"]["task_retrieval"]["top1_accuracy"] == 1.0


def test_per_task_reference_sensitivity_uses_the_same_fixed_inputs() -> None:
    result = compute_per_task_reference_sensitivity(
        observations=_observations(), per_task=_policies().per_task
    )

    assert result["scene_count"] == 1
    assert result["pair_count"] == 15
    assert result["mean_full_chunk_rms"] > 0.0


def test_policy_contexts_require_all_references_in_canonical_order() -> None:
    contexts = _policies()
    reversed_mapping = dict(reversed(tuple(contexts.per_task.items())))
    with pytest.raises(M43AuditRuntimeError, match="canonical task order"):
        SemanticPolicyContexts(
            per_task=reversed_mapping,
            state_onehot=contexts.state_onehot,
            task_token=contexts.task_token,
        )


def test_combined_audit_keeps_validation_and_development_explicit() -> None:
    validation = compute_fixed_observation_audits(
        observations=_observations(), policies=_policies(), distance_config=_distance_config()
    )
    development = compute_fixed_observation_audits(
        observations=_observations(scope="m42_dev_v0"),
        policies=_policies(),
        distance_config=_distance_config(),
    )
    combined = combine_fixed_audit_computations(validation, development)
    assert combined.source_scope == "combined"
    assert combined.scene_count == 2
    assert combined.ordered_observation_ids == (
        *validation.ordered_observation_ids,
        *development.ordered_observation_ids,
    )
    for label in ("state_onehot", "task_token"):
        assert combined.policy_audits[label].observation_count == 12
        summary = combined.policy_runtime_summaries[label]
        assert summary["task_retrieval"]["top1_accuracy"] == 1.0
        assert summary["task_sensitivity"]["scene_count"] == 2


def test_combined_audit_rejects_wrong_or_overlapping_scopes() -> None:
    validation = compute_fixed_observation_audits(
        observations=_observations(), policies=_policies(), distance_config=_distance_config()
    )
    with pytest.raises(M43AuditRuntimeError, match="development evidence second"):
        combine_fixed_audit_computations(validation, validation)
