from __future__ import annotations

import json
from dataclasses import replace

import pytest

from langmani.policies.act_conditioning import CANONICAL_TASK_IDS
from langmani.policies.m42_types import (
    ExecutionHorizonConfig,
    M42ContractError,
    M42SelectionKind,
    M42SelectionRecord,
    RuntimeAblationResult,
    TaskTokenConfig,
    TaskTokenMapping,
)

DIGEST = "sha256:" + "a" * 64


def test_horizon_and_task_token_contracts_are_stable_json() -> None:
    horizon = ExecutionHorizonConfig(5)
    token = TaskTokenConfig()
    assert horizon.to_dict()["actions_per_query"] == 5
    assert len(horizon.fingerprint) == 71
    assert token.mapping.ordered_task_ids == CANONICAL_TASK_IDS
    assert token.panda_state_dimension == 9
    assert token.task_feature_key == "observation.environment_state"
    json.dumps(token.to_dict(), allow_nan=False)

    with pytest.raises(M42ContractError, match="1, 5, or 10"):
        ExecutionHorizonConfig(2)
    with pytest.raises(M42ContractError, match="ordering"):
        TaskTokenMapping(ordered_task_ids=tuple(reversed(CANONICAL_TASK_IDS)))
    with pytest.raises(M42ContractError, match="9D"):
        TaskTokenConfig(panda_state_dimension=15)


def test_selection_records_quarantine_final_and_development_from_checkpoint_selection() -> None:
    runtime = M42SelectionRecord(
        selection_kind=M42SelectionKind.EXECUTION_HORIZON,
        selected_value="5",
        candidate_values=("5", "10", "1"),
        ranking_version="fixture",
        evidence_fingerprints=(DIGEST,),
        development_schedule_fingerprint=DIGEST,
        locked_at_utc="2026-07-15T00:00:00Z",
    )
    assert runtime.final_schedule_accessed is False
    with pytest.raises(M42ContractError, match="never access"):
        M42SelectionRecord(
            selection_kind=M42SelectionKind.EXECUTION_HORIZON,
            selected_value="5",
            candidate_values=("5",),
            ranking_version="fixture",
            evidence_fingerprints=(DIGEST,),
            development_schedule_fingerprint=DIGEST,
            final_schedule_accessed=True,
            locked_at_utc="2026-07-15T00:00:00Z",
        )
    with pytest.raises(M42ContractError, match="cannot use m42_dev"):
        M42SelectionRecord(
            selection_kind=M42SelectionKind.TASK_TOKEN_CHECKPOINT,
            selected_value=DIGEST,
            candidate_values=(DIGEST,),
            ranking_version="fixture",
            evidence_fingerprints=(DIGEST,),
            development_schedule_fingerprint=DIGEST,
            selected_checkpoint_fingerprint=DIGEST,
            validation_split_digest=DIGEST,
            locked_at_utc="2026-07-15T00:00:00Z",
        )


def test_nested_result_metrics_are_recursively_immutable_and_serializable() -> None:
    result = RuntimeAblationResult(
        config_fingerprint=DIGEST,
        schedule_id="m42_dev_v0",
        schedule_fingerprint=DIGEST,
        model_label="mixed_task_onehot",
        task_id=None,
        execution_horizon=10,
        gripper_mode="project",
        episode_count=1,
        successes=1,
        post_grasp_timeouts=0,
        wrong_object_interactions=0,
        wrong_object_grasp_count=0,
        wrong_object_in_target_bin_count=0,
        target_in_wrong_bin_count=0,
        target_off_table_count=0,
        invalid_action_count=0,
        successful_episode_steps=(12,),
        policy_query_count=2,
        release_sign_transitions=0,
        grasp_sign_transitions=1,
        unnecessary_gripper_sign_transitions=0,
        action_metrics={"nested": {"count": 1}},
    )
    with pytest.raises(TypeError):
        result.action_metrics["new"] = 2  # type: ignore[index]
    nested = result.action_metrics["nested"]
    assert hasattr(nested, "__getitem__")
    with pytest.raises(TypeError):
        nested["count"] = 2  # type: ignore[index]
    json.dumps(result.to_dict(), allow_nan=False)

    with pytest.raises(M42ContractError, match="episode-level union"):
        replace(result, wrong_object_grasp_count=1)
