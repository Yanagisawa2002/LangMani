"""CPU-safe contracts for immutable M4.3a semantic-audit evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import BIN_IDS, OBJECT_IDS, stable_task_id
from langmani.policies.act_semantic_audit import (
    build_semantic_alignment_audit,
    retrieve_bin,
    retrieve_object,
    retrieve_task,
)
from langmani.policies.m42_types import GripperRuntimeMode
from langmani.policies.m43_audit_runtime import _complete_distance_record
from langmani.policies.m43_evidence import (
    M43_CANDIDATE_POLICY_LABELS,
    M43_TASK_TOKEN_M42_STATUS,
    M43AuditConfig,
    M43AuditScope,
    M43AuditScopeArtifact,
    M43EvidenceError,
    M43ObservationSource,
    ObservationSourceIdentity,
    build_policy_semantic_summary,
    build_rollout_aggregate_evidence,
    stage_and_promote_audit_evidence,
    unavailable_rollout_evidence,
    validate_completed_audit_evidence,
)
from langmani.policies.m43_types import (
    M43_CANONICAL_TASK_IDS,
    ActionChunkDistanceConfig,
    BinRetrievalResult,
    ObjectRetrievalResult,
    SemanticAlignmentAudit,
    SemanticFailureClass,
    SemanticRetrievalResult,
)

_DIGEST = "sha256:" + "a" * 64
_OTHER_DIGEST = "sha256:" + "b" * 64
_GIT_COMMIT = "c" * 40


def _distance_config() -> ActionChunkDistanceConfig:
    return ActionChunkDistanceConfig(
        action_lower_bounds=(-1.0,) * 8,
        action_upper_bounds=(1.0,) * 8,
        train_action_std=(0.5,) * 8,
        locked_execution_horizon=5,
    )


def _observations(source: M43ObservationSource) -> tuple[ObservationSourceIdentity, ...]:
    group_count = 6 if source is M43ObservationSource.M3B_VALIDATION else 12
    base_seed = 1_000 if source is M43ObservationSource.M3B_VALIDATION else 2_000
    values: list[ObservationSourceIdentity] = []
    for group_index in range(group_count):
        source_task_id = M43_CANONICAL_TASK_IDS[0]
        episode_id = f"{source.value}-episode-{group_index:03d}"
        image_digest = f"sha256:{group_index + 1:064x}"
        state_digest = f"sha256:{group_index + 101:064x}"
        for task_index, task_id in enumerate(M43_CANONICAL_TASK_IDS):
            values.append(
                ObservationSourceIdentity(
                    source=source,
                    observation_id=f"{source.value}-scene-{group_index:03d}-query-{task_index}",
                    scene_id=f"scene-{group_index:03d}",
                    scene_group_id=f"group-{group_index:03d}",
                    scene_seed=base_seed + group_index,
                    source_task_id=source_task_id,
                    source_episode_id=episode_id,
                    frame_index=0,
                    image_fingerprint=image_digest,
                    policy_state_fingerprint=state_digest,
                    queried_task_id=task_id,
                )
            )
    return tuple(values)


def _config(scope: M43AuditScope = M43AuditScope.M3B_VALIDATION) -> M43AuditConfig:
    observations = _observations(M43ObservationSource.M3B_VALIDATION)
    if scope is M43AuditScope.M42_DEVELOPMENT:
        observations = _observations(M43ObservationSource.M42_DEVELOPMENT)
    elif scope is M43AuditScope.COMBINED:
        observations = (*observations, *_observations(M43ObservationSource.M42_DEVELOPMENT))
    return M43AuditConfig(
        scope=scope,
        git_commit=_GIT_COMMIT,
        m3b_fingerprint=_DIGEST,
        m3b_split_manifest_digest=_OTHER_DIGEST,
        per_task_checkpoint_fingerprints={
            task_id: f"sha256:{index + 10:064x}"
            for index, task_id in enumerate(M43_CANONICAL_TASK_IDS)
        },
        state_onehot_checkpoint_fingerprint="sha256:" + "d" * 64,
        task_token_checkpoint_fingerprint="sha256:" + "e" * 64,
        selected_execution_horizon=5,
        selected_gripper_runtime=GripperRuntimeMode.PROJECT,
        distance_config=_distance_config(),
        ordered_observations=observations,
    )


def _retrievals(
    observations: tuple[ObservationSourceIdentity, ...],
) -> tuple[
    tuple[SemanticRetrievalResult, ...],
    tuple[ObjectRetrievalResult, ...],
    tuple[BinRetrievalResult, ...],
]:
    references = _reference_chunks()
    task_results: list[SemanticRetrievalResult] = []
    object_results: list[ObjectRetrievalResult] = []
    bin_results: list[BinRetrievalResult] = []
    for observation in observations:
        task_id = observation.queried_task_id
        arguments = {
            "observation_id": observation.observation_id,
            "requested_task_id": task_id,
            "candidate_chunk": references[task_id],
            "per_task_chunks": references,
            "config": _distance_config(),
        }
        task_results.append(retrieve_task(**arguments))
        object_results.append(retrieve_object(**arguments))
        bin_results.append(retrieve_bin(**arguments))
    return tuple(task_results), tuple(object_results), tuple(bin_results)


def _reference_chunks() -> dict[str, np.ndarray]:
    chunks: dict[str, np.ndarray] = {}
    for task_spec in CANONICAL_TASK_SPECS:
        value = np.zeros((50, 8), dtype=np.float64)
        value[:, 0] = float(OBJECT_IDS.index(task_spec.target_object_id) * 4)
        value[:, 1] = float(BIN_IDS.index(task_spec.target_bin_id))
        chunks[stable_task_id(task_spec)] = value
    return chunks


def _audit(
    label: str, observations: tuple[ObservationSourceIdentity, ...]
) -> SemanticAlignmentAudit:
    task_results, object_results, bin_results = _retrievals(observations)
    references = _reference_chunks()
    complete_distances = {
        observation.observation_id: _complete_distance_record(
            candidate_chunk=references[observation.queried_task_id],
            requested_task_id=observation.queried_task_id,
            per_task_chunks=references,
            config=_distance_config(),
        )
        for observation in observations
    }
    return build_semantic_alignment_audit(
        policy_label=label,
        distance_config=_distance_config(),
        task_results=task_results,
        object_results=object_results,
        bin_results=bin_results,
        first_interactions=(),
        conclusions=(),
        complete_distance_results=complete_distances,
        raw_policy_action_metrics={"mean_pairwise_condition_distance": 0.25},
        runtime_action_metrics={},
        deterministic_reload_validated=False,
        deterministic_reset_validated=True,
    )


def _scope_artifact(config: M43AuditConfig, scope: M43AuditScope) -> M43AuditScopeArtifact:
    observations = config.observations_for_scope(scope)
    audits = {label: _audit(label, observations) for label in M43_CANDIDATE_POLICY_LABELS}
    summaries = {
        label: build_policy_semantic_summary(audit, task_sensitivity_magnitude=0.25)
        for label, audit in audits.items()
    }
    return M43AuditScopeArtifact(
        scope=scope,
        ordered_observation_ids=tuple(item.observation_id for item in observations),
        policy_audits=audits,
        policy_summaries=summaries,
        first_interaction_evidence_available=False,
        rollout_evidence=unavailable_rollout_evidence(),
    )


def _artifact_set(config: M43AuditConfig) -> dict[M43AuditScope, M43AuditScopeArtifact]:
    return {scope: _scope_artifact(config, scope) for scope in config.required_artifact_scopes}


def test_config_is_portable_balanced_and_round_trips() -> None:
    config = _config()

    assert config.fingerprint.startswith("sha256:")
    assert len(config.ordered_observations) == 36
    assert config.task_token_m42_status == M43_TASK_TOKEN_M42_STATUS
    first_group = config.ordered_observations[:6]
    assert len({item.source_episode_id for item in first_group}) == 1
    assert len({item.image_fingerprint for item in first_group}) == 1
    assert tuple(item.queried_task_id for item in first_group) == M43_CANONICAL_TASK_IDS
    assert M43AuditConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()


def test_config_rejects_paths_forbidden_sources_and_unfixed_observations() -> None:
    with pytest.raises(M43EvidenceError, match="absolute path"):
        replace(
            _observations(M43ObservationSource.M3B_VALIDATION)[0],
            scene_id=r"C:\private\scene",
        )

    with pytest.raises(M43EvidenceError, match="prohibited"):
        replace(
            _observations(M43ObservationSource.M3B_VALIDATION)[0],
            observation_id="m42_final_v0-query",
        )

    observations = list(_observations(M43ObservationSource.M3B_VALIDATION))
    observations[1] = replace(observations[1], policy_state_fingerprint="sha256:" + "f" * 64)
    with pytest.raises(M43EvidenceError, match="hold source RGB"):
        replace(_config(), ordered_observations=tuple(observations))


def test_config_requires_all_six_checkpoint_references_and_explicit_scope_counts() -> None:
    with pytest.raises(M43EvidenceError, match="six references"):
        replace(
            _config(),
            per_task_checkpoint_fingerprints={M43_CANONICAL_TASK_IDS[0]: _DIGEST},
        )
    with pytest.raises(M43EvidenceError, match="36 validation"):
        replace(_config(), ordered_observations=_config().ordered_observations[:-1])

    combined = _config(M43AuditScope.COMBINED)
    assert len(combined.ordered_observations) == 108
    assert combined.required_artifact_scopes == (
        M43AuditScope.M3B_VALIDATION,
        M43AuditScope.M42_DEVELOPMENT,
        M43AuditScope.COMBINED,
    )
    with pytest.raises(M43EvidenceError, match="validation then development"):
        replace(combined, ordered_observations=tuple(reversed(combined.ordered_observations)))


def test_policy_summary_keeps_semantic_interpretations_separate() -> None:
    audit = _audit("state_onehot", _config().ordered_observations)
    summary = build_policy_semantic_summary(audit, task_sensitivity_magnitude=0.25)

    assert summary["full_task_top1_retrieval"] == 1.0
    assert summary["target_object_retrieval"] == 1.0
    assert summary["conditional_bin_retrieval"] == 1.0
    assert summary["first_interaction_evidence_available"] is False
    interpretation = summary["interpretation"]
    assert isinstance(interpretation, dict)
    assert interpretation["output_changes_across_task_conditions"] is True
    assert interpretation["correct_semantic_alignment"] is True
    assert summary["deterministic_reload"] == {
        "validated": False,
        "available": False,
        "reason": "fresh checkpoint instances were not reloaded by this audit",
    }
    assert interpretation["nonzero_action_distance_is_not_semantic_correctness"] is True


def test_real_rollout_aggregate_overrides_empty_trace_distribution_without_fabrication() -> None:
    config = _config(M43AuditScope.M42_DEVELOPMENT)
    observations = config.ordered_observations
    audits = {label: _audit(label, observations) for label in M43_CANDIDATE_POLICY_LABELS}
    distribution = {failure.value: 0 for failure in SemanticFailureClass}
    distribution[SemanticFailureClass.TIMED_OUT_AFTER_TARGET_GRASP.value] = 7
    rollout = build_rollout_aggregate_evidence(
        source_fingerprint=_DIGEST,
        per_policy_post_grasp_failure_distribution={
            label: distribution for label in M43_CANDIDATE_POLICY_LABELS
        },
        per_policy_metrics={label: {"successes": 12} for label in M43_CANDIDATE_POLICY_LABELS},
    )
    summaries = {
        label: build_policy_semantic_summary(
            audit,
            task_sensitivity_magnitude=0.25,
            external_failure_distribution=distribution,
        )
        for label, audit in audits.items()
    }

    artifact = M43AuditScopeArtifact(
        scope=M43AuditScope.M42_DEVELOPMENT,
        ordered_observation_ids=tuple(item.observation_id for item in observations),
        policy_audits=audits,
        policy_summaries=summaries,
        first_interaction_evidence_available=False,
        rollout_evidence=rollout,
    )

    assert artifact.first_interaction_evidence_available is False
    assert artifact.rollout_evidence["available"] is True
    state_summary = artifact.policy_summaries["state_onehot"]
    assert isinstance(state_summary, Mapping)
    assert cast(Mapping[str, object], state_summary)["post_grasp_failure_distribution"] == (
        distribution
    )
    with pytest.raises(M43EvidenceError, match="validation-only"):
        replace(artifact, scope=M43AuditScope.M3B_VALIDATION)


def test_lifecycle_stages_validates_promotes_and_reuses_immutably(tmp_path: Path) -> None:
    config = _config()
    completed = stage_and_promote_audit_evidence(
        output_root=tmp_path / "audit",
        config=config,
        scope_artifacts=_artifact_set(config),
    )

    assert completed.root.name == config.fingerprint.removeprefix("sha256:")
    assert (completed.root / "complete.json").is_file()
    assert (completed.root / "summary.md").is_file()
    assert (completed.root / "scopes" / "m3b_validation.json").is_file()
    assert completed.completion["semantic_audit_completed"] is True
    assert completed.completion["final_schedule_accessed"] is False
    assert completed.completion["smolvla_go"] is False
    human_summary = (completed.root / "summary.md").read_text(encoding="utf-8")
    for expected in (
        "Mean correct-reference margin",
        "Per-TaskSpec retrieval",
        "Task confusion",
        "Object confusion",
        "Bin confusion",
        "Post-grasp distribution",
        "Semantic classification",
        "First-interaction records: unavailable",
        "Deterministic fresh-instance reload validated: false",
    ):
        assert expected in human_summary
    loaded = validate_completed_audit_evidence(completed.root, expected_config=config)
    assert loaded.evidence_fingerprint == completed.evidence_fingerprint

    reused = stage_and_promote_audit_evidence(
        output_root=tmp_path / "audit",
        config=config,
        scope_artifacts=_artifact_set(config),
    )
    assert reused.root == completed.root
    assert not tuple((tmp_path / "audit" / "evidence").glob(".staging-*"))


def test_checksum_and_extra_artifact_tampering_are_rejected(tmp_path: Path) -> None:
    config = _config()
    completed = stage_and_promote_audit_evidence(
        output_root=tmp_path / "audit",
        config=config,
        scope_artifacts=_artifact_set(config),
    )
    summary = completed.root / "summary.md"
    summary.write_text(summary.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    with pytest.raises(M43EvidenceError, match="checksum or size mismatch"):
        validate_completed_audit_evidence(completed.root)

    second = stage_and_promote_audit_evidence(
        output_root=tmp_path / "second",
        config=config,
        scope_artifacts=_artifact_set(config),
    )
    (second.root / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(M43EvidenceError, match="incomplete or contain extras"):
        validate_completed_audit_evidence(second.root)


def test_matching_staging_requires_explicit_cleanup_and_rejects_other_owner(
    tmp_path: Path,
) -> None:
    config = _config()
    root = tmp_path / "audit"
    evidence = root / "evidence"
    staging = evidence / f".staging-{config.fingerprint.removeprefix('sha256:')}"
    staging.mkdir(parents=True)
    (staging / "owner.json").write_text(
        json.dumps({"config_fingerprint": config.fingerprint}), encoding="utf-8"
    )
    with pytest.raises(M43EvidenceError, match="explicit clean_matching_staging"):
        stage_and_promote_audit_evidence(
            output_root=root,
            config=config,
            scope_artifacts=_artifact_set(config),
        )
    completed = stage_and_promote_audit_evidence(
        output_root=root,
        config=config,
        scope_artifacts=_artifact_set(config),
        clean_matching_staging=True,
    )
    assert completed.root.is_dir()

    other_root = tmp_path / "other"
    other_staging = (
        other_root / "evidence" / f".staging-{config.fingerprint.removeprefix('sha256:')}"
    )
    other_staging.mkdir(parents=True)
    (other_staging / "owner.json").write_text(
        json.dumps({"config_fingerprint": _OTHER_DIGEST}), encoding="utf-8"
    )
    with pytest.raises(M43EvidenceError, match="another audit config"):
        stage_and_promote_audit_evidence(
            output_root=other_root,
            config=config,
            scope_artifacts=_artifact_set(config),
            clean_matching_staging=True,
        )


def test_combined_scope_cannot_silently_omit_development_or_combined_results() -> None:
    config = _config(M43AuditScope.COMBINED)
    validation_only = {
        M43AuditScope.M3B_VALIDATION: _scope_artifact(config, M43AuditScope.M3B_VALIDATION)
    }
    with pytest.raises(M43EvidenceError, match="keep validation, development, and combined"):
        stage_and_promote_audit_evidence(
            output_root=Path("unused-output"),
            config=config,
            scope_artifacts=validation_only,
        )


def test_scope_artifact_rejects_malformed_summary_and_access_flags() -> None:
    config = _config()
    artifact = _scope_artifact(config, M43AuditScope.M3B_VALIDATION)
    bad_summaries = {label: dict(summary) for label, summary in artifact.policy_summaries.items()}
    bad_summaries["state_onehot"]["full_task_top1_retrieval"] = 0.0
    with pytest.raises(M43EvidenceError, match="summary disagrees"):
        replace(artifact, policy_summaries=bad_summaries)
    with pytest.raises(M43EvidenceError, match="cannot access or authorize"):
        replace(artifact, final_schedule_accessed=True)


def test_machine_specific_paths_are_rejected_inside_audit_metrics(tmp_path: Path) -> None:
    config = _config()
    observations = config.ordered_observations
    audit = _audit("state_onehot", observations)
    unsafe = replace(audit, runtime_action_metrics={"source": "/root/private/report.json"})
    audits = {
        "state_onehot": unsafe,
        "task_token": _audit("task_token", observations),
    }
    summaries = {
        label: build_policy_semantic_summary(value, task_sensitivity_magnitude=0.25)
        for label, value in audits.items()
    }
    with pytest.raises(M43EvidenceError, match="absolute path"):
        artifact = M43AuditScopeArtifact(
            scope=M43AuditScope.M3B_VALIDATION,
            ordered_observation_ids=tuple(item.observation_id for item in observations),
            policy_audits=audits,
            policy_summaries=summaries,
            first_interaction_evidence_available=False,
            rollout_evidence=unavailable_rollout_evidence(),
        )
        stage_and_promote_audit_evidence(
            output_root=tmp_path,
            config=config,
            scope_artifacts={M43AuditScope.M3B_VALIDATION: artifact},
        )
