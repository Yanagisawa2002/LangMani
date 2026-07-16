"""CPU-safe tests for immutable M4.3b FactorFiLM evaluation evidence."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from langmani.policies.act_factor_film_evaluation import FactorFiLMValidationCountResult
from langmani.policies.act_factor_film_evidence import (
    FACTOR_FILM_EVALUATION_STAGE_ORDER,
    FactorFiLMEvaluationEvidenceConfig,
    FactorFiLMEvaluationStage,
    FactorFiLMEvaluationStageArtifact,
    FactorFiLMEvidenceError,
    stage_and_promote_factor_film_evaluation_evidence,
    validate_completed_factor_film_evaluation_evidence,
)
from langmani.policies.act_factor_film_types import (
    FactorFiLMSelectionRecord,
    FactorFiLMValidationQueue,
    FactorFiLMValidationQueueItem,
    canonical_fingerprint,
)


def _digest(label: str) -> str:
    return canonical_fingerprint({"fixture": label})


def _queue() -> FactorFiLMValidationQueue:
    return FactorFiLMValidationQueue(
        run_fingerprint=_digest("run"),
        validation_schedule_digest=_digest("validation-schedule"),
        items=tuple(
            FactorFiLMValidationQueueItem(
                checkpoint_fingerprint=_digest(f"checkpoint-{step}"),
                checkpoint_step=step,
                checkpoint_relative_path=f"checkpoints/step-{step:08d}",
                offline_validation_action_loss=float(21 - index) / 100.0,
            )
            for index, step in enumerate(range(5_000, 100_001, 5_000), start=1)
        ),
    )


def _config(queue: FactorFiLMValidationQueue) -> FactorFiLMEvaluationEvidenceConfig:
    return FactorFiLMEvaluationEvidenceConfig(
        run_fingerprint=queue.run_fingerprint,
        training_manifest_fingerprint=_digest("training-manifest"),
        training_completion_fingerprint=_digest("training-completion"),
        validation_queue_fingerprint=queue.queue_fingerprint,
        dataset_fingerprint=_digest("m3b"),
        split_fingerprint=_digest("split"),
        semantic_audit_evidence_fingerprint=_digest("semantic-audit"),
        validation_schedule_digest=queue.validation_schedule_digest,
        development_schedule_fingerprint=_digest("m42-dev"),
        training_git_commit="8ee0f1babf36b91d1ee2a39701e4a6db6003b660",
        evaluation_git_commit="a" * 40,
    )


def _results(queue: FactorFiLMValidationQueue) -> tuple[FactorFiLMValidationCountResult, ...]:
    return tuple(
        FactorFiLMValidationCountResult(
            checkpoint_fingerprint=item.checkpoint_fingerprint,
            checkpoint_step=item.checkpoint_step,
            schedule_digest=queue.validation_schedule_digest,
            success_count=(18 if index != 7 else 27),
            wrong_object_interaction_count=0,
            wrong_object_in_target_bin_count=0,
            target_off_table_count=0,
            timeout_count=0,
            offline_validation_action_loss=item.offline_validation_action_loss,
        )
        for index, item in enumerate(queue.items)
    )


def test_evidence_config_fingerprint_binds_training_and_evaluation_git() -> None:
    config = _config(_queue())
    assert replace(config, training_git_commit="b" * 40).fingerprint != config.fingerprint
    assert replace(config, evaluation_git_commit="c" * 40).fingerprint != config.fingerprint


def _selection(
    queue: FactorFiLMValidationQueue,
    results: tuple[FactorFiLMValidationCountResult, ...],
) -> FactorFiLMSelectionRecord:
    ranked = tuple(sorted(results, key=lambda value: value.ranking_key))
    return FactorFiLMSelectionRecord(
        run_fingerprint=queue.run_fingerprint,
        validation_queue_fingerprint=queue.queue_fingerprint,
        validation_queue=queue,
        validation_schedule_digest=queue.validation_schedule_digest,
        candidates=tuple(value.to_selection_result() for value in results),
        ranked_checkpoint_fingerprints=tuple(value.checkpoint_fingerprint for value in ranked),
        selected_checkpoint_fingerprint=ranked[0].checkpoint_fingerprint,
        selected_checkpoint_step=ranked[0].checkpoint_step,
    )


def _stage_artifacts(
    config: FactorFiLMEvaluationEvidenceConfig,
    queue: FactorFiLMValidationQueue,
) -> dict[FactorFiLMEvaluationStage, FactorFiLMEvaluationStageArtifact]:
    results = _results(queue)
    selection = _selection(queue, results)
    selected = selection.selected_checkpoint_fingerprint
    step = selection.selected_checkpoint_step
    validation = FactorFiLMEvaluationStageArtifact(
        stage=FactorFiLMEvaluationStage.VALIDATION,
        run_fingerprint=config.run_fingerprint,
        input_artifact_fingerprints={
            "validation_queue": config.validation_queue_fingerprint,
            "validation_schedule": config.validation_schedule_digest,
        },
        payload={
            "schema_version": "langmani-m43-factor-film-validation-evidence-v0",
            "validation_queue_fingerprint": config.validation_queue_fingerprint,
            "validation_schedule_digest": config.validation_schedule_digest,
            "checkpoint_count": 20,
            "results": [value.to_dict() for value in results],
            "benchmark_reports": [
                {"checkpoint_step": value.checkpoint_step, "episode_count": 36} for value in results
            ],
        },
    )
    selection_artifact = FactorFiLMEvaluationStageArtifact(
        stage=FactorFiLMEvaluationStage.SELECTION,
        run_fingerprint=config.run_fingerprint,
        input_artifact_fingerprints={"validation": validation.artifact_fingerprint},
        payload=selection.to_dict(),
        selected_checkpoint_fingerprint=selected,
        selected_checkpoint_step=step,
    )
    reload_artifact = FactorFiLMEvaluationStageArtifact(
        stage=FactorFiLMEvaluationStage.RELOAD,
        run_fingerprint=config.run_fingerprint,
        input_artifact_fingerprints={
            "selection": selection_artifact.artifact_fingerprint,
        },
        payload={
            "reload_validated": True,
            "selected_checkpoint_fingerprint": selected,
            "selected_checkpoint_step": step,
            "atol": 1e-6,
            "rtol": 1e-6,
        },
        selected_checkpoint_fingerprint=selected,
        selected_checkpoint_step=step,
    )
    development = FactorFiLMEvaluationStageArtifact(
        stage=FactorFiLMEvaluationStage.DEVELOPMENT,
        run_fingerprint=config.run_fingerprint,
        input_artifact_fingerprints={
            "reload": reload_artifact.artifact_fingerprint,
            "development_schedule": config.development_schedule_fingerprint,
        },
        payload={
            "development_benchmark_completed": True,
            "physical_execution": True,
            "device": "cuda",
            "total_episode_count": 216,
            "paired_episode_identities_validated": True,
            "benchmark_reports": {
                "per_task": [{"episode_count": 12} for _ in range(6)],
                "state_onehot": {"episode_count": 72},
                "factor_film": {"episode_count": 72},
            },
        },
        selected_checkpoint_fingerprint=selected,
        selected_checkpoint_step=step,
    )
    semantic = FactorFiLMEvaluationStageArtifact(
        stage=FactorFiLMEvaluationStage.SEMANTIC,
        run_fingerprint=config.run_fingerprint,
        input_artifact_fingerprints={
            "validation": validation.artifact_fingerprint,
            "development": development.artifact_fingerprint,
            "semantic_audit": config.semantic_audit_evidence_fingerprint,
        },
        payload={
            "semantic_evaluation_completed": True,
            "post_grasp_analysis_completed": True,
            "first_interaction_analysis_completed": True,
            "correct_task_retrieval_validated": True,
            "object_retrieval_validated": True,
            "bin_retrieval_validated": True,
            "raw_action_metrics_validated": True,
            "runtime_action_metrics_validated": True,
        },
        selected_checkpoint_fingerprint=selected,
        selected_checkpoint_step=step,
    )
    gate = FactorFiLMEvaluationStageArtifact(
        stage=FactorFiLMEvaluationStage.QUALITY_GATE,
        run_fingerprint=config.run_fingerprint,
        input_artifact_fingerprints={
            "development": development.artifact_fingerprint,
            "semantic": semantic.artifact_fingerprint,
        },
        payload={
            "development_quality_gate_passed": False,
            "final_benchmark_authorized": False,
            "final_schedule_accessed": False,
            "smolvla_go": False,
        },
        selected_checkpoint_fingerprint=selected,
        selected_checkpoint_step=step,
    )
    return {
        FactorFiLMEvaluationStage.VALIDATION: validation,
        FactorFiLMEvaluationStage.SELECTION: selection_artifact,
        FactorFiLMEvaluationStage.RELOAD: reload_artifact,
        FactorFiLMEvaluationStage.DEVELOPMENT: development,
        FactorFiLMEvaluationStage.SEMANTIC: semantic,
        FactorFiLMEvaluationStage.QUALITY_GATE: gate,
    }


def _publish(tmp_path: Path):
    queue = _queue()
    config = _config(queue)
    artifacts = _stage_artifacts(config, queue)
    training = tmp_path / "models" / "run"
    training.mkdir(parents=True)
    (training / "sentinel.bin").write_bytes(b"immutable training data")
    completed = stage_and_promote_factor_film_evaluation_evidence(
        output_root=tmp_path / "diagnostics",
        training_run_root=training,
        config=config,
        stage_artifacts=artifacts,
    )
    return completed, config, artifacts, training


def test_config_is_canonical_immutable_and_rejects_sealed_sources() -> None:
    queue = _queue()
    config = _config(queue)
    assert FactorFiLMEvaluationEvidenceConfig.from_dict(config.to_dict()) == config
    assert config.fingerprint == canonical_fingerprint(config.to_dict())
    with pytest.raises(FactorFiLMEvidenceError, match="sealed source"):
        replace(config, test_split_accessed=True)
    with pytest.raises(FactorFiLMEvidenceError, match="H=10/project"):
        replace(config, selected_execution_horizon=20)


def test_publish_checksums_all_six_stages_and_never_writes_training_run(
    tmp_path: Path,
) -> None:
    completed, config, artifacts, training = _publish(tmp_path)
    assert completed.config == config
    assert tuple(completed.stage_artifacts) == FACTOR_FILM_EVALUATION_STAGE_ORDER
    assert completed.completion["complete"] is True
    assert completed.completion["development_quality_gate_passed"] is False
    assert completed.completion["final_benchmark_authorized"] is False
    assert completed.completion["test_split_accessed"] is False
    assert completed.completion["fresh_seed_accessed"] is False
    assert completed.completion["final_schedule_accessed"] is False
    assert completed.completion["smolvla_go"] is False
    assert (training / "sentinel.bin").read_bytes() == b"immutable training data"
    assert tuple(training.iterdir()) == (training / "sentinel.bin",)
    assert len(cast(tuple[object, ...], completed.manifest["artifacts"])) == 8
    assert (
        completed.stage_artifacts[FactorFiLMEvaluationStage.QUALITY_GATE].to_dict()
        == artifacts[FactorFiLMEvaluationStage.QUALITY_GATE].to_dict()
    )
    with pytest.raises(TypeError):
        completed.completion["complete"] = False  # type: ignore[index]


def test_identical_reuse_is_idempotent_but_different_payload_is_rejected(
    tmp_path: Path,
) -> None:
    completed, config, artifacts, training = _publish(tmp_path)
    reused = stage_and_promote_factor_film_evaluation_evidence(
        output_root=tmp_path / "diagnostics",
        training_run_root=training,
        config=config,
        stage_artifacts=artifacts,
    )
    assert reused.evidence_fingerprint == completed.evidence_fingerprint
    changed = dict(artifacts)
    gate = changed[FactorFiLMEvaluationStage.QUALITY_GATE]
    changed[FactorFiLMEvaluationStage.QUALITY_GATE] = replace(
        gate,
        payload={
            **dict(gate.payload),
            "development_quality_gate_passed": True,
            "final_benchmark_authorized": True,
        },
        artifact_fingerprint="",
    )
    with pytest.raises(FactorFiLMEvidenceError, match="differ from expected evidence"):
        stage_and_promote_factor_film_evaluation_evidence(
            output_root=tmp_path / "diagnostics",
            training_run_root=training,
            config=config,
            stage_artifacts=changed,
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        "stages/development.json",
        "manifest.json",
        "complete.json",
    ],
)
def test_tamper_is_rejected(tmp_path: Path, relative_path: str) -> None:
    completed, _, _, training = _publish(tmp_path)
    path = completed.root / relative_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tampered"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FactorFiLMEvidenceError):
        validate_completed_factor_film_evaluation_evidence(
            completed.root,
            training_run_root=training,
        )


def test_extra_file_is_rejected(tmp_path: Path) -> None:
    completed, _, _, training = _publish(tmp_path)
    (completed.root / "unowned.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FactorFiLMEvidenceError, match="incomplete or contain extras"):
        validate_completed_factor_film_evaluation_evidence(
            completed.root,
            training_run_root=training,
        )


@pytest.mark.parametrize("relation", ["same", "inside", "contains"])
def test_output_must_not_overlap_training_run(tmp_path: Path, relation: str) -> None:
    queue = _queue()
    config = _config(queue)
    artifacts = _stage_artifacts(config, queue)
    training = tmp_path / "training"
    training.mkdir()
    if relation == "same":
        output = training
    elif relation == "inside":
        output = training / "diagnostics"
    else:
        output = tmp_path
    with pytest.raises(FactorFiLMEvidenceError, match="must not overlap"):
        stage_and_promote_factor_film_evaluation_evidence(
            output_root=output,
            training_run_root=training,
            config=config,
            stage_artifacts=artifacts,
        )
    assert not (output / "factor-film-evaluation-evidence").exists()


def test_matching_staging_requires_explicit_cleanup_and_owner_match(tmp_path: Path) -> None:
    queue = _queue()
    config = _config(queue)
    artifacts = _stage_artifacts(config, queue)
    output = tmp_path / "diagnostics"
    training = tmp_path / "models" / "run"
    training.mkdir(parents=True)
    evidence = output / "factor-film-evaluation-evidence"
    evidence.mkdir(parents=True)
    staging = evidence / f".staging-{config.fingerprint.removeprefix('sha256:')}"
    staging.mkdir()
    (staging / "owner.json").write_text(
        json.dumps(
            {
                "schema_version": "langmani-m43-factor-film-evaluation-owner-v0",
                "config_fingerprint": config.fingerprint,
                "config": config.to_dict(),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FactorFiLMEvidenceError, match="explicit clean_matching_staging"):
        stage_and_promote_factor_film_evaluation_evidence(
            output_root=output,
            training_run_root=training,
            config=config,
            stage_artifacts=artifacts,
        )
    completed = stage_and_promote_factor_film_evaluation_evidence(
        output_root=output,
        training_run_root=training,
        config=config,
        stage_artifacts=artifacts,
        clean_matching_staging=True,
    )
    assert completed.root.is_dir() and not staging.exists()

    other_output = tmp_path / "other-diagnostics"
    other_evidence = other_output / "factor-film-evaluation-evidence"
    other_evidence.mkdir(parents=True)
    other_staging = other_evidence / staging.name
    other_staging.mkdir()
    (other_staging / "owner.json").write_text(
        json.dumps({"config_fingerprint": _digest("foreign")}), encoding="utf-8"
    )
    with pytest.raises(FactorFiLMEvidenceError, match="owned by another config"):
        stage_and_promote_factor_film_evaluation_evidence(
            output_root=other_output,
            training_run_root=training,
            config=config,
            stage_artifacts=artifacts,
            clean_matching_staging=True,
        )


def test_missing_stage_and_noncanonical_reference_are_rejected(tmp_path: Path) -> None:
    queue = _queue()
    config = _config(queue)
    artifacts = _stage_artifacts(config, queue)
    training = tmp_path / "run"
    training.mkdir()
    missing = dict(artifacts)
    missing.pop(FactorFiLMEvaluationStage.SEMANTIC)
    with pytest.raises(FactorFiLMEvidenceError, match="exactly six"):
        stage_and_promote_factor_film_evaluation_evidence(
            output_root=tmp_path / "evidence-a",
            training_run_root=training,
            config=config,
            stage_artifacts=missing,
        )
    changed = dict(artifacts)
    selection = changed[FactorFiLMEvaluationStage.SELECTION]
    changed[FactorFiLMEvaluationStage.SELECTION] = replace(
        selection,
        input_artifact_fingerprints={"validation": _digest("foreign")},
        artifact_fingerprint="",
    )
    with pytest.raises(FactorFiLMEvidenceError, match="not canonical"):
        stage_and_promote_factor_film_evaluation_evidence(
            output_root=tmp_path / "evidence-b",
            training_run_root=training,
            config=config,
            stage_artifacts=changed,
        )


def test_forbidden_test_fresh_or_final_identity_is_rejected() -> None:
    queue = _queue()
    config = _config(queue)
    with pytest.raises(FactorFiLMEvidenceError, match="prohibited evaluation identity"):
        FactorFiLMEvaluationStageArtifact(
            stage=FactorFiLMEvaluationStage.VALIDATION,
            run_fingerprint=config.run_fingerprint,
            input_artifact_fingerprints={
                "validation_queue": config.validation_queue_fingerprint,
                "validation_schedule": config.validation_schedule_digest,
            },
            payload={"source": "m3b_test"},
        )
    with pytest.raises(FactorFiLMEvidenceError, match="prohibited source flag"):
        FactorFiLMEvaluationStageArtifact(
            stage=FactorFiLMEvaluationStage.VALIDATION,
            run_fingerprint=config.run_fingerprint,
            input_artifact_fingerprints={
                "validation_queue": config.validation_queue_fingerprint,
                "validation_schedule": config.validation_schedule_digest,
            },
            payload={"fresh_seed_accessed": True},
        )


def test_reload_and_quality_gate_must_be_truthfully_bound(tmp_path: Path) -> None:
    queue = _queue()
    config = _config(queue)
    artifacts = _stage_artifacts(config, queue)
    training = tmp_path / "run"
    training.mkdir()
    reload = artifacts[FactorFiLMEvaluationStage.RELOAD]
    bad_reload = dict(artifacts)
    bad_reload[FactorFiLMEvaluationStage.RELOAD] = replace(
        reload,
        payload={**dict(reload.payload), "reload_validated": False},
        artifact_fingerprint="",
    )
    development = bad_reload[FactorFiLMEvaluationStage.DEVELOPMENT]
    bad_reload[FactorFiLMEvaluationStage.DEVELOPMENT] = replace(
        development,
        input_artifact_fingerprints={
            "reload": bad_reload[FactorFiLMEvaluationStage.RELOAD].artifact_fingerprint,
            "development_schedule": config.development_schedule_fingerprint,
        },
        artifact_fingerprint="",
    )
    semantic = bad_reload[FactorFiLMEvaluationStage.SEMANTIC]
    bad_reload[FactorFiLMEvaluationStage.SEMANTIC] = replace(
        semantic,
        input_artifact_fingerprints={
            "validation": bad_reload[FactorFiLMEvaluationStage.VALIDATION].artifact_fingerprint,
            "development": bad_reload[FactorFiLMEvaluationStage.DEVELOPMENT].artifact_fingerprint,
            "semantic_audit": config.semantic_audit_evidence_fingerprint,
        },
        artifact_fingerprint="",
    )
    gate = bad_reload[FactorFiLMEvaluationStage.QUALITY_GATE]
    bad_reload[FactorFiLMEvaluationStage.QUALITY_GATE] = replace(
        gate,
        input_artifact_fingerprints={
            "development": bad_reload[FactorFiLMEvaluationStage.DEVELOPMENT].artifact_fingerprint,
            "semantic": bad_reload[FactorFiLMEvaluationStage.SEMANTIC].artifact_fingerprint,
        },
        artifact_fingerprint="",
    )
    with pytest.raises(FactorFiLMEvidenceError, match="fresh-instance reload"):
        stage_and_promote_factor_film_evaluation_evidence(
            output_root=tmp_path / "bad-reload",
            training_run_root=training,
            config=config,
            stage_artifacts=bad_reload,
        )
    gate = artifacts[FactorFiLMEvaluationStage.QUALITY_GATE]
    bad_gate = dict(artifacts)
    bad_gate[FactorFiLMEvaluationStage.QUALITY_GATE] = replace(
        gate,
        payload={
            **dict(gate.payload),
            "development_quality_gate_passed": True,
            "final_benchmark_authorized": False,
        },
        artifact_fingerprint="",
    )
    with pytest.raises(FactorFiLMEvidenceError, match="inconsistent"):
        stage_and_promote_factor_film_evaluation_evidence(
            output_root=tmp_path / "bad-gate",
            training_run_root=training,
            config=config,
            stage_artifacts=bad_gate,
        )
