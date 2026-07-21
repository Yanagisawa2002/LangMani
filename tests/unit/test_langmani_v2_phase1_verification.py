from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from langmani.v2.act_adapter import ACT_ADAPTER_NAME, ACT_RUNTIME_MANIFEST_SCHEMA, ActAdapterConfig
from langmani.v2.evaluator import EpisodeOutcome, EvaluationEpisodeResult, persist_evaluation
from langmani.v2.phase1_verification import (
    Phase1VerificationError,
    validate_phase1_runtime_evidence,
    verify_phase1_preconditions,
)
from langmani.v2.taxonomy import canonical_pick_and_place_tasks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEEDS = (41001, 41002, 41003)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _create_phase1_fixture(tmp_path: Path) -> dict[str, Path | ActAdapterConfig]:
    task = canonical_pick_and_place_tasks()[0]
    run_root = tmp_path / "outputs/models/act/canonical"
    checkpoint = run_root / "checkpoints/selected"
    hashes: dict[str, str] = {}
    for relative in (
        "pretrained_model/model.safetensors",
        "pretrained_model/policy_preprocessor_step_3_normalizer_processor.safetensors",
        "pretrained_model/policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ):
        artifact = checkpoint / relative
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(relative.encode())
        hashes[relative] = hashlib.sha256(relative.encode()).hexdigest()
    config = ActAdapterConfig(
        policy_id="canonical-act",
        adapter_name=ACT_ADAPTER_NAME,
        run_root="outputs/models/act/canonical",
        selected_checkpoint_relative_path="checkpoints/selected",
        expected_run_fingerprint="sha256:" + "1" * 64,
        expected_checkpoint_fingerprint="sha256:" + "2" * 64,
        expected_dataset_fingerprint="sha256:" + "3" * 64,
        expected_task_id=task.canonical_task_id,
        execution_horizon=10,
        action_bound_mode="project",
        device="cuda",
        dtype="float32",
        artifact_sha256=hashes,
    )
    policy_config = tmp_path / "configs/policy.json"
    _write_json(policy_config, config.to_dict())
    task_config = tmp_path / "configs/tasks.json"
    task_config.parent.mkdir(parents=True, exist_ok=True)
    task_config.write_text(
        (PROJECT_ROOT / "configs/langmani_v2/tasks/pick_and_place_v0.json").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    runtime_manifest = {
        "schema_version": ACT_RUNTIME_MANIFEST_SCHEMA,
        "policy_identity": {
            "policy_id": config.policy_id,
            "adapter_name": ACT_ADAPTER_NAME,
            "checkpoint_identity": config.expected_checkpoint_fingerprint,
        },
        "repository_config": "configs/policy.json",
        "execution_horizon": 10,
        "action_shape": [8],
        "task_compatibility": [task.canonical_task_id],
    }
    results = tuple(
        EvaluationEpisodeResult(
            evaluation_id=f"fixture:{seed}",
            seed=seed,
            task_identity=task.canonical_task_id,
            skill_family="pick_and_place",
            policy_identity=config.policy_id,
            checkpoint_identity=config.expected_checkpoint_fingerprint,
            outcome=EpisodeOutcome.TIMEOUT,
            success=False,
            timeout=True,
            wrong_object_interaction=False,
            object_drop_or_loss=False,
            invalid_action=False,
            episode_length=200,
            action_projection_count=0,
            action_projection_component_count=0,
            policy_query_count=20,
            policy_inference_latency_ms={
                "mean": 2.0,
                "p50": 2.0,
                "p95": 3.0,
                "maximum": 3.0,
            },
            environment_step_latency_ms={
                "mean": 4.0,
                "p50": 4.0,
                "p95": 5.0,
                "maximum": 5.0,
            },
            final_evaluation={"success": False},
            failure_reason="episode step budget exhausted",
        )
        for seed in SEEDS
    )
    evidence_root = tmp_path / "outputs/diagnostics/v2/phase1/evaluation"
    persist_evaluation(
        output_root=evidence_root,
        runtime_manifest=runtime_manifest,
        results=results,
    )
    frozen = tmp_path / "frozen.txt"
    frozen.write_text("frozen\n", encoding="utf-8")
    release_manifest = tmp_path / "releases/langmani_v1/manifest.yaml"
    _write_json(
        release_manifest,
        {
            "schema_version": "langmani-v1-release-manifest-v0",
            "release": {"tag": "v1.0.0", "commit": "a" * 40},
            "artifact_identities": {"dataset": "sha256:" + "b" * 64},
            "frozen_files": [
                {
                    "path": "frozen.txt",
                    "sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
                }
            ],
        },
    )
    return {
        "policy_config": policy_config,
        "task_config": task_config,
        "evidence_root": evidence_root,
        "release_manifest": release_manifest,
        "config": config,
    }


def test_phase1_runtime_evidence_requires_three_real_policy_rollouts(tmp_path: Path) -> None:
    fixture = _create_phase1_fixture(tmp_path)

    result = validate_phase1_runtime_evidence(
        project_root=tmp_path,
        policy_config_path=fixture["policy_config"],
        task_config_path=fixture["task_config"],
        evidence_root=fixture["evidence_root"],
        expected_seeds=SEEDS,
    )

    assert result.episode_count == 3
    assert result.policy_query_count == 60
    assert result.environment_step_count == 600
    assert result.success_count == 0


def test_phase1_runtime_evidence_rejects_no_policy_query(tmp_path: Path) -> None:
    fixture = _create_phase1_fixture(tmp_path)
    episodes_path = fixture["evidence_root"] / "episodes.jsonl"
    records = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines()]
    records[0]["policy_query_count"] = 0
    episodes_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(Phase1VerificationError, match="no real policy query"):
        validate_phase1_runtime_evidence(
            project_root=tmp_path,
            policy_config_path=fixture["policy_config"],
            task_config_path=fixture["task_config"],
            evidence_root=fixture["evidence_root"],
            expected_seeds=SEEDS,
        )


def test_phase2_precondition_fails_closed_when_runtime_evidence_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _create_phase1_fixture(tmp_path)
    monkeypatch.setattr("langmani.v2.release._git", lambda *_args: "a" * 40)

    result = verify_phase1_preconditions(
        project_root=tmp_path,
        release_manifest_path=fixture["release_manifest"],
        policy_config_path=fixture["policy_config"],
        task_config_path=fixture["task_config"],
        evidence_root=tmp_path / "outputs/diagnostics/v2/phase1/missing",
        expected_seeds=SEEDS,
    )

    assert result.v1_release_validated
    assert not result.phase1_runtime_evidence_validated
    assert not result.real_policy_inference_validated
    assert not result.real_environment_steps_validated
    assert not result.phase2_authorized
    assert result.errors and "evidence is missing" in result.errors[0]


def test_phase2_precondition_fails_closed_when_policy_config_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _create_phase1_fixture(tmp_path)
    monkeypatch.setattr("langmani.v2.release._git", lambda *_args: "a" * 40)

    result = verify_phase1_preconditions(
        project_root=tmp_path,
        release_manifest_path=fixture["release_manifest"],
        policy_config_path=tmp_path / "configs/missing-policy.json",
        task_config_path=fixture["task_config"],
        evidence_root=fixture["evidence_root"],
        expected_seeds=SEEDS,
    )

    assert result.v1_release_validated
    assert not result.phase1_runtime_evidence_validated
    assert not result.phase2_authorized
    assert result.errors and "cannot read ACT adapter config" in result.errors[0]
