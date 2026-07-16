"""Independent M4.3b target-development verifier contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import stable_scene_id, stable_task_id
from langmani.policies.act_factor_film_evaluation import (
    FACTOR_FILM_DEVELOPMENT_POLICY_LABELS,
    DevelopmentEpisodeIdentity,
    DevelopmentPolicyMetrics,
    DevelopmentSemanticMetrics,
    FactorFiLMValidationCountResult,
    evaluate_development_quality_gate,
    validate_paired_development_identities,
)
from langmani.policies.act_factor_film_types import canonical_fingerprint
from langmani.policies.act_factor_film_verification import (
    M43B_ARCHITECTURE_BASELINE_GIT_COMMIT,
    FactorFiLMIndependentVerificationError,
    FactorFiLMIndependentVerificationReport,
    _parse_structural_flags,
    _validate_development_payload,
    _validate_metrics_file,
    _validate_semantic_payload,
    _validate_validation_benchmark_reports,
    verify_factor_film_target_development,
)
from langmani.policies.m42_schedule import M42_DEV_SCHEDULE_FINGERPRINT

_TASK_IDS = tuple(stable_task_id(value) for value in CANONICAL_TASK_SPECS)


def _digest(label: str) -> str:
    return canonical_fingerprint({"fixture": label})


def _paired_identities():
    identities = tuple(
        DevelopmentEpisodeIdentity(
            episode_index=index,
            scene_seed=1_000 + index // 6,
            scene_id=stable_scene_id(1_000 + index // 6),
            task_id=_TASK_IDS[index % 6],
        )
        for index in range(72)
    )
    return validate_paired_development_identities(
        per_task=identities,
        state_onehot=identities,
        factor_film=identities,
        schedule_fingerprint=M42_DEV_SCHEDULE_FINGERPRINT,
    )


def _development_metrics(
    *, success_count: int = 54, report_fingerprint: str | None = None
) -> DevelopmentPolicyMetrics:
    per_task = success_count // 6
    return DevelopmentPolicyMetrics(
        report_fingerprint=report_fingerprint or _digest("development"),
        success_count=success_count,
        per_task_success_counts={task_id: per_task for task_id in _TASK_IDS},
        wrong_object_grasp_count=2,
        wrong_object_in_target_bin_count=0,
        target_in_wrong_bin_count=0,
        target_off_table_count=0,
        timeout_count=12,
        arm_projection_count=0,
        nan_count=0,
        inf_count=0,
        malformed_action_count=0,
    )


def _semantic_metrics() -> DevelopmentSemanticMetrics:
    return DevelopmentSemanticMetrics(
        semantic_report_fingerprint=_digest("semantic"),
        full_task_top1_retrieval=0.75,
        target_object_retrieval=0.85,
        task_sensitivity_ratio_relative_to_per_task=0.8,
    )


def _development_payload() -> dict[str, object]:
    identities = _paired_identities().ordered_episode_identities
    per_task_reports = [
        _benchmark_report(
            tuple(item for item in identities if item.task_id == task_id),
            successes=10,
        )
        for task_id in _TASK_IDS
    ]
    onehot_report = _benchmark_report(identities, successes=48)
    factor_report = _benchmark_report(identities, successes=54)
    metrics = _development_metrics(report_fingerprint=canonical_fingerprint(factor_report))
    return {
        "development_benchmark_completed": True,
        "total_episode_count": 216,
        "per_policy_episode_counts": {label: 72 for label in FACTOR_FILM_DEVELOPMENT_POLICY_LABELS},
        "paired_episode_identities_validated": True,
        "paired_identities": _paired_identities().to_dict(),
        "factor_film_metrics": metrics.to_dict(),
        "physical_execution": True,
        "device": "cuda",
        "environment_id": "LangMani-PickPlaceByInstruction-v0",
        "schedule_id": "m42_dev_v0",
        "schedule_fingerprint": M42_DEV_SCHEDULE_FINGERPRINT,
        "execution_horizon": 10,
        "gripper_mode": "project",
        "m2_expert_invoked": False,
        "benchmark_reports": {
            "per_task": per_task_reports,
            "state_onehot": onehot_report,
            "factor_film": factor_report,
        },
    }


def _benchmark_report(
    identities: tuple[DevelopmentEpisodeIdentity, ...], *, successes: int
) -> dict[str, object]:
    per_task: dict[str, dict[str, int]] = {}
    for task_id in _TASK_IDS:
        count = sum(item.task_id == task_id for item in identities)
        if count:
            per_task[task_id] = {
                "episodes": count,
                "successes": 9 if len(identities) == 72 else successes,
            }
    if len(identities) == 72:
        per_task = {task_id: {"episodes": 12, "successes": 9} for task_id in _TASK_IDS}
    return {
        "schedule_id": "m42_dev_v0",
        "schedule_fingerprint": M42_DEV_SCHEDULE_FINGERPRINT,
        "execution_horizon": 10,
        "gripper_mode": "project",
        "episodes": [
            {
                "episode_index": item.episode_index,
                "rollout": {
                    "split": "development",
                    "scene_seed": item.scene_seed,
                    "scene_id": item.scene_id,
                    "task_id": item.task_id,
                },
            }
            for item in identities
        ],
        "aggregate": {
            "episode_count": len(identities),
            "successes": successes,
            "wrong_object_grasp_count": 2 if len(identities) == 72 else 0,
            "wrong_object_in_target_bin_count": 0,
            "target_in_wrong_bin_count": 0,
            "target_off_table_count": 0,
            "timeout_count": 12 if len(identities) == 72 else 0,
            "per_task": per_task,
            "action_metrics": {
                "arm_projected_component_count": 0,
                "nan_count": 0,
                "inf_count": 0,
                "malformed_action_count": 0,
            },
        },
    }


def _semantic_payload() -> dict[str, object]:
    return {
        "semantic_evaluation_completed": True,
        "post_grasp_analysis_completed": True,
        "first_interaction_analysis_completed": True,
        "raw_action_metrics_validated": True,
        "runtime_action_metrics_validated": True,
        "validation_observation_count": 36,
        "development_observation_count": 72,
        "first_interaction_count": 72,
        "first_interaction_confusions": {
            "requested_object_to_first_grasped_object": {},
            "requested_task_to_nearest_per_task_chunk": {},
            "nearest_per_task_chunk_to_actual_first_grasp": {},
            "requested_bin_to_first_approached_bin": {},
            "requested_bin_to_object_entry_bin": {},
        },
        "semantic_error_timing": {
            "visible_in_initial_chunk": 3,
            "emerged_later": 4,
            "no_semantic_error": 60,
            "no_first_interaction": 5,
        },
        "semantic_metrics": _semantic_metrics().to_dict(),
    }


def test_strict_development_and_semantic_payloads_cover_real_counts() -> None:
    paired, development, per_task_successes = _validate_development_payload(_development_payload())
    semantic = _validate_semantic_payload(_semantic_payload())
    assert paired.episode_count == 72
    assert development.episode_count == 72
    assert per_task_successes == 60
    assert semantic.observation_count == 72


def test_validation_counts_are_recomputed_from_checksum_bound_reports() -> None:
    schedule_digest = _digest("validation-schedule")
    results = tuple(
        FactorFiLMValidationCountResult(
            checkpoint_fingerprint=_digest(f"checkpoint-{index}"),
            checkpoint_step=(index + 1) * 5_000,
            schedule_digest=schedule_digest,
            success_count=30,
            wrong_object_interaction_count=1,
            wrong_object_in_target_bin_count=0,
            target_off_table_count=0,
            timeout_count=6,
            offline_validation_action_loss=1.0 + index,
        )
        for index in range(20)
    )
    episodes = [
        {
            "episode_index": scene_index * 6 + task_index,
            "rollout": {
                "split": "validation",
                "scene_id": f"scene-{scene_index}",
                "task_id": task_id,
            },
        }
        for scene_index in range(6)
        for task_index, task_id in enumerate(_TASK_IDS)
    ]
    reports = [
        {
            "schedule_id": "m3b_validation_v0",
            "schedule_fingerprint": schedule_digest,
            "execution_horizon": 10,
            "gripper_mode": "project",
            "checkpoint": {
                "checkpoint_fingerprint": result.checkpoint_fingerprint,
            },
            "episodes": episodes,
            "aggregate": {
                "episode_count": 36,
                "successes": 30,
                "wrong_object_interaction_count": 1,
                "wrong_object_in_target_bin_count": 0,
                "target_off_table_count": 0,
                "timeout_count": 6,
            },
        }
        for result in results
    ]
    payload = {"benchmark_reports": reports}
    _validate_validation_benchmark_reports(payload, results, schedule_digest=schedule_digest)

    malformed = json.loads(json.dumps(payload))
    malformed["benchmark_reports"][0]["episodes"][0]["rollout"]["split"] = "fresh_seed"
    with pytest.raises(FactorFiLMIndependentVerificationError, match="non-validation"):
        _validate_validation_benchmark_reports(malformed, results, schedule_digest=schedule_digest)


def test_development_rejects_factor_only_or_unpaired_evidence() -> None:
    partial = _development_payload()
    partial["total_episode_count"] = 72
    with pytest.raises(FactorFiLMIndependentVerificationError, match="216"):
        _validate_development_payload(partial)

    unpaired = _development_payload()
    unpaired["per_policy_episode_counts"] = {"ACT-Mixed-FactorFiLM": 72}
    with pytest.raises(FactorFiLMIndependentVerificationError, match="each contain 72"):
        _validate_development_payload(unpaired)


def test_semantic_rejects_partial_first_interaction_or_missing_timing() -> None:
    partial = _semantic_payload()
    partial["first_interaction_count"] = 71
    with pytest.raises(FactorFiLMIndependentVerificationError, match="cover 72"):
        _validate_semantic_payload(partial)

    missing = _semantic_payload()
    missing["semantic_error_timing"] = {
        "visible_in_initial_chunk": 1,
        "emerged_later": 2,
    }
    with pytest.raises(FactorFiLMIndependentVerificationError, match="timing"):
        _validate_semantic_payload(missing)


def test_quality_failure_does_not_falsify_experiment_or_physical_flags() -> None:
    failing_metrics = _development_metrics(success_count=48)
    gate = evaluate_development_quality_gate(
        factor_film_metrics=failing_metrics,
        semantic_metrics=_semantic_metrics(),
        per_task_reference_success_count=60,
    )
    assert gate.development_quality_gate_passed is False

    report = FactorFiLMIndependentVerificationReport(
        implementation_validated=True,
        semantic_audit_completed=True,
        factor_film_implementation_validated=True,
        factor_film_fixture_training_validated=True,
        factor_film_training_completed=True,
        factor_film_checkpoints_complete=True,
        factor_film_checkpoint_selected=True,
        validation_only_selection_validated=True,
        factor_film_reload_validated=True,
        development_benchmark_completed=True,
        post_grasp_analysis_completed=True,
        first_interaction_analysis_completed=True,
        raw_action_metrics_validated=True,
        runtime_action_metrics_validated=True,
        development_quality_gate_passed=False,
        final_benchmark_authorized=False,
        physical_target_validated=True,
    )
    assert report.passed is True
    assert report.to_dict()["development_quality_gate_passed"] is False
    assert report.to_dict()["physical_target_validated"] is True


def test_metrics_validation_is_read_only_and_rejects_nonfinite(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps({"step": 1, "total_loss": 0.5, "validation_loss": None})
        + "\n"
        + json.dumps({"step": 2, "total_loss": 0.25, "validation_loss": 0.4})
        + "\n",
        encoding="utf-8",
    )
    before = path.read_bytes()
    _validate_metrics_file(tmp_path, final_step=2)
    assert path.read_bytes() == before

    path.write_text('{"step": 2, "total_loss": NaN}\n', encoding="utf-8")
    with pytest.raises(FactorFiLMIndependentVerificationError, match="non-finite"):
        _validate_metrics_file(tmp_path, final_step=2)


def test_independent_verifier_requires_explicit_full_training_commit(tmp_path: Path) -> None:
    with pytest.raises(FactorFiLMIndependentVerificationError, match="full lowercase Git hex"):
        verify_factor_film_target_development(
            training_run_root=tmp_path / "training",
            evaluation_evidence_root=tmp_path / "evidence",
            expected_training_git_commit="8ee0f1b",
            structural_verification_path=tmp_path / "structural.json",
        )


def test_structural_verification_stays_bound_to_architecture_baseline(tmp_path: Path) -> None:
    path = tmp_path / "structural.json"
    payload = {
        "passed": True,
        "implementation_git_commit": M43B_ARCHITECTURE_BASELINE_GIT_COMMIT,
        "implementation_git_dirty": False,
        "factor_film_implementation_validated": True,
        "factor_film_fixture_training_validated": True,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _parse_structural_flags(path) == (True, True)

    payload["implementation_git_commit"] = "a" * 40
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FactorFiLMIndependentVerificationError, match="architecture baseline"):
        _parse_structural_flags(path)
