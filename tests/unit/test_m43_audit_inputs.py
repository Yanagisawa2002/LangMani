from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import stable_task_id
from langmani.policies import m42_evidence
from langmani.policies.act_conditioning import CANONICAL_TASK_IDS
from langmani.policies.act_data import (
    COMPLETION_SCHEMA,
    FeatureStatistics,
    StatisticsLeakageAudit,
    TrainOnlyStatistics,
)
from langmani.policies.act_evaluation import (
    create_checkpoint_selection,
)
from langmani.policies.act_types import (
    ActDataConfig,
    ActExperimentConfig,
    ActModelConfig,
    ActRunIdentity,
    ActVariant,
    CheckpointRecord,
    ExperimentMode,
    TrainingState,
    ValidationResult,
)
from langmani.policies.m42_analysis import create_task_token_checkpoint_selection
from langmani.policies.m42_training import (
    RUNTIME_SELECTION_SCHEMA,
    TaskTokenRunIdentity,
    TaskTokenTrainingManifest,
    TaskTokenValidationQueue,
    TaskTokenValidationQueueItem,
    canonical_fingerprint,
)
from langmani.policies.m42_types import TaskTokenValidationResult
from langmani.policies.m43_audit_inputs import (
    M43AuditInputError,
    _evaluation_implementation_fingerprint,
    load_restricted_audit_inputs,
)


def _digest(index: int) -> str:
    return f"sha256:{index:064x}"


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _validation_schedule() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for scene_index in range(6):
        for task_index, (task_id, task_spec) in enumerate(
            zip(CANONICAL_TASK_IDS, CANONICAL_TASK_SPECS, strict=True)
        ):
            result.append(
                {
                    "episode_index": 288 + scene_index * 6 + task_index,
                    "scene_seed": 10_000 + scene_index,
                    "scene_group_id": f"validation-group-{scene_index}",
                    "task_id": task_id,
                    "task_spec": task_spec.to_dict(),
                }
            )
    return result


def _statistics(
    *,
    variant: ActVariant,
    fingerprint_seed: int,
    action_std: tuple[float, ...],
    task_id: str | None = None,
) -> TrainOnlyStatistics:
    state_dimension = 15 if variant is ActVariant.MIXED_TASK_ONEHOT else 9

    def feature(
        name: str, count: int, *, std: tuple[float, ...] | None = None
    ) -> FeatureStatistics:
        values = std or tuple(1.0 for _ in range(count))
        return FeatureStatistics(
            feature_name=name,
            components=tuple(f"{name}_{index}" for index in range(count)),
            count=288,
            minimum=tuple(-1.0 for _ in range(count)),
            maximum=tuple(1.0 for _ in range(count)),
            mean=tuple(0.0 for _ in range(count)),
            std=values,
        )

    # fingerprint_seed keeps the helper signature explicit; the canonical
    # typed contract owns the actual fingerprint.
    assert fingerprint_seed >= 0
    return TrainOnlyStatistics(
        variant=variant,
        task_id=task_id,
        image=feature("observation.image", 3),
        state=feature("observation.state", state_dimension),
        action=feature("action", 8, std=action_std),
        leakage_audit=StatisticsLeakageAudit(
            source_episode_indices=tuple(range(288)),
            train_episode_indices=tuple(range(288)),
            validation_episode_indices=tuple(range(288, 324)),
            test_episode_indices=tuple(range(324, 360)),
        ),
        lerobot_version="0.6.0",
        torch_version="2.11.0+cu128",
    )


def _m4_run(
    root: Path,
    *,
    run_index: int,
    variant: ActVariant,
    task_id: str | None,
    dataset_fingerprint: str,
    split_digest: str,
    action_std: tuple[float, ...],
    mode: ExperimentMode = ExperimentMode.FULL,
    complete: bool = True,
) -> dict[str, object]:
    config = ActExperimentConfig(
        variant=variant,
        task_id=task_id,
        data=ActDataConfig(dataset_root="fixture/m3b"),
        model=ActModelConfig.for_variant(variant),
        mode=mode,
        device="cuda",
    )
    if variant is ActVariant.PER_TASK:
        task_index = CANONICAL_TASK_IDS.index(task_id)
        train_indices = tuple(range(task_index * 48, (task_index + 1) * 48))
        validation_indices = tuple(range(288 + task_index, 324, 6))
    else:
        train_indices = tuple(range(288))
        validation_indices = tuple(range(288, 324))
    statistics = _statistics(
        variant=variant,
        fingerprint_seed=run_index,
        action_std=action_std,
        task_id=task_id,
    )
    identity = ActRunIdentity(
        m3b_export_fingerprint=dataset_fingerprint,
        m3b_split_manifest_digest=split_digest,
        ordered_train_episode_indices=train_indices,
        ordered_validation_episode_indices=validation_indices,
        variant=variant,
        task_id=task_id,
        task_onehot_mapping_version="CanonicalTaskOneHotV0",
        model_config=config.model.to_dict(),
        data_contract={
            "evaluation_schedules": {
                "validation": [],
                # These declarations deliberately mimic historical manifests;
                # the restricted loader must not follow their sibling paths.
                "test": [{"declaration_only": True}],
                "fresh_seed": [{"declaration_only": True}],
            }
        },
        train_statistics_fingerprint=statistics.statistics_fingerprint,
        optimization_config=config.optimization.to_dict(),
        training_seed=0,
        experiment_mode=mode,
        device="cuda",
        dtype="float32",
        lerobot_version="0.6.0",
        torch_version="2.11.0+cu128",
        cuda_version="12.8",
        git_commit="a" * 40,
        git_dirty=False,
        dirty_development_override=False,
    )
    checkpoint = CheckpointRecord(
        checkpoint_fingerprint=_digest(100 + run_index),
        run_fingerprint=identity.run_fingerprint,
        global_step=100_000,
        relative_path="checkpoints/step-00100000",
        policy_config_digest=_digest(400 + run_index),
        statistics_fingerprint=statistics.statistics_fingerprint,
        split_digest=split_digest,
        git_commit=identity.git_commit,
        complete=True,
    )
    manifest = {
        "identity": identity,
        "config": config,
        "git_dirty": False,
        "dirty_development_override": False,
        "train_statistics_fingerprint": statistics.statistics_fingerprint,
        "training_state": TrainingState(
            global_step=100_000,
            examples_processed=3_200_000,
            last_checkpoint_fingerprint=checkpoint.checkpoint_fingerprint,
            completed=complete,
        ),
        "checkpoints": (checkpoint,),
        "selected_checkpoint_fingerprint": checkpoint.checkpoint_fingerprint,
        "complete": complete,
    }
    from langmani.policies.act_types import ActExperimentManifest

    typed_manifest = ActExperimentManifest(**manifest)
    candidate = ValidationResult(
        checkpoint_fingerprint=checkpoint.checkpoint_fingerprint,
        checkpoint_step=checkpoint.global_step,
        schedule_digest=_digest(500 + run_index),
        success_rate=0.5,
        wrong_object_interaction_rate=0.0,
        target_off_table_rate=0.0,
        offline_validation_action_loss=1.0,
    )
    selection = create_checkpoint_selection(
        run_fingerprint=identity.run_fingerprint,
        candidates=(candidate,),
        selection_timestamp_utc="2026-07-15T00:00:00Z",
    )
    run_root = root / f"{run_index + 1:064x}"
    _write(run_root / "run_manifest.json", typed_manifest.to_dict())
    _write(run_root / "checkpoint_selection.json", selection.to_dict())
    (run_root / checkpoint.relative_path).mkdir(parents=True)
    if variant is ActVariant.MIXED_TASK_ONEHOT:
        _write(run_root / "train_stats.json", statistics.to_dict())
    return {
        "run_root": run_root,
        "run_fingerprint": identity.run_fingerprint,
        "checkpoint_fingerprint": checkpoint.checkpoint_fingerprint,
        "checkpoint_relative_path": checkpoint.relative_path,
        "statistics_fingerprint": statistics.statistics_fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
        "split_digest": split_digest,
        "architecture_fingerprint": None,
        "git_commit": identity.git_commit,
        "task_id": task_id,
    }


def _fixture(tmp_path: Path, *, task_token_std: tuple[float, ...] | None = None) -> dict[str, Any]:
    dataset_root = tmp_path / "dataset"
    m4_root = tmp_path / "m4-models"
    token_root = tmp_path / "task-token-models"
    m42_root = tmp_path / "m42-diagnostics"
    dataset_fingerprint = _digest(1)
    split_digest = _digest(2)
    action_std = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    evaluation_git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    runtime_evaluation_git_commit = "b" * 40
    evaluation_implementation_fingerprint = _evaluation_implementation_fingerprint(
        evaluation_git_commit
    )
    _write(
        dataset_root / "langmani" / "complete.json",
        {"schema_version": COMPLETION_SCHEMA, "export_fingerprint": dataset_fingerprint},
    )

    per_task = [
        _m4_run(
            m4_root,
            run_index=index,
            variant=ActVariant.PER_TASK,
            task_id=task_id,
            dataset_fingerprint=dataset_fingerprint,
            split_digest=split_digest,
            action_std=action_std,
        )
        for index, task_id in enumerate(CANONICAL_TASK_IDS)
    ]
    onehot = _m4_run(
        m4_root,
        run_index=6,
        variant=ActVariant.MIXED_TASK_ONEHOT,
        task_id=None,
        dataset_fingerprint=dataset_fingerprint,
        split_digest=split_digest,
        action_std=action_std,
    )

    runtime_raw = {
        "schema_version": RUNTIME_SELECTION_SCHEMA,
        "execution_horizon": 5,
        "gripper_mode": "project",
        "horizon_selection_fingerprint": _digest(20),
        "gripper_selection_fingerprint": _digest(21),
        "runtime_fingerprint": _digest(22),
        "development_schedule_fingerprint": _digest(23),
        "m3b_dataset_fingerprint": dataset_fingerprint,
        "mixed_task_onehot_checkpoint_fingerprint": onehot["checkpoint_fingerprint"],
        "representative_per_task_checkpoint_fingerprint": per_task[2]["checkpoint_fingerprint"],
        "implementation_fingerprint": _digest(24),
        "experiment_manifest_fingerprint": _digest(25),
        "evaluation_git_commit": runtime_evaluation_git_commit,
        "selection_evidence": {"experiment_manifest": "experiment_manifest.json"},
        "locked": True,
        "final_schedule_accessed": False,
    }
    runtime_source_fingerprint = canonical_fingerprint(runtime_raw)
    runtime_path = m42_root / "runtime_ablation" / "runtime_selection.json"
    _write(runtime_path, runtime_raw)

    token_statistics = _statistics(
        variant=ActVariant.MIXED_UNCONDITIONED,
        fingerprint_seed=50,
        action_std=task_token_std or action_std,
    )
    validation_schedule = _validation_schedule()
    validation_schedule_fingerprint = canonical_fingerprint(
        {
            "schema_version": "langmani-m42-m3b-validation-schedule-v0",
            "m3b_dataset_fingerprint": dataset_fingerprint,
            "split_digest": split_digest,
            "episodes": validation_schedule,
        }
    )
    token_identity = TaskTokenRunIdentity(
        m3b_export_fingerprint=dataset_fingerprint,
        m3b_split_manifest_digest=split_digest,
        ordered_train_episode_indices=tuple(range(288)),
        ordered_validation_episode_indices=tuple(range(288, 324)),
        model_config={"fixture": "task-token"},
        data_contract={
            "validation_schedule_fingerprint": validation_schedule_fingerprint,
            "validation_schedule": validation_schedule,
            "selection_forbidden_sources": [
                "m3b_test",
                "m4_fresh_seed",
                "m42_final_v0",
            ],
        },
        train_statistics_fingerprint=token_statistics.statistics_fingerprint,
        optimization_config={"fixture": True},
        runtime_selection_fingerprint=runtime_raw["runtime_fingerprint"],
        runtime_selection_source_fingerprint=runtime_source_fingerprint,
        experiment_manifest_fingerprint=runtime_raw["experiment_manifest_fingerprint"],
        task_token_architecture_fingerprint=_digest(26),
        training_seed=0,
        device="cuda",
        dtype="float32",
        lerobot_version="0.6.0",
        torch_version="2.11.0+cu128",
        cuda_version="12.8",
        git_commit=evaluation_git_commit,
        git_dirty=False,
    )
    token_checkpoints = tuple(
        CheckpointRecord(
            checkpoint_fingerprint=_digest(600 + index),
            run_fingerprint=token_identity.run_fingerprint,
            global_step=(index + 1) * 5_000,
            relative_path=f"checkpoints/step-{(index + 1) * 5_000:08d}",
            policy_config_digest=_digest(700 + index),
            statistics_fingerprint=token_statistics.statistics_fingerprint,
            split_digest=split_digest,
            git_commit=token_identity.git_commit,
            complete=True,
        )
        for index in range(20)
    )
    token_manifest = TaskTokenTrainingManifest(
        identity=token_identity,
        task_token_experiment={},
        training_state=TrainingState(
            global_step=100_000,
            examples_processed=3_200_000,
            last_checkpoint_fingerprint=token_checkpoints[-1].checkpoint_fingerprint,
            completed=True,
        ),
        checkpoints=token_checkpoints,
        training_complete=True,
    )
    token_run_root = token_root / f"{99:064x}"
    _write(token_run_root / "run_manifest.json", token_manifest.to_dict())
    _write(token_run_root / "train_stats.json", token_statistics.to_dict())
    (token_run_root / token_checkpoints[-1].relative_path).mkdir(parents=True)
    queue = TaskTokenValidationQueue(
        run_fingerprint=token_identity.run_fingerprint,
        m3b_dataset_fingerprint=dataset_fingerprint,
        split_digest=split_digest,
        train_statistics_fingerprint=token_statistics.statistics_fingerprint,
        runtime_selection_fingerprint=token_identity.runtime_selection_fingerprint,
        experiment_manifest_fingerprint=token_identity.experiment_manifest_fingerprint,
        task_token_architecture_fingerprint=token_identity.task_token_architecture_fingerprint,
        validation_schedule_fingerprint=validation_schedule_fingerprint,
        ordered_validation_episode_indices=token_identity.ordered_validation_episode_indices,
        checkpoints=tuple(
            TaskTokenValidationQueueItem(
                global_step=item.global_step,
                checkpoint_fingerprint=item.checkpoint_fingerprint,
                checkpoint_relative_path=item.relative_path,
                offline_validation_loss=float(20 - index),
            )
            for index, item in enumerate(token_checkpoints)
        ),
        git_commit=token_identity.git_commit,
        complete=True,
    )
    _write(token_run_root / "validation_queue.json", queue.to_dict())

    validation_results = tuple(
        TaskTokenValidationResult(
            checkpoint_fingerprint=item.checkpoint_fingerprint,
            checkpoint_step=item.global_step,
            validation_schedule_fingerprint=validation_schedule_fingerprint,
            validation_split_digest=split_digest,
            episode_count=36,
            success_count=index + 1,
            wrong_object_interaction_count=0,
            target_off_table_count=0,
            timeout_count=0,
            offline_validation_action_loss=queue.checkpoints[index].offline_validation_loss,
        )
        for index, item in enumerate(token_checkpoints)
    )
    selection = create_task_token_checkpoint_selection(
        validation_results, locked_at_utc="2026-07-15T00:00:00+00:00"
    )
    development_root = m42_root / "development"
    _write(
        development_root / "task_token_checkpoint_selection.json",
        {
            "schema_version": "langmani-m42-task-token-selection-artifact-v0",
            "selection": selection.to_dict(),
            "selection_fingerprint": selection.fingerprint,
            "validation_result_fingerprints": list(selection.evidence_fingerprints),
            "selection_source": "m3b_validation_only",
            "test_split_accessed": False,
            "development_schedule_accessed": False,
            "final_schedule_accessed": False,
            "locked": True,
        },
    )
    for item, result in zip(queue.checkpoints, validation_results, strict=True):
        identity = {
            "schema_version": "langmani-m42-evaluation-artifact-v0",
            "stage": "validation_selection",
            "evaluation_git_commit": evaluation_git_commit,
            "implementation_fingerprint": evaluation_implementation_fingerprint,
            "experiment_manifest_fingerprint": token_identity.experiment_manifest_fingerprint,
            "run_fingerprint": token_identity.run_fingerprint,
            "checkpoint_fingerprint": item.checkpoint_fingerprint,
            "checkpoint_step": item.global_step,
            "validation_schedule_fingerprint": validation_schedule_fingerprint,
            "validation_split_digest": split_digest,
            "execution_horizon": runtime_raw["execution_horizon"],
            "gripper_mode": runtime_raw["gripper_mode"],
            "maximum_episode_steps": 200,
            "test_split_accessed": False,
            "development_schedule_accessed": False,
            "final_schedule_accessed": False,
        }
        identity_fingerprint = canonical_fingerprint(identity)
        artifact = {
            "schema_version": "langmani-m42-evaluation-artifact-v0",
            "identity": identity,
            "payload": {
                "benchmark": {},
                "benchmark_fingerprint": _digest(900 + item.global_step),
                "validation_result": result.to_dict(),
                "validation_result_fingerprint": result.fingerprint,
            },
            "passed": True,
        }
        evidence_root = development_root / "evidence" / identity_fingerprint.removeprefix("sha256:")
        _write(evidence_root / "artifact.json", artifact)
        _write(
            evidence_root / "complete.json",
            {
                "schema_version": "langmani-m42-evaluation-artifact-v0",
                "identity_fingerprint": identity_fingerprint,
                "artifact_fingerprint": canonical_fingerprint(artifact),
                "passed": True,
            },
        )

    def benchmark(checkpoint: dict[str, object], policy_kind: str) -> dict[str, object]:
        return {
            "schedule_id": "m42_dev_v0",
            "schedule_fingerprint": runtime_raw["development_schedule_fingerprint"],
            "checkpoint": {
                "policy_kind": policy_kind,
                "run_fingerprint": checkpoint["run_fingerprint"],
                "checkpoint_fingerprint": checkpoint["checkpoint_fingerprint"],
                "checkpoint_relative_path": checkpoint["checkpoint_relative_path"],
                "task_id": checkpoint["task_id"],
                "dataset_fingerprint": checkpoint["dataset_fingerprint"],
                "split_digest": checkpoint["split_digest"],
                "statistics_fingerprint": checkpoint["statistics_fingerprint"],
                "architecture_fingerprint": checkpoint["architecture_fingerprint"],
                "git_commit": checkpoint["git_commit"],
            },
            "aggregate": {"episode_count": 12},
            "episodes": [],
        }

    comparison = {
        "schema_version": "langmani-m42-development-comparison-v0",
        "experiment_manifest_fingerprint": runtime_raw["experiment_manifest_fingerprint"],
        "schedule_id": "m42_dev_v0",
        "schedule_fingerprint": runtime_raw["development_schedule_fingerprint"],
        "execution_horizon": runtime_raw["execution_horizon"],
        "gripper_mode": runtime_raw["gripper_mode"],
        "selected_task_token_checkpoint_fingerprint": (
            token_checkpoints[-1].checkpoint_fingerprint
        ),
        "selection_fingerprint": selection.fingerprint,
        "models": {
            "state_onehot": benchmark(onehot, "state_onehot"),
            "task_token": benchmark(
                {
                    "run_fingerprint": token_identity.run_fingerprint,
                    "checkpoint_fingerprint": token_checkpoints[-1].checkpoint_fingerprint,
                    "checkpoint_relative_path": token_checkpoints[-1].relative_path,
                    "dataset_fingerprint": dataset_fingerprint,
                    "split_digest": split_digest,
                    "statistics_fingerprint": token_statistics.statistics_fingerprint,
                    "architecture_fingerprint": (
                        token_identity.task_token_architecture_fingerprint
                    ),
                    "git_commit": token_identity.git_commit,
                    "task_id": None,
                },
                "task_token",
            ),
            "per_task": {
                "benchmarks": [benchmark(item, "per_task") for item in per_task],
                "aggregate": {"episode_count": 72},
            },
        },
        "paired_comparisons": {},
        "task_sensitivity": {},
        "selection_source": "m3b_validation_only",
        "test_split_accessed": False,
        "final_schedule_accessed": False,
        "complete": True,
    }
    _write(development_root / "development_comparison.json", comparison)
    completion = {
        "schema_version": "langmani-m42-development-complete-v0",
        "experiment_manifest_fingerprint": runtime_raw["experiment_manifest_fingerprint"],
        "development_schedule_fingerprint": runtime_raw["development_schedule_fingerprint"],
        "implementation_fingerprint": _digest(30),
        "evaluation_git_commit": evaluation_git_commit,
        "runtime_selection_fingerprint": runtime_raw["runtime_fingerprint"],
        "runtime_selection_source_fingerprint": runtime_source_fingerprint,
        "task_token_run_fingerprint": token_identity.run_fingerprint,
        "task_token_selection_fingerprint": selection.fingerprint,
        "selected_task_token_checkpoint_fingerprint": (
            token_checkpoints[-1].checkpoint_fingerprint
        ),
        "development_comparison_fingerprint": canonical_fingerprint(comparison),
        "final_schedule_accessed": False,
        "passed": True,
    }
    _write(development_root / "development_complete.json", completion)
    return {
        "dataset_root": dataset_root,
        "m4_root": m4_root,
        "token_root": token_root,
        "m42_root": m42_root,
        "runtime_path": runtime_path,
        "comparison_path": development_root / "development_comparison.json",
        "completion_path": development_root / "development_complete.json",
        "selection_path": development_root / "task_token_checkpoint_selection.json",
        "queue_path": token_run_root / "validation_queue.json",
        "validation_results": validation_results,
        "validation_evidence_root": development_root / "evidence",
        "dataset_fingerprint": dataset_fingerprint,
        "split_digest": split_digest,
        "runtime_evaluation_git_commit": runtime_evaluation_git_commit,
        "task_token_evaluation_git_commit": evaluation_git_commit,
        "action_std": action_std,
        "m4_runs": per_task + [onehot],
        "token_run_root": token_run_root,
    }


def _load(paths: dict[str, Any], *, mode: str = "combined") -> object:
    return load_restricted_audit_inputs(
        mode=mode,
        dataset_root=paths["dataset_root"],
        m4_checkpoint_root=paths["m4_root"],
        task_token_checkpoint_root=paths["token_root"],
        m42_diagnostics_root=paths["m42_root"],
        runtime_selection_path=paths["runtime_path"],
    )


def test_restricted_loader_opens_only_allowlisted_validation_and_development_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    forbidden_files = []
    for run in paths["m4_runs"]:
        for relative in ("test/benchmark.json", "fresh_seed/benchmark.json"):
            path = run["run_root"] / relative
            _write(path, {"must_not_open": True})
            forbidden_files.append(path.resolve())
    final_path = paths["m42_root"] / "m42_final_v0" / "comparison.json"
    _write(final_path, {"must_not_open": True})
    forbidden_files.append(final_path.resolve())

    opened: list[Path] = []
    original = __import__("os").open

    def traced_open(path: object, *args: object, **kwargs: object) -> object:
        resolved = Path(path).resolve()
        normalized = tuple(part.casefold().replace("-", "_") for part in resolved.parts)
        if any(part in {"test", "fresh_seed", "m42_final_v0"} for part in normalized):
            raise AssertionError(f"forbidden audit path was opened: {resolved}")
        opened.append(resolved)
        return original(path, *args, **kwargs)

    monkeypatch.setattr("os.open", traced_open)
    result = _load(paths)
    assert len(opened) == 63
    assert not set(opened) & set(forbidden_files)
    assert len(result.access_trace) == 63
    assert Counter(item.authority for item in result.access_trace) == Counter(
        {
            "m3b_dataset": 1,
            "m4_checkpoints": 15,
            "task_token_checkpoints": 3,
            "m42_checkpoint_provenance": 42,
            "m42_development_rollout": 2,
        }
    )
    assert result.train_action_std == paths["action_std"]
    assert result.development.task_token_m42_status == "rejected"
    portable = result.portable_dict()
    encoded = json.dumps(portable, sort_keys=True)
    assert str(tmp_path.resolve()) not in encoded
    assert portable["test_split_accessed"] is False
    assert portable["fresh_seed_accessed"] is False
    assert portable["final_schedule_accessed"] is False


def test_validation_mode_does_not_read_development_rollout_evidence(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["comparison_path"].write_text("not-json", encoding="utf-8")
    paths["completion_path"].write_text("not-json", encoding="utf-8")

    result = _load(paths, mode="validation")

    assert result.audit_mode == "validation"
    assert result.development is None
    assert result.runtime.runtime_fingerprint == _digest(22)
    assert result.task_token_selection.selected_checkpoint_fingerprint == (
        result.task_token_checkpoint.checkpoint_fingerprint
    )
    assert "m42_development_rollout" not in {item.authority for item in result.access_trace}


def test_task_token_selection_is_recomputed_from_all_validation_results(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    artifact = json.loads(paths["selection_path"].read_text(encoding="utf-8"))
    record = artifact["selection"]
    record["selected_value"] = record["candidate_values"][-1]
    record["selected_checkpoint_fingerprint"] = record["candidate_values"][-1]
    artifact["selection_fingerprint"] = canonical_fingerprint(record)
    _write(paths["selection_path"], artifact)

    with pytest.raises(M43AuditInputError, match="reproduce all 20"):
        _load(paths, mode="validation")


def test_development_descriptor_binds_full_checkpoint_provenance(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    comparison = json.loads(paths["comparison_path"].read_text(encoding="utf-8"))
    comparison["models"]["state_onehot"]["checkpoint"]["split_digest"] = _digest(999)
    _write(paths["comparison_path"], comparison)
    completion = json.loads(paths["completion_path"].read_text(encoding="utf-8"))
    completion["development_comparison_fingerprint"] = canonical_fingerprint(comparison)
    _write(paths["completion_path"], completion)

    with pytest.raises(M43AuditInputError, match="different checkpoints/tasks"):
        _load(paths)


def test_authorized_reader_detects_path_replacement_between_check_and_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    target = paths["dataset_root"] / "langmani" / "complete.json"
    replacement = target.with_name("replacement.json")
    _write(
        replacement,
        {"schema_version": COMPLETION_SCHEMA, "export_fingerprint": _digest(999)},
    )
    backup = target.with_name("original.json")
    original_open = __import__("os").open
    swapped = False

    def replacing_open(path: object, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        if not swapped and Path(path).resolve() == target.resolve():
            swapped = True
            target.rename(backup)
            replacement.rename(target)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr("os.open", replacing_open)
    with pytest.raises(M43AuditInputError, match="changed during its authorized read"):
        _load(paths, mode="validation")


def test_deployment_parent_named_test_is_not_an_evidence_identity(tmp_path: Path) -> None:
    paths = _fixture(tmp_path / "test" / "deployment")
    result = _load(paths, mode="validation")
    assert result.dataset_fingerprint == _digest(1)


def test_loader_never_calls_the_broad_prior_m4_evidence_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    monkeypatch.setattr(
        m42_evidence,
        "load_prior_m4_evidence",
        lambda **_kwargs: pytest.fail("restricted loader called broad M4 evidence gate"),
    )
    result = _load(paths)
    assert len(result.per_task_checkpoints) == 6
    assert result.state_onehot_checkpoint.policy_kind.value == "state_onehot"
    assert result.task_token_checkpoint.policy_kind.value == "task_token"


def test_loader_ignores_valid_historical_non_full_and_incomplete_m4_runs(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    historical = (
        (
            100,
            ActVariant.PER_TASK,
            CANONICAL_TASK_IDS[0],
            ExperimentMode.TINY_OVERFIT,
            True,
        ),
        (101, ActVariant.MIXED_TASK_ONEHOT, None, ExperimentMode.TINY_OVERFIT, False),
        (102, ActVariant.PER_TASK, CANONICAL_TASK_IDS[1], ExperimentMode.FULL, False),
    )
    for run_index, variant, task_id, mode, complete in historical:
        _m4_run(
            paths["m4_root"],
            run_index=run_index,
            variant=variant,
            task_id=task_id,
            dataset_fingerprint=paths["dataset_fingerprint"],
            split_digest=paths["split_digest"],
            action_std=paths["action_std"],
            mode=mode,
            complete=complete,
        )

    result = _load(paths, mode="validation")

    assert len(result.per_task_checkpoints) == 6
    assert result.state_onehot_checkpoint.policy_kind.value == "state_onehot"


def test_loader_rejects_duplicate_completed_full_m4_semantic_run(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _m4_run(
        paths["m4_root"],
        run_index=103,
        variant=ActVariant.PER_TASK,
        task_id=CANONICAL_TASK_IDS[0],
        dataset_fingerprint=_digest(901),
        split_digest=paths["split_digest"],
        action_std=paths["action_std"],
    )

    with pytest.raises(M43AuditInputError, match="duplicate M4 semantic run"):
        _load(paths, mode="validation")


def test_loader_rejects_dirty_completed_full_m4_candidate(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    candidate = _m4_run(
        paths["m4_root"],
        run_index=104,
        variant=ActVariant.PER_TASK,
        task_id=CANONICAL_TASK_IDS[0],
        dataset_fingerprint=paths["dataset_fingerprint"],
        split_digest=paths["split_digest"],
        action_std=paths["action_std"],
    )
    manifest_path = candidate["run_root"] / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["git_dirty"] = True
    manifest["identity"]["git_dirty"] = True
    _write(manifest_path, manifest)

    with pytest.raises(M43AuditInputError, match="M4 run manifest is invalid"):
        _load(paths, mode="validation")


def test_loader_binds_validation_selection_to_completed_full_run(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    run = paths["m4_runs"][0]
    candidate = ValidationResult(
        checkpoint_fingerprint=run["checkpoint_fingerprint"],
        checkpoint_step=100_000,
        schedule_digest=_digest(902),
        success_rate=0.5,
        wrong_object_interaction_rate=0.0,
        target_off_table_rate=0.0,
        offline_validation_action_loss=1.0,
    )
    foreign_selection = create_checkpoint_selection(
        run_fingerprint=_digest(903),
        candidates=(candidate,),
        selection_timestamp_utc="2026-07-15T00:00:00Z",
    )
    _write(run["run_root"] / "checkpoint_selection.json", foreign_selection.to_dict())

    with pytest.raises(M43AuditInputError, match="immutable validation-only lock"):
        _load(paths, mode="validation")


def test_task_token_validation_uses_its_producer_lineage_not_runtime_lineage(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)

    assert paths["runtime_evaluation_git_commit"] != paths["task_token_evaluation_git_commit"]
    result = _load(paths, mode="validation")

    assert result.task_token_checkpoint.git_commit == paths["task_token_evaluation_git_commit"]


def test_task_token_validation_queue_rejects_a_different_producer_commit(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    queue = json.loads(paths["queue_path"].read_text(encoding="utf-8"))
    queue["git_commit"] = "c" * 40
    queue["queue_fingerprint"] = ""
    _write(paths["queue_path"], queue)

    with pytest.raises(
        M43AuditInputError,
        match="validation queue differs from its complete training identity",
    ):
        _load(paths, mode="validation")


def test_task_token_validation_artifact_rejects_a_different_producer_commit(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    artifact_path = next(paths["validation_evidence_root"].glob("*/artifact.json"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["identity"]["evaluation_git_commit"] = paths["runtime_evaluation_git_commit"]
    _write(artifact_path, artifact)

    with pytest.raises(
        M43AuditInputError,
        match="validation artifact differs from its immutable queue/identity",
    ):
        _load(paths, mode="validation")


def test_state_onehot_and_task_token_action_std_must_match(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, task_token_std=(0.2,) * 8)
    with pytest.raises(M43AuditInputError, match="action train statistics must match"):
        _load(paths)


def test_development_payload_rejects_final_identity_before_completion(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    comparison = json.loads(paths["comparison_path"].read_text(encoding="utf-8"))
    comparison["source_identity"] = "m42_final_v0"
    _write(paths["comparison_path"], comparison)
    with pytest.raises(M43AuditInputError, match="prohibited evidence identity"):
        _load(paths)


def test_explicit_runtime_path_cannot_point_into_final_area(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    forbidden = paths["m42_root"] / "m42_final_v0" / "runtime_selection.json"
    _write(forbidden, {})
    with pytest.raises(M43AuditInputError, match="runtime selection must be"):
        load_restricted_audit_inputs(
            dataset_root=paths["dataset_root"],
            m4_checkpoint_root=paths["m4_root"],
            task_token_checkpoint_root=paths["token_root"],
            m42_diagnostics_root=paths["m42_root"],
            runtime_selection_path=forbidden,
        )


def test_validation_schedule_is_balanced_and_canonical(tmp_path: Path) -> None:
    result = _load(_fixture(tmp_path))
    assert tuple(item.episode_index for item in result.validation_episodes) == tuple(
        range(288, 324)
    )
    assert Counter(item.task_id for item in result.validation_episodes) == Counter(
        {task_id: 6 for task_id in CANONICAL_TASK_IDS}
    )
    assert all(
        item.task_id == stable_task_id(item.task_spec) for item in result.validation_episodes
    )
