"""Command-boundary tests for M4 training, evaluation, comparison, and verification."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from langmani.datasets.lerobot_types import IMAGE_FEATURE_KEY, STATE_FEATURE_KEY, DatasetSplit
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import stable_scene_id
from langmani.policies.act_action_bounds import (
    ActionBoundConfig,
    EvaluationRuntimeIdentity,
    EvaluationRuntimeManifest,
)
from langmani.policies.act_analysis import compute_counterfactual_sensitivity, task_identity_effects
from langmani.policies.act_conditioning import CANONICAL_TASK_IDS
from langmani.policies.act_data import DatasetEpisodeView
from langmani.policies.act_evaluation import (
    authorize_test_evaluation,
    create_checkpoint_selection,
    rollout_schedule_digest,
    summarize_rollout_benchmark,
    write_checkpoint_selection_atomic,
)
from langmani.policies.act_types import (
    TASK_ONEHOT_MAPPING_VERSION,
    ActDataConfig,
    ActExperimentConfig,
    ActExperimentManifest,
    ActModelConfig,
    ActOptimizationConfig,
    ActRunIdentity,
    ActVariant,
    CheckpointRecord,
    EvaluationSplit,
    ExperimentMode,
    RolloutEpisodeResult,
    RolloutStatus,
    TrainingState,
    ValidationResult,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _runtime_manifest(
    *, checkpoint_fingerprint: str, split: EvaluationSplit
) -> EvaluationRuntimeManifest:
    identity = EvaluationRuntimeIdentity(
        checkpoint_fingerprint=checkpoint_fingerprint,
        policy_preprocessor_fingerprint="sha256:" + "a" * 64,
        policy_postprocessor_fingerprint="sha256:" + "b" * 64,
        action_bound_config=ActionBoundConfig(),
        environment_id="LangMani-PickPlaceByInstruction-v0",
        action_space_contract={
            "shape": [8],
            "dtype": "float32",
            "low": [-1.0] * 8,
            "high": [1.0] * 8,
        },
        task_conditioning_mapping_version=TASK_ONEHOT_MAPPING_VERSION,
        rollout_config={"split": split.value, "sim_backend": "fixture"},
        code_git_commit="1" * 40,
    )
    return EvaluationRuntimeManifest(
        identity=identity,
        checkpoint_model_reload_validated=True,
        policy_processor_reload_validated=True,
        action_bound_processor_reload_validated=True,
        deterministic_raw_action_matched=True,
        raw_action_match_tolerance=1e-6,
    )


def _load(name: str, relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


train_cli = _load("langmani_test_train_act", "scripts/train_act.py")
evaluate_cli = _load("langmani_test_evaluate_act", "scripts/evaluate_act.py")
compare_cli = _load("langmani_test_compare_act", "scripts/compare_act_baselines.py")
verify_m4 = _load("langmani_test_verify_m4", "environment/verify_m4.py")


def test_train_cli_requires_one_explicit_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_act.py", "--variant", ActVariant.MIXED_UNCONDITIONED.value],
    )
    with pytest.raises(SystemExit) as error:
        train_cli.parse_args()
    assert error.value.code == 2


def test_train_dry_run_can_plan_the_exact_full_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_act.py",
            "--variant",
            ActVariant.MIXED_TASK_ONEHOT.value,
            "--dry-run",
            "--planned-mode",
            ExperimentMode.FULL.value,
        ],
    )
    args = train_cli.parse_args()
    config, task_id = train_cli._configs(args)
    assert config.mode is ExperimentMode.FULL
    assert config.optimization == ActOptimizationConfig()
    assert task_id is None

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_act.py",
            "--variant",
            ActVariant.MIXED_UNCONDITIONED.value,
            "--full",
            "--planned-mode",
            ExperimentMode.FULL.value,
        ],
    )
    with pytest.raises(ValueError, match="valid only with --dry-run"):
        train_cli._configs(train_cli.parse_args())


def test_train_cli_resolves_semantic_alias_without_language_parsing() -> None:
    assert train_cli._task_id("red_cube__left_bin") == CANONICAL_TASK_IDS[0]
    assert train_cli._task_id(CANONICAL_TASK_IDS[5]) == CANONICAL_TASK_IDS[5]
    with pytest.raises(ValueError, match="unknown task ID"):
        train_cli._task_id("pick up the red cube")


def test_tiny_overfit_loss_view_is_not_mislabeled_as_held_out_validation() -> None:
    task_id = CANONICAL_TASK_IDS[0]
    config = ActExperimentConfig(
        variant=ActVariant.PER_TASK,
        data=ActDataConfig(dataset_root="fixture"),
        model=ActModelConfig.for_variant(ActVariant.PER_TASK),
        task_id=task_id,
        mode=ExperimentMode.TINY_OVERFIT,
    )
    train_view = DatasetEpisodeView(
        split=DatasetSplit.TRAIN,
        episode_indices=(0,),
        task_id=task_id,
    )
    formal_validation, contract = train_cli._offline_loss_identity_contract(
        config=config,
        train_view=train_view,
        offline_loss_view=train_view,
    )
    assert formal_validation == ()
    assert contract == {
        "role": "train_tiny_overfit_diagnostic",
        "episode_indices": [0],
    }


def test_full_run_rejects_any_train_validation_overlap() -> None:
    task_id = CANONICAL_TASK_IDS[0]
    config = ActExperimentConfig(
        variant=ActVariant.PER_TASK,
        data=ActDataConfig(dataset_root="fixture"),
        model=ActModelConfig.for_variant(ActVariant.PER_TASK),
        task_id=task_id,
        mode=ExperimentMode.FULL,
    )
    train_view = DatasetEpisodeView(
        split=DatasetSplit.TRAIN,
        episode_indices=(0, 1),
        task_id=task_id,
    )
    overlapping_view = DatasetEpisodeView(
        split=DatasetSplit.VALIDATION,
        episode_indices=(1, 2),
        task_id=task_id,
    )
    with pytest.raises(RuntimeError, match="offline loss may overlap training"):
        train_cli._offline_loss_identity_contract(
            config=config,
            train_view=train_view,
            offline_loss_view=overlapping_view,
        )


def test_resume_recovers_checkpoint_bound_metric_after_process_interruption(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.jsonl"
    recovery = {
        "step": 5,
        "total_loss": 0.25,
        "checkpoint_path": "checkpoints/step-00000005-fixture",
    }
    train_cli._truncate_metrics_to_checkpoint(metrics, 5, recovery_record=recovery)
    assert json.loads(metrics.read_text(encoding="utf-8")) == recovery
    metrics.write_text(
        json.dumps({**recovery, "checkpoint_path": None}) + "\n",
        encoding="utf-8",
    )
    train_cli._truncate_metrics_to_checkpoint(metrics, 5, recovery_record=recovery)
    assert json.loads(metrics.read_text(encoding="utf-8")) == recovery
    tampered = {**recovery, "total_loss": 0.1}
    metrics.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checkpoint-bound evidence"):
        train_cli._truncate_metrics_to_checkpoint(metrics, 5, recovery_record=recovery)


def test_validation_selection_uses_checkpoint_bound_metric(tmp_path: Path) -> None:
    relative = "checkpoints/step-00000005-fixture"
    bound = {"step": 5, "validation_loss": 0.25, "checkpoint_path": None}
    external = {**bound, "checkpoint_path": relative}
    (tmp_path / "metrics.jsonl").write_text(json.dumps(external) + "\n", encoding="utf-8")
    assert evaluate_cli._offline_validation_loss(
        tmp_path,
        5,
        checkpoint_metric=bound,
        checkpoint_relative_path=relative,
    ) == pytest.approx(0.25)
    external["validation_loss"] = 0.0
    (tmp_path / "metrics.jsonl").write_text(json.dumps(external) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="offline validation loss"):
        evaluate_cli._offline_validation_loss(
            tmp_path,
            5,
            checkpoint_metric=bound,
            checkpoint_relative_path=relative,
        )


def test_resume_adopts_only_the_sole_promoted_manifest_orphan(tmp_path: Path) -> None:
    declared = "checkpoints/step-00000005-declared"
    orphan = "checkpoints/step-00000010-orphan"
    for relative in (declared, orphan):
        directory = tmp_path / relative
        directory.mkdir(parents=True)
        (directory / train_cli.CHECKPOINT_COMPLETION_MARKER).write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="newer promoted checkpoint"):
        train_cli._resume_checkpoint_is_recoverable_orphan(
            tmp_path,
            requested_relative_path=declared,
            declared_relative_paths=(declared,),
        )
    assert train_cli._resume_checkpoint_is_recoverable_orphan(
        tmp_path,
        requested_relative_path=orphan,
        declared_relative_paths=(declared,),
    )


def test_verify_report_exposes_independent_flags_and_structural_is_not_physical(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "verification.json"
    monkeypatch.setattr(verify_m4, "REPORT_PATH", path)
    report = verify_m4.Report(
        implementation_validated=True,
        git_baseline_validated=True,
        fixture_training_validated=True,
        checkpoint_reload_validated=True,
        train_stats_leakage_validated=True,
        validation_selection_validated=True,
        test_lock_validated=True,
    )
    report.write(mode="structural", dataset_root=tmp_path / "dataset")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "implementation_validated",
        "git_baseline_validated",
        "source_dataset_validated",
        "fixture_training_validated",
        "cuda_training_validated",
        "tiny_overfit_validated",
        "checkpoint_reload_validated",
        "checkpoint_model_reload_validated",
        "policy_processor_reload_validated",
        "action_bound_processor_reload_validated",
        "raw_action_bounds_validated",
        "projected_action_bounds_validated",
        "closed_loop_inference_validated",
        "strict_unprojected_rollout_validated",
        "tiny_overfit_task_success_validated",
        "train_stats_leakage_validated",
        "validation_selection_validated",
        "test_lock_validated",
        "per_task_experiment_completed",
        "mixed_unconditioned_experiment_completed",
        "mixed_task_onehot_experiment_completed",
        "fresh_seed_benchmark_completed",
        "full_dry_run_validated",
        "planned_full_run_count",
        "full_experiment_validated",
        "baseline_quality_validated",
        "physical_target_validated",
    }
    assert expected <= payload.keys()
    assert payload["passed"] is True
    assert payload["physical_target_validated"] is False

    dry_report = verify_m4.Report(
        implementation_validated=True,
        git_baseline_validated=True,
        source_dataset_validated=True,
        fixture_training_validated=True,
        checkpoint_reload_validated=True,
        train_stats_leakage_validated=True,
        validation_selection_validated=True,
        test_lock_validated=True,
        full_dry_run_validated=True,
        planned_full_run_count=8,
    )
    assert dry_report.accepted("target_full", dry_run=True)
    assert not dry_report.accepted("target_full")


def _benchmark_payload(
    *,
    run_fingerprint: str,
    checkpoint_fingerprint: str,
    variant: ActVariant,
    task_id: str | None,
    split: EvaluationSplit,
) -> dict[str, object]:
    runtime_manifest = _runtime_manifest(
        checkpoint_fingerprint=checkpoint_fingerprint,
        split=split,
    )
    task_ids = (task_id,) if variant is ActVariant.PER_TASK else CANONICAL_TASK_IDS
    scene_count = 30 if split is EvaluationSplit.FRESH_SEED else 6
    task_specs = dict(zip(CANONICAL_TASK_IDS, CANONICAL_TASK_SPECS, strict=True))
    schedule = tuple(
        {
            "scene_seed": scene_seed,
            "task_spec": task_specs[current_task].to_dict(),
        }
        for scene_seed in range(scene_count)
        for current_task in task_ids
    )
    schedule_digest = rollout_schedule_digest(
        m3b_export_fingerprint="sha256:" + "d" * 64,
        variant=variant.value,
        task_id=task_id,
        split=split.value,
        episodes=schedule,
    )
    episodes = tuple(
        RolloutEpisodeResult(
            evaluation_id=f"{split.value}:{index:04d}",
            run_fingerprint=run_fingerprint,
            checkpoint_fingerprint=checkpoint_fingerprint,
            schedule_digest=schedule_digest,
            split=split,
            scene_seed=int(item["scene_seed"]),
            scene_id=stable_scene_id(int(item["scene_seed"])),
            task_id=current_task,
            status=RolloutStatus.TIMEOUT,
            success=False,
            episode_steps=1,
            final_evaluation={
                "success": False,
                "target_in_target_bin": False,
                "target_in_wrong_bin": False,
                "wrong_object_in_target_bin": False,
                "target_off_table": False,
                "fail": False,
            },
            target_in_target_bin=False,
            target_in_wrong_bin=False,
            wrong_object_in_target_bin=False,
            target_grasped_any=False,
            wrong_object_grasped_any=False,
            target_off_table=False,
            timeout=True,
            invalid_action=False,
            inference_failure=False,
            time_to_first_target_grasp_s=None,
            time_to_release_s=None,
            time_to_success_s=None,
            cumulative_action_magnitude=0.0,
            action_min=(0.0,) * 8,
            action_max=(0.0,) * 8,
            inference_latency_ms=(1.0,),
            environment_step_latency_ms=(1.0,),
            total_episode_duration_s=0.1,
            failure_reason="fixture timeout",
            runtime_fingerprint=runtime_manifest.identity.runtime_fingerprint,
        )
        for index, (item, current_task) in enumerate(
            zip(schedule, task_ids * scene_count, strict=True)
        )
    )
    benchmark = summarize_rollout_benchmark(episodes)
    return {
        **benchmark.to_dict(),
        "schema_version": "langmani-m4.1-rollout-benchmark-v2",
        "passed": True,
        "quality_validated": False,
        "infrastructure_failure_count": 0,
        "environment_action_applied": True,
        "evaluation_runtime_fingerprint": (runtime_manifest.identity.runtime_fingerprint),
        "action_projection_artifact": {
            "relative_path": evaluate_cli.ACTION_PROJECTION_RECORDS_FILE,
            "sha256": hashlib.sha256(b"").hexdigest(),
            "record_count": 0,
        },
        "evaluation_git": {
            "commit": "1" * 40,
            "dirty": False,
            "changed_paths": [],
            "baseline_tracked": True,
        },
    }


def _write_run(
    root: Path,
    *,
    variant: ActVariant,
    run_character: str,
    task_id: str | None,
) -> Path:
    selected = "sha256:" + "e" * 64
    statistics = "sha256:" + "5" * 64
    split_digest = "sha256:" + "6" * 64
    optimization = ActOptimizationConfig()
    schedule_digests = {
        split.value: _benchmark_payload(
            run_fingerprint="sha256:" + "0" * 64,
            checkpoint_fingerprint="sha256:" + "0" * 64,
            variant=variant,
            task_id=task_id,
            split=split,
        )["schedule_digest"]
        for split in (
            EvaluationSplit.VALIDATION,
            EvaluationSplit.TEST,
            EvaluationSplit.FRESH_SEED,
        )
    }
    config = ActExperimentConfig(
        variant=variant,
        data=ActDataConfig(dataset_root="portable-fixture"),
        model=ActModelConfig.for_variant(variant),
        optimization=optimization,
        task_id=task_id,
        mode=ExperimentMode.FULL,
        device="cuda",
    )
    identity = ActRunIdentity(
        m3b_export_fingerprint="sha256:" + "d" * 64,
        m3b_split_manifest_digest=split_digest,
        ordered_train_episode_indices=(0,),
        ordered_validation_episode_indices=(1,),
        variant=variant,
        task_id=task_id,
        task_onehot_mapping_version=TASK_ONEHOT_MAPPING_VERSION,
        model_config=config.model.to_dict(),
        data_contract={
            "fixture": run_character,
            "evaluation_schedules": {"digests": schedule_digests},
        },
        train_statistics_fingerprint=statistics,
        optimization_config=optimization.to_dict(),
        training_seed=0,
        experiment_mode=config.mode,
        device=config.device,
        dtype=config.dtype,
        lerobot_version="0.6.0",
        torch_version="2.11.0+cu128",
        cuda_version="12.8",
        git_commit="1" * 40,
        git_dirty=False,
        dirty_development_override=False,
    )
    run_fingerprint = identity.run_fingerprint
    checkpoint_fingerprints = tuple("sha256:" + f"{index:064x}" for index in range(1, 20)) + (
        selected,
    )
    checkpoints = tuple(
        CheckpointRecord(
            checkpoint_fingerprint=fingerprint,
            run_fingerprint=run_fingerprint,
            global_step=index * 5_000,
            relative_path=(
                f"checkpoints/step-{index * 5_000:08d}-{fingerprint.removeprefix('sha256:')[:12]}"
            ),
            policy_config_digest="sha256:" + "7" * 64,
            statistics_fingerprint=statistics,
            split_digest=split_digest,
            git_commit=identity.git_commit,
            complete=True,
        )
        for index, fingerprint in enumerate(checkpoint_fingerprints, start=1)
    )
    manifest = ActExperimentManifest(
        identity=identity,
        config=config,
        git_dirty=False,
        dirty_development_override=False,
        train_statistics_fingerprint=statistics,
        training_state=TrainingState(
            global_step=100_000,
            examples_processed=3_200_000,
            last_checkpoint_fingerprint=selected,
            completed=True,
        ),
        checkpoints=checkpoints,
        selected_checkpoint_fingerprint=selected,
        complete=True,
    )
    root.mkdir(parents=True)
    manifest_path = root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    validation_digest = _benchmark_payload(
        run_fingerprint=run_fingerprint,
        checkpoint_fingerprint=selected,
        variant=variant,
        task_id=task_id,
        split=EvaluationSplit.VALIDATION,
    )["schedule_digest"]
    candidates = tuple(
        ValidationResult(
            checkpoint_fingerprint=checkpoint.checkpoint_fingerprint,
            checkpoint_step=checkpoint.global_step,
            schedule_digest=str(validation_digest),
            success_rate=index / len(checkpoints),
            wrong_object_interaction_rate=0.0,
            target_off_table_rate=0.0,
            offline_validation_action_loss=1.0,
        )
        for index, checkpoint in enumerate(checkpoints, start=1)
    )
    selection = create_checkpoint_selection(
        run_fingerprint=run_fingerprint,
        candidates=candidates,
    )
    write_checkpoint_selection_atomic(root / "checkpoint_selection.json", selection)
    required = [
        f"validation/{selected.removeprefix('sha256:')}/benchmark.json",
        "test/benchmark.json",
        "fresh_seed/benchmark.json",
    ]
    for relative in required:
        split = (
            EvaluationSplit.VALIDATION
            if relative.startswith("validation/")
            else EvaluationSplit(relative.split("/", maxsplit=1)[0])
        )
        benchmark = _benchmark_payload(
            run_fingerprint=run_fingerprint,
            checkpoint_fingerprint=selected,
            variant=variant,
            task_id=task_id,
            split=split,
        )
        authorization = (
            authorize_test_evaluation(
                run_fingerprint=run_fingerprint,
                checkpoint_fingerprint=selected,
                actual_schedule_digest=str(benchmark["schedule_digest"]),
                expected_schedule_digest=str(benchmark["schedule_digest"]),
                mode=ExperimentMode.FULL,
                selection=selection,
            )
            if split in {EvaluationSplit.TEST, EvaluationSplit.FRESH_SEED}
            else None
        )
        benchmark["test_authorization"] = (
            authorization.to_dict() if authorization is not None else None
        )
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(benchmark), encoding="utf-8")
        runtime_manifest = _runtime_manifest(
            checkpoint_fingerprint=selected,
            split=split,
        )
        (path.parent / evaluate_cli.EVALUATION_RUNTIME_MANIFEST_FILE).write_text(
            json.dumps(runtime_manifest.to_dict()), encoding="utf-8"
        )
        (path.parent / evaluate_cli.ACTION_PROJECTION_RECORDS_FILE).write_text("", encoding="utf-8")
    report_path = (
        root
        / "reports"
        / (
            "per_task_reference.json"
            if variant is ActVariant.PER_TASK
            else "counterfactual_sensitivity.json"
        )
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if variant is ActVariant.PER_TASK:
        task_index = CANONICAL_TASK_IDS.index(task_id)
        actions = np.full((2, 8), float(task_index), dtype=np.float32)
        analysis_payload = {
            "schema_version": "langmani-m4-per-task-reference-v1",
            "passed": True,
            "run_fingerprint": run_fingerprint,
            "checkpoint_fingerprint": selected,
            "evaluation_git": {
                "commit": "1" * 40,
                "dirty": False,
                "changed_paths": [],
                "baseline_tracked": True,
            },
            "task_id": task_id,
            "scene_seed": 7,
            "scene_id": "scene:7",
            "initial_rgb_digest": "8" * 64,
            "expert_decoded_rgb_digest": "9" * 64,
            "panda_state_equivalent": True,
            "initial_panda_state": [[0.0] * 9],
            "first_action": actions[0].tolist(),
            "action_chunk": actions.tolist(),
            "expert_chunk_distance": float(task_index),
        }
    else:
        actions = (
            np.zeros((6, 8), dtype=np.float32)
            if variant is ActVariant.MIXED_UNCONDITIONED
            else np.stack([np.full(8, float(index), dtype=np.float32) for index in range(6)])
        )
        chunks = np.repeat(actions[:, None, :], 2, axis=1)
        policy_inputs = []
        for index in range(6):
            state = np.zeros(15 if variant is ActVariant.MIXED_TASK_ONEHOT else 9, np.float32)
            if variant is ActVariant.MIXED_TASK_ONEHOT:
                state[9 + index] = 1.0
            policy_inputs.append(
                {
                    IMAGE_FEATURE_KEY: np.zeros((3, 1, 1), dtype=np.uint8),
                    STATE_FEATURE_KEY: state,
                }
            )
        sensitivity = compute_counterfactual_sensitivity(
            variant=variant,
            scene_id="scene:7",
            ordered_task_ids=CANONICAL_TASK_IDS,
            first_actions=actions,
            action_chunks=chunks,
            policy_inputs=policy_inputs,
        )
        analysis_payload = {
            **sensitivity.to_dict(),
            **task_identity_effects(sensitivity.pairwise_chunk_distances),
            "expert_chunk_distances_by_task": {
                task: float(index) for index, task in enumerate(CANONICAL_TASK_IDS)
            },
            "expert_reference_evidence": {
                "panda_state_equivalent": True,
                "reset_rgb_digest": "8" * 64,
                "expert_decoded_rgb_digests": ["9" * 64 for _ in range(6)],
                "rgb_comparison_note": "fixture",
            },
            "passed": True,
            "run_fingerprint": run_fingerprint,
            "checkpoint_fingerprint": selected,
            "evaluation_git": {
                "commit": "1" * 40,
                "dirty": False,
                "changed_paths": [],
                "baseline_tracked": True,
            },
        }
    report_path.write_text(json.dumps(analysis_payload), encoding="utf-8")
    summary_path = root / "reports" / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "passed": True,
                "run_fingerprint": run_fingerprint,
                "variant": variant.value,
                "task_id": task_id,
                "parameter_count": 1,
                "training_duration_s": 1.0,
                "training_duration_complete": True,
                "measured_dataloader_and_optimizer_duration_s": 0.5,
                "initial_train_loss": 1.0,
                "final_train_loss": 0.5,
                "initial_validation_loss": 1.0,
                "final_validation_loss": 0.5,
                "peak_gpu_allocated_bytes": 1,
                "peak_gpu_reserved_bytes": 1,
                "mean_throughput_examples_per_s": 1.0,
                "effective_model_config": config.model.to_dict(),
                "effective_optimization_config": config.optimization.to_dict(),
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_run_finalization_is_idempotent_and_rejects_tampered_episode_evidence(
    tmp_path: Path,
) -> None:
    manifest_path = _write_run(
        tmp_path / "run",
        variant=ActVariant.PER_TASK,
        run_character="idempotent",
        task_id=CANONICAL_TASK_IDS[0],
    )
    manifest = ActExperimentManifest.from_dict(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    assert evaluate_cli._maybe_finalize_run(
        manifest_path.parent,
        manifest.identity,
        mode=ExperimentMode.FULL,
    )
    assert evaluate_cli._maybe_finalize_run(
        manifest_path.parent,
        manifest.identity,
        mode=ExperimentMode.FULL,
    )
    benchmark_path = manifest_path.parent / "test" / "benchmark.json"
    original_benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    benchmark = dict(original_benchmark)
    benchmark["episodes"] = benchmark["episodes"][:-1]
    benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
    with pytest.raises(RuntimeError, match="episode contract"):
        evaluate_cli._maybe_finalize_run(
            manifest_path.parent,
            manifest.identity,
            mode=ExperimentMode.FULL,
        )
    contradictory = json.loads(json.dumps(original_benchmark))
    contradictory["episodes"][0]["final_evaluation"]["success"] = True
    contradictory["episodes"][0]["final_evaluation"]["target_in_target_bin"] = True
    benchmark_path.write_text(json.dumps(contradictory), encoding="utf-8")
    with pytest.raises(RuntimeError, match="final_evaluation disagrees"):
        evaluate_cli._maybe_finalize_run(
            manifest_path.parent,
            manifest.identity,
            mode=ExperimentMode.FULL,
        )
    contradictory_status = json.loads(json.dumps(original_benchmark))
    contradictory_status["episodes"][0]["invalid_action"] = True
    benchmark_path.write_text(json.dumps(contradictory_status), encoding="utf-8")
    with pytest.raises(RuntimeError, match="final_evaluation disagrees"):
        evaluate_cli._maybe_finalize_run(
            manifest_path.parent,
            manifest.identity,
            mode=ExperimentMode.FULL,
        )


def test_evaluation_rejects_runtime_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = _write_run(
        tmp_path / "run",
        variant=ActVariant.PER_TASK,
        run_character="runtime",
        task_id=CANONICAL_TASK_IDS[0],
    )
    identity = ActExperimentManifest.from_dict(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    ).identity
    monkeypatch.setattr(
        evaluate_cli,
        "runtime_versions",
        lambda: {"lerobot": "0.6.0", "torch": "stale", "cuda": "12.8"},
    )
    with pytest.raises(RuntimeError, match="runtime versions"):
        evaluate_cli._validate_runtime_identity(identity)


def test_target_full_archives_only_matching_precheckpoint_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "models"
    manifest_path = _write_run(
        model_root / "run",
        variant=ActVariant.PER_TASK,
        run_character="precheckpoint",
        task_id=CANONICAL_TASK_IDS[0],
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["checkpoints"] = []
    payload["selected_checkpoint_fingerprint"] = None
    payload["complete"] = False
    payload["training_state"] = TrainingState(global_step=0, examples_processed=0).to_dict()
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    (manifest_path.parent / "reports" / "training_summary.json").unlink()
    monkeypatch.setattr(
        verify_m4,
        "inspect_git_state",
        lambda _root: type("Git", (), {"commit": "1" * 40})(),
    )
    monkeypatch.setattr(
        verify_m4,
        "runtime_versions",
        lambda: {"lerobot": "0.6.0", "torch": "2.11.0+cu128", "cuda": "12.8"},
    )
    command, reused = verify_m4._full_training_command_or_reuse(
        dataset_root=tmp_path / "dataset",
        dataset_fingerprint="sha256:" + "d" * 64,
        model_root=model_root,
        variant=ActVariant.PER_TASK,
        task_id=CANONICAL_TASK_IDS[0],
        report_path=tmp_path / "train.json",
    )
    assert command is not None
    assert reused is None
    assert not manifest_path.parent.exists()
    abandoned = tuple((model_root / "_abandoned_precheckpoint").glob("*/abandoned.json"))
    assert len(abandoned) == 1
    assert json.loads(abandoned[0].read_text(encoding="utf-8"))["reason"] == (
        "interrupted_before_first_promoted_checkpoint"
    )


def test_evaluation_and_full_reuse_reject_linked_output_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    external_validation = tmp_path / "external-validation"
    external_validation.mkdir()
    try:
        (run_root / "validation").symlink_to(external_validation, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")
    with pytest.raises(RuntimeError, match="escapes the run root"):
        evaluate_cli._validate_evaluation_output_path(
            run_root,
            run_root / "validation" / ("a" * 64),
        )

    model_root = tmp_path / "models"
    model_root.mkdir()
    external_run = tmp_path / "external-run"
    _write_run(
        external_run,
        variant=ActVariant.PER_TASK,
        run_character="external",
        task_id=CANONICAL_TASK_IDS[0],
    )
    (model_root / "linked-run").symlink_to(external_run, target_is_directory=True)
    monkeypatch.setattr(
        verify_m4,
        "inspect_git_state",
        lambda _root: type("Git", (), {"commit": "1" * 40})(),
    )
    monkeypatch.setattr(
        verify_m4,
        "runtime_versions",
        lambda: {"lerobot": "0.6.0", "torch": "2.11.0+cu128", "cuda": "12.8"},
    )
    with pytest.raises(RuntimeError, match="escapes the model root"):
        verify_m4._full_training_command_or_reuse(
            dataset_root=tmp_path / "dataset",
            dataset_fingerprint="sha256:" + "d" * 64,
            model_root=model_root,
            variant=ActVariant.PER_TASK,
            task_id=CANONICAL_TASK_IDS[0],
            report_path=tmp_path / "train.json",
        )


def test_comparison_requires_complete_result_evidence_and_does_not_invent_quality(
    tmp_path: Path,
) -> None:
    per_task = [
        _write_run(
            tmp_path / f"per-{index}",
            variant=ActVariant.PER_TASK,
            run_character=str(index + 1),
            task_id=task_id,
        )
        for index, task_id in enumerate(CANONICAL_TASK_IDS)
    ]
    unconditioned = _write_run(
        tmp_path / "unconditioned",
        variant=ActVariant.MIXED_UNCONDITIONED,
        run_character="a",
        task_id=None,
    )
    conditioned = _write_run(
        tmp_path / "conditioned",
        variant=ActVariant.MIXED_TASK_ONEHOT,
        run_character="b",
        task_id=None,
    )
    result = compare_cli.compare(
        argparse.Namespace(
            per_task_manifest=list(reversed(per_task)),
            mixed_unconditioned_manifest=unconditioned,
            mixed_task_onehot_manifest=conditioned,
        )
    )
    assert result["full_experiment_validated"] is True
    assert result["baseline_quality_validated"] is False
    assert result["interpretation"]["language_understanding"] == "not tested in M4"
    assert [item["task_id"] for item in result["runs"][:6]] == list(CANONICAL_TASK_IDS)


def test_verify_target_modes_are_mutually_exclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["verify_m4.py", "--target-smoke", "--target-full"])
    with pytest.raises(SystemExit) as error:
        verify_m4.parse_args()
    assert error.value.code == 2


def test_target_smoke_uses_checkpoint_bound_dataset_after_current_machine_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    current_gate_root = tmp_path / "current-machine-smoke"
    checkpoint_root = tmp_path / "checkpoint-bound-smoke"
    fingerprint = "sha256:" + "a" * 64
    completed = SimpleNamespace(
        summary=SimpleNamespace(total_episodes=6),
        export_fingerprint=fingerprint,
    )
    calls: list[Path] = []

    def fake_load(root: Path, *, require_full: bool, validate_storage: bool) -> object:
        calls.append(root)
        assert require_full is False
        assert validate_storage is True
        return completed

    monkeypatch.setattr(verify_m4, "load_completed_m3b_dataset", fake_load)
    training = {
        verify_m4._CHECKPOINT_DATASET_ROOT_KEY: str(checkpoint_root),
        verify_m4._CHECKPOINT_DATASET_FINGERPRINT_KEY: fingerprint,
    }
    report = verify_m4.Report()
    selected = verify_m4._checkpoint_bound_smoke_dataset(
        report,
        prior_gate_dataset_root=current_gate_root,
        per_task_training=training,
        onehot_training=dict(training),
    )

    assert selected == checkpoint_root.resolve()
    assert calls == [checkpoint_root.resolve()]
    assert report.source_dataset_validated is True
    assert report.checkpoint_bound_dataset_root == str(checkpoint_root.resolve())
    assert report.checkpoint_bound_dataset_fingerprint == fingerprint
    assert str(current_gate_root.resolve()) in report.checks[-1]["detail"]


def test_target_smoke_rejects_different_checkpoint_dataset_identities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        verify_m4,
        "load_completed_m3b_dataset",
        lambda *_args, **_kwargs: pytest.fail("mismatched identities must fail before loading"),
    )
    fingerprint = "sha256:" + "a" * 64
    per_task = {
        verify_m4._CHECKPOINT_DATASET_ROOT_KEY: str(tmp_path / "per-task"),
        verify_m4._CHECKPOINT_DATASET_FINGERPRINT_KEY: fingerprint,
    }
    onehot = {
        verify_m4._CHECKPOINT_DATASET_ROOT_KEY: str(tmp_path / "onehot"),
        verify_m4._CHECKPOINT_DATASET_FINGERPRINT_KEY: fingerprint,
    }
    with pytest.raises(RuntimeError, match="different M3B dataset roots"):
        verify_m4._checkpoint_bound_smoke_dataset(
            verify_m4.Report(),
            prior_gate_dataset_root=tmp_path / "current",
            per_task_training=per_task,
            onehot_training=onehot,
        )


def test_runtime_specific_analysis_preserves_clean_canonical_history(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    output = run_root / "train-new-runtime"
    reports = run_root / "reports"
    output.mkdir(parents=True)
    reports.mkdir()
    run_fingerprint = "sha256:" + "a" * 64
    checkpoint_fingerprint = "sha256:" + "b" * 64
    old_git = {
        "commit": "1" * 40,
        "dirty": False,
        "changed_paths": [],
        "baseline_tracked": True,
    }
    new_git = {**old_git, "commit": "2" * 40}
    canonical = {
        "schema_version": "langmani-m4-counterfactual-sensitivity-v1",
        "passed": True,
        "run_fingerprint": run_fingerprint,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "evaluation_git": old_git,
        "first_actions": [[0.0] * 8],
    }
    current = {
        **canonical,
        "evaluation_git": new_git,
        "first_actions": [[1.0] * 8],
    }
    canonical_path = reports / "counterfactual_sensitivity.json"
    canonical_path.write_text(json.dumps(canonical), encoding="utf-8")
    (output / evaluate_cli.EVALUATION_ANALYSIS_FILE).write_text(
        json.dumps(current), encoding="utf-8"
    )
    identity = SimpleNamespace(
        variant=ActVariant.MIXED_TASK_ONEHOT,
        run_fingerprint=run_fingerprint,
    )

    evaluate_cli._publish_analysis(
        output=output,
        run_root=run_root,
        identity=identity,
        checkpoint_fingerprint=checkpoint_fingerprint,
        evaluation_git=new_git,
    )

    assert json.loads(canonical_path.read_text(encoding="utf-8")) == canonical
    assert json.loads((output / "analysis.json").read_text(encoding="utf-8")) == current


def test_runtime_specific_analysis_rejects_wrong_canonical_checkpoint(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    output = run_root / "train-new-runtime"
    reports = run_root / "reports"
    output.mkdir(parents=True)
    reports.mkdir()
    run_fingerprint = "sha256:" + "a" * 64
    checkpoint_fingerprint = "sha256:" + "b" * 64
    evaluation_git = {
        "commit": "1" * 40,
        "dirty": False,
        "changed_paths": [],
        "baseline_tracked": True,
    }
    current = {
        "schema_version": "langmani-m4-counterfactual-sensitivity-v1",
        "passed": True,
        "run_fingerprint": run_fingerprint,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "evaluation_git": evaluation_git,
    }
    (output / evaluate_cli.EVALUATION_ANALYSIS_FILE).write_text(
        json.dumps(current), encoding="utf-8"
    )
    canonical = {**current, "checkpoint_fingerprint": "sha256:" + "c" * 64}
    (reports / "counterfactual_sensitivity.json").write_text(
        json.dumps(canonical), encoding="utf-8"
    )
    identity = SimpleNamespace(
        variant=ActVariant.MIXED_TASK_ONEHOT,
        run_fingerprint=run_fingerprint,
    )

    with pytest.raises(RuntimeError, match="existing immutable evaluation artifact differs"):
        evaluate_cli._publish_analysis(
            output=output,
            run_root=run_root,
            identity=identity,
            checkpoint_fingerprint=checkpoint_fingerprint,
            evaluation_git=evaluation_git,
        )


def test_verify_dry_run_requires_target_full(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["verify_m4.py", "--dry-run"])
    with pytest.raises(SystemExit) as error:
        verify_m4.parse_args()
    assert error.value.code == 2


def test_full_commands_bind_planned_identity_and_projected_action_boundary(
    tmp_path: Path,
) -> None:
    train_command = verify_m4._training_command(
        dataset_root=tmp_path / "dataset",
        model_root=tmp_path / "models",
        variant=ActVariant.MIXED_TASK_ONEHOT,
        task_id=None,
        mode="dry-run",
        planned_mode=ExperimentMode.FULL,
        steps=100_000,
        report_path=tmp_path / "train.json",
    )
    assert "--dry-run" in train_command
    assert train_command[train_command.index("--planned-mode") + 1] == "full"
    evaluation_command = verify_m4._evaluation_command(
        checkpoint=tmp_path / "checkpoint",
        dataset_root=tmp_path / "dataset",
        split="fresh_seed",
        action_bound_mode="project",
        report_path=tmp_path / "evaluation.json",
    )
    assert evaluation_command[evaluation_command.index("--action-bound-mode") + 1] == "project"


def test_full_dry_run_validates_eight_unique_semantic_runs_without_training(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    model_root = tmp_path / "models"
    report_path = tmp_path / "diagnostics" / "verification.json"
    monkeypatch.setattr(verify_m4, "REPORT_PATH", report_path)
    completed = SimpleNamespace(
        summary=SimpleNamespace(total_episodes=360),
        export_fingerprint="sha256:" + "d" * 64,
        split_manifest_digest="sha256:" + "e" * 64,
    )
    monkeypatch.setattr(
        verify_m4, "load_completed_m3b_dataset", lambda *_args, **_kwargs: completed
    )
    git_payload = {
        "commit": "1" * 40,
        "dirty": False,
        "changed_paths": [],
        "baseline_tracked": True,
    }
    monkeypatch.setattr(
        verify_m4,
        "inspect_git_state",
        lambda _root: SimpleNamespace(to_dict=lambda: git_payload),
    )
    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> tuple[bool, dict[str, object]]:
        commands.append(command)
        index = len(commands) - 1
        variant = ActVariant(command[command.index("--variant") + 1])
        task_id = command[command.index("--task-id") + 1] if "--task-id" in command else None
        fingerprint = "sha256:" + f"{index + 1:064x}"
        state_dimension = 15 if variant is ActVariant.MIXED_TASK_ONEHOT else 9
        train_count = 48 if variant is ActVariant.PER_TASK else 288
        validation_count = 6 if variant is ActVariant.PER_TASK else 36
        return True, {
            "passed": True,
            "dry_run": True,
            "training_started": False,
            "fixture_evidence": False,
            "planned_experiment_mode": "full",
            "dataset_fingerprint": completed.export_fingerprint,
            "split_digest": completed.split_manifest_digest,
            "variant": variant.value,
            "task_id": task_id,
            "train_episode_count": train_count,
            "validation_episode_count": validation_count,
            "offline_loss_episode_count": validation_count,
            "offline_loss_role": "held_out_validation",
            "optimization": ActOptimizationConfig().to_dict(),
            "checkpoint_schedule": list(range(5_000, 100_001, 5_000)),
            "evaluation_schedule": list(range(5_000, 100_001, 5_000)),
            "input_shapes": {STATE_FEATURE_KEY: [state_dimension]},
            "output_shape": [8],
            "act_config": ActModelConfig.for_variant(variant).to_dict(),
            "git": git_payload,
            "run_fingerprint": fingerprint,
            "expected_output_directory": str(
                model_root.resolve() / fingerprint.removeprefix("sha256:")
            ),
            "train_statistics_fingerprint": "sha256:" + f"{index + 20:064x}",
        }

    monkeypatch.setattr(verify_m4, "_run", fake_run)
    report = verify_m4.Report()
    verify_m4._run_target_full_dry_run(
        report,
        dataset_root,
        model_root,
        action_bound_mode="project",
    )
    assert report.full_dry_run_validated
    assert report.planned_full_run_count == 8
    assert len(commands) == 8
    assert all("--dry-run" in command and "--full" not in command for command in commands)
    plan = json.loads(
        (report_path.parent / "target_full" / "dry_run_plan.json").read_text(encoding="utf-8")
    )
    assert plan["passed"] is True
    assert plan["training_started"] is False
    assert plan["action_bound_mode"] == "project"
    assert len({item["run_fingerprint"] for item in plan["runs"]}) == 8


def test_strict_probe_rejects_then_executes_one_explicit_projection() -> None:
    class Space:
        low = np.full(8, -1.0, dtype=np.float32)
        high = np.full(8, 1.0, dtype=np.float32)
        shape = (8,)
        dtype = np.dtype(np.float32)

    class Environment:
        action_space = Space()
        calls: list[np.ndarray]

        def __init__(self) -> None:
            self.calls = []

        def step(self, action: np.ndarray) -> tuple[None, float, bool, bool, dict[str, object]]:
            self.calls.append(action.copy())
            return None, 0.0, False, False, {}

    env = Environment()
    raw = np.zeros((1, 8), dtype=np.float32)
    raw[0, -1] = 1.0508
    result = evaluate_cli._strict_bound_probe(
        env=env,
        raw_action=evaluate_cli.torch.from_numpy(raw),
        runtime_manifest=_runtime_manifest(
            checkpoint_fingerprint="sha256:" + "e" * 64,
            split=EvaluationSplit.TRAIN,
        ),
    )
    assert result["reject_blocked_before_env_step"] is True
    assert result["real_projected_env_step_executed"] is True
    assert len(env.calls) == 1
    assert env.calls[0][-1] == 1.0


def test_pre_m41_interrupted_evaluation_is_preserved_with_stable_legacy_identity(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    staging = run_root / ".train.staging"
    staging.mkdir(parents=True)
    legacy_owner = {
        "schema_version": "langmani-m4-evaluation-owner-v1",
        "run_fingerprint": "sha256:" + "a" * 64,
        "checkpoint_fingerprint": "sha256:" + "b" * 64,
        "split": "train",
        "schedule_digest": "sha256:" + "c" * 64,
    }
    (staging / evaluate_cli.EVALUATION_OWNER_FILE).write_text(
        json.dumps(legacy_owner), encoding="utf-8"
    )
    (staging / "failure.txt").write_text("preserve me", encoding="utf-8")
    archived = evaluate_cli._archive_interrupted_evaluation(run_root, staging)
    assert not staging.exists()
    assert archived.parent == run_root / "failed_evaluations"
    assert "legacy-" in archived.name
    assert (archived / "failure.txt").read_text(encoding="utf-8") == "preserve me"
