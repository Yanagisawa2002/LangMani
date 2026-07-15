from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from langmani.policies.act_checkpoint import load_act_checkpoint, save_act_checkpoint
from langmani.policies.act_runtime import GitState
from langmani.policies.act_types import (
    ActDataConfig,
    ActModelConfig,
    ActOptimizationConfig,
    CheckpointRecord,
    TrainingState,
)
from langmani.policies.m42_schedule import (
    M42_DEV_SCHEDULE_FINGERPRINT,
    M42_FINAL_SCHEDULE_FINGERPRINT,
)
from langmani.policies.m42_training import (
    RUNTIME_SELECTION_SCHEMA,
    M42TrainingContractError,
    RuntimeSelectionContract,
    TaskTokenFairComparisonContract,
    TaskTokenRunIdentity,
    TaskTokenTrainingManifest,
    TaskTokenValidationQueue,
    TaskTokenValidationQueueItem,
    fingerprint_owned_task_token_runs,
    load_runtime_selection,
    validate_completed_task_token_run,
)
from langmani.policies.m42_types import (
    M42ExperimentManifest,
    M42Stage,
    TaskTokenConfig,
    TaskTokenExperimentConfig,
)


def _load_cli() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "train_act_task_token.py"
    spec = importlib.util.spec_from_file_location("langmani_test_train_act_task_token", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cli = _load_cli()


def _digest(character: str) -> str:
    return f"sha256:{character * 64}"


def _runtime_payload(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": RUNTIME_SELECTION_SCHEMA,
        "execution_horizon": 5,
        "gripper_mode": "binary",
        "horizon_selection_fingerprint": _digest("a"),
        "gripper_selection_fingerprint": _digest("b"),
        "runtime_fingerprint": _digest("c"),
        "development_schedule_fingerprint": _digest("d"),
        "m3b_dataset_fingerprint": _digest("e"),
        "mixed_task_onehot_checkpoint_fingerprint": _digest("f"),
        "representative_per_task_checkpoint_fingerprint": _digest("1"),
        "implementation_fingerprint": _digest("9"),
        "experiment_manifest_fingerprint": _digest("8"),
        "evaluation_git_commit": "a" * 40,
        "selection_evidence": {
            "horizon_selection": "horizon_selection.json",
            "gripper_selection": "gripper_selection.json",
            "experiment_manifest": "experiment_manifest.json",
            "task_token_fair_comparison_contract": _fair_comparison_payload(),
        },
        "locked": True,
        "final_schedule_accessed": False,
    }
    value.update(overrides)
    return value


def _fair_comparison_payload() -> dict[str, object]:
    optimization = ActOptimizationConfig().to_dict()
    train_indices = tuple(range(288))
    validation_indices = tuple(range(288, 324))
    return TaskTokenFairComparisonContract(
        source_run_fingerprint=_digest("6"),
        source_checkpoint_fingerprint=_digest("f"),
        m3b_export_fingerprint=_digest("e"),
        m3b_split_manifest_digest=_digest("2"),
        ordered_train_episode_indices=train_indices,
        ordered_validation_episode_indices=validation_indices,
        source_model_config=ActModelConfig(state_dimension=15).to_dict(),
        source_optimization_config=optimization,
        source_common_data_contract={
            **ActDataConfig(dataset_root="fixture").identity_dict(),
            "m3b_feature_contract": {"fixture": "features"},
            "policy_state_schema": {"fixture": "state"},
        },
        data_order_contract={
            "schema_version": "langmani-m42-deterministic-data-order-v0",
            "sampler": "DeterministicResumeBatchSampler",
            "ordered_train_episode_indices": list(train_indices),
            "ordered_validation_episode_indices": list(validation_indices),
            "training_seed": 0,
            "batch_size": optimization["batch_size"],
            "epoch_seed_rule": "(training_seed + epoch) % (2**63 - 1)",
            "resume_cursor": "completed_optimizer_step",
        },
        source_training_seed=0,
        source_device="cuda",
        source_dtype="float32",
        allowed_input_difference={
            "source_state": {
                "feature_type": "STATE",
                "dimension": 15,
                "contents": "PandaPolicyStateV0+CanonicalTaskOneHotV0",
            },
            "task_token_state": {
                "feature_type": "STATE",
                "dimension": 9,
                "contents": "PandaPolicyStateV0",
            },
            "task_token_condition": {
                "feature_type": "ENV",
                "dimension": 6,
                "mapping": "CanonicalTaskTokenV0",
            },
            "all_other_differences_permitted": False,
        },
    ).to_dict()


def _runtime(tmp_path: Path) -> RuntimeSelectionContract:
    manifest = M42ExperimentManifest(
        stage=M42Stage.RUNTIME_ABLATION,
        implementation_git_commit="a" * 40,
        m3b_dataset_fingerprint=_digest("e"),
        prior_m4_verification_fingerprint=_digest("7"),
        schedule_fingerprints={
            "m42_dev_v0": M42_DEV_SCHEDULE_FINGERPRINT,
            "m42_final_v0": M42_FINAL_SCHEDULE_FINGERPRINT,
        },
        m4_checkpoint_fingerprints=tuple(_digest(value) for value in "2345f1ab"),
        m41_runtime_processor_fingerprint=_digest("6"),
        runtime_selection_fingerprints={
            "execution_horizon": _digest("a"),
            "gripper_runtime": _digest("b"),
        },
        task_token_experiment_fingerprint=None,
        artifact_paths={
            "runtime_selection": "runtime_selection.json",
            "horizon_selection": "horizon_selection.json",
            "gripper_selection": "gripper_selection.json",
            "post_grasp_analysis": "post_grasp_analysis.json",
        },
        completed=True,
    )
    (tmp_path / "experiment_manifest.json").write_text(
        json.dumps(manifest.to_dict()), encoding="utf-8"
    )
    path = tmp_path / "runtime_selection.json"
    path.write_text(
        json.dumps(_runtime_payload(experiment_manifest_fingerprint=manifest.fingerprint)),
        encoding="utf-8",
    )
    return load_runtime_selection(path)


def _identity(**overrides: object) -> TaskTokenRunIdentity:
    values: dict[str, object] = {
        "m3b_export_fingerprint": _digest("e"),
        "m3b_split_manifest_digest": _digest("2"),
        "ordered_train_episode_indices": tuple(range(288)),
        "ordered_validation_episode_indices": tuple(range(288, 324)),
        "model_config": {
            "base_m4_model": {"state_dimension": 9, "chunk_size": 50},
            "task_token": {"input_dimension": 6, "hidden_dimension": 512},
        },
        "data_contract": {
            "selection_source": "m3b_validation_only",
            "test_episode_indices_accessed": [],
        },
        "train_statistics_fingerprint": _digest("3"),
        "optimization_config": {
            "training_steps": 100_000,
            "checkpoint_interval": 5_000,
            "batch_size": 32,
            "mixed_precision": "bfloat16",
        },
        "runtime_selection_fingerprint": _digest("c"),
        "runtime_selection_source_fingerprint": _digest("4"),
        "experiment_manifest_fingerprint": _digest("8"),
        "task_token_architecture_fingerprint": _digest("5"),
        "training_seed": 0,
        "device": "cuda",
        "dtype": "float32",
        "lerobot_version": "0.6.0",
        "torch_version": "2.7.1+cu128",
        "cuda_version": "12.8",
        "git_commit": "6" * 40,
        "git_dirty": False,
    }
    values.update(overrides)
    return TaskTokenRunIdentity(**values)


def _checkpoint(step: int, identity: TaskTokenRunIdentity) -> CheckpointRecord:
    return CheckpointRecord(
        checkpoint_fingerprint=_digest(format(step // 5_000, "x")[-1]),
        run_fingerprint=identity.run_fingerprint,
        global_step=step,
        relative_path=f"checkpoints/step-{step:08d}-fixture",
        policy_config_digest=_digest("7"),
        statistics_fingerprint=identity.train_statistics_fingerprint,
        split_digest=identity.m3b_split_manifest_digest,
        git_commit=identity.git_commit,
        complete=True,
    )


def _precheckpoint_run(
    output_root: Path,
    identity: TaskTokenRunIdentity,
    experiment: TaskTokenExperimentConfig,
    *,
    metric_steps: int = 17,
) -> Path:
    run_root = cli.safe_run_directory(output_root, identity.run_fingerprint)
    (run_root / "reports").mkdir(parents=True)
    manifest = TaskTokenTrainingManifest(
        identity=identity,
        task_token_experiment=experiment.to_dict(),
        training_state=TrainingState(global_step=0, examples_processed=0),
        checkpoints=(),
        training_complete=False,
    )
    artifacts = {
        "run_manifest.json": manifest.to_dict(),
        "task_token_experiment.json": experiment.to_dict(),
        "dataset_contract.json": identity.to_dict()["data_contract"],
        "runtime_selection.json": {"runtime_fingerprint": identity.runtime_selection_fingerprint},
        "task_token_contract.json": {
            "architecture_fingerprint": identity.task_token_architecture_fingerprint
        },
        "train_stats.json": {"statistics_fingerprint": identity.train_statistics_fingerprint},
    }
    for relative, payload in artifacts.items():
        (run_root / relative).write_text(json.dumps(payload), encoding="utf-8")
    (run_root / "reports" / "initial_validation.json").write_text(
        json.dumps(
            {
                "run_fingerprint": identity.run_fingerprint,
                "loss": 0.5,
                "test_accessed": False,
                "development_schedule_accessed": False,
                "final_schedule_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    if metric_steps:
        (run_root / "metrics.jsonl").write_text(
            "".join(
                json.dumps({"step": step, "total_loss": 1.0 / step}) + "\n"
                for step in range(1, metric_steps + 1)
            ),
            encoding="utf-8",
        )
    return run_root


class _CheckpointPolicy:
    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        push_to_hub: bool = False,
        **kwargs: object,
    ) -> None:
        assert push_to_hub is False and not kwargs
        root = Path(save_directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / "policy.json").write_text("{}\n", encoding="utf-8")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        local_files_only: bool,
        strict: bool,
    ) -> _CheckpointPolicy:
        assert local_files_only and strict
        assert (Path(pretrained_name_or_path) / "policy.json").is_file()
        return cls()


class _CheckpointProcessor:
    def __init__(self, label: str) -> None:
        self.label = label

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        push_to_hub: bool = False,
        config_filename: str | None = None,
        **kwargs: object,
    ) -> None:
        assert push_to_hub is False and config_filename is not None and not kwargs
        root = Path(save_directory)
        (root / config_filename).write_text("{}\n", encoding="utf-8")
        (root / f"{self.label}.safetensors").write_bytes(self.label.encode())


def _checkpoint_processor_loader(
    policy: object, root: Path
) -> tuple[_CheckpointProcessor, _CheckpointProcessor]:
    assert isinstance(policy, _CheckpointPolicy)
    assert (root / "policy_preprocessor.json").is_file()
    assert (root / "policy_postprocessor.json").is_file()
    return _CheckpointProcessor("loaded_pre"), _CheckpointProcessor("loaded_post")


def test_runtime_selection_is_content_bound_and_rejects_final_access(tmp_path: Path) -> None:
    first = _runtime(tmp_path)
    assert first.execution_horizon == 5
    assert first.gripper_mode == "binary"
    assert first.source_fingerprint.startswith("sha256:")

    changed_root = tmp_path / "changed"
    changed_root.mkdir()
    (changed_root / "experiment_manifest.json").write_bytes(
        (tmp_path / "experiment_manifest.json").read_bytes()
    )
    changed_path = changed_root / "runtime_selection.json"
    changed_path.write_text(
        json.dumps(
            _runtime_payload(
                execution_horizon=1,
                experiment_manifest_fingerprint=first.experiment_manifest_fingerprint,
            )
        ),
        encoding="utf-8",
    )
    changed = load_runtime_selection(changed_path)
    assert changed.source_fingerprint != first.source_fingerprint

    forbidden_root = tmp_path / "forbidden"
    forbidden_root.mkdir()
    (forbidden_root / "experiment_manifest.json").write_bytes(
        (tmp_path / "experiment_manifest.json").read_bytes()
    )
    forbidden = forbidden_root / "runtime_selection.json"
    forbidden.write_text(
        json.dumps(
            _runtime_payload(
                final_schedule_accessed=True,
                experiment_manifest_fingerprint=first.experiment_manifest_fingerprint,
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(M42TrainingContractError, match="must not consume m42_final"):
        load_runtime_selection(forbidden)

    embedded_root = tmp_path / "embedded"
    embedded_root.mkdir()
    (embedded_root / "experiment_manifest.json").write_bytes(
        (tmp_path / "experiment_manifest.json").read_bytes()
    )
    embedded = embedded_root / "runtime_selection.json"
    embedded.write_text(
        json.dumps(
            _runtime_payload(
                final_schedule_fingerprint=_digest("9"),
                experiment_manifest_fingerprint=first.experiment_manifest_fingerprint,
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(M42TrainingContractError, match="must not embed or reference"):
        load_runtime_selection(embedded)


def test_task_token_default_runtime_selection_chains_from_runtime_ablation() -> None:
    assert cli.DEFAULT_RUNTIME_SELECTION.parts[-2:] == (
        "runtime_ablation",
        "runtime_selection.json",
    )


def test_fair_comparison_allows_only_9d_state_plus_6d_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = cli._internal_experiment(tmp_path, device="cuda")
    completed = SimpleNamespace(
        export_fingerprint=_digest("e"),
        split_manifest_digest=_digest("2"),
        views=SimpleNamespace(
            train=SimpleNamespace(episode_indices=tuple(range(288))),
            validation=SimpleNamespace(episode_indices=tuple(range(288, 324))),
        ),
        manifest=SimpleNamespace(
            config=SimpleNamespace(
                feature_contract=SimpleNamespace(to_dict=lambda: {"fixture": "features"}),
                policy_state_schema=SimpleNamespace(to_dict=lambda: {"fixture": "state"}),
            )
        ),
    )
    effective = {
        "input_features": {
            experiment.model.image_feature_key: {
                "type": "VISUAL",
                "shape": list(experiment.model.image_shape_chw),
            },
            experiment.model.state_feature_key: {"type": "STATE", "shape": [9]},
            cli.TASK_TOKEN_FEATURE_KEY: {"type": "ENV", "shape": [6]},
        },
        "output_features": {experiment.model.action_feature_key: {"type": "ACTION", "shape": [8]}},
        "normalization_mapping": {
            "VISUAL": "MEAN_STD",
            "STATE": "MEAN_STD",
            "ACTION": "MEAN_STD",
            "ENV": "IDENTITY",
        },
        "device": "cuda",
        "use_amp": True,
        "chunk_size": 50,
        "n_action_steps": 10,
        "dim_model": 512,
    }
    monkeypatch.setattr(cli, "_effective_act_config_dict", lambda _config: effective)
    contract = TaskTokenFairComparisonContract.from_dict(_fair_comparison_payload())
    assert (
        cli._validate_fair_comparison(
            contract=contract,
            experiment=experiment,
            completed=completed,
            effective_act_config=object(),
            dry_run=False,
        )
        == contract.contract_fingerprint
    )

    changed_payload = _fair_comparison_payload()
    optimization = dict(changed_payload["source_optimization_config"])
    optimization["learning_rate"] = 2e-5
    changed_payload["source_optimization_config"] = optimization
    changed_payload["contract_fingerprint"] = ""
    changed = TaskTokenFairComparisonContract.from_dict(changed_payload)
    with pytest.raises(RuntimeError, match="optimization differs"):
        cli._validate_fair_comparison(
            contract=changed,
            experiment=experiment,
            completed=completed,
            effective_act_config=object(),
            dry_run=False,
        )


def test_task_token_identity_is_separate_stable_and_resume_sensitive() -> None:
    identity = _identity()
    assert TaskTokenRunIdentity.from_dict(identity.to_dict()) == identity
    assert identity.model_label == "act_mixed_task_token"
    assert "variant" not in identity.to_dict()

    changed_runtime = _identity(runtime_selection_fingerprint=_digest("8"))
    changed_token = _identity(task_token_architecture_fingerprint=_digest("9"))
    changed_stats = _identity(train_statistics_fingerprint=_digest("a"))
    assert (
        len(
            {
                identity.run_fingerprint,
                changed_runtime.run_fingerprint,
                changed_token.run_fingerprint,
                changed_stats.run_fingerprint,
            }
        )
        == 4
    )
    with pytest.raises(M42TrainingContractError, match="clean Git"):
        _identity(git_dirty=True)


def test_manifest_and_validation_queue_bind_twenty_validation_only_checkpoints() -> None:
    identity = _identity()
    experiment = TaskTokenExperimentConfig(
        dataset_fingerprint=identity.m3b_export_fingerprint,
        split_digest=identity.m3b_split_manifest_digest,
        train_statistics_fingerprint=identity.train_statistics_fingerprint,
        implementation_git_commit=identity.git_commit,
        runtime_selection_fingerprint=identity.runtime_selection_fingerprint,
    )
    checkpoints = tuple(_checkpoint(step, identity) for step in range(5_000, 100_001, 5_000))
    state = TrainingState(
        global_step=100_000,
        examples_processed=3_200_000,
        last_checkpoint_fingerprint=checkpoints[-1].checkpoint_fingerprint,
        completed=True,
    )
    manifest = TaskTokenTrainingManifest(
        identity=identity,
        task_token_experiment=experiment.to_dict(),
        training_state=state,
        checkpoints=checkpoints,
        training_complete=True,
    )
    restored = TaskTokenTrainingManifest.from_dict(manifest.to_dict())
    assert restored == manifest
    assert manifest.to_dict()["selected_checkpoint_fingerprint"] is None

    items = tuple(
        TaskTokenValidationQueueItem(
            global_step=record.global_step,
            checkpoint_fingerprint=record.checkpoint_fingerprint,
            checkpoint_relative_path=record.relative_path,
            offline_validation_loss=1.0 / index,
        )
        for index, record in enumerate(checkpoints, start=1)
    )
    queue = TaskTokenValidationQueue(
        run_fingerprint=identity.run_fingerprint,
        m3b_dataset_fingerprint=identity.m3b_export_fingerprint,
        split_digest=identity.m3b_split_manifest_digest,
        train_statistics_fingerprint=identity.train_statistics_fingerprint,
        runtime_selection_fingerprint=identity.runtime_selection_fingerprint,
        experiment_manifest_fingerprint=identity.experiment_manifest_fingerprint,
        task_token_architecture_fingerprint=identity.task_token_architecture_fingerprint,
        validation_schedule_fingerprint=_digest("b"),
        ordered_validation_episode_indices=identity.ordered_validation_episode_indices,
        checkpoints=items,
        git_commit=identity.git_commit,
        complete=True,
    )
    payload = queue.to_dict()
    assert TaskTokenValidationQueue.from_dict(payload) == queue
    assert len(payload["checkpoints"]) == 20
    assert payload["selection_source"] == "m3b_validation_only"
    assert payload["forbidden_selection_sources"] == [
        "m3b_test",
        "m4_fresh_seed",
        "m42_dev_v0",
        "m42_final_v0",
    ]


def test_completed_training_is_integrity_reused_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _identity(device="cpu", cuda_version=None)
    experiment = TaskTokenExperimentConfig(
        dataset_fingerprint=identity.m3b_export_fingerprint,
        split_digest=identity.m3b_split_manifest_digest,
        train_statistics_fingerprint=identity.train_statistics_fingerprint,
        implementation_git_commit=identity.git_commit,
        runtime_selection_fingerprint=identity.runtime_selection_fingerprint,
    )
    checkpoints = tuple(_checkpoint(step, identity) for step in range(5_000, 100_001, 5_000))
    manifest = TaskTokenTrainingManifest(
        identity=identity,
        task_token_experiment=experiment.to_dict(),
        training_state=TrainingState(
            global_step=100_000,
            examples_processed=3_200_000,
            last_checkpoint_fingerprint=checkpoints[-1].checkpoint_fingerprint,
            completed=True,
        ),
        checkpoints=checkpoints,
        training_complete=True,
    )
    queue = TaskTokenValidationQueue(
        run_fingerprint=identity.run_fingerprint,
        m3b_dataset_fingerprint=identity.m3b_export_fingerprint,
        split_digest=identity.m3b_split_manifest_digest,
        train_statistics_fingerprint=identity.train_statistics_fingerprint,
        runtime_selection_fingerprint=identity.runtime_selection_fingerprint,
        experiment_manifest_fingerprint=identity.experiment_manifest_fingerprint,
        task_token_architecture_fingerprint=identity.task_token_architecture_fingerprint,
        validation_schedule_fingerprint=_digest("b"),
        ordered_validation_episode_indices=identity.ordered_validation_episode_indices,
        checkpoints=tuple(
            TaskTokenValidationQueueItem(
                global_step=record.global_step,
                checkpoint_fingerprint=record.checkpoint_fingerprint,
                checkpoint_relative_path=record.relative_path,
                offline_validation_loss=1.0 / index,
            )
            for index, record in enumerate(checkpoints, start=1)
        ),
        git_commit=identity.git_commit,
        complete=True,
    )
    run_root = tmp_path / identity.run_fingerprint.replace(":", "-")
    (run_root / "reports").mkdir(parents=True)
    for record in checkpoints:
        checkpoint_root = run_root / record.relative_path
        checkpoint_root.mkdir(parents=True)
        (checkpoint_root / cli.CHECKPOINT_COMPLETION_MARKER).write_text(
            "complete\n", encoding="utf-8"
        )
    summary = {
        "schema_version": cli.TRAINING_SUMMARY_SCHEMA,
        "passed": True,
        "run_fingerprint": identity.run_fingerprint,
        "experiment_manifest_fingerprint": identity.experiment_manifest_fingerprint,
        "validation_queue_fingerprint": queue.queue_fingerprint,
        "final_checkpoint_fingerprint": checkpoints[-1].checkpoint_fingerprint,
        "task_token_training_completed": True,
        "task_token_checkpoint_selected": False,
        "checkpoint_count": 20,
        "test_accessed": False,
        "development_schedule_accessed_for_selection": False,
        "final_schedule_accessed": False,
    }
    completion = {
        "schema_version": cli.TRAINING_COMPLETION_SCHEMA,
        "run_fingerprint": identity.run_fingerprint,
        "experiment_manifest_fingerprint": identity.experiment_manifest_fingerprint,
        "training_summary_fingerprint": f"sha256:{cli.sha256_hex(summary)}",
        "validation_queue_fingerprint": queue.queue_fingerprint,
        "final_checkpoint_fingerprint": checkpoints[-1].checkpoint_fingerprint,
        "checkpoint_selection_pending": True,
    }
    (run_root / "run_manifest.json").write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    (run_root / "validation_queue.json").write_text(json.dumps(queue.to_dict()), encoding="utf-8")
    (run_root / "reports" / "training_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    assert cli._latest_safe_resume_checkpoint(run_root, manifest) == checkpoints[-1].relative_path
    cli._repair_missing_completion_marker(
        run_root=run_root,
        identity=identity,
        task_experiment=experiment,
    )
    assert (
        json.loads((run_root / "training_complete.json").read_text(encoding="utf-8")) == completion
    )
    validated = validate_completed_task_token_run(run_root, fresh_reload=False)
    assert validated.manifest == manifest
    assert validated.queue == queue
    monkeypatch.setattr(
        cli,
        "load_act_checkpoint",
        lambda **_kwargs: SimpleNamespace(policy=object(), record=checkpoints[-1]),
    )
    monkeypatch.setattr(cli, "validate_task_token_policy", lambda *_args, **_kwargs: None)
    result = cli._reuse_completed_training(
        run_root=run_root,
        identity=identity,
        task_experiment=experiment,
        report={"schema_version": "fixture"},
        task_token=TaskTokenConfig(),
    )
    assert result["passed"] is True
    assert result["training_reused"] is True
    assert result["training_started"] is False
    assert result["checkpoint_count"] == 20


def test_promoted_100k_resume_can_finalize_without_an_extra_training_step() -> None:
    state = TrainingState(global_step=100_000, examples_processed=3_200_000)
    outcome = cli._finalization_only_outcome(state)
    assert outcome is not None
    assert outcome.final_step == 100_000
    assert outcome.examples_processed == 3_200_000
    assert outcome.metrics == ()
    assert (
        cli._finalization_only_outcome(
            TrainingState(global_step=95_000, examples_processed=3_040_000)
        )
        is None
    )


def test_precheckpoint_interruption_is_archived_then_rebuilt_from_step_zero(
    tmp_path: Path,
) -> None:
    identity = _identity(device="cpu", cuda_version=None)
    experiment = TaskTokenExperimentConfig(
        dataset_fingerprint=identity.m3b_export_fingerprint,
        split_digest=identity.m3b_split_manifest_digest,
        train_statistics_fingerprint=identity.train_statistics_fingerprint,
        implementation_git_commit=identity.git_commit,
        runtime_selection_fingerprint=identity.runtime_selection_fingerprint,
    )
    output_root = tmp_path / "models"
    run_root = _precheckpoint_run(output_root, identity, experiment, metric_steps=37)
    original_evidence = {
        path.relative_to(run_root).as_posix(): path.read_bytes()
        for path in run_root.rglob("*")
        if path.is_file()
    }

    resume_checkpoint, archive = cli._recover_incomplete_attempt(
        output_root=output_root,
        run_root=run_root,
        identity=identity,
        task_experiment=experiment,
    )

    assert resume_checkpoint is None
    assert archive is not None and archive.is_dir()
    assert not run_root.exists()
    assert archive.parent == output_root.resolve() / ".recovery" / run_root.name
    for relative, expected in original_evidence.items():
        assert (archive / relative).read_bytes() == expected
    recovery = json.loads((archive / "precheckpoint_recovery.json").read_text(encoding="utf-8"))
    assert recovery["schema_version"] == cli.PRECHECKPOINT_RECOVERY_SCHEMA
    assert recovery["run_fingerprint"] == identity.run_fingerprint
    assert recovery["metric_record_count"] == 37
    assert recovery["last_completed_metric_step"] == 37
    assert recovery["declared_checkpoint_count"] == 0
    assert recovery["promoted_checkpoint_count"] == 0

    # The canonical fingerprint-owned path is now free for the caller to
    # reconstruct the exact same identity from step zero; the archive remains.
    run_root.mkdir(parents=True)
    assert archive.is_dir()


def test_precheckpoint_recovery_resumes_after_marker_write_before_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity(device="cpu", cuda_version=None)
    experiment = TaskTokenExperimentConfig(
        dataset_fingerprint=identity.m3b_export_fingerprint,
        split_digest=identity.m3b_split_manifest_digest,
        train_statistics_fingerprint=identity.train_statistics_fingerprint,
        implementation_git_commit=identity.git_commit,
        runtime_selection_fingerprint=identity.runtime_selection_fingerprint,
    )
    output_root = tmp_path / "models"
    run_root = _precheckpoint_run(output_root, identity, experiment, metric_steps=43)
    original_rename = cli.os.rename

    def interrupt_after_marker(_source: Path, _destination: Path) -> None:
        raise OSError("simulated interruption after immutable marker write")

    monkeypatch.setattr(cli.os, "rename", interrupt_after_marker)
    with pytest.raises(RuntimeError, match="evidence archival failed"):
        cli._recover_incomplete_attempt(
            output_root=output_root,
            run_root=run_root,
            identity=identity,
            task_experiment=experiment,
        )

    marker_path = run_root / "precheckpoint_recovery.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    archive = output_root.resolve() / marker["archive_directory"]
    assert run_root.is_dir()
    assert not archive.exists()

    monkeypatch.setattr(cli.os, "rename", original_rename)
    resume_checkpoint, resumed_archive = cli._recover_incomplete_attempt(
        output_root=output_root,
        run_root=run_root,
        identity=identity,
        task_experiment=experiment,
    )

    assert resume_checkpoint is None
    assert resumed_archive == archive
    assert not run_root.exists()
    assert (
        json.loads((archive / "precheckpoint_recovery.json").read_text(encoding="utf-8")) == marker
    )
    assert len(cli._metrics(archive / "metrics.jsonl")) == 43


def test_precheckpoint_recovery_rejects_identity_mismatch_without_moving_evidence(
    tmp_path: Path,
) -> None:
    identity = _identity(device="cpu", cuda_version=None)
    experiment = TaskTokenExperimentConfig(
        dataset_fingerprint=identity.m3b_export_fingerprint,
        split_digest=identity.m3b_split_manifest_digest,
        train_statistics_fingerprint=identity.train_statistics_fingerprint,
        implementation_git_commit=identity.git_commit,
        runtime_selection_fingerprint=identity.runtime_selection_fingerprint,
    )
    output_root = tmp_path / "models"
    run_root = _precheckpoint_run(output_root, identity, experiment)
    mismatched = _identity(
        device="cpu",
        cuda_version=None,
        runtime_selection_fingerprint=_digest("8"),
    )
    mismatched_manifest = TaskTokenTrainingManifest(
        identity=mismatched,
        task_token_experiment=experiment.to_dict(),
        training_state=TrainingState(global_step=0, examples_processed=0),
    )
    (run_root / "run_manifest.json").write_text(
        json.dumps(mismatched_manifest.to_dict()),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="identity differs"):
        cli._recover_incomplete_attempt(
            output_root=output_root,
            run_root=run_root,
            identity=identity,
            task_experiment=experiment,
        )

    assert run_root.is_dir()
    assert not (output_root / ".recovery").exists()


def test_precheckpoint_recovery_rejects_unsafe_layout(tmp_path: Path) -> None:
    identity = _identity(device="cpu", cuda_version=None)
    experiment = TaskTokenExperimentConfig(
        dataset_fingerprint=identity.m3b_export_fingerprint,
        split_digest=identity.m3b_split_manifest_digest,
        train_statistics_fingerprint=identity.train_statistics_fingerprint,
        implementation_git_commit=identity.git_commit,
        runtime_selection_fingerprint=identity.runtime_selection_fingerprint,
    )
    output_root = tmp_path / "models"
    run_root = _precheckpoint_run(output_root, identity, experiment)
    (run_root / "unexpected.bin").write_bytes(b"unsafe")

    with pytest.raises(RuntimeError, match="unsafe artifact layout"):
        cli._recover_incomplete_attempt(
            output_root=output_root,
            run_root=run_root,
            identity=identity,
            task_experiment=experiment,
        )

    assert run_root.is_dir()
    assert not (output_root / ".recovery").exists()


def test_existing_checkpoint_lifecycle_accepts_and_strictly_reloads_m42_identity(
    tmp_path: Path,
) -> None:
    identity = _identity(device="cpu", cuda_version=None)
    optimizer = torch.optim.AdamW(torch.nn.Linear(1, 1).parameters())
    record = save_act_checkpoint(
        run_root=tmp_path,
        identity=identity,
        training_state=TrainingState(global_step=5_000, examples_processed=160_000),
        policy=_CheckpointPolicy(),
        preprocessor=_CheckpointProcessor("pre"),
        postprocessor=_CheckpointProcessor("post"),
        optimizer=optimizer,
        training_metric={"step": 5_000, "validation_loss": 0.25},
    )
    loaded = load_act_checkpoint(
        run_root=tmp_path,
        checkpoint_relative_path=record.relative_path,
        expected_identity=identity,
        policy_class=_CheckpointPolicy,
        processor_loader=_checkpoint_processor_loader,
    )
    assert loaded.record == record
    assert loaded.training_metric == {"step": 5_000, "validation_loss": 0.25}
    assert loaded.component_fingerprints.model.startswith("sha256:")


def test_dry_run_reports_exact_plan_without_starting_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime(tmp_path)
    identity = _identity(
        device="cpu", runtime_selection_source_fingerprint=runtime.source_fingerprint
    )
    task_experiment = TaskTokenExperimentConfig(
        dataset_fingerprint=identity.m3b_export_fingerprint,
        split_digest=identity.m3b_split_manifest_digest,
        train_statistics_fingerprint=identity.train_statistics_fingerprint,
        implementation_git_commit=identity.git_commit,
        runtime_selection_fingerprint=identity.runtime_selection_fingerprint,
    )
    completed = SimpleNamespace(
        export_fingerprint=identity.m3b_export_fingerprint,
        split_manifest_digest=identity.m3b_split_manifest_digest,
        views=SimpleNamespace(
            train=SimpleNamespace(episode_indices=tuple(range(288))),
            validation=SimpleNamespace(episode_indices=tuple(range(288, 324))),
        ),
    )
    statistics = SimpleNamespace(statistics_fingerprint=identity.train_statistics_fingerprint)
    monkeypatch.setattr(
        cli,
        "inspect_git_state",
        lambda _root: GitState(
            commit=identity.git_commit,
            dirty=False,
            changed_paths=(),
            baseline_tracked=True,
        ),
    )
    monkeypatch.setattr(cli, "load_runtime_selection", lambda _path: runtime)
    monkeypatch.setattr(
        cli,
        "_load_data_and_statistics",
        lambda _root: (completed, statistics),
    )
    monkeypatch.setattr(cli, "build_task_token_act_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        cli,
        "_effective_act_config_dict",
        lambda _config: {"fixture_effective_act_config": True},
    )
    monkeypatch.setattr(cli, "_effective_act_config_dict", lambda _config: {})
    fair = TaskTokenFairComparisonContract.from_dict(_fair_comparison_payload())
    monkeypatch.setattr(cli, "_fair_comparison_contract", lambda _runtime: fair)
    monkeypatch.setattr(
        cli,
        "_validate_fair_comparison",
        lambda **_kwargs: fair.contract_fingerprint,
    )
    monkeypatch.setattr(
        cli,
        "_identity",
        lambda **kwargs: (identity, task_experiment, _digest("c")),
    )
    args = cli.parse_args(
        [
            "--dry-run",
            "--device",
            "cpu",
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--runtime-selection",
            str(tmp_path / "runtime_selection.json"),
            "--output-root",
            str(tmp_path / "models"),
            "--report",
            str(tmp_path / "report.json"),
        ]
    )
    report = cli.execute(args)
    assert report["passed"] is True
    assert report["training_started"] is False
    assert report["checkpoint_schedule"] == list(range(5_000, 100_001, 5_000))
    assert report["train_episode_count"] == 288
    assert report["validation_episode_count"] == 36
    assert report["test_episode_count_used_for_training_or_selection"] == 0
    assert report["task_token_training_completed"] is False
    assert report["task_token_checkpoint_selected"] is False
    assert report["validation_only_selection_validated"] is False
    assert not (tmp_path / "models").exists()


def test_task_token_output_refuses_a_second_full_run_identity(tmp_path: Path) -> None:
    output_root = tmp_path / "models"
    expected = output_root / ("a" * 64)
    existing = output_root / ("b" * 64)
    existing.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="exactly one ACT-Mixed-TaskToken"):
        cli._enforce_single_task_token_run(output_root, expected)

    existing.rename(expected)
    cli._enforce_single_task_token_run(output_root, expected)


def test_recovery_namespace_is_not_a_second_task_token_run(tmp_path: Path) -> None:
    output_root = tmp_path / "models"
    expected = output_root / ("a" * 64)
    expected.mkdir(parents=True)
    (output_root / ".recovery" / ("b" * 64)).mkdir(parents=True)
    assert fingerprint_owned_task_token_runs(output_root) == (expected.resolve(),)


def test_training_output_and_report_cannot_overlap_immutable_inputs(tmp_path: Path) -> None:
    runtime_path = tmp_path / "runtime_selection.json"
    runtime_path.write_text(json.dumps(_runtime_payload()), encoding="utf-8")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    overlapping_output = cli.parse_args(
        [
            "--dry-run",
            "--device",
            "cpu",
            "--dataset-root",
            str(dataset),
            "--runtime-selection",
            str(runtime_path),
            "--output-root",
            str(dataset / "models"),
            "--report",
            str(tmp_path / "report.json"),
        ]
    )
    with pytest.raises(RuntimeError, match="must not overlap M3B"):
        cli._validate_paths(overlapping_output)

    overlapping_report = cli.parse_args(
        [
            "--dry-run",
            "--device",
            "cpu",
            "--dataset-root",
            str(dataset),
            "--runtime-selection",
            str(runtime_path),
            "--output-root",
            str(tmp_path / "models"),
            "--report",
            str(runtime_path),
        ]
    )
    with pytest.raises(RuntimeError, match="must not overlap"):
        cli._validate_paths(overlapping_report)

    overlapping_report.report = overlapping_report.output_root / "command.json"
    with pytest.raises(RuntimeError, match="must not overlap"):
        cli._validate_paths(overlapping_report)


def test_training_output_and_report_cannot_overlap_source_or_git(tmp_path: Path) -> None:
    runtime_path = tmp_path / "runtime_selection.json"
    runtime_path.write_text(json.dumps(_runtime_payload()), encoding="utf-8")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    args = cli.parse_args(
        [
            "--dry-run",
            "--device",
            "cpu",
            "--dataset-root",
            str(dataset),
            "--runtime-selection",
            str(runtime_path),
            "--output-root",
            str(cli.PROJECT_ROOT / ".git" / "unsafe-task-token-output"),
            "--report",
            str(tmp_path / "report.json"),
        ]
    )
    with pytest.raises(RuntimeError, match="source, or Git"):
        cli._validate_paths(args)

    args.output_root = tmp_path / "models"
    args.report = cli.PROJECT_ROOT / "src" / "langmani" / "unsafe-report.json"
    with pytest.raises(RuntimeError, match="source, or Git"):
        cli._validate_paths(args)


def test_training_rejects_linked_input_or_output_ancestor(tmp_path: Path) -> None:
    runtime_path = tmp_path / "runtime_selection.json"
    runtime_path.write_text(json.dumps(_runtime_payload()), encoding="utf-8")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    real = tmp_path / "real-output-parent"
    real.mkdir()
    linked = tmp_path / "linked-output-parent"
    try:
        linked.symlink_to(real, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")
    args = cli.parse_args(
        [
            "--dry-run",
            "--device",
            "cpu",
            "--dataset-root",
            str(dataset),
            "--runtime-selection",
            str(runtime_path),
            "--output-root",
            str(linked / "models"),
            "--report",
            str(tmp_path / "report.json"),
        ]
    )
    with pytest.raises(RuntimeError, match="symlink or junction"):
        cli._validate_paths(args)


def test_cli_failure_is_nonzero_and_writes_machine_readable_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path = tmp_path / "failure.json"

    def fail(_args: object) -> dict[str, object]:
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(cli, "execute", fail)
    status = cli.main(["--dry-run", "--device", "cpu", "--report", str(report_path)])
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert status == 1
    assert payload["passed"] is False
    assert payload["error_type"] == "RuntimeError"
    assert payload["error_message"] == "fixture failure"
