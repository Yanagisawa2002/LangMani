from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from environment import verify_m42
from langmani.policies.m42_source_fingerprint import (
    TASK_TOKEN_TRAINING_SOURCE_FILES,
    task_token_training_source_fingerprint,
)
from langmani.policies.m42_training import canonical_fingerprint


def _args(tmp_path: Path, *, dry_run: bool) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_root=tmp_path / "dataset",
        m4_model_root=tmp_path / "m4-models",
        task_token_model_root=tmp_path / "task-token-models",
        m4_diagnostics_root=tmp_path / "m4-diagnostics",
        output_root=tmp_path / "m42-output",
        target_development=True,
        target_final=False,
        dry_run=dry_run,
    )


def _runtime_report(*, physical: bool) -> dict[str, object]:
    return {
        "passed": True,
        "physical_execution": physical,
        "final_schedule_accessed": False,
        "horizon_ablation_completed": physical,
        "horizon_selection_locked": physical,
        "gripper_ablation_completed": physical,
        "gripper_selection_locked": physical,
        "post_grasp_analysis_completed": physical,
        "raw_action_metrics_validated": physical,
        "runtime_action_metrics_validated": physical,
    }


def _development_report(*, physical: bool) -> dict[str, object]:
    return {
        "passed": True,
        "physical_execution": physical,
        "final_schedule_accessed": False,
        "final_benchmark_completed": False,
        "go_no_go_decision_completed": False,
        "smolvla_go": False,
        "development_benchmark_completed": physical,
        "task_token_checkpoint_selected": physical,
        "validation_only_selection_validated": physical,
        "post_grasp_analysis_completed": physical,
        "raw_action_metrics_validated": physical,
        "runtime_action_metrics_validated": physical,
    }


def test_target_development_dry_run_has_independent_truthful_acceptance(
    tmp_path: Path, monkeypatch
) -> None:
    args = _args(tmp_path, dry_run=True)
    selection = args.output_root / "runtime_ablation" / "runtime_selection.json"
    selection.parent.mkdir(parents=True)
    selection.write_text("{}\n", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(command: list[str], _report: Path) -> dict[str, object]:
        commands.append(command)
        if any(item.endswith("run_m42_runtime_ablation.py") for item in command):
            return _runtime_report(physical=False)
        if any(item.endswith("train_act_task_token.py") for item in command):
            return {
                "passed": True,
                "training_started": False,
                "task_token_training_completed": False,
            }
        return _development_report(physical=False)

    monkeypatch.setattr(verify_m42, "_run_json", fake_run)
    report = verify_m42.Report(
        implementation_validated=True,
        prior_m4_evidence_validated=True,
        development_schedule_locked=True,
        final_schedule_locked=True,
    )
    verify_m42._run_target_development(report, args)

    assert len(commands) == 3
    assert all("--dry-run" in command for command in commands)
    assert report.dry_run_validated
    assert not report.physical_target_validated
    assert not report.development_benchmark_completed
    assert report.accepted("target_development", dry_run=True)
    assert not report.accepted("target_development")


def test_fresh_target_development_dry_run_defers_training_without_fake_selection(
    tmp_path: Path, monkeypatch
) -> None:
    args = _args(tmp_path, dry_run=True)
    commands: list[list[str]] = []

    def fake_run(command: list[str], _report: Path) -> dict[str, object]:
        commands.append(command)
        if any(item.endswith("run_m42_runtime_ablation.py") for item in command):
            return _runtime_report(physical=False)
        assert any(item.endswith("evaluate_m42.py") for item in command)
        return _development_report(physical=False)

    monkeypatch.setattr(verify_m42, "_run_json", fake_run)
    report = verify_m42.Report(
        implementation_validated=True,
        prior_m4_evidence_validated=True,
        development_schedule_locked=True,
        final_schedule_locked=True,
    )
    verify_m42._run_target_development(report, args)

    assert len(commands) == 2
    training = report.dry_run_plan["task_token_training"]
    assert isinstance(training, dict)
    assert training["deferred"] is True
    assert training["executed"] is False
    assert "does not fabricate" in str(training["deferred_reason"])
    assert not args.output_root.exists()
    assert report.accepted("target_development", dry_run=True)


def test_target_development_reuses_content_bound_training_without_full_retrain(
    tmp_path: Path, monkeypatch
) -> None:
    args = _args(tmp_path, dry_run=False)
    commands: list[list[str]] = []

    def fake_run(command: list[str], _report: Path) -> dict[str, object]:
        commands.append(command)
        if any(item.endswith("run_m42_runtime_ablation.py") for item in command):
            return _runtime_report(physical=True)
        if any(item.endswith("train_act_task_token.py") for item in command):
            assert "--dry-run" in command
            return {"passed": True, "training_started": False}
        return _development_report(physical=True)

    monkeypatch.setattr(verify_m42, "_run_json", fake_run)
    monkeypatch.setattr(
        verify_m42,
        "_find_content_bound_task_token_training",
        lambda **_kwargs: {
            "passed": True,
            "training_reused": True,
            "task_token_training_completed": True,
        },
    )
    report = verify_m42.Report()
    verify_m42._run_target_development(report, args)

    training_commands = [
        item for item in commands if any(part.endswith("train_act_task_token.py") for part in item)
    ]
    assert len(training_commands) == 1
    assert "--dry-run" in training_commands[0]
    assert report.task_token_training_completed
    assert report.development_benchmark_completed
    assert report.physical_target_validated


@pytest.mark.parametrize(
    "runtime_override",
    (
        {"physical_execution": False},
        {"final_schedule_accessed": True},
    ),
)
def test_target_development_rejects_unphysical_or_final_runtime_evidence(
    tmp_path: Path, monkeypatch, runtime_override: dict[str, object]
) -> None:
    args = _args(tmp_path, dry_run=False)
    runtime = {**_runtime_report(physical=True), **runtime_override}
    monkeypatch.setattr(verify_m42, "_run_json", lambda _command, _report: runtime)

    with pytest.raises(RuntimeError, match="development-only execution boundary"):
        verify_m42._run_target_development(verify_m42.Report(), args)


def test_target_development_rejects_evaluation_that_touched_final(
    tmp_path: Path, monkeypatch
) -> None:
    args = _args(tmp_path, dry_run=False)

    def fake_run(command: list[str], _report: Path) -> dict[str, object]:
        if any(item.endswith("run_m42_runtime_ablation.py") for item in command):
            return _runtime_report(physical=True)
        if any(item.endswith("train_act_task_token.py") for item in command):
            return {"passed": True, "training_started": False}
        return {**_development_report(physical=True), "final_schedule_accessed": True}

    monkeypatch.setattr(verify_m42, "_run_json", fake_run)
    monkeypatch.setattr(
        verify_m42,
        "_find_content_bound_task_token_training",
        lambda **_kwargs: {
            "passed": True,
            "training_reused": True,
            "task_token_training_completed": True,
        },
    )
    with pytest.raises(RuntimeError, match="crossed the sealed-final boundary"):
        verify_m42._run_target_development(verify_m42.Report(), args)


def test_target_final_dry_run_then_explicit_authorized_physical_execution(
    tmp_path: Path, monkeypatch
) -> None:
    args = _args(tmp_path, dry_run=False)
    args.target_development = False
    args.target_final = True
    commands: list[list[str]] = []
    implementation = "sha256:" + "a" * 64

    def fake_run(command: list[str], _report: Path) -> dict[str, object]:
        commands.append(command)
        if "--dry-run" in command:
            return {
                "passed": True,
                "implementation_fingerprint": implementation,
                "physical_execution": False,
                "final_schedule_accessed": False,
                "final_benchmark_completed": False,
                "go_no_go_decision_completed": False,
            }
        return {
            "passed": True,
            "physical_execution": True,
            "task_token_checkpoint_selected": True,
            "validation_only_selection_validated": True,
            "development_benchmark_completed": True,
            "final_benchmark_completed": True,
            "post_grasp_analysis_completed": True,
            "raw_action_metrics_validated": True,
            "runtime_action_metrics_validated": True,
            "go_no_go_decision_completed": True,
            "smolvla_go": False,
        }

    monkeypatch.setattr(verify_m42, "_run_json", fake_run)
    report = verify_m42.Report()
    verify_m42._run_target_final(report, args)

    assert len(commands) == 2
    assert "--dry-run" in commands[0]
    assert "--authorize-sealed-final" not in commands[0]
    assert "--authorize-sealed-final" in commands[1]
    fingerprint_index = commands[1].index("--expected-implementation-fingerprint") + 1
    assert commands[1][fingerprint_index] == implementation
    assert report.final_benchmark_completed
    assert report.go_no_go_decision_completed
    assert report.physical_target_validated
    assert not report.smolvla_go


def test_target_final_dry_run_never_authorizes_or_materializes_final(
    tmp_path: Path, monkeypatch
) -> None:
    args = _args(tmp_path, dry_run=True)
    args.target_development = False
    args.target_final = True
    commands: list[list[str]] = []

    def fake_run(command: list[str], _report: Path) -> dict[str, object]:
        commands.append(command)
        return {
            "passed": True,
            "implementation_fingerprint": "sha256:" + "b" * 64,
            "physical_execution": False,
            "final_schedule_accessed": False,
            "final_benchmark_completed": False,
            "go_no_go_decision_completed": False,
        }

    monkeypatch.setattr(verify_m42, "_run_json", fake_run)
    report = verify_m42.Report(
        implementation_validated=True,
        prior_m4_evidence_validated=True,
        development_schedule_locked=True,
        final_schedule_locked=True,
    )
    verify_m42._run_target_final(report, args)

    assert len(commands) == 1
    assert "--dry-run" in commands[0]
    assert "--authorize-sealed-final" not in commands[0]
    assert report.dry_run_validated
    assert report.accepted("target_final", dry_run=True)
    assert not report.final_benchmark_completed
    assert not report.physical_target_validated


def test_content_bound_training_reuse_rejects_changed_effective_act_architecture() -> None:
    effective = {
        "input_features": {
            "observation.images.base_camera": {"type": "visual", "shape": [3, 256, 256]},
            "observation.state": {"type": "state", "shape": [9]},
            "observation.environment_state": {"type": "environment", "shape": [6]},
        },
        "output_features": {"action": {"type": "action", "shape": [8]}},
        "chunk_size": 50,
        "n_action_steps": 50,
        "dim_model": 512,
        "n_encoder_layers": 4,
    }
    task_experiment = {
        "dataset_fingerprint": "sha256:" + "1" * 64,
        "split_digest": "sha256:" + "2" * 64,
        "train_statistics_fingerprint": "sha256:" + "3" * 64,
        "implementation_git_commit": "a" * 40,
        "runtime_selection_fingerprint": "sha256:" + "4" * 64,
    }
    model_contract = {"effective_act_config": effective, "task_token": {"hidden_dimension": 512}}
    data_contract = {"validation_schedule": "m3b-validation"}
    identity_dict = {
        "m3b_export_fingerprint": "sha256:" + "1" * 64,
        "m3b_split_manifest_digest": "sha256:" + "2" * 64,
        "ordered_train_episode_indices": list(range(288)),
        "ordered_validation_episode_indices": list(range(288, 324)),
        "model_config": model_contract,
        "data_contract": data_contract,
        "train_statistics_fingerprint": "sha256:" + "3" * 64,
        "optimization_config": {"batch_size": 32},
        "runtime_selection_fingerprint": "sha256:" + "4" * 64,
        "runtime_selection_source_fingerprint": "sha256:" + "5" * 64,
        "experiment_manifest_fingerprint": "sha256:" + "0" * 64,
        "task_token_architecture_fingerprint": "sha256:" + "6" * 64,
        "git_dirty": False,
    }
    identity = SimpleNamespace(to_dict=lambda: identity_dict, run_fingerprint="sha256:" + "7" * 64)
    manifest = SimpleNamespace(
        identity=identity,
        task_token_experiment=task_experiment,
        training_complete=True,
        checkpoints=(),
    )
    queue = SimpleNamespace(
        complete=True,
        run_fingerprint=identity.run_fingerprint,
        queue_fingerprint="sha256:" + "8" * 64,
        checkpoints=(),
    )
    content_experiment = dict(task_experiment)
    content_experiment.pop("implementation_git_commit")
    plan = {
        "dataset_fingerprint": identity_dict["m3b_export_fingerprint"],
        "split_digest": identity_dict["m3b_split_manifest_digest"],
        "train_statistics_fingerprint": identity_dict["train_statistics_fingerprint"],
        "runtime_selection_fingerprint": identity_dict["runtime_selection_fingerprint"],
        "runtime_selection_source_fingerprint": identity_dict[
            "runtime_selection_source_fingerprint"
        ],
        "experiment_manifest_fingerprint": identity_dict["experiment_manifest_fingerprint"],
        "task_token_architecture_fingerprint": identity_dict["task_token_architecture_fingerprint"],
        "optimization": identity_dict["optimization_config"],
        "model_contract_fingerprint": canonical_fingerprint(model_contract),
        "data_contract_fingerprint": canonical_fingerprint(data_contract),
        "train_episode_count": 288,
        "validation_episode_count": 36,
        "effective_act_config": effective,
        "task_token_experiment_content_fingerprint": canonical_fingerprint(content_experiment),
        "input_shapes": {key: value["shape"] for key, value in effective["input_features"].items()},
        "output_shape": [8],
        "checkpoint_schedule": [],
        "task_token_training_source_fingerprint": "sha256:" + "9" * 64,
        "task_token_training_source_files": list(verify_m42.TASK_TOKEN_TRAINING_SOURCE_FILES),
    }
    assert verify_m42._task_token_content_matches_plan(
        manifest=manifest,
        queue=queue,
        plan=plan,
        historical_source_fingerprint="sha256:" + "9" * 64,
    )

    changed = dict(plan)
    changed_effective = dict(effective)
    changed_effective["n_encoder_layers"] = 5
    changed["effective_act_config"] = changed_effective
    assert not verify_m42._task_token_content_matches_plan(
        manifest=manifest,
        queue=queue,
        plan=changed,
        historical_source_fingerprint="sha256:" + "9" * 64,
    )

    assert not verify_m42._task_token_content_matches_plan(
        manifest=manifest,
        queue=queue,
        plan=plan,
        historical_source_fingerprint="sha256:" + "0" * 64,
    )

    sources = {name: f"original:{name}".encode() for name in TASK_TOKEN_TRAINING_SOURCE_FILES}
    original_fingerprint = task_token_training_source_fingerprint(sources)
    plan["task_token_training_source_fingerprint"] = original_fingerprint
    assert verify_m42._task_token_content_matches_plan(
        manifest=manifest,
        queue=queue,
        plan=plan,
        historical_source_fingerprint=original_fingerprint,
    )

    changed_sources = dict(sources)
    changed_sources["src/langmani/policies/act_data.py"] = b"changed data semantics"
    changed_fingerprint = task_token_training_source_fingerprint(changed_sources)
    assert changed_fingerprint != original_fingerprint
    assert not verify_m42._task_token_content_matches_plan(
        manifest=manifest,
        queue=queue,
        plan=plan,
        historical_source_fingerprint=changed_fingerprint,
    )


def test_task_token_training_source_closure_covers_semantic_dependencies() -> None:
    required = {
        "src/langmani/policies/act_training.py",
        "src/langmani/policies/act_task_token.py",
        "src/langmani/policies/act_checkpoint.py",
        "src/langmani/policies/act_data.py",
        "src/langmani/policies/act_runtime.py",
        "src/langmani/policies/act_types.py",
        "src/langmani/policies/m42_training.py",
        "src/langmani/policies/m42_types.py",
        "src/langmani/policies/m42_source_fingerprint.py",
        "src/langmani/datasets/lerobot_types.py",
        "src/langmani/datasets/policy_state.py",
        "src/langmani/datasets/schedule.py",
        "src/langmani/datasets/splits.py",
    }
    assert len(TASK_TOKEN_TRAINING_SOURCE_FILES) == 20
    assert required <= set(TASK_TOKEN_TRAINING_SOURCE_FILES)
    assert verify_m42.TASK_TOKEN_TRAINING_SOURCE_FILES is TASK_TOKEN_TRAINING_SOURCE_FILES


def test_content_bound_reuse_rejects_complete_and_incomplete_sibling_runs(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "models"
    first = model_root / ("a" * 64)
    second = model_root / ("b" * 64)
    first.mkdir(parents=True)
    second.mkdir()
    (first / "training_complete.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="complete or incomplete"):
        verify_m42._find_content_bound_task_token_training(
            model_root=model_root,
            plan={"run_fingerprint": "sha256:" + "a" * 64},
        )


def test_content_bound_reuse_rejects_a_different_incomplete_identity(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    (model_root / ("b" * 64)).mkdir(parents=True)
    with pytest.raises(RuntimeError, match="identity differs"):
        verify_m42._find_content_bound_task_token_training(
            model_root=model_root,
            plan={"run_fingerprint": "sha256:" + "a" * 64},
        )


def test_historical_training_source_uses_same_closure_and_rejects_missing(
    monkeypatch,
) -> None:
    sources = {name: f"historical:{name}".encode() for name in TASK_TOKEN_TRAINING_SOURCE_FILES}
    seen: list[str] = []

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        relative_name = command[-1].split(":", maxsplit=1)[1]
        seen.append(relative_name)
        content = sources.get(relative_name)
        if content is None:
            return SimpleNamespace(returncode=128, stdout=b"", stderr=b"path absent")
        return SimpleNamespace(returncode=0, stdout=content, stderr=b"")

    monkeypatch.setattr(verify_m42.subprocess, "run", fake_run)
    expected = task_token_training_source_fingerprint(sources)
    assert verify_m42._task_token_training_source_fingerprint_at_commit("a" * 40) == expected
    assert seen == list(TASK_TOKEN_TRAINING_SOURCE_FILES)

    del sources["src/langmani/policies/act_data.py"]
    with pytest.raises(RuntimeError, match="cannot audit historical TaskToken source.*act_data"):
        verify_m42._task_token_training_source_fingerprint_at_commit("a" * 40)
