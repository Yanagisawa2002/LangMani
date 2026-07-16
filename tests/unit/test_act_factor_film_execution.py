from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from langmani.policies.act_factor_film_execution import (
    FactorFiLMExecutionConfig,
    FactorFiLMExecutionError,
    FactorFiLMExecutionPaths,
    _raw_actions_by_step,
    _runtime_fingerprint,
    _safe_child_directory,
    _write_or_validate,
)
from langmani.policies.act_factor_film_types import canonical_fingerprint
from scripts import evaluate_act_factor_film as cli


def _paths(tmp_path: Path) -> FactorFiLMExecutionPaths:
    return FactorFiLMExecutionPaths(
        dataset_root=tmp_path / "dataset",
        factor_film_model_root=tmp_path / "models",
        semantic_audit_evidence_root=tmp_path / "semantic",
        m4_checkpoint_root=tmp_path / "m4",
        task_token_checkpoint_root=tmp_path / "token",
        m42_diagnostics_root=tmp_path / "m42",
        runtime_selection_path=tmp_path / "runtime.json",
        output_root=tmp_path / "evaluation",
    )


def test_execution_config_binds_clean_evaluator_commit(tmp_path: Path) -> None:
    config = FactorFiLMExecutionConfig(
        paths=_paths(tmp_path),
        evaluation_git_commit="a" * 40,
        expected_training_git_commit="b" * 40,
        device="cuda",
    )
    assert config.evaluation_git_commit == "a" * 40
    with pytest.raises(FactorFiLMExecutionError, match="evaluation_git_commit"):
        FactorFiLMExecutionConfig(
            paths=_paths(tmp_path),
            evaluation_git_commit="8ee0f1b",
            expected_training_git_commit="b" * 40,
        )
    with pytest.raises(FactorFiLMExecutionError, match="expected_training_git_commit"):
        FactorFiLMExecutionConfig(
            paths=_paths(tmp_path),
            evaluation_git_commit="a" * 40,
            expected_training_git_commit="8ee0f1b",
        )
    with pytest.raises(FactorFiLMExecutionError, match="requires CUDA"):
        FactorFiLMExecutionConfig(
            paths=_paths(tmp_path),
            evaluation_git_commit="a" * 40,
            expected_training_git_commit="b" * 40,
            device="cpu",
        )


def test_cli_default_uses_promoted_semantic_audit_evidence() -> None:
    assert cli.DEFAULT_SEMANTIC_EVIDENCE_ROOT.parts[-3:] == (
        "semantic-audit",
        "evidence",
        "930ed848f8700a5ebc8734b1d3eb0fe22fd8fd4a3592c65825f2d813a32205e2",
    )


def test_work_artifacts_are_immutable_and_resume_identically(tmp_path: Path) -> None:
    path = tmp_path / "stage" / "result.json"
    payload = {"schema_version": "fixture-v0", "passed": True}
    _write_or_validate(path, payload)
    _write_or_validate(path, payload)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    with pytest.raises(FactorFiLMExecutionError, match="immutable generated artifact changed"):
        _write_or_validate(path, {**payload, "passed": False})


def test_fingerprint_owned_work_rejects_preexisting_directory_links(tmp_path: Path) -> None:
    real = tmp_path / "outside"
    real.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(real, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")

    with pytest.raises(FactorFiLMExecutionError, match="traverses a link"):
        _safe_child_directory(tmp_path, "linked", "linked work root")


def test_runtime_fingerprint_uses_config_and_action_space_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Config:
        def to_dict(self) -> dict[str, object]:
            return {"mode": "project", "schema_version": "fixture"}

    processor = SimpleNamespace(
        config=_Config(),
        action_space_contract=lambda: {
            "lower_bounds": [-1.0] * 8,
            "upper_bounds": [1.0] * 8,
        },
    )
    monkeypatch.setattr(
        "langmani.policies.act_factor_film_execution."
        "BoundedActionEnvPostprocessorV0.from_environment",
        lambda *args, **kwargs: processor,
    )
    observed = _runtime_fingerprint(object())
    assert observed == canonical_fingerprint(
        {
            "component": "BoundedActionEnvPostprocessorV0",
            "config": processor.config.to_dict(),
            "action_space_contract": processor.action_space_contract(),
        }
    )


def test_raw_release_actions_preserve_rollout_steps_and_raw_values() -> None:
    episode = {
        "runtime_projection_audits": [
            {"rollout_step": 1, "raw_action": [[0.1] * 7 + [-1.0]]},
            {"rollout_step": 2, "raw_action": [[0.2] * 7 + [1.0]]},
        ]
    }
    actions = _raw_actions_by_step(episode)
    assert tuple(actions) == (1, 2)
    assert actions[1][-1] == -1.0
    assert actions[2][-1] == 1.0

    duplicate = {
        "runtime_projection_audits": [
            {"rollout_step": 1, "raw_action": [0.0] * 8},
            {"rollout_step": 1, "raw_action": [0.0] * 8},
        ]
    }
    with pytest.raises(FactorFiLMExecutionError, match="duplicate step"):
        _raw_actions_by_step(duplicate)


def test_all_cli_uses_three_processes_with_fresh_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = cli.parse_args(
        [
            "--target-development",
            "--training-git-commit",
            "a" * 40,
            "--stage",
            "all",
            "--device",
            "cpu",
            "--report",
            str(tmp_path / "report.json"),
        ]
    )
    monkeypatch.setattr(cli, "validate_evaluator_git", lambda root: {"commit": "b" * 40})
    observed: list[str] = []

    def run(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        assert command[command.index("--training-git-commit") + 1] == "a" * 40
        stage = command[command.index("--stage") + 1]
        observed.append(stage)
        if stage == "development":
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text('{"passed":true,"stage":"development"}', encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", run)
    report = cli.execute(args)
    assert observed == ["validation", "reload", "development"]
    assert report["fresh_process_reload_validated"] is True
    assert report["subprocess_stages"] == observed


def test_cli_rejects_command_report_inside_immutable_evidence(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    args = cli.parse_args(
        [
            "--target-development",
            "--training-git-commit",
            "a" * 40,
            "--output-root",
            str(output),
            "--report",
            str(output / "command.json"),
        ]
    )
    with pytest.raises(RuntimeError, match="outside generated evidence"):
        cli._validate_command_report_path(args)
