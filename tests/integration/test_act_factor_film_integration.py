"""Installed LeRobot 0.6.0 integration tests for M4.3b FactorFiLM."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

from langmani.datasets.lerobot_types import IMAGE_FEATURE_KEY, STATE_FEATURE_KEY
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.policies.act_factor_film_conditioning import (
    attach_factor_film_runtime_input,
    factorized_runtime_input,
)
from langmani.policies.act_factor_film_training import (
    AUTHORIZED_M43_AUDIT_EVIDENCE_FINGERPRINT,
    FactorFiLMTrainingConfig,
    FactorFiLMTrainingMode,
    build_factor_film_policy_and_processors,
    run_factor_film_fixture,
)
from langmani.policies.act_factor_film_types import FactorFiLMConfig, FactorFiLMContractError
from langmani.policies.act_training import prepare_raw_batch
from langmani.policies.act_types import (
    ActDataConfig,
    ActModelConfig,
    ActOptimizationConfig,
)

_GIT_COMMIT = "dfea8b3d7d28274909ff178cb9087a9a90e17ee7"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CLI = _PROJECT_ROOT / "scripts" / "train_act_factor_film.py"


def _model() -> ActModelConfig:
    return ActModelConfig(
        state_dimension=9,
        chunk_size=4,
        n_action_steps=2,
        dim_model=32,
        n_heads=2,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        latent_dim=8,
        n_vae_encoder_layers=1,
        dropout=0.0,
    )


def _training_config() -> FactorFiLMTrainingConfig:
    return FactorFiLMTrainingConfig(
        data=ActDataConfig(dataset_root="fixture-only"),
        base_model=_model(),
        optimization=ActOptimizationConfig(
            mixed_precision="none",
            batch_size=1,
            dataloader_workers=0,
            training_steps=1,
            checkpoint_interval=1,
            validation_interval=1,
        ),
        semantic_audit_evidence_fingerprint=(AUTHORIZED_M43_AUDIT_EVIDENCE_FINGERPRINT),
        training_seed=43,
        mode=FactorFiLMTrainingMode.FIXTURE,
        device="cpu",
    )


def _stats() -> dict[str, dict[str, torch.Tensor]]:
    return {
        IMAGE_FEATURE_KEY: {
            "mean": torch.zeros(3),
            "std": torch.ones(3),
            "min": torch.zeros(3),
            "max": torch.ones(3),
        },
        STATE_FEATURE_KEY: {
            "mean": torch.zeros(9),
            "std": torch.ones(9),
            "min": -torch.ones(9),
            "max": torch.ones(9),
        },
        "action": {
            "mean": torch.zeros(8),
            "std": torch.ones(8),
            "min": -torch.ones(8),
            "max": torch.ones(8),
        },
    }


@pytest.mark.fixture
@pytest.mark.integration
def test_factor_film_hooks_are_separate_and_identity_like() -> None:
    torch.manual_seed(43)
    _, policy, preprocessor, _, _ = build_factor_film_policy_and_processors(
        training_config=_training_config(),
        statistics=_stats(),
        factor_film_config=FactorFiLMConfig(state_context_dimension=32),
    )
    generator = torch.Generator().manual_seed(91)
    raw = {
        IMAGE_FEATURE_KEY: torch.randint(
            0,
            256,
            (1, 3, 256, 256),
            dtype=torch.uint8,
            generator=generator,
        ),
        STATE_FEATURE_KEY: torch.linspace(-0.2, 0.2, 9).reshape(1, 9),
    }
    processed = preprocessor(prepare_raw_batch(raw))
    assert isinstance(processed, Mapping)

    raw_visual: list[torch.Tensor] = []
    conditioned_visual: list[torch.Tensor] = []
    raw_state: list[torch.Tensor] = []
    conditioned_state: list[torch.Tensor] = []

    def capture_raw_visual(
        _module: torch.nn.Module,
        _inputs: tuple[object, ...],
        output: object,
    ) -> None:
        assert isinstance(output, Mapping)
        value = output["feature_map"]
        assert isinstance(value, torch.Tensor)
        raw_visual.append(value.detach().clone())

    def capture_conditioned_visual(
        _module: torch.nn.Module,
        _inputs: tuple[object, ...],
        output: object,
    ) -> None:
        assert isinstance(output, Mapping)
        value = output["feature_map"]
        assert isinstance(value, torch.Tensor)
        conditioned_visual.append(value.detach().clone())

    def capture_raw_state(
        _module: torch.nn.Module,
        _inputs: tuple[object, ...],
        output: object,
    ) -> None:
        assert isinstance(output, torch.Tensor)
        raw_state.append(output.detach().clone())

    def capture_conditioned_state(
        _module: torch.nn.Module,
        _inputs: tuple[object, ...],
        output: object,
    ) -> None:
        assert isinstance(output, torch.Tensor)
        conditioned_state.append(output.detach().clone())

    handles = (
        policy.model.backbone.register_forward_hook(capture_raw_visual, prepend=True),
        policy.model.backbone.register_forward_hook(capture_conditioned_visual),
        policy.model.encoder_robot_state_input_proj.register_forward_hook(
            capture_raw_state, prepend=True
        ),
        policy.model.encoder_robot_state_input_proj.register_forward_hook(
            capture_conditioned_state
        ),
    )
    outputs: list[torch.Tensor] = []
    # Same observation: object-only change, then bin-only change.
    for task_spec in (
        CANONICAL_TASK_SPECS[0],
        CANONICAL_TASK_SPECS[4],
        CANONICAL_TASK_SPECS[1],
    ):
        conditioned = attach_factor_film_runtime_input(
            processed,
            factorized_runtime_input(task_spec),
        )
        policy.eval()
        torch.manual_seed(7)
        outputs.append(policy.predict_action_chunk(conditioned))
    for handle in handles:
        handle.remove()

    assert len(raw_visual) == len(conditioned_visual) == 3
    assert len(raw_state) == len(conditioned_state) == 3
    torch.testing.assert_close(raw_visual[0], raw_visual[1], rtol=0, atol=0)
    torch.testing.assert_close(raw_visual[0], raw_visual[2], rtol=0, atol=0)
    torch.testing.assert_close(raw_state[0], raw_state[1], rtol=0, atol=0)
    torch.testing.assert_close(raw_state[0], raw_state[2], rtol=0, atol=0)

    # Object changed while bin stayed left: only the visual FiLM result changes.
    assert not torch.equal(conditioned_visual[0], conditioned_visual[1])
    torch.testing.assert_close(conditioned_state[0], conditioned_state[1], rtol=0, atol=0)
    # Bin changed while object stayed red: only the state FiLM result changes.
    torch.testing.assert_close(conditioned_visual[0], conditioned_visual[2], rtol=0, atol=0)
    assert not torch.equal(conditioned_state[0], conditioned_state[2])
    assert not torch.equal(outputs[0], outputs[1])
    assert not torch.equal(outputs[0], outputs[2])
    assert float((conditioned_visual[0] - raw_visual[0]).abs().max()) < 1e-3
    assert float((conditioned_state[0] - raw_state[0]).abs().max()) < 1e-3
    assert policy._active_object_indices is None
    assert policy._active_bin_indices is None


@pytest.mark.fixture
@pytest.mark.integration
@pytest.mark.training
def test_factor_film_fixture_forward_backward_optimizer_and_strict_reload() -> None:
    report = run_factor_film_fixture(_GIT_COMMIT)

    assert report["passed"] is True
    assert report["fixture_training_only"] is True
    assert report["finite_loss"] is True
    assert report["optimizer_step_completed"] is True
    assert report["processor_reload_validated"] is True
    assert report["checkpoint_reload_validated"] is True
    assert report["deterministic_inference_reload_validated"] is True
    assert report["gradient_evidence"] == {
        "base_act": True,
        "object_embedding": True,
        "bin_embedding": True,
        "object_film_projection": True,
        "bin_film_projection": True,
    }
    assert report["factor_film_training_completed"] is False
    assert report["factor_film_checkpoints_complete"] is False
    assert report["factor_film_checkpoint_selected"] is False
    assert report["development_benchmark_completed"] is False
    assert report["final_schedule_accessed"] is False
    assert report["smolvla_go"] is False
    assert report["physical_target_validated"] is False


@pytest.mark.fixture
@pytest.mark.integration
def test_factor_film_direct_save_refuses_mutation_of_existing_directory(
    tmp_path: Path,
) -> None:
    _, policy, _, _, _ = build_factor_film_policy_and_processors(
        training_config=_training_config(),
        statistics=_stats(),
        factor_film_config=FactorFiLMConfig(state_context_dimension=32),
    )
    destination = tmp_path / "pretrained"
    policy.save_pretrained(destination)

    def fingerprints() -> dict[str, str]:
        return {
            value.relative_to(destination).as_posix(): hashlib.sha256(
                value.read_bytes()
            ).hexdigest()
            for value in sorted(destination.rglob("*"))
            if value.is_file()
        }

    before = fingerprints()
    with torch.no_grad():
        policy.target_object_condition.embedding.weight.add_(1.0)
    with pytest.raises(FactorFiLMContractError, match="new or empty"):
        policy.save_pretrained(destination)
    assert fingerprints() == before


def _run_cli(tmp_path: Path, *arguments: str) -> tuple[subprocess.CompletedProcess[str], dict]:
    report = tmp_path / f"report-{len(tuple(tmp_path.glob('report-*.json')))}.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(_CLI),
            "--dataset-root",
            str(tmp_path / "dataset-source"),
            "--output-root",
            str(tmp_path / "model-output"),
            "--report",
            str(report),
            "--device",
            "cpu",
            *arguments,
        ],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = (
        json.loads(report.read_text(encoding="utf-8"))
        if report.is_file()
        else json.loads(completed.stdout.splitlines()[-1])
    )
    assert isinstance(payload, dict)
    return completed, payload


@pytest.mark.fixture
@pytest.mark.integration
def test_factor_film_cli_dry_run_and_failure_reports_are_truthful(tmp_path: Path) -> None:
    completed, report = _run_cli(tmp_path, "--dry-run", "--fixture-contract")
    assert completed.returncode == 0, completed.stderr
    assert report["passed"] is True
    assert report["training_mode"] == "dry_run_fixture_contract"
    assert report["dataset_accessed"] is False
    assert report["training_started"] is False
    assert report["factor_film_training_completed"] is False
    assert report["test_accessible"] is False
    assert report["historical_fresh_accessible"] is False
    assert report["final_schedule_accessed"] is False
    assert report["smolvla_go"] is False
    assert report["physical_target_validated"] is False

    failed, failure = _run_cli(tmp_path, "--fixture", "--fixture-contract")
    assert failed.returncode != 0
    assert failure["passed"] is False
    assert failure["error_type"] == "FactorFiLMContractError"
    assert failure["factor_film_training_completed"] is False
    assert failure["factor_film_checkpoint_selected"] is False
    assert failure["final_schedule_accessed"] is False
    assert failure["smolvla_go"] is False
    assert failure["physical_target_validated"] is False

    overlap_root = tmp_path / "protected-dataset"
    overlap, overlap_failure = _run_cli(
        tmp_path,
        "--dry-run",
        "--fixture-contract",
        "--dataset-root",
        str(overlap_root),
        "--output-root",
        str(overlap_root / "models"),
    )
    assert overlap.returncode != 0
    assert overlap_failure["passed"] is False
    assert overlap_failure["error_type"] == "FactorFiLMContractError"
    assert "must not contain or overlap" in overlap_failure["error_message"]

    unsafe_report = overlap_root / "must-not-be-created.json"
    unsafe, unsafe_failure = _run_cli(
        tmp_path,
        "--dry-run",
        "--fixture-contract",
        "--dataset-root",
        str(overlap_root),
        "--report",
        str(unsafe_report),
    )
    assert unsafe.returncode != 0
    assert unsafe_failure["passed"] is False
    assert unsafe_failure["report_written"] is False
    assert not unsafe_report.exists()

    tracked_report = _PROJECT_ROOT / "README.md"
    tracked_before = tracked_report.read_bytes()
    protected, protected_failure = _run_cli(
        tmp_path,
        "--dry-run",
        "--fixture-contract",
        "--report",
        str(tracked_report),
    )
    assert protected.returncode != 0
    assert protected_failure["passed"] is False
    assert protected_failure["report_written"] is False
    assert tracked_report.read_bytes() == tracked_before


@pytest.mark.fixture
@pytest.mark.integration
@pytest.mark.training
def test_factor_film_cli_fixture_runs_real_local_lifecycle(tmp_path: Path) -> None:
    completed, report = _run_cli(tmp_path, "--fixture")
    assert completed.returncode == 0, completed.stderr
    assert report["passed"] is True
    assert report["training_mode"] == "fixture"
    assert report["dataset_accessed"] is False
    assert report["fixture_training_only"] is True
    assert report["finite_loss"] is True
    assert all(report["gradient_evidence"].values())
    assert report["optimizer_step_completed"] is True
    assert report["checkpoint_reload_validated"] is True
    assert report["processor_reload_validated"] is True
    assert report["deterministic_inference_reload_validated"] is True
    assert report["factor_film_training_completed"] is False
    assert report["factor_film_checkpoints_complete"] is False
    assert report["factor_film_checkpoint_selected"] is False
    assert report["development_benchmark_completed"] is False
    assert report["final_schedule_accessed"] is False
    assert report["smolvla_go"] is False
    assert report["physical_target_validated"] is False
