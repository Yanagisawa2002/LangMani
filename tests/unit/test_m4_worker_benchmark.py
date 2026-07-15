"""Tests for the isolated M4 evaluation-worker throughput benchmark."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "langmani_test_benchmark_act_workers",
        PROJECT_ROOT / "scripts" / "benchmark_act_evaluation_workers.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark_cli = _load()


def test_worker_selection_uses_worst_gpu_rate_and_five_percent_tie_break() -> None:
    phases = [
        {"passed": True, "worker_count_per_gpu": 1, "worst_gpu_episodes_per_minute": 10.0},
        {"passed": True, "worker_count_per_gpu": 2, "worst_gpu_episodes_per_minute": 19.2},
        {"passed": True, "worker_count_per_gpu": 4, "worst_gpu_episodes_per_minute": 20.0},
        {"passed": False, "worker_count_per_gpu": 6, "worst_gpu_episodes_per_minute": 25.0},
    ]
    winner, runner = benchmark_cli.select_worker_count(phases)
    assert winner == 2
    assert runner == 4


def test_worker_selection_rejects_when_no_physical_phase_passes() -> None:
    with pytest.raises(RuntimeError, match="no worker-count phase"):
        benchmark_cli.select_worker_count(
            [
                {
                    "passed": False,
                    "worker_count_per_gpu": 1,
                    "worst_gpu_episodes_per_minute": 10.0,
                }
            ]
        )


def test_worker_clone_contains_only_training_identity_and_selected_checkpoint(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    checkpoint_relative = "checkpoints/step-00000001-selected"
    checkpoint = source / checkpoint_relative
    checkpoint.mkdir(parents=True)
    (source / "run_manifest.json").write_text("{}", encoding="utf-8")
    (source / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
    (source / "checkpoint_selection.json").write_text("{}", encoding="utf-8")
    (source / "complete.json").write_text("{}", encoding="utf-8")
    source_weight = checkpoint / "model.safetensors"
    source_weight.write_bytes(b"immutable")
    worker_root = tmp_path / "worker"
    worker_root.mkdir()

    cloned_checkpoint = benchmark_cli._clone_worker_run(
        source_run=source,
        checkpoint_relative_path=checkpoint_relative,
        worker_root=worker_root,
    )

    run = worker_root / "run"
    assert cloned_checkpoint == run / checkpoint_relative
    assert (run / "run_manifest.json").is_file()
    assert (run / "metrics.jsonl").is_file()
    assert not (run / "checkpoint_selection.json").exists()
    assert not (run / "complete.json").exists()
    assert os.stat(source_weight).st_ino == os.stat(cloned_checkpoint / source_weight.name).st_ino
