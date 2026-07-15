"""Tests for the isolated M4 evaluation-worker throughput benchmark."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
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


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_recovered_phase(
    *,
    root: Path,
    source: Path,
    checkpoint_relative: str,
    phase: str,
    worker_count: int,
    gpu_ids: tuple[int, ...] = (0, 1),
) -> None:
    phase_root = root / phase
    phase_root.mkdir(parents=True)
    schedule_digest = "sha256:" + "a" * 64
    episodes = [
        {
            "scene_seed": index,
            "scene_id": f"scene-{index}",
            "task_id": "red_cube__left_bin",
            "status": "success",
            "success": True,
            "task_success": True,
            "episode_steps": 10,
            "target_in_wrong_bin": False,
            "wrong_object_in_target_bin": False,
            "target_off_table": False,
            "total_episode_duration_s": 1.0,
        }
        for index in range(benchmark_cli.EPISODES_PER_WORKER)
    ]
    for gpu_id in gpu_ids:
        for worker_index in range(worker_count):
            worker_root = phase_root / f"gpu{gpu_id}" / f"worker{worker_index:02d}"
            worker_root.mkdir(parents=True)
            clone_run = worker_root / "run"
            clone_run.mkdir()
            for name in ("run_manifest.json", "metrics.jsonl"):
                benchmark_cli.shutil.copy2(source / name, clone_run / name)
            clone_checkpoint = clone_run / checkpoint_relative
            clone_checkpoint.parent.mkdir(parents=True)
            benchmark_cli.shutil.copytree(
                source / checkpoint_relative,
                clone_checkpoint,
                copy_function=os.link,
            )
            output = worker_root / "run" / "evaluations" / "validation" / "fixture"
            _write_json(
                output / "benchmark.json",
                {
                    "passed": True,
                    "environment_action_applied": True,
                    "infrastructure_failure_count": 0,
                    "episodes": episodes,
                },
            )
            report = worker_root / "report.json"
            _write_json(
                report,
                {
                    "passed": True,
                    "split": "validation",
                    "run_fingerprint": "sha256:" + "b" * 64,
                    "checkpoint_fingerprint": "sha256:" + "c" * 64,
                    "schedule_digest": schedule_digest,
                    "episode_count": benchmark_cli.EPISODES_PER_WORKER,
                    "success_rate": 1.0,
                    "output": str(output.resolve()),
                    "physical_execution": True,
                    "infrastructure_failure_count": 0,
                    "action_bound_mode": "project",
                    "reused_existing_evaluation": False,
                },
            )
            future = time.time() + 5.0
            os.utime(report, (future, future))


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
    assert not os.path.samefile(source_weight, cloned_checkpoint / source_weight.name)

    audit = benchmark_cli._clone_source_equivalence(
        source_run=source,
        checkpoint_relative_path=checkpoint_relative,
        worker_root=worker_root,
        allow_historical_hardlinks=False,
    )
    assert audit["passed"] is True
    assert audit["source_mutation_isolation_validated"] is True

    cloned_weight = cloned_checkpoint / source_weight.name
    cloned_weight.write_bytes(b"different")
    with pytest.raises(RuntimeError, match="differs from source"):
        benchmark_cli._clone_source_equivalence(
            source_run=source,
            checkpoint_relative_path=checkpoint_relative,
            worker_root=worker_root,
            allow_historical_hardlinks=False,
        )
    assert source_weight.read_bytes() == b"immutable"


def test_cpu_quota_partitions_unmeasured_worker_counts(tmp_path: Path) -> None:
    cpu_max = tmp_path / "cpu.max"
    cpu_max.write_text("5000000 100000\n", encoding="utf-8")
    quota = benchmark_cli._cpu_quota_cores(cpu_max)
    eligible, skipped = benchmark_cli._partition_worker_counts(
        worker_counts=(1, 2, 4, 6),
        gpu_ids=(0, 1),
        cpu_quota_cores=quota,
        estimated_cpu_cores_per_worker=12.0,
    )

    assert quota == 50.0
    assert eligible == (1, 2)
    assert [item["worker_count_per_gpu"] for item in skipped] == [4, 6]
    assert all(item["measured"] is False for item in skipped)
    assert all(item["reason_code"] == "cpu_cgroup_quota_exceeded" for item in skipped)
    assert skipped[0]["estimated_total_cpu_cores"] == 96.0
    assert skipped[1]["estimated_total_cpu_cores"] == 144.0


def test_recover_completed_phase_preserves_unknown_return_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    checkpoint_relative = "checkpoints/step-00000001-selected"
    checkpoint = source / checkpoint_relative
    checkpoint.mkdir(parents=True)
    (source / "run_manifest.json").write_text("{}", encoding="utf-8")
    (source / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"immutable")
    output_root = tmp_path / "benchmark"
    _make_recovered_phase(
        root=output_root,
        source=source,
        checkpoint_relative=checkpoint_relative,
        phase="timed-n1",
        worker_count=1,
    )
    monkeypatch.setattr(
        benchmark_cli,
        "_path_creation_time",
        lambda _path: (time.time(), "test_birth_time"),
    )

    recovered = benchmark_cli._recover_completed_phase(
        phase="timed-n1",
        worker_count=1,
        gpu_ids=(0, 1),
        source_run=source,
        checkpoint_relative_path=checkpoint_relative,
        output_root=output_root,
        expected_run_fingerprint="sha256:" + "b" * 64,
        expected_checkpoint_fingerprint="sha256:" + "c" * 64,
    )

    assert recovered["passed"] is False
    assert recovered["artifact_measurements_validated"] is True
    assert recovered["measurement_validated"] is False
    assert recovered["recovered_from_existing_artifacts"] is True
    assert recovered["subprocess_return_codes_observed"] is False
    assert recovered["clone_source_equivalence_validated"] is True
    assert recovered["source_mutation_isolation_validated"] is False
    assert recovered["wall_duration_s"] > 0
    assert all(worker["wall_duration_s"] is None for worker in recovered["workers"])
    assert all(worker["return_code"] is None for worker in recovered["workers"])
    assert all(worker["return_code_observed"] is False for worker in recovered["workers"])
    assert all(
        worker["completion_provenance"] == "passed_report_recovery"
        for worker in recovered["workers"]
    )


def test_recover_completed_phase_rejects_output_outside_worker_clone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    checkpoint_relative = "checkpoints/step-00000001-selected"
    checkpoint = source / checkpoint_relative
    checkpoint.mkdir(parents=True)
    (source / "run_manifest.json").write_text("{}", encoding="utf-8")
    (source / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"immutable")
    output_root = tmp_path / "benchmark"
    _make_recovered_phase(
        root=output_root,
        source=source,
        checkpoint_relative=checkpoint_relative,
        phase="timed-n1",
        worker_count=1,
    )
    escaped = tmp_path / "escaped"
    _write_json(escaped / "benchmark.json", {"passed": True})
    report_path = output_root / "timed-n1" / "gpu0" / "worker00" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["output"] = str(escaped.resolve())
    _write_json(report_path, report)
    monkeypatch.setattr(
        benchmark_cli,
        "_path_creation_time",
        lambda _path: (time.time(), "test_birth_time"),
    )

    with pytest.raises(RuntimeError, match="escaped its clone"):
        benchmark_cli._recover_completed_phase(
            phase="timed-n1",
            worker_count=1,
            gpu_ids=(0, 1),
            source_run=source,
            checkpoint_relative_path=checkpoint_relative,
            output_root=output_root,
            expected_run_fingerprint="sha256:" + "b" * 64,
            expected_checkpoint_fingerprint="sha256:" + "c" * 64,
        )


def test_missing_runner_repeat_keeps_recommendation_nonfinal() -> None:
    original = {
        "passed": True,
        "worker_count_per_gpu": 2,
        "worst_gpu_episodes_per_minute": 1.0,
        "semantic_digest": "sha256:" + "d" * 64,
    }
    record = benchmark_cli._stability_record(
        worker_count=2,
        original=original,
        repeat=None,
        repeat_source=None,
    )

    assert record["completed"] is False
    assert record["stable"] is False


def test_skip_retest_is_measurement_only_even_when_retests_are_stable() -> None:
    stable_retests = [
        {"worker_count_per_gpu": 1, "stable": True},
        {"worker_count_per_gpu": 2, "stable": True},
    ]

    assert (
        benchmark_cli._selection_stability_validated(
            retests=stable_retests,
            winner=1,
            runner=2,
            skip_retest=True,
        )
        is False
    )
    assert (
        benchmark_cli._selection_stability_validated(
            retests=stable_retests,
            winner=1,
            runner=2,
            skip_retest=False,
        )
        is True
    )


def test_creation_time_rejects_unreliable_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Stat:
        st_ctime = 123.0

    class _Path:
        def stat(self) -> _Stat:
            return _Stat()

        def __str__(self) -> str:
            return "fixture"

    monkeypatch.setattr(benchmark_cli.sys, "platform", "unsupported")
    with pytest.raises(RuntimeError, match="birth time is unavailable"):
        benchmark_cli._path_creation_time(_Path())
