"""CPU-safe command-boundary tests for the M1 verifier."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_verifier(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "environment/verify_m1.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_cpu_and_gpu_physx_workers_use_separate_subprocesses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _load_verifier("langmani_test_verify_m1_workers")
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def completed_worker(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        worker = command[command.index("--_worker") + 1]
        output_path = Path(command[command.index("--_worker-output") + 1])
        output_path.write_text(
            json.dumps(
                {
                    "checks": [
                        {
                            "name": f"{worker} simulator check",
                            "status": "pass",
                            "detail": "isolated",
                            "required": True,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout=f"{worker} output\n", stderr="")

    monkeypatch.setattr(verifier.subprocess, "run", completed_worker)
    report = verifier.Report()
    verifier.run_isolated_worker(report, "cpu", seed=123, verbose=False)
    verifier.run_isolated_worker(report, "gpu", seed=123, verbose=True)

    assert [command[command.index("--_worker") + 1] for command, _ in calls] == ["cpu", "gpu"]
    assert calls[0][0][0] == sys.executable
    assert "--verbose" not in calls[0][0]
    assert "--verbose" in calls[1][0]
    assert all(kwargs["timeout"] == verifier.WORKER_TIMEOUT_SECONDS for _, kwargs in calls)
    assert all(kwargs["env"]["PYTHONUNBUFFERED"] == "1" for _, kwargs in calls)
    assert {check["name"] for check in report.checks} == {
        "cpu simulator check",
        "CPU PhysX worker process",
        "gpu simulator check",
        "GPU PhysX worker process",
    }
    assert not report.failed


def test_missing_worker_result_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    verifier = _load_verifier("langmani_test_verify_m1_missing_worker_result")

    def failed_worker(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 7, stdout="", stderr="worker crashed")

    monkeypatch.setattr(verifier.subprocess, "run", failed_worker)
    report = verifier.Report()
    verifier.run_isolated_worker(report, "gpu", seed=123, verbose=False)

    assert report.failed
    assert report.checks[-1] == {
        "name": "GPU PhysX worker process",
        "status": "fail",
        "detail": "exit code 7; worker result file was not written",
        "required": True,
    }


def test_malformed_worker_result_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    verifier = _load_verifier("langmani_test_verify_m1_malformed_worker_result")

    def malformed_worker(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        output_path = Path(command[command.index("--_worker-output") + 1])
        output_path.write_text('{"checks": "not-a-list"}', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(verifier.subprocess, "run", malformed_worker)
    report = verifier.Report()
    verifier.run_isolated_worker(report, "cpu", seed=123, verbose=False)

    assert report.failed
    assert report.checks[-1]["name"] == "CPU PhysX worker process"
    assert report.checks[-1]["detail"].startswith(
        "malformed worker result: TypeError: worker result checks"
    )


def test_target_main_schedules_cpu_and_gpu_workers_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _load_verifier("langmani_test_verify_m1_main_workers")
    args = argparse.Namespace(
        target=True,
        seed=123,
        verbose=False,
        _worker=None,
        _worker_output=None,
    )
    workers: list[str] = []

    monkeypatch.setattr(verifier, "parse_args", lambda: args)
    monkeypatch.setattr(verifier, "run_contract_checks", lambda report: None)
    monkeypatch.setattr(verifier, "native_linux", lambda: True)
    monkeypatch.setattr(verifier.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(verifier.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(verifier.Report, "write", lambda self, *, target: None)

    def record_worker(
        report: Any,
        worker: str,
        *,
        seed: int,
        verbose: bool,
    ) -> None:
        del seed, verbose
        workers.append(worker)
        report.check(f"{worker} worker", True, "scheduled")

    monkeypatch.setattr(verifier, "run_isolated_worker", record_worker)

    assert verifier.main() == 0
    assert workers == ["cpu", "gpu"]
