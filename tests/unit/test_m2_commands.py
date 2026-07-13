"""CPU-safe command-boundary tests for M2 diagnostics and acceptance."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from langmani.experts.command_support import validated_output_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = (PROJECT_ROOT / "outputs").resolve()


def _load_script(module_name: str, relative_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, PROJECT_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_diagnostic_output_is_confined_to_outputs_and_json() -> None:
    path = validated_output_path(
        "outputs/diagnostics/m2/result.json",
        output_root=OUTPUT_ROOT,
    )
    path.relative_to(OUTPUT_ROOT)

    with pytest.raises(ValueError, match="must stay under"):
        validated_output_path("outputs/../README.md", output_root=OUTPUT_ROOT)
    with pytest.raises(ValueError, match=".json suffix"):
        validated_output_path(
            "outputs/diagnostics/m2/result.txt",
            output_root=OUTPUT_ROOT,
        )


def test_benchmark_seed_parser_rejects_ambiguous_lists() -> None:
    benchmark = _load_script("langmani_test_benchmark_expert", "environment/benchmark_expert.py")

    assert benchmark._parse_seeds("0,2,5") == (0, 2, 5)
    for malformed in ("", "0,,1", ",0", "0,", "-1", "1,1", "one"):
        with pytest.raises(argparse.ArgumentTypeError):
            benchmark._parse_seeds(malformed)


def test_benchmark_fatal_error_is_not_fabricated_as_a_task_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = _load_script(
        "langmani_test_benchmark_fatal",
        "environment/benchmark_expert.py",
    )
    output = OUTPUT_ROOT / "diagnostics" / "m2" / "test_benchmark.json"
    args = SimpleNamespace(
        seeds=(0,),
        sim_backend="physx_cpu",
        diagnostic_rendering=False,
        output=output,
    )
    captured: dict[str, Any] = {}
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    monkeypatch.setattr(
        benchmark,
        "create_expert_environment",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("environment failed")),
    )
    monkeypatch.setattr(benchmark.traceback, "print_exc", lambda: None)
    monkeypatch.setattr(
        benchmark,
        "write_json",
        lambda path, payload, **kwargs: captured.update(payload=payload) or Path(path),
    )

    assert benchmark.main() == 1
    payload = captured["payload"]
    assert payload["expected_rollouts"] == 6
    assert payload["completed_rollouts"] == 0
    assert payload["results"] == []
    assert payload["command_errors"] == [{"type": "RuntimeError", "message": "environment failed"}]


def test_target_verifier_runs_prior_gates_before_any_m2_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _load_script("langmani_test_verify_order", "environment/verify_m2.py")
    events: list[str] = []
    monkeypatch.setattr(verifier, "parse_args", lambda: SimpleNamespace(target=True))
    monkeypatch.setattr(verifier, "_is_native_linux", lambda: True)
    monkeypatch.setattr(
        verifier,
        "_run_command",
        lambda report, name, arguments: events.append(name) or True,
    )
    monkeypatch.setattr(
        verifier,
        "_structural_checks",
        lambda report: events.append("M2 structural checks"),
    )
    monkeypatch.setattr(
        verifier,
        "_compare_benchmarks",
        lambda report: events.append("M2 comparison"),
    )
    monkeypatch.setattr(
        verifier,
        "_run_balanced_benchmark",
        lambda report, arguments: events.append("M2 balanced benchmark") or True,
    )
    monkeypatch.setattr(
        verifier,
        "_validate_balanced_benchmark",
        lambda report: events.append("M2 balanced validation"),
    )
    monkeypatch.setattr(
        verifier,
        "_verify_rendered_rollout",
        lambda report: events.append("M2 rendered verification"),
    )
    monkeypatch.setattr(verifier.Report, "write", lambda self, target: None)

    assert verifier.main() == 0
    assert events == [
        "M0 target gate",
        "M1 target gate",
        "M2 structural checks",
        "M2 benchmark run A",
        "M2 benchmark run B",
        "M2 comparison",
        "M2 balanced benchmark",
        "M2 balanced validation",
        "M2 rendered rollout",
        "M2 rendered verification",
    ]


def test_target_verifier_does_not_start_m2_after_failed_prior_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _load_script("langmani_test_verify_stop", "environment/verify_m2.py")
    events: list[str] = []
    monkeypatch.setattr(verifier, "parse_args", lambda: SimpleNamespace(target=True))
    monkeypatch.setattr(verifier, "_is_native_linux", lambda: True)

    def fail_first(report: Any, name: str, arguments: list[str]) -> bool:
        del arguments
        events.append(name)
        report.check(name, False, "deliberate prior-gate failure")
        return False

    monkeypatch.setattr(verifier, "_run_command", fail_first)
    monkeypatch.setattr(
        verifier,
        "_structural_checks",
        lambda report: events.append("M2 structural checks"),
    )
    monkeypatch.setattr(verifier.Report, "write", lambda self, target: None)

    assert verifier.main() == 1
    assert events == ["M0 target gate"]


def test_batch_benchmark_collects_one_unexpected_failure_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = _load_script(
        "langmani_test_benchmark_continue",
        "environment/benchmark_expert.py",
    )
    output = OUTPUT_ROOT / "diagnostics" / "m2" / "test_continue.json"
    args = SimpleNamespace(
        seeds=(0,),
        sim_backend="physx_cpu",
        diagnostic_rendering=False,
        output=output,
    )
    attempts: list[dict[str, Any]] = []
    captured: dict[str, Any] = {}

    class FakeEnv:
        def reset(self, *, seed: int, options: dict[str, Any]) -> None:
            self.seed = seed
            self.options = options
            attempts.append(options)

        def close(self) -> None:
            return None

    class CompactResult:
        def __init__(self, status: str, *, exception: Exception | None = None) -> None:
            self.success = status == "success"
            self.status = SimpleNamespace(value=status)
            self.exception = exception

        def to_dict(self) -> dict[str, Any]:
            return {
                "success": self.success,
                "status": self.status.value,
                "exception_type": (
                    type(self.exception).__name__ if self.exception is not None else None
                ),
                "exception_message": str(self.exception) if self.exception is not None else None,
            }

    class FakeExpert:
        calls = 0

        def __init__(self, env: FakeEnv, **kwargs: object) -> None:
            del env, kwargs

        def run(self) -> CompactResult:
            type(self).calls += 1
            if type(self).calls == 1:
                raise ArithmeticError("first rollout failed unexpectedly")
            return CompactResult("success")

        def unexpected_exception_result(self, error: Exception) -> CompactResult:
            return CompactResult("unexpected_exception", exception=error)

    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    monkeypatch.setattr(benchmark, "create_expert_environment", lambda **kwargs: FakeEnv())
    monkeypatch.setattr(benchmark, "PickPlaceExpert", FakeExpert)
    monkeypatch.setattr(benchmark.traceback, "print_exc", lambda: None)
    monkeypatch.setattr(
        benchmark,
        "write_json",
        lambda path, payload, **kwargs: captured.update(payload=payload) or Path(path),
    )

    assert benchmark.main() == 1
    payload = captured["payload"]
    assert len(attempts) == 6
    assert payload["completed_rollouts"] == 6
    assert payload["successful_rollouts"] == 5
    assert len(payload["results"]) == 6
    assert payload["results"][0]["status"] == "unexpected_exception"
    assert payload["results"][0]["exception_type"] == "ArithmeticError"
    assert payload["results"][0]["exception_message"] == "first rollout failed unexpectedly"
    assert payload["status_counts"] == {"success": 5, "unexpected_exception": 1}
    assert payload["command_errors"] == []


def test_single_command_preserves_unexpected_exception_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _load_script("langmani_test_run_expert", "environment/run_expert.py")
    output = OUTPUT_ROOT / "diagnostics" / "m2" / "test_single.json"
    args = SimpleNamespace(
        seed=0,
        object_id="red_cube",
        bin_id="left_bin",
        sim_backend="physx_cpu",
        diagnostic_rendering=False,
        output=output,
    )
    captured: dict[str, Any] = {}

    class FakeEnv:
        def reset(self, **kwargs: object) -> None:
            del kwargs

        def close(self) -> None:
            return None

    class CompactResult:
        success = False
        status = SimpleNamespace(value="unexpected_exception")

        def to_dict(self) -> dict[str, Any]:
            return {
                "status": "unexpected_exception",
                "exception_type": "ArithmeticError",
                "exception_message": "single rollout defect",
            }

    class FakeExpert:
        def __init__(self, env: FakeEnv, **kwargs: object) -> None:
            del env, kwargs

        def run(self) -> CompactResult:
            raise ArithmeticError("single rollout defect")

        def unexpected_exception_result(self, error: Exception) -> CompactResult:
            assert isinstance(error, ArithmeticError)
            return CompactResult()

    monkeypatch.setattr(command, "parse_args", lambda: args)
    monkeypatch.setattr(command, "create_expert_environment", lambda **kwargs: FakeEnv())
    monkeypatch.setattr(command, "PickPlaceExpert", FakeExpert)
    monkeypatch.setattr(command.traceback, "print_exc", lambda: None)
    monkeypatch.setattr(
        command,
        "write_json",
        lambda path, payload, **kwargs: captured.update(payload=payload) or Path(path),
    )

    assert command.main() == 1
    assert captured["payload"] == {
        "status": "unexpected_exception",
        "exception_type": "ArithmeticError",
        "exception_message": "single rollout defect",
    }


def test_target_comparison_requires_the_exact_six_unique_semantic_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _load_script("langmani_test_verify_matrix", "environment/verify_m2.py")
    path_a = OUTPUT_ROOT / "diagnostics" / "m2" / "test_compare_a.json"
    path_b = OUTPUT_ROOT / "diagnostics" / "m2" / "test_compare_b.json"
    path_a.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(verifier, "BENCHMARK_A_PATH", path_a)
    monkeypatch.setattr(verifier, "BENCHMARK_B_PATH", path_b)

    def result(object_id: str, bin_id: str, index: int) -> dict[str, Any]:
        return {
            "success": True,
            "status": "success",
            "scene_seed": 0,
            "scene_id": "scene:0",
            "task_id": f"task:{index}",
            "canonical_instruction": f"{object_id}/{bin_id}",
            "target_object_id": object_id,
            "target_bin_id": bin_id,
            "total_environment_steps": 10,
            "total_planning_calls": 6,
            "total_replans": 0,
            "completed_phases": ["initialize"],
            "failed_phase": None,
            "final_environment_evaluation": {"success": True},
            "phase_results": [],
        }

    duplicates = [result("red_cube", "left_bin", index) for index in range(6)]
    duplicate_payload = {
        "expected_rollouts": 6,
        "completed_rollouts": 6,
        "command_errors": [],
        "results": duplicates,
    }
    path_a.write_text(json.dumps(duplicate_payload), encoding="utf-8")
    path_b.write_text(json.dumps(duplicate_payload), encoding="utf-8")
    duplicate_report = verifier.Report()

    verifier._compare_benchmarks(duplicate_report)

    assert duplicate_report.failed
    matrix = [
        (object_id, bin_id)
        for object_id in ("red_cube", "green_cube", "blue_cube")
        for bin_id in ("left_bin", "right_bin")
    ]
    exact_results = [
        result(object_id, bin_id, index) for index, (object_id, bin_id) in enumerate(matrix)
    ]
    exact_payload = {
        "expected_rollouts": 6,
        "completed_rollouts": 6,
        "command_errors": [],
        "results": exact_results,
    }
    path_a.write_text(json.dumps(exact_payload), encoding="utf-8")
    path_b.write_text(json.dumps(exact_payload), encoding="utf-8")
    exact_report = verifier.Report()

    verifier._compare_benchmarks(exact_report)

    assert not exact_report.failed


def _balanced_payload(verifier: ModuleType) -> dict[str, Any]:
    results = []
    for scene_seed in verifier.TARGET_BALANCED_SEEDS:
        for task_index, (object_id, bin_id) in enumerate(verifier.CANONICAL_TASKS):
            results.append(
                {
                    "success": True,
                    "status": "success",
                    "scene_seed": scene_seed,
                    "scene_id": f"scene:{scene_seed}",
                    "task_id": f"task:{task_index}",
                    "target_object_id": object_id,
                    "target_bin_id": bin_id,
                    "final_environment_evaluation": {
                        "success": True,
                        "wrong_object_in_target_bin": False,
                        "target_in_wrong_bin": False,
                    },
                }
            )
    return {
        "schema_version": "langmani-m2-benchmark-v0",
        "environment_id": verifier.ENV_ID,
        "seeds": list(verifier.TARGET_BALANCED_SEEDS),
        "expected_rollouts": verifier.TARGET_BALANCED_EPISODES,
        "completed_rollouts": verifier.TARGET_BALANCED_EPISODES,
        "successful_rollouts": verifier.TARGET_BALANCED_EPISODES,
        "command_errors": [],
        "results": results,
    }


def _mark_classified_failures(payload: dict[str, Any], per_task: tuple[int, ...]) -> None:
    marked = [0] * len(per_task)
    for index, result in enumerate(payload["results"]):
        task_index = index % len(per_task)
        if marked[task_index] >= per_task[task_index]:
            continue
        result["success"] = False
        result["status"] = "planning_failure"
        result["final_environment_evaluation"]["success"] = False
        marked[task_index] += 1
    payload["successful_rollouts"] = sum(result["success"] for result in payload["results"])


def _check_status(report: Any, name: str) -> str:
    return next(check["status"] for check in report.checks if check["name"] == name)


def test_balanced_target_accepts_inclusive_success_thresholds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    verifier = _load_script("langmani_test_verify_balanced_pass", "environment/verify_m2.py")
    benchmark_path = tmp_path / "balanced.json"
    monkeypatch.setattr(verifier, "BALANCED_BENCHMARK_PATH", benchmark_path)
    payload = _balanced_payload(verifier)
    # 9 classified failures: exactly 95% overall, and the first TaskSpec is
    # exactly 90%. Every other TaskSpec remains above 90%.
    _mark_classified_failures(payload, (3, 2, 1, 1, 1, 1))
    benchmark_path.write_text(json.dumps(payload), encoding="utf-8")
    report = verifier.Report()

    verifier._validate_balanced_benchmark(report)

    assert not report.failed
    assert _check_status(report, "balanced benchmark shape") == "pass"
    assert _check_status(report, "balanced TaskSpec counts") == "pass"
    assert _check_status(report, "balanced overall success rate") == "pass"
    assert _check_status(report, "balanced per-TaskSpec success rates") == "pass"


def test_balanced_command_requires_a_fresh_report_but_allows_classified_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    verifier = _load_script("langmani_test_verify_balanced_command", "environment/verify_m2.py")
    benchmark_path = tmp_path / "balanced.json"
    monkeypatch.setattr(verifier, "BALANCED_BENCHMARK_PATH", benchmark_path)

    def completed_with_classified_failures(*args: object, **kwargs: object) -> Any:
        del args, kwargs
        benchmark_path.write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(verifier.subprocess, "run", completed_with_classified_failures)
    report = verifier.Report()
    assert verifier._run_balanced_benchmark(report, ["python", "benchmark.py"])
    assert not report.failed

    # A stale artifact must not make a crashed/non-writing command look complete.
    monkeypatch.setattr(
        verifier.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1),
    )
    stale_report = verifier.Report()
    assert not verifier._run_balanced_benchmark(stale_report, ["python", "benchmark.py"])
    assert stale_report.failed


def test_balanced_target_rejects_each_required_safety_and_reliability_violation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    verifier = _load_script("langmani_test_verify_balanced_fail", "environment/verify_m2.py")
    benchmark_path = tmp_path / "balanced.json"
    monkeypatch.setattr(verifier, "BALANCED_BENCHMARK_PATH", benchmark_path)

    def validate(payload: dict[str, Any]) -> Any:
        benchmark_path.write_text(json.dumps(payload), encoding="utf-8")
        report = verifier.Report()
        verifier._validate_balanced_benchmark(report)
        return report

    incomplete = _balanced_payload(verifier)
    incomplete["results"].pop()
    incomplete["completed_rollouts"] -= 1
    incomplete["successful_rollouts"] -= 1
    incomplete_report = validate(incomplete)
    assert _check_status(incomplete_report, "balanced benchmark shape") == "fail"
    assert _check_status(incomplete_report, "balanced TaskSpec counts") == "fail"
    assert _check_status(incomplete_report, "zero benchmark crashes") == "fail"

    low_overall = _balanced_payload(verifier)
    _mark_classified_failures(low_overall, (2, 2, 2, 2, 1, 1))
    assert _check_status(validate(low_overall), "balanced overall success rate") == "fail"

    low_task = _balanced_payload(verifier)
    _mark_classified_failures(low_task, (4, 0, 0, 0, 0, 0))
    assert _check_status(validate(low_task), "balanced per-TaskSpec success rates") == "fail"

    wrong_object = _balanced_payload(verifier)
    wrong_object["results"][0]["final_environment_evaluation"]["wrong_object_in_target_bin"] = True
    assert (
        _check_status(validate(wrong_object), "zero wrong-object successful completions") == "fail"
    )

    wrong_bin = _balanced_payload(verifier)
    wrong_bin["results"][0]["final_environment_evaluation"]["target_in_wrong_bin"] = True
    assert _check_status(validate(wrong_bin), "zero wrong-bin successful completions") == "fail"

    unclassified = _balanced_payload(verifier)
    unclassified["results"][0]["success"] = False
    unclassified["results"][0]["status"] = "unexpected_exception"
    unclassified["results"][0]["final_environment_evaluation"]["success"] = False
    unclassified["successful_rollouts"] -= 1
    unclassified_report = validate(unclassified)
    assert _check_status(unclassified_report, "zero unclassified failures") == "fail"
    assert _check_status(unclassified_report, "zero benchmark crashes") == "fail"

    crashed = _balanced_payload(verifier)
    crashed["command_errors"] = [{"type": "RuntimeError", "message": "benchmark crashed"}]
    assert _check_status(validate(crashed), "zero benchmark crashes") == "fail"
