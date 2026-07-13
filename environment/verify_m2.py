"""Verify M2 contracts and native-target expert acceptance.

The strict ``--target`` path first runs the M0 and M1 target gates.  Only after
both pass does it run a repeatable six-task smoke, a balanced 180-rollout
benchmark, and one rendered diagnostic rollout.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import os
import platform
import subprocess
import sys
import traceback
from collections import Counter
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

from langmani.environments.pick_place_by_instruction import ENV_ID
from langmani.environments.specs import BIN_IDS, OBJECT_IDS
from langmani.experts import (
    EXPERT_PHASE_SEQUENCE,
    PHASE_CONTRACTS,
    ExpertConfig,
    ExpertStatus,
)
from langmani.experts.planner import EXPECTED_MPLIB_VERSION, MplibPandaPlannerAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "diagnostics" / "m2"
REPORT_PATH = OUTPUT_DIR / "verification.json"
BENCHMARK_A_PATH = OUTPUT_DIR / "verification_benchmark_a.json"
BENCHMARK_B_PATH = OUTPUT_DIR / "verification_benchmark_b.json"
BALANCED_BENCHMARK_PATH = OUTPUT_DIR / "verification_balanced_180.json"
RENDERED_ROLLOUT_PATH = OUTPUT_DIR / "target_rendered" / "result.json"

TARGET_BALANCED_SEEDS = tuple(range(30))
TARGET_EPISODES_PER_TASK = 30
TARGET_BALANCED_EPISODES = len(OBJECT_IDS) * len(BIN_IDS) * TARGET_EPISODES_PER_TASK
TARGET_MINIMUM_OVERALL_SUCCESS_RATE = 0.95
TARGET_MINIMUM_TASK_SUCCESS_RATE = 0.90
CANONICAL_TASKS = tuple((object_id, bin_id) for object_id in OBJECT_IDS for bin_id in BIN_IDS)


@dataclass(slots=True)
class Report:
    """Collect explicit pass/fail/skip evidence for one verifier run."""

    checks: list[dict[str, Any]] = field(default_factory=list)

    def record(self, name: str, status: str, detail: str, *, required: bool = True) -> None:
        print(f"[{status.upper()}] {name}: {detail}")
        self.checks.append({"name": name, "status": status, "detail": detail, "required": required})

    def check(self, name: str, condition: bool, detail: str) -> None:
        self.record(name, "pass" if condition else "fail", detail)

    @property
    def failed(self) -> bool:
        return any(check["required"] and check["status"] == "fail" for check in self.checks)

    def write(self, *, target: bool) -> None:
        payload = {
            "schema_version": "langmani-m2-verification-v0",
            "environment_id": ENV_ID,
            "target_mode": target,
            "validation_scope": "native_target" if target else "non_target_diagnostic",
            "physical_acceptance": target and not self.failed,
            "checks": self.checks,
            "passed": not self.failed,
        }
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _is_native_linux() -> bool:
    release = platform.release().lower()
    is_wsl = "microsoft" in release or bool(os.environ.get("WSL_INTEROP"))
    return platform.system() == "Linux" and not is_wsl


def _run_command(report: Report, name: str, arguments: list[str]) -> bool:
    print(f"[INFO] {name}: {' '.join(arguments)}")
    completed = subprocess.run(arguments, cwd=PROJECT_ROOT, check=False)
    passed = completed.returncode == 0
    report.check(
        name,
        passed,
        f"command={' '.join(arguments)}; exit code {completed.returncode}",
    )
    return passed


def _run_balanced_benchmark(report: Report, arguments: list[str]) -> bool:
    """Run the statistical benchmark without requiring every rollout to pass.

    ``benchmark_expert.py`` deliberately returns one when any rollout fails. The
    target acceptance policy permits a bounded number of *classified* failures,
    so this boundary accepts exit code zero or one only when a fresh JSON report
    was produced. The report is then judged by the explicit M2 thresholds below.
    """
    BALANCED_BENCHMARK_PATH.unlink(missing_ok=True)
    print(f"[INFO] M2 balanced 180-episode benchmark: {' '.join(arguments)}")
    completed = subprocess.run(arguments, cwd=PROJECT_ROOT, check=False)
    report_written = BALANCED_BENCHMARK_PATH.is_file()
    completed_normally = completed.returncode in (0, 1) and report_written
    report.check(
        "M2 balanced 180-episode benchmark command",
        completed_normally,
        f"command={' '.join(arguments)}; exit code {completed.returncode}; "
        f"fresh report={report_written}",
    )
    return completed_normally


def _result_signature(result: dict[str, Any]) -> dict[str, Any]:
    phase_signature = [
        {
            key: phase[key]
            for key in (
                "phase",
                "success",
                "status",
                "attempts",
                "environment_steps",
                "planning_calls",
                "replans",
                "planner_status",
            )
        }
        for phase in result["phase_results"]
    ]
    return {
        key: result[key]
        for key in (
            "success",
            "status",
            "scene_seed",
            "scene_id",
            "task_id",
            "canonical_instruction",
            "target_object_id",
            "target_bin_id",
            "total_environment_steps",
            "total_planning_calls",
            "total_replans",
            "completed_phases",
            "failed_phase",
            "final_environment_evaluation",
        )
    } | {"phase_results": phase_signature}


def _compare_benchmarks(report: Report) -> None:
    payload_a = json.loads(BENCHMARK_A_PATH.read_text(encoding="utf-8"))
    payload_b = json.loads(BENCHMARK_B_PATH.read_text(encoding="utf-8"))
    results_a = payload_a.get("results", [])
    results_b = payload_b.get("results", [])
    expected_tasks = list(CANONICAL_TASKS)
    tasks_a = [
        (result.get("target_object_id"), result.get("target_bin_id")) for result in results_a
    ]
    tasks_b = [
        (result.get("target_object_id"), result.get("target_bin_id")) for result in results_b
    ]

    def is_valid_smoke_result(result: dict[str, Any]) -> bool:
        evaluation = result.get("final_environment_evaluation")
        return (
            result.get("success") is True
            and result.get("status") == ExpertStatus.SUCCESS.value
            and isinstance(evaluation, dict)
            and evaluation.get("success") is True
            and evaluation.get("wrong_object_in_target_bin") is not True
            and evaluation.get("target_in_wrong_bin") is not True
        )

    all_six_succeeded = (
        len(results_a) == len(results_b) == 6
        and payload_a.get("expected_rollouts") == payload_a.get("completed_rollouts") == 6
        and payload_b.get("expected_rollouts") == payload_b.get("completed_rollouts") == 6
        and not payload_a.get("command_errors")
        and not payload_b.get("command_errors")
        and tasks_a == tasks_b == expected_tasks
        and len({result.get("task_id") for result in results_a}) == 6
        and len({result.get("scene_id") for result in results_a}) == 1
        and all(is_valid_smoke_result(result) for result in results_a)
        and all(is_valid_smoke_result(result) for result in results_b)
    )
    report.check(
        "six-episode all-task smoke",
        all_six_succeeded,
        f"run A successes={sum(bool(r.get('success')) for r in results_a)}/6; "
        f"run B successes={sum(bool(r.get('success')) for r in results_b)}/6",
    )
    signatures_match = len(results_a) == len(results_b) and [
        _result_signature(result) for result in results_a
    ] == [_result_signature(result) for result in results_b]
    report.check(
        "repeatability signature",
        signatures_match,
        "identical seed/task runs match after excluding wall-clock durations and artifact paths",
    )


def _validate_balanced_benchmark(report: Report) -> None:
    """Apply the exact statistical and safety gates to the 180-rollout report."""
    payload = json.loads(BALANCED_BENCHMARK_PATH.read_text(encoding="utf-8"))
    results_value = payload.get("results")
    results = results_value if isinstance(results_value, list) else []
    command_errors_value = payload.get("command_errors")
    command_errors = command_errors_value if isinstance(command_errors_value, list) else []

    shape_valid = (
        isinstance(results_value, list)
        and isinstance(command_errors_value, list)
        and payload.get("schema_version") == "langmani-m2-benchmark-v0"
        and payload.get("environment_id") == ENV_ID
        and payload.get("seeds") == list(TARGET_BALANCED_SEEDS)
        and payload.get("expected_rollouts") == TARGET_BALANCED_EPISODES
        and payload.get("completed_rollouts") == TARGET_BALANCED_EPISODES
        and len(results) == TARGET_BALANCED_EPISODES
    )
    report.check(
        "balanced benchmark shape",
        shape_valid,
        f"results={len(results)}/{TARGET_BALANCED_EPISODES}; "
        f"seeds={len(payload.get('seeds', [])) if isinstance(payload.get('seeds'), list) else 0}"
        f"/{len(TARGET_BALANCED_SEEDS)}",
    )

    task_counts: Counter[tuple[object, object]] = Counter()
    success_counts: Counter[tuple[object, object]] = Counter()
    semantic_rows_valid = True
    successful_evaluations_valid = True
    wrong_object_successes = 0
    wrong_bin_successes = 0
    unclassified_failures = 0
    unexpected_exceptions = 0
    successful_rollouts = 0
    classified_failure_statuses = {
        status.value
        for status in ExpertStatus
        if status not in (ExpertStatus.SUCCESS, ExpertStatus.UNEXPECTED_EXCEPTION)
    }

    expected_rows = [
        (scene_seed, object_id, bin_id)
        for scene_seed in TARGET_BALANCED_SEEDS
        for object_id, bin_id in CANONICAL_TASKS
    ]
    for index, result_value in enumerate(results):
        if not isinstance(result_value, dict):
            semantic_rows_valid = False
            unclassified_failures += 1
            continue
        result = result_value
        task = (result.get("target_object_id"), result.get("target_bin_id"))
        task_counts[task] += 1
        success_value = result.get("success")
        success = success_value is True
        status = result.get("status")
        evaluation_value = result.get("final_environment_evaluation")
        evaluation = evaluation_value if isinstance(evaluation_value, dict) else {}
        semantic_rows_valid &= (
            isinstance(success_value, bool)
            and isinstance(status, str)
            and isinstance(evaluation_value, dict)
        )
        if (
            index >= len(expected_rows)
            or (
                result.get("scene_seed"),
                result.get("target_object_id"),
                result.get("target_bin_id"),
            )
            != expected_rows[index]
        ):
            semantic_rows_valid = False

        if success:
            successful_rollouts += 1
            success_counts[task] += 1
            successful_evaluations_valid &= (
                status == ExpertStatus.SUCCESS.value
                and evaluation.get("success") is True
                and evaluation.get("wrong_object_in_target_bin") is False
                and evaluation.get("target_in_wrong_bin") is False
            )
            wrong_object_successes += evaluation.get("wrong_object_in_target_bin") is True
            wrong_bin_successes += evaluation.get("target_in_wrong_bin") is True
        elif status not in classified_failure_statuses:
            unclassified_failures += 1

        if status == ExpertStatus.UNEXPECTED_EXCEPTION.value:
            unexpected_exceptions += 1

    expected_task_counts = {task: TARGET_EPISODES_PER_TASK for task in CANONICAL_TASKS}
    actual_task_counts = {task: task_counts.get(task, 0) for task in CANONICAL_TASKS}
    no_unknown_tasks = set(task_counts).issubset(CANONICAL_TASKS)
    report.check(
        "balanced TaskSpec counts",
        no_unknown_tasks and actual_task_counts == expected_task_counts,
        ", ".join(
            f"{object_id}/{bin_id}={actual_task_counts[(object_id, bin_id)]}"
            for object_id, bin_id in CANONICAL_TASKS
        ),
    )
    report.check(
        "balanced semantic and result schema",
        semantic_rows_valid
        and successful_evaluations_valid
        and payload.get("successful_rollouts") == successful_rollouts,
        "ordered rows match seed-major canonical TaskSpec order; successful results agree with "
        "M1 evaluation",
    )

    overall_rate = successful_rollouts / TARGET_BALANCED_EPISODES
    report.check(
        "balanced overall success rate",
        overall_rate >= TARGET_MINIMUM_OVERALL_SUCCESS_RATE,
        f"{successful_rollouts}/{TARGET_BALANCED_EPISODES}={overall_rate:.3%}; "
        f"required>={TARGET_MINIMUM_OVERALL_SUCCESS_RATE:.1%}",
    )
    task_rates = {
        task: success_counts.get(task, 0) / TARGET_EPISODES_PER_TASK for task in CANONICAL_TASKS
    }
    report.check(
        "balanced per-TaskSpec success rates",
        all(rate >= TARGET_MINIMUM_TASK_SUCCESS_RATE for rate in task_rates.values()),
        ", ".join(
            f"{object_id}/{bin_id}={task_rates[(object_id, bin_id)]:.1%}"
            for object_id, bin_id in CANONICAL_TASKS
        )
        + f"; required>={TARGET_MINIMUM_TASK_SUCCESS_RATE:.1%}",
    )
    report.check(
        "zero wrong-object successful completions",
        wrong_object_successes == 0,
        f"count={wrong_object_successes}",
    )
    report.check(
        "zero wrong-bin successful completions",
        wrong_bin_successes == 0,
        f"count={wrong_bin_successes}",
    )
    report.check(
        "zero unclassified failures",
        unclassified_failures == 0,
        f"count={unclassified_failures}",
    )
    missing_rollouts = max(0, TARGET_BALANCED_EPISODES - len(results))
    benchmark_crashes = len(command_errors) + unexpected_exceptions + missing_rollouts
    report.check(
        "zero benchmark crashes",
        benchmark_crashes == 0,
        f"command_errors={len(command_errors)}; unexpected_exceptions={unexpected_exceptions}; "
        f"missing_rollouts={missing_rollouts}",
    )


def _verify_rendered_rollout(report: Report) -> None:
    payload = json.loads(RENDERED_ROLLOUT_PATH.read_text(encoding="utf-8"))
    artifact_paths = payload.get("diagnostic_artifact_paths", [])
    output_root = (PROJECT_ROOT / "outputs").resolve()
    valid_paths: list[Path] = []
    for raw_path in artifact_paths:
        if not isinstance(raw_path, str):
            continue
        path = Path(raw_path).resolve()
        try:
            path.relative_to(output_root)
        except ValueError:
            continue
        if path.suffix.lower() == ".png" and path.is_file() and path.stat().st_size > 0:
            valid_paths.append(path)
    report.check(
        "M2 rendered diagnostic rollout",
        payload.get("success") is True
        and len(artifact_paths) == len(EXPERT_PHASE_SEQUENCE)
        and len(valid_paths) == len(EXPERT_PHASE_SEQUENCE),
        f"success={payload.get('success')!r}; valid phase PNGs="
        f"{len(valid_paths)}/{len(EXPERT_PHASE_SEQUENCE)}",
    )


def _structural_checks(report: Report) -> None:
    config = ExpertConfig()
    report.check(
        "expert control mode",
        config.control_mode == "pd_joint_pos",
        f"configured {config.control_mode!r}",
    )
    report.check(
        "twelve explicit phases",
        len(EXPERT_PHASE_SEQUENCE) == 12 and tuple(PHASE_CONTRACTS) == EXPERT_PHASE_SEQUENCE,
        ", ".join(phase.value for phase in EXPERT_PHASE_SEQUENCE),
    )
    adapter_source = inspect.getsource(MplibPandaPlannerAdapter)
    report.check(
        "planner adapter boundary",
        "mani_skill.examples" not in adapter_source and "plan_screw" in adapter_source,
        "project adapter uses direct lazy mplib screw planning and no ManiSkill example import",
    )
    installed_mplib = None
    with contextlib.suppress(metadata.PackageNotFoundError):
        installed_mplib = metadata.version("mplib")
    if _is_native_linux():
        report.check(
            "mplib version",
            installed_mplib == EXPECTED_MPLIB_VERSION,
            f"installed={installed_mplib!r}, expected={EXPECTED_MPLIB_VERSION!r}",
        )
    else:
        report.record(
            "mplib version",
            "skip",
            "mplib is a Linux-only ManiSkill dependency; no physical planner import attempted",
            required=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = Report()
    try:
        native_linux = _is_native_linux()
        prerequisites_passed = True
        if args.target:
            report.check(
                "native Linux target",
                native_linux,
                f"platform={platform.system()} release={platform.release()}",
            )
            if native_linux and not report.failed:
                prior_commands = (
                    (
                        "M0 target gate",
                        [sys.executable, "environment/verify_install.py", "--target"],
                    ),
                    (
                        "M1 target gate",
                        [sys.executable, "environment/verify_m1.py", "--target"],
                    ),
                )
                for name, command in prior_commands:
                    if not _run_command(report, name, command):
                        prerequisites_passed = False
                        break
            else:
                prerequisites_passed = False

        if prerequisites_passed:
            _structural_checks(report)
        elif args.target:
            report.record(
                "M2 structural and physical checks",
                "skip",
                "not started because a required platform or prior milestone gate failed",
                required=False,
            )

        if native_linux and prerequisites_passed and not report.failed:
            benchmark_commands = (
                (
                    "M2 benchmark run A",
                    [
                        sys.executable,
                        "environment/benchmark_expert.py",
                        "--seeds",
                        "0",
                        "--output",
                        str(BENCHMARK_A_PATH),
                    ],
                ),
                (
                    "M2 benchmark run B",
                    [
                        sys.executable,
                        "environment/benchmark_expert.py",
                        "--seeds",
                        "0",
                        "--output",
                        str(BENCHMARK_B_PATH),
                    ],
                ),
            )
            benchmark_passed = True
            for name, command in benchmark_commands:
                if not _run_command(report, name, command):
                    benchmark_passed = False
            if benchmark_passed:
                _compare_benchmarks(report)
            if args.target and not report.failed:
                balanced_command = [
                    sys.executable,
                    "environment/benchmark_expert.py",
                    "--seeds",
                    ",".join(str(seed) for seed in TARGET_BALANCED_SEEDS),
                    "--output",
                    str(BALANCED_BENCHMARK_PATH),
                ]
                if _run_balanced_benchmark(report, balanced_command):
                    _validate_balanced_benchmark(report)
            if args.target and not report.failed:
                rendered_command = [
                    sys.executable,
                    "environment/run_expert.py",
                    "--seed",
                    "0",
                    "--object-id",
                    "red_cube",
                    "--bin-id",
                    "left_bin",
                    "--diagnostic-rendering",
                    "--output",
                    str(RENDERED_ROLLOUT_PATH),
                ]
                if _run_command(report, "M2 rendered rollout", rendered_command):
                    _verify_rendered_rollout(report)
        elif not native_linux:
            report.record(
                "M2 physical rollouts",
                "skip",
                "native Linux is required; structural review only",
                required=args.target,
            )
    except Exception as error:  # noqa: BLE001 - verifier command boundary
        traceback.print_exc()
        report.record(
            "unexpected verifier exception",
            "fail",
            f"{type(error).__name__}: {error}",
        )
    report.write(target=args.target)
    print(f"[INFO] report: {REPORT_PATH}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
