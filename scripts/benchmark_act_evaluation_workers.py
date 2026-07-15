"""Benchmark independent ACT validation workers without touching final M4 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from langmani.policies.act_runtime import atomic_write_json
from langmani.policies.act_types import ActExperimentManifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "outputs" / "datasets" / "m3b" / "langmani-pick-place-lerobot-v1"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "benchmarks" / "m4_eval_worker_scaling"
EPISODES_PER_WORKER = 6
BENCHMARK_SCHEMA_VERSION = "langmani-m4-evaluation-worker-benchmark-v2"


@dataclass(frozen=True, slots=True)
class WorkerResult:
    phase: str
    worker_count_per_gpu: int
    gpu_id: int
    worker_index: int
    return_code: int | None
    return_code_observed: bool
    completion_provenance: str
    wall_duration_s: float | None
    report_path: str
    output_path: str | None
    artifact_measurement_validated: bool
    passed: bool
    physical_execution: bool
    infrastructure_failure_count: int
    episode_count: int
    success_rate: float | None
    schedule_digest: str | None
    semantic_digest: str | None
    summed_episode_duration_s: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--worker-counts", type=int, nargs="+", default=(1, 2, 4, 6))
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--skip-retest", action="store_true")
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Recover an immutable summary from already completed phase directories.",
    )
    parser.add_argument(
        "--estimated-cpu-cores-per-worker",
        type=float,
        help=(
            "Observed CPU demand used with the cgroup cpu.max quota to skip infeasible "
            "worker counts before execution or recovered summarization."
        ),
    )
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_snapshot(source_run: Path, checkpoint_relative_path: str) -> dict[str, str]:
    paths = [
        source_run / "run_manifest.json",
        source_run / "metrics.jsonl",
        source_run / "checkpoint_selection.json",
        source_run / "complete.json",
        *(path for path in (source_run / checkpoint_relative_path).rglob("*") if path.is_file()),
    ]
    if any(not path.is_file() for path in paths):
        raise RuntimeError("source run lacks required immutable benchmark inputs")
    return {path.relative_to(source_run).as_posix(): _sha256_file(path) for path in sorted(paths)}


def _selected_checkpoint(source_run: Path) -> tuple[ActExperimentManifest, str]:
    manifest = ActExperimentManifest.from_dict(_read(source_run / "run_manifest.json"))
    selection = _read(source_run / "checkpoint_selection.json")
    selected = selection.get("selected_checkpoint_fingerprint")
    if (
        not manifest.complete
        or not manifest.training_state.completed
        or selected != manifest.selected_checkpoint_fingerprint
    ):
        raise RuntimeError("worker benchmark requires one completed selection-locked source run")
    records = [item for item in manifest.checkpoints if item.checkpoint_fingerprint == selected]
    if len(records) != 1 or not records[0].complete:
        raise RuntimeError("selected checkpoint is missing or not immutable")
    return manifest, records[0].relative_path


def _cpu_quota_cores(
    cpu_max_path: Path = Path("/sys/fs/cgroup/cpu.max"),
) -> float | None:
    """Return the cgroup-v2 CPU quota in cores, or None when it is unlimited/unavailable."""
    if not cpu_max_path.is_file():
        return None
    fields = cpu_max_path.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2:
        raise RuntimeError(f"malformed cgroup cpu.max: {cpu_max_path}")
    quota, period = fields
    if quota == "max":
        return None
    try:
        quota_value = int(quota)
        period_value = int(period)
    except ValueError as error:
        raise RuntimeError(f"malformed cgroup cpu.max: {cpu_max_path}") from error
    if quota_value <= 0 or period_value <= 0:
        raise RuntimeError(f"invalid cgroup cpu.max values: {cpu_max_path}")
    return quota_value / period_value


def _partition_worker_counts(
    *,
    worker_counts: tuple[int, ...],
    gpu_ids: tuple[int, ...],
    cpu_quota_cores: float | None,
    estimated_cpu_cores_per_worker: float | None,
) -> tuple[tuple[int, ...], list[dict[str, object]]]:
    """Partition counts without turning quota-skipped candidates into failed measurements."""
    if estimated_cpu_cores_per_worker is not None and estimated_cpu_cores_per_worker <= 0:
        raise ValueError("estimated CPU cores per worker must be positive")
    eligible: list[int] = []
    skipped: list[dict[str, object]] = []
    for worker_count in worker_counts:
        estimated_total = (
            len(gpu_ids) * worker_count * estimated_cpu_cores_per_worker
            if estimated_cpu_cores_per_worker is not None
            else None
        )
        if (
            cpu_quota_cores is not None
            and estimated_total is not None
            and estimated_total > cpu_quota_cores
        ):
            skipped.append(
                {
                    "worker_count_per_gpu": worker_count,
                    "measured": False,
                    "reason_code": "cpu_cgroup_quota_exceeded",
                    "cpu_quota_cores": cpu_quota_cores,
                    "estimated_cpu_cores_per_worker": estimated_cpu_cores_per_worker,
                    "estimated_total_cpu_cores": estimated_total,
                }
            )
        else:
            eligible.append(worker_count)
    if not eligible:
        raise RuntimeError("cgroup CPU quota excludes every requested worker count")
    return tuple(eligible), skipped


def _checkpoint_file_map(root: Path) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _clone_source_equivalence(
    *,
    source_run: Path,
    checkpoint_relative_path: str,
    worker_root: Path,
    allow_historical_hardlinks: bool,
) -> dict[str, object]:
    """Verify that one minimal worker clone still references the immutable source bytes."""
    clone_run = worker_root / "run"
    if (clone_run / "checkpoint_selection.json").exists() or (clone_run / "complete.json").exists():
        raise RuntimeError("worker clone contains forbidden selection or completion evidence")
    copied_inputs: dict[str, str] = {}
    for name in ("run_manifest.json", "metrics.jsonl"):
        source = source_run / name
        clone = clone_run / name
        if not source.is_file() or not clone.is_file():
            raise RuntimeError(f"worker clone input is missing: {name}")
        source_digest = _sha256_file(source)
        if _sha256_file(clone) != source_digest:
            raise RuntimeError(f"worker clone input differs from source: {name}")
        copied_inputs[name] = source_digest

    source_checkpoint = source_run / checkpoint_relative_path
    clone_checkpoint = clone_run / checkpoint_relative_path
    source_files = _checkpoint_file_map(source_checkpoint)
    clone_files = _checkpoint_file_map(clone_checkpoint)
    if not source_files or source_files.keys() != clone_files.keys():
        raise RuntimeError("worker clone checkpoint file set differs from source")
    checkpoint_hashes: dict[str, str] = {}
    hardlinked_files: list[str] = []
    for relative_path, source in source_files.items():
        clone = clone_files[relative_path]
        digest = _sha256_file(source)
        if _sha256_file(clone) != digest:
            raise RuntimeError(f"worker clone checkpoint differs from source: {relative_path}")
        try:
            same_file = os.path.samefile(source, clone)
        except OSError:
            same_file = False
        if same_file:
            hardlinked_files.append(relative_path)
        checkpoint_hashes[relative_path] = digest
    source_mutation_isolation_validated = not hardlinked_files
    if hardlinked_files and not allow_historical_hardlinks:
        raise RuntimeError(
            "new worker clone checkpoint is hard-linked to immutable source evidence"
        )
    return {
        "passed": True,
        "copied_input_sha256": copied_inputs,
        "checkpoint_sha256": checkpoint_hashes,
        "hardlinked_checkpoint_files": hardlinked_files,
        "source_mutation_isolation_validated": source_mutation_isolation_validated,
    }


def _clone_worker_run(
    *,
    source_run: Path,
    checkpoint_relative_path: str,
    worker_root: Path,
) -> Path:
    run_root = worker_root / "run"
    run_root.mkdir(parents=True)
    for name in ("run_manifest.json", "metrics.jsonl"):
        shutil.copy2(source_run / name, run_root / name)
    source_checkpoint = source_run / checkpoint_relative_path
    target_checkpoint = run_root / checkpoint_relative_path
    target_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_checkpoint, target_checkpoint, copy_function=shutil.copy2)
    audit = _clone_source_equivalence(
        source_run=source_run,
        checkpoint_relative_path=checkpoint_relative_path,
        worker_root=worker_root,
        allow_historical_hardlinks=False,
    )
    if audit.get("source_mutation_isolation_validated") is not True:
        raise RuntimeError("new worker clone does not isolate source checkpoint mutations")
    return target_checkpoint


def _semantic_digest(benchmark: dict[str, Any]) -> str:
    episodes = benchmark.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != EPISODES_PER_WORKER:
        raise RuntimeError("worker validation benchmark has the wrong episode count")
    semantic = [
        {
            "scene_seed": item.get("scene_seed"),
            "scene_id": item.get("scene_id"),
            "task_id": item.get("task_id"),
            "status": item.get("status"),
            "success": item.get("success"),
            "task_success": item.get("task_success"),
            "episode_steps": item.get("episode_steps"),
            "target_in_wrong_bin": item.get("target_in_wrong_bin"),
            "wrong_object_in_target_bin": item.get("wrong_object_in_target_bin"),
            "target_off_table": item.get("target_off_table"),
        }
        for item in episodes
    ]
    canonical = json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _worker_result(
    *,
    phase: str,
    worker_count: int,
    gpu_id: int,
    worker_index: int,
    return_code: int | None,
    duration: float | None,
    report_path: Path,
    completion_provenance: str = "observed_subprocess_exit",
) -> WorkerResult:
    report = _read(report_path) if report_path.is_file() else {}
    output_value = report.get("output")
    benchmark = (
        _read(Path(str(output_value)) / "benchmark.json")
        if isinstance(output_value, str) and (Path(output_value) / "benchmark.json").is_file()
        else {}
    )
    episodes = benchmark.get("episodes")
    summed_duration = (
        sum(float(item["total_episode_duration_s"]) for item in episodes)
        if isinstance(episodes, list)
        and all(isinstance(item, dict) and "total_episode_duration_s" in item for item in episodes)
        else None
    )
    semantic_digest = _semantic_digest(benchmark) if benchmark else None
    episode_count = int(report.get("episode_count", -1))
    return_code_observed = return_code is not None
    artifact_measurement_validated = all(
        (
            report.get("passed") is True,
            report.get("physical_execution") is True,
            report.get("infrastructure_failure_count") == 0,
            report.get("action_bound_mode") == "project",
            report.get("reused_existing_evaluation") is not True,
            episode_count == EPISODES_PER_WORKER,
            benchmark.get("passed") is True,
            benchmark.get("environment_action_applied") is True,
            benchmark.get("infrastructure_failure_count") == 0,
        )
    )
    passed = return_code_observed and return_code == 0 and artifact_measurement_validated
    return WorkerResult(
        phase=phase,
        worker_count_per_gpu=worker_count,
        gpu_id=gpu_id,
        worker_index=worker_index,
        return_code=return_code,
        return_code_observed=return_code_observed,
        completion_provenance=completion_provenance,
        wall_duration_s=duration,
        report_path=str(report_path.resolve()),
        output_path=str(output_value) if isinstance(output_value, str) else None,
        artifact_measurement_validated=artifact_measurement_validated,
        passed=passed,
        physical_execution=report.get("physical_execution") is True,
        infrastructure_failure_count=int(report.get("infrastructure_failure_count", -1)),
        episode_count=episode_count,
        success_rate=(float(report["success_rate"]) if "success_rate" in report else None),
        schedule_digest=(str(report["schedule_digest"]) if "schedule_digest" in report else None),
        semantic_digest=semantic_digest,
        summed_episode_duration_s=summed_duration,
    )


def _summarize_phase(
    *,
    phase: str,
    worker_count: int,
    gpu_ids: tuple[int, ...],
    wall_duration_s: float,
    results: list[WorkerResult],
    gpu_samples: list[dict[str, object]],
    duration_source: str,
) -> dict[str, object]:
    if wall_duration_s <= 0:
        raise RuntimeError(f"phase {phase} has a non-positive wall duration")
    expected_workers = {(gpu_id, index) for gpu_id in gpu_ids for index in range(worker_count)}
    actual_workers = {(item.gpu_id, item.worker_index) for item in results}
    if actual_workers != expected_workers:
        raise RuntimeError(f"phase {phase} worker set is incomplete or contains extras")
    card_rates = {
        str(gpu_id): worker_count * EPISODES_PER_WORKER / wall_duration_s * 60.0
        for gpu_id in gpu_ids
    }
    schedule_digests = {item.schedule_digest for item in results}
    semantic_digests = {item.semantic_digest for item in results}
    artifact_measurements_validated = (
        all(item.artifact_measurement_validated for item in results)
        and len(schedule_digests) == 1
        and None not in schedule_digests
        and len(semantic_digests) == 1
        and None not in semantic_digests
    )
    passed = all(item.passed for item in results) and artifact_measurements_validated
    return {
        "phase": phase,
        "worker_count_per_gpu": worker_count,
        "gpu_ids": list(gpu_ids),
        "wall_duration_s": wall_duration_s,
        "duration_source": duration_source,
        "total_episodes": len(gpu_ids) * worker_count * EPISODES_PER_WORKER,
        "cluster_episodes_per_minute": (
            len(gpu_ids) * worker_count * EPISODES_PER_WORKER / wall_duration_s * 60.0
        ),
        "per_gpu_episodes_per_minute": card_rates,
        "worst_gpu_episodes_per_minute": min(card_rates.values()),
        "artifact_measurements_validated": artifact_measurements_validated,
        "measurement_validated": passed,
        "passed": passed,
        "schedule_digest": next(iter(schedule_digests)) if len(schedule_digests) == 1 else None,
        "semantic_digest": next(iter(semantic_digests)) if len(semantic_digests) == 1 else None,
        "workers": [asdict(item) for item in results],
        "gpu_samples": gpu_samples,
    }


def _path_creation_time(path: Path) -> tuple[float, str]:
    stat_result = path.stat()
    birth_time = getattr(stat_result, "st_birthtime", None)
    if isinstance(birth_time, (int, float)) and birth_time > 0:
        return float(birth_time), "filesystem_birth_time"
    if sys.platform.startswith("linux"):
        completed = subprocess.run(
            ["stat", "--format=%W", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            try:
                linux_birth_time = float(completed.stdout.strip())
            except ValueError:
                linux_birth_time = 0.0
            if linux_birth_time > 0:
                return linux_birth_time, "filesystem_birth_time"
    raise RuntimeError(f"reliable filesystem birth time is unavailable: {path}")


def _recover_completed_phase(
    *,
    phase: str,
    worker_count: int,
    gpu_ids: tuple[int, ...],
    source_run: Path,
    checkpoint_relative_path: str,
    output_root: Path,
    expected_run_fingerprint: str,
    expected_checkpoint_fingerprint: str,
) -> dict[str, object]:
    """Recover one already completed physical phase without inventing process exit codes."""
    phase_root = output_root / phase
    if not phase_root.is_dir():
        raise RuntimeError(f"completed phase directory is missing: {phase_root}")
    expected_roots = {
        (phase_root / f"gpu{gpu_id}" / f"worker{worker_index:02d}").resolve()
        for gpu_id in gpu_ids
        for worker_index in range(worker_count)
    }
    actual_roots = {
        path.resolve()
        for gpu_root in phase_root.glob("gpu*")
        if gpu_root.is_dir()
        for path in gpu_root.glob("worker*")
        if path.is_dir()
    }
    if actual_roots != expected_roots:
        raise RuntimeError(f"phase {phase} worker directories do not match the expected set")

    phase_started_at, start_source = _path_creation_time(phase_root)
    report_mtimes: list[float] = []
    results: list[WorkerResult] = []
    clone_audits: list[dict[str, object]] = []
    for gpu_id in gpu_ids:
        for worker_index in range(worker_count):
            worker_root = phase_root / f"gpu{gpu_id}" / f"worker{worker_index:02d}"
            report_path = worker_root / "report.json"
            if not report_path.is_file():
                raise RuntimeError(f"completed worker report is missing: {report_path}")
            report = _read(report_path)
            if (
                report.get("run_fingerprint") != expected_run_fingerprint
                or report.get("checkpoint_fingerprint") != expected_checkpoint_fingerprint
                or report.get("split") != "validation"
            ):
                raise RuntimeError(f"completed worker identity is incompatible: {report_path}")
            output_value = report.get("output")
            if not isinstance(output_value, str):
                raise RuntimeError(f"completed worker output path is missing: {report_path}")
            evaluation_output = Path(output_value).resolve()
            if not evaluation_output.is_relative_to(worker_root.resolve()):
                raise RuntimeError(f"completed worker output escaped its clone: {report_path}")
            if not (evaluation_output / "benchmark.json").is_file():
                raise RuntimeError(f"completed worker benchmark is missing: {evaluation_output}")
            clone_audit = _clone_source_equivalence(
                source_run=source_run,
                checkpoint_relative_path=checkpoint_relative_path,
                worker_root=worker_root,
                allow_historical_hardlinks=True,
            )
            clone_audits.append(
                {
                    "gpu_id": gpu_id,
                    "worker_index": worker_index,
                    **clone_audit,
                }
            )
            report_mtimes.append(report_path.stat().st_mtime)
            results.append(
                _worker_result(
                    phase=phase,
                    worker_count=worker_count,
                    gpu_id=gpu_id,
                    worker_index=worker_index,
                    return_code=None,
                    duration=None,
                    report_path=report_path,
                    completion_provenance="passed_report_recovery",
                )
            )
    phase_ended_at = max(report_mtimes)
    wall_duration_s = phase_ended_at - phase_started_at
    if wall_duration_s <= 0:
        raise RuntimeError(f"phase {phase} filesystem timestamps cannot recover a duration")
    summary = _summarize_phase(
        phase=phase,
        worker_count=worker_count,
        gpu_ids=gpu_ids,
        wall_duration_s=wall_duration_s,
        results=results,
        gpu_samples=[],
        duration_source=f"{start_source}_to_latest_report_mtime",
    )
    summary["recovered_from_existing_artifacts"] = True
    summary["subprocess_return_codes_observed"] = False
    summary["clone_source_equivalence_validated"] = all(
        item.get("passed") is True for item in clone_audits
    )
    summary["source_mutation_isolation_validated"] = all(
        item.get("source_mutation_isolation_validated") is True for item in clone_audits
    )
    summary["clone_source_audits"] = clone_audits
    summary["artifact_measurements_validated"] = (
        summary.get("artifact_measurements_validated") is True
        and summary["clone_source_equivalence_validated"]
    )
    summary["measurement_validated"] = False
    summary["passed"] = False
    return summary


def _gpu_sample(gpu_ids: tuple[int, ...]) -> list[dict[str, float | int]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return []
    samples: list[dict[str, float | int]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5 or int(fields[0]) not in gpu_ids:
            continue
        samples.append(
            {
                "gpu_id": int(fields[0]),
                "utilization_percent": float(fields[1]),
                "memory_used_mib": float(fields[2]),
                "temperature_c": float(fields[3]),
                "power_w": float(fields[4]),
            }
        )
    return samples


def _run_phase(
    *,
    phase: str,
    worker_count: int,
    gpu_ids: tuple[int, ...],
    source_run: Path,
    checkpoint_relative_path: str,
    dataset_root: Path,
    output_root: Path,
    python: Path,
) -> dict[str, object]:
    phase_root = output_root / phase
    phase_root.mkdir(parents=True)
    jobs: list[dict[str, object]] = []
    started = time.monotonic()
    for gpu_id in gpu_ids:
        for worker_index in range(worker_count):
            worker_root = phase_root / f"gpu{gpu_id}" / f"worker{worker_index:02d}"
            worker_root.mkdir(parents=True)
            checkpoint = _clone_worker_run(
                source_run=source_run,
                checkpoint_relative_path=checkpoint_relative_path,
                worker_root=worker_root,
            )
            report_path = worker_root / "report.json"
            log_path = worker_root / "worker.log"
            log_stream = log_path.open("w", encoding="utf-8")
            command = [
                str(python),
                "scripts/evaluate_act.py",
                "--checkpoint",
                str(checkpoint),
                "--dataset-root",
                str(dataset_root),
                "--split",
                "validation",
                "--sim-backend",
                "physx_cpu",
                "--action-bound-mode",
                "project",
                "--report",
                str(report_path),
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            environment["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
            )
            jobs.append(
                {
                    "gpu_id": gpu_id,
                    "worker_index": worker_index,
                    "process": process,
                    "started": time.monotonic(),
                    "report_path": report_path,
                    "log_stream": log_stream,
                }
            )
    samples: list[dict[str, object]] = []
    while any(job["process"].poll() is None for job in jobs):
        samples.append({"elapsed_s": time.monotonic() - started, "gpus": _gpu_sample(gpu_ids)})
        time.sleep(1.0)
    ended = time.monotonic()
    results: list[WorkerResult] = []
    for job in jobs:
        process = job["process"]
        job["log_stream"].close()
        results.append(
            _worker_result(
                phase=phase,
                worker_count=worker_count,
                gpu_id=int(job["gpu_id"]),
                worker_index=int(job["worker_index"]),
                return_code=int(process.returncode),
                duration=ended - float(job["started"]),
                report_path=Path(job["report_path"]),
            )
        )
    return _summarize_phase(
        phase=phase,
        worker_count=worker_count,
        gpu_ids=gpu_ids,
        wall_duration_s=ended - started,
        results=results,
        gpu_samples=samples,
        duration_source="observed_monotonic_clock",
    )


def select_worker_count(
    phases: list[dict[str, object]],
    *,
    eligibility_field: str = "passed",
) -> tuple[int, int | None]:
    eligible = [phase for phase in phases if phase.get(eligibility_field) is True]
    if not eligible:
        raise RuntimeError(f"no worker-count phase passed the {eligibility_field} gate")
    maximum = max(float(phase["worst_gpu_episodes_per_minute"]) for phase in eligible)
    within_five_percent = [
        phase
        for phase in eligible
        if float(phase["worst_gpu_episodes_per_minute"]) >= maximum * 0.95
    ]
    winner = min(int(phase["worker_count_per_gpu"]) for phase in within_five_percent)
    ranked = sorted(
        eligible,
        key=lambda phase: (
            -float(phase["worst_gpu_episodes_per_minute"]),
            int(phase["worker_count_per_gpu"]),
        ),
    )
    runner = next(
        (
            int(phase["worker_count_per_gpu"])
            for phase in ranked
            if int(phase["worker_count_per_gpu"]) != winner
        ),
        None,
    )
    return winner, runner


def _stability_record(
    *,
    worker_count: int,
    original: dict[str, object],
    repeat: dict[str, object] | None,
    repeat_source: str | None,
) -> dict[str, object]:
    if repeat is None:
        return {
            "worker_count_per_gpu": worker_count,
            "completed": False,
            "stable": False,
            "repeat_source": repeat_source,
            "reason": "no completed temporal repeat was available",
        }
    rate = float(original["worst_gpu_episodes_per_minute"])
    repeat_rate = float(repeat["worst_gpu_episodes_per_minute"])
    relative_change = abs(repeat_rate - rate) / rate
    semantics_match = repeat.get("semantic_digest") == original.get("semantic_digest")
    stable = all(
        (
            repeat.get("passed") is True,
            semantics_match,
            relative_change <= 0.10,
        )
    )
    return {
        "worker_count_per_gpu": worker_count,
        "completed": True,
        "stable": stable,
        "repeat_source": repeat_source,
        "relative_throughput_change": relative_change,
        "matches_reference_semantics": semantics_match,
        "repeat": repeat,
    }


def _selection_stability_validated(
    *,
    retests: list[dict[str, object]],
    winner: int,
    runner: int | None,
    skip_retest: bool,
) -> bool:
    expected = {winner, runner} - {None}
    return (
        not skip_retest
        and {item.get("worker_count_per_gpu") for item in retests} == expected
        and all(item.get("stable") is True for item in retests)
    )


def _validate_common_args(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, tuple[int, ...], tuple[int, ...]]:
    source_run = args.source_run.resolve()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    gpu_ids = tuple(args.gpu_ids)
    worker_counts = tuple(args.worker_counts)
    if (
        not source_run.is_dir()
        or not dataset_root.is_dir()
        or len(gpu_ids) != len(set(gpu_ids))
        or not gpu_ids
        or len(worker_counts) != len(set(worker_counts))
        or any(value < 1 for value in worker_counts)
    ):
        raise ValueError("invalid source, dataset, GPU, or worker-count configuration")
    return source_run, dataset_root, output_root, gpu_ids, worker_counts


def summarize_existing(args: argparse.Namespace) -> dict[str, object]:
    """Build a fail-closed report from completed phases after an interrupted parent process."""
    source_run, dataset_root, output_root, gpu_ids, worker_counts = _validate_common_args(args)
    if not output_root.is_dir():
        raise FileNotFoundError("existing benchmark output root is missing")
    if args.estimated_cpu_cores_per_worker is None:
        raise ValueError("recovered summarization requires --estimated-cpu-cores-per-worker")
    manifest, checkpoint_relative_path = _selected_checkpoint(source_run)
    selected_checkpoint = manifest.selected_checkpoint_fingerprint
    if selected_checkpoint is None:
        raise RuntimeError("completed source manifest does not name a selected checkpoint")
    cpu_quota = _cpu_quota_cores()
    eligible_counts, skipped_counts = _partition_worker_counts(
        worker_counts=worker_counts,
        gpu_ids=gpu_ids,
        cpu_quota_cores=cpu_quota,
        estimated_cpu_cores_per_worker=args.estimated_cpu_cores_per_worker,
    )
    before = _source_snapshot(source_run, checkpoint_relative_path)
    warmup = None
    if not args.skip_warmup:
        warmup = _recover_completed_phase(
            phase="warmup-n1",
            worker_count=1,
            gpu_ids=gpu_ids,
            source_run=source_run,
            checkpoint_relative_path=checkpoint_relative_path,
            output_root=output_root,
            expected_run_fingerprint=manifest.identity.run_fingerprint,
            expected_checkpoint_fingerprint=selected_checkpoint,
        )
    phases: list[dict[str, object]] = []
    reference_digest = warmup.get("semantic_digest") if isinstance(warmup, dict) else None
    for worker_count in eligible_counts:
        phase = _recover_completed_phase(
            phase=f"timed-n{worker_count}",
            worker_count=worker_count,
            gpu_ids=gpu_ids,
            source_run=source_run,
            checkpoint_relative_path=checkpoint_relative_path,
            output_root=output_root,
            expected_run_fingerprint=manifest.identity.run_fingerprint,
            expected_checkpoint_fingerprint=selected_checkpoint,
        )
        if reference_digest is None:
            reference_digest = phase.get("semantic_digest")
        phase["matches_reference_semantics"] = phase.get("semantic_digest") == reference_digest
        phase["artifact_measurements_validated"] = (
            phase.get("artifact_measurements_validated") is True
            and phase["matches_reference_semantics"]
        )
        phase["measurement_validated"] = False
        phase["passed"] = False
        phases.append(phase)
    winner, runner = select_worker_count(
        phases,
        eligibility_field="artifact_measurements_validated",
    )
    stability_records: list[dict[str, object]] = []
    for worker_count in (winner, runner):
        if worker_count is None:
            continue
        original = next(phase for phase in phases if phase["worker_count_per_gpu"] == worker_count)
        repeat: dict[str, object] | None = None
        repeat_source: str | None = None
        retest_root = output_root / f"retest-n{worker_count}"
        if retest_root.is_dir() and not args.skip_retest:
            repeat_source = f"retest-n{worker_count}"
            repeat = _recover_completed_phase(
                phase=repeat_source,
                worker_count=worker_count,
                gpu_ids=gpu_ids,
                source_run=source_run,
                checkpoint_relative_path=checkpoint_relative_path,
                output_root=output_root,
                expected_run_fingerprint=manifest.identity.run_fingerprint,
                expected_checkpoint_fingerprint=selected_checkpoint,
            )
        elif worker_count == 1 and warmup is not None and not args.skip_retest:
            repeat_source = "warmup-n1"
            repeat = warmup
        stability_records.append(
            _stability_record(
                worker_count=worker_count,
                original=original,
                repeat=repeat,
                repeat_source=repeat_source,
            )
        )
    after = _source_snapshot(source_run, checkpoint_relative_path)
    source_unchanged_during_summary = after == before
    artifact_measurements_validated = all(
        (
            source_unchanged_during_summary,
            warmup is None or warmup.get("artifact_measurements_validated") is True,
            all(phase.get("artifact_measurements_validated") is True for phase in phases),
        )
    )
    source_mutation_isolation_validated = all(
        phase.get("source_mutation_isolation_validated") is True
        for phase in ([warmup] if warmup is not None else []) + phases
    )
    measurement_validated = False
    selection_stability_validated = False
    recommendation_supported = artifact_measurements_validated
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "passed": False,
        "artifact_measurements_validated": artifact_measurements_validated,
        "measurement_validated": measurement_validated,
        "selection_stability_validated": selection_stability_validated,
        "recommendation_supported": recommendation_supported,
        "measurement_only": True,
        "skip_retest_requested": args.skip_retest,
        "source_mutation_isolation_validated": source_mutation_isolation_validated,
        "source_run": str(source_run),
        "source_run_fingerprint": manifest.identity.run_fingerprint,
        "selected_checkpoint_fingerprint": selected_checkpoint,
        "dataset_root": str(dataset_root),
        "gpu_ids": list(gpu_ids),
        "worker_counts": list(worker_counts),
        "eligible_measured_worker_counts": list(eligible_counts),
        "skipped_worker_counts": skipped_counts,
        "cpu_cgroup_quota_cores": cpu_quota,
        "estimated_cpu_cores_per_worker": args.estimated_cpu_cores_per_worker,
        "cpu_constraint_scope": (
            "current cgroup quota plus an operator-supplied per-worker CPU estimate; "
            "not a global worker-count proof"
        ),
        "episodes_per_worker": EPISODES_PER_WORKER,
        "selection_scope": "eligible physically measured worker counts within the CPU quota",
        "selection_rule": (
            "maximize worst-GPU episodes/min; within 5% choose fewer workers; "
            "winner/runner-up temporal repeat must stay within 10% and preserve semantics"
        ),
        "warmup": warmup,
        "phases": phases,
        "stability_records": stability_records,
        "recommended_worker_count_per_gpu": winner,
        "selected_worker_count_per_gpu": None,
        "source_snapshot_sha256": before,
        "source_unchanged_during_summary": source_unchanged_during_summary,
        "benchmark_window_source_unchanged": None,
        "benchmark_window_source_unchanged_reason": (
            "the interrupted parent did not persist its in-memory pre-benchmark snapshot"
        ),
    }


def execute(args: argparse.Namespace) -> dict[str, object]:
    source_run, dataset_root, output_root, gpu_ids, requested_worker_counts = _validate_common_args(
        args
    )
    if output_root.exists():
        raise FileExistsError("worker benchmark output root is immutable and must not exist")
    cpu_quota = _cpu_quota_cores()
    worker_counts, skipped_counts = _partition_worker_counts(
        worker_counts=requested_worker_counts,
        gpu_ids=gpu_ids,
        cpu_quota_cores=cpu_quota,
        estimated_cpu_cores_per_worker=args.estimated_cpu_cores_per_worker,
    )
    output_root.mkdir(parents=True)
    manifest, checkpoint_relative_path = _selected_checkpoint(source_run)
    before = _source_snapshot(source_run, checkpoint_relative_path)
    warmup = None
    if not args.skip_warmup:
        warmup = _run_phase(
            phase="warmup-n1",
            worker_count=1,
            gpu_ids=gpu_ids,
            source_run=source_run,
            checkpoint_relative_path=checkpoint_relative_path,
            dataset_root=dataset_root,
            output_root=output_root,
            python=args.python.resolve(),
        )
        if warmup.get("passed") is not True:
            raise RuntimeError("worker benchmark warmup failed")
    phases: list[dict[str, object]] = []
    reference_digest = warmup.get("semantic_digest") if isinstance(warmup, dict) else None
    for worker_count in worker_counts:
        phase = _run_phase(
            phase=f"timed-n{worker_count}",
            worker_count=worker_count,
            gpu_ids=gpu_ids,
            source_run=source_run,
            checkpoint_relative_path=checkpoint_relative_path,
            dataset_root=dataset_root,
            output_root=output_root,
            python=args.python.resolve(),
        )
        if reference_digest is None:
            reference_digest = phase.get("semantic_digest")
        phase["matches_reference_semantics"] = phase.get("semantic_digest") == reference_digest
        phase["passed"] = phase.get("passed") is True and phase["matches_reference_semantics"]
        phases.append(phase)
        if _source_snapshot(source_run, checkpoint_relative_path) != before:
            raise RuntimeError("worker benchmark modified immutable source evidence")
    winner, runner = select_worker_count(phases)
    retests: list[dict[str, object]] = []
    if not args.skip_retest:
        for worker_count in (winner, runner):
            if worker_count is None:
                continue
            retest = _run_phase(
                phase=f"retest-n{worker_count}",
                worker_count=worker_count,
                gpu_ids=gpu_ids,
                source_run=source_run,
                checkpoint_relative_path=checkpoint_relative_path,
                dataset_root=dataset_root,
                output_root=output_root,
                python=args.python.resolve(),
            )
            original = next(item for item in phases if item["worker_count_per_gpu"] == worker_count)
            rate = float(original["worst_gpu_episodes_per_minute"])
            retest_rate = float(retest["worst_gpu_episodes_per_minute"])
            retest["relative_throughput_change"] = abs(retest_rate - rate) / rate
            retest["matches_reference_semantics"] = (
                retest.get("semantic_digest") == reference_digest
            )
            retest["stable"] = all(
                (
                    retest.get("passed") is True,
                    retest["matches_reference_semantics"],
                    float(retest["relative_throughput_change"]) <= 0.10,
                )
            )
            retests.append(retest)
    after = _source_snapshot(source_run, checkpoint_relative_path)
    source_unchanged = after == before
    if not source_unchanged:
        raise RuntimeError("worker benchmark changed immutable source run files")
    selection_stability_validated = _selection_stability_validated(
        retests=retests,
        winner=winner,
        runner=runner,
        skip_retest=args.skip_retest,
    )
    measurement_validated = source_unchanged and all(
        phase.get("passed") is True for phase in phases
    )
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "passed": measurement_validated and selection_stability_validated,
        "artifact_measurements_validated": measurement_validated,
        "measurement_validated": measurement_validated,
        "selection_stability_validated": selection_stability_validated,
        "recommendation_supported": measurement_validated,
        "measurement_only": args.skip_retest,
        "source_mutation_isolation_validated": True,
        "source_run": str(source_run),
        "source_run_fingerprint": manifest.identity.run_fingerprint,
        "selected_checkpoint_fingerprint": manifest.selected_checkpoint_fingerprint,
        "dataset_root": str(dataset_root),
        "gpu_ids": list(gpu_ids),
        "worker_counts": list(requested_worker_counts),
        "eligible_measured_worker_counts": list(worker_counts),
        "skipped_worker_counts": skipped_counts,
        "cpu_cgroup_quota_cores": cpu_quota,
        "estimated_cpu_cores_per_worker": args.estimated_cpu_cores_per_worker,
        "cpu_constraint_scope": (
            "current cgroup quota plus an optional operator-supplied per-worker CPU estimate; "
            "not a global worker-count proof"
        ),
        "episodes_per_worker": EPISODES_PER_WORKER,
        "selection_rule": (
            "maximize worst-GPU episodes/min; within 5% choose fewer workers; "
            "winner/runner-up retest must stay within 10% and preserve semantics"
        ),
        "warmup": warmup,
        "phases": phases,
        "retests": retests,
        "recommended_worker_count_per_gpu": winner,
        "selected_worker_count_per_gpu": winner if selection_stability_validated else None,
        "source_snapshot_sha256": before,
        "source_unchanged": source_unchanged,
    }


def main() -> int:
    args = parse_args()
    output = args.output_root.resolve() / "benchmark.json"
    if output.exists():
        report = {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "passed": False,
            "error_type": "FileExistsError",
            "error_message": f"immutable benchmark summary already exists: {output}",
        }
        print(json.dumps({**report, "output": str(output)}, sort_keys=True))
        return 1
    try:
        report = summarize_existing(args) if args.summarize_existing else execute(args)
    except Exception as error:  # noqa: BLE001 - benchmark boundary preserves diagnostics
        import traceback

        traceback.print_exc()
        report = {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "passed": False,
            "error_type": type(error).__name__,
            "error_message": str(error) or repr(error),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, report)
    print(json.dumps({**report, "output": str(output)}, sort_keys=True))
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
