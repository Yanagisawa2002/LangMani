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


@dataclass(frozen=True, slots=True)
class WorkerResult:
    phase: str
    worker_count_per_gpu: int
    gpu_id: int
    worker_index: int
    return_code: int
    wall_duration_s: float
    report_path: str
    output_path: str | None
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
    shutil.copytree(source_checkpoint, target_checkpoint, copy_function=os.link)
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
    return_code: int,
    duration: float,
    report_path: Path,
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
    passed = all(
        (
            return_code == 0,
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
    return WorkerResult(
        phase=phase,
        worker_count_per_gpu=worker_count,
        gpu_id=gpu_id,
        worker_index=worker_index,
        return_code=return_code,
        wall_duration_s=duration,
        report_path=str(report_path.resolve()),
        output_path=str(output_value) if isinstance(output_value, str) else None,
        passed=passed,
        physical_execution=report.get("physical_execution") is True,
        infrastructure_failure_count=int(report.get("infrastructure_failure_count", -1)),
        episode_count=episode_count,
        success_rate=(float(report["success_rate"]) if "success_rate" in report else None),
        schedule_digest=(str(report["schedule_digest"]) if "schedule_digest" in report else None),
        semantic_digest=semantic_digest,
        summed_episode_duration_s=summed_duration,
    )


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
    wall_duration = ended - started
    card_rates = {
        str(gpu_id): worker_count * EPISODES_PER_WORKER / wall_duration * 60.0 for gpu_id in gpu_ids
    }
    schedule_digests = {item.schedule_digest for item in results}
    semantic_digests = {item.semantic_digest for item in results}
    passed = (
        all(item.passed for item in results)
        and len(schedule_digests) == 1
        and None not in schedule_digests
        and len(semantic_digests) == 1
        and None not in semantic_digests
    )
    return {
        "phase": phase,
        "worker_count_per_gpu": worker_count,
        "gpu_ids": list(gpu_ids),
        "wall_duration_s": wall_duration,
        "total_episodes": len(gpu_ids) * worker_count * EPISODES_PER_WORKER,
        "cluster_episodes_per_minute": (
            len(gpu_ids) * worker_count * EPISODES_PER_WORKER / wall_duration * 60.0
        ),
        "per_gpu_episodes_per_minute": card_rates,
        "worst_gpu_episodes_per_minute": min(card_rates.values()),
        "passed": passed,
        "schedule_digest": next(iter(schedule_digests)) if len(schedule_digests) == 1 else None,
        "semantic_digest": next(iter(semantic_digests)) if len(semantic_digests) == 1 else None,
        "workers": [asdict(item) for item in results],
        "gpu_samples": samples,
    }


def select_worker_count(phases: list[dict[str, object]]) -> tuple[int, int | None]:
    eligible = [phase for phase in phases if phase.get("passed") is True]
    if not eligible:
        raise RuntimeError("no worker-count phase passed the physical stability gate")
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


def execute(args: argparse.Namespace) -> dict[str, object]:
    source_run = args.source_run.resolve()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    gpu_ids = tuple(args.gpu_ids)
    worker_counts = tuple(args.worker_counts)
    if output_root.exists():
        raise FileExistsError("worker benchmark output root is immutable and must not exist")
    if (
        not source_run.is_dir()
        or not dataset_root.is_dir()
        or len(gpu_ids) != len(set(gpu_ids))
        or not gpu_ids
        or len(worker_counts) != len(set(worker_counts))
        or any(value < 1 for value in worker_counts)
    ):
        raise ValueError("invalid source, dataset, GPU, or worker-count configuration")
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
        winner_retest = next(item for item in retests if item["worker_count_per_gpu"] == winner)
        if winner_retest.get("stable") is not True:
            stable_lower = [
                int(item["worker_count_per_gpu"])
                for item in retests
                if item.get("stable") is True and int(item["worker_count_per_gpu"]) < winner
            ]
            if not stable_lower:
                raise RuntimeError("winning worker count was unstable and no lower retest passed")
            winner = max(stable_lower)
    after = _source_snapshot(source_run, checkpoint_relative_path)
    source_unchanged = after == before
    if not source_unchanged:
        raise RuntimeError("worker benchmark changed immutable source run files")
    return {
        "schema_version": "langmani-m4-evaluation-worker-benchmark-v1",
        "passed": source_unchanged
        and all(phase.get("passed") is True for phase in phases)
        and all(item.get("stable") is True for item in retests),
        "source_run": str(source_run),
        "source_run_fingerprint": manifest.identity.run_fingerprint,
        "selected_checkpoint_fingerprint": manifest.selected_checkpoint_fingerprint,
        "dataset_root": str(dataset_root),
        "gpu_ids": list(gpu_ids),
        "worker_counts": list(worker_counts),
        "episodes_per_worker": EPISODES_PER_WORKER,
        "selection_rule": (
            "maximize worst-GPU episodes/min; within 5% choose fewer workers; "
            "winner/runner-up retest must stay within 10% and preserve semantics"
        ),
        "warmup": warmup,
        "phases": phases,
        "retests": retests,
        "selected_worker_count_per_gpu": winner,
        "source_snapshot_sha256": before,
        "source_unchanged": source_unchanged,
    }


def main() -> int:
    args = parse_args()
    try:
        report = execute(args)
    except Exception as error:  # noqa: BLE001 - benchmark boundary preserves diagnostics
        import traceback

        traceback.print_exc()
        report = {
            "schema_version": "langmani-m4-evaluation-worker-benchmark-v1",
            "passed": False,
            "error_type": type(error).__name__,
            "error_message": str(error) or repr(error),
        }
    output = args.output_root.resolve() / "benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, report)
    print(json.dumps({**report, "output": str(output)}, sort_keys=True))
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
