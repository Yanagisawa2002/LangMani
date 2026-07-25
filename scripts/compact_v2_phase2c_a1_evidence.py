"""Create compact source artifacts from external Phase 2C-A.1 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

from langmani.v2.phase2c_a import PACKAGE_FINGERPRINT, TASK_IDS, ModelKind

MODEL_REPORT_NAMES = {
    ModelKind.PICK.value: "train-pick.json",
    ModelKind.STACK.value: "train-stack.json",
    ModelKind.PUSH.value: "train-push.json",
    ModelKind.SHARED.value: "train-shared.json",
}
FINAL_SPLITS = ("test_unseen_reset", "test_visual_shift")


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain one JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _external_file(path: Path) -> dict[str, object]:
    return {
        "external_path": path.resolve().as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _write_new(root: Path, name: str, value: dict[str, object]) -> None:
    path = root / name
    if path.exists():
        if _read_object(path) != value:
            raise RuntimeError(f"existing immutable artifact differs: {path}")
        return
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _prohibited_processes() -> list[dict[str, object]]:
    matches = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            command = (
                (process / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        lowered = command.lower()
        if any(token in lowered for token in ("smolvla", "vla-jepa", "vla_jepa")):
            matches.append({"pid": int(process.name), "command": command})
    return matches


def _evaluation_registry(root: Path, *, final: bool) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    splits = FINAL_SPLITS if final else ("validation",)
    stage = "final" if final else "development"
    for scope in ("per_task", "shared"):
        for split in splits:
            for task_id in TASK_IDS:
                path = (
                    root / stage / scope / split / task_id
                    if final
                    else root / stage / scope / task_id
                )
                summary_path = path / "summary.json"
                episodes_path = path / "episodes.jsonl"
                records.append(
                    {
                        "scope": scope,
                        "split": split,
                        "task_id": task_id,
                        "summary": _read_object(summary_path),
                        "summary_file": _external_file(summary_path),
                        "episodes_file": _external_file(episodes_path),
                    }
                )
    return records


def _summary_episode_count(record: dict[str, object]) -> int:
    summary = record.get("summary")
    count = summary.get("episode_count") if isinstance(summary, dict) else None
    if not isinstance(count, int):
        raise RuntimeError("evaluation summary lacks an integer episode count")
    return count


def _screen_component_fingerprints(
    screens: list[dict[str, object]],
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for screen in screens:
        checkpoint = screen.get("checkpoint_fingerprint")
        components = screen.get("checkpoint_components")
        if (
            not isinstance(checkpoint, str)
            or checkpoint in result
            or not isinstance(components, dict)
            or set(components) != {"model", "preprocessor", "postprocessor"}
            or not all(isinstance(value, str) for value in components.values())
        ):
            raise RuntimeError("checkpoint screen component registry is invalid")
        result[checkpoint] = {name: str(value) for name, value in components.items()}
    return result


def _integer_field(record: dict[str, object], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int):
        raise RuntimeError(f"evaluation record lacks an integer {key}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--training-git-commit", required=True)
    parser.add_argument("--pre-final-evaluation-git-commit", required=True)
    parser.add_argument("--final-evaluation-git-commit", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    artifact_root = args.artifact_root.resolve()
    evidence_root = args.evidence_root.resolve()
    evaluation_root = args.evaluation_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    repository_status = _git(repository_root, "status", "--porcelain")
    repository_commit = _git(repository_root, "rev-parse", "HEAD")

    training_reports = {
        model_kind: _read_object(evidence_root / "reports" / filename)
        for model_kind, filename in MODEL_REPORT_NAMES.items()
    }
    screen_paths = sorted((evidence_root / "reports/screens").glob("*.json"))
    screens = [_read_object(path) for path in screen_paths]
    screen_components = _screen_component_fingerprints(screens)
    training_configs = {
        model_kind: {
            "identity": report["identity"],
            "report_file": _external_file(
                evidence_root / "reports" / MODEL_REPORT_NAMES[model_kind]
            ),
        }
        for model_kind, report in training_reports.items()
    }
    _write_new(
        artifact_root,
        "training_configs.json",
        {
            "schema_version": "langmani-v2-phase2c-a1-training-config-registry-v0",
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "training_git_commit": args.training_git_commit,
            "models": training_configs,
        },
    )

    metric_records = {}
    checkpoint_records = []
    processor_records = []
    for model_kind, report in training_reports.items():
        result = report["result"]
        assert isinstance(result, dict)
        run_root = Path(str(result["run_root"]))
        metrics_path = run_root / "metrics.jsonl"
        rows = _read_jsonl(metrics_path)
        metric_records[model_kind] = {
            "run_fingerprint": result["run_fingerprint"],
            "final_step": result["final_step"],
            "examples_processed": result["examples_processed"],
            "effective_samples_by_task": result["effective_samples_by_task"],
            "training_duration_s": result["training_duration_s"],
            "mean_examples_per_second": result["mean_examples_per_second"],
            "peak_gpu_allocated_bytes": result["peak_gpu_allocated_bytes"],
            "peak_gpu_reserved_bytes": result["peak_gpu_reserved_bytes"],
            "metric_row_count": len(rows),
            "first_metric": rows[0],
            "last_metric": rows[-1],
            "metrics_file": _external_file(metrics_path),
        }
        checkpoints = result["checkpoint_records"]
        assert isinstance(checkpoints, list)
        for checkpoint in checkpoints:
            assert isinstance(checkpoint, dict)
            checkpoint_records.append(
                {
                    "model_kind": model_kind,
                    "run_fingerprint": result["run_fingerprint"],
                    **checkpoint,
                }
            )
            components = screen_components.get(str(checkpoint["checkpoint_fingerprint"]))
            if components is None:
                raise RuntimeError("training checkpoint lacks a matching screen")
            processor_records.append(
                {
                    "model_kind": model_kind,
                    "checkpoint_fingerprint": checkpoint["checkpoint_fingerprint"],
                    "checkpoint_step": checkpoint["global_step"],
                    "preprocessor": components["preprocessor"],
                    "postprocessor": components["postprocessor"],
                    "model": components["model"],
                }
            )
    _write_new(
        artifact_root,
        "training_metrics.json",
        {
            "schema_version": "langmani-v2-phase2c-a1-training-metrics-registry-v0",
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "models": metric_records,
            "raw_metrics_committed": False,
        },
    )
    _write_new(
        artifact_root,
        "checkpoint_registry.json",
        {
            "schema_version": "langmani-v2-phase2c-a1-checkpoint-registry-v0",
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "checkpoint_count": len(checkpoint_records),
            "checkpoints": checkpoint_records,
            "checkpoint_bytes_committed": False,
        },
    )
    _write_new(
        artifact_root,
        "processor_registry.json",
        {
            "schema_version": "langmani-v2-phase2c-a1-processor-registry-v0",
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "records": processor_records,
        },
    )

    selection_paths = sorted((evidence_root / "reports/selections").glob("*.json"))
    _write_new(
        artifact_root,
        "checkpoint_screen.json",
        {
            "schema_version": "langmani-v2-phase2c-a1-checkpoint-screen-registry-v0",
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "screens": [
                {"result": screen, "file": _external_file(path)}
                for path, screen in zip(screen_paths, screens, strict=True)
            ],
            "selections": [
                {"result": _read_object(path), "file": _external_file(path)}
                for path in selection_paths
            ],
        },
    )

    closed_loop_smoke_root = evaluation_root / "closed_loop_smoke"
    closed_loop_smoke_summary = _read_object(closed_loop_smoke_root / "summary.json")
    closed_loop_smoke_rows = _read_jsonl(closed_loop_smoke_root / "episodes.jsonl")
    closed_loop_smoke_actions = sum(
        _integer_field(row, "actions_executed") for row in closed_loop_smoke_rows
    )
    _write_new(
        artifact_root,
        "closed_loop_smoke.json",
        {
            "schema_version": "langmani-v2-phase2c-a1-closed-loop-smoke-registry-v0",
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "summary": closed_loop_smoke_summary,
            "runtime_manifest": _read_object(closed_loop_smoke_root / "runtime_manifest.json"),
            "episodes_file": _external_file(closed_loop_smoke_root / "episodes.jsonl"),
            "actions_executed": closed_loop_smoke_actions,
            "passed": (
                len(closed_loop_smoke_rows) == 1
                and closed_loop_smoke_summary.get("episode_count") == 1
                and closed_loop_smoke_actions >= 1
                and closed_loop_smoke_summary.get("invalid_action_count") == 0
                and closed_loop_smoke_summary.get("simulator_error_count") == 0
            ),
        },
    )

    horizon = _read_object(evidence_root / "reports/horizon-selection.json")
    _write_new(artifact_root, "execution_horizon_comparison.json", horizon)
    development = _evaluation_registry(evaluation_root, final=False)
    final = _evaluation_registry(evaluation_root, final=True)
    _write_new(
        artifact_root,
        "development_evaluation.json",
        {
            "schema_version": "langmani-v2-phase2c-a1-development-registry-v0",
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "episode_count": sum(_summary_episode_count(record) for record in development),
            "groups": development,
        },
    )
    final_lock = _read_object(evidence_root / "reports/final-policy-lock.json")
    _write_new(artifact_root, "final_policy_manifests.json", final_lock)
    _write_new(
        artifact_root,
        "final_evaluation.json",
        {
            "schema_version": "langmani-v2-phase2c-a1-final-evaluation-registry-v0",
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "episode_count": sum(_summary_episode_count(record) for record in final),
            "groups": final,
        },
    )

    intervention_lock_path = evidence_root / "reports/task-intervention-lock.json"
    intervention_analysis_path = evidence_root / "reports/task-intervention-analysis.json"
    _write_new(
        artifact_root,
        "task_id_intervention.json",
        {
            "schema_version": "langmani-v2-phase2c-a1-task-intervention-registry-v0",
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "lock": _read_object(intervention_lock_path),
            "analysis": _read_object(intervention_analysis_path),
            "lock_file": _external_file(intervention_lock_path),
            "analysis_file": _external_file(intervention_analysis_path),
        },
    )
    analysis = _read_object(evidence_root / "reports/result-analysis.json")
    _write_new(
        artifact_root,
        "multiskill_interference.json",
        {
            "schema_version": "langmani-v2-phase2c-a1-multiskill-interference-v0",
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "success_rate_difference": analysis["multi_skill_interference_success_rate_difference"],
            "average": analysis["multi_skill_interference_average"],
            "timeout_rate_difference": analysis["multi_skill_timeout_rate_difference"],
            "action_variance_difference_by_dimension": analysis[
                "multi_skill_action_variance_difference_by_dimension"
            ],
            "failure_category_comparison": analysis["multi_skill_failure_category_comparison"],
            "visual_shift_success_rate_difference": analysis[
                "visual_shift_success_rate_difference"
            ],
        },
    )
    summaries = analysis["summaries"]
    assert isinstance(summaries, dict)
    _write_new(
        artifact_root,
        "failure_taxonomy.json",
        {
            "schema_version": "langmani-v2-phase2c-a1-failure-taxonomy-v0",
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "groups": {
                scope: {
                    split: {task: value["failure_categories"] for task, value in by_task.items()}
                    for split, by_task in by_split.items()
                }
                for scope, by_split in summaries.items()
            },
        },
    )
    _write_new(
        artifact_root,
        "latency.json",
        {
            "schema_version": "langmani-v2-phase2c-a1-latency-v0",
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "groups": {
                scope: {
                    split: {
                        task: {
                            "p50_ms": value["inference_latency_p50_ms"],
                            "p95_ms": value["inference_latency_p95_ms"],
                        }
                        for task, value in by_task.items()
                    }
                    for split, by_task in by_split.items()
                }
                for scope, by_split in summaries.items()
            },
        },
    )
    _write_new(artifact_root, "result_classification.json", analysis)
    closure = {
        key: analysis[key]
        for key in (
            "result",
            "act_baselines_validated",
            "act_policy_quality_weak",
            "shared_act_failed",
            "act_phase_closed",
            "further_act_architecture_authorized",
        )
    }
    _write_new(
        artifact_root,
        "act_closure_state.json",
        {
            "schema_version": "langmani-v2-phase2c-a1-act-closure-v0",
            **closure,
        },
    )
    authorization = {
        "schema_version": "langmani-v2-phase2c-a1-authorization-state-v0",
        "smolvla_phase_eligible": analysis["smolvla_phase_eligible"],
        "smolvla_training_authorized": False,
        "vla_jepa_training_authorized": False,
        "phase2c_a1_training_authorized": False,
        "additional_act_phase_authorized": False,
    }
    _write_new(artifact_root, "smolvla_eligibility.json", authorization)
    _write_new(artifact_root, "authorization_state.json", authorization)

    prohibited_processes = _prohibited_processes()
    remote_audit = {
        "schema_version": "langmani-v2-phase2c-a1-remote-execution-audit-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "hostname": platform.node(),
        "python": platform.python_version(),
        "repository_commit_before_compaction": repository_commit,
        "repository_clean_before_compaction": repository_status == "",
        "training_git_commit": args.training_git_commit,
        "pre_final_evaluation_git_commit": args.pre_final_evaluation_git_commit,
        "final_evaluation_git_commit": args.final_evaluation_git_commit,
        "evidence_root": evidence_root.as_posix(),
        "evaluation_root": evaluation_root.as_posix(),
        "prohibited_processes": prohibited_processes,
        "smolvla_or_vla_jepa_process_found": bool(prohibited_processes),
        "smolvla_or_vla_jepa_loaded": False,
        "server_shutdown_requested": False,
        "credentials_recorded": False,
        "process_id": os.getpid(),
    }
    _write_new(artifact_root, "remote_execution_audit.json", remote_audit)
    _write_new(
        artifact_root,
        "repository_environment_audit.json",
        {
            "schema_version": "langmani-v2-phase2c-a1-repository-environment-audit-v0",
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "repository_commit": repository_commit,
            "repository_clean_before_compaction": repository_status == "",
            "training_git_commit": args.training_git_commit,
            "pre_final_evaluation_git_commit": args.pre_final_evaluation_git_commit,
            "final_evaluation_git_commit": args.final_evaluation_git_commit,
            "runtime": training_reports[ModelKind.PICK.value]["runtime"],
            "remote_audit": remote_audit,
        },
    )
    if prohibited_processes:
        raise RuntimeError("SmolVLA or VLA-JEPA process found during compact audit")
    print(
        json.dumps(
            {
                "artifact_root": artifact_root.as_posix(),
                "training_model_count": len(training_reports),
                "checkpoint_count": len(checkpoint_records),
                "screen_count": len(screen_paths),
                "development_group_count": len(development),
                "final_group_count": len(final),
                "result": analysis["result"],
                "passed": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
