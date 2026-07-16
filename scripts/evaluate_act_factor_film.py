"""Run M4.3b FactorFiLM validation, reload and development evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

from langmani.policies.act_factor_film_execution import (
    FACTOR_FILM_EXECUTION_SCHEMA_VERSION,
    FactorFiLMExecutionConfig,
    FactorFiLMExecutionPaths,
    execute_factor_film_stage,
    validate_evaluator_git,
)
from langmani.policies.act_runtime import atomic_write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "outputs" / "datasets" / "m3b" / "langmani-pick-place-lerobot-v1"
)
DEFAULT_FACTOR_MODEL_ROOT = PROJECT_ROOT / "outputs" / "models" / "act-factor-film"
DEFAULT_M4_CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "models" / "act"
DEFAULT_TASK_TOKEN_ROOT = PROJECT_ROOT / "outputs" / "models" / "act-task-token"
DEFAULT_M42_DIAGNOSTICS_ROOT = PROJECT_ROOT / "outputs" / "diagnostics" / "m42"
DEFAULT_RUNTIME_SELECTION = (
    DEFAULT_M42_DIAGNOSTICS_ROOT / "runtime_ablation" / "runtime_selection.json"
)
DEFAULT_SEMANTIC_EVIDENCE_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "diagnostics"
    / "m43"
    / "semantic-audit"
    / "evidence"
    / "930ed848f8700a5ebc8734b1d3eb0fe22fd8fd4a3592c65825f2d813a32205e2"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "diagnostics" / "m43" / "factor-film-evaluation"
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "diagnostics" / "m43" / "evaluate-factor-film.json"


def _resolved_unlinked(path: Path, label: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    for component in (lexical, *lexical.parents):
        if component.is_symlink() or component.is_junction():
            raise RuntimeError(f"{label} traverses a symlink or junction: {component}")
    return lexical.resolve(strict=False)


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_command_report_path(args: argparse.Namespace) -> None:
    report = _resolved_unlinked(args.report, "FactorFiLM command report")
    output = _resolved_unlinked(args.output_root, "FactorFiLM evaluation output")
    protected = (
        output,
        *(
            PROJECT_ROOT / name
            for name in ("src", "scripts", "environment", "tests", "docs", ".git")
        ),
        args.dataset_root,
        args.factor_film_model_root,
        args.semantic_audit_evidence_root,
        args.m4_checkpoint_root,
        args.task_token_checkpoint_root,
        args.m42_diagnostics_root,
    )
    if any(_within(report, Path(root).resolve(strict=False)) for root in protected):
        raise RuntimeError(
            "command report must remain outside generated evidence, immutable inputs, and source"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", "validation", "reload", "development"),
        default="all",
    )
    parser.add_argument("--target-development", action="store_true", required=True)
    parser.add_argument(
        "--training-git-commit",
        required=True,
        help="exact clean Git commit that produced the authorized FactorFiLM training run",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--factor-film-model-root", type=Path, default=DEFAULT_FACTOR_MODEL_ROOT)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument(
        "--semantic-audit-evidence-root",
        type=Path,
        default=DEFAULT_SEMANTIC_EVIDENCE_ROOT,
    )
    parser.add_argument("--m4-checkpoint-root", type=Path, default=DEFAULT_M4_CHECKPOINT_ROOT)
    parser.add_argument("--task-token-checkpoint-root", type=Path, default=DEFAULT_TASK_TOKEN_ROOT)
    parser.add_argument("--m42-diagnostics-root", type=Path, default=DEFAULT_M42_DIAGNOSTICS_ROOT)
    parser.add_argument("--runtime-selection", type=Path, default=DEFAULT_RUNTIME_SELECTION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--clean-matching-staging", action="store_true")
    return parser.parse_args(argv)


def _config(args: argparse.Namespace, *, evaluation_git_commit: str) -> FactorFiLMExecutionConfig:
    return FactorFiLMExecutionConfig(
        paths=FactorFiLMExecutionPaths(
            dataset_root=args.dataset_root,
            factor_film_model_root=args.factor_film_model_root,
            semantic_audit_evidence_root=args.semantic_audit_evidence_root,
            m4_checkpoint_root=args.m4_checkpoint_root,
            task_token_checkpoint_root=args.task_token_checkpoint_root,
            m42_diagnostics_root=args.m42_diagnostics_root,
            runtime_selection_path=args.runtime_selection,
            output_root=args.output_root,
            run_root=args.run_root,
        ),
        evaluation_git_commit=evaluation_git_commit,
        expected_training_git_commit=args.training_git_commit,
        device=args.device,
        clean_matching_staging=args.clean_matching_staging,
    )


def _child_arguments(args: argparse.Namespace, stage: str) -> list[str]:
    values = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--target-development",
        "--stage",
        stage,
        "--device",
        args.device,
        "--training-git-commit",
        args.training_git_commit,
        "--dataset-root",
        str(args.dataset_root),
        "--factor-film-model-root",
        str(args.factor_film_model_root),
        "--semantic-audit-evidence-root",
        str(args.semantic_audit_evidence_root),
        "--m4-checkpoint-root",
        str(args.m4_checkpoint_root),
        "--task-token-checkpoint-root",
        str(args.task_token_checkpoint_root),
        "--m42-diagnostics-root",
        str(args.m42_diagnostics_root),
        "--runtime-selection",
        str(args.runtime_selection),
        "--output-root",
        str(args.output_root),
        "--report",
        str(args.report),
    ]
    if args.run_root is not None:
        values.extend(("--run-root", str(args.run_root)))
    if args.clean_matching_staging:
        values.append("--clean-matching-staging")
    return values


def execute(args: argparse.Namespace) -> dict[str, object]:
    _validate_command_report_path(args)
    git = validate_evaluator_git(PROJECT_ROOT)
    if args.stage == "all":
        completed: list[str] = []
        for stage in ("validation", "reload", "development"):
            result = subprocess.run(_child_arguments(args, stage), check=False)
            if result.returncode:
                raise RuntimeError(f"FactorFiLM {stage} subprocess exited with {result.returncode}")
            completed.append(stage)
        final = json.loads(args.report.read_text(encoding="utf-8"))
        if not isinstance(final, dict) or final.get("passed") is not True:
            raise RuntimeError("FactorFiLM development subprocess report is incomplete")
        return {
            **final,
            "stage": "all",
            "subprocess_stages": completed,
            "fresh_process_reload_validated": True,
            "evaluator_git": git,
        }
    commit = git.get("commit")
    if not isinstance(commit, str):
        raise RuntimeError("evaluator Git report lacks its full commit")
    result = execute_factor_film_stage(_config(args, evaluation_git_commit=commit), args.stage)
    return {
        "schema_version": FACTOR_FILM_EXECUTION_SCHEMA_VERSION,
        **result,
        "target_development": True,
        "evaluator_git": git,
        "test_split_accessed": False,
        "fresh_seed_accessed": False,
        "final_schedule_accessed": False,
        "smolvla_go": False,
    }


def main() -> int:
    args = parse_args()
    try:
        report = execute(args)
        atomic_write_json(args.report, report)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:  # noqa: BLE001 - outer command boundary preserves diagnostics
        failure = {
            "schema_version": FACTOR_FILM_EXECUTION_SCHEMA_VERSION,
            "passed": False,
            "stage": args.stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "test_split_accessed": False,
            "fresh_seed_accessed": False,
            "final_schedule_accessed": False,
            "smolvla_go": False,
        }
        try:
            _validate_command_report_path(args)
        except Exception:  # noqa: BLE001 - unsafe failure path must remain unwritten
            pass
        else:
            atomic_write_json(args.report, failure)
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
