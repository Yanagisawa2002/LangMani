"""Independently verify completed M4.3b FactorFiLM target-development evidence."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from langmani.policies.act_factor_film_verification import (  # noqa: E402
    M43B_INDEPENDENT_VERIFICATION_SCHEMA_VERSION,
    FactorFiLMIndependentVerificationReport,
    verify_factor_film_target_development,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run-root", type=Path, required=True)
    parser.add_argument("--evaluation-evidence-root", type=Path, required=True)
    parser.add_argument("--structural-verification", type=Path, required=True)
    parser.add_argument(
        "--training-git-commit",
        required=True,
        help="exact clean Git commit that produced the authorized FactorFiLM training run",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="optional separate report path; source artifacts remain read-only",
    )
    return parser.parse_args(argv)


def _write_report(
    path: Path,
    payload: dict[str, object],
    *,
    read_only_roots: tuple[Path, ...],
) -> None:
    destination = path.resolve(strict=False)
    source_roots = (
        (PROJECT_ROOT / "src").resolve(),
        (PROJECT_ROOT / "scripts").resolve(),
        (PROJECT_ROOT / "environment").resolve(),
        (PROJECT_ROOT / "tests").resolve(),
        (PROJECT_ROOT / "docs").resolve(),
        (PROJECT_ROOT / ".git").resolve(),
    )
    if any(destination == root or root in destination.parents for root in source_roots):
        raise RuntimeError("independent verification report cannot overwrite source or Git content")
    protected = tuple(root.resolve(strict=True) for root in read_only_roots)
    if any(destination == root or root in destination.parents for root in protected):
        raise RuntimeError(
            "independent verification report cannot modify training or evaluation evidence"
        )
    if destination.exists():
        raise RuntimeError("independent verification report is immutable and already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report: FactorFiLMIndependentVerificationReport | None = None
    try:
        report = verify_factor_film_target_development(
            training_run_root=args.training_run_root,
            evaluation_evidence_root=args.evaluation_evidence_root,
            expected_training_git_commit=args.training_git_commit,
            structural_verification_path=args.structural_verification,
        )
        payload = report.to_dict()
    except Exception as error:  # noqa: BLE001 - command boundary preserves diagnostics
        traceback.print_exc()
        payload = {
            "schema_version": M43B_INDEPENDENT_VERIFICATION_SCHEMA_VERSION,
            "passed": False,
            "error_type": type(error).__name__,
            "error_message": str(error) or repr(error),
            "implementation_validated": False,
            "factor_film_training_completed": False,
            "factor_film_checkpoints_complete": False,
            "factor_film_checkpoint_selected": False,
            "validation_only_selection_validated": False,
            "factor_film_reload_validated": False,
            "development_benchmark_completed": False,
            "development_quality_gate_passed": False,
            "final_benchmark_authorized": False,
            "final_schedule_accessed": False,
            "test_split_accessed": False,
            "fresh_seed_accessed": False,
            "smolvla_go": False,
            "physical_target_validated": False,
        }
    try:
        if args.report is not None:
            _write_report(
                args.report,
                payload,
                read_only_roots=(args.training_run_root, args.evaluation_evidence_root),
            )
    except Exception:
        traceback.print_exc()
        return 1
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0 if report is not None and report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
