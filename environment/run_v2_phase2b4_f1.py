"""Run one explicitly selected bounded Phase 2B.4-F1 diagnostic or PPO probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langmani.v2.phase2b4_f1_runtime import (
    audit_initial_exploration,
    compare_global_and_residual_physics,
    diagnose_f0_checkpoint,
    evaluate_probe_checkpoint,
    run_probe_training,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "diagnostics" / "v2" / "phase2b4_f1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--audit-initial-exploration", action="store_true")
    stage.add_argument("--diagnose-f0", type=Path, metavar="CHECKPOINT")
    stage.add_argument("--compare-actions", action="store_true")
    stage.add_argument("--train-probe-a", action="store_true")
    stage.add_argument("--evaluate-probe-a", type=Path, metavar="CHECKPOINT")
    stage.add_argument("--train-probe-b", action="store_true")
    stage.add_argument("--evaluate-probe-b", type=Path, metavar="CHECKPOINT")
    parser.add_argument("--probe-a-evaluation", type=Path)
    parser.add_argument("--reward-revision", type=int, choices=(0,), default=0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    if args.audit_initial_exploration:
        report = audit_initial_exploration()
        report_path = output_root / "exploration_distribution_audit.json"
    elif args.diagnose_f0 is not None:
        report = diagnose_f0_checkpoint(checkpoint=args.diagnose_f0.resolve())
        report_path = output_root / "contact_credit_audit.json"
    elif args.compare_actions:
        report = compare_global_and_residual_physics()
        report_path = output_root / "global_versus_residual_comparison.json"
    elif args.train_probe_a:
        if args.reward_revision != 0:
            raise PermissionError("Probe A is frozen to F0 reward revision 0")
        report = run_probe_training(
            probe="A",
            output_dir=output_root / "probe_a",
            reward_revision=0,
        )
        report_path = output_root / "probe_a_training_metrics.json"
    elif args.evaluate_probe_a is not None:
        report = evaluate_probe_checkpoint(
            probe="A",
            checkpoint=args.evaluate_probe_a.resolve(),
            reward_revision=0,
        )
        report_path = output_root / "probe_a_evaluation.json"
    elif args.train_probe_b:
        if args.probe_a_evaluation is None:
            raise PermissionError("Probe B requires an explicit Probe A evaluation report")
        probe_a = _read(args.probe_a_evaluation.resolve())
        if probe_a.get("passed") is not True:
            raise PermissionError("Probe A did not authorize Probe B")
        report = run_probe_training(
            probe="B",
            output_dir=output_root / "probe_b",
            reward_revision=args.reward_revision,
        )
        report_path = output_root / "probe_b_training_metrics.json"
    elif args.evaluate_probe_b is not None:
        report = evaluate_probe_checkpoint(
            probe="B",
            checkpoint=args.evaluate_probe_b.resolve(),
            reward_revision=args.reward_revision,
        )
        report_path = output_root / "probe_b_evaluation.json"
    else:
        raise AssertionError("argparse did not select an F1 stage")
    _write(report_path, report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    failed_gate = report.get("passed") is False
    failed_safety = report.get("residual_materially_safer") is False
    return 2 if failed_gate or failed_safety else 0


if __name__ == "__main__":
    raise SystemExit(main())
