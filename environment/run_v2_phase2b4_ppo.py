"""Run one explicitly selected Phase 2B.4-F0 PPO stage on the native CUDA target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langmani.v2.phase2b4_ppo import PPOConfig
from langmani.v2.phase2b4_runtime import (
    evaluate_checkpoint,
    ppo_smoke_bundle,
    run_ppo_training,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "diagnostics" / "v2" / "phase2b4_f0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--smoke", action="store_true")
    stage.add_argument("--train-micro", action="store_true")
    stage.add_argument("--evaluate-micro", type=Path, metavar="CHECKPOINT")
    stage.add_argument("--evaluate-development", type=Path, metavar="CHECKPOINT")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    if args.smoke:
        report = ppo_smoke_bundle(output_dir=output_root / "smoke")
        report_path = output_root / "ppo_smoke_test_result.json"
    elif args.train_micro:
        report = run_ppo_training(
            config=PPOConfig(),
            output_dir=output_root / "micro",
            smoke=False,
        )
        report_path = output_root / "micro_training_metrics.json"
    elif args.evaluate_micro is not None:
        report = evaluate_checkpoint(
            checkpoint=args.evaluate_micro.resolve(),
            mode="micro",
        )
        report_path = output_root / "micro_evaluation_result.json"
    elif args.evaluate_development is not None:
        report = evaluate_checkpoint(
            checkpoint=args.evaluate_development.resolve(),
            mode="development",
        )
        report_path = output_root / "feasibility_development_result.json"
    else:
        raise AssertionError("argparse did not select a Phase 2B.4-F0 stage")
    write_json(report_path, report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    succeeded = report.get("passed") is True or report.get("pipeline_execution_passed") is True
    return 0 if succeeded else 2


if __name__ == "__main__":
    raise SystemExit(main())
