"""Aggregate frozen Phase 2C-A.1 development and final ACT evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langmani.v2.phase2c_a import TASK_IDS
from langmani.v2.phase2c_a1_analysis import analyze_phase2c_a1_results

FINAL_SPLITS = ("test_unseen_reset", "test_visual_shift")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _load_scope(root: Path, scope: str) -> dict[str, dict[str, list[dict[str, object]]]]:
    result = {
        "validation": {
            task_id: _read_jsonl(root / "development" / scope / task_id / "episodes.jsonl")
            for task_id in TASK_IDS
        }
    }
    result.update(
        {
            split: {
                task_id: _read_jsonl(root / "final" / scope / split / task_id / "episodes.jsonl")
                for task_id in TASK_IDS
            }
            for split in FINAL_SPLITS
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--training-runs-complete", action="store_true")
    parser.add_argument("--checkpoints-recoverable", action="store_true")
    parser.add_argument("--padding-mask-verified", action="store_true")
    parser.add_argument("--closed-loop-smoke-passed", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.evaluation_root.resolve()
    result = analyze_phase2c_a1_results(
        per_task=_load_scope(root, "per_task"),
        shared=_load_scope(root, "shared"),
        training_runs_complete=args.training_runs_complete,
        checkpoints_recoverable=args.checkpoints_recoverable,
        padding_mask_verified=args.padding_mask_verified,
        closed_loop_smoke_passed=args.closed_loop_smoke_passed,
    )
    destination = args.report.resolve()
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
