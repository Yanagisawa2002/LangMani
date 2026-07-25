"""Freeze one common ACT execution horizon from validation-only summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langmani.v2.phase2c_a_evaluator import select_execution_horizon


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain one JSON object")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = select_execution_horizon([_read_object(path.resolve()) for path in args.summary])
    destination = args.report.resolve()
    if destination.exists():
        if _read_object(destination) != result:
            raise RuntimeError(f"existing immutable artifact differs: {destination}")
    else:
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
