"""Verify the compact Phase 2B.3.1-RR Result D package without simulator execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langmani.v2.phase2b3_rr import verify_phase2b3_rr_result_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2b3_rr"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs" / "diagnostics" / "v2" / "phase2b3_rr" / "verification.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = verify_phase2b3_rr_result_artifacts(args.artifact_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
