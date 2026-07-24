"""Independently verify compact Phase 2B.6-v2 Result A evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langmani.v2.phase2b6_v2_finalize import verify_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_ROOT = PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2b6_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    return parser.parse_args()


def main() -> int:
    result = verify_artifacts(parse_args().evidence_root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
