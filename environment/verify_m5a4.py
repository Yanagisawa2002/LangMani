"""Independently verify one immutable M5A.4 neuro-symbolic evidence root."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for value in (PROJECT_ROOT, SRC_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from langmani.language.neuro_symbolic_verifier import (  # noqa: E402
    verify_neuro_symbolic_evidence,
)
from langmani.policies.act_runtime import atomic_write_json  # noqa: E402

DEFAULT_REPORT = PROJECT_ROOT / "outputs/diagnostics/m5a/neuro-symbolic-verification.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload: dict[str, object] = {"passed": False}
    try:
        payload = cast(dict[str, object], verify_neuro_symbolic_evidence(args.evidence_root))
    except Exception as error:  # noqa: BLE001 - verifier boundary preserves exact evidence
        traceback.print_exc()
        payload["error"] = {"type": type(error).__name__, "message": str(error)}
    atomic_write_json(args.report.resolve(strict=False), payload)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0 if payload.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
