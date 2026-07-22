"""Audit explicitly named Phase 2B dataset candidates without modifying them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langmani.v2.phase2c import write_json_once
from langmani.v2.phase2c_recovery import audit_dataset_candidate, build_asset_audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the read-only asset-audit command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, action="append", default=[])
    parser.add_argument("--phase2b-result", type=Path)
    parser.add_argument("--verifier-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Write one immutable machine-readable candidate audit."""

    args = parse_args(argv)
    if (args.phase2b_result is None) != (args.verifier_report is None):
        raise ValueError("phase2b result and verifier report must be supplied together")
    candidates = [
        audit_dataset_candidate(
            path,
            phase2b_result_path=args.phase2b_result,
            verifier_report_path=args.verifier_report,
        )
        for path in args.candidate
    ]
    audit = build_asset_audit(candidates)
    write_json_once(args.output.resolve(), audit)
    print(json.dumps(audit, sort_keys=True))
    return 0 if audit["passed"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
