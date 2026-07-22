"""Build the content-bound Candidate E--P failure registry."""

from __future__ import annotations

import argparse
from pathlib import Path

from langmani.v2.push_expert_recovery import (
    build_failure_registry,
    load_json,
    root_cause_summary,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "langmani_v2" / "phase_2b3" / "failure_audit.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root-cause-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = build_failure_registry(config=load_json(args.config), history_root=args.history_root)
    write_json(args.output, registry)
    write_json(args.root_cause_output, root_cause_summary(registry))
    print(
        f"historical={registry['historical_result_count']} "
        f"failures={registry['failure_count']} digest={registry['records_digest']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
