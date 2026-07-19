"""Strictly audit the available stages of one LatentGuard bridge probe."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from langmani.integrations.latentguard_bridge import (  # noqa: E402
    CANDIDATE_REQUEST_SCHEMA,
    INITIAL_PROPOSAL_SCHEMA,
    POLICY_BINDING_SCHEMA,
    PROJECTED_CANDIDATES_SCHEMA,
    LatentGuardBridgeError,
    audit_bridge,
    read_envelope,
    write_envelope,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse bridge artifact paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-binding", type=Path, required=True)
    parser.add_argument("--initial-proposal", type=Path)
    parser.add_argument("--raw-candidates", type=Path)
    parser.add_argument("--projected-candidates", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def _optional(path: Path | None, schema: str) -> dict[str, object] | None:
    if path is None:
        return None
    return read_envelope(path, expected_schema=schema)[0]


def main(argv: Sequence[str] | None = None) -> int:
    """Audit and persist the compact zero-action bridge result."""

    args = parse_args(argv)
    try:
        binding = read_envelope(args.policy_binding, expected_schema=POLICY_BINDING_SCHEMA)[0]
        report = audit_bridge(
            binding_envelope=binding,
            proposal_envelope=_optional(args.initial_proposal, INITIAL_PROPOSAL_SCHEMA),
            candidate_envelope=_optional(args.raw_candidates, CANDIDATE_REQUEST_SCHEMA),
            projected_envelope=_optional(args.projected_candidates, PROJECTED_CANDIDATES_SCHEMA),
        )
        write_envelope(args.output, report)
        payload = report["payload"]
        assert isinstance(payload, dict)
        passed = payload["passed"] is True
        print(
            "audit-latentguard-bridge "
            f"{'OK' if passed else 'BLOCKED'} digest={report['content_digest']}"
        )
        return 0 if passed or not args.strict else 1
    except (LatentGuardBridgeError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"audit-latentguard-bridge failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
