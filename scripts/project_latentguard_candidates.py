"""Project a frozen four-candidate request without opening a simulator."""

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
    POLICY_BINDING_SCHEMA,
    LatentGuardBridgeError,
    project_candidates,
    read_envelope,
    write_envelope,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the projection command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-binding", type=Path, required=True)
    parser.add_argument("--raw-candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate, project, and persist the four executable prefixes."""

    args = parse_args(argv)
    try:
        binding, _ = read_envelope(args.policy_binding, expected_schema=POLICY_BINDING_SCHEMA)
        candidates, _ = read_envelope(args.raw_candidates, expected_schema=CANDIDATE_REQUEST_SCHEMA)
        report = project_candidates(binding_envelope=binding, candidate_envelope=candidates)
        write_envelope(args.output, report)
        print(
            "project-latentguard-candidates OK "
            f"candidates=4 steps=0 digest={report['content_digest']}"
        )
        return 0
    except (LatentGuardBridgeError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"project-latentguard-candidates failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
