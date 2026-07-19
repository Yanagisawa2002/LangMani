"""Export one deterministic accepted PerTask controller binding for LatentGuard."""

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
    LatentGuardBridgeError,
    export_policy_binding,
    write_envelope,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the path-only policy binding command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-registry", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--runtime-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Export and strictly persist the selected controller identity."""

    args = parse_args(argv)
    try:
        report = export_policy_binding(
            registry_path=args.controller_registry,
            checkpoint_root=args.checkpoint_root,
            dataset_root=args.dataset_root,
            runtime_selection_path=args.runtime_selection,
            repository_root=PROJECT_ROOT,
        )
        write_envelope(args.output, report)
        payload = report["payload"]
        assert isinstance(payload, dict)
        print(
            "export-latentguard-policy-binding OK "
            f"task={payload['task_id']} digest={report['content_digest']}"
        )
        return 0
    except (LatentGuardBridgeError, OSError, TypeError, ValueError) as error:
        print(f"export-latentguard-policy-binding failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
