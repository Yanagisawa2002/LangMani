"""Run one reset-only ACT query and export zero-action LatentGuard evidence."""

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
    POLICY_BINDING_SCHEMA,
    LatentGuardBridgeError,
    derive_probe_seed,
    export_initial_proposal,
    load_runtime_registry,
    read_envelope,
    write_envelope,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the one-probe command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-binding", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--runtime-selection", type=Path, required=True)
    parser.add_argument("--probe-seed", type=int, default=derive_probe_seed())
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Export exactly one proposal without calling the environment step API."""

    args = parse_args(argv)
    try:
        binding, _ = read_envelope(args.policy_binding, expected_schema=POLICY_BINDING_SCHEMA)
        registry, locators = load_runtime_registry(
            checkpoint_root=args.checkpoint_root,
            dataset_root=args.dataset_root,
            runtime_selection_path=args.runtime_selection,
        )
        report = export_initial_proposal(
            binding_envelope=binding,
            registry=registry,
            locators=locators,
            repository_root=PROJECT_ROOT,
            probe_seed=args.probe_seed,
            device=args.device,
        )
        write_envelope(args.output, report)
        print(
            "export-latentguard-initial-proposal OK "
            f"reset=1 steps=0 digest={report['content_digest']}"
        )
        return 0
    except (LatentGuardBridgeError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"export-latentguard-initial-proposal failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
