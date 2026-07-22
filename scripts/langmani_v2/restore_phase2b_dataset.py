"""Atomically restore one already accepted Phase 2B dataset package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langmani.v2.phase2c_recovery import (
    load_dataset_package,
    restore_dataset_atomically,
    verify_dataset_package,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the atomic recovery command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Verify, stage, atomically activate, and reverify exact dataset bytes."""

    args = parse_args(argv)
    package = load_dataset_package(args.package.resolve())
    restore_dataset_atomically(
        source_root=args.source.resolve(),
        destination_root=args.destination.resolve(),
        package=package,
    )
    verify_dataset_package(package, dataset_root=args.destination.resolve())
    result = {
        "schema_version": "langmani-v2-phase2c1-atomic-recovery-v0",
        "semantic_sha256": package.semantic_sha256,
        "destination_root_name": args.destination.name,
        "passed": True,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
