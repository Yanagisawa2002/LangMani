"""Verify an accepted Phase 2B package against source or restored bytes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langmani.v2.phase2c_recovery import load_dataset_package, verify_dataset_package


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse package-validation arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Recompute byte and schema identity and return nonzero on drift."""

    args = parse_args(argv)
    package = load_dataset_package(args.package.resolve())
    verify_dataset_package(package, dataset_root=args.dataset_root)
    result = {
        "schema_version": "langmani-v2-phase2c1-package-verification-v0",
        "semantic_sha256": package.semantic_sha256,
        "passed": True,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
