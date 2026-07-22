"""Write the frozen and disjoint Phase 2B.3 seed registry."""

from __future__ import annotations

import argparse
from pathlib import Path

from langmani.v2.push_expert_recovery import build_seed_registry, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2b3" / "seed_registry.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = build_seed_registry()
    write_json(args.output, registry)
    print(registry["registry_digest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
