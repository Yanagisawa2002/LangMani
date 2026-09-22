"""Reevaluate frozen M4B actions/checkpoint with independently validated stage diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from langmani.policies.m4b1_runtime import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.source, args.output, args.checkpoint, smoke=args.smoke)


if __name__ == "__main__":
    main()
