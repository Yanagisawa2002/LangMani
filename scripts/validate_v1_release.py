"""Validate the source-controlled LangMani v1 release freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langmani.v2.release import validate_v1_release

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "releases" / "langmani_v1" / "manifest.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate_v1_release(
        project_root=PROJECT_ROOT,
        manifest_path=args.manifest.resolve(),
    ).to_dict()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
