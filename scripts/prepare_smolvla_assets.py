"""Download the two pinned public upstream snapshots and record their exact file hashes."""

from __future__ import annotations

import argparse
from pathlib import Path

from langmani.policies.m4a_data import file_digest, write_json
from langmani.policies.m4b_protocol import (
    BACKBONE_MODEL,
    BACKBONE_REVISION,
    BASE_MODEL,
    BASE_REVISION,
)


def main() -> None:
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    receipt = {}
    for label, repo, revision, patterns in (
        ("base", BASE_MODEL, BASE_REVISION, ["*.json", "*.safetensors"]),
        ("backbone", BACKBONE_MODEL, BACKBONE_REVISION, ["*.json", "*.txt"]),
    ):
        directory = root / f"{label}-{revision}"
        snapshot_download(
            repo_id=repo, revision=revision, local_dir=directory, allow_patterns=patterns
        )
        files = {
            p.relative_to(directory).as_posix(): file_digest(p)
            for p in sorted(directory.rglob("*"))
            if p.is_file() and ".cache" not in p.relative_to(directory).parts
        }
        receipt[label] = {
            "repo_id": repo,
            "revision": revision,
            "path": str(directory),
            "files": files,
        }
    write_json(root / "assets.json", receipt)


if __name__ == "__main__":
    main()
