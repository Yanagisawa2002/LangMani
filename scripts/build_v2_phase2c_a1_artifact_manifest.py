"""Content-bind the compact source-controlled Phase 2C-A.1 artifact set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from langmani.datasets.identity import sha256_hex
from langmani.v2.phase2c_a import PACKAGE_FINGERPRINT


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.artifact_root.resolve()
    report = args.report.resolve()
    if not root.is_dir() or report.parent != root:
        raise RuntimeError("artifact manifest must be written directly inside its root")
    files = []
    for path in sorted(root.rglob("*")):
        if path == report or not path.is_file():
            continue
        if path.is_symlink():
            raise RuntimeError(f"compact artifact cannot be a symlink: {path}")
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not files:
        raise RuntimeError("compact artifact root is empty")
    semantic = {
        "schema_version": "langmani-v2-phase2c-a1-artifact-manifest-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "artifact_root": "artifacts/langmani_v2/phase_2c_a1",
        "file_count": len(files),
        "total_bytes": sum(int(record["size_bytes"]) for record in files),
        "files": files,
        "generated_datasets_committed": False,
        "model_checkpoints_committed": False,
        "videos_committed": False,
        "raw_episode_logs_committed": False,
        "external_evidence_referenced_by_content_hash": True,
    }
    result = {**semantic, "fingerprint": f"sha256:{sha256_hex(semantic)}"}
    if report.exists():
        existing = json.loads(report.read_text(encoding="utf-8"))
        if existing != result:
            raise RuntimeError(f"existing immutable artifact differs: {report}")
    else:
        report.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
