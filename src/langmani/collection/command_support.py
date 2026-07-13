"""Safe output-path and JSON helpers for M3A commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def validated_dataset_root(path: str | Path, *, output_root: str | Path) -> Path:
    """Require generated M3A archives to live below the ignored outputs tree."""
    allowed = Path(output_root).resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(allowed)
    except ValueError as error:
        raise ValueError(f"raw dataset root must stay under {allowed}") from error
    if relative == Path("."):
        raise ValueError("raw dataset root must be a child of outputs, not outputs itself")
    return resolved


def write_json_atomic(path: str | Path, payload: Any) -> Path:
    """Write a small command report without a partial JSON window."""
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


__all__ = ["validated_dataset_root", "write_json_atomic"]
