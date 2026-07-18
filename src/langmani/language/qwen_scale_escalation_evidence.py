"""Atomic immutable evidence lifecycle for the M5A.3 capacity comparison."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import cast

from langmani.datasets.identity import sha256_hex

QWEN_SCALE_EVIDENCE_SCHEMA = "langmani-m5a3-qwen-scale-evidence-v0"


class QwenScaleEvidenceError(RuntimeError):
    """Raised when M5A.3 evidence is unsafe, mutable, or checksum-invalid."""


def _real_path(path: str | Path, *, label: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    for component in (lexical, *lexical.parents):
        if component.is_symlink() or component.is_junction():
            raise QwenScaleEvidenceError(f"{label} traverses a linked path")
    return lexical.resolve(strict=False)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")
        )
        + "\n"
    ).encode("utf-8")


def _write_new(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(
                payload.encode("utf-8") if isinstance(payload, str) else _json_bytes(payload)
            )
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise QwenScaleEvidenceError(f"refusing to overwrite immutable evidence: {path}") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _records(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()):
        if path.is_symlink() or path.is_junction():
            raise QwenScaleEvidenceError("evidence contains a linked entry")
        if not path.is_file() or path.name in {"manifest.json", "complete.json"}:
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return records


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QwenScaleEvidenceError(f"could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise QwenScaleEvidenceError(f"evidence file must be one object: {path}")
    return cast(dict[str, object], value)


def validate_qwen_scale_evidence(
    evidence_root: str | Path,
    *,
    expected_owner: Mapping[str, object] | None = None,
) -> dict[str, object]:
    root = _real_path(evidence_root, label="M5A.3 evidence")
    if not root.is_dir():
        raise QwenScaleEvidenceError("M5A.3 evidence root is missing")
    owner = _read_object(root / "owner.json")
    manifest = _read_object(root / "manifest.json")
    complete = _read_object(root / "complete.json")
    if expected_owner is not None and owner != dict(expected_owner):
        raise QwenScaleEvidenceError("M5A.3 evidence owner differs")
    records = _records(root)
    fingerprint = f"sha256:{sha256_hex({'owner': owner, 'artifacts': records})}"
    if manifest != {
        "schema_version": QWEN_SCALE_EVIDENCE_SCHEMA,
        "artifact_fingerprint": fingerprint,
        "artifacts": records,
    }:
        raise QwenScaleEvidenceError("M5A.3 manifest/checksums differ")
    if (
        complete.get("schema_version") != QWEN_SCALE_EVIDENCE_SCHEMA
        or complete.get("passed") is not True
        or complete.get("runtime_fingerprint") != owner.get("runtime_fingerprint")
        or complete.get("artifact_fingerprint") != fingerprint
        or not isinstance(complete.get("stopped_after_stage"), str)
        or not isinstance(complete.get("stopped_at_validation"), bool)
        or not isinstance(complete.get("flags"), dict)
    ):
        raise QwenScaleEvidenceError("M5A.3 completion marker differs")
    return {
        "passed": True,
        "root": str(root),
        "owner": owner,
        "complete": complete,
        "artifacts": records,
        "artifact_fingerprint": fingerprint,
    }


def write_qwen_scale_evidence(
    output_root: str | Path,
    *,
    owner: Mapping[str, object],
    artifacts: Mapping[str, object],
    stopped_after_stage: str,
    stopped_at_validation: bool,
    flags: Mapping[str, bool],
) -> dict[str, object]:
    root = _real_path(output_root, label="M5A.3 output root")
    runtime_fingerprint = owner.get("runtime_fingerprint")
    if (
        not isinstance(runtime_fingerprint, str)
        or not runtime_fingerprint.startswith("sha256:")
        or len(runtime_fingerprint) != 71
    ):
        raise QwenScaleEvidenceError("owner requires one runtime fingerprint")
    token = runtime_fingerprint.removeprefix("sha256:")
    destination = root / token
    if destination.exists():
        return {
            **validate_qwen_scale_evidence(destination, expected_owner=owner),
            "evidence_reused": True,
        }
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".{token}.{uuid.uuid4().hex}.staging"
    staging.mkdir(exist_ok=False)
    try:
        _write_new(staging / "owner.json", dict(owner))
        for relative, payload in sorted(artifacts.items()):
            pure = PurePosixPath(relative)
            if pure.is_absolute() or ".." in pure.parts or pure.suffix not in {".json", ".md"}:
                raise QwenScaleEvidenceError("artifact path must be a safe JSON or Markdown path")
            if pure.name in {"owner.json", "manifest.json", "complete.json"}:
                raise QwenScaleEvidenceError("reserved evidence filename")
            _write_new(staging.joinpath(*pure.parts), payload)
        records = _records(staging)
        fingerprint = f"sha256:{sha256_hex({'owner': dict(owner), 'artifacts': records})}"
        _write_new(
            staging / "manifest.json",
            {
                "schema_version": QWEN_SCALE_EVIDENCE_SCHEMA,
                "artifact_fingerprint": fingerprint,
                "artifacts": records,
            },
        )
        _write_new(
            staging / "complete.json",
            {
                "schema_version": QWEN_SCALE_EVIDENCE_SCHEMA,
                "runtime_fingerprint": runtime_fingerprint,
                "artifact_fingerprint": fingerprint,
                "passed": True,
                "stopped_after_stage": stopped_after_stage,
                "stopped_at_validation": stopped_at_validation,
                "flags": dict(flags),
            },
        )
        validate_qwen_scale_evidence(staging, expected_owner=owner)
        if staging.stat().st_dev != root.stat().st_dev:
            raise QwenScaleEvidenceError(
                "staging and evidence destination use different filesystems"
            )
        os.replace(staging, destination)
        return {
            **validate_qwen_scale_evidence(destination, expected_owner=owner),
            "evidence_reused": False,
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "QWEN_SCALE_EVIDENCE_SCHEMA",
    "QwenScaleEvidenceError",
    "validate_qwen_scale_evidence",
    "write_qwen_scale_evidence",
]
