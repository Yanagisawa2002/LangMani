"""Immutable M5A.4.1 evidence and pre-development lock lifecycle."""

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

M5A41_EVIDENCE_SCHEMA = "langmani-m5a41-neuro-symbolic-safety-evidence-v1"
M5A41_LOCK_SCHEMA = "langmani-m5a41-neuro-symbolic-router-lock-v1"


class M5A41EvidenceError(RuntimeError):
    """Raised when M5A.4.1 evidence is mutable, linked, or checksum-invalid."""


def real_unlinked(path: str | Path, *, label: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    for component in (lexical, *lexical.parents):
        if component.is_symlink() or component.is_junction():
            raise M5A41EvidenceError(f"{label} traverses a linked path")
    return lexical.resolve(strict=False)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")
        )
        + "\n"
    ).encode("utf-8")


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(value.encode("utf-8") if isinstance(value, str) else _json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise M5A41EvidenceError(f"refusing to overwrite immutable file: {path}") from error


def read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise M5A41EvidenceError(f"could not read evidence object {path}: {error}") from error
    if not isinstance(value, dict):
        raise M5A41EvidenceError(f"evidence file is not one object: {path}")
    return cast(dict[str, object], value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _artifact_records(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink() or path.is_junction():
            raise M5A41EvidenceError("evidence contains a linked entry")
        if not path.is_file() or path.name in {"manifest.json", "complete.json"}:
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def write_router_lock(
    output_root: str | Path,
    *,
    runtime_fingerprint: str,
    lock_payload: Mapping[str, object],
) -> dict[str, object]:
    """Write the immutable router lock before validation or development is opened."""

    if not runtime_fingerprint.startswith("sha256:") or len(runtime_fingerprint) != 71:
        raise M5A41EvidenceError("runtime fingerprint is malformed")
    root = real_unlinked(output_root, label="M5A.4.1 output")
    token = runtime_fingerprint.removeprefix("sha256:")
    lock_root = root / ".locks" / token
    payload = {
        "schema_version": M5A41_LOCK_SCHEMA,
        "runtime_fingerprint": runtime_fingerprint,
        "neuro_symbolic_router_locked": True,
        "exact_rejection_taxonomy_not_guaranteed": True,
        **dict(lock_payload),
    }
    payload["lock_fingerprint"] = f"sha256:{sha256_hex(payload)}"
    path = lock_root / "router_lock.json"
    if path.exists():
        observed = read_object(path)
        if observed != payload:
            raise M5A41EvidenceError("existing M5A.4.1 router lock differs")
        return observed
    lock_root.mkdir(parents=True, exist_ok=False)
    _write_new(path, payload)
    return read_object(path)


def validate_m5a41_evidence(
    evidence_root: str | Path,
    *,
    expected_owner: Mapping[str, object] | None = None,
) -> dict[str, object]:
    root = real_unlinked(evidence_root, label="M5A.4.1 evidence")
    if not root.is_dir():
        raise M5A41EvidenceError("M5A.4.1 evidence root is missing")
    owner = read_object(root / "owner.json")
    manifest = read_object(root / "manifest.json")
    complete = read_object(root / "complete.json")
    if expected_owner is not None and owner != dict(expected_owner):
        raise M5A41EvidenceError("M5A.4.1 evidence owner differs")
    records = _artifact_records(root)
    fingerprint = f"sha256:{sha256_hex({'owner': owner, 'artifacts': records})}"
    if manifest != {
        "schema_version": M5A41_EVIDENCE_SCHEMA,
        "artifact_fingerprint": fingerprint,
        "artifacts": records,
    }:
        raise M5A41EvidenceError("M5A.4.1 evidence checksums differ")
    if (
        complete.get("schema_version") != M5A41_EVIDENCE_SCHEMA
        or complete.get("passed") is not True
        or complete.get("runtime_fingerprint") != owner.get("runtime_fingerprint")
        or complete.get("artifact_fingerprint") != fingerprint
        or not isinstance(complete.get("flags"), dict)
    ):
        raise M5A41EvidenceError("M5A.4.1 completion marker differs")
    if (root / "complete.json").stat().st_mtime_ns < (root / "manifest.json").stat().st_mtime_ns:
        raise M5A41EvidenceError("M5A.4.1 completion marker was not written last")
    return {
        "passed": True,
        "root": str(root),
        "owner": owner,
        "complete": complete,
        "artifacts": records,
        "artifact_fingerprint": fingerprint,
    }


def write_m5a41_evidence(
    output_root: str | Path,
    *,
    owner: Mapping[str, object],
    artifacts: Mapping[str, object],
    flags: Mapping[str, bool],
) -> dict[str, object]:
    """Write checksum-owned evidence through same-filesystem atomic promotion."""

    root = real_unlinked(output_root, label="M5A.4.1 output")
    runtime_fingerprint = owner.get("runtime_fingerprint")
    if not isinstance(runtime_fingerprint, str) or len(runtime_fingerprint) != 71:
        raise M5A41EvidenceError("M5A.4.1 owner requires one SHA-256 runtime fingerprint")
    token = runtime_fingerprint.removeprefix("sha256:")
    destination = root / token
    if destination.exists():
        return {
            **validate_m5a41_evidence(destination, expected_owner=owner),
            "evidence_reused": True,
        }
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".{token}.{uuid.uuid4().hex}.staging"
    staging.mkdir(exist_ok=False)
    try:
        _write_new(staging / "owner.json", dict(owner))
        for relative, value in sorted(artifacts.items()):
            pure = PurePosixPath(relative)
            if pure.is_absolute() or ".." in pure.parts or pure.suffix not in {".json", ".md"}:
                raise M5A41EvidenceError("M5A.4.1 artifact path is unsafe")
            if pure.name in {"owner.json", "manifest.json", "complete.json"}:
                raise M5A41EvidenceError("M5A.4.1 artifact uses a reserved filename")
            _write_new(staging.joinpath(*pure.parts), value)
        records = _artifact_records(staging)
        fingerprint = f"sha256:{sha256_hex({'owner': dict(owner), 'artifacts': records})}"
        _write_new(
            staging / "manifest.json",
            {
                "schema_version": M5A41_EVIDENCE_SCHEMA,
                "artifact_fingerprint": fingerprint,
                "artifacts": records,
            },
        )
        _write_new(
            staging / "complete.json",
            {
                "schema_version": M5A41_EVIDENCE_SCHEMA,
                "runtime_fingerprint": runtime_fingerprint,
                "artifact_fingerprint": fingerprint,
                "flags": dict(flags),
                "passed": True,
            },
        )
        validate_m5a41_evidence(staging, expected_owner=owner)
        if staging.stat().st_dev != root.stat().st_dev:
            raise M5A41EvidenceError("M5A.4.1 staging and output are on different filesystems")
        os.replace(staging, destination)
        return {
            **validate_m5a41_evidence(destination, expected_owner=owner),
            "evidence_reused": False,
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "M5A41_EVIDENCE_SCHEMA",
    "M5A41EvidenceError",
    "read_object",
    "real_unlinked",
    "sha256_file",
    "validate_m5a41_evidence",
    "write_m5a41_evidence",
    "write_router_lock",
]
