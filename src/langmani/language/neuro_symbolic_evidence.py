"""Immutable lock and evidence lifecycle for the M5A.4 offline experiment."""

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

NEURO_SYMBOLIC_EVIDENCE_SCHEMA = "langmani-m5a4-neuro-symbolic-evidence-v0"
DEVELOPMENT_LOCK_SCHEMA = "langmani-m5a4-development-access-lock-v0"


class NeuroSymbolicEvidenceError(RuntimeError):
    """Raised when M5A.4 locks or evidence are unsafe or checksum-invalid."""


def _real_unlinked(path: str | Path, *, label: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    for component in (lexical, *lexical.parents):
        if component.is_symlink() or component.is_junction():
            raise NeuroSymbolicEvidenceError(f"{label} traverses a linked path")
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
        raise NeuroSymbolicEvidenceError(f"refusing to overwrite immutable file: {path}") from error


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NeuroSymbolicEvidenceError(
            f"could not read evidence object {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise NeuroSymbolicEvidenceError(f"evidence file is not one object: {path}")
    return cast(dict[str, object], value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _artifact_records(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink() or path.is_junction():
            raise NeuroSymbolicEvidenceError("evidence contains a linked entry")
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


def write_development_access_lock(
    output_root: str | Path,
    *,
    runtime_fingerprint: str,
    prompt_lock: Mapping[str, object],
) -> dict[str, object]:
    """Persist the immutable semantic lock before any development example is opened."""

    if not runtime_fingerprint.startswith("sha256:") or len(runtime_fingerprint) != 71:
        raise NeuroSymbolicEvidenceError("runtime fingerprint is malformed")
    root = _real_unlinked(output_root, label="M5A.4 output")
    token = runtime_fingerprint.removeprefix("sha256:")
    lock_root = root / ".locks" / token
    payload = {
        "schema_version": DEVELOPMENT_LOCK_SCHEMA,
        "runtime_fingerprint": runtime_fingerprint,
        "neuro_symbolic_router_locked": True,
        **dict(prompt_lock),
    }
    payload["lock_fingerprint"] = f"sha256:{sha256_hex(payload)}"
    path = lock_root / "prompt_lock.json"
    if path.exists():
        observed = _read_object(path)
        if observed != payload:
            raise NeuroSymbolicEvidenceError("existing development lock differs")
        return observed
    lock_root.mkdir(parents=True, exist_ok=False)
    _write_new(path, payload)
    return _read_object(path)


def validate_neuro_symbolic_evidence(
    evidence_root: str | Path,
    *,
    expected_owner: Mapping[str, object] | None = None,
) -> dict[str, object]:
    root = _real_unlinked(evidence_root, label="M5A.4 evidence")
    if not root.is_dir():
        raise NeuroSymbolicEvidenceError("M5A.4 evidence root is missing")
    owner = _read_object(root / "owner.json")
    manifest = _read_object(root / "manifest.json")
    complete = _read_object(root / "complete.json")
    if expected_owner is not None and owner != dict(expected_owner):
        raise NeuroSymbolicEvidenceError("evidence owner differs")
    records = _artifact_records(root)
    fingerprint = f"sha256:{sha256_hex({'owner': owner, 'artifacts': records})}"
    if manifest != {
        "schema_version": NEURO_SYMBOLIC_EVIDENCE_SCHEMA,
        "artifact_fingerprint": fingerprint,
        "artifacts": records,
    }:
        raise NeuroSymbolicEvidenceError("evidence checksums differ")
    if (
        complete.get("schema_version") != NEURO_SYMBOLIC_EVIDENCE_SCHEMA
        or complete.get("passed") is not True
        or complete.get("runtime_fingerprint") != owner.get("runtime_fingerprint")
        or complete.get("artifact_fingerprint") != fingerprint
        or not isinstance(complete.get("flags"), dict)
    ):
        raise NeuroSymbolicEvidenceError("completion marker differs")
    if (root / "complete.json").stat().st_mtime_ns < (root / "manifest.json").stat().st_mtime_ns:
        raise NeuroSymbolicEvidenceError("completion marker was not written last")
    return {
        "passed": True,
        "root": str(root),
        "owner": owner,
        "complete": complete,
        "artifacts": records,
        "artifact_fingerprint": fingerprint,
    }


def write_neuro_symbolic_evidence(
    output_root: str | Path,
    *,
    owner: Mapping[str, object],
    artifacts: Mapping[str, object],
    flags: Mapping[str, bool],
) -> dict[str, object]:
    """Write immutable evidence through same-filesystem staging and atomic promotion."""

    root = _real_unlinked(output_root, label="M5A.4 output")
    runtime_fingerprint = owner.get("runtime_fingerprint")
    if not isinstance(runtime_fingerprint, str) or len(runtime_fingerprint) != 71:
        raise NeuroSymbolicEvidenceError("owner requires one SHA-256 runtime fingerprint")
    token = runtime_fingerprint.removeprefix("sha256:")
    destination = root / token
    if destination.exists():
        return {
            **validate_neuro_symbolic_evidence(destination, expected_owner=owner),
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
                raise NeuroSymbolicEvidenceError("artifact path is unsafe")
            if pure.name in {"owner.json", "manifest.json", "complete.json"}:
                raise NeuroSymbolicEvidenceError("artifact uses a reserved filename")
            _write_new(staging.joinpath(*pure.parts), value)
        records = _artifact_records(staging)
        fingerprint = f"sha256:{sha256_hex({'owner': dict(owner), 'artifacts': records})}"
        _write_new(
            staging / "manifest.json",
            {
                "schema_version": NEURO_SYMBOLIC_EVIDENCE_SCHEMA,
                "artifact_fingerprint": fingerprint,
                "artifacts": records,
            },
        )
        _write_new(
            staging / "complete.json",
            {
                "schema_version": NEURO_SYMBOLIC_EVIDENCE_SCHEMA,
                "runtime_fingerprint": runtime_fingerprint,
                "artifact_fingerprint": fingerprint,
                "flags": dict(flags),
                "passed": True,
            },
        )
        validate_neuro_symbolic_evidence(staging, expected_owner=owner)
        if staging.stat().st_dev != root.stat().st_dev:
            raise NeuroSymbolicEvidenceError("staging and output are on different filesystems")
        os.replace(staging, destination)
        return {
            **validate_neuro_symbolic_evidence(destination, expected_owner=owner),
            "evidence_reused": False,
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "DEVELOPMENT_LOCK_SCHEMA",
    "NEURO_SYMBOLIC_EVIDENCE_SCHEMA",
    "NeuroSymbolicEvidenceError",
    "validate_neuro_symbolic_evidence",
    "write_development_access_lock",
    "write_neuro_symbolic_evidence",
]
