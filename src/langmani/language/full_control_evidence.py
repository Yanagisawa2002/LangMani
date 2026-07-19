"""Immutable evidence lifecycle for M5A full control development."""

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
from langmani.language.full_control_development import FULL_CONTROL_SCHEMA


class FullControlEvidenceError(RuntimeError):
    """Raised when full-development evidence is unsafe or inconsistent."""


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")
        )
        + "\n"
    ).encode("utf-8")


def _real_unlinked(value: str | Path, *, label: str) -> Path:
    path = Path(os.path.abspath(os.fspath(Path(value).expanduser())))
    for item in (path, *path.parents):
        if item.is_symlink() or item.is_junction():
            raise FullControlEvidenceError(f"{label} traverses a link")
    return path.resolve(strict=False)


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.encode("utf-8") if isinstance(value, str) else _json_bytes(value)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _records(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or path.is_junction():
            raise FullControlEvidenceError("full evidence contains a linked entry")
        if path.is_dir():
            continue
        relative = PurePosixPath(path.relative_to(root)).as_posix()
        if relative in {"manifest.json", "complete.json"}:
            continue
        records.append(
            {"path": relative, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    return records


def _object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FullControlEvidenceError(f"cannot read {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise FullControlEvidenceError(f"{path.name} must contain one JSON object")
    return cast(dict[str, object], value)


def validate_full_control_evidence(
    evidence_root: str | Path,
    *,
    expected_owner: Mapping[str, object] | None = None,
) -> dict[str, object]:
    root = _real_unlinked(evidence_root, label="full-development evidence")
    if not root.is_dir():
        raise FullControlEvidenceError("full-development evidence root is missing")
    owner = _object(root / "owner.json")
    manifest = _object(root / "manifest.json")
    complete = _object(root / "complete.json")
    if expected_owner is not None and owner != dict(expected_owner):
        raise FullControlEvidenceError("full-development owner differs")
    records = _records(root)
    artifact_fingerprint = f"sha256:{sha256_hex({'owner': owner, 'artifacts': records})}"
    if manifest != {
        "schema_version": FULL_CONTROL_SCHEMA,
        "artifact_fingerprint": artifact_fingerprint,
        "artifacts": records,
    }:
        raise FullControlEvidenceError("full-development manifest checksums differ")
    completion_base = dict(complete)
    completion_fingerprint = completion_base.pop("completion_fingerprint", None)
    if (
        completion_fingerprint != f"sha256:{sha256_hex(completion_base)}"
        or complete.get("schema_version") != FULL_CONTROL_SCHEMA
        or complete.get("passed") is not True
        or complete.get("artifact_fingerprint") != artifact_fingerprint
        or complete.get("run_fingerprint") != owner.get("run_fingerprint")
        or not isinstance(complete.get("flags"), dict)
    ):
        raise FullControlEvidenceError("full-development completion marker differs")
    if (root / "complete.json").stat().st_mtime_ns < (root / "manifest.json").stat().st_mtime_ns:
        raise FullControlEvidenceError("full-development completion marker was not written last")
    return {
        "passed": True,
        "root": str(root),
        "owner": owner,
        "manifest": manifest,
        "complete": complete,
        "artifact_fingerprint": artifact_fingerprint,
    }


def write_full_control_evidence(
    output_root: str | Path,
    *,
    owner: Mapping[str, object],
    artifacts: Mapping[str, object],
    flags: Mapping[str, bool],
) -> dict[str, object]:
    """Stage, fsync, validate, and atomically promote completed full evidence."""

    root = _real_unlinked(output_root, label="full-development output")
    run_fingerprint = owner.get("run_fingerprint")
    if not isinstance(run_fingerprint, str) or not run_fingerprint.startswith("sha256:"):
        raise FullControlEvidenceError("owner requires a SHA-256 run fingerprint")
    token = run_fingerprint.removeprefix("sha256:")
    destination = root / token
    if destination.exists():
        return {
            **validate_full_control_evidence(destination, expected_owner=owner),
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
                raise FullControlEvidenceError("full-development artifact path is unsafe")
            if pure.name in {"owner.json", "manifest.json", "complete.json"}:
                raise FullControlEvidenceError("full-development artifact uses a reserved name")
            _write_new(staging.joinpath(*pure.parts), value)
        records = _records(staging)
        artifact_fingerprint = f"sha256:{sha256_hex({'owner': dict(owner), 'artifacts': records})}"
        _write_new(
            staging / "manifest.json",
            {
                "schema_version": FULL_CONTROL_SCHEMA,
                "artifact_fingerprint": artifact_fingerprint,
                "artifacts": records,
            },
        )
        completion_base: dict[str, object] = {
            "schema_version": FULL_CONTROL_SCHEMA,
            "run_fingerprint": run_fingerprint,
            "artifact_fingerprint": artifact_fingerprint,
            "flags": dict(flags),
            "passed": True,
        }
        _write_new(
            staging / "complete.json",
            {
                **completion_base,
                "completion_fingerprint": f"sha256:{sha256_hex(completion_base)}",
            },
        )
        validate_full_control_evidence(staging, expected_owner=owner)
        if staging.stat().st_dev != root.stat().st_dev:
            raise FullControlEvidenceError("staging and destination use different filesystems")
        os.replace(staging, destination)
        return {
            **validate_full_control_evidence(destination, expected_owner=owner),
            "evidence_reused": False,
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "FullControlEvidenceError",
    "validate_full_control_evidence",
    "write_full_control_evidence",
]
