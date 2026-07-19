"""Immutable access ledger and evidence lifecycle for the M5A sealed final."""

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
from langmani.language.sealed_final import SEALED_FINAL_EVIDENCE_SCHEMA


class SealedFinalEvidenceError(RuntimeError):
    """Raised when final access or immutable evidence is unsafe."""


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")
        )
        + "\n"
    ).encode("utf-8")


def real_unlinked(value: str | Path, *, label: str) -> Path:
    path = Path(os.path.abspath(os.fspath(Path(value).expanduser())))
    for item in (path, *path.parents):
        if item.is_symlink() or item.is_junction():
            raise SealedFinalEvidenceError(f"{label} traverses a link")
    return path.resolve(strict=False)


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.encode("utf-8") if isinstance(value, str) else _json_bytes(value)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def read_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SealedFinalEvidenceError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise SealedFinalEvidenceError(f"{label} must contain one JSON object")
    return cast(dict[str, object], value)


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
            raise SealedFinalEvidenceError("sealed-final evidence contains a linked entry")
        if path.is_dir():
            continue
        relative = PurePosixPath(path.relative_to(root)).as_posix()
        if relative in {"manifest.json", "complete.json"}:
            continue
        records.append(
            {"path": relative, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    return records


def final_access_count(output_root: str | Path) -> int:
    root = real_unlinked(output_root, label="sealed-final output")
    ledger = root / "access-ledger"
    if not ledger.exists():
        return 0
    if not ledger.is_dir():
        raise SealedFinalEvidenceError("final access ledger is not a directory")
    attempts = [item for item in ledger.iterdir() if item.is_dir()]
    if any(item.is_symlink() or item.is_junction() for item in attempts):
        raise SealedFinalEvidenceError("final access ledger contains linked attempts")
    return len(attempts)


def open_final_attempt(
    output_root: str | Path,
    *,
    run_identity: Mapping[str, object],
) -> dict[str, object]:
    """Record the one access before any final command or seed is materialized."""

    root = real_unlinked(output_root, label="sealed-final output")
    if final_access_count(root) != 0:
        raise SealedFinalEvidenceError("sealed final already has an access attempt")
    run_fingerprint = run_identity.get("run_fingerprint")
    if not isinstance(run_fingerprint, str) or not run_fingerprint.startswith("sha256:"):
        raise SealedFinalEvidenceError("final attempt requires a run fingerprint")
    attempt_root = root / "access-ledger" / run_fingerprint.removeprefix("sha256:")
    if attempt_root.exists():
        raise SealedFinalEvidenceError("final attempt root already exists")
    attempt_root.mkdir(parents=True, exist_ok=False)
    base: dict[str, object] = {
        "schema_version": "langmani-m5a-final-attempt-open-v0",
        "attempt_index": 1,
        "run_identity": dict(run_identity),
        "single_valid_attempt_policy": True,
        "final_access_opened": True,
    }
    opened = {**base, "attempt_fingerprint": f"sha256:{sha256_hex(base)}"}
    write_new(attempt_root / "opened.json", opened)
    return {**opened, "attempt_root": str(attempt_root)}


def close_final_attempt(
    attempt_root: str | Path,
    *,
    completed: bool,
    evidence_fingerprint: str | None = None,
    error: Mapping[str, object] | None = None,
) -> dict[str, object]:
    root = real_unlinked(attempt_root, label="sealed-final attempt")
    opened = read_object(root / "opened.json", label="final attempt owner")
    if (root / "completed.json").exists() or (root / "invalid.json").exists():
        raise SealedFinalEvidenceError("final attempt already has a terminal marker")
    if completed and (not isinstance(evidence_fingerprint, str) or error is not None):
        raise SealedFinalEvidenceError("completed final attempt requires only evidence identity")
    if not completed and (evidence_fingerprint is not None or not isinstance(error, Mapping)):
        raise SealedFinalEvidenceError("invalid final attempt requires exact failure evidence")
    base: dict[str, object] = {
        "schema_version": (
            "langmani-m5a-final-attempt-completed-v0"
            if completed
            else "langmani-m5a-final-attempt-invalid-v0"
        ),
        "attempt_fingerprint": opened["attempt_fingerprint"],
        "completed": completed,
        "evidence_fingerprint": evidence_fingerprint,
        "error": None if error is None else dict(error),
    }
    marker = {**base, "terminal_fingerprint": f"sha256:{sha256_hex(base)}"}
    write_new(root / ("completed.json" if completed else "invalid.json"), marker)
    return marker


def validate_sealed_final_evidence(
    evidence_root: str | Path,
    *,
    expected_owner: Mapping[str, object] | None = None,
) -> dict[str, object]:
    root = real_unlinked(evidence_root, label="sealed-final evidence")
    if not root.is_dir():
        raise SealedFinalEvidenceError("sealed-final evidence root is missing")
    owner = read_object(root / "owner.json", label="final owner")
    manifest = read_object(root / "manifest.json", label="final manifest")
    complete = read_object(root / "complete.json", label="final completion")
    if expected_owner is not None and owner != dict(expected_owner):
        raise SealedFinalEvidenceError("sealed-final owner differs")
    records = _records(root)
    artifact_fingerprint = f"sha256:{sha256_hex({'owner': owner, 'artifacts': records})}"
    if manifest != {
        "schema_version": SEALED_FINAL_EVIDENCE_SCHEMA,
        "artifact_fingerprint": artifact_fingerprint,
        "artifacts": records,
    }:
        raise SealedFinalEvidenceError("sealed-final manifest checksums differ")
    base = dict(complete)
    declared = base.pop("completion_fingerprint", None)
    if (
        declared != f"sha256:{sha256_hex(base)}"
        or complete.get("schema_version") != SEALED_FINAL_EVIDENCE_SCHEMA
        or complete.get("passed") is not True
        or complete.get("artifact_fingerprint") != artifact_fingerprint
        or complete.get("run_fingerprint") != owner.get("run_fingerprint")
        or not isinstance(complete.get("flags"), dict)
    ):
        raise SealedFinalEvidenceError("sealed-final completion differs")
    if (root / "complete.json").stat().st_mtime_ns < (root / "manifest.json").stat().st_mtime_ns:
        raise SealedFinalEvidenceError("sealed-final completion was not written last")
    return {
        "passed": True,
        "root": str(root),
        "owner": owner,
        "manifest": manifest,
        "complete": complete,
        "artifact_fingerprint": artifact_fingerprint,
    }


def write_sealed_final_evidence(
    output_root: str | Path,
    *,
    owner: Mapping[str, object],
    artifacts: Mapping[str, object],
    flags: Mapping[str, bool],
) -> dict[str, object]:
    """Stage, fsync, checksum, and atomically promote one final archive."""

    root = real_unlinked(output_root, label="sealed-final output")
    run_fingerprint = owner.get("run_fingerprint")
    if not isinstance(run_fingerprint, str) or not run_fingerprint.startswith("sha256:"):
        raise SealedFinalEvidenceError("sealed-final owner lacks its run fingerprint")
    destination = root / run_fingerprint.removeprefix("sha256:")
    if destination.exists():
        raise SealedFinalEvidenceError("completed sealed-final output is immutable")
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".{destination.name}.{uuid.uuid4().hex}.staging"
    staging.mkdir(exist_ok=False)
    try:
        write_new(staging / "owner.json", dict(owner))
        for relative, value in sorted(artifacts.items()):
            pure = PurePosixPath(relative)
            if pure.is_absolute() or ".." in pure.parts or pure.suffix not in {".json", ".md"}:
                raise SealedFinalEvidenceError("sealed-final artifact path is unsafe")
            if pure.name in {"owner.json", "manifest.json", "complete.json"}:
                raise SealedFinalEvidenceError("sealed-final artifact uses a reserved name")
            write_new(staging.joinpath(*pure.parts), value)
        records = _records(staging)
        artifact_fingerprint = f"sha256:{sha256_hex({'owner': dict(owner), 'artifacts': records})}"
        write_new(
            staging / "manifest.json",
            {
                "schema_version": SEALED_FINAL_EVIDENCE_SCHEMA,
                "artifact_fingerprint": artifact_fingerprint,
                "artifacts": records,
            },
        )
        completion_base: dict[str, object] = {
            "schema_version": SEALED_FINAL_EVIDENCE_SCHEMA,
            "run_fingerprint": run_fingerprint,
            "artifact_fingerprint": artifact_fingerprint,
            "flags": dict(flags),
            "passed": True,
        }
        write_new(
            staging / "complete.json",
            {
                **completion_base,
                "completion_fingerprint": f"sha256:{sha256_hex(completion_base)}",
            },
        )
        validate_sealed_final_evidence(staging, expected_owner=owner)
        if staging.stat().st_dev != root.stat().st_dev:
            raise SealedFinalEvidenceError(
                "final staging and destination use different filesystems"
            )
        os.replace(staging, destination)
        return validate_sealed_final_evidence(destination, expected_owner=owner)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "SealedFinalEvidenceError",
    "close_final_attempt",
    "final_access_count",
    "open_final_attempt",
    "read_object",
    "real_unlinked",
    "validate_sealed_final_evidence",
    "write_sealed_final_evidence",
]
