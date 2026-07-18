"""Immutable M5A.1 rejection-analysis artifact writing and validation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from langmani.datasets.identity import sha256_hex

REJECTION_ANALYSIS_ARTIFACT_SCHEMA = "langmani-m5a-rejection-analysis-artifact-v0"


class RejectionReportError(RuntimeError):
    """Raised when immutable M5A.1 evidence is malformed or would be overwritten."""


def _real_path(path: str | Path, *, label: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    for component in (lexical, *lexical.parents):
        if component.is_symlink() or component.is_junction():
            raise RejectionReportError(f"{label} traverses a linked path: {component}")
    return lexical.resolve(strict=False)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(values: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_json_bytes(dict(value)) for value in values)


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise RejectionReportError(f"refusing to overwrite immutable evidence: {path}") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _records(root: Path) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()):
        if path.is_symlink() or path.is_junction():
            raise RejectionReportError(f"analysis artifact contains a linked entry: {path}")
        if not path.is_file() or path.name in {"artifact_manifest.json", "complete.json"}:
            continue
        values.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return values


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RejectionReportError(f"could not read immutable evidence {path}: {error}") from error
    if not isinstance(value, dict):
        raise RejectionReportError(f"immutable evidence must be one object: {path}")
    return cast(dict[str, object], value)


def write_rejection_analysis_artifact(
    output_root: str | Path,
    *,
    json_artifacts: Mapping[str, Mapping[str, object]],
    per_example_records: Sequence[Mapping[str, object]],
    input_identity: Mapping[str, object],
    conclusion: str,
    classifier_full_quality_gate_passed: bool,
) -> dict[str, object]:
    root = _real_path(output_root, label="M5A.1 output root")
    if root.exists():
        raise RejectionReportError("immutable M5A.1 output root already exists")
    parent = root.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{root.name}.{uuid.uuid4().hex}.staging"
    staging.mkdir()
    try:
        for name, payload in sorted(json_artifacts.items()):
            if Path(name).name != name or not name.endswith(".json"):
                raise RejectionReportError("analysis JSON artifact names must be flat .json files")
            _write_new(staging / name, _json_bytes(dict(payload)))
        _write_new(staging / "per_example.jsonl", _jsonl_bytes(per_example_records))
        records = _records(staging)
        analysis_fingerprint = (
            f"sha256:{sha256_hex({'input_identity': dict(input_identity), 'artifacts': records})}"
        )
        manifest = {
            "schema_version": REJECTION_ANALYSIS_ARTIFACT_SCHEMA,
            "input_identity": dict(input_identity),
            "artifacts": records,
            "analysis_fingerprint": analysis_fingerprint,
        }
        complete = {
            "schema_version": REJECTION_ANALYSIS_ARTIFACT_SCHEMA,
            "passed": True,
            "analysis_fingerprint": analysis_fingerprint,
            "conclusion": conclusion,
            "classifier_full_quality_gate_passed": classifier_full_quality_gate_passed,
        }
        _write_new(staging / "artifact_manifest.json", _json_bytes(manifest))
        _write_new(staging / "complete.json", _json_bytes(complete))
        validate_rejection_analysis_artifact(staging)
        if staging.stat().st_dev != parent.stat().st_dev:
            raise RejectionReportError(
                "analysis staging and destination are on different filesystems"
            )
        os.replace(staging, root)
        return validate_rejection_analysis_artifact(root)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def validate_rejection_analysis_artifact(output_root: str | Path) -> dict[str, object]:
    root = _real_path(output_root, label="M5A.1 analysis artifact")
    if not root.is_dir():
        raise RejectionReportError("M5A.1 analysis artifact is missing")
    manifest = _read_object(root / "artifact_manifest.json")
    complete = _read_object(root / "complete.json")
    records = _records(root)
    input_identity = manifest.get("input_identity")
    if not isinstance(input_identity, Mapping):
        raise RejectionReportError("analysis input identity is malformed")
    expected = (
        f"sha256:{sha256_hex({'input_identity': dict(input_identity), 'artifacts': records})}"
    )
    if (
        manifest.get("schema_version") != REJECTION_ANALYSIS_ARTIFACT_SCHEMA
        or manifest.get("artifacts") != records
        or manifest.get("analysis_fingerprint") != expected
    ):
        raise RejectionReportError("analysis checksum manifest differs")
    conclusion = complete.get("conclusion")
    quality = complete.get("classifier_full_quality_gate_passed")
    expected_conclusion = (
        "classifier_promoted_after_posthoc_calibration"
        if quality is True
        else "classifier_rejected_after_posthoc_calibration"
    )
    if (
        complete.get("schema_version") != REJECTION_ANALYSIS_ARTIFACT_SCHEMA
        or complete.get("passed") is not True
        or complete.get("analysis_fingerprint") != expected
        or conclusion != expected_conclusion
        or not isinstance(quality, bool)
    ):
        raise RejectionReportError("analysis completion marker differs")
    return {
        "root": str(root),
        "analysis_fingerprint": expected,
        "conclusion": conclusion,
        "classifier_full_quality_gate_passed": quality,
        "artifacts": records,
        "input_identity": dict(input_identity),
        "passed": True,
    }


__all__ = [
    "REJECTION_ANALYSIS_ARTIFACT_SCHEMA",
    "RejectionReportError",
    "validate_rejection_analysis_artifact",
    "write_rejection_analysis_artifact",
]
