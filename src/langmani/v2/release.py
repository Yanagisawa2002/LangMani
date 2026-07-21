"""Simple validation for the immutable, source-controlled LangMani v1 release."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

V1_RELEASE_SCHEMA = "langmani-v1-release-manifest-v0"


class V1ReleaseError(RuntimeError):
    """Raised when the frozen v1 tag or evidence files have changed."""


def _read_manifest(path: Path) -> dict[str, Any]:
    """Read the manifest's canonical JSON, which is valid YAML 1.2 syntax."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V1ReleaseError(f"cannot read v1 release manifest {path}: {error}") from error
    if not isinstance(value, dict) or value.get("schema_version") != V1_RELEASE_SCHEMA:
        raise V1ReleaseError("v1 release manifest is not the supported object schema")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise V1ReleaseError(f"cannot hash frozen evidence {path}: {error}") from error
    return digest.hexdigest()


def _git(project_root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise V1ReleaseError(f"cannot inspect Git v1 release identity: {error}") from error


def _safe_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise V1ReleaseError("frozen evidence path must be a non-empty string")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "\\" in value:
        raise V1ReleaseError("frozen evidence path must be repository-relative")
    path = (root / pure).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise V1ReleaseError("frozen evidence path escapes the repository") from error
    return path


@dataclass(frozen=True, slots=True)
class V1ReleaseValidation:
    manifest_path: str
    release_tag: str
    expected_commit: str
    resolved_commit: str
    frozen_file_count: int
    frozen_files_valid: bool
    identities_valid: bool
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "langmani-v1-release-validation-v0",
            "manifest_path": self.manifest_path,
            "release_tag": self.release_tag,
            "expected_commit": self.expected_commit,
            "resolved_commit": self.resolved_commit,
            "frozen_file_count": self.frozen_file_count,
            "frozen_files_valid": self.frozen_files_valid,
            "identities_valid": self.identities_valid,
            "passed": self.passed,
        }


def validate_v1_release(*, project_root: Path, manifest_path: Path) -> V1ReleaseValidation:
    """Fail closed on tag drift, evidence edits, or malformed frozen identities."""

    root = project_root.resolve()
    manifest = _read_manifest(manifest_path)
    release = manifest.get("release")
    files = manifest.get("frozen_files")
    identities = manifest.get("artifact_identities")
    if not isinstance(release, Mapping) or not isinstance(files, list):
        raise V1ReleaseError("v1 release manifest lacks release or frozen_files")
    if not isinstance(identities, Mapping):
        raise V1ReleaseError("v1 release manifest lacks artifact_identities")
    tag = release.get("tag")
    expected_commit = release.get("commit")
    if not isinstance(tag, str) or not isinstance(expected_commit, str):
        raise V1ReleaseError("v1 release tag and commit must be strings")
    resolved_commit = _git(root, "rev-parse", f"{tag}^{{}}")
    if resolved_commit != expected_commit:
        raise V1ReleaseError(
            f"v1 release tag {tag} resolves to {resolved_commit}, expected {expected_commit}"
        )
    seen: set[str] = set()
    for record in files:
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise V1ReleaseError("frozen file records must contain only path and sha256")
        relative = record["path"]
        expected = record["sha256"]
        if not isinstance(relative, str) or relative in seen:
            raise V1ReleaseError("frozen evidence paths must be unique strings")
        if not isinstance(expected, str) or len(expected) != 64:
            raise V1ReleaseError("frozen evidence hashes must be SHA-256 strings")
        seen.add(relative)
        actual = _sha256_file(_safe_path(root, relative))
        if actual != expected:
            raise V1ReleaseError(f"frozen v1 evidence changed: {relative}")
    for label, value in identities.items():
        if not isinstance(label, str):
            raise V1ReleaseError("artifact identity labels must be strings")
        if isinstance(value, str) and value.startswith("sha256:") and len(value) != 71:
            raise V1ReleaseError(f"artifact identity {label} is not a SHA-256 digest")
    relative_manifest = manifest_path.resolve().relative_to(root).as_posix()
    return V1ReleaseValidation(
        manifest_path=relative_manifest,
        release_tag=tag,
        expected_commit=expected_commit,
        resolved_commit=resolved_commit,
        frozen_file_count=len(files),
        frozen_files_valid=True,
        identities_valid=True,
        passed=True,
    )


__all__ = [
    "V1_RELEASE_SCHEMA",
    "V1ReleaseError",
    "V1ReleaseValidation",
    "validate_v1_release",
]
