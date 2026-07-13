"""Atomic manifest persistence and resume checks for M3A collections."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from langmani.datasets.schedule import build_collection_schedule
from langmani.datasets.types import (
    CollectionConfig,
    CollectionManifest,
    CollectionStatus,
    RawDatasetSummary,
    detected_runtime_versions,
)

MANIFEST_RELATIVE_PATH = Path("manifests") / "collection_manifest.json"
SCHEDULE_RELATIVE_PATH = Path("manifests") / "collection_schedule.json"
SUMMARY_RELATIVE_PATH = Path("manifests") / "dataset_summary.json"
ATTEMPTS_RELATIVE_PATH = Path("manifests") / "attempts.jsonl"
EPISODES_RELATIVE_PATH = Path("manifests") / "episodes.jsonl"
SCENE_GROUPS_RELATIVE_PATH = Path("manifests") / "scene_groups.jsonl"
SHARDS_RELATIVE_PATH = Path("manifests") / "source_shards.jsonl"
RUN_MARKER = ".langmani-m3a-root"


class ManifestError(RuntimeError):
    """Raised when a collection root cannot be initialized or resumed safely."""


def installed_runtime_versions() -> dict[str, str | None]:
    """Return the exact package versions relevant to raw replay provenance."""
    return detected_runtime_versions()


def initialize_or_resume_manifest(config: CollectionConfig) -> CollectionManifest:
    """Create a new manifest or load the exact compatible incomplete run."""
    root = Path(config.raw_output_root).resolve()
    manifest_path = root / MANIFEST_RELATIVE_PATH
    marker_path = root / RUN_MARKER

    if root.exists() and config.overwrite_policy.value == "replace":
        _replace_existing_root(
            root,
            marker_path=marker_path,
            manifest_path=manifest_path,
            schema_version=config.collection_schema_version,
        )

    if manifest_path.exists():
        if config.resume_policy.value != "resume":
            raise ManifestError(
                f"collection manifest already exists and resume is disabled: {manifest_path}"
            )
        manifest = load_manifest(root)
        if manifest.config.identity_dict() != config.identity_dict():
            raise ManifestError("existing manifest configuration differs from requested collection")
        if Path(manifest.config.raw_output_root).resolve() != root:
            raise ManifestError("existing manifest raw_output_root does not resolve to this root")
        if dict(manifest.runtime_versions) != dict(config.runtime_versions):
            raise ManifestError("runtime package versions differ from the existing collection")
        return manifest

    if root.exists() and any(root.iterdir()):
        if config.resume_policy.value == "resume" and _is_safe_initialization_residue(
            root,
            marker_path=marker_path,
            schema_version=config.collection_schema_version,
        ):
            shutil.rmtree(root)
        else:
            raise ManifestError(f"non-empty collection root has no resumable M3A manifest: {root}")
    root.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(config.collection_schema_version + "\n", encoding="utf-8")
    schedule = build_collection_schedule(config)
    manifest = CollectionManifest(
        collection_schema_version=config.collection_schema_version,
        collection_run_id=schedule.collection_run_id,
        config=config,
        schedule=schedule,
        status=CollectionStatus.IN_PROGRESS,
        runtime_versions=config.runtime_versions,
        next_candidate_scene_index=0,
    )
    persist_manifest(root, manifest)
    return manifest


def load_manifest(root: str | Path) -> CollectionManifest:
    """Load and fully validate the authoritative collection manifest."""
    path = Path(root).resolve() / MANIFEST_RELATIVE_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read collection manifest {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ManifestError("collection manifest root must be a mapping")
    try:
        return CollectionManifest.from_dict(payload)
    except (TypeError, ValueError, KeyError) as error:
        raise ManifestError(f"invalid collection manifest {path}: {error}") from error


def persist_manifest(root: str | Path, manifest: CollectionManifest) -> None:
    """Commit deterministic audit projections, then the authoritative manifest.

    The manifest is written last so it is the commit marker for the projection
    generation.  A process interruption can leave newer projections beside an
    older manifest, but a resume can deterministically regenerate them from that
    older authority; it can never expose a newer authority with older projections.
    """
    destination_root = Path(root).resolve()
    _write_json_atomic(destination_root / SCHEDULE_RELATIVE_PATH, manifest.schedule.to_dict())
    summary = RawDatasetSummary.from_manifest(manifest)
    _write_json_atomic(destination_root / SUMMARY_RELATIVE_PATH, summary.to_dict())
    _write_jsonl_atomic(
        destination_root / ATTEMPTS_RELATIVE_PATH,
        [record.to_dict() for record in manifest.attempts],
    )
    _write_jsonl_atomic(
        destination_root / EPISODES_RELATIVE_PATH,
        [record.to_dict() for record in manifest.raw_episodes],
    )
    _write_jsonl_atomic(
        destination_root / SCENE_GROUPS_RELATIVE_PATH,
        [record.to_dict() for record in manifest.scene_groups],
    )
    _write_jsonl_atomic(
        destination_root / SHARDS_RELATIVE_PATH,
        [record.to_dict() for record in manifest.source_shards],
    )
    _write_json_atomic(destination_root / MANIFEST_RELATIVE_PATH, manifest.to_dict())


def validate_manifest_projections(
    root: str | Path,
    manifest: CollectionManifest | None = None,
) -> None:
    """Require every human-readable projection to exactly match the authority."""
    destination_root = Path(root).resolve()
    authoritative = manifest if manifest is not None else load_manifest(destination_root)
    _require_json_projection(
        destination_root / SCHEDULE_RELATIVE_PATH,
        authoritative.schedule.to_dict(),
    )
    _require_json_projection(
        destination_root / SUMMARY_RELATIVE_PATH,
        RawDatasetSummary.from_manifest(authoritative).to_dict(),
    )
    _require_jsonl_projection(
        destination_root / ATTEMPTS_RELATIVE_PATH,
        [record.to_dict() for record in authoritative.attempts],
    )
    _require_jsonl_projection(
        destination_root / EPISODES_RELATIVE_PATH,
        [record.to_dict() for record in authoritative.raw_episodes],
    )
    _require_jsonl_projection(
        destination_root / SCENE_GROUPS_RELATIVE_PATH,
        [record.to_dict() for record in authoritative.scene_groups],
    )
    _require_jsonl_projection(
        destination_root / SHARDS_RELATIVE_PATH,
        [record.to_dict() for record in authoritative.source_shards],
    )


def _replace_existing_root(
    root: Path,
    *,
    marker_path: Path,
    manifest_path: Path,
    schema_version: str,
) -> None:
    if not root.exists():
        return
    if root.parent == root:
        raise ManifestError("refusing to replace a filesystem root")
    safe_initialization = _is_safe_initialization_residue(
        root,
        marker_path=marker_path,
        schema_version=schema_version,
    )
    if manifest_path.is_file():
        try:
            marker_matches = marker_path.read_text(encoding="utf-8") == schema_version + "\n"
        except (OSError, UnicodeError):
            marker_matches = False
        if not marker_matches:
            raise ManifestError("overwrite=replace requires an exact LangMani M3A marker")
        existing = load_manifest(root)
        if (
            existing.collection_schema_version != schema_version
            or Path(existing.config.raw_output_root).resolve() != root.resolve()
        ):
            raise ManifestError("overwrite=replace manifest does not own the requested root")
    elif not safe_initialization:
        raise ManifestError(
            "overwrite=replace requires an existing LangMani M3A marker and manifest"
        )
    shutil.rmtree(root)


def _is_safe_initialization_residue(
    root: Path,
    *,
    marker_path: Path,
    schema_version: str,
) -> bool:
    """Recognize only pre-authority files emitted by an interrupted first commit."""
    try:
        if marker_path.read_text(encoding="utf-8") != schema_version + "\n":
            return False
    except (OSError, UnicodeError):
        return False
    allowed_root_names = {RUN_MARKER, "manifests"}
    if any(path.name not in allowed_root_names for path in root.iterdir()):
        return False
    manifests = root / "manifests"
    if not manifests.exists():
        return True
    if not manifests.is_dir():
        return False
    allowed_names = {
        MANIFEST_RELATIVE_PATH.name,
        SCHEDULE_RELATIVE_PATH.name,
        SUMMARY_RELATIVE_PATH.name,
        ATTEMPTS_RELATIVE_PATH.name,
        EPISODES_RELATIVE_PATH.name,
        SCENE_GROUPS_RELATIVE_PATH.name,
        SHARDS_RELATIVE_PATH.name,
    }
    for path in manifests.iterdir():
        if not path.is_file():
            return False
        if path.name not in allowed_names and not (
            path.name.startswith(".") and path.name.endswith(".tmp")
        ):
            return False
    return True


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                )
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _require_json_projection(path: Path, expected: Any) -> None:
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read manifest projection {path}: {error}") from error
    if actual != expected:
        raise ManifestError(f"manifest projection differs from authority: {path}")


def _require_jsonl_projection(path: Path, expected: list[dict[str, Any]]) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        actual = [json.loads(line) for line in lines]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read manifest projection {path}: {error}") from error
    if actual != expected:
        raise ManifestError(f"manifest projection differs from authority: {path}")


__all__ = [
    "ATTEMPTS_RELATIVE_PATH",
    "EPISODES_RELATIVE_PATH",
    "MANIFEST_RELATIVE_PATH",
    "ManifestError",
    "SCENE_GROUPS_RELATIVE_PATH",
    "SHARDS_RELATIVE_PATH",
    "SUMMARY_RELATIVE_PATH",
    "initialize_or_resume_manifest",
    "installed_runtime_versions",
    "load_manifest",
    "persist_manifest",
    "validate_manifest_projections",
]
