"""Independently verify the Phase 2C-A dataset identity resolution."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = (
    PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2c_a" / "identity_resolution.json"
)
DEFAULT_PRIMARY = Path("/root/autodl-tmp/langmani-external/phase2b6-v2/production_v2/primary")
DEFAULT_ARCHIVE = Path(
    "/root/autodl-tmp/langmani-external/phase2b6-v2-archive/"
    "LangManiOfficialMultiSkill-v2-"
    "b0e255d8b3f7f77fe33538cb2c313620d1d00adf90562b33dad06b10674474b3.tar.gz"
)
DEFAULT_RESTORE = Path(
    "/root/autodl-tmp/langmani-external/phase2b6-v2-restore-scratch/"
    "b0e255d8b3f7f77fe33538cb2c313620d1d00adf90562b33dad06b10674474b3/"
    "LangManiOfficialMultiSkill-v2"
)
SIDECAR = "metadata/accepted_multiskill_dataset_package.json"
CANONICAL_PACKAGE_FINGERPRINT = (
    "sha256:77675e2134e4886a97e4bdac2230c64c3da30c080e647433b7c701a79544ed04"
)
INVALID_TRANSCRIPTION = "sha256:77675e2134e4886a97e4bdac2230c64c3da30cc080e647433b7c701a79544ed04"
CORRECTED_DOCUMENTS = (
    "docs/DECISIONS.md",
    "docs/langmani_v2/phase_2b6_v2_dataset_card.md",
    "docs/langmani_v2/phase_2b6_v2_result.md",
)
ERRATUM = "docs/langmani_v2/phase_2b6_v2_identity_erratum.md"


class IdentityResolutionError(RuntimeError):
    """Raised when the identity-resolution evidence cannot be verified."""


def canonical_json_sha256(payload: Mapping[str, object]) -> str:
    """Return the repository's canonical JSON fingerprint."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash one ordinary file without following a symlink."""

    if path.is_symlink() or not path.is_file():
        raise IdentityResolutionError(f"expected ordinary file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise IdentityResolutionError(f"failed to read JSON object: {path}") from error
    if not isinstance(value, dict):
        raise IdentityResolutionError(f"expected JSON object: {path}")
    return value


def legacy_tree_manifest(root: Path) -> dict[str, object]:
    """Reproduce the frozen Phase 2B.2 tree-identity algorithm exactly."""

    source = root.resolve()
    if not source.is_dir():
        raise IdentityResolutionError(f"tree root is unavailable: {source}")
    entries: list[dict[str, object]] = []
    total_bytes = 0
    for path in sorted(source.rglob("*"), key=lambda candidate: candidate.as_posix()):
        if path.is_symlink():
            raise IdentityResolutionError(f"dataset tree contains symlink: {path}")
        if not path.is_file():
            continue
        size_bytes = path.stat().st_size
        relative = path.relative_to(source).as_posix()
        # The accepted tree identity was produced by Phase 2B.2's historical
        # double-scheme entry serialization. Preserve it byte-for-byte here.
        entries.append(
            {
                "path": relative,
                "sha256": "sha256:" + sha256_file(path),
                "size_bytes": size_bytes,
            }
        )
        total_bytes += size_bytes
    if not entries:
        raise IdentityResolutionError(f"tree root is empty: {source}")
    return {
        "entries": entries,
        "file_count": len(entries),
        "schema_version": "langmani-v2-phase2b2-file-tree-v0",
        "total_bytes": total_bytes,
        "tree_digest": _entries_digest(entries),
    }


def _entries_digest(entries: list[dict[str, object]]) -> str:
    """Match ``langmani.v2.push_dataset.sha256_json(entries)``."""

    encoded = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _entry_map(manifest: Mapping[str, object]) -> dict[str, tuple[int, str]]:
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise IdentityResolutionError("tree manifest entries are unavailable")
    result: dict[str, tuple[int, str]] = {}
    for raw in entries:
        if not isinstance(raw, dict):
            raise IdentityResolutionError("tree manifest entry is invalid")
        path = raw.get("path")
        size_bytes = raw.get("size_bytes")
        digest = raw.get("sha256")
        if (
            not isinstance(path, str)
            or not isinstance(size_bytes, int)
            or not isinstance(digest, str)
        ):
            raise IdentityResolutionError("tree manifest entry fields are invalid")
        result[path] = (size_bytes, digest)
    return result


def _git(command: list[str], *, repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", *command],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise IdentityResolutionError(f"git {' '.join(command)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def verify_identity_resolution(
    *,
    repo_root: Path,
    audit_path: Path,
    primary_root: Path,
    archive_path: Path,
    restore_root: Path,
) -> dict[str, object]:
    """Verify machine identity, payload bytes, sidecar, archive, and restore."""

    audit = read_json(audit_path)
    audit_body = dict(audit)
    audit_fingerprint = audit_body.pop("fingerprint", None)
    package_path = (
        repo_root
        / "artifacts"
        / "langmani_v2"
        / "phase_2b6_v2"
        / "accepted_multiskill_dataset_package.json"
    )
    result_path = (
        repo_root / "artifacts" / "langmani_v2" / "phase_2b6_v2" / "result_classification.json"
    )
    archive_manifest_path = (
        repo_root / "artifacts" / "langmani_v2" / "phase_2b6_v2" / "archive_manifest.json"
    )
    restore_manifest_path = (
        repo_root / "artifacts" / "langmani_v2" / "phase_2b6_v2" / "restore_validation_result.json"
    )
    package = read_json(package_path)
    result = read_json(result_path)
    archive_manifest = read_json(archive_manifest_path)
    restore_manifest = read_json(restore_manifest_path)
    package_body = dict(package)
    package_fingerprint = package_body.pop("fingerprint", None)
    recomputed_package_fingerprint = canonical_json_sha256(package_body)

    primary = legacy_tree_manifest(primary_root)
    restore = legacy_tree_manifest(restore_root)
    primary_entries = _entry_map(primary)
    restore_entries = _entry_map(restore)
    primary_only = sorted(set(primary_entries) - set(restore_entries))
    restore_only = sorted(set(restore_entries) - set(primary_entries))
    common_paths = sorted(set(primary_entries) & set(restore_entries))
    common_mismatches = [
        path for path in common_paths if primary_entries[path] != restore_entries[path]
    ]

    sidecar_path = primary_root / SIDECAR
    sidecar = read_json(sidecar_path)
    sidecar_body = dict(sidecar)
    sidecar_fingerprint = sidecar_body.pop("fingerprint", None)
    sidecar_recomputed = canonical_json_sha256(sidecar_body)
    sidecar_sha256 = sha256_file(sidecar_path)
    source_package_sha256 = sha256_file(package_path)

    canonical_audit = audit.get("canonical_package")
    frozen_audit = audit.get("frozen_payload_tree")
    live_audit = audit.get("live_primary_tree")
    sidecar_audit = audit.get("sidecar")
    archive_audit = audit.get("archive")
    restore_audit = audit.get("restore")
    if not all(
        isinstance(value, dict)
        for value in (
            canonical_audit,
            frozen_audit,
            live_audit,
            sidecar_audit,
            archive_audit,
            restore_audit,
        )
    ):
        raise IdentityResolutionError("identity-resolution audit sections are invalid")

    corrected_docs = {
        relative: (repo_root / relative).read_text(encoding="utf-8")
        for relative in CORRECTED_DOCUMENTS
    }
    erratum_text = (repo_root / ERRATUM).read_text(encoding="utf-8")
    checks = {
        "audit_fingerprint_recomputed": audit_fingerprint == canonical_json_sha256(audit_body),
        "canonical_package_hex_length": len(CANONICAL_PACKAGE_FINGERPRINT.removeprefix("sha256:"))
        == 64,
        "invalid_transcription_hex_length": len(INVALID_TRANSCRIPTION.removeprefix("sha256:"))
        == 65,
        "machine_package_fingerprint_matches": package_fingerprint == CANONICAL_PACKAGE_FINGERPRINT,
        "machine_package_fingerprint_recomputed": recomputed_package_fingerprint
        == CANONICAL_PACKAGE_FINGERPRINT,
        "result_fingerprint_matches": result.get("accepted_package_fingerprint")
        == CANONICAL_PACKAGE_FINGERPRINT,
        "audit_canonical_fingerprint_matches": canonical_audit.get("fingerprint")
        == CANONICAL_PACKAGE_FINGERPRINT,
        "corrected_documents_use_canonical": all(
            CANONICAL_PACKAGE_FINGERPRINT in text for text in corrected_docs.values()
        ),
        "corrected_documents_exclude_invalid": all(
            INVALID_TRANSCRIPTION not in text for text in corrected_docs.values()
        ),
        "erratum_records_both_values": (
            CANONICAL_PACKAGE_FINGERPRINT in erratum_text and INVALID_TRANSCRIPTION in erratum_text
        ),
        "actual_archive_path_matches": archive_path.resolve().as_posix()
        == archive_audit.get("path"),
        "actual_archive_size_matches": archive_path.stat().st_size
        == archive_audit.get("size_bytes"),
        "actual_archive_sha256_matches": sha256_file(archive_path) == archive_audit.get("sha256"),
        "archive_manifest_tree_matches": archive_manifest.get("primary_tree_digest")
        == frozen_audit.get("tree_digest"),
        "archive_manifest_file_count_matches": archive_manifest.get("primary_file_count")
        == frozen_audit.get("file_count"),
        "archive_manifest_total_bytes_matches": archive_manifest.get("primary_total_bytes")
        == frozen_audit.get("total_bytes"),
        "restore_manifest_fingerprint_matches": restore_manifest.get("fingerprint")
        == restore_audit.get("fingerprint"),
        "restore_manifest_tree_matches": restore_manifest.get("restored_tree_digest")
        == frozen_audit.get("tree_digest"),
        "restore_actual_file_count_matches": restore.get("file_count")
        == frozen_audit.get("file_count"),
        "restore_actual_total_bytes_matches": restore.get("total_bytes")
        == frozen_audit.get("total_bytes"),
        "restore_actual_tree_matches": restore.get("tree_digest")
        == frozen_audit.get("tree_digest"),
        "primary_actual_file_count_matches": primary.get("file_count")
        == live_audit.get("file_count"),
        "primary_actual_total_bytes_matches": primary.get("total_bytes")
        == live_audit.get("total_bytes"),
        "primary_actual_tree_matches": primary.get("tree_digest") == live_audit.get("tree_digest"),
        "only_primary_difference_is_sidecar": primary_only == [SIDECAR],
        "restore_has_no_additional_paths": not restore_only,
        "all_common_payload_bytes_match": not common_mismatches
        and len(common_paths) == frozen_audit.get("file_count"),
        "sidecar_size_matches": sidecar_path.stat().st_size == sidecar_audit.get("size_bytes"),
        "sidecar_sha256_matches": sidecar_sha256 == sidecar_audit.get("sha256"),
        "sidecar_source_controlled_copy_matches": sidecar_sha256 == source_package_sha256,
        "sidecar_fingerprint_matches": sidecar_fingerprint == CANONICAL_PACKAGE_FINGERPRINT,
        "sidecar_fingerprint_recomputed": sidecar_recomputed == CANONICAL_PACKAGE_FINGERPRINT,
        "sidecar_audit_reference_matches": sidecar_audit.get("referenced_package_fingerprint")
        == CANONICAL_PACKAGE_FINGERPRINT,
    }
    report: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-a-identity-verification-v0",
        "source_commit": "19de56781e029c20859839a1e3493920449ff9bf",
        "branch": _git(["branch", "--show-current"], repo_root=repo_root),
        "head": _git(["rev-parse", "HEAD"], repo_root=repo_root),
        "canonical_package_fingerprint": CANONICAL_PACKAGE_FINGERPRINT,
        "observed": {
            "archive": {
                "path": archive_path.resolve().as_posix(),
                "sha256": sha256_file(archive_path),
                "size_bytes": archive_path.stat().st_size,
            },
            "frozen_payload_tree": {
                key: restore[key] for key in ("file_count", "total_bytes", "tree_digest")
            },
            "live_primary_tree": {
                key: primary[key] for key in ("file_count", "total_bytes", "tree_digest")
            },
            "primary_only_paths": primary_only,
            "restore_only_paths": restore_only,
            "common_file_count": len(common_paths),
            "common_mismatch_paths": common_mismatches,
            "sidecar": {
                "path": SIDECAR,
                "sha256": sidecar_sha256,
                "size_bytes": sidecar_path.stat().st_size,
                "fingerprint": sidecar_fingerprint,
            },
        },
        "protected_payload_classes_unchanged": {
            "accepted_episode_and_frame_inventory": not common_mismatches,
            "action": not common_mismatches,
            "image_and_media": not common_mismatches,
            "normalization": not common_mismatches,
            "source_lineage": not common_mismatches,
            "split": not common_mismatches,
            "state": not common_mismatches,
            "statistics": not common_mismatches,
        },
        "checks": checks,
        "optimizer_created": False,
        "training_started": False,
        "passed": all(checks.values()),
    }
    report["fingerprint"] = canonical_json_sha256(report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--primary-root", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--restore-root", type=Path, default=DEFAULT_RESTORE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = verify_identity_resolution(
        repo_root=args.repo_root.resolve(),
        audit_path=args.audit.resolve(),
        primary_root=args.primary_root.resolve(),
        archive_path=args.archive.resolve(),
        restore_root=args.restore_root.resolve(),
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
