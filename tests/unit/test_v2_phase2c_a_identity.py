"""Unit tests for the Phase 2C-A identity-resolution verifier."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERIFIER = PROJECT_ROOT / "environment" / "verify_v2_phase2c_a_identity.py"


def _load_verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_v2_phase2c_a_identity", VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_package_fingerprint_recomputes() -> None:
    verifier = _load_verifier()
    package_path = (
        PROJECT_ROOT
        / "artifacts"
        / "langmani_v2"
        / "phase_2b6_v2"
        / "accepted_multiskill_dataset_package.json"
    )
    package = json.loads(package_path.read_text(encoding="utf-8"))
    fingerprint = package.pop("fingerprint")
    assert fingerprint == verifier.CANONICAL_PACKAGE_FINGERPRINT
    assert verifier.canonical_json_sha256(package) == fingerprint
    assert len(fingerprint.removeprefix("sha256:")) == 64
    assert len(verifier.INVALID_TRANSCRIPTION.removeprefix("sha256:")) == 65


def test_legacy_tree_manifest_detects_only_authorized_sidecar(tmp_path: Path) -> None:
    verifier = _load_verifier()
    restored = tmp_path / "restore"
    primary = tmp_path / "primary"
    for root in (restored, primary):
        (root / "data").mkdir(parents=True)
        (root / "data" / "episode.bin").write_bytes(b"immutable-payload")
    sidecar = primary / verifier.SIDECAR
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text('{"fingerprint":"sha256:test"}\n', encoding="utf-8")

    restored_manifest = verifier.legacy_tree_manifest(restored)
    primary_manifest = verifier.legacy_tree_manifest(primary)
    restored_entries = verifier._entry_map(restored_manifest)
    primary_entries = verifier._entry_map(primary_manifest)

    assert sorted(set(primary_entries) - set(restored_entries)) == [verifier.SIDECAR]
    assert not (set(restored_entries) - set(primary_entries))
    common = set(primary_entries) & set(restored_entries)
    assert all(primary_entries[path] == restored_entries[path] for path in common)


def test_tree_manifest_fails_on_symlink_when_supported(tmp_path: Path) -> None:
    verifier = _load_verifier()
    root = tmp_path / "tree"
    root.mkdir()
    target = root / "payload.bin"
    target.write_bytes(b"payload")
    link = root / "link.bin"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(verifier.IdentityResolutionError, match="symlink"):
        verifier.legacy_tree_manifest(root)
