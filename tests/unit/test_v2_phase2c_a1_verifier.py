"""Phase 2C-A.1 independent-verifier checkpoint contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from environment.verify_v2_phase2c_a1 import _verify_checkpoint


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_verifier_reads_checkpoint_fingerprint_from_manifest_record(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "step-00000001"
    artifact = checkpoint / "pretrained_model" / "model.safetensors"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"bounded-act")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    fingerprint = "sha256:" + "1" * 64
    _write_json(
        checkpoint / "checkpoint_manifest.json",
        {
            "record": {"checkpoint_fingerprint": fingerprint},
            "artifacts": [
                {
                    "path": "pretrained_model/model.safetensors",
                    "size_bytes": artifact.stat().st_size,
                    "sha256": digest,
                }
            ],
        },
    )
    _write_json(
        checkpoint / "complete.json",
        {"checkpoint_fingerprint": fingerprint},
    )
    result = _verify_checkpoint(tmp_path, "checkpoints/step-00000001")
    assert result["checkpoint_fingerprint"] == fingerprint
    assert result["artifact_count"] == 1
    assert result["all_artifacts_rehashed"] is True
