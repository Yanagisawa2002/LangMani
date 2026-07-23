from __future__ import annotations

import json
import shutil
from pathlib import Path

from langmani.v2.phase2b4_evidence import (
    REQUIRED_ARTIFACTS,
    verify_phase2b4_f0_artifacts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_result_c_artifacts_verify_and_detect_tampering(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2b4_f0"
    copied = tmp_path / "phase_2b4_f0"
    shutil.copytree(source, copied)
    report = verify_phase2b4_f0_artifacts(copied)
    assert report["passed"] is True
    assert report["result"] == "RESULT_C"
    assert report["ppo_full_training_eligible"] is False
    assert report["ppo_full_training_authorized"] is False

    manifest = json.loads((copied / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert {entry["path"] for entry in manifest["files"]} == REQUIRED_ARTIFACTS
    authorization = copied / "authorization_state.json"
    lf_bytes = authorization.read_bytes().replace(b"\r\n", b"\n")
    authorization.write_bytes(lf_bytes.replace(b"\n", b"\r\n"))
    assert verify_phase2b4_f0_artifacts(copied)["passed"] is True

    evaluation = copied / "micro_evaluation_result.json"
    evaluation.write_bytes(evaluation.read_bytes() + b" ")
    tampered = verify_phase2b4_f0_artifacts(copied)
    assert tampered["passed"] is False
    assert tampered["artifact_hash_checks"]["micro_evaluation_result.json"] is False
