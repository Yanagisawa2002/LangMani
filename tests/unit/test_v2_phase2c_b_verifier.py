from __future__ import annotations

import json
from pathlib import Path

from environment.verify_v2_phase2c_b import main as verify_phase2c_b
from langmani.v2.phase2c_a import PACKAGE_FINGERPRINT
from langmani.v2.phase2c_b import canonical_fingerprint


def _write(
    path: Path,
    semantic: dict[str, object],
    *,
    fingerprint: bool = True,
) -> Path:
    value = (
        {**semantic, "fingerprint": canonical_fingerprint(semantic)} if fingerprint else semantic
    )
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def test_pick_gate_verifier_closes_selected_checkpoint_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_sha256 = "sha256:" + "4" * 64
    static = _write(
        tmp_path / "static.json",
        {
            "schema_version": "langmani-v2-phase2c-b-static-preparation-complete-v0",
            "passed": True,
        },
    )
    base = _write(
        tmp_path / "base.json",
        {
            "schema_version": "langmani-v2-phase2c-b-base-audit-complete-v0",
            "passed": True,
        },
    )
    smoke = _write(
        tmp_path / "smoke.json",
        {
            "schema_version": "langmani-v2-phase2c-b-training-result-v0",
            "stage": "smoke",
            "optimizer_steps": 1,
            "passed": True,
        },
    )
    smoke_evaluation = _write(
        tmp_path / "smoke-evaluation.json",
        {
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "episode_count": 1,
            "invalid_action_count": 0,
            "simulator_error_count": 0,
            "mean_episode_length": 1.0,
            "mean_policy_query_count": 1.0,
        },
        fingerprint=False,
    )
    pick_micro = _write(
        tmp_path / "pick-micro.json",
        {
            "stage": "micro",
            "model_kind": "pick_smolvla",
            "optimizer_steps": 500,
            "passed": True,
        },
    )
    shared_micro = _write(
        tmp_path / "shared-micro.json",
        {
            "stage": "micro",
            "model_kind": "shared_language_smolvla",
            "optimizer_steps": 500,
            "passed": True,
        },
    )
    training = _write(
        tmp_path / "training.json",
        {
            "stage": "full",
            "model_kind": "pick_smolvla",
            "optimizer_steps": 20_000,
            "checkpoint_records": [{"sha256": checkpoint_sha256}],
            "passed": True,
        },
    )
    selected = _write(
        tmp_path / "selected.json",
        {
            "schema_version": "langmani-v2-phase2c-b-selected-policy-v0",
            "model_kind": "pick_smolvla",
            "task_id": "PickCube-v1",
            "selection_split": "validation",
            "selected_checkpoint_sha256": checkpoint_sha256,
            "selected_execution_horizon": 4,
            "final_test_opened": False,
            "passed": True,
        },
    )
    validation = _write(
        tmp_path / "validation.json",
        {
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "model_kind": "pick_smolvla",
            "task_id": "PickCube-v1",
            "split": "validation",
            "instruction_condition": "correct",
            "episode_count": 30,
            "success_count": 3,
            "invalid_action_count": 0,
            "simulator_error_count": 0,
            "checkpoint_identities": [checkpoint_sha256],
            "execution_horizon": 4,
            "completed": True,
        },
    )
    output = tmp_path / "verification.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "verify",
            "--verify-stage",
            "pick_gate",
            "--static-preparation",
            str(static),
            "--base-audit",
            str(base),
            "--smoke-training-result",
            str(smoke),
            "--smoke-evaluation-summary",
            str(smoke_evaluation),
            "--pick-micro-result",
            str(pick_micro),
            "--shared-micro-result",
            str(shared_micro),
            "--pick-training-result",
            str(training),
            "--selected-policy",
            str(selected),
            "--pick-validation-summary",
            str(validation),
            "--output",
            str(output),
        ],
    )
    assert verify_phase2c_b() == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["passed"] is True
    assert result["pick_competence_gate"]["success_count"] == 3
    assert result["other_full_models_authorized"] is True
    assert result["vla_jepa_training_authorized"] is False
