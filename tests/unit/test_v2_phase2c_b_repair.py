from __future__ import annotations

import json
from pathlib import Path

from environment.verify_v2_phase2c_b import main as verify_phase2c_b
from langmani.v2.phase2c_a import PACKAGE_FINGERPRINT
from langmani.v2.phase2c_b import canonical_fingerprint
from scripts.select_v2_phase2c_b_repair import main as select_repair


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


def _base_evidence(tmp_path: Path, checkpoint_sha256: str) -> dict[str, Path]:
    return {
        "static": _write(
            tmp_path / "static.json",
            {
                "schema_version": ("langmani-v2-phase2c-b-static-preparation-complete-v0"),
                "passed": True,
            },
        ),
        "base": _write(
            tmp_path / "base.json",
            {
                "schema_version": "langmani-v2-phase2c-b-base-audit-complete-v0",
                "passed": True,
            },
        ),
        "smoke": _write(
            tmp_path / "smoke.json",
            {
                "schema_version": "langmani-v2-phase2c-b-training-result-v0",
                "stage": "smoke",
                "optimizer_steps": 1,
                "passed": True,
            },
        ),
        "smoke_evaluation": _write(
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
        ),
        "pick_micro": _write(
            tmp_path / "pick-micro.json",
            {
                "stage": "micro",
                "model_kind": "pick_smolvla",
                "optimizer_steps": 500,
                "passed": True,
            },
        ),
        "shared_micro": _write(
            tmp_path / "shared-micro.json",
            {
                "stage": "micro",
                "model_kind": "shared_language_smolvla",
                "optimizer_steps": 500,
                "passed": True,
            },
        ),
        "training": _write(
            tmp_path / "training.json",
            {
                "stage": "full",
                "model_kind": "pick_smolvla",
                "optimizer_steps": 20_000,
                "checkpoint_records": [{"sha256": checkpoint_sha256}],
                "passed": True,
            },
        ),
    }


def _repair_evidence(
    tmp_path: Path,
    checkpoint_sha256: str,
) -> tuple[Path, Path, Path]:
    screen_semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-a-evaluation-summary-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "model_kind": "pick_smolvla",
        "task_id": "PickCube-v1",
        "split": "validation",
        "instruction_condition": "correct",
        "episode_count": 6,
        "success_count": 0,
        "invalid_action_count": 0,
        "simulator_error_count": 0,
        "checkpoint_identities": [checkpoint_sha256],
        "execution_horizon": 8,
        "failure_categories": {"failed_grasp": 5, "no_initial_motion": 1},
        "post_contact_divergence_mean_m": 0.007,
        "completed": True,
    }
    screen = _write(tmp_path / "screen.json", screen_semantic)
    screen_document = json.loads(screen.read_text(encoding="utf-8"))
    selected = _write(
        tmp_path / "selected.json",
        {
            "schema_version": "langmani-v2-phase2c-b-selected-policy-v0",
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "model_kind": "pick_smolvla",
            "task_id": "PickCube-v1",
            "selection_split": "validation",
            "selection_subset_episode_count": 6,
            "selected_checkpoint_sha256": checkpoint_sha256,
            "selected_execution_horizon": 1,
            "candidates": [
                {
                    "checkpoint_sha256": checkpoint_sha256,
                    "execution_horizon": 8,
                    "summary_fingerprint": screen_document["fingerprint"],
                }
            ],
            "final_test_opened": False,
            "passed": True,
        },
    )
    initial = _write(
        tmp_path / "initial.json",
        {
            "schema_version": "langmani-v2-phase2c-a-evaluation-summary-v0",
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "model_kind": "pick_smolvla",
            "task_id": "PickCube-v1",
            "split": "validation",
            "instruction_condition": "correct",
            "episode_count": 30,
            "success_count": 0,
            "invalid_action_count": 0,
            "simulator_error_count": 0,
            "checkpoint_identities": [checkpoint_sha256],
            "execution_horizon": 1,
            "failure_categories": {"failed_grasp": 4, "no_initial_motion": 26},
            "post_contact_divergence_mean_m": 0.001,
            "completed": True,
        },
    )
    return selected, initial, screen


def test_horizon_repair_binds_demonstrated_motion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_sha256 = "sha256:" + "4" * 64
    selected, initial, screen = _repair_evidence(tmp_path, checkpoint_sha256)
    repair = tmp_path / "repair.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "repair",
            "--selected-policy",
            str(selected),
            "--initial-validation-summary",
            str(initial),
            "--repair-screen-summary",
            str(screen),
            "--output",
            str(repair),
        ],
    )
    assert select_repair() == 0
    result = json.loads(repair.read_text(encoding="utf-8"))
    assert result["repair_ordinal"] == 1
    assert result["repair_kind"] == "action_chunk_execution_horizon"
    assert result["original_execution_horizon"] == 1
    assert result["repaired_execution_horizon"] == 8
    assert result["model_bytes_changed"] is False
    assert result["training_started"] is False


def test_verifier_closes_failed_single_repair_as_result_d(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_sha256 = "sha256:" + "4" * 64
    base = _base_evidence(tmp_path, checkpoint_sha256)
    selected, initial, screen = _repair_evidence(tmp_path, checkpoint_sha256)
    repair = tmp_path / "repair.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "repair",
            "--selected-policy",
            str(selected),
            "--initial-validation-summary",
            str(initial),
            "--repair-screen-summary",
            str(screen),
            "--output",
            str(repair),
        ],
    )
    assert select_repair() == 0
    repaired_validation = _write(
        tmp_path / "repaired-validation.json",
        {
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "model_kind": "pick_smolvla",
            "task_id": "PickCube-v1",
            "split": "validation",
            "instruction_condition": "correct",
            "episode_count": 30,
            "success_count": 0,
            "invalid_action_count": 0,
            "simulator_error_count": 0,
            "checkpoint_identities": [checkpoint_sha256],
            "execution_horizon": 8,
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
            str(base["static"]),
            "--base-audit",
            str(base["base"]),
            "--smoke-training-result",
            str(base["smoke"]),
            "--smoke-evaluation-summary",
            str(base["smoke_evaluation"]),
            "--pick-micro-result",
            str(base["pick_micro"]),
            "--shared-micro-result",
            str(base["shared_micro"]),
            "--pick-training-result",
            str(base["training"]),
            "--selected-policy",
            str(selected),
            "--initial-pick-validation-summary",
            str(initial),
            "--repair-policy",
            str(repair),
            "--pick-validation-summary",
            str(repaired_validation),
            "--repair-used",
            "--output",
            str(output),
        ],
    )
    assert verify_phase2c_b() == 2
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["passed"] is False
    assert result["repair_used"] is True
    assert result["original_selected_execution_horizon"] == 1
    assert result["selected_execution_horizon"] == 8
    assert result["pick_competence_gate"]["bounded_repair_permitted"] is False
    assert result["pick_competence_gate"]["result_d_stop"] is True
    assert result["other_full_models_authorized"] is False
