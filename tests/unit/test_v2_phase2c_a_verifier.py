"""Unit tests for the Phase 2C-A Result D verifier."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERIFIER = PROJECT_ROOT / "environment" / "verify_v2_phase2c_a.py"


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("verify_v2_phase2c_a", VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_result(module: Any) -> dict[str, object]:
    return {
        "schema_version": "langmani-v2-phase2c-a-result-v0",
        "package_fingerprint": module.CANONICAL_PACKAGE,
        "result": "RESULT_D",
        "result_reason": (
            "selected_pick_act_emitted_native_pd_joint_pos_out_of_bounds_gripper_actions"
        ),
        "hard_stop_stage": "closed_loop_infrastructure_smoke",
        "training_complete": True,
        "offline_diagnostics_complete": True,
        "closed_loop_development_started": False,
        "final_evaluation_started": False,
        "shared_seed1_started": False,
        "act_baselines_validated": False,
        "smolvla_phase_eligible": False,
        "smolvla_training_authorized": False,
        "vla_jepa_training_authorized": False,
        "unavailable_metrics": [
            "validation_success",
            "unseen_reset_success",
            "visual_shift_success",
        ],
    }


def test_result_d_contract_keeps_unrun_metrics_unavailable() -> None:
    module = _module()
    module.validate_result_classification(_valid_result(module))


def test_result_d_contract_rejects_training_authorization() -> None:
    module = _module()
    value = _valid_result(module)
    value["smolvla_training_authorized"] = True
    with pytest.raises(module.Phase2CAVerificationError, match="classification"):
        module.validate_result_classification(value)


def test_result_d_contract_rejects_zero_for_unrun_metrics() -> None:
    module = _module()
    value = _valid_result(module)
    value["unavailable_metrics"] = ["unseen_reset_success", "visual_shift_success"]
    value["validation_success"] = 0.0
    with pytest.raises(module.Phase2CAVerificationError, match="classification"):
        module.validate_result_classification(value)
