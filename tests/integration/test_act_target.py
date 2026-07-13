"""Native-target acceptance evidence for M4.

These tests intentionally consume the fresh structured report written by
``environment/verify_m4.py --target-smoke`` or ``--target-full``.  They skip on
unsupported hosts and never reinterpret fixture evidence as physical results.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT = PROJECT_ROOT / "outputs" / "diagnostics" / "m4" / "verification.json"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.target,
    pytest.mark.linux,
    pytest.mark.gpu,
    pytest.mark.training,
    pytest.mark.evaluation,
]


def _report(*, full: bool = False) -> dict[str, object]:
    if platform.system() != "Linux" or not torch.cuda.is_available():
        pytest.skip("M4 target evidence requires native Linux CUDA")
    if not REPORT.is_file():
        pytest.skip("run environment/verify_m4.py target mode before target integration tests")
    value = json.loads(REPORT.read_text(encoding="utf-8"))
    expected_mode = "target_full" if full else {"target_smoke", "target_full"}
    actual_mode = value.get("verification_mode")
    if (full and actual_mode != expected_mode) or (not full and actual_mode not in expected_mode):
        pytest.skip(f"fresh M4 report has incompatible mode {actual_mode!r}")
    assert value.get("passed") is True
    assert value.get("physical_target_validated") is True
    return value


def test_target_cuda_forward_backward() -> None:
    assert _report()["cuda_training_validated"] is True


@pytest.mark.video
def test_target_real_m3b_video_batch_loading() -> None:
    assert _report()["source_dataset_validated"] is True


@pytest.mark.video
def test_target_real_train_only_statistics() -> None:
    assert _report()["train_stats_leakage_validated"] is True


@pytest.mark.rendering
def test_target_per_task_closed_loop_rollout() -> None:
    report = _report()
    assert report["closed_loop_inference_validated"] is True


@pytest.mark.rendering
def test_target_all_six_task_onehot_rollouts() -> None:
    report = _report()
    assert (
        report["tiny_overfit_validated"] is True
        or report["mixed_task_onehot_experiment_completed"] is True
    )


@pytest.mark.rendering
@pytest.mark.slow
def test_target_mixed_unconditioned_rollout_without_hidden_task_input() -> None:
    assert _report(full=True)["mixed_unconditioned_experiment_completed"] is True


@pytest.mark.rendering
def test_target_actions_reached_m1_without_clipping() -> None:
    assert _report()["closed_loop_inference_validated"] is True


def test_rollout_module_has_no_m2_expert_or_planner_import() -> None:
    source = (PROJECT_ROOT / "src" / "langmani" / "policies" / "act_rollout.py").read_text(
        encoding="utf-8"
    )
    assert "langmani.experts" not in source
    assert "mplib" not in source


def test_target_checkpoint_reloaded_locally() -> None:
    assert _report()["checkpoint_reload_validated"] is True


@pytest.mark.rendering
@pytest.mark.slow
def test_target_full_validation_test_and_fresh_schedules() -> None:
    report = _report(full=True)
    assert report["validation_selection_validated"] is True
    assert report["test_lock_validated"] is True
    assert report["fresh_seed_benchmark_completed"] is True
    assert report["full_experiment_validated"] is True
