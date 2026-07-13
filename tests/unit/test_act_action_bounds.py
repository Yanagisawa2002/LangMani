"""Pure M4.1 action-bound processor, audit, and runtime-identity tests."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from langmani.policies.act_action_bounds import (
    ActionBoundConfig,
    ActionBoundErrorKind,
    ActionBoundMode,
    ActionBoundProcessingError,
    ActionProjectionRecord,
    ActionProjectionSummary,
    BoundedActionEnvPostprocessorV0,
    EvaluationRuntimeIdentity,
    EvaluationRuntimeManifest,
    resolve_environment_action_bounds,
)

SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
COMMIT = "1" * 40


def _processor(
    mode: ActionBoundMode = ActionBoundMode.PROJECT,
    *,
    low: np.ndarray | None = None,
    high: np.ndarray | None = None,
) -> BoundedActionEnvPostprocessorV0:
    return BoundedActionEnvPostprocessorV0(
        ActionBoundConfig(mode=mode),
        low=np.full(8, -1.0, dtype=np.float32) if low is None else low,
        high=np.full(8, 1.0, dtype=np.float32) if high is None else high,
    )


def _runtime(mode: ActionBoundMode) -> EvaluationRuntimeIdentity:
    processor = _processor(mode)
    return EvaluationRuntimeIdentity(
        checkpoint_fingerprint=SHA_A,
        policy_preprocessor_fingerprint=SHA_B,
        policy_postprocessor_fingerprint=SHA_C,
        action_bound_config=processor.config,
        environment_id="LangMani-PickPlaceByInstruction-v0",
        action_space_contract=processor.action_space_contract(),
        task_conditioning_mapping_version="CanonicalTaskOneHotV0",
        rollout_config={"maximum_episode_steps": 200, "sim_backend": "physx_cpu"},
        code_git_commit=COMMIT,
    )


def test_action_bound_mode_is_exact_and_configuration_roundtrips() -> None:
    assert tuple(mode.value for mode in ActionBoundMode) == ("reject", "project")
    config = ActionBoundConfig(mode=ActionBoundMode.PROJECT)
    assert ActionBoundConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError):
        ActionBoundMode("clip")


def test_in_range_numpy_action_is_bitwise_unchanged_and_uses_environment_bounds() -> None:
    low = np.full(8, -2.0, dtype=np.float32)
    high = np.full(8, 2.0, dtype=np.float32)
    action = np.linspace(-1.5, 1.5, 8, dtype=np.float32)
    result = _processor(low=low, high=high).process(action, rollout_step=1)
    assert result.executed_action is action
    assert np.array_equal(result.executed_action, action)
    assert not result.audit_record.was_projected
    assert result.audit_record.projected_component_count == 0


@pytest.mark.parametrize("value", (1.0508, 1.1280))
def test_upper_bound_violation_is_projected_exactly(value: float) -> None:
    action = torch.zeros((1, 8), dtype=torch.float32)
    action[0, -1] = value
    result = _processor().process(action, rollout_step=3)
    assert isinstance(result.executed_action, torch.Tensor)
    assert result.executed_action[0, -1] == 1.0
    assert result.audit_record.was_projected
    assert result.audit_record.projected_component_count == 1
    assert result.audit_record.maximum_upper_excess == pytest.approx(value - 1.0)


def test_lower_bound_violation_and_correction_norms_are_exact() -> None:
    action = np.zeros(8, dtype=np.float64)
    action[0] = -1.5
    action[1] = 1.25
    result = _processor().process(action, rollout_step=2)
    assert np.array_equal(result.executed_action[:2], np.asarray([-1.0, 1.0]))
    record = result.audit_record
    assert np.array_equal(record.violation_mask[:2], np.asarray([True, True]))
    assert record.projected_component_count == 2
    assert record.l1_correction_magnitude == pytest.approx(0.75)
    assert record.l2_correction_magnitude == pytest.approx(np.hypot(0.5, 0.25))
    assert record.linf_correction_magnitude == pytest.approx(0.5)
    assert record.maximum_lower_excess == pytest.approx(0.5)
    assert record.maximum_upper_excess == pytest.approx(0.25)


@pytest.mark.parametrize("value", (1.01, -1.01))
def test_reject_mode_blocks_both_bound_directions(value: float) -> None:
    action = torch.zeros((1, 8), dtype=torch.float32)
    action[0, 2] = value
    with pytest.raises(ActionBoundProcessingError) as captured:
        _processor(ActionBoundMode.REJECT).process(action, rollout_step=1)
    assert captured.value.kind is ActionBoundErrorKind.BOUND_VIOLATION
    assert captured.value.record is not None
    assert captured.value.record.was_rejected
    assert captured.value.record.executed_action is None


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
@pytest.mark.parametrize("mode", tuple(ActionBoundMode))
def test_nonfinite_actions_are_hard_failures_in_all_modes(
    value: float,
    mode: ActionBoundMode,
) -> None:
    action = np.zeros(8, dtype=np.float32)
    action[3] = value
    with pytest.raises(ActionBoundProcessingError) as captured:
        _processor(mode).process(action, rollout_step=1)
    assert captured.value.kind is ActionBoundErrorKind.NONFINITE_ACTION
    assert captured.value.record is not None
    assert captured.value.record.was_rejected
    serialized = captured.value.record.to_dict()
    json.dumps(serialized, allow_nan=False)
    assert ActionProjectionRecord.from_dict(serialized) == captured.value.record


@pytest.mark.parametrize(
    "action",
    (
        np.zeros(7, dtype=np.float32),
        np.zeros((1, 1, 8), dtype=np.float32),
        np.zeros(8, dtype=np.int64),
    ),
)
def test_malformed_shape_or_dtype_is_rejected(action: np.ndarray) -> None:
    with pytest.raises(ActionBoundProcessingError) as captured:
        _processor().process(action, rollout_step=1)
    assert captured.value.kind is ActionBoundErrorKind.MALFORMED_ACTION
    assert captured.value.record is not None
    assert captured.value.record.was_rejected


def test_invalid_bounds_are_rejected() -> None:
    with pytest.raises(ActionBoundProcessingError, match="same-shape"):
        BoundedActionEnvPostprocessorV0(
            ActionBoundConfig(),
            low=np.zeros(7, dtype=np.float32),
            high=np.ones(8, dtype=np.float32),
        )
    with pytest.raises(ActionBoundProcessingError, match="must not exceed"):
        BoundedActionEnvPostprocessorV0(
            ActionBoundConfig(),
            low=np.ones(8, dtype=np.float32),
            high=np.zeros(8, dtype=np.float32),
        )
    with pytest.raises(ActionBoundProcessingError, match="finite"):
        BoundedActionEnvPostprocessorV0(
            ActionBoundConfig(),
            low=np.full(8, -np.inf, dtype=np.float32),
            high=np.ones(8, dtype=np.float32),
        )


def test_dtype_device_and_batch_structure_are_preserved() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action = torch.zeros((2, 8), dtype=torch.float64, device=device)
    action[0, 0] = 1.25
    action[1, 1] = -1.5
    result = _processor().process(action, rollout_step=1)
    assert isinstance(result.executed_action, torch.Tensor)
    assert result.executed_action.dtype == action.dtype
    assert result.executed_action.device == action.device
    assert result.executed_action.shape == action.shape
    assert result.audit_record.projected_component_count == 2


def test_raw_executed_bounds_and_mask_survive_json_roundtrip() -> None:
    action = np.zeros(8, dtype=np.float32)
    action[-1] = 1.0508
    record = _processor().process(action, rollout_step=4).audit_record
    payload = record.to_dict()
    assert payload["raw_action"][-1] == pytest.approx(1.0508)
    assert payload["executed_action"][-1] == 1.0
    assert payload["lower_bounds"] == [-1.0] * 8
    assert payload["upper_bounds"] == [1.0] * 8
    assert payload["violation_mask"] == [False] * 7 + [True]
    assert ActionProjectionRecord.from_dict(payload) == record


def test_projection_summary_counts_actions_components_and_dimensions() -> None:
    processor = _processor()
    first = np.zeros(8, dtype=np.float32)
    first[0] = 1.5
    second = np.zeros(8, dtype=np.float32)
    second[0] = -1.25
    second[7] = 1.1
    records = (
        processor.process(first, rollout_step=1).audit_record,
        processor.process(second, rollout_step=2).audit_record,
        processor.process(np.zeros(8, dtype=np.float32), rollout_step=3).audit_record,
    )
    summary = ActionProjectionSummary.from_records(records, action_dimension=8)
    assert summary.total_policy_actions == 3
    assert summary.projected_action_count == 2
    assert summary.projected_action_rate == pytest.approx(2 / 3)
    assert summary.projected_component_count == 3
    assert summary.per_action_dimension_projection_counts == (2, 0, 0, 0, 0, 0, 0, 1)
    assert summary.first_projected_rollout_step == 1
    assert ActionProjectionSummary.from_dict(summary.to_dict()) == summary


def test_environment_wrapper_bounds_are_resolved_without_hard_coding() -> None:
    base = SimpleNamespace(
        single_action_space=SimpleNamespace(
            low=np.arange(8, dtype=np.float32) - 10,
            high=np.arange(8, dtype=np.float32) + 10,
        )
    )

    class Wrapper:
        unwrapped = base

        @staticmethod
        def get_wrapper_attr(name: str) -> object:
            return getattr(base, name)

    low, high = resolve_environment_action_bounds(Wrapper(), expected_action_components=8)
    assert np.array_equal(low, np.arange(8, dtype=np.float32) - 10)
    assert np.array_equal(high, np.arange(8, dtype=np.float32) + 10)


def test_reloaded_processor_produces_identical_projection() -> None:
    config = ActionBoundConfig(mode=ActionBoundMode.PROJECT)
    original = _processor(config.mode)
    reloaded = BoundedActionEnvPostprocessorV0(
        ActionBoundConfig.from_dict(config.to_dict()),
        low=np.full(8, -1.0, dtype=np.float32),
        high=np.full(8, 1.0, dtype=np.float32),
    )
    action = torch.tensor([[0.0] * 7 + [1.0508]], dtype=torch.float32)
    first = original.process(action, rollout_step=1)
    second = reloaded.process(action, rollout_step=1)
    torch.testing.assert_close(first.executed_action, second.executed_action, rtol=0, atol=0)
    assert first.audit_record == second.audit_record


def test_runtime_fingerprint_changes_with_mode_and_ignores_output_path() -> None:
    reject = _runtime(ActionBoundMode.REJECT)
    project = _runtime(ActionBoundMode.PROJECT)
    assert reject.runtime_fingerprint != project.runtime_fingerprint
    first = EvaluationRuntimeManifest(
        identity=project,
        checkpoint_model_reload_validated=True,
        policy_processor_reload_validated=True,
        action_bound_processor_reload_validated=True,
        deterministic_raw_action_matched=True,
        raw_action_match_tolerance=1e-6,
        evaluation_output_path="C:/machine/a/output",
    )
    second = replace(first, evaluation_output_path="/mnt/machine-b/output")
    assert first.identity.runtime_fingerprint == second.identity.runtime_fingerprint
    assert EvaluationRuntimeManifest.from_dict(first.to_dict()) == first


def test_checkpoint_and_processor_fingerprints_are_runtime_inputs() -> None:
    base = _runtime(ActionBoundMode.PROJECT)
    changed = replace(
        base,
        policy_postprocessor_fingerprint="sha256:" + "d" * 64,
        runtime_fingerprint="",
    )
    assert changed.runtime_fingerprint != base.runtime_fingerprint
