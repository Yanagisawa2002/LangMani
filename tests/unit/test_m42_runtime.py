"""Pure tests for the M4.2 execution-horizon and gripper runtime adapters."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from langmani.policies.act_action_bounds import (
    ActionBoundErrorKind,
    ActionBoundProcessingError,
)
from langmani.policies.m42_runtime import (
    BinaryGripperEnvPostprocessorV0,
    ExecutionHorizonPolicyV0,
    M42ActionRuntimeProcessorV0,
    M42ActionRuntimeSummary,
    M42RuntimeContractError,
)
from langmani.policies.m42_types import ExecutionHorizonConfig, GripperRuntimeMode


class _ChunkPolicy:
    def __init__(self) -> None:
        self.query_count = 0
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        assert "observation.state" in batch
        offset = self.query_count * 100
        self.query_count += 1
        chunk = torch.zeros((1, 50, 8), dtype=torch.float32)
        chunk[0, :, 0] = torch.arange(50, dtype=torch.float32) + offset
        return chunk


def _bounds(value: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.full(8, -value, dtype=np.float32),
        np.full(8, value, dtype=np.float32),
    )


def _batch() -> dict[str, torch.Tensor]:
    return {"observation.state": torch.zeros((1, 9), dtype=torch.float32)}


@pytest.mark.parametrize("horizon", (1, 5, 10))
def test_execution_horizon_consumes_exactly_h_actions_per_query(horizon: int) -> None:
    policy = _ChunkPolicy()
    adapter = ExecutionHorizonPolicyV0(policy, ExecutionHorizonConfig(horizon))
    adapter.begin_task("task-a")
    actions = [
        float(adapter.select_action(_batch(), task_id="task-a")[0, 0]) for _ in range(horizon + 1)
    ]
    assert actions[:horizon] == [float(index) for index in range(horizon)]
    assert actions[-1] == 100.0
    metrics = adapter.metrics()
    assert metrics.policy_query_count == 2
    assert metrics.executed_action_count == horizon + 1
    assert metrics.pending_action_count == horizon - 1
    assert metrics.maximum_queue_depth == horizon
    json.dumps(metrics.to_dict(), allow_nan=False)


def test_execution_horizon_reset_drops_queue_and_prevents_cross_task_residue() -> None:
    policy = _ChunkPolicy()
    adapter = ExecutionHorizonPolicyV0(policy, ExecutionHorizonConfig(5))
    adapter.begin_task("task-a")
    assert float(adapter.select_action(_batch(), task_id="task-a")[0, 0]) == 0.0
    assert adapter.metrics().pending_action_count == 4

    adapter.begin_task("task-b")
    metrics = adapter.metrics()
    assert metrics.discarded_on_last_reset == 4
    assert metrics.policy_query_count == 0
    assert metrics.executed_action_count == 0
    with pytest.raises(M42RuntimeContractError, match="different task"):
        adapter.select_action(_batch(), task_id="task-a")
    assert float(adapter.select_action(_batch(), task_id="task-b")[0, 0]) == 100.0
    assert policy.reset_count == 2


def test_execution_horizon_requires_explicit_task_binding_for_task_checked_calls() -> None:
    adapter = ExecutionHorizonPolicyV0(_ChunkPolicy(), ExecutionHorizonConfig(1))
    with pytest.raises(M42RuntimeContractError, match="begin_task"):
        adapter.select_action(_batch(), task_id="task-a")
    with pytest.raises(ValueError, match="non-empty"):
        adapter.begin_task("")


@pytest.mark.parametrize(
    ("chunk", "message"),
    (
        (torch.zeros((1, 49, 8), dtype=torch.float32), "50-action"),
        (torch.zeros((2, 50, 8), dtype=torch.float32), r"\[1, chunk, 8\]"),
        (torch.zeros((1, 50, 7), dtype=torch.float32), r"\[1, chunk, 8\]"),
        (torch.zeros((1, 50, 8), dtype=torch.int64), "floating"),
    ),
)
def test_execution_horizon_rejects_malformed_policy_chunks(
    chunk: torch.Tensor,
    message: str,
) -> None:
    class BadPolicy:
        @staticmethod
        def reset() -> None:
            return None

        @staticmethod
        def predict_action_chunk(batch: dict[str, torch.Tensor]) -> torch.Tensor:
            del batch
            return chunk

    adapter = ExecutionHorizonPolicyV0(BadPolicy(), ExecutionHorizonConfig(5))
    with pytest.raises(ActionBoundProcessingError, match=message) as captured:
        adapter.select_action(_batch())
    assert captured.value.kind is ActionBoundErrorKind.MALFORMED_ACTION


def test_execution_horizon_rejects_nonfinite_chunk() -> None:
    class NonfinitePolicy(_ChunkPolicy):
        def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
            chunk = super().predict_action_chunk(batch)
            chunk[0, 0, 0] = float("nan")
            return chunk

    adapter = ExecutionHorizonPolicyV0(NonfinitePolicy(), ExecutionHorizonConfig(1))
    with pytest.raises(ActionBoundProcessingError, match="NaN or infinity") as captured:
        adapter.select_action(_batch())
    assert captured.value.kind is ActionBoundErrorKind.NONFINITE_ACTION


@pytest.mark.parametrize(
    ("raw_gripper", "expected"),
    ((0.75, 1.0), (0.0, 1.0), (-0.001, -1.0), (-0.8, -1.0)),
)
def test_binary_gripper_sign_mapping(raw_gripper: float, expected: float) -> None:
    low, high = _bounds()
    processor = BinaryGripperEnvPostprocessorV0(low=low, high=high)
    action = torch.zeros((1, 8), dtype=torch.float32)
    action[0, 7] = raw_gripper
    result = processor.process(action, rollout_step=1)
    assert isinstance(result.executed_action, torch.Tensor)
    assert float(result.executed_action[0, 7]) == expected
    assert result.binary_gripper_audit is not None
    assert result.binary_gripper_audit.changed == (raw_gripper != expected)


def test_binary_gripper_leaves_all_arm_values_bitwise_unchanged() -> None:
    low, high = _bounds()
    processor = BinaryGripperEnvPostprocessorV0(low=low, high=high)
    action = np.asarray([0.25, -0.5, 0.75, -0.25, 0.0, 0.1, -0.1, 0.2], dtype=np.float32)
    result = processor.process(action, rollout_step=2)
    assert np.array_equal(result.executed_action[:7], action[:7])
    assert result.executed_action[7] == 1.0
    assert result.runtime_projection_audit.projected_component_count == 0


def test_binary_transform_hook_only_changes_gripper_and_resets_audit() -> None:
    low, high = _bounds()
    processor = BinaryGripperEnvPostprocessorV0(low=low, high=high)
    action = torch.tensor(
        [[0.25, -0.5, 0.75, -0.25, 0.0, 0.1, -0.1, 0.0]],
        dtype=torch.float32,
    )
    transformed = processor.transform(action, rollout_step=4)
    torch.testing.assert_close(transformed[0, :7], action[0, :7], rtol=0, atol=0)
    assert float(transformed[0, 7]) == 1.0
    assert float(action[0, 7]) == 0.0
    assert len(processor.audit_records) == 1
    assert processor.last_audit_record is not None
    assert processor.last_audit_record.rollout_step == 4
    processor.reset()
    assert processor.audit_records == ()
    assert processor.last_audit_record is None


@pytest.mark.parametrize(
    "action",
    (
        torch.zeros(8, dtype=torch.float32),
        torch.zeros((1, 7), dtype=torch.float32),
        torch.zeros((1, 8), dtype=torch.int64),
    ),
)
def test_binary_transform_hook_rejects_malformed_tensors(action: torch.Tensor) -> None:
    low, high = _bounds()
    processor = BinaryGripperEnvPostprocessorV0(low=low, high=high)
    with pytest.raises(M42RuntimeContractError):
        processor.transform(action, rollout_step=1)


def test_binary_transform_hook_rejects_nonfinite_tensor() -> None:
    low, high = _bounds()
    processor = BinaryGripperEnvPostprocessorV0(low=low, high=high)
    action = torch.zeros((1, 8), dtype=torch.float32)
    action[0, 7] = float("nan")
    with pytest.raises(M42RuntimeContractError, match="NaN or infinity"):
        processor.transform(action, rollout_step=1)


def test_binary_then_project_has_separate_raw_binary_and_runtime_audits() -> None:
    low, high = _bounds(0.8)
    processor = M42ActionRuntimeProcessorV0(
        GripperRuntimeMode.BINARY,
        low=low,
        high=high,
    )
    action = np.zeros(8, dtype=np.float32)
    action[0] = 1.2
    action[7] = 0.1
    result = processor.process(action, rollout_step=1)
    assert np.array_equal(
        result.executed_action,
        np.asarray([0.8] + [0.0] * 6 + [0.8], dtype=np.float32),
    )
    assert result.raw_bound_audit.projected_component_count == 1
    assert result.binary_gripper_audit is not None
    assert result.binary_gripper_audit.changed_command_count == 1
    assert result.runtime_projection_audit.projected_component_count == 2

    summary = M42ActionRuntimeSummary.from_results((result,))
    assert summary.raw_out_of_bounds_component_count == 1
    assert summary.binary_changed_command_count == 1
    assert summary.arm_projected_component_count == 1
    assert summary.gripper_projected_component_count == 1
    assert not summary.raw_action_bounds_validated
    assert summary.runtime_action_bounds_validated
    json.dumps(summary.to_dict(), allow_nan=False)


def test_project_runtime_preserves_existing_processor_behavior() -> None:
    low, high = _bounds()
    processor = M42ActionRuntimeProcessorV0(
        GripperRuntimeMode.PROJECT,
        low=low,
        high=high,
    )
    action = torch.zeros((1, 8), dtype=torch.float32)
    action[0, 7] = 1.2
    result = processor.process(action, rollout_step=1)
    assert float(result.executed_action[0, 7]) == 1.0
    assert result.binary_gripper_audit is None
    assert result.raw_bound_audit is result.runtime_projection_audit
    summary = M42ActionRuntimeSummary.from_results((result,))
    assert summary.binary_changed_command_count == 0
    assert summary.gripper_projected_component_count == 1


@pytest.mark.parametrize("bad_value", (float("nan"), float("inf"), float("-inf")))
def test_binary_gripper_hard_fails_nonfinite_input(bad_value: float) -> None:
    low, high = _bounds()
    processor = BinaryGripperEnvPostprocessorV0(low=low, high=high)
    action = np.zeros(8, dtype=np.float32)
    action[7] = bad_value
    with pytest.raises(ActionBoundProcessingError) as captured:
        processor.process(action, rollout_step=1)
    assert captured.value.kind is ActionBoundErrorKind.NONFINITE_ACTION


@pytest.mark.parametrize(
    "action",
    (
        np.zeros(7, dtype=np.float32),
        np.zeros((1, 1, 8), dtype=np.float32),
        np.zeros(8, dtype=np.int64),
    ),
)
def test_binary_gripper_hard_fails_malformed_action(action: np.ndarray) -> None:
    low, high = _bounds()
    processor = BinaryGripperEnvPostprocessorV0(low=low, high=high)
    with pytest.raises(ActionBoundProcessingError) as captured:
        processor.process(action, rollout_step=1)
    assert captured.value.kind is ActionBoundErrorKind.MALFORMED_ACTION


def test_binary_configuration_roundtrip_and_fingerprints_are_stable() -> None:
    low, high = _bounds()
    binary = BinaryGripperEnvPostprocessorV0(low=low, high=high)
    reloaded = BinaryGripperEnvPostprocessorV0.from_configuration(
        binary.configuration(),
        low=low,
        high=high,
    )
    assert reloaded.configuration() == binary.configuration()
    assert reloaded.runtime_fingerprint == binary.runtime_fingerprint
    assert json.loads(json.dumps(binary.configuration(), allow_nan=False)) == binary.configuration()

    project_runtime = M42ActionRuntimeProcessorV0(
        GripperRuntimeMode.PROJECT,
        low=low,
        high=high,
    )
    binary_runtime = M42ActionRuntimeProcessorV0(
        GripperRuntimeMode.BINARY,
        low=low,
        high=high,
    )
    assert project_runtime.runtime_fingerprint != binary_runtime.runtime_fingerprint
    assert ExecutionHorizonConfig(1).fingerprint != ExecutionHorizonConfig(5).fingerprint


def test_runtime_summary_rejects_mixed_runtime_identities() -> None:
    low, high = _bounds()
    action = np.zeros(8, dtype=np.float32)
    project = M42ActionRuntimeProcessorV0(
        GripperRuntimeMode.PROJECT,
        low=low,
        high=high,
    ).process(action, rollout_step=1)
    binary = M42ActionRuntimeProcessorV0(
        GripperRuntimeMode.BINARY,
        low=low,
        high=high,
    ).process(action, rollout_step=1)
    with pytest.raises(ValueError, match="cannot mix"):
        M42ActionRuntimeSummary.from_results((project, binary))
