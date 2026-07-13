"""CanonicalTaskOneHotV0 contract tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import stable_task_id
from langmani.policies.act_conditioning import (
    CANONICAL_TASK_IDS,
    append_canonical_task_onehot,
    canonical_task_index,
    canonical_task_onehot,
    canonical_task_onehot_from_spec,
    task_onehot_contract,
    validate_canonical_task_onehot,
)


def test_canonical_task_order_is_exact_object_major_bin_minor() -> None:
    assert tuple(stable_task_id(spec) for spec in CANONICAL_TASK_SPECS) == CANONICAL_TASK_IDS
    assert [spec.target_object_id for spec in CANONICAL_TASK_SPECS] == [
        "red_cube",
        "red_cube",
        "green_cube",
        "green_cube",
        "blue_cube",
        "blue_cube",
    ]
    assert [spec.target_bin_id for spec in CANONICAL_TASK_SPECS] == [
        "left_bin",
        "right_bin",
        "left_bin",
        "right_bin",
        "left_bin",
        "right_bin",
    ]


@pytest.mark.parametrize("index", range(6))
def test_onehot_has_one_active_component_and_round_trips_task_spec(index: int) -> None:
    task_id = CANONICAL_TASK_IDS[index]
    from_id = canonical_task_onehot(task_id)
    from_spec = canonical_task_onehot_from_spec(CANONICAL_TASK_SPECS[index])

    assert canonical_task_index(task_id) == index
    assert from_id.dtype == np.float32
    assert from_id.shape == (6,)
    assert np.array_equal(from_id, np.eye(6, dtype=np.float32)[index])
    assert np.array_equal(from_id, from_spec)
    validate_canonical_task_onehot(from_id)


def test_training_and_inference_append_use_the_same_mapping() -> None:
    states = np.arange(18, dtype=np.float32).reshape(2, 9)
    task_ids = (CANONICAL_TASK_IDS[1], CANONICAL_TASK_IDS[4])
    conditioned = append_canonical_task_onehot(states, task_ids)

    assert conditioned.dtype == np.float32
    assert conditioned.shape == (2, 15)
    assert np.array_equal(conditioned[:, :9], states)
    assert np.array_equal(
        conditioned[0, 9:], canonical_task_onehot_from_spec(CANONICAL_TASK_SPECS[1])
    )
    assert np.array_equal(
        conditioned[1, 9:], canonical_task_onehot_from_spec(CANONICAL_TASK_SPECS[4])
    )

    torch_state = torch.from_numpy(states)
    torch_result = append_canonical_task_onehot(torch_state, task_ids)
    assert isinstance(torch_result, torch.Tensor)
    assert torch_result.dtype == torch.float32
    assert torch.equal(torch_result, torch.from_numpy(conditioned))


def test_single_state_preserves_panda_prefix_and_contract_serializes() -> None:
    state = np.linspace(0.0, 1.0, 9, dtype=np.float32)
    result = append_canonical_task_onehot(state, CANONICAL_TASK_IDS[0])

    assert result.shape == (15,)
    assert np.array_equal(result[:9], state)
    validate_canonical_task_onehot(result[9:])
    assert task_onehot_contract() == {
        "mapping_version": "CanonicalTaskOneHotV0",
        "ordered_task_ids": list(CANONICAL_TASK_IDS),
        "dtype": "float32",
        "shape": [6],
    }


def test_unknown_task_and_malformed_state_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown stable task ID"):
        canonical_task_onehot("red_cube__left_bin")
    with pytest.raises(ValueError, match="dtype float32"):
        append_canonical_task_onehot(np.zeros(9, dtype=np.float64), CANONICAL_TASK_IDS[0])
    with pytest.raises(ValueError, match=r"shape \(9,\)"):
        append_canonical_task_onehot(np.zeros(8, dtype=np.float32), CANONICAL_TASK_IDS[0])
    with pytest.raises(ValueError, match="exactly one active"):
        validate_canonical_task_onehot(np.ones(6, dtype=np.float32))
