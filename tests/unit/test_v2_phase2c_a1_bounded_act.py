"""Phase 2C-A.1 bounded-action and serialization contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("lerobot")

from lerobot.configs import FeatureType, NormalizationMode  # noqa: E402
from lerobot.policies import make_pre_post_processors  # noqa: E402

from langmani.datasets.lerobot_types import (  # noqa: E402
    ACTION_FEATURE_KEY,
    IMAGE_FEATURE_KEY,
    STATE_FEATURE_KEY,
)
from langmani.policies.act_training import (  # noqa: E402
    build_fixture_act_config,
    build_policy_and_processors,
)
from langmani.v2.phase2b5 import (  # noqa: E402
    PANDA_ACTION_HIGH,
    PANDA_ACTION_LOW,
)
from langmani.v2.phase2c_a1_bounded_act import (  # noqa: E402
    BOUNDED_ACTION_HEAD_CONFIG_FILE,
    BoundedActionHeadV1,
    BoundedACTPolicyV1,
    build_bounded_act_config,
    independent_bounded_action_audit,
    validate_bounded_act_policy,
)
from langmani.v2.phase2c_a_act import masked_l1_loss  # noqa: E402


def _stats() -> dict[str, dict[str, torch.Tensor]]:
    return {
        IMAGE_FEATURE_KEY: {
            "mean": torch.zeros(3, 1, 1),
            "std": torch.ones(3, 1, 1),
            "min": torch.zeros(3, 1, 1),
            "max": torch.ones(3, 1, 1),
        },
        STATE_FEATURE_KEY: {
            "mean": torch.zeros(9),
            "std": torch.ones(9),
            "min": -torch.ones(9),
            "max": torch.ones(9),
        },
        ACTION_FEATURE_KEY: {
            "mean": torch.zeros(8),
            "std": torch.ones(8),
            "min": torch.from_numpy(PANDA_ACTION_LOW.copy()),
            "max": torch.from_numpy(PANDA_ACTION_HIGH.copy()),
        },
    }


def test_exact_bounded_mapping_handles_finite_extremes_without_correction() -> None:
    head = BoundedActionHeadV1(8)
    with torch.no_grad():
        head.linear.weight.copy_(torch.eye(8))
        head.linear.bias.zero_()
    raw = torch.tensor(
        [
            [-1.0e6, 1.0e6, -100.0, 100.0, -10.0, 10.0, -1.0, 1.0],
            [0.0, -0.0, 0.25, -0.25, 4.0, -4.0, 20.0, -20.0],
        ],
        dtype=torch.float32,
    )
    action = head(raw)
    lower = torch.from_numpy(PANDA_ACTION_LOW)
    upper = torch.from_numpy(PANDA_ACTION_HIGH)
    expected = lower + 0.5 * (torch.tanh(raw) + 1.0) * (upper - lower)
    torch.testing.assert_close(action, expected, rtol=0, atol=0)
    assert bool(torch.isfinite(action).all())
    assert bool(torch.all(action >= lower))
    assert bool(torch.all(action <= upper))


def test_independent_100k_chunk_audit_has_no_bound_events() -> None:
    result = independent_bounded_action_audit()
    assert result["sampled_chunks"] == 100_000
    assert result["sampled_actions"] == 1_600_000
    assert result["nonfinite_actions"] == 0
    assert result["lower_bound_violations"] == 0
    assert result["upper_bound_violations"] == 0
    assert result["clipping_events"] == 0
    assert result["projection_events"] == 0
    assert result["passed"] is True


def test_padding_mask_zeroes_padded_and_all_padding_batches() -> None:
    expected = torch.zeros(2, 16, 8)
    predicted = torch.ones_like(expected)
    partial = torch.zeros(2, 16, dtype=torch.bool)
    partial[:, -5:] = True
    baseline = masked_l1_loss(predicted, expected, partial)
    predicted[:, -5:] = 1.0e9
    assert masked_l1_loss(predicted, expected, partial) == baseline
    all_padding = torch.ones(2, 16, dtype=torch.bool)
    all_padding_loss = masked_l1_loss(predicted, expected, all_padding)
    assert torch.isfinite(all_padding_loss)
    assert float(all_padding_loss) == 0.0


def test_action_processor_is_identity_and_gripper_targets_remain_finite() -> None:
    config = build_bounded_act_config(build_fixture_act_config())
    normalized = {
        FeatureType(key) if isinstance(key, str) else key: NormalizationMode(value)
        for key, value in config.normalization_mapping.items()
    }
    assert normalized[FeatureType.ACTION] is NormalizationMode.IDENTITY
    _, preprocessor, postprocessor = build_policy_and_processors(
        config,
        _stats(),
        policy_factory=BoundedACTPolicyV1,
    )
    targets = torch.zeros(2, config.chunk_size, 8)
    targets[0, :, -1] = 1.0 - 1.0e-6
    targets[1, :, -1] = -1.0 + 1.0e-6
    batch = {
        IMAGE_FEATURE_KEY: torch.zeros(2, 3, 256, 256),
        STATE_FEATURE_KEY: torch.zeros(2, 9),
        ACTION_FEATURE_KEY: targets,
        "action_is_pad": torch.zeros(2, config.chunk_size, dtype=torch.bool),
    }
    processed = preprocessor(batch)
    torch.testing.assert_close(processed[ACTION_FEATURE_KEY], targets, rtol=0, atol=0)
    restored = postprocessor(processed[ACTION_FEATURE_KEY])
    torch.testing.assert_close(restored, targets, rtol=0, atol=0)
    assert bool(torch.isfinite(restored).all())


@pytest.mark.integration
def test_policy_and_processors_reload_with_bounded_semantics(tmp_path: Path) -> None:
    torch.manual_seed(47)
    config = build_bounded_act_config(build_fixture_act_config())
    policy, preprocessor, postprocessor = build_policy_and_processors(
        config,
        _stats(),
        policy_factory=BoundedACTPolicyV1,
    )
    policy.eval()
    observation = {
        IMAGE_FEATURE_KEY: torch.zeros(1, 3, 256, 256),
        STATE_FEATURE_KEY: torch.zeros(1, 9),
    }
    processed = preprocessor(observation)
    expected = postprocessor(policy.predict_action_chunk(processed))

    checkpoint = tmp_path / "checkpoint"
    policy.save_pretrained(checkpoint, push_to_hub=False)
    preprocessor.save_pretrained(checkpoint, push_to_hub=False)
    postprocessor.save_pretrained(checkpoint, push_to_hub=False)
    assert (checkpoint / BOUNDED_ACTION_HEAD_CONFIG_FILE).is_file()
    loaded = BoundedACTPolicyV1.from_pretrained(
        checkpoint,
        local_files_only=True,
        strict=True,
    )
    loaded_pre, loaded_post = make_pre_post_processors(
        loaded.config,
        pretrained_path=str(checkpoint),
    )
    actual = loaded_post(loaded.predict_action_chunk(loaded_pre(observation)))
    validate_bounded_act_policy(loaded)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    lower = torch.from_numpy(PANDA_ACTION_LOW)
    upper = torch.from_numpy(PANDA_ACTION_HIGH)
    assert bool(torch.all(actual >= lower))
    assert bool(torch.all(actual <= upper))
