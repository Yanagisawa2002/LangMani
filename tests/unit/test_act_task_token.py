"""CanonicalTaskTokenV0 public LeRobot ACT integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from lerobot.configs import FeatureType, NormalizationMode
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy

from langmani.datasets.lerobot_types import ACTION_FEATURE_KEY, IMAGE_FEATURE_KEY, STATE_FEATURE_KEY
from langmani.policies.act_conditioning import CANONICAL_TASK_IDS
from langmani.policies.act_task_token import (
    TASK_TOKEN_FEATURE_KEY,
    TaskTokenContractError,
    build_task_token_act_config,
    canonical_task_token,
    inject_task_token_observation,
    inject_task_token_training_batch,
    learned_task_embeddings,
    task_token_architecture_fingerprint,
    task_token_contract,
    validate_task_token_policy,
)
from langmani.policies.act_training import build_policy_and_processors, prepare_raw_batch
from langmani.policies.act_types import ActModelConfig
from langmani.policies.m42_types import TaskTokenConfig, TaskTokenMapping


def _model() -> ActModelConfig:
    return ActModelConfig(
        chunk_size=4,
        n_action_steps=2,
        dim_model=32,
        n_heads=2,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        latent_dim=8,
        n_vae_encoder_layers=1,
        dropout=0.0,
    )


def _token_config() -> TaskTokenConfig:
    return TaskTokenConfig(hidden_dimension=32)


def _statistics() -> dict[str, dict[str, torch.Tensor]]:
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
            "min": -torch.ones(8),
            "max": torch.ones(8),
        },
    }


def _raw_batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(42)
    return {
        IMAGE_FEATURE_KEY: torch.randint(
            0,
            256,
            (1, 3, 256, 256),
            dtype=torch.uint8,
            generator=generator,
        ),
        STATE_FEATURE_KEY: torch.linspace(-0.2, 0.2, 9).reshape(1, 9),
        ACTION_FEATURE_KEY: torch.zeros(1, 4, 8),
        "action_is_pad": torch.tensor([[False, False, False, True]]),
        "episode_index": torch.tensor([12], dtype=torch.int64),
    }


def test_task_token_mapping_and_architecture_fingerprint_are_canonical() -> None:
    config = _token_config()
    payload = task_token_contract(config)

    assert config.mapping.ordered_task_ids == CANONICAL_TASK_IDS
    assert payload["normalization_mode"] == "IDENTITY"
    assert payload["upstream_feature_type"] == "ENV"
    fingerprint = task_token_architecture_fingerprint(config)
    assert fingerprint.startswith("sha256:")
    assert fingerprint == task_token_architecture_fingerprint(_token_config())

    with pytest.raises(ValueError, match="ordering"):
        TaskTokenMapping(ordered_task_ids=tuple(reversed(CANONICAL_TASK_IDS)))
    with pytest.raises(ValueError, match="shape"):
        TaskTokenMapping(shape=(5,))


def test_training_and_inference_injection_share_mapping_and_preserve_9d_state() -> None:
    state = torch.arange(9, dtype=torch.float32)
    observation = inject_task_token_observation(
        {STATE_FEATURE_KEY: state},
        task=CANONICAL_TASK_IDS[4],
    )
    assert observation[STATE_FEATURE_KEY] is state
    assert state.shape == (9,)
    torch.testing.assert_close(
        observation[TASK_TOKEN_FEATURE_KEY],
        canonical_task_token(CANONICAL_TASK_IDS[4]),
        rtol=0,
        atol=0,
    )

    raw = _raw_batch()
    augmented = inject_task_token_training_batch(
        raw,
        task_id_by_episode={12: CANONICAL_TASK_IDS[4]},
    )
    assert augmented[STATE_FEATURE_KEY] is raw[STATE_FEATURE_KEY]
    assert raw[STATE_FEATURE_KEY].shape == (1, 9)
    torch.testing.assert_close(
        augmented[TASK_TOKEN_FEATURE_KEY][0],
        observation[TASK_TOKEN_FEATURE_KEY],
        rtol=0,
        atol=0,
    )


def test_task_token_injection_rejects_missing_or_precomputed_provenance() -> None:
    raw = _raw_batch()
    with pytest.raises(TaskTokenContractError, match="no stable M3B task mapping"):
        inject_task_token_training_batch(raw, task_id_by_episode={})
    with pytest.raises(TaskTokenContractError, match="precomputed"):
        inject_task_token_training_batch(
            {**raw, TASK_TOKEN_FEATURE_KEY: torch.zeros(1, 6)},
            task_id_by_episode={12: CANONICAL_TASK_IDS[0]},
        )
    with pytest.raises(ValueError, match="unknown stable task ID"):
        canonical_task_token("unknown-task")
    with pytest.raises(TaskTokenContractError, match="already contains"):
        inject_task_token_observation(
            {
                STATE_FEATURE_KEY: torch.zeros(9),
                TASK_TOKEN_FEATURE_KEY: torch.zeros(6),
            },
            task=CANONICAL_TASK_IDS[0],
        )


def test_public_act_env_feature_is_dedicated_hidden_dimension_token() -> None:
    config = build_task_token_act_config(
        _model(),
        device="cpu",
        use_amp=False,
        token_config=_token_config(),
    )
    policy = ACTPolicy(config)
    validate_task_token_policy(policy, token_config=_token_config())

    assert config.input_features[STATE_FEATURE_KEY].shape == (9,)
    assert config.input_features[STATE_FEATURE_KEY].type is FeatureType.STATE
    assert config.input_features[TASK_TOKEN_FEATURE_KEY].shape == (6,)
    assert config.input_features[TASK_TOKEN_FEATURE_KEY].type is FeatureType.ENV
    assert NormalizationMode(config.normalization_mapping["ENV"]) is NormalizationMode.IDENTITY
    assert policy.model.encoder_env_state_input_proj.in_features == 6
    assert policy.model.encoder_env_state_input_proj.out_features == 32
    assert learned_task_embeddings(policy).shape == (6, 32)
    assert not torch.equal(learned_task_embeddings(policy)[0], learned_task_embeddings(policy)[1])


@pytest.mark.fixture
@pytest.mark.integration
@pytest.mark.training
def test_task_token_forward_backward_and_fresh_pretrained_reload(tmp_path: Path) -> None:
    torch.manual_seed(17)
    config = build_task_token_act_config(
        _model(),
        device="cpu",
        use_amp=False,
        token_config=_token_config(),
    )
    policy, preprocessor, postprocessor = build_policy_and_processors(config, _statistics())
    raw = _raw_batch()
    augmented = inject_task_token_training_batch(
        raw,
        task_id_by_episode={12: CANONICAL_TASK_IDS[0]},
    )
    projected = prepare_raw_batch(augmented)
    assert projected[STATE_FEATURE_KEY].shape == (1, 9)
    assert projected[TASK_TOKEN_FEATURE_KEY].shape == (1, 6)
    processed = preprocessor(projected)
    loss, loss_dict = policy.forward(processed)
    assert torch.isfinite(loss)
    assert {"l1_loss", "kld_loss"} <= loss_dict.keys()
    loss.backward()
    projection_gradient = policy.model.encoder_env_state_input_proj.weight.grad
    assert projection_gradient is not None and torch.isfinite(projection_gradient).all()

    checkpoint = tmp_path / "task-token-checkpoint"
    policy.save_pretrained(checkpoint, push_to_hub=False)
    preprocessor.save_pretrained(checkpoint, push_to_hub=False)
    postprocessor.save_pretrained(checkpoint, push_to_hub=False)
    reloaded = ACTPolicy.from_pretrained(checkpoint, local_files_only=True, strict=True)
    loaded_preprocessor, loaded_postprocessor = make_pre_post_processors(
        reloaded.config,
        pretrained_path=str(checkpoint),
    )
    validate_task_token_policy(reloaded, token_config=_token_config())

    policy.eval()
    reloaded.eval()
    inference_raw = {
        IMAGE_FEATURE_KEY: projected[IMAGE_FEATURE_KEY][0],
        STATE_FEATURE_KEY: projected[STATE_FEATURE_KEY][0],
    }
    action_by_task: list[torch.Tensor] = []
    for task_id in CANONICAL_TASK_IDS[:2]:
        observation = inject_task_token_observation(inference_raw, task=task_id)
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        reloaded.reset()
        loaded_preprocessor.reset()
        loaded_postprocessor.reset()
        expected = postprocessor(policy.select_action(preprocessor(observation)))
        actual = loaded_postprocessor(reloaded.select_action(loaded_preprocessor(observation)))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        action_by_task.append(actual)
    assert not torch.equal(action_by_task[0], action_by_task[1])


def test_legacy_raw_batch_projection_remains_task_token_free() -> None:
    raw = _raw_batch()
    raw.pop("episode_index")
    projected = prepare_raw_batch(raw)
    assert set(projected) == {
        IMAGE_FEATURE_KEY,
        STATE_FEATURE_KEY,
        ACTION_FEATURE_KEY,
        "action_is_pad",
    }
    assert projected[STATE_FEATURE_KEY].shape == (1, 9)
