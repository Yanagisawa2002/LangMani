"""Public LeRobot ACT task-token integration for M4.2.

LeRobot 0.6.0 exposes one supported environment-state feature on ACT.  ACT
projects that feature with ``nn.Linear(env_dim, dim_model)`` and inserts the
result as its own Transformer encoder token after the Panda state token and
before image tokens.  M4.2 uses that public path with a canonical six-way
one-hot command.  The columns of the public linear projection (plus its shared
bias) are therefore six learned hidden-dimensional task embeddings without
changing ``PandaPolicyStateV0`` from nine components.

This module never patches or subclasses installed LeRobot code.  Its strict
checks intentionally fail if the upstream public ENV feature stops satisfying
the recorded architecture contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import cast

import torch
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act import ACTConfig, ACTPolicy

from langmani.datasets.identity import sha256_hex
from langmani.datasets.lerobot_types import STATE_FEATURE_KEY
from langmani.environments.specs import TaskSpec, stable_task_id
from langmani.policies.act_conditioning import (
    CANONICAL_TASK_COUNT,
    CANONICAL_TASK_IDS,
    canonical_task_onehot,
)
from langmani.policies.act_types import ActModelConfig, ActOptimizationConfig
from langmani.policies.m42_types import TaskTokenConfig, TaskTokenMapping

TASK_TOKEN_FEATURE_KEY = "observation.environment_state"
TASK_TOKEN_NORMALIZATION_MODE = NormalizationMode.IDENTITY
TASK_TOKEN_UPSTREAM_INTERFACE = "lerobot.configs.FeatureType.ENV"
TASK_TOKEN_INJECTION_LOCATION = "act_transformer_encoder_environment_token"


class TaskTokenContractError(RuntimeError):
    """Raised when TaskToken data or the installed ACT interface is incompatible."""


def task_token_contract(config: TaskTokenConfig) -> dict[str, object]:
    """Return the canonical serialized mapping and public integration contract."""
    _validate_token_config(config)
    return {
        "token_config": config.to_dict(),
        "normalization_mode": TASK_TOKEN_NORMALIZATION_MODE.value,
        "upstream_feature_type": FeatureType.ENV.value,
        "upstream_interface": TASK_TOKEN_UPSTREAM_INTERFACE,
    }


def task_token_architecture_fingerprint(config: TaskTokenConfig) -> str:
    """Fingerprint all semantics that affect the TaskToken architecture."""
    return f"sha256:{sha256_hex(task_token_contract(config))}"


def canonical_task_token(
    task: str | TaskSpec,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Encode one stable task as the exact float32 ENV-token input."""
    if isinstance(task, TaskSpec):
        task_id = stable_task_id(task)
    elif isinstance(task, str):
        task_id = task
    else:
        raise TypeError("task must be a stable task ID or TaskSpec")
    return torch.as_tensor(canonical_task_onehot(task_id), dtype=torch.float32, device=device)


def inject_task_token_observation(
    observation: Mapping[str, object],
    *,
    task: str | TaskSpec,
) -> dict[str, object]:
    """Add CanonicalTaskTokenV0 for inference without changing Panda qpos."""
    state = observation.get(STATE_FEATURE_KEY)
    if not isinstance(state, torch.Tensor):
        raise TaskTokenContractError("TaskToken observation requires a Panda state tensor")
    batched = state.ndim == 2
    _validate_panda_state(state, batched=batched)
    if TASK_TOKEN_FEATURE_KEY in observation:
        raise TaskTokenContractError("TaskToken observation already contains the ENV feature")
    result = dict(observation)
    token = canonical_task_token(task, device=state.device)
    result[TASK_TOKEN_FEATURE_KEY] = (
        token.unsqueeze(0).expand(state.shape[0], -1).clone() if batched else token
    )
    if result[STATE_FEATURE_KEY] is not state:
        raise RuntimeError("TaskToken injection unexpectedly replaced PandaPolicyStateV0")
    return result


def inject_task_token_training_batch(
    batch: Mapping[str, object],
    *,
    task_id_by_episode: Mapping[int, str],
) -> dict[str, object]:
    """Resolve each training token only through immutable M3B episode provenance."""
    state = batch.get(STATE_FEATURE_KEY)
    episode_index = batch.get("episode_index")
    if not isinstance(state, torch.Tensor) or not isinstance(episode_index, torch.Tensor):
        raise TaskTokenContractError(
            "TaskToken training requires state and numeric episode_index tensors"
        )
    _validate_panda_state(state, batched=True)
    if TASK_TOKEN_FEATURE_KEY in batch:
        raise TaskTokenContractError("raw TaskToken batch must not contain a precomputed token")
    if episode_index.ndim not in (1, 2):
        raise TaskTokenContractError("episode_index must be a one-dimensional batch")
    flattened = episode_index.reshape(-1)
    if flattened.shape[0] != state.shape[0]:
        raise TaskTokenContractError("episode_index count must equal the Panda state batch size")
    if flattened.dtype.is_floating_point or flattened.dtype == torch.bool:
        raise TaskTokenContractError("episode_index must use an integer dtype")
    indices = tuple(int(item) for item in flattened.detach().cpu().tolist())
    try:
        task_ids = tuple(task_id_by_episode[index] for index in indices)
    except KeyError as error:
        raise TaskTokenContractError(
            f"batch episode {error.args[0]} has no stable M3B task mapping"
        ) from error
    tokens = torch.stack(
        [canonical_task_token(task_id, device=state.device) for task_id in task_ids],
        dim=0,
    )
    if tokens.dtype != torch.float32 or tuple(tokens.shape) != (
        state.shape[0],
        CANONICAL_TASK_COUNT,
    ):
        raise RuntimeError("CanonicalTaskTokenV0 failed to produce float32[B,6]")
    result = dict(batch)
    result[TASK_TOKEN_FEATURE_KEY] = tokens
    if result[STATE_FEATURE_KEY] is not state:
        raise RuntimeError("TaskToken injection unexpectedly replaced PandaPolicyStateV0")
    return result


def build_task_token_act_config(
    model: ActModelConfig,
    *,
    device: str,
    use_amp: bool,
    optimization: ActOptimizationConfig | None = None,
    token_config: TaskTokenConfig | None = None,
) -> ACTConfig:
    """Add the public ACT ENV feature to an otherwise unchanged M4 config."""
    from langmani.policies.act_training import build_act_config

    effective_token_config = token_config or TaskTokenConfig(hidden_dimension=model.dim_model)
    _validate_token_config(effective_token_config)
    if model.state_dimension != effective_token_config.panda_state_dimension:
        raise TaskTokenContractError("TaskToken requires the unchanged 9D Panda policy state")
    if model.dim_model != effective_token_config.hidden_dimension:
        raise TaskTokenContractError("TaskToken hidden dimension must equal ACT dim_model")
    if model.action_dimension != effective_token_config.action_dimension:
        raise TaskTokenContractError("TaskToken action dimension must equal the M1 action schema")

    base = build_act_config(
        model,
        device=device,
        use_amp=use_amp,
        optimization=optimization,
    )
    input_features = dict(base.input_features)
    if TASK_TOKEN_FEATURE_KEY in input_features:
        raise TaskTokenContractError("base ACT config unexpectedly already contains TaskToken")
    input_features[TASK_TOKEN_FEATURE_KEY] = PolicyFeature(
        FeatureType.ENV,
        (effective_token_config.task_input_dimension,),
    )
    # ACTConfig's serialized normalization mapping uses FeatureType string
    # values.  Keep every key in that format so the public processor can
    # reconstruct the complete enum map, rather than creating a mixed-key map.
    normalization_mapping = {
        (key.value if isinstance(key, FeatureType) else str(key)): NormalizationMode(value)
        for key, value in base.normalization_mapping.items()
    }
    normalization_mapping[FeatureType.ENV.value] = TASK_TOKEN_NORMALIZATION_MODE
    result = replace(
        base,
        input_features=input_features,
        normalization_mapping=normalization_mapping,
    )
    validate_task_token_act_config(result, token_config=effective_token_config)
    return result


def validate_task_token_act_config(
    config: ACTConfig,
    *,
    token_config: TaskTokenConfig | None = None,
) -> None:
    """Fail if ACT no longer exposes the exact dedicated ENV token contract."""
    effective = token_config or TaskTokenConfig(hidden_dimension=config.dim_model)
    _validate_token_config(effective)
    if not isinstance(config, ACTConfig):
        raise TypeError("config must be an ACTConfig")
    state_features = [
        item for item in config.input_features.values() if item.type is FeatureType.STATE
    ]
    env_features = [item for item in config.input_features.values() if item.type is FeatureType.ENV]
    if len(state_features) != 1 or state_features[0].shape != (effective.panda_state_dimension,):
        raise TaskTokenContractError("ACT must retain one exact 9D Panda STATE feature")
    if set(key for key, item in config.input_features.items() if item.type is FeatureType.ENV) != {
        effective.task_feature_key
    }:
        raise TaskTokenContractError("ACT must expose TaskToken only at the canonical ENV key")
    if len(env_features) != 1 or env_features[0].shape != (effective.task_input_dimension,):
        raise TaskTokenContractError("ACT ENV TaskToken input must have shape (6,)")
    if config.dim_model != effective.hidden_dimension:
        raise TaskTokenContractError("ACT dim_model differs from the TaskToken hidden dimension")
    normalization = _normalization_mode(config, FeatureType.ENV)
    if normalization is not TASK_TOKEN_NORMALIZATION_MODE:
        raise TaskTokenContractError("TaskToken ENV input must use IDENTITY normalization")


def validate_task_token_policy(
    policy: ACTPolicy,
    *,
    token_config: TaskTokenConfig | None = None,
) -> None:
    """Audit the public linear projection and dedicated encoder-token allocation."""
    if not isinstance(policy, ACTPolicy):
        raise TypeError("policy must be an ACTPolicy")
    effective = token_config or TaskTokenConfig(hidden_dimension=policy.config.dim_model)
    validate_task_token_act_config(policy.config, token_config=effective)
    projection = getattr(policy.model, "encoder_env_state_input_proj", None)
    if not isinstance(projection, torch.nn.Linear):
        raise TaskTokenContractError("installed ACT lacks the public ENV input projection")
    if (projection.in_features, projection.out_features) != (
        effective.task_input_dimension,
        effective.hidden_dimension,
    ):
        raise TaskTokenContractError("ACT ENV projection is not linear_6_to_hidden_dimension")
    positions = getattr(policy.model, "encoder_1d_feature_pos_embed", None)
    if not isinstance(positions, torch.nn.Embedding) or positions.num_embeddings != 3:
        raise TaskTokenContractError(
            "ACT must allocate encoder positions for latent, Panda state, and ENV TaskToken"
        )


def learned_task_embeddings(policy: ACTPolicy) -> torch.Tensor:
    """Return six learned hidden vectors represented by the public ENV projection."""
    validate_task_token_policy(policy)
    projection = cast(torch.nn.Linear, policy.model.encoder_env_state_input_proj)
    result = projection.weight.transpose(0, 1)
    if projection.bias is not None:
        result = result + projection.bias.unsqueeze(0)
    if tuple(result.shape) != (CANONICAL_TASK_COUNT, policy.config.dim_model):
        raise RuntimeError("TaskToken projection produced an unexpected embedding table")
    return result


def _validate_token_config(config: TaskTokenConfig) -> None:
    if not isinstance(config, TaskTokenConfig):
        raise TypeError("token_config must be a TaskTokenConfig")
    if not isinstance(config.mapping, TaskTokenMapping):
        raise TaskTokenContractError("TaskToken config mapping is malformed")
    if tuple(config.mapping.ordered_task_ids) != CANONICAL_TASK_IDS:
        raise TaskTokenContractError("TaskToken mapping order differs from CanonicalTaskOneHotV0")
    if (
        config.task_feature_key != TASK_TOKEN_FEATURE_KEY
        or config.injection_location != TASK_TOKEN_INJECTION_LOCATION
        or config.task_input_dimension != CANONICAL_TASK_COUNT
        or config.panda_state_dimension != 9
        or config.action_dimension != 8
    ):
        raise TaskTokenContractError("TaskToken config differs from the public ACT ENV contract")


def _validate_panda_state(state: torch.Tensor, *, batched: bool = False) -> None:
    expected_shape = (9,) if not batched else None
    valid_shape = (
        tuple(state.shape) == expected_shape
        if not batched
        else state.ndim == 2 and state.shape[1] == 9
    )
    if state.dtype != torch.float32 or not valid_shape or not torch.isfinite(state).all():
        expected = "(9,)" if not batched else "(B,9)"
        raise TaskTokenContractError(f"PandaPolicyStateV0 must be finite float32{expected}")


def _normalization_mode(config: ACTConfig, feature_type: FeatureType) -> NormalizationMode:
    for key, value in config.normalization_mapping.items():
        normalized_key = FeatureType(key) if isinstance(key, str) else key
        if normalized_key is feature_type:
            return NormalizationMode(value)
    return NormalizationMode.IDENTITY


__all__ = [
    "TASK_TOKEN_FEATURE_KEY",
    "TASK_TOKEN_INJECTION_LOCATION",
    "TASK_TOKEN_NORMALIZATION_MODE",
    "TASK_TOKEN_UPSTREAM_INTERFACE",
    "TaskTokenContractError",
    "build_task_token_act_config",
    "canonical_task_token",
    "inject_task_token_observation",
    "inject_task_token_training_batch",
    "learned_task_embeddings",
    "task_token_architecture_fingerprint",
    "task_token_contract",
    "validate_task_token_act_config",
    "validate_task_token_policy",
]
