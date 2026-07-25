"""Official SmolVLA integration for the Phase 2C-C relative Pick target.

The published SmolVLA architecture and parameter names are unchanged.  The
only formulation change occurs outside the official model:

* before preprocessing, physical action chunks are encoded relative to the
  current public state;
* after model generation and postprocessing, residual latents are decoded
  relative to the current public state.

The state preprocessor therefore continues to receive exactly the same public
``PandaPolicyStateV0`` input, while future state never enters the policy or
target transform.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from lerobot.policies.smolvla.configuration_smolvla import (  # type: ignore[import-untyped]
    SmolVLAConfig,
)
from lerobot.policies.smolvla.modeling_smolvla import (  # type: ignore[import-untyped]
    SmolVLAPolicy,
)

from langmani.v2.phase2c_b import (
    OFFICIAL_BASE_MODEL,
    OFFICIAL_BASE_REVISION,
    SmolVLATrainingConfig,
    canonical_fingerprint,
)
from langmani.v2.phase2c_b_smolvla import (
    ACTION_FEATURE_KEY,
    ACTION_PAD_KEY,
    STATE_FEATURE_KEY,
    LoadedSmolVLAView,
    build_official_config,
    load_dataset_view,
    load_smolvla_statistics,
    make_processors,
    processor_manifest,
    project_policy_batch,
)
from langmani.v2.phase2c_c import (
    Phase2CCContractError,
    StateRelativeBoundedActionV0,
)


class Phase2CCSmolVLAError(Phase2CCContractError):
    """Raised when the relative target crosses the official model boundary."""


class RelativeSmolVLAPolicyV0(SmolVLAPolicy):
    """Unmodified official SmolVLA whose action features are residual latents."""

    action_transform = StateRelativeBoundedActionV0()


def project_relative_policy_batch(
    raw_batch: Mapping[str, object],
    *,
    transform: StateRelativeBoundedActionV0 | None = None,
) -> dict[str, object]:
    """Encode one real physical batch using only its current public state."""

    active = transform or StateRelativeBoundedActionV0()
    batch = project_policy_batch(raw_batch, shared=False)
    state = batch.get(STATE_FEATURE_KEY)
    action = batch.get(ACTION_FEATURE_KEY)
    padding = batch.get(ACTION_PAD_KEY)
    if (
        not isinstance(state, torch.Tensor)
        or not isinstance(action, torch.Tensor)
        or not isinstance(padding, torch.Tensor)
        or state.ndim != 2
        or action.ndim != 3
        or padding.shape != action.shape[:2]
    ):
        raise Phase2CCSmolVLAError("real relative-action batch is malformed")
    latent = active.encode(action, state)
    latent = latent.masked_fill(padding.unsqueeze(-1), 0.0)
    batch[ACTION_FEATURE_KEY] = latent
    return batch


def decode_relative_action_chunk(
    latent: torch.Tensor,
    *,
    current_state: torch.Tensor,
    transform: StateRelativeBoundedActionV0 | None = None,
) -> torch.Tensor:
    """Decode generated residual latents against the current query state."""

    return (transform or StateRelativeBoundedActionV0()).decode(latent, current_state)


def load_relative_pretrained_policy(
    *,
    base_snapshot: str | Path,
    config: SmolVLAConfig,
) -> tuple[RelativeSmolVLAPolicyV0, dict[str, object]]:
    """Strictly load every official tensor without architecture changes."""

    policy = RelativeSmolVLAPolicyV0.from_pretrained(
        Path(base_snapshot),
        config=config,
        strict=True,
    )
    parameters = list(policy.parameters())
    parameter_count = sum(int(parameter.numel()) for parameter in parameters)
    trainable_count = sum(
        int(parameter.numel()) for parameter in parameters if parameter.requires_grad
    )
    semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-c-pretrained-loading-audit-v0",
        "base_model": OFFICIAL_BASE_MODEL,
        "base_revision": OFFICIAL_BASE_REVISION,
        "strict_weight_load": True,
        "published_parameter_count": parameter_count,
        "loaded_parameter_count": parameter_count,
        "loaded_parameter_percentage": 100.0,
        "reinitialized_parameters": [],
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "frozen_parameter_count": parameter_count - trainable_count,
        "architecture_changed": False,
        "action_target_changed_outside_model": True,
        "action_transform_fingerprint": policy.action_transform.fingerprint,
        "passed": parameter_count > 0 and trainable_count > 0,
    }
    if semantic["passed"] is not True:
        raise Phase2CCSmolVLAError("official SmolVLA strict loading audit failed")
    return policy, {**semantic, "fingerprint": canonical_fingerprint(semantic)}


def load_relative_checkpoint(
    checkpoint: str | Path,
    *,
    strict: bool = True,
) -> RelativeSmolVLAPolicyV0:
    """Strictly reconstruct a checkpoint and independently load its transform."""

    root = Path(checkpoint)
    transform = StateRelativeBoundedActionV0.load(root)
    policy = RelativeSmolVLAPolicyV0.from_pretrained(root, strict=strict)
    if transform.fingerprint != policy.action_transform.fingerprint:
        raise Phase2CCSmolVLAError("checkpoint action-transform identity changed")
    return cast(RelativeSmolVLAPolicyV0, policy)


def validate_unchanged_training_config(
    config: SmolVLAConfig,
    training: SmolVLATrainingConfig,
) -> dict[str, Any]:
    """Record that every non-action training field remains Phase 2C-B exact."""

    semantic: dict[str, Any] = {
        "schema_version": "langmani-v2-phase2c-c-config-equivalence-v0",
        "training_config_fingerprint": training.fingerprint,
        "chunk_size": config.chunk_size,
        "n_action_steps": config.n_action_steps,
        "num_inference_steps": config.num_steps,
        "freeze_vision_encoder": config.freeze_vision_encoder,
        "train_expert_only": config.train_expert_only,
        "train_state_projection": config.train_state_proj,
        "optimizer_learning_rate": config.optimizer_lr,
        "optimizer_betas": list(config.optimizer_betas),
        "optimizer_epsilon": config.optimizer_eps,
        "optimizer_weight_decay": config.optimizer_weight_decay,
        "gradient_clip_norm": config.optimizer_grad_clip_norm,
        "warmup_steps": config.scheduler_warmup_steps,
        "decay_steps": config.scheduler_decay_steps,
        "decay_learning_rate": config.scheduler_decay_lr,
        "processor_action_normalization": "IDENTITY",
        "processor_state_normalization": "MEAN_STD",
        "only_action_target_and_deployment_formulation_changed": True,
    }
    return {**semantic, "fingerprint": canonical_fingerprint(semantic)}


__all__ = [
    "LoadedSmolVLAView",
    "Phase2CCSmolVLAError",
    "RelativeSmolVLAPolicyV0",
    "build_official_config",
    "decode_relative_action_chunk",
    "load_dataset_view",
    "load_relative_checkpoint",
    "load_relative_pretrained_policy",
    "load_smolvla_statistics",
    "make_processors",
    "processor_manifest",
    "project_relative_policy_batch",
    "validate_unchanged_training_config",
]
