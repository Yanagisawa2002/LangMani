"""Isolated LeRobot 0.6.0 adapter for ACT-Mixed-FactorFiLM.

All access to ACT's semi-stable internal modules is confined here.  The
extension keeps the installed ACT implementation, loss, queue, Transformer,
and serialization.  Instance-local PyTorch forward hooks condition only the
ResNet feature map and the encoded Panda-state token.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies.act import ACTConfig, ACTPolicy, make_act_pre_post_processors
from torch import nn

from langmani.policies.act_factor_film_conditioning import (
    extract_factor_film_runtime_tensors,
)
from langmani.policies.act_factor_film_types import (
    FACTOR_FILM_ARCHITECTURE_NAME,
    FACTOR_FILM_INITIALIZATION,
    SUPPORTED_LEROBOT_VERSION,
    FactorFiLMArchitectureIdentity,
    FactorFiLMConfig,
    FactorFiLMContractError,
    canonical_fingerprint,
)

FACTOR_FILM_CONFIG_FILENAME = "langmani_factor_film_config.json"
FACTOR_FILM_POLICY_NAME = "act_factor_film"
FACTOR_FILM_FEATURE_MAP_KEY = "feature_map"
_FILM_WEIGHT_STD = 1e-5
_P = "POSITIONAL_OR_KEYWORD"
_K = "KEYWORD_ONLY"
_V = "VAR_KEYWORD"
_EXPECTED_PUBLIC_SIGNATURES: dict[str, tuple[tuple[str, str, bool, str | None], ...]] = {
    "ACTPolicy.__init__": (
        ("self", _P, False, None),
        ("config", _P, False, None),
        ("kwargs", _V, False, None),
    ),
    "ACTPolicy.forward": (("self", _P, False, None), ("batch", _P, False, None)),
    "ACTPolicy.predict_action_chunk": (
        ("self", _P, False, None),
        ("batch", _P, False, None),
    ),
    "ACTPolicy.select_action": (
        ("self", _P, False, None),
        ("batch", _P, False, None),
    ),
    "ACTPolicy.save_pretrained": (
        ("self", _P, False, None),
        ("save_directory", _P, False, None),
        ("state_dict", _K, True, "None"),
        ("repo_id", _K, True, "None"),
        ("push_to_hub", _K, True, "False"),
        ("card_kwargs", _K, True, "None"),
        ("push_to_hub_kwargs", _V, False, None),
    ),
    "ACTPolicy.from_pretrained": (
        ("pretrained_name_or_path", _P, False, None),
        ("config", _K, True, "None"),
        ("force_download", _K, True, "False"),
        ("resume_download", _K, True, "None"),
        ("proxies", _K, True, "None"),
        ("token", _K, True, "None"),
        ("cache_dir", _K, True, "None"),
        ("local_files_only", _K, True, "False"),
        ("revision", _K, True, "None"),
        ("strict", _K, True, "False"),
        ("kwargs", _V, False, None),
    ),
    "PreTrainedConfig.from_pretrained": (
        ("pretrained_name_or_path", _P, False, None),
        ("force_download", _K, True, "False"),
        ("resume_download", _K, True, "None"),
        ("proxies", _K, True, "None"),
        ("token", _K, True, "None"),
        ("cache_dir", _K, True, "None"),
        ("local_files_only", _K, True, "False"),
        ("revision", _K, True, "None"),
        ("policy_kwargs", _V, False, None),
    ),
    "make_act_pre_post_processors": (
        ("config", _P, False, None),
        ("dataset_stats", _P, True, "None"),
    ),
}


class FactorFiLMUpstreamContractError(RuntimeError):
    """Raised when installed LeRobot no longer matches the inspected ACT path."""


def _signature_contract(value: Callable[..., object]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for parameter in inspect.signature(value).parameters.values():
        default = None if parameter.default is inspect.Parameter.empty else repr(parameter.default)
        result.append(
            {
                "name": parameter.name,
                "kind": parameter.kind.name,
                "has_default": parameter.default is not inspect.Parameter.empty,
                "default": default,
            }
        )
    return result


def _signature_tuple(
    value: Callable[..., object],
) -> tuple[tuple[str, str, bool, str | None], ...]:
    return tuple(
        (
            cast(str, item["name"]),
            cast(str, item["kind"]),
            cast(bool, item["has_default"]),
            cast(str | None, item["default"]),
        )
        for item in _signature_contract(value)
    )


def factor_film_upstream_contract() -> dict[str, object]:
    """Record the exact public symbols and semi-stable methods consumed here."""
    installed = version("lerobot")
    if installed != SUPPORTED_LEROBOT_VERSION:
        raise FactorFiLMUpstreamContractError(
            f"FactorFiLM requires LeRobot {SUPPORTED_LEROBOT_VERSION}, found {installed}"
        )
    callables: dict[str, Callable[..., object]] = {
        "ACTPolicy.__init__": ACTPolicy.__init__,
        "ACTPolicy.forward": ACTPolicy.forward,
        "ACTPolicy.predict_action_chunk": ACTPolicy.predict_action_chunk,
        "ACTPolicy.select_action": ACTPolicy.select_action,
        "ACTPolicy.save_pretrained": ACTPolicy.save_pretrained,
        "ACTPolicy.from_pretrained": ACTPolicy.from_pretrained,
        "PreTrainedConfig.from_pretrained": PreTrainedConfig.from_pretrained,
        "make_act_pre_post_processors": make_act_pre_post_processors,
    }
    for name, expected in _EXPECTED_PUBLIC_SIGNATURES.items():
        if _signature_tuple(callables[name]) != expected:
            raise FactorFiLMUpstreamContractError(f"installed LeRobot signature changed for {name}")
    return {
        "lerobot_version": installed,
        "public_symbols": {
            "PreTrainedConfig": "lerobot.configs.PreTrainedConfig",
            "ACTConfig": "lerobot.policies.act.ACTConfig",
            "ACTPolicy": "lerobot.policies.act.ACTPolicy",
            "make_act_pre_post_processors": ("lerobot.policies.act.make_act_pre_post_processors"),
            "ACTPolicy.save_pretrained": (
                "lerobot.policies.pretrained.PreTrainedPolicy.save_pretrained"
            ),
            "ACTPolicy.from_pretrained": (
                "lerobot.policies.pretrained.PreTrainedPolicy.from_pretrained"
            ),
        },
        "public_signatures": {
            name: _signature_contract(value) for name, value in callables.items()
        },
        "semi_stable_model_attributes": {
            "model": "lerobot.policies.act.modeling_act.ACT",
            "visual_backbone": "policy.model.backbone",
            "visual_backbone_output_key": FACTOR_FILM_FEATURE_MAP_KEY,
            "image_projection": "policy.model.encoder_img_feat_input_proj",
            "state_projection": "policy.model.encoder_robot_state_input_proj",
            "one_dimensional_positions": "policy.model.encoder_1d_feature_pos_embed",
            "decoder_positions": "policy.model.decoder_pos_embed",
            "action_head": "policy.model.action_head",
        },
        "expected_tensor_structure": {
            "image": ["B", 3, 256, 256],
            "resnet18_feature_map": ["B", 512, 8, 8],
            "encoded_state": ["B", "dim_model"],
            "transformer_tokens": ["latent", "state", "64_image_tokens"],
            "action_chunk": ["B", "chunk_size", 8],
        },
        "private_module_imported": False,
        "global_monkey_patch": False,
        "integration_mechanism": "instance_local_torch_forward_hooks",
    }


def _validate_indices(
    indices: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    count: int,
    label: str,
) -> None:
    if not isinstance(indices, torch.Tensor):
        raise TypeError(f"{label} indices must be a torch Tensor")
    if indices.dtype != torch.long or tuple(indices.shape) != (batch_size,):
        raise FactorFiLMContractError(f"{label} indices must be torch.long[B]")
    if indices.device != device:
        raise FactorFiLMContractError(f"{label} indices and features use different devices")
    if bool(torch.any((indices < 0) | (indices >= count))):
        raise FactorFiLMContractError(f"{label} condition index is out of range")


class TargetObjectConditionV0(nn.Module):
    """Learned target-object embedding and residual FiLM over ResNet features."""

    def __init__(self, config: FactorFiLMConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            config.target_object_count,
            config.object_embedding_dimension,
        )
        self.film_projection = nn.Linear(
            config.object_embedding_dimension,
            2 * config.visual_feature_channels,
        )
        nn.init.normal_(self.film_projection.weight, mean=0.0, std=_FILM_WEIGHT_STD)
        nn.init.zeros_(self.film_projection.bias)

    def gamma_beta(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.film_projection(self.embedding(indices))
        return values.chunk(2, dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(features, torch.Tensor) or features.ndim != 4:
            raise FactorFiLMContractError("object FiLM features must have shape [B,C,H,W]")
        if features.shape[1] != self.config.visual_feature_channels:
            raise FactorFiLMContractError("object FiLM visual channel count changed")
        if not features.dtype.is_floating_point:
            raise FactorFiLMContractError("object FiLM features must be floating tensors")
        _validate_indices(
            indices,
            batch_size=features.shape[0],
            device=features.device,
            count=self.config.target_object_count,
            label="target-object",
        )
        gamma, beta = self.gamma_beta(indices)
        if tuple(gamma.shape) != (features.shape[0], features.shape[1]):
            raise FactorFiLMContractError("object gamma/beta shape differs from [B,C]")
        gamma = gamma.to(dtype=features.dtype).unsqueeze(-1).unsqueeze(-1)
        beta = beta.to(dtype=features.dtype).unsqueeze(-1).unsqueeze(-1)
        if tuple(gamma.shape) != (features.shape[0], features.shape[1], 1, 1):
            raise RuntimeError("object FiLM broadcasting failed")
        return features * (1.0 + gamma) + beta


class DestinationBinConditionV0(nn.Module):
    """Learned destination-bin embedding and residual FiLM over the state token."""

    def __init__(self, config: FactorFiLMConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            config.destination_bin_count,
            config.bin_embedding_dimension,
        )
        self.film_projection = nn.Linear(
            config.bin_embedding_dimension,
            2 * config.state_context_dimension,
        )
        nn.init.normal_(self.film_projection.weight, mean=0.0, std=_FILM_WEIGHT_STD)
        nn.init.zeros_(self.film_projection.bias)

    def gamma_beta(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.film_projection(self.embedding(indices))
        return values.chunk(2, dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise FactorFiLMContractError("bin FiLM features must have shape [B,D]")
        if features.shape[1] != self.config.state_context_dimension:
            raise FactorFiLMContractError("bin FiLM state/context dimension changed")
        if not features.dtype.is_floating_point:
            raise FactorFiLMContractError("bin FiLM features must be floating tensors")
        _validate_indices(
            indices,
            batch_size=features.shape[0],
            device=features.device,
            count=self.config.destination_bin_count,
            label="destination-bin",
        )
        gamma, beta = self.gamma_beta(indices)
        if tuple(gamma.shape) != tuple(features.shape):
            raise FactorFiLMContractError("bin gamma/beta shape differs from encoded state")
        return features * (1.0 + gamma.to(dtype=features.dtype)) + beta.to(dtype=features.dtype)


class FactorFiLMACTPolicy(ACTPolicy):
    """One ACT subclass with separated object/visual and bin/state FiLM paths."""

    config_class = ACTConfig
    name = FACTOR_FILM_POLICY_NAME

    def __init__(
        self,
        config: ACTConfig,
        *,
        factor_film_config: FactorFiLMConfig | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(config, **kwargs)
        visual_projection = getattr(self.model, "encoder_img_feat_input_proj", None)
        if not isinstance(visual_projection, nn.Conv2d):
            raise FactorFiLMUpstreamContractError("installed ACT lacks its image projection")
        effective = factor_film_config or FactorFiLMConfig(
            visual_feature_channels=visual_projection.in_channels,
            state_context_dimension=config.dim_model,
        )
        if effective.visual_feature_channels != visual_projection.in_channels:
            raise FactorFiLMUpstreamContractError(
                "FactorFiLM visual channels differ from ACT backbone output"
            )
        if effective.state_context_dimension != config.dim_model:
            raise FactorFiLMUpstreamContractError(
                "FactorFiLM state dimension differs from ACT dim_model"
            )
        self.factor_film_config = effective
        self.target_object_condition = TargetObjectConditionV0(effective)
        self.destination_bin_condition = DestinationBinConditionV0(effective)
        self._active_object_indices: torch.Tensor | None = None
        self._active_bin_indices: torch.Tensor | None = None
        self._visual_hook_calls = 0
        self._state_hook_calls = 0
        self._visual_hook_handle = self.model.backbone.register_forward_hook(
            self._condition_visual_hook
        )
        self._state_hook_handle = self.model.encoder_robot_state_input_proj.register_forward_hook(
            self._condition_state_hook
        )
        validate_factor_film_policy_structure(self)

    def _condition_visual_hook(
        self,
        _module: nn.Module,
        _inputs: tuple[object, ...],
        output: object,
    ) -> object:
        if self._active_object_indices is None:
            raise FactorFiLMContractError("visual FiLM hook ran without an active object condition")
        if not isinstance(output, Mapping) or set(output) != {FACTOR_FILM_FEATURE_MAP_KEY}:
            raise FactorFiLMUpstreamContractError(
                "ACT backbone no longer returns only the feature_map tensor"
            )
        feature_map = output[FACTOR_FILM_FEATURE_MAP_KEY]
        if not isinstance(feature_map, torch.Tensor):
            raise FactorFiLMUpstreamContractError("ACT feature_map is not a tensor")
        if tuple(feature_map.shape[1:]) != (512, 8, 8):
            raise FactorFiLMUpstreamContractError(
                "ACT ResNet-18 feature_map must have trailing shape (512,8,8)"
            )
        self._visual_hook_calls += 1
        conditioned = self.target_object_condition(feature_map, self._active_object_indices)
        result = dict(output)
        result[FACTOR_FILM_FEATURE_MAP_KEY] = conditioned
        return result

    def _condition_state_hook(
        self,
        _module: nn.Module,
        _inputs: tuple[object, ...],
        output: object,
    ) -> object:
        if self._active_bin_indices is None:
            raise FactorFiLMContractError("state FiLM hook ran without an active bin condition")
        if not isinstance(output, torch.Tensor):
            raise FactorFiLMUpstreamContractError("ACT encoded state is not a tensor")
        if output.ndim != 2 or output.shape[1] != self.factor_film_config.state_context_dimension:
            raise FactorFiLMUpstreamContractError(
                "ACT encoded state no longer has shape [B,dim_model]"
            )
        self._state_hook_calls += 1
        return self.destination_bin_condition(output, self._active_bin_indices)

    def _conditioned_call(
        self,
        batch: Mapping[str, object],
        call: Callable[[dict[str, torch.Tensor]], Any],
    ) -> Any:
        if self._active_object_indices is not None or self._active_bin_indices is not None:
            raise FactorFiLMContractError("FactorFiLM policy calls must not be nested or reentrant")
        clean, object_indices, bin_indices = extract_factor_film_runtime_tensors(batch)
        self._active_object_indices = object_indices
        self._active_bin_indices = bin_indices
        self._visual_hook_calls = 0
        self._state_hook_calls = 0
        completed = False
        try:
            result = call(cast(dict[str, torch.Tensor], clean))
            completed = True
            return result
        finally:
            visual_calls = self._visual_hook_calls
            state_calls = self._state_hook_calls
            self._active_object_indices = None
            self._active_bin_indices = None
            self._visual_hook_calls = 0
            self._state_hook_calls = 0
            if completed and (visual_calls, state_calls) != (1, 1):
                raise FactorFiLMUpstreamContractError(
                    "ACT FactorFiLM expected exactly one visual and one state hook call"
                )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return cast(
            tuple[torch.Tensor, dict[str, float]],
            self._conditioned_call(batch, super().forward),
        )

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self._conditioned_call(batch, super().predict_action_chunk),
        )

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        push_to_hub: bool = False,
        **kwargs: object,
    ) -> str | None:
        if push_to_hub:
            raise FactorFiLMContractError("FactorFiLM artifacts are local-only")
        destination = Path(save_directory)
        if destination.exists() and (
            not destination.is_dir()
            or destination.is_symlink()
            or destination.is_junction()
            or any(destination.iterdir())
        ):
            raise FactorFiLMContractError(
                "FactorFiLM save destination must be a new or empty real directory"
            )
        result = super().save_pretrained(
            destination,
            push_to_hub=False,
            **kwargs,
        )
        path = destination / FACTOR_FILM_CONFIG_FILENAME
        payload = {
            "architecture_name": FACTOR_FILM_ARCHITECTURE_NAME,
            "factor_film_config": self.factor_film_config.to_dict(),
            "factor_film_config_fingerprint": self.factor_film_config.fingerprint,
            "initialization": FACTOR_FILM_INITIALIZATION,
        }
        try:
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
                stream.write("\n")
        except FileExistsError as error:
            raise FactorFiLMContractError(
                "refusing to overwrite FactorFiLM config sidecar"
            ) from error
        return result

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: ACTConfig | None = None,
        local_files_only: bool = True,
        strict: bool = True,
        **kwargs: object,
    ) -> FactorFiLMACTPolicy:
        if not local_files_only or not strict:
            raise FactorFiLMContractError(
                "FactorFiLM reload requires local_files_only=True and strict=True"
            )
        root = Path(pretrained_name_or_path)
        sidecar = root / FACTOR_FILM_CONFIG_FILENAME
        if (
            not root.is_dir()
            or root.is_symlink()
            or root.is_junction()
            or sidecar.is_symlink()
            or sidecar.is_junction()
        ):
            raise FactorFiLMContractError("FactorFiLM reload root is missing or linked")
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise FactorFiLMContractError(
                "FactorFiLM config sidecar is missing or invalid"
            ) from error
        expected_sidecar_fields = {
            "architecture_name",
            "factor_film_config",
            "factor_film_config_fingerprint",
            "initialization",
        }
        if not isinstance(payload, dict) or set(payload) != expected_sidecar_fields:
            raise FactorFiLMContractError("FactorFiLM config sidecar fields changed")
        if payload.get("architecture_name") != FACTOR_FILM_ARCHITECTURE_NAME:
            raise FactorFiLMContractError("FactorFiLM config sidecar has the wrong architecture")
        if payload.get("initialization") != FACTOR_FILM_INITIALIZATION:
            raise FactorFiLMContractError("FactorFiLM config sidecar initialization changed")
        raw_config = payload.get("factor_film_config")
        if not isinstance(raw_config, Mapping):
            raise FactorFiLMContractError("FactorFiLM sidecar lacks a configuration object")
        film_config = FactorFiLMConfig.from_dict(raw_config)
        if payload.get("factor_film_config_fingerprint") != film_config.fingerprint:
            raise FactorFiLMContractError("FactorFiLM config sidecar fingerprint mismatch")
        effective_act_config = config
        if effective_act_config is None:
            loaded = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=root,
                local_files_only=True,
            )
            if not isinstance(loaded, ACTConfig):
                raise FactorFiLMContractError("saved FactorFiLM ACT config is malformed")
            effective_act_config = loaded
        policy = super().from_pretrained(
            root,
            config=effective_act_config,
            factor_film_config=film_config,
            local_files_only=True,
            strict=True,
            **kwargs,
        )
        if not isinstance(policy, cls):
            raise FactorFiLMContractError("FactorFiLM reload returned the wrong policy type")
        validate_factor_film_policy_structure(policy)
        return policy


def validate_factor_film_policy_structure(policy: FactorFiLMACTPolicy) -> None:
    """Fail closed if the two semi-stable ACT injection points drift."""
    if version("lerobot") != SUPPORTED_LEROBOT_VERSION:
        raise FactorFiLMUpstreamContractError("installed LeRobot version changed")
    factor_film_upstream_contract()
    if not isinstance(policy, FactorFiLMACTPolicy):
        raise TypeError("policy must be FactorFiLMACTPolicy")
    config = policy.config
    if config.robot_state_feature is None or config.robot_state_feature.shape != (9,):
        raise FactorFiLMUpstreamContractError("FactorFiLM requires one 9D ACT state feature")
    if config.env_state_feature is not None:
        raise FactorFiLMUpstreamContractError("FactorFiLM must not add an ACT ENV/task token")
    if len(config.image_features) != 1 or config.chunk_size < 1:
        raise FactorFiLMUpstreamContractError("FactorFiLM requires one ACT policy camera")
    camera = config.image_features.get("observation.images.base_camera")
    if camera is None or camera.shape != (3, 256, 256):
        raise FactorFiLMUpstreamContractError(
            "FactorFiLM requires base_camera with shape (3,256,256)"
        )
    model = policy.model
    state_projection = getattr(model, "encoder_robot_state_input_proj", None)
    image_projection = getattr(model, "encoder_img_feat_input_proj", None)
    positions = getattr(model, "encoder_1d_feature_pos_embed", None)
    decoder_positions = getattr(model, "decoder_pos_embed", None)
    action_head = getattr(model, "action_head", None)
    if not isinstance(state_projection, nn.Linear) or (
        state_projection.in_features,
        state_projection.out_features,
    ) != (9, config.dim_model):
        raise FactorFiLMUpstreamContractError("ACT state projection signature changed")
    if not isinstance(image_projection, nn.Conv2d) or (
        image_projection.in_channels,
        image_projection.out_channels,
        image_projection.kernel_size,
    ) != (policy.factor_film_config.visual_feature_channels, config.dim_model, (1, 1)):
        raise FactorFiLMUpstreamContractError("ACT image projection signature changed")
    if not isinstance(positions, nn.Embedding) or (
        positions.num_embeddings,
        positions.embedding_dim,
    ) != (2, config.dim_model):
        raise FactorFiLMUpstreamContractError(
            "ACT must keep exactly latent and Panda-state 1D tokens"
        )
    if (
        not isinstance(decoder_positions, nn.Embedding)
        or decoder_positions.num_embeddings != config.chunk_size
    ):
        raise FactorFiLMUpstreamContractError("ACT decoder action-chunk structure changed")
    if not isinstance(action_head, nn.Linear) or (
        action_head.in_features,
        action_head.out_features,
    ) != (config.dim_model, 8):
        raise FactorFiLMUpstreamContractError("ACT action head changed")
    if (
        type(model).__module__ != "lerobot.policies.act.modeling_act"
        or type(model).__name__ != "ACT"
    ):
        raise FactorFiLMUpstreamContractError("ACT underlying model type changed")


def factor_film_parameter_counts(policy: FactorFiLMACTPolicy) -> tuple[int, int, int]:
    """Return base ACT, complete FactorFiLM policy, and exact increase counts."""
    validate_factor_film_policy_structure(policy)
    base_count = sum(parameter.numel() for parameter in policy.model.parameters())
    factor_count = sum(parameter.numel() for parameter in policy.parameters())
    increase = sum(
        parameter.numel()
        for module in (policy.target_object_condition, policy.destination_bin_condition)
        for parameter in module.parameters()
    )
    if factor_count - base_count != increase:
        raise FactorFiLMContractError("only the two FiLM paths may add parameters")
    return base_count, factor_count, increase


def build_factor_film_architecture_identity(
    policy: FactorFiLMACTPolicy,
    *,
    base_act_configuration: Mapping[str, object],
) -> FactorFiLMArchitectureIdentity:
    """Bind the inspected upstream path and measured parameter increase."""
    base_count, factor_count, increase = factor_film_parameter_counts(policy)
    return FactorFiLMArchitectureIdentity(
        base_act_configuration_fingerprint=canonical_fingerprint(base_act_configuration),
        factor_film_config=policy.factor_film_config,
        upstream_contract=factor_film_upstream_contract(),
        base_parameter_count=base_count,
        factor_film_parameter_count=factor_count,
        parameter_count_increase=increase,
    )


__all__ = [
    "FACTOR_FILM_CONFIG_FILENAME",
    "DestinationBinConditionV0",
    "FactorFiLMACTPolicy",
    "FactorFiLMUpstreamContractError",
    "TargetObjectConditionV0",
    "build_factor_film_architecture_identity",
    "factor_film_parameter_counts",
    "factor_film_upstream_contract",
    "validate_factor_film_policy_structure",
]
