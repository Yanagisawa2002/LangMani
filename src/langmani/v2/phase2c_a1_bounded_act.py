"""Structurally bounded ACT output path for LangMani 2.0 Phase 2C-A.1.

The bounded mapping is part of the trainable policy:

``raw = Linear(decoder_token)``
``z = tanh(raw)``
``action = lower + 0.5 * (z + 1) * (upper - lower)``

The matching ACT processor configuration keeps actions in physical
``pd_joint_pos`` coordinates.  No clipping, projection, gripper thresholding,
or evaluator/environment correction is used.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Final

import numpy as np
import torch
from lerobot.configs import FeatureType, NormalizationMode, PreTrainedConfig
from lerobot.policies.act import ACTConfig, ACTPolicy
from torch import nn

from langmani.datasets.identity import sha256_hex
from langmani.v2.phase2b5 import PANDA_ACTION_HIGH, PANDA_ACTION_LOW

BOUNDED_ACTION_HEAD_IDENTITY: Final = "bounded_action_head_v1"
BOUNDED_ACTION_HEAD_CONFIG_FILE: Final = "langmani_bounded_action_head_v1.json"
BOUNDED_ACTION_HEAD_SCHEMA: Final = "langmani-v2-phase2c-a1-bounded-action-head-v1"
ACTION_BOUNDS_SCHEMA: Final = "langmani-v2-phase2c-a1-native-action-bounds-v0"
ACTION_BOUNDS_SOURCE: Final = (
    "ManiSkill 3.0.1 Panda pd_joint_pos action space; independently observed "
    "identical for PickCube-v1, StackCube-v1, and PushCube-v1"
)


class BoundedACTContractError(RuntimeError):
    """Raised when the bounded ACT construction or serialization drifts."""


def _float_list(values: np.ndarray | Sequence[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (8,) or not np.isfinite(array).all():
        raise BoundedACTContractError("native action bounds must be finite float32[8]")
    return [float(value) for value in array]


def native_action_bounds_manifest() -> dict[str, object]:
    """Return the content-bound, native Panda action-space contract."""

    semantic = {
        "schema_version": ACTION_BOUNDS_SCHEMA,
        "source": ACTION_BOUNDS_SOURCE,
        "control_mode": "pd_joint_pos",
        "dtype": "float32",
        "shape": [8],
        "tasks": ["PickCube-v1", "StackCube-v1", "PushCube-v1"],
        "task_consistency_verified": True,
        "lower": _float_list(PANDA_ACTION_LOW),
        "upper": _float_list(PANDA_ACTION_HIGH),
    }
    return {
        **semantic,
        "fingerprint": f"sha256:{sha256_hex(semantic)}",
    }


def bounded_action_head_manifest() -> dict[str, object]:
    """Return the immutable semantic identity of the only authorized repair."""

    bounds = native_action_bounds_manifest()
    semantic = {
        "schema_version": BOUNDED_ACTION_HEAD_SCHEMA,
        "identity": BOUNDED_ACTION_HEAD_IDENTITY,
        "raw_head": "torch.nn.Linear(dim_model, 8)",
        "normalized_mapping": "z = tanh(u)",
        "physical_mapping": ("a = lower + 0.5 * (z + 1) * (upper - lower)"),
        "training_action_representation": "native_physical_pd_joint_pos_float32",
        "inference_action_representation": "native_physical_pd_joint_pos_float32",
        "action_normalization_mode": "IDENTITY",
        "loss": "padding_masked_physical_action_l1_plus_existing_act_kl",
        "action_bounds_fingerprint": bounds["fingerprint"],
        "clipping": False,
        "projection": False,
        "rejection_sampling": False,
        "zero_or_previous_action_replacement": False,
        "gripper_thresholding": False,
        "environment_correction_relied_upon": False,
    }
    return {
        **semantic,
        "fingerprint": f"sha256:{sha256_hex(semantic)}",
    }


class BoundedActionHeadV1(nn.Module):
    """Trainable linear ACT head followed by the frozen physical bound map."""

    def __init__(
        self,
        in_features: int,
        *,
        lower: Sequence[float] = tuple(float(value) for value in PANDA_ACTION_LOW),
        upper: Sequence[float] = tuple(float(value) for value in PANDA_ACTION_HIGH),
    ) -> None:
        super().__init__()
        if in_features < 1:
            raise BoundedACTContractError("bounded action head needs positive input width")
        lower_tensor = torch.tensor(tuple(lower), dtype=torch.float32)
        upper_tensor = torch.tensor(tuple(upper), dtype=torch.float32)
        if (
            tuple(lower_tensor.shape) != (8,)
            or tuple(upper_tensor.shape) != (8,)
            or not bool(torch.isfinite(lower_tensor).all())
            or not bool(torch.isfinite(upper_tensor).all())
            or not bool(torch.all(lower_tensor < upper_tensor))
        ):
            raise BoundedACTContractError("invalid native action bounds")
        self.linear = nn.Linear(in_features, 8)
        self.register_buffer("lower", lower_tensor, persistent=True)
        self.register_buffer("upper", upper_tensor, persistent=True)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
    ) -> BoundedActionHeadV1:
        """Replace the installed head while preserving its seeded initialization."""

        if linear.out_features != 8 or linear.bias is None:
            raise BoundedACTContractError("installed ACT action head is not Linear(D,8)")
        head = cls(linear.in_features)
        with torch.no_grad():
            head.linear.weight.copy_(linear.weight)
            head.linear.bias.copy_(linear.bias)
        return head

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.linear(features)
        normalized = torch.tanh(raw)
        lower = self.lower.to(device=raw.device, dtype=raw.dtype)
        upper = self.upper.to(device=raw.device, dtype=raw.dtype)
        return lower + 0.5 * (normalized + 1.0) * (upper - lower)


def build_bounded_act_config(base: ACTConfig) -> ACTConfig:
    """Keep state/image preprocessing but train actions in physical coordinates."""

    normalization_mapping = {
        (key.value if isinstance(key, FeatureType) else str(key)): NormalizationMode(value)
        for key, value in base.normalization_mapping.items()
    }
    normalization_mapping[FeatureType.ACTION.value] = NormalizationMode.IDENTITY
    config = replace(base, normalization_mapping=normalization_mapping)
    validate_bounded_act_config(config)
    return config


def validate_bounded_act_config(config: ACTConfig) -> None:
    normalized = {
        FeatureType(key) if isinstance(key, str) else key: NormalizationMode(value)
        for key, value in config.normalization_mapping.items()
    }
    if normalized.get(FeatureType.ACTION) is not NormalizationMode.IDENTITY:
        raise BoundedACTContractError("bounded ACT actions must use IDENTITY normalization")
    action = config.output_features.get("action")
    if action is None or action.type is not FeatureType.ACTION or action.shape != (8,):
        raise BoundedACTContractError("bounded ACT output feature must be action float[8]")


class BoundedACTPolicyV1(ACTPolicy):
    """Maintained ACT with the only authorized structurally bounded output head."""

    config_class = ACTConfig
    name = BOUNDED_ACTION_HEAD_IDENTITY

    def __init__(self, config: ACTConfig, **kwargs: object) -> None:
        validate_bounded_act_config(config)
        super().__init__(config, **kwargs)
        installed_head = getattr(self.model, "action_head", None)
        if not isinstance(installed_head, nn.Linear):
            raise BoundedACTContractError("installed ACT action head structure changed")
        self.model.action_head = BoundedActionHeadV1.from_linear(installed_head)
        validate_bounded_act_policy(self)

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        push_to_hub: bool = False,
        **kwargs: object,
    ) -> str | None:
        if push_to_hub:
            raise BoundedACTContractError("bounded ACT artifacts are local-only")
        destination = Path(save_directory)
        if destination.exists() and (
            not destination.is_dir()
            or destination.is_symlink()
            or destination.is_junction()
            or any(destination.iterdir())
        ):
            raise BoundedACTContractError("bounded ACT save destination must be new or empty")
        result = super().save_pretrained(
            destination,
            push_to_hub=False,
            **kwargs,
        )
        payload = {
            "bounded_action_head": bounded_action_head_manifest(),
            "native_action_bounds": native_action_bounds_manifest(),
        }
        try:
            with (destination / BOUNDED_ACTION_HEAD_CONFIG_FILE).open(
                "x", encoding="utf-8", newline="\n"
            ) as stream:
                json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
                stream.write("\n")
        except FileExistsError as error:
            raise BoundedACTContractError("refusing to overwrite bounded ACT sidecar") from error
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
    ) -> BoundedACTPolicyV1:
        if not local_files_only or not strict:
            raise BoundedACTContractError(
                "bounded ACT reload requires local_files_only=True and strict=True"
            )
        root = Path(pretrained_name_or_path)
        sidecar = root / BOUNDED_ACTION_HEAD_CONFIG_FILE
        if (
            not root.is_dir()
            or root.is_symlink()
            or root.is_junction()
            or sidecar.is_symlink()
            or sidecar.is_junction()
        ):
            raise BoundedACTContractError("bounded ACT reload root is missing or linked")
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise BoundedACTContractError("bounded ACT sidecar is missing or invalid") from error
        expected = {
            "bounded_action_head": bounded_action_head_manifest(),
            "native_action_bounds": native_action_bounds_manifest(),
        }
        if payload != expected:
            raise BoundedACTContractError("bounded ACT sidecar identity changed")
        effective_config = config
        if effective_config is None:
            loaded = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=root,
                local_files_only=True,
            )
            if not isinstance(loaded, ACTConfig):
                raise BoundedACTContractError("saved ACT config is malformed")
            effective_config = loaded
        policy = super().from_pretrained(
            root,
            config=effective_config,
            local_files_only=True,
            strict=True,
            **kwargs,
        )
        if not isinstance(policy, cls):
            raise BoundedACTContractError("reload returned the wrong policy type")
        validate_bounded_act_policy(policy)
        return policy


def validate_bounded_act_policy(policy: BoundedACTPolicyV1) -> None:
    if not isinstance(policy, BoundedACTPolicyV1):
        raise TypeError("policy must be BoundedACTPolicyV1")
    validate_bounded_act_config(policy.config)
    head = getattr(policy.model, "action_head", None)
    if (
        not isinstance(head, BoundedActionHeadV1)
        or head.linear.in_features != policy.config.dim_model
        or head.linear.out_features != 8
    ):
        raise BoundedACTContractError("bounded ACT head structure changed")
    expected_lower = torch.from_numpy(PANDA_ACTION_LOW)
    expected_upper = torch.from_numpy(PANDA_ACTION_HIGH)
    if not torch.equal(head.lower.detach().cpu(), expected_lower) or not torch.equal(
        head.upper.detach().cpu(), expected_upper
    ):
        raise BoundedACTContractError("bounded ACT head buffers changed")


def model_identity_with_bounded_head(
    base_model_config: Mapping[str, object],
) -> dict[str, object]:
    """Bind the repair and physical representation into the run identity."""

    return {
        **dict(base_model_config),
        "action_head": bounded_action_head_manifest(),
        "native_action_bounds": native_action_bounds_manifest(),
    }


def identity_uses_bounded_action_head(model_config: Mapping[str, object]) -> bool:
    value = model_config.get("action_head")
    return (
        isinstance(value, Mapping)
        and value.get("identity") == BOUNDED_ACTION_HEAD_IDENTITY
        and dict(value) == bounded_action_head_manifest()
    )


def independent_bounded_action_audit(
    *,
    sample_count: int = 100_000,
    chunk_size: int = 16,
    seed: int = 20260725,
) -> dict[str, object]:
    """Audit the exact pure mapping over at least 100k random action chunks."""

    if sample_count < 100_000 or chunk_size < 1:
        raise ValueError("bounded audit needs at least 100,000 nonempty chunks")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    lower = torch.from_numpy(PANDA_ACTION_LOW)
    upper = torch.from_numpy(PANDA_ACTION_HIGH)
    nonfinite = 0
    lower_violations = 0
    upper_violations = 0
    observed_min = torch.full((8,), torch.inf)
    observed_max = torch.full((8,), -torch.inf)
    remaining = sample_count
    while remaining:
        count = min(remaining, 2_048)
        raw = torch.randn(count, chunk_size, 8, generator=generator)
        action = lower + 0.5 * (torch.tanh(raw) + 1.0) * (upper - lower)
        nonfinite += int(torch.count_nonzero(~torch.isfinite(action)))
        lower_violations += int(torch.count_nonzero(action < lower))
        upper_violations += int(torch.count_nonzero(action > upper))
        observed_min = torch.minimum(observed_min, action.amin(dim=(0, 1)))
        observed_max = torch.maximum(observed_max, action.amax(dim=(0, 1)))
        remaining -= count
    semantic = {
        "schema_version": "langmani-v2-phase2c-a1-100k-action-audit-v0",
        "mapping": BOUNDED_ACTION_HEAD_IDENTITY,
        "seed": seed,
        "raw_distribution": "torch.randn float32",
        "sampled_chunks": sample_count,
        "chunk_size": chunk_size,
        "sampled_actions": sample_count * chunk_size,
        "nonfinite_actions": nonfinite,
        "lower_bound_violations": lower_violations,
        "upper_bound_violations": upper_violations,
        "clipping_events": 0,
        "projection_events": 0,
        "observed_minimum": [float(value) for value in observed_min],
        "observed_maximum": [float(value) for value in observed_max],
        "native_action_bounds": native_action_bounds_manifest(),
        "passed": nonfinite == lower_violations == upper_violations == 0,
    }
    return {
        **semantic,
        "fingerprint": f"sha256:{sha256_hex(semantic)}",
    }


__all__ = [
    "ACTION_BOUNDS_SCHEMA",
    "BOUNDED_ACTION_HEAD_CONFIG_FILE",
    "BOUNDED_ACTION_HEAD_IDENTITY",
    "BoundedACTContractError",
    "BoundedACTPolicyV1",
    "BoundedActionHeadV1",
    "bounded_action_head_manifest",
    "build_bounded_act_config",
    "identity_uses_bounded_action_head",
    "independent_bounded_action_audit",
    "model_identity_with_bounded_head",
    "native_action_bounds_manifest",
    "validate_bounded_act_config",
    "validate_bounded_act_policy",
]
