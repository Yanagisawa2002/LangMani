"""Frozen contracts for the Phase 2C-C Pick action-formulation experiment.

This module defines the sole authorized alternative to the failed absolute
Pick SmolVLA action path.  A model predicts a bounded residual in normalized
joint coordinates.  The residual is composed with the current public Panda
state and decoded to the unchanged native ``pd_joint_pos`` action contract.

The current observation at a policy query anchors the complete predicted
action chunk.  Future state, object pose, goal pose, and task diagnostics are
never inputs to the transform or policy.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Self

import numpy as np
import torch

from langmani.datasets.identity import sha256_hex
from langmani.datasets.policy_state import PANDA_POLICY_STATE_COMPONENTS
from langmani.v2.phase2c_a import ACTION_DIMENSION, PACKAGE_FINGERPRINT, STATE_DIMENSION
from langmani.v2.phase2c_b import ACTION_LOWER, ACTION_UPPER, CHUNK_SIZE, SmolVLATrainingConfig

SOURCE_BRANCH: Final = "codex/langmani-v2-phase2c-b-smolvla"
SOURCE_COMMIT: Final = "15daf852238edac39a65f6837340dfc5c9817ed3"
TARGET_BRANCH: Final = "codex/langmani-v2-phase2c-c-action-formulation"

RELATIVE_ACTION_TRANSFORM_ID: Final = "pick_state_relative_bounded_residual_v1"
RELATIVE_ACTION_TRANSFORM_SCHEMA: Final = (
    "langmani-v2-phase2c-c-pick-state-relative-bounded-residual-v1"
)
RELATIVE_ACTION_TRANSFORM_FILE: Final = "pick_state_relative_bounded_residual_v1.json"

# The safe inverse-tanh margin is one float32 half-ULP at 1.0.  The resulting
# full accepted-Pick reconstruction error is required to remain <= 1e-6.
SAFE_ATANH_EPSILON: Final = 2.0**-24
FLOAT32_RECONSTRUCTION_ATOL: Final = 1e-6
RESIDUAL_SCALE_MARGIN: Final = 1.01

# These two float32 values are the single pre-registered scale choice.  They
# are 1.01 times the maximum train-only, query-anchored chunk safe-logit
# displacement, using one common arm scale and one gripper scale.  Static
# preparation must recompute and exactly verify them before any optimizer is
# created.
ARM_RESIDUAL_SCALE: Final = 0.6531111598014832
GRIPPER_RESIDUAL_SCALE: Final = 16.947166442871094

# ManiSkill Panda's normalized mimic command maps [-1, 1] to the physical
# finger-drive target range [-0.01, 0.04] metres.  PandaPolicyStateV0 records
# both measured finger joint positions.  Their arithmetic mean is the
# deterministic measured/current mimic reference.
GRIPPER_PHYSICAL_LOWER: Final = -0.01
GRIPPER_PHYSICAL_UPPER: Final = 0.04
STATIC_RANDOM_CHUNKS: Final = 100_000
STATIC_RANDOM_SEED: Final = 2_026_072_6


class Phase2CCContractError(RuntimeError):
    """Raised when the relative formulation crosses its frozen boundary."""


def canonical_fingerprint(value: Mapping[str, object]) -> str:
    return f"sha256:{sha256_hex(dict(value))}"


def _float32(value: float) -> float:
    return float(np.float32(value))


@dataclass(frozen=True, slots=True)
class StateRelativeBoundedActionV0:
    """Query-anchored bounded residual transform for Panda ``pd_joint_pos``.

    ``encode`` maps an unchanged physical action target to the latent residual
    supervised by SmolVLA.  ``decode`` composes any model output with the
    current public state and always returns an action inside the native bounds.
    """

    arm_scale: float = ARM_RESIDUAL_SCALE
    gripper_scale: float = GRIPPER_RESIDUAL_SCALE
    epsilon: float = SAFE_ATANH_EPSILON
    identity: str = RELATIVE_ACTION_TRANSFORM_ID
    schema_version: str = RELATIVE_ACTION_TRANSFORM_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema_version != RELATIVE_ACTION_TRANSFORM_SCHEMA
            or self.identity != RELATIVE_ACTION_TRANSFORM_ID
            or self.arm_scale != ARM_RESIDUAL_SCALE
            or self.gripper_scale != GRIPPER_RESIDUAL_SCALE
            or self.epsilon != SAFE_ATANH_EPSILON
            or not math.isfinite(self.arm_scale)
            or not math.isfinite(self.gripper_scale)
            or self.arm_scale <= 0.0
            or self.gripper_scale <= 0.0
        ):
            raise Phase2CCContractError("relative action transform configuration changed")

    def _bounds(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lower = torch.as_tensor(ACTION_LOWER, dtype=torch.float32, device=reference.device)
        upper = torch.as_tensor(ACTION_UPPER, dtype=torch.float32, device=reference.device)
        return lower, upper

    def _scales(self, reference: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(
            [self.arm_scale] * 7 + [self.gripper_scale],
            dtype=torch.float32,
            device=reference.device,
        )

    def current_action_reference(self, state: torch.Tensor) -> torch.Tensor:
        """Convert public Panda state to the current native action reference."""

        if (
            not isinstance(state, torch.Tensor)
            or not state.is_floating_point()
            or state.shape[-1] != STATE_DIMENSION
            or not bool(torch.isfinite(state).all())
        ):
            raise Phase2CCContractError("current state must be finite floating [...,9]")
        source = state.to(dtype=torch.float32)
        measured_finger_mean = source[..., 7:9].mean(dim=-1)
        gripper_reference = (
            2.0
            * (measured_finger_mean - GRIPPER_PHYSICAL_LOWER)
            / (GRIPPER_PHYSICAL_UPPER - GRIPPER_PHYSICAL_LOWER)
            - 1.0
        )
        reference = torch.cat((source[..., :7], gripper_reference.unsqueeze(-1)), dim=-1)
        lower, upper = self._bounds(reference)
        if bool(torch.any(reference < lower)) or bool(torch.any(reference > upper)):
            raise Phase2CCContractError(
                "public state cannot be mapped unambiguously inside native action bounds"
            )
        return reference

    def _broadcast_reference(
        self,
        state: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        reference = self.current_action_reference(state)
        while reference.ndim < target.ndim:
            reference = reference.unsqueeze(-2)
        try:
            shape = torch.broadcast_shapes(reference.shape, target.shape)
        except RuntimeError as error:
            raise Phase2CCContractError(
                "state/action leading dimensions cannot be broadcast"
            ) from error
        if shape != target.shape:
            raise Phase2CCContractError("state reference does not align with the action target")
        return reference.expand_as(target)

    def physical_residual(
        self,
        physical_target: torch.Tensor,
        current_state: torch.Tensor,
    ) -> torch.Tensor:
        """Return the metadata/action-only derived physical delta target."""

        physical = self._validate_physical(physical_target)
        reference = self._broadcast_reference(current_state, physical)
        return physical - reference

    def reconstruct_from_physical_residual(
        self,
        residual: torch.Tensor,
        current_state: torch.Tensor,
    ) -> torch.Tensor:
        """Inverse the derived physical-delta view without modifying bytes."""

        if (
            not isinstance(residual, torch.Tensor)
            or not residual.is_floating_point()
            or residual.shape[-1] != ACTION_DIMENSION
            or not bool(torch.isfinite(residual).all())
        ):
            raise Phase2CCContractError("physical residual must be finite floating [...,8]")
        source = residual.to(dtype=torch.float32)
        reference = self._broadcast_reference(current_state, source)
        return self._validate_physical(reference + source)

    def _validate_physical(self, physical: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(physical, torch.Tensor)
            or not physical.is_floating_point()
            or physical.shape[-1] != ACTION_DIMENSION
            or not bool(torch.isfinite(physical).all())
        ):
            raise Phase2CCContractError("physical action must be finite floating [...,8]")
        source = physical.to(dtype=torch.float32)
        lower, upper = self._bounds(source)
        if bool(torch.any(source < lower)) or bool(torch.any(source > upper)):
            raise Phase2CCContractError("physical action lies outside native Panda bounds")
        return source

    def _safe_logit(self, physical: torch.Tensor) -> torch.Tensor:
        lower, upper = self._bounds(physical)
        normalized = 2.0 * (physical - lower) / (upper - lower) - 1.0
        safe = normalized * (1.0 - self.epsilon)
        result = torch.atanh(safe)
        if not bool(torch.isfinite(result).all()):
            raise Phase2CCContractError("safe inverse-tanh produced a non-finite value")
        return result

    def encode(
        self,
        physical_target: torch.Tensor,
        current_state: torch.Tensor,
    ) -> torch.Tensor:
        """Encode physical targets as finite bounded-residual latent targets."""

        physical = self._validate_physical(physical_target)
        reference = self._broadcast_reference(current_state, physical)
        displacement = self._safe_logit(physical) - self._safe_logit(reference)
        ratio = displacement / self._scales(displacement)
        if bool(torch.any(torch.abs(ratio) >= 1.0)):
            raise Phase2CCContractError(
                "physical target exceeds the frozen train-derived residual scale"
            )
        latent = torch.atanh(ratio)
        if not bool(torch.isfinite(latent).all()):
            raise Phase2CCContractError("relative action encoder produced a non-finite target")
        return latent

    def decode(
        self,
        latent: torch.Tensor,
        current_state: torch.Tensor,
    ) -> torch.Tensor:
        """Decode any non-NaN latent to a structurally bounded physical action."""

        if (
            not isinstance(latent, torch.Tensor)
            or not latent.is_floating_point()
            or latent.shape[-1] != ACTION_DIMENSION
            or bool(torch.isnan(latent).any())
        ):
            raise Phase2CCContractError("relative latent must be floating [...,8] without NaN")
        source = latent.to(dtype=torch.float32)
        reference = self._broadcast_reference(current_state, source)
        lower, upper = self._bounds(source)
        normalized = torch.tanh(
            self._safe_logit(reference) + self._scales(source) * torch.tanh(source)
        )
        physical = torch.lerp(lower, upper, 0.5 * (normalized + 1.0))
        if (
            not bool(torch.isfinite(physical).all())
            or bool(torch.any(physical < lower))
            or bool(torch.any(physical > upper))
        ):
            raise Phase2CCContractError("bounded relative decoder invariant failed")
        return physical

    def semantic_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "identity": self.identity,
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "state_schema": "PandaPolicyStateV0",
            "state_components": list(PANDA_POLICY_STATE_COMPONENTS),
            "action_schema": "PandaJointPositionActionV0",
            "action_dimension": ACTION_DIMENSION,
            "native_lower": list(ACTION_LOWER),
            "native_upper": list(ACTION_UPPER),
            "gripper_reference": {
                "measured_state_fields": [
                    "panda_finger_joint1",
                    "panda_finger_joint2",
                ],
                "aggregation": "arithmetic_mean",
                "physical_lower_metres": GRIPPER_PHYSICAL_LOWER,
                "physical_upper_metres": GRIPPER_PHYSICAL_UPPER,
                "normalized_command_lower": -1.0,
                "normalized_command_upper": 1.0,
            },
            "epsilon": self.epsilon,
            "arm_residual_scale": self.arm_scale,
            "gripper_residual_scale": self.gripper_scale,
            "scale_margin_over_train_maximum": RESIDUAL_SCALE_MARGIN,
            "chunk_anchor": "current_observation_state_at_policy_query",
            "future_state_used": False,
            "object_or_goal_state_used": False,
            "encoder": (
                "atanh((safe_logit(physical_target)-safe_logit(current_reference))/residual_scale)"
            ),
            "decoder": (
                "native_affine(tanh(safe_logit(current_reference)"
                "+residual_scale*tanh(model_output)))"
            ),
            "physical_delta_view": "physical_target-current_action_reference",
            "post_hoc_clipping": False,
            "projection": False,
            "replacement_action": False,
            "reconstruction_atol": FLOAT32_RECONSTRUCTION_ATOL,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.semantic_dict())

    def save(self, directory: str | Path) -> Path:
        target = Path(directory) / RELATIVE_ACTION_TRANSFORM_FILE
        payload = {**self.semantic_dict(), "fingerprint": self.fingerprint}
        encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_text(encoding="utf-8") != encoded:
                raise Phase2CCContractError("refusing to overwrite a different transform")
        else:
            target.write_text(encoded, encoding="utf-8", newline="\n")
        return target

    @classmethod
    def load(cls, directory: str | Path) -> Self:
        path = Path(directory) / RELATIVE_ACTION_TRANSFORM_FILE
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise Phase2CCContractError(
                f"cannot load relative action transform: {error}"
            ) from error
        if not isinstance(value, dict):
            raise Phase2CCContractError("relative action transform must be one JSON object")
        semantic = dict(value)
        fingerprint = semantic.pop("fingerprint", None)
        instance = cls()
        if fingerprint != canonical_fingerprint(semantic) or semantic != instance.semantic_dict():
            raise Phase2CCContractError("relative action transform identity changed")
        return instance


def derive_frozen_scales(
    physical_targets: torch.Tensor,
    current_states: torch.Tensor,
    *,
    train_query_count: int | None = None,
) -> dict[str, object]:
    """Recompute the sole scale choice from valid query-anchored train targets."""

    transform = StateRelativeBoundedActionV0()
    physical = transform._validate_physical(physical_targets)
    if physical.ndim != 2 or current_states.shape != (physical.shape[0], STATE_DIMENSION):
        raise Phase2CCContractError("train scale inputs must be aligned valid chunk targets")
    query_count = physical.shape[0] if train_query_count is None else train_query_count
    if not 1 <= query_count <= physical.shape[0]:
        raise Phase2CCContractError("train query count is invalid")
    reference = transform._broadcast_reference(current_states, physical)
    displacement = transform._safe_logit(physical) - transform._safe_logit(reference)
    arm_maximum = float(torch.abs(displacement[..., :7]).max())
    gripper_maximum = float(torch.abs(displacement[..., 7]).max())
    derived_arm = _float32(arm_maximum * RESIDUAL_SCALE_MARGIN)
    derived_gripper = _float32(gripper_maximum * RESIDUAL_SCALE_MARGIN)
    semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-c-train-scale-audit-v1",
        "scale_source": "all_valid_query_anchored_train_chunk_targets",
        "chunk_size": CHUNK_SIZE,
        "train_query_count": int(query_count),
        "train_valid_chunk_target_count": int(physical.shape[0]),
        "safe_logit_arm_absolute_maximum": arm_maximum,
        "safe_logit_gripper_absolute_maximum": gripper_maximum,
        "margin": RESIDUAL_SCALE_MARGIN,
        "derived_arm_scale_float32": derived_arm,
        "derived_gripper_scale_float32": derived_gripper,
        "frozen_arm_scale": ARM_RESIDUAL_SCALE,
        "frozen_gripper_scale": GRIPPER_RESIDUAL_SCALE,
        "exact_scale_match": (
            derived_arm == ARM_RESIDUAL_SCALE and derived_gripper == GRIPPER_RESIDUAL_SCALE
        ),
    }
    return {**semantic, "fingerprint": canonical_fingerprint(semantic)}


def build_static_relative_action_audit(
    *,
    transform: StateRelativeBoundedActionV0 | None = None,
    chunk_count: int = STATIC_RANDOM_CHUNKS,
    batch_chunks: int = 1_000,
) -> dict[str, object]:
    """Exercise at least 100k random residual chunks without simulator work."""

    if chunk_count < STATIC_RANDOM_CHUNKS or batch_chunks < 1:
        raise Phase2CCContractError("static relative audit requires at least 100,000 chunks")
    active = transform or StateRelativeBoundedActionV0()
    generator = torch.Generator(device="cpu").manual_seed(STATIC_RANDOM_SEED)
    lower = torch.as_tensor(ACTION_LOWER, dtype=torch.float32)
    upper = torch.as_tensor(ACTION_UPPER, dtype=torch.float32)
    nonfinite = 0
    lower_violations = 0
    upper_violations = 0
    maximum_zero_error = 0.0
    processed = 0
    while processed < chunk_count:
        count = min(batch_chunks, chunk_count - processed)
        current = lower + torch.rand((count, 8), generator=generator) * (upper - lower)
        finger_position = GRIPPER_PHYSICAL_LOWER + 0.5 * (current[:, 7] + 1.0) * (
            GRIPPER_PHYSICAL_UPPER - GRIPPER_PHYSICAL_LOWER
        )
        state = torch.cat((current[:, :7], finger_position[:, None].repeat(1, 2)), dim=1)
        latent = torch.randn((count, CHUNK_SIZE, 8), generator=generator) * 3.0
        decoded = active.decode(latent, state)
        nonfinite += int(torch.count_nonzero(~torch.isfinite(decoded)))
        lower_violations += int(torch.count_nonzero(decoded < lower))
        upper_violations += int(torch.count_nonzero(decoded > upper))
        zero = active.decode(torch.zeros((count, CHUNK_SIZE, 8)), state)
        zero_error = torch.abs(zero - current[:, None, :])
        maximum_zero_error = max(maximum_zero_error, float(zero_error.max()))
        processed += count
    semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-c-static-relative-action-audit-v0",
        "transform_fingerprint": active.fingerprint,
        "random_seed": STATIC_RANDOM_SEED,
        "chunk_count": chunk_count,
        "chunk_size": CHUNK_SIZE,
        "action_count": chunk_count * CHUNK_SIZE,
        "component_count": chunk_count * CHUNK_SIZE * ACTION_DIMENSION,
        "nonfinite_component_count": nonfinite,
        "lower_bound_violation_count": lower_violations,
        "upper_bound_violation_count": upper_violations,
        "clipping_event_count": 0,
        "projection_event_count": 0,
        "replacement_action_count": 0,
        "maximum_zero_residual_identity_error": maximum_zero_error,
        "zero_residual_identity_atol": FLOAT32_RECONSTRUCTION_ATOL,
        "passed": (
            nonfinite == 0
            and lower_violations == 0
            and upper_violations == 0
            and maximum_zero_error <= FLOAT32_RECONSTRUCTION_ATOL
        ),
    }
    return {**semantic, "fingerprint": canonical_fingerprint(semantic)}


def relative_training_config() -> dict[str, object]:
    """Bind Phase 2C-C to the unchanged Phase 2C-B optimization schedule."""

    training = SmolVLATrainingConfig()
    semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-c-relative-training-config-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "task_id": "PickCube-v1",
        "training_config": training.to_dict(),
        "training_config_fingerprint": training.fingerprint,
        "action_formulation": RELATIVE_ACTION_TRANSFORM_ID,
        "action_transform_fingerprint": StateRelativeBoundedActionV0().fingerprint,
        "only_change_from_absolute_baseline": (
            "absolute_action_target_to_query_state_relative_bounded_residual"
        ),
        "training_seed_count": 1,
        "training_seed": training.seed,
        "checkpoint_steps": list(training.checkpoint_steps),
        "total_optimizer_steps": training.total_steps,
        "vla_jepa_authorized": False,
        "shared_smolvla_authorized": False,
        "stack_smolvla_authorized": False,
        "push_smolvla_authorized": False,
    }
    return {**semantic, "fingerprint": canonical_fingerprint(semantic)}


__all__ = [
    "ARM_RESIDUAL_SCALE",
    "FLOAT32_RECONSTRUCTION_ATOL",
    "GRIPPER_PHYSICAL_LOWER",
    "GRIPPER_PHYSICAL_UPPER",
    "GRIPPER_RESIDUAL_SCALE",
    "RELATIVE_ACTION_TRANSFORM_FILE",
    "RELATIVE_ACTION_TRANSFORM_ID",
    "SOURCE_BRANCH",
    "SOURCE_COMMIT",
    "StateRelativeBoundedActionV0",
    "TARGET_BRANCH",
    "Phase2CCContractError",
    "build_static_relative_action_audit",
    "canonical_fingerprint",
    "derive_frozen_scales",
    "relative_training_config",
]
