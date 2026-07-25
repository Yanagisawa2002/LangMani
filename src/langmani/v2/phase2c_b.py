"""Frozen contracts for LangMani 2.0 Phase 2C-B SmolVLA.

The module contains only model-family-independent experiment contracts and the
isolated embodiment action transform.  The transform is intentionally outside
LeRobot's normalization processors:

``physical -> affine[-1, 1] -> safe atanh -> SmolVLA latent``
``SmolVLA latent -> tanh -> affine[native Panda bounds] -> environment``

No clipping, projection, rejection sampling, or replacement action exists in
the decoding path.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, Self

import torch

from langmani.datasets.identity import sha256_hex
from langmani.v2.phase2b5 import PANDA_ACTION_HIGH, PANDA_ACTION_LOW
from langmani.v2.phase2c_a import (
    ACTION_DIMENSION,
    IMAGE_SHAPE_CHW,
    NORMALIZATION_FINGERPRINT,
    NORMALIZATION_IDENTITY,
    PACKAGE_FINGERPRINT,
    PACKAGE_IDENTITY,
    STATE_DIMENSION,
    TASK_IDS,
    TASK_SLUGS,
    TRAIN_EPISODES,
    TRAIN_FRAMES,
    VISUAL_SHIFT_IDENTITY,
)

PHASE2C_B_SCHEMA_VERSION: Final = "langmani-v2-phase2c-b-v0"
SOURCE_BRANCH: Final = "codex/langmani-v2-phase2c-a1-bounded-act"
SOURCE_COMMIT: Final = "4c38190a36ec41349ef39774ee95dc666537eed6"
TARGET_BRANCH: Final = "codex/langmani-v2-phase2c-b-smolvla"
OFFICIAL_LEROBOT_VERSION: Final = "0.6.0"
OFFICIAL_BASE_MODEL: Final = "lerobot/smolvla_base"
OFFICIAL_BASE_REVISION: Final = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
OFFICIAL_BASE_CONFIG_SHA256: Final = (
    "650584b56c104720f7a3c91d1ec6bec9e8de8ac11e60c92ba2fa82d93eda147d"
)
OFFICIAL_BASE_MODEL_SHA256: Final = (
    "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb"
)
OFFICIAL_VLM_MODEL: Final = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
OFFICIAL_VLM_REVISION: Final = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
OFFICIAL_VLM_CONFIG_SHA256: Final = (
    "ea6bc1237e96247f6258de3e202e2e62b93d6f386dc47e7b36b5588bf3a15e17"
)
OFFICIAL_VLM_MODEL_SHA256: Final = (
    "b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3"
)
BOUNDED_ACTION_TRANSFORM_ID: Final = "smolvla_bounded_action_latent_v1"
BOUNDED_ACTION_TRANSFORM_SCHEMA: Final = "langmani-v2-phase2c-b-smolvla-bounded-action-latent-v1"
BOUNDED_ACTION_TRANSFORM_FILE: Final = "smolvla_bounded_action_latent_v1.json"
CHUNK_SIZE: Final = 50
EXECUTION_HORIZONS: Final = (1, 4, 8)
VALIDATION_EPISODES_PER_TASK: Final = 30
FULL_TRAINING_STEPS: Final = 20_000
STATIC_ACTION_AUDIT_CHUNKS: Final = 100_000
ACTION_EPSILON: Final = 1e-6
ACTION_LOWER: Final = tuple(float(value) for value in PANDA_ACTION_LOW)
ACTION_UPPER: Final = tuple(float(value) for value in PANDA_ACTION_HIGH)


class Phase2CBContractError(RuntimeError):
    """Raised when a frozen Phase 2C-B boundary is violated."""


class ModelKind(StrEnum):
    """The four pre-registered SmolVLA models."""

    PICK = "pick_smolvla"
    STACK = "stack_smolvla"
    PUSH = "push_smolvla"
    SHARED = "shared_language_smolvla"

    @property
    def task_id(self) -> str | None:
        return {
            ModelKind.PICK: "PickCube-v1",
            ModelKind.STACK: "StackCube-v1",
            ModelKind.PUSH: "PushCube-v1",
            ModelKind.SHARED: None,
        }[self]

    @property
    def compatible_task_ids(self) -> tuple[str, ...]:
        return TASK_IDS if self is ModelKind.SHARED else (str(self.task_id),)

    @property
    def slug(self) -> str:
        return {
            ModelKind.PICK: "model-p",
            ModelKind.STACK: "model-s",
            ModelKind.PUSH: "model-u",
            ModelKind.SHARED: "model-m",
        }[self]


class Phase2CBResult(StrEnum):
    RESULT_A = "RESULT_A"
    RESULT_B = "RESULT_B"
    RESULT_C = "RESULT_C"
    RESULT_D = "RESULT_D"
    RESULT_E = "RESULT_E"


FAILURE_CATEGORIES: Final = (
    "no_initial_motion",
    "incorrect_approach",
    "failed_grasp",
    "object_drop",
    "contact_without_completion",
    "failed_stack_alignment",
    "stack_collapse",
    "insufficient_push",
    "push_overshoot",
    "wrong_skill_behavior",
    "instruction_ignored",
    "action_oscillation",
    "timeout",
    "invalid_policy_output",
    "simulator_error",
    "other",
)


def canonical_fingerprint(value: Mapping[str, object]) -> str:
    return f"sha256:{sha256_hex(dict(value))}"


def _validate_sha256(value: str, label: str) -> None:
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise Phase2CBContractError(f"{label} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class BoundedActionLatentV1:
    """Deterministic, structurally bounded Panda action representation.

    Exact native endpoints are embedded at finite ``atanh(1-epsilon)`` targets.
    Decoding those targets returns the deterministic nearest interior value;
    the maximum round-trip error is ``epsilon * half_range``.  This keeps the
    binary gripper endpoints finite and distinguishable while every possible
    finite or infinite latent still decodes inside the native bounds.
    """

    lower: tuple[float, ...] = ACTION_LOWER
    upper: tuple[float, ...] = ACTION_UPPER
    epsilon: float = ACTION_EPSILON
    identity: str = BOUNDED_ACTION_TRANSFORM_ID
    schema_version: str = BOUNDED_ACTION_TRANSFORM_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema_version != BOUNDED_ACTION_TRANSFORM_SCHEMA
            or self.identity != BOUNDED_ACTION_TRANSFORM_ID
            or len(self.lower) != ACTION_DIMENSION
            or len(self.upper) != ACTION_DIMENSION
            or any(not math.isfinite(value) for value in (*self.lower, *self.upper))
            or any(low >= high for low, high in zip(self.lower, self.upper, strict=True))
            or not math.isfinite(self.epsilon)
            or not 0.0 < self.epsilon < 1e-3
        ):
            raise Phase2CBContractError("bounded action transform configuration is invalid")
        if self.lower != ACTION_LOWER or self.upper != ACTION_UPPER:
            raise Phase2CBContractError("bounded action transform changed native Panda bounds")

    def _bounds(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lower = torch.as_tensor(self.lower, dtype=torch.float32, device=reference.device)
        upper = torch.as_tensor(self.upper, dtype=torch.float32, device=reference.device)
        return lower, upper

    def encode(self, physical: torch.Tensor) -> torch.Tensor:
        """Map a finite native action to a finite SmolVLA latent target."""

        if (
            not isinstance(physical, torch.Tensor)
            or not physical.is_floating_point()
            or physical.shape[-1] != ACTION_DIMENSION
            or not bool(torch.isfinite(physical).all())
        ):
            raise Phase2CBContractError("physical actions must be finite floating [...,8]")
        source = physical.to(dtype=torch.float32)
        lower, upper = self._bounds(source)
        if bool(torch.any(source < lower)) or bool(torch.any(source > upper)):
            raise Phase2CBContractError("physical action lies outside native Panda bounds")
        normalized = 2.0 * (source - lower) / (upper - lower) - 1.0
        safe = normalized * (1.0 - self.epsilon)
        latent = torch.atanh(safe)
        if not bool(torch.isfinite(latent).all()):
            raise Phase2CBContractError("bounded action encoder produced a non-finite latent")
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Map any floating latent to a structurally bounded physical action."""

        if (
            not isinstance(latent, torch.Tensor)
            or not latent.is_floating_point()
            or latent.shape[-1] != ACTION_DIMENSION
            or bool(torch.isnan(latent).any())
        ):
            raise Phase2CBContractError("latent actions must be floating [...,8] without NaN")
        source = latent.to(dtype=torch.float32)
        lower, upper = self._bounds(source)
        normalized = torch.tanh(source)
        physical = torch.lerp(lower, upper, 0.5 * (normalized + 1.0))
        if (
            not bool(torch.isfinite(physical).all())
            or bool(torch.any(physical < lower))
            or bool(torch.any(physical > upper))
        ):
            raise Phase2CBContractError("bounded action decoder invariant failed")
        return physical

    def semantic_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "identity": self.identity,
            "action_dimension": ACTION_DIMENSION,
            "lower": list(self.lower),
            "upper": list(self.upper),
            "epsilon": self.epsilon,
            "encoder": "atanh((1-epsilon)*(2*(physical-lower)/(upper-lower)-1))",
            "decoder": "lower+0.5*(tanh(latent)+1)*(upper-lower)",
            "endpoint_treatment": "finite_nearest_interior_v0",
            "post_hoc_clipping": False,
            "projection": False,
            "replacement_action": False,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.semantic_dict())

    def save(self, directory: str | Path) -> Path:
        target = Path(directory) / BOUNDED_ACTION_TRANSFORM_FILE
        payload = {**self.semantic_dict(), "fingerprint": self.fingerprint}
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if target.exists():
            if target.read_text(encoding="utf-8") != encoded:
                raise Phase2CBContractError("refusing to overwrite a different action transform")
        else:
            target.write_text(encoded, encoding="utf-8", newline="\n")
        return target

    @classmethod
    def load(cls, directory: str | Path) -> Self:
        path = Path(directory) / BOUNDED_ACTION_TRANSFORM_FILE
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise Phase2CBContractError(f"cannot load bounded action transform: {error}") from error
        if not isinstance(value, dict):
            raise Phase2CBContractError("bounded action transform must be one JSON object")
        fingerprint = value.pop("fingerprint", None)
        allowed = {
            "schema_version",
            "identity",
            "lower",
            "upper",
            "epsilon",
            "action_dimension",
            "encoder",
            "decoder",
            "endpoint_treatment",
            "post_hoc_clipping",
            "projection",
            "replacement_action",
        }
        if set(value) != allowed:
            raise Phase2CBContractError("bounded action transform fields changed")
        semantic = dict(value)
        if fingerprint != canonical_fingerprint(semantic):
            raise Phase2CBContractError("bounded action transform fingerprint changed")
        instance = cls(
            lower=tuple(float(item) for item in value["lower"]),
            upper=tuple(float(item) for item in value["upper"]),
            epsilon=float(value["epsilon"]),
            identity=str(value["identity"]),
            schema_version=str(value["schema_version"]),
        )
        if (
            value["action_dimension"] != ACTION_DIMENSION
            or value["encoder"] != instance.semantic_dict()["encoder"]
            or value["decoder"] != instance.semantic_dict()["decoder"]
            or value["endpoint_treatment"] != "finite_nearest_interior_v0"
            or value["post_hoc_clipping"] is not False
            or value["projection"] is not False
            or value["replacement_action"] is not False
        ):
            raise Phase2CBContractError("bounded action transform semantics changed")
        return instance


@dataclass(frozen=True, slots=True)
class SmolVLATrainingConfig:
    """The one pre-registered primary training configuration."""

    chunk_size: int = CHUNK_SIZE
    n_action_steps: int = CHUNK_SIZE
    execution_horizon_candidates: tuple[int, ...] = EXECUTION_HORIZONS
    image_shape_chw: tuple[int, int, int] = IMAGE_SHAPE_CHW
    state_dimension: int = STATE_DIMENSION
    action_dimension: int = ACTION_DIMENSION
    batch_size: int = 4
    gradient_accumulation: int = 4
    precision: str = "bfloat16"
    optimizer: str = "AdamW"
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    optimizer_epsilon: float = 1e-8
    weight_decay: float = 1e-10
    gradient_clip_norm: float = 10.0
    scheduler: str = "cosine_decay_with_warmup"
    warmup_steps: int = 1_000
    decay_steps: int = 30_000
    decay_learning_rate: float = 2.5e-6
    augmentation: str = "none"
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_projection: bool = True
    seed: int = 0
    checkpoint_steps: tuple[int, ...] = (5_000, 10_000, 20_000)
    validation_steps: tuple[int, ...] = (5_000, 10_000, 20_000)
    total_steps: int = FULL_TRAINING_STEPS
    num_workers: int = 4
    num_inference_steps: int = 10
    tokenizer_max_length: int = 48
    resize_images_with_padding: tuple[int, int] = (512, 512)

    def __post_init__(self) -> None:
        if (
            self.chunk_size != CHUNK_SIZE
            or self.n_action_steps != CHUNK_SIZE
            or self.execution_horizon_candidates != EXECUTION_HORIZONS
            or self.image_shape_chw != IMAGE_SHAPE_CHW
            or self.state_dimension != STATE_DIMENSION
            or self.action_dimension != ACTION_DIMENSION
            or self.batch_size < 1
            or self.gradient_accumulation < 1
            or self.precision != "bfloat16"
            or self.optimizer != "AdamW"
            or self.learning_rate != 1e-4
            or self.betas != (0.9, 0.95)
            or self.optimizer_epsilon != 1e-8
            or self.weight_decay != 1e-10
            or self.gradient_clip_norm != 10.0
            or self.scheduler != "cosine_decay_with_warmup"
            or self.warmup_steps != 1_000
            or self.decay_steps != 30_000
            or self.decay_learning_rate != 2.5e-6
            or self.augmentation != "none"
            or not self.freeze_vision_encoder
            or not self.train_expert_only
            or not self.train_state_projection
            or self.seed != 0
            or self.checkpoint_steps != (5_000, 10_000, 20_000)
            or self.validation_steps != self.checkpoint_steps
            or self.total_steps != FULL_TRAINING_STEPS
            or self.num_workers < 0
            or self.num_inference_steps != 10
            or self.tokenizer_max_length != 48
            or self.resize_images_with_padding != (512, 512)
        ):
            raise Phase2CBContractError("primary SmolVLA training configuration changed")

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_dict())


def apply_padding_mask(
    losses: torch.Tensor,
    action_is_pad: torch.Tensor,
) -> torch.Tensor:
    """Reference loss reduction matching official SmolVLA 0.6 padding semantics."""

    if (
        not losses.is_floating_point()
        or losses.ndim != 3
        or losses.shape[-1] != ACTION_DIMENSION
        or action_is_pad.dtype is not torch.bool
        or action_is_pad.shape != losses.shape[:2]
        or not bool(torch.isfinite(losses).all())
    ):
        raise Phase2CBContractError("padding audit tensors are malformed")
    valid = (~action_is_pad).unsqueeze(-1)
    denominator = ((~action_is_pad).sum() * losses.shape[-1]).clamp_min(1)
    return (losses * valid).sum() / denominator


def build_static_action_audit(
    *,
    chunk_count: int = STATIC_ACTION_AUDIT_CHUNKS,
    batch_chunks: int = 1_000,
    seed: int = 20_260_725,
) -> dict[str, object]:
    """Exercise at least 100k synthetic latent chunks through the exact decoder."""

    if chunk_count < STATIC_ACTION_AUDIT_CHUNKS or batch_chunks < 1:
        raise Phase2CBContractError("static action audit requires at least 100,000 chunks")
    transform = BoundedActionLatentV1()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    lower = torch.tensor(transform.lower, dtype=torch.float32)
    upper = torch.tensor(transform.upper, dtype=torch.float32)
    nonfinite = lower_violations = upper_violations = 0
    minimum = torch.full((ACTION_DIMENSION,), torch.inf, dtype=torch.float32)
    maximum = torch.full((ACTION_DIMENSION,), -torch.inf, dtype=torch.float32)
    completed = 0
    while completed < chunk_count:
        current = min(batch_chunks, chunk_count - completed)
        latent = torch.randn(
            (current, CHUNK_SIZE, ACTION_DIMENSION),
            generator=generator,
            dtype=torch.float32,
        )
        physical = transform.decode(latent)
        nonfinite += int(torch.count_nonzero(~torch.isfinite(physical)))
        lower_violations += int(torch.count_nonzero(physical < lower))
        upper_violations += int(torch.count_nonzero(physical > upper))
        minimum = torch.minimum(minimum, physical.amin(dim=(0, 1)))
        maximum = torch.maximum(maximum, physical.amax(dim=(0, 1)))
        completed += current
    endpoints = torch.stack((lower, upper))
    endpoint_latent = transform.encode(endpoints)
    endpoint_roundtrip = transform.decode(endpoint_latent)
    endpoint_error = torch.abs(endpoint_roundtrip - endpoints)
    semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-b-static-action-audit-v0",
        "transform_fingerprint": transform.fingerprint,
        "seed": seed,
        "chunk_count": chunk_count,
        "action_count": chunk_count * CHUNK_SIZE,
        "component_count": chunk_count * CHUNK_SIZE * ACTION_DIMENSION,
        "latent_distribution": "torch_standard_normal_float32",
        "nonfinite_physical_component_count": nonfinite,
        "lower_bound_violation_count": lower_violations,
        "upper_bound_violation_count": upper_violations,
        "clipping_event_count": 0,
        "projection_event_count": 0,
        "replacement_event_count": 0,
        "observed_minimum": minimum.tolist(),
        "observed_maximum": maximum.tolist(),
        "endpoint_latent_finite": bool(torch.isfinite(endpoint_latent).all()),
        "endpoint_roundtrip_max_abs_error": float(endpoint_error.max()),
        "gripper_endpoint_roundtrip_errors": endpoint_error[:, -1].tolist(),
        "passed": nonfinite == lower_violations == upper_violations == 0,
    }
    return {**semantic, "fingerprint": canonical_fingerprint(semantic)}


def build_padding_audit() -> dict[str, object]:
    """Verify the exact masked reduction, including an all-padding batch."""

    losses = torch.ones((2, CHUNK_SIZE, ACTION_DIMENSION), dtype=torch.float32)
    padding = torch.zeros((2, CHUNK_SIZE), dtype=torch.bool)
    padding[0, -7:] = True
    losses[0, -7:] = 1_000_000.0
    mixed = apply_padding_mask(losses, padding)
    all_padding = torch.ones((2, CHUNK_SIZE), dtype=torch.bool)
    all_padding_loss = apply_padding_mask(losses, all_padding)
    endpoint_actions = torch.tensor([ACTION_LOWER, ACTION_UPPER], dtype=torch.float32)
    endpoint_latent = BoundedActionLatentV1().encode(endpoint_actions)
    semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-b-padding-audit-v0",
        "chunk_size": CHUNK_SIZE,
        "action_dimension": ACTION_DIMENSION,
        "explicit_action_is_pad": True,
        "padded_timestep_loss_contribution": 0.0,
        "unpadded_reference_loss": float(mixed),
        "all_padding_loss": float(all_padding_loss),
        "all_padding_finite": bool(torch.isfinite(all_padding_loss)),
        "endpoint_latent_finite": bool(torch.isfinite(endpoint_latent).all()),
        "passed": float(mixed) == 1.0 and float(all_padding_loss) == 0.0,
    }
    return {**semantic, "fingerprint": canonical_fingerprint(semantic)}


def build_model_view_manifest(model_kind: ModelKind | str) -> dict[str, object]:
    kind = ModelKind(model_kind)
    task_ids = kind.compatible_task_ids
    episodes = {task_id: TRAIN_EPISODES[task_id] for task_id in task_ids}
    frames = {task_id: TRAIN_FRAMES[task_id] for task_id in task_ids}
    semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-b-model-view-v0",
        "package_identity": PACKAGE_IDENTITY,
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "normalization_identity": NORMALIZATION_IDENTITY,
        "normalization_fingerprint": NORMALIZATION_FINGERPRINT,
        "model_kind": kind.value,
        "task_ids": list(task_ids),
        "episodes_by_task": episodes,
        "frames_by_task": frames,
        "episode_count": sum(episodes.values()),
        "frame_count": sum(frames.values()),
        "language_input": "natural_language_task_field",
        "task_id_model_input": False,
        "uniform_task_sampling": kind is ModelKind.SHARED,
        "uniform_task_probabilities": (
            {task_id: 1.0 / len(TASK_IDS) for task_id in TASK_IDS}
            if kind is ModelKind.SHARED
            else None
        ),
        "excluded_episodes_absent": True,
        "privileged_policy_fields": [],
    }
    return {**semantic, "fingerprint": canonical_fingerprint(semantic)}


def build_evaluation_schedule(
    split_manifest: Mapping[str, object],
    source_inventory: Mapping[str, object],
) -> dict[str, object]:
    """Freeze all validation/final reset and language identities before results."""

    assignments = split_manifest.get("assignments")
    sources = source_inventory.get("episodes")
    if not isinstance(assignments, list) or not isinstance(sources, list):
        raise Phase2CBContractError("evaluation schedule sources are malformed")
    reset_by_source: dict[tuple[str, int], Mapping[str, object]] = {}
    for raw in sources:
        if not isinstance(raw, Mapping):
            raise Phase2CBContractError("source inventory episode is malformed")
        task_id = raw.get("task_id")
        source_episode_id = raw.get("source_episode_id")
        if task_id in TASK_IDS and isinstance(source_episode_id, int):
            reset_by_source[(str(task_id), source_episode_id)] = raw
    counts = {
        "validation": VALIDATION_EPISODES_PER_TASK,
        "test_unseen_reset": 50,
        "test_unseen_task_language": 50,
        "test_visual_shift": 50,
    }
    schedules: dict[str, dict[str, list[dict[str, object]]]] = {}
    for split, maximum in counts.items():
        schedules[split] = {}
        for task_id in TASK_IDS:
            selected = [
                raw
                for raw in assignments
                if isinstance(raw, Mapping)
                and raw.get("task_id") == task_id
                and raw.get("primary_split") == split
            ]
            selected.sort(key=lambda value: int(value["source_episode_id"]))
            selected = selected[:maximum]
            if len(selected) != maximum:
                raise Phase2CBContractError(f"{task_id}/{split} lacks {maximum} identities")
            rows: list[dict[str, object]] = []
            for ordinal, assignment in enumerate(selected):
                source_episode_id = assignment.get("source_episode_id")
                instruction = assignment.get("instruction")
                template_id = assignment.get("instruction_template_id")
                if (
                    not isinstance(source_episode_id, int)
                    or not isinstance(instruction, str)
                    or not instruction.strip()
                    or not isinstance(template_id, str)
                    or not template_id
                ):
                    raise Phase2CBContractError("scheduled language identity is malformed")
                source = reset_by_source.get((task_id, source_episode_id))
                if source is None:
                    raise Phase2CBContractError("scheduled episode lacks source reset metadata")
                rows.append(
                    {
                        "evaluation_id": (f"phase2c-b:{split}:{TASK_SLUGS[task_id]}:{ordinal:03d}"),
                        "task_id": task_id,
                        "split": split,
                        "source_episode_id": source_episode_id,
                        "derived_episode_identity": assignment.get("derived_episode_identity"),
                        "reset_identity": assignment.get("reset_identity"),
                        "reset_kwargs": source.get("reset_kwargs"),
                        "language_instruction": instruction,
                        "instruction_template_id": template_id,
                        "language_bank": assignment.get("language_bank"),
                        "visual_transform": (
                            VISUAL_SHIFT_IDENTITY if split == "test_visual_shift" else None
                        ),
                    }
                )
            schedules[split][task_id] = rows
    semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-b-evaluation-schedule-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "source_split_manifest_fingerprint": split_manifest.get("fingerprint"),
        "source_inventory_fingerprint": source_inventory.get("fingerprint"),
        "validation_role": "checkpoint_and_execution_horizon_selection_only",
        "final_splits": [
            "test_unseen_reset",
            "test_unseen_task_language",
            "test_visual_shift",
        ],
        "final_settings_mutable_after_results": False,
        "execution_horizons": list(EXECUTION_HORIZONS),
        "schedules": schedules,
    }
    return {**semantic, "fingerprint": canonical_fingerprint(semantic)}


def build_language_intervention_manifest(
    evaluation_schedule: Mapping[str, object],
    *,
    seed: int = 20_260_725,
    episodes_per_task: int = 20,
) -> dict[str, object]:
    """Freeze correct/wrong/blank/shuffled instructions before model results."""

    schedules = evaluation_schedule.get("schedules")
    validation = schedules.get("validation") if isinstance(schedules, Mapping) else None
    if (
        evaluation_schedule.get("schema_version") != "langmani-v2-phase2c-b-evaluation-schedule-v0"
        or evaluation_schedule.get("package_fingerprint") != PACKAGE_FINGERPRINT
        or not isinstance(validation, Mapping)
        or episodes_per_task < 20
    ):
        raise Phase2CBContractError("language intervention schedule is malformed")
    base: list[dict[str, object]] = []
    for task_id in TASK_IDS:
        rows = validation.get(task_id)
        if not isinstance(rows, list) or len(rows) < episodes_per_task:
            raise Phase2CBContractError("language intervention lacks validation rows")
        for ordinal, raw in enumerate(rows[:episodes_per_task]):
            if (
                not isinstance(raw, Mapping)
                or not isinstance(raw.get("evaluation_id"), str)
                or not isinstance(raw.get("language_instruction"), str)
            ):
                raise Phase2CBContractError("language intervention row is malformed")
            base.append(
                {
                    "task_id": task_id,
                    "ordinal": ordinal,
                    "evaluation_id": raw["evaluation_id"],
                    "correct_instruction": raw["language_instruction"],
                }
            )
    import random

    rng = random.Random(seed)
    shuffled_sources = list(range(len(base)))
    while all(
        base[index]["task_id"] == base[source]["task_id"]
        for index, source in enumerate(shuffled_sources)
    ):
        rng.shuffle(shuffled_sources)
    records: list[dict[str, object]] = []
    for index, row in enumerate(base):
        task_id = str(row["task_id"])
        raw_ordinal = row["ordinal"]
        if not isinstance(raw_ordinal, int):
            raise Phase2CBContractError("language intervention ordinal is malformed")
        ordinal = raw_ordinal
        wrong_task = TASK_IDS[(TASK_IDS.index(task_id) + 1) % len(TASK_IDS)]
        wrong_row = base[TASK_IDS.index(wrong_task) * episodes_per_task + ordinal]
        shuffled_row = base[shuffled_sources[index]]
        records.append(
            {
                **row,
                "wrong_skill_source_task_id": wrong_task,
                "wrong_skill_instruction": wrong_row["correct_instruction"],
                "blank_instruction": "",
                "shuffled_source_task_id": shuffled_row["task_id"],
                "shuffled_source_evaluation_id": shuffled_row["evaluation_id"],
                "shuffled_instruction": shuffled_row["correct_instruction"],
            }
        )
    semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-b-language-intervention-lock-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "evaluation_schedule_fingerprint": evaluation_schedule.get("fingerprint"),
        "split": "validation",
        "seed": seed,
        "episodes_per_task": episodes_per_task,
        "conditions": ["correct", "wrong_skill", "blank", "shuffled"],
        "records": records,
        "settings_mutable_after_results": False,
    }
    return {**semantic, "fingerprint": canonical_fingerprint(semantic)}


def validate_model_batch(
    batch: Mapping[str, object],
    *,
    shared: bool,
) -> None:
    """Reject privileged or task-ID model inputs before preprocessing."""

    required = {
        "observation.images.base_camera",
        "observation.state",
        "action",
        "action_is_pad",
        "task",
    }
    if set(batch) != required:
        raise Phase2CBContractError("SmolVLA model batch must contain the exact policy allowlist")
    task = batch["task"]
    if not isinstance(task, Sequence) or isinstance(task, (str, bytes)):
        raise Phase2CBContractError("batched natural-language task field is malformed")
    if not task or any(not isinstance(item, str) or not item.strip() for item in task):
        raise Phase2CBContractError("SmolVLA requires non-empty natural-language instructions")
    if shared and len(set(task)) < 1:
        raise Phase2CBContractError("shared SmolVLA language batch is empty")


def validate_prerequisite_completion(
    document: Mapping[str, object],
    *,
    schema_version: str,
) -> dict[str, object]:
    """Validate one immutable, self-fingerprinted Phase 2C-B prerequisite."""

    semantic = dict(document)
    fingerprint = semantic.pop("fingerprint", None)
    if (
        semantic.get("schema_version") != schema_version
        or semantic.get("passed") is not True
        or fingerprint != canonical_fingerprint(semantic)
    ):
        raise Phase2CBContractError("Phase 2C-B prerequisite completion identity is invalid")
    return dict(document)


def classify_pick_gate(
    *,
    success_count: int,
    episode_count: int,
    invalid_action_episode_count: int,
    simulator_error_episode_count: int,
    repair_already_used: bool,
) -> dict[str, object]:
    if (
        episode_count != VALIDATION_EPISODES_PER_TASK
        or not 0 <= success_count <= episode_count
        or invalid_action_episode_count < 0
        or simulator_error_episode_count < 0
    ):
        raise Phase2CBContractError("Pick competence evidence is malformed")
    passed = (
        success_count >= 3
        and invalid_action_episode_count == 0
        and simulator_error_episode_count == 0
    )
    result_d = success_count == 0 and repair_already_used
    semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-b-pick-competence-gate-v0",
        "success_count": success_count,
        "episode_count": episode_count,
        "minimum_success_count": 3,
        "invalid_action_episode_count": invalid_action_episode_count,
        "simulator_error_episode_count": simulator_error_episode_count,
        "repair_already_used": repair_already_used,
        "passed": passed,
        "bounded_repair_permitted": (
            success_count == 0
            and not repair_already_used
            and invalid_action_episode_count == 0
            and simulator_error_episode_count == 0
        ),
        "result_d_stop": result_d,
        "other_full_models_authorized": passed,
    }
    return {**semantic, "fingerprint": canonical_fingerprint(semantic)}


__all__ = [
    "ACTION_EPSILON",
    "ACTION_LOWER",
    "ACTION_UPPER",
    "BOUNDED_ACTION_TRANSFORM_FILE",
    "BOUNDED_ACTION_TRANSFORM_ID",
    "BOUNDED_ACTION_TRANSFORM_SCHEMA",
    "BoundedActionLatentV1",
    "CHUNK_SIZE",
    "EXECUTION_HORIZONS",
    "FAILURE_CATEGORIES",
    "FULL_TRAINING_STEPS",
    "ModelKind",
    "OFFICIAL_BASE_CONFIG_SHA256",
    "OFFICIAL_BASE_MODEL",
    "OFFICIAL_BASE_MODEL_SHA256",
    "OFFICIAL_BASE_REVISION",
    "OFFICIAL_LEROBOT_VERSION",
    "OFFICIAL_VLM_CONFIG_SHA256",
    "OFFICIAL_VLM_MODEL",
    "OFFICIAL_VLM_MODEL_SHA256",
    "OFFICIAL_VLM_REVISION",
    "PHASE2C_B_SCHEMA_VERSION",
    "Phase2CBContractError",
    "Phase2CBResult",
    "SOURCE_BRANCH",
    "SOURCE_COMMIT",
    "STATIC_ACTION_AUDIT_CHUNKS",
    "SmolVLATrainingConfig",
    "TARGET_BRANCH",
    "VALIDATION_EPISODES_PER_TASK",
    "apply_padding_mask",
    "build_model_view_manifest",
    "build_evaluation_schedule",
    "build_language_intervention_manifest",
    "build_padding_audit",
    "build_static_action_audit",
    "canonical_fingerprint",
    "classify_pick_gate",
    "validate_model_batch",
]
