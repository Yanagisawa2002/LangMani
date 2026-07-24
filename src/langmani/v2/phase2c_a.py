"""Frozen contracts for LangMani 2.0 Phase 2C-A ACT baselines.

This module is intentionally independent of LeRobot and ManiSkill.  It owns
the accepted-package consumer boundary, fixed data views, deterministic
uniform-task sampling, evaluation schedules, result classification, and
compact statistical summaries.  Model construction and native simulation
live behind separate modules.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Self, cast

from langmani.datasets.identity import sha256_hex

PHASE2C_A_SCHEMA_VERSION = "langmani-v2-phase2c-a-v0"
PACKAGE_IDENTITY = "LangManiOfficialMultiSkill-v2"
PACKAGE_FINGERPRINT = "sha256:77675e2134e4886a97e4bdac2230c64c3da30c080e647433b7c701a79544ed04"
INVALID_TRANSCRIBED_PACKAGE_FINGERPRINT = (
    "sha256:77675e2134e4886a97e4bdac2230c64c3da30cc080e647433b7c701a79544ed04"
)
NORMALIZATION_IDENTITY = "LangManiOfficialMultiSkill-v2-primary-train-pooled-natural-frame-v0"
NORMALIZATION_FINGERPRINT = (
    "sha256:9e08cc0cc6f719298655d250e75b59fda85f9a8414fa3d53e99397a6c4ee58a6"
)
SOURCE_COMMIT = "19de56781e029c20859839a1e3493920449ff9bf"
TARGET_BRANCH = "codex/langmani-v2-phase2c-a-act-baselines"
CHUNK_SIZE = 16
CONTROL_FREQUENCY_HZ = 20
IMAGE_SHAPE_CHW = (3, 256, 256)
STATE_DIMENSION = 9
ACTION_DIMENSION = 8
TASK_IDS = ("PickCube-v1", "StackCube-v1", "PushCube-v1")
TASK_SLUGS = {
    "PickCube-v1": "pickcube",
    "StackCube-v1": "stackcube",
    "PushCube-v1": "pushcube",
}
SKILL_FAMILIES = {
    "PickCube-v1": "pick",
    "StackCube-v1": "stack",
    "PushCube-v1": "push",
}
TRAIN_EPISODES = {"PickCube-v1": 700, "StackCube-v1": 700, "PushCube-v1": 699}
TRAIN_FRAMES = {"PickCube-v1": 54_608, "StackCube-v1": 74_891, "PushCube-v1": 48_219}
VALIDATION_EPISODES = {task_id: 100 for task_id in TASK_IDS}
UNSEEN_RESET_EPISODES = {"PickCube-v1": 100, "StackCube-v1": 99, "PushCube-v1": 100}
VISUAL_SHIFT_EPISODES = {task_id: 50 for task_id in TASK_IDS}
EXCLUDED_EPISODES = {
    ("StackCube-v1", 938),
    ("PushCube-v1", 202),
}
PRIMARY_SPLITS = (
    "train",
    "validation",
    "test_unseen_reset",
    "test_unseen_task_language",
    "test_visual_shift",
)
POLICY_FEATURE_ALLOWLIST = frozenset(
    {
        "observation.images.base_camera",
        "observation.state",
        "action",
        "action_is_pad",
    }
)
PRIVILEGED_FEATURE_DENYLIST = frozenset(
    {
        "object_pose",
        "target_pose",
        "goal_pose",
        "full_simulator_state",
        "contacts",
        "reward",
        "success",
        "success_internals",
        "expert_phase",
        "future_observation",
    }
)
EXECUTION_HORIZONS = (1, 4, 8)
FINAL_EPISODES_PER_TASK = 50
VALIDATION_EPISODES_PER_TASK = 30
TARGET_EFFECTIVE_PASSES = 20
VISUAL_SHIFT_IDENTITY = "langmani-v2-phase2b6-postrender-appearance-v0"
VISUAL_SHIFT_EXPOSURE = 0.9
VISUAL_SHIFT_RGB_MULTIPLIERS = (1.03, 1.0, 0.94)


class Phase2CAContractError(RuntimeError):
    """Raised when a Phase 2C-A frozen boundary is violated."""


class ModelKind(StrEnum):
    PICK = "pick_act"
    STACK = "stack_act"
    PUSH = "push_act"
    SHARED = "shared_task_id_act"

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


class Phase2CAResult(StrEnum):
    RESULT_A = "RESULT_A"
    RESULT_B = "RESULT_B"
    RESULT_C = "RESULT_C"
    RESULT_D = "RESULT_D"


FAILURE_CATEGORIES = (
    "no_initial_motion",
    "incorrect_approach",
    "contact_without_completion",
    "failed_grasp",
    "object_drop",
    "failed_stack_alignment",
    "stack_collapse",
    "insufficient_push",
    "push_overshoot",
    "wrong_task_behavior",
    "action_oscillation",
    "timeout",
    "invalid_policy_output",
    "simulator_error",
    "other",
)


def _canonical_fingerprint(payload: Mapping[str, object]) -> str:
    return f"sha256:{sha256_hex(dict(payload))}"


def _positive_int(value: int, label: str, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")


def _digest(value: str, label: str) -> None:
    candidate = value.removeprefix("sha256:")
    if len(candidate) != 64 or any(character not in "0123456789abcdef" for character in candidate):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class Phase2CAModelConfig:
    """The one shared ACT architecture used by all four primary baselines."""

    image_shape_chw: tuple[int, int, int] = IMAGE_SHAPE_CHW
    state_dimension: int = STATE_DIMENSION
    action_dimension: int = ACTION_DIMENSION
    chunk_size: int = CHUNK_SIZE
    n_action_steps: int = CHUNK_SIZE
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: None = None
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4
    dropout: float = 0.1
    kl_weight: float = 10.0
    augmentation: str = "none"
    task_condition_dimension: int | None = None
    task_condition_feature: str | None = None

    def __post_init__(self) -> None:
        if (
            self.image_shape_chw != IMAGE_SHAPE_CHW
            or self.state_dimension != STATE_DIMENSION
            or self.action_dimension != ACTION_DIMENSION
            or self.chunk_size != CHUNK_SIZE
            or self.n_action_steps != CHUNK_SIZE
        ):
            raise Phase2CAContractError("Phase 2C-A ACT input/action/chunk contract changed")
        if (
            self.vision_backbone != "resnet18"
            or self.pretrained_backbone_weights is not None
            or self.dim_model != 512
            or self.n_heads != 8
            or self.dim_feedforward != 3200
            or self.n_encoder_layers != 4
            or self.n_decoder_layers != 1
            or not self.use_vae
            or self.latent_dim != 32
            or self.n_vae_encoder_layers != 4
            or self.dropout != 0.1
            or self.kl_weight != 10.0
            or self.augmentation != "none"
        ):
            raise Phase2CAContractError("Phase 2C-A primary ACT architecture changed")
        conditioned = self.task_condition_dimension is not None
        if conditioned != (self.task_condition_feature is not None):
            raise Phase2CAContractError("task-conditioning dimension and feature must be paired")
        if conditioned and (
            self.task_condition_dimension != len(TASK_IDS)
            or self.task_condition_feature != "observation.environment_state"
        ):
            raise Phase2CAContractError("shared ACT requires the exact three-way ENV task token")

    @classmethod
    def for_model(cls, model: ModelKind | str) -> Self:
        kind = ModelKind(model)
        if kind is ModelKind.SHARED:
            return cls(
                task_condition_dimension=len(TASK_IDS),
                task_condition_feature="observation.environment_state",
            )
        return cls()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["image_shape_chw"] = list(self.image_shape_chw)
        return payload


@dataclass(frozen=True, slots=True)
class Phase2CAOptimizationConfig:
    """Frozen optimization and checkpoint schedule for one primary run."""

    batch_size: int
    training_steps: int
    checkpoint_steps: tuple[int, int, int, int]
    target_samples_by_task: Mapping[str, int]
    effective_passes_by_task: Mapping[str, float]
    optimizer: str = "adamw"
    learning_rate: float = 1e-5
    backbone_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    scheduler: str = "none"
    warmup_steps: int = 0
    gradient_clip_norm: float = 10.0
    precision: str = "bfloat16"
    dataloader_workers: int = 8
    validation_steps: tuple[int, int, int, int] = ()
    seed: int = 0

    def __post_init__(self) -> None:
        for value, label in (
            (self.batch_size, "batch_size"),
            (self.training_steps, "training_steps"),
            (self.dataloader_workers, "dataloader_workers"),
            (self.seed, "seed"),
        ):
            _positive_int(value, label, allow_zero=label in {"dataloader_workers", "seed"})
        if self.optimizer != "adamw" or self.scheduler != "none" or self.warmup_steps != 0:
            raise Phase2CAContractError("Phase 2C-A pins AdamW without warmup or scheduler")
        if (
            self.learning_rate != 1e-5
            or self.backbone_learning_rate != 1e-5
            or self.weight_decay != 1e-4
            or self.gradient_clip_norm != 10.0
            or self.precision != "bfloat16"
        ):
            raise Phase2CAContractError("Phase 2C-A optimization constants changed")
        expected_checkpoints = checkpoint_schedule(self.training_steps)
        if self.checkpoint_steps != expected_checkpoints:
            raise Phase2CAContractError("checkpoint schedule must be the 25/50/75/100% schedule")
        if self.validation_steps and self.validation_steps != expected_checkpoints:
            raise Phase2CAContractError("validation schedule must match retained checkpoints")
        if set(self.target_samples_by_task) - set(TASK_IDS):
            raise Phase2CAContractError("optimization config contains an unknown task")
        if set(self.target_samples_by_task) != set(self.effective_passes_by_task):
            raise Phase2CAContractError("sample and effective-pass task sets differ")
        tasks = set(self.target_samples_by_task)
        if len(tasks) == 1:
            task_id = next(iter(tasks))
            expected_samples = TRAIN_FRAMES[task_id] * TARGET_EFFECTIVE_PASSES
            if self.target_samples_by_task[task_id] != expected_samples:
                raise Phase2CAContractError(
                    "task-specific sample budget must be exactly 20 accepted-data passes"
                )
            expected_steps = (
                math.ceil(TRAIN_FRAMES[task_id] / self.batch_size) * TARGET_EFFECTIVE_PASSES
            )
        elif tasks == set(TASK_IDS):
            if self.batch_size % len(TASK_IDS):
                raise Phase2CAContractError(
                    "shared ACT batch size must preserve exact per-batch 1/3 sampling"
                )
            budgets = set(self.target_samples_by_task.values())
            per_task_batch = self.batch_size // len(TASK_IDS)
            if len(budgets) != 1 or next(iter(budgets)) % per_task_batch:
                raise Phase2CAContractError(
                    "shared ACT sample budget must end on an exact balanced batch"
                )
            expected_steps = next(iter(budgets)) // per_task_batch
        else:
            raise Phase2CAContractError(
                "optimization config must cover one task or all three tasks"
            )
        if self.training_steps != expected_steps:
            raise Phase2CAContractError("training steps do not match the effective sample budget")
        expected_passes = {
            task_id: sample_count / TRAIN_FRAMES[task_id]
            for task_id, sample_count in self.target_samples_by_task.items()
        }
        if dict(self.effective_passes_by_task) != expected_passes:
            raise Phase2CAContractError("effective passes disagree with target sample counts")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key in ("checkpoint_steps", "validation_steps"):
            payload[key] = list(payload[key])
        payload["target_samples_by_task"] = dict(self.target_samples_by_task)
        payload["effective_passes_by_task"] = dict(self.effective_passes_by_task)
        return payload


@dataclass(frozen=True, slots=True)
class Phase2CARunIdentity:
    """Semantic identity consumed by the immutable ACT checkpoint helper."""

    model_kind: ModelKind
    run_role: str
    model_config: Mapping[str, object]
    optimization_config: Mapping[str, object]
    sampler_fingerprint: str
    train_statistics_fingerprint: str
    m3b_split_manifest_digest: str
    git_commit: str
    runtime_fingerprint: str
    package_fingerprint: str = PACKAGE_FINGERPRINT
    normalization_fingerprint: str = NORMALIZATION_FINGERPRINT
    schema_version: str = "langmani-v2-phase2c-a-act-run-v0"
    run_fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_kind", ModelKind(self.model_kind))
        if self.run_role not in {"gpu_smoke", "micro_overfit", "primary_training"}:
            raise Phase2CAContractError("unknown Phase 2C-A ACT run role")
        for value, label in (
            (self.sampler_fingerprint, "sampler_fingerprint"),
            (self.train_statistics_fingerprint, "train_statistics_fingerprint"),
            (self.m3b_split_manifest_digest, "m3b_split_manifest_digest"),
            (self.runtime_fingerprint, "runtime_fingerprint"),
            (self.package_fingerprint, "package_fingerprint"),
            (self.normalization_fingerprint, "normalization_fingerprint"),
        ):
            _digest(value, label)
        if self.package_fingerprint != PACKAGE_FINGERPRINT:
            raise Phase2CAContractError("ACT run is not bound to the canonical package fingerprint")
        if self.normalization_fingerprint != NORMALIZATION_FINGERPRINT:
            raise Phase2CAContractError("ACT run is not bound to the canonical normalization")
        if len(self.git_commit) != 40:
            raise ValueError("git_commit must be a full SHA-1")
        expected = _canonical_fingerprint(self.identity_payload())
        if not self.run_fingerprint:
            object.__setattr__(self, "run_fingerprint", expected)
        elif self.run_fingerprint != expected:
            raise Phase2CAContractError("run_fingerprint disagrees with the semantic identity")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_kind": self.model_kind.value,
            "run_role": self.run_role,
            "model_config": dict(self.model_config),
            "optimization_config": dict(self.optimization_config),
            "sampler_fingerprint": self.sampler_fingerprint,
            "train_statistics_fingerprint": self.train_statistics_fingerprint,
            "m3b_split_manifest_digest": self.m3b_split_manifest_digest,
            "git_commit": self.git_commit,
            "runtime_fingerprint": self.runtime_fingerprint,
            "package_fingerprint": self.package_fingerprint,
            "normalization_fingerprint": self.normalization_fingerprint,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.identity_payload(), "run_fingerprint": self.run_fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = dict(value)
        for key in ("model_config", "optimization_config"):
            if not isinstance(payload.get(key), Mapping):
                raise TypeError(f"{key} must be an object")
            payload[key] = dict(cast(Mapping[str, object], payload[key]))
        return cls(**payload)


def checkpoint_schedule(training_steps: int) -> tuple[int, int, int, int]:
    _positive_int(training_steps, "training_steps")
    values = tuple(math.ceil(training_steps * fraction / 4) for fraction in range(1, 5))
    if len(set(values)) != 4:
        raise ValueError("training budget is too small for four distinct checkpoints")
    return values


def primary_optimization_config(
    model: ModelKind | str,
    *,
    batch_size: int,
    dataloader_workers: int = 8,
    seed: int = 0,
) -> Phase2CAOptimizationConfig:
    """Derive optimizer steps from frozen accepted-frame budgets."""

    kind = ModelKind(model)
    if kind is ModelKind.SHARED:
        if batch_size % len(TASK_IDS):
            raise Phase2CAContractError(
                "shared ACT batch size must be divisible by the three task families"
            )
        total_per_task_budget = sum(
            TRAIN_FRAMES[task_id] * TARGET_EFFECTIVE_PASSES for task_id in TASK_IDS
        )
        requested_common_budget = math.ceil(total_per_task_budget / len(TASK_IDS))
        per_task_batch = batch_size // len(TASK_IDS)
        common_budget = math.ceil(requested_common_budget / per_task_batch) * per_task_batch
        samples = {task_id: common_budget for task_id in TASK_IDS}
        steps = common_budget // per_task_batch
    else:
        task_id = str(kind.task_id)
        samples = {task_id: TRAIN_FRAMES[task_id] * TARGET_EFFECTIVE_PASSES}
        steps = math.ceil(TRAIN_FRAMES[task_id] / batch_size) * TARGET_EFFECTIVE_PASSES
    passes = {
        task_id: sample_count / TRAIN_FRAMES[task_id] for task_id, sample_count in samples.items()
    }
    schedule = checkpoint_schedule(steps)
    return Phase2CAOptimizationConfig(
        batch_size=batch_size,
        training_steps=steps,
        checkpoint_steps=schedule,
        validation_steps=schedule,
        target_samples_by_task=samples,
        effective_passes_by_task=passes,
        dataloader_workers=dataloader_workers,
        seed=seed,
    )


def validate_consumed_training_samples(
    optimization: Phase2CAOptimizationConfig,
    observed_samples: Mapping[str, int],
) -> dict[str, int]:
    """Fail closed unless a completed run consumed its exact frozen sample budget."""

    observed = {task_id: int(observed_samples.get(task_id, 0)) for task_id in TASK_IDS}
    if set(observed_samples) - set(TASK_IDS) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in observed_samples.values()
    ):
        raise Phase2CAContractError("training reported invalid task sample counts")
    expected = {
        task_id: int(optimization.target_samples_by_task.get(task_id, 0)) for task_id in TASK_IDS
    }
    if observed != expected:
        raise Phase2CAContractError(
            f"training sample budget mismatch: expected={expected}, observed={observed}"
        )
    return observed


def task_onehot(task_id: str) -> tuple[float, float, float]:
    try:
        index = TASK_IDS.index(task_id)
    except ValueError as error:
        raise Phase2CAContractError(f"unknown Phase 2C-A task ID {task_id!r}") from error
    return tuple(1.0 if position == index else 0.0 for position in range(len(TASK_IDS)))


class UniformTaskBatchSampler:
    """Deterministic infinite sampler with exact long-run 1/3 task frequency.

    The sampler consumes task-local frame indices.  For batch sizes not
    divisible by three, the one- or two-sample remainder rotates across tasks,
    so every three consecutive batches contain exactly equal task counts.
    Each task owns an independent seeded permutation stream.
    """

    def __init__(
        self,
        task_lengths: Mapping[str, int],
        *,
        batch_size: int,
        seed: int,
        start_batch: int = 0,
    ) -> None:
        if tuple(task_lengths) != TASK_IDS:
            raise Phase2CAContractError("uniform sampler requires canonical task order")
        if any(
            isinstance(length, bool) or not isinstance(length, int) or length < 1
            for length in task_lengths.values()
        ):
            raise ValueError("every task dataset length must be positive")
        _positive_int(batch_size, "batch_size")
        _positive_int(seed, "seed", allow_zero=True)
        _positive_int(start_batch, "start_batch", allow_zero=True)
        self.task_lengths = dict(task_lengths)
        self.batch_size = batch_size
        self.seed = seed
        self.start_batch = start_batch
        self.offsets: dict[str, int] = {}
        offset = 0
        for task_id in TASK_IDS:
            self.offsets[task_id] = offset
            offset += self.task_lengths[task_id]

    def _task_counts(self, batch_index: int) -> dict[str, int]:
        base, remainder = divmod(self.batch_size, len(TASK_IDS))
        counts = {task_id: base for task_id in TASK_IDS}
        for position in range(remainder):
            task_id = TASK_IDS[(batch_index + position) % len(TASK_IDS)]
            counts[task_id] += 1
        return counts

    def __iter__(self) -> Iterator[list[int]]:
        import torch

        permutations: dict[str, list[int]] = {}
        cursors: dict[str, int] = {}
        epochs: dict[str, int] = {}

        def refill(task_id: str) -> None:
            generator = torch.Generator().manual_seed(
                (self.seed + TASK_IDS.index(task_id) * 1_000_003 + epochs[task_id]) % (2**63 - 1)
            )
            permutations[task_id] = torch.randperm(
                self.task_lengths[task_id], generator=generator
            ).tolist()
            cursors[task_id] = 0
            epochs[task_id] += 1

        for task_id in TASK_IDS:
            epochs[task_id] = 0
            refill(task_id)

        batch_index = 0
        while batch_index < self.start_batch:
            for task_id, count in self._task_counts(batch_index).items():
                for _ in range(count):
                    if cursors[task_id] == len(permutations[task_id]):
                        refill(task_id)
                    cursors[task_id] += 1
            batch_index += 1

        while True:
            batch: list[int] = []
            for task_id, count in self._task_counts(batch_index).items():
                offset = self.offsets[task_id]
                for _ in range(count):
                    if cursors[task_id] == len(permutations[task_id]):
                        refill(task_id)
                    batch.append(offset + permutations[task_id][cursors[task_id]])
                    cursors[task_id] += 1
            batch_generator = torch.Generator().manual_seed(
                (self.seed + 7_000_001 + batch_index) % (2**63 - 1)
            )
            order = torch.randperm(len(batch), generator=batch_generator).tolist()
            yield [batch[index] for index in order]
            batch_index += 1

    def audit(self, batch_count: int) -> dict[str, object]:
        _positive_int(batch_count, "batch_count")
        counts = Counter[str]()
        for batch_index in range(batch_count):
            counts.update(self._task_counts(batch_index))
        total = sum(counts.values())
        semantic = {
            "task_order": list(TASK_IDS),
            "task_lengths": dict(self.task_lengths),
            "batch_size": self.batch_size,
            "seed": self.seed,
            "start_batch": self.start_batch,
            "audited_batch_count": batch_count,
            "sample_counts": {task_id: counts[task_id] for task_id in TASK_IDS},
            "sample_fractions": {task_id: counts[task_id] / total for task_id in TASK_IDS},
            "maximum_count_difference": max(counts.values()) - min(counts.values()),
            "uniform_task_sampling": max(counts.values()) - min(counts.values()) <= 1,
        }
        return {
            "schema_version": "langmani-v2-phase2c-a-uniform-sampler-audit-v0",
            **semantic,
            "fingerprint": _canonical_fingerprint(semantic),
            "passed": semantic["uniform_task_sampling"],
        }


def validate_package_document(package: Mapping[str, object]) -> dict[str, object]:
    """Verify immutable fields required before a policy consumer can exist."""

    exclusions = package.get("exclusions")
    tasks = package.get("tasks")
    gates = package.get("gates")
    if (
        package.get("package_identity") != PACKAGE_IDENTITY
        or package.get("fingerprint") != PACKAGE_FINGERPRINT
        or package.get("accepted_episode_count") != 2_998
        or package.get("accepted_frame_count") != 254_200
        or package.get("contains_trained_model") is not False
        or package.get("student_policy_training_started") is not False
        or package.get("optimizer_steps") != 0
    ):
        raise Phase2CAContractError("accepted package top-level identity changed")
    if (
        not isinstance(exclusions, list)
        or {
            (item.get("task_id"), item.get("source_episode_id"))
            for item in exclusions
            if isinstance(item, Mapping)
        }
        != EXCLUDED_EPISODES
    ):
        raise Phase2CAContractError("accepted package exclusion set changed")
    if not isinstance(tasks, Mapping) or set(tasks) != set(TASK_IDS):
        raise Phase2CAContractError("accepted package task set changed")
    for task_id in TASK_IDS:
        value = tasks[task_id]
        if not isinstance(value, Mapping):
            raise Phase2CAContractError("accepted package task record is malformed")
        if (
            value.get("episode_count")
            != {
                "PickCube-v1": 1_000,
                "StackCube-v1": 999,
                "PushCube-v1": 999,
            }[task_id]
        ):
            raise Phase2CAContractError(f"{task_id} accepted episode count changed")
        split_records = value.get("splits")
        if not isinstance(split_records, Mapping):
            raise Phase2CAContractError(f"{task_id} split records are missing")
        expected = {
            "train": TRAIN_EPISODES[task_id],
            "validation": VALIDATION_EPISODES[task_id],
            "test_unseen_reset": UNSEEN_RESET_EPISODES[task_id],
            "test_unseen_task_language": 50,
            "test_visual_shift": VISUAL_SHIFT_EPISODES[task_id],
        }
        for split, episode_count in expected.items():
            record = split_records.get(split)
            if not isinstance(record, Mapping) or record.get("episode_count") != episode_count:
                raise Phase2CAContractError(f"{task_id}/{split} count changed")
    required_gates = {
        "accepted_action_frame_alignment",
        "accepted_replay_outcome_agreement",
        "archive_created",
        "invalid_and_nonfinite_actions_zero",
        "lerobot_full_readback",
        "padding_masks_verified",
        "primary_split_leakage_zero",
        "privileged_fields_excluded",
        "restore_validated",
        "source_to_derived_equality",
        "train_only_normalization",
    }
    if not isinstance(gates, Mapping) or any(gates.get(key) is not True for key in required_gates):
        raise Phase2CAContractError("accepted package required gates no longer pass")
    semantic = {
        "package_identity": PACKAGE_IDENTITY,
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "episode_count": 2_998,
        "frame_count": 254_200,
        "tasks": list(TASK_IDS),
        "excluded_episodes": [
            {"task_id": task_id, "source_episode_id": episode_id}
            for task_id, episode_id in sorted(EXCLUDED_EPISODES)
        ],
        "required_gates": sorted(required_gates),
    }
    return {
        "schema_version": "langmani-v2-phase2c-a-package-verification-v0",
        **semantic,
        "fingerprint": _canonical_fingerprint(semantic),
        "passed": True,
    }


def build_training_view_manifest(
    split_manifest: Mapping[str, object],
    *,
    package_fingerprint: str = PACKAGE_FINGERPRINT,
) -> dict[str, object]:
    assignments = split_manifest.get("assignments")
    if package_fingerprint != PACKAGE_FINGERPRINT:
        raise Phase2CAContractError("training view requested a noncanonical package")
    if not isinstance(assignments, list):
        raise Phase2CAContractError("accepted split manifest assignments are missing")
    records: list[dict[str, object]] = []
    for raw in assignments:
        if not isinstance(raw, Mapping) or raw.get("primary_split") != "train":
            continue
        task_id = raw.get("task_id")
        source_episode_id = raw.get("source_episode_id")
        if (
            task_id not in TASK_IDS
            or not isinstance(source_episode_id, int)
            or (str(task_id), source_episode_id) in EXCLUDED_EPISODES
        ):
            raise Phase2CAContractError("training view contains an invalid or excluded episode")
        records.append(
            {
                "task_id": task_id,
                "skill_family": raw.get("skill_family"),
                "source_episode_id": source_episode_id,
                "derived_episode_identity": raw.get("derived_episode_identity"),
                "source_trajectory_identity": raw.get("source_trajectory_identity"),
                "action_sha256": raw.get("action_sha256"),
                "reset_identity": raw.get("reset_identity"),
                "frame_count": raw.get("transition_count"),
                "transition_count": raw.get("transition_count"),
                "primary_split": "train",
                "instruction_template_id": raw.get("instruction_template_id"),
            }
        )
    records.sort(
        key=lambda value: (TASK_IDS.index(str(value["task_id"])), value["source_episode_id"])
    )
    counts = Counter(str(record["task_id"]) for record in records)
    frames = Counter()
    for record in records:
        frame_count = record["frame_count"]
        if not isinstance(frame_count, int) or frame_count < 1:
            raise Phase2CAContractError("training episode has an invalid frame count")
        frames[str(record["task_id"])] += frame_count
    if dict(counts) != TRAIN_EPISODES or dict(frames) != TRAIN_FRAMES:
        raise Phase2CAContractError("training-view counts disagree with the frozen accepted split")
    semantic = {
        "package_identity": PACKAGE_IDENTITY,
        "package_fingerprint": package_fingerprint,
        "source_split_manifest_fingerprint": split_manifest.get("fingerprint"),
        "episode_count": len(records),
        "frame_count": sum(frames.values()),
        "episodes_by_task": dict(counts),
        "frames_by_task": dict(frames),
        "excluded_episodes_absent": True,
        "privileged_policy_fields": [],
        "records": records,
    }
    return {
        "schema_version": "langmani-v2-phase2c-a-training-view-v0",
        **semantic,
        "fingerprint": _canonical_fingerprint(semantic),
        "passed": True,
    }


def build_evaluation_schedule(
    split_manifest: Mapping[str, object],
    source_inventory: Mapping[str, object],
) -> dict[str, object]:
    """Freeze validation and final reset identities before any rollout result exists."""

    assignments = split_manifest.get("assignments")
    sources = source_inventory.get("episodes")
    if not isinstance(assignments, list) or not isinstance(sources, list):
        raise Phase2CAContractError("evaluation schedule sources are malformed")
    reset_by_source: dict[tuple[str, int], Mapping[str, object]] = {}
    for raw in sources:
        if not isinstance(raw, Mapping):
            raise Phase2CAContractError("source inventory episode is malformed")
        task_id = raw.get("task_id")
        source_episode_id = raw.get("source_episode_id")
        if task_id in TASK_IDS and isinstance(source_episode_id, int):
            reset_by_source[(str(task_id), source_episode_id)] = raw
    schedules: dict[str, dict[str, list[dict[str, object]]]] = {}
    target_counts = {
        "validation": VALIDATION_EPISODES_PER_TASK,
        "test_unseen_reset": FINAL_EPISODES_PER_TASK,
        "test_visual_shift": FINAL_EPISODES_PER_TASK,
    }
    for split, maximum in target_counts.items():
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
                raise Phase2CAContractError(f"{task_id}/{split} lacks {maximum} identities")
            rows: list[dict[str, object]] = []
            for ordinal, assignment in enumerate(selected):
                source_episode_id = assignment.get("source_episode_id")
                if not isinstance(source_episode_id, int):
                    raise Phase2CAContractError("scheduled source episode ID is malformed")
                source = reset_by_source.get((task_id, source_episode_id))
                if source is None:
                    raise Phase2CAContractError("scheduled episode lacks source reset metadata")
                if (task_id, source_episode_id) in EXCLUDED_EPISODES:
                    raise Phase2CAContractError("excluded episode entered an evaluation schedule")
                rows.append(
                    {
                        "evaluation_id": (f"phase2c-a:{split}:{TASK_SLUGS[task_id]}:{ordinal:03d}"),
                        "task_id": task_id,
                        "split": split,
                        "source_episode_id": source_episode_id,
                        "derived_episode_identity": assignment.get("derived_episode_identity"),
                        "reset_identity": assignment.get("reset_identity"),
                        "reset_kwargs": source.get("reset_kwargs"),
                        "visual_transform": (
                            VISUAL_SHIFT_IDENTITY if split == "test_visual_shift" else None
                        ),
                    }
                )
            schedules[split][task_id] = rows
    semantic = {
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "source_split_manifest_fingerprint": split_manifest.get("fingerprint"),
        "source_inventory_fingerprint": source_inventory.get("fingerprint"),
        "validation_role": "checkpoint_and_execution_horizon_selection_only",
        "final_splits": ["test_unseen_reset", "test_visual_shift"],
        "final_settings_mutable_after_results": False,
        "execution_horizons": list(EXECUTION_HORIZONS),
        "schedules": schedules,
    }
    return {
        "schema_version": "langmani-v2-phase2c-a-evaluation-schedule-v0",
        **semantic,
        "fingerprint": _canonical_fingerprint(semantic),
        "passed": True,
    }


def task_condition_intervention(
    task_ids: Sequence[str],
    *,
    seed: int = 20_260_724,
) -> dict[str, object]:
    """Freeze correct/incorrect/shuffled task-ID assignments for a fixed subset."""

    if not task_ids or any(task_id not in TASK_IDS for task_id in task_ids):
        raise Phase2CAContractError("intervention requires canonical task IDs")
    import random

    rng = random.Random(seed)
    shuffled = list(task_ids)
    while shuffled == list(task_ids):
        rng.shuffle(shuffled)
    records = []
    for index, task_id in enumerate(task_ids):
        incorrect = TASK_IDS[(TASK_IDS.index(task_id) + 1) % len(TASK_IDS)]
        records.append(
            {
                "index": index,
                "environment_task_id": task_id,
                "correct_task_id": task_id,
                "incorrect_task_id": incorrect,
                "shuffled_task_id": shuffled[index],
            }
        )
    semantic = {
        "seed": seed,
        "conditions": ["correct_task_id", "incorrect_task_id", "shuffled_task_id"],
        "natural_language_grounding_claim": False,
        "records": records,
    }
    return {
        "schema_version": "langmani-v2-phase2c-a-task-condition-intervention-v0",
        **semantic,
        "fingerprint": _canonical_fingerprint(semantic),
    }


def wilson_interval(
    success_count: int, episode_count: int, *, z: float = 1.959963984540054
) -> tuple[float, float]:
    _positive_int(episode_count, "episode_count")
    _positive_int(success_count, "success_count", allow_zero=True)
    if success_count > episode_count:
        raise ValueError("success_count cannot exceed episode_count")
    probability = success_count / episode_count
    denominator = 1 + z * z / episode_count
    center = (probability + z * z / (2 * episode_count)) / denominator
    radius = (
        z
        * math.sqrt(
            probability * (1 - probability) / episode_count
            + z * z / (4 * episode_count * episode_count)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def classify_result(
    *,
    pipeline_valid: bool,
    training_runs_complete: bool,
    checkpoints_recoverable: bool,
    padding_mask_verified: bool,
    closed_loop_complete: bool,
    per_task_success_count: Mapping[str, int],
    shared_success_count: Mapping[str, int],
    invalid_action_episode_count: int,
) -> dict[str, object]:
    """Apply the immutable Result A/B/C/D decision tree."""

    _positive_int(invalid_action_episode_count, "invalid_action_episode_count", allow_zero=True)
    if set(per_task_success_count) != set(TASK_IDS) or set(shared_success_count) != set(TASK_IDS):
        raise Phase2CAContractError("result classification requires all three task counts")
    common_valid = (
        pipeline_valid
        and training_runs_complete
        and checkpoints_recoverable
        and padding_mask_verified
        and closed_loop_complete
        and invalid_action_episode_count == 0
    )
    per_task_nonzero = all(per_task_success_count[task_id] > 0 for task_id in TASK_IDS)
    shared_nonzero = all(shared_success_count[task_id] > 0 for task_id in TASK_IDS)
    if not common_valid:
        result = Phase2CAResult.RESULT_D
    elif per_task_nonzero and shared_nonzero:
        result = Phase2CAResult.RESULT_A
    elif per_task_nonzero:
        result = Phase2CAResult.RESULT_B
    else:
        result = Phase2CAResult.RESULT_C
    semantic: dict[str, Any] = {
        "result": result.value,
        "pipeline_valid": pipeline_valid,
        "training_runs_complete": training_runs_complete,
        "checkpoints_recoverable": checkpoints_recoverable,
        "padding_mask_verified": padding_mask_verified,
        "closed_loop_complete": closed_loop_complete,
        "per_task_success_count": dict(per_task_success_count),
        "shared_success_count": dict(shared_success_count),
        "per_task_nonzero_all_tasks": per_task_nonzero,
        "shared_nonzero_all_tasks": shared_nonzero,
        "invalid_action_episode_count": invalid_action_episode_count,
        "act_baselines_validated": result in {Phase2CAResult.RESULT_A, Phase2CAResult.RESULT_B},
        "shared_act_baseline_failed": result is Phase2CAResult.RESULT_B,
        "smolvla_phase_eligible": result
        in {
            Phase2CAResult.RESULT_A,
            Phase2CAResult.RESULT_B,
            Phase2CAResult.RESULT_C,
        }
        and pipeline_valid,
        "smolvla_training_authorized": False,
        "vla_jepa_training_authorized": False,
    }
    return {
        "schema_version": "langmani-v2-phase2c-a-result-classification-v0",
        **semantic,
        "fingerprint": _canonical_fingerprint(semantic),
    }


def summarize_binary_results(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not records:
        raise Phase2CAContractError("cannot summarize empty episode results")
    successes = sum(record.get("success") is True for record in records)
    lower, upper = wilson_interval(successes, len(records))
    outcomes = Counter(str(record.get("outcome")) for record in records)
    failures = Counter(str(record.get("failure_category")) for record in records)
    return {
        "episode_count": len(records),
        "success_count": successes,
        "success_rate": successes / len(records),
        "success_wilson_95": [lower, upper],
        "outcomes": dict(sorted(outcomes.items())),
        "failure_categories": dict(sorted(failures.items())),
        "invalid_action_count": sum(record.get("invalid_action") is True for record in records),
        "simulator_error_count": sum(
            record.get("outcome") == "simulator_error" for record in records
        ),
    }


__all__ = [
    "ACTION_DIMENSION",
    "CHUNK_SIZE",
    "CONTROL_FREQUENCY_HZ",
    "EXECUTION_HORIZONS",
    "EXCLUDED_EPISODES",
    "FAILURE_CATEGORIES",
    "FINAL_EPISODES_PER_TASK",
    "INVALID_TRANSCRIBED_PACKAGE_FINGERPRINT",
    "ModelKind",
    "NORMALIZATION_FINGERPRINT",
    "NORMALIZATION_IDENTITY",
    "PACKAGE_FINGERPRINT",
    "PACKAGE_IDENTITY",
    "POLICY_FEATURE_ALLOWLIST",
    "PRIVILEGED_FEATURE_DENYLIST",
    "Phase2CAContractError",
    "Phase2CAModelConfig",
    "Phase2CAOptimizationConfig",
    "Phase2CAResult",
    "Phase2CARunIdentity",
    "PRIMARY_SPLITS",
    "SKILL_FAMILIES",
    "SOURCE_COMMIT",
    "STATE_DIMENSION",
    "TARGET_BRANCH",
    "TARGET_EFFECTIVE_PASSES",
    "TASK_IDS",
    "TASK_SLUGS",
    "TRAIN_EPISODES",
    "TRAIN_FRAMES",
    "UniformTaskBatchSampler",
    "VALIDATION_EPISODES_PER_TASK",
    "VISUAL_SHIFT_EXPOSURE",
    "VISUAL_SHIFT_IDENTITY",
    "VISUAL_SHIFT_RGB_MULTIPLIERS",
    "build_evaluation_schedule",
    "build_training_view_manifest",
    "checkpoint_schedule",
    "classify_result",
    "primary_optimization_config",
    "summarize_binary_results",
    "task_condition_intervention",
    "task_onehot",
    "validate_package_document",
    "wilson_interval",
]
