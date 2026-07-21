"""Typed metadata for the LangMani 2.0 planar-pushing task."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

type PushObjectId = Literal["blue_cube", "orange_cylinder"]
type TargetRegionId = Literal["left", "right", "forward_left", "forward_right"]
type PushDifficulty = Literal["standard", "hard"]
type PushInstructionTemplateId = Literal["canonical_push_v0"]

PUSH_OBJECT_IDS: tuple[PushObjectId, ...] = ("blue_cube", "orange_cylinder")
TARGET_REGION_IDS: tuple[TargetRegionId, ...] = (
    "left",
    "right",
    "forward_left",
    "forward_right",
)
PUSH_DIFFICULTIES: tuple[PushDifficulty, ...] = ("standard", "hard")
PUSH_INSTRUCTION_TEMPLATE_IDS: tuple[PushInstructionTemplateId, ...] = ("canonical_push_v0",)

PUSH_SCENE_ID_PREFIX = "langmani-push-scene-v0"
PUSH_TASK_ID_PREFIX = "langmani-push-task-v0"

_PUSH_TASK_FIELDS = frozenset(
    {"target_object_id", "target_region_id", "difficulty", "instruction_template_id"}
)


def _choice(value: object, *, label: str, allowed: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string, got {type(value).__name__}")
    if value not in allowed:
        choices = ", ".join(repr(item) for item in allowed)
        raise ValueError(f"unknown {label} {value!r}; expected one of: {choices}")
    return value


@dataclass(frozen=True, slots=True)
class PushTaskSpec:
    """Semantic pushing target, independent of one sampled physical scene."""

    target_object_id: PushObjectId
    target_region_id: TargetRegionId
    difficulty: PushDifficulty
    instruction_template_id: PushInstructionTemplateId = "canonical_push_v0"

    def __post_init__(self) -> None:
        _choice(self.target_object_id, label="target_object_id", allowed=PUSH_OBJECT_IDS)
        _choice(self.target_region_id, label="target_region_id", allowed=TARGET_REGION_IDS)
        _choice(self.difficulty, label="difficulty", allowed=PUSH_DIFFICULTIES)
        _choice(
            self.instruction_template_id,
            label="instruction_template_id",
            allowed=PUSH_INSTRUCTION_TEMPLATE_IDS,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PushTaskSpec:
        if not isinstance(value, Mapping):
            raise TypeError(f"push task_spec must be a mapping, got {type(value).__name__}")
        fields = frozenset(value)
        if not all(isinstance(field, str) for field in fields):
            raise TypeError("push task_spec keys must be strings")
        missing = sorted(_PUSH_TASK_FIELDS - fields)
        extra = sorted(fields - _PUSH_TASK_FIELDS)
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append(f"missing fields: {', '.join(missing)}")
            if extra:
                details.append(f"unexpected fields: {', '.join(cast(list[str], extra))}")
            raise ValueError("malformed push task_spec (" + "; ".join(details) + ")")
        return cls(
            target_object_id=cast(
                PushObjectId,
                _choice(
                    value["target_object_id"],
                    label="target_object_id",
                    allowed=PUSH_OBJECT_IDS,
                ),
            ),
            target_region_id=cast(
                TargetRegionId,
                _choice(
                    value["target_region_id"],
                    label="target_region_id",
                    allowed=TARGET_REGION_IDS,
                ),
            ),
            difficulty=cast(
                PushDifficulty,
                _choice(value["difficulty"], label="difficulty", allowed=PUSH_DIFFICULTIES),
            ),
            instruction_template_id=cast(
                PushInstructionTemplateId,
                _choice(
                    value["instruction_template_id"],
                    label="instruction_template_id",
                    allowed=PUSH_INSTRUCTION_TEMPLATE_IDS,
                ),
            ),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "target_object_id": self.target_object_id,
            "target_region_id": self.target_region_id,
            "difficulty": self.difficulty,
            "instruction_template_id": self.instruction_template_id,
        }


def canonical_push_instruction(task_spec: PushTaskSpec) -> str:
    geometry = "blue cube" if task_spec.target_object_id == "blue_cube" else "orange cylinder"
    region = task_spec.target_region_id.replace("_", "-")
    return f"Push the {geometry} into the {region} target region without picking it up."


def stable_push_scene_id(scene_seed: int) -> str:
    if isinstance(scene_seed, bool) or not isinstance(scene_seed, int):
        raise TypeError(f"scene_seed must be an integer, got {type(scene_seed).__name__}")
    if scene_seed < 0:
        raise ValueError("scene_seed must be non-negative")
    return f"{PUSH_SCENE_ID_PREFIX}:{scene_seed}"


def stable_push_task_id(task_spec: PushTaskSpec) -> str:
    return ":".join(
        (
            PUSH_TASK_ID_PREFIX,
            task_spec.target_object_id,
            task_spec.target_region_id,
            task_spec.difficulty,
            task_spec.instruction_template_id,
        )
    )


@dataclass(frozen=True, slots=True)
class PushEpisodeSpec:
    """Immutable language and identity metadata for one pushing episode."""

    scene_seed: int
    task_spec: PushTaskSpec
    canonical_instruction: str
    scene_id: str
    task_id: str

    def __post_init__(self) -> None:
        if self.canonical_instruction != canonical_push_instruction(self.task_spec):
            raise ValueError("canonical_instruction is inconsistent with push task_spec")
        if self.scene_id != stable_push_scene_id(self.scene_seed):
            raise ValueError("scene_id is inconsistent with scene_seed")
        if self.task_id != stable_push_task_id(self.task_spec):
            raise ValueError("task_id is inconsistent with push task_spec")

    @classmethod
    def create(cls, *, scene_seed: int, task_spec: PushTaskSpec) -> PushEpisodeSpec:
        return cls(
            scene_seed=scene_seed,
            task_spec=task_spec,
            canonical_instruction=canonical_push_instruction(task_spec),
            scene_id=stable_push_scene_id(scene_seed),
            task_id=stable_push_task_id(task_spec),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_seed": self.scene_seed,
            "task_spec": self.task_spec.to_dict(),
            "canonical_instruction": self.canonical_instruction,
            "scene_id": self.scene_id,
            "task_id": self.task_id,
        }


__all__ = [
    "PUSH_DIFFICULTIES",
    "PUSH_INSTRUCTION_TEMPLATE_IDS",
    "PUSH_OBJECT_IDS",
    "TARGET_REGION_IDS",
    "PushDifficulty",
    "PushEpisodeSpec",
    "PushInstructionTemplateId",
    "PushObjectId",
    "PushTaskSpec",
    "TargetRegionId",
    "canonical_push_instruction",
    "stable_push_scene_id",
    "stable_push_task_id",
]
