"""Typed language-task metadata for LangMani environments.

These values intentionally live outside ManiSkill's numeric observation and step-info
paths. They are immutable project contracts with JSON-serializable ``to_dict()``
representations that can be materialized on demand by environment accessors.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

type ObjectId = Literal["red_cube", "green_cube", "blue_cube"]
type BinId = Literal["left_bin", "right_bin"]
type InstructionTemplateId = Literal["canonical_v0"]

OBJECT_IDS: tuple[ObjectId, ...] = ("red_cube", "green_cube", "blue_cube")
BIN_IDS: tuple[BinId, ...] = ("left_bin", "right_bin")
INSTRUCTION_TEMPLATE_IDS: tuple[InstructionTemplateId, ...] = ("canonical_v0",)

SCENE_ID_PREFIX = "langmani-pick-place-scene-v0"
TASK_ID_PREFIX = "langmani-pick-place-task-v0"

_TASK_SPEC_FIELDS = frozenset({"target_object_id", "target_bin_id", "instruction_template_id"})


def _validated_string_choice(
    value: object,
    *,
    field_name: str,
    allowed: tuple[str, ...],
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    if value not in allowed:
        choices = ", ".join(repr(choice) for choice in allowed)
        raise ValueError(f"unknown {field_name} {value!r}; expected one of: {choices}")
    return value


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Semantic task selection, independent of the physical scene layout."""

    target_object_id: ObjectId
    target_bin_id: BinId
    instruction_template_id: InstructionTemplateId

    def __post_init__(self) -> None:
        _validated_string_choice(
            self.target_object_id,
            field_name="target_object_id",
            allowed=OBJECT_IDS,
        )
        _validated_string_choice(
            self.target_bin_id,
            field_name="target_bin_id",
            allowed=BIN_IDS,
        )
        _validated_string_choice(
            self.instruction_template_id,
            field_name="instruction_template_id",
            allowed=INSTRUCTION_TEMPLATE_IDS,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> TaskSpec:
        """Parse the exact public reset-option shape without applying defaults."""
        if not isinstance(value, Mapping):
            raise TypeError(f"task_spec must be a mapping, got {type(value).__name__}")

        fields = frozenset(value.keys())
        if not all(isinstance(field, str) for field in fields):
            raise TypeError("task_spec keys must be strings")
        missing = sorted(_TASK_SPEC_FIELDS - fields)
        extra = sorted(fields - _TASK_SPEC_FIELDS)
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append(f"missing fields: {', '.join(missing)}")
            if extra:
                details.append(f"unexpected fields: {', '.join(cast(list[str], extra))}")
            raise ValueError("malformed task_spec (" + "; ".join(details) + ")")

        object_id = _validated_string_choice(
            value["target_object_id"],
            field_name="target_object_id",
            allowed=OBJECT_IDS,
        )
        bin_id = _validated_string_choice(
            value["target_bin_id"],
            field_name="target_bin_id",
            allowed=BIN_IDS,
        )
        template_id = _validated_string_choice(
            value["instruction_template_id"],
            field_name="instruction_template_id",
            allowed=INSTRUCTION_TEMPLATE_IDS,
        )
        return cls(
            target_object_id=cast(ObjectId, object_id),
            target_bin_id=cast(BinId, bin_id),
            instruction_template_id=cast(InstructionTemplateId, template_id),
        )

    def to_dict(self) -> dict[str, str]:
        """Return a fresh JSON-serializable representation."""
        return {
            "target_object_id": self.target_object_id,
            "target_bin_id": self.target_bin_id,
            "instruction_template_id": self.instruction_template_id,
        }


def canonical_instruction(task_spec: TaskSpec) -> str:
    """Render the only language template supported in M1."""
    color = task_spec.target_object_id.removesuffix("_cube")
    side = task_spec.target_bin_id.removesuffix("_bin")
    return f"Pick up the {color} cube and place it in the {side} bin."


def stable_scene_id(scene_seed: int) -> str:
    """Return a scene identifier that depends only on the physical-layout seed."""
    if isinstance(scene_seed, bool) or not isinstance(scene_seed, int):
        raise TypeError(f"scene_seed must be an integer, got {type(scene_seed).__name__}")
    if scene_seed < 0:
        raise ValueError("scene_seed must be non-negative")
    return f"{SCENE_ID_PREFIX}:{scene_seed}"


def stable_task_id(task_spec: TaskSpec) -> str:
    """Return a process-stable semantic identifier without Python hashing."""
    return ":".join(
        (
            TASK_ID_PREFIX,
            task_spec.target_object_id,
            task_spec.target_bin_id,
            task_spec.instruction_template_id,
        )
    )


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    """Immutable metadata describing one scene/task pairing."""

    scene_seed: int
    task_spec: TaskSpec
    canonical_instruction: str
    scene_id: str
    task_id: str

    def __post_init__(self) -> None:
        if self.canonical_instruction != canonical_instruction(self.task_spec):
            raise ValueError("canonical_instruction is inconsistent with task_spec")
        if self.scene_id != stable_scene_id(self.scene_seed):
            raise ValueError("scene_id is inconsistent with scene_seed")
        if self.task_id != stable_task_id(self.task_spec):
            raise ValueError("task_id is inconsistent with task_spec")

    @classmethod
    def create(cls, *, scene_seed: int, task_spec: TaskSpec) -> EpisodeSpec:
        """Build an EpisodeSpec with validated derived language and identifiers."""
        return cls(
            scene_seed=scene_seed,
            task_spec=task_spec,
            canonical_instruction=canonical_instruction(task_spec),
            scene_id=stable_scene_id(scene_seed),
            task_id=stable_task_id(task_spec),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh nested mapping accepted by ``json.dumps``."""
        return {
            "scene_seed": self.scene_seed,
            "task_spec": self.task_spec.to_dict(),
            "canonical_instruction": self.canonical_instruction,
            "scene_id": self.scene_id,
            "task_id": self.task_id,
        }
