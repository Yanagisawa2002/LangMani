"""Explicit skill-family and task-instance taxonomy for LangMani 2.0."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self, cast

from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.push_specs import (
    PUSH_DIFFICULTIES,
    PUSH_OBJECT_IDS,
    TARGET_REGION_IDS,
    PushTaskSpec,
    canonical_push_instruction,
    stable_push_task_id,
)
from langmani.environments.specs import (
    BIN_IDS,
    OBJECT_IDS,
    TaskSpec,
    canonical_instruction,
    stable_task_id,
)

TAXONOMY_SCHEMA_VERSION = "langmani-v2-task-taxonomy-v0"
PICK_AND_PLACE_SKILL_ID = "pick_and_place"
PUSH_TO_REGION_SKILL_ID = "push_to_region"
LEGACY_ENVIRONMENT_ID = "LangMani-PickPlaceByInstruction-v0"
PUSH_ENVIRONMENT_ID = "LangMani-PushToRegion-v0"


class TaskTaxonomyError(ValueError):
    """Raised when a v2 task taxonomy record is malformed or incompatible."""


@dataclass(frozen=True, slots=True)
class SkillFamilySpec:
    """One semantic manipulation family, independent of task attributes."""

    skill_family_id: str
    description: str
    object_attribute_name: str
    target_attribute_name: str
    future_instance_axes: tuple[str, ...] = ()
    schema_version: str = TAXONOMY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TAXONOMY_SCHEMA_VERSION:
            raise TaskTaxonomyError("unknown task taxonomy schema version")
        for value, label in (
            (self.skill_family_id, "skill_family_id"),
            (self.description, "description"),
            (self.object_attribute_name, "object_attribute_name"),
            (self.target_attribute_name, "target_attribute_name"),
        ):
            if not isinstance(value, str) or not value:
                raise TaskTaxonomyError(f"{label} must be a non-empty string")
        if len(set(self.future_instance_axes)) != len(self.future_instance_axes):
            raise TaskTaxonomyError("future_instance_axes must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "skill_family_id": self.skill_family_id,
            "description": self.description,
            "object_attribute_name": self.object_attribute_name,
            "target_attribute_name": self.target_attribute_name,
            "future_instance_axes": list(self.future_instance_axes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = dict(value)
        axes = payload.get("future_instance_axes", [])
        if not isinstance(axes, list) or not all(isinstance(item, str) for item in axes):
            raise TaskTaxonomyError("future_instance_axes must be a string list")
        payload["future_instance_axes"] = tuple(axes)
        try:
            return cls(**payload)
        except TypeError as error:
            raise TaskTaxonomyError(f"malformed skill family: {error}") from error


PICK_AND_PLACE_SKILL = SkillFamilySpec(
    skill_family_id=PICK_AND_PLACE_SKILL_ID,
    description="Move one selected cube into one selected destination bin.",
    object_attribute_name="color",
    target_attribute_name="target_region",
    future_instance_axes=("shape", "size", "material"),
)

PUSH_TO_REGION_SKILL = SkillFamilySpec(
    skill_family_id=PUSH_TO_REGION_SKILL_ID,
    description="Move one selected object into one planar target region by pushing.",
    object_attribute_name="geometry",
    target_attribute_name="target_region",
    future_instance_axes=("difficulty", "material", "size"),
)


@dataclass(frozen=True, slots=True)
class TaskInstanceSpec:
    """One attribute binding within a skill family, not a separate skill."""

    canonical_task_id: str
    skill_family_id: str
    object_attribute: str
    target_region: str
    language_instruction: str
    environment_task_spec: TaskSpec
    success_predicate: str = "m1_conservative_pick_place_success_v0"
    maximum_episode_steps: int = 200
    embodiment: str = "panda"
    camera_setup: str = "base_camera_v0"
    environment_id: str = LEGACY_ENVIRONMENT_ID
    schema_version: str = TAXONOMY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TAXONOMY_SCHEMA_VERSION:
            raise TaskTaxonomyError("unknown task taxonomy schema version")
        if self.skill_family_id != PICK_AND_PLACE_SKILL_ID:
            raise TaskTaxonomyError("Phase 1 supports only the pick_and_place skill family")
        if self.canonical_task_id != stable_task_id(self.environment_task_spec):
            raise TaskTaxonomyError("canonical_task_id disagrees with the M1 TaskSpec")
        if self.object_attribute != self.environment_task_spec.target_object_id.removesuffix(
            "_cube"
        ):
            raise TaskTaxonomyError("object_attribute disagrees with target_object_id")
        if self.target_region != self.environment_task_spec.target_bin_id:
            raise TaskTaxonomyError("target_region disagrees with target_bin_id")
        if self.language_instruction != canonical_instruction(self.environment_task_spec):
            raise TaskTaxonomyError("language_instruction disagrees with the canonical M1 text")
        if self.maximum_episode_steps != 200:
            raise TaskTaxonomyError("Phase 1 preserves the 200-step M1 rollout limit")
        if self.embodiment != "panda" or self.camera_setup != "base_camera_v0":
            raise TaskTaxonomyError("Phase 1 preserves Panda and base_camera_v0")
        if self.environment_id != LEGACY_ENVIRONMENT_ID:
            raise TaskTaxonomyError(f"task instance requires {LEGACY_ENVIRONMENT_ID}")

    @classmethod
    def from_task_spec(cls, task_spec: TaskSpec) -> Self:
        return cls(
            canonical_task_id=stable_task_id(task_spec),
            skill_family_id=PICK_AND_PLACE_SKILL_ID,
            object_attribute=task_spec.target_object_id.removesuffix("_cube"),
            target_region=task_spec.target_bin_id,
            language_instruction=canonical_instruction(task_spec),
            environment_task_spec=task_spec,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "canonical_task_id": self.canonical_task_id,
            "skill_family_id": self.skill_family_id,
            "object_attribute": self.object_attribute,
            "target_region": self.target_region,
            "language_instruction": self.language_instruction,
            "environment_task_spec": self.environment_task_spec.to_dict(),
            "success_predicate": self.success_predicate,
            "maximum_episode_steps": self.maximum_episode_steps,
            "embodiment": self.embodiment,
            "camera_setup": self.camera_setup,
            "environment_id": self.environment_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = dict(value)
        task_value = payload.get("environment_task_spec")
        if not isinstance(task_value, Mapping):
            raise TaskTaxonomyError("environment_task_spec must be an object")
        payload["environment_task_spec"] = TaskSpec.from_mapping(task_value)
        try:
            return cls(**payload)
        except TypeError as error:
            raise TaskTaxonomyError(f"malformed task instance: {error}") from error


PUSH_FAILURE_PREDICATES = (
    "timeout",
    "wrong_object_contact",
    "wrong_object_displacement",
    "target_outside_workspace",
    "target_overshoot",
    "target_toppled",
    "target_lifted",
    "invalid_action",
    "no_progress_stall",
    "robot_collision",
    "action_projection_or_clipping",
)


@dataclass(frozen=True, slots=True)
class PushTaskInstanceSpec:
    """One geometry/region/difficulty binding in the pushing skill family."""

    canonical_task_id: str
    skill_family_id: str
    object_attribute: str
    target_region: str
    difficulty: str
    language_instruction: str
    environment_task_spec: PushTaskSpec
    success_predicate: str = "stable_full_containment_planar_push_success_v0"
    failure_predicates: tuple[str, ...] = PUSH_FAILURE_PREDICATES
    maximum_episode_steps: int = 250
    embodiment: str = "panda"
    camera_setup: str = "base_camera_v0"
    environment_id: str = PUSH_ENVIRONMENT_ID
    compatible_policy_action_space: str = "pd_joint_pos_8d_v0"
    schema_version: str = TAXONOMY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TAXONOMY_SCHEMA_VERSION:
            raise TaskTaxonomyError("unknown task taxonomy schema version")
        if self.skill_family_id != PUSH_TO_REGION_SKILL_ID:
            raise TaskTaxonomyError("push task requires the push_to_region skill family")
        if self.canonical_task_id != stable_push_task_id(self.environment_task_spec):
            raise TaskTaxonomyError("canonical_task_id disagrees with PushTaskSpec")
        if self.object_attribute != self.environment_task_spec.target_object_id:
            raise TaskTaxonomyError("object_attribute disagrees with push target_object_id")
        if self.target_region != self.environment_task_spec.target_region_id:
            raise TaskTaxonomyError("target_region disagrees with PushTaskSpec")
        if self.difficulty != self.environment_task_spec.difficulty:
            raise TaskTaxonomyError("difficulty disagrees with PushTaskSpec")
        if self.language_instruction != canonical_push_instruction(self.environment_task_spec):
            raise TaskTaxonomyError("language_instruction disagrees with canonical push text")
        if self.failure_predicates != PUSH_FAILURE_PREDICATES:
            raise TaskTaxonomyError("push failure predicates changed")
        if self.maximum_episode_steps != 250:
            raise TaskTaxonomyError("push task requires the 250-step rollout limit")
        if self.embodiment != "panda" or self.camera_setup != "base_camera_v0":
            raise TaskTaxonomyError("push task requires Panda and base_camera_v0")
        if self.environment_id != PUSH_ENVIRONMENT_ID:
            raise TaskTaxonomyError(f"push task instance requires {PUSH_ENVIRONMENT_ID}")
        if self.compatible_policy_action_space != "pd_joint_pos_8d_v0":
            raise TaskTaxonomyError("push task requires pd_joint_pos_8d_v0")

    @classmethod
    def from_task_spec(cls, task_spec: PushTaskSpec) -> Self:
        return cls(
            canonical_task_id=stable_push_task_id(task_spec),
            skill_family_id=PUSH_TO_REGION_SKILL_ID,
            object_attribute=task_spec.target_object_id,
            target_region=task_spec.target_region_id,
            difficulty=task_spec.difficulty,
            language_instruction=canonical_push_instruction(task_spec),
            environment_task_spec=task_spec,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "canonical_task_id": self.canonical_task_id,
            "skill_family_id": self.skill_family_id,
            "object_attribute": self.object_attribute,
            "target_region": self.target_region,
            "difficulty": self.difficulty,
            "language_instruction": self.language_instruction,
            "environment_task_spec": self.environment_task_spec.to_dict(),
            "success_predicate": self.success_predicate,
            "failure_predicates": list(self.failure_predicates),
            "maximum_episode_steps": self.maximum_episode_steps,
            "embodiment": self.embodiment,
            "camera_setup": self.camera_setup,
            "environment_id": self.environment_id,
            "compatible_policy_action_space": self.compatible_policy_action_space,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = dict(value)
        task_value = payload.get("environment_task_spec")
        failures = payload.get("failure_predicates", list(PUSH_FAILURE_PREDICATES))
        if not isinstance(task_value, Mapping):
            raise TaskTaxonomyError("environment_task_spec must be an object")
        if not isinstance(failures, list) or not all(isinstance(item, str) for item in failures):
            raise TaskTaxonomyError("failure_predicates must be a string list")
        payload["environment_task_spec"] = PushTaskSpec.from_mapping(task_value)
        payload["failure_predicates"] = tuple(failures)
        try:
            return cls(**payload)
        except TypeError as error:
            raise TaskTaxonomyError(f"malformed push task instance: {error}") from error


type AnyTaskInstanceSpec = TaskInstanceSpec | PushTaskInstanceSpec


@dataclass(frozen=True, slots=True)
class EvaluationTask:
    """One deterministic scene variation paired with a semantic task instance."""

    task_instance: AnyTaskInstanceSpec
    scene_seed: int

    def __post_init__(self) -> None:
        if isinstance(self.scene_seed, bool) or not isinstance(self.scene_seed, int):
            raise TaskTaxonomyError("scene_seed must be an integer")
        if self.scene_seed < 0:
            raise TaskTaxonomyError("scene_seed must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "task_instance": self.task_instance.to_dict(),
            "scene_seed": self.scene_seed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        if set(value) != {"task_instance", "scene_seed"}:
            raise TaskTaxonomyError("evaluation task fields are invalid")
        task = value["task_instance"]
        if not isinstance(task, Mapping):
            raise TaskTaxonomyError("task_instance must be an object")
        family = task.get("skill_family_id")
        if family == PICK_AND_PLACE_SKILL_ID:
            task_instance: AnyTaskInstanceSpec = TaskInstanceSpec.from_dict(task)
        elif family == PUSH_TO_REGION_SKILL_ID:
            task_instance = PushTaskInstanceSpec.from_dict(task)
        else:
            raise TaskTaxonomyError(f"unknown skill family {family!r}")
        return cls(task_instance=task_instance, scene_seed=cast(int, value["scene_seed"]))


def canonical_pick_and_place_tasks() -> tuple[TaskInstanceSpec, ...]:
    """Return the six legacy combinations as instances of one skill family."""

    result = tuple(TaskInstanceSpec.from_task_spec(item) for item in CANONICAL_TASK_SPECS)
    expected = {
        (object_id.removesuffix("_cube"), bin_id) for object_id in OBJECT_IDS for bin_id in BIN_IDS
    }
    actual = {(item.object_attribute, item.target_region) for item in result}
    if actual != expected or len(result) != 6:
        raise TaskTaxonomyError("canonical M1 tasks do not form the expected 3 x 2 product")
    return result


def canonical_push_to_region_tasks() -> tuple[PushTaskInstanceSpec, ...]:
    """Return the fixed 2 x 4 x 2 pushing task product in canonical order."""

    result = tuple(
        PushTaskInstanceSpec.from_task_spec(
            PushTaskSpec(
                target_object_id=object_id,
                target_region_id=region_id,
                difficulty=difficulty,
            )
        )
        for object_id in PUSH_OBJECT_IDS
        for region_id in TARGET_REGION_IDS
        for difficulty in PUSH_DIFFICULTIES
    )
    if len(result) != 16 or len({item.canonical_task_id for item in result}) != 16:
        raise TaskTaxonomyError("canonical push tasks do not form the expected 2 x 4 x 2 product")
    return result


def load_task_catalog(
    value: Mapping[str, Any],
) -> tuple[SkillFamilySpec, tuple[TaskInstanceSpec, ...]]:
    """Load the repository-controlled catalog without accepting hidden defaults."""

    if set(value) != {"schema_version", "skill_family", "task_instances"}:
        raise TaskTaxonomyError("task catalog fields are invalid")
    if value["schema_version"] != TAXONOMY_SCHEMA_VERSION:
        raise TaskTaxonomyError("task catalog schema version changed")
    family_value = value["skill_family"]
    tasks_value = value["task_instances"]
    if not isinstance(family_value, Mapping) or not isinstance(tasks_value, list):
        raise TaskTaxonomyError("task catalog has malformed family or task instances")
    family = SkillFamilySpec.from_dict(family_value)
    tasks = tuple(
        TaskInstanceSpec.from_dict(item) for item in tasks_value if isinstance(item, Mapping)
    )
    if len(tasks) != len(tasks_value) or tasks != canonical_pick_and_place_tasks():
        raise TaskTaxonomyError("task catalog must contain the canonical six instances in order")
    if family != PICK_AND_PLACE_SKILL:
        raise TaskTaxonomyError("task catalog changed the pick_and_place family")
    return family, tasks


def load_push_task_catalog(
    value: Mapping[str, Any],
) -> tuple[SkillFamilySpec, tuple[PushTaskInstanceSpec, ...]]:
    """Load the exact repository-controlled pushing catalog."""

    if set(value) != {"schema_version", "skill_family", "task_instances"}:
        raise TaskTaxonomyError("task catalog fields are invalid")
    if value["schema_version"] != TAXONOMY_SCHEMA_VERSION:
        raise TaskTaxonomyError("task catalog schema version changed")
    family_value = value["skill_family"]
    tasks_value = value["task_instances"]
    if not isinstance(family_value, Mapping) or not isinstance(tasks_value, list):
        raise TaskTaxonomyError("task catalog has malformed family or task instances")
    family = SkillFamilySpec.from_dict(family_value)
    tasks = tuple(
        PushTaskInstanceSpec.from_dict(item) for item in tasks_value if isinstance(item, Mapping)
    )
    if len(tasks) != len(tasks_value) or tasks != canonical_push_to_region_tasks():
        raise TaskTaxonomyError("push catalog must contain the canonical 16 instances in order")
    if family != PUSH_TO_REGION_SKILL:
        raise TaskTaxonomyError("task catalog changed the push_to_region family")
    return family, tasks


def load_any_task_catalog(
    value: Mapping[str, Any],
) -> tuple[SkillFamilySpec, tuple[AnyTaskInstanceSpec, ...]]:
    """Dispatch one exact catalog by its declared skill-family identity."""

    family = value.get("skill_family")
    if not isinstance(family, Mapping):
        raise TaskTaxonomyError("task catalog has malformed skill_family")
    family_id = family.get("skill_family_id")
    if family_id == PICK_AND_PLACE_SKILL_ID:
        return load_task_catalog(value)
    if family_id == PUSH_TO_REGION_SKILL_ID:
        return load_push_task_catalog(value)
    raise TaskTaxonomyError(f"unknown skill family {family_id!r}")


__all__ = [
    "PUSH_ENVIRONMENT_ID",
    "PUSH_FAILURE_PREDICATES",
    "PUSH_TO_REGION_SKILL",
    "PUSH_TO_REGION_SKILL_ID",
    "PICK_AND_PLACE_SKILL",
    "PICK_AND_PLACE_SKILL_ID",
    "TAXONOMY_SCHEMA_VERSION",
    "EvaluationTask",
    "AnyTaskInstanceSpec",
    "PushTaskInstanceSpec",
    "SkillFamilySpec",
    "TaskInstanceSpec",
    "TaskTaxonomyError",
    "canonical_pick_and_place_tasks",
    "canonical_push_to_region_tasks",
    "load_any_task_catalog",
    "load_push_task_catalog",
    "load_task_catalog",
]
