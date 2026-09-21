"""Immutable, JSON-ready contracts for deterministic M3A raw collections."""

from __future__ import annotations

import math
import re
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Self, cast

from langmani.datasets.identity import (
    COLLECTION_SCHEMA_VERSION,
    environment_version,
    expert_config_fingerprint,
    sha256_hex,
    stable_accepted_scene_group_id,
    stable_attempt_id,
    stable_candidate_scene_id,
    stable_collection_run_id,
    stable_raw_trajectory_id,
    stable_scheduled_episode_id,
    stable_source_shard_id,
)
from langmani.environments.specs import (
    BIN_IDS,
    OBJECT_IDS,
    EpisodeSpec,
    TaskSpec,
    stable_scene_id,
    stable_task_id,
)
from langmani.experts.runtime import (
    PLANNER_RUNTIME_MODULES,
    query_planner_runtime_versions,
)
from langmani.experts.types import (
    ExpertConfig,
    ExpertPhase,
    ExpertResult,
    ExpertStatus,
    PhaseResult,
)

RUNTIME_VERSION_KEYS = frozenset(PLANNER_RUNTIME_MODULES)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ReplayValidationMode(StrEnum):
    """Supported single-environment replay validation depths."""

    ACTION = "action"
    ACTION_AND_STATE_AUDIT = "action_and_state_audit"


class ReplayFailureCode(StrEnum):
    """Stable machine-readable classifications for one replay failure."""

    RECORDED_TRAJECTORY_INVALID = "recorded_trajectory_invalid"
    INITIALIZATION_FAILURE = "initialization_failure"
    TASK_SPEC_MISMATCH = "task_spec_mismatch"
    INITIAL_STATE_MISMATCH = "initial_state_mismatch"
    ACTION_OUT_OF_BOUNDS = "action_out_of_bounds"
    ACTION_EXECUTION_FAILURE = "action_execution_failure"
    ACTION_COUNT_MISMATCH = "action_count_mismatch"
    RECORDED_SUCCESS_INVALID = "recorded_success_invalid"
    FINAL_SUCCESS_FAILURE = "final_success_failure"
    WRONG_OBJECT_IN_TARGET_BIN = "wrong_object_in_target_bin"
    TARGET_IN_WRONG_BIN = "target_in_wrong_bin"
    TARGET_OFF_TABLE = "target_off_table"
    FINAL_STATE_COMPARISON_FAILURE = "final_state_comparison_failure"
    FINAL_POSITION_TOLERANCE = "final_position_tolerance"
    FINAL_ORIENTATION_TOLERANCE = "final_orientation_tolerance"
    FINAL_JOINT_TOLERANCE = "final_joint_tolerance"
    STATE_AUDIT_FAILURE = "state_audit_failure"
    ENVIRONMENT_CLOSE_FAILURE = "environment_close_failure"


class OverwritePolicy(StrEnum):
    """Behavior when an incompatible collection already occupies the output root."""

    ERROR = "error"
    REPLACE = "replace"


class ResumePolicy(StrEnum):
    """Behavior when a compatible, incomplete manifest is present."""

    ERROR = "error"
    RESUME = "resume"


class AttemptOutcome(StrEnum):
    """Stable classification of one expert/record/replay attempt."""

    ACCEPTED = "accepted"
    EXPERT_FAILURE = "expert_failure"
    EXPERT_CONTRACT_FAILURE = "expert_contract_failure"
    RECORDING_FAILURE = "recording_failure"
    STRUCTURAL_FAILURE = "structural_failure"
    REPLAY_FAILURE = "replay_failure"
    GROUP_REJECTED = "group_rejected"
    UNEXPECTED_EXCEPTION = "unexpected_exception"


class CollectionStatus(StrEnum):
    """Persistent collection-run lifecycle state."""

    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool")
    return value


def _require_int(value: object, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < minimum:
        qualifier = "positive" if minimum == 1 else f">= {minimum}"
        raise ValueError(f"{field_name} must be {qualifier}")
    return value


def _require_float(value: object, field_name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    if positive and normalized <= 0:
        raise ValueError(f"{field_name} must be positive")
    if not positive and normalized < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return normalized


def _require_string(value: object, field_name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        suffix = " or None" if optional else ""
        raise TypeError(f"{field_name} must be a non-empty string{suffix}")
    return value


def _require_string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not all(isinstance(item, str) and item for item in value):
        raise TypeError(f"{field_name} must be a tuple of non-empty strings")
    return value


def _enum(value: object, enum_type: type[StrEnum], field_name: str) -> StrEnum:
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or {enum_type.__name__}")
    try:
        return enum_type(value)
    except ValueError as error:
        raise ValueError(f"unknown {field_name} {value!r}") from error


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field_name} keys must be strings")
    return cast(Mapping[str, object], value)


def _exact_fields(value: Mapping[str, object], expected: set[str], type_name: str) -> None:
    fields = set(value)
    missing = sorted(expected - fields)
    extra = sorted(fields - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing fields: " + ", ".join(missing))
        if extra:
            details.append("unexpected fields: " + ", ".join(extra))
        raise ValueError(f"malformed {type_name} (" + "; ".join(details) + ")")


def _enum_value(value: StrEnum) -> str:
    return value.value


def _string_tuple_from_json(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise TypeError(f"{field_name} must be a list of non-empty strings")
    return tuple(value)


def _bool_mapping(value: object, field_name: str) -> Mapping[str, bool]:
    source = _mapping(value, field_name)
    result: dict[str, bool] = {}
    for key, item in source.items():
        result[key] = _require_bool(item, f"{field_name}[{key!r}]")
    return MappingProxyType(result)


def _count_mapping(value: object, field_name: str) -> Mapping[str, int]:
    source = _mapping(value, field_name)
    result: dict[str, int] = {}
    for key, item in source.items():
        result[key] = _require_int(item, f"{field_name}[{key!r}]")
    return MappingProxyType(result)


def _runtime_versions(value: object) -> Mapping[str, str | None]:
    source = _mapping(value, "runtime_versions")
    missing = sorted(RUNTIME_VERSION_KEYS - set(source))
    if missing:
        raise ValueError("runtime_versions missing fields: " + ", ".join(missing))
    result: dict[str, str | None] = {}
    for key, item in source.items():
        if item is not None and (not isinstance(item, str) or not item):
            raise TypeError(f"runtime_versions[{key!r}] must be a non-empty string or None")
        result[key] = cast(str | None, item)
    return MappingProxyType(dict(sorted(result.items())))


def _default_runtime_versions() -> Mapping[str, str | None]:
    return detected_runtime_versions()


def detected_runtime_versions() -> dict[str, str | None]:
    """Capture effective imported modules used for collection/replay provenance."""
    return query_planner_runtime_versions(sys.executable)


def _parse_expert_result(value: object) -> ExpertResult:
    payload = _mapping(value, "expert_result")
    phase_results_value = payload.get("phase_results")
    if not isinstance(phase_results_value, list):
        raise TypeError("expert_result.phase_results must be a list")
    phase_results = tuple(_parse_phase_result(item) for item in phase_results_value)
    completed_value = payload.get("completed_phases")
    if not isinstance(completed_value, list):
        raise TypeError("expert_result.completed_phases must be a list")
    completed = tuple(
        cast(ExpertPhase, _enum(item, ExpertPhase, "completed_phase")) for item in completed_value
    )
    failed_value = payload.get("failed_phase")
    failed = (
        None
        if failed_value is None
        else cast(ExpertPhase, _enum(failed_value, ExpertPhase, "failed_phase"))
    )
    artifact_paths = payload.get("diagnostic_artifact_paths")
    if not isinstance(artifact_paths, list):
        raise TypeError("expert_result.diagnostic_artifact_paths must be a list")
    return ExpertResult(
        success=cast(bool, payload.get("success")),
        status=cast(ExpertStatus, _enum(payload.get("status"), ExpertStatus, "status")),
        scene_seed=cast(int | None, payload.get("scene_seed")),
        scene_id=cast(str | None, payload.get("scene_id")),
        task_id=cast(str | None, payload.get("task_id")),
        canonical_instruction=cast(str | None, payload.get("canonical_instruction")),
        target_object_id=cast(str | None, payload.get("target_object_id")),
        target_bin_id=cast(str | None, payload.get("target_bin_id")),
        total_environment_steps=cast(int, payload.get("total_environment_steps")),
        total_planning_calls=cast(int, payload.get("total_planning_calls")),
        total_replans=cast(int, payload.get("total_replans")),
        completed_phases=completed,
        failed_phase=failed,
        final_environment_evaluation=cast(
            Mapping[str, bool], payload.get("final_environment_evaluation")
        ),
        phase_results=phase_results,
        planning_duration_seconds=cast(float, payload.get("planning_duration_seconds")),
        execution_duration_seconds=cast(float, payload.get("execution_duration_seconds")),
        diagnostic_artifact_paths=tuple(cast(list[str], artifact_paths)),
        exception_type=cast(str | None, payload.get("exception_type")),
        exception_message=cast(str | None, payload.get("exception_message")),
    )


def _parse_phase_result(value: object) -> PhaseResult:
    payload = _mapping(value, "phase_result")
    return PhaseResult(
        phase=cast(ExpertPhase, _enum(payload.get("phase"), ExpertPhase, "phase")),
        success=cast(bool, payload.get("success")),
        status=cast(ExpertStatus, _enum(payload.get("status"), ExpertStatus, "status")),
        attempts=cast(int, payload.get("attempts")),
        environment_steps=cast(int, payload.get("environment_steps")),
        planning_calls=cast(int, payload.get("planning_calls")),
        replans=cast(int, payload.get("replans")),
        planning_duration_seconds=cast(float, payload.get("planning_duration_seconds")),
        execution_duration_seconds=cast(float, payload.get("execution_duration_seconds")),
        message=cast(str, payload.get("message")),
        planner_status=cast(str | None, payload.get("planner_status")),
    )


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    """All explicit bounds and policies for one reproducible raw collection."""

    environment_id: str = "LangMani-PickPlaceByInstruction-v0"
    control_mode: str = "pd_joint_pos"
    sim_backend: str = "physx_cpu"
    candidate_scene_seed_start: int = 0
    target_complete_scene_count: int = 60
    maximum_candidate_scene_count: int = 120
    maximum_expert_attempts_per_task: int = 3
    raw_output_root: str = "outputs/datasets/m3a"
    shard_size: int = 60
    retain_failed_raw_trajectories: bool = False
    replay_validation_mode: ReplayValidationMode = ReplayValidationMode.ACTION_AND_STATE_AUDIT
    final_position_tolerance_m: float = 0.005
    final_orientation_tolerance_rad: float = 0.05
    final_joint_tolerance_rad: float = 0.05
    overwrite_policy: OverwritePolicy = OverwritePolicy.ERROR
    resume_policy: ResumePolicy = ResumePolicy.RESUME
    collection_schema_version: str = COLLECTION_SCHEMA_VERSION
    expert_config: ExpertConfig = field(default_factory=ExpertConfig)
    runtime_versions: Mapping[str, str | None] = field(default_factory=_default_runtime_versions)

    def __post_init__(self) -> None:
        _require_string(self.environment_id, "environment_id")
        if self.environment_id != "LangMani-PickPlaceByInstruction-v0":
            raise ValueError("environment_id must be LangMani-PickPlaceByInstruction-v0")
        if self.control_mode != "pd_joint_pos":
            raise ValueError("control_mode must be pd_joint_pos")
        if self.sim_backend not in ("physx_cpu", "physx_cuda"):
            raise ValueError("sim_backend must be 'physx_cpu' or 'physx_cuda'")
        _require_int(self.candidate_scene_seed_start, "candidate_scene_seed_start")
        _require_int(self.target_complete_scene_count, "target_complete_scene_count", minimum=1)
        _require_int(
            self.maximum_candidate_scene_count,
            "maximum_candidate_scene_count",
            minimum=1,
        )
        if self.maximum_candidate_scene_count < self.target_complete_scene_count:
            raise ValueError(
                "maximum_candidate_scene_count cannot be smaller than target_complete_scene_count"
            )
        _require_int(
            self.maximum_expert_attempts_per_task,
            "maximum_expert_attempts_per_task",
            minimum=1,
        )
        _require_string(self.raw_output_root, "raw_output_root")
        _require_int(self.shard_size, "shard_size", minimum=1)
        if self.shard_size % 6:
            raise ValueError("shard_size must be a multiple of six to preserve scene groups")
        _require_bool(self.retain_failed_raw_trajectories, "retain_failed_raw_trajectories")
        object.__setattr__(
            self,
            "replay_validation_mode",
            _enum(
                self.replay_validation_mode,
                ReplayValidationMode,
                "replay_validation_mode",
            ),
        )
        for field_name in (
            "final_position_tolerance_m",
            "final_orientation_tolerance_rad",
            "final_joint_tolerance_rad",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_float(getattr(self, field_name), field_name, positive=True),
            )
        object.__setattr__(
            self,
            "overwrite_policy",
            _enum(self.overwrite_policy, OverwritePolicy, "overwrite_policy"),
        )
        object.__setattr__(
            self,
            "resume_policy",
            _enum(self.resume_policy, ResumePolicy, "resume_policy"),
        )
        if self.collection_schema_version != COLLECTION_SCHEMA_VERSION:
            raise ValueError(f"collection_schema_version must be {COLLECTION_SCHEMA_VERSION!r}")
        if not isinstance(self.expert_config, ExpertConfig):
            raise TypeError("expert_config must be an ExpertConfig")
        if self.expert_config.control_mode != self.control_mode:
            raise ValueError("expert_config control mode must match collection control_mode")
        object.__setattr__(self, "runtime_versions", _runtime_versions(self.runtime_versions))

    @property
    def expert_config_fingerprint(self) -> str:
        return expert_config_fingerprint(self.expert_config)

    def identity_dict(self) -> dict[str, Any]:
        """Return content-affecting fields used to identify/resume a run."""
        payload = self.to_dict()
        for field_name in ("raw_output_root", "overwrite_policy", "resume_policy"):
            payload.pop(field_name)
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_id": self.environment_id,
            "control_mode": self.control_mode,
            "sim_backend": self.sim_backend,
            "candidate_scene_seed_start": self.candidate_scene_seed_start,
            "target_complete_scene_count": self.target_complete_scene_count,
            "maximum_candidate_scene_count": self.maximum_candidate_scene_count,
            "maximum_expert_attempts_per_task": self.maximum_expert_attempts_per_task,
            "raw_output_root": self.raw_output_root,
            "shard_size": self.shard_size,
            "retain_failed_raw_trajectories": self.retain_failed_raw_trajectories,
            "replay_validation_mode": self.replay_validation_mode.value,
            "final_position_tolerance_m": self.final_position_tolerance_m,
            "final_orientation_tolerance_rad": self.final_orientation_tolerance_rad,
            "final_joint_tolerance_rad": self.final_joint_tolerance_rad,
            "overwrite_policy": self.overwrite_policy.value,
            "resume_policy": self.resume_policy.value,
            "collection_schema_version": self.collection_schema_version,
            "expert_config": self.expert_config.to_dict(),
            "expert_config_fingerprint": self.expert_config_fingerprint,
            "runtime_versions": dict(self.runtime_versions),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = _mapping(value, "collection_config")
        expected = {
            "environment_id",
            "control_mode",
            "sim_backend",
            "candidate_scene_seed_start",
            "target_complete_scene_count",
            "maximum_candidate_scene_count",
            "maximum_expert_attempts_per_task",
            "raw_output_root",
            "shard_size",
            "retain_failed_raw_trajectories",
            "replay_validation_mode",
            "final_position_tolerance_m",
            "final_orientation_tolerance_rad",
            "final_joint_tolerance_rad",
            "overwrite_policy",
            "resume_policy",
            "collection_schema_version",
            "expert_config",
            "expert_config_fingerprint",
            "runtime_versions",
        }
        _exact_fields(payload, expected, "collection_config")
        expert_payload = _mapping(payload["expert_config"], "expert_config")
        config = cls(
            environment_id=cast(str, payload["environment_id"]),
            control_mode=cast(str, payload["control_mode"]),
            sim_backend=cast(str, payload["sim_backend"]),
            candidate_scene_seed_start=cast(int, payload["candidate_scene_seed_start"]),
            target_complete_scene_count=cast(int, payload["target_complete_scene_count"]),
            maximum_candidate_scene_count=cast(int, payload["maximum_candidate_scene_count"]),
            maximum_expert_attempts_per_task=cast(int, payload["maximum_expert_attempts_per_task"]),
            raw_output_root=cast(str, payload["raw_output_root"]),
            shard_size=cast(int, payload["shard_size"]),
            retain_failed_raw_trajectories=cast(bool, payload["retain_failed_raw_trajectories"]),
            replay_validation_mode=cast(
                ReplayValidationMode,
                _enum(
                    payload["replay_validation_mode"],
                    ReplayValidationMode,
                    "replay_validation_mode",
                ),
            ),
            final_position_tolerance_m=cast(float, payload["final_position_tolerance_m"]),
            final_orientation_tolerance_rad=cast(float, payload["final_orientation_tolerance_rad"]),
            final_joint_tolerance_rad=cast(float, payload["final_joint_tolerance_rad"]),
            overwrite_policy=cast(
                OverwritePolicy,
                _enum(payload["overwrite_policy"], OverwritePolicy, "overwrite_policy"),
            ),
            resume_policy=cast(
                ResumePolicy,
                _enum(payload["resume_policy"], ResumePolicy, "resume_policy"),
            ),
            collection_schema_version=cast(str, payload["collection_schema_version"]),
            expert_config=ExpertConfig(**dict(expert_payload)),
            runtime_versions=cast(Mapping[str, str | None], payload["runtime_versions"]),
        )
        if payload["expert_config_fingerprint"] != config.expert_config_fingerprint:
            raise ValueError("expert_config_fingerprint does not match expert_config")
        return config


@dataclass(frozen=True, slots=True)
class ScheduledEpisode:
    """One deterministic task request in the candidate-seed schedule."""

    collection_run_id: str
    candidate_scene_index: int
    candidate_scene_id: str
    scene_seed: int
    scene_id: str
    task_index: int
    task_spec: TaskSpec
    task_id: str
    canonical_instruction: str
    environment_id: str
    environment_version: str
    control_mode: str
    collection_schema_version: str
    expert_config_fingerprint: str
    scheduled_episode_id: str

    def __post_init__(self) -> None:
        for field_name in (
            "collection_run_id",
            "candidate_scene_id",
            "scene_id",
            "task_id",
            "canonical_instruction",
            "environment_id",
            "environment_version",
            "control_mode",
            "collection_schema_version",
            "expert_config_fingerprint",
            "scheduled_episode_id",
        ):
            _require_string(getattr(self, field_name), field_name)
        _require_int(self.candidate_scene_index, "candidate_scene_index")
        _require_int(self.scene_seed, "scene_seed")
        _require_int(self.task_index, "task_index")
        if self.task_index >= 6:
            raise ValueError("task_index must be in [0, 5]")
        if not isinstance(self.task_spec, TaskSpec):
            raise TypeError("task_spec must be a TaskSpec")
        episode = EpisodeSpec.create(scene_seed=self.scene_seed, task_spec=self.task_spec)
        if (self.scene_id, self.task_id, self.canonical_instruction) != (
            episode.scene_id,
            episode.task_id,
            episode.canonical_instruction,
        ):
            raise ValueError("scheduled episode semantic metadata is inconsistent")
        if self.environment_version != environment_version(self.environment_id):
            raise ValueError("environment_version is inconsistent with environment_id")
        expected_candidate_id = stable_candidate_scene_id(
            environment_id=self.environment_id,
            scene_seed=self.scene_seed,
            scene_id=self.scene_id,
            expert_fingerprint=self.expert_config_fingerprint,
            control_mode=self.control_mode,
            schema_version=self.collection_schema_version,
        )
        if self.candidate_scene_id != expected_candidate_id:
            raise ValueError("candidate_scene_id is inconsistent with semantic inputs")
        expected_episode_id = stable_scheduled_episode_id(
            environment_id=self.environment_id,
            scene_seed=self.scene_seed,
            scene_id=self.scene_id,
            task_spec=self.task_spec,
            task_id=self.task_id,
            expert_fingerprint=self.expert_config_fingerprint,
            control_mode=self.control_mode,
            schema_version=self.collection_schema_version,
        )
        if self.scheduled_episode_id != expected_episode_id:
            raise ValueError("scheduled_episode_id is inconsistent with semantic inputs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection_run_id": self.collection_run_id,
            "candidate_scene_index": self.candidate_scene_index,
            "candidate_scene_id": self.candidate_scene_id,
            "scene_seed": self.scene_seed,
            "scene_id": self.scene_id,
            "task_index": self.task_index,
            "task_spec": self.task_spec.to_dict(),
            "task_id": self.task_id,
            "canonical_instruction": self.canonical_instruction,
            "environment_id": self.environment_id,
            "environment_version": self.environment_version,
            "control_mode": self.control_mode,
            "collection_schema_version": self.collection_schema_version,
            "expert_config_fingerprint": self.expert_config_fingerprint,
            "scheduled_episode_id": self.scheduled_episode_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = _mapping(value, "scheduled_episode")
        expected = set(cls.__dataclass_fields__)
        _exact_fields(payload, expected, "scheduled_episode")
        return cls(
            collection_run_id=cast(str, payload["collection_run_id"]),
            candidate_scene_index=cast(int, payload["candidate_scene_index"]),
            candidate_scene_id=cast(str, payload["candidate_scene_id"]),
            scene_seed=cast(int, payload["scene_seed"]),
            scene_id=cast(str, payload["scene_id"]),
            task_index=cast(int, payload["task_index"]),
            task_spec=TaskSpec.from_mapping(_mapping(payload["task_spec"], "task_spec")),
            task_id=cast(str, payload["task_id"]),
            canonical_instruction=cast(str, payload["canonical_instruction"]),
            environment_id=cast(str, payload["environment_id"]),
            environment_version=cast(str, payload["environment_version"]),
            control_mode=cast(str, payload["control_mode"]),
            collection_schema_version=cast(str, payload["collection_schema_version"]),
            expert_config_fingerprint=cast(str, payload["expert_config_fingerprint"]),
            scheduled_episode_id=cast(str, payload["scheduled_episode_id"]),
        )


@dataclass(frozen=True, slots=True)
class CollectionSchedule:
    """Complete, deterministic seed-major/object-major collection schedule."""

    collection_run_id: str
    collection_schema_version: str
    config_fingerprint: str
    canonical_task_specs: tuple[TaskSpec, ...]
    candidate_scene_seeds: tuple[int, ...]
    episodes: tuple[ScheduledEpisode, ...]

    def __post_init__(self) -> None:
        _require_string(self.collection_run_id, "collection_run_id")
        _require_string(self.collection_schema_version, "collection_schema_version")
        _require_string(self.config_fingerprint, "config_fingerprint")
        if not isinstance(self.canonical_task_specs, tuple) or len(self.canonical_task_specs) != 6:
            raise ValueError("canonical_task_specs must contain exactly six tasks")
        if not all(isinstance(task, TaskSpec) for task in self.canonical_task_specs):
            raise TypeError("canonical_task_specs must contain TaskSpec values")
        if len(set(self.canonical_task_specs)) != 6:
            raise ValueError("canonical_task_specs must be unique")
        if not isinstance(self.candidate_scene_seeds, tuple) or not all(
            isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0
            for seed in self.candidate_scene_seeds
        ):
            raise TypeError("candidate_scene_seeds must be non-negative integer tuple")
        if len(set(self.candidate_scene_seeds)) != len(self.candidate_scene_seeds):
            raise ValueError("candidate_scene_seeds must be unique")
        if not isinstance(self.episodes, tuple) or not all(
            isinstance(episode, ScheduledEpisode) for episode in self.episodes
        ):
            raise TypeError("episodes must be a tuple of ScheduledEpisode values")
        if len(self.episodes) != 6 * len(self.candidate_scene_seeds):
            raise ValueError("schedule must contain exactly six episodes per candidate scene")
        for flat_index, episode in enumerate(self.episodes):
            scene_index, task_index = divmod(flat_index, 6)
            if episode.collection_run_id != self.collection_run_id:
                raise ValueError("episode collection_run_id differs from schedule")
            if episode.candidate_scene_index != scene_index or episode.task_index != task_index:
                raise ValueError("episodes must use seed-major, canonical task order")
            if episode.scene_seed != self.candidate_scene_seeds[scene_index]:
                raise ValueError("episode scene seed differs from schedule")
            if episode.task_spec != self.canonical_task_specs[task_index]:
                raise ValueError("episode task differs from canonical task order")

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection_run_id": self.collection_run_id,
            "collection_schema_version": self.collection_schema_version,
            "config_fingerprint": self.config_fingerprint,
            "canonical_task_specs": [task.to_dict() for task in self.canonical_task_specs],
            "candidate_scene_seeds": list(self.candidate_scene_seeds),
            "episodes": [episode.to_dict() for episode in self.episodes],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = _mapping(value, "collection_schedule")
        _exact_fields(payload, set(cls.__dataclass_fields__), "collection_schedule")
        tasks_value = payload["canonical_task_specs"]
        seeds_value = payload["candidate_scene_seeds"]
        episodes_value = payload["episodes"]
        if not isinstance(tasks_value, list):
            raise TypeError("canonical_task_specs must be a list")
        if not isinstance(seeds_value, list):
            raise TypeError("candidate_scene_seeds must be a list")
        if not isinstance(episodes_value, list):
            raise TypeError("episodes must be a list")
        return cls(
            collection_run_id=cast(str, payload["collection_run_id"]),
            collection_schema_version=cast(str, payload["collection_schema_version"]),
            config_fingerprint=cast(str, payload["config_fingerprint"]),
            canonical_task_specs=tuple(
                TaskSpec.from_mapping(_mapping(task, "task_spec")) for task in tasks_value
            ),
            candidate_scene_seeds=tuple(cast(list[int], seeds_value)),
            episodes=tuple(
                ScheduledEpisode.from_dict(_mapping(item, "scheduled_episode"))
                for item in episodes_value
            ),
        )


@dataclass(frozen=True, slots=True)
class ReplayValidationResult:
    """Compact evidence from deterministic action replay and optional state audit."""

    passed: bool
    mode: ReplayValidationMode
    recorded_success: bool
    replay_success: bool
    task_spec_matches: bool
    wrong_object_in_target_bin: bool
    target_in_wrong_bin: bool
    target_off_table: bool
    final_environment_evaluation: Mapping[str, bool]
    recorded_action_steps: int
    replayed_action_steps: int
    final_position_error_m: float | None
    final_orientation_error_rad: float | None
    final_joint_error_rad: float | None
    state_audit_performed: bool
    state_audit_passed: bool | None
    failure_codes: tuple[ReplayFailureCode, ...] = ()
    failure_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "passed",
            "recorded_success",
            "replay_success",
            "task_spec_matches",
            "wrong_object_in_target_bin",
            "target_in_wrong_bin",
            "target_off_table",
            "state_audit_performed",
        ):
            _require_bool(getattr(self, field_name), field_name)
        object.__setattr__(
            self,
            "mode",
            _enum(self.mode, ReplayValidationMode, "mode"),
        )
        object.__setattr__(
            self,
            "final_environment_evaluation",
            _bool_mapping(self.final_environment_evaluation, "final_environment_evaluation"),
        )
        _require_int(self.recorded_action_steps, "recorded_action_steps", minimum=1)
        _require_int(self.replayed_action_steps, "replayed_action_steps")
        for field_name in (
            "final_position_error_m",
            "final_orientation_error_rad",
            "final_joint_error_rad",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _require_float(value, field_name),
                )
        if self.state_audit_passed is not None:
            _require_bool(self.state_audit_passed, "state_audit_passed")
        if self.state_audit_performed != (self.state_audit_passed is not None):
            raise ValueError(
                "state_audit_passed must be set exactly when state_audit_performed is true"
            )
        if self.mode is ReplayValidationMode.ACTION and self.state_audit_performed:
            raise ValueError("action-only replay mode cannot report a state audit")
        if (
            self.mode is ReplayValidationMode.ACTION_AND_STATE_AUDIT
            and not self.state_audit_performed
        ):
            raise ValueError("action_and_state_audit mode requires a state audit result")
        object.__setattr__(
            self,
            "failure_codes",
            tuple(_enum(code, ReplayFailureCode, "failure_codes") for code in self.failure_codes),
        )
        _require_string_tuple(self.failure_reasons, "failure_reasons")
        if bool(self.failure_codes) != bool(self.failure_reasons):
            raise ValueError(
                "failure_codes and failure_reasons must either both be empty or both be non-empty"
            )

        required_pass = (
            self.recorded_success
            and self.replay_success
            and self.task_spec_matches
            and not self.wrong_object_in_target_bin
            and not self.target_in_wrong_bin
            and not self.target_off_table
            and self.recorded_action_steps == self.replayed_action_steps
            and self.final_environment_evaluation.get("success") is True
            and self.final_environment_evaluation.get("fail") is False
            and (self.state_audit_passed is not False)
            and not self.failure_codes
            and not self.failure_reasons
        )
        if self.passed != required_pass:
            raise ValueError("passed is inconsistent with replay evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "mode": self.mode.value,
            "recorded_success": self.recorded_success,
            "replay_success": self.replay_success,
            "task_spec_matches": self.task_spec_matches,
            "wrong_object_in_target_bin": self.wrong_object_in_target_bin,
            "target_in_wrong_bin": self.target_in_wrong_bin,
            "target_off_table": self.target_off_table,
            "final_environment_evaluation": dict(self.final_environment_evaluation),
            "recorded_action_steps": self.recorded_action_steps,
            "replayed_action_steps": self.replayed_action_steps,
            "final_position_error_m": self.final_position_error_m,
            "final_orientation_error_rad": self.final_orientation_error_rad,
            "final_joint_error_rad": self.final_joint_error_rad,
            "state_audit_performed": self.state_audit_performed,
            "state_audit_passed": self.state_audit_passed,
            "failure_codes": [code.value for code in self.failure_codes],
            "failure_reasons": list(self.failure_reasons),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = _mapping(value, "replay_validation_result")
        _exact_fields(payload, set(cls.__dataclass_fields__), "replay_validation_result")
        return cls(
            passed=cast(bool, payload["passed"]),
            mode=cast(
                ReplayValidationMode,
                _enum(payload["mode"], ReplayValidationMode, "mode"),
            ),
            recorded_success=cast(bool, payload["recorded_success"]),
            replay_success=cast(bool, payload["replay_success"]),
            task_spec_matches=cast(bool, payload["task_spec_matches"]),
            wrong_object_in_target_bin=cast(bool, payload["wrong_object_in_target_bin"]),
            target_in_wrong_bin=cast(bool, payload["target_in_wrong_bin"]),
            target_off_table=cast(bool, payload["target_off_table"]),
            final_environment_evaluation=cast(
                Mapping[str, bool], payload["final_environment_evaluation"]
            ),
            recorded_action_steps=cast(int, payload["recorded_action_steps"]),
            replayed_action_steps=cast(int, payload["replayed_action_steps"]),
            final_position_error_m=cast(float | None, payload["final_position_error_m"]),
            final_orientation_error_rad=cast(float | None, payload["final_orientation_error_rad"]),
            final_joint_error_rad=cast(float | None, payload["final_joint_error_rad"]),
            state_audit_performed=cast(bool, payload["state_audit_performed"]),
            state_audit_passed=cast(bool | None, payload["state_audit_passed"]),
            failure_codes=tuple(
                cast(
                    ReplayFailureCode,
                    _enum(code, ReplayFailureCode, "failure_codes"),
                )
                for code in _list_of(payload["failure_codes"], "failure_codes")
            ),
            failure_reasons=_string_tuple_from_json(payload["failure_reasons"], "failure_reasons"),
        )


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    """Persistent small record for every bounded expert attempt, including failures."""

    attempt_id: str
    scheduled_episode_id: str
    attempt_index: int
    scene_seed: int
    scene_id: str
    task_spec: TaskSpec
    task_id: str
    outcome: AttemptOutcome
    expert_result: ExpertResult
    raw_trajectory_id: str | None = None
    replay_validation: ReplayValidationResult | None = None
    trajectory_retained: bool = False
    failure_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("attempt_id", "scheduled_episode_id", "scene_id", "task_id"):
            _require_string(getattr(self, field_name), field_name)
        _require_int(self.attempt_index, "attempt_index", minimum=1)
        _require_int(self.scene_seed, "scene_seed")
        if not isinstance(self.task_spec, TaskSpec):
            raise TypeError("task_spec must be a TaskSpec")
        if self.scene_id != stable_scene_id(self.scene_seed):
            raise ValueError("scene_id is inconsistent with scene_seed")
        if self.task_id != stable_task_id(self.task_spec):
            raise ValueError("task_id is inconsistent with task_spec")
        object.__setattr__(self, "outcome", _enum(self.outcome, AttemptOutcome, "outcome"))
        if not isinstance(self.expert_result, ExpertResult):
            raise TypeError("expert_result must be an ExpertResult")
        episode = EpisodeSpec.create(scene_seed=self.scene_seed, task_spec=self.task_spec)
        semantic_pairs = (
            (self.expert_result.scene_seed, episode.scene_seed),
            (self.expert_result.scene_id, episode.scene_id),
            (self.expert_result.task_id, episode.task_id),
            (self.expert_result.canonical_instruction, episode.canonical_instruction),
            (self.expert_result.target_object_id, episode.task_spec.target_object_id),
            (self.expert_result.target_bin_id, episode.task_spec.target_bin_id),
        )
        _require_string(self.raw_trajectory_id, "raw_trajectory_id", optional=True)
        if self.replay_validation is not None and not isinstance(
            self.replay_validation, ReplayValidationResult
        ):
            raise TypeError("replay_validation must be ReplayValidationResult or None")
        _require_bool(self.trajectory_retained, "trajectory_retained")
        _require_string_tuple(self.failure_reasons, "failure_reasons")

        if self.outcome is AttemptOutcome.ACCEPTED:
            if not self.expert_result.success:
                raise ValueError("an accepted attempt requires expert success")
            if any(actual is None for actual, _ in semantic_pairs):
                raise ValueError("an accepted attempt requires complete expert semantic identity")
            if any(actual != expected for actual, expected in semantic_pairs):
                raise ValueError("accepted expert_result semantic identity differs from attempt")
            if self.raw_trajectory_id is None or not self.trajectory_retained:
                raise ValueError("an accepted attempt requires a retained raw trajectory")
            if self.replay_validation is None or not self.replay_validation.passed:
                raise ValueError("an accepted attempt requires passed replay validation")
            if self.failure_reasons:
                raise ValueError("an accepted attempt cannot have failure_reasons")
        elif not self.failure_reasons:
            raise ValueError("a failed attempt requires at least one failure reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "scheduled_episode_id": self.scheduled_episode_id,
            "attempt_index": self.attempt_index,
            "scene_seed": self.scene_seed,
            "scene_id": self.scene_id,
            "task_spec": self.task_spec.to_dict(),
            "task_id": self.task_id,
            "outcome": self.outcome.value,
            "expert_result": self.expert_result.to_dict(),
            "raw_trajectory_id": self.raw_trajectory_id,
            "replay_validation": (
                self.replay_validation.to_dict() if self.replay_validation is not None else None
            ),
            "trajectory_retained": self.trajectory_retained,
            "failure_reasons": list(self.failure_reasons),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = _mapping(value, "attempt_record")
        _exact_fields(payload, set(cls.__dataclass_fields__), "attempt_record")
        replay = payload["replay_validation"]
        return cls(
            attempt_id=cast(str, payload["attempt_id"]),
            scheduled_episode_id=cast(str, payload["scheduled_episode_id"]),
            attempt_index=cast(int, payload["attempt_index"]),
            scene_seed=cast(int, payload["scene_seed"]),
            scene_id=cast(str, payload["scene_id"]),
            task_spec=TaskSpec.from_mapping(_mapping(payload["task_spec"], "task_spec")),
            task_id=cast(str, payload["task_id"]),
            outcome=cast(
                AttemptOutcome,
                _enum(payload["outcome"], AttemptOutcome, "outcome"),
            ),
            expert_result=_parse_expert_result(payload["expert_result"]),
            raw_trajectory_id=cast(str | None, payload["raw_trajectory_id"]),
            replay_validation=(
                None
                if replay is None
                else ReplayValidationResult.from_dict(_mapping(replay, "replay_validation"))
            ),
            trajectory_retained=cast(bool, payload["trajectory_retained"]),
            failure_reasons=_string_tuple_from_json(payload["failure_reasons"], "failure_reasons"),
        )


@dataclass(frozen=True, slots=True)
class RawEpisodeRecord:
    """Authoritative metadata locating one validated native ManiSkill episode."""

    scheduled_episode_id: str
    attempt_id: str
    raw_trajectory_id: str
    source_shard_id: str
    native_episode_id: int
    h5_group: str
    h5_path: str
    json_path: str
    h5_sha256: str
    json_sha256: str
    elapsed_steps: int
    scene_seed: int
    scene_id: str
    task_spec: TaskSpec
    task_id: str
    canonical_instruction: str
    expert_result: ExpertResult
    replay_validation: ReplayValidationResult
    final_environment_evaluation: Mapping[str, bool]
    structural_valid: bool
    accepted: bool

    def __post_init__(self) -> None:
        for field_name in (
            "scheduled_episode_id",
            "attempt_id",
            "raw_trajectory_id",
            "source_shard_id",
            "h5_group",
            "h5_path",
            "json_path",
            "scene_id",
            "task_id",
            "canonical_instruction",
        ):
            _require_string(getattr(self, field_name), field_name)
        _require_int(self.native_episode_id, "native_episode_id")
        if self.h5_group != f"traj_{self.native_episode_id}":
            raise ValueError("h5_group must equal traj_<native_episode_id>")
        for field_name in ("h5_sha256", "json_sha256"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
                raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
        _require_int(self.elapsed_steps, "elapsed_steps", minimum=1)
        _require_int(self.scene_seed, "scene_seed")
        if not isinstance(self.task_spec, TaskSpec):
            raise TypeError("task_spec must be a TaskSpec")
        episode = EpisodeSpec.create(scene_seed=self.scene_seed, task_spec=self.task_spec)
        if (self.scene_id, self.task_id, self.canonical_instruction) != (
            episode.scene_id,
            episode.task_id,
            episode.canonical_instruction,
        ):
            raise ValueError("raw episode semantic metadata is inconsistent")
        if not isinstance(self.expert_result, ExpertResult):
            raise TypeError("expert_result must be an ExpertResult")
        if self.expert_result.total_environment_steps != self.elapsed_steps:
            raise ValueError("expert_result step count differs from raw episode elapsed_steps")
        if not isinstance(self.replay_validation, ReplayValidationResult):
            raise TypeError("replay_validation must be a ReplayValidationResult")
        object.__setattr__(
            self,
            "final_environment_evaluation",
            _bool_mapping(self.final_environment_evaluation, "final_environment_evaluation"),
        )
        _require_bool(self.structural_valid, "structural_valid")
        _require_bool(self.accepted, "accepted")
        if self.accepted and not (
            self.structural_valid
            and self.expert_result.success
            and self.replay_validation.passed
            and self.final_environment_evaluation.get("success") is True
            and self.final_environment_evaluation.get("fail") is False
        ):
            raise ValueError("accepted raw episode lacks required success evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheduled_episode_id": self.scheduled_episode_id,
            "attempt_id": self.attempt_id,
            "raw_trajectory_id": self.raw_trajectory_id,
            "source_shard_id": self.source_shard_id,
            "native_episode_id": self.native_episode_id,
            "h5_group": self.h5_group,
            "h5_path": self.h5_path,
            "json_path": self.json_path,
            "h5_sha256": self.h5_sha256,
            "json_sha256": self.json_sha256,
            "elapsed_steps": self.elapsed_steps,
            "scene_seed": self.scene_seed,
            "scene_id": self.scene_id,
            "task_spec": self.task_spec.to_dict(),
            "task_id": self.task_id,
            "canonical_instruction": self.canonical_instruction,
            "expert_result": self.expert_result.to_dict(),
            "replay_validation": self.replay_validation.to_dict(),
            "final_environment_evaluation": dict(self.final_environment_evaluation),
            "structural_valid": self.structural_valid,
            "accepted": self.accepted,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = _mapping(value, "raw_episode_record")
        _exact_fields(payload, set(cls.__dataclass_fields__), "raw_episode_record")
        return cls(
            scheduled_episode_id=cast(str, payload["scheduled_episode_id"]),
            attempt_id=cast(str, payload["attempt_id"]),
            raw_trajectory_id=cast(str, payload["raw_trajectory_id"]),
            source_shard_id=cast(str, payload["source_shard_id"]),
            native_episode_id=cast(int, payload["native_episode_id"]),
            h5_group=cast(str, payload["h5_group"]),
            h5_path=cast(str, payload["h5_path"]),
            json_path=cast(str, payload["json_path"]),
            h5_sha256=cast(str, payload["h5_sha256"]),
            json_sha256=cast(str, payload["json_sha256"]),
            elapsed_steps=cast(int, payload["elapsed_steps"]),
            scene_seed=cast(int, payload["scene_seed"]),
            scene_id=cast(str, payload["scene_id"]),
            task_spec=TaskSpec.from_mapping(_mapping(payload["task_spec"], "task_spec")),
            task_id=cast(str, payload["task_id"]),
            canonical_instruction=cast(str, payload["canonical_instruction"]),
            expert_result=_parse_expert_result(payload["expert_result"]),
            replay_validation=ReplayValidationResult.from_dict(
                _mapping(payload["replay_validation"], "replay_validation")
            ),
            final_environment_evaluation=cast(
                Mapping[str, bool], payload["final_environment_evaluation"]
            ),
            structural_valid=cast(bool, payload["structural_valid"]),
            accepted=cast(bool, payload["accepted"]),
        )


@dataclass(frozen=True, slots=True)
class SourceShardRecord:
    """Checksummed native HDF5/JSON source-pair metadata."""

    source_shard_id: str
    shard_index: int
    h5_path: str
    json_path: str
    h5_sha256: str
    json_sha256: str
    h5_size_bytes: int
    json_size_bytes: int
    episode_count: int
    raw_trajectory_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in ("source_shard_id", "h5_path", "json_path"):
            _require_string(getattr(self, field_name), field_name)
        _require_int(self.shard_index, "shard_index")
        for field_name in ("h5_sha256", "json_sha256"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
                raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
        _require_int(self.h5_size_bytes, "h5_size_bytes", minimum=1)
        _require_int(self.json_size_bytes, "json_size_bytes", minimum=1)
        _require_int(self.episode_count, "episode_count", minimum=1)
        _require_string_tuple(self.raw_trajectory_ids, "raw_trajectory_ids")
        if self.episode_count != len(self.raw_trajectory_ids):
            raise ValueError("episode_count must equal len(raw_trajectory_ids)")
        if self.episode_count % 6:
            raise ValueError("accepted source shards must contain complete six-episode groups")
        if len(set(self.raw_trajectory_ids)) != self.episode_count:
            raise ValueError("raw_trajectory_ids must be unique within a shard")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_shard_id": self.source_shard_id,
            "shard_index": self.shard_index,
            "h5_path": self.h5_path,
            "json_path": self.json_path,
            "h5_sha256": self.h5_sha256,
            "json_sha256": self.json_sha256,
            "h5_size_bytes": self.h5_size_bytes,
            "json_size_bytes": self.json_size_bytes,
            "episode_count": self.episode_count,
            "raw_trajectory_ids": list(self.raw_trajectory_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = _mapping(value, "source_shard_record")
        _exact_fields(payload, set(cls.__dataclass_fields__), "source_shard_record")
        return cls(
            source_shard_id=cast(str, payload["source_shard_id"]),
            shard_index=cast(int, payload["shard_index"]),
            h5_path=cast(str, payload["h5_path"]),
            json_path=cast(str, payload["json_path"]),
            h5_sha256=cast(str, payload["h5_sha256"]),
            json_sha256=cast(str, payload["json_sha256"]),
            h5_size_bytes=cast(int, payload["h5_size_bytes"]),
            json_size_bytes=cast(int, payload["json_size_bytes"]),
            episode_count=cast(int, payload["episode_count"]),
            raw_trajectory_ids=_string_tuple_from_json(
                payload["raw_trajectory_ids"], "raw_trajectory_ids"
            ),
        )


@dataclass(frozen=True, slots=True)
class SceneGroupRecord:
    """All attempt and acceptance evidence for one counterfactual scene group."""

    candidate_scene_index: int
    candidate_scene_id: str
    accepted_scene_group_id: str | None
    scene_seed: int
    scene_id: str
    scheduled_episode_ids: tuple[str, ...]
    attempt_ids: tuple[str, ...]
    episodes: tuple[RawEpisodeRecord, ...]
    complete: bool
    accepted: bool
    bundle_h5_sha256: str | None = None
    bundle_json_sha256: str | None = None
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_int(self.candidate_scene_index, "candidate_scene_index")
        _require_string(self.candidate_scene_id, "candidate_scene_id")
        _require_string(
            self.accepted_scene_group_id,
            "accepted_scene_group_id",
            optional=True,
        )
        _require_int(self.scene_seed, "scene_seed")
        if self.scene_id != stable_scene_id(self.scene_seed):
            raise ValueError("scene_id is inconsistent with scene_seed")
        _require_string_tuple(self.scheduled_episode_ids, "scheduled_episode_ids")
        if len(self.scheduled_episode_ids) != 6 or len(set(self.scheduled_episode_ids)) != 6:
            raise ValueError("scheduled_episode_ids must contain exactly six unique IDs")
        _require_string_tuple(self.attempt_ids, "attempt_ids")
        if len(set(self.attempt_ids)) != len(self.attempt_ids):
            raise ValueError("attempt_ids must be unique")
        if not isinstance(self.episodes, tuple) or not all(
            isinstance(episode, RawEpisodeRecord) for episode in self.episodes
        ):
            raise TypeError("episodes must be a tuple of RawEpisodeRecord values")
        _require_bool(self.complete, "complete")
        _require_bool(self.accepted, "accepted")
        for field_name in ("bundle_h5_sha256", "bundle_json_sha256"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None
            ):
                raise ValueError(f"{field_name} must be None or a lowercase SHA-256 digest")
        if (self.bundle_h5_sha256 is None) != (self.bundle_json_sha256 is None):
            raise ValueError("scene-group bundle checksums must be provided together")
        _require_string_tuple(self.rejection_reasons, "rejection_reasons")
        if self.accepted != self.complete:
            raise ValueError("a scene group is accepted exactly when it is complete")
        if self.accepted:
            if self.accepted_scene_group_id is None:
                raise ValueError("accepted scene group requires accepted_scene_group_id")
            if self.bundle_h5_sha256 is None:
                raise ValueError("accepted scene group requires trusted bundle checksums")
            if len(self.episodes) != 6 or not all(episode.accepted for episode in self.episodes):
                raise ValueError("accepted scene group requires six accepted raw episodes")
            if tuple(episode.scheduled_episode_id for episode in self.episodes) != (
                self.scheduled_episode_ids
            ):
                raise ValueError("accepted episodes must follow scheduled task order")
            if len({episode.task_id for episode in self.episodes}) != 6:
                raise ValueError("accepted scene group must contain all six unique TaskSpecs")
            if any(
                episode.scene_seed != self.scene_seed or episode.scene_id != self.scene_id
                for episode in self.episodes
            ):
                raise ValueError("accepted episodes must share the group's physical scene")
            if self.rejection_reasons:
                raise ValueError("accepted scene group cannot have rejection_reasons")
        else:
            if self.accepted_scene_group_id is not None:
                raise ValueError("rejected scene group cannot have accepted_scene_group_id")
            if self.bundle_h5_sha256 is not None:
                raise ValueError("rejected scene group cannot have bundle checksums")
            if not self.rejection_reasons:
                raise ValueError("rejected scene group requires rejection_reasons")

    @property
    def scene_group_id(self) -> str:
        """Return the accepted ID when present, otherwise the candidate ID."""
        return self.accepted_scene_group_id or self.candidate_scene_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_scene_index": self.candidate_scene_index,
            "candidate_scene_id": self.candidate_scene_id,
            "accepted_scene_group_id": self.accepted_scene_group_id,
            "scene_group_id": self.scene_group_id,
            "scene_seed": self.scene_seed,
            "scene_id": self.scene_id,
            "scheduled_episode_ids": list(self.scheduled_episode_ids),
            "attempt_ids": list(self.attempt_ids),
            "episodes": [episode.to_dict() for episode in self.episodes],
            "complete": self.complete,
            "accepted": self.accepted,
            "bundle_h5_sha256": self.bundle_h5_sha256,
            "bundle_json_sha256": self.bundle_json_sha256,
            "rejection_reasons": list(self.rejection_reasons),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = _mapping(value, "scene_group_record")
        expected = set(cls.__dataclass_fields__) | {"scene_group_id"}
        _exact_fields(payload, expected, "scene_group_record")
        episodes_value = payload["episodes"]
        if not isinstance(episodes_value, list):
            raise TypeError("episodes must be a list")
        result = cls(
            candidate_scene_index=cast(int, payload["candidate_scene_index"]),
            candidate_scene_id=cast(str, payload["candidate_scene_id"]),
            accepted_scene_group_id=cast(str | None, payload["accepted_scene_group_id"]),
            scene_seed=cast(int, payload["scene_seed"]),
            scene_id=cast(str, payload["scene_id"]),
            scheduled_episode_ids=_string_tuple_from_json(
                payload["scheduled_episode_ids"], "scheduled_episode_ids"
            ),
            attempt_ids=_string_tuple_from_json(payload["attempt_ids"], "attempt_ids"),
            episodes=tuple(
                RawEpisodeRecord.from_dict(_mapping(item, "raw_episode_record"))
                for item in episodes_value
            ),
            complete=cast(bool, payload["complete"]),
            accepted=cast(bool, payload["accepted"]),
            bundle_h5_sha256=cast(str | None, payload["bundle_h5_sha256"]),
            bundle_json_sha256=cast(str | None, payload["bundle_json_sha256"]),
            rejection_reasons=_string_tuple_from_json(
                payload["rejection_reasons"], "rejection_reasons"
            ),
        )
        if payload["scene_group_id"] != result.scene_group_id:
            raise ValueError("scene_group_id is inconsistent")
        return result


@dataclass(frozen=True, slots=True)
class CollectionManifest:
    """Resumable authoritative index of schedule, attempts, groups, and shards."""

    collection_schema_version: str
    collection_run_id: str
    config: CollectionConfig
    schedule: CollectionSchedule
    status: CollectionStatus
    runtime_versions: Mapping[str, str | None]
    next_candidate_scene_index: int
    attempts: tuple[AttemptRecord, ...] = ()
    raw_episodes: tuple[RawEpisodeRecord, ...] = ()
    scene_groups: tuple[SceneGroupRecord, ...] = ()
    source_shards: tuple[SourceShardRecord, ...] = ()
    failure_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.collection_schema_version != COLLECTION_SCHEMA_VERSION:
            raise ValueError("unsupported collection_schema_version")
        _require_string(self.collection_run_id, "collection_run_id")
        if not isinstance(self.config, CollectionConfig):
            raise TypeError("config must be a CollectionConfig")
        if not isinstance(self.schedule, CollectionSchedule):
            raise TypeError("schedule must be a CollectionSchedule")
        if self.schedule.collection_run_id != self.collection_run_id:
            raise ValueError("schedule collection_run_id differs from manifest")
        if self.collection_run_id != stable_collection_run_id(self.config.identity_dict()):
            raise ValueError("collection_run_id is inconsistent with collection config")
        expected_fingerprint = f"sha256:{sha256_hex(self.config.identity_dict())}"
        if self.schedule.config_fingerprint != expected_fingerprint:
            raise ValueError("schedule config_fingerprint differs from config")
        expected_tasks = tuple(
            TaskSpec(
                target_object_id=object_id,
                target_bin_id=bin_id,
                instruction_template_id="canonical_v0",
            )
            for object_id in OBJECT_IDS
            for bin_id in BIN_IDS
        )
        expected_seeds = tuple(
            self.config.candidate_scene_seed_start + offset
            for offset in range(self.config.maximum_candidate_scene_count)
        )
        if (
            self.schedule.collection_schema_version != self.config.collection_schema_version
            or self.schedule.canonical_task_specs != expected_tasks
            or self.schedule.candidate_scene_seeds != expected_seeds
        ):
            raise ValueError("schedule content differs from collection config")
        if any(
            episode.environment_id != self.config.environment_id
            or episode.control_mode != self.config.control_mode
            or episode.collection_schema_version != self.config.collection_schema_version
            or episode.expert_config_fingerprint != self.config.expert_config_fingerprint
            for episode in self.schedule.episodes
        ):
            raise ValueError("scheduled episode runtime identity differs from config")
        object.__setattr__(self, "status", _enum(self.status, CollectionStatus, "status"))
        object.__setattr__(self, "runtime_versions", _runtime_versions(self.runtime_versions))
        if dict(self.runtime_versions) != dict(self.config.runtime_versions):
            raise ValueError("manifest runtime_versions must match collection config")
        _require_int(self.next_candidate_scene_index, "next_candidate_scene_index")
        if self.next_candidate_scene_index > self.config.maximum_candidate_scene_count:
            raise ValueError("next_candidate_scene_index exceeds candidate bound")
        typed_sequences: tuple[tuple[str, object, type[object]], ...] = (
            ("attempts", self.attempts, AttemptRecord),
            ("raw_episodes", self.raw_episodes, RawEpisodeRecord),
            ("scene_groups", self.scene_groups, SceneGroupRecord),
            ("source_shards", self.source_shards, SourceShardRecord),
        )
        for field_name, values, item_type in typed_sequences:
            if not isinstance(values, tuple) or not all(
                isinstance(item, item_type) for item in values
            ):
                raise TypeError(f"{field_name} must be a tuple of {item_type.__name__}")
        _require_string_tuple(self.failure_reasons, "failure_reasons")
        if len({record.attempt_id for record in self.attempts}) != len(self.attempts):
            raise ValueError("attempt IDs must be unique")
        if len({record.raw_trajectory_id for record in self.raw_episodes}) != len(
            self.raw_episodes
        ):
            raise ValueError("raw trajectory IDs must be unique")
        if len({record.candidate_scene_id for record in self.scene_groups}) != len(
            self.scene_groups
        ):
            raise ValueError("candidate scene IDs must be unique")
        if len({record.source_shard_id for record in self.source_shards}) != len(
            self.source_shards
        ):
            raise ValueError("source shard IDs must be unique")

        schedule_by_id = {
            episode.scheduled_episode_id: episode for episode in self.schedule.episodes
        }
        attempts_by_id = {record.attempt_id: record for record in self.attempts}
        raw_by_id = {record.raw_trajectory_id: record for record in self.raw_episodes}
        shards_by_id = {record.source_shard_id: record for record in self.source_shards}

        for attempt in self.attempts:
            scheduled = schedule_by_id.get(attempt.scheduled_episode_id)
            if scheduled is None:
                raise ValueError("attempt references an unknown scheduled episode")
            if attempt.attempt_index > self.config.maximum_expert_attempts_per_task:
                raise ValueError("attempt exceeds the configured retry bound")
            if attempt.attempt_id != stable_attempt_id(
                scheduled_episode_id=attempt.scheduled_episode_id,
                attempt_index=attempt.attempt_index,
            ):
                raise ValueError("attempt_id is inconsistent with scheduled episode and index")
            if (
                attempt.scene_seed != scheduled.scene_seed
                or attempt.scene_id != scheduled.scene_id
                or attempt.task_spec != scheduled.task_spec
                or attempt.task_id != scheduled.task_id
            ):
                raise ValueError("attempt provenance differs from its scheduled episode")
            if attempt.raw_trajectory_id is not None and attempt.raw_trajectory_id != (
                stable_raw_trajectory_id(
                    scheduled_episode_id=attempt.scheduled_episode_id,
                    attempt_id=attempt.attempt_id,
                )
            ):
                raise ValueError("attempt raw_trajectory_id is inconsistent")

        for episode in self.raw_episodes:
            scheduled = schedule_by_id.get(episode.scheduled_episode_id)
            attempt = attempts_by_id.get(episode.attempt_id)
            if scheduled is None or attempt is None:
                raise ValueError("raw episode lacks scheduled or attempt provenance")
            if episode.raw_trajectory_id != stable_raw_trajectory_id(
                scheduled_episode_id=episode.scheduled_episode_id,
                attempt_id=episode.attempt_id,
            ):
                raise ValueError("raw_trajectory_id is inconsistent with semantic inputs")
            if (
                attempt.outcome is not AttemptOutcome.ACCEPTED
                or attempt.raw_trajectory_id != episode.raw_trajectory_id
                or not attempt.trajectory_retained
                or attempt.expert_result.to_dict() != episode.expert_result.to_dict()
                or attempt.replay_validation is None
                or attempt.replay_validation.to_dict() != episode.replay_validation.to_dict()
            ):
                raise ValueError("raw episode disagrees with its accepted attempt")
            replay_errors = (
                episode.replay_validation.final_position_error_m,
                episode.replay_validation.final_orientation_error_rad,
                episode.replay_validation.final_joint_error_rad,
            )
            replay_tolerances = (
                self.config.final_position_tolerance_m,
                self.config.final_orientation_tolerance_rad,
                self.config.final_joint_tolerance_rad,
            )
            if any(
                error is None or error > tolerance
                for error, tolerance in zip(replay_errors, replay_tolerances, strict=True)
            ):
                raise ValueError("accepted replay exceeds configured final-state tolerances")
            if (
                episode.scene_seed != scheduled.scene_seed
                or episode.scene_id != scheduled.scene_id
                or episode.task_spec != scheduled.task_spec
                or episode.task_id != scheduled.task_id
                or episode.canonical_instruction != scheduled.canonical_instruction
            ):
                raise ValueError("raw episode provenance differs from its schedule")
            if episode.source_shard_id not in shards_by_id:
                raise ValueError("raw episode references an unknown source shard")
            if episode.replay_validation.mode is not self.config.replay_validation_mode:
                raise ValueError("accepted replay mode differs from collection config")

        accepted_attempt_raw_ids = [
            attempt.raw_trajectory_id
            for attempt in self.attempts
            if attempt.outcome is AttemptOutcome.ACCEPTED
        ]
        if (
            None in accepted_attempt_raw_ids
            or len(accepted_attempt_raw_ids) != len(raw_by_id)
            or set(accepted_attempt_raw_ids) != set(raw_by_id)
        ):
            raise ValueError("accepted attempts and raw episodes must form an exact one-to-one set")

        shard_indices = tuple(shard.shard_index for shard in self.source_shards)
        if shard_indices != tuple(range(len(self.source_shards))):
            raise ValueError("source shards must use contiguous ordered indices")
        sharded_raw_ids: list[str] = []
        for shard in self.source_shards:
            if shard.source_shard_id != stable_source_shard_id(
                collection_run_id=self.collection_run_id,
                shard_index=shard.shard_index,
            ):
                raise ValueError("source_shard_id is inconsistent")
            for raw_id in shard.raw_trajectory_ids:
                episode = raw_by_id.get(raw_id)
                if episode is None or episode.source_shard_id != shard.source_shard_id:
                    raise ValueError("source shard raw trajectory provenance is inconsistent")
                sharded_raw_ids.append(raw_id)
        if set(sharded_raw_ids) != set(raw_by_id) or len(sharded_raw_ids) != len(raw_by_id):
            raise ValueError("source shards must contain every raw episode exactly once")

        expected_group_indices = tuple(range(self.next_candidate_scene_index))
        if (
            tuple(group.candidate_scene_index for group in self.scene_groups)
            != expected_group_indices
        ):
            raise ValueError("scene groups must cover every committed candidate in order")
        grouped_attempt_ids: list[str] = []
        for group in self.scene_groups:
            scheduled_group = self.schedule.episodes[
                group.candidate_scene_index * 6 : (group.candidate_scene_index + 1) * 6
            ]
            first = scheduled_group[0]
            if (
                group.candidate_scene_id != first.candidate_scene_id
                or group.scene_seed != first.scene_seed
                or group.scene_id != first.scene_id
                or group.scheduled_episode_ids
                != tuple(episode.scheduled_episode_id for episode in scheduled_group)
            ):
                raise ValueError("scene group provenance differs from the schedule")
            for attempt_id in group.attempt_ids:
                attempt = attempts_by_id.get(attempt_id)
                if (
                    attempt is None
                    or attempt.scheduled_episode_id not in group.scheduled_episode_ids
                ):
                    raise ValueError("scene group references an unrelated attempt")
                grouped_attempt_ids.append(attempt_id)
            if group.accepted:
                raw_ids = tuple(episode.raw_trajectory_id for episode in group.episodes)
                if any(
                    raw_by_id.get(episode.raw_trajectory_id) != episode
                    for episode in group.episodes
                ):
                    raise ValueError("accepted scene group episode differs from raw authority")
                if group.accepted_scene_group_id != stable_accepted_scene_group_id(
                    candidate_scene_id=group.candidate_scene_id,
                    raw_trajectory_ids=raw_ids,
                ):
                    raise ValueError("accepted_scene_group_id is inconsistent")
        if set(grouped_attempt_ids) != set(attempts_by_id) or len(grouped_attempt_ids) != len(
            attempts_by_id
        ):
            raise ValueError("scene groups must contain every attempt exactly once")

        accepted_groups = tuple(group for group in self.scene_groups if group.accepted)
        if len(accepted_groups) > self.config.target_complete_scene_count:
            raise ValueError("manifest exceeds target complete scene count")
        accepted_raw_ids = {
            episode.raw_trajectory_id for episode in self.raw_episodes if episode.accepted
        }
        grouped_raw_ids = {
            episode.raw_trajectory_id for group in accepted_groups for episode in group.episodes
        }
        if accepted_raw_ids != grouped_raw_ids:
            raise ValueError("accepted raw episodes must belong to accepted scene groups")
        if self.status is CollectionStatus.COMPLETE:
            if len(accepted_groups) != self.config.target_complete_scene_count:
                raise ValueError("complete manifest has wrong accepted scene count")
            if len(accepted_raw_ids) != 6 * self.config.target_complete_scene_count:
                raise ValueError("complete manifest has wrong accepted episode count")
            if self.failure_reasons:
                raise ValueError("complete manifest cannot have failure_reasons")
        if self.status is CollectionStatus.FAILED and not self.failure_reasons:
            raise ValueError("failed manifest requires failure_reasons")

    @property
    def accepted_scene_group_count(self) -> int:
        return sum(group.accepted for group in self.scene_groups)

    @property
    def accepted_episode_count(self) -> int:
        return sum(episode.accepted for episode in self.raw_episodes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection_schema_version": self.collection_schema_version,
            "collection_run_id": self.collection_run_id,
            "config": self.config.to_dict(),
            "schedule": self.schedule.to_dict(),
            "status": self.status.value,
            "runtime_versions": dict(self.runtime_versions),
            "next_candidate_scene_index": self.next_candidate_scene_index,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "raw_episodes": [episode.to_dict() for episode in self.raw_episodes],
            "scene_groups": [group.to_dict() for group in self.scene_groups],
            "source_shards": [shard.to_dict() for shard in self.source_shards],
            "accepted_scene_group_count": self.accepted_scene_group_count,
            "accepted_episode_count": self.accepted_episode_count,
            "failure_reasons": list(self.failure_reasons),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = _mapping(value, "collection_manifest")
        expected = set(cls.__dataclass_fields__) | {
            "accepted_scene_group_count",
            "accepted_episode_count",
        }
        _exact_fields(payload, expected, "collection_manifest")
        attempts = _list_of(payload["attempts"], "attempts")
        episodes = _list_of(payload["raw_episodes"], "raw_episodes")
        groups = _list_of(payload["scene_groups"], "scene_groups")
        shards = _list_of(payload["source_shards"], "source_shards")
        result = cls(
            collection_schema_version=cast(str, payload["collection_schema_version"]),
            collection_run_id=cast(str, payload["collection_run_id"]),
            config=CollectionConfig.from_dict(_mapping(payload["config"], "config")),
            schedule=CollectionSchedule.from_dict(_mapping(payload["schedule"], "schedule")),
            status=cast(
                CollectionStatus,
                _enum(payload["status"], CollectionStatus, "status"),
            ),
            runtime_versions=cast(Mapping[str, str | None], payload["runtime_versions"]),
            next_candidate_scene_index=cast(int, payload["next_candidate_scene_index"]),
            attempts=tuple(
                AttemptRecord.from_dict(_mapping(item, "attempt_record")) for item in attempts
            ),
            raw_episodes=tuple(
                RawEpisodeRecord.from_dict(_mapping(item, "raw_episode_record"))
                for item in episodes
            ),
            scene_groups=tuple(
                SceneGroupRecord.from_dict(_mapping(item, "scene_group_record")) for item in groups
            ),
            source_shards=tuple(
                SourceShardRecord.from_dict(_mapping(item, "source_shard_record"))
                for item in shards
            ),
            failure_reasons=_string_tuple_from_json(payload["failure_reasons"], "failure_reasons"),
        )
        if payload["accepted_scene_group_count"] != result.accepted_scene_group_count:
            raise ValueError("accepted_scene_group_count is inconsistent")
        if payload["accepted_episode_count"] != result.accepted_episode_count:
            raise ValueError("accepted_episode_count is inconsistent")
        return result


def _list_of(value: object, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    return value


@dataclass(frozen=True, slots=True)
class RawDatasetSummary:
    """Small inspection report derived from one collection manifest."""

    collection_schema_version: str
    collection_run_id: str
    status: CollectionStatus
    target_complete_scene_count: int
    candidate_scene_count: int
    accepted_scene_group_count: int
    rejected_scene_group_count: int
    accepted_episode_count: int
    total_attempt_count: int
    total_action_steps: int
    source_shard_count: int
    source_bytes: int
    task_episode_counts: Mapping[str, int]
    attempt_outcome_counts: Mapping[str, int]
    rejection_reason_counts: Mapping[str, int]
    complete_group_acceptance_rate: float

    def __post_init__(self) -> None:
        if self.collection_schema_version != COLLECTION_SCHEMA_VERSION:
            raise ValueError("unsupported collection_schema_version")
        _require_string(self.collection_run_id, "collection_run_id")
        object.__setattr__(self, "status", _enum(self.status, CollectionStatus, "status"))
        for field_name in (
            "target_complete_scene_count",
            "candidate_scene_count",
            "accepted_scene_group_count",
            "rejected_scene_group_count",
            "accepted_episode_count",
            "total_attempt_count",
            "total_action_steps",
            "source_shard_count",
            "source_bytes",
        ):
            _require_int(getattr(self, field_name), field_name)
        if self.candidate_scene_count != (
            self.accepted_scene_group_count + self.rejected_scene_group_count
        ):
            raise ValueError("candidate scene count is inconsistent")
        if self.accepted_episode_count != 6 * self.accepted_scene_group_count:
            raise ValueError("accepted episode count is inconsistent")
        for field_name in (
            "task_episode_counts",
            "attempt_outcome_counts",
            "rejection_reason_counts",
        ):
            object.__setattr__(
                self,
                field_name,
                _count_mapping(getattr(self, field_name), field_name),
            )
        rate = _require_float(
            self.complete_group_acceptance_rate,
            "complete_group_acceptance_rate",
        )
        if rate > 1:
            raise ValueError("complete_group_acceptance_rate cannot exceed 1")
        expected_rate = (
            self.accepted_scene_group_count / self.candidate_scene_count
            if self.candidate_scene_count
            else 0.0
        )
        if not math.isclose(rate, expected_rate, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("complete_group_acceptance_rate is inconsistent")
        object.__setattr__(self, "complete_group_acceptance_rate", rate)

    @classmethod
    def from_manifest(cls, manifest: CollectionManifest) -> Self:
        accepted_groups = tuple(group for group in manifest.scene_groups if group.accepted)
        rejected_groups = tuple(group for group in manifest.scene_groups if not group.accepted)
        accepted_episodes = tuple(episode for episode in manifest.raw_episodes if episode.accepted)
        return cls(
            collection_schema_version=manifest.collection_schema_version,
            collection_run_id=manifest.collection_run_id,
            status=manifest.status,
            target_complete_scene_count=manifest.config.target_complete_scene_count,
            candidate_scene_count=len(manifest.scene_groups),
            accepted_scene_group_count=len(accepted_groups),
            rejected_scene_group_count=len(rejected_groups),
            accepted_episode_count=len(accepted_episodes),
            total_attempt_count=len(manifest.attempts),
            total_action_steps=sum(episode.elapsed_steps for episode in accepted_episodes),
            source_shard_count=len(manifest.source_shards),
            source_bytes=sum(
                shard.h5_size_bytes + shard.json_size_bytes for shard in manifest.source_shards
            ),
            task_episode_counts=Counter(episode.task_id for episode in accepted_episodes),
            attempt_outcome_counts=Counter(attempt.outcome.value for attempt in manifest.attempts),
            rejection_reason_counts=Counter(
                reason for group in rejected_groups for reason in group.rejection_reasons
            ),
            complete_group_acceptance_rate=(
                len(accepted_groups) / len(manifest.scene_groups) if manifest.scene_groups else 0.0
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection_schema_version": self.collection_schema_version,
            "collection_run_id": self.collection_run_id,
            "status": self.status.value,
            "target_complete_scene_count": self.target_complete_scene_count,
            "candidate_scene_count": self.candidate_scene_count,
            "accepted_scene_group_count": self.accepted_scene_group_count,
            "rejected_scene_group_count": self.rejected_scene_group_count,
            "accepted_episode_count": self.accepted_episode_count,
            "total_attempt_count": self.total_attempt_count,
            "total_action_steps": self.total_action_steps,
            "source_shard_count": self.source_shard_count,
            "source_bytes": self.source_bytes,
            "task_episode_counts": dict(self.task_episode_counts),
            "attempt_outcome_counts": dict(self.attempt_outcome_counts),
            "rejection_reason_counts": dict(self.rejection_reason_counts),
            "complete_group_acceptance_rate": self.complete_group_acceptance_rate,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = _mapping(value, "raw_dataset_summary")
        _exact_fields(payload, set(cls.__dataclass_fields__), "raw_dataset_summary")
        return cls(
            collection_schema_version=cast(str, payload["collection_schema_version"]),
            collection_run_id=cast(str, payload["collection_run_id"]),
            status=cast(
                CollectionStatus,
                _enum(payload["status"], CollectionStatus, "status"),
            ),
            target_complete_scene_count=cast(int, payload["target_complete_scene_count"]),
            candidate_scene_count=cast(int, payload["candidate_scene_count"]),
            accepted_scene_group_count=cast(int, payload["accepted_scene_group_count"]),
            rejected_scene_group_count=cast(int, payload["rejected_scene_group_count"]),
            accepted_episode_count=cast(int, payload["accepted_episode_count"]),
            total_attempt_count=cast(int, payload["total_attempt_count"]),
            total_action_steps=cast(int, payload["total_action_steps"]),
            source_shard_count=cast(int, payload["source_shard_count"]),
            source_bytes=cast(int, payload["source_bytes"]),
            task_episode_counts=cast(Mapping[str, int], payload["task_episode_counts"]),
            attempt_outcome_counts=cast(Mapping[str, int], payload["attempt_outcome_counts"]),
            rejection_reason_counts=cast(Mapping[str, int], payload["rejection_reason_counts"]),
            complete_group_acceptance_rate=cast(float, payload["complete_group_acceptance_rate"]),
        )
