"""Committed, leakage-safe M4.2 development and final evaluation schedules.

The two seed lists are generated together from audited M3A/M3B/M4/M4.1
provenance.  Loading validates the committed locks, while materializing the
sealed final schedule additionally requires an explicit authorization that
binds all development selections and a clean implementation fingerprint.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from types import MappingProxyType
from typing import Any, Self, cast

from langmani.datasets.identity import sha256_hex
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import TaskSpec, stable_scene_id, stable_task_id

M42_EXCLUSION_SCHEMA_VERSION = "langmani-m42-seed-exclusion-v0"
M42_SCHEDULE_SCHEMA_VERSION = "langmani-m42-seed-schedule-v0"
M42_GENERATION_SCHEMA_VERSION = "langmani-m42-joint-seed-generation-v0"
M42_EVALUATION_CONFIG_SCHEMA_VERSION = "langmani-m42-evaluation-config-v0"
M42_GENERATION_NAMESPACE = "langmani-m42-dev-final-v0"
M42_DEV_SCHEDULE_ID = "m42_dev_v0"
M42_FINAL_SCHEDULE_ID = "m42_final_v0"
M42_M3B_EXPORT_FINGERPRINT = (
    "sha256:3f4d81471ac7c3ecc034206bc207524c4cbfbd1f7874b25eca894a5b607acfb4"
)
M42_EXCLUSION_DIGEST = "sha256:801e5b9d0595855729a45fa3bab25b85449afab2aec06065dd69233eec0f5610"
M42_EVALUATION_CONFIG_FINGERPRINT = (
    "sha256:0746ecb022ac67e73bc04d19db39c6e11e76ce264669d5e1dec9474a34cbb6d4"
)
M42_DEV_SCHEDULE_FINGERPRINT = (
    "sha256:981547e771b2b5cd3a77e2788bb49d29fc45b3f59c607021a03a4e2ce70b43f1"
)
M42_FINAL_SCHEDULE_FINGERPRINT = (
    "sha256:b2aef313e076201f7a94875c835c2d606d8f255e3f75d53f7ac7fadcdbb267fc"
)
M42_MAXIMUM_EPISODE_STEPS = 200
M42_CANDIDATE_ATTEMPTS = 42
_MAX_SCENE_SEED = 2**31 - 1
_RESOURCE_DIRECTORY = "m42_schedules"


class M42ScheduleError(ValueError):
    """Raised when a committed M4.2 schedule or its provenance is invalid."""


class FinalScheduleAccessError(PermissionError):
    """Raised when code tries to materialize the sealed final benchmark early."""


def _fingerprint(payload: object) -> str:
    return f"sha256:{sha256_hex(payload)}"


def _require_fingerprint(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA-256 fingerprint")
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 fingerprint")
    return value


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _int_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list | tuple):
        raise TypeError(f"{name} must be a JSON array")
    result = tuple(value)
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in result):
        raise TypeError(f"{name} must contain integers")
    if any(seed < 0 or seed > _MAX_SCENE_SEED for seed in result):
        raise ValueError(f"{name} must contain non-negative int31 values")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return cast(tuple[int, ...], result)


def _task_specs(value: object) -> tuple[TaskSpec, ...]:
    if not isinstance(value, list | tuple):
        raise TypeError("ordered_task_specs must be a JSON array")
    result = tuple(TaskSpec.from_mapping(_mapping(item, "task_spec")) for item in value)
    if result != CANONICAL_TASK_SPECS:
        raise M42ScheduleError("M4.2 schedules must use canonical object-major/bin-minor order")
    return result


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def evaluation_config_payload() -> dict[str, object]:
    """Return the exact simulator/evaluation payload bound by both seed locks."""
    return {
        "schema_version": M42_EVALUATION_CONFIG_SCHEMA_VERSION,
        "environment_id": "LangMani-PickPlaceByInstruction-v0",
        "control_mode": "pd_joint_pos",
        "num_envs": 1,
        "sim_backend": "physx_cpu",
        "obs_mode": "rgb",
        "reward_mode": "none",
        "render_mode": "rgb_array",
        "camera_name": "base_camera",
        "maximum_episode_steps": M42_MAXIMUM_EPISODE_STEPS,
        "control_frequency_hz": 20,
        "ordered_task_specs": [task.to_dict() for task in CANONICAL_TASK_SPECS],
    }


@dataclass(frozen=True, slots=True)
class M42ExclusionSources:
    """Exact prior seed evidence used by the joint deterministic generator."""

    payload: Mapping[str, object]
    exclusion_digest: str

    def __post_init__(self) -> None:
        payload = dict(_mapping(self.payload, "exclusion payload"))
        if payload.get("schema_version") != M42_EXCLUSION_SCHEMA_VERSION:
            raise M42ScheduleError("unsupported M4.2 exclusion schema")
        digest = _require_fingerprint(self.exclusion_digest, "exclusion_digest")
        if _fingerprint(payload) != digest or digest != M42_EXCLUSION_DIGEST:
            raise M42ScheduleError("M4.2 exclusion-source digest is invalid")
        object.__setattr__(self, "payload", _freeze_json(payload))

    @property
    def excluded_scene_seeds(self) -> frozenset[int]:
        """Return the exact union of all prior observed or predeclared seeds."""
        source = self.payload
        m3a = _mapping(source["m3a_full"], "m3a_full")
        m3b = _mapping(source["m3b_full"], "m3b_full")
        m4 = _mapping(source["m4_full_fresh"], "m4_full_fresh")
        tiny = _mapping(source["m4_smoke_tiny"], "m4_smoke_tiny")
        m41 = _mapping(source["m41_diagnostic_target_smoke"], "m41 diagnostics")
        groups = (
            _int_tuple(m3a["accepted_scene_seeds"], "M3A accepted seeds"),
            _int_tuple(m3a["rejected_candidate_scene_seeds"], "M3A rejected seeds"),
            _int_tuple(m3b["train_scene_seeds"], "M3B train seeds"),
            _int_tuple(m3b["validation_scene_seeds"], "M3B validation seeds"),
            _int_tuple(m3b["test_scene_seeds"], "M3B test seeds"),
            _int_tuple(m4["ordered_scene_seeds"], "M4 full fresh seeds"),
            _int_tuple(tiny["actual_scene_seeds"], "M4 smoke seeds"),
            _int_tuple(tiny["predeclared_fresh_scene_seeds"], "M4 tiny fresh seeds"),
            _int_tuple(m41["actual_scene_seeds"], "M4.1 diagnostic seeds"),
        )
        result = frozenset(seed for group in groups for seed in group)
        accepted = frozenset(groups[0])
        rejected = frozenset(groups[1])
        if accepted | rejected != frozenset(range(65)):
            raise M42ScheduleError("M3A attempted seeds must be exactly 0 through 64")
        if frozenset(groups[2]) | frozenset(groups[3]) | frozenset(groups[4]) != accepted:
            raise M42ScheduleError("M3B split seeds must exactly partition M3A accepted seeds")
        if len(result) != 125:
            raise M42ScheduleError("M4.2 exclusion union must contain exactly 125 seeds")
        return result

    def to_dict(self) -> dict[str, object]:
        payload = _thaw_json(self.payload)
        if not isinstance(payload, dict):
            raise AssertionError("frozen exclusion payload did not thaw to a mapping")
        return {**payload, "exclusion_digest": self.exclusion_digest}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        source = dict(_mapping(value, "M4.2 exclusion lock"))
        expected = {
            "schema_version",
            "m3a_full",
            "m3b_full",
            "m4_full_fresh",
            "m4_smoke_tiny",
            "m41_diagnostic_target_smoke",
            "exclusion_digest",
        }
        if set(source) != expected:
            raise M42ScheduleError("M4.2 exclusion lock fields are invalid")
        digest = source.pop("exclusion_digest")
        return cls(payload=source, exclusion_digest=cast(str, digest))


@dataclass(frozen=True, slots=True)
class M42SeedSchedule:
    """One committed seed lock; it deliberately does not expose rollout episodes."""

    schedule_id: str
    stage: str
    ordered_scene_seeds: tuple[int, ...]
    counterpart_schedule_id: str
    counterpart_scene_seeds: tuple[int, ...]
    ordered_task_specs: tuple[TaskSpec, ...]
    schedule_fingerprint: str
    base_exclusion_digest: str = M42_EXCLUSION_DIGEST
    evaluation_config_fingerprint: str = M42_EVALUATION_CONFIG_FINGERPRINT
    m3b_export_fingerprint: str = M42_M3B_EXPORT_FINGERPRINT
    maximum_episode_steps: int = M42_MAXIMUM_EPISODE_STEPS
    candidate_attempts: int = M42_CANDIDATE_ATTEMPTS
    schema_version: str = M42_SCHEDULE_SCHEMA_VERSION
    generation_schema_version: str = M42_GENERATION_SCHEMA_VERSION
    generation_namespace: str = M42_GENERATION_NAMESPACE

    def __post_init__(self) -> None:
        expected = {
            M42_DEV_SCHEDULE_ID: ("development", 12, M42_FINAL_SCHEDULE_ID, 30),
            M42_FINAL_SCHEDULE_ID: ("final", 30, M42_DEV_SCHEDULE_ID, 12),
        }.get(self.schedule_id)
        if expected is None:
            raise M42ScheduleError("unknown M4.2 schedule ID")
        stage, seed_count, counterpart_id, counterpart_count = expected
        if (
            self.stage != stage
            or len(self.ordered_scene_seeds) != seed_count
            or self.counterpart_schedule_id != counterpart_id
            or len(self.counterpart_scene_seeds) != counterpart_count
        ):
            raise M42ScheduleError("M4.2 schedule stage or seed counts are invalid")
        _int_tuple(self.ordered_scene_seeds, "ordered_scene_seeds")
        _int_tuple(self.counterpart_scene_seeds, "counterpart_scene_seeds")
        if set(self.ordered_scene_seeds) & set(self.counterpart_scene_seeds):
            raise M42ScheduleError("development and final seed schedules must be disjoint")
        if self.ordered_task_specs != CANONICAL_TASK_SPECS:
            raise M42ScheduleError("M4.2 schedule TaskSpec order is invalid")
        if (
            self.schema_version != M42_SCHEDULE_SCHEMA_VERSION
            or self.generation_schema_version != M42_GENERATION_SCHEMA_VERSION
            or self.generation_namespace != M42_GENERATION_NAMESPACE
            or self.m3b_export_fingerprint != M42_M3B_EXPORT_FINGERPRINT
            or self.base_exclusion_digest != M42_EXCLUSION_DIGEST
            or self.evaluation_config_fingerprint != M42_EVALUATION_CONFIG_FINGERPRINT
            or self.maximum_episode_steps != M42_MAXIMUM_EPISODE_STEPS
            or self.candidate_attempts != M42_CANDIDATE_ATTEMPTS
        ):
            raise M42ScheduleError("M4.2 schedule identity differs from the committed contract")
        expected_fingerprint = {
            M42_DEV_SCHEDULE_ID: M42_DEV_SCHEDULE_FINGERPRINT,
            M42_FINAL_SCHEDULE_ID: M42_FINAL_SCHEDULE_FINGERPRINT,
        }[self.schedule_id]
        if self.compute_fingerprint() != self.schedule_fingerprint:
            raise M42ScheduleError("M4.2 schedule fingerprint is invalid")
        if self.schedule_fingerprint != expected_fingerprint:
            raise M42ScheduleError("M4.2 schedule fingerprint differs from the source lock")

    def fingerprint_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generation_schema_version": self.generation_schema_version,
            "generation_namespace": self.generation_namespace,
            "m3b_export_fingerprint": self.m3b_export_fingerprint,
            "base_exclusion_digest": self.base_exclusion_digest,
            "candidate_attempts": self.candidate_attempts,
            "maximum_episode_steps": self.maximum_episode_steps,
            "evaluation_config_fingerprint": self.evaluation_config_fingerprint,
            "ordered_task_specs": [task.to_dict() for task in self.ordered_task_specs],
            "schedule_id": self.schedule_id,
            "stage": self.stage,
            "ordered_scene_seeds": list(self.ordered_scene_seeds),
            "counterpart_schedule_id": self.counterpart_schedule_id,
            "counterpart_scene_seeds": list(self.counterpart_scene_seeds),
        }

    def compute_fingerprint(self) -> str:
        return _fingerprint(self.fingerprint_payload())

    def to_dict(self) -> dict[str, object]:
        return {**self.fingerprint_payload(), "schedule_fingerprint": self.schedule_fingerprint}

    @classmethod
    def from_dict(cls, value: object) -> Self:
        source = _mapping(value, "M4.2 schedule lock")
        expected = {
            "schema_version",
            "generation_schema_version",
            "generation_namespace",
            "m3b_export_fingerprint",
            "base_exclusion_digest",
            "candidate_attempts",
            "maximum_episode_steps",
            "evaluation_config_fingerprint",
            "ordered_task_specs",
            "schedule_id",
            "stage",
            "ordered_scene_seeds",
            "counterpart_schedule_id",
            "counterpart_scene_seeds",
            "schedule_fingerprint",
        }
        if set(source) != expected:
            raise M42ScheduleError("M4.2 schedule lock fields are invalid")
        return cls(
            schema_version=cast(str, source["schema_version"]),
            generation_schema_version=cast(str, source["generation_schema_version"]),
            generation_namespace=cast(str, source["generation_namespace"]),
            m3b_export_fingerprint=cast(str, source["m3b_export_fingerprint"]),
            base_exclusion_digest=cast(str, source["base_exclusion_digest"]),
            candidate_attempts=cast(int, source["candidate_attempts"]),
            maximum_episode_steps=cast(int, source["maximum_episode_steps"]),
            evaluation_config_fingerprint=cast(str, source["evaluation_config_fingerprint"]),
            ordered_task_specs=_task_specs(source["ordered_task_specs"]),
            schedule_id=cast(str, source["schedule_id"]),
            stage=cast(str, source["stage"]),
            ordered_scene_seeds=_int_tuple(source["ordered_scene_seeds"], "ordered_scene_seeds"),
            counterpart_schedule_id=cast(str, source["counterpart_schedule_id"]),
            counterpart_scene_seeds=_int_tuple(
                source["counterpart_scene_seeds"], "counterpart_scene_seeds"
            ),
            schedule_fingerprint=cast(str, source["schedule_fingerprint"]),
        )


@dataclass(frozen=True, slots=True)
class FinalScheduleAuthorization:
    """Explicit proof that every development choice is frozen before final rollout."""

    final_schedule_fingerprint: str
    expected_implementation_fingerprint: str
    actual_implementation_fingerprint: str
    explicit_final_authorization: bool
    clean_git: bool
    horizon_selection_locked: bool
    gripper_selection_locked: bool
    task_token_checkpoint_selection_locked: bool

    def __post_init__(self) -> None:
        for name in (
            "final_schedule_fingerprint",
            "expected_implementation_fingerprint",
            "actual_implementation_fingerprint",
        ):
            _require_fingerprint(getattr(self, name), name)
        for name in (
            "explicit_final_authorization",
            "clean_git",
            "horizon_selection_locked",
            "gripper_selection_locked",
            "task_token_checkpoint_selection_locked",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")

    def require_valid(self, schedule: M42SeedSchedule) -> None:
        failures: list[str] = []
        if schedule.schedule_id != M42_FINAL_SCHEDULE_ID:
            failures.append("authorization applies only to m42_final_v0")
        if self.final_schedule_fingerprint != schedule.schedule_fingerprint:
            failures.append("final schedule fingerprint mismatch")
        if self.expected_implementation_fingerprint != self.actual_implementation_fingerprint:
            failures.append("implementation fingerprint mismatch")
        if not self.explicit_final_authorization:
            failures.append("explicit final flag is absent")
        if not self.clean_git:
            failures.append("Git worktree is dirty")
        if not self.horizon_selection_locked:
            failures.append("horizon selection is not locked")
        if not self.gripper_selection_locked:
            failures.append("gripper selection is not locked")
        if not self.task_token_checkpoint_selection_locked:
            failures.append("TaskToken checkpoint selection is not locked")
        if failures:
            raise FinalScheduleAccessError("; ".join(failures))


@dataclass(frozen=True, slots=True)
class M42ScheduledEpisode:
    """One deterministic scene/task pair materialized for an authorized rollout."""

    schedule_id: str
    episode_index: int
    scene_index: int
    task_index: int
    scene_seed: int
    scene_id: str
    task_spec: TaskSpec
    task_id: str


def _read_resource(name: str) -> dict[str, Any]:
    path = resources.files("langmani.policies").joinpath(_RESOURCE_DIRECTORY, name)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise M42ScheduleError(
            f"cannot read committed M4.2 schedule resource {name}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise M42ScheduleError(f"committed M4.2 schedule resource {name} must be an object")
    return payload


def load_exclusion_sources() -> M42ExclusionSources:
    """Load and validate the exact committed exclusion provenance."""
    return M42ExclusionSources.from_dict(_read_resource("exclusion_sources_v0.json"))


def load_locked_schedule(schedule_id: str) -> M42SeedSchedule:
    """Validate a lock without authorizing or creating final rollout episodes."""
    names = {
        M42_DEV_SCHEDULE_ID: "m42_dev_v0.json",
        M42_FINAL_SCHEDULE_ID: "m42_final_v0.json",
    }
    try:
        name = names[schedule_id]
    except (KeyError, TypeError) as error:
        raise M42ScheduleError(f"unknown M4.2 schedule ID {schedule_id!r}") from error
    return M42SeedSchedule.from_dict(_read_resource(name))


def generate_locked_scene_seeds(
    exclusions: M42ExclusionSources | None = None,
    *,
    max_attempts: int = 1_000_000,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """Regenerate both committed lists from the audited SHA-256 counter stream."""
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 42:
        raise M42ScheduleError("max_attempts must be an integer of at least 42")
    source = exclusions if exclusions is not None else load_exclusion_sources()
    excluded = source.excluded_scene_seeds
    selected: list[int] = []
    for counter in range(max_attempts):
        digest = sha256_hex(
            {
                "schema_version": M42_GENERATION_SCHEMA_VERSION,
                "namespace": M42_GENERATION_NAMESPACE,
                "m3b_export_fingerprint": M42_M3B_EXPORT_FINGERPRINT,
                "base_exclusion_digest": source.exclusion_digest,
                "candidate_counter": counter,
            }
        )
        candidate = int(digest[:16], 16) % (_MAX_SCENE_SEED + 1)
        if candidate in excluded or candidate in selected:
            continue
        selected.append(candidate)
        if len(selected) == 42:
            attempts = counter + 1
            return tuple(selected[:12]), tuple(selected[12:]), attempts
    raise M42ScheduleError(
        f"could not generate 42 unique unexcluded scene seeds within {max_attempts} attempts"
    )


def validate_locked_schedules() -> tuple[M42SeedSchedule, M42SeedSchedule]:
    """Validate both locks and their joint provenance without final materialization."""
    exclusions = load_exclusion_sources()
    if _fingerprint(evaluation_config_payload()) != M42_EVALUATION_CONFIG_FINGERPRINT:
        raise M42ScheduleError("M4.2 evaluation configuration fingerprint is invalid")
    development = load_locked_schedule(M42_DEV_SCHEDULE_ID)
    final = load_locked_schedule(M42_FINAL_SCHEDULE_ID)
    regenerated_dev, regenerated_final, attempts = generate_locked_scene_seeds(exclusions)
    if (
        development.ordered_scene_seeds != regenerated_dev
        or final.ordered_scene_seeds != regenerated_final
        or development.counterpart_scene_seeds != final.ordered_scene_seeds
        or final.counterpart_scene_seeds != development.ordered_scene_seeds
        or attempts != M42_CANDIDATE_ATTEMPTS
    ):
        raise M42ScheduleError("committed M4.2 schedules differ from joint regeneration")
    if exclusions.excluded_scene_seeds & (
        set(development.ordered_scene_seeds) | set(final.ordered_scene_seeds)
    ):
        raise M42ScheduleError("committed M4.2 schedules overlap prior known seeds")
    return development, final


def materialize_schedule(
    schedule: M42SeedSchedule,
    *,
    final_authorization: FinalScheduleAuthorization | None = None,
) -> tuple[M42ScheduledEpisode, ...]:
    """Expand a lock into seed-major/task-minor episodes after the final gate."""
    if not isinstance(schedule, M42SeedSchedule):
        raise TypeError("schedule must be an M42SeedSchedule")
    if schedule.schedule_id == M42_FINAL_SCHEDULE_ID:
        if final_authorization is None:
            raise FinalScheduleAccessError(
                "m42_final_v0 cannot be materialized without explicit final authorization"
            )
        final_authorization.require_valid(schedule)
    elif final_authorization is not None:
        raise FinalScheduleAccessError("final authorization cannot be used for development")
    episodes: list[M42ScheduledEpisode] = []
    for scene_index, scene_seed in enumerate(schedule.ordered_scene_seeds):
        for task_index, task_spec in enumerate(schedule.ordered_task_specs):
            episodes.append(
                M42ScheduledEpisode(
                    schedule_id=schedule.schedule_id,
                    episode_index=len(episodes),
                    scene_index=scene_index,
                    task_index=task_index,
                    scene_seed=scene_seed,
                    scene_id=stable_scene_id(scene_seed),
                    task_spec=task_spec,
                    task_id=stable_task_id(task_spec),
                )
            )
    return tuple(episodes)


def materialize_locked_schedule(
    schedule_id: str,
    *,
    final_authorization: FinalScheduleAuthorization | None = None,
) -> tuple[M42ScheduledEpisode, ...]:
    """Load, validate, then materialize one committed schedule."""
    development, final = validate_locked_schedules()
    schedule = development if schedule_id == M42_DEV_SCHEDULE_ID else final
    if schedule_id not in {M42_DEV_SCHEDULE_ID, M42_FINAL_SCHEDULE_ID}:
        raise M42ScheduleError(f"unknown M4.2 schedule ID {schedule_id!r}")
    return materialize_schedule(schedule, final_authorization=final_authorization)


__all__ = [
    "FinalScheduleAccessError",
    "FinalScheduleAuthorization",
    "M42_CANDIDATE_ATTEMPTS",
    "M42_DEV_SCHEDULE_FINGERPRINT",
    "M42_DEV_SCHEDULE_ID",
    "M42_EVALUATION_CONFIG_FINGERPRINT",
    "M42_EXCLUSION_DIGEST",
    "M42_FINAL_SCHEDULE_FINGERPRINT",
    "M42_FINAL_SCHEDULE_ID",
    "M42_MAXIMUM_EPISODE_STEPS",
    "M42ExclusionSources",
    "M42ScheduleError",
    "M42ScheduledEpisode",
    "M42SeedSchedule",
    "evaluation_config_payload",
    "generate_locked_scene_seeds",
    "load_exclusion_sources",
    "load_locked_schedule",
    "materialize_locked_schedule",
    "materialize_schedule",
    "validate_locked_schedules",
]
