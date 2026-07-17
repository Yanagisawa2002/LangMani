"""Deterministic, leakage-safe M5A language and control schedule locks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from langmani.datasets.identity import sha256_hex
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import TaskSpec, stable_scene_id, stable_task_id
from langmani.language.corpus import (
    M5A_SEALED_FINAL_EXAMPLE_ID_PREFIX,
    GeneratedLanguageCorpus,
)
from langmani.language.router_types import LanguageSplit
from langmani.policies.m42_schedule import load_exclusion_sources, validate_locked_schedules

M5A_SCHEDULE_SCHEMA_VERSION = "langmani-m5a-schedules-v1"
M5A_TARGET_DEVELOPMENT_SCHEDULE_LOCK_SCHEMA = "langmani-m5a-target-development-schedule-lock-v1"
M5A_SEED_GENERATION_VERSION = "ordered-unexcluded-int31-v1"
M5A_LANGUAGE_DEV_SCHEDULE_ID = "m5a_language_dev_v0"
M5A_LANGUAGE_FINAL_SCHEDULE_ID = "m5a_language_final_v0"
M5A_CONTROL_DEV_SCHEDULE_ID = "m5a_control_dev_v0"
M5A_CONTROL_FINAL_SCHEDULE_ID = "m5a_control_final_v0"
M5A_DEVELOPMENT_SCENE_COUNT = 12
M5A_FINAL_SCENE_COUNT = 30
M5A_DEFAULT_CANDIDATE_SEED_START = 2_000_000_000
_MAX_SCENE_SEED = 2**31 - 1

REQUIRED_EXCLUSION_SOURCE_IDS: tuple[str, ...] = (
    "m3a_accepted",
    "m3a_rejected_candidates",
    "m3b_train",
    "m3b_validation",
    "m3b_test",
    "m4_fresh",
    "m4_smoke",
    "m4_tiny_overfit",
    "m42_dev_v0",
    "m42_final_v0",
    "m43_development",
)


class M5AScheduleError(ValueError):
    """Raised when M5A schedule identity or exclusion provenance is invalid."""


class FinalControlScheduleAccessError(PermissionError):
    """Raised when target-development tries to materialize final control episodes."""


def _fingerprint(payload: object) -> str:
    return f"sha256:{sha256_hex(payload)}"


def _require_fingerprint(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise M5AScheduleError(f"{name} must be a SHA-256 fingerprint")
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise M5AScheduleError(f"{name} must be a lowercase SHA-256 fingerprint")
    return value


@dataclass(frozen=True, slots=True)
class SeedExclusionSource:
    """One provenance-bound collection of scene seeds that M5A must not reuse."""

    source_id: str
    source_fingerprint: str
    scene_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.source_id not in REQUIRED_EXCLUSION_SOURCE_IDS:
            raise M5AScheduleError(f"unknown M5A exclusion source {self.source_id!r}")
        _require_fingerprint(self.source_fingerprint, name="source_fingerprint")
        if not isinstance(self.scene_seeds, tuple):
            raise M5AScheduleError("scene_seeds must be a tuple")
        if any(
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
            or seed > _MAX_SCENE_SEED
            for seed in self.scene_seeds
        ):
            raise M5AScheduleError("excluded scene seeds must be non-negative int31 values")
        if len(set(self.scene_seeds)) != len(self.scene_seeds):
            raise M5AScheduleError("one exclusion source cannot contain duplicate seeds")
        object.__setattr__(self, "scene_seeds", tuple(sorted(self.scene_seeds)))

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "source_fingerprint": self.source_fingerprint,
            "scene_seeds": list(self.scene_seeds),
        }


@dataclass(frozen=True, slots=True)
class ControlScheduleConfig:
    """Complete deterministic seed-selection input for both control schedules."""

    exclusion_sources: tuple[SeedExclusionSource, ...]
    candidate_seed_start: int = M5A_DEFAULT_CANDIDATE_SEED_START
    maximum_candidate_count: int = 1_000_000
    generation_version: str = M5A_SEED_GENERATION_VERSION
    schema_version: str = M5A_SCHEDULE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.exclusion_sources, tuple):
            raise M5AScheduleError("exclusion_sources must be a tuple")
        source_ids = tuple(source.source_id for source in self.exclusion_sources)
        if set(source_ids) != set(REQUIRED_EXCLUSION_SOURCE_IDS) or len(source_ids) != len(
            REQUIRED_EXCLUSION_SOURCE_IDS
        ):
            raise M5AScheduleError(
                "exclusion_sources must declare every required source exactly once"
            )
        if (
            isinstance(self.candidate_seed_start, bool)
            or not isinstance(self.candidate_seed_start, int)
            or not 0 <= self.candidate_seed_start <= _MAX_SCENE_SEED
        ):
            raise M5AScheduleError("candidate_seed_start must be a non-negative int31 value")
        if (
            isinstance(self.maximum_candidate_count, bool)
            or not isinstance(self.maximum_candidate_count, int)
            or self.maximum_candidate_count < M5A_DEVELOPMENT_SCENE_COUNT + M5A_FINAL_SCENE_COUNT
        ):
            raise M5AScheduleError("maximum_candidate_count is too small for both schedules")
        if self.generation_version != M5A_SEED_GENERATION_VERSION:
            raise M5AScheduleError("unknown M5A seed generation version")
        if self.schema_version != M5A_SCHEDULE_SCHEMA_VERSION:
            raise M5AScheduleError("unknown M5A schedule schema version")
        object.__setattr__(
            self,
            "exclusion_sources",
            tuple(sorted(self.exclusion_sources, key=lambda source: source.source_id)),
        )

    @property
    def excluded_scene_seeds(self) -> frozenset[int]:
        return frozenset(seed for source in self.exclusion_sources for seed in source.scene_seeds)

    @property
    def exclusion_digest(self) -> str:
        return _fingerprint(
            {
                "schema_version": self.schema_version,
                "sources": [source.to_dict() for source in self.exclusion_sources],
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "exclusion_sources": [source.to_dict() for source in self.exclusion_sources],
            "candidate_seed_start": self.candidate_seed_start,
            "maximum_candidate_count": self.maximum_candidate_count,
            "generation_version": self.generation_version,
            "schema_version": self.schema_version,
            "exclusion_digest": self.exclusion_digest,
        }


@dataclass(frozen=True, slots=True)
class LanguageScheduleLock:
    """A corpus split lock containing IDs and fingerprints, never command text."""

    schedule_id: str
    split: LanguageSplit
    ordered_example_ids: tuple[str, ...]
    corpus_fingerprint: str
    split_fingerprint: str
    schedule_fingerprint: str = ""
    sealed: bool = False
    schema_version: str = M5A_SCHEDULE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        expected = {
            M5A_LANGUAGE_DEV_SCHEDULE_ID: (LanguageSplit.DEVELOPMENT, False),
            M5A_LANGUAGE_FINAL_SCHEDULE_ID: (LanguageSplit.FINAL, True),
        }.get(self.schedule_id)
        if expected is None or (self.split, self.sealed) != expected:
            raise M5AScheduleError("language schedule ID, split, or seal state is invalid")
        if not self.ordered_example_ids or len(set(self.ordered_example_ids)) != len(
            self.ordered_example_ids
        ):
            raise M5AScheduleError("language schedule example IDs must be non-empty and unique")
        if not all(isinstance(value, str) and value for value in self.ordered_example_ids):
            raise M5AScheduleError("language schedule example IDs must be non-empty strings")
        _require_fingerprint(self.corpus_fingerprint, name="corpus_fingerprint")
        _require_fingerprint(self.split_fingerprint, name="split_fingerprint")
        if self.schema_version != M5A_SCHEDULE_SCHEMA_VERSION:
            raise M5AScheduleError("unknown M5A language schedule schema")
        expected_fingerprint = _fingerprint(self.fingerprint_payload())
        if self.schedule_fingerprint and self.schedule_fingerprint != expected_fingerprint:
            raise M5AScheduleError("language schedule fingerprint is invalid")
        object.__setattr__(self, "schedule_fingerprint", expected_fingerprint)

    def fingerprint_payload(self) -> dict[str, object]:
        return {
            "schedule_id": self.schedule_id,
            "split": self.split.value,
            "ordered_example_ids": list(self.ordered_example_ids),
            "corpus_fingerprint": self.corpus_fingerprint,
            "split_fingerprint": self.split_fingerprint,
            "sealed": self.sealed,
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.fingerprint_payload(), "schedule_fingerprint": self.schedule_fingerprint}


@dataclass(frozen=True, slots=True)
class ControlScheduleLock:
    """One seed/task/language-ID lock; final command text is not present."""

    schedule_id: str
    split: LanguageSplit
    ordered_scene_seeds: tuple[int, ...]
    counterpart_scene_seeds: tuple[int, ...]
    ordered_task_specs: tuple[TaskSpec, ...]
    associated_language_example_ids: tuple[str, ...]
    exclusion_digest: str
    language_schedule_fingerprint: str
    schedule_fingerprint: str = ""
    sealed: bool = False
    schema_version: str = M5A_SCHEDULE_SCHEMA_VERSION
    generation_version: str = M5A_SEED_GENERATION_VERSION

    def __post_init__(self) -> None:
        expected = {
            M5A_CONTROL_DEV_SCHEDULE_ID: (
                LanguageSplit.DEVELOPMENT,
                M5A_DEVELOPMENT_SCENE_COUNT,
                M5A_FINAL_SCENE_COUNT,
                False,
            ),
            M5A_CONTROL_FINAL_SCHEDULE_ID: (
                LanguageSplit.FINAL,
                M5A_FINAL_SCENE_COUNT,
                M5A_DEVELOPMENT_SCENE_COUNT,
                True,
            ),
        }.get(self.schedule_id)
        if expected is None:
            raise M5AScheduleError("unknown M5A control schedule ID")
        split, scene_count, counterpart_count, sealed = expected
        if (
            self.split is not split
            or len(self.ordered_scene_seeds) != scene_count
            or len(self.counterpart_scene_seeds) != counterpart_count
            or self.sealed is not sealed
        ):
            raise M5AScheduleError("control schedule split, size, counterpart, or seal is invalid")
        if len(set(self.ordered_scene_seeds)) != scene_count:
            raise M5AScheduleError("control schedule scene seeds must be unique")
        if len(set(self.counterpart_scene_seeds)) != counterpart_count:
            raise M5AScheduleError("counterpart scene seeds must be unique")
        if set(self.ordered_scene_seeds) & set(self.counterpart_scene_seeds):
            raise M5AScheduleError("development and final control schedules must be disjoint")
        if self.ordered_task_specs != CANONICAL_TASK_SPECS:
            raise M5AScheduleError("control schedule must use canonical object-major task order")
        episode_count = scene_count * len(CANONICAL_TASK_SPECS)
        if len(self.associated_language_example_ids) != episode_count:
            raise M5AScheduleError("control schedule must associate one language ID per episode")
        if not all(
            isinstance(example_id, str) and example_id
            for example_id in self.associated_language_example_ids
        ):
            raise M5AScheduleError("associated language example IDs must be non-empty strings")
        _require_fingerprint(self.exclusion_digest, name="exclusion_digest")
        _require_fingerprint(
            self.language_schedule_fingerprint,
            name="language_schedule_fingerprint",
        )
        if (
            self.schema_version != M5A_SCHEDULE_SCHEMA_VERSION
            or self.generation_version != M5A_SEED_GENERATION_VERSION
        ):
            raise M5AScheduleError("unknown M5A control schedule version")
        expected_fingerprint = _fingerprint(self.fingerprint_payload())
        if self.schedule_fingerprint and self.schedule_fingerprint != expected_fingerprint:
            raise M5AScheduleError("control schedule fingerprint is invalid")
        object.__setattr__(self, "schedule_fingerprint", expected_fingerprint)

    @property
    def episode_count(self) -> int:
        return len(self.associated_language_example_ids)

    def fingerprint_payload(self) -> dict[str, object]:
        return {
            "schedule_id": self.schedule_id,
            "split": self.split.value,
            "ordered_scene_seeds": list(self.ordered_scene_seeds),
            "counterpart_scene_seeds": list(self.counterpart_scene_seeds),
            "ordered_task_specs": [task_spec.to_dict() for task_spec in self.ordered_task_specs],
            "associated_language_example_ids": list(self.associated_language_example_ids),
            "exclusion_digest": self.exclusion_digest,
            "language_schedule_fingerprint": self.language_schedule_fingerprint,
            "sealed": self.sealed,
            "schema_version": self.schema_version,
            "generation_version": self.generation_version,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.fingerprint_payload(), "schedule_fingerprint": self.schedule_fingerprint}


@dataclass(frozen=True, slots=True)
class ScheduledControlEpisode:
    """One authorized scene/task pair associated with an opaque language-example ID."""

    schedule_id: str
    episode_index: int
    scene_index: int
    task_index: int
    scene_seed: int
    scene_id: str
    task_spec: TaskSpec
    task_id: str
    language_example_id: str

    def __post_init__(self) -> None:
        if self.scene_id != stable_scene_id(self.scene_seed):
            raise M5AScheduleError("scheduled scene_id is inconsistent with scene_seed")
        if self.task_id != stable_task_id(self.task_spec):
            raise M5AScheduleError("scheduled task_id is inconsistent with task_spec")
        if not self.language_example_id:
            raise M5AScheduleError("scheduled episodes require a language example ID")

    def to_dict(self) -> dict[str, object]:
        return {
            "schedule_id": self.schedule_id,
            "episode_index": self.episode_index,
            "scene_index": self.scene_index,
            "task_index": self.task_index,
            "scene_seed": self.scene_seed,
            "scene_id": self.scene_id,
            "task_spec": self.task_spec.to_dict(),
            "task_id": self.task_id,
            "language_example_id": self.language_example_id,
        }


@dataclass(frozen=True, slots=True)
class M5AScheduleBundle:
    """All four predeclared locks plus development-only materialization."""

    language_development: LanguageScheduleLock
    language_final: LanguageScheduleLock
    control_development: ControlScheduleLock
    control_final: ControlScheduleLock
    development_episodes: tuple[ScheduledControlEpisode, ...]

    def __post_init__(self) -> None:
        if len(self.development_episodes) != 72:
            raise M5AScheduleError("M5A development must contain exactly 72 episodes")
        if self.language_final.sealed is not True or self.control_final.sealed is not True:
            raise M5AScheduleError("both final M5A schedules must remain sealed")


def _select_scene_seeds(config: ControlScheduleConfig) -> tuple[tuple[int, ...], tuple[int, ...]]:
    selected: list[int] = []
    excluded = config.excluded_scene_seeds
    for offset in range(config.maximum_candidate_count):
        candidate = config.candidate_seed_start + offset
        if candidate > _MAX_SCENE_SEED:
            break
        if candidate in excluded:
            continue
        selected.append(candidate)
        if len(selected) == M5A_DEVELOPMENT_SCENE_COUNT + M5A_FINAL_SCENE_COUNT:
            return (
                tuple(selected[:M5A_DEVELOPMENT_SCENE_COUNT]),
                tuple(selected[M5A_DEVELOPMENT_SCENE_COUNT:]),
            )
    raise M5AScheduleError("could not generate 42 unexcluded scene seeds within the bound")


def build_language_schedule_locks(
    corpus: GeneratedLanguageCorpus,
) -> tuple[LanguageScheduleLock, LanguageScheduleLock]:
    """Lock development and final corpus IDs without revealing any final text."""
    development_manifest = corpus.manifest.split_manifests[LanguageSplit.DEVELOPMENT]
    final_manifest = corpus.manifest.split_manifests[LanguageSplit.FINAL]
    development = LanguageScheduleLock(
        schedule_id=M5A_LANGUAGE_DEV_SCHEDULE_ID,
        split=LanguageSplit.DEVELOPMENT,
        ordered_example_ids=development_manifest.example_ids,
        corpus_fingerprint=corpus.manifest.corpus_fingerprint,
        split_fingerprint=development_manifest.content_fingerprint,
        sealed=False,
    )
    final = LanguageScheduleLock(
        schedule_id=M5A_LANGUAGE_FINAL_SCHEDULE_ID,
        split=LanguageSplit.FINAL,
        ordered_example_ids=final_manifest.example_ids,
        corpus_fingerprint=corpus.manifest.corpus_fingerprint,
        split_fingerprint=final_manifest.content_fingerprint,
        sealed=True,
    )
    if not all(
        example_id.startswith(M5A_SEALED_FINAL_EXAMPLE_ID_PREFIX)
        for example_id in final.ordered_example_ids
    ):
        raise M5AScheduleError("final language lock must use only opaque sealed example IDs")
    if set(development.ordered_example_ids) & set(final.ordered_example_ids):
        raise M5AScheduleError("development and final language schedules overlap")
    return development, final


def _associate_language_ids(
    *,
    split: LanguageSplit,
    scene_count: int,
    corpus: GeneratedLanguageCorpus,
) -> tuple[str, ...]:
    associated: list[str] = []
    by_task = corpus.routeable_example_ids_by_task[split]
    for scene_index in range(scene_count):
        for task_spec in CANONICAL_TASK_SPECS:
            task_id = stable_task_id(task_spec)
            candidates = by_task.get(task_id, ())
            if not candidates:
                raise M5AScheduleError(
                    f"language split {split.value} has no routeable examples for {task_id}"
                )
            if split is LanguageSplit.FINAL and not all(
                candidate.startswith(M5A_SEALED_FINAL_EXAMPLE_ID_PREFIX) for candidate in candidates
            ):
                raise M5AScheduleError(
                    "final control association must use opaque sealed language IDs"
                )
            associated.append(candidates[scene_index % len(candidates)])
    return tuple(associated)


def build_m5a_schedule_bundle(
    *,
    corpus: GeneratedLanguageCorpus,
    config: ControlScheduleConfig,
) -> M5AScheduleBundle:
    """Jointly lock all four schedules before router training or development."""
    language_development, language_final = build_language_schedule_locks(corpus)
    development_seeds, final_seeds = _select_scene_seeds(config)
    development_ids = _associate_language_ids(
        split=LanguageSplit.DEVELOPMENT,
        scene_count=M5A_DEVELOPMENT_SCENE_COUNT,
        corpus=corpus,
    )
    final_ids = _associate_language_ids(
        split=LanguageSplit.FINAL,
        scene_count=M5A_FINAL_SCENE_COUNT,
        corpus=corpus,
    )
    development = ControlScheduleLock(
        schedule_id=M5A_CONTROL_DEV_SCHEDULE_ID,
        split=LanguageSplit.DEVELOPMENT,
        ordered_scene_seeds=development_seeds,
        counterpart_scene_seeds=final_seeds,
        ordered_task_specs=CANONICAL_TASK_SPECS,
        associated_language_example_ids=development_ids,
        exclusion_digest=config.exclusion_digest,
        language_schedule_fingerprint=language_development.schedule_fingerprint,
        sealed=False,
    )
    final = ControlScheduleLock(
        schedule_id=M5A_CONTROL_FINAL_SCHEDULE_ID,
        split=LanguageSplit.FINAL,
        ordered_scene_seeds=final_seeds,
        counterpart_scene_seeds=development_seeds,
        ordered_task_specs=CANONICAL_TASK_SPECS,
        associated_language_example_ids=final_ids,
        exclusion_digest=config.exclusion_digest,
        language_schedule_fingerprint=language_final.schedule_fingerprint,
        sealed=True,
    )
    if set(development.ordered_scene_seeds) & config.excluded_scene_seeds:
        raise M5AScheduleError("development control schedule overlaps excluded seeds")
    if set(final.ordered_scene_seeds) & config.excluded_scene_seeds:
        raise M5AScheduleError("final control schedule overlaps excluded seeds")
    development_episodes = materialize_control_schedule(development)
    return M5AScheduleBundle(
        language_development=language_development,
        language_final=language_final,
        control_development=development,
        control_final=final,
        development_episodes=development_episodes,
    )


def materialize_control_schedule(
    schedule: ControlScheduleLock,
    *,
    authorize_final: bool = False,
) -> tuple[ScheduledControlEpisode, ...]:
    """Materialize development freely and final only after an explicit later gate."""
    if schedule.split is LanguageSplit.FINAL and authorize_final is not True:
        raise FinalControlScheduleAccessError(
            "m5a_control_final_v0 is sealed during target-development"
        )
    if schedule.split is not LanguageSplit.FINAL and authorize_final:
        raise FinalControlScheduleAccessError(
            "final authorization cannot be applied to a development schedule"
        )
    episodes: list[ScheduledControlEpisode] = []
    for scene_index, scene_seed in enumerate(schedule.ordered_scene_seeds):
        for task_index, task_spec in enumerate(schedule.ordered_task_specs):
            episode_index = scene_index * len(schedule.ordered_task_specs) + task_index
            episodes.append(
                ScheduledControlEpisode(
                    schedule_id=schedule.schedule_id,
                    episode_index=episode_index,
                    scene_index=scene_index,
                    task_index=task_index,
                    scene_seed=scene_seed,
                    scene_id=stable_scene_id(scene_seed),
                    task_spec=task_spec,
                    task_id=stable_task_id(task_spec),
                    language_example_id=schedule.associated_language_example_ids[episode_index],
                )
            )
    return tuple(episodes)


def exclusion_sources_from_mapping(
    sources: Mapping[str, tuple[str, Sequence[int]]],
) -> tuple[SeedExclusionSource, ...]:
    """Build strict typed exclusion sources from audited artifact outputs."""
    return tuple(
        SeedExclusionSource(
            source_id=source_id,
            source_fingerprint=fingerprint,
            scene_seeds=tuple(scene_seeds),
        )
        for source_id, (fingerprint, scene_seeds) in sorted(sources.items())
    )


def build_authoritative_seed_exclusions(
    *,
    m43_development_fingerprint: str,
    m43_development_scene_seeds: Sequence[int],
) -> tuple[SeedExclusionSource, ...]:
    """Bind every historical M5A seed exclusion to committed prior evidence.

    M4.3 development deliberately reused ``m42_dev_v0``.  Its separate
    evidence fingerprint remains required so the exclusion audit proves both
    prior consumers without inventing another seed schedule.
    """

    _require_fingerprint(
        m43_development_fingerprint,
        name="m43_development_fingerprint",
    )
    prior = load_exclusion_sources()
    m42_development, m42_final = validate_locked_schedules()
    m43_seeds = tuple(sorted(set(m43_development_scene_seeds)))
    if m43_seeds != tuple(sorted(m42_development.ordered_scene_seeds)):
        raise M5AScheduleError(
            "M4.3 development exclusions must exactly reuse m42_dev_v0 scene seeds"
        )
    payload = prior.payload
    m3a = payload["m3a_full"]
    m3b = payload["m3b_full"]
    m4_fresh = payload["m4_full_fresh"]
    m4_smoke_tiny = payload["m4_smoke_tiny"]
    m41_smoke = payload["m41_diagnostic_target_smoke"]
    if not all(
        isinstance(value, Mapping) for value in (m3a, m3b, m4_fresh, m4_smoke_tiny, m41_smoke)
    ):
        raise M5AScheduleError("committed prior seed evidence is malformed")

    def seeds(value: object, *, label: str) -> tuple[int, ...]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise M5AScheduleError(f"{label} must be a seed sequence")
        result = tuple(value)
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in result):
            raise M5AScheduleError(f"{label} contains a non-integer seed")
        return tuple(sorted(set(result)))

    def source_fingerprint(source_id: str, authority: object, values: tuple[int, ...]) -> str:
        return _fingerprint(
            {
                "source_id": source_id,
                "authority": authority,
                "scene_seeds": list(values),
            }
        )

    m3a_mapping = dict(m3a)
    m3b_mapping = dict(m3b)
    m4_fresh_mapping = dict(m4_fresh)
    m4_smoke_mapping = dict(m4_smoke_tiny)
    m41_mapping = dict(m41_smoke)
    values: dict[str, tuple[object, tuple[int, ...]]] = {
        "m3a_accepted": (
            m3a_mapping.get("canonical_manifest_digest"),
            seeds(m3a_mapping.get("accepted_scene_seeds"), label="M3A accepted seeds"),
        ),
        "m3a_rejected_candidates": (
            m3a_mapping.get("canonical_manifest_digest"),
            seeds(
                m3a_mapping.get("rejected_candidate_scene_seeds"),
                label="M3A rejected seeds",
            ),
        ),
        "m3b_train": (
            m3b_mapping.get("split_manifest_digest"),
            seeds(m3b_mapping.get("train_scene_seeds"), label="M3B train seeds"),
        ),
        "m3b_validation": (
            m3b_mapping.get("split_manifest_digest"),
            seeds(
                m3b_mapping.get("validation_scene_seeds"),
                label="M3B validation seeds",
            ),
        ),
        "m3b_test": (
            m3b_mapping.get("split_manifest_digest"),
            seeds(m3b_mapping.get("test_scene_seeds"), label="M3B test seeds"),
        ),
        "m4_fresh": (
            m4_fresh_mapping.get("fresh_seed_schedule_digest"),
            seeds(
                m4_fresh_mapping.get("ordered_scene_seeds"),
                label="M4 fresh seeds",
            ),
        ),
        "m4_smoke": (
            {
                "m4_smoke_tiny": m4_smoke_mapping.get("run_fingerprints"),
                "m41_diagnostic": m41_mapping.get("per_task_rollout_schedule_digest"),
            },
            tuple(
                sorted(
                    set(
                        seeds(
                            m4_smoke_mapping.get("actual_scene_seeds"),
                            label="M4 smoke seeds",
                        )
                    )
                    | set(
                        seeds(
                            m41_mapping.get("actual_scene_seeds"),
                            label="M4.1 smoke seeds",
                        )
                    )
                )
            ),
        ),
        "m4_tiny_overfit": (
            m4_smoke_mapping.get("predeclared_fresh_seed_schedule_digest"),
            seeds(
                m4_smoke_mapping.get("predeclared_fresh_scene_seeds"),
                label="M4 tiny-overfit seeds",
            ),
        ),
        "m42_dev_v0": (
            m42_development.schedule_fingerprint,
            tuple(m42_development.ordered_scene_seeds),
        ),
        "m42_final_v0": (
            m42_final.schedule_fingerprint,
            tuple(m42_final.ordered_scene_seeds),
        ),
        "m43_development": (m43_development_fingerprint, m43_seeds),
    }
    return tuple(
        SeedExclusionSource(
            source_id=source_id,
            source_fingerprint=source_fingerprint(source_id, authority, scene_seeds),
            scene_seeds=scene_seeds,
        )
        for source_id, (authority, scene_seeds) in sorted(values.items())
    )
