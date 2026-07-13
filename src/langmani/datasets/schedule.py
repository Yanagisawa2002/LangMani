"""Deterministic counterfactual scene-group scheduling for M3A."""

from __future__ import annotations

from langmani.datasets.identity import (
    environment_version,
    sha256_hex,
    stable_candidate_scene_id,
    stable_collection_run_id,
    stable_scheduled_episode_id,
)
from langmani.datasets.types import CollectionConfig, CollectionSchedule, ScheduledEpisode
from langmani.environments.specs import BIN_IDS, OBJECT_IDS, EpisodeSpec, TaskSpec

CANONICAL_TASK_SPECS: tuple[TaskSpec, ...] = tuple(
    TaskSpec.from_mapping(
        {
            "target_object_id": object_id,
            "target_bin_id": bin_id,
            "instruction_template_id": "canonical_v0",
        }
    )
    for object_id in OBJECT_IDS
    for bin_id in BIN_IDS
)


def build_collection_schedule(config: CollectionConfig) -> CollectionSchedule:
    """Build the complete bounded seed-major, object-major schedule.

    Candidate seeds are consecutive starting at ``candidate_scene_seed_start``.
    Every seed is paired with all six canonical TaskSpecs before the next seed.
    """
    if not isinstance(config, CollectionConfig):
        raise TypeError("config must be a CollectionConfig")

    identity_payload = config.identity_dict()
    config_fingerprint = f"sha256:{sha256_hex(identity_payload)}"
    collection_run_id = stable_collection_run_id(identity_payload)
    seeds = tuple(
        config.candidate_scene_seed_start + offset
        for offset in range(config.maximum_candidate_scene_count)
    )
    episodes: list[ScheduledEpisode] = []
    env_version = environment_version(config.environment_id)
    for candidate_scene_index, scene_seed in enumerate(seeds):
        first_episode = EpisodeSpec.create(
            scene_seed=scene_seed,
            task_spec=CANONICAL_TASK_SPECS[0],
        )
        candidate_scene_id = stable_candidate_scene_id(
            environment_id=config.environment_id,
            scene_seed=scene_seed,
            scene_id=first_episode.scene_id,
            expert_fingerprint=config.expert_config_fingerprint,
            control_mode=config.control_mode,
            schema_version=config.collection_schema_version,
        )
        for task_index, task_spec in enumerate(CANONICAL_TASK_SPECS):
            episode_spec = EpisodeSpec.create(scene_seed=scene_seed, task_spec=task_spec)
            scheduled_episode_id = stable_scheduled_episode_id(
                environment_id=config.environment_id,
                scene_seed=scene_seed,
                scene_id=episode_spec.scene_id,
                task_spec=task_spec,
                task_id=episode_spec.task_id,
                expert_fingerprint=config.expert_config_fingerprint,
                control_mode=config.control_mode,
                schema_version=config.collection_schema_version,
            )
            episodes.append(
                ScheduledEpisode(
                    collection_run_id=collection_run_id,
                    candidate_scene_index=candidate_scene_index,
                    candidate_scene_id=candidate_scene_id,
                    scene_seed=scene_seed,
                    scene_id=episode_spec.scene_id,
                    task_index=task_index,
                    task_spec=task_spec,
                    task_id=episode_spec.task_id,
                    canonical_instruction=episode_spec.canonical_instruction,
                    environment_id=config.environment_id,
                    environment_version=env_version,
                    control_mode=config.control_mode,
                    collection_schema_version=config.collection_schema_version,
                    expert_config_fingerprint=config.expert_config_fingerprint,
                    scheduled_episode_id=scheduled_episode_id,
                )
            )
    return CollectionSchedule(
        collection_run_id=collection_run_id,
        collection_schema_version=config.collection_schema_version,
        config_fingerprint=config_fingerprint,
        canonical_task_specs=CANONICAL_TASK_SPECS,
        candidate_scene_seeds=seeds,
        episodes=tuple(episodes),
    )


def scheduled_scene_group(
    schedule: CollectionSchedule, candidate_scene_index: int
) -> tuple[ScheduledEpisode, ...]:
    """Return the six scheduled tasks for one zero-based candidate index."""
    if not isinstance(schedule, CollectionSchedule):
        raise TypeError("schedule must be a CollectionSchedule")
    if isinstance(candidate_scene_index, bool) or not isinstance(candidate_scene_index, int):
        raise TypeError("candidate_scene_index must be an integer")
    if not 0 <= candidate_scene_index < len(schedule.candidate_scene_seeds):
        raise IndexError("candidate_scene_index is out of range")
    start = candidate_scene_index * 6
    return schedule.episodes[start : start + 6]
