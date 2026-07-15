from __future__ import annotations

import json
from dataclasses import replace

import pytest

import langmani.policies.m42_schedule as schedule_module
from langmani.datasets.identity import sha256_hex
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import stable_scene_id, stable_task_id
from langmani.policies.m42_schedule import (
    M42_DEV_SCHEDULE_FINGERPRINT,
    M42_DEV_SCHEDULE_ID,
    M42_EVALUATION_CONFIG_FINGERPRINT,
    M42_EXCLUSION_DIGEST,
    M42_FINAL_SCHEDULE_FINGERPRINT,
    M42_FINAL_SCHEDULE_ID,
    FinalScheduleAccessError,
    FinalScheduleAuthorization,
    M42ScheduleError,
    M42SeedSchedule,
    evaluation_config_payload,
    generate_locked_scene_seeds,
    load_exclusion_sources,
    load_locked_schedule,
    materialize_locked_schedule,
    materialize_schedule,
    validate_locked_schedules,
)

EXPECTED_DEV_SEEDS = (
    722950248,
    1851299512,
    713553274,
    1707433820,
    1405538489,
    1953463567,
    1994548171,
    1805875340,
    520095534,
    400105183,
    1993348904,
    584965131,
)

EXPECTED_FINAL_SEEDS = (
    893589536,
    982761247,
    1647023593,
    694109670,
    145992996,
    544351154,
    2073139116,
    908051858,
    651466276,
    2055735584,
    1942539159,
    978613149,
    64981643,
    1732076572,
    47175200,
    451960684,
    1196965561,
    272982124,
    737985780,
    1699051323,
    138690626,
    1098325036,
    1544603825,
    214610042,
    905027018,
    568018298,
    742414486,
    1073839369,
    2086293999,
    824728764,
)


def _authorization(**changes: object) -> FinalScheduleAuthorization:
    fingerprint = "sha256:" + "a" * 64
    values: dict[str, object] = {
        "final_schedule_fingerprint": M42_FINAL_SCHEDULE_FINGERPRINT,
        "expected_implementation_fingerprint": fingerprint,
        "actual_implementation_fingerprint": fingerprint,
        "explicit_final_authorization": True,
        "clean_git": True,
        "horizon_selection_locked": True,
        "gripper_selection_locked": True,
        "task_token_checkpoint_selection_locked": True,
    }
    values.update(changes)
    return FinalScheduleAuthorization(**values)  # type: ignore[arg-type]


def test_exclusion_lock_covers_every_prior_seed_source() -> None:
    lock = load_exclusion_sources()
    excluded = lock.excluded_scene_seeds
    assert lock.exclusion_digest == M42_EXCLUSION_DIGEST
    assert len(excluded) == 125
    assert set(range(65)) <= excluded
    assert {13, 26, 43, 50, 60} <= excluded
    assert 257025151 in excluded
    assert 1590420839 in excluded


def test_exclusion_contract_is_immutable_and_json_serializable() -> None:
    lock = load_exclusion_sources()
    with pytest.raises(TypeError):
        lock.payload["schema_version"] = "tampered"  # type: ignore[index]
    assert json.loads(json.dumps(lock.to_dict()))["exclusion_digest"] == M42_EXCLUSION_DIGEST


def test_evaluation_configuration_fingerprint_is_stable() -> None:
    assert "sha256:" + sha256_hex(evaluation_config_payload()) == M42_EVALUATION_CONFIG_FINGERPRINT


def test_joint_generation_reproduces_both_committed_lists() -> None:
    first = generate_locked_scene_seeds()
    second = generate_locked_scene_seeds()
    assert first == second
    assert first == (EXPECTED_DEV_SEEDS, EXPECTED_FINAL_SEEDS, 42)


def test_joint_generation_fails_at_the_explicit_attempt_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exclusions = load_exclusion_sources()
    monkeypatch.setattr(schedule_module, "sha256_hex", lambda _payload: "0" * 64)
    with pytest.raises(M42ScheduleError, match="within 42 attempts"):
        generate_locked_scene_seeds(exclusions, max_attempts=42)


def test_locked_schedules_are_disjoint_and_exclude_all_prior_seeds() -> None:
    development, final = validate_locked_schedules()
    excluded = load_exclusion_sources().excluded_scene_seeds
    assert development.ordered_scene_seeds == EXPECTED_DEV_SEEDS
    assert final.ordered_scene_seeds == EXPECTED_FINAL_SEEDS
    assert development.schedule_fingerprint == M42_DEV_SCHEDULE_FINGERPRINT
    assert final.schedule_fingerprint == M42_FINAL_SCHEDULE_FINGERPRINT
    assert not set(EXPECTED_DEV_SEEDS) & set(EXPECTED_FINAL_SEEDS)
    assert not excluded & (set(EXPECTED_DEV_SEEDS) | set(EXPECTED_FINAL_SEEDS))


def test_schedule_task_order_is_canonical_object_major_bin_minor() -> None:
    development = load_locked_schedule(M42_DEV_SCHEDULE_ID)
    assert development.ordered_task_specs == CANONICAL_TASK_SPECS
    assert tuple(
        (task.target_object_id, task.target_bin_id) for task in development.ordered_task_specs
    ) == (
        ("red_cube", "left_bin"),
        ("red_cube", "right_bin"),
        ("green_cube", "left_bin"),
        ("green_cube", "right_bin"),
        ("blue_cube", "left_bin"),
        ("blue_cube", "right_bin"),
    )


def test_development_materialization_is_seed_major_and_contains_72_episodes() -> None:
    episodes = materialize_locked_schedule(M42_DEV_SCHEDULE_ID)
    assert len(episodes) == 72
    assert [item.episode_index for item in episodes] == list(range(72))
    assert tuple(item.task_spec for item in episodes[:6]) == CANONICAL_TASK_SPECS
    assert {item.scene_seed for item in episodes[:6]} == {EXPECTED_DEV_SEEDS[0]}
    assert episodes[0].scene_id == stable_scene_id(EXPECTED_DEV_SEEDS[0])
    assert episodes[0].task_id == stable_task_id(CANONICAL_TASK_SPECS[0])


def test_development_can_validate_final_lock_without_materializing_it() -> None:
    development, final = validate_locked_schedules()
    assert development.schedule_id == M42_DEV_SCHEDULE_ID
    assert final.schedule_id == M42_FINAL_SCHEDULE_ID
    with pytest.raises(FinalScheduleAccessError, match="explicit final authorization"):
        materialize_schedule(final)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"explicit_final_authorization": False}, "explicit final flag"),
        ({"clean_git": False}, "dirty"),
        ({"horizon_selection_locked": False}, "horizon"),
        ({"gripper_selection_locked": False}, "gripper"),
        ({"task_token_checkpoint_selection_locked": False}, "TaskToken"),
        (
            {"actual_implementation_fingerprint": "sha256:" + "b" * 64},
            "implementation fingerprint",
        ),
        ({"final_schedule_fingerprint": "sha256:" + "c" * 64}, "schedule fingerprint"),
    ],
)
def test_final_materialization_rejects_incomplete_authorization(
    changes: dict[str, object], message: str
) -> None:
    final = load_locked_schedule(M42_FINAL_SCHEDULE_ID)
    with pytest.raises(FinalScheduleAccessError, match=message):
        materialize_schedule(final, final_authorization=_authorization(**changes))


def test_explicit_final_authorization_materializes_exactly_180_episodes() -> None:
    episodes = materialize_locked_schedule(
        M42_FINAL_SCHEDULE_ID,
        final_authorization=_authorization(),
    )
    assert len(episodes) == 180
    assert tuple(item.scene_seed for item in episodes[::6]) == EXPECTED_FINAL_SEEDS
    assert all(item.schedule_id == M42_FINAL_SCHEDULE_ID for item in episodes)


def test_authorization_cannot_be_reused_for_development() -> None:
    development = load_locked_schedule(M42_DEV_SCHEDULE_ID)
    with pytest.raises(FinalScheduleAccessError, match="cannot be used for development"):
        materialize_schedule(development, final_authorization=_authorization())


def test_schedule_fingerprint_detects_seed_tampering() -> None:
    schedule = load_locked_schedule(M42_DEV_SCHEDULE_ID)
    payload = schedule.to_dict()
    payload["ordered_scene_seeds"] = [*EXPECTED_DEV_SEEDS[:-1], 123456]
    with pytest.raises(M42ScheduleError, match="fingerprint"):
        M42SeedSchedule.from_dict(payload)


def test_dataclass_replacement_cannot_bypass_source_lock() -> None:
    schedule = load_locked_schedule(M42_DEV_SCHEDULE_ID)
    with pytest.raises(M42ScheduleError, match="fingerprint"):
        replace(schedule, ordered_scene_seeds=(*EXPECTED_DEV_SEEDS[:-1], 123456))
