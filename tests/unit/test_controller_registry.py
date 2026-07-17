from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import langmani.language.controller_registry as controller_registry
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import stable_task_id
from langmani.language.controller_registry import (
    LOCKED_EXECUTION_HORIZON,
    ControllerLocation,
    ControllerRegistry,
    ControllerRegistryError,
    ControllerRegistryLocators,
    build_fixture_controller_registry,
)
from langmani.policies.act_action_bounds import ActionBoundConfig, ActionBoundMode


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def test_fixture_registry_contains_exactly_six_canonical_per_task_controllers() -> None:
    registry = build_fixture_controller_registry()

    assert tuple(entry.task_spec for entry in registry.entries) == CANONICAL_TASK_SPECS
    assert tuple(entry.task_id for entry in registry.entries) == tuple(
        stable_task_id(spec) for spec in CANONICAL_TASK_SPECS
    )
    assert len({entry.checkpoint_fingerprint for entry in registry.entries}) == 6
    assert all(entry.execution_horizon == LOCKED_EXECUTION_HORIZON for entry in registry.entries)
    assert all(entry.gripper_mode == "project" for entry in registry.entries)
    assert all(
        entry.action_bound_config.mode is ActionBoundMode.PROJECT for entry in registry.entries
    )


def test_registry_round_trip_preserves_path_independent_fingerprint() -> None:
    registry = build_fixture_controller_registry()

    restored = ControllerRegistry.from_dict(registry.to_dict())

    assert restored == registry
    assert restored.registry_fingerprint == registry.registry_fingerprint
    assert restored.environment_contract_fingerprint == registry.environment_contract_fingerprint
    assert restored.rollout_config_fingerprint == registry.rollout_config_fingerprint


def test_registry_runtime_contract_aggregates_all_six_entries() -> None:
    registry = build_fixture_controller_registry()
    changed = replace(
        registry,
        entries=(
            *registry.entries[:-1],
            replace(
                registry.entries[-1],
                rollout_config={"maximum_episode_steps": 201, "control_frequency_hz": 20},
            ),
        ),
        registry_fingerprint="",
    )

    assert changed.rollout_config_fingerprint != registry.rollout_config_fingerprint
    assert changed.environment_contract_fingerprint == registry.environment_contract_fingerprint


def test_registry_contract_mappings_are_recursively_immutable() -> None:
    entry = build_fixture_controller_registry().entries[0]

    with pytest.raises(TypeError):
        entry.model_config["state_dimension"] = 15  # type: ignore[index]
    normalization = entry.model_config["normalization_mapping"]
    assert isinstance(normalization, dict | tuple) is False
    with pytest.raises(TypeError):
        normalization["STATE"] = "MIN_MAX"  # type: ignore[index]


def test_registry_serialization_never_expands_historical_test_or_fresh_contracts() -> None:
    registry = build_fixture_controller_registry()

    serialized = str(registry.to_dict()).casefold()

    assert "evaluation_schedules" not in serialized
    assert "fresh_seed_schedule" not in serialized
    assert "split': 'test" not in serialized
    with pytest.raises(ControllerRegistryError, match="only deployable I/O metadata"):
        replace(
            registry.entries[0],
            data_contract={
                **dict(registry.entries[0].data_contract),
                "evaluation_schedules": {"test": [123], "fresh_seed": [456]},
            },
        )


def test_registry_fingerprint_excludes_machine_local_run_roots(tmp_path) -> None:
    registry = build_fixture_controller_registry()
    first = ControllerRegistryLocators(
        registry_fingerprint=registry.registry_fingerprint,
        locations=tuple(
            ControllerLocation(
                task_id=entry.task_id,
                run_root=tmp_path / "machine-a" / str(index),
                checkpoint_relative_path="checkpoints/step-100000",
            )
            for index, entry in enumerate(registry.entries)
        ),
    )
    second = ControllerRegistryLocators(
        registry_fingerprint=registry.registry_fingerprint,
        locations=tuple(
            ControllerLocation(
                task_id=entry.task_id,
                run_root=tmp_path / "machine-b" / str(index),
                checkpoint_relative_path="checkpoints/step-100000",
            )
            for index, entry in enumerate(registry.entries)
        ),
    )

    assert first.registry_fingerprint == second.registry_fingerprint
    assert first.locations[0].run_root != second.locations[0].run_root


def test_registry_rejects_missing_duplicate_or_reordered_controller() -> None:
    registry = build_fixture_controller_registry()

    with pytest.raises(ControllerRegistryError, match="exactly six"):
        ControllerRegistry(entries=registry.entries[:-1])
    with pytest.raises(ControllerRegistryError, match="canonical task order"):
        ControllerRegistry(
            entries=(registry.entries[1], registry.entries[0], *registry.entries[2:])
        )
    with pytest.raises(ControllerRegistryError, match="distinct PerTask run"):
        ControllerRegistry(
            entries=(
                registry.entries[0],
                replace(
                    registry.entries[1],
                    run_fingerprint=registry.entries[0].run_fingerprint,
                ),
                *registry.entries[2:],
            )
        )


def test_registry_rejects_unlocked_horizon_or_action_runtime() -> None:
    registry = build_fixture_controller_registry()
    entry = registry.entries[0]

    with pytest.raises(ControllerRegistryError, match="H=10"):
        replace(entry, execution_horizon=5)
    with pytest.raises(ControllerRegistryError, match="project action-bound"):
        replace(entry, action_bound_config=ActionBoundConfig(mode=ActionBoundMode.REJECT))


def test_registry_semantic_change_changes_fingerprint() -> None:
    registry = build_fixture_controller_registry()
    changed = replace(
        registry,
        entries=(
            replace(registry.entries[0], checkpoint_fingerprint=_digest("f")),
            *registry.entries[1:],
        ),
        registry_fingerprint="",
    )

    assert changed.registry_fingerprint != registry.registry_fingerprint


def test_registry_rejects_path_traversal_and_unknown_task() -> None:
    registry = build_fixture_controller_registry()

    with pytest.raises(ControllerRegistryError, match="canonical relative path"):
        ControllerLocation(
            task_id=registry.entries[0].task_id,
            run_root="fixture",
            checkpoint_relative_path="../checkpoint",
        )
    with pytest.raises(ControllerRegistryError, match="unsupported controller task"):
        registry.require("langmani-pick-place-task-v0:yellow_cube:left_bin:canonical_v0")


@pytest.mark.parametrize(
    "relative",
    (
        "test/evaluation_runtime_manifest.json",
        "fresh_seed/evaluation_runtime_manifest.json",
        "comparison.json",
        "verification.json",
    ),
)
def test_metadata_reader_rejects_historical_result_paths(tmp_path: Path, relative: str) -> None:
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ControllerRegistryError, match="prohibited test/fresh/final"):
        controller_registry._read_json_object(path, label="forbidden historical artifact")
