from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from langmani.environments.specs import TaskSpec, stable_task_id
from langmani.language import dispatcher as dispatcher_module
from langmani.language.controller_registry import (
    ControllerLocation,
    ControllerRegistryLocators,
    build_fixture_controller_registry,
)
from langmani.language.dispatcher import (
    ControllerDispatcher,
    ControllerLoadError,
    FixturePerTaskControllerLoader,
    StrictPerTaskControllerLoader,
)
from langmani.language.failure_attribution import FailureAttribution
from langmani.language.router_types import (
    RouterConfidence,
    RouterDecision,
    RouterRejectionReason,
    RouterStatus,
)
from langmani.policies.act_checkpoint import CheckpointComponentFingerprints


@dataclass
class _SpyEnvironment:
    reset_count: int = 0
    step_count: int = 0
    reset_task_specs: list[dict[str, str]] = field(default_factory=list)
    executed_actions: list[np.ndarray] = field(default_factory=list)

    def reset(self, *, seed: int, options: dict[str, object]) -> tuple[object, dict[str, object]]:
        assert seed >= 0
        self.reset_count += 1
        self.reset_task_specs.append(dict(options["task_spec"]))
        return {}, {}

    def step(self, action: np.ndarray) -> tuple[object, float, bool, bool, dict[str, object]]:
        self.step_count += 1
        self.executed_actions.append(np.array(action, copy=True))
        return {}, 0.0, False, False, {}


def _route(task_spec: TaskSpec) -> RouterDecision:
    return RouterDecision.route(
        task_spec=task_spec,
        confidence=RouterConfidence.unavailable(),
        router_name="fixture-router",
        router_version="v0",
    )


def _reject() -> RouterDecision:
    return RouterDecision.reject(
        status=RouterStatus.REJECT_AMBIGUOUS,
        rejection_reason=RouterRejectionReason.CONFLICTING_OBJECTS,
        confidence=RouterConfidence.unavailable(),
        router_name="fixture-router",
        router_version="v0",
    )


def test_correct_route_selects_exactly_one_controller() -> None:
    registry = build_fixture_controller_registry()
    loader = FixturePerTaskControllerLoader()
    dispatcher = ControllerDispatcher(registry=registry, loader=loader)
    environment = _SpyEnvironment()
    task_spec = TaskSpec("green_cube", "right_bin", "canonical_v0")

    result = dispatcher.dispatch(
        _route(task_spec),
        environment=environment,
        evaluation_id="correct-route",
        oracle_task_spec=task_spec,
        scene_seed=123,
    )

    assert loader.load_count == 1
    assert result.dispatch.dispatched
    assert result.dispatch.controller_task_id == stable_task_id(task_spec)
    assert result.failure_attribution is FailureAttribution.ROUTING_CORRECT_CONTROL_SUCCESS
    assert environment.reset_count == environment.step_count == 1


def test_rejection_returns_before_lookup_load_policy_reset_and_environment() -> None:
    registry = build_fixture_controller_registry()
    loader = FixturePerTaskControllerLoader()
    dispatcher = ControllerDispatcher(registry=registry, loader=loader)
    environment = _SpyEnvironment()
    oracle = TaskSpec("red_cube", "left_bin", "canonical_v0")

    result = dispatcher.dispatch(
        _reject(),
        environment=environment,
        evaluation_id="false-rejection",
        oracle_task_spec=oracle,
        scene_seed=2,
    )

    assert loader.load_count == 0
    assert loader.executors == {}
    assert environment.reset_count == environment.step_count == 0
    assert result.dispatch.safe_rejection
    assert result.dispatch.policy_called is False
    assert result.failure_attribution is FailureAttribution.ROUTING_FALSE_REJECTION


def test_correct_language_rejection_has_no_control_attribution() -> None:
    registry = build_fixture_controller_registry()
    loader = FixturePerTaskControllerLoader()
    environment = _SpyEnvironment()

    result = ControllerDispatcher(registry=registry, loader=loader).dispatch(
        _reject(),
        environment=environment,
        evaluation_id="correct-rejection",
        oracle_task_spec=None,
        scene_seed=None,
    )

    assert result.failure_attribution is None
    assert result.dispatch.safe_rejection
    assert loader.load_count == environment.reset_count == environment.step_count == 0


def test_wrong_route_uses_predicted_controller_but_resets_oracle_task() -> None:
    registry = build_fixture_controller_registry()
    loader = FixturePerTaskControllerLoader()
    dispatcher = ControllerDispatcher(registry=registry, loader=loader)
    environment = _SpyEnvironment()
    oracle = TaskSpec("red_cube", "left_bin", "canonical_v0")
    predicted = TaskSpec("blue_cube", "right_bin", "canonical_v0")

    result = dispatcher.dispatch(
        _route(predicted),
        environment=environment,
        evaluation_id="wrong-route",
        oracle_task_spec=oracle,
        scene_seed=8,
    )

    assert result.dispatch.oracle_task_id == stable_task_id(oracle)
    assert result.dispatch.predicted_task_id == stable_task_id(predicted)
    assert result.dispatch.controller_task_id == stable_task_id(predicted)
    assert environment.reset_task_specs == [oracle.to_dict()]
    assert result.failure_attribution is FailureAttribution.ROUTING_WRONG_TASK


def test_policy_state_resets_for_every_episode_and_loader_may_reuse_controller() -> None:
    registry = build_fixture_controller_registry()
    loader = FixturePerTaskControllerLoader()
    dispatcher = ControllerDispatcher(registry=registry, loader=loader)
    environment = _SpyEnvironment()
    task_spec = TaskSpec("blue_cube", "left_bin", "canonical_v0")

    for scene_seed in (10, 11):
        dispatcher.dispatch(
            _route(task_spec),
            environment=environment,
            evaluation_id=f"episode-{scene_seed}",
            oracle_task_spec=task_spec,
            scene_seed=scene_seed,
        )

    executor = loader.executors[(id(environment), stable_task_id(task_spec))]
    assert len(loader.executors) == 1
    assert executor.episode_reset_count == 2
    assert executor.policy_call_count == 2
    assert environment.reset_count == environment.step_count == 2


def test_all_four_action_stream_contracts_remain_separate() -> None:
    registry = build_fixture_controller_registry()
    loader = FixturePerTaskControllerLoader()
    environment = _SpyEnvironment()
    task_spec = TaskSpec("red_cube", "right_bin", "canonical_v0")

    result = ControllerDispatcher(registry=registry, loader=loader).dispatch(
        _route(task_spec),
        environment=environment,
        evaluation_id="action-evidence",
        oracle_task_spec=task_spec,
        scene_seed=17,
    )

    assert result.control is not None
    evidence = result.control.action_evidence
    assert evidence.binary_transformed_actions == (None,)
    assert evidence.raw_actions is not evidence.projected_actions
    assert evidence.projected_actions is not evidence.executed_actions
    assert evidence.binary_transformed_fingerprint not in {
        evidence.raw_fingerprint,
        evidence.projected_fingerprint,
        evidence.executed_fingerprint,
    }
    assert evidence.raw_fingerprint == evidence.projected_fingerprint
    assert evidence.projected_fingerprint == evidence.executed_fingerprint


def test_routed_rejection_only_record_is_missed_rejection_without_execution() -> None:
    registry = build_fixture_controller_registry()
    loader = FixturePerTaskControllerLoader()
    environment = _SpyEnvironment()

    result = ControllerDispatcher(registry=registry, loader=loader).dispatch(
        _route(TaskSpec("green_cube", "left_bin", "canonical_v0")),
        environment=environment,
        evaluation_id="missed-rejection",
        oracle_task_spec=None,
        scene_seed=None,
    )

    assert result.failure_attribution is FailureAttribution.ROUTING_MISSED_REJECTION
    assert loader.load_count == environment.reset_count == environment.step_count == 0


def _fixture_locators(registry, tmp_path: Path) -> ControllerRegistryLocators:
    locations = []
    for index, entry in enumerate(registry.entries):
        run_root = tmp_path / f"run-{index}"
        run_root.mkdir()
        locations.append(
            ControllerLocation(
                task_id=entry.task_id,
                run_root=run_root,
                checkpoint_relative_path="checkpoints/step-100000",
            )
        )
    return ControllerRegistryLocators(
        registry_fingerprint=registry.registry_fingerprint,
        locations=tuple(locations),
    )


def _bounded_environment() -> object:
    return SimpleNamespace(
        single_action_space=SimpleNamespace(
            low=np.full((8,), -1.0, dtype=np.float32),
            high=np.full((8,), 1.0, dtype=np.float32),
        )
    )


def test_strict_loader_verifies_all_registry_fingerprints_and_caches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry = build_fixture_controller_registry()
    entry = registry.entries[0]
    locators = _fixture_locators(registry, tmp_path)
    calls: list[dict[str, object]] = []
    loaded = SimpleNamespace(
        policy=object(),
        preprocessor=object(),
        postprocessor=object(),
        record=SimpleNamespace(
            global_step=entry.checkpoint_step,
            relative_path="checkpoints/step-100000",
        ),
        component_fingerprints=CheckpointComponentFingerprints(
            model=entry.model_component_fingerprint,
            preprocessor=entry.preprocessor_fingerprint,
            postprocessor=entry.postprocessor_fingerprint,
        ),
    )
    context = SimpleNamespace(
        loaded=loaded,
        descriptor=SimpleNamespace(
            statistics_fingerprint=entry.train_statistics_fingerprint,
            split_digest=entry.m3b_split_manifest_digest,
            git_commit=entry.producer_git_commit,
        ),
    )

    def fake_checkpoint_load(run_root: Path, **kwargs: object) -> object:
        calls.append({"run_root": run_root, **kwargs})
        return context

    class FakeExecutor:
        def __init__(self, **kwargs: object) -> None:
            selected = kwargs["entry"]
            self.controller_task_id = selected.task_id

    monkeypatch.setattr(dispatcher_module, "load_m4_checkpoint_context", fake_checkpoint_load)
    monkeypatch.setattr(dispatcher_module, "_ActPerTaskExecutor", FakeExecutor)
    loader = StrictPerTaskControllerLoader(
        registry=registry,
        locators=locators,
        schedule_digest="sha256:" + "d" * 64,
    )
    environment = _bounded_environment()

    first = loader.load(entry, environment=environment)
    second = loader.load(entry, environment=environment)

    assert first is second
    assert len(calls) == 1
    assert calls[0]["expected_checkpoint_fingerprint"] == entry.checkpoint_fingerprint
    assert calls[0]["expected_run_fingerprint"] == entry.run_fingerprint
    assert calls[0]["expected_dataset_fingerprint"] == entry.m3b_export_fingerprint
    assert calls[0]["expected_task_id"] == entry.task_id


def test_strict_loader_rejects_component_fingerprint_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry = build_fixture_controller_registry()
    entry = registry.entries[0]
    locators = _fixture_locators(registry, tmp_path)
    loaded = SimpleNamespace(
        record=SimpleNamespace(
            global_step=entry.checkpoint_step,
            relative_path="checkpoints/step-100000",
        ),
        component_fingerprints=CheckpointComponentFingerprints(
            model="sha256:" + "0" * 64,
            preprocessor=entry.preprocessor_fingerprint,
            postprocessor=entry.postprocessor_fingerprint,
        ),
    )
    context = SimpleNamespace(
        loaded=loaded,
        descriptor=SimpleNamespace(
            statistics_fingerprint=entry.train_statistics_fingerprint,
            split_digest=entry.m3b_split_manifest_digest,
            git_commit=entry.producer_git_commit,
        ),
    )
    monkeypatch.setattr(
        dispatcher_module,
        "load_m4_checkpoint_context",
        lambda *_args, **_kwargs: context,
    )
    loader = StrictPerTaskControllerLoader(
        registry=registry,
        locators=locators,
        schedule_digest="sha256:" + "d" * 64,
    )

    with pytest.raises(ControllerLoadError, match="differs from registry"):
        loader.load(entry, environment=_bounded_environment())


def test_dispatcher_has_no_m2_expert_dependency() -> None:
    source = Path(dispatcher_module.__file__).read_text(encoding="utf-8")

    assert "langmani.experts" not in source
    assert "PickPlaceExpert" not in source
