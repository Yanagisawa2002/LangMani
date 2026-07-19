from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

from langmani.datasets.identity import sha256_hex
from langmani.environments.specs import EpisodeSpec, TaskSpec
from langmani.language.controller_registry import build_fixture_controller_registry
from langmani.language.corpus import build_language_corpus
from langmani.language.dispatcher import ControllerDispatcher, FixturePerTaskControllerLoader
from langmani.language.router_types import (
    RouterConfidence,
    RouterDecision,
    RouterRejectionReason,
    RouterStatus,
)
from langmani.language.schedules import (
    REQUIRED_EXCLUSION_SOURCE_IDS,
    ControlScheduleConfig,
    SeedExclusionSource,
    build_m5a_schedule_bundle,
)
from langmani.language.three_scene_control import (
    analyze_three_scene_records,
    initial_scene_state_fingerprint,
)
from langmani.language.three_scene_control_evidence import (
    validate_three_scene_evidence,
    write_three_scene_evidence,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _command() -> ModuleType:
    path = PROJECT_ROOT / "scripts" / "run_language_control.py"
    spec = importlib.util.spec_from_file_location("langmani_test_three_scene_command", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Environment:
    def __init__(self) -> None:
        self.last_seed: int | None = None
        self.last_task: dict[str, str] | None = None
        self.reset_count = 0
        self.step_count = 0

    def reset(self, *, seed: int, options: dict[str, object]):  # type: ignore[no-untyped-def]
        self.last_seed = seed
        self.last_task = dict(options["task_spec"])
        self.reset_count += 1
        return {}, {}

    def step(self, action: np.ndarray):  # type: ignore[no-untyped-def]
        assert action.shape == (8,)
        self.step_count += 1
        return {}, 0.0, False, False, {}

    def get_episode_specs(self) -> tuple[EpisodeSpec, ...]:
        assert self.last_seed is not None and self.last_task is not None
        return (
            EpisodeSpec.create(
                scene_seed=self.last_seed,
                task_spec=TaskSpec.from_mapping(self.last_task),
            ),
        )

    def get_expert_initial_scene_state(self) -> dict[str, object]:
        assert self.last_seed is not None
        offset = self.last_seed / 1_000_000.0
        return {
            "object_poses": {
                "red_cube": [0.1 + offset, 0.0, 0.03, 1.0, 0.0, 0.0, 0.0],
                "green_cube": [0.2 + offset, 0.0, 0.03, 1.0, 0.0, 0.0, 0.0],
                "blue_cube": [0.3 + offset, 0.0, 0.03, 1.0, 0.0, 0.0, 0.0],
            },
            "bin_poses": {
                "left_bin": [0.4, 0.2, 0.02, 1.0, 0.0, 0.0, 0.0],
                "right_bin": [0.4, -0.2, 0.02, 1.0, 0.0, 0.0, 0.0],
            },
            "panda_qpos": [0.0] * 9,
        }


class _RejectingRouter:
    def route(self, _command: str) -> RouterDecision:
        return RouterDecision.reject(
            status=RouterStatus.REJECT_UNSUPPORTED,
            rejection_reason=RouterRejectionReason.UNSUPPORTED_ACTION,
            confidence=RouterConfidence.unavailable(),
            router_name="NeuroSymbolicRouterV0",
            router_version="neuro-symbolic-router-v0",
        )


def _inputs(command: ModuleType):
    corpus = build_language_corpus()
    exclusions = tuple(
        SeedExclusionSource(
            source_id=source_id,
            source_fingerprint=f"sha256:{sha256_hex({'source': source_id})}",
            scene_seeds=(),
        )
        for source_id in REQUIRED_EXCLUSION_SOURCE_IDS
    )
    bundle = build_m5a_schedule_bundle(
        corpus=corpus,
        config=ControlScheduleConfig(exclusion_sources=exclusions),
    )
    return command.prepare_development_inputs(
        schedule=bundle.control_development,
        corpus=corpus,
        stage=command.M5AStage.THREE_SCENE_CONTROL_SCREEN,
    )


def _run_fixture(tmp_path: Path):
    command = _command()
    inputs = _inputs(command)
    registry = build_fixture_controller_registry()
    environment = _Environment()
    loader = FixturePerTaskControllerLoader()
    dispatcher = ControllerDispatcher(registry=registry, loader=loader)
    calls: list[tuple[int, str]] = []

    def oracle(episode, example):  # type: ignore[no-untyped-def]
        del example
        calls.append((episode.episode_index, "oracle"))
        return command._oracle_provider(episode, None)

    def learned(episode, example):  # type: ignore[no-untyped-def]
        del example
        calls.append((episode.episode_index, "neuro_symbolic"))
        return RouterDecision.route(
            task_spec=episode.task_spec,
            confidence=RouterConfidence.unavailable(),
            router_name="NeuroSymbolicRouterV0",
            router_version="neuro-symbolic-router-v0",
        )

    default_probe = command.run_rejection_noop_probe(
        registry=registry,
        loader=loader,
        environment=environment,
        router=_RejectingRouter(),
    )
    probe_set = command.run_three_scene_rejection_probes(
        registry=registry,
        loader=loader,
        environment=environment,
        router=_RejectingRouter(),
    )
    result = command.run_development_control(
        inputs=inputs,
        registry=registry,
        dispatcher=dispatcher,
        environment=environment,
        providers={"oracle": oracle, "neuro_symbolic": learned},
        output_root=tmp_path / "control",
        run_identity={"fixture": "three-scene"},
        require_active_episode_spec=True,
        rejection_noop_probe=default_probe,
    )
    root = Path(result["evidence_root"])
    oracle_records = [
        json.loads((root / "episodes" / "oracle" / f"{index:03d}.json").read_text())
        for index in range(18)
    ]
    learned_records = [
        json.loads((root / "episodes" / "neuro_symbolic" / f"{index:03d}.json").read_text())
        for index in range(18)
    ]
    return inputs, calls, probe_set, oracle_records, learned_records


def test_three_scene_runs_oracle_then_learned_with_exact_pairing(tmp_path: Path) -> None:
    inputs, calls, probes, oracle, learned = _run_fixture(tmp_path)
    analysis = analyze_three_scene_records(
        oracle_records=oracle,
        neuro_symbolic_records=learned,
        rejection_probe_set=probes,
    )

    assert len(inputs.schedule.ordered_scene_seeds) == 3
    assert len(inputs.episodes) == 18
    assert calls == [
        (index, router) for index in range(18) for router in ("oracle", "neuro_symbolic")
    ]
    assert analysis["paired_initial_state_count"] == 18
    assert analysis["controller_identity_agreement_count"] == 18
    assert analysis["policy_reset_count"] == 36
    assert analysis["three_scene_control_screen_passed"] is True
    assert analysis["full_control_development_authorized"] is True
    assert analysis["full_control_development_completed"] is False
    assert probes["probe_count"] == 6
    assert probes["environment_step_count"] == 0


def test_three_scene_gate_detects_initial_state_drift(tmp_path: Path) -> None:
    _inputs_value, _calls, probes, oracle, learned = _run_fixture(tmp_path)
    changed = copy.deepcopy(learned)
    state = changed[0]["paired_execution_audit"]["initial_physical_state"]
    state["object_poses"]["red_cube"][0] += 0.01
    changed[0]["paired_execution_audit"]["initial_physical_state_fingerprint"] = (
        initial_scene_state_fingerprint(state)
    )
    analysis = analyze_three_scene_records(
        oracle_records=oracle,
        neuro_symbolic_records=changed,
        rejection_probe_set=probes,
    )

    assert analysis["paired_initial_state_count"] == 17
    assert analysis["gate_items"]["paired_initial_states_complete"] is False
    assert analysis["three_scene_control_screen_passed"] is False


def test_three_scene_evidence_is_atomic_immutable_and_reusable(tmp_path: Path) -> None:
    owner = {
        "schema_version": "owner-v0",
        "run_fingerprint": f"sha256:{'a' * 64}",
    }
    artifacts = {
        "input_contract.json": {"pair_count": 18},
        "summary.md": "# Three-scene fixture\n",
    }
    flags = {"three_scene_control_screen_completed": True}

    first = write_three_scene_evidence(
        tmp_path / "evidence", owner=owner, artifacts=artifacts, flags=flags
    )
    second = write_three_scene_evidence(
        tmp_path / "evidence", owner=owner, artifacts=artifacts, flags=flags
    )

    assert first["evidence_reused"] is False
    assert second["evidence_reused"] is True
    assert validate_three_scene_evidence(first["root"])["passed"] is True
    assert not list((tmp_path / "evidence").glob("*.staging"))
