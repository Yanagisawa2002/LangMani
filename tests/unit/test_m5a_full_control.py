from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from langmani.datasets.identity import sha256_hex
from langmani.environments.specs import EpisodeSpec, TaskSpec
from langmani.language.controller_registry import build_fixture_controller_registry
from langmani.language.corpus import build_language_corpus
from langmani.language.dispatcher import ControllerDispatcher, FixturePerTaskControllerLoader
from langmani.language.full_control_development import (
    analyze_full_control_records,
    build_final_authorization,
    initial_scene_state_fingerprint,
)
from langmani.language.full_control_evidence import (
    FullControlEvidenceError,
    validate_full_control_evidence,
    write_full_control_evidence,
)
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _command() -> ModuleType:
    path = PROJECT_ROOT / "scripts" / "run_language_control.py"
    spec = importlib.util.spec_from_file_location("langmani_test_full_control_command", path)
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


def _run_fixture(tmp_path: Path):
    command = _command()
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
    inputs = command.prepare_development_inputs(
        schedule=bundle.control_development,
        corpus=corpus,
        stage=command.M5AStage.FULL_CONTROL_DEVELOPMENT,
    )
    registry = build_fixture_controller_registry()
    environment = _Environment()
    loader = FixturePerTaskControllerLoader()
    dispatcher = ControllerDispatcher(registry=registry, loader=loader)
    calls: list[tuple[int, str]] = []

    def oracle(episode, _example):  # type: ignore[no-untyped-def]
        calls.append((episode.episode_index, "oracle"))
        return command._oracle_provider(episode, None)

    def learned(episode, _example):  # type: ignore[no-untyped-def]
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
    probes = command.run_full_control_rejection_probes(
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
        run_identity={"fixture": "full-control"},
        require_active_episode_spec=True,
        rejection_noop_probe=default_probe,
    )
    root = Path(result["evidence_root"])
    oracle_records = [
        json.loads((root / "episodes" / "oracle" / f"{index:03d}.json").read_text())
        for index in range(36)
    ]
    learned_records = [
        json.loads((root / "episodes" / "neuro_symbolic" / f"{index:03d}.json").read_text())
        for index in range(36)
    ]
    for record in (*oracle_records, *learned_records):
        audit = record["paired_execution_audit"]
        rollout = audit["rollout"]
        projection = rollout["action_projection_summary"]
        projection.update(
            {
                "total_policy_actions": 1,
                "projected_action_count": 0,
                "projected_component_count": 0,
                "maximum_bound_excess": 0.0,
                "per_action_dimension_projection_counts": [0] * 8,
            }
        )
        rollout.update(
            {
                "target_grasped_any": True,
                "wrong_object_grasped_any": False,
                "target_in_wrong_bin": False,
                "target_off_table": False,
                "inference_latency_ms": [],
                "environment_step_latency_ms": [],
                "total_episode_duration_s": 0.05,
            }
        )
    analysis = analyze_full_control_records(
        oracle_records=oracle_records,
        neuro_symbolic_records=learned_records,
        rejection_probe_set=probes,
        examples_by_id=inputs.examples_by_id,
        router_summaries=result["summaries"],
    )
    return inputs, environment, calls, probes, oracle_records, learned_records, analysis


def test_full_control_runs_exact_paired_budget_and_gate(tmp_path: Path) -> None:
    inputs, environment, calls, probes, _oracle, _learned, analysis = _run_fixture(tmp_path)

    assert len(inputs.schedule.ordered_scene_seeds) == 6
    assert len(inputs.episodes) == 36
    assert calls == [
        (index, router) for index in range(36) for router in ("oracle", "neuro_symbolic")
    ]
    assert environment.reset_count == 72
    assert probes["probe_count"] == 9
    assert probes["controller_lookup_count"] == 0
    assert probes["environment_reset_count"] == 0
    assert probes["environment_step_count"] == 0
    assert analysis["paired_initial_state_count"] == 36
    assert analysis["controller_identity_agreement_count"] == 36
    assert analysis["policy_reset_count"] == 72
    assert analysis["routing_correct_count"] == 36
    assert analysis["development_quality_gate_passed"] is True
    assert all(analysis["gate_items"].values())


def test_full_control_gate_recomputes_state_and_exact_route_threshold(tmp_path: Path) -> None:
    inputs, _environment, _calls, probes, oracle, learned, _analysis = _run_fixture(tmp_path)
    changed = copy.deepcopy(learned)
    state = changed[0]["paired_execution_audit"]["initial_physical_state"]
    state["object_poses"]["red_cube"][0] += 0.01
    changed[0]["paired_execution_audit"]["initial_physical_state_fingerprint"] = (
        initial_scene_state_fingerprint(state)
    )
    analysis = analyze_full_control_records(
        oracle_records=oracle,
        neuro_symbolic_records=changed,
        rejection_probe_set=probes,
        examples_by_id=inputs.examples_by_id,
        router_summaries=_analysis["router_summaries"],
    )

    assert analysis["paired_initial_state_count"] == 35
    assert analysis["development_quality_gate_passed"] is False


def test_full_control_final_authorization_is_bounded_and_explicit() -> None:
    common = {
        "router_fingerprint": f"sha256:{'1' * 64}",
        "controller_registry_fingerprint": f"sha256:{'2' * 64}",
        "runtime_fingerprint": f"sha256:{'3' * 64}",
        "full_schedule_fingerprint": f"sha256:{'4' * 64}",
        "development_evidence_fingerprint": f"sha256:{'5' * 64}",
        "selected_router_identity": "NeuroSymbolicRouterV0",
        "final_language_schedule_fingerprint": f"sha256:{'6' * 64}",
        "final_control_schedule_fingerprint": f"sha256:{'7' * 64}",
        "git_commit": "a" * 40,
    }
    accepted = build_final_authorization(authorized=True, **common)
    rejected = build_final_authorization(authorized=False, **common)

    assert accepted["authorized"] is True
    assert rejected["authorized"] is False
    assert accepted["final_data_accessed"] is False
    assert accepted["automatic_final_execution"] is False
    assert accepted["final_scene_seeds_materialized"] is False
    assert accepted["authorization_fingerprint"] != rejected["authorization_fingerprint"]


def test_full_control_evidence_is_atomic_immutable_and_path_safe(tmp_path: Path) -> None:
    owner = {"schema_version": "owner-v0", "run_fingerprint": f"sha256:{'a' * 64}"}
    artifacts = {"input_contract.json": {"pair_count": 36}, "summary.md": "# Full\n"}
    flags = {"full_control_development_completed": True}
    first = write_full_control_evidence(
        tmp_path / "evidence", owner=owner, artifacts=artifacts, flags=flags
    )
    second = write_full_control_evidence(
        tmp_path / "evidence", owner=owner, artifacts=artifacts, flags=flags
    )

    assert first["evidence_reused"] is False
    assert second["evidence_reused"] is True
    assert validate_full_control_evidence(first["root"])["passed"] is True
    assert not list((tmp_path / "evidence").glob("*.staging"))
    with pytest.raises(FullControlEvidenceError, match="unsafe"):
        write_full_control_evidence(
            tmp_path / "unsafe",
            owner={**owner, "run_fingerprint": f"sha256:{'b' * 64}"},
            artifacts={"../escape.json": {}},
            flags=flags,
        )
