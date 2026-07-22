from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from langmani.environments.push_simulation_state import (
    PushSimulationStateError,
    capture_push_simulation_state,
    restore_push_simulation_state,
)
from langmani.environments.push_specs import PushTaskSpec
from langmani.v2.phase2b3 import (
    PHASE2B3_DEVELOPMENT_SEED_START,
    PHASE2B3_FORMAL_SEED_START,
    MPCRolloutMetrics,
    SimulatorMPCPushConfig,
    build_phase2b3_schedule,
    classify_phase2b3_result,
    collection_authorization,
    generate_mpc_candidates,
    safety_veto_reasons,
    score_mpc_rollout,
    select_best_candidate,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_MANIFEST = PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2b3" / "input_manifest.json"


class _Spec:
    def to_dict(self) -> dict[str, object]:
        return {
            "scene_seed": 69_000,
            "task_spec": PushTaskSpec("blue_cube", "left", "standard").to_dict(),
        }


class _Agent:
    def __init__(self) -> None:
        self.state: dict[str, object] = {}

    def get_controller_state(self) -> dict[str, object]:
        return dict(self.state)

    def set_controller_state(self, state: object) -> None:
        assert isinstance(state, dict)
        self.state = dict(state)


class _BatchedRNG:
    def __init__(self, seed: int) -> None:
        self.rngs = [np.random.RandomState(seed)]


class _FakePushEnvironment:
    def __init__(self) -> None:
        self.num_envs = 1
        self.device = torch.device("cpu")
        self.agent = _Agent()
        self._simulation = {
            "actors": {"target": torch.arange(13, dtype=torch.float32).reshape(1, 13)},
            "articulations": {"panda": torch.arange(31, dtype=torch.float32).reshape(1, 31)},
        }
        for index, name in enumerate(
            (
                "_elapsed_steps",
                "_scene_seeds",
                "_target_object_indices",
                "_target_region_indices",
                "_difficulty_indices",
                "_target_region_centers",
                "_target_region_radii",
                "_initial_object_positions",
                "_initial_target_positions",
                "_stable_success_count",
                "_best_target_distance",
                "_steps_without_progress",
                "_event_wrong_object_contact",
                "_event_wrong_object_displaced",
                "_event_target_outside_workspace",
                "_event_target_overshoot",
                "_event_target_toppled",
                "_event_target_lifted",
                "_event_invalid_action",
                "_event_action_out_of_bounds",
                "_event_no_progress_stall",
                "_event_robot_collision",
                "_last_wrong_object_contact",
                "_last_invalid_action",
                "_last_action_out_of_bounds",
                "_last_robot_collision",
            )
        ):
            setattr(self, name, torch.full((1,), index, dtype=torch.float32))
        self._episode_seed = np.asarray([69_000], dtype=np.int64)
        self._main_seed = [69_000]
        self._episode_rng = np.random.RandomState(1)
        self._main_rng = np.random.RandomState(2)
        self._batched_episode_rng = _BatchedRNG(3)
        self._batched_main_rng = _BatchedRNG(4)

    @property
    def unwrapped(self) -> _FakePushEnvironment:
        return self

    def get_state_dict(self) -> dict[str, object]:
        return self._simulation

    def set_state_dict(self, value: dict[str, object]) -> None:
        self._simulation = value

    def get_episode_specs(self) -> tuple[_Spec, ...]:
        return (_Spec(),)

    def get_push_expert_action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.full(8, -1.0), np.full(8, 1.0)


def _metrics(**overrides: object) -> MPCRolloutMetrics:
    values: dict[str, object] = {
        "target_distance_reduction": 0.02,
        "directional_progress": 0.018,
        "target_containment_progress": 0.01,
        "contact_stability": 1.0,
        "lateral_error": 0.002,
        "harmful_rotation": 0.0,
        "overshoot_risk": 0.0,
        "robot_motion_cost": 0.1,
        "candidate_duration_steps": 12,
    }
    values.update(overrides)
    return MPCRolloutMetrics(**values)  # type: ignore[arg-type]


def test_input_manifest_freezes_prior_evidence_and_authorizations() -> None:
    payload = json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))

    assert payload["source_commit"] == "8476a72b3ce7504e7fef0a08114c6c567da77ba7"
    assert payload["prior_result"] == "RESULT_B_EXPERT_GATE_FAILED"
    assert payload["v2_collection"]["attempted_episodes"] == 0
    assert payload["v2_collection"]["accepted_episodes"] == 0
    assert payload["authorizations"]["phase2b4_collection_authorized"] is False
    assert payload["authorizations"]["phase2c2_training_authorized"] is False
    assert payload["authorizations"]["smolvla_started"] is False


def test_complete_simulation_snapshot_round_trip_and_tamper_detection() -> None:
    source = _FakePushEnvironment()
    destination = _FakePushEnvironment()
    snapshot = capture_push_simulation_state(source)

    destination._simulation["actors"]["target"] += 100  # type: ignore[index,operator]
    destination._elapsed_steps += 8
    destination._episode_rng.uniform()
    restore_push_simulation_state(destination, snapshot)

    assert torch.equal(
        destination._simulation["actors"]["target"],  # type: ignore[index]
        source._simulation["actors"]["target"],  # type: ignore[index]
    )
    assert torch.equal(destination._elapsed_steps, source._elapsed_steps)
    assert destination._episode_rng.get_state()[2] == source._episode_rng.get_state()[2]

    snapshot.task_tensors["_elapsed_steps"].add_(1)
    with pytest.raises(PushSimulationStateError, match="contents changed"):
        restore_push_simulation_state(destination, snapshot)


def test_candidate_generation_is_deterministic_and_geometry_aware() -> None:
    config = SimulatorMPCPushConfig()
    assert config.contact_free_replan_snapshots is True
    common = {
        "object_position": (-0.15, 0.0, 0.025),
        "target_center": (0.12, 0.22, 0.001),
        "distractor_position": (-0.28, 0.24, 0.025),
        "full_containment_center_radius": 0.05,
        "config": config,
    }
    cube = generate_mpc_candidates(task=PushTaskSpec("blue_cube", "left", "standard"), **common)
    cylinder = generate_mpc_candidates(
        task=PushTaskSpec("orange_cylinder", "left", "standard"), **common
    )

    assert cube == generate_mpc_candidates(
        task=PushTaskSpec("blue_cube", "left", "standard"), **common
    )
    assert len(cube) == len(cylinder) == 6
    assert {item.contact_height for item in cube} == {0.025}
    assert {item.contact_height for item in cylinder} == {0.015}
    assert {item.angle_offset_degrees for item in cube} == {-12.0, 0.0, 12.0}
    assert {item.angle_offset_degrees for item in cylinder} == {-8.0, 0.0, 8.0}
    assert all(np.isfinite(item.estimated_distractor_clearance) for item in cube + cylinder)


def test_safety_veto_is_hard_and_scoring_is_reproducible() -> None:
    config = SimulatorMPCPushConfig()
    safe = _metrics()
    unsafe = _metrics(workspace_violation=True)

    assert safety_veto_reasons(safe) == ()
    assert safety_veto_reasons(unsafe) == ("workspace_violation",)
    assert score_mpc_rollout(safe, config) == score_mpc_rollout(safe, config)
    with pytest.raises(ValueError, match="vetoed"):
        score_mpc_rollout(unsafe, config)


def test_candidate_selection_uses_stable_identifier_tie_break() -> None:
    config = SimulatorMPCPushConfig()
    candidates = generate_mpc_candidates(
        task=PushTaskSpec("blue_cube", "left", "standard"),
        object_position=(-0.15, 0.0, 0.025),
        target_center=(0.12, 0.22, 0.001),
        distractor_position=(-0.28, 0.24, 0.025),
        full_containment_center_radius=0.05,
        config=config,
    )
    chosen = select_best_candidate(
        ((candidates[1], _metrics()), (candidates[0], _metrics())), config
    )

    assert chosen is not None
    assert chosen[0].candidate_id == min(candidates[0].candidate_id, candidates[1].candidate_id)


def test_schedules_are_fixed_disjoint_and_cover_all_tasks() -> None:
    development = build_phase2b3_schedule(mode="development")
    formal = build_phase2b3_schedule(mode="formal")

    assert len(development) == 24
    assert len(formal) == 100
    assert development[0][0] == PHASE2B3_DEVELOPMENT_SEED_START
    assert formal[0][0] == PHASE2B3_FORMAL_SEED_START
    assert {seed for seed, _ in development}.isdisjoint(seed for seed, _ in formal)
    assert {task.target_object_id for _, task in development} == {
        "blue_cube",
        "orange_cylinder",
    }
    assert {task.target_region_id for _, task in development} == {
        "left",
        "right",
        "forward_left",
        "forward_right",
    }
    assert {task.difficulty for _, task in development} == {"standard", "hard"}


def test_result_classification_and_authorization_are_fail_closed() -> None:
    zeros = {
        "simulator_errors": 0,
        "nonfinite_actions": 0,
        "action_bound_violations": 0,
        "workspace_violations": 0,
        "wrong_object_interactions": 0,
        "prohibited_object_lifts": 0,
        "unsafe_collisions": 0,
    }
    assert (
        classify_phase2b3_result(
            architecture_valid=False,
            safety_counts=zeros,
            formal_completed_episodes=0,
            formal_successes=0,
        )
        == "RESULT_C"
    )
    assert (
        classify_phase2b3_result(
            architecture_valid=True,
            safety_counts={**zeros, "workspace_violations": 1},
            formal_completed_episodes=1,
            formal_successes=1,
        )
        == "RESULT_C"
    )
    assert (
        classify_phase2b3_result(
            architecture_valid=True,
            safety_counts=zeros,
            formal_completed_episodes=100,
            formal_successes=94,
        )
        == "RESULT_B"
    )
    assert collection_authorization("RESULT_B")["phase2b4_collection_authorized"] is False
    accepted = collection_authorization("RESULT_A")
    assert accepted["phase2b4_collection_authorized"] is True
    assert accepted["phase2c2_training_authorized"] is False
