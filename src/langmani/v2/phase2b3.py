"""Frozen contracts for the Phase 2B.3 simulator-backed MPC push expert."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

import numpy as np

from langmani.environments.push_specs import (
    PUSH_OBJECT_IDS,
    TARGET_REGION_IDS,
    PushTaskSpec,
)

PHASE2B3_EXPERT_ID = "SimulatorMPCPushExpertV1"
PHASE2B3_DATASET_ID = "langmani/phase2b-push-v2"
PHASE2B3_SOURCE_COMMIT = "8476a72b3ce7504e7fef0a08114c6c567da77ba7"
PHASE2B3_FORMAL_SEED_START = 66_300
PHASE2B3_DEVELOPMENT_SEED_START = 69_000
PHASE2B3_ZERO_TOLERANCE_FIELDS = (
    "simulator_errors",
    "nonfinite_actions",
    "action_bound_violations",
    "workspace_violations",
    "wrong_object_interactions",
    "prohibited_object_lifts",
    "unsafe_collisions",
)

Phase2B3Result = Literal["RESULT_A", "RESULT_B", "RESULT_C"]


class MPCStatePhase(StrEnum):
    INITIALIZE = "initialize"
    SELECT_CONTACT_MODE = "select_contact_mode"
    PLAN_PRECONTACT = "plan_precontact"
    EVALUATE_CANDIDATES = "evaluate_candidates"
    EXECUTE_PREFIX = "execute_prefix"
    ASSESS_PROGRESS = "assess_progress"
    REPLAN = "replan"
    VERIFY = "verify"
    TERMINATE = "terminate"


class MPCContactMode(StrEnum):
    DIRECT_CONTINUATION = "direct_continuation"
    RETRACT_AND_REAPPROACH = "retract_and_reapproach"
    ALTERNATE_SIDE = "alternate_side"


@dataclass(frozen=True, slots=True)
class SimulatorMPCPushConfig:
    """Small, bounded and fingerprintable MPC search configuration."""

    control_mode: str = "pd_joint_pos"
    maximum_episode_steps: int = 250
    candidate_horizon_steps: int = 12
    executed_prefix_steps: int = 3
    maximum_mpc_decisions: int = 40
    maximum_recontact_attempts: int = 2
    maximum_no_progress_replans: int = 3
    maximum_planner_retries: int = 1
    candidate_angle_offsets_degrees: tuple[float, ...] = (-12.0, 0.0, 12.0)
    cylinder_angle_offsets_degrees: tuple[float, ...] = (-8.0, 0.0, 8.0)
    candidate_distance_scales: tuple[float, ...] = (0.5, 1.0)
    minimum_push_distance: float = 0.015
    maximum_push_distance: float = 0.06
    cube_contact_height: float = 0.025
    cylinder_contact_height: float = 0.015
    contact_offset: float = 0.045
    precontact_clearance: float = 0.075
    minimum_distractor_clearance: float = 0.015
    minimum_progress: float = 0.001
    translation_agreement_tolerance: float = 2e-4
    orientation_agreement_tolerance_radians: float = 2e-3
    target_distance_agreement_tolerance: float = 2e-4
    development_minimum_success_rate: float = 0.85
    development_minimum_prefix_agreement: float = 0.99
    formal_minimum_success_rate: float = 0.95
    maximum_bounded_repairs: int = 1
    contact_free_replan_snapshots: bool = True
    score_progress_weight: float = 1.0
    score_alignment_weight: float = 0.4
    score_target_weight: float = 0.6
    score_contact_weight: float = 0.15
    score_lateral_weight: float = 0.5
    score_rotation_weight: float = 0.15
    score_overshoot_weight: float = 1.0
    score_motion_weight: float = 0.02
    score_time_weight: float = 0.01

    def __post_init__(self) -> None:
        if self.control_mode != "pd_joint_pos":
            raise ValueError("MPC push expert requires control_mode='pd_joint_pos'")
        if self.maximum_episode_steps != 250:
            raise ValueError("MPC push expert preserves the 250-step episode budget")
        for name in (
            "candidate_horizon_steps",
            "executed_prefix_steps",
            "maximum_mpc_decisions",
            "maximum_recontact_attempts",
            "maximum_no_progress_replans",
            "maximum_planner_retries",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.executed_prefix_steps > self.candidate_horizon_steps:
            raise ValueError("executed prefix cannot exceed candidate horizon")
        if self.maximum_bounded_repairs != 1:
            raise ValueError("Phase 2B.3 permits exactly one bounded repair")
        if self.contact_free_replan_snapshots is not True:
            raise ValueError("MPC sandbox snapshots must be captured at contact-free boundaries")
        if len(self.candidate_angle_offsets_degrees) * len(self.candidate_distance_scales) > 12:
            raise ValueError("MPC candidate set cannot exceed 12 physical rollouts")
        for name in (
            "development_minimum_success_rate",
            "development_minimum_prefix_agreement",
            "formal_minimum_success_rate",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.formal_minimum_success_rate != 0.95:
            raise ValueError("formal expert gate must remain 95%")

    @property
    def maximum_physical_candidates(self) -> int:
        return len(self.candidate_angle_offsets_degrees) * len(self.candidate_distance_scales)

    def to_dict(self) -> dict[str, object]:
        return {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in (
                (field, getattr(self, field)) for field in self.__dataclass_fields__
            )
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class MPCPushCandidate:
    candidate_id: str
    contact_mode: MPCContactMode
    angle_offset_degrees: float
    push_distance: float
    contact_height: float
    push_direction_xy: tuple[float, float]
    precontact_position: tuple[float, float, float]
    contact_position: tuple[float, float, float]
    endpoint_position: tuple[float, float, float]
    estimated_distractor_clearance: float

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "contact_mode": self.contact_mode.value,
            "angle_offset_degrees": self.angle_offset_degrees,
            "push_distance": self.push_distance,
            "contact_height": self.contact_height,
            "push_direction_xy": list(self.push_direction_xy),
            "precontact_position": list(self.precontact_position),
            "contact_position": list(self.contact_position),
            "endpoint_position": list(self.endpoint_position),
            "estimated_distractor_clearance": self.estimated_distractor_clearance,
        }


@dataclass(frozen=True, slots=True)
class MPCRolloutMetrics:
    target_distance_reduction: float
    directional_progress: float
    target_containment_progress: float
    contact_stability: float
    lateral_error: float
    harmful_rotation: float
    overshoot_risk: float
    robot_motion_cost: float
    candidate_duration_steps: int
    workspace_violation: bool = False
    invalid_action: bool = False
    action_out_of_bounds: bool = False
    simulator_error: bool = False
    wrong_object_interaction: bool = False
    prohibited_object_lift: bool = False
    unsafe_collision: bool = False
    target_outside_workspace: bool = False

    def __post_init__(self) -> None:
        for name in (
            "target_distance_reduction",
            "directional_progress",
            "target_containment_progress",
            "contact_stability",
            "lateral_error",
            "harmful_rotation",
            "overshoot_risk",
            "robot_motion_cost",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.candidate_duration_steps < 0:
            raise ValueError("candidate_duration_steps cannot be negative")


def generate_mpc_candidates(
    *,
    task: PushTaskSpec,
    object_position: Sequence[float],
    target_center: Sequence[float],
    distractor_position: Sequence[float],
    full_containment_center_radius: float,
    config: SimulatorMPCPushConfig,
) -> tuple[MPCPushCandidate, ...]:
    """Generate the fixed geometry-aware contact-and-push candidate grid."""

    object_xyz = _vector(object_position, 3, "object_position")
    target_xyz = _vector(target_center, 3, "target_center")
    distractor_xyz = _vector(distractor_position, 3, "distractor_position")
    delta = target_xyz[:2] - object_xyz[:2]
    target_distance = float(np.linalg.norm(delta))
    if target_distance < 1e-6:
        return ()
    direct = delta / target_distance
    angles = (
        config.cylinder_angle_offsets_degrees
        if task.target_object_id == "orange_cylinder"
        else config.candidate_angle_offsets_degrees
    )
    height = (
        config.cylinder_contact_height
        if task.target_object_id == "orange_cylinder"
        else config.cube_contact_height
    )
    nominal_distance = float(
        np.clip(
            target_distance - max(0.0, full_containment_center_radius * 0.5),
            config.minimum_push_distance,
            config.maximum_push_distance,
        )
    )
    values: list[MPCPushCandidate] = []
    for angle in angles:
        direction = _rotate(direct, math.radians(angle))
        contact_xy = object_xyz[:2] - direction * config.contact_offset
        precontact_xy = object_xyz[:2] - direction * (
            config.contact_offset + config.precontact_clearance
        )
        for scale in config.candidate_distance_scales:
            distance = float(
                np.clip(
                    nominal_distance * scale,
                    config.minimum_push_distance,
                    config.maximum_push_distance,
                )
            )
            endpoint_xy = contact_xy + direction * distance
            clearance = _segment_point_clearance(
                precontact_xy,
                endpoint_xy,
                distractor_xyz[:2],
            )
            identifier = f"angle_{angle:+05.1f}_distance_{distance:.3f}"
            values.append(
                MPCPushCandidate(
                    candidate_id=identifier,
                    contact_mode=MPCContactMode.DIRECT_CONTINUATION,
                    angle_offset_degrees=float(angle),
                    push_distance=distance,
                    contact_height=height,
                    push_direction_xy=(float(direction[0]), float(direction[1])),
                    precontact_position=(
                        float(precontact_xy[0]),
                        float(precontact_xy[1]),
                        height,
                    ),
                    contact_position=(
                        float(contact_xy[0]),
                        float(contact_xy[1]),
                        height,
                    ),
                    endpoint_position=(
                        float(endpoint_xy[0]),
                        float(endpoint_xy[1]),
                        height,
                    ),
                    estimated_distractor_clearance=clearance,
                )
            )
    return tuple(values)


def safety_veto_reasons(metrics: MPCRolloutMetrics) -> tuple[str, ...]:
    fields = (
        ("workspace_violation", "workspace_violation"),
        ("invalid_action", "invalid_action"),
        ("action_out_of_bounds", "action_out_of_bounds"),
        ("simulator_error", "simulator_error"),
        ("wrong_object_interaction", "wrong_object_interaction"),
        ("prohibited_object_lift", "prohibited_object_lift"),
        ("unsafe_collision", "unsafe_collision"),
        ("target_outside_workspace", "target_outside_workspace"),
    )
    return tuple(reason for field, reason in fields if getattr(metrics, field))


def score_mpc_rollout(
    metrics: MPCRolloutMetrics,
    config: SimulatorMPCPushConfig,
) -> float:
    """Score one physically observed candidate after every hard veto passes."""

    reasons = safety_veto_reasons(metrics)
    if reasons:
        raise ValueError("vetoed candidate cannot be scored: " + ", ".join(reasons))
    return float(
        config.score_progress_weight * metrics.target_distance_reduction
        + config.score_alignment_weight * metrics.directional_progress
        + config.score_target_weight * metrics.target_containment_progress
        + config.score_contact_weight * metrics.contact_stability
        - config.score_lateral_weight * metrics.lateral_error
        - config.score_rotation_weight * metrics.harmful_rotation
        - config.score_overshoot_weight * metrics.overshoot_risk
        - config.score_motion_weight * metrics.robot_motion_cost
        - config.score_time_weight * metrics.candidate_duration_steps
    )


def select_best_candidate(
    candidates: Sequence[tuple[MPCPushCandidate, MPCRolloutMetrics]],
    config: SimulatorMPCPushConfig,
) -> tuple[MPCPushCandidate, MPCRolloutMetrics, float] | None:
    scored = [
        (candidate, metrics, score_mpc_rollout(metrics, config))
        for candidate, metrics in candidates
        if not safety_veto_reasons(metrics)
    ]
    if not scored:
        return None
    return sorted(scored, key=lambda item: (-item[2], item[0].candidate_id))[0]


def build_phase2b3_schedule(
    *,
    mode: Literal["development", "formal"],
) -> tuple[tuple[int, PushTaskSpec], ...]:
    if mode == "development":
        start, counts = PHASE2B3_DEVELOPMENT_SEED_START, (("standard", 12), ("hard", 12))
    else:
        start, counts = PHASE2B3_FORMAL_SEED_START, (("standard", 60), ("hard", 40))
    combinations = tuple(
        (object_id, region_id) for object_id in PUSH_OBJECT_IDS for region_id in TARGET_REGION_IDS
    )
    values: list[tuple[int, PushTaskSpec]] = []
    index = 0
    for difficulty, count in counts:
        for offset in range(count):
            object_id, region_id = combinations[offset % len(combinations)]
            values.append(
                (
                    start + index,
                    PushTaskSpec(object_id, region_id, difficulty),  # type: ignore[arg-type]
                )
            )
            index += 1
    return tuple(values)


def classify_phase2b3_result(
    *,
    architecture_valid: bool,
    safety_counts: Mapping[str, int],
    formal_completed_episodes: int,
    formal_successes: int,
) -> Phase2B3Result:
    if not architecture_valid or any(
        int(safety_counts.get(field, 0)) != 0 for field in PHASE2B3_ZERO_TOLERANCE_FIELDS
    ):
        return "RESULT_C"
    if formal_completed_episodes == 100 and formal_successes >= 95:
        return "RESULT_A"
    return "RESULT_B"


def collection_authorization(result: Phase2B3Result) -> dict[str, bool]:
    return {
        "phase2b4_collection_authorized": result == "RESULT_A",
        "phase2c2_training_authorized": False,
        "smolvla_started": False,
    }


def verify_phase2b3_result_artifacts(root: Path) -> dict[str, object]:
    """Independently rehash and validate the fail-closed Phase 2B.3 result package."""

    root = root.resolve()
    manifest = _load_json_object(root / "artifact_manifest.json")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("artifact manifest files must be a list")

    hash_checks: dict[str, bool] = {}
    crlf_normalized_hash_checks: dict[str, bool] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("artifact manifest entries must be objects")
        relative = entry.get("path")
        expected = entry.get("sha256")
        if not isinstance(relative, str) or not relative or Path(relative).name != relative:
            raise ValueError("artifact manifest paths must be safe basenames")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"invalid SHA-256 for artifact {relative!r}")
        path = (root / relative).resolve()
        if path.parent != root:
            raise ValueError("artifact path escapes result root")
        contents = path.read_bytes() if path.is_file() else b""
        actual = hashlib.sha256(contents).hexdigest()
        matched = actual == expected
        normalized_match = False
        if not matched and path.suffix == ".json" and b"\r\n" in contents:
            normalized = contents.replace(b"\r\n", b"\n")
            normalized_match = hashlib.sha256(normalized).hexdigest() == expected
            matched = normalized_match
        hash_checks[relative] = matched
        crlf_normalized_hash_checks[relative] = normalized_match

    classification = _load_json_object(root / "result_classification.json")
    clone = _load_json_object(root / "simulator_clone_audit.json")
    sandbox = _load_json_object(root / "sandbox_equivalence_audit.json")
    development = _load_json_object(root / "development_pilot_result.json")
    prediction = _load_json_object(root / "prediction_agreement_audit.json")
    formal = _load_json_object(root / "formal_qualification_result.json")
    authorization = _load_json_object(root / "collection_authorization_decision.json")
    prior = _load_json_object(root / "prior_evidence_verification.json")
    remote = _load_json_object(root / "remote_execution_audit.json")

    attempts = clone.get("attempts")
    final_attempt = attempts[-1] if isinstance(attempts, list) and attempts else {}
    checks = {
        "artifact_manifest_schema_valid": manifest.get("schema_version")
        == "langmani-v2-phase2b3-artifact-manifest-v0",
        "artifact_count_matches": manifest.get("file_count") == len(entries),
        "all_artifact_hashes_valid": bool(hash_checks) and all(hash_checks.values()),
        "prior_evidence_validated": prior.get("verified") is True,
        "phase2b2_result_preserved": prior.get("phase2b2_result") == "RESULT_B_EXPERT_GATE_FAILED",
        "clone_precondition_failed": clone.get("passed") is False
        and clone.get("architecture_valid") is False,
        "three_clone_attempts_preserved": isinstance(attempts, list) and len(attempts) == 3,
        "contact_free_attempt_failed": isinstance(final_attempt, Mapping)
        and final_attempt.get("snapshot_target_contact") is False
        and final_attempt.get("passed") is False,
        "categorical_outcomes_agree": final_attempt.get("categorical_agreement") == 1.0,
        "material_live_pose_divergence_recorded": float(
            final_attempt.get("live_continuation_object_pose_max_abs_error_m", 0.0)
        )
        > float(final_attempt.get("object_pose_tolerance_m", float("inf"))),
        "sandbox_configuration_equivalent": sandbox.get("configuration_equivalent") is True,
        "sandbox_isolation_validated": sandbox.get("main_environment_isolation_validated") is True
        and sandbox.get("other_sandbox_isolation_validated") is True,
        "sandbox_physics_not_equivalent": sandbox.get("physics_equivalent_for_mpc_prediction")
        is False,
        "development_not_run": development.get("status") == "not_run_clone_precondition_failed"
        and development.get("completed_episodes") == 0,
        "prediction_gate_not_run": prediction.get("status") == "not_run_clone_precondition_failed"
        and prediction.get("development_prefixes") == 0,
        "formal_gate_not_run": formal.get("status") == "not_run_architecture_invalid"
        and formal.get("completed_episodes") == 0
        and formal.get("formal_seed_range_accessed") is False,
        "result_c_classified": classification.get("result") == "RESULT_C"
        and classification.get("architecture_valid") is False,
        "collection_blocked": authorization.get("phase2b4_collection_authorized") is False
        and authorization.get("accepted_dataset_package_created") is False
        and authorization.get("raw_episode_archives_created") is False
        and authorization.get("lerobot_dataset_created") is False,
        "training_and_smolvla_blocked": authorization.get("phase2c2_training_authorized") is False
        and authorization.get("smolvla_started") is False
        and authorization.get("optimizer_steps") == 0,
        "remote_audit_failed_closed": remote.get("clone_audit_exit_code") == 2
        and remote.get("development_pilot_started") is False
        and remote.get("formal_qualification_started") is False,
    }
    return {
        "schema_version": "langmani-v2-phase2b3-independent-verification-v0",
        "passed": all(checks.values()),
        "result": "RESULT_C",
        "checks": checks,
        "artifact_hash_checks": hash_checks,
        "crlf_normalized_hash_checks": crlf_normalized_hash_checks,
        "phase2b4_collection_authorized": False,
        "phase2c2_training_authorized": False,
        "smolvla_started": False,
    }


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def _vector(value: Sequence[float], size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite vector of size {size}")
    return result


def _rotate(vector: np.ndarray, angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        (cosine * vector[0] - sine * vector[1], sine * vector[0] + cosine * vector[1]),
        dtype=np.float64,
    )


def _segment_point_clearance(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> float:
    segment = end - start
    denominator = float(np.dot(segment, segment))
    fraction = 0.0 if denominator < 1e-12 else float(np.dot(point - start, segment) / denominator)
    closest = start + np.clip(fraction, 0.0, 1.0) * segment
    return float(np.linalg.norm(point - closest))


__all__ = [
    "MPCContactMode",
    "MPCPushCandidate",
    "MPCRolloutMetrics",
    "MPCStatePhase",
    "PHASE2B3_DATASET_ID",
    "PHASE2B3_DEVELOPMENT_SEED_START",
    "PHASE2B3_EXPERT_ID",
    "PHASE2B3_FORMAL_SEED_START",
    "PHASE2B3_SOURCE_COMMIT",
    "PHASE2B3_ZERO_TOLERANCE_FIELDS",
    "Phase2B3Result",
    "SimulatorMPCPushConfig",
    "build_phase2b3_schedule",
    "classify_phase2b3_result",
    "collection_authorization",
    "generate_mpc_candidates",
    "safety_veto_reasons",
    "score_mpc_rollout",
    "select_best_candidate",
    "verify_phase2b3_result_artifacts",
]
