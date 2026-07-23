"""Write deterministic pre-learning contracts for Phase 2B.4-F0."""

from __future__ import annotations

import argparse
from pathlib import Path

from langmani.v2.phase2b4_ppo import (
    F0_ALLOWED_OPERATIONS,
    F0_PROHIBITED_OPERATIONS,
    OFFICIAL_PPO_REVISION,
    SOURCE_COMMIT,
    PPOConfig,
    audit_reward_hacking_scenarios,
    canonical_json_sha256,
    normalization_manifest,
    reward_manifest,
    seed_disjointness_audit,
    student_privilege_exclusion_audit,
    teacher_observation_manifest,
)
from langmani.v2.phase2b4_runtime import write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2b4_f0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def _implementation_audit() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "langmani-v2-phase2b4-f0-ppo-implementation-audit-v0",
        "official_project": "ManiSkill",
        "official_version": "3.0.1",
        "official_revision": OFFICIAL_PPO_REVISION,
        "official_source": (
            "https://github.com/haosulab/ManiSkill/blob/v3.0.1/examples/baselines/ppo/ppo.py"
        ),
        "adapted_components": [
            "GPU vectorized state-environment structure",
            "separate three-layer tanh actor and critic MLPs",
            "rollout tensor storage",
            "generalized advantage estimation",
            "clipped PPO surrogate",
            "Adam optimization",
            "deterministic mean-policy evaluation",
        ],
        "semantic_changes": {
            "action_distribution": (
                "unbounded Normal plus post-hoc clamp replaced by one tanh-affine "
                "distribution shared by rollout, update, reload, and evaluation"
            ),
            "observation": "fixed project-owned current privileged state schema",
            "normalization": "fixed component scales without fitting or clipping",
            "reset": (
                "project-owned explicit full seed vector plus physically masked partial reset"
            ),
            "checkpoint": "full policy, optimizer, config, fingerprints, and RNG state",
            "resume": "checkpoint is reconstructable but F0 exposes no resume command",
            "evaluation": "single-environment deterministic action with unchanged physical task",
        },
        "official_baseline_clips_actions": True,
        "f0_clips_or_projects_actions": False,
        "source_commit": SOURCE_COMMIT,
    }
    payload["fingerprint"] = canonical_json_sha256(payload)
    return payload


def _micro_configuration() -> dict[str, object]:
    config = PPOConfig()
    payload: dict[str, object] = {
        "schema_version": "langmani-v2-phase2b4-f0-micro-training-configuration-v0",
        "frozen_before_learning": True,
        "ppo": config.to_dict(),
        "environment_id": "LangMani-PushToRegion-v0",
        "control_mode": "pd_joint_pos",
        "action_shape": [8],
        "episode_budget": 250,
        "sim_backend": "physx_cuda",
        "render_backend": "none",
        "task_distribution": [
            {
                "target_object_id": object_id,
                "target_region_id": direction,
                "difficulty": "standard",
            }
            for object_id in ("blue_cube", "orange_cylinder")
            for direction in ("left", "forward_right")
        ],
        "micro_evaluation_episodes": 48,
        "development_evaluation_episodes": 60,
        "reward_revision_limit": 1,
        "reward_revision_used": 0,
        "hyperparameter_sweep": False,
        "full_curriculum": False,
        "formal_qualification": False,
    }
    payload["fingerprint"] = canonical_json_sha256(payload)
    return payload


def _operation_boundary() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "langmani-v2-phase2b4-f0-operation-boundary-v0",
        "allowed": sorted(F0_ALLOWED_OPERATIONS),
        "prohibited": sorted(F0_PROHIBITED_OPERATIONS),
        "demonstrations_generated": False,
        "datasets_created": False,
        "student_training_started": False,
        "formal_qualification_started": False,
    }
    payload["fingerprint"] = canonical_json_sha256(payload)
    return payload


def main() -> int:
    output_root = parse_args().output_root.resolve()
    artifacts = {
        "ppo_implementation_audit.json": _implementation_audit(),
        "teacher_observation_manifest.json": teacher_observation_manifest(),
        "student_privilege_exclusion_audit.json": student_privilege_exclusion_audit(),
        "state_normalization_manifest.json": normalization_manifest(),
        "reward_manifest.json": reward_manifest(),
        "reward_hacking_audit.json": audit_reward_hacking_scenarios(),
        "seed_disjointness_audit.json": seed_disjointness_audit(),
        "micro_training_configuration.json": _micro_configuration(),
        "operation_boundary.json": _operation_boundary(),
    }
    if artifacts["reward_hacking_audit.json"]["passed"] is not True:
        raise RuntimeError("reward-hacking audit failed before learning")
    if artifacts["seed_disjointness_audit.json"]["passed"] is not True:
        raise RuntimeError("seed-disjointness audit failed before learning")
    if artifacts["student_privilege_exclusion_audit.json"]["passed"] is not True:
        raise RuntimeError("future student privilege-exclusion audit failed")
    for name, value in artifacts.items():
        write_json(output_root / name, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
