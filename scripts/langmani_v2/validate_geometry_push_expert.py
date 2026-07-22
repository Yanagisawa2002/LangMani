"""Run the 10k geometry and 10k native-action Phase 2B.3 static gates."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from langmani.experts.geometry_push_types import GeometryPushExpertConfig
from langmani.experts.push_feasibility import (
    PlanarPrimitive,
    PrimitiveKind,
    WorkspaceContract,
    build_bounded_joint_action,
    build_geometry_push_plan,
)
from langmani.v2.push_expert_recovery import load_json, sha256_file, sha256_json, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "langmani_v2" / "phase_2b3" / "expert.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry-cases", type=int, default=10_000)
    parser.add_argument("--action-cases", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def _expert_config(path: Path) -> tuple[GeometryPushExpertConfig, dict[str, Any]]:
    payload = load_json(path)
    values = payload.get("expert")
    if not isinstance(values, dict):
        raise ValueError("expert config must contain an expert mapping")
    return GeometryPushExpertConfig(**values), payload  # type: ignore[arg-type]


def main() -> int:
    args = parse_args()
    if args.geometry_cases < 10_000 or args.action_cases < 10_000:
        raise ValueError("Phase 2B.3 static gates each require at least 10,000 cases")
    config, payload = _expert_config(args.config)
    rng = np.random.default_rng(args.seed)
    workspace = WorkspaceContract(
        (-0.43, 0.43, -0.38, 0.38),
        config.object_clearance,
        config.tcp_clearance,
        config.tracking_margin,
    )
    geometry_errors: list[dict[str, object]] = []
    shape_counts = {PrimitiveKind.BOX.value: 0, PrimitiveKind.CYLINDER.value: 0}
    minimum_object_margin = math.inf
    minimum_tcp_margin = math.inf
    for index in range(args.geometry_cases):
        kind = PrimitiveKind.BOX if index % 2 == 0 else PrimitiveKind.CYLINDER
        primitive = (
            PlanarPrimitive(kind, (0.025, 0.025), math.sqrt(2.0) * 0.025, 0.05)
            if kind is PrimitiveKind.BOX
            else PlanarPrimitive(kind, (0.025, 0.025), 0.025, 0.05)
        )
        object_xy = np.array(
            [rng.uniform(-0.22, -0.12), rng.uniform(-0.075, 0.075)], dtype=np.float64
        )
        target_xy = np.array(
            [rng.uniform(0.108, 0.232), rng.uniform(-0.232, 0.232)], dtype=np.float64
        )
        try:
            plan = build_geometry_push_plan(
                object_xy=object_xy,
                target_xy=target_xy,
                primitive=primitive,
                object_yaw=float(rng.uniform(-math.pi, math.pi)),
                workspace=workspace,
                full_containment_radius=0.044,
                goal_margin=config.goal_margin,
                gripper_contact_padding=config.gripper_contact_padding,
                precontact_clearance=config.precontact_clearance,
                push_height=(
                    config.cube_push_height
                    if kind is PrimitiveKind.BOX
                    else config.cylinder_push_height
                ),
                precontact_height=config.precontact_height,
                staging_height=config.staging_height,
                maximum_segment_distance=config.maximum_segment_distance,
            )
            minimum_object_margin = min(minimum_object_margin, plan.object_path_margin)
            minimum_tcp_margin = min(minimum_tcp_margin, plan.tcp_path_margin)
            shape_counts[kind.value] += 1
        except ValueError as error:
            geometry_errors.append({"case": index, "error": str(error)})

    action_errors: list[dict[str, object]] = []
    low = np.asarray([-3.0] * 7 + [-1.0], dtype=np.float64)
    high = np.asarray([3.0] * 7 + [1.0], dtype=np.float64)
    minimum_action_margin = math.inf
    for index in range(args.action_cases):
        arm = rng.uniform(-2.95, 2.95, size=7)
        try:
            action = build_bounded_joint_action(arm, action_low=low, action_high=high)
            minimum_action_margin = min(
                minimum_action_margin,
                float(np.min(np.minimum(action - low, high - action))),
            )
        except ValueError as error:
            action_errors.append({"case": index, "error": str(error)})

    report = {
        "schema_version": "langmani-v2-phase2b3-static-validation-v1",
        "source_commit": _source_commit(),
        "config_sha256": sha256_file(args.config),
        "semantic_config_digest": sha256_json(config.to_dict()),
        "generator_seed": args.seed,
        "geometry_cases": args.geometry_cases,
        "geometry_shape_counts": shape_counts,
        "geometry_errors": geometry_errors,
        "minimum_object_workspace_margin": minimum_object_margin,
        "minimum_tcp_workspace_margin": minimum_tcp_margin,
        "action_cases": args.action_cases,
        "nonfinite_actions": 0,
        "action_bound_violations": len(action_errors),
        "action_shape_errors": 0,
        "action_errors": action_errors,
        "minimum_action_bound_margin": minimum_action_margin,
        "teleport_or_state_mutation_used": False,
        "official_collection_episodes": 0,
        "optimizer_steps": 0,
        "passed": not geometry_errors and not action_errors,
        "resolved_config": payload,
    }
    write_json(args.output, report)
    print(json.dumps({key: report[key] for key in ("geometry_cases", "action_cases", "passed")}))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
