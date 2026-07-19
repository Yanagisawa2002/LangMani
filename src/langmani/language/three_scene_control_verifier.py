"""Fresh-process verification for the M5A three-scene paired screen."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from langmani.language.artifact_validation import (
    DevelopmentControlInputsLike,
    validate_development_control_evidence,
)
from langmani.language.controller_registry import ControllerRegistry
from langmani.language.neuro_symbolic_dispatch import (
    NEURO_SYMBOLIC_DISPATCH_ROUTER_LABEL,
    SelectedNeuroSymbolicDispatchSource,
    bind_selected_router_to_controller_registry,
)
from langmani.language.stage_protocol import M5AStage
from langmani.language.three_scene_control import analyze_three_scene_records
from langmani.language.three_scene_control_evidence import validate_three_scene_evidence

THREE_SCENE_VERIFICATION_SCHEMA = "langmani-m5a-three-scene-independent-verification-v0"


class ThreeSceneVerificationError(RuntimeError):
    """Raised when evidence cannot support the claimed physical stage."""


def _object(root: Path, name: str) -> dict[str, object]:
    try:
        value = json.loads((root / name).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ThreeSceneVerificationError(f"cannot read {name}: {error}") from error
    if not isinstance(value, dict):
        raise ThreeSceneVerificationError(f"{name} must contain one object")
    return cast(dict[str, object], value)


def _episodes(root: Path, name: str) -> list[Mapping[str, object]]:
    payload = _object(root, name)
    values = payload.get("episodes")
    if not isinstance(values, list) or any(not isinstance(value, Mapping) for value in values):
        raise ThreeSceneVerificationError(f"{name} has malformed episode records")
    return cast(list[Mapping[str, object]], values)


def verify_three_scene_control_evidence(
    evidence_root: str | Path,
    *,
    inputs: DevelopmentControlInputsLike,
    registry: ControllerRegistry,
    source: SelectedNeuroSymbolicDispatchSource,
    prior_one_scene_authority: Mapping[str, object],
    expected_git_commit: str,
) -> dict[str, object]:
    """Recompute raw atoms, pairing, action metrics, and all quality predicates."""

    if inputs.schedule.stage is not M5AStage.THREE_SCENE_CONTROL_SCREEN:
        raise ThreeSceneVerificationError("independent verifier requires the three-scene schedule")
    if len(inputs.episodes) != 18 or len(inputs.schedule.ordered_scene_seeds) != 3:
        raise ThreeSceneVerificationError("three-scene schedule must contain 18 atoms")
    archive = validate_three_scene_evidence(evidence_root)
    root = Path(cast(str, archive["root"]))
    expected_files = {
        "owner.json",
        "input_contract.json",
        "schedule.json",
        "router_identity.json",
        "controller_registry.json",
        "runtime_identity.json",
        "oracle_episodes.json",
        "neuro_symbolic_episodes.json",
        "paired_results.json",
        "rejection_probes.json",
        "failure_attribution.json",
        "benchmark_summary.json",
        "gate_result.json",
        "summary.md",
        "manifest.json",
        "complete.json",
    }
    observed_files = {path.name for path in root.iterdir() if path.is_file()}
    if observed_files != expected_files:
        raise ThreeSceneVerificationError("three-scene compact evidence file set differs")
    owner = cast(Mapping[str, object], archive["owner"])
    if (
        owner.get("implementation_git_commit") != expected_git_commit
        or owner.get("schedule_fingerprint") != inputs.schedule.schedule_fingerprint
        or owner.get("corpus_fingerprint") != inputs.corpus_fingerprint
        or owner.get("controller_registry_fingerprint") != registry.registry_fingerprint
        or owner.get("dispatch_source_fingerprint") != source.dispatch_source_fingerprint
        or prior_one_scene_authority.get("validated") is not True
        or owner.get("prior_one_scene_completion_fingerprint")
        != prior_one_scene_authority.get("completion_fingerprint")
    ):
        raise ThreeSceneVerificationError("three-scene owner identity differs")
    binding = bind_selected_router_to_controller_registry(source, registry)
    if (
        _object(root, "router_identity.json") != source.to_dict()
        or _object(root, "controller_registry.json") != registry.to_dict()
    ):
        raise ThreeSceneVerificationError("router or controller registry identity differs")
    runtime = _object(root, "runtime_identity.json")
    if (
        runtime.get("controller_binding") != binding.to_dict()
        or runtime.get("control_mode") != "pd_joint_pos"
        or runtime.get("execution_horizon") != 10
        or runtime.get("action_bound_mode") != "project"
        or runtime.get("m2_expert_call_count") != 0
    ):
        raise ThreeSceneVerificationError("paired runtime identity differs")
    atom_root = runtime.get("control_atom_evidence_root")
    if not isinstance(atom_root, str):
        raise ThreeSceneVerificationError("paired runtime omitted atom evidence root")
    inner = validate_development_control_evidence(
        atom_root,
        inputs=inputs,
        registry=registry,
        rejection_probe_router_name="NeuroSymbolicRouterV0",
    )
    if (
        inner.identity.get("router_order") != ["oracle", NEURO_SYMBOLIC_DISPATCH_ROUTER_LABEL]
        or inner.record_count != 36
        or inner.run_fingerprint != owner.get("control_run_fingerprint")
        or inner.completion_fingerprint != owner.get("control_completion_fingerprint")
    ):
        raise ThreeSceneVerificationError("raw paired control evidence identity differs")
    schedule_artifact = _object(root, "schedule.json")
    if any(schedule_artifact.get(key) != value for key, value in inputs.schedule.to_dict().items()):
        raise ThreeSceneVerificationError("compact three-scene schedule differs")
    oracle_records = _episodes(root, "oracle_episodes.json")
    learned_records = _episodes(root, "neuro_symbolic_episodes.json")
    probes = _object(root, "rejection_probes.json")
    analysis = analyze_three_scene_records(
        oracle_records=oracle_records,
        neuro_symbolic_records=learned_records,
        rejection_probe_set=probes,
    )
    if _object(root, "benchmark_summary.json") != analysis or owner.get(
        "analysis_fingerprint"
    ) != analysis.get("analysis_fingerprint"):
        raise ThreeSceneVerificationError("paired benchmark summary differs from raw atoms")
    pair_artifact = _object(root, "paired_results.json")
    if pair_artifact.get("pairs") != analysis.get("paired_results") or pair_artifact.get(
        "paired_outcome_counts"
    ) != analysis.get("paired_outcome_counts"):
        raise ThreeSceneVerificationError("paired-result sidecar differs")
    gate = _object(root, "gate_result.json")
    if gate.get("gate_items") != analysis.get("gate_items") or gate.get(
        "three_scene_control_screen_passed"
    ) != analysis.get("three_scene_control_screen_passed"):
        raise ThreeSceneVerificationError("three-scene quality gate differs")
    completion = cast(Mapping[str, object], archive["complete"])
    flags = cast(Mapping[str, object], completion["flags"])
    quality_passed = analysis.get("three_scene_control_screen_passed") is True
    expected_flags = {
        "prior_one_scene_evidence_validated": True,
        "neuro_symbolic_router_identity_validated": True,
        "controller_registry_validated": True,
        "three_scene_schedule_validated": True,
        "paired_initial_states_validated": analysis.get("paired_initial_state_count") == 18,
        "oracle_control_executed": True,
        "neuro_symbolic_control_executed": True,
        "six_tasks_per_scene_validated": True,
        "routing_results_validated": True,
        "rejection_no_dispatch_validated": True,
        "failure_attribution_validated": True,
        "paired_comparison_validated": True,
        "action_runtime_validated": True,
        "three_scene_control_screen_completed": True,
        "three_scene_control_screen_passed": quality_passed,
        "selected_router_locked": quality_passed,
        "full_control_development_authorized": quality_passed,
        "full_control_development_completed": False,
        "language_final_accessed": False,
        "control_final_accessed": False,
        "test_split_accessed": False,
        "historical_fresh_accessed": False,
        "m42_final_accessed": False,
        "smolvla_go": False,
        "physical_target_validated": True,
    }
    if dict(flags) != expected_flags:
        raise ThreeSceneVerificationError("three-scene completion flags differ")
    return {
        "schema_version": THREE_SCENE_VERIFICATION_SCHEMA,
        "passed": True,
        **expected_flags,
        "implementation_validated": True,
        "evidence_checksum_validated": True,
        "raw_metric_recomputation_validated": True,
        "oracle_success_count": analysis["oracle_success_count"],
        "neuro_symbolic_success_count": analysis["neuro_symbolic_success_count"],
        "paired_outcome_counts": analysis["paired_outcome_counts"],
        "gate_items": analysis["gate_items"],
        "run_fingerprint": owner["run_fingerprint"],
        "artifact_fingerprint": archive["artifact_fingerprint"],
        "completion_fingerprint": completion["completion_fingerprint"],
        "control_run_fingerprint": inner.run_fingerprint,
        "control_record_set_fingerprint": inner.record_set_fingerprint,
        "verifier_git_commit": expected_git_commit,
    }


__all__ = [
    "THREE_SCENE_VERIFICATION_SCHEMA",
    "ThreeSceneVerificationError",
    "verify_three_scene_control_evidence",
]
