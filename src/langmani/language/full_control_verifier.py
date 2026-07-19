"""Fresh-process verifier for M5A full paired control development."""

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
from langmani.language.full_control_development import (
    analyze_full_control_records,
    build_final_authorization,
)
from langmani.language.full_control_evidence import validate_full_control_evidence
from langmani.language.neuro_symbolic_dispatch import (
    NEURO_SYMBOLIC_DISPATCH_ROUTER_LABEL,
    SelectedNeuroSymbolicDispatchSource,
    bind_selected_router_to_controller_registry,
)
from langmani.language.stage_protocol import M5AStage

FULL_CONTROL_VERIFICATION_SCHEMA = "langmani-m5a-full-control-independent-verification-v0"


class FullControlVerificationError(RuntimeError):
    """Raised when full-development evidence cannot support its claims."""


def _object(root: Path, name: str) -> dict[str, object]:
    try:
        value = json.loads((root / name).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FullControlVerificationError(f"cannot read {name}: {error}") from error
    if not isinstance(value, dict):
        raise FullControlVerificationError(f"{name} must contain one object")
    return cast(dict[str, object], value)


def _episodes(root: Path, name: str) -> list[Mapping[str, object]]:
    payload = _object(root, name)
    values = payload.get("episodes")
    if not isinstance(values, list) or any(not isinstance(value, Mapping) for value in values):
        raise FullControlVerificationError(f"{name} has malformed episode records")
    return cast(list[Mapping[str, object]], values)


def verify_full_control_evidence(
    evidence_root: str | Path,
    *,
    inputs: DevelopmentControlInputsLike,
    registry: ControllerRegistry,
    source: SelectedNeuroSymbolicDispatchSource,
    prior_three_scene_authority: Mapping[str, object],
    sealed_final_authority: Mapping[str, object],
    expected_git_commit: str,
) -> dict[str, object]:
    """Rehash all 72 atoms and recompute the full gate and final authorization."""

    if inputs.schedule.stage is not M5AStage.FULL_CONTROL_DEVELOPMENT:
        raise FullControlVerificationError("independent verifier requires the full schedule")
    if len(inputs.episodes) != 36 or len(inputs.schedule.ordered_scene_seeds) != 6:
        raise FullControlVerificationError("full schedule must contain six scenes and 36 atoms")
    archive = validate_full_control_evidence(evidence_root)
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
        "final_authorization.json",
        "summary.md",
        "manifest.json",
        "complete.json",
    }
    if {path.name for path in root.iterdir() if path.is_file()} != expected_files:
        raise FullControlVerificationError("full compact evidence file set differs")
    owner = cast(Mapping[str, object], archive["owner"])
    if (
        owner.get("implementation_git_commit") != expected_git_commit
        or owner.get("schedule_fingerprint") != inputs.schedule.schedule_fingerprint
        or owner.get("corpus_fingerprint") != inputs.corpus_fingerprint
        or owner.get("controller_registry_fingerprint") != registry.registry_fingerprint
        or owner.get("dispatch_source_fingerprint") != source.dispatch_source_fingerprint
        or prior_three_scene_authority.get("validated") is not True
        or owner.get("prior_three_scene_completion_fingerprint")
        != prior_three_scene_authority.get("completion_fingerprint")
    ):
        raise FullControlVerificationError("full-development owner identity differs")
    binding = bind_selected_router_to_controller_registry(source, registry)
    if (
        _object(root, "router_identity.json") != source.to_dict()
        or _object(root, "controller_registry.json") != registry.to_dict()
    ):
        raise FullControlVerificationError("router or controller registry identity differs")
    runtime = _object(root, "runtime_identity.json")
    if (
        runtime.get("controller_binding") != binding.to_dict()
        or runtime.get("control_mode") != "pd_joint_pos"
        or runtime.get("execution_horizon") != 10
        or runtime.get("action_bound_mode") != "project"
        or runtime.get("policy_reset_per_episode") is not True
        or runtime.get("m2_expert_call_count") != 0
        or not isinstance(runtime.get("runtime_contract_fingerprint"), str)
    ):
        raise FullControlVerificationError("full paired runtime identity differs")
    atom_root = runtime.get("control_atom_evidence_root")
    if not isinstance(atom_root, str):
        raise FullControlVerificationError("full runtime omitted atom evidence root")
    inner = validate_development_control_evidence(
        atom_root,
        inputs=inputs,
        registry=registry,
        rejection_probe_router_name="NeuroSymbolicRouterV0",
    )
    if (
        inner.identity.get("router_order") != ["oracle", NEURO_SYMBOLIC_DISPATCH_ROUTER_LABEL]
        or inner.record_count != 72
        or inner.run_fingerprint != owner.get("control_run_fingerprint")
        or inner.completion_fingerprint != owner.get("control_completion_fingerprint")
    ):
        raise FullControlVerificationError("raw full control evidence identity differs")
    schedule_artifact = _object(root, "schedule.json")
    if any(schedule_artifact.get(key) != value for key, value in inputs.schedule.to_dict().items()):
        raise FullControlVerificationError("compact full schedule differs")
    oracle_records = _episodes(root, "oracle_episodes.json")
    learned_records = _episodes(root, "neuro_symbolic_episodes.json")
    probes = _object(root, "rejection_probes.json")
    analysis = analyze_full_control_records(
        oracle_records=oracle_records,
        neuro_symbolic_records=learned_records,
        rejection_probe_set=probes,
        examples_by_id=inputs.examples_by_id,
        router_summaries=inner.summaries,
    )
    if _object(root, "benchmark_summary.json") != analysis or owner.get(
        "analysis_fingerprint"
    ) != analysis.get("analysis_fingerprint"):
        raise FullControlVerificationError("full benchmark summary differs from raw atoms")
    paired = _object(root, "paired_results.json")
    if paired.get("pairs") != analysis.get("paired_results") or paired.get(
        "paired_outcome_counts"
    ) != analysis.get("paired_outcome_counts"):
        raise FullControlVerificationError("full paired-result sidecar differs")
    gate = _object(root, "gate_result.json")
    if (
        gate.get("gate_items") != analysis.get("gate_items")
        or gate.get("development_quality_gate_passed")
        != analysis.get("development_quality_gate_passed")
        or gate.get("final_benchmark_authorized") != analysis.get("final_benchmark_authorized")
    ):
        raise FullControlVerificationError("full quality gate differs")
    quality_passed = analysis.get("development_quality_gate_passed") is True
    final_authorization = _object(root, "final_authorization.json")
    expected_authorization = build_final_authorization(
        authorized=quality_passed,
        router_fingerprint=source.router_lock_fingerprint,
        controller_registry_fingerprint=registry.registry_fingerprint,
        runtime_fingerprint=cast(str, runtime["runtime_contract_fingerprint"]),
        full_schedule_fingerprint=inputs.schedule.schedule_fingerprint,
        development_evidence_fingerprint=cast(str, owner["run_fingerprint"]),
        selected_router_identity="NeuroSymbolicRouterV0",
        final_language_schedule_fingerprint=cast(
            str, sealed_final_authority["final_language_schedule_fingerprint"]
        ),
        final_control_schedule_fingerprint=cast(
            str, sealed_final_authority["final_control_schedule_fingerprint"]
        ),
        git_commit=expected_git_commit,
    )
    if final_authorization != expected_authorization:
        raise FullControlVerificationError("final authorization record differs")
    completion = cast(Mapping[str, object], archive["complete"])
    flags = cast(Mapping[str, object], completion["flags"])
    expected_flags = {
        "prior_one_scene_evidence_validated": True,
        "prior_three_scene_evidence_validated": True,
        "selected_router_identity_validated": True,
        "controller_registry_validated": True,
        "full_development_schedule_validated": True,
        "paired_initial_states_validated": analysis.get("paired_initial_state_count") == 36,
        "oracle_control_development_completed": True,
        "neuro_symbolic_control_development_completed": True,
        "routing_results_validated": True,
        "rejection_no_dispatch_validated": True,
        "failure_attribution_validated": True,
        "paired_comparison_validated": True,
        "action_runtime_validated": True,
        "full_control_development_completed": True,
        "development_quality_gate_passed": quality_passed,
        "selected_router_locked": quality_passed,
        "final_benchmark_authorized": quality_passed,
        "final_authorization_created": quality_passed,
        "language_final_accessed": False,
        "control_final_accessed": False,
        "test_split_accessed": False,
        "historical_fresh_accessed": False,
        "m42_final_accessed": False,
        "smolvla_go": False,
        "physical_target_validated": True,
    }
    if dict(flags) != expected_flags:
        raise FullControlVerificationError("full completion flags differ")
    return {
        "schema_version": FULL_CONTROL_VERIFICATION_SCHEMA,
        "passed": True,
        **expected_flags,
        "implementation_validated": True,
        "evidence_checksum_validated": True,
        "raw_metric_recomputation_validated": True,
        "oracle_success_count": analysis["oracle_success_count"],
        "neuro_symbolic_success_count": analysis["neuro_symbolic_success_count"],
        "paired_outcome_counts": analysis["paired_outcome_counts"],
        "routing_correct_count": analysis["routing_correct_count"],
        "gate_items": analysis["gate_items"],
        "final_authorization_fingerprint": final_authorization["authorization_fingerprint"],
        "run_fingerprint": owner["run_fingerprint"],
        "artifact_fingerprint": archive["artifact_fingerprint"],
        "completion_fingerprint": completion["completion_fingerprint"],
        "control_run_fingerprint": inner.run_fingerprint,
        "control_record_set_fingerprint": inner.record_set_fingerprint,
        "verifier_git_commit": expected_git_commit,
    }


__all__ = [
    "FULL_CONTROL_VERIFICATION_SCHEMA",
    "FullControlVerificationError",
    "verify_full_control_evidence",
]
