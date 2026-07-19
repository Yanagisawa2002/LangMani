"""Independent verification for the M5A.4.1 one-scene control smoke."""

from __future__ import annotations

from collections.abc import Mapping
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

NEURO_SYMBOLIC_CONTROL_VERIFICATION_SCHEMA = "langmani-m5a41-neuro-symbolic-control-verification-v0"


class NeuroSymbolicControlVerificationError(RuntimeError):
    """Raised when physical dispatch evidence cannot support the claimed result."""


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise NeuroSymbolicControlVerificationError(f"{label} must be one object")
    return cast(Mapping[str, object], value)


def verify_neuro_symbolic_control_evidence(
    report: Mapping[str, object],
    *,
    inputs: DevelopmentControlInputsLike,
    registry: ControllerRegistry,
    source: SelectedNeuroSymbolicDispatchSource,
) -> dict[str, object]:
    """Rehash all control atoms and recompute the six-task dispatch contract."""

    if inputs.schedule.stage is not M5AStage.ONE_SCENE_CONTROL_SMOKE:
        raise NeuroSymbolicControlVerificationError(
            "M5A.4.1 verifier accepts only one_scene_control_smoke"
        )
    if len(inputs.episodes) != 6:
        raise NeuroSymbolicControlVerificationError("one-scene schedule must contain six tasks")
    if (
        report.get("passed") is not True
        or report.get("physical_execution") is not True
        or report.get("stage") != M5AStage.ONE_SCENE_CONTROL_SMOKE.value
        or report.get("router_source_mode") != NEURO_SYMBOLIC_DISPATCH_ROUTER_LABEL
    ):
        raise NeuroSymbolicControlVerificationError("control stage did not complete physically")
    for name in (
        "language_final_accessed",
        "control_final_accessed",
        "m42_final_accessed",
        "test_split_accessed",
        "historical_fresh_accessed",
        "smolvla_go",
    ):
        if report.get(name) is not False:
            raise NeuroSymbolicControlVerificationError(
                f"control stage accessed prohibited field {name}"
            )
    evidence_root = report.get("evidence_root")
    if not isinstance(evidence_root, str):
        raise NeuroSymbolicControlVerificationError("control report omitted evidence_root")
    validated = validate_development_control_evidence(
        evidence_root,
        inputs=inputs,
        registry=registry,
    )
    if validated.identity.get("router_order") != [
        "oracle",
        NEURO_SYMBOLIC_DISPATCH_ROUTER_LABEL,
    ]:
        raise NeuroSymbolicControlVerificationError(
            "control evidence is not Oracle plus the selected neuro-symbolic router"
        )
    source_identity = validated.identity.get("neuro_symbolic_dispatch_source")
    binding_identity = validated.identity.get("neuro_symbolic_controller_binding")
    expected_binding = bind_selected_router_to_controller_registry(source, registry)
    if (
        source_identity != source.identity_dict()
        or binding_identity != expected_binding.to_dict()
        or report.get("neuro_symbolic_dispatch_source") != source.to_dict()
        or report.get("neuro_symbolic_controller_binding") != expected_binding.to_dict()
    ):
        raise NeuroSymbolicControlVerificationError(
            "selected router/controller binding differs from immutable control evidence"
        )
    summaries = validated.summaries
    oracle = _mapping(summaries.get("oracle"), label="oracle summary")
    learned = _mapping(
        summaries.get(NEURO_SYMBOLIC_DISPATCH_ROUTER_LABEL),
        label="neuro-symbolic summary",
    )
    route_checks = {
        "six_routeable_commands": learned.get("episode_count") == 6,
        "six_controller_dispatches": learned.get("dispatched_count") == 6,
        "six_correct_task_routes": learned.get("routing_correct_count") == 6,
        "zero_false_rejections": learned.get("routing_false_rejection_count") == 0,
        "zero_wrong_object_routes": learned.get("routing_wrong_object_count") == 0,
        "zero_wrong_bin_routes": learned.get("routing_wrong_bin_count") == 0,
        "zero_wrong_task_routes": learned.get("routing_wrong_task_count") == 0,
        "zero_dispatch_after_rejection": learned.get("dispatch_after_rejection_count") == 0,
        "zero_invalid_actions": learned.get("invalid_action_count") == 0,
        "zero_malformed_actions": learned.get("malformed_action_count") == 0,
        "zero_nonfinite_actions": learned.get("nonfinite_action_count") == 0,
        "zero_infrastructure_failures": learned.get("infrastructure_failure_count") == 0,
        "zero_m2_expert_calls": learned.get("m2_expert_call_count") == 0,
        "rejection_noop_probe": report.get("rejection_noop_probe_validated") is True,
        "zero_dispatch_validated": report.get("zero_dispatch_after_rejection_validated") is True,
        "oracle_six_tasks_executed": oracle.get("dispatched_count") == 6,
    }
    if not all(route_checks.values()):
        failed = sorted(name for name, passed in route_checks.items() if not passed)
        raise NeuroSymbolicControlVerificationError(
            f"six-task dispatch checks failed: {', '.join(failed)}"
        )
    learned_successes = learned.get("end_to_end_success_count")
    oracle_successes = oracle.get("end_to_end_success_count")
    if not isinstance(learned_successes, int) or not isinstance(oracle_successes, int):
        raise NeuroSymbolicControlVerificationError("control success counts are malformed")
    return {
        "schema_version": NEURO_SYMBOLIC_CONTROL_VERIFICATION_SCHEMA,
        "passed": True,
        "implementation_validated": True,
        "selected_router_evidence_validated": True,
        "read_only_dispatch_integration_validated": True,
        "controller_registry_validated": True,
        "rejection_noop_probe_validated": True,
        "six_task_routes_validated": True,
        "six_task_control_executed": True,
        "one_scene_control_smoke_completed": True,
        "stage_quality_gate_passed": report.get("stage_quality_gate_passed") is True,
        "physical_target_validated": True,
        "oracle_control_success_count": oracle_successes,
        "neuro_symbolic_control_success_count": learned_successes,
        "neuro_symbolic_control_failure_count": 6 - learned_successes,
        "route_checks": route_checks,
        "controller_registry_fingerprint": registry.registry_fingerprint,
        "dispatch_source_fingerprint": source.dispatch_source_fingerprint,
        "controller_binding_fingerprint": expected_binding.binding_fingerprint,
        "control_run_fingerprint": validated.run_fingerprint,
        "control_record_set_fingerprint": validated.record_set_fingerprint,
        "control_completion_fingerprint": validated.completion_fingerprint,
        "language_final_accessed": False,
        "control_final_accessed": False,
        "test_split_accessed": False,
        "m42_final_accessed": False,
        "smolvla_go": False,
        "final_benchmark_authorized": False,
    }


__all__ = [
    "NEURO_SYMBOLIC_CONTROL_VERIFICATION_SCHEMA",
    "NeuroSymbolicControlVerificationError",
    "verify_neuro_symbolic_control_evidence",
]
