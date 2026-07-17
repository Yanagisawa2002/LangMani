from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from langmani.environments.specs import TaskSpec, stable_task_id
from langmani.language.router_types import (
    M5ADevelopmentGate,
    M5AExperimentManifest,
    RouterConfidence,
    RouterContractError,
    RouterDecision,
    RouterRejectionReason,
    RouterRuntimeIdentity,
    RouterStatus,
)


def _task() -> TaskSpec:
    return TaskSpec(
        target_object_id="red_cube",
        target_bin_id="left_bin",
        instruction_template_id="canonical_v0",
    )


def test_router_status_contains_exactly_the_four_contract_values() -> None:
    assert tuple(status.value for status in RouterStatus) == (
        "route",
        "reject_ambiguous",
        "reject_unsupported",
        "reject_malformed",
    )
    with pytest.raises(ValueError):
        RouterStatus("guess")


def test_stable_task_decomposition_and_route_decision_round_trip() -> None:
    task_spec = _task()
    decision = RouterDecision.route(
        task_spec=task_spec,
        confidence=RouterConfidence.probability(
            0.97,
            definition="calibrated full-route probability",
            calibrated=True,
        ),
        router_name="FactorizedTextClassifierV0",
        router_version="v0",
        evidence={"matched_fields": ["red_cube", "left_bin"]},
    )

    assert decision.target_object_id == "red_cube"
    assert decision.target_bin_id == "left_bin"
    assert decision.task_spec == task_spec
    assert decision.task_id == stable_task_id(task_spec)
    assert decision.rejection_reason is None
    assert decision.decision_fingerprint.startswith("langmani-m5a-router-decision-")
    assert RouterDecision.from_dict(decision.to_dict()) == decision
    json.dumps(decision.to_dict())


def test_rejected_decision_contains_no_executable_task_fields() -> None:
    decision = RouterDecision.reject(
        status=RouterStatus.REJECT_AMBIGUOUS,
        rejection_reason=RouterRejectionReason.CONFLICTING_OBJECTS,
        confidence=RouterConfidence.unavailable(definition="deterministic rule rejection"),
        router_name="RuleRouterV0",
        router_version="v0",
        evidence={"matched_objects": ["red_cube", "blue_cube"]},
    )

    assert decision.target_object_id is None
    assert decision.target_bin_id is None
    assert decision.task_spec is None
    assert decision.task_id is None
    assert decision.rejection_reason is RouterRejectionReason.CONFLICTING_OBJECTS
    assert RouterDecision.from_dict(decision.to_dict()) == decision
    json.dumps(decision.to_dict())


def test_rejection_cannot_smuggle_an_executable_task() -> None:
    with pytest.raises(RouterContractError, match="cannot contain executable"):
        RouterDecision(
            status=RouterStatus.REJECT_UNSUPPORTED,
            target_object_id="red_cube",
            target_bin_id="left_bin",
            task_spec=_task(),
            task_id=stable_task_id(_task()),
            confidence=RouterConfidence.unavailable(),
            router_name="bad-router",
            router_version="v0",
            rejection_reason=RouterRejectionReason.UNSUPPORTED_ACTION,
        )


def test_route_decision_requires_consistent_canonical_task() -> None:
    with pytest.raises(RouterContractError, match="target_object_id is inconsistent"):
        RouterDecision(
            status=RouterStatus.ROUTE,
            target_object_id="blue_cube",
            target_bin_id="left_bin",
            task_spec=_task(),
            task_id=stable_task_id(_task()),
            confidence=RouterConfidence.unavailable(),
            router_name="bad-router",
            router_version="v0",
        )


def test_confidence_is_explicit_and_never_fabricated() -> None:
    unavailable = RouterConfidence.unavailable()
    assert unavailable.available is False
    assert unavailable.score is None
    assert unavailable.calibrated is False

    with pytest.raises(RouterContractError, match=r"in \[0, 1\]"):
        RouterConfidence.probability(
            1.1,
            definition="invalid probability",
            calibrated=False,
        )
    with pytest.raises(FrozenInstanceError):
        unavailable.score = 0.5  # type: ignore[misc]


def test_decision_fingerprint_binds_router_evidence() -> None:
    first = RouterDecision.route(
        task_spec=_task(),
        confidence=RouterConfidence.unavailable(),
        router_name="RuleRouterV0",
        router_version="v0",
        evidence={"rule": "direct"},
    )
    second = RouterDecision.route(
        task_spec=_task(),
        confidence=RouterConfidence.unavailable(),
        router_name="RuleRouterV0",
        router_version="v0",
        evidence={"rule": "synonym"},
    )
    assert first.decision_fingerprint != second.decision_fingerprint

    tampered = first.to_dict()
    tampered["router_version"] = "v1"
    with pytest.raises(RouterContractError, match="decision_fingerprint"):
        RouterDecision.from_dict(tampered)


def _runtime_identity(router_index: int) -> RouterRuntimeIdentity:
    return RouterRuntimeIdentity(
        router_fingerprint=f"sha256:{router_index + 1:064x}",
        controller_registry_fingerprint=f"sha256:{10:064x}",
        environment_id="LangMani-PickPlaceByInstruction-v0",
        environment_contract_fingerprint=f"sha256:{11:064x}",
        rollout_config_fingerprint=f"sha256:{12:064x}",
        action_runtime_fingerprint=f"sha256:{13:064x}",
        git_commit="a" * 40,
    )


def test_runtime_and_experiment_identities_bind_all_four_schedule_locks() -> None:
    router_names = (
        "RuleRouterV0",
        "FactorizedTextClassifierV0",
        "StructuredLocalLLMRouterV0",
    )
    manifest = M5AExperimentManifest(
        corpus_fingerprint=f"sha256:{20:064x}",
        schedule_fingerprints={
            "m5a_language_dev_v0": f"sha256:{21:064x}",
            "m5a_language_final_v0": f"sha256:{22:064x}",
            "m5a_control_dev_v0": f"sha256:{23:064x}",
            "m5a_control_final_v0": f"sha256:{24:064x}",
        },
        controller_registry_fingerprint=f"sha256:{10:064x}",
        router_runtime_identities={
            router_name: _runtime_identity(index) for index, router_name in enumerate(router_names)
        },
        git_commit="a" * 40,
        dependency_versions={"python": "3.12.11"},
    )
    assert manifest.experiment_fingerprint.startswith("sha256:")
    assert manifest.language_final_accessed is False
    assert manifest.control_final_accessed is False
    assert manifest.historical_fresh_accessed is False
    json.dumps(manifest.to_dict())


def test_development_gate_is_computed_and_rule_router_cannot_authorize_final() -> None:
    values = {
        "candidate_router_fingerprint": f"sha256:{30:064x}",
        "full_task_spec_accuracy": 0.96,
        "object_accuracy": 0.98,
        "bin_accuracy": 0.98,
        "false_route_rate": 0.02,
        "ambiguous_rejection_recall": 0.91,
        "unsupported_rejection_recall": 0.96,
        "malformed_rejection_recall": 0.96,
        "schema_valid_output_rate": 0.995,
        "deterministic_repeatability": 1.0,
        "llm_malformed_output_rate": None,
        "oracle_success_rate": 0.80,
        "predicted_success_rate": 0.76,
        "wrong_object_routing_count": 1,
        "wrong_bin_routing_count": 1,
        "false_rejection_count": 2,
        "target_in_wrong_bin_count": 0,
        "target_off_table_count": 0,
        "arm_projection_count": 0,
        "invalid_action_count": 0,
        "dispatch_after_rejection_count": 0,
        "m2_expert_call_count": 0,
    }
    learned = M5ADevelopmentGate(
        candidate_router_name="FactorizedTextClassifierV0",
        candidate_is_learned=True,
        **values,  # type: ignore[arg-type]
    )
    rule = M5ADevelopmentGate(
        candidate_router_name="RuleRouterV0",
        candidate_is_learned=False,
        **values,  # type: ignore[arg-type]
    )
    assert learned.language_gate_passed is True
    assert learned.end_to_end_gate_passed is True
    assert learned.final_benchmark_authorized is True
    assert rule.final_benchmark_authorized is False
    json.dumps(learned.to_dict())


def test_development_gate_uses_absolute_oracle_success_gap() -> None:
    values = {
        "candidate_router_name": "FactorizedTextClassifierV0",
        "candidate_router_fingerprint": f"sha256:{31:064x}",
        "candidate_is_learned": True,
        "full_task_spec_accuracy": 0.96,
        "object_accuracy": 0.98,
        "bin_accuracy": 0.98,
        "false_route_rate": 0.02,
        "ambiguous_rejection_recall": 0.91,
        "unsupported_rejection_recall": 0.96,
        "malformed_rejection_recall": 0.96,
        "schema_valid_output_rate": 0.995,
        "deterministic_repeatability": 1.0,
        "llm_malformed_output_rate": None,
        "oracle_success_rate": 0.70,
        "predicted_success_rate": 0.80,
        "wrong_object_routing_count": 0,
        "wrong_bin_routing_count": 0,
        "false_rejection_count": 0,
        "target_in_wrong_bin_count": 0,
        "target_off_table_count": 0,
        "arm_projection_count": 0,
        "invalid_action_count": 0,
        "dispatch_after_rejection_count": 0,
        "m2_expert_call_count": 0,
    }

    gate = M5ADevelopmentGate(**values)  # type: ignore[arg-type]

    assert gate.end_to_end_gate_passed is False
    assert gate.final_benchmark_authorized is False
