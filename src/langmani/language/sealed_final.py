"""Pure contracts for the separately authorized M5A sealed final benchmark."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from langmani.datasets.identity import sha256_hex
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import stable_task_id
from langmani.language.failure_attribution import FailureAttribution
from langmani.language.full_control_development import (
    FINAL_AUTHORIZATION_SCHEMA,
    final_evaluation_protocol_fingerprint,
)
from langmani.language.neuro_symbolic_safety import (
    M5A41GateResult,
    RejectionTaxonomyEvaluation,
    SafetyRoutingEvaluation,
    evaluate_development_safety_gate,
)
from langmani.language.router_types import LanguageExample
from langmani.language.three_scene_control import (
    RejectionProbeCase,
    initial_scene_state_fingerprint,
    normalize_initial_scene_state,
)

SEALED_FINAL_RUN_IDENTITY_SCHEMA = "langmani-m5a-final-run-identity-v0"
SEALED_FINAL_LANGUAGE_GATE_SCHEMA = "langmani-m5a-final-language-quality-gate-v0"
SEALED_FINAL_CONTROL_ANALYSIS_SCHEMA = "langmani-m5a-final-control-analysis-v0"
SEALED_FINAL_CONTROL_GATE_SCHEMA = "langmani-m5a-final-control-quality-gate-v0"
SEALED_FINAL_RESULT_SCHEMA = "langmani-m5a-final-result-v0"
SEALED_FINAL_EVIDENCE_SCHEMA = "langmani-m5a-sealed-final-evidence-v0"
SEALED_FINAL_VERIFICATION_SCHEMA = "langmani-m5a-sealed-final-verification-v0"


class SealedFinalContractError(ValueError):
    """Raised when final inputs, metrics, or paired evidence differ from the lock."""


def _fingerprint(value: object) -> str:
    return f"sha256:{sha256_hex(value)}"


def _require_fingerprint(value: str, *, label: str) -> None:
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise SealedFinalContractError(f"{label} must be one SHA-256 fingerprint")


@dataclass(frozen=True, slots=True)
class M5AFinalRunIdentityV0:
    """Path- and time-independent identity for the one valid final attempt."""

    implementation_git: str
    final_authorization_fingerprint: str
    router_lock_fingerprint: str
    controller_registry_fingerprint: str
    runtime_fingerprint: str
    language_schedule_fingerprint: str
    control_schedule_fingerprint: str
    corpus_fingerprint: str
    split_fingerprints: Mapping[str, str]
    model_tokenizer_identity_fingerprint: str
    prompt_fingerprint: str
    semantic_schema_fingerprint: str
    symbolic_parser_fingerprint: str
    safety_arbiter_fingerprint: str
    metric_contract_fingerprint: str
    evidence_schema_fingerprint: str

    def __post_init__(self) -> None:
        if len(self.implementation_git) != 40 or any(
            character not in "0123456789abcdef" for character in self.implementation_git
        ):
            raise SealedFinalContractError("implementation_git must be one full Git SHA")
        for name in (
            "final_authorization_fingerprint",
            "router_lock_fingerprint",
            "controller_registry_fingerprint",
            "runtime_fingerprint",
            "language_schedule_fingerprint",
            "control_schedule_fingerprint",
            "corpus_fingerprint",
            "model_tokenizer_identity_fingerprint",
            "prompt_fingerprint",
            "semantic_schema_fingerprint",
            "symbolic_parser_fingerprint",
            "safety_arbiter_fingerprint",
            "metric_contract_fingerprint",
            "evidence_schema_fingerprint",
        ):
            _require_fingerprint(cast(str, getattr(self, name)), label=name)
        if not self.split_fingerprints or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.split_fingerprints.items()
        ):
            raise SealedFinalContractError("split fingerprints must be one non-empty mapping")
        for key, value in self.split_fingerprints.items():
            _require_fingerprint(value, label=f"split_fingerprints[{key}]")

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": SEALED_FINAL_RUN_IDENTITY_SCHEMA,
            "implementation_git": self.implementation_git,
            "final_authorization_fingerprint": self.final_authorization_fingerprint,
            "router_lock_fingerprint": self.router_lock_fingerprint,
            "controller_registry_fingerprint": self.controller_registry_fingerprint,
            "runtime_fingerprint": self.runtime_fingerprint,
            "language_schedule_fingerprint": self.language_schedule_fingerprint,
            "control_schedule_fingerprint": self.control_schedule_fingerprint,
            "corpus_fingerprint": self.corpus_fingerprint,
            "split_fingerprints": dict(sorted(self.split_fingerprints.items())),
            "model_tokenizer_identity_fingerprint": self.model_tokenizer_identity_fingerprint,
            "prompt_fingerprint": self.prompt_fingerprint,
            "semantic_schema_fingerprint": self.semantic_schema_fingerprint,
            "symbolic_parser_fingerprint": self.symbolic_parser_fingerprint,
            "safety_arbiter_fingerprint": self.safety_arbiter_fingerprint,
            "metric_contract_fingerprint": self.metric_contract_fingerprint,
            "evidence_schema_fingerprint": self.evidence_schema_fingerprint,
        }

    @property
    def run_fingerprint(self) -> str:
        return _fingerprint(self.semantic_payload())

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "run_fingerprint": self.run_fingerprint}


def validate_final_authorization(
    authorization: Mapping[str, object],
    *,
    expected_fingerprint: str,
    router_fingerprint: str,
    controller_registry_fingerprint: str,
    runtime_fingerprint: str,
    language_schedule_fingerprint: str,
    control_schedule_fingerprint: str,
) -> dict[str, object]:
    """Validate the positive development-derived authorization before final access."""

    payload = dict(authorization)
    declared = payload.pop("authorization_fingerprint", None)
    if (
        authorization.get("schema_version") != FINAL_AUTHORIZATION_SCHEMA
        or authorization.get("authorized") is not True
        or declared != _fingerprint(payload)
        or declared != expected_fingerprint
        or authorization.get("locked_router_fingerprint") != router_fingerprint
        or authorization.get("controller_registry_fingerprint") != controller_registry_fingerprint
        or authorization.get("runtime_fingerprint") != runtime_fingerprint
        or authorization.get("final_language_schedule_lock_fingerprint")
        != language_schedule_fingerprint
        or authorization.get("final_control_schedule_lock_fingerprint")
        != control_schedule_fingerprint
        or authorization.get("final_evaluation_protocol_fingerprint")
        != final_evaluation_protocol_fingerprint()
        or authorization.get("selected_router_identity") != "NeuroSymbolicRouterV0"
        or authorization.get("final_data_accessed") is not False
        or authorization.get("final_command_texts_materialized") is not False
        or authorization.get("final_scene_seeds_materialized") is not False
        or authorization.get("automatic_final_execution") is not False
    ):
        raise SealedFinalContractError("M5AFinalAuthorizationV0 identity or access state differs")
    return dict(authorization)


def final_metric_contract() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "langmani-m5a-final-metric-contract-v0",
        "language": {
            "full_task_spec_accuracy_min": 0.95,
            "object_accuracy_min": 0.97,
            "bin_accuracy_min": 0.97,
            "every_task_accuracy_min": 0.90,
            "unsafe_false_route_rate_max": 0.03,
            "safe_rejection_recall_min": 0.97,
            "schema_valid_rate": 1.0,
            "deterministic_repeatability": 1.0,
            "family_unsafe_false_route_rate_max": 0.10,
        },
        "control": {
            "correct_routes_min": 69,
            "pair_count": 72,
            "wrong_object_routes_max": 3,
            "wrong_bin_routes_max": 3,
            "false_rejections_max": 2,
            "success_gap_max": 4,
            "initial_state_pairs": 72,
        },
        "taxonomy_is_diagnostic_only": True,
    }
    return {**payload, "contract_fingerprint": _fingerprint(payload)}


def evaluate_final_language_quality(
    *,
    metrics: Mapping[str, object],
    safety: SafetyRoutingEvaluation,
    taxonomy: RejectionTaxonomyEvaluation,
    special_route_counts: Mapping[str, int],
    rejection_family_unsafe_false_route_rates: Mapping[str, float],
    no_prohibited_source_access: bool,
) -> dict[str, object]:
    """Apply the frozen final safety gate and keep taxonomy separate."""

    safety_gate: M5A41GateResult = evaluate_development_safety_gate(
        metrics=metrics,
        safety=safety,
        special_route_counts=special_route_counts,
        rejection_family_unsafe_false_route_rates=rejection_family_unsafe_false_route_rates,
        no_prohibited_source_access=no_prohibited_source_access,
    )
    base: dict[str, object] = {
        "schema_version": SEALED_FINAL_LANGUAGE_GATE_SCHEMA,
        "safety_gate_items": dict(safety_gate.checks),
        "final_language_safety_quality_passed": safety_gate.passed,
        "taxonomy_metrics": taxonomy.to_dict(),
        "final_rejection_taxonomy_quality_passed": taxonomy.exact_quality_passed,
        "taxonomy_diagnostic_only_for_physical_dispatch": True,
    }
    return {**base, "gate_fingerprint": _fingerprint(base)}


FINAL_REJECTION_CASES: tuple[RejectionProbeCase, ...] = (
    RejectionProbeCase(
        "unsupported_action", "unsupported_action", "Push the red cube toward the left bin."
    ),
    RejectionProbeCase(
        "unsupported_object", "unsupported_object", "Put the yellow cube in the left bin."
    ),
    RejectionProbeCase(
        "unsupported_destination", "unsupported_destination", "Move the red cube to the shelf."
    ),
    RejectionProbeCase(
        "conflicting_objects",
        "conflicting_objects",
        "Pick up the red cube and the blue cube and place them in the left bin.",
    ),
    RejectionProbeCase(
        "conflicting_bins",
        "conflicting_bins",
        "Pick up the red cube and place it in both the left bin and the right bin.",
    ),
    RejectionProbeCase(
        "multiple_sequential_tasks",
        "multiple_sequential_tasks",
        "Move red to right and then blue to left.",
    ),
    RejectionProbeCase(
        "unresolved_correction",
        "unresolved_correction",
        "Actually never mind, move the red cube left.",
    ),
    RejectionProbeCase(
        "contradictory_negation",
        "contradictory_negation",
        "Do not move the red cube to the left bin.",
    ),
    RejectionProbeCase("empty_noise", "empty_or_noise", "..."),
    RejectionProbeCase(
        "unsupported_spatial_reference",
        "unsupported_spatial_reference",
        "Move the red cube over there.",
    ),
)


def build_final_rejection_probe_set(
    items: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(items) != len(FINAL_REJECTION_CASES):
        raise SealedFinalContractError("final rejection set must contain exactly ten probes")
    validated: list[dict[str, object]] = []
    for expected, value in zip(FINAL_REJECTION_CASES, items, strict=True):
        item = dict(value)
        if any(
            item.get(key) != expected_value for key, expected_value in expected.to_dict().items()
        ):
            raise SealedFinalContractError("final rejection probe identity or order differs")
        probe = item.get("probe")
        if not isinstance(probe, Mapping):
            raise SealedFinalContractError("final rejection probe lacks dispatch evidence")
        result = probe.get("result")
        if not isinstance(result, Mapping):
            raise SealedFinalContractError("final rejection probe result is malformed")
        decision = result.get("decision")
        dispatch = result.get("dispatch")
        if not isinstance(decision, Mapping) or not isinstance(dispatch, Mapping):
            raise SealedFinalContractError("final rejection decision/dispatch is malformed")
        if not (
            probe.get("passed") is True
            and probe.get("controller_lookup_count") == 0
            and probe.get("environment_reset_count") == 0
            and probe.get("environment_step_count") == 0
            and item.get("policy_reset_count") == 0
            and decision.get("status") != "route"
            and decision.get("task_spec") is None
            and decision.get("task_id") is None
            and decision.get("target_object_id") is None
            and decision.get("target_bin_id") is None
            and dispatch.get("dispatched") is False
            and dispatch.get("controller_loaded") is False
            and dispatch.get("policy_called") is False
            and dispatch.get("environment_reset_called") is False
            and dispatch.get("environment_step_count") == 0
            and dispatch.get("safe_rejection") is True
        ):
            raise SealedFinalContractError("a final rejection probe entered the runtime")
        validated.append(item)
    base: dict[str, object] = {
        "schema_version": "langmani-m5a-final-rejection-probes-v0",
        "probe_count": len(validated),
        "cases": validated,
        "all_safe_rejections": True,
        "controller_lookup_count": 0,
        "policy_reset_count": 0,
        "environment_reset_count": 0,
        "environment_step_count": 0,
    }
    return {**base, "probe_set_fingerprint": _fingerprint(base)}


def _object(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SealedFinalContractError(f"{label} must be one object")
    return cast(Mapping[str, object], value)


def _indexed(
    records: Sequence[Mapping[str, object]], *, label: str
) -> dict[int, Mapping[str, object]]:
    result: dict[int, Mapping[str, object]] = {}
    for record in records:
        identity = _object(record.get("identity"), label=f"{label} identity")
        index = identity.get("episode_index")
        if isinstance(index, bool) or not isinstance(index, int) or index in result:
            raise SealedFinalContractError(f"{label} indices are malformed")
        result[index] = record
    if set(result) != set(range(72)):
        raise SealedFinalContractError(f"{label} must contain exactly indices 0..71")
    return result


def analyze_final_control_records(
    *,
    oracle_records: Sequence[Mapping[str, object]],
    neuro_symbolic_records: Sequence[Mapping[str, object]],
    rejection_probe_set: Mapping[str, object],
    examples_by_id: Mapping[str, LanguageExample],
    router_summaries: Mapping[str, object],
) -> dict[str, object]:
    """Recompute the 72-pair final control gate from compact and raw atoms."""

    probes = build_final_rejection_probe_set(
        cast(Sequence[Mapping[str, object]], rejection_probe_set.get("cases", ()))
    )
    if probes != dict(rejection_probe_set):
        raise SealedFinalContractError("final rejection probe fingerprint differs")
    oracle = _indexed(oracle_records, label="Oracle final evidence")
    learned = _indexed(neuro_symbolic_records, label="NeuroSymbolic final evidence")
    oracle_summary = _object(router_summaries.get("oracle"), label="Oracle summary")
    learned_summary = _object(router_summaries.get("neuro_symbolic"), label="Neuro summary")
    expected_tasks = tuple(stable_task_id(value) for value in CANONICAL_TASK_SPECS)
    scene_tasks: defaultdict[str, list[str]] = defaultdict(list)
    pairs: list[dict[str, object]] = []
    outcomes: Counter[str] = Counter()
    attributions: Counter[str] = Counter()
    per_task: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {
            "episodes": 0,
            "routing_correct": 0,
            "oracle_success": 0,
            "neuro_symbolic_success": 0,
        }
    )
    per_family: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"episodes": 0, "routing_correct": 0}
    )
    routing_correct = object_correct = bin_correct = 0
    wrong_object = wrong_bin = false_rejection = malformed_executable = 0
    paired_states = controller_agreement = 0
    target_grasp_oracle = target_grasp_learned = 0

    for index in range(72):
        oracle_record, learned_record = oracle[index], learned[index]
        oracle_identity = _object(oracle_record.get("identity"), label="Oracle identity")
        learned_identity = _object(learned_record.get("identity"), label="learned identity")
        shared = (
            "schedule_id",
            "schedule_fingerprint",
            "episode_index",
            "scene_seed",
            "scene_id",
            "oracle_task_id",
            "language_example_id",
            "command_fingerprint",
        )
        if any(oracle_identity.get(key) != learned_identity.get(key) for key in shared):
            raise SealedFinalContractError("paired final identities differ")
        task_id = cast(str, oracle_identity["oracle_task_id"])
        scene_id = cast(str, oracle_identity["scene_id"])
        example_id = cast(str, oracle_identity["language_example_id"])
        example = examples_by_id.get(example_id)
        if example is None or example.task_id != task_id:
            raise SealedFinalContractError("final command slot differs from scheduled TaskSpec")
        scene_tasks[scene_id].append(task_id)
        oracle_result = _object(oracle_record.get("result"), label="Oracle result")
        learned_result = _object(learned_record.get("result"), label="learned result")
        decision = _object(learned_result.get("decision"), label="learned decision")
        oracle_task = _object(learned_result.get("oracle_task_spec"), label="oracle task")
        oracle_dispatch = _object(oracle_result.get("dispatch"), label="Oracle dispatch")
        learned_dispatch = _object(learned_result.get("dispatch"), label="learned dispatch")
        routed = decision.get("status") == "route"
        object_match = routed and decision.get("target_object_id") == oracle_task.get(
            "target_object_id"
        )
        bin_match = routed and decision.get("target_bin_id") == oracle_task.get("target_bin_id")
        correct = routed and object_match and bin_match and decision.get("task_id") == task_id
        routing_correct += int(correct)
        object_correct += int(object_match)
        bin_correct += int(bin_match)
        wrong_object += int(routed and not object_match)
        wrong_bin += int(routed and not bin_match)
        false_rejection += int(not routed)
        malformed_executable += int(
            routed
            and (
                decision.get("task_id") is None
                or decision.get("task_spec") is None
                or learned_dispatch.get("controller_task_id") is None
            )
        )
        same_controller = (
            correct
            and oracle_dispatch.get("controller_task_id") == task_id
            and learned_dispatch.get("controller_task_id") == task_id
            and oracle_dispatch.get("controller_checkpoint_fingerprint")
            == learned_dispatch.get("controller_checkpoint_fingerprint")
        )
        controller_agreement += int(same_controller)
        oracle_audit = _object(oracle_record.get("paired_execution_audit"), label="Oracle audit")
        learned_audit_value = learned_record.get("paired_execution_audit")
        learned_audit = (
            None
            if learned_audit_value is None
            else _object(learned_audit_value, label="learned audit")
        )
        if routed and learned_audit is None:
            raise SealedFinalContractError("a routed final episode lacks an execution audit")
        rejected_without_runtime = (
            learned_audit is None
            and learned_record.get("active_episode_spec") is None
            and learned_result.get("control") is None
            and learned_result.get("failure_attribution")
            == FailureAttribution.ROUTING_FALSE_REJECTION.value
            and decision.get("task_spec") is None
            and decision.get("task_id") is None
            and decision.get("target_object_id") is None
            and decision.get("target_bin_id") is None
            and learned_dispatch.get("dispatched") is False
            and learned_dispatch.get("controller_loaded") is False
            and learned_dispatch.get("policy_called") is False
            and learned_dispatch.get("environment_reset_called") is False
            and learned_dispatch.get("environment_step_count") == 0
        )
        if not routed and not rejected_without_runtime:
            raise SealedFinalContractError("a rejected final episode entered the runtime")
        oracle_state = normalize_initial_scene_state(oracle_audit.get("initial_physical_state"))
        learned_state = (
            None
            if learned_audit is None
            else normalize_initial_scene_state(learned_audit.get("initial_physical_state"))
        )
        state_equal = (
            learned_audit is not None
            and learned_state is not None
            and oracle_state == learned_state
            and oracle_audit.get("initial_physical_state_fingerprint")
            == initial_scene_state_fingerprint(oracle_state)
            and learned_audit.get("initial_physical_state_fingerprint")
            == initial_scene_state_fingerprint(learned_state)
        )
        paired_states += int(state_equal)
        oracle_success = oracle_result.get("end_to_end_success") is True
        learned_success = learned_result.get("end_to_end_success") is True
        outcome = (
            "both_succeed"
            if oracle_success and learned_success
            else "both_fail"
            if not oracle_success and not learned_success
            else "oracle_succeeds_neuro_symbolic_fails"
            if oracle_success
            else "oracle_fails_neuro_symbolic_succeeds"
        )
        outcomes[outcome] += 1
        attribution = learned_result.get("failure_attribution")
        if attribution not in {value.value for value in FailureAttribution}:
            raise SealedFinalContractError("final episode lacks one valid failure attribution")
        attributions[cast(str, attribution)] += 1
        per_task[task_id]["episodes"] += 1
        per_task[task_id]["routing_correct"] += int(correct)
        per_task[task_id]["oracle_success"] += int(oracle_success)
        per_task[task_id]["neuro_symbolic_success"] += int(learned_success)
        per_family[example.template_family_id]["episodes"] += 1
        per_family[example.template_family_id]["routing_correct"] += int(correct)
        audits = [("oracle", oracle_audit)]
        if learned_audit is not None:
            audits.append(("neuro_symbolic", learned_audit))
        for label, audit in audits:
            rollout = _object(audit.get("rollout"), label=f"{label} rollout")
            if rollout.get("action_bound_mode") != "project":
                raise SealedFinalContractError("final rollout changed the project action runtime")
            grasped = rollout.get("target_grasped_any") is True
            target_grasp_oracle += int(label == "oracle" and grasped)
            target_grasp_learned += int(label == "neuro_symbolic" and grasped)
        pairs.append(
            {
                "episode_index": index,
                "scene_seed": oracle_identity["scene_seed"],
                "scene_id": scene_id,
                "task_id": task_id,
                "language_example_id": example_id,
                "template_family_id": example.template_family_id,
                "oracle_success": oracle_success,
                "neuro_symbolic_success": learned_success,
                "routing_correct": correct,
                "same_controller": same_controller,
                "initial_physical_state_equal": state_equal,
                "paired_outcome": outcome,
                "failure_attribution": attribution,
            }
        )

    if len(scene_tasks) != 12 or any(
        tuple(values) != expected_tasks for values in scene_tasks.values()
    ):
        raise SealedFinalContractError("final evidence lacks 12 canonical six-task scenes")
    oracle_success = int(oracle_summary.get("end_to_end_success_count", -1))
    learned_success = int(learned_summary.get("end_to_end_success_count", -1))
    success_gap = abs(oracle_success - learned_success)
    wrong_bin_interaction = int(oracle_summary.get("target_in_wrong_bin_count", 0)) + int(
        learned_summary.get("target_in_wrong_bin_count", 0)
    )
    off_table = int(oracle_summary.get("target_off_table_count", 0)) + int(
        learned_summary.get("target_off_table_count", 0)
    )
    arm_projection = int(oracle_summary.get("arm_projected_component_count", 0)) + int(
        learned_summary.get("arm_projected_component_count", 0)
    )
    malformed_actions = int(oracle_summary.get("malformed_action_count", 0)) + int(
        learned_summary.get("malformed_action_count", 0)
    )
    nonfinite = int(oracle_summary.get("nonfinite_action_count", 0)) + int(
        learned_summary.get("nonfinite_action_count", 0)
    )
    infrastructure = int(oracle_summary.get("infrastructure_failure_count", 0)) + int(
        learned_summary.get("infrastructure_failure_count", 0)
    )
    m2_calls = int(oracle_summary.get("m2_expert_call_count", 0)) + int(
        learned_summary.get("m2_expert_call_count", 0)
    )
    gate_items = {
        "correct_full_task_routes_at_least_69_of_72": routing_correct >= 69,
        "routing_accuracy_at_least_0_95": routing_correct / 72 >= 0.95,
        "wrong_object_routes_at_most_3_of_72": wrong_object <= 3,
        "wrong_bin_routes_at_most_3_of_72": wrong_bin <= 3,
        "false_rejections_at_most_2_of_72": false_rejection <= 2,
        "malformed_executable_decisions_zero": malformed_executable == 0,
        "neuro_symbolic_success_gap_to_oracle_at_most_4_of_72": success_gap <= 4,
        "controller_identity_agreement_for_correct_routes_complete": (
            controller_agreement == routing_correct
        ),
        "initial_state_pairing_72_of_72": paired_states == 72,
        "unsafe_rejection_probe_routes_zero": probes["all_safe_rejections"] is True,
        "lookup_after_rejection_zero": probes["controller_lookup_count"] == 0,
        "policy_reset_after_rejection_zero": probes["policy_reset_count"] == 0,
        "environment_reset_after_rejection_zero": probes["environment_reset_count"] == 0,
        "env_step_after_rejection_zero": probes["environment_step_count"] == 0,
        "target_in_wrong_bin_zero": wrong_bin_interaction == 0,
        "target_off_table_zero": off_table == 0,
        "arm_projection_zero": arm_projection == 0,
        "nan_zero": nonfinite == 0,
        "inf_zero": nonfinite == 0,
        "malformed_action_zero": malformed_actions == 0,
        "m2_calls_zero": m2_calls == 0,
        "infrastructure_failures_zero": infrastructure == 0,
    }
    quality_passed = all(gate_items.values())

    def rates(values: Mapping[str, dict[str, int]]) -> dict[str, object]:
        return {
            key: {
                **counts,
                "routing_accuracy": counts["routing_correct"] / counts["episodes"],
                "oracle_success_rate": counts.get("oracle_success", 0) / counts["episodes"],
                "neuro_symbolic_success_rate": counts.get("neuro_symbolic_success", 0)
                / counts["episodes"],
            }
            for key, counts in sorted(values.items())
        }

    base: dict[str, object] = {
        "schema_version": SEALED_FINAL_CONTROL_ANALYSIS_SCHEMA,
        "scene_count": 12,
        "pair_count": 72,
        "oracle_episode_count": 72,
        "neuro_symbolic_episode_count": 72,
        "six_tasks_per_scene_validated": True,
        "paired_results": pairs,
        "paired_outcome_counts": dict(sorted(outcomes.items())),
        "paired_disagreement_episode_indices": [
            cast(int, pair["episode_index"])
            for pair in pairs
            if pair["paired_outcome"]
            in {
                "oracle_succeeds_neuro_symbolic_fails",
                "oracle_fails_neuro_symbolic_succeeds",
            }
        ],
        "oracle_success_count": oracle_success,
        "neuro_symbolic_success_count": learned_success,
        "paired_success_difference": learned_success - oracle_success,
        "absolute_success_gap": success_gap,
        "routing_correct_count": routing_correct,
        "full_task_spec_accuracy": routing_correct / 72,
        "object_accuracy": object_correct / 72,
        "bin_accuracy": bin_correct / 72,
        "wrong_object_route_count": wrong_object,
        "wrong_bin_route_count": wrong_bin,
        "false_rejection_count": false_rejection,
        "malformed_executable_decision_count": malformed_executable,
        "per_task_results": rates(per_task),
        "per_command_family_routing": rates(per_family),
        "controller_identity_agreement_count": controller_agreement,
        "paired_initial_state_count": paired_states,
        "failure_attribution_counts": dict(sorted(attributions.items())),
        "oracle_target_grasp_count": target_grasp_oracle,
        "neuro_symbolic_target_grasp_count": target_grasp_learned,
        "oracle_post_grasp_success_rate": (
            oracle_success / target_grasp_oracle if target_grasp_oracle else None
        ),
        "neuro_symbolic_post_grasp_success_rate": (
            learned_success / target_grasp_learned if target_grasp_learned else None
        ),
        "router_summaries": {
            "oracle": dict(oracle_summary),
            "neuro_symbolic": dict(learned_summary),
        },
        "rejection_probe_summary": probes,
        "gate_items": gate_items,
        "final_control_quality_passed": quality_passed,
    }
    return {**base, "analysis_fingerprint": _fingerprint(base)}


def build_final_result(
    *,
    language_gate: Mapping[str, object],
    control_analysis: Mapping[str, object],
    independent_evidence_valid: bool,
) -> dict[str, object]:
    language_safety = language_gate.get("final_language_safety_quality_passed") is True
    taxonomy = language_gate.get("final_rejection_taxonomy_quality_passed") is True
    control = control_analysis.get("final_control_quality_passed") is True
    pipeline = independent_evidence_valid
    base: dict[str, object] = {
        "schema_version": SEALED_FINAL_RESULT_SCHEMA,
        "final_language_safety_quality_passed": language_safety,
        "final_rejection_taxonomy_quality_passed": taxonomy,
        "final_control_quality_passed": control,
        "final_pipeline_validated": pipeline,
        "physical_target_validated": pipeline,
        "final_quality_gate_passed": language_safety and control,
        "test_split_accessed": False,
        "historical_fresh_accessed": False,
        "m42_final_accessed": False,
        "smolvla_go": False,
    }
    return {**base, "result_fingerprint": _fingerprint(base)}


def finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SealedFinalContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SealedFinalContractError(f"{label} must be finite")
    return result


__all__ = [
    "FINAL_REJECTION_CASES",
    "M5AFinalRunIdentityV0",
    "SEALED_FINAL_CONTROL_ANALYSIS_SCHEMA",
    "SEALED_FINAL_EVIDENCE_SCHEMA",
    "SEALED_FINAL_RESULT_SCHEMA",
    "SEALED_FINAL_RUN_IDENTITY_SCHEMA",
    "SEALED_FINAL_VERIFICATION_SCHEMA",
    "SealedFinalContractError",
    "analyze_final_control_records",
    "build_final_rejection_probe_set",
    "build_final_result",
    "evaluate_final_language_quality",
    "final_metric_contract",
    "validate_final_authorization",
]
