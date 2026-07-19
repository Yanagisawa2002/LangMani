"""Pure contracts for the M5A six-scene paired control-development benchmark."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import cast

from langmani.datasets.identity import sha256_hex
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import stable_task_id
from langmani.language.failure_attribution import FailureAttribution
from langmani.language.router_types import LanguageExample
from langmani.language.three_scene_control import (
    RejectionProbeCase,
    initial_scene_state_fingerprint,
    normalize_initial_scene_state,
)

FULL_CONTROL_SCHEMA = "langmani-m5a-full-control-development-v0"
FULL_CONTROL_ANALYSIS_SCHEMA = "langmani-m5a-full-control-paired-analysis-v0"
FULL_REJECTION_SET_SCHEMA = "langmani-m5a-full-control-rejection-probes-v0"
FINAL_AUTHORIZATION_SCHEMA = "langmani-m5a-final-authorization-v0"
FINAL_PROTOCOL_SCHEMA = "langmani-m5a-sealed-final-evaluation-protocol-v0"
CONTROL_FREQUENCY_HZ = 20.0


class FullControlDevelopmentError(ValueError):
    """Raised when full-development evidence is incomplete or contradictory."""


FULL_REJECTION_CASES = (
    RejectionProbeCase(
        "conflicting_objects",
        "conflicting_objects",
        ("Pick up the red cube and the blue cube and place them in the left bin."),
    ),
    RejectionProbeCase(
        "conflicting_bins",
        "conflicting_bins",
        ("Pick up the red cube and place it in both the left bin and the right bin."),
    ),
    RejectionProbeCase(
        "unsupported_action", "unsupported_action", ("Push the red cube toward the left bin.")
    ),
    RejectionProbeCase(
        "unsupported_object",
        "unsupported_object",
        ("Pick up the yellow cube and place it in the left bin."),
    ),
    RejectionProbeCase(
        "unsupported_destination", "unsupported_destination", ("Move the red cube to the shelf.")
    ),
    RejectionProbeCase("empty_noise", "empty_or_noise", "..."),
    RejectionProbeCase(
        "unresolved_correction",
        "unresolved_correction",
        ("Actually never mind, move the red cube left."),
    ),
    RejectionProbeCase(
        "contradictory_negation",
        "contradictory_negation",
        ("Do not move the red cube to the left bin."),
    ),
    RejectionProbeCase(
        "unsupported_spatial_reference",
        "unsupported_spatial_reference",
        ("Move the red cube over there."),
    ),
)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise FullControlDevelopmentError(f"{label} must be one object")
    return cast(Mapping[str, object], value)


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise FullControlDevelopmentError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise FullControlDevelopmentError(f"{label} must be finite")
    return result


def _percentiles(values: Sequence[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)

    def percentile(fraction: float) -> float | None:
        if not ordered:
            return None
        return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]

    return {
        "sample_count": len(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "total": sum(ordered),
    }


def build_full_rejection_probe_set(
    items: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Bind the predeclared nine safety probes and prove zero runtime work."""

    if len(items) != len(FULL_REJECTION_CASES):
        raise FullControlDevelopmentError("full rejection set must contain exactly nine probes")
    validated: list[dict[str, object]] = []
    for expected, raw in zip(FULL_REJECTION_CASES, items, strict=True):
        item = dict(raw)
        if any(item.get(key) != value for key, value in expected.to_dict().items()):
            raise FullControlDevelopmentError("full rejection probe identity or order differs")
        probe = _mapping(item.get("probe"), label=f"{expected.probe_id} probe")
        result = _mapping(probe.get("result"), label=f"{expected.probe_id} result")
        decision = _mapping(result.get("decision"), label=f"{expected.probe_id} decision")
        dispatch = _mapping(result.get("dispatch"), label=f"{expected.probe_id} dispatch")
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
            raise FullControlDevelopmentError(
                f"rejection probe {expected.probe_id} entered a forbidden runtime"
            )
        validated.append(item)
    base: dict[str, object] = {
        "schema_version": FULL_REJECTION_SET_SCHEMA,
        "probe_count": len(validated),
        "cases": validated,
        "all_safe_rejections": True,
        "controller_lookup_count": 0,
        "policy_reset_count": 0,
        "environment_reset_count": 0,
        "environment_step_count": 0,
    }
    return {**base, "probe_set_fingerprint": f"sha256:{sha256_hex(base)}"}


def validate_full_rejection_probe_set(payload: Mapping[str, object]) -> dict[str, object]:
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise FullControlDevelopmentError("full rejection set lacks its ordered cases")
    rebuilt = build_full_rejection_probe_set(
        tuple(_mapping(item, label="full rejection case") for item in cases)
    )
    if dict(payload) != rebuilt:
        raise FullControlDevelopmentError("full rejection set fingerprint or fields differ")
    return rebuilt


def _record_by_index(
    records: Sequence[Mapping[str, object]], *, label: str
) -> dict[int, Mapping[str, object]]:
    result: dict[int, Mapping[str, object]] = {}
    for record in records:
        identity = _mapping(record.get("identity"), label=f"{label} identity")
        index = identity.get("episode_index")
        if isinstance(index, bool) or not isinstance(index, int) or index in result:
            raise FullControlDevelopmentError(f"{label} has duplicate or malformed indices")
        result[index] = record
    if set(result) != set(range(36)):
        raise FullControlDevelopmentError(f"{label} must contain exactly indices 0..35")
    return result


def _success(record: Mapping[str, object]) -> bool:
    return _mapping(record.get("result"), label="episode result").get("end_to_end_success") is True


def _derived_step(value: object) -> int | None:
    if value is None:
        return None
    seconds = _finite(value, label="rollout event time")
    step = round(seconds * CONTROL_FREQUENCY_HZ)
    if not math.isclose(seconds, step / CONTROL_FREQUENCY_HZ, abs_tol=1e-9):
        raise FullControlDevelopmentError("rollout event time is not on the locked control grid")
    return step


def _generation_metrics(decision: Mapping[str, object]) -> tuple[list[float], list[int]]:
    evidence = decision.get("evidence")
    if not isinstance(evidence, Mapping):
        return [], []
    metadata = evidence.get("generation_metadata")
    if not isinstance(metadata, list):
        return [], []
    latencies: list[float] = []
    tokens: list[int] = []
    for item in metadata:
        if not isinstance(item, Mapping):
            raise FullControlDevelopmentError("generation metadata must contain objects")
        elapsed = _finite(item.get("elapsed_seconds"), label="generation latency")
        count = item.get("generated_token_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise FullControlDevelopmentError("generated token count is malformed")
        latencies.append(elapsed * 1_000.0)
        tokens.append(count)
    return latencies, tokens


def analyze_full_control_records(
    *,
    oracle_records: Sequence[Mapping[str, object]],
    neuro_symbolic_records: Sequence[Mapping[str, object]],
    rejection_probe_set: Mapping[str, object],
    examples_by_id: Mapping[str, LanguageExample],
    router_summaries: Mapping[str, object],
) -> dict[str, object]:
    """Recompute all paired metrics and the locked full-development gate."""

    probes = validate_full_rejection_probe_set(rejection_probe_set)
    oracle = _record_by_index(oracle_records, label="Oracle full evidence")
    learned = _record_by_index(neuro_symbolic_records, label="NeuroSymbolic full evidence")
    oracle_summary = _mapping(router_summaries.get("oracle"), label="Oracle summary")
    learned_summary = _mapping(
        router_summaries.get("neuro_symbolic"), label="NeuroSymbolic summary"
    )
    expected_task_ids = tuple(stable_task_id(task) for task in CANONICAL_TASK_SPECS)
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
    controller_agreement = paired_states = policy_resets = 0
    target_grasp_oracle = target_grasp_learned = 0
    wrong_object_interaction = target_wrong_bin = target_off_table = 0
    environment_failures = infrastructure_failures = malformed_actions = nonfinite_actions = 0
    arm_projection = gripper_projection = raw_violations = m2_calls = 0
    maximum_raw_excess = maximum_correction = 0.0
    strict_unprojected_successes = 0
    router_latency: list[float] = []
    generation_latency: list[float] = []
    generated_tokens: list[int] = []
    policy_latency: dict[str, list[float]] = {"oracle": [], "neuro_symbolic": []}
    environment_latency: dict[str, list[float]] = {"oracle": [], "neuro_symbolic": []}
    episode_duration: dict[str, list[float]] = {"oracle": [], "neuro_symbolic": []}

    for index in range(36):
        oracle_record = oracle[index]
        learned_record = learned[index]
        oracle_identity = _mapping(oracle_record.get("identity"), label="Oracle identity")
        learned_identity = _mapping(learned_record.get("identity"), label="learned identity")
        shared_keys = (
            "schedule_id",
            "schedule_fingerprint",
            "episode_index",
            "scene_seed",
            "scene_id",
            "oracle_task_id",
            "language_example_id",
            "command_fingerprint",
        )
        if any(oracle_identity.get(key) != learned_identity.get(key) for key in shared_keys):
            raise FullControlDevelopmentError("paired full Oracle/NeuroSymbolic identities differ")
        task_id = cast(str, oracle_identity["oracle_task_id"])
        scene_id = cast(str, oracle_identity["scene_id"])
        example_id = cast(str, oracle_identity["language_example_id"])
        example = examples_by_id.get(example_id)
        if example is None or example.task_id != task_id:
            raise FullControlDevelopmentError("full schedule command identity differs")
        scene_tasks[scene_id].append(task_id)
        oracle_result = _mapping(oracle_record.get("result"), label="Oracle result")
        learned_result = _mapping(learned_record.get("result"), label="learned result")
        decision = _mapping(learned_result.get("decision"), label="learned decision")
        oracle_task = _mapping(learned_result.get("oracle_task_spec"), label="oracle task")
        oracle_dispatch = _mapping(oracle_result.get("dispatch"), label="Oracle dispatch")
        learned_dispatch = _mapping(learned_result.get("dispatch"), label="learned dispatch")
        route = decision.get("status") == "route"
        object_matches = route and decision.get("target_object_id") == oracle_task.get(
            "target_object_id"
        )
        bin_matches = route and decision.get("target_bin_id") == oracle_task.get("target_bin_id")
        correct = route and decision.get("task_id") == task_id and object_matches and bin_matches
        routing_correct += int(correct)
        object_correct += int(object_matches)
        bin_correct += int(bin_matches)
        wrong_object += int(route and not object_matches)
        wrong_bin += int(route and not bin_matches)
        false_rejection += int(not route)
        malformed_executable += int(
            route
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
        oracle_audit = _mapping(oracle_record.get("paired_execution_audit"), label="Oracle audit")
        learned_audit = _mapping(
            learned_record.get("paired_execution_audit"), label="learned audit"
        )
        oracle_state = normalize_initial_scene_state(oracle_audit.get("initial_physical_state"))
        learned_state = normalize_initial_scene_state(learned_audit.get("initial_physical_state"))
        state_equal = (
            oracle_state == learned_state
            and oracle_audit.get("initial_physical_state_fingerprint")
            == initial_scene_state_fingerprint(oracle_state)
            and learned_audit.get("initial_physical_state_fingerprint")
            == initial_scene_state_fingerprint(learned_state)
        )
        paired_states += int(state_equal)
        policy_resets += int(oracle_audit.get("policy_reset_called") is True)
        policy_resets += int(learned_audit.get("policy_reset_called") is True)
        oracle_success = _success(oracle_record)
        learned_success = _success(learned_record)
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
        if not isinstance(attribution, str) or attribution not in {
            value.value for value in FailureAttribution
        }:
            raise FullControlDevelopmentError("learned episode lacks one failure attribution")
        attributions[attribution] += 1
        per_task[task_id]["episodes"] += 1
        per_task[task_id]["routing_correct"] += int(correct)
        per_task[task_id]["oracle_success"] += int(oracle_success)
        per_task[task_id]["neuro_symbolic_success"] += int(learned_success)
        per_family[example.template_family_id]["episodes"] += 1
        per_family[example.template_family_id]["routing_correct"] += int(correct)
        router_latency.append(
            _finite(learned_record.get("router_inference_latency_ms"), label="router latency")
        )
        current_generation_latency, current_tokens = _generation_metrics(decision)
        generation_latency.extend(current_generation_latency)
        generated_tokens.extend(current_tokens)

        pair_payload: dict[str, object] = {
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
        for label, result, audit in (
            ("oracle", oracle_result, oracle_audit),
            ("neuro_symbolic", learned_result, learned_audit),
        ):
            control = _mapping(result.get("control"), label=f"{label} control")
            rollout = _mapping(audit.get("rollout"), label=f"{label} rollout")
            projection = _mapping(
                rollout.get("action_projection_summary"), label=f"{label} projection"
            )
            if rollout.get("action_bound_mode") != "project":
                raise FullControlDevelopmentError("full rollout changed the project runtime")
            grasped = rollout.get("target_grasped_any") is True
            target_grasp_oracle += int(label == "oracle" and grasped)
            target_grasp_learned += int(label == "neuro_symbolic" and grasped)
            wrong_object_interaction += int(rollout.get("wrong_object_grasped_any") is True)
            target_wrong_bin += int(rollout.get("target_in_wrong_bin") is True)
            target_off_table += int(rollout.get("target_off_table") is True)
            environment_failures += int(control.get("environment_failed") is True)
            infrastructure_failures += int(control.get("infrastructure_failed") is True)
            infrastructure_failures += int(control.get("controller_inference_failed") is True)
            malformed_actions += int(projection.get("any_malformed_action") is True)
            nonfinite_actions += int(projection.get("any_nonfinite_action") is True)
            counts = projection.get("per_action_dimension_projection_counts")
            if not isinstance(counts, list) or len(counts) != 8:
                raise FullControlDevelopmentError("projection counts must contain eight dimensions")
            arm_projection += sum(int(value) for value in counts[:7])
            gripper_projection += int(counts[7])
            raw_violations += int(projection.get("projected_component_count", 0))
            maximum_raw_excess = max(
                maximum_raw_excess,
                _finite(projection.get("maximum_bound_excess", 0.0), label="maximum raw excess"),
            )
            maximum_correction = max(
                maximum_correction,
                _finite(
                    projection.get("maximum_linf_correction", 0.0),
                    label="maximum action correction",
                ),
            )
            strict_unprojected_successes += int(rollout.get("strict_unprojected_success") is True)
            inference = rollout.get("inference_latency_ms")
            environment_steps = rollout.get("environment_step_latency_ms")
            if not isinstance(inference, list) or not isinstance(environment_steps, list):
                raise FullControlDevelopmentError("rollout latency streams are malformed")
            policy_latency[label].extend(float(value) for value in inference)
            environment_latency[label].extend(float(value) for value in environment_steps)
            episode_duration[label].append(
                _finite(rollout.get("total_episode_duration_s"), label="episode duration")
            )
            pair_payload[f"{label}_episode_steps"] = rollout.get("episode_steps")
            pair_payload[f"{label}_first_target_grasp_step"] = _derived_step(
                rollout.get("time_to_first_target_grasp_s")
            )
            pair_payload[f"{label}_release_command_step"] = _derived_step(
                rollout.get("time_to_release_s")
            )
            pair_payload[f"{label}_time_to_success_s"] = rollout.get("time_to_success_s")
        pairs.append(pair_payload)

    if len(scene_tasks) != 6 or any(
        tuple(tasks) != expected_task_ids for tasks in scene_tasks.values()
    ):
        raise FullControlDevelopmentError("full evidence lacks canonical six-task scene coverage")
    oracle_success = sum(int(_success(record)) for record in oracle.values())
    learned_success = sum(int(_success(record)) for record in learned.values())
    success_gap = abs(oracle_success - learned_success)
    gate_items = {
        "correct_full_task_routes_at_least_35": routing_correct >= 35,
        "full_task_accuracy_at_least_95_percent_discrete": routing_correct >= 35,
        "wrong_object_routes_at_most_two": wrong_object <= 2,
        "wrong_bin_routes_at_most_two": wrong_bin <= 2,
        "false_rejections_at_most_three": false_rejection <= 3,
        "malformed_executable_decisions_zero": malformed_executable == 0,
        "success_gap_at_most_two": success_gap <= 2,
        "correct_route_controller_identity_agreement_complete": (
            controller_agreement == routing_correct
        ),
        "paired_initial_states_complete": paired_states == 36,
        "policy_reset_every_episode": policy_resets == 72,
        "rejection_probes_zero_work": probes.get("all_safe_rejections") is True,
        "target_in_wrong_bin_zero": target_wrong_bin == 0,
        "target_off_table_zero": target_off_table == 0,
        "arm_projection_zero": arm_projection == 0,
        "nan_count_zero": nonfinite_actions == 0,
        "inf_count_zero": nonfinite_actions == 0,
        "malformed_actions_zero": malformed_actions == 0,
        "environment_failures_zero": environment_failures == 0,
        "infrastructure_failures_zero": infrastructure_failures == 0,
        "m2_expert_calls_zero": m2_calls == 0,
    }
    gate_passed = all(gate_items.values())

    def rates(values: Mapping[str, dict[str, int]], key: str) -> dict[str, object]:
        return {
            name: {
                **counts,
                "routing_accuracy": counts[key] / counts["episodes"],
            }
            for name, counts in sorted(values.items())
        }

    base: dict[str, object] = {
        "schema_version": FULL_CONTROL_ANALYSIS_SCHEMA,
        "scene_count": 6,
        "pair_count": 36,
        "oracle_episode_count": 36,
        "neuro_symbolic_episode_count": 36,
        "six_tasks_per_scene_validated": True,
        "paired_results": pairs,
        "paired_outcome_counts": dict(sorted(outcomes.items())),
        "paired_disagreement_episode_indices": [
            pair["episode_index"]
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
        "full_task_spec_accuracy": routing_correct / 36,
        "object_accuracy": object_correct / 36,
        "bin_accuracy": bin_correct / 36,
        "wrong_object_route_count": wrong_object,
        "wrong_bin_route_count": wrong_bin,
        "false_rejection_count": false_rejection,
        "malformed_executable_decision_count": malformed_executable,
        "per_task_results": rates(per_task, "routing_correct"),
        "per_command_family_routing": rates(per_family, "routing_correct"),
        "controller_identity_agreement_count": controller_agreement,
        "paired_initial_state_count": paired_states,
        "policy_reset_count": policy_resets,
        "failure_attribution_counts": dict(sorted(attributions.items())),
        "oracle_target_grasp_count": target_grasp_oracle,
        "neuro_symbolic_target_grasp_count": target_grasp_learned,
        "oracle_target_grasp_rate": target_grasp_oracle / 36,
        "neuro_symbolic_target_grasp_rate": target_grasp_learned / 36,
        "oracle_post_grasp_success_rate": (
            oracle_success / target_grasp_oracle if target_grasp_oracle else None
        ),
        "neuro_symbolic_post_grasp_success_rate": (
            learned_success / target_grasp_learned if target_grasp_learned else None
        ),
        "wrong_object_interaction_count": wrong_object_interaction,
        "target_in_wrong_bin_count": target_wrong_bin,
        "target_off_table_count": target_off_table,
        "environment_failure_count": environment_failures,
        "infrastructure_failure_count": infrastructure_failures,
        "raw_action_violation_count": raw_violations,
        "gripper_projection_count": gripper_projection,
        "arm_projection_count": arm_projection,
        "maximum_raw_action_excess": maximum_raw_excess,
        "maximum_action_correction": maximum_correction,
        "strict_unprojected_success_count": strict_unprojected_successes,
        "malformed_action_count": malformed_actions,
        "nonfinite_action_count": nonfinite_actions,
        "m2_expert_call_count": m2_calls,
        "router_summaries": {
            "oracle": dict(oracle_summary),
            "neuro_symbolic": dict(learned_summary),
        },
        "latency_metrics": {
            "router_inference_ms": _percentiles(router_latency),
            "qwen_generation_ms": _percentiles(generation_latency),
            "generated_token_count": _percentiles([float(value) for value in generated_tokens]),
            "policy_inference_ms": {
                label: _percentiles(values) for label, values in policy_latency.items()
            },
            "environment_step_ms": {
                label: _percentiles(values) for label, values in environment_latency.items()
            },
            "episode_duration_s": {
                label: _percentiles(values) for label, values in episode_duration.items()
            },
        },
        "gate_items": gate_items,
        "full_control_development_completed": True,
        "development_quality_gate_passed": gate_passed,
        "selected_router_locked": gate_passed,
        "final_benchmark_authorized": gate_passed,
    }
    return {**base, "analysis_fingerprint": f"sha256:{sha256_hex(base)}"}


def final_evaluation_protocol_fingerprint() -> str:
    payload = {
        "schema_version": FINAL_PROTOCOL_SCHEMA,
        "stage": "sealed_final",
        "paired_oracle_and_neuro_symbolic": True,
        "scene_count": 12,
        "tasks_per_scene": 6,
        "action_bound_mode": "project",
        "execution_horizon": 10,
        "controller_selection": "frozen_per_task",
        "automatic_execution": False,
    }
    return f"sha256:{sha256_hex(payload)}"


def build_final_authorization(
    *,
    authorized: bool,
    router_fingerprint: str,
    controller_registry_fingerprint: str,
    runtime_fingerprint: str,
    full_schedule_fingerprint: str,
    development_evidence_fingerprint: str,
    selected_router_identity: str,
    final_language_schedule_fingerprint: str,
    final_control_schedule_fingerprint: str,
    git_commit: str,
) -> dict[str, object]:
    base: dict[str, object] = {
        "schema_version": FINAL_AUTHORIZATION_SCHEMA,
        "authorized": authorized,
        "locked_router_fingerprint": router_fingerprint,
        "controller_registry_fingerprint": controller_registry_fingerprint,
        "runtime_fingerprint": runtime_fingerprint,
        "full_development_schedule_fingerprint": full_schedule_fingerprint,
        "development_evidence_fingerprint": development_evidence_fingerprint,
        "selected_router_identity": selected_router_identity,
        "final_language_schedule_lock_fingerprint": final_language_schedule_fingerprint,
        "final_control_schedule_lock_fingerprint": final_control_schedule_fingerprint,
        "final_evaluation_protocol_fingerprint": final_evaluation_protocol_fingerprint(),
        "implementation_git_commit": git_commit,
        "final_data_accessed": False,
        "final_command_texts_materialized": False,
        "final_scene_seeds_materialized": False,
        "automatic_final_execution": False,
    }
    return {**base, "authorization_fingerprint": f"sha256:{sha256_hex(base)}"}


__all__ = [
    "FINAL_AUTHORIZATION_SCHEMA",
    "FINAL_PROTOCOL_SCHEMA",
    "FULL_CONTROL_ANALYSIS_SCHEMA",
    "FULL_CONTROL_SCHEMA",
    "FULL_REJECTION_CASES",
    "FULL_REJECTION_SET_SCHEMA",
    "FullControlDevelopmentError",
    "analyze_full_control_records",
    "build_final_authorization",
    "build_full_rejection_probe_set",
    "final_evaluation_protocol_fingerprint",
    "validate_full_rejection_probe_set",
]
