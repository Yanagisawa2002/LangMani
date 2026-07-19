"""Pure contracts for the M5A three-scene paired control screen."""

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

THREE_SCENE_SCREEN_SCHEMA = "langmani-m5a-three-scene-control-screen-v0"
THREE_SCENE_ANALYSIS_SCHEMA = "langmani-m5a-three-scene-paired-analysis-v0"
THREE_SCENE_REJECTION_SET_SCHEMA = "langmani-m5a-three-scene-rejection-probes-v0"


class ThreeSceneControlError(ValueError):
    """Raised when paired screen evidence is incomplete or contradictory."""


@dataclass(frozen=True, slots=True)
class RejectionProbeCase:
    probe_id: str
    category: str
    command: str

    def to_dict(self) -> dict[str, str]:
        return {
            "probe_id": self.probe_id,
            "category": self.category,
            "command": self.command,
        }


THREE_SCENE_REJECTION_CASES = (
    RejectionProbeCase("unsupported_action", "unsupported_action", "Open the drawer."),
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
        "unsupported_object",
        "unsupported_object",
        "Pick up the yellow cube and place it in the left bin.",
    ),
    RejectionProbeCase("empty_noise", "empty_or_noise", "..."),
    RejectionProbeCase(
        "unresolved_correction",
        "unresolved_correction",
        "Pick up the red cube, no, actually...",
    ),
)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ThreeSceneControlError(f"{label} must be one object")
    return cast(Mapping[str, object], value)


def _finite_vector(value: object, *, label: str, length: int) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ThreeSceneControlError(f"{label} must be one numeric vector")
    result = [float(item) for item in value]
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise ThreeSceneControlError(f"{label} must contain {length} finite values")
    return result


def normalize_initial_scene_state(value: object) -> dict[str, object]:
    """Validate the privileged reset snapshot used only for paired audit."""

    payload = _mapping(value, label="initial physical state")
    if set(payload) != {"object_poses", "bin_poses", "panda_qpos"}:
        raise ThreeSceneControlError("initial physical state has an invalid field set")

    def poses(
        raw: object,
        *,
        keys: tuple[str, ...],
        label: str,
    ) -> dict[str, list[float]]:
        values = _mapping(raw, label=label)
        if set(values) != set(keys):
            raise ThreeSceneControlError(f"{label} has an invalid semantic ID set")
        return {key: _finite_vector(values[key], label=f"{label}.{key}", length=7) for key in keys}

    return {
        "object_poses": poses(
            payload["object_poses"],
            keys=("red_cube", "green_cube", "blue_cube"),
            label="object_poses",
        ),
        "bin_poses": poses(
            payload["bin_poses"],
            keys=("left_bin", "right_bin"),
            label="bin_poses",
        ),
        "panda_qpos": _finite_vector(payload["panda_qpos"], label="panda_qpos", length=9),
    }


def initial_scene_state_fingerprint(value: object) -> str:
    return f"sha256:{sha256_hex(normalize_initial_scene_state(value))}"


def build_rejection_probe_set(items: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Validate the predeclared six probes and bind them into one immutable set."""

    if len(items) != len(THREE_SCENE_REJECTION_CASES):
        raise ThreeSceneControlError("three-scene rejection set must contain exactly six probes")
    validated: list[dict[str, object]] = []
    for expected, raw in zip(THREE_SCENE_REJECTION_CASES, items, strict=True):
        item = dict(raw)
        if any(item.get(key) != value for key, value in expected.to_dict().items()):
            raise ThreeSceneControlError("rejection probe identity or order differs")
        probe = _mapping(item.get("probe"), label=f"{expected.probe_id} probe")
        result = _mapping(probe.get("result"), label=f"{expected.probe_id} result")
        decision = _mapping(result.get("decision"), label=f"{expected.probe_id} decision")
        dispatch = _mapping(result.get("dispatch"), label=f"{expected.probe_id} dispatch")
        zero_work = (
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
        )
        if not zero_work:
            raise ThreeSceneControlError(
                f"rejection probe {expected.probe_id} entered a forbidden runtime"
            )
        validated.append(item)
    base: dict[str, object] = {
        "schema_version": THREE_SCENE_REJECTION_SET_SCHEMA,
        "probe_count": len(validated),
        "cases": validated,
        "all_safe_rejections": True,
        "controller_lookup_count": 0,
        "policy_reset_count": 0,
        "environment_reset_count": 0,
        "environment_step_count": 0,
    }
    return {**base, "probe_set_fingerprint": f"sha256:{sha256_hex(base)}"}


def validate_rejection_probe_set(payload: Mapping[str, object]) -> dict[str, object]:
    base = dict(payload)
    fingerprint = base.pop("probe_set_fingerprint", None)
    cases = base.get("cases")
    if not isinstance(cases, list):
        raise ThreeSceneControlError("rejection probe set lacks its ordered cases")
    rebuilt = build_rejection_probe_set(tuple(_mapping(item, label="probe case") for item in cases))
    if fingerprint != rebuilt["probe_set_fingerprint"] or dict(payload) != rebuilt:
        raise ThreeSceneControlError("rejection probe set fingerprint or fields differ")
    return rebuilt


def _record_by_index(
    records: Sequence[Mapping[str, object]], *, label: str
) -> dict[int, Mapping[str, object]]:
    result: dict[int, Mapping[str, object]] = {}
    for record in records:
        identity = _mapping(record.get("identity"), label=f"{label} identity")
        index = identity.get("episode_index")
        if isinstance(index, bool) or not isinstance(index, int) or index in result:
            raise ThreeSceneControlError(f"{label} has duplicate or malformed episode indices")
        result[index] = record
    if set(result) != set(range(18)):
        raise ThreeSceneControlError(f"{label} must contain exactly episode indices 0..17")
    return result


def _success(record: Mapping[str, object]) -> bool:
    return _mapping(record.get("result"), label="episode result").get("end_to_end_success") is True


def analyze_three_scene_records(
    *,
    oracle_records: Sequence[Mapping[str, object]],
    neuro_symbolic_records: Sequence[Mapping[str, object]],
    rejection_probe_set: Mapping[str, object],
) -> dict[str, object]:
    """Recompute pairing, routing/control decomposition, and the conjunctive gate."""

    probes = validate_rejection_probe_set(rejection_probe_set)
    oracle = _record_by_index(oracle_records, label="Oracle evidence")
    learned = _record_by_index(neuro_symbolic_records, label="NeuroSymbolic evidence")
    expected_task_ids = tuple(stable_task_id(task) for task in CANONICAL_TASK_SPECS)
    scene_tasks: defaultdict[str, list[str]] = defaultdict(list)
    pairs: list[dict[str, object]] = []
    outcomes: Counter[str] = Counter()
    attributions: Counter[str] = Counter()
    per_task: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"episodes": 0, "oracle_success": 0, "neuro_symbolic_success": 0}
    )
    routing_wrong_object = routing_wrong_bin = routing_wrong_episode = false_rejection = 0
    malformed_executable = controller_agreement = paired_states = 0
    target_wrong_bin = target_off_table = arm_projection = malformed_actions = 0
    nonfinite_actions = infrastructure_failures = m2_calls = dispatch_after_rejection = 0
    raw_violations = gripper_projection = 0
    maximum_correction = 0.0
    policy_resets = 0

    for index in range(18):
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
            raise ThreeSceneControlError("paired Oracle/NeuroSymbolic identities differ")
        task_id = cast(str, oracle_identity["oracle_task_id"])
        scene_id = cast(str, oracle_identity["scene_id"])
        scene_tasks[scene_id].append(task_id)
        oracle_result = _mapping(oracle_record.get("result"), label="Oracle result")
        learned_result = _mapping(learned_record.get("result"), label="learned result")
        decision = _mapping(learned_result.get("decision"), label="learned decision")
        oracle_dispatch = _mapping(oracle_result.get("dispatch"), label="Oracle dispatch")
        learned_dispatch = _mapping(learned_result.get("dispatch"), label="learned dispatch")
        route = decision.get("status") == "route"
        wrong_object = route and decision.get("target_object_id") != _mapping(
            learned_result.get("oracle_task_spec"), label="oracle task"
        ).get("target_object_id")
        wrong_bin = route and decision.get("target_bin_id") != _mapping(
            learned_result.get("oracle_task_spec"), label="oracle task"
        ).get("target_bin_id")
        routing_wrong_object += int(wrong_object)
        routing_wrong_bin += int(wrong_bin)
        routing_wrong_episode += int(wrong_object or wrong_bin)
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
            route
            and decision.get("task_id") == task_id
            and oracle_dispatch.get("controller_task_id") == task_id
            and learned_dispatch.get("controller_task_id") == task_id
            and oracle_dispatch.get("controller_checkpoint_fingerprint")
            == learned_dispatch.get("controller_checkpoint_fingerprint")
        )
        controller_agreement += int(same_controller)
        oracle_audit = _mapping(oracle_record.get("paired_execution_audit"), label="Oracle audit")
        learned_audit_value = learned_record.get("paired_execution_audit")
        learned_audit = (
            None
            if learned_audit_value is None
            else _mapping(learned_audit_value, label="learned audit")
        )
        oracle_state = normalize_initial_scene_state(oracle_audit.get("initial_physical_state"))
        learned_state = (
            None
            if learned_audit is None
            else normalize_initial_scene_state(learned_audit.get("initial_physical_state"))
        )
        state_equal = (
            learned_state is not None
            and oracle_state == learned_state
            and oracle_audit.get("initial_physical_state_fingerprint")
            == initial_scene_state_fingerprint(oracle_state)
            and learned_audit.get("initial_physical_state_fingerprint")
            == initial_scene_state_fingerprint(learned_state)
        )
        paired_states += int(state_equal)
        policy_resets += int(oracle_audit.get("policy_reset_called") is True)
        policy_resets += int(
            learned_audit is not None and learned_audit.get("policy_reset_called") is True
        )
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
            raise ThreeSceneControlError("learned episode lacks exactly one failure attribution")
        attributions[attribution] += 1
        per_task[task_id]["episodes"] += 1
        per_task[task_id]["oracle_success"] += int(oracle_success)
        per_task[task_id]["neuro_symbolic_success"] += int(learned_success)

        for result in (oracle_result, learned_result):
            dispatch = _mapping(result.get("dispatch"), label="dispatch")
            if _mapping(result.get("decision"), label="decision").get("status") != "route":
                dispatch_after_rejection += int(
                    dispatch.get("dispatched") is True
                    or dispatch.get("controller_loaded") is True
                    or dispatch.get("policy_called") is True
                    or dispatch.get("environment_reset_called") is True
                    or int(dispatch.get("environment_step_count", 0)) > 0
                )
            control_value = result.get("control")
            if not isinstance(control_value, Mapping):
                infrastructure_failures += int(dispatch.get("controller_load_failed") is True)
                continue
            control = cast(Mapping[str, object], control_value)
            final = _mapping(control.get("final_evaluation"), label="final evaluation")
            action = _mapping(control.get("action_evidence"), label="action evidence")
            target_wrong_bin += int(final.get("target_in_wrong_bin") is True)
            target_off_table += int(final.get("target_off_table") is True)
            arm_projection += int(action.get("arm_projected_component_count", 0))
            gripper_projection += int(action.get("gripper_projected_component_count", 0))
            raw_violations += int(action.get("raw_violation_count", 0))
            malformed_actions += int(action.get("malformed_action_count", 0))
            nonfinite_actions += int(action.get("nonfinite_action_count", 0))
            infrastructure_failures += int(control.get("infrastructure_failed") is True)
            infrastructure_failures += int(control.get("controller_inference_failed") is True)
        for audit in (oracle_audit, learned_audit):
            if audit is None:
                continue
            rollout = _mapping(audit.get("rollout"), label="rollout audit")
            if rollout.get("action_bound_mode") != "project":
                raise ThreeSceneControlError("paired rollout changed the project action runtime")
            projection = _mapping(
                rollout.get("action_projection_summary"), label="projection summary"
            )
            maximum_correction = max(
                maximum_correction, float(projection.get("maximum_linf_correction", 0.0))
            )
        pairs.append(
            {
                "episode_index": index,
                "scene_seed": oracle_identity["scene_seed"],
                "scene_id": scene_id,
                "task_id": task_id,
                "language_example_id": oracle_identity["language_example_id"],
                "oracle_success": oracle_success,
                "neuro_symbolic_success": learned_success,
                "routing_correct": decision.get("task_id") == task_id,
                "same_controller": same_controller,
                "initial_physical_state_equal": state_equal,
                "paired_outcome": outcome,
                "failure_attribution": attribution,
            }
        )

    if len(scene_tasks) != 3 or any(
        tuple(tasks) != expected_task_ids for tasks in scene_tasks.values()
    ):
        raise ThreeSceneControlError("three-scene evidence lacks canonical six-task coverage")
    oracle_success_count = sum(int(_success(record)) for record in oracle.values())
    learned_success_count = sum(int(_success(record)) for record in learned.values())
    success_gap = abs(oracle_success_count - learned_success_count)
    gate_items = {
        "routing_errors_at_most_one": routing_wrong_episode <= 1,
        "wrong_object_routes_at_most_one": routing_wrong_object <= 1,
        "wrong_bin_routes_at_most_one": routing_wrong_bin <= 1,
        "false_rejections_at_most_one": false_rejection <= 1,
        "malformed_executable_decisions_zero": malformed_executable == 0,
        "success_gap_at_most_one": success_gap <= 1,
        "controller_identity_agreement_complete": controller_agreement == 18,
        "paired_initial_states_complete": paired_states == 18,
        "policy_reset_every_episode": policy_resets == 36,
        "dispatch_after_rejection_zero": dispatch_after_rejection == 0,
        "rejection_probes_zero_work": probes.get("all_safe_rejections") is True,
        "target_in_wrong_bin_zero": target_wrong_bin == 0,
        "target_off_table_zero": target_off_table == 0,
        "arm_projection_zero": arm_projection == 0,
        "nan_count_zero": nonfinite_actions == 0,
        "inf_count_zero": nonfinite_actions == 0,
        "malformed_actions_zero": malformed_actions == 0,
        "m2_expert_calls_zero": m2_calls == 0,
        "infrastructure_failures_zero": infrastructure_failures == 0,
    }
    gate_passed = all(gate_items.values())
    base: dict[str, object] = {
        "schema_version": THREE_SCENE_ANALYSIS_SCHEMA,
        "pair_count": 18,
        "scene_count": 3,
        "six_tasks_per_scene_validated": True,
        "paired_results": pairs,
        "paired_outcome_counts": dict(sorted(outcomes.items())),
        "oracle_success_count": oracle_success_count,
        "neuro_symbolic_success_count": learned_success_count,
        "paired_success_difference": learned_success_count - oracle_success_count,
        "absolute_success_gap": success_gap,
        "exact_success_gap_confidence_interval": {
            "available": False,
            "reason": "no pre-existing exact paired-difference interval contract",
        },
        "routing_error_count": routing_wrong_episode,
        "wrong_object_route_count": routing_wrong_object,
        "wrong_bin_route_count": routing_wrong_bin,
        "false_rejection_count": false_rejection,
        "malformed_executable_decision_count": malformed_executable,
        "controller_identity_agreement_count": controller_agreement,
        "paired_initial_state_count": paired_states,
        "policy_reset_count": policy_resets,
        "failure_attribution_counts": dict(sorted(attributions.items())),
        "per_task_results": dict(sorted(per_task.items())),
        "raw_action_violation_count": raw_violations,
        "gripper_projection_count": gripper_projection,
        "arm_projection_count": arm_projection,
        "maximum_action_correction": maximum_correction,
        "malformed_action_count": malformed_actions,
        "nonfinite_action_count": nonfinite_actions,
        "target_in_wrong_bin_count": target_wrong_bin,
        "target_off_table_count": target_off_table,
        "infrastructure_failure_count": infrastructure_failures,
        "m2_expert_call_count": m2_calls,
        "gate_items": gate_items,
        "three_scene_control_screen_completed": True,
        "three_scene_control_screen_passed": gate_passed,
        "selected_router_locked": gate_passed,
        "full_control_development_authorized": gate_passed,
        "full_control_development_completed": False,
    }
    return {**base, "analysis_fingerprint": f"sha256:{sha256_hex(base)}"}


__all__ = [
    "THREE_SCENE_ANALYSIS_SCHEMA",
    "THREE_SCENE_REJECTION_CASES",
    "THREE_SCENE_REJECTION_SET_SCHEMA",
    "THREE_SCENE_SCREEN_SCHEMA",
    "RejectionProbeCase",
    "ThreeSceneControlError",
    "analyze_three_scene_records",
    "build_rejection_probe_set",
    "initial_scene_state_fingerprint",
    "normalize_initial_scene_state",
    "validate_rejection_probe_set",
]
