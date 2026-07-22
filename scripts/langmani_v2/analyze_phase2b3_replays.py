"""Merge deterministic replay telemetry into the Phase 2B.3 failure audit."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import cast

from langmani.v2.push_expert_recovery import load_json, sha256_file, sha256_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--raw-replay-root", type=Path, required=True)
    parser.add_argument("--enriched-registry", type=Path, required=True)
    parser.add_argument("--root-cause-summary", type=Path, required=True)
    return parser.parse_args()


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return cast(Mapping[str, object], value)


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a sequence")
    return cast(Sequence[object], value)


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    return float(value)


def _case_key(value: Mapping[str, object]) -> tuple[str, int]:
    return str(value.get("candidate_id")), _integer(value.get("seed"), "seed")


def _raw_case(root: Path, case: Mapping[str, object]) -> dict[str, object]:
    candidate, seed = _case_key(case)
    path = root / f"{candidate}_{seed}_r0.json"
    raw = load_json(path)
    raw_case = _mapping(raw.get("case"), "raw case")
    if _case_key(raw_case) != (candidate, seed):
        raise ValueError(f"raw replay identity mismatch: {path}")
    if raw_case.get("source_commit") != case.get("source_commit"):
        raise ValueError(f"raw replay source mismatch: {path}")
    return raw


def _trace(raw: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    return tuple(
        _mapping(item, "step snapshot") for item in _sequence(raw.get("step_trace"), "step trace")
    )


def _phase_results(raw: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    result = _mapping(raw.get("result"), "result")
    return tuple(
        _mapping(item, "phase result")
        for item in _sequence(result.get("phase_results", []), "phase results")
    )


def _first_irrecoverable_step(
    record: Mapping[str, object], raw: Mapping[str, object]
) -> tuple[int | None, str]:
    trace = _trace(raw)
    category = str(record.get("failure_category"))
    if category == "OBJECT_WORKSPACE_VIOLATION":
        for item in trace:
            evaluation = _mapping(item.get("evaluation", {}), "evaluation")
            if evaluation.get("target_outside_workspace") is True:
                return _integer(item.get("step"), "step"), "FIRST_LATCHED_WORKSPACE_VIOLATION"
    if category == "STALLED_PROGRESS":
        for item in trace:
            evaluation = _mapping(item.get("evaluation", {}), "evaluation")
            if evaluation.get("no_progress_stall") is True:
                return _integer(item.get("step"), "step"), "FIRST_LATCHED_STALL"
    if category == "LOSS_OF_CONTACT":
        previous = False
        for item in trace:
            current = (
                item.get("target_contact") is True
                if isinstance(item.get("target_contact"), bool)
                else item.get("contact_proxy") is True
            )
            if previous and not current:
                return _integer(item.get("step"), "step"), "FIRST_REPLAYED_CONTACT_LOSS"
            previous = current
    completed_steps = 0
    for phase in _phase_results(raw):
        if phase.get("success") is not True:
            return completed_steps, "FAILED_PHASE_ENTRY_BOUNDARY"
        completed_steps += _integer(phase.get("environment_steps", 0), "environment steps")
    return None, "UNAVAILABLE"


def _planning_record(case: Mapping[str, object], raw: Mapping[str, object]) -> dict[str, object]:
    result = _mapping(raw.get("result"), "result")
    failed = next(
        (item for item in _phase_results(raw) if item.get("success") is not True),
        {},
    )
    calls = [
        dict(_mapping(item, "planner call"))
        for item in _sequence(raw.get("planner_calls", []), "planner calls")
    ]
    return {
        "candidate_id": case.get("candidate_id"),
        "seed": case.get("seed"),
        "object_shape": case.get("object_shape"),
        "failed_phase": result.get("failed_phase"),
        "status": result.get("status"),
        "planner_status": failed.get("planner_status"),
        "phase_environment_steps": failed.get("environment_steps"),
        "zero_step_failure": failed.get("environment_steps") == 0,
        "planner_calls": calls,
    }


def _timeout_record(case: Mapping[str, object], raw: Mapping[str, object]) -> dict[str, object]:
    result = _mapping(raw.get("result"), "result")
    trace = _trace(raw)
    distances = [
        _number(
            _mapping(item.get("evaluation", {}), "evaluation").get("target_distance"), "distance"
        )
        for item in trace
    ]
    contact_events = _sequence(case.get("contact_events", []), "contact events")
    final_evaluation = _mapping(result.get("final_environment_evaluation", {}), "evaluation")
    return {
        "candidate_id": case.get("candidate_id"),
        "seed": case.get("seed"),
        "object_shape": case.get("object_shape"),
        "failed_phase": result.get("failed_phase"),
        "episode_steps": result.get("total_environment_steps"),
        "initial_target_distance": distances[0],
        "minimum_target_distance": min(distances),
        "final_target_distance": distances[-1],
        "distance_improvement": distances[0] - distances[-1],
        "latched_no_progress_stall": final_evaluation.get("no_progress_stall") is True,
        "contact_event_count": len(contact_events),
        "contact_lost": any(
            _mapping(event, "contact event").get("event") == "CONTACT_LOST"
            for event in contact_events
        ),
        "maximum_object_displacement": case.get("maximum_object_displacement"),
    }


def _enrich_registry(
    registry: Mapping[str, object],
    validation_cases: Sequence[Mapping[str, object]],
    raw_by_key: Mapping[tuple[str, int], Mapping[str, object]],
) -> dict[str, object]:
    cases = {_case_key(case): case for case in validation_cases}
    enriched_records: list[dict[str, object]] = []
    for raw_record in _sequence(registry.get("records"), "records"):
        record = dict(_mapping(raw_record, "record"))
        key = _case_key(record)
        case = cases.get(key)
        if case is not None:
            first_step, semantic = _first_irrecoverable_step(record, raw_by_key[key])
            for field in (
                "object_initial_pose",
                "target_pose",
                "robot_initial_state",
                "maximum_workspace_margin",
                "minimum_workspace_margin",
                "minimum_action_bound_margin",
                "final_object_target_distance",
                "maximum_object_displacement",
                "contact_events",
            ):
                record[field] = case.get(field)
            record["first_irrecoverable_step"] = first_step
            record["first_irrecoverable_step_semantic"] = semantic
            record["replay_evidence"] = {
                "original_outcome_reproduced": case.get("original_outcome_reproduced"),
                "repeat_semantics_identical": case.get("repeat_semantics_identical"),
                "maximum_target_pose_repeat_error": case.get("maximum_target_pose_repeat_error"),
                "workspace_violation_detail": case.get("workspace_violation_detail"),
            }
            record["evidence_completeness"] = "DETERMINISTIC_STEP_REPLAY"
        enriched_records.append(record)
    payload = dict(registry)
    payload["schema_version"] = "langmani-v2-phase2b3-failure-registry-v1"
    payload["historical_records_digest"] = registry.get("records_digest")
    payload["records"] = enriched_records
    payload["records_digest"] = f"sha256:{sha256_json(enriched_records)}"
    payload["deterministically_replayed_record_count"] = len(validation_cases)
    return payload


def _root_cause_summary(
    *,
    registry: Mapping[str, object],
    validation: Mapping[str, object],
    cases: Sequence[Mapping[str, object]],
    raw_by_key: Mapping[tuple[str, int], Mapping[str, object]],
    validation_path: Path,
) -> dict[str, object]:
    planning: list[dict[str, object]] = []
    timeouts: list[dict[str, object]] = []
    shapes: dict[str, Counter[str]] = defaultdict(Counter)
    first_steps: Counter[str] = Counter()
    registry_by_key = {
        _case_key(_mapping(item, "record")): _mapping(item, "record")
        for item in _sequence(registry.get("records"), "records")
    }
    for case in cases:
        key = _case_key(case)
        raw = raw_by_key[key]
        record = registry_by_key[key]
        result = _mapping(raw.get("result"), "result")
        category = str(record.get("failure_category"))
        shapes[str(case.get("object_shape"))][category] += 1
        _, semantic = _first_irrecoverable_step(record, raw)
        first_steps[semantic] += 1
        if result.get("status") in {"planning_failure", "ik_failure"}:
            planning.append(_planning_record(case, raw))
        if result.get("status") == "timeout":
            timeouts.append(_timeout_record(case, raw))
    planner_phase_counts = Counter(str(item.get("failed_phase")) for item in planning)
    planner_status_counts = Counter(str(item.get("planner_status")) for item in planning)
    workspace = [
        {
            "candidate_id": case.get("candidate_id"),
            "seed": case.get("seed"),
            "object_shape": case.get("object_shape"),
            "entity": case.get("workspace_violation_entity"),
            "detail": case.get("workspace_violation_detail"),
        }
        for case in cases
        if case.get("workspace_violation_entity") is not None
    ]
    return {
        "schema_version": "langmani-v2-phase2b3-root-cause-summary-v1",
        "failure_registry_digest": registry.get("records_digest"),
        "replay_validation_sha256": f"sha256:{sha256_file(validation_path)}",
        "replay_runner_source_commit": validation.get("runner_source_commit"),
        "historical_source_commits": validation.get("historical_source_commits"),
        "replay_passed": validation.get("passed"),
        "replayed_case_count": len(cases),
        "replayed_worker_episode_count": validation.get("worker_episode_count"),
        "first_irrecoverable_step_semantics": dict(sorted(first_steps.items())),
        "workspace_violation_analysis": {
            "count": len(workspace),
            "entity_counts": dict(
                sorted(Counter(str(item.get("entity")) for item in workspace).items())
            ),
            "records": workspace,
        },
        "planning_failure_analysis": {
            "count": len(planning),
            "failed_phase_counts": dict(sorted(planner_phase_counts.items())),
            "planner_status_counts": dict(sorted(planner_status_counts.items())),
            "zero_step_failure_count": sum(
                item.get("zero_step_failure") is True for item in planning
            ),
            "records": planning,
        },
        "timeout_analysis": {
            "count": len(timeouts),
            "latched_no_progress_stall_count": sum(
                item.get("latched_no_progress_stall") is True for item in timeouts
            ),
            "contact_loss_count": sum(item.get("contact_lost") is True for item in timeouts),
            "mean_distance_improvement": (
                fmean(float(cast(float, item["distance_improvement"])) for item in timeouts)
                if timeouts
                else None
            ),
            "records": timeouts,
        },
        "shape_failure_counts": {
            shape: dict(sorted(counts.items())) for shape, counts in sorted(shapes.items())
        },
        "deterministic_replay_contract": {
            "all_original_outcomes_reproduced": all(
                case.get("original_outcome_reproduced") is True for case in cases
            ),
            "all_repeat_semantics_identical": all(
                case.get("repeat_semantics_identical") is True for case in cases
            ),
            "all_pose_repeats_within_tolerance": all(
                case.get("pose_repeat_within_tolerance") is True for case in cases
            ),
        },
        "official_collection_episodes": 0,
        "optimizer_steps": 0,
    }


def main() -> int:
    args = parse_args()
    registry = load_json(args.registry)
    validation = load_json(args.validation)
    if validation.get("passed") is not True:
        raise ValueError("replay validation must pass before root-cause analysis")
    cases = tuple(
        _mapping(item, "validation case")
        for item in _sequence(validation.get("cases"), "validation cases")
    )
    raw_by_key = {_case_key(case): _raw_case(args.raw_replay_root, case) for case in cases}
    enriched = _enrich_registry(registry, cases, raw_by_key)
    summary = _root_cause_summary(
        registry=enriched,
        validation=validation,
        cases=cases,
        raw_by_key=raw_by_key,
        validation_path=args.validation,
    )
    write_json(args.enriched_registry, enriched)
    write_json(args.root_cause_summary, summary)
    workspace_analysis = _mapping(summary["workspace_violation_analysis"], "workspace analysis")
    planning_analysis = _mapping(summary["planning_failure_analysis"], "planning analysis")
    timeout_analysis = _mapping(summary["timeout_analysis"], "timeout analysis")
    print(
        json.dumps(
            {
                "replayed_cases": len(cases),
                "workspace_violations": workspace_analysis["count"],
                "planning_failures": planning_analysis["count"],
                "timeouts": timeout_analysis["count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
