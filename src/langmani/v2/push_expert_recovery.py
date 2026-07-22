"""Evidence and seed contracts for the Phase 2B.3 Push expert recovery."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast


class PushExpertRecoveryError(ValueError):
    """Raised when Phase 2B.3 evidence violates a frozen contract."""


class PushFailureCategory(StrEnum):
    """Versioned first-irrevocable failure taxonomy."""

    PREAPPROACH_PLANNING_FAILURE = "PREAPPROACH_PLANNING_FAILURE"
    IK_OR_MOTION_PLANNING_FAILURE = "IK_OR_MOTION_PLANNING_FAILURE"
    PRECONTACT_ALIGNMENT_FAILURE = "PRECONTACT_ALIGNMENT_FAILURE"
    MISSED_CONTACT = "MISSED_CONTACT"
    WRONG_CONTACT_SIDE = "WRONG_CONTACT_SIDE"
    OBJECT_ROTATION_OR_ROLLOUT = "OBJECT_ROTATION_OR_ROLLOUT"
    LOSS_OF_CONTACT = "LOSS_OF_CONTACT"
    PUSH_DIRECTION_ERROR = "PUSH_DIRECTION_ERROR"
    ROBOT_WORKSPACE_VIOLATION = "ROBOT_WORKSPACE_VIOLATION"
    OBJECT_WORKSPACE_VIOLATION = "OBJECT_WORKSPACE_VIOLATION"
    ACTION_BOUND_REJECTION = "ACTION_BOUND_REJECTION"
    STALLED_PROGRESS = "STALLED_PROGRESS"
    TIMEOUT = "TIMEOUT"
    FALSE_SUCCESS = "FALSE_SUCCESS"
    SIMULATOR_ERROR = "SIMULATOR_ERROR"
    UNKNOWN = "UNKNOWN"


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise PushExpertRecoveryError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise PushExpertRecoveryError(f"{label} must be a sequence")
    return cast(Sequence[object], value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PushExpertRecoveryError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PushExpertRecoveryError(f"{label} must be an integer")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PushExpertRecoveryError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise PushExpertRecoveryError(f"{label} must be finite")
    return result


def load_json(path: str | Path) -> dict[str, object]:
    """Load one JSON object without schema coercion."""

    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PushExpertRecoveryError(f"cannot load JSON {source}: {error}") from error
    return dict(_mapping(value, str(source)))


def write_json(path: str | Path, value: Mapping[str, object]) -> None:
    """Atomically write compact review evidence."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(target)


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    """Hash a JSON value using canonical path-independent encoding."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def classify_historical_failure(result: Mapping[str, object]) -> PushFailureCategory:
    """Classify the earliest evidenced failure without inventing telemetry."""

    if result.get("success") is True:
        raise PushExpertRecoveryError("successful results have no failure category")
    status = str(result.get("status", ""))
    failed_phase = result.get("failed_phase")
    phase = str(failed_phase) if isinstance(failed_phase, str) else None
    evaluation = _mapping(result.get("final_environment_evaluation", {}), "evaluation")
    if status == "unexpected_exception":
        return PushFailureCategory.SIMULATOR_ERROR
    if evaluation.get("invalid_action") is True or evaluation.get("action_out_of_bounds") is True:
        return PushFailureCategory.ACTION_BOUND_REJECTION
    if evaluation.get("target_outside_workspace") is True:
        return PushFailureCategory.OBJECT_WORKSPACE_VIOLATION
    if evaluation.get("target_toppled") is True or status in {"target_lifted", "target_toppled"}:
        return PushFailureCategory.OBJECT_ROTATION_OR_ROLLOUT
    if evaluation.get("success") is True and status != "success":
        return PushFailureCategory.FALSE_SUCCESS
    if status == "ik_failure":
        return PushFailureCategory.IK_OR_MOTION_PLANNING_FAILURE
    if status == "planning_failure":
        if phase == "move_to_precontact":
            return PushFailureCategory.PREAPPROACH_PLANNING_FAILURE
        return PushFailureCategory.IK_OR_MOTION_PLANNING_FAILURE
    if status == "contact_failure":
        return PushFailureCategory.MISSED_CONTACT
    if evaluation.get("target_overshoot") is True or status == "push_failure":
        return PushFailureCategory.PUSH_DIRECTION_ERROR
    if status == "correction_failure":
        return PushFailureCategory.LOSS_OF_CONTACT
    if status == "timeout":
        if evaluation.get("no_progress_stall") is True:
            return PushFailureCategory.STALLED_PROGRESS
        return PushFailureCategory.TIMEOUT
    return PushFailureCategory.UNKNOWN


def _phase_boundary_step(result: Mapping[str, object]) -> int | None:
    total = 0
    for raw in _sequence(result.get("phase_results", []), "phase_results"):
        phase = _mapping(raw, "phase result")
        if phase.get("success") is not True:
            return total
        total += _integer(phase.get("environment_steps", 0), "environment_steps")
    return None


def _trace_metrics(trace_value: object, object_id: str) -> dict[str, object]:
    trace = tuple(_mapping(item, "trace item") for item in _sequence(trace_value, "trace"))
    if not trace:
        return {
            "object_initial_pose": None,
            "target_pose": None,
            "robot_initial_state": None,
            "maximum_workspace_margin": None,
            "minimum_workspace_margin": None,
            "maximum_action_bound_margin": None,
            "maximum_object_displacement": None,
            "contact_events": [],
        }
    initial = trace[0]
    margins = [_number(item.get("workspace_margin"), "workspace_margin") for item in trace]
    poses: list[Sequence[object]] = []
    for item in trace:
        pose = _mapping(item.get("object_poses", {}), "object poses").get(object_id)
        if isinstance(pose, Sequence) and not isinstance(pose, (str, bytes, bytearray)):
            poses.append(cast(Sequence[object], pose))
    displacement: float | None = None
    if poses:
        origin = [_number(value, "object pose") for value in poses[0][:2]]
        displacement = max(
            math.hypot(
                _number(pose[0], "object pose") - origin[0],
                _number(pose[1], "object pose") - origin[1],
            )
            for pose in poses
        )
    contact_events: list[dict[str, object]] = []
    previous = False
    for item in trace:
        current = item.get("contact_proxy") is True
        if current != previous:
            contact_events.append(
                {
                    "environment_step": item.get("environment_steps"),
                    "event": "CONTACT_ESTABLISHED" if current else "CONTACT_LOST",
                    "phase": item.get("phase"),
                }
            )
        previous = current
    return {
        "object_initial_pose": initial.get("target_object_pose"),
        "target_pose": initial.get("target_center"),
        "robot_initial_state": {"tcp_pose": initial.get("tcp_pose"), "joint_state": None},
        "maximum_workspace_margin": max(margins),
        "minimum_workspace_margin": min(margins),
        "maximum_action_bound_margin": None,
        "maximum_object_displacement": displacement,
        "contact_events": contact_events,
    }


def build_failure_registry(
    *, config: Mapping[str, object], history_root: str | Path
) -> dict[str, object]:
    """Rehash configured Candidate E--P reports and preserve every result."""

    root = Path(history_root)
    records: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    for raw_run in _sequence(config.get("runs"), "runs"):
        run = _mapping(raw_run, "run")
        run_id = _string(run.get("run_id"), "run_id")
        candidate_id = _string(run.get("candidate_id"), "candidate_id")
        source_commit = _string(run.get("source_commit"), "source_commit")
        report_path = root / run_id / _string(run.get("report"), "report")
        observed_digest = sha256_file(report_path)
        if observed_digest != _string(run.get("sha256"), "sha256"):
            raise PushExpertRecoveryError(f"historical report digest mismatch for {run_id}")
        report = load_json(report_path)
        runtime = _mapping(report.get("runtime"), "runtime")
        if runtime.get("git_commit") != source_commit:
            raise PushExpertRecoveryError(f"historical source mismatch for {run_id}")
        results = _sequence(report.get("results"), "results")
        sources.append(
            {
                "candidate_id": candidate_id,
                "run_id": run_id,
                "source_commit": source_commit,
                "report_sha256": f"sha256:{observed_digest}",
                "episode_count": len(results),
            }
        )
        for raw_result in results:
            result = _mapping(raw_result, "result")
            seed = _integer(result.get("scene_seed"), "scene_seed")
            object_id = _string(result.get("target_object_id"), "target_object_id")
            trace_path = root / run_id / f"seed{seed}_diagnostic_trace.json"
            trace_file = load_json(trace_path) if trace_path.is_file() else None
            trace = trace_file.get("diagnostic_trace", []) if trace_file is not None else []
            metrics = _trace_metrics(trace, object_id)
            phase_results = tuple(
                _mapping(item, "phase result")
                for item in _sequence(result.get("phase_results", []), "phase_results")
            )
            evaluation = _mapping(result.get("final_environment_evaluation", {}), "evaluation")
            records.append(
                {
                    "candidate_id": candidate_id,
                    "run_id": run_id,
                    "source_commit": source_commit,
                    "seed": seed,
                    "object_shape": "cylinder" if "cylinder" in object_id else "box",
                    "object_id": object_id,
                    "target_region_id": result.get("target_region_id"),
                    "difficulty": result.get("difficulty"),
                    "object_initial_pose": metrics["object_initial_pose"],
                    "target_pose": metrics["target_pose"],
                    "robot_initial_state": metrics["robot_initial_state"],
                    "episode_result": "SUCCESS" if result.get("success") is True else "FAILURE",
                    "termination_reason": result.get("status"),
                    "episode_steps": result.get("total_environment_steps"),
                    "planning_attempts": sum(
                        _integer(item.get("attempts", 0), "attempts") for item in phase_results
                    ),
                    "recovery_attempts": sum(
                        _integer(item.get("attempts", 0), "attempts")
                        for item in phase_results
                        if str(item.get("phase", "")).startswith("corrective_push")
                    ),
                    "maximum_workspace_margin": metrics["maximum_workspace_margin"],
                    "minimum_workspace_margin": metrics["minimum_workspace_margin"],
                    "maximum_action_bound_margin": metrics["maximum_action_bound_margin"],
                    "final_object_target_distance": evaluation.get("target_distance"),
                    "maximum_object_displacement": metrics["maximum_object_displacement"],
                    "contact_events": metrics["contact_events"],
                    "phase_at_failure": result.get("failed_phase"),
                    "first_irrecoverable_step": (
                        None if result.get("success") is True else _phase_boundary_step(result)
                    ),
                    "failure_category": (
                        None
                        if result.get("success") is True
                        else classify_historical_failure(result).value
                    ),
                    "evidence_completeness": (
                        "PHASE_AND_TRACE" if trace_file is not None else "PHASE_BOUNDARY_ONLY"
                    ),
                }
            )
    records.sort(key=lambda item: (str(item["candidate_id"]), _integer(item["seed"], "seed")))
    failures = [item for item in records if item["episode_result"] == "FAILURE"]
    counts = Counter(str(item["failure_category"]) for item in failures)
    return {
        "schema_version": "langmani-v2-phase2b3-failure-registry-v0",
        "historical_result_count": len(records),
        "success_count": len(records) - len(failures),
        "failure_count": len(failures),
        "failure_category_counts": dict(sorted(counts.items())),
        "source_reports": sources,
        "records": records,
        "records_digest": f"sha256:{sha256_json(records)}",
        "official_collection_episodes": 0,
        "optimizer_steps": 0,
    }


@dataclass(frozen=True, slots=True)
class SeedBank:
    """One contiguous and fingerprintable Phase 2B.3 seed bank."""

    name: str
    seed_start: int
    count: int
    access: str

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.seed_start, self.seed_start + self.count))

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "seed_start": self.seed_start,
            "seed_end_inclusive": self.seed_start + self.count - 1,
            "count": self.count,
            "access": self.access,
            "seeds_digest": f"sha256:{sha256_json(self.seeds)}",
        }


def build_seed_registry() -> dict[str, object]:
    """Build the frozen, pairwise-disjoint recovery seed contract."""

    development = SeedBank("development", 69_000, 100, "REPEATABLE_AFTER_FREEZE")
    acceptance = SeedBank("acceptance", 69_200, 100, "SEALED_UNTIL_DEVELOPMENT_PASSES")
    exclusions = (
        SeedBank("candidate_e", 66_000, 100, "HISTORICAL"),
        SeedBank("candidate_f", 66_100, 100, "HISTORICAL"),
        SeedBank("candidate_g", 66_200, 32, "HISTORICAL"),
        SeedBank("phase2b2_formal", 66_300, 100, "RESERVED_UNTOUCHED"),
        SeedBank("candidate_h", 66_400, 64, "HISTORICAL"),
        SeedBank("candidate_i", 66_500, 64, "HISTORICAL"),
        SeedBank("candidate_j", 66_600, 64, "HISTORICAL"),
        SeedBank("candidate_k", 66_700, 64, "HISTORICAL"),
        SeedBank("candidate_l", 66_800, 64, "HISTORICAL"),
        SeedBank("candidate_m", 66_900, 64, "HISTORICAL"),
        SeedBank("phase2b2_collection", 67_000, 810, "RESERVED_UNTOUCHED"),
        SeedBank("candidate_n", 68_000, 64, "HISTORICAL"),
        SeedBank("candidate_o", 68_100, 64, "HISTORICAL"),
        SeedBank("candidate_p", 68_200, 64, "HISTORICAL"),
    )
    all_banks = (*exclusions, development, acceptance)
    for index, bank in enumerate(all_banks):
        for other in all_banks[index + 1 :]:
            if set(bank.seeds).intersection(other.seeds):
                raise PushExpertRecoveryError(f"seed banks overlap: {bank.name}, {other.name}")
    payload = {
        "schema_version": "langmani-v2-phase2b3-seed-registry-v0",
        "failure_regression_semantic": "historical_failures_only_v0",
        "development": development.to_dict(),
        "acceptance": acceptance.to_dict(),
        "exclusions": [bank.to_dict() for bank in exclusions],
        "official_collection_episodes": 0,
    }
    return payload | {"registry_digest": f"sha256:{sha256_json(payload)}"}


def read_seed_bank(
    registry: Mapping[str, object],
    bank_name: str,
    *,
    development_report: Mapping[str, object] | None = None,
    expected_source_commit: str | None = None,
) -> tuple[int, ...]:
    """Read a bank while keeping acceptance sealed before development passes."""

    if bank_name not in {"development", "acceptance"}:
        raise PushExpertRecoveryError(f"unknown seed bank {bank_name!r}")
    bank = _mapping(registry.get(bank_name), bank_name)
    if bank_name == "acceptance":
        if development_report is None or expected_source_commit is None:
            raise PushExpertRecoveryError("acceptance bank is sealed before development passes")
        required = {
            "passed": True,
            "source_commit": expected_source_commit,
            "completed_episodes": 100,
            "workspace_violations": 0,
            "action_bound_violations": 0,
            "nonfinite_actions": 0,
            "simulator_errors": 0,
            "false_successes": 0,
        }
        for key, value in required.items():
            if development_report.get(key) != value:
                raise PushExpertRecoveryError(f"development report failed acceptance key {key}")
        if _integer(development_report.get("successes"), "successes") < 95:
            raise PushExpertRecoveryError("development report is below 95 successes")
    start = _integer(bank.get("seed_start"), "seed_start")
    count = _integer(bank.get("count"), "count")
    return tuple(range(start, start + count))


def root_cause_summary(registry: Mapping[str, object]) -> dict[str, object]:
    """Aggregate historical phase evidence without overclaiming absent traces."""

    records = tuple(
        _mapping(item, "record") for item in _sequence(registry.get("records"), "records")
    )
    failures = tuple(item for item in records if item.get("episode_result") == "FAILURE")
    category_counts = Counter(str(item.get("failure_category")) for item in failures)
    phase_counts = Counter(str(item.get("phase_at_failure")) for item in failures)
    shape_counts: dict[str, Counter[str]] = {}
    for item in failures:
        shape = str(item.get("object_shape"))
        shape_counts.setdefault(shape, Counter())[str(item.get("failure_category"))] += 1
    traced = tuple(
        item for item in failures if item.get("evidence_completeness") == "PHASE_AND_TRACE"
    )
    margins = [_number(item.get("minimum_workspace_margin"), "workspace margin") for item in traced]
    return {
        "schema_version": "langmani-v2-phase2b3-root-cause-summary-v0",
        "historical_failure_count": len(failures),
        "failure_category_counts": dict(sorted(category_counts.items())),
        "failure_phase_counts": dict(sorted(phase_counts.items())),
        "shape_failure_counts": {
            shape: dict(sorted(counts.items())) for shape, counts in sorted(shape_counts.items())
        },
        "phase_boundary_only_count": sum(
            item.get("evidence_completeness") == "PHASE_BOUNDARY_ONLY" for item in failures
        ),
        "phase_and_trace_count": len(traced),
        "observed_minimum_workspace_margin": min(margins) if margins else None,
        "workspace_violation_interpretation": (
            "target object footprint crossed the native environment boundary; historical reports "
            "contain no robot-link workspace violation flag"
        ),
        "planning_failure_interpretation": (
            "phase evidence distinguishes preapproach from later planning failure; deterministic "
            "replay is required for finer IK/target-pose attribution"
        ),
        "timeout_interpretation": (
            "no_progress_stall timeouts are STALLED_PROGRESS; contact-loss and rollout attribution "
            "requires step telemetry"
        ),
        "requires_deterministic_replay": True,
        "official_collection_episodes": 0,
    }


__all__ = [
    "PushExpertRecoveryError",
    "PushFailureCategory",
    "SeedBank",
    "build_failure_registry",
    "build_seed_registry",
    "classify_historical_failure",
    "load_json",
    "read_seed_bank",
    "root_cause_summary",
    "sha256_file",
    "sha256_json",
    "write_json",
]
