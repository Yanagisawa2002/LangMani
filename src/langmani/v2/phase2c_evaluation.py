"""Validation gates, confidence intervals, and failure analysis for Phase 2C."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langmani.datasets.identity import sha256_hex
from langmani.v2.evaluator import EpisodeOutcome
from langmani.v2.phase2c import Phase2CContractError

PHASE2C_FAILURE_TAXONOMY = (
    "no_initial_motion",
    "incorrect_approach_direction",
    "premature_contact",
    "contact_loss",
    "insufficient_push",
    "overshoot",
    "lateral_drift",
    "cylinder_rotation_failure",
    "wrong_object_interaction",
    "workspace_exit",
    "action_saturation",
    "action_oscillation",
    "gripper_misuse",
    "language_target_confusion",
    "timeout",
    "verification_boundary_failure",
    "invalid_policy_output",
    "other",
)


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Phase2CContractError(f"{label} must be an integer")
    return value


def _float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise Phase2CContractError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise Phase2CContractError(f"{label} must be finite")
    return result


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> list[float]:
    """Return a two-sided 95% Wilson score interval for a binomial rate."""

    if total < 1 or successes < 0 or successes > total:
        raise Phase2CContractError("Wilson interval requires 0 <= successes <= total")
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total**2)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def development_competence_gate(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Apply the predeclared validation-only gate without consulting test outcomes."""

    if not records:
        raise Phase2CContractError("development gate requires validation episodes")
    if any(item.get("split") != "validation" for item in records):
        raise Phase2CContractError("development selection may consume validation episodes only")
    total = len(records)
    success_count = sum(item.get("success") is True for item in records)
    standard = [item for item in records if item.get("difficulty") == "standard"]
    standard_successes = sum(item.get("success") is True for item in standard)
    cylinder_successes = sum(
        item.get("success") is True and item.get("geometry") == "horizontal_cylinder"
        for item in records
    )
    lateral_successes = sum(
        item.get("success") is True and item.get("direction") in {"left", "right"}
        for item in records
    )
    invalid_count = sum(
        item.get("invalid_policy_output") is True or item.get("action_bound_failure") is True
        for item in records
    )
    nonfinite_count = sum(item.get("nonfinite_action") is True for item in records)
    checks = {
        "overall_success": success_count / total >= 0.50,
        "standard_success": bool(standard) and standard_successes / len(standard) >= 0.60,
        "cylinder_success": cylinder_successes >= 1,
        "lateral_success": lateral_successes >= 1,
        "finite_actions": nonfinite_count == 0,
        "invalid_or_action_bound_rate": invalid_count / total <= 0.02,
    }
    return {
        "schema_version": "langmani-v2-phase2c-development-gate-v0",
        "episode_count": total,
        "success_count": success_count,
        "success_rate": success_count / total,
        "standard_episode_count": len(standard),
        "standard_success_count": standard_successes,
        "standard_success_rate": standard_successes / len(standard) if standard else 0.0,
        "cylinder_success_count": cylinder_successes,
        "lateral_success_count": lateral_successes,
        "nonfinite_action_episode_count": nonfinite_count,
        "invalid_or_action_bound_episode_count": invalid_count,
        "checks": checks,
        "passed": all(checks.values()),
        "final_test_authorized": all(checks.values()),
    }


def classify_failure(record: Mapping[str, object]) -> str | None:
    """Assign one conservative primary Phase 2C failure category."""

    if record.get("success") is True:
        return None
    final = record.get("final_evaluation")
    final = final if isinstance(final, Mapping) else {}
    reason = str(record.get("failure_reason") or "").lower()
    if record.get("invalid_action") is True or "non-finite" in reason or "policy" in reason:
        return "invalid_policy_output"
    if record.get("wrong_object_interaction") is True:
        return "wrong_object_interaction"
    if final.get("target_outside_workspace") is True or final.get("target_off_table") is True:
        return "workspace_exit"
    if final.get("target_overshoot") is True:
        return "overshoot"
    if final.get("target_toppled") is True:
        return "cylinder_rotation_failure"
    if final.get("target_lifted") is True or final.get("target_is_grasped") is True:
        return "gripper_misuse"
    if final.get("no_progress_stall") is True:
        return (
            "no_initial_motion"
            if _integer(record.get("episode_length", 0), "episode_length") < 25
            else "insufficient_push"
        )
    if (
        record.get("action_saturation_rate", 0.0)
        and _float(record["action_saturation_rate"], "action_saturation_rate") > 0.1
    ):
        return "action_saturation"
    if record.get("timeout") is True or record.get("outcome") == EpisodeOutcome.TIMEOUT.value:
        return "timeout"
    if "verification" in reason:
        return "verification_boundary_failure"
    return "other"


def summarize_sealed_results(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate frozen final episodes without changing any selection setting."""

    if not records:
        raise Phase2CContractError("sealed result summary requires episodes")
    groups: dict[str, dict[str, list[Mapping[str, object]]]] = {
        name: defaultdict(list)
        for name in ("split", "geometry", "direction", "difficulty", "template_group")
    }
    for record in records:
        for name, values in groups.items():
            value = record.get(name)
            if isinstance(value, str) and value:
                values[value].append(record)

    def aggregate(values: Sequence[Mapping[str, object]]) -> dict[str, object]:
        success = sum(item.get("success") is True for item in values)
        total = len(values)
        return {
            "episodes": total,
            "successes": success,
            "success_rate": success / total,
            "wilson_95": wilson_interval(success, total),
        }

    failure_counts = Counter(
        category for item in records if (category := classify_failure(item)) is not None
    )
    summary: dict[str, Any] = {
        "schema_version": "langmani-v2-phase2c-final-summary-v0",
        "overall": aggregate(records),
        "by": {
            name: {key: aggregate(values) for key, values in sorted(items.items())}
            for name, items in groups.items()
        },
        "failure_taxonomy": {key: failure_counts.get(key, 0) for key in PHASE2C_FAILURE_TAXONOMY},
        "invalid_or_action_bound_episode_count": sum(
            item.get("invalid_action") is True
            or item.get("invalid_policy_output") is True
            or item.get("action_bound_failure") is True
            for item in records
        ),
        "mean_action_saturation_rate": math.fsum(
            _float(item.get("action_saturation_rate", 0.0), "action_saturation_rate")
            for item in records
        )
        / len(records),
    }
    summary["semantic_sha256"] = f"sha256:{sha256_hex(summary)}"
    return summary


def quality_level(
    summary: Mapping[str, object],
    *,
    development_gate: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Classify the frozen policy by the predeclared Level 0--3 thresholds."""

    by = summary.get("by")
    if not isinstance(by, Mapping) or not isinstance(by.get("split"), Mapping):
        raise Phase2CContractError("final summary lacks split results")
    splits = by["split"]

    def rate(name: str) -> float:
        value = splits.get(name)
        return float(value.get("success_rate", 0.0)) if isinstance(value, Mapping) else 0.0

    invalid_count = _integer(
        summary.get("invalid_or_action_bound_episode_count", 0),
        "invalid_or_action_bound_episode_count",
    )
    overall = summary.get("overall")
    total = (
        _integer(overall.get("episodes", 0), "overall episodes")
        if isinstance(overall, Mapping)
        else 0
    )
    invalid_rate = invalid_count / total if total else 1.0
    validation_standard = rate("validation_standard")
    if development_gate is not None:
        validation_standard = _float(
            development_gate.get("standard_success_rate", 0.0),
            "development standard_success_rate",
        )
    standard = max(rate("train_distribution_sanity"), validation_standard)
    level3 = (
        validation_standard >= 0.80
        and rate("test_unseen_scene") >= 0.65
        and rate("test_unseen_language") >= 0.65
        and rate("test_hard") >= 0.50
        and rate("test_visual_shift") >= 0.60
        and invalid_rate <= 0.01
    )
    level2 = (
        standard >= 0.70
        and rate("test_unseen_scene") >= 0.55
        and rate("test_unseen_language") >= 0.55
        and rate("test_hard") >= 0.35
        and rate("test_visual_shift") >= 0.45
        and invalid_rate <= 0.02
    )
    level1 = standard >= 0.50
    level = 3 if level3 else (2 if level2 else (1 if level1 else 0))
    return {
        "schema_version": "langmani-v2-phase2c-quality-decision-v0",
        "quality_level": level,
        "phase2d_authorized_by_quality": level >= 2,
        "invalid_or_action_bound_rate": invalid_rate,
        "validation_standard_success_rate": validation_standard,
        "tested_not_tuned": True,
    }


@dataclass(frozen=True, slots=True)
class FrozenPolicyManifest:
    """Immutable validation-selected policy identity opened before sealed tests."""

    checkpoint_sha256: str
    base_model_revision: str
    training_seed: int
    training_config_sha256: str
    train_view_sha256: str
    normalization_sha256: str
    processor_sha256: str
    action_chunk_size: int
    execution_horizon: int
    control_frequency_hz: int
    inference_seed: int
    final_test_identity_sha256: str
    schema_version: str = "langmani-v2-phase2c-final-policy-v0"

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_sha256",
            "training_config_sha256",
            "train_view_sha256",
            "normalization_sha256",
            "processor_sha256",
            "final_test_identity_sha256",
        ):
            value = str(getattr(self, name))
            if len(value) != 64 or any(item not in "0123456789abcdef" for item in value):
                raise Phase2CContractError(f"{name} must be a lowercase SHA-256 digest")
        if self.action_chunk_size != 50 or self.execution_horizon not in {1, 4, 8}:
            raise Phase2CContractError("frozen chunk or execution horizon is invalid")
        if self.control_frequency_hz != 20:
            raise Phase2CContractError("Phase 2C control frequency must remain 20 Hz")
        if len(self.base_model_revision) not in range(7, 65):
            raise Phase2CContractError("frozen base model revision must be pinned")
        for name in ("training_seed", "inference_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise Phase2CContractError(f"{name} must be a non-negative integer")

    def to_dict(self) -> dict[str, object]:
        value = {name: getattr(self, name) for name in self.__dataclass_fields__}
        value["semantic_sha256"] = f"sha256:{sha256_hex(value)}"
        return value


__all__ = [
    "PHASE2C_FAILURE_TAXONOMY",
    "FrozenPolicyManifest",
    "classify_failure",
    "development_competence_gate",
    "quality_level",
    "summarize_sealed_results",
    "wilson_interval",
]
