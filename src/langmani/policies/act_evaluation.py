"""Leakage-safe evaluation scheduling and checkpoint selection for M4 ACT.

This module is deliberately simulator-free.  It turns immutable rollout
records into benchmark summaries, owns the validation-only checkpoint ranking,
locks full-mode test evaluation to the selected checkpoint, and constructs the
predeclared fresh-scene schedule.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Self, cast
from uuid import uuid4

import numpy as np

from langmani.datasets.identity import sha256_hex
from langmani.policies.act_types import (
    ExperimentMode,
    RolloutBenchmarkResult,
    RolloutEpisodeResult,
    RolloutStatus,
    ValidationResult,
)

CHECKPOINT_SELECTION_SCHEMA_VERSION = "langmani-m4-checkpoint-selection-v1"
FRESH_SEED_SCHEDULE_SCHEMA_VERSION = "langmani-m4-fresh-seeds-v1"
TEST_AUTHORIZATION_SCHEMA_VERSION = "langmani-m4-test-authorization-v1"
FRESH_SEED_COUNT = 30
_MAX_SCENE_SEED = 2**31 - 1


def rollout_schedule_digest(
    *,
    m3b_export_fingerprint: str,
    variant: str,
    task_id: str | None,
    split: str,
    episodes: tuple[Mapping[str, object], ...],
) -> str:
    """Fingerprint one predeclared rollout schedule independently of a run path."""
    _digest(m3b_export_fingerprint, "m3b_export_fingerprint")
    if not variant or not split or not episodes:
        raise EvaluationContractError("rollout schedule identity fields must be non-empty")
    return _canonical_digest(
        {
            "schema_version": "langmani-m4-rollout-schedule-v1",
            "m3b_export_fingerprint": m3b_export_fingerprint,
            "variant": variant,
            "task_id": task_id,
            "split": split,
            "episodes": list(episodes),
        }
    )


class EvaluationContractError(ValueError):
    """Raised when evaluation evidence violates the declared M4 protocol."""


class SelectionLockError(RuntimeError):
    """Raised when code attempts to replace an immutable selection record."""


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA-256 string")
    normalized = value.removeprefix("sha256:")
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _timestamp(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError("selection_timestamp_utc must be a non-empty string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("selection_timestamp_utc must use ISO 8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError("selection_timestamp_utc must be UTC")
    return value


def _canonical_digest(payload: object) -> str:
    return f"sha256:{sha256_hex(payload)}"


def validation_ranking_key(result: ValidationResult) -> tuple[float, float, float, float, int]:
    """Return the exact, predeclared validation-only ranking key.

    Lower tuples rank first: highest success, then fewer wrong-object
    interactions, fewer off-table failures, lower offline action loss, and
    finally the earlier checkpoint.
    """

    if not isinstance(result, ValidationResult):
        raise TypeError("checkpoint candidates must be ValidationResult values")
    return (
        -result.success_rate,
        result.wrong_object_interaction_rate,
        result.target_off_table_rate,
        result.offline_validation_action_loss,
        result.checkpoint_step,
    )


def rank_validation_results(
    candidates: tuple[ValidationResult, ...],
) -> tuple[ValidationResult, ...]:
    """Rank unique checkpoints evaluated on one identical validation schedule."""

    if not isinstance(candidates, tuple) or not candidates:
        raise EvaluationContractError("checkpoint selection requires at least one candidate")
    if not all(isinstance(item, ValidationResult) for item in candidates):
        raise TypeError("checkpoint candidates must contain ValidationResult values")
    schedules = {item.schedule_digest for item in candidates}
    if len(schedules) != 1:
        raise EvaluationContractError("all candidates must use one validation schedule digest")
    fingerprints = [item.checkpoint_fingerprint for item in candidates]
    steps = [item.checkpoint_step for item in candidates]
    if len(set(fingerprints)) != len(fingerprints):
        raise EvaluationContractError("checkpoint candidate fingerprints must be unique")
    if len(set(steps)) != len(steps):
        raise EvaluationContractError("checkpoint candidate steps must be unique")
    return tuple(sorted(candidates, key=validation_ranking_key))


@dataclass(frozen=True, slots=True)
class CheckpointSelectionRecord:
    """Immutable evidence that validation, and only validation, selected a checkpoint."""

    run_fingerprint: str
    validation_schedule_digest: str
    candidates: tuple[ValidationResult, ...]
    ranked_checkpoint_fingerprints: tuple[str, ...]
    selected_checkpoint_fingerprint: str
    selected_checkpoint_step: int
    selection_timestamp_utc: str
    selection_fingerprint: str
    selection_locked: bool = True
    test_evaluation_status: str = "pending"
    schema_version: str = CHECKPOINT_SELECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _digest(self.run_fingerprint, "run_fingerprint")
        _digest(self.validation_schedule_digest, "validation_schedule_digest")
        _digest(self.selected_checkpoint_fingerprint, "selected_checkpoint_fingerprint")
        _digest(self.selection_fingerprint, "selection_fingerprint")
        _timestamp(self.selection_timestamp_utc)
        if self.schema_version != CHECKPOINT_SELECTION_SCHEMA_VERSION:
            raise EvaluationContractError("unknown checkpoint-selection schema")
        if self.selection_locked is not True:
            raise EvaluationContractError("checkpoint selection must be immutable and locked")
        if self.test_evaluation_status != "pending":
            raise EvaluationContractError(
                "selection record is immutable; test status belongs in a separate result"
            )
        ranked = rank_validation_results(self.candidates)
        if self.candidates != tuple(sorted(self.candidates, key=lambda item: item.checkpoint_step)):
            raise EvaluationContractError(
                "selection candidates must use canonical ascending checkpoint order"
            )
        if any(item.schedule_digest != self.validation_schedule_digest for item in ranked):
            raise EvaluationContractError("candidate schedule differs from selection schedule")
        expected_ranking = tuple(item.checkpoint_fingerprint for item in ranked)
        if self.ranked_checkpoint_fingerprints != expected_ranking:
            raise EvaluationContractError("stored checkpoint ranking is not the declared ranking")
        if self.selected_checkpoint_fingerprint != ranked[0].checkpoint_fingerprint:
            raise EvaluationContractError("selected checkpoint is not the highest-ranked candidate")
        if self.selected_checkpoint_step != ranked[0].checkpoint_step:
            raise EvaluationContractError(
                "selected checkpoint step differs from validation evidence"
            )
        if self.selection_fingerprint != self.compute_fingerprint():
            raise EvaluationContractError("selection fingerprint does not match semantic evidence")

    def fingerprint_payload(self) -> dict[str, object]:
        """Return stable selection semantics, excluding the wall-clock timestamp."""

        return {
            "schema_version": self.schema_version,
            "run_fingerprint": self.run_fingerprint,
            "validation_schedule_digest": self.validation_schedule_digest,
            "candidates": [item.to_dict() for item in self.candidates],
            "ranked_checkpoint_fingerprints": list(self.ranked_checkpoint_fingerprints),
            "selected_checkpoint_fingerprint": self.selected_checkpoint_fingerprint,
            "selected_checkpoint_step": self.selected_checkpoint_step,
            "selection_locked": self.selection_locked,
            "test_evaluation_status": self.test_evaluation_status,
        }

    def compute_fingerprint(self) -> str:
        return _canonical_digest(self.fingerprint_payload())

    def to_dict(self) -> dict[str, object]:
        return {
            **self.fingerprint_payload(),
            "selection_timestamp_utc": self.selection_timestamp_utc,
            "selection_fingerprint": self.selection_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if not isinstance(value, dict):
            raise TypeError("checkpoint-selection JSON must be an object")
        expected = {
            "schema_version",
            "run_fingerprint",
            "validation_schedule_digest",
            "candidates",
            "ranked_checkpoint_fingerprints",
            "selected_checkpoint_fingerprint",
            "selected_checkpoint_step",
            "selection_timestamp_utc",
            "selection_fingerprint",
            "selection_locked",
            "test_evaluation_status",
        }
        if set(value) != expected:
            raise EvaluationContractError("checkpoint-selection JSON fields are invalid")
        raw_candidates = value["candidates"]
        if not isinstance(raw_candidates, list):
            raise TypeError("checkpoint-selection candidates must be a list")
        candidates = tuple(_validation_result_from_dict(item) for item in raw_candidates)
        raw_ranking = value["ranked_checkpoint_fingerprints"]
        if not isinstance(raw_ranking, list) or not all(
            isinstance(item, str) for item in raw_ranking
        ):
            raise TypeError("ranked checkpoint fingerprints must be a string list")
        return cls(
            schema_version=cast(str, value["schema_version"]),
            run_fingerprint=cast(str, value["run_fingerprint"]),
            validation_schedule_digest=cast(str, value["validation_schedule_digest"]),
            candidates=candidates,
            ranked_checkpoint_fingerprints=tuple(raw_ranking),
            selected_checkpoint_fingerprint=cast(str, value["selected_checkpoint_fingerprint"]),
            selected_checkpoint_step=cast(int, value["selected_checkpoint_step"]),
            selection_timestamp_utc=cast(str, value["selection_timestamp_utc"]),
            selection_fingerprint=cast(str, value["selection_fingerprint"]),
            selection_locked=cast(bool, value["selection_locked"]),
            test_evaluation_status=cast(str, value["test_evaluation_status"]),
        )


def create_checkpoint_selection(
    *,
    run_fingerprint: str,
    candidates: tuple[ValidationResult, ...],
    selection_timestamp_utc: str | None = None,
) -> CheckpointSelectionRecord:
    """Create deterministic ranking evidence with a non-semantic audit timestamp."""

    _digest(run_fingerprint, "run_fingerprint")
    ranked = rank_validation_results(candidates)
    ordered_candidates = tuple(sorted(candidates, key=lambda item: item.checkpoint_step))
    timestamp = selection_timestamp_utc or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    partial = {
        "schema_version": CHECKPOINT_SELECTION_SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "validation_schedule_digest": ranked[0].schedule_digest,
        "candidates": ordered_candidates,
        "ranked_checkpoint_fingerprints": tuple(item.checkpoint_fingerprint for item in ranked),
        "selected_checkpoint_fingerprint": ranked[0].checkpoint_fingerprint,
        "selected_checkpoint_step": ranked[0].checkpoint_step,
        "selection_locked": True,
        "test_evaluation_status": "pending",
    }
    fingerprint_payload = {
        key: [item.to_dict() for item in value] if key == "candidates" else value
        for key, value in partial.items()
    }
    return CheckpointSelectionRecord(
        **partial,
        selection_timestamp_utc=timestamp,
        selection_fingerprint=_canonical_digest(fingerprint_payload),
    )


def load_checkpoint_selection(path: Path) -> CheckpointSelectionRecord:
    """Load and fully revalidate one local immutable selection record."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SelectionLockError(f"cannot load checkpoint selection {path}: {error}") from error
    return CheckpointSelectionRecord.from_dict(value)


def write_checkpoint_selection_atomic(
    path: Path, record: CheckpointSelectionRecord
) -> CheckpointSelectionRecord:
    """Publish a complete record exactly once without replacing existing evidence.

    A fully written sibling is hard-linked into place.  Link creation is atomic
    and fails if another process already published the destination.
    """

    if not isinstance(path, Path):
        raise TypeError("selection path must be a pathlib.Path")
    if not isinstance(record, CheckpointSelectionRecord):
        raise TypeError("record must be a CheckpointSelectionRecord")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = load_checkpoint_selection(path)
        if existing == record:
            return existing
        raise SelectionLockError(f"immutable checkpoint selection already exists: {path}")

    temporary = path.with_name(f".{path.name}.{uuid4().hex}.staging")
    payload = json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = load_checkpoint_selection(path)
            if existing == record:
                return existing
            raise SelectionLockError(
                f"another process published a different checkpoint selection: {path}"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)
    return load_checkpoint_selection(path)


@dataclass(frozen=True, slots=True)
class TestEvaluationAuthorization:
    """Machine-readable result of the full-mode test lock guard."""

    run_fingerprint: str
    checkpoint_fingerprint: str
    selection_fingerprint: str | None
    actual_schedule_digest: str
    expected_schedule_digest: str
    mode: ExperimentMode
    authorized: bool
    final_eligible: bool
    development_override: bool
    reason: str
    schema_version: str = TEST_AUTHORIZATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _digest(self.run_fingerprint, "run_fingerprint")
        _digest(self.checkpoint_fingerprint, "checkpoint_fingerprint")
        _digest(self.actual_schedule_digest, "actual_schedule_digest")
        _digest(self.expected_schedule_digest, "expected_schedule_digest")
        if self.selection_fingerprint is not None:
            _digest(self.selection_fingerprint, "selection_fingerprint")
        object.__setattr__(self, "mode", ExperimentMode(self.mode))
        for name in ("authorized", "final_eligible", "development_override"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if self.schema_version != TEST_AUTHORIZATION_SCHEMA_VERSION:
            raise EvaluationContractError("unknown test-authorization schema")
        if not isinstance(self.reason, str) or not self.reason:
            raise EvaluationContractError("test authorization requires a reason")
        if self.final_eligible and (
            not self.authorized or self.development_override or self.mode is not ExperimentMode.FULL
        ):
            raise EvaluationContractError("only a locked full-mode test may be final-eligible")
        if self.development_override and self.mode is not ExperimentMode.DEVELOPMENT:
            raise EvaluationContractError("test override is development-only")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["mode"] = self.mode.value
        return value

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if not isinstance(value, dict):
            raise TypeError("test-authorization JSON must be an object")
        expected = {
            "run_fingerprint",
            "checkpoint_fingerprint",
            "selection_fingerprint",
            "actual_schedule_digest",
            "expected_schedule_digest",
            "mode",
            "authorized",
            "final_eligible",
            "development_override",
            "reason",
            "schema_version",
        }
        if set(value) != expected:
            raise EvaluationContractError("test-authorization JSON fields are invalid")
        return cls(**value)


def authorize_test_evaluation(
    *,
    run_fingerprint: str,
    checkpoint_fingerprint: str,
    actual_schedule_digest: str,
    expected_schedule_digest: str,
    mode: ExperimentMode,
    selection: CheckpointSelectionRecord | None,
    development_override_reason: str | None = None,
) -> TestEvaluationAuthorization:
    """Enforce selected-checkpoint and fixed-schedule locks before test access."""

    mode = ExperimentMode(mode)
    _digest(run_fingerprint, "run_fingerprint")
    _digest(checkpoint_fingerprint, "checkpoint_fingerprint")
    _digest(actual_schedule_digest, "actual_schedule_digest")
    _digest(expected_schedule_digest, "expected_schedule_digest")
    failures: list[str] = []
    if mode not in {ExperimentMode.FULL, ExperimentMode.DEVELOPMENT}:
        failures.append("test evaluation is permitted only in full or development mode")
    if selection is None:
        failures.append("checkpoint selection record is missing")
    elif run_fingerprint != selection.run_fingerprint:
        failures.append("test run differs from the checkpoint-selection run")
    elif checkpoint_fingerprint != selection.selected_checkpoint_fingerprint:
        failures.append("test checkpoint is not the validation-selected checkpoint")
    if actual_schedule_digest != expected_schedule_digest:
        failures.append("test episode schedule differs from the predeclared schedule")

    if mode is ExperimentMode.FULL:
        if development_override_reason is not None:
            failures.append("development override cannot authorize full mode")
        authorized = not failures
        return TestEvaluationAuthorization(
            run_fingerprint=run_fingerprint,
            checkpoint_fingerprint=checkpoint_fingerprint,
            selection_fingerprint=None if selection is None else selection.selection_fingerprint,
            actual_schedule_digest=actual_schedule_digest,
            expected_schedule_digest=expected_schedule_digest,
            mode=mode,
            authorized=authorized,
            final_eligible=authorized,
            development_override=False,
            reason="full test lock satisfied" if authorized else "; ".join(failures),
        )

    override = development_override_reason is not None
    if override and (
        not isinstance(development_override_reason, str) or not development_override_reason
    ):
        raise EvaluationContractError("development override reason must be non-empty")
    override_applied = bool(failures and mode is ExperimentMode.DEVELOPMENT and override)
    authorized = not failures or override_applied
    reason = (
        "development evaluation satisfies the lock but remains nonfinal"
        if not failures
        else "; ".join(failures)
    )
    if failures and override:
        reason += f"; development-only override: {development_override_reason}"
    return TestEvaluationAuthorization(
        run_fingerprint=run_fingerprint,
        checkpoint_fingerprint=checkpoint_fingerprint,
        selection_fingerprint=None if selection is None else selection.selection_fingerprint,
        actual_schedule_digest=actual_schedule_digest,
        expected_schedule_digest=expected_schedule_digest,
        mode=mode,
        authorized=authorized,
        final_eligible=False,
        development_override=override_applied,
        reason=reason,
    )


@dataclass(frozen=True, slots=True)
class FreshSeedSchedule:
    """Thirty deterministic unseen scene seeds and their complete exclusion evidence."""

    m3b_export_fingerprint: str
    namespace: str
    ordered_accepted_source_seeds: tuple[int, ...]
    ordered_rejected_source_seeds: tuple[int, ...]
    rejected_source_seeds_available: bool
    exclusion_digest: str
    ordered_fresh_seeds: tuple[int, ...]
    candidate_attempts: int
    schedule_digest: str
    schema_version: str = FRESH_SEED_SCHEDULE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _digest(self.m3b_export_fingerprint, "m3b_export_fingerprint")
        _digest(self.exclusion_digest, "exclusion_digest")
        _digest(self.schedule_digest, "schedule_digest")
        if self.schema_version != FRESH_SEED_SCHEDULE_SCHEMA_VERSION:
            raise EvaluationContractError("unknown fresh-seed schedule schema")
        if not isinstance(self.namespace, str) or not self.namespace:
            raise EvaluationContractError("fresh-seed namespace must be non-empty")
        if not isinstance(self.rejected_source_seeds_available, bool):
            raise TypeError("rejected_source_seeds_available must be a boolean")
        if isinstance(self.candidate_attempts, bool) or not isinstance(
            self.candidate_attempts, int
        ):
            raise TypeError("candidate_attempts must be an integer")
        for name, seeds in (
            ("accepted", self.ordered_accepted_source_seeds),
            ("rejected", self.ordered_rejected_source_seeds),
            ("fresh", self.ordered_fresh_seeds),
        ):
            if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
                raise TypeError(f"{name} scene seeds must be integers")
            if any(seed < 0 or seed > _MAX_SCENE_SEED for seed in seeds):
                raise EvaluationContractError(f"{name} scene seeds must be non-negative int31")
            if len(set(seeds)) != len(seeds):
                raise EvaluationContractError(f"{name} scene seeds must be unique")
        if self.ordered_accepted_source_seeds != tuple(sorted(self.ordered_accepted_source_seeds)):
            raise EvaluationContractError("accepted source seeds must use sorted canonical order")
        if self.ordered_rejected_source_seeds != tuple(sorted(self.ordered_rejected_source_seeds)):
            raise EvaluationContractError("rejected source seeds must use sorted canonical order")
        if not self.rejected_source_seeds_available and self.ordered_rejected_source_seeds:
            raise EvaluationContractError(
                "unavailable rejected seeds must be represented by an empty tuple"
            )
        if len(self.ordered_fresh_seeds) != FRESH_SEED_COUNT:
            raise EvaluationContractError("fresh benchmark requires exactly 30 scene seeds")
        excluded = set(self.ordered_accepted_source_seeds) | set(self.ordered_rejected_source_seeds)
        if excluded & set(self.ordered_fresh_seeds):
            raise EvaluationContractError("fresh scene seeds overlap source candidates")
        if self.candidate_attempts < FRESH_SEED_COUNT:
            raise EvaluationContractError("candidate_attempts cannot be less than fresh seed count")
        if self.exclusion_digest != self.compute_exclusion_digest():
            raise EvaluationContractError("fresh-seed exclusion digest is invalid")
        if self.schedule_digest != self.compute_schedule_digest():
            raise EvaluationContractError("fresh-seed schedule digest is invalid")

    def exclusion_payload(self) -> dict[str, object]:
        return {
            "ordered_accepted_source_seeds": list(self.ordered_accepted_source_seeds),
            "ordered_rejected_source_seeds": list(self.ordered_rejected_source_seeds),
            "rejected_source_seeds_available": self.rejected_source_seeds_available,
        }

    def compute_exclusion_digest(self) -> str:
        return _canonical_digest(self.exclusion_payload())

    def schedule_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "m3b_export_fingerprint": self.m3b_export_fingerprint,
            "namespace": self.namespace,
            "exclusion_digest": self.exclusion_digest,
            "ordered_fresh_seeds": list(self.ordered_fresh_seeds),
            "candidate_attempts": self.candidate_attempts,
        }

    def compute_schedule_digest(self) -> str:
        return _canonical_digest(self.schedule_payload())

    def to_dict(self) -> dict[str, object]:
        return {
            **self.schedule_payload(),
            **self.exclusion_payload(),
            "schedule_digest": self.schedule_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        if not isinstance(value, dict):
            raise TypeError("fresh-seed schedule JSON must be an object")
        expected = {
            "schema_version",
            "m3b_export_fingerprint",
            "namespace",
            "ordered_accepted_source_seeds",
            "ordered_rejected_source_seeds",
            "rejected_source_seeds_available",
            "exclusion_digest",
            "ordered_fresh_seeds",
            "candidate_attempts",
            "schedule_digest",
        }
        if set(value) != expected:
            raise EvaluationContractError("fresh-seed schedule JSON fields are invalid")
        for key in (
            "ordered_accepted_source_seeds",
            "ordered_rejected_source_seeds",
            "ordered_fresh_seeds",
        ):
            if not isinstance(value[key], list):
                raise TypeError(f"{key} must be a list")
        return cls(
            schema_version=cast(str, value["schema_version"]),
            m3b_export_fingerprint=cast(str, value["m3b_export_fingerprint"]),
            namespace=cast(str, value["namespace"]),
            ordered_accepted_source_seeds=tuple(value["ordered_accepted_source_seeds"]),
            ordered_rejected_source_seeds=tuple(value["ordered_rejected_source_seeds"]),
            rejected_source_seeds_available=cast(bool, value["rejected_source_seeds_available"]),
            exclusion_digest=cast(str, value["exclusion_digest"]),
            ordered_fresh_seeds=tuple(value["ordered_fresh_seeds"]),
            candidate_attempts=cast(int, value["candidate_attempts"]),
            schedule_digest=cast(str, value["schedule_digest"]),
        )


def build_fresh_seed_schedule(
    *,
    m3b_export_fingerprint: str,
    accepted_source_seeds: tuple[int, ...],
    rejected_source_seeds: tuple[int, ...] | None,
    namespace: str,
) -> FreshSeedSchedule:
    """Derive the fixed 30-scene benchmark without consulting model results."""

    _digest(m3b_export_fingerprint, "m3b_export_fingerprint")
    if not isinstance(namespace, str) or not namespace:
        raise EvaluationContractError("fresh-seed namespace must be non-empty")
    accepted = _canonical_seed_set(accepted_source_seeds, "accepted_source_seeds")
    rejected_available = rejected_source_seeds is not None
    rejected = (
        _canonical_seed_set(rejected_source_seeds, "rejected_source_seeds")
        if rejected_source_seeds is not None
        else ()
    )
    excluded = set(accepted) | set(rejected)
    fresh: list[int] = []
    fresh_set: set[int] = set()
    counter = 0
    while len(fresh) < FRESH_SEED_COUNT:
        digest = sha256_hex(
            {
                "schema_version": FRESH_SEED_SCHEDULE_SCHEMA_VERSION,
                "m3b_export_fingerprint": m3b_export_fingerprint,
                "namespace": namespace,
                "candidate_counter": counter,
            }
        )
        candidate = int(digest[:16], 16) % (_MAX_SCENE_SEED + 1)
        counter += 1
        if candidate in excluded or candidate in fresh_set:
            continue
        fresh.append(candidate)
        fresh_set.add(candidate)

    exclusion_payload = {
        "ordered_accepted_source_seeds": list(accepted),
        "ordered_rejected_source_seeds": list(rejected),
        "rejected_source_seeds_available": rejected_available,
    }
    exclusion_digest = _canonical_digest(exclusion_payload)
    schedule_payload = {
        "schema_version": FRESH_SEED_SCHEDULE_SCHEMA_VERSION,
        "m3b_export_fingerprint": m3b_export_fingerprint,
        "namespace": namespace,
        "exclusion_digest": exclusion_digest,
        "ordered_fresh_seeds": fresh,
        "candidate_attempts": counter,
    }
    return FreshSeedSchedule(
        m3b_export_fingerprint=m3b_export_fingerprint,
        namespace=namespace,
        ordered_accepted_source_seeds=accepted,
        ordered_rejected_source_seeds=rejected,
        rejected_source_seeds_available=rejected_available,
        exclusion_digest=exclusion_digest,
        ordered_fresh_seeds=tuple(fresh),
        candidate_attempts=counter,
        schedule_digest=_canonical_digest(schedule_payload),
    )


def _canonical_seed_set(seeds: object, name: str) -> tuple[int, ...]:
    if not isinstance(seeds, tuple):
        raise TypeError(f"{name} must be a tuple")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise TypeError(f"{name} must contain integers")
    if any(seed < 0 or seed > _MAX_SCENE_SEED for seed in seeds):
        raise EvaluationContractError(f"{name} must contain non-negative int31 values")
    return tuple(sorted(set(seeds)))


@dataclass(frozen=True, slots=True)
class WilsonInterval:
    successes: int
    trials: int
    confidence_level: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if isinstance(self.successes, bool) or not isinstance(self.successes, int):
            raise TypeError("successes must be an integer")
        if isinstance(self.trials, bool) or not isinstance(self.trials, int):
            raise TypeError("trials must be an integer")
        if self.trials < 1 or self.successes < 0 or self.successes > self.trials:
            raise EvaluationContractError("Wilson counts must satisfy 0 <= successes <= trials")
        if not 0.0 < self.confidence_level < 1.0:
            raise EvaluationContractError("confidence_level must be between zero and one")
        if not 0.0 <= self.lower <= self.upper <= 1.0:
            raise EvaluationContractError("Wilson interval bounds are invalid")


def wilson_interval(successes: int, trials: int, confidence_level: float = 0.95) -> WilsonInterval:
    """Compute a two-sided Wilson score interval without optional dependencies."""

    if isinstance(successes, bool) or not isinstance(successes, int):
        raise TypeError("successes must be an integer")
    if isinstance(trials, bool) or not isinstance(trials, int):
        raise TypeError("trials must be an integer")
    if trials < 1 or successes < 0 or successes > trials:
        raise EvaluationContractError("Wilson counts must satisfy 0 <= successes <= trials")
    if not 0.0 < confidence_level < 1.0:
        raise EvaluationContractError("confidence_level must be between zero and one")
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return WilsonInterval(
        successes=successes,
        trials=trials,
        confidence_level=confidence_level,
        lower=max(0.0, center - half_width),
        upper=min(1.0, center + half_width),
    )


def summarize_rollout_benchmark(
    episodes: tuple[RolloutEpisodeResult, ...],
    *,
    confidence_level: float = 0.95,
) -> RolloutBenchmarkResult:
    """Aggregate immutable episode evidence with identity and schedule checks."""

    if not isinstance(episodes, tuple) or not episodes:
        raise EvaluationContractError("benchmark requires at least one rollout episode")
    if not all(isinstance(item, RolloutEpisodeResult) for item in episodes):
        raise TypeError("benchmark episodes must be RolloutEpisodeResult values")
    identities = {
        (item.run_fingerprint, item.checkpoint_fingerprint, item.schedule_digest, item.split)
        for item in episodes
    }
    if len(identities) != 1:
        raise EvaluationContractError(
            "benchmark episodes must share run, checkpoint, schedule, and split"
        )
    evaluation_ids = [item.evaluation_id for item in episodes]
    if len(set(evaluation_ids)) != len(evaluation_ids):
        raise EvaluationContractError("rollout evaluation IDs must be unique")
    run_fingerprint, checkpoint_fingerprint, schedule_digest, split = identities.pop()
    successes = sum(item.success for item in episodes)
    interval = wilson_interval(successes, len(episodes), confidence_level)
    task_ids = sorted({item.task_id for item in episodes})
    per_task = {
        task_id: sum(item.success for item in episodes if item.task_id == task_id)
        / sum(item.task_id == task_id for item in episodes)
        for task_id in task_ids
    }
    scene_ids = sorted({item.scene_id for item in episodes})
    per_scene = {
        scene_id: sum(item.success for item in episodes if item.scene_id == scene_id)
        / sum(item.scene_id == scene_id for item in episodes)
        for scene_id in scene_ids
    }
    metric_rates = {
        "target_in_target_bin": sum(item.target_in_target_bin for item in episodes) / len(episodes),
        "target_in_wrong_bin": sum(item.target_in_wrong_bin for item in episodes) / len(episodes),
        "wrong_object_in_target_bin": sum(item.wrong_object_in_target_bin for item in episodes)
        / len(episodes),
        "target_grasped_any": sum(item.target_grasped_any for item in episodes) / len(episodes),
        "wrong_object_grasped_any": sum(item.wrong_object_grasped_any for item in episodes)
        / len(episodes),
        "target_off_table": sum(item.target_off_table for item in episodes) / len(episodes),
        "timeout": sum(item.timeout for item in episodes) / len(episodes),
        "invalid_action": sum(item.invalid_action for item in episodes) / len(episodes),
        "inference_failure": sum(item.inference_failure for item in episodes) / len(episodes),
    }

    def latency_summary(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"p50": None, "p95": None, "p99": None}
        array = np.asarray(values, dtype=np.float64)
        return {
            "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)),
            "p99": float(np.percentile(array, 99)),
        }

    failure_counts = {
        status.value: sum(item.status is status for item in episodes)
        for status in RolloutStatus
        if status is not RolloutStatus.SUCCESS and any(item.status is status for item in episodes)
    }
    benchmark_id = _canonical_digest(
        {
            "run_fingerprint": run_fingerprint,
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "schedule_digest": schedule_digest,
            "split": split.value,
            "ordered_evaluation_ids": evaluation_ids,
        }
    )
    return RolloutBenchmarkResult(
        benchmark_id=benchmark_id,
        run_fingerprint=run_fingerprint,
        checkpoint_fingerprint=checkpoint_fingerprint,
        schedule_digest=schedule_digest,
        split=split,
        episodes=episodes,
        success_rate=successes / len(episodes),
        success_wilson_low=interval.lower,
        success_wilson_high=interval.upper,
        per_task_success_rates=per_task,
        per_scene_group_success_rates=per_scene,
        metric_rates=metric_rates,
        failure_counts=failure_counts,
        inference_latency_ms=latency_summary(
            [value for item in episodes for value in item.inference_latency_ms]
        ),
        environment_step_latency_ms=latency_summary(
            [value for item in episodes for value in item.environment_step_latency_ms]
        ),
    )


def _validation_result_from_dict(value: object) -> ValidationResult:
    if not isinstance(value, dict):
        raise TypeError("validation result must be an object")
    expected = {
        "checkpoint_fingerprint",
        "checkpoint_step",
        "schedule_digest",
        "success_rate",
        "wrong_object_interaction_rate",
        "target_off_table_rate",
        "offline_validation_action_loss",
    }
    if set(value) != expected:
        raise EvaluationContractError("validation result fields are invalid")
    return ValidationResult(**value)


__all__ = [
    "CHECKPOINT_SELECTION_SCHEMA_VERSION",
    "FRESH_SEED_COUNT",
    "FRESH_SEED_SCHEDULE_SCHEMA_VERSION",
    "TEST_AUTHORIZATION_SCHEMA_VERSION",
    "CheckpointSelectionRecord",
    "EvaluationContractError",
    "FreshSeedSchedule",
    "SelectionLockError",
    "TestEvaluationAuthorization",
    "WilsonInterval",
    "authorize_test_evaluation",
    "build_fresh_seed_schedule",
    "create_checkpoint_selection",
    "load_checkpoint_selection",
    "rank_validation_results",
    "rollout_schedule_digest",
    "summarize_rollout_benchmark",
    "validation_ranking_key",
    "wilson_interval",
    "write_checkpoint_selection_atomic",
]
