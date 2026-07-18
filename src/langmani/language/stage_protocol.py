"""Compute-bounded, provenance-owned promotion contracts for LangMani M5A."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from langmani.datasets.identity import sha256_hex

M5A_STAGE_PROTOCOL_SCHEMA = "langmani-m5a-stage-protocol-v0"
M5A_CLASSIFIER_TRAINING_SEED = 0
M5A_CLASSIFIER_TRAINING_SEEDS = 1
M5A_MAXIMUM_EPOCHS = 5
M5A_EARLY_STOPPING_PATIENCE = 1


class M5AStageProtocolError(ValueError):
    """Raised when a staged M5A identity or promotion is inconsistent."""


class M5AStage(StrEnum):
    IMPLEMENTATION = "implementation"
    CLASSIFIER_FIXTURE = "classifier_fixture"
    CLASSIFIER_TINY_OVERFIT = "classifier_tiny_overfit"
    CLASSIFIER_PILOT = "classifier_pilot"
    CLASSIFIER_TRAINING = "classifier_training"
    LANGUAGE_DEVELOPMENT = "language_development"
    ONE_SCENE_CONTROL_SMOKE = "one_scene_control_smoke"
    THREE_SCENE_CONTROL_SCREEN = "three_scene_control_screen"
    FULL_CONTROL_DEVELOPMENT = "full_control_development"
    SEALED_FINAL = "sealed_final"


class CandidatePromotionState(StrEnum):
    DECLARED = "declared"
    FIXTURE_PASSED = "fixture_passed"
    TINY_OVERFIT_PASSED = "tiny_overfit_passed"
    PILOT_PASSED = "pilot_passed"
    TRAINING_PASSED = "training_passed"
    LANGUAGE_PROMOTED = "language_promoted"
    ONE_SCENE_PROMOTED = "one_scene_promoted"
    THREE_SCENE_PROMOTED = "three_scene_promoted"
    SELECTED_FOR_FULL_DEVELOPMENT = "selected_for_full_development"
    FINAL_LOCKED = "final_locked"
    REJECTED = "rejected"


_STATE_ORDER = {
    state: index
    for index, state in enumerate(
        (
            CandidatePromotionState.DECLARED,
            CandidatePromotionState.FIXTURE_PASSED,
            CandidatePromotionState.TINY_OVERFIT_PASSED,
            CandidatePromotionState.PILOT_PASSED,
            CandidatePromotionState.TRAINING_PASSED,
            CandidatePromotionState.LANGUAGE_PROMOTED,
            CandidatePromotionState.ONE_SCENE_PROMOTED,
            CandidatePromotionState.THREE_SCENE_PROMOTED,
            CandidatePromotionState.SELECTED_FOR_FULL_DEVELOPMENT,
            CandidatePromotionState.FINAL_LOCKED,
        )
    )
}


def _probability(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise M5AStageProtocolError(f"{name} must be finite and in [0, 1]")
    return float(value)


def _fingerprint(payload: object) -> str:
    return f"sha256:{sha256_hex(payload)}"


@dataclass(frozen=True, slots=True)
class ClassifierStageMetrics:
    """Validation/tiny metrics shared by the explicit classifier gates."""

    full_task_spec_accuracy: float
    object_accuracy: float
    bin_accuracy: float
    false_route_rate: float
    schema_valid_rate: float
    ambiguous_rejection_recall: float
    unsupported_rejection_recall: float
    malformed_rejection_recall: float
    status_accuracy: float = 0.0
    finite: bool = True

    def __post_init__(self) -> None:
        for name in (
            "full_task_spec_accuracy",
            "object_accuracy",
            "bin_accuracy",
            "false_route_rate",
            "schema_valid_rate",
            "ambiguous_rejection_recall",
            "unsupported_rejection_recall",
            "malformed_rejection_recall",
            "status_accuracy",
        ):
            object.__setattr__(self, name, _probability(getattr(self, name), name=name))
        if not isinstance(self.finite, bool):
            raise M5AStageProtocolError("finite must be a boolean")

    @property
    def all_rejection_classes_have_nonzero_recall(self) -> bool:
        return all(
            value > 0.0
            for value in (
                self.ambiguous_rejection_recall,
                self.unsupported_rejection_recall,
                self.malformed_rejection_recall,
            )
        )

    @property
    def pilot_gate_passed(self) -> bool:
        return (
            self.finite
            and self.full_task_spec_accuracy >= 0.85
            and self.object_accuracy >= 0.90
            and self.bin_accuracy >= 0.90
            and self.false_route_rate <= 0.10
            and self.schema_valid_rate == 1.0
            and self.all_rejection_classes_have_nonzero_recall
        )

    @property
    def training_gate_passed(self) -> bool:
        return (
            self.finite
            and self.full_task_spec_accuracy >= 0.95
            and self.object_accuracy >= 0.97
            and self.bin_accuracy >= 0.97
            and self.false_route_rate <= 0.03
            and self.ambiguous_rejection_recall >= 0.90
            and self.unsupported_rejection_recall >= 0.95
            and self.malformed_rejection_recall >= 0.95
            and self.schema_valid_rate == 1.0
        )

    @property
    def tiny_overfit_gate_passed(self) -> bool:
        return (
            self.finite
            and self.status_accuracy >= 0.98
            and self.full_task_spec_accuracy >= 0.98
            and self.false_route_rate == 0.0
            and self.schema_valid_rate == 1.0
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "full_task_spec_accuracy": self.full_task_spec_accuracy,
            "object_accuracy": self.object_accuracy,
            "bin_accuracy": self.bin_accuracy,
            "false_route_rate": self.false_route_rate,
            "schema_valid_rate": self.schema_valid_rate,
            "ambiguous_rejection_recall": self.ambiguous_rejection_recall,
            "unsupported_rejection_recall": self.unsupported_rejection_recall,
            "malformed_rejection_recall": self.malformed_rejection_recall,
            "status_accuracy": self.status_accuracy,
            "all_rejection_classes_have_nonzero_recall": (
                self.all_rejection_classes_have_nonzero_recall
            ),
            "finite": self.finite,
            "tiny_overfit_gate_passed": self.tiny_overfit_gate_passed,
            "pilot_gate_passed": self.pilot_gate_passed,
            "training_gate_passed": self.training_gate_passed,
        }


@dataclass(frozen=True, slots=True)
class ClassifierStageTrainingConfig:
    """One-seed epoch-bounded training identity shared by pilot and resume."""

    train_example_count: int
    batch_size: int = 32
    maximum_epochs: int = M5A_MAXIMUM_EPOCHS
    original_maximum_steps: int = 600
    seed: int = M5A_CLASSIFIER_TRAINING_SEED
    early_stopping_patience: int = M5A_EARLY_STOPPING_PATIENCE
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    maximum_gradient_norm: float = 1.0
    scheduler: str = "constant"
    schema_version: str = M5A_STAGE_PROTOCOL_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "train_example_count",
            "batch_size",
            "maximum_epochs",
            "original_maximum_steps",
            "early_stopping_patience",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise M5AStageProtocolError(f"{name} must be a positive integer")
        if self.seed != M5A_CLASSIFIER_TRAINING_SEED:
            raise M5AStageProtocolError("M5A classifier development permits exactly seed 0")
        if self.maximum_epochs > M5A_MAXIMUM_EPOCHS:
            raise M5AStageProtocolError("M5A classifier development is capped at five epochs")
        if self.early_stopping_patience != M5A_EARLY_STOPPING_PATIENCE:
            raise M5AStageProtocolError("M5A validation patience is fixed at one interval")
        if self.scheduler != "constant":
            raise M5AStageProtocolError("M5A staged classifier uses the constant scheduler")
        for name in ("learning_rate", "maximum_gradient_norm"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise M5AStageProtocolError(f"{name} must be finite and positive")
        if not math.isfinite(float(self.weight_decay)) or self.weight_decay < 0.0:
            raise M5AStageProtocolError("weight_decay must be finite and non-negative")
        if self.schema_version != M5A_STAGE_PROTOCOL_SCHEMA:
            raise M5AStageProtocolError("unknown M5A stage protocol schema")

    @property
    def steps_per_epoch(self) -> int:
        return math.ceil(self.train_example_count / self.batch_size)

    @property
    def maximum_steps(self) -> int:
        return min(self.original_maximum_steps, self.maximum_epochs * self.steps_per_epoch)

    @property
    def validation_steps(self) -> tuple[int, ...]:
        steps = {
            *range(self.steps_per_epoch, self.maximum_steps + 1, self.steps_per_epoch),
            self.pilot_step,
            self.maximum_steps,
        }
        return tuple(sorted(steps))

    @property
    def pilot_step(self) -> int:
        twenty_percent = max(1, math.ceil(self.maximum_steps * 0.20))
        return min(self.steps_per_epoch, twenty_percent)

    def to_dict(self) -> dict[str, object]:
        return {
            "train_example_count": self.train_example_count,
            "batch_size": self.batch_size,
            "maximum_epochs": self.maximum_epochs,
            "original_maximum_steps": self.original_maximum_steps,
            "maximum_steps": self.maximum_steps,
            "steps_per_epoch": self.steps_per_epoch,
            "validation_steps": list(self.validation_steps),
            "pilot_step": self.pilot_step,
            "seed": self.seed,
            "classifier_training_seeds": M5A_CLASSIFIER_TRAINING_SEEDS,
            "robustness_across_training_seeds_not_evaluated": True,
            "early_stopping_patience": self.early_stopping_patience,
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "maximum_gradient_norm": float(self.maximum_gradient_norm),
            "optimizer": "adamw",
            "scheduler": self.scheduler,
            "schema_version": self.schema_version,
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())


@dataclass(frozen=True, slots=True)
class CheckpointRetentionPolicy:
    """Fixed-name recovery slots prevent checkpoint growth across five epochs."""

    retained_roles: tuple[str, ...] = ("pilot", "latest", "validation_best")
    maximum_retained_checkpoints: int = 3
    schema_version: str = M5A_STAGE_PROTOCOL_SCHEMA

    def __post_init__(self) -> None:
        if self.retained_roles != ("pilot", "latest", "validation_best"):
            raise M5AStageProtocolError("checkpoint roles are fixed for M5A development")
        if self.maximum_retained_checkpoints != len(self.retained_roles):
            raise M5AStageProtocolError("checkpoint retention must match the fixed role slots")

    def to_dict(self) -> dict[str, object]:
        return {
            "retained_roles": list(self.retained_roles),
            "maximum_retained_checkpoints": self.maximum_retained_checkpoints,
            "overwrite_policy": "atomic_replace_same_role_only",
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class CandidatePromotionRecord:
    """Immutable candidate state transition bound to its evidence fingerprint."""

    candidate_id: str
    previous_state: CandidatePromotionState
    state: CandidatePromotionState
    gate_passed: bool
    evidence_fingerprint: str
    reason: str
    record_fingerprint: str = ""
    schema_version: str = M5A_STAGE_PROTOCOL_SCHEMA

    def __post_init__(self) -> None:
        if self.candidate_id not in {"classifier", "llm"}:
            raise M5AStageProtocolError("only classifier and llm are learned M5A candidates")
        if not isinstance(self.previous_state, CandidatePromotionState) or not isinstance(
            self.state, CandidatePromotionState
        ):
            raise M5AStageProtocolError("promotion states must be CandidatePromotionState values")
        if not isinstance(self.gate_passed, bool) or not self.reason.strip():
            raise M5AStageProtocolError("promotion decisions require a boolean gate and reason")
        digest = self.evidence_fingerprint.removeprefix("sha256:")
        if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
            raise M5AStageProtocolError("promotion evidence requires a full SHA-256")
        if self.gate_passed:
            if self.state is CandidatePromotionState.REJECTED:
                raise M5AStageProtocolError("a passed gate cannot reject the candidate")
            previous = _STATE_ORDER.get(self.previous_state)
            current = _STATE_ORDER.get(self.state)
            if previous is None or current is None or current != previous + 1:
                raise M5AStageProtocolError("promotion must advance exactly one declared state")
        elif self.state is not CandidatePromotionState.REJECTED:
            raise M5AStageProtocolError("a failed gate must stop the candidate as rejected")
        expected = _fingerprint(self.identity_dict())
        if self.record_fingerprint and self.record_fingerprint != expected:
            raise M5AStageProtocolError("promotion record fingerprint differs")
        object.__setattr__(self, "record_fingerprint", expected)

    def identity_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "previous_state": self.previous_state.value,
            "state": self.state.value,
            "gate_passed": self.gate_passed,
            "evidence_fingerprint": self.evidence_fingerprint,
            "reason": self.reason,
            "schema_version": self.schema_version,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.identity_dict(), "record_fingerprint": self.record_fingerprint}


@dataclass(frozen=True, slots=True)
class PhysicalStageBudget:
    stage: M5AStage
    scene_count: int
    oracle_episodes: int
    maximum_learned_candidates: int
    learned_episodes_per_candidate: int
    development_scene_offset: int
    final: bool = False

    def __post_init__(self) -> None:
        if self.stage not in {
            M5AStage.ONE_SCENE_CONTROL_SMOKE,
            M5AStage.THREE_SCENE_CONTROL_SCREEN,
            M5AStage.FULL_CONTROL_DEVELOPMENT,
            M5AStage.SEALED_FINAL,
        }:
            raise M5AStageProtocolError("physical budget uses a non-physical stage")
        if self.scene_count * 6 != self.oracle_episodes:
            raise M5AStageProtocolError("Oracle must cover all six tasks in every scene")
        if self.learned_episodes_per_candidate != self.oracle_episodes:
            raise M5AStageProtocolError(
                "each promoted candidate must use the Oracle scene/task view"
            )
        if self.final is not (self.stage is M5AStage.SEALED_FINAL):
            raise M5AStageProtocolError("only the sealed-final budget may be final")

    @property
    def maximum_episode_count(self) -> int:
        return self.oracle_episodes + (
            self.maximum_learned_candidates * self.learned_episodes_per_candidate
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage.value,
            "scene_count": self.scene_count,
            "oracle_episodes": self.oracle_episodes,
            "maximum_learned_candidates": self.maximum_learned_candidates,
            "learned_episodes_per_candidate": self.learned_episodes_per_candidate,
            "maximum_episode_count": self.maximum_episode_count,
            "development_scene_offset": self.development_scene_offset,
            "final": self.final,
        }


M5A_PHYSICAL_STAGE_BUDGETS: Mapping[M5AStage, PhysicalStageBudget] = MappingProxyType(
    {
        M5AStage.ONE_SCENE_CONTROL_SMOKE: PhysicalStageBudget(
            stage=M5AStage.ONE_SCENE_CONTROL_SMOKE,
            scene_count=1,
            oracle_episodes=6,
            maximum_learned_candidates=2,
            learned_episodes_per_candidate=6,
            development_scene_offset=0,
        ),
        M5AStage.THREE_SCENE_CONTROL_SCREEN: PhysicalStageBudget(
            stage=M5AStage.THREE_SCENE_CONTROL_SCREEN,
            scene_count=3,
            oracle_episodes=18,
            maximum_learned_candidates=2,
            learned_episodes_per_candidate=18,
            development_scene_offset=1,
        ),
        M5AStage.FULL_CONTROL_DEVELOPMENT: PhysicalStageBudget(
            stage=M5AStage.FULL_CONTROL_DEVELOPMENT,
            scene_count=6,
            oracle_episodes=36,
            maximum_learned_candidates=1,
            learned_episodes_per_candidate=36,
            development_scene_offset=4,
        ),
        M5AStage.SEALED_FINAL: PhysicalStageBudget(
            stage=M5AStage.SEALED_FINAL,
            scene_count=12,
            oracle_episodes=72,
            maximum_learned_candidates=1,
            learned_episodes_per_candidate=72,
            development_scene_offset=0,
            final=True,
        ),
    }
)


def maximum_pre_final_physical_episodes() -> int:
    return sum(
        M5A_PHYSICAL_STAGE_BUDGETS[stage].maximum_episode_count
        for stage in (
            M5AStage.ONE_SCENE_CONTROL_SMOKE,
            M5AStage.THREE_SCENE_CONTROL_SCREEN,
            M5AStage.FULL_CONTROL_DEVELOPMENT,
        )
    )


def promoted_candidates_for_stage(
    records: Sequence[CandidatePromotionRecord], *, stage: M5AStage
) -> tuple[str, ...]:
    """Return only candidates whose immutable state authorizes the requested stage."""

    latest: dict[str, CandidatePromotionRecord] = {}
    for record in records:
        previous = latest.get(record.candidate_id)
        if previous is not None and record.previous_state is not previous.state:
            raise M5AStageProtocolError("candidate promotion chain is discontinuous")
        latest[record.candidate_id] = record
    minimum = {
        M5AStage.ONE_SCENE_CONTROL_SMOKE: CandidatePromotionState.LANGUAGE_PROMOTED,
        M5AStage.THREE_SCENE_CONTROL_SCREEN: CandidatePromotionState.ONE_SCENE_PROMOTED,
        M5AStage.FULL_CONTROL_DEVELOPMENT: (CandidatePromotionState.SELECTED_FOR_FULL_DEVELOPMENT),
        M5AStage.SEALED_FINAL: CandidatePromotionState.FINAL_LOCKED,
    }.get(stage)
    if minimum is None:
        raise M5AStageProtocolError("requested stage is not a physical promotion stage")
    result = tuple(
        candidate
        for candidate in ("classifier", "llm")
        if (record := latest.get(candidate)) is not None
        and record.state is not CandidatePromotionState.REJECTED
        and _STATE_ORDER[record.state] >= _STATE_ORDER[minimum]
    )
    maximum = M5A_PHYSICAL_STAGE_BUDGETS[stage].maximum_learned_candidates
    if len(result) > maximum:
        raise M5AStageProtocolError("too many learned candidates were promoted to the stage")
    if stage in {M5AStage.FULL_CONTROL_DEVELOPMENT, M5AStage.SEALED_FINAL} and len(result) != 1:
        raise M5AStageProtocolError("expanded development/final requires exactly one locked router")
    return result


def stage_protocol_manifest() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": M5A_STAGE_PROTOCOL_SCHEMA,
        "classifier_training_seeds": M5A_CLASSIFIER_TRAINING_SEEDS,
        "classifier_training_seed": M5A_CLASSIFIER_TRAINING_SEED,
        "robustness_across_training_seeds_not_evaluated": True,
        "maximum_epochs": M5A_MAXIMUM_EPOCHS,
        "early_stopping_patience": M5A_EARLY_STOPPING_PATIENCE,
        "checkpoint_retention": CheckpointRetentionPolicy().to_dict(),
        "physical_stages": {
            stage.value: budget.to_dict() for stage, budget in M5A_PHYSICAL_STAGE_BUDGETS.items()
        },
        "maximum_pre_final_physical_episodes": maximum_pre_final_physical_episodes(),
        "final_automatically_executed": False,
    }
    return {**payload, "protocol_fingerprint": _fingerprint(payload)}


__all__ = [
    "CandidatePromotionRecord",
    "CandidatePromotionState",
    "CheckpointRetentionPolicy",
    "ClassifierStageMetrics",
    "ClassifierStageTrainingConfig",
    "M5A_CLASSIFIER_TRAINING_SEED",
    "M5A_CLASSIFIER_TRAINING_SEEDS",
    "M5A_EARLY_STOPPING_PATIENCE",
    "M5A_MAXIMUM_EPOCHS",
    "M5A_PHYSICAL_STAGE_BUDGETS",
    "M5A_STAGE_PROTOCOL_SCHEMA",
    "M5AStage",
    "M5AStageProtocolError",
    "PhysicalStageBudget",
    "maximum_pre_final_physical_episodes",
    "promoted_candidates_for_stage",
    "stage_protocol_manifest",
]
