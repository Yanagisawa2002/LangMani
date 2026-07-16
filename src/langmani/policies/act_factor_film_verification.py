"""Independent, read-only verification of real M4.3b target-development evidence.

The verifier never imports an evaluator entry point, constructs an environment,
loads model weights, or mutates training/evaluation artifacts.  It validates the
completed FactorFiLM training run, every checkpoint's immutable bytes, and the
separately promoted six-stage evaluation evidence bundle.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

from langmani.environments.pick_place_by_instruction import ENV_ID
from langmani.policies.act_checkpoint import validate_act_checkpoint_artifacts
from langmani.policies.act_factor_film_evaluation import (
    FACTOR_FILM_DEVELOPMENT_EPISODE_COUNT,
    FACTOR_FILM_DEVELOPMENT_POLICY_LABELS,
    FACTOR_FILM_VALIDATION_EPISODE_COUNT,
    DevelopmentEpisodeIdentity,
    DevelopmentPolicyMetrics,
    DevelopmentQualityGateResult,
    DevelopmentSemanticMetrics,
    FactorFiLMReloadValidationRecord,
    FactorFiLMValidationCountResult,
    PairedDevelopmentIdentityRecord,
    factor_film_validation_count_result,
    rank_factor_film_validation_count_results,
    validate_paired_development_identities,
    validate_reload_against_selection,
)
from langmani.policies.act_factor_film_evidence import (
    FactorFiLMEvaluationStage,
    validate_completed_factor_film_evaluation_evidence,
)
from langmani.policies.act_factor_film_training import (
    AUTHORIZED_M3B_FINGERPRINT,
    AUTHORIZED_M3B_SPLIT_FINGERPRINT,
    AUTHORIZED_M43_AUDIT_EVIDENCE_FINGERPRINT,
)
from langmani.policies.act_factor_film_types import (
    FACTOR_FILM_TARGET_MODEL_CONFIG_FINGERPRINT,
    FACTOR_FILM_TARGET_OPTIMIZATION_CONFIG_FINGERPRINT,
    FactorFiLMSelectionRecord,
    FactorFiLMTrainingManifest,
    FactorFiLMTrainingMode,
    FactorFiLMValidationQueue,
    canonical_fingerprint,
)
from langmani.policies.m42_schedule import M42_DEV_SCHEDULE_FINGERPRINT
from langmani.policies.m43_types import M43_CANONICAL_TASK_IDS

M43B_ARCHITECTURE_BASELINE_GIT_COMMIT = "8ee0f1babf36b91d1ee2a39701e4a6db6003b660"
M43B_TARGET_TRAINING_STEPS = 100_000
M43B_TARGET_CHECKPOINT_STEPS = tuple(range(5_000, 100_001, 5_000))
M43B_TARGET_FACTOR_FILM_PARAMETER_INCREASE = 67_744
M43B_TARGET_DEVELOPMENT_TOTAL_EPISODES = 216
M43B_INDEPENDENT_VERIFICATION_SCHEMA_VERSION = (
    "langmani-m43b-independent-target-development-verification-v0"
)


class FactorFiLMIndependentVerificationError(RuntimeError):
    """Raised when persisted target-development evidence is incomplete or inconsistent."""


@dataclass(slots=True)
class FactorFiLMIndependentVerificationReport:
    """Portable independent verification flags and compact provenance."""

    checks: list[dict[str, object]] = field(default_factory=list)
    implementation_validated: bool = False
    semantic_audit_completed: bool = False
    factor_film_implementation_validated: bool = False
    factor_film_fixture_training_validated: bool = False
    factor_film_training_completed: bool = False
    factor_film_checkpoints_complete: bool = False
    factor_film_checkpoint_selected: bool = False
    validation_only_selection_validated: bool = False
    factor_film_reload_validated: bool = False
    development_benchmark_completed: bool = False
    post_grasp_analysis_completed: bool = False
    first_interaction_analysis_completed: bool = False
    correct_task_retrieval_validated: bool = False
    object_retrieval_validated: bool = False
    bin_retrieval_validated: bool = False
    raw_action_metrics_validated: bool = False
    runtime_action_metrics_validated: bool = False
    development_quality_gate_passed: bool = False
    final_benchmark_authorized: bool = False
    final_schedule_accessed: bool = False
    test_split_accessed: bool = False
    fresh_seed_accessed: bool = False
    smolvla_go: bool = False
    physical_target_validated: bool = False
    run_fingerprint: str | None = None
    selected_checkpoint_fingerprint: str | None = None
    selected_checkpoint_step: int | None = None
    evaluation_evidence_fingerprint: str | None = None
    training_git_commit: str | None = None
    evaluation_git_commit: str | None = None
    quality_failed_criteria: tuple[str, ...] = ()
    schema_version: str = M43B_INDEPENDENT_VERIFICATION_SCHEMA_VERSION

    def check(self, name: str, condition: bool, detail: str) -> None:
        self.checks.append(
            {"name": name, "status": "pass" if condition else "fail", "detail": detail}
        )
        if not condition:
            raise FactorFiLMIndependentVerificationError(f"{name}: {detail}")

    @property
    def passed(self) -> bool:
        required_true = (
            self.implementation_validated,
            self.semantic_audit_completed,
            self.factor_film_implementation_validated,
            self.factor_film_fixture_training_validated,
            self.factor_film_training_completed,
            self.factor_film_checkpoints_complete,
            self.factor_film_checkpoint_selected,
            self.validation_only_selection_validated,
            self.factor_film_reload_validated,
            self.development_benchmark_completed,
            self.post_grasp_analysis_completed,
            self.first_interaction_analysis_completed,
            self.raw_action_metrics_validated,
            self.runtime_action_metrics_validated,
            self.physical_target_validated,
        )
        forbidden = (
            self.final_schedule_accessed,
            self.test_split_accessed,
            self.fresh_seed_accessed,
            self.smolvla_go,
        )
        return (
            all(required_true)
            and not any(forbidden)
            and self.final_benchmark_authorized == self.development_quality_gate_passed
            and not any(value["status"] == "fail" for value in self.checks)
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


@dataclass(frozen=True, slots=True)
class _ValidatedTraining:
    manifest: FactorFiLMTrainingManifest
    queue: FactorFiLMValidationQueue
    completion: Mapping[str, object]
    manifest_fingerprint: str
    completion_fingerprint: str


def _read_object(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise FactorFiLMIndependentVerificationError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise FactorFiLMIndependentVerificationError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise FactorFiLMIndependentVerificationError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise FactorFiLMIndependentVerificationError(f"{label} must be a JSON object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, list | tuple):
        raise FactorFiLMIndependentVerificationError(f"{label} must be a JSON array")
    return cast(Sequence[object], value)


def _first(payload: Mapping[str, object], *names: str) -> object:
    for name in names:
        if name in payload:
            return payload[name]
    raise FactorFiLMIndependentVerificationError(
        f"evidence is missing required field ({' or '.join(names)})"
    )


def _validate_metrics_file(run_root: Path, *, final_step: int) -> None:
    path = run_root / "metrics.jsonl"
    if path.is_symlink() or not path.is_file():
        raise FactorFiLMIndependentVerificationError("completed training lacks metrics.jsonl")
    previous_step = -1
    observed_final_step = -1
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            try:
                record = json.loads(
                    raw,
                    parse_constant=lambda token: (_ for _ in ()).throw(
                        ValueError(f"non-finite JSON constant {token}")
                    ),
                )
            except (json.JSONDecodeError, ValueError) as error:
                raise FactorFiLMIndependentVerificationError(
                    f"metrics.jsonl line {line_number} is invalid: {error}"
                ) from error
            if not isinstance(record, dict):
                raise FactorFiLMIndependentVerificationError("training metric must be an object")
            _validate_finite_metric_values(record, path=f"metrics line {line_number}")
            step = record.get("step")
            if isinstance(step, bool) or not isinstance(step, int) or step < previous_step:
                raise FactorFiLMIndependentVerificationError(
                    "training metric steps are malformed or not monotonic"
                )
            previous_step = step
            observed_final_step = max(observed_final_step, step)
            for name, value in record.items():
                lowered = name.lower()
                if (
                    value is not None
                    and any(token in lowered for token in ("loss", "gradient", "grad_norm"))
                    and (
                        isinstance(value, bool)
                        or not isinstance(value, int | float)
                        or not math.isfinite(float(value))
                    )
                ):
                    raise FactorFiLMIndependentVerificationError(
                        f"training metric {name} is non-finite at step {step}"
                    )
    if observed_final_step != final_step:
        raise FactorFiLMIndependentVerificationError(
            f"metrics.jsonl ends at {observed_final_step}, expected {final_step}"
        )


def _validate_finite_metric_values(value: object, *, path: str) -> None:
    if isinstance(value, Mapping):
        for name, item in value.items():
            _validate_finite_metric_values(item, path=f"{path}.{name}")
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, item in enumerate(value):
            _validate_finite_metric_values(item, path=f"{path}[{index}]")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise FactorFiLMIndependentVerificationError(f"training metric {path} is non-finite")


def _validate_training_run(
    run_root: Path,
    *,
    expected_training_git_commit: str,
) -> _ValidatedTraining:
    root = run_root.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise FactorFiLMIndependentVerificationError("training run root is missing or unsafe")
    manifest_payload = _read_object(root / "run_manifest.json", "training manifest")
    queue_payload = _read_object(root / "validation_queue.json", "validation queue")
    completion = _read_object(root / "complete.json", "training completion")
    manifest = FactorFiLMTrainingManifest.from_dict(manifest_payload)
    queue = FactorFiLMValidationQueue.from_dict(queue_payload)
    identity = manifest.identity

    if identity.git_commit != expected_training_git_commit or identity.git_dirty:
        raise FactorFiLMIndependentVerificationError(
            "training Git identity is not the explicitly authorized clean producer commit"
        )
    if identity.experiment_mode is not FactorFiLMTrainingMode.TARGET_DEVELOPMENT:
        raise FactorFiLMIndependentVerificationError("training run is not target_development")
    if identity.device != "cuda" or manifest.training_config.device != "cuda":
        raise FactorFiLMIndependentVerificationError("real target training did not declare CUDA")
    if (
        identity.m3b_export_fingerprint != AUTHORIZED_M3B_FINGERPRINT
        or identity.m3b_split_manifest_digest != AUTHORIZED_M3B_SPLIT_FINGERPRINT
        or identity.semantic_audit_evidence_fingerprint != AUTHORIZED_M43_AUDIT_EVIDENCE_FINGERPRINT
    ):
        raise FactorFiLMIndependentVerificationError("training authority fingerprints changed")
    if (
        canonical_fingerprint(identity.model_config.get("base_act"))
        != FACTOR_FILM_TARGET_MODEL_CONFIG_FINGERPRINT
        or canonical_fingerprint(identity.optimization_config)
        != FACTOR_FILM_TARGET_OPTIMIZATION_CONFIG_FINGERPRINT
    ):
        raise FactorFiLMIndependentVerificationError("model or optimization identity changed")
    if manifest.architecture_identity.parameter_count_increase != (
        M43B_TARGET_FACTOR_FILM_PARAMETER_INCREASE
    ):
        raise FactorFiLMIndependentVerificationError("FactorFiLM parameter increase is not 67,744")
    steps = tuple(value.global_step for value in manifest.checkpoints)
    if (
        not manifest.training_complete
        or manifest.training_state.global_step != M43B_TARGET_TRAINING_STEPS
        or steps != M43B_TARGET_CHECKPOINT_STEPS
    ):
        raise FactorFiLMIndependentVerificationError(
            "training is not the exact 100k/20-checkpoint run"
        )
    if queue.run_fingerprint != identity.run_fingerprint:
        raise FactorFiLMIndependentVerificationError("validation queue belongs to another run")
    queue_by_step = {value.checkpoint_step: value for value in queue.items}
    for checkpoint in manifest.checkpoints:
        validated = validate_act_checkpoint_artifacts(
            run_root=root,
            checkpoint_relative_path=checkpoint.relative_path,
            expected_identity=identity,
        )
        if validated.record != checkpoint:
            raise FactorFiLMIndependentVerificationError("checkpoint record differs from manifest")
        item = queue_by_step[checkpoint.global_step]
        if (
            item.checkpoint_fingerprint != checkpoint.checkpoint_fingerprint
            or item.checkpoint_relative_path != checkpoint.relative_path
        ):
            raise FactorFiLMIndependentVerificationError("queue checkpoint identity changed")
        metric = validated.training_metric
        if not isinstance(metric, Mapping):
            raise FactorFiLMIndependentVerificationError("checkpoint lacks its bound metric")
        loss = metric.get("validation_loss")
        if (
            isinstance(loss, bool)
            or not isinstance(loss, int | float)
            or not math.isfinite(float(loss))
            or float(loss) != float(item.offline_validation_action_loss)
        ):
            raise FactorFiLMIndependentVerificationError(
                "queued validation loss is not checkpoint-bound"
            )

    manifest_fingerprint = canonical_fingerprint(manifest.to_dict())
    completion_fingerprint = canonical_fingerprint(completion)
    required_completion = {
        "passed": True,
        "run_fingerprint": identity.run_fingerprint,
        "global_step": M43B_TARGET_TRAINING_STEPS,
        "checkpoint_count": 20,
        "manifest_fingerprint": manifest_fingerprint,
        "validation_queue_fingerprint": queue.queue_fingerprint,
        "factor_film_training_completed": True,
        "factor_film_checkpoint_selected": False,
        "development_benchmark_completed": False,
        "final_schedule_accessed": False,
        "smolvla_go": False,
    }
    if any(completion.get(name) != value for name, value in required_completion.items()):
        raise FactorFiLMIndependentVerificationError("training completion is inconsistent")
    _validate_metrics_file(root, final_step=M43B_TARGET_TRAINING_STEPS)
    return _ValidatedTraining(
        manifest=manifest,
        queue=queue,
        completion=completion,
        manifest_fingerprint=manifest_fingerprint,
        completion_fingerprint=completion_fingerprint,
    )


def _parse_structural_flags(path: Path) -> tuple[bool, bool]:
    payload = _read_object(path, "M4.3 structural verification")
    if payload.get("passed") is not True:
        raise FactorFiLMIndependentVerificationError("structural verification did not pass")
    if (
        payload.get("implementation_git_commit") != M43B_ARCHITECTURE_BASELINE_GIT_COMMIT
        or payload.get("implementation_git_dirty") is not False
    ):
        raise FactorFiLMIndependentVerificationError(
            "structural verification is not bound to the clean FactorFiLM architecture baseline"
        )
    return (
        payload.get("factor_film_implementation_validated") is True,
        payload.get("factor_film_fixture_training_validated") is True,
    )


def _validate_development_payload(
    payload: Mapping[str, object],
) -> tuple[PairedDevelopmentIdentityRecord, DevelopmentPolicyMetrics, int]:
    if payload.get("development_benchmark_completed") is not True:
        raise FactorFiLMIndependentVerificationError("development benchmark is incomplete")
    total = _first(payload, "total_episode_count", "episode_count")
    if total != M43B_TARGET_DEVELOPMENT_TOTAL_EPISODES:
        raise FactorFiLMIndependentVerificationError(
            "development evidence must contain 216 rollouts"
        )
    counts = _mapping(
        _first(payload, "per_policy_episode_counts", "policy_episode_counts"),
        "per-policy episode counts",
    )
    if dict(counts) != {
        label: FACTOR_FILM_DEVELOPMENT_EPISODE_COUNT
        for label in FACTOR_FILM_DEVELOPMENT_POLICY_LABELS
    }:
        raise FactorFiLMIndependentVerificationError(
            "development policies must each contain 72 rollouts"
        )
    paired = PairedDevelopmentIdentityRecord.from_dict(
        _mapping(
            _first(payload, "paired_identities", "paired_episode_identities"), "paired identities"
        )
    )
    metrics = DevelopmentPolicyMetrics.from_dict(
        _mapping(
            _first(payload, "factor_film_metrics", "development_metrics"), "FactorFiLM metrics"
        )
    )
    if payload.get("paired_episode_identities_validated") is not True:
        raise FactorFiLMIndependentVerificationError("paired development identity flag is false")
    physical_contract = {
        "physical_execution": True,
        "device": "cuda",
        "environment_id": ENV_ID,
        "schedule_id": "m42_dev_v0",
        "schedule_fingerprint": M42_DEV_SCHEDULE_FINGERPRINT,
        "execution_horizon": 10,
        "gripper_mode": "project",
        "m2_expert_invoked": False,
    }
    if any(payload.get(name) != expected for name, expected in physical_contract.items()):
        raise FactorFiLMIndependentVerificationError(
            "development evidence lacks the exact physical H10/project contract"
        )

    reports = _mapping(payload.get("benchmark_reports"), "development benchmark reports")
    per_task_reports = _sequence(reports.get("per_task"), "PerTask benchmark reports")
    if len(per_task_reports) != 6:
        raise FactorFiLMIndependentVerificationError("development requires six PerTask reports")
    per_task_identities: list[DevelopmentEpisodeIdentity] = []
    per_task_successes = 0
    for report in per_task_reports:
        parsed = _mapping(report, "PerTask benchmark")
        identities, aggregate = _validate_physical_benchmark(parsed, expected_episodes=12)
        per_task_identities.extend(identities)
        per_task_successes += int(cast(int, aggregate["successes"]))
    onehot_report = _mapping(reports.get("state_onehot"), "State-OneHot benchmark")
    factor_report = _mapping(reports.get("factor_film"), "FactorFiLM benchmark")
    onehot_identities, _ = _validate_physical_benchmark(onehot_report, expected_episodes=72)
    factor_identities, factor_aggregate = _validate_physical_benchmark(
        factor_report, expected_episodes=72
    )
    recomputed = validate_paired_development_identities(
        per_task=tuple(per_task_identities),
        state_onehot=onehot_identities,
        factor_film=factor_identities,
        schedule_fingerprint=M42_DEV_SCHEDULE_FINGERPRINT,
    )
    if recomputed.to_dict() != paired.to_dict():
        raise FactorFiLMIndependentVerificationError(
            "paired identities differ from the embedded physical reports"
        )
    recomputed_metrics = _development_metrics_from_benchmark(factor_report, factor_aggregate)
    if recomputed_metrics.to_dict() != metrics.to_dict():
        raise FactorFiLMIndependentVerificationError(
            "FactorFiLM metrics differ from the embedded physical report"
        )
    return paired, metrics, per_task_successes


def _validate_physical_benchmark(
    report: Mapping[str, object], *, expected_episodes: int
) -> tuple[tuple[DevelopmentEpisodeIdentity, ...], Mapping[str, object]]:
    if (
        report.get("schedule_id") != "m42_dev_v0"
        or report.get("schedule_fingerprint") != M42_DEV_SCHEDULE_FINGERPRINT
        or report.get("execution_horizon") != 10
        or report.get("gripper_mode") != "project"
    ):
        raise FactorFiLMIndependentVerificationError(
            "development benchmark changed schedule or runtime"
        )
    episodes = _sequence(report.get("episodes"), "development benchmark episodes")
    aggregate = _mapping(report.get("aggregate"), "development benchmark aggregate")
    if len(episodes) != expected_episodes or aggregate.get("episode_count") != expected_episodes:
        raise FactorFiLMIndependentVerificationError(
            "development benchmark episode count is incomplete"
        )
    identities: list[DevelopmentEpisodeIdentity] = []
    for item in episodes:
        episode = _mapping(item, "development episode")
        rollout = _mapping(episode.get("rollout"), "development rollout")
        if rollout.get("split") != "development":
            raise FactorFiLMIndependentVerificationError(
                "m42_dev_v0 rollout is mislabeled as a non-development source"
            )
        identities.append(
            DevelopmentEpisodeIdentity(
                episode_index=cast(int, episode["episode_index"]),
                scene_seed=cast(int, rollout["scene_seed"]),
                scene_id=cast(str, rollout["scene_id"]),
                task_id=cast(str, rollout["task_id"]),
            )
        )
    return tuple(identities), aggregate


def _development_metrics_from_benchmark(
    report: Mapping[str, object], aggregate: Mapping[str, object]
) -> DevelopmentPolicyMetrics:
    action = _mapping(aggregate.get("action_metrics"), "FactorFiLM action metrics")
    per_task = _mapping(aggregate.get("per_task"), "FactorFiLM per-task metrics")
    counts = {
        task_id: cast(int, _mapping(per_task.get(task_id), "per-task result")["successes"])
        for task_id in M43_CANONICAL_TASK_IDS
    }
    return DevelopmentPolicyMetrics(
        report_fingerprint=canonical_fingerprint(report),
        success_count=cast(int, aggregate["successes"]),
        per_task_success_counts=counts,
        wrong_object_grasp_count=cast(int, aggregate["wrong_object_grasp_count"]),
        wrong_object_in_target_bin_count=cast(int, aggregate["wrong_object_in_target_bin_count"]),
        target_in_wrong_bin_count=cast(int, aggregate["target_in_wrong_bin_count"]),
        target_off_table_count=cast(int, aggregate["target_off_table_count"]),
        timeout_count=cast(int, aggregate["timeout_count"]),
        arm_projection_count=cast(int, action["arm_projected_component_count"]),
        nan_count=cast(int, action["nan_count"]),
        inf_count=cast(int, action["inf_count"]),
        malformed_action_count=cast(int, action["malformed_action_count"]),
    )


def _validate_semantic_payload(
    payload: Mapping[str, object],
) -> DevelopmentSemanticMetrics:
    required_true = (
        "semantic_evaluation_completed",
        "post_grasp_analysis_completed",
        "first_interaction_analysis_completed",
        "raw_action_metrics_validated",
        "runtime_action_metrics_validated",
    )
    if any(payload.get(name) is not True for name in required_true):
        raise FactorFiLMIndependentVerificationError("semantic/action evidence is incomplete")
    validation_count = _first(
        payload,
        "validation_observation_count",
        "semantic_validation_observation_count",
    )
    development_count = _first(
        payload,
        "development_observation_count",
        "semantic_development_observation_count",
    )
    if (
        validation_count != FACTOR_FILM_VALIDATION_EPISODE_COUNT
        or development_count != FACTOR_FILM_DEVELOPMENT_EPISODE_COUNT
    ):
        raise FactorFiLMIndependentVerificationError("semantic evidence is not exact 36/72")
    first_count = _first(payload, "first_interaction_count", "first_interaction_episode_count")
    if first_count != FACTOR_FILM_DEVELOPMENT_EPISODE_COUNT:
        raise FactorFiLMIndependentVerificationError(
            "first-interaction evidence must cover 72 episodes"
        )
    confusions = _mapping(
        payload.get("first_interaction_confusions"), "first-interaction confusions"
    )
    expected_confusions = {
        "requested_object_to_first_grasped_object",
        "requested_task_to_nearest_per_task_chunk",
        "nearest_per_task_chunk_to_actual_first_grasp",
        "requested_bin_to_first_approached_bin",
        "requested_bin_to_object_entry_bin",
    }
    if set(confusions) != expected_confusions:
        raise FactorFiLMIndependentVerificationError(
            "first-interaction confusion set is incomplete"
        )
    timing = _mapping(payload.get("semantic_error_timing"), "semantic-error timing")
    expected_timing = {
        "visible_in_initial_chunk",
        "emerged_later",
        "no_semantic_error",
        "no_first_interaction",
    }
    if (
        set(timing) != expected_timing
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in timing.values()
        )
        or sum(cast(int, value) for value in timing.values()) != 72
    ):
        raise FactorFiLMIndependentVerificationError(
            "semantic-error timing must distinguish initial, later, absent, and no interaction"
        )
    return DevelopmentSemanticMetrics.from_dict(
        _mapping(
            _first(payload, "semantic_metrics", "development_semantic_metrics"), "semantic metrics"
        )
    )


def _validate_validation_benchmark_reports(
    payload: Mapping[str, object],
    results: Sequence[FactorFiLMValidationCountResult],
    *,
    schedule_digest: str,
) -> None:
    reports = _sequence(payload.get("benchmark_reports"), "validation benchmark reports")
    if len(reports) != 20 or len(results) != 20:
        raise FactorFiLMIndependentVerificationError(
            "validation evidence requires 20 physical benchmark reports"
        )
    for result, raw_report in zip(results, reports, strict=True):
        report = _mapping(raw_report, "validation benchmark")
        checkpoint = _mapping(report.get("checkpoint"), "validation checkpoint")
        aggregate = _mapping(report.get("aggregate"), "validation aggregate")
        episodes = _sequence(report.get("episodes"), "validation episodes")
        if (
            report.get("schedule_id") != "m3b_validation_v0"
            or report.get("schedule_fingerprint") != schedule_digest
            or report.get("execution_horizon") != 10
            or report.get("gripper_mode") != "project"
            or checkpoint.get("checkpoint_fingerprint") != result.checkpoint_fingerprint
            or len(episodes) != 36
            or aggregate.get("episode_count") != 36
        ):
            raise FactorFiLMIndependentVerificationError(
                "validation benchmark changed checkpoint, schedule, runtime, or count"
            )
        scene_tasks: dict[str, set[str]] = {}
        for raw_episode in episodes:
            episode = _mapping(raw_episode, "validation episode")
            rollout = _mapping(episode.get("rollout"), "validation rollout")
            if rollout.get("split") != "validation":
                raise FactorFiLMIndependentVerificationError(
                    "checkpoint selection contains a non-validation rollout"
                )
            scene_id = cast(str, rollout.get("scene_id"))
            task_id = cast(str, rollout.get("task_id"))
            if not isinstance(scene_id, str) or task_id not in M43_CANONICAL_TASK_IDS:
                raise FactorFiLMIndependentVerificationError(
                    "validation rollout semantic identity is malformed"
                )
            scene_tasks.setdefault(scene_id, set()).add(task_id)
        if len(scene_tasks) != 6 or any(
            task_ids != set(M43_CANONICAL_TASK_IDS) for task_ids in scene_tasks.values()
        ):
            raise FactorFiLMIndependentVerificationError(
                "validation benchmark lacks six complete counterfactual groups"
            )
        recomputed = factor_film_validation_count_result(
            checkpoint_fingerprint=result.checkpoint_fingerprint,
            checkpoint_step=result.checkpoint_step,
            schedule_digest=schedule_digest,
            aggregate=aggregate,
            offline_validation_action_loss=result.offline_validation_action_loss,
        )
        if recomputed.to_dict() != result.to_dict():
            raise FactorFiLMIndependentVerificationError(
                "validation count differs from its checksum-bound benchmark"
            )


def verify_factor_film_target_development(
    *,
    training_run_root: str | Path,
    evaluation_evidence_root: str | Path,
    expected_training_git_commit: str,
    structural_verification_path: str | Path,
) -> FactorFiLMIndependentVerificationReport:
    """Validate completed real target-development evidence without executing it."""
    if re.fullmatch(r"[0-9a-f]{40}", expected_training_git_commit) is None:
        raise FactorFiLMIndependentVerificationError(
            "expected training Git commit must be full lowercase Git hex"
        )
    report = FactorFiLMIndependentVerificationReport()
    training_root = Path(training_run_root).resolve(strict=True)
    training = _validate_training_run(
        training_root,
        expected_training_git_commit=expected_training_git_commit,
    )
    report.check(
        "training contract",
        True,
        "explicit clean producer Git, CUDA, exact M3B and 100k steps",
    )
    report.factor_film_training_completed = True
    report.factor_film_checkpoints_complete = True
    report.run_fingerprint = training.manifest.identity.run_fingerprint
    report.training_git_commit = training.manifest.identity.git_commit

    completed = validate_completed_factor_film_evaluation_evidence(
        evaluation_evidence_root,
        training_run_root=training_root,
    )
    config = completed.config
    if (
        config.run_fingerprint != training.manifest.identity.run_fingerprint
        or config.training_manifest_fingerprint != training.manifest_fingerprint
        or config.training_completion_fingerprint != training.completion_fingerprint
        or config.validation_queue_fingerprint != training.queue.queue_fingerprint
        or config.dataset_fingerprint != AUTHORIZED_M3B_FINGERPRINT
        or config.split_fingerprint != AUTHORIZED_M3B_SPLIT_FINGERPRINT
        or config.semantic_audit_evidence_fingerprint != AUTHORIZED_M43_AUDIT_EVIDENCE_FINGERPRINT
        or config.development_schedule_fingerprint != M42_DEV_SCHEDULE_FINGERPRINT
        or config.training_git_commit != expected_training_git_commit
        or config.selected_execution_horizon != 10
        or config.action_bound_mode != "project"
    ):
        raise FactorFiLMIndependentVerificationError(
            "evaluation config differs from training authorities"
        )
    report.check("evidence lifecycle", True, "six-stage DAG, checksums and completion are valid")

    stages = completed.stage_artifacts
    validation_payload = stages[FactorFiLMEvaluationStage.VALIDATION].payload
    validation_results = tuple(
        FactorFiLMValidationCountResult.from_dict(_mapping(value, "validation result"))
        for value in _sequence(validation_payload["results"], "validation results")
    )
    _validate_validation_benchmark_reports(
        validation_payload,
        validation_results,
        schedule_digest=config.validation_schedule_digest,
    )
    ranked = rank_factor_film_validation_count_results(validation_results)
    if any(value.episode_count != 36 for value in validation_results):
        raise FactorFiLMIndependentVerificationError(
            "a validation checkpoint lacks all 36 episodes"
        )
    selection = FactorFiLMSelectionRecord.from_dict(
        stages[FactorFiLMEvaluationStage.SELECTION].payload
    )
    if (
        selection.selected_checkpoint_fingerprint != ranked[0].checkpoint_fingerprint
        or selection.selected_checkpoint_step != ranked[0].checkpoint_step
    ):
        raise FactorFiLMIndependentVerificationError(
            "selection does not use the seven-level winner"
        )
    report.check(
        "validation selection", True, "20 checkpoints x 36 episodes and seven-level ranking"
    )
    report.factor_film_checkpoint_selected = True
    report.validation_only_selection_validated = True
    report.selected_checkpoint_fingerprint = selection.selected_checkpoint_fingerprint
    report.selected_checkpoint_step = selection.selected_checkpoint_step

    reload_record = FactorFiLMReloadValidationRecord.from_dict(
        stages[FactorFiLMEvaluationStage.RELOAD].payload
    )
    validate_reload_against_selection(reload_record, selection)
    report.check(
        "fresh reload", True, "selected checkpoint and processors reproduce fixed inference"
    )
    report.factor_film_reload_validated = True

    paired, development_metrics, per_task_reference_successes = _validate_development_payload(
        stages[FactorFiLMEvaluationStage.DEVELOPMENT].payload
    )
    if paired.schedule_fingerprint != config.development_schedule_fingerprint:
        raise FactorFiLMIndependentVerificationError(
            "paired identities use another development schedule"
        )
    report.check("development benchmark", True, "three paired policies x 72 m42_dev_v0 episodes")
    report.development_benchmark_completed = True

    semantic_payload = stages[FactorFiLMEvaluationStage.SEMANTIC].payload
    semantic_metrics = _validate_semantic_payload(semantic_payload)
    report.post_grasp_analysis_completed = True
    report.first_interaction_analysis_completed = True
    report.raw_action_metrics_validated = True
    report.runtime_action_metrics_validated = True
    report.correct_task_retrieval_validated = cast(
        bool, semantic_payload["correct_task_retrieval_validated"]
    )
    report.object_retrieval_validated = cast(bool, semantic_payload["object_retrieval_validated"])
    report.bin_retrieval_validated = cast(bool, semantic_payload["bin_retrieval_validated"])
    report.check("semantic evidence", True, "paired 36/72 retrieval and 72 first interactions")

    gate_payload = stages[FactorFiLMEvaluationStage.QUALITY_GATE].payload
    gate = DevelopmentQualityGateResult.from_dict(
        _mapping(_first(gate_payload, "quality_gate", "development_quality_gate"), "quality gate")
    )
    if (
        gate.factor_film_metrics.to_dict() != development_metrics.to_dict()
        or gate.semantic_metrics.to_dict() != semantic_metrics.to_dict()
        or gate.per_task_reference_success_count != per_task_reference_successes
        or gate_payload.get("development_quality_gate_passed")
        is not gate.development_quality_gate_passed
        or gate_payload.get("final_benchmark_authorized") is not gate.final_benchmark_authorized
        or len(gate.checks) != 16
    ):
        raise FactorFiLMIndependentVerificationError("quality-gate evidence is not exact or bound")
    report.development_quality_gate_passed = gate.development_quality_gate_passed
    report.final_benchmark_authorized = gate.final_benchmark_authorized
    report.quality_failed_criteria = gate.failed_criteria
    report.check("quality gate", True, "all 16 criteria were computed exactly; quality is separate")

    implementation, fixture = _parse_structural_flags(Path(structural_verification_path))
    report.implementation_validated = True
    report.semantic_audit_completed = True
    report.factor_film_implementation_validated = implementation
    report.factor_film_fixture_training_validated = fixture
    report.physical_target_validated = True
    report.evaluation_evidence_fingerprint = completed.evidence_fingerprint
    report.evaluation_git_commit = config.evaluation_git_commit
    report.check(
        "sealed sources", True, "test, fresh seed, final schedule and SmolVLA remain untouched"
    )
    if not report.passed:
        raise FactorFiLMIndependentVerificationError(
            "independent verification flags are inconsistent"
        )
    return report


__all__ = [
    "M43B_INDEPENDENT_VERIFICATION_SCHEMA_VERSION",
    "M43B_ARCHITECTURE_BASELINE_GIT_COMMIT",
    "FactorFiLMIndependentVerificationError",
    "FactorFiLMIndependentVerificationReport",
    "verify_factor_film_target_development",
]
