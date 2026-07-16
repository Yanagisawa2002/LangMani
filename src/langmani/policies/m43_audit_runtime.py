"""Runtime-neutral execution core for the M4.3a semantic audit.

The module deliberately separates three concerns:

* a fixed physical policy input (one RGB image and one 9-D Panda state),
* deterministic post-policy-processor action-chunk inference, and
* semantic retrieval against the six frozen PerTask references.

It does not step an environment, select a checkpoint, materialize a schedule,
or write evidence.  The command layer supplies validated observations and
checkpoint contexts, while :mod:`langmani.policies.m43_evidence` owns the
immutable artifact lifecycle.
"""

from __future__ import annotations

import hashlib
import math
from argparse import Namespace
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from langmani.datasets.lerobot_types import IMAGE_FEATURE_KEY, STATE_FEATURE_KEY
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import BIN_IDS, OBJECT_IDS, stable_task_id
from langmani.policies.act_conditioning import append_canonical_task_onehot
from langmani.policies.act_semantic_audit import (
    build_semantic_alignment_audit,
    classify_semantic_conclusions,
    compute_action_chunk_distance,
    retrieve_bin,
    retrieve_object,
    retrieve_task,
    summarize_bin_retrieval,
    summarize_object_retrieval,
    summarize_task_retrieval,
)
from langmani.policies.m42_evaluation import M42CheckpointContext, M42PolicyKind
from langmani.policies.m43_types import (
    M43_ACTION_CHUNK_SIZE,
    M43_ACTION_COMPONENTS,
    M43_CANONICAL_TASK_IDS,
    ActionChunkDistanceConfig,
    BinRetrievalResult,
    ObjectRetrievalResult,
    SemanticAlignmentAudit,
    SemanticRetrievalResult,
)

M43_AUDIT_RUNTIME_VERSION = "LangManiSemanticAuditRuntimeV0"
M43_POLICY_LABELS = ("state_onehot", "task_token")
M43_REQUESTED_SOURCES = {
    "validation": ("m3b_validation",),
    "m42_dev_v0": ("m42_dev_v0",),
    "combined": ("m3b_validation", "m42_dev_v0"),
}
_TASK_SPEC_BY_ID = {stable_task_id(value): value for value in CANONICAL_TASK_SPECS}
_TASK_ID_BY_SEMANTICS = {
    (value.target_object_id, value.target_bin_id): stable_task_id(value)
    for value in CANONICAL_TASK_SPECS
}


class M43AuditRuntimeError(RuntimeError):
    """Raised when fixed-input inference is not a controlled intervention."""


def plan_semantic_audit(args: Namespace) -> dict[str, object]:
    """Return a truthful structural plan without loading data, models, or schedules."""

    mode = getattr(args, "mode", None)
    if mode not in M43_REQUESTED_SOURCES:
        raise M43AuditRuntimeError("unknown M4.3a audit mode")
    device = getattr(args, "device", None)
    if device not in {"cpu", "cuda"}:
        raise M43AuditRuntimeError("M4.3a audit device must be cpu or cuda")
    return {
        "schema_version": "langmani-m43-semantic-audit-plan-v0",
        "passed": True,
        "requested_sources": list(M43_REQUESTED_SOURCES[mode]),
        "audited_sources": [],
        "device": device,
        "task_token_m42_status": "rejected",
        "semantic_audit_implementation_validated": True,
        "semantic_audit_completed": False,
        "factor_film_training_completed": False,
        "physical_execution": False,
        "test_split_accessed": False,
        "fresh_seed_accessed": False,
        "final_schedule_accessed": False,
        "final_benchmark_authorized": False,
        "go_no_go_decision_completed": False,
        "smolvla_go": False,
        "inference_deferred": True,
        "inference_deferred_reason": (
            "dry-run validates M4.3a contracts without opening data, checkpoints, "
            "rendering, or schedules"
        ),
    }


def run_semantic_audit(args: Namespace) -> dict[str, object]:
    """Import and execute the heavyweight target path only after CLI preflight."""

    from langmani.policies.m43_audit_execution import run_semantic_audit as execute

    return execute(args)


def _as_float_tensor(value: object, *, shape: tuple[int, ...], name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch Tensor")
    if value.dtype != torch.float32 or tuple(value.shape) != shape:
        raise M43AuditRuntimeError(f"{name} must be float32{shape}")
    if not bool(torch.isfinite(value).all()):
        raise M43AuditRuntimeError(f"{name} must contain only finite values")
    return value.detach().clone()


def _tensor_digest(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True, slots=True)
class FixedPolicyObservation:
    """One portable identity plus runtime-only fixed RGB/state tensors."""

    source_scope: str
    scene_key: str
    observation_id: str
    requested_task_id: str
    source_identity: Mapping[str, object]
    image: torch.Tensor
    state: torch.Tensor

    def __post_init__(self) -> None:
        if self.source_scope not in {"m3b_validation", "m42_dev_v0"}:
            raise M43AuditRuntimeError("fixed observation uses a forbidden audit source")
        for value, name in (
            (self.scene_key, "scene_key"),
            (self.observation_id, "observation_id"),
        ):
            if not isinstance(value, str) or not value:
                raise TypeError(f"{name} must be a non-empty string")
        if self.requested_task_id not in M43_CANONICAL_TASK_IDS:
            raise M43AuditRuntimeError("requested task is outside canonical M1 ordering")
        if not isinstance(self.source_identity, Mapping):
            raise TypeError("source_identity must be a portable mapping")
        if any(
            key in str(self.source_identity).lower()
            for key in ("m3b_test", "fresh_seed", "m42_final_v0")
        ):
            raise M43AuditRuntimeError("source identity references a sealed audit source")
        object.__setattr__(
            self,
            "source_identity",
            {str(key): item for key, item in sorted(self.source_identity.items())},
        )
        object.__setattr__(
            self,
            "image",
            _as_float_tensor(self.image, shape=(3, 256, 256), name="fixed RGB"),
        )
        object.__setattr__(
            self,
            "state",
            _as_float_tensor(self.state, shape=(9,), name="fixed Panda state"),
        )

    @property
    def image_digest(self) -> str:
        return _tensor_digest(self.image)

    @property
    def state_digest(self) -> str:
        return _tensor_digest(self.state)


@dataclass(frozen=True, slots=True)
class SemanticPolicyContexts:
    """Exactly six references and the two frozen shared-policy candidates."""

    per_task: Mapping[str, M42CheckpointContext]
    state_onehot: M42CheckpointContext
    task_token: M42CheckpointContext

    def __post_init__(self) -> None:
        if tuple(self.per_task) != M43_CANONICAL_TASK_IDS:
            raise M43AuditRuntimeError("PerTask contexts must preserve canonical task order")
        for task_id, context in self.per_task.items():
            if (
                context.descriptor.policy_kind is not M42PolicyKind.PER_TASK
                or context.descriptor.task_id != task_id
            ):
                raise M43AuditRuntimeError("PerTask context semantic identity mismatch")
        if self.state_onehot.descriptor.policy_kind is not M42PolicyKind.STATE_ONEHOT:
            raise M43AuditRuntimeError("State-OneHot context has the wrong policy kind")
        if self.task_token.descriptor.policy_kind is not M42PolicyKind.TASK_TOKEN:
            raise M43AuditRuntimeError("TaskToken context has the wrong policy kind")
        dataset_fingerprints = {
            context.descriptor.dataset_fingerprint
            for context in (*self.per_task.values(), self.state_onehot, self.task_token)
        }
        if len(dataset_fingerprints) != 1:
            raise M43AuditRuntimeError("audit checkpoints do not bind one M3B dataset")


@dataclass(frozen=True, slots=True)
class FixedAuditComputation:
    """Typed audit objects plus compact runtime summaries for one source scope."""

    source_scope: str
    scene_count: int
    ordered_observation_ids: tuple[str, ...]
    policy_audits: Mapping[str, SemanticAlignmentAudit]
    policy_runtime_summaries: Mapping[str, Mapping[str, object]]
    deterministic_reset_validated: bool
    locked_execution_horizon: int

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": M43_AUDIT_RUNTIME_VERSION,
            "source_scope": self.source_scope,
            "scene_count": self.scene_count,
            "observation_count": len(self.ordered_observation_ids),
            "ordered_observation_ids": list(self.ordered_observation_ids),
            "fixed_input_validated": True,
            "policy_reset_validated": True,
            "deterministic_reset_validated": self.deterministic_reset_validated,
            "primary_gripper_excluded": True,
            "locked_execution_horizon": self.locked_execution_horizon,
            "audits": {label: self.policy_audits[label].to_dict() for label in M43_POLICY_LABELS},
            "summaries": {
                label: dict(self.policy_runtime_summaries[label]) for label in M43_POLICY_LABELS
            },
            "test_split_accessed": False,
            "fresh_seed_schedule_accessed": False,
            "final_schedule_accessed": False,
        }


def reset_policy_components(context: M42CheckpointContext) -> None:
    """Reset policy, preprocessor, and postprocessor before every query."""

    for component, name in (
        (context.loaded.policy, "policy"),
        (context.loaded.preprocessor, "preprocessor"),
        (context.loaded.postprocessor, "postprocessor"),
    ):
        reset = getattr(component, "reset", None)
        if not callable(reset):
            raise M43AuditRuntimeError(f"{name} lacks the required reset() contract")
        reset()


def predict_postprocessed_action_chunk(
    *,
    context: M42CheckpointContext,
    observation: FixedPolicyObservation,
    task_id: str,
) -> NDArray[np.float64]:
    """Infer one 50x8 chunk after the policy postprocessor and before runtime transforms."""

    if task_id not in M43_CANONICAL_TASK_IDS:
        raise M43AuditRuntimeError("action-chunk query uses an unknown task ID")
    reset_policy_components(context)
    state = observation.state.clone()
    kind = context.descriptor.policy_kind
    if kind is M42PolicyKind.PER_TASK:
        if context.descriptor.task_id != task_id:
            raise M43AuditRuntimeError("PerTask reference queried with the wrong TaskSpec")
    elif kind is M42PolicyKind.STATE_ONEHOT:
        state = cast(torch.Tensor, append_canonical_task_onehot(state, task_id))
    elif kind is not M42PolicyKind.TASK_TOKEN:
        raise M43AuditRuntimeError("unsupported policy kind in semantic audit")

    raw = {
        IMAGE_FEATURE_KEY: observation.image.clone(),
        STATE_FEATURE_KEY: state,
    }
    processed = context.loaded.preprocessor(raw)
    if not isinstance(processed, Mapping):
        raise M43AuditRuntimeError("policy preprocessor returned a non-mapping")
    batch = cast(dict[str, torch.Tensor], dict(processed))
    if kind is M42PolicyKind.TASK_TOKEN:
        from langmani.policies.act_task_token import TASK_TOKEN_FEATURE_KEY, canonical_task_token

        processed_state = batch.get(STATE_FEATURE_KEY)
        if not isinstance(processed_state, torch.Tensor):
            raise M43AuditRuntimeError("TaskToken query lacks processed Panda state")
        batch[TASK_TOKEN_FEATURE_KEY] = canonical_task_token(
            task_id, device=processed_state.device
        ).unsqueeze(0)

    with torch.inference_mode():
        prediction = cast(Any, context.loaded.policy).predict_action_chunk(batch)
        action = context.loaded.postprocessor(prediction)
    if not isinstance(action, torch.Tensor):
        raise M43AuditRuntimeError("policy postprocessor returned a non-tensor")
    array = action.detach().cpu().numpy()
    expected = (1, M43_ACTION_CHUNK_SIZE, M43_ACTION_COMPONENTS)
    if array.shape != expected or not np.all(np.isfinite(array)):
        raise M43AuditRuntimeError(f"postprocessed action chunk must be finite {expected}")
    return np.asarray(array[0], dtype=np.float64).copy()


def _validate_fixed_scene_group(
    values: Sequence[FixedPolicyObservation],
) -> tuple[FixedPolicyObservation, ...]:
    if len(values) != len(M43_CANONICAL_TASK_IDS):
        raise M43AuditRuntimeError("each scene requires exactly six task queries")
    by_task = {value.requested_task_id: value for value in values}
    if len(by_task) != len(values) or tuple(by_task) != M43_CANONICAL_TASK_IDS:
        raise M43AuditRuntimeError("scene queries must use canonical TaskSpec ordering")
    ordered = tuple(by_task[task_id] for task_id in M43_CANONICAL_TASK_IDS)
    reference = ordered[0]
    if any(value.scene_key != reference.scene_key for value in ordered):
        raise M43AuditRuntimeError("one audit group contains multiple physical scenes")
    if any(
        value.source_scope != reference.source_scope
        or value.image_digest != reference.image_digest
        or value.state_digest != reference.state_digest
        or not torch.equal(value.image, reference.image)
        or not torch.equal(value.state, reference.state)
        for value in ordered[1:]
    ):
        raise M43AuditRuntimeError("RGB and Panda state must remain fixed across task queries")
    return ordered


def validate_fixed_observation_set(
    observations: Sequence[FixedPolicyObservation],
) -> tuple[tuple[FixedPolicyObservation, ...], ...]:
    """Validate deterministic source/task ordering and return scene groups."""

    if not observations:
        raise M43AuditRuntimeError("semantic audit requires fixed observations")
    grouped: dict[str, list[FixedPolicyObservation]] = {}
    scope: str | None = None
    seen_ids: set[str] = set()
    for value in observations:
        if value.observation_id in seen_ids:
            raise M43AuditRuntimeError("observation identities must be unique")
        seen_ids.add(value.observation_id)
        scope = value.source_scope if scope is None else scope
        if value.source_scope != scope:
            raise M43AuditRuntimeError(
                "validation and development observations cannot mix silently"
            )
        grouped.setdefault(value.scene_key, []).append(value)
    groups = tuple(_validate_fixed_scene_group(grouped[key]) for key in sorted(grouped))
    flattened = tuple(value.observation_id for group in groups for value in group)
    if flattened != tuple(value.observation_id for value in observations):
        raise M43AuditRuntimeError("observations must use sorted scene and canonical task order")
    return groups


def _task_sensitivity(
    chunks_by_scene: Sequence[Mapping[str, NDArray[np.float64]]],
) -> dict[str, object]:
    per_scene: list[dict[str, object]] = []
    all_distances: list[float] = []
    for index, chunks in enumerate(chunks_by_scene):
        if tuple(chunks) != M43_CANONICAL_TASK_IDS:
            raise M43AuditRuntimeError("task sensitivity requires six canonical chunks")
        distances = []
        for left, right in combinations(M43_CANONICAL_TASK_IDS, 2):
            delta = chunks[left] - chunks[right]
            distance = float(np.sqrt(np.mean(np.square(delta, dtype=np.float64))))
            if not math.isfinite(distance):
                raise M43AuditRuntimeError("task sensitivity produced a non-finite value")
            distances.append(distance)
        all_distances.extend(distances)
        per_scene.append(
            {
                "scene_index": index,
                "pair_count": len(distances),
                "mean_full_chunk_rms": float(np.mean(distances)),
                "maximum_full_chunk_rms": max(distances),
            }
        )
    return {
        "version": "M42FullChunkRmsTaskSensitivityV0",
        "scene_count": len(chunks_by_scene),
        "pair_count": len(all_distances),
        "mean_full_chunk_rms": float(np.mean(all_distances)),
        "maximum_full_chunk_rms": max(all_distances),
        "nonzero_pair_fraction": sum(value > 0.0 for value in all_distances) / len(all_distances),
        "per_scene": per_scene,
    }


def _raw_action_metrics(
    chunks: Sequence[NDArray[np.float64]], config: ActionChunkDistanceConfig
) -> dict[str, object]:
    values = np.stack(chunks)
    low = np.asarray(config.action_lower_bounds, dtype=np.float64)
    high = np.asarray(config.action_upper_bounds, dtype=np.float64)
    below = values < low
    above = values > high
    violations = below | above
    correction = np.maximum(low - values, 0.0) + np.maximum(values - high, 0.0)
    per_dimension = violations.reshape(-1, M43_ACTION_COMPONENTS).sum(axis=0)
    return {
        "action_chunk_count": len(chunks),
        "action_count": int(values.shape[0] * values.shape[1]),
        "nonfinite_count": int((~np.isfinite(values)).sum()),
        "out_of_bounds_action_count": int(violations.any(axis=2).sum()),
        "out_of_bounds_component_count": int(violations.sum()),
        "per_action_dimension_violation_counts": [int(value) for value in per_dimension],
        "maximum_absolute_bound_excess": float(correction.max(initial=0.0)),
    }


def _complete_distance_record(
    *,
    candidate_chunk: NDArray[np.float64],
    requested_task_id: str,
    per_task_chunks: Mapping[str, NDArray[np.float64]],
    config: ActionChunkDistanceConfig,
) -> dict[str, object]:
    """Persist every ActionChunkDistanceV0 family used by semantic retrieval."""

    def centroid(task_ids: Sequence[str]) -> NDArray[np.float64]:
        return np.mean(np.stack([per_task_chunks[task_id] for task_id in task_ids]), axis=0)

    task_references = {
        task_id: compute_action_chunk_distance(
            candidate_chunk, per_task_chunks[task_id], config
        ).to_dict()
        for task_id in M43_CANONICAL_TASK_IDS
    }
    object_centroids = {
        object_id: compute_action_chunk_distance(
            candidate_chunk,
            centroid(
                tuple(
                    stable_task_id(task_spec)
                    for task_spec in CANONICAL_TASK_SPECS
                    if task_spec.target_object_id == object_id
                )
            ),
            config,
        ).to_dict()
        for object_id in OBJECT_IDS
    }
    bin_centroids = {
        bin_id: compute_action_chunk_distance(
            candidate_chunk,
            centroid(
                tuple(
                    stable_task_id(task_spec)
                    for task_spec in CANONICAL_TASK_SPECS
                    if task_spec.target_bin_id == bin_id
                )
            ),
            config,
        ).to_dict()
        for bin_id in BIN_IDS
    }
    requested_object = _TASK_SPEC_BY_ID[requested_task_id].target_object_id
    conditional_bin_references = {
        bin_id: compute_action_chunk_distance(
            candidate_chunk,
            per_task_chunks[_TASK_ID_BY_SEMANTICS[(requested_object, bin_id)]],
            config,
        ).to_dict()
        for bin_id in BIN_IDS
    }
    return {
        "task_references": task_references,
        "object_centroids": object_centroids,
        "bin_centroids": bin_centroids,
        "conditional_bin_references": conditional_bin_references,
    }


def _candidate_audit(
    *,
    policy_label: str,
    task_results: Sequence[SemanticRetrievalResult],
    object_results: Sequence[ObjectRetrievalResult],
    bin_results: Sequence[BinRetrievalResult],
    chunks_by_scene: Sequence[Mapping[str, NDArray[np.float64]]],
    complete_distance_results: Mapping[str, object],
    distance_config: ActionChunkDistanceConfig,
    deterministic_reset_validated: bool,
) -> tuple[SemanticAlignmentAudit, dict[str, object]]:
    task_summary = summarize_task_retrieval(task_results)
    object_summary = summarize_object_retrieval(object_results)
    bin_summary = summarize_bin_retrieval(bin_results)
    sensitivity = _task_sensitivity(chunks_by_scene)
    condition_changes = cast(float, sensitivity["mean_full_chunk_rms"]) > 0.0
    correct_task = all(value.top1_correct for value in task_results)
    correct_object = all(value.correct for value in object_results)
    correct_bin = all(value.correct and value.conditional_correct for value in bin_results)
    correct_semantic_alignment = correct_task and correct_object and correct_bin
    conclusions = classify_semantic_conclusions(
        sufficient_evidence=True,
        condition_changes_output=condition_changes,
        correct_task_retrieval=correct_task,
        correct_object_retrieval=correct_object,
        correct_bin_retrieval=correct_bin,
        shared_control_degraded_after_correct_selection=False,
        post_grasp_execution_failed=False,
    )
    raw_metrics = _raw_action_metrics(
        [chunk for scene in chunks_by_scene for chunk in scene.values()], distance_config
    )
    runtime_metrics = {
        "available": False,
        "reason": "zero_training_audit_stops_before_runtime_action_transform",
    }
    audit = build_semantic_alignment_audit(
        policy_label=policy_label,
        distance_config=distance_config,
        task_results=task_results,
        object_results=object_results,
        bin_results=bin_results,
        first_interactions=(),
        conclusions=conclusions,
        complete_distance_results=complete_distance_results,
        raw_policy_action_metrics=raw_metrics,
        runtime_action_metrics=runtime_metrics,
        deterministic_reload_validated=False,
        deterministic_reset_validated=deterministic_reset_validated,
    )
    summary = {
        "policy_label": policy_label,
        "task_retrieval": task_summary,
        "object_retrieval": object_summary,
        "bin_retrieval": bin_summary,
        "task_confusion": audit.task_confusion.to_dict(),
        "object_confusion": audit.object_confusion.to_dict(),
        "bin_confusion": audit.bin_confusion.to_dict(),
        "first_interaction": {
            "available": False,
            "reason": "zero_training_fixed-observation audit does not execute rollouts",
        },
        "post_grasp_failure_distribution": {
            "available": False,
            "reason": "must be joined from immutable development rollout evidence",
        },
        "task_sensitivity": sensitivity,
        "semantic_interpretation": {
            "output_changes_across_task_conditions": condition_changes,
            "correct_semantic_alignment": correct_semantic_alignment,
            "full_task_retrieval_consistent": correct_task,
            "object_retrieval_consistent": correct_object,
            "bin_retrieval_consistent": correct_bin,
            "target_object_confusion": not correct_object,
            "destination_bin_confusion": not correct_bin,
            "post_grasp_control_failure": None,
            "nonzero_distance_is_not_semantic_correctness": True,
        },
        "raw_policy_action_metrics": raw_metrics,
        "runtime_action_metrics": runtime_metrics,
        "conclusions": [value.value for value in conclusions],
        "deterministic_reload": {
            "validated": False,
            "reason": "fresh checkpoint instances were not reloaded by this audit",
        },
    }
    return audit, summary


def _combined_sensitivity(
    left: Mapping[str, object], right: Mapping[str, object]
) -> dict[str, object]:
    """Combine two explicitly scoped sensitivity summaries without hiding provenance."""

    if left.get("version") != right.get("version") or left.get("version") != (
        "M42FullChunkRmsTaskSensitivityV0"
    ):
        raise M43AuditRuntimeError("cannot combine incompatible task-sensitivity evidence")
    left_pairs = int(cast(int, left["pair_count"]))
    right_pairs = int(cast(int, right["pair_count"]))
    pair_count = left_pairs + right_pairs
    if left_pairs < 1 or right_pairs < 1 or pair_count < 1:
        raise M43AuditRuntimeError("combined sensitivity requires both source scopes")
    left_scenes = cast(Sequence[Mapping[str, object]], left["per_scene"])
    right_scenes = cast(Sequence[Mapping[str, object]], right["per_scene"])
    per_scene = [
        {**dict(value), "scene_index": index}
        for index, value in enumerate((*left_scenes, *right_scenes))
    ]
    return {
        "version": left["version"],
        "scene_count": int(cast(int, left["scene_count"])) + int(cast(int, right["scene_count"])),
        "pair_count": pair_count,
        "mean_full_chunk_rms": (
            float(cast(float, left["mean_full_chunk_rms"])) * left_pairs
            + float(cast(float, right["mean_full_chunk_rms"])) * right_pairs
        )
        / pair_count,
        "maximum_full_chunk_rms": max(
            float(cast(float, left["maximum_full_chunk_rms"])),
            float(cast(float, right["maximum_full_chunk_rms"])),
        ),
        "nonzero_pair_fraction": (
            float(cast(float, left["nonzero_pair_fraction"])) * left_pairs
            + float(cast(float, right["nonzero_pair_fraction"])) * right_pairs
        )
        / pair_count,
        "per_scene": per_scene,
    }


def _combined_raw_action_metrics(
    left: Mapping[str, object], right: Mapping[str, object]
) -> dict[str, object]:
    dimensions = M43_ACTION_COMPONENTS
    left_by_dimension = cast(Sequence[int], left["per_action_dimension_violation_counts"])
    right_by_dimension = cast(Sequence[int], right["per_action_dimension_violation_counts"])
    if len(left_by_dimension) != dimensions or len(right_by_dimension) != dimensions:
        raise M43AuditRuntimeError("raw action metrics have the wrong action dimension")
    return {
        name: int(cast(int, left[name])) + int(cast(int, right[name]))
        for name in (
            "action_chunk_count",
            "action_count",
            "nonfinite_count",
            "out_of_bounds_action_count",
            "out_of_bounds_component_count",
        )
    } | {
        "per_action_dimension_violation_counts": [
            int(left_by_dimension[index]) + int(right_by_dimension[index])
            for index in range(dimensions)
        ],
        "maximum_absolute_bound_excess": max(
            float(cast(float, left["maximum_absolute_bound_excess"])),
            float(cast(float, right["maximum_absolute_bound_excess"])),
        ),
    }


def _combined_candidate_audit(
    *,
    policy_label: str,
    left: SemanticAlignmentAudit,
    right: SemanticAlignmentAudit,
    sensitivity: Mapping[str, object],
) -> tuple[SemanticAlignmentAudit, Mapping[str, object]]:
    if left.policy_label != policy_label or right.policy_label != policy_label:
        raise M43AuditRuntimeError("combined candidate policy labels disagree")
    if left.distance_config.to_dict() != right.distance_config.to_dict():
        raise M43AuditRuntimeError("combined source scopes use different distance contracts")
    task_results = (*left.task_retrieval, *right.task_retrieval)
    object_results = (*left.object_retrieval, *right.object_retrieval)
    bin_results = (*left.bin_retrieval, *right.bin_retrieval)
    task_summary = summarize_task_retrieval(task_results)
    object_summary = summarize_object_retrieval(object_results)
    bin_summary = summarize_bin_retrieval(bin_results)
    condition_changes = float(cast(float, sensitivity["mean_full_chunk_rms"])) > 0.0
    correct_task = all(value.top1_correct for value in task_results)
    correct_object = all(value.correct for value in object_results)
    correct_bin = all(value.correct and value.conditional_correct for value in bin_results)
    correct_semantic_alignment = correct_task and correct_object and correct_bin
    conclusions = classify_semantic_conclusions(
        sufficient_evidence=True,
        condition_changes_output=condition_changes,
        correct_task_retrieval=correct_task,
        correct_object_retrieval=correct_object,
        correct_bin_retrieval=correct_bin,
        shared_control_degraded_after_correct_selection=False,
        post_grasp_execution_failed=False,
    )
    raw_metrics = _combined_raw_action_metrics(
        left.raw_policy_action_metrics, right.raw_policy_action_metrics
    )
    runtime_metrics = {
        "available": False,
        "reason": "zero_training_audit_stops_before_runtime_action_transform",
    }
    audit = build_semantic_alignment_audit(
        policy_label=policy_label,
        distance_config=left.distance_config,
        task_results=task_results,
        object_results=object_results,
        bin_results=bin_results,
        first_interactions=(),
        conclusions=conclusions,
        complete_distance_results={
            **dict(left.complete_distance_results),
            **dict(right.complete_distance_results),
        },
        raw_policy_action_metrics=raw_metrics,
        runtime_action_metrics=runtime_metrics,
        deterministic_reload_validated=(
            left.deterministic_reload_validated and right.deterministic_reload_validated
        ),
        deterministic_reset_validated=(
            left.deterministic_reset_validated and right.deterministic_reset_validated
        ),
    )
    summary = {
        "policy_label": policy_label,
        "task_retrieval": task_summary,
        "object_retrieval": object_summary,
        "bin_retrieval": bin_summary,
        "task_confusion": audit.task_confusion.to_dict(),
        "object_confusion": audit.object_confusion.to_dict(),
        "bin_confusion": audit.bin_confusion.to_dict(),
        "first_interaction": {
            "available": False,
            "reason": "zero_training_fixed-observation audit does not execute rollouts",
        },
        "post_grasp_failure_distribution": {
            "available": False,
            "reason": "must be joined from immutable development rollout evidence",
        },
        "task_sensitivity": dict(sensitivity),
        "semantic_interpretation": {
            "output_changes_across_task_conditions": condition_changes,
            "correct_semantic_alignment": correct_semantic_alignment,
            "full_task_retrieval_consistent": correct_task,
            "object_retrieval_consistent": correct_object,
            "bin_retrieval_consistent": correct_bin,
            "target_object_confusion": not correct_object,
            "destination_bin_confusion": not correct_bin,
            "post_grasp_control_failure": None,
            "nonzero_distance_is_not_semantic_correctness": True,
        },
        "raw_policy_action_metrics": raw_metrics,
        "runtime_action_metrics": runtime_metrics,
        "conclusions": [value.value for value in conclusions],
        "deterministic_reload": {
            "validated": False,
            "reason": "fresh checkpoint instances were not reloaded by this audit",
        },
    }
    return audit, summary


def combine_fixed_audit_computations(
    validation: FixedAuditComputation,
    development: FixedAuditComputation,
) -> FixedAuditComputation:
    """Create a labeled combined view from two independently computed scopes."""

    if validation.source_scope != "m3b_validation":
        raise M43AuditRuntimeError("combined audit requires validation evidence first")
    if development.source_scope != "m42_dev_v0":
        raise M43AuditRuntimeError("combined audit requires development evidence second")
    if validation.locked_execution_horizon != development.locked_execution_horizon:
        raise M43AuditRuntimeError("source scopes use different execution horizons")
    if set(validation.ordered_observation_ids) & set(development.ordered_observation_ids):
        raise M43AuditRuntimeError("combined source observation identities overlap")
    audits: dict[str, SemanticAlignmentAudit] = {}
    summaries: dict[str, Mapping[str, object]] = {}
    for label in M43_POLICY_LABELS:
        left_summary = validation.policy_runtime_summaries[label]
        right_summary = development.policy_runtime_summaries[label]
        sensitivity = _combined_sensitivity(
            cast(Mapping[str, object], left_summary["task_sensitivity"]),
            cast(Mapping[str, object], right_summary["task_sensitivity"]),
        )
        audit, summary = _combined_candidate_audit(
            policy_label=label,
            left=validation.policy_audits[label],
            right=development.policy_audits[label],
            sensitivity=sensitivity,
        )
        audits[label] = audit
        summaries[label] = summary
    return FixedAuditComputation(
        source_scope="combined",
        scene_count=validation.scene_count + development.scene_count,
        ordered_observation_ids=(
            *validation.ordered_observation_ids,
            *development.ordered_observation_ids,
        ),
        policy_audits=audits,
        policy_runtime_summaries=summaries,
        deterministic_reset_validated=(
            validation.deterministic_reset_validated and development.deterministic_reset_validated
        ),
        locked_execution_horizon=validation.locked_execution_horizon,
    )


def compute_fixed_observation_audits(
    *,
    observations: Sequence[FixedPolicyObservation],
    policies: SemanticPolicyContexts,
    distance_config: ActionChunkDistanceConfig,
) -> FixedAuditComputation:
    """Run both candidate policies against six references on fixed scene inputs."""

    groups = validate_fixed_observation_set(observations)
    task_results: dict[str, list[SemanticRetrievalResult]] = {
        label: [] for label in M43_POLICY_LABELS
    }
    object_results: dict[str, list[ObjectRetrievalResult]] = {
        label: [] for label in M43_POLICY_LABELS
    }
    bin_results: dict[str, list[BinRetrievalResult]] = {label: [] for label in M43_POLICY_LABELS}
    candidate_chunks: dict[str, list[dict[str, NDArray[np.float64]]]] = {
        label: [] for label in M43_POLICY_LABELS
    }
    complete_distance_results: dict[str, dict[str, object]] = {
        label: {} for label in M43_POLICY_LABELS
    }

    deterministic_reset_validated = True
    for group_index, group in enumerate(groups):
        base = group[0]
        references = {
            task_id: predict_postprocessed_action_chunk(
                context=policies.per_task[task_id], observation=base, task_id=task_id
            )
            for task_id in M43_CANONICAL_TASK_IDS
        }
        per_policy_scene = {label: {} for label in M43_POLICY_LABELS}
        for query in group:
            chunks = {
                "state_onehot": predict_postprocessed_action_chunk(
                    context=policies.state_onehot,
                    observation=query,
                    task_id=query.requested_task_id,
                ),
                "task_token": predict_postprocessed_action_chunk(
                    context=policies.task_token,
                    observation=query,
                    task_id=query.requested_task_id,
                ),
            }
            if group_index == 0:
                for label, context in (
                    ("state_onehot", policies.state_onehot),
                    ("task_token", policies.task_token),
                ):
                    repeated = predict_postprocessed_action_chunk(
                        context=context,
                        observation=query,
                        task_id=query.requested_task_id,
                    )
                    deterministic_reset_validated &= np.array_equal(chunks[label], repeated)
            for label, chunk in chunks.items():
                per_policy_scene[label][query.requested_task_id] = chunk
                complete_distance_results[label][query.observation_id] = _complete_distance_record(
                    candidate_chunk=chunk,
                    requested_task_id=query.requested_task_id,
                    per_task_chunks=references,
                    config=distance_config,
                )
                task_results[label].append(
                    retrieve_task(
                        observation_id=query.observation_id,
                        requested_task_id=query.requested_task_id,
                        candidate_chunk=chunk,
                        per_task_chunks=references,
                        config=distance_config,
                    )
                )
                object_results[label].append(
                    retrieve_object(
                        observation_id=query.observation_id,
                        requested_task_id=query.requested_task_id,
                        candidate_chunk=chunk,
                        per_task_chunks=references,
                        config=distance_config,
                    )
                )
                bin_results[label].append(
                    retrieve_bin(
                        observation_id=query.observation_id,
                        requested_task_id=query.requested_task_id,
                        candidate_chunk=chunk,
                        per_task_chunks=references,
                        config=distance_config,
                    )
                )
        for label in M43_POLICY_LABELS:
            candidate_chunks[label].append(per_policy_scene[label])

    if not deterministic_reset_validated:
        raise M43AuditRuntimeError("reset deterministic inference produced different chunks")
    audits: dict[str, SemanticAlignmentAudit] = {}
    summaries: dict[str, Mapping[str, object]] = {}
    for label in M43_POLICY_LABELS:
        audit, summary = _candidate_audit(
            policy_label=label,
            task_results=task_results[label],
            object_results=object_results[label],
            bin_results=bin_results[label],
            chunks_by_scene=candidate_chunks[label],
            complete_distance_results=complete_distance_results[label],
            distance_config=distance_config,
            deterministic_reset_validated=deterministic_reset_validated,
        )
        audits[label] = audit
        summaries[label] = summary
    return FixedAuditComputation(
        source_scope=groups[0][0].source_scope,
        scene_count=len(groups),
        ordered_observation_ids=tuple(value.observation_id for value in observations),
        policy_audits=audits,
        policy_runtime_summaries=summaries,
        deterministic_reset_validated=deterministic_reset_validated,
        locked_execution_horizon=distance_config.locked_execution_horizon,
    )


def audit_fixed_observations(
    *,
    observations: Sequence[FixedPolicyObservation],
    policies: SemanticPolicyContexts,
    distance_config: ActionChunkDistanceConfig,
) -> dict[str, object]:
    """JSON-ready wrapper around :func:`compute_fixed_observation_audits`."""

    return compute_fixed_observation_audits(
        observations=observations,
        policies=policies,
        distance_config=distance_config,
    ).to_dict()


__all__ = [
    "M43_AUDIT_RUNTIME_VERSION",
    "M43AuditRuntimeError",
    "FixedPolicyObservation",
    "FixedAuditComputation",
    "SemanticPolicyContexts",
    "audit_fixed_observations",
    "combine_fixed_audit_computations",
    "compute_fixed_observation_audits",
    "plan_semantic_audit",
    "predict_postprocessed_action_chunk",
    "reset_policy_components",
    "run_semantic_audit",
    "validate_fixed_observation_set",
]
