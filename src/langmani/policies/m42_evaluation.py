"""M4.2 checkpoint loading, privileged rollout diagnostics, and paired reports.

The policy path in this module sees only the M1 RGB image, PandaPolicyStateV0,
and its declared oracle command representation.  The environment diagnostic
proxy samples privileged state *after* an action has been chosen and is used
only to classify failures.  It never calls or imports the M2 expert.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from statistics import median
from typing import Any, Protocol, cast

import numpy as np
import torch

from langmani.datasets.identity import sha256_hex
from langmani.datasets.lerobot_types import STATE_FEATURE_KEY
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.pick_place_by_instruction import (
    BIN_INTERIOR_HALF_SIZE,
    CUBE_HALF_SIZE,
)
from langmani.environments.specs import TaskSpec, stable_scene_id, stable_task_id
from langmani.policies.act_action_bounds import (
    ActionBoundConfig,
    ActionBoundErrorKind,
    ActionBoundMode,
    ActionBoundProcessingError,
    ActionProjectionRecord,
)
from langmani.policies.act_checkpoint import LoadedActCheckpoint, load_act_checkpoint
from langmani.policies.act_conditioning import (
    CANONICAL_TASK_IDS,
    append_canonical_task_onehot,
)
from langmani.policies.act_evaluation import (
    load_checkpoint_selection,
    wilson_interval,
)
from langmani.policies.act_factor_film_conditioning import (
    attach_factor_film_runtime_input,
    factorized_runtime_input,
)
from langmani.policies.act_factor_film_types import FactorFiLMContractError
from langmani.policies.act_rollout import ActManiSkillRolloutAdapter, latency_percentiles
from langmani.policies.act_task_token import (
    TASK_TOKEN_FEATURE_KEY,
    canonical_task_token,
    validate_task_token_policy,
)
from langmani.policies.act_types import (
    ActEvaluationConfig,
    ActExperimentManifest,
    ActVariant,
    EvaluationSplit,
    RolloutEpisodeResult,
)
from langmani.policies.m42_analysis import (
    PostGraspTrace,
    classify_post_grasp_phase,
    paired_success_metrics,
)
from langmani.policies.m42_runtime import (
    BinaryGripperEnvPostprocessorV0,
    ExecutionHorizonPolicyV0,
)
from langmani.policies.m42_schedule import (
    M42_DEV_SCHEDULE_ID,
    M42_FINAL_SCHEDULE_ID,
    FinalScheduleAuthorization,
    M42ScheduledEpisode,
    load_locked_schedule,
)
from langmani.policies.m42_types import (
    ExecutionHorizonConfig,
    GripperRuntimeMode,
    PostGraspFailureRecord,
    PostGraspPhase,
    RuntimeAblationResult,
    TaskTokenValidationResult,
)

M42_EVALUATION_SCHEMA_VERSION = "langmani-m42-evaluation-v0"
M42_POST_GRASP_DIAGNOSTIC_VERSION = "M42PostGraspDiagnosticV0"
M42_LIFT_DELTA_METERS = 0.04
M42_DESTINATION_NEAR_XY_METERS = BIN_INTERIOR_HALF_SIZE + CUBE_HALF_SIZE
M42_DESCENT_HEIGHT_METERS = CUBE_HALF_SIZE + 0.035
_DIGEST_PREFIX = "sha256:"


class M42EvaluationError(RuntimeError):
    """Raised when evaluation evidence is incomplete, unsafe, or inconsistent."""


class M42PolicyKind(StrEnum):
    PER_TASK = "per_task"
    STATE_ONEHOT = "state_onehot"
    TASK_TOKEN = "task_token"
    FACTOR_FILM = "factor_film"


class _CheckpointIdentity(Protocol):
    run_fingerprint: str
    m3b_export_fingerprint: str
    m3b_split_manifest_digest: str
    train_statistics_fingerprint: str
    git_commit: str
    model_config: Mapping[str, object]

    def to_dict(self) -> dict[str, object]: ...


def _fingerprint(value: object) -> str:
    return f"{_DIGEST_PREFIX}{sha256_hex(value)}"


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise M42EvaluationError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise M42EvaluationError(f"{label} must be a JSON object")
    return value


def _require_digest(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.startswith(_DIGEST_PREFIX) or len(value) != 71:
        raise M42EvaluationError(f"{label} must be a prefixed SHA-256 digest")


@dataclass(frozen=True, slots=True)
class M42CheckpointDescriptor:
    policy_kind: M42PolicyKind
    run_fingerprint: str
    checkpoint_fingerprint: str
    checkpoint_relative_path: str
    dataset_fingerprint: str
    split_digest: str
    statistics_fingerprint: str
    architecture_fingerprint: str | None
    task_id: str | None
    git_commit: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.run_fingerprint, "run_fingerprint"),
            (self.checkpoint_fingerprint, "checkpoint_fingerprint"),
            (self.dataset_fingerprint, "dataset_fingerprint"),
            (self.split_digest, "split_digest"),
            (self.statistics_fingerprint, "statistics_fingerprint"),
        ):
            _require_digest(value, label)
        if self.architecture_fingerprint is not None:
            _require_digest(self.architecture_fingerprint, "architecture_fingerprint")
        if self.policy_kind is M42PolicyKind.PER_TASK:
            if self.task_id not in CANONICAL_TASK_IDS:
                raise M42EvaluationError("PerTask checkpoint requires one canonical task ID")
        elif self.task_id is not None:
            raise M42EvaluationError("mixed checkpoint descriptors cannot bind one task ID")
        if self.policy_kind in {M42PolicyKind.TASK_TOKEN, M42PolicyKind.FACTOR_FILM} and (
            self.architecture_fingerprint is None
        ):
            raise M42EvaluationError("custom mixed checkpoints require an architecture fingerprint")

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_kind": self.policy_kind.value,
            "run_fingerprint": self.run_fingerprint,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "checkpoint_relative_path": self.checkpoint_relative_path,
            "dataset_fingerprint": self.dataset_fingerprint,
            "split_digest": self.split_digest,
            "statistics_fingerprint": self.statistics_fingerprint,
            "architecture_fingerprint": self.architecture_fingerprint,
            "task_id": self.task_id,
            "git_commit": self.git_commit,
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())


@dataclass(frozen=True, slots=True)
class M42CheckpointContext:
    descriptor: M42CheckpointDescriptor
    loaded: LoadedActCheckpoint


def _assert_expected(actual: str, expected: str | None, label: str) -> None:
    if expected is not None and actual != expected:
        raise M42EvaluationError(f"{label} differs from the frozen expected value")


def load_m4_checkpoint_context(
    run_root: str | Path,
    *,
    policy_kind: M42PolicyKind,
    expected_checkpoint_fingerprint: str,
    expected_run_fingerprint: str | None = None,
    expected_dataset_fingerprint: str | None = None,
    expected_task_id: str | None = None,
) -> M42CheckpointContext:
    """Integrity-load one selected historical M4 checkpoint without changing it."""
    if policy_kind not in {M42PolicyKind.PER_TASK, M42PolicyKind.STATE_ONEHOT}:
        raise M42EvaluationError("historical loader supports only PerTask and State-OneHot")
    root = Path(run_root).resolve()
    manifest = ActExperimentManifest.from_dict(
        _read_json_object(root / "run_manifest.json", "M4 run manifest")
    )
    selection = load_checkpoint_selection(root / "checkpoint_selection.json")
    selected = selection.selected_checkpoint_fingerprint
    if manifest.selected_checkpoint_fingerprint != selected:
        raise M42EvaluationError("M4 run manifest and checkpoint selection disagree")
    _assert_expected(selected, expected_checkpoint_fingerprint, "checkpoint fingerprint")
    _assert_expected(manifest.identity.run_fingerprint, expected_run_fingerprint, "run fingerprint")
    _assert_expected(
        manifest.identity.m3b_export_fingerprint,
        expected_dataset_fingerprint,
        "M3B dataset fingerprint",
    )
    expected_variant = (
        ActVariant.PER_TASK
        if policy_kind is M42PolicyKind.PER_TASK
        else ActVariant.MIXED_TASK_ONEHOT
    )
    if manifest.identity.variant is not expected_variant:
        raise M42EvaluationError("historical checkpoint has the wrong ACT variant")
    _assert_expected(manifest.identity.task_id or "", expected_task_id, "task ID")
    record = next(
        (item for item in manifest.checkpoints if item.checkpoint_fingerprint == selected), None
    )
    if record is None:
        raise M42EvaluationError("selected checkpoint is absent from the M4 run manifest")
    loaded = load_act_checkpoint(
        run_root=root,
        checkpoint_relative_path=record.relative_path,
        expected_identity=manifest.identity,
        restore_rng=False,
    )
    if loaded.record.checkpoint_fingerprint != selected:
        raise M42EvaluationError("loaded M4 checkpoint fingerprint changed during reload")
    descriptor = M42CheckpointDescriptor(
        policy_kind=policy_kind,
        run_fingerprint=manifest.identity.run_fingerprint,
        checkpoint_fingerprint=selected,
        checkpoint_relative_path=record.relative_path,
        dataset_fingerprint=manifest.identity.m3b_export_fingerprint,
        split_digest=manifest.identity.m3b_split_manifest_digest,
        statistics_fingerprint=manifest.identity.train_statistics_fingerprint,
        architecture_fingerprint=None,
        task_id=manifest.identity.task_id,
        git_commit=manifest.identity.git_commit,
    )
    return M42CheckpointContext(descriptor=descriptor, loaded=loaded)


def load_task_token_checkpoint_context(
    run_root: str | Path,
    *,
    expected_identity: _CheckpointIdentity,
    checkpoint_relative_path: str,
    expected_checkpoint_fingerprint: str,
    expected_architecture_fingerprint: str,
) -> M42CheckpointContext:
    """Integrity-load one TaskToken checkpoint through the existing M4 lifecycle."""
    _require_digest(expected_architecture_fingerprint, "architecture fingerprint")
    actual_architecture = getattr(expected_identity, "task_token_architecture_fingerprint", None)
    if actual_architecture != expected_architecture_fingerprint:
        raise M42EvaluationError("TaskToken identity architecture fingerprint mismatch")
    loaded = load_act_checkpoint(
        run_root=run_root,
        checkpoint_relative_path=checkpoint_relative_path,
        expected_identity=cast(Any, expected_identity),
        restore_rng=False,
    )
    if loaded.record.checkpoint_fingerprint != expected_checkpoint_fingerprint:
        raise M42EvaluationError("loaded TaskToken checkpoint differs from selection")
    validate_task_token_policy(cast(Any, loaded.policy))
    descriptor = M42CheckpointDescriptor(
        policy_kind=M42PolicyKind.TASK_TOKEN,
        run_fingerprint=expected_identity.run_fingerprint,
        checkpoint_fingerprint=expected_checkpoint_fingerprint,
        checkpoint_relative_path=checkpoint_relative_path,
        dataset_fingerprint=expected_identity.m3b_export_fingerprint,
        split_digest=expected_identity.m3b_split_manifest_digest,
        statistics_fingerprint=expected_identity.train_statistics_fingerprint,
        architecture_fingerprint=expected_architecture_fingerprint,
        task_id=None,
        git_commit=expected_identity.git_commit,
    )
    return M42CheckpointContext(descriptor=descriptor, loaded=loaded)


def load_task_token_training_checkpoint_context(
    run_root: str | Path,
    *,
    expected_checkpoint_fingerprint: str,
    expected_run_fingerprint: str | None = None,
    expected_dataset_fingerprint: str | None = None,
    expected_architecture_fingerprint: str | None = None,
) -> M42CheckpointContext:
    """Load one TaskToken checkpoint directly from its immutable training manifest."""
    from langmani.policies.m42_training import TaskTokenTrainingManifest

    root = Path(run_root).resolve()
    manifest = TaskTokenTrainingManifest.from_dict(
        _read_json_object(root / "run_manifest.json", "TaskToken training manifest")
    )
    if not manifest.training_complete:
        raise M42EvaluationError("TaskToken checkpoint evaluation requires completed training")
    identity = manifest.identity
    _assert_expected(identity.run_fingerprint, expected_run_fingerprint, "run fingerprint")
    _assert_expected(
        identity.m3b_export_fingerprint,
        expected_dataset_fingerprint,
        "M3B dataset fingerprint",
    )
    _assert_expected(
        identity.task_token_architecture_fingerprint,
        expected_architecture_fingerprint,
        "TaskToken architecture fingerprint",
    )
    record = next(
        (
            item
            for item in manifest.checkpoints
            if item.checkpoint_fingerprint == expected_checkpoint_fingerprint
        ),
        None,
    )
    if record is None:
        raise M42EvaluationError("selected TaskToken checkpoint is absent from training manifest")
    return load_task_token_checkpoint_context(
        root,
        expected_identity=identity,
        checkpoint_relative_path=record.relative_path,
        expected_checkpoint_fingerprint=expected_checkpoint_fingerprint,
        expected_architecture_fingerprint=identity.task_token_architecture_fingerprint,
    )


def load_factor_film_training_checkpoint_context(
    run_root: str | Path,
    *,
    expected_checkpoint_fingerprint: str,
    expected_run_fingerprint: str | None = None,
    expected_dataset_fingerprint: str | None = None,
    expected_architecture_fingerprint: str | None = None,
) -> M42CheckpointContext:
    """Strictly load one FactorFiLM checkpoint from its immutable training manifest."""
    from langmani.policies.act_factor_film_adapter import (
        FactorFiLMACTPolicy,
        validate_factor_film_policy_structure,
    )
    from langmani.policies.act_factor_film_types import FactorFiLMTrainingManifest

    root = Path(run_root).resolve()
    manifest = FactorFiLMTrainingManifest.from_dict(
        _read_json_object(root / "run_manifest.json", "FactorFiLM training manifest")
    )
    if not manifest.training_complete:
        raise M42EvaluationError("FactorFiLM checkpoint evaluation requires completed training")
    identity = manifest.identity
    architecture_fingerprint = manifest.architecture_identity.architecture_fingerprint
    _assert_expected(identity.run_fingerprint, expected_run_fingerprint, "run fingerprint")
    _assert_expected(
        identity.m3b_export_fingerprint,
        expected_dataset_fingerprint,
        "M3B dataset fingerprint",
    )
    _assert_expected(
        architecture_fingerprint,
        expected_architecture_fingerprint,
        "FactorFiLM architecture fingerprint",
    )
    record = next(
        (
            item
            for item in manifest.checkpoints
            if item.checkpoint_fingerprint == expected_checkpoint_fingerprint
        ),
        None,
    )
    if record is None:
        raise M42EvaluationError("selected FactorFiLM checkpoint is absent from training manifest")
    loaded = load_act_checkpoint(
        run_root=root,
        checkpoint_relative_path=record.relative_path,
        expected_identity=cast(Any, identity),
        restore_rng=False,
        policy_class=FactorFiLMACTPolicy,
    )
    if loaded.record.checkpoint_fingerprint != expected_checkpoint_fingerprint:
        raise M42EvaluationError("loaded FactorFiLM checkpoint differs from selection")
    if not isinstance(loaded.policy, FactorFiLMACTPolicy):
        raise M42EvaluationError("FactorFiLM strict reload returned the wrong policy type")
    validate_factor_film_policy_structure(loaded.policy)
    descriptor = M42CheckpointDescriptor(
        policy_kind=M42PolicyKind.FACTOR_FILM,
        run_fingerprint=identity.run_fingerprint,
        checkpoint_fingerprint=expected_checkpoint_fingerprint,
        checkpoint_relative_path=record.relative_path,
        dataset_fingerprint=identity.m3b_export_fingerprint,
        split_digest=identity.m3b_split_manifest_digest,
        statistics_fingerprint=identity.train_statistics_fingerprint,
        architecture_fingerprint=architecture_fingerprint,
        task_id=None,
        git_commit=identity.git_commit,
    )
    return M42CheckpointContext(descriptor=descriptor, loaded=loaded)


def _single_numpy(value: object, *, label: str) -> np.ndarray:
    candidate = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
    result = np.asarray(candidate)
    if result.shape[0] != 1:
        raise M42EvaluationError(f"{label} must contain num_envs=1")
    return np.array(result[0], copy=True)


def _single_bool_mapping(value: object) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        raise M42EvaluationError("M1 policy evaluation accessor must return a mapping")
    result: dict[str, bool] = {}
    for key, item in value.items():
        array = _single_numpy(item, label=str(key)).reshape(-1)
        if array.size != 1:
            raise M42EvaluationError(f"evaluation field {key!r} must be scalar")
        result[str(key)] = bool(array[0])
    return result


@dataclass(frozen=True, slots=True)
class _DiagnosticSnapshot:
    step: int
    target_position: tuple[float, float, float]
    target_velocity: tuple[float, float, float]
    bin_center: tuple[float, float, float]
    target_to_bin_distance: float
    object_is_grasped: tuple[bool, bool, bool]
    evaluation: Mapping[str, bool]
    action: tuple[float, ...] | None
    semantic_interaction: Mapping[str, object] | None

    def semantic_interaction_dict(self) -> dict[str, object]:
        """Return compact per-step privileged evidence outside the policy path."""
        if self.semantic_interaction is None:
            raise M42EvaluationError("semantic interaction capture was not authorized")
        return dict(self.semantic_interaction)


class _DiagnosticEnvironmentProxy:
    """Observe post-action privileged state without exposing it to the policy."""

    def __init__(self, env: object, *, capture_semantic_interactions: bool = False) -> None:
        self._env = env
        self._base = getattr(env, "unwrapped", env)
        self._capture_semantic_interactions = capture_semantic_interactions
        self.snapshots: list[_DiagnosticSnapshot] = []

    @property
    def unwrapped(self) -> object:
        return self._base

    @property
    def spec(self) -> object:
        return getattr(self._env, "spec", None)

    def reset(self, *args: object, **kwargs: object) -> object:
        self.snapshots.clear()
        result = cast(Any, self._env).reset(*args, **kwargs)
        self._capture(step=0, action=None)
        return result

    def step(self, action: object) -> object:
        result = cast(Any, self._env).step(action)
        self._capture(step=len(self.snapshots), action=np.asarray(action))
        return result

    def _capture(self, *, step: int, action: np.ndarray | None) -> None:
        diagnostic_accessor = getattr(self._base, "get_policy_rollout_diagnostics", None)
        evaluation_accessor = getattr(self._base, "get_policy_rollout_evaluation", None)
        if not callable(diagnostic_accessor) or not callable(evaluation_accessor):
            raise M42EvaluationError("M1 environment lacks M4.2 rollout diagnostic accessors")
        diagnostic = diagnostic_accessor()
        if not isinstance(diagnostic, Mapping):
            raise M42EvaluationError("M1 rollout diagnostics must be a mapping")
        position = _single_numpy(diagnostic["target_position"], label="target_position")
        velocity = _single_numpy(
            diagnostic["target_linear_velocity"], label="target_linear_velocity"
        )
        center = _single_numpy(
            diagnostic["target_bin_floor_center"], label="target_bin_floor_center"
        )
        distance = _single_numpy(
            diagnostic["target_to_bin_center_distance"],
            label="target_to_bin_center_distance",
        ).reshape(-1)
        grasped = _single_numpy(diagnostic["object_is_grasped"], label="object_is_grasped").reshape(
            -1
        )
        if position.shape != (3,) or velocity.shape != (3,) or center.shape != (3,):
            raise M42EvaluationError("M1 target diagnostic vectors must be three-dimensional")
        if grasped.shape != (3,) or distance.size != 1:
            raise M42EvaluationError("M1 object grasp/distance diagnostics are malformed")
        action_tuple = None
        if action is not None:
            flattened = np.asarray(action, dtype=np.float32).reshape(-1)
            if flattened.shape != (8,) or not np.all(np.isfinite(flattened)):
                raise M42EvaluationError("executed M4.2 action must be finite float[8]")
            action_tuple = tuple(float(item) for item in flattened)
        evaluation = _single_bool_mapping(evaluation_accessor())
        semantic_interaction: dict[str, object] | None = None
        if self._capture_semantic_interactions:
            cube_positions = _single_numpy(diagnostic["cube_positions"], label="cube_positions")
            cube_orientations = _single_numpy(
                diagnostic["cube_orientations"], label="cube_orientations"
            )
            cube_velocities = _single_numpy(
                diagnostic["cube_linear_velocities"], label="cube_linear_velocities"
            )
            cube_angular_velocities = _single_numpy(
                diagnostic["cube_angular_velocities"], label="cube_angular_velocities"
            )
            bin_centers = _single_numpy(diagnostic["bin_floor_centers"], label="bin_floor_centers")
            tcp_position = _single_numpy(diagnostic["tcp_position"], label="tcp_position")
            object_in_bin = _single_numpy(diagnostic["object_in_bin"], label="object_in_bin")
            if (
                cube_positions.shape != (3, 3)
                or cube_orientations.shape != (3, 4)
                or cube_velocities.shape != (3, 3)
                or cube_angular_velocities.shape != (3, 3)
                or bin_centers.shape != (2, 3)
                or tcp_position.shape != (3,)
                or object_in_bin.shape != (3, 2)
            ):
                raise M42EvaluationError("M1 semantic-interaction diagnostics are malformed")
            continuous = (
                cube_positions,
                cube_orientations,
                cube_velocities,
                cube_angular_velocities,
                bin_centers,
                tcp_position,
            )
            if not all(np.all(np.isfinite(value)) for value in continuous):
                raise M42EvaluationError("M1 semantic-interaction diagnostics must be finite")
            semantic_interaction = {
                "step": step,
                "cube_positions": cube_positions.tolist(),
                "cube_orientations": cube_orientations.tolist(),
                "cube_linear_velocities": cube_velocities.tolist(),
                "cube_angular_velocities": cube_angular_velocities.tolist(),
                "bin_floor_centers": bin_centers.tolist(),
                "tcp_position": tcp_position.tolist(),
                "object_is_grasped": [bool(item) for item in grasped],
                "object_in_bin": object_in_bin.astype(np.bool_).tolist(),
                "executed_action": None if action_tuple is None else list(action_tuple),
                "evaluation": evaluation,
            }
        self.snapshots.append(
            _DiagnosticSnapshot(
                step=step,
                target_position=tuple(float(item) for item in position),
                target_velocity=tuple(float(item) for item in velocity),
                bin_center=tuple(float(item) for item in center),
                target_to_bin_distance=float(distance[0]),
                object_is_grasped=tuple(bool(item) for item in grasped),
                evaluation=evaluation,
                action=action_tuple,
                semantic_interaction=semantic_interaction,
            )
        )

    def __getattr__(self, name: str) -> object:
        return getattr(self._env, name)


class _TaskTokenBatchAugmenter:
    def __call__(self, batch: dict[str, torch.Tensor], task_id: str) -> dict[str, torch.Tensor]:
        state = batch.get(STATE_FEATURE_KEY)
        if not isinstance(state, torch.Tensor) or state.dtype != torch.float32:
            raise M42EvaluationError("TaskToken rollout requires processed float32 Panda state")
        if tuple(state.shape) != (1, 9):
            raise M42EvaluationError("TaskToken rollout must keep Panda state at [1,9]")
        if TASK_TOKEN_FEATURE_KEY in batch:
            raise M42EvaluationError("TaskToken must be injected exactly once")
        result = dict(batch)
        result[TASK_TOKEN_FEATURE_KEY] = canonical_task_token(
            task_id, device=state.device
        ).unsqueeze(0)
        return result


class _FactorFiLMBatchAugmenter:
    def __call__(self, batch: dict[str, torch.Tensor], task_id: str) -> dict[str, torch.Tensor]:
        state = batch.get(STATE_FEATURE_KEY)
        if not isinstance(state, torch.Tensor) or state.dtype != torch.float32:
            raise M42EvaluationError("FactorFiLM rollout requires processed float32 Panda state")
        if tuple(state.shape) != (1, 9):
            raise M42EvaluationError("FactorFiLM rollout must keep Panda state at [1,9]")
        try:
            result = attach_factor_film_runtime_input(
                batch,
                factorized_runtime_input(task_id),
            )
        except (FactorFiLMContractError, TypeError, ValueError) as error:
            raise M42EvaluationError(f"invalid FactorFiLM rollout task: {error}") from error
        return cast(dict[str, torch.Tensor], result)


class _M42RawActionTransform:
    """Audit exact raw-action failures before the unchanged M4.1 bound processor."""

    def __init__(self, env: object, *, binary: bool) -> None:
        self.processor = BinaryGripperEnvPostprocessorV0.from_environment(env) if binary else None
        self.nan_count = 0
        self.inf_count = 0
        self.malformed_action_count = 0

    def reset(self) -> None:
        self.nan_count = 0
        self.inf_count = 0
        self.malformed_action_count = 0
        if self.processor is not None:
            self.processor.reset()

    def __call__(self, action: torch.Tensor, *, rollout_step: int) -> torch.Tensor:
        if (
            not isinstance(action, torch.Tensor)
            or tuple(action.shape) != (1, 8)
            or not action.is_floating_point()
        ):
            self.malformed_action_count += 1
            raise ActionBoundProcessingError(
                ActionBoundErrorKind.MALFORMED_ACTION,
                "M4.2 raw environment action must be one floating tensor with shape [1,8]",
            )
        if not bool(torch.isfinite(action).all()):
            self.nan_count += int(torch.count_nonzero(torch.isnan(action)))
            self.inf_count += int(torch.count_nonzero(torch.isinf(action)))
            raise ActionBoundProcessingError(
                ActionBoundErrorKind.NONFINITE_ACTION,
                "M4.2 raw environment action contains NaN or infinity",
            )
        return (
            action
            if self.processor is None
            else self.processor.transform(action, rollout_step=rollout_step)
        )


def _task_conditioner(state: np.ndarray, task_id: str) -> np.ndarray:
    return cast(np.ndarray, append_canonical_task_onehot(state, task_id))


def _sign_transition_metrics(actions: Sequence[tuple[float, ...]]) -> tuple[int, int]:
    signs = tuple(action[7] >= 0.0 for action in actions)
    release = sum((not left) and right for left, right in zip(signs, signs[1:], strict=False))
    grasp = sum(left and (not right) for left, right in zip(signs, signs[1:], strict=False))
    return release, grasp


def _post_grasp_diagnostics(
    *,
    snapshots: Sequence[_DiagnosticSnapshot],
    rollout: RolloutEpisodeResult,
    checkpoint_fingerprint: str,
    task_spec: Mapping[str, object],
    execution_horizon: int,
) -> tuple[dict[str, object], PostGraspFailureRecord | None]:
    if not snapshots:
        raise M42EvaluationError("post-grasp analysis requires at least the reset snapshot")
    initial = snapshots[0]
    target_index = CANONICAL_TASK_IDS.index(rollout.task_id) // 2
    first_grasp = next(
        (item.step for item in snapshots if item.object_is_grasped[target_index]), None
    )
    actions = tuple(item.action for item in snapshots if item.action is not None)
    release_transitions, grasp_transitions = _sign_transition_metrics(actions)
    release_step = next(
        (
            item.step
            for item in snapshots
            if first_grasp is not None
            and item.step >= first_grasp
            and item.action is not None
            and item.action[7] >= 0.0
        ),
        None,
    )
    maximum_height = max(item.target_position[2] for item in snapshots)
    closest_distance = min(item.target_to_bin_distance for item in snapshots)
    ever_lifted = maximum_height >= initial.target_position[2] + M42_LIFT_DELTA_METERS
    near_destination = tuple(
        item
        for item in snapshots
        if math.dist(item.target_position[:2], item.bin_center[:2])
        <= M42_DESTINATION_NEAR_XY_METERS
    )
    ever_transported = bool(near_destination)
    ever_descended = any(
        item.target_position[2] <= item.bin_center[2] + M42_DESCENT_HEIGHT_METERS
        for item in near_destination
    )
    released_inside = any(
        item.step >= (release_step or 10**9) and item.evaluation.get("target_in_target_bin", False)
        for item in snapshots
    )
    success_seen = any(item.evaluation.get("success", False) for item in snapshots)
    wrong_interaction = rollout.wrong_object_grasped_any or rollout.wrong_object_in_target_bin
    trace = PostGraspTrace(
        success=rollout.success,
        environment_failure=rollout.inference_failure
        or rollout.status.value == "environment_failure",
        wrong_object_interaction=wrong_interaction,
        ever_grasped_target=first_grasp is not None,
        ever_lifted_target=ever_lifted,
        ever_transported_to_destination=ever_transported,
        ever_descended_at_destination=ever_descended,
        release_command_observed=release_step is not None,
        released_inside_success_region=released_inside,
        final_target_static=snapshots[-1].evaluation.get("target_is_static", False),
        success_observed_before_final_step=success_seen and not rollout.success,
        timed_out=rollout.timeout,
    )
    phase = classify_post_grasp_phase(trace)
    query_steps = tuple(range(1, rollout.episode_steps + 1, execution_horizon))
    requery_after_grasp = (
        sum(step > first_grasp for step in query_steps) if first_grasp is not None else 0
    )
    near_steps = {item.step for item in near_destination}
    requery_near_destination = sum(step in near_steps for step in query_steps)
    details: dict[str, object] = {
        "version": M42_POST_GRASP_DIAGNOSTIC_VERSION,
        "phase": phase.value,
        "first_target_grasp_step": first_grasp,
        "maximum_target_height": maximum_height,
        "closest_target_to_bin_distance": closest_distance,
        "first_release_command_step": release_step,
        "release_sign_transitions": release_transitions,
        "grasp_sign_transitions": grasp_transitions,
        "policy_requeries_after_target_grasp": requery_after_grasp,
        "policy_requeries_near_destination": requery_near_destination,
        "steps_from_first_grasp_to_success": (
            rollout.episode_steps - first_grasp
            if first_grasp is not None and rollout.success
            else None
        ),
        "steps_from_first_grasp_to_timeout": (
            rollout.episode_steps - first_grasp
            if first_grasp is not None and rollout.timeout
            else None
        ),
        "final_target_position": list(snapshots[-1].target_position),
        "final_target_velocity": list(snapshots[-1].target_velocity),
    }
    failure_record = None
    if first_grasp is not None and not rollout.success:
        failure_record = PostGraspFailureRecord(
            scene_seed=rollout.scene_seed,
            task_spec=task_spec,
            selected_checkpoint_fingerprint=checkpoint_fingerprint,
            execution_horizon=execution_horizon,
            first_target_grasp_step=first_grasp,
            maximum_target_height=maximum_height,
            closest_target_to_bin_distance=closest_distance,
            first_release_command_step=release_step,
            final_target_position=snapshots[-1].target_position,
            final_target_velocity=snapshots[-1].target_velocity,
            final_environment_evaluation=rollout.final_evaluation,
            inferred_failure_phase=phase,
            projection_summary=rollout.action_projection_summary,
        )
    return details, failure_record


@dataclass(frozen=True, slots=True)
class M42EpisodeReport:
    scheduled_episode_id: str
    schedule_id: str
    episode_index: int
    model_label: str
    rollout: RolloutEpisodeResult
    execution_horizon_metrics: Mapping[str, object]
    post_grasp_diagnostics: Mapping[str, object]
    binary_gripper_audits: tuple[Mapping[str, object], ...]
    post_grasp_failure_record: PostGraspFailureRecord | None
    runtime_projection_audits: tuple[Mapping[str, object], ...] = ()
    semantic_interaction_snapshots: tuple[Mapping[str, object], ...] = ()

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "scheduled_episode_id": self.scheduled_episode_id,
            "schedule_id": self.schedule_id,
            "episode_index": self.episode_index,
            "model_label": self.model_label,
            "rollout": self.rollout.to_dict(),
            "execution_horizon_metrics": dict(self.execution_horizon_metrics),
            "post_grasp_diagnostics": dict(self.post_grasp_diagnostics),
            "binary_gripper_audits": [dict(item) for item in self.binary_gripper_audits],
            "runtime_projection_audits": [dict(item) for item in self.runtime_projection_audits],
            "post_grasp_failure_record": (
                None
                if self.post_grasp_failure_record is None
                else self.post_grasp_failure_record.to_dict()
            ),
        }
        if self.semantic_interaction_snapshots:
            result["semantic_interaction_snapshots"] = [
                dict(item) for item in self.semantic_interaction_snapshots
            ]
        return result

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())


def _sum_projection(reports: Sequence[M42EpisodeReport]) -> dict[str, object]:
    summaries = [report.rollout.action_projection_summary for report in reports]
    horizon_metrics = [report.execution_horizon_metrics for report in reports]
    nan_count = sum(
        int(item.get("predicted_chunk_nan_count", 0))
        + int(item.get("raw_environment_nan_count", 0))
        for item in horizon_metrics
    )
    inf_count = sum(
        int(item.get("predicted_chunk_inf_count", 0))
        + int(item.get("raw_environment_inf_count", 0))
        for item in horizon_metrics
    )
    malformed_action_count = sum(
        int(item.get("malformed_chunk_count", 0))
        + int(item.get("raw_environment_malformed_action_count", 0))
        for item in horizon_metrics
    )
    dimensions = [0] * 8
    for summary in summaries:
        counts = summary.get("per_action_dimension_projection_counts")
        if not isinstance(counts, Sequence) or len(counts) != 8:
            raise M42EvaluationError("rollout projection summary is malformed")
        for index, count in enumerate(counts):
            dimensions[index] += int(count)
    binary = [audit for report in reports for audit in report.binary_gripper_audits]
    if binary:
        raw_audits = [cast(Mapping[str, object], item["raw_bound_audit"]) for item in binary]
        raw_dimension_counts = [0] * 8
        for audit in raw_audits:
            mask = np.asarray(audit["violation_mask"], dtype=np.bool_).reshape(-1, 8)
            for index, count in enumerate(mask.sum(axis=0)):
                raw_dimension_counts[index] += int(count)
        raw_out_of_bounds_action_count = sum(bool(item["was_projected"]) for item in raw_audits)
        raw_out_of_bounds_component_count = sum(
            int(item["projected_component_count"]) for item in raw_audits
        )
        maximum_raw_bound_excess = max(
            float(item["maximum_absolute_bound_excess"]) for item in raw_audits
        )
    else:
        raw_dimension_counts = list(dimensions)
        raw_out_of_bounds_action_count = sum(
            int(item["projected_action_count"]) for item in summaries
        )
        raw_out_of_bounds_component_count = sum(dimensions)
        maximum_raw_bound_excess = max(
            (float(item["maximum_bound_excess"]) for item in summaries), default=0.0
        )
    return {
        "total_policy_actions": sum(int(item["total_policy_actions"]) for item in summaries),
        "projected_action_count": sum(int(item["projected_action_count"]) for item in summaries),
        "projected_component_count": sum(
            int(item["projected_component_count"]) for item in summaries
        ),
        "runtime_projected_action_count": sum(
            int(item["projected_action_count"]) for item in summaries
        ),
        "runtime_projected_component_count": sum(
            int(item["projected_component_count"]) for item in summaries
        ),
        "arm_projected_component_count": sum(dimensions[:7]),
        "gripper_projected_component_count": dimensions[7],
        "per_action_dimension_projection_counts": dimensions,
        "maximum_runtime_projection_correction": max(
            (float(item["maximum_linf_correction"]) for item in summaries), default=0.0
        ),
        "raw_out_of_bounds_action_count": raw_out_of_bounds_action_count,
        "raw_out_of_bounds_component_count": raw_out_of_bounds_component_count,
        "raw_per_action_dimension_violation_counts": raw_dimension_counts,
        "maximum_raw_bound_excess": maximum_raw_bound_excess,
        "binary_changed_action_count": sum(bool(item["changed"]) for item in binary),
        "binary_changed_command_count": sum(int(item["changed_command_count"]) for item in binary),
        "maximum_binary_gripper_change": max(
            (float(item["maximum_absolute_gripper_change"]) for item in binary), default=0.0
        ),
        "raw_action_metrics_validated": all(
            not bool(item["any_nonfinite_action"]) and not bool(item["any_malformed_action"])
            for item in summaries
        )
        and nan_count == 0
        and inf_count == 0
        and malformed_action_count == 0,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "malformed_action_count": malformed_action_count,
        "nonfinite_action_episode_count": sum(
            bool(item["any_nonfinite_action"]) for item in summaries
        ),
        "malformed_action_episode_count": sum(
            bool(item["any_malformed_action"]) for item in summaries
        ),
        "raw_action_bounds_validated": raw_out_of_bounds_component_count == 0,
        "arm_action_bounds_validated": sum(dimensions[:7]) == 0,
        "runtime_action_bounds_validated": True,
    }


def _aggregate(reports: Sequence[M42EpisodeReport]) -> dict[str, object]:
    if not reports:
        raise M42EvaluationError("M4.2 benchmark cannot aggregate zero episodes")
    phase_counts = {phase.value: 0 for phase in PostGraspPhase}
    per_task: dict[str, dict[str, int | float]] = {}
    inference: list[float] = []
    environment: list[float] = []
    for report in reports:
        rollout = report.rollout
        phase = cast(str, report.post_grasp_diagnostics["phase"])
        phase_counts[phase] += 1
        item = per_task.setdefault(rollout.task_id, {"episodes": 0, "successes": 0})
        item["episodes"] = int(item["episodes"]) + 1
        item["successes"] = int(item["successes"]) + int(rollout.success)
        inference.extend(rollout.inference_latency_ms)
        environment.extend(rollout.environment_step_latency_ms)
    for item in per_task.values():
        item["success_rate"] = int(item["successes"]) / int(item["episodes"])
    successes = sum(report.rollout.success for report in reports)
    interval = wilson_interval(successes, len(reports))
    successful_steps = [
        report.rollout.episode_steps for report in reports if report.rollout.success
    ]
    strict_raw_unprojected = sum(
        report.rollout.success
        and not any(
            bool(cast(Mapping[str, object], item["raw_bound_audit"])["was_projected"])
            for item in report.binary_gripper_audits
        )
        and not any(bool(item["was_projected"]) for item in report.runtime_projection_audits)
        for report in reports
    )
    strict_raw_unmodified = sum(
        report.rollout.success
        and not any(bool(item["changed"]) for item in report.binary_gripper_audits)
        and not any(bool(item["was_projected"]) for item in report.runtime_projection_audits)
        for report in reports
    )
    return {
        "episode_count": len(reports),
        "successes": successes,
        "success_rate": successes / len(reports),
        "success_wilson_interval": {
            "successes": interval.successes,
            "trials": interval.trials,
            "confidence_level": interval.confidence_level,
            "lower": interval.lower,
            "upper": interval.upper,
        },
        "per_task": per_task,
        "target_grasped_count": sum(report.rollout.target_grasped_any for report in reports),
        "post_grasp_success_count": sum(
            report.rollout.target_grasped_any and report.rollout.success for report in reports
        ),
        "wrong_object_interaction_count": sum(
            report.rollout.wrong_object_grasped_any or report.rollout.wrong_object_in_target_bin
            for report in reports
        ),
        "wrong_object_grasp_count": sum(
            report.rollout.wrong_object_grasped_any for report in reports
        ),
        "target_in_wrong_bin_count": sum(report.rollout.target_in_wrong_bin for report in reports),
        "wrong_object_in_target_bin_count": sum(
            report.rollout.wrong_object_in_target_bin for report in reports
        ),
        "target_off_table_count": sum(report.rollout.target_off_table for report in reports),
        "timeout_count": sum(report.rollout.timeout for report in reports),
        "invalid_action_count": sum(report.rollout.invalid_action for report in reports),
        # M4.2 evaluates this against the original policy action, before an
        # optional binary gripper transform can hide a raw bound violation.
        "strict_unprojected_success_count": strict_raw_unprojected,
        "strict_runtime_unprojected_success_count": sum(
            bool(report.rollout.strict_unprojected_success) for report in reports
        ),
        "strict_raw_unmodified_success_count": strict_raw_unmodified,
        "successful_episode_steps": successful_steps,
        "median_successful_episode_steps": (
            float(median(successful_steps)) if successful_steps else None
        ),
        "policy_query_count": sum(
            int(report.execution_horizon_metrics["policy_query_count"]) for report in reports
        ),
        "release_sign_transitions": sum(
            int(report.post_grasp_diagnostics["release_sign_transitions"]) for report in reports
        ),
        "grasp_sign_transitions": sum(
            int(report.post_grasp_diagnostics["grasp_sign_transitions"]) for report in reports
        ),
        "unnecessary_gripper_sign_transitions": sum(
            max(0, int(report.post_grasp_diagnostics["release_sign_transitions"]) - 1)
            + max(0, int(report.post_grasp_diagnostics["grasp_sign_transitions"]) - 1)
            for report in reports
        ),
        "phase_counts": phase_counts,
        "post_grasp_timeout_count": phase_counts[PostGraspPhase.TIMED_OUT_AFTER_TARGET_GRASP.value],
        "inference_latency_ms": latency_percentiles(tuple(inference)),
        "environment_step_latency_ms": latency_percentiles(tuple(environment)),
        "action_metrics": _sum_projection(reports),
    }


def _runtime_projection_audit_count_valid(
    rollout: RolloutEpisodeResult, runtime_audit_count: int
) -> bool:
    """Accept one missing audit only for a classified pre-bound action failure."""

    if isinstance(runtime_audit_count, bool) or runtime_audit_count < 0:
        return False
    summary = rollout.action_projection_summary
    total_policy_actions = int(summary["total_policy_actions"])
    if runtime_audit_count == total_policy_actions:
        return True
    classified_pre_bound_failure = rollout.invalid_action and (
        bool(summary["any_nonfinite_action"]) or bool(summary["any_malformed_action"])
    )
    return classified_pre_bound_failure and runtime_audit_count + 1 == total_policy_actions


@dataclass(frozen=True, slots=True)
class M42BenchmarkReport:
    schema_version: str
    schedule_id: str
    schedule_fingerprint: str
    model_label: str
    checkpoint: M42CheckpointDescriptor
    execution_horizon: int
    gripper_mode: GripperRuntimeMode
    runtime_fingerprint: str
    episodes: tuple[M42EpisodeReport, ...]
    aggregate: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "schedule_id": self.schedule_id,
            "schedule_fingerprint": self.schedule_fingerprint,
            "model_label": self.model_label,
            "checkpoint": self.checkpoint.to_dict(),
            "execution_horizon": self.execution_horizon,
            "gripper_mode": self.gripper_mode.value,
            "runtime_fingerprint": self.runtime_fingerprint,
            "episodes": [item.to_dict() for item in self.episodes],
            "aggregate": dict(self.aggregate),
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def to_runtime_ablation_result(self, config_fingerprint: str) -> RuntimeAblationResult:
        aggregate = self.aggregate
        action_metrics = cast(Mapping[str, object], aggregate["action_metrics"])
        return RuntimeAblationResult(
            config_fingerprint=config_fingerprint,
            schedule_id=self.schedule_id,
            schedule_fingerprint=self.schedule_fingerprint,
            model_label=self.model_label,
            task_id=self.checkpoint.task_id,
            execution_horizon=self.execution_horizon,
            gripper_mode=self.gripper_mode,
            episode_count=int(aggregate["episode_count"]),
            successes=int(aggregate["successes"]),
            post_grasp_timeouts=int(aggregate["post_grasp_timeout_count"]),
            wrong_object_interactions=int(aggregate["wrong_object_interaction_count"]),
            wrong_object_grasp_count=int(aggregate["wrong_object_grasp_count"]),
            wrong_object_in_target_bin_count=int(aggregate["wrong_object_in_target_bin_count"]),
            target_in_wrong_bin_count=int(aggregate["target_in_wrong_bin_count"]),
            target_off_table_count=int(aggregate["target_off_table_count"]),
            invalid_action_count=int(aggregate["invalid_action_count"]),
            successful_episode_steps=tuple(
                int(item) for item in cast(Sequence[object], aggregate["successful_episode_steps"])
            ),
            policy_query_count=int(aggregate["policy_query_count"]),
            release_sign_transitions=int(aggregate["release_sign_transitions"]),
            grasp_sign_transitions=int(aggregate["grasp_sign_transitions"]),
            unnecessary_gripper_sign_transitions=int(
                aggregate["unnecessary_gripper_sign_transitions"]
            ),
            action_metrics=action_metrics,
            latency_metrics={
                "inference_latency_ms": aggregate["inference_latency_ms"],
                "environment_step_latency_ms": aggregate["environment_step_latency_ms"],
            },
            report_fingerprint=self.fingerprint,
        )


def _validate_episode_schedule(
    episodes: Sequence[M42ScheduledEpisode],
    *,
    schedule_fingerprint: str,
    policy_kind: M42PolicyKind,
    task_id: str | None,
    final_authorization: FinalScheduleAuthorization | None,
) -> str:
    if not episodes:
        raise M42EvaluationError("M4.2 benchmark requires scheduled episodes")
    schedule_ids = {item.schedule_id for item in episodes}
    if len(schedule_ids) != 1:
        raise M42EvaluationError("benchmark episodes cannot mix schedules")
    schedule_id = next(iter(schedule_ids))
    if policy_kind is M42PolicyKind.FACTOR_FILM and schedule_id != M42_DEV_SCHEDULE_ID:
        raise M42EvaluationError(
            "FactorFiLM target-development permits only the locked m42_dev_v0 schedule"
        )
    schedule = load_locked_schedule(schedule_id)
    if schedule.schedule_fingerprint != schedule_fingerprint:
        raise M42EvaluationError("benchmark schedule fingerprint differs from committed lock")
    if schedule_id == M42_FINAL_SCHEDULE_ID:
        if final_authorization is None:
            raise M42EvaluationError("m42_final_v0 evaluation requires explicit authorization")
        final_authorization.require_valid(schedule)
    elif schedule_id != M42_DEV_SCHEDULE_ID or final_authorization is not None:
        raise M42EvaluationError("invalid development/final schedule authorization")
    expected_count = len(schedule.ordered_scene_seeds) * (
        1 if policy_kind is M42PolicyKind.PER_TASK else len(schedule.ordered_task_specs)
    )
    if len(episodes) != expected_count:
        raise M42EvaluationError("benchmark must contain one complete declared schedule view")
    if policy_kind is M42PolicyKind.PER_TASK and (
        task_id is None or {item.task_id for item in episodes} != {task_id}
    ):
        raise M42EvaluationError("PerTask benchmark contains the wrong scheduled task")
    expected_pairs = tuple(
        (scene_seed, task_spec)
        for scene_seed in schedule.ordered_scene_seeds
        for task_spec in schedule.ordered_task_specs
        if policy_kind is not M42PolicyKind.PER_TASK or stable_task_id(task_spec) == task_id
    )
    actual_pairs = tuple((item.scene_seed, item.task_spec) for item in episodes)
    if actual_pairs != expected_pairs:
        raise M42EvaluationError("benchmark episode ordering differs from the committed lock")
    if any(
        item.scene_id != stable_scene_id(item.scene_seed)
        or item.task_id != stable_task_id(item.task_spec)
        for item in episodes
    ):
        raise M42EvaluationError("benchmark episode semantic identifiers are inconsistent")
    return schedule_id


def _run_checkpoint_benchmark(
    *,
    env: object,
    checkpoint: M42CheckpointContext,
    episodes: Sequence[M42ScheduledEpisode],
    schedule_id: str,
    schedule_fingerprint: str,
    split: EvaluationSplit,
    execution_horizon: int,
    gripper_mode: GripperRuntimeMode,
    model_label: str,
    maximum_episode_steps: int = 200,
) -> M42BenchmarkReport:
    capture_semantic_interactions = (
        checkpoint.descriptor.policy_kind is M42PolicyKind.FACTOR_FILM
        and schedule_id == M42_DEV_SCHEDULE_ID
    )
    proxy = _DiagnosticEnvironmentProxy(
        env, capture_semantic_interactions=capture_semantic_interactions
    )
    horizon_policy = ExecutionHorizonPolicyV0(
        cast(Any, checkpoint.loaded.policy), ExecutionHorizonConfig(execution_horizon)
    )
    raw_action_transform = _M42RawActionTransform(
        proxy, binary=gripper_mode is GripperRuntimeMode.BINARY
    )
    variant = {
        M42PolicyKind.PER_TASK: ActVariant.PER_TASK,
        M42PolicyKind.STATE_ONEHOT: ActVariant.MIXED_TASK_ONEHOT,
        M42PolicyKind.TASK_TOKEN: ActVariant.MIXED_UNCONDITIONED,
        M42PolicyKind.FACTOR_FILM: ActVariant.MIXED_UNCONDITIONED,
    }[checkpoint.descriptor.policy_kind]
    runtime_fingerprint = _fingerprint(
        {
            "schema_version": M42_EVALUATION_SCHEMA_VERSION,
            "checkpoint_descriptor_fingerprint": checkpoint.descriptor.fingerprint,
            "execution_horizon": execution_horizon,
            "gripper_mode": gripper_mode.value,
            "binary_processor_fingerprint": (
                None
                if raw_action_transform.processor is None
                else raw_action_transform.processor.runtime_fingerprint
            ),
        }
    )
    projection_records: dict[str, list[ActionProjectionRecord]] = {}

    def retain_projection_record(evaluation_id: str, record: ActionProjectionRecord) -> None:
        projection_records.setdefault(evaluation_id, []).append(record)

    adapter = ActManiSkillRolloutAdapter(
        env=proxy,
        policy=horizon_policy,
        preprocessor=cast(Any, checkpoint.loaded.preprocessor),
        postprocessor=cast(Any, checkpoint.loaded.postprocessor),
        variant=variant,
        run_fingerprint=checkpoint.descriptor.run_fingerprint,
        checkpoint_fingerprint=checkpoint.descriptor.checkpoint_fingerprint,
        schedule_digest=schedule_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
        action_bound_config=ActionBoundConfig(mode=ActionBoundMode.PROJECT),
        evaluation=ActEvaluationConfig(maximum_episode_steps=maximum_episode_steps),
        task_conditioner=(
            _task_conditioner
            if checkpoint.descriptor.policy_kind is M42PolicyKind.STATE_ONEHOT
            else None
        ),
        processed_observation_augmenter=(
            _TaskTokenBatchAugmenter()
            if checkpoint.descriptor.policy_kind is M42PolicyKind.TASK_TOKEN
            else (
                _FactorFiLMBatchAugmenter()
                if checkpoint.descriptor.policy_kind is M42PolicyKind.FACTOR_FILM
                else None
            )
        ),
        pre_bound_action_transform=raw_action_transform,
        projection_record_sink=retain_projection_record,
    )
    reports: list[M42EpisodeReport] = []
    for scheduled in episodes:
        evaluation_id = _fingerprint(
            {
                "schema_version": M42_EVALUATION_SCHEMA_VERSION,
                "schedule_fingerprint": schedule_fingerprint,
                "episode_index": scheduled.episode_index,
                "checkpoint_fingerprint": checkpoint.descriptor.checkpoint_fingerprint,
                "runtime_fingerprint": runtime_fingerprint,
            }
        )
        rollout = adapter.run_episode(
            evaluation_id=evaluation_id,
            split=split,
            scene_seed=scheduled.scene_seed,
            task_spec=scheduled.task_spec,
        )
        metrics = {
            **horizon_policy.metrics().to_dict(),
            "raw_environment_nan_count": raw_action_transform.nan_count,
            "raw_environment_inf_count": raw_action_transform.inf_count,
            "raw_environment_malformed_action_count": (raw_action_transform.malformed_action_count),
        }
        diagnostics, failure = _post_grasp_diagnostics(
            snapshots=proxy.snapshots,
            rollout=rollout,
            checkpoint_fingerprint=checkpoint.descriptor.checkpoint_fingerprint,
            task_spec=scheduled.task_spec.to_dict(),
            execution_horizon=execution_horizon,
        )
        binary_audits = (
            ()
            if raw_action_transform.processor is None
            else tuple(item.to_dict() for item in raw_action_transform.processor.audit_records)
        )
        runtime_audits = tuple(item.to_dict() for item in projection_records.pop(evaluation_id, ()))
        if not _runtime_projection_audit_count_valid(rollout, len(runtime_audits)):
            raise M42EvaluationError(
                "per-step runtime projection evidence differs from its rollout summary"
            )
        reports.append(
            M42EpisodeReport(
                scheduled_episode_id=evaluation_id,
                schedule_id=schedule_id,
                episode_index=scheduled.episode_index,
                model_label=model_label,
                rollout=rollout,
                execution_horizon_metrics=metrics,
                post_grasp_diagnostics=diagnostics,
                binary_gripper_audits=binary_audits,
                post_grasp_failure_record=failure,
                runtime_projection_audits=runtime_audits,
                semantic_interaction_snapshots=(
                    tuple(item.semantic_interaction_dict() for item in proxy.snapshots)
                    if capture_semantic_interactions
                    else ()
                ),
            )
        )
    if projection_records:
        raise M42EvaluationError("orphaned runtime projection records remain after benchmark")
    episode_tuple = tuple(reports)
    return M42BenchmarkReport(
        schema_version=M42_EVALUATION_SCHEMA_VERSION,
        schedule_id=schedule_id,
        schedule_fingerprint=schedule_fingerprint,
        model_label=model_label,
        checkpoint=checkpoint.descriptor,
        execution_horizon=execution_horizon,
        gripper_mode=gripper_mode,
        runtime_fingerprint=runtime_fingerprint,
        episodes=episode_tuple,
        aggregate=_aggregate(episode_tuple),
    )


def run_m42_checkpoint_benchmark(
    *,
    env: object,
    checkpoint: M42CheckpointContext,
    episodes: Sequence[M42ScheduledEpisode],
    schedule_fingerprint: str,
    execution_horizon: int,
    gripper_mode: GripperRuntimeMode,
    model_label: str,
    maximum_episode_steps: int = 200,
    final_authorization: FinalScheduleAuthorization | None = None,
) -> M42BenchmarkReport:
    """Run one frozen checkpoint over one complete, authorized M4.2 schedule."""
    schedule_id = _validate_episode_schedule(
        episodes,
        schedule_fingerprint=schedule_fingerprint,
        policy_kind=checkpoint.descriptor.policy_kind,
        task_id=checkpoint.descriptor.task_id,
        final_authorization=final_authorization,
    )
    return _run_checkpoint_benchmark(
        env=env,
        checkpoint=checkpoint,
        episodes=episodes,
        schedule_id=schedule_id,
        schedule_fingerprint=schedule_fingerprint,
        split=(
            EvaluationSplit.DEVELOPMENT
            if schedule_id == M42_DEV_SCHEDULE_ID
            else EvaluationSplit.FRESH_SEED
        ),
        execution_horizon=execution_horizon,
        gripper_mode=gripper_mode,
        model_label=model_label,
        maximum_episode_steps=maximum_episode_steps,
    )


def run_m42_validation_benchmark(
    *,
    env: object,
    checkpoint: M42CheckpointContext,
    schedule_records: Sequence[Mapping[str, object]],
    validation_schedule_fingerprint: str,
    execution_horizon: int,
    gripper_mode: GripperRuntimeMode,
    maximum_episode_steps: int = 200,
) -> M42BenchmarkReport:
    """Evaluate TaskToken or FactorFiLM on its exact frozen 36-episode validation view."""
    policy_kind = checkpoint.descriptor.policy_kind
    if policy_kind not in {M42PolicyKind.TASK_TOKEN, M42PolicyKind.FACTOR_FILM}:
        raise M42EvaluationError(
            "mixed checkpoint selection accepts only TaskToken or FactorFiLM models"
        )
    _require_digest(validation_schedule_fingerprint, "validation_schedule_fingerprint")
    if len(schedule_records) != 36:
        raise M42EvaluationError(
            "mixed checkpoint selection requires exactly 36 validation episodes"
        )
    canonical_records = [dict(item) for item in schedule_records]
    if policy_kind is M42PolicyKind.FACTOR_FILM:
        expected_fingerprint = _fingerprint(
            {
                "schema_version": "langmani-m43-factor-film-validation-schedule-v0",
                "m3b_export_fingerprint": checkpoint.descriptor.dataset_fingerprint,
                "split_manifest_digest": checkpoint.descriptor.split_digest,
                "source": "m3b_validation",
                "episodes": canonical_records,
            }
        )
    else:
        expected_fingerprint = _fingerprint(
            {
                "schema_version": "langmani-m42-m3b-validation-schedule-v0",
                "m3b_dataset_fingerprint": checkpoint.descriptor.dataset_fingerprint,
                "split_digest": checkpoint.descriptor.split_digest,
                "episodes": canonical_records,
            }
        )
    if expected_fingerprint != validation_schedule_fingerprint:
        raise M42EvaluationError("M3B validation schedule fingerprint is inconsistent")
    parsed: list[tuple[int, TaskSpec, str | None]] = []
    for index, record in enumerate(canonical_records):
        if policy_kind is M42PolicyKind.FACTOR_FILM:
            if set(record) != {"episode_index", "scene_group_id", "scene_seed", "task_id"}:
                raise M42EvaluationError(
                    "FactorFiLM validation records must use the exact schedule fields"
                )
            task_id = record.get("task_id")
            scene_group_id = record.get("scene_group_id")
            if (
                task_id not in CANONICAL_TASK_IDS
                or not isinstance(scene_group_id, str)
                or not (scene_group_id)
            ):
                raise M42EvaluationError("FactorFiLM validation semantic record is malformed")
            task_spec = CANONICAL_TASK_SPECS[CANONICAL_TASK_IDS.index(cast(str, task_id))]
        else:
            task_value = record.get("task_spec")
            if not isinstance(task_value, Mapping):
                raise M42EvaluationError("validation schedule TaskSpec is malformed")
            task_spec = TaskSpec.from_mapping(task_value)
            scene_group_id = None
        scene_seed = record.get("scene_seed")
        episode_index = record.get("episode_index")
        if (
            isinstance(scene_seed, bool)
            or not isinstance(scene_seed, int)
            or isinstance(episode_index, bool)
            or not isinstance(episode_index, int)
            or record.get("task_id") != stable_task_id(task_spec)
        ):
            raise M42EvaluationError("validation schedule semantic record is malformed")
        if index and episode_index <= int(canonical_records[index - 1]["episode_index"]):
            raise M42EvaluationError("validation episode indices must be strictly ordered")
        parsed.append((scene_seed, task_spec, scene_group_id))
    if policy_kind is M42PolicyKind.FACTOR_FILM:
        grouped: dict[str, list[tuple[int, TaskSpec]]] = {}
        for scene_seed, task_spec, scene_group_id in parsed:
            grouped.setdefault(cast(str, scene_group_id), []).append((scene_seed, task_spec))
        if len(grouped) != 6 or any(
            len(values) != 6
            or len({scene_seed for scene_seed, _ in values}) != 1
            or tuple(task_spec for _, task_spec in values) != CANONICAL_TASK_SPECS
            for values in grouped.values()
        ):
            raise M42EvaluationError(
                "FactorFiLM validation requires six complete counterfactual scene groups"
            )
        group_indices = {scene_group_id: index for index, scene_group_id in enumerate(grouped)}
    else:
        group_indices = {}
    episodes = tuple(
        M42ScheduledEpisode(
            schedule_id="m3b_validation_v0",
            episode_index=index,
            scene_index=(
                group_indices[cast(str, scene_group_id)]
                if policy_kind is M42PolicyKind.FACTOR_FILM
                else index // 6
            ),
            task_index=CANONICAL_TASK_IDS.index(stable_task_id(task_spec)),
            scene_seed=scene_seed,
            scene_id=stable_scene_id(scene_seed),
            task_spec=task_spec,
            task_id=stable_task_id(task_spec),
        )
        for index, (scene_seed, task_spec, scene_group_id) in enumerate(parsed)
    )
    if len({(item.scene_id, item.task_id) for item in episodes}) != 36:
        raise M42EvaluationError("M3B validation schedule contains duplicate semantic episodes")
    if {item.task_id for item in episodes} != set(CANONICAL_TASK_IDS):
        raise M42EvaluationError("M3B validation schedule omits a canonical task")
    return _run_checkpoint_benchmark(
        env=env,
        checkpoint=checkpoint,
        episodes=episodes,
        schedule_id="m3b_validation_v0",
        schedule_fingerprint=validation_schedule_fingerprint,
        split=EvaluationSplit.VALIDATION,
        execution_horizon=execution_horizon,
        gripper_mode=gripper_mode,
        model_label=policy_kind.value,
        maximum_episode_steps=maximum_episode_steps,
    )


def task_token_validation_result(
    report: M42BenchmarkReport,
    *,
    checkpoint_step: int,
    validation_split_digest: str,
    offline_validation_action_loss: float,
) -> TaskTokenValidationResult:
    """Project a complete closed-loop report into the validation-only ranking record."""
    if report.schedule_id != "m3b_validation_v0" or len(report.episodes) != 36:
        raise M42EvaluationError("TaskToken ranking requires one complete validation report")
    aggregate = report.aggregate
    return TaskTokenValidationResult(
        checkpoint_fingerprint=report.checkpoint.checkpoint_fingerprint,
        checkpoint_step=checkpoint_step,
        validation_schedule_fingerprint=report.schedule_fingerprint,
        validation_split_digest=validation_split_digest,
        episode_count=int(aggregate["episode_count"]),
        success_count=int(aggregate["successes"]),
        wrong_object_interaction_count=int(aggregate["wrong_object_interaction_count"]),
        target_off_table_count=int(aggregate["target_off_table_count"]),
        timeout_count=int(aggregate["timeout_count"]),
        offline_validation_action_loss=offline_validation_action_loss,
    )


def paired_episode_comparison(
    left: M42BenchmarkReport, right: M42BenchmarkReport
) -> dict[str, object]:
    """Compute a paired semantic-episode comparison on one unchanged schedule."""
    if left.schedule_fingerprint != right.schedule_fingerprint:
        raise M42EvaluationError("paired reports must share one schedule fingerprint")
    left_by_id = {
        (item.rollout.scene_id, item.rollout.task_id): item.rollout.success
        for item in left.episodes
    }
    right_by_id = {
        (item.rollout.scene_id, item.rollout.task_id): item.rollout.success
        for item in right.episodes
    }
    if left_by_id.keys() != right_by_id.keys():
        raise M42EvaluationError("paired reports do not cover identical semantic episodes")
    keys = tuple(sorted(left_by_id))
    metrics = paired_success_metrics(
        tuple(left_by_id[key] for key in keys), tuple(right_by_id[key] for key in keys)
    )
    return {
        "left_report_fingerprint": left.fingerprint,
        "right_report_fingerprint": right.fingerprint,
        "schedule_fingerprint": left.schedule_fingerprint,
        **metrics,
    }


def task_action_chunk_distances(
    chunks_by_task: Mapping[str, torch.Tensor | np.ndarray],
) -> dict[str, object]:
    """Measure all 15 counterfactual pair distances for one physical observation."""
    if set(chunks_by_task) != set(CANONICAL_TASK_IDS):
        raise M42EvaluationError("task sensitivity requires exactly the canonical six tasks")
    normalized: dict[str, np.ndarray] = {}
    for task_id in CANONICAL_TASK_IDS:
        value = chunks_by_task[task_id]
        array = (
            value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        )
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.shape != (50, 8) or not np.all(np.isfinite(array)):
            raise M42EvaluationError("task-sensitivity chunks must be finite [50,8]")
        normalized[task_id] = np.asarray(array, dtype=np.float64)
    pairs: list[dict[str, object]] = []
    for left_index, left in enumerate(CANONICAL_TASK_IDS):
        for right in CANONICAL_TASK_IDS[left_index + 1 :]:
            distance = float(np.sqrt(np.mean((normalized[left] - normalized[right]) ** 2)))
            pairs.append({"left_task_id": left, "right_task_id": right, "rms_distance": distance})
    distances = [float(item["rms_distance"]) for item in pairs]
    return {
        "distance_definition": "root_mean_square_over_50x8_action_chunk",
        "pair_count": len(pairs),
        "pairs": pairs,
        "mean_pair_distance": float(np.mean(distances)),
        "median_pair_distance": float(np.median(distances)),
        "maximum_pair_distance": max(distances),
    }


def task_sensitivity_report(
    *,
    conditioned_chunks_by_task: Mapping[str, torch.Tensor | np.ndarray],
    per_task_oracle_chunks_by_task: Mapping[str, torch.Tensor | np.ndarray],
) -> dict[str, object]:
    conditioned = task_action_chunk_distances(conditioned_chunks_by_task)
    oracle = task_action_chunk_distances(per_task_oracle_chunks_by_task)
    denominator = float(oracle["mean_pair_distance"])
    if denominator <= 0:
        raise M42EvaluationError("PerTask oracle action distance must be positive")
    return {
        "conditioned": conditioned,
        "per_task_oracle": oracle,
        "task_sensitivity_ratio": float(conditioned["mean_pair_distance"]) / denominator,
    }


__all__ = [
    "M42BenchmarkReport",
    "M42CheckpointContext",
    "M42CheckpointDescriptor",
    "M42EpisodeReport",
    "M42EvaluationError",
    "M42PolicyKind",
    "load_factor_film_training_checkpoint_context",
    "load_m4_checkpoint_context",
    "load_task_token_checkpoint_context",
    "load_task_token_training_checkpoint_context",
    "paired_episode_comparison",
    "run_m42_checkpoint_benchmark",
    "run_m42_validation_benchmark",
    "task_action_chunk_distances",
    "task_sensitivity_report",
    "task_token_validation_result",
]
