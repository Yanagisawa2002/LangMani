"""Heavy, target-side execution path for the zero-training M4.3a audit.

The thin CLI and :mod:`m43_audit_runtime` remain importable without a
simulator.  This module is imported only for a real audit.  It opens the M3B
validation view, materializes only ``m42_dev_v0`` when requested, integrity-
loads the eight frozen policies, and promotes project-owned immutable
evidence.  It never selects a checkpoint or executes a policy action.
"""

from __future__ import annotations

import json
import math
from argparse import Namespace
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from langmani.datasets.identity import sha256_hex
from langmani.datasets.lerobot_types import IMAGE_FEATURE_KEY, STATE_FEATURE_KEY
from langmani.datasets.observation_reconstruction import extract_base_camera_rgb
from langmani.datasets.policy_state import extract_panda_policy_state_v0
from langmani.environments import ENV_ID
from langmani.environments.specs import stable_scene_id
from langmani.policies.act_action_bounds import (
    ActionBoundConfig,
    ActionBoundMode,
    BoundedActionEnvPostprocessorV0,
)
from langmani.policies.act_runtime import inspect_git_state
from langmani.policies.act_training import image_to_policy_float
from langmani.policies.m42_evaluation import (
    M42CheckpointContext,
    M42PolicyKind,
    load_m4_checkpoint_context,
    load_task_token_training_checkpoint_context,
)
from langmani.policies.m42_schedule import (
    M42_DEV_SCHEDULE_ID,
    load_locked_schedule,
    materialize_schedule,
)
from langmani.policies.m42_types import GripperRuntimeMode
from langmani.policies.m43_audit_inputs import (
    M42DevelopmentInputs,
    RestrictedAuditInputs,
    RestrictedCheckpointInput,
    ValidationEpisodeInput,
    load_restricted_audit_inputs,
)
from langmani.policies.m43_audit_runtime import (
    M43_POLICY_LABELS,
    M43_REQUESTED_SOURCES,
    FixedAuditComputation,
    FixedPolicyObservation,
    M43AuditRuntimeError,
    SemanticPolicyContexts,
    combine_fixed_audit_computations,
    compute_fixed_observation_audits,
)
from langmani.policies.m43_evidence import (
    M43_CANDIDATE_POLICY_LABELS,
    M43AuditConfig,
    M43AuditScope,
    M43AuditScopeArtifact,
    M43ObservationSource,
    ObservationSourceIdentity,
    build_policy_semantic_summary,
    build_rollout_aggregate_evidence,
    stage_and_promote_audit_evidence,
    unavailable_rollout_evidence,
    validate_completed_audit_evidence,
)
from langmani.policies.m43_types import (
    M43_ACTION_COMPONENTS,
    M43_CANONICAL_TASK_IDS,
    ActionChunkDistanceConfig,
    SemanticAuditConclusion,
    SemanticFailureClass,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
M43_EXECUTION_SCHEMA_VERSION = "langmani-m43-semantic-audit-execution-v0"
_LEROBOT_CODEBASE_VERSION = "v3.0"
_LEROBOT_FPS = 20
_VIDEO_TIMESTAMP_TOLERANCE_S = 1e-4


def _progress(args: Namespace) -> object | None:
    return getattr(args, "_m43_command_progress", None)


def _mark_progress(args: Namespace, name: str) -> None:
    progress = _progress(args)
    if progress is not None and hasattr(progress, name):
        setattr(progress, name, True)


def _create_environment(*, rgb: bool) -> object:
    import gymnasium as gym

    import langmani.environments  # noqa: F401 - registers ENV_ID

    kwargs: dict[str, object] = {
        "num_envs": 1,
        "obs_mode": "rgb" if rgb else "none",
        "reward_mode": "none",
        "control_mode": "pd_joint_pos",
        "sim_backend": "physx_cpu",
    }
    if rgb:
        kwargs["render_mode"] = "rgb_array"
    return gym.make(ENV_ID, **kwargs)


def _action_bounds(env: object) -> tuple[tuple[float, ...], tuple[float, ...]]:
    processor = BoundedActionEnvPostprocessorV0.from_environment(
        env,
        ActionBoundConfig(mode=ActionBoundMode.PROJECT),
        expected_action_components=M43_ACTION_COMPONENTS,
    )
    contract = processor.action_space_contract()
    low = tuple(float(value) for value in cast(Sequence[object], contract["lower_bounds"]))
    high = tuple(float(value) for value in cast(Sequence[object], contract["upper_bounds"]))
    if len(low) != M43_ACTION_COMPONENTS or len(high) != M43_ACTION_COMPONENTS:
        raise M43AuditRuntimeError("M1 environment action bounds are not eight-dimensional")
    return low, high


def _retarget_context(context: M42CheckpointContext, *, device: str) -> None:
    target = torch.device(device)
    policy = cast(Any, context.loaded.policy)
    move = getattr(policy, "to", None)
    evaluate = getattr(policy, "eval", None)
    if not callable(move) or not callable(evaluate):
        raise M43AuditRuntimeError("loaded ACT policy lacks to()/eval()")
    config = getattr(policy, "config", None)
    if config is not None and hasattr(config, "device"):
        config.device = device
    move(target)
    evaluate()
    for pipeline in (context.loaded.preprocessor, context.loaded.postprocessor):
        steps = getattr(pipeline, "steps", None)
        if not isinstance(steps, Sequence):
            raise M43AuditRuntimeError("loaded ACT processor lacks a public steps sequence")
        for step in steps:
            if hasattr(step, "device"):
                step.device = target
            if hasattr(step, "stats") and hasattr(step, "_tensor_stats"):
                initialize = getattr(step, "__post_init__", None)
                if not callable(initialize):
                    raise M43AuditRuntimeError("normalization processor cannot retarget its stats")
                initialize()


def _load_context(value: RestrictedCheckpointInput, *, device: str) -> M42CheckpointContext:
    if value.policy_kind in {M42PolicyKind.PER_TASK, M42PolicyKind.STATE_ONEHOT}:
        context = load_m4_checkpoint_context(
            value.run_root,
            policy_kind=value.policy_kind,
            expected_checkpoint_fingerprint=value.checkpoint_fingerprint,
            expected_run_fingerprint=value.run_fingerprint,
            expected_dataset_fingerprint=value.dataset_fingerprint,
            expected_task_id=value.task_id,
        )
    elif value.policy_kind is M42PolicyKind.TASK_TOKEN:
        context = load_task_token_training_checkpoint_context(
            value.run_root,
            expected_checkpoint_fingerprint=value.checkpoint_fingerprint,
            expected_run_fingerprint=value.run_fingerprint,
            expected_dataset_fingerprint=value.dataset_fingerprint,
        )
    else:  # pragma: no cover - enum gate is independently tested
        raise M43AuditRuntimeError("unsupported checkpoint kind in M4.3a")
    _retarget_context(context, device=device)
    return context


def _load_policy_contexts(inputs: RestrictedAuditInputs, *, device: str) -> SemanticPolicyContexts:
    if device == "cuda" and not torch.cuda.is_available():
        raise M43AuditRuntimeError("M4.3a requested CUDA but torch.cuda is unavailable")
    per_task = {
        value.task_id: _load_context(value, device=device)
        for value in inputs.per_task_checkpoints
        if value.task_id is not None
    }
    contexts = SemanticPolicyContexts(
        per_task=per_task,
        state_onehot=_load_context(inputs.state_onehot_checkpoint, device=device),
        task_token=_load_context(inputs.task_token_checkpoint, device=device),
    )
    return contexts


def _action_std_from_context(context: M42CheckpointContext) -> tuple[float, ...]:
    records: list[object] = []
    steps = getattr(context.loaded.postprocessor, "steps", None)
    if not isinstance(steps, Sequence):
        raise M43AuditRuntimeError("postprocessor lacks public steps for statistics audit")
    for step in steps:
        stats = getattr(step, "stats", None)
        if not isinstance(stats, Mapping):
            continue
        action = stats.get("action")
        if not isinstance(action, Mapping):
            continue
        if "std" in action:
            records.append(action["std"])
    if len(records) != 1:
        raise M43AuditRuntimeError("checkpoint must expose exactly one 8D action std")
    raw = records[0]
    if isinstance(raw, torch.Tensor):
        if raw.dtype != torch.float32 or tuple(raw.shape) != (M43_ACTION_COMPONENTS,):
            raise M43AuditRuntimeError("saved Torch action std must be float32[8]")
        result = tuple(float(value) for value in raw.detach().to(device="cpu").tolist())
    elif isinstance(raw, np.ndarray):
        if raw.dtype != np.dtype(np.float32) or raw.shape != (M43_ACTION_COMPONENTS,):
            raise M43AuditRuntimeError("saved NumPy action std must be float32[8]")
        result = tuple(float(value) for value in raw.tolist())
    elif isinstance(raw, list | tuple):
        if len(raw) != M43_ACTION_COMPONENTS or any(
            isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating)
            for value in raw
        ):
            raise M43AuditRuntimeError("saved sequence action std must be numeric float32[8]")
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(raw, dtype=np.float32)
        result = tuple(float(value) for value in array.tolist())
    else:
        raise M43AuditRuntimeError("saved action std uses an unsupported public representation")
    if any(not math.isfinite(value) or value <= 0.0 for value in result):
        raise M43AuditRuntimeError("train-only action std must be finite and positive")
    return result


def _distance_config(
    *,
    env: object,
    policies: SemanticPolicyContexts,
    inputs: RestrictedAuditInputs,
) -> ActionChunkDistanceConfig:
    low, high = _action_bounds(env)
    onehot_std = _action_std_from_context(policies.state_onehot)
    token_std = _action_std_from_context(policies.task_token)
    with np.errstate(over="ignore", invalid="ignore"):
        canonical_input_std = tuple(
            float(value) for value in np.asarray(inputs.train_action_std, dtype=np.float32).tolist()
        )
    if any(not math.isfinite(value) or value <= 0.0 for value in canonical_input_std):
        raise M43AuditRuntimeError("canonical train-only action std must be finite and positive")
    if onehot_std != token_std or onehot_std != canonical_input_std:
        raise M43AuditRuntimeError(
            "loaded processors differ from the fingerprint-validated train action std"
        )
    runtime = inputs.runtime
    return ActionChunkDistanceConfig(
        action_lower_bounds=low,
        action_upper_bounds=high,
        train_action_std=inputs.train_action_std,
        locked_execution_horizon=runtime.execution_horizon,
    )


def _tensor_state(value: object) -> torch.Tensor:
    try:
        result = torch.as_tensor(value, dtype=torch.float32).reshape(-1).cpu()
    except (TypeError, ValueError) as error:
        raise M43AuditRuntimeError("M3B policy state must be numeric") from error
    if result.shape != (9,) or not bool(torch.isfinite(result).all()):
        raise M43AuditRuntimeError("M3B policy state must be finite float32[9]")
    return result.clone()


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise M43AuditRuntimeError(f"{label} is missing or linked")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise M43AuditRuntimeError(f"{label} is not strict JSON") from error
    if not isinstance(value, dict):
        raise M43AuditRuntimeError(f"{label} must contain one JSON object")
    return value


def _unlinked_parquet_files(root: Path, *, label: str) -> tuple[Path, ...]:
    if not root.is_dir() or root.is_symlink():
        raise M43AuditRuntimeError(f"{label} directory is missing or linked")
    files = tuple(sorted(path for path in root.glob("*/*.parquet") if path.is_file()))
    if not files:
        raise M43AuditRuntimeError(f"{label} contains no Parquet files")
    for path in files:
        current = root
        for part in path.relative_to(root).parts:
            current /= part
            if current.is_symlink():
                raise M43AuditRuntimeError(f"{label} contains a linked path component")
    return files


def _safe_video_path(dataset_root: Path, template: str, *, chunk: int, file: int) -> Path:
    if not template or Path(template).is_absolute() or ".." in Path(template).parts:
        raise M43AuditRuntimeError("LeRobot video path template is unsafe")
    try:
        relative = template.format(
            video_key=IMAGE_FEATURE_KEY,
            chunk_index=chunk,
            file_index=file,
        )
    except (IndexError, KeyError, ValueError) as error:
        raise M43AuditRuntimeError("LeRobot video path template is malformed") from error
    path = dataset_root / relative
    if not path.is_file() or path.is_symlink():
        raise M43AuditRuntimeError("requested M3B validation video is missing or linked")
    resolved = path.resolve()
    try:
        resolved.relative_to(dataset_root.resolve())
    except ValueError as error:
        raise M43AuditRuntimeError(
            "requested M3B validation video escapes the dataset root"
        ) from error
    return resolved


def _validation_frame_rows(
    dataset_root: Path,
    scheduled: Sequence[ValidationEpisodeInput],
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Predicate-load only six canonical M3B validation source frames.

    LeRobot's high-level dataset eagerly loads metadata for every episode,
    including the locked test split. M4.3a instead filters both episode and
    frame Parquet tables before materialization and decodes only validation
    videos named by the content-bound TaskToken validation schedule.
    """

    try:
        import pyarrow.dataset as pa_dataset
        from lerobot.datasets.video_utils import decode_video_frames
    except ImportError as error:  # pragma: no cover - dependency target gate
        raise M43AuditRuntimeError("M3B validation audit requires PyArrow and LeRobot") from error

    info = _read_json_object(dataset_root / "meta" / "info.json", label="LeRobot info")
    features = info.get("features")
    video_template = info.get("video_path")
    if (
        info.get("codebase_version") != _LEROBOT_CODEBASE_VERSION
        or info.get("fps") != _LEROBOT_FPS
        or not isinstance(features, Mapping)
        or IMAGE_FEATURE_KEY not in features
        or STATE_FEATURE_KEY not in features
        or not isinstance(video_template, str)
    ):
        raise M43AuditRuntimeError("LeRobot metadata differs from the M3B policy contract")

    groups: dict[str, list[ValidationEpisodeInput]] = {}
    for item in scheduled:
        groups.setdefault(item.scene_group_id, []).append(item)
    if len(groups) != 6:
        raise M43AuditRuntimeError("M3B validation schedule must contain six scene groups")
    source_indices = tuple(values[0].episode_index for values in groups.values())
    if len(set(source_indices)) != 6:
        raise M43AuditRuntimeError("M3B validation source episodes must be unique")

    episode_files = _unlinked_parquet_files(
        dataset_root / "meta" / "episodes", label="LeRobot validation episode metadata"
    )
    video_chunk = f"videos/{IMAGE_FEATURE_KEY}/chunk_index"
    video_file = f"videos/{IMAGE_FEATURE_KEY}/file_index"
    video_start = f"videos/{IMAGE_FEATURE_KEY}/from_timestamp"
    episode_dataset = pa_dataset.dataset([str(path) for path in episode_files], format="parquet")
    required_episode_columns = {"episode_index", video_chunk, video_file, video_start}
    if not required_episode_columns <= set(episode_dataset.schema.names):
        raise M43AuditRuntimeError("M3B episode metadata lacks base-camera video references")
    episode_table = episode_dataset.to_table(
        columns=sorted(required_episode_columns),
        filter=pa_dataset.field("episode_index").isin(source_indices),
    )
    episode_rows = {int(row["episode_index"]): row for row in episode_table.to_pylist()}
    if set(episode_rows) != set(source_indices) or len(episode_rows) != episode_table.num_rows:
        raise M43AuditRuntimeError("filtered M3B episode metadata differs from validation sources")

    data_files = _unlinked_parquet_files(dataset_root / "data", label="LeRobot validation data")
    data_dataset = pa_dataset.dataset([str(path) for path in data_files], format="parquet")
    required_data_columns = {"episode_index", "frame_index", "timestamp", STATE_FEATURE_KEY}
    if not required_data_columns <= set(data_dataset.schema.names):
        raise M43AuditRuntimeError("M3B frame data lacks the policy state contract")
    data_table = data_dataset.to_table(
        columns=sorted(required_data_columns),
        filter=(pa_dataset.field("episode_index").isin(source_indices))
        & (pa_dataset.field("frame_index") == 0),
    )
    data_rows = {int(row["episode_index"]): row for row in data_table.to_pylist()}
    if set(data_rows) != set(source_indices) or len(data_rows) != data_table.num_rows:
        raise M43AuditRuntimeError("filtered M3B data differs from initial validation frames")

    result: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for episode_index in source_indices:
        episode = episode_rows[episode_index]
        row = data_rows[episode_index]
        timestamp = float(row["timestamp"])
        from_timestamp = float(episode[video_start])
        if not math.isfinite(timestamp) or not math.isfinite(from_timestamp):
            raise M43AuditRuntimeError("M3B validation video timestamp is not finite")
        video_path = _safe_video_path(
            dataset_root,
            video_template,
            chunk=int(episode[video_chunk]),
            file=int(episode[video_file]),
        )
        decoded = decode_video_frames(
            video_path,
            [from_timestamp + timestamp],
            _VIDEO_TIMESTAMP_TOLERANCE_S,
            backend="pyav",
            return_uint8=True,
        )
        if decoded.shape[0] != 1:
            raise M43AuditRuntimeError("M3B validation decoder returned the wrong frame count")
        image = image_to_policy_float(decoded[0].detach().cpu()).clone()
        state = _tensor_state(row[STATE_FEATURE_KEY])
        result[episode_index] = (image, state)
    return result


def _validation_observations(
    dataset_root: Path,
    scheduled: Sequence[ValidationEpisodeInput],
) -> tuple[tuple[FixedPolicyObservation, ...], tuple[ObservationSourceIdentity, ...]]:
    frames = _validation_frame_rows(dataset_root, scheduled)
    groups: dict[str, list[ValidationEpisodeInput]] = {}
    for item in scheduled:
        groups.setdefault(item.scene_group_id, []).append(item)

    observations: list[FixedPolicyObservation] = []
    identities: list[ObservationSourceIdentity] = []
    for group_id in sorted(groups):
        values = groups[group_id]
        if tuple(item.task_id for item in values) != M43_CANONICAL_TASK_IDS:
            raise M43AuditRuntimeError("validation scene group lost canonical task order")
        source = values[0]
        image, state = frames[source.episode_index]
        for query in values:
            observation_id = f"m3b-validation:{group_id}:{query.task_id}"
            fixed = FixedPolicyObservation(
                source_scope=M43ObservationSource.M3B_VALIDATION.value,
                scene_key=group_id,
                observation_id=observation_id,
                requested_task_id=query.task_id,
                source_identity={
                    "source": M43ObservationSource.M3B_VALIDATION.value,
                    "scene_group_id": group_id,
                    "source_episode_index": source.episode_index,
                    "source_task_id": source.task_id,
                    "queried_task_id": query.task_id,
                },
                image=image.clone(),
                state=state.clone(),
            )
            observations.append(fixed)
            identities.append(
                ObservationSourceIdentity(
                    source=M43ObservationSource.M3B_VALIDATION,
                    observation_id=observation_id,
                    scene_id=stable_scene_id(source.scene_seed),
                    scene_group_id=group_id,
                    scene_seed=source.scene_seed,
                    source_task_id=source.task_id,
                    source_episode_id=None,
                    frame_index=0,
                    image_fingerprint=fixed.image_digest,
                    policy_state_fingerprint=fixed.state_digest,
                    queried_task_id=query.task_id,
                )
            )
    return tuple(observations), tuple(identities)


def _development_observations(
    env: object,
) -> tuple[tuple[FixedPolicyObservation, ...], tuple[ObservationSourceIdentity, ...]]:
    schedule = load_locked_schedule(M42_DEV_SCHEDULE_ID)
    episodes = materialize_schedule(schedule)
    by_scene: dict[int, list[object]] = {}
    for item in episodes:
        by_scene.setdefault(item.scene_index, []).append(item)
    if len(by_scene) != 12:
        raise M43AuditRuntimeError("m42_dev_v0 must contain exactly twelve scene groups")

    observations: list[FixedPolicyObservation] = []
    identities: list[ObservationSourceIdentity] = []
    base = cast(Any, getattr(env, "unwrapped", env))
    for scene_index in sorted(by_scene):
        values = cast(Sequence[Any], by_scene[scene_index])
        if tuple(item.task_id for item in values) != M43_CANONICAL_TASK_IDS:
            raise M43AuditRuntimeError("development scene lost canonical task order")
        source = values[0]
        observation, _ = cast(Any, env).reset(
            seed=source.scene_seed,
            options={"task_spec": source.task_spec.to_dict()},
        )
        specs = base.get_episode_specs()
        if len(specs) != 1 or specs[0].scene_id != source.scene_id:
            raise M43AuditRuntimeError("development reset changed the locked scene identity")
        rgb_hwc = extract_base_camera_rgb(observation)
        image = image_to_policy_float(torch.from_numpy(np.transpose(rgb_hwc, (2, 0, 1)).copy()))
        state_array = extract_panda_policy_state_v0(base.agent.robot)
        state = torch.from_numpy(np.asarray(state_array, dtype=np.float32).copy())
        scene_key = f"{scene_index:02d}:{source.scene_id}"
        for query in values:
            observation_id = f"m42-dev:{source.scene_id}:{query.task_id}"
            fixed = FixedPolicyObservation(
                source_scope=M43ObservationSource.M42_DEVELOPMENT.value,
                scene_key=scene_key,
                observation_id=observation_id,
                requested_task_id=query.task_id,
                source_identity={
                    "source": M43ObservationSource.M42_DEVELOPMENT.value,
                    "scene_id": source.scene_id,
                    "source_task_id": source.task_id,
                    "queried_task_id": query.task_id,
                },
                image=image.clone(),
                state=state.clone(),
            )
            observations.append(fixed)
            identities.append(
                ObservationSourceIdentity(
                    source=M43ObservationSource.M42_DEVELOPMENT,
                    observation_id=observation_id,
                    scene_id=source.scene_id,
                    scene_group_id=None,
                    scene_seed=source.scene_seed,
                    source_task_id=source.task_id,
                    source_episode_id=None,
                    frame_index=0,
                    image_fingerprint=fixed.image_digest,
                    policy_state_fingerprint=fixed.state_digest,
                    queried_task_id=query.task_id,
                )
            )
    return tuple(observations), tuple(identities)


_M42_TO_M43_FAILURE = {
    "never_grasped_target": SemanticFailureClass.NEVER_GRASPED_TARGET,
    "target_grasped_not_lifted": SemanticFailureClass.TARGET_GRASPED_NOT_LIFTED,
    "lifted_not_transported": SemanticFailureClass.LIFTED_NOT_TRANSPORTED,
    "transported_not_descended": SemanticFailureClass.TRANSPORTED_NOT_DESCENDED,
    "descended_not_released": SemanticFailureClass.DESCENDED_NOT_RELEASED,
    "released_outside_success_region": SemanticFailureClass.RELEASED_OUTSIDE_SUCCESS_REGION,
    "released_but_not_static": SemanticFailureClass.RELEASED_BUT_NOT_STATIC,
    "target_success_then_lost": SemanticFailureClass.TARGET_SUCCESS_THEN_LOST,
    "timed_out_after_target_grasp": SemanticFailureClass.TIMED_OUT_AFTER_TARGET_GRASP,
    "wrong_object_interaction": SemanticFailureClass.WRONG_OBJECT_INTERACTION,
    "environment_failure": SemanticFailureClass.ENVIRONMENT_FAILURE,
    "success": SemanticFailureClass.SUCCESS,
}


def _rollout_distribution(benchmark: Mapping[str, object]) -> dict[str, int]:
    raw_episodes = benchmark.get("episodes")
    if isinstance(raw_episodes, Sequence) and not isinstance(raw_episodes, str | bytes):
        episodes = tuple(raw_episodes)
    else:
        children = benchmark.get("benchmarks")
        if not isinstance(children, Sequence) or isinstance(children, str | bytes):
            raise M43AuditRuntimeError("M4.2 development benchmark lacks episode evidence")
        flattened: list[object] = []
        for child in children:
            if not isinstance(child, Mapping):
                raise M43AuditRuntimeError("M4.2 PerTask benchmark descriptor is malformed")
            child_episodes = child.get("episodes")
            if not isinstance(child_episodes, Sequence) or isinstance(child_episodes, str | bytes):
                raise M43AuditRuntimeError("M4.2 PerTask benchmark lacks episode evidence")
            flattened.extend(child_episodes)
        episodes = tuple(flattened)
    result = {value.value: 0 for value in SemanticFailureClass}
    for raw in episodes:
        if not isinstance(raw, Mapping):
            raise M43AuditRuntimeError("M4.2 development episode is malformed")
        diagnostics = raw.get("post_grasp_diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise M43AuditRuntimeError("M4.2 episode lacks post-grasp diagnostics")
        phase = diagnostics.get("phase")
        if not isinstance(phase, str) or phase not in _M42_TO_M43_FAILURE:
            raise M43AuditRuntimeError("M4.2 episode has an unknown post-grasp phase")
        result[_M42_TO_M43_FAILURE[phase].value] += 1
    if len(episodes) != 72 or sum(result.values()) != 72:
        raise M43AuditRuntimeError("development rollout evidence must cover 72 episodes")
    return result


def _rollout_metrics(benchmark: Mapping[str, object]) -> dict[str, object]:
    aggregate = benchmark.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise M43AuditRuntimeError("M4.2 development benchmark lacks aggregate metrics")
    names = (
        "episode_count",
        "successes",
        "success_rate",
        "timeout_count",
        "target_grasped_count",
        "wrong_object_interaction_count",
        "wrong_object_grasp_count",
        "wrong_object_in_target_bin_count",
        "target_in_wrong_bin_count",
        "target_off_table_count",
        "invalid_action_count",
    )
    result = {name: aggregate[name] for name in names if name in aggregate}
    if result.get("episode_count") != 72:
        raise M43AuditRuntimeError("development aggregate must report 72 episodes")
    action_metrics = aggregate.get("action_metrics")
    if not isinstance(action_metrics, Mapping):
        raise M43AuditRuntimeError("development aggregate lacks separated action metrics")
    result["action_metrics"] = dict(action_metrics)
    strict_names = (
        "strict_unprojected_success_count",
        "strict_runtime_unprojected_success_count",
        "strict_raw_unmodified_success_count",
    )
    children = benchmark.get("benchmarks")
    for name in strict_names:
        if name in aggregate:
            result[name] = aggregate[name]
            continue
        if not isinstance(children, Sequence) or isinstance(children, str | bytes):
            raise M43AuditRuntimeError(f"development aggregate lacks {name}")
        total = 0
        for child in children:
            if not isinstance(child, Mapping) or not isinstance(child.get("aggregate"), Mapping):
                raise M43AuditRuntimeError("M4.2 PerTask aggregate is malformed")
            child_aggregate = cast(Mapping[str, object], child["aggregate"])
            if name not in child_aggregate:
                raise M43AuditRuntimeError(f"M4.2 PerTask aggregate lacks {name}")
            total += int(cast(int, child_aggregate[name]))
        result[name] = total
    return result


def _development_rollout_evidence(development: M42DevelopmentInputs) -> dict[str, object]:
    comparison = development.comparison
    models = comparison.get("models")
    if not isinstance(models, Mapping):
        raise M43AuditRuntimeError("development comparison lacks model evidence")
    benchmarks: dict[str, Mapping[str, object]] = {}
    for label in M43_CANDIDATE_POLICY_LABELS:
        value = models.get(label)
        if not isinstance(value, Mapping):
            raise M43AuditRuntimeError(f"development comparison lacks {label}")
        benchmarks[label] = value
    comparison_fingerprint = f"sha256:{sha256_hex(development.comparison)}"
    return build_rollout_aggregate_evidence(
        source_fingerprint=comparison_fingerprint,
        per_policy_post_grasp_failure_distribution={
            label: _rollout_distribution(benchmarks[label]) for label in M43_CANDIDATE_POLICY_LABELS
        },
        per_policy_metrics={
            label: _rollout_metrics(benchmarks[label]) for label in M43_CANDIDATE_POLICY_LABELS
        },
    )


def _scope_artifact(
    computation: FixedAuditComputation,
    *,
    scope: M43AuditScope,
    rollout_evidence: Mapping[str, object],
) -> M43AuditScopeArtifact:
    available = rollout_evidence.get("available") is True
    distributions = rollout_evidence.get("per_policy_post_grasp_failure_distribution")
    if available and not isinstance(distributions, Mapping):
        raise M43AuditRuntimeError("available rollout evidence lacks failure distributions")
    summaries: dict[str, object] = {}
    audits = dict(computation.policy_audits)
    for label in M43_POLICY_LABELS:
        runtime = computation.policy_runtime_summaries[label]
        sensitivity = runtime.get("task_sensitivity")
        if not isinstance(sensitivity, Mapping):
            raise M43AuditRuntimeError("fixed audit lacks task-sensitivity evidence")
        external = None
        if available:
            external = cast(Mapping[str, object], cast(Mapping[str, object], distributions)[label])
            post_grasp_names = {
                SemanticFailureClass.TARGET_GRASPED_NOT_LIFTED.value,
                SemanticFailureClass.LIFTED_NOT_TRANSPORTED.value,
                SemanticFailureClass.TRANSPORTED_NOT_DESCENDED.value,
                SemanticFailureClass.DESCENDED_NOT_RELEASED.value,
                SemanticFailureClass.RELEASED_OUTSIDE_SUCCESS_REGION.value,
                SemanticFailureClass.RELEASED_BUT_NOT_STATIC.value,
                SemanticFailureClass.TARGET_SUCCESS_THEN_LOST.value,
                SemanticFailureClass.TIMED_OUT_AFTER_TARGET_GRASP.value,
            }
            if sum(int(external[name]) for name in post_grasp_names) > 0:
                conclusions = list(audits[label].conclusions)
                if SemanticAuditConclusion.POST_GRASP_EXECUTION_FAILURE not in conclusions:
                    conclusions.append(SemanticAuditConclusion.POST_GRASP_EXECUTION_FAILURE)
                audits[label] = replace(audits[label], conclusions=tuple(conclusions))
        summaries[label] = build_policy_semantic_summary(
            audits[label],
            task_sensitivity_magnitude=float(sensitivity["mean_full_chunk_rms"]),
            external_failure_distribution=external,
        )
    return M43AuditScopeArtifact(
        scope=scope,
        ordered_observation_ids=computation.ordered_observation_ids,
        policy_audits=audits,
        policy_summaries=summaries,
        first_interaction_evidence_available=False,
        rollout_evidence=rollout_evidence,
    )


def _requested_scope(mode: str) -> M43AuditScope:
    return {
        "validation": M43AuditScope.M3B_VALIDATION,
        "m42_dev_v0": M43AuditScope.M42_DEVELOPMENT,
        "combined": M43AuditScope.COMBINED,
    }[mode]


def run_semantic_audit(args: Namespace) -> dict[str, object]:
    """Execute one complete validation/development semantic audit."""

    mode = getattr(args, "mode", None)
    device = getattr(args, "device", None)
    if mode not in M43_REQUESTED_SOURCES or device not in {"cpu", "cuda"}:
        raise M43AuditRuntimeError("invalid M4.3a execution mode or device")
    git = inspect_git_state(PROJECT_ROOT)
    if not git.baseline_tracked or git.dirty:
        raise M43AuditRuntimeError("real M4.3a evidence requires one clean tracked Git commit")

    inputs = load_restricted_audit_inputs(
        mode=mode,
        dataset_root=args.dataset_root,
        m4_checkpoint_root=args.m4_checkpoint_root,
        task_token_checkpoint_root=args.task_token_checkpoint_root,
        m42_diagnostics_root=args.m42_diagnostics_root,
        runtime_selection_path=args.runtime_selection,
    )
    policies = _load_policy_contexts(inputs, device=device)
    need_development = mode in {"m42_dev_v0", "combined"}
    env = _create_environment(rgb=need_development)
    try:
        distance_config = _distance_config(env=env, policies=policies, inputs=inputs)
        computations: dict[M43AuditScope, FixedAuditComputation] = {}
        identities: dict[M43AuditScope, tuple[ObservationSourceIdentity, ...]] = {}
        if mode in {"validation", "combined"}:
            _mark_progress(args, "validation_evidence_accessed")
            observations, source_identities = _validation_observations(
                inputs.dataset_root, inputs.validation_episodes
            )
            computations[M43AuditScope.M3B_VALIDATION] = compute_fixed_observation_audits(
                observations=observations,
                policies=policies,
                distance_config=distance_config,
            )
            identities[M43AuditScope.M3B_VALIDATION] = source_identities
        if need_development:
            _mark_progress(args, "development_evidence_accessed")
            _mark_progress(args, "physical_execution_started")
            observations, source_identities = _development_observations(env)
            computations[M43AuditScope.M42_DEVELOPMENT] = compute_fixed_observation_audits(
                observations=observations,
                policies=policies,
                distance_config=distance_config,
            )
            identities[M43AuditScope.M42_DEVELOPMENT] = source_identities
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    rollout = unavailable_rollout_evidence()
    if need_development:
        if inputs.development is None:
            raise M43AuditRuntimeError("development audit lacks its restricted rollout evidence")
        rollout = _development_rollout_evidence(inputs.development)
    artifacts: dict[M43AuditScope, M43AuditScopeArtifact] = {}
    if M43AuditScope.M3B_VALIDATION in computations:
        artifacts[M43AuditScope.M3B_VALIDATION] = _scope_artifact(
            computations[M43AuditScope.M3B_VALIDATION],
            scope=M43AuditScope.M3B_VALIDATION,
            rollout_evidence=unavailable_rollout_evidence(),
        )
    if M43AuditScope.M42_DEVELOPMENT in computations:
        artifacts[M43AuditScope.M42_DEVELOPMENT] = _scope_artifact(
            computations[M43AuditScope.M42_DEVELOPMENT],
            scope=M43AuditScope.M42_DEVELOPMENT,
            rollout_evidence=rollout,
        )
    if mode == "combined":
        combined = combine_fixed_audit_computations(
            computations[M43AuditScope.M3B_VALIDATION],
            computations[M43AuditScope.M42_DEVELOPMENT],
        )
        computations[M43AuditScope.COMBINED] = combined
        identities[M43AuditScope.COMBINED] = (
            *identities[M43AuditScope.M3B_VALIDATION],
            *identities[M43AuditScope.M42_DEVELOPMENT],
        )
        artifacts[M43AuditScope.COMBINED] = _scope_artifact(
            combined,
            scope=M43AuditScope.COMBINED,
            rollout_evidence=rollout,
        )

    scope = _requested_scope(mode)
    ordered_identities = (
        identities[scope]
        if scope is not M43AuditScope.COMBINED
        else (
            *identities[M43AuditScope.M3B_VALIDATION],
            *identities[M43AuditScope.M42_DEVELOPMENT],
        )
    )
    config = M43AuditConfig(
        scope=scope,
        git_commit=git.commit,
        m3b_fingerprint=inputs.dataset_fingerprint,
        m3b_split_manifest_digest=inputs.split_digest,
        per_task_checkpoint_fingerprints={
            value.task_id: value.checkpoint_fingerprint
            for value in inputs.per_task_checkpoints
            if value.task_id is not None
        },
        state_onehot_checkpoint_fingerprint=(inputs.state_onehot_checkpoint.checkpoint_fingerprint),
        task_token_checkpoint_fingerprint=inputs.task_token_checkpoint.checkpoint_fingerprint,
        selected_execution_horizon=inputs.runtime.execution_horizon,
        selected_gripper_runtime=GripperRuntimeMode(inputs.runtime.gripper_mode),
        distance_config=distance_config,
        ordered_observations=ordered_identities,
    )
    evidence = stage_and_promote_audit_evidence(
        output_root=args.output_root,
        config=config,
        scope_artifacts=artifacts,
    )
    validated = validate_completed_audit_evidence(evidence.root, expected_config=config)
    requested = list(M43_REQUESTED_SOURCES[mode])
    scope_counts = {
        item.value: len(artifacts[item].ordered_observation_ids)
        for item in config.required_artifact_scopes
    }
    return {
        "schema_version": M43_EXECUTION_SCHEMA_VERSION,
        "passed": True,
        "requested_sources": requested,
        "audited_sources": requested,
        "device": device,
        "audit_identity": {
            "config_fingerprint": config.fingerprint,
            "evidence_fingerprint": validated.evidence_fingerprint,
            "restricted_input_fingerprint": inputs.fingerprint,
            "git_commit": git.commit,
        },
        "fingerprint_inputs": {
            "m3b_fingerprint": inputs.dataset_fingerprint,
            "m3b_split_manifest_digest": inputs.split_digest,
            "per_task_checkpoint_fingerprints": {
                value.task_id: value.checkpoint_fingerprint
                for value in inputs.per_task_checkpoints
                if value.task_id is not None
            },
            "state_onehot_checkpoint_fingerprint": (
                inputs.state_onehot_checkpoint.checkpoint_fingerprint
            ),
            "task_token_checkpoint_fingerprint": (
                inputs.task_token_checkpoint.checkpoint_fingerprint
            ),
            "selected_execution_horizon": inputs.runtime.execution_horizon,
            "selected_gripper_runtime": inputs.runtime.gripper_mode,
            "primary_distance_metric": distance_config.primary_metric,
        },
        "scope_observation_counts": scope_counts,
        "evidence_directory": str(validated.root),
        "task_token_m42_status": "rejected",
        "semantic_audit_implementation_validated": True,
        "semantic_audit_completed": True,
        "factor_film_training_completed": False,
        "physical_execution": need_development,
        "physical_target_validated": False,
        "test_split_accessed": False,
        "fresh_seed_accessed": False,
        "final_schedule_accessed": False,
        "final_benchmark_authorized": False,
        "go_no_go_decision_completed": False,
        "smolvla_go": False,
        "inference_deferred": False,
    }


__all__ = ["M43_EXECUTION_SCHEMA_VERSION", "run_semantic_audit"]
