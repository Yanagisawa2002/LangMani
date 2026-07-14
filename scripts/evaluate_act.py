"""Run closed-loop M1 evaluation for one locally saved M4 ACT checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import traceback
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from langmani.datasets.lerobot_types import DatasetSplit, EpisodeExportRecord
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.pick_place_by_instruction import ENV_ID
from langmani.environments.specs import TaskSpec, stable_scene_id, stable_task_id
from langmani.policies.act_action_bounds import (
    ActionBoundConfig,
    ActionBoundMode,
    ActionBoundProcessingError,
    ActionProjectionRecord,
    BoundedActionEnvPostprocessorV0,
    EvaluationRuntimeIdentity,
    EvaluationRuntimeManifest,
)
from langmani.policies.act_analysis import (
    CounterfactualDemonstration,
    compute_counterfactual_sensitivity,
    expert_chunk_distances,
    load_m3b_reference_counterfactual_group,
    task_identity_effects,
)
from langmani.policies.act_checkpoint import (
    CHECKPOINT_MANIFEST,
    LoadedActCheckpoint,
    load_act_checkpoint,
)
from langmani.policies.act_conditioning import (
    CANONICAL_TASK_IDS,
    append_canonical_task_onehot,
)
from langmani.policies.act_data import load_completed_m3b_dataset
from langmani.policies.act_evaluation import (
    FreshSeedSchedule,
    authorize_test_evaluation,
    build_fresh_seed_schedule,
    create_checkpoint_selection,
    load_checkpoint_selection,
    rollout_schedule_digest,
    summarize_rollout_benchmark,
    write_checkpoint_selection_atomic,
)
from langmani.policies.act_rollout import (
    ActManiSkillRolloutAdapter,
    build_policy_observation,
)
from langmani.policies.act_runtime import (
    atomic_write_json,
    inspect_git_state,
    runtime_versions,
    validate_git_for_run,
)
from langmani.policies.act_types import (
    TASK_ONEHOT_MAPPING_VERSION,
    ActExperimentConfig,
    ActExperimentManifest,
    ActRunIdentity,
    ActVariant,
    EvaluationSplit,
    ExperimentMode,
    RolloutBenchmarkResult,
    RolloutEpisodeResult,
    RolloutStatus,
    ValidationResult,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "outputs" / "datasets" / "m3b" / "langmani-pick-place-lerobot-v1"
)
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "diagnostics" / "m4" / "evaluate_act.json"
EVALUATION_OWNER_FILE = "evaluation_owner.json"
EVALUATION_ANALYSIS_FILE = "analysis.json"
EVALUATION_RUNTIME_MANIFEST_FILE = "evaluation_runtime_manifest.json"
ACTION_PROJECTION_RECORDS_FILE = "action_projection_records.jsonl"


def _is_link_like(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def _validate_runtime_identity(identity: ActRunIdentity) -> None:
    versions = runtime_versions()
    if (
        versions["lerobot"] != identity.lerobot_version
        or versions["torch"] != identity.torch_version
        or versions["cuda"] != identity.cuda_version
    ):
        raise RuntimeError(
            "evaluation runtime versions differ from the checkpoint training identity"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test", "fresh_seed"),
        required=True,
    )
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--development-test-override")
    parser.add_argument("--lock-selection", action="store_true")
    parser.add_argument("--counterfactual-sensitivity", action="store_true")
    parser.add_argument("--rollout-video", action="store_true")
    parser.add_argument(
        "--action-bound-mode",
        choices=tuple(mode.value for mode in ActionBoundMode),
        default=ActionBoundMode.REJECT.value,
        help="explicit environment-action handling after the ACT policy postprocessor",
    )
    parser.add_argument(
        "--strict-bound-probe",
        action="store_true",
        help=(
            "reproduce the first strict bound rejection, verify deterministic projection, "
            "and execute exactly one projected environment step without a rollout"
        ),
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    if _is_link_like(path):
        raise RuntimeError(f"refusing linked evaluation artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def _write_or_validate_immutable(path: Path, payload: dict[str, object]) -> None:
    if _is_link_like(path):
        raise RuntimeError(f"refusing linked immutable evaluation artifact: {path}")
    if path.is_file():
        if _read(path) != payload:
            raise RuntimeError(f"existing immutable evaluation artifact differs: {path}")
        return
    atomic_write_json(path, payload, immutable=True)


def _write_projection_records(
    path: Path,
    records: list[tuple[str, ActionProjectionRecord]],
) -> dict[str, object]:
    """Write per-step raw/executed evidence outside compact benchmark JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing to replace action-projection artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for evaluation_id, record in records:
                stream.write(
                    json.dumps(
                        {"evaluation_id": evaluation_id, **record.to_dict()},
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "relative_path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "record_count": len(records),
    }


def _validate_projection_artifact(
    output: Path,
    payload: object,
    *,
    expected_runtime_fingerprint: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise RuntimeError("benchmark lacks action-projection artifact metadata")
    relative = payload.get("relative_path")
    digest = payload.get("sha256")
    count = payload.get("record_count")
    if (
        relative != ACTION_PROJECTION_RECORDS_FILE
        or not isinstance(digest, str)
        or not isinstance(count, int)
    ):
        raise RuntimeError("action-projection artifact metadata is malformed")
    path = output / relative
    if _is_link_like(path) or not path.is_file() or path.resolve().parent != output.resolve():
        raise RuntimeError("action-projection artifact is missing or unsafe")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise RuntimeError("action-projection artifact checksum mismatch")
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != count:
        raise RuntimeError("action-projection artifact record count mismatch")
    for line in lines:
        value = json.loads(line)
        if not isinstance(value, dict) or not isinstance(value.get("evaluation_id"), str):
            raise RuntimeError("action-projection artifact record is malformed")
        record = dict(value)
        record.pop("evaluation_id")
        ActionProjectionRecord.from_dict(record)
    runtime = EvaluationRuntimeManifest.from_dict(_read(output / EVALUATION_RUNTIME_MANIFEST_FILE))
    if runtime.identity.runtime_fingerprint != expected_runtime_fingerprint:
        raise RuntimeError("evaluation runtime manifest fingerprint mismatch")


def _checkpoint_context(
    checkpoint: Path,
) -> tuple[Path, str, ActRunIdentity, ActExperimentConfig]:
    checkpoint = checkpoint.resolve()
    if checkpoint.parent.name != "checkpoints":
        raise ValueError("checkpoint must be one direct child of a run's checkpoints directory")
    run_root = checkpoint.parent.parent
    relative = checkpoint.relative_to(run_root).as_posix()
    checkpoint_manifest = _read(checkpoint / CHECKPOINT_MANIFEST)
    identity_value = checkpoint_manifest.get("identity")
    if not isinstance(identity_value, dict):
        raise RuntimeError("checkpoint is missing its run identity")
    identity = ActRunIdentity.from_dict(identity_value)
    run_manifest = ActExperimentManifest.from_dict(_read(run_root / "run_manifest.json"))
    config = run_manifest.config
    if run_manifest.identity != identity:
        raise RuntimeError("run and checkpoint identities disagree")
    return run_root, relative, identity, config


def _task_spec(record: EpisodeExportRecord) -> TaskSpec:
    return TaskSpec(
        target_object_id=record.target_object_id,
        target_bin_id=record.target_bin_id,
        instruction_template_id=record.instruction_template_id,
    )


def _split_schedule(
    *,
    completed: object,
    identity: ActRunIdentity,
    config: ActExperimentConfig,
    schedule_contract: dict[str, object],
    split: EvaluationSplit,
) -> tuple[tuple[int, TaskSpec], ...]:
    manifest = completed.manifest
    if split is EvaluationSplit.FRESH_SEED:
        accepted_seeds = tuple(sorted({item.source_scene_seed for item in manifest.episodes}))
        schedule = build_fresh_seed_schedule(
            m3b_export_fingerprint=manifest.export_fingerprint,
            accepted_source_seeds=accepted_seeds,
            rejected_source_seeds=None,
            namespace=config.evaluation.fresh_seed_namespace,
        )
        stored_value = schedule_contract.get("fresh_seed_schedule")
        stored = FreshSeedSchedule.from_dict(stored_value)
        if stored != schedule:
            raise RuntimeError("fresh-seed schedule differs from the run's predeclared contract")
        first_train_index = completed.views.train.episode_indices[0]
        first_group_id = completed.views.scene_group_id_by_episode[first_train_index]
        first_group = tuple(
            item for item in manifest.episodes if item.source_scene_group_id == first_group_id
        )
        by_task_id = {item.task_id: _task_spec(item) for item in first_group}
        if set(by_task_id) != set(CANONICAL_TASK_IDS):
            raise RuntimeError("M3B first scene group does not contain the canonical six tasks")
        if identity.variant is ActVariant.PER_TASK:
            selected = (by_task_id[identity.task_id],)
        else:
            selected = tuple(by_task_id[task_id] for task_id in CANONICAL_TASK_IDS)
        return tuple((seed, task) for seed in schedule.ordered_fresh_seeds for task in selected)
    dataset_split = DatasetSplit(split.value)
    records = tuple(item for item in manifest.episodes if item.split is dataset_split)
    if identity.variant is ActVariant.PER_TASK:
        records = tuple(item for item in records if item.task_id == identity.task_id)
    return tuple((item.source_scene_seed, _task_spec(item)) for item in records)


def _schedule_digest(
    identity: ActRunIdentity,
    split: EvaluationSplit,
    schedule: tuple[tuple[int, TaskSpec], ...],
) -> str:
    return rollout_schedule_digest(
        m3b_export_fingerprint=identity.m3b_export_fingerprint,
        variant=identity.variant.value,
        task_id=identity.task_id,
        split=split.value,
        episodes=tuple(
            {"scene_seed": seed, "task_spec": task.to_dict()} for seed, task in schedule
        ),
    )


def _create_environment(sim_backend: str, *, record_video: bool, output: Path) -> object:
    import gymnasium as gym

    import langmani.environments  # noqa: F401 - registers the M1 environment

    env = gym.make(
        ENV_ID,
        num_envs=1,
        obs_mode="rgb",
        reward_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend=sim_backend,
    )
    if not record_video:
        return env
    from mani_skill.utils.wrappers.record import RecordEpisode

    return RecordEpisode(
        env,
        output_dir=str(output),
        save_trajectory=False,
        save_video=True,
        save_on_reset=True,
        record_reward=False,
        record_env_state=False,
        video_fps=20,
        avoid_overwriting_video=True,
    )


def _raw_first_action(
    *,
    env: object,
    loaded: LoadedActCheckpoint,
    identity: ActRunIdentity,
    scene_seed: int,
    task_spec: TaskSpec,
) -> torch.Tensor:
    base = getattr(env, "unwrapped", env)
    loaded.policy.reset()
    loaded.preprocessor.reset()
    loaded.postprocessor.reset()
    observation, _ = env.reset(seed=scene_seed, options={"task_spec": task_spec.to_dict()})
    raw = build_policy_observation(
        observation,
        base_environment=base,
        variant=identity.variant,
        task_id=stable_task_id(task_spec),
        task_conditioner=(
            append_canonical_task_onehot
            if identity.variant is ActVariant.MIXED_TASK_ONEHOT
            else None
        ),
    )
    processed = loaded.preprocessor(raw)
    if not isinstance(processed, Mapping):
        raise RuntimeError("reloaded ACT preprocessor did not return a tensor mapping")
    with torch.inference_mode():
        predicted = loaded.policy.select_action(dict(processed))
        action = loaded.postprocessor(predicted)
    if (
        not isinstance(action, torch.Tensor)
        or action.dtype != torch.float32
        or action.shape
        != (
            1,
            8,
        )
    ):
        raise RuntimeError("reloaded ACT stack did not produce float32[1,8]")
    return action.detach().clone()


def _evaluation_runtime_manifest(
    *,
    env: object,
    loaded: LoadedActCheckpoint,
    run_root: Path,
    checkpoint_relative_path: str,
    identity: ActRunIdentity,
    config: ActExperimentConfig,
    action_bound_config: ActionBoundConfig,
    evaluation_git: dict[str, object],
    split: EvaluationSplit,
    sim_backend: str,
    record_video: bool,
    first_episode: tuple[int, TaskSpec],
    output: Path,
) -> tuple[EvaluationRuntimeManifest, torch.Tensor]:
    processor = BoundedActionEnvPostprocessorV0.from_environment(
        env,
        action_bound_config,
        expected_action_components=8,
    )
    reloaded_bound_processor = BoundedActionEnvPostprocessorV0.from_environment(
        env,
        ActionBoundConfig.from_dict(action_bound_config.to_dict()),
        expected_action_components=8,
    )
    if processor.action_space_contract() != reloaded_bound_processor.action_space_contract():
        raise RuntimeError("serialized action-bound processor changed the action-space contract")
    fresh_loaded = load_act_checkpoint(
        run_root=run_root,
        checkpoint_relative_path=checkpoint_relative_path,
        expected_identity=identity,
    )
    first_seed, first_task = first_episode
    first_raw = _raw_first_action(
        env=env,
        loaded=loaded,
        identity=identity,
        scene_seed=first_seed,
        task_spec=first_task,
    )
    second_raw = _raw_first_action(
        env=env,
        loaded=fresh_loaded,
        identity=identity,
        scene_seed=first_seed,
        task_spec=first_task,
    )
    tolerance = action_bound_config.comparison_tolerance
    raw_match = bool(torch.allclose(first_raw, second_raw, atol=tolerance, rtol=0))
    if not raw_match:
        raise RuntimeError("fresh checkpoint/processor reload changed deterministic raw action")

    def process(processor_value: BoundedActionEnvPostprocessorV0) -> object:
        try:
            return processor_value.process(first_raw, rollout_step=1)
        except ActionBoundProcessingError as error:
            return error

    first_bound = process(processor)
    second_bound = process(reloaded_bound_processor)
    if type(first_bound) is not type(second_bound):
        raise RuntimeError("reloaded action-bound processor changed first-action behavior")
    if hasattr(first_bound, "audit_record"):
        first_record = first_bound.audit_record.to_dict()
        second_record = second_bound.audit_record.to_dict()
    else:
        first_record = getattr(first_bound, "record", None)
        second_record = getattr(second_bound, "record", None)
        first_record = None if first_record is None else first_record.to_dict()
        second_record = None if second_record is None else second_record.to_dict()
    if first_record != second_record:
        raise RuntimeError("reloaded action-bound processor changed projection audit output")

    runtime_identity = EvaluationRuntimeIdentity(
        checkpoint_fingerprint=loaded.record.checkpoint_fingerprint,
        policy_preprocessor_fingerprint=loaded.component_fingerprints.preprocessor,
        policy_postprocessor_fingerprint=loaded.component_fingerprints.postprocessor,
        action_bound_config=action_bound_config,
        environment_id=ENV_ID,
        action_space_contract=processor.action_space_contract(),
        task_conditioning_mapping_version=TASK_ONEHOT_MAPPING_VERSION,
        rollout_config={
            "evaluation": config.evaluation.to_dict(),
            "sim_backend": sim_backend,
            "split": split.value,
            "rollout_video": record_video,
        },
        code_git_commit=str(evaluation_git["commit"]),
    )
    return (
        EvaluationRuntimeManifest(
            identity=runtime_identity,
            checkpoint_model_reload_validated=True,
            policy_processor_reload_validated=True,
            action_bound_processor_reload_validated=True,
            deterministic_raw_action_matched=raw_match,
            raw_action_match_tolerance=tolerance,
            evaluation_output_path=str(output.resolve()),
        ),
        first_raw,
    )


def _offline_validation_loss(
    run_root: Path,
    checkpoint_step: int,
    *,
    checkpoint_metric: Mapping[str, object] | None,
    checkpoint_relative_path: str,
) -> float:
    values: list[dict[str, object]] = []
    metrics_path = run_root / "metrics.jsonl"
    _validate_owned_artifact_path(run_root, metrics_path)
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if isinstance(item, dict) and item.get("step") == checkpoint_step:
            values.append(item)
    expected_metric = (
        {**checkpoint_metric, "checkpoint_path": checkpoint_relative_path}
        if checkpoint_metric is not None
        else None
    )
    if (
        len(values) != 1
        or expected_metric is None
        or values[0] != expected_metric
        or expected_metric.get("validation_loss") is None
    ):
        raise RuntimeError("checkpoint has no unique stored offline validation loss")
    return float(expected_metric["validation_loss"])


def _result_directory(
    run_root: Path,
    split: EvaluationSplit,
    checkpoint_fingerprint: str,
) -> Path:
    name = checkpoint_fingerprint.removeprefix("sha256:")
    if split is EvaluationSplit.VALIDATION:
        return run_root / "validation" / name
    return run_root / split.value


def _runtime_result_directory(
    base: Path,
    *,
    runtime_fingerprint: str,
) -> Path:
    """Keep immutable evaluation evidence addressable across runtime revisions."""

    if not base.exists():
        return base
    manifest_path = base / EVALUATION_RUNTIME_MANIFEST_FILE
    if manifest_path.is_file():
        existing = EvaluationRuntimeManifest.from_dict(_read(manifest_path))
        if existing.identity.runtime_fingerprint == runtime_fingerprint:
            return base
    suffix = runtime_fingerprint.removeprefix("sha256:")[:12]
    candidate = base.with_name(f"{base.name}-{suffix}")
    if candidate.exists():
        candidate_manifest = candidate / EVALUATION_RUNTIME_MANIFEST_FILE
        if not candidate_manifest.is_file():
            raise RuntimeError("runtime-specific evaluation directory lacks its manifest")
        existing = EvaluationRuntimeManifest.from_dict(_read(candidate_manifest))
        if existing.identity.runtime_fingerprint != runtime_fingerprint:
            raise RuntimeError("runtime evaluation prefix collision")
    return candidate


def _validate_evaluation_output_path(run_root: Path, output: Path) -> None:
    if _is_link_like(run_root) or not run_root.is_dir():
        raise RuntimeError("evaluation run root must be a real directory")
    root = run_root.resolve()
    try:
        output.absolute().relative_to(run_root.absolute())
    except ValueError as error:
        raise RuntimeError("evaluation output is not owned by the run root") from error
    ancestor = output.parent
    while not ancestor.exists() and ancestor != run_root:
        ancestor = ancestor.parent
    if (
        _is_link_like(ancestor)
        or not ancestor.is_dir()
        or not ancestor.resolve().is_relative_to(root)
    ):
        raise RuntimeError("evaluation output parent escapes the run root")
    if output.exists() and (
        _is_link_like(output) or not output.is_dir() or not output.resolve().is_relative_to(root)
    ):
        raise RuntimeError("existing evaluation output is not an owned directory")


def _archive_interrupted_evaluation(run_root: Path, staging_output: Path) -> Path:
    """Preserve an interrupted rollout instead of deleting its diagnostic evidence."""

    if (
        _is_link_like(staging_output)
        or not staging_output.is_dir()
        or staging_output.resolve().parent != run_root.resolve()
        or not (staging_output / EVALUATION_OWNER_FILE).is_file()
    ):
        raise RuntimeError("unsafe evaluation staging directory")
    owner = _read(staging_output / EVALUATION_OWNER_FILE)
    runtime = owner.get("evaluation_runtime_fingerprint")
    if isinstance(runtime, str) and runtime.startswith("sha256:"):
        archive_identity = runtime.removeprefix("sha256:")[:12]
    else:
        # Pre-M4.1 staging has no runtime fingerprint. Preserve it under a
        # stable digest of its complete legacy owner instead of deleting it.
        archive_identity = (
            "legacy-"
            + hashlib.sha256(
                json.dumps(
                    owner,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()[:12]
        )
    archive_root = run_root / "failed_evaluations"
    if archive_root.exists() and (
        _is_link_like(archive_root)
        or not archive_root.is_dir()
        or archive_root.resolve().parent != run_root.resolve()
    ):
        raise RuntimeError("unsafe interrupted-evaluation archive")
    archive_root.mkdir(exist_ok=True)
    base = f"{staging_output.name.removeprefix('.')}-{archive_identity}"
    destination = archive_root / base
    suffix = 0
    while destination.exists():
        suffix += 1
        destination = archive_root / f"{base}-{suffix:04d}"
    os.replace(staging_output, destination)
    return destination


def _validate_owned_artifact_path(run_root: Path, path: Path) -> None:
    root = run_root.resolve()
    try:
        path.absolute().relative_to(run_root.absolute())
    except ValueError as error:
        raise RuntimeError("evaluation artifact is not owned by the run root") from error
    if (
        _is_link_like(path)
        or _is_link_like(path.parent)
        or not path.resolve().is_relative_to(root)
        or not path.parent.resolve().is_relative_to(root)
    ):
        raise RuntimeError("evaluation artifact path escapes the run root")


def _maybe_lock_selection(
    run_root: Path,
    current: ValidationResult,
    *,
    run_fingerprint: str,
    expected_candidate_count: int,
) -> object:
    candidates = [current]
    validation_root = run_root / "validation"
    if (
        _is_link_like(validation_root)
        or not validation_root.is_dir()
        or validation_root.resolve().parent != run_root.resolve()
    ):
        raise RuntimeError("validation evidence root is not owned by the run")
    for path in validation_root.glob("*/validation_result.json"):
        if (
            _is_link_like(path)
            or _is_link_like(path.parent)
            or not path.parent.is_dir()
            or path.parent.resolve().parent != validation_root.resolve()
        ):
            raise RuntimeError("validation candidate directory escapes the run")
        value = _read(path)
        if value.get("checkpoint_fingerprint") == current.checkpoint_fingerprint:
            continue
        candidate = ValidationResult(**value)
        if path.parent.name != candidate.checkpoint_fingerprint.removeprefix("sha256:"):
            raise RuntimeError("validation artifact directory disagrees with its fingerprint")
        candidates.append(candidate)
    manifest = ActExperimentManifest.from_dict(_read(run_root / "run_manifest.json"))
    expected_steps = set(
        range(
            manifest.config.optimization.validation_interval,
            manifest.config.optimization.training_steps + 1,
            manifest.config.optimization.validation_interval,
        )
    )
    declared = {
        (checkpoint.checkpoint_fingerprint, checkpoint.global_step)
        for checkpoint in manifest.checkpoints
        if checkpoint.global_step in expected_steps
    }
    actual = {
        (candidate.checkpoint_fingerprint, candidate.checkpoint_step) for candidate in candidates
    }
    if (
        len(candidates) != expected_candidate_count
        or len(candidates) != len(actual)
        or actual != declared
    ):
        raise RuntimeError(
            "checkpoint selection candidates do not exactly match the run manifest's "
            "declared validation checkpoint fingerprints and steps"
        )
    selection_path = run_root / "checkpoint_selection.json"
    _validate_owned_artifact_path(run_root, selection_path)
    if selection_path.is_file():
        if _is_link_like(selection_path):
            raise RuntimeError("checkpoint selection must be an owned regular file")
        locked = load_checkpoint_selection(selection_path)
        if locked.run_fingerprint != run_fingerprint or set(locked.candidates) != set(candidates):
            raise RuntimeError("existing checkpoint selection differs from validation evidence")
    else:
        record = create_checkpoint_selection(
            run_fingerprint=run_fingerprint,
            candidates=tuple(candidates),
        )
        locked = write_checkpoint_selection_atomic(selection_path, record)
    manifest_path = run_root / "run_manifest.json"
    _validate_owned_artifact_path(run_root, manifest_path)
    manifest = _read(manifest_path)
    manifest["selected_checkpoint_fingerprint"] = locked.selected_checkpoint_fingerprint
    atomic_write_json(manifest_path, manifest)
    return locked


def _validated_final_benchmark(
    payload: dict[str, object],
    *,
    output: Path,
    identity: ActRunIdentity,
    checkpoint_fingerprint: str,
    split: EvaluationSplit,
) -> dict[str, object]:
    runtime_manifest = EvaluationRuntimeManifest.from_dict(
        _read(output / EVALUATION_RUNTIME_MANIFEST_FILE)
    )
    raw_episodes = payload.get("episodes")
    if not isinstance(raw_episodes, list):
        raise RuntimeError(f"{split.value} benchmark lacks raw episode evidence")
    episodes = tuple(_rollout_episode_from_dict(value) for value in raw_episodes)
    expected_count = {
        EvaluationSplit.VALIDATION: 6 if identity.variant is ActVariant.PER_TASK else 36,
        EvaluationSplit.TEST: 6 if identity.variant is ActVariant.PER_TASK else 36,
        EvaluationSplit.FRESH_SEED: 30 if identity.variant is ActVariant.PER_TASK else 180,
    }[split]
    task_by_id = dict(zip(CANONICAL_TASK_IDS, CANONICAL_TASK_SPECS, strict=True))
    expected_tasks = (
        {identity.task_id} if identity.variant is ActVariant.PER_TASK else set(task_by_id)
    )
    if (
        len(episodes) != expected_count
        or {episode.task_id for episode in episodes} != expected_tasks
        or any(
            episode.evaluation_id != f"{split.value}:{index:04d}"
            or episode.split is not split
            or episode.run_fingerprint != identity.run_fingerprint
            or episode.checkpoint_fingerprint != checkpoint_fingerprint
            or episode.runtime_fingerprint != runtime_manifest.identity.runtime_fingerprint
            or episode.scene_id != stable_scene_id(episode.scene_seed)
            for index, episode in enumerate(episodes)
        )
    ):
        raise RuntimeError(f"{split.value} benchmark does not match its declared episode contract")
    rebuilt = summarize_rollout_benchmark(episodes).to_dict()
    if any(payload.get(key) != expected for key, expected in rebuilt.items()):
        raise RuntimeError(f"{split.value} benchmark aggregates disagree with raw episodes")
    schedule_digest = rollout_schedule_digest(
        m3b_export_fingerprint=identity.m3b_export_fingerprint,
        variant=identity.variant.value,
        task_id=identity.task_id,
        split=split.value,
        episodes=tuple(
            {
                "scene_seed": episode.scene_seed,
                "task_spec": task_by_id[episode.task_id].to_dict(),
            }
            for episode in episodes
        ),
    )
    raw_schedule_contract = identity.data_contract.get("evaluation_schedules")
    if not isinstance(raw_schedule_contract, Mapping):
        raise RuntimeError("run identity lacks evaluation schedule evidence")
    raw_digests = raw_schedule_contract.get("digests")
    if (
        not isinstance(raw_digests, Mapping)
        or raw_digests.get(split.value) != schedule_digest
        or payload.get("schedule_digest") != schedule_digest
    ):
        raise RuntimeError(f"{split.value} benchmark schedule differs from the run identity")
    evaluation_git = payload.get("evaluation_git")
    if (
        payload.get("schema_version") != "langmani-m4.1-rollout-benchmark-v2"
        or payload.get("passed") is not True
        or payload.get("quality_validated") is not False
        or payload.get("infrastructure_failure_count") != 0
        or payload.get("environment_action_applied") is not True
        or not isinstance(evaluation_git, dict)
        or evaluation_git.get("commit") != runtime_manifest.identity.code_git_commit
        or evaluation_git.get("dirty") is not False
        or payload.get("evaluation_runtime_fingerprint")
        != runtime_manifest.identity.runtime_fingerprint
    ):
        raise RuntimeError(f"{split.value} benchmark metadata is not final-eligible")
    _validate_projection_artifact(
        output,
        payload.get("action_projection_artifact"),
        expected_runtime_fingerprint=runtime_manifest.identity.runtime_fingerprint,
    )
    return payload


def _maybe_finalize_run(
    run_root: Path,
    identity: ActRunIdentity,
    *,
    mode: ExperimentMode,
) -> bool:
    selection_path = run_root / "checkpoint_selection.json"
    if not selection_path.is_file():
        return False
    selection = load_checkpoint_selection(selection_path)
    selected_name = selection.selected_checkpoint_fingerprint.removeprefix("sha256:")
    benchmark_paths = {
        EvaluationSplit.VALIDATION: run_root / "validation" / selected_name / "benchmark.json",
        EvaluationSplit.TEST: run_root / "test" / "benchmark.json",
        EvaluationSplit.FRESH_SEED: run_root / "fresh_seed" / "benchmark.json",
    }
    analysis_path = (
        run_root
        / "reports"
        / (
            "per_task_reference.json"
            if identity.variant is ActVariant.PER_TASK
            else "counterfactual_sensitivity.json"
        )
    )
    required = [*benchmark_paths.values(), analysis_path]
    for path in required:
        _validate_owned_artifact_path(run_root, path)
    if not all(path.is_file() for path in required):
        return False
    benchmark_payloads = {
        split: _validated_final_benchmark(
            _read(path),
            output=path.parent,
            identity=identity,
            checkpoint_fingerprint=selection.selected_checkpoint_fingerprint,
            split=split,
        )
        for split, path in benchmark_paths.items()
    }
    analysis_payload = _read(analysis_path)
    required_payloads = (*benchmark_payloads.values(), analysis_payload)
    if (
        analysis_payload.get("passed") is not True
        or analysis_payload.get("checkpoint_fingerprint")
        != selection.selected_checkpoint_fingerprint
        or analysis_payload.get("run_fingerprint") != identity.run_fingerprint
    ):
        raise RuntimeError("final sensitivity artifact does not use the selected checkpoint")
    if any(
        not isinstance(payload.get("evaluation_git"), dict)
        or payload["evaluation_git"].get("dirty") is not False
        for payload in required_payloads
    ):
        raise RuntimeError("final evaluation artifacts do not use clean runtime Git evidence")
    for split in (EvaluationSplit.TEST, EvaluationSplit.FRESH_SEED):
        payload = benchmark_payloads[split]
        authorization = payload.get("test_authorization")
        if not isinstance(authorization, dict) or authorization.get("final_eligible") is not True:
            if mode is ExperimentMode.FULL:
                raise RuntimeError(f"{split.value} artifact lacks a final-eligible selection lock")
            return False
    completion = {
        "schema_version": "langmani-m4-run-completion-v1",
        "run_fingerprint": identity.run_fingerprint,
        "selected_checkpoint_fingerprint": selection.selected_checkpoint_fingerprint,
    }
    completion_path = run_root / "complete.json"
    _validate_owned_artifact_path(run_root, completion_path)
    if completion_path.is_file() and _read(completion_path) != completion:
        raise RuntimeError("existing run completion marker differs from final evaluation evidence")
    manifest_path = run_root / "run_manifest.json"
    _validate_owned_artifact_path(run_root, manifest_path)
    manifest = _read(manifest_path)
    manifest["selected_checkpoint_fingerprint"] = selection.selected_checkpoint_fingerprint
    manifest["complete"] = True
    training_state = manifest.get("training_state")
    if not isinstance(training_state, dict):
        raise RuntimeError("run manifest is missing training_state at finalization")
    training_state["completed"] = True
    atomic_write_json(manifest_path, manifest)
    _write_or_validate_immutable(completion_path, completion)
    return True


def _sensitivity(
    *,
    env: object,
    policy: object,
    preprocessor: object,
    postprocessor: object,
    identity: ActRunIdentity,
    scene_seed: int,
    tasks: tuple[TaskSpec, ...],
    expert_demonstrations: tuple[CounterfactualDemonstration, ...],
) -> tuple[object, np.ndarray, dict[str, object]]:
    if identity.variant is ActVariant.PER_TASK:
        raise ValueError("per-task sensitivity requires the separate six-policy comparison command")
    base = getattr(env, "unwrapped", env)
    raw_inputs: list[dict[str, torch.Tensor]] = []
    first_actions: list[np.ndarray] = []
    chunks: list[np.ndarray] = []
    for task in tasks:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        observation, _ = env.reset(seed=scene_seed, options={"task_spec": task.to_dict()})
        raw = build_policy_observation(
            observation,
            base_environment=base,
            variant=identity.variant,
            task_id=CANONICAL_TASK_IDS[tasks.index(task)],
            task_conditioner=(
                append_canonical_task_onehot
                if identity.variant is ActVariant.MIXED_TASK_ONEHOT
                else None
            ),
        )
        processed = preprocessor(raw)
        predicted = policy.predict_action_chunk(processed)
        actions = postprocessor(predicted).detach().cpu().numpy()[0]
        raw_inputs.append(raw)
        chunks.append(actions)
        first_actions.append(actions[0])
    stacked_chunks = np.stack(chunks)
    result = compute_counterfactual_sensitivity(
        variant=identity.variant,
        scene_id=f"fresh-sensitivity-scene:{scene_seed}",
        ordered_task_ids=CANONICAL_TASK_IDS,
        first_actions=np.stack(first_actions),
        action_chunks=stacked_chunks,
        policy_inputs=raw_inputs,
    )
    reset_state = raw_inputs[0]["observation.state"].detach().cpu().numpy().reshape(-1)[:9]
    expert_states = np.asarray(
        [item.panda_state for item in expert_demonstrations],
        dtype=np.float32,
    )
    if expert_states.shape != (6, 9) or not np.allclose(
        expert_states,
        reset_state[None, :],
        atol=1e-6,
        rtol=0,
    ):
        raise RuntimeError("policy reset Panda state differs from the M3B expert reference scene")
    reset_image = raw_inputs[0]["observation.images.base_camera"].detach().cpu().numpy()
    evidence = {
        "panda_state_equivalent": True,
        "reset_rgb_digest": hashlib.sha256(np.ascontiguousarray(reset_image).tobytes()).hexdigest(),
        "expert_decoded_rgb_digests": [item.rgb_digest for item in expert_demonstrations],
        "rgb_comparison_note": (
            "decoded M3B video may differ from the raw reset frame because of video compression; "
            "scene seed/ID and Panda state are the equality gate"
        ),
    }
    return result, stacked_chunks, evidence


def _per_task_reference(
    *,
    env: object,
    policy: object,
    preprocessor: object,
    postprocessor: object,
    identity: ActRunIdentity,
    checkpoint_fingerprint: str,
    scene_seed: int,
    task: TaskSpec,
    expert_demonstration: CounterfactualDemonstration,
) -> dict[str, object]:
    if identity.variant is not ActVariant.PER_TASK or identity.task_id is None:
        raise ValueError("per-task reference requires one per_task checkpoint")
    if expert_demonstration.task_id != identity.task_id:
        raise RuntimeError("per-task expert reference has the wrong stable task ID")
    base = getattr(env, "unwrapped", env)
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    observation, _ = env.reset(seed=scene_seed, options={"task_spec": task.to_dict()})
    episode_spec = base.get_episode_specs()[0]
    if episode_spec.task_id != identity.task_id:
        raise RuntimeError("per-task reference reset disagrees with checkpoint task")
    raw = build_policy_observation(
        observation,
        base_environment=base,
        variant=identity.variant,
        task_id=identity.task_id,
        task_conditioner=None,
    )
    predicted = policy.predict_action_chunk(preprocessor(raw))
    actions = postprocessor(predicted).detach().cpu().numpy()[0]
    image = raw["observation.images.base_camera"].detach().cpu().numpy()
    state = raw["observation.state"].detach().cpu().numpy()
    expert_state = np.asarray(expert_demonstration.panda_state, dtype=np.float32)
    if expert_state.shape != (9,) or not np.allclose(
        state.reshape(-1), expert_state, atol=1e-6, rtol=0
    ):
        raise RuntimeError("per-task reset Panda state differs from its M3B expert reference")
    expert = np.asarray(expert_demonstration.action_chunk, dtype=np.float64)
    padding = np.asarray(expert_demonstration.action_is_pad, dtype=np.bool_)
    if expert.shape != actions.shape or padding.shape != (actions.shape[0],):
        raise RuntimeError("per-task policy and expert reference chunks do not share shape")
    if not np.any(~padding):
        raise RuntimeError("per-task expert reference chunk is entirely padded")
    expert_distance = float(np.linalg.norm(actions[~padding] - expert[~padding], axis=1).mean())
    return {
        "schema_version": "langmani-m4-per-task-reference-v1",
        "passed": True,
        "run_fingerprint": identity.run_fingerprint,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "task_id": identity.task_id,
        "scene_seed": scene_seed,
        "scene_id": episode_spec.scene_id,
        "initial_rgb_digest": hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest(),
        "expert_decoded_rgb_digest": expert_demonstration.rgb_digest,
        "panda_state_equivalent": True,
        "initial_panda_state": state.tolist(),
        "first_action": actions[0].tolist(),
        "action_chunk": actions.tolist(),
        "expert_chunk_distance": expert_distance,
    }


def _rollout_episode_from_dict(value: object) -> RolloutEpisodeResult:
    if not isinstance(value, dict):
        raise RuntimeError("rollout episode record is malformed")
    payload = dict(value)
    for key in (
        "action_min",
        "action_max",
        "inference_latency_ms",
        "environment_step_latency_ms",
    ):
        raw = payload.get(key)
        if not isinstance(raw, list):
            raise RuntimeError(f"rollout episode {key} must be a JSON list")
        payload[key] = tuple(raw)
    episode = RolloutEpisodeResult(**payload)
    final = episode.final_evaluation
    required = {
        "success",
        "target_in_target_bin",
        "target_in_wrong_bin",
        "wrong_object_in_target_bin",
        "target_off_table",
        "fail",
    }
    if not required <= final.keys() or any(not isinstance(final[key], bool) for key in required):
        raise RuntimeError("rollout final_evaluation lacks required boolean M1 fields")
    if (
        final["success"] != episode.success
        or final["target_off_table"] != episode.target_off_table
        or final["fail"] != episode.target_off_table
        or episode.target_off_table != (episode.status is RolloutStatus.TARGET_OFF_TABLE)
        or episode.timeout != (episode.status in {RolloutStatus.TIMEOUT, RolloutStatus.TRUNCATED})
        or episode.invalid_action != (episode.status is RolloutStatus.INVALID_ACTION)
        or episode.inference_failure != (episode.status is RolloutStatus.INFERENCE_FAILURE)
        or (final["target_in_target_bin"] and not episode.target_in_target_bin)
        or (final["target_in_wrong_bin"] and not episode.target_in_wrong_bin)
        or (final["wrong_object_in_target_bin"] and not episode.wrong_object_in_target_bin)
        or (
            final["success"]
            and (
                not final["target_in_target_bin"]
                or final["target_in_wrong_bin"]
                or final["wrong_object_in_target_bin"]
                or final["target_off_table"]
            )
        )
    ):
        raise RuntimeError("rollout final_evaluation disagrees with status or accumulated M1 flags")
    return episode


def _validated_published_benchmark(
    payload: dict[str, object],
    *,
    identity: ActRunIdentity,
    checkpoint_fingerprint: str,
    split: EvaluationSplit,
    digest: str,
    schedule: tuple[tuple[int, TaskSpec], ...],
    evaluation_git: dict[str, object],
    authorization: dict[str, object] | None,
    runtime_manifest: EvaluationRuntimeManifest,
    output: Path,
) -> tuple[tuple[RolloutEpisodeResult, ...], RolloutBenchmarkResult]:
    raw_episodes = payload.get("episodes")
    if not isinstance(raw_episodes, list) or len(raw_episodes) != len(schedule):
        raise RuntimeError("published benchmark does not contain its exact rollout schedule")
    episodes = tuple(_rollout_episode_from_dict(value) for value in raw_episodes)
    for index, (episode, (scene_seed, task_spec)) in enumerate(
        zip(episodes, schedule, strict=True)
    ):
        if (
            episode.evaluation_id != f"{split.value}:{index:04d}"
            or episode.run_fingerprint != identity.run_fingerprint
            or episode.checkpoint_fingerprint != checkpoint_fingerprint
            or episode.runtime_fingerprint != runtime_manifest.identity.runtime_fingerprint
            or episode.action_bound_mode is not runtime_manifest.identity.action_bound_config.mode
            or episode.schedule_digest != digest
            or episode.split is not split
            or episode.scene_seed != scene_seed
            or episode.scene_id != stable_scene_id(scene_seed)
            or episode.task_id != stable_task_id(task_spec)
        ):
            raise RuntimeError("published benchmark episode order or semantic identity is invalid")
    rebuilt = summarize_rollout_benchmark(episodes)
    if any(payload.get(key) != expected for key, expected in rebuilt.to_dict().items()):
        raise RuntimeError("published benchmark aggregates disagree with raw rollout episodes")
    infrastructure_failure_count = sum(
        episode.status in {RolloutStatus.INFERENCE_FAILURE, RolloutStatus.ENVIRONMENT_FAILURE}
        for episode in episodes
    )
    environment_action_applied = any(episode.episode_steps > 0 for episode in episodes)
    passed = infrastructure_failure_count == 0 and environment_action_applied
    expected_metadata = {
        "schema_version": "langmani-m4.1-rollout-benchmark-v2",
        "passed": passed,
        "quality_validated": False,
        "infrastructure_failure_count": infrastructure_failure_count,
        "environment_action_applied": environment_action_applied,
        "evaluation_git": evaluation_git,
        "test_authorization": authorization,
        "evaluation_runtime_fingerprint": runtime_manifest.identity.runtime_fingerprint,
        "action_bound_mode": runtime_manifest.identity.action_bound_config.mode.value,
    }
    if any(payload.get(key) != expected for key, expected in expected_metadata.items()):
        raise RuntimeError("published benchmark metadata disagrees with rollout evidence")
    if not passed:
        raise RuntimeError("published benchmark is not eligible physical execution evidence")
    if _read(output / EVALUATION_RUNTIME_MANIFEST_FILE) != runtime_manifest.to_dict():
        raise RuntimeError("published evaluation runtime manifest differs from active runtime")
    _validate_projection_artifact(
        output,
        payload.get("action_projection_artifact"),
        expected_runtime_fingerprint=runtime_manifest.identity.runtime_fingerprint,
    )
    return episodes, rebuilt


def _publish_analysis(
    *,
    output: Path,
    run_root: Path,
    identity: ActRunIdentity,
    checkpoint_fingerprint: str,
    evaluation_git: dict[str, object],
) -> None:
    analysis = _read(output / EVALUATION_ANALYSIS_FILE)
    if (
        analysis.get("passed") is not True
        or analysis.get("run_fingerprint") != identity.run_fingerprint
        or analysis.get("checkpoint_fingerprint") != checkpoint_fingerprint
        or analysis.get("evaluation_git") != evaluation_git
    ):
        raise RuntimeError("published counterfactual analysis evidence is mismatched")
    analysis_name = (
        "per_task_reference.json"
        if identity.variant is ActVariant.PER_TASK
        else "counterfactual_sensitivity.json"
    )
    analysis_path = run_root / "reports" / analysis_name
    _validate_owned_artifact_path(run_root, analysis_path)
    if analysis_path.is_file():
        existing = _read(analysis_path)
        existing_git = existing.get("evaluation_git")
        same_artifact_identity = all(
            (
                existing.get("schema_version") == analysis.get("schema_version"),
                existing.get("passed") is True,
                existing.get("run_fingerprint") == identity.run_fingerprint,
                existing.get("checkpoint_fingerprint") == checkpoint_fingerprint,
                isinstance(existing_git, dict),
                isinstance(existing_git, dict) and existing_git.get("baseline_tracked") is True,
                isinstance(existing_git, dict) and existing_git.get("dirty") is False,
                isinstance(existing_git, dict) and existing_git.get("changed_paths") == [],
            )
        )
        if not same_artifact_identity:
            raise RuntimeError(f"existing immutable evaluation artifact differs: {analysis_path}")
        return
    _write_or_validate_immutable(analysis_path, analysis)


def _counterfactual_analysis_payload(
    *,
    env: object,
    completed: Any,
    loaded: LoadedActCheckpoint,
    identity: ActRunIdentity,
    evaluation_git: dict[str, object],
) -> dict[str, object]:
    first_train_index = completed.views.train.episode_indices[0]
    first_group_id = completed.views.scene_group_id_by_episode[first_train_index]
    by_task_id = {
        record.task_id: record
        for record in completed.manifest.episodes
        if record.source_scene_group_id == first_group_id
    }
    if set(by_task_id) != set(CANONICAL_TASK_IDS):
        raise RuntimeError("M3B sensitivity scene lacks the canonical six tasks")
    records = tuple(by_task_id[task_id] for task_id in CANONICAL_TASK_IDS)
    tasks = tuple(_task_spec(record) for record in records)
    expert_demonstrations = load_m3b_reference_counterfactual_group(
        completed,
        policy_config=loaded.policy.config,
    )
    if identity.variant is ActVariant.PER_TASK:
        if identity.task_id is None:
            raise RuntimeError("per-task checkpoint lacks its stable task ID")
        selected_record = by_task_id[identity.task_id]
        selected_expert = expert_demonstrations[CANONICAL_TASK_IDS.index(identity.task_id)]
        payload = _per_task_reference(
            env=env,
            policy=loaded.policy,
            preprocessor=loaded.preprocessor,
            postprocessor=loaded.postprocessor,
            identity=identity,
            checkpoint_fingerprint=loaded.record.checkpoint_fingerprint,
            scene_seed=selected_record.source_scene_seed,
            task=_task_spec(selected_record),
            expert_demonstration=selected_expert,
        )
        payload["evaluation_git"] = evaluation_git
        return payload
    sensitivity, predicted_chunks, reference_evidence = _sensitivity(
        env=env,
        policy=loaded.policy,
        preprocessor=loaded.preprocessor,
        postprocessor=loaded.postprocessor,
        identity=identity,
        scene_seed=records[0].source_scene_seed,
        tasks=tasks,
        expert_demonstrations=expert_demonstrations,
    )
    return {
        **sensitivity.to_dict(),
        "expert_chunk_distances_by_task": expert_chunk_distances(
            predicted_chunks,
            expert_demonstrations,
        ),
        **task_identity_effects(sensitivity.pairwise_chunk_distances),
        "expert_reference_evidence": reference_evidence,
        "run_fingerprint": identity.run_fingerprint,
        "checkpoint_fingerprint": loaded.record.checkpoint_fingerprint,
        "evaluation_git": evaluation_git,
        "passed": True,
    }


def _resume_existing_evaluation(
    *,
    args: argparse.Namespace,
    output: Path,
    run_root: Path,
    identity: ActRunIdentity,
    config: ActExperimentConfig,
    loaded: LoadedActCheckpoint,
    split: EvaluationSplit,
    digest: str,
    schedule: tuple[tuple[int, TaskSpec], ...],
    evaluation_git: dict[str, object],
    authorization: dict[str, object] | None,
    runtime_manifest: EvaluationRuntimeManifest,
) -> dict[str, object]:
    payload = _read(output / "benchmark.json")
    checkpoint_fingerprint = loaded.record.checkpoint_fingerprint
    episodes, benchmark = _validated_published_benchmark(
        payload,
        identity=identity,
        checkpoint_fingerprint=checkpoint_fingerprint,
        split=split,
        digest=digest,
        schedule=schedule,
        evaluation_git=evaluation_git,
        authorization=authorization,
        runtime_manifest=runtime_manifest,
        output=output,
    )
    if split is EvaluationSplit.VALIDATION:
        validation = ValidationResult(**_read(output / "validation_result.json"))
        expected_validation = ValidationResult(
            checkpoint_fingerprint=checkpoint_fingerprint,
            checkpoint_step=loaded.record.global_step,
            schedule_digest=digest,
            success_rate=benchmark.success_rate,
            wrong_object_interaction_rate=sum(
                episode.wrong_object_grasped_any or episode.wrong_object_in_target_bin
                for episode in episodes
            )
            / len(episodes),
            target_off_table_rate=sum(episode.target_off_table for episode in episodes)
            / len(episodes),
            offline_validation_action_loss=_offline_validation_loss(
                run_root,
                loaded.record.global_step,
                checkpoint_metric=loaded.training_metric,
                checkpoint_relative_path=loaded.record.relative_path,
            ),
        )
        if validation != expected_validation:
            raise RuntimeError(
                "published validation result disagrees with benchmark/checkpoint evidence"
            )
        if args.lock_selection:
            expected_candidates = len(
                range(
                    config.optimization.validation_interval,
                    config.optimization.training_steps + 1,
                    config.optimization.validation_interval,
                )
            )
            _maybe_lock_selection(
                run_root,
                validation,
                run_fingerprint=identity.run_fingerprint,
                expected_candidate_count=expected_candidates,
            )
    if args.counterfactual_sensitivity:
        _publish_analysis(
            output=output,
            run_root=run_root,
            identity=identity,
            checkpoint_fingerprint=checkpoint_fingerprint,
            evaluation_git=evaluation_git,
        )
    finalized = _maybe_finalize_run(run_root, identity, mode=config.mode)
    return {
        "schema_version": "langmani-m4-evaluation-command-v1",
        "passed": True,
        "split": split.value,
        "run_fingerprint": identity.run_fingerprint,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "schedule_digest": digest,
        "episode_count": len(schedule),
        "success_rate": payload["success_rate"],
        "output": str(output),
        "physical_execution": payload["environment_action_applied"],
        "infrastructure_failure_count": 0,
        "evaluation_git": payload["evaluation_git"],
        "evaluation_runtime_fingerprint": runtime_manifest.identity.runtime_fingerprint,
        "action_bound_mode": runtime_manifest.identity.action_bound_config.mode.value,
        "checkpoint_model_reload_validated": (runtime_manifest.checkpoint_model_reload_validated),
        "policy_processor_reload_validated": (runtime_manifest.policy_processor_reload_validated),
        "action_bound_processor_reload_validated": (
            runtime_manifest.action_bound_processor_reload_validated
        ),
        "raw_action_bounds_validated": payload["raw_action_bounds_validated"],
        "projected_action_bounds_validated": payload["projected_action_bounds_validated"],
        "task_success_count": payload["task_success_count"],
        "strict_unprojected_success_count": payload["strict_unprojected_success_count"],
        "action_projection_summary": payload["action_projection_summary"],
        "run_finalized": finalized,
        "reused_existing_evaluation": True,
    }


def _strict_bound_probe(
    *,
    env: object,
    raw_action: torch.Tensor,
    runtime_manifest: EvaluationRuntimeManifest,
) -> dict[str, object]:
    """Reproduce strict rejection, then execute one explicitly projected action."""

    reject_config = runtime_manifest.identity.action_bound_config
    if reject_config.mode is not ActionBoundMode.REJECT:
        raise RuntimeError("strict bound probe requires --action-bound-mode reject")
    reject_processor = BoundedActionEnvPostprocessorV0.from_environment(
        env,
        ActionBoundConfig.from_dict(reject_config.to_dict()),
        expected_action_components=raw_action.shape[-1],
    )
    rejected_record: ActionProjectionRecord | None = None
    try:
        reject_processor.process(raw_action, rollout_step=1)
    except ActionBoundProcessingError as error:
        rejected_record = error.record
    if rejected_record is None or not rejected_record.was_rejected:
        raise RuntimeError("strict bound probe did not reproduce an out-of-bounds action")

    project_config = ActionBoundConfig.from_dict(
        ActionBoundConfig(mode=ActionBoundMode.PROJECT).to_dict()
    )
    project_processor = BoundedActionEnvPostprocessorV0.from_environment(
        env,
        project_config,
        expected_action_components=raw_action.shape[-1],
    )
    project_result = project_processor.process(raw_action, rollout_step=1)
    if not project_result.audit_record.was_projected:
        raise RuntimeError("project bound probe did not project the known invalid action")
    project_identity = replace(
        runtime_manifest.identity,
        action_bound_config=project_config,
        runtime_fingerprint="",
    )
    project_manifest = EvaluationRuntimeManifest(
        identity=project_identity,
        checkpoint_model_reload_validated=(runtime_manifest.checkpoint_model_reload_validated),
        policy_processor_reload_validated=(runtime_manifest.policy_processor_reload_validated),
        action_bound_processor_reload_validated=True,
        deterministic_raw_action_matched=(runtime_manifest.deterministic_raw_action_matched),
        raw_action_match_tolerance=runtime_manifest.raw_action_match_tolerance,
    )
    executed = project_result.executed_action
    if not isinstance(executed, torch.Tensor):
        raise RuntimeError("strict bound probe expected a Torch action")
    # Conversion occurs only after complete bound processing, immediately before env.step.
    env.step(executed.detach().cpu().numpy()[0])
    return {
        "schema_version": "langmani-m4.1-strict-bound-probe-v0",
        "passed": True,
        "checkpoint_fingerprint": runtime_manifest.identity.checkpoint_fingerprint,
        "checkpoint_model_reload_validated": (runtime_manifest.checkpoint_model_reload_validated),
        "policy_processor_reload_validated": (runtime_manifest.policy_processor_reload_validated),
        "action_bound_processor_reload_validated": True,
        "deterministic_raw_action_matched": (runtime_manifest.deterministic_raw_action_matched),
        "reject_blocked_before_env_step": True,
        "raw_action_bounds_validated": False,
        "projected_action_bounds_validated": True,
        "real_projected_env_step_executed": True,
        "raw_action": rejected_record.to_dict()["raw_action"],
        "executed_action": project_result.audit_record.to_dict()["executed_action"],
        "reject_record": rejected_record.to_dict(),
        "project_record": project_result.audit_record.to_dict(),
        "reject_runtime_manifest": runtime_manifest.to_dict(),
        "project_runtime_manifest": project_manifest.to_dict(),
    }


def execute(args: argparse.Namespace) -> dict[str, object]:
    run_root, relative, identity, config = _checkpoint_context(args.checkpoint)
    _validate_runtime_identity(identity)
    evaluation_git = inspect_git_state(PROJECT_ROOT)
    validate_git_for_run(
        evaluation_git,
        mode=config.mode,
        allow_dirty_development=config.allow_dirty_development,
    )
    if evaluation_git.dirty:
        raise RuntimeError("M4.1 evaluation-runtime evidence requires a clean Git worktree")
    completed = load_completed_m3b_dataset(
        args.dataset_root,
        require_full=config.mode is ExperimentMode.FULL,
        validate_storage=True,
    )
    if completed.export_fingerprint != identity.m3b_export_fingerprint:
        raise RuntimeError("evaluation dataset differs from the checkpoint identity")
    loaded = load_act_checkpoint(
        run_root=run_root,
        checkpoint_relative_path=relative,
        expected_identity=identity,
    )
    split = EvaluationSplit(args.split)
    raw_schedule_contract = identity.data_contract.get("evaluation_schedules")
    if not isinstance(raw_schedule_contract, Mapping):
        raise RuntimeError("run identity lacks predeclared evaluation schedules")
    schedule_contract = dict(raw_schedule_contract)
    schedule = _split_schedule(
        completed=completed,
        identity=identity,
        config=config,
        schedule_contract=schedule_contract,
        split=split,
    )
    digest = _schedule_digest(identity, split, schedule)
    raw_digests = schedule_contract.get("digests")
    if not isinstance(raw_digests, Mapping) or raw_digests.get(split.value) != digest:
        raise RuntimeError("actual rollout schedule differs from the predeclared digest")
    expected_digest = str(raw_digests[split.value])
    selection = None
    authorization = None
    if split in {EvaluationSplit.TEST, EvaluationSplit.FRESH_SEED}:
        selection_path = run_root / "checkpoint_selection.json"
        _validate_owned_artifact_path(run_root, selection_path)
        if selection_path.exists() and _is_link_like(selection_path):
            raise RuntimeError("checkpoint selection must be an owned regular file")
        selection = load_checkpoint_selection(selection_path) if selection_path.is_file() else None
        authorization = authorize_test_evaluation(
            run_fingerprint=identity.run_fingerprint,
            checkpoint_fingerprint=loaded.record.checkpoint_fingerprint,
            actual_schedule_digest=digest,
            expected_schedule_digest=expected_digest,
            mode=config.mode,
            selection=selection,
            development_override_reason=args.development_test_override,
        )
        if not authorization.authorized:
            raise RuntimeError(
                f"{split.value} selection lock rejected evaluation: {authorization.reason}"
            )
    output = _result_directory(run_root, split, loaded.record.checkpoint_fingerprint)
    _validate_evaluation_output_path(run_root, output)
    evaluation_git_payload = evaluation_git.to_dict()
    authorization_payload = authorization.to_dict() if authorization is not None else None
    action_bound_config = ActionBoundConfig(mode=ActionBoundMode(args.action_bound_mode))
    probe_env = _create_environment(
        args.sim_backend,
        record_video=False,
        output=output,
    )
    strict_probe_result: dict[str, object] | None = None
    try:
        runtime_manifest, deterministic_raw_action = _evaluation_runtime_manifest(
            env=probe_env,
            loaded=loaded,
            run_root=run_root,
            checkpoint_relative_path=relative,
            identity=identity,
            config=config,
            action_bound_config=action_bound_config,
            evaluation_git=evaluation_git_payload,
            split=split,
            sim_backend=args.sim_backend,
            record_video=args.rollout_video,
            first_episode=schedule[0],
            output=output,
        )
        if args.strict_bound_probe:
            strict_probe_result = _strict_bound_probe(
                env=probe_env,
                raw_action=deterministic_raw_action,
                runtime_manifest=runtime_manifest,
            )
    finally:
        probe_env.close()
    if strict_probe_result is not None:
        return strict_probe_result
    runtime_fingerprint = runtime_manifest.identity.runtime_fingerprint
    output = _runtime_result_directory(
        output,
        runtime_fingerprint=runtime_fingerprint,
    )
    _validate_evaluation_output_path(run_root, output)
    runtime_manifest = replace(
        runtime_manifest,
        evaluation_output_path=str(output.resolve()),
    )
    if output.exists():
        existing_runtime = EvaluationRuntimeManifest.from_dict(
            _read(output / EVALUATION_RUNTIME_MANIFEST_FILE)
        )
        if existing_runtime != runtime_manifest:
            raise RuntimeError(
                "existing immutable evaluation uses a different action-bound runtime"
            )
        analysis_file = output / EVALUATION_ANALYSIS_FILE
        if args.counterfactual_sensitivity and not analysis_file.is_file():
            _validated_published_benchmark(
                _read(output / "benchmark.json"),
                identity=identity,
                checkpoint_fingerprint=loaded.record.checkpoint_fingerprint,
                split=split,
                digest=digest,
                schedule=schedule,
                evaluation_git=evaluation_git_payload,
                authorization=authorization_payload,
                runtime_manifest=runtime_manifest,
                output=output,
            )
            analysis_env = _create_environment(
                args.sim_backend,
                record_video=False,
                output=output,
            )
            try:
                recovered_analysis = _counterfactual_analysis_payload(
                    env=analysis_env,
                    completed=completed,
                    loaded=loaded,
                    identity=identity,
                    evaluation_git=evaluation_git_payload,
                )
            finally:
                analysis_env.close()
            _write_or_validate_immutable(analysis_file, recovered_analysis)
        return _resume_existing_evaluation(
            args=args,
            output=output,
            run_root=run_root,
            identity=identity,
            config=config,
            loaded=loaded,
            split=split,
            digest=digest,
            schedule=schedule,
            evaluation_git=evaluation_git_payload,
            authorization=authorization_payload,
            runtime_manifest=runtime_manifest,
        )
    staging_output = output.with_name(f".{output.name}.staging")
    owner = {
        "schema_version": "langmani-m4-evaluation-owner-v1",
        "run_fingerprint": identity.run_fingerprint,
        "checkpoint_fingerprint": loaded.record.checkpoint_fingerprint,
        "split": split.value,
        "schedule_digest": digest,
        "evaluation_runtime_fingerprint": runtime_fingerprint,
        "action_bound_mode": action_bound_config.mode.value,
    }
    if staging_output.exists():
        archived = _archive_interrupted_evaluation(run_root, staging_output)
        print(f"[INFO] preserved interrupted evaluation at {archived}")
    staging_output.mkdir(parents=True)
    atomic_write_json(staging_output / EVALUATION_OWNER_FILE, owner, immutable=True)
    atomic_write_json(
        staging_output / EVALUATION_RUNTIME_MANIFEST_FILE,
        runtime_manifest.to_dict(),
        immutable=True,
    )
    env = _create_environment(
        args.sim_backend,
        record_video=args.rollout_video,
        output=staging_output / "videos",
    )
    analysis_payload: dict[str, object] | None = None
    validation: ValidationResult | None = None
    projection_records: list[tuple[str, ActionProjectionRecord]] = []
    try:
        adapter = ActManiSkillRolloutAdapter(
            env=env,
            policy=loaded.policy,
            preprocessor=loaded.preprocessor,
            postprocessor=loaded.postprocessor,
            variant=identity.variant,
            run_fingerprint=identity.run_fingerprint,
            checkpoint_fingerprint=loaded.record.checkpoint_fingerprint,
            schedule_digest=digest,
            runtime_fingerprint=runtime_fingerprint,
            action_bound_config=action_bound_config,
            evaluation=config.evaluation,
            task_conditioner=(
                append_canonical_task_onehot
                if identity.variant is ActVariant.MIXED_TASK_ONEHOT
                else None
            ),
            projection_record_sink=lambda evaluation_id, record: projection_records.append(
                (evaluation_id, record)
            ),
        )
        episodes = tuple(
            adapter.run_episode(
                evaluation_id=f"{split.value}:{index:04d}",
                split=split,
                scene_seed=scene_seed,
                task_spec=task_spec,
            )
            for index, (scene_seed, task_spec) in enumerate(schedule)
        )
        benchmark = summarize_rollout_benchmark(episodes)
        infrastructure_failure_count = sum(
            episode.status in {RolloutStatus.INFERENCE_FAILURE, RolloutStatus.ENVIRONMENT_FAILURE}
            for episode in episodes
        )
        environment_action_applied = any(episode.episode_steps > 0 for episode in episodes)
        benchmark_passed = infrastructure_failure_count == 0 and environment_action_applied
        projection_artifact = _write_projection_records(
            staging_output / ACTION_PROJECTION_RECORDS_FILE,
            projection_records,
        )
        payload = {
            **benchmark.to_dict(),
            "schema_version": "langmani-m4.1-rollout-benchmark-v2",
            "passed": benchmark_passed,
            "quality_validated": False,
            "infrastructure_failure_count": infrastructure_failure_count,
            "environment_action_applied": environment_action_applied,
            "evaluation_git": evaluation_git_payload,
            "test_authorization": authorization_payload,
            "evaluation_runtime_fingerprint": runtime_fingerprint,
            "action_bound_mode": action_bound_config.mode.value,
            "action_projection_artifact": projection_artifact,
            "deterministic_reload_raw_action": deterministic_raw_action.detach()
            .cpu()
            .numpy()[0]
            .tolist(),
        }
        atomic_write_json(staging_output / "benchmark.json", payload, immutable=True)
        if not benchmark_passed:
            raise RuntimeError(
                "closed-loop benchmark is not eligible physical execution evidence: "
                f"infrastructure_failures={infrastructure_failure_count}; "
                f"environment_action_applied={environment_action_applied}"
            )
        if split is EvaluationSplit.VALIDATION:
            validation = ValidationResult(
                checkpoint_fingerprint=loaded.record.checkpoint_fingerprint,
                checkpoint_step=loaded.record.global_step,
                schedule_digest=digest,
                success_rate=benchmark.success_rate,
                wrong_object_interaction_rate=sum(
                    episode.wrong_object_grasped_any or episode.wrong_object_in_target_bin
                    for episode in episodes
                )
                / len(episodes),
                target_off_table_rate=sum(episode.target_off_table for episode in episodes)
                / len(episodes),
                offline_validation_action_loss=_offline_validation_loss(
                    run_root,
                    loaded.record.global_step,
                    checkpoint_metric=loaded.training_metric,
                    checkpoint_relative_path=loaded.record.relative_path,
                ),
            )
            atomic_write_json(
                staging_output / "validation_result.json",
                validation.to_dict(),
                immutable=True,
            )
        if args.counterfactual_sensitivity:
            analysis_payload = _counterfactual_analysis_payload(
                env=env,
                completed=completed,
                loaded=loaded,
                identity=identity,
                evaluation_git=evaluation_git_payload,
            )
    finally:
        env.close()
    if analysis_payload is not None:
        atomic_write_json(
            staging_output / EVALUATION_ANALYSIS_FILE,
            analysis_payload,
            immutable=True,
        )
    _validated_published_benchmark(
        _read(staging_output / "benchmark.json"),
        identity=identity,
        checkpoint_fingerprint=loaded.record.checkpoint_fingerprint,
        split=split,
        digest=digest,
        schedule=schedule,
        evaluation_git=evaluation_git_payload,
        authorization=authorization_payload,
        runtime_manifest=runtime_manifest,
        output=staging_output,
    )
    if (
        validation is not None
        and ValidationResult(**_read(staging_output / "validation_result.json")) != validation
    ):
        raise RuntimeError("staged validation result differs from the in-memory benchmark evidence")
    if (
        analysis_payload is not None
        and _read(staging_output / EVALUATION_ANALYSIS_FILE) != analysis_payload
    ):
        raise RuntimeError("staged counterfactual analysis differs from the computed evidence")
    os.replace(staging_output, output)
    if validation is not None and args.lock_selection:
        expected_candidates = len(
            range(
                config.optimization.validation_interval,
                config.optimization.training_steps + 1,
                config.optimization.validation_interval,
            )
        )
        _maybe_lock_selection(
            run_root,
            validation,
            run_fingerprint=identity.run_fingerprint,
            expected_candidate_count=expected_candidates,
        )
    if args.counterfactual_sensitivity:
        _publish_analysis(
            output=output,
            run_root=run_root,
            identity=identity,
            checkpoint_fingerprint=loaded.record.checkpoint_fingerprint,
            evaluation_git=evaluation_git_payload,
        )
    finalized = _maybe_finalize_run(run_root, identity, mode=config.mode)
    return {
        "schema_version": "langmani-m4-evaluation-command-v1",
        "passed": True,
        "split": split.value,
        "run_fingerprint": identity.run_fingerprint,
        "checkpoint_fingerprint": loaded.record.checkpoint_fingerprint,
        "schedule_digest": digest,
        "episode_count": len(schedule),
        "success_rate": benchmark.success_rate,
        "output": str(output),
        "physical_execution": environment_action_applied,
        "infrastructure_failure_count": infrastructure_failure_count,
        "evaluation_git": evaluation_git_payload,
        "evaluation_runtime_fingerprint": runtime_fingerprint,
        "action_bound_mode": action_bound_config.mode.value,
        "checkpoint_model_reload_validated": (runtime_manifest.checkpoint_model_reload_validated),
        "policy_processor_reload_validated": runtime_manifest.policy_processor_reload_validated,
        "action_bound_processor_reload_validated": (
            runtime_manifest.action_bound_processor_reload_validated
        ),
        "raw_action_bounds_validated": benchmark.raw_action_bounds_validated,
        "projected_action_bounds_validated": benchmark.projected_action_bounds_validated,
        "task_success_count": benchmark.task_success_count,
        "strict_unprojected_success_count": benchmark.strict_unprojected_success_count,
        "action_projection_summary": benchmark.to_dict()["action_projection_summary"],
        "run_finalized": finalized,
    }


def main() -> int:
    args = parse_args()
    try:
        report = execute(args)
    except Exception as error:  # noqa: BLE001 - command boundary preserves diagnostics
        traceback.print_exc()
        try:
            failed_git = inspect_git_state(PROJECT_ROOT).to_dict()
        except Exception as git_error:  # noqa: BLE001 - preserve secondary diagnostic
            failed_git = {
                "error_type": type(git_error).__name__,
                "error_message": str(git_error) or repr(git_error),
            }
        report = {
            "schema_version": "langmani-m4-evaluation-command-v1",
            "passed": False,
            "error_type": type(error).__name__,
            "error_message": str(error) or repr(error),
            "physical_execution": False,
            "evaluation_git": failed_git,
        }
    atomic_write_json(args.report, report)
    print(json.dumps({**report, "report": str(args.report.resolve())}, sort_keys=True))
    return 0 if report["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
