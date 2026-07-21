"""Deterministic native demonstration collection for the Phase 2 push task."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from langmani.environments.push_specs import PushTaskSpec
from langmani.environments.push_to_region import ENV_ID
from langmani.experts.push import PushToRegionExpert
from langmani.experts.push_types import PushExpertConfig, PushExpertResult
from langmani.v2.push_archive import (
    sha256_file,
    validate_push_native_episode,
)
from langmani.v2.push_dataset import (
    PushCollectionConfig,
    PushDatasetContractError,
    PushScheduledAttempt,
    build_collection_schedule,
    build_top_up_schedule,
    canonical_json,
    generation_acceptance_failures,
    quota_deficits,
    sha256_json,
)

COLLECTION_RUN_SCHEMA = "langmani-v2-phase2b-run-v0"
ATTEMPT_RECORD_SCHEMA = "langmani-v2-phase2b-attempt-record-v0"


def inspect_phase2b_runtime() -> dict[str, object]:
    """Inspect exact software/hardware identity before simulator construction."""

    import torch

    versions: dict[str, str] = {}
    for distribution in ("numpy", "mplib", "mani-skill", "lerobot"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise PushDatasetContractError(
                f"required runtime distribution is missing: {distribution}"
            ) from error
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    try:
        subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                "59ca88e9f0514187252a6286ab1b8e06c4318fb4",
                commit,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise PushDatasetContractError("accepted Candidate E is not an ancestor of HEAD") from error
    gpu = _nvidia_query()
    result: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_runtime": torch.version.cuda,
        "torch": torch.__version__,
        "numpy": versions["numpy"],
        "mplib": versions["mplib"],
        "mani_skill": versions["mani-skill"],
        "lerobot": versions["lerobot"],
        "git_commit": commit,
        "git_clean": status == "",
        "accepted_expert_commit": "59ca88e9f0514187252a6286ab1b8e06c4318fb4",
        "gpu": gpu,
    }
    if (result["numpy"], result["mplib"], result["mani_skill"], result["lerobot"]) != (
        "1.26.4",
        "0.1.1",
        "3.0.1",
        "0.6.0",
    ):
        raise PushDatasetContractError("runtime dependency versions differ from the frozen config")
    if status:
        raise PushDatasetContractError("real collection requires a clean Git worktree")
    if not gpu:
        raise PushDatasetContractError("real collection requires one visible NVIDIA GPU")
    return result


def collect_push_attempts(
    *,
    config: PushCollectionConfig,
    output_root: str | Path,
    stage: str,
    sim_backend: str = "physx_cuda",
) -> dict[str, object]:
    """Collect one immutable pilot or full schedule into closed native shards."""

    if stage not in {"pilot", "full", "top_up"}:
        raise PushDatasetContractError("collector stage must be pilot, full, or top_up")
    if sim_backend != "physx_cuda":
        raise PushDatasetContractError("Phase 2B real collection requires physx_cuda")
    runtime = inspect_phase2b_runtime()
    root = Path(output_root).resolve()
    _ensure_root_matches_config(config, root)
    if stage == "top_up":
        accepted = _read_json(root / "accepted" / "full.json").get("records")
        if not isinstance(accepted, list) or not all(isinstance(item, dict) for item in accepted):
            raise PushDatasetContractError("full replay acceptance must precede top-up")
        deficits = quota_deficits(config, cast(list[dict[str, object]], accepted))
        preferred = config.integer("preferred_accepted_episodes")
        if len(accepted) < preferred:
            deficits["total"] = preferred - len(accepted)
        schedule = build_top_up_schedule(
            config,
            deficits=deficits,
            start_index=config.integer("full_standard_attempts")
            + config.integer("full_hard_attempts"),
        )
    else:
        schedule = build_collection_schedule(config, cast(Any, stage))
    manifests = root / "manifests"
    attempts_root = root / "attempts" / stage
    stage_manifest = manifests / f"{stage}_collection.json"
    if stage_manifest.exists():
        return _read_json(stage_manifest)
    if attempts_root.exists():
        raise FileExistsError(f"incomplete immutable {stage} collection exists: {attempts_root}")
    for directory in (
        manifests,
        attempts_root,
        root / "accepted",
        root / "rejected",
        root / "audits",
        root / "videos",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    specification_path = manifests / "collection_specification.json"
    specification_payload = config.to_dict()
    _write_or_verify_json(specification_path, specification_payload)
    runtime_path = manifests / "runtime_environment.json"
    _write_or_verify_json(runtime_path, runtime)
    schedule_payload = {
        "schema_version": COLLECTION_RUN_SCHEMA,
        "collection_fingerprint": config.fingerprint,
        "stage": stage,
        "attempt_count": len(schedule),
        "attempts": [item.to_dict() for item in schedule],
    }
    _write_or_verify_json(manifests / f"{stage}_schedule.json", schedule_payload)
    run_id = (
        "langmani-v2-phase2b-"
        + sha256_json(
            {
                "config": config.fingerprint,
                "stage": stage,
                "schedule": sha256_json(schedule_payload["attempts"]),
                "code": runtime["git_commit"],
            }
        ).removeprefix("sha256:")[:24]
    )
    owner = {
        "schema_version": COLLECTION_RUN_SCHEMA,
        "collection_run_id": run_id,
        "stage": stage,
        "collection_fingerprint": config.fingerprint,
        "schedule_fingerprint": sha256_json(schedule_payload["attempts"]),
        "runtime": runtime,
        "created_unix_seconds": time.time(),
    }
    _write_or_verify_json(manifests / f"{stage}_owner.json", owner)

    records: list[dict[str, object]] = []
    shard_size = config.integer("shard_size")
    for shard_index, start in enumerate(range(0, len(schedule), shard_size)):
        shard_schedule = schedule[start : start + shard_size]
        records.extend(
            _collect_shard(
                shard_schedule,
                shard_index=shard_index,
                attempts_root=attempts_root,
                run_id=run_id,
                sim_backend=sim_backend,
            )
        )
    generation_accepted = sum(record["generation_accepted"] is True for record in records)
    summary = {
        "schema_version": COLLECTION_RUN_SCHEMA,
        "collection_run_id": run_id,
        "stage": stage,
        "collection_fingerprint": config.fingerprint,
        "runtime_environment_sha256": sha256_file(runtime_path),
        "attempt_count": len(records),
        "generation_accepted_count": generation_accepted,
        "generation_rejected_count": len(records) - generation_accepted,
        "attempt_records": records,
        "completed": True,
    }
    _write_new_json(stage_manifest, summary)
    _write_jsonl(manifests / f"{stage}_attempts.jsonl", records)
    return summary


def load_attempt_records(root: str | Path, stages: Sequence[str]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for stage in stages:
        manifest = _read_json(Path(root) / "manifests" / f"{stage}_collection.json")
        values = manifest.get("attempt_records")
        if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
            raise PushDatasetContractError(f"{stage} collection has malformed attempt records")
        records.extend(cast(list[dict[str, object]], values))
    ids = [record.get("episode_id") for record in records]
    if len(ids) != len(set(ids)):
        raise PushDatasetContractError("combined attempt records contain duplicate identities")
    return records


def _collect_shard(
    schedule: Sequence[PushScheduledAttempt],
    *,
    shard_index: int,
    attempts_root: Path,
    run_id: str,
    sim_backend: str,
) -> list[dict[str, object]]:
    import gymnasium as gym
    from mani_skill.utils.wrappers.record import RecordEpisode

    import langmani.environments  # noqa: F401

    shard = attempts_root / f"shard-{shard_index:04d}"
    shard.mkdir(parents=True, exist_ok=False)
    stem = "attempts"
    environment: Any = gym.make(
        ENV_ID,
        num_envs=1,
        obs_mode="state_dict",
        reward_mode="none",
        control_mode="pd_joint_pos",
        render_mode=None,
        sim_backend=sim_backend,
    )
    recorder = RecordEpisode(
        environment,
        output_dir=str(shard),
        trajectory_name=stem,
        save_trajectory=True,
        save_video=False,
        save_on_reset=False,
        clean_on_close=True,
        record_reward=False,
        record_env_state=True,
        source_type="motionplanning",
        source_desc="LangMani 2.0 Phase 2B accepted Candidate E push demonstrations",
    )
    pending: list[tuple[PushScheduledAttempt, PushExpertResult, int | None, dict[str, object]]] = []
    try:
        for scheduled in schedule:
            started = time.perf_counter()
            recorder.reset(
                seed=scheduled.seed,
                options={"task_spec": scheduled.task_spec.to_dict()},
            )
            base = recorder.unwrapped
            episode = base.get_episode_specs()[0]
            initial_pose = _target_pose(base, scheduled.task_spec)
            expert = PushToRegionExpert(recorder, config=PushExpertConfig())
            try:
                result = expert.run()
            except Exception as error:  # noqa: BLE001 - outer attempt boundary preserves evidence
                result = expert.unexpected_exception_result(error)
            before = _native_episode_count(shard / f"{stem}.json")
            recorder.flush_trajectory(save=True)
            after = _native_episode_count(shard / f"{stem}.json")
            native_id = after - 1 if after == before + 1 else None
            final_pose = _target_pose(base, scheduled.task_spec)
            diagnostics = {
                "episode_spec": episode.to_dict(),
                "initial_target_pose": initial_pose,
                "final_target_pose": final_pose,
                "expert_action_count": len(expert.action_trace),
                "expert_diagnostic_trace": [item.to_dict() for item in expert.diagnostic_trace],
                "approach_candidates": list(expert.approach_candidate_diagnostics),
                "wall_clock_seconds": time.perf_counter() - started,
            }
            pending.append((scheduled, result, native_id, diagnostics))
    finally:
        recorder.close()

    h5_path = shard / f"{stem}.h5"
    json_path = shard / f"{stem}.json"
    records: list[dict[str, object]] = []
    for scheduled, result, native_id, diagnostics in pending:
        validation = (
            validate_push_native_episode(h5_path, json_path, native_episode_id=native_id)
            if native_id is not None
            else None
        )
        records.append(
            _attempt_record(
                scheduled,
                result,
                diagnostics,
                validation=validation.to_dict() if validation is not None else None,
                run_id=run_id,
                h5_path=h5_path,
                json_path=json_path,
            )
        )
    shard_manifest = {
        "schema_version": COLLECTION_RUN_SCHEMA,
        "collection_run_id": run_id,
        "shard_index": shard_index,
        "h5_path": h5_path.as_posix(),
        "json_path": json_path.as_posix(),
        "h5_sha256": sha256_file(h5_path),
        "json_sha256": sha256_file(json_path),
        "attempt_records": records,
        "completed": True,
    }
    _write_new_json(shard / "manifest.json", shard_manifest)
    return records


def _attempt_record(
    scheduled: PushScheduledAttempt,
    result: PushExpertResult,
    diagnostics: Mapping[str, object],
    *,
    validation: Mapping[str, object] | None,
    run_id: str,
    h5_path: Path,
    json_path: Path,
) -> dict[str, object]:
    evaluation = dict(result.final_environment_evaluation)
    task_id = scheduled.to_dict()["task_id"]
    metadata_valid = bool(
        result.scene_seed == scheduled.seed
        and result.scene_id == scheduled.scene_group_id
        and result.task_id == task_id
        and result.target_object_id == scheduled.task_spec.target_object_id
        and result.target_region_id == scheduled.task_spec.target_region_id
        and result.difficulty == scheduled.task_spec.difficulty
    )
    native = dict(validation or {})
    record: dict[str, object] = {
        "schema_version": ATTEMPT_RECORD_SCHEMA,
        "collection_run_id": run_id,
        **scheduled.to_dict(),
        "skill_family": "pick_and_place/push_to_region",
        "object_geometry": (
            "cube" if scheduled.task_spec.target_object_id == "blue_cube" else "horizontal_cylinder"
        ),
        "expert_identity": "PushToRegionExpert/CandidateE@59ca88e9",
        "expert_result": result.to_dict(),
        "expert_success": result.success,
        "final_evaluation": evaluation,
        "success_agreement": result.success is bool(evaluation.get("success", False)),
        "failure_category": None if result.success else result.status.value,
        "episode_length": int(native.get("action_count", result.total_environment_steps)),
        "final_object_to_target_distance": evaluation.get("target_distance"),
        "stable_success_steps": evaluation.get("stable_success_steps"),
        "wrong_object_interaction": bool(
            evaluation.get("wrong_object_contact", False)
            or evaluation.get("wrong_object_displaced", False)
        ),
        "workspace_exit": bool(evaluation.get("target_outside_workspace", False)),
        "object_lift": bool(evaluation.get("target_lifted", False)),
        "object_topple": bool(evaluation.get("target_toppled", False)),
        "action_bound_event": bool(evaluation.get("action_out_of_bounds", False)),
        "action_projection_count": 0,
        "planner_retry_count": max(
            0,
            result.total_planning_calls
            - sum(item.planning_calls > 0 for item in result.phase_results),
        ),
        "correction_push_count": sum(
            item.phase.value.startswith("corrective_push") and item.environment_steps > 0
            for item in result.phase_results
        ),
        "wall_clock_seconds": diagnostics["wall_clock_seconds"],
        "task_metadata_valid": metadata_valid,
        "instruction_valid": bool(scheduled.instruction),
        "raw_h5_path": h5_path.as_posix(),
        "raw_json_path": json_path.as_posix(),
        "raw_h5_group": native.get("h5_group"),
        "native_episode_id": native.get("native_episode_id"),
        "initial_target_pose": diagnostics["initial_target_pose"],
        "final_target_pose": diagnostics["final_target_pose"],
        "expert_diagnostic_trace": diagnostics["expert_diagnostic_trace"],
        "approach_candidates": diagnostics["approach_candidates"],
        "trajectory_sha256": native.get("trajectory_sha256"),
        "initial_state_sha256": native.get("initial_state_sha256"),
        "final_state_sha256": native.get("final_state_sha256"),
        "actions_finite": native.get("actions_finite", False),
        "observations_finite": native.get("observations_finite", False),
        "time_contract_valid": native.get("time_contract_valid", False),
        "action_contract_valid": native.get("action_contract_valid", False),
        "native_final_success": native.get("native_final_success", False),
        "native_final_fail": native.get("native_final_fail", False),
        "replay_validated": False,
        "accepted": False,
    }
    failures = list(generation_acceptance_failures(record))
    if record["success_agreement"] is not True:
        failures.append("expert_environment_success_disagreement")
    if record["native_final_success"] is not result.success:
        failures.append("native_expert_success_disagreement")
    record["generation_acceptance_failures"] = failures
    record["generation_accepted"] = not failures
    record["record_fingerprint"] = sha256_json(record)
    return record


def _target_pose(base: object, task: PushTaskSpec) -> list[float]:
    actor = base.push_objects[("blue_cube", "orange_cylinder").index(task.target_object_id)]
    value = actor.pose.raw_pose.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (1, 7) or not np.all(np.isfinite(array)):
        raise PushDatasetContractError("target pose accessor returned malformed data")
    return [float(item) for item in array[0]]


def _native_episode_count(path: Path) -> int:
    if not path.exists():
        return 0
    value = _read_json(path)
    episodes = value.get("episodes")
    if not isinstance(episodes, list):
        raise PushDatasetContractError("native JSON episodes must be a list")
    return len(episodes)


def _ensure_root_matches_config(config: PushCollectionConfig, root: Path) -> None:
    configured = config.payload.get("dataset")
    if not isinstance(configured, Mapping):
        raise PushDatasetContractError("dataset configuration must be an object")
    suffix = Path(str(configured.get("output_directory"))).as_posix()
    if not root.as_posix().endswith(suffix):
        raise PushDatasetContractError("output root differs from frozen collection specification")


def _nvidia_query() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise PushDatasetContractError(f"cannot query NVIDIA runtime: {error}") from error
    rows = []
    for line in result.stdout.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) == 4:
            rows.append(
                {"name": values[0], "uuid": values[1], "driver": values[2], "memory_mib": values[3]}
            )
    return rows


def _git(*args: str) -> str:
    try:
        result = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise PushDatasetContractError(f"Git identity check failed: {error}") from error
    return result.stdout.strip()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PushDatasetContractError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise PushDatasetContractError(f"JSON root must be an object: {path}")
    return value


def _write_or_verify_json(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        if _read_json(path) != dict(value):
            raise PushDatasetContractError(f"immutable manifest differs: {path}")
        return
    _write_new_json(path, value)


def _write_new_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _write_jsonl(path: Path, values: Sequence[Mapping[str, object]]) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            for value in values:
                stream.write(canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


__all__ = [
    "ATTEMPT_RECORD_SCHEMA",
    "COLLECTION_RUN_SCHEMA",
    "collect_push_attempts",
    "inspect_phase2b_runtime",
    "load_attempt_records",
]
