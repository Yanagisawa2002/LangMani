"""Native-runtime qualification tools for official ManiSkill demonstrations.

The runtime consumes only pinned official trajectories.  It replays their
recorded actions, reconstructs a small nonprivileged visual pilot, and converts
only that pilot through the public LeRobot 0.6 API.  It contains no planner,
expert, optimizer, policy, or training path.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
import zipfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path, PurePosixPath
from typing import Any, cast

import h5py  # type: ignore[import-untyped]
import numpy as np

from langmani.datasets.policy_state import extract_panda_policy_state_v0
from langmani.v2.phase2b5 import (
    CANDIDATE_TASKS,
    CONTROL_FREQUENCY_HZ,
    CONTROL_MODE,
    HF_REPOSITORY,
    HF_REVISION,
    IMAGE_SHAPE,
    MANISKILL_SOURCE_REVISION,
    MANISKILL_VERSION,
    PANDA_ACTION_HIGH,
    PANDA_ACTION_LOW,
    PANDA_POLICY_STATE_COMPONENTS,
    PINNED_SOURCE_FILES,
    SELECTED_TASK_IDS,
    SOURCE_LICENSE,
    Phase2B5ContractError,
    aggregate_replay_gate,
    canonical_json_sha256,
    compare_replay_episode,
    padding_audit,
    select_stratified_episode_ids,
    sha256_file,
    stable_reset_identity,
    validate_action_array,
    validate_metadata_document,
    validate_policy_frame,
    validate_transition_lengths,
)

ACTION_NAMES = (
    "panda_joint1_position_target",
    "panda_joint2_position_target",
    "panda_joint3_position_target",
    "panda_joint4_position_target",
    "panda_joint5_position_target",
    "panda_joint6_position_target",
    "panda_joint7_position_target",
    "panda_gripper_normalized_command",
)
IMAGE_FEATURE = "observation.images.base_camera"
STATE_FEATURE = "observation.state"
ACTION_FEATURE = "action"
TASK_ID_FEATURE = "task_id"


class Phase2B5RuntimeError(RuntimeError):
    """Raised when native qualification cannot produce trustworthy evidence."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2B5RuntimeError(f"failed to read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise Phase2B5RuntimeError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _fingerprinted(payload: dict[str, object]) -> dict[str, object]:
    payload["fingerprint"] = canonical_json_sha256(payload)
    return payload


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _selected_paths(source_root: Path, task_id: str) -> tuple[Path, Path]:
    directory = source_root / "expanded" / task_id / "motionplanning"
    return directory / "trajectory.h5", directory / "trajectory.json"


def verify_and_extract_source_archives(source_root: Path) -> dict[str, object]:
    """Verify pinned ZIPs and safely materialize a read-only working expansion."""
    source_root = source_root.resolve()
    source_directory = source_root / "sources"
    expanded = source_root / "expanded"
    entries: list[dict[str, object]] = []
    for task_id, identity in PINNED_SOURCE_FILES.items():
        archive = source_directory / identity.file_name
        if not identity.verify(archive):
            raise Phase2B5RuntimeError(f"immutable source verification failed: {archive}")
        with zipfile.ZipFile(archive) as bundle:
            members: list[dict[str, object]] = []
            for info in bundle.infolist():
                pure = PurePosixPath(info.filename)
                mode = info.external_attr >> 16
                if pure.is_absolute() or ".." in pure.parts or (mode & 0o170000) == 0o120000:
                    raise Phase2B5RuntimeError(
                        f"unsafe official ZIP member {identity.file_name}:{info.filename}"
                    )
                members.append(
                    {
                        "path": info.filename,
                        "size_bytes": info.file_size,
                        "compressed_size_bytes": info.compress_size,
                    }
                )
            target_task = expanded / task_id
            if not target_task.exists():
                bundle.extractall(expanded)
            entries.append(
                {
                    **identity.to_dict(),
                    "official_url": (
                        f"https://huggingface.co/datasets/{HF_REPOSITORY}/resolve/"
                        f"{HF_REVISION}/demos/{identity.file_name}"
                    ),
                    "downloaded_at_utc": datetime.fromtimestamp(
                        archive.stat().st_mtime, tz=UTC
                    ).isoformat(),
                    "member_count": len(members),
                    "members": members,
                    "motionplanning_pd_joint_pos_present": (
                        f"{task_id}/motionplanning/trajectory.h5"
                        in {str(item["path"]) for item in members}
                    ),
                }
            )
    payload: dict[str, object] = {
        "schema_version": "langmani-v2-phase2b5-source-download-manifest-v0",
        "repository": HF_REPOSITORY,
        "revision": HF_REVISION,
        "license": SOURCE_LICENSE,
        "source_root": source_root.as_posix(),
        "source_root_committed": False,
        "files": entries,
        "total_downloaded_bytes": sum(item.size_bytes for item in PINNED_SOURCE_FILES.values()),
        "source_files_modified": False,
    }
    return _fingerprinted(payload)


def _leaf_lengths(group: h5py.Group) -> tuple[int, ...]:
    values: list[int] = []
    for item in group.values():
        if isinstance(item, h5py.Dataset):
            if not item.shape:
                raise Phase2B5RuntimeError(f"state dataset {item.name} is scalar")
            values.append(int(item.shape[0]))
        else:
            values.extend(_leaf_lengths(item))
    return tuple(values)


def _action_digest(actions: np.ndarray) -> str:
    array = np.ascontiguousarray(actions, dtype=np.float32)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def inspect_official_sources(
    *,
    source_root: Path,
    output_root: Path,
    task_ids: Sequence[str] = SELECTED_TASK_IDS,
) -> dict[str, object]:
    """Read every metadata record and HDF5 trajectory for selected direct sources."""
    source_manifest = verify_and_extract_source_archives(source_root)
    reports: list[dict[str, object]] = []
    for task_id in task_ids:
        if task_id not in SELECTED_TASK_IDS:
            raise Phase2B5RuntimeError(f"task is not a selected direct source: {task_id}")
        h5_path, json_path = _selected_paths(source_root, task_id)
        metadata_document = _read_json(json_path)
        episodes = validate_metadata_document(metadata_document, expected_task_id=task_id)
        episodes_by_id = {int(row["episode_id"]): row for row in episodes}
        lengths: list[int] = []
        action_digests: list[str] = []
        action_min: np.ndarray = np.full(8, np.inf, dtype=np.float64)
        action_max: np.ndarray = np.full(8, -np.inf, dtype=np.float64)
        action_values = 0
        nonfinite_values = 0
        bound_violation_values = 0
        malformed: list[dict[str, object]] = []
        total_transitions = 0
        terminal_success_agreements = 0
        with h5py.File(h5_path, "r") as trajectories:
            expected_groups = {f"traj_{episode_id}" for episode_id in episodes_by_id}
            actual_groups = set(trajectories)
            if expected_groups != actual_groups:
                raise Phase2B5RuntimeError(
                    f"{task_id} metadata/HDF5 episode group identities differ"
                )
            for episode_id, episode in sorted(episodes_by_id.items()):
                group = trajectories[f"traj_{episode_id}"]
                try:
                    actions = np.asarray(group["actions"])
                    action_report = validate_action_array(actions)
                    transition_lengths = {
                        name: int(value.shape[0])
                        for name, value in group.items()
                        if isinstance(value, h5py.Dataset) and name not in {"actions"}
                    }
                    state_lengths = _leaf_lengths(group["env_states"])
                    validate_transition_lengths(
                        action_count=int(actions.shape[0]),
                        transition_lengths=transition_lengths,
                        env_state_lengths=state_lengths,
                    )
                    if int(episode["elapsed_steps"]) != int(actions.shape[0]):
                        raise Phase2B5ContractError("elapsed_steps differs from action count")
                    h5_success = bool(np.asarray(group["success"])[-1])
                    terminal_success_agreements += int(h5_success is bool(episode["success"]))
                    lengths.append(int(actions.shape[0]))
                    total_transitions += int(actions.shape[0])
                    action_digests.append(_action_digest(actions))
                    action_min = np.minimum(
                        action_min, np.asarray(action_report["minimum"], dtype=np.float64)
                    )
                    action_max = np.maximum(
                        action_max, np.asarray(action_report["maximum"], dtype=np.float64)
                    )
                    action_values += int(actions.size)
                    nonfinite = action_report["nonfinite_values"]
                    violations = action_report["bound_violation_values"]
                    if not isinstance(nonfinite, int) or not isinstance(violations, int):
                        raise Phase2B5ContractError("action count fields must be integers")
                    nonfinite_values += nonfinite
                    bound_violation_values += violations
                except (KeyError, TypeError, ValueError, Phase2B5ContractError) as error:
                    malformed.append(
                        {
                            "episode_id": episode_id,
                            "error_type": type(error).__name__,
                            "message": str(error),
                        }
                    )
        duplicate_count = len(action_digests) - len(set(action_digests))
        sorted_lengths = sorted(lengths)
        report: dict[str, object] = {
            "task_id": task_id,
            "source_h5_relative_path": (f"expanded/{task_id}/motionplanning/trajectory.h5"),
            "source_json_relative_path": (f"expanded/{task_id}/motionplanning/trajectory.json"),
            "h5_sha256": "sha256:" + sha256_file(h5_path),
            "h5_size_bytes": h5_path.stat().st_size,
            "metadata_sha256": "sha256:" + sha256_file(json_path),
            "metadata_size_bytes": json_path.stat().st_size,
            "total_trajectories": len(episodes),
            "successful_trajectories": sum(row["success"] is True for row in episodes),
            "failed_trajectories": sum(row["success"] is False for row in episodes),
            "malformed_or_incomplete_trajectories": malformed,
            "malformed_or_incomplete_count": len(malformed),
            "total_transitions": total_transitions,
            "length": {
                "minimum": min(sorted_lengths),
                "median": statistics.median(sorted_lengths),
                "maximum": max(sorted_lengths),
            },
            "unique_episode_ids": len({int(row["episode_id"]) for row in episodes}),
            "unique_reset_identities": len(
                {stable_reset_identity(row["reset_kwargs"]) for row in episodes}
            ),
            "terminal_metadata_h5_agreement_count": terminal_success_agreements,
            "duplicate_action_trajectory_hashes": duplicate_count,
            "action": {
                "shape": ["T", 8],
                "dtype": "float32",
                "control_mode": CONTROL_MODE,
                "arm_units": "radians absolute joint-position targets",
                "joint_order": list(ACTION_NAMES),
                "gripper_semantics": "normalized mimic command in [-1,1]",
                "frequency_hz": CONTROL_FREQUENCY_HZ,
                "simulator_substeps_per_action": 5,
                "bounds_low": [float(value) for value in PANDA_ACTION_LOW],
                "bounds_high": [float(value) for value in PANDA_ACTION_HIGH],
                "minimum": [float(value) for value in action_min],
                "maximum": [float(value) for value in action_max],
                "finite_value_rate": (
                    (action_values - nonfinite_values) / action_values if action_values else 0.0
                ),
                "nonfinite_values": nonfinite_values,
                "bound_violation_values": bound_violation_values,
                "bound_violation_rate": (
                    bound_violation_values / action_values if action_values else 0.0
                ),
                "padding_in_source": False,
                "terminal_action_handling": (
                    "The final recorded action remains an executed transition; "
                    "there is no synthetic terminal action."
                ),
                "posthoc_clipping_or_repair": False,
            },
            "source_schema_valid": (
                not malformed
                and terminal_success_agreements == len(episodes)
                and nonfinite_values == 0
                and bound_violation_values == 0
            ),
        }
        reports.append(_fingerprinted(report))

    payload = _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b5-source-schema-statistics-v0",
            "created_at_utc": _timestamp(),
            "source_manifest_fingerprint": source_manifest["fingerprint"],
            "tasks": reports,
            "all_selected_sources_valid": all(
                report["source_schema_valid"] is True for report in reports
            ),
        }
    )
    _write_json(output_root / "source_download_manifest.json", source_manifest)
    _write_json(output_root / "source_schema_statistics.json", payload)
    _write_json(
        output_root / "task_candidate_runtime_audit.json",
        inspect_candidate_task_sources(source_root=source_root),
    )
    return payload


def inspect_candidate_task_sources(*, source_root: Path) -> dict[str, object]:
    """Inspect all five required official candidates, including incompatible modes."""
    try:
        import gymnasium as gym
        import mani_skill.envs  # type: ignore[import-untyped]  # noqa: F401
    except (ImportError, RuntimeError, OSError) as error:
        raise Phase2B5RuntimeError(
            "candidate audit requires the native ManiSkill runtime"
        ) from error
    objective = {
        "PickCube-v1": "cube position within 0.025 m of goal and robot static",
        "StackCube-v1": "cube A stably stacked on cube B and robot static",
        "PushCube-v1": "cube center within 0.05 m of goal region",
        "PokeCube-v1": "cube center within 0.05 m of goal after tool-mediated contact",
        "PullCube-v1": "cube center within 0.05 m of goal behind the initial object",
    }
    randomization = {
        "PickCube-v1": "cube xy pose and goal pose vary by reset seed",
        "StackCube-v1": "both cube xy poses vary by reset seed",
        "PushCube-v1": "cube and goal xy poses vary by reset seed",
        "PokeCube-v1": "cube, peg/tool, and goal placements vary by reset seed",
        "PullCube-v1": "cube and goal placements vary by reset seed",
    }
    task_reports: list[dict[str, object]] = []
    for task_id in (
        "PickCube-v1",
        "StackCube-v1",
        "PushCube-v1",
        "PokeCube-v1",
        "PullCube-v1",
    ):
        task_root = source_root / "expanded" / task_id
        sources: list[dict[str, object]] = []
        for json_path in sorted(task_root.rglob("*.json")):
            if "trajectory" not in json_path.name:
                continue
            h5_path = json_path.with_suffix(".h5")
            if not h5_path.is_file():
                continue
            document = _read_json(json_path)
            env_info = document.get("env_info")
            episodes = document.get("episodes")
            if not isinstance(env_info, dict) or not isinstance(episodes, list) or not episodes:
                raise Phase2B5RuntimeError(f"malformed candidate metadata: {json_path}")
            env_kwargs = env_info.get("env_kwargs")
            if not isinstance(env_kwargs, dict):
                raise Phase2B5RuntimeError(f"candidate env_kwargs missing: {json_path}")
            action_shape: list[object] | None = None
            action_dtype: str | None = None
            transition_count = 0
            with h5py.File(h5_path, "r") as trajectories:
                for group in trajectories.values():
                    actions = np.asarray(group["actions"])
                    if action_shape is None:
                        action_shape = ["T", int(actions.shape[1])]
                        action_dtype = str(actions.dtype)
                    elif actions.ndim != 2 or action_shape[1] != int(actions.shape[1]):
                        raise Phase2B5RuntimeError(
                            f"inconsistent candidate action shapes: {h5_path}"
                        )
                    transition_count += int(actions.shape[0])
            relative = json_path.relative_to(task_root)
            source_type = relative.parts[0] if len(relative.parts) > 1 else "unknown"
            sources.append(
                {
                    "source_type": source_type,
                    "metadata_relative_path": relative.as_posix(),
                    "metadata_sha256": "sha256:" + sha256_file(json_path),
                    "h5_relative_path": h5_path.relative_to(task_root).as_posix(),
                    "h5_sha256": "sha256:" + sha256_file(h5_path),
                    "control_mode": env_kwargs.get("control_mode"),
                    "sim_backend": env_kwargs.get("sim_backend", "physx_cpu_inferred"),
                    "action_shape": action_shape,
                    "action_dtype": action_dtype,
                    "trajectory_count": len(episodes),
                    "successful_trajectory_count": sum(
                        isinstance(row, dict) and row.get("success") is True for row in episodes
                    ),
                    "total_transitions": transition_count,
                    "official_replay_tooling_supported": True,
                }
            )
        env = gym.make(
            task_id,
            num_envs=1,
            obs_mode="rgb",
            control_mode=CONTROL_MODE,
            sim_backend="physx_cpu",
            render_backend="sapien_cuda",
            sensor_configs={"width": 256, "height": 256},
        )
        try:
            observation, _ = env.reset(seed=0)
            base: Any = env.unwrapped
            sensor_data = observation.get("sensor_data", {})
            cameras = sorted(sensor_data) if isinstance(sensor_data, Mapping) else []
            action_space: Any = env.action_space
            max_episode_steps = getattr(env.spec, "max_episode_steps", None)
            robot_uid = str(base.robot_uids)
            state_available = extract_panda_policy_state_v0(base.agent.robot).shape == (9,)
        finally:
            env.close()
        direct = [
            source
            for source in sources
            if source["control_mode"] == CONTROL_MODE
            and source["action_shape"] == ["T", 8]
            and source["action_dtype"] == "float32"
        ]
        task_reports.append(
            {
                "task_id": task_id,
                "environment_registered": True,
                "official_source_repository": HF_REPOSITORY,
                "official_source_revision": HF_REVISION,
                "license": SOURCE_LICENSE,
                "supported_robot_uid": robot_uid,
                "common_panda_actuation": True,
                "pd_joint_pos_action_space": {
                    "shape": list(action_space.shape),
                    "dtype": str(action_space.dtype),
                    "low": [float(value) for value in action_space.low],
                    "high": [float(value) for value in action_space.high],
                },
                "maximum_episode_length_registered": max_episode_steps,
                "success_predicate_summary": objective[task_id],
                "success_predicate_modified": False,
                "reset_randomization_summary": randomization[task_id],
                "rgb_camera_names": cameras,
                "base_camera_256_rgb_available": "base_camera" in cameras,
                "state_observation_available": state_available,
                "required_assets": "built-in procedural rigid-body task assets",
                "downloaded_source_count": len(sources),
                "sources": sources,
                "direct_pd_joint_pos_source_count": len(direct),
                "compatibility": ("directly_compatible" if direct else "incompatible"),
                "incompatibility_reason": (
                    None
                    if direct
                    else (
                        "Downloaded official sources contain only delta-action RL "
                        "trajectories; Phase 2B.5 does not convert or repair them."
                    )
                ),
            }
        )
    return _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b5-task-candidate-runtime-audit-v0",
            "created_at_utc": _timestamp(),
            "tasks": task_reports,
            "selected_task_ids": list(SELECTED_TASK_IDS),
            "selected_skill_families": [
                CANDIDATE_TASKS[task_id].skill_family for task_id in SELECTED_TASK_IDS
            ],
        }
    )


def _to_numpy(value: object) -> np.ndarray:
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    numpy = getattr(candidate, "numpy", None)
    if callable(numpy):
        candidate = numpy()
    return np.asarray(candidate)


def _scalar_bool(value: object) -> bool:
    array = _to_numpy(value)
    if array.size != 1:
        raise Phase2B5RuntimeError(f"expected one success value, got shape {array.shape}")
    return bool(array.reshape(-1)[0])


def _flatten_state(value: object, *, prefix: str = "") -> dict[str, np.ndarray]:
    if isinstance(value, Mapping):
        result: dict[str, np.ndarray] = {}
        for key, item in value.items():
            child = f"{prefix}/{key}" if prefix else str(key)
            result.update(_flatten_state(item, prefix=child))
        return result
    array = _to_numpy(value)
    if array.ndim > 0 and array.shape[0] == 1:
        array = array[0]
    return {prefix: np.asarray(array)}


def _h5_state_at(group: h5py.Group, index: int) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            result[key] = _h5_state_at(item, index)
        else:
            result[key] = np.asarray(item[index])
    return result


def _state_error(reference: object, observed: object) -> dict[str, object]:
    expected = _flatten_state(reference)
    actual = _flatten_state(observed)
    common = sorted(set(expected) & set(actual))
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    errors: list[np.ndarray] = []
    shape_mismatches: list[str] = []
    for key in common:
        lhs = expected[key]
        rhs = actual[key]
        if lhs.shape != rhs.shape:
            shape_mismatches.append(key)
            continue
        if np.issubdtype(lhs.dtype, np.number) and np.issubdtype(rhs.dtype, np.number):
            errors.append(np.abs(lhs.astype(np.float64) - rhs.astype(np.float64)).reshape(-1))
    concatenated = np.concatenate(errors) if errors else np.asarray([], dtype=np.float64)
    return {
        "matched_leaf_count": len(common) - len(shape_mismatches),
        "missing_leaves": missing,
        "unexpected_leaves": unexpected,
        "shape_mismatch_leaves": shape_mismatches,
        "compared_numeric_values": int(concatenated.size),
        "maximum_absolute_error": (float(np.max(concatenated)) if concatenated.size else None),
        "mean_absolute_error": (float(np.mean(concatenated)) if concatenated.size else None),
        "structures_match": not missing and not unexpected and not shape_mismatches,
    }


def _prepare_replay_initial_state(
    *,
    env: Any,
    base: Any,
    reset_kwargs: Mapping[str, Any],
    source_initial: Mapping[str, object],
) -> tuple[dict[str, object], bool, dict[str, object] | None]:
    """Prefer an exact semantic reset and use the official state anchor only as fallback."""
    env.reset(**reset_kwargs)
    semantic_error = _state_error(source_initial, base.get_state_dict())
    semantic_exact = semantic_error["structures_match"] is True and semantic_error[
        "maximum_absolute_error"
    ] in {0.0, None}
    if semantic_exact:
        return semantic_error, False, None
    base.set_state_dict(source_initial)
    anchor_error = _state_error(source_initial, base.get_state_dict())
    if anchor_error["structures_match"] is not True:
        raise Phase2B5RuntimeError("official first-state anchor structure differs")
    return semantic_error, True, anchor_error


def _load_episode_metadata(json_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document = _read_json(json_path)
    task_id = str(document.get("env_info", {}).get("env_id"))
    episodes = validate_metadata_document(document, expected_task_id=task_id)
    return document, episodes


def run_official_action_replays(
    *,
    source_root: Path,
    output_root: Path,
    replay_count: int,
    report_name: str,
    task_ids: Sequence[str] = SELECTED_TASK_IDS,
    episode_ids_by_task: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, object]:
    """Replay recorded actions from a source-state anchor with official predicates."""
    try:
        import gymnasium as gym
        import mani_skill.envs  # type: ignore[import-untyped]  # noqa: F401
    except (ImportError, RuntimeError, OSError) as import_error:
        raise Phase2B5RuntimeError(
            "official replay requires the native ManiSkill runtime"
        ) from import_error

    task_reports: list[dict[str, object]] = []
    for task_id in task_ids:
        h5_path, json_path = _selected_paths(source_root, task_id)
        document, episodes = _load_episode_metadata(json_path)
        episodes_by_id = {cast(int, row["episode_id"]): row for row in episodes}
        if episode_ids_by_task is None:
            selected_ids = select_stratified_episode_ids(episodes, count=replay_count)
        else:
            selected_ids = tuple(int(value) for value in episode_ids_by_task[task_id])
        if len(selected_ids) < min(replay_count, len(episodes)) and episode_ids_by_task is None:
            raise Phase2B5RuntimeError("stratified replay selection is unexpectedly short")
        env_kwargs = dict(document["env_info"]["env_kwargs"])
        env_kwargs.update(
            {
                "obs_mode": "state",
                "reward_mode": "none",
                "render_mode": None,
                "sim_backend": "physx_cpu",
                "num_envs": 1,
            }
        )
        started = time.perf_counter()
        env = gym.make(task_id, **env_kwargs)
        base: Any = env.unwrapped
        if str(base.control_mode) != CONTROL_MODE or int(base.control_freq) != 20:
            env.close()
            raise Phase2B5RuntimeError("native replay environment violates action timing")
        episode_reports: list[dict[str, object]] = []
        try:
            with h5py.File(h5_path, "r") as trajectories:
                for episode_id in selected_ids:
                    episode = episodes_by_id[episode_id]
                    group = trajectories[f"traj_{episode_id}"]
                    actions = np.asarray(group["actions"], dtype=np.float32)
                    action_report = validate_action_array(actions)
                    invalid_rows = int(
                        np.count_nonzero(
                            np.any(~np.isfinite(actions), axis=1)
                            | np.any(actions < PANDA_ACTION_LOW, axis=1)
                            | np.any(actions > PANDA_ACTION_HIGH, axis=1)
                        )
                    )
                    source_initial = _h5_state_at(group["env_states"], 0)
                    source_terminal = _h5_state_at(group["env_states"], len(actions))
                    source_success_values = np.asarray(group["success"], dtype=np.bool_)
                    source_success_indices = np.flatnonzero(source_success_values)
                    source_success_step = (
                        int(source_success_indices[0] + 1) if source_success_indices.size else None
                    )
                    simulator_error_count = 0
                    replayed_actions = 0
                    replay_success = False
                    replay_success_step: int | None = None
                    semantic_reset_error: dict[str, object] | None = None
                    first_state_anchor_used = False
                    first_state_anchor_error: dict[str, object] | None = None
                    terminal_error: dict[str, object] | None = None
                    episode_error: dict[str, object] | None = None
                    try:
                        (
                            semantic_reset_error,
                            first_state_anchor_used,
                            first_state_anchor_error,
                        ) = _prepare_replay_initial_state(
                            env=env,
                            base=base,
                            reset_kwargs=cast(Mapping[str, Any], episode["reset_kwargs"]),
                            source_initial=source_initial,
                        )
                        for step, action in enumerate(actions, start=1):
                            _, _, _, _, info = env.step(action)
                            replayed_actions += 1
                            current_success = _scalar_bool(info["success"])
                            if current_success and replay_success_step is None:
                                replay_success_step = step
                            replay_success = current_success
                        terminal_error = _state_error(source_terminal, base.get_state_dict())
                    except Exception as caught:
                        simulator_error_count = 1
                        episode_error = {
                            "type": type(caught).__name__,
                            "message": str(caught),
                        }
                    comparison = compare_replay_episode(
                        source_success=bool(episode["success"]),
                        replay_success=replay_success,
                        action_count=len(actions),
                        replayed_action_count=replayed_actions,
                        generated_frame_count=len(actions) + 1,
                        invalid_action_count=invalid_rows,
                        simulator_error_count=simulator_error_count,
                    )
                    episode_reports.append(
                        {
                            "episode_id": episode_id,
                            "episode_seed": episode["episode_seed"],
                            "reset_identity": stable_reset_identity(episode["reset_kwargs"]),
                            "source_length": len(actions),
                            "source_success_step": source_success_step,
                            "replay_success_step": replay_success_step,
                            "success_timing_error_steps": (
                                replay_success_step - source_success_step
                                if replay_success_step is not None
                                and source_success_step is not None
                                else None
                            ),
                            "semantic_reset_state_error": semantic_reset_error,
                            "first_state_anchor_used": first_state_anchor_used,
                            "first_state_anchor_error": first_state_anchor_error,
                            "terminal_state_error": terminal_error,
                            "action_validation": action_report,
                            "error": episode_error,
                            **comparison,
                        }
                    )
        finally:
            env.close()
        gate = aggregate_replay_gate(episode_reports)
        task_reports.append(
            _fingerprinted(
                {
                    "task_id": task_id,
                    "selection_protocol": (
                        "deterministic length-spanning sample with distinct reset identities"
                    ),
                    "selected_episode_ids": list(selected_ids),
                    "first_state_anchor": (
                        "Semantic reset is used when public state values match exactly; "
                        "ManiSkill's official first-state mechanism is a recorded fallback "
                        "only when reset reconstruction differs."
                    ),
                    "success_predicate_modified": False,
                    "planner_or_expert_invoked": False,
                    "elapsed_seconds": time.perf_counter() - started,
                    "episodes": episode_reports,
                    "gate": gate,
                }
            )
        )
    task_gate_passes: list[bool] = []
    for task_report in task_reports:
        gate_value = task_report.get("gate")
        task_gate_passes.append(
            isinstance(gate_value, Mapping) and gate_value.get("passed") is True
        )
    payload = _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b5-official-replay-results-v0",
            "created_at_utc": _timestamp(),
            "maniskill_version": MANISKILL_VERSION,
            "maniskill_source_revision": MANISKILL_SOURCE_REVISION,
            "sim_backend": "physx_cpu",
            "recorded_actions_only": True,
            "task_reports": task_reports,
            "all_tasks_passed": all(task_gate_passes),
        }
    )
    _write_json(output_root / report_name, payload)
    return payload


def _extract_rgb(observation: object) -> np.ndarray:
    if not isinstance(observation, Mapping):
        raise Phase2B5RuntimeError("RGB observation must be a mapping")
    sensor_data = observation.get("sensor_data")
    if not isinstance(sensor_data, Mapping):
        raise Phase2B5RuntimeError("RGB observation lacks sensor_data")
    camera = sensor_data.get("base_camera")
    if not isinstance(camera, Mapping) or "rgb" not in camera:
        raise Phase2B5RuntimeError("official task lacks a stable base_camera RGB field")
    rgb = _to_numpy(camera["rgb"])
    if rgb.shape != (1, *IMAGE_SHAPE) or rgb.dtype != np.dtype(np.uint8):
        raise Phase2B5RuntimeError(
            f"base_camera must provide uint8[1,256,256,3], got {rgb.shape}/{rgb.dtype}"
        )
    return np.array(rgb[0], dtype=np.uint8, copy=True)


def _write_contact_sheet(path: Path, frames: Sequence[np.ndarray]) -> None:
    try:
        from PIL import Image
    except ImportError as error:
        raise Phase2B5RuntimeError("contact sheet generation requires Pillow") from error
    if len(frames) != 4:
        raise Phase2B5RuntimeError("contact sheet requires first/middle/last/terminal frames")
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    canvas[:256, :256] = frames[0]
    canvas[:256, 256:] = frames[1]
    canvas[256:, :256] = frames[2]
    canvas[256:, 256:] = frames[3]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="RGB").save(path, format="PNG", optimize=True)


def generate_visual_policy_pilot(
    *,
    source_root: Path,
    output_root: Path,
    pilot_root: Path,
    episodes_per_task: int = 5,
    task_ids: Sequence[str] = SELECTED_TASK_IDS,
) -> dict[str, object]:
    """Replay five accepted episodes per task and capture pre-action policy frames."""
    try:
        import gymnasium as gym
        import mani_skill.envs  # type: ignore[import-untyped]  # noqa: F401
    except (ImportError, RuntimeError, OSError) as error:
        raise Phase2B5RuntimeError("visual pilot requires the native ManiSkill runtime") from error
    if pilot_root.exists():
        raise Phase2B5RuntimeError(f"pilot root is immutable and already exists: {pilot_root}")
    pilot_root.mkdir(parents=True)
    task_reports: list[dict[str, object]] = []
    for task_id in task_ids:
        h5_path, json_path = _selected_paths(source_root, task_id)
        document, episodes = _load_episode_metadata(json_path)
        selected_ids = select_stratified_episode_ids(episodes, count=episodes_per_task)
        episodes_by_id = {cast(int, row["episode_id"]): row for row in episodes}
        env_kwargs = dict(document["env_info"]["env_kwargs"])
        env_kwargs.update(
            {
                "obs_mode": "rgb",
                "reward_mode": "none",
                "render_mode": None,
                "sim_backend": "physx_cpu",
                "render_backend": "sapien_cuda",
                "sensor_configs": {"width": 256, "height": 256},
                "num_envs": 1,
            }
        )
        env = gym.make(task_id, **env_kwargs)
        base: Any = env.unwrapped
        episode_reports: list[dict[str, object]] = []
        try:
            with h5py.File(h5_path, "r") as trajectories:
                for episode_id in selected_ids:
                    episode = episodes_by_id[episode_id]
                    group = trajectories[f"traj_{episode_id}"]
                    actions = np.asarray(group["actions"], dtype=np.float32)
                    source_initial = _h5_state_at(group["env_states"], 0)
                    _prepare_replay_initial_state(
                        env=env,
                        base=base,
                        reset_kwargs=cast(Mapping[str, Any], episode["reset_kwargs"]),
                        source_initial=source_initial,
                    )
                    images: list[np.ndarray] = []
                    states: list[np.ndarray] = []
                    timestamps: list[float] = []
                    replay_success = False
                    for frame_index, action in enumerate(actions):
                        observation = base.get_obs()
                        rgb = _extract_rgb(observation)
                        state = extract_panda_policy_state_v0(base.agent.robot)
                        frame = {
                            "rgb": rgb,
                            "state": state,
                            "action": action,
                            "task_instruction": (CANDIDATE_TASKS[task_id].canonical_instruction),
                            "task_id": task_id,
                            "timestamp": frame_index / CONTROL_FREQUENCY_HZ,
                        }
                        validate_policy_frame(frame)
                        images.append(rgb)
                        states.append(state)
                        timestamps.append(frame_index / CONTROL_FREQUENCY_HZ)
                        _, _, _, _, info = env.step(action)
                        replay_success = _scalar_bool(info["success"])
                    terminal_rgb = _extract_rgb(base.get_obs())
                    if replay_success is not bool(episode["success"]):
                        raise Phase2B5RuntimeError(
                            f"{task_id} episode {episode_id} failed during visual replay"
                        )
                    episode_directory = pilot_root / task_id
                    episode_directory.mkdir(parents=True, exist_ok=True)
                    episode_path = episode_directory / f"episode_{episode_id:06d}.npz"
                    np.savez_compressed(
                        episode_path,
                        rgb=np.stack(images).astype(np.uint8, copy=False),
                        state=np.stack(states).astype(np.float32, copy=False),
                        action=actions,
                        timestamp=np.asarray(timestamps, dtype=np.float64),
                        terminal_rgb=terminal_rgb,
                    )
                    contact_sheet = (
                        output_root / "contact_sheets" / f"{task_id}_episode_{episode_id:06d}.png"
                    )
                    _write_contact_sheet(
                        contact_sheet,
                        (
                            images[0],
                            images[len(images) // 2],
                            images[-1],
                            terminal_rgb,
                        ),
                    )
                    episode_reports.append(
                        {
                            "episode_id": episode_id,
                            "source_length": len(actions),
                            "policy_frame_count": len(images),
                            "terminal_diagnostic_frame_count": 1,
                            "first_frame_semantics": "pre-action state for action[0]",
                            "last_policy_frame_semantics": (
                                "pre-action state for the final recorded action"
                            ),
                            "terminal_frame_semantics": (
                                "post-final-action diagnostic only; excluded from training frames"
                            ),
                            "frame_action_aligned": len(images) == len(actions),
                            "student_fields": [
                                "rgb",
                                "state",
                                "action",
                                "task_instruction",
                                "task_id",
                                "timestamp",
                            ],
                            "privileged_fields": [],
                            "npz_relative_path": episode_path.relative_to(pilot_root).as_posix(),
                            "npz_sha256": "sha256:" + sha256_file(episode_path),
                            "contact_sheet_relative_path": (
                                contact_sheet.relative_to(output_root).as_posix()
                            ),
                            "contact_sheet_sha256": "sha256:" + sha256_file(contact_sheet),
                            "success": replay_success,
                        }
                    )
        finally:
            env.close()
        task_reports.append(
            {
                "task_id": task_id,
                "camera_name": "base_camera",
                "robot_uid": str(base.robot_uids),
                "episodes": episode_reports,
            }
        )
    all_episode_reports = [
        episode
        for task_report in task_reports
        for episode in cast(list[dict[str, object]], task_report["episodes"])
    ]
    payload = _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b5-visual-pilot-manifest-v0",
            "created_at_utc": _timestamp(),
            "pilot_root": pilot_root.as_posix(),
            "pilot_root_committed": False,
            "image_contract": {
                "dtype": "uint8",
                "layout": "HWC_RGB",
                "shape": list(IMAGE_SHAPE),
                "camera": "base_camera",
                "crop": "none",
            },
            "state_contract": {
                "schema": "PandaPolicyStateV0",
                "dtype": "float32",
                "shape": [9],
                "components": list(PANDA_POLICY_STATE_COMPONENTS),
            },
            "action_contract": {
                "control_mode": CONTROL_MODE,
                "dtype": "float32",
                "shape": [8],
                "timing": "each frame is captured immediately before its recorded action",
            },
            "tasks": task_reports,
            "episode_count": len(all_episode_reports),
            "policy_frame_count": sum(
                cast(int, episode["policy_frame_count"]) for episode in all_episode_reports
            ),
            "privileged_fields_in_student_schema": [],
            "passed": all(
                episode["frame_action_aligned"] is True and episode["success"] is True
                for episode in all_episode_reports
            ),
        }
    )
    _write_json(output_root / "visual_pilot_manifest.json", payload)
    _write_json(
        output_root / "privilege_exclusion_audit.json",
        _fingerprinted(
            {
                "schema_version": "langmani-v2-phase2b5-privilege-exclusion-v0",
                "student_field_allowlist": [
                    "rgb",
                    "state",
                    "action",
                    "task_instruction",
                    "task_id",
                    "timestamp",
                ],
                "panda_state_components": list(PANDA_POLICY_STATE_COMPONENTS),
                "object_pose_in_student_fields": False,
                "goal_pose_in_student_fields": False,
                "environment_state_in_student_fields": False,
                "diagnostic_env_states_remain_external": True,
                "passed": True,
            }
        ),
    )
    return payload


def _distribution_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "not_installed"


def lerobot_environment_manifest() -> dict[str, object]:
    """Record the isolated runtime identity used for conversion and readback."""
    try:
        import av
        import torch
    except (ImportError, RuntimeError, OSError) as error:
        raise Phase2B5RuntimeError("LeRobot pilot requires working Torch and PyAV") from error
    versions = {
        name: _distribution_version(name)
        for name in (
            "lerobot",
            "torch",
            "transformers",
            "diffusers",
            "numpy",
            "av",
            "pillow",
            "datasets",
            "pyarrow",
            "pandas",
        )
    }
    if versions["lerobot"] != "0.6.0":
        raise Phase2B5RuntimeError(f"expected LeRobot 0.6.0, got {versions['lerobot']}")
    torch_major_minor = tuple(int(item) for item in versions["torch"].split("+")[0].split(".")[:2])
    if torch_major_minor < (2, 7):
        raise Phase2B5RuntimeError(f"expected Torch >=2.7, got {versions['torch']}")
    return _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b5-lerobot-060-environment-v0",
            "created_at_utc": _timestamp(),
            "environment_prefix": os.environ.get("CONDA_PREFIX") or os.environ.get("VIRTUAL_ENV"),
            "python": platform.python_version(),
            "packages": versions,
            "cuda_runtime_reported_by_torch": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "lerobot_tag": "v0.6.0",
            "lerobot_commit": "installed PyPI sdist/wheel 0.6.0; no Git checkout",
            "pyav_libraries": {
                name: ".".join(str(value) for value in version)
                for name, version in sorted(av.library_versions.items())
            },
            "isolated_from_planner_environment": True,
            "package_installation_into_planner_or_ppo_environment": False,
        }
    )


def convert_visual_pilot_to_lerobot(
    *,
    pilot_root: Path,
    dataset_root: Path,
    output_root: Path,
    visual_manifest_path: Path,
) -> dict[str, object]:
    """Convert only the bounded NPZ pilot through LeRobot's public writer."""
    try:
        from lerobot.configs import RGBEncoderConfig  # type: ignore[import-untyped]
        from lerobot.datasets import LeRobotDataset  # type: ignore[import-untyped]
    except (ImportError, RuntimeError, OSError) as error:
        raise Phase2B5RuntimeError("LeRobot 0.6 public dataset API is unavailable") from error
    environment = lerobot_environment_manifest()
    if dataset_root.exists():
        raise Phase2B5RuntimeError(f"pilot dataset root already exists: {dataset_root}")
    visual = _read_json(visual_manifest_path)
    features = {
        IMAGE_FEATURE: {
            "dtype": "video",
            "shape": IMAGE_SHAPE,
            "names": ["height", "width", "channels"],
        },
        STATE_FEATURE: {
            "dtype": "float32",
            "shape": (9,),
            "names": list(PANDA_POLICY_STATE_COMPONENTS),
        },
        ACTION_FEATURE: {
            "dtype": "float32",
            "shape": (8,),
            "names": list(ACTION_NAMES),
        },
        TASK_ID_FEATURE: {
            "dtype": "string",
            "shape": (1,),
            "names": None,
        },
    }
    encoder = RGBEncoderConfig(
        vcodec="h264",
        pix_fmt="yuv444p",
        g=2,
        crf=18,
        preset="medium",
        fast_decode=0,
        video_backend="pyav",
    )
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(
        repo_id="langmani/phase2b5-official-pilot-v0",
        fps=CONTROL_FREQUENCY_HZ,
        features=features,
        root=dataset_root,
        robot_type="panda",
        use_videos=True,
        video_backend="pyav",
        rgb_encoder=encoder,
        encoder_threads=1,
        image_writer_processes=0,
        image_writer_threads=0,
        batch_encoding_size=1,
        streaming_encoding=False,
        data_files_size_in_mb=100,
        video_files_size_in_mb=500,
    )
    episode_rows: list[dict[str, object]] = []
    total_frames = 0
    try:
        for task in visual["tasks"]:
            task_id = str(task["task_id"])
            instruction = CANDIDATE_TASKS[task_id].canonical_instruction
            for episode in task["episodes"]:
                path = pilot_root / str(episode["npz_relative_path"])
                with np.load(path, allow_pickle=False) as payload:
                    rgb = np.asarray(payload["rgb"], dtype=np.uint8)
                    state = np.asarray(payload["state"], dtype=np.float32)
                    actions = np.asarray(payload["action"], dtype=np.float32)
                    timestamps = np.asarray(payload["timestamp"], dtype=np.float64)
                if not (
                    len(rgb) == len(state) == len(actions) == len(timestamps)
                    and np.allclose(
                        timestamps,
                        np.arange(len(actions), dtype=np.float64) / CONTROL_FREQUENCY_HZ,
                    )
                ):
                    raise Phase2B5RuntimeError("pilot frame/action/timestamp alignment failed")
                for frame_index in range(len(actions)):
                    frame = {
                        "rgb": rgb[frame_index],
                        "state": state[frame_index],
                        "action": actions[frame_index],
                        "task_instruction": instruction,
                        "task_id": task_id,
                        "timestamp": float(timestamps[frame_index]),
                    }
                    validate_policy_frame(frame)
                    dataset.add_frame(
                        {
                            IMAGE_FEATURE: rgb[frame_index],
                            STATE_FEATURE: state[frame_index],
                            ACTION_FEATURE: actions[frame_index],
                            TASK_ID_FEATURE: task_id,
                            "task": instruction,
                        }
                    )
                dataset.save_episode(parallel_encoding=False)
                total_frames += len(actions)
                episode_rows.append(
                    {
                        "task_id": task_id,
                        "source_episode_id": episode["episode_id"],
                        "converted_episode_index": len(episode_rows),
                        "frame_count": len(actions),
                        "canonical_instruction": instruction,
                    }
                )
        dataset.finalize()
    except Exception:
        if getattr(dataset, "has_pending_frames", lambda: False)():
            dataset.clear_episode_buffer(delete_images=True)
        raise
    _write_json(
        dataset_root / "langmani_phase2b5_episode_index.json",
        {
            "schema_version": "langmani-v2-phase2b5-pilot-episode-index-v0",
            "episodes": episode_rows,
        },
    )
    manifest = _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b5-lerobot-conversion-pilot-v0",
            "created_at_utc": _timestamp(),
            "dataset_root": dataset_root.as_posix(),
            "dataset_root_committed": False,
            "repo_id": "langmani/phase2b5-official-pilot-v0",
            "scope": "bounded conversion pilot; not full dataset production",
            "episode_count": len(episode_rows),
            "frame_count": total_frames,
            "task_ids": sorted({str(row["task_id"]) for row in episode_rows}),
            "features": {
                IMAGE_FEATURE: {"dtype": "video", "shape": list(IMAGE_SHAPE)},
                STATE_FEATURE: {"dtype": "float32", "shape": [9]},
                ACTION_FEATURE: {"dtype": "float32", "shape": [8]},
                TASK_ID_FEATURE: {"dtype": "string"},
                "task": {"semantic": "canonical natural-language instruction"},
                "episode_index": {"managed_by_lerobot": True},
                "frame_index": {"managed_by_lerobot": True},
                "timestamp": {"managed_by_lerobot": True},
            },
            "episodes": episode_rows,
            "processor_compatibility": (
                "Dataset features use public LeRobot 0.6 conventions; no policy processor "
                "was loaded because model loading and training are outside this phase."
            ),
            "full_dataset_production_started": False,
            "passed": len(episode_rows) == 5 * len(SELECTED_TASK_IDS),
        }
    )
    _write_json(output_root / "lerobot_060_environment_manifest.json", environment)
    _write_json(output_root / "conversion_pilot_manifest.json", manifest)
    return manifest


def readback_lerobot_pilot(
    *,
    dataset_root: Path,
    output_root: Path,
    conversion_manifest_path: Path,
) -> dict[str, object]:
    """Load and random-access every converted frame through LeRobot 0.6."""
    try:
        from lerobot.datasets import LeRobotDataset  # type: ignore[import-untyped]
    except (ImportError, RuntimeError, OSError) as error:
        raise Phase2B5RuntimeError("LeRobot 0.6 readback API is unavailable") from error
    conversion = _read_json(conversion_manifest_path)
    dataset = LeRobotDataset(
        repo_id=str(conversion["repo_id"]),
        root=dataset_root,
        download_videos=False,
    )
    row_count = len(dataset)
    image_failures = 0
    state_failures = 0
    action_failures = 0
    task_id_failures = 0
    index_failures = 0
    observed_tasks: set[str] = set()
    sampled_indices = sorted({0, row_count // 4, row_count // 2, 3 * row_count // 4, row_count - 1})
    sampled_rows: list[dict[str, object]] = []
    previous_episode = -1
    previous_frame = -1
    for index in range(row_count):
        row = dataset[index]
        image = _to_numpy(row[IMAGE_FEATURE])
        state = _to_numpy(row[STATE_FEATURE])
        action = _to_numpy(row[ACTION_FEATURE])
        task_id = row[TASK_ID_FEATURE]
        if image.shape not in {(3, 256, 256), IMAGE_SHAPE} or not np.all(np.isfinite(image)):
            image_failures += 1
        if state.shape != (9,) or not np.all(np.isfinite(state)):
            state_failures += 1
        if action.shape != (8,) or not np.all(np.isfinite(action)):
            action_failures += 1
        if not isinstance(task_id, str) or task_id not in SELECTED_TASK_IDS:
            task_id_failures += 1
        else:
            observed_tasks.add(task_id)
        episode_index = int(_to_numpy(row["episode_index"]).reshape(-1)[0])
        frame_index = int(_to_numpy(row["frame_index"]).reshape(-1)[0])
        timestamp = float(_to_numpy(row["timestamp"]).reshape(-1)[0])
        if episode_index == previous_episode:
            if frame_index != previous_frame + 1:
                index_failures += 1
        elif frame_index != 0 or episode_index != previous_episode + 1:
            index_failures += 1
        if abs(timestamp - frame_index / CONTROL_FREQUENCY_HZ) > 1e-6:
            index_failures += 1
        previous_episode = episode_index
        previous_frame = frame_index
        if index in sampled_indices:
            sampled_rows.append(
                {
                    "index": index,
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "task_id": task_id,
                    "image_shape": list(image.shape),
                    "state_shape": list(state.shape),
                    "action_shape": list(action.shape),
                }
            )
    metadata_episode_count = int(dataset.meta.total_episodes)
    metadata_frame_count = int(dataset.meta.total_frames)
    report = _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b5-lerobot-readback-v0",
            "created_at_utc": _timestamp(),
            "lerobot_version": _distribution_version("lerobot"),
            "dataset_length": row_count,
            "metadata_episode_count": metadata_episode_count,
            "metadata_frame_count": metadata_frame_count,
            "expected_episode_count": conversion["episode_count"],
            "expected_frame_count": conversion["frame_count"],
            "observed_task_ids": sorted(observed_tasks),
            "image_decode_failures": image_failures,
            "state_shape_or_value_failures": state_failures,
            "action_shape_or_value_failures": action_failures,
            "task_id_failures": task_id_failures,
            "index_or_timestamp_failures": index_failures,
            "sampled_random_access_rows": sampled_rows,
            "passed": (
                row_count == int(conversion["frame_count"])
                and metadata_episode_count == int(conversion["episode_count"])
                and metadata_frame_count == int(conversion["frame_count"])
                and observed_tasks == set(SELECTED_TASK_IDS)
                and image_failures == 0
                and state_failures == 0
                and action_failures == 0
                and task_id_failures == 0
                and index_failures == 0
            ),
        }
    )
    _write_json(output_root / "lerobot_readback_result.json", report)
    return report


def write_padding_report(*, visual_manifest_path: Path, output_root: Path) -> dict[str, object]:
    visual = _read_json(visual_manifest_path)
    lengths = [
        int(episode["policy_frame_count"])
        for task in visual["tasks"]
        for episode in task["episodes"]
    ]
    report = _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b5-padding-audit-v0",
            **padding_audit(lengths),
        }
    )
    _write_json(output_root / "padding_audit.json", report)
    return report


def runtime_audit(*, execution_commit: str, network_turbo_sourced: bool) -> dict[str, object]:
    """Record the read-only qualification runtime and explicit no-training state."""
    gpu: dict[str, object] = {"available": False}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        fields = [part.strip() for part in result.stdout.strip().split(",", maxsplit=4)]
        gpu = {
            "available": True,
            "uuid": fields[0],
            "name": fields[1],
            "driver": fields[2],
            "memory_used_mib": int(fields[3]),
            "memory_total_mib": int(fields[4]),
        }
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    return _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b5-remote-execution-audit-v0",
            "created_at_utc": _timestamp(),
            "hostname": platform.node(),
            "execution_commit": execution_commit,
            "network_turbo_sourced": network_turbo_sourced,
            "runtime_python": platform.python_version(),
            "maniskill_version": _distribution_version("mani_skill"),
            "torch_version": _distribution_version("torch"),
            "gpu": gpu,
            "recorded_action_replay_only": True,
            "official_source_download_only": True,
            "custom_expert_started": False,
            "planner_started": False,
            "optimizer_created": False,
            "backward_passes": 0,
            "training_started": False,
            "full_dataset_export_started": False,
        }
    )


__all__ = [
    "ACTION_FEATURE",
    "IMAGE_FEATURE",
    "STATE_FEATURE",
    "TASK_ID_FEATURE",
    "Phase2B5RuntimeError",
    "convert_visual_pilot_to_lerobot",
    "generate_visual_policy_pilot",
    "inspect_official_sources",
    "lerobot_environment_manifest",
    "readback_lerobot_pilot",
    "run_official_action_replays",
    "runtime_audit",
    "verify_and_extract_source_archives",
    "write_padding_report",
]
