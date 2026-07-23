"""LeRobot 0.6 conversion, readback, archive, and restore for Phase 2B.6.

This module consumes completed nonprivileged replay arrays.  It never imports
or instantiates a policy, model, optimizer, checkpoint, or training processor.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, cast

import h5py  # type: ignore[import-untyped]
import numpy as np

from langmani.datasets.policy_state import PANDA_POLICY_STATE_COMPONENTS
from langmani.v2.phase2b2 import (
    create_deterministic_tar_gz,
    file_tree_manifest,
    restore_validate_archive,
)
from langmani.v2.phase2b5 import (
    CANDIDATE_TASKS,
    CONTROL_FREQUENCY_HZ,
    IMAGE_SHAPE,
    sha256_file,
)
from langmani.v2.phase2b5_runtime import ACTION_NAMES
from langmani.v2.phase2b6 import (
    DATASET_PACKAGE_ID,
    EXPECTED_EPISODES,
    EXPECTED_TRANSITIONS,
    PRIMARY_SPLITS,
    TASK_IDS,
    apply_visual_shift,
    audit_primary_split_leakage,
    authorization_state,
    classify_result,
    create_accepted_dataset_package,
    fingerprinted,
    load_production_spec,
    validate_student_feature_names,
)

IMAGE_FEATURE = "observation.images.base_camera"
STATE_FEATURE = "observation.state"
ACTION_FEATURE = "action"
TASK_ID_FEATURE = "task_id"
SKILL_FEATURE = "skill_family"
SOURCE_IDENTITY_FEATURE = "source_episode_identity"
DERIVED_IDENTITY_FEATURE = "derived_episode_identity"
TEMPLATE_ID_FEATURE = "instruction_template_id"
SPLIT_FEATURE = "primary_split"
PRIMARY_DERIVED_IDENTITY_FEATURE = "primary_derived_episode_identity"
VISUAL_SHIFT_ID_FEATURE = "visual_shift_identity"


class Phase2B6LeRobotError(RuntimeError):
    """Raised when conversion, readback, archive, or restore fails closed."""


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2B6LeRobotError(f"failed to read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise Phase2B6LeRobotError(f"{path} must contain one object")
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.parent / f".{path.name}.partial"
    if staging.exists():
        raise Phase2B6LeRobotError(f"stale JSON staging file exists: {staging}")
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with staging.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()


def _distribution_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "not_installed"


def lerobot_environment_manifest() -> dict[str, object]:
    """Record and enforce the isolated public LeRobot 0.6 runtime."""

    try:
        import av
        import torch
    except (ImportError, OSError, RuntimeError) as error:
        raise Phase2B6LeRobotError("LeRobot conversion requires Torch and PyAV") from error
    packages = {
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
            "h5py",
        )
    }
    if packages["lerobot"] != "0.6.0":
        raise Phase2B6LeRobotError(f"expected LeRobot 0.6.0, got {packages['lerobot']}")
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-lerobot-environment-v0",
            "created_at_utc": _timestamp(),
            "python": platform.python_version(),
            "executable": Path(os.sys.executable).resolve().as_posix(),
            "environment_prefix": (
                os.environ.get("CONDA_PREFIX")
                or os.environ.get("VIRTUAL_ENV")
                or Path(os.sys.executable).resolve().parents[1].as_posix()
            ),
            "packages": packages,
            "cuda_runtime_reported_by_torch": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "lerobot_tag": "v0.6.0",
            "lerobot_distribution_source": "installed PyPI sdist/wheel 0.6.0",
            "pyav_libraries": {
                name: ".".join(str(value) for value in version)
                for name, version in sorted(av.library_versions.items())
            },
            "isolated_from_maniskill_replay_environment": True,
            "model_or_policy_loaded": False,
            "optimizer_created": False,
            "backward_passes": 0,
            "passed": True,
        }
    )


def _task_slug(task_id: str) -> str:
    return task_id.removesuffix("-v1").lower()


def _array_sha256(value: object, *, dtype: np.dtype[Any] | None = None) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return "sha256:" + hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _load_assignments(production_root: Path) -> list[dict[str, object]]:
    document = _read_json(production_root / "work" / "primary_split_manifest.json")
    rows = document.get("assignments")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise Phase2B6LeRobotError("primary split assignments are malformed")
    return cast(list[dict[str, object]], rows)


def _load_replay_rows(production_root: Path) -> list[dict[str, object]]:
    document = _read_json(production_root / "primary" / "metadata" / "replay_inventory.json")
    rows = document.get("episodes")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise Phase2B6LeRobotError("replay inventory is malformed")
    return cast(list[dict[str, object]], rows)


def _base_features() -> dict[str, dict[str, object]]:
    return {
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
        TASK_ID_FEATURE: {"dtype": "string", "shape": (1,), "names": None},
        SKILL_FEATURE: {"dtype": "string", "shape": (1,), "names": None},
        SOURCE_IDENTITY_FEATURE: {"dtype": "string", "shape": (1,), "names": None},
        DERIVED_IDENTITY_FEATURE: {"dtype": "string", "shape": (1,), "names": None},
        TEMPLATE_ID_FEATURE: {"dtype": "string", "shape": (1,), "names": None},
        SPLIT_FEATURE: {"dtype": "string", "shape": (1,), "names": None},
    }


def _writer(*, repo_id: str, root: Path, visual_shift: bool = False) -> Any:
    try:
        from lerobot.configs import RGBEncoderConfig  # type: ignore[import-untyped]
        from lerobot.datasets import LeRobotDataset  # type: ignore[import-untyped]
    except (ImportError, OSError, RuntimeError) as error:
        raise Phase2B6LeRobotError("LeRobot 0.6 public writer is unavailable") from error
    features = _base_features()
    if visual_shift:
        features[PRIMARY_DERIVED_IDENTITY_FEATURE] = {
            "dtype": "string",
            "shape": (1,),
            "names": None,
        }
        features[VISUAL_SHIFT_ID_FEATURE] = {
            "dtype": "string",
            "shape": (1,),
            "names": None,
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
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=CONTROL_FREQUENCY_HZ,
        features=features,
        root=root,
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
        metadata_buffer_size=25,
    )


def _validate_npz(
    *,
    path: Path,
    replay: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if path.is_symlink() or not path.is_file():
        raise Phase2B6LeRobotError(f"raw replay episode is unavailable: {path}")
    if "sha256:" + sha256_file(path) != replay["raw_sha256"]:
        raise Phase2B6LeRobotError(f"raw replay episode hash changed: {path}")
    with np.load(path, allow_pickle=False) as payload:
        rgb = np.asarray(payload["rgb"], dtype=np.uint8)
        state = np.asarray(payload["state"], dtype=np.float32)
        action = np.asarray(payload["action"], dtype=np.float32)
        timestamp = np.asarray(payload["timestamp"], dtype=np.float64)
    length = int(replay["source_action_count"])
    if (
        rgb.shape != (length, *IMAGE_SHAPE)
        or state.shape != (length, 9)
        or action.shape != (length, 8)
        or timestamp.shape != (length,)
    ):
        raise Phase2B6LeRobotError("raw replay arrays violate frame/action schema")
    if rgb.dtype != np.dtype(np.uint8):
        raise Phase2B6LeRobotError("raw replay RGB dtype changed")
    if state.dtype != np.dtype(np.float32) or action.dtype != np.dtype(np.float32):
        raise Phase2B6LeRobotError("raw replay state/action dtype changed")
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
        raise Phase2B6LeRobotError("raw replay state/action contains nonfinite values")
    expected_timestamp = np.arange(length, dtype=np.float64) / CONTROL_FREQUENCY_HZ
    if not np.array_equal(timestamp, expected_timestamp):
        raise Phase2B6LeRobotError("raw replay timestamp convention changed")
    if _array_sha256(action, dtype=np.dtype(np.float32)) != replay["source_action_sha256"]:
        raise Phase2B6LeRobotError("raw replay action bytes differ from official source")
    return rgb, state, action, timestamp


def _split_repo_id(task_id: str, split: str) -> str:
    return f"langmani/official-{_task_slug(task_id)}-v1-{split.replace('_', '-')}"


def _visual_repo_id(task_id: str) -> str:
    return f"langmani/official-{_task_slug(task_id)}-v1-visual-shift"


def convert_task_split_roots(
    *,
    production_root: Path,
    evidence_root: Path,
    spec_path: Path,
) -> dict[str, object]:
    """Build one task container with five disjoint LeRobot roots per task."""

    spec = load_production_spec(spec_path)
    environment = lerobot_environment_manifest()
    validate_student_feature_names(
        set(_base_features()) | {"task", "episode_index", "frame_index", "timestamp"}
    )
    assignments = _load_assignments(production_root)
    replay_rows = _load_replay_rows(production_root)
    replay_by_identity = {str(row["derived_episode_identity"]): row for row in replay_rows}
    primary = production_root / "primary"
    task_reports: dict[str, object] = {}
    converted_episodes = 0
    converted_frames = 0
    started = time.perf_counter()
    for task_id in TASK_IDS:
        split_reports: dict[str, object] = {}
        for split in PRIMARY_SPLITS:
            rows = sorted(
                [
                    row
                    for row in assignments
                    if row["task_id"] == task_id and row["primary_split"] == split
                ],
                key=lambda row: int(row["source_episode_id"]),
            )
            final_root = primary / "task_roots" / _task_slug(task_id) / "splits" / split
            staging_root = final_root.parent / f".{split}.partial"
            if final_root.exists():
                sidecar_path = final_root / "langmani_phase2b6_split_manifest.json"
                if not sidecar_path.is_file():
                    raise Phase2B6LeRobotError(
                        f"existing task split root is incomplete: {final_root}"
                    )
                sidecar = _read_json(sidecar_path)
                if sidecar.get("passed") is not True:
                    raise Phase2B6LeRobotError(f"existing task split root failed: {final_root}")
                split_reports[split] = sidecar
                converted_episodes += int(sidecar["episode_count"])
                converted_frames += int(sidecar["frame_count"])
                continue
            if staging_root.exists():
                raise Phase2B6LeRobotError(
                    f"preserved partial conversion requires review: {staging_root}"
                )
            staging_root.parent.mkdir(parents=True, exist_ok=True)
            repo_id = _split_repo_id(task_id, split)
            dataset = _writer(repo_id=repo_id, root=staging_root)
            episode_index_rows: list[dict[str, object]] = []
            global_frame = 0
            try:
                for local_episode_index, assignment in enumerate(rows):
                    identity = str(assignment["derived_episode_identity"])
                    replay = replay_by_identity[identity]
                    path = production_root / str(replay["raw_relative_path"])
                    rgb, state, action, timestamp = _validate_npz(path=path, replay=replay)
                    instruction = str(assignment["instruction"])
                    for frame_index in range(len(action)):
                        dataset.add_frame(
                            {
                                IMAGE_FEATURE: rgb[frame_index],
                                STATE_FEATURE: state[frame_index],
                                ACTION_FEATURE: action[frame_index],
                                TASK_ID_FEATURE: task_id,
                                SKILL_FEATURE: assignment["skill_family"],
                                SOURCE_IDENTITY_FEATURE: assignment["source_trajectory_identity"],
                                DERIVED_IDENTITY_FEATURE: identity,
                                TEMPLATE_ID_FEATURE: assignment["instruction_template_id"],
                                SPLIT_FEATURE: split,
                                "task": instruction,
                            }
                        )
                    dataset.save_episode(parallel_encoding=False)
                    episode_index_rows.append(
                        {
                            "local_episode_index": local_episode_index,
                            "global_frame_start": global_frame,
                            "frame_count": len(action),
                            "source_episode_id": assignment["source_episode_id"],
                            "source_trajectory_identity": assignment["source_trajectory_identity"],
                            "derived_episode_identity": identity,
                            "reset_identity": assignment["reset_identity"],
                            "action_sha256": assignment["action_sha256"],
                            "instruction_template_id": assignment["instruction_template_id"],
                            "instruction": instruction,
                            "primary_split": split,
                        }
                    )
                    global_frame += len(action)
                    converted_episodes += 1
                    converted_frames += len(action)
                    if converted_episodes % 25 == 0:
                        print(
                            json.dumps(
                                {
                                    "phase": "lerobot_conversion",
                                    "completed_episodes": converted_episodes,
                                    "total_episodes": EXPECTED_EPISODES,
                                    "elapsed_seconds": time.perf_counter() - started,
                                    "task_id": task_id,
                                    "split": split,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                dataset.finalize()
            except Exception:
                if getattr(dataset, "has_pending_frames", lambda: False)():
                    dataset.clear_episode_buffer(delete_images=True)
                raise
            payload_tree = file_tree_manifest(staging_root)
            sidecar = fingerprinted(
                {
                    "schema_version": "langmani-v2-phase2b6-task-split-root-v0",
                    "task_id": task_id,
                    "skill_family": CANDIDATE_TASKS[task_id].skill_family,
                    "primary_split": split,
                    "repo_id": repo_id,
                    "episode_count": len(rows),
                    "frame_count": global_frame,
                    "features": {
                        IMAGE_FEATURE: {"dtype": "video", "shape": list(IMAGE_SHAPE)},
                        STATE_FEATURE: {"dtype": "float32", "shape": [9]},
                        ACTION_FEATURE: {"dtype": "float32", "shape": [8]},
                        TASK_ID_FEATURE: {"dtype": "string"},
                        SKILL_FEATURE: {"dtype": "string"},
                        SOURCE_IDENTITY_FEATURE: {"dtype": "string"},
                        DERIVED_IDENTITY_FEATURE: {"dtype": "string"},
                        TEMPLATE_ID_FEATURE: {"dtype": "string"},
                        SPLIT_FEATURE: {"dtype": "string"},
                        "task": {"semantic": "deterministic natural-language instruction"},
                        "episode_index": {"managed_by_lerobot": True},
                        "frame_index": {"managed_by_lerobot": True},
                        "timestamp": {"managed_by_lerobot": True},
                    },
                    "episode_index": episode_index_rows,
                    "payload_tree_digest_before_sidecar": payload_tree["tree_digest"],
                    "payload_file_count_before_sidecar": payload_tree["file_count"],
                    "payload_bytes_before_sidecar": payload_tree["total_bytes"],
                    "source_bytes_modified": False,
                    "optimizer_steps": 0,
                    "passed": len(rows) == len(episode_index_rows),
                }
            )
            _write_json(staging_root / "langmani_phase2b6_split_manifest.json", sidecar)
            os.replace(staging_root, final_root)
            split_reports[split] = sidecar
        task_container = primary / "task_roots" / _task_slug(task_id)
        task_tree = file_tree_manifest(task_container)
        task_spec = cast(Mapping[str, object], cast(Mapping[str, object], spec["tasks"])[task_id])
        task_reports[task_id] = {
            "task_id": task_id,
            "skill_family": CANDIDATE_TASKS[task_id].skill_family,
            "dataset_identity": task_spec["dataset_id"],
            "container_relative_path": task_container.relative_to(primary).as_posix(),
            "episode_count": sum(
                int(cast(Mapping[str, object], report)["episode_count"])
                for report in split_reports.values()
            ),
            "frame_count": sum(
                int(cast(Mapping[str, object], report)["frame_count"])
                for report in split_reports.values()
            ),
            "tree_digest": task_tree["tree_digest"],
            "file_count": task_tree["file_count"],
            "total_bytes": task_tree["total_bytes"],
            "splits": split_reports,
            "passed": all(
                cast(Mapping[str, object], report)["passed"] is True
                for report in split_reports.values()
            ),
        }
    report = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-task-specific-roots-v0",
            "created_at_utc": _timestamp(),
            "lerobot_environment": environment,
            "task_roots": task_reports,
            "episode_count": sum(
                int(cast(Mapping[str, object], report)["episode_count"])
                for report in task_reports.values()
            ),
            "frame_count": sum(
                int(cast(Mapping[str, object], report)["frame_count"])
                for report in task_reports.values()
            ),
            "physical_media_partitioning": (
                "one LeRobot 0.6 subroot per task and primary split; media files never "
                "cross primary split boundaries"
            ),
            "student_policy_training_started": False,
            "optimizer_steps": 0,
            "passed": (
                all(
                    cast(Mapping[str, object], report)["passed"] is True
                    for report in task_reports.values()
                )
                and sum(
                    int(cast(Mapping[str, object], report)["episode_count"])
                    for report in task_reports.values()
                )
                == EXPECTED_EPISODES
                and sum(
                    int(cast(Mapping[str, object], report)["frame_count"])
                    for report in task_reports.values()
                )
                == EXPECTED_TRANSITIONS
            ),
        }
    )
    _write_json(evidence_root / "lerobot_environment_manifest.json", environment)
    _write_json(evidence_root / "task_specific_root_manifests.json", report)
    _write_json(primary / "metadata" / "task_specific_root_manifests.json", report)
    if report["passed"] is not True:
        raise Phase2B6LeRobotError("task-specific LeRobot conversion hard stop")
    return report


def convert_visual_shift_roots(
    *,
    production_root: Path,
    evidence_root: Path,
    spec_path: Path,
) -> dict[str, object]:
    """Build evaluation-only shifted RGB roots for the disjoint visual split."""

    spec = load_production_spec(spec_path)
    visual = cast(Mapping[str, object], spec["visual_shift"])
    assignments = _load_assignments(production_root)
    replay_rows = _load_replay_rows(production_root)
    replay_by_identity = {str(row["derived_episode_identity"]): row for row in replay_rows}
    primary = production_root / "primary"
    task_reports: dict[str, object] = {}
    total_changed_values = 0
    for task_id in TASK_IDS:
        rows = sorted(
            [
                row
                for row in assignments
                if row["task_id"] == task_id and row["primary_split"] == "test_visual_shift"
            ],
            key=lambda row: int(row["source_episode_id"]),
        )
        final_root = primary / "visual_shift_roots" / _task_slug(task_id)
        staging_root = final_root.parent / f".{_task_slug(task_id)}.partial"
        if final_root.exists():
            sidecar = _read_json(final_root / "langmani_phase2b6_visual_shift_manifest.json")
            if sidecar.get("passed") is not True:
                raise Phase2B6LeRobotError("existing visual-shift root failed")
            task_reports[task_id] = sidecar
            total_changed_values += int(sidecar["changed_channel_values"])
            continue
        if staging_root.exists():
            raise Phase2B6LeRobotError(
                f"preserved partial visual conversion requires review: {staging_root}"
            )
        staging_root.parent.mkdir(parents=True, exist_ok=True)
        repo_id = _visual_repo_id(task_id)
        dataset = _writer(repo_id=repo_id, root=staging_root, visual_shift=True)
        episode_index_rows: list[dict[str, object]] = []
        global_frame = 0
        changed_values = 0
        for local_episode_index, assignment in enumerate(rows):
            primary_identity = str(assignment["derived_episode_identity"])
            replay = replay_by_identity[primary_identity]
            path = production_root / str(replay["raw_relative_path"])
            rgb, state, action, _ = _validate_npz(path=path, replay=replay)
            shifted = apply_visual_shift(
                rgb,
                exposure_multiplier=float(visual["exposure_multiplier"]),
                rgb_channel_multipliers=cast(Sequence[float], visual["rgb_channel_multipliers"]),
            )
            changed_values += int(np.count_nonzero(shifted != rgb))
            shifted_identity = (
                "sha256:"
                + hashlib.sha256(
                    json.dumps(
                        {
                            "primary_derived_episode_identity": primary_identity,
                            "visual_shift_identity": visual["identity"],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
            )
            for frame_index in range(len(action)):
                dataset.add_frame(
                    {
                        IMAGE_FEATURE: shifted[frame_index],
                        STATE_FEATURE: state[frame_index],
                        ACTION_FEATURE: action[frame_index],
                        TASK_ID_FEATURE: task_id,
                        SKILL_FEATURE: assignment["skill_family"],
                        SOURCE_IDENTITY_FEATURE: assignment["source_trajectory_identity"],
                        DERIVED_IDENTITY_FEATURE: shifted_identity,
                        TEMPLATE_ID_FEATURE: assignment["instruction_template_id"],
                        SPLIT_FEATURE: "test_visual_shift",
                        PRIMARY_DERIVED_IDENTITY_FEATURE: primary_identity,
                        VISUAL_SHIFT_ID_FEATURE: visual["identity"],
                        "task": assignment["instruction"],
                    }
                )
            dataset.save_episode(parallel_encoding=False)
            episode_index_rows.append(
                {
                    "local_episode_index": local_episode_index,
                    "global_frame_start": global_frame,
                    "frame_count": len(action),
                    "source_episode_id": assignment["source_episode_id"],
                    "source_trajectory_identity": assignment["source_trajectory_identity"],
                    "primary_derived_episode_identity": primary_identity,
                    "shifted_derived_episode_identity": shifted_identity,
                    "action_sha256": assignment["action_sha256"],
                    "instruction_template_id": assignment["instruction_template_id"],
                    "instruction": assignment["instruction"],
                }
            )
            global_frame += len(action)
        dataset.finalize()
        payload_tree = file_tree_manifest(staging_root)
        sidecar = fingerprinted(
            {
                "schema_version": "langmani-v2-phase2b6-visual-shift-task-root-v0",
                "task_id": task_id,
                "repo_id": repo_id,
                "visual_shift_identity": visual["identity"],
                "appearance_contract": visual,
                "primary_split": "test_visual_shift",
                "evaluation_only": True,
                "episode_count": len(rows),
                "frame_count": global_frame,
                "changed_channel_values": changed_values,
                "physics_changed": False,
                "actions_changed": False,
                "states_changed": False,
                "timestamps_changed": False,
                "episode_index": episode_index_rows,
                "payload_tree_digest_before_sidecar": payload_tree["tree_digest"],
                "payload_file_count_before_sidecar": payload_tree["file_count"],
                "payload_bytes_before_sidecar": payload_tree["total_bytes"],
                "passed": len(rows) == 50 and changed_values > 0,
            }
        )
        _write_json(staging_root / "langmani_phase2b6_visual_shift_manifest.json", sidecar)
        os.replace(staging_root, final_root)
        task_reports[task_id] = sidecar
        total_changed_values += changed_values
    report = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-visual-shift-manifest-v0",
            "created_at_utc": _timestamp(),
            "visual_shift_identity": visual["identity"],
            "appearance_contract": visual,
            "task_roots": task_reports,
            "episode_count": sum(
                int(cast(Mapping[str, object], report)["episode_count"])
                for report in task_reports.values()
            ),
            "frame_count": sum(
                int(cast(Mapping[str, object], report)["frame_count"])
                for report in task_reports.values()
            ),
            "changed_channel_values": total_changed_values,
            "evaluation_only": True,
            "physics_changed": False,
            "actions_changed": False,
            "training_episode_count": 0,
            "passed": (
                len(task_reports) == 3
                and all(
                    cast(Mapping[str, object], task_report)["passed"] is True
                    for task_report in task_reports.values()
                )
            ),
        }
    )
    _write_json(evidence_root / "visual_shift_manifest.json", report)
    _write_json(primary / "metadata" / "visual_shift_manifest.json", report)
    if report["passed"] is not True:
        raise Phase2B6LeRobotError("visual-shift production hard stop")
    return report


def build_unified_package(
    *,
    production_root: Path,
    evidence_root: Path,
    spec_path: Path,
) -> dict[str, object]:
    """Create a content-bound multi-root index without rewriting task roots."""

    spec = load_production_spec(spec_path)
    primary = production_root / "primary"
    task_roots = _read_json(evidence_root / "task_specific_root_manifests.json")
    visual = _read_json(evidence_root / "visual_shift_manifest.json")
    metadata = primary / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    copied = {
        "primary_split_manifest.json": production_root / "work" / "primary_split_manifest.json",
        "language_template_manifest.json": production_root
        / "work"
        / "language_template_manifest.json",
        "cross_skill_fold_manifests.json": production_root
        / "work"
        / "cross_skill_fold_manifests.json",
        "padding_audit.json": evidence_root / "padding_audit.json",
        "task_balance_manifest.json": evidence_root / "task_balance_manifest.json",
        "dataset_statistics.json": evidence_root / "dataset_statistics.json",
        "normalization_manifest.json": evidence_root / "normalization_manifest.json",
        "production_run_manifest.json": evidence_root / "production_run_manifest.json",
        "source_integrity_audit.json": evidence_root / "source_integrity_audit.json",
        "source_schema_result.json": evidence_root / "source_schema_result.json",
        "observation_action_contract.json": evidence_root / "observation_action_contract.json",
    }
    metadata_hashes: dict[str, str] = {}
    for name, source in copied.items():
        destination = metadata / name
        if destination.exists():
            if sha256_file(destination) != sha256_file(source):
                raise Phase2B6LeRobotError(f"existing package metadata differs: {destination}")
        else:
            shutil.copyfile(source, destination)
        metadata_hashes[name] = "sha256:" + sha256_file(destination)
    source_lineage = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-source-lineage-v0",
            "source_authorization": spec["source_authorization"],
            "tasks": {
                task_id: {
                    "source_zip_relative_path": cast(
                        Mapping[str, object], cast(Mapping[str, object], spec["tasks"])[task_id]
                    )["source_zip_relative_path"],
                    "source_zip_sha256": cast(
                        Mapping[str, object], cast(Mapping[str, object], spec["tasks"])[task_id]
                    )["source_zip_sha256"],
                    "source_zip_size_bytes": cast(
                        Mapping[str, object], cast(Mapping[str, object], spec["tasks"])[task_id]
                    )["source_zip_size_bytes"],
                    "source_zip_bytes_in_archive": False,
                    "source_zip_reference_in_archive": True,
                }
                for task_id in TASK_IDS
            },
            "source_bytes_modified": False,
        }
    )
    _write_json(metadata / "source_lineage.json", source_lineage)
    task_index: list[dict[str, object]] = []
    for task_id, raw in cast(Mapping[str, object], task_roots["task_roots"]).items():
        task_report = cast(Mapping[str, object], raw)
        split_index: list[dict[str, object]] = []
        for split, split_raw in cast(Mapping[str, object], task_report["splits"]).items():
            split_report = cast(Mapping[str, object], split_raw)
            split_index.append(
                {
                    "primary_split": split,
                    "repo_id": split_report["repo_id"],
                    "relative_root": (f"task_roots/{_task_slug(task_id)}/splits/{split}"),
                    "episode_count": split_report["episode_count"],
                    "frame_count": split_report["frame_count"],
                    "manifest_fingerprint": split_report["fingerprint"],
                }
            )
        task_index.append(
            {
                "task_id": task_id,
                "skill_family": task_report["skill_family"],
                "dataset_identity": task_report["dataset_identity"],
                "container_relative_root": task_report["container_relative_path"],
                "tree_digest": task_report["tree_digest"],
                "splits": split_index,
            }
        )
    visual_index = [
        {
            "task_id": task_id,
            "repo_id": cast(Mapping[str, object], raw)["repo_id"],
            "relative_root": f"visual_shift_roots/{_task_slug(task_id)}",
            "episode_count": cast(Mapping[str, object], raw)["episode_count"],
            "frame_count": cast(Mapping[str, object], raw)["frame_count"],
            "manifest_fingerprint": cast(Mapping[str, object], raw)["fingerprint"],
        }
        for task_id, raw in cast(Mapping[str, object], visual["task_roots"]).items()
    ]
    unified = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-unified-multi-root-v0",
            "package_identity": DATASET_PACKAGE_ID,
            "materialization": "immutable_task_and_split_multi_root_index",
            "primary_root": primary.as_posix(),
            "primary_root_committed": False,
            "tasks": task_index,
            "visual_shift_roots": visual_index,
            "metadata": metadata_hashes,
            "source_lineage_sha256": "sha256:" + sha256_file(metadata / "source_lineage.json"),
            "load_filters": {
                "all_tasks": "iterate every task/split root in tasks",
                "one_task": "select tasks[].task_id",
                "one_split": "select tasks[].splits[].primary_split",
                "one_skill": "select tasks[].skill_family",
                "one_cross_skill_fold": (
                    "resolve immutable episode identities from "
                    "metadata/cross_skill_fold_manifests.json"
                ),
            },
            "source_task_roots_modified": False,
            "destructive_concatenation": False,
            "episode_count": EXPECTED_EPISODES,
            "frame_count": EXPECTED_TRANSITIONS,
            "visual_shift_episode_count": 150,
            "student_policy_training_started": False,
            "optimizer_steps": 0,
            "passed": (
                task_roots.get("passed") is True
                and visual.get("passed") is True
                and len(task_index) == 3
            ),
        }
    )
    _write_json(metadata / "unified_multi_root_manifest.json", unified)
    _write_json(evidence_root / "unified_multi_root_manifest.json", unified)
    if unified["passed"] is not True:
        raise Phase2B6LeRobotError("unified package construction hard stop")
    return unified


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


def _scalar(value: object) -> object:
    array = _to_numpy(value)
    if array.size != 1:
        raise Phase2B6LeRobotError(f"expected scalar, got {array.shape}")
    return array.reshape(-1)[0].item()


def _decode_all_videos(root: Path) -> dict[str, object]:
    try:
        import av
    except ImportError as error:
        raise Phase2B6LeRobotError("full video readback requires PyAV") from error
    videos = sorted(root.glob(f"videos/{IMAGE_FEATURE}/**/*.mp4"))
    if not videos:
        raise Phase2B6LeRobotError(f"LeRobot root has no video files: {root}")
    decoded_frames = 0
    failures: list[dict[str, object]] = []
    shapes: Counter[str] = Counter()
    for path in videos:
        try:
            with av.open(str(path)) as container:
                for frame in container.decode(video=0):
                    image = frame.to_ndarray(format="rgb24")
                    shapes[str(tuple(image.shape))] += 1
                    if image.shape != IMAGE_SHAPE or image.dtype != np.dtype(np.uint8):
                        raise Phase2B6LeRobotError("decoded video frame schema changed")
                    decoded_frames += 1
        except Exception as error:
            failures.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
    return {
        "video_file_count": len(videos),
        "decoded_frame_count": decoded_frames,
        "decoded_shapes": dict(shapes),
        "decoding_failures": failures,
        "passed": not failures,
    }


def _readback_one_root(
    *,
    root: Path,
    sidecar: Mapping[str, object],
    visual_shift: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    try:
        from lerobot.datasets import LeRobotDataset  # type: ignore[import-untyped]
    except (ImportError, OSError, RuntimeError) as error:
        raise Phase2B6LeRobotError("LeRobot 0.6 readback API is unavailable") from error
    dataset = LeRobotDataset(
        repo_id=str(sidecar["repo_id"]),
        root=root,
        download_videos=False,
        video_backend="pyav",
        return_uint8=True,
    )
    episode_rows = cast(list[dict[str, object]], sidecar["episode_index"])
    hf = dataset.hf_dataset.with_format("numpy")
    failures = {
        "state": 0,
        "action": 0,
        "metadata": 0,
        "index_or_timestamp": 0,
        "api_image": 0,
        "api_instruction": 0,
    }
    equality_rows: list[dict[str, object]] = []
    api_episode_count = 0
    for episode in episode_rows:
        start = int(episode["global_frame_start"])
        length = int(episode["frame_count"])
        stop = start + length
        rows = hf[start:stop]
        state = np.asarray(rows[STATE_FEATURE], dtype=np.float32)
        action = np.asarray(rows[ACTION_FEATURE], dtype=np.float32)
        episode_indices = np.asarray(rows["episode_index"]).reshape(-1)
        frame_indices = np.asarray(rows["frame_index"]).reshape(-1)
        timestamps = np.asarray(rows["timestamp"], dtype=np.float64).reshape(-1)
        task_ids = list(rows[TASK_ID_FEATURE])
        skills = list(rows[SKILL_FEATURE])
        source_ids = list(rows[SOURCE_IDENTITY_FEATURE])
        derived_ids = list(rows[DERIVED_IDENTITY_FEATURE])
        template_ids = list(rows[TEMPLATE_ID_FEATURE])
        splits = list(rows[SPLIT_FEATURE])
        if state.shape != (length, 9) or state.dtype != np.dtype(np.float32):
            failures["state"] += 1
        if action.shape != (length, 8) or action.dtype != np.dtype(np.float32):
            failures["action"] += 1
        if not np.all(np.isfinite(state)):
            failures["state"] += 1
        if not np.all(np.isfinite(action)):
            failures["action"] += 1
        expected_episode_index = int(episode["local_episode_index"])
        if (
            not np.all(episode_indices == expected_episode_index)
            or not np.array_equal(frame_indices, np.arange(length))
            or not np.allclose(
                timestamps,
                np.arange(length, dtype=np.float64) / CONTROL_FREQUENCY_HZ,
                rtol=0.0,
                atol=1e-6,
            )
        ):
            failures["index_or_timestamp"] += 1
        expected_task = str(sidecar["task_id"])
        expected_skill = CANDIDATE_TASKS[expected_task].skill_family
        expected_source = str(episode["source_trajectory_identity"])
        expected_template = str(episode["instruction_template_id"])
        expected_split = str(sidecar.get("primary_split", "test_visual_shift"))
        expected_derived = str(
            episode[
                "shifted_derived_episode_identity" if visual_shift else "derived_episode_identity"
            ]
        )
        metadata_ok = (
            set(map(str, task_ids)) == {expected_task}
            and set(map(str, skills)) == {expected_skill}
            and set(map(str, source_ids)) == {expected_source}
            and set(map(str, derived_ids)) == {expected_derived}
            and set(map(str, template_ids)) == {expected_template}
            and set(map(str, splits)) == {expected_split}
        )
        if not metadata_ok:
            failures["metadata"] += 1
        api_row = dataset[start]
        api_image = _to_numpy(api_row[IMAGE_FEATURE])
        if api_image.shape not in {(3, 256, 256), IMAGE_SHAPE}:
            failures["api_image"] += 1
        if api_row.get("task") != episode["instruction"]:
            failures["api_instruction"] += 1
        api_episode_count += 1
        equality_rows.append(
            {
                "task_id": expected_task,
                "primary_split": expected_split,
                "local_episode_index": expected_episode_index,
                "source_episode_id": episode["source_episode_id"],
                "source_trajectory_identity": expected_source,
                "derived_episode_identity": expected_derived,
                "frame_count": length,
                "derived_action_sha256": _array_sha256(action, dtype=np.dtype(np.float32)),
                "expected_action_sha256": episode["action_sha256"],
                "action_exact": (
                    _array_sha256(action, dtype=np.dtype(np.float32)) == episode["action_sha256"]
                ),
                "timestamp_sequence_valid": np.allclose(
                    timestamps,
                    np.arange(length, dtype=np.float64) / CONTROL_FREQUENCY_HZ,
                    rtol=0.0,
                    atol=1e-6,
                ),
                "metadata_exact": metadata_ok,
            }
        )
    videos = _decode_all_videos(root)
    expected_episodes = int(sidecar["episode_count"])
    expected_frames = int(sidecar["frame_count"])
    report = {
        "repo_id": sidecar["repo_id"],
        "root": root.as_posix(),
        "episode_count": int(dataset.num_episodes),
        "frame_count": int(dataset.num_frames),
        "expected_episode_count": expected_episodes,
        "expected_frame_count": expected_frames,
        "api_episode_readback_count": api_episode_count,
        "frame_level_structural_validation_count": sum(
            int(row["frame_count"]) for row in equality_rows
        ),
        "failures": failures,
        "video": videos,
        "passed": (
            int(dataset.num_episodes) == expected_episodes
            and int(dataset.num_frames) == expected_frames
            and api_episode_count == expected_episodes
            and sum(int(row["frame_count"]) for row in equality_rows) == expected_frames
            and all(value == 0 for value in failures.values())
            and videos["passed"] is True
            and videos["decoded_frame_count"] == expected_frames
            and all(row["action_exact"] is True for row in equality_rows)
        ),
    }
    return report, equality_rows


def run_full_readback(
    *,
    production_root: Path,
    evidence_root: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Read every episode via LeRobot and decode every video frame."""

    primary = production_root / "primary"
    task_roots = _read_json(evidence_root / "task_specific_root_manifests.json")
    root_reports: list[dict[str, object]] = []
    visual_root_reports: list[dict[str, object]] = []
    equality_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for task_id, raw_task in cast(Mapping[str, object], task_roots["task_roots"]).items():
        task = cast(Mapping[str, object], raw_task)
        for split, raw_sidecar in cast(Mapping[str, object], task["splits"]).items():
            sidecar = cast(Mapping[str, object], raw_sidecar)
            root = primary / "task_roots" / _task_slug(task_id) / "splits" / split
            report, rows = _readback_one_root(root=root, sidecar=sidecar, visual_shift=False)
            root_reports.append(report)
            equality_rows.extend(rows)
            print(
                json.dumps(
                    {
                        "phase": "full_readback",
                        "completed_root_count": len(root_reports),
                        "total_root_count": 15,
                        "elapsed_seconds": time.perf_counter() - started,
                        "task_id": task_id,
                        "split": split,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    visual = _read_json(evidence_root / "visual_shift_manifest.json")
    for task_id, raw_sidecar in cast(Mapping[str, object], visual["task_roots"]).items():
        sidecar = cast(Mapping[str, object], raw_sidecar)
        root = primary / "visual_shift_roots" / _task_slug(task_id)
        report, _ = _readback_one_root(
            root=root,
            sidecar=sidecar,
            visual_shift=True,
        )
        visual_root_reports.append(report)
        print(
            json.dumps(
                {
                    "phase": "full_readback_visual_shift",
                    "completed_root_count": len(visual_root_reports),
                    "total_root_count": 3,
                    "elapsed_seconds": time.perf_counter() - started,
                    "task_id": task_id,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    total_episodes = sum(int(report["episode_count"]) for report in root_reports)
    total_frames = sum(int(report["frame_count"]) for report in root_reports)
    visual_episodes = sum(int(report["episode_count"]) for report in visual_root_reports)
    visual_frames = sum(int(report["frame_count"]) for report in visual_root_reports)
    decoded_primary_frames = sum(
        int(cast(Mapping[str, object], report["video"])["decoded_frame_count"])
        for report in root_reports
    )
    decoded_visual_frames = sum(
        int(cast(Mapping[str, object], report["video"])["decoded_frame_count"])
        for report in visual_root_reports
    )
    all_reports = [*root_reports, *visual_root_reports]
    readback = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-full-lerobot-readback-v0",
            "created_at_utc": _timestamp(),
            "lerobot_version": _distribution_version("lerobot"),
            "root_count": len(all_reports),
            "primary_root_count": len(root_reports),
            "visual_shift_root_count": len(visual_root_reports),
            "episode_count": total_episodes,
            "frame_count": total_frames,
            "visual_shift_episode_count": visual_episodes,
            "visual_shift_frame_count": visual_frames,
            "all_generated_episode_readback_count": total_episodes + visual_episodes,
            "all_generated_frame_readback_count": total_frames + visual_frames,
            "api_episode_readback_count": sum(
                int(report["api_episode_readback_count"]) for report in all_reports
            ),
            "frame_level_structural_validation_count": sum(
                int(report["frame_level_structural_validation_count"]) for report in all_reports
            ),
            "decoded_frame_count": decoded_primary_frames + decoded_visual_frames,
            "decoded_primary_frame_count": decoded_primary_frames,
            "decoded_visual_shift_frame_count": decoded_visual_frames,
            "decoding_failure_count": sum(
                len(
                    cast(
                        list[object],
                        cast(Mapping[str, object], report["video"])["decoding_failures"],
                    )
                )
                for report in all_reports
            ),
            "state_failure_count": sum(
                int(cast(Mapping[str, object], report["failures"])["state"])
                for report in all_reports
            ),
            "action_failure_count": sum(
                int(cast(Mapping[str, object], report["failures"])["action"])
                for report in all_reports
            ),
            "metadata_failure_count": sum(
                int(cast(Mapping[str, object], report["failures"])["metadata"])
                for report in all_reports
            ),
            "primary_roots": root_reports,
            "visual_shift_roots": visual_root_reports,
            "student_policy_training_started": False,
            "optimizer_steps": 0,
            "passed": (
                len(root_reports) == 15
                and len(visual_root_reports) == 3
                and total_episodes == EXPECTED_EPISODES
                and total_frames == EXPECTED_TRANSITIONS
                and visual_episodes == 150
                and decoded_primary_frames == EXPECTED_TRANSITIONS
                and decoded_visual_frames == visual_frames
                and all(report["passed"] is True for report in all_reports)
            ),
        }
    )
    equality = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-derived-equality-intermediate-v0",
            "episode_count": len(equality_rows),
            "action_exact_count": sum(row["action_exact"] is True for row in equality_rows),
            "timestamp_sequence_valid_count": sum(
                row["timestamp_sequence_valid"] is True for row in equality_rows
            ),
            "metadata_exact_count": sum(row["metadata_exact"] is True for row in equality_rows),
            "episode_rows_digest": _canonical_rows_digest(equality_rows),
            "episode_rows": equality_rows,
            "passed": all(
                row["action_exact"] is True
                and row["timestamp_sequence_valid"] is True
                and row["metadata_exact"] is True
                for row in equality_rows
            ),
        }
    )
    _write_json(evidence_root / "full_readback_result.json", readback)
    _write_json(
        production_root / "primary" / "metadata" / "full_readback_result.json",
        readback,
    )
    _write_json(production_root / "work" / "derived_equality_intermediate.json", equality)
    if readback["passed"] is not True or equality["passed"] is not True:
        raise Phase2B6LeRobotError("full LeRobot readback hard stop")
    return readback, equality


def _canonical_rows_digest(rows: Sequence[Mapping[str, object]]) -> str:
    encoded = json.dumps(
        {"rows": list(rows)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def run_source_to_derived_verification(
    *,
    source_root: Path,
    production_root: Path,
    evidence_root: Path,
) -> dict[str, object]:
    """Independently compare every source HDF5 action to every derived episode."""

    assignments = _load_assignments(production_root)
    equality = _read_json(production_root / "work" / "derived_equality_intermediate.json")
    derived_rows = cast(list[dict[str, object]], equality["episode_rows"])
    derived_by_source = {
        (str(row["task_id"]), int(row["source_episode_id"])): row for row in derived_rows
    }
    replay_rows = _load_replay_rows(production_root)
    replay_by_source = {
        (str(row["task_id"]), int(row["source_episode_id"])): row for row in replay_rows
    }
    split_by_source = {
        (str(row["task_id"]), int(row["source_episode_id"])): row for row in assignments
    }
    failures: list[dict[str, object]] = []
    checked = 0
    action_values_checked = 0
    for task_id in TASK_IDS:
        h5_path = source_root / "expanded" / task_id / "motionplanning" / "trajectory.h5"
        with h5py.File(h5_path, "r") as trajectories:
            for source_episode_id in range(1000):
                source_actions = np.asarray(
                    trajectories[f"traj_{source_episode_id}"]["actions"],
                    dtype=np.float32,
                )
                source_hash = _array_sha256(source_actions, dtype=np.dtype(np.float32))
                derived = derived_by_source[(task_id, source_episode_id)]
                replay = replay_by_source[(task_id, source_episode_id)]
                split = split_by_source[(task_id, source_episode_id)]
                checks = {
                    "source_action_count": len(source_actions) == int(derived["frame_count"]),
                    "derived_action_count": int(derived["frame_count"])
                    == int(replay["source_action_count"]),
                    "action_hash": source_hash == derived["derived_action_sha256"],
                    "action_values_exact": derived["action_exact"] is True,
                    "source_success": replay["source_success"] is True,
                    "replay_success": replay["replay_success"] is True,
                    "task_id": split["task_id"] == task_id,
                    "reset_identity": replay["reset_identity"] == split["reset_identity"],
                    "language_identity": (
                        replay["instruction_template_id"] == split["instruction_template_id"]
                    ),
                    "frame_count": replay["policy_frame_count"] == len(source_actions),
                    "timestamp_sequence": derived["timestamp_sequence_valid"] is True,
                }
                if not all(checks.values()):
                    failures.append(
                        {
                            "task_id": task_id,
                            "source_episode_id": source_episode_id,
                            "checks": checks,
                        }
                    )
                checked += 1
                action_values_checked += int(source_actions.size)
    report = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-source-to-derived-verifier-v0",
            "created_at_utc": _timestamp(),
            "independent_from_production_writer": True,
            "source_episode_count": checked,
            "derived_episode_count": len(derived_rows),
            "source_action_values_checked": action_values_checked,
            "exact_action_equality_count": checked - len(failures),
            "approximate_action_tolerance_used": False,
            "failure_count": len(failures),
            "failures": failures,
            "student_policy_training_started": False,
            "optimizer_steps": 0,
            "passed": checked == EXPECTED_EPISODES and not failures,
        }
    )
    _write_json(evidence_root / "source_to_derived_verifier_result.json", report)
    _write_json(
        production_root / "primary" / "metadata" / "source_to_derived_verifier_result.json",
        report,
    )
    if report["passed"] is not True:
        raise Phase2B6LeRobotError("source-to-derived equality hard stop")
    return report


def run_leakage_audit(
    *,
    production_root: Path,
    evidence_root: Path,
) -> dict[str, object]:
    """Audit primary identities, physical media, visual lineage, and fold references."""

    assignments = _load_assignments(production_root)
    primary_summary = audit_primary_split_leakage(assignments)
    primary = production_root / "primary"
    media_by_split: dict[str, set[str]] = {}
    media_hash_by_split: dict[str, set[str]] = {}
    for split in PRIMARY_SPLITS:
        paths: set[str] = set()
        hashes: set[str] = set()
        for task_id in TASK_IDS:
            root = primary / "task_roots" / _task_slug(task_id) / "splits" / split
            for path in root.glob(f"videos/{IMAGE_FEATURE}/**/*.mp4"):
                paths.add(path.relative_to(primary).as_posix())
                hashes.add("sha256:" + sha256_file(path))
        media_by_split[split] = paths
        media_hash_by_split[split] = hashes
    media_matrix: dict[str, dict[str, int]] = {}
    for left_index, left in enumerate(PRIMARY_SPLITS):
        for right in PRIMARY_SPLITS[left_index + 1 :]:
            media_matrix[f"{left}__{right}"] = {
                "relative_path_overlap": len(media_by_split[left] & media_by_split[right]),
                "file_hash_overlap": len(media_hash_by_split[left] & media_hash_by_split[right]),
            }
    visual = _read_json(evidence_root / "visual_shift_manifest.json")
    visual_source_ids = {
        str(row["source_trajectory_identity"])
        for row in assignments
        if row["primary_split"] == "test_visual_shift"
    }
    visual_manifest_source_ids = {
        str(episode["source_trajectory_identity"])
        for raw in cast(Mapping[str, object], visual["task_roots"]).values()
        for episode in cast(
            list[dict[str, object]], cast(Mapping[str, object], raw)["episode_index"]
        )
    }
    train_template_ids = {
        str(row["instruction_template_id"])
        for row in assignments
        if row["primary_split"] == "train"
    }
    heldout_template_ids = {
        str(row["instruction_template_id"])
        for row in assignments
        if row["primary_split"] == "test_unseen_task_language"
    }
    folds = _read_json(production_root / "work" / "cross_skill_fold_manifests.json")
    media_overlap_count = sum(value for pair in media_matrix.values() for value in pair.values())
    report = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-leakage-audit-v0",
            "created_at_utc": _timestamp(),
            "primary_identity_audit": primary_summary,
            "primary_media_overlap_matrix": media_matrix,
            "primary_media_overlap_count": media_overlap_count,
            "train_heldout_language_template_overlap_count": len(
                train_template_ids & heldout_template_ids
            ),
            "visual_shift_lineage": {
                "expected_source_episode_count": len(visual_source_ids),
                "manifest_source_episode_count": len(visual_manifest_source_ids),
                "exact_match": visual_source_ids == visual_manifest_source_ids,
                "training_source_overlap_count": len(
                    visual_manifest_source_ids
                    & {
                        str(row["source_trajectory_identity"])
                        for row in assignments
                        if row["primary_split"] == "train"
                    }
                ),
            },
            "cross_skill_folds": {
                "materialization": folds["materialization"],
                "primary_leakage": False,
                "alternate_view_reference_only": True,
                "fold_count": len(cast(Mapping[str, object], folds["folds"])),
            },
            "primary_split_overlap_count": (
                int(primary_summary["primary_split_overlap_count"])
                + media_overlap_count
                + len(train_template_ids & heldout_template_ids)
            ),
            "passed": (
                primary_summary["passed"] is True
                and media_overlap_count == 0
                and not (train_template_ids & heldout_template_ids)
                and visual_source_ids == visual_manifest_source_ids
            ),
        }
    )
    _write_json(evidence_root / "leakage_audit.json", report)
    _write_json(production_root / "primary" / "metadata" / "leakage_audit.json", report)
    if report["passed"] is not True:
        raise Phase2B6LeRobotError("primary split leakage hard stop")
    return report


def create_archive(
    *,
    production_root: Path,
    archive_root: Path,
    evidence_root: Path,
    spec_path: Path,
) -> dict[str, object]:
    """Create a deterministic tree-addressed archive outside the primary root."""

    spec = load_production_spec(spec_path)
    primary = (production_root / "primary").resolve()
    archive_directory = archive_root.resolve()
    try:
        archive_directory.relative_to(primary)
    except ValueError:
        pass
    else:
        raise Phase2B6LeRobotError("archive location must be outside the primary dataset")
    tree = file_tree_manifest(primary)
    tree_digest = str(tree["tree_digest"]).removeprefix("sha256:")
    archive_path = archive_directory / f"{DATASET_PACKAGE_ID}-{tree_digest}.tar.gz"
    if archive_path.exists():
        raise Phase2B6LeRobotError(f"immutable archive already exists: {archive_path}")
    created = create_deterministic_tar_gz(
        source_root=primary,
        archive_path=archive_path,
        prefix=str(cast(Mapping[str, object], spec["archive"])["prefix"]),
    )
    report = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-archive-manifest-v0",
            "created_at_utc": _timestamp(),
            "package_identity": DATASET_PACKAGE_ID,
            "primary_dataset_location": primary.as_posix(),
            "primary_tree_digest": tree["tree_digest"],
            "primary_file_count": tree["file_count"],
            "primary_total_bytes": tree["total_bytes"],
            "content_addressed_archive_location": archive_path.as_posix(),
            "archive_size_bytes": created["size_bytes"],
            "archive_sha256": created["sha256"],
            "archive_prefix": created["prefix"],
            "archive_content_address_kind": "primary_tree_sha256",
            "source_zip_references_included": True,
            "source_zip_bytes_included": False,
            "primary_and_archive_distinct": primary.parent != archive_directory,
            "symlinks_in_primary": False,
            "recovery_command": (
                "python environment/run_v2_phase2b6.py restore "
                f"--archive {archive_path.as_posix()} --destination <clean-scratch>"
            ),
            "passed": (
                created["source_tree_digest"] == tree["tree_digest"] and archive_path.is_file()
            ),
        }
    )
    _write_json(evidence_root / "archive_manifest.json", report)
    if report["passed"] is not True:
        raise Phase2B6LeRobotError("archive creation hard stop")
    return report


def restore_and_validate(
    *,
    archive_manifest_path: Path,
    restore_root: Path,
    evidence_root: Path,
) -> dict[str, object]:
    """Restore into clean physical bytes and load every task/split inventory."""

    archive = _read_json(archive_manifest_path)
    tree_digest = str(archive["primary_tree_digest"])
    destination = restore_root.resolve() / tree_digest.removeprefix("sha256:")
    restored = restore_validate_archive(
        archive_path=Path(str(archive["content_addressed_archive_location"])),
        destination=destination,
        prefix=str(archive["archive_prefix"]),
        expected_tree_digest=tree_digest,
    )
    package_root = destination / str(archive["archive_prefix"])
    task_manifest = _read_json(package_root / "metadata" / "task_specific_root_manifests.json")
    sampled: list[dict[str, object]] = []
    inventory_episode_count = 0
    inventory_frame_count = 0
    try:
        from lerobot.datasets import LeRobotDataset  # type: ignore[import-untyped]
    except (ImportError, OSError, RuntimeError) as error:
        raise Phase2B6LeRobotError("restored readback requires LeRobot 0.6") from error
    for task_id, raw_task in cast(Mapping[str, object], task_manifest["task_roots"]).items():
        task = cast(Mapping[str, object], raw_task)
        for split, raw_sidecar in cast(Mapping[str, object], task["splits"]).items():
            sidecar = cast(Mapping[str, object], raw_sidecar)
            root = package_root / "task_roots" / _task_slug(task_id) / "splits" / split
            dataset = LeRobotDataset(
                repo_id=str(sidecar["repo_id"]),
                root=root,
                download_videos=False,
                video_backend="pyav",
                return_uint8=True,
            )
            episodes = cast(list[dict[str, object]], sidecar["episode_index"])
            sample_episode_indices = sorted({0, len(episodes) // 2, len(episodes) - 1})
            for local_index in sample_episode_indices:
                episode = episodes[local_index]
                global_index = int(episode["global_frame_start"])
                row = dataset[global_index]
                image = _to_numpy(row[IMAGE_FEATURE])
                state = _to_numpy(row[STATE_FEATURE])
                action = _to_numpy(row[ACTION_FEATURE])
                sampled.append(
                    {
                        "task_id": task_id,
                        "split": split,
                        "local_episode_index": local_index,
                        "image_shape": list(image.shape),
                        "state_shape": list(state.shape),
                        "action_shape": list(action.shape),
                        "instruction": row["task"],
                        "source_identity": row[SOURCE_IDENTITY_FEATURE],
                        "passed": (
                            image.shape in {(3, 256, 256), IMAGE_SHAPE}
                            and state.shape == (9,)
                            and action.shape == (8,)
                            and row["task"] == episode["instruction"]
                            and row[SOURCE_IDENTITY_FEATURE]
                            == episode["source_trajectory_identity"]
                        ),
                    }
                )
            inventory_episode_count += int(dataset.num_episodes)
            inventory_frame_count += int(dataset.num_frames)
    report = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-restore-validation-v0",
            "created_at_utc": _timestamp(),
            "archive_sha256": archive["archive_sha256"],
            "restored_location": package_root.as_posix(),
            "restored_tree_digest": restored["restored_tree_digest"],
            "full_file_hash_tree_verified": (
                restored["restored_tree_digest"] == archive["primary_tree_digest"]
            ),
            "restored_file_count": restored["file_count"],
            "restored_total_bytes": restored["total_bytes"],
            "symlink_to_primary_used": False,
            "task_split_root_count": 15,
            "full_episode_inventory_count": inventory_episode_count,
            "full_frame_inventory_count": inventory_frame_count,
            "stratified_sample_count": len(sampled),
            "stratified_samples": sampled,
            "lerobot_api_loading": True,
            "scratch_deleted_after_evidence_finalization": False,
            "passed": (
                restored["passed"] is True
                and restored["restored_tree_digest"] == archive["primary_tree_digest"]
                and inventory_episode_count == EXPECTED_EPISODES
                and inventory_frame_count == EXPECTED_TRANSITIONS
                and len(sampled) == 45
                and all(row["passed"] is True for row in sampled)
            ),
        }
    )
    _write_json(evidence_root / "restore_validation_result.json", report)
    if report["passed"] is not True:
        raise Phase2B6LeRobotError("restore validation hard stop")
    return report


def finalize_acceptance(
    *,
    production_root: Path,
    evidence_root: Path,
    repo_root: Path,
    command_log: Sequence[str],
) -> dict[str, object]:
    """Create Result A evidence only when all immutable inputs pass."""

    names = {
        "source": "source_integrity_audit.json",
        "schema": "source_schema_result.json",
        "production": "production_run_manifest.json",
        "readback": "full_readback_result.json",
        "equality": "source_to_derived_verifier_result.json",
        "leakage": "leakage_audit.json",
        "archive": "archive_manifest.json",
        "restore": "restore_validation_result.json",
        "padding": "padding_audit.json",
        "normalization": "normalization_manifest.json",
        "observation": "observation_action_contract.json",
        "unified": "unified_multi_root_manifest.json",
        "visual": "visual_shift_manifest.json",
        "task_roots": "task_specific_root_manifests.json",
    }
    reports = {key: _read_json(evidence_root / name) for key, name in names.items()}
    gates = {
        "official_source_hashes": reports["source"]["passed"] is True,
        "source_trajectory_accounting": (
            reports["schema"]["passed"] is True
            and reports["schema"]["episode_count"] == EXPECTED_EPISODES
        ),
        "replay_outcome_agreement": (
            reports["production"]["passed"] is True
            and cast(Mapping[str, object], reports["production"]["aggregate_gate"])[
                "categorical_outcome_agreement_rate"
            ]
            == 1.0
        ),
        "action_integrity": (
            reports["equality"]["passed"] is True
            and reports["equality"]["exact_action_equality_count"] == EXPECTED_EPISODES
        ),
        "finite_values": (
            cast(Mapping[str, object], reports["production"]["aggregate_gate"])[
                "nonfinite_value_count"
            ]
            == 0
        ),
        "lerobot_readback": reports["readback"]["passed"] is True,
        "primary_split_leakage": (
            reports["leakage"]["passed"] is True
            and reports["leakage"]["primary_split_overlap_count"] == 0
        ),
        "privileged_field_exclusion": reports["observation"]["passed"] is True,
        "train_only_normalization": (
            reports["normalization"]["passed"] is True
            and reports["normalization"]["source_view"] == "primary_train_only"
        ),
        "action_padding_masks": (
            reports["padding"]["passed"] is True
            and reports["padding"]["masked_loss_contract_verified"] is True
        ),
        "archive": reports["archive"]["passed"] is True,
        "restore": reports["restore"]["passed"] is True,
    }
    package = create_accepted_dataset_package(
        gates=gates,
        package_fields={
            "production_run_fingerprint": reports["production"]["fingerprint"],
            "source_identities": {
                "official_source_revision": reports["source"]["official_source_revision"],
                "source_schema_fingerprint": reports["schema"]["fingerprint"],
            },
            "derived_roots": reports["task_roots"]["task_roots"],
            "unified_multi_root_fingerprint": reports["unified"]["fingerprint"],
            "observation_contract": reports["observation"]["observation"],
            "action_contract": reports["observation"]["action"],
            "language_contract": "language_template_manifest.json",
            "split_contract": "primary_split_manifest.json",
            "statistics": {
                "dataset_statistics": "dataset_statistics.json",
                "canonical_normalization_fingerprint": reports["normalization"]["fingerprint"],
            },
            "padding_behavior": {
                "mask_feature": reports["padding"]["mask_feature"],
                "horizons": list(reports["padding"]["audits"]),
            },
            "visual_shift_fingerprint": reports["visual"]["fingerprint"],
            "source_to_derived_fingerprint": reports["equality"]["fingerprint"],
            "readback_fingerprint": reports["readback"]["fingerprint"],
            "leakage_fingerprint": reports["leakage"]["fingerprint"],
            "archive_identity": {
                "location": reports["archive"]["content_addressed_archive_location"],
                "sha256": reports["archive"]["archive_sha256"],
                "primary_tree_digest": reports["archive"]["primary_tree_digest"],
            },
            "restoration_identity": {
                "restored_tree_digest": reports["restore"]["restored_tree_digest"],
                "fingerprint": reports["restore"]["fingerprint"],
            },
            "known_limitations": [
                "Official sources expose no broader reset-family label beyond per-episode reset identity.",
                "Visual shift is a deterministic post-render RGB transform and is evaluation-only.",
                "Cross-skill folds are metadata views; separate future models are required for unseen-skill claims.",
                "Eligibility does not authorize or start any policy training.",
            ],
        },
    )
    result = classify_result(
        source_integrity_valid=reports["source"]["passed"] is True,
        replay_and_derived_integrity_valid=all(
            reports[name]["passed"] is True
            for name in (
                "schema",
                "production",
                "readback",
                "equality",
                "leakage",
                "padding",
                "normalization",
                "observation",
                "unified",
                "visual",
                "task_roots",
            )
        ),
        accepted_skill_count=3,
        archive_created=reports["archive"]["passed"] is True,
        restore_valid=reports["restore"]["passed"] is True,
    )
    if result.value != "RESULT_A":
        raise Phase2B6LeRobotError(f"final result is not accepted: {result.value}")
    authorization = fingerprinted(authorization_state(accepted=True))
    classification = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-result-classification-v0",
            "result": result.value,
            "label": "Full multi-skill dataset accepted",
            "accepted_package_fingerprint": package["fingerprint"],
            "gates": gates,
            "accepted_multiskill_dataset_validated": True,
            "student_policy_training_started": False,
            "optimizer_steps": 0,
            "passed": True,
        }
    )
    remote = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-remote-execution-audit-v0",
            "created_at_utc": _timestamp(),
            "hostname": platform.node(),
            "producer_commit": _git(repo_root, "rev-parse", "HEAD"),
            "producer_branch": _git(repo_root, "branch", "--show-current"),
            "producer_worktree_clean": not _git(repo_root, "status", "--porcelain=v1"),
            "lerobot_environment": lerobot_environment_manifest(),
            "commands": list(command_log),
            "official_action_replay_only": True,
            "custom_expert_started": False,
            "model_loaded": False,
            "optimizer_created": False,
            "backward_passes": 0,
            "optimizer_steps": 0,
            "student_policy_training_started": False,
            "passed": True,
        }
    )
    _write_json(evidence_root / "accepted_multiskill_dataset_package.json", package)
    _write_json(evidence_root / "result_classification.json", classification)
    _write_json(evidence_root / "authorization_state.json", authorization)
    _write_json(evidence_root / "remote_execution_audit.json", remote)
    return classification


def write_artifact_manifest(*, evidence_root: Path) -> dict[str, object]:
    """Hash every compact evidence file except the manifest itself."""

    manifest_path = evidence_root / "artifact_manifest.json"
    if manifest_path.exists():
        raise Phase2B6LeRobotError("artifact manifest is immutable and already exists")
    entries: list[dict[str, object]] = []
    for path in sorted(evidence_root.glob("*.json")):
        if path.name == manifest_path.name:
            continue
        entries.append(
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": "sha256:" + sha256_file(path),
            }
        )
    manifest = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-artifact-manifest-v0",
            "file_count": len(entries),
            "files": entries,
            "large_dataset_bytes_committed": False,
            "passed": len(entries) >= 25,
        }
    )
    _write_json(manifest_path, manifest)
    return manifest


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


__all__ = [
    "Phase2B6LeRobotError",
    "build_unified_package",
    "convert_task_split_roots",
    "convert_visual_shift_roots",
    "create_archive",
    "finalize_acceptance",
    "lerobot_environment_manifest",
    "restore_and_validate",
    "run_full_readback",
    "run_leakage_audit",
    "run_source_to_derived_verification",
    "write_artifact_manifest",
]
