"""Identity, quality, archival, and acceptance contracts for Phase 2B.2."""

from __future__ import annotations

import gzip
import json
import os
import shutil
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Self, cast

import numpy as np

from langmani.datasets.lerobot_types import (
    ACTION_FEATURE_KEY,
    IMAGE_FEATURE_KEY,
    STATE_FEATURE_KEY,
)
from langmani.v2.push_archive import sha256_file, validate_push_native_episode
from langmani.v2.push_collection import load_attempt_records
from langmani.v2.push_dataset import (
    PushCollectionConfig,
    PushDatasetContractError,
    audit_split_leakage,
    build_split_manifest,
    sha256_json,
)

PHASE2B2_DATASET_ID = "langmani/phase2b-push-v2"
PHASE2B1_DATASET_ID = "langmani/phase2b-push-v1"
PHASE2B2_PACKAGE_SCHEMA = "langmani-v2-phase2b2-accepted-dataset-package-v0"
AcceptanceStatus = Literal["ACCEPTED", "REJECTED", "INCOMPLETE"]


@dataclass(frozen=True, slots=True)
class AcceptedDatasetPackage:
    """Complete content identity for one accepted Phase 2B.2 dataset."""

    dataset_id: str
    dataset_version: str
    source_commit: str
    environment_identity: str
    raw_manifest_sha256: str
    lerobot_tree_digest: str
    meta_info_sha256: str
    split_manifest_sha256: str
    raw_archive_sha256: str
    lerobot_archive_sha256: str
    episode_count: int
    frame_count: int
    total_bytes: int
    fps: float
    camera_keys: tuple[str, ...]
    state_dimension: int
    action_dimension: int
    lerobot_version: str
    acceptance_status: AcceptanceStatus

    def __post_init__(self) -> None:
        if self.dataset_id != PHASE2B2_DATASET_ID or self.dataset_version != "v2":
            raise PushDatasetContractError("accepted package must use the independent v2 identity")
        if self.dataset_id == PHASE2B1_DATASET_ID:
            raise PushDatasetContractError("v1 identity cannot be promoted as Phase 2B.2")
        if len(self.source_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.source_commit
        ):
            raise PushDatasetContractError("accepted package source_commit is not a full Git SHA")
        for name in (
            "raw_manifest_sha256",
            "lerobot_tree_digest",
            "meta_info_sha256",
            "split_manifest_sha256",
            "raw_archive_sha256",
            "lerobot_archive_sha256",
        ):
            _require_digest(getattr(self, name), name)
        if self.episode_count < 0 or self.frame_count < 0 or self.total_bytes < 0:
            raise PushDatasetContractError("accepted package counts cannot be negative")
        if self.fps != 20.0 or self.camera_keys != (IMAGE_FEATURE_KEY,):
            raise PushDatasetContractError("accepted package camera/fps contract changed")
        if self.state_dimension != 9 or self.action_dimension != 8:
            raise PushDatasetContractError("accepted package state/action contract changed")
        if self.lerobot_version != "0.6.0":
            raise PushDatasetContractError("accepted package LeRobot version changed")
        if self.acceptance_status not in {"ACCEPTED", "REJECTED", "INCOMPLETE"}:
            raise PushDatasetContractError("unknown accepted package status")
        if self.acceptance_status == "ACCEPTED" and (
            self.episode_count < 400 or self.frame_count < 50_000
        ):
            raise PushDatasetContractError("undersized Phase 2B.2 package cannot be accepted")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": PHASE2B2_PACKAGE_SCHEMA,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "source_commit": self.source_commit,
            "environment_identity": self.environment_identity,
            "raw_manifest_sha256": self.raw_manifest_sha256,
            "lerobot_tree_digest": self.lerobot_tree_digest,
            "meta_info_sha256": self.meta_info_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "raw_archive_sha256": self.raw_archive_sha256,
            "lerobot_archive_sha256": self.lerobot_archive_sha256,
            "episode_count": self.episode_count,
            "frame_count": self.frame_count,
            "total_bytes": self.total_bytes,
            "fps": self.fps,
            "camera_keys": list(self.camera_keys),
            "state_dimension": self.state_dimension,
            "action_dimension": self.action_dimension,
            "lerobot_version": self.lerobot_version,
            "acceptance_status": self.acceptance_status,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        if value.get("schema_version") != PHASE2B2_PACKAGE_SCHEMA:
            raise PushDatasetContractError("unsupported Phase 2B.2 package schema")
        camera_keys = value.get("camera_keys")
        if not isinstance(camera_keys, list) or not all(
            isinstance(item, str) for item in camera_keys
        ):
            raise PushDatasetContractError("package camera_keys are malformed")
        return cls(
            dataset_id=_string(value.get("dataset_id"), "dataset_id"),
            dataset_version=_string(value.get("dataset_version"), "dataset_version"),
            source_commit=_string(value.get("source_commit"), "source_commit"),
            environment_identity=_string(value.get("environment_identity"), "environment_identity"),
            raw_manifest_sha256=_string(value.get("raw_manifest_sha256"), "raw_manifest_sha256"),
            lerobot_tree_digest=_string(value.get("lerobot_tree_digest"), "lerobot_tree_digest"),
            meta_info_sha256=_string(value.get("meta_info_sha256"), "meta_info_sha256"),
            split_manifest_sha256=_string(
                value.get("split_manifest_sha256"), "split_manifest_sha256"
            ),
            raw_archive_sha256=_string(value.get("raw_archive_sha256"), "raw_archive_sha256"),
            lerobot_archive_sha256=_string(
                value.get("lerobot_archive_sha256"), "lerobot_archive_sha256"
            ),
            episode_count=_integer(value.get("episode_count"), "episode_count"),
            frame_count=_integer(value.get("frame_count"), "frame_count"),
            total_bytes=_integer(value.get("total_bytes"), "total_bytes"),
            fps=_number(value.get("fps"), "fps"),
            camera_keys=tuple(cast(list[str], camera_keys)),
            state_dimension=_integer(value.get("state_dimension"), "state_dimension"),
            action_dimension=_integer(value.get("action_dimension"), "action_dimension"),
            lerobot_version=_string(value.get("lerobot_version"), "lerobot_version"),
            acceptance_status=cast(
                AcceptanceStatus, _string(value.get("acceptance_status"), "acceptance_status")
            ),
        )


def file_tree_manifest(root: str | Path) -> dict[str, object]:
    """Hash every regular file under root using path-independent relative names."""

    source = Path(root).resolve()
    if not source.is_dir():
        raise PushDatasetContractError(f"tree root is unavailable: {source}")
    entries: list[dict[str, object]] = []
    total = 0
    for path in sorted(source.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise PushDatasetContractError(f"dataset trees cannot contain symlinks: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(source).as_posix()
        size = path.stat().st_size
        entries.append(
            {"path": relative, "size_bytes": size, "sha256": f"sha256:{sha256_file(path)}"}
        )
        total += size
    if not entries:
        raise PushDatasetContractError("cannot identify an empty dataset tree")
    return {
        "schema_version": "langmani-v2-phase2b2-file-tree-v0",
        "file_count": len(entries),
        "total_bytes": total,
        "entries": entries,
        "tree_digest": sha256_json(entries),
    }


def build_raw_episode_manifest(
    *,
    config: PushCollectionConfig,
    source_root: str | Path,
    stages: Sequence[str],
) -> dict[str, object]:
    """Rehash every native v2 episode and bind the clean producer commits."""

    root = Path(source_root).resolve()
    records = load_attempt_records(root, stages)
    accepted_path = root / "accepted" / f"{'_'.join(stages)}.json"
    accepted = _read_json(accepted_path)
    accepted_records = accepted.get("records")
    if not isinstance(accepted_records, list) or not all(
        isinstance(item, dict) for item in accepted_records
    ):
        raise PushDatasetContractError("accepted raw records are malformed")
    runtime_commits: set[str] = set()
    for stage in stages:
        owner = _read_json(root / "manifests" / f"{stage}_owner.json")
        runtime = owner.get("runtime")
        if not isinstance(runtime, Mapping) or runtime.get("git_clean") is not True:
            raise PushDatasetContractError("raw collection lacks a clean runtime identity")
        runtime_commits.add(_string(runtime.get("git_commit"), "runtime git_commit"))
    if len(runtime_commits) != 1:
        raise PushDatasetContractError("Phase 2B.2 stages use different source commits")
    episodes: list[dict[str, object]] = []
    for record in records:
        h5_path = _source_path(root, record.get("raw_h5_path"))
        json_path = _source_path(root, record.get("raw_json_path"))
        native_id = record.get("native_episode_id")
        if isinstance(native_id, bool) or not isinstance(native_id, int):
            if record.get("generation_accepted") is True:
                raise PushDatasetContractError("generation-accepted record lacks native episode")
            continue
        validation = validate_push_native_episode(h5_path, json_path, native_episode_id=native_id)
        episodes.append(
            {
                "episode_id": record["episode_id"],
                "accepted": record.get("accepted") is True,
                "seed": record["seed"],
                "h5_path": h5_path.relative_to(root).as_posix(),
                "json_path": json_path.relative_to(root).as_posix(),
                "h5_sha256": f"sha256:{sha256_file(h5_path)}",
                "json_sha256": f"sha256:{sha256_file(json_path)}",
                "trajectory_sha256": validation.trajectory_sha256,
                "action_count": validation.action_count,
            }
        )
    episode_ids = [str(item["episode_id"]) for item in episodes]
    if len(episode_ids) != len(set(episode_ids)):
        raise PushDatasetContractError("raw manifest contains duplicate episode IDs")
    accepted_count = sum(item["accepted"] is True for item in episodes)
    return {
        "schema_version": "langmani-v2-phase2b2-raw-episode-manifest-v0",
        "dataset_id": PHASE2B2_DATASET_ID,
        "collection_fingerprint": config.fingerprint,
        "source_commit": next(iter(runtime_commits)),
        "stages": list(stages),
        "attempt_count": len(records),
        "accepted_episode_count": accepted_count,
        "rejected_episode_count": len(records) - accepted_count,
        "accepted_manifest_sha256": f"sha256:{sha256_file(accepted_path)}",
        "episodes": episodes,
        "episodes_digest": sha256_json(episodes),
        "optimizer_steps": 0,
    }


def scan_lerobot_quality(
    *,
    config: PushCollectionConfig,
    source_root: str | Path,
    export_root: str | Path,
    stages: Sequence[str],
    expert_report: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Read every exported frame and evaluate the complete Phase 2B.2 data gate."""

    from lerobot.datasets import LeRobotDataset  # type: ignore[import-untyped]

    source = Path(source_root).resolve()
    exported = Path(export_root).resolve()
    accepted = _read_json(source / "accepted" / f"{'_'.join(stages)}.json")
    records = accepted.get("records")
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise PushDatasetContractError("accepted records are malformed")
    accepted_records = cast(list[dict[str, object]], records)
    split_manifest = build_split_manifest(accepted_records)
    leakage = audit_split_leakage(accepted_records, split_manifest)
    export_manifest = _read_json(exported / "langmani" / "export_manifest.json")
    repo_prefix = _string(export_manifest.get("repo_id_prefix"), "repo_id_prefix")
    if repo_prefix != PHASE2B2_DATASET_ID:
        raise PushDatasetContractError("export repo ID is not Phase 2B.2 v2")
    bounds = cast(Mapping[str, object], config.payload["action_bounds"])
    low = np.asarray(bounds["low"], dtype=np.float64)
    high = np.asarray(bounds["high"], dtype=np.float64)
    state_values: list[np.ndarray] = []
    action_values: list[np.ndarray] = []
    frame_count = 0
    black_frames = 0
    duplicate_frames = 0
    duplicate_actions = 0
    missing_tasks = 0
    nonfinite_observations = 0
    nonfinite_states = 0
    nonfinite_actions = 0
    action_boundary_violations = 0
    timestamp_errors = 0
    frame_index_errors = 0
    previous_frame: dict[tuple[str, int], bytes] = {}
    previous_action: dict[tuple[str, int], bytes] = {}
    expected_frame: dict[tuple[str, int], int] = {}
    previous_timestamp: dict[tuple[str, int], float] = {}
    split_counts: dict[str, dict[str, int]] = {}
    assignments = cast(Mapping[str, object], split_manifest["assignments"])
    split_names = tuple(cast(Mapping[str, int], split_manifest["counts"]))
    for split in split_names:
        dataset = LeRobotDataset(
            repo_id=f"{repo_prefix}-{split}",
            root=exported / "splits" / split,
            video_backend="pyav",
            return_uint8=True,
        )
        split_counts[split] = {
            "episodes": int(dataset.num_episodes),
            "frames": int(dataset.num_frames),
        }
        for index in range(int(dataset.num_frames)):
            item = dataset[index]
            rgb = _numpy(item[IMAGE_FEATURE_KEY])
            state = _numpy(item[STATE_FEATURE_KEY]).astype(np.float64)
            action = _numpy(item[ACTION_FEATURE_KEY]).astype(np.float64)
            if rgb.shape != (3, 256, 256) or rgb.dtype != np.uint8:
                raise PushDatasetContractError("decoded RGB contract changed")
            if state.shape != (9,) or action.shape != (8,):
                raise PushDatasetContractError("decoded state/action contract changed")
            episode_index = int(_scalar(item["episode_index"]))
            frame_index = int(_scalar(item["frame_index"]))
            timestamp = float(_scalar(item["timestamp"]))
            episode_key = (split, episode_index)
            expected = expected_frame.get(episode_key, 0)
            if frame_index != expected:
                frame_index_errors += 1
            if episode_key in previous_timestamp and timestamp <= previous_timestamp[episode_key]:
                timestamp_errors += 1
            expected_frame[episode_key] = frame_index + 1
            previous_timestamp[episode_key] = timestamp
            rgb_bytes = rgb.tobytes(order="C")
            action_bytes = action.astype(np.float32).tobytes(order="C")
            if previous_frame.get(episode_key) == rgb_bytes:
                duplicate_frames += 1
            if previous_action.get(episode_key) == action_bytes:
                duplicate_actions += 1
            previous_frame[episode_key] = rgb_bytes
            previous_action[episode_key] = action_bytes
            black_frames += int(not np.any(rgb))
            nonfinite_observations += int(not np.all(np.isfinite(rgb)))
            nonfinite_states += int(not np.all(np.isfinite(state)))
            nonfinite_actions += int(not np.all(np.isfinite(action)))
            action_boundary_violations += int(np.any(action < low) or np.any(action > high))
            task = item.get("task")
            missing_tasks += int(not isinstance(task, str) or not task.strip())
            state_values.append(state)
            action_values.append(action)
            frame_count += 1
    state_array = np.stack(state_values)
    action_array = np.stack(action_values)
    tree = file_tree_manifest(exported)
    complete = _read_json(exported / "langmani" / "complete.json")
    minimum_frames = config.integer("minimum_total_frames")
    minimum_episodes = config.integer("minimum_accepted_episodes")
    gates = {
        "expert_success_rate": _number(expert_report.get("success_rate"), "success_rate") >= 0.95,
        "expert_gate_passed": expert_report.get("passed") is True,
        "accepted_successful_episodes": len(accepted_records) >= minimum_episodes,
        "total_frames": frame_count >= minimum_frames,
        "meta_info_present": all(
            (exported / "splits" / split / "meta" / "info.json").is_file() for split in split_names
        ),
        "episode_integrity": frame_index_errors == 0 and timestamp_errors == 0,
        "camera_contract_matches": True,
        "state_dimension": state_array.shape[1] == 9,
        "action_dimension": action_array.shape[1] == 8,
        "action_boundary_violations": action_boundary_violations == 0,
        "nonfinite_observations": nonfinite_observations == 0,
        "nonfinite_states": nonfinite_states == 0,
        "nonfinite_actions": nonfinite_actions == 0,
        "duplicate_episodes": leakage.get("duplicate_episode_id_count") == 0,
        "duplicate_trajectories": leakage.get("duplicate_trajectory_hash_count") == 0,
        "split_leakage": leakage.get("passed") is True,
        "tasks_readable": missing_tasks == 0,
        "export_complete": complete.get("export_manifest_sha256")
        == f"sha256:{sha256_file(exported / 'langmani' / 'export_manifest.json')}",
        "optimizer_steps_zero": True,
    }
    report = {
        "schema_version": "langmani-v2-phase2b2-data-quality-v0",
        "dataset_id": PHASE2B2_DATASET_ID,
        "attempted_episodes": len(load_attempt_records(source, stages)),
        "accepted_successful_episodes": len(accepted_records),
        "failed_episodes": len(load_attempt_records(source, stages)) - len(accepted_records),
        "total_frames": frame_count,
        "total_bytes": _integer(tree.get("total_bytes"), "total_bytes"),
        "split_counts": split_counts,
        "image": {
            "shape": [3, 256, 256],
            "dtype": "uint8",
            "black_frames": black_frames,
            "consecutive_duplicate_frames": duplicate_frames,
            "nonfinite": nonfinite_observations,
        },
        "state": {
            "shape": [9],
            "dtype": "float32",
            "min": state_array.min(axis=0).tolist(),
            "max": state_array.max(axis=0).tolist(),
            "mean": state_array.mean(axis=0).tolist(),
            "std": state_array.std(axis=0).tolist(),
            "nonfinite": nonfinite_states,
        },
        "action": {
            "shape": [8],
            "dtype": "float32",
            "min": action_array.min(axis=0).tolist(),
            "max": action_array.max(axis=0).tolist(),
            "mean": action_array.mean(axis=0).tolist(),
            "std": action_array.std(axis=0).tolist(),
            "boundary_violations": action_boundary_violations,
            "nonfinite": nonfinite_actions,
            "consecutive_duplicate_actions": duplicate_actions,
        },
        "language": {"missing_or_empty": missing_tasks},
        "episode_integrity": {
            "frame_index_errors": frame_index_errors,
            "timestamp_errors": timestamp_errors,
        },
        "leakage": leakage,
        "tree": tree,
        "gates": gates,
        "optimizer_steps": 0,
        "passed": all(gates.values()),
    }
    manifest = {
        "schema_version": "langmani-v2-phase2b2-dataset-manifest-v0",
        "dataset_id": PHASE2B2_DATASET_ID,
        "collection_fingerprint": config.fingerprint,
        "stages": list(stages),
        "episode_count": len(accepted_records),
        "frame_count": frame_count,
        "total_bytes": tree["total_bytes"],
        "lerobot_tree_digest": tree["tree_digest"],
        "split_manifest_sha256": f"sha256:{sha256_file(exported / 'langmani' / 'split_manifest.json')}",
        "meta_info_sha256": _meta_info_digest(exported, split_names),
        "export_manifest_sha256": f"sha256:{sha256_file(exported / 'langmani' / 'export_manifest.json')}",
        "split_assignments_digest": sha256_json(assignments),
        "quality_passed": report["passed"],
        "optimizer_steps": 0,
    }
    return manifest, report


def create_deterministic_tar_gz(
    *, source_root: str | Path, archive_path: str | Path, prefix: str
) -> dict[str, object]:
    """Create an atomic deterministic gzip-compressed tar archive."""

    source = Path(source_root).resolve()
    destination = Path(archive_path).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    archive_prefix = PurePosixPath(prefix)
    if archive_prefix.is_absolute() or ".." in archive_prefix.parts or not archive_prefix.parts:
        raise PushDatasetContractError("archive prefix must be one safe relative path")
    tree = file_tree_manifest(source)
    staging = destination.parent / f".{destination.name}.partial"
    if staging.exists():
        raise FileExistsError(staging)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        staging.open("xb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for entry in cast(list[dict[str, object]], tree["entries"]):
            relative = PurePosixPath(str(entry["path"]))
            path = source / Path(*relative.parts)
            info = tarfile.TarInfo((archive_prefix / relative).as_posix())
            info.size = path.stat().st_size
            info.mode = 0o444
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as stream:
                archive.addfile(info, stream)
    with staging.open("r+b") as completed:
        completed.flush()
        os.fsync(completed.fileno())
    os.replace(staging, destination)
    return {
        "path": destination.as_posix(),
        "size_bytes": destination.stat().st_size,
        "sha256": f"sha256:{sha256_file(destination)}",
        "source_tree_digest": tree["tree_digest"],
        "prefix": prefix,
    }


def restore_validate_archive(
    *,
    archive_path: str | Path,
    destination: str | Path,
    prefix: str,
    expected_tree_digest: str,
) -> dict[str, object]:
    """Safely extract an archive and rehash its complete restored tree."""

    archive_file = Path(archive_path).resolve()
    root = Path(destination).resolve()
    if root.exists():
        raise FileExistsError(root)
    staging = root.parent / f".{root.name}.partial"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        with tarfile.open(archive_file, mode="r:gz") as archive:
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or not member.isfile():
                    raise PushDatasetContractError("archive contains an unsafe or non-file member")
                stream = archive.extractfile(member)
                if stream is None:
                    raise PushDatasetContractError("archive member cannot be read")
                target = staging / Path(*path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as output:
                    while chunk := stream.read(1024 * 1024):
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
        restored = staging / prefix
        tree = file_tree_manifest(restored)
        if tree["tree_digest"] != expected_tree_digest:
            raise PushDatasetContractError("restored archive tree digest changed")
        os.replace(staging, root)
        return {
            "archive_sha256": f"sha256:{sha256_file(archive_file)}",
            "restored_tree_digest": tree["tree_digest"],
            "file_count": tree["file_count"],
            "total_bytes": tree["total_bytes"],
            "loader_readback_pending": True,
            "passed": True,
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_replication_report(
    *,
    expected_sha256: str,
    replicas: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Require hashes for every completed replica and report third-copy truthfully."""

    _require_digest(expected_sha256, "expected_sha256")
    values: list[dict[str, object]] = []
    for item in replicas:
        location = _string(item.get("location"), "replica location")
        completed = item.get("copy_completed") is True
        digest = item.get("sha256")
        if completed and digest != expected_sha256:
            raise PushDatasetContractError(f"replica hash changed: {location}")
        values.append(
            {
                "location": location,
                "copy_completed": completed,
                "file_size": _integer(item.get("file_size", 0), "file_size"),
                "sha256": digest,
                "restore_test_passed": item.get("restore_test_passed") is True,
            }
        )
    complete = [item for item in values if item["copy_completed"] is True]
    restored = [item for item in complete if item["restore_test_passed"] is True]
    declared_pending = [item for item in values if item["copy_completed"] is not True]
    return {
        "schema_version": "langmani-v2-phase2b2-replication-v0",
        "expected_sha256": expected_sha256,
        "replicas": values,
        "completed_copy_count": len(complete),
        "restore_tested_copy_count": len(restored),
        "three_independent_copies": len(complete) >= 3,
        "declared_pending_copy_count": len(declared_pending),
        "minimum_temporary_persistence_gate": len(restored) >= 2,
        "future_training_persistence_gate": len(complete) >= 3,
        "passed": len(restored) >= 2 and len(values) >= 3,
    }


def write_json_once(path: str | Path, value: Mapping[str, object]) -> None:
    destination = Path(path)
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if destination.exists():
        if destination.read_bytes() != encoded:
            raise PushDatasetContractError(f"refusing to overwrite changed artifact: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _source_path(root: Path, value: object) -> Path:
    path = Path(str(value))
    resolved = path if path.is_absolute() else root / path
    try:
        resolved.resolve().relative_to(root)
    except ValueError as error:
        raise PushDatasetContractError("raw episode path escapes the source root") from error
    return resolved


def _meta_info_digest(root: Path, splits: Sequence[str]) -> str:
    entries = [
        {
            "split": split,
            "sha256": f"sha256:{sha256_file(root / 'splits' / split / 'meta' / 'info.json')}",
        }
        for split in splits
    ]
    return sha256_json(entries)


def _numpy(value: object) -> np.ndarray:
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    return np.asarray(candidate)


def _scalar(value: object) -> Any:
    array = _numpy(value)
    if array.size != 1:
        raise PushDatasetContractError("LeRobot scalar field contains multiple values")
    return array.reshape(-1)[0]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PushDatasetContractError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise PushDatasetContractError(f"JSON root must be an object: {path}")
    return value


def _require_digest(value: str, label: str) -> None:
    if not value.startswith("sha256:") or len(value) != 71:
        raise PushDatasetContractError(f"{label} must be a prefixed SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value.removeprefix("sha256:")):
        raise PushDatasetContractError(f"{label} is not lowercase hexadecimal")


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PushDatasetContractError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PushDatasetContractError(f"{label} must be a non-negative integer")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PushDatasetContractError(f"{label} must be numeric")
    return float(value)


__all__ = [
    "PHASE2B1_DATASET_ID",
    "PHASE2B2_DATASET_ID",
    "PHASE2B2_PACKAGE_SCHEMA",
    "AcceptedDatasetPackage",
    "build_raw_episode_manifest",
    "build_replication_report",
    "create_deterministic_tar_gz",
    "file_tree_manifest",
    "restore_validate_archive",
    "scan_lerobot_quality",
    "write_json_once",
]
