"""Native ManiSkill production runtime for LangMani 2.0 Phase 2B.6.

The runtime replays only pinned official actions and writes nonprivileged
pre-action observations.  It contains no planner, expert, policy, optimizer,
checkpoint, backward pass, or training path.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import time
import zipfile
from collections import defaultdict
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
    HF_REVISION,
    IMAGE_SHAPE,
    MANISKILL_SOURCE_REVISION,
    MANISKILL_VERSION,
    PANDA_ACTION_HIGH,
    PANDA_ACTION_LOW,
    SOURCE_LICENSE,
    canonical_json_sha256,
    sha256_file,
    stable_reset_identity,
    validate_action_array,
    validate_metadata_document,
    validate_transition_lengths,
)
from langmani.v2.phase2b5_runtime import (
    _extract_rgb,
    _h5_state_at,
    _prepare_replay_initial_state,
    _scalar_bool,
    _state_error,
    _task_object_pose_error,
    _task_object_raw_poses,
)
from langmani.v2.phase2b6 import (
    EXPECTED_EPISODES,
    EXPECTED_TRANSITIONS,
    PRIMARY_SPLITS,
    SOURCE_COMMIT,
    TARGET_BRANCH,
    TASK_IDS,
    Phase2B6ContractError,
    apply_visual_shift,
    build_cross_skill_folds,
    build_language_template_manifest,
    build_padding_audit,
    build_primary_split_manifest,
    build_task_balance_manifest,
    fingerprinted,
    load_production_spec,
    source_reset_identity,
    source_trajectory_identity,
    summarize_replay_gate,
    verify_phase2b5_package,
)


class Phase2B6RuntimeError(RuntimeError):
    """Raised when native production cannot preserve the frozen contract."""


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2B6RuntimeError(f"failed to read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise Phase2B6RuntimeError(f"{path} must contain one object")
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.parent / f".{path.name}.partial"
    if staging.exists():
        raise Phase2B6RuntimeError(f"stale JSON staging file exists: {staging}")
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


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise Phase2B6RuntimeError(
                    f"invalid progress JSONL at line {line_number}"
                ) from error
            if not isinstance(value, dict):
                raise Phase2B6RuntimeError("progress JSONL entries must be objects")
            rows.append(value)
    return rows


def _selected_paths(source_root: Path, task_id: str) -> tuple[Path, Path]:
    directory = source_root / "expanded" / task_id / "motionplanning"
    return directory / "trajectory.h5", directory / "trajectory.json"


def _task_slug(task_id: str) -> str:
    return task_id.removesuffix("-v1").lower()


def _sha256_stream(stream: Any, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(chunk_size):
        digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: object, *, dtype: np.dtype[Any] | None = None) -> str:
    array = np.asarray(value, dtype=dtype)
    contiguous = np.ascontiguousarray(array)
    return "sha256:" + hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _h5_state_sha256(group: h5py.Group, index: int) -> str:
    digest = hashlib.sha256()

    def visit(current: h5py.Group, prefix: str) -> None:
        for key in sorted(current):
            item = current[key]
            name = f"{prefix}/{key}" if prefix else key
            if isinstance(item, h5py.Group):
                visit(item, name)
            else:
                array = np.ascontiguousarray(np.asarray(item[index]))
                digest.update(name.encode())
                digest.update(str(array.dtype).encode())
                digest.update(json.dumps(list(array.shape)).encode())
                digest.update(array.tobytes(order="C"))

    visit(group, "")
    return "sha256:" + digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _runtime_versions() -> dict[str, object]:
    packages: dict[str, str] = {}
    for name in ("mani-skill", "sapien", "torch", "numpy", "h5py", "pillow"):
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = "not_installed"
    return {
        "python": platform.python_version(),
        "executable": Path(os.sys.executable).resolve().as_posix(),
        "environment_prefix": (
            os.environ.get("CONDA_PREFIX")
            or os.environ.get("VIRTUAL_ENV")
            or Path(os.sys.executable).resolve().parents[1].as_posix()
        ),
        "packages": packages,
    }


def _gpu_audit() -> dict[str, object]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        rows: list[dict[str, object]] = []
        for line in result.stdout.strip().splitlines():
            fields = [part.strip() for part in line.split(",")]
            rows.append(
                {
                    "uuid": fields[0],
                    "name": fields[1],
                    "driver_version": fields[2],
                    "memory_used_mib": int(fields[3]),
                    "memory_total_mib": int(fields[4]),
                    "utilization_percent": int(fields[5]),
                }
            )
        return {"available": bool(rows), "devices": rows}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}


def _process_audit() -> dict[str, object]:
    patterns = (
        "train_act",
        "smolvla",
        "vla_jepa",
        "collect_",
        "phase2b6",
        "lerobot",
        "optimizer",
    )
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,args="],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"query_succeeded": False, "error": f"{type(error).__name__}: {error}"}
    own = {os.getpid(), os.getppid()}
    matches: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) != 3:
            continue
        pid = int(fields[0])
        command = fields[2]
        lower = command.lower()
        if pid not in own and any(pattern in lower for pattern in patterns):
            matches.append({"pid": pid, "ppid": int(fields[1]), "command": command})
    prohibited = [
        row
        for row in matches
        if any(
            pattern in str(row["command"]).lower()
            for pattern in ("train_act", "smolvla", "vla_jepa", "collect_", "optimizer")
        )
    ]
    return {
        "query_succeeded": True,
        "matching_processes": matches,
        "prohibited_active_processes": prohibited,
        "passed": not prohibited,
    }


def _storage_audit(paths: Mapping[str, Path]) -> dict[str, object]:
    reports: dict[str, object] = {}
    for label, raw_path in paths.items():
        path = raw_path.resolve()
        existing = path
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        usage = shutil.disk_usage(existing)
        reports[label] = {
            "requested_path": path.as_posix(),
            "measured_at": existing.as_posix(),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        }
    minimum_free = min(
        int(cast(Mapping[str, object], report)["free_bytes"]) for report in reports.values()
    )
    estimated_peak_increment_bytes = 60 * 1024**3
    return {
        "volumes": reports,
        "minimum_free_bytes": minimum_free,
        "estimated_peak_increment_bytes": estimated_peak_increment_bytes,
        "estimate_components": {
            "compressed_replay_intermediates": "approximately 28-35 GiB",
            "primary_lerobot_and_visual_shift": "approximately 3-8 GiB",
            "archive_and_restore": "approximately 6-16 GiB combined",
        },
        "passed": minimum_free >= estimated_peak_increment_bytes,
    }


def freeze_production_spec(*, spec_path: Path, production_root: Path) -> dict[str, object]:
    """Create or revalidate the immutable production-run specification lock."""

    spec = load_production_spec(spec_path)
    lock_directory = production_root / "run"
    lock_directory.mkdir(parents=True, exist_ok=True)
    frozen_path = lock_directory / "frozen_production_specification.json"
    lock_path = lock_directory / "specification_lock.json"
    frozen = {key: value for key, value in spec.items() if not key.startswith("_")}
    expected_payload = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-specification-lock-v0",
            "production_run_id": spec["production_run_id"],
            "specification_sha256": spec["_spec_sha256"],
            "frozen_specification_fingerprint": canonical_json_sha256(frozen),
            "source_commit": SOURCE_COMMIT,
            "full_production_started": False,
        }
    )
    if frozen_path.exists() or lock_path.exists():
        if not frozen_path.is_file() or not lock_path.is_file():
            raise Phase2B6RuntimeError("production specification lock is incomplete")
        if _read_json(frozen_path) != frozen or _read_json(lock_path) != expected_payload:
            raise Phase2B6RuntimeError("existing production specification lock differs")
    else:
        _write_json(frozen_path, frozen)
        _write_json(lock_path, expected_payload)
    return expected_payload


def repository_environment_audit(
    *,
    repo_root: Path,
    source_root: Path,
    production_root: Path,
    archive_root: Path,
    restore_root: Path,
    spec: Mapping[str, object],
    network_turbo_sourced: bool,
) -> dict[str, object]:
    """Verify the exact clean producer checkout and server capacity."""

    head = _git(repo_root, "rev-parse", "HEAD")
    branch = _git(repo_root, "branch", "--show-current")
    status = _git(repo_root, "status", "--porcelain=v1")
    upstream = _git(repo_root, "rev-parse", "@{upstream}")
    remote = _git(repo_root, "remote", "get-url", "origin")
    merge_base = _git(repo_root, "merge-base", SOURCE_COMMIT, head)
    merge_commits = _git(repo_root, "rev-list", "--merges", f"{SOURCE_COMMIT}..{head}")
    try:
        github_line = _git(
            repo_root,
            "ls-remote",
            "origin",
            f"refs/heads/{TARGET_BRANCH}",
        )
        github_sha = github_line.split()[0] if github_line else None
    except subprocess.CalledProcessError:
        github_sha = None
    storage = _storage_audit(
        {
            "official_source": source_root,
            "production": production_root,
            "archive": archive_root,
            "restore": restore_root,
        }
    )
    process = _process_audit()
    checks = {
        "target_branch": branch == TARGET_BRANCH,
        "source_commit_is_ancestor": merge_base == SOURCE_COMMIT,
        "no_merge_commits_since_source": not merge_commits,
        "worktree_clean": not status,
        "upstream_matches_head": upstream == head,
        "github_matches_head": github_sha == head,
        "phase2b5_package_identity": (
            cast(Mapping[str, object], spec["source_authorization"])["accepted_package_sha256"]
            == "sha256:22e93d609fc4564d0bdb3e5fc7571cf8971c8a86b3cf3bdaa910108621bcbcc2"
        ),
        "process_state": process.get("passed") is True,
        "storage": storage["passed"] is True,
        "network_turbo_sourced": network_turbo_sourced,
    }
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-repository-environment-audit-v0",
            "created_at_utc": _timestamp(),
            "repository_path": repo_root.resolve().as_posix(),
            "worktree_path": repo_root.resolve().as_posix(),
            "branch": branch,
            "source_commit": SOURCE_COMMIT,
            "head": head,
            "upstream": upstream,
            "github_branch_sha": github_sha,
            "remote": remote,
            "source_package_identity": cast(Mapping[str, object], spec["source_authorization"])[
                "accepted_package_sha256"
            ],
            "runtime_environments": {"maniskill": _runtime_versions()},
            "storage": storage,
            "gpu": _gpu_audit(),
            "process_state": process,
            "authorization_state": {
                "student_policy_training_started": False,
                "optimizer_steps": 0,
                "act_training_started": False,
                "smolvla_training_started": False,
                "vla_jepa_training_started": False,
                "custom_push_expert_route_active": False,
            },
            "checks": checks,
            "passed": all(checks.values()),
        }
    )


def verify_source_bytes(*, source_root: Path, spec: Mapping[str, object]) -> dict[str, object]:
    """Rehash source ZIPs and prove selected extracted files match ZIP members."""

    task_specs = cast(Mapping[str, object], spec["tasks"])
    task_reports: list[dict[str, object]] = []
    for task_id in TASK_IDS:
        task = cast(Mapping[str, object], task_specs[task_id])
        archive = source_root / str(task["source_zip_relative_path"])
        expected_hash = str(task["source_zip_sha256"])
        checks = {
            "ordinary_file": archive.is_file() and not archive.is_symlink(),
            "size": archive.is_file() and archive.stat().st_size == task["source_zip_size_bytes"],
            "sha256": archive.is_file() and "sha256:" + sha256_file(archive) == expected_hash,
        }
        members: list[dict[str, object]] = []
        extraction_matches: dict[str, bool] = {}
        with zipfile.ZipFile(archive) as bundle:
            member_names = {info.filename for info in bundle.infolist()}
            for info in bundle.infolist():
                pure = PurePosixPath(info.filename)
                mode = info.external_attr >> 16
                if pure.is_absolute() or ".." in pure.parts or (mode & 0o170000) == 0o120000:
                    raise Phase2B6RuntimeError(
                        f"unsafe source ZIP member {archive.name}:{info.filename}"
                    )
                members.append(
                    {
                        "path": info.filename,
                        "size_bytes": info.file_size,
                        "compressed_size_bytes": info.compress_size,
                    }
                )
            for relative in (
                f"{task_id}/motionplanning/trajectory.h5",
                f"{task_id}/motionplanning/trajectory.json",
            ):
                if relative not in member_names:
                    raise Phase2B6RuntimeError(f"source ZIP lacks {relative}")
                expanded = source_root / "expanded" / relative
                if expanded.is_symlink() or not expanded.is_file():
                    raise Phase2B6RuntimeError(f"selected extracted source is missing: {expanded}")
                with bundle.open(relative) as stream:
                    member_hash = _sha256_stream(stream)
                extraction_matches[relative] = member_hash == sha256_file(expanded)
        checks["safe_zip_members"] = True
        checks["selected_members_present"] = len(extraction_matches) == 2
        checks["reproducible_extraction"] = all(extraction_matches.values())
        task_reports.append(
            {
                "task_id": task_id,
                "source_zip": archive.as_posix(),
                "size_bytes": archive.stat().st_size,
                "sha256": "sha256:" + sha256_file(archive),
                "official_repository": cast(Mapping[str, object], spec["source_authorization"])[
                    "official_source_repository"
                ],
                "official_revision": HF_REVISION,
                "license": SOURCE_LICENSE,
                "member_count": len(members),
                "selected_extraction_matches": extraction_matches,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-source-integrity-audit-v0",
            "created_at_utc": _timestamp(),
            "official_source_revision": HF_REVISION,
            "maniskill_source_revision": MANISKILL_SOURCE_REVISION,
            "source_files_modified": False,
            "tasks": task_reports,
            "passed": all(report["passed"] is True for report in task_reports),
        }
    )


def validate_source_schema(
    *,
    source_root: Path,
    production_root: Path,
    spec: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Validate all 3,000 source trajectories and create the identity inventory."""

    task_specs = cast(Mapping[str, object], spec["tasks"])
    inventory: list[dict[str, object]] = []
    task_reports: list[dict[str, object]] = []
    for task_id in TASK_IDS:
        h5_path, json_path = _selected_paths(source_root, task_id)
        document = _read_json(json_path)
        episodes = validate_metadata_document(document, expected_task_id=task_id)
        by_id = {int(row["episode_id"]): row for row in episodes}
        action_hashes: list[str] = []
        reset_hashes: list[str] = []
        lengths: list[int] = []
        malformed: list[dict[str, object]] = []
        nonfinite_values = 0
        bound_violations = 0
        terminal_agreements = 0
        with h5py.File(h5_path, "r") as trajectories:
            expected_groups = {f"traj_{episode_id}" for episode_id in by_id}
            if set(trajectories) != expected_groups:
                raise Phase2B6RuntimeError(f"{task_id} HDF5/metadata groups differ")
            for episode_id, metadata in sorted(by_id.items()):
                group = trajectories[f"traj_{episode_id}"]
                try:
                    actions = np.asarray(group["actions"])
                    action_report = validate_action_array(actions)
                    transition_lengths = {
                        name: int(item.shape[0])
                        for name, item in group.items()
                        if isinstance(item, h5py.Dataset) and name != "actions"
                    }
                    env_state_lengths = _leaf_lengths(cast(h5py.Group, group["env_states"]))
                    validate_transition_lengths(
                        action_count=len(actions),
                        transition_lengths=transition_lengths,
                        env_state_lengths=env_state_lengths,
                    )
                    if metadata["elapsed_steps"] != len(actions):
                        raise Phase2B6ContractError("elapsed_steps differs from action count")
                    source_success = bool(metadata["success"])
                    h5_success = bool(np.asarray(group["success"], dtype=np.bool_)[-1])
                    if not source_success or not h5_success:
                        raise Phase2B6ContractError("selected official source is not successful")
                    terminal_agreements += int(source_success is h5_success)
                    action_hash = _array_sha256(actions, dtype=np.dtype(np.float32))
                    reset_kwargs_hash = stable_reset_identity(
                        cast(Mapping[str, object], metadata["reset_kwargs"])
                    )
                    reset_hash = source_reset_identity(
                        task_id=task_id,
                        source_episode_id=episode_id,
                        reset_kwargs_sha256=reset_kwargs_hash,
                    )
                    initial_hash = _h5_state_sha256(cast(h5py.Group, group["env_states"]), 0)
                    action_hashes.append(action_hash)
                    reset_hashes.append(reset_hash)
                    lengths.append(len(actions))
                    nonfinite_values += int(cast(int, action_report["nonfinite_values"]))
                    bound_violations += int(cast(int, action_report["bound_violation_values"]))
                    source_identity = source_trajectory_identity(
                        spec=spec,
                        task_id=task_id,
                        source_episode_id=episode_id,
                        reset_identity=reset_hash,
                        action_sha256=action_hash,
                    )
                    inventory.append(
                        {
                            "task_id": task_id,
                            "skill_family": CANDIDATE_TASKS[task_id].skill_family,
                            "source_episode_id": episode_id,
                            "episode_seed": int(metadata["episode_seed"]),
                            "source_success": source_success,
                            "reset_kwargs": metadata["reset_kwargs"],
                            "reset_kwargs_sha256": reset_kwargs_hash,
                            "reset_identity": reset_hash,
                            "initial_state_sha256": initial_hash,
                            "action_sha256": action_hash,
                            "transition_count": len(actions),
                            "source_trajectory_identity": source_identity,
                        }
                    )
                except (KeyError, TypeError, ValueError, Phase2B6ContractError) as error:
                    malformed.append(
                        {
                            "source_episode_id": episode_id,
                            "error_type": type(error).__name__,
                            "message": str(error),
                        }
                    )
        task_expected = cast(Mapping[str, object], task_specs[task_id])
        report = {
            "task_id": task_id,
            "source_h5_sha256": "sha256:" + sha256_file(h5_path),
            "source_json_sha256": "sha256:" + sha256_file(json_path),
            "trajectory_count": len(by_id),
            "transition_count": sum(lengths),
            "successful_count": sum(row["success"] is True for row in episodes),
            "terminal_metadata_h5_agreements": terminal_agreements,
            "unique_source_episode_ids": len(by_id),
            "unique_reset_identities": len(set(reset_hashes)),
            "unique_action_hashes": len(set(action_hashes)),
            "duplicate_action_hash_count": len(action_hashes) - len(set(action_hashes)),
            "malformed_or_missing": malformed,
            "malformed_count": len(malformed),
            "nonfinite_action_values": nonfinite_values,
            "action_bound_violations": bound_violations,
            "episode_length": {
                "minimum": min(lengths),
                "median": statistics.median(lengths),
                "maximum": max(lengths),
            },
            "expected_trajectory_count": task_expected["expected_episodes"],
            "expected_transition_count": task_expected["expected_transitions"],
            "passed": (
                len(by_id) == task_expected["expected_episodes"]
                and sum(lengths) == task_expected["expected_transitions"]
                and terminal_agreements == len(by_id)
                and len(set(reset_hashes)) == len(by_id)
                and len(set(action_hashes)) == len(by_id)
                and not malformed
                and nonfinite_values == 0
                and bound_violations == 0
            ),
        }
        task_reports.append(report)
    inventory.sort(
        key=lambda row: (TASK_IDS.index(str(row["task_id"])), int(row["source_episode_id"]))
    )
    inventory_digest = canonical_json_sha256({"episodes": inventory})
    inventory_path = production_root / "work" / "source_episode_inventory.json"
    _write_json(
        inventory_path,
        {
            "schema_version": "langmani-v2-phase2b6-source-episode-inventory-v0",
            "episode_count": len(inventory),
            "transition_count": sum(int(row["transition_count"]) for row in inventory),
            "episodes_digest": inventory_digest,
            "episodes": inventory,
        },
    )
    payload = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-source-schema-result-v0",
            "created_at_utc": _timestamp(),
            "source_inventory_relative_path": inventory_path.relative_to(
                production_root
            ).as_posix(),
            "source_inventory_sha256": "sha256:" + sha256_file(inventory_path),
            "source_inventory_digest": inventory_digest,
            "episode_count": len(inventory),
            "transition_count": sum(int(row["transition_count"]) for row in inventory),
            "tasks": task_reports,
            "malformed_trajectory_count": sum(
                int(report["malformed_count"]) for report in task_reports
            ),
            "passed": (
                len(inventory) == EXPECTED_EPISODES
                and sum(int(row["transition_count"]) for row in inventory) == EXPECTED_TRANSITIONS
                and all(report["passed"] is True for report in task_reports)
            ),
        }
    )
    return payload, inventory


def _leaf_lengths(group: h5py.Group) -> tuple[int, ...]:
    values: list[int] = []
    for item in group.values():
        if isinstance(item, h5py.Dataset):
            if not item.shape:
                raise Phase2B6RuntimeError(f"state dataset {item.name} is scalar")
            values.append(int(item.shape[0]))
        else:
            values.extend(_leaf_lengths(item))
    return tuple(values)


def prepare_production(
    *,
    repo_root: Path,
    spec_path: Path,
    phase2b5_artifact_root: Path,
    source_root: Path,
    production_root: Path,
    archive_root: Path,
    restore_root: Path,
    evidence_root: Path,
    network_turbo_sourced: bool,
) -> dict[str, object]:
    """Execute the complete no-simulator preflight and freeze identities/splits."""

    spec = load_production_spec(spec_path)
    lock = freeze_production_spec(spec_path=spec_path, production_root=production_root)
    repository = repository_environment_audit(
        repo_root=repo_root,
        source_root=source_root,
        production_root=production_root,
        archive_root=archive_root,
        restore_root=restore_root,
        spec=spec,
        network_turbo_sourced=network_turbo_sourced,
    )
    phase2b5 = verify_phase2b5_package(artifact_root=phase2b5_artifact_root, spec=spec)
    source_integrity = verify_source_bytes(source_root=source_root, spec=spec)
    if (
        repository["passed"] is not True
        or phase2b5["passed"] is not True
        or source_integrity["passed"] is not True
    ):
        raise Phase2B6RuntimeError("Phase 2B.6 preflight hard stop")
    source_schema, inventory = validate_source_schema(
        source_root=source_root,
        production_root=production_root,
        spec=spec,
    )
    if source_schema["passed"] is not True:
        raise Phase2B6RuntimeError("source schema hard stop")
    language = build_language_template_manifest(spec)
    splits = build_primary_split_manifest(spec=spec, episodes=inventory)
    folds = build_cross_skill_folds(spec=spec, split_manifest=splits)
    padding = build_padding_audit(cast(list[dict[str, object]], splits["assignments"]))
    balance = build_task_balance_manifest(cast(list[dict[str, object]], splits["assignments"]))
    observation_action = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-observation-action-contract-v0",
            "observation": spec["observation"],
            "camera": spec["camera"],
            "action": spec["action"],
            "pre_action_frame_count_equals_action_count": True,
            "post_terminal_frame_in_policy_dataset": False,
            "fabricated_terminal_action": False,
            "privileged_fields_excluded": True,
            "optimizer_steps": 0,
            "passed": True,
        }
    )
    artifacts = {
        "repository_environment_audit.json": repository,
        "phase2b5_source_package_verification.json": phase2b5,
        "frozen_production_specification.json": fingerprinted(
            {
                "schema_version": "langmani-v2-phase2b6-frozen-specification-evidence-v0",
                "specification_sha256": spec["_spec_sha256"],
                "specification_lock": lock,
                "production_run_id": spec["production_run_id"],
                "passed": True,
            }
        ),
        "source_integrity_audit.json": source_integrity,
        "source_schema_result.json": source_schema,
        "observation_action_contract.json": observation_action,
        "language_template_manifest.json": language,
        "primary_split_manifest.json": splits,
        "cross_skill_fold_manifests.json": folds,
        "padding_audit.json": padding,
        "task_balance_manifest.json": balance,
    }
    for name, payload in artifacts.items():
        _write_json(evidence_root / name, payload)
    work = production_root / "work"
    _write_json(work / "primary_split_manifest.json", splits)
    _write_json(work / "language_template_manifest.json", language)
    _write_json(work / "cross_skill_fold_manifests.json", folds)
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-preflight-complete-v0",
            "production_run_id": spec["production_run_id"],
            "specification_sha256": spec["_spec_sha256"],
            "repository_audit_fingerprint": repository["fingerprint"],
            "phase2b5_verification_fingerprint": phase2b5["fingerprint"],
            "source_integrity_fingerprint": source_integrity["fingerprint"],
            "source_schema_fingerprint": source_schema["fingerprint"],
            "split_manifest_fingerprint": splits["fingerprint"],
            "episode_count": EXPECTED_EPISODES,
            "transition_count": EXPECTED_TRANSITIONS,
            "training_started": False,
            "optimizer_steps": 0,
            "passed": True,
        }
    )


def _load_inventory(production_root: Path) -> list[dict[str, object]]:
    document = _read_json(production_root / "work" / "source_episode_inventory.json")
    rows = document.get("episodes")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise Phase2B6RuntimeError("source episode inventory is malformed")
    if len(rows) != EXPECTED_EPISODES:
        raise Phase2B6RuntimeError("source episode inventory count changed")
    return cast(list[dict[str, object]], rows)


def _load_assignments(production_root: Path) -> list[dict[str, object]]:
    document = _read_json(production_root / "work" / "primary_split_manifest.json")
    rows = document.get("assignments")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise Phase2B6RuntimeError("primary split assignments are malformed")
    return cast(list[dict[str, object]], rows)


def _environment_kwargs(document: Mapping[str, object]) -> dict[str, object]:
    env_info = cast(Mapping[str, object], document["env_info"])
    kwargs = dict(cast(Mapping[str, object], env_info["env_kwargs"]))
    kwargs.update(
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
    return kwargs


def _sensor_parameter_report(base: Any) -> dict[str, object]:
    parameters = base.get_sensor_params()
    camera = parameters.get("base_camera")
    if not isinstance(camera, Mapping):
        raise Phase2B6RuntimeError("base_camera sensor parameters are unavailable")
    result: dict[str, object] = {}
    for key, value in camera.items():
        array = _to_numpy(value)
        if array.ndim > 0 and array.shape[0] == 1:
            array = array[0]
        if np.issubdtype(array.dtype, np.number):
            result[str(key)] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "value": array.tolist(),
            }
    sensor_config = getattr(base, "_sensor_configs", None)
    base_config = None
    if isinstance(sensor_config, Mapping):
        base_config = sensor_config.get("base_camera")
    for name in ("near", "far", "width", "height", "fov", "fovy"):
        value = getattr(base_config, name, None)
        if isinstance(value, (float, int)):
            result[f"config_{name}"] = value
    return result


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


def _replay_one(
    *,
    env: Any,
    base: Any,
    task_id: str,
    metadata: Mapping[str, object],
    group: h5py.Group,
    capture_arrays: bool,
) -> tuple[dict[str, object], dict[str, np.ndarray] | None]:
    actions = np.asarray(group["actions"], dtype=np.float32)
    action_report = validate_action_array(actions)
    invalid_rows = int(
        np.count_nonzero(
            np.any(~np.isfinite(actions), axis=1)
            | np.any(actions < PANDA_ACTION_LOW, axis=1)
            | np.any(actions > PANDA_ACTION_HIGH, axis=1)
        )
    )
    source_initial = _h5_state_at(cast(h5py.Group, group["env_states"]), 0)
    source_terminal = _h5_state_at(cast(h5py.Group, group["env_states"]), len(actions))
    started = time.perf_counter()
    (
        semantic_reset_error,
        first_state_anchor_used,
        first_state_anchor_error,
    ) = _prepare_replay_initial_state(
        env=env,
        base=base,
        reset_kwargs=cast(Mapping[str, Any], metadata["reset_kwargs"]),
        source_initial=source_initial,
    )
    rgb = np.empty((len(actions), *IMAGE_SHAPE), dtype=np.uint8) if capture_arrays else None
    states = np.empty((len(actions), 9), dtype=np.float32) if capture_arrays else None
    timestamps = (
        np.arange(len(actions), dtype=np.float64) / CONTROL_FREQUENCY_HZ if capture_arrays else None
    )
    replay_success = False
    replayed_actions = 0
    terminated_count = 0
    truncated_count = 0
    for frame_index, action in enumerate(actions):
        observation = base.get_obs()
        policy_rgb = _extract_rgb(observation)
        policy_state = extract_panda_policy_state_v0(base.agent.robot)
        if capture_arrays:
            assert rgb is not None and states is not None
            rgb[frame_index] = policy_rgb
            states[frame_index] = policy_state
        _, _, terminated, truncated, info = env.step(action)
        replayed_actions += 1
        replay_success = _scalar_bool(info["success"])
        terminated_count += int(_scalar_bool(terminated))
        truncated_count += int(_scalar_bool(truncated))
    terminal_rgb = _extract_rgb(base.get_obs())
    replay_terminal_object_poses = _task_object_raw_poses(base, task_id)
    terminal_state_error = _state_error(source_terminal, base.get_state_dict())
    base.set_state_dict(source_terminal)
    source_terminal_object_poses = _task_object_raw_poses(base, task_id)
    terminal_object_pose_error = _task_object_pose_error(
        source_terminal_object_poses, replay_terminal_object_poses
    )
    source_success = bool(metadata["success"])
    record = {
        "source_success": source_success,
        "replay_success": replay_success,
        "categorical_outcome_agreement": source_success is replay_success,
        "source_action_count": len(actions),
        "replayed_action_count": replayed_actions,
        "policy_frame_count": len(actions) if capture_arrays else 0,
        "terminal_diagnostic_frame_count": 1,
        "action_frame_alignment": replayed_actions == len(actions),
        "invalid_action_count": invalid_rows,
        "nonfinite_value_count": int(cast(int, action_report["nonfinite_values"])),
        "simulator_error_count": 0,
        "terminated_signal_count": terminated_count,
        "truncated_signal_count": truncated_count,
        "source_action_sha256": _array_sha256(actions, dtype=np.dtype(np.float32)),
        "semantic_reset_state_error": semantic_reset_error,
        "first_state_anchor_used": first_state_anchor_used,
        "first_state_anchor_error": first_state_anchor_error,
        "terminal_state_error": terminal_state_error,
        "terminal_object_pose_error": terminal_object_pose_error,
        "terminal_rgb_sha256": _array_sha256(terminal_rgb),
        "wall_clock_seconds": time.perf_counter() - started,
        "success_predicate_modified": False,
        "action_clipping_or_projection": False,
        "planner_or_expert_invoked": False,
        "fabricated_terminal_action": False,
        "passed": (
            source_success
            and replay_success
            and replayed_actions == len(actions)
            and invalid_rows == 0
            and action_report["nonfinite_values"] == 0
        ),
    }
    arrays = (
        {
            "rgb": cast(np.ndarray, rgb),
            "state": cast(np.ndarray, states),
            "action": actions,
            "timestamp": cast(np.ndarray, timestamps),
            "terminal_rgb": terminal_rgb,
        }
        if capture_arrays
        else None
    )
    return record, arrays


def run_visual_shift_pilot(
    *,
    source_root: Path,
    production_root: Path,
    evidence_root: Path,
    spec_path: Path,
) -> dict[str, object]:
    """Physically replay one visual-split episode per task before full production."""

    try:
        import gymnasium as gym
        import mani_skill.envs  # type: ignore[import-untyped]  # noqa: F401
    except (ImportError, OSError, RuntimeError) as error:
        raise Phase2B6RuntimeError("visual-shift pilot requires ManiSkill") from error
    spec = load_production_spec(spec_path)
    visual = cast(Mapping[str, object], spec["visual_shift"])
    assignments = _load_assignments(production_root)
    task_reports: list[dict[str, object]] = []
    for task_id in TASK_IDS:
        selected = next(
            row
            for row in assignments
            if row["task_id"] == task_id and row["primary_split"] == "test_visual_shift"
        )
        h5_path, json_path = _selected_paths(source_root, task_id)
        document = _read_json(json_path)
        episodes = validate_metadata_document(document, expected_task_id=task_id)
        metadata = {int(row["episode_id"]): row for row in episodes}[
            int(selected["source_episode_id"])
        ]
        env = gym.make(task_id, **_environment_kwargs(document))
        base: Any = env.unwrapped
        try:
            with h5py.File(h5_path, "r") as trajectories:
                record, arrays = _replay_one(
                    env=env,
                    base=base,
                    task_id=task_id,
                    metadata=metadata,
                    group=cast(
                        h5py.Group,
                        trajectories[f"traj_{int(selected['source_episode_id'])}"],
                    ),
                    capture_arrays=True,
                )
            if arrays is None:
                raise Phase2B6RuntimeError("visual pilot did not capture arrays")
            shifted = apply_visual_shift(
                arrays["rgb"],
                exposure_multiplier=float(visual["exposure_multiplier"]),
                rgb_channel_multipliers=cast(Sequence[float], visual["rgb_channel_multipliers"]),
            )
            changed = int(np.count_nonzero(shifted != arrays["rgb"]))
            task_reports.append(
                {
                    "task_id": task_id,
                    "source_episode_id": selected["source_episode_id"],
                    "source_trajectory_identity": selected["source_trajectory_identity"],
                    "derived_episode_identity": selected["derived_episode_identity"],
                    "primary_split": selected["primary_split"],
                    "physics_replay": record,
                    "original_rgb_sha256": _array_sha256(arrays["rgb"]),
                    "shifted_rgb_sha256": _array_sha256(shifted),
                    "changed_channel_values": changed,
                    "action_sha256": _array_sha256(arrays["action"], dtype=np.dtype(np.float32)),
                    "state_sha256": _array_sha256(arrays["state"], dtype=np.dtype(np.float32)),
                    "post_render_only": True,
                    "physics_input_changed": False,
                    "categorical_physical_outcome_preserved": record["passed"],
                    "passed": record["passed"] is True and changed > 0,
                }
            )
        finally:
            env.close()
    report = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-visual-shift-pilot-v0",
            "created_at_utc": _timestamp(),
            "visual_shift_identity": visual["identity"],
            "appearance_contract": visual,
            "episode_count": len(task_reports),
            "tasks": task_reports,
            "physics_configuration_changed": False,
            "action_sequence_changed": False,
            "categorical_outcome_agreement_rate": (
                sum(row["categorical_physical_outcome_preserved"] is True for row in task_reports)
                / len(task_reports)
            ),
            "passed": all(row["passed"] is True for row in task_reports),
        }
    )
    _write_json(evidence_root / "visual_shift_pilot.json", report)
    if report["passed"] is not True:
        raise Phase2B6RuntimeError("visual-shift pilot hard stop")
    return report


def _save_episode_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise Phase2B6RuntimeError(f"episode output already exists without progress row: {path}")
    staging = path.parent / f".{path.name}.partial.npz"
    if staging.exists():
        raise Phase2B6RuntimeError(f"stale episode staging file exists: {staging}")
    try:
        np.savez_compressed(
            staging,
            rgb=arrays["rgb"],
            state=arrays["state"],
            action=arrays["action"],
            timestamp=arrays["timestamp"],
        )
        with staging.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()


def _contact_sheet(path: Path, frames: Sequence[np.ndarray]) -> None:
    try:
        from PIL import Image
    except ImportError as error:
        raise Phase2B6RuntimeError("contact sheets require Pillow") from error
    if len(frames) != 4:
        raise Phase2B6RuntimeError("contact sheet requires four frames")
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    canvas[:256, :256] = frames[0]
    canvas[:256, 256:] = frames[1]
    canvas[256:, :256] = frames[2]
    canvas[256:, 256:] = frames[3]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="RGB").save(path, format="PNG", optimize=True)


def _contact_sheet_episode_ids(
    assignments: Sequence[Mapping[str, object]], task_id: str
) -> set[int]:
    rows = sorted(
        [row for row in assignments if row["task_id"] == task_id],
        key=lambda row: (int(row["transition_count"]), int(row["source_episode_id"])),
    )
    indices = {0, len(rows) // 4, len(rows) // 2, 3 * len(rows) // 4, len(rows) - 1}
    return {int(rows[index]["source_episode_id"]) for index in indices}


def _mark_full_production_started(
    *, repo_root: Path, spec_path: Path, production_root: Path
) -> dict[str, object]:
    spec = load_production_spec(spec_path)
    marker = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-production-start-v0",
            "started_at_utc": _timestamp(),
            "production_run_id": spec["production_run_id"],
            "specification_sha256": spec["_spec_sha256"],
            "producer_commit": _git(repo_root, "rev-parse", "HEAD"),
            "producer_worktree_clean": not _git(repo_root, "status", "--porcelain=v1"),
            "student_policy_training_started": False,
            "optimizer_steps": 0,
        }
    )
    path = production_root / "run" / "full_production_started.json"
    if path.exists():
        existing = _read_json(path)
        stable_keys = (
            "production_run_id",
            "specification_sha256",
            "producer_commit",
            "student_policy_training_started",
            "optimizer_steps",
        )
        if any(existing.get(key) != marker.get(key) for key in stable_keys):
            raise Phase2B6RuntimeError("existing production-start marker differs")
        return existing
    _write_json(path, marker)
    return marker


def run_full_replay_production(
    *,
    repo_root: Path,
    source_root: Path,
    production_root: Path,
    evidence_root: Path,
    spec_path: Path,
) -> dict[str, object]:
    """Replay all 3,000 official trajectories and save pre-action policy arrays."""

    try:
        import gymnasium as gym
        import mani_skill.envs  # type: ignore[import-untyped]  # noqa: F401
    except (ImportError, OSError, RuntimeError) as error:
        raise Phase2B6RuntimeError("full production requires ManiSkill") from error
    marker = _mark_full_production_started(
        repo_root=repo_root,
        spec_path=spec_path,
        production_root=production_root,
    )
    assignments = _load_assignments(production_root)
    by_key = {(str(row["task_id"]), int(row["source_episode_id"])): row for row in assignments}
    progress_path = production_root / "work" / "replay_records.jsonl"
    progress = _read_jsonl(progress_path)
    completed: dict[str, dict[str, Any]] = {}
    for row in progress:
        identity = str(row.get("derived_episode_identity"))
        if identity in completed:
            raise Phase2B6RuntimeError("duplicate episode identity in replay progress")
        completed[identity] = row
    failed_rows = [row for row in progress if row.get("passed") is not True]
    if failed_rows:
        raise Phase2B6RuntimeError("existing failed replay record triggers hard stop")
    camera_reports: dict[str, object] = {}
    total_started = time.perf_counter()
    for task_id in TASK_IDS:
        h5_path, json_path = _selected_paths(source_root, task_id)
        document = _read_json(json_path)
        episodes = validate_metadata_document(document, expected_task_id=task_id)
        env = gym.make(task_id, **_environment_kwargs(document))
        base: Any = env.unwrapped
        if str(base.control_mode) != CONTROL_MODE or int(base.control_freq) != 20:
            env.close()
            raise Phase2B6RuntimeError("production environment action timing changed")
        env.reset(**cast(Mapping[str, Any], episodes[0]["reset_kwargs"]))
        camera_reports[task_id] = _sensor_parameter_report(base)
        contact_episode_ids = _contact_sheet_episode_ids(assignments, task_id)
        try:
            with h5py.File(h5_path, "r") as trajectories:
                for source_metadata in sorted(episodes, key=lambda row: int(row["episode_id"])):
                    source_episode_id = int(source_metadata["episode_id"])
                    assignment = by_key[(task_id, source_episode_id)]
                    derived_identity = str(assignment["derived_episode_identity"])
                    existing = completed.get(derived_identity)
                    if existing is not None:
                        output_path = production_root / str(existing["raw_relative_path"])
                        if (
                            not output_path.is_file()
                            or "sha256:" + sha256_file(output_path) != existing["raw_sha256"]
                        ):
                            raise Phase2B6RuntimeError(
                                f"completed episode output changed: {derived_identity}"
                            )
                        continue
                    output_path = (
                        production_root
                        / "work"
                        / "raw"
                        / _task_slug(task_id)
                        / f"episode_{source_episode_id:06d}.npz"
                    )
                    try:
                        record, arrays = _replay_one(
                            env=env,
                            base=base,
                            task_id=task_id,
                            metadata=source_metadata,
                            group=cast(h5py.Group, trajectories[f"traj_{source_episode_id}"]),
                            capture_arrays=True,
                        )
                        if arrays is None:
                            raise Phase2B6RuntimeError("production replay did not capture arrays")
                        if record["passed"] is not True:
                            raise Phase2B6RuntimeError("episode replay gate failed")
                        _save_episode_npz(output_path, arrays)
                        raw_hash = "sha256:" + sha256_file(output_path)
                        if source_episode_id in contact_episode_ids:
                            rgb = arrays["rgb"]
                            contact_path = (
                                production_root
                                / "primary"
                                / "contact_sheets"
                                / f"{task_id}_episode_{source_episode_id:06d}.png"
                            )
                            _contact_sheet(
                                contact_path,
                                (
                                    rgb[0],
                                    rgb[len(rgb) // 2],
                                    rgb[-1],
                                    arrays["terminal_rgb"],
                                ),
                            )
                        progress_row: dict[str, object] = {
                            "attempt_index": 0,
                            "retry": False,
                            "retry_reason": None,
                            "task_id": task_id,
                            "skill_family": assignment["skill_family"],
                            "source_episode_id": source_episode_id,
                            "source_trajectory_identity": assignment["source_trajectory_identity"],
                            "derived_episode_identity": derived_identity,
                            "reset_identity": assignment["reset_identity"],
                            "instruction_template_id": assignment["instruction_template_id"],
                            "instruction": assignment["instruction"],
                            "primary_split": assignment["primary_split"],
                            "raw_relative_path": output_path.relative_to(
                                production_root
                            ).as_posix(),
                            "raw_sha256": raw_hash,
                            "rgb_sha256": _array_sha256(arrays["rgb"]),
                            "state_sha256": _array_sha256(
                                arrays["state"], dtype=np.dtype(np.float32)
                            ),
                            "derived_action_sha256": _array_sha256(
                                arrays["action"], dtype=np.dtype(np.float32)
                            ),
                            "timestamp_sha256": _array_sha256(
                                arrays["timestamp"], dtype=np.dtype(np.float64)
                            ),
                            "first_frame_semantics": "pre-action observation for action[0]",
                            "last_policy_frame_semantics": (
                                "pre-action observation for the final recorded action"
                            ),
                            "post_terminal_frame_policy_dataset": False,
                            "error": None,
                            **record,
                        }
                    except Exception as error:
                        progress_row = {
                            "attempt_index": 0,
                            "retry": False,
                            "retry_reason": None,
                            "task_id": task_id,
                            "source_episode_id": source_episode_id,
                            "source_trajectory_identity": assignment["source_trajectory_identity"],
                            "derived_episode_identity": derived_identity,
                            "primary_split": assignment["primary_split"],
                            "raw_relative_path": output_path.relative_to(
                                production_root
                            ).as_posix(),
                            "error": {
                                "type": type(error).__name__,
                                "message": str(error),
                            },
                            "source_success": bool(source_metadata["success"]),
                            "replay_success": False,
                            "categorical_outcome_agreement": False,
                            "action_frame_alignment": False,
                            "invalid_action_count": 0,
                            "nonfinite_value_count": 0,
                            "simulator_error_count": 1,
                            "passed": False,
                        }
                    _append_jsonl(progress_path, progress_row)
                    completed[derived_identity] = cast(dict[str, Any], progress_row)
                    if progress_row["passed"] is not True:
                        rejected = fingerprinted(
                            {
                                "schema_version": ("langmani-v2-phase2b6-rejected-production-v0"),
                                "episode_count": 1,
                                "episodes": [progress_row],
                                "failure_training_corpus_created": False,
                                "passed": False,
                            }
                        )
                        _write_json(evidence_root / "rejected_production_manifest.json", rejected)
                        raise Phase2B6RuntimeError(
                            f"full production hard stop at {task_id}/{source_episode_id}"
                        )
                    completed_count = len(completed)
                    if completed_count % 25 == 0:
                        print(
                            json.dumps(
                                {
                                    "phase": "full_replay",
                                    "completed_episodes": completed_count,
                                    "total_episodes": EXPECTED_EPISODES,
                                    "elapsed_seconds": time.perf_counter() - total_started,
                                    "latest_task": task_id,
                                    "latest_source_episode_id": source_episode_id,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
        finally:
            env.close()
    rows = _read_jsonl(progress_path)
    if len(rows) != EXPECTED_EPISODES:
        raise Phase2B6RuntimeError("full production episode count is incomplete")
    task_summaries: dict[str, object] = {}
    for task_id in TASK_IDS:
        task_rows = [row for row in rows if row.get("task_id") == task_id]
        gate = summarize_replay_gate(task_rows)
        task_summary = fingerprinted(
            {
                "schema_version": "langmani-v2-phase2b6-task-replay-summary-v0",
                "task_id": task_id,
                "skill_family": CANDIDATE_TASKS[task_id].skill_family,
                "episode_count": len(task_rows),
                "frame_count": sum(int(row["source_action_count"]) for row in task_rows),
                "retry_count": sum(row.get("retry") is True for row in task_rows),
                "gate": gate,
                "camera_parameters": camera_reports[task_id],
                "passed": gate["passed"],
            }
        )
        task_summaries[task_id] = task_summary
        _write_json(
            evidence_root / f"replay_summary_{_task_slug(task_id)}.json",
            task_summary,
        )
    all_gate = summarize_replay_gate(rows)
    replay_inventory_path = production_root / "primary" / "metadata" / "replay_inventory.json"
    _write_json(
        replay_inventory_path,
        {
            "schema_version": "langmani-v2-phase2b6-replay-inventory-v0",
            "production_run_id": marker["production_run_id"],
            "episode_count": len(rows),
            "frame_count": sum(int(row["source_action_count"]) for row in rows),
            "episodes_digest": canonical_json_sha256({"episodes": rows}),
            "episodes": rows,
        },
    )
    rejected = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-rejected-production-v0",
            "episode_count": 0,
            "episodes": [],
            "failure_training_corpus_created": False,
            "passed": True,
        }
    )
    manifest = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-production-run-manifest-v0",
            "created_at_utc": _timestamp(),
            "production_start": marker,
            "maniskill_version": MANISKILL_VERSION,
            "maniskill_source_revision": MANISKILL_SOURCE_REVISION,
            "source_revision": HF_REVISION,
            "recorded_official_actions_only": True,
            "episode_count": len(rows),
            "frame_count": sum(int(row["source_action_count"]) for row in rows),
            "task_summaries": {
                task_id: {
                    "episode_count": cast(Mapping[str, object], summary)["episode_count"],
                    "frame_count": cast(Mapping[str, object], summary)["frame_count"],
                    "fingerprint": cast(Mapping[str, object], summary)["fingerprint"],
                    "passed": cast(Mapping[str, object], summary)["passed"],
                }
                for task_id, summary in task_summaries.items()
            },
            "replay_inventory_relative_path": replay_inventory_path.relative_to(
                production_root
            ).as_posix(),
            "replay_inventory_sha256": "sha256:" + sha256_file(replay_inventory_path),
            "camera_parameters": camera_reports,
            "aggregate_gate": all_gate,
            "rejected_production_count": 0,
            "student_policy_training_started": False,
            "optimizer_steps": 0,
            "elapsed_seconds_this_invocation": time.perf_counter() - total_started,
            "passed": (
                len(rows) == EXPECTED_EPISODES
                and sum(int(row["source_action_count"]) for row in rows) == EXPECTED_TRANSITIONS
                and all_gate["passed"] is True
            ),
        }
    )
    _write_json(evidence_root / "production_run_manifest.json", manifest)
    _write_json(evidence_root / "rejected_production_manifest.json", rejected)
    if manifest["passed"] is not True:
        raise Phase2B6RuntimeError("full production aggregate hard stop")
    return manifest


class _Moments:
    def __init__(self, dimension: int) -> None:
        self.count = 0
        self.sum = np.zeros(dimension, dtype=np.float64)
        self.squared_sum = np.zeros(dimension, dtype=np.float64)
        self.minimum = np.full(dimension, np.inf, dtype=np.float64)
        self.maximum = np.full(dimension, -np.inf, dtype=np.float64)

    def add(self, values: object) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(self.sum):
            raise Phase2B6RuntimeError("statistics array has wrong shape")
        if not np.all(np.isfinite(array)):
            raise Phase2B6RuntimeError("statistics array contains nonfinite values")
        self.count += len(array)
        self.sum += array.sum(axis=0)
        self.squared_sum += np.square(array).sum(axis=0)
        self.minimum = np.minimum(self.minimum, array.min(axis=0))
        self.maximum = np.maximum(self.maximum, array.max(axis=0))

    def report(self) -> dict[str, object]:
        if self.count == 0:
            raise Phase2B6RuntimeError("cannot report empty statistics")
        mean = self.sum / self.count
        variance = np.maximum(self.squared_sum / self.count - np.square(mean), 0.0)
        std = np.sqrt(variance)
        return {
            "count": self.count,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "minimum": self.minimum.tolist(),
            "maximum": self.maximum.tolist(),
            "low_variance_dimensions": [index for index, value in enumerate(std) if value < 1e-6],
        }


def compute_dataset_statistics(
    *,
    production_root: Path,
    evidence_root: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Compute split statistics and canonical normalization from primary train only."""

    assignments = _load_assignments(production_root)
    replay_document = _read_json(production_root / "primary" / "metadata" / "replay_inventory.json")
    replay_rows = cast(list[dict[str, object]], replay_document["episodes"])
    replay_by_identity = {str(row["derived_episode_identity"]): row for row in replay_rows}
    state_moments: dict[tuple[str, str], _Moments] = defaultdict(lambda: _Moments(9))
    action_moments: dict[tuple[str, str], _Moments] = defaultdict(lambda: _Moments(8))
    pooled_train_state = _Moments(9)
    pooled_train_action = _Moments(8)
    train_image = _Moments(3)
    endpoint_counts = np.zeros(8, dtype=np.int64)
    train_action_count = 0
    gripper_values: list[np.ndarray] = []
    episode_lengths: dict[str, list[int]] = defaultdict(list)
    for index, assignment in enumerate(assignments, start=1):
        identity = str(assignment["derived_episode_identity"])
        replay = replay_by_identity[identity]
        path = production_root / str(replay["raw_relative_path"])
        with np.load(path, allow_pickle=False) as payload:
            rgb = np.asarray(payload["rgb"], dtype=np.uint8)
            state = np.asarray(payload["state"], dtype=np.float32)
            action = np.asarray(payload["action"], dtype=np.float32)
        task_id = str(assignment["task_id"])
        split = str(assignment["primary_split"])
        state_moments[(task_id, split)].add(state)
        action_moments[(task_id, split)].add(action)
        episode_lengths[task_id].append(len(action))
        if split == "train":
            pooled_train_state.add(state)
            pooled_train_action.add(action)
            pixels = rgb.reshape(-1, 3).astype(np.float64) / 255.0
            train_image.add(pixels)
            at_low = np.isclose(action, PANDA_ACTION_LOW, rtol=0.0, atol=1e-5)
            at_high = np.isclose(action, PANDA_ACTION_HIGH, rtol=0.0, atol=1e-5)
            endpoint_counts += np.count_nonzero(at_low | at_high, axis=0)
            train_action_count += len(action)
            gripper_values.append(action[:, 7].copy())
        if index % 100 == 0:
            print(
                json.dumps(
                    {
                        "phase": "statistics",
                        "completed_episodes": index,
                        "total_episodes": EXPECTED_EPISODES,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    by_task_split = {
        task_id: {
            split: {
                "state": state_moments[(task_id, split)].report(),
                "action": action_moments[(task_id, split)].report(),
            }
            for split in PRIMARY_SPLITS
        }
        for task_id in TASK_IDS
    }
    per_task_train = {
        task_id: {
            "state": state_moments[(task_id, "train")].report(),
            "action": action_moments[(task_id, "train")].report(),
        }
        for task_id in TASK_IDS
    }
    equal_task_state_mean = np.mean(
        [np.asarray(per_task_train[task]["state"]["mean"]) for task in TASK_IDS], axis=0
    )
    equal_task_action_mean = np.mean(
        [np.asarray(per_task_train[task]["action"]["mean"]) for task in TASK_IDS], axis=0
    )
    gripper = np.concatenate(gripper_values)
    stats = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-dataset-statistics-v0",
            "created_at_utc": _timestamp(),
            "all_splits_by_task": by_task_split,
            "train_only": {
                "per_task": per_task_train,
                "pooled_natural_frame_frequency": {
                    "state": pooled_train_state.report(),
                    "action": pooled_train_action.report(),
                    "image_rgb_unit_interval": train_image.report(),
                },
                "equal_task_weighted_diagnostic": {
                    "state_mean_of_task_means": equal_task_state_mean.tolist(),
                    "action_mean_of_task_means": equal_task_action_mean.tolist(),
                    "canonical": False,
                },
                "action_endpoint_rates": (
                    endpoint_counts.astype(np.float64) / train_action_count
                ).tolist(),
                "gripper_distribution": {
                    "minimum": float(gripper.min()),
                    "maximum": float(gripper.max()),
                    "mean": float(gripper.mean()),
                    "std": float(gripper.std()),
                    "negative_fraction": float(np.mean(gripper < 0)),
                    "zero_fraction": float(np.mean(gripper == 0)),
                    "positive_fraction": float(np.mean(gripper > 0)),
                    "quantiles": {
                        str(value): float(np.quantile(gripper, value))
                        for value in (0.0, 0.25, 0.5, 0.75, 1.0)
                    },
                },
            },
            "episode_length_by_task": {
                task_id: {
                    "count": len(values),
                    "minimum": min(values),
                    "median": statistics.median(values),
                    "mean": statistics.mean(values),
                    "maximum": max(values),
                    "transition_count": sum(values),
                }
                for task_id, values in episode_lengths.items()
            },
            "validation_or_test_in_train_normalization": False,
            "student_policy_training_started": False,
            "optimizer_steps": 0,
            "passed": pooled_train_state.count > 0 and pooled_train_action.count > 0,
        }
    )
    normalization = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-normalization-v0",
            "canonical_identity": (
                "LangManiOfficialMultiSkill-v1-primary-train-pooled-natural-frame-v0"
            ),
            "source_view": "primary_train_only",
            "weighting": "natural_training_frame_frequency",
            "state": pooled_train_state.report(),
            "action": pooled_train_action.report(),
            "standard_deviation_floor_for_future_consumers": 1e-8,
            "validation_episode_count": 0,
            "test_episode_count": 0,
            "equal_task_weighted_view_is_diagnostic_only": True,
            "training_started": False,
            "optimizer_steps": 0,
            "passed": True,
        }
    )
    metadata = production_root / "primary" / "metadata"
    _write_json(metadata / "dataset_statistics.json", stats)
    _write_json(metadata / "normalization.json", normalization)
    _write_json(evidence_root / "dataset_statistics.json", stats)
    _write_json(evidence_root / "normalization_manifest.json", normalization)
    return stats, normalization


__all__ = [
    "Phase2B6RuntimeError",
    "compute_dataset_statistics",
    "freeze_production_spec",
    "prepare_production",
    "repository_environment_audit",
    "run_full_replay_production",
    "run_visual_shift_pilot",
    "validate_source_schema",
    "verify_source_bytes",
]
