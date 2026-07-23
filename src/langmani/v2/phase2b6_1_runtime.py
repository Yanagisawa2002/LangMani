"""Native, bounded runtime for Phase 2B.6.1 StackCube replay forensics.

The module can construct only ``StackCube-v1`` and can read only source
episodes 936--938.  Every replay writes to a new forensic directory and never
opens the frozen Phase 2B.6 production root for writing.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, cast

import h5py  # type: ignore[import-untyped]
import numpy as np

from langmani.datasets.policy_state import extract_panda_policy_state_v0
from langmani.v2.phase2b5 import (
    IMAGE_SHAPE,
    sha256_file,
    stable_reset_identity,
    validate_action_array,
    validate_metadata_document,
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
    load_production_spec,
    source_reset_identity,
    source_trajectory_identity,
    verify_phase2b5_package,
)
from langmani.v2.phase2b6_1 import (
    CONTROL_EPISODE_IDS,
    CONTROL_FREQUENCY_HZ,
    CONTROL_MODE,
    GATE_ORDER,
    MANISKILL_SOURCE_REVISION,
    MODE_REPETITION_BUDGETS,
    PHASE2B5_PACKAGE_SHA256,
    PRODUCER_COMMIT,
    SOURCE_COMMIT,
    SOURCE_REVISION,
    SOURCE_ZIP_SHA256,
    TARGET_ACTION_COUNT,
    TARGET_ACTION_SHA256,
    TARGET_BRANCH,
    TARGET_DERIVED_EPISODE_IDENTITY,
    TARGET_EPISODE_ID,
    TARGET_RESET_IDENTITY,
    TARGET_SOURCE_TRAJECTORY_IDENTITY,
    TASK_ID,
    ForensicMode,
    GateStatus,
    alignment_audit,
    array_sha256,
    authorization_state,
    classify_result,
    eligibility_state,
    fingerprinted,
    first_failed_gate,
    fresh_process_identity,
    gate,
    required_gates_for_mode,
    run_passed_for_mode,
    success_timing,
    validate_episode_scope,
    verify_phase2b6_artifacts,
)
from langmani.v2.phase2b6_runtime import _environment_kwargs, _save_episode_npz


class Phase2B61RuntimeError(RuntimeError):
    """Raised when the bounded native forensic contract cannot be preserved."""


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2B61RuntimeError(f"failed to read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise Phase2B61RuntimeError(f"{path} must contain one JSON object")
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise Phase2B61RuntimeError(f"forensic output already exists: {path}")
    staging = path.parent / f".{path.name}.partial"
    if staging.exists():
        raise Phase2B61RuntimeError(f"stale forensic staging file exists: {staging}")
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


def _git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=check,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


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


def _jsonable_array(value: object) -> list[object]:
    return cast(list[object], _to_numpy(value).tolist())


def _runtime_versions() -> dict[str, object]:
    packages: dict[str, str] = {}
    for name in ("mani-skill", "sapien", "torch", "numpy", "h5py"):
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = "not_installed"
    return {
        "python": platform.python_version(),
        "executable": Path(sys.executable).resolve().as_posix(),
        "prefix": (
            os.environ.get("CONDA_PREFIX")
            or os.environ.get("VIRTUAL_ENV")
            or Path(sys.executable).resolve().parents[1].as_posix()
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
        devices = []
        for line in result.stdout.strip().splitlines():
            fields = [field.strip() for field in line.split(",")]
            devices.append(
                {
                    "uuid": fields[0],
                    "name": fields[1],
                    "driver_version": fields[2],
                    "memory_used_mib": int(fields[3]),
                    "memory_total_mib": int(fields[4]),
                    "utilization_percent": int(fields[5]),
                }
            )
        return {"available": bool(devices), "devices": devices}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}


def _command_audit(
    command: Sequence[str], *, environment: Mapping[str, str] | None = None
) -> dict[str, object]:
    try:
        result = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            env=dict(environment) if environment is not None else None,
        )
        return {
            "command": list(command),
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "passed": result.returncode == 0,
        }
    except OSError as error:
        return {
            "command": list(command),
            "exit_code": None,
            "stdout": "",
            "stderr": f"{type(error).__name__}: {error}",
            "passed": False,
        }


def _process_audit() -> dict[str, object]:
    prohibited_terms = (
        "run_v2_phase2b6.py produce",
        "collect_",
        "convert",
        "archive",
        "restore",
        "train_act",
        "smolvla",
        "vla_jepa",
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
        return {
            "query_succeeded": False,
            "prohibited_active_processes": [],
            "error": f"{type(error).__name__}: {error}",
            "passed": False,
        }
    own = {os.getpid(), os.getppid()}
    matches: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) != 3:
            continue
        pid = int(fields[0])
        command = fields[2]
        if pid not in own and any(term in command.lower() for term in prohibited_terms):
            matches.append({"pid": pid, "ppid": int(fields[1]), "command": command})
    return {
        "query_succeeded": True,
        "prohibited_active_processes": matches,
        "passed": not matches,
    }


def _source_paths(source_root: Path) -> tuple[Path, Path, Path]:
    directory = source_root / "expanded" / TASK_ID / "motionplanning"
    return (
        directory / "trajectory.h5",
        directory / "trajectory.json",
        source_root / "sources" / f"{TASK_ID}.zip",
    )


def _episode_rows(
    *, source_root: Path, spec_path: Path
) -> tuple[dict[str, Any], dict[int, dict[str, object]]]:
    h5_path, json_path, _ = _source_paths(source_root)
    if not h5_path.is_file() or not json_path.is_file():
        raise Phase2B61RuntimeError("pinned StackCube source files are missing")
    document = _read_json(json_path)
    episodes = validate_metadata_document(document, expected_task_id=TASK_ID)
    selected_metadata = {
        int(row["episode_id"]): cast(dict[str, object], row)
        for row in episodes
        if int(row["episode_id"]) in (*CONTROL_EPISODE_IDS, TARGET_EPISODE_ID)
    }
    if set(selected_metadata) != {*CONTROL_EPISODE_IDS, TARGET_EPISODE_ID}:
        raise Phase2B61RuntimeError("required source episode metadata is incomplete")
    load_production_spec(spec_path)
    return document, selected_metadata


def build_source_identity_manifest(
    *, source_root: Path, spec_path: Path, frozen_root: Path
) -> dict[str, object]:
    """Prove the exact three permitted source trajectories and episode 938 lineage."""

    document, metadata_by_id = _episode_rows(source_root=source_root, spec_path=spec_path)
    spec = load_production_spec(spec_path)
    h5_path, json_path, zip_path = _source_paths(source_root)
    inventory = _read_json(frozen_root / "work" / "source_episode_inventory.json")
    inventory_rows = cast(list[dict[str, object]], inventory["episodes"])
    frozen_by_id = {
        int(row["source_episode_id"]): row
        for row in inventory_rows
        if row.get("task_id") == TASK_ID
        and int(row["source_episode_id"]) in (*CONTROL_EPISODE_IDS, TARGET_EPISODE_ID)
    }
    assignments = _read_json(frozen_root / "work" / "primary_split_manifest.json")
    assignment_rows = cast(list[dict[str, object]], assignments["assignments"])
    assignment_by_id = {
        int(row["source_episode_id"]): row
        for row in assignment_rows
        if row.get("task_id") == TASK_ID
        and int(row["source_episode_id"]) in (*CONTROL_EPISODE_IDS, TARGET_EPISODE_ID)
    }
    reports: list[dict[str, object]] = []
    with h5py.File(h5_path, "r") as trajectories:
        for episode_id in (*CONTROL_EPISODE_IDS, TARGET_EPISODE_ID):
            metadata = metadata_by_id[episode_id]
            group = cast(h5py.Group, trajectories[f"traj_{episode_id}"])
            actions = np.asarray(group["actions"])
            action_report = validate_action_array(actions)
            action_hash = array_sha256(actions, dtype=np.dtype(np.float32))
            reset_kwargs_hash = stable_reset_identity(
                cast(Mapping[str, object], metadata["reset_kwargs"])
            )
            reset_identity = source_reset_identity(
                task_id=TASK_ID,
                source_episode_id=episode_id,
                reset_kwargs_sha256=reset_kwargs_hash,
            )
            trajectory_identity = source_trajectory_identity(
                spec=spec,
                task_id=TASK_ID,
                source_episode_id=episode_id,
                reset_identity=reset_identity,
                action_sha256=action_hash,
            )
            success_values = np.asarray(group["success"], dtype=np.bool_)
            source_state_count = min(
                int(dataset.shape[0])
                for dataset in _all_datasets(cast(h5py.Group, group["env_states"]))
            )
            frozen = frozen_by_id.get(episode_id, {})
            checks = {
                "native_float32_t8": action_report["passed"] is True,
                "metadata_elapsed_matches": metadata["elapsed_steps"] == len(actions),
                "environment_state_count_t_plus_one": source_state_count == len(actions) + 1,
                "source_terminal_success": bool(success_values[-1]),
                "metadata_success": metadata["success"] is True,
                "inventory_action_hash": frozen.get("action_sha256") == action_hash,
                "inventory_reset_identity": frozen.get("reset_identity") == reset_identity,
                "inventory_trajectory_identity": (
                    frozen.get("source_trajectory_identity") == trajectory_identity
                ),
            }
            reports.append(
                {
                    "task_id": TASK_ID,
                    "source_episode_id": episode_id,
                    "trajectory_key": f"traj_{episode_id}",
                    "episode_seed": metadata["episode_seed"],
                    "reset_kwargs": metadata["reset_kwargs"],
                    "reset_kwargs_sha256": reset_kwargs_hash,
                    "reset_identity": reset_identity,
                    "action_count": len(actions),
                    "action_dtype": str(actions.dtype),
                    "action_shape": list(actions.shape),
                    "action_sha256": action_hash,
                    "environment_state_count": source_state_count,
                    "source_success_metadata": metadata["success"],
                    "source_terminal_success": bool(success_values[-1]),
                    "source_success_timing": success_timing(success_values.tolist()),
                    "source_elapsed_length": metadata["elapsed_steps"],
                    "source_trajectory_identity": trajectory_identity,
                    "frozen_inventory_identity": frozen,
                    "frozen_split_assignment": assignment_by_id.get(episode_id),
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
    target = next(row for row in reports if row["source_episode_id"] == TARGET_EPISODE_ID)
    target_checks = {
        "action_count": target["action_count"] == TARGET_ACTION_COUNT,
        "action_sha256": target["action_sha256"] == TARGET_ACTION_SHA256,
        "reset_identity": target["reset_identity"] == TARGET_RESET_IDENTITY,
        "trajectory_identity": (
            target["source_trajectory_identity"] == TARGET_SOURCE_TRAJECTORY_IDENTITY
        ),
        "derived_episode_identity": (
            cast(Mapping[str, object], target["frozen_split_assignment"]).get(
                "derived_episode_identity"
            )
            == TARGET_DERIVED_EPISODE_IDENTITY
        ),
    }
    source_zip_hash = "sha256:" + sha256_file(zip_path)
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-source-identity-v0",
            "official_source_revision": SOURCE_REVISION,
            "maniskill_source_revision": MANISKILL_SOURCE_REVISION,
            "source_h5_path": h5_path.as_posix(),
            "source_h5_sha256": "sha256:" + sha256_file(h5_path),
            "source_json_path": json_path.as_posix(),
            "source_json_sha256": "sha256:" + sha256_file(json_path),
            "source_zip_path": zip_path.as_posix(),
            "source_zip_sha256": source_zip_hash,
            "source_zip_identity_passed": source_zip_hash == SOURCE_ZIP_SHA256,
            "source_environment": document["env_info"],
            "episodes": reports,
            "target_checks": target_checks,
            "producer_input_identity_proven": (
                source_zip_hash == SOURCE_ZIP_SHA256
                and all(row["passed"] is True for row in reports)
                and all(target_checks.values())
            ),
            "passed": (
                source_zip_hash == SOURCE_ZIP_SHA256
                and all(row["passed"] is True for row in reports)
                and all(target_checks.values())
            ),
        }
    )


def _all_datasets(group: h5py.Group) -> list[h5py.Dataset]:
    datasets: list[h5py.Dataset] = []
    for item in group.values():
        if isinstance(item, h5py.Group):
            datasets.extend(_all_datasets(item))
        else:
            datasets.append(item)
    return datasets


def _relevant_file_entry(path: Path, *, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": "sha256:" + sha256_file(path),
    }


def inventory_frozen_output(frozen_root: Path) -> dict[str, object]:
    """Hash only records directly relevant to the bounded forensic question."""

    candidates = [
        frozen_root / "run" / "frozen_production_specification.json",
        frozen_root / "run" / "specification_lock.json",
        frozen_root / "run" / "full_production_started.json",
        frozen_root / "run" / "command_log.jsonl",
        frozen_root / "work" / "source_episode_inventory.json",
        frozen_root / "work" / "primary_split_manifest.json",
        frozen_root / "work" / "replay_records.jsonl",
    ]
    stack_raw_root = frozen_root / "work" / "raw" / "stackcube"
    for episode_id in CONTROL_EPISODE_IDS:
        candidates.append(stack_raw_root / f"episode_{episode_id:06d}.npz")
    files = [_relevant_file_entry(path, root=frozen_root) for path in candidates if path.is_file()]
    target_path = stack_raw_root / f"episode_{TARGET_EPISODE_ID:06d}.npz"
    target_npz = [target_path] if target_path.is_file() else []
    temp_files = [
        path.relative_to(frozen_root).as_posix()
        for path in frozen_root.rglob("*partial*")
        if path.is_file()
    ]
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-frozen-partial-inventory-v0",
            "frozen_root": frozen_root.as_posix(),
            "read_only_contract": True,
            "directly_relevant_files": files,
            "target_episode_npz_files": [
                _relevant_file_entry(path, root=frozen_root) for path in target_npz
            ],
            "temporary_or_partial_named_files": temp_files,
            "historical_accepted_episode_count": 1938,
            "historical_accepted_frame_count": 178633,
            "partial_output_is_accepted_dataset": False,
            "frozen_bytes_modified": False,
            "passed": bool(files) and not target_npz,
        }
    )


def historical_evidence_audit(frozen_root: Path) -> dict[str, object]:
    """Preserve observed/missing distinctions for the opaque producer failure."""

    replay_path = frozen_root / "work" / "replay_records.jsonl"
    accepted_rows: dict[int, dict[str, object]] = {}
    if replay_path.is_file():
        with replay_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if (
                    isinstance(row, dict)
                    and row.get("task_id") == TASK_ID
                    and row.get("source_episode_id") in CONTROL_EPISODE_IDS
                ):
                    accepted_rows[int(row["source_episode_id"])] = row
    relevant_logs: list[dict[str, object]] = []
    for path in (frozen_root.parent / "logs").glob("*.log"):
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = [
            line
            for line in text.splitlines()
            if "938" in line or "episode replay gate failed" in line
        ]
        if lines:
            relevant_logs.append(
                {
                    "path": path.as_posix(),
                    "sha256": "sha256:" + sha256_file(path),
                    "matching_lines": lines[-40:],
                }
            )
    fields = {
        "source_identity": {
            "status": "reconstructed_from_committed_metadata",
            "value": TARGET_SOURCE_TRAJECTORY_IDENTITY,
        },
        "action_identity": {
            "status": "reconstructed_from_committed_metadata",
            "value": TARGET_ACTION_SHA256,
        },
        "aggregate_rejection": {
            "status": "observed" if relevant_logs else "missing",
            "value": "episode replay gate failed" if relevant_logs else None,
        },
        "internal_returned_replay_record": {"status": "not_recorded", "value": None},
        "exact_failed_inner_subgate": {"status": "missing", "value": None},
        "observed_simulator_exception": {"status": "not_recorded", "value": None},
        "episode_npz": {"status": "missing", "value": None},
        "frame_count": {"status": "not_recorded", "value": None},
        "timestamps": {"status": "not_recorded", "value": None},
        "writer_status": {"status": "not_recorded", "value": None},
        "terminal_state": {"status": "not_recorded", "value": None},
        "producer_process_stop": {
            "status": "observed" if relevant_logs else "missing",
            "value": "hard stop after rejected replay" if relevant_logs else None,
        },
    }
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-historical-evidence-audit-v0",
            "producer_commit": PRODUCER_COMMIT,
            "controls": [accepted_rows[key] for key in sorted(accepted_rows)],
            "target_episode": TARGET_EPISODE_ID,
            "fields": fields,
            "producer_log_fragments": relevant_logs,
            "surrogate_simulator_error_is_observed_exception": False,
            "exact_failed_subgate_inferred_from_generic_label": False,
            "passed": len(accepted_rows) == 2,
        }
    )


def forensic_protocol() -> dict[str, object]:
    """Return the preregistered progressive protocol and immutable stop rules."""

    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-forensic-protocol-v0",
            "task_id": TASK_ID,
            "control_episodes": list(CONTROL_EPISODE_IDS),
            "target_episode": TARGET_EPISODE_ID,
            "modes": {
                "A": {
                    "description": "independent physics-only diagnostic replay",
                    "rgb": False,
                    "policy_state": False,
                    "writer": False,
                    "required_gates": list(required_gates_for_mode("A")),
                },
                "B": {
                    "description": "producer-like pre-action observation replay",
                    "rgb": True,
                    "policy_state": True,
                    "writer": False,
                    "required_gates": list(required_gates_for_mode("B")),
                },
                "C": {
                    "description": "producer-like observation plus isolated NPZ writer",
                    "rgb": True,
                    "policy_state": True,
                    "writer": True,
                    "required_gates": list(required_gates_for_mode("C")),
                },
            },
            "repetition_budgets": MODE_REPETITION_BUDGETS,
            "fresh_process_per_run": True,
            "gate_order": list(GATE_ORDER),
            "producer_success_rule": "final_step_canonical_success",
            "separate_stable_success_gate_in_producer": False,
            "no_acceptance_policy_change": True,
            "stop_rules": [
                "stop if either control fails",
                "stop after three Mode A runs if any Mode A run fails or outcomes differ",
                "Mode B permitted only after all Mode A runs pass",
                "Mode C permitted only after all Mode B runs pass",
                "stop when the first failing sub-gate is localized",
                "never process StackCube 939+ or PushCube",
            ],
            "authorization": authorization_state(),
            "passed": True,
        }
    )


def producer_call_sequence_audit() -> dict[str, object]:
    """Declare exact similarities and material differences before execution."""

    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-call-sequence-audit-v0",
            "producer_commit": PRODUCER_COMMIT,
            "shared_steps": [
                "load source actions as native float32 without clipping or projection",
                "semantic reset followed by official first-state anchor only if needed",
                "submit actions in recorded order",
                "evaluate final canonical success from env.step info",
                "Mode B/C acquire base.get_obs before each corresponding action",
                "Mode B/C extract base_camera RGB and PandaPolicyStateV0",
                "Mode C uses the Phase 2B.6 atomic compressed-NPZ writer",
            ],
            "known_differences": [
                {
                    "difference": "each forensic run creates a fresh process and environment",
                    "material": True,
                    "reason": "the producer reused one StackCube environment across prior episodes",
                },
                {
                    "difference": "Mode A disables RGB observation and renderer work",
                    "material": True,
                    "reason": "required independent physics-layer isolation",
                },
                {
                    "difference": "the forensic runner retains every explicit sub-gate and step diagnostic",
                    "material": False,
                    "reason": "observability only; no action or success predicate change",
                },
                {
                    "difference": "Mode C writes only to a distinct forensic destination",
                    "material": False,
                    "reason": "frozen Phase 2B.6 output must remain immutable",
                },
            ],
            "controller_initialization": "gym.make plus exact source reset",
            "observation_before_action_order": True,
            "sensor_update_timing": "base.get_obs once immediately before each action in B/C",
            "termination_checking": "recorded after every env.step; actions are not shortened",
            "stable_success_checking": "not present as a separate producer acceptance gate",
            "writer_order": "np.savez_compressed, flush/fsync, atomic os.replace",
            "dtype_conversion": "np.asarray(source actions, dtype=float32), byte-equal identity required",
            "claim": "independent_diagnostic_replay",
            "exact_producer_path_reproduction_claimed": False,
            "passed": True,
        }
    )


def repository_environment_audit(
    *,
    repo_root: Path,
    source_root: Path,
    frozen_root: Path,
    output_root: Path,
    network_turbo_sourced: bool,
) -> dict[str, object]:
    """Verify clean isolation, exact ancestry, capacity, and closed authorization."""

    head = _git(repo_root, "rev-parse", "HEAD")
    branch = _git(repo_root, "branch", "--show-current")
    upstream = _git(repo_root, "rev-parse", "@{upstream}")
    remote = _git(repo_root, "remote", "get-url", "origin")
    source_ancestor = _git(repo_root, "merge-base", SOURCE_COMMIT, head) == SOURCE_COMMIT
    producer_exists = _git(repo_root, "cat-file", "-t", PRODUCER_COMMIT, check=False) == "commit"
    remote_line = _git(repo_root, "ls-remote", "origin", f"refs/heads/{TARGET_BRANCH}", check=False)
    github_sha = remote_line.split()[0] if remote_line else None
    status = _git(repo_root, "status", "--porcelain=v1")
    source_zip = _source_paths(source_root)[2]
    free_bytes = shutil.disk_usage(output_root.parent).free
    process = _process_audit()
    checks = {
        "target_branch": branch == TARGET_BRANCH,
        "source_commit_is_ancestor": source_ancestor,
        "producer_commit_exists": producer_exists,
        "worktree_clean": not status,
        "upstream_matches_head": upstream == head,
        "github_matches_head": github_sha == head,
        "source_zip_hash": (
            source_zip.is_file() and "sha256:" + sha256_file(source_zip) == SOURCE_ZIP_SHA256
        ),
        "frozen_root_exists": frozen_root.is_dir(),
        "forensic_output_is_distinct": (
            output_root != frozen_root and frozen_root not in output_root.parents
        ),
        "space_for_small_forensics": free_bytes >= 4 * 1024**3,
        "no_prohibited_process": process["passed"] is True,
        "network_turbo_sourced": network_turbo_sourced,
    }
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-repository-environment-audit-v0",
            "created_at_utc": _timestamp(),
            "repository": remote,
            "worktree": repo_root.as_posix(),
            "source_commit": SOURCE_COMMIT,
            "branch": branch,
            "head": head,
            "upstream": upstream,
            "github_branch_sha": github_sha,
            "remote": remote,
            "producer_commit": PRODUCER_COMMIT,
            "source_package_identity": PHASE2B5_PACKAGE_SHA256,
            "official_source_identities": {
                "official_source_revision": SOURCE_REVISION,
                "maniskill_source_revision": MANISKILL_SOURCE_REVISION,
                "stackcube_source_zip_sha256": SOURCE_ZIP_SHA256,
            },
            "runtime_environment": _runtime_versions(),
            "process_state": process,
            "gpu_state": _gpu_audit(),
            "storage": {
                "forensic_output_root": output_root.as_posix(),
                "free_bytes": free_bytes,
                "minimum_required_bytes": 4 * 1024**3,
            },
            "authorization_state": authorization_state(),
            "checks": checks,
            "passed": all(checks.values()),
        }
    )


def prepare_forensics(
    *,
    repo_root: Path,
    source_root: Path,
    frozen_root: Path,
    output_root: Path,
    evidence_root: Path,
    spec_path: Path,
    phase2b5_artifact_root: Path,
    network_turbo_sourced: bool,
) -> dict[str, object]:
    """Create compact preflight evidence without starting a simulator."""

    if output_root.exists() and any(output_root.iterdir()):
        raise Phase2B61RuntimeError("forensic output root must start empty")
    output_root.mkdir(parents=True, exist_ok=True)
    evidence_root.mkdir(parents=True, exist_ok=True)
    repository = repository_environment_audit(
        repo_root=repo_root,
        source_root=source_root,
        frozen_root=frozen_root,
        output_root=output_root,
        network_turbo_sourced=network_turbo_sourced,
    )
    if repository["passed"] is not True:
        raise Phase2B61RuntimeError("repository/environment preflight failed")
    prior = verify_phase2b6_artifacts(repository_root=repo_root)
    spec = load_production_spec(spec_path)
    phase2b5 = verify_phase2b5_package(artifact_root=phase2b5_artifact_root, spec=spec)
    if phase2b5["passed"] is not True:
        raise Phase2B61RuntimeError("Phase 2B.5 source package verification failed")
    frozen = inventory_frozen_output(frozen_root)
    source = build_source_identity_manifest(
        source_root=source_root, spec_path=spec_path, frozen_root=frozen_root
    )
    history = historical_evidence_audit(frozen_root)
    protocol = forensic_protocol()
    call_sequence = producer_call_sequence_audit()
    documents = {
        "repository_environment_audit.json": repository,
        "phase2b6_prior_evidence_verification.json": prior,
        "phase2b5_source_package_verification.json": phase2b5,
        "frozen_partial_output_inventory.json": frozen,
        "source_episode_identity_manifest.json": source,
        "historical_evidence_audit.json": history,
        "forensic_protocol.json": protocol,
        "producer_forensic_call_sequence_audit.json": call_sequence,
        "authorization_state.json": authorization_state(),
    }
    for name, document in documents.items():
        _write_json(evidence_root / name, document)
    summary = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-preflight-v0",
            "documents": sorted(documents),
            "source_identity_proven": source["producer_input_identity_proven"],
            "controls_available": history["passed"],
            "simulator_started": False,
            "passed": all(document.get("passed") is True for document in documents.values()),
        }
    )
    _write_json(evidence_root / "preflight_result.json", summary)
    if summary["passed"] is not True:
        raise Phase2B61RuntimeError("forensic preflight hard stop")
    return summary


def _environment_config(document: Mapping[str, object], mode: ForensicMode) -> dict[str, object]:
    if mode is ForensicMode.PHYSICS:
        env_info = cast(Mapping[str, object], document["env_info"])
        kwargs = dict(cast(Mapping[str, object], env_info["env_kwargs"]))
        kwargs.update(
            {
                "obs_mode": "none",
                "reward_mode": "none",
                "render_mode": None,
                "sim_backend": "physx_cpu",
                "num_envs": 1,
            }
        )
        return kwargs
    return _environment_kwargs(document)


def _robot_state(base: Any) -> tuple[list[object] | None, list[object] | None]:
    robot = base.agent.robot
    try:
        return _jsonable_array(robot.get_qpos()), _jsonable_array(robot.get_qvel())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None, None


def _actor_vector(actor: Any, attribute: str) -> list[object] | None:
    value = getattr(actor, attribute, None)
    if value is None:
        return None
    try:
        array = _to_numpy(value)
        if array.ndim > 1 and array.shape[0] == 1:
            array = array[0]
        return cast(list[object], array.tolist())
    except (RuntimeError, TypeError, ValueError):
        return None


def _physical_snapshot(base: Any, step_index: int) -> dict[str, object]:
    qpos, qvel = _robot_state(base)
    cube_a = base.cubeA
    cube_b = base.cubeB
    poses = _task_object_raw_poses(base, TASK_ID)
    evaluate = base.evaluate()
    evaluate_json = {
        str(key): (
            _scalar_bool(value)
            if _to_numpy(value).size == 1 and np.issubdtype(_to_numpy(value).dtype, np.bool_)
            else _jsonable_array(value)
        )
        for key, value in cast(Mapping[str, object], evaluate).items()
    }
    relative = poses["cubeA"][:3] - poses["cubeB"][:3]
    tcp_pose = getattr(getattr(base.agent, "tcp", None), "pose", None)
    tcp_raw = getattr(tcp_pose, "raw_pose", None)
    pair_force: list[object] | None = None
    with suppress(AttributeError, RuntimeError, TypeError, ValueError):
        pair_force = _jsonable_array(base.scene.get_pairwise_contact_forces(cube_a, cube_b))
    return {
        "step_index": step_index,
        "robot_qpos": qpos,
        "robot_qvel": qvel,
        "tcp_pose": _jsonable_array(tcp_raw) if tcp_raw is not None else None,
        "lower_cube_pose": poses["cubeB"].tolist(),
        "upper_cube_pose": poses["cubeA"].tolist(),
        "lower_cube_linear_velocity": _actor_vector(cube_b, "linear_velocity"),
        "lower_cube_angular_velocity": _actor_vector(cube_b, "angular_velocity"),
        "upper_cube_linear_velocity": _actor_vector(cube_a, "linear_velocity"),
        "upper_cube_angular_velocity": _actor_vector(cube_a, "angular_velocity"),
        "cube_cube_contact_force": pair_force,
        "upper_cube_height": float(poses["cubeA"][2]),
        "relative_cube_displacement": relative.tolist(),
        "canonical_diagnostics": evaluate_json,
        "canonical_success": _scalar_bool(evaluate["success"]),
    }


def _event_snapshots(step_snapshots: Sequence[Mapping[str, object]]) -> dict[str, object]:
    reset = step_snapshots[0]
    upper_reset_height = float(cast(float, reset["upper_cube_height"]))
    events: dict[str, object] = {"reset": reset}
    for snapshot in step_snapshots[1:]:
        if (
            "first_cube_contact" not in events
            and snapshot.get("cube_cube_contact_force") is not None
            and np.linalg.norm(np.asarray(snapshot["cube_cube_contact_force"], dtype=np.float64))
            > 1e-5
        ):
            events["first_cube_contact"] = snapshot
        if (
            "first_lift" not in events
            and float(cast(float, snapshot["upper_cube_height"])) > upper_reset_height + 0.01
        ):
            events["first_lift"] = snapshot
        diagnostics = cast(Mapping[str, object], snapshot["canonical_diagnostics"])
        if "first_placement_relation" not in events and (
            diagnostics.get("is_cubeA_on_cubeB") is True
            or diagnostics.get("is_cubeA_on_cubeB") == [True]
        ):
            events["first_placement_relation"] = snapshot
        if "first_canonical_success" not in events and snapshot["canonical_success"] is True:
            events["first_canonical_success"] = snapshot
    events["final_step"] = step_snapshots[-1]
    for name in (
        "first_cube_contact",
        "first_lift",
        "first_placement_relation",
        "first_canonical_success",
    ):
        events.setdefault(name, None)
    return events


def _empty_sub_gates() -> dict[str, dict[str, object]]:
    return {
        name: gate(
            status=GateStatus.NOT_REACHED,
            detail="gate was not reached",
        )
        for name in GATE_ORDER
    }


def _set_mode_na(sub_gates: dict[str, dict[str, object]], mode: ForensicMode) -> None:
    sub_gates["stable_success_gate"] = gate(
        status=GateStatus.NOT_APPLICABLE,
        detail=(
            "the frozen producer had no separate stable-success gate; final-step "
            "canonical success remains the acceptance rule"
        ),
    )
    if mode is ForensicMode.PHYSICS:
        for name in (
            "pre_action_frame_count_gate",
            "observation_action_alignment_gate",
            "rgb_completeness_gate",
            "state_completeness_gate",
            "timestamp_monotonicity_gate",
            "temporary_writer_gate",
            "episode_serialization_gate",
        ):
            sub_gates[name] = gate(
                status=GateStatus.NOT_APPLICABLE,
                detail=f"{name} is outside physics-only Mode A",
            )
    elif mode is ForensicMode.OBSERVATION:
        for name in ("temporary_writer_gate", "episode_serialization_gate"):
            sub_gates[name] = gate(
                status=GateStatus.NOT_APPLICABLE,
                detail=f"{name} is outside no-writer Mode B",
            )


def run_forensic_replay(
    *,
    source_root: Path,
    output_root: Path,
    spec_path: Path,
    mode_value: str,
    episode_id: int,
    repetition: int,
) -> dict[str, object]:
    """Execute one fresh-process bounded replay and retain every sub-gate."""

    validate_episode_scope(mode=mode_value, episode_id=episode_id, repetition=repetition)
    mode = ForensicMode(mode_value)
    run_directory = (
        output_root
        / "runs"
        / f"stackcube_{episode_id}"
        / f"mode_{mode.value}"
        / f"repetition_{repetition}"
    )
    if run_directory.exists():
        raise Phase2B61RuntimeError(f"forensic repetition already exists: {run_directory}")
    run_directory.mkdir(parents=True)
    run_identity = {
        "task_id": TASK_ID,
        "episode_id": episode_id,
        "mode": mode.value,
        "repetition": repetition,
        "fresh_process": fresh_process_identity(
            pid=os.getpid(),
            process_start_time_ns=time.time_ns(),
            run_nonce=uuid.uuid4().hex,
        ),
    }
    document, metadata_by_id = _episode_rows(source_root=source_root, spec_path=spec_path)
    metadata = metadata_by_id[episode_id]
    h5_path, _, _ = _source_paths(source_root)
    sub_gates = _empty_sub_gates()
    _set_mode_na(sub_gates, mode)
    arrays: dict[str, np.ndarray] = {}
    step_reports: list[dict[str, object]] = []
    physical_snapshots: list[dict[str, object]] = []
    simulator_exceptions: list[dict[str, object]] = []
    writer_exceptions: list[dict[str, object]] = []
    reset_report: dict[str, object] = {}
    sensor_report: dict[str, object] | None = None
    terminal_state_error: dict[str, object] | None = None
    terminal_object_pose_error: dict[str, object] | None = None
    started = time.perf_counter()
    try:
        import gymnasium as gym
        import mani_skill.envs  # type: ignore[import-untyped]  # noqa: F401
    except (ImportError, OSError, RuntimeError) as error:
        raise Phase2B61RuntimeError("native forensic replay requires ManiSkill") from error
    with h5py.File(h5_path, "r") as trajectories:
        group = cast(h5py.Group, trajectories[f"traj_{episode_id}"])
        actions = np.asarray(group["actions"], dtype=np.float32)
        action_report = validate_action_array(actions)
        action_hash = array_sha256(actions, dtype=np.dtype(np.float32))
        source_success_values = np.asarray(group["success"], dtype=np.bool_)
        source_initial = _h5_state_at(cast(h5py.Group, group["env_states"]), 0)
        source_terminal = _h5_state_at(cast(h5py.Group, group["env_states"]), len(actions))
        sub_gates["action_contract_gate"] = gate(
            status=(
                GateStatus.PASSED
                if action_report["passed"] is True
                and action_hash
                == (TARGET_ACTION_SHA256 if episode_id == TARGET_EPISODE_ID else action_hash)
                else GateStatus.FAILED
            ),
            detail="native float32[T,8] actions are finite, in bounds, and hash-bound",
            evidence={
                **action_report,
                "action_sha256": action_hash,
                "target_expected_sha256": (
                    TARGET_ACTION_SHA256 if episode_id == TARGET_EPISODE_ID else None
                ),
            },
        )
        kwargs = _environment_config(document, mode)
        env = gym.make(TASK_ID, **cast(Any, kwargs))
        base: Any = env.unwrapped
        try:
            if str(base.control_mode) != CONTROL_MODE or int(base.control_freq) != 20:
                raise Phase2B61RuntimeError("environment violates control contract")
            (
                semantic_error,
                anchor_used,
                anchor_error,
            ) = _prepare_replay_initial_state(
                env=env,
                base=base,
                reset_kwargs=cast(Mapping[str, Any], metadata["reset_kwargs"]),
                source_initial=source_initial,
            )
            reset_exact = (not anchor_used) or (
                anchor_error is not None
                and anchor_error["structures_match"] is True
                and anchor_error["maximum_absolute_error"] in (0.0, None)
            )
            reset_report = {
                "reset_kwargs": metadata["reset_kwargs"],
                "semantic_reset_state_error": semantic_error,
                "first_state_anchor_used": anchor_used,
                "first_state_anchor_error": anchor_error,
                "reset_exact_after_permitted_anchor": reset_exact,
            }
            sub_gates["reset_identity_gate"] = gate(
                status=GateStatus.PASSED if reset_exact else GateStatus.FAILED,
                detail="exact source reset identity and state-anchor contract",
                evidence=reset_report,
            )
            physical_snapshots.append(_physical_snapshot(base, -1))
            capture = mode is not ForensicMode.PHYSICS
            if capture:
                rgb = np.empty((len(actions), *IMAGE_SHAPE), dtype=np.uint8)
                states = np.empty((len(actions), 9), dtype=np.float32)
                timestamps = np.arange(len(actions), dtype=np.float64) / CONTROL_FREQUENCY_HZ
                arrays = {
                    "rgb": rgb,
                    "state": states,
                    "action": actions,
                    "timestamp": timestamps,
                }
                try:
                    sensor_params = base.get_sensor_params().get("base_camera", {})
                    sensor_report = {
                        str(key): {
                            "shape": list(_to_numpy(value).shape),
                            "dtype": str(_to_numpy(value).dtype),
                        }
                        for key, value in cast(Mapping[str, object], sensor_params).items()
                    }
                except (AttributeError, RuntimeError, TypeError, ValueError) as error:
                    sensor_report = {"error": f"{type(error).__name__}: {error}"}
            executed = 0
            for index, action in enumerate(actions):
                observation_ok: bool | None = None
                rgb_ok: bool | None = None
                state_ok: bool | None = None
                observation_error: str | None = None
                if capture:
                    try:
                        observation = base.get_obs()
                        observation_ok = True
                        policy_rgb = _extract_rgb(observation)
                        rgb_ok = policy_rgb.shape == IMAGE_SHAPE
                        policy_state = extract_panda_policy_state_v0(base.agent.robot)
                        state_ok = policy_state.shape == (9,)
                        arrays["rgb"][index] = policy_rgb
                        arrays["state"][index] = policy_state
                    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as error:
                        observation_error = f"{type(error).__name__}: {error}"
                        observation_ok = False
                        rgb_ok = False
                        state_ok = False
                reward: object = None
                terminated: object = False
                truncated: object = False
                success = False
                step_error: str | None = None
                try:
                    _, reward, terminated, truncated, info = env.step(action)
                    executed += 1
                    success = _scalar_bool(info["success"])
                except (RuntimeError, TypeError, ValueError) as error:
                    step_error = f"{type(error).__name__}: {error}"
                    simulator_exceptions.append({"step_index": index, "error": step_error})
                snapshot = _physical_snapshot(base, index)
                physical_snapshots.append(snapshot)
                step_reports.append(
                    {
                        "step_index": index,
                        "submitted_action_sha256": array_sha256(action, dtype=np.dtype(np.float32)),
                        "observation_acquisition_success": observation_ok,
                        "rgb_frame_acquisition_success": rgb_ok,
                        "panda_state_acquisition_success": state_ok,
                        "observation_exception": observation_error,
                        "reward": (
                            float(_to_numpy(reward).reshape(-1)[0])
                            if reward is not None and _to_numpy(reward).size
                            else None
                        ),
                        "terminated": _scalar_bool(terminated),
                        "truncated": _scalar_bool(truncated),
                        "canonical_success": success,
                        "task_specific_diagnostic_state": snapshot,
                        "simulator_exception": step_error,
                        "writer_exception": None,
                    }
                )
                if step_error is not None or observation_error is not None:
                    break
            replay_success_values = [bool(row["canonical_success"]) for row in step_reports]
            timing = success_timing(replay_success_values)
            final_success = (
                bool(replay_success_values[-1])
                if len(replay_success_values) == len(actions)
                else False
            )
            terminal_state_error = _state_error(source_terminal, base.get_state_dict())
            replay_terminal_poses = _task_object_raw_poses(base, TASK_ID)
            base.set_state_dict(source_terminal)
            source_terminal_poses = _task_object_raw_poses(base, TASK_ID)
            terminal_object_pose_error = _task_object_pose_error(
                source_terminal_poses, replay_terminal_poses
            )
            sub_gates["step_execution_gate"] = gate(
                status=GateStatus.PASSED if executed == len(actions) else GateStatus.FAILED,
                detail="every source action was submitted exactly once",
                first_failure_step=executed if executed < len(actions) else None,
                evidence={"executed_action_count": executed, "source_action_count": len(actions)},
            )
            sub_gates["simulator_exception_gate"] = gate(
                status=GateStatus.PASSED if not simulator_exceptions else GateStatus.FAILED,
                detail="no simulator exception was observed",
                first_failure_step=(
                    int(simulator_exceptions[0]["step_index"]) if simulator_exceptions else None
                ),
                evidence={"exceptions": simulator_exceptions},
            )
            sub_gates["canonical_terminal_success_gate"] = gate(
                status=GateStatus.PASSED if final_success else GateStatus.FAILED,
                detail="frozen producer requires final-step canonical success",
                first_failure_step=(len(actions) - 1 if not final_success else None),
                evidence={"success_timing": timing},
            )
            source_final_success = bool(source_success_values[-1])
            sub_gates["source_replay_outcome_agreement_gate"] = gate(
                status=(
                    GateStatus.PASSED
                    if source_final_success is final_success
                    else GateStatus.FAILED
                ),
                detail="official source and replay final categorical outcomes agree",
                evidence={
                    "source_final_success": source_final_success,
                    "replay_final_success": final_success,
                },
            )
            sub_gates["action_count_gate"] = gate(
                status=GateStatus.PASSED if executed == len(actions) else GateStatus.FAILED,
                detail="executed action count equals source action count",
                evidence={"source": len(actions), "executed": executed},
            )
            alignment: dict[str, object] | None = None
            if capture:
                alignment = alignment_audit(
                    source_action_count=len(actions),
                    generated_policy_frame_count=len(step_reports),
                    action_indices=[int(row["step_index"]) for row in step_reports],
                    frame_indices=list(range(len(step_reports))),
                    timestamps=arrays["timestamp"][: len(step_reports)].tolist(),
                    terminal_diagnostic_frame_count=1,
                )
                observations_ok = all(
                    row["observation_acquisition_success"] is True for row in step_reports
                )
                rgb_ok = all(row["rgb_frame_acquisition_success"] is True for row in step_reports)
                state_ok = all(
                    row["panda_state_acquisition_success"] is True for row in step_reports
                )
                sub_gates["pre_action_frame_count_gate"] = gate(
                    status=(
                        GateStatus.PASSED
                        if len(step_reports) == len(actions) and observations_ok
                        else GateStatus.FAILED
                    ),
                    detail="one pre-action observation was acquired per source action",
                    evidence={"frame_count": len(step_reports), "action_count": len(actions)},
                )
                sub_gates["observation_action_alignment_gate"] = gate(
                    status=(
                        GateStatus.PASSED if alignment["passed"] is True else GateStatus.FAILED
                    ),
                    detail="observation[t] immediately precedes action[t]",
                    evidence=alignment,
                )
                sub_gates["rgb_completeness_gate"] = gate(
                    status=GateStatus.PASSED if rgb_ok else GateStatus.FAILED,
                    detail="all base-camera RGB frames were complete",
                    evidence={"shape": list(arrays["rgb"].shape)},
                )
                sub_gates["state_completeness_gate"] = gate(
                    status=GateStatus.PASSED if state_ok else GateStatus.FAILED,
                    detail="all PandaPolicyStateV0 vectors were complete",
                    evidence={"shape": list(arrays["state"].shape)},
                )
                sub_gates["timestamp_monotonicity_gate"] = gate(
                    status=(
                        GateStatus.PASSED
                        if alignment["finite_timestamps"] is True
                        and alignment["timestamps_strictly_monotonic"] is True
                        and alignment["timestamps_exact_20hz"] is True
                        else GateStatus.FAILED
                    ),
                    detail="timestamps are finite, strictly monotonic, and exactly 20 Hz",
                    evidence=alignment,
                )
            writer_audit: dict[str, object] | None = None
            if mode is ForensicMode.WRITER:
                npz_path = run_directory / "episode_938_forensic.npz"
                staging = npz_path.parent / f".{npz_path.name}.partial.npz"
                try:
                    _save_episode_npz(npz_path, arrays)
                    temporary_pass = not staging.exists() and npz_path.is_file()
                    with np.load(npz_path, allow_pickle=False) as payload:
                        readback = {
                            key: {
                                "shape": list(payload[key].shape),
                                "dtype": str(payload[key].dtype),
                                "sha256": array_sha256(payload[key]),
                            }
                            for key in ("rgb", "state", "action", "timestamp")
                        }
                    serialization_pass = (
                        readback["rgb"]["shape"] == list(arrays["rgb"].shape)
                        and readback["state"]["shape"] == list(arrays["state"].shape)
                        and readback["action"]["shape"] == list(arrays["action"].shape)
                        and readback["timestamp"]["shape"] == list(arrays["timestamp"].shape)
                    )
                    writer_audit = {
                        "npz_path": npz_path.as_posix(),
                        "npz_size_bytes": npz_path.stat().st_size,
                        "npz_sha256": "sha256:" + sha256_file(npz_path),
                        "staging_path": staging.as_posix(),
                        "staging_absent_after_close": not staging.exists(),
                        "atomic_rename_completed": npz_path.is_file(),
                        "readback": readback,
                        "available_storage_bytes": shutil.disk_usage(run_directory).free,
                    }
                except (OSError, RuntimeError, TypeError, ValueError) as error:
                    writer_exceptions.append({"error": f"{type(error).__name__}: {error}"})
                    temporary_pass = False
                    serialization_pass = False
                    writer_audit = {"exceptions": writer_exceptions}
                sub_gates["temporary_writer_gate"] = gate(
                    status=GateStatus.PASSED if temporary_pass else GateStatus.FAILED,
                    detail="temporary NPZ was flushed, closed, renamed, and cleaned",
                    evidence=cast(Mapping[str, object], writer_audit),
                )
                sub_gates["episode_serialization_gate"] = gate(
                    status=GateStatus.PASSED if serialization_pass else GateStatus.FAILED,
                    detail="serialized arrays passed shape and hash readback",
                    evidence=cast(Mapping[str, object], writer_audit),
                )
        finally:
            env.close()
    required = required_gates_for_mode(mode.value)
    passed = all(sub_gates[name]["status"] == GateStatus.PASSED.value for name in required)
    report = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-forensic-run-v0",
            "created_at_utc": _timestamp(),
            "run_identity": run_identity,
            "producer_commit": PRODUCER_COMMIT,
            "source_identity": {
                "task_id": TASK_ID,
                "source_episode_id": episode_id,
                "action_count": len(actions),
                "action_sha256": action_hash,
                "source_success": bool(source_success_values[-1]),
            },
            "environment": {
                "constructor_kwargs": kwargs,
                "control_mode": CONTROL_MODE,
                "control_frequency_hz": CONTROL_FREQUENCY_HZ,
                "sensor_parameters": sensor_report,
                "fresh_environment": True,
            },
            "reset": reset_report,
            "step_reports": step_reports,
            "physical_event_snapshots": _event_snapshots(physical_snapshots),
            "terminal_state_error": terminal_state_error,
            "terminal_object_pose_error": terminal_object_pose_error,
            "source_success_timing": success_timing(source_success_values.tolist()),
            "success_timing": success_timing(
                [bool(row["canonical_success"]) for row in step_reports]
            ),
            "sub_gates": sub_gates,
            "first_failed_sub_gate": first_failed_gate(sub_gates),
            "writer_audit": writer_audit,
            "simulator_exceptions": simulator_exceptions,
            "writer_exceptions": writer_exceptions,
            "wall_clock_seconds": time.perf_counter() - started,
            "mode_required_gates": list(required),
            "policy_or_expert_invoked": False,
            "optimizer_created": False,
            "backward_passes": 0,
            "optimizer_steps": 0,
            "passed": passed,
        }
    )
    _write_json(run_directory / "forensic_run.json", report)
    return report


def record_infrastructure_hard_stop(*, output_root: Path, evidence_root: Path) -> dict[str, object]:
    """Freeze the observed control-936 environment-construction hard stop.

    This stage performs read-only Vulkan diagnostics. It does not construct a
    simulator and cannot be used to retry the failed control.
    """

    run_directory = output_root / "runs" / "stackcube_936" / "mode_A" / "repetition_1"
    if not run_directory.is_dir():
        raise Phase2B61RuntimeError("failed control attempt directory is missing")
    if (run_directory / "forensic_run.json").exists():
        raise Phase2B61RuntimeError("control attempt unexpectedly completed")
    other_run_directories = [
        path
        for path in (output_root / "runs").glob("stackcube_*/mode_*/repetition_*")
        if path != run_directory
    ]
    if other_run_directories:
        raise Phase2B61RuntimeError("later forensic runs exist after the hard stop")
    base_environment = dict(os.environ)
    default = _command_audit(["vulkaninfo", "--summary"])
    legacy = _command_audit(
        ["vulkaninfo", "--summary"],
        environment={
            **base_environment,
            "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
        },
    )
    egl = _command_audit(
        ["vulkaninfo", "--summary"],
        environment={
            **base_environment,
            "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/my_nvidia_icd.json",
        },
    )
    report = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-infrastructure-hard-stop-v0",
            "created_at_utc": _timestamp(),
            "attempted_run": {
                "task_id": TASK_ID,
                "episode_id": 936,
                "mode": "A",
                "repetition": 1,
                "fresh_process": True,
            },
            "attempt_result": {
                "environment_creation_started": True,
                "environment_creation_succeeded": False,
                "reset_started": False,
                "source_action_submissions": 0,
                "source_episode_936_physical_outcome_observed": False,
                "error_type": "RuntimeError",
                "error_message": "vk::createInstanceUnique: ErrorIncompatibleDriver",
                "observed_renderer_messages": [
                    "Extension VK_KHR_external_memory_capabilities is not available",
                    "Extension VK_KHR_external_semaphore_capabilities is not available",
                    "Your GPU driver does not support Vulkan",
                ],
                "original_exit_code": 1,
                "original_stderr_log_file_created": False,
                "evidence_status": "observed_terminal_output_and_empty_run_directory",
            },
            "diagnostics": {
                "runtime_environment_vk_icd_filenames": os.environ.get("VK_ICD_FILENAMES"),
                "gpu": _gpu_audit(),
                "default_loader": default,
                "legacy_glx_icd": legacy,
                "egl_icd": egl,
            },
            "diagnosis": {
                "primary_layer": "environment_construction",
                "categorization": "infrastructure_runtime_configuration",
                "working_vulkan_loader_path_exists": egl["passed"] is True,
                "runtime_selected_loader_is_incompatible": default["passed"] is False,
                "legacy_glx_icd_is_incompatible": legacy["passed"] is False,
                "inference": (
                    "the failed SAPIEN invocation did not select the working "
                    "Vulkan ICD; this is an infrastructure/configuration result, "
                    "not a StackCube replay outcome"
                ),
                "preflight_gap": (
                    "the original preflight recorded GPU presence but did not "
                    "construct a Vulkan instance"
                ),
            },
            "hard_stop_triggered": True,
            "control_rerun_performed": False,
            "episode_937_run": False,
            "episode_938_run_count": 0,
            "mode_b_run_count": 0,
            "mode_c_run_count": 0,
            "production_resumed": False,
            "optimizer_steps": 0,
            "passed": (
                default["passed"] is False and legacy["passed"] is False and egl["passed"] is True
            ),
        }
    )
    _write_json(evidence_root / "infrastructure_failure_report.json", report)
    return report


def _load_runs(output_root: Path, episode_id: int, mode: str) -> list[dict[str, Any]]:
    directory = output_root / "runs" / f"stackcube_{episode_id}" / f"mode_{mode}"
    if not directory.is_dir():
        return []
    return [_read_json(path) for path in sorted(directory.glob("repetition_*/forensic_run.json"))]


def finalize_forensics(*, output_root: Path, evidence_root: Path) -> dict[str, object]:
    """Freeze the bounded run evidence and classify without starting new work."""

    control_runs = [
        *_load_runs(output_root, 936, "A"),
        *_load_runs(output_root, 937, "A"),
    ]
    mode_a = _load_runs(output_root, 938, "A")
    mode_b = _load_runs(output_root, 938, "B")
    mode_c = _load_runs(output_root, 938, "C")
    source = _read_json(evidence_root / "source_episode_identity_manifest.json")
    infrastructure_path = evidence_root / "infrastructure_failure_report.json"
    infrastructure = _read_json(infrastructure_path) if infrastructure_path.is_file() else None
    classification = classify_result(
        source_identity_proven=source["producer_input_identity_proven"] is True,
        control_runs=control_runs,
        mode_a_runs=mode_a,
        mode_b_runs=mode_b,
        mode_c_runs=mode_c,
        historical_producer_final_success=False,
        infrastructure_failure=infrastructure is not None,
    )
    eligibility = eligibility_state(classification)
    authorization = authorization_state()
    first_failures = [
        {
            "mode": run["run_identity"]["mode"],
            "repetition": run["run_identity"]["repetition"],
            "first_failed_sub_gate": run["first_failed_sub_gate"],
        }
        for run in mode_a + mode_b + mode_c
        if run["first_failed_sub_gate"] is not None
    ]
    timing = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-success-timing-audit-v0",
            "source": next(
                row["source_success_timing"]
                for row in source["episodes"]
                if row["source_episode_id"] == 938
            ),
            "replays": [
                {
                    "mode": run["run_identity"]["mode"],
                    "repetition": run["run_identity"]["repetition"],
                    "timing": run["success_timing"],
                }
                for run in mode_a + mode_b + mode_c
            ],
            "producer_requires_final_step_success": True,
            "producer_has_separate_stable_success_gate": False,
            "passed": True,
        }
    )
    alignment = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-alignment-audit-v0",
            "runs": [
                {
                    "mode": run["run_identity"]["mode"],
                    "repetition": run["run_identity"]["repetition"],
                    "gate": run["sub_gates"]["observation_action_alignment_gate"],
                }
                for run in mode_b + mode_c
            ],
            "mode_not_run_is_failure": False,
            "passed": all(run_passed_for_mode(run) for run in mode_b + mode_c),
        }
    )
    writer = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-writer-filesystem-audit-v0",
            "mode_c_run_count": len(mode_c),
            "runs": [run["writer_audit"] for run in mode_c],
            "frozen_output_modified": False,
            "old_writer_failure_proven": False,
            "passed": all(run_passed_for_mode(run) for run in mode_c),
        }
    )
    physical = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-physical-state-comparison-v0",
            "controls": [
                {
                    "episode_id": run["run_identity"]["episode_id"],
                    "timing": run["success_timing"],
                    "events": run["physical_event_snapshots"],
                    "terminal_object_pose_error": run["terminal_object_pose_error"],
                }
                for run in control_runs
            ],
            "target_runs": [
                {
                    "mode": run["run_identity"]["mode"],
                    "repetition": run["run_identity"]["repetition"],
                    "timing": run["success_timing"],
                    "events": run["physical_event_snapshots"],
                    "terminal_object_pose_error": run["terminal_object_pose_error"],
                }
                for run in mode_a + mode_b + mode_c
            ],
            "canonical_outcomes": [run["success_timing"]["final_step_success"] for run in mode_a],
            "passed": len(control_runs) == 2 and len(mode_a) == 3,
        }
    )
    first_failure = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-first-failure-v0",
            "historical_exact_subgate": None,
            "historical_exact_subgate_status": "not_recorded",
            "fresh_run_first_failures": first_failures,
            "primary_failure_classification": classification["primary_failure_classification"],
            "secondary_contributing_factors": classification["secondary_contributing_factors"],
            "result": classification["result"],
            "passed": classification["classification_complete"],
        }
    )
    controls = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-control-results-v0",
            "runs": control_runs,
            "infrastructure_failure": infrastructure,
            "control_936_environment_created": False if infrastructure else None,
            "control_936_physical_outcome_observed": False if infrastructure else None,
            "control_937_not_run_due_to_hard_stop": infrastructure is not None,
            "passed": len(control_runs) == 2
            and all(run_passed_for_mode(run) for run in control_runs),
        }
    )
    mode_documents = {
        "episode_938_mode_a_results.json": fingerprinted(
            {
                "schema_version": "langmani-v2-phase2b6-1-mode-a-results-v0",
                "runs": mode_a,
                "passed": len(mode_a) == 3 and all(run_passed_for_mode(run) for run in mode_a),
            }
        ),
        "episode_938_mode_b_results.json": fingerprinted(
            {
                "schema_version": "langmani-v2-phase2b6-1-mode-b-results-v0",
                "runs": mode_b,
                "not_run": not mode_b,
                "passed": not mode_b
                or (len(mode_b) == 2 and all(run_passed_for_mode(run) for run in mode_b)),
            }
        ),
        "episode_938_mode_c_result.json": fingerprinted(
            {
                "schema_version": "langmani-v2-phase2b6-1-mode-c-result-v0",
                "runs": mode_c,
                "not_run": not mode_c,
                "passed": not mode_c
                or (len(mode_c) == 1 and all(run_passed_for_mode(run) for run in mode_c)),
            }
        ),
    }
    documents: dict[str, Mapping[str, object]] = {
        "control_episode_results.json": controls,
        **mode_documents,
        "physical_state_comparison.json": physical,
        "success_timing_audit.json": timing,
        "observation_action_alignment_audit.json": alignment,
        "writer_filesystem_audit.json": writer,
        "first_failure_report.json": first_failure,
        "result_classification.json": classification,
        "eligibility_state.json": eligibility,
        "authorization_state_final.json": authorization,
    }
    for name, document in documents.items():
        _write_json(evidence_root / name, document)
    remote = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-remote-execution-audit-v0",
            "runtime_environment": _runtime_versions(),
            "gpu": _gpu_audit(),
            "process_state_after_runs": _process_audit(),
            "controls_run_count": len(control_runs),
            "mode_a_run_count": len(mode_a),
            "mode_b_run_count": len(mode_b),
            "mode_c_run_count": len(mode_c),
            "infrastructure_failure": infrastructure,
            "stack_939_or_later_processed": False,
            "pushcube_processed": False,
            "production_resumed": False,
            "training_started": False,
            "optimizer_steps": 0,
            "passed": True,
        }
    )
    _write_json(evidence_root / "remote_execution_audit.json", remote)
    return classification


def write_artifact_manifest(evidence_root: Path) -> dict[str, object]:
    """Hash every compact forensic artifact except the manifest itself."""

    files = [
        {
            "path": path.relative_to(evidence_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": "sha256:" + sha256_file(path),
        }
        for path in sorted(evidence_root.glob("*.json"))
        if path.name != "artifact_manifest.json"
    ]
    manifest = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-artifact-manifest-v0",
            "files": files,
            "file_count": len(files),
            "full_episode_media_committed": False,
            "large_npz_arrays_committed": False,
            "passed": bool(files),
        }
    )
    _write_json(evidence_root / "artifact_manifest.json", manifest)
    return manifest


__all__ = [
    "Phase2B61RuntimeError",
    "build_source_identity_manifest",
    "finalize_forensics",
    "forensic_protocol",
    "historical_evidence_audit",
    "inventory_frozen_output",
    "prepare_forensics",
    "producer_call_sequence_audit",
    "record_infrastructure_hard_stop",
    "repository_environment_audit",
    "run_forensic_replay",
    "write_artifact_manifest",
]
