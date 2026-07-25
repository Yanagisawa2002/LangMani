"""Prepare and fail-closed audit the Phase 2C-C relative Pick experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from langmani.v2.phase2c import sha256_file
from langmani.v2.phase2c_a import PACKAGE_FINGERPRINT
from langmani.v2.phase2c_b import (
    BoundedActionLatentV1,
    ModelKind,
    SmolVLATrainingConfig,
    build_model_view_manifest,
)
from langmani.v2.phase2c_b_adapter import CHECKPOINT_MANIFEST as ABSOLUTE_CHECKPOINT_MANIFEST
from langmani.v2.phase2c_c import (
    FLOAT32_RECONSTRUCTION_ATOL,
    SOURCE_COMMIT,
    TARGET_BRANCH,
    Phase2CCContractError,
    StateRelativeBoundedActionV0,
    build_static_relative_action_audit,
    canonical_fingerprint,
    derive_frozen_scales,
    relative_training_config,
)
from langmani.v2.smolvla_adapter import sha256_directory

PICK_SPLITS = ("train", "validation", "test_unseen_reset")
EXPECTED_EPISODES = {"train": 700, "validation": 100, "test_unseen_reset": 100}
PHASES = (
    "approach",
    "pre_grasp",
    "gripper_close",
    "grasp_contact",
    "lift",
    "completion",
)
PHASE_DIAGNOSTIC_FRAMES = 64
CANONICAL_PICK_INSTRUCTION = "Pick up the cube and place it on the target."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--phase2c-a1-artifact-root", type=Path, required=True)
    parser.add_argument("--phase2c-b-artifact-root", type=Path, required=True)
    parser.add_argument("--absolute-checkpoint", type=Path, required=True)
    parser.add_argument("--absolute-checkpoint-sha256", required=True)
    parser.add_argument("--original-validation-summary", type=Path, required=True)
    parser.add_argument("--repaired-validation-summary", type=Path, required=True)
    parser.add_argument("--baseline-evaluation-schedule", type=Path, required=True)
    parser.add_argument("--source-pick-h5", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    return parser.parse_args()


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2CCContractError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise Phase2CCContractError(f"{path} must contain one JSON object")
    return value


def _write_new_or_equal(path: Path, value: Mapping[str, object]) -> None:
    encoded = json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise Phase2CCContractError(f"existing immutable artifact differs: {path}")
        return
    path.write_text(encoded, encoding="utf-8", newline="\n")


def _repository_identity(expected_commit: str) -> dict[str, object]:
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise Phase2CCContractError("--expected-git-commit must be one full Git SHA")

    def run(*command: str) -> str:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()

    head = run("git", "rev-parse", "HEAD")
    branch = run("git", "branch", "--show-current")
    status = run("git", "status", "--porcelain")
    upstream = run("git", "rev-parse", "@{upstream}")
    if head != expected_commit or branch != TARGET_BRANCH or status or upstream != head:
        raise Phase2CCContractError("repository identity or cleanliness gate failed")
    return {
        "branch": branch,
        "git_commit": head,
        "upstream": upstream,
        "clean": True,
        "source_commit": SOURCE_COMMIT,
    }


def _verify_artifact_manifest(root: Path) -> dict[str, object]:
    manifest_path = root / "artifact_manifest.json"
    manifest = _read_object(manifest_path)
    records = manifest.get("files")
    if not isinstance(records, list):
        raise Phase2CCContractError(f"artifact manifest lacks file records: {root}")
    total = 0
    checked = 0
    for raw in records:
        if not isinstance(raw, Mapping):
            raise Phase2CCContractError("artifact record is malformed")
        relative = raw.get("path")
        expected_hash = raw.get("sha256")
        expected_size = raw.get("bytes", raw.get("size_bytes"))
        if (
            not isinstance(relative, str)
            or not isinstance(expected_hash, str)
            or not isinstance(expected_size, int)
        ):
            raise Phase2CCContractError("artifact record fields are malformed")
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != expected_size
            or sha256_file(path) != expected_hash.removeprefix("sha256:")
        ):
            raise Phase2CCContractError(f"artifact bytes changed: {path}")
        total += expected_size
        checked += 1
    if manifest.get("file_count") != checked:
        raise Phase2CCContractError("artifact manifest file count changed")
    return {
        "root": root.as_posix(),
        "manifest_sha256": f"sha256:{sha256_file(manifest_path)}",
        "manifest_schema": manifest.get("schema_version"),
        "manifest_fingerprint": manifest.get("fingerprint"),
        "record_digest": manifest.get("record_digest"),
        "verified_file_count": checked,
        "verified_total_bytes": total,
    }


def _running_forbidden_processes() -> list[str]:
    completed = subprocess.run(
        ("ps", "-eo", "pid=,args="),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    own_pid = os.getpid()
    tokens = (
        "train_v2_phase2c",
        "evaluate_v2_phase2c",
        "lerobot-train",
        "PickCube-v1",
        "StackCube-v1",
        "PushCube-v1",
    )
    rows = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        raw_pid, _, command = stripped.partition(" ")
        if raw_pid.isdigit() and int(raw_pid) == own_pid:
            continue
        if any(token in command for token in tokens):
            rows.append(stripped)
    return rows


def _split_root(primary: Path, split: str) -> Path:
    root = primary / "task_roots" / "pickcube" / "splits" / split
    if not root.is_dir():
        raise Phase2CCContractError(f"accepted Pick split is missing: {root}")
    return root


def _split_data(primary: Path, split: str) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    root = _split_root(primary, split)
    paths = sorted((root / "data").rglob("*.parquet"))
    if not paths:
        raise Phase2CCContractError(f"accepted Pick split has no data Parquet: {root}")
    columns = (
        "observation.state",
        "action",
        "source_episode_identity",
        "derived_episode_identity",
        "instruction_template_id",
        "primary_split",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
    )
    table = pa.concat_tables([pq.read_table(path, columns=columns) for path in paths])
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    arrays: dict[str, np.ndarray] = {
        "state": states,
        "action": actions,
        "source_episode_identity": np.asarray(
            table["source_episode_identity"].to_pylist(),
            dtype=object,
        ),
        "derived_episode_identity": np.asarray(
            table["derived_episode_identity"].to_pylist(),
            dtype=object,
        ),
        "instruction_template_id": np.asarray(
            table["instruction_template_id"].to_pylist(),
            dtype=object,
        ),
        "primary_split": np.asarray(table["primary_split"].to_pylist(), dtype=object),
        "timestamp": np.asarray(table["timestamp"].to_pylist(), dtype=np.float32),
        "frame_index": np.asarray(table["frame_index"].to_pylist(), dtype=np.int64),
        "episode_index": np.asarray(table["episode_index"].to_pylist(), dtype=np.int64),
        "index": np.asarray(table["index"].to_pylist(), dtype=np.int64),
    }
    episode_count = len(np.unique(arrays["episode_index"]))
    if (
        states.shape != (len(table), 9)
        or actions.shape != (len(table), 8)
        or episode_count != EXPECTED_EPISODES[split]
        or any(value != split for value in arrays["primary_split"])
    ):
        raise Phase2CCContractError(f"accepted Pick split contract changed: {split}")
    manifest_path = root / "langmani_phase2b6_split_manifest.json"
    manifest = _read_object(manifest_path)
    identity = {
        "split": split,
        "episode_count": episode_count,
        "frame_count": len(table),
        "split_manifest_sha256": f"sha256:{sha256_file(manifest_path)}",
        "split_manifest_fingerprint": manifest.get("fingerprint"),
        "repo_id": manifest.get("repo_id"),
        "data_files": [
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": f"sha256:{sha256_file(path)}",
            }
            for path in paths
        ],
    }
    identity["fingerprint"] = canonical_fingerprint(identity)
    return arrays, identity


def _image_identity(split: str, derived_identity: str, frame_index: int) -> str:
    payload = (
        f"{PACKAGE_FINGERPRINT}\0PickCube-v1\0{split}\0"
        f"{derived_identity}\0{frame_index}\0base_camera"
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_derived_view(
    root: Path,
    split: str,
    arrays: Mapping[str, np.ndarray],
    residual: np.ndarray,
) -> dict[str, object]:
    target = root / "derived_action_view" / f"{split}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = len(residual)
    table = pa.table(
        {
            "source_episode_identity": arrays["source_episode_identity"].tolist(),
            "derived_episode_identity": arrays["derived_episode_identity"].tolist(),
            "primary_split": [split] * rows,
            "frame_index": arrays["frame_index"],
            "episode_index": arrays["episode_index"],
            "timestamp": arrays["timestamp"],
            "instruction_template_id": arrays["instruction_template_id"].tolist(),
            "image_identity": [
                _image_identity(
                    split,
                    str(arrays["derived_episode_identity"][index]),
                    int(arrays["frame_index"][index]),
                )
                for index in range(rows)
            ],
            "delta_action": [value.tolist() for value in residual],
        }
    )
    staging = target.with_name(f".{target.name}.staging")
    pq.write_table(table, staging, compression="zstd")
    if target.exists():
        if sha256_file(target) != sha256_file(staging):
            staging.unlink()
            raise Phase2CCContractError(f"existing derived view differs: {target}")
        staging.unlink()
    else:
        os.replace(staging, target)
    return {
        "split": split,
        "path": target.as_posix(),
        "frame_count": rows,
        "bytes": target.stat().st_size,
        "sha256": f"sha256:{sha256_file(target)}",
        "media_bytes_duplicated": 0,
        "state_values_stored": False,
        "future_state_used": False,
    }


def _accepted_and_inventory(primary: Path) -> tuple[dict[str, object], dict[str, object]]:
    metadata = primary / "metadata"
    return (
        _read_object(metadata / "accepted_episode_manifest.json"),
        _read_object(metadata / "source_inventory.json"),
    )


def _phase_labels(
    h5: h5py.File,
    source_episode_id: int,
) -> list[str]:
    group = h5[f"traj_{source_episode_id}"]
    actions = np.asarray(group["actions"], dtype=np.float32)
    cube = np.asarray(group["env_states/actors/cube"], dtype=np.float32)[:-1, :3]
    goal = np.asarray(group["env_states/actors/goal_site"], dtype=np.float32)[:-1, :3]
    if len(actions) != len(cube) or len(cube) != len(goal):
        raise Phase2CCContractError("source privileged trajectory alignment changed")
    close_candidates = np.flatnonzero(actions[:, 7] < 0.0)
    first_close = int(close_candidates[0]) if len(close_candidates) else len(actions)
    initial_cube = cube[0]
    labels: list[str] = []
    for frame in range(len(actions)):
        goal_distance = float(np.linalg.norm(cube[frame] - goal[frame]))
        cube_displacement = float(np.linalg.norm(cube[frame] - initial_cube))
        lift_height = float(cube[frame, 2] - initial_cube[2])
        if goal_distance <= 0.025:
            phase = "completion"
        elif lift_height >= 0.015:
            phase = "lift"
        elif frame >= first_close + 4:
            phase = "grasp_contact"
        elif frame >= first_close:
            phase = "gripper_close"
        elif frame >= max(0, first_close - 5):
            phase = "pre_grasp"
        else:
            phase = "approach"
        if phase == "grasp_contact" and cube_displacement < 0.001:
            phase = "gripper_close"
        labels.append(phase)
    return labels


def _diagnostic_selection(
    *,
    primary: Path,
    validation: Mapping[str, np.ndarray],
    accepted: Mapping[str, object],
    source_h5: Path,
) -> dict[str, object]:
    episodes = accepted.get("episodes")
    if not isinstance(episodes, list):
        raise Phase2CCContractError("accepted manifest lacks episodes")
    source_by_derived = {
        str(row["derived_episode_identity"]): int(row["source_episode_id"])
        for row in episodes
        if isinstance(row, Mapping)
        and row.get("task_id") == "PickCube-v1"
        and row.get("primary_split") == "validation"
    }
    by_phase: dict[str, list[int]] = defaultdict(list)
    label_records: dict[int, tuple[str, int]] = {}
    cached: dict[int, list[str]] = {}
    with h5py.File(source_h5, "r") as h5:
        for index, (derived, frame) in enumerate(
            zip(
                validation["derived_episode_identity"],
                validation["frame_index"],
                strict=True,
            )
        ):
            source_id = source_by_derived.get(str(derived))
            if source_id is None:
                raise Phase2CCContractError("validation row lacks accepted source identity")
            if source_id not in cached:
                cached[source_id] = _phase_labels(h5, source_id)
            frame_index = int(frame)
            labels = cached[source_id]
            if not 0 <= frame_index < len(labels):
                raise Phase2CCContractError("validation frame exceeds source trajectory")
            phase = labels[frame_index]
            by_phase[phase].append(index)
            label_records[index] = (phase, source_id)
    records: list[dict[str, object]] = []
    for phase in PHASES:
        candidates = by_phase.get(phase, [])
        if not candidates:
            raise Phase2CCContractError(f"privileged phase has no validation frames: {phase}")
        count = min(PHASE_DIAGNOSTIC_FRAMES, len(candidates))
        selected = (
            [candidates[0]]
            if count == 1
            else [
                candidates[round(position * (len(candidates) - 1) / (count - 1))]
                for position in range(count)
            ]
        )
        for index in selected:
            label, source_id = label_records[index]
            records.append(
                {
                    "dataset_index": index,
                    "phase": label,
                    "source_episode_id": source_id,
                    "derived_episode_identity": str(validation["derived_episode_identity"][index]),
                    "frame_index": int(validation["frame_index"][index]),
                }
            )
    semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-c-phase-diagnostic-lock-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "task_id": "PickCube-v1",
        "split": "validation",
        "phase_order": list(PHASES),
        "maximum_frames_per_phase": PHASE_DIAGNOSTIC_FRAMES,
        "records": records,
        "frame_count_by_phase": {
            phase: sum(record["phase"] == phase for record in records) for phase in PHASES
        },
        "labels_use_privileged_state_offline_only": True,
        "privileged_policy_input": False,
        "source_h5_sha256": f"sha256:{sha256_file(source_h5)}",
        "settings_mutable_after_model_results": False,
    }
    return {**semantic, "fingerprint": canonical_fingerprint(semantic)}


def _evaluation_schedule(
    *,
    accepted: Mapping[str, object],
    inventory: Mapping[str, object],
    baseline: Mapping[str, object],
) -> dict[str, object]:
    accepted_rows = accepted.get("episodes")
    inventory_rows = inventory.get("episodes")
    schedules = baseline.get("schedules")
    if (
        not isinstance(accepted_rows, list)
        or not isinstance(inventory_rows, list)
        or not isinstance(schedules, Mapping)
    ):
        raise Phase2CCContractError("evaluation schedule inputs are malformed")
    inventory_by_source = {
        (str(row.get("task_id")), int(cast(int, row.get("source_episode_id")))): row
        for row in inventory_rows
        if isinstance(row, Mapping) and isinstance(row.get("source_episode_id"), int)
    }
    train_candidates = sorted(
        (
            row
            for row in accepted_rows
            if isinstance(row, Mapping)
            and row.get("task_id") == "PickCube-v1"
            and row.get("primary_split") == "train"
        ),
        key=lambda row: int(cast(int, row["source_episode_id"])),
    )[:30]
    train_rows = []
    for ordinal, row in enumerate(train_candidates):
        source_id = int(cast(int, row["source_episode_id"]))
        source = inventory_by_source.get(("PickCube-v1", source_id))
        if not isinstance(source, Mapping):
            raise Phase2CCContractError("training reset lacks source inventory")
        train_rows.append(
            {
                "evaluation_id": f"phase2c-c:train-reset:{ordinal:02d}",
                "split": "train_reset",
                "task_id": "PickCube-v1",
                "source_episode_id": source_id,
                "source_episode_identity": row.get("source_trajectory_identity"),
                "reset_identity": source.get("reset_identity"),
                "reset_kwargs": source.get("reset_kwargs"),
                "language_instruction": CANONICAL_PICK_INSTRUCTION,
                "instruction_role": "canonical_pick",
                "visual_transform": None,
            }
        )
    validation = cast(Mapping[str, object], schedules.get("validation"))
    test = cast(Mapping[str, object], schedules.get("test_unseen_reset"))
    validation_rows = cast(Sequence[Mapping[str, object]], validation.get("PickCube-v1"))
    test_rows = cast(Sequence[Mapping[str, object]], test.get("PickCube-v1"))
    if len(validation_rows) < 30 or len(test_rows) < 50 or len(train_rows) != 30:
        raise Phase2CCContractError("evaluation schedules lack frozen coverage")
    semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-c-evaluation-schedule-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "task_id": "PickCube-v1",
        "canonical_instruction": CANONICAL_PICK_INSTRUCTION,
        "training_reset": train_rows,
        "validation_horizon_screen": [dict(row) for row in validation_rows[:6]],
        "validation": [dict(row) for row in validation_rows[:30]],
        "test_unseen_reset": [dict(row) for row in test_rows[:50]],
        "execution_horizons": [1, 8],
        "horizon_selection_rule": (
            "maximize_success_then_grasp_then_lift_then_minimize_timeout_then_prefer_h1"
        ),
        "final_open_condition": "relative_validation_success_count>=3_of_30",
        "settings_mutable_after_results": False,
    }
    return {**semantic, "fingerprint": canonical_fingerprint(semantic)}


def main() -> int:
    args = parse_args()
    primary = args.primary_root.resolve()
    output = args.output_root.resolve()
    repository = _repository_identity(str(args.expected_git_commit))
    package_path = primary / "metadata" / "accepted_multiskill_dataset_package.json"
    package = _read_object(package_path)
    phase2c_a1 = _verify_artifact_manifest(args.phase2c_a1_artifact_root.resolve())
    phase2c_b = _verify_artifact_manifest(args.phase2c_b_artifact_root.resolve())
    original = _read_object(args.original_validation_summary.resolve())
    repaired = _read_object(args.repaired_validation_summary.resolve())
    absolute_checkpoint = args.absolute_checkpoint.resolve()
    absolute_digest = str(args.absolute_checkpoint_sha256).removeprefix("sha256:")
    if (
        package.get("fingerprint") != PACKAGE_FINGERPRINT
        or phase2c_b.get("record_digest")
        != "sha256:f7ef0b16bc885221900c68697c2378332e39f50258dbe253cbb16abb124b946f"
        or original.get("success_count") != 0
        or original.get("episode_count") != 30
        or original.get("execution_horizon") != 1
        or repaired.get("success_count") != 0
        or repaired.get("episode_count") != 30
        or repaired.get("execution_horizon") != 8
        or original.get("invalid_action_count") != 0
        or original.get("simulator_error_count") != 0
        or repaired.get("invalid_action_count") != 0
        or repaired.get("simulator_error_count") != 0
        or sha256_directory(absolute_checkpoint) != absolute_digest
    ):
        raise Phase2CCContractError("frozen Phase 2C-C input evidence changed")
    absolute_manifest = _read_object(absolute_checkpoint / ABSOLUTE_CHECKPOINT_MANIFEST)
    BoundedActionLatentV1.load(absolute_checkpoint)
    processor_files = sorted(
        path.name for path in absolute_checkpoint.iterdir() if "processor" in path.name
    )
    running = _running_forbidden_processes()
    if running:
        raise Phase2CCContractError("model or simulator process is active: " + repr(running))
    input_semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-c-input-verification-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "repository": repository,
        "dataset_package": {
            "path": package_path.as_posix(),
            "sha256": f"sha256:{sha256_file(package_path)}",
            "fingerprint": package.get("fingerprint"),
        },
        "phase2c_a1_artifacts": phase2c_a1,
        "phase2c_b_artifacts": phase2c_b,
        "absolute_checkpoint": {
            "path": absolute_checkpoint.as_posix(),
            "sha256": f"sha256:{absolute_digest}",
            "manifest_fingerprint": absolute_manifest.get("fingerprint"),
            "optimizer_step": absolute_manifest.get("optimizer_step"),
            "processor_files": processor_files,
            "transform_fingerprint": BoundedActionLatentV1().fingerprint,
        },
        "original_validation": {
            "path": args.original_validation_summary.resolve().as_posix(),
            "sha256": f"sha256:{sha256_file(args.original_validation_summary.resolve())}",
            "fingerprint": original.get("fingerprint"),
            "success_count": 0,
            "episode_count": 30,
            "execution_horizon": 1,
        },
        "repaired_validation": {
            "path": args.repaired_validation_summary.resolve().as_posix(),
            "sha256": f"sha256:{sha256_file(args.repaired_validation_summary.resolve())}",
            "fingerprint": repaired.get("fingerprint"),
            "success_count": 0,
            "episode_count": 30,
            "execution_horizon": 8,
        },
        "active_model_or_simulator_processes": [],
        "passed": True,
    }
    input_report = {**input_semantic, "fingerprint": canonical_fingerprint(input_semantic)}
    _write_new_or_equal(output / "input_verification.json", input_report)

    accepted, inventory = _accepted_and_inventory(primary)
    transform = StateRelativeBoundedActionV0()
    split_arrays: dict[str, dict[str, np.ndarray]] = {}
    split_identities: dict[str, dict[str, object]] = {}
    reconstruction_by_split: dict[str, object] = {}
    derived_views: list[dict[str, object]] = []
    train_scale: dict[str, object] | None = None
    train_residual: np.ndarray | None = None
    for split in PICK_SPLITS:
        arrays, identity = _split_data(primary, split)
        split_arrays[split] = arrays
        split_identities[split] = identity
        state = torch.from_numpy(cast(np.ndarray, arrays["state"]))
        action = torch.from_numpy(cast(np.ndarray, arrays["action"]))
        residual = transform.physical_residual(action, state)
        reconstructed_delta = transform.reconstruct_from_physical_residual(residual, state)
        latent = transform.encode(action, state)
        reconstructed_latent = transform.decode(latent, state)
        delta_error = torch.abs(reconstructed_delta - action)
        latent_error = torch.abs(reconstructed_latent - action)
        reconstruction_by_split[split] = {
            "frame_count": len(action),
            "physical_delta_inverse_maximum_absolute_error": float(delta_error.max()),
            "bounded_latent_inverse_maximum_absolute_error": float(latent_error.max()),
            "physical_delta_inverse_over_tolerance_count": int(
                torch.count_nonzero(delta_error > FLOAT32_RECONSTRUCTION_ATOL)
            ),
            "bounded_latent_inverse_over_tolerance_count": int(
                torch.count_nonzero(latent_error > FLOAT32_RECONSTRUCTION_ATOL)
            ),
            "nonfinite_count": int(torch.count_nonzero(~torch.isfinite(reconstructed_latent))),
        }
        residual_numpy = residual.numpy()
        derived_views.append(_write_derived_view(output, split, arrays, residual_numpy))
        if split == "train":
            train_scale = derive_frozen_scales(action, state)
            train_residual = residual_numpy
    if train_scale is None or train_residual is None:
        raise Phase2CCContractError("train-only residual statistics were not computed")
    if train_scale["exact_scale_match"] is not True:
        raise Phase2CCContractError("frozen residual scales do not match train statistics")
    total_frames = sum(int(value["frame_count"]) for value in reconstruction_by_split.values())
    maximum_error = max(
        float(value["bounded_latent_inverse_maximum_absolute_error"])
        for value in reconstruction_by_split.values()
    )
    reconstruction_semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-c-full-reconstruction-audit-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "task_id": "PickCube-v1",
        "splits": list(PICK_SPLITS),
        "total_frame_count": total_frames,
        "float32_absolute_tolerance": FLOAT32_RECONSTRUCTION_ATOL,
        "by_split": reconstruction_by_split,
        "maximum_absolute_error": maximum_error,
        "over_tolerance_count": sum(
            int(value["bounded_latent_inverse_over_tolerance_count"])
            for value in reconstruction_by_split.values()
        ),
        "native_bound_violation_count": 0,
        "future_state_used": False,
        "passed": maximum_error <= FLOAT32_RECONSTRUCTION_ATOL,
    }
    reconstruction = {
        **reconstruction_semantic,
        "fingerprint": canonical_fingerprint(reconstruction_semantic),
    }
    if reconstruction["passed"] is not True:
        raise Phase2CCContractError("full Pick reconstruction audit failed")
    _write_new_or_equal(output / "reconstruction_audit.json", reconstruction)

    residual_stats = {
        "minimum": train_residual.min(axis=0).tolist(),
        "maximum": train_residual.max(axis=0).tolist(),
        "mean": train_residual.mean(axis=0, dtype=np.float64).tolist(),
        "standard_deviation": train_residual.std(axis=0, dtype=np.float64).tolist(),
        "low_variance_dimensions_below_1e-6": [
            index
            for index, value in enumerate(train_residual.std(axis=0, dtype=np.float64))
            if value < 1e-6
        ],
    }
    state_action_semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-c-state-action-audit-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "task_id": "PickCube-v1",
        "observation_action_timing": "observation[t]_immediately_precedes_action[t]",
        "state_schema": "PandaPolicyStateV0",
        "state_field_order": [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
            "panda_finger_joint1",
            "panda_finger_joint2",
        ],
        "additional_state_fields": [],
        "action_semantics": {
            "action[0:7]": "absolute Panda arm joint-position targets in radians",
            "action[7]": "normalized Panda gripper mimic command in [-1,1]",
        },
        "gripper_current_reference": (
            "normalize arithmetic mean of measured finger joint positions "
            "from physical range [-0.01,0.04] metres to [-1,1]"
        ),
        "derived_action": "physical_target_action-current_action_reference",
        "chunk_anchor": "current_observation_state_at_policy_query",
        "future_state_used": False,
        "object_or_goal_state_used": False,
        "observations_changed": False,
        "source_media_duplicated": False,
        "split_identities": split_identities,
        "derived_views": derived_views,
        "train_only_delta_statistics": residual_stats,
        "train_scale_audit": train_scale,
        "current_reference_unambiguous": True,
        "passed": True,
    }
    state_action = {
        **state_action_semantic,
        "fingerprint": canonical_fingerprint(state_action_semantic),
    }
    _write_new_or_equal(output / "state_action_audit.json", state_action)
    _write_new_or_equal(
        output / "residual_transform_manifest.json",
        {**transform.semantic_dict(), "fingerprint": transform.fingerprint},
    )
    static_audit = build_static_relative_action_audit()
    if static_audit["passed"] is not True:
        raise Phase2CCContractError("100k relative transform audit failed")
    _write_new_or_equal(output / "action_100k_audit.json", static_audit)
    training_config = relative_training_config()
    _write_new_or_equal(output / "relative_training_config.json", training_config)

    diagnostic_lock = _diagnostic_selection(
        primary=primary,
        validation=split_arrays["validation"],
        accepted=accepted,
        source_h5=args.source_pick_h5.resolve(),
    )
    _write_new_or_equal(output / "phase_diagnostic_lock.json", diagnostic_lock)
    baseline_schedule = _read_object(args.baseline_evaluation_schedule.resolve())
    schedule = _evaluation_schedule(
        accepted=accepted,
        inventory=inventory,
        baseline=baseline_schedule,
    )
    _write_new_or_equal(output / "evaluation_schedule.json", schedule)
    real_view = build_model_view_manifest(ModelKind.PICK)
    real_view_semantic = {
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "model_view_fingerprint": real_view["fingerprint"],
        "split_identity_fingerprints": {
            split: value["fingerprint"] for split, value in split_identities.items()
        },
        "derived_action_transform_fingerprint": transform.fingerprint,
        "media_bytes_duplicated": 0,
    }
    real_view_fingerprint = canonical_fingerprint(real_view_semantic)
    training = SmolVLATrainingConfig()
    completion_semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-c-static-preparation-complete-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "input_verification_fingerprint": input_report["fingerprint"],
        "state_action_audit_fingerprint": state_action["fingerprint"],
        "action_transform_fingerprint": transform.fingerprint,
        "static_action_audit_fingerprint": static_audit["fingerprint"],
        "reconstruction_audit_fingerprint": reconstruction["fingerprint"],
        "phase_diagnostic_lock_fingerprint": diagnostic_lock["fingerprint"],
        "evaluation_schedule_fingerprint": schedule["fingerprint"],
        "training_config_fingerprint": training.fingerprint,
        "relative_training_config_fingerprint": training_config["fingerprint"],
        "real_view_fingerprint": real_view_fingerprint,
        "train_scale_exact_match": train_scale["exact_scale_match"],
        "full_reconstruction_passed": reconstruction["passed"],
        "optimizer_created": False,
        "training_started": False,
        "vla_jepa_authorized": False,
        "other_smolvla_models_authorized": False,
        "passed": True,
    }
    completion = {
        **completion_semantic,
        "fingerprint": canonical_fingerprint(completion_semantic),
    }
    _write_new_or_equal(output / "static_preparation_complete.json", completion)
    print(json.dumps(completion, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
