"""Fail-closed contracts for LangMani 2.0 Phase 2B.5.

This module is deliberately independent of ManiSkill's renderer and replay
runtime.  It validates immutable official demonstration bytes, defines the
future student-facing contract, and classifies evidence produced by the native
Linux runtime.  It never creates an optimizer or a policy.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import numpy as np

from langmani.datasets.policy_state import PANDA_POLICY_STATE_COMPONENTS

SOURCE_BRANCH: Final = "codex/langmani-v2-phase2b4-f1-ppo-diagnosis"
SOURCE_COMMIT: Final = "6ca3786702ada4e92be607a5293b9c3c31d2a7d4"
TARGET_BRANCH: Final = "codex/langmani-v2-phase2b5-official-demos"
HF_REPOSITORY: Final = "haosulab/ManiSkill_Demonstrations"
HF_REVISION: Final = "d674485bbffdd533914e52d272fdda34c0515608"
MANISKILL_VERSION: Final = "3.0.1"
MANISKILL_SOURCE_REVISION: Final = "a4a4f9272ad64b1564035874b605ceb687b63ed8"
SOURCE_LICENSE: Final = "Apache-2.0"
CONTROL_MODE: Final = "pd_joint_pos"
ACTION_DTYPE: Final = np.dtype(np.float32)
ACTION_DIMENSION: Final = 8
CONTROL_FREQUENCY_HZ: Final = 20
SIMULATION_FREQUENCY_HZ: Final = 100
IMAGE_SHAPE: Final = (256, 256, 3)
PANDA_ACTION_LOW: Final = np.asarray(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -1.0],
    dtype=np.float32,
)
PANDA_ACTION_HIGH: Final = np.asarray(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1.0],
    dtype=np.float32,
)

SELECTED_TASK_IDS: Final = ("PickCube-v1", "StackCube-v1", "PushCube-v1")
PROHIBITED_OPERATIONS: Final = frozenset(
    {
        "act_training",
        "smolvla_training",
        "vla_jepa_training",
        "diffusion_policy_training",
        "sarm_training",
        "ppo_training",
        "lora_training",
        "optimizer_creation",
        "backward_pass",
        "custom_push_expert_development",
        "custom_mpc",
        "custom_reset_replay",
    }
)


class Phase2B5ContractError(ValueError):
    """Raised when evidence violates the frozen Phase 2B.5 contract."""


class Compatibility(StrEnum):
    """Compatibility with the preferred common Panda action contract."""

    DIRECT = "directly_compatible"
    CONVERTIBLE = "compatible_after_documented_deterministic_conversion"
    INCOMPATIBLE = "incompatible"
    INSUFFICIENT = "insufficient_evidence"


class Phase2B5Result(StrEnum):
    """The only allowed terminal result classifications."""

    RESULT_A = "RESULT_A"
    RESULT_B = "RESULT_B"
    RESULT_C = "RESULT_C"
    RESULT_D = "RESULT_D"


@dataclass(frozen=True, slots=True)
class CandidateTask:
    """Static official task identity; byte-derived fields are audited separately."""

    task_id: str
    skill_family: str
    canonical_instruction: str
    official_demo_registered: bool
    preferred: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "skill_family": self.skill_family,
            "canonical_instruction": self.canonical_instruction,
            "official_demo_registered": self.official_demo_registered,
            "preferred": self.preferred,
        }


CANDIDATE_TASKS: Final = {
    task.task_id: task
    for task in (
        CandidateTask(
            "PickCube-v1",
            "pick_and_place",
            "Pick up the cube and move it to the target location.",
            True,
            True,
        ),
        CandidateTask(
            "StackCube-v1",
            "stacking",
            "Pick up one cube and stack it on top of the other cube.",
            True,
            True,
        ),
        CandidateTask(
            "PushCube-v1",
            "planar_pushing",
            "Push the cube onto the target region.",
            True,
            True,
        ),
        CandidateTask(
            "PokeCube-v1",
            "tool_mediated_pushing",
            "Use the peg to push the cube to the goal region.",
            True,
            True,
        ),
        CandidateTask(
            "PullCube-v1",
            "pulling",
            "Pull the cube to the target position.",
            True,
            False,
        ),
        CandidateTask(
            "PegInsertionSide-v1",
            "insertion",
            "Insert the peg into the matching side hole.",
            True,
            False,
        ),
        CandidateTask(
            "PlugCharger-v1",
            "insertion",
            "Insert the charger plug into the receptacle.",
            True,
            False,
        ),
        CandidateTask(
            "PushT-v1",
            "orientation_or_reorientation",
            "Push the T-shaped object into the goal pose.",
            True,
            False,
        ),
    )
}


@dataclass(frozen=True, slots=True)
class SourceFileIdentity:
    """Content address and official attribution for one immutable ZIP."""

    task_id: str
    file_name: str
    size_bytes: int
    sha256: str
    repository: str = HF_REPOSITORY
    revision: str = HF_REVISION
    license: str = SOURCE_LICENSE

    def __post_init__(self) -> None:
        if self.task_id not in CANDIDATE_TASKS:
            raise Phase2B5ContractError(f"unknown official task {self.task_id!r}")
        if not self.file_name.endswith(".zip") or Path(self.file_name).name != self.file_name:
            raise Phase2B5ContractError("source file name must be a simple ZIP name")
        if self.size_bytes <= 0:
            raise Phase2B5ContractError("source size must be positive")
        if len(self.sha256) != 64 or any(ch not in "0123456789abcdef" for ch in self.sha256):
            raise Phase2B5ContractError("source SHA-256 must be lowercase hexadecimal")
        if self.revision != HF_REVISION or self.repository != HF_REPOSITORY:
            raise Phase2B5ContractError("official source repository or revision changed")
        if self.license != SOURCE_LICENSE:
            raise Phase2B5ContractError("official demonstration license changed")

    def verify(self, path: Path) -> bool:
        return (
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == self.size_bytes
            and sha256_file(path) == self.sha256
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "file_name": self.file_name,
            "size_bytes": self.size_bytes,
            "sha256": "sha256:" + self.sha256,
            "repository": self.repository,
            "revision": self.revision,
            "license": self.license,
        }


PINNED_SOURCE_FILES: Final = {
    item.task_id: item
    for item in (
        SourceFileIdentity(
            "PickCube-v1",
            "PickCube-v1.zip",
            36_590_010,
            "b2d4afb30fa309755862b98c342e6ee18918253c93f3bbac16ed6670748f26d8",
        ),
        SourceFileIdentity(
            "StackCube-v1",
            "StackCube-v1.zip",
            43_486_191,
            "f9b7d34b9aa418a04aa8e4322d4dea5aa27e8ae81757f60b210c1ffc54bf9c1b",
        ),
        SourceFileIdentity(
            "PushCube-v1",
            "PushCube-v1.zip",
            24_573_577,
            "1ce24408bf93658501faaa0ef164a0cdc29539cc412612a413fac746697eeab2",
        ),
        SourceFileIdentity(
            "PokeCube-v1",
            "PokeCube-v1.zip",
            22_891_027,
            "e682d878123da9156d655a948df356c6d35bbe3d2b9527335647233ffea70675",
        ),
        SourceFileIdentity(
            "PullCube-v1",
            "PullCube-v1.zip",
            11_141_008,
            "582ca678e1d28691d408cb738fa1723e8ec28d8c6390715eacc33c5dd740cbd7",
        ),
    )
}


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash one ordinary file without following a source-level symlink."""
    if path.is_symlink() or not path.is_file():
        raise Phase2B5ContractError(f"source must be an ordinary file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Mapping[str, object]) -> str:
    """Return a stable SHA-256 fingerprint for JSON-ready evidence."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def enforce_source_commit(parent_commit: str, branch: str) -> None:
    """Reject a branch created from any non-authoritative parent."""
    if parent_commit != SOURCE_COMMIT:
        raise Phase2B5ContractError(
            f"Phase 2B.5 must start from {SOURCE_COMMIT}, got {parent_commit}"
        )
    if branch != TARGET_BRANCH:
        raise Phase2B5ContractError(f"Phase 2B.5 branch must be {TARGET_BRANCH!r}")


def custom_route_closure() -> dict[str, object]:
    """Return the frozen custom-source closure without reopening old evidence."""
    payload: dict[str, object] = {
        "schema_version": "langmani-v2-phase2b5-custom-route-closure-v0",
        "custom_task_identity": "LangMani-PushToRegion-v0",
        "scripted_expert": {
            "candidate_e_formal": {"successes": 75, "episodes": 100, "gate": 95},
            "candidate_f_formal": {"successes": 11, "episodes": 100},
            "candidate_p_diagnostic": {"successes": 2, "episodes": 5},
        },
        "geometry_closed_loop": "COMPLETE_REJECTED",
        "simulator_mpc": "RESULT_C_INFRASTRUCTURE_PRECONDITION_FAILURE",
        "reset_replay": "RESULT_D_ORIGINAL_EXECUTED_PREFIX_UNAVAILABLE",
        "ppo_f0": {"successes": 0, "episodes": 48, "result": "RESULT_C"},
        "ppo_f1_probe_a": {
            "successes": 0,
            "episodes": 32,
            "correct_contacts": 0,
            "unsafe_wrong_object_interactions": True,
            "probe_b_ran": False,
            "result": "RESULT_C",
        },
        "custom_v2_dataset": {
            "attempted_episodes": 0,
            "accepted_episodes": 0,
            "frames": 0,
            "bytes": 0,
        },
        "historical_v1_bytes_available": False,
        "f2_authorized": False,
        "probe_c_authorized": False,
        "new_custom_expert_authorized": False,
        "custom_push_expert_route_active": False,
        "custom_push_expert_reactivation_authorized": False,
    }
    payload["fingerprint"] = canonical_json_sha256(payload)
    return payload


def authorization_state(
    *,
    official_demo_source_validated: bool = False,
    phase2b6_dataset_production_eligible: bool = False,
) -> dict[str, bool]:
    """Keep eligibility separate from every training authorization."""
    if phase2b6_dataset_production_eligible and not official_demo_source_validated:
        raise Phase2B5ContractError("production eligibility requires a validated source")
    return {
        "official_demo_source_validated": official_demo_source_validated,
        "phase2b6_dataset_production_eligible": phase2b6_dataset_production_eligible,
        "act_training_authorized": False,
        "smolvla_training_authorized": False,
        "vla_jepa_training_authorized": False,
        "student_policy_training_started": False,
        "custom_push_expert_route_active": False,
        "custom_push_expert_reactivation_authorized": False,
        "full_dataset_production_started": False,
    }


def validate_metadata_document(metadata: object, *, expected_task_id: str) -> list[dict[str, Any]]:
    """Validate the public JSON sidecar and return its episode records."""
    if not isinstance(metadata, dict):
        raise Phase2B5ContractError("official trajectory metadata must be a JSON object")
    env_info = metadata.get("env_info")
    episodes = metadata.get("episodes")
    if not isinstance(env_info, dict) or not isinstance(episodes, list) or not episodes:
        raise Phase2B5ContractError("metadata requires env_info and a non-empty episodes list")
    if env_info.get("env_id") != expected_task_id:
        raise Phase2B5ContractError("metadata task identity does not match its source")
    env_kwargs = env_info.get("env_kwargs")
    if not isinstance(env_kwargs, dict):
        raise Phase2B5ContractError("metadata env_kwargs must be an object")
    if env_kwargs.get("control_mode") != CONTROL_MODE:
        raise Phase2B5ContractError("selected source must record direct pd_joint_pos actions")
    if env_kwargs.get("obs_mode") != "none":
        raise Phase2B5ContractError("official compressed source must declare obs_mode='none'")
    if not isinstance(env_info.get("max_episode_steps"), int):
        raise Phase2B5ContractError("metadata must declare max_episode_steps")

    normalized: list[dict[str, Any]] = []
    episode_ids: set[int] = set()
    for raw in episodes:
        if not isinstance(raw, dict):
            raise Phase2B5ContractError("episode metadata must be an object")
        required = {
            "episode_id",
            "episode_seed",
            "control_mode",
            "elapsed_steps",
            "reset_kwargs",
            "success",
        }
        if not required <= set(raw):
            raise Phase2B5ContractError("episode metadata is missing required fields")
        episode_id = raw["episode_id"]
        if isinstance(episode_id, bool) or not isinstance(episode_id, int) or episode_id < 0:
            raise Phase2B5ContractError("episode_id must be a non-negative integer")
        if episode_id in episode_ids:
            raise Phase2B5ContractError(f"duplicate episode_id {episode_id}")
        episode_ids.add(episode_id)
        if raw["control_mode"] != CONTROL_MODE:
            raise Phase2B5ContractError("episode control mode differs from source contract")
        if not isinstance(raw["reset_kwargs"], dict):
            raise Phase2B5ContractError("episode reset_kwargs must be an object")
        if not isinstance(raw["success"], bool):
            raise Phase2B5ContractError("episode success must be boolean")
        if (
            isinstance(raw["elapsed_steps"], bool)
            or not isinstance(raw["elapsed_steps"], int)
            or raw["elapsed_steps"] <= 0
        ):
            raise Phase2B5ContractError("elapsed_steps must be positive")
        normalized.append(dict(raw))
    return normalized


def validate_action_array(actions: object) -> dict[str, object]:
    """Validate actual executed actions without clipping, projection, or repair."""
    array = np.asarray(actions)
    if array.ndim != 2 or array.shape[1] != ACTION_DIMENSION or array.shape[0] == 0:
        raise Phase2B5ContractError(
            f"actions must have shape [T,{ACTION_DIMENSION}], got {array.shape}"
        )
    if array.dtype != ACTION_DTYPE:
        raise Phase2B5ContractError(f"actions must use float32, got {array.dtype}")
    finite = np.isfinite(array)
    nonfinite_count = int(array.size - np.count_nonzero(finite))
    below = array < PANDA_ACTION_LOW
    above = array > PANDA_ACTION_HIGH
    violation_mask = below | above
    violation_count = int(np.count_nonzero(violation_mask))
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite_values": int(np.count_nonzero(finite)),
        "nonfinite_values": nonfinite_count,
        "finite_value_rate": float(np.count_nonzero(finite) / array.size),
        "bound_violation_values": violation_count,
        "bound_violation_rate": float(violation_count / array.size),
        "minimum": [float(value) for value in np.min(array, axis=0)],
        "maximum": [float(value) for value in np.max(array, axis=0)],
        "passed": nonfinite_count == 0 and violation_count == 0,
    }


def validate_transition_lengths(
    *,
    action_count: int,
    transition_lengths: Mapping[str, int],
    env_state_lengths: Iterable[int],
) -> None:
    """Require T transitions and T+1 state snapshots for every episode."""
    if action_count <= 0:
        raise Phase2B5ContractError("action_count must be positive")
    mismatches = {
        name: length for name, length in transition_lengths.items() if length != action_count
    }
    if mismatches:
        raise Phase2B5ContractError(f"transition/action length mismatch: {mismatches}")
    state_lengths = tuple(env_state_lengths)
    if not state_lengths or any(length != action_count + 1 for length in state_lengths):
        raise Phase2B5ContractError("all environment-state leaves must have T+1 records")


def stable_reset_identity(reset_kwargs: Mapping[str, object]) -> str:
    """Fingerprint reset semantics for stratification and future split design."""
    encoded = json.dumps(
        reset_kwargs, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def select_stratified_episode_ids(
    episode_rows: Sequence[Mapping[str, object]], *, count: int
) -> tuple[int, ...]:
    """Select deterministic length-spanning episodes with distinct reset identities."""
    if count <= 0:
        raise Phase2B5ContractError("replay sample count must be positive")
    rows: list[tuple[int, int, str]] = []
    for row in episode_rows:
        episode_id = row.get("episode_id")
        length = row.get("elapsed_steps")
        reset = row.get("reset_kwargs")
        if (
            isinstance(episode_id, bool)
            or not isinstance(episode_id, int)
            or isinstance(length, bool)
            or not isinstance(length, int)
            or not isinstance(reset, Mapping)
        ):
            raise Phase2B5ContractError("malformed episode row for replay stratification")
        rows.append((episode_id, length, stable_reset_identity(reset)))
    if count > len(rows):
        count = len(rows)
    ordered = sorted(rows, key=lambda row: (row[1], row[2], row[0]))
    if count == len(ordered):
        return tuple(row[0] for row in ordered)
    targets = np.linspace(0, len(ordered) - 1, count, dtype=np.int64)
    selected: list[tuple[int, int, str]] = []
    seen_ids: set[int] = set()
    for index in targets:
        entry = ordered[int(index)]
        if entry[0] not in seen_ids:
            selected.append(entry)
            seen_ids.add(entry[0])
    for entry in ordered:
        if len(selected) == count:
            break
        if entry[0] not in seen_ids:
            selected.append(entry)
            seen_ids.add(entry[0])
    return tuple(entry[0] for entry in sorted(selected, key=lambda item: item[0]))


def compare_replay_episode(
    *,
    source_success: bool,
    replay_success: bool,
    action_count: int,
    replayed_action_count: int,
    generated_frame_count: int,
    invalid_action_count: int,
    simulator_error_count: int,
) -> dict[str, object]:
    """Compare one official action replay without changing the success predicate."""
    if min(action_count, replayed_action_count, generated_frame_count) < 0:
        raise Phase2B5ContractError("replay counts cannot be negative")
    aligned = action_count == replayed_action_count and generated_frame_count in {
        0,
        action_count + 1,
    }
    return {
        "source_success": source_success,
        "replay_success": replay_success,
        "categorical_outcome_agreement": source_success is replay_success,
        "action_count": action_count,
        "replayed_action_count": replayed_action_count,
        "generated_frame_count": generated_frame_count,
        "action_frame_alignment": aligned,
        "invalid_action_count": invalid_action_count,
        "simulator_error_count": simulator_error_count,
        "passed": (
            source_success is replay_success
            and aligned
            and invalid_action_count == 0
            and simulator_error_count == 0
        ),
    }


def aggregate_replay_gate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Apply the frozen 99%/100%/zero-error replay gate."""
    if not rows:
        raise Phase2B5ContractError("replay gate requires at least one episode")
    agreement_count = sum(row.get("categorical_outcome_agreement") is True for row in rows)
    alignment_count = sum(row.get("action_frame_alignment") is True for row in rows)
    invalid_values = [row.get("invalid_action_count", 0) for row in rows]
    simulator_error_values = [row.get("simulator_error_count", 0) for row in rows]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in invalid_values):
        raise Phase2B5ContractError("invalid action counts must be integers")
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in simulator_error_values
    ):
        raise Phase2B5ContractError("simulator error counts must be integers")
    invalid = sum(value for value in invalid_values if isinstance(value, int))
    simulator_errors = sum(value for value in simulator_error_values if isinstance(value, int))
    agreement_rate = agreement_count / len(rows)
    alignment_rate = alignment_count / len(rows)
    return {
        "episode_count": len(rows),
        "categorical_outcome_agreements": agreement_count,
        "categorical_outcome_agreement_rate": agreement_rate,
        "action_frame_alignment_rate": alignment_rate,
        "invalid_action_count": invalid,
        "simulator_error_count": simulator_errors,
        "thresholds": {
            "categorical_outcome_agreement_rate": 0.99,
            "action_frame_alignment_rate": 1.0,
            "invalid_action_count": 0,
            "simulator_error_count": 0,
        },
        "passed": (
            agreement_rate >= 0.99
            and alignment_rate == 1.0
            and invalid == 0
            and simulator_errors == 0
        ),
    }


def validate_policy_frame(frame: Mapping[str, object]) -> None:
    """Enforce the exact nonprivileged future-student frame allowlist."""
    required = {
        "rgb",
        "state",
        "action",
        "task_instruction",
        "task_id",
        "timestamp",
    }
    if set(frame) != required:
        raise Phase2B5ContractError(
            "policy frame must use the exact nonprivileged allowlist; "
            f"unexpected={sorted(set(frame) - required)}, missing={sorted(required - set(frame))}"
        )
    rgb = np.asarray(frame["rgb"])
    state = np.asarray(frame["state"])
    action = np.asarray(frame["action"])
    if rgb.shape != IMAGE_SHAPE or rgb.dtype != np.dtype(np.uint8):
        raise Phase2B5ContractError("RGB must be uint8 HWC 256x256x3")
    if state.shape != (9,) or state.dtype != np.dtype(np.float32):
        raise Phase2B5ContractError("PandaPolicyStateV0 must be float32[9]")
    if action.shape != (8,) or action.dtype != ACTION_DTYPE:
        raise Phase2B5ContractError("action must be float32[8]")
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
        raise Phase2B5ContractError("student state/action fields must be finite")
    task_id = frame["task_id"]
    instruction = frame["task_instruction"]
    if not isinstance(task_id, str) or task_id not in CANDIDATE_TASKS:
        raise Phase2B5ContractError("policy task_id must be a canonical official task")
    if instruction != CANDIDATE_TASKS[task_id].canonical_instruction:
        raise Phase2B5ContractError("task instruction does not match the frozen language contract")
    timestamp = frame["timestamp"]
    if isinstance(timestamp, bool) or not isinstance(timestamp, (float, int)):
        raise Phase2B5ContractError("timestamp must be numeric")
    if not math.isfinite(float(timestamp)) or float(timestamp) < 0:
        raise Phase2B5ContractError("timestamp must be finite and non-negative")


def padding_audit(
    episode_lengths: Sequence[int], *, chunk_lengths: Sequence[int] = (10, 50)
) -> dict[str, object]:
    """Compute future end-of-episode action padding without creating training data."""
    if not episode_lengths or any(
        isinstance(length, bool) or not isinstance(length, int) or length <= 0
        for length in episode_lengths
    ):
        raise Phase2B5ContractError("episode lengths must be positive integers")
    audits: dict[str, object] = {}
    total_frames = sum(episode_lengths)
    for horizon in chunk_lengths:
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise Phase2B5ContractError("chunk lengths must be positive integers")
        padded_chunks = 0
        padded_timesteps = 0
        for length in episode_lengths:
            for frame_index in range(length):
                missing = max(0, horizon - (length - frame_index))
                padded_chunks += int(missing > 0)
                padded_timesteps += missing
        audits[str(horizon)] = {
            "chunk_length": horizon,
            "total_chunks": total_frames,
            "padded_chunks": padded_chunks,
            "padded_chunk_percentage": 100.0 * padded_chunks / total_frames,
            "total_chunk_timesteps": total_frames * horizon,
            "padded_timesteps": padded_timesteps,
            "padded_timestep_percentage": (100.0 * padded_timesteps / (total_frames * horizon)),
            "required_mask_field": "action_is_pad",
        }
    return {
        "episode_lengths": list(episode_lengths),
        "episode_count": len(episode_lengths),
        "audits": audits,
        "lerobot_060_loss_mask_behavior": (
            "Dataset storage does not define model loss masking; future policy adapters must "
            "materialize and consume action_is_pad explicitly."
        ),
        "historical_act_comparison": "historical_act_retrain_required",
        "historical_act_reason": (
            "Historical ACT used the frozen LangMani M3B distribution and H=50. "
            "Official multi-skill tasks, reset distribution, camera content, episode lengths, "
            "and language supervision differ, so old results are context only until retrained "
            "under an identical data and runtime contract."
        ),
    }


def validate_split_disjointness(
    splits: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    """Reject frame-level or overlapping future split specifications."""
    required = {
        "train",
        "validation",
        "test_unseen_reset",
        "test_unseen_task_language",
        "test_cross_skill",
    }
    if not required <= set(splits):
        raise Phase2B5ContractError("split design is missing a required partition")
    owners: dict[str, str] = {}
    overlap: list[str] = []
    frame_level_entries = 0
    for split_name, rows in splits.items():
        for row in rows:
            if "frame_index" in row:
                frame_level_entries += 1
            identity = row.get("episode_identity")
            if not isinstance(identity, str) or not identity:
                raise Phase2B5ContractError("each split row needs an episode_identity")
            previous = owners.setdefault(identity, split_name)
            if previous != split_name:
                overlap.append(identity)
    return {
        "required_splits_present": required <= set(splits),
        "episode_identity_overlap_count": len(set(overlap)),
        "frame_level_entries": frame_level_entries,
        "passed": not overlap and frame_level_entries == 0,
    }


def classify_result(
    *,
    source_identity_valid: bool,
    replay_valid: bool,
    common_contract_valid: bool,
    observation_valid: bool,
    lerobot_readback_valid: bool,
    privilege_exclusion_valid: bool,
    split_design_valid: bool,
    accepted_skill_families: Iterable[str],
) -> Phase2B5Result:
    """Classify evidence in the fixed A/B/C/D precedence order."""
    families = frozenset(accepted_skill_families)
    if not source_identity_valid or not replay_valid:
        return Phase2B5Result.RESULT_C
    if not common_contract_valid or not observation_valid or not lerobot_readback_valid:
        return Phase2B5Result.RESULT_B
    if len(families) < 3:
        return Phase2B5Result.RESULT_D
    if not privilege_exclusion_valid or not split_design_valid:
        return Phase2B5Result.RESULT_B
    return Phase2B5Result.RESULT_A


def accepted_source_package(
    *,
    selected_task_ids: Sequence[str],
    gates: Mapping[str, bool],
    package_fields: Mapping[str, object],
) -> dict[str, object]:
    """Create the package only after every required gate and diversity check passes."""
    required_gates = {
        "source_identity",
        "source_schema",
        "common_action_contract",
        "replay",
        "observation_generation",
        "privilege_exclusion",
        "lerobot_conversion",
        "lerobot_readback",
        "split_design",
    }
    if set(gates) != required_gates or not all(gates.values()):
        raise Phase2B5ContractError("accepted package requires every frozen gate")
    if any(task_id not in CANDIDATE_TASKS for task_id in selected_task_ids):
        raise Phase2B5ContractError("accepted package contains an unknown task")
    families = {CANDIDATE_TASKS[task_id].skill_family for task_id in selected_task_ids}
    if len(families) < 3:
        raise Phase2B5ContractError("accepted package requires three genuine skill families")
    prohibited = {"model", "checkpoint", "optimizer", "training_run"}
    if prohibited & set(package_fields):
        raise Phase2B5ContractError("accepted source package cannot contain trained-model fields")
    payload: dict[str, object] = {
        "schema_version": "langmani-v2-phase2b5-accepted-official-source-package-v0",
        "selected_task_ids": list(selected_task_ids),
        "skill_families": sorted(families),
        "gates": dict(gates),
        "contains_trained_model": False,
        "conversion_scope": "five-episode-per-task pilot only",
        **dict(package_fields),
    }
    payload["fingerprint"] = canonical_json_sha256(payload)
    return payload


__all__ = [
    "ACTION_DIMENSION",
    "CANDIDATE_TASKS",
    "CONTROL_FREQUENCY_HZ",
    "CONTROL_MODE",
    "HF_REPOSITORY",
    "HF_REVISION",
    "IMAGE_SHAPE",
    "MANISKILL_SOURCE_REVISION",
    "MANISKILL_VERSION",
    "PANDA_ACTION_HIGH",
    "PANDA_ACTION_LOW",
    "PANDA_POLICY_STATE_COMPONENTS",
    "PINNED_SOURCE_FILES",
    "PROHIBITED_OPERATIONS",
    "SELECTED_TASK_IDS",
    "SOURCE_BRANCH",
    "SOURCE_COMMIT",
    "SOURCE_LICENSE",
    "TARGET_BRANCH",
    "CandidateTask",
    "Compatibility",
    "Phase2B5ContractError",
    "Phase2B5Result",
    "SourceFileIdentity",
    "accepted_source_package",
    "aggregate_replay_gate",
    "authorization_state",
    "canonical_json_sha256",
    "classify_result",
    "compare_replay_episode",
    "custom_route_closure",
    "enforce_source_commit",
    "padding_audit",
    "select_stratified_episode_ids",
    "sha256_file",
    "stable_reset_identity",
    "validate_action_array",
    "validate_metadata_document",
    "validate_policy_frame",
    "validate_split_disjointness",
    "validate_transition_lengths",
]
