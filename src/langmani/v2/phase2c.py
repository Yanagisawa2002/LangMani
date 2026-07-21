"""Fail-closed data, normalization, and manifest contracts for Phase 2C."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from langmani.datasets.identity import sha256_hex
from langmani.datasets.lerobot_types import (
    ACTION_FEATURE_KEY,
    IMAGE_FEATURE_KEY,
    STATE_FEATURE_KEY,
)
from langmani.v2.policy import ObservationBatch, PolicyContractError
from langmani.v2.push_dataset import DATASET_SPLITS

PHASE2C_INPUT_SCHEMA = "langmani-v2-phase2c-input-v0"
PHASE2C_TRAIN_VIEW_SCHEMA = "langmani-v2-phase2c-train-view-v0"
PHASE2C_NORMALIZATION_SCHEMA = "langmani-v2-phase2c-normalization-v0"
PHASE2C_FEATURE_MAPPING_SCHEMA = "langmani-v2-phase2c-feature-mapping-v0"
PHASE2B_SOURCE_COMMIT = "7394faba1ed2e29ab26b70dbc4058f46e5164041"
PUSH_SKILL_FAMILY = "push_to_region"
TRAIN_EPISODES = 203
TRAIN_FRAMES = 26_968
EXPECTED_SPLIT_COUNTS: Mapping[str, tuple[int, int]] = {
    "train": (203, 26_968),
    "validation": (77, 10_425),
    "test_unseen_scene": (26, 3_382),
    "test_unseen_language": (40, 5_277),
    "test_hard": (23, 3_210),
    "test_visual_shift": (28, 4_035),
}
EXPECTED_PHASE2B_HASHES: Mapping[str, str] = {
    "replay_validation_sha256": (
        "sha256:dacb7db3b980997f104b3b580b8ea27a58dbacaec8e5307ad3b16d06b1638c65"
    ),
    "export_manifest_sha256": (
        "sha256:4bf966a83da6bcf6af7c4d53ed45e4efca8dea659d53618fef846c4ab42147f2"
    ),
    "dataset_statistics_sha256": (
        "sha256:cd16ad2179a2c225f28bf9801d1ab0cd0ec2103e7c6eb756cdddac5875b627c7"
    ),
    "split_manifest_sha256": (
        "sha256:5d9df54e3ac06c4e031c450f9677f069392d2639d0314b75d7477827567d1833"
    ),
    "leakage_audit_sha256": (
        "sha256:231f061c9362ad7345b419c58be084536cc65ab35feabd3b90577719ad7916b5"
    ),
    "unified_dataset_index_sha256": (
        "sha256:864b8d74593143fb907f1d838b243c27b0d2e239745fdf90ed3ac9bbb84b9313"
    ),
}
POLICY_FEATURE_KEYS = frozenset({IMAGE_FEATURE_KEY, STATE_FEATURE_KEY, "task"})
PRIVILEGED_FEATURE_FRAGMENTS = (
    "object_position",
    "target_position",
    "object_identity",
    "target_identity",
    "success",
    "failure",
    "expert_phase",
    "planner",
    "segmentation",
    "future",
)


class Phase2CContractError(ValueError):
    """Raised when Phase 2C input, split, or feature evidence is invalid."""


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 digest for one required file."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise Phase2CContractError(f"cannot hash required artifact {path}: {error}") from error
    return digest.hexdigest()


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    """Read one JSON object without coercing malformed content."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2CContractError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise Phase2CContractError(f"{label} must be a JSON object")
    return value


def write_json_once(path: Path, value: Mapping[str, object]) -> None:
    """Create one immutable JSON artifact, or accept an identical existing file."""

    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise Phase2CContractError(f"refusing to overwrite changed artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _prefixed_digest(path: Path) -> str:
    return f"sha256:{sha256_file(path)}"


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Phase2CContractError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Phase2CContractError(f"{label} must be a non-negative integer")
    return value


def _records(value: object, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise Phase2CContractError(f"{label} must be a list of objects")
    return cast(list[dict[str, Any]], value)


def verify_phase2b_for_phase2c(
    *,
    export_root: Path,
    phase2b_result_path: Path,
    verifier_report_path: Path,
) -> dict[str, object]:
    """Re-bind accepted Phase 2B evidence and the immutable exported split roots."""

    result = read_json_object(phase2b_result_path, "Phase 2B result manifest")
    verifier = read_json_object(verifier_report_path, "independent Phase 2B verifier report")
    for key, expected in EXPECTED_PHASE2B_HASHES.items():
        if result.get(key) != expected:
            raise Phase2CContractError(f"accepted Phase 2B identity changed: {key}")
    if result.get("completed") is not True or result.get("smolvla_phase2c_authorized") is not True:
        raise Phase2CContractError("Phase 2B did not authorize Phase 2C")
    if result.get("accepted_episode_count") != 397 or result.get("lerobot_frame_count") != 53_297:
        raise Phase2CContractError("Phase 2B accepted counts changed")
    if verifier.get("passed") is not True:
        raise Phase2CContractError("the independently rerun Phase 2B verifier did not pass")
    if verifier.get("smolvla_phase2c_authorized") is not True:
        raise Phase2CContractError("the independent verifier did not authorize SmolVLA")
    if verifier.get("accepted_count") != 397 or verifier.get("readback_frame_count") != 53_297:
        raise Phase2CContractError("independent verifier counts differ from accepted Phase 2B")

    sidecar = export_root / "langmani"
    paths = {
        "export_manifest": sidecar / "export_manifest.json",
        "dataset_schema": sidecar / "dataset_schema.json",
        "source_statistics": sidecar / "dataset_statistics.json",
        "split_manifest": sidecar / "split_manifest.json",
        "leakage_audit": sidecar / "leakage_audit.json",
        "episodes": sidecar / "episodes.json",
        "complete": sidecar / "complete.json",
    }
    export_manifest = read_json_object(paths["export_manifest"], "push export manifest")
    split_manifest = read_json_object(paths["split_manifest"], "push split manifest")
    leakage = read_json_object(paths["leakage_audit"], "push leakage audit")
    schema = read_json_object(paths["dataset_schema"], "push feature schema")
    episode_manifest = read_json_object(paths["episodes"], "push episode inventory")
    complete = read_json_object(paths["complete"], "push export completion marker")

    if _prefixed_digest(paths["export_manifest"]) != result["export_manifest_sha256"]:
        raise Phase2CContractError("export manifest bytes differ from accepted Phase 2B")
    if _prefixed_digest(paths["source_statistics"]) != result["dataset_statistics_sha256"]:
        raise Phase2CContractError("source statistics bytes differ from accepted Phase 2B")
    if _prefixed_digest(paths["split_manifest"]) != result["split_manifest_sha256"]:
        raise Phase2CContractError("split manifest bytes differ from accepted Phase 2B")
    if _prefixed_digest(paths["leakage_audit"]) != result["leakage_audit_sha256"]:
        raise Phase2CContractError("leakage audit bytes differ from accepted Phase 2B")
    if complete.get("export_manifest_sha256") != sha256_file(paths["export_manifest"]):
        raise Phase2CContractError("export completion marker does not bind the export manifest")
    if leakage.get("passed") is not True:
        raise Phase2CContractError("Phase 2B leakage audit is not passing")

    assignments = split_manifest.get("assignments")
    counts = split_manifest.get("counts")
    if not isinstance(assignments, Mapping) or not isinstance(counts, Mapping):
        raise Phase2CContractError("split manifest is malformed")
    records = _records(episode_manifest.get("records"), "episode inventory records")
    if len(records) != 397 or len(assignments) != 397:
        raise Phase2CContractError("episode inventory must contain exactly 397 unique assignments")
    by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in DATASET_SPLITS}
    seen: set[str] = set()
    for record in records:
        episode_id = _string(record.get("episode_id"), "episode_id")
        if episode_id in seen:
            raise Phase2CContractError(f"duplicate episode identity: {episode_id}")
        seen.add(episode_id)
        split = _string(record.get("split"), f"split[{episode_id}]")
        if split not in by_split or assignments.get(episode_id) != split:
            raise Phase2CContractError(f"episode split disagrees for {episode_id}")
        by_split[split].append(record)
    observed_counts: dict[str, dict[str, int]] = {}
    for split, (expected_episodes, expected_frames) in EXPECTED_SPLIT_COUNTS.items():
        selected = by_split[split]
        episodes = len(selected)
        frames = sum(_integer(item.get("frame_count"), "frame_count") for item in selected)
        if episodes != expected_episodes or frames != expected_frames:
            raise Phase2CContractError(
                f"{split} count drift: got {episodes}/{frames}, "
                f"expected {expected_episodes}/{expected_frames}"
            )
        if counts.get(split) != expected_episodes:
            raise Phase2CContractError(f"split count manifest changed for {split}")
        split_root = export_root / "splits" / split
        if not split_root.is_dir():
            raise Phase2CContractError(f"exported split root is missing: {split_root}")
        observed_counts[split] = {"episodes": episodes, "frames": frames}

    policy_features = schema.get("policy_features")
    if not isinstance(policy_features, Mapping):
        raise Phase2CContractError("Phase 2B policy feature schema is malformed")
    if set(policy_features) != {IMAGE_FEATURE_KEY, STATE_FEATURE_KEY, ACTION_FEATURE_KEY}:
        raise Phase2CContractError("Phase 2B policy feature allowlist changed")
    if schema.get("privileged_policy_fields") != []:
        raise Phase2CContractError("Phase 2B export exposes privileged policy fields")

    episode_ids = {
        split: sorted(_string(item["episode_id"], "episode_id") for item in selected)
        for split, selected in by_split.items()
    }
    source_hashes = {key: result[key] for key in EXPECTED_PHASE2B_HASHES}
    source_hashes.update(
        {
            "phase2b_result_sha256": _prefixed_digest(phase2b_result_path),
            "phase2b_verifier_report_sha256": _prefixed_digest(verifier_report_path),
            "episode_inventory_sha256": _prefixed_digest(paths["episodes"]),
            "dataset_schema_sha256": _prefixed_digest(paths["dataset_schema"]),
            "export_complete_sha256": _prefixed_digest(paths["complete"]),
        }
    )
    semantic = {
        "source_git_commit": PHASE2B_SOURCE_COMMIT,
        "dataset_version": export_manifest.get("export_fingerprint"),
        "split_episode_ids": episode_ids,
        "counts": observed_counts,
        "feature_schema": policy_features,
        "task_field": "task",
        "control_frequency_hz": 20,
        "control_mode": "pd_joint_pos",
        "action_representation": "PandaJointPositionActionV0/float32[8]",
        "image_mapping": {IMAGE_FEATURE_KEY: "base_camera/uint8[3,256,256]"},
        "state_mapping": {STATE_FEATURE_KEY: "PandaPolicyStateV0/float32[9]"},
        "source_manifest_hashes": source_hashes,
    }
    return {
        "schema_version": PHASE2C_INPUT_SCHEMA,
        **semantic,
        "dataset_roots": {
            split: (export_root / "splits" / split).resolve().as_posix() for split in DATASET_SPLITS
        },
        "phase2b_verifier_passed": True,
        "test_splits_frozen": True,
        "privileged_policy_fields": [],
        "semantic_sha256": f"sha256:{sha256_hex(semantic)}",
        "passed": True,
    }


def build_train_view_manifest(input_manifest: Mapping[str, object]) -> dict[str, object]:
    """Describe the already materialized Phase 2B train split as the only training view."""

    if input_manifest.get("passed") is not True:
        raise Phase2CContractError("cannot build a train view from an unverified input")
    counts = input_manifest.get("counts")
    identities = input_manifest.get("split_episode_ids")
    roots = input_manifest.get("dataset_roots")
    if not all(isinstance(item, Mapping) for item in (counts, identities, roots)):
        raise Phase2CContractError("Phase 2C input split fields are malformed")
    counts = cast(Mapping[str, object], counts)
    identities = cast(Mapping[str, object], identities)
    roots = cast(Mapping[str, object], roots)
    train_counts = counts.get("train")
    train_ids = identities.get("train")
    if train_counts != {"episodes": TRAIN_EPISODES, "frames": TRAIN_FRAMES}:
        raise Phase2CContractError("training split must remain 203 episodes and 26,968 frames")
    if not isinstance(train_ids, list) or len(train_ids) != TRAIN_EPISODES:
        raise Phase2CContractError("training identity inventory is malformed")
    excluded = {
        split: cast(list[object], identities[split]) for split in DATASET_SPLITS if split != "train"
    }
    if set(train_ids).intersection(item for values in excluded.values() for item in values):
        raise Phase2CContractError("training identities overlap a non-training split")
    semantic = {
        "source_input_semantic_sha256": input_manifest.get("semantic_sha256"),
        "selection": {"split": "train", "skill_family": PUSH_SKILL_FAMILY},
        "episode_ids": list(train_ids),
        "episode_count": TRAIN_EPISODES,
        "frame_count": TRAIN_FRAMES,
        "excluded_splits": [split for split in DATASET_SPLITS if split != "train"],
        "failed_demonstrations_included": False,
        "replay_copies_included": False,
        "privileged_features_included": False,
        "materialization": "existing_immutable_lerobot_train_split",
    }
    return {
        "schema_version": PHASE2C_TRAIN_VIEW_SCHEMA,
        **semantic,
        "dataset_root": _string(roots.get("train"), "train dataset root"),
        "semantic_sha256": f"sha256:{sha256_hex(semantic)}",
        "passed": True,
    }


@dataclass(slots=True)
class StreamingMoments:
    """Numerically stable per-dimension statistics without retaining frames."""

    dimension: int
    count: int = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None
    minimum: np.ndarray | None = None
    maximum: np.ndarray | None = None

    def add(self, value: object) -> None:
        array = np.asarray(value, dtype=np.float64).reshape(-1, self.dimension)
        if array.shape[0] < 1 or not np.isfinite(array).all():
            raise Phase2CContractError("statistics input must contain finite complete rows")
        for row in array:
            self.count += 1
            if self.mean is None:
                self.mean = row.copy()
                self.m2 = np.zeros_like(row)
                self.minimum = row.copy()
                self.maximum = row.copy()
                continue
            assert self.m2 is not None and self.minimum is not None and self.maximum is not None
            delta = row - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (row - self.mean)
            self.minimum = np.minimum(self.minimum, row)
            self.maximum = np.maximum(self.maximum, row)

    def to_dict(self, *, low_variance_threshold: float) -> dict[str, object]:
        if self.count < 1 or self.mean is None or self.m2 is None:
            raise Phase2CContractError("cannot finalize empty statistics")
        assert self.minimum is not None and self.maximum is not None
        variance = self.m2 / self.count
        standard_deviation = np.sqrt(np.maximum(variance, 0.0))
        return {
            "count": self.count,
            "mean": self.mean.tolist(),
            "std": standard_deviation.tolist(),
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
            "low_variance_threshold": low_variance_threshold,
            "low_variance_dimensions": [
                index
                for index, value in enumerate(standard_deviation)
                if value <= low_variance_threshold
            ],
        }


def compute_train_only_normalization(
    samples: Iterable[Mapping[str, object]],
    *,
    expected_frames: int = TRAIN_FRAMES,
    low_variance_threshold: float = 1e-8,
) -> dict[str, object]:
    """Compute state/action statistics from an already verified train-only iterator."""

    state = StreamingMoments(9)
    action = StreamingMoments(8)
    frame_count = 0
    for sample in samples:
        if STATE_FEATURE_KEY not in sample or ACTION_FEATURE_KEY not in sample:
            raise Phase2CContractError("training sample lacks state or action")
        state.add(_numpy(sample[STATE_FEATURE_KEY]))
        action.add(_numpy(sample[ACTION_FEATURE_KEY]))
        frame_count += 1
    if frame_count != expected_frames:
        raise Phase2CContractError(
            f"train-only statistics saw {frame_count} frames, expected {expected_frames}"
        )
    semantic = {
        "source": "phase2c_verified_train_view_only",
        "episode_count": TRAIN_EPISODES,
        "frame_count": frame_count,
        "features": {
            STATE_FEATURE_KEY: state.to_dict(low_variance_threshold=low_variance_threshold),
            ACTION_FEATURE_KEY: action.to_dict(low_variance_threshold=low_variance_threshold),
        },
        "image": {
            "feature": IMAGE_FEATURE_KEY,
            "source_dtype": "uint8",
            "source_range": [0, 255],
            "model_input_dtype": "float32",
            "model_input_range": [0.0, 1.0],
            "normalization_mode": "identity_before_smolvla_resize_and_minus1_plus1_transform",
        },
        "gripper_semantics": "Panda mimic joint position target; final action component",
        "state_units": "7 joint positions plus 2 gripper positions in radians/meters",
        "action_units": "7 joint-position targets plus 1 gripper mimic command",
        "artificial_noise_added": False,
    }
    return {
        "schema_version": PHASE2C_NORMALIZATION_SCHEMA,
        **semantic,
        "semantic_sha256": f"sha256:{sha256_hex(semantic)}",
        "passed": True,
    }


def normalization_round_trip(
    values: np.ndarray,
    statistics: Mapping[str, object],
) -> np.ndarray:
    """Normalize then invert values, using exact zero-variance handling for audit tests."""

    mean = np.asarray(statistics.get("mean"), dtype=np.float64)
    std = np.asarray(statistics.get("std"), dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    if array.shape[-1] != mean.size or std.shape != mean.shape:
        raise Phase2CContractError("normalization shape differs from feature statistics")
    scale = np.where(std > 0.0, std, 1.0)
    restored = ((array - mean) / scale) * scale + mean
    if not np.allclose(restored, array, rtol=0.0, atol=1e-12):
        raise Phase2CContractError("normalization round trip changed values")
    return restored


def feature_mapping_manifest() -> dict[str, object]:
    """Return the explicit deployable LangMani-to-SmolVLA feature contract."""

    semantic = {
        "inputs": {
            IMAGE_FEATURE_KEY: {
                "source": "base_camera RGB",
                "source_dtype": "uint8 or decoded float32",
                "source_shape": [3, 256, 256],
                "adapter_dtype": "float32",
                "adapter_range": [0.0, 1.0],
                "channel_order": "RGB/CHW",
                "official_processor": "resize-with-pad to 512x512 then map to [-1,1]",
            },
            STATE_FEATURE_KEY: {
                "schema": "PandaPolicyStateV0",
                "dtype": "float32",
                "shape": [9],
            },
            "task": {
                "source": "episode instruction or PolicyContext language_instruction",
                "dtype": "string",
                "official_processor": "append newline then tokenize to at most 48 tokens",
            },
        },
        "output": {
            ACTION_FEATURE_KEY: {
                "schema": "PandaJointPositionActionV0",
                "dtype": "float32",
                "shape": [8],
            }
        },
        "timestamp": "LeRobot training metadata only; not passed by the runtime adapter",
        "privileged_features": [],
    }
    return {
        "schema_version": PHASE2C_FEATURE_MAPPING_SCHEMA,
        **semantic,
        "semantic_sha256": f"sha256:{sha256_hex(semantic)}",
    }


def map_smolvla_observation(observation: ObservationBatch, *, task: str) -> dict[str, Any]:
    """Map one deployable observation to the official SmolVLA processor input."""

    if set(observation.features) != {IMAGE_FEATURE_KEY, STATE_FEATURE_KEY}:
        extra = sorted(set(observation.features) - {IMAGE_FEATURE_KEY, STATE_FEATURE_KEY})
        if any(
            fragment in key.lower() for key in extra for fragment in PRIVILEGED_FEATURE_FRAGMENTS
        ):
            raise PolicyContractError("privileged feature reached the SmolVLA adapter")
        raise PolicyContractError("SmolVLA observation keys differ from the frozen allowlist")
    image = observation.features[IMAGE_FEATURE_KEY]
    state = observation.features[STATE_FEATURE_KEY]
    if tuple(image.shape) != (3, 256, 256):
        raise PolicyContractError("SmolVLA image must have shape [3,256,256]")
    if image.dtype is torch.uint8:
        image = image.to(dtype=torch.float32).div(255.0)
    elif image.dtype is torch.float32:
        image = image.detach().clone()
    else:
        raise PolicyContractError("SmolVLA image must be uint8 or float32")
    if not bool(torch.isfinite(image).all()) or float(image.min()) < 0 or float(image.max()) > 1:
        raise PolicyContractError("SmolVLA image must be finite in [0,1]")
    if tuple(state.shape) != (9,) or state.dtype is not torch.float32:
        raise PolicyContractError("SmolVLA state must be float32[9]")
    if not bool(torch.isfinite(state).all()):
        raise PolicyContractError("SmolVLA state contains non-finite values")
    instruction = _string(task, "task")
    return {
        IMAGE_FEATURE_KEY: image,
        STATE_FEATURE_KEY: state.detach().clone(),
        "task": instruction,
    }


def _numpy(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def require_finite_manifest(value: Mapping[str, object]) -> None:
    """Reject NaN/Inf before a machine-readable Phase 2C artifact is persisted."""

    def walk(item: object) -> None:
        if isinstance(item, Mapping):
            for child in item.values():
                walk(child)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for child in item:
                walk(child)
        elif isinstance(item, float) and not math.isfinite(item):
            raise Phase2CContractError("manifest contains a non-finite number")

    walk(value)


__all__ = [
    "EXPECTED_PHASE2B_HASHES",
    "EXPECTED_SPLIT_COUNTS",
    "PHASE2B_SOURCE_COMMIT",
    "PHASE2C_FEATURE_MAPPING_SCHEMA",
    "PHASE2C_INPUT_SCHEMA",
    "PHASE2C_NORMALIZATION_SCHEMA",
    "PHASE2C_TRAIN_VIEW_SCHEMA",
    "Phase2CContractError",
    "StreamingMoments",
    "build_train_view_manifest",
    "compute_train_only_normalization",
    "feature_mapping_manifest",
    "map_smolvla_observation",
    "normalization_round_trip",
    "read_json_object",
    "require_finite_manifest",
    "sha256_file",
    "verify_phase2b_for_phase2c",
    "write_json_once",
]
