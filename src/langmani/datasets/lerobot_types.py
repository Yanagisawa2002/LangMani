"""Immutable, JSON-ready contracts for deterministic M3B LeRobot exports.

The module is deliberately independent of LeRobot's import-heavy dataset
package.  It describes the exact public writer inputs and the project-owned
sidecar records without creating datasets, importing codecs, or touching the
authoritative M3A archive.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Self, cast

import numpy as np

from langmani.datasets.identity import (
    COLLECTION_SCHEMA_VERSION,
    environment_version,
    sha256_hex,
)
from langmani.datasets.policy_state import PANDA_POLICY_STATE_COMPONENTS

LEROBOT_EXPORT_SCHEMA_VERSION = "langmani-m3b-lerobot-v1"
SUPPORTED_LEROBOT_VERSION = "0.6.0"
ENVIRONMENT_ID = "LangMani-PickPlaceByInstruction-v0"
CONTROL_MODE = "pd_joint_pos"
POLICY_CAMERA_NAME = "base_camera"
POLICY_FPS = 20
POLICY_IMAGE_HEIGHT = 256
POLICY_IMAGE_WIDTH = 256
IMAGE_FEATURE_KEY = "observation.images.base_camera"
STATE_FEATURE_KEY = "observation.state"
ACTION_FEATURE_KEY = "action"

PANDA_ACTION_COMPONENTS: tuple[str, ...] = (
    "panda_joint1_position_target",
    "panda_joint2_position_target",
    "panda_joint3_position_target",
    "panda_joint4_position_target",
    "panda_joint5_position_target",
    "panda_joint6_position_target",
    "panda_joint7_position_target",
    "panda_gripper_mimic_command",
)

POLICY_FEATURE_KEYS = frozenset({IMAGE_FEATURE_KEY, STATE_FEATURE_KEY, ACTION_FEATURE_KEY})
LEROBOT_MANAGED_FEATURE_KEYS = frozenset(
    {"timestamp", "frame_index", "episode_index", "index", "task_index"}
)

_SHA256_PATTERN = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_REPO_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


class ExportMode(StrEnum):
    """Supported M3B export sizes."""

    SMOKE = "smoke"
    FULL = "full"


class DatasetSplit(StrEnum):
    """Scene-level dataset partitions."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class EpisodeExportStatus(StrEnum):
    """Stable final status for one derived episode."""

    EXPORTED = "exported"
    FAILED = "failed"


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field_name} must be a non-empty string")
    return value


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool")
    return value


def _require_int(value: object, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < minimum:
        qualifier = "positive" if minimum == 1 else f">= {minimum}"
        raise ValueError(f"{field_name} must be {qualifier}")
    return value


def _require_float(
    value: object,
    field_name: str,
    *,
    minimum: float = 0.0,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    invalid = result <= minimum if strict_minimum else result < minimum
    if invalid:
        operator = ">" if strict_minimum else ">="
        raise ValueError(f"{field_name} must be {operator} {minimum}")
    return result


def _require_string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not all(isinstance(item, str) and item for item in value):
        raise TypeError(f"{field_name} must be a tuple of non-empty strings")
    return value


def _string_tuple_from_json(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise TypeError(f"{field_name} must be a list of non-empty strings")
    return tuple(value)


def _int_tuple_from_json(value: object, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list of integers")
    return tuple(_require_int(item, f"{field_name} item") for item in value)


def _shape(value: object, field_name: str, *, dimensions: int) -> tuple[int, ...]:
    if not isinstance(value, tuple) or len(value) != dimensions:
        raise TypeError(f"{field_name} must be a {dimensions}-element tuple")
    return tuple(_require_int(item, f"{field_name} item", minimum=1) for item in value)


def _shape_from_json(value: object, field_name: str, *, dimensions: int) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    return _shape(tuple(value), field_name, dimensions=dimensions)


def _float_tuple(value: object, field_name: str, *, length: int) -> tuple[float, ...]:
    if not isinstance(value, tuple) or len(value) != length:
        raise TypeError(f"{field_name} must be a {length}-element tuple")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{field_name} items must be real numbers")
        normalized = float(item)
        if not math.isfinite(normalized):
            raise ValueError(f"{field_name} items must be finite")
        result.append(normalized)
    return tuple(result)


def _float_tuple_from_json(value: object, field_name: str, *, length: int) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    return _float_tuple(tuple(value), field_name, length=length)


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field_name} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _string_mapping(value: object, field_name: str) -> Mapping[str, str]:
    source = _mapping(value, field_name)
    result = {key: _require_string(item, f"{field_name}[{key!r}]") for key, item in source.items()}
    return MappingProxyType(dict(sorted(result.items())))


def _count_mapping(value: object, field_name: str) -> Mapping[str, int]:
    source = _mapping(value, field_name)
    result = {key: _require_int(item, f"{field_name}[{key!r}]") for key, item in source.items()}
    return MappingProxyType(dict(sorted(result.items())))


def _exact_fields(value: Mapping[str, object], expected: set[str], type_name: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing fields: " + ", ".join(missing))
        if extra:
            details.append("unexpected fields: " + ", ".join(extra))
        raise ValueError(f"malformed {type_name} (" + "; ".join(details) + ")")


def _enum(value: object, enum_type: type[StrEnum], field_name: str) -> StrEnum:
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or {enum_type.__name__}")
    try:
        return enum_type(value)
    except ValueError as error:
        raise ValueError(f"unknown {field_name} {value!r}") from error


def _require_sha256(value: object, field_name: str, *, prefixed: bool = False) -> str:
    result = _require_string(value, field_name)
    if _SHA256_PATTERN.fullmatch(result) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    if prefixed and not result.startswith("sha256:"):
        raise ValueError(f"{field_name} must start with 'sha256:'")
    return result


def _require_relative_path(value: object, field_name: str) -> str:
    result = _require_string(value, field_name)
    if "\\" in result or ":" in result:
        raise ValueError(f"{field_name} must use a relative POSIX path")
    path = PurePosixPath(result)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{field_name} must be a safe relative POSIX path")
    return result


@dataclass(frozen=True, slots=True)
class VideoCodecConfig:
    """Exact deterministic RGB encoder settings passed through LeRobot 0.6.0."""

    codec: str = "h264"
    pixel_format: str = "yuv444p"
    crf: int = 18
    gop_size: int = 2
    preset: int | str = "medium"
    fast_decode: int = 0
    backend: str = "pyav"
    encoder_threads: int = 1

    def __post_init__(self) -> None:
        if self.codec != "h264":
            raise ValueError("M3B fixes the RGB codec to h264")
        if self.pixel_format != "yuv444p":
            raise ValueError("M3B fixes the RGB pixel format to yuv444p")
        _require_int(self.crf, "crf")
        _require_int(self.gop_size, "gop_size", minimum=1)
        if isinstance(self.preset, bool) or not isinstance(self.preset, (int, str)):
            raise TypeError("preset must be an integer or non-empty string")
        if isinstance(self.preset, str) and not self.preset:
            raise TypeError("preset must be an integer or non-empty string")
        _require_int(self.fast_decode, "fast_decode")
        if self.backend != "pyav":
            raise ValueError("M3B requires the explicit pyav video backend")
        _require_int(self.encoder_threads, "encoder_threads", minimum=1)

    def to_dict(self) -> dict[str, object]:
        return {
            "codec": self.codec,
            "pixel_format": self.pixel_format,
            "crf": self.crf,
            "gop_size": self.gop_size,
            "preset": self.preset,
            "fast_decode": self.fast_decode,
            "backend": self.backend,
            "encoder_threads": self.encoder_threads,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = _mapping(value, "VideoCodecConfig")
        _exact_fields(payload, set(cls().to_dict()), "VideoCodecConfig")
        return cls(
            codec=cast(str, payload["codec"]),
            pixel_format=cast(str, payload["pixel_format"]),
            crf=cast(int, payload["crf"]),
            gop_size=cast(int, payload["gop_size"]),
            preset=cast(int | str, payload["preset"]),
            fast_decode=cast(int, payload["fast_decode"]),
            backend=cast(str, payload["backend"]),
            encoder_threads=cast(int, payload["encoder_threads"]),
        )


@dataclass(frozen=True, slots=True)
class PolicyStateSchema:
    """Stable, nonprivileged Panda qpos contract used by M3B."""

    name: str = "PandaPolicyStateV0"
    components: tuple[str, ...] = PANDA_POLICY_STATE_COMPONENTS
    dtype: str = "float32"
    shape: tuple[int, ...] = (9,)

    def __post_init__(self) -> None:
        if self.name != "PandaPolicyStateV0":
            raise ValueError("policy-state schema must be PandaPolicyStateV0")
        if self.components != PANDA_POLICY_STATE_COMPONENTS:
            raise ValueError("PandaPolicyStateV0 components must use the stable active-joint order")
        if self.dtype != "float32":
            raise ValueError("PandaPolicyStateV0 dtype must be float32")
        if _shape(self.shape, "shape", dimensions=1) != (len(self.components),):
            raise ValueError("PandaPolicyStateV0 shape must match its components")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "components": list(self.components),
            "dtype": self.dtype,
            "shape": list(self.shape),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = _mapping(value, "PolicyStateSchema")
        _exact_fields(payload, {"name", "components", "dtype", "shape"}, "PolicyStateSchema")
        return cls(
            name=cast(str, payload["name"]),
            components=_string_tuple_from_json(payload["components"], "components"),
            dtype=cast(str, payload["dtype"]),
            shape=_shape_from_json(payload["shape"], "shape", dimensions=1),
        )


@dataclass(frozen=True, slots=True)
class FeatureContract:
    """Exact three-feature policy allowlist supplied to LeRobot's writer."""

    image_feature_key: str = IMAGE_FEATURE_KEY
    state_feature_key: str = STATE_FEATURE_KEY
    action_feature_key: str = ACTION_FEATURE_KEY
    image_shape: tuple[int, ...] = (POLICY_IMAGE_HEIGHT, POLICY_IMAGE_WIDTH, 3)
    state_shape: tuple[int, ...] = (9,)
    action_shape: tuple[int, ...] = (8,)
    image_dtype: str = "video"
    state_dtype: str = "float32"
    action_dtype: str = "float32"
    image_names: tuple[str, ...] = ("height", "width", "channels")
    state_names: tuple[str, ...] = PANDA_POLICY_STATE_COMPONENTS
    action_names: tuple[str, ...] = PANDA_ACTION_COMPONENTS

    def __post_init__(self) -> None:
        keys = {self.image_feature_key, self.state_feature_key, self.action_feature_key}
        if keys != POLICY_FEATURE_KEYS:
            raise ValueError("M3B policy features must be exactly the three-feature allowlist")
        if _shape(self.image_shape, "image_shape", dimensions=3) != (256, 256, 3):
            raise ValueError("base_camera feature shape must be (256, 256, 3)")
        if _shape(self.state_shape, "state_shape", dimensions=1) != (9,):
            raise ValueError("observation.state shape must be (9,)")
        if _shape(self.action_shape, "action_shape", dimensions=1) != (8,):
            raise ValueError("action shape must be (8,)")
        if (self.image_dtype, self.state_dtype, self.action_dtype) != (
            "video",
            "float32",
            "float32",
        ):
            raise ValueError("feature dtypes must be video, float32, float32")
        if self.image_names != ("height", "width", "channels"):
            raise ValueError("image_names must describe HWC ordering")
        if self.state_names != PANDA_POLICY_STATE_COMPONENTS:
            raise ValueError("state_names must match PandaPolicyStateV0")
        if self.action_names != PANDA_ACTION_COMPONENTS:
            raise ValueError("action_names must match the 8D pd_joint_pos schema")

    @property
    def policy_feature_keys(self) -> frozenset[str]:
        return frozenset({self.image_feature_key, self.state_feature_key, self.action_feature_key})

    def to_lerobot_features(self) -> dict[str, dict[str, object]]:
        """Return the exact feature mapping accepted by LeRobotDataset.create()."""
        return {
            self.image_feature_key: {
                "dtype": self.image_dtype,
                "shape": self.image_shape,
                "names": list(self.image_names),
            },
            self.state_feature_key: {
                "dtype": self.state_dtype,
                "shape": self.state_shape,
                "names": list(self.state_names),
            },
            self.action_feature_key: {
                "dtype": self.action_dtype,
                "shape": self.action_shape,
                "names": list(self.action_names),
            },
        }

    def validate_frame(
        self,
        *,
        rgb: object,
        state: object,
        action: object,
    ) -> None:
        """Validate one writer frame without converting, normalizing, or clipping it."""
        if not isinstance(rgb, np.ndarray):
            raise TypeError("rgb must be a numpy.ndarray")
        if rgb.dtype != np.dtype(np.uint8):
            raise ValueError(f"rgb dtype must be uint8, got {rgb.dtype}")
        if rgb.shape != self.image_shape:
            raise ValueError(f"rgb shape must be {self.image_shape}, got {rgb.shape}")

        if not isinstance(state, np.ndarray):
            raise TypeError("state must be a numpy.ndarray")
        if state.dtype != np.dtype(np.float32):
            raise ValueError(f"state dtype must be float32, got {state.dtype}")
        if state.shape != self.state_shape:
            raise ValueError(f"state shape must be {self.state_shape}, got {state.shape}")
        if not np.all(np.isfinite(state)):
            raise ValueError("state must contain only finite values")

        if not isinstance(action, np.ndarray):
            raise TypeError("action must be a numpy.ndarray")
        if action.dtype != np.dtype(np.float32):
            raise ValueError(f"action dtype must be float32, got {action.dtype}")
        if action.shape != self.action_shape:
            raise ValueError(f"action shape must be {self.action_shape}, got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("action must contain only finite values")

    def to_dict(self) -> dict[str, object]:
        return {
            "image_feature_key": self.image_feature_key,
            "state_feature_key": self.state_feature_key,
            "action_feature_key": self.action_feature_key,
            "image_shape": list(self.image_shape),
            "state_shape": list(self.state_shape),
            "action_shape": list(self.action_shape),
            "image_dtype": self.image_dtype,
            "state_dtype": self.state_dtype,
            "action_dtype": self.action_dtype,
            "image_names": list(self.image_names),
            "state_names": list(self.state_names),
            "action_names": list(self.action_names),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = _mapping(value, "FeatureContract")
        _exact_fields(payload, set(cls().to_dict()), "FeatureContract")
        return cls(
            image_feature_key=cast(str, payload["image_feature_key"]),
            state_feature_key=cast(str, payload["state_feature_key"]),
            action_feature_key=cast(str, payload["action_feature_key"]),
            image_shape=_shape_from_json(payload["image_shape"], "image_shape", dimensions=3),
            state_shape=_shape_from_json(payload["state_shape"], "state_shape", dimensions=1),
            action_shape=_shape_from_json(payload["action_shape"], "action_shape", dimensions=1),
            image_dtype=cast(str, payload["image_dtype"]),
            state_dtype=cast(str, payload["state_dtype"]),
            action_dtype=cast(str, payload["action_dtype"]),
            image_names=_string_tuple_from_json(payload["image_names"], "image_names"),
            state_names=_string_tuple_from_json(payload["state_names"], "state_names"),
            action_names=_string_tuple_from_json(payload["action_names"], "action_names"),
        )


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Scene-group counts and seed for one deterministic partition."""

    train_scene_groups: int = 48
    validation_scene_groups: int = 6
    test_scene_groups: int = 6
    split_seed: int = 0

    def __post_init__(self) -> None:
        _require_int(self.train_scene_groups, "train_scene_groups")
        _require_int(self.validation_scene_groups, "validation_scene_groups")
        _require_int(self.test_scene_groups, "test_scene_groups")
        _require_int(self.split_seed, "split_seed")
        if self.total_scene_groups < 1:
            raise ValueError("a split must contain at least one scene group")

    @property
    def total_scene_groups(self) -> int:
        return self.train_scene_groups + self.validation_scene_groups + self.test_scene_groups

    @classmethod
    def full(cls, *, split_seed: int = 0) -> Self:
        return cls(48, 6, 6, split_seed)

    @classmethod
    def smoke(cls, *, split_seed: int = 0) -> Self:
        return cls(1, 0, 0, split_seed)

    def to_dict(self) -> dict[str, int]:
        return {
            "train_scene_groups": self.train_scene_groups,
            "validation_scene_groups": self.validation_scene_groups,
            "test_scene_groups": self.test_scene_groups,
            "split_seed": self.split_seed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = _mapping(value, "SplitConfig")
        _exact_fields(payload, set(cls().to_dict()), "SplitConfig")
        return cls(
            train_scene_groups=cast(int, payload["train_scene_groups"]),
            validation_scene_groups=cast(int, payload["validation_scene_groups"]),
            test_scene_groups=cast(int, payload["test_scene_groups"]),
            split_seed=cast(int, payload["split_seed"]),
        )


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    """One scene group's deterministic digest rank and resulting split."""

    scene_group_id: str
    split: DatasetSplit
    digest: str
    rank: int

    def __post_init__(self) -> None:
        _require_string(self.scene_group_id, "scene_group_id")
        object.__setattr__(self, "split", _enum(self.split, DatasetSplit, "split"))
        _require_sha256(self.digest, "digest")
        _require_int(self.rank, "rank")

    def to_dict(self) -> dict[str, object]:
        return {
            "scene_group_id": self.scene_group_id,
            "split": self.split.value,
            "digest": self.digest,
            "rank": self.rank,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = _mapping(value, "SplitAssignment")
        _exact_fields(payload, {"scene_group_id", "split", "digest", "rank"}, "SplitAssignment")
        return cls(
            scene_group_id=cast(str, payload["scene_group_id"]),
            split=cast(DatasetSplit, _enum(payload["split"], DatasetSplit, "split")),
            digest=cast(str, payload["digest"]),
            rank=cast(int, payload["rank"]),
        )


@dataclass(frozen=True, slots=True)
class CameraContract:
    """Complete fixed M1 policy-camera geometry and rendering contract."""

    name: str = POLICY_CAMERA_NAME
    eye: tuple[float, ...] = (0.65, -0.75, 0.70)
    target: tuple[float, ...] = (-0.04, 0.0, 0.08)
    width: int = POLICY_IMAGE_WIDTH
    height: int = POLICY_IMAGE_HEIGHT
    fov_radians: float = 1.05
    near_plane_m: float = 0.01
    far_plane_m: float = 10.0
    shader_pack: str = "minimal"
    render_backend: str = "sapien_cuda"

    def __post_init__(self) -> None:
        if self.name != POLICY_CAMERA_NAME:
            raise ValueError(f"camera name must be {POLICY_CAMERA_NAME}")
        if _float_tuple(self.eye, "eye", length=3) != (0.65, -0.75, 0.70):
            raise ValueError("base_camera eye must match the M1 camera contract")
        if _float_tuple(self.target, "target", length=3) != (-0.04, 0.0, 0.08):
            raise ValueError("base_camera target must match the M1 camera contract")
        if (self.width, self.height) != (POLICY_IMAGE_WIDTH, POLICY_IMAGE_HEIGHT):
            raise ValueError("base_camera resolution must be 256 x 256")
        object.__setattr__(
            self,
            "fov_radians",
            _require_float(self.fov_radians, "fov_radians", strict_minimum=True),
        )
        object.__setattr__(
            self,
            "near_plane_m",
            _require_float(self.near_plane_m, "near_plane_m", strict_minimum=True),
        )
        object.__setattr__(
            self,
            "far_plane_m",
            _require_float(self.far_plane_m, "far_plane_m", strict_minimum=True),
        )
        if self.fov_radians != 1.05 or self.near_plane_m != 0.01 or self.far_plane_m != 10.0:
            raise ValueError("base_camera projection must match the M1 camera contract")
        if self.shader_pack != "minimal":
            raise ValueError("base_camera shader_pack must be minimal")
        if self.render_backend != "sapien_cuda":
            raise ValueError("M3B target rendering requires sapien_cuda")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "eye": list(self.eye),
            "target": list(self.target),
            "width": self.width,
            "height": self.height,
            "fov_radians": self.fov_radians,
            "near_plane_m": self.near_plane_m,
            "far_plane_m": self.far_plane_m,
            "shader_pack": self.shader_pack,
            "render_backend": self.render_backend,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = _mapping(value, "CameraContract")
        _exact_fields(payload, set(cls().to_dict()), "CameraContract")
        return cls(
            name=cast(str, payload["name"]),
            eye=_float_tuple_from_json(payload["eye"], "eye", length=3),
            target=_float_tuple_from_json(payload["target"], "target", length=3),
            width=cast(int, payload["width"]),
            height=cast(int, payload["height"]),
            fov_radians=cast(float, payload["fov_radians"]),
            near_plane_m=cast(float, payload["near_plane_m"]),
            far_plane_m=cast(float, payload["far_plane_m"]),
            shader_pack=cast(str, payload["shader_pack"]),
            render_backend=cast(str, payload["render_backend"]),
        )


@dataclass(frozen=True, slots=True)
class LeRobotExportConfig:
    """All semantic and operational inputs to one deterministic M3B export."""

    source_root: str
    output_root: str
    repo_id: str
    expected_source_collection_run_id: str
    expected_source_run_fingerprint: str
    expected_source_archive_digest: str
    expected_source_schema_version: str = COLLECTION_SCHEMA_VERSION
    mode: ExportMode = ExportMode.FULL
    feature_contract: FeatureContract = field(default_factory=FeatureContract)
    policy_state_schema: PolicyStateSchema = field(default_factory=PolicyStateSchema)
    video_codec: VideoCodecConfig = field(default_factory=VideoCodecConfig)
    split_config: SplitConfig = field(default_factory=SplitConfig.full)
    camera_contract: CameraContract = field(default_factory=CameraContract)
    environment_id: str = ENVIRONMENT_ID
    control_mode: str = CONTROL_MODE
    camera_name: str = POLICY_CAMERA_NAME
    fps: int = POLICY_FPS
    image_height: int = POLICY_IMAGE_HEIGHT
    image_width: int = POLICY_IMAGE_WIDTH
    data_files_size_in_mb: int = 100
    video_files_size_in_mb: int = 200
    overwrite_policy: str = "error"
    staging_policy: str = "restart"
    action_tolerance: float = 0.0
    state_tolerance: float = 1e-6
    timestamp_tolerance: float = 1e-4
    video_mean_absolute_error_tolerance: float = 5.0
    video_min_psnr_db: float = 30.0
    output_schema_version: str = LEROBOT_EXPORT_SCHEMA_VERSION
    lerobot_version: str = SUPPORTED_LEROBOT_VERSION
    pyav_version: str = "15.1.0"
    libavcodec_version: str = "61.19.101"

    def __post_init__(self) -> None:
        _require_string(self.source_root, "source_root")
        _require_string(self.output_root, "output_root")
        if _REPO_ID_PATTERN.fullmatch(self.repo_id) is None:
            raise ValueError("repo_id must have the form 'owner/dataset-name'")
        _require_string(self.expected_source_collection_run_id, "expected_source_collection_run_id")
        _require_sha256(
            self.expected_source_run_fingerprint,
            "expected_source_run_fingerprint",
            prefixed=True,
        )
        _require_sha256(
            self.expected_source_archive_digest,
            "expected_source_archive_digest",
            prefixed=True,
        )
        if self.expected_source_schema_version != COLLECTION_SCHEMA_VERSION:
            raise ValueError(
                f"expected_source_schema_version must be {COLLECTION_SCHEMA_VERSION!r}"
            )
        object.__setattr__(self, "mode", _enum(self.mode, ExportMode, "mode"))
        if not isinstance(self.feature_contract, FeatureContract):
            raise TypeError("feature_contract must be a FeatureContract")
        if not isinstance(self.policy_state_schema, PolicyStateSchema):
            raise TypeError("policy_state_schema must be a PolicyStateSchema")
        if not isinstance(self.video_codec, VideoCodecConfig):
            raise TypeError("video_codec must be a VideoCodecConfig")
        if not isinstance(self.split_config, SplitConfig):
            raise TypeError("split_config must be a SplitConfig")
        if not isinstance(self.camera_contract, CameraContract):
            raise TypeError("camera_contract must be a CameraContract")
        expected_split = SplitConfig.full(split_seed=self.split_config.split_seed)
        if self.mode is ExportMode.SMOKE:
            expected_split = SplitConfig.smoke(split_seed=self.split_config.split_seed)
        if self.split_config != expected_split:
            expected = "48/6/6" if self.mode is ExportMode.FULL else "1/0/0"
            raise ValueError(f"{self.mode.value} mode requires the {expected} scene-group split")
        if self.environment_id != ENVIRONMENT_ID:
            raise ValueError(f"environment_id must be {ENVIRONMENT_ID}")
        if self.control_mode != CONTROL_MODE:
            raise ValueError(f"control_mode must be {CONTROL_MODE}")
        if self.camera_name != POLICY_CAMERA_NAME:
            raise ValueError(f"camera_name must be {POLICY_CAMERA_NAME}")
        if self.fps != POLICY_FPS:
            raise ValueError(f"fps must be {POLICY_FPS}")
        if (self.image_height, self.image_width) != (POLICY_IMAGE_HEIGHT, POLICY_IMAGE_WIDTH):
            raise ValueError("output resolution must be 256 x 256")
        if self.feature_contract.image_shape != (self.image_height, self.image_width, 3):
            raise ValueError("feature contract image shape must match configured resolution")
        if (
            self.camera_contract.name != self.camera_name
            or self.camera_contract.height != self.image_height
            or self.camera_contract.width != self.image_width
        ):
            raise ValueError("camera contract must match configured name and resolution")
        if self.feature_contract.state_names != self.policy_state_schema.components:
            raise ValueError("feature contract state names must match policy-state schema")
        _require_int(self.data_files_size_in_mb, "data_files_size_in_mb", minimum=1)
        _require_int(self.video_files_size_in_mb, "video_files_size_in_mb", minimum=1)
        if self.overwrite_policy != "error":
            raise ValueError(
                "M3B complete destinations are immutable; overwrite_policy must be error"
            )
        if self.staging_policy != "restart":
            raise ValueError("M3B does not resume LeRobot writers; staging_policy must be restart")
        object.__setattr__(
            self, "action_tolerance", _require_float(self.action_tolerance, "action_tolerance")
        )
        if self.action_tolerance != 0.0:
            raise ValueError("action_tolerance must be 0.0 for exact raw-action equality")
        object.__setattr__(
            self, "state_tolerance", _require_float(self.state_tolerance, "state_tolerance")
        )
        object.__setattr__(
            self,
            "timestamp_tolerance",
            _require_float(self.timestamp_tolerance, "timestamp_tolerance", strict_minimum=True),
        )
        object.__setattr__(
            self,
            "video_mean_absolute_error_tolerance",
            _require_float(
                self.video_mean_absolute_error_tolerance,
                "video_mean_absolute_error_tolerance",
            ),
        )
        object.__setattr__(
            self,
            "video_min_psnr_db",
            _require_float(self.video_min_psnr_db, "video_min_psnr_db", strict_minimum=True),
        )
        if self.output_schema_version != LEROBOT_EXPORT_SCHEMA_VERSION:
            raise ValueError(f"output_schema_version must be {LEROBOT_EXPORT_SCHEMA_VERSION!r}")
        if self.lerobot_version != SUPPORTED_LEROBOT_VERSION:
            raise ValueError(f"lerobot_version must be {SUPPORTED_LEROBOT_VERSION}")
        _require_string(self.pyav_version, "pyav_version")
        _require_string(self.libavcodec_version, "libavcodec_version")

    def identity_dict(self) -> dict[str, object]:
        """Return every semantic input while excluding machine/operational paths."""
        payload = self.to_dict()
        for name in ("source_root", "output_root", "overwrite_policy", "staging_policy"):
            payload.pop(name)
        return payload

    def portable_dict(self) -> dict[str, object]:
        """Return a sidecar-safe config without machine-specific absolute paths."""
        payload = self.to_dict()
        payload["source_root"] = "<authoritative-m3a-source>"
        payload["output_root"] = "."
        return payload

    def to_dict(self) -> dict[str, object]:
        return {
            "source_root": self.source_root,
            "output_root": self.output_root,
            "repo_id": self.repo_id,
            "expected_source_collection_run_id": self.expected_source_collection_run_id,
            "expected_source_run_fingerprint": self.expected_source_run_fingerprint,
            "expected_source_archive_digest": self.expected_source_archive_digest,
            "expected_source_schema_version": self.expected_source_schema_version,
            "mode": self.mode.value,
            "feature_contract": self.feature_contract.to_dict(),
            "policy_state_schema": self.policy_state_schema.to_dict(),
            "video_codec": self.video_codec.to_dict(),
            "split_config": self.split_config.to_dict(),
            "camera_contract": self.camera_contract.to_dict(),
            "environment_id": self.environment_id,
            "control_mode": self.control_mode,
            "camera_name": self.camera_name,
            "fps": self.fps,
            "image_height": self.image_height,
            "image_width": self.image_width,
            "data_files_size_in_mb": self.data_files_size_in_mb,
            "video_files_size_in_mb": self.video_files_size_in_mb,
            "overwrite_policy": self.overwrite_policy,
            "staging_policy": self.staging_policy,
            "action_tolerance": self.action_tolerance,
            "state_tolerance": self.state_tolerance,
            "timestamp_tolerance": self.timestamp_tolerance,
            "video_mean_absolute_error_tolerance": self.video_mean_absolute_error_tolerance,
            "video_min_psnr_db": self.video_min_psnr_db,
            "output_schema_version": self.output_schema_version,
            "lerobot_version": self.lerobot_version,
            "pyav_version": self.pyav_version,
            "libavcodec_version": self.libavcodec_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = _mapping(value, "LeRobotExportConfig")
        # The required constructor fields prevent instantiating cls() just to list fields.
        expected = {
            "source_root",
            "output_root",
            "repo_id",
            "expected_source_collection_run_id",
            "expected_source_run_fingerprint",
            "expected_source_archive_digest",
            "expected_source_schema_version",
            "mode",
            "feature_contract",
            "policy_state_schema",
            "video_codec",
            "split_config",
            "camera_contract",
            "environment_id",
            "control_mode",
            "camera_name",
            "fps",
            "image_height",
            "image_width",
            "data_files_size_in_mb",
            "video_files_size_in_mb",
            "overwrite_policy",
            "staging_policy",
            "action_tolerance",
            "state_tolerance",
            "timestamp_tolerance",
            "video_mean_absolute_error_tolerance",
            "video_min_psnr_db",
            "output_schema_version",
            "lerobot_version",
            "pyav_version",
            "libavcodec_version",
        }
        _exact_fields(payload, expected, "LeRobotExportConfig")
        return cls(
            source_root=cast(str, payload["source_root"]),
            output_root=cast(str, payload["output_root"]),
            repo_id=cast(str, payload["repo_id"]),
            expected_source_collection_run_id=cast(
                str, payload["expected_source_collection_run_id"]
            ),
            expected_source_run_fingerprint=cast(str, payload["expected_source_run_fingerprint"]),
            expected_source_archive_digest=cast(str, payload["expected_source_archive_digest"]),
            expected_source_schema_version=cast(str, payload["expected_source_schema_version"]),
            mode=cast(ExportMode, _enum(payload["mode"], ExportMode, "mode")),
            feature_contract=FeatureContract.from_dict(
                _mapping(payload["feature_contract"], "feature_contract")
            ),
            policy_state_schema=PolicyStateSchema.from_dict(
                _mapping(payload["policy_state_schema"], "policy_state_schema")
            ),
            video_codec=VideoCodecConfig.from_dict(_mapping(payload["video_codec"], "video_codec")),
            split_config=SplitConfig.from_dict(_mapping(payload["split_config"], "split_config")),
            camera_contract=CameraContract.from_dict(
                _mapping(payload["camera_contract"], "camera_contract")
            ),
            environment_id=cast(str, payload["environment_id"]),
            control_mode=cast(str, payload["control_mode"]),
            camera_name=cast(str, payload["camera_name"]),
            fps=cast(int, payload["fps"]),
            image_height=cast(int, payload["image_height"]),
            image_width=cast(int, payload["image_width"]),
            data_files_size_in_mb=cast(int, payload["data_files_size_in_mb"]),
            video_files_size_in_mb=cast(int, payload["video_files_size_in_mb"]),
            overwrite_policy=cast(str, payload["overwrite_policy"]),
            staging_policy=cast(str, payload["staging_policy"]),
            action_tolerance=cast(float, payload["action_tolerance"]),
            state_tolerance=cast(float, payload["state_tolerance"]),
            timestamp_tolerance=cast(float, payload["timestamp_tolerance"]),
            video_mean_absolute_error_tolerance=cast(
                float, payload["video_mean_absolute_error_tolerance"]
            ),
            video_min_psnr_db=cast(float, payload["video_min_psnr_db"]),
            output_schema_version=cast(str, payload["output_schema_version"]),
            lerobot_version=cast(str, payload["lerobot_version"]),
            pyav_version=cast(str, payload["pyav_version"]),
            libavcodec_version=cast(str, payload["libavcodec_version"]),
        )


def stable_export_fingerprint(
    config: LeRobotExportConfig, ordered_source_episode_ids: Sequence[str]
) -> str:
    """Fingerprint all semantic export inputs without machine-specific paths."""
    if not isinstance(config, LeRobotExportConfig):
        raise TypeError("config must be a LeRobotExportConfig")
    episode_ids = tuple(ordered_source_episode_ids)
    if not episode_ids or not all(isinstance(item, str) and item for item in episode_ids):
        raise TypeError("ordered_source_episode_ids must contain non-empty strings")
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("ordered_source_episode_ids must be unique")
    payload = {
        "config": config.identity_dict(),
        "environment_version": environment_version(config.environment_id),
        "ordered_source_episode_ids": list(episode_ids),
    }
    return f"sha256:{sha256_hex(payload)}"


@dataclass(frozen=True, slots=True)
class EpisodeExportRecord:
    """Complete one-to-one source-to-derived episode mapping."""

    source_collection_run_id: str
    source_run_fingerprint: str
    source_archive_digest: str
    source_episode_id: str
    source_scene_group_id: str
    source_scene_seed: int
    scene_id: str
    task_id: str
    target_object_id: str
    target_bin_id: str
    instruction_template_id: str
    canonical_instruction: str
    source_shard_id: str
    source_shard_path: str
    source_trajectory_key: str
    source_h5_sha256: str
    source_json_sha256: str
    source_checksum: str
    source_frame_count: int
    lerobot_episode_index: int
    split: DatasetSplit
    raw_render_digest: str
    output_frame_count: int
    status: EpisodeExportStatus = EpisodeExportStatus.EXPORTED

    def __post_init__(self) -> None:
        for name in (
            "source_collection_run_id",
            "source_run_fingerprint",
            "source_episode_id",
            "source_scene_group_id",
            "scene_id",
            "task_id",
            "target_object_id",
            "target_bin_id",
            "instruction_template_id",
            "canonical_instruction",
            "source_shard_id",
            "source_trajectory_key",
        ):
            _require_string(getattr(self, name), name)
        _require_sha256(self.source_run_fingerprint, "source_run_fingerprint", prefixed=True)
        _require_sha256(self.source_archive_digest, "source_archive_digest", prefixed=True)
        _require_int(self.source_scene_seed, "source_scene_seed")
        object.__setattr__(
            self,
            "source_shard_path",
            _require_relative_path(self.source_shard_path, "source_shard_path"),
        )
        _require_sha256(self.source_h5_sha256, "source_h5_sha256")
        _require_sha256(self.source_json_sha256, "source_json_sha256")
        _require_sha256(self.source_checksum, "source_checksum")
        _require_int(self.source_frame_count, "source_frame_count", minimum=1)
        _require_int(self.lerobot_episode_index, "lerobot_episode_index")
        object.__setattr__(self, "split", _enum(self.split, DatasetSplit, "split"))
        _require_sha256(self.raw_render_digest, "raw_render_digest")
        _require_int(self.output_frame_count, "output_frame_count", minimum=1)
        object.__setattr__(self, "status", _enum(self.status, EpisodeExportStatus, "status"))
        if (
            self.status is EpisodeExportStatus.EXPORTED
            and self.output_frame_count != self.source_frame_count
        ):
            raise ValueError("an exported episode must preserve the exact source frame count")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_collection_run_id": self.source_collection_run_id,
            "source_run_fingerprint": self.source_run_fingerprint,
            "source_archive_digest": self.source_archive_digest,
            "source_episode_id": self.source_episode_id,
            "source_scene_group_id": self.source_scene_group_id,
            "source_scene_seed": self.source_scene_seed,
            "scene_id": self.scene_id,
            "task_id": self.task_id,
            "target_object_id": self.target_object_id,
            "target_bin_id": self.target_bin_id,
            "instruction_template_id": self.instruction_template_id,
            "canonical_instruction": self.canonical_instruction,
            "source_shard_id": self.source_shard_id,
            "source_shard_path": self.source_shard_path,
            "source_trajectory_key": self.source_trajectory_key,
            "source_h5_sha256": self.source_h5_sha256,
            "source_json_sha256": self.source_json_sha256,
            "source_checksum": self.source_checksum,
            "source_frame_count": self.source_frame_count,
            "lerobot_episode_index": self.lerobot_episode_index,
            "split": self.split.value,
            "raw_render_digest": self.raw_render_digest,
            "output_frame_count": self.output_frame_count,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = _mapping(value, "EpisodeExportRecord")
        expected = {
            "source_collection_run_id",
            "source_run_fingerprint",
            "source_archive_digest",
            "source_episode_id",
            "source_scene_group_id",
            "source_scene_seed",
            "scene_id",
            "task_id",
            "target_object_id",
            "target_bin_id",
            "instruction_template_id",
            "canonical_instruction",
            "source_shard_id",
            "source_shard_path",
            "source_trajectory_key",
            "source_h5_sha256",
            "source_json_sha256",
            "source_checksum",
            "source_frame_count",
            "lerobot_episode_index",
            "split",
            "raw_render_digest",
            "output_frame_count",
            "status",
        }
        _exact_fields(payload, expected, "EpisodeExportRecord")
        return cls(
            source_collection_run_id=cast(str, payload["source_collection_run_id"]),
            source_run_fingerprint=cast(str, payload["source_run_fingerprint"]),
            source_archive_digest=cast(str, payload["source_archive_digest"]),
            source_episode_id=cast(str, payload["source_episode_id"]),
            source_scene_group_id=cast(str, payload["source_scene_group_id"]),
            source_scene_seed=cast(int, payload["source_scene_seed"]),
            scene_id=cast(str, payload["scene_id"]),
            task_id=cast(str, payload["task_id"]),
            target_object_id=cast(str, payload["target_object_id"]),
            target_bin_id=cast(str, payload["target_bin_id"]),
            instruction_template_id=cast(str, payload["instruction_template_id"]),
            canonical_instruction=cast(str, payload["canonical_instruction"]),
            source_shard_id=cast(str, payload["source_shard_id"]),
            source_shard_path=cast(str, payload["source_shard_path"]),
            source_trajectory_key=cast(str, payload["source_trajectory_key"]),
            source_h5_sha256=cast(str, payload["source_h5_sha256"]),
            source_json_sha256=cast(str, payload["source_json_sha256"]),
            source_checksum=cast(str, payload["source_checksum"]),
            source_frame_count=cast(int, payload["source_frame_count"]),
            lerobot_episode_index=cast(int, payload["lerobot_episode_index"]),
            split=cast(DatasetSplit, _enum(payload["split"], DatasetSplit, "split")),
            raw_render_digest=cast(str, payload["raw_render_digest"]),
            output_frame_count=cast(int, payload["output_frame_count"]),
            status=cast(
                EpisodeExportStatus,
                _enum(payload["status"], EpisodeExportStatus, "status"),
            ),
        )


@dataclass(frozen=True, slots=True)
class VideoValidationResult:
    """Independent structural and sampled-quality evidence for one episode video."""

    lerobot_episode_index: int
    video_path: str
    expected_frame_count: int
    decoded_frame_count: int
    decoded_height: int
    decoded_width: int
    sampled_frame_indices: tuple[int, ...]
    mean_absolute_error: float
    minimum_psnr_db: float
    frames_nonempty: bool
    frames_nonuniform: bool
    passed: bool
    failure_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_int(self.lerobot_episode_index, "lerobot_episode_index")
        object.__setattr__(
            self, "video_path", _require_relative_path(self.video_path, "video_path")
        )
        _require_int(self.expected_frame_count, "expected_frame_count", minimum=1)
        _require_int(self.decoded_frame_count, "decoded_frame_count")
        _require_int(self.decoded_height, "decoded_height", minimum=1)
        _require_int(self.decoded_width, "decoded_width", minimum=1)
        if not isinstance(self.sampled_frame_indices, tuple):
            raise TypeError("sampled_frame_indices must be a tuple")
        for index in self.sampled_frame_indices:
            _require_int(index, "sampled_frame_indices item")
            if index >= self.expected_frame_count:
                raise ValueError("sampled frame index is outside the episode")
        if len(set(self.sampled_frame_indices)) != len(self.sampled_frame_indices):
            raise ValueError("sampled_frame_indices must be unique")
        object.__setattr__(
            self,
            "mean_absolute_error",
            _require_float(self.mean_absolute_error, "mean_absolute_error"),
        )
        object.__setattr__(
            self,
            "minimum_psnr_db",
            _require_float(self.minimum_psnr_db, "minimum_psnr_db"),
        )
        for name in ("frames_nonempty", "frames_nonuniform", "passed"):
            _require_bool(getattr(self, name), name)
        _require_string_tuple(
            self.failure_reasons, "failure_reasons"
        ) if self.failure_reasons else ()
        if self.passed and self.failure_reasons:
            raise ValueError("a passing video result cannot contain failure reasons")
        if not self.passed and not self.failure_reasons:
            raise ValueError("a failing video result must contain failure reasons")

    def to_dict(self) -> dict[str, object]:
        return {
            "lerobot_episode_index": self.lerobot_episode_index,
            "video_path": self.video_path,
            "expected_frame_count": self.expected_frame_count,
            "decoded_frame_count": self.decoded_frame_count,
            "decoded_height": self.decoded_height,
            "decoded_width": self.decoded_width,
            "sampled_frame_indices": list(self.sampled_frame_indices),
            "mean_absolute_error": self.mean_absolute_error,
            "minimum_psnr_db": self.minimum_psnr_db,
            "frames_nonempty": self.frames_nonempty,
            "frames_nonuniform": self.frames_nonuniform,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = _mapping(value, "VideoValidationResult")
        expected = {
            "lerobot_episode_index",
            "video_path",
            "expected_frame_count",
            "decoded_frame_count",
            "decoded_height",
            "decoded_width",
            "sampled_frame_indices",
            "mean_absolute_error",
            "minimum_psnr_db",
            "frames_nonempty",
            "frames_nonuniform",
            "passed",
            "failure_reasons",
        }
        _exact_fields(payload, expected, "VideoValidationResult")
        return cls(
            lerobot_episode_index=cast(int, payload["lerobot_episode_index"]),
            video_path=cast(str, payload["video_path"]),
            expected_frame_count=cast(int, payload["expected_frame_count"]),
            decoded_frame_count=cast(int, payload["decoded_frame_count"]),
            decoded_height=cast(int, payload["decoded_height"]),
            decoded_width=cast(int, payload["decoded_width"]),
            sampled_frame_indices=_int_tuple_from_json(
                payload["sampled_frame_indices"], "sampled_frame_indices"
            ),
            mean_absolute_error=cast(float, payload["mean_absolute_error"]),
            minimum_psnr_db=cast(float, payload["minimum_psnr_db"]),
            frames_nonempty=cast(bool, payload["frames_nonempty"]),
            frames_nonuniform=cast(bool, payload["frames_nonuniform"]),
            passed=cast(bool, payload["passed"]),
            failure_reasons=_string_tuple_from_json(payload["failure_reasons"], "failure_reasons")
            if payload["failure_reasons"]
            else (),
        )


@dataclass(frozen=True, slots=True)
class SourceAlignmentResult:
    """Every independent source-to-derived alignment check for one episode."""

    lerobot_episode_index: int
    source_episode_id: str
    task_identity_matches: bool
    scene_identity_matches: bool
    frame_count_matches: bool
    actions_match: bool
    policy_states_match: bool
    raw_render_digest_matches: bool
    decoded_video_frame_count_matches: bool
    decoded_video_quality_passed: bool
    timestamps_match: bool
    passed: bool
    failure_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_int(self.lerobot_episode_index, "lerobot_episode_index")
        _require_string(self.source_episode_id, "source_episode_id")
        checks = (
            self.task_identity_matches,
            self.scene_identity_matches,
            self.frame_count_matches,
            self.actions_match,
            self.policy_states_match,
            self.raw_render_digest_matches,
            self.decoded_video_frame_count_matches,
            self.decoded_video_quality_passed,
            self.timestamps_match,
        )
        for name, value in zip(
            (
                "task_identity_matches",
                "scene_identity_matches",
                "frame_count_matches",
                "actions_match",
                "policy_states_match",
                "raw_render_digest_matches",
                "decoded_video_frame_count_matches",
                "decoded_video_quality_passed",
                "timestamps_match",
            ),
            checks,
            strict=True,
        ):
            _require_bool(value, name)
        _require_bool(self.passed, "passed")
        if self.passed != all(checks):
            raise ValueError("passed must equal the conjunction of all alignment checks")
        _require_string_tuple(
            self.failure_reasons, "failure_reasons"
        ) if self.failure_reasons else ()
        if self.passed and self.failure_reasons:
            raise ValueError("a passing alignment cannot contain failure reasons")
        if not self.passed and not self.failure_reasons:
            raise ValueError("a failing alignment must contain failure reasons")

    def to_dict(self) -> dict[str, object]:
        return {
            "lerobot_episode_index": self.lerobot_episode_index,
            "source_episode_id": self.source_episode_id,
            "task_identity_matches": self.task_identity_matches,
            "scene_identity_matches": self.scene_identity_matches,
            "frame_count_matches": self.frame_count_matches,
            "actions_match": self.actions_match,
            "policy_states_match": self.policy_states_match,
            "raw_render_digest_matches": self.raw_render_digest_matches,
            "decoded_video_frame_count_matches": self.decoded_video_frame_count_matches,
            "decoded_video_quality_passed": self.decoded_video_quality_passed,
            "timestamps_match": self.timestamps_match,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = _mapping(value, "SourceAlignmentResult")
        _exact_fields(payload, set(cls._field_names()), "SourceAlignmentResult")
        return cls(
            lerobot_episode_index=cast(int, payload["lerobot_episode_index"]),
            source_episode_id=cast(str, payload["source_episode_id"]),
            task_identity_matches=cast(bool, payload["task_identity_matches"]),
            scene_identity_matches=cast(bool, payload["scene_identity_matches"]),
            frame_count_matches=cast(bool, payload["frame_count_matches"]),
            actions_match=cast(bool, payload["actions_match"]),
            policy_states_match=cast(bool, payload["policy_states_match"]),
            raw_render_digest_matches=cast(bool, payload["raw_render_digest_matches"]),
            decoded_video_frame_count_matches=cast(
                bool, payload["decoded_video_frame_count_matches"]
            ),
            decoded_video_quality_passed=cast(bool, payload["decoded_video_quality_passed"]),
            timestamps_match=cast(bool, payload["timestamps_match"]),
            passed=cast(bool, payload["passed"]),
            failure_reasons=_string_tuple_from_json(payload["failure_reasons"], "failure_reasons")
            if payload["failure_reasons"]
            else (),
        )

    @staticmethod
    def _field_names() -> tuple[str, ...]:
        return (
            "lerobot_episode_index",
            "source_episode_id",
            "task_identity_matches",
            "scene_identity_matches",
            "frame_count_matches",
            "actions_match",
            "policy_states_match",
            "raw_render_digest_matches",
            "decoded_video_frame_count_matches",
            "decoded_video_quality_passed",
            "timestamps_match",
            "passed",
            "failure_reasons",
        )


@dataclass(frozen=True, slots=True)
class LeRobotExportManifest:
    """Final immutable sidecar index for one fully staged export."""

    export_schema_version: str
    export_fingerprint: str
    source_collection_run_id: str
    source_run_fingerprint: str
    source_archive_digest: str
    source_schema_version: str
    repo_id: str
    config: LeRobotExportConfig
    ordered_source_episode_ids: tuple[str, ...]
    split_assignments: tuple[SplitAssignment, ...]
    episodes: tuple[EpisodeExportRecord, ...]
    runtime_versions: Mapping[str, str]
    finalized: bool

    def __post_init__(self) -> None:
        if self.export_schema_version != LEROBOT_EXPORT_SCHEMA_VERSION:
            raise ValueError(f"export_schema_version must be {LEROBOT_EXPORT_SCHEMA_VERSION!r}")
        _require_sha256(self.export_fingerprint, "export_fingerprint", prefixed=True)
        _require_string(self.source_collection_run_id, "source_collection_run_id")
        _require_sha256(self.source_run_fingerprint, "source_run_fingerprint", prefixed=True)
        _require_sha256(self.source_archive_digest, "source_archive_digest", prefixed=True)
        if self.source_schema_version != COLLECTION_SCHEMA_VERSION:
            raise ValueError(f"source_schema_version must be {COLLECTION_SCHEMA_VERSION!r}")
        if not isinstance(self.config, LeRobotExportConfig):
            raise TypeError("config must be a LeRobotExportConfig")
        if self.repo_id != self.config.repo_id:
            raise ValueError("manifest repo_id must match export config")
        _require_string_tuple(self.ordered_source_episode_ids, "ordered_source_episode_ids")
        if len(set(self.ordered_source_episode_ids)) != len(self.ordered_source_episode_ids):
            raise ValueError("ordered_source_episode_ids must be unique")
        if not isinstance(self.split_assignments, tuple) or not all(
            isinstance(item, SplitAssignment) for item in self.split_assignments
        ):
            raise TypeError("split_assignments must be a tuple of SplitAssignment")
        if not isinstance(self.episodes, tuple) or not all(
            isinstance(item, EpisodeExportRecord) for item in self.episodes
        ):
            raise TypeError("episodes must be a tuple of EpisodeExportRecord")
        object.__setattr__(
            self, "runtime_versions", _string_mapping(self.runtime_versions, "runtime_versions")
        )
        _require_bool(self.finalized, "finalized")
        expected_fingerprint = stable_export_fingerprint(
            self.config, self.ordered_source_episode_ids
        )
        if self.export_fingerprint != expected_fingerprint:
            raise ValueError("export_fingerprint does not match config and source ordering")
        if self.source_run_fingerprint != self.config.expected_source_run_fingerprint:
            raise ValueError("source run fingerprint must match export config")
        if self.source_collection_run_id != self.config.expected_source_collection_run_id:
            raise ValueError("source collection run ID must match export config")
        if self.source_archive_digest != self.config.expected_source_archive_digest:
            raise ValueError("source archive digest must match export config")
        if (
            tuple(item.source_episode_id for item in self.episodes)
            != self.ordered_source_episode_ids
        ):
            raise ValueError("episode records must preserve ordered source episode IDs")
        if tuple(item.lerobot_episode_index for item in self.episodes) != tuple(
            range(len(self.episodes))
        ):
            raise ValueError("LeRobot episode indices must be contiguous and ordered")
        expected_episode_count = self.config.split_config.total_scene_groups * 6
        if len(self.episodes) != expected_episode_count:
            raise ValueError("manifest must contain six episodes per scene group")
        if len(self.split_assignments) != self.config.split_config.total_scene_groups:
            raise ValueError("manifest split assignment count does not match split config")
        assignment_by_group = {item.scene_group_id: item for item in self.split_assignments}
        if len(assignment_by_group) != len(self.split_assignments):
            raise ValueError("scene groups must have exactly one split assignment")
        for episode in self.episodes:
            if (
                episode.source_collection_run_id != self.source_collection_run_id
                or episode.source_run_fingerprint != self.source_run_fingerprint
                or episode.source_archive_digest != self.source_archive_digest
            ):
                raise ValueError("episode source provenance must match manifest provenance")
            assignment = assignment_by_group.get(episode.source_scene_group_id)
            if assignment is None or assignment.split is not episode.split:
                raise ValueError("episode split must match its scene-group assignment")
        if self.finalized and any(
            item.status is not EpisodeExportStatus.EXPORTED for item in self.episodes
        ):
            raise ValueError("a finalized manifest cannot contain failed episodes")

    def to_dict(self) -> dict[str, object]:
        return {
            "export_schema_version": self.export_schema_version,
            "export_fingerprint": self.export_fingerprint,
            "source_collection_run_id": self.source_collection_run_id,
            "source_run_fingerprint": self.source_run_fingerprint,
            "source_archive_digest": self.source_archive_digest,
            "source_schema_version": self.source_schema_version,
            "repo_id": self.repo_id,
            "config": self.config.portable_dict(),
            "ordered_source_episode_ids": list(self.ordered_source_episode_ids),
            "split_assignments": [item.to_dict() for item in self.split_assignments],
            "episodes": [item.to_dict() for item in self.episodes],
            "runtime_versions": dict(self.runtime_versions),
            "finalized": self.finalized,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = _mapping(value, "LeRobotExportManifest")
        expected = {
            "export_schema_version",
            "export_fingerprint",
            "source_collection_run_id",
            "source_run_fingerprint",
            "source_archive_digest",
            "source_schema_version",
            "repo_id",
            "config",
            "ordered_source_episode_ids",
            "split_assignments",
            "episodes",
            "runtime_versions",
            "finalized",
        }
        _exact_fields(payload, expected, "LeRobotExportManifest")
        assignments_value = payload["split_assignments"]
        episodes_value = payload["episodes"]
        if not isinstance(assignments_value, list) or not isinstance(episodes_value, list):
            raise TypeError("manifest split_assignments and episodes must be lists")
        return cls(
            export_schema_version=cast(str, payload["export_schema_version"]),
            export_fingerprint=cast(str, payload["export_fingerprint"]),
            source_collection_run_id=cast(str, payload["source_collection_run_id"]),
            source_run_fingerprint=cast(str, payload["source_run_fingerprint"]),
            source_archive_digest=cast(str, payload["source_archive_digest"]),
            source_schema_version=cast(str, payload["source_schema_version"]),
            repo_id=cast(str, payload["repo_id"]),
            config=LeRobotExportConfig.from_dict(_mapping(payload["config"], "config")),
            ordered_source_episode_ids=_string_tuple_from_json(
                payload["ordered_source_episode_ids"], "ordered_source_episode_ids"
            ),
            split_assignments=tuple(
                SplitAssignment.from_dict(_mapping(item, "split_assignment"))
                for item in assignments_value
            ),
            episodes=tuple(
                EpisodeExportRecord.from_dict(_mapping(item, "episode")) for item in episodes_value
            ),
            runtime_versions=cast(Mapping[str, str], payload["runtime_versions"]),
            finalized=cast(bool, payload["finalized"]),
        )


@dataclass(frozen=True, slots=True)
class LeRobotDatasetSummary:
    """Compact counts and schema facts for one derived local dataset."""

    export_fingerprint: str
    repo_id: str
    mode: ExportMode
    total_scene_groups: int
    total_episodes: int
    total_frames: int
    task_episode_counts: Mapping[str, int]
    split_scene_group_counts: Mapping[str, int]
    split_episode_counts: Mapping[str, int]
    feature_keys: tuple[str, ...]
    fps: int
    image_shape: tuple[int, ...]
    state_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    parquet_file_count: int
    video_file_count: int

    def __post_init__(self) -> None:
        _require_sha256(self.export_fingerprint, "export_fingerprint", prefixed=True)
        if _REPO_ID_PATTERN.fullmatch(self.repo_id) is None:
            raise ValueError("repo_id must have the form 'owner/dataset-name'")
        object.__setattr__(self, "mode", _enum(self.mode, ExportMode, "mode"))
        _require_int(self.total_scene_groups, "total_scene_groups", minimum=1)
        _require_int(self.total_episodes, "total_episodes", minimum=1)
        _require_int(self.total_frames, "total_frames", minimum=1)
        object.__setattr__(
            self,
            "task_episode_counts",
            _count_mapping(self.task_episode_counts, "task_episode_counts"),
        )
        object.__setattr__(
            self,
            "split_scene_group_counts",
            _count_mapping(self.split_scene_group_counts, "split_scene_group_counts"),
        )
        object.__setattr__(
            self,
            "split_episode_counts",
            _count_mapping(self.split_episode_counts, "split_episode_counts"),
        )
        if set(self.split_scene_group_counts) != {item.value for item in DatasetSplit}:
            raise ValueError("split_scene_group_counts must contain train, validation, and test")
        if set(self.split_episode_counts) != {item.value for item in DatasetSplit}:
            raise ValueError("split_episode_counts must contain train, validation, and test")
        if self.total_episodes != self.total_scene_groups * 6:
            raise ValueError("summary must contain six episodes per scene group")
        if sum(self.task_episode_counts.values()) != self.total_episodes:
            raise ValueError("task episode counts must sum to total_episodes")
        if len(self.task_episode_counts) != 6 or len(set(self.task_episode_counts.values())) != 1:
            raise ValueError("the six task combinations must be perfectly balanced")
        if sum(self.split_scene_group_counts.values()) != self.total_scene_groups:
            raise ValueError("split scene-group counts must sum to total_scene_groups")
        if sum(self.split_episode_counts.values()) != self.total_episodes:
            raise ValueError("split episode counts must sum to total_episodes")
        expected_split_episodes = {
            key: value * 6 for key, value in self.split_scene_group_counts.items()
        }
        if dict(self.split_episode_counts) != expected_split_episodes:
            raise ValueError("each split must contain all six tasks per scene group")
        if set(self.feature_keys) != POLICY_FEATURE_KEYS | LEROBOT_MANAGED_FEATURE_KEYS:
            raise ValueError("feature_keys do not match the policy and LeRobot-managed allowlist")
        if self.fps != POLICY_FPS:
            raise ValueError(f"fps must be {POLICY_FPS}")
        if _shape(self.image_shape, "image_shape", dimensions=3) != (256, 256, 3):
            raise ValueError("image_shape must be (256, 256, 3)")
        if _shape(self.state_shape, "state_shape", dimensions=1) != (9,):
            raise ValueError("state_shape must be (9,)")
        if _shape(self.action_shape, "action_shape", dimensions=1) != (8,):
            raise ValueError("action_shape must be (8,)")
        _require_int(self.parquet_file_count, "parquet_file_count", minimum=1)
        _require_int(self.video_file_count, "video_file_count", minimum=1)

    def to_dict(self) -> dict[str, object]:
        return {
            "export_fingerprint": self.export_fingerprint,
            "repo_id": self.repo_id,
            "mode": self.mode.value,
            "total_scene_groups": self.total_scene_groups,
            "total_episodes": self.total_episodes,
            "total_frames": self.total_frames,
            "task_episode_counts": dict(self.task_episode_counts),
            "split_scene_group_counts": dict(self.split_scene_group_counts),
            "split_episode_counts": dict(self.split_episode_counts),
            "feature_keys": list(self.feature_keys),
            "fps": self.fps,
            "image_shape": list(self.image_shape),
            "state_shape": list(self.state_shape),
            "action_shape": list(self.action_shape),
            "parquet_file_count": self.parquet_file_count,
            "video_file_count": self.video_file_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = _mapping(value, "LeRobotDatasetSummary")
        expected = {
            "export_fingerprint",
            "repo_id",
            "mode",
            "total_scene_groups",
            "total_episodes",
            "total_frames",
            "task_episode_counts",
            "split_scene_group_counts",
            "split_episode_counts",
            "feature_keys",
            "fps",
            "image_shape",
            "state_shape",
            "action_shape",
            "parquet_file_count",
            "video_file_count",
        }
        _exact_fields(payload, expected, "LeRobotDatasetSummary")
        return cls(
            export_fingerprint=cast(str, payload["export_fingerprint"]),
            repo_id=cast(str, payload["repo_id"]),
            mode=cast(ExportMode, _enum(payload["mode"], ExportMode, "mode")),
            total_scene_groups=cast(int, payload["total_scene_groups"]),
            total_episodes=cast(int, payload["total_episodes"]),
            total_frames=cast(int, payload["total_frames"]),
            task_episode_counts=cast(Mapping[str, int], payload["task_episode_counts"]),
            split_scene_group_counts=cast(Mapping[str, int], payload["split_scene_group_counts"]),
            split_episode_counts=cast(Mapping[str, int], payload["split_episode_counts"]),
            feature_keys=_string_tuple_from_json(payload["feature_keys"], "feature_keys"),
            fps=cast(int, payload["fps"]),
            image_shape=_shape_from_json(payload["image_shape"], "image_shape", dimensions=3),
            state_shape=_shape_from_json(payload["state_shape"], "state_shape", dimensions=1),
            action_shape=_shape_from_json(payload["action_shape"], "action_shape", dimensions=1),
            parquet_file_count=cast(int, payload["parquet_file_count"]),
            video_file_count=cast(int, payload["video_file_count"]),
        )


@dataclass(frozen=True, slots=True)
class LeRobotValidationReport:
    """Independent validation evidence for a completed derived dataset."""

    export_fingerprint: str
    mode: ExportMode
    dataset_load_validated: bool
    feature_schema_validated: bool
    parquet_validated: bool
    video_decode_validated: bool
    source_alignment_validated: bool
    split_integrity_validated: bool
    privileged_leakage_validated: bool
    dataloader_validated: bool
    source_alignments: tuple[SourceAlignmentResult, ...]
    video_results: tuple[VideoValidationResult, ...]
    passed: bool
    failure_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_sha256(self.export_fingerprint, "export_fingerprint", prefixed=True)
        object.__setattr__(self, "mode", _enum(self.mode, ExportMode, "mode"))
        flags = (
            self.dataset_load_validated,
            self.feature_schema_validated,
            self.parquet_validated,
            self.video_decode_validated,
            self.source_alignment_validated,
            self.split_integrity_validated,
            self.privileged_leakage_validated,
            self.dataloader_validated,
        )
        for name, value in zip(
            (
                "dataset_load_validated",
                "feature_schema_validated",
                "parquet_validated",
                "video_decode_validated",
                "source_alignment_validated",
                "split_integrity_validated",
                "privileged_leakage_validated",
                "dataloader_validated",
            ),
            flags,
            strict=True,
        ):
            _require_bool(value, name)
        if not isinstance(self.source_alignments, tuple) or not all(
            isinstance(item, SourceAlignmentResult) for item in self.source_alignments
        ):
            raise TypeError("source_alignments must be a tuple of SourceAlignmentResult")
        if not isinstance(self.video_results, tuple) or not all(
            isinstance(item, VideoValidationResult) for item in self.video_results
        ):
            raise TypeError("video_results must be a tuple of VideoValidationResult")
        _require_bool(self.passed, "passed")
        expected_passed = (
            all(flags)
            and all(item.passed for item in self.source_alignments)
            and all(item.passed for item in self.video_results)
        )
        if self.passed != expected_passed:
            raise ValueError("passed must equal all report, source-alignment, and video checks")
        _require_string_tuple(
            self.failure_reasons, "failure_reasons"
        ) if self.failure_reasons else ()
        if self.passed and self.failure_reasons:
            raise ValueError("a passing report cannot contain failure reasons")
        if not self.passed and not self.failure_reasons:
            raise ValueError("a failing report must contain failure reasons")

    def to_dict(self) -> dict[str, object]:
        return {
            "export_fingerprint": self.export_fingerprint,
            "mode": self.mode.value,
            "dataset_load_validated": self.dataset_load_validated,
            "feature_schema_validated": self.feature_schema_validated,
            "parquet_validated": self.parquet_validated,
            "video_decode_validated": self.video_decode_validated,
            "source_alignment_validated": self.source_alignment_validated,
            "split_integrity_validated": self.split_integrity_validated,
            "privileged_leakage_validated": self.privileged_leakage_validated,
            "dataloader_validated": self.dataloader_validated,
            "source_alignments": [item.to_dict() for item in self.source_alignments],
            "video_results": [item.to_dict() for item in self.video_results],
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = _mapping(value, "LeRobotValidationReport")
        expected = {
            "export_fingerprint",
            "mode",
            "dataset_load_validated",
            "feature_schema_validated",
            "parquet_validated",
            "video_decode_validated",
            "source_alignment_validated",
            "split_integrity_validated",
            "privileged_leakage_validated",
            "dataloader_validated",
            "source_alignments",
            "video_results",
            "passed",
            "failure_reasons",
        }
        _exact_fields(payload, expected, "LeRobotValidationReport")
        source_value = payload["source_alignments"]
        video_value = payload["video_results"]
        if not isinstance(source_value, list) or not isinstance(video_value, list):
            raise TypeError("source_alignments and video_results must be lists")
        return cls(
            export_fingerprint=cast(str, payload["export_fingerprint"]),
            mode=cast(ExportMode, _enum(payload["mode"], ExportMode, "mode")),
            dataset_load_validated=cast(bool, payload["dataset_load_validated"]),
            feature_schema_validated=cast(bool, payload["feature_schema_validated"]),
            parquet_validated=cast(bool, payload["parquet_validated"]),
            video_decode_validated=cast(bool, payload["video_decode_validated"]),
            source_alignment_validated=cast(bool, payload["source_alignment_validated"]),
            split_integrity_validated=cast(bool, payload["split_integrity_validated"]),
            privileged_leakage_validated=cast(bool, payload["privileged_leakage_validated"]),
            dataloader_validated=cast(bool, payload["dataloader_validated"]),
            source_alignments=tuple(
                SourceAlignmentResult.from_dict(_mapping(item, "source_alignment"))
                for item in source_value
            ),
            video_results=tuple(
                VideoValidationResult.from_dict(_mapping(item, "video_result"))
                for item in video_value
            ),
            passed=cast(bool, payload["passed"]),
            failure_reasons=_string_tuple_from_json(payload["failure_reasons"], "failure_reasons")
            if payload["failure_reasons"]
            else (),
        )


__all__ = [
    "ACTION_FEATURE_KEY",
    "CameraContract",
    "CONTROL_MODE",
    "ENVIRONMENT_ID",
    "IMAGE_FEATURE_KEY",
    "LEROBOT_EXPORT_SCHEMA_VERSION",
    "LEROBOT_MANAGED_FEATURE_KEYS",
    "POLICY_CAMERA_NAME",
    "POLICY_FEATURE_KEYS",
    "POLICY_FPS",
    "POLICY_IMAGE_HEIGHT",
    "POLICY_IMAGE_WIDTH",
    "PANDA_ACTION_COMPONENTS",
    "STATE_FEATURE_KEY",
    "SUPPORTED_LEROBOT_VERSION",
    "DatasetSplit",
    "EpisodeExportRecord",
    "EpisodeExportStatus",
    "ExportMode",
    "FeatureContract",
    "LeRobotDatasetSummary",
    "LeRobotExportConfig",
    "LeRobotExportManifest",
    "LeRobotValidationReport",
    "PolicyStateSchema",
    "SourceAlignmentResult",
    "SplitAssignment",
    "SplitConfig",
    "VideoCodecConfig",
    "VideoValidationResult",
    "stable_export_fingerprint",
]
