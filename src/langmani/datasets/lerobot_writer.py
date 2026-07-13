"""Narrow, stateful adapter around the installed LeRobot 0.6.0 writer.

LeRobot remains responsible for Parquet metadata and PyAV video encoding.  This
module adds only the lifecycle and staging guarantees that M3B needs: a writer
cannot be used after failure/finalization, successful finalization happens once,
and a staged directory is never confused with a completed derived dataset.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import shutil
from dataclasses import dataclass
from enum import StrEnum
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from langmani.datasets.lerobot_types import FeatureContract, VideoCodecConfig

_FINGERPRINT_PATTERN = re.compile(r"^sha256:([0-9a-f]{64})$")
_STAGING_MARKER = "langmani-m3b-staging-v1"
SIDECAR_DIRECTORY = "langmani"
COMPLETION_MARKER = Path(SIDECAR_DIRECTORY) / "complete.json"


class LeRobotWriterError(RuntimeError):
    """Raised when the public LeRobot writer or M3B lifecycle is incompatible."""


class WriterState(StrEnum):
    """Project-owned protection around LeRobot's permissive writer object."""

    OPEN = "open"
    FINALIZED = "finalized"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class LeRobotRuntimeInfo:
    """Exact installed writer/codec versions recorded in M3B provenance."""

    lerobot: str
    datasets: str
    pandas: str
    pyarrow: str
    av: str
    av_libraries: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "lerobot": self.lerobot,
            "datasets": self.datasets,
            "pandas": self.pandas,
            "pyarrow": self.pyarrow,
            "av": self.av,
            "av_libraries": dict(self.av_libraries),
        }


@dataclass(frozen=True, slots=True)
class StagingLayout:
    """A fingerprint-owned staging container and its not-yet-created dataset."""

    container: Path
    dataset_root: Path
    final_root: Path
    export_fingerprint: str


def inspect_lerobot_runtime() -> LeRobotRuntimeInfo:
    """Verify the installed 0.6.0 public surface and explicit PyAV backend."""
    try:
        import av
        from lerobot.datasets import LeRobotDataset
    except (ImportError, RuntimeError, OSError) as error:
        raise LeRobotWriterError(
            "LeRobot dataset APIs require lerobot[dataset]==0.6.0 and a working PyAV install"
        ) from error

    versions = {
        name: _distribution_version(name)
        for name in ("lerobot", "datasets", "pandas", "pyarrow", "av")
    }
    if versions["lerobot"] != "0.6.0":
        raise LeRobotWriterError(
            f"M3B supports exactly lerobot 0.6.0, found {versions['lerobot']!r}"
        )
    required_create = {
        "repo_id",
        "fps",
        "features",
        "root",
        "use_videos",
        "video_backend",
        "rgb_encoder",
        "data_files_size_in_mb",
        "video_files_size_in_mb",
    }
    create_parameters = set(inspect.signature(LeRobotDataset.create).parameters)
    missing = sorted(required_create - create_parameters)
    if missing:
        raise LeRobotWriterError(
            "installed LeRobotDataset.create is missing required parameters: " + ", ".join(missing)
        )
    for method_name, expected in (
        ("add_frame", {"self", "frame"}),
        ("save_episode", {"self", "episode_data", "parallel_encoding"}),
        ("finalize", {"self"}),
    ):
        actual = set(inspect.signature(getattr(LeRobotDataset, method_name)).parameters)
        if not expected <= actual:
            raise LeRobotWriterError(
                f"installed LeRobotDataset.{method_name} signature is unsupported"
            )
    libraries = tuple(
        sorted(
            (name, ".".join(str(part) for part in version))
            for name, version in av.library_versions.items()
        )
    )
    return LeRobotRuntimeInfo(
        lerobot=versions["lerobot"],
        datasets=versions["datasets"],
        pandas=versions["pandas"],
        pyarrow=versions["pyarrow"],
        av=versions["av"],
        av_libraries=libraries,
    )


class LeRobotWriterAdapter:
    """Enforce one-way writer state transitions around public LeRobot methods."""

    def __init__(self, dataset: Any, feature_contract: FeatureContract) -> None:
        self._dataset = dataset
        self._features = feature_contract
        self._state = WriterState.OPEN
        self._finalize_calls = 0
        self._saved_episodes = 0

    @classmethod
    def create(
        cls,
        *,
        root: Path,
        repo_id: str,
        fps: int,
        feature_contract: FeatureContract,
        codec: VideoCodecConfig,
        data_files_size_in_mb: int,
        video_files_size_in_mb: int,
    ) -> LeRobotWriterAdapter:
        """Create one write-only dataset at a child path that must not exist."""
        inspect_lerobot_runtime()
        if root.exists():
            raise LeRobotWriterError(f"staging dataset root must not already exist: {root}")
        root.parent.mkdir(parents=True, exist_ok=True)
        try:
            from lerobot.configs import RGBEncoderConfig
            from lerobot.datasets import LeRobotDataset

            rgb_encoder = RGBEncoderConfig(
                vcodec=codec.codec,
                pix_fmt=codec.pixel_format,
                g=codec.gop_size,
                crf=codec.crf,
                preset=codec.preset,
                fast_decode=codec.fast_decode,
                video_backend=codec.backend,
            )
            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=fps,
                features=feature_contract.to_lerobot_features(),
                root=root,
                robot_type="panda",
                use_videos=True,
                video_backend="pyav",
                rgb_encoder=rgb_encoder,
                encoder_threads=codec.encoder_threads,
                image_writer_processes=0,
                image_writer_threads=0,
                batch_encoding_size=1,
                streaming_encoding=False,
                data_files_size_in_mb=data_files_size_in_mb,
                video_files_size_in_mb=video_files_size_in_mb,
            )
        except Exception as error:
            raise LeRobotWriterError(
                f"failed to create staged LeRobotDataset: {type(error).__name__}: {error}"
            ) from error
        return cls(dataset, feature_contract)

    @property
    def state(self) -> WriterState:
        return self._state

    @property
    def saved_episodes(self) -> int:
        return self._saved_episodes

    @property
    def finalize_calls(self) -> int:
        return self._finalize_calls

    def add_policy_frame(
        self,
        *,
        rgb: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        task: str,
    ) -> None:
        """Add a fresh, strictly typed pre-action policy frame."""
        self._require_open("add_policy_frame")
        self._features.validate_frame(rgb=rgb, state=state, action=action)
        if not isinstance(task, str) or not task:
            raise LeRobotWriterError("task must be a non-empty canonical instruction")
        frame = {
            self._features.image_feature_key: rgb,
            self._features.state_feature_key: state,
            self._features.action_feature_key: action,
            "task": task,
        }
        try:
            self._dataset.add_frame(frame)
        except Exception as error:
            self._state = WriterState.FAILED
            raise LeRobotWriterError(
                f"LeRobot add_frame failed: {type(error).__name__}: {error}"
            ) from error

    def save_episode(self) -> None:
        """Commit the current episode exactly once through the public writer."""
        self._require_open("save_episode")
        if not self._dataset.has_pending_frames():
            raise LeRobotWriterError("cannot save an empty LeRobot episode")
        try:
            self._dataset.save_episode(parallel_encoding=False)
        except Exception as error:
            self._state = WriterState.FAILED
            raise LeRobotWriterError(
                f"LeRobot save_episode failed: {type(error).__name__}: {error}"
            ) from error
        self._saved_episodes += 1

    def finalize_once(self) -> None:
        """Finalize a successful writer once and permanently close project writes."""
        self._require_open("finalize_once")
        if self._dataset.has_pending_frames():
            raise LeRobotWriterError("cannot finalize with an unsaved episode")
        try:
            self._dataset.finalize()
        except Exception as error:
            self._state = WriterState.FAILED
            raise LeRobotWriterError(
                f"LeRobot finalize failed: {type(error).__name__}: {error}"
            ) from error
        self._finalize_calls += 1
        self._state = WriterState.FINALIZED

    def close_failed(self) -> tuple[str, ...]:
        """Best-effort public cleanup while preserving every secondary error."""
        if self._state is WriterState.FINALIZED:
            return ("writer was already finalized before failure cleanup",)
        errors: list[str] = []
        try:
            if self._dataset.has_pending_frames():
                self._dataset.clear_episode_buffer(delete_images=True)
        except Exception as error:  # cleanup must report, not replace, the primary failure
            errors.append(f"clear_episode_buffer: {type(error).__name__}: {error}")
        try:
            self._dataset.finalize()
            self._finalize_calls += 1
        except Exception as error:  # cleanup must report, not replace, the primary failure
            errors.append(f"finalize: {type(error).__name__}: {error}")
        self._state = WriterState.FAILED
        return tuple(errors)

    def _require_open(self, operation: str) -> None:
        if self._state is not WriterState.OPEN:
            raise LeRobotWriterError(
                f"cannot {operation} when writer state is {self._state.value!r}"
            )


def prepare_staging_layout(
    final_root: str | Path,
    export_fingerprint: str,
    *,
    clean_staging: bool = False,
) -> StagingLayout:
    """Reserve a fingerprint-owned, same-volume staging container.

    LeRobot 0.6.0/PyAV's Windows ffconcat path handling fails for non-ASCII
    dataset roots.  On that platform only, an explicit same-volume ASCII root is
    used so the later directory rename remains atomic.  The final dataset can be
    read normally from a Unicode path after promotion.
    """
    match = _FINGERPRINT_PATTERN.fullmatch(export_fingerprint)
    if match is None:
        raise LeRobotWriterError("export_fingerprint must be sha256:<64 lowercase hex>")
    destination = Path(final_root).resolve()
    if destination.exists():
        if not destination.is_dir() or (destination / COMPLETION_MARKER).exists():
            raise LeRobotWriterError(
                f"derived destination is immutable and already exists: {destination}"
            )
        owned = _read_export_fingerprint(destination) == export_fingerprint
        if not clean_staging:
            raise LeRobotWriterError(
                "incomplete promoted destination exists; rerun with explicit "
                f"clean-staging: {destination}"
            )
        if not owned:
            raise LeRobotWriterError(
                f"refusing to clean unowned incomplete destination: {destination}"
            )
        shutil.rmtree(destination)
    digest = match.group(1)
    staging_parent = _staging_parent(destination)
    container = staging_parent / digest
    dataset_root = container / "dataset"
    marker = container / "staging.json"
    if container.exists():
        owned = _read_staging_marker(marker) == {
            "export_fingerprint": export_fingerprint,
            "marker": _STAGING_MARKER,
        }
        if not clean_staging:
            raise LeRobotWriterError(
                f"incomplete staging exists; rerun with explicit clean-staging: {container}"
            )
        if not owned:
            raise LeRobotWriterError(f"refusing to clean unowned staging directory: {container}")
        shutil.rmtree(container)
    container.mkdir(parents=True, exist_ok=False)
    _write_json(marker, {"marker": _STAGING_MARKER, "export_fingerprint": export_fingerprint})
    return StagingLayout(
        container=container,
        dataset_root=dataset_root,
        final_root=destination,
        export_fingerprint=export_fingerprint,
    )


def promote_staged_dataset(layout: StagingLayout) -> Path:
    """Atomically rename a fully validated dataset into its immutable destination."""
    if not layout.dataset_root.is_dir():
        raise LeRobotWriterError("staged dataset root is missing")
    if layout.final_root.exists():
        raise LeRobotWriterError("destination appeared before atomic promotion")
    layout.final_root.parent.mkdir(parents=True, exist_ok=True)
    if layout.dataset_root.stat().st_dev != layout.final_root.parent.stat().st_dev:
        raise LeRobotWriterError("staging and destination must be on the same filesystem")
    try:
        os.replace(layout.dataset_root, layout.final_root)
    except OSError as error:
        raise LeRobotWriterError(f"atomic dataset promotion failed: {error}") from error
    return layout.final_root


def write_completion_marker(final_root: str | Path, export_fingerprint: str) -> Path:
    """Write the completion marker exclusively as the final mutating operation."""
    if _FINGERPRINT_PATTERN.fullmatch(export_fingerprint) is None:
        raise LeRobotWriterError("invalid export fingerprint for completion marker")
    root = Path(final_root).resolve()
    marker = root / COMPLETION_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "langmani-m3b-completion-v1",
        "export_fingerprint": export_fingerprint,
    }
    try:
        with marker.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise LeRobotWriterError("completion marker already exists") from error
    return marker


def _staging_parent(destination: Path) -> Path:
    parent = destination.parent / ".langmani-m3b-staging"
    if os.name != "nt" or str(parent).isascii():
        parent.mkdir(parents=True, exist_ok=True)
        return parent
    anchor = Path(destination.anchor)
    if not destination.anchor or not str(anchor).isascii():
        raise LeRobotWriterError(
            "Windows non-ASCII output requires an ASCII same-volume staging anchor"
        )
    parent = anchor / ".langmani-m3b-staging"
    parent.mkdir(parents=True, exist_ok=True)
    return parent


def _read_staging_marker(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_export_fingerprint(destination: Path) -> str | None:
    manifest = _read_staging_marker(destination / SIDECAR_DIRECTORY / "export_manifest.json")
    if manifest is None:
        return None
    value = manifest.get("export_fingerprint")
    return value if isinstance(value, str) else None


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError as error:
        raise LeRobotWriterError(f"required distribution is not installed: {name}") from error


__all__ = [
    "COMPLETION_MARKER",
    "SIDECAR_DIRECTORY",
    "LeRobotRuntimeInfo",
    "LeRobotWriterAdapter",
    "LeRobotWriterError",
    "StagingLayout",
    "WriterState",
    "inspect_lerobot_runtime",
    "prepare_staging_layout",
    "promote_staged_dataset",
    "write_completion_marker",
]
