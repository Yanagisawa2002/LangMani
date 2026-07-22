"""Byte-bound recovery contracts for the accepted Phase 2B push dataset."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from langmani.datasets.identity import sha256_hex
from langmani.datasets.lerobot_types import (
    ACTION_FEATURE_KEY,
    IMAGE_FEATURE_KEY,
    STATE_FEATURE_KEY,
)
from langmani.v2.phase2c import (
    EXPECTED_PHASE2B_HASHES,
    EXPECTED_SPLIT_COUNTS,
    PHASE2B_SOURCE_COMMIT,
    Phase2CContractError,
    read_json_object,
    sha256_file,
    verify_phase2b_for_phase2c,
)
from langmani.v2.push_dataset import DATASET_SPLITS

ACCEPTED_DATASET_PACKAGE_SCHEMA = "langmani-v2-phase2c1-accepted-dataset-package-v0"
DATASET_ASSET_AUDIT_SCHEMA = "langmani-v2-phase2c1-dataset-asset-audit-v0"
EXPECTED_DATASET_ID = "langmani/phase2b-push-v1"
EXPECTED_DATASET_ROOT_NAME = "phase2b-push-lerobot-v1"
EXPECTED_LEROBOT_VERSION = "0.6.0"
EXPECTED_EPISODE_COUNT = 397
EXPECTED_FRAME_COUNT = 53_297
EXPECTED_SIZE_BYTES = 141_422_866
EXPECTED_FPS = 20.0
EXPECTED_CAMERA_KEYS = (IMAGE_FEATURE_KEY,)
EXPECTED_STATE_DIMENSION = 9
EXPECTED_ACTION_DIMENSION = 8
EXPECTED_META_INFO_FIELDS = frozenset(
    {
        "codebase_version",
        "robot_type",
        "total_episodes",
        "total_frames",
        "fps",
        "features",
    }
)
REQUIRED_ACCEPTANCE_FLAGS = frozenset(
    {
        "accepted_phase_2b_identity",
        "meta_info_present",
        "episode_count_matches",
        "frame_count_matches",
        "feature_schema_matches",
        "single_camera_contract_matches",
        "task_or_instruction_schema_matches",
        "tree_integrity_passed",
        "split_or_grouping_identity_matches",
    }
)
DatasetAssetStatus = Literal[
    "EXACT_ACCEPTED_DATASET",
    "POTENTIALLY_MATCHING_UNVERIFIED",
    "INCOMPLETE_COPY",
    "REGENERATED_OR_DIFFERENT",
    "WRONG_DATASET",
    "UNREADABLE",
    "NOT_FOUND",
]


def _prefixed_digest(path: Path) -> str:
    return f"sha256:{sha256_file(path)}"


def _require_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise Phase2CContractError(f"{label} must be a lowercase prefixed SHA-256")
    return value


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise Phase2CContractError(f"{label} must be a positive integer")
    return value


def _positive_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
        raise Phase2CContractError(f"{label} must be positive")
    return float(value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Phase2CContractError(f"{label} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class DatasetTreeIdentity:
    """Complete deterministic inventory for one external dataset directory."""

    file_count: int
    total_bytes: int
    tree_digest: str
    files: tuple[tuple[str, int, str], ...]

    def __post_init__(self) -> None:
        _positive_integer(self.file_count, "file_count")
        _positive_integer(self.total_bytes, "total_bytes")
        _require_digest(self.tree_digest, "tree_digest")
        if len(self.files) != self.file_count:
            raise Phase2CContractError("tree file inventory count changed")


@dataclass(frozen=True, slots=True)
class AcceptedDatasetPackage:
    """Read-only content identity for one independently accepted dataset root."""

    dataset_id: str
    source_phase: str
    source_commit: str
    dataset_root: str
    manifest_sha256: str
    meta_info_sha256: str
    tree_digest: str
    file_count: int
    total_bytes: int
    episode_count: int
    frame_count: int
    fps: float
    camera_keys: tuple[str, ...]
    state_dimension: int
    action_dimension: int
    lerobot_version: str
    acceptance_evidence: Mapping[str, object]
    semantic_sha256: str

    def __post_init__(self) -> None:
        for label, value in (
            ("dataset_id", self.dataset_id),
            ("source_phase", self.source_phase),
            ("source_commit", self.source_commit),
            ("dataset_root", self.dataset_root),
            ("lerobot_version", self.lerobot_version),
        ):
            _string(value, label)
        for label, value in (
            ("manifest_sha256", self.manifest_sha256),
            ("meta_info_sha256", self.meta_info_sha256),
            ("tree_digest", self.tree_digest),
            ("semantic_sha256", self.semantic_sha256),
        ):
            _require_digest(value, label)
        for label, integer_value in (
            ("file_count", self.file_count),
            ("total_bytes", self.total_bytes),
            ("episode_count", self.episode_count),
            ("frame_count", self.frame_count),
            ("state_dimension", self.state_dimension),
            ("action_dimension", self.action_dimension),
        ):
            _positive_integer(integer_value, label)
        _positive_float(self.fps, "fps")
        if not self.camera_keys or len(set(self.camera_keys)) != len(self.camera_keys):
            raise Phase2CContractError("camera_keys must be unique and non-empty")
        expected = _package_semantic(self)
        if self.semantic_sha256 != f"sha256:{sha256_hex(expected)}":
            raise Phase2CContractError("accepted dataset package semantic digest changed")

    def to_dict(self) -> dict[str, object]:
        """Return the exact JSON-ready package representation."""

        return {
            "schema_version": ACCEPTED_DATASET_PACKAGE_SCHEMA,
            "dataset_id": self.dataset_id,
            "source_phase": self.source_phase,
            "source_commit": self.source_commit,
            "dataset_root": self.dataset_root,
            "manifest_sha256": self.manifest_sha256,
            "meta_info_sha256": self.meta_info_sha256,
            "tree_digest": self.tree_digest,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "episode_count": self.episode_count,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "camera_keys": list(self.camera_keys),
            "state_dimension": self.state_dimension,
            "action_dimension": self.action_dimension,
            "lerobot_version": self.lerobot_version,
            "acceptance_evidence": dict(self.acceptance_evidence),
            "semantic_sha256": self.semantic_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> AcceptedDatasetPackage:
        """Load a package without accepting missing or additional fields."""

        expected = set(cls.__dataclass_fields__) | {"schema_version"}
        if set(value) != expected or value.get("schema_version") != ACCEPTED_DATASET_PACKAGE_SCHEMA:
            raise Phase2CContractError("accepted dataset package schema or fields changed")
        cameras = value["camera_keys"]
        evidence = value["acceptance_evidence"]
        if not isinstance(cameras, list) or not all(isinstance(item, str) for item in cameras):
            raise Phase2CContractError("camera_keys must be a list of strings")
        if not isinstance(evidence, Mapping):
            raise Phase2CContractError("acceptance_evidence must be an object")
        return cls(
            dataset_id=cast(str, value["dataset_id"]),
            source_phase=cast(str, value["source_phase"]),
            source_commit=cast(str, value["source_commit"]),
            dataset_root=cast(str, value["dataset_root"]),
            manifest_sha256=cast(str, value["manifest_sha256"]),
            meta_info_sha256=cast(str, value["meta_info_sha256"]),
            tree_digest=cast(str, value["tree_digest"]),
            file_count=cast(int, value["file_count"]),
            total_bytes=cast(int, value["total_bytes"]),
            episode_count=cast(int, value["episode_count"]),
            frame_count=cast(int, value["frame_count"]),
            fps=cast(float, value["fps"]),
            camera_keys=tuple(cast(list[str], cameras)),
            state_dimension=cast(int, value["state_dimension"]),
            action_dimension=cast(int, value["action_dimension"]),
            lerobot_version=cast(str, value["lerobot_version"]),
            acceptance_evidence=cast(Mapping[str, object], evidence),
            semantic_sha256=cast(str, value["semantic_sha256"]),
        )


def _package_semantic(package: AcceptedDatasetPackage) -> dict[str, object]:
    return {
        "dataset_id": package.dataset_id,
        "source_phase": package.source_phase,
        "source_commit": package.source_commit,
        "manifest_sha256": package.manifest_sha256,
        "meta_info_sha256": package.meta_info_sha256,
        "tree_digest": package.tree_digest,
        "file_count": package.file_count,
        "total_bytes": package.total_bytes,
        "episode_count": package.episode_count,
        "frame_count": package.frame_count,
        "fps": package.fps,
        "camera_keys": list(package.camera_keys),
        "state_dimension": package.state_dimension,
        "action_dimension": package.action_dimension,
        "lerobot_version": package.lerobot_version,
        "acceptance_evidence": dict(package.acceptance_evidence),
    }


def dataset_tree_identity(root: Path) -> DatasetTreeIdentity:
    """Hash every regular file by relative path, byte count, and content."""

    root = root.resolve()
    if not root.is_dir():
        raise Phase2CContractError(f"dataset root is unavailable: {root}")
    files: list[tuple[str, int, str]] = []
    total_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise Phase2CContractError(f"dataset trees must not contain symlinks: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = _prefixed_digest(path)
        files.append((relative, size, digest))
        total_bytes += size
    if not files:
        raise Phase2CContractError("dataset tree is empty")
    payload = [{"path": path, "size": size, "sha256": digest} for path, size, digest in files]
    return DatasetTreeIdentity(
        file_count=len(files),
        total_bytes=total_bytes,
        tree_digest=f"sha256:{sha256_hex(payload)}",
        files=tuple(files),
    )


def _meta_info_paths(root: Path) -> tuple[Path, ...]:
    paths = tuple(root / "splits" / split / "meta" / "info.json" for split in DATASET_SPLITS)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise Phase2CContractError(
            "required LeRobot meta/info.json is missing: "
            + ", ".join(path.as_posix() for path in missing)
        )
    return paths


def _meta_info_digest(paths: Sequence[Path], root: Path) -> str:
    payload = [
        {"path": path.relative_to(root).as_posix(), "sha256": _prefixed_digest(path)}
        for path in paths
    ]
    return f"sha256:{sha256_hex(payload)}"


def validate_lerobot_meta_info(root: Path) -> dict[str, object]:
    """Validate all six split metadata files against the accepted public contract."""

    root = root.resolve()
    paths = _meta_info_paths(root)
    total_episodes = 0
    total_frames = 0
    observed_fps: set[float] = set()
    camera_keys: set[str] = set()
    for split, path in zip(DATASET_SPLITS, paths, strict=True):
        info = read_json_object(path, f"{split} LeRobot meta/info.json")
        missing_fields = sorted(EXPECTED_META_INFO_FIELDS - set(info))
        if missing_fields:
            raise Phase2CContractError(
                f"{split} meta/info.json lacks required fields: {', '.join(missing_fields)}"
            )
        expected_episodes, expected_frames = EXPECTED_SPLIT_COUNTS[split]
        if info.get("total_episodes") != expected_episodes:
            raise Phase2CContractError(f"{split} episode count differs from accepted Phase 2B")
        if info.get("total_frames") != expected_frames:
            raise Phase2CContractError(f"{split} frame count differs from accepted Phase 2B")
        fps = _positive_float(info.get("fps"), f"{split}.fps")
        if fps != EXPECTED_FPS:
            raise Phase2CContractError(f"{split} fps differs from accepted Phase 2B")
        features = info.get("features")
        if not isinstance(features, Mapping):
            raise Phase2CContractError(f"{split} feature schema is malformed")
        image_features = {
            key
            for key, value in features.items()
            if isinstance(value, Mapping) and value.get("dtype") == "video"
        }
        if image_features != set(EXPECTED_CAMERA_KEYS):
            raise Phase2CContractError(f"{split} camera keys differ from accepted Phase 2B")
        image = features.get(IMAGE_FEATURE_KEY)
        if not isinstance(image, Mapping) or image.get("shape") != [256, 256, 3]:
            raise Phase2CContractError(f"{split} camera shape differs from accepted Phase 2B")
        _validate_feature(features, STATE_FEATURE_KEY, "float32", EXPECTED_STATE_DIMENSION)
        _validate_feature(features, ACTION_FEATURE_KEY, "float32", EXPECTED_ACTION_DIMENSION)
        task_index = features.get("task_index")
        if not isinstance(task_index, Mapping) or task_index.get("dtype") != "int64":
            raise Phase2CContractError(f"{split} task_index feature must be int64")
        if not (root / "splits" / split / "meta" / "tasks.parquet").is_file():
            raise Phase2CContractError(f"{split} task metadata file is missing")
        total_episodes += expected_episodes
        total_frames += expected_frames
        observed_fps.add(fps)
        camera_keys.update(image_features)
    if total_episodes != EXPECTED_EPISODE_COUNT or total_frames != EXPECTED_FRAME_COUNT:
        raise Phase2CContractError("aggregate split counts differ from accepted Phase 2B")
    return {
        "meta_info_sha256": _meta_info_digest(paths, root),
        "episode_count": total_episodes,
        "frame_count": total_frames,
        "fps": next(iter(observed_fps)),
        "camera_keys": sorted(camera_keys),
        "state_dimension": EXPECTED_STATE_DIMENSION,
        "action_dimension": EXPECTED_ACTION_DIMENSION,
        "task_schema": "LeRobot task/string",
    }


def _validate_feature(
    features: Mapping[str, object], key: str, expected_dtype: str, expected_dimension: int
) -> None:
    value = features.get(key)
    if not isinstance(value, Mapping) or value.get("dtype") != expected_dtype:
        raise Phase2CContractError(f"{key} dtype differs from the accepted contract")
    shape = value.get("shape")
    if shape != [expected_dimension]:
        raise Phase2CContractError(f"{key} shape differs from the accepted contract")


def build_accepted_dataset_package(
    *,
    dataset_root: Path,
    phase2b_result_path: Path,
    verifier_report_path: Path,
) -> AcceptedDatasetPackage:
    """Create a package only after all accepted Phase 2B gates pass again."""

    root = dataset_root.resolve()
    input_manifest = verify_phase2b_for_phase2c(
        export_root=root,
        phase2b_result_path=phase2b_result_path.resolve(),
        verifier_report_path=verifier_report_path.resolve(),
    )
    result = read_json_object(phase2b_result_path.resolve(), "Phase 2B result manifest")
    if result.get("lerobot_size_bytes") != EXPECTED_SIZE_BYTES:
        raise Phase2CContractError("accepted Phase 2B byte count changed")
    export_manifest_path = root / "langmani" / "export_manifest.json"
    export_manifest = read_json_object(export_manifest_path, "Phase 2B export manifest")
    runtime = export_manifest.get("lerobot_runtime")
    if not isinstance(runtime, Mapping) or runtime.get("lerobot") != EXPECTED_LEROBOT_VERSION:
        raise Phase2CContractError("LeRobot runtime differs from accepted Phase 2B")
    if export_manifest.get("repo_id_prefix") != EXPECTED_DATASET_ID:
        raise Phase2CContractError("dataset ID differs from accepted Phase 2B")
    meta = validate_lerobot_meta_info(root)
    tree = dataset_tree_identity(root)
    if result.get("lerobot_size_bytes") != tree.total_bytes:
        raise Phase2CContractError("dataset byte count differs from accepted Phase 2B")
    evidence = {
        "accepted_phase_2b_identity": True,
        "meta_info_present": True,
        "episode_count_matches": True,
        "frame_count_matches": True,
        "feature_schema_matches": True,
        "single_camera_contract_matches": True,
        "task_or_instruction_schema_matches": True,
        "tree_integrity_passed": True,
        "split_or_grouping_identity_matches": True,
        "phase2c_input_semantic_sha256": input_manifest["semantic_sha256"],
        "phase2b_result_sha256": _prefixed_digest(phase2b_result_path.resolve()),
        "phase2b_verifier_sha256": _prefixed_digest(verifier_report_path.resolve()),
        "accepted_sidecar_hashes": dict(EXPECTED_PHASE2B_HASHES),
    }
    semantic = {
        "dataset_id": EXPECTED_DATASET_ID,
        "source_phase": "Phase 2B",
        "source_commit": PHASE2B_SOURCE_COMMIT,
        "manifest_sha256": _prefixed_digest(export_manifest_path),
        "meta_info_sha256": meta["meta_info_sha256"],
        "tree_digest": tree.tree_digest,
        "file_count": tree.file_count,
        "total_bytes": tree.total_bytes,
        "episode_count": meta["episode_count"],
        "frame_count": meta["frame_count"],
        "fps": meta["fps"],
        "camera_keys": meta["camera_keys"],
        "state_dimension": meta["state_dimension"],
        "action_dimension": meta["action_dimension"],
        "lerobot_version": EXPECTED_LEROBOT_VERSION,
        "acceptance_evidence": evidence,
    }
    return AcceptedDatasetPackage(
        dataset_id=EXPECTED_DATASET_ID,
        source_phase="Phase 2B",
        source_commit=PHASE2B_SOURCE_COMMIT,
        dataset_root=root.as_posix(),
        manifest_sha256=cast(str, semantic["manifest_sha256"]),
        meta_info_sha256=cast(str, semantic["meta_info_sha256"]),
        tree_digest=tree.tree_digest,
        file_count=tree.file_count,
        total_bytes=tree.total_bytes,
        episode_count=cast(int, meta["episode_count"]),
        frame_count=cast(int, meta["frame_count"]),
        fps=cast(float, meta["fps"]),
        camera_keys=tuple(cast(list[str], meta["camera_keys"])),
        state_dimension=EXPECTED_STATE_DIMENSION,
        action_dimension=EXPECTED_ACTION_DIMENSION,
        lerobot_version=EXPECTED_LEROBOT_VERSION,
        acceptance_evidence=evidence,
        semantic_sha256=f"sha256:{sha256_hex(semantic)}",
    )


def verify_dataset_package(
    package: AcceptedDatasetPackage, *, dataset_root: Path | None = None
) -> None:
    """Rehash a source or restored copy without repairing any content."""

    root = (dataset_root or Path(package.dataset_root)).resolve()
    tree = dataset_tree_identity(root)
    meta = validate_lerobot_meta_info(root)
    checks = {
        "tree_digest": tree.tree_digest == package.tree_digest,
        "file_count": tree.file_count == package.file_count,
        "total_bytes": tree.total_bytes == package.total_bytes,
        "meta_info_sha256": meta["meta_info_sha256"] == package.meta_info_sha256,
        "episode_count": meta["episode_count"] == package.episode_count,
        "frame_count": meta["frame_count"] == package.frame_count,
        "fps": meta["fps"] == package.fps,
        "camera_keys": tuple(cast(list[str], meta["camera_keys"])) == package.camera_keys,
        "state_dimension": meta["state_dimension"] == package.state_dimension,
        "action_dimension": meta["action_dimension"] == package.action_dimension,
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise Phase2CContractError("dataset package verification failed: " + ", ".join(failed))
    for flag in REQUIRED_ACCEPTANCE_FLAGS:
        if package.acceptance_evidence.get(flag) is not True:
            raise Phase2CContractError(f"dataset package lacks acceptance flag: {flag}")


def restore_dataset_atomically(
    *, source_root: Path, destination_root: Path, package: AcceptedDatasetPackage
) -> None:
    """Copy an exact source through an unaddressed staging path, then atomically activate it."""

    source = source_root.resolve()
    destination = destination_root.resolve()
    if destination.name != EXPECTED_DATASET_ROOT_NAME:
        raise Phase2CContractError("restored dataset destination root name changed")
    if destination.exists():
        raise Phase2CContractError(f"refusing to overwrite dataset destination: {destination}")
    verify_dataset_package(package, dataset_root=source)
    staging = (
        destination.parent / f".{destination.name}.phase2c1-staging-{package.tree_digest[-12:]}"
    )
    if staging.exists():
        raise Phase2CContractError(f"owned recovery staging path already exists: {staging}")
    try:
        shutil.copytree(source, staging, copy_function=shutil.copy2)
        verify_dataset_package(package, dataset_root=staging)
        os.replace(staging, destination)
    except Exception:
        _remove_owned_staging(staging, destination.parent)
        raise
    verify_dataset_package(package, dataset_root=destination)


def _remove_owned_staging(staging: Path, parent: Path) -> None:
    resolved = staging.resolve()
    expected_parent = parent.resolve()
    if resolved.parent != expected_parent or not resolved.name.startswith(
        f".{EXPECTED_DATASET_ROOT_NAME}.phase2c1-staging-"
    ):
        raise Phase2CContractError("refusing to remove an unowned recovery path")
    if resolved.exists():
        shutil.rmtree(resolved)


def audit_dataset_candidate(
    location: Path,
    *,
    phase2b_result_path: Path | None = None,
    verifier_report_path: Path | None = None,
) -> dict[str, object]:
    """Classify one candidate using content evidence rather than its path name."""

    path = location.expanduser()
    base: dict[str, object] = {
        "location": path.as_posix(),
        "exists": path.exists(),
        "readable": False,
        "file_count": 0,
        "total_bytes": 0,
        "meta_files_present": 0,
        "episode_files_present": False,
        "video_or_image_files_present": False,
        "manifest_present": False,
        "candidate_identity": None,
        "compatibility_status": "NOT_FOUND",
        "failure": None,
    }
    if not path.exists():
        return base
    try:
        if path.is_file():
            base.update(
                {
                    "readable": True,
                    "file_count": 1,
                    "total_bytes": path.stat().st_size,
                    "compatibility_status": "POTENTIALLY_MATCHING_UNVERIFIED",
                    "failure": "archive bytes require explicit extraction into an unaddressed staging root",
                }
            )
            return base
        tree = dataset_tree_identity(path)
        file_names = {item[0] for item in tree.files}
        meta_count = sum(name.endswith("/meta/info.json") for name in file_names)
        manifests = {
            "langmani/export_manifest.json",
            "langmani/dataset_schema.json",
            "langmani/split_manifest.json",
            "langmani/episodes.json",
            "langmani/complete.json",
        }
        base.update(
            {
                "readable": True,
                "file_count": tree.file_count,
                "total_bytes": tree.total_bytes,
                "meta_files_present": meta_count,
                "episode_files_present": any("/data/" in f"/{name}" for name in file_names),
                "video_or_image_files_present": any(
                    "/videos/" in f"/{name}" or "/images/" in f"/{name}" for name in file_names
                ),
                "manifest_present": manifests.issubset(file_names),
                "candidate_identity": tree.tree_digest,
            }
        )
        if phase2b_result_path is not None and verifier_report_path is not None:
            package = build_accepted_dataset_package(
                dataset_root=path,
                phase2b_result_path=phase2b_result_path,
                verifier_report_path=verifier_report_path,
            )
            base["candidate_identity"] = package.semantic_sha256
            base["compatibility_status"] = "EXACT_ACCEPTED_DATASET"
        elif meta_count == len(DATASET_SPLITS) and manifests.issubset(file_names):
            base["compatibility_status"] = "POTENTIALLY_MATCHING_UNVERIFIED"
            base["failure"] = "accepted verifier evidence was not supplied"
        elif meta_count or manifests.intersection(file_names):
            base["compatibility_status"] = "INCOMPLETE_COPY"
            base["failure"] = "required split metadata or LangMani sidecars are incomplete"
        else:
            base["compatibility_status"] = "WRONG_DATASET"
            base["failure"] = "no accepted Phase 2B structure was found"
    except (OSError, Phase2CContractError, ValueError) as error:
        base["compatibility_status"] = (
            "UNREADABLE" if isinstance(error, OSError) else "REGENERATED_OR_DIFFERENT"
        )
        base["failure"] = f"{type(error).__name__}: {error}"
    return base


def build_asset_audit(candidates: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Summarize candidate records without turning absence into acceptance."""

    exact = sum(item.get("compatibility_status") == "EXACT_ACCEPTED_DATASET" for item in candidates)
    return {
        "schema_version": DATASET_ASSET_AUDIT_SCHEMA,
        "expected_dataset_id": EXPECTED_DATASET_ID,
        "expected_root_name": EXPECTED_DATASET_ROOT_NAME,
        "candidate_count": len(candidates),
        "exact_accepted_dataset_count": exact,
        "candidates": [dict(item) for item in candidates],
        "accepted_phase_2b_identity": exact == 1,
        "optimizer_steps_executed": 0,
        "passed": exact == 1,
    }


def load_dataset_package(path: Path) -> AcceptedDatasetPackage:
    """Load a package JSON and enforce its complete schema and semantic digest."""

    return AcceptedDatasetPackage.from_dict(read_json_object(path, "accepted dataset package"))


__all__ = [
    "ACCEPTED_DATASET_PACKAGE_SCHEMA",
    "DATASET_ASSET_AUDIT_SCHEMA",
    "EXPECTED_DATASET_ID",
    "EXPECTED_DATASET_ROOT_NAME",
    "EXPECTED_EPISODE_COUNT",
    "EXPECTED_FRAME_COUNT",
    "EXPECTED_FPS",
    "EXPECTED_LEROBOT_VERSION",
    "EXPECTED_META_INFO_FIELDS",
    "EXPECTED_SIZE_BYTES",
    "AcceptedDatasetPackage",
    "DatasetAssetStatus",
    "DatasetTreeIdentity",
    "audit_dataset_candidate",
    "build_accepted_dataset_package",
    "build_asset_audit",
    "dataset_tree_identity",
    "load_dataset_package",
    "restore_dataset_atomically",
    "validate_lerobot_meta_info",
    "verify_dataset_package",
]
