from __future__ import annotations

import json
from pathlib import Path

import pytest

from langmani.datasets.identity import sha256_hex
from langmani.datasets.lerobot_types import (
    ACTION_FEATURE_KEY,
    IMAGE_FEATURE_KEY,
    STATE_FEATURE_KEY,
)
from langmani.v2.phase2c import EXPECTED_SPLIT_COUNTS, Phase2CContractError, sha256_file
from langmani.v2.phase2c_recovery import (
    ACCEPTED_DATASET_PACKAGE_SCHEMA,
    EXPECTED_ACTION_DIMENSION,
    EXPECTED_DATASET_ROOT_NAME,
    EXPECTED_STATE_DIMENSION,
    REQUIRED_ACCEPTANCE_FLAGS,
    AcceptedDatasetPackage,
    audit_dataset_candidate,
    build_asset_audit,
    dataset_tree_identity,
    restore_dataset_atomically,
    validate_lerobot_meta_info,
    verify_dataset_package,
)
from langmani.v2.push_dataset import DATASET_SPLITS


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _info(split: str) -> dict[str, object]:
    episodes, frames = EXPECTED_SPLIT_COUNTS[split]
    return {
        "codebase_version": "v3.0",
        "robot_type": "panda",
        "total_episodes": episodes,
        "total_frames": frames,
        "fps": 20,
        "features": {
            IMAGE_FEATURE_KEY: {
                "dtype": "video",
                "shape": [256, 256, 3],
                "names": ["height", "width", "channels"],
            },
            STATE_FEATURE_KEY: {"dtype": "float32", "shape": [9]},
            ACTION_FEATURE_KEY: {"dtype": "float32", "shape": [8]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }


def _dataset(root: Path) -> Path:
    for split in DATASET_SPLITS:
        split_root = root / "splits" / split
        _write_json(split_root / "meta" / "info.json", _info(split))
        (split_root / "meta" / "tasks.parquet").write_bytes(b"task-fixture")
        (split_root / "data").mkdir()
        (split_root / "data" / "chunk.parquet").write_bytes(f"data-{split}".encode())
        (split_root / "videos").mkdir()
        (split_root / "videos" / "chunk.mp4").write_bytes(f"video-{split}".encode())
    for name, value in {
        "export_manifest.json": {"repo_id_prefix": "langmani/phase2b-push-v1"},
        "dataset_schema.json": {"policy_features": {}},
        "split_manifest.json": {"counts": {}},
        "episodes.json": {"records": []},
        "complete.json": {"completed": True},
    }.items():
        _write_json(root / "langmani" / name, value)
    return root


def _package(root: Path) -> AcceptedDatasetPackage:
    meta = validate_lerobot_meta_info(root)
    tree = dataset_tree_identity(root)
    evidence = {flag: True for flag in REQUIRED_ACCEPTANCE_FLAGS}
    semantic = {
        "dataset_id": "langmani/phase2b-push-v1",
        "source_phase": "Phase 2B",
        "source_commit": "7" * 40,
        "manifest_sha256": f"sha256:{sha256_file(root / 'langmani' / 'export_manifest.json')}",
        "meta_info_sha256": meta["meta_info_sha256"],
        "tree_digest": tree.tree_digest,
        "file_count": tree.file_count,
        "total_bytes": tree.total_bytes,
        "episode_count": meta["episode_count"],
        "frame_count": meta["frame_count"],
        "fps": meta["fps"],
        "camera_keys": meta["camera_keys"],
        "state_dimension": EXPECTED_STATE_DIMENSION,
        "action_dimension": EXPECTED_ACTION_DIMENSION,
        "lerobot_version": "0.6.0",
        "acceptance_evidence": evidence,
    }
    return AcceptedDatasetPackage(
        dataset_id="langmani/phase2b-push-v1",
        source_phase="Phase 2B",
        source_commit="7" * 40,
        dataset_root=root.as_posix(),
        manifest_sha256=str(semantic["manifest_sha256"]),
        meta_info_sha256=str(meta["meta_info_sha256"]),
        tree_digest=tree.tree_digest,
        file_count=tree.file_count,
        total_bytes=tree.total_bytes,
        episode_count=int(meta["episode_count"]),
        frame_count=int(meta["frame_count"]),
        fps=float(meta["fps"]),
        camera_keys=tuple(meta["camera_keys"]),
        state_dimension=EXPECTED_STATE_DIMENSION,
        action_dimension=EXPECTED_ACTION_DIMENSION,
        lerobot_version="0.6.0",
        acceptance_evidence=evidence,
        semantic_sha256=f"sha256:{sha256_hex(semantic)}",
    )


def test_accepted_dataset_package_hash_round_trip(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "source")
    package = _package(root)
    restored = AcceptedDatasetPackage.from_dict(package.to_dict())
    assert restored == package
    assert restored.to_dict()["schema_version"] == ACCEPTED_DATASET_PACKAGE_SCHEMA
    verify_dataset_package(restored)


def test_package_rejects_changed_semantic_hash(tmp_path: Path) -> None:
    value = _package(_dataset(tmp_path / "source")).to_dict()
    value["semantic_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(Phase2CContractError, match="semantic digest changed"):
        AcceptedDatasetPackage.from_dict(value)


def test_missing_meta_info_fails_closed(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "source")
    (root / "splits" / "train" / "meta" / "info.json").unlink()
    with pytest.raises(Phase2CContractError, match="meta/info.json is missing"):
        validate_lerobot_meta_info(root)


def test_wrong_episode_or_frame_count_is_rejected(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "source")
    path = root / "splits" / "train" / "meta" / "info.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    info["total_frames"] += 1
    _write_json(path, info)
    with pytest.raises(Phase2CContractError, match="frame count differs"):
        validate_lerobot_meta_info(root)


def test_camera_key_mismatch_is_rejected(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "source")
    path = root / "splits" / "validation" / "meta" / "info.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    info["features"]["observation.images.other"] = info["features"].pop(IMAGE_FEATURE_KEY)
    _write_json(path, info)
    with pytest.raises(Phase2CContractError, match="camera keys differ"):
        validate_lerobot_meta_info(root)


@pytest.mark.parametrize(
    ("key", "shape", "message"),
    [
        (STATE_FEATURE_KEY, [8], "observation.state shape differs"),
        (ACTION_FEATURE_KEY, [7], "action shape differs"),
    ],
)
def test_state_and_action_dimensions_are_rejected(
    tmp_path: Path, key: str, shape: list[int], message: str
) -> None:
    root = _dataset(tmp_path / "source")
    path = root / "splits" / "test_hard" / "meta" / "info.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    info["features"][key]["shape"] = shape
    _write_json(path, info)
    with pytest.raises(Phase2CContractError, match=message):
        validate_lerobot_meta_info(root)


def test_interrupted_copy_does_not_activate_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _dataset(tmp_path / "source")
    package = _package(source)
    destination = tmp_path / EXPECTED_DATASET_ROOT_NAME

    def fail_copy(source_path: Path, destination_path: Path, **_: object) -> None:
        del source_path
        destination_path.mkdir(parents=True)
        (destination_path / "partial.bin").write_bytes(b"partial")
        raise OSError("injected interruption")

    monkeypatch.setattr("langmani.v2.phase2c_recovery.shutil.copytree", fail_copy)
    with pytest.raises(OSError, match="injected interruption"):
        restore_dataset_atomically(
            source_root=source,
            destination_root=destination,
            package=package,
        )
    assert not destination.exists()
    assert not any(tmp_path.glob(f".{EXPECTED_DATASET_ROOT_NAME}.phase2c1-staging-*"))


def test_atomic_dataset_activation_preserves_identity(tmp_path: Path) -> None:
    source = _dataset(tmp_path / "source")
    package = _package(source)
    destination = tmp_path / EXPECTED_DATASET_ROOT_NAME
    restore_dataset_atomically(
        source_root=source,
        destination_root=destination,
        package=package,
    )
    assert destination.is_dir()
    verify_dataset_package(package, dataset_root=destination)


def test_asset_audit_does_not_promote_unverified_candidate(tmp_path: Path) -> None:
    candidate = audit_dataset_candidate(_dataset(tmp_path / "candidate"))
    assert candidate["compatibility_status"] == "POTENTIALLY_MATCHING_UNVERIFIED"
    audit = build_asset_audit([candidate])
    assert audit["accepted_phase_2b_identity"] is False
    assert audit["optimizer_steps_executed"] == 0
    assert audit["passed"] is False
