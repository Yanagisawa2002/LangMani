"""State-restored RGB export of validated Phase 2B push demonstrations."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from langmani.datasets.lerobot_types import FeatureContract, VideoCodecConfig
from langmani.datasets.lerobot_writer import (
    LeRobotWriterAdapter,
    inspect_lerobot_runtime,
)
from langmani.datasets.observation_reconstruction import extract_base_camera_rgb
from langmani.datasets.policy_state import extract_panda_policy_state_v0
from langmani.environments.push_specs import PushTaskSpec
from langmani.environments.push_to_region import ENV_ID
from langmani.v2.push_archive import load_native_actions, load_native_state, sha256_file
from langmani.v2.push_dataset import (
    DATASET_SPLITS,
    PushCollectionConfig,
    PushDatasetContractError,
    audit_split_leakage,
    build_split_manifest,
    sha256_json,
)

PUSH_LEROBOT_SCHEMA = "langmani-v2-phase2b-lerobot-v0"


def export_push_lerobot_dataset(
    *,
    config: PushCollectionConfig,
    source_root: str | Path,
    stages: Sequence[str],
    output_root: str | Path,
    sim_backend: str = "physx_cuda",
) -> dict[str, object]:
    """Export accepted actions with nonprivileged RGB and Panda policy state."""

    source = Path(source_root).resolve()
    destination = Path(output_root).resolve()
    if destination.exists():
        complete = destination / "langmani" / "complete.json"
        if complete.is_file():
            return _read_json(destination / "langmani" / "export_manifest.json")
        raise FileExistsError(f"incomplete export destination exists: {destination}")
    records = _load_accepted_records(source, stages)
    if not records:
        raise PushDatasetContractError("no independently replay-validated records to export")
    dataset_config = config.payload.get("dataset")
    if not isinstance(dataset_config, Mapping) or not isinstance(
        dataset_config.get("repo_id"), str
    ):
        raise PushDatasetContractError("collection dataset repo_id is malformed")
    repo_id_prefix = str(dataset_config["repo_id"])
    pilot = tuple(stages) == ("pilot",)
    if pilot:
        split_manifest: dict[str, object] = {
            "schema_version": "langmani-v2-phase2b-pilot-split-v0",
            "assignments": {str(record["episode_id"]): "pilot" for record in records},
            "counts": {"pilot": len(records)},
            "fingerprint": sha256_json([record["episode_id"] for record in records]),
        }
        leakage = {
            "schema_version": "langmani-v2-phase2b-pilot-leakage-v0",
            "passed": len({record["episode_id"] for record in records}) == len(records),
            "errors": [],
            "accepted_episode_count": len(records),
        }
        split_names = ("pilot",)
    else:
        split_manifest = build_split_manifest(records)
        leakage = audit_split_leakage(records, split_manifest)
        if leakage["passed"] is not True:
            raise PushDatasetContractError("split leakage audit rejected accepted records")
        split_names = DATASET_SPLITS
    assignments = cast(Mapping[str, object], split_manifest["assignments"])
    export_identity = {
        "schema_version": PUSH_LEROBOT_SCHEMA,
        "collection_fingerprint": config.fingerprint,
        "stages": list(stages),
        "accepted_fingerprints": [record["final_record_fingerprint"] for record in records],
        "split_fingerprint": split_manifest["fingerprint"],
        "feature_contract": FeatureContract().to_dict(),
    }
    export_fingerprint = sha256_json(export_identity)
    staging = destination.parent / f".{destination.name}.staging-{export_fingerprint[-12:]}"
    if staging.exists():
        raise FileExistsError(f"owned staging already exists: {staging}")
    staging.mkdir(parents=True)
    runtime = inspect_lerobot_runtime()
    started = time.perf_counter()
    episode_sidecars: list[dict[str, object]] = []
    statistics = _StreamingStatistics()
    try:
        with PushObservationReconstructor(sim_backend=sim_backend) as reconstructor:
            for split in split_names:
                selected = [
                    record for record in records if assignments[str(record["episode_id"])] == split
                ]
                writer = LeRobotWriterAdapter.create(
                    root=staging / "splits" / split,
                    repo_id=f"{repo_id_prefix}-{split}",
                    fps=20,
                    feature_contract=FeatureContract(),
                    codec=VideoCodecConfig(),
                    data_files_size_in_mb=100,
                    video_files_size_in_mb=200,
                )
                for split_episode_index, record in enumerate(selected):
                    task_value = record.get("task_spec")
                    if not isinstance(task_value, Mapping):
                        raise PushDatasetContractError("accepted task_spec is malformed")
                    task = PushTaskSpec.from_mapping(task_value)
                    seed = _integer(record.get("seed"), "seed")
                    native_id = _integer(record.get("native_episode_id"), "native_episode_id")
                    actions = load_native_actions(
                        str(record["raw_h5_path"]), native_episode_id=native_id
                    )
                    reconstructor.begin_episode(seed=seed, task=task)
                    rendered_hash = _RenderedFrameHash()
                    for frame_index, action in enumerate(actions):
                        state_dict = load_native_state(
                            str(record["raw_h5_path"]),
                            native_episode_id=native_id,
                            state_index=frame_index,
                        )
                        rgb, state = reconstructor.reconstruct(state_dict)
                        writer.add_policy_frame(
                            rgb=rgb,
                            state=state,
                            action=action,
                            task=str(record["instruction"]),
                        )
                        rendered_hash.add(rgb)
                        statistics.add(rgb=rgb, state=state, action=action)
                    writer.save_episode()
                    episode_sidecars.append(
                        {
                            "episode_id": record["episode_id"],
                            "task_id": record["task_id"],
                            "task_spec": dict(task_value),
                            "instruction": record["instruction"],
                            "template_group": record["template_group"],
                            "template_id": record["template_id"],
                            "scene_group_id": record["scene_group_id"],
                            "seed": seed,
                            "split": split,
                            "lerobot_repo_id": f"{repo_id_prefix}-{split}",
                            "lerobot_episode_index": split_episode_index,
                            "frame_count": len(actions),
                            "raw_h5_path": record["raw_h5_path"],
                            "raw_h5_group": record["raw_h5_group"],
                            "trajectory_sha256": record["trajectory_sha256"],
                            "rendered_rgb_sha256": rendered_hash.hexdigest(),
                            "source_record_fingerprint": record["final_record_fingerprint"],
                        }
                    )
                writer.finalize_once()
                if writer.saved_episodes != len(selected):
                    raise PushDatasetContractError("LeRobot writer episode count changed")

        readback = _validate_real_readback(
            staging / "splits", split_names, episode_sidecars, repo_id_prefix=repo_id_prefix
        )
        sidecar = staging / "langmani"
        sidecar.mkdir(parents=True)
        schema = {
            "schema_version": PUSH_LEROBOT_SCHEMA,
            "policy_features": FeatureContract().to_lerobot_features(),
            "implicit_lerobot_fields": [
                "task",
                "episode_index",
                "frame_index",
                "timestamp",
            ],
            "sidecar_task_metadata": [
                "task_id",
                "task_spec",
                "instruction",
                "template_group",
                "template_id",
            ],
            "privileged_policy_fields": [],
            "raw_authority": "ManiSkill state_dict T+1 with exact action T",
            "image_source": "base_camera RGB reconstructed from state[t] without env.step",
        }
        _write_json(sidecar / "dataset_schema.json", schema)
        _write_json(sidecar / "split_manifest.json", split_manifest)
        _write_json(sidecar / "leakage_audit.json", leakage)
        _write_json(
            sidecar / "episodes.json",
            {"records": episode_sidecars, "fingerprint": sha256_json(episode_sidecars)},
        )
        stats = statistics.to_dict()
        _write_json(sidecar / "dataset_statistics.json", stats)
        export_manifest = {
            "schema_version": PUSH_LEROBOT_SCHEMA,
            "export_fingerprint": export_fingerprint,
            "collection_fingerprint": config.fingerprint,
            "repo_id_prefix": repo_id_prefix,
            "source_root": source.as_posix(),
            "stages": list(stages),
            "episode_count": len(records),
            "frame_count": statistics.frame_count,
            "split_counts": dict(Counter(item["split"] for item in episode_sidecars)),
            "feature_schema_sha256": sha256_file(sidecar / "dataset_schema.json"),
            "statistics_sha256": sha256_file(sidecar / "dataset_statistics.json"),
            "split_manifest_sha256": sha256_file(sidecar / "split_manifest.json"),
            "leakage_audit_sha256": sha256_file(sidecar / "leakage_audit.json"),
            "episodes_sha256": sha256_file(sidecar / "episodes.json"),
            "lerobot_runtime": runtime.to_dict(),
            "real_lerobot_readback": readback,
            "elapsed_seconds": time.perf_counter() - started,
            "completed": True,
        }
        _write_json(sidecar / "export_manifest.json", export_manifest)
        _fsync_tree(staging)
        os.replace(staging, destination)
        size = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
        complete = {
            "schema_version": "langmani-v2-phase2b-export-complete-v0",
            "export_fingerprint": export_fingerprint,
            "dataset_size_bytes": size,
            "export_manifest_sha256": sha256_file(
                destination / "langmani" / "export_manifest.json"
            ),
        }
        _write_json(destination / "langmani" / "complete.json", complete)
        return export_manifest
    except Exception:
        # Keep staging for forensic inspection. A caller must never mistake it for final data.
        raise


class PushObservationReconstructor:
    def __init__(self, *, sim_backend: str) -> None:
        self._sim_backend = sim_backend
        self._environment: Any | None = None
        self._base: Any | None = None
        self._active = False

    def __enter__(self) -> PushObservationReconstructor:
        import gymnasium as gym

        import langmani.environments  # noqa: F401

        self._environment = gym.make(
            ENV_ID,
            num_envs=1,
            obs_mode="rgb",
            reward_mode="none",
            control_mode="pd_joint_pos",
            render_mode=None,
            sim_backend=self._sim_backend,
            render_backend="sapien_cuda",
        )
        self._base = self._environment.unwrapped
        if self._base.control_freq != 20:
            raise PushDatasetContractError("push reconstruction requires 20Hz control")
        return self

    def __exit__(self, *args: object) -> None:
        del args
        if self._environment is not None:
            self._environment.close()
        self._environment = None
        self._base = None
        self._active = False

    def begin_episode(self, *, seed: int, task: PushTaskSpec) -> None:
        if self._environment is None or self._base is None:
            raise PushDatasetContractError("push reconstructor is not open")
        self._environment.reset(seed=seed, options={"task_spec": task.to_dict()})
        specs = self._base.get_episode_specs()
        if len(specs) != 1 or specs[0].task_spec != task or specs[0].scene_seed != seed:
            raise PushDatasetContractError("push reconstruction reset identity changed")
        self._active = True

    def reconstruct(self, state_dict: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        if not self._active or self._base is None:
            raise PushDatasetContractError("begin_episode must precede reconstruction")
        self._base.set_state_dict(state_dict)
        observation = self._base.get_obs()
        rgb = extract_base_camera_rgb(observation)
        state = extract_panda_policy_state_v0(self._base.agent.robot)
        FeatureContract().validate_frame(rgb=rgb, state=state, action=np.zeros(8, dtype=np.float32))
        return rgb, state


class _StreamingStatistics:
    def __init__(self) -> None:
        self.frame_count = 0
        self._state_sum = np.zeros(9, dtype=np.float64)
        self._state_sq = np.zeros(9, dtype=np.float64)
        self._state_min = np.full(9, np.inf)
        self._state_max = np.full(9, -np.inf)
        self._action_sum = np.zeros(8, dtype=np.float64)
        self._action_sq = np.zeros(8, dtype=np.float64)
        self._action_min = np.full(8, np.inf)
        self._action_max = np.full(8, -np.inf)
        self._image_sum = np.zeros(3, dtype=np.float64)
        self._image_sq = np.zeros(3, dtype=np.float64)
        self._image_pixels = 0

    def add(self, *, rgb: np.ndarray, state: np.ndarray, action: np.ndarray) -> None:
        self.frame_count += 1
        state64 = state.astype(np.float64)
        action64 = action.astype(np.float64)
        pixels = rgb.reshape(-1, 3).astype(np.float64) / 255.0
        self._state_sum += state64
        self._state_sq += state64**2
        self._state_min = np.minimum(self._state_min, state64)
        self._state_max = np.maximum(self._state_max, state64)
        self._action_sum += action64
        self._action_sq += action64**2
        self._action_min = np.minimum(self._action_min, action64)
        self._action_max = np.maximum(self._action_max, action64)
        self._image_sum += pixels.sum(axis=0)
        self._image_sq += (pixels**2).sum(axis=0)
        self._image_pixels += len(pixels)

    def to_dict(self) -> dict[str, object]:
        if self.frame_count < 1 or self._image_pixels < 1:
            raise PushDatasetContractError("cannot finalize empty dataset statistics")
        return {
            "schema_version": "langmani-v2-phase2b-statistics-v0",
            "frame_count": self.frame_count,
            "state": _moments(
                self._state_sum,
                self._state_sq,
                self._state_min,
                self._state_max,
                self.frame_count,
            ),
            "action": _moments(
                self._action_sum,
                self._action_sq,
                self._action_min,
                self._action_max,
                self.frame_count,
            ),
            "image_rgb_0_1": _moments(
                self._image_sum,
                self._image_sq,
                np.zeros(3),
                np.ones(3),
                self._image_pixels,
            ),
        }


class _RenderedFrameHash:
    def __init__(self) -> None:
        import hashlib

        self._digest = hashlib.sha256()

    def add(self, rgb: np.ndarray) -> None:
        self._digest.update(rgb.tobytes(order="C"))

    def hexdigest(self) -> str:
        return "sha256:" + self._digest.hexdigest()


def _validate_real_readback(
    splits_root: Path,
    split_names: Sequence[str],
    episodes: Sequence[Mapping[str, object]],
    *,
    repo_id_prefix: str,
) -> dict[str, object]:
    from lerobot.datasets import LeRobotDataset

    expected = Counter(str(item["split"]) for item in episodes)
    expected_frames = Counter()
    for item in episodes:
        expected_frames[str(item["split"])] += int(item["frame_count"])
    reports: dict[str, object] = {}
    for split in split_names:
        dataset = LeRobotDataset(
            repo_id=f"{repo_id_prefix}-{split}",
            root=splits_root / split,
            video_backend="pyav",
            return_uint8=True,
        )
        if dataset.num_episodes != expected[split] or dataset.num_frames != expected_frames[split]:
            raise PushDatasetContractError("real LeRobot readback counts changed")
        item = dataset[0]
        if set(FeatureContract().policy_feature_keys) - set(item):
            raise PushDatasetContractError("real LeRobot readback is missing policy fields")
        if item["observation.images.base_camera"].shape != (3, 256, 256):
            raise PushDatasetContractError("decoded base_camera shape changed")
        if item["observation.state"].shape != (9,) or item["action"].shape != (8,):
            raise PushDatasetContractError("decoded state/action shape changed")
        reports[split] = {
            "episode_count": dataset.num_episodes,
            "frame_count": dataset.num_frames,
            "sample_task": item["task"],
            "passed": True,
        }
    return {"splits": reports, "passed": True}


def _load_accepted_records(root: Path, stages: Sequence[str]) -> list[dict[str, object]]:
    name = "_".join(stages)
    value = _read_json(root / "accepted" / f"{name}.json")
    records = value.get("records")
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise PushDatasetContractError("accepted manifest records are malformed")
    return cast(list[dict[str, object]], records)


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PushDatasetContractError(f"{label} must be a non-negative integer")
    return value


def _moments(
    total: np.ndarray,
    square: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
    count: int,
) -> dict[str, list[float]]:
    mean = total / count
    std = np.sqrt(np.maximum(square / count - mean**2, 0.0))
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min": minimum.tolist(),
        "max": maximum.tolist(),
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PushDatasetContractError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())


__all__ = ["PUSH_LEROBOT_SCHEMA", "PushObservationReconstructor", "export_push_lerobot_dataset"]
