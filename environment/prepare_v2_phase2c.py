"""Verify Phase 2B and freeze the Phase 2C train view, statistics, and sanity artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from langmani.datasets.lerobot_types import (
    ACTION_FEATURE_KEY,
    IMAGE_FEATURE_KEY,
    STATE_FEATURE_KEY,
)
from langmani.v2.phase2c import (
    TRAIN_EPISODES,
    TRAIN_FRAMES,
    Phase2CContractError,
    build_train_view_manifest,
    compute_train_only_normalization,
    feature_mapping_manifest,
    normalization_round_trip,
    read_json_object,
    sha256_file,
    verify_phase2b_for_phase2c,
    write_json_once,
)
from langmani.v2.phase2c_schedule import build_phase2c_schedules

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = PROJECT_ROOT / "docs" / "langmani_v2" / "phase_2b_result_manifest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--phase2b-verifier-report", type=Path, required=True)
    parser.add_argument("--phase2b-result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sanity-samples", type=int, default=12)
    return parser.parse_args(argv)


def _dataset(train_root: Path, repo_id: str) -> object:
    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError as error:
        raise Phase2CContractError("LeRobot 0.6 is required to prepare Phase 2C") from error
    return LeRobotDataset(
        repo_id=repo_id,
        root=train_root,
        video_backend="pyav",
        return_uint8=True,
    )


def _sample(dataset: Any, index: int) -> dict[str, object]:
    value = dataset[index]
    if not isinstance(value, dict):
        raise Phase2CContractError("LeRobotDataset item must be a mapping")
    return value


def _task(value: object) -> str:
    if isinstance(value, str) and value:
        return value
    raise Phase2CContractError("real LeRobot sample task must be a non-empty string")


def _chw_uint8(value: object) -> np.ndarray:
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if array.shape != (3, 256, 256):
        raise Phase2CContractError("sanity image must have shape [3,256,256]")
    if array.dtype != np.uint8:
        if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
            raise Phase2CContractError("sanity image must be uint8 or finite floating data")
        if float(array.min()) < 0 or float(array.max()) > 1:
            raise Phase2CContractError("floating sanity image must be in [0,1]")
        array = np.rint(array * 255).astype(np.uint8)
    return array


def _write_sanity_grid(dataset: Any, path: Path, *, count: int) -> dict[str, object]:
    count = min(max(count, 1), int(dataset.num_frames))
    indices = np.linspace(0, int(dataset.num_frames) - 1, count, dtype=np.int64).tolist()
    tiles: list[Image.Image] = []
    samples: list[dict[str, object]] = []
    for index in indices:
        item = _sample(dataset, index)
        image = _chw_uint8(item[IMAGE_FEATURE_KEY])
        task = _task(item.get("task"))
        state = np.asarray(item[STATE_FEATURE_KEY], dtype=np.float32)
        action = np.asarray(item[ACTION_FEATURE_KEY], dtype=np.float32)
        if state.shape != (9,) or action.shape != (8,):
            raise Phase2CContractError("real sample state/action shapes differ from Phase 2B")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise Phase2CContractError("real sample state/action values must be finite")
        tile = Image.new("RGB", (256, 300), "white")
        tile.paste(Image.fromarray(np.transpose(image, (1, 2, 0)), mode="RGB"), (0, 0))
        draw = ImageDraw.Draw(tile)
        draw.text((4, 260), f"frame={index}", fill="black")
        draw.text((4, 276), task[:42], fill="black")
        tiles.append(tile)
        samples.append(
            {
                "frame_index": index,
                "task": task,
                "state": state.tolist(),
                "action": action.tolist(),
            }
        )
    columns = 3
    rows = (len(tiles) + columns - 1) // columns
    grid = Image.new("RGB", (columns * 256, rows * 300), "white")
    for index, tile in enumerate(tiles):
        grid.paste(tile, ((index % columns) * 256, (index // columns) * 300))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise Phase2CContractError(f"refusing to overwrite sanity grid: {path}")
    grid.save(path, format="PNG", optimize=True)
    return {
        "schema_version": "langmani-v2-phase2c-sanity-samples-v0",
        "image_grid": path.name,
        "image_grid_sha256": f"sha256:{sha256_file(path)}",
        "samples": samples,
    }


def _verify_lerobot_train_statistics(dataset: Any, normalization: dict[str, object]) -> None:
    """Prove that the official trainer will consume the recomputed train-only moments."""

    metadata = getattr(getattr(dataset, "meta", None), "stats", None)
    features = normalization.get("features")
    if not isinstance(metadata, Mapping) or not isinstance(features, Mapping):
        raise Phase2CContractError("LeRobot train metadata statistics are unavailable")
    for key in (STATE_FEATURE_KEY, ACTION_FEATURE_KEY):
        observed = metadata.get(key)
        expected = features.get(key)
        if not isinstance(observed, Mapping) or not isinstance(expected, Mapping):
            raise Phase2CContractError(f"LeRobot train statistics are missing {key}")
        for statistic in ("mean", "std", "min", "max"):
            if statistic not in observed:
                raise Phase2CContractError(f"LeRobot train statistics lack {key}.{statistic}")
            left = np.asarray(observed[statistic], dtype=np.float64).reshape(-1)
            right = np.asarray(expected[statistic], dtype=np.float64).reshape(-1)
            if left.shape != right.shape or not np.allclose(left, right, rtol=1e-5, atol=1e-6):
                raise Phase2CContractError(
                    f"official LeRobot train statistic differs from recomputation: {key}.{statistic}"
                )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    export_root = args.export_root.resolve()
    output_root = args.output_root.resolve()
    input_manifest = verify_phase2b_for_phase2c(
        export_root=export_root,
        phase2b_result_path=args.phase2b_result.resolve(),
        verifier_report_path=args.phase2b_verifier_report.resolve(),
    )
    train_view = build_train_view_manifest(input_manifest)
    export = read_json_object(
        export_root / "langmani" / "export_manifest.json", "push export manifest"
    )
    repo_prefix = export.get("repo_id_prefix")
    if not isinstance(repo_prefix, str) or not repo_prefix:
        raise Phase2CContractError("push export lacks repo_id_prefix")
    dataset = _dataset(Path(str(train_view["dataset_root"])), f"{repo_prefix}-train")
    if int(dataset.num_episodes) != TRAIN_EPISODES or int(dataset.num_frames) != TRAIN_FRAMES:
        raise Phase2CContractError("real LeRobot train readback counts changed")
    normalization = compute_train_only_normalization(
        _sample(dataset, index) for index in range(int(dataset.num_frames))
    )
    _verify_lerobot_train_statistics(dataset, normalization)
    features = normalization["features"]
    assert isinstance(features, dict)
    for key in (STATE_FEATURE_KEY, ACTION_FEATURE_KEY):
        item = _sample(dataset, 0)
        restored = normalization_round_trip(
            np.asarray(item[key]),
            features[key],
        )
        if not np.array_equal(restored.astype(np.float32), np.asarray(item[key], dtype=np.float32)):
            raise Phase2CContractError(f"{key} normalization round trip changed float32 values")
    sanity = _write_sanity_grid(
        dataset,
        output_root / "sanity" / "training_images.png",
        count=args.sanity_samples,
    )
    sanity["train_view_semantic_sha256"] = train_view["semantic_sha256"]
    episodes = read_json_object(
        export_root / "langmani" / "episodes.json", "push episode inventory"
    ).get("records")
    if not isinstance(episodes, list) or not all(isinstance(item, Mapping) for item in episodes):
        raise Phase2CContractError("push episode inventory records are malformed")
    schedule = build_phase2c_schedules(episodes)
    write_json_once(output_root / "smolvla_push_input_manifest.json", input_manifest)
    write_json_once(output_root / "smolvla_push_train_view_manifest.json", train_view)
    write_json_once(output_root / "smolvla_push_normalization_manifest.json", normalization)
    write_json_once(output_root / "smolvla_push_feature_mapping.json", feature_mapping_manifest())
    write_json_once(output_root / "smolvla_push_evaluation_schedule.json", schedule)
    write_json_once(output_root / "sanity" / "samples.json", sanity)
    complete = {
        "schema_version": "langmani-v2-phase2c-prepare-complete-v0",
        "input_manifest_sha256": f"sha256:{sha256_file(output_root / 'smolvla_push_input_manifest.json')}",
        "train_view_manifest_sha256": f"sha256:{sha256_file(output_root / 'smolvla_push_train_view_manifest.json')}",
        "normalization_manifest_sha256": f"sha256:{sha256_file(output_root / 'smolvla_push_normalization_manifest.json')}",
        "feature_mapping_sha256": f"sha256:{sha256_file(output_root / 'smolvla_push_feature_mapping.json')}",
        "evaluation_schedule_sha256": f"sha256:{sha256_file(output_root / 'smolvla_push_evaluation_schedule.json')}",
        "real_lerobot_readback": {
            "episodes": int(dataset.num_episodes),
            "frames": int(dataset.num_frames),
        },
        "passed": True,
    }
    write_json_once(output_root / "complete.json", complete)
    print(json.dumps(complete, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
