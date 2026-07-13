"""Inspect one finalized M3B episode without modifying the dataset."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from langmani.collection.command_support import validated_dataset_root, write_json_atomic
from langmani.datasets.lerobot_types import (
    LeRobotDatasetSummary,
    LeRobotExportManifest,
    LeRobotValidationReport,
)
from langmani.datasets.lerobot_writer import COMPLETION_MARKER

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_DATASET_ROOT = OUTPUT_ROOT / "datasets" / "m3b" / "langmani-pick-place-lerobot-v1"
DEFAULT_REPORT = OUTPUT_ROOT / "diagnostics" / "m3b" / "episode_inspection.json"


def _dataset_root(value: str) -> Path:
    try:
        return validated_dataset_root(value, output_root=OUTPUT_ROOT)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=_dataset_root, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--episode-index", type=_non_negative_int, required=True)
    parser.add_argument(
        "--extract-frame",
        type=_non_negative_int,
        metavar="EPISODE_FRAME_INDEX",
        help="Write one episode-relative decoded PNG under outputs/diagnostics/m3b.",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return payload


def _inspect(dataset_root: Path, episode_index: int, extract_frame: int | None) -> dict[str, Any]:
    sidecar = dataset_root / "langmani"
    manifest = LeRobotExportManifest.from_dict(_read_json(sidecar / "export_manifest.json"))
    summary = LeRobotDatasetSummary.from_dict(_read_json(sidecar / "dataset_summary.json"))
    validation = LeRobotValidationReport.from_dict(_read_json(sidecar / "validation_report.json"))
    completion = _read_json(dataset_root / COMPLETION_MARKER)
    if completion.get("export_fingerprint") != manifest.export_fingerprint:
        raise RuntimeError("completion marker fingerprint differs from export manifest")
    if not 0 <= episode_index < len(manifest.episodes):
        raise IndexError(f"episode index must be in [0, {len(manifest.episodes) - 1}]")
    record = manifest.episodes[episode_index]
    alignment = next(
        (
            item
            for item in validation.source_alignments
            if item.lerobot_episode_index == episode_index
        ),
        None,
    )
    video_result = next(
        (item for item in validation.video_results if item.lerobot_episode_index == episode_index),
        None,
    )

    from lerobot.datasets import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id=manifest.repo_id,
        root=dataset_root,
        video_backend="pyav",
        return_uint8=True,
    )
    episode_meta = dataset.meta.episodes[episode_index]
    video_key = manifest.config.feature_contract.image_feature_key
    chunk_index = episode_meta[f"videos/{video_key}/chunk_index"]
    file_index = episode_meta[f"videos/{video_key}/file_index"]
    video_path = dataset.meta.video_path.format(
        video_key=video_key,
        chunk_index=chunk_index,
        file_index=file_index,
    )
    result: dict[str, Any] = {
        "command": "inspect_lerobot_episode",
        "export_fingerprint": manifest.export_fingerprint,
        "lerobot_episode_index": episode_index,
        "source_m3a_episode_id": record.source_episode_id,
        "source_scene_group_id": record.source_scene_group_id,
        "split": record.split.value,
        "task_spec": {
            "target_object_id": record.target_object_id,
            "target_bin_id": record.target_bin_id,
            "instruction_template_id": record.instruction_template_id,
        },
        "canonical_instruction": record.canonical_instruction,
        "episode_frame_count": record.output_frame_count,
        "fps": summary.fps,
        "state_schema": manifest.config.policy_state_schema.to_dict(),
        "action_schema": {
            "dtype": manifest.config.feature_contract.action_dtype,
            "shape": list(manifest.config.feature_contract.action_shape),
            "components": list(manifest.config.feature_contract.action_names),
        },
        "source_checksum": record.source_checksum,
        "source_h5_sha256": record.source_h5_sha256,
        "source_json_sha256": record.source_json_sha256,
        "raw_render_digest": record.raw_render_digest,
        "video": {
            "relative_path": video_path,
            "from_timestamp": episode_meta[f"videos/{video_key}/from_timestamp"],
            "to_timestamp": episode_meta[f"videos/{video_key}/to_timestamp"],
            "feature_info": dataset.features[video_key].get("info", {}),
            "validation": None if video_result is None else video_result.to_dict(),
        },
        "source_alignment": None if alignment is None else alignment.to_dict(),
        "dataset_validation_passed": validation.passed,
        "passed": True,
    }
    if extract_frame is not None:
        if extract_frame >= record.output_frame_count:
            raise IndexError("extract-frame is outside the selected episode")
        absolute_index = int(episode_meta["dataset_from_index"]) + extract_frame
        item = dataset[absolute_index]
        image = item[video_key]
        array = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
        if array.shape != (3, summary.image_shape[0], summary.image_shape[1]):
            raise RuntimeError(f"decoded image has unexpected shape {array.shape}")
        rgb = np.transpose(array, (1, 2, 0))
        output = (
            OUTPUT_ROOT
            / "diagnostics"
            / "m3b"
            / (f"episode-{episode_index:03d}-frame-{extract_frame:04d}.png")
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb.astype(np.uint8, copy=False)).save(output)
        result["extracted_frame_path"] = str(output)
    return result


def main() -> int:
    args = parse_args()
    report_path = args.report.resolve()
    report_path.unlink(missing_ok=True)
    try:
        result = _inspect(args.dataset_root.resolve(), args.episode_index, args.extract_frame)
        exit_code = 0
    except Exception as error:  # noqa: BLE001 - outer command boundary preserves diagnostics
        traceback.print_exc()
        result = {
            "command": "inspect_lerobot_episode",
            "episode_index": args.episode_index,
            "passed": False,
            "exception_type": type(error).__name__,
            "exception_message": str(error) or repr(error),
        }
        exit_code = 1
    written = write_json_atomic(report_path, result)
    print(json.dumps({"passed": result["passed"], "report": str(written)}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
