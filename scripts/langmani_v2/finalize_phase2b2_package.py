"""Issue the content-bound Phase 2B.2 acceptance package after every gate is fixed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from langmani.datasets.lerobot_types import IMAGE_FEATURE_KEY
from langmani.v2.phase2b2 import (
    PHASE2B2_DATASET_ID,
    AcceptedDatasetPackage,
    write_json_once,
)
from langmani.v2.phase2c import sha256_file
from langmani.v2.push_dataset import PushCollectionConfig, PushDatasetContractError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "langmani_v2" / "phase_2b2" / "dataset_contract.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--expert-report", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--archive-report", type=Path, required=True)
    parser.add_argument("--replication-report", type=Path, required=True)
    parser.add_argument("--compatibility-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-output", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PushDatasetContractError(f"cannot read finalization input {path}: {error}") from error
    if not isinstance(value, dict):
        raise PushDatasetContractError(f"finalization input must be an object: {path}")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PushDatasetContractError(f"{label} must be a non-negative integer")
    return value


def main() -> int:
    args = parse_args()
    config = PushCollectionConfig.load(args.config)
    expert = _read(args.expert_report)
    raw = _read(args.raw_manifest)
    dataset = _read(args.dataset_manifest)
    quality = _read(args.quality_report)
    archives = _read(args.archive_report)
    replication = _read(args.replication_report)
    compatibility = _read(args.compatibility_report)
    archive_values = archives.get("archives")
    if not isinstance(archive_values, dict):
        raise PushDatasetContractError("archive report lacks archive identities")
    raw_archive = archive_values.get("raw")
    lerobot_archive = archive_values.get("lerobot")
    if not isinstance(raw_archive, dict) or not isinstance(lerobot_archive, dict):
        raise PushDatasetContractError("raw or LeRobot archive identity is missing")
    replication_assets = replication.get("assets")
    if not isinstance(replication_assets, dict):
        raise PushDatasetContractError("replication report lacks asset identities")
    raw_replication = replication_assets.get("raw")
    lerobot_replication = replication_assets.get("lerobot")
    if not isinstance(raw_replication, dict) or not isinstance(lerobot_replication, dict):
        raise PushDatasetContractError("replication report is missing raw or LeRobot evidence")
    gates = {
        "dataset_identity": raw.get("dataset_id") == PHASE2B2_DATASET_ID
        and dataset.get("dataset_id") == PHASE2B2_DATASET_ID,
        "expert": expert.get("passed") is True,
        "quality": quality.get("passed") is True and dataset.get("quality_passed") is True,
        "raw_archive_restore": raw_archive.get("restore_passed") is True,
        "lerobot_archive_restore": lerobot_archive.get("restore_passed") is True,
        "replication": replication.get("passed") is True,
        "replication_hashes": raw_replication.get("expected_sha256") == raw_archive.get("sha256")
        and lerobot_replication.get("expected_sha256") == lerobot_archive.get("sha256"),
        "smolvla_real_batch": compatibility.get("passed") is True,
        "optimizer_steps_zero": compatibility.get("optimizer_steps") == 0
        and raw.get("optimizer_steps") == 0
        and dataset.get("optimizer_steps") == 0,
    }
    accepted = all(gates.values())
    package = AcceptedDatasetPackage(
        dataset_id=PHASE2B2_DATASET_ID,
        dataset_version="v2",
        source_commit=str(raw["source_commit"]),
        environment_identity=str(config.payload["environment_id"]),
        raw_manifest_sha256=f"sha256:{sha256_file(args.raw_manifest)}",
        lerobot_tree_digest=str(dataset["lerobot_tree_digest"]),
        meta_info_sha256=str(dataset["meta_info_sha256"]),
        split_manifest_sha256=str(dataset["split_manifest_sha256"]),
        raw_archive_sha256=str(raw_archive["sha256"]),
        lerobot_archive_sha256=str(lerobot_archive["sha256"]),
        episode_count=_integer(dataset.get("episode_count"), "episode_count"),
        frame_count=_integer(dataset.get("frame_count"), "frame_count"),
        total_bytes=_integer(dataset.get("total_bytes"), "total_bytes"),
        fps=20.0,
        camera_keys=(IMAGE_FEATURE_KEY,),
        state_dimension=9,
        action_dimension=8,
        lerobot_version="0.6.0",
        acceptance_status="ACCEPTED" if accepted else "REJECTED",
    )
    gate_report = {
        "schema_version": "langmani-v2-phase2b2-final-gate-v0",
        "dataset_id": PHASE2B2_DATASET_ID,
        "gates": gates,
        "optimizer_steps": 0,
        "passed": accepted,
    }
    write_json_once(args.output.resolve(), package.to_dict())
    write_json_once(args.gate_output.resolve(), gate_report)
    print(json.dumps(gate_report, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
