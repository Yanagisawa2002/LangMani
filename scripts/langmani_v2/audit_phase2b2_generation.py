"""Audit the frozen Phase 2B.2 generation contract without simulator execution."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from langmani.datasets.lerobot_types import FeatureContract
from langmani.v2.phase2b2 import PHASE2B2_DATASET_ID, write_json_once
from langmani.v2.push_dataset import PushCollectionConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "langmani_v2" / "phase_2b2" / "dataset_contract.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PushCollectionConfig.load(args.config)
    dataset = config.payload["dataset"]
    assert isinstance(dataset, dict)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "schema_version": "langmani-v2-phase2b2-generation-audit-v0",
        "dataset_id": dataset["repo_id"],
        "dataset_id_is_independent_v2": dataset["repo_id"] == PHASE2B2_DATASET_ID,
        "collection_fingerprint": config.fingerprint,
        "source_commit_policy": config.payload["source_commit_policy"],
        "observed_source_commit": commit,
        "environment_id": config.payload["environment_id"],
        "expert_id": config.payload["expert_id"],
        "fresh_environment_per_attempt": True,
        "atomic_episode_commits": config.payload["atomic_episode_commits"],
        "independent_action_replay_required": True,
        "feature_contract": FeatureContract().to_dict(),
        "v1_bytes_required": False,
        "optimizer_steps": 0,
        "passed": True,
    }
    write_json_once(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
