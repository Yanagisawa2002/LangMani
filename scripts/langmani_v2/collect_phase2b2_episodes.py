"""Collect or validate resumable atomic Phase 2B.2 native episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langmani.v2.push_collection import collect_push_attempts
from langmani.v2.push_dataset import PushCollectionConfig, PushDatasetContractError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "langmani_v2" / "phase_2b2" / "dataset_contract.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("full", "top_up"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--seed-count", type=int)
    parser.add_argument("--target-successes", type=int)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _validate_overrides(args: argparse.Namespace, config: PushCollectionConfig) -> None:
    full_count = config.integer("full_standard_attempts") + config.integer("full_hard_attempts")
    expected = {
        "seed_start": (
            config.integer("full_seed_start")
            if args.stage == "full"
            else config.integer("full_seed_start") + full_count
        ),
        "seed_count": (
            full_count if args.stage == "full" else config.integer("maximum_top_up_attempts")
        ),
        "target_successes": config.integer("preferred_accepted_episodes"),
        "max_attempts": (
            full_count if args.stage == "full" else config.integer("maximum_top_up_attempts")
        ),
    }
    for name, frozen in expected.items():
        observed = getattr(args, name)
        if observed is not None and observed != frozen:
            raise PushDatasetContractError(
                f"--{name.replace('_', '-')}={observed} differs from frozen value {frozen}"
            )


def main() -> int:
    args = parse_args()
    config = PushCollectionConfig.load(args.config)
    _validate_overrides(args, config)
    report = collect_push_attempts(
        config=config,
        output_root=args.output_root,
        stage=args.stage,
        resume=args.resume,
        validate_only=args.validate_only,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
