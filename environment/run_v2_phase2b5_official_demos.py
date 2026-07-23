"""Execute bounded, no-training Phase 2B.5 official-demo qualification stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langmani.v2.phase2b5_runtime import (
    convert_visual_pilot_to_lerobot,
    generate_visual_policy_pilot,
    inspect_official_sources,
    lerobot_environment_manifest,
    readback_lerobot_pilot,
    run_official_action_replays,
    runtime_audit,
    write_padding_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "diagnostics" / "v2" / "phase2b5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "schema",
            "bounded-replay",
            "strong-replay",
            "pilot",
            "convert",
            "readback",
            "padding",
            "lerobot-environment",
            "runtime-audit",
        ),
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pilot-root", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--execution-commit")
    parser.add_argument("--network-turbo-sourced", action="store_true")
    return parser.parse_args()


def _required(path: Path | None, name: str) -> Path:
    if path is None:
        raise SystemExit(f"{name} is required for this stage")
    return path.resolve()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.stage == "schema":
        report = inspect_official_sources(
            source_root=_required(args.source_root, "--source-root"),
            output_root=output_root,
        )
    elif args.stage in {"bounded-replay", "strong-replay"}:
        count = 20 if args.stage == "bounded-replay" else 100
        report = run_official_action_replays(
            source_root=_required(args.source_root, "--source-root"),
            output_root=output_root,
            replay_count=count,
            report_name=(
                "bounded_replay_results.json"
                if args.stage == "bounded-replay"
                else "strong_replay_results.json"
            ),
        )
    elif args.stage == "pilot":
        report = generate_visual_policy_pilot(
            source_root=_required(args.source_root, "--source-root"),
            output_root=output_root,
            pilot_root=_required(args.pilot_root, "--pilot-root"),
        )
    elif args.stage == "convert":
        report = convert_visual_pilot_to_lerobot(
            pilot_root=_required(args.pilot_root, "--pilot-root"),
            dataset_root=_required(args.dataset_root, "--dataset-root"),
            output_root=output_root,
            visual_manifest_path=output_root / "visual_pilot_manifest.json",
        )
    elif args.stage == "readback":
        report = readback_lerobot_pilot(
            dataset_root=_required(args.dataset_root, "--dataset-root"),
            output_root=output_root,
            conversion_manifest_path=output_root / "conversion_pilot_manifest.json",
        )
    elif args.stage == "padding":
        report = write_padding_report(
            visual_manifest_path=output_root / "visual_pilot_manifest.json",
            output_root=output_root,
        )
    elif args.stage == "lerobot-environment":
        report = lerobot_environment_manifest()
        target = output_root / "lerobot_060_environment_manifest.json"
        target.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    else:
        if not args.execution_commit:
            raise SystemExit("--execution-commit is required for runtime-audit")
        report = runtime_audit(
            execution_commit=args.execution_commit,
            network_turbo_sourced=args.network_turbo_sourced,
        )
        target = output_root / "remote_execution_audit.json"
        target.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, sort_keys=True))
    return 0 if report.get("passed", report.get("all_tasks_passed", True)) is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())
