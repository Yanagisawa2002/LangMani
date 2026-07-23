"""Execute bounded LangMani 2.0 Phase 2B.6 dataset-production stages."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langmani.v2.phase2b6_lerobot import (
    build_unified_package,
    convert_task_split_roots,
    convert_visual_shift_roots,
    create_archive,
    finalize_acceptance,
    restore_and_validate,
    run_full_readback,
    run_leakage_audit,
    run_source_to_derived_verification,
    write_artifact_manifest,
)
from langmani.v2.phase2b6_runtime import (
    compute_dataset_statistics,
    prepare_production,
    run_full_replay_production,
    run_visual_shift_pilot,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = PROJECT_ROOT / "configs" / "langmani_v2" / "phase2b6_dataset_production.yaml"
DEFAULT_PHASE2B5_ARTIFACTS = PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2b5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "preflight",
            "visual-shift-pilot",
            "produce",
            "statistics",
            "convert",
            "visual-shift",
            "unified-package",
            "readback",
            "source-equality",
            "leakage",
            "archive",
            "restore",
            "finalize",
            "artifact-manifest",
        ),
    )
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument(
        "--phase2b5-artifact-root",
        type=Path,
        default=DEFAULT_PHASE2B5_ARTIFACTS,
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--restore-root", type=Path)
    parser.add_argument("--network-turbo-sourced", action="store_true")
    return parser.parse_args()


def _required(value: Path | None, label: str) -> Path:
    if value is None:
        raise ValueError(f"{label} is required for this stage")
    return value.resolve()


def _record_command(production_root: Path) -> None:
    path = production_root / "run" / "command_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "executable": Path(sys.executable).resolve().as_posix(),
        "argv": list(sys.argv),
        "shell_command": shlex.join([sys.executable, *sys.argv]),
        "optimizer_steps": 0,
    }
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _command_log(production_root: Path) -> list[str]:
    path = production_root / "run" / "command_log.jsonl"
    commands: list[str] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict) or not isinstance(value.get("shell_command"), str):
                raise ValueError("command log is malformed")
            commands.append(value["shell_command"])
    return commands


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    production_root = args.production_root.resolve()
    evidence_root = args.evidence_root.resolve()
    evidence_root.mkdir(parents=True, exist_ok=True)
    _record_command(production_root)
    result: dict[str, Any]
    if args.stage == "preflight":
        result = prepare_production(
            repo_root=repo_root,
            spec_path=args.spec.resolve(),
            phase2b5_artifact_root=args.phase2b5_artifact_root.resolve(),
            source_root=_required(args.source_root, "--source-root"),
            production_root=production_root,
            archive_root=_required(args.archive_root, "--archive-root"),
            restore_root=_required(args.restore_root, "--restore-root"),
            evidence_root=evidence_root,
            network_turbo_sourced=args.network_turbo_sourced,
        )
    elif args.stage == "visual-shift-pilot":
        result = run_visual_shift_pilot(
            source_root=_required(args.source_root, "--source-root"),
            production_root=production_root,
            evidence_root=evidence_root,
            spec_path=args.spec.resolve(),
        )
    elif args.stage == "produce":
        result = run_full_replay_production(
            repo_root=repo_root,
            source_root=_required(args.source_root, "--source-root"),
            production_root=production_root,
            evidence_root=evidence_root,
            spec_path=args.spec.resolve(),
        )
    elif args.stage == "statistics":
        statistics, normalization = compute_dataset_statistics(
            production_root=production_root,
            evidence_root=evidence_root,
        )
        result = {
            "schema_version": "langmani-v2-phase2b6-statistics-stage-v0",
            "statistics_fingerprint": statistics["fingerprint"],
            "normalization_fingerprint": normalization["fingerprint"],
            "passed": statistics["passed"] is True and normalization["passed"] is True,
        }
    elif args.stage == "convert":
        result = convert_task_split_roots(
            production_root=production_root,
            evidence_root=evidence_root,
            spec_path=args.spec.resolve(),
        )
    elif args.stage == "visual-shift":
        result = convert_visual_shift_roots(
            production_root=production_root,
            evidence_root=evidence_root,
            spec_path=args.spec.resolve(),
        )
    elif args.stage == "unified-package":
        result = build_unified_package(
            production_root=production_root,
            evidence_root=evidence_root,
            spec_path=args.spec.resolve(),
        )
    elif args.stage == "readback":
        readback, equality = run_full_readback(
            production_root=production_root,
            evidence_root=evidence_root,
        )
        result = {
            "schema_version": "langmani-v2-phase2b6-readback-stage-v0",
            "readback_fingerprint": readback["fingerprint"],
            "equality_intermediate_fingerprint": equality["fingerprint"],
            "passed": readback["passed"] is True and equality["passed"] is True,
        }
    elif args.stage == "source-equality":
        result = run_source_to_derived_verification(
            source_root=_required(args.source_root, "--source-root"),
            production_root=production_root,
            evidence_root=evidence_root,
        )
    elif args.stage == "leakage":
        result = run_leakage_audit(
            production_root=production_root,
            evidence_root=evidence_root,
        )
    elif args.stage == "archive":
        result = create_archive(
            production_root=production_root,
            archive_root=_required(args.archive_root, "--archive-root"),
            evidence_root=evidence_root,
            spec_path=args.spec.resolve(),
        )
    elif args.stage == "restore":
        result = restore_and_validate(
            archive_manifest_path=evidence_root / "archive_manifest.json",
            restore_root=_required(args.restore_root, "--restore-root"),
            evidence_root=evidence_root,
        )
    elif args.stage == "finalize":
        result = finalize_acceptance(
            production_root=production_root,
            evidence_root=evidence_root,
            repo_root=repo_root,
            command_log=_command_log(production_root),
        )
    elif args.stage == "artifact-manifest":
        result = write_artifact_manifest(evidence_root=evidence_root)
    else:
        raise AssertionError(args.stage)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
