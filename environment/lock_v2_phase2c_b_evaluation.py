"""Freeze the Phase 2C-B validation-only horizon and intervention identities."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from langmani.v2.phase2c_b import (
    Phase2CBContractError,
    build_horizon_selection_lock,
    build_language_intervention_manifest,
    canonical_fingerprint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-schedule", type=Path, required=True)
    parser.add_argument("--language-intervention-lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _read_fingerprinted(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase2CBContractError(f"{path} must contain one object")
    semantic = dict(value)
    fingerprint = semantic.pop("fingerprint", None)
    if fingerprint != canonical_fingerprint(semantic):
        raise Phase2CBContractError(f"artifact fingerprint changed: {path}")
    return value


def _write_new_or_equal(path: Path, value: Mapping[str, object]) -> None:
    encoded = json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise Phase2CBContractError(f"existing immutable evaluation lock differs: {path}")
        return
    path.write_text(encoded, encoding="utf-8", newline="\n")


def main() -> int:
    args = parse_args()
    schedule = _read_fingerprinted(args.evaluation_schedule.resolve())
    existing_language = _read_fingerprinted(args.language_intervention_lock.resolve())
    expected_language = build_language_intervention_manifest(schedule)
    if existing_language != expected_language:
        raise Phase2CBContractError("preflight language-intervention lock changed")
    horizon = build_horizon_selection_lock(schedule)
    output_root = args.output_root.resolve()
    _write_new_or_equal(output_root / "horizon_selection_lock.json", horizon)
    completion_semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-b-evaluation-lock-complete-v0",
        "evaluation_schedule_fingerprint": schedule["fingerprint"],
        "horizon_selection_fingerprint": horizon["fingerprint"],
        "language_intervention_fingerprint": existing_language["fingerprint"],
        "validation_results_opened": False,
        "final_test_opened": False,
        "passed": True,
    }
    completion = {
        **completion_semantic,
        "fingerprint": canonical_fingerprint(completion_semantic),
    }
    _write_new_or_equal(output_root / "evaluation_lock_complete.json", completion)
    print(json.dumps(completion, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
