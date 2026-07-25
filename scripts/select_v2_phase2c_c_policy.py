"""Apply the pre-registered Phase 2C-C checkpoint or horizon selection rule."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from langmani.v2.phase2c_a import PACKAGE_FINGERPRINT
from langmani.v2.phase2c_c import Phase2CCContractError, canonical_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    checkpoint = subparsers.add_parser("checkpoint")
    checkpoint.add_argument("--diagnostic", type=Path, action="append", required=True)
    checkpoint.add_argument("--output", type=Path, required=True)
    horizon = subparsers.add_parser("horizon")
    horizon.add_argument("--checkpoint-selection", type=Path, required=True)
    horizon.add_argument("--summary", type=Path, action="append", required=True)
    horizon.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2CCContractError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise Phase2CCContractError(f"{path} must contain one object")
    semantic = dict(value)
    fingerprint = semantic.pop("fingerprint", None)
    if fingerprint != canonical_fingerprint(semantic):
        raise Phase2CCContractError(f"evidence fingerprint changed: {path}")
    return value


def _write(path: Path, value: Mapping[str, object]) -> None:
    encoded = json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise Phase2CCContractError(f"existing immutable selection differs: {path}")
        return
    path.write_text(encoded, encoding="utf-8", newline="\n")


def _checkpoint_selection(paths: list[Path]) -> dict[str, object]:
    rows = [_read(path.resolve()) for path in paths]
    if (
        len(rows) != 3
        or {row.get("optimizer_step") for row in rows} != {5_000, 10_000, 20_000}
        or any(
            row.get("schema_version") != "langmani-v2-phase2c-c-offline-checkpoint-diagnostic-v0"
            or row.get("package_fingerprint") != PACKAGE_FINGERPRINT
            or row.get("formulation") != "relative"
            or row.get("split") != "validation"
            or row.get("passed") is not True
            for row in rows
        )
    ):
        raise Phase2CCContractError("checkpoint diagnostics do not cover the frozen queue")
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["physical_first_action_mae"]),
            float(row["residual_action_mae"]),
            float(row["validation_flow_matching_loss"]),
            int(row["optimizer_step"]),
        ),
    )
    selected = ranked[0]
    semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-c-checkpoint-selection-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "model_kind": "pick_smolvla_relative",
        "selection_split": "validation",
        "selection_rule": [
            "minimum_physical_first_action_mae",
            "minimum_residual_action_mae",
            "minimum_validation_flow_matching_loss",
            "earliest_optimizer_step",
        ],
        "candidates": [
            {
                "optimizer_step": row["optimizer_step"],
                "checkpoint": row["checkpoint"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "diagnostic_fingerprint": row["fingerprint"],
                "physical_first_action_mae": row["physical_first_action_mae"],
                "residual_action_mae": row["residual_action_mae"],
                "validation_flow_matching_loss": row["validation_flow_matching_loss"],
            }
            for row in sorted(rows, key=lambda row: int(row["optimizer_step"]))
        ],
        "selected_optimizer_step": selected["optimizer_step"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_diagnostic_fingerprint": selected["fingerprint"],
        "test_results_used": False,
        "closed_loop_results_used": False,
        "passed": True,
    }
    return {**semantic, "fingerprint": canonical_fingerprint(semantic)}


def _integer(row: Mapping[str, object], key: str) -> int:
    value = row.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise Phase2CCContractError(f"horizon summary lacks integer {key}")
    return value


def _horizon_selection(
    checkpoint_path: Path,
    summary_paths: list[Path],
) -> dict[str, object]:
    checkpoint = _read(checkpoint_path.resolve())
    summaries = [_read(path.resolve()) for path in summary_paths]
    if (
        checkpoint.get("schema_version") != "langmani-v2-phase2c-c-checkpoint-selection-v0"
        or checkpoint.get("passed") is not True
        or len(summaries) != 2
        or {summary.get("execution_horizon") for summary in summaries} != {1, 8}
        or any(
            summary.get("package_fingerprint") != PACKAGE_FINGERPRINT
            or summary.get("split_role") != "validation_horizon_screen"
            or summary.get("episode_count") != 6
            or summary.get("checkpoint_identities")
            != [checkpoint.get("selected_checkpoint_sha256")]
            or summary.get("invalid_action_count") != 0
            or summary.get("simulator_error_count") != 0
            for summary in summaries
        )
    ):
        raise Phase2CCContractError("horizon screen evidence changed")
    ranked = sorted(
        summaries,
        key=lambda row: (
            -_integer(row, "success_count"),
            -_integer(row, "ever_grasped_count"),
            -_integer(row, "ever_lifted_count"),
            _integer(row, "timeout_count"),
            int(row["execution_horizon"]),
        ),
    )
    selected = ranked[0]
    semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-c-policy-selection-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "model_kind": "pick_smolvla_relative",
        "selection_split": "validation",
        "checkpoint_selection_fingerprint": checkpoint["fingerprint"],
        "horizon_selection_rule": [
            "maximum_success_count",
            "maximum_ever_grasped_count",
            "maximum_ever_lifted_count",
            "minimum_timeout_count",
            "prefer_h1",
        ],
        "horizon_candidates": [
            {
                "execution_horizon": row["execution_horizon"],
                "summary_fingerprint": row["fingerprint"],
                "success_count": row["success_count"],
                "ever_grasped_count": row["ever_grasped_count"],
                "ever_lifted_count": row["ever_lifted_count"],
                "timeout_count": row["timeout_count"],
            }
            for row in sorted(summaries, key=lambda row: int(row["execution_horizon"]))
        ],
        "selected_checkpoint": checkpoint["selected_checkpoint"],
        "selected_checkpoint_sha256": checkpoint["selected_checkpoint_sha256"],
        "selected_optimizer_step": checkpoint["selected_optimizer_step"],
        "selected_execution_horizon": selected["execution_horizon"],
        "test_results_used": False,
        "full_validation_results_used": False,
        "settings_mutable_after_selection": False,
        "passed": True,
    }
    return {**semantic, "fingerprint": canonical_fingerprint(semantic)}


def main() -> int:
    args = parse_args()
    if args.mode == "checkpoint":
        result = _checkpoint_selection(args.diagnostic)
    else:
        result = _horizon_selection(args.checkpoint_selection, args.summary)
    _write(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
