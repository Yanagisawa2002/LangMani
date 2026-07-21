"""Freeze the validation-selected Phase 2C policy before any sealed outcome is opened."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from langmani.v2.phase2c import Phase2CContractError, read_json_object, sha256_file, write_json_once
from langmani.v2.phase2c_evaluation import FrozenPolicyManifest
from langmani.v2.smolvla_adapter import SmolVLAAdapterConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-config", type=Path, required=True)
    parser.add_argument("--checkpoint-selection", type=Path, required=True)
    parser.add_argument("--development-summary", type=Path, required=True)
    parser.add_argument("--evaluation-schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _selected_digests(value: dict[str, Any]) -> set[str]:
    selected = value.get("selected")
    if not isinstance(selected, list) or not all(isinstance(item, dict) for item in selected):
        raise Phase2CContractError("offline checkpoint selection is malformed")
    return {
        str(item.get("checkpoint_sha256"))
        for item in selected
        if isinstance(item.get("checkpoint_sha256"), str)
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    adapter = SmolVLAAdapterConfig.load(args.adapter_config.resolve())
    selection = read_json_object(
        args.checkpoint_selection.resolve(), "offline checkpoint selection"
    )
    if selection.get("passed") is not True:
        raise Phase2CContractError("offline checkpoint selection is not accepted")
    if adapter.checkpoint_sha256 not in _selected_digests(selection):
        raise Phase2CContractError("development checkpoint was not selected by offline validation")
    development = read_json_object(
        args.development_summary.resolve(), "development competence summary"
    )
    if development.get("passed") is not True or development.get("mode") != "development":
        raise Phase2CContractError("development competence gate did not pass")
    schedule_path = args.evaluation_schedule.resolve()
    schedule = read_json_object(schedule_path, "Phase 2C evaluation schedule")
    if schedule.get("passed") is not True or schedule.get("sealed_count") != 230:
        raise Phase2CContractError("final evaluation schedule is not accepted")
    schedule_digest = sha256_file(schedule_path)
    if development.get("schedule_sha256") != f"sha256:{schedule_digest}":
        raise Phase2CContractError("development evidence used a different schedule")
    manifest = FrozenPolicyManifest(
        checkpoint_sha256=adapter.checkpoint_sha256,
        base_model_revision=adapter.base_model_revision,
        training_seed=adapter.training_seed,
        training_config_sha256=adapter.training_config_sha256,
        train_view_sha256=adapter.train_view_sha256,
        normalization_sha256=adapter.normalization_sha256,
        processor_sha256=adapter.processor_sha256,
        action_chunk_size=50,
        execution_horizon=adapter.execution_horizon,
        control_frequency_hz=20,
        inference_seed=adapter.inference_seed,
        final_test_identity_sha256=schedule_digest,
    )
    payload = manifest.to_dict()
    write_json_once(args.output.resolve(), payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
