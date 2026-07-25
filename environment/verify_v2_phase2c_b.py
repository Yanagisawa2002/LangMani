"""Independently verify staged LangMani 2.0 Phase 2C-B evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from langmani.v2.phase2c import sha256_file
from langmani.v2.phase2c_a import PACKAGE_FINGERPRINT
from langmani.v2.phase2c_b import (
    Phase2CBContractError,
    canonical_fingerprint,
    classify_pick_gate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-stage", choices=("pick_gate",), required=True)
    parser.add_argument("--static-preparation", type=Path, required=True)
    parser.add_argument("--base-audit", type=Path, required=True)
    parser.add_argument("--smoke-training-result", type=Path, required=True)
    parser.add_argument("--smoke-evaluation-summary", type=Path, required=True)
    parser.add_argument("--pick-micro-result", type=Path, required=True)
    parser.add_argument("--shared-micro-result", type=Path, required=True)
    parser.add_argument("--pick-training-result", type=Path, required=True)
    parser.add_argument("--selected-policy", type=Path, required=True)
    parser.add_argument("--pick-validation-summary", type=Path, required=True)
    parser.add_argument("--initial-pick-validation-summary", type=Path)
    parser.add_argument("--repair-policy", type=Path)
    parser.add_argument("--repair-used", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path, *, require_self_fingerprint: bool = True) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2CBContractError(f"cannot read evidence {path}: {error}") from error
    if not isinstance(value, dict):
        raise Phase2CBContractError(f"{path} must contain one object")
    if require_self_fingerprint:
        semantic = dict(value)
        fingerprint = semantic.pop("fingerprint", None)
        if fingerprint != canonical_fingerprint(semantic):
            raise Phase2CBContractError(f"evidence fingerprint changed: {path}")
    return value


def _integer(document: Mapping[str, object], key: str) -> int:
    value = document.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise Phase2CBContractError(f"evidence lacks valid integer {key}")
    return value


def _write_new_or_equal(path: Path, value: Mapping[str, object]) -> None:
    encoded = json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise Phase2CBContractError(f"existing immutable verification differs: {path}")
        return
    path.write_text(encoded, encoding="utf-8", newline="\n")


def _path_record(path: Path, document: Mapping[str, object]) -> dict[str, object]:
    return {
        "path": path.as_posix(),
        "sha256": f"sha256:{sha256_file(path)}",
        "fingerprint": document.get("fingerprint"),
    }


def main() -> int:
    args = parse_args()
    paths = {
        "static_preparation": args.static_preparation.resolve(),
        "base_audit": args.base_audit.resolve(),
        "smoke_training_result": args.smoke_training_result.resolve(),
        "smoke_evaluation_summary": args.smoke_evaluation_summary.resolve(),
        "pick_micro_result": args.pick_micro_result.resolve(),
        "shared_micro_result": args.shared_micro_result.resolve(),
        "pick_training_result": args.pick_training_result.resolve(),
        "selected_policy": args.selected_policy.resolve(),
        "pick_validation_summary": args.pick_validation_summary.resolve(),
    }
    if bool(args.initial_pick_validation_summary) != bool(args.repair_policy):
        raise Phase2CBContractError(
            "repair verification requires both initial summary and repair policy"
        )
    if args.repair_used != bool(args.repair_policy):
        raise Phase2CBContractError(
            "--repair-used must exactly match the presence of repair evidence"
        )
    if args.repair_policy is not None and args.initial_pick_validation_summary is not None:
        paths["initial_pick_validation_summary"] = args.initial_pick_validation_summary.resolve()
        paths["repair_policy"] = args.repair_policy.resolve()
    documents = {
        key: _read(
            path,
            require_self_fingerprint=key != "smoke_evaluation_summary",
        )
        for key, path in paths.items()
    }
    static = documents["static_preparation"]
    base = documents["base_audit"]
    smoke = documents["smoke_training_result"]
    smoke_evaluation = documents["smoke_evaluation_summary"]
    pick_micro = documents["pick_micro_result"]
    shared_micro = documents["shared_micro_result"]
    training = documents["pick_training_result"]
    selected = documents["selected_policy"]
    validation = documents["pick_validation_summary"]
    if (
        static.get("schema_version") != "langmani-v2-phase2c-b-static-preparation-complete-v0"
        or static.get("passed") is not True
        or base.get("schema_version") != "langmani-v2-phase2c-b-base-audit-complete-v0"
        or base.get("passed") is not True
        or smoke.get("schema_version") != "langmani-v2-phase2c-b-training-result-v0"
        or smoke.get("stage") != "smoke"
        or smoke.get("passed") is not True
        or smoke.get("optimizer_steps") != 1
        or smoke_evaluation.get("package_fingerprint") != PACKAGE_FINGERPRINT
        or smoke_evaluation.get("episode_count") != 1
        or smoke_evaluation.get("invalid_action_count") != 0
        or smoke_evaluation.get("simulator_error_count") != 0
        or smoke_evaluation.get("mean_episode_length") != 1.0
        or smoke_evaluation.get("mean_policy_query_count") != 1.0
        or pick_micro.get("stage") != "micro"
        or pick_micro.get("model_kind") != "pick_smolvla"
        or pick_micro.get("optimizer_steps") != 500
        or pick_micro.get("passed") is not True
        or shared_micro.get("stage") != "micro"
        or shared_micro.get("model_kind") != "shared_language_smolvla"
        or shared_micro.get("optimizer_steps") != 500
        or shared_micro.get("passed") is not True
        or training.get("stage") != "full"
        or training.get("model_kind") != "pick_smolvla"
        or training.get("optimizer_steps") != 20_000
        or training.get("passed") is not True
        or selected.get("schema_version") != "langmani-v2-phase2c-b-selected-policy-v0"
        or selected.get("model_kind") != "pick_smolvla"
        or selected.get("task_id") != "PickCube-v1"
        or selected.get("selection_split") != "validation"
        or selected.get("final_test_opened") is not False
        or selected.get("passed") is not True
        or validation.get("package_fingerprint") != PACKAGE_FINGERPRINT
        or validation.get("model_kind") != "pick_smolvla"
        or validation.get("task_id") != "PickCube-v1"
        or validation.get("split") != "validation"
        or validation.get("instruction_condition") != "correct"
        or validation.get("episode_count") != 30
        or validation.get("completed") is not True
    ):
        raise Phase2CBContractError("Pick-gate evidence contract failed")
    selected_checkpoint = selected.get("selected_checkpoint_sha256")
    original_selected_horizon = selected.get("selected_execution_horizon")
    selected_horizon = original_selected_horizon
    if args.repair_used:
        repair = documents["repair_policy"]
        initial_validation = documents["initial_pick_validation_summary"]
        initial_identities = initial_validation.get("checkpoint_identities")
        repair_inputs = repair.get("inputs")
        repair_initial = (
            repair_inputs.get("initial_validation_summary")
            if isinstance(repair_inputs, Mapping)
            else None
        )
        repair_selected = (
            repair_inputs.get("selected_policy") if isinstance(repair_inputs, Mapping) else None
        )
        if (
            repair.get("schema_version") != "langmani-v2-phase2c-b-bounded-repair-policy-v0"
            or repair.get("package_fingerprint") != PACKAGE_FINGERPRINT
            or repair.get("repair_ordinal") != 1
            or repair.get("repair_kind") != "action_chunk_execution_horizon"
            or repair.get("selected_checkpoint_sha256") != selected_checkpoint
            or repair.get("original_execution_horizon") != original_selected_horizon
            or not isinstance(repair.get("repaired_execution_horizon"), int)
            or isinstance(repair.get("repaired_execution_horizon"), bool)
            or repair.get("repaired_execution_horizon") == original_selected_horizon
            or repair.get("model_bytes_changed") is not False
            or repair.get("optimizer_created") is not False
            or repair.get("training_started") is not False
            or repair.get("dataset_changed") is not False
            or repair.get("split_changed") is not False
            or repair.get("observation_contract_changed") is not False
            or repair.get("physical_action_contract_changed") is not False
            or repair.get("success_predicate_changed") is not False
            or repair.get("final_test_opened") is not False
            or repair.get("passed") is not True
            or initial_validation.get("package_fingerprint") != PACKAGE_FINGERPRINT
            or initial_validation.get("model_kind") != "pick_smolvla"
            or initial_validation.get("task_id") != "PickCube-v1"
            or initial_validation.get("split") != "validation"
            or initial_validation.get("instruction_condition") != "correct"
            or initial_validation.get("episode_count") != 30
            or initial_validation.get("success_count") != 0
            or initial_validation.get("invalid_action_count") != 0
            or initial_validation.get("simulator_error_count") != 0
            or initial_validation.get("completed") is not True
            or initial_identities != [selected_checkpoint]
            or initial_validation.get("execution_horizon") != original_selected_horizon
            or not isinstance(repair_initial, Mapping)
            or repair_initial.get("fingerprint") != initial_validation.get("fingerprint")
            or not isinstance(repair_selected, Mapping)
            or repair_selected.get("fingerprint") != selected.get("fingerprint")
        ):
            raise Phase2CBContractError("bounded repair evidence contract failed")
        selected_horizon = repair["repaired_execution_horizon"]
    validation_checkpoints = validation.get("checkpoint_identities")
    checkpoint_records = training.get("checkpoint_records")
    if (
        not isinstance(selected_checkpoint, str)
        or not isinstance(selected_horizon, int)
        or not isinstance(validation_checkpoints, list)
        or validation_checkpoints != [selected_checkpoint]
        or validation.get("execution_horizon") != selected_horizon
        or not isinstance(checkpoint_records, list)
        or selected_checkpoint
        not in {
            str(record.get("sha256"))
            for record in checkpoint_records
            if isinstance(record, Mapping)
        }
    ):
        raise Phase2CBContractError("selected Pick policy identity is not closed")
    gate = classify_pick_gate(
        success_count=_integer(validation, "success_count"),
        episode_count=_integer(validation, "episode_count"),
        invalid_action_episode_count=_integer(validation, "invalid_action_count"),
        simulator_error_episode_count=_integer(validation, "simulator_error_count"),
        repair_already_used=bool(args.repair_used),
    )
    semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-b-pick-stage-verification-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "verify_stage": "pick_gate",
        "inputs": {key: _path_record(paths[key], documents[key]) for key in sorted(paths)},
        "selected_checkpoint_sha256": selected_checkpoint,
        "original_selected_execution_horizon": original_selected_horizon,
        "selected_execution_horizon": selected_horizon,
        "pick_competence_gate": gate,
        "repair_used": bool(args.repair_used),
        "other_full_models_authorized": gate["other_full_models_authorized"],
        "vla_jepa_training_authorized": False,
        "latentguard_authorized": False,
        "passed": gate["passed"],
    }
    result = {**semantic, "fingerprint": canonical_fingerprint(semantic)}
    _write_new_or_equal(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
