"""Create the compact, result-conditioned Phase 2C-C evidence package."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path

from langmani.v2.phase2c_a import PACKAGE_FINGERPRINT
from langmani.v2.phase2c_c import (
    Phase2CCContractError,
    canonical_fingerprint,
    classify_action_formulation,
)

STATIC_DOCUMENTS = {
    "input_verification.json": "input_verification.json",
    "state_action_audit.json": "state_action_audit.json",
    "residual_transform_manifest.json": "residual_transform_manifest.json",
    "action_100k_audit.json": "action_100k_audit.json",
    "reconstruction_audit.json": "reconstruction_audit.json",
    "relative_training_config.json": "relative_training_config.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-root", type=Path, required=True)
    parser.add_argument("--baseline-phase-diagnostic", type=Path, required=True)
    parser.add_argument("--absolute-training-reset", type=Path, required=True)
    parser.add_argument("--smoke-training", type=Path, required=True)
    parser.add_argument("--smoke-environment-step", type=Path, required=True)
    parser.add_argument("--micro-training", type=Path, required=True)
    parser.add_argument("--micro-environment-step", type=Path, required=True)
    parser.add_argument("--full-training", type=Path, required=True)
    parser.add_argument("--checkpoint-registry", type=Path, required=True)
    parser.add_argument("--offline-diagnostic", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint-selection", type=Path, required=True)
    parser.add_argument("--horizon-screen", type=Path, action="append", required=True)
    parser.add_argument("--policy-selection", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--relative-training-reset", type=Path, required=True)
    parser.add_argument("--unseen-reset", type=Path)
    parser.add_argument("--final-git-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


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
        raise Phase2CCContractError(f"artifact fingerprint changed: {path}")
    return value


def _write(path: Path, value: Mapping[str, object]) -> None:
    encoded = json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise Phase2CCContractError(f"existing compact artifact differs: {path}")
        return
    path.write_text(encoded, encoding="utf-8", newline="\n")


def _fingerprinted(semantic: Mapping[str, object]) -> dict[str, object]:
    value = dict(semantic)
    return {**value, "fingerprint": canonical_fingerprint(value)}


def _require_summary(
    path: Path,
    *,
    split_role: str,
    episode_count: int,
    formulation: str,
) -> dict[str, object]:
    value = _read(path)
    if (
        value.get("schema_version") != "langmani-v2-phase2c-c-evaluation-summary-v0"
        or value.get("package_fingerprint") != PACKAGE_FINGERPRINT
        or value.get("split_role") != split_role
        or value.get("episode_count") != episode_count
        or value.get("formulation") != formulation
        or value.get("completed") is not True
        or value.get("invalid_action_count") != 0
        or value.get("simulator_error_count") != 0
    ):
        raise Phase2CCContractError(f"evaluation summary is invalid: {path}")
    return value


def _training_result(path: Path, *, stage: str, steps: int) -> dict[str, object]:
    value = _read(path)
    if (
        value.get("schema_version") != "langmani-v2-phase2c-c-training-result-v0"
        or value.get("package_fingerprint") != PACKAGE_FINGERPRINT
        or value.get("stage") != stage
        or value.get("optimizer_steps") != steps
        or value.get("completed") is not True
        or value.get("passed") is not True
        or value.get("clipping_events") != 0
        or value.get("projection_events") != 0
    ):
        raise Phase2CCContractError(f"relative training result is invalid: {path}")
    return value


def _evaluation_reference(value: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "fingerprint",
        "split_role",
        "episode_count",
        "success_count",
        "timeout_count",
        "entered_grasp_region_count",
        "ever_grasped_count",
        "ever_lifted_count",
        "failed_grasp_count",
        "no_motion_count",
        "object_drop_count",
        "invalid_action_count",
        "simulator_error_count",
        "mean_episode_length",
        "inference_latency_p50_ms",
        "inference_latency_p95_ms",
        "execution_horizon",
        "checkpoint_identities",
        "schedule_fingerprint",
    )
    return {key: value.get(key) for key in keys}


def _training_reference(value: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "fingerprint",
        "stage",
        "optimizer_steps",
        "initial_fixed_batch_loss",
        "final_fixed_batch_loss",
        "fixed_batch_loss_decreased",
        "physical_action_error_decreased",
        "residual_action_error_decreased",
        "last_training_loss",
        "last_gradient_norm",
        "training_duration_seconds",
        "peak_gpu_allocated_bytes",
        "peak_gpu_reserved_bytes",
        "checkpoint_records",
        "final_checkpoint_reload",
    )
    return {key: value.get(key) for key in keys}


def _registry(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise Phase2CCContractError("checkpoint registry contains a non-object")
        rows.append(value)
    if (
        len(rows) != 3
        or [row.get("optimizer_step") for row in rows] != [5_000, 10_000, 20_000]
        or any(not isinstance(row.get("sha256"), str) for row in rows)
    ):
        raise Phase2CCContractError("checkpoint registry does not contain the frozen queue")
    return rows


def main() -> int:
    args = parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.final_git_commit) is None:
        raise Phase2CCContractError("--final-git-commit must be one full SHA")
    static_root = args.static_root.resolve()
    static = {name: _read(static_root / source) for name, source in STATIC_DOCUMENTS.items()}
    completion = _read(static_root / "static_preparation_complete.json")
    if (
        completion.get("schema_version") != "langmani-v2-phase2c-c-static-preparation-complete-v1"
        or completion.get("passed") is not True
        or completion.get("optimizer_created") is not False
        or completion.get("training_started") is not False
        or any(
            value.get("package_fingerprint") != PACKAGE_FINGERPRINT
            for name, value in static.items()
            if name != "action_100k_audit.json"
        )
        or static["action_100k_audit.json"].get("transform_fingerprint")
        != static["residual_transform_manifest.json"].get("fingerprint")
    ):
        raise Phase2CCContractError("accepted Phase 2C-C static preparation changed")

    baseline = _read(args.baseline_phase_diagnostic.resolve())
    absolute_train = _require_summary(
        args.absolute_training_reset.resolve(),
        split_role="training_reset",
        episode_count=30,
        formulation="absolute",
    )
    smoke_training = _training_result(args.smoke_training.resolve(), stage="smoke", steps=1)
    smoke_step = _require_summary(
        args.smoke_environment_step.resolve(),
        split_role="smoke",
        episode_count=1,
        formulation="relative",
    )
    micro_training = _training_result(args.micro_training.resolve(), stage="micro", steps=500)
    micro_step = _require_summary(
        args.micro_environment_step.resolve(),
        split_role="smoke",
        episode_count=1,
        formulation="relative",
    )
    full_training = _training_result(args.full_training.resolve(), stage="full", steps=20_000)
    registry = _registry(args.checkpoint_registry.resolve())

    diagnostics = [_read(path.resolve()) for path in args.offline_diagnostic]
    if (
        len(diagnostics) != 3
        or {row.get("optimizer_step") for row in diagnostics} != {5_000, 10_000, 20_000}
        or any(
            row.get("schema_version") != "langmani-v2-phase2c-c-offline-checkpoint-diagnostic-v0"
            or row.get("formulation") != "relative"
            or row.get("passed") is not True
            for row in diagnostics
        )
    ):
        raise Phase2CCContractError("relative offline diagnostics are incomplete")
    checkpoint_selection = _read(args.checkpoint_selection.resolve())
    horizon_screens = [
        _require_summary(
            path.resolve(),
            split_role="validation_horizon_screen",
            episode_count=6,
            formulation="relative",
        )
        for path in args.horizon_screen
    ]
    policy_selection = _read(args.policy_selection.resolve())
    if (
        len(horizon_screens) != 2
        or {row.get("execution_horizon") for row in horizon_screens} != {1, 8}
        or checkpoint_selection.get("schema_version")
        != "langmani-v2-phase2c-c-checkpoint-selection-v0"
        or policy_selection.get("schema_version") != "langmani-v2-phase2c-c-policy-selection-v0"
        or checkpoint_selection.get("passed") is not True
        or policy_selection.get("passed") is not True
    ):
        raise Phase2CCContractError("validation-only policy selection changed")
    selected_sha = policy_selection.get("selected_checkpoint_sha256")
    selected_horizon = policy_selection.get("selected_execution_horizon")
    validation = _require_summary(
        args.validation.resolve(),
        split_role="validation",
        episode_count=30,
        formulation="relative",
    )
    relative_train = _require_summary(
        args.relative_training_reset.resolve(),
        split_role="training_reset",
        episode_count=30,
        formulation="relative",
    )
    if (
        validation.get("checkpoint_identities") != [selected_sha]
        or relative_train.get("checkpoint_identities") != [selected_sha]
        or validation.get("execution_horizon") != selected_horizon
        or relative_train.get("execution_horizon") != selected_horizon
        or validation.get("schedule_fingerprint") != relative_train.get("schedule_fingerprint")
        or absolute_train.get("schedule_fingerprint") != relative_train.get("schedule_fingerprint")
    ):
        raise Phase2CCContractError("selected policy evaluation identities changed")
    validation_success = int(validation["success_count"])
    training_success = int(relative_train["success_count"])
    classification = classify_action_formulation(
        validation_success_count=validation_success,
        training_reset_success_count=training_success,
        pipeline_valid=True,
    )
    unseen = None
    if classification["case"] == "CASE_A":
        if args.unseen_reset is None:
            raise Phase2CCContractError("Case A requires the authorized unseen-reset result")
        unseen = _require_summary(
            args.unseen_reset.resolve(),
            split_role="test_unseen_reset",
            episode_count=50,
            formulation="relative",
        )
        if (
            unseen.get("checkpoint_identities") != [selected_sha]
            or unseen.get("execution_horizon") != selected_horizon
        ):
            raise Phase2CCContractError("unseen-reset policy identity changed")
    elif args.unseen_reset is not None:
        raise Phase2CCContractError("sub-threshold validation cannot package unseen-reset results")

    output = args.output_root.resolve()
    payloads: dict[str, dict[str, object]] = dict(static)
    payloads["baseline_phase_diagnostics.json"] = baseline
    payloads["absolute_training_reset_evaluation.json"] = absolute_train
    payloads["smoke_micro_result.json"] = _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2c-c-smoke-micro-result-v0",
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "smoke_training": _training_reference(smoke_training),
            "smoke_environment_step": _evaluation_reference(smoke_step),
            "micro_training": _training_reference(micro_training),
            "micro_environment_step": _evaluation_reference(micro_step),
            "real_environment_steps_completed": 2,
            "passed": True,
        }
    )
    payloads["training_metrics.json"] = _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2c-c-training-metrics-v0",
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "full_training": _training_reference(full_training),
            "offline_diagnostics": sorted(
                diagnostics, key=lambda value: int(value["optimizer_step"])
            ),
            "checkpoint_selection": checkpoint_selection,
            "passed": True,
        }
    )
    payloads["validation_result.json"] = _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2c-c-validation-result-v0",
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "horizon_screens": sorted(
                (_evaluation_reference(value) for value in horizon_screens),
                key=lambda value: int(value["execution_horizon"]),
            ),
            "policy_selection": policy_selection,
            "validation": _evaluation_reference(validation),
            "passed": True,
        }
    )
    payloads["relative_training_reset_result.json"] = relative_train
    payloads["result_classification.json"] = classification
    case = str(classification["case"])
    recommendation = {
        "CASE_A": "rebuild_multi_skill_smolvla_around_relative_action_contract",
        "CASE_B": "observation_view_ablation_or_pivot_to_standard_benchmark",
        "CASE_C": "pivot_to_standard_benchmark_with_known_working_baseline",
        "CASE_D": "repair_consumer_pipeline_without_model_quality_claim",
    }[case]
    payloads["next_route_decision.json"] = _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2c-c-next-route-decision-v0",
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "case": case,
            "recommendation": recommendation,
            "start_next_stage_automatically": False,
            "add_data_automatically": False,
            "passed": True,
        }
    )
    payloads["authorization_state.json"] = _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2c-c-authorization-state-v0",
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "case": case,
            "relative_action_formulation_validated": classification[
                "relative_action_formulation_validated"
            ],
            "langmani_custom_model_route_eligible": classification[
                "langmani_custom_model_route_eligible"
            ],
            "shared_smolvla_training_started": False,
            "stack_smolvla_training_started": False,
            "push_smolvla_training_started": False,
            "vla_jepa_authorized": False,
            "vla_jepa_training_started": False,
            "phase2c_c_1_authorized": False,
            "passed": True,
        }
    )
    if unseen is not None:
        payloads["unseen_reset_result.json"] = unseen

    for name, value in payloads.items():
        _write(output / name, value)
    registry_target = output / "checkpoint_registry.jsonl"
    registry_encoded = "".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in registry
    )
    registry_target.parent.mkdir(parents=True, exist_ok=True)
    if registry_target.exists():
        if registry_target.read_text(encoding="utf-8") != registry_encoded:
            raise Phase2CCContractError("existing compact checkpoint registry differs")
    else:
        registry_target.write_text(registry_encoded, encoding="utf-8", newline="\n")

    allowed = set(payloads) | {"checkpoint_registry.jsonl", "artifact_manifest.json"}
    unexpected = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.relative_to(output).as_posix() not in allowed
    }
    if unexpected:
        raise Phase2CCContractError(
            f"compact artifact root contains unexpected files: {unexpected}"
        )
    records = []
    for path in sorted(
        (
            path
            for path in output.iterdir()
            if path.is_file() and path.name != "artifact_manifest.json"
        ),
        key=lambda path: path.name,
    ):
        records.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2c-c-artifact-manifest-v0",
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "final_git_commit": args.final_git_commit,
            "case": case,
            "file_count": len(records),
            "total_bytes": sum(int(record["bytes"]) for record in records),
            "records": records,
            "generated_data_or_checkpoint_included": False,
            "passed": True,
        }
    )
    _write(output / "artifact_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
