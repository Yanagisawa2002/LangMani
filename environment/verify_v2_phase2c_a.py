"""Independently verify the Phase 2C-A ACT Result D evidence package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2c_a"
CANONICAL_PACKAGE = "sha256:77675e2134e4886a97e4bdac2230c64c3da30c080e647433b7c701a79544ed04"
DIAGNOSTIC_COMMIT = "a3a09d30cba8a1f85c79953dc0c39e22712e4e9a"
TRAINING_COMMIT = "c0b5106cb1fff5b43f46500321264cec33613673"

EXPECTED_RUNS: dict[str, dict[str, object]] = {
    "pick": {
        "report": "train-pick-c0b5106.json",
        "steps": 2680,
        "examples": 1_092_160,
        "task_samples": {"PickCube-v1": 1_092_160, "StackCube-v1": 0, "PushCube-v1": 0},
        "selected": ("sha256:da7128d019689d190523c1f4c6f7f1e4060371ef24f203821afd98dbe316e0f3"),
    },
    "stack": {
        "report": "train-stack-c0b5106.json",
        "steps": 3680,
        "examples": 1_497_820,
        "task_samples": {"PickCube-v1": 0, "StackCube-v1": 1_497_820, "PushCube-v1": 0},
        "selected": ("sha256:9458aa3c9e499526282e3c48534de0562306d4a346973e0c594e7e7097415a61"),
    },
    "push": {
        "report": "train-push-c0b5106.json",
        "steps": 2380,
        "examples": 964_380,
        "task_samples": {"PickCube-v1": 0, "StackCube-v1": 0, "PushCube-v1": 964_380},
        "selected": ("sha256:6b51e3af0d4d42dc20c1175e6196b1eb3f7a2b15895555ee71e24ae0d0c167bd"),
    },
    "shared_seed0": {
        "report": "train-shared-seed0-c0b5106.json",
        "steps": 8712,
        "examples": 3_554_496,
        "task_samples": {
            "PickCube-v1": 1_184_832,
            "StackCube-v1": 1_184_832,
            "PushCube-v1": 1_184_832,
        },
        "selected": ("sha256:ada3cb1bd9dd9f752f60f2c9059746c8d0e2ae902fa4c5a04c88b8e96242d870"),
    },
}


class Phase2CAVerificationError(RuntimeError):
    """Raised when immutable Phase 2C-A evidence is incomplete or inconsistent."""


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2CAVerificationError(f"could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise Phase2CAVerificationError(f"{path} must contain one object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise Phase2CAVerificationError(f"{path}:{line_number} must contain one object")
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2CAVerificationError(f"could not read {path}: {error}") from error
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise Phase2CAVerificationError(f"could not hash {path}: {error}") from error
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    """Hash one JSON-compatible value without importing simulator-owned packages."""

    serialized = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Phase2CAVerificationError(f"{label} must be an object")
    return value


def validate_result_classification(document: Mapping[str, object]) -> None:
    """Validate Result D semantics without consulting external generated evidence."""

    unavailable = document.get("unavailable_metrics")
    if (
        document.get("schema_version") != "langmani-v2-phase2c-a-result-v0"
        or document.get("package_fingerprint") != CANONICAL_PACKAGE
        or document.get("result") != "RESULT_D"
        or document.get("result_reason")
        != "selected_pick_act_emitted_native_pd_joint_pos_out_of_bounds_gripper_actions"
        or document.get("hard_stop_stage") != "closed_loop_infrastructure_smoke"
        or document.get("training_complete") is not True
        or document.get("offline_diagnostics_complete") is not True
        or document.get("closed_loop_development_started") is not False
        or document.get("final_evaluation_started") is not False
        or document.get("shared_seed1_started") is not False
        or document.get("act_baselines_validated") is not False
        or document.get("smolvla_phase_eligible") is not False
        or document.get("smolvla_training_authorized") is not False
        or document.get("vla_jepa_training_authorized") is not False
        or not isinstance(unavailable, list)
        or "validation_success" not in unavailable
        or "unseen_reset_success" not in unavailable
        or "visual_shift_success" not in unavailable
    ):
        raise Phase2CAVerificationError("Result D classification contract differs")


def _verify_checkpoint_bytes(
    *,
    run_root: Path,
    report_record: Mapping[str, object],
) -> None:
    relative = report_record.get("relative_path")
    if not isinstance(relative, str):
        raise Phase2CAVerificationError("training report checkpoint path is malformed")
    checkpoint = (run_root / relative).resolve()
    if run_root.resolve() not in checkpoint.parents:
        raise Phase2CAVerificationError("checkpoint path escapes its run")
    manifest = _read_object(checkpoint / "checkpoint_manifest.json")
    complete = _read_object(checkpoint / "complete.json")
    record = _mapping(manifest.get("record"), label="checkpoint record")
    if (
        record.get("checkpoint_fingerprint") != report_record.get("checkpoint_fingerprint")
        or record.get("global_step") != report_record.get("global_step")
        or record.get("complete") is not True
        or complete.get("checkpoint_fingerprint") != record.get("checkpoint_fingerprint")
    ):
        raise Phase2CAVerificationError("checkpoint record or completion marker differs")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise Phase2CAVerificationError("checkpoint artifact manifest is empty")
    for item in artifacts:
        values = _mapping(item, label="checkpoint artifact")
        path_value = values.get("path")
        if not isinstance(path_value, str):
            raise Phase2CAVerificationError("checkpoint artifact path is malformed")
        path = (checkpoint / path_value).resolve()
        if checkpoint not in path.parents:
            raise Phase2CAVerificationError("checkpoint artifact escapes checkpoint root")
        if path.stat().st_size != values.get("size_bytes") or _sha256_file(path) != values.get(
            "sha256"
        ):
            raise Phase2CAVerificationError(f"checkpoint artifact differs: {path}")


def verify_phase2c_a(
    *,
    artifact_root: Path,
    evidence_root: Path,
    verify_checkpoint_bytes: bool = True,
) -> dict[str, object]:
    """Verify compact source evidence against immutable server outputs."""

    checks: dict[str, bool] = {}
    result = _read_object(artifact_root / "result_classification.json")
    validate_result_classification(result)
    checks["result_classification_valid"] = True

    identity = _read_object(artifact_root / "identity_resolution_verification.json")
    if identity.get("passed") is not True or identity.get("canonical_package_fingerprint") != (
        CANONICAL_PACKAGE
    ):
        raise Phase2CAVerificationError("identity-resolution verification differs")
    checks["identity_resolution_passed"] = True

    compact_training = _read_object(artifact_root / "training_summary.json")
    compact_runs = _mapping(compact_training.get("runs"), label="compact training runs")
    reports_root = evidence_root / "reports"
    observed_checkpoint_count = 0
    for model, expected in EXPECTED_RUNS.items():
        report = _read_object(reports_root / str(expected["report"]))
        outcome = _mapping(report.get("result"), label=f"{model} training outcome")
        compact = _mapping(compact_runs.get(model), label=f"{model} compact training outcome")
        if (
            report.get("passed") is not True
            or outcome.get("final_step") != expected["steps"]
            or outcome.get("examples_processed") != expected["examples"]
            or outcome.get("effective_samples_by_task") != expected["task_samples"]
            or compact.get("final_step") != expected["steps"]
            or compact.get("examples_processed") != expected["examples"]
            or compact.get("samples_by_task") != expected["task_samples"]
        ):
            raise Phase2CAVerificationError(f"{model} training identity or sample budget differs")
        records = outcome.get("checkpoint_records")
        if not isinstance(records, list) or len(records) != 4:
            raise Phase2CAVerificationError(f"{model} does not have four checkpoint records")
        run_root_value = outcome.get("run_root")
        if not isinstance(run_root_value, str):
            raise Phase2CAVerificationError(f"{model} run root is malformed")
        run_root = Path(run_root_value).resolve()
        for record_value in records:
            record = _mapping(record_value, label=f"{model} checkpoint record")
            if record.get("git_commit") != TRAINING_COMMIT or record.get("complete") is not True:
                raise Phase2CAVerificationError(f"{model} checkpoint producer differs")
            if verify_checkpoint_bytes:
                _verify_checkpoint_bytes(run_root=run_root, report_record=record)
            observed_checkpoint_count += 1
    checks["four_training_runs_complete"] = True
    checks["sixteen_checkpoints_recoverable"] = observed_checkpoint_count == 16
    checks["checkpoint_bytes_rehashed"] = verify_checkpoint_bytes

    diagnostic_paths = sorted((reports_root / "offline-a3a09d3").glob("*.json"))
    if len(diagnostic_paths) != 16:
        raise Phase2CAVerificationError("offline diagnostic count differs")
    diagnostic_by_fingerprint: dict[str, Mapping[str, object]] = {}
    for path in diagnostic_paths:
        diagnostic = _read_object(path)
        fingerprint = diagnostic.get("checkpoint_fingerprint")
        if (
            not isinstance(fingerprint, str)
            or diagnostic.get("passed") is not True
            or diagnostic.get("split") != "validation"
            or diagnostic.get("package_fingerprint") != CANONICAL_PACKAGE
            or diagnostic.get("output_finite") is not True
            or diagnostic.get("constant_action_prediction") is not False
            or not isinstance(diagnostic.get("total_action_loss"), int | float)
            or not math.isfinite(float(diagnostic["total_action_loss"]))
        ):
            raise Phase2CAVerificationError(f"offline diagnostic differs: {path}")
        diagnostic_by_fingerprint[fingerprint] = diagnostic
    checks["sixteen_offline_diagnostics_passed"] = len(diagnostic_by_fingerprint) == 16

    selection_root = reports_root / "checkpoint-selections-a3a09d3"
    for model, expected in EXPECTED_RUNS.items():
        external_name = "shared-seed0" if model == "shared_seed0" else model
        selection = _read_object(selection_root / f"{external_name}.json")
        selected_record = _mapping(
            selection.get("selected_checkpoint_record"),
            label=f"{model} selected checkpoint",
        )
        if (
            selection.get("package_fingerprint") != CANONICAL_PACKAGE
            or selection.get("producer_git_commit") != DIAGNOSTIC_COMMIT
            or selected_record.get("checkpoint_fingerprint") != expected["selected"]
            or expected["selected"] not in diagnostic_by_fingerprint
        ):
            raise Phase2CAVerificationError(f"{model} selection lock differs")
    checks["validation_only_selection_locks_passed"] = True

    smoke_root = evidence_root / "evaluation-smoke" / "pick-h4-a3a09d3"
    summary = _read_object(smoke_root / "summary.json")
    episodes = _read_jsonl(smoke_root / "episodes.jsonl")
    if len(episodes) != 1:
        raise Phase2CAVerificationError("closed-loop smoke episode count differs")
    episode = episodes[0]
    if (
        summary.get("episode_count") != 1
        or summary.get("invalid_action_count") != 1
        or summary.get("simulator_error_count") != 0
        or summary.get("success_count") != 0
        or summary.get("execution_horizon") != 4
        or episode.get("checkpoint_identity") != EXPECTED_RUNS["pick"]["selected"]
        or episode.get("outcome") != "invalid_policy_output"
        or episode.get("failure_reason") != "policy action exceeded native pd_joint_pos bounds"
        or episode.get("episode_length") != 0
        or episode.get("actions_executed") != 0
        or episode.get("policy_query_count") != 1
        or episode.get("simulator_error") is not False
    ):
        raise Phase2CAVerificationError("closed-loop hard-stop evidence differs")
    checks["closed_loop_smoke_hard_stopped_before_step"] = True

    forbidden_patterns = (
        "train-shared-seed1-*.json",
        "*test_unseen_reset*",
        "*test_visual_shift*",
        "*task-condition-intervention*result*",
    )
    if any(list(evidence_root.rglob(pattern)) for pattern in forbidden_patterns):
        raise Phase2CAVerificationError("post-hard-stop evidence unexpectedly exists")
    checks["post_hard_stop_stages_absent"] = True

    semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-a-result-verification-v0",
        "package_fingerprint": CANONICAL_PACKAGE,
        "result": "RESULT_D",
        "checks": checks,
        "passed": all(checks.values()),
        "training_commit": TRAINING_COMMIT,
        "diagnostic_commit": DIAGNOSTIC_COMMIT,
        "smolvla_training_authorized": False,
        "vla_jepa_training_authorized": False,
    }
    return {**semantic, "fingerprint": f"sha256:{_sha256_json(semantic)}"}


def _write_new_or_equal(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        if _read_object(path) != value:
            raise Phase2CAVerificationError(f"existing verification report differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.staging")
    staging.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(staging, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--skip-checkpoint-bytes",
        action="store_true",
        help="Fixture-only mode; production verification must not use this option.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = verify_phase2c_a(
        artifact_root=args.artifact_root.resolve(),
        evidence_root=args.evidence_root.resolve(),
        verify_checkpoint_bytes=not args.skip_checkpoint_bytes,
    )
    _write_new_or_equal(args.report.resolve(), report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
