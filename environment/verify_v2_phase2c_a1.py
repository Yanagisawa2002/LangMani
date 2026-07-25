"""Independently verify Phase 2C-A.1 training, evaluation, and ACT closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

import numpy as np

from langmani.datasets.identity import sha256_hex
from langmani.v2.phase2b5 import PANDA_ACTION_HIGH, PANDA_ACTION_LOW
from langmani.v2.phase2c_a import PACKAGE_FINGERPRINT, TASK_IDS, ModelKind

EXPECTED_STEPS = {
    ModelKind.PICK.value: (670, 1340, 2010, 2680),
    ModelKind.STACK.value: (920, 1840, 2760, 3680),
    ModelKind.PUSH.value: (595, 1190, 1785, 2380),
    ModelKind.SHARED.value: (2178, 4356, 6534, 8712),
}
TRAIN_REPORTS = {
    ModelKind.PICK.value: "train-pick.json",
    ModelKind.STACK.value: "train-stack.json",
    ModelKind.PUSH.value: "train-push.json",
    ModelKind.SHARED.value: "train-shared.json",
}


class Phase2CA1VerificationError(RuntimeError):
    """Raised when final evidence does not satisfy the independent verifier."""


class CheckpointVerification(TypedDict):
    checkpoint_fingerprint: object
    artifact_count: int
    artifact_bytes: int
    all_artifacts_rehashed: bool


class EvaluationGroupVerification(TypedDict):
    episode_count: int
    action_count: int
    invalid_action_count: int
    simulator_error_count: int
    runtime_manifest_fingerprint: object


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2CA1VerificationError(f"could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise Phase2CA1VerificationError(f"{path} must contain one object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise Phase2CA1VerificationError(f"{path}:{line_number} is not one object")
            rows.append(value)
    if not rows:
        raise Phase2CA1VerificationError(f"{path} is empty")
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _fingerprint_matches(document: Mapping[str, object]) -> bool:
    semantic = dict(document)
    fingerprint = semantic.pop("fingerprint", None)
    return fingerprint == f"sha256:{sha256_hex(semantic)}"


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _valid_selection(selection: Mapping[str, object]) -> bool:
    selected = selection.get("selected")
    return (
        selection.get("selection_split") == "validation"
        and selection.get("test_outcomes_available") is False
        and isinstance(selected, list)
        and len(selected) == 1
        and selection.get("passed") is True
    )


def _verify_checkpoint(run_root: Path, relative_path: str) -> CheckpointVerification:
    checkpoint_root = run_root / relative_path
    manifest = _read_object(checkpoint_root / "checkpoint_manifest.json")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise Phase2CA1VerificationError("checkpoint manifest has no artifact registry")
    total_bytes = 0
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise Phase2CA1VerificationError("checkpoint artifact record is malformed")
        raw_path = artifact.get("path")
        size = artifact.get("size_bytes")
        digest = artifact.get("sha256")
        if (
            not isinstance(raw_path, str)
            or not isinstance(size, int)
            or not isinstance(digest, str)
        ):
            raise Phase2CA1VerificationError("checkpoint artifact identity is malformed")
        path = checkpoint_root / raw_path
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != size
            or _sha256_file(path) != digest
        ):
            raise Phase2CA1VerificationError(f"checkpoint artifact changed: {path}")
        total_bytes += size
    complete = _read_object(checkpoint_root / "complete.json")
    if complete.get("checkpoint_fingerprint") != manifest.get("checkpoint_fingerprint"):
        raise Phase2CA1VerificationError("checkpoint completion identity changed")
    return {
        "checkpoint_fingerprint": manifest.get("checkpoint_fingerprint"),
        "artifact_count": len(artifacts),
        "artifact_bytes": total_bytes,
        "all_artifacts_rehashed": True,
    }


def _verify_action_sequence(record: Mapping[str, object]) -> int:
    value = record.get("action_sequence")
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        return 0
    if (
        array.ndim != 2
        or array.shape[1:] != (8,)
        or not np.isfinite(array).all()
        or np.any(array < PANDA_ACTION_LOW)
        or np.any(array > PANDA_ACTION_HIGH)
    ):
        raise Phase2CA1VerificationError("evaluation action trace violates native bounds")
    semantic = tuple(tuple(float(component) for component in row) for row in array)
    if record.get("action_sequence_fingerprint") != f"sha256:{sha256_hex(semantic)}":
        raise Phase2CA1VerificationError("evaluation action trace fingerprint changed")
    return len(array)


def _verify_evaluation_group(
    path: Path,
    *,
    expected_count: int,
    expected_evaluation_commit: str,
    final_policy_lock: str | None,
) -> EvaluationGroupVerification:
    rows = _read_jsonl(path / "episodes.jsonl")
    manifest = _read_object(path / "runtime_manifest.json")
    summary = _read_object(path / "summary.json")
    if (
        len(rows) != expected_count
        or summary.get("episode_count") != expected_count
        or summary.get("completed") is not True
        or manifest.get("package_fingerprint") != PACKAGE_FINGERPRINT
        or manifest.get("action_clipping") is not False
        or manifest.get("action_projection") is not False
        or manifest.get("bounded_action_head_v1_required") is not True
        or manifest.get("evaluation_git_commit") != expected_evaluation_commit
        or manifest.get("final_policy_lock_fingerprint") != final_policy_lock
        or not _fingerprint_matches(manifest)
    ):
        raise Phase2CA1VerificationError(f"evaluation group is incomplete: {path}")
    action_count = sum(_verify_action_sequence(record) for record in rows)
    invalid_count = sum(record.get("invalid_action") is True for record in rows)
    simulator_count = sum(record.get("simulator_error") is True for record in rows)
    return {
        "episode_count": len(rows),
        "action_count": action_count,
        "invalid_action_count": invalid_count,
        "simulator_error_count": simulator_count,
        "runtime_manifest_fingerprint": manifest.get("fingerprint"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--expected-training-commit", required=True)
    parser.add_argument("--expected-evaluation-commit", required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    artifact_root = args.artifact_root.resolve()
    evidence_root = args.evidence_root.resolve()
    evaluation_root = args.evaluation_root.resolve()
    for commit in (args.expected_training_commit, args.expected_evaluation_commit):
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise Phase2CA1VerificationError("expected commits must be full Git SHAs")
    checks: dict[str, object] = {}

    input_verification = _read_object(artifact_root / "input_verification.json")
    checks["input_verification"] = (
        input_verification.get("canonical_package_fingerprint") == PACKAGE_FINGERPRINT
        and _mapping(input_verification.get("phase2c_a_result_verification")).get("passed") is True
        and _mapping(input_verification.get("phase2c_a_terminal_observation")).get(
            "environment_steps"
        )
        == 0
    )
    bounds = _read_object(artifact_root / "action_bound_manifest.json")
    checks["native_action_bounds"] = (
        _fingerprint_matches(bounds)
        and bounds.get("lower") == PANDA_ACTION_LOW.tolist()
        and bounds.get("upper") == PANDA_ACTION_HIGH.tolist()
        and bounds.get("task_consistency_verified") is True
    )
    bounded_head = _read_object(artifact_root / "bounded_head_manifest.json")
    checks["bounded_head"] = (
        _fingerprint_matches(bounded_head)
        and bounded_head.get("identity") == "bounded_action_head_v1"
        and bounded_head.get("clipping") is False
        and bounded_head.get("projection") is False
        and bounded_head.get("action_normalization_mode") == "IDENTITY"
    )
    random_audit = _read_object(artifact_root / "action_100k_audit.json")
    checks["random_action_audit"] = (
        _fingerprint_matches(random_audit)
        and random_audit.get("sampled_chunks") == 100_000
        and random_audit.get("sampled_actions") == 1_600_000
        and random_audit.get("nonfinite_actions") == 0
        and random_audit.get("lower_bound_violations") == 0
        and random_audit.get("upper_bound_violations") == 0
        and random_audit.get("clipping_events") == 0
        and random_audit.get("projection_events") == 0
        and random_audit.get("passed") is True
    )
    padding = _read_object(artifact_root / "padding_loss_audit.json")
    checks["padding_loss"] = (
        padding.get("padded_timesteps_contribute_zero_action_loss") is True
        and padding.get("unpadded_timesteps_contribute_normally") is True
        and padding.get("passed") is True
    )
    smoke = _read_object(artifact_root / "smoke_result.json")
    pick_micro = _read_object(artifact_root / "micro_pick_result.json")
    shared_micro = _read_object(artifact_root / "micro_shared_result.json")
    checks["gpu_smoke_and_micro"] = all(
        document.get("passed") is True
        and document.get("bounded_action_head_v1") is True
        and document.get("package_fingerprint") == PACKAGE_FINGERPRINT
        for document in (smoke, pick_micro, shared_micro)
    )

    checkpoint_verification: list[dict[str, object]] = []
    run_fingerprints: set[str] = set()
    for model_kind, filename in TRAIN_REPORTS.items():
        report = _read_object(evidence_root / "reports" / filename)
        identity = report.get("identity")
        result = report.get("result")
        if (
            report.get("passed") is not True
            or report.get("model_kind") != model_kind
            or not isinstance(identity, dict)
            or identity.get("git_commit") != args.expected_training_commit
            or identity.get("package_fingerprint") != PACKAGE_FINGERPRINT
            or not isinstance(result, dict)
            or result.get("final_step") != EXPECTED_STEPS[model_kind][-1]
        ):
            raise Phase2CA1VerificationError(f"training report is invalid: {filename}")
        run_root = Path(str(result["run_root"])).resolve()
        records = result.get("checkpoint_records")
        if not isinstance(records, list) or [
            record.get("global_step") for record in records if isinstance(record, dict)
        ] != list(EXPECTED_STEPS[model_kind]):
            raise Phase2CA1VerificationError("training checkpoint schedule changed")
        run_fingerprints.add(str(result["run_fingerprint"]))
        for record in records:
            assert isinstance(record, dict)
            verified = _verify_checkpoint(run_root, str(record["relative_path"]))
            if verified["checkpoint_fingerprint"] != record.get("checkpoint_fingerprint"):
                raise Phase2CA1VerificationError("checkpoint record identity changed")
            checkpoint_verification.append({"model_kind": model_kind, **verified})
    checks["training_runs"] = len(run_fingerprints) == 4
    checks["sixteen_checkpoints_rehashed"] = len(checkpoint_verification) == 16

    screens = [
        _read_object(path) for path in sorted((evidence_root / "reports/screens").glob("*.json"))
    ]
    checks["checkpoint_screens"] = len(screens) == 16 and all(
        screen.get("schema_version") == "langmani-v2-phase2c-a1-checkpoint-screen-v0"
        and _fingerprint_matches(screen)
        and screen.get("policy_query_count") == 10_000
        and screen.get("lower_bound_violations") == 0
        and screen.get("upper_bound_violations") == 0
        and screen.get("passed") is True
        for screen in screens
    )
    selections = [
        _read_object(evidence_root / "reports/selections" / f"{kind}.json")
        for kind in EXPECTED_STEPS
    ]
    checks["checkpoint_selections"] = all(
        _fingerprint_matches(selection) and _valid_selection(selection) for selection in selections
    )

    smoke_group = _verify_evaluation_group(
        evaluation_root / "closed_loop_smoke",
        expected_count=1,
        expected_evaluation_commit=args.expected_evaluation_commit,
        final_policy_lock=None,
    )
    checks["closed_loop_smoke"] = (
        smoke_group["action_count"] >= 1
        and smoke_group["invalid_action_count"] == 0
        and smoke_group["simulator_error_count"] == 0
    )
    horizon = _read_object(evidence_root / "reports/horizon-selection.json")
    checks["execution_horizon"] = (
        horizon.get("selection_split") == "validation"
        and horizon.get("selected_execution_horizon") in {1, 4, 8}
        and horizon.get("final_test_outcomes_available") is False
        and horizon.get("passed") is True
    )
    final_lock = _read_object(evidence_root / "reports/final-policy-lock.json")
    final_lock_fingerprint = final_lock.get("fingerprint")
    checks["final_policy_lock"] = (
        _fingerprint_matches(final_lock)
        and final_lock.get("evaluation_git_commit") == args.expected_evaluation_commit
        and final_lock.get("execution_horizon") == horizon.get("selected_execution_horizon")
        and final_lock.get("final_results_available") is False
        and final_lock.get("settings_mutable_after_final_results") is False
    )

    evaluation_groups: list[EvaluationGroupVerification] = []
    for scope in ("per_task", "shared"):
        for task_id in TASK_IDS:
            evaluation_groups.append(
                _verify_evaluation_group(
                    evaluation_root / "development" / scope / task_id,
                    expected_count=30,
                    expected_evaluation_commit=args.expected_evaluation_commit,
                    final_policy_lock=None,
                )
            )
            for split in ("test_unseen_reset", "test_visual_shift"):
                evaluation_groups.append(
                    _verify_evaluation_group(
                        evaluation_root / "final" / scope / split / task_id,
                        expected_count=30,
                        expected_evaluation_commit=args.expected_evaluation_commit,
                        final_policy_lock=str(final_lock_fingerprint),
                    )
                )
    checks["development_and_final_evaluations"] = (
        len(evaluation_groups) == 18
        and sum(group["episode_count"] for group in evaluation_groups) == 540
        and sum(group["invalid_action_count"] for group in evaluation_groups) == 0
        and sum(group["simulator_error_count"] for group in evaluation_groups) == 0
        and all(group["action_count"] > 0 for group in evaluation_groups)
    )
    intervention = _read_object(evidence_root / "reports/task-intervention-analysis.json")
    checks["task_intervention"] = (
        _fingerprint_matches(intervention)
        and intervention.get("split") == "validation"
        and intervention.get("language_grounding_claimed") is False
        and intervention.get("passed") is True
    )
    analysis = _read_object(evidence_root / "reports/result-analysis.json")
    checks["result_and_closure"] = (
        _fingerprint_matches(analysis)
        and analysis.get("result") in {"RESULT_A", "RESULT_B", "RESULT_C"}
        and analysis.get("act_baselines_validated") is True
        and analysis.get("act_phase_closed") is True
        and analysis.get("further_act_architecture_authorized") is False
        and analysis.get("smolvla_phase_eligible") is True
        and analysis.get("smolvla_training_authorized") is False
        and analysis.get("vla_jepa_training_authorized") is False
    )

    current_commit = _git(repository_root, "rev-parse", "HEAD")
    checks["repository"] = (
        re.fullmatch(r"[0-9a-f]{40}", current_commit) is not None
        and _git(repository_root, "status", "--porcelain") == ""
    )
    failed = [name for name, passed in checks.items() if passed is not True]
    result_semantic = {
        "schema_version": "langmani-v2-phase2c-a1-final-verification-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "training_git_commit": args.expected_training_commit,
        "evaluation_git_commit": args.expected_evaluation_commit,
        "verifier_git_commit": current_commit,
        "checks": checks,
        "checkpoint_verification": checkpoint_verification,
        "result": analysis.get("result"),
        "act_phase_closed": analysis.get("act_phase_closed"),
        "smolvla_phase_eligible": analysis.get("smolvla_phase_eligible"),
        "smolvla_training_authorized": False,
        "vla_jepa_training_authorized": False,
        "failed_checks": failed,
        "passed": not failed,
    }
    result = {
        **result_semantic,
        "fingerprint": f"sha256:{sha256_hex(result_semantic)}",
    }
    destination = args.report.resolve()
    if destination.exists():
        if _read_object(destination) != result:
            raise Phase2CA1VerificationError(
                f"existing immutable verifier report differs: {destination}"
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
