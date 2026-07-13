"""Verify M3A contracts, one-group target smoke, or full target archive acceptance."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from mani_skill.utils.wrappers.record import RecordEpisode

from langmani.collection.command_support import validated_dataset_root, write_json_atomic
from langmani.collection.inspection import inspect_raw_dataset, source_archive_digest
from langmani.collection.manifest import load_manifest
from langmani.datasets.schedule import CANONICAL_TASK_SPECS, build_collection_schedule
from langmani.datasets.types import (
    CollectionConfig,
    CollectionStatus,
    ReplayValidationMode,
    ReplayValidationResult,
)
from langmani.experts.runtime import query_planner_runtime_versions, resolve_planner_python

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
OUTPUT_DIR = OUTPUT_ROOT / "diagnostics" / "m3a"
REPORT_PATH = OUTPUT_DIR / "verification.json"
DEFAULT_DATASET_ROOT = OUTPUT_ROOT / "datasets" / "m3a" / "langmani-pick-place-raw-v1"
SMOKE_RUNS_ROOT = OUTPUT_ROOT / "datasets" / "m3a" / "target-smoke-runs"

PRIOR_TARGET_GATE = "ordered M0/M1/M2 target gate"
SMOKE_COLLECTION = "M3A one-group smoke collection"
SMOKE_INSPECTION = "M3A one-group smoke structural inspection"
SMOKE_REPLAY = "M3A independent six-episode smoke action replay"
FULL_COLLECTION = "M3A 60-group collection/resume"
FULL_INSPECTION = "M3A independent full-dataset structural inspection"
FULL_REPLAY = "M3A independent 360-episode action replay"

COMMAND_REPORTS = {
    PRIOR_TARGET_GATE: OUTPUT_ROOT / "diagnostics" / "m2" / "verification.json",
    SMOKE_COLLECTION: OUTPUT_DIR / "collection_command.json",
    SMOKE_INSPECTION: OUTPUT_DIR / "inspection.json",
    SMOKE_REPLAY: OUTPUT_DIR / "replay_validation.json",
    FULL_COLLECTION: OUTPUT_DIR / "collection_command.json",
    FULL_INSPECTION: OUTPUT_DIR / "inspection.json",
    FULL_REPLAY: OUTPUT_DIR / "replay_validation.json",
}
ARCHIVE_REPORT_EXPECTATIONS = {
    SMOKE_COLLECTION: (1, 6),
    SMOKE_INSPECTION: (1, 6),
    FULL_COLLECTION: (60, 360),
    FULL_INSPECTION: (60, 360),
}
REPLAY_REPORT_EXPECTATIONS = {
    SMOKE_REPLAY: (1, 6),
    FULL_REPLAY: (60, 360),
}

VerificationMode = Literal["structural", "target_smoke", "target_full"]
FullArchiveAction = Literal["collect", "reuse", "fail"]


def _validated_source_archive_identity(dataset_root: Path) -> dict[str, str | None]:
    """Return content-bound provenance only for a currently valid M3A archive."""
    manifest_path = dataset_root / "manifests" / "collection_manifest.json"
    if not manifest_path.is_file():
        return {
            "source_archive_identity_status": "absent",
            "source_archive_identity_error": None,
            "collection_run_id": None,
            "config_fingerprint": None,
            "source_archive_digest": None,
        }
    try:
        # Inspection is intentionally repeated at report time. This prevents a
        # successful verification report from binding only a parseable manifest
        # after a source shard or deterministic projection has changed.
        inspect_raw_dataset(dataset_root)
        manifest = load_manifest(dataset_root)
        return {
            "source_archive_identity_status": "validated",
            "source_archive_identity_error": None,
            "collection_run_id": manifest.collection_run_id,
            "config_fingerprint": manifest.schedule.config_fingerprint,
            "source_archive_digest": source_archive_digest(manifest),
        }
    except (OSError, TypeError, ValueError, KeyError, RuntimeError) as error:
        return {
            "source_archive_identity_status": "invalid",
            "source_archive_identity_error": (
                f"{type(error).__name__}: {str(error) or repr(error)}"
            ),
            "collection_run_id": None,
            "config_fingerprint": None,
            "source_archive_digest": None,
        }


@dataclass(slots=True)
class Report:
    checks: list[dict[str, Any]] = field(default_factory=list)
    implementation_validated: bool = False
    prior_target_gates_validated: bool = False
    expert_collection_smoke_validated: bool = False
    action_replay_validated: bool = False
    full_dataset_validated: bool = False

    def record(self, name: str, status: str, detail: str, *, required: bool = True) -> None:
        print(f"[{status.upper()}] {name}: {detail}")
        self.checks.append({"name": name, "status": status, "detail": detail, "required": required})

    def check(self, name: str, condition: bool, detail: str) -> None:
        self.record(name, "pass" if condition else "fail", detail)

    @property
    def failed(self) -> bool:
        return any(item["required"] and item["status"] == "fail" for item in self.checks)

    def new_checks_passed(self, start_index: int) -> bool:
        return not any(
            item["required"] and item["status"] == "fail" for item in self.checks[start_index:]
        )

    def accepted(self, mode: VerificationMode) -> bool:
        if self.failed or not self.implementation_validated:
            return False
        if mode == "structural":
            return True
        if not self.prior_target_gates_validated or not self.action_replay_validated:
            return False
        if mode == "target_smoke":
            return self.expert_collection_smoke_validated
        return self.full_dataset_validated

    def write(self, *, mode: VerificationMode, dataset_root: Path) -> None:
        target_mode = mode != "structural"
        physical_target_validated = target_mode and self.accepted(mode)
        source_identity = _validated_source_archive_identity(dataset_root)
        write_json_atomic(
            REPORT_PATH,
            {
                "schema_version": "langmani-m3a-verification-v3",
                "verification_mode": mode,
                "target_mode": target_mode,
                "validation_scope": mode,
                "dataset_root": str(dataset_root),
                "implementation_validated": self.implementation_validated,
                "prior_target_gates_validated": self.prior_target_gates_validated,
                "expert_collection_smoke_validated": self.expert_collection_smoke_validated,
                "action_replay_validated": self.action_replay_validated,
                "full_dataset_validated": self.full_dataset_validated,
                "physical_target_validated": physical_target_validated,
                # Kept as a compatibility alias; the six fields above are authoritative.
                "physical_acceptance": physical_target_validated,
                **source_identity,
                "checks": self.checks,
                "passed": self.accepted(mode),
            },
        )


def _is_native_linux() -> bool:
    release = platform.release().lower()
    return (
        platform.system() == "Linux" and "microsoft" not in release and not os.getenv("WSL_INTEROP")
    )


def _dataset_root(value: str) -> Path:
    try:
        return validated_dataset_root(value, output_root=OUTPUT_ROOT)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--target-smoke",
        action="store_true",
        help="Create and validate one fresh six-task scene group on the native target.",
    )
    target_group.add_argument(
        "--target-full",
        action="store_true",
        help="Resume or validate the authoritative 60-group native-target archive.",
    )
    parser.add_argument(
        "--dataset-root",
        type=_dataset_root,
        help=(
            "Archive root. Full mode defaults to the authoritative M3A root; smoke mode "
            "defaults to a fresh diagnostic root."
        ),
    )
    parser.add_argument(
        "--create-new-run",
        action="store_true",
        help="Authorize target-full to create a missing archive; never replaces an existing root.",
    )
    args = parser.parse_args()
    if args.create_new_run and not args.target_full:
        parser.error("--create-new-run requires --target-full")
    return args


def _verification_mode(args: argparse.Namespace) -> VerificationMode:
    if args.target_smoke:
        return "target_smoke"
    if args.target_full:
        return "target_full"
    return "structural"


def _fresh_smoke_dataset_root() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    candidate = SMOKE_RUNS_ROOT / f"smoke-{stamp}-{uuid4().hex[:8]}"
    return validated_dataset_root(candidate, output_root=OUTPUT_ROOT)


def _resolved_dataset_root(args: argparse.Namespace, mode: VerificationMode) -> Path:
    if args.dataset_root is not None:
        return validated_dataset_root(args.dataset_root, output_root=OUTPUT_ROOT)
    if mode == "target_smoke":
        return _fresh_smoke_dataset_root()
    return validated_dataset_root(DEFAULT_DATASET_ROOT, output_root=OUTPUT_ROOT)


def _invalidate_m3a_command_reports() -> None:
    for report_path in {
        path for name, path in COMMAND_REPORTS.items() if name != PRIOR_TARGET_GATE
    }:
        report_path.unlink(missing_ok=True)


def _run_command(report: Report, name: str, arguments: list[str]) -> bool:
    print(f"[INFO] {name}: {' '.join(arguments)}")
    report_path = COMMAND_REPORTS.get(name)
    if report_path is None:
        report.check(name, False, "verifier has no fresh-report contract for this command")
        return False
    report_path.unlink(missing_ok=True)
    completed = subprocess.run(arguments, cwd=PROJECT_ROOT, check=False)
    artifact_ok, artifact_detail = _validate_command_report(name, report_path, arguments)
    passed = completed.returncode == 0 and artifact_ok
    report.check(name, passed, f"exit code {completed.returncode}; {artifact_detail}")
    return passed


def _command_dataset_root(name: str, arguments: list[str]) -> Path:
    dataset_flag = (
        "--output-root" if name in (SMOKE_COLLECTION, FULL_COLLECTION) else "--dataset-root"
    )
    try:
        return Path(arguments[arguments.index(dataset_flag) + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise ValueError(f"command is missing {dataset_flag}") from error


def _validate_command_report(
    name: str,
    report_path: Path,
    arguments: list[str],
) -> tuple[bool, str]:
    if not report_path.is_file():
        return False, f"fresh report missing: {report_path}"
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return False, f"fresh report unreadable: {type(error).__name__}: {error}"
    if not isinstance(payload, dict) or payload.get("passed") is not True:
        return False, "fresh report does not declare passed=true"
    if name == PRIOR_TARGET_GATE:
        valid = payload.get("target_mode") is True and payload.get("physical_acceptance") is True
        return (
            valid,
            "fresh ordered M2 target report validated" if valid else "invalid M2 target report",
        )

    try:
        dataset_root = _command_dataset_root(name, arguments)
    except ValueError as error:
        return False, str(error)
    if payload.get("dataset_root") != str(dataset_root):
        return False, "fresh report dataset_root differs from command"

    replay_expectation = REPLAY_REPORT_EXPECTATIONS.get(name)
    if replay_expectation is not None:
        expected_group_count, expected_episode_count = replay_expectation
        results = payload.get("results")
        try:
            manifest = load_manifest(dataset_root)
            expected_records = manifest.raw_episodes
            validations: list[ReplayValidationResult] = []
            identities_match = isinstance(results, list) and len(results) == len(expected_records)
            if identities_match:
                for item, record in zip(results, expected_records, strict=True):
                    if not isinstance(item, dict) or (
                        item.get("raw_trajectory_id") != record.raw_trajectory_id
                        or item.get("source_shard_id") != record.source_shard_id
                        or item.get("native_episode_id") != record.native_episode_id
                        or not isinstance(item.get("validation"), dict)
                    ):
                        identities_match = False
                        break
                    validations.append(ReplayValidationResult.from_dict(item["validation"]))
            tolerance_match = identities_match and all(
                validation.passed
                and validation.mode is ReplayValidationMode.ACTION_AND_STATE_AUDIT
                and validation.state_audit_performed
                and validation.state_audit_passed is True
                and validation.final_position_error_m is not None
                and validation.final_position_error_m <= manifest.config.final_position_tolerance_m
                and validation.final_orientation_error_rad is not None
                and validation.final_orientation_error_rad
                <= manifest.config.final_orientation_tolerance_rad
                and validation.final_joint_error_rad is not None
                and validation.final_joint_error_rad <= manifest.config.final_joint_tolerance_rad
                for validation in validations
            )
        except (OSError, TypeError, ValueError, KeyError, RuntimeError) as error:
            return False, f"replay evidence is malformed: {type(error).__name__}: {error}"
        valid = (
            payload.get("schema_version") == "langmani-m3a-replay-report-v1"
            and payload.get("selected_episode_count") == expected_episode_count
            and payload.get("passed_episode_count") == expected_episode_count
            and payload.get("command_errors") == []
            and len(expected_records) == expected_episode_count
            and len(tuple(group for group in manifest.scene_groups if group.accepted))
            == expected_group_count
            and manifest.config.target_complete_scene_count == expected_group_count
            and manifest.config.replay_validation_mode
            is ReplayValidationMode.ACTION_AND_STATE_AUDIT
            and identities_match
            and len(validations) == expected_episode_count
            and tolerance_match
            and "--limit" not in arguments
            and "--raw-trajectory-id" not in arguments
        )
        detail = f"fresh {expected_episode_count}-episode replay report validated"
        return valid, detail if valid else "invalid replay report"

    archive_expectation = ARCHIVE_REPORT_EXPECTATIONS.get(name)
    if archive_expectation is None:
        return False, "verifier has no report-schema expectation for this command"
    expected_group_count, expected_episode_count = archive_expectation
    summary = payload.get("summary")
    valid = (
        isinstance(summary, dict)
        and summary.get("status") == "complete"
        and summary.get("target_complete_scene_count") == expected_group_count
        and summary.get("accepted_scene_group_count") == expected_group_count
        and summary.get("accepted_episode_count") == expected_episode_count
    )
    detail = f"fresh complete {expected_group_count}-group archive report validated"
    return valid, detail if valid else "invalid archive report"


def _structural_checks(report: Report) -> None:
    config = CollectionConfig(runtime_versions=query_planner_runtime_versions())
    schedule = build_collection_schedule(config)
    report.check(
        "authoritative M3A target size",
        config.target_complete_scene_count == 60
        and len(CANONICAL_TASK_SPECS) == 6
        and 60 * 6 == 360,
        "60 complete scene groups x 6 canonical TaskSpecs = 360 accepted episodes",
    )
    report.check(
        "bounded deterministic schedule",
        config.maximum_candidate_scene_count == 120
        and config.maximum_expert_attempts_per_task == 3
        and len(schedule.episodes) == 120 * 6
        and schedule.candidate_scene_seeds == tuple(range(120)),
        "120 ordered candidate seeds, six tasks per seed, three attempts per task",
    )
    report.check(
        "group-aligned source shards",
        config.shard_size == 60 and config.shard_size % 6 == 0,
        f"shard_size={config.shard_size}",
    )
    signature = inspect.signature(RecordEpisode)
    source = inspect.getsource(RecordEpisode)
    report.check(
        "ManiSkill RecordEpisode integration contract",
        "record_env_state" in signature.parameters
        and "save_on_reset" in signature.parameters
        and 'h5py.File(self.output_dir / f"{trajectory_name}.h5", "w")' in source,
        "installed recorder exposes env-state recording and overwrite-only native HDF5",
    )


def _is_expected_full_config(config: CollectionConfig, dataset_root: Path) -> bool:
    expected = CollectionConfig(
        raw_output_root=str(dataset_root),
        runtime_versions=query_planner_runtime_versions(),
    )
    actual_payload = config.to_dict()
    expected_payload = expected.to_dict()
    actual_root = Path(str(actual_payload.pop("raw_output_root"))).resolve()
    expected_root = Path(str(expected_payload.pop("raw_output_root"))).resolve()
    return (
        actual_root == expected_root == dataset_root.resolve()
        and actual_payload == expected_payload
    )


def _prepare_full_archive(
    report: Report,
    dataset_root: Path,
    *,
    create_new_run: bool,
) -> FullArchiveAction:
    manifest_path = dataset_root / "manifests" / "collection_manifest.json"
    if create_new_run:
        root_absent = not dataset_root.exists()
        report.check(
            "explicit new full-run root",
            root_absent,
            (
                f"new archive root is absent: {dataset_root}"
                if root_absent
                else "--create-new-run never replaces or reuses an existing dataset root"
            ),
        )
        return "collect" if root_absent else "fail"
    if not manifest_path.is_file():
        report.check(
            "existing full archive or explicit creation",
            False,
            "full archive is missing; pass --create-new-run to authorize a new collection run",
        )
        return "fail"
    try:
        manifest = load_manifest(dataset_root)
    except (OSError, TypeError, ValueError, KeyError, RuntimeError) as error:
        report.check(
            "existing full archive manifest",
            False,
            f"manifest cannot be reused: {type(error).__name__}: {error}",
        )
        return "fail"
    config_matches = _is_expected_full_config(manifest.config, dataset_root)
    report.check(
        "existing full archive identity",
        config_matches,
        (
            "schema, collection config, control mode, and expert fingerprint match"
            if config_matches
            else "existing archive differs from the authoritative full-run identity"
        ),
    )
    if not config_matches:
        return "fail"
    if manifest.status is CollectionStatus.COMPLETE:
        report.record(
            FULL_COLLECTION,
            "pass",
            "existing immutable complete archive selected; collector was not invoked",
        )
        return "reuse"
    if manifest.status is CollectionStatus.IN_PROGRESS:
        report.record(
            "legal full-run resume",
            "pass",
            "compatible in-progress manifest selected; accepted groups remain immutable",
        )
        return "collect"
    report.check(
        "legal full-run resume",
        False,
        f"collection status {manifest.status.value!r} cannot be resumed for acceptance",
    )
    return "fail"


def _collection_command(dataset_root: Path, *, smoke: bool) -> list[str]:
    return [
        resolve_planner_python(),
        "environment/collect_raw_demos.py",
        "--output-root",
        str(dataset_root),
        "--candidate-seed-start",
        "0",
        "--target-complete-scene-count",
        "1" if smoke else "60",
        "--maximum-candidate-scene-count",
        "20" if smoke else "120",
        "--maximum-expert-attempts-per-task",
        "3",
        "--shard-size",
        "6" if smoke else "60",
        "--sim-backend",
        "physx_cpu",
        "--replay-validation-mode",
        "action_and_state_audit",
    ]


def _inspection_command(dataset_root: Path) -> list[str]:
    return [
        sys.executable,
        "environment/inspect_raw_demos.py",
        "--dataset-root",
        str(dataset_root),
    ]


def _replay_command(dataset_root: Path) -> list[str]:
    return [
        resolve_planner_python(),
        "environment/replay_raw_demos.py",
        "--dataset-root",
        str(dataset_root),
        "--mode",
        "action_and_state_audit",
    ]


def _validate_target_archive(
    report: Report,
    dataset_root: Path,
    *,
    expected_group_count: int,
) -> bool:
    start_index = len(report.checks)
    expected_episode_count = expected_group_count * 6
    expected_task_count = expected_group_count
    summary = inspect_raw_dataset(dataset_root)
    manifest = load_manifest(dataset_root)
    accepted_groups = tuple(group for group in manifest.scene_groups if group.accepted)
    accepted_episodes = tuple(episode for episode in manifest.raw_episodes if episode.accepted)
    task_counts = tuple(sorted(summary.task_episode_counts.values()))
    partial_groups = tuple(
        group
        for group in accepted_groups
        if not group.complete or len(group.episodes) != 6 or len(group.scheduled_episode_ids) != 6
    )
    expert_failures = tuple(
        episode for episode in accepted_episodes if not episode.expert_result.success
    )
    replay_failures = tuple(
        episode for episode in accepted_episodes if not episode.replay_validation.passed
    )
    false_successes = tuple(
        episode
        for episode in accepted_episodes
        if episode.final_environment_evaluation.get("success") is not True
        or episode.final_environment_evaluation.get("fail") is not False
        or episode.final_environment_evaluation.get("target_in_wrong_bin", False)
        or episode.final_environment_evaluation.get("wrong_object_in_target_bin", False)
    )
    unclassified = tuple(
        attempt
        for attempt in manifest.attempts
        if (attempt.outcome.value != "accepted" and not attempt.failure_reasons)
        or (
            attempt.replay_validation is not None
            and not attempt.replay_validation.passed
            and not getattr(attempt.replay_validation, "failure_codes", ())
        )
    )

    report.check(
        f"{expected_episode_count} accepted replay-validated episodes",
        summary.status is CollectionStatus.COMPLETE
        and summary.accepted_scene_group_count == expected_group_count
        and summary.accepted_episode_count == expected_episode_count
        and len(accepted_groups) == expected_group_count
        and len(accepted_episodes) == expected_episode_count,
        f"status={summary.status.value}; groups={summary.accepted_scene_group_count}; "
        f"episodes={summary.accepted_episode_count}",
    )
    report.check(
        f"exactly {expected_task_count} episodes per TaskSpec",
        len(task_counts) == 6 and task_counts == (expected_task_count,) * 6,
        f"counts={dict(summary.task_episode_counts)}",
    )
    report.check(
        "zero partial accepted scene groups",
        not partial_groups,
        f"partial_accepted_groups={len(partial_groups)}",
    )
    report.check(
        "zero accepted expert failures",
        not expert_failures,
        f"accepted_expert_failures={len(expert_failures)}",
    )
    report.check(
        "zero accepted action-replay failures",
        not replay_failures,
        f"accepted_replay_failures={len(replay_failures)}",
    )
    report.check(
        "zero accepted wrong-object, wrong-bin, fail, or non-success outcomes",
        not false_successes,
        f"accepted_false_successes={len(false_successes)}",
    )
    report.check(
        "zero checksum or schema failures",
        True,
        "independent archive inspection completed without corruption or schema exceptions",
    )
    report.check(
        "six-task actual initial states identical within every accepted group",
        True,
        "independent archive inspection enforced recorded initial-scene equivalence",
    )
    report.check(
        "zero unclassified validation failures",
        not unclassified,
        f"unclassified_validation_failures={len(unclassified)}",
    )
    return report.new_checks_passed(start_index)


def _run_target_smoke(report: Report, dataset_root: Path) -> None:
    if dataset_root.exists():
        report.check(
            "fresh smoke archive root",
            False,
            "target smoke never overwrites or reuses an existing archive; choose another root",
        )
        return
    collection_ok = _run_command(
        report,
        SMOKE_COLLECTION,
        _collection_command(dataset_root, smoke=True),
    )
    inspection_ok = collection_ok and _run_command(
        report,
        SMOKE_INSPECTION,
        _inspection_command(dataset_root),
    )
    archive_ok = inspection_ok and _validate_target_archive(
        report,
        dataset_root,
        expected_group_count=1,
    )
    report.expert_collection_smoke_validated = collection_ok and inspection_ok and archive_ok
    replay_ok = archive_ok and _run_command(
        report,
        SMOKE_REPLAY,
        _replay_command(dataset_root),
    )
    report.action_replay_validated = replay_ok


def _run_target_full(
    report: Report,
    dataset_root: Path,
    *,
    create_new_run: bool,
) -> None:
    action = _prepare_full_archive(
        report,
        dataset_root,
        create_new_run=create_new_run,
    )
    if action == "fail":
        return
    collection_ok = action == "reuse" or _run_command(
        report,
        FULL_COLLECTION,
        _collection_command(dataset_root, smoke=False),
    )
    inspection_ok = collection_ok and _run_command(
        report,
        FULL_INSPECTION,
        _inspection_command(dataset_root),
    )
    archive_ok = inspection_ok and _validate_target_archive(
        report,
        dataset_root,
        expected_group_count=60,
    )
    report.full_dataset_validated = collection_ok and inspection_ok and archive_ok
    replay_ok = archive_ok and _run_command(
        report,
        FULL_REPLAY,
        _replay_command(dataset_root),
    )
    report.action_replay_validated = replay_ok


def main() -> int:
    REPORT_PATH.unlink(missing_ok=True)
    _invalidate_m3a_command_reports()
    args = parse_args()
    mode = _verification_mode(args)
    dataset_root = _resolved_dataset_root(args, mode)
    report = Report()
    try:
        native_linux = _is_native_linux()
        if mode != "structural":
            report.check(
                "native Linux target",
                native_linux,
                f"platform={platform.system()} release={platform.release()}",
            )

        structural_start = len(report.checks)
        _structural_checks(report)
        report.implementation_validated = report.new_checks_passed(structural_start)

        if mode != "structural" and native_linux and report.implementation_validated:
            report.prior_target_gates_validated = _run_command(
                report,
                PRIOR_TARGET_GATE,
                [sys.executable, "environment/verify_m2.py", "--target"],
            )
            if report.prior_target_gates_validated:
                if mode == "target_smoke":
                    _run_target_smoke(report, dataset_root)
                else:
                    _run_target_full(
                        report,
                        dataset_root,
                        create_new_run=args.create_new_run,
                    )
        elif mode != "structural":
            report.record(
                "M3A physical collection and replay",
                "skip",
                "not started because native-target or implementation prerequisites failed",
                required=False,
            )
        elif dataset_root.joinpath("manifests", "collection_manifest.json").is_file():
            summary = inspect_raw_dataset(dataset_root)
            report.record(
                "existing M3A archive inspection",
                "pass",
                f"status={summary.status.value}; episodes={summary.accepted_episode_count}",
                required=False,
            )
        else:
            report.record(
                "M3A physical collection and replay",
                "skip",
                "no dataset requested in non-target structural mode",
                required=False,
            )
    except Exception as error:  # noqa: BLE001 - verifier command boundary
        traceback.print_exc()
        report.record(
            "unexpected verifier exception",
            "fail",
            f"{type(error).__name__}: {str(error) or repr(error)}",
        )
    report.write(mode=mode, dataset_root=dataset_root)
    print(f"[INFO] report: {REPORT_PATH}")
    return 0 if report.accepted(mode) else 1


if __name__ == "__main__":
    raise SystemExit(main())
