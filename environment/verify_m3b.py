"""Verify M3B contracts, one-group target export, or the full derived dataset."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from langmani.collection.command_support import validated_dataset_root, write_json_atomic
from langmani.datasets.lerobot_types import (
    ExportMode,
    FeatureContract,
    LeRobotExportConfig,
    SplitConfig,
    stable_export_fingerprint,
)
from langmani.datasets.lerobot_writer import COMPLETION_MARKER, inspect_lerobot_runtime
from langmani.datasets.splits import build_split_assignments

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
OUTPUT_DIR = OUTPUT_ROOT / "diagnostics" / "m3b"
REPORT_PATH = OUTPUT_DIR / "verification.json"
M3A_REPORT = OUTPUT_ROOT / "diagnostics" / "m3a" / "verification.json"
EXPORT_REPORT = OUTPUT_DIR / "export_command.json"
VALIDATION_REPORT = OUTPUT_DIR / "validation_command.json"
DEFAULT_SOURCE_ROOT = OUTPUT_ROOT / "datasets" / "m3a" / "langmani-pick-place-raw-v1"
DEFAULT_DATASET_ROOT = OUTPUT_ROOT / "datasets" / "m3b" / "langmani-pick-place-lerobot-v1"
SMOKE_OUTPUT_PARENT = OUTPUT_ROOT / "datasets" / "m3b" / "target-smoke-runs"

VerificationMode = Literal["structural", "target_smoke", "target_full"]


@dataclass(slots=True)
class Report:
    checks: list[dict[str, Any]] = field(default_factory=list)
    implementation_validated: bool = False
    source_archive_validated: bool = False
    smoke_export_validated: bool = False
    full_export_validated: bool = False
    lerobot_load_validated: bool = False
    parquet_validated: bool = False
    video_decode_validated: bool = False
    source_alignment_validated: bool = False
    split_integrity_validated: bool = False
    privileged_leakage_validated: bool = False
    physical_target_validated: bool = False
    fixture_export_validated: bool = False

    def check(self, name: str, condition: bool, detail: str) -> None:
        status = "pass" if condition else "fail"
        print(f"[{status.upper()}] {name}: {detail}")
        self.checks.append({"name": name, "status": status, "detail": detail})

    @property
    def failed(self) -> bool:
        return any(item["status"] == "fail" for item in self.checks)

    def accepted(self, mode: VerificationMode) -> bool:
        if self.failed or not self.implementation_validated:
            return False
        if mode == "structural":
            return True
        export_ok = (
            self.smoke_export_validated if mode == "target_smoke" else self.full_export_validated
        )
        return all(
            (
                self.source_archive_validated,
                export_ok,
                self.lerobot_load_validated,
                self.parquet_validated,
                self.video_decode_validated,
                self.source_alignment_validated,
                self.split_integrity_validated,
                self.privileged_leakage_validated,
                self.physical_target_validated,
            )
        )

    def write(self, *, mode: VerificationMode, source_root: Path, dataset_root: Path) -> None:
        write_json_atomic(
            REPORT_PATH,
            {
                "schema_version": "langmani-m3b-verification-v1",
                "verification_mode": mode,
                "source_root": str(source_root),
                "dataset_root": str(dataset_root),
                "implementation_validated": self.implementation_validated,
                "source_archive_validated": self.source_archive_validated,
                "smoke_export_validated": self.smoke_export_validated,
                "full_export_validated": self.full_export_validated,
                "lerobot_load_validated": self.lerobot_load_validated,
                "parquet_validated": self.parquet_validated,
                "video_decode_validated": self.video_decode_validated,
                "source_alignment_validated": self.source_alignment_validated,
                "split_integrity_validated": self.split_integrity_validated,
                "privileged_leakage_validated": self.privileged_leakage_validated,
                "physical_target_validated": self.physical_target_validated,
                "fixture_export_validated": self.fixture_export_validated,
                "checks": self.checks,
                "passed": self.accepted(mode),
            },
        )


def _dataset_root(value: str) -> Path:
    try:
        return validated_dataset_root(value, output_root=OUTPUT_ROOT)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--target-smoke", action="store_true")
    mode.add_argument("--target-full", action="store_true")
    parser.add_argument("--source-root", type=_dataset_root, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--dataset-root", type=_dataset_root, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--create-new-source", action="store_true")
    parser.add_argument("--clean-staging", action="store_true")
    args = parser.parse_args()
    if args.create_new_source and not args.target_full:
        parser.error("--create-new-source requires --target-full")
    return args


def _mode(args: argparse.Namespace) -> VerificationMode:
    if args.target_smoke:
        return "target_smoke"
    if args.target_full:
        return "target_full"
    return "structural"


def _native_linux() -> bool:
    return (
        platform.system() == "Linux"
        and "microsoft" not in platform.release().lower()
        and not os.getenv("WSL_INTEROP")
    )


def _run(report: Report, name: str, arguments: list[str]) -> bool:
    print("[INFO]", name, "::", " ".join(arguments))
    completed = subprocess.run(arguments, cwd=PROJECT_ROOT, check=False)
    passed = completed.returncode == 0
    report.check(name, passed, f"exit code {completed.returncode}")
    return passed


def _read_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read fresh report {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"report root must be a JSON object: {path}")
    return payload


def _structural_checks(report: Report) -> None:
    start = len(report.checks)
    runtime = inspect_lerobot_runtime()
    report.check(
        "LeRobot 0.6.0 public dataset API", runtime.lerobot == "0.6.0", str(runtime.to_dict())
    )
    feature = FeatureContract()
    report.check(
        "three-feature policy allowlist",
        feature.policy_feature_keys
        == {
            "observation.images.base_camera",
            "observation.state",
            "action",
        },
        repr(sorted(feature.policy_feature_keys)),
    )
    config = LeRobotExportConfig(
        source_root="C:/machine-a/raw",
        output_root="C:/machine-a/derived",
        repo_id="langmani/structural",
        expected_source_collection_run_id="collection-run-structural",
        expected_source_run_fingerprint="sha256:" + "1" * 64,
        expected_source_archive_digest="sha256:" + "2" * 64,
        mode=ExportMode.FULL,
        split_config=SplitConfig.full(split_seed=17),
        pyav_version=runtime.av,
        libavcodec_version=dict(runtime.av_libraries)["libavcodec"],
    )
    episode_ids = tuple(f"episode-{index}" for index in range(360))
    first_fingerprint = stable_export_fingerprint(config, episode_ids)
    moved = LeRobotExportConfig.from_dict(
        {
            **config.to_dict(),
            "source_root": "/different-machine/raw",
            "output_root": "/different-machine/derived",
        }
    )
    report.check(
        "path-independent stable export fingerprint",
        first_fingerprint == stable_export_fingerprint(moved, episode_ids),
        first_fingerprint,
    )
    assignments = build_split_assignments(
        tuple(f"group-{index}" for index in range(60)),
        config.split_config,
    )
    counts = {
        split: sum(item.split.value == split for item in assignments)
        for split in ("train", "validation", "test")
    }
    report.check(
        "deterministic 48/6/6 scene split",
        counts == {"train": 48, "validation": 6, "test": 6},
        repr(counts),
    )

    test_files = [
        "tests/unit/test_lerobot_contract.py",
        "tests/unit/test_dataset_splits.py",
        "tests/unit/test_lerobot_source.py",
        "tests/unit/test_policy_state.py",
        "tests/unit/test_observation_reconstruction.py",
        "tests/unit/test_frame_alignment.py",
        "tests/unit/test_lerobot_writer.py",
        "tests/integration/test_lerobot_real_writer.py",
    ]
    validator_test = PROJECT_ROOT / "tests" / "unit" / "test_lerobot_validation.py"
    if validator_test.is_file():
        test_files.append(str(validator_test.relative_to(PROJECT_ROOT)))
    fixture_ok = _run(
        report,
        "M3B contract and real LeRobot fixture suite",
        [sys.executable, "-m", "pytest", "-q", *test_files],
    )
    report.fixture_export_validated = fixture_ok
    report.implementation_validated = not any(
        item["status"] == "fail" for item in report.checks[start:]
    )


def _fresh_smoke_output() -> Path:
    candidate = SMOKE_OUTPUT_PARENT / f"run-{uuid4().hex}"
    return validated_dataset_root(candidate, output_root=OUTPUT_ROOT)


def _run_m3a_gate(
    report: Report, mode: VerificationMode, source_root: Path, create: bool
) -> Path | None:
    M3A_REPORT.unlink(missing_ok=True)
    command = [sys.executable, "environment/verify_m3a.py"]
    if mode == "target_smoke":
        command.append("--target-smoke")
    else:
        command.extend(["--target-full", "--dataset-root", str(source_root)])
        if create:
            command.append("--create-new-run")
    if not _run(report, "ordered M0/M1/M2 plus M3A target gate", command):
        return None
    payload = _read_report(M3A_REPORT)
    expected_mode = "target_smoke" if mode == "target_smoke" else "target_full"
    valid = all(
        (
            payload.get("schema_version") == "langmani-m3a-verification-v3",
            payload.get("verification_mode") == expected_mode,
            payload.get("passed") is True,
            payload.get("prior_target_gates_validated") is True,
            payload.get("action_replay_validated") is True,
            payload.get("physical_target_validated") is True,
            payload.get("source_archive_identity_status") == "validated",
        )
    )
    valid = valid and (
        payload.get("expert_collection_smoke_validated") is True
        if mode == "target_smoke"
        else payload.get("full_dataset_validated") is True
    )
    report.check("content-bound M3A target evidence", valid, f"mode={expected_mode}")
    if not valid:
        return None
    return Path(str(payload["dataset_root"])).resolve()


def _apply_validation_flags(report: Report, payload: dict[str, Any]) -> None:
    validation = payload.get("validation")
    if not isinstance(validation, dict):
        report.check("M3B validation report contract", False, "validation object missing")
        return
    report.lerobot_load_validated = validation.get("dataset_load_validated") is True
    report.parquet_validated = validation.get("parquet_validated") is True
    report.video_decode_validated = validation.get("video_decode_validated") is True
    report.source_alignment_validated = validation.get("source_alignment_validated") is True
    report.split_integrity_validated = validation.get("split_integrity_validated") is True
    report.privileged_leakage_validated = validation.get("privileged_leakage_validated") is True
    report.check(
        "independent M3B validation report", payload.get("passed") is True, "full report consumed"
    )


def _run_target(
    report: Report, mode: VerificationMode, args: argparse.Namespace
) -> tuple[Path, Path]:
    requested_source = args.source_root.resolve()
    source_root = _run_m3a_gate(
        report,
        mode,
        requested_source,
        create=bool(args.create_new_source),
    )
    dataset_root = _fresh_smoke_output() if mode == "target_smoke" else args.dataset_root.resolve()
    if source_root is None:
        return requested_source, dataset_root
    report.source_archive_validated = True

    expected_export_mode = "--smoke" if mode == "target_smoke" else "--full"
    completion_exists = (dataset_root / COMPLETION_MARKER).is_file()
    if not completion_exists:
        EXPORT_REPORT.unlink(missing_ok=True)
        command = [
            sys.executable,
            "scripts/export_lerobot_dataset.py",
            "--source-root",
            str(source_root),
            "--output-root",
            str(dataset_root),
            expected_export_mode,
            "--source-verification-report",
            str(M3A_REPORT),
        ]
        if args.clean_staging:
            command.append("--clean-staging")
        export_ok = _run(report, "staged M3B export", command)
        if not export_ok:
            return source_root, dataset_root
        export_payload = _read_report(EXPORT_REPORT)
        export_ok = export_payload.get("passed") is True
    else:
        export_ok = True
        report.check("reuse immutable completed M3B dataset", True, str(dataset_root))
    VALIDATION_REPORT.unlink(missing_ok=True)
    validation_ok = _run(
        report,
        "independent finalized M3B validation",
        [
            sys.executable,
            "scripts/validate_lerobot_dataset.py",
            "--dataset-root",
            str(dataset_root),
            "--source-root",
            str(source_root),
            expected_export_mode,
            "--source-verification-report",
            str(M3A_REPORT),
        ],
    )
    if validation_ok:
        _apply_validation_flags(report, _read_report(VALIDATION_REPORT))
    completed_export_validated = export_ok and validation_ok
    if mode == "target_smoke":
        report.smoke_export_validated = completed_export_validated
    else:
        report.full_export_validated = completed_export_validated
    report.physical_target_validated = validation_ok and all(
        (
            report.source_archive_validated,
            report.lerobot_load_validated,
            report.parquet_validated,
            report.video_decode_validated,
            report.source_alignment_validated,
            report.split_integrity_validated,
            report.privileged_leakage_validated,
        )
    )
    return source_root, dataset_root


def main() -> int:
    REPORT_PATH.unlink(missing_ok=True)
    args = parse_args()
    mode = _mode(args)
    report = Report()
    source_root = args.source_root.resolve()
    dataset_root = args.dataset_root.resolve()
    try:
        _structural_checks(report)
        if mode != "structural":
            if not _native_linux():
                report.check("native Linux target runtime", False, platform.platform())
            elif report.implementation_validated:
                source_root, dataset_root = _run_target(report, mode, args)
    except Exception as error:  # noqa: BLE001 - verifier boundary records unexpected failures
        traceback.print_exc()
        report.check(
            "unexpected verifier exception",
            False,
            f"{type(error).__name__}: {str(error) or repr(error)}",
        )
    report.write(mode=mode, source_root=source_root, dataset_root=dataset_root)
    print(f"[INFO] report: {REPORT_PATH}")
    return 0 if report.accepted(mode) else 1


if __name__ == "__main__":
    raise SystemExit(main())
