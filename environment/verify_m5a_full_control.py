"""Verify M5A six-scene paired control-development contracts and target evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for value in (PROJECT_ROOT, SRC_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from langmani.language.controller_registry import load_controller_registry_metadata  # noqa: E402
from langmani.language.corpus import build_language_corpus  # noqa: E402
from langmani.language.full_control_development import FULL_REJECTION_CASES  # noqa: E402
from langmani.language.full_control_verifier import verify_full_control_evidence  # noqa: E402
from langmani.language.neuro_symbolic_dispatch import (  # noqa: E402
    bind_selected_router_to_controller_registry,
    validate_selected_neuro_symbolic_dispatch_source,
)
from langmani.language.stage_protocol import M5AStage  # noqa: E402
from langmani.policies.act_runtime import inspect_git_state  # noqa: E402
from scripts.run_language_control import (  # noqa: E402
    _validate_prior_three_scene_authority,
    load_authoritative_development_schedule,
    load_sealed_final_authority,
    prepare_development_inputs,
)

DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "outputs" / "datasets" / "m3b" / "langmani-pick-place-lerobot-v1"
)
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "models" / "act"
DEFAULT_RUNTIME_SELECTION = (
    PROJECT_ROOT / "outputs" / "diagnostics" / "m42" / "runtime_ablation" / "runtime_selection.json"
)
DEFAULT_REPORT = (
    PROJECT_ROOT / "outputs" / "diagnostics" / "m5a" / "full-control-independent-verification.json"
)


class FullControlVerificationCommandError(RuntimeError):
    """Raised when full target-verification inputs are incomplete or unsafe."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-evidence", action="store_true")
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--prior-stage-report", type=Path)
    parser.add_argument("--prior-stage-verification", type=Path)
    parser.add_argument("--control-schedule", type=Path)
    parser.add_argument("--neuro-symbolic-evidence-root", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--runtime-selection", type=Path, default=DEFAULT_RUNTIME_SELECTION)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def _write(path: Path, value: dict[str, object]) -> None:
    destination = Path(os.path.abspath(os.fspath(path.expanduser())))
    if any(
        item.is_symlink() or item.is_junction()
        for item in (destination, *destination.parents)
        if item.exists()
    ):
        raise FullControlVerificationCommandError("verification report traverses a link")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")
        )
        + "\n"
    ).encode("utf-8")
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _structural() -> dict[str, object]:
    required = {
        "conflicting_objects",
        "conflicting_bins",
        "unsupported_action",
        "unsupported_object",
        "unsupported_destination",
        "empty_or_noise",
        "unresolved_correction",
        "contradictory_negation",
        "unsupported_spatial_reference",
    }
    categories = {case.category for case in FULL_REJECTION_CASES}
    passed = len(FULL_REJECTION_CASES) == 9 and categories == required
    return {
        "schema_version": "langmani-m5a-full-control-structural-verification-v0",
        "passed": passed,
        "implementation_validated": passed,
        "fixed_rejection_set_validated": passed,
        "full_control_development_completed": False,
        "final_benchmark_authorized": False,
        "physical_target_validated": False,
        "language_final_accessed": False,
        "control_final_accessed": False,
        "test_split_accessed": False,
        "historical_fresh_accessed": False,
        "m42_final_accessed": False,
        "smolvla_go": False,
    }


def _target(args: argparse.Namespace) -> dict[str, object]:
    required = (
        "evidence_root",
        "prior_stage_report",
        "prior_stage_verification",
        "control_schedule",
        "neuro_symbolic_evidence_root",
    )
    for name in required:
        if getattr(args, name) is None:
            raise FullControlVerificationCommandError(f"--{name.replace('_', '-')} is required")
    git = inspect_git_state(PROJECT_ROOT)
    if not git.baseline_tracked or git.dirty:
        raise FullControlVerificationCommandError(
            "target verification requires a clean tracked Git worktree"
        )
    corpus = build_language_corpus()
    source = validate_selected_neuro_symbolic_dispatch_source(
        args.neuro_symbolic_evidence_root,
        corpus=corpus,
    )
    schedule = load_authoritative_development_schedule(args.control_schedule, corpus=corpus)
    inputs = prepare_development_inputs(
        schedule=schedule,
        corpus=corpus,
        stage=M5AStage.FULL_CONTROL_DEVELOPMENT,
    )
    registry, _locators = load_controller_registry_metadata(
        checkpoint_root=args.checkpoint_root,
        dataset_root=args.dataset_root,
        runtime_selection_path=args.runtime_selection,
    )
    bind_selected_router_to_controller_registry(source, registry)
    prior = _validate_prior_three_scene_authority(
        report_path=args.prior_stage_report,
        verification_path=args.prior_stage_verification,
        schedule=schedule,
        corpus=corpus,
        registry=registry,
        source=source,
    )
    sealed_final = load_sealed_final_authority(args.control_schedule, corpus=corpus)
    return verify_full_control_evidence(
        args.evidence_root,
        inputs=inputs,
        registry=registry,
        source=source,
        prior_three_scene_authority=prior,
        sealed_final_authority=sealed_final,
        expected_git_commit=git.commit,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = _target(args) if args.target_evidence else _structural()
    except Exception as error:  # noqa: BLE001 - preserve exact verification failure
        traceback.print_exc()
        report = {
            "schema_version": "langmani-m5a-full-control-verification-failure-v0",
            "passed": False,
            "error_type": type(error).__name__,
            "error_message": str(error) or repr(error),
            "full_control_development_completed": False,
            "final_benchmark_authorized": False,
            "physical_target_validated": False,
            "language_final_accessed": False,
            "control_final_accessed": False,
            "test_split_accessed": False,
            "historical_fresh_accessed": False,
            "m42_final_accessed": False,
            "smolvla_go": False,
        }
    _write(args.report, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
