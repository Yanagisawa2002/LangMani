"""Verify M5A.4.1 read-only dispatch integration or one real six-task smoke."""

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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from langmani.datasets.schedule import CANONICAL_TASK_SPECS  # noqa: E402
from langmani.language.controller_registry import (  # noqa: E402
    build_fixture_controller_registry,
    load_controller_registry_metadata,
)
from langmani.language.corpus import build_language_corpus  # noqa: E402
from langmani.language.dispatcher import (  # noqa: E402
    ControllerDispatcher,
    FixturePerTaskControllerLoader,
)
from langmani.language.neuro_symbolic_control_verifier import (  # noqa: E402
    verify_neuro_symbolic_control_evidence,
)
from langmani.language.neuro_symbolic_dispatch import (  # noqa: E402
    bind_selected_router_to_controller_registry,
    validate_selected_neuro_symbolic_dispatch_source,
)
from langmani.language.router_types import (  # noqa: E402
    RouterConfidence,
    RouterDecision,
    RouterRejectionReason,
    RouterStatus,
)
from langmani.language.stage_protocol import M5AStage  # noqa: E402
from langmani.policies.act_runtime import inspect_git_state  # noqa: E402
from scripts.run_language_control import (  # noqa: E402
    load_authoritative_development_schedule,
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
    PROJECT_ROOT / "outputs" / "diagnostics" / "m5a" / "neuro-symbolic-dispatch-verification.json"
)


class M5ADispatchVerificationCommandError(RuntimeError):
    """Raised when dispatch verification inputs are unsafe or incomplete."""


class _FixtureEnvironment:
    def __init__(self) -> None:
        self.reset_count = 0
        self.step_count = 0

    def reset(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        self.reset_count += 1
        return {}, {}

    def step(self, action):  # type: ignore[no-untyped-def]
        del action
        self.step_count += 1
        return {}, 0.0, True, False, {"success": True}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-evidence", action="store_true")
    parser.add_argument("--stage-report", type=Path)
    parser.add_argument("--control-schedule", type=Path)
    parser.add_argument("--neuro-symbolic-evidence-root", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--runtime-selection", type=Path, default=DEFAULT_RUNTIME_SELECTION)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def _read_object(path: Path, *, label: str) -> dict[str, object]:
    source = path.resolve(strict=True)
    if source.is_symlink() or source.is_junction() or not source.is_file():
        raise M5ADispatchVerificationCommandError(f"unsafe or missing {label}: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise M5ADispatchVerificationCommandError(f"{label} must be one JSON object")
    return value


def _write_report(path: Path, value: dict[str, object]) -> None:
    destination = Path(os.path.abspath(os.fspath(path.expanduser())))
    if any(
        candidate.is_symlink() or candidate.is_junction()
        for candidate in (destination, *destination.parents)
        if candidate.exists()
    ):
        raise M5ADispatchVerificationCommandError("verification report traverses a link")
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


def _structural_verification() -> dict[str, object]:
    registry = build_fixture_controller_registry()
    loader = FixturePerTaskControllerLoader()
    environment = _FixtureEnvironment()
    dispatcher = ControllerDispatcher(registry=registry, loader=loader)
    routed = []
    for index, task in enumerate(CANONICAL_TASK_SPECS):
        decision = RouterDecision.route(
            task_spec=task,
            confidence=RouterConfidence.unavailable(),
            router_name="NeuroSymbolicRouterV0",
            router_version="neuro-symbolic-router-v0",
        )
        result = dispatcher.dispatch(
            decision,
            environment=environment,
            evaluation_id=f"m5a41-dispatch-fixture-{index}",
            oracle_task_spec=task,
            scene_seed=10_000 + index,
        )
        routed.append(
            result.dispatch.dispatched
            and result.dispatch.controller_task_id == decision.task_id
            and result.dispatch.oracle_task_id == decision.task_id
        )
    loads_before = loader.load_count
    resets_before = environment.reset_count
    steps_before = environment.step_count
    rejection = RouterDecision.reject(
        status=RouterStatus.REJECT_UNSUPPORTED,
        rejection_reason=RouterRejectionReason.UNSUPPORTED_ACTION,
        confidence=RouterConfidence.unavailable(),
        router_name="NeuroSymbolicRouterV0",
        router_version="neuro-symbolic-router-v0",
    )
    rejected = dispatcher.dispatch(
        rejection,
        environment=environment,
        evaluation_id="m5a41-dispatch-fixture-rejection",
        oracle_task_spec=None,
        scene_seed=None,
    )
    zero_work = (
        not rejected.dispatch.dispatched
        and rejected.dispatch.safe_rejection
        and loader.load_count == loads_before
        and environment.reset_count == resets_before
        and environment.step_count == steps_before
    )
    passed = all(routed) and zero_work and loader.load_count == 6
    return {
        "schema_version": "langmani-m5a41-dispatch-structural-verification-v0",
        "passed": passed,
        "implementation_validated": passed,
        "read_only_dispatch_integration_validated": passed,
        "canonical_six_task_mapping_validated": all(routed),
        "zero_work_rejection_validated": zero_work,
        "controller_load_count": loader.load_count,
        "environment_reset_count": environment.reset_count,
        "environment_step_count": environment.step_count,
        "real_six_task_control_validated": False,
        "physical_target_validated": False,
        "language_final_accessed": False,
        "control_final_accessed": False,
        "test_split_accessed": False,
        "m42_final_accessed": False,
        "smolvla_go": False,
    }


def _target_verification(args: argparse.Namespace) -> dict[str, object]:
    for name in ("stage_report", "control_schedule", "neuro_symbolic_evidence_root"):
        if getattr(args, name) is None:
            raise M5ADispatchVerificationCommandError(f"--{name.replace('_', '-')} is required")
    git = inspect_git_state(PROJECT_ROOT)
    if not git.baseline_tracked or git.dirty:
        raise M5ADispatchVerificationCommandError(
            "target evidence verification requires a clean tracked Git worktree"
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
        stage=M5AStage.ONE_SCENE_CONTROL_SMOKE,
    )
    registry, _locators = load_controller_registry_metadata(
        checkpoint_root=args.checkpoint_root,
        dataset_root=args.dataset_root,
        runtime_selection_path=args.runtime_selection,
    )
    bind_selected_router_to_controller_registry(source, registry)
    report = _read_object(args.stage_report, label="one-scene control stage report")
    return {
        **verify_neuro_symbolic_control_evidence(
            report,
            inputs=inputs,
            registry=registry,
            source=source,
        ),
        "verifier_git_commit": git.commit,
        "stage_report": str(args.stage_report.resolve()),
        "selected_router_evidence_root": str(source.evidence_root),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = _target_verification(args) if args.target_evidence else _structural_verification()
    except Exception as error:  # noqa: BLE001 - preserve exact verification failure
        traceback.print_exc()
        report = {
            "schema_version": "langmani-m5a41-dispatch-verification-failure-v0",
            "passed": False,
            "error_type": type(error).__name__,
            "error_message": str(error) or repr(error),
            "physical_target_validated": False,
            "language_final_accessed": False,
            "control_final_accessed": False,
            "test_split_accessed": False,
            "m42_final_accessed": False,
            "smolvla_go": False,
        }
    _write_report(args.report, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
