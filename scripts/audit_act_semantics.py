"""Run the frozen-policy M4.3a semantic-alignment audit.

This command is deliberately a thin, scope-safe boundary.  Project-owned
runtime code performs checkpoint loading and inference; this module owns only
argument parsing, path safety, audit-scope separation, machine-readable
reporting, and a truthful non-zero process status.

M4.3a may inspect the M3B validation view and ``m42_dev_v0`` only.  It cannot
represent the M3B test split, the original M4 fresh-seed benchmark, or
``m42_final_v0``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import cast

from langmani.policies.act_runtime import atomic_write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMAND_SCHEMA = "langmani-m43-semantic-audit-command-v0"
AUDIT_MODES = ("validation", "m42_dev_v0", "combined")
EXPECTED_SOURCES: Mapping[str, tuple[str, ...]] = {
    "validation": ("m3b_validation",),
    "m42_dev_v0": ("m42_dev_v0",),
    "combined": ("m3b_validation", "m42_dev_v0"),
}

PROTECTED_SOURCE_ROOTS = (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "environment",
    PROJECT_ROOT / "tests",
    PROJECT_ROOT / "docs",
    PROJECT_ROOT / ".git",
)

_DIRECTORY_INPUT_FIELDS = (
    "dataset_root",
    "m4_checkpoint_root",
    "task_token_checkpoint_root",
    "m42_diagnostics_root",
)
_FILE_INPUT_FIELDS = ("runtime_selection",)
_FORBIDDEN_ACCESS_FLAGS = (
    "test_split_accessed",
    "fresh_seed_accessed",
    "fresh_seed_schedule_accessed",
    "final_schedule_accessed",
    "m42_final_schedule_accessed",
)
_FORBIDDEN_AUTHORIZATION_FLAGS = (
    "final_benchmark_authorized",
    "go_no_go_decision_completed",
    "smolvla_go",
    "factor_film_training_completed",
)
_REQUIRED_FALSE_RUNTIME_FLAGS = (
    "test_split_accessed",
    "fresh_seed_accessed",
    "final_schedule_accessed",
    *_FORBIDDEN_AUTHORIZATION_FLAGS,
)
_PORTABLE_SECTION_KEYS = frozenset(
    {
        "audit_identity",
        "evidence_identity",
        "fingerprint_inputs",
        "portable_manifest",
        "semantic_identity",
    }
)


@dataclass(frozen=True, slots=True)
class ResolvedAuditPaths:
    """Resolved, link-free paths passed to the project-owned audit runtime."""

    dataset_root: Path
    m4_checkpoint_root: Path
    task_token_checkpoint_root: Path
    m42_diagnostics_root: Path
    runtime_selection: Path
    output_root: Path
    report: Path


@dataclass(slots=True)
class _CommandProgress:
    """Facts about this process only; stale artifacts never set these fields."""

    validation_evidence_accessed: bool = False
    development_evidence_accessed: bool = False
    test_split_accessed: bool = False
    fresh_seed_accessed: bool = False
    final_schedule_accessed: bool = False
    physical_execution_started: bool = False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=AUDIT_MODES, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--m4-checkpoint-root", type=Path, required=True)
    parser.add_argument("--task-token-checkpoint-root", type=Path, required=True)
    parser.add_argument("--m42-diagnostics-root", type=Path, required=True)
    parser.add_argument("--runtime-selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Ephemeral command report; it must remain outside the immutable evidence root.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _command_progress(args: argparse.Namespace) -> _CommandProgress:
    value = getattr(args, "_m43_command_progress", None)
    if not isinstance(value, _CommandProgress):
        value = _CommandProgress()
        args._m43_command_progress = value
    return value


def _lexical_absolute(path: Path) -> Path:
    """Make a path absolute without following links."""

    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _resolved_unlinked(path: Path, *, label: str) -> Path:
    """Resolve only after rejecting every existing linked path component."""

    lexical = _lexical_absolute(path)
    for component in (lexical, *lexical.parents):
        if component.is_symlink() or component.is_junction():
            raise RuntimeError(f"{label} traverses a symlink or junction: {component}")
    return lexical.resolve(strict=False)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _overlaps(left: Path, right: Path) -> bool:
    return _is_within(left, right) or _is_within(right, left)


def _normalized_identity(value: str) -> str:
    return value.strip().casefold().replace("-", "_")


def _is_forbidden_identity(value: str) -> bool:
    normalized = _normalized_identity(value)
    return (
        normalized == "test"
        or "fresh_seed" in normalized
        or normalized == "m42_final_v0"
        or normalized.startswith("m42_final_")
    )


def _reject_forbidden_path_identity(
    path: Path, *, label: str, include_parent: bool = False
) -> None:
    # Evidence roots may legitimately live below deployment directories named
    # ``test``.  Scope the lexical identity check to the declared endpoint;
    # file endpoints additionally bind their immediate evidence namespace.
    inspected = path.parts[-2:] if include_parent else path.parts[-1:]
    for component in inspected:
        if _is_forbidden_identity(component):
            raise RuntimeError(f"{label} names a prohibited test, fresh-seed, or final source")


def _input_paths(args: argparse.Namespace) -> tuple[Path, ...]:
    return tuple(
        _resolved_unlinked(cast(Path, getattr(args, field)), label=f"M4.3a {field}")
        for field in (*_DIRECTORY_INPUT_FIELDS, *_FILE_INPUT_FIELDS)
    )


def _safe_report_path(args: argparse.Namespace) -> Path:
    report = _resolved_unlinked(args.report, label="M4.3a command report")
    output = _resolved_unlinked(args.output_root, label="M4.3a evidence output")
    _reject_forbidden_path_identity(report, label="report", include_parent=True)
    _reject_forbidden_path_identity(output, label="output_root", include_parent=True)
    inputs = _input_paths(args)
    protected = tuple(
        _resolved_unlinked(path, label="protected source path") for path in PROTECTED_SOURCE_ROOTS
    )
    if _overlaps(report, output):
        raise RuntimeError("command report must remain outside the immutable evidence output")
    if any(_overlaps(report, path) for path in (*inputs, *protected)):
        raise RuntimeError("command report overlaps protected input, source, or Git content")
    if report.exists() and not report.is_file():
        raise RuntimeError("command report path must be a real file")
    return report


def validate_command(args: argparse.Namespace) -> ResolvedAuditPaths:
    """Validate paths without creating or mutating any artifact."""

    if args.mode not in EXPECTED_SOURCES:
        raise ValueError(f"unsupported M4.3a audit mode: {args.mode!r}")
    directories = {
        field: _resolved_unlinked(cast(Path, getattr(args, field)), label=f"M4.3a {field}")
        for field in _DIRECTORY_INPUT_FIELDS
    }
    files = {
        field: _resolved_unlinked(cast(Path, getattr(args, field)), label=f"M4.3a {field}")
        for field in _FILE_INPUT_FIELDS
    }
    for field, path in (*directories.items(), *files.items()):
        _reject_forbidden_path_identity(
            path, label=field, include_parent=field in _FILE_INPUT_FIELDS
        )
    if not args.dry_run:
        for field, path in directories.items():
            if not path.is_dir():
                raise RuntimeError(f"{field} must be an existing real directory: {path}")
        for field, path in files.items():
            if not path.is_file():
                raise RuntimeError(f"{field} must be an existing real file: {path}")
    else:
        for field, path in directories.items():
            if path.exists() and not path.is_dir():
                raise RuntimeError(f"{field} must be a directory when present: {path}")
        for field, path in files.items():
            if path.exists() and not path.is_file():
                raise RuntimeError(f"{field} must be a file when present: {path}")

    output = _resolved_unlinked(args.output_root, label="M4.3a evidence output")
    _reject_forbidden_path_identity(output, label="output_root", include_parent=True)
    protected = tuple(
        _resolved_unlinked(path, label="protected source path") for path in PROTECTED_SOURCE_ROOTS
    )
    if any(
        _overlaps(output, path) for path in (*directories.values(), *files.values(), *protected)
    ):
        raise RuntimeError(
            "M4.3a evidence output must not overlap immutable input, source, or Git content"
        )
    if output.exists() and not output.is_dir():
        raise RuntimeError("M4.3a evidence output must be a real directory when present")
    report = _safe_report_path(args)
    return ResolvedAuditPaths(
        **directories,
        **files,
        output_root=output,
        report=report,
    )


def _validate_paths(args: argparse.Namespace) -> ResolvedAuditPaths:
    """Repository-style private alias retained for verifier contract tests."""

    return validate_command(args)


def _portable_json_value(value: object, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise RuntimeError(f"{label} requires string JSON keys")
            _portable_json_value(item, label=f"{label}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, item in enumerate(value):
            _portable_json_value(item, label=f"{label}[{index}]")
        return
    if isinstance(value, str) and (
        PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()
    ):
        raise RuntimeError(f"{label} contains a machine-specific absolute path")


def _reject_forbidden_runtime_identity(value: object, *, parent_key: str = "") -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise RuntimeError("semantic audit report requires string JSON keys")
            key = _normalized_identity(raw_key)
            if key in _FORBIDDEN_ACCESS_FLAGS and item is not False:
                raise RuntimeError(f"semantic audit must keep {raw_key}=false")
            if key in _FORBIDDEN_AUTHORIZATION_FLAGS and item is not False:
                raise RuntimeError(f"semantic audit must keep {raw_key}=false")
            if key not in (*_FORBIDDEN_ACCESS_FLAGS, *_FORBIDDEN_AUTHORIZATION_FLAGS) and (
                _is_forbidden_identity(raw_key)
            ):
                raise RuntimeError("semantic audit report contains a prohibited evidence section")
            if key in _PORTABLE_SECTION_KEYS:
                _portable_json_value(item, label=raw_key)
            _reject_forbidden_runtime_identity(item, parent_key=key)
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for item in value:
            _reject_forbidden_runtime_identity(item, parent_key=parent_key)
        return
    identity_key = any(
        token in parent_key
        for token in ("split", "schedule", "source", "identity", "namespace", "view")
    )
    if isinstance(value, str) and identity_key and _is_forbidden_identity(value):
        raise RuntimeError(
            "semantic audit report references a prohibited test, fresh-seed, or final identity"
        )


def _json_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("semantic audit runtime must return a mapping")
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"semantic audit runtime report is not finite JSON: {error}") from error
    if not isinstance(decoded, dict):
        raise RuntimeError("semantic audit runtime report must encode one JSON object")
    return cast(dict[str, object], decoded)


def _validate_runtime_result(
    value: object,
    *,
    mode: str,
    dry_run: bool,
) -> dict[str, object]:
    result = _json_mapping(value)
    _reject_forbidden_runtime_identity(result)
    if not isinstance(result.get("passed"), bool):
        raise RuntimeError("semantic audit runtime report requires a boolean passed field")
    if result["passed"] is not True:
        return result

    missing_false_flags = [
        name for name in _REQUIRED_FALSE_RUNTIME_FLAGS if result.get(name) is not False
    ]
    if missing_false_flags:
        raise RuntimeError(
            "successful semantic audit runtime must explicitly report false for: "
            + ", ".join(missing_false_flags)
        )

    expected = list(EXPECTED_SOURCES[mode])
    if result.get("requested_sources") != expected:
        raise RuntimeError("semantic audit runtime changed or silently mixed the requested sources")
    if result.get("task_token_m42_status") != "rejected":
        raise RuntimeError("semantic audit must preserve TaskToken as rejected by M4.2")
    if dry_run:
        if result.get("semantic_audit_completed") is not False:
            raise RuntimeError("dry-run cannot claim a completed semantic audit")
        if result.get("physical_execution") is not False:
            raise RuntimeError("dry-run cannot claim physical execution")
    else:
        if result.get("audited_sources") != expected:
            raise RuntimeError("completed semantic audit did not preserve source separation")
        if result.get("semantic_audit_completed") is not True:
            raise RuntimeError("successful runtime did not complete its requested semantic audit")
    return result


def _load_runtime_entrypoint(*, dry_run: bool) -> Callable[[argparse.Namespace], object]:
    """Delay the heavyweight checkpoint/runtime import until after preflight."""

    module = importlib.import_module("langmani.policies.m43_audit_runtime")
    name = "plan_semantic_audit" if dry_run else "run_semantic_audit"
    entrypoint = getattr(module, name, None)
    if not callable(entrypoint):
        raise RuntimeError(f"M4.3a runtime does not provide {name}()")
    return cast(Callable[[argparse.Namespace], object], entrypoint)


def _runtime_namespace(args: argparse.Namespace, paths: ResolvedAuditPaths) -> argparse.Namespace:
    result = argparse.Namespace(**vars(args))
    for field in ResolvedAuditPaths.__dataclass_fields__:
        setattr(result, field, getattr(paths, field))
    return result


def execute(args: argparse.Namespace) -> dict[str, object]:
    paths = _validate_paths(args)
    progress = _command_progress(args)
    runtime_args = _runtime_namespace(args, paths)
    runtime_args._m43_command_progress = progress
    runtime = _load_runtime_entrypoint(dry_run=bool(args.dry_run))
    result = _validate_runtime_result(
        runtime(runtime_args), mode=args.mode, dry_run=bool(args.dry_run)
    )
    return {
        **result,
        "schema_version": COMMAND_SCHEMA,
        "mode": args.mode,
        "device": args.device,
        "dry_run": bool(args.dry_run),
    }


def _failure_report(
    args: argparse.Namespace,
    progress: _CommandProgress,
    error: BaseException,
) -> dict[str, object]:
    mode = args.mode if args.mode in EXPECTED_SOURCES else "invalid"
    return {
        "schema_version": COMMAND_SCHEMA,
        "mode": mode,
        "device": args.device,
        "requested_sources": list(EXPECTED_SOURCES.get(mode, ())),
        "dry_run": bool(args.dry_run),
        "passed": False,
        "error_type": type(error).__name__,
        "error_message": str(error) or repr(error),
        "semantic_audit_implementation_validated": False,
        "semantic_audit_completed": False,
        "factor_film_training_completed": False,
        "test_split_accessed": progress.test_split_accessed,
        "fresh_seed_accessed": progress.fresh_seed_accessed,
        "final_schedule_accessed": progress.final_schedule_accessed,
        "final_benchmark_authorized": False,
        "go_no_go_decision_completed": False,
        "smolvla_go": False,
        "physical_execution": progress.physical_execution_started,
    }


def _write_report(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, value)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    progress = _CommandProgress()
    args._m43_command_progress = progress
    try:
        report = execute(args)
    except Exception as error:  # noqa: BLE001 - outer CLI preserves exact diagnostics
        traceback.print_exc()
        report = _failure_report(args, progress, error)
    report_written = False
    report_path: Path | None = None
    try:
        report_path = _safe_report_path(args)
        _write_report(report_path, report)
        report_written = True
    except Exception:  # noqa: BLE001 - machine stdout still reports an unsafe report path
        traceback.print_exc()
    stdout = {
        **report,
        "report_written": report_written,
        "report": str(report_path) if report_written and report_path is not None else None,
    }
    print(json.dumps(stdout, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if report.get("passed") is True and report_written else 1


if __name__ == "__main__":
    raise SystemExit(main())
