from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_cli() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "audit_act_semantics.py"
    spec = importlib.util.spec_from_file_location("langmani_test_audit_act_semantics", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cli = _load_cli()


def _argv(tmp_path: Path, *, mode: str = "combined", dry_run: bool = True) -> list[str]:
    values = [
        "--mode",
        mode,
        "--device",
        "cpu",
        "--dataset-root",
        str(tmp_path / "dataset"),
        "--m4-checkpoint-root",
        str(tmp_path / "m4-models"),
        "--task-token-checkpoint-root",
        str(tmp_path / "task-token"),
        "--m42-diagnostics-root",
        str(tmp_path / "m42-diagnostics"),
        "--runtime-selection",
        str(tmp_path / "runtime-selection.json"),
        "--output-root",
        str(tmp_path / "semantic-evidence"),
        "--report",
        str(tmp_path / "semantic-command.json"),
    ]
    if dry_run:
        values.append("--dry-run")
    return values


def _runtime_result(*, mode: str, dry_run: bool) -> dict[str, object]:
    sources = list(cli.EXPECTED_SOURCES[mode])
    return {
        "passed": True,
        "requested_sources": sources,
        "audited_sources": [] if dry_run else sources,
        "task_token_m42_status": "rejected",
        "semantic_audit_implementation_validated": True,
        "semantic_audit_completed": not dry_run,
        "physical_execution": False,
        "test_split_accessed": False,
        "fresh_seed_accessed": False,
        "final_schedule_accessed": False,
        "final_benchmark_authorized": False,
        "go_no_go_decision_completed": False,
        "factor_film_training_completed": False,
        "smolvla_go": False,
    }


def _materialize_inputs(args: object) -> None:
    for field in cli._DIRECTORY_INPUT_FIELDS:
        getattr(args, field).mkdir(parents=True)
    args.runtime_selection.write_text("{}\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("mode", "sources"),
    (
        ("validation", ["m3b_validation"]),
        ("m42_dev_v0", ["m42_dev_v0"]),
        ("combined", ["m3b_validation", "m42_dev_v0"]),
    ),
)
def test_dry_run_dispatches_only_the_planner_and_preserves_source_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    sources: list[str],
) -> None:
    args = cli.parse_args(_argv(tmp_path, mode=mode))
    calls: list[object] = []

    def plan(runtime_args: object) -> dict[str, object]:
        calls.append(runtime_args)
        assert runtime_args.dataset_root.is_absolute()
        assert runtime_args.output_root.is_absolute()
        return _runtime_result(mode=mode, dry_run=True)

    monkeypatch.setattr(
        cli,
        "_load_runtime_entrypoint",
        lambda *, dry_run: (
            plan if dry_run else pytest.fail("dry-run dispatched physical audit runtime")
        ),
    )
    result = cli.execute(args)
    assert len(calls) == 1
    assert result["requested_sources"] == sources
    assert result["semantic_audit_completed"] is False
    assert result["physical_execution"] is False
    assert result["final_schedule_accessed"] is False
    assert result["factor_film_training_completed"] is False
    assert not args.output_root.exists()


def test_completed_command_dispatches_runner_and_requires_exact_completed_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = cli.parse_args(_argv(tmp_path, mode="validation", dry_run=False))
    _materialize_inputs(args)
    monkeypatch.setattr(
        cli,
        "_load_runtime_entrypoint",
        lambda *, dry_run: (
            pytest.fail("real command dispatched planner")
            if dry_run
            else lambda _args: _runtime_result(mode="validation", dry_run=False)
        ),
    )
    result = cli.execute(args)
    assert result["passed"] is True
    assert result["audited_sources"] == ["m3b_validation"]
    assert result["semantic_audit_completed"] is True
    assert result["task_token_m42_status"] == "rejected"


def test_all_roots_are_explicit_and_mode_cannot_name_a_forbidden_schedule(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--mode", "combined"])
    with pytest.raises(SystemExit):
        cli.parse_args(_argv(tmp_path, mode="m42_final_v0"))
    with pytest.raises(SystemExit):
        cli.parse_args(_argv(tmp_path, mode="test"))


def test_non_dry_run_requires_every_explicit_input(tmp_path: Path) -> None:
    args = cli.parse_args(_argv(tmp_path, dry_run=False))
    with pytest.raises(RuntimeError, match="must be an existing real"):
        cli.validate_command(args)


@pytest.mark.parametrize(
    "field",
    (
        "dataset_root",
        "m4_checkpoint_root",
        "task_token_checkpoint_root",
        "m42_diagnostics_root",
        "runtime_selection",
    ),
)
@pytest.mark.parametrize("relation", ("equal", "child", "parent"))
def test_output_rejects_bidirectional_overlap_with_every_input(
    tmp_path: Path, field: str, relation: str
) -> None:
    args = cli.parse_args(_argv(tmp_path))
    protected = getattr(args, field)
    if relation == "equal":
        args.output_root = protected
    elif relation == "child":
        args.output_root = protected / "evidence"
    else:
        container = tmp_path / f"contains-{field}"
        setattr(args, field, container / "input")
        args.output_root = container
    with pytest.raises(RuntimeError, match="must not overlap"):
        cli.validate_command(args)


def test_output_and_report_reject_source_git_and_evidence_overlap(tmp_path: Path) -> None:
    args = cli.parse_args(_argv(tmp_path))
    args.output_root = cli.PROJECT_ROOT / "src" / "unsafe-evidence"
    with pytest.raises(RuntimeError, match="must not overlap"):
        cli.validate_command(args)

    args = cli.parse_args(_argv(tmp_path))
    args.report = args.output_root / "command.json"
    with pytest.raises(RuntimeError, match="outside the immutable evidence output"):
        cli.validate_command(args)

    args = cli.parse_args(_argv(tmp_path))
    args.report = args.dataset_root / "command.json"
    with pytest.raises(RuntimeError, match="overlaps protected"):
        cli.validate_command(args)


@pytest.mark.parametrize("leaf", ("test", "fresh_seed", "m42_final_v0"))
def test_explicit_input_path_cannot_name_forbidden_evidence(tmp_path: Path, leaf: str) -> None:
    args = cli.parse_args(_argv(tmp_path))
    args.m42_diagnostics_root = tmp_path / leaf
    with pytest.raises(RuntimeError, match="prohibited test, fresh-seed, or final"):
        cli.validate_command(args)


def test_deployment_parent_named_test_is_not_rejected(tmp_path: Path) -> None:
    deployment = tmp_path / "test" / "deployment"
    args = cli.parse_args(_argv(deployment))
    resolved = cli.validate_command(args)
    assert resolved.dataset_root == (deployment / "dataset").resolve()


@pytest.mark.parametrize("field", ("output_root", "report"))
def test_output_and_report_cannot_be_mislabeled_as_final_or_test(
    tmp_path: Path, field: str
) -> None:
    args = cli.parse_args(_argv(tmp_path))
    suffix = "command.json" if field == "report" else "evidence"
    setattr(args, field, tmp_path / "m42_final_v0" / suffix)
    with pytest.raises(RuntimeError, match="prohibited test, fresh-seed, or final"):
        cli.validate_command(args)


def test_linked_path_component_is_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(real, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")
    args = cli.parse_args(_argv(tmp_path))
    args.output_root = linked / "evidence"
    with pytest.raises(RuntimeError, match="symlink or junction"):
        cli.validate_command(args)


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("test_split_accessed", True),
        ("fresh_seed_accessed", True),
        ("fresh_seed_schedule_accessed", True),
        ("final_schedule_accessed", True),
        ("final_benchmark_authorized", True),
        ("go_no_go_decision_completed", True),
        ("factor_film_training_completed", True),
        ("smolvla_go", True),
    ),
)
def test_runtime_cannot_claim_forbidden_access_or_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: bool,
) -> None:
    args = cli.parse_args(_argv(tmp_path))
    runtime = _runtime_result(mode="combined", dry_run=True)
    runtime[key] = value
    monkeypatch.setattr(cli, "_load_runtime_entrypoint", lambda *, dry_run: lambda _args: runtime)
    with pytest.raises(RuntimeError, match="must keep"):
        cli.execute(args)


@pytest.mark.parametrize(
    "key", ("test_split_accessed", "fresh_seed_accessed", "final_schedule_accessed")
)
def test_runtime_must_explicitly_report_each_forbidden_access_flag_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    args = cli.parse_args(_argv(tmp_path))
    runtime = _runtime_result(mode="combined", dry_run=True)
    runtime.pop(key)
    monkeypatch.setattr(cli, "_load_runtime_entrypoint", lambda *, dry_run: lambda _args: runtime)
    with pytest.raises(RuntimeError, match="explicitly report false"):
        cli.execute(args)


@pytest.mark.parametrize("identity", ("test", "original_m4_fresh_seed", "m42_final_v0"))
def test_runtime_cannot_reference_forbidden_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity: str,
) -> None:
    args = cli.parse_args(_argv(tmp_path))
    runtime = _runtime_result(mode="combined", dry_run=True)
    runtime["source_identity"] = identity
    monkeypatch.setattr(cli, "_load_runtime_entrypoint", lambda *, dry_run: lambda _args: runtime)
    with pytest.raises(RuntimeError, match="prohibited"):
        cli.execute(args)


def test_runtime_cannot_silently_mix_validation_and_development(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = cli.parse_args(_argv(tmp_path, mode="validation"))
    runtime = _runtime_result(mode="combined", dry_run=True)
    monkeypatch.setattr(cli, "_load_runtime_entrypoint", lambda *, dry_run: lambda _args: runtime)
    with pytest.raises(RuntimeError, match="silently mixed"):
        cli.execute(args)


@pytest.mark.parametrize(
    "section", ("audit_identity", "evidence_identity", "fingerprint_inputs", "portable_manifest")
)
def test_portable_identity_sections_reject_absolute_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    section: str,
) -> None:
    args = cli.parse_args(_argv(tmp_path))
    runtime = _runtime_result(mode="combined", dry_run=True)
    runtime[section] = {"dataset": "C:/machine-specific/dataset"}
    monkeypatch.setattr(cli, "_load_runtime_entrypoint", lambda *, dry_run: lambda _args: runtime)
    with pytest.raises(RuntimeError, match="machine-specific absolute path"):
        cli.execute(args)


def test_task_token_rejection_is_a_required_runtime_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = cli.parse_args(_argv(tmp_path))
    runtime = _runtime_result(mode="combined", dry_run=True)
    runtime["task_token_m42_status"] = "accepted"
    monkeypatch.setattr(cli, "_load_runtime_entrypoint", lambda *, dry_run: lambda _args: runtime)
    with pytest.raises(RuntimeError, match="rejected by M4.2"):
        cli.execute(args)


def test_main_writes_one_machine_report_and_returns_nonzero_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "_load_runtime_entrypoint",
        lambda *, dry_run: lambda _args: (_ for _ in ()).throw(RuntimeError("fixture failure")),
    )
    assert cli.main(_argv(tmp_path)) == 1
    stdout = json.loads(capsys.readouterr().out.strip())
    report = json.loads((tmp_path / "semantic-command.json").read_text(encoding="utf-8"))
    assert stdout["passed"] is False
    assert stdout["report_written"] is True
    assert report["error_type"] == "RuntimeError"
    assert report["semantic_audit_completed"] is False
    assert report["final_schedule_accessed"] is False
    assert report["smolvla_go"] is False


def test_main_success_is_machine_readable_and_keeps_report_outside_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "_load_runtime_entrypoint",
        lambda *, dry_run: lambda _args: _runtime_result(mode="combined", dry_run=True),
    )
    assert cli.main(_argv(tmp_path)) == 0
    stdout = json.loads(capsys.readouterr().out.strip())
    report = json.loads((tmp_path / "semantic-command.json").read_text(encoding="utf-8"))
    assert stdout["passed"] is True
    assert stdout["report_written"] is True
    assert report["schema_version"] == cli.COMMAND_SCHEMA
    assert report["semantic_audit_completed"] is False
    assert not (tmp_path / "semantic-evidence").exists()


def test_unsafe_report_path_is_nonzero_but_stdout_remains_machine_readable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    values = _argv(tmp_path)
    report_index = values.index("--report") + 1
    values[report_index] = str(tmp_path / "semantic-evidence" / "command.json")
    assert cli.main(values) == 1
    stdout = json.loads(capsys.readouterr().out.strip())
    assert stdout["passed"] is False
    assert stdout["report_written"] is False
    assert stdout["report"] is None
