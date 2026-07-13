"""Command-boundary tests for M3B export, validation, inspection, and verification."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from langmani.datasets.lerobot_types import ExportMode

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_script(module_name: str, relative_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, PROJECT_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _LazyScript:
    """Load a command script on first use while forwarding monkeypatch writes."""

    def __init__(self, module_name: str, relative_path: str) -> None:
        object.__setattr__(self, "_module_name", module_name)
        object.__setattr__(self, "_relative_path", relative_path)
        object.__setattr__(self, "_module", None)

    def _loaded(self) -> ModuleType:
        module = object.__getattribute__(self, "_module")
        if module is None:
            module = _load_script(
                object.__getattribute__(self, "_module_name"),
                object.__getattribute__(self, "_relative_path"),
            )
            object.__setattr__(self, "_module", module)
        return module

    def __getattr__(self, name: str) -> object:
        return getattr(self._loaded(), name)

    def __setattr__(self, name: str, value: object) -> None:
        if name in {"_module_name", "_relative_path", "_module"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self._loaded(), name, value)


export_cli = _LazyScript("langmani_test_export_lerobot_cli", "scripts/export_lerobot_dataset.py")
validate_cli = _LazyScript(
    "langmani_test_validate_lerobot_cli", "scripts/validate_lerobot_dataset.py"
)
inspect_cli = _LazyScript("langmani_test_inspect_lerobot_cli", "scripts/inspect_lerobot_episode.py")
verify_m3b = _LazyScript("langmani_test_verify_m3b_cli", "environment/verify_m3b.py")


def _export_args(tmp_path: Path, *, dry_run: bool) -> argparse.Namespace:
    return argparse.Namespace(
        source_root=tmp_path / "source",
        output_root=tmp_path / "output",
        repo_id="langmani/test",
        smoke=True,
        full=False,
        dry_run=dry_run,
        split_seed=0,
        clean_staging=False,
        source_verification_report=tmp_path / "m3a.json",
        report=tmp_path / "export-report.json",
    )


def _validation_payload(*, passed: bool = True) -> dict[str, object]:
    return {
        "dataset_load_validated": passed,
        "feature_schema_validated": passed,
        "parquet_validated": passed,
        "video_decode_validated": passed,
        "source_alignment_validated": passed,
        "split_integrity_validated": passed,
        "privileged_leakage_validated": passed,
        "dataloader_validated": passed,
        "passed": passed,
    }


@pytest.mark.parametrize(
    "argv",
    [
        ["export_lerobot_dataset.py"],
        ["export_lerobot_dataset.py", "--smoke", "--full"],
    ],
)
def test_export_cli_requires_exactly_one_smoke_or_full_mode(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as error:
        export_cli.parse_args()
    assert error.value.code == 2


def test_export_cli_dry_run_only_builds_plan_and_writes_machine_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = _export_args(tmp_path, dry_run=True)
    calls: list[str] = []
    config = object()

    class _Plan:
        def to_dict(self, candidate: object) -> dict[str, object]:
            assert candidate is config
            return {
                "schema_version": "langmani-m3b-export-plan-v1",
                "dry_run": True,
                "source_episode_count": 6,
            }

    monkeypatch.setattr(export_cli, "parse_args", lambda: args)
    monkeypatch.setattr(export_cli, "_config", lambda parsed: config)

    def _build(candidate: object, *, source_verification_report: Path) -> _Plan:
        assert candidate is config
        assert source_verification_report == args.source_verification_report
        calls.append("build_plan")
        return _Plan()

    def _forbidden_export(*_args: object, **_kwargs: object) -> None:
        calls.append("export")
        raise AssertionError("dry-run must not create a LeRobot dataset")

    monkeypatch.setattr(export_cli, "build_export_plan", _build)
    monkeypatch.setattr(export_cli, "export_lerobot_dataset", _forbidden_export)

    assert export_cli.main() == 0
    assert calls == ["build_plan"]
    report = json.loads(args.report.read_text(encoding="utf-8"))
    stdout = json.loads(capsys.readouterr().out.strip())
    assert report["passed"] is True
    assert report["dry_run"] is True
    assert report["source_episode_count"] == 6
    assert stdout["passed"] is True
    assert Path(stdout["report"]).resolve() == args.report.resolve()


def test_validate_metadata_only_cannot_claim_full_acceptance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        dataset_root=tmp_path / "dataset",
        source_root=tmp_path / "source",
        smoke=True,
        full=False,
        metadata_only=True,
        source_verification_report=tmp_path / "m3a.json",
        report=tmp_path / "validation-report.json",
    )
    validation = SimpleNamespace(
        dataset_load_validated=True,
        feature_schema_validated=True,
        parquet_validated=True,
        video_decode_validated=True,
        source_alignment_validated=True,
        split_integrity_validated=True,
        privileged_leakage_validated=True,
        dataloader_validated=True,
        passed=True,
        to_dict=lambda: _validation_payload(passed=True),
    )
    summary = SimpleNamespace(
        mode=ExportMode.SMOKE,
        to_dict=lambda: {"mode": "smoke", "total_episodes": 6},
    )
    monkeypatch.setattr(validate_cli, "parse_args", lambda: args)
    monkeypatch.setattr(
        validate_cli,
        "validate_lerobot_dataset",
        lambda *_args, **_kwargs: (validation, summary),
    )

    assert validate_cli.main() == 0
    report = json.loads(args.report.read_text(encoding="utf-8"))
    assert report["metadata_only"] is True
    assert report["passed"] is True
    assert report["full_acceptance"] is False


def _patch_inspection_sidecars(
    monkeypatch: pytest.MonkeyPatch,
    *,
    completion_fingerprint: str,
    manifest_fingerprint: str,
    episode_count: int,
) -> None:
    records = tuple(SimpleNamespace() for _ in range(episode_count))
    manifest = SimpleNamespace(export_fingerprint=manifest_fingerprint, episodes=records)
    summary = SimpleNamespace()
    validation = SimpleNamespace(source_alignments=(), video_results=())
    monkeypatch.setattr(
        inspect_cli,
        "LeRobotExportManifest",
        SimpleNamespace(from_dict=lambda _payload: manifest),
    )
    monkeypatch.setattr(
        inspect_cli,
        "LeRobotDatasetSummary",
        SimpleNamespace(from_dict=lambda _payload: summary),
    )
    monkeypatch.setattr(
        inspect_cli,
        "LeRobotValidationReport",
        SimpleNamespace(from_dict=lambda _payload: validation),
    )
    monkeypatch.setattr(
        inspect_cli,
        "_read_json",
        lambda path: (
            {"export_fingerprint": completion_fingerprint}
            if path.name == Path(inspect_cli.COMPLETION_MARKER).name
            else {}
        ),
    )


def test_inspect_rejects_missing_or_mismatched_completion_before_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_inspection_sidecars(
        monkeypatch,
        completion_fingerprint="sha256:" + "a" * 64,
        manifest_fingerprint="sha256:" + "b" * 64,
        episode_count=1,
    )

    with pytest.raises(RuntimeError, match="completion marker fingerprint"):
        inspect_cli._inspect(tmp_path, 0, None)


def test_inspect_rejects_missing_completion_marker_before_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fingerprint = "sha256:" + "a" * 64
    _patch_inspection_sidecars(
        monkeypatch,
        completion_fingerprint=fingerprint,
        manifest_fingerprint=fingerprint,
        episode_count=1,
    )

    def _missing_marker(path: Path) -> dict[str, object]:
        if path.name == Path(inspect_cli.COMPLETION_MARKER).name:
            raise RuntimeError("completion marker is missing")
        return {}

    monkeypatch.setattr(inspect_cli, "_read_json", _missing_marker)
    with pytest.raises(RuntimeError, match="completion marker is missing"):
        inspect_cli._inspect(tmp_path, 0, None)


@pytest.mark.parametrize("episode_index", [-1, 1])
def test_inspect_rejects_episode_bounds_before_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    episode_index: int,
) -> None:
    fingerprint = "sha256:" + "a" * 64
    _patch_inspection_sidecars(
        monkeypatch,
        completion_fingerprint=fingerprint,
        manifest_fingerprint=fingerprint,
        episode_count=1,
    )

    with pytest.raises(IndexError, match="episode index"):
        inspect_cli._inspect(tmp_path, episode_index, None)


def test_verify_report_has_all_independent_flags_and_structural_is_not_physical(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "verification.json"
    monkeypatch.setattr(verify_m3b, "REPORT_PATH", report_path)
    report = verify_m3b.Report()
    report.implementation_validated = True

    report.write(
        mode="structural",
        source_root=tmp_path / "source",
        dataset_root=tmp_path / "dataset",
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    required_flags = {
        "implementation_validated",
        "source_archive_validated",
        "smoke_export_validated",
        "full_export_validated",
        "lerobot_load_validated",
        "parquet_validated",
        "video_decode_validated",
        "source_alignment_validated",
        "split_integrity_validated",
        "privileged_leakage_validated",
        "physical_target_validated",
    }
    assert required_flags <= payload.keys()
    assert payload["implementation_validated"] is True
    assert payload["physical_target_validated"] is False
    assert payload["passed"] is True


def test_verify_target_smoke_and_full_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify_m3b.py", "--target-smoke", "--target-full"],
    )
    with pytest.raises(SystemExit) as error:
        verify_m3b.parse_args()
    assert error.value.code == 2


def test_verify_m3a_gate_deletes_stale_report_and_consumes_fresh_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    m3a_report = tmp_path / "m3a-verification.json"
    m3a_report.write_text('{"stale": true}', encoding="utf-8")
    monkeypatch.setattr(verify_m3b, "M3A_REPORT", m3a_report)
    source_root = tmp_path / "source"
    calls: list[str] = []

    def _run(
        report: verify_m3b.Report,
        name: str,
        arguments: list[str],
    ) -> bool:
        assert not m3a_report.exists()
        calls.append(name)
        assert arguments[1:] == ["environment/verify_m3a.py", "--target-smoke"]
        m3a_report.write_text(
            json.dumps(
                {
                    "schema_version": "langmani-m3a-verification-v3",
                    "verification_mode": "target_smoke",
                    "passed": True,
                    "prior_target_gates_validated": True,
                    "action_replay_validated": True,
                    "physical_target_validated": True,
                    "source_archive_identity_status": "validated",
                    "expert_collection_smoke_validated": True,
                    "dataset_root": str(source_root),
                }
            ),
            encoding="utf-8",
        )
        return True

    monkeypatch.setattr(verify_m3b, "_run", _run)
    report = verify_m3b.Report()

    assert verify_m3b._run_m3a_gate(report, "target_smoke", source_root, False) == source_root
    assert calls == ["ordered M0/M1/M2 plus M3A target gate"]


def _target_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        source_root=tmp_path / "source",
        dataset_root=tmp_path / "dataset",
        create_new_source=False,
        clean_staging=False,
    )


def _patch_target_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        verify_m3b,
        "_read_report",
        lambda path: (
            {"passed": True}
            if path == verify_m3b.EXPORT_REPORT
            else {"passed": True, "validation": _validation_payload(passed=True)}
        ),
    )


def test_target_runs_fresh_m3a_gate_before_export_then_independent_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args = _target_args(tmp_path)
    source_root = args.source_root.resolve()
    smoke_root = tmp_path / "fresh-smoke-output"
    events: list[str] = []

    def _m3a_gate(*_args: object, **_kwargs: object) -> Path:
        events.append("m3a")
        return source_root

    def _run(report: verify_m3b.Report, name: str, arguments: list[str]) -> bool:
        del report, arguments
        events.append(name)
        return True

    monkeypatch.setattr(verify_m3b, "_run_m3a_gate", _m3a_gate)
    monkeypatch.setattr(verify_m3b, "_fresh_smoke_output", lambda: smoke_root)
    monkeypatch.setattr(verify_m3b, "_run", _run)
    _patch_target_reports(monkeypatch)
    report = verify_m3b.Report(implementation_validated=True)

    returned_source, returned_dataset = verify_m3b._run_target(report, "target_smoke", args)

    assert events == ["m3a", "staged M3B export", "independent finalized M3B validation"]
    assert returned_source == source_root
    assert returned_dataset == smoke_root
    assert report.smoke_export_validated is True
    assert report.physical_target_validated is True


def test_existing_complete_full_is_reused_but_always_independently_validated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args = _target_args(tmp_path)
    args.dataset_root.mkdir(parents=True)
    completion = args.dataset_root / verify_m3b.COMPLETION_MARKER
    completion.parent.mkdir(parents=True)
    completion.write_text("{}", encoding="utf-8")
    events: list[str] = []
    monkeypatch.setattr(
        verify_m3b,
        "_run_m3a_gate",
        lambda *_args, **_kwargs: args.source_root.resolve(),
    )

    def _run(report: verify_m3b.Report, name: str, arguments: list[str]) -> bool:
        del report, arguments
        events.append(name)
        return True

    monkeypatch.setattr(verify_m3b, "_run", _run)
    _patch_target_reports(monkeypatch)
    report = verify_m3b.Report(implementation_validated=True)

    verify_m3b._run_target(report, "target_full", args)

    assert events == ["independent finalized M3B validation"]
    assert report.full_export_validated is True
    assert report.physical_target_validated is True


def test_reused_full_dataset_is_not_validated_when_independent_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args = _target_args(tmp_path)
    args.dataset_root.mkdir(parents=True)
    completion = args.dataset_root / verify_m3b.COMPLETION_MARKER
    completion.parent.mkdir(parents=True)
    completion.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        verify_m3b,
        "_run_m3a_gate",
        lambda *_args, **_kwargs: args.source_root.resolve(),
    )
    monkeypatch.setattr(verify_m3b, "_run", lambda *_args, **_kwargs: False)
    report = verify_m3b.Report(implementation_validated=True)

    verify_m3b._run_target(report, "target_full", args)

    assert report.full_export_validated is False
    assert report.physical_target_validated is False
