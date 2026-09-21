"""CPU-safe command and target-gate tests for M3A."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from langmani.collection.command_support import validated_dataset_root
from langmani.datasets.types import (
    CollectionConfig,
    OverwritePolicy,
    ReplayValidationMode,
    ReplayValidationResult,
    ResumePolicy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = (PROJECT_ROOT / "outputs").resolve()


def _verifier_args(
    dataset_root: Path,
    *,
    mode: str,
    create_new_run: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        target_smoke=mode == "target_smoke",
        target_full=mode == "target_full",
        dataset_root=dataset_root,
        create_new_run=create_new_run,
    )


def _load_script(module_name: str, relative_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, PROJECT_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_dataset_commands_confine_generated_roots_below_outputs() -> None:
    accepted = validated_dataset_root(
        OUTPUT_ROOT / "datasets" / "m3a" / "test",
        output_root=OUTPUT_ROOT,
    )
    accepted.relative_to(OUTPUT_ROOT)

    with pytest.raises(ValueError, match="must stay under"):
        validated_dataset_root(PROJECT_ROOT / "datasets", output_root=OUTPUT_ROOT)
    with pytest.raises(ValueError, match="child of outputs"):
        validated_dataset_root(OUTPUT_ROOT, output_root=OUTPUT_ROOT)


def test_verifier_cli_separates_target_modes_and_scopes_new_run_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _load_script("langmani_test_verify_m3a_cli", "environment/verify_m3a.py")
    monkeypatch.setattr(sys, "argv", ["verify_m3a.py", "--target-smoke", "--target-full"])
    with pytest.raises(SystemExit) as error:
        verifier.parse_args()
    assert error.value.code == 2

    monkeypatch.setattr(sys, "argv", ["verify_m3a.py", "--create-new-run"])
    with pytest.raises(SystemExit) as error:
        verifier.parse_args()
    assert error.value.code == 2

    monkeypatch.setattr(sys, "argv", ["verify_m3a.py", "--target-full", "--create-new-run"])
    args = verifier.parse_args()
    assert args.target_full
    assert args.create_new_run


def test_target_smoke_orders_prior_milestones_before_one_group_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _load_script("langmani_test_verify_m3a_order", "environment/verify_m3a.py")
    dataset_root = OUTPUT_ROOT / "datasets" / "m3a" / "order-test"
    events: list[str] = []
    monkeypatch.setattr(
        verifier,
        "parse_args",
        lambda: _verifier_args(dataset_root, mode="target_smoke"),
    )
    monkeypatch.setattr(verifier, "_is_native_linux", lambda: True)
    monkeypatch.setattr(
        verifier,
        "_structural_checks",
        lambda report: events.append("M3A structural checks"),
    )
    monkeypatch.setattr(
        verifier,
        "_run_command",
        lambda report, name, arguments: events.append(name) or True,
    )
    monkeypatch.setattr(
        verifier,
        "_validate_target_archive",
        lambda report, root, **kwargs: events.append("M3A archive validation") or True,
    )
    monkeypatch.setattr(verifier.Report, "write", lambda self, **kwargs: None)

    assert verifier.main() == 0
    assert events == [
        "M3A structural checks",
        "ordered M0/M1/M2 target gate",
        "M3A one-group smoke collection",
        "M3A one-group smoke structural inspection",
        "M3A archive validation",
        "M3A independent six-episode smoke action replay",
    ]


def test_target_smoke_stops_after_failed_prior_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _load_script("langmani_test_verify_m3a_stop", "environment/verify_m3a.py")
    dataset_root = OUTPUT_ROOT / "datasets" / "m3a" / "stop-test"
    events: list[str] = []
    monkeypatch.setattr(
        verifier,
        "parse_args",
        lambda: _verifier_args(dataset_root, mode="target_smoke"),
    )
    monkeypatch.setattr(verifier, "_is_native_linux", lambda: True)
    monkeypatch.setattr(verifier, "_structural_checks", lambda report: None)

    def fail_prior(report: Any, name: str, arguments: list[str]) -> bool:
        del arguments
        events.append(name)
        report.check(name, False, "deliberate failure")
        return False

    monkeypatch.setattr(verifier, "_run_command", fail_prior)
    monkeypatch.setattr(verifier.Report, "write", lambda self, **kwargs: None)

    assert verifier.main() == 1
    assert events == ["ordered M0/M1/M2 target gate"]


def test_collection_command_materializes_all_explicit_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _load_script("langmani_test_collect_m3a", "environment/collect_raw_demos.py")
    dataset_root = OUTPUT_ROOT / "datasets" / "m3a" / "command-test"
    args = SimpleNamespace(
        output_root=dataset_root,
        candidate_seed_start=7,
        target_complete_scene_count=2,
        maximum_candidate_scene_count=4,
        maximum_expert_attempts_per_task=2,
        shard_size=12,
        sim_backend="physx_cpu",
        replay_validation_mode="action",
        final_position_tolerance_m=0.004,
        final_orientation_tolerance_rad=0.04,
        final_joint_tolerance_rad=0.03,
        retain_failed_raw_trajectories=True,
        overwrite=False,
        no_resume=False,
    )
    captured: dict[str, Any] = {}

    class Summary:
        status = command.CollectionStatus.COMPLETE

        def to_dict(self) -> dict[str, Any]:
            return {"status": "complete"}

    class Collector:
        def __init__(self, config: Any) -> None:
            captured["config"] = config

        def collect(self) -> Summary:
            return Summary()

    monkeypatch.setattr(command, "parse_args", lambda: args)
    monkeypatch.setattr(
        command,
        "installed_runtime_versions",
        lambda: {
            "gymnasium": "1.2.3",
            "mani_skill": "3.0.1",
            "h5py": "3.16.0",
            "numpy": "1.26.4",
            "opencv_python": "4.11.0.86",
            "pillow": "12.3.0",
            "sapien": "3.0.3",
            "scipy": "1.15.3",
            "torch": "2.11.0+cpu",
            "mplib": None,
        },
    )
    monkeypatch.setattr(command, "RawDemonstrationCollector", Collector)
    monkeypatch.setattr(
        command,
        "write_json_atomic",
        lambda path, payload: captured.update(report=payload) or Path(path),
    )

    assert command.main() == 0
    config = captured["config"]
    assert config.candidate_scene_seed_start == 7
    assert config.target_complete_scene_count == 2
    assert config.maximum_candidate_scene_count == 4
    assert config.maximum_expert_attempts_per_task == 2
    assert config.shard_size == 12
    assert config.retain_failed_raw_trajectories
    assert config.replay_validation_mode.value == "action"
    assert config.sim_backend == "physx_cpu"
    assert captured["report"]["passed"] is True


def test_target_full_uses_exact_prior_gate_and_unlimited_360_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _load_script("langmani_test_verify_m3a_arguments", "environment/verify_m3a.py")
    dataset_root = OUTPUT_ROOT / "datasets" / "m3a" / "argument-test"
    commands: dict[str, list[str]] = {}
    monkeypatch.setattr(
        verifier,
        "parse_args",
        lambda: _verifier_args(dataset_root, mode="target_full"),
    )
    monkeypatch.setattr(verifier, "_is_native_linux", lambda: True)
    monkeypatch.setattr(verifier, "resolve_planner_python", lambda: "planner-side-python")
    monkeypatch.setattr(verifier, "_structural_checks", lambda report: None)
    monkeypatch.setattr(verifier, "_prepare_full_archive", lambda *args, **kwargs: "collect")

    def capture(report: Any, name: str, arguments: list[str]) -> bool:
        del report
        commands[name] = arguments
        return True

    monkeypatch.setattr(verifier, "_run_command", capture)
    monkeypatch.setattr(
        verifier,
        "_validate_target_archive",
        lambda report, root, **kwargs: True,
    )
    monkeypatch.setattr(verifier.Report, "write", lambda self, **kwargs: None)

    assert verifier.main() == 0
    assert commands["ordered M0/M1/M2 target gate"] == [
        sys.executable,
        "environment/verify_m2.py",
        "--target",
    ]
    replay = commands["M3A independent 360-episode action replay"]
    assert replay[0] == "planner-side-python"
    assert "--limit" not in replay
    assert replay[replay.index("--mode") + 1] == "action_and_state_audit"
    collection = commands["M3A 60-group collection/resume"]
    assert collection[0] == "planner-side-python"
    assert "--overwrite" not in collection
    assert "--no-resume" not in collection


def test_smoke_collection_has_one_group_and_never_overwrites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _load_script("langmani_test_verify_m3a_smoke_args", "environment/verify_m3a.py")
    dataset_root = OUTPUT_ROOT / "datasets" / "m3a" / "fresh-smoke-test"
    monkeypatch.setattr(verifier, "resolve_planner_python", lambda: "planner-side-python")

    command = verifier._collection_command(dataset_root, smoke=True)

    assert command[0] == "planner-side-python"
    assert command[command.index("--target-complete-scene-count") + 1] == "1"
    assert command[command.index("--maximum-candidate-scene-count") + 1] == "20"
    assert command[command.index("--shard-size") + 1] == "6"
    assert "--overwrite" not in command
    assert "--no-resume" not in command
    assert verifier._replay_command(dataset_root)[0] == "planner-side-python"
    assert verifier._inspection_command(dataset_root)[0] == sys.executable


@pytest.mark.parametrize(
    ("status", "expected_action"),
    [
        ("complete", "reuse"),
        ("in_progress", "collect"),
        ("failed", "fail"),
    ],
)
def test_full_archive_reuses_complete_or_resumes_only_legal_incomplete_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    expected_action: str,
) -> None:
    verifier = _load_script(
        f"langmani_test_verify_m3a_resume_{status}",
        "environment/verify_m3a.py",
    )
    dataset_root = tmp_path / "dataset"
    manifest_path = dataset_root / "manifests" / "collection_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        verifier,
        "load_manifest",
        lambda root: SimpleNamespace(status=verifier.CollectionStatus(status), config=object()),
    )
    monkeypatch.setattr(verifier, "_is_expected_full_config", lambda config, root: True)
    report = verifier.Report()

    action = verifier._prepare_full_archive(
        report,
        dataset_root,
        create_new_run=False,
    )

    assert action == expected_action
    assert report.failed is (status == "failed")


def test_full_archive_requires_explicit_creation_and_never_replaces_existing_root(
    tmp_path: Path,
) -> None:
    verifier = _load_script("langmani_test_verify_m3a_create", "environment/verify_m3a.py")
    missing_root = tmp_path / "missing"

    report = verifier.Report()
    assert verifier._prepare_full_archive(report, missing_root, create_new_run=False) == "fail"
    assert report.failed

    report = verifier.Report()
    assert verifier._prepare_full_archive(report, missing_root, create_new_run=True) == "collect"
    assert not report.failed

    missing_root.mkdir()
    report = verifier.Report()
    assert verifier._prepare_full_archive(report, missing_root, create_new_run=True) == "fail"
    assert report.failed


def test_full_archive_reuse_requires_the_complete_default_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = _load_script(
        "langmani_test_verify_m3a_full_identity",
        "environment/verify_m3a.py",
    )
    dataset_root = (tmp_path / "dataset").resolve()
    expected = CollectionConfig(raw_output_root=str(dataset_root))
    # This contract test must not compare the caller's packages with an external
    # LANGMANI_PLANNER_PYTHON environment selected by the machine running pytest.
    monkeypatch.setattr(
        verifier, "query_planner_runtime_versions", lambda: dict(expected.runtime_versions)
    )

    assert verifier._is_expected_full_config(expected, dataset_root)
    assert not verifier._is_expected_full_config(
        replace(expected, raw_output_root=str(tmp_path / "copied-elsewhere")),
        dataset_root,
    )
    assert not verifier._is_expected_full_config(
        replace(expected, overwrite_policy=OverwritePolicy.REPLACE),
        dataset_root,
    )
    assert not verifier._is_expected_full_config(
        replace(expected, resume_policy=ResumePolicy.ERROR),
        dataset_root,
    )
    drifted_versions = dict(expected.runtime_versions)
    drifted_versions["mani_skill"] = "different-version"
    assert not verifier._is_expected_full_config(
        replace(expected, runtime_versions=drifted_versions),
        dataset_root,
    )


def test_full_validation_of_existing_complete_archive_skips_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _load_script("langmani_test_verify_m3a_reuse", "environment/verify_m3a.py")
    dataset_root = OUTPUT_ROOT / "datasets" / "m3a" / "reuse-test"
    commands: list[str] = []
    monkeypatch.setattr(verifier, "_prepare_full_archive", lambda *args, **kwargs: "reuse")
    monkeypatch.setattr(
        verifier,
        "_run_command",
        lambda report, name, arguments: commands.append(name) or True,
    )
    monkeypatch.setattr(verifier, "_validate_target_archive", lambda *args, **kwargs: True)
    report = verifier.Report(implementation_validated=True, prior_target_gates_validated=True)

    verifier._run_target_full(report, dataset_root, create_new_run=False)

    assert commands == [
        "M3A independent full-dataset structural inspection",
        "M3A independent 360-episode action replay",
    ]
    assert report.full_dataset_validated
    assert report.action_replay_validated


def test_verification_report_contains_independent_structured_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _load_script("langmani_test_verify_m3a_flags", "environment/verify_m3a.py")
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        verifier,
        "write_json_atomic",
        lambda path, payload: captured.update(payload=payload) or Path(path),
    )
    report = verifier.Report(
        implementation_validated=True,
        prior_target_gates_validated=True,
        expert_collection_smoke_validated=True,
        action_replay_validated=False,
        full_dataset_validated=False,
    )

    report.write(mode="target_smoke", dataset_root=tmp_path / "missing-source")

    payload = captured["payload"]
    assert payload["schema_version"] == "langmani-m3a-verification-v3"
    assert payload["implementation_validated"] is True
    assert payload["prior_target_gates_validated"] is True
    assert payload["expert_collection_smoke_validated"] is True
    assert payload["action_replay_validated"] is False
    assert payload["full_dataset_validated"] is False
    assert payload["physical_target_validated"] is False
    assert payload["passed"] is False
    assert payload["source_archive_identity_status"] == "absent"
    assert payload["source_archive_identity_error"] is None
    assert payload["collection_run_id"] is None
    assert payload["config_fingerprint"] is None
    assert payload["source_archive_digest"] is None


def test_verification_report_binds_current_validated_source_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _load_script("langmani_test_verify_m3a_source_identity", "environment/verify_m3a.py")
    dataset_root = tmp_path / "dataset"
    manifest_path = dataset_root / "manifests" / "collection_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}\n", encoding="utf-8")
    manifest = SimpleNamespace(
        collection_run_id="collection-run",
        schedule=SimpleNamespace(config_fingerprint="sha256:" + "1" * 64),
    )
    calls: list[str] = []
    monkeypatch.setattr(
        verifier,
        "inspect_raw_dataset",
        lambda root: calls.append(f"inspect:{Path(root).resolve()}") or object(),
    )
    monkeypatch.setattr(verifier, "load_manifest", lambda root: manifest)
    monkeypatch.setattr(verifier, "source_archive_digest", lambda value: "sha256:" + "2" * 64)

    identity = verifier._validated_source_archive_identity(dataset_root)

    assert calls == [f"inspect:{dataset_root.resolve()}"]
    assert identity == {
        "source_archive_identity_status": "validated",
        "source_archive_identity_error": None,
        "collection_run_id": "collection-run",
        "config_fingerprint": "sha256:" + "1" * 64,
        "source_archive_digest": "sha256:" + "2" * 64,
    }


def test_verification_report_marks_invalid_source_identity_without_stale_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _load_script(
        "langmani_test_verify_m3a_invalid_identity", "environment/verify_m3a.py"
    )
    dataset_root = tmp_path / "dataset"
    manifest_path = dataset_root / "manifests" / "collection_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        verifier,
        "inspect_raw_dataset",
        lambda root: (_ for _ in ()).throw(RuntimeError("changed source shard")),
    )

    identity = verifier._validated_source_archive_identity(dataset_root)

    assert identity["source_archive_identity_status"] == "invalid"
    assert identity["source_archive_identity_error"] == "RuntimeError: changed source shard"
    assert identity["collection_run_id"] is None
    assert identity["config_fingerprint"] is None
    assert identity["source_archive_digest"] is None


def test_target_command_rejects_stale_success_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _load_script("langmani_test_verify_m3a_fresh", "environment/verify_m3a.py")
    name = "M3A independent full-dataset structural inspection"
    report_path = tmp_path / "stale.json"
    report_path.write_text(json.dumps({"passed": True}), encoding="utf-8")
    verifier.COMMAND_REPORTS[name] = report_path
    monkeypatch.setattr(
        verifier.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    report = verifier.Report()

    assert not verifier._run_command(
        report,
        name,
        [
            sys.executable,
            "environment/inspect_raw_demos.py",
            "--dataset-root",
            str(tmp_path / "dataset"),
        ],
    )
    assert not report_path.exists()
    assert report.failed


@pytest.mark.parametrize(
    ("command_name", "episode_count", "group_count"),
    [
        ("M3A independent six-episode smoke action replay", 6, 1),
        ("M3A independent 360-episode action replay", 360, 60),
    ],
)
def test_replay_report_contract_requires_exact_manifest_without_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command_name: str,
    episode_count: int,
    group_count: int,
) -> None:
    verifier = _load_script("langmani_test_verify_m3a_replay_report", "environment/verify_m3a.py")
    dataset_root = (tmp_path / "dataset").resolve()
    report_path = tmp_path / "replay.json"
    arguments = [
        sys.executable,
        "environment/replay_raw_demos.py",
        "--dataset-root",
        str(dataset_root),
        "--mode",
        "action_and_state_audit",
    ]
    records = tuple(
        SimpleNamespace(
            raw_trajectory_id=f"raw-{index}",
            source_shard_id=f"shard-{index // 60}",
            native_episode_id=index % 60,
        )
        for index in range(episode_count)
    )
    config = SimpleNamespace(
        replay_validation_mode=ReplayValidationMode.ACTION_AND_STATE_AUDIT,
        target_complete_scene_count=group_count,
        final_position_tolerance_m=0.005,
        final_orientation_tolerance_rad=0.05,
        final_joint_tolerance_rad=0.05,
    )
    monkeypatch.setattr(
        verifier,
        "load_manifest",
        lambda root: SimpleNamespace(
            raw_episodes=records,
            scene_groups=tuple(SimpleNamespace(accepted=True) for _ in range(group_count)),
            config=config,
        ),
    )
    payload: dict[str, Any] = {
        "schema_version": "langmani-m3a-replay-report-v1",
        "dataset_root": str(dataset_root),
        "selected_episode_count": episode_count,
        "passed_episode_count": episode_count,
        "command_errors": [],
        "results": [{} for _ in range(episode_count)],
        "passed": True,
    }
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    valid, _ = verifier._validate_command_report(
        command_name,
        report_path,
        arguments,
    )
    assert not valid

    validation = ReplayValidationResult(
        passed=True,
        mode=ReplayValidationMode.ACTION_AND_STATE_AUDIT,
        recorded_success=True,
        replay_success=True,
        task_spec_matches=True,
        wrong_object_in_target_bin=False,
        target_in_wrong_bin=False,
        target_off_table=False,
        final_environment_evaluation={"success": True, "fail": False},
        recorded_action_steps=2,
        replayed_action_steps=2,
        final_position_error_m=0.0,
        final_orientation_error_rad=0.0,
        final_joint_error_rad=0.0,
        state_audit_performed=True,
        state_audit_passed=True,
    ).to_dict()
    payload["results"] = [
        {
            "raw_trajectory_id": record.raw_trajectory_id,
            "source_shard_id": record.source_shard_id,
            "native_episode_id": record.native_episode_id,
            "validation": validation,
        }
        for record in records
    ]
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    valid, _ = verifier._validate_command_report(
        command_name,
        report_path,
        arguments,
    )
    assert valid

    payload["results"][1] = dict(payload["results"][0])
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    valid, _ = verifier._validate_command_report(
        command_name,
        report_path,
        arguments,
    )
    assert not valid


def test_target_mode_on_windows_is_honest_and_starts_no_subcommands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = _load_script("langmani_test_verify_m3a_windows", "environment/verify_m3a.py")
    dataset_root = OUTPUT_ROOT / "datasets" / "m3a" / "windows-test"
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        verifier,
        "parse_args",
        lambda: _verifier_args(dataset_root, mode="target_full"),
    )
    monkeypatch.setattr(verifier, "_is_native_linux", lambda: False)
    monkeypatch.setattr(
        verifier,
        "_run_command",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("non-Linux target must not start subcommands")
        ),
    )
    monkeypatch.setattr(
        verifier,
        "write_json_atomic",
        lambda path, payload: captured.update(payload=payload) or Path(path),
    )

    assert verifier.main() == 1
    assert captured["payload"]["validation_scope"] == "target_full"
    assert captured["payload"]["implementation_validated"] is True
    assert captured["payload"]["prior_target_gates_validated"] is False
    assert captured["payload"]["expert_collection_smoke_validated"] is False
    assert captured["payload"]["action_replay_validated"] is False
    assert captured["payload"]["full_dataset_validated"] is False
    assert captured["payload"]["physical_target_validated"] is False
    assert captured["payload"]["physical_acceptance"] is False
    assert captured["payload"]["passed"] is False


@pytest.mark.parametrize(
    ("relative_path", "report_attribute"),
    [
        ("environment/verify_m3a.py", "REPORT_PATH"),
        ("environment/collect_raw_demos.py", "COMMAND_REPORT"),
        ("environment/inspect_raw_demos.py", "DEFAULT_REPORT"),
        ("environment/replay_raw_demos.py", "DEFAULT_REPORT"),
    ],
)
def test_m3a_commands_invalidate_stale_report_before_argument_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
    report_attribute: str,
) -> None:
    module_name = "langmani_test_stale_" + Path(relative_path).stem
    command = _load_script(module_name, relative_path)
    report_path = tmp_path / f"{Path(relative_path).stem}.json"
    report_path.write_text('{"passed":true}\n', encoding="utf-8")
    monkeypatch.setattr(command, report_attribute, report_path)
    monkeypatch.setattr(
        command,
        "parse_args",
        lambda: (_ for _ in ()).throw(SystemExit(2)),
    )

    with pytest.raises(SystemExit) as error:
        command.main()
    assert error.value.code == 2
    assert not report_path.exists()
