from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from langmani.policies.act_runtime import GitState
from langmani.policies.m42_schedule import M42_DEV_SCHEDULE_ID, M42_FINAL_SCHEDULE_ID
from langmani.policies.m42_types import TaskTokenValidationResult


def _load_cli() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_m42.py"
    spec = importlib.util.spec_from_file_location("langmani_test_evaluate_m42", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cli = _load_cli()


def test_evaluation_rejects_completed_plus_incomplete_task_token_siblings(
    tmp_path: Path,
) -> None:
    root = tmp_path / "task-token-models"
    complete = root / ("a" * 64)
    incomplete = root / ("b" * 64)
    complete.mkdir(parents=True)
    incomplete.mkdir()
    (complete / "training_complete.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="exactly one fingerprint-owned"):
        cli._load_unique_task_token_run(root)


def test_failed_development_metrics_never_write_a_passed_completion(tmp_path: Path) -> None:
    comparison, completion = cli._persist_development_success(
        output_root=tmp_path,
        comparison={"complete": False, "metrics_validated": False},
        completion={"passed": False},
        passed=False,
    )
    assert comparison is None and completion is None
    assert not (tmp_path / "development_comparison.json").exists()
    assert not (tmp_path / "development_complete.json").exists()


def _digest(character: str) -> str:
    return f"sha256:{character * 64}"


def _git(*, dirty: bool = False) -> GitState:
    return GitState(
        commit="a" * 40,
        dirty=dirty,
        changed_paths=("M fixture",) if dirty else (),
        baseline_tracked=True,
    )


def _args(tmp_path: Path, *, stage: str, schedule: str, dry_run: bool) -> object:
    values = [
        "--stage",
        stage,
        "--schedule",
        schedule,
        "--output-root",
        str(tmp_path / "evaluation"),
        "--report",
        str(tmp_path / "command.json"),
    ]
    if dry_run:
        values.append("--dry-run")
    return cli.parse_args(values)


def test_stage_and_schedule_are_an_exact_pair(tmp_path: Path) -> None:
    args = _args(
        tmp_path,
        stage="development",
        schedule=M42_FINAL_SCHEDULE_ID,
        dry_run=True,
    )
    with pytest.raises(ValueError, match="development stage requires"):
        cli.execute(args)

    args = _args(
        tmp_path,
        stage="final",
        schedule=M42_DEV_SCHEDULE_ID,
        dry_run=True,
    )
    with pytest.raises(ValueError, match="final stage requires"):
        cli.execute(args)


@pytest.mark.parametrize(
    ("stage", "schedule", "episode_count", "authorization"),
    (
        ("development", M42_DEV_SCHEDULE_ID, 72, False),
        ("final", M42_FINAL_SCHEDULE_ID, 180, True),
    ),
)
def test_dry_run_never_materializes_or_executes_a_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    schedule: str,
    episode_count: int,
    authorization: bool,
) -> None:
    monkeypatch.setattr(cli, "inspect_git_state", lambda _root: _git(dirty=True))
    monkeypatch.setattr(
        cli,
        "materialize_schedule",
        lambda *_args, **_kwargs: pytest.fail("dry-run must not materialize episodes"),
    )
    monkeypatch.setattr(
        cli,
        "_create_environment",
        lambda: pytest.fail("dry-run must not create ManiSkill"),
    )
    args = _args(tmp_path, stage=stage, schedule=schedule, dry_run=True)
    result = cli.execute(args)
    assert result["passed"] is True
    assert result["expected_physical_schedule_episode_count"] == episode_count
    assert result["final_authorization_required"] is authorization
    assert result["final_schedule_accessed"] is False
    assert result["physical_execution"] is False
    assert result["task_token_checkpoint_selected"] is False
    assert not (tmp_path / "evaluation").exists()


def test_output_path_cannot_overlap_any_immutable_input(tmp_path: Path) -> None:
    for name in ("dataset", "checkpoints", "task-token", "m4"):
        (tmp_path / name).mkdir()
    runtime = tmp_path / "runtime.json"
    runtime.write_text("{}\n", encoding="utf-8")
    args = cli.parse_args(
        [
            "--stage",
            "development",
            "--schedule",
            M42_DEV_SCHEDULE_ID,
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--checkpoint-root",
            str(tmp_path / "checkpoints"),
            "--task-token-root",
            str(tmp_path / "task-token"),
            "--m4-diagnostics-root",
            str(tmp_path / "m4"),
            "--runtime-selection",
            str(runtime),
            "--output-root",
            str(tmp_path / "dataset" / "evaluation"),
            "--report",
            str(tmp_path / "command.json"),
        ]
    )
    with pytest.raises(RuntimeError, match="must not overlap"):
        cli._validate_paths(args)


def test_command_report_path_cannot_overwrite_an_input(tmp_path: Path) -> None:
    args = _args(
        tmp_path,
        stage="development",
        schedule=M42_DEV_SCHEDULE_ID,
        dry_run=True,
    )
    args.dataset_root = tmp_path / "dataset"
    args.report = args.dataset_root / "manifest.json"
    with pytest.raises(RuntimeError, match="overlaps protected"):
        cli._safe_report_path(args)


def test_output_cannot_contain_an_input_or_source_tree(tmp_path: Path) -> None:
    args = _args(
        tmp_path,
        stage="development",
        schedule=M42_DEV_SCHEDULE_ID,
        dry_run=True,
    )
    container = tmp_path / "protected-container"
    dataset = container / "dataset"
    dataset.mkdir(parents=True)
    args.dataset_root = dataset
    args.output_root = container
    with pytest.raises(RuntimeError, match="must not overlap"):
        cli._validate_paths(args)

    args.output_root = cli.PROJECT_ROOT / "environment" / "m42-unsafe-output"
    with pytest.raises(RuntimeError, match="must not overlap"):
        cli._validate_paths(args)


def test_command_report_must_remain_outside_evaluation_evidence(tmp_path: Path) -> None:
    args = _args(
        tmp_path,
        stage="development",
        schedule=M42_DEV_SCHEDULE_ID,
        dry_run=True,
    )
    args.report = args.output_root / "command.json"
    with pytest.raises(RuntimeError, match="outside the evaluation evidence output"):
        cli._safe_report_path(args)


def test_linked_input_or_output_ancestor_is_rejected(tmp_path: Path) -> None:
    args = _args(
        tmp_path,
        stage="development",
        schedule=M42_DEV_SCHEDULE_ID,
        dry_run=True,
    )
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(real, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")
    args.output_root = linked / "evaluation"
    with pytest.raises(RuntimeError, match="symlink or junction"):
        cli._validate_paths(args)


def test_linked_evidence_child_is_rejected_without_following_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "evaluation"
    output.mkdir()
    linked_evidence = (output / "evidence").absolute()
    path_type = type(linked_evidence)
    original = path_type.is_symlink
    monkeypatch.setattr(
        path_type,
        "is_symlink",
        lambda self: self == linked_evidence or original(self),
    )
    with pytest.raises(RuntimeError, match="symlink or junction"):
        cli._artifact_paths(output, {"schema_version": "fixture-v0"})


def test_runtime_selection_revalidates_both_selection_locks(tmp_path: Path) -> None:
    horizon = {"selected_value": "5", "kind": "horizon"}
    gripper = {"selected_value": "binary", "kind": "gripper"}
    (tmp_path / "horizon.json").write_text(json.dumps(horizon), encoding="utf-8")
    (tmp_path / "gripper.json").write_text(json.dumps(gripper), encoding="utf-8")
    runtime = SimpleNamespace(
        selection_evidence={
            "horizon_selection": "horizon.json",
            "gripper_selection": "gripper.json",
        },
        horizon_selection_fingerprint=cli.canonical_fingerprint(horizon),
        gripper_selection_fingerprint=cli.canonical_fingerprint(gripper),
        execution_horizon=5,
        gripper_mode="binary",
    )
    cli._verify_runtime_selection_sources(runtime, tmp_path / "runtime.json")

    gripper["selected_value"] = "project"
    (tmp_path / "gripper.json").write_text(json.dumps(gripper), encoding="utf-8")
    with pytest.raises(RuntimeError, match="fingerprint differs"):
        cli._verify_runtime_selection_sources(runtime, tmp_path / "runtime.json")


def test_development_metrics_reject_any_arm_joint_projection() -> None:
    benchmark: dict[str, object] = {
        "aggregate": {
            "episode_count": 1,
            "phase_counts": {"success": 1},
            "action_metrics": {
                "raw_action_metrics_validated": True,
                "runtime_action_bounds_validated": True,
                "arm_action_bounds_validated": True,
                "arm_projected_component_count": 0,
            },
        }
    }
    assert cli._metrics_valid(benchmark) == (True, True, True)
    aggregate = benchmark["aggregate"]
    assert isinstance(aggregate, dict)
    actions = aggregate["action_metrics"]
    assert isinstance(actions, dict)
    actions["arm_projected_component_count"] = 1
    assert cli._metrics_valid(benchmark) == (True, False, True)


def test_evaluation_requires_training_bound_runtime_source_fingerprint() -> None:
    runtime = SimpleNamespace(
        runtime_fingerprint=_digest("a"),
        source_fingerprint=_digest("b"),
    )
    experiment_manifest = SimpleNamespace(fingerprint=_digest("e"))
    identity = SimpleNamespace(
        m3b_export_fingerprint=_digest("c"),
        runtime_selection_fingerprint=runtime.runtime_fingerprint,
        runtime_selection_source_fingerprint=runtime.source_fingerprint,
        experiment_manifest_fingerprint=experiment_manifest.fingerprint,
    )
    evidence = SimpleNamespace(dataset_fingerprint=identity.m3b_export_fingerprint)
    manifest = SimpleNamespace(identity=identity)
    queue = SimpleNamespace(
        runtime_selection_fingerprint=runtime.runtime_fingerprint,
        experiment_manifest_fingerprint=experiment_manifest.fingerprint,
    )
    cli._validate_runtime_training_binding(
        evidence=evidence,
        runtime=runtime,
        experiment_manifest=experiment_manifest,
        manifest=manifest,
        queue=queue,
    )

    runtime.source_fingerprint = _digest("d")
    with pytest.raises(RuntimeError, match="selected runtime"):
        cli._validate_runtime_training_binding(
            evidence=evidence,
            runtime=runtime,
            experiment_manifest=experiment_manifest,
            manifest=manifest,
            queue=queue,
        )


def test_completed_evaluation_artifact_is_content_bound_and_reused(tmp_path: Path) -> None:
    output = tmp_path / "evaluation"
    output.mkdir()
    calls = 0

    def runner() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"value": 7}

    identity = {"schema_version": "fixture-v0", "checkpoint": _digest("a")}
    first, first_path = cli._run_or_reuse_artifact(
        output_root=output,
        identity=identity,
        runner=runner,
    )
    second, second_path = cli._run_or_reuse_artifact(
        output_root=output,
        identity=identity,
        runner=lambda: pytest.fail("completed evidence must be reused"),
    )
    assert calls == 1
    assert first == second
    assert first_path == second_path
    assert first["payload"] == {"value": 7}

    artifact_path = output / first_path / "artifact.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["payload"] = {"value": 8}
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from requested identity"):
        cli._run_or_reuse_artifact(
            output_root=output,
            identity=identity,
            runner=runner,
        )


def test_staging_cleanup_rejects_a_different_owner(tmp_path: Path) -> None:
    output = tmp_path / "evaluation"
    output.mkdir()
    identity = {"schema_version": "fixture-v0", "checkpoint": _digest("b")}
    destination, staging, expected = cli._artifact_paths(output, identity)
    assert not destination.exists()
    staging.mkdir()
    (staging / "owner.json").write_text(
        json.dumps({"identity_fingerprint": _digest("c")}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="another evaluation identity"):
        cli._run_or_reuse_artifact(
            output_root=output,
            identity=identity,
            runner=lambda: {"unexpected": True},
        )
    assert staging.is_dir()
    assert expected != _digest("c")


def _validation_result(index: int) -> TaskTokenValidationResult:
    # The final checkpoint is intentionally best.  No development, final, or
    # test evidence can be represented by this ranking type.
    return TaskTokenValidationResult(
        checkpoint_fingerprint=f"sha256:{index:064x}",
        checkpoint_step=(index + 1) * 5_000,
        validation_schedule_fingerprint=_digest("d"),
        validation_split_digest=_digest("e"),
        episode_count=36,
        success_count=index,
        wrong_object_interaction_count=0,
        target_off_table_count=0,
        timeout_count=36 - index,
        offline_validation_action_loss=1.0 / (index + 1),
    )


def test_checkpoint_selection_is_validation_only_immutable_and_idempotent(
    tmp_path: Path,
) -> None:
    values = tuple(_validation_result(index) for index in range(20))
    path = tmp_path / "selection.json"
    first = cli._selection_lock(path, values)
    second = cli._selection_lock(path, values)
    assert first == second
    assert first.selected_checkpoint_fingerprint == values[-1].checkpoint_fingerprint
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["selection_source"] == "m3b_validation_only"
    assert payload["test_split_accessed"] is False
    assert payload["development_schedule_accessed"] is False
    assert payload["final_schedule_accessed"] is False
    assert payload["locked"] is True


def test_queue_parser_requires_exact_twenty_checkpoint_schedule() -> None:
    payload = {
        "schema_version": "langmani-m42-task-token-validation-queue-v0",
        "run_fingerprint": _digest("1"),
        "m3b_dataset_fingerprint": _digest("2"),
        "split_digest": _digest("3"),
        "train_statistics_fingerprint": _digest("4"),
        "runtime_selection_fingerprint": _digest("5"),
        "experiment_manifest_fingerprint": _digest("9"),
        "task_token_architecture_fingerprint": _digest("6"),
        "validation_schedule_fingerprint": _digest("7"),
        "ordered_validation_episode_indices": list(range(36)),
        "checkpoints": [
            {
                "global_step": step,
                "checkpoint_fingerprint": f"sha256:{step:064x}",
                "checkpoint_relative_path": f"checkpoints/step-{step:08d}",
                "offline_validation_loss": 1.0,
            }
            for step in range(5_000, 100_001, 5_000)
        ],
        "git_commit": "8" * 40,
        "complete": True,
    }
    # The queue fingerprint is part of the canonical payload and therefore is
    # allowed to be computed by the immutable type on first construction.
    payload["queue_fingerprint"] = ""
    queue = cli._queue_from_dict(payload)
    assert queue.complete
    assert len(queue.checkpoints) == 20
    assert queue.checkpoints[-1].global_step == 100_000

    payload["checkpoints"] = payload["checkpoints"][:-1]
    with pytest.raises(RuntimeError, match="all 20 checkpoints"):
        cli._queue_from_dict(payload)


def test_validation_schedule_preserves_all_thirty_six_canonical_mapping_records() -> None:
    records = [
        {
            "episode_index": 288 + index,
            "scene_seed": 10_000 + index // 6,
            "scene_group_id": f"group-{index // 6}",
            "task_id": f"task-{index % 6}",
            "task_spec": {
                "target_object_id": "red_cube",
                "target_bin_id": "left_bin",
                "instruction_template_id": "canonical_v0",
            },
        }
        for index in range(36)
    ]
    identity = SimpleNamespace(
        ordered_validation_episode_indices=tuple(range(288, 324)),
        to_dict=lambda: {"data_contract": {"validation_schedule": records}},
    )
    manifest = SimpleNamespace(identity=identity)
    result = cli._validation_schedule(manifest)
    assert len(result) == 36
    assert all(isinstance(item, dict) for item in result)
    assert list(result) == records
    assert result[0]["scene_group_id"] == "group-0"


def test_validation_rollout_bridge_uses_schedule_records_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = tuple({"episode_index": index} for index in range(36))
    captured: dict[str, object] = {}

    def fake_validation_benchmark(**kwargs: object) -> str:
        captured.update(kwargs)
        return "sentinel"

    monkeypatch.setattr(cli, "run_m42_validation_benchmark", fake_validation_benchmark)
    result = cli._run_validation_checkpoint(
        env=object(),
        checkpoint=object(),
        schedule_records=records,
        validation_schedule_fingerprint=_digest("f"),
        execution_horizon=5,
        gripper_mode=cli.GripperRuntimeMode.PROJECT,
    )
    assert result == "sentinel"
    assert captured["schedule_records"] == records
    assert "schedule" not in captured
    assert captured["maximum_episode_steps"] == 200


def test_development_acceptance_hard_rejects_any_final_access() -> None:
    clean = SimpleNamespace(
        development_schedule_accessed=False,
        final_schedule_accessed=False,
        test_split_accessed=False,
    )
    cli._require_development_without_final_access(
        schedule_id=M42_DEV_SCHEDULE_ID,
        validation_results=(clean,),
        benchmarks=({"schedule_id": M42_DEV_SCHEDULE_ID},),
    )
    with pytest.raises(RuntimeError, match="sealed-final"):
        cli._require_development_without_final_access(
            schedule_id=M42_DEV_SCHEDULE_ID,
            validation_results=(
                SimpleNamespace(
                    development_schedule_accessed=False,
                    final_schedule_accessed=True,
                    test_split_accessed=False,
                ),
            ),
            benchmarks=({"schedule_id": M42_DEV_SCHEDULE_ID},),
        )
    with pytest.raises(RuntimeError, match="sealed-final"):
        cli._require_development_without_final_access(
            schedule_id=M42_DEV_SCHEDULE_ID,
            validation_results=(clean,),
            benchmarks=({"schedule_id": M42_FINAL_SCHEDULE_ID},),
        )


def test_paired_report_has_deterministic_exact_bootstrap_interval() -> None:
    keys = [(f"scene-{index // 6}", f"task-{index % 6}") for index in range(180)]
    left = {key: index % 3 != 0 for index, key in enumerate(keys)}
    right = {key: index % 5 != 0 for index, key in enumerate(reversed(keys))}
    result = cli._paired(left, right)
    reordered = cli._paired(
        dict(reversed(tuple(left.items()))), dict(reversed(tuple(right.items())))
    )
    interval = result["paired_success_rate_difference_interval"]
    assert result == reordered
    assert interval["method"] == "exact_empirical_paired_bootstrap_v0"
    assert interval["confidence_level"] == 0.95
    assert interval["lower"] <= result["paired_success_rate_difference"] <= interval["upper"]

    identical = cli._paired(left, left)
    identical_interval = identical["paired_success_rate_difference_interval"]
    assert identical_interval["lower"] == 0.0
    assert identical_interval["upper"] == 0.0


def _final_identity_fixture(
    *, implementation: str = "e", git_commit: str = "f" * 40
) -> dict[str, object]:
    checkpoints = [_digest(character) for character in "123456ab"]
    return {
        "schema_version": cli.FINAL_IDENTITY_SCHEMA,
        "experiment_manifest_fingerprint": _digest("c"),
        "final_schedule_fingerprint": _digest("d"),
        "implementation_fingerprint": _digest(implementation),
        "evaluation_git_commit": git_commit,
        "prior_m4_evidence_fingerprint": _digest("1"),
        "runtime_selection_fingerprint": _digest("2"),
        "runtime_selection_source_fingerprint": _digest("3"),
        "task_token_run_fingerprint": _digest("4"),
        "task_token_selection_fingerprint": _digest("5"),
        "selected_task_token_checkpoint_fingerprint": checkpoints[-1],
        "evaluated_model_checkpoint_fingerprints": checkpoints,
        "execution_horizon": 5,
        "gripper_mode": "project",
    }


def test_partial_final_attempt_is_preserved_and_next_attempt_is_clean(tmp_path: Path) -> None:
    output = tmp_path / "final"
    output.mkdir()
    identity = _final_identity_fixture()
    first_progress = cli._CommandProgress()
    first, _ = cli._start_final_attempt(
        output_root=output, identity=identity, progress=first_progress
    )
    cli._run_or_reuse_artifact(
        output_root=first,
        identity={"schema_version": "completed-model-v0"},
        runner=lambda: {"model": "complete"},
    )
    partial_identity = {"schema_version": "partial-model-v0"}

    def fail_after_partial_write() -> dict[str, object]:
        _, staging, _ = cli._artifact_paths(first, partial_identity)
        (staging / "partial-episode-evidence.jsonl").write_text("preserved\n", encoding="utf-8")
        raise RuntimeError("fixture interruption")

    with pytest.raises(RuntimeError, match="fixture interruption"):
        cli._run_or_reuse_artifact(
            output_root=first, identity=partial_identity, runner=fail_after_partial_write
        )
    second_progress = cli._CommandProgress()
    second, _ = cli._start_final_attempt(
        output_root=output, identity=identity, progress=second_progress
    )
    incomplete = json.loads((first / "incomplete.json").read_text(encoding="utf-8"))
    assert incomplete["state"] == "invalid_incomplete"
    assert incomplete["completed_artifacts"]
    assert incomplete["partial_staging_paths"]
    assert (first / incomplete["partial_staging_paths"][0]).is_dir()
    assert second != first
    assert not (second / "evidence").exists()


def test_incomplete_final_allows_only_semantically_identical_infrastructure_repair(
    tmp_path: Path,
) -> None:
    output = tmp_path / "final"
    output.mkdir()
    original_identity = _final_identity_fixture()
    original_progress = cli._CommandProgress(final_schedule_accessed=True)
    original, original_fingerprint = cli._start_final_attempt(
        output_root=output, identity=original_identity, progress=original_progress
    )
    cli._record_incomplete_final_attempt(
        attempt_root=original,
        identity_fingerprint=original_fingerprint,
        progress=original_progress,
        error=RuntimeError("infrastructure fixture"),
        reason="sealed_final_command_did_not_complete",
    )

    repaired_identity = _final_identity_fixture(implementation="f", git_commit="e" * 40)
    repaired, _ = cli._start_final_attempt(
        output_root=output,
        identity=repaired_identity,
        progress=cli._CommandProgress(),
    )
    repaired_owner = json.loads((repaired / "attempt.json").read_text(encoding="utf-8"))
    assert repaired_owner["supersedes"][0]["attempt"].endswith("attempt-0001")
    assert repaired_owner["supersedes"][0]["repair_kind"] == "infrastructure_implementation_repair"

    changed_model_identity = dict(repaired_identity)
    changed_model_identity["gripper_mode"] = "binary"
    with pytest.raises(RuntimeError, match="changed schedule, model, selection, or runtime"):
        cli._start_final_attempt(
            output_root=output,
            identity=changed_model_identity,
            progress=cli._CommandProgress(),
        )


def _checkpoint_context(character: str) -> object:
    return SimpleNamespace(
        descriptor=SimpleNamespace(
            fingerprint=_digest(character),
            checkpoint_fingerprint=_digest(character),
        )
    )


def test_final_sensitivity_is_a_content_bound_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "final"
    attempt = output / "final_attempts" / "attempt-0001"
    attempt.mkdir(parents=True)
    calls = 0

    def sensitivity(**_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"scene_count": 2, "task_token_ratio_relative_to_per_task": 0.5}

    monkeypatch.setattr(cli, "_sensitivity", sensitivity)
    per_task = tuple(_checkpoint_context(character) for character in "123456")
    onehot = _checkpoint_context("a")
    task_token = _checkpoint_context("b")
    first, first_reference = cli._run_final_sensitivity_artifact(
        output_root=output,
        attempt_root=attempt,
        final_identity_fingerprint=_digest("c"),
        env=object(),
        scene_seeds=(11, 12),
        per_task=per_task,
        onehot=onehot,
        task_token=task_token,
    )
    second, second_reference = cli._run_final_sensitivity_artifact(
        output_root=output,
        attempt_root=attempt,
        final_identity_fingerprint=_digest("c"),
        env=object(),
        scene_seeds=(11, 12),
        per_task=per_task,
        onehot=onehot,
        task_token=task_token,
    )
    assert calls == 1
    assert first == second
    assert first_reference == second_reference
    assert (output / first_reference["path"] / "complete.json").is_file()

    _, changed_reference = cli._run_final_sensitivity_artifact(
        output_root=output,
        attempt_root=attempt,
        final_identity_fingerprint=_digest("c"),
        env=object(),
        scene_seeds=(11, 13),
        per_task=per_task,
        onehot=onehot,
        task_token=task_token,
    )
    assert calls == 2
    assert changed_reference["path"] != first_reference["path"]


def test_completed_final_is_reused_before_materialization_or_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "final"
    output.mkdir()
    selection = SimpleNamespace(
        selected_checkpoint_fingerprint=_digest("1"), fingerprint=_digest("2")
    )
    manifest = SimpleNamespace(
        identity=SimpleNamespace(
            experiment_manifest_fingerprint=_digest("3"),
            runtime_selection_fingerprint=_digest("4"),
            runtime_selection_source_fingerprint=_digest("5"),
            run_fingerprint=_digest("6"),
        )
    )
    evidence = SimpleNamespace(
        fingerprint=_digest("7"),
        checkpoints=tuple(
            SimpleNamespace(checkpoint_fingerprint=f"sha256:{index:064x}") for index in range(8)
        ),
        mixed_task_onehot=SimpleNamespace(checkpoint_fingerprint=_digest("8")),
    )
    monkeypatch.setattr(
        cli,
        "_validate_final_development_lock",
        lambda **_kwargs: ({"locked": True}, selection, object()),
    )
    monkeypatch.setattr(cli, "_load_reusable_final_completion", lambda **_kwargs: {"fixture": True})

    def reuse_report(**kwargs: object) -> dict[str, object]:
        assert kwargs["completion_reused"] is True
        return {"passed": True, "final_benchmark_completed": True}

    monkeypatch.setattr(cli, "_final_success_report", reuse_report)
    for name in (
        "materialize_schedule",
        "_task_token_context",
        "_m4_contexts",
        "_create_environment",
    ):
        monkeypatch.setattr(
            cli, name, lambda *_args, _name=name, **_kwargs: pytest.fail(f"called {_name}")
        )
    actual = _digest("9")
    args = SimpleNamespace(
        expected_implementation_fingerprint=actual,
        authorize_sealed_final=True,
        runtime_selection=tmp_path / "runtime.json",
    )
    result = cli._execute_final(
        args=args,
        output_root=output,
        report={"implementation_fingerprint": actual},
        evidence=evidence,
        run_root=tmp_path,
        manifest=manifest,
        queue=object(),
        horizon=5,
        gripper=cli.GripperRuntimeMode.PROJECT,
        git=SimpleNamespace(commit="a" * 40, dirty=False),
    )
    assert result["passed"] is True
    assert result["final_benchmark_completed"] is True


def test_reusable_final_completion_is_bound_to_all_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "final"
    attempt = output / "final_attempts" / "attempt-0001"
    attempt.mkdir(parents=True)
    checkpoints = [_digest(character) for character in "123456ab"]
    identity = {
        "schema_version": cli.FINAL_IDENTITY_SCHEMA,
        "experiment_manifest_fingerprint": _digest("c"),
        "final_schedule_fingerprint": _digest("d"),
        "implementation_fingerprint": _digest("e"),
        "evaluation_git_commit": "f" * 40,
        "prior_m4_evidence_fingerprint": _digest("1"),
        "runtime_selection_fingerprint": _digest("2"),
        "runtime_selection_source_fingerprint": _digest("3"),
        "task_token_run_fingerprint": _digest("4"),
        "task_token_selection_fingerprint": _digest("5"),
        "selected_task_token_checkpoint_fingerprint": checkpoints[-1],
        "evaluated_model_checkpoint_fingerprints": checkpoints,
        "execution_horizon": 5,
        "gripper_mode": "project",
    }
    identity_fingerprint = cli.canonical_fingerprint(identity)
    attempt_owner = {
        "schema_version": cli.FINAL_ATTEMPT_SCHEMA,
        "identity": identity,
        "identity_fingerprint": identity_fingerprint,
        "semantic_identity_fingerprint": cli.canonical_fingerprint(
            cli._final_semantic_identity(identity)
        ),
        "supersedes": [],
        "complete": False,
    }
    cli.atomic_write_json(
        attempt / "attempt.json",
        attempt_owner,
        immutable=True,
    )

    def model_artifact(
        checkpoint: str, episode_count: int, label: str
    ) -> tuple[dict[str, object], dict[str, str], dict[str, object]]:
        model_identity = {
            "schema_version": cli.EVIDENCE_SCHEMA,
            "stage": "final",
            "schedule_fingerprint": identity["final_schedule_fingerprint"],
            "implementation_fingerprint": identity["implementation_fingerprint"],
            "experiment_manifest_fingerprint": identity["experiment_manifest_fingerprint"],
            "runtime_selection_fingerprint": identity["runtime_selection_fingerprint"],
            "runtime_selection_source_fingerprint": identity[
                "runtime_selection_source_fingerprint"
            ],
            "checkpoint_fingerprint": checkpoint,
            "execution_horizon": 5,
            "gripper_mode": "project",
            "ordered_episode_indices": list(range(episode_count)),
            "model_label": label,
        }
        benchmark = {"model_label": label, "episodes": [], "aggregate": {}}
        artifact, relative = cli._run_or_reuse_artifact(
            output_root=attempt,
            identity=model_identity,
            runner=lambda: {"benchmark": benchmark},
        )
        reference = cli._attempt_artifact_reference(
            output_root=output,
            attempt_root=attempt,
            relative_path=relative,
            artifact=artifact,
        )
        return artifact, reference, benchmark

    onehot_artifact, onehot_reference, onehot = model_artifact(checkpoints[6], 180, "state_onehot")
    token_artifact, token_reference, task_token = model_artifact(checkpoints[7], 180, "task_token")
    per_task = [
        model_artifact(checkpoint, 30, f"per_task_{index}")
        for index, checkpoint in enumerate(checkpoints[:6])
    ]
    sensitivity = {"scene_count": 30, "task_token_ratio_relative_to_per_task": 0.5}
    sensitivity_artifact, sensitivity_relative = cli._run_or_reuse_artifact(
        output_root=attempt,
        identity={
            "schema_version": cli.FINAL_SENSITIVITY_SCHEMA,
            "final_identity_fingerprint": identity_fingerprint,
            "checkpoint_fingerprints": checkpoints,
        },
        runner=lambda: {"sensitivity": sensitivity},
    )
    sensitivity_reference = cli._attempt_artifact_reference(
        output_root=output,
        attempt_root=attempt,
        relative_path=sensitivity_relative,
        artifact=sensitivity_artifact,
    )
    final_config = {"fixture": "sealed-final"}
    final_config_fingerprint = cli.canonical_fingerprint(final_config)
    final_result = {
        "config_fingerprint": final_config_fingerprint,
        "model_results": {
            "state_onehot": onehot,
            "task_token": task_token,
            "per_task": {
                "benchmarks": [item[2] for item in per_task],
                "aggregate": {},
            },
        },
        "task_sensitivity": sensitivity,
    }
    final_result_fingerprint = cli.canonical_fingerprint(final_result)
    go_no_go = {
        "decision": "remain_in_oracle_control_layer",
        "final_benchmark_fingerprint": final_result_fingerprint,
    }
    completion_path = attempt / "complete.json"
    payload = {
        "schema_version": cli.FINAL_SCHEMA,
        "experiment_manifest_fingerprint": identity["experiment_manifest_fingerprint"],
        "final_identity": identity,
        "final_identity_fingerprint": identity_fingerprint,
        "final_config": final_config,
        "final_config_fingerprint": final_config_fingerprint,
        "final_result": final_result,
        "final_result_fingerprint": final_result_fingerprint,
        "go_no_go": go_no_go,
        "go_no_go_fingerprint": cli.canonical_fingerprint(go_no_go),
        "attempt_completion": str(completion_path.relative_to(output)),
        "evidence_artifacts": {
            "state_onehot": onehot_reference,
            "task_token": token_reference,
            "per_task": [item[1] for item in per_task],
        },
        "sensitivity_artifact": sensitivity_reference,
        "complete": True,
        "final_schedule_accessed": True,
    }
    cli.atomic_write_json(
        completion_path,
        {
            "schema_version": cli.FINAL_ATTEMPT_COMPLETION_SCHEMA,
            "identity_fingerprint": identity_fingerprint,
            "attempt_owner_fingerprint": cli.canonical_fingerprint(attempt_owner),
            "final_comparison_fingerprint": cli.canonical_fingerprint(payload),
            "completed_artifact_count": 9,
            "sensitivity_artifact": sensitivity_reference,
            "complete": True,
        },
        immutable=True,
    )
    cli.atomic_write_json(output / "final_comparison.json", payload, immutable=True)
    assert cli._load_reusable_final_completion(output_root=output, identity=identity) == payload

    sensitivity_path = output / sensitivity_reference["path"] / "artifact.json"
    tampered = json.loads(sensitivity_path.read_text(encoding="utf-8"))
    tampered["payload"]["sensitivity"]["scene_count"] = 29
    sensitivity_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="content differs"):
        cli._load_reusable_final_completion(output_root=output, identity=identity)


def _task_token_failure_benchmark(*, classified: bool) -> dict[str, object]:
    return {
        "episodes": [
            {
                "rollout": {
                    "invalid_action": True,
                    "action_projection_summary": {
                        "any_nonfinite_action": classified,
                        "any_malformed_action": False,
                    },
                }
            }
        ],
        "aggregate": {
            "episode_count": 1,
            "invalid_action_count": 1,
            "phase_counts": {"environment_failure": 1},
            "action_metrics": {
                "nan_count": 2 if classified else 0,
                "inf_count": 1 if classified else 0,
                "malformed_action_count": 0,
                "nonfinite_action_episode_count": 1 if classified else 0,
                "malformed_action_episode_count": 0,
                "raw_action_metrics_validated": not classified,
                "runtime_action_bounds_validated": True,
                "arm_action_bounds_validated": True,
                "arm_projected_component_count": 0,
                "raw_per_action_dimension_violation_counts": [0] * 8,
                "maximum_raw_bound_excess": 0.0,
            },
        },
    }


def test_classified_nonfinite_actions_are_exact_no_go_metrics_not_infrastructure() -> None:
    benchmark = _task_token_failure_benchmark(classified=True)
    assert cli._classified_task_token_action_counts(benchmark) == (2, 1, 0)
    raw_complete, runtime_valid, post_grasp = cli._final_metric_evidence_complete(benchmark)
    assert raw_complete
    assert runtime_valid
    assert post_grasp

    with pytest.raises(RuntimeError, match="unclassified invalid"):
        cli._classified_task_token_action_counts(_task_token_failure_benchmark(classified=False))


def _main_argv(tmp_path: Path) -> list[str]:
    return [
        "--stage",
        "final",
        "--schedule",
        M42_FINAL_SCHEDULE_ID,
        "--output-root",
        str(tmp_path / "final"),
        "--report",
        str(tmp_path / "command.json"),
    ]


def test_final_failure_ignores_stale_attempt_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "final"
    (output / "evidence" / "old").mkdir(parents=True)
    (output / "final_access_attempt.json").write_text("{}\n", encoding="utf-8")
    (output / "evidence" / "old" / "complete.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        cli, "execute", lambda _args: (_ for _ in ()).throw(RuntimeError("preflight"))
    )
    assert cli.main(_main_argv(tmp_path)) == 1
    report = json.loads((tmp_path / "command.json").read_text(encoding="utf-8"))
    assert report["final_schedule_accessed"] is False
    assert report["physical_execution"] is False
    assert report["physical_execution_partial"] is False
    assert report["completed_evaluation_artifact_count"] == 0


def test_final_failure_reports_only_current_command_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_after_progress(args: object) -> dict[str, object]:
        progress = cli._command_progress(args)
        progress.final_schedule_accessed = True
        progress.physical_execution_started = True
        progress.completed_evaluation_artifact_count = 3
        progress.final_attempt_relative_path = "final_attempts/attempt-0002"
        raise RuntimeError("current attempt")

    monkeypatch.setattr(cli, "execute", fail_after_progress)
    assert cli.main(_main_argv(tmp_path)) == 1
    report = json.loads((tmp_path / "command.json").read_text(encoding="utf-8"))
    assert report["final_schedule_accessed"] is True
    assert report["physical_execution_partial"] is True
    assert report["completed_evaluation_artifact_count"] == 3
    assert report["final_attempt"] == "final_attempts/attempt-0002"
