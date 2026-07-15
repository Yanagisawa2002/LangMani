from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from langmani.policies.act_runtime import GitState
from langmani.policies.m42_schedule import (
    M42_DEV_SCHEDULE_ID,
    load_locked_schedule,
    materialize_schedule,
)
from langmani.policies.m42_training import load_runtime_selection
from langmani.policies.m42_types import GripperRuntimeMode, RuntimeAblationResult


def _load_cli() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_m42_runtime_ablation.py"
    spec = importlib.util.spec_from_file_location("run_m42_runtime_ablation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cli = _load_cli()


def _digest(character: str) -> str:
    return f"sha256:{character * 64}"


def _checkpoint(*, task_id: str | None, marker: str) -> SimpleNamespace:
    return SimpleNamespace(
        task_id=task_id,
        run_root=Path(f"run-{marker}").resolve(),
        run_fingerprint=_digest(marker),
        checkpoint_fingerprint=_digest("f" if marker == "c" else "1"),
    )


def _evidence() -> SimpleNamespace:
    representative_task_id = materialize_schedule(load_locked_schedule(M42_DEV_SCHEDULE_ID))[
        2
    ].task_id
    mixed = _checkpoint(task_id=None, marker="c")
    representative = _checkpoint(task_id=representative_task_id, marker="d")
    return SimpleNamespace(
        fingerprint=_digest("a"),
        dataset_fingerprint=_digest("b"),
        verification_fingerprint=_digest("2"),
        m41_runtime_processor_fingerprint=_digest("3"),
        checkpoints=tuple(
            SimpleNamespace(checkpoint_fingerprint=_digest(value)) for value in "2456ab"
        )
        + (representative, mixed),
        mixed_task_onehot=mixed,
        representative_per_task=representative,
    )


def _args(tmp_path: Path, *, dry_run: bool) -> object:
    for name in ("dataset", "checkpoints", "m4"):
        (tmp_path / name).mkdir(exist_ok=True)
    values = [
        "--dataset-root",
        str(tmp_path / "dataset"),
        "--checkpoint-root",
        str(tmp_path / "checkpoints"),
        "--m4-diagnostics-root",
        str(tmp_path / "m4"),
        "--output-root",
        str(tmp_path / "runtime"),
        "--report",
        str(tmp_path / "command.json"),
    ]
    if dry_run:
        values.append("--dry-run")
    return cli.parse_args(values)


def _patch_inputs(monkeypatch: pytest.MonkeyPatch, evidence: object) -> None:
    monkeypatch.setattr(
        cli,
        "inspect_git_state",
        lambda _root: GitState(
            commit="e" * 40,
            dirty=False,
            changed_paths=(),
            baseline_tracked=True,
        ),
    )
    monkeypatch.setattr(cli, "load_prior_m4_evidence", lambda **_kwargs: evidence)
    monkeypatch.setattr(
        cli,
        "_task_token_fair_comparison",
        lambda _evidence: SimpleNamespace(
            contract_fingerprint=_digest("7"),
            to_dict=lambda: {
                "schema_version": "langmani-m42-task-token-fair-comparison-v0",
                "contract_fingerprint": _digest("7"),
            },
        ),
    )


def _runtime_result(
    *,
    config_fingerprint: str,
    schedule_fingerprint: str,
    model_label: str,
    task_id: str | None,
    horizon: int,
    mode: GripperRuntimeMode,
    episode_count: int,
) -> RuntimeAblationResult:
    mixed = task_id is None
    base_successes = {10: 20, 5: 40, 1: 30}[horizon] if mixed else {10: 5, 5: 8, 1: 6}[horizon]
    successes = min(episode_count, base_successes + (4 if mode is GripperRuntimeMode.BINARY else 0))
    marker = f"{horizon}{mode.value}{model_label}{task_id}"
    from langmani.policies.m42_training import canonical_fingerprint

    return RuntimeAblationResult(
        config_fingerprint=config_fingerprint,
        schedule_id=M42_DEV_SCHEDULE_ID,
        schedule_fingerprint=schedule_fingerprint,
        model_label=model_label,
        task_id=task_id,
        execution_horizon=horizon,
        gripper_mode=mode,
        episode_count=episode_count,
        successes=successes,
        post_grasp_timeouts=2 if mode is GripperRuntimeMode.PROJECT else 1,
        wrong_object_interactions=0,
        wrong_object_grasp_count=0,
        wrong_object_in_target_bin_count=0,
        target_in_wrong_bin_count=0,
        target_off_table_count=0,
        invalid_action_count=0,
        successful_episode_steps=tuple(100 + index for index in range(successes)),
        policy_query_count=episode_count * (200 // horizon),
        release_sign_transitions=episode_count,
        grasp_sign_transitions=episode_count,
        unnecessary_gripper_sign_transitions=0,
        action_metrics={
            "total_policy_actions": episode_count * 100,
            "raw_out_of_bounds_action_count": 2,
            "raw_out_of_bounds_component_count": 2,
            "maximum_raw_bound_excess": 0.2,
            "raw_action_metrics_validated": True,
            "runtime_projected_action_count": 2,
            "runtime_projected_component_count": 2,
            "arm_projected_component_count": 0,
            "arm_action_bounds_validated": True,
            "maximum_runtime_projection_correction": 0.2,
            "runtime_action_bounds_validated": True,
        },
        latency_metrics={"mean_policy_query_s": 0.01},
        report_fingerprint=canonical_fingerprint({"marker": marker}),
    )


def test_cli_rejects_final_schedule_before_any_materialization() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--schedule", "m42_final_v0"])


def test_runtime_metrics_reject_any_arm_joint_projection() -> None:
    result = _runtime_result(
        config_fingerprint=_digest("1"),
        schedule_fingerprint=_digest("2"),
        model_label="ACT-Mixed-TaskOneHot",
        task_id=None,
        horizon=5,
        mode=GripperRuntimeMode.PROJECT,
        episode_count=72,
    )
    assert cli._metrics_present(result) == (True, True)
    changed = dict(result.action_metrics)
    changed["arm_projected_component_count"] = 1
    changed["arm_action_bounds_validated"] = False
    assert cli._metrics_present(replace(result, action_metrics=changed)) == (True, False)


def test_plan_declares_logical_counts_and_content_bound_project_reuse() -> None:
    episodes = materialize_schedule(load_locked_schedule(M42_DEV_SCHEDULE_ID))
    plan = cli._plan(episodes)
    assert plan["horizon_physical_episode_count"] == 252
    assert plan["gripper_logical_episode_count"] == 168
    assert plan["gripper_additional_physical_episode_count"] == 84
    assert plan["total_logical_episode_count"] == 420
    assert plan["total_unique_physical_episode_count"] == 336
    assert plan["final_schedule_materialized"] is False


def test_benchmark_identity_uses_integrity_loaded_descriptor() -> None:
    episodes = materialize_schedule(load_locked_schedule(M42_DEV_SCHEDULE_ID))
    checkpoint = SimpleNamespace(
        descriptor=SimpleNamespace(
            run_fingerprint=_digest("1"), checkpoint_fingerprint=_digest("2")
        )
    )
    identity = cli._benchmark_identity(
        checkpoint=checkpoint,
        model_label="ACT-Mixed-TaskOneHot",
        task_id=None,
        schedule_fingerprint=load_locked_schedule(M42_DEV_SCHEDULE_ID).schedule_fingerprint,
        episodes=episodes,
        horizon=5,
        gripper_mode=GripperRuntimeMode.PROJECT,
        config_fingerprint=_digest("3"),
    )
    assert identity["run_fingerprint"] == _digest("1")
    assert identity["checkpoint_fingerprint"] == _digest("2")
    assert identity["ordered_episode_indices"] == list(range(72))


def test_dry_run_never_creates_environment_or_locks_a_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _evidence()
    _patch_inputs(monkeypatch, evidence)
    monkeypatch.setattr(
        cli,
        "_create_environment",
        lambda: pytest.fail("dry-run must not construct ManiSkill"),
    )
    args = _args(tmp_path, dry_run=True)
    report = cli.execute(args)
    assert report["passed"] is True
    assert report["physical_execution"] is False
    assert report["horizon_ablation_completed"] is False
    assert report["gripper_selection_locked"] is False
    assert report["final_schedule_accessed"] is False
    assert not (tmp_path / "runtime").exists()
    assert not (tmp_path / "runtime" / "runtime_selection.json").exists()
    persisted = json.loads((tmp_path / "command.json").read_text(encoding="utf-8"))
    assert persisted == report


def test_output_must_not_overlap_dataset(tmp_path: Path) -> None:
    args = _args(tmp_path, dry_run=True)
    args.output_root = tmp_path / "dataset" / "runtime"
    with pytest.raises(RuntimeError, match="must not overlap"):
        cli._validate_paths(args)


def test_output_must_not_contain_an_input_or_source_tree(tmp_path: Path) -> None:
    args = _args(tmp_path, dry_run=True)
    container = tmp_path / "protected-container"
    dataset = container / "dataset"
    dataset.mkdir(parents=True)
    args.dataset_root = dataset
    args.output_root = container
    with pytest.raises(RuntimeError, match="must not overlap"):
        cli._validate_paths(args)

    args.output_root = cli.PROJECT_ROOT / "src" / "m42-unsafe-output"
    with pytest.raises(RuntimeError, match="must not overlap"):
        cli._validate_paths(args)


def test_command_report_must_remain_outside_evidence_output(tmp_path: Path) -> None:
    args = _args(tmp_path, dry_run=True)
    args.report = args.output_root / "command.json"
    with pytest.raises(RuntimeError, match="outside the evidence output"):
        cli._validate_paths(args)


def test_linked_output_ancestor_is_rejected(tmp_path: Path) -> None:
    args = _args(tmp_path, dry_run=True)
    real = tmp_path / "real-output-parent"
    real.mkdir()
    linked = tmp_path / "linked-output-parent"
    try:
        linked.symlink_to(real, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")
    args.output_root = linked / "runtime"
    with pytest.raises(RuntimeError, match="symlink or junction"):
        cli._validate_paths(args)


def test_staging_cleanup_requires_matching_owner_fingerprint(tmp_path: Path) -> None:
    evidence_root = tmp_path / "evidence"
    staging = evidence_root / ".staging-token"
    staging.mkdir(parents=True)
    (staging / cli.BENCHMARK_OWNER_FILE).write_text(
        json.dumps({"identity_fingerprint": _digest("1")}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="another runtime identity"):
        cli._clean_matching_staging(
            staging,
            identity_fingerprint=_digest("2"),
            evidence_root=evidence_root,
        )
    assert staging.is_dir()


def test_physical_orchestration_locks_selection_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _evidence()
    _patch_inputs(monkeypatch, evidence)

    class FakeEnv:
        closed = False

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(cli, "_create_environment", FakeEnv)
    monkeypatch.setattr(cli, "_load_contexts", lambda _evidence: (object(), object()))

    def fake_run_or_reuse(**values: object):
        selected_episodes = values["episodes"]
        assert isinstance(selected_episodes, tuple)
        result = _runtime_result(
            config_fingerprint=values["config_fingerprint"],
            schedule_fingerprint=values["schedule_fingerprint"],
            model_label=values["model_label"],
            task_id=values["task_id"],
            horizon=values["horizon"],
            mode=values["gripper_mode"],
            episode_count=len(selected_episodes),
        )
        artifact = {
            "benchmark_fingerprint": result.report_fingerprint,
            "benchmark": {
                "post_grasp_summary": {"episode_count": len(selected_episodes)},
                "post_grasp_failure_records": [],
            },
        }
        relative = f"evidence/{result.report_fingerprint.removeprefix('sha256:')}"
        return result, artifact, relative

    monkeypatch.setattr(cli, "_run_or_reuse", fake_run_or_reuse)
    monkeypatch.setattr(
        cli,
        "_post_grasp_payload",
        lambda artifacts, *, schedule_fingerprint: {
            "schema_version": cli.POST_GRASP_SCHEMA,
            "schedule_fingerprint": schedule_fingerprint,
            "unique_physical_benchmark_count": len(artifacts),
            "complete": True,
            "final_schedule_accessed": False,
        },
    )
    args = _args(tmp_path, dry_run=False)
    first = cli.execute(args)
    second = cli.execute(args)
    assert first == second
    assert first["passed"] is True
    assert first["horizon_ablation_completed"] is True
    assert first["gripper_ablation_completed"] is True
    assert first["unique_physical_episode_count"] == 336
    assert first["unique_physical_benchmark_count"] == 8
    runtime = load_runtime_selection(tmp_path / "runtime" / "runtime_selection.json")
    assert runtime.execution_horizon == 5
    assert runtime.gripper_mode in {"project", "binary"}
    assert runtime.final_schedule_accessed is False
