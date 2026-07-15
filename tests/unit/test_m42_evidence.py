"""M4.2 shared provenance-manifest and historical M4.1 processor tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import langmani.policies.m42_evidence as evidence_module
from langmani.policies.act_action_bounds import (
    ActionBoundConfig,
    ActionBoundMode,
    EvaluationRuntimeIdentity,
    EvaluationRuntimeManifest,
)
from langmani.policies.act_checkpoint import ActCheckpointError
from langmani.policies.m42_evidence import (
    M42EvidenceError,
    m41_runtime_processor_fingerprint,
    validate_m42_experiment_manifest,
)
from langmani.policies.m42_schedule import (
    M42_DEV_SCHEDULE_FINGERPRINT,
    M42_FINAL_SCHEDULE_FINGERPRINT,
)
from langmani.policies.m42_types import M42ExperimentManifest, M42Stage


def _digest(character: str) -> str:
    return f"sha256:{character * 64}"


def _runtime(index: int, *, upper_gripper: float = 1.0) -> EvaluationRuntimeManifest:
    identity = EvaluationRuntimeIdentity(
        checkpoint_fingerprint=f"sha256:{index + 1:064x}",
        policy_preprocessor_fingerprint=_digest("a"),
        policy_postprocessor_fingerprint=_digest("b"),
        action_bound_config=ActionBoundConfig(mode=ActionBoundMode.PROJECT),
        environment_id="LangMani-PickPlaceByInstruction-v0",
        action_space_contract={
            "bounds_source": "environment_action_space",
            "single_action_shape": [8],
            "lower_dtype": "float32",
            "upper_dtype": "float32",
            "lower_bounds": [-1.0] * 8,
            "upper_bounds": [1.0] * 7 + [upper_gripper],
        },
        task_conditioning_mapping_version="CanonicalTaskOneHotV0",
        rollout_config={"maximum_episode_steps": 200},
        code_git_commit="c" * 40,
    )
    return EvaluationRuntimeManifest(
        identity=identity,
        checkpoint_model_reload_validated=True,
        policy_processor_reload_validated=True,
        action_bound_processor_reload_validated=True,
        deterministic_raw_action_matched=True,
        raw_action_match_tolerance=1e-6,
    )


def _manifest(checkpoints: tuple[str, ...], processor: str) -> M42ExperimentManifest:
    return M42ExperimentManifest(
        stage=M42Stage.RUNTIME_ABLATION,
        implementation_git_commit="d" * 40,
        m3b_dataset_fingerprint=_digest("e"),
        prior_m4_verification_fingerprint=_digest("f"),
        schedule_fingerprints={
            "m42_dev_v0": M42_DEV_SCHEDULE_FINGERPRINT,
            "m42_final_v0": M42_FINAL_SCHEDULE_FINGERPRINT,
        },
        m4_checkpoint_fingerprints=checkpoints,
        m41_runtime_processor_fingerprint=processor,
        runtime_selection_fingerprints={
            "execution_horizon": _digest("1"),
            "gripper_runtime": _digest("2"),
        },
        task_token_experiment_fingerprint=None,
        artifact_paths={
            "runtime_selection": "runtime_selection.json",
            "horizon_selection": "horizon_selection.json",
            "gripper_selection": "gripper_selection.json",
            "post_grasp_analysis": "post_grasp_analysis.json",
        },
        completed=True,
    )


def test_m41_processor_fingerprint_uses_actual_config_and_action_space() -> None:
    runtimes = tuple(_runtime(index) for index in range(8))
    fingerprint = m41_runtime_processor_fingerprint(runtimes)
    assert fingerprint.startswith("sha256:")
    changed = (*runtimes[:-1], _runtime(7, upper_gripper=0.9))
    with pytest.raises(M42EvidenceError, match="disagree"):
        m41_runtime_processor_fingerprint(changed)


def test_experiment_manifest_round_trip_binds_all_eight_historical_controls() -> None:
    runtimes = tuple(_runtime(index) for index in range(8))
    checkpoints = tuple(item.identity.checkpoint_fingerprint for item in runtimes)
    processor = m41_runtime_processor_fingerprint(runtimes)
    manifest = _manifest(checkpoints, processor)
    assert M42ExperimentManifest.from_dict(manifest.to_dict()) == manifest
    evidence = SimpleNamespace(
        dataset_fingerprint=manifest.m3b_dataset_fingerprint,
        verification_fingerprint=manifest.prior_m4_verification_fingerprint,
        m41_runtime_processor_fingerprint=processor,
        checkpoints=tuple(
            SimpleNamespace(checkpoint_fingerprint=fingerprint) for fingerprint in checkpoints
        ),
    )
    validate_m42_experiment_manifest(manifest, evidence)
    evidence.m41_runtime_processor_fingerprint = _digest("3")
    with pytest.raises(M42EvidenceError, match="differs"):
        validate_m42_experiment_manifest(manifest, evidence)


def _disk_checkpoint(root: Path, index: int, *, materialize: bool = True) -> SimpleNamespace:
    run_root = root / f"run-{index}"
    relative_path = f"checkpoints/step-00100000-{index:012x}"
    checkpoint_path = run_root / Path(relative_path)
    if materialize:
        checkpoint_path.mkdir(parents=True)
    record = SimpleNamespace(
        checkpoint_fingerprint=f"sha256:{index + 1:064x}",
        relative_path=relative_path,
    )
    return SimpleNamespace(
        run_root=run_root,
        checkpoint_path=checkpoint_path,
        selected_checkpoint=record,
        checkpoint_fingerprint=record.checkpoint_fingerprint,
        manifest=SimpleNamespace(identity=SimpleNamespace(run_fingerprint=f"run-{index}")),
    )


def test_all_eight_selected_checkpoints_are_strictly_reloaded_sequentially(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoints = tuple(_disk_checkpoint(tmp_path, index) for index in range(8))
    calls: list[str] = []

    def _load(**kwargs: object) -> SimpleNamespace:
        calls.append(str(kwargs["checkpoint_relative_path"]))
        checkpoint = checkpoints[len(calls) - 1]
        return SimpleNamespace(record=checkpoint.selected_checkpoint)

    monkeypatch.setattr(evidence_module, "load_act_checkpoint", _load)
    evidence_module._validate_all_frozen_checkpoints(checkpoints)

    assert calls == [item.selected_checkpoint.relative_path for item in checkpoints]
    assert calls[-2:] == [
        checkpoints[6].selected_checkpoint.relative_path,
        checkpoints[7].selected_checkpoint.relative_path,
    ]


def test_missing_eighth_selected_checkpoint_is_rejected_before_acceptance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoints = tuple(
        _disk_checkpoint(tmp_path, index, materialize=index != 7) for index in range(8)
    )
    calls: list[str] = []

    def _load(**kwargs: object) -> SimpleNamespace:
        calls.append(str(kwargs["checkpoint_relative_path"]))
        return SimpleNamespace(record=checkpoints[len(calls) - 1].selected_checkpoint)

    monkeypatch.setattr(evidence_module, "load_act_checkpoint", _load)
    with pytest.raises(M42EvidenceError, match="selected checkpoint"):
        evidence_module._validate_all_frozen_checkpoints(checkpoints)

    assert len(calls) == 7


def test_corrupt_eighth_selected_checkpoint_is_rejected_by_strict_reload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoints = tuple(_disk_checkpoint(tmp_path, index) for index in range(8))
    calls: list[str] = []

    def _load(**kwargs: object) -> SimpleNamespace:
        calls.append(str(kwargs["checkpoint_relative_path"]))
        if len(calls) == 8:
            raise ActCheckpointError("checkpoint artifact integrity mismatch")
        return SimpleNamespace(record=checkpoints[len(calls) - 1].selected_checkpoint)

    monkeypatch.setattr(evidence_module, "load_act_checkpoint", _load)
    with pytest.raises(M42EvidenceError, match="atomic marker, hash, or strict reload"):
        evidence_module._validate_all_frozen_checkpoints(checkpoints)

    assert len(calls) == 8


def test_strict_reload_record_must_equal_the_frozen_selected_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoints = tuple(_disk_checkpoint(tmp_path, index) for index in range(8))

    def _load(**kwargs: object) -> SimpleNamespace:
        del kwargs
        return SimpleNamespace(
            record=SimpleNamespace(
                checkpoint_fingerprint="sha256:" + "f" * 64,
                relative_path="checkpoints/other",
            )
        )

    monkeypatch.setattr(evidence_module, "load_act_checkpoint", _load)
    with pytest.raises(M42EvidenceError, match="record differs"):
        evidence_module._validate_all_frozen_checkpoints(checkpoints)


def _directory_symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")


def test_evidence_reader_rejects_a_linked_ancestor_before_resolution(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "verification.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    linked = tmp_path / "linked-diagnostics"
    _directory_symlink_or_skip(linked, outside)

    with pytest.raises(M42EvidenceError, match="traverses a symlink or junction"):
        evidence_module._read_object(linked / "verification.json")


def test_run_discovery_rejects_a_linked_child_run(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    outside_run = tmp_path / "outside-run"
    outside_run.mkdir()
    (outside_run / "run_manifest.json").write_text("{}", encoding="utf-8")
    _directory_symlink_or_skip(checkpoint_root / "linked-run", outside_run)

    with pytest.raises(M42EvidenceError, match="linked child run"):
        evidence_module._discover_runs(checkpoint_root)
