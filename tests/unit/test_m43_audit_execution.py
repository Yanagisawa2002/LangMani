from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import stable_task_id
from langmani.policies.m43_audit_execution import (
    _action_std_from_context,
    _rollout_distribution,
    _rollout_metrics,
    _validation_frame_rows,
    run_semantic_audit,
)
from langmani.policies.m43_audit_inputs import ValidationEpisodeInput
from langmani.policies.m43_audit_runtime import M43AuditRuntimeError
from langmani.policies.m43_types import SemanticFailureClass


def _episode(phase: str) -> dict[str, object]:
    return {"post_grasp_diagnostics": {"phase": phase}}


def test_rollout_distribution_maps_real_m42_phases_without_fabricated_traces() -> None:
    episodes = [_episode("success") for _ in range(70)]
    episodes.extend((_episode("wrong_object_interaction"), _episode("target_grasped_not_lifted")))
    result = _rollout_distribution({"episodes": episodes})
    assert set(result) == {value.value for value in SemanticFailureClass}
    assert result[SemanticFailureClass.SUCCESS.value] == 70
    assert result[SemanticFailureClass.WRONG_OBJECT_INTERACTION.value] == 1
    assert result[SemanticFailureClass.TARGET_GRASPED_NOT_LIFTED.value] == 1
    assert sum(result.values()) == 72


def test_rollout_distribution_requires_complete_development_schedule() -> None:
    with pytest.raises(M43AuditRuntimeError, match="cover 72 episodes"):
        _rollout_distribution({"episodes": [_episode("success")]})
    with pytest.raises(M43AuditRuntimeError, match="unknown post-grasp phase"):
        _rollout_distribution({"episodes": [_episode("unknown")] * 72})


def test_rollout_distribution_flattens_six_historical_per_task_benchmarks() -> None:
    benchmark = {
        "benchmarks": [{"episodes": [_episode("success") for _ in range(12)]} for _ in range(6)]
    }
    result = _rollout_distribution(benchmark)
    assert result[SemanticFailureClass.SUCCESS.value] == 72
    assert sum(result.values()) == 72


def test_rollout_metrics_preserve_action_contract_and_recover_per_task_strict_counts() -> None:
    child_aggregate = {
        "strict_unprojected_success_count": 1,
        "strict_runtime_unprojected_success_count": 2,
        "strict_raw_unmodified_success_count": 3,
    }
    result = _rollout_metrics(
        {
            "aggregate": {
                "episode_count": 72,
                "successes": 60,
                "success_rate": 60 / 72,
                "action_metrics": {
                    "raw_action_metrics_validated": False,
                    "runtime_action_bounds_validated": True,
                    "per_action_dimension_projection_counts": [0] * 8,
                    "raw_per_action_dimension_violation_counts": [1] * 8,
                },
            },
            "benchmarks": [{"aggregate": child_aggregate} for _ in range(6)],
        }
    )
    assert result["strict_unprojected_success_count"] == 6
    assert result["strict_runtime_unprojected_success_count"] == 12
    assert result["strict_raw_unmodified_success_count"] == 18
    assert result["action_metrics"] == {
        "raw_action_metrics_validated": False,
        "runtime_action_bounds_validated": True,
        "per_action_dimension_projection_counts": [0] * 8,
        "raw_per_action_dimension_violation_counts": [1] * 8,
    }


def test_action_std_comes_from_one_saved_train_only_postprocessor_record() -> None:
    step = SimpleNamespace(stats={"action": {"std": torch.arange(1, 9)}})
    context = SimpleNamespace(loaded=SimpleNamespace(postprocessor=SimpleNamespace(steps=[step])))
    result = _action_std_from_context(context)
    assert result == tuple(float(value) for value in range(1, 9))


def test_action_std_rejects_missing_duplicate_or_nonpositive_values() -> None:
    def context(*steps: object) -> object:
        return SimpleNamespace(
            loaded=SimpleNamespace(postprocessor=SimpleNamespace(steps=list(steps)))
        )

    with pytest.raises(M43AuditRuntimeError, match="exactly one"):
        _action_std_from_context(context())
    first = SimpleNamespace(stats={"action": {"std": [1.0] * 8}})
    second = SimpleNamespace(stats={"action": {"std": [2.0] * 8}})
    with pytest.raises(M43AuditRuntimeError, match="exactly one"):
        _action_std_from_context(context(first, second))
    invalid = SimpleNamespace(stats={"action": {"std": [1.0] * 7 + [0.0]}})
    with pytest.raises(M43AuditRuntimeError, match="finite and positive"):
        _action_std_from_context(context(invalid))


def _validation_schedule() -> tuple[ValidationEpisodeInput, ...]:
    values: list[ValidationEpisodeInput] = []
    for scene_index in range(6):
        for task_spec in CANONICAL_TASK_SPECS:
            values.append(
                ValidationEpisodeInput(
                    episode_index=288 + len(values),
                    scene_seed=10_000 + scene_index,
                    scene_group_id=f"group-{scene_index}",
                    task_id=stable_task_id(task_spec),
                    task_spec=task_spec,
                )
            )
    return tuple(values)


def test_validation_reader_predicate_loads_only_six_source_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = {
        "codebase_version": "v3.0",
        "fps": 20,
        "video_path": ("videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"),
        "features": {
            "observation.images.base_camera": {"dtype": "video"},
            "observation.state": {"dtype": "float32", "shape": [9]},
        },
    }
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    episode_dir = tmp_path / "meta" / "episodes" / "chunk-000"
    episode_dir.mkdir(parents=True)
    image_key = "observation.images.base_camera"
    pq.write_table(
        pa.table(
            {
                "episode_index": list(range(360)),
                f"videos/{image_key}/chunk_index": [0] * 360,
                f"videos/{image_key}/file_index": [0] * 360,
                f"videos/{image_key}/from_timestamp": [0.0] * 360,
            }
        ),
        episode_dir / "file-000.parquet",
    )
    data_dir = tmp_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": list(range(360)),
                "frame_index": [0] * 360,
                "timestamp": [0.0] * 360,
                "observation.state": [[float(index)] * 9 for index in range(360)],
            }
        ),
        data_dir / "file-000.parquet",
    )
    video = tmp_path / "videos" / image_key / "chunk-000" / "file-000.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fixture")
    decoded: list[Path] = []

    def fake_decode(path: Path, *_args: object, **_kwargs: object) -> torch.Tensor:
        decoded.append(Path(path))
        return torch.zeros((1, 3, 256, 256), dtype=torch.uint8)

    monkeypatch.setattr("lerobot.datasets.video_utils.decode_video_frames", fake_decode)
    frames = _validation_frame_rows(tmp_path, _validation_schedule())

    assert tuple(frames) == (288, 294, 300, 306, 312, 318)
    assert len(decoded) == 6
    assert frames[318][1].tolist() == [318.0] * 9


def test_real_audit_passes_mode_to_restricted_loader_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ExpectedStop(RuntimeError):
        pass

    monkeypatch.setattr(
        "langmani.policies.m43_audit_execution.inspect_git_state",
        lambda _root: SimpleNamespace(baseline_tracked=True, dirty=False, commit="a" * 40),
    )

    def restricted(**kwargs: object) -> object:
        assert kwargs["mode"] == "validation"
        return SimpleNamespace(
            dataset_root=tmp_path,
            dataset_fingerprint="sha256:" + "1" * 64,
            split_digest="sha256:" + "2" * 64,
        )

    monkeypatch.setattr(
        "langmani.policies.m43_audit_execution.load_restricted_audit_inputs", restricted
    )

    def stop_before_checkpoint_load(*_args: object, **_kwargs: object) -> object:
        raise ExpectedStop

    monkeypatch.setattr(
        "langmani.policies.m43_audit_execution._load_policy_contexts", stop_before_checkpoint_load
    )
    args = Namespace(
        mode="validation",
        device="cpu",
        dataset_root=tmp_path,
        m4_checkpoint_root=tmp_path,
        task_token_checkpoint_root=tmp_path,
        m42_diagnostics_root=tmp_path,
        runtime_selection=tmp_path,
    )
    with pytest.raises(ExpectedStop):
        run_semantic_audit(args)
