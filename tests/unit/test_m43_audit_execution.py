from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import UnnormalizerProcessorStep

from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import stable_task_id
from langmani.policies.m43_audit_execution import (
    _action_std_from_context,
    _distance_config,
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
    step = SimpleNamespace(stats={"action": {"std": torch.arange(1, 9, dtype=torch.float32)}})
    context = SimpleNamespace(loaded=SimpleNamespace(postprocessor=SimpleNamespace(steps=[step])))
    result = _action_std_from_context(context)
    assert result == tuple(float(value) for value in range(1, 9))


def test_action_std_accepts_exact_public_lerobot_numpy_float32_record() -> None:
    high_precision = (
        0.1710442851280501,
        0.24617334989705913,
        0.04556946773446931,
        0.4108679144550478,
        0.04032323625685756,
        0.20265467286995067,
        0.23601910385043248,
        0.9999948224584017,
    )
    saved = np.asarray(high_precision, dtype=np.float32)
    step = SimpleNamespace(stats={"action": {"std": saved}})
    context = SimpleNamespace(loaded=SimpleNamespace(postprocessor=SimpleNamespace(steps=[step])))

    result = _action_std_from_context(context)

    assert result == tuple(float(value) for value in saved.tolist())
    assert result != high_precision


def test_installed_lerobot_unnormalizer_restores_public_numpy_float32_stats() -> None:
    step = UnnormalizerProcessorStep(
        features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(8,))},
        norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
    )
    expected = torch.linspace(0.125, 1.0, 8, dtype=torch.float32)
    step.load_state_dict({"action.std": expected})
    context = SimpleNamespace(loaded=SimpleNamespace(postprocessor=SimpleNamespace(steps=[step])))

    assert isinstance(step.stats["action"]["std"], np.ndarray)
    assert step.stats["action"]["std"].dtype == np.float32
    assert step.stats["action"]["std"].shape == (8,)
    assert _action_std_from_context(context) == tuple(float(value) for value in expected.tolist())


def test_distance_config_compares_processors_at_exact_float32_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    high_precision = (
        0.1710442851280501,
        0.24617334989705913,
        0.04556946773446931,
        0.4108679144550478,
        0.04032323625685756,
        0.20265467286995067,
        0.23601910385043248,
        0.9999948224584017,
    )
    saved = np.asarray(high_precision, dtype=np.float32)

    def context(values: np.ndarray) -> object:
        step = SimpleNamespace(stats={"action": {"std": values}})
        return SimpleNamespace(loaded=SimpleNamespace(postprocessor=SimpleNamespace(steps=[step])))

    monkeypatch.setattr(
        "langmani.policies.m43_audit_execution._action_bounds",
        lambda _env: ((-1.0,) * 8, (1.0,) * 8),
    )
    inputs = SimpleNamespace(
        train_action_std=high_precision,
        runtime=SimpleNamespace(execution_horizon=10),
    )
    policies = SimpleNamespace(state_onehot=context(saved), task_token=context(saved.copy()))

    result = _distance_config(env=object(), policies=policies, inputs=inputs)

    assert result.train_action_std == high_precision
    changed = saved.copy()
    changed[0] = np.nextafter(changed[0], np.float32(np.inf))
    mismatched = SimpleNamespace(state_onehot=context(changed), task_token=context(changed.copy()))
    with pytest.raises(M43AuditRuntimeError, match="fingerprint-validated train action std"):
        _distance_config(env=object(), policies=mismatched, inputs=inputs)


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
    numpy_first = SimpleNamespace(stats={"action": {"std": np.ones(8, dtype=np.float32)}})
    numpy_second = SimpleNamespace(stats={"action": {"std": np.full(8, 2.0, dtype=np.float32)}})
    with pytest.raises(M43AuditRuntimeError, match="exactly one"):
        _action_std_from_context(context(numpy_first, numpy_second))
    invalid = SimpleNamespace(stats={"action": {"std": [1.0] * 7 + [0.0]}})
    with pytest.raises(M43AuditRuntimeError, match="finite and positive"):
        _action_std_from_context(context(invalid))
    unsupported = SimpleNamespace(stats={"action": {"std": object()}})
    with pytest.raises(M43AuditRuntimeError, match="unsupported public representation"):
        _action_std_from_context(context(unsupported))
    with pytest.raises(M43AuditRuntimeError, match="exactly one"):
        _action_std_from_context(context(first, unsupported))


@pytest.mark.parametrize(
    "raw",
    (
        np.ones(8, dtype=np.float64),
        np.ones((2, 4), dtype=np.float32),
        np.ones(7, dtype=np.float32),
    ),
)
def test_action_std_rejects_wrong_numpy_dtype_or_shape(raw: np.ndarray) -> None:
    step = SimpleNamespace(stats={"action": {"std": raw}})
    context = SimpleNamespace(loaded=SimpleNamespace(postprocessor=SimpleNamespace(steps=[step])))

    with pytest.raises(M43AuditRuntimeError, match=r"NumPy action std must be float32\[8\]"):
        _action_std_from_context(context)


def test_action_std_rejects_wrong_torch_dtype_or_shape() -> None:
    def context(raw: torch.Tensor) -> object:
        step = SimpleNamespace(stats={"action": {"std": raw}})
        return SimpleNamespace(loaded=SimpleNamespace(postprocessor=SimpleNamespace(steps=[step])))

    with pytest.raises(M43AuditRuntimeError, match=r"Torch action std must be float32\[8\]"):
        _action_std_from_context(context(torch.ones(8, dtype=torch.float64)))
    with pytest.raises(M43AuditRuntimeError, match=r"Torch action std must be float32\[8\]"):
        _action_std_from_context(context(torch.ones((2, 4), dtype=torch.float32)))


@pytest.mark.parametrize("raw", (["1"] * 8, [True] * 8, [[1.0]] * 8))
def test_action_std_rejects_non_numeric_or_nested_sequences(raw: list[object]) -> None:
    step = SimpleNamespace(stats={"action": {"std": raw}})
    context = SimpleNamespace(loaded=SimpleNamespace(postprocessor=SimpleNamespace(steps=[step])))

    with pytest.raises(
        M43AuditRuntimeError, match=r"sequence action std must be numeric float32\[8\]"
    ):
        _action_std_from_context(context)


def test_distance_config_rejects_input_that_overflows_float32() -> None:
    raw = np.ones(8, dtype=np.float32)
    step = SimpleNamespace(stats={"action": {"std": raw}})
    context = SimpleNamespace(loaded=SimpleNamespace(postprocessor=SimpleNamespace(steps=[step])))
    policies = SimpleNamespace(state_onehot=context, task_token=context)
    inputs = SimpleNamespace(
        train_action_std=(1.0,) * 7 + (float(np.finfo(np.float32).max) * 2.0,),
        runtime=SimpleNamespace(execution_horizon=10),
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "langmani.policies.m43_audit_execution._action_bounds",
            lambda _env: ((-1.0,) * 8, (1.0,) * 8),
        )
        with pytest.raises(M43AuditRuntimeError, match="canonical train-only action std"):
            _distance_config(env=object(), policies=policies, inputs=inputs)


@pytest.mark.parametrize("invalid", (float("nan"), float("inf"), float("-inf"), 0.0, -1.0))
def test_action_std_rejects_nonfinite_numpy_values(invalid: float) -> None:
    raw = np.ones(8, dtype=np.float32)
    raw[3] = invalid
    step = SimpleNamespace(stats={"action": {"std": raw}})
    context = SimpleNamespace(loaded=SimpleNamespace(postprocessor=SimpleNamespace(steps=[step])))

    with pytest.raises(M43AuditRuntimeError, match="finite and positive"):
        _action_std_from_context(context)


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
