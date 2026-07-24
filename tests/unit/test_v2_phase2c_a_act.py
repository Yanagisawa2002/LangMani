"""Targeted ACT/adapter tests; skipped when LeRobot 0.6 is unavailable."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

pytest.importorskip("lerobot")

from langmani.datasets.lerobot_types import (  # noqa: E402
    ACTION_FEATURE_KEY,
    IMAGE_FEATURE_KEY,
    STATE_FEATURE_KEY,
)
from langmani.v2.phase2c_a import (  # noqa: E402
    TASK_IDS,
    ModelKind,
    Phase2CARunIdentity,
    primary_optimization_config,
)
from langmani.v2.phase2c_a_act import (  # noqa: E402
    TASK_INDEX_KEY,
    TASK_TOKEN_FEATURE_KEY,
    Phase2CAActError,
    _repo_id,
    build_phase2c_a_act_config,
    load_processor_statistics,
    masked_l1_loss,
    prepare_policy_batch,
    validate_shared_act_config,
)
from langmani.v2.phase2c_a_adapter import (  # noqa: E402
    Phase2CAActPolicyAdapter,
    Phase2CAAdapterError,
)
from langmani.v2.policy import ObservationBatch, PolicyContext  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2b6_v2"


def test_repo_id_comes_from_accepted_split_manifest(tmp_path: Path) -> None:
    manifest = {
        "repo_id": "langmani/official-pickcube-v2-train",
        "task_id": "PickCube-v1",
        "primary_split": "train",
    }
    (tmp_path / "langmani_phase2b6_split_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    assert (
        _repo_id(tmp_path, task_id="PickCube-v1", split="train")
        == "langmani/official-pickcube-v2-train"
    )
    with pytest.raises(Phase2CAActError, match="accepted LeRobot split identity"):
        _repo_id(tmp_path, task_id="PushCube-v1", split="train")


def _raw_batch(batch_size: int = 2) -> dict[str, object]:
    action = torch.zeros(batch_size, 16, 8, dtype=torch.float32)
    padding = torch.zeros(batch_size, 16, dtype=torch.bool)
    padding[:, -3:] = True
    return {
        IMAGE_FEATURE_KEY: torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8),
        STATE_FEATURE_KEY: torch.zeros(batch_size, 9, dtype=torch.float32),
        ACTION_FEATURE_KEY: action,
        "action_is_pad": padding,
        TASK_INDEX_KEY: torch.tensor([0, 2], dtype=torch.int64),
        "object_pose": torch.ones(batch_size, 7),
        "reward": torch.ones(batch_size),
    }


def test_masked_l1_loss_gives_padding_zero_weight() -> None:
    expected = torch.zeros(2, 16, 8)
    predicted = torch.ones_like(expected)
    padding = torch.zeros(2, 16, dtype=torch.bool)
    padding[:, -4:] = True
    baseline = masked_l1_loss(predicted, expected, padding)
    predicted[:, -4:] = 1_000_000
    assert masked_l1_loss(predicted, expected, padding) == baseline
    assert float(baseline) == 1.0


def test_policy_batch_projects_no_privileged_fields_and_keeps_padding() -> None:
    projected = prepare_policy_batch(_raw_batch(), model_kind=ModelKind.SHARED)
    assert set(projected) == {
        IMAGE_FEATURE_KEY,
        STATE_FEATURE_KEY,
        ACTION_FEATURE_KEY,
        "action_is_pad",
        TASK_TOKEN_FEATURE_KEY,
    }
    assert projected[TASK_TOKEN_FEATURE_KEY].tolist() == [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    assert projected["action_is_pad"].dtype is torch.bool


def test_shared_config_uses_public_three_way_env_token() -> None:
    optimization = primary_optimization_config(ModelKind.SHARED, batch_size=128)
    config = build_phase2c_a_act_config(
        ModelKind.SHARED,
        device="cpu",
        use_amp=False,
        optimization=optimization,
    )
    validate_shared_act_config(config)
    assert config.input_features[TASK_TOKEN_FEATURE_KEY].shape == (3,)
    assert config.input_features[STATE_FEATURE_KEY].shape == (9,)


def test_accepted_normalization_loader_uses_push_std_floor() -> None:
    loaded = load_processor_statistics(
        EVIDENCE / "dataset_statistics.json",
        EVIDENCE / "normalization_manifest.json",
        model_kind=ModelKind.PUSH,
    )
    action_std = loaded.tensors[ACTION_FEATURE_KEY]["std"]
    assert float(action_std[-1]) == pytest.approx(1e-8)
    assert loaded.manifest["validation_or_test_rows_used"] is False


class _Processor:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def __call__(self, value: object) -> object:
        return value


class _Policy:
    def __init__(self, *, nonfinite: bool = False) -> None:
        self.reset_calls = 0
        self.nonfinite = nonfinite
        self.config = object()

    def to(self, device: object) -> _Policy:
        del device
        return self

    def eval(self) -> _Policy:
        return self

    def reset(self) -> None:
        self.reset_calls += 1

    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        del batch
        result = torch.arange(16 * 8, dtype=torch.float32).reshape(1, 16, 8) / 100
        if self.nonfinite:
            result[0, 0, 0] = torch.nan
        return result


def _adapter(*, horizon: int = 4, nonfinite: bool = False) -> Phase2CAActPolicyAdapter:
    return Phase2CAActPolicyAdapter(
        model_kind=ModelKind.PICK,
        policy=_Policy(nonfinite=nonfinite),  # type: ignore[arg-type]
        preprocessor=_Processor(),  # type: ignore[arg-type]
        postprocessor=_Processor(),  # type: ignore[arg-type]
        checkpoint_identity="sha256:" + "1" * 64,
        run_fingerprint="sha256:" + "2" * 64,
        execution_horizon=horizon,
        device="cpu",
    )


def _observation() -> ObservationBatch:
    return ObservationBatch(
        features={
            IMAGE_FEATURE_KEY: torch.zeros(3, 256, 256),
            STATE_FEATURE_KEY: torch.zeros(9),
        }
    )


def test_adapter_reset_clears_query_state_and_execution_queue() -> None:
    adapter = _adapter(horizon=4)
    context = PolicyContext(
        evaluation_task=None,
        evaluation_id="episode-1",
        task_id="PickCube-v1",
    )
    adapter.reset(context)
    chunk = adapter.act(_observation())
    assert chunk.valid_action_count == 4
    assert adapter.query_count == 1
    adapter.reset(
        PolicyContext(
            evaluation_task=None,
            evaluation_id="episode-2",
            task_id="PickCube-v1",
        )
    )
    assert adapter.query_count == 0
    assert adapter.reset_count == 2


def test_adapter_rejects_nonfinite_outputs() -> None:
    adapter = _adapter(nonfinite=True)
    adapter.reset(
        PolicyContext(
            evaluation_task=None,
            evaluation_id="episode",
            task_id="PickCube-v1",
        )
    )
    with pytest.raises(Phase2CAAdapterError, match="nonfinite"):
        adapter.act(_observation())


def test_adapter_rejects_wrong_task_and_non_registered_horizon() -> None:
    with pytest.raises(Phase2CAAdapterError, match="execution horizon"):
        _adapter(horizon=2)
    adapter = _adapter()
    with pytest.raises(Phase2CAAdapterError, match="incompatible"):
        adapter.reset(
            PolicyContext(
                evaluation_task=None,
                evaluation_id="wrong",
                task_id="PushCube-v1",
            )
        )


def test_run_identity_binds_role_and_canonical_package() -> None:
    optimization = primary_optimization_config(ModelKind.PICK, batch_size=128)
    value = Phase2CARunIdentity(
        model_kind=ModelKind.PICK,
        run_role="gpu_smoke",
        model_config={"chunk_size": 16},
        optimization_config=optimization.to_dict(),
        sampler_fingerprint="sha256:" + "3" * 64,
        train_statistics_fingerprint="sha256:" + "4" * 64,
        m3b_split_manifest_digest="sha256:" + "5" * 64,
        git_commit="6" * 40,
        runtime_fingerprint="sha256:" + "7" * 64,
    )
    roundtrip = Phase2CARunIdentity.from_dict(value.to_dict())
    assert roundtrip == value
    assert roundtrip.run_role == "gpu_smoke"


def test_shared_task_mapping_contains_exactly_three_official_tasks() -> None:
    assert TASK_IDS == ("PickCube-v1", "StackCube-v1", "PushCube-v1")
    raw = _raw_batch()
    raw[TASK_INDEX_KEY] = torch.tensor([1, 1])
    result = prepare_policy_batch(raw, model_kind=ModelKind.SHARED)
    assert result[TASK_TOKEN_FEATURE_KEY].shape == (2, 3)


def test_adapter_runtime_manifest_never_authorizes_projection() -> None:
    manifest: Any = _adapter().runtime_manifest
    assert manifest["action_projection"] is False
    assert manifest["action_clipping"] is False
    assert manifest["execution_horizon"] == 4
