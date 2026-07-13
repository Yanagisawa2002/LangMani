"""Real installed LeRobot 0.6.0 ACT API fixture tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy

from langmani.datasets.lerobot_types import ACTION_FEATURE_KEY, IMAGE_FEATURE_KEY, STATE_FEATURE_KEY
from langmani.policies.act_conditioning import CANONICAL_TASK_IDS
from langmani.policies.act_training import (
    append_task_condition_to_batch,
    build_fixture_act_config,
    build_policy_and_processors,
    make_optimizer,
    prepare_raw_batch,
    train_act,
)
from langmani.policies.act_types import (
    ActDataConfig,
    ActExperimentConfig,
    ActModelConfig,
    ActOptimizationConfig,
    ActVariant,
    ExperimentMode,
)
from scripts import train_act as train_cli


def _stats(state_dimension: int = 9) -> dict[str, dict[str, torch.Tensor]]:
    return {
        IMAGE_FEATURE_KEY: {
            "mean": torch.zeros(3, 1, 1),
            "std": torch.ones(3, 1, 1),
            "min": torch.zeros(3, 1, 1),
            "max": torch.ones(3, 1, 1),
        },
        STATE_FEATURE_KEY: {
            "mean": torch.zeros(state_dimension),
            "std": torch.ones(state_dimension),
            "min": -torch.ones(state_dimension),
            "max": torch.ones(state_dimension),
        },
        ACTION_FEATURE_KEY: {
            "mean": torch.zeros(8),
            "std": torch.ones(8),
            "min": -torch.ones(8),
            "max": torch.ones(8),
        },
    }


def _batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    return {
        IMAGE_FEATURE_KEY: torch.randint(
            0, 256, (1, 3, 256, 256), dtype=torch.uint8, generator=generator
        ),
        STATE_FEATURE_KEY: torch.zeros(1, 9),
        ACTION_FEATURE_KEY: torch.zeros(1, 4, 8),
        "action_is_pad": torch.tensor([[False, False, False, True]]),
        "task": ["this string must never enter ACT"],
    }


@pytest.mark.fixture
@pytest.mark.integration
@pytest.mark.training
def test_real_act_forward_backward_optimizer_and_local_reload(tmp_path: Path) -> None:
    torch.manual_seed(3)
    config = build_fixture_act_config()
    policy, preprocessor, postprocessor = build_policy_and_processors(config, _stats())
    projected = prepare_raw_batch(_batch())
    assert "task" not in projected
    processed = preprocessor(projected)
    loss, loss_dict = policy.forward(processed)
    assert torch.isfinite(loss)
    assert {"l1_loss", "kld_loss"} <= loss_dict.keys()
    loss.backward()
    assert any(parameter.grad is not None for parameter in policy.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in policy.parameters()
        if parameter.grad is not None
    )
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-5, weight_decay=1e-4)
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
    optimizer.step()

    checkpoint = tmp_path / "checkpoint"
    policy.save_pretrained(checkpoint, push_to_hub=False)
    preprocessor.save_pretrained(checkpoint, push_to_hub=False)
    postprocessor.save_pretrained(checkpoint, push_to_hub=False)
    reloaded = ACTPolicy.from_pretrained(
        checkpoint,
        local_files_only=True,
        strict=True,
    )
    loaded_pre, loaded_post = make_pre_post_processors(
        reloaded.config,
        pretrained_path=str(checkpoint),
    )

    policy.eval()
    reloaded.eval()
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    reloaded.reset()
    loaded_pre.reset()
    loaded_post.reset()
    observation = {
        IMAGE_FEATURE_KEY: projected[IMAGE_FEATURE_KEY][0],
        STATE_FEATURE_KEY: projected[STATE_FEATURE_KEY][0],
    }
    expected = postprocessor(policy.select_action(preprocessor(observation)))
    actual = loaded_post(reloaded.select_action(loaded_pre(observation)))
    assert expected.shape == actual.shape == (1, 8)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.fixture
@pytest.mark.integration
@pytest.mark.training
def test_training_metric_serializes_checkpoint_path(tmp_path: Path) -> None:
    model = ActModelConfig(
        chunk_size=4,
        n_action_steps=2,
        dim_model=32,
        n_heads=2,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        latent_dim=8,
        n_vae_encoder_layers=1,
        dropout=0.0,
    )
    optimization = ActOptimizationConfig(
        mixed_precision="none",
        batch_size=1,
        dataloader_workers=0,
        training_steps=1,
        checkpoint_interval=1,
        validation_interval=1,
    )
    experiment = ActExperimentConfig(
        variant=ActVariant.MIXED_UNCONDITIONED,
        data=ActDataConfig(dataset_root="fixture"),
        model=model,
        optimization=optimization,
        mode=ExperimentMode.DRY_RUN,
        device="cpu",
    )
    policy, preprocessor, postprocessor = build_policy_and_processors(
        build_fixture_act_config(), _stats()
    )
    optimizer = make_optimizer(policy, experiment)
    metrics_path = tmp_path / "metrics.jsonl"
    outcome = train_act(
        experiment=experiment,
        policy=policy,
        preprocessor=preprocessor,
        optimizer=optimizer,
        train_batches=(_batch(),),
        metrics_path=metrics_path,
        checkpoint_callback=lambda **_values: "checkpoints/step-00000001-fixture",
        postprocessor=postprocessor,
        initial_examples_processed=10,
    )
    metric = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert outcome.metrics[0].checkpoint_path == "checkpoints/step-00000001-fixture"
    assert metric["checkpoint_path"] == "checkpoints/step-00000001-fixture"
    assert metric["examples_processed"] == outcome.examples_processed == 11


@pytest.mark.fixture
@pytest.mark.integration
def test_resume_accepts_unique_first_promoted_checkpoint_orphan(tmp_path: Path) -> None:
    orphan = "checkpoints/step-00000005-orphan"
    orphan_root = tmp_path / orphan
    orphan_root.mkdir(parents=True)
    (orphan_root / train_cli.CHECKPOINT_COMPLETION_MARKER).write_text("{}", encoding="utf-8")

    assert train_cli._resume_checkpoint_is_recoverable_orphan(
        tmp_path,
        requested_relative_path=orphan,
        declared_relative_paths=(),
    )
    train_cli._validate_recoverable_orphan_step(
        5,
        declared_steps=(),
        checkpoint_interval=5,
        training_steps=12,
    )
    with pytest.raises(RuntimeError, match="next declared checkpoint/final step"):
        train_cli._validate_recoverable_orphan_step(
            10,
            declared_steps=(),
            checkpoint_interval=5,
            training_steps=12,
        )

    second_orphan = tmp_path / "checkpoints" / "step-00000010-second-orphan"
    second_orphan.mkdir()
    (second_orphan / train_cli.CHECKPOINT_COMPLETION_MARKER).write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="sole recoverable promoted orphan"):
        train_cli._resume_checkpoint_is_recoverable_orphan(
            tmp_path,
            requested_relative_path=orphan,
            declared_relative_paths=(),
        )


@pytest.mark.fixture
@pytest.mark.integration
def test_resume_accepts_final_step_as_first_promoted_checkpoint_orphan() -> None:
    train_cli._validate_recoverable_orphan_step(
        3,
        declared_steps=(),
        checkpoint_interval=5,
        training_steps=3,
    )
    with pytest.raises(RuntimeError, match="next declared checkpoint/final step"):
        train_cli._validate_recoverable_orphan_step(
            5,
            declared_steps=(),
            checkpoint_interval=5,
            training_steps=3,
        )


@pytest.mark.fixture
@pytest.mark.integration
def test_make_optimizer_uses_exact_act_parameter_groups_and_learning_rates() -> None:
    policy = ACTPolicy(build_fixture_act_config())
    optimization = ActOptimizationConfig(
        learning_rate=3e-4,
        backbone_learning_rate=7e-5,
        mixed_precision="none",
    )
    experiment = ActExperimentConfig(
        variant=ActVariant.MIXED_UNCONDITIONED,
        data=ActDataConfig(dataset_root="fixture"),
        model=ActModelConfig.for_variant(ActVariant.MIXED_UNCONDITIONED),
        optimization=optimization,
        mode=ExperimentMode.DRY_RUN,
        device="cpu",
    )

    optimizer = make_optimizer(policy, experiment)
    expected_non_backbone = {
        id(parameter)
        for name, parameter in policy.named_parameters()
        if not name.startswith("model.backbone") and parameter.requires_grad
    }
    expected_backbone = {
        id(parameter)
        for name, parameter in policy.named_parameters()
        if name.startswith("model.backbone") and parameter.requires_grad
    }
    actual_groups = [
        {id(parameter) for parameter in group["params"]} for group in optimizer.param_groups
    ]

    assert len(optimizer.param_groups) == 2
    assert expected_non_backbone
    assert expected_backbone
    assert actual_groups == [expected_non_backbone, expected_backbone]
    assert actual_groups[0].isdisjoint(actual_groups[1])
    assert optimizer.param_groups[0]["lr"] == optimization.learning_rate
    assert optimizer.param_groups[1]["lr"] == optimization.backbone_learning_rate
    assert all(
        group["weight_decay"] == optimization.weight_decay for group in optimizer.param_groups
    )


@pytest.mark.fixture
@pytest.mark.integration
def test_policy_reset_discards_queued_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    config = build_fixture_act_config()
    policy, preprocessor, postprocessor = build_policy_and_processors(config, _stats())
    projected = prepare_raw_batch(_batch())
    observation = {
        IMAGE_FEATURE_KEY: projected[IMAGE_FEATURE_KEY][0],
        STATE_FEATURE_KEY: projected[STATE_FEATURE_KEY][0],
    }
    processed = preprocessor(observation)
    calls = 0
    original = policy.predict_action_chunk

    def wrapped(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(batch)

    monkeypatch.setattr(policy, "predict_action_chunk", wrapped)
    policy.select_action(processed)
    policy.select_action(processed)
    assert calls == 1
    policy.reset()
    policy.select_action(processed)
    assert calls == 2
    postprocessor.reset()


@pytest.mark.fixture
@pytest.mark.integration
def test_real_processor_preserves_task_onehot_identity_suffix() -> None:
    config = build_fixture_act_config(state_dimension=15)
    _, preprocessor, _ = build_policy_and_processors(config, _stats(state_dimension=15))
    batch = _batch()
    batch["episode_index"] = torch.tensor([42])
    conditioned = append_task_condition_to_batch(
        batch,
        task_id_by_episode={42: CANONICAL_TASK_IDS[3]},
    )
    processed = preprocessor(prepare_raw_batch(conditioned))
    suffix = processed[STATE_FEATURE_KEY][0, 9:]
    torch.testing.assert_close(
        suffix,
        torch.tensor([0, 0, 0, 1, 0, 0], dtype=torch.float32),
        rtol=0,
        atol=0,
    )
