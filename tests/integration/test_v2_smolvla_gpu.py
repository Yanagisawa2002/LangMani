from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from langmani.datasets.lerobot_types import IMAGE_FEATURE_KEY, STATE_FEATURE_KEY
from langmani.environments.push_specs import PushTaskSpec
from langmani.environments.push_to_region import ENV_ID
from langmani.policies.act_action_bounds import (
    ActionBoundConfig,
    ActionBoundMode,
    BoundedActionEnvPostprocessorV0,
)
from langmani.v2.policy import ObservationBatch, PolicyContext
from langmani.v2.smolvla_adapter import SmolVLAAdapterConfig, SmolVLAPolicyAdapter
from langmani.v2.taxonomy import EvaluationTask, PushTaskInstanceSpec


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.linux
@pytest.mark.target
def test_official_lerobot_smolvla_dependency_is_installed() -> None:
    """Target gate; the full real batch/model/simulator smoke writes external evidence."""

    import lerobot
    import num2words
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

    assert lerobot.__version__ == "0.6.0"
    assert SmolVLAConfig.__name__ == "SmolVLAConfig"
    assert SmolVLAPolicy.__name__ == "SmolVLAPolicy"
    assert callable(make_smolvla_pre_post_processors)
    assert callable(num2words.num2words)


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.linux
@pytest.mark.target
@pytest.mark.training
@pytest.mark.evaluation
def test_real_dataset_smolvla_generation_and_simulator_step() -> None:
    """Target-only real dataset -> official model -> LangMani -> ManiSkill gate."""

    root = os.environ.get("LANGMANI_PHASE2C_TRAIN_ROOT")
    repo_id = os.environ.get("LANGMANI_PHASE2C_TRAIN_REPO_ID")
    adapter_path = os.environ.get("LANGMANI_PHASE2C_ADAPTER_CONFIG")
    if not all((root, repo_id, adapter_path)):
        pytest.skip("real Phase 2C dataset/checkpoint environment variables are not configured")
    import gymnasium as gym
    from lerobot.datasets import LeRobotDataset

    import langmani.environments  # noqa: F401 - registers ENV_ID

    dataset = LeRobotDataset(
        repo_id=str(repo_id),
        root=Path(str(root)),
        video_backend="pyav",
        return_uint8=True,
    )
    sample = dataset[0]
    image = sample[IMAGE_FEATURE_KEY]
    if image.dtype is torch.uint8:
        image = image.to(torch.float32) / 255.0
    observation = ObservationBatch(
        features={
            IMAGE_FEATURE_KEY: image,
            STATE_FEATURE_KEY: sample[STATE_FEATURE_KEY].to(torch.float32),
        }
    )
    task_spec = PushTaskSpec("blue_cube", "left", "standard")
    task = EvaluationTask(
        task_instance=PushTaskInstanceSpec.from_task_spec(task_spec),
        scene_seed=2_031_337,
    )
    adapter = SmolVLAPolicyAdapter(config=SmolVLAAdapterConfig.load(Path(str(adapter_path))))
    environment = gym.make(
        ENV_ID,
        num_envs=1,
        obs_mode="rgb",
        reward_mode="none",
        control_mode="pd_joint_pos",
        render_mode=None,
        sim_backend="physx_cuda",
    )
    try:
        adapter.reset(
            PolicyContext(
                evaluation_task=task,
                evaluation_id="phase2c-real-gpu-integration",
                language_instruction=str(sample["task"]),
            )
        )
        chunk = adapter.act(observation)
        assert chunk.actions.shape[1] == 8
        assert bool(torch.isfinite(chunk.actions).all())
        environment.reset(seed=task.scene_seed, options={"task_spec": task_spec.to_dict()})
        processor = BoundedActionEnvPostprocessorV0.from_environment(
            environment,
            ActionBoundConfig(mode=ActionBoundMode.REJECT),
            expected_action_components=8,
        )
        bounded = processor.process(chunk.actions[0].reshape(1, 8), rollout_step=1)
        action = bounded.executed_action.detach().cpu().numpy()[0]
        step = environment.step(action)
        assert isinstance(step, tuple) and len(step) == 5
    finally:
        adapter.close()
        environment.close()
