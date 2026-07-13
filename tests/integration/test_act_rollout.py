"""Fixture closed-loop tests for the ACT-to-M1 inference boundary."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from langmani.datasets.lerobot_types import IMAGE_FEATURE_KEY, STATE_FEATURE_KEY
from langmani.datasets.policy_state import PANDA_POLICY_STATE_COMPONENTS
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.pick_place_by_instruction import ENV_ID
from langmani.environments.specs import EpisodeSpec
from langmani.policies.act_action_bounds import ActionBoundConfig, ActionBoundMode
from langmani.policies.act_rollout import ActManiSkillRolloutAdapter, RolloutContractError
from langmani.policies.act_types import ActVariant, EvaluationSplit, RolloutStatus

SHA = "sha256:" + "a" * 64


class _Joint:
    def __init__(self, name: str) -> None:
        self.name = name


class _Robot:
    def get_active_joints(self) -> list[_Joint]:
        return [_Joint(name) for name in PANDA_POLICY_STATE_COMPONENTS]

    def get_qpos(self) -> torch.Tensor:
        return torch.zeros((1, 9), dtype=torch.float32)


class _Processor:
    def __init__(self, *, add_batch: bool = False) -> None:
        self.reset_count = 0
        self.last_input: object = None
        self.add_batch = add_batch

    def reset(self) -> None:
        self.reset_count += 1

    def __call__(self, value: object) -> object:
        self.last_input = value
        if self.add_batch and isinstance(value, dict):
            return {
                key: item.unsqueeze(0) if isinstance(item, torch.Tensor) else item
                for key, item in value.items()
            }
        return value


class _Policy:
    def __init__(self, action: torch.Tensor | None = None) -> None:
        self.action = action if action is not None else torch.zeros((1, 8), dtype=torch.float32)
        self.reset_count = 0
        self.inputs: list[dict[str, torch.Tensor]] = []

    def reset(self) -> None:
        self.reset_count += 1

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.inputs.append(batch)
        return self.action.clone()


class _Env:
    def __init__(self) -> None:
        self.unwrapped = self
        self.num_envs = 1
        self.control_mode = "pd_joint_pos"
        self.control_freq = 20
        self.spec = SimpleNamespace(id=ENV_ID)
        self.agent = SimpleNamespace(robot=_Robot())
        self.single_action_space = SimpleNamespace(
            low=np.full(8, -1.0, dtype=np.float32),
            high=np.full(8, 1.0, dtype=np.float32),
        )
        self.step_calls = 0
        self.last_action: np.ndarray | None = None
        self._episode: EpisodeSpec | None = None

    def get_policy_rollout_evaluation(self) -> dict[str, torch.Tensor]:
        success = self.step_calls == 2
        return {
            "success": torch.tensor([success]),
            "target_is_grasped": torch.tensor([self.step_calls == 1]),
            "wrong_object_is_grasped": torch.tensor([False]),
            "target_off_table": torch.tensor([False]),
        }

    def _observation(self) -> dict[str, object]:
        return {
            "sensor_data": {
                "base_camera": {"rgb": torch.zeros((1, 256, 256, 3), dtype=torch.uint8)}
            }
        }

    def reset(self, *, seed: int, options: dict[str, object]) -> tuple[object, dict[str, object]]:
        from langmani.environments.specs import TaskSpec

        task_spec = TaskSpec.from_mapping(options["task_spec"])
        self._episode = EpisodeSpec.create(scene_seed=seed, task_spec=task_spec)
        self.step_calls = 0
        return self._observation(), {"success": torch.tensor([False])}

    def get_episode_specs(self) -> tuple[EpisodeSpec, ...]:
        assert self._episode is not None
        return (self._episode,)

    def step(self, action: np.ndarray) -> tuple[object, float, object, object, dict[str, object]]:
        assert action.shape == (8,)
        self.last_action = np.array(action, copy=True)
        self.step_calls += 1
        success = self.step_calls == 2
        info = {
            "success": torch.tensor([success]),
            "target_is_grasped": torch.tensor([self.step_calls == 1]),
            "wrong_object_is_grasped": torch.tensor([False]),
            "target_off_table": torch.tensor([False]),
        }
        return (
            self._observation(),
            0.0,
            torch.tensor([success]),
            torch.tensor([False]),
            info,
        )


class _TimeLimitLikeWrapper:
    """Gymnasium-style wrapper that does not proxy arbitrary attributes directly."""

    def __init__(self, env: _Env) -> None:
        self.env = env
        self.unwrapped = env
        self.spec = env.spec

    def get_wrapper_attr(self, name: str) -> object:
        return getattr(self.env, name)

    def reset(self, *, seed: int, options: dict[str, object]) -> tuple[object, dict[str, object]]:
        return self.env.reset(seed=seed, options=options)

    def step(self, action: np.ndarray) -> tuple[object, float, object, object, dict[str, object]]:
        return self.env.step(action)


def _adapter(
    env: object,
    policy: _Policy,
    *,
    action_bound_mode: ActionBoundMode = ActionBoundMode.REJECT,
) -> tuple[ActManiSkillRolloutAdapter, _Processor, _Processor]:
    pre = _Processor(add_batch=True)
    post = _Processor()
    adapter = ActManiSkillRolloutAdapter(
        env=env,
        policy=policy,
        preprocessor=pre,
        postprocessor=post,
        variant=ActVariant.MIXED_UNCONDITIONED,
        run_fingerprint=SHA,
        checkpoint_fingerprint=SHA,
        schedule_digest=SHA,
        runtime_fingerprint=SHA,
        action_bound_config=ActionBoundConfig(mode=action_bound_mode),
    )
    return adapter, pre, post


@pytest.mark.fixture
@pytest.mark.evaluation
def test_rollout_requires_explicit_environment_contract() -> None:
    env = _Env()
    env.control_freq = 10
    with pytest.raises(RolloutContractError, match="20 Hz"):
        _adapter(env, _Policy())
    env.control_freq = 20
    env.control_mode = None
    with pytest.raises(RolloutContractError, match="pd_joint_pos"):
        _adapter(env, _Policy())


@pytest.mark.fixture
@pytest.mark.evaluation
def test_rollout_resets_all_state_and_executes_raw_actions() -> None:
    env = _Env()
    policy = _Policy()
    adapter, pre, post = _adapter(env, policy)
    result = adapter.run_episode(
        evaluation_id="fixture-evaluation",
        split=EvaluationSplit.VALIDATION,
        scene_seed=3,
        task_spec=CANONICAL_TASK_SPECS[0],
    )
    assert result.status is RolloutStatus.SUCCESS
    assert result.episode_steps == 2
    assert result.target_grasped_any
    assert result.time_to_first_target_grasp_s == 1 / 20
    assert result.time_to_release_s == 2 / 20
    assert policy.reset_count == pre.reset_count == post.reset_count == 1
    assert env.step_calls == 2
    assert len(policy.inputs) == 2
    assert set(policy.inputs[0]) == {IMAGE_FEATURE_KEY, STATE_FEATURE_KEY}
    assert policy.inputs[0][IMAGE_FEATURE_KEY].shape == (1, 3, 256, 256)
    assert policy.inputs[0][IMAGE_FEATURE_KEY].dtype == torch.float32
    assert policy.inputs[0][STATE_FEATURE_KEY].shape == (1, 9)
    assert result.task_success
    assert result.strict_unprojected_success
    assert result.action_projection_summary["projected_action_count"] == 0


@pytest.mark.fixture
@pytest.mark.evaluation
def test_rollout_resolves_single_action_space_through_timelimit_wrapper() -> None:
    base = _Env()
    wrapper = _TimeLimitLikeWrapper(base)
    adapter, _, _ = _adapter(wrapper, _Policy())
    result = adapter.run_episode(
        evaluation_id="fixture-wrapped-environment",
        split=EvaluationSplit.VALIDATION,
        scene_seed=3,
        task_spec=CANONICAL_TASK_SPECS[0],
    )
    assert result.status is RolloutStatus.SUCCESS
    assert base.step_calls == 2


@pytest.mark.fixture
@pytest.mark.evaluation
def test_out_of_bounds_action_is_rejected_without_environment_step() -> None:
    env = _Env()
    policy = _Policy(torch.full((1, 8), 2.0, dtype=torch.float32))
    adapter, _, _ = _adapter(env, policy)
    result = adapter.run_episode(
        evaluation_id="fixture-invalid-action",
        split=EvaluationSplit.VALIDATION,
        scene_seed=4,
        task_spec=CANONICAL_TASK_SPECS[1],
    )
    assert result.status is RolloutStatus.INVALID_ACTION
    assert result.invalid_action
    assert env.step_calls == 0
    assert result.action_projection_summary["maximum_bound_excess"] == 1.0


@pytest.mark.fixture
@pytest.mark.evaluation
def test_project_mode_sends_the_explicitly_bounded_action_to_environment() -> None:
    env = _Env()
    action = torch.zeros((1, 8), dtype=torch.float32)
    action[0, -1] = 1.0508
    adapter, _, _ = _adapter(env, _Policy(action), action_bound_mode=ActionBoundMode.PROJECT)
    result = adapter.run_episode(
        evaluation_id="fixture-projected-action",
        split=EvaluationSplit.VALIDATION,
        scene_seed=4,
        task_spec=CANONICAL_TASK_SPECS[1],
    )
    assert result.success
    assert env.step_calls == 2
    assert env.last_action is not None and env.last_action[-1] == 1.0
    assert result.action_projection_summary["projected_action_count"] == 2
    assert not result.strict_unprojected_success


@pytest.mark.fixture
@pytest.mark.evaluation
def test_task_onehot_adapter_uses_command_task_id_only() -> None:
    env = _Env()
    policy = _Policy()
    pre = _Processor(add_batch=True)
    post = _Processor()
    seen_task_ids: list[str] = []

    def conditioner(state: np.ndarray, task_id: str) -> np.ndarray:
        seen_task_ids.append(task_id)
        return np.concatenate((state, np.asarray([1, 0, 0, 0, 0, 0], dtype=np.float32)))

    adapter = ActManiSkillRolloutAdapter(
        env=env,
        policy=policy,
        preprocessor=pre,
        postprocessor=post,
        variant=ActVariant.MIXED_TASK_ONEHOT,
        run_fingerprint=SHA,
        checkpoint_fingerprint=SHA,
        schedule_digest=SHA,
        runtime_fingerprint=SHA,
        action_bound_config=ActionBoundConfig(mode=ActionBoundMode.REJECT),
        task_conditioner=conditioner,
    )
    result = adapter.run_episode(
        evaluation_id="fixture-onehot",
        split=EvaluationSplit.VALIDATION,
        scene_seed=5,
        task_spec=CANONICAL_TASK_SPECS[0],
    )
    assert result.success
    assert policy.inputs[0][STATE_FEATURE_KEY].shape == (1, 15)
    assert seen_task_ids == [result.task_id, result.task_id]
