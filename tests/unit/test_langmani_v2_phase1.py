from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from langmani.environments.specs import EpisodeSpec, TaskSpec
from langmani.v2 import act_adapter as act_adapter_module
from langmani.v2.act_adapter import ActAdapterConfig, ActAdapterError, ActPerTaskPolicyAdapter
from langmani.v2.evaluator import (
    EpisodeOutcome,
    UnifiedPolicyEvaluator,
    persist_evaluation,
)
from langmani.v2.policy import (
    ActionChunk,
    ObservationBatch,
    PolicyContext,
    PolicyContractError,
    PolicyIdentity,
    PolicyRegistry,
)
from langmani.v2.release import V1ReleaseError, validate_v1_release
from langmani.v2.taxonomy import (
    PICK_AND_PLACE_SKILL,
    EvaluationTask,
    TaskInstanceSpec,
    canonical_pick_and_place_tasks,
    load_task_catalog,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _red_left() -> TaskInstanceSpec:
    return TaskInstanceSpec.from_task_spec(TaskSpec("red_cube", "left_bin", "canonical_v0"))


def test_six_legacy_combinations_are_one_skill_family_and_round_trip() -> None:
    tasks = canonical_pick_and_place_tasks()

    assert len(tasks) == 6
    assert {item.skill_family_id for item in tasks} == {"pick_and_place"}
    assert len({item.canonical_task_id for item in tasks}) == 6
    assert tuple(TaskInstanceSpec.from_dict(item.to_dict()) for item in tasks) == tasks

    catalog = json.loads(
        (PROJECT_ROOT / "configs/langmani_v2/tasks/pick_and_place_v0.json").read_text(
            encoding="utf-8"
        )
    )
    family, loaded = load_task_catalog(catalog)
    assert family == PICK_AND_PLACE_SKILL
    assert loaded == tasks


def test_canonical_runtime_config_contains_no_absolute_paths() -> None:
    path = PROJECT_ROOT / "configs/langmani_v2/policies/act_per_task_red_left_v0.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    config = ActAdapterConfig.from_dict(value)

    assert not Path(config.run_root).is_absolute()
    assert not Path(config.selected_checkpoint_relative_path).is_absolute()
    assert ".." not in Path(config.run_root).parts
    assert all(not Path(item).is_absolute() for item in config.artifact_sha256)

    value["run_root"] = "/root/private/checkpoint"
    with pytest.raises(ActAdapterError, match="repository-relative"):
        ActAdapterConfig.from_dict(value)


def test_action_chunk_requires_one_true_prefix() -> None:
    actions = np.zeros((3, 8), dtype=np.float32)
    chunk = ActionChunk(
        actions=actions,
        valid_mask=np.asarray([True, True, False]),
        horizon=3,
    )

    assert chunk.valid_action_count == 2
    assert len(tuple(chunk.iter_valid_actions())) == 2

    with pytest.raises(PolicyContractError, match="true prefix"):
        ActionChunk(
            actions=actions,
            valid_mask=np.asarray([True, False, True]),
            horizon=3,
        )


class _FixturePolicy:
    def __init__(self, task_id: str) -> None:
        self._identity = PolicyIdentity(
            policy_id="fixture-policy",
            adapter_name="fixture",
            implementation="tests.FixturePolicy",
            checkpoint_identity="fixture-checkpoint",
            compatible_skill_families=("pick_and_place",),
            compatible_task_ids=(task_id,),
        )
        self.reset_count = 0
        self.act_count = 0

    @property
    def identity(self) -> PolicyIdentity:
        return self._identity

    @property
    def runtime_manifest(self) -> dict[str, object]:
        return {"policy_identity": self.identity.to_dict()}

    def reset(self, context: PolicyContext) -> None:
        assert context.evaluation_task.task_instance.canonical_task_id in (
            self.identity.compatible_task_ids
        )
        self.reset_count += 1

    def act(self, observation: ObservationBatch) -> ActionChunk:
        assert set(observation.features) == {"image", "state"}
        self.act_count += 1
        return ActionChunk(
            actions=np.zeros((2, 8), dtype=np.float32),
            valid_mask=np.asarray([True, True]),
            horizon=2,
        )

    def close(self) -> None:
        pass


class _FixtureExtractor:
    def extract(self, observation: object, *, environment: object) -> ObservationBatch:
        del observation, environment
        return ObservationBatch(
            features={
                "image": torch.zeros((3, 256, 256), dtype=torch.float32),
                "state": torch.zeros(9, dtype=torch.float32),
            }
        )


class _FixtureEnvironment:
    def __init__(self) -> None:
        self.unwrapped = self
        self.spec = SimpleNamespace(id="LangMani-PickPlaceByInstruction-v0")
        self.num_envs = 1
        self.control_mode = "pd_joint_pos"
        self.single_action_space = SimpleNamespace(
            low=np.full((8,), -1.0, dtype=np.float32),
            high=np.full((8,), 1.0, dtype=np.float32),
        )
        self._episode_spec: EpisodeSpec | None = None
        self.step_count = 0

    def reset(self, *, seed: int, options: dict[str, object]):
        task_spec = TaskSpec.from_mapping(options["task_spec"])
        self._episode_spec = EpisodeSpec.create(scene_seed=seed, task_spec=task_spec)
        self.step_count = 0
        return {}, {"success": False}

    def get_episode_specs(self):
        assert self._episode_spec is not None
        return (self._episode_spec,)

    def get_policy_rollout_evaluation(self):
        return {
            "success": self.step_count >= 2,
            "target_off_table": False,
            "wrong_object_is_grasped": False,
            "wrong_object_in_target_bin": False,
        }

    def step(self, action: np.ndarray):
        assert action.shape == (8,)
        self.step_count += 1
        return {}, 0.0, False, False, self.get_policy_rollout_evaluation()


def test_unified_evaluator_is_policy_agnostic_and_persists_schema(tmp_path: Path) -> None:
    task_instance = _red_left()
    policy = _FixturePolicy(task_instance.canonical_task_id)
    evaluator = UnifiedPolicyEvaluator(
        environment=_FixtureEnvironment(),
        observation_extractor=_FixtureExtractor(),
    )

    result = evaluator.run_episode(
        policy=policy,
        task=EvaluationTask(task_instance=task_instance, scene_seed=17),
        evaluation_id="fixture:17",
    )

    assert result.outcome is EpisodeOutcome.SUCCESS
    assert result.success
    assert result.episode_length == 2
    assert result.policy_query_count == 1
    assert result.action_projection_count == 0
    assert result.task_identity == task_instance.canonical_task_id
    assert result.skill_family == "pick_and_place"
    assert policy.reset_count == policy.act_count == 1

    summary = persist_evaluation(
        output_root=tmp_path / "evaluation",
        runtime_manifest=policy.runtime_manifest,
        results=(result,),
    )
    record = json.loads((tmp_path / "evaluation/episodes.jsonl").read_text(encoding="utf-8"))
    assert summary["success_count"] == 1
    assert record["checkpoint_identity"] == "fixture-checkpoint"
    assert record["policy_inference_latency_ms"]["p95"] is not None


def test_unified_evaluator_rejects_incompatible_policy_before_reset() -> None:
    task = _red_left()
    policy = _FixturePolicy("langmani-pick-place-task-v0:blue_cube:right_bin:canonical_v0")
    evaluator = UnifiedPolicyEvaluator(
        environment=_FixtureEnvironment(),
        observation_extractor=_FixtureExtractor(),
    )

    with pytest.raises(Exception, match="incompatible"):
        evaluator.run_episode(
            policy=policy,
            task=EvaluationTask(task_instance=task, scene_seed=0),
            evaluation_id="incompatible",
        )
    assert policy.reset_count == 0


def test_policy_registry_loads_only_registered_adapter() -> None:
    registry = PolicyRegistry()
    task_id = _red_left().canonical_task_id
    registry.register("fixture", lambda **_kwargs: _FixturePolicy(task_id))

    adapter = registry.create("fixture")
    assert adapter.identity.adapter_name == "fixture"
    with pytest.raises(PolicyContractError, match="unregistered"):
        registry.create("smolvla")


class _Component:
    def __init__(self, fn=None) -> None:
        self.reset_count = 0
        self.fn = fn

    def reset(self) -> None:
        self.reset_count += 1

    def __call__(self, value):
        return self.fn(value) if self.fn is not None else value


class _ACTPolicy(_Component):
    config = SimpleNamespace()

    def to(self, _device):
        return self

    def eval(self):
        return self

    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        assert set(batch) == {"observation.images.base_camera", "observation.state"}
        return torch.zeros((1, 50, 8), dtype=torch.float32)


def test_act_adapter_reset_and_real_chunk_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "outputs/models/act/run"
    checkpoint = run_root / "checkpoints/selected"
    hashes: dict[str, str] = {}
    for relative in (
        "pretrained_model/model.safetensors",
        "pretrained_model/policy_preprocessor_step_3_normalizer_processor.safetensors",
        "pretrained_model/policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ):
        path = checkpoint / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
        hashes[relative] = hashlib.sha256(relative.encode()).hexdigest()
    (run_root / "run_manifest.json").write_text("{}", encoding="utf-8")
    config_path = tmp_path / "configs/policy.json"
    config_path.parent.mkdir(parents=True)
    task_id = _red_left().canonical_task_id
    config = ActAdapterConfig(
        policy_id="fixture-act",
        adapter_name="act_per_task_v0",
        run_root="outputs/models/act/run",
        selected_checkpoint_relative_path="checkpoints/selected",
        expected_run_fingerprint="sha256:" + "1" * 64,
        expected_checkpoint_fingerprint="sha256:" + "2" * 64,
        expected_dataset_fingerprint="sha256:" + "3" * 64,
        expected_task_id=task_id,
        execution_horizon=10,
        action_bound_mode="project",
        device="cpu",
        dtype="float32",
        artifact_sha256=hashes,
    )
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    policy = _ACTPolicy()
    preprocessor = _Component()
    postprocessor = _Component()
    context = SimpleNamespace(
        descriptor=SimpleNamespace(checkpoint_relative_path="checkpoints/selected"),
        loaded=SimpleNamespace(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
        ),
    )
    monkeypatch.setattr(act_adapter_module, "load_m4_checkpoint_context", lambda *_a, **_k: context)
    fake_identity = SimpleNamespace(
        model_config={"normalization_mapping": {"ACTION": "MEAN_STD"}},
        git_commit="a" * 40,
        m3b_export_fingerprint="sha256:" + "3" * 64,
        train_statistics_fingerprint="sha256:" + "4" * 64,
    )
    fake_manifest = SimpleNamespace(
        identity=fake_identity,
        config=SimpleNamespace(to_dict=lambda: {"variant": "per_task"}),
    )
    monkeypatch.setattr(
        act_adapter_module.ActExperimentManifest,
        "from_dict",
        lambda _value: fake_manifest,
    )

    adapter = ActPerTaskPolicyAdapter(
        config=config,
        project_root=tmp_path,
        config_path=config_path,
    )
    adapter.reset(
        PolicyContext(
            evaluation_task=EvaluationTask(task_instance=_red_left(), scene_seed=1),
            evaluation_id="act-fixture",
        )
    )
    chunk = adapter.act(
        ObservationBatch(
            features={
                "observation.images.base_camera": torch.zeros((3, 256, 256), dtype=torch.float32),
                "observation.state": torch.zeros(9, dtype=torch.float32),
            }
        )
    )

    assert chunk.actions.shape == (10, 8)
    assert chunk.valid_action_count == 10
    assert policy.reset_count == preprocessor.reset_count == postprocessor.reset_count == 1
    assert adapter.runtime_manifest["action_shape"] == [8]
    assert adapter.runtime_manifest["normalization"] == {"ACTION": "MEAN_STD"}


def test_v1_release_manifest_validates_and_detects_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("frozen\n", encoding="utf-8")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "langmani-v1-release-manifest-v0",
                "release": {"tag": "v1.0.0", "commit": "a" * 40},
                "artifact_identities": {"dataset": "sha256:" + "b" * 64},
                "frozen_files": [{"path": "evidence.txt", "sha256": digest}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("langmani.v2.release._git", lambda *_args: "a" * 40)

    result = validate_v1_release(project_root=tmp_path, manifest_path=manifest)
    assert result.passed

    evidence.write_text("changed\n", encoding="utf-8")
    with pytest.raises(V1ReleaseError, match="evidence changed"):
        validate_v1_release(project_root=tmp_path, manifest_path=manifest)
