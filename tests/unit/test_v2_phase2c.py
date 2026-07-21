from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from langmani.datasets.lerobot_types import (
    ACTION_FEATURE_KEY,
    IMAGE_FEATURE_KEY,
    STATE_FEATURE_KEY,
)
from langmani.environments.push_specs import PushTaskSpec
from langmani.v2.phase2c import (
    DATASET_SPLITS,
    TRAIN_EPISODES,
    Phase2CContractError,
    build_train_view_manifest,
    compute_train_only_normalization,
    feature_mapping_manifest,
    map_smolvla_observation,
    normalization_round_trip,
    require_finite_manifest,
    write_json_once,
)
from langmani.v2.phase2c_baselines import HoldPositionPolicyAdapter
from langmani.v2.phase2c_evaluation import (
    FrozenPolicyManifest,
    development_competence_gate,
    quality_level,
    summarize_sealed_results,
    wilson_interval,
)
from langmani.v2.phase2c_offline import OfflineMetricAccumulator, select_offline_checkpoints
from langmani.v2.phase2c_schedule import build_phase2c_schedules
from langmani.v2.policy import ObservationBatch, PolicyContext, PolicyContractError
from langmani.v2.registry import default_policy_registry
from langmani.v2.smolvla_adapter import (
    OFFICIAL_BASE_MODEL,
    SMOLVLA_ADAPTER_NAME,
    SmolVLAAdapterConfig,
    SmolVLAAdapterError,
    SmolVLAPolicyAdapter,
)
from langmani.v2.taxonomy import EvaluationTask, PushTaskInstanceSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_training_launcher():
    path = PROJECT_ROOT / "environment" / "launch_v2_smolvla_training.py"
    spec = importlib.util.spec_from_file_location("phase2c_training_launcher_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _input_manifest() -> dict[str, object]:
    identities = {
        split: [f"{split}-{index}" for index in range(count)]
        for split, count in {
            "train": 203,
            "validation": 77,
            "test_unseen_scene": 26,
            "test_unseen_language": 40,
            "test_hard": 23,
            "test_visual_shift": 28,
        }.items()
    }
    return {
        "passed": True,
        "semantic_sha256": "sha256:" + "a" * 64,
        "counts": {
            split: {"episodes": len(values), "frames": 26_968 if split == "train" else 1}
            for split, values in identities.items()
        },
        "split_episode_ids": identities,
        "dataset_roots": {split: f"/external/{split}" for split in DATASET_SPLITS},
    }


def _sample(index: int) -> dict[str, object]:
    return {
        STATE_FEATURE_KEY: np.asarray([index, 1, 2, 3, 4, 5, 6, 7, 0], dtype=np.float32),
        ACTION_FEATURE_KEY: np.asarray([index, 1, 2, 3, 4, 5, 6, 0], dtype=np.float32),
    }


def _context() -> PolicyContext:
    task = PushTaskInstanceSpec.from_task_spec(PushTaskSpec("blue_cube", "left", "standard"))
    return PolicyContext(
        evaluation_task=EvaluationTask(task_instance=task, scene_seed=42),
        evaluation_id="phase2c-fixture",
    )


def _observation(*, image: torch.Tensor | None = None) -> ObservationBatch:
    return ObservationBatch(
        features={
            IMAGE_FEATURE_KEY: (
                image if image is not None else torch.zeros((3, 256, 256), dtype=torch.float32)
            ),
            STATE_FEATURE_KEY: torch.arange(9, dtype=torch.float32),
        }
    )


def _config(tmp_path: Path, *, horizon: int = 4) -> SmolVLAAdapterConfig:
    return SmolVLAAdapterConfig(
        policy_id="smolvla-push-fixture",
        checkpoint_path=str(tmp_path / "checkpoint"),
        checkpoint_sha256="1" * 64,
        base_model=OFFICIAL_BASE_MODEL,
        base_model_revision="c83c3163b8ca9b7e67c509fffd9121e66cb96205",
        training_config_sha256="2" * 64,
        train_view_sha256="3" * 64,
        normalization_sha256="4" * 64,
        processor_sha256="5" * 64,
        training_seed=0,
        execution_horizon=horizon,
        device="cpu",
        dtype="float32",
        inference_seed=17,
    )


class _Component:
    def __init__(self, fn=None) -> None:
        self.reset_count = 0
        self.fn = fn or (lambda value: value)

    def reset(self) -> None:
        self.reset_count += 1

    def __call__(self, value):
        return self.fn(value)


class _Policy(_Component):
    def __init__(self, *, invalid: bool = False) -> None:
        super().__init__()
        self.config = SimpleNamespace(chunk_size=50, n_action_steps=50)
        self.invalid = invalid
        self.last_batch = None
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def to(self, *args, **kwargs):
        dtype = kwargs.get("dtype")
        if dtype is not None:
            self.weight.data = self.weight.data.to(dtype=dtype)
        return self

    def eval(self):
        return self

    def parameters(self):
        return [self.weight]

    def predict_action_chunk(self, batch):
        self.last_batch = batch
        value = torch.zeros((1, 50, 8), dtype=torch.float32)
        if self.invalid:
            value[0, 0, 0] = float("nan")
        return value


def test_train_view_is_exact_push_train_only_and_excludes_all_tests() -> None:
    manifest = build_train_view_manifest(_input_manifest())
    assert manifest["episode_count"] == TRAIN_EPISODES
    assert manifest["selection"] == {"split": "train", "skill_family": "push_to_region"}
    assert manifest["excluded_splits"] == [split for split in DATASET_SPLITS if split != "train"]
    assert manifest["failed_demonstrations_included"] is False
    assert manifest["privileged_features_included"] is False


def test_train_view_rejects_any_test_identity_overlap() -> None:
    value = _input_manifest()
    value["split_episode_ids"]["validation"][0] = value["split_episode_ids"]["train"][0]
    with pytest.raises(Phase2CContractError, match="overlap"):
        build_train_view_manifest(value)


def test_train_only_normalization_and_round_trip_detect_low_variance() -> None:
    manifest = compute_train_only_normalization(
        (_sample(index) for index in range(3)),
        expected_frames=3,
    )
    state = manifest["features"][STATE_FEATURE_KEY]
    action = manifest["features"][ACTION_FEATURE_KEY]
    assert state["count"] == 3
    assert state["mean"][0] == 1.0
    assert state["low_variance_dimensions"] == list(range(1, 9))
    assert action["low_variance_dimensions"] == list(range(1, 8))
    values = np.asarray([[0, 1, 2, 3, 4, 5, 6, 7, 0]], dtype=np.float64)
    assert np.allclose(normalization_round_trip(values, state), values, rtol=0, atol=1e-12)


def test_train_only_normalization_rejects_wrong_frame_count() -> None:
    with pytest.raises(Phase2CContractError, match="expected"):
        compute_train_only_normalization([_sample(0)], expected_frames=2)


def test_feature_mapping_accepts_uint8_and_float_but_no_privileged_keys() -> None:
    mapped = map_smolvla_observation(
        _observation(image=torch.full((3, 256, 256), 255, dtype=torch.uint8)),
        task="push the blue cube left",
    )
    assert mapped[IMAGE_FEATURE_KEY].dtype is torch.float32
    assert float(mapped[IMAGE_FEATURE_KEY].max()) == 1.0
    assert mapped["task"] == "push the blue cube left"
    with pytest.raises(PolicyContractError, match="privileged"):
        map_smolvla_observation(
            ObservationBatch(
                features={
                    **_observation().features,
                    "object_position": torch.zeros(3),
                }
            ),
            task="push",
        )
    assert feature_mapping_manifest()["privileged_features"] == []


def test_smolvla_config_requires_pinned_official_checkpoint_and_reject_mode(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    payload = config.to_dict()
    config_path = tmp_path / "adapter.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    assert SmolVLAAdapterConfig.load(config_path) == config
    with pytest.raises(SmolVLAAdapterError, match="execution_horizon"):
        _config(tmp_path, horizon=3)
    payload["action_bound_mode"] = "project"
    with pytest.raises(SmolVLAAdapterError, match="never projected"):
        SmolVLAAdapterConfig(**payload)


def test_smolvla_adapter_reset_chunk_mask_horizon_and_runtime(tmp_path: Path) -> None:
    policy = _Policy()
    preprocessor = _Component()
    postprocessor = _Component()
    adapter = SmolVLAPolicyAdapter(
        config=_config(tmp_path),
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        verify_checkpoint=False,
    )
    adapter.reset(_context())
    chunk = adapter.act(_observation())
    assert chunk.actions.shape == (4, 8)
    assert chunk.valid_mask.tolist() == [True, True, True, True]
    assert chunk.metadata["source_chunk_size"] == 50
    assert "left target region" in policy.last_batch["task"]
    assert policy.reset_count == preprocessor.reset_count == postprocessor.reset_count == 1
    runtime = adapter.runtime_manifest
    assert runtime["policy_query_count"] == 1
    assert runtime["actions_emitted"] == 4
    assert runtime["queue_clear_count"] == 1
    assert runtime["queue_underrun_count"] == 0
    assert runtime["action_bound_mode"] == "reject"
    assert runtime["parameter_count"] == 1
    assert runtime["observed_parameter_dtypes"] == ["float32"]


def test_smolvla_adapter_reset_clears_episode_query_state(tmp_path: Path) -> None:
    adapter = SmolVLAPolicyAdapter(
        config=_config(tmp_path, horizon=1),
        policy=_Policy(),
        preprocessor=_Component(),
        postprocessor=_Component(),
        verify_checkpoint=False,
    )
    adapter.reset(_context())
    adapter.act(_observation())
    adapter.reset(_context())
    assert adapter.runtime_manifest["policy_query_count"] == 0
    assert adapter.runtime_manifest["actions_emitted"] == 0
    assert adapter.runtime_manifest["queue_clear_count"] == 2


def test_smolvla_runtime_identity_excludes_checkpoint_path(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = SmolVLAPolicyAdapter(
        config=config,
        policy=_Policy(),
        preprocessor=_Component(),
        postprocessor=_Component(),
        verify_checkpoint=False,
    )
    payload = config.to_dict()
    payload["checkpoint_path"] = str(tmp_path / "relocated" / "checkpoint")
    second = SmolVLAPolicyAdapter(
        config=SmolVLAAdapterConfig(**payload),
        policy=_Policy(),
        preprocessor=_Component(),
        postprocessor=_Component(),
        verify_checkpoint=False,
    )
    assert (
        first.runtime_manifest["runtime_fingerprint"]
        == second.runtime_manifest["runtime_fingerprint"]
    )


def test_smolvla_adapter_rejects_invalid_output(tmp_path: Path) -> None:
    adapter = SmolVLAPolicyAdapter(
        config=_config(tmp_path),
        policy=_Policy(invalid=True),
        preprocessor=_Component(),
        postprocessor=_Component(),
        verify_checkpoint=False,
    )
    adapter.reset(_context())
    with pytest.raises(SmolVLAAdapterError, match="non-finite"):
        adapter.act(_observation())
    assert adapter.runtime_manifest["invalid_output_count"] == 1


def test_registry_exposes_real_smolvla_adapter_name() -> None:
    assert SMOLVLA_ADAPTER_NAME in default_policy_registry().registered_names


def test_immutable_json_and_nonfinite_manifest_guards(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_json_once(path, {"passed": True})
    write_json_once(path, {"passed": True})
    with pytest.raises(Phase2CContractError, match="overwrite"):
        write_json_once(path, {"passed": False})
    with pytest.raises(Phase2CContractError, match="non-finite"):
        require_finite_manifest({"loss": float("nan")})


def _schedule_records() -> list[dict[str, object]]:
    counts = {
        "train": 203,
        "validation": 77,
        "test_unseen_scene": 26,
        "test_unseen_language": 40,
        "test_hard": 23,
        "test_visual_shift": 28,
    }
    result: list[dict[str, object]] = []
    seed = 1000
    for split, count in counts.items():
        for index in range(count):
            cylinder = index % 2
            directions = ("left", "right", "forward_left", "forward_right")
            difficulty = "hard" if split == "test_hard" or index % 3 == 0 else "standard"
            result.append(
                {
                    "episode_id": f"{split}-{index}",
                    "split": split,
                    "seed": seed,
                    "task_spec": {
                        "target_object_id": "orange_cylinder" if cylinder else "blue_cube",
                        "target_region_id": directions[index % 4],
                        "difficulty": difficulty,
                        "instruction_template_id": "canonical_push_v0",
                    },
                    "instruction": f"held-out instruction {split} {index}",
                    "template_group": (
                        "held_out_paraphrase"
                        if split == "test_unseen_language"
                        else ("validation" if split == "validation" else "base")
                    ),
                    "template_id": f"template-{split}-{index % 4}",
                    "scene_group_id": f"scene-{split}-{index}",
                }
            )
            seed += 1
    return result


def test_outcome_free_schedule_balances_and_supplements_final_splits() -> None:
    schedule = build_phase2c_schedules(_schedule_records())
    assert schedule["development_count"] == 40
    assert schedule["sealed_count"] == 230
    assert schedule["supplemental_counts"] == {
        "test_hard": 27,
        "test_unseen_language": 10,
        "test_unseen_scene": 24,
        "test_visual_shift": 22,
    }
    sealed = schedule["sealed"]
    assert sum(item["split"] == "test_visual_shift" for item in sealed) == 50
    assert all(
        item["visual_domain"] == "photometric_shift_v0"
        for item in sealed
        if item["split"] == "test_visual_shift"
    )
    assert len({item["evaluation_id"] for item in schedule["development"] + sealed}) == 270


def test_development_gate_is_validation_only_and_conjunctive() -> None:
    records = []
    for index in range(40):
        records.append(
            {
                "split": "validation",
                "success": index < 24,
                "difficulty": "standard" if index < 20 else "hard",
                "geometry": "horizontal_cylinder" if index == 0 else "cube",
                "direction": "left" if index == 0 else "forward_left",
                "invalid_policy_output": False,
                "action_bound_failure": False,
                "nonfinite_action": False,
            }
        )
    gate = development_competence_gate(records)
    assert gate["passed"] is True
    records[0]["split"] = "test_hard"
    with pytest.raises(Phase2CContractError, match="validation"):
        development_competence_gate(records)


def test_final_summary_confidence_and_quality_are_predeclared() -> None:
    records: list[dict[str, object]] = []
    rates = {
        "validation_standard": 0.8,
        "test_unseen_scene": 0.66,
        "test_unseen_language": 0.66,
        "test_hard": 0.5,
        "test_visual_shift": 0.6,
    }
    for split, rate in rates.items():
        for index in range(100):
            records.append(
                {
                    "split": split,
                    "success": index < int(rate * 100),
                    "geometry": "cube",
                    "direction": "left",
                    "difficulty": "standard",
                    "template_group": "base",
                    "timeout": index >= int(rate * 100),
                    "outcome": "timeout",
                    "invalid_action": False,
                    "action_saturation_rate": 0.0,
                    "episode_length": 250,
                    "final_evaluation": {},
                }
            )
    summary = summarize_sealed_results(records)
    assert summary["overall"]["wilson_95"][0] < summary["overall"]["success_rate"]
    assert quality_level(summary)["quality_level"] == 3
    assert wilson_interval(5, 10)[0] < 0.5 < wilson_interval(5, 10)[1]


def test_quality_level_can_bind_validation_only_development_evidence() -> None:
    records = []
    for split, successes in {
        "train_distribution_sanity": 21,
        "test_unseen_scene": 33,
        "test_unseen_language": 33,
        "test_hard": 25,
        "test_visual_shift": 30,
    }.items():
        for index in range(50):
            records.append(
                {
                    "split": split,
                    "success": index < successes,
                    "geometry": "cube",
                    "direction": "left",
                    "difficulty": "standard",
                    "template_group": "base",
                    "timeout": index >= successes,
                    "outcome": "timeout",
                    "invalid_action": False,
                    "action_saturation_rate": 0.0,
                }
            )
    summary = summarize_sealed_results(records)
    decision = quality_level(summary, development_gate={"standard_success_rate": 0.8})
    assert decision["quality_level"] == 3


def test_frozen_policy_manifest_binds_all_final_settings() -> None:
    manifest = FrozenPolicyManifest(
        checkpoint_sha256="1" * 64,
        base_model_revision="c83c3163b8ca9b7e67c509fffd9121e66cb96205",
        training_seed=0,
        training_config_sha256="2" * 64,
        train_view_sha256="3" * 64,
        normalization_sha256="4" * 64,
        processor_sha256="5" * 64,
        action_chunk_size=50,
        execution_horizon=4,
        control_frequency_hz=20,
        inference_seed=271828,
        final_test_identity_sha256="6" * 64,
    )
    assert manifest.to_dict()["semantic_sha256"].startswith("sha256:")


def test_offline_metrics_mask_padding_and_select_validation_only() -> None:
    accumulator = OfflineMetricAccumulator(horizon=2)
    expected = np.zeros((1, 2, 8), dtype=np.float32)
    predicted = np.ones((1, 2, 8), dtype=np.float32)
    accumulator.add(
        validation_loss=0.25,
        predicted_normalized=predicted,
        expected_normalized=expected,
        predicted_raw=predicted,
        expected_raw=expected,
        action_is_pad=np.asarray([[False, True]], dtype=bool),
    )
    metrics = accumulator.finalize()
    assert metrics["validation_loss"] == 0.25
    assert metrics["inverse_normalized_action_mae"] == 1.0
    assert metrics["inverse_normalized_mae_by_horizon_position"] == [1.0, None]
    reports = [
        {
            "completed": True,
            "split": "validation",
            "checkpoint_sha256": character * 64,
            "metrics": {
                "validation_loss": loss,
                "inverse_normalized_first_action_mae": first,
            },
        }
        for character, loss, first in (("a", 0.2, 0.1), ("b", 0.1, 0.2), ("c", 0.1, 0.1))
    ]
    selection = select_offline_checkpoints(reports)
    assert [item["checkpoint_sha256"] for item in selection["selected"]] == [
        "c" * 64,
        "b" * 64,
    ]


def test_hold_position_baseline_uses_only_deployable_state() -> None:
    policy = HoldPositionPolicyAdapter()
    policy.reset(_context())
    chunk = policy.act(_observation())
    assert chunk.actions.shape == (1, 8)
    assert torch.equal(chunk.actions[0, :7], torch.arange(7, dtype=torch.float32))
    assert chunk.actions[0, 7].item() == 7.5
    assert policy.runtime_manifest["privileged_inputs"] is False


def test_training_launcher_replaces_base_camera_mapping_instead_of_merging(tmp_path: Path) -> None:
    launcher = _load_training_launcher()
    protocol = json.loads(
        (PROJECT_ROOT / "configs" / "langmani_v2" / "phase2c_smolvla_push.json").read_text(
            encoding="utf-8"
        )
    )
    args = SimpleNamespace(
        stage="smoke",
        batch_size=1,
        num_workers=0,
        seed=0,
        max_steps=1,
        limit_samples=None,
        checkpoint_every=1,
        episodes=None,
        resume_checkpoint=None,
        dry_run=True,
        lerobot_train="lerobot-train",
        repo_id="phase2c-train-view",
        train_root=tmp_path / "train",
        output_dir=tmp_path / "output",
    )
    command, steps = launcher.build_command(args, protocol)
    assert steps == 1
    assert "--policy.input_features=null" in command
    assert "--policy.optimizer_betas=[0.9,0.95]" in command
    assert "0.95" not in command
    assert not any(item.startswith("--policy.output_features=") for item in command)
    assert not any("observation.images.camera" in item for item in command)
