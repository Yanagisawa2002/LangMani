"""Pure M4 ACT variant, identity, and feature-contract tests."""

from __future__ import annotations

import dataclasses
import json

import pytest

from langmani.datasets.lerobot_types import ACTION_FEATURE_KEY, IMAGE_FEATURE_KEY, STATE_FEATURE_KEY
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import stable_task_id
from langmani.policies.act_runtime import (
    GitState,
    RuntimeContractError,
    safe_run_directory,
    validate_git_for_run,
)
from langmani.policies.act_training import DeterministicResumeBatchSampler
from langmani.policies.act_types import (
    ActDataConfig,
    ActExperimentConfig,
    ActModelConfig,
    ActRunIdentity,
    ActVariant,
    ExperimentMode,
)

SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
COMMIT = "1" * 40


def _identity(*, root: str = "C:/machine/a", seed: int = 0) -> ActRunIdentity:
    data = ActDataConfig(dataset_root=root)
    model = ActModelConfig.for_variant(ActVariant.MIXED_UNCONDITIONED)
    return ActRunIdentity(
        m3b_export_fingerprint=SHA_A,
        m3b_split_manifest_digest=SHA_B,
        ordered_train_episode_indices=(0, 1, 2),
        ordered_validation_episode_indices=(3,),
        variant=ActVariant.MIXED_UNCONDITIONED,
        task_id=None,
        task_onehot_mapping_version="CanonicalTaskOneHotV0",
        model_config=model.to_dict(),
        data_contract=data.identity_dict(),
        train_statistics_fingerprint=SHA_C,
        optimization_config={"optimizer": "adamw", "lr": 1e-5},
        training_seed=seed,
        experiment_mode=ExperimentMode.DRY_RUN,
        device="cpu",
        dtype="float32",
        lerobot_version="0.6.0",
        torch_version="2.11.0+cpu",
        cuda_version=None,
        git_commit=COMMIT,
        git_dirty=False,
        dirty_development_override=False,
    )


def test_act_variant_is_exact_and_rejects_unknown() -> None:
    assert tuple(item.value for item in ActVariant) == (
        "per_task",
        "mixed_unconditioned",
        "mixed_task_onehot",
    )
    with pytest.raises(ValueError):
        ActVariant("language_act")


def test_per_task_requires_stable_task_id_and_mixed_rejects_it() -> None:
    data = ActDataConfig(dataset_root="dataset")
    per_task_model = ActModelConfig.for_variant(ActVariant.PER_TASK)
    with pytest.raises(ValueError, match="require one stable task_id"):
        ActExperimentConfig(
            variant=ActVariant.PER_TASK,
            data=data,
            model=per_task_model,
        )
    task_id = stable_task_id(CANONICAL_TASK_SPECS[0])
    config = ActExperimentConfig(
        variant=ActVariant.PER_TASK,
        task_id=task_id,
        data=data,
        model=per_task_model,
    )
    assert config.task_id == task_id
    with pytest.raises(ValueError, match="must not select"):
        ActExperimentConfig(
            variant=ActVariant.MIXED_UNCONDITIONED,
            task_id=task_id,
            data=data,
            model=ActModelConfig.for_variant(ActVariant.MIXED_UNCONDITIONED),
        )


def test_model_input_contracts_are_9d_or_explicit_15d() -> None:
    assert ActModelConfig.for_variant(ActVariant.PER_TASK).state_dimension == 9
    assert ActModelConfig.for_variant(ActVariant.MIXED_UNCONDITIONED).state_dimension == 9
    assert ActModelConfig.for_variant(ActVariant.MIXED_TASK_ONEHOT).state_dimension == 15
    model = ActModelConfig.for_variant(ActVariant.MIXED_TASK_ONEHOT)
    assert {
        model.image_feature_key,
        model.state_feature_key,
        model.action_feature_key,
    } == {IMAGE_FEATURE_KEY, STATE_FEATURE_KEY, ACTION_FEATURE_KEY}


def test_run_fingerprint_is_stable_and_ignores_machine_paths() -> None:
    first = _identity(root="C:/users/a/dataset")
    second = _identity(root="/mnt/data/dataset")
    assert first.run_fingerprint == second.run_fingerprint
    assert first.run_fingerprint == first.compute_fingerprint()


def test_run_fingerprint_changes_for_semantic_config_changes() -> None:
    first = _identity(seed=0)
    second = _identity(seed=1)
    assert first.run_fingerprint != second.run_fingerprint
    changed = dataclasses.replace(
        first,
        model_config={**first.model_config, "chunk_size": 25},
        run_fingerprint="",
    )
    assert changed.run_fingerprint != first.run_fingerprint


def test_run_identity_recursively_freezes_semantic_mappings() -> None:
    source = {"nested": {"chunk": [1, 2, 3]}}
    identity = dataclasses.replace(
        _identity(),
        model_config=source,
        run_fingerprint="",
    )
    fingerprint = identity.run_fingerprint
    source["nested"]["chunk"].append(4)
    assert identity.run_fingerprint == fingerprint == identity.compute_fingerprint()
    with pytest.raises(TypeError):
        identity.model_config["nested"] = {}  # type: ignore[index]


def test_dirty_development_and_clean_full_have_distinct_run_identities() -> None:
    base = _identity()
    dirty_development = dataclasses.replace(
        base,
        experiment_mode=ExperimentMode.DEVELOPMENT,
        git_dirty=True,
        dirty_development_override=True,
        run_fingerprint="",
    )
    clean_full = dataclasses.replace(
        base,
        experiment_mode=ExperimentMode.FULL,
        run_fingerprint="",
    )
    assert dirty_development.run_fingerprint != clean_full.run_fingerprint


def test_experiment_and_identity_json_roundtrip() -> None:
    config = ActExperimentConfig(
        variant=ActVariant.MIXED_UNCONDITIONED,
        data=ActDataConfig(dataset_root="dataset"),
        model=ActModelConfig.for_variant(ActVariant.MIXED_UNCONDITIONED),
    )
    assert ActExperimentConfig.from_dict(config.to_dict()) == config
    identity = _identity()
    assert ActRunIdentity.from_dict(identity.to_dict()) == identity


def test_dirty_override_is_development_only() -> None:
    with pytest.raises(ValueError, match="development-only"):
        ActExperimentConfig(
            variant=ActVariant.MIXED_UNCONDITIONED,
            data=ActDataConfig(dataset_root="dataset"),
            model=ActModelConfig.for_variant(ActVariant.MIXED_UNCONDITIONED),
            mode=ExperimentMode.FULL,
            allow_dirty_development=True,
        )


def test_dirty_git_full_run_is_rejected_and_development_is_recorded() -> None:
    state = GitState(
        commit=COMMIT,
        dirty=True,
        changed_paths=(" M file.py",),
        baseline_tracked=True,
    )
    with pytest.raises(RuntimeContractError, match="clean Git"):
        validate_git_for_run(
            state,
            mode=ExperimentMode.FULL,
            allow_dirty_development=False,
        )
    validate_git_for_run(
        state,
        mode=ExperimentMode.DEVELOPMENT,
        allow_dirty_development=True,
    )
    assert state.to_dict()["changed_paths"] == [" M file.py"]
    assert json.loads(json.dumps(state.to_dict())) == state.to_dict()


def test_output_path_is_contained_by_root(tmp_path) -> None:
    destination = safe_run_directory(tmp_path, SHA_A)
    assert destination.parent == tmp_path.resolve()


def test_resume_batch_sampler_matches_uninterrupted_sequence_across_epochs() -> None:
    uninterrupted = iter(
        DeterministicResumeBatchSampler(
            dataset_size=11,
            batch_size=4,
            seed=17,
        )
    )
    expected = [next(uninterrupted) for _ in range(10)]
    resumed = iter(
        DeterministicResumeBatchSampler(
            dataset_size=11,
            batch_size=4,
            seed=17,
            start_step=5,
        )
    )
    assert expected[5:] == [next(resumed) for _ in range(5)]
