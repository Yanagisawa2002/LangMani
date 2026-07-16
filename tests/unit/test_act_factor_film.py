"""CPU-safe contracts for the single M4.3b FactorFiLM ACT policy."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from lerobot.policies.act import ACTPolicy

from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import stable_task_id
from langmani.policies.act_factor_film_adapter import (
    DestinationBinConditionV0,
    FactorFiLMUpstreamContractError,
    TargetObjectConditionV0,
    build_factor_film_architecture_identity,
    factor_film_parameter_counts,
    factor_film_upstream_contract,
    validate_factor_film_policy_structure,
)
from langmani.policies.act_factor_film_conditioning import (
    attach_factor_film_runtime_input,
    extract_factor_film_runtime_tensors,
    factorized_runtime_input,
    factorized_task_condition,
    runtime_input_from_episode_indices,
    runtime_input_tensors,
)
from langmani.policies.act_factor_film_training import (
    AUTHORIZED_M43_AUDIT_EVIDENCE_FINGERPRINT,
    build_factor_film_checkpoint_contract,
    build_factor_film_policy_and_processors,
    build_factor_film_validation_queue,
    create_factor_film_selection,
    rank_factor_film_validation_results,
)
from langmani.policies.act_factor_film_types import (
    DESTINATION_BIN_IDS,
    FACTOR_FILM_BIN_INDEX_KEY,
    FACTOR_FILM_OBJECT_INDEX_KEY,
    TARGET_OBJECT_IDS,
    DestinationBinConditionMapping,
    FactorFiLMArchitectureIdentity,
    FactorFiLMCheckpointContract,
    FactorFiLMConfig,
    FactorFiLMContractError,
    FactorFiLMRunIdentity,
    FactorFiLMRuntimeInput,
    FactorFiLMSelectionRecord,
    FactorFiLMTrainingConfig,
    FactorFiLMTrainingManifest,
    FactorFiLMTrainingMode,
    FactorFiLMValidationQueue,
    FactorFiLMValidationQueueItem,
    FactorFiLMValidationResult,
    FactorizedTaskCondition,
    TargetObjectConditionMapping,
    canonical_fingerprint,
)
from langmani.policies.act_types import (
    ActDataConfig,
    ActModelConfig,
    ActOptimizationConfig,
    CheckpointRecord,
    TrainingState,
)
from scripts.train_act_factor_film import _resume_is_atomic_orphan

_DIGEST_A = "sha256:" + "a" * 64
_DIGEST_B = "sha256:" + "b" * 64
_DIGEST_C = "sha256:" + "c" * 64
_DIGEST_D = "sha256:" + "d" * 64
_DIGEST_E = "sha256:" + "e" * 64
_GIT_COMMIT = "dfea8b3d7d28274909ff178cb9087a9a90e17ee7"


def _small_model() -> ActModelConfig:
    return ActModelConfig(
        state_dimension=9,
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


def _small_training_config() -> FactorFiLMTrainingConfig:
    return FactorFiLMTrainingConfig(
        data=ActDataConfig(dataset_root="fixture-only"),
        base_model=_small_model(),
        optimization=ActOptimizationConfig(
            mixed_precision="none",
            batch_size=2,
            dataloader_workers=0,
            training_steps=1,
            checkpoint_interval=1,
            validation_interval=1,
        ),
        semantic_audit_evidence_fingerprint=(AUTHORIZED_M43_AUDIT_EVIDENCE_FINGERPRINT),
        training_seed=43,
        mode=FactorFiLMTrainingMode.FIXTURE,
        device="cpu",
    )


def _stats() -> dict[str, dict[str, torch.Tensor]]:
    return {
        "observation.images.base_camera": {
            "mean": torch.zeros(3),
            "std": torch.ones(3),
            "min": torch.zeros(3),
            "max": torch.ones(3),
        },
        "observation.state": {
            "mean": torch.zeros(9),
            "std": torch.ones(9),
            "min": -torch.ones(9),
            "max": torch.ones(9),
        },
        "action": {
            "mean": torch.zeros(8),
            "std": torch.ones(8),
            "min": -torch.ones(8),
            "max": torch.ones(8),
        },
    }


@pytest.fixture(scope="module")
def small_policy_bundle():
    return build_factor_film_policy_and_processors(
        training_config=_small_training_config(),
        statistics=_stats(),
        factor_film_config=FactorFiLMConfig(state_context_dimension=32),
    )


def _run_identity(**overrides: object) -> FactorFiLMRunIdentity:
    values: dict[str, object] = {
        "m3b_export_fingerprint": _DIGEST_A,
        "m3b_split_manifest_digest": _DIGEST_B,
        "ordered_train_episode_indices": (0, 1),
        "ordered_validation_episode_indices": (2,),
        "architecture_identity": {
            "architecture_name": "ACT-Mixed-FactorFiLM",
            "architecture_fingerprint": _DIGEST_C,
        },
        "model_config": {
            "state_dimension": 9,
            "object_embedding_dimension": 32,
            "bin_embedding_dimension": 32,
        },
        "data_contract": {
            "checkpoint_selection_source": "m3b_validation",
            "test_accessible": False,
            "sealed_schedule_accessed": False,
        },
        "train_statistics_fingerprint": _DIGEST_D,
        "optimization_config": _small_training_config().optimization.to_dict(),
        "semantic_audit_evidence_fingerprint": (AUTHORIZED_M43_AUDIT_EVIDENCE_FINGERPRINT),
        "training_seed": 43,
        "experiment_mode": FactorFiLMTrainingMode.FIXTURE,
        "device": "cpu",
        "dtype": "float32",
        "lerobot_version": "0.6.0",
        "torch_version": torch.__version__,
        "cuda_version": None,
        "git_commit": _GIT_COMMIT,
        "git_dirty": False,
    }
    values.update(overrides)
    return FactorFiLMRunIdentity(**values)  # type: ignore[arg-type]


def _validation_candidates() -> tuple[FactorFiLMValidationResult, ...]:
    schedule = "sha256:" + "f" * 64
    values: list[FactorFiLMValidationResult] = []
    for index, step in enumerate(range(5_000, 100_001, 5_000), start=1):
        values.append(
            FactorFiLMValidationResult(
                checkpoint_fingerprint=f"sha256:{index:064x}",
                checkpoint_step=step,
                source_split="m3b_validation",
                schedule_digest=schedule,
                task_success_rate=0.5,
                wrong_object_interaction_rate=0.2,
                wrong_object_in_target_bin_rate=0.1,
                target_off_table_rate=0.05,
                timeout_rate=0.03,
                offline_validation_action_loss=0.5,
            )
        )
    # Later step wins on success; step 10k then beats 15k via wrong-object rate.
    values[1] = replace(
        values[1],
        task_success_rate=0.9,
        wrong_object_interaction_rate=0.05,
    )
    values[2] = replace(
        values[2],
        task_success_rate=0.9,
        wrong_object_interaction_rate=0.10,
    )
    return tuple(values)


def _validation_queue() -> FactorFiLMValidationQueue:
    candidates = _validation_candidates()
    return FactorFiLMValidationQueue(
        run_fingerprint=_DIGEST_A,
        validation_schedule_digest=candidates[0].schedule_digest,
        items=tuple(
            FactorFiLMValidationQueueItem(
                checkpoint_fingerprint=value.checkpoint_fingerprint,
                checkpoint_step=value.checkpoint_step,
                checkpoint_relative_path=(f"checkpoints/step-{value.checkpoint_step:08d}-fixture"),
                offline_validation_action_loss=value.offline_validation_action_loss,
            )
            for value in candidates
        ),
    )


def _target_identity(
    *,
    train_indices: tuple[int, ...] = tuple(range(288)),
    validation_indices: tuple[int, ...] = tuple(range(288, 324)),
) -> FactorFiLMRunIdentity:
    return _run_identity(
        ordered_train_episode_indices=train_indices,
        ordered_validation_episode_indices=validation_indices,
        data_contract={
            "checkpoint_selection_source": "m3b_validation",
            "execution_horizon": 10,
            "gripper_runtime_mode": "project",
            "train_episode_count": len(train_indices),
            "validation_episode_count": len(validation_indices),
            "train_episode_indices_fingerprint": canonical_fingerprint(train_indices),
            "validation_episode_indices_fingerprint": canonical_fingerprint(validation_indices),
        },
        training_seed=0,
        experiment_mode=FactorFiLMTrainingMode.TARGET_DEVELOPMENT,
        device="cuda",
    )


def test_canonical_mappings_are_stable_json_contracts() -> None:
    object_mapping = TargetObjectConditionMapping()
    bin_mapping = DestinationBinConditionMapping()

    assert TARGET_OBJECT_IDS == ("red_cube", "green_cube", "blue_cube")
    assert DESTINATION_BIN_IDS == ("left_bin", "right_bin")
    assert object_mapping.ordered_object_ids == TARGET_OBJECT_IDS
    assert bin_mapping.ordered_bin_ids == DESTINATION_BIN_IDS
    assert object_mapping.fingerprint == TargetObjectConditionMapping().fingerprint
    assert bin_mapping.fingerprint == DestinationBinConditionMapping().fingerprint
    assert TargetObjectConditionMapping.from_dict(object_mapping.to_dict()) == object_mapping
    assert DestinationBinConditionMapping.from_dict(bin_mapping.to_dict()) == bin_mapping
    json.dumps(object_mapping.to_dict(), sort_keys=True)
    json.dumps(bin_mapping.to_dict(), sort_keys=True)

    with pytest.raises(FactorFiLMContractError, match="target-object order"):
        TargetObjectConditionMapping(ordered_object_ids=tuple(reversed(TARGET_OBJECT_IDS)))
    with pytest.raises(FactorFiLMContractError, match="destination-bin order"):
        DestinationBinConditionMapping(ordered_bin_ids=tuple(reversed(DESTINATION_BIN_IDS)))


def test_task_decomposition_uses_stable_metadata_and_rejects_unknown_values() -> None:
    for task_spec in CANONICAL_TASK_SPECS:
        condition = factorized_task_condition(task_spec)
        by_id = factorized_task_condition(stable_task_id(task_spec))
        assert condition == by_id
        assert condition.target_object_index == TARGET_OBJECT_IDS.index(task_spec.target_object_id)
        assert condition.destination_bin_index == DESTINATION_BIN_IDS.index(task_spec.target_bin_id)

    with pytest.raises(FactorFiLMContractError, match="unknown stable task ID"):
        factorized_task_condition("not-a-task")
    with pytest.raises(TypeError, match="task must"):
        factorized_task_condition("")
    with pytest.raises(FactorFiLMContractError, match="unknown target-object"):
        FactorizedTaskCondition(
            task_id="malformed",
            target_object_id="orange_cube",
            destination_bin_id="left_bin",
            target_object_index=0,
            destination_bin_index=0,
        )
    with pytest.raises(FactorFiLMContractError, match="unknown destination-bin"):
        FactorizedTaskCondition(
            task_id="malformed",
            target_object_id="red_cube",
            destination_bin_id="center_bin",
            target_object_index=0,
            destination_bin_index=0,
        )
    valid = factorized_task_condition(CANONICAL_TASK_SPECS[0])
    with pytest.raises(FactorFiLMContractError, match="stable task ID does not match"):
        replace(
            valid,
            target_object_id="blue_cube",
            target_object_index=2,
            destination_bin_id="right_bin",
            destination_bin_index=1,
        )
    malformed = valid.to_dict()
    malformed["task_id"] = "unknown-task"
    with pytest.raises(FactorFiLMContractError, match="stable task ID does not match"):
        FactorizedTaskCondition.from_dict(malformed)


def test_batched_condition_tensors_have_exact_shape_dtype_device_and_provenance() -> None:
    task_ids = tuple(stable_task_id(value) for value in CANONICAL_TASK_SPECS)
    runtime = factorized_runtime_input(task_ids)
    object_indices, bin_indices = runtime_input_tensors(runtime, device="cpu")

    assert runtime.batch_size == 6
    assert object_indices.shape == bin_indices.shape == (6,)
    assert object_indices.dtype == bin_indices.dtype == torch.long
    assert object_indices.device == bin_indices.device == torch.device("cpu")
    assert object_indices.tolist() == [0, 0, 1, 1, 2, 2]
    assert bin_indices.tolist() == [0, 1, 0, 1, 0, 1]
    assert FactorFiLMRuntimeInput.from_dict(runtime.to_dict()) == runtime
    json.dumps(runtime.to_dict(), sort_keys=True)
    assert runtime_input_from_episode_indices(
        torch.tensor([8, 9], dtype=torch.int32),
        task_id_by_episode={8: task_ids[0], 9: task_ids[-1]},
    ).conditions == (runtime.conditions[0], runtime.conditions[-1])

    with pytest.raises(FactorFiLMContractError, match="no stable M3B task mapping"):
        runtime_input_from_episode_indices(torch.tensor([8]), task_id_by_episode={})


def test_condition_attachment_keeps_state_9d_and_uses_project_only_keys() -> None:
    state = torch.zeros(2, 9, dtype=torch.float32)
    batch: dict[str, object] = {
        "observation.state": state,
        "observation.images.base_camera": torch.zeros(2, 3, 256, 256),
    }
    runtime = factorized_runtime_input(CANONICAL_TASK_SPECS[:2])
    conditioned = attach_factor_film_runtime_input(batch, runtime)

    assert conditioned["observation.state"] is state
    assert state.shape == (2, 9)
    assert FACTOR_FILM_OBJECT_INDEX_KEY in conditioned
    assert FACTOR_FILM_BIN_INDEX_KEY in conditioned
    assert not any("onehot" in key or "task_token" in key for key in conditioned)
    clean, object_indices, bin_indices = extract_factor_film_runtime_tensors(conditioned)
    assert clean == batch
    assert object_indices.tolist() == [0, 0]
    assert bin_indices.tolist() == [0, 1]

    with pytest.raises(FactorFiLMContractError, match="batch size"):
        attach_factor_film_runtime_input(batch, factorized_runtime_input(CANONICAL_TASK_SPECS[:1]))
    with pytest.raises(FactorFiLMContractError, match="must be float32"):
        attach_factor_film_runtime_input({"observation.state": state.double()}, runtime)


def test_film_modules_have_separate_shapes_identity_like_init_and_broadcasting() -> None:
    torch.manual_seed(43)
    config = FactorFiLMConfig(state_context_dimension=16)
    object_film = TargetObjectConditionV0(config)
    bin_film = DestinationBinConditionV0(config)
    visual = torch.randn(2, 512, 3, 5)
    state = torch.randn(2, 16)
    object_indices = torch.tensor([0, 2])
    bin_indices = torch.tensor([0, 1])

    object_gamma, object_beta = object_film.gamma_beta(object_indices)
    bin_gamma, bin_beta = bin_film.gamma_beta(bin_indices)
    assert object_film.embedding.weight.shape == (3, 32)
    assert bin_film.embedding.weight.shape == (2, 32)
    assert object_gamma.shape == object_beta.shape == (2, 512)
    assert bin_gamma.shape == bin_beta.shape == (2, 16)

    visual_conditioned = object_film(visual, object_indices)
    state_conditioned = bin_film(state, bin_indices)
    visual_manual = visual * (1 + object_gamma[:, :, None, None]) + object_beta[:, :, None, None]
    state_manual = state * (1 + bin_gamma) + bin_beta
    torch.testing.assert_close(visual_conditioned, visual_manual)
    torch.testing.assert_close(state_conditioned, state_manual)
    assert float((visual_conditioned - visual).abs().max().detach()) < 1e-3
    assert float((state_conditioned - state).abs().max().detach()) < 1e-3
    assert not torch.equal(
        object_film(visual, torch.tensor([0, 0])),
        object_film(visual, torch.tensor([2, 2])),
    )
    assert not torch.equal(
        bin_film(state, torch.tensor([0, 0])),
        bin_film(state, torch.tensor([1, 1])),
    )


def test_film_modules_fail_closed_on_batch_dtype_device_and_range() -> None:
    config = FactorFiLMConfig(state_context_dimension=16)
    object_film = TargetObjectConditionV0(config)
    bin_film = DestinationBinConditionV0(config)
    visual = torch.zeros(2, 512, 2, 2)
    state = torch.zeros(2, 16)

    with pytest.raises(FactorFiLMContractError, match=r"torch.long\[B\]"):
        object_film(visual, torch.tensor([0, 1], dtype=torch.int32))
    with pytest.raises(FactorFiLMContractError, match=r"torch.long\[B\]"):
        bin_film(state, torch.tensor([0]))
    with pytest.raises(FactorFiLMContractError, match="out of range"):
        object_film(visual, torch.tensor([0, 3]))
    with pytest.raises(FactorFiLMContractError, match="out of range"):
        bin_film(state, torch.tensor([0, 2]))
    with pytest.raises(FactorFiLMContractError, match="different devices"):
        object_film(visual, torch.empty(2, dtype=torch.long, device="meta"))


def test_installed_act_contract_identity_and_parameter_increase(
    small_policy_bundle,
) -> None:
    _, policy, _, _, architecture = small_policy_bundle
    validate_factor_film_policy_structure(policy)
    upstream = factor_film_upstream_contract()
    base_count, factor_count, increase = factor_film_parameter_counts(policy)
    rebuilt = build_factor_film_architecture_identity(
        policy, base_act_configuration=_small_model().to_dict()
    )

    assert upstream["lerobot_version"] == "0.6.0"
    assert {
        "ACTConfig": "lerobot.policies.act.ACTConfig",
        "ACTPolicy": "lerobot.policies.act.ACTPolicy",
        "make_act_pre_post_processors": "lerobot.policies.act.make_act_pre_post_processors",
    }.items() <= upstream["public_symbols"].items()
    assert {
        "ACTPolicy.__init__",
        "ACTPolicy.forward",
        "ACTPolicy.predict_action_chunk",
        "ACTPolicy.select_action",
        "make_act_pre_post_processors",
    } <= upstream["public_signatures"].keys()
    assert upstream["private_module_imported"] is False
    assert upstream["global_monkey_patch"] is False
    assert architecture == rebuilt
    assert (base_count, factor_count, increase) == (
        architecture.base_parameter_count,
        architecture.factor_film_parameter_count,
        architecture.parameter_count_increase,
    )
    expected_increase = (3 * 32 + 2 * 512 * 32 + 2 * 512) + (2 * 32 + 2 * 32 * 32 + 2 * 32)
    assert increase == expected_increase == 36_064
    assert json.loads(json.dumps(architecture.to_dict()))["architecture_name"] == (
        "ACT-Mixed-FactorFiLM"
    )
    assert architecture.architecture_fingerprint == rebuilt.architecture_fingerprint
    assert FactorFiLMArchitectureIdentity.from_dict(architecture.to_dict()) == architecture


def test_upstream_contract_rejects_same_version_signature_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ACTPolicy.forward

    def incompatible(self, batch, unexpected_required_argument):
        return original(self, batch), unexpected_required_argument

    monkeypatch.setattr(ACTPolicy, "forward", incompatible)
    with pytest.raises(FactorFiLMUpstreamContractError, match="signature changed"):
        factor_film_upstream_contract()


def test_factor_film_rejects_noncanonical_policy_camera_shape() -> None:
    with pytest.raises(ValueError, match="base_camera"):
        replace(_small_model(), image_shape_chw=(3, 128, 128))


def test_architecture_and_run_fingerprints_are_stable_and_path_free(
    small_policy_bundle,
) -> None:
    _, _, _, _, architecture = small_policy_bundle
    identity = _run_identity(architecture_identity=architecture.to_dict())
    roundtrip = FactorFiLMRunIdentity.from_dict(identity.to_dict())

    assert roundtrip == identity
    assert roundtrip.run_fingerprint == identity.run_fingerprint
    assert identity.run_fingerprint == canonical_fingerprint(identity.identity_payload())
    serialized = json.dumps(identity.to_dict(), sort_keys=True)
    assert "output_root" not in serialized
    assert "checkpoint_root" not in serialized

    changed = _run_identity(
        architecture_identity={
            **architecture.to_dict(),
            "factor_film_config": FactorFiLMConfig(
                object_embedding_dimension=16,
                state_context_dimension=32,
            ).to_dict(),
        }
    )
    assert changed.run_fingerprint != identity.run_fingerprint
    assert (
        FactorFiLMConfig().fingerprint
        != FactorFiLMConfig(object_embedding_dimension=16).fingerprint
    )


def test_checkpoint_contract_and_manifest_roundtrip_reject_foreign_resume(
    small_policy_bundle,
) -> None:
    _, _, _, _, architecture = small_policy_bundle
    identity = _run_identity(architecture_identity=architecture.to_dict())
    contract = build_factor_film_checkpoint_contract(
        identity=identity,
        architecture=architecture,
    )
    checkpoint = CheckpointRecord(
        checkpoint_fingerprint=_DIGEST_C,
        run_fingerprint=identity.run_fingerprint,
        global_step=1,
        relative_path="checkpoints/step-00000001-fixture",
        policy_config_digest=canonical_fingerprint(identity.model_config),
        statistics_fingerprint=identity.train_statistics_fingerprint,
        split_digest=identity.m3b_split_manifest_digest,
        git_commit=identity.git_commit,
        complete=True,
    )
    manifest = FactorFiLMTrainingManifest(
        identity=identity,
        training_config=_small_training_config(),
        architecture_identity=architecture,
        checkpoint_contract=contract,
        training_state=TrainingState(
            global_step=1,
            examples_processed=2,
            last_checkpoint_fingerprint=checkpoint.checkpoint_fingerprint,
            completed=True,
        ),
        checkpoints=(checkpoint,),
        training_complete=True,
    )

    assert FactorFiLMCheckpointContract.from_dict(contract.to_dict()) == contract
    assert FactorFiLMTrainingManifest.from_dict(manifest.to_dict()) == manifest
    json.dumps(manifest.to_dict(), sort_keys=True)
    with pytest.raises(FactorFiLMContractError, match="another run"):
        replace(
            manifest,
            checkpoints=(replace(checkpoint, run_fingerprint=_DIGEST_A),),
        )
    with pytest.raises(FactorFiLMContractError, match="checkpoint provenance"):
        replace(
            manifest,
            checkpoints=(replace(checkpoint, statistics_fingerprint=_DIGEST_E),),
        )
    with pytest.raises(FactorFiLMContractError, match="state fingerprint"):
        replace(
            manifest,
            training_state=replace(
                manifest.training_state,
                last_checkpoint_fingerprint=_DIGEST_E,
            ),
        )


@pytest.mark.parametrize("forbidden", ["m3b_test", "fresh_seed", "m42_final_v0"])
def test_run_identity_rejects_test_fresh_and_final_sources(forbidden: str) -> None:
    with pytest.raises(FactorFiLMContractError, match="forbidden source"):
        _run_identity(data_contract={forbidden: False})


def test_target_run_rejects_dirty_git_and_resume_identity_mismatches() -> None:
    with pytest.raises(FactorFiLMContractError, match="clean Git"):
        _run_identity(
            experiment_mode=FactorFiLMTrainingMode.TARGET_DEVELOPMENT,
            device="cuda",
            git_dirty=True,
        )
    with pytest.raises(FactorFiLMContractError, match="seed 0"):
        replace(
            _small_training_config(),
            base_model=ActModelConfig(),
            optimization=ActOptimizationConfig(),
            training_seed=1,
            mode=FactorFiLMTrainingMode.TARGET_DEVELOPMENT,
            device="cuda",
        )

    identity = _run_identity()
    payload = identity.to_dict()
    payload["m3b_export_fingerprint"] = "sha256:" + "0" * 64
    with pytest.raises(FactorFiLMContractError, match="run fingerprint mismatch"):
        FactorFiLMRunIdentity.from_dict(payload)

    payload = identity.to_dict()
    payload["m3b_split_manifest_digest"] = "sha256:" + "1" * 64
    with pytest.raises(FactorFiLMContractError, match="run fingerprint mismatch"):
        FactorFiLMRunIdentity.from_dict(payload)

    payload = identity.to_dict()
    payload["train_statistics_fingerprint"] = "sha256:" + "2" * 64
    with pytest.raises(FactorFiLMContractError, match="run fingerprint mismatch"):
        FactorFiLMRunIdentity.from_dict(payload)


def test_target_run_identity_locks_exact_canonical_288_36_episode_contract() -> None:
    identity = _target_identity()

    assert len(identity.ordered_train_episode_indices) == 288
    assert len(identity.ordered_validation_episode_indices) == 36
    with pytest.raises(FactorFiLMContractError, match="exactly 288 train and 36 validation"):
        _target_identity(train_indices=tuple(range(287)))
    with pytest.raises(FactorFiLMContractError, match="strictly increasing and unique"):
        _target_identity(train_indices=(*range(287), 286))
    with pytest.raises(FactorFiLMContractError, match="strictly increasing and unique"):
        _target_identity(validation_indices=tuple(reversed(range(288, 324))))
    with pytest.raises(FactorFiLMContractError, match="episode identity"):
        replace(
            identity,
            data_contract={
                **dict(identity.data_contract),
                "train_episode_indices_fingerprint": _DIGEST_E,
            },
            run_fingerprint="",
        )


def test_validation_selection_requires_exact_20_and_uses_declared_ranking() -> None:
    candidates = _validation_candidates()
    queue = _validation_queue()
    ranked = rank_factor_film_validation_results(candidates)
    selection = create_factor_film_selection(
        validation_queue=queue,
        candidates=candidates,
    )

    assert ranked[0].checkpoint_step == 10_000
    assert ranked[1].checkpoint_step == 15_000
    assert selection.selected_checkpoint_step == 10_000
    assert selection.test_accessed is False
    assert selection.development_accessed_for_selection is False
    assert selection.semantic_audit_accessed_for_selection is False
    assert selection.final_schedule_accessed is False
    assert selection.validation_queue_fingerprint == queue.queue_fingerprint
    assert FactorFiLMSelectionRecord.from_dict(selection.to_dict()) == selection

    with pytest.raises(FactorFiLMContractError, match="all 20 checkpoints"):
        rank_factor_film_validation_results(candidates[:-1])
    with pytest.raises(FactorFiLMContractError, match="different validation schedules"):
        rank_factor_film_validation_results(
            (*candidates[:-1], replace(candidates[-1], schedule_digest=_DIGEST_B))
        )
    with pytest.raises(FactorFiLMContractError, match="validation-only"):
        replace(candidates[0], source_split="test")
    with pytest.raises(FactorFiLMContractError, match="published validation queue"):
        create_factor_film_selection(
            validation_queue=queue,
            candidates=(replace(candidates[0], checkpoint_fingerprint=_DIGEST_E), *candidates[1:]),
        )
    with pytest.raises(FactorFiLMContractError, match="published validation queue"):
        create_factor_film_selection(
            validation_queue=queue,
            candidates=(
                replace(candidates[0], offline_validation_action_loss=0.25),
                *candidates[1:],
            ),
        )


def test_validation_queue_builder_binds_each_checkpoint_offline_loss() -> None:
    identity = _run_identity()
    candidates = _validation_candidates()
    checkpoints = tuple(
        CheckpointRecord(
            checkpoint_fingerprint=value.checkpoint_fingerprint,
            run_fingerprint=identity.run_fingerprint,
            global_step=value.checkpoint_step,
            relative_path=f"checkpoints/step-{value.checkpoint_step:08d}-fixture",
            policy_config_digest=_DIGEST_B,
            statistics_fingerprint=identity.train_statistics_fingerprint,
            split_digest=identity.m3b_split_manifest_digest,
            git_commit=identity.git_commit,
            complete=True,
        )
        for value in candidates
    )
    losses = {
        value.checkpoint_fingerprint: value.offline_validation_action_loss for value in candidates
    }
    queue = build_factor_film_validation_queue(
        identity=identity,
        validation_schedule_digest=candidates[0].schedule_digest,
        checkpoints=checkpoints,
        offline_validation_action_losses=losses,
    )

    assert tuple(value.offline_validation_action_loss for value in queue.items) == tuple(
        value.offline_validation_action_loss for value in candidates
    )
    with pytest.raises(FactorFiLMContractError, match="cover exactly"):
        build_factor_film_validation_queue(
            identity=identity,
            validation_schedule_digest=candidates[0].schedule_digest,
            checkpoints=checkpoints,
            offline_validation_action_losses={
                key: value
                for key, value in losses.items()
                if key != candidates[-1].checkpoint_fingerprint
            },
        )


def test_validation_queue_rejects_development_test_final_and_unsafe_paths() -> None:
    queue = _validation_queue()
    item = queue.items[0]
    assert queue.selection_source == "m3b_validation"
    assert FactorFiLMValidationQueue.from_dict(queue.to_dict()) == queue

    for field in (
        "test_accessed",
        "development_accessed_for_selection",
        "semantic_audit_accessed_for_selection",
        "final_schedule_accessed",
    ):
        with pytest.raises(FactorFiLMContractError, match="forbidden source"):
            replace(queue, **{field: True})
    with pytest.raises(FactorFiLMContractError, match="forbidden split"):
        replace(item, source_split="m42_dev_v0")
    with pytest.raises(FactorFiLMContractError, match="unsafe"):
        replace(item, checkpoint_relative_path="../outside")
    with pytest.raises(FactorFiLMContractError, match="exactly 20"):
        replace(queue, items=queue.items[:-1], queue_fingerprint="")


def test_resume_accepts_only_the_sole_promoted_atomic_orphan(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    declared = "checkpoints/step-00005000-declared"
    orphan = "checkpoints/step-00010000-orphan"
    for relative in (declared, orphan):
        marker = run_root / relative / "complete.json"
        marker.parent.mkdir(parents=True)
        marker.write_text("{}\n", encoding="utf-8")
    manifest = SimpleNamespace(checkpoints=(SimpleNamespace(relative_path=declared),))

    assert (
        _resume_is_atomic_orphan(
            run_root,
            requested=orphan,
            manifest=manifest,
        )
        is True
    )
    with pytest.raises(RuntimeError, match="newer promoted"):
        _resume_is_atomic_orphan(
            run_root,
            requested=declared,
            manifest=manifest,
        )
    second = run_root / "checkpoints/step-00015000-orphan/complete.json"
    second.parent.mkdir(parents=True)
    second.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="multiple promoted"):
        _resume_is_atomic_orphan(
            run_root,
            requested=orphan,
            manifest=manifest,
        )
