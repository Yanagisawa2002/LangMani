from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import torch

from langmani.policies.act_checkpoint import (
    CHECKPOINT_COMPLETION_MARKER,
    CHECKPOINT_MANIFEST,
    PREPROCESSOR_CONFIG,
    ActCheckpointError,
    assert_resume_compatible,
    load_act_checkpoint,
    load_act_checkpoint_manifest,
    save_act_checkpoint,
)
from langmani.policies.act_training import DeterministicResumeBatchSampler
from langmani.policies.act_types import (
    ActRunIdentity,
    ActVariant,
    ExperimentMode,
    TrainingState,
)


def _digest(character: str) -> str:
    return f"sha256:{character * 64}"


def _identity(*, training_seed: int = 7) -> ActRunIdentity:
    return ActRunIdentity(
        m3b_export_fingerprint=_digest("a"),
        m3b_split_manifest_digest=_digest("b"),
        ordered_train_episode_indices=(0, 6),
        ordered_validation_episode_indices=(1,),
        variant=ActVariant.MIXED_UNCONDITIONED,
        task_id=None,
        task_onehot_mapping_version="CanonicalTaskOneHotV0",
        model_config={"chunk_size": 4, "state_dimension": 9},
        data_contract={"fps": 20, "feature_allowlist": ["observation.state", "action"]},
        train_statistics_fingerprint=_digest("c"),
        optimization_config={"optimizer": "adamw", "learning_rate": 1e-5},
        training_seed=training_seed,
        experiment_mode=ExperimentMode.DEVELOPMENT,
        device="cpu",
        dtype="float32",
        lerobot_version="0.6.0",
        torch_version=torch.__version__,
        cuda_version=None,
        git_commit="d" * 40,
        git_dirty=False,
        dirty_development_override=False,
    )


class _FakePolicy:
    load_calls: ClassVar[list[dict[str, object]]] = []

    def __init__(self) -> None:
        self.config = {"type": "fake-act"}
        self.save_calls: list[dict[str, object]] = []

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        push_to_hub: bool = False,
        config_filename: str | None = None,
        **kwargs: object,
    ) -> None:
        self.save_calls.append(
            {
                "push_to_hub": push_to_hub,
                "config_filename": config_filename,
                "kwargs": kwargs,
            }
        )
        root = Path(save_directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / "fake_policy.json").write_text('{"policy":"act"}\n', encoding="utf-8")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        local_files_only: bool,
        strict: bool,
    ) -> _FakePolicy:
        root = Path(pretrained_name_or_path)
        cls.load_calls.append(
            {
                "path": root,
                "local_files_only": local_files_only,
                "strict": strict,
            }
        )
        if not (root / "fake_policy.json").is_file():
            raise FileNotFoundError("fake policy artifact is missing")
        return cls()


class _FakeProcessor:
    def __init__(self, name: str) -> None:
        self.name = name
        self.save_calls: list[dict[str, object]] = []

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        push_to_hub: bool = False,
        config_filename: str | None = None,
        **kwargs: object,
    ) -> None:
        self.save_calls.append(
            {
                "push_to_hub": push_to_hub,
                "config_filename": config_filename,
                "kwargs": kwargs,
            }
        )
        if config_filename is None:
            raise AssertionError("checkpoint must pin processor config filenames")
        root = Path(save_directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / config_filename).write_text(
            json.dumps({"processor": self.name}) + "\n",
            encoding="utf-8",
        )
        (root / f"{self.name}_stats.safetensors").write_bytes(self.name.encode("ascii"))


class _TinyResumePolicy(torch.nn.Module):
    """Small serializable policy used to prove exact CPU resume continuity."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        push_to_hub: bool = False,
        **kwargs: object,
    ) -> None:
        assert push_to_hub is False
        assert not kwargs
        root = Path(save_directory)
        root.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), root / "tiny_policy.pt")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        local_files_only: bool,
        strict: bool,
    ) -> _TinyResumePolicy:
        assert local_files_only
        policy = cls()
        state = torch.load(
            Path(pretrained_name_or_path) / "tiny_policy.pt",
            map_location="cpu",
            weights_only=True,
        )
        policy.load_state_dict(state, strict=strict)
        return policy


def _processor_loader(policy: object, root: Path) -> tuple[object, object]:
    assert isinstance(policy, _FakePolicy)
    assert (root / "policy_preprocessor.json").is_file()
    assert (root / "policy_postprocessor.json").is_file()
    return _FakeProcessor("loaded_pre"), _FakeProcessor("loaded_post")


def _tiny_processor_loader(policy: object, root: Path) -> tuple[object, object]:
    assert isinstance(policy, _TinyResumePolicy)
    assert (root / "policy_preprocessor.json").is_file()
    assert (root / "policy_postprocessor.json").is_file()
    return _FakeProcessor("loaded_pre"), _FakeProcessor("loaded_post")


def _seed_tiny_run(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)


def _train_tiny_steps(
    policy: _TinyResumePolicy,
    optimizer: torch.optim.AdamW,
    *,
    start_step: int,
    target_step: int,
) -> tuple[tuple[int, ...], ...]:
    features = torch.tensor(
        [
            [-1.0, 0.5],
            [-0.5, 1.0],
            [0.0, -1.0],
            [0.5, -0.5],
            [1.0, 0.0],
            [1.5, 0.5],
            [2.0, 1.0],
        ]
    )
    targets = (features[:, :1] * 0.75) - (features[:, 1:] * 0.25)
    sampler = iter(
        DeterministicResumeBatchSampler(
            dataset_size=len(features),
            batch_size=3,
            seed=29,
            start_step=start_step,
        )
    )
    consumed: list[tuple[int, ...]] = []
    for _ in range(start_step, target_step):
        batch = tuple(next(sampler))
        consumed.append(batch)
        indices = torch.tensor(batch, dtype=torch.long)
        python_noise = random.random()
        numpy_noise = float(np.random.random())
        inputs = features[indices] + torch.rand(len(batch), 2) * 0.01
        inputs = inputs + (python_noise + numpy_noise) * 0.001
        loss = torch.nn.functional.mse_loss(policy(inputs), targets[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return tuple(consumed)


def _next_rng_values() -> tuple[float, float, torch.Tensor]:
    return random.random(), float(np.random.random()), torch.rand(4)


def _assert_nested_state_equal(actual: object, expected: object) -> None:
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return
    if isinstance(actual, dict) and isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_nested_state_equal(actual[key], expected[key])
        return
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_state_equal(actual_item, expected_item)
        return
    assert actual == expected


def _optimizer_and_scheduler() -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.StepLR]:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    loss = model(torch.ones(2, 2)).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    return optimizer, scheduler


def _save_fixture(
    root: Path,
    *,
    identity: ActRunIdentity | None = None,
    training_state: TrainingState | None = None,
) -> tuple[object, _FakePolicy, _FakeProcessor, _FakeProcessor]:
    policy = _FakePolicy()
    preprocessor = _FakeProcessor("pre")
    postprocessor = _FakeProcessor("post")
    optimizer, scheduler = _optimizer_and_scheduler()
    record = save_act_checkpoint(
        run_root=root,
        identity=identity or _identity(),
        training_state=training_state or TrainingState(global_step=3, examples_processed=12),
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    return record, policy, preprocessor, postprocessor


def test_checkpoint_is_staged_promoted_manifested_and_strictly_reloaded(tmp_path: Path) -> None:
    _FakePolicy.load_calls.clear()
    identity = _identity()
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    source_optimizer, source_scheduler = _optimizer_and_scheduler()
    policy = _FakePolicy()
    preprocessor = _FakeProcessor("pre")
    postprocessor = _FakeProcessor("post")
    record = save_act_checkpoint(
        run_root=tmp_path,
        identity=identity,
        training_state=TrainingState(global_step=5, examples_processed=40),
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        optimizer=source_optimizer,
        scheduler=source_scheduler,
        training_metric={"step": 5, "total_loss": 0.25, "checkpoint_path": None},
    )
    expected_rng = (random.random(), float(np.random.random()), torch.rand(3))
    checkpoint_root = tmp_path / Path(record.relative_path)

    assert record.complete
    assert checkpoint_root.is_dir()
    assert (checkpoint_root / CHECKPOINT_MANIFEST).is_file()
    assert (checkpoint_root / CHECKPOINT_COMPLETION_MARKER).is_file()
    assert not any((tmp_path / ".checkpoint-staging").iterdir())
    assert policy.save_calls == [{"push_to_hub": False, "config_filename": None, "kwargs": {}}]
    assert preprocessor.save_calls[0]["push_to_hub"] is False
    assert preprocessor.save_calls[0]["config_filename"] == PREPROCESSOR_CONFIG
    assert postprocessor.save_calls[0]["push_to_hub"] is False

    metadata = load_act_checkpoint_manifest(
        run_root=tmp_path,
        checkpoint_relative_path=record.relative_path,
        expected_identity=identity,
    )
    assert metadata.record == record
    assert metadata.training_state.global_step == 5
    assert metadata.training_metric == {
        "step": 5,
        "total_loss": 0.25,
        "checkpoint_path": None,
    }
    assert _FakePolicy.load_calls == []

    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    resumed_optimizer, resumed_scheduler = _optimizer_and_scheduler()
    loaded = load_act_checkpoint(
        run_root=tmp_path,
        checkpoint_relative_path=record.relative_path,
        expected_identity=identity,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        for_resume=True,
        restore_rng=True,
        policy_class=_FakePolicy,
        processor_loader=_processor_loader,
    )

    assert loaded.training_state.global_step == 5
    assert loaded.record == record
    assert loaded.training_metric == {"step": 5, "total_loss": 0.25, "checkpoint_path": None}
    assert _FakePolicy.load_calls[-1]["local_files_only"] is True
    assert _FakePolicy.load_calls[-1]["strict"] is True
    assert len(loaded.optimizer.state_dict()["state"]) > 0  # type: ignore[union-attr]
    actual_rng = (random.random(), float(np.random.random()), torch.rand(3))
    assert actual_rng[0] == expected_rng[0]
    assert actual_rng[1] == expected_rng[1]
    torch.testing.assert_close(actual_rng[2], expected_rng[2], rtol=0, atol=0)


def test_resume_matches_uninterrupted_sampler_parameters_optimizer_and_rng(
    tmp_path: Path,
) -> None:
    total_steps = 8
    checkpoint_step = 4

    _seed_tiny_run(31)
    uninterrupted_policy = _TinyResumePolicy()
    uninterrupted_optimizer = torch.optim.AdamW(uninterrupted_policy.parameters(), lr=2e-3)
    uninterrupted_batches = _train_tiny_steps(
        uninterrupted_policy,
        uninterrupted_optimizer,
        start_step=0,
        target_step=total_steps,
    )
    uninterrupted_rng = _next_rng_values()

    _seed_tiny_run(31)
    interrupted_policy = _TinyResumePolicy()
    interrupted_optimizer = torch.optim.AdamW(interrupted_policy.parameters(), lr=2e-3)
    before_checkpoint_batches = _train_tiny_steps(
        interrupted_policy,
        interrupted_optimizer,
        start_step=0,
        target_step=checkpoint_step,
    )
    identity = _identity(training_seed=31)
    record = save_act_checkpoint(
        run_root=tmp_path,
        identity=identity,
        training_state=TrainingState(
            global_step=checkpoint_step,
            examples_processed=sum(len(batch) for batch in before_checkpoint_batches),
        ),
        policy=interrupted_policy,
        preprocessor=_FakeProcessor("pre"),
        postprocessor=_FakeProcessor("post"),
        optimizer=interrupted_optimizer,
    )

    _seed_tiny_run(999)
    resumed_policy = _TinyResumePolicy()
    resumed_optimizer = torch.optim.AdamW(resumed_policy.parameters(), lr=2e-3)
    loaded = load_act_checkpoint(
        run_root=tmp_path,
        checkpoint_relative_path=record.relative_path,
        expected_identity=identity,
        optimizer=resumed_optimizer,
        for_resume=True,
        policy_class=_TinyResumePolicy,
        processor_loader=_tiny_processor_loader,
        existing_policy=resumed_policy,
    )
    assert isinstance(loaded.policy, _TinyResumePolicy)
    assert isinstance(loaded.optimizer, torch.optim.AdamW)
    after_checkpoint_batches = _train_tiny_steps(
        loaded.policy,
        loaded.optimizer,
        start_step=loaded.training_state.global_step,
        target_step=total_steps,
    )
    resumed_rng = _next_rng_values()

    assert before_checkpoint_batches + after_checkpoint_batches == uninterrupted_batches
    _assert_nested_state_equal(loaded.policy.state_dict(), uninterrupted_policy.state_dict())
    _assert_nested_state_equal(
        loaded.optimizer.state_dict(),
        uninterrupted_optimizer.state_dict(),
    )
    assert resumed_rng[:2] == uninterrupted_rng[:2]
    torch.testing.assert_close(resumed_rng[2], uninterrupted_rng[2], rtol=0, atol=0)


def test_checkpoint_semantic_mismatch_is_rejected_before_policy_load(tmp_path: Path) -> None:
    record, _, _, _ = _save_fixture(tmp_path)
    _FakePolicy.load_calls.clear()
    incompatible = _identity(training_seed=8)

    with pytest.raises(ActCheckpointError, match="semantic identity"):
        load_act_checkpoint(
            run_root=tmp_path,
            checkpoint_relative_path=record.relative_path,
            expected_identity=incompatible,
            policy_class=_FakePolicy,
            processor_loader=_processor_loader,
        )

    assert not _FakePolicy.load_calls
    with pytest.raises(ActCheckpointError, match="resume identity mismatch"):
        assert_resume_compatible(incompatible, _identity().to_dict())


@pytest.mark.parametrize(
    "unsafe_path",
    ("../escape", "/absolute", "checkpoints\\windows", "checkpoints/./alias"),
)
def test_checkpoint_load_rejects_unsafe_paths(tmp_path: Path, unsafe_path: str) -> None:
    with pytest.raises(ActCheckpointError, match="path"):
        load_act_checkpoint(
            run_root=tmp_path,
            checkpoint_relative_path=unsafe_path,
            expected_identity=_identity(),
            policy_class=_FakePolicy,
            processor_loader=_processor_loader,
        )


def test_completed_checkpoint_step_and_run_are_immutable(tmp_path: Path) -> None:
    record, _, _, _ = _save_fixture(tmp_path)
    assert record.complete

    with pytest.raises(ActCheckpointError, match="already exists"):
        _save_fixture(tmp_path)

    (tmp_path / "complete.json").write_text("{}\n", encoding="utf-8")
    optimizer, scheduler = _optimizer_and_scheduler()
    with pytest.raises(ActCheckpointError, match="completed ACT runs are immutable"):
        save_act_checkpoint(
            run_root=tmp_path,
            identity=_identity(),
            training_state=TrainingState(global_step=4, examples_processed=16),
            policy=_FakePolicy(),
            preprocessor=_FakeProcessor("pre"),
            postprocessor=_FakeProcessor("post"),
            optimizer=optimizer,
            scheduler=scheduler,
        )


def test_completed_training_state_can_be_loaded_for_inference_but_not_resumed(
    tmp_path: Path,
) -> None:
    identity = _identity()
    record, _, _, _ = _save_fixture(
        tmp_path,
        identity=identity,
        training_state=TrainingState(global_step=3, examples_processed=12, completed=True),
    )
    loaded = load_act_checkpoint(
        run_root=tmp_path,
        checkpoint_relative_path=record.relative_path,
        expected_identity=identity,
        policy_class=_FakePolicy,
        processor_loader=_processor_loader,
    )
    assert loaded.training_state.completed

    optimizer, scheduler = _optimizer_and_scheduler()
    with pytest.raises(ActCheckpointError, match="cannot be resumed"):
        load_act_checkpoint(
            run_root=tmp_path,
            checkpoint_relative_path=record.relative_path,
            expected_identity=identity,
            optimizer=optimizer,
            scheduler=scheduler,
            for_resume=True,
            policy_class=_FakePolicy,
            processor_loader=_processor_loader,
        )


def test_artifact_corruption_and_scheduler_mismatch_are_rejected(tmp_path: Path) -> None:
    identity = _identity()
    record, _, _, _ = _save_fixture(tmp_path, identity=identity)
    optimizer, _ = _optimizer_and_scheduler()
    with pytest.raises(ActCheckpointError, match="scheduler state is incompatible"):
        load_act_checkpoint(
            run_root=tmp_path,
            checkpoint_relative_path=record.relative_path,
            expected_identity=identity,
            optimizer=optimizer,
            scheduler=None,
            for_resume=True,
            policy_class=_FakePolicy,
            processor_loader=_processor_loader,
        )

    artifact = tmp_path / Path(record.relative_path) / "pretrained_model" / "fake_policy.json"
    artifact.write_text("corrupt\n", encoding="utf-8")
    with pytest.raises(ActCheckpointError, match="artifact integrity mismatch"):
        load_act_checkpoint(
            run_root=tmp_path,
            checkpoint_relative_path=record.relative_path,
            expected_identity=identity,
            policy_class=_FakePolicy,
            processor_loader=_processor_loader,
        )


def test_training_state_mismatch_changes_checkpoint_fingerprint(tmp_path: Path) -> None:
    identity = _identity()
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    first, _, _, _ = _save_fixture(tmp_path / "first", identity=identity)
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    second, _, _, _ = _save_fixture(
        tmp_path / "second",
        identity=identity,
        training_state=replace(
            TrainingState(global_step=3, examples_processed=12),
            global_step=4,
        ),
    )
    assert first.checkpoint_fingerprint != second.checkpoint_fingerprint
