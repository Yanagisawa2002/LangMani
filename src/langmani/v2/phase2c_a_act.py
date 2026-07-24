"""Maintained LeRobot 0.6 ACT integration for Phase 2C-A.

The installed ACT core is never patched or subclassed.  This module provides
only the accepted-dataset view, a three-way public ``FeatureType.ENV`` task
token, exact frozen statistics, bounded training orchestration, and offline
diagnostics.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act import ACTConfig, ACTPolicy
from lerobot.processor import DataProcessorPipeline, PolicyProcessorPipeline
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from langmani.datasets.identity import sha256_hex
from langmani.datasets.lerobot_types import (
    ACTION_FEATURE_KEY,
    IMAGE_FEATURE_KEY,
    STATE_FEATURE_KEY,
)
from langmani.policies.act_checkpoint import load_act_checkpoint, save_act_checkpoint
from langmani.policies.act_training import (
    DeterministicResumeBatchSampler,
    build_act_config,
    build_policy_and_processors,
    image_to_policy_float,
    seed_dataloader_worker,
    seed_everything,
)
from langmani.policies.act_types import (
    ActModelConfig,
    ActOptimizationConfig,
    TrainingState,
)
from langmani.v2.phase2c_a import (
    ACTION_DIMENSION,
    CHUNK_SIZE,
    CONTROL_FREQUENCY_HZ,
    NORMALIZATION_FINGERPRINT,
    NORMALIZATION_IDENTITY,
    PACKAGE_FINGERPRINT,
    POLICY_FEATURE_ALLOWLIST,
    PRIVILEGED_FEATURE_DENYLIST,
    TASK_IDS,
    TASK_SLUGS,
    TRAIN_FRAMES,
    ModelKind,
    Phase2CAModelConfig,
    Phase2CAOptimizationConfig,
    Phase2CARunIdentity,
    UniformTaskBatchSampler,
    validate_consumed_training_samples,
)

TASK_INDEX_KEY = "_phase2c_a_task_index"
TASK_TOKEN_FEATURE_KEY = "observation.environment_state"
SOURCE_SPLIT_KEY = "_phase2c_a_split"
MODEL_CONFIG_FILE = "phase2c_a_model_config.json"
RUN_MANIFEST_FILE = "run_manifest.json"
TRAINING_METRICS_FILE = "metrics.jsonl"
TRAINING_SUMMARY_FILE = "training_summary.json"


class Phase2CAActError(RuntimeError):
    """Raised when installed ACT or accepted data violates Phase 2C-A."""


def act_config_dict(config: ACTConfig) -> dict[str, object]:
    """Return the installed dataclass config as deterministic JSON-safe data."""

    def convert(value: object) -> object:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Mapping):
            return {
                str(convert(key)): convert(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, tuple | list):
            return [convert(item) for item in value]
        if value is None or isinstance(value, str | int | float | bool):
            return value
        raise Phase2CAActError(f"ACT config contains a non-JSON value: {type(value)!r}")

    converted = convert(asdict(config))
    if not isinstance(converted, dict):
        raise Phase2CAActError("ACT config did not serialize to an object")
    return converted


@dataclass(frozen=True, slots=True)
class LoadedPhase2CAView:
    dataset: Dataset[Mapping[str, object]]
    task_lengths: Mapping[str, int]
    task_roots: Mapping[str, str]
    split: str


@dataclass(frozen=True, slots=True)
class ProcessorStatistics:
    tensors: Mapping[str, Mapping[str, torch.Tensor]]
    manifest: Mapping[str, object]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    run_root: str
    run_fingerprint: str
    final_step: int
    examples_processed: int
    effective_samples_by_task: Mapping[str, int]
    checkpoint_records: tuple[Mapping[str, object], ...]
    training_duration_s: float
    peak_gpu_allocated_bytes: int
    peak_gpu_reserved_bytes: int
    mean_examples_per_second: float

    def to_dict(self) -> dict[str, object]:
        return {
            "run_root": self.run_root,
            "run_fingerprint": self.run_fingerprint,
            "final_step": self.final_step,
            "examples_processed": self.examples_processed,
            "effective_samples_by_task": dict(self.effective_samples_by_task),
            "checkpoint_records": [dict(item) for item in self.checkpoint_records],
            "training_duration_s": self.training_duration_s,
            "peak_gpu_allocated_bytes": self.peak_gpu_allocated_bytes,
            "peak_gpu_reserved_bytes": self.peak_gpu_reserved_bytes,
            "mean_examples_per_second": self.mean_examples_per_second,
        }


class _TaskAnnotatedDataset(Dataset[Mapping[str, object]]):
    def __init__(self, dataset: Dataset[Mapping[str, object]], task_id: str, split: str) -> None:
        if task_id not in TASK_IDS:
            raise Phase2CAActError(f"unknown task dataset {task_id!r}")
        self.dataset = dataset
        self.task_id = task_id
        self.task_index = TASK_IDS.index(task_id)
        self.split = split

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Mapping[str, object]:
        raw = self.dataset[index]
        if not isinstance(raw, Mapping):
            raise Phase2CAActError("LeRobot row must be a mapping")
        result = dict(raw)
        result[TASK_INDEX_KEY] = torch.tensor(self.task_index, dtype=torch.int64)
        result[SOURCE_SPLIT_KEY] = self.split
        return result


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2CAActError(f"could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise Phase2CAActError(f"{path} must contain one object")
    return value


def _write_new_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise Phase2CAActError(f"refusing to overwrite immutable artifact {path}") from error


def _task_root(primary_root: Path, task_id: str, split: str) -> Path:
    if task_id not in TASK_IDS or split not in {
        "train",
        "validation",
        "test_unseen_reset",
        "test_unseen_task_language",
        "test_visual_shift",
    }:
        raise Phase2CAActError("unknown task/split dataset root")
    return primary_root / "task_roots" / TASK_SLUGS[task_id] / "splits" / split


def _repo_id(root: Path, *, task_id: str, split: str) -> str:
    manifest = _read_object(root / "langmani_phase2b6_split_manifest.json")
    repo_id = manifest.get("repo_id")
    if (
        not isinstance(repo_id, str)
        or not repo_id
        or manifest.get("task_id") != task_id
        or manifest.get("primary_split") != split
    ):
        raise Phase2CAActError(f"{root} has no matching accepted LeRobot split identity")
    return repo_id


def load_dataset_view(
    primary_root: str | Path,
    *,
    model_kind: ModelKind | str,
    split: str,
    chunk_size: int = CHUNK_SIZE,
) -> LoadedPhase2CAView:
    """Load accepted split roots through the real LeRobot 0.6 API."""

    if chunk_size != CHUNK_SIZE:
        raise Phase2CAActError("Phase 2C-A does not permit a chunk-size sweep")
    primary = Path(primary_root).resolve()
    kind = ModelKind(model_kind)
    task_ids = kind.compatible_task_ids if kind is not ModelKind.SHARED else TASK_IDS
    delta_timestamps = {
        ACTION_FEATURE_KEY: [index / CONTROL_FREQUENCY_HZ for index in range(chunk_size)]
    }
    datasets: list[_TaskAnnotatedDataset] = []
    lengths: dict[str, int] = {}
    roots: dict[str, str] = {}
    for task_id in task_ids:
        root = _task_root(primary, task_id, split)
        if not root.is_dir():
            raise Phase2CAActError(f"accepted LeRobot split root is missing: {root}")
        dataset = LeRobotDataset(
            repo_id=_repo_id(root, task_id=task_id, split=split),
            root=root,
            delta_timestamps=delta_timestamps,
            video_backend="pyav",
            return_uint8=True,
            download_videos=False,
        )
        annotated = _TaskAnnotatedDataset(dataset, task_id, split)
        datasets.append(annotated)
        lengths[task_id] = len(annotated)
        roots[task_id] = root.as_posix()
    combined: Dataset[Mapping[str, object]]
    if len(datasets) == 1:
        combined = datasets[0]
    else:
        combined = cast(Dataset[Mapping[str, object]], ConcatDataset(datasets))
    return LoadedPhase2CAView(
        dataset=combined,
        task_lengths=lengths,
        task_roots=roots,
        split=split,
    )


def audit_consumer_view(
    primary_root: str | Path,
    *,
    package_fingerprint: str = PACKAGE_FINGERPRINT,
) -> dict[str, object]:
    """Read boundary samples from all train/validation roots and validate H=16 padding."""

    if package_fingerprint != PACKAGE_FINGERPRINT:
        raise Phase2CAActError("consumer audit requested a noncanonical package")
    reports: list[dict[str, object]] = []
    for split in ("train", "validation"):
        for kind in (ModelKind.PICK, ModelKind.STACK, ModelKind.PUSH):
            task_id = str(kind.task_id)
            view = load_dataset_view(primary_root, model_kind=kind, split=split)
            if len(view.dataset) != (
                TRAIN_FRAMES[task_id]
                if split == "train"
                else {
                    "PickCube-v1": 7_803,
                    "StackCube-v1": 10_555,
                    "PushCube-v1": 6_893,
                }[task_id]
            ):
                raise Phase2CAActError(f"{task_id}/{split} frame count changed")
            indices = (0, len(view.dataset) - 1)
            samples = []
            for index in indices:
                row = view.dataset[index]
                image = row.get(IMAGE_FEATURE_KEY)
                state = row.get(STATE_FEATURE_KEY)
                action = row.get(ACTION_FEATURE_KEY)
                padding = row.get("action_is_pad")
                timestamp = row.get("timestamp")
                if (
                    not isinstance(image, torch.Tensor)
                    or image.dtype is not torch.uint8
                    or tuple(image.shape) != (3, 256, 256)
                    or not isinstance(state, torch.Tensor)
                    or state.dtype is not torch.float32
                    or tuple(state.shape) != (9,)
                    or not isinstance(action, torch.Tensor)
                    or action.dtype is not torch.float32
                    or tuple(action.shape) != (CHUNK_SIZE, ACTION_DIMENSION)
                    or not isinstance(padding, torch.Tensor)
                    or padding.dtype is not torch.bool
                    or tuple(padding.shape) != (CHUNK_SIZE,)
                    or bool(padding[0])
                ):
                    raise Phase2CAActError(f"{task_id}/{split} sample contract failed")
                if not torch.isfinite(state).all() or not torch.isfinite(action).all():
                    raise Phase2CAActError("consumer view contains nonfinite state/action values")
                samples.append(
                    {
                        "dataset_index": index,
                        "episode_index": int(cast(torch.Tensor, row["episode_index"]).item()),
                        "frame_index": int(cast(torch.Tensor, row["frame_index"]).item()),
                        "timestamp": float(cast(torch.Tensor, timestamp).item()),
                        "padded_timestep_count": int(torch.count_nonzero(padding)),
                    }
                )
            reports.append(
                {
                    "task_id": task_id,
                    "split": split,
                    "root": view.task_roots[task_id],
                    "frame_count": len(view.dataset),
                    "boundary_samples": samples,
                }
            )
    semantic = {
        "package_fingerprint": package_fingerprint,
        "lerobot_version": _distribution_version("lerobot"),
        "video_backend": "pyav",
        "download_videos": False,
        "return_uint8": True,
        "chunk_size": CHUNK_SIZE,
        "reports": reports,
        "all_train_roots_loaded": True,
        "all_validation_roots_loaded": True,
        "images_decoded": True,
        "state_shape": [9],
        "action_shape": [CHUNK_SIZE, ACTION_DIMENSION],
        "padding_mask_dtype": "bool",
        "excluded_episodes_absent": True,
        "privileged_policy_fields": [],
    }
    return {
        "schema_version": "langmani-v2-phase2c-a-consumer-audit-v0",
        **semantic,
        "fingerprint": f"sha256:{sha256_hex(semantic)}",
        "passed": True,
    }


def _distribution_version(name: str) -> str:
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not_installed"


def _stat_record(
    raw: Mapping[str, object],
    *,
    feature_name: str,
    image: bool = False,
    std_floor: float = 1e-8,
) -> dict[str, torch.Tensor]:
    keys = {
        "min": "minimum",
        "max": "maximum",
        "mean": "mean",
        "std": "std",
    }
    result: dict[str, torch.Tensor] = {}
    expected_dimension = 3 if image else (9 if feature_name == STATE_FEATURE_KEY else 8)
    for target, source in keys.items():
        value = raw.get(source)
        if not isinstance(value, list) or len(value) != expected_dimension:
            raise Phase2CAActError(f"{feature_name} {source} statistics are malformed")
        tensor = torch.tensor(value, dtype=torch.float32)
        if target == "std":
            tensor = tensor.clamp_min(std_floor)
        if image:
            tensor = tensor.reshape(expected_dimension, 1, 1)
        result[target] = tensor
    count = raw.get("count")
    if not isinstance(count, int) or count < 1:
        raise Phase2CAActError(f"{feature_name} statistics count is malformed")
    result["count"] = torch.tensor([count], dtype=torch.int64)
    return result


def load_processor_statistics(
    statistics_path: str | Path,
    normalization_path: str | Path,
    *,
    model_kind: ModelKind | str,
) -> ProcessorStatistics:
    """Load accepted train-only statistics without scanning media."""

    statistics = _read_object(Path(statistics_path))
    normalization = _read_object(Path(normalization_path))
    if (
        statistics.get("passed") is not True
        or statistics.get("validation_or_test_in_train_normalization") is not False
        or normalization.get("passed") is not True
        or normalization.get("fingerprint") != NORMALIZATION_FINGERPRINT
        or normalization.get("canonical_identity") != NORMALIZATION_IDENTITY
        or normalization.get("validation_episode_count") != 0
        or normalization.get("test_episode_count") != 0
    ):
        raise Phase2CAActError("accepted train-only normalization identity changed")
    train_only = statistics.get("train_only")
    if not isinstance(train_only, Mapping):
        raise Phase2CAActError("dataset statistics train_only record is missing")
    pooled = train_only.get("pooled_natural_frame_frequency")
    per_task = train_only.get("per_task")
    if not isinstance(pooled, Mapping) or not isinstance(per_task, Mapping):
        raise Phase2CAActError("dataset statistics accepted views are missing")
    image_stats = pooled.get("image_rgb_unit_interval")
    if not isinstance(image_stats, Mapping):
        raise Phase2CAActError("accepted pooled image statistics are missing")
    kind = ModelKind(model_kind)
    selected: Mapping[str, object]
    selection: str
    if kind is ModelKind.SHARED:
        selected = pooled
        selection = "pooled_natural_training_frame_frequency"
    else:
        task_id = str(kind.task_id)
        raw_selected = per_task.get(task_id)
        if not isinstance(raw_selected, Mapping):
            raise Phase2CAActError(f"accepted task statistics are missing for {task_id}")
        selected = raw_selected
        selection = f"task_specific_train_only:{task_id}"
    state_stats = selected.get("state")
    action_stats = selected.get("action")
    if not isinstance(state_stats, Mapping) or not isinstance(action_stats, Mapping):
        raise Phase2CAActError("selected state/action statistics are missing")
    tensors = {
        IMAGE_FEATURE_KEY: _stat_record(image_stats, feature_name=IMAGE_FEATURE_KEY, image=True),
        STATE_FEATURE_KEY: _stat_record(state_stats, feature_name=STATE_FEATURE_KEY),
        ACTION_FEATURE_KEY: _stat_record(action_stats, feature_name=ACTION_FEATURE_KEY),
    }
    manifest = {
        "schema_version": "langmani-v2-phase2c-a-processor-statistics-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "model_kind": kind.value,
        "selection": selection,
        "image_selection": "pooled_natural_training_frame_frequency",
        "normalization_identity": NORMALIZATION_IDENTITY,
        "normalization_fingerprint": NORMALIZATION_FINGERPRINT,
        "dataset_statistics_fingerprint": statistics.get("fingerprint"),
        "standard_deviation_floor": 1e-8,
        "state": dict(state_stats),
        "action": dict(action_stats),
        "image_rgb_unit_interval": dict(image_stats),
        "validation_or_test_rows_used": False,
    }
    fingerprint = f"sha256:{sha256_hex(manifest)}"
    return ProcessorStatistics(tensors=tensors, manifest=manifest, fingerprint=fingerprint)


def build_phase2c_a_act_config(
    model_kind: ModelKind | str,
    *,
    device: str,
    use_amp: bool,
    optimization: Phase2CAOptimizationConfig,
) -> ACTConfig:
    kind = ModelKind(model_kind)
    portable = Phase2CAModelConfig.for_model(kind)
    model = ActModelConfig(
        state_dimension=portable.state_dimension,
        chunk_size=portable.chunk_size,
        n_action_steps=portable.n_action_steps,
        vision_backbone=portable.vision_backbone,
        pretrained_backbone_weights=portable.pretrained_backbone_weights,
        dim_model=portable.dim_model,
        n_heads=portable.n_heads,
        dim_feedforward=portable.dim_feedforward,
        n_encoder_layers=portable.n_encoder_layers,
        n_decoder_layers=portable.n_decoder_layers,
        use_vae=portable.use_vae,
        latent_dim=portable.latent_dim,
        n_vae_encoder_layers=portable.n_vae_encoder_layers,
        dropout=portable.dropout,
        kl_weight=portable.kl_weight,
    )
    legacy_optimization = ActOptimizationConfig(
        learning_rate=optimization.learning_rate,
        backbone_learning_rate=optimization.backbone_learning_rate,
        weight_decay=optimization.weight_decay,
        scheduler=optimization.scheduler,
        warmup_steps=optimization.warmup_steps,
        gradient_clip_norm=optimization.gradient_clip_norm,
        mixed_precision=optimization.precision,
        batch_size=optimization.batch_size,
        dataloader_workers=optimization.dataloader_workers,
        training_steps=optimization.training_steps,
        checkpoint_interval=max(1, optimization.checkpoint_steps[0]),
        validation_interval=max(1, optimization.validation_steps[0]),
    )
    base = build_act_config(
        model,
        device=device,
        use_amp=use_amp,
        optimization=legacy_optimization,
    )
    if kind is not ModelKind.SHARED:
        return base
    features = dict(base.input_features)
    features[TASK_TOKEN_FEATURE_KEY] = PolicyFeature(FeatureType.ENV, (len(TASK_IDS),))
    normalization_mapping = {
        (key.value if isinstance(key, FeatureType) else str(key)): NormalizationMode(value)
        for key, value in base.normalization_mapping.items()
    }
    normalization_mapping[FeatureType.ENV.value] = NormalizationMode.IDENTITY
    config = replace(
        base,
        input_features=features,
        normalization_mapping=normalization_mapping,
    )
    validate_shared_act_config(config)
    return config


def validate_shared_act_config(config: ACTConfig, policy: ACTPolicy | None = None) -> None:
    env_features = {
        key: value for key, value in config.input_features.items() if value.type is FeatureType.ENV
    }
    state_features = [
        value for value in config.input_features.values() if value.type is FeatureType.STATE
    ]
    if env_features != {TASK_TOKEN_FEATURE_KEY: PolicyFeature(FeatureType.ENV, (len(TASK_IDS),))}:
        raise Phase2CAActError("shared ACT lacks the exact three-way public ENV feature")
    if len(state_features) != 1 or state_features[0].shape != (9,):
        raise Phase2CAActError("shared ACT changed PandaPolicyStateV0")
    normalized = {
        FeatureType(key) if isinstance(key, str) else key: NormalizationMode(value)
        for key, value in config.normalization_mapping.items()
    }
    if normalized.get(FeatureType.ENV) is not NormalizationMode.IDENTITY:
        raise Phase2CAActError("shared ACT task identity must use IDENTITY normalization")
    if policy is not None:
        projection = getattr(policy.model, "encoder_env_state_input_proj", None)
        positions = getattr(policy.model, "encoder_1d_feature_pos_embed", None)
        if not isinstance(projection, torch.nn.Linear) or (
            projection.in_features,
            projection.out_features,
        ) != (len(TASK_IDS), config.dim_model):
            raise Phase2CAActError("installed ACT ENV projection changed")
        if not isinstance(positions, torch.nn.Embedding) or positions.num_embeddings != 3:
            raise Phase2CAActError("installed ACT ENV encoder-token allocation changed")


def _task_indices(raw_batch: Mapping[str, object]) -> torch.Tensor:
    value = raw_batch.get(TASK_INDEX_KEY)
    if not isinstance(value, torch.Tensor):
        raise Phase2CAActError("training batch lacks its nonprivileged task identity")
    flattened = value.reshape(-1)
    if flattened.dtype.is_floating_point or flattened.dtype is torch.bool:
        raise Phase2CAActError("task index must use an integer dtype")
    if bool(torch.any(flattened < 0)) or bool(torch.any(flattened >= len(TASK_IDS))):
        raise Phase2CAActError("task index is outside the three-way mapping")
    return flattened.to(dtype=torch.int64)


def prepare_policy_batch(
    raw_batch: Mapping[str, object],
    *,
    model_kind: ModelKind | str,
    include_action: bool = True,
) -> dict[str, torch.Tensor]:
    """Project exactly the policy allowlist and optional three-way task token."""

    kind = ModelKind(model_kind)
    image = raw_batch.get(IMAGE_FEATURE_KEY)
    state = raw_batch.get(STATE_FEATURE_KEY)
    if not isinstance(image, torch.Tensor) or not isinstance(state, torch.Tensor):
        raise Phase2CAActError("policy batch lacks tensor image/state")
    if state.dtype is not torch.float32 or state.ndim not in (1, 2) or state.shape[-1] != 9:
        raise Phase2CAActError("PandaPolicyStateV0 must be float32[...,9]")
    batch: dict[str, torch.Tensor] = {
        IMAGE_FEATURE_KEY: image_to_policy_float(image),
        STATE_FEATURE_KEY: state,
    }
    if kind is ModelKind.SHARED:
        indices = _task_indices(raw_batch)
        onehot = torch.nn.functional.one_hot(indices, num_classes=len(TASK_IDS)).to(
            dtype=torch.float32
        )
        if state.ndim == 1:
            onehot = onehot.reshape(len(TASK_IDS))
        batch[TASK_TOKEN_FEATURE_KEY] = onehot
    if include_action:
        action = raw_batch.get(ACTION_FEATURE_KEY)
        padding = raw_batch.get("action_is_pad")
        if (
            not isinstance(action, torch.Tensor)
            or action.dtype is not torch.float32
            or action.shape[-2:] != (CHUNK_SIZE, ACTION_DIMENSION)
            or not isinstance(padding, torch.Tensor)
            or padding.dtype is not torch.bool
            or padding.shape[-1:] != (CHUNK_SIZE,)
        ):
            raise Phase2CAActError("training batch action/padding contract failed")
        if bool(torch.any(padding[..., 0])):
            raise Phase2CAActError("the current action cannot be padded")
        batch[ACTION_FEATURE_KEY] = action
        batch["action_is_pad"] = padding
    if set(batch) - (
        POLICY_FEATURE_ALLOWLIST | ({TASK_TOKEN_FEATURE_KEY} if kind is ModelKind.SHARED else set())
    ):
        raise Phase2CAActError("unexpected feature crossed the policy boundary")
    if any(key in batch for key in PRIVILEGED_FEATURE_DENYLIST):
        raise Phase2CAActError("privileged simulator data entered the policy batch")
    return batch


def validate_processed_batch(
    batch: Mapping[str, object],
    *,
    model_kind: ModelKind | str,
    include_action: bool = True,
) -> None:
    kind = ModelKind(model_kind)
    expected = {
        IMAGE_FEATURE_KEY: (3, 256, 256),
        STATE_FEATURE_KEY: (9,),
    }
    if include_action:
        expected[ACTION_FEATURE_KEY] = (CHUNK_SIZE, ACTION_DIMENSION)
        expected["action_is_pad"] = (CHUNK_SIZE,)
    if kind is ModelKind.SHARED:
        expected[TASK_TOKEN_FEATURE_KEY] = (len(TASK_IDS),)
    for key, trailing in expected.items():
        value = batch.get(key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape[-len(trailing) :]) != trailing:
            raise Phase2CAActError(f"processed {key} does not end with {trailing}")
        if value.dtype is not torch.bool and not bool(torch.isfinite(value).all()):
            raise Phase2CAActError(f"processed {key} contains nonfinite values")


def masked_l1_loss(
    predicted: torch.Tensor,
    expected: torch.Tensor,
    action_is_pad: torch.Tensor,
) -> torch.Tensor:
    """Reference masked ACT action loss used by the independent audit."""

    if (
        predicted.shape != expected.shape
        or predicted.ndim != 3
        or predicted.shape[-2:] != (CHUNK_SIZE, ACTION_DIMENSION)
        or action_is_pad.dtype is not torch.bool
        or action_is_pad.shape != predicted.shape[:2]
    ):
        raise Phase2CAActError("masked L1 inputs have incompatible shapes")
    valid = (~action_is_pad).unsqueeze(-1)
    denominator = valid.sum() * ACTION_DIMENSION
    return (torch.abs(predicted - expected) * valid).sum() / denominator.clamp_min(1)


def _make_dataloader(
    view: LoadedPhase2CAView,
    *,
    model_kind: ModelKind,
    optimization: Phase2CAOptimizationConfig,
    start_step: int = 0,
) -> DataLoader[Mapping[str, object]]:
    if model_kind is ModelKind.SHARED:
        sampler: Iterable[list[int]] = UniformTaskBatchSampler(
            {task_id: view.task_lengths[task_id] for task_id in TASK_IDS},
            batch_size=optimization.batch_size,
            seed=optimization.seed,
            start_batch=start_step,
        )
    else:
        sampler = DeterministicResumeBatchSampler(
            dataset_size=len(view.dataset),
            batch_size=optimization.batch_size,
            seed=optimization.seed,
            start_step=start_step,
        )
    generator = torch.Generator().manual_seed(optimization.seed)
    return DataLoader(
        view.dataset,
        batch_sampler=cast(Any, sampler),
        num_workers=optimization.dataloader_workers,
        pin_memory=True,
        persistent_workers=optimization.dataloader_workers > 0,
        worker_init_fn=seed_dataloader_worker,
        generator=generator,
    )


def _autocast(precision: str, device: str) -> Any:
    from contextlib import nullcontext

    if precision == "none":
        return nullcontext()
    if precision != "bfloat16" or device != "cuda":
        raise Phase2CAActError("Phase 2C-A supports only BF16 CUDA or explicit no-AMP")
    if not torch.cuda.is_bf16_supported():
        raise Phase2CAActError("selected CUDA device does not support BF16")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _optimizer(
    policy: ACTPolicy,
    optimization: Phase2CAOptimizationConfig,
) -> torch.optim.AdamW:
    groups = policy.get_optim_params()
    if len(groups) != 2:
        raise Phase2CAActError("installed ACT optimizer groups changed")
    groups[1]["lr"] = optimization.backbone_learning_rate
    return torch.optim.AdamW(
        groups,
        lr=optimization.learning_rate,
        weight_decay=optimization.weight_decay,
    )


def _task_counts(raw_batch: Mapping[str, object], model_kind: ModelKind) -> Counter[str]:
    if model_kind is ModelKind.SHARED:
        indices = _task_indices(raw_batch).detach().cpu().tolist()
        return Counter(TASK_IDS[int(index)] for index in indices)
    state = raw_batch.get(STATE_FEATURE_KEY)
    if not isinstance(state, torch.Tensor):
        raise Phase2CAActError("batch lacks state for sample accounting")
    return Counter({str(model_kind.task_id): int(state.shape[0])})


def create_policy_and_processors(
    *,
    model_kind: ModelKind | str,
    optimization: Phase2CAOptimizationConfig,
    statistics: ProcessorStatistics,
    device: str,
) -> tuple[ACTPolicy, PolicyProcessorPipeline, PolicyProcessorPipeline, ACTConfig]:
    kind = ModelKind(model_kind)
    config = build_phase2c_a_act_config(
        kind,
        device=device,
        use_amp=optimization.precision == "bfloat16",
        optimization=optimization,
    )
    policy, preprocessor, postprocessor = build_policy_and_processors(
        config,
        statistics.tensors,
    )
    if kind is ModelKind.SHARED:
        validate_shared_act_config(config, policy)
    return policy, preprocessor, postprocessor, config


def _postprocess_action_chunk(
    postprocessor: PolicyProcessorPipeline,
    predicted: torch.Tensor,
) -> torch.Tensor:
    value = postprocessor(predicted)
    if isinstance(value, Mapping):
        value = value.get(ACTION_FEATURE_KEY)
    if not isinstance(value, torch.Tensor):
        raise Phase2CAActError("ACT postprocessor did not return an action tensor")
    return value


@torch.no_grad()
def _raw_prediction_error(
    *,
    policy: ACTPolicy,
    postprocessor: PolicyProcessorPipeline,
    processed: Mapping[str, object],
    projected: Mapping[str, torch.Tensor],
) -> tuple[float, np.ndarray]:
    policy.eval()
    observations = {
        key: value
        for key, value in processed.items()
        if key not in {ACTION_FEATURE_KEY, "action_is_pad"}
    }
    predicted = policy.predict_action_chunk(cast(dict[str, torch.Tensor], observations))
    raw = _postprocess_action_chunk(postprocessor, predicted)
    expected = projected[ACTION_FEATURE_KEY]
    padding = projected["action_is_pad"]
    if raw.shape != expected.shape:
        raise Phase2CAActError("predicted and expected raw chunks differ in shape")
    valid = ~padding
    error = torch.abs(raw - expected)[valid].mean()
    return float(error.cpu()), raw.float().cpu().numpy()


def _save_and_reload_probe(
    *,
    run_root: Path,
    run_identity: Phase2CARunIdentity,
    policy: ACTPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    optimizer: torch.optim.Optimizer,
    step: int,
    examples_processed: int,
    metric: Mapping[str, object],
) -> tuple[Mapping[str, object], ACTPolicy, PolicyProcessorPipeline, PolicyProcessorPipeline]:
    record = save_act_checkpoint(
        run_root=run_root,
        identity=run_identity,
        training_state=TrainingState(
            global_step=step,
            examples_processed=examples_processed,
            completed=False,
        ),
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        optimizer=optimizer,
        training_metric=metric,
    )
    loaded = load_act_checkpoint(
        run_root=run_root,
        checkpoint_relative_path=record.relative_path,
        expected_identity=run_identity,
        for_resume=False,
    )
    if not isinstance(loaded.policy, ACTPolicy):
        raise Phase2CAActError("checkpoint smoke did not reload ACTPolicy")
    if not isinstance(loaded.preprocessor, DataProcessorPipeline) or not isinstance(
        loaded.postprocessor, DataProcessorPipeline
    ):
        raise Phase2CAActError("checkpoint smoke did not reload both processors")
    loaded.policy.to(policy.config.device)
    return (
        {
            **record.to_dict(),
            "component_fingerprints": {
                "model": loaded.component_fingerprints.model,
                "preprocessor": loaded.component_fingerprints.preprocessor,
                "postprocessor": loaded.component_fingerprints.postprocessor,
            },
        },
        loaded.policy,
        loaded.preprocessor,
        loaded.postprocessor,
    )


def run_one_batch_gpu_smoke(
    *,
    primary_root: str | Path,
    model_kind: ModelKind | str,
    optimization: Phase2CAOptimizationConfig,
    statistics: ProcessorStatistics,
    run_identity: Phase2CARunIdentity,
    output_root: str | Path,
) -> dict[str, object]:
    """Exercise one real full-architecture batch through save/reload/inference."""

    kind = ModelKind(model_kind)
    root = Path(output_root).resolve()
    if root.exists():
        raise Phase2CAActError(f"smoke output already exists: {root}")
    root.mkdir(parents=True)
    seed_everything(optimization.seed)
    view = load_dataset_view(primary_root, model_kind=kind, split="train")
    loader = _make_dataloader(
        view,
        model_kind=kind,
        optimization=replace(optimization, dataloader_workers=0),
    )
    raw_batch = next(iter(loader))
    projected = prepare_policy_batch(raw_batch, model_kind=kind)
    padding = projected["action_is_pad"]
    if not bool(torch.any(padding)):
        raise Phase2CAActError("real smoke batch unexpectedly contains no padded timestep")
    policy, preprocessor, postprocessor, config = create_policy_and_processors(
        model_kind=kind,
        optimization=optimization,
        statistics=statistics,
        device="cuda",
    )
    optimizer = _optimizer(policy, optimization)
    processed = preprocessor(projected)
    if not isinstance(processed, Mapping):
        raise Phase2CAActError("smoke preprocessor did not return a mapping")
    validate_processed_batch(processed, model_kind=kind)
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(optimization.seed + 17)
    torch.cuda.manual_seed_all(optimization.seed + 17)
    with _autocast(optimization.precision, "cuda"):
        loss, loss_dict = policy.forward(cast(dict[str, torch.Tensor], processed))
    if not bool(torch.isfinite(loss)):
        raise Phase2CAActError("GPU smoke loss is nonfinite")

    perturbed = dict(processed)
    actions = cast(torch.Tensor, processed[ACTION_FEATURE_KEY])
    padding_device = cast(torch.Tensor, processed["action_is_pad"])
    perturbed_actions = actions.clone()
    perturbed_actions[padding_device] = perturbed_actions[padding_device] + 123.0
    perturbed[ACTION_FEATURE_KEY] = perturbed_actions
    torch.manual_seed(optimization.seed + 17)
    torch.cuda.manual_seed_all(optimization.seed + 17)
    with _autocast(optimization.precision, "cuda"):
        perturbed_loss, perturbed_loss_dict = policy.forward(
            cast(dict[str, torch.Tensor], perturbed)
        )
    masking_difference = abs(float(loss.detach().cpu()) - float(perturbed_loss.detach().cpu()))
    if masking_difference > 1e-6:
        raise Phase2CAActError("padded target perturbation changed the installed ACT loss")

    loss.backward()
    gradients = [parameter.grad for parameter in policy.parameters() if parameter.grad is not None]
    if not gradients or not all(bool(torch.isfinite(value).all()) for value in gradients):
        raise Phase2CAActError("GPU smoke backward produced nonfinite gradients")
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(), optimization.gradient_clip_norm
    )
    optimizer.step()
    torch.cuda.synchronize()
    policy.eval()
    reprocessed = preprocessor(projected)
    if not isinstance(reprocessed, Mapping):
        raise Phase2CAActError("smoke reprocessing failed")
    before_error, before_prediction = _raw_prediction_error(
        policy=policy,
        postprocessor=postprocessor,
        processed=reprocessed,
        projected=projected,
    )
    metric = {
        "step": 1,
        "total_loss": float(loss.detach().cpu()),
        "action_loss": float(loss_dict.get("l1_loss", math.nan)),
        "kl_loss": float(loss_dict.get("kld_loss", math.nan)),
        "gradient_norm": float(gradient_norm.detach().cpu()),
    }
    (
        checkpoint,
        reloaded_policy,
        reloaded_preprocessor,
        reloaded_postprocessor,
    ) = _save_and_reload_probe(
        run_root=root,
        run_identity=run_identity,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        optimizer=optimizer,
        step=1,
        examples_processed=int(projected[STATE_FEATURE_KEY].shape[0]),
        metric=metric,
    )
    reloaded_processed = reloaded_preprocessor(projected)
    if not isinstance(reloaded_processed, Mapping):
        raise Phase2CAActError("reloaded preprocessor failed")
    reloaded_error, reloaded_prediction = _raw_prediction_error(
        policy=reloaded_policy,
        postprocessor=reloaded_postprocessor,
        processed=reloaded_processed,
        projected=projected,
    )
    reload_max_error = float(np.max(np.abs(before_prediction - reloaded_prediction)))
    if reload_max_error > 1e-6:
        raise Phase2CAActError("checkpoint reload changed deterministic ACT inference")
    semantic = {
        "schema_version": "langmani-v2-phase2c-a-gpu-smoke-v0",
        "model_kind": kind.value,
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "batch_size": int(projected[STATE_FEATURE_KEY].shape[0]),
        "real_dataset_batch": True,
        "real_preprocessing": True,
        "finite_total_loss": float(loss.detach().cpu()),
        "action_loss": float(loss_dict.get("l1_loss", math.nan)),
        "kl_loss": float(loss_dict.get("kld_loss", math.nan)),
        "padded_timestep_count": int(torch.count_nonzero(padding).cpu()),
        "padded_target_perturbation_loss_difference": masking_difference,
        "padded_target_perturbation_action_loss_difference": abs(
            float(loss_dict.get("l1_loss", math.nan))
            - float(perturbed_loss_dict.get("l1_loss", math.nan))
        ),
        "padding_mask_applied": masking_difference <= 1e-6,
        "backward_completed": True,
        "finite_gradient_norm": float(gradient_norm.detach().cpu()),
        "optimizer_step_completed": True,
        "checkpoint": checkpoint,
        "checkpoint_reload_consistent_max_abs_error": reload_max_error,
        "raw_prediction_error_before_reload": before_error,
        "raw_prediction_error_after_reload": reloaded_error,
        "real_action_inference": True,
        "prediction_finite": bool(np.isfinite(reloaded_prediction).all()),
        "prediction_nonconstant": bool(float(np.var(reloaded_prediction)) > 1e-12),
        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "effective_act_config": act_config_dict(config),
    }
    result = {
        **semantic,
        "fingerprint": f"sha256:{sha256_hex(semantic)}",
        "passed": True,
    }
    _write_new_json(root / "smoke_result.json", result)
    return result


def _micro_batch(
    view: LoadedPhase2CAView,
    *,
    model_kind: ModelKind,
) -> Mapping[str, object]:
    from torch.utils.data._utils.collate import default_collate

    indices: list[int] = []
    offset = 0
    for task_id in model_kind.compatible_task_ids:
        length = view.task_lengths[task_id]
        count = 6 if model_kind is ModelKind.SHARED else 16
        selected = {
            min(length - 1, round(position * (length - 1) / max(count - 1, 1)))
            for position in range(count)
        }
        selected.add(length - 1)
        indices.extend(offset + index for index in sorted(selected))
        offset += length
    rows = [view.dataset[index] for index in indices]
    batch = default_collate(rows)
    if not isinstance(batch, Mapping):
        raise Phase2CAActError("micro-overfit collate did not return a mapping")
    return batch


def run_micro_overfit(
    *,
    primary_root: str | Path,
    model_kind: ModelKind | str,
    optimization: Phase2CAOptimizationConfig,
    statistics: ProcessorStatistics,
    run_identity: Phase2CARunIdentity,
    output_root: str | Path,
    steps: int = 200,
) -> dict[str, object]:
    """Overfit one deterministic tiny real-data batch using the primary model."""

    kind = ModelKind(model_kind)
    if kind not in {ModelKind.PICK, ModelKind.SHARED}:
        raise Phase2CAActError("micro-overfit gate requires Pick ACT or shared Task-ID ACT")
    if steps < 20:
        raise ValueError("micro-overfit requires at least 20 optimization steps")
    root = Path(output_root).resolve()
    if root.exists():
        raise Phase2CAActError(f"micro-overfit output already exists: {root}")
    root.mkdir(parents=True)
    seed_everything(optimization.seed)
    view = load_dataset_view(primary_root, model_kind=kind, split="train")
    raw_batch = _micro_batch(view, model_kind=kind)
    projected = prepare_policy_batch(raw_batch, model_kind=kind)
    if not bool(torch.any(projected["action_is_pad"])):
        raise Phase2CAActError("micro-overfit batch lacks padded action timesteps")
    policy, preprocessor, postprocessor, config = create_policy_and_processors(
        model_kind=kind,
        optimization=optimization,
        statistics=statistics,
        device="cuda",
    )
    optimizer = _optimizer(policy, optimization)
    initial_processed = preprocessor(projected)
    if not isinstance(initial_processed, Mapping):
        raise Phase2CAActError("micro-overfit preprocessing failed")
    initial_error, initial_prediction = _raw_prediction_error(
        policy=policy,
        postprocessor=postprocessor,
        processed=initial_processed,
        projected=projected,
    )
    losses: list[float] = []
    action_losses: list[float] = []
    kl_losses: list[float] = []
    for _ in range(steps):
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        processed = preprocessor(projected)
        if not isinstance(processed, Mapping):
            raise Phase2CAActError("micro-overfit processor returned a non-mapping")
        with _autocast(optimization.precision, "cuda"):
            loss, loss_dict = policy.forward(cast(dict[str, torch.Tensor], processed))
        if not bool(torch.isfinite(loss)):
            raise Phase2CAActError("micro-overfit produced nonfinite loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), optimization.gradient_clip_norm
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise Phase2CAActError("micro-overfit produced nonfinite gradients")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        action_losses.append(float(loss_dict.get("l1_loss", math.nan)))
        kl_losses.append(float(loss_dict.get("kld_loss", math.nan)))
    torch.cuda.synchronize()
    final_processed = preprocessor(projected)
    if not isinstance(final_processed, Mapping):
        raise Phase2CAActError("micro-overfit final preprocessing failed")
    final_error, final_prediction = _raw_prediction_error(
        policy=policy,
        postprocessor=postprocessor,
        processed=final_processed,
        projected=projected,
    )
    window = max(5, steps // 10)
    initial_window_loss = float(np.mean(losses[:window]))
    final_window_loss = float(np.mean(losses[-window:]))
    if final_window_loss >= initial_window_loss or final_error >= initial_error:
        raise Phase2CAActError("micro-overfit did not reduce both loss and unpadded raw error")
    metric = {
        "step": steps,
        "total_loss": losses[-1],
        "action_loss": action_losses[-1],
        "kl_loss": kl_losses[-1],
    }
    checkpoint, loaded_policy, loaded_preprocessor, loaded_postprocessor = _save_and_reload_probe(
        run_root=root,
        run_identity=run_identity,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        optimizer=optimizer,
        step=steps,
        examples_processed=steps * int(projected[STATE_FEATURE_KEY].shape[0]),
        metric=metric,
    )
    loaded_processed = loaded_preprocessor(projected)
    if not isinstance(loaded_processed, Mapping):
        raise Phase2CAActError("micro-overfit reloaded preprocessing failed")
    loaded_error, loaded_prediction = _raw_prediction_error(
        policy=loaded_policy,
        postprocessor=loaded_postprocessor,
        processed=loaded_processed,
        projected=projected,
    )
    reload_error = float(np.max(np.abs(final_prediction - loaded_prediction)))
    if reload_error > 1e-6:
        raise Phase2CAActError("micro-overfit checkpoint reload changed predictions")
    padding = projected["action_is_pad"]
    reference = masked_l1_loss(
        torch.from_numpy(final_prediction).to(projected[ACTION_FEATURE_KEY].device),
        projected[ACTION_FEATURE_KEY],
        padding,
    )
    semantic = {
        "schema_version": "langmani-v2-phase2c-a-micro-overfit-v0",
        "model_kind": kind.value,
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "pipeline_evidence_only": True,
        "policy_quality_result": False,
        "steps": steps,
        "tiny_batch_size": int(projected[STATE_FEATURE_KEY].shape[0]),
        "padded_timestep_count": int(torch.count_nonzero(padding).cpu()),
        "initial_window_total_loss": initial_window_loss,
        "final_window_total_loss": final_window_loss,
        "total_loss_decreased": final_window_loss < initial_window_loss,
        "initial_unpadded_raw_action_error": initial_error,
        "final_unpadded_raw_action_error": final_error,
        "unpadded_raw_action_error_decreased": final_error < initial_error,
        "reference_masked_raw_l1": float(reference.cpu()),
        "output_finite": bool(np.isfinite(final_prediction).all()),
        "output_nonconstant": bool(float(np.var(final_prediction)) > 1e-12),
        "checkpoint": checkpoint,
        "checkpoint_reload_max_abs_error": reload_error,
        "checkpoint_reload_error": loaded_error,
        "effective_act_config": act_config_dict(config),
        "loss_curve": losses,
        "action_loss_curve": action_losses,
        "kl_loss_curve": kl_losses,
    }
    result = {
        **semantic,
        "fingerprint": f"sha256:{sha256_hex(semantic)}",
        "passed": True,
    }
    _write_new_json(root / "micro_overfit_result.json", result)
    return result


def train_primary_act(
    *,
    primary_root: str | Path,
    model_kind: ModelKind | str,
    optimization: Phase2CAOptimizationConfig,
    statistics: ProcessorStatistics,
    run_identity: Phase2CARunIdentity,
    output_root: str | Path,
    device: str = "cuda",
    maximum_steps: int | None = None,
) -> TrainingOutcome:
    """Run one new immutable primary ACT training job."""

    kind = ModelKind(model_kind)
    if run_identity.model_kind is not kind:
        raise Phase2CAActError("run identity and requested model kind differ")
    if device != "cuda":
        raise Phase2CAActError("primary Phase 2C-A training requires the target CUDA runtime")
    target_steps = optimization.training_steps if maximum_steps is None else maximum_steps
    if target_steps < 1 or target_steps > optimization.training_steps:
        raise ValueError("maximum_steps must lie within the frozen primary budget")
    run_root = Path(output_root).resolve() / run_identity.run_fingerprint.removeprefix("sha256:")
    if run_root.exists():
        raise Phase2CAActError(f"run root already exists: {run_root}")
    run_root.mkdir(parents=True)
    view = load_dataset_view(primary_root, model_kind=kind, split="train")
    policy, preprocessor, postprocessor, config = create_policy_and_processors(
        model_kind=kind,
        optimization=optimization,
        statistics=statistics,
        device=device,
    )
    optimizer = _optimizer(policy, optimization)
    run_manifest = {
        "schema_version": "langmani-v2-phase2c-a-act-training-run-v0",
        "identity": run_identity.to_dict(),
        "model_kind": kind.value,
        "model_config": Phase2CAModelConfig.for_model(kind).to_dict(),
        "effective_act_config": act_config_dict(config),
        "optimization_config": optimization.to_dict(),
        "processor_statistics": dict(statistics.manifest),
        "processor_statistics_fingerprint": statistics.fingerprint,
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "dataset_roots": dict(view.task_roots),
        "task_lengths": dict(view.task_lengths),
        "maximum_steps": target_steps,
        "primary_budget_complete": target_steps == optimization.training_steps,
        "smolvla_training_authorized": False,
        "vla_jepa_training_authorized": False,
    }
    _write_new_json(run_root / RUN_MANIFEST_FILE, run_manifest)
    seed_everything(optimization.seed)
    loader = _make_dataloader(view, model_kind=kind, optimization=optimization)
    iterator = iter(loader)
    checkpoint_records: list[Mapping[str, object]] = []
    sample_counts = Counter[str]()
    metrics_path = run_root / TRAINING_METRICS_FILE
    start_time = time.perf_counter()
    last_time = start_time
    peak_allocated = 0
    peak_reserved = 0
    throughput_sum = 0.0
    metrics_count = 0
    policy.train()
    with metrics_path.open("x", encoding="utf-8", newline="\n") as metrics_stream:
        for step in range(1, target_steps + 1):
            batch_start = time.perf_counter()
            raw_batch = next(iterator)
            counts = _task_counts(raw_batch, kind)
            sample_counts.update(counts)
            projected = prepare_policy_batch(raw_batch, model_kind=kind)
            processed = preprocessor(projected)
            if not isinstance(processed, Mapping):
                raise Phase2CAActError("ACT preprocessor did not return a mapping")
            validate_processed_batch(processed, model_kind=kind)
            optimizer.zero_grad(set_to_none=True)
            optimization_start = time.perf_counter()
            with _autocast(optimization.precision, device):
                loss, loss_dict = policy.forward(cast(dict[str, torch.Tensor], processed))
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise Phase2CAActError("ACT produced a nonfinite training loss")
            if not all(math.isfinite(float(value)) for value in loss_dict.values()):
                raise Phase2CAActError("ACT produced a nonfinite loss component")
            loss.backward()
            gradients = [
                parameter.grad for parameter in policy.parameters() if parameter.grad is not None
            ]
            if not gradients or not all(bool(torch.isfinite(value).all()) for value in gradients):
                raise Phase2CAActError("ACT backward produced missing or nonfinite gradients")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), optimization.gradient_clip_norm
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise Phase2CAActError("ACT gradient norm is nonfinite")
            optimizer.step()
            torch.cuda.synchronize()
            step_end = time.perf_counter()
            allocated = int(torch.cuda.memory_allocated())
            reserved = int(torch.cuda.memory_reserved())
            peak_allocated = max(peak_allocated, allocated)
            peak_reserved = max(peak_reserved, reserved)
            batch_size = sum(counts.values())
            step_duration = step_end - optimization_start
            throughput = batch_size / max(step_duration, 1e-12)
            throughput_sum += throughput
            metrics_count += 1
            padding = cast(torch.Tensor, projected["action_is_pad"])
            metric: dict[str, object] = {
                "schema_version": "langmani-v2-phase2c-a-training-metric-v0",
                "step": step,
                "examples_processed": sum(sample_counts.values()),
                "samples_by_task": {task_id: sample_counts[task_id] for task_id in TASK_IDS},
                "batch_samples_by_task": {task_id: counts[task_id] for task_id in TASK_IDS},
                "total_loss": float(loss.detach().cpu()),
                "action_loss": (float(loss_dict["l1_loss"]) if "l1_loss" in loss_dict else None),
                "kl_loss": (float(loss_dict["kld_loss"]) if "kld_loss" in loss_dict else None),
                "padding_timestep_fraction": float(padding.float().mean().cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "dataloader_time_s": optimization_start - batch_start,
                "step_time_s": step_duration,
                "throughput_examples_per_s": throughput,
                "gpu_allocated_bytes": allocated,
                "gpu_reserved_bytes": reserved,
                "wall_elapsed_s": step_end - start_time,
                "checkpoint_path": None,
            }
            if step in optimization.checkpoint_steps or step == target_steps:
                state = TrainingState(
                    global_step=step,
                    examples_processed=sum(sample_counts.values()),
                    completed=step == optimization.training_steps,
                )
                record = save_act_checkpoint(
                    run_root=run_root,
                    identity=run_identity,
                    training_state=state,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    optimizer=optimizer,
                    training_metric=metric,
                )
                checkpoint_records.append(record.to_dict())
                metric["checkpoint_path"] = record.relative_path
                metrics_stream.write(json.dumps(metric, sort_keys=True, allow_nan=False) + "\n")
                metrics_stream.flush()
                os.fsync(metrics_stream.fileno())
            else:
                metrics_stream.write(json.dumps(metric, sort_keys=True, allow_nan=False) + "\n")
                if step % 50 == 0:
                    metrics_stream.flush()
            last_time = step_end
        metrics_stream.flush()
        os.fsync(metrics_stream.fileno())
    duration = last_time - start_time
    effective_samples = validate_consumed_training_samples(
        optimization,
        sample_counts,
    )
    outcome = TrainingOutcome(
        run_root=run_root.as_posix(),
        run_fingerprint=run_identity.run_fingerprint,
        final_step=target_steps,
        examples_processed=sum(sample_counts.values()),
        effective_samples_by_task=effective_samples,
        checkpoint_records=tuple(checkpoint_records),
        training_duration_s=duration,
        peak_gpu_allocated_bytes=peak_allocated,
        peak_gpu_reserved_bytes=peak_reserved,
        mean_examples_per_second=throughput_sum / metrics_count,
    )
    _write_new_json(
        run_root / TRAINING_SUMMARY_FILE,
        {
            "schema_version": "langmani-v2-phase2c-a-training-summary-v0",
            **outcome.to_dict(),
            "complete": target_steps == optimization.training_steps,
            "target_samples_by_task": dict(optimization.target_samples_by_task),
            "effective_passes_by_task": {
                task_id: sample_counts[task_id] / TRAIN_FRAMES[task_id]
                for task_id in kind.compatible_task_ids
            },
        },
    )
    return outcome


@torch.no_grad()
def offline_diagnostics(
    *,
    model_kind: ModelKind | str,
    policy: ACTPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    validation_batches: Iterable[Mapping[str, object]],
    maximum_batches: int | None = None,
) -> dict[str, object]:
    """Compute validation-only ACT diagnostics with explicit padding masks."""

    kind = ModelKind(model_kind)
    policy.eval()
    total_samples = 0
    valid_steps = 0
    weighted_loss = 0.0
    normalized_abs = 0.0
    raw_abs = 0.0
    raw_squared = 0.0
    first_abs = 0.0
    per_horizon_abs = np.zeros(CHUNK_SIZE, dtype=np.float64)
    per_horizon_count = np.zeros(CHUNK_SIZE, dtype=np.int64)
    per_dimension_abs = np.zeros(ACTION_DIMENSION, dtype=np.float64)
    per_dimension_count = np.zeros(ACTION_DIMENSION, dtype=np.int64)
    prediction_values: list[np.ndarray] = []
    saturation_components = 0
    predicted_components = 0
    by_task: dict[str, dict[str, float]] = {
        task_id: {"absolute_sum": 0.0, "component_count": 0.0}
        for task_id in kind.compatible_task_ids
    }
    for batch_index, raw_batch in enumerate(validation_batches):
        if maximum_batches is not None and batch_index >= maximum_batches:
            break
        projected = prepare_policy_batch(raw_batch, model_kind=kind)
        processed = preprocessor(projected)
        if not isinstance(processed, Mapping):
            raise Phase2CAActError("validation processor did not return a mapping")
        validate_processed_batch(processed, model_kind=kind)
        loss, loss_dict = policy.forward(cast(dict[str, torch.Tensor], processed))
        if not bool(torch.isfinite(loss)) or not all(
            math.isfinite(float(value)) for value in loss_dict.values()
        ):
            raise Phase2CAActError("offline validation produced nonfinite loss")
        observations = {
            key: value
            for key, value in processed.items()
            if key not in {ACTION_FEATURE_KEY, "action_is_pad"}
        }
        predicted_normalized = policy.predict_action_chunk(
            cast(dict[str, torch.Tensor], observations)
        )
        predicted_raw_tensor = _postprocess_action_chunk(
            postprocessor,
            predicted_normalized,
        )
        expected_normalized = cast(torch.Tensor, processed[ACTION_FEATURE_KEY])
        expected_raw = cast(torch.Tensor, projected[ACTION_FEATURE_KEY])
        padding = cast(torch.Tensor, projected["action_is_pad"])
        valid = ~padding
        if (
            predicted_normalized.shape != expected_normalized.shape
            or predicted_raw_tensor.shape != expected_raw.shape
        ):
            raise Phase2CAActError("offline prediction shape differs from H=16 targets")
        predicted_np = predicted_raw_tensor.float().cpu().numpy()
        expected_np = expected_raw.float().cpu().numpy()
        predicted_norm_np = predicted_normalized.float().cpu().numpy()
        expected_norm_np = expected_normalized.float().cpu().numpy()
        valid_np = valid.cpu().numpy()
        if not np.isfinite(predicted_np).all() or not np.isfinite(predicted_norm_np).all():
            raise Phase2CAActError("offline ACT predictions are nonfinite")
        raw_error = np.abs(predicted_np - expected_np)
        norm_error = np.abs(predicted_norm_np - expected_norm_np)
        batch_size = predicted_np.shape[0]
        total_samples += batch_size
        valid_rows = int(np.count_nonzero(valid_np))
        valid_steps += valid_rows
        weighted_loss += float(loss.cpu()) * batch_size
        normalized_abs += float(norm_error[valid_np].sum())
        raw_abs += float(raw_error[valid_np].sum())
        raw_squared += float(np.square(predicted_np - expected_np)[valid_np].sum())
        first_abs += float(raw_error[:, 0, :].sum())
        for horizon in range(CHUNK_SIZE):
            position_valid = valid_np[:, horizon]
            per_horizon_abs[horizon] += float(raw_error[:, horizon, :][position_valid].sum())
            per_horizon_count[horizon] += int(np.count_nonzero(position_valid)) * ACTION_DIMENSION
        for dimension in range(ACTION_DIMENSION):
            values = raw_error[:, :, dimension][valid_np]
            per_dimension_abs[dimension] += float(values.sum())
            per_dimension_count[dimension] += values.size
        prediction_values.append(predicted_np[valid_np])
        low = np.array([-3.0] * 7 + [-1.0], dtype=np.float32)
        high = np.array([3.0] * 7 + [1.0], dtype=np.float32)
        margin = np.maximum((high - low) * 0.01, 1e-8)
        valid_predictions = predicted_np[valid_np]
        saturation_components += int(
            np.count_nonzero(
                (valid_predictions <= low + margin) | (valid_predictions >= high - margin)
            )
        )
        predicted_components += int(valid_predictions.size)
        indices = _task_indices(raw_batch).cpu().numpy()
        for row, task_index in enumerate(indices):
            task_id = TASK_IDS[int(task_index)]
            row_valid = valid_np[row]
            by_task[task_id]["absolute_sum"] += float(raw_error[row][row_valid].sum())
            by_task[task_id]["component_count"] += float(
                np.count_nonzero(row_valid) * ACTION_DIMENSION
            )
    if total_samples < 1 or valid_steps < 1:
        raise Phase2CAActError("offline validation consumed no samples")
    predictions = np.concatenate(prediction_values, axis=0)
    component_count = valid_steps * ACTION_DIMENSION
    return {
        "schema_version": "langmani-v2-phase2c-a-offline-diagnostics-v0",
        "model_kind": kind.value,
        "split": "validation",
        "sample_count": total_samples,
        "valid_action_count": valid_steps,
        "total_action_loss": weighted_loss / total_samples,
        "unpadded_normalized_l1": normalized_abs / component_count,
        "unpadded_raw_l1": raw_abs / component_count,
        "unpadded_raw_rmse": math.sqrt(raw_squared / component_count),
        "first_action_raw_l1": first_abs / (total_samples * ACTION_DIMENSION),
        "raw_l1_by_horizon": [
            per_horizon_abs[index] / per_horizon_count[index] for index in range(CHUNK_SIZE)
        ],
        "raw_l1_by_dimension": (per_dimension_abs / per_dimension_count).tolist(),
        "arm_joint_raw_l1": float(per_dimension_abs[:7].sum() / per_dimension_count[:7].sum()),
        "gripper_raw_l1": float(per_dimension_abs[7] / per_dimension_count[7]),
        "predicted_action_variance_by_dimension": np.var(predictions, axis=0).tolist(),
        "constant_action_prediction": bool(np.all(np.var(predictions, axis=0) <= 1e-12)),
        "saturation_rate": (
            saturation_components / predicted_components if predicted_components else 0.0
        ),
        "output_finite": True,
        "by_task_raw_l1": {
            task_id: (
                values["absolute_sum"] / values["component_count"]
                if values["component_count"]
                else None
            )
            for task_id, values in by_task.items()
        },
        "padding_excluded_from_metrics": True,
        "passed": True,
    }


def select_checkpoint_diagnostics(
    reports: Sequence[Mapping[str, object]],
    *,
    maximum: int = 1,
) -> dict[str, object]:
    """Frozen validation-only rule used before any simulator rollout."""

    if maximum not in {1, 2} or not reports:
        raise Phase2CAActError("checkpoint selection requires one or two retained candidates")
    candidates = []
    for report in reports:
        if report.get("split") != "validation" or report.get("passed") is not True:
            raise Phase2CAActError("checkpoint selection received non-validation evidence")
        checkpoint = report.get("checkpoint_fingerprint")
        loss = report.get("total_action_loss")
        first = report.get("first_action_raw_l1")
        if (
            not isinstance(checkpoint, str)
            or not isinstance(loss, int | float)
            or not isinstance(first, int | float)
            or not math.isfinite(float(loss))
            or not math.isfinite(float(first))
        ):
            raise Phase2CAActError("checkpoint diagnostic is malformed")
        candidates.append((float(loss), float(first), checkpoint))
    candidates.sort()
    semantic = {
        "selection_split": "validation",
        "selection_rule": (
            "lowest padding-masked full validation ACT total loss; "
            "tie-break by inverse-normalized first-action L1; retain at most one primary candidate"
        ),
        "candidate_count": len(candidates),
        "maximum_selected": maximum,
        "selected": [
            {
                "rank": index + 1,
                "checkpoint_fingerprint": checkpoint,
                "total_action_loss": loss,
                "first_action_raw_l1": first,
            }
            for index, (loss, first, checkpoint) in enumerate(candidates[:maximum])
        ],
        "test_outcomes_available": False,
    }
    return {
        "schema_version": "langmani-v2-phase2c-a-checkpoint-selection-v0",
        **semantic,
        "fingerprint": f"sha256:{sha256_hex(semantic)}",
        "passed": True,
    }


__all__ = [
    "MODEL_CONFIG_FILE",
    "RUN_MANIFEST_FILE",
    "SOURCE_SPLIT_KEY",
    "TASK_INDEX_KEY",
    "TASK_TOKEN_FEATURE_KEY",
    "LoadedPhase2CAView",
    "Phase2CAActError",
    "ProcessorStatistics",
    "TrainingOutcome",
    "audit_consumer_view",
    "build_phase2c_a_act_config",
    "create_policy_and_processors",
    "load_dataset_view",
    "load_processor_statistics",
    "masked_l1_loss",
    "offline_diagnostics",
    "prepare_policy_batch",
    "run_micro_overfit",
    "run_one_batch_gpu_smoke",
    "select_checkpoint_diagnostics",
    "train_primary_act",
    "validate_processed_batch",
    "validate_shared_act_config",
]
