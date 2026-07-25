"""Official LeRobot 0.6 SmolVLA integration for Phase 2C-B.

The only model-path change is :class:`BoundedSmolVLAPolicyV1`, a parameter-free
subclass that converts physical training actions to the frozen latent
representation and converts generated latent chunks back to physical actions.
All visual-language and flow-matching modules are the official implementation
and retain their published parameter names and checkpoint bytes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import torch
from lerobot.configs import (  # type: ignore[import-untyped]
    FeatureType,
    NormalizationMode,
    PolicyFeature,
)
from lerobot.configs.policies import PreTrainedConfig  # type: ignore[import-untyped]
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore[import-untyped]
from lerobot.policies.factory import (  # type: ignore[import-untyped]
    make_pre_post_processors,
)
from lerobot.policies.smolvla.configuration_smolvla import (  # type: ignore[import-untyped]
    SmolVLAConfig,
)
from lerobot.policies.smolvla.modeling_smolvla import (  # type: ignore[import-untyped]
    SmolVLAPolicy,
)
from lerobot.processor import PolicyProcessorPipeline  # type: ignore[import-untyped]
from torch.utils.data import ConcatDataset, Dataset

from langmani.v2.phase2c_a import (
    ACTION_DIMENSION,
    CONTROL_FREQUENCY_HZ,
    IMAGE_SHAPE_CHW,
    PACKAGE_FINGERPRINT,
    STATE_DIMENSION,
    TASK_IDS,
    TASK_SLUGS,
    TRAIN_FRAMES,
)
from langmani.v2.phase2c_a import ModelKind as ActModelKind
from langmani.v2.phase2c_a_act import (
    ProcessorStatistics,
    load_processor_statistics,
)
from langmani.v2.phase2c_b import (
    BOUNDED_ACTION_TRANSFORM_FILE,
    CHUNK_SIZE,
    OFFICIAL_BASE_MODEL,
    OFFICIAL_BASE_REVISION,
    BoundedActionLatentV1,
    ModelKind,
    Phase2CBContractError,
    SmolVLATrainingConfig,
    canonical_fingerprint,
    validate_model_batch,
)

IMAGE_FEATURE_KEY: Final = "observation.images.base_camera"
STATE_FEATURE_KEY: Final = "observation.state"
ACTION_FEATURE_KEY: Final = "action"
ACTION_PAD_KEY: Final = "action_is_pad"
LANGUAGE_FEATURE_KEY: Final = "task"
POLICY_BATCH_KEYS: Final = (
    IMAGE_FEATURE_KEY,
    STATE_FEATURE_KEY,
    ACTION_FEATURE_KEY,
    ACTION_PAD_KEY,
    LANGUAGE_FEATURE_KEY,
)


class Phase2CBSmolVLAError(Phase2CBContractError):
    """Raised when the official SmolVLA boundary is violated."""


class BoundedSmolVLAPolicyV1(SmolVLAPolicy):
    """Official SmolVLA with a parameter-free bounded embodiment action path."""

    action_transform = BoundedActionLatentV1()

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> Any:
        physical = batch.get(ACTION_FEATURE_KEY)
        if not isinstance(physical, torch.Tensor):
            raise Phase2CBSmolVLAError("training batch lacks physical actions")
        transformed = dict(batch)
        latent = self.action_transform.encode(physical)
        padding = batch.get(ACTION_PAD_KEY)
        if padding is not None:
            if (
                not isinstance(padding, torch.Tensor)
                or padding.dtype is not torch.bool
                or padding.shape != physical.shape[:2]
            ):
                raise Phase2CBSmolVLAError("training action_is_pad is malformed")
            latent = latent.masked_fill(padding.unsqueeze(-1), 0.0)
        transformed[ACTION_FEATURE_KEY] = latent
        return super().forward(transformed, noise=noise, time=time, reduction=reduction)

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        latent = super().predict_action_chunk(batch, noise=noise, **kwargs)
        return self.action_transform.decode(latent)


@dataclass(frozen=True, slots=True)
class LoadedSmolVLAView:
    dataset: Dataset[Mapping[str, object]]
    task_lengths: Mapping[str, int]
    task_roots: Mapping[str, str]
    split: str


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2CBSmolVLAError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise Phase2CBSmolVLAError(f"{path} must contain one JSON object")
    return value


def task_root(primary_root: str | Path, task_id: str, split: str) -> Path:
    if task_id not in TASK_IDS or split not in {
        "train",
        "validation",
        "test_unseen_reset",
        "test_unseen_task_language",
        "test_visual_shift",
    }:
        raise Phase2CBSmolVLAError("unknown accepted task/split root")
    return Path(primary_root).resolve() / "task_roots" / TASK_SLUGS[task_id] / "splits" / split


def _repo_id(root: Path, *, task_id: str, split: str) -> str:
    manifest = _read_object(root / "langmani_phase2b6_split_manifest.json")
    repo_id = manifest.get("repo_id")
    if (
        not isinstance(repo_id, str)
        or not repo_id
        or manifest.get("task_id") != task_id
        or manifest.get("primary_split") != split
        or manifest.get("package_fingerprint") not in {None, PACKAGE_FINGERPRINT}
    ):
        raise Phase2CBSmolVLAError("accepted LeRobot split identity changed")
    return repo_id


def load_dataset_view(
    primary_root: str | Path,
    *,
    model_kind: ModelKind | str,
    split: str,
    chunk_size: int = CHUNK_SIZE,
) -> LoadedSmolVLAView:
    """Load immutable LeRobot split roots with the official 50-action target."""

    if chunk_size != CHUNK_SIZE:
        raise Phase2CBSmolVLAError("Phase 2C-B does not permit a chunk-size sweep")
    kind = ModelKind(model_kind)
    datasets: list[Dataset[Mapping[str, object]]] = []
    lengths: dict[str, int] = {}
    roots: dict[str, str] = {}
    delta_timestamps = {
        ACTION_FEATURE_KEY: [index / CONTROL_FREQUENCY_HZ for index in range(chunk_size)]
    }
    for task_id in kind.compatible_task_ids:
        root = task_root(primary_root, task_id, split)
        if not root.is_dir():
            raise Phase2CBSmolVLAError(f"accepted LeRobot root is missing: {root}")
        dataset = LeRobotDataset(
            repo_id=_repo_id(root, task_id=task_id, split=split),
            root=root,
            delta_timestamps=delta_timestamps,
            video_backend="pyav",
            return_uint8=True,
            download_videos=False,
        )
        datasets.append(cast(Dataset[Mapping[str, object]], dataset))
        lengths[task_id] = len(dataset)
        roots[task_id] = root.as_posix()
    combined: Dataset[Mapping[str, object]]
    if len(datasets) == 1:
        combined = datasets[0]
    else:
        combined = cast(Dataset[Mapping[str, object]], ConcatDataset(datasets))
    return LoadedSmolVLAView(
        dataset=combined,
        task_lengths=lengths,
        task_roots=roots,
        split=split,
    )


def project_policy_batch(raw_batch: Mapping[str, object], *, shared: bool) -> dict[str, object]:
    """Remove metadata and convert immutable uint8 camera bytes to model float input."""

    missing = [key for key in POLICY_BATCH_KEYS if key not in raw_batch]
    if missing:
        raise Phase2CBSmolVLAError("dataset batch lacks: " + ", ".join(missing))
    batch = {key: raw_batch[key] for key in POLICY_BATCH_KEYS}
    validate_model_batch(batch, shared=shared)
    image = batch[IMAGE_FEATURE_KEY]
    state = batch[STATE_FEATURE_KEY]
    action = batch[ACTION_FEATURE_KEY]
    padding = batch[ACTION_PAD_KEY]
    if (
        not isinstance(image, torch.Tensor)
        or image.dtype not in {torch.uint8, torch.float32}
        or image.ndim != 4
        or tuple(image.shape[1:]) != IMAGE_SHAPE_CHW
        or not isinstance(state, torch.Tensor)
        or state.dtype is not torch.float32
        or state.ndim != 2
        or state.shape[1] != STATE_DIMENSION
        or not isinstance(action, torch.Tensor)
        or action.dtype is not torch.float32
        or action.ndim != 3
        or tuple(action.shape[1:]) != (CHUNK_SIZE, ACTION_DIMENSION)
        or not isinstance(padding, torch.Tensor)
        or padding.dtype is not torch.bool
        or padding.shape != action.shape[:2]
        or not bool(torch.isfinite(state).all())
        or not bool(torch.isfinite(action).all())
    ):
        raise Phase2CBSmolVLAError("real SmolVLA batch violates the observation/action contract")
    model_image = image.to(dtype=torch.float32).div_(255.0) if image.dtype is torch.uint8 else image
    if (
        not bool(torch.isfinite(model_image).all())
        or float(model_image.amin()) < 0.0
        or float(model_image.amax()) > 1.0
    ):
        raise Phase2CBSmolVLAError("SmolVLA camera input must lie in [0,1]")
    batch[IMAGE_FEATURE_KEY] = model_image
    return batch


def _act_kind(kind: ModelKind) -> ActModelKind:
    return {
        ModelKind.PICK: ActModelKind.PICK,
        ModelKind.STACK: ActModelKind.STACK,
        ModelKind.PUSH: ActModelKind.PUSH,
        ModelKind.SHARED: ActModelKind.SHARED,
    }[kind]


def load_smolvla_statistics(
    primary_root: str | Path,
    *,
    model_kind: ModelKind | str,
) -> ProcessorStatistics:
    root = Path(primary_root).resolve()
    return load_processor_statistics(
        root / "metadata" / "dataset_statistics.json",
        root / "metadata" / "normalization.json",
        model_kind=_act_kind(ModelKind(model_kind)),
    )


def build_official_config(
    *,
    base_snapshot: str | Path,
    vlm_snapshot: str | Path,
    device: str,
    training: SmolVLATrainingConfig,
) -> SmolVLAConfig:
    """Bind the published config to the exact LangMani public feature contract."""

    raw = PreTrainedConfig.from_pretrained(
        Path(base_snapshot),
        cli_overrides=["--input_features=null"],
    )
    if not isinstance(raw, SmolVLAConfig) or raw.input_features is not None:
        raise Phase2CBSmolVLAError("official SmolVLA config replacement semantic changed")
    raw.input_features = {
        IMAGE_FEATURE_KEY: PolicyFeature(FeatureType.VISUAL, IMAGE_SHAPE_CHW),
        STATE_FEATURE_KEY: PolicyFeature(FeatureType.STATE, (STATE_DIMENSION,)),
    }
    raw.output_features = {
        ACTION_FEATURE_KEY: PolicyFeature(FeatureType.ACTION, (ACTION_DIMENSION,))
    }
    raw.normalization_mapping = {
        FeatureType.VISUAL: NormalizationMode.IDENTITY,
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.IDENTITY,
    }
    raw.vlm_model_name = Path(vlm_snapshot).resolve().as_posix()
    raw.chunk_size = training.chunk_size
    raw.n_action_steps = training.n_action_steps
    raw.num_steps = training.num_inference_steps
    raw.freeze_vision_encoder = training.freeze_vision_encoder
    raw.train_expert_only = training.train_expert_only
    raw.train_state_proj = training.train_state_projection
    raw.resize_imgs_with_padding = training.resize_images_with_padding
    raw.tokenizer_max_length = training.tokenizer_max_length
    raw.optimizer_lr = training.learning_rate
    raw.optimizer_betas = training.betas
    raw.optimizer_eps = training.optimizer_epsilon
    raw.optimizer_weight_decay = training.weight_decay
    raw.optimizer_grad_clip_norm = training.gradient_clip_norm
    raw.scheduler_warmup_steps = training.warmup_steps
    raw.scheduler_decay_steps = training.decay_steps
    raw.scheduler_decay_lr = training.decay_learning_rate
    raw.device = device
    raw.use_amp = False
    raw.push_to_hub = False
    raw.validate_features()
    if set(raw.input_features) != {IMAGE_FEATURE_KEY, STATE_FEATURE_KEY}:
        raise Phase2CBSmolVLAError("official SmolVLA config added an unexpected input")
    return raw


def load_pretrained_policy(
    *,
    base_snapshot: str | Path,
    config: SmolVLAConfig,
) -> tuple[BoundedSmolVLAPolicyV1, dict[str, object]]:
    """Strictly load every official tensor without reinitializing a parameter."""

    policy = BoundedSmolVLAPolicyV1.from_pretrained(
        Path(base_snapshot),
        config=config,
        strict=True,
    )
    parameters = list(policy.parameters())
    parameter_count = sum(int(parameter.numel()) for parameter in parameters)
    trainable_count = sum(
        int(parameter.numel()) for parameter in parameters if parameter.requires_grad
    )
    semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-b-pretrained-loading-audit-v0",
        "base_model": OFFICIAL_BASE_MODEL,
        "base_revision": OFFICIAL_BASE_REVISION,
        "strict_weight_load": True,
        "published_parameter_count": parameter_count,
        "loaded_parameter_count": parameter_count,
        "loaded_parameter_percentage": 100.0,
        "reinitialized_parameters": [],
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "frozen_parameter_count": parameter_count - trainable_count,
        "bounded_transform_trainable_parameters": 0,
        "state_feature_dimension": STATE_DIMENSION,
        "action_feature_dimension": ACTION_DIMENSION,
        "internal_max_state_dimension": int(config.max_state_dim),
        "internal_max_action_dimension": int(config.max_action_dim),
        "camera_features": [IMAGE_FEATURE_KEY],
        "action_transform_fingerprint": policy.action_transform.fingerprint,
        "passed": parameter_count > 0 and 0 < trainable_count < parameter_count,
    }
    if semantic["passed"] is not True:
        raise Phase2CBSmolVLAError("official pretrained loading/trainability audit failed")
    return policy, {**semantic, "fingerprint": canonical_fingerprint(semantic)}


def make_processors(
    config: SmolVLAConfig,
    statistics: ProcessorStatistics,
) -> tuple[PolicyProcessorPipeline[Any, Any], PolicyProcessorPipeline[Any, Any]]:
    """Construct official processors with state-only numeric normalization."""

    processors = make_pre_post_processors(config, dataset_stats=dict(statistics.tensors))
    preprocessor = cast(PolicyProcessorPipeline[Any, Any], processors[0])
    postprocessor = cast(PolicyProcessorPipeline[Any, Any], processors[1])
    return preprocessor, postprocessor


def processor_manifest(
    *,
    config: SmolVLAConfig,
    statistics: ProcessorStatistics,
) -> dict[str, object]:
    normalization = {
        (key.value if isinstance(key, FeatureType) else str(key)): (
            value.value if isinstance(value, NormalizationMode) else str(value)
        )
        for key, value in config.normalization_mapping.items()
    }
    semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-b-processor-contract-v0",
        "input_features": {
            key: {"type": value.type.value, "shape": list(value.shape)}
            for key, value in config.input_features.items()
        },
        "output_features": {
            key: {"type": value.type.value, "shape": list(value.shape)}
            for key, value in config.output_features.items()
        },
        "normalization_mapping": normalization,
        "statistics_fingerprint": statistics.fingerprint,
        "state_statistics_source": statistics.manifest["selection"],
        "action_statistics_consumed": False,
        "action_normalization": "IDENTITY_before_bounded_latent_transform",
        "dataset_image_dtype": "uint8",
        "model_image_dtype": "float32",
        "image_conversion": "exact_uint8_divide_255",
        "task_input": LANGUAGE_FEATURE_KEY,
        "task_id_input": False,
        "action_is_pad": True,
        "bounded_transform_file": BOUNDED_ACTION_TRANSFORM_FILE,
    }
    if normalization != {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "IDENTITY"}:
        raise Phase2CBSmolVLAError("SmolVLA processor normalization contract changed")
    return {**semantic, "fingerprint": canonical_fingerprint(semantic)}


def audit_real_view(
    primary_root: str | Path,
    *,
    model_kind: ModelKind | str,
    split: str = "train",
) -> dict[str, object]:
    """Decode real boundary rows through LeRobot without scanning the full media tree."""

    kind = ModelKind(model_kind)
    view = load_dataset_view(primary_root, model_kind=kind, split=split)
    reports: list[dict[str, object]] = []
    offset = 0
    for task_id in kind.compatible_task_ids:
        length = view.task_lengths[task_id]
        for local_index in (0, length - 1):
            row = view.dataset[offset + local_index]
            batch: dict[str, object] = {}
            for key in POLICY_BATCH_KEYS:
                value = row[key]
                batch[key] = value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value]
            projected = project_policy_batch(batch, shared=kind is ModelKind.SHARED)
            padding = cast(torch.Tensor, projected[ACTION_PAD_KEY])
            reports.append(
                {
                    "task_id": task_id,
                    "dataset_index": offset + local_index,
                    "padded_timestep_count": int(torch.count_nonzero(padding)),
                    "instruction": cast(Sequence[str], projected[LANGUAGE_FEATURE_KEY])[0],
                }
            )
        offset += length
    expected = {task_id: TRAIN_FRAMES[task_id] for task_id in kind.compatible_task_ids}
    if split == "train" and dict(view.task_lengths) != expected:
        raise Phase2CBSmolVLAError("SmolVLA train view frame counts changed")
    semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-b-real-view-audit-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "model_kind": kind.value,
        "split": split,
        "chunk_size": CHUNK_SIZE,
        "task_lengths": dict(view.task_lengths),
        "task_roots": dict(view.task_roots),
        "boundary_rows": reports,
        "policy_batch_keys": list(POLICY_BATCH_KEYS),
        "task_id_model_input": False,
        "privileged_policy_fields": [],
        "passed": True,
    }
    return {**semantic, "fingerprint": canonical_fingerprint(semantic)}


__all__ = [
    "ACTION_FEATURE_KEY",
    "ACTION_PAD_KEY",
    "BoundedSmolVLAPolicyV1",
    "IMAGE_FEATURE_KEY",
    "LANGUAGE_FEATURE_KEY",
    "LoadedSmolVLAView",
    "POLICY_BATCH_KEYS",
    "Phase2CBSmolVLAError",
    "STATE_FEATURE_KEY",
    "audit_real_view",
    "build_official_config",
    "load_dataset_view",
    "load_pretrained_policy",
    "load_smolvla_statistics",
    "make_processors",
    "processor_manifest",
    "project_policy_batch",
    "task_root",
]
