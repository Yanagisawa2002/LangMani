"""Thin single-process adapter around LeRobot 0.6's official training command.

The only replaced upstream seam is its dataset factory: the selected episodes are
unchanged, but their state/action statistics must not come from whole-dataset metadata.
Optimization, chunk padding, RNG/optimizer resume and checkpoint writing stay upstream.
"""

from __future__ import annotations

import platform
import sys
import time
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

from langmani.policies.m4a_data import (
    ACTION,
    IMAGE,
    STATE,
    Moments,
    file_digest,
    local_dataset_only,
    numpy,
    read_json,
    require_lerobot,
    scalar,
    validate_features,
    write_json,
)


def act_config(device: str) -> Any:
    require_lerobot()
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.act import ACTConfig

    validate_device(device)
    config = ACTConfig(
        input_features={
            IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            STATE: PolicyFeature(type=FeatureType.STATE, shape=(9,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(8,))},
        chunk_size=50,
        n_action_steps=10,
        device=device,
        push_to_hub=False,
    )
    validate_act_config(config)
    return config


def validate_device(device: str) -> None:
    if device not in {"cpu", "cuda", "cuda:0"}:
        raise ValueError("use cpu or cuda; select a physical GPU with CUDA_VISIBLE_DEVICES")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"requested device is unavailable: {device}")


def validate_act_config(config: Any) -> None:
    from lerobot.configs import FeatureType

    expected = {IMAGE: (FeatureType.VISUAL, (3, 256, 256)), STATE: (FeatureType.STATE, (9,))}
    if set(config.input_features) != set(expected) or set(config.output_features) != {ACTION}:
        raise ValueError("ACT observation allowlist forbids privileged or language features")
    for key, (kind, shape) in expected.items():
        if (
            config.input_features[key].type != kind
            or tuple(config.input_features[key].shape) != shape
        ):
            raise ValueError(f"ACT input schema mismatch: {key}")
    action = config.output_features[ACTION]
    if action.type != FeatureType.ACTION or tuple(action.shape) != (8,):
        raise ValueError("ACT action schema must be eight absolute pd_joint_pos components")
    if config.n_obs_steps != 1 or config.chunk_size != 50 or config.n_action_steps != 10:
        raise ValueError("M4A uses one observation, chunk_size=50 and n_action_steps=10")
    if config.temporal_ensemble_coeff is not None:
        raise ValueError("M4A uses the standard ACT action queue without temporal ensembling")


def train_statistics(dataset: Any, episode_ids: list[int]) -> dict[str, dict[str, Any]]:
    """Use only current-frame rows from the exact train view; images use fixed ImageNet stats."""
    from lerobot.utils.constants import IMAGENET_STATS

    validate_features(dataset.features)
    stats = {STATE: Moments(9), ACTION: Moments(8)}
    seen: dict[int, int] = {}
    for row in dataset.hf_dataset:
        ep = scalar(row["episode_index"], integer=True)
        if ep not in episode_ids or scalar(row["frame_index"], integer=True) != seen.get(ep, 0):
            raise ValueError("train statistics contain held-out data or an invalid frame sequence")
        seen[ep] = seen.get(ep, 0) + 1
        for key in stats:
            value = numpy(row[key])
            if (
                value.shape != stats[key].mean.shape
                or not torch.isfinite(torch.as_tensor(value)).all()
            ):
                raise ValueError("invalid train statistics vector")
            stats[key].update(value)
    if set(seen) != set(episode_ids) or sum(seen.values()) != len(dataset):
        raise ValueError("train statistics do not cover the exact episode view")
    for ep, count in seen.items():
        if count != dataset.meta.episodes[ep]["length"]:
            raise ValueError("partial episode in train statistics")
    result = {key: value.report() for key, value in stats.items()}
    result[IMAGE] = {key: numpy(value).tolist() for key, value in IMAGENET_STATS.items()}
    return result


def checkpoint_directory(path: Path) -> Path:
    """Accept an upstream step directory, pretrained_model directory, or train_config.json."""
    path = path.resolve(strict=True)
    if path.is_file():
        if path.name != "train_config.json":
            raise ValueError("checkpoint file must be train_config.json")
        path = path.parent
    if path.name != "pretrained_model":
        path = path / "pretrained_model"
    for name in (
        "config.json",
        "model.safetensors",
        "train_config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        if not (path / name).is_file():
            raise ValueError(f"checkpoint is incomplete: missing {name}")
    return path


def load_checkpoint(path: Path, device: str) -> tuple[Any, Any, Any]:
    require_lerobot()
    validate_device(device)
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act import ACTPolicy

    path = checkpoint_directory(path)
    config = PreTrainedConfig.from_pretrained(path, local_files_only=True)
    if config.type != "act":
        raise ValueError("checkpoint is not upstream ACT")
    config.device = device
    validate_act_config(config)
    policy = ACTPolicy.from_pretrained(path, config=config, local_files_only=True).to(device).eval()
    pre, post = make_pre_post_processors(
        config,
        pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


def policy_observation(image: Any, state: Any) -> dict[str, torch.Tensor]:
    """Construct an allowlisted policy input; task text and simulator extras cannot pass through."""
    image = torch.as_tensor(image)
    state = torch.as_tensor(state)
    if image.dtype != torch.uint8 or tuple(image.shape) != (3, 256, 256):
        raise ValueError("policy camera must be uint8[3,256,256]")
    if (
        state.dtype != torch.float32
        or tuple(state.shape) != (9,)
        or not torch.isfinite(state).all()
    ):
        raise ValueError("policy proprioception must be finite float32[9]")
    return {IMAGE: image.float() / 255, STATE: state}


@local_dataset_only()
def train_official(
    *,
    root: Path,
    output: Path,
    split: Mapping[str, Any],
    mode: str,
    device: str,
    batch_size: int,
    steps: int,
    seed: int,
    wandb: bool = False,
    resume: Path | None = None,
) -> dict[str, Any]:
    """Run upstream training and then reload its actual model and processor checkpoint."""
    require_lerobot()
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets import LeRobotDataset
    from lerobot.scripts import lerobot_train as upstream

    if mode not in {"smoke", "full"} or min(batch_size, steps) < 1 or seed < 0:
        raise ValueError("invalid training mode, batch size, steps or seed")
    if mode == "smoke" and steps > 20:
        raise ValueError("smoke is capped at 20 optimizer steps; use full for baseline training")
    validate_device(device)
    train_ids = list(split["train_episode_ids"])
    if not train_ids or set(train_ids) & set(split["held_out_episode_ids"]):
        raise ValueError("empty or overlapping train view")
    training_dir = output / "training"
    # This reader has no temporal expansion. It never selects held-out/test frames.
    current = LeRobotDataset(
        split["repo_id"], root=root, episodes=train_ids, video_backend="pyav", return_uint8=True
    )
    stats = train_statistics(current, train_ids)
    if len(current) != split["train_frames"]:
        raise ValueError("train frame count differs from split manifest")
    stats_payload = {
        "episode_ids": train_ids,
        "visual_stats": "fixed ImageNet",
        "state_action_stats": "train only",
        "statistics": stats,
    }
    if resume is not None:
        checkpoint = checkpoint_directory(resume)
        if not checkpoint.is_relative_to(training_dir.resolve()):
            raise ValueError("resume checkpoint must belong to this run's training directory")
        if read_json(output / "normalization.json") != stats_payload:
            raise ValueError("resume normalization differs from original training")
        cfg = TrainPipelineConfig.from_pretrained(
            checkpoint / "train_config.json", local_files_only=True
        )
        if cfg.seed != seed or cfg.batch_size != batch_size or cfg.dataset.episodes != train_ids:
            raise ValueError("resume seed, batch size or episode view changed")
        if cfg.dataset.repo_id != split["repo_id"] or cfg.policy.device != device:
            raise ValueError("resume dataset identity or device changed")
        if steps <= read_json(checkpoint.parent / "training_state/training_step.json")["step"]:
            raise ValueError("resume --steps must exceed the saved global step")
        cfg.dataset.root = str(root.resolve())
        cfg.output_dir = training_dir.resolve()
        cfg.resume, cfg.steps = True, steps
        argv = ["m4a", f"--config_path={checkpoint / 'train_config.json'}"]
    else:
        if training_dir.exists():
            raise FileExistsError("training directory already exists; use --resume")
        cfg = TrainPipelineConfig(
            dataset=DatasetConfig(
                repo_id=split["repo_id"],
                root=str(root.resolve()),
                episodes=train_ids,
                video_backend="pyav",
                use_imagenet_stats=False,
            ),
            policy=act_config(device),
            output_dir=training_dir.resolve(),
            batch_size=batch_size,
            steps=steps,
            seed=seed,
            num_workers=0,
            env_eval_freq=0,
            eval_steps=0,
            save_freq=min(steps, 10_000),
            log_freq=min(steps, 100),
            cudnn_deterministic=True,
            wandb=WandBConfig(enable=wandb, project="langmani-m4a", disable_artifact=True),
        )
        argv = ["m4a"]
        write_json(output / "normalization.json", stats_payload)
    validate_act_config(cfg.policy)
    factory = upstream.make_train_eval_datasets

    def dataset_factory(config: Any) -> tuple[Any, None]:
        dataset, evaluation = factory(config)
        if evaluation is not None or dataset.episodes != train_ids:
            raise ValueError("upstream changed the explicit train episode view")
        dataset.meta.stats = {
            key: {
                name: torch.as_tensor(value, dtype=torch.float32)
                for name, value in values.items()
                if name != "count"
            }
            for key, values in stats.items()
        }
        return dataset, None

    # Scoped in-process adapter, restored even on failure; no installed source is edited.
    started = time.perf_counter()

    def windows_checkpoint_pointer(path: Path) -> Path:
        # Windows review hosts may not grant symlink privileges. The checkpoint itself is still
        # saved by upstream; only its optional convenience link becomes an explicit JSON pointer.
        pointer = path.parent / "last_checkpoint.json"
        write_json(pointer, {"checkpoint": path.name, "reason": "Windows symlink-free pointer"})
        return pointer

    with ExitStack() as stack:
        stack.enter_context(patch.object(upstream, "make_train_eval_datasets", dataset_factory))
        stack.enter_context(patch.object(sys, "argv", argv))
        if platform.system() == "Windows":
            print(
                "Windows fixture runtime: last_checkpoint.json replaces the optional last symlink"
            )
            stack.enter_context(
                patch.object(upstream, "update_last_checkpoint", windows_checkpoint_pointer)
            )
        upstream.train(cfg)
    elapsed = time.perf_counter() - started
    from lerobot.common.train_utils import get_step_checkpoint_dir

    checkpoint = checkpoint_directory(get_step_checkpoint_dir(training_dir, steps, steps))
    saved_step = read_json(checkpoint.parent / "training_state/training_step.json")["step"]
    if saved_step != steps:
        raise RuntimeError("upstream did not save the requested final optimization step")
    policy, pre, post = load_checkpoint(checkpoint, device)
    row = current[0]
    with torch.inference_mode():
        observation = policy_observation(row[IMAGE], row[STATE])
        policy.reset()
        action = post(policy.select_action(pre(observation)))
        if tuple(action.shape) != (1, 8) or not torch.isfinite(action).all():
            raise RuntimeError("reloaded ACT/processor pair produced invalid actions")
    result = {
        "mode": mode,
        "steps": steps,
        "checkpoint": str(checkpoint.relative_to(output)),
        "checkpoint_sha256": file_digest(checkpoint / "model.safetensors"),
        "checkpoint_files": {p.name: file_digest(p) for p in checkpoint.iterdir() if p.is_file()},
        "checkpoint_reloaded": True,
        "train_frames": len(current),
        "sample_presentations": steps * batch_size,
        "equivalent_train_passes": steps * batch_size / len(current),
        "training_invocation_seconds": elapsed,
        "lerobot_version": "0.6.0",
        "torch_version": torch.__version__,
        "last_step_and_reload_peak_allocated_vram_bytes": torch.cuda.max_memory_allocated()
        if device.startswith("cuda")
        else None,
        "full_act_trained": mode == "full",
        "closed_loop_evaluated": False,
    }
    write_json(output / "checkpoint_metadata.json", result)
    return result
