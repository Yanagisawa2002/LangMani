"""Thin SmolVLA adapter: upstream model, tokenizer, processors, optimizer and trainer."""

from __future__ import annotations

import platform
import sys
import time
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

from langmani.policies.m4a_data import (
    ACTION,
    IMAGE,
    STATE,
    digest,
    file_digest,
    local_dataset_only,
    read_json,
    require_lerobot,
    write_json,
)
from langmani.policies.m4a_training import checkpoint_directory, train_statistics, validate_device
from langmani.policies.m4a_training import policy_observation as visual_observation
from langmani.policies.m4b_protocol import (
    BACKBONE_MODEL,
    BACKBONE_REVISION,
    BASE_MODEL,
    BASE_REVISION,
)


def verify_assets(path: Path) -> dict[str, Any]:
    receipt = read_json(path)
    for label, repo, revision in (
        ("base", BASE_MODEL, BASE_REVISION),
        ("backbone", BACKBONE_MODEL, BACKBONE_REVISION),
    ):
        item = receipt[label]
        if item["repo_id"] != repo or item["revision"] != revision or not item["files"]:
            raise ValueError("pretrained asset identity changed")
        root = Path(item["path"]).resolve(strict=True)
        actual = {
            p.relative_to(root).as_posix(): file_digest(p)
            for p in sorted(root.rglob("*"))
            if p.is_file() and ".cache" not in p.relative_to(root).parts
        }
        if actual != item["files"]:
            raise ValueError(f"pinned {label} snapshot bytes changed")
    return receipt


def validate_config(config: Any) -> None:
    from lerobot.configs import FeatureType, NormalizationMode

    if (
        config.type != "smolvla"
        or set(config.input_features) != {IMAGE, STATE}
        or set(config.output_features) != {ACTION}
    ):
        raise ValueError("only upstream SmolVLA RGB/qpos inputs and joint actions are permitted")
    for key, kind, shape in (
        (IMAGE, FeatureType.VISUAL, (3, 256, 256)),
        (STATE, FeatureType.STATE, (9,)),
        (ACTION, FeatureType.ACTION, (8,)),
    ):
        item = (config.input_features | config.output_features)[key]
        if item.type != kind or tuple(item.shape) != shape:
            raise ValueError(f"invalid policy feature: {key}")
    if (config.n_obs_steps, config.chunk_size, config.n_action_steps, config.num_steps) != (
        1,
        50,
        10,
        10,
    ):
        raise ValueError("M4B observation/chunk/execution/flow protocol changed")
    if (
        config.use_delta_joint_actions_aloha
        or config.adapt_to_pi_aloha
        or config.load_vlm_weights
        or config.empty_cameras
    ):
        raise ValueError(
            "M4B forbids action adaptation, extra cameras or separately loaded VLM weights"
        )
    if config.normalization_mapping["VISUAL"] != NormalizationMode.IDENTITY:
        raise ValueError("SmolVLA visual preprocessing must remain upstream identity")
    if not Path(config.vlm_model_name).is_dir():
        raise ValueError("backbone/tokenizer must be an immutable verified local snapshot")


def smolvla_config(device: str, assets: dict[str, Any], learning_rate: float) -> Any:
    require_lerobot()
    validate_device(device)
    from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
    from lerobot.policies.smolvla import SmolVLAConfig  # noqa: F401

    config = PreTrainedConfig.from_pretrained(assets["base"]["path"], local_files_only=True)
    config.input_features = {
        IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        STATE: PolicyFeature(type=FeatureType.STATE, shape=(9,)),
    }
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(8,))}
    config.device, config.push_to_hub = device, False
    config.pretrained_path = Path(assets["base"]["path"])
    config.pretrained_revision = assets["base"]["revision"]
    config.vlm_model_name = assets["backbone"]["path"]
    config.n_action_steps, config.num_steps = 10, 10
    config.optimizer_lr = learning_rate
    config.load_vlm_weights = False
    validate_config(config)
    return config


def policy_observation(image: Any, state: Any, instruction: str) -> dict[str, Any]:
    if not isinstance(instruction, str):
        raise ValueError("instruction must be explicit text, including an explicit blank string")
    return {**visual_observation(image, state), "task": instruction}


def load_checkpoint(path: Path, device: str) -> tuple[Any, Any, Any]:
    require_lerobot()
    validate_device(device)
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla import SmolVLAPolicy

    path = checkpoint_directory(path)
    config = PreTrainedConfig.from_pretrained(path, local_files_only=True)
    config.device = device
    validate_config(config)
    policy = (
        SmolVLAPolicy.from_pretrained(path, config=config, local_files_only=True).to(device).eval()
    )
    pre, post = make_pre_post_processors(
        config,
        pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


@local_dataset_only()
def train_official(
    *,
    root: Path,
    output: Path,
    split: dict[str, Any],
    assets: dict[str, Any],
    mode: str,
    device: str,
    batch_size: int,
    steps: int,
    seed: int,
    learning_rate: float = 1e-4,
    wandb: bool = False,
    resume: Path | None = None,
) -> dict[str, Any]:
    require_lerobot()
    validate_device(device)
    from lerobot.common.train_utils import get_step_checkpoint_dir
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets import LeRobotDataset
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from lerobot.scripts import lerobot_train as upstream

    if (
        mode not in {"smoke", "full"}
        or min(batch_size, steps) < 1
        or seed < 0
        or not 0 < learning_rate < 1
        or (mode == "smoke" and steps > 20)
    ):
        raise ValueError("invalid training budget/configuration")
    train_ids = split["train_episode_ids"]
    if not train_ids or set(train_ids) & set(
        split["held_out_episode_ids"] + split["excluded_test_episode_ids"]
    ):
        raise ValueError("train view overlaps held-out data")
    current = LeRobotDataset(
        split["repo_id"], root=root, episodes=train_ids, video_backend="pyav", return_uint8=True
    )
    stats = train_statistics(current, train_ids)
    if len(current) != split["train_frames"]:
        raise ValueError("train frames differ from split")
    stats_payload = {
        "episode_ids": train_ids,
        "visual_normalization": "upstream IDENTITY; unused fixed ImageNet metadata",
        "state_action_stats": "train only",
        "statistics": stats,
    }
    training_dir = output / "training"
    if resume is not None:
        checkpoint = checkpoint_directory(resume)
        if (
            not checkpoint.is_relative_to(training_dir.resolve())
            or read_json(output / "normalization.json") != stats_payload
        ):
            raise ValueError("resume checkpoint or train-only normalization differs")
        cfg = TrainPipelineConfig.from_pretrained(
            checkpoint / "train_config.json", local_files_only=True
        )
        if (
            cfg.seed != seed
            or cfg.batch_size != batch_size
            or cfg.dataset.episodes != train_ids
            or cfg.dataset.repo_id != split["repo_id"]
            or cfg.policy.device != device
            or cfg.policy.optimizer_lr != learning_rate
        ):
            raise ValueError("resume run settings changed")
        if steps <= read_json(checkpoint.parent / "training_state/training_step.json")["step"]:
            raise ValueError("resume must increase the saved optimization step")
        cfg.dataset.root, cfg.output_dir = str(root.resolve()), training_dir.resolve()
        cfg.resume, cfg.steps = True, steps
        argv = ["m4b", f"--config_path={checkpoint / 'train_config.json'}"]
    else:
        if training_dir.exists():
            raise FileExistsError("training directory exists; use resume")
        cfg = TrainPipelineConfig(
            dataset=DatasetConfig(
                repo_id=split["repo_id"],
                root=str(root.resolve()),
                episodes=train_ids,
                video_backend="pyav",
                use_imagenet_stats=False,
            ),
            policy=smolvla_config(device, assets, learning_rate),
            output_dir=training_dir.resolve(),
            batch_size=batch_size,
            steps=steps,
            seed=seed,
            num_workers=0,
            env_eval_freq=0,
            eval_steps=0,
            save_freq=min(steps, 5000),
            log_freq=min(steps, 100),
            cudnn_deterministic=True,
            wandb=WandBConfig(enable=wandb, project="langmani-m4b", disable_artifact=True),
        )
        argv = ["m4b"]
        write_json(output / "normalization.json", stats_payload)
    validate_config(cfg.policy)
    factory = upstream.make_train_eval_datasets
    processor_factory = upstream.make_pre_post_processors
    update = upstream.update_policy
    peak = 0

    def dataset_factory(config: Any) -> tuple[Any, None]:
        dataset, evaluation = factory(config)
        if evaluation is not None or dataset.episodes != train_ids:
            raise ValueError("upstream changed explicit train episodes")
        dataset.meta.stats = {
            key: {
                name: torch.as_tensor(value, dtype=torch.float32)
                for name, value in values.items()
                if name != "count"
            }
            for key, values in stats.items()
        }
        return dataset, None

    def processors(policy_cfg: Any, **kwargs: Any) -> Any:
        # A new robot needs fresh official processors with this train view's statistics.
        # Loading base-robot processors would retain unrelated features/tokenizer settings.
        if resume is None:
            return make_smolvla_pre_post_processors(
                policy_cfg, dataset_stats=kwargs["dataset_stats"]
            )
        return processor_factory(policy_cfg, **kwargs)

    def measured_update(*args: Any, **kwargs: Any) -> Any:
        nonlocal peak
        result = update(*args, **kwargs)
        if device.startswith("cuda"):
            peak = max(peak, torch.cuda.max_memory_allocated())
        return result

    started = time.perf_counter()
    with ExitStack() as stack:
        stack.enter_context(patch.object(upstream, "make_train_eval_datasets", dataset_factory))
        stack.enter_context(patch.object(upstream, "make_pre_post_processors", processors))
        stack.enter_context(patch.object(upstream, "update_policy", measured_update))
        stack.enter_context(patch.object(sys, "argv", argv))
        if platform.system() == "Windows":
            stack.enter_context(
                patch.object(
                    upstream,
                    "update_last_checkpoint",
                    lambda p: write_json(p.parent / "last_checkpoint.json", {"checkpoint": p.name}),
                )
            )
        upstream.train(cfg)
    elapsed = time.perf_counter() - started
    checkpoint = checkpoint_directory(get_step_checkpoint_dir(training_dir, steps, steps))
    if read_json(checkpoint.parent / "training_state/training_step.json")["step"] != steps:
        raise RuntimeError("upstream final step is missing")
    policy, pre, post = load_checkpoint(checkpoint, device)
    row = current[0]
    with torch.inference_mode():
        policy.reset()
        action = post(
            policy.select_action(pre(policy_observation(row[IMAGE], row[STATE], row["task"])))
        )
        if tuple(action.shape) != (1, 8) or not torch.isfinite(action).all():
            raise RuntimeError("reloaded SmolVLA/processor pair emitted invalid action")
    files = {
        p.relative_to(checkpoint.parent).as_posix(): file_digest(p)
        for p in checkpoint.parent.rglob("*")
        if p.is_file()
    }
    result = {
        "mode": mode,
        "steps": steps,
        "checkpoint": str(checkpoint.relative_to(output)),
        "checkpoint_sha256": file_digest(checkpoint / "model.safetensors"),
        "checkpoint_files": files,
        "checkpoint_tree_sha256": digest(files),
        "checkpoint_reloaded": True,
        "train_frames": len(current),
        "sample_presentations": steps * batch_size,
        "equivalent_train_passes": steps * batch_size / len(current),
        "training_invocation_seconds": elapsed,
        "optimizer_step_peak_allocated_vram_bytes": peak if device.startswith("cuda") else None,
        "versions": {
            name: version(name)
            for name in ("lerobot", "torch", "transformers", "tokenizers", "accelerate", "numpy")
        },
        "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None,
        "full_smolvla_trained": mode == "full",
        "closed_loop_evaluated": False,
    }
    write_json(output / "checkpoint_metadata.json", result)
    return result
