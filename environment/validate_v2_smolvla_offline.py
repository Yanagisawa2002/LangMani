"""Evaluate frozen SmolVLA checkpoints on the immutable Phase 2B validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from langmani.datasets.lerobot_types import ACTION_FEATURE_KEY
from langmani.v2.phase2c import Phase2CContractError, sha256_file, write_json_once
from langmani.v2.phase2c_offline import OfflineMetricAccumulator, select_offline_checkpoints
from langmani.v2.smolvla_adapter import sha256_directory

VALIDATION_EPISODES = 77
VALIDATION_FRAMES = 10_425


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--select", action="store_true")
    return parser.parse_args(argv)


def _numpy(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _slice_batch(batch: dict[str, Any], size: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in batch.items():
        try:
            result[key] = value[:size]
        except (TypeError, IndexError) as error:
            raise Phase2CContractError(f"cannot bound validation batch field {key}") from error
    return result


def _dataset(*, root: Path, repo_id: str, policy_config: object) -> object:
    try:
        from lerobot.datasets import (  # type: ignore[import-untyped]
            LeRobotDataset,
            LeRobotDatasetMetadata,
        )
        from lerobot.datasets.factory import (  # type: ignore[import-untyped]
            resolve_delta_timestamps,
        )
    except ImportError as error:
        raise Phase2CContractError("LeRobot 0.6 dataset APIs are required") from error
    metadata = LeRobotDatasetMetadata(repo_id, root=root)
    delta_timestamps = resolve_delta_timestamps(policy_config, metadata)
    return LeRobotDataset(
        repo_id=repo_id,
        root=root,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
        return_uint8=True,
    )


def _load_components(checkpoint: Path, *, device: str, dtype: str) -> tuple[Any, Any, Any]:
    try:
        from lerobot.policies.factory import (  # type: ignore[import-untyped]
            make_pre_post_processors,
        )
        from lerobot.policies.smolvla.modeling_smolvla import (  # type: ignore[import-untyped]
            SmolVLAPolicy,
        )
    except ImportError as error:
        raise Phase2CContractError("official LeRobot SmolVLA APIs are required") from error
    policy = SmolVLAPolicy.from_pretrained(str(checkpoint))
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
    )
    target_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    policy.to(device=torch.device(device), dtype=target_dtype)
    policy.eval()
    return policy, preprocessor, postprocessor


def _evaluate_checkpoint(args: argparse.Namespace, checkpoint: Path) -> dict[str, object]:
    policy, preprocessor, postprocessor = _load_components(
        checkpoint,
        device=args.device,
        dtype=args.dtype,
    )
    dataset: Any = _dataset(
        root=args.validation_root.resolve(),
        repo_id=args.repo_id,
        policy_config=policy.config,
    )
    if int(dataset.num_episodes) != VALIDATION_EPISODES:
        raise Phase2CContractError("validation episode count changed from 77")
    if int(dataset.num_frames) != VALIDATION_FRAMES:
        raise Phase2CContractError("validation frame count changed from 10,425")
    if args.batch_size < 1 or args.num_workers < 0 or args.seed < 0:
        raise Phase2CContractError("offline validation runtime arguments are invalid")
    if args.max_samples is not None and args.max_samples < 1:
        raise Phase2CContractError("max_samples must be positive")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    accumulator = OfflineMetricAccumulator(
        horizon=int(policy.config.chunk_size),
        action_dimension=8,
        per_dimension_absolute_sum=np.zeros(8, dtype=np.float64),
        per_dimension_count=np.zeros(8, dtype=np.int64),
        per_horizon_absolute_sum=np.zeros(int(policy.config.chunk_size), dtype=np.float64),
        per_horizon_count=np.zeros(int(policy.config.chunk_size), dtype=np.int64),
        prediction_sum=np.zeros(8, dtype=np.float64),
        prediction_squared_sum=np.zeros(8, dtype=np.float64),
    )
    seen = 0
    for batch_ordinal, raw_batch in enumerate(loader):
        if not isinstance(raw_batch, dict):
            raise Phase2CContractError("validation loader returned a non-mapping batch")
        remaining = (
            int(args.max_samples) - seen if args.max_samples is not None else len(raw_batch["task"])
        )
        if remaining <= 0:
            break
        batch_size = min(len(raw_batch["task"]), remaining)
        raw_batch = _slice_batch(raw_batch, batch_size)
        raw_actions = _numpy(raw_batch[ACTION_FEATURE_KEY])
        reset = getattr(policy, "reset", None)
        if callable(reset):
            reset()
        for processor in (preprocessor, postprocessor):
            reset_processor = getattr(processor, "reset", None)
            if callable(reset_processor):
                reset_processor()
        cuda_devices = [torch.cuda.current_device()] if args.device == "cuda" else []
        with torch.inference_mode(), torch.random.fork_rng(devices=cuda_devices):
            seed = args.seed + batch_ordinal
            torch.manual_seed(seed)
            if args.device == "cuda":
                torch.cuda.manual_seed_all(seed)
            processed = preprocessor(raw_batch)
            if not isinstance(processed, dict):
                raise Phase2CContractError("official preprocessor returned a non-mapping")
            forward = policy(processed)
            if not isinstance(forward, tuple) or len(forward) != 2:
                raise Phase2CContractError("official SmolVLA forward result contract changed")
            loss = float(forward[0].detach().cpu())
            predicted_normalized = policy.predict_action_chunk(dict(processed))
            predicted_raw = postprocessor(predicted_normalized)
        expected_normalized = processed[ACTION_FEATURE_KEY]
        action_is_pad = processed.get("action_is_pad")
        accumulator.add(
            validation_loss=loss,
            predicted_normalized=_numpy(predicted_normalized),
            expected_normalized=_numpy(expected_normalized),
            predicted_raw=_numpy(predicted_raw),
            expected_raw=raw_actions,
            action_is_pad=_numpy(action_is_pad) if action_is_pad is not None else None,
        )
        seen += batch_size
        print(
            json.dumps(
                {
                    "checkpoint": checkpoint.name,
                    "samples": seen,
                    "total": args.max_samples or VALIDATION_FRAMES,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    metrics = accumulator.finalize()
    complete = seen == VALIDATION_FRAMES and args.max_samples is None
    return {
        "schema_version": "langmani-v2-phase2c-offline-validation-v0",
        "split": "validation",
        "checkpoint_path": checkpoint.as_posix(),
        "checkpoint_sha256": sha256_directory(checkpoint),
        "repo_id": args.repo_id,
        "validation_root": args.validation_root.resolve().as_posix(),
        "dataset_counts": {"episodes": VALIDATION_EPISODES, "frames": VALIDATION_FRAMES},
        "processed_samples": seen,
        "deterministic_seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "metrics": metrics,
        "partial_smoke": not complete,
        "completed": complete,
        "selection_eligible": complete,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, object]] = []
    for checkpoint_value in args.checkpoint:
        checkpoint = checkpoint_value.resolve()
        report = _evaluate_checkpoint(args, checkpoint)
        digest = str(report["checkpoint_sha256"])
        report_path = output / f"checkpoint-{digest[:12]}.json"
        write_json_once(report_path, report)
        report["report_sha256"] = f"sha256:{sha256_file(report_path)}"
        reports.append(report)
    registry = {
        "schema_version": "langmani-v2-phase2c-offline-registry-v0",
        "reports": reports,
        "completed": all(item["completed"] is True for item in reports),
    }
    write_json_once(output / "checkpoint_registry.json", registry)
    if args.select:
        if not bool(registry["completed"]):
            raise Phase2CContractError("partial offline smoke reports cannot select checkpoints")
        selection = select_offline_checkpoints(reports, maximum=2)
        write_json_once(output / "checkpoint_selection.json", selection)
        print(json.dumps(selection, sort_keys=True))
    else:
        print(json.dumps(registry, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
