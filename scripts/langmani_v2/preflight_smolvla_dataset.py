"""Run one real Phase 2B.2 SmolVLA batch through the official processor with zero optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from langmani.datasets.lerobot_types import ACTION_FEATURE_KEY, IMAGE_FEATURE_KEY, STATE_FEATURE_KEY
from langmani.v2.phase2b2 import PHASE2B2_DATASET_ID, write_json_once
from langmani.v2.phase2c import read_json_object, sha256_file
from langmani.v2.push_dataset import PushDatasetContractError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--base-model-root", type=Path, required=True)
    parser.add_argument("--base-model-manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _shape(value: object) -> list[int]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise PushDatasetContractError("processed batch field lacks a tensor shape")
    return [int(item) for item in shape]


def _verify_base_snapshot(root: Path, manifest: dict[str, Any]) -> dict[str, object]:
    base = manifest.get("base_model")
    if not isinstance(base, dict) or base.get("repo_id") != "lerobot/smolvla_base":
        raise PushDatasetContractError("reviewed SmolVLA base manifest is malformed")
    config = base.get("config")
    weights = base.get("weights")
    if not isinstance(config, dict) or not isinstance(weights, dict):
        raise PushDatasetContractError("reviewed SmolVLA file identities are malformed")
    expected = {
        "config.json": config.get("sha256"),
        "model.safetensors": weights.get("sha256"),
        "policy_preprocessor.json": base.get("preprocessor_sha256"),
        "policy_postprocessor.json": base.get("postprocessor_sha256"),
    }
    files: dict[str, object] = {}
    for name, digest in expected.items():
        path = root / name
        actual = f"sha256:{sha256_file(path)}"
        if actual != digest:
            raise PushDatasetContractError(f"reviewed SmolVLA base file changed: {name}")
        files[name] = {"size_bytes": path.stat().st_size, "sha256": actual}
    return {
        "repo_id": base["repo_id"],
        "revision": base["revision"],
        "files": files,
    }


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise PushDatasetContractError("batch-size must be positive")
    try:
        import lerobot  # type: ignore[import-untyped]
        from lerobot.datasets import (  # type: ignore[import-untyped]
            LeRobotDataset,
            LeRobotDatasetMetadata,
        )
        from lerobot.datasets.factory import (  # type: ignore[import-untyped]
            resolve_delta_timestamps,
        )
        from lerobot.policies.factory import (  # type: ignore[import-untyped]
            make_pre_post_processors,
        )
        from lerobot.policies.smolvla.modeling_smolvla import (  # type: ignore[import-untyped]
            SmolVLAPolicy,
        )
    except ImportError as error:
        raise PushDatasetContractError(
            "official LeRobot 0.6 SmolVLA APIs are unavailable"
        ) from error
    if getattr(lerobot, "__version__", None) != "0.6.0":
        raise PushDatasetContractError("SmolVLA preflight requires LeRobot 0.6.0")
    base_root = args.base_model_root.resolve()
    manifest_path = args.base_model_manifest.resolve()
    manifest = read_json_object(manifest_path, "reviewed SmolVLA base manifest")
    base_identity = _verify_base_snapshot(base_root, manifest)
    policy = SmolVLAPolicy.from_pretrained(str(base_root))
    preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path=str(base_root))
    target_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    device = torch.device(args.device)
    policy.to(device=device, dtype=target_dtype)
    policy.eval()
    train_root = args.export_root.resolve() / "splits" / "train"
    repo_id = f"{PHASE2B2_DATASET_ID}-train"
    metadata = LeRobotDatasetMetadata(repo_id, root=train_root)
    deltas = resolve_delta_timestamps(policy.config, metadata)
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=train_root,
        delta_timestamps=deltas,
        video_backend="pyav",
        return_uint8=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    raw = next(iter(loader))
    if not isinstance(raw, dict):
        raise PushDatasetContractError("real LeRobot loader returned a non-mapping batch")
    if _shape(raw[IMAGE_FEATURE_KEY])[1:] != [3, 256, 256]:
        raise PushDatasetContractError("real batch RGB shape changed")
    if _shape(raw[STATE_FEATURE_KEY])[-1:] != [9]:
        raise PushDatasetContractError("real batch state shape changed")
    if _shape(raw[ACTION_FEATURE_KEY])[-1:] != [8]:
        raise PushDatasetContractError("real batch action shape changed")
    if not all(isinstance(item, str) and item.strip() for item in raw["task"]):
        raise PushDatasetContractError("real batch task text is unreadable")
    processed = preprocessor(raw)
    if not isinstance(processed, dict):
        raise PushDatasetContractError("official SmolVLA processor returned a non-mapping")
    construction = manifest.get("construction")
    if not isinstance(construction, dict) or construction.get("passed") is not True:
        raise PushDatasetContractError("reviewed SmolVLA construction evidence is not accepted")
    parameters = list(policy.parameters())
    parameter_count = sum(int(item.numel()) for item in parameters)
    trainable_count = sum(int(item.numel()) for item in parameters if item.requires_grad)
    if parameter_count != construction.get(
        "parameter_count"
    ) or trainable_count != construction.get("trainable_parameter_count"):
        raise PushDatasetContractError("loaded SmolVLA parameter identity changed")
    report = {
        "schema_version": "langmani-v2-phase2b2-smolvla-real-batch-preflight-v0",
        "dataset_id": PHASE2B2_DATASET_ID,
        "dataset": {
            "repo_id": repo_id,
            "episodes": int(dataset.num_episodes),
            "frames": int(dataset.num_frames),
            "batch_size": int(_shape(raw[STATE_FEATURE_KEY])[0]),
        },
        "raw_batch_shapes": {
            IMAGE_FEATURE_KEY: _shape(raw[IMAGE_FEATURE_KEY]),
            STATE_FEATURE_KEY: _shape(raw[STATE_FEATURE_KEY]),
            ACTION_FEATURE_KEY: _shape(raw[ACTION_FEATURE_KEY]),
        },
        "processed_shapes": {
            key: _shape(value)
            for key, value in processed.items()
            if isinstance(value, torch.Tensor)
        },
        "base_model": base_identity,
        "base_model_manifest_sha256": f"sha256:{sha256_file(manifest_path)}",
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "device": args.device,
        "dtype": args.dtype,
        "data_loader_workers": 0,
        "forward_calls": 0,
        "backward_calls": 0,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "training_started": False,
        "passed": True,
    }
    write_json_once(args.output.resolve(), report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
