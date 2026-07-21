"""Content-bind one official SmolVLA checkpoint and emit its external adapter config."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from langmani.v2.phase2c import Phase2CContractError, read_json_object, sha256_file, write_json_once
from langmani.v2.smolvla_adapter import (
    OFFICIAL_BASE_MODEL,
    SmolVLAAdapterConfig,
    sha256_directory,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "langmani_v2" / "phase2c_smolvla_push.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--train-view-manifest", type=Path, required=True)
    parser.add_argument("--normalization-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execution-horizon", type=int, choices=(1, 4, 8), required=True)
    parser.add_argument("--inference-seed", type=int, required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _processor_digest(root: Path) -> str:
    files = sorted(
        item
        for item in root.rglob("*")
        if item.is_file()
        and (
            item.name.startswith("policy_preprocessor")
            or item.name.startswith("policy_postprocessor")
        )
    )
    if not files:
        raise Phase2CContractError("checkpoint lacks serialized policy processors")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(root).as_posix().encode()
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint = args.checkpoint.resolve()
    required = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    )
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise Phase2CContractError("checkpoint is incomplete: " + ", ".join(missing))
    protocol = read_json_object(args.protocol.resolve(), "Phase 2C protocol")
    base = protocol.get("base_model")
    if not isinstance(base, dict) or base.get("repo_id") != OFFICIAL_BASE_MODEL:
        raise Phase2CContractError("protocol does not bind the official SmolVLA base")
    launch = read_json_object(args.launch_manifest.resolve(), "training launch manifest")
    if launch.get("dry_run") is True:
        raise Phase2CContractError("a dry-run launch cannot own a real checkpoint")
    training_seed = launch.get("seed")
    if isinstance(training_seed, bool) or not isinstance(training_seed, int) or training_seed < 0:
        raise Phase2CContractError("training launch lacks a valid seed")
    train_view = read_json_object(args.train_view_manifest.resolve(), "train view manifest")
    normalization = read_json_object(
        args.normalization_manifest.resolve(), "normalization manifest"
    )
    if train_view.get("passed") is not True or normalization.get("passed") is not True:
        raise Phase2CContractError("checkpoint inputs are not accepted")
    config = SmolVLAAdapterConfig(
        policy_id=args.policy_id,
        checkpoint_path=checkpoint.as_posix(),
        checkpoint_sha256=sha256_directory(checkpoint),
        base_model=OFFICIAL_BASE_MODEL,
        base_model_revision=str(base["revision"]),
        training_config_sha256=sha256_file(args.launch_manifest.resolve()),
        train_view_sha256=sha256_file(args.train_view_manifest.resolve()),
        normalization_sha256=sha256_file(args.normalization_manifest.resolve()),
        processor_sha256=_processor_digest(checkpoint),
        training_seed=training_seed,
        execution_horizon=args.execution_horizon,
        device=args.device,
        dtype=args.dtype,
        inference_seed=args.inference_seed,
    )
    write_json_once(args.output.resolve(), config.to_dict())
    print(json.dumps(config.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
