"""Launch one bounded official LeRobot SmolVLA Phase 2C training stage."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from langmani.v2.phase2c import Phase2CContractError, read_json_object, sha256_file, write_json_once

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "langmani_v2" / "phase2c_smolvla_push.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "micro", "full"), required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--prepare-root", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--episodes", type=int, nargs="*")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--lerobot-train", default="lerobot-train")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _positive(value: int | None, label: str) -> int | None:
    if value is not None and (isinstance(value, bool) or value < 1):
        raise Phase2CContractError(f"{label} must be positive")
    return value


def _stage_steps(protocol: dict[str, Any], stage: str) -> int:
    if stage == "smoke":
        return int(protocol["smoke"]["steps"])
    if stage == "micro":
        return int(protocol["micro_overfit"]["steps"])
    return int(protocol["full_training"]["total_steps"])


def build_command(args: argparse.Namespace, protocol: dict[str, Any]) -> tuple[list[str], int]:
    """Construct the exact installed LeRobot 0.6 command without shell interpolation."""

    if args.batch_size < 1 or args.num_workers < 0 or args.seed < 0:
        raise Phase2CContractError("batch size, workers, and seed are invalid")
    requested = _positive(args.max_steps, "max_steps") or _stage_steps(protocol, args.stage)
    ceiling = _stage_steps(protocol, args.stage)
    if requested > ceiling:
        raise Phase2CContractError(f"{args.stage} cannot exceed its frozen {ceiling}-step ceiling")
    limit = _positive(args.limit_samples, "limit_samples")
    if limit is not None and requested * args.batch_size > limit:
        requested = limit // args.batch_size
        if requested < 1:
            raise Phase2CContractError("limit_samples is smaller than one batch")
    checkpoint_every = _positive(args.checkpoint_every, "checkpoint_every")
    if checkpoint_every is None:
        checkpoint_every = 1 if args.stage == "smoke" else (100 if args.stage == "micro" else 5000)
    if checkpoint_every > requested:
        checkpoint_every = requested
    if args.stage == "micro" and (not args.episodes or len(args.episodes) != 16):
        raise Phase2CContractError("micro-overfit requires exactly 16 preselected train episodes")
    if args.stage == "full" and args.episodes:
        raise Phase2CContractError("full training must consume the complete train view")

    base = protocol["base_model"]
    model = protocol["model"]
    optimizer = protocol["optimizer"]
    scheduler = protocol["scheduler"]
    policy_arguments = [
        f"--policy.path={base['repo_id']}",
        f"--policy.pretrained_revision={base['revision']}",
        # LeRobot 0.6 recursively merges dictionary CLI overrides into the published
        # three-camera config. Null is the documented replacement semantic: make_policy
        # then derives the complete input mapping from the already verified train view.
        "--policy.input_features=null",
        f"--policy.chunk_size={model['chunk_size']}",
        f"--policy.n_action_steps={model['n_action_steps']}",
        f"--policy.num_steps={model['num_inference_steps']}",
        f"--policy.freeze_vision_encoder={str(model['freeze_vision_encoder']).lower()}",
        f"--policy.train_expert_only={str(model['train_expert_only']).lower()}",
        f"--policy.train_state_proj={str(model['train_state_proj']).lower()}",
        f"--policy.dtype={model['dtype']}",
        f"--policy.use_amp={str(model['use_amp']).lower()}",
        "--policy.load_vlm_weights=true",
        f"--policy.optimizer_lr={optimizer['learning_rate']}",
        f"--policy.optimizer_betas={optimizer['betas'][0]}",
        str(optimizer["betas"][1]),
        f"--policy.optimizer_eps={optimizer['epsilon']}",
        f"--policy.optimizer_weight_decay={optimizer['weight_decay']}",
        f"--policy.optimizer_grad_clip_norm={optimizer['gradient_clip_norm']}",
        f"--policy.scheduler_warmup_steps={scheduler['warmup_steps']}",
        f"--policy.scheduler_decay_steps={scheduler['decay_steps']}",
        f"--policy.scheduler_decay_lr={scheduler['decay_learning_rate']}",
    ]
    if args.resume_checkpoint is not None:
        train_config = args.resume_checkpoint.resolve() / "pretrained_model" / "train_config.json"
        if not train_config.is_file() and not args.dry_run:
            raise Phase2CContractError("resume checkpoint lacks pretrained_model/train_config.json")
        policy_arguments = ["--resume=true", f"--config_path={train_config}"]
    command = [
        args.lerobot_train,
        *policy_arguments,
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={args.train_root.resolve()}",
        "--dataset.return_uint8=true",
        "--dataset.video_backend=pyav",
        "--dataset.eval_split=0.0",
        f"--output_dir={args.output_dir.resolve()}",
        f"--job_name=phase2c-{args.stage}-seed{args.seed}",
        f"--seed={args.seed}",
        f"--num_workers={args.num_workers}",
        f"--batch_size={args.batch_size}",
        f"--steps={requested}",
        f"--save_freq={checkpoint_every}",
        "--eval_steps=0",
        "--env_eval_freq=0",
        "--wandb.enable=false",
        "--policy.push_to_hub=false",
        "--cudnn_deterministic=true",
    ]
    if args.episodes:
        command.append("--dataset.episodes=" + json.dumps(args.episodes, separators=(",", ":")))
    return command, requested


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol_path = args.protocol.resolve()
    protocol = read_json_object(protocol_path, "Phase 2C experiment protocol")
    complete_path = args.prepare_root.resolve() / "complete.json"
    complete = read_json_object(complete_path, "Phase 2C preparation completion marker")
    if complete.get("passed") is not True:
        raise Phase2CContractError("Phase 2C input preparation has not passed")
    if not args.train_root.resolve().is_dir():
        raise Phase2CContractError("immutable train root is unavailable")
    command, steps = build_command(args, protocol)
    output = args.output_dir.resolve()
    if output.exists() and args.resume_checkpoint is None:
        raise Phase2CContractError(f"refusing to overwrite training output: {output}")
    output.mkdir(parents=True, exist_ok=args.resume_checkpoint is not None)
    environment = {
        key: os.environ.get(key)
        for key in ("CUDA_VISIBLE_DEVICES", "HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE")
        if os.environ.get(key)
    }
    manifest = {
        "schema_version": "langmani-v2-phase2c-training-launch-v0",
        "stage": args.stage,
        "protocol_sha256": f"sha256:{sha256_file(protocol_path)}",
        "prepare_complete_sha256": f"sha256:{sha256_file(complete_path)}",
        "train_root": args.train_root.resolve().as_posix(),
        "repo_id": args.repo_id,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "effective_batch_size": args.batch_size,
        "gradient_accumulation": 1,
        "steps": steps,
        "sample_updates": steps * args.batch_size,
        "num_workers": args.num_workers,
        "episodes": args.episodes,
        "resume_checkpoint": (
            args.resume_checkpoint.resolve().as_posix() if args.resume_checkpoint else None
        ),
        "command": command,
        "selected_environment": environment,
        "dry_run": args.dry_run,
        "checkpoint_resume_supported": True,
        "periodic_offline_validation": "run on immutable validation root after each frozen checkpoint",
        "feature_binding": {
            "semantic": "complete_dataset_metadata_inference_v0",
            "expected_input_features": protocol["feature_override"]["input_features"],
            "expected_output_features": protocol["feature_override"]["output_features"],
            "dictionary_cli_merge_prohibited": True,
        },
    }
    write_json_once(output / "launch_manifest.json", manifest)
    if args.dry_run:
        print(json.dumps(manifest, sort_keys=True))
        return 0
    started = time.time()
    log_path = output / "train.log"
    with log_path.open("x", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env={**os.environ, "HF_HUB_DISABLE_XET": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            raise
        return_code = process.wait()
    result = {
        "schema_version": "langmani-v2-phase2c-training-exit-v0",
        "stage": args.stage,
        "return_code": return_code,
        "elapsed_seconds": time.time() - started,
        "log_sha256": f"sha256:{sha256_file(log_path)}",
        "completed": return_code == 0,
    }
    write_json_once(output / "exit.json", result)
    print(json.dumps(result, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
