"""M4A: validate, split, train upstream ACT, and compare paired closed-loop rollouts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from langmani.policies.m4a_data import (
    TASK_ID,
    check_split,
    file_digest,
    load_manifest,
    make_split,
    read_json,
    validate_root,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def provenance() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(PROJECT_ROOT), *args], text=True).strip()

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python": sys.version,
        "command": sys.argv,
    }


def safe_output(output: Path, dataset_root: Path) -> Path:
    output = output.resolve()
    dataset_root = dataset_root.resolve()
    if output.is_relative_to(dataset_root) or dataset_root.is_relative_to(output):
        raise ValueError("outputs must not overlap the immutable source dataset")
    for name in ("src", "tests", "docs", "scripts", "environment", ".git", "configs"):
        protected = (PROJECT_ROOT / name).resolve()
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("output overlaps repository source")
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "split", "train", "evaluate"):
        child = sub.add_parser(command)
        child.add_argument("--dataset-root", type=Path, required=True)
        if command in {"validate", "split"}:
            child.add_argument("--repo-id", help="must match the M3B local manifest")
        if command == "validate":
            child.add_argument("--report", type=Path)
        else:
            child.add_argument(
                "--output",
                type=Path,
                required=True,
                help="split JSON file, or a new training/evaluation run directory",
            )
        if command in {"train", "evaluate"}:
            child.add_argument("--split", type=Path, required=True)
            child.add_argument("--device", default="cuda")
            child.add_argument("--seed", type=int, default=0)
        if command == "train":
            child.add_argument("--mode", choices=("smoke", "full"), required=True)
            child.add_argument("--steps", type=int, help="default: smoke=3, full=100000")
            child.add_argument("--batch-size", type=int, default=8)
            child.add_argument("--wandb", action="store_true")
            child.add_argument("--resume", type=Path)
        if command == "evaluate":
            child.add_argument("--checkpoint", type=Path, required=True)
            child.add_argument("--episodes", type=int, default=20)
            child.add_argument(
                "--sim-backend", choices=("physx_cpu", "physx_cuda"), default="physx_cpu"
            )
    return parser.parse_args(argv)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    root = args.dataset_root.resolve(strict=True)
    if args.command in {"validate", "split"}:
        report = validate_root(root, args.repo_id)
        if args.command == "split":
            split = make_split(load_manifest(root), report)
            output = safe_output(args.output, root)
            if output.exists() and read_json(output) != split:
                raise FileExistsError("split exists with different contents")
            write_json(output, split)
            print(
                f"split: train={len(split['train_episode_ids'])} / held-out={len(split['held_out_episode_ids'])}"
            )
        elif args.report:
            write_json(safe_output(args.report, root), report)
        print(
            f"PASS: {report['num_episodes']} episodes, {report['num_frames']} frames; "
            "20 FPS, RGB uint8[3,256,256], state float32[9], action float32[8]"
        )
        print(json.dumps(report["statistics"], sort_keys=True))
        return {
            "passed": True,
            "full_dataset_validated": load_manifest(root).config.mode.value == "full",
        }

    output = safe_output(args.output, root)
    split, validation = check_split(root, args.split)
    source = provenance()
    if args.command == "train":
        from langmani.policies.m4a_training import train_official

        steps = args.steps if args.steps is not None else (3 if args.mode == "smoke" else 100_000)
        if source["git_dirty"] and args.mode == "full":
            raise ValueError("full training requires a clean committed implementation")
        identity = {
            "task_name": TASK_ID,
            "mode": args.mode,
            "split_sha256": split["split_sha256"],
            "device": args.device,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "git_commit": source["git_commit"],
        }
        if args.resume:
            if read_json(output / "config.json")["identity"] != identity:
                raise ValueError("resume run identity differs (source/split/config/commit)")
        else:
            if output.exists():
                raise FileExistsError("output exists; use a new run ID or --resume")
            write_json(
                output / "config.json",
                {
                    "identity": identity,
                    "steps": steps,
                    "dataset_root": str(root),
                    "provenance": source,
                    "language_conditioning": False,
                    "observation_keys": ["observation.images.base_camera", "observation.state"],
                },
            )
            write_json(output / "split.json", split)
            write_json(output / "validation.json", validation)
        result = train_official(
            root=root,
            output=output,
            split=split,
            mode=args.mode,
            device=args.device,
            batch_size=args.batch_size,
            steps=steps,
            seed=args.seed,
            wandb=args.wandb,
            resume=args.resume,
        )
        result.update(
            passed=True,
            smoke_tested=True,
            full_dataset_validated=load_manifest(root).config.mode.value == "full",
        )
        write_json(output / "status.json", result)
        return result

    from langmani.policies.m4a_evaluation import evaluate
    from langmani.policies.m4a_training import checkpoint_directory

    checkpoint = checkpoint_directory(args.checkpoint)
    training_run = checkpoint.parents[3]
    if read_json(training_run / "split.json") != split:
        raise ValueError("checkpoint belongs to a different dataset split")
    checkpoint_metadata = read_json(training_run / "checkpoint_metadata.json")
    if file_digest(checkpoint / "model.safetensors") != checkpoint_metadata["checkpoint_sha256"]:
        raise ValueError("checkpoint model checksum differs from its training receipt")
    for name, expected in checkpoint_metadata["checkpoint_files"].items():
        if file_digest(checkpoint / name) != expected:
            raise ValueError(f"checkpoint processor/config bytes changed: {name}")
    return evaluate(
        checkpoint=checkpoint,
        output=output,
        split=split,
        count=args.episodes,
        seed=args.seed,
        device=args.device,
        sim_backend=args.sim_backend,
        provenance=source,
    )


def main() -> int:
    args = parse_args()
    try:
        report = execute(args)
    except Exception as error:
        traceback.print_exc()
        report = {
            "passed": False,
            "command": args.command,
            "error": f"{type(error).__name__}: {error}",
        }
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
