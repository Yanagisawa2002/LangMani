"""M4B: validate, freeze splits/scenes, verify native pairing, train and evaluate SmolVLA."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from langmani.policies.m4a_data import digest, file_digest, load_manifest, read_json, write_json
from langmani.policies.m4b_data import check_split, make_split, validate_dataset
from langmani.policies.m4b_protocol import make_schedule

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


def safe_output(path: Path, source: Path) -> Path:
    path, source = path.resolve(), source.resolve()
    if not any(path.is_relative_to(PROJECT_ROOT / name) for name in ("outputs", "results")):
        raise ValueError("M4B outputs must stay in this checkout's outputs/ or results/")
    if path.is_relative_to(source) or source.is_relative_to(path):
        raise ValueError("output overlaps immutable source")
    return path


def check_schedule(path: Path, split: dict[str, Any]) -> dict[str, Any]:
    schedule = read_json(path)
    expected = make_schedule(
        split["all_source_scene_seeds"],
        schedule["excluded_m4a_seeds"],
        len(schedule["scenes"]),
        schedule["seed"],
    )
    if schedule != expected:
        raise ValueError("evaluation schedule is not reproducible from its frozen exclusions")
    return schedule


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "prepare", "gate", "train", "evaluate"):
        child = sub.add_parser(command)
        child.add_argument("--dataset-root", required=True, type=Path)
        child.add_argument("--output", required=True, type=Path)
        if command != "validate":
            child.add_argument("--validation", required=True, type=Path)
        if command == "prepare":
            child.add_argument("--m4a-evaluation-config", required=True, type=Path)
            child.add_argument("--scenes", type=int, default=20)
            child.add_argument("--seed", type=int, default=43000)
        if command in {"gate", "train", "evaluate"}:
            child.add_argument("--split", required=True, type=Path)
            child.add_argument("--schedule", required=True, type=Path)
        if command in {"gate", "evaluate"}:
            child.add_argument(
                "--sim-backend", choices=("physx_cpu", "physx_cuda"), default="physx_cpu"
            )
        if command in {"train", "evaluate"}:
            child.add_argument("--assets", required=True, type=Path)
            child.add_argument("--gate", required=True, type=Path)
            child.add_argument("--device", default="cuda")
        if command == "train":
            child.add_argument("--mode", choices=("smoke", "full"), required=True)
            child.add_argument("--steps", type=int)
            child.add_argument("--batch-size", type=int, default=8)
            child.add_argument("--learning-rate", type=float, default=1e-4)
            child.add_argument("--seed", type=int, default=0)
            child.add_argument("--resume", type=Path)
            child.add_argument("--wandb", action="store_true")
        if command == "evaluate":
            child.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args(argv)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    root = args.dataset_root.resolve(strict=True)
    output = safe_output(args.output, root)
    if args.command == "validate":
        if output.exists():
            raise FileExistsError("preserve prior validation; choose a new receipt")
        report = validate_dataset(root)
        report["provenance"] = provenance()
        write_json(output, report)
        return {
            "passed": True,
            "full_dataset_validated": load_manifest(root).config.mode.value == "full",
            "episodes": report["num_episodes"],
            "frames": report["num_frames"],
        }
    if args.command == "prepare":
        if output.exists():
            raise FileExistsError("prepare output exists")
        report = read_json(args.validation)
        split = make_split(load_manifest(root), report)
        # Publish only after checking all bytes against the decoded validation receipt.
        actual = {
            p.relative_to(root).as_posix(): file_digest(p)
            for p in sorted(root.rglob("*"))
            if p.is_file()
        }
        if (
            actual != report["files"]
            or digest(actual) != report["content_sha256"]
            or "m4b_pair_audit" not in report
        ):
            raise ValueError("validation no longer matches source")
        prior = read_json(args.m4a_evaluation_config)
        if (
            digest(prior["seeds"]) != prior["schedule_sha256"]
            or prior["schedule_sha256"]
            != "d12a28d002aca9cca203a78dde12b6fd761272ba9d4f9f0528b576c63d074ae9"
        ):
            raise ValueError("M4A exclusion schedule differs from frozen final evaluation")
        schedule = make_schedule(
            split["all_source_scene_seeds"], prior["seeds"], args.scenes, args.seed
        )
        write_json(output / "split.json", split)
        write_json(output / "schedule.json", schedule)
        return {
            "passed": True,
            "selected_episodes": len(split["episodes"]),
            "selected_frames": split["total_frames"],
            "schedule_sha256": schedule["schedule_sha256"],
        }
    split, validation = check_split(root, args.split, args.validation)
    schedule = check_schedule(args.schedule, split)
    source = provenance()
    if args.command == "gate":
        from langmani.policies.m4b_evaluation import native_gate

        result = native_gate(schedule, output, args.sim_backend)
        write_json(output / "provenance.json", source)
        return result
    from langmani.policies.m4b_training import verify_assets

    assets = verify_assets(args.assets)
    gate = read_json(args.gate / "gate.json")
    if (
        gate["passed"] is not True
        or gate["schedule_sha256"] != schedule["schedule_sha256"]
        or digest(read_json(args.gate / "episodes.json")["episodes"]) != gate["episodes_sha256"]
    ):
        raise ValueError("accepted native paired-input gate is required")
    if args.command == "train":
        from langmani.policies.m4b_training import train_official

        steps = args.steps if args.steps is not None else (3 if args.mode == "smoke" else 20_000)
        if source["git_dirty"] and args.mode == "full":
            raise ValueError("full training requires a clean committed implementation")
        if args.mode == "full" and load_manifest(root).config.mode.value != "full":
            raise ValueError("full training requires formal full M3B data")
        identity = {
            "mode": args.mode,
            "split_sha256": split["split_sha256"],
            "schedule_sha256": schedule["schedule_sha256"],
            "assets_sha256": digest(assets),
            "device": args.device,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "git_commit": source["git_commit"],
        }
        if args.resume:
            if read_json(output / "config.json")["identity"] != identity:
                raise ValueError("resume identity differs from original source/config/data/assets")
        else:
            if output.exists():
                raise FileExistsError("output exists; use new run or resume")
            write_json(
                output / "config.json",
                {
                    "identity": identity,
                    "steps": steps,
                    "dataset_root": str(root),
                    "provenance": source,
                    "language_conditioning": True,
                    "observation_keys": [
                        "observation.images.base_camera",
                        "observation.state",
                        "task",
                    ],
                },
            )
            write_json(output / "split.json", split)
            write_json(output / "assets.json", assets)
            write_json(output / "validation.json", validation)
            write_json(output / "schedule.json", schedule)
        result = train_official(
            root=root,
            output=output,
            split=split,
            assets=assets,
            mode=args.mode,
            device=args.device,
            batch_size=args.batch_size,
            steps=steps,
            seed=args.seed,
            learning_rate=args.learning_rate,
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
    from langmani.policies.m4a_training import checkpoint_directory
    from langmani.policies.m4b_evaluation import evaluate

    checkpoint = checkpoint_directory(args.checkpoint)
    run = checkpoint.parents[3]
    if (
        read_json(run / "split.json") != split
        or read_json(run / "schedule.json") != schedule
        or read_json(run / "assets.json") != assets
    ):
        raise ValueError("checkpoint provenance differs from requested data/schedule/assets")
    metadata = read_json(run / "checkpoint_metadata.json")
    actual = {
        p.relative_to(checkpoint.parent).as_posix(): file_digest(p)
        for p in checkpoint.parent.rglob("*")
        if p.is_file()
    }
    if (
        actual != metadata["checkpoint_files"]
        or file_digest(checkpoint / "model.safetensors") != metadata["checkpoint_sha256"]
    ):
        raise ValueError("checkpoint model/processors/optimizer bytes changed")
    return evaluate(
        checkpoint=checkpoint,
        output=output,
        schedule=schedule,
        gate_path=args.gate,
        device=args.device,
        sim_backend=args.sim_backend,
        provenance=source,
    )


def main() -> int:
    args = parse_args()
    try:
        result = execute(args)
    except Exception as error:
        traceback.print_exc()
        result = {
            "passed": False,
            "command": args.command,
            "error": f"{type(error).__name__}: {error}",
        }
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
