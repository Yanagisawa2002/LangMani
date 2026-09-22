"""Prepare, smoke-test, train and evaluate the controlled L1/L5/L10 comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from langmani.policies.m4a_data import digest, file_digest, load_manifest, read_json, write_json
from langmani.policies.m4b_data import check_split
from langmani.policies.m4b_protocol import make_schedule as m4b_schedule
from langmani.policies.m4b_training import train_official, verify_assets
from langmani.policies.m4c_language import check_manifest, make_manifest, make_schedule

ROOT = Path(__file__).resolve().parents[1]


def provenance() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()

    if git("status", "--porcelain"):
        raise ValueError("native experiment requires a clean implementation commit")
    return {"git_commit": git("rev-parse", "HEAD"), "python": sys.version, "command": sys.argv}


def safe_output(path: Path) -> Path:
    path = path.resolve()
    if not any(
        path.is_relative_to(ROOT / name) and path != ROOT / name for name in ("results", "outputs")
    ):
        raise ValueError("new M4C output must stay inside this checkout's results/ or outputs/")
    return path


def source_data(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = Path(config["dataset_root"])
    split, _ = check_split(root, Path(config["split"]), Path(config["validation"]))
    if (
        load_manifest(root).config.mode.value != "full"
        or len(split["train_episode_ids"]) != 96
        or split["train_frames"] != 17149
        or split["dataset_total_frames"] != 64548
    ):
        raise ValueError("formal frozen M4B robot-data view required")
    schedule = read_json(config["m4b_schedule"])
    if schedule != m4b_schedule(
        split["all_source_scene_seeds"], schedule["excluded_m4a_seeds"], 20, 43000
    ):
        raise ValueError("frozen 20-scene schedule differs")
    gate = Path(config["gate"])
    receipt = read_json(gate / "gate.json")
    if (
        not receipt["passed"]
        or receipt["schedule_sha256"] != schedule["schedule_sha256"]
        or read_json(gate / "schedule.json") != schedule
        or digest(read_json(gate / "episodes.json")["episodes"]) != receipt["episodes_sha256"]
    ):
        raise ValueError("frozen paired reference differs")
    assets = verify_assets(Path(config["assets"]))
    return split, schedule, assets


def protocol(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = read_json(path / "source.json")
    split, original, assets = source_data(config)
    manifest = read_json(path / "language_manifest.json")
    check_manifest(manifest, split)
    schedule = make_schedule(manifest, original)
    if read_json(path / "schedule.json") != schedule:
        raise ValueError("language intervention schedule differs")
    validation = read_json(path / "language_validation.json")
    if (
        not validation["passed"]
        or validation["language_manifest_sha256"] != manifest["language_manifest_sha256"]
        or validation["assets_sha256"] != digest(assets)
    ):
        raise ValueError("tokenizer/language validation differs")
    return config, split, manifest, schedule, assets


def checkpoint(run: Path) -> tuple[Path, dict[str, Any]]:
    metadata = read_json(run / "checkpoint_metadata.json")
    path = (run / metadata["checkpoint"]).resolve(strict=True)
    if not path.is_relative_to((run / "training/checkpoints").resolve()):
        raise ValueError("checkpoint escapes its own run")
    actual = {
        p.relative_to(path.parent).as_posix(): file_digest(p)
        for p in path.parent.rglob("*")
        if p.is_file()
    }
    if (
        actual != metadata["checkpoint_files"]
        or file_digest(path / "model.safetensors") != metadata["checkpoint_sha256"]
    ):
        raise ValueError("checkpoint model/processors/optimizer inventory differs")
    return path, metadata


def compare_runs(
    runs: list[Path], manifest: dict[str, Any], *, smoke: bool, control: Path | None = None
) -> dict[str, Any]:
    metadata = [read_json(p / "m4c_training_metadata.json") for p in runs]
    if {m["language"]["level"] for m in metadata} != {"L1", "L5", "L10"} or len(metadata) != 3:
        raise ValueError("one complete run for each L1/L5/L10 is required")
    steps = 20 if smoke else 20000
    identities = []
    for run, m in zip(runs, metadata, strict=True):
        _, frozen = checkpoint(run)
        identity = read_json(run / "config.json")["identity"]
        if (
            identity["level"] != m["language"]["level"]
            or identity["steps"] != steps
            or identity["smoke"] != smoke
            or identity["batch_size"] != 8
            or identity["learning_rate"] != 1e-4
            or identity["seed"] != 0
            or any(m[k] != v for k, v in frozen.items())
        ):
            raise ValueError("run identity/checkpoint provenance differs")
        identities.append({k: v for k, v in identity.items() if k != "level"})
        if (
            m["steps"] != steps
            or frozen["steps"] != steps
            or m["language"]["start_step"] != 0
            or m["language"]["sample_presentations"] != steps * 8
            or m["language"]["language_manifest_sha256"] != manifest["language_manifest_sha256"]
        ):
            raise ValueError("incomplete or incompatible fixed-budget group")
        samples = run / m["language_samples_file"]
        if file_digest(samples) != m["language"]["samples_csv_sha256"]:
            raise ValueError("sample trace differs")
    if len({digest(i) for i in identities}) != 1:
        raise ValueError("comparison confounded by different run configuration/provenance")
    for field in ("initial_model_state_sha256", "robot_sample_sequence_sha256"):
        if len({m["language"][field] for m in metadata}) != 1:
            raise ValueError(f"comparison confounded by different {field}")
    if len({digest(read_json(run / "normalization.json")) for run in runs}) != 1:
        raise ValueError("normalization differs across language groups")
    matched = None
    if control is not None:
        _, control_meta = checkpoint(control)
        l1 = next(m for m in metadata if m["language"]["level"] == "L1")
        if (
            control_meta["steps"] != 20
            or control_meta["checkpoint_sha256"] != l1["checkpoint_sha256"]
        ):
            raise ValueError(
                "L1 wrapper is not numerically equivalent to unwrapped official training"
            )
        matched = l1["checkpoint_sha256"]
    return {
        "passed": True,
        "smoke": smoke,
        "language_manifest_sha256": manifest["language_manifest_sha256"],
        "groups": [
            {
                "level": m["language"]["level"],
                "run": str(p.resolve()),
                "metadata_sha256": file_digest(p / "m4c_training_metadata.json"),
            }
            for p, m in zip(runs, metadata, strict=True)
        ],
        "identical_initial_model_state_sha256": metadata[0]["language"][
            "initial_model_state_sha256"
        ],
        "identical_robot_sample_sequence_sha256": metadata[0]["language"][
            "robot_sample_sequence_sha256"
        ],
        "sample_presentations_per_group": steps * 8,
        "l1_unwrapped_checkpoint_match": matched,
    }


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    for name in ("dataset-root", "validation", "split", "m4b-schedule", "assets", "gate"):
        prepare.add_argument(f"--{name}", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    for name in ("train", "control-smoke", "evaluate", "compare"):
        p = commands.add_parser(name)
        p.add_argument("--protocol", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        if name in ("train", "evaluate", "compare"):
            p.add_argument("--smoke", action="store_true")
        if name == "train":
            p.add_argument("--level", choices=("L1", "L5", "L10"), required=True)
            p.add_argument("--smoke-validation", type=Path)
            p.add_argument("--resume", type=Path)
        if name == "evaluate":
            p.add_argument("--run", type=Path, required=True)
        if name == "compare":
            p.add_argument("--runs", nargs=3, type=Path, required=True)
            p.add_argument("--control", type=Path)
    return parser.parse_args()


def execute(args: argparse.Namespace) -> dict[str, Any]:
    origin = provenance()
    output = safe_output(args.output)
    if args.command == "prepare":
        if output.exists():
            raise FileExistsError("protocol exists")
        config = {
            key: str(getattr(args, key).resolve(strict=True))
            for key in ("dataset_root", "validation", "split", "m4b_schedule", "assets", "gate")
        }
        source = Path(config["dataset_root"])
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("output overlaps immutable data")
        split, original, assets = source_data(config)
        manifest = make_manifest(split)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(assets["backbone"]["path"], local_files_only=True)
        lengths = []
        for row in manifest["training_catalog"] + manifest["held_out_catalog"]:
            text = row["instruction_text"] + "\n"  # exact upstream NewLineTaskProcessor contract
            tokens = tokenizer(text, truncation=False)["input_ids"]
            if (
                len(tokens) > 48
                or tokens != tokenizer(text, truncation=True, max_length=48)["input_ids"]
            ):
                raise ValueError("instruction is truncated by the frozen tokenizer limit")
            lengths.append(
                {
                    "template_id": row["template_id"],
                    "goal": row["semantic_goal_id"],
                    "tokens": len(tokens),
                }
            )
        validation = {
            "passed": True,
            "language_manifest_sha256": manifest["language_manifest_sha256"],
            "assets_sha256": digest(assets),
            "maximum_tokens": max(r["tokens"] for r in lengths),
            "token_lengths": lengths,
            "leakage": manifest["validation"],
            "provenance": origin,
        }
        write_json(output / "source.json", config)
        write_json(output / "language_manifest.json", manifest)
        write_json(output / "schedule.json", make_schedule(manifest, original))
        write_json(output / "language_validation.json", validation)
        return validation
    config, split, manifest, schedule, assets = protocol(args.protocol)
    source = Path(config["dataset_root"])
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("output overlaps immutable data")
    if args.command == "compare":
        if output.exists():
            raise FileExistsError("comparison receipt exists")
        result = compare_runs(args.runs, manifest, smoke=args.smoke, control=args.control)
        write_json(output, result)
        return result
    if args.command in ("train", "control-smoke"):
        smoke = args.command == "control-smoke" or args.smoke
        if not smoke:
            if args.smoke_validation is None:
                raise ValueError(
                    "native wrapper-equivalence smoke is required before full training"
                )
            receipt = read_json(args.smoke_validation)
            if (
                not receipt["passed"]
                or not receipt["smoke"]
                or not receipt["l1_unwrapped_checkpoint_match"]
                or receipt["language_manifest_sha256"] != manifest["language_manifest_sha256"]
            ):
                raise ValueError("smoke receipt is missing or incompatible")
            for group in receipt["groups"]:
                if (
                    file_digest(Path(group["run"]) / "m4c_training_metadata.json")
                    != group["metadata_sha256"]
                ):
                    raise ValueError("validated smoke metadata changed")
        identity = {
            "level": getattr(args, "level", "unwrapped-control"),
            "smoke": smoke,
            "language_manifest_sha256": manifest["language_manifest_sha256"],
            "dataset_content_sha256": split["content_sha256"],
            "assets_sha256": digest(assets),
            "schedule_sha256": schedule["schedule_sha256"],
            "steps": 20 if smoke else 20000,
            "batch_size": 8,
            "learning_rate": 1e-4,
            "seed": 0,
            "git_commit": origin["git_commit"],
        }
        resume = getattr(args, "resume", None)
        if resume is not None:
            if (output / "m4c_training_metadata.json").exists():
                raise FileExistsError("completed training must not be repeated")
            selected_step = read_json(resume.parent / "training_state/training_step.json")["step"]
            if any(
                p.is_dir() and p.name.isdigit() and int(p.name) > selected_step
                for p in (output / "training/checkpoints").iterdir()
            ):
                raise ValueError(
                    "later checkpoint directory exists; preserve/verify it before resume"
                )
            if read_json(output / "config.json")["identity"] != identity:
                raise ValueError("resume identity differs")
        else:
            if output.exists():
                raise FileExistsError("run exists; choose new output or compatible resume")
            write_json(
                output / "config.json",
                {
                    "identity": identity,
                    "provenance": origin,
                    "protocol": str(args.protocol.resolve()),
                },
            )
        if args.command == "control-smoke":
            result = train_official(
                root=Path(config["dataset_root"]),
                output=output,
                split=split,
                assets=assets,
                mode="smoke",
                device="cuda",
                batch_size=8,
                steps=20,
                seed=0,
                learning_rate=1e-4,
                wandb=False,
            )
        else:
            from langmani.policies.m4c_training import train_language

            result = train_language(
                root=Path(config["dataset_root"]),
                output=output,
                split=split,
                assets=assets,
                manifest=manifest,
                level=args.level,
                smoke=smoke,
                resume=resume,
            )
        result.update(passed=True, full_dataset_validated=True, smoke_tested=smoke)
        write_json(output / "status.json", result)
        return result
    path, metadata = checkpoint(args.run)
    identity = read_json(args.run / "config.json")["identity"]
    if (
        identity["language_manifest_sha256"] != manifest["language_manifest_sha256"]
        or identity["schedule_sha256"] != schedule["schedule_sha256"]
        or (not args.smoke and (metadata["steps"] != 20000 or identity["smoke"]))
    ):
        raise ValueError("evaluation checkpoint belongs to a different language protocol/budget")
    from langmani.policies.m4c_evaluation import evaluate

    return evaluate(
        checkpoint=path,
        output=output,
        schedule=schedule,
        gate=Path(config["gate"]),
        provenance={**origin, "training_identity": identity},
        smoke=args.smoke,
    )


def main() -> int:
    try:
        result = execute(parse())
    except Exception as error:
        traceback.print_exc()
        print(
            json.dumps({"passed": False, "error": f"{type(error).__name__}: {error}"}), flush=True
        )
        return 1
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                in (
                    "passed",
                    "maximum_tokens",
                    "steps",
                    "checkpoint_sha256",
                    "closed_loop_evaluated",
                    "sample_presentations_per_group",
                )
            }
        ),
        flush=True,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
