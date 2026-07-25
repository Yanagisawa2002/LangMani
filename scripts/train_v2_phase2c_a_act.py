"""Smoke, micro-overfit, train, or diagnose a Phase 2C-A/A.1 ACT baseline."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from langmani.datasets.identity import sha256_hex
from langmani.policies.act_checkpoint import load_act_checkpoint
from langmani.v2.phase2c_a import (
    PACKAGE_FINGERPRINT,
    TASK_IDS,
    TRAIN_FRAMES,
    ModelKind,
    Phase2CAModelConfig,
    Phase2CARunIdentity,
    UniformTaskBatchSampler,
    primary_optimization_config,
)
from langmani.v2.phase2c_a_act import (
    ProcessorStatistics,
    load_dataset_view,
    load_processor_statistics,
    offline_diagnostics,
    run_micro_overfit,
    run_one_batch_gpu_smoke,
    train_primary_act,
)
from langmani.v2.phase2c_a1_bounded_act import (
    BoundedACTPolicyV1,
    identity_uses_bounded_action_head,
    model_identity_with_bounded_head,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATISTICS_PATH = (
    PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2b6_v2" / "dataset_statistics.json"
)
NORMALIZATION_PATH = (
    PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2b6_v2" / "normalization_manifest.json"
)
SPLIT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "langmani_v2"
    / "phase_2b6_v2"
    / "accepted_primary_split_manifest.json"
)


def _version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not_installed"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain one object")
    return value


def _write_new_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite report {path}")
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _runtime_identity() -> tuple[dict[str, object], str]:
    semantic = {
        "python": platform.python_version(),
        "python_executable": Path(__import__("sys").executable).resolve().as_posix(),
        "environment_prefix": Path(__import__("sys").prefix).resolve().as_posix(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "lerobot": _version("lerobot"),
        "transformers": _version("transformers"),
        "diffusers": _version("diffusers"),
        "numpy": _version("numpy"),
        "torchvision": _version("torchvision"),
        "av": _version("av"),
        "gpu": (
            {
                "name": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            }
            if torch.cuda.is_available()
            else None
        ),
    }
    return semantic, f"sha256:{sha256_hex(semantic)}"


def _sampler_fingerprint(
    kind: ModelKind,
    *,
    batch_size: int,
    seed: int,
    training_steps: int,
) -> str:
    if kind is ModelKind.SHARED:
        audit = UniformTaskBatchSampler(
            {task_id: TRAIN_FRAMES[task_id] for task_id in TASK_IDS},
            batch_size=batch_size,
            seed=seed,
        ).audit(training_steps)
        return str(audit["fingerprint"])
    semantic = {
        "sampler": "DeterministicResumeBatchSampler",
        "model_kind": kind.value,
        "task_id": kind.task_id,
        "dataset_size": TRAIN_FRAMES[str(kind.task_id)],
        "batch_size": batch_size,
        "seed": seed,
        "training_steps": training_steps,
    }
    return f"sha256:{sha256_hex(semantic)}"


def _identity(
    *,
    kind: ModelKind,
    role: str,
    optimization: Any,
    statistics: ProcessorStatistics,
    bounded_action_head_v1: bool,
) -> Phase2CARunIdentity:
    branch = _git("branch", "--show-current")
    status = _git("status", "--porcelain")
    expected_branch = (
        "codex/langmani-v2-phase2c-a1-bounded-act"
        if bounded_action_head_v1
        else "codex/langmani-v2-phase2c-a-act-baselines"
    )
    if branch != expected_branch or status:
        raise RuntimeError(f"ACT evidence requires the clean {expected_branch} branch")
    git_commit = _git("rev-parse", "HEAD")
    split_manifest = _read_object(SPLIT_MANIFEST_PATH)
    split_fingerprint = split_manifest.get("fingerprint")
    if not isinstance(split_fingerprint, str):
        raise RuntimeError("accepted split manifest lacks its fingerprint")
    runtime, runtime_fingerprint = _runtime_identity()
    identity = Phase2CARunIdentity(
        model_kind=kind,
        run_role=role,
        model_config=(
            model_identity_with_bounded_head(
                Phase2CAModelConfig.for_model(kind).to_dict()
            )
            if bounded_action_head_v1
            else Phase2CAModelConfig.for_model(kind).to_dict()
        ),
        optimization_config=optimization.to_dict(),
        sampler_fingerprint=_sampler_fingerprint(
            kind,
            batch_size=optimization.batch_size,
            seed=optimization.seed,
            training_steps=optimization.training_steps,
        ),
        train_statistics_fingerprint=statistics.fingerprint,
        m3b_split_manifest_digest=split_fingerprint,
        git_commit=git_commit,
        runtime_fingerprint=runtime_fingerprint,
    )
    del runtime
    return identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("config", "smoke", "micro", "train"):
        child = commands.add_parser(command)
        child.add_argument("--dataset-root", type=Path, required=command != "config")
        child.add_argument(
            "--model-kind", choices=[item.value for item in ModelKind], required=True
        )
        child.add_argument("--batch-size", type=int, required=True)
        child.add_argument("--dataloader-workers", type=int, default=8)
        child.add_argument("--seed", type=int, default=0)
        child.add_argument("--bounded-action-head-v1", action="store_true")
        child.add_argument("--output-root", type=Path, required=command != "config")
        child.add_argument("--report", type=Path, required=command != "config")
        if command == "micro":
            child.add_argument("--steps", type=int, default=200)
    diagnose = commands.add_parser("diagnose")
    diagnose.add_argument("--dataset-root", type=Path, required=True)
    diagnose.add_argument("--run-root", type=Path, required=True)
    diagnose.add_argument("--checkpoint-relative-path", required=True)
    diagnose.add_argument("--batch-size", type=int, default=128)
    diagnose.add_argument("--dataloader-workers", type=int, default=8)
    diagnose.add_argument("--maximum-batches", type=int)
    diagnose.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def _configuration(args: argparse.Namespace, kind: ModelKind) -> Any:
    if args.batch_size < 1 or args.dataloader_workers < 0 or args.seed < 0:
        raise ValueError("batch size/workers/seed are invalid")
    return primary_optimization_config(
        kind,
        batch_size=args.batch_size,
        dataloader_workers=args.dataloader_workers,
        seed=args.seed,
    )


def _diagnose(args: argparse.Namespace) -> dict[str, object]:
    run_root = args.run_root.resolve()
    manifest = _read_object(run_root / "run_manifest.json")
    identity_value = manifest.get("identity")
    if not isinstance(identity_value, dict):
        raise RuntimeError("training run lacks identity")
    identity = Phase2CARunIdentity.from_dict(identity_value)
    loaded = load_act_checkpoint(
        run_root=run_root,
        checkpoint_relative_path=args.checkpoint_relative_path,
        expected_identity=identity,
        for_resume=False,
        policy_class=(
            BoundedACTPolicyV1
            if identity_uses_bounded_action_head(identity.model_config)
            else None
        ),
    )
    policy = loaded.policy
    if not hasattr(policy, "forward"):
        raise RuntimeError("checkpoint did not reload an ACT policy")
    policy.to("cuda")
    view = load_dataset_view(
        args.dataset_root,
        model_kind=identity.model_kind,
        split="validation",
    )
    loader = DataLoader(
        view.dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataloader_workers,
        pin_memory=True,
        persistent_workers=args.dataloader_workers > 0,
    )
    report = offline_diagnostics(
        model_kind=identity.model_kind,
        policy=policy,
        preprocessor=loaded.preprocessor,
        postprocessor=loaded.postprocessor,
        validation_batches=loader,
        maximum_batches=args.maximum_batches,
    )
    result = {
        **report,
        "run_fingerprint": identity.run_fingerprint,
        "checkpoint_fingerprint": loaded.record.checkpoint_fingerprint,
        "checkpoint_relative_path": loaded.record.relative_path,
        "checkpoint_step": loaded.record.global_step,
        "package_fingerprint": PACKAGE_FINGERPRINT,
    }
    _write_new_json(args.report.resolve(), result)
    return result


def main() -> int:
    args = parse_args()
    if args.command == "diagnose":
        result = _diagnose(args)
        print(json.dumps(result, sort_keys=True))
        return 0
    kind = ModelKind(args.model_kind)
    optimization = _configuration(args, kind)
    if args.command == "config":
        print(
            json.dumps(
                {
                    "model_kind": kind.value,
                    "model_config": (
                        model_identity_with_bounded_head(
                            Phase2CAModelConfig.for_model(kind).to_dict()
                        )
                        if args.bounded_action_head_v1
                        else Phase2CAModelConfig.for_model(kind).to_dict()
                    ),
                    "optimization_config": optimization.to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    statistics = load_processor_statistics(
        STATISTICS_PATH,
        NORMALIZATION_PATH,
        model_kind=kind,
    )
    role = {
        "smoke": "gpu_smoke",
        "micro": "micro_overfit",
        "train": "primary_training",
    }[args.command]
    identity = _identity(
        kind=kind,
        role=role,
        optimization=optimization,
        statistics=statistics,
        bounded_action_head_v1=args.bounded_action_head_v1,
    )
    started = __import__("time").perf_counter()
    if args.command == "smoke":
        result = run_one_batch_gpu_smoke(
            primary_root=args.dataset_root,
            model_kind=kind,
            optimization=optimization,
            statistics=statistics,
            run_identity=identity,
            output_root=args.output_root,
            bounded_action_head_v1=args.bounded_action_head_v1,
        )
    elif args.command == "micro":
        result = run_micro_overfit(
            primary_root=args.dataset_root,
            model_kind=kind,
            optimization=optimization,
            statistics=statistics,
            run_identity=identity,
            output_root=args.output_root,
            steps=args.steps,
            bounded_action_head_v1=args.bounded_action_head_v1,
        )
    else:
        outcome = train_primary_act(
            primary_root=args.dataset_root,
            model_kind=kind,
            optimization=optimization,
            statistics=statistics,
            run_identity=identity,
            output_root=args.output_root,
            bounded_action_head_v1=args.bounded_action_head_v1,
        )
        result = outcome.to_dict()
    elapsed = __import__("time").perf_counter() - started
    report = {
        "schema_version": "langmani-v2-phase2c-a-act-command-v0",
        "command": args.command,
        "model_kind": kind.value,
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "bounded_action_head_v1": args.bounded_action_head_v1,
        "identity": identity.to_dict(),
        "runtime": _runtime_identity()[0],
        "wall_time_s": elapsed,
        "result": result,
        "passed": True,
    }
    _write_new_json(args.report.resolve(), report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
