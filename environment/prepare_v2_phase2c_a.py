"""Run the fail-closed Phase 2C-A repository, runtime, and consumer preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from langmani.datasets.identity import sha256_hex
from langmani.v2.phase2c_a import (
    PACKAGE_FINGERPRINT,
    SOURCE_COMMIT,
    TARGET_BRANCH,
    TASK_IDS,
    TRAIN_FRAMES,
    ModelKind,
    UniformTaskBatchSampler,
    build_evaluation_schedule,
    build_training_view_manifest,
    primary_optimization_config,
    task_condition_intervention,
    validate_package_document,
)
from langmani.v2.phase2c_a_act import audit_consumer_view

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2b6_v2"
IDENTITY_RESOLUTION = (
    PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2c_a" / "identity_resolution.json"
)


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain one object")
    return value


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite preflight artifact {path}")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _run(
    *args: str, cwd: Path | None = None, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )


def _git(*args: str) -> str:
    return _run("git", *args, cwd=PROJECT_ROOT).stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not_installed"


def _nvidia_smi() -> dict[str, object]:
    result = _run(
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    devices = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 7:
            raise RuntimeError("nvidia-smi returned an unexpected device row")
        devices.append(
            {
                "index": int(values[0]),
                "name": values[1],
                "uuid": values[2],
                "driver_version": values[3],
                "memory_total_mib": int(values[4]),
                "memory_used_mib": int(values[5]),
                "utilization_percent": int(values[6]),
            }
        )
    return {"available": bool(devices), "devices": devices}


def _process_audit() -> dict[str, object]:
    result = _run("ps", "-eo", "pid=,args=", check=False)
    prohibited = (
        "collect",
        "replay",
        "convert",
        "archive",
        "restore",
        "train_v2_phase2c_a",
        "train_act",
        "ffmpeg",
    )
    matches = []
    own_pid = os.getpid()
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        first, _, command = stripped.partition(" ")
        try:
            pid = int(first)
        except ValueError:
            continue
        if pid == own_pid or "prepare_v2_phase2c_a.py" in command:
            continue
        if any(token in command.lower() for token in prohibited):
            matches.append({"pid": pid, "command": command})
    return {"prohibited_processes": matches, "passed": not matches}


def _installed_source_audit() -> dict[str, object]:
    import lerobot.policies.act.configuration_act as configuration
    import lerobot.policies.act.modeling_act as modeling
    import lerobot.policies.act.processor_act as processor

    files = {
        "configuration_act": Path(configuration.__file__).resolve(),
        "modeling_act": Path(modeling.__file__).resolve(),
        "processor_act": Path(processor.__file__).resolve(),
    }
    records = {
        name: {
            "path": path.as_posix(),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in files.items()
    }
    source = files["modeling_act"].read_text(encoding="utf-8")
    required_fragments = (
        'valid_mask = ~batch["action_is_pad"].unsqueeze(-1)',
        "abs_err = F.l1_loss",
        "(abs_err * valid_mask).sum()",
        "loss = l1_loss + mean_kld * self.config.kl_weight",
    )
    missing = [fragment for fragment in required_fragments if fragment not in source]
    if missing:
        raise RuntimeError(f"installed ACT padding-loss source changed: missing={missing}")
    semantic = {
        "lerobot_version": _version("lerobot"),
        "installed_source_files": records,
        "maintained_core_modified": False,
        "policy_class": "lerobot.policies.act.ACTPolicy",
        "configuration_class": "lerobot.policies.act.ACTConfig",
        "image_encoder": "torchvision_resnet18_without_pretrained_weights",
        "transformer": {
            "dim_model": 512,
            "heads": 8,
            "feedforward_dimension": 3200,
            "encoder_layers": 4,
            "decoder_layers": 1,
            "pre_norm": False,
            "dropout": 0.1,
        },
        "action_chunk_size": 16,
        "runtime_execution_horizons": [1, 4, 8],
        "temporal_ensemble": False,
        "vae": {
            "enabled": True,
            "latent_dimension": 32,
            "encoder_layers": 4,
            "kl_weight": 10.0,
        },
        "action_loss": "padding-masked component-mean L1",
        "padding_mask_feature": "action_is_pad",
        "padding_loss_source_fragments_verified": list(required_fragments),
        "normalization": "MEAN_STD for VISUAL/STATE/ACTION; IDENTITY for shared ENV token",
        "checkpoint_serialization": "ACTPolicy.save_pretrained",
        "processor_serialization": (
            "PolicyProcessorPipeline.save_pretrained using policy_preprocessor.json and "
            "policy_postprocessor.json"
        ),
        "resume_state": "optimizer plus Python/NumPy/Torch CPU/CUDA RNG state",
        "deterministic_evaluation": (
            "policy.eval, ACT deterministic zero latent, fixed reset identity, no temporal ensemble"
        ),
        "dataset_adapter": (
            "isolated accepted-root view and three-way public FeatureType.ENV one-hot token"
        ),
    }
    return {
        "schema_version": "langmani-v2-phase2c-a-act-implementation-audit-v0",
        **semantic,
        "fingerprint": f"sha256:{sha256_hex(semantic)}",
        "passed": True,
    }


def _repository_environment_audit(dataset_root: Path, output_root: Path) -> dict[str, object]:
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    if branch != TARGET_BRANCH:
        raise RuntimeError(f"expected {TARGET_BRANCH}, got {branch}")
    if status:
        raise RuntimeError("Phase 2C-A target worktree must be clean for preflight")
    upstream = _git("rev-parse", "@{upstream}")
    remote = _git("remote", "get-url", "origin")
    if head != upstream:
        raise RuntimeError("local and upstream Phase 2C-A commits differ")
    disk = shutil.disk_usage(output_root.parent)
    package_lock = _run(sys.executable, "-m", "pip", "freeze").stdout
    lock_path = output_root / "package_lock.txt"
    lock_path.write_text(package_lock, encoding="utf-8", newline="\n")
    identity = _read_object(IDENTITY_RESOLUTION)
    processes = _process_audit()
    if processes["passed"] is not True:
        raise RuntimeError("a prohibited writer/replay/training process is active")
    runtime = {
        "python": platform.python_version(),
        "python_executable": Path(sys.executable).resolve().as_posix(),
        "environment_prefix": Path(sys.prefix).resolve().as_posix(),
        "pytorch": _version("torch"),
        "torch_cuda": __import__("torch").version.cuda,
        "lerobot": _version("lerobot"),
        "transformers": _version("transformers"),
        "diffusers": _version("diffusers"),
        "numpy": _version("numpy"),
        "torchvision": _version("torchvision"),
        "av": _version("av"),
        "pillow": _version("pillow"),
        "opencv_python": _version("opencv-python"),
        "gymnasium": _version("gymnasium"),
        "mani_skill": _version("mani-skill"),
        "sapien": _version("sapien"),
    }
    semantic: dict[str, Any] = {
        "source_commit": SOURCE_COMMIT,
        "branch": branch,
        "head": head,
        "upstream": upstream,
        "worktree_clean": not status,
        "remote": remote,
        "dataset_root": dataset_root.as_posix(),
        "dataset_root_exists": dataset_root.is_dir(),
        "canonical_package_fingerprint": PACKAGE_FINGERPRINT,
        "identity_resolution_fingerprint": identity.get("fingerprint"),
        "archive": identity.get("archive_identity"),
        "restore": identity.get("restore_identity"),
        "runtime": runtime,
        "gpu": _nvidia_smi(),
        "disk": {
            "path": output_root.parent.resolve().as_posix(),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "process_audit": processes,
        "package_lock": {
            "path": lock_path.name,
            "sha256": _sha256_file(lock_path),
            "line_count": len(package_lock.splitlines()),
        },
        "training_environment_isolated_from_replay_environment": (
            Path(sys.prefix).resolve().as_posix() != "/root/autodl-tmp/conda-envs/langmani"
        ),
        "policy_simulator_transport": (
            "content-bound checkpoint and processors loaded into an in-process "
            "PolicyAdapter; observation/action tensors cross the generic interface only"
        ),
        "smolvla_loaded": False,
        "vla_jepa_loaded": False,
    }
    if (
        semantic["dataset_root_exists"] is not True
        or semantic["training_environment_isolated_from_replay_environment"] is not True
        or semantic["gpu"]["available"] is not True
        or disk.free < 20 * 1024**3
    ):
        raise RuntimeError("repository/runtime resource preflight failed")
    return {
        "schema_version": "langmani-v2-phase2c-a-repository-environment-audit-v0",
        **semantic,
        "fingerprint": f"sha256:{sha256_hex(semantic)}",
        "passed": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--dataloader-workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise RuntimeError(f"preflight output already exists: {output_root}")
    output_root.mkdir(parents=True)
    package = _read_object(SOURCE_ARTIFACT_ROOT / "accepted_multiskill_dataset_package.json")
    split = _read_object(SOURCE_ARTIFACT_ROOT / "accepted_primary_split_manifest.json")
    sources = _read_object(SOURCE_ARTIFACT_ROOT / "source_inventory.json")
    package_result = validate_package_document(package)
    training_view = build_training_view_manifest(split)
    schedule = build_evaluation_schedule(split, sources)
    intervention_tasks = tuple(task_id for task_id in TASK_IDS for _ in range(5))
    intervention = task_condition_intervention(intervention_tasks)
    consumer = audit_consumer_view(dataset_root)
    act_audit = _installed_source_audit()
    repository = _repository_environment_audit(dataset_root, output_root)
    outputs = {
        "accepted_dataset_verification.json": package_result,
        "training_view_manifest.json": training_view,
        "evaluation_schedule.json": schedule,
        "task_condition_intervention_schedule.json": intervention,
        "consumer_view_audit.json": consumer,
        "act_implementation_audit.json": act_audit,
        "repository_environment_audit.json": repository,
    }
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        configurations = {
            kind.value: primary_optimization_config(
                kind,
                batch_size=args.batch_size,
                dataloader_workers=args.dataloader_workers,
            ).to_dict()
            for kind in ModelKind
        }
        shared_sampler = UniformTaskBatchSampler(
            {task_id: TRAIN_FRAMES[task_id] for task_id in TASK_IDS},
            batch_size=args.batch_size,
            seed=0,
        ).audit(configurations[ModelKind.SHARED.value]["training_steps"])
        sampler_semantic = {
            "batch_size": args.batch_size,
            "task_specific_sampler": (
                "independent deterministic epoch permutations addressable by optimizer step"
            ),
            "shared_sampler": shared_sampler,
            "strict_uniform_task_frequency": True,
            "shared_budget_resolution": (
                "strict 1/3 task sampling uses the arithmetic mean of the three task-specific "
                "20-pass sample budgets for each task; actual effective passes are recorded"
            ),
        }
        outputs["four_training_configurations.json"] = {
            "schema_version": "langmani-v2-phase2c-a-four-training-configurations-v0",
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "configurations": configurations,
            "fingerprint": f"sha256:{sha256_hex(configurations)}",
            "passed": True,
        }
        outputs["sampler_manifest.json"] = {
            "schema_version": "langmani-v2-phase2c-a-sampler-manifest-v0",
            **sampler_semantic,
            "fingerprint": f"sha256:{sha256_hex(sampler_semantic)}",
            "passed": True,
        }
    for name, value in outputs.items():
        _write_json(output_root / name, value)
    manifest_records = [
        {
            "path": path.name,
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(output_root.iterdir())
        if path.is_file()
    ]
    _write_json(
        output_root / "preflight_manifest.json",
        {
            "schema_version": "langmani-v2-phase2c-a-preflight-manifest-v0",
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "artifacts": manifest_records,
            "passed": True,
        },
    )
    print(json.dumps({"output_root": output_root.as_posix(), "passed": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
