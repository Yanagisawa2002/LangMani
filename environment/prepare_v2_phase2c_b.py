"""Create the compact Phase 2C-B input and static-contract artifacts."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

from langmani.v2.phase2c import sha256_file
from langmani.v2.phase2c_a import (
    PACKAGE_FINGERPRINT,
    UniformTaskBatchSampler,
    validate_package_document,
)
from langmani.v2.phase2c_b import (
    SOURCE_COMMIT,
    TARGET_BRANCH,
    BoundedActionLatentV1,
    ModelKind,
    SmolVLATrainingConfig,
    build_model_view_manifest,
    build_padding_audit,
    build_static_action_audit,
    canonical_fingerprint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2b6_v2"
ACT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2c_a1"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "langmani_v2" / "phase2c_b_smolvla.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--allow-noncuda-static", action="store_true")
    return parser.parse_args()


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain one JSON object")
    return value


def _write_new_or_equal(path: Path, value: Mapping[str, object]) -> None:
    encoded = json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"existing immutable artifact differs: {path}")
        return
    path.write_text(encoded, encoding="utf-8", newline="\n")


def _run(*command: str) -> str:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _act_closure() -> dict[str, object]:
    closure = _read_object(ACT_ARTIFACT_ROOT / "act_closure_state.json")
    verification = _read_object(ACT_ARTIFACT_ROOT / "final_verification.json")
    final = _read_object(ACT_ARTIFACT_ROOT / "final_evaluation.json")
    groups = final.get("groups")
    if not isinstance(groups, list) or not all(isinstance(group, Mapping) for group in groups):
        raise RuntimeError("Phase 2C-A.1 final evaluation registry is malformed")
    summaries = [group.get("summary") for group in groups]
    if not all(isinstance(summary, Mapping) for summary in summaries):
        raise RuntimeError("Phase 2C-A.1 final summaries are malformed")
    episode_count = sum(int(cast_summary["episode_count"]) for cast_summary in summaries)
    success_count = sum(int(cast_summary["success_count"]) for cast_summary in summaries)
    invalid_count = sum(int(cast_summary["invalid_action_count"]) for cast_summary in summaries)
    simulator_count = sum(int(cast_summary["simulator_error_count"]) for cast_summary in summaries)
    action_count = sum(
        int(cast_summary["episode_count"]) * int(round(float(cast_summary["mean_episode_length"])))
        for cast_summary in summaries
    )
    if (
        closure
        != {
            "act_baselines_validated": True,
            "act_phase_closed": True,
            "act_policy_quality_weak": True,
            "further_act_architecture_authorized": False,
            "result": "RESULT_B",
            "schema_version": "langmani-v2-phase2c-a1-act-closure-v0",
            "shared_act_failed": False,
        }
        or verification.get("passed") is not True
        or verification.get("result") != "RESULT_B"
        or verification.get("act_phase_closed") is not True
        or verification.get("package_fingerprint") != PACKAGE_FINGERPRINT
        or verification.get("smolvla_training_authorized") is not False
        or verification.get("vla_jepa_training_authorized") is not False
        or final.get("episode_count") != 360
        or episode_count != 360
        or success_count != 0
        or invalid_count != 0
        or simulator_count != 0
        or action_count != 18_000
    ):
        raise RuntimeError("Phase 2C-A.1 immutable closure facts changed")
    return {
        "result": "RESULT_B",
        "pipeline_valid": True,
        "act_phase_closed": True,
        "episode_count": episode_count,
        "physical_action_count": action_count,
        "success_count": success_count,
        "invalid_action_count": invalid_count,
        "simulator_error_count": simulator_count,
        "final_verification_fingerprint": verification.get("fingerprint"),
        "act_retraining_authorized": False,
    }


def _running_processes() -> list[str]:
    if os.name == "nt":
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process | % CommandLine",
        ]
    else:
        command = ["ps", "-eo", "args="]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    needles = (
        "train_v2_phase2c",
        "evaluate_v2_phase2c",
        "collect_",
        "replay_",
        "vla_jepa",
        "lerobot-train",
    )
    return [
        line.strip()
        for line in completed.stdout.splitlines()
        if any(needle in line.lower() for needle in needles)
        and "prepare_v2_phase2c_b.py" not in line
    ]


def _repository_audit() -> dict[str, object]:
    head = _run("git", "rev-parse", "HEAD")
    branch = _run("git", "branch", "--show-current")
    status = _run("git", "status", "--porcelain")
    origin = _run("git", "remote", "get-url", "origin")
    upstream = _run("git", "rev-parse", "@{upstream}")
    source_is_ancestor = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, head],
            cwd=PROJECT_ROOT,
            check=False,
        ).returncode
        == 0
    )
    if (
        branch != TARGET_BRANCH
        or upstream != head
        or not source_is_ancestor
        or status
        or origin != "https://github.com/Yanagisawa2002/LangMani.git"
    ):
        raise RuntimeError("Phase 2C-B repository isolation audit failed")
    return {
        "branch": branch,
        "source_commit": SOURCE_COMMIT,
        "source_is_ancestor": source_is_ancestor,
        "head": head,
        "upstream": upstream,
        "origin": origin,
        "clean": True,
    }


def main() -> int:
    args = parse_args()
    primary_root = args.primary_root.resolve()
    output_root = args.output_root.resolve()
    config_path = args.config.resolve()
    package = _read_object(primary_root / "metadata" / "accepted_multiskill_dataset_package.json")
    source_package = _read_object(SOURCE_ARTIFACT_ROOT / "accepted_multiskill_dataset_package.json")
    verified = validate_package_document(package)
    if (
        source_package.get("fingerprint") != PACKAGE_FINGERPRINT
        or package != source_package
        or verified.get("passed") is not True
    ):
        raise RuntimeError("live accepted package differs from the canonical committed package")
    config = _read_object(config_path)
    if config.get("dataset", {}).get("fingerprint") != PACKAGE_FINGERPRINT:
        raise RuntimeError("Phase 2C-B configuration uses a noncanonical package")
    if not torch.cuda.is_available() and not args.allow_noncuda_static:
        raise RuntimeError("Phase 2C-B target preparation requires an available CUDA device")
    processes = _running_processes()
    if processes:
        raise RuntimeError("conflicting data/model process is already running: " + repr(processes))
    repository = _repository_audit()
    disk = shutil.disk_usage(output_root.parent if output_root.parent.exists() else primary_root)
    cuda = (
        {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        }
        if torch.cuda.is_available()
        else {"available": False}
    )
    closure = _act_closure()
    input_semantic: dict[str, Any] = {
        "schema_version": "langmani-v2-phase2c-b-input-verification-v0",
        "repository": repository,
        "package_verification": verified,
        "live_package_path": (
            primary_root / "metadata" / "accepted_multiskill_dataset_package.json"
        ).as_posix(),
        "live_package_matches_committed_authority": True,
        "canonical_package_fingerprint": PACKAGE_FINGERPRINT,
        "act_closure": closure,
        "conflicting_processes": [],
        "runtime": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "lerobot": _package_version("lerobot"),
            "transformers": _package_version("transformers"),
            "diffusers": _package_version("diffusers"),
            "tokenizers": _package_version("tokenizers"),
        },
        "gpu": cuda,
        "disk": {
            "path": output_root.parent.as_posix(),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "config_path": config_path.as_posix(),
        "config_sha256": f"sha256:{sha256_file(config_path)}",
        "dataset_writer_started": False,
        "replay_started": False,
        "act_started": False,
        "smolvla_started": False,
        "vla_jepa_started": False,
        "passed": True,
    }
    input_report = {
        **input_semantic,
        "fingerprint": canonical_fingerprint(input_semantic),
    }
    _write_new_or_equal(output_root / "input_verification.json", input_report)

    action_audit = build_static_action_audit()
    padding_audit = build_padding_audit()
    transform = BoundedActionLatentV1()
    transform_manifest = {
        **transform.semantic_dict(),
        "fingerprint": transform.fingerprint,
        "passed": True,
    }
    training = SmolVLATrainingConfig()
    training_manifest = {
        "schema_version": "langmani-v2-phase2c-b-training-configs-v0",
        "primary": training.to_dict(),
        "primary_fingerprint": training.fingerprint,
        "model_order": [kind.value for kind in ModelKind],
        "one_seed_per_model": True,
        "second_shared_seed_condition": "nonzero_primary_shared_success_on_all_three_tasks",
        "large_hyperparameter_sweep": False,
        "vla_jepa_authorized": False,
    }
    _write_new_or_equal(output_root / "action_100k_audit.json", action_audit)
    _write_new_or_equal(output_root / "padding_audit.json", padding_audit)
    _write_new_or_equal(output_root / "bounded_action_manifest.json", transform_manifest)
    _write_new_or_equal(output_root / "training_configs.json", training_manifest)
    views_root = output_root / "model_views"
    for kind in ModelKind:
        _write_new_or_equal(
            views_root / f"{kind.slug}.json",
            build_model_view_manifest(kind),
        )
    sampler = UniformTaskBatchSampler(
        {"PickCube-v1": 54_608, "StackCube-v1": 74_891, "PushCube-v1": 48_219},
        batch_size=training.batch_size,
        seed=training.seed,
    ).audit(training.total_steps * training.gradient_accumulation)
    _write_new_or_equal(output_root / "sampler_manifest.json", sampler)
    completion_semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-b-static-preparation-complete-v0",
        "input_verification_fingerprint": input_report["fingerprint"],
        "action_audit_fingerprint": action_audit["fingerprint"],
        "padding_audit_fingerprint": padding_audit["fingerprint"],
        "bounded_action_fingerprint": transform.fingerprint,
        "training_config_fingerprint": training.fingerprint,
        "sampler_fingerprint": sampler["fingerprint"],
        "model_view_fingerprints": {
            kind.value: build_model_view_manifest(kind)["fingerprint"] for kind in ModelKind
        },
        "passed": True,
    }
    completion = {
        **completion_semantic,
        "fingerprint": canonical_fingerprint(completion_semantic),
    }
    _write_new_or_equal(output_root / "static_preparation_complete.json", completion)
    print(json.dumps(completion, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
