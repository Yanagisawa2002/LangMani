"""Train the sole authorized Phase 2C-C relative-action Pick SmolVLA."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from accelerate import Accelerator
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, Subset

from langmani.policies.act_training import DeterministicResumeBatchSampler
from langmani.v2.phase2c_a import PACKAGE_FINGERPRINT
from langmani.v2.phase2c_b import (
    OFFICIAL_BASE_MODEL,
    OFFICIAL_BASE_REVISION,
    ModelKind,
    SmolVLATrainingConfig,
)
from langmani.v2.phase2c_b_smolvla import (
    ACTION_FEATURE_KEY,
    ACTION_PAD_KEY,
    LoadedSmolVLAView,
    build_official_config,
    load_dataset_view,
    load_smolvla_statistics,
    make_processors,
    processor_manifest,
    project_policy_batch,
)
from langmani.v2.phase2c_c import (
    TARGET_BRANCH,
    Phase2CCContractError,
    StateRelativeBoundedActionV0,
    canonical_fingerprint,
)
from langmani.v2.phase2c_c_adapter import CHECKPOINT_MANIFEST
from langmani.v2.phase2c_c_smolvla import (
    RelativeSmolVLAPolicyV0,
    load_relative_checkpoint,
    load_relative_pretrained_policy,
    project_relative_policy_batch,
    validate_unchanged_training_config,
)
from langmani.v2.smolvla_adapter import sha256_directory

MICRO_STEPS = 500
MICRO_FRAMES = 64


def _write_new_or_equal(path: Path, value: Mapping[str, object]) -> None:
    encoded = json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise Phase2CCContractError(f"existing immutable artifact differs: {path}")
        return
    path.write_text(encoded, encoding="utf-8", newline="\n")


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(value), sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _micro_view(view: LoadedSmolVLAView, kind: ModelKind) -> LoadedSmolVLAView:
    if kind is not ModelKind.PICK:
        raise Phase2CCContractError("Phase 2C-C micro data must remain Pick-only")
    length = view.task_lengths["PickCube-v1"]
    selected = min(MICRO_FRAMES, length)
    return LoadedSmolVLAView(
        dataset=cast(Dataset[Mapping[str, object]], Subset(view.dataset, range(selected))),
        task_lengths={"PickCube-v1": selected},
        task_roots=view.task_roots,
        split=view.split,
    )


def _dataloader(
    view: LoadedSmolVLAView,
    *,
    kind: ModelKind,
    training: SmolVLATrainingConfig,
    start_optimizer_step: int,
    num_workers: int,
) -> DataLoader[Mapping[str, object]]:
    if kind is not ModelKind.PICK:
        raise Phase2CCContractError("Phase 2C-C training must remain Pick-only")
    start_batch = start_optimizer_step * training.gradient_accumulation
    sampler = DeterministicResumeBatchSampler(
        dataset_size=len(cast(Any, view.dataset)),
        batch_size=training.batch_size,
        seed=training.seed,
        start_step=start_batch,
    )
    generator = torch.Generator().manual_seed(training.seed)
    return DataLoader(
        view.dataset,
        batch_sampler=cast(Any, sampler),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def _lr_factor(training: SmolVLATrainingConfig, step: int) -> float:
    if step < training.warmup_steps:
        return (step + 1) / training.warmup_steps
    progress = min(
        1.0,
        (step - training.warmup_steps) / max(1, training.decay_steps - training.warmup_steps),
    )
    minimum = training.decay_learning_rate / training.learning_rate
    return minimum + 0.5 * (1.0 - minimum) * (1.0 + math.cos(math.pi * progress))


def _checkpoint_path(run_root: Path, step: int) -> Path:
    return run_root / "checkpoints" / f"step-{step:08d}"


def _rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng(value: Mapping[str, object]) -> None:
    random.setstate(cast(tuple[Any, ...], value["python"]))
    np.random.set_state(cast(tuple[Any, ...], value["numpy"]))
    torch.set_rng_state(cast(torch.Tensor, value["torch"]))
    torch.cuda.set_rng_state_all(cast(list[torch.Tensor], value["cuda"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--base-snapshot", type=Path, required=True)
    parser.add_argument("--vlm-snapshot", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "micro", "full"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--training-git-commit", required=True)
    parser.add_argument("--static-preparation", type=Path, required=True)
    parser.add_argument("--base-audit-completion", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--num-workers", type=int)
    return parser.parse_args()


def _read_fingerprinted(
    path: Path,
    *,
    schema_version: str,
) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2CCContractError(f"cannot read prerequisite {path}: {error}") from error
    if not isinstance(value, dict):
        raise Phase2CCContractError(f"prerequisite {path} must contain one object")
    semantic = dict(value)
    fingerprint = semantic.pop("fingerprint", None)
    if (
        value.get("schema_version") != schema_version
        or value.get("passed") is not True
        or fingerprint != canonical_fingerprint(semantic)
    ):
        raise Phase2CCContractError(f"prerequisite identity is invalid: {path}")
    return value


def _repository_identity(expected_commit: str) -> dict[str, object]:
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise Phase2CCContractError("--training-git-commit must be one full Git SHA")

    def run(*command: str) -> str:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()

    head = run("git", "rev-parse", "HEAD")
    branch = run("git", "branch", "--show-current")
    status = run("git", "status", "--porcelain")
    upstream = run("git", "rev-parse", "@{upstream}")
    if head != expected_commit or branch != TARGET_BRANCH or status or upstream != head:
        raise Phase2CCContractError("training repository identity/cleanliness check failed")
    return {"git_commit": head, "branch": branch, "upstream": upstream, "clean": True}


def _physical_fixed_batch(view: LoadedSmolVLAView) -> dict[str, object]:
    rows = [view.dataset[0], view.dataset[1]]
    raw: dict[str, object] = {}
    for key in (
        "observation.images.base_camera",
        "observation.state",
        ACTION_FEATURE_KEY,
        ACTION_PAD_KEY,
        "task",
    ):
        values = [row[key] for row in rows]
        raw[key] = (
            torch.stack(cast(list[torch.Tensor], values))
            if isinstance(values[0], torch.Tensor)
            else values
        )
    return project_policy_batch(raw, shared=False)


def _evaluate_fixed_loss(
    policy: RelativeSmolVLAPolicyV0,
    preprocessor: object,
    physical_batch: Mapping[str, object],
) -> float:
    policy.eval()
    relative = project_relative_policy_batch(physical_batch)
    processed = cast(Any, preprocessor)(relative)
    if not isinstance(processed, Mapping):
        raise Phase2CCContractError("fixed-batch preprocessor returned a non-mapping")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = policy(dict(processed))
    value = float(loss)
    if not math.isfinite(value):
        raise Phase2CCContractError("fixed-batch loss is non-finite")
    policy.train()
    return value


def _predict_diagnostics(
    policy: RelativeSmolVLAPolicyV0,
    preprocessor: object,
    postprocessor: object,
    physical_batch: Mapping[str, object],
) -> dict[str, object]:
    policy.eval()
    transform = StateRelativeBoundedActionV0()
    relative = project_relative_policy_batch(physical_batch)
    processed = cast(Any, preprocessor)(relative)
    if not isinstance(processed, Mapping):
        raise Phase2CCContractError("diagnostic preprocessor returned a non-mapping")
    processed_state = processed.get("observation.state")
    current_state = physical_batch.get("observation.state")
    target_physical = physical_batch.get(ACTION_FEATURE_KEY)
    target_latent = relative.get(ACTION_FEATURE_KEY)
    padding = physical_batch.get(ACTION_PAD_KEY)
    if not all(
        isinstance(value, torch.Tensor)
        for value in (
            processed_state,
            current_state,
            target_physical,
            target_latent,
            padding,
        )
    ):
        raise Phase2CCContractError("diagnostic batch is malformed")
    assert isinstance(processed_state, torch.Tensor)
    assert isinstance(current_state, torch.Tensor)
    assert isinstance(target_physical, torch.Tensor)
    assert isinstance(target_latent, torch.Tensor)
    assert isinstance(padding, torch.Tensor)
    generator = torch.Generator(device=processed_state.device).manual_seed(27_182)
    base_noise = torch.randn(
        (1, int(policy.config.chunk_size), int(policy.config.max_action_dim)),
        generator=generator,
        device=processed_state.device,
        dtype=processed_state.dtype,
    )
    noise = base_noise.expand(processed_state.shape[0], -1, -1).clone()
    with torch.inference_mode():
        generated = cast(Any, postprocessor)(
            policy.predict_action_chunk(dict(processed), noise=noise)
        )
    if not isinstance(generated, torch.Tensor) or generated.shape != target_latent.shape:
        raise Phase2CCContractError("diagnostic residual shape changed")
    generated_cpu = generated.to(dtype=torch.float32, device="cpu")
    current_cpu = current_state.to(dtype=torch.float32, device="cpu")
    physical_predicted = transform.decode(generated_cpu, current_cpu)
    target_physical_cpu = target_physical.to(dtype=torch.float32, device="cpu")
    target_latent_cpu = target_latent.to(dtype=torch.float32, device="cpu")
    padding_cpu = padding.to(device="cpu")
    valid = (~padding_cpu).unsqueeze(-1).expand_as(target_physical_cpu)
    reference = transform.current_action_reference(current_cpu)[:, None, :]
    predicted_delta = physical_predicted - reference
    target_delta = target_physical_cpu - reference
    physical_error = torch.abs(physical_predicted - target_physical_cpu)
    residual_error = torch.abs(predicted_delta - target_delta)
    latent_error = torch.abs(generated_cpu - target_latent_cpu)
    observation_difference = float(
        torch.max(torch.abs(physical_predicted[0] - physical_predicted[1]))
    )
    policy.train()
    return {
        "physical_action_mae": float(physical_error[valid].mean()),
        "physical_first_action_mae": float(physical_error[:, 0].mean()),
        "physical_arm_mae": float(physical_error[..., :7][valid[..., :7]].mean()),
        "physical_gripper_mae": float(physical_error[..., 7:][valid[..., 7:]].mean()),
        "residual_action_mae": float(residual_error[valid].mean()),
        "residual_latent_mae": float(latent_error[valid].mean()),
        "different_observation_max_abs_difference": observation_difference,
        "outputs_differ_across_observations": observation_difference > 1e-8,
        "all_actions_bounded": True,
    }


def _save_checkpoint(
    *,
    accelerator: Accelerator,
    policy: RelativeSmolVLAPolicyV0,
    preprocessor: object,
    postprocessor: object,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    run_root: Path,
    stage: str,
    step: int,
    training: SmolVLATrainingConfig,
    processor_fingerprint: str,
    training_git_commit: str,
    static_preparation_fingerprint: str,
    base_audit_fingerprint: str,
    real_view_fingerprint: str,
    run_manifest_fingerprint: str,
) -> dict[str, object]:
    accelerator.wait_for_everyone()
    checkpoint = _checkpoint_path(run_root, step)
    if accelerator.is_main_process:
        if checkpoint.exists():
            raise Phase2CCContractError(f"refusing to overwrite checkpoint {checkpoint}")
        checkpoint.mkdir(parents=True)
        unwrapped = cast(RelativeSmolVLAPolicyV0, accelerator.unwrap_model(policy))
        unwrapped.save_pretrained(checkpoint, safe_serialization=True)
        cast(Any, preprocessor).save_pretrained(checkpoint)
        cast(Any, postprocessor).save_pretrained(checkpoint)
        transform = StateRelativeBoundedActionV0()
        transform.save(checkpoint)
        semantic: dict[str, object] = {
            "schema_version": "langmani-v2-phase2c-c-checkpoint-manifest-v0",
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "model_kind": "pick_smolvla_relative",
            "task_id": "PickCube-v1",
            "stage": stage,
            "optimizer_step": step,
            "training_config_fingerprint": training.fingerprint,
            "processor_contract_fingerprint": processor_fingerprint,
            "training_git_commit": training_git_commit,
            "static_preparation_fingerprint": static_preparation_fingerprint,
            "base_audit_fingerprint": base_audit_fingerprint,
            "real_view_fingerprint": real_view_fingerprint,
            "run_manifest_fingerprint": run_manifest_fingerprint,
            "action_transform_fingerprint": transform.fingerprint,
            "base_model": OFFICIAL_BASE_MODEL,
            "base_revision": OFFICIAL_BASE_REVISION,
            "task_id_model_input": False,
            "future_state_model_input": False,
            "language_input": "task",
            "post_hoc_clipping": False,
            "projection": False,
            "replacement_action": False,
        }
        _write_new_or_equal(
            checkpoint / CHECKPOINT_MANIFEST,
            {**semantic, "fingerprint": canonical_fingerprint(semantic)},
        )
        torch.save(
            {
                "schema_version": "langmani-v2-phase2c-c-training-state-v0",
                "optimizer_step": step,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": _rng_state(),
            },
            checkpoint / "training_state.pt",
        )
        digest = sha256_directory(checkpoint)
        record = {
            "optimizer_step": step,
            "relative_path": checkpoint.relative_to(run_root).as_posix(),
            "sha256": f"sha256:{digest}",
            "file_count": sum(1 for item in checkpoint.rglob("*") if item.is_file()),
            "size_bytes": sum(
                item.stat().st_size for item in checkpoint.rglob("*") if item.is_file()
            ),
        }
        _append_jsonl(run_root / "checkpoint_registry.jsonl", record)
    else:
        record = {}
    accelerator.wait_for_everyone()
    return cast(dict[str, object], record)


def _reload_audit(
    checkpoint: Path,
    *,
    physical_batch: Mapping[str, object],
    expected_checkpoint_sha256: str,
) -> dict[str, object]:
    if sha256_directory(checkpoint) != expected_checkpoint_sha256.removeprefix("sha256:"):
        raise Phase2CCContractError("checkpoint changed before reload audit")
    transform = StateRelativeBoundedActionV0.load(checkpoint)
    policy = load_relative_checkpoint(checkpoint, strict=True)
    from lerobot.policies.factory import (  # type: ignore[import-untyped]
        make_pre_post_processors,
    )

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=checkpoint,
    )
    relative = project_relative_policy_batch(physical_batch, transform=transform)
    processed = preprocessor(relative)
    if not isinstance(processed, Mapping):
        raise Phase2CCContractError("reloaded processor returned a non-mapping")
    state = processed.get("observation.state")
    current_state = physical_batch.get("observation.state")
    if not isinstance(state, torch.Tensor) or not isinstance(current_state, torch.Tensor):
        raise Phase2CCContractError("reloaded real state is malformed")
    policy.to(device=torch.device("cuda"))
    policy.eval()
    generator = torch.Generator(device=state.device).manual_seed(31_415)
    noise = torch.randn(
        (
            state.shape[0],
            int(policy.config.chunk_size),
            int(policy.config.max_action_dim),
        ),
        generator=generator,
        device=state.device,
        dtype=state.dtype,
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        latent = postprocessor(policy.predict_action_chunk(dict(processed), noise=noise))
    if not isinstance(latent, torch.Tensor) or not bool(torch.isfinite(latent).all()):
        raise Phase2CCContractError("reloaded checkpoint generated invalid residuals")
    actions = transform.decode(
        latent.to(dtype=torch.float32, device="cpu"),
        current_state.to(dtype=torch.float32, device="cpu"),
    )
    return {
        "checkpoint_sha256": expected_checkpoint_sha256,
        "checkpoint_reconstructed": True,
        "processor_reconstructed": True,
        "action_transform_reconstructed": True,
        "generated_shape": list(actions.shape),
        "generated_nonconstant": bool(torch.any(torch.abs(actions - actions[:, :1]) > 0)),
        "generated_minimum": actions.amin(dim=(0, 1)).tolist(),
        "generated_maximum": actions.amax(dim=(0, 1)).tolist(),
        "bounded": True,
        "passed": True,
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise Phase2CCContractError("relative SmolVLA training requires CUDA")
    stage = str(args.stage)
    training = SmolVLATrainingConfig(
        num_workers=(
            SmolVLATrainingConfig().num_workers if args.num_workers is None else args.num_workers
        )
    )
    repository = _repository_identity(str(args.training_git_commit))
    static = _read_fingerprinted(
        args.static_preparation.resolve(),
        schema_version="langmani-v2-phase2c-c-static-preparation-complete-v1",
    )
    base_audit = _read_fingerprinted(
        args.base_audit_completion.resolve(),
        schema_version="langmani-v2-phase2c-b-base-audit-complete-v0",
    )
    transform = StateRelativeBoundedActionV0()
    if (
        static.get("training_config_fingerprint") != training.fingerprint
        or static.get("action_transform_fingerprint") != transform.fingerprint
        or static.get("train_scale_exact_match") is not True
        or static.get("full_reconstruction_passed") is not True
        or not isinstance(static.get("real_view_fingerprint"), str)
    ):
        raise Phase2CCContractError("static preparation does not authorize training")
    target_steps = {"smoke": 1, "micro": MICRO_STEPS, "full": training.total_steps}[stage]
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    random.seed(training.seed)
    np.random.seed(training.seed)
    torch.manual_seed(training.seed)
    torch.cuda.manual_seed_all(training.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()

    view = load_dataset_view(
        args.primary_root.resolve(),
        model_kind=ModelKind.PICK,
        split="train",
    )
    if stage == "micro":
        view = _micro_view(view, ModelKind.PICK)
    statistics = load_smolvla_statistics(
        args.primary_root.resolve(),
        model_kind=ModelKind.PICK,
    )
    start_step = 0
    resume_state: dict[str, object] | None = None
    loading_audit: dict[str, object] | None = None
    if args.resume_checkpoint is None:
        config = build_official_config(
            base_snapshot=args.base_snapshot.resolve(),
            vlm_snapshot=args.vlm_snapshot.resolve(),
            device="cuda",
            training=training,
        )
        policy, loading_audit = load_relative_pretrained_policy(
            base_snapshot=args.base_snapshot.resolve(),
            config=config,
        )
        preprocessor, postprocessor = make_processors(config, statistics)
    else:
        checkpoint = args.resume_checkpoint.resolve()
        policy = load_relative_checkpoint(checkpoint, strict=True)
        from lerobot.policies.factory import (  # type: ignore[import-untyped]
            make_pre_post_processors,
        )

        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=checkpoint,
        )
        resume_state = torch.load(
            checkpoint / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(resume_state, dict):
            raise Phase2CCContractError("resume training state is malformed")
        raw_start = resume_state.get("optimizer_step")
        if not isinstance(raw_start, int) or not 0 < raw_start < target_steps:
            raise Phase2CCContractError("resume step lies outside this stage")
        start_step = raw_start
        _restore_rng(cast(Mapping[str, object], resume_state["rng"]))
    config_equivalence = validate_unchanged_training_config(policy.config, training)
    processor = processor_manifest(config=policy.config, statistics=statistics)
    policy.to(device=torch.device("cuda"))
    loader = _dataloader(
        view,
        kind=ModelKind.PICK,
        training=training,
        start_optimizer_step=start_step,
        num_workers=training.num_workers,
    )
    fixed_physical = _physical_fixed_batch(view)
    initial_loss = _evaluate_fixed_loss(policy, preprocessor, fixed_physical)
    initial_diagnostics = _predict_diagnostics(
        policy,
        preprocessor,
        postprocessor,
        fixed_physical,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=training.learning_rate,
        betas=training.betas,
        eps=training.optimizer_epsilon,
        weight_decay=training.weight_decay,
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: _lr_factor(training, step))
    if resume_state is not None:
        optimizer.load_state_dict(cast(dict[str, Any], resume_state["optimizer"]))
        scheduler.load_state_dict(cast(dict[str, Any], resume_state["scheduler"]))
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=training.gradient_accumulation,
    )
    policy, optimizer, loader, scheduler = accelerator.prepare(
        policy,
        optimizer,
        loader,
        scheduler,
    )
    manifest_semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-c-training-run-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "model_kind": "pick_smolvla_relative",
        "task_id": "PickCube-v1",
        "stage": stage,
        "repository": repository,
        "static_preparation_fingerprint": static["fingerprint"],
        "base_audit_fingerprint": base_audit["fingerprint"],
        "real_view_fingerprint": static["real_view_fingerprint"],
        "target_optimizer_steps": target_steps,
        "start_optimizer_step": start_step,
        "training_config": training.to_dict(),
        "training_config_fingerprint": training.fingerprint,
        "config_equivalence": config_equivalence,
        "processor_contract": processor,
        "action_transform": transform.semantic_dict(),
        "base_snapshot": args.base_snapshot.resolve().as_posix(),
        "vlm_snapshot": args.vlm_snapshot.resolve().as_posix(),
        "dataset_roots": dict(view.task_roots),
        "task_lengths": dict(view.task_lengths),
        "task_id_model_input": False,
        "future_state_model_input": False,
        "privileged_policy_fields": [],
        "pretrained_loading": loading_audit,
        "resume_checkpoint": (
            args.resume_checkpoint.resolve().as_posix()
            if args.resume_checkpoint is not None
            else None
        ),
    }
    manifest = {**manifest_semantic, "fingerprint": canonical_fingerprint(manifest_semantic)}
    _write_new_or_equal(run_root / "run_manifest.json", manifest)
    started = time.perf_counter()
    global_step = start_step
    last_loss: float | None = None
    last_grad_norm: float | None = None
    sample_counts = Counter[str]()
    for raw_batch in cast(Any, loader):
        batch = project_relative_policy_batch(raw_batch)
        task_ids = raw_batch.get("task_id")
        if isinstance(task_ids, list):
            sample_counts.update(str(item) for item in task_ids)
        processed = preprocessor(batch)
        if not isinstance(processed, Mapping):
            raise Phase2CCContractError("real preprocessor returned a non-mapping")
        with accelerator.accumulate(policy):
            with accelerator.autocast():
                loss, loss_details = policy(dict(processed))
            if not bool(torch.isfinite(loss)):
                raise Phase2CCContractError("relative SmolVLA loss became non-finite")
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(
                    policy.parameters(),
                    training.gradient_clip_norm,
                )
                if not bool(torch.isfinite(grad_norm)):
                    raise Phase2CCContractError("relative gradient norm became non-finite")
                last_grad_norm = float(grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        if not accelerator.sync_gradients:
            continue
        global_step += 1
        last_loss = float(loss.detach())
        if accelerator.is_main_process and (
            global_step == 1 or global_step % 10 == 0 or global_step == target_steps
        ):
            _append_jsonl(
                run_root / "training_metrics.jsonl",
                {
                    "optimizer_step": global_step,
                    "loss": last_loss,
                    "grad_norm": last_grad_norm,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "loss_details": dict(loss_details),
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                },
            )
        checkpoint_now = global_step == target_steps or (
            stage == "full" and global_step in training.checkpoint_steps
        )
        if checkpoint_now:
            _save_checkpoint(
                accelerator=accelerator,
                policy=cast(RelativeSmolVLAPolicyV0, policy),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                optimizer=optimizer,
                scheduler=cast(LambdaLR, scheduler),
                run_root=run_root,
                stage=stage,
                step=global_step,
                training=training,
                processor_fingerprint=str(processor["fingerprint"]),
                training_git_commit=str(repository["git_commit"]),
                static_preparation_fingerprint=str(static["fingerprint"]),
                base_audit_fingerprint=str(base_audit["fingerprint"]),
                real_view_fingerprint=str(static["real_view_fingerprint"]),
                run_manifest_fingerprint=str(manifest["fingerprint"]),
            )
        if global_step >= target_steps:
            break
    if global_step != target_steps or last_loss is None:
        raise Phase2CCContractError("training ended before the target optimizer step")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = cast(RelativeSmolVLAPolicyV0, accelerator.unwrap_model(policy))
        final_loss = _evaluate_fixed_loss(unwrapped, preprocessor, fixed_physical)
        final_diagnostics = _predict_diagnostics(
            unwrapped,
            preprocessor,
            postprocessor,
            fixed_physical,
        )
        registry_rows = [
            json.loads(line)
            for line in (run_root / "checkpoint_registry.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        final_record = registry_rows[-1]
        reload = _reload_audit(
            _checkpoint_path(run_root, global_step),
            physical_batch=fixed_physical,
            expected_checkpoint_sha256=str(final_record["sha256"]),
        )
        initial_physical = float(initial_diagnostics["physical_action_mae"])
        final_physical = float(final_diagnostics["physical_action_mae"])
        initial_residual = float(initial_diagnostics["residual_action_mae"])
        final_residual = float(final_diagnostics["residual_action_mae"])
        semantic: dict[str, object] = {
            "schema_version": "langmani-v2-phase2c-c-training-result-v0",
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "model_kind": "pick_smolvla_relative",
            "task_id": "PickCube-v1",
            "stage": stage,
            "completed": True,
            "optimizer_steps": global_step,
            "initial_fixed_batch_loss": initial_loss,
            "final_fixed_batch_loss": final_loss,
            "fixed_batch_loss_decreased": final_loss < initial_loss,
            "initial_diagnostics": initial_diagnostics,
            "final_diagnostics": final_diagnostics,
            "physical_action_error_decreased": final_physical < initial_physical,
            "residual_action_error_decreased": final_residual < initial_residual,
            "last_training_loss": last_loss,
            "last_gradient_norm": last_grad_norm,
            "sample_counts_by_task": dict(sample_counts),
            "checkpoint_records": registry_rows,
            "final_checkpoint_reload": reload,
            "training_duration_seconds": time.perf_counter() - started,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
            "action_bounds_structural": True,
            "clipping_events": 0,
            "projection_events": 0,
            "task_id_model_input": False,
            "future_state_model_input": False,
            "passed": reload["passed"] is True
            and math.isfinite(last_loss)
            and (
                stage == "smoke"
                or (stage == "full" and final_loss < initial_loss)
                or (
                    stage == "micro"
                    and final_loss < initial_loss
                    and final_physical < initial_physical
                    and final_residual < initial_residual
                    and final_diagnostics["outputs_differ_across_observations"] is True
                )
            ),
        }
        result = {**semantic, "fingerprint": canonical_fingerprint(semantic)}
        _write_new_or_equal(run_root / "training_result.json", result)
        if result["passed"] is not True:
            raise Phase2CCContractError("relative training stage failed its gate")
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
