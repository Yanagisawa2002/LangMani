"""Run one ordered Phase 2C-B SmolVLA smoke, micro, or full training stage."""

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
from accelerate import Accelerator  # type: ignore[import-untyped]
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, Subset

from langmani.policies.act_training import DeterministicResumeBatchSampler
from langmani.v2.phase2c_a import (
    PACKAGE_FINGERPRINT,
    TASK_IDS,
    UniformTaskBatchSampler,
)
from langmani.v2.phase2c_b import (
    OFFICIAL_BASE_MODEL,
    OFFICIAL_BASE_REVISION,
    BoundedActionLatentV1,
    ModelKind,
    Phase2CBContractError,
    SmolVLATrainingConfig,
    canonical_fingerprint,
    validate_prerequisite_completion,
)
from langmani.v2.phase2c_b_adapter import CHECKPOINT_MANIFEST
from langmani.v2.phase2c_b_smolvla import (
    ACTION_FEATURE_KEY,
    ACTION_PAD_KEY,
    BoundedSmolVLAPolicyV1,
    LoadedSmolVLAView,
    build_official_config,
    load_dataset_view,
    load_pretrained_policy,
    load_smolvla_statistics,
    make_processors,
    processor_manifest,
    project_policy_batch,
)
from langmani.v2.smolvla_adapter import sha256_directory

MICRO_STEPS = 500
MICRO_FRAMES_PER_TASK = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--base-snapshot", type=Path, required=True)
    parser.add_argument("--vlm-snapshot", type=Path, required=True)
    parser.add_argument("--model-kind", choices=[kind.value for kind in ModelKind], required=True)
    parser.add_argument("--stage", choices=("smoke", "micro", "full"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--training-git-commit", required=True)
    parser.add_argument("--static-preparation", type=Path, required=True)
    parser.add_argument("--base-audit-completion", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--num-workers", type=int)
    return parser.parse_args()


def _write_new_or_equal(path: Path, value: Mapping[str, object]) -> None:
    encoded = json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise Phase2CBContractError(f"existing immutable artifact differs: {path}")
        return
    path.write_text(encoded, encoding="utf-8", newline="\n")


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(value), sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_fingerprinted_completion(
    path: Path,
    *,
    schema_version: str,
) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2CBContractError(f"cannot read prerequisite {path}: {error}") from error
    if not isinstance(value, dict):
        raise Phase2CBContractError(f"prerequisite {path} must contain one object")
    try:
        return validate_prerequisite_completion(
            value,
            schema_version=schema_version,
        )
    except Phase2CBContractError as error:
        raise Phase2CBContractError(f"prerequisite identity is invalid: {path}") from error


def _repository_identity(expected_commit: str) -> dict[str, object]:
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise Phase2CBContractError("--training-git-commit must be one full Git SHA")

    def run(*command: str) -> str:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return completed.stdout.strip()

    head = run("git", "rev-parse", "HEAD")
    branch = run("git", "branch", "--show-current")
    status = run("git", "status", "--porcelain")
    upstream = run("git", "rev-parse", "@{upstream}")
    if (
        head != expected_commit
        or branch != "codex/langmani-v2-phase2c-b-smolvla"
        or status
        or upstream != head
    ):
        raise Phase2CBContractError("training repository identity/cleanliness check failed")
    return {
        "git_commit": head,
        "branch": branch,
        "upstream": upstream,
        "clean": True,
    }


def _micro_view(view: LoadedSmolVLAView, kind: ModelKind) -> LoadedSmolVLAView:
    indices: list[int] = []
    lengths: dict[str, int] = {}
    offset = 0
    for task_id in kind.compatible_task_ids:
        length = view.task_lengths[task_id]
        selected = min(MICRO_FRAMES_PER_TASK, length)
        indices.extend(range(offset, offset + selected))
        lengths[task_id] = selected
        offset += length
    return LoadedSmolVLAView(
        dataset=cast(Dataset[Mapping[str, object]], Subset(view.dataset, indices)),
        task_lengths=lengths,
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
    start_batch = start_optimizer_step * training.gradient_accumulation
    sampler: Any
    if kind is ModelKind.SHARED:
        sampler = UniformTaskBatchSampler(
            {task_id: view.task_lengths[task_id] for task_id in TASK_IDS},
            batch_size=training.batch_size,
            seed=training.seed,
            start_batch=start_batch,
        )
    else:
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


def _save_checkpoint(
    *,
    accelerator: Accelerator,
    policy: BoundedSmolVLAPolicyV1,
    preprocessor: object,
    postprocessor: object,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    run_root: Path,
    kind: ModelKind,
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
            raise Phase2CBContractError(f"refusing to overwrite checkpoint {checkpoint}")
        checkpoint.mkdir(parents=True)
        unwrapped = cast(BoundedSmolVLAPolicyV1, accelerator.unwrap_model(policy))
        unwrapped.save_pretrained(checkpoint, safe_serialization=True)
        cast(Any, preprocessor).save_pretrained(checkpoint)
        cast(Any, postprocessor).save_pretrained(checkpoint)
        unwrapped.action_transform.save(checkpoint)
        manifest_semantic: dict[str, object] = {
            "schema_version": "langmani-v2-phase2c-b-checkpoint-manifest-v0",
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "model_kind": kind.value,
            "stage": stage,
            "optimizer_step": step,
            "training_config_fingerprint": training.fingerprint,
            "processor_contract_fingerprint": processor_fingerprint,
            "training_git_commit": training_git_commit,
            "static_preparation_fingerprint": static_preparation_fingerprint,
            "base_audit_fingerprint": base_audit_fingerprint,
            "real_view_fingerprint": real_view_fingerprint,
            "run_manifest_fingerprint": run_manifest_fingerprint,
            "action_transform_fingerprint": unwrapped.action_transform.fingerprint,
            "base_model": OFFICIAL_BASE_MODEL,
            "base_revision": OFFICIAL_BASE_REVISION,
            "task_id_model_input": False,
            "language_input": "task",
            "post_hoc_clipping": False,
            "projection": False,
        }
        _write_new_or_equal(
            checkpoint / CHECKPOINT_MANIFEST,
            {
                **manifest_semantic,
                "fingerprint": canonical_fingerprint(manifest_semantic),
            },
        )
        torch.save(
            {
                "schema_version": "langmani-v2-phase2c-b-training-state-v0",
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
    raw_batch: Mapping[str, object],
    expected_checkpoint_sha256: str,
) -> dict[str, object]:
    if sha256_directory(checkpoint) != expected_checkpoint_sha256.removeprefix("sha256:"):
        raise Phase2CBContractError("checkpoint changed before reload audit")
    BoundedActionLatentV1.load(checkpoint)
    policy = BoundedSmolVLAPolicyV1.from_pretrained(checkpoint, strict=True)
    from lerobot.policies.factory import (  # type: ignore[import-untyped]
        make_pre_post_processors,
    )

    processors = make_pre_post_processors(policy.config, pretrained_path=checkpoint)
    policy.to(device=torch.device("cuda"))
    policy.eval()
    processed = processors[0](dict(raw_batch))
    if not isinstance(processed, Mapping):
        raise Phase2CBContractError("reloaded processor returned a non-mapping")
    state = processed["observation.state"]
    assert isinstance(state, torch.Tensor)
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
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        actions = processors[1](policy.predict_action_chunk(dict(processed), noise=noise))
    if (
        not isinstance(actions, torch.Tensor)
        or not bool(torch.isfinite(actions).all())
        or actions.shape[-1] != 8
    ):
        raise Phase2CBContractError("reloaded checkpoint failed real action generation")
    return {
        "checkpoint_sha256": expected_checkpoint_sha256,
        "checkpoint_reconstructed": True,
        "processor_reconstructed": True,
        "generated_shape": list(actions.shape),
        "generated_nonconstant": bool(torch.any(torch.abs(actions - actions[:, :1]) > 0)),
        "generated_minimum": actions.amin(dim=(0, 1)).tolist(),
        "generated_maximum": actions.amax(dim=(0, 1)).tolist(),
        "bounded": True,
        "passed": True,
    }


def _fixed_batch(view: LoadedSmolVLAView, *, kind: ModelKind) -> dict[str, object]:
    rows = []
    if kind is ModelKind.SHARED:
        offsets = [0]
        for task_id in kind.compatible_task_ids[:-1]:
            offsets.append(offsets[-1] + view.task_lengths[task_id])
    else:
        offsets = [0, 1]
    for index in offsets:
        rows.append(view.dataset[index])
    batch: dict[str, object] = {}
    for key in (
        "observation.images.base_camera",
        "observation.state",
        ACTION_FEATURE_KEY,
        ACTION_PAD_KEY,
        "task",
    ):
        values = [row[key] for row in rows]
        batch[key] = (
            torch.stack(cast(list[torch.Tensor], values))
            if isinstance(values[0], torch.Tensor)
            else values
        )
    return project_policy_batch(batch, shared=kind is ModelKind.SHARED)


def _evaluate_fixed_loss(
    policy: BoundedSmolVLAPolicyV1,
    preprocessor: object,
    raw_batch: Mapping[str, object],
) -> float:
    policy.eval()
    processed = cast(Any, preprocessor)(dict(raw_batch))
    if not isinstance(processed, Mapping):
        raise Phase2CBContractError("fixed-batch preprocessor returned a non-mapping")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = policy(dict(processed))
    value = float(loss)
    if not math.isfinite(value):
        raise Phase2CBContractError("fixed-batch loss is non-finite")
    policy.train()
    return value


def _predict_diagnostics(
    policy: BoundedSmolVLAPolicyV1,
    preprocessor: object,
    postprocessor: object,
    raw_batch: Mapping[str, object],
    *,
    kind: ModelKind,
) -> dict[str, object]:
    policy.eval()
    processed = cast(Any, preprocessor)(dict(raw_batch))
    if not isinstance(processed, Mapping):
        raise Phase2CBContractError("diagnostic preprocessor returned a non-mapping")
    state = processed["observation.state"]
    target = raw_batch[ACTION_FEATURE_KEY]
    padding = raw_batch[ACTION_PAD_KEY]
    if (
        not isinstance(state, torch.Tensor)
        or not isinstance(target, torch.Tensor)
        or not isinstance(padding, torch.Tensor)
    ):
        raise Phase2CBContractError("diagnostic batch is malformed")
    generator = torch.Generator(device=state.device).manual_seed(27_182)
    base_noise = torch.randn(
        (1, int(policy.config.chunk_size), int(policy.config.max_action_dim)),
        generator=generator,
        device=state.device,
        dtype=state.dtype,
    )
    noise = base_noise.expand(state.shape[0], -1, -1).clone()
    with torch.inference_mode():
        predicted = cast(Any, postprocessor)(
            policy.predict_action_chunk(dict(processed), noise=noise)
        )
    if not isinstance(predicted, torch.Tensor) or tuple(predicted.shape) != tuple(target.shape):
        raise Phase2CBContractError("diagnostic generated action shape changed")
    predicted_cpu = predicted.to(dtype=torch.float32, device="cpu")
    target_cpu = target.to(dtype=torch.float32, device="cpu")
    padding_cpu = padding.to(device="cpu")
    valid = (~padding_cpu).unsqueeze(-1).expand_as(target_cpu)
    physical_mae = float(torch.abs(predicted_cpu - target_cpu)[valid].mean())
    first_action_mae = float(torch.abs(predicted_cpu[:, 0] - target_cpu[:, 0]).mean())
    observation_difference = float(torch.max(torch.abs(predicted_cpu[0] - predicted_cpu[1])))
    instruction_difference: float | None = None
    if kind is ModelKind.SHARED:
        intervention_raw: dict[str, object] = {}
        for key, value in raw_batch.items():
            if isinstance(value, torch.Tensor):
                intervention_raw[key] = value[:1].expand(2, *value.shape[1:]).clone()
            else:
                values = cast(list[str], value)
                intervention_raw[key] = [values[0], values[1]]
        intervention_processed = cast(Any, preprocessor)(intervention_raw)
        if not isinstance(intervention_processed, Mapping):
            raise Phase2CBContractError("language diagnostic preprocessing failed")
        intervention_state = intervention_processed["observation.state"]
        assert isinstance(intervention_state, torch.Tensor)
        intervention_noise = base_noise.expand(2, -1, -1).clone()
        with torch.inference_mode():
            intervention_prediction = cast(Any, postprocessor)(
                policy.predict_action_chunk(
                    dict(intervention_processed),
                    noise=intervention_noise,
                )
            )
        if not isinstance(intervention_prediction, torch.Tensor):
            raise Phase2CBContractError("language diagnostic prediction is malformed")
        instruction_difference = float(
            torch.max(torch.abs(intervention_prediction[0] - intervention_prediction[1]))
        )
    policy.train()
    return {
        "physical_action_mae": physical_mae,
        "physical_first_action_mae": first_action_mae,
        "different_observation_max_abs_difference": observation_difference,
        "outputs_differ_across_observations": observation_difference > 1e-8,
        "same_observation_instruction_max_abs_difference": instruction_difference,
        "shared_outputs_differ_across_instructions": (
            instruction_difference > 1e-8 if instruction_difference is not None else None
        ),
        "all_actions_bounded": True,
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise Phase2CBContractError("SmolVLA training requires CUDA")
    kind = ModelKind(args.model_kind)
    stage = str(args.stage)
    if stage == "micro" and kind not in {ModelKind.PICK, ModelKind.SHARED}:
        raise Phase2CBContractError("micro-overfit is authorized only for Pick and shared")
    training = SmolVLATrainingConfig(
        num_workers=(
            SmolVLATrainingConfig().num_workers if args.num_workers is None else args.num_workers
        )
    )
    repository = _repository_identity(str(args.training_git_commit))
    static_preparation = _read_fingerprinted_completion(
        args.static_preparation.resolve(),
        schema_version="langmani-v2-phase2c-b-static-preparation-complete-v0",
    )
    base_audit = _read_fingerprinted_completion(
        args.base_audit_completion.resolve(),
        schema_version="langmani-v2-phase2c-b-base-audit-complete-v0",
    )
    if (
        static_preparation.get("training_config_fingerprint") != training.fingerprint
        or static_preparation.get("bounded_action_fingerprint")
        != BoundedActionLatentV1().fingerprint
        or not isinstance(static_preparation.get("real_view_fingerprints"), Mapping)
        or kind.value
        not in cast(
            Mapping[str, object],
            static_preparation["real_view_fingerprints"],
        )
    ):
        raise Phase2CBContractError("static preparation does not authorize this training config")
    static_preparation_fingerprint = str(static_preparation["fingerprint"])
    base_audit_fingerprint = str(base_audit["fingerprint"])
    real_view_fingerprint = str(
        cast(Mapping[str, object], static_preparation["real_view_fingerprints"])[kind.value]
    )
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
        model_kind=kind,
        split="train",
    )
    if stage == "micro":
        view = _micro_view(view, kind)
    statistics = load_smolvla_statistics(args.primary_root.resolve(), model_kind=kind)
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
        policy, loading_audit = load_pretrained_policy(
            base_snapshot=args.base_snapshot.resolve(),
            config=config,
        )
        preprocessor, postprocessor = make_processors(config, statistics)
    else:
        checkpoint = args.resume_checkpoint.resolve()
        policy = BoundedSmolVLAPolicyV1.from_pretrained(checkpoint, strict=True)
        from lerobot.policies.factory import make_pre_post_processors

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
            raise Phase2CBContractError("resume training state is malformed")
        raw_start_step = resume_state["optimizer_step"]
        if not isinstance(raw_start_step, int):
            raise Phase2CBContractError("resume optimizer step is malformed")
        start_step = raw_start_step
        if not 0 < start_step < target_steps:
            raise Phase2CBContractError("resume step lies outside this stage")
        _restore_rng(cast(Mapping[str, object], resume_state["rng"]))
    processor = processor_manifest(config=policy.config, statistics=statistics)
    policy.to(device=torch.device("cuda"))
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
    loader = _dataloader(
        view,
        kind=kind,
        training=training,
        start_optimizer_step=start_step,
        num_workers=training.num_workers,
    )
    fixed_raw = _fixed_batch(view, kind=kind)
    initial_loss = _evaluate_fixed_loss(policy, preprocessor, fixed_raw)
    initial_diagnostics = _predict_diagnostics(
        policy,
        preprocessor,
        postprocessor,
        fixed_raw,
        kind=kind,
    )
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
        "schema_version": "langmani-v2-phase2c-b-training-run-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "model_kind": kind.value,
        "stage": stage,
        "repository": repository,
        "static_preparation": {
            "path": args.static_preparation.resolve().as_posix(),
            "fingerprint": static_preparation_fingerprint,
        },
        "base_audit": {
            "path": args.base_audit_completion.resolve().as_posix(),
            "fingerprint": base_audit_fingerprint,
        },
        "real_view_fingerprint": real_view_fingerprint,
        "target_optimizer_steps": target_steps,
        "start_optimizer_step": start_step,
        "training_config": training.to_dict(),
        "training_config_fingerprint": training.fingerprint,
        "processor_contract": processor,
        "action_transform": BoundedActionLatentV1().semantic_dict(),
        "base_snapshot": args.base_snapshot.resolve().as_posix(),
        "vlm_snapshot": args.vlm_snapshot.resolve().as_posix(),
        "dataset_roots": dict(view.task_roots),
        "task_lengths": dict(view.task_lengths),
        "task_id_model_input": False,
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
    checkpoint_records: list[dict[str, object]] = []
    for raw_batch in cast(Any, loader):
        batch = project_policy_batch(raw_batch, shared=kind is ModelKind.SHARED)
        task_ids = raw_batch.get("task_id")
        if isinstance(task_ids, list):
            sample_counts.update(str(item) for item in task_ids)
        processed = preprocessor(batch)
        if not isinstance(processed, Mapping):
            raise Phase2CBContractError("real preprocessor returned a non-mapping")
        with accelerator.accumulate(policy):
            with accelerator.autocast():
                loss, loss_details = policy(dict(processed))
            if not bool(torch.isfinite(loss)):
                raise Phase2CBContractError("SmolVLA training loss became non-finite")
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(
                    policy.parameters(),
                    training.gradient_clip_norm,
                )
                if not bool(torch.isfinite(grad_norm)):
                    raise Phase2CBContractError("SmolVLA gradient norm became non-finite")
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
            record = _save_checkpoint(
                accelerator=accelerator,
                policy=cast(BoundedSmolVLAPolicyV1, policy),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                optimizer=optimizer,
                scheduler=cast(LambdaLR, scheduler),
                run_root=run_root,
                kind=kind,
                stage=stage,
                step=global_step,
                training=training,
                processor_fingerprint=str(processor["fingerprint"]),
                training_git_commit=str(repository["git_commit"]),
                static_preparation_fingerprint=static_preparation_fingerprint,
                base_audit_fingerprint=base_audit_fingerprint,
                real_view_fingerprint=real_view_fingerprint,
                run_manifest_fingerprint=str(manifest["fingerprint"]),
            )
            if accelerator.is_main_process:
                checkpoint_records.append(record)
        if global_step >= target_steps:
            break
    if global_step != target_steps or last_loss is None:
        raise Phase2CBContractError("SmolVLA training ended before the target optimizer step")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = cast(BoundedSmolVLAPolicyV1, accelerator.unwrap_model(policy))
        final_loss = _evaluate_fixed_loss(unwrapped, preprocessor, fixed_raw)
        final_diagnostics = _predict_diagnostics(
            unwrapped,
            preprocessor,
            postprocessor,
            fixed_raw,
            kind=kind,
        )
        final_checkpoint = _checkpoint_path(run_root, global_step)
        registry_rows = [
            json.loads(line)
            for line in (run_root / "checkpoint_registry.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        final_record = registry_rows[-1]
        reload = _reload_audit(
            final_checkpoint,
            raw_batch=fixed_raw,
            expected_checkpoint_sha256=str(final_record["sha256"]),
        )
        duration = time.perf_counter() - started
        initial_action_mae = cast(float, initial_diagnostics["physical_action_mae"])
        final_action_mae = cast(float, final_diagnostics["physical_action_mae"])
        semantic: dict[str, object] = {
            "schema_version": "langmani-v2-phase2c-b-training-result-v0",
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "model_kind": kind.value,
            "stage": stage,
            "completed": True,
            "optimizer_steps": global_step,
            "initial_fixed_batch_loss": initial_loss,
            "final_fixed_batch_loss": final_loss,
            "fixed_batch_loss_decreased": final_loss < initial_loss,
            "initial_diagnostics": initial_diagnostics,
            "final_diagnostics": final_diagnostics,
            "physical_action_error_decreased": final_action_mae < initial_action_mae,
            "last_training_loss": last_loss,
            "last_gradient_norm": last_grad_norm,
            "sample_counts_by_task": dict(sample_counts),
            "checkpoint_records": registry_rows,
            "final_checkpoint_reload": reload,
            "training_duration_seconds": duration,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
            "action_bounds_structural": True,
            "clipping_events": 0,
            "projection_events": 0,
            "task_id_model_input": False,
            "passed": reload["passed"] is True
            and math.isfinite(last_loss)
            and (
                stage == "smoke"
                or (stage == "full" and final_loss < initial_loss)
                or (
                    stage == "micro"
                    and final_loss < initial_loss
                    and final_action_mae < initial_action_mae
                    and final_diagnostics["outputs_differ_across_observations"] is True
                    and (
                        kind is not ModelKind.SHARED
                        or final_diagnostics["shared_outputs_differ_across_instructions"] is True
                    )
                )
            ),
        }
        result = {**semantic, "fingerprint": canonical_fingerprint(semantic)}
        _write_new_or_equal(run_root / "training_result.json", result)
        if result["passed"] is not True:
            raise Phase2CBContractError("SmolVLA training stage failed its gate")
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
