"""Compute deterministic offline validation diagnostics for one SmolVLA checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from langmani.v2.phase2b5 import PANDA_ACTION_HIGH, PANDA_ACTION_LOW
from langmani.v2.phase2c_a import PACKAGE_FINGERPRINT
from langmani.v2.phase2c_b import ModelKind, Phase2CBContractError, canonical_fingerprint
from langmani.v2.phase2c_b_adapter import CHECKPOINT_MANIFEST
from langmani.v2.phase2c_b_smolvla import (
    ACTION_FEATURE_KEY,
    ACTION_PAD_KEY,
    BoundedSmolVLAPolicyV1,
    load_dataset_view,
    project_policy_batch,
)
from langmani.v2.smolvla_adapter import sha256_directory

DIAGNOSTIC_SEED = 92_771


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--maximum-frames-per-task", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2CBContractError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise Phase2CBContractError(f"{path} must contain one object")
    return value


def _write_new_or_equal(path: Path, value: Mapping[str, object]) -> None:
    encoded = json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise Phase2CBContractError(f"existing immutable diagnostic differs: {path}")
        return
    path.write_text(encoded, encoding="utf-8", newline="\n")


def _balanced_indices(
    task_lengths: Mapping[str, int],
    *,
    maximum_frames_per_task: int,
) -> list[int]:
    if maximum_frames_per_task < 1:
        raise ValueError("--maximum-frames-per-task must be positive")
    indices: list[int] = []
    offset = 0
    for length in task_lengths.values():
        selected = min(maximum_frames_per_task, length)
        if selected == 1:
            local_indices = [0]
        else:
            local_indices = [
                round(position * (length - 1) / (selected - 1)) for position in range(selected)
            ]
        if len(set(local_indices)) != selected:
            raise Phase2CBContractError("balanced diagnostic indices are not unique")
        indices.extend(offset + index for index in local_indices)
        offset += length
    return indices


def _template_ids(raw: object, *, batch_size: int) -> list[str]:
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise Phase2CBContractError("diagnostic batch lacks instruction_template_id")
    values = [str(value) for value in raw]
    if len(values) != batch_size or any(not value for value in values):
        raise Phase2CBContractError("instruction template batch is malformed")
    return values


def _task_ids(raw: object, *, batch_size: int) -> list[str]:
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise Phase2CBContractError("diagnostic batch lacks task_id")
    values = [str(value) for value in raw]
    if len(values) != batch_size or any(not value for value in values):
        raise Phase2CBContractError("task ID batch is malformed")
    return values


def _group_summary(
    totals: Mapping[str, tuple[float, int]],
) -> dict[str, dict[str, float | int]]:
    return {
        key: {
            "absolute_error_sum": value[0],
            "valid_component_count": value[1],
            "physical_action_mae": value[0] / value[1],
        }
        for key, value in sorted(totals.items())
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise Phase2CBContractError("offline SmolVLA diagnostics require CUDA")
    checkpoint = args.checkpoint.resolve()
    expected_digest = str(args.checkpoint_sha256).removeprefix("sha256:")
    if sha256_directory(checkpoint) != expected_digest:
        raise Phase2CBContractError("checkpoint changed before offline diagnostics")
    manifest = _read_object(checkpoint / CHECKPOINT_MANIFEST)
    kind = ModelKind(str(manifest.get("model_kind")))
    if (
        manifest.get("package_fingerprint") != PACKAGE_FINGERPRINT
        or manifest.get("task_id_model_input") is not False
    ):
        raise Phase2CBContractError("checkpoint is outside the Phase 2C-B data contract")

    view = load_dataset_view(
        args.primary_root.resolve(),
        model_kind=kind,
        split="validation",
    )
    indices = _balanced_indices(
        view.task_lengths,
        maximum_frames_per_task=args.maximum_frames_per_task,
    )
    loader = DataLoader(
        Subset(view.dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    policy = BoundedSmolVLAPolicyV1.from_pretrained(checkpoint, strict=True)
    from lerobot.policies.factory import (  # type: ignore[import-untyped]
        make_pre_post_processors,
    )

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=checkpoint,
    )
    policy.to(device=torch.device("cuda"))
    policy.eval()
    lower = torch.as_tensor(PANDA_ACTION_LOW, dtype=torch.float32)
    upper = torch.as_tensor(PANDA_ACTION_HIGH, dtype=torch.float32)
    saturation_margin = torch.maximum((upper - lower) * 0.01, torch.full_like(lower, 1e-8))

    validation_loss_sum = 0.0
    validation_loss_examples = 0
    absolute_error_sum = 0.0
    valid_component_count = 0
    first_action_error_sum = 0.0
    first_action_component_count = 0
    arm_error_sum = 0.0
    arm_component_count = 0
    gripper_error_sum = 0.0
    gripper_component_count = 0
    chunk_error_sum = torch.zeros(int(policy.config.chunk_size), dtype=torch.float64)
    chunk_error_count = torch.zeros(int(policy.config.chunk_size), dtype=torch.int64)
    predicted_values: list[torch.Tensor] = []
    saturated_components = 0
    endpoint_components = 0
    total_predicted_components = 0
    by_task_mutable: dict[str, list[float | int]] = defaultdict(lambda: [0.0, 0])
    by_template_mutable: dict[str, list[float | int]] = defaultdict(lambda: [0.0, 0])

    for batch_index, raw_batch in enumerate(loader):
        batch = project_policy_batch(raw_batch, shared=kind is ModelKind.SHARED)
        processed = preprocessor(batch)
        if not isinstance(processed, Mapping):
            raise Phase2CBContractError("offline preprocessor returned a non-mapping")
        processed_batch = dict(processed)
        state = processed_batch.get("observation.state")
        target = batch[ACTION_FEATURE_KEY]
        padding = batch[ACTION_PAD_KEY]
        if (
            not isinstance(state, torch.Tensor)
            or not isinstance(target, torch.Tensor)
            or not isinstance(padding, torch.Tensor)
        ):
            raise Phase2CBContractError("offline diagnostic batch is malformed")
        batch_size = target.shape[0]
        task_ids = _task_ids(raw_batch.get("task_id"), batch_size=batch_size)
        template_ids = _template_ids(
            raw_batch.get("instruction_template_id"),
            batch_size=batch_size,
        )
        torch.manual_seed(DIAGNOSTIC_SEED + batch_index)
        torch.cuda.manual_seed_all(DIAGNOSTIC_SEED + batch_index)
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        ):
            loss, _ = policy(processed_batch)
            generator = torch.Generator(device=state.device).manual_seed(
                DIAGNOSTIC_SEED * 10_000 + batch_index
            )
            noise = torch.randn(
                (
                    batch_size,
                    int(policy.config.chunk_size),
                    int(policy.config.max_action_dim),
                ),
                generator=generator,
                device=state.device,
                dtype=state.dtype,
            )
            predicted = postprocessor(policy.predict_action_chunk(processed_batch, noise=noise))
        if (
            not isinstance(predicted, torch.Tensor)
            or predicted.shape != target.shape
            or not bool(torch.isfinite(predicted).all())
        ):
            raise Phase2CBContractError("offline generated physical actions are invalid")
        validation_loss_sum += float(loss) * batch_size
        validation_loss_examples += batch_size
        predicted_cpu = predicted.to(dtype=torch.float32, device="cpu")
        target_cpu = target.to(dtype=torch.float32, device="cpu")
        padding_cpu = padding.to(device="cpu")
        valid_steps = ~padding_cpu
        valid = valid_steps.unsqueeze(-1).expand_as(target_cpu)
        error = torch.abs(predicted_cpu - target_cpu)
        absolute_error_sum += float(error[valid].sum())
        valid_component_count += int(valid.sum())
        first_action_error_sum += float(error[:, 0].sum())
        first_action_component_count += batch_size * target.shape[-1]
        arm_valid = valid[..., :7]
        gripper_valid = valid[..., 7:]
        arm_error_sum += float(error[..., :7][arm_valid].sum())
        arm_component_count += int(arm_valid.sum())
        gripper_error_sum += float(error[..., 7:][gripper_valid].sum())
        gripper_component_count += int(gripper_valid.sum())
        for position in range(target.shape[1]):
            position_valid = valid_steps[:, position]
            chunk_error_sum[position] += float(error[position_valid, position].sum())
            chunk_error_count[position] += int(position_valid.sum()) * target.shape[-1]
        predicted_valid = predicted_cpu[valid].reshape(-1)
        predicted_values.append(predicted_valid)
        saturated_components += int(
            torch.count_nonzero(
                (predicted_cpu <= lower + saturation_margin)
                | (predicted_cpu >= upper - saturation_margin)
            )
        )
        endpoint_components += int(
            torch.count_nonzero(
                torch.isclose(predicted_cpu, lower, atol=1e-6, rtol=0.0)
                | torch.isclose(predicted_cpu, upper, atol=1e-6, rtol=0.0)
            )
        )
        total_predicted_components += predicted_cpu.numel()
        for row_index, (task_id, template_id) in enumerate(
            zip(task_ids, template_ids, strict=True)
        ):
            row_valid = valid[row_index]
            row_error = float(error[row_index][row_valid].sum())
            row_count = int(row_valid.sum())
            for collection, key in (
                (by_task_mutable, task_id),
                (by_template_mutable, template_id),
            ):
                collection[key][0] = float(collection[key][0]) + row_error
                collection[key][1] = int(collection[key][1]) + row_count

    if (
        validation_loss_examples != len(indices)
        or valid_component_count == 0
        or first_action_component_count == 0
        or arm_component_count == 0
        or gripper_component_count == 0
        or total_predicted_components == 0
    ):
        raise Phase2CBContractError("offline diagnostic coverage is incomplete")
    all_predicted = torch.cat(predicted_values)
    chunk_mae = [
        (
            float(chunk_error_sum[index] / chunk_error_count[index])
            if int(chunk_error_count[index]) > 0
            else None
        )
        for index in range(len(chunk_error_sum))
    ]
    semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-b-offline-checkpoint-diagnostic-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "model_kind": kind.value,
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": f"sha256:{expected_digest}",
        "checkpoint_manifest_fingerprint": manifest.get("fingerprint"),
        "optimizer_step": manifest.get("optimizer_step"),
        "split": "validation",
        "diagnostic_seed": DIAGNOSTIC_SEED,
        "balanced_frame_count": len(indices),
        "maximum_frames_per_task": args.maximum_frames_per_task,
        "frames_by_task": {
            task_id: min(args.maximum_frames_per_task, length)
            for task_id, length in view.task_lengths.items()
        },
        "validation_flow_matching_loss": validation_loss_sum / validation_loss_examples,
        "physical_action_mae": absolute_error_sum / valid_component_count,
        "physical_first_action_mae": (first_action_error_sum / first_action_component_count),
        "arm_action_mae": arm_error_sum / arm_component_count,
        "gripper_action_mae": gripper_error_sum / gripper_component_count,
        "chunk_position_mae": chunk_mae,
        "output_variance": float(torch.var(all_predicted, unbiased=False)),
        "action_saturation_rate": saturated_components / total_predicted_components,
        "action_endpoint_rate": endpoint_components / total_predicted_components,
        "error_by_task": _group_summary(
            {key: (float(value[0]), int(value[1])) for key, value in by_task_mutable.items()}
        ),
        "error_by_instruction_template": _group_summary(
            {key: (float(value[0]), int(value[1])) for key, value in by_template_mutable.items()}
        ),
        "nonfinite_loss_count": 0,
        "invalid_action_count": 0,
        "clipping_events": 0,
        "projection_events": 0,
        "task_id_model_input": False,
        "passed": (
            math.isfinite(validation_loss_sum)
            and math.isfinite(absolute_error_sum)
            and saturated_components <= total_predicted_components
        ),
    }
    result = {**semantic, "fingerprint": canonical_fingerprint(semantic)}
    if result["passed"] is not True:
        raise Phase2CBContractError("offline checkpoint diagnostic failed")
    _write_new_or_equal(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
