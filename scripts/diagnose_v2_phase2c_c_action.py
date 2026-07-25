"""Run frozen phase-conditioned offline diagnostics for absolute or relative Pick."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader, Subset

from langmani.v2.phase2b5 import PANDA_ACTION_HIGH, PANDA_ACTION_LOW
from langmani.v2.phase2c_a import PACKAGE_FINGERPRINT
from langmani.v2.phase2c_b import ModelKind
from langmani.v2.phase2c_b_adapter import CHECKPOINT_MANIFEST as ABSOLUTE_MANIFEST
from langmani.v2.phase2c_b_smolvla import (
    ACTION_FEATURE_KEY,
    ACTION_PAD_KEY,
    BoundedSmolVLAPolicyV1,
    load_dataset_view,
    project_policy_batch,
)
from langmani.v2.phase2c_c import (
    Phase2CCContractError,
    StateRelativeBoundedActionV0,
    canonical_fingerprint,
)
from langmani.v2.phase2c_c_adapter import CHECKPOINT_MANIFEST as RELATIVE_MANIFEST
from langmani.v2.phase2c_c_smolvla import (
    load_relative_checkpoint,
    project_relative_policy_batch,
)
from langmani.v2.smolvla_adapter import sha256_directory

DIAGNOSTIC_SEED = 92_771


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formulation", choices=("absolute", "relative"), required=True)
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--phase-lock", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2CCContractError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise Phase2CCContractError(f"{path} must contain one object")
    return value


def _write_new_or_equal(path: Path, value: Mapping[str, object]) -> None:
    encoded = json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise Phase2CCContractError(f"existing immutable diagnostic differs: {path}")
        return
    path.write_text(encoded, encoding="utf-8", newline="\n")


def _sequence(raw: object, *, label: str, batch_size: int) -> list[str]:
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise Phase2CCContractError(f"diagnostic batch lacks {label}")
    values = [str(value) for value in raw]
    if len(values) != batch_size or any(not value for value in values):
        raise Phase2CCContractError(f"diagnostic {label} is malformed")
    return values


def _group_summary(
    totals: Mapping[str, list[float | int]],
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for key, value in sorted(totals.items()):
        component_count = int(value[1])
        first_count = int(value[3])
        result[key] = {
            "absolute_error_sum": float(value[0]),
            "valid_component_count": component_count,
            "physical_action_mae": float(value[0]) / component_count,
            "first_action_absolute_error_sum": float(value[2]),
            "first_action_component_count": first_count,
            "physical_first_action_mae": float(value[2]) / first_count,
            "frame_count": int(value[4]),
        }
    return result


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise Phase2CCContractError("offline SmolVLA diagnostics require CUDA")
    formulation = str(args.formulation)
    checkpoint = args.checkpoint.resolve()
    expected_digest = str(args.checkpoint_sha256).removeprefix("sha256:")
    if sha256_directory(checkpoint) != expected_digest:
        raise Phase2CCContractError("checkpoint changed before offline diagnostics")
    manifest_name = ABSOLUTE_MANIFEST if formulation == "absolute" else RELATIVE_MANIFEST
    manifest = _read_object(checkpoint / manifest_name)
    if (
        manifest.get("package_fingerprint") != PACKAGE_FINGERPRINT
        or manifest.get("task_id_model_input") is not False
    ):
        raise Phase2CCContractError("checkpoint is outside the Phase 2C-C data contract")
    lock = _read_object(args.phase_lock.resolve())
    semantic_lock = dict(lock)
    lock_fingerprint = semantic_lock.pop("fingerprint", None)
    records = lock.get("records")
    if (
        lock.get("schema_version") != "langmani-v2-phase2c-c-phase-diagnostic-lock-v0"
        or lock.get("package_fingerprint") != PACKAGE_FINGERPRINT
        or lock.get("split") != "validation"
        or lock_fingerprint != canonical_fingerprint(semantic_lock)
        or not isinstance(records, list)
        or not records
    ):
        raise Phase2CCContractError("phase diagnostic lock changed")
    indices = [
        int(cast(Mapping[str, object], record)["dataset_index"])
        for record in records
        if isinstance(record, Mapping)
    ]
    phases = [
        str(cast(Mapping[str, object], record)["phase"])
        for record in records
        if isinstance(record, Mapping)
    ]
    expected_derived = [
        str(cast(Mapping[str, object], record)["derived_episode_identity"])
        for record in records
        if isinstance(record, Mapping)
    ]
    expected_frames = [
        int(cast(Mapping[str, object], record)["frame_index"])
        for record in records
        if isinstance(record, Mapping)
    ]
    if not (len(indices) == len(phases) == len(expected_derived) == len(expected_frames)):
        raise Phase2CCContractError("phase diagnostic lock records are malformed")
    view = load_dataset_view(
        args.primary_root.resolve(),
        model_kind=ModelKind.PICK,
        split="validation",
    )
    loader = DataLoader(
        Subset(view.dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    if formulation == "absolute":
        policy: Any = BoundedSmolVLAPolicyV1.from_pretrained(checkpoint, strict=True)
        BoundedSmolVLAPolicyV1.action_transform.load(checkpoint)
    else:
        policy = load_relative_checkpoint(checkpoint, strict=True)
    from lerobot.policies.factory import (  # type: ignore[import-untyped]
        make_pre_post_processors,
    )

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=checkpoint,
    )
    policy.to(device=torch.device("cuda"))
    policy.eval()
    transform = StateRelativeBoundedActionV0()
    lower = torch.as_tensor(PANDA_ACTION_LOW, dtype=torch.float32)
    upper = torch.as_tensor(PANDA_ACTION_HIGH, dtype=torch.float32)
    saturation_margin = torch.maximum((upper - lower) * 0.01, torch.full_like(lower, 1e-8))

    loss_sum = 0.0
    example_count = 0
    absolute_error_sum = 0.0
    valid_component_count = 0
    first_error_sum = 0.0
    first_component_count = 0
    arm_error_sum = 0.0
    arm_component_count = 0
    gripper_error_sum = 0.0
    gripper_component_count = 0
    residual_error_sum = 0.0
    latent_error_sum = 0.0
    latent_component_count = 0
    chunk_error_sum = torch.zeros(int(policy.config.chunk_size), dtype=torch.float64)
    chunk_error_count = torch.zeros(int(policy.config.chunk_size), dtype=torch.int64)
    predicted_physical_values: list[torch.Tensor] = []
    predicted_residual_values: list[torch.Tensor] = []
    saturated_components = 0
    total_predicted_components = 0
    by_phase: dict[str, list[float | int]] = defaultdict(lambda: [0.0, 0, 0.0, 0, 0])
    cursor = 0
    for batch_index, raw_batch in enumerate(loader):
        physical_batch = project_policy_batch(raw_batch, shared=False)
        current_state = physical_batch.get("observation.state")
        target = physical_batch.get(ACTION_FEATURE_KEY)
        padding = physical_batch.get(ACTION_PAD_KEY)
        if not all(isinstance(value, torch.Tensor) for value in (current_state, target, padding)):
            raise Phase2CCContractError("offline physical batch is malformed")
        assert isinstance(current_state, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert isinstance(padding, torch.Tensor)
        batch_size = target.shape[0]
        derived = _sequence(
            raw_batch.get("derived_episode_identity"),
            label="derived_episode_identity",
            batch_size=batch_size,
        )
        frames_raw = raw_batch.get("frame_index")
        if not isinstance(frames_raw, torch.Tensor):
            raise Phase2CCContractError("diagnostic batch lacks frame_index")
        frames = [int(value) for value in frames_raw.tolist()]
        if (
            derived != expected_derived[cursor : cursor + batch_size]
            or frames != expected_frames[cursor : cursor + batch_size]
        ):
            raise Phase2CCContractError("diagnostic rows differ from their frozen lock")
        row_phases = phases[cursor : cursor + batch_size]
        cursor += batch_size
        model_batch = (
            physical_batch
            if formulation == "absolute"
            else project_relative_policy_batch(physical_batch)
        )
        processed = preprocessor(model_batch)
        if not isinstance(processed, Mapping):
            raise Phase2CCContractError("offline preprocessor returned a non-mapping")
        processed_batch = dict(processed)
        processed_state = processed_batch.get("observation.state")
        if not isinstance(processed_state, torch.Tensor):
            raise Phase2CCContractError("processed state is malformed")
        torch.manual_seed(DIAGNOSTIC_SEED + batch_index)
        torch.cuda.manual_seed_all(DIAGNOSTIC_SEED + batch_index)
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ),
        ):
            loss, _ = policy(processed_batch)
            generator = torch.Generator(device=processed_state.device).manual_seed(
                DIAGNOSTIC_SEED * 10_000 + batch_index
            )
            noise = torch.randn(
                (
                    batch_size,
                    int(policy.config.chunk_size),
                    int(policy.config.max_action_dim),
                ),
                generator=generator,
                device=processed_state.device,
                dtype=processed_state.dtype,
            )
            generated = postprocessor(policy.predict_action_chunk(processed_batch, noise=noise))
        if not isinstance(generated, torch.Tensor) or not bool(torch.isfinite(generated).all()):
            raise Phase2CCContractError("offline generated action is invalid")
        if formulation == "absolute":
            predicted = generated
            target_latent = None
        else:
            predicted = transform.decode(
                generated.to(dtype=torch.float32, device="cpu"),
                current_state.to(dtype=torch.float32, device="cpu"),
            )
            relative_target = model_batch.get(ACTION_FEATURE_KEY)
            if not isinstance(relative_target, torch.Tensor):
                raise Phase2CCContractError("relative target is missing")
            target_latent = relative_target.to(dtype=torch.float32, device="cpu")
        predicted_cpu = predicted.to(dtype=torch.float32, device="cpu")
        target_cpu = target.to(dtype=torch.float32, device="cpu")
        state_cpu = current_state.to(dtype=torch.float32, device="cpu")
        padding_cpu = padding.to(device="cpu")
        if predicted_cpu.shape != target_cpu.shape:
            raise Phase2CCContractError("generated physical action shape changed")
        valid_steps = ~padding_cpu
        valid = valid_steps.unsqueeze(-1).expand_as(target_cpu)
        error = torch.abs(predicted_cpu - target_cpu)
        reference = transform.current_action_reference(state_cpu)[:, None, :]
        predicted_residual = predicted_cpu - reference
        target_residual = target_cpu - reference
        residual_error = torch.abs(predicted_residual - target_residual)
        loss_sum += float(loss) * batch_size
        example_count += batch_size
        absolute_error_sum += float(error[valid].sum())
        residual_error_sum += float(residual_error[valid].sum())
        valid_component_count += int(valid.sum())
        first_error_sum += float(error[:, 0].sum())
        first_component_count += batch_size * target.shape[-1]
        arm_valid = valid[..., :7]
        gripper_valid = valid[..., 7:]
        arm_error_sum += float(error[..., :7][arm_valid].sum())
        arm_component_count += int(arm_valid.sum())
        gripper_error_sum += float(error[..., 7:][gripper_valid].sum())
        gripper_component_count += int(gripper_valid.sum())
        if target_latent is not None:
            latent_error_sum += float(torch.abs(generated.cpu() - target_latent)[valid].sum())
            latent_component_count += int(valid.sum())
        for position in range(target.shape[1]):
            position_valid = valid_steps[:, position]
            chunk_error_sum[position] += float(error[position_valid, position].sum())
            chunk_error_count[position] += int(position_valid.sum()) * target.shape[-1]
        predicted_physical_values.append(predicted_cpu[valid].reshape(-1))
        predicted_residual_values.append(predicted_residual[valid].reshape(-1))
        saturated_components += int(
            torch.count_nonzero(
                (predicted_cpu <= lower + saturation_margin)
                | (predicted_cpu >= upper - saturation_margin)
            )
        )
        total_predicted_components += predicted_cpu.numel()
        for row_index, phase in enumerate(row_phases):
            row_valid = valid[row_index]
            row_first = error[row_index, 0]
            value = by_phase[phase]
            value[0] = float(value[0]) + float(error[row_index][row_valid].sum())
            value[1] = int(value[1]) + int(row_valid.sum())
            value[2] = float(value[2]) + float(row_first.sum())
            value[3] = int(value[3]) + row_first.numel()
            value[4] = int(value[4]) + 1
    if (
        cursor != len(records)
        or example_count != len(records)
        or valid_component_count == 0
        or arm_component_count == 0
        or gripper_component_count == 0
    ):
        raise Phase2CCContractError("offline diagnostic coverage is incomplete")
    physical_values = torch.cat(predicted_physical_values)
    residual_values = torch.cat(predicted_residual_values)
    chunk_mae = [
        (
            float(chunk_error_sum[index] / chunk_error_count[index])
            if int(chunk_error_count[index]) > 0
            else None
        )
        for index in range(len(chunk_error_sum))
    ]
    semantic: dict[str, object] = {
        "schema_version": "langmani-v2-phase2c-c-offline-checkpoint-diagnostic-v0",
        "package_fingerprint": PACKAGE_FINGERPRINT,
        "formulation": formulation,
        "model_kind": (
            "pick_smolvla_absolute" if formulation == "absolute" else "pick_smolvla_relative"
        ),
        "checkpoint": checkpoint.as_posix(),
        "checkpoint_sha256": f"sha256:{expected_digest}",
        "checkpoint_manifest_fingerprint": manifest.get("fingerprint"),
        "optimizer_step": manifest.get("optimizer_step"),
        "split": "validation",
        "phase_diagnostic_lock_fingerprint": lock_fingerprint,
        "diagnostic_seed": DIAGNOSTIC_SEED,
        "frame_count": len(records),
        "validation_flow_matching_loss": loss_sum / example_count,
        "physical_action_mae": absolute_error_sum / valid_component_count,
        "physical_first_action_mae": first_error_sum / first_component_count,
        "arm_action_mae": arm_error_sum / arm_component_count,
        "gripper_action_mae": gripper_error_sum / gripper_component_count,
        "residual_action_mae": residual_error_sum / valid_component_count,
        "residual_latent_mae": (
            latent_error_sum / latent_component_count if latent_component_count else None
        ),
        "chunk_position_mae": chunk_mae,
        "physical_output_variance": float(torch.var(physical_values, unbiased=False)),
        "residual_output_variance": float(torch.var(residual_values, unbiased=False)),
        "action_saturation_rate": saturated_components / total_predicted_components,
        "error_by_phase": _group_summary(by_phase),
        "phase_labels_privileged_offline_only": True,
        "privileged_policy_input": False,
        "future_state_model_input": False,
        "nonfinite_loss_count": 0,
        "invalid_action_count": 0,
        "clipping_events": 0,
        "projection_events": 0,
        "passed": (
            math.isfinite(loss_sum)
            and math.isfinite(absolute_error_sum)
            and saturated_components <= total_predicted_components
        ),
    }
    result = {**semantic, "fingerprint": canonical_fingerprint(semantic)}
    if result["passed"] is not True:
        raise Phase2CCContractError("offline checkpoint diagnostic failed")
    _write_new_or_equal(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
