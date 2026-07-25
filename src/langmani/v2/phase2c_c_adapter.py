"""Inference adapter for the sole Phase 2C-C relative Pick SmolVLA."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from lerobot.policies.factory import (  # type: ignore[import-untyped]
    make_pre_post_processors,
)

from langmani.v2.phase2b5 import PANDA_ACTION_HIGH, PANDA_ACTION_LOW
from langmani.v2.phase2c_a import PACKAGE_FINGERPRINT, SKILL_FAMILIES
from langmani.v2.phase2c_b import (
    OFFICIAL_BASE_MODEL,
    OFFICIAL_BASE_REVISION,
    OFFICIAL_LEROBOT_VERSION,
)
from langmani.v2.phase2c_b_adapter import DEFAULT_INSTRUCTIONS
from langmani.v2.phase2c_c import (
    Phase2CCContractError,
    StateRelativeBoundedActionV0,
    canonical_fingerprint,
)
from langmani.v2.phase2c_c_smolvla import load_relative_checkpoint
from langmani.v2.policy import ActionChunk, ObservationBatch, PolicyContext, PolicyIdentity
from langmani.v2.smolvla_adapter import map_smolvla_observation, sha256_directory

ADAPTER_NAME = "smolvla_pick_state_relative_bounded_v0"
CHECKPOINT_MANIFEST = "phase2c_c_checkpoint_manifest.json"


class Phase2CCSmolVLAAdapterError(Phase2CCContractError):
    """Raised when a relative checkpoint violates its runtime contract."""


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2CCSmolVLAAdapterError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise Phase2CCSmolVLAAdapterError(f"{path} must contain one JSON object")
    return value


class Phase2CCRelativeSmolVLAAdapter:
    """Expose query-state-relative predictions as native physical chunks."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        expected_checkpoint_sha256: str,
        execution_horizon: int,
        device: str = "cuda",
        dtype: str = "bfloat16",
        inference_seed: int = 314_159,
    ) -> None:
        self.checkpoint = Path(checkpoint).resolve()
        digest = expected_checkpoint_sha256.removeprefix("sha256:")
        if sha256_directory(self.checkpoint) != digest:
            raise Phase2CCSmolVLAAdapterError("relative checkpoint directory hash changed")
        if execution_horizon not in {1, 8}:
            raise Phase2CCSmolVLAAdapterError("relative execution horizon must be 1 or 8")
        if device not in {"cpu", "cuda"} or dtype not in {"float32", "bfloat16"}:
            raise Phase2CCSmolVLAAdapterError("unsupported SmolVLA runtime device/dtype")
        if device == "cuda" and not torch.cuda.is_available():
            raise Phase2CCSmolVLAAdapterError("CUDA SmolVLA runtime has no CUDA device")
        self.manifest = _read_object(self.checkpoint / CHECKPOINT_MANIFEST)
        self.transform = StateRelativeBoundedActionV0.load(self.checkpoint)
        if (
            self.manifest.get("schema_version") != "langmani-v2-phase2c-c-checkpoint-manifest-v0"
            or self.manifest.get("package_fingerprint") != PACKAGE_FINGERPRINT
            or self.manifest.get("model_kind") != "pick_smolvla_relative"
            or self.manifest.get("task_id") != "PickCube-v1"
            or self.manifest.get("base_model") != OFFICIAL_BASE_MODEL
            or self.manifest.get("base_revision") != OFFICIAL_BASE_REVISION
            or self.manifest.get("action_transform_fingerprint") != self.transform.fingerprint
            or self.manifest.get("task_id_model_input") is not False
            or self.manifest.get("future_state_model_input") is not False
            or re.fullmatch(
                r"[0-9a-f]{40}",
                str(self.manifest.get("training_git_commit")),
            )
            is None
            or any(
                re.fullmatch(r"sha256:[0-9a-f]{64}", str(self.manifest.get(key))) is None
                for key in (
                    "static_preparation_fingerprint",
                    "base_audit_fingerprint",
                    "real_view_fingerprint",
                    "run_manifest_fingerprint",
                )
            )
        ):
            raise Phase2CCSmolVLAAdapterError("relative checkpoint manifest changed")
        self.policy = load_relative_checkpoint(self.checkpoint, strict=True)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=self.checkpoint,
        )
        self.policy.to(device=torch.device(device))
        self.policy.eval()
        for component in (self.policy, self.preprocessor, self.postprocessor):
            if not callable(getattr(component, "reset", None)):
                raise Phase2CCSmolVLAAdapterError("SmolVLA component lacks reset()")
        self.execution_horizon = execution_horizon
        self.device = device
        self.dtype = dtype
        self.inference_seed = inference_seed
        self._context: PolicyContext | None = None
        self._closed = False
        self._query_count = 0
        self._actions_emitted = 0
        self._invalid_output_count = 0
        self._latencies_ms: list[float] = []
        self._identity = PolicyIdentity(
            policy_id="phase2c-c:pick_smolvla_relative",
            adapter_name=ADAPTER_NAME,
            implementation=f"{type(self.policy).__module__}.{type(self.policy).__qualname__}",
            checkpoint_identity=f"sha256:{digest}",
            compatible_skill_families=(SKILL_FAMILIES["PickCube-v1"],),
            compatible_task_ids=("PickCube-v1",),
        )
        static: dict[str, object] = {
            "schema_version": "langmani-v2-phase2c-c-relative-runtime-manifest-v0",
            "policy_identity": self._identity.to_dict(),
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "model_kind": "pick_smolvla_relative",
            "base_model": OFFICIAL_BASE_MODEL,
            "base_revision": OFFICIAL_BASE_REVISION,
            "lerobot_version": OFFICIAL_LEROBOT_VERSION,
            "checkpoint_sha256": f"sha256:{digest}",
            "training_git_commit": self.manifest["training_git_commit"],
            "static_preparation_fingerprint": self.manifest["static_preparation_fingerprint"],
            "base_audit_fingerprint": self.manifest["base_audit_fingerprint"],
            "real_view_fingerprint": self.manifest["real_view_fingerprint"],
            "run_manifest_fingerprint": self.manifest["run_manifest_fingerprint"],
            "action_transform_fingerprint": self.transform.fingerprint,
            "action_transform": "query_state_relative_safe_logit_composition",
            "chunk_anchor": "current_observation_state_at_policy_query",
            "future_state_model_input": False,
            "post_hoc_clipping": False,
            "projection": False,
            "replacement_action": False,
            "task_id_model_input": False,
            "language_input": "task",
            "chunk_size": int(self.policy.config.chunk_size),
            "execution_horizon": execution_horizon,
            "device": device,
            "dtype": dtype,
            "inference_seed": inference_seed,
        }
        self._runtime_static = {**static, "fingerprint": canonical_fingerprint(static)}

    @property
    def identity(self) -> PolicyIdentity:
        return self._identity

    @property
    def runtime_manifest(self) -> Mapping[str, Any]:
        return {
            **self._runtime_static,
            "policy_query_count": self._query_count,
            "actions_emitted": self._actions_emitted,
            "invalid_output_count": self._invalid_output_count,
            "inference_latency_ms": _latency_summary(self._latencies_ms),
        }

    def reset(self, context: PolicyContext) -> None:
        if self._closed:
            raise Phase2CCSmolVLAAdapterError("closed relative adapter cannot reset")
        if context.canonical_task_id != "PickCube-v1":
            raise Phase2CCSmolVLAAdapterError("relative policy is Pick-only")
        for component in (self.policy, self.preprocessor, self.postprocessor):
            cast(Any, component).reset()
        self._context = context
        self._query_count = 0
        self._actions_emitted = 0
        self._invalid_output_count = 0
        self._latencies_ms = []

    def act(self, observation: ObservationBatch) -> ActionChunk:
        if self._closed or self._context is None:
            raise Phase2CCSmolVLAAdapterError("relative adapter requires reset() before act()")
        instruction = (
            self._context.language_instruction
            if self._context.language_instruction is not None
            else DEFAULT_INSTRUCTIONS["PickCube-v1"]
        )
        mapped = map_smolvla_observation(observation, task=instruction)
        raw_state = mapped.get("observation.state")
        if not isinstance(raw_state, torch.Tensor) or raw_state.shape != (9,):
            raise Phase2CCSmolVLAAdapterError("mapped current public state is malformed")
        started = time.perf_counter()
        restored: torch.Tensor | None = None
        try:
            inference_autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if self.device == "cuda" and self.dtype == "bfloat16"
                else nullcontext()
            )
            with torch.inference_mode(), inference_autocast:
                processed = self.preprocessor(mapped)
                if not isinstance(processed, Mapping):
                    raise Phase2CCSmolVLAAdapterError("preprocessor returned a non-mapping")
                batch = dict(processed)
                processed_state = batch.get("observation.state")
                if not isinstance(processed_state, torch.Tensor) or processed_state.ndim != 2:
                    raise Phase2CCSmolVLAAdapterError("processed state is malformed")
                generator = torch.Generator(device=processed_state.device).manual_seed(
                    self.inference_seed + self._query_count
                )
                noise = torch.randn(
                    (
                        processed_state.shape[0],
                        int(self.policy.config.chunk_size),
                        int(self.policy.config.max_action_dim),
                    ),
                    generator=generator,
                    device=processed_state.device,
                    dtype=processed_state.dtype,
                )
                predicted_latent = self.policy.predict_action_chunk(batch, noise=noise)
                restored_latent = self.postprocessor(predicted_latent)
                if (
                    not isinstance(restored_latent, torch.Tensor)
                    or restored_latent.ndim != 3
                    or restored_latent.shape[0] != 1
                    or restored_latent.shape[1] < self.execution_horizon
                    or restored_latent.shape[2] != 8
                    or not bool(torch.isfinite(restored_latent).all())
                ):
                    raise Phase2CCSmolVLAAdapterError(
                        "SmolVLA generated an invalid relative latent chunk"
                    )
                current_state = raw_state.reshape(1, 9).to(
                    device=restored_latent.device,
                    dtype=torch.float32,
                )
                restored = self.transform.decode(
                    restored_latent.to(dtype=torch.float32),
                    current_state,
                )
                actions = (
                    restored[0, : self.execution_horizon]
                    .to(dtype=torch.float32, device="cpu")
                    .detach()
                    .clone()
                )
                lower = torch.as_tensor(PANDA_ACTION_LOW, dtype=torch.float32)
                upper = torch.as_tensor(PANDA_ACTION_HIGH, dtype=torch.float32)
                if bool(torch.any(actions < lower)) or bool(torch.any(actions > upper)):
                    raise Phase2CCSmolVLAAdapterError(
                        "structurally bounded relative action escaped native bounds"
                    )
                constant_within_chunk = not bool(torch.any(torch.abs(actions - actions[0]) > 0))
        except Exception:
            self._invalid_output_count += 1
            raise
        finally:
            if self.device == "cuda":
                torch.cuda.synchronize()
            self._latencies_ms.append((time.perf_counter() - started) * 1_000.0)
        if restored is None:
            raise Phase2CCSmolVLAAdapterError("relative inference produced no action chunk")
        query_ordinal = self._query_count
        self._query_count += 1
        self._actions_emitted += self.execution_horizon
        return ActionChunk(
            actions=actions,
            valid_mask=torch.ones(self.execution_horizon, dtype=torch.bool),
            horizon=self.execution_horizon,
            metadata={
                "source_chunk_size": int(restored.shape[1]),
                "execution_horizon": self.execution_horizon,
                "policy_query_ordinal": query_ordinal,
                "inference_seed": self.inference_seed + query_ordinal,
                "task_id": "PickCube-v1",
                "instruction_sha256": canonical_fingerprint({"instruction": instruction}),
                "constant_within_chunk": constant_within_chunk,
                "chunk_anchor": "current_observation_state_at_policy_query",
            },
        )

    def close(self) -> None:
        self._context = None
        self._closed = True
        if self.device == "cuda":
            torch.cuda.empty_cache()


def _latency_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "maximum": None}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
        return ordered[index]

    return {
        "mean": float(np.mean(ordered)),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "maximum": ordered[-1],
    }


__all__ = [
    "ADAPTER_NAME",
    "CHECKPOINT_MANIFEST",
    "Phase2CCRelativeSmolVLAAdapter",
    "Phase2CCSmolVLAAdapterError",
]
