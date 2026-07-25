"""Generic PolicyAdapter for bounded Phase 2C-B SmolVLA checkpoints."""

from __future__ import annotations

import json
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
from langmani.v2.phase2c_a import SKILL_FAMILIES
from langmani.v2.phase2c_b import (
    OFFICIAL_BASE_MODEL,
    OFFICIAL_BASE_REVISION,
    OFFICIAL_LEROBOT_VERSION,
    PACKAGE_FINGERPRINT,
    BoundedActionLatentV1,
    ModelKind,
    Phase2CBContractError,
    canonical_fingerprint,
)
from langmani.v2.phase2c_b_smolvla import (
    BoundedSmolVLAPolicyV1,
)
from langmani.v2.policy import (
    ActionChunk,
    ObservationBatch,
    PolicyContext,
    PolicyIdentity,
)
from langmani.v2.smolvla_adapter import map_smolvla_observation, sha256_directory

ADAPTER_NAME = "smolvla_multiskill_bounded_v1"
CHECKPOINT_MANIFEST = "phase2c_b_checkpoint_manifest.json"
DEFAULT_INSTRUCTIONS = {
    "PickCube-v1": "Pick up the cube and place it on the target.",
    "StackCube-v1": "Stack one cube on top of the other cube.",
    "PushCube-v1": "Push the cube to the goal region.",
}


class Phase2CBSmolVLAAdapterError(Phase2CBContractError):
    """Raised when a deployable SmolVLA checkpoint violates its runtime contract."""


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2CBSmolVLAAdapterError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise Phase2CBSmolVLAAdapterError(f"{path} must contain one JSON object")
    return value


class Phase2CBSmolVLAPolicyAdapter:
    """Inference-only adapter that exposes physical bounded action chunks."""

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
            raise Phase2CBSmolVLAAdapterError("SmolVLA checkpoint directory hash changed")
        if execution_horizon not in {1, 4, 8}:
            raise Phase2CBSmolVLAAdapterError("execution horizon must be 1, 4, or 8")
        if device not in {"cpu", "cuda"} or dtype not in {"float32", "bfloat16"}:
            raise Phase2CBSmolVLAAdapterError("unsupported SmolVLA runtime device/dtype")
        if device == "cuda" and not torch.cuda.is_available():
            raise Phase2CBSmolVLAAdapterError("CUDA SmolVLA runtime has no CUDA device")
        self.manifest = _read_object(self.checkpoint / CHECKPOINT_MANIFEST)
        kind = ModelKind(str(self.manifest.get("model_kind")))
        if (
            self.manifest.get("schema_version") != "langmani-v2-phase2c-b-checkpoint-manifest-v0"
            or self.manifest.get("package_fingerprint") != PACKAGE_FINGERPRINT
            or self.manifest.get("base_model") != OFFICIAL_BASE_MODEL
            or self.manifest.get("base_revision") != OFFICIAL_BASE_REVISION
            or self.manifest.get("action_transform_fingerprint")
            != BoundedActionLatentV1().fingerprint
            or self.manifest.get("task_id_model_input") is not False
        ):
            raise Phase2CBSmolVLAAdapterError("SmolVLA checkpoint manifest changed")
        BoundedActionLatentV1.load(self.checkpoint)
        self.policy = BoundedSmolVLAPolicyV1.from_pretrained(self.checkpoint, strict=True)
        processors = make_pre_post_processors(
            self.policy.config,
            pretrained_path=self.checkpoint,
        )
        self.preprocessor = processors[0]
        self.postprocessor = processors[1]
        self.policy.to(device=torch.device(device))
        self.policy.eval()
        for component in (self.policy, self.preprocessor, self.postprocessor):
            reset = getattr(component, "reset", None)
            if not callable(reset):
                raise Phase2CBSmolVLAAdapterError("SmolVLA component lacks reset()")
        self.model_kind = kind
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
            policy_id=f"phase2c-b:{kind.value}",
            adapter_name=ADAPTER_NAME,
            implementation=(f"{type(self.policy).__module__}.{type(self.policy).__qualname__}"),
            checkpoint_identity=f"sha256:{digest}",
            compatible_skill_families=tuple(
                SKILL_FAMILIES[task_id] for task_id in kind.compatible_task_ids
            ),
            compatible_task_ids=kind.compatible_task_ids,
        )
        static = {
            "schema_version": "langmani-v2-phase2c-b-runtime-manifest-v0",
            "policy_identity": self._identity.to_dict(),
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "model_kind": kind.value,
            "base_model": OFFICIAL_BASE_MODEL,
            "base_revision": OFFICIAL_BASE_REVISION,
            "lerobot_version": OFFICIAL_LEROBOT_VERSION,
            "checkpoint_sha256": f"sha256:{digest}",
            "action_transform_fingerprint": BoundedActionLatentV1().fingerprint,
            "action_transform": "structural_tanh_then_native_affine",
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
        self._runtime_static = {
            **static,
            "fingerprint": canonical_fingerprint(static),
        }

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
            raise Phase2CBSmolVLAAdapterError("closed SmolVLA adapter cannot reset")
        task_id = context.canonical_task_id
        if task_id not in self.identity.compatible_task_ids:
            raise Phase2CBSmolVLAAdapterError("policy is incompatible with requested task")
        for component in (self.policy, self.preprocessor, self.postprocessor):
            cast(Any, component).reset()
        self._context = context
        self._query_count = 0
        self._actions_emitted = 0
        self._invalid_output_count = 0
        self._latencies_ms = []

    def act(self, observation: ObservationBatch) -> ActionChunk:
        if self._closed or self._context is None:
            raise Phase2CBSmolVLAAdapterError("SmolVLA adapter requires reset() before act()")
        instruction = (
            self._context.language_instruction
            if self._context.language_instruction is not None
            else DEFAULT_INSTRUCTIONS[self._context.canonical_task_id]
        )
        mapped = map_smolvla_observation(observation, task=instruction)
        started = time.perf_counter()
        try:
            inference_autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if self.device == "cuda" and self.dtype == "bfloat16"
                else nullcontext()
            )
            with torch.inference_mode(), inference_autocast:
                processed = self.preprocessor(mapped)
                if not isinstance(processed, Mapping):
                    raise Phase2CBSmolVLAAdapterError("preprocessor returned a non-mapping")
                batch = dict(processed)
                state = batch.get("observation.state")
                if not isinstance(state, torch.Tensor) or state.ndim != 2:
                    raise Phase2CBSmolVLAAdapterError("processed state is malformed")
                generator = torch.Generator(device=state.device).manual_seed(
                    self.inference_seed + self._query_count
                )
                noise = torch.randn(
                    (
                        state.shape[0],
                        int(self.policy.config.chunk_size),
                        int(self.policy.config.max_action_dim),
                    ),
                    generator=generator,
                    device=state.device,
                    dtype=state.dtype,
                )
                predicted = self.policy.predict_action_chunk(batch, noise=noise)
                restored = self.postprocessor(predicted)
                if (
                    not isinstance(restored, torch.Tensor)
                    or restored.ndim != 3
                    or tuple(restored.shape[:1]) != (1,)
                    or restored.shape[1] < self.execution_horizon
                    or restored.shape[2] != 8
                    or not restored.is_floating_point()
                    or not bool(torch.isfinite(restored).all())
                ):
                    raise Phase2CBSmolVLAAdapterError(
                        "SmolVLA generated an invalid physical action chunk"
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
                    raise Phase2CBSmolVLAAdapterError(
                        "structurally bounded SmolVLA action escaped native bounds"
                    )
                if not bool(torch.any(torch.abs(actions - actions[0]) > 0)):
                    constant_within_chunk = True
                else:
                    constant_within_chunk = False
        except Exception:
            self._invalid_output_count += 1
            raise
        finally:
            if self.device == "cuda":
                torch.cuda.synchronize()
            self._latencies_ms.append((time.perf_counter() - started) * 1_000.0)
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
                "task_id": self._context.canonical_task_id,
                "instruction_sha256": canonical_fingerprint({"instruction": instruction}),
                "constant_within_chunk": constant_within_chunk,
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
    "DEFAULT_INSTRUCTIONS",
    "Phase2CBSmolVLAAdapterError",
    "Phase2CBSmolVLAPolicyAdapter",
]
