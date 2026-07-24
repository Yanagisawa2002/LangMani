"""Generic PolicyAdapter implementation for trained Phase 2C-A ACT policies."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from lerobot.policies.act import ACTPolicy
from lerobot.processor import DataProcessorPipeline, PolicyProcessorPipeline

from langmani.datasets.identity import sha256_hex
from langmani.datasets.lerobot_types import ACTION_FEATURE_KEY
from langmani.policies.act_checkpoint import (
    CheckpointComponentFingerprints,
    load_act_checkpoint,
)
from langmani.v2.phase2c_a import (
    ACTION_DIMENSION,
    CHUNK_SIZE,
    EXECUTION_HORIZONS,
    PACKAGE_FINGERPRINT,
    SKILL_FAMILIES,
    TASK_IDS,
    ModelKind,
    Phase2CARunIdentity,
)
from langmani.v2.phase2c_a_act import (
    TASK_INDEX_KEY,
    prepare_policy_batch,
    validate_shared_act_config,
)
from langmani.v2.policy import (
    ActionChunk,
    ObservationBatch,
    PolicyContext,
    PolicyIdentity,
)


class Phase2CAAdapterError(RuntimeError):
    """Raised when a trained ACT cannot satisfy the generic policy contract."""


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2CAAdapterError(f"could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise Phase2CAAdapterError(f"{path} must contain one object")
    return value


def _reset_component(component: object, label: str) -> None:
    reset = getattr(component, "reset", None)
    if not callable(reset):
        raise Phase2CAAdapterError(f"{label} does not expose reset()")
    reset()


def _postprocess_chunk(
    postprocessor: PolicyProcessorPipeline,
    predicted: torch.Tensor,
) -> torch.Tensor:
    result = postprocessor(predicted)
    if isinstance(result, Mapping):
        result = result.get(ACTION_FEATURE_KEY)
    if not isinstance(result, torch.Tensor):
        raise Phase2CAAdapterError("ACT postprocessor did not return an action tensor")
    tensor = result.detach()
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.dtype is not torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    if tuple(tensor.shape) != (CHUNK_SIZE, ACTION_DIMENSION):
        raise Phase2CAAdapterError(
            f"ACT action chunk must be [{CHUNK_SIZE},{ACTION_DIMENSION}], got {tuple(tensor.shape)}"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise Phase2CAAdapterError("ACT action chunk contains nonfinite values")
    return tensor


class Phase2CAActPolicyAdapter:
    """Load, preprocess, infer, and emit validated ACT action chunks."""

    def __init__(
        self,
        *,
        model_kind: ModelKind | str,
        policy: ACTPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        checkpoint_identity: str,
        run_fingerprint: str,
        execution_horizon: int,
        component_fingerprints: CheckpointComponentFingerprints | None = None,
        device: str = "cuda",
    ) -> None:
        self.model_kind = ModelKind(model_kind)
        if execution_horizon not in EXECUTION_HORIZONS:
            raise Phase2CAAdapterError("execution horizon must be one of 1/4/8")
        if device not in {"cpu", "cuda"}:
            raise Phase2CAAdapterError("adapter device must be cpu or cuda")
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.execution_horizon = execution_horizon
        self.device = torch.device(device)
        self.policy.to(self.device)
        self.policy.eval()
        if self.model_kind is ModelKind.SHARED:
            validate_shared_act_config(self.policy.config, self.policy)
        self._checkpoint_identity = checkpoint_identity
        self._run_fingerprint = run_fingerprint
        self._component_fingerprints = component_fingerprints
        self._active_task_id: str | None = None
        self._evaluation_id: str | None = None
        self._query_count = 0
        self._reset_count = 0
        self._latencies_ms: list[float] = []
        compatible = self.model_kind.compatible_task_ids
        self._identity = PolicyIdentity(
            policy_id=f"phase2c-a:{self.model_kind.value}:{run_fingerprint}",
            adapter_name="phase2c_a_act",
            implementation="lerobot-0.6.0-maintained-act",
            checkpoint_identity=checkpoint_identity,
            compatible_skill_families=tuple(SKILL_FAMILIES[task_id] for task_id in compatible),
            compatible_task_ids=compatible,
        )

    @classmethod
    def from_checkpoint(
        cls,
        *,
        run_root: str | Path,
        checkpoint_relative_path: str,
        execution_horizon: int,
        device: str = "cuda",
    ) -> Phase2CAActPolicyAdapter:
        root = Path(run_root).resolve()
        run_manifest = _read_object(root / "run_manifest.json")
        identity_value = run_manifest.get("identity")
        if not isinstance(identity_value, Mapping):
            raise Phase2CAAdapterError("run manifest lacks its semantic identity")
        identity = Phase2CARunIdentity.from_dict(identity_value)
        loaded = load_act_checkpoint(
            run_root=root,
            checkpoint_relative_path=checkpoint_relative_path,
            expected_identity=identity,
            for_resume=False,
        )
        if not isinstance(loaded.policy, ACTPolicy):
            raise Phase2CAAdapterError("checkpoint did not reload an ACTPolicy")
        if not isinstance(loaded.preprocessor, DataProcessorPipeline) or not isinstance(
            loaded.postprocessor, DataProcessorPipeline
        ):
            raise Phase2CAAdapterError("checkpoint did not reload both policy processors")
        return cls(
            model_kind=identity.model_kind,
            policy=loaded.policy,
            preprocessor=loaded.preprocessor,
            postprocessor=loaded.postprocessor,
            checkpoint_identity=loaded.record.checkpoint_fingerprint,
            run_fingerprint=identity.run_fingerprint,
            execution_horizon=execution_horizon,
            component_fingerprints=loaded.component_fingerprints,
            device=device,
        )

    @property
    def identity(self) -> PolicyIdentity:
        return self._identity

    @property
    def runtime_manifest(self) -> Mapping[str, Any]:
        components = self._component_fingerprints
        semantic = {
            "schema_version": "langmani-v2-phase2c-a-act-runtime-v0",
            "package_fingerprint": PACKAGE_FINGERPRINT,
            "model_kind": self.model_kind.value,
            "policy_identity": self.identity.to_dict(),
            "run_fingerprint": self._run_fingerprint,
            "checkpoint_identity": self._checkpoint_identity,
            "execution_horizon": self.execution_horizon,
            "predicted_chunk_size": CHUNK_SIZE,
            "task_conditioning": (
                "three_way_onehot_public_env_token"
                if self.model_kind is ModelKind.SHARED
                else "none"
            ),
            "action_handling": "hard_reject_nonfinite_malformed_or_out_of_bounds",
            "action_projection": False,
            "action_clipping": False,
            "device": str(self.device),
            "processor_components": (
                {
                    "model": components.model,
                    "preprocessor": components.preprocessor,
                    "postprocessor": components.postprocessor,
                }
                if components is not None
                else None
            ),
        }
        return {**semantic, "fingerprint": f"sha256:{sha256_hex(semantic)}"}

    @property
    def query_count(self) -> int:
        return self._query_count

    @property
    def reset_count(self) -> int:
        return self._reset_count

    @property
    def inference_latencies_ms(self) -> tuple[float, ...]:
        return tuple(self._latencies_ms)

    def reset(self, context: PolicyContext) -> None:
        task_id = context.canonical_task_id
        if task_id not in self.identity.compatible_task_ids:
            raise Phase2CAAdapterError(
                f"{self.model_kind.value} is incompatible with task {task_id}"
            )
        _reset_component(self.policy, "ACT policy")
        _reset_component(self.preprocessor, "ACT preprocessor")
        _reset_component(self.postprocessor, "ACT postprocessor")
        self._active_task_id = task_id
        self._evaluation_id = context.evaluation_id
        self._query_count = 0
        self._latencies_ms.clear()
        self._reset_count += 1

    @torch.no_grad()
    def act(self, observation: ObservationBatch) -> ActionChunk:
        if self._active_task_id is None or self._evaluation_id is None:
            raise Phase2CAAdapterError("adapter must be reset before inference")
        raw: dict[str, object] = dict(observation.features)
        if self.model_kind is ModelKind.SHARED:
            raw[TASK_INDEX_KEY] = torch.tensor(
                TASK_IDS.index(self._active_task_id), dtype=torch.int64
            )
        projected = prepare_policy_batch(
            raw,
            model_kind=self.model_kind,
            include_action=False,
        )
        processed = self.preprocessor(projected)
        if not isinstance(processed, Mapping):
            raise Phase2CAAdapterError("ACT preprocessor did not return a mapping")
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        predicted = self.policy.predict_action_chunk(cast(dict[str, torch.Tensor], dict(processed)))
        actions = _postprocess_chunk(self.postprocessor, predicted)
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000.0
        if not math_is_finite_nonnegative(latency_ms):
            raise Phase2CAAdapterError("inference latency is invalid")
        self._latencies_ms.append(latency_ms)
        self._query_count += 1
        valid_mask = torch.zeros(CHUNK_SIZE, dtype=torch.bool, device=actions.device)
        valid_mask[: self.execution_horizon] = True
        return ActionChunk(
            actions=actions,
            valid_mask=valid_mask,
            horizon=CHUNK_SIZE,
            metadata={
                "evaluation_id": self._evaluation_id,
                "task_id": self._active_task_id,
                "policy_query_index": self._query_count,
                "predicted_chunk_size": CHUNK_SIZE,
                "execution_horizon": self.execution_horizon,
                "inference_latency_ms": latency_ms,
            },
        )

    def close(self) -> None:
        self._active_task_id = None
        self._evaluation_id = None
        self._latencies_ms.clear()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def math_is_finite_nonnegative(value: float) -> bool:
    return bool(np.isfinite(value) and value >= 0.0)


__all__ = [
    "Phase2CAActPolicyAdapter",
    "Phase2CAAdapterError",
]
