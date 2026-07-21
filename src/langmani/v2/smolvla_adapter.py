"""Official LeRobot 0.6 SmolVLA adapter for the generic LangMani policy API."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Self, cast

import torch

from langmani.datasets.identity import sha256_hex
from langmani.datasets.lerobot_types import ACTION_FEATURE_KEY
from langmani.v2.phase2c import map_smolvla_observation
from langmani.v2.policy import (
    ActionChunk,
    ObservationBatch,
    PolicyContext,
    PolicyContractError,
    PolicyIdentity,
    finite_latency_ms,
)
from langmani.v2.taxonomy import PUSH_TO_REGION_SKILL_ID, canonical_push_to_region_tasks

SMOLVLA_ADAPTER_NAME = "smolvla_push_v0"
SMOLVLA_ADAPTER_CONFIG_SCHEMA = "langmani-v2-smolvla-adapter-config-v0"
SMOLVLA_RUNTIME_MANIFEST_SCHEMA = "langmani-v2-smolvla-runtime-manifest-v0"
OFFICIAL_BASE_MODEL = "lerobot/smolvla_base"
SUPPORTED_LEROBOT_VERSION = "0.6.0"


class SmolVLAAdapterError(PolicyContractError):
    """Raised when official SmolVLA loading or inference violates Phase 2C."""


def sha256_directory(path: Path) -> str:
    """Hash all checkpoint files by stable relative path and exact bytes."""

    if not path.is_dir():
        raise SmolVLAAdapterError(f"checkpoint directory does not exist: {path}")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise SmolVLAAdapterError("checkpoint directory is empty")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SmolVLAAdapterConfig:
    """Content-bound runtime configuration for one fine-tuned push checkpoint."""

    policy_id: str
    checkpoint_path: str
    checkpoint_sha256: str
    base_model: str
    base_model_revision: str
    training_config_sha256: str
    train_view_sha256: str
    normalization_sha256: str
    processor_sha256: str
    training_seed: int
    execution_horizon: int
    device: str
    dtype: str
    inference_seed: int
    adapter_name: str = SMOLVLA_ADAPTER_NAME
    action_bound_mode: str = "reject"
    schema_version: str = SMOLVLA_ADAPTER_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SMOLVLA_ADAPTER_CONFIG_SCHEMA:
            raise SmolVLAAdapterError("unknown SmolVLA adapter config schema")
        if self.adapter_name != SMOLVLA_ADAPTER_NAME:
            raise SmolVLAAdapterError(f"adapter_name must be {SMOLVLA_ADAPTER_NAME}")
        if not self.policy_id or not self.checkpoint_path:
            raise SmolVLAAdapterError("policy_id and checkpoint_path cannot be empty")
        if self.base_model != OFFICIAL_BASE_MODEL:
            raise SmolVLAAdapterError(f"Phase 2C requires {OFFICIAL_BASE_MODEL}")
        if len(self.base_model_revision) not in range(7, 65):
            raise SmolVLAAdapterError("base model revision must be pinned, not floating")
        for value, label in (
            (self.checkpoint_sha256, "checkpoint_sha256"),
            (self.training_config_sha256, "training_config_sha256"),
            (self.train_view_sha256, "train_view_sha256"),
            (self.normalization_sha256, "normalization_sha256"),
            (self.processor_sha256, "processor_sha256"),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise SmolVLAAdapterError(f"{label} must be a lowercase SHA-256 digest")
        if self.execution_horizon not in {1, 4, 8}:
            raise SmolVLAAdapterError(
                "execution_horizon must be one of the frozen candidates 1,4,8"
            )
        if isinstance(self.training_seed, bool) or self.training_seed < 0:
            raise SmolVLAAdapterError("training_seed must be a non-negative integer")
        if self.device not in {"cpu", "cuda"}:
            raise SmolVLAAdapterError("SmolVLA device must be cpu or cuda")
        if self.dtype not in {"float32", "bfloat16"}:
            raise SmolVLAAdapterError("SmolVLA dtype must be float32 or bfloat16")
        if isinstance(self.inference_seed, bool) or self.inference_seed < 0:
            raise SmolVLAAdapterError("inference_seed must be a non-negative integer")
        if self.action_bound_mode != "reject":
            raise SmolVLAAdapterError("Phase 2C policy actions must be rejected, never projected")

    @classmethod
    def load(cls, path: Path) -> Self:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise SmolVLAAdapterError(f"cannot read SmolVLA adapter config: {error}") from error
        if not isinstance(value, dict):
            raise SmolVLAAdapterError("SmolVLA adapter config must be a JSON object")
        try:
            return cls(**value)
        except TypeError as error:
            raise SmolVLAAdapterError(f"malformed SmolVLA adapter config: {error}") from error

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "adapter_name": self.adapter_name,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "base_model": self.base_model,
            "base_model_revision": self.base_model_revision,
            "training_config_sha256": self.training_config_sha256,
            "train_view_sha256": self.train_view_sha256,
            "normalization_sha256": self.normalization_sha256,
            "processor_sha256": self.processor_sha256,
            "training_seed": self.training_seed,
            "execution_horizon": self.execution_horizon,
            "action_bound_mode": self.action_bound_mode,
            "device": self.device,
            "dtype": self.dtype,
            "inference_seed": self.inference_seed,
        }

    def semantic_dict(self) -> dict[str, object]:
        """Return the portable runtime identity, excluding the local checkpoint path."""

        value = self.to_dict()
        value.pop("checkpoint_path")
        return value


class SmolVLAPolicyAdapter:
    """Inference-only wrapper around the installed official SmolVLAPolicy."""

    def __init__(
        self,
        *,
        config: SmolVLAAdapterConfig,
        policy: object | None = None,
        preprocessor: object | None = None,
        postprocessor: object | None = None,
        verify_checkpoint: bool = True,
    ) -> None:
        self.config = config
        checkpoint = Path(config.checkpoint_path).expanduser().resolve()
        if verify_checkpoint and sha256_directory(checkpoint) != config.checkpoint_sha256:
            raise SmolVLAAdapterError("fine-tuned checkpoint content hash changed")
        self._load_seconds = 0.0
        if policy is None or preprocessor is None or postprocessor is None:
            started = time.perf_counter()
            policy, preprocessor, postprocessor = self._load_official_components(checkpoint)
            self._load_seconds = time.perf_counter() - started
        self._policy = cast(Any, policy)
        self._preprocessor = cast(Any, preprocessor)
        self._postprocessor = cast(Any, postprocessor)
        for component, label in (
            (self._policy, "policy"),
            (self._preprocessor, "preprocessor"),
            (self._postprocessor, "postprocessor"),
        ):
            if component is None:
                raise SmolVLAAdapterError(f"official SmolVLA {label} is missing")
        to_device = getattr(self._policy, "to", None)
        if callable(to_device):
            target_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
            to_device(device=torch.device(config.device), dtype=target_dtype)
        eval_mode = getattr(self._policy, "eval", None)
        if callable(eval_mode):
            eval_mode()
        self._model_config = getattr(self._policy, "config", None)
        self._chunk_size = int(getattr(self._model_config, "chunk_size", 50))
        self._n_action_steps = int(getattr(self._model_config, "n_action_steps", 50))
        if self._chunk_size < config.execution_horizon:
            raise SmolVLAAdapterError("execution horizon exceeds model chunk size")
        self._context: PolicyContext | None = None
        self._closed = False
        self._query_count = 0
        self._actions_emitted = 0
        self._queue_clear_count = 0
        self._queue_underrun_count = 0
        self._invalid_output_count = 0
        self._latencies_ms: list[float] = []
        compatible_tasks = tuple(
            item.canonical_task_id for item in canonical_push_to_region_tasks()
        )
        self._identity = PolicyIdentity(
            policy_id=config.policy_id,
            adapter_name=SMOLVLA_ADAPTER_NAME,
            implementation=f"{type(self._policy).__module__}.{type(self._policy).__qualname__}",
            checkpoint_identity=f"sha256:{config.checkpoint_sha256}",
            compatible_skill_families=(PUSH_TO_REGION_SKILL_ID,),
            compatible_task_ids=compatible_tasks,
        )
        parameter_count, trainable_parameter_count = self._parameter_counts()
        parameter_dtypes = self._parameter_dtypes()
        if parameter_dtypes and parameter_dtypes != [config.dtype]:
            raise SmolVLAAdapterError(
                "loaded SmolVLA parameter dtype differs from the adapter configuration: "
                + ",".join(parameter_dtypes)
            )
        identity_payload = {
            "adapter_config": config.semantic_dict(),
            "policy_identity": self._identity.to_dict(),
            "model_chunk_size": self._chunk_size,
            "n_action_steps": self._n_action_steps,
        }
        self._runtime_static: dict[str, Any] = {
            "schema_version": SMOLVLA_RUNTIME_MANIFEST_SCHEMA,
            "runtime_fingerprint": f"sha256:{sha256_hex(identity_payload)}",
            "policy_identity": self._identity.to_dict(),
            "official_implementation": "LeRobot SmolVLAPolicy",
            "lerobot_version": SUPPORTED_LEROBOT_VERSION,
            "base_model": config.base_model,
            "base_model_revision": config.base_model_revision,
            "fine_tuned_checkpoint_sha256": config.checkpoint_sha256,
            "training_config_sha256": config.training_config_sha256,
            "train_view_sha256": config.train_view_sha256,
            "normalization_sha256": config.normalization_sha256,
            "processor_sha256": config.processor_sha256,
            "training_seed": config.training_seed,
            "observation_schema": {
                "observation.images.base_camera": [3, 256, 256],
                "observation.state": [9],
                "task": "non-empty string",
            },
            "action_schema": {ACTION_FEATURE_KEY: [8]},
            "model_chunk_size": self._chunk_size,
            "official_n_action_steps": self._n_action_steps,
            "execution_horizon": config.execution_horizon,
            "action_bound_mode": config.action_bound_mode,
            "queue_owner": "LangMani ActionChunk true-prefix; official select_action queue bypassed",
            "stochastic_generation": (
                "flow-matching noise is deterministically seeded per episode query"
            ),
            "device": config.device,
            "dtype": config.dtype,
            "observed_parameter_dtypes": parameter_dtypes,
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "load_seconds": self._load_seconds,
        }

    @classmethod
    def from_config(
        cls,
        *,
        config_path: Path,
        project_root: Path,
    ) -> Self:
        """Load the official adapter from one content-bound external config."""

        config = SmolVLAAdapterConfig.load(config_path)
        checkpoint = Path(config.checkpoint_path)
        if not checkpoint.is_absolute():
            config = replace(config, checkpoint_path=str(project_root / checkpoint))
        return cls(config=config)

    def _load_official_components(self, checkpoint: Path) -> tuple[object, object, object]:
        try:
            import lerobot  # type: ignore[import-untyped]
            from lerobot.policies.factory import (  # type: ignore[import-untyped]
                make_pre_post_processors,
            )
            from lerobot.policies.smolvla.modeling_smolvla import (  # type: ignore[import-untyped]
                SmolVLAPolicy,
            )
        except ImportError as error:
            raise SmolVLAAdapterError(
                "LeRobot 0.6 with the official SmolVLA dependencies is required"
            ) from error
        if getattr(lerobot, "__version__", None) != SUPPORTED_LEROBOT_VERSION:
            raise SmolVLAAdapterError(
                f"installed LeRobot must be exactly {SUPPORTED_LEROBOT_VERSION}"
            )
        try:
            policy = SmolVLAPolicy.from_pretrained(str(checkpoint))
            processors = make_pre_post_processors(
                policy.config,
                pretrained_path=str(checkpoint),
            )
        except Exception as error:
            raise SmolVLAAdapterError(
                f"official SmolVLA checkpoint load failed: {type(error).__name__}: {error}"
            ) from error
        return policy, processors[0], processors[1]

    def _parameter_counts(self) -> tuple[int, int]:
        parameters = getattr(self._policy, "parameters", None)
        if not callable(parameters):
            return 0, 0
        values = list(parameters())
        return (
            sum(int(item.numel()) for item in values),
            sum(int(item.numel()) for item in values if bool(item.requires_grad)),
        )

    def _parameter_dtypes(self) -> list[str]:
        parameters = getattr(self._policy, "parameters", None)
        if not callable(parameters):
            return []
        return sorted(
            {
                str(item.dtype).removeprefix("torch.")
                for item in parameters()
                if bool(item.is_floating_point())
            }
        )

    @property
    def identity(self) -> PolicyIdentity:
        return self._identity

    @property
    def runtime_manifest(self) -> Mapping[str, Any]:
        return {
            **self._runtime_static,
            "policy_query_count": self._query_count,
            "actions_emitted": self._actions_emitted,
            "queue_clear_count": self._queue_clear_count,
            "queue_underrun_count": self._queue_underrun_count,
            "invalid_output_count": self._invalid_output_count,
            "inference_latency_ms": _latency_summary(self._latencies_ms),
        }

    def reset(self, context: PolicyContext) -> None:
        if self._closed:
            raise SmolVLAAdapterError("closed SmolVLA adapter cannot be reset")
        task = context.evaluation_task.task_instance
        if task.skill_family_id != PUSH_TO_REGION_SKILL_ID:
            raise SmolVLAAdapterError("Phase 2C SmolVLA is push-only")
        if task.canonical_task_id not in self.identity.compatible_task_ids:
            raise SmolVLAAdapterError("SmolVLA policy received an unknown push task")
        for component, label in (
            (self._policy, "policy"),
            (self._preprocessor, "preprocessor"),
            (self._postprocessor, "postprocessor"),
        ):
            reset = getattr(component, "reset", None)
            if not callable(reset):
                raise SmolVLAAdapterError(f"official SmolVLA {label} does not expose reset()")
            reset()
        self._context = context
        self._query_count = 0
        self._actions_emitted = 0
        self._queue_underrun_count = 0
        self._invalid_output_count = 0
        self._latencies_ms = []
        self._queue_clear_count += 1

    def act(self, observation: ObservationBatch) -> ActionChunk:
        if self._closed or self._context is None:
            raise SmolVLAAdapterError("SmolVLA adapter requires reset(context) before inference")
        instruction = (
            self._context.language_instruction
            or self._context.evaluation_task.task_instance.language_instruction
        )
        mapped = map_smolvla_observation(observation, task=instruction)
        cuda_devices = [torch.cuda.current_device()] if self.config.device == "cuda" else []
        started = time.perf_counter()
        try:
            with torch.inference_mode(), torch.random.fork_rng(devices=cuda_devices):
                seed = self.config.inference_seed + self._query_count
                torch.manual_seed(seed)
                if self.config.device == "cuda":
                    torch.cuda.manual_seed_all(seed)
                processed = self._preprocessor(mapped)
                if not isinstance(processed, Mapping):
                    raise SmolVLAAdapterError("SmolVLA preprocessor returned a non-mapping")
                predicted = self._policy.predict_action_chunk(dict(processed))
                if not isinstance(predicted, torch.Tensor):
                    raise SmolVLAAdapterError("SmolVLA predict_action_chunk returned a non-tensor")
                if predicted.ndim != 3 or predicted.shape[0] != 1 or predicted.shape[2] != 8:
                    raise SmolVLAAdapterError("SmolVLA predicted chunk must have shape [1,chunk,8]")
                if predicted.shape[1] < self.config.execution_horizon:
                    self._queue_underrun_count += 1
                    raise SmolVLAAdapterError("SmolVLA predicted fewer actions than requested")
                if not predicted.is_floating_point() or not bool(torch.isfinite(predicted).all()):
                    self._invalid_output_count += 1
                    raise SmolVLAAdapterError(
                        "SmolVLA predicted non-finite or non-floating actions"
                    )
                restored = self._postprocessor(predicted)
                if not isinstance(restored, torch.Tensor):
                    raise SmolVLAAdapterError("SmolVLA postprocessor returned a non-tensor")
                if tuple(restored.shape) != tuple(predicted.shape):
                    raise SmolVLAAdapterError("SmolVLA postprocessor changed the chunk shape")
                if not bool(torch.isfinite(restored).all()):
                    self._invalid_output_count += 1
                    raise SmolVLAAdapterError("postprocessed SmolVLA actions are non-finite")
                actions = (
                    restored[0, : self.config.execution_horizon]
                    .to(dtype=torch.float32, device="cpu")
                    .detach()
                    .clone()
                )
        finally:
            if self.config.device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            self._latencies_ms.append(finite_latency_ms(started, time.perf_counter()))
        self._query_count += 1
        self._actions_emitted += self.config.execution_horizon
        return ActionChunk(
            actions=actions,
            valid_mask=torch.ones(self.config.execution_horizon, dtype=torch.bool),
            horizon=self.config.execution_horizon,
            metadata={
                "source_chunk_size": int(predicted.shape[1]),
                "execution_horizon": self.config.execution_horizon,
                "policy_query_ordinal": self._query_count - 1,
                "inference_seed": self.config.inference_seed + self._query_count - 1,
                "task_id": self._context.evaluation_task.task_instance.canonical_task_id,
            },
        )

    def close(self) -> None:
        self._context = None
        self._closed = True
        if self.config.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _latency_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "maximum": None}
    ordered = sorted(values)

    def item(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
        return ordered[index]

    return {
        "mean": sum(ordered) / len(ordered),
        "p50": item(0.50),
        "p95": item(0.95),
        "maximum": ordered[-1],
    }


__all__ = [
    "OFFICIAL_BASE_MODEL",
    "SMOLVLA_ADAPTER_CONFIG_SCHEMA",
    "SMOLVLA_ADAPTER_NAME",
    "SMOLVLA_RUNTIME_MANIFEST_SCHEMA",
    "SUPPORTED_LEROBOT_VERSION",
    "SmolVLAAdapterConfig",
    "SmolVLAAdapterError",
    "SmolVLAPolicyAdapter",
    "sha256_directory",
]
