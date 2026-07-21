"""Generic-policy adapter for one frozen historical PerTask ACT checkpoint."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Self, cast

import torch

from langmani.datasets.identity import sha256_hex
from langmani.datasets.lerobot_types import (
    ACTION_FEATURE_KEY,
    IMAGE_FEATURE_KEY,
    STATE_FEATURE_KEY,
)
from langmani.policies.act_types import ACT_ACTION_COMPONENTS, ActExperimentManifest
from langmani.policies.m42_evaluation import M42PolicyKind, load_m4_checkpoint_context
from langmani.v2.policy import (
    ActionChunk,
    ObservationBatch,
    PolicyContext,
    PolicyContractError,
    PolicyIdentity,
)
from langmani.v2.taxonomy import PICK_AND_PLACE_SKILL_ID

ACT_ADAPTER_NAME = "act_per_task_v0"
ACT_ADAPTER_CONFIG_SCHEMA = "langmani-v2-act-adapter-config-v0"
ACT_RUNTIME_MANIFEST_SCHEMA = "langmani-v2-policy-runtime-manifest-v0"


class ActAdapterError(PolicyContractError):
    """Raised when the canonical ACT baseline cannot be loaded or executed."""


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ActAdapterError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ActAdapterError(f"{label} must be a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ActAdapterError(f"cannot hash ACT artifact {path}: {error}") from error
    return digest.hexdigest()


def _safe_repo_relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ActAdapterError(f"{label} must be a non-empty repository-relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "\\" in value:
        raise ActAdapterError(f"{label} must be a safe POSIX repository-relative path")
    return value


@dataclass(frozen=True, slots=True)
class ActAdapterConfig:
    """Repository-controlled locator and exact identity for one accepted ACT policy."""

    policy_id: str
    adapter_name: str
    run_root: str
    selected_checkpoint_relative_path: str
    expected_run_fingerprint: str
    expected_checkpoint_fingerprint: str
    expected_dataset_fingerprint: str
    expected_task_id: str
    execution_horizon: int
    action_bound_mode: str
    device: str
    dtype: str
    artifact_sha256: Mapping[str, str]
    schema_version: str = ACT_ADAPTER_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ACT_ADAPTER_CONFIG_SCHEMA:
            raise ActAdapterError("unknown ACT adapter config schema")
        if self.adapter_name != ACT_ADAPTER_NAME:
            raise ActAdapterError(f"ACT adapter_name must be {ACT_ADAPTER_NAME}")
        _safe_repo_relative_path(self.run_root, "run_root")
        _safe_repo_relative_path(
            self.selected_checkpoint_relative_path,
            "selected_checkpoint_relative_path",
        )
        if self.execution_horizon != 10:
            raise ActAdapterError("canonical v2 ACT baseline preserves execution horizon H=10")
        if self.action_bound_mode != "project":
            raise ActAdapterError("canonical v2 ACT baseline preserves project action handling")
        if self.device not in {"cpu", "cuda"} or self.dtype != "float32":
            raise ActAdapterError("ACT runtime supports cpu/cuda with float32 outputs")
        if not self.policy_id or not self.expected_task_id:
            raise ActAdapterError("policy_id and expected_task_id cannot be empty")
        for value, label in (
            (self.expected_run_fingerprint, "expected_run_fingerprint"),
            (self.expected_checkpoint_fingerprint, "expected_checkpoint_fingerprint"),
            (self.expected_dataset_fingerprint, "expected_dataset_fingerprint"),
        ):
            if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
                raise ActAdapterError(f"{label} must be a prefixed SHA-256 digest")
        required_artifacts = {
            "pretrained_model/model.safetensors",
            "pretrained_model/policy_preprocessor_step_3_normalizer_processor.safetensors",
            "pretrained_model/policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        }
        if set(self.artifact_sha256) != required_artifacts:
            raise ActAdapterError("artifact_sha256 must contain the exact deployable ACT files")
        for relative, digest in self.artifact_sha256.items():
            _safe_repo_relative_path(relative, "artifact path")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ActAdapterError("artifact hashes must be unprefixed SHA-256 digests")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        payload = dict(value)
        hashes = payload.get("artifact_sha256")
        if not isinstance(hashes, Mapping) or not all(
            isinstance(key, str) and isinstance(item, str) for key, item in hashes.items()
        ):
            raise ActAdapterError("artifact_sha256 must be a string mapping")
        payload["artifact_sha256"] = dict(hashes)
        try:
            return cls(**payload)
        except TypeError as error:
            raise ActAdapterError(f"malformed ACT adapter config: {error}") from error

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.from_dict(_read_json_object(path, "ACT adapter config"))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "adapter_name": self.adapter_name,
            "run_root": self.run_root,
            "selected_checkpoint_relative_path": self.selected_checkpoint_relative_path,
            "expected_run_fingerprint": self.expected_run_fingerprint,
            "expected_checkpoint_fingerprint": self.expected_checkpoint_fingerprint,
            "expected_dataset_fingerprint": self.expected_dataset_fingerprint,
            "expected_task_id": self.expected_task_id,
            "execution_horizon": self.execution_horizon,
            "action_bound_mode": self.action_bound_mode,
            "device": self.device,
            "dtype": self.dtype,
            "artifact_sha256": dict(self.artifact_sha256),
        }


class ActPerTaskPolicyAdapter:
    """Expose frozen LeRobot ACT inference through the generic v2 interface."""

    def __init__(
        self,
        *,
        config: ActAdapterConfig,
        project_root: Path,
        config_path: Path,
    ) -> None:
        self.config = config
        self.project_root = project_root.resolve()
        self.config_path = config_path.resolve()
        try:
            self.config_path.relative_to(self.project_root)
        except ValueError as error:
            raise ActAdapterError("ACT config must live inside the repository") from error
        self.run_root = (self.project_root / PurePosixPath(config.run_root)).resolve()
        try:
            self.run_root.relative_to(self.project_root)
        except ValueError as error:
            raise ActAdapterError("ACT run_root escapes the repository") from error
        if not self.run_root.is_dir():
            raise ActAdapterError(f"ACT run_root is missing: {self.run_root}")
        if config.device == "cuda" and not torch.cuda.is_available():
            raise ActAdapterError("ACT config requires CUDA but no CUDA device is visible")

        try:
            context = load_m4_checkpoint_context(
                self.run_root,
                policy_kind=M42PolicyKind.PER_TASK,
                expected_checkpoint_fingerprint=config.expected_checkpoint_fingerprint,
                expected_run_fingerprint=config.expected_run_fingerprint,
                expected_dataset_fingerprint=config.expected_dataset_fingerprint,
                expected_task_id=config.expected_task_id,
            )
        except Exception as error:
            raise ActAdapterError(
                f"strict canonical ACT reload failed: {type(error).__name__}: {error}"
            ) from error
        if context.descriptor.checkpoint_relative_path != config.selected_checkpoint_relative_path:
            raise ActAdapterError("selected checkpoint path differs from repository config")
        self._loaded = context.loaded
        self._policy = cast(Any, context.loaded.policy)
        self._preprocessor = cast(Any, context.loaded.preprocessor)
        self._postprocessor = cast(Any, context.loaded.postprocessor)
        to_device = getattr(self._policy, "to", None)
        if callable(to_device):
            to_device(torch.device(config.device))
        eval_mode = getattr(self._policy, "eval", None)
        if callable(eval_mode):
            eval_mode()
        self._context: PolicyContext | None = None
        self._closed = False
        self._validate_artifact_hashes()
        self._identity = PolicyIdentity(
            policy_id=config.policy_id,
            adapter_name=ACT_ADAPTER_NAME,
            implementation=f"{type(self._policy).__module__}.{type(self._policy).__qualname__}",
            checkpoint_identity=config.expected_checkpoint_fingerprint,
            compatible_skill_families=(PICK_AND_PLACE_SKILL_ID,),
            compatible_task_ids=(config.expected_task_id,),
        )
        manifest = ActExperimentManifest.from_dict(
            _read_json_object(self.run_root / "run_manifest.json", "ACT run manifest")
        )
        config_relative = self.config_path.relative_to(self.project_root).as_posix()
        checkpoint_root = self.run_root / PurePosixPath(config.selected_checkpoint_relative_path)
        self._runtime_manifest: Mapping[str, Any] = {
            "schema_version": ACT_RUNTIME_MANIFEST_SCHEMA,
            "policy_identity": self._identity.to_dict(),
            "runtime_fingerprint": f"sha256:{sha256_hex({'config': config.to_dict(), 'identity': self._identity.to_dict()})}",
            "repository_config": config_relative,
            "checkpoint_path": (checkpoint_root.relative_to(self.project_root).as_posix()),
            "checkpoint_fingerprint": config.expected_checkpoint_fingerprint,
            "artifact_sha256": dict(config.artifact_sha256),
            "policy_class": self._identity.implementation,
            "skill_compatibility": [PICK_AND_PLACE_SKILL_ID],
            "task_compatibility": [config.expected_task_id],
            "observation_keys": [IMAGE_FEATURE_KEY, STATE_FEATURE_KEY],
            "observation_shapes": {
                IMAGE_FEATURE_KEY: [3, 256, 256],
                STATE_FEATURE_KEY: [9],
            },
            "action_key": ACTION_FEATURE_KEY,
            "action_shape": [ACT_ACTION_COMPONENTS],
            "predicted_chunk_size": 50,
            "execution_horizon": config.execution_horizon,
            "normalization": dict(manifest.identity.model_config.normalization_mapping),
            "preprocessing": {
                "pipeline": "LeRobot PolicyProcessorPipeline",
                "config": "pretrained_model/policy_preprocessor.json",
                "state_schema": "PandaPolicyStateV0",
                "image_conversion": "uint8 CHW to float32 [0,1]",
            },
            "postprocessing": {
                "pipeline": "LeRobot PolicyProcessorPipeline",
                "config": "pretrained_model/policy_postprocessor.json",
                "action_bound_mode": config.action_bound_mode,
                "action_projection_owner": "UnifiedPolicyEvaluator",
            },
            "device": config.device,
            "dtype": config.dtype,
            "source_training_configuration": manifest.config.to_dict(),
            "source_training_git_commit": manifest.identity.git_commit,
            "source_dataset_fingerprint": manifest.identity.m3b_export_fingerprint,
            "train_statistics_fingerprint": manifest.identity.train_statistics_fingerprint,
        }

    @classmethod
    def from_config(
        cls,
        *,
        config_path: Path,
        project_root: Path,
    ) -> Self:
        return cls(
            config=ActAdapterConfig.load(config_path),
            project_root=project_root,
            config_path=config_path,
        )

    @property
    def identity(self) -> PolicyIdentity:
        return self._identity

    @property
    def runtime_manifest(self) -> Mapping[str, Any]:
        return dict(self._runtime_manifest)

    def _validate_artifact_hashes(self) -> None:
        checkpoint_root = self.run_root / PurePosixPath(
            self.config.selected_checkpoint_relative_path
        )
        for relative, expected in self.config.artifact_sha256.items():
            artifact = checkpoint_root / PurePosixPath(relative)
            if not artifact.is_file():
                raise ActAdapterError(f"required ACT artifact is missing: {artifact}")
            if _sha256_file(artifact) != expected:
                raise ActAdapterError(f"ACT artifact hash changed: {relative}")

    def reset(self, context: PolicyContext) -> None:
        if self._closed:
            raise ActAdapterError("closed ACT adapter cannot be reset")
        task = context.evaluation_task.task_instance
        if task.skill_family_id not in self.identity.compatible_skill_families:
            raise ActAdapterError("ACT policy is incompatible with the requested skill family")
        if task.canonical_task_id not in self.identity.compatible_task_ids:
            raise ActAdapterError("PerTask ACT policy is incompatible with the requested task")
        for component, label in (
            (self._policy, "policy"),
            (self._preprocessor, "preprocessor"),
            (self._postprocessor, "postprocessor"),
        ):
            reset = getattr(component, "reset", None)
            if not callable(reset):
                raise ActAdapterError(f"ACT {label} does not expose reset()")
            reset()
        self._context = context

    def act(self, observation: ObservationBatch) -> ActionChunk:
        if self._closed or self._context is None:
            raise ActAdapterError("ACT adapter requires reset(context) before inference")
        expected = {IMAGE_FEATURE_KEY, STATE_FEATURE_KEY}
        if set(observation.features) != expected:
            raise ActAdapterError("ACT observation keys differ from the frozen feature contract")
        image = observation.features[IMAGE_FEATURE_KEY]
        state = observation.features[STATE_FEATURE_KEY]
        if tuple(image.shape) != (3, 256, 256) or image.dtype is not torch.float32:
            raise ActAdapterError("ACT image must be float32[3,256,256]")
        if tuple(state.shape) != (9,) or state.dtype is not torch.float32:
            raise ActAdapterError("ACT state must be float32[9]")
        with torch.inference_mode():
            processed = self._preprocessor(dict(observation.features))
            if not isinstance(processed, Mapping):
                raise ActAdapterError("ACT preprocessor must return a tensor mapping")
            chunk = self._policy.predict_action_chunk(dict(processed))
            if not isinstance(chunk, torch.Tensor):
                raise ActAdapterError("ACT predict_action_chunk must return a Torch tensor")
            if tuple(chunk.shape) != (1, 50, ACT_ACTION_COMPONENTS):
                raise ActAdapterError("ACT predicted chunk must have shape [1,50,8]")
            if not chunk.is_floating_point() or not bool(torch.isfinite(chunk).all()):
                raise ActAdapterError("ACT predicted chunk contains malformed actions")
            postprocessed = self._postprocessor(chunk)
            if not isinstance(postprocessed, torch.Tensor):
                raise ActAdapterError("ACT postprocessor must return a Torch tensor")
            if tuple(postprocessed.shape) != (1, 50, ACT_ACTION_COMPONENTS):
                raise ActAdapterError("postprocessed ACT chunk must have shape [1,50,8]")
            if not bool(torch.isfinite(postprocessed).all()):
                raise ActAdapterError("postprocessed ACT chunk contains malformed actions")
            actions = (
                postprocessed[0, : self.config.execution_horizon]
                .to(dtype=torch.float32)
                .detach()
                .clone()
            )
        return ActionChunk(
            actions=actions,
            valid_mask=torch.ones(self.config.execution_horizon, dtype=torch.bool),
            horizon=self.config.execution_horizon,
            metadata={
                "source_chunk_size": 50,
                "execution_horizon": self.config.execution_horizon,
                "task_id": self._context.evaluation_task.task_instance.canonical_task_id,
            },
        )

    def close(self) -> None:
        self._context = None
        self._closed = True
        if self.config.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = [
    "ACT_ADAPTER_CONFIG_SCHEMA",
    "ACT_ADAPTER_NAME",
    "ACT_RUNTIME_MANIFEST_SCHEMA",
    "ActAdapterConfig",
    "ActAdapterError",
    "ActPerTaskPolicyAdapter",
]
