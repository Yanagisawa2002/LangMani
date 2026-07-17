"""Immutable six-controller registry for the M5A modular language runtime.

The registry identity is deliberately independent of filesystem locations.  Real
run roots live in :class:`ControllerRegistryLocators`, while every semantic input
that can change controller behavior is bound into the registry fingerprint.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, cast

from langmani.datasets.identity import canonical_json, sha256_hex
from langmani.datasets.lerobot_writer import COMPLETION_MARKER
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import TaskSpec, stable_task_id
from langmani.policies.act_action_bounds import (
    ActionBoundConfig,
    ActionBoundMode,
    BoundedActionEnvPostprocessorV0,
    EvaluationRuntimeManifest,
)
from langmani.policies.act_checkpoint import (
    CheckpointComponentFingerprints,
    ValidatedActCheckpointArtifacts,
    validate_act_checkpoint_artifacts,
)
from langmani.policies.act_data import COMPLETION_SCHEMA
from langmani.policies.act_evaluation import CheckpointSelectionRecord
from langmani.policies.act_types import (
    ACT_ACTION_COMPONENTS,
    ACT_STATE_COMPONENTS,
    ActEvaluationConfig,
    ActExperimentManifest,
    ActModelConfig,
    ActVariant,
    ExperimentMode,
)
from langmani.policies.m42_evidence import (
    M42_M41_RUNTIME_PROCESSOR_SCHEMA,
    FrozenM4Checkpoint,
)
from langmani.policies.m42_training import RuntimeSelectionContract, load_runtime_selection
from langmani.policies.m42_types import ExecutionHorizonConfig, M42ExperimentManifest

CONTROLLER_ENTRY_SCHEMA_VERSION = "langmani-m5a-controller-entry-v0"
CONTROLLER_REGISTRY_SCHEMA_VERSION = "langmani-m5a-controller-registry-v0"
CONTROLLER_LOCATORS_SCHEMA_VERSION = "langmani-m5a-controller-locators-v0"
LOCKED_EXECUTION_HORIZON = 10
LOCKED_GRIPPER_MODE = "project"
LOCKED_ENVIRONMENT_ID = "LangMani-PickPlaceByInstruction-v0"
LOCKED_CONTROL_MODE = "pd_joint_pos"
LOCKED_SIM_BACKEND = "physx_cpu"
LOCKED_OBSERVATION_MODE = "rgb"
LOCKED_REWARD_MODE = "none"
LOCKED_RENDER_MODE = "rgb_array"
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_RE = re.compile(r"^[0-9a-f]{40,64}$")
_RUN_DIRECTORY_RE = re.compile(r"^[0-9a-f]{64}$")


class ControllerRegistryError(RuntimeError):
    """Raised when frozen controller evidence is incomplete or inconsistent."""


def _fingerprint(value: object) -> str:
    return f"sha256:{sha256_hex(value)}"


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ControllerRegistryError(f"{label} must be sha256:<64 lowercase hex>")
    return value


def _require_git(value: object, label: str) -> str:
    if not isinstance(value, str) or _GIT_RE.fullmatch(value) is None:
        raise ControllerRegistryError(f"{label} must be a full lowercase Git object ID")
    return value


def _json_copy(value: object, label: str) -> object:
    try:
        return json.loads(canonical_json(value))
    except (TypeError, ValueError) as error:
        raise ControllerRegistryError(f"{label} must be finite JSON data") from error


def _json_mapping(value: Mapping[str, object], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ControllerRegistryError(f"{label} must be a string-keyed mapping")
    copied = _json_copy(dict(value), label)
    if not isinstance(copied, dict):
        raise ControllerRegistryError(f"{label} must be a JSON object")
    return cast(Mapping[str, object], _freeze_json(copied))


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_json(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    return value


def _safe_relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ControllerRegistryError(f"{label} must be a portable relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ControllerRegistryError(f"{label} must be a canonical relative path")
    return value


@dataclass(frozen=True, slots=True)
class ControllerRegistryEntry:
    """Path-independent semantic identity for one frozen PerTask ACT controller."""

    task_spec: TaskSpec
    task_id: str
    run_fingerprint: str
    checkpoint_fingerprint: str
    checkpoint_step: int
    selection_fingerprint: str
    model_component_fingerprint: str
    preprocessor_fingerprint: str
    postprocessor_fingerprint: str
    train_statistics_fingerprint: str
    producer_git_commit: str
    m3b_export_fingerprint: str
    m3b_split_manifest_digest: str
    model_config: Mapping[str, object]
    data_contract: Mapping[str, object]
    source_data_contract_fingerprint: str
    controller_runtime_fingerprint: str
    environment_id: str
    action_space_contract: Mapping[str, object]
    rollout_config: Mapping[str, object]
    action_bound_config: ActionBoundConfig
    m41_runtime_processor_fingerprint: str
    execution_horizon: int
    execution_horizon_fingerprint: str
    gripper_mode: str
    action_runtime_fingerprint: str
    runtime_selection_fingerprint: str
    runtime_implementation_fingerprint: str
    runtime_evaluation_git_commit: str
    horizon_selection_fingerprint: str
    gripper_selection_fingerprint: str
    runtime_experiment_manifest_fingerprint: str
    schema_version: str = CONTROLLER_ENTRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.task_spec, TaskSpec):
            raise ControllerRegistryError("controller task_spec must be a TaskSpec")
        if self.task_id != stable_task_id(self.task_spec):
            raise ControllerRegistryError("controller task_id disagrees with its TaskSpec")
        for name in (
            "run_fingerprint",
            "checkpoint_fingerprint",
            "selection_fingerprint",
            "model_component_fingerprint",
            "preprocessor_fingerprint",
            "postprocessor_fingerprint",
            "train_statistics_fingerprint",
            "m3b_export_fingerprint",
            "m3b_split_manifest_digest",
            "source_data_contract_fingerprint",
            "controller_runtime_fingerprint",
            "m41_runtime_processor_fingerprint",
            "execution_horizon_fingerprint",
            "action_runtime_fingerprint",
            "runtime_selection_fingerprint",
            "runtime_implementation_fingerprint",
            "horizon_selection_fingerprint",
            "gripper_selection_fingerprint",
            "runtime_experiment_manifest_fingerprint",
        ):
            _require_digest(getattr(self, name), name)
        _require_git(self.producer_git_commit, "producer_git_commit")
        _require_git(self.runtime_evaluation_git_commit, "runtime_evaluation_git_commit")
        if isinstance(self.checkpoint_step, bool) or not isinstance(self.checkpoint_step, int):
            raise ControllerRegistryError("checkpoint_step must be an integer")
        if self.checkpoint_step < 1:
            raise ControllerRegistryError("checkpoint_step must be positive")
        if self.schema_version != CONTROLLER_ENTRY_SCHEMA_VERSION:
            raise ControllerRegistryError("unsupported controller-entry schema")
        model_config = _json_mapping(self.model_config, "model_config")
        model = ActModelConfig.from_dict(model_config)
        if (
            model.state_dimension != ACT_STATE_COMPONENTS
            or model.action_dimension != ACT_ACTION_COMPONENTS
            or model.image_shape_chw != (3, 256, 256)
            or model.chunk_size != 50
        ):
            raise ControllerRegistryError("controller model violates the frozen M4 contracts")
        object.__setattr__(self, "model_config", model_config)
        data_contract = _json_mapping(self.data_contract, "data_contract")
        if set(data_contract) != {
            "video_backend",
            "download_videos",
            "return_uint8",
            "fps",
            "m3b_feature_contract",
            "policy_state_schema",
        }:
            raise ControllerRegistryError(
                "controller data contract must contain only deployable I/O metadata"
            )
        object.__setattr__(self, "data_contract", data_contract)
        object.__setattr__(
            self,
            "action_space_contract",
            _json_mapping(self.action_space_contract, "action_space_contract"),
        )
        object.__setattr__(
            self,
            "rollout_config",
            _json_mapping(self.rollout_config, "rollout_config"),
        )
        if not isinstance(self.action_bound_config, ActionBoundConfig):
            raise ControllerRegistryError("action_bound_config must be an ActionBoundConfig")
        if self.action_bound_config.mode is not ActionBoundMode.PROJECT:
            raise ControllerRegistryError("M5A requires the locked project action-bound runtime")
        horizon = ExecutionHorizonConfig(self.execution_horizon)
        if self.execution_horizon != LOCKED_EXECUTION_HORIZON:
            raise ControllerRegistryError("M5A requires the locked H=10 execution horizon")
        if horizon.fingerprint != self.execution_horizon_fingerprint:
            raise ControllerRegistryError("execution-horizon fingerprint is inconsistent")
        if self.gripper_mode != LOCKED_GRIPPER_MODE:
            raise ControllerRegistryError("M5A requires the locked project gripper runtime")
        if not isinstance(self.environment_id, str) or not self.environment_id:
            raise ControllerRegistryError("environment_id must be non-empty")

    @property
    def model_config_fingerprint(self) -> str:
        return _fingerprint(dict(self.model_config))

    @property
    def data_contract_fingerprint(self) -> str:
        return _fingerprint(dict(self.data_contract))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "task_spec": self.task_spec.to_dict(),
            "task_id": self.task_id,
            "run_fingerprint": self.run_fingerprint,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "checkpoint_step": self.checkpoint_step,
            "selection_fingerprint": self.selection_fingerprint,
            "model_component_fingerprint": self.model_component_fingerprint,
            "preprocessor_fingerprint": self.preprocessor_fingerprint,
            "postprocessor_fingerprint": self.postprocessor_fingerprint,
            "train_statistics_fingerprint": self.train_statistics_fingerprint,
            "producer_git_commit": self.producer_git_commit,
            "m3b_export_fingerprint": self.m3b_export_fingerprint,
            "m3b_split_manifest_digest": self.m3b_split_manifest_digest,
            "model_config": _json_copy(dict(self.model_config), "model_config"),
            "model_config_fingerprint": self.model_config_fingerprint,
            "data_contract": _json_copy(dict(self.data_contract), "data_contract"),
            "data_contract_fingerprint": self.data_contract_fingerprint,
            "source_data_contract_fingerprint": self.source_data_contract_fingerprint,
            "controller_runtime_fingerprint": self.controller_runtime_fingerprint,
            "environment_id": self.environment_id,
            "action_space_contract": _json_copy(
                dict(self.action_space_contract), "action_space_contract"
            ),
            "rollout_config": _json_copy(dict(self.rollout_config), "rollout_config"),
            "action_bound_config": self.action_bound_config.to_dict(),
            "m41_runtime_processor_fingerprint": self.m41_runtime_processor_fingerprint,
            "execution_horizon": self.execution_horizon,
            "execution_horizon_fingerprint": self.execution_horizon_fingerprint,
            "gripper_mode": self.gripper_mode,
            "action_runtime_fingerprint": self.action_runtime_fingerprint,
            "runtime_selection_fingerprint": self.runtime_selection_fingerprint,
            "runtime_implementation_fingerprint": self.runtime_implementation_fingerprint,
            "runtime_evaluation_git_commit": self.runtime_evaluation_git_commit,
            "horizon_selection_fingerprint": self.horizon_selection_fingerprint,
            "gripper_selection_fingerprint": self.gripper_selection_fingerprint,
            "runtime_experiment_manifest_fingerprint": (
                self.runtime_experiment_manifest_fingerprint
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ControllerRegistryEntry:
        payload = dict(value)
        declared_model_fingerprint = payload.pop("model_config_fingerprint", None)
        declared_data_fingerprint = payload.pop("data_contract_fingerprint", None)
        task_spec = payload.get("task_spec")
        if not isinstance(task_spec, Mapping):
            raise ControllerRegistryError("controller entry task_spec must be a mapping")
        payload["task_spec"] = TaskSpec.from_mapping(task_spec)
        payload["action_bound_config"] = ActionBoundConfig.from_dict(
            payload.get("action_bound_config")
        )
        try:
            result = cls(**payload)
        except (TypeError, ValueError) as error:
            raise ControllerRegistryError("malformed controller-registry entry") from error
        if (
            declared_model_fingerprint != result.model_config_fingerprint
            or declared_data_fingerprint != result.data_contract_fingerprint
        ):
            raise ControllerRegistryError("controller contract fingerprint is inconsistent")
        return result


@dataclass(frozen=True, slots=True)
class ControllerRegistry:
    """Exactly six immutable PerTask entries in canonical TaskSpec order."""

    entries: tuple[ControllerRegistryEntry, ...]
    schema_version: str = CONTROLLER_REGISTRY_SCHEMA_VERSION
    registry_fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != CONTROLLER_REGISTRY_SCHEMA_VERSION:
            raise ControllerRegistryError("unsupported controller-registry schema")
        canonical_ids = tuple(stable_task_id(spec) for spec in CANONICAL_TASK_SPECS)
        if len(self.entries) != len(canonical_ids):
            raise ControllerRegistryError("controller registry must contain exactly six entries")
        if tuple(entry.task_id for entry in self.entries) != canonical_ids:
            raise ControllerRegistryError("controller registry must preserve canonical task order")
        if tuple(entry.task_spec for entry in self.entries) != tuple(CANONICAL_TASK_SPECS):
            raise ControllerRegistryError(
                "controller registry TaskSpecs are incomplete or reordered"
            )
        if len({entry.run_fingerprint for entry in self.entries}) != len(self.entries):
            raise ControllerRegistryError("each controller must use one distinct PerTask run")
        if len({entry.checkpoint_fingerprint for entry in self.entries}) != len(self.entries):
            raise ControllerRegistryError("each controller must use one distinct checkpoint")
        common_fields = (
            "m3b_export_fingerprint",
            "m3b_split_manifest_digest",
            "m41_runtime_processor_fingerprint",
            "execution_horizon_fingerprint",
            "runtime_selection_fingerprint",
            "action_runtime_fingerprint",
            "runtime_implementation_fingerprint",
            "runtime_evaluation_git_commit",
            "horizon_selection_fingerprint",
            "gripper_selection_fingerprint",
            "runtime_experiment_manifest_fingerprint",
            "environment_id",
        )
        for name in common_fields:
            if len({getattr(entry, name) for entry in self.entries}) != 1:
                raise ControllerRegistryError(f"controller entries disagree on {name}")
        expected = self.compute_fingerprint()
        if not self.registry_fingerprint:
            object.__setattr__(self, "registry_fingerprint", expected)
        elif self.registry_fingerprint != expected:
            raise ControllerRegistryError("registry fingerprint disagrees with semantic contents")

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "canonical_task_ids": [entry.task_id for entry in self.entries],
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def compute_fingerprint(self) -> str:
        return _fingerprint(self.identity_payload())

    @property
    def environment_contract_fingerprint(self) -> str:
        """Bind every task's environment-facing action/state/image contract."""

        return _fingerprint(
            [
                {
                    "task_id": entry.task_id,
                    "environment_id": entry.environment_id,
                    "action_space_contract": dict(entry.action_space_contract),
                    "data_contract": dict(entry.data_contract),
                    "model_config": dict(entry.model_config),
                }
                for entry in self.entries
            ]
        )

    @property
    def rollout_config_fingerprint(self) -> str:
        """Bind every task's rollout configuration without locator paths."""

        return _fingerprint(
            [
                {
                    "task_id": entry.task_id,
                    "rollout_config": dict(entry.rollout_config),
                }
                for entry in self.entries
            ]
        )

    def to_dict(self) -> dict[str, object]:
        return {**self.identity_payload(), "registry_fingerprint": self.registry_fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ControllerRegistry:
        payload = dict(value)
        if (
            not isinstance(payload.get("registry_fingerprint"), str)
            or not payload["registry_fingerprint"]
        ):
            raise ControllerRegistryError("serialized registry requires its immutable fingerprint")
        canonical_task_ids = payload.pop("canonical_task_ids", None)
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list):
            raise ControllerRegistryError("registry entries must be a list")
        entries = tuple(
            ControllerRegistryEntry.from_dict(cast(Mapping[str, object], item))
            for item in raw_entries
            if isinstance(item, Mapping)
        )
        if len(entries) != len(raw_entries):
            raise ControllerRegistryError("registry entries must be JSON objects")
        payload["entries"] = entries
        result = cls(**payload)
        if canonical_task_ids != [entry.task_id for entry in result.entries]:
            raise ControllerRegistryError("serialized canonical task order is inconsistent")
        return result

    def require(self, task_id: str) -> ControllerRegistryEntry:
        for entry in self.entries:
            if entry.task_id == task_id:
                return entry
        raise ControllerRegistryError(f"unsupported controller task ID: {task_id!r}")


@dataclass(frozen=True, slots=True)
class ControllerLocation:
    """Non-semantic local locator for one registry entry."""

    task_id: str
    run_root: Path
    checkpoint_relative_path: str

    def __post_init__(self) -> None:
        raw_root = Path(self.run_root).expanduser()
        if ".." in raw_root.parts:
            raise ControllerRegistryError("controller run root must not contain parent traversal")
        root = raw_root.absolute()
        if root.is_symlink() or root.is_junction():
            raise ControllerRegistryError("controller run root must not be a link")
        object.__setattr__(self, "run_root", root)
        _safe_relative_path(self.checkpoint_relative_path, "checkpoint_relative_path")


@dataclass(frozen=True, slots=True)
class ControllerRegistryLocators:
    """Machine-local paths bound to, but excluded from, registry identity."""

    registry_fingerprint: str
    locations: tuple[ControllerLocation, ...]
    schema_version: str = CONTROLLER_LOCATORS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_digest(self.registry_fingerprint, "registry_fingerprint")
        if self.schema_version != CONTROLLER_LOCATORS_SCHEMA_VERSION:
            raise ControllerRegistryError("unsupported controller-locators schema")
        canonical_ids = tuple(stable_task_id(spec) for spec in CANONICAL_TASK_SPECS)
        if tuple(location.task_id for location in self.locations) != canonical_ids:
            raise ControllerRegistryError(
                "controller locators must preserve all six canonical tasks"
            )

    def require(self, task_id: str) -> ControllerLocation:
        for location in self.locations:
            if location.task_id == task_id:
                return location
        raise ControllerRegistryError(f"missing controller location for {task_id!r}")


CheckpointArtifactValidator = Callable[..., ValidatedActCheckpointArtifacts]


def _lexical_absolute(path: Path) -> Path:
    expanded = path.expanduser()
    if ".." in expanded.parts:
        raise ControllerRegistryError(f"controller evidence path contains traversal: {path}")
    return expanded if expanded.is_absolute() else Path.cwd() / expanded


def _resolved_unlinked(path: Path, *, label: str, directory: bool) -> Path:
    lexical = _lexical_absolute(path)
    for component in (lexical, *lexical.parents):
        if component.is_symlink() or component.is_junction():
            raise ControllerRegistryError(f"{label} traverses a symlink or junction: {component}")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise ControllerRegistryError(f"missing or inaccessible {label}: {lexical}") from error
    if resolved != lexical:
        raise ControllerRegistryError(f"{label} does not resolve to its lexical path")
    if directory and not resolved.is_dir():
        raise ControllerRegistryError(f"{label} must be a directory")
    if not directory and not resolved.is_file():
        raise ControllerRegistryError(f"{label} must be a file")
    return resolved


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    source = _resolved_unlinked(path, label=label, directory=False)
    if any(
        part.casefold().replace("-", "_") in {"test", "fresh_seed", "m42_final_v0"}
        for part in source.parts
    ) or source.name.casefold() in {"comparison.json", "verification.json"}:
        raise ControllerRegistryError(f"{label} is in a prohibited test/fresh/final path")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ControllerRegistryError(f"cannot read {label}: {source}") from error
    if not isinstance(value, dict):
        raise ControllerRegistryError(f"{label} must contain one JSON object")
    return value


def inspect_controller_action_space_contract(environment: object) -> Mapping[str, object]:
    """Read the active M1 action bounds without stepping or reading task content."""

    processor = BoundedActionEnvPostprocessorV0.from_environment(
        environment,
        ActionBoundConfig(mode=ActionBoundMode.PROJECT),
        expected_action_components=ACT_ACTION_COMPONENTS,
    )
    return _json_mapping(processor.action_space_contract(), "action_space_contract")


def validate_active_controller_environment(
    environment: object,
    registry: ControllerRegistry,
) -> None:
    """Require the active M1 environment to reproduce the frozen validation bounds."""

    active = dict(inspect_controller_action_space_contract(environment))
    expected = dict(registry.entries[0].action_space_contract)
    if active != expected:
        raise ControllerRegistryError(
            "active M1 action-space contract differs from the frozen PerTask runtime"
        )


def locked_controller_rollout_config(
    evaluation: ActEvaluationConfig | None = None,
) -> Mapping[str, object]:
    """Project only the M5A-relevant rollout fields out of the historical config."""

    config = evaluation or ActEvaluationConfig()
    return _json_mapping(
        {
            "maximum_episode_steps": config.maximum_episode_steps,
            "control_frequency_hz": config.control_frequency_hz,
            "invalid_action_policy": config.invalid_action_policy,
            "control_mode": LOCKED_CONTROL_MODE,
            "sim_backend": LOCKED_SIM_BACKEND,
            "observation_mode": LOCKED_OBSERVATION_MODE,
            "reward_mode": LOCKED_REWARD_MODE,
            "render_mode": LOCKED_RENDER_MODE,
            "execution_horizon": LOCKED_EXECUTION_HORIZON,
        },
        "rollout_config",
    )


def _controller_data_contract(source: Mapping[str, object]) -> Mapping[str, object]:
    """Project deployable I/O metadata without test/fresh schedule material."""

    allowed = (
        "video_backend",
        "download_videos",
        "return_uint8",
        "fps",
        "m3b_feature_contract",
        "policy_state_schema",
    )
    missing = [name for name in allowed if name not in source]
    if missing:
        raise ControllerRegistryError(
            "M4 data contract lacks controller I/O fields: " + ",".join(missing)
        )
    return _json_mapping(
        {name: source[name] for name in allowed},
        "controller_data_contract",
    )


def _dataset_fingerprint(dataset_root: Path) -> str:
    completion = _read_json_object(
        _resolved_unlinked(dataset_root, label="M3B dataset root", directory=True)
        / COMPLETION_MARKER,
        label="M3B completion marker",
    )
    if (
        set(completion) != {"schema_version", "export_fingerprint"}
        or completion.get("schema_version") != COMPLETION_SCHEMA
    ):
        raise ControllerRegistryError("M3B completion marker fields are invalid")
    return _require_digest(completion.get("export_fingerprint"), "M3B export fingerprint")


def _discover_per_task_checkpoints(checkpoint_root: Path) -> tuple[FrozenM4Checkpoint, ...]:
    root = _resolved_unlinked(checkpoint_root, label="M4 checkpoint root", directory=True)
    found: dict[str, FrozenM4Checkpoint] = {}
    for child in sorted(root.iterdir(), key=lambda candidate: candidate.name):
        if _RUN_DIRECTORY_RE.fullmatch(child.name) is None:
            continue
        if child.is_symlink() or child.is_junction() or not child.is_dir():
            raise ControllerRegistryError(f"unsafe fingerprint-owned M4 run: {child}")
        manifest_path = child / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = ActExperimentManifest.from_dict(
                _read_json_object(manifest_path, label="M4 run manifest")
            )
        except (TypeError, ValueError) as error:
            raise ControllerRegistryError("invalid M4 run manifest") from error
        if manifest.identity.variant is not ActVariant.PER_TASK:
            continue
        if (
            manifest.identity.experiment_mode is not ExperimentMode.FULL
            or manifest.git_dirty
            or not manifest.complete
            or not manifest.training_state.completed
        ):
            continue
        task_id = manifest.identity.task_id
        canonical_ids = {stable_task_id(spec) for spec in CANONICAL_TASK_SPECS}
        if task_id not in canonical_ids:
            raise ControllerRegistryError("completed PerTask run has a noncanonical task ID")
        if task_id in found:
            raise ControllerRegistryError(f"duplicate completed PerTask run for {task_id}")
        try:
            selection = CheckpointSelectionRecord.from_dict(
                _read_json_object(
                    child / "checkpoint_selection.json",
                    label="M4 validation-only checkpoint selection",
                )
            )
        except (TypeError, ValueError) as error:
            raise ControllerRegistryError("invalid M4 validation-only selection") from error
        if (
            selection.run_fingerprint != manifest.identity.run_fingerprint
            or not selection.selection_locked
            or selection.test_evaluation_status != "pending"
            or manifest.selected_checkpoint_fingerprint != selection.selected_checkpoint_fingerprint
        ):
            raise ControllerRegistryError("M4 checkpoint was not locked by validation only")
        selected = next(
            (
                item
                for item in manifest.checkpoints
                if item.checkpoint_fingerprint == selection.selected_checkpoint_fingerprint
            ),
            None,
        )
        if selected is None or not selected.complete:
            raise ControllerRegistryError("selected M4 checkpoint is absent or incomplete")
        checkpoint_path = child.joinpath(*PurePosixPath(selected.relative_path).parts)
        _resolved_unlinked(checkpoint_path, label="selected M4 checkpoint", directory=True)
        found[cast(str, task_id)] = FrozenM4Checkpoint(
            variant=ActVariant.PER_TASK,
            task_id=task_id,
            run_root=child,
            manifest=manifest,
            selected_checkpoint=selected,
            selection_fingerprint=selection.selection_fingerprint,
        )
    ordered_ids = tuple(stable_task_id(spec) for spec in CANONICAL_TASK_SPECS)
    if set(found) != set(ordered_ids):
        missing = sorted(set(ordered_ids) - set(found))
        raise ControllerRegistryError(
            "M5A requires exactly six completed PerTask controllers; missing=" + ",".join(missing)
        )
    return tuple(found[task_id] for task_id in ordered_ids)


def _m41_processor_fingerprint(action_space_contract: Mapping[str, object]) -> str:
    return _fingerprint(
        {
            "schema_version": M42_M41_RUNTIME_PROCESSOR_SCHEMA,
            "processor_class": "BoundedActionEnvPostprocessorV0",
            "action_bound_config": ActionBoundConfig(mode=ActionBoundMode.PROJECT).to_dict(),
            "action_space_contract": dict(action_space_contract),
        }
    )


def _validation_runtime(checkpoint: FrozenM4Checkpoint) -> EvaluationRuntimeManifest:
    runtime_path = (
        checkpoint.run_root
        / "validation"
        / checkpoint.checkpoint_fingerprint.removeprefix("sha256:")
        / "evaluation_runtime_manifest.json"
    )
    try:
        runtime = EvaluationRuntimeManifest.from_dict(
            _read_json_object(runtime_path, label="M4 selected validation runtime manifest")
        )
    except (TypeError, ValueError) as error:
        raise ControllerRegistryError("invalid selected M4 validation runtime") from error
    if (
        not runtime.checkpoint_model_reload_validated
        or not runtime.policy_processor_reload_validated
        or not runtime.action_bound_processor_reload_validated
        or not runtime.deterministic_raw_action_matched
    ):
        raise ControllerRegistryError(
            "selected M4 validation runtime reload evidence is incomplete"
        )
    return runtime


def _validate_runtime_authority(
    *,
    runtime_selection_path: Path,
    runtime_selection: RuntimeSelectionContract,
    checkpoints: tuple[FrozenM4Checkpoint, ...],
    m41_runtime_processor_fingerprint: str,
) -> None:
    manifest_relative = runtime_selection.selection_evidence.get("experiment_manifest")
    if (
        not isinstance(manifest_relative, str)
        or not manifest_relative
        or Path(manifest_relative).is_absolute()
        or ".." in Path(manifest_relative).parts
    ):
        raise ControllerRegistryError("runtime selection lacks a safe experiment manifest")
    selection_path = _resolved_unlinked(
        runtime_selection_path,
        label="M4.2 runtime selection",
        directory=False,
    )
    manifest_path = selection_path.parent / manifest_relative
    try:
        manifest = M42ExperimentManifest.from_dict(
            _read_json_object(manifest_path, label="M4.2 runtime experiment manifest")
        )
    except (TypeError, ValueError) as error:
        raise ControllerRegistryError("invalid M4.2 runtime experiment manifest") from error
    checkpoint_fingerprints = tuple(checkpoint.checkpoint_fingerprint for checkpoint in checkpoints)
    if (
        manifest.m3b_dataset_fingerprint != runtime_selection.m3b_dataset_fingerprint
        or manifest.m4_checkpoint_fingerprints[:6] != checkpoint_fingerprints
        or manifest.m41_runtime_processor_fingerprint != m41_runtime_processor_fingerprint
        or manifest.fingerprint != runtime_selection.experiment_manifest_fingerprint
    ):
        raise ControllerRegistryError(
            "M4.2 authority differs from the frozen PerTask/action-runtime evidence"
        )


def _entry_from_evidence(
    checkpoint: FrozenM4Checkpoint,
    *,
    dataset_fingerprint: str,
    runtime_selection: RuntimeSelectionContract,
    validation_runtime: EvaluationRuntimeManifest,
    rollout_config: Mapping[str, object],
    artifact_validator: CheckpointArtifactValidator,
) -> ControllerRegistryEntry:
    if checkpoint.variant is not ActVariant.PER_TASK or checkpoint.task_id is None:
        raise ControllerRegistryError("M5A registry accepts only PerTask ACT checkpoints")
    identity = checkpoint.manifest.identity
    artifacts = artifact_validator(
        run_root=checkpoint.run_root,
        checkpoint_relative_path=checkpoint.selected_checkpoint.relative_path,
        expected_identity=identity,
    )
    if artifacts.record != checkpoint.selected_checkpoint:
        raise ControllerRegistryError("validated checkpoint record differs from M4 selection")
    components: CheckpointComponentFingerprints = artifacts.component_fingerprints
    if identity.m3b_export_fingerprint != dataset_fingerprint:
        raise ControllerRegistryError("selected M4 checkpoint binds a different M3B archive")
    runtime_identity = validation_runtime.identity
    if (
        runtime_identity.checkpoint_fingerprint != checkpoint.checkpoint_fingerprint
        or runtime_identity.policy_preprocessor_fingerprint != components.preprocessor
        or runtime_identity.policy_postprocessor_fingerprint != components.postprocessor
        or runtime_identity.action_bound_config.mode is not ActionBoundMode.PROJECT
        or runtime_identity.environment_id != LOCKED_ENVIRONMENT_ID
    ):
        raise ControllerRegistryError(
            "selected validation runtime differs from checkpoint or locked environment"
        )
    action_space_contract = runtime_identity.action_space_contract
    task_spec = next(
        spec for spec in CANONICAL_TASK_SPECS if stable_task_id(spec) == checkpoint.task_id
    )
    horizon = ExecutionHorizonConfig(LOCKED_EXECUTION_HORIZON)
    action_bound_config = ActionBoundConfig(mode=ActionBoundMode.PROJECT)
    controller_runtime_fingerprint = _fingerprint(
        {
            "schema_version": "langmani-m5a-frozen-controller-runtime-v0",
            "checkpoint_fingerprint": checkpoint.checkpoint_fingerprint,
            "model_component_fingerprint": components.model,
            "preprocessor_fingerprint": components.preprocessor,
            "postprocessor_fingerprint": components.postprocessor,
            "validation_runtime_fingerprint": runtime_identity.runtime_fingerprint,
            "environment_id": LOCKED_ENVIRONMENT_ID,
            "action_space_contract": dict(action_space_contract),
            "rollout_config": dict(rollout_config),
            "action_bound_config": action_bound_config.to_dict(),
            "runtime_selection_fingerprint": runtime_selection.source_fingerprint,
        }
    )
    return ControllerRegistryEntry(
        task_spec=task_spec,
        task_id=checkpoint.task_id,
        run_fingerprint=checkpoint.run_fingerprint,
        checkpoint_fingerprint=checkpoint.checkpoint_fingerprint,
        checkpoint_step=checkpoint.selected_checkpoint.global_step,
        selection_fingerprint=checkpoint.selection_fingerprint,
        model_component_fingerprint=components.model,
        preprocessor_fingerprint=components.preprocessor,
        postprocessor_fingerprint=components.postprocessor,
        train_statistics_fingerprint=identity.train_statistics_fingerprint,
        producer_git_commit=identity.git_commit,
        m3b_export_fingerprint=identity.m3b_export_fingerprint,
        m3b_split_manifest_digest=identity.m3b_split_manifest_digest,
        model_config=identity.model_config,
        data_contract=_controller_data_contract(identity.data_contract),
        source_data_contract_fingerprint=_fingerprint(dict(identity.data_contract)),
        controller_runtime_fingerprint=controller_runtime_fingerprint,
        environment_id=LOCKED_ENVIRONMENT_ID,
        action_space_contract=action_space_contract,
        rollout_config=rollout_config,
        action_bound_config=action_bound_config,
        m41_runtime_processor_fingerprint=_m41_processor_fingerprint(action_space_contract),
        execution_horizon=LOCKED_EXECUTION_HORIZON,
        execution_horizon_fingerprint=horizon.fingerprint,
        gripper_mode=LOCKED_GRIPPER_MODE,
        action_runtime_fingerprint=runtime_selection.runtime_fingerprint,
        runtime_selection_fingerprint=runtime_selection.source_fingerprint,
        runtime_implementation_fingerprint=runtime_selection.implementation_fingerprint,
        runtime_evaluation_git_commit=runtime_selection.evaluation_git_commit,
        horizon_selection_fingerprint=runtime_selection.horizon_selection_fingerprint,
        gripper_selection_fingerprint=runtime_selection.gripper_selection_fingerprint,
        runtime_experiment_manifest_fingerprint=(runtime_selection.experiment_manifest_fingerprint),
    )


def build_controller_registry(
    *,
    checkpoints: tuple[FrozenM4Checkpoint, ...],
    dataset_fingerprint: str,
    runtime_selection: RuntimeSelectionContract,
    validation_runtimes: Mapping[str, EvaluationRuntimeManifest],
    artifact_validator: CheckpointArtifactValidator = validate_act_checkpoint_artifacts,
) -> tuple[ControllerRegistry, ControllerRegistryLocators]:
    """Build the six-entry registry after validating immutable M4/M4.2 evidence."""

    if runtime_selection.execution_horizon != LOCKED_EXECUTION_HORIZON:
        raise ControllerRegistryError("runtime selection is not the locked H=10 result")
    if runtime_selection.gripper_mode != LOCKED_GRIPPER_MODE:
        raise ControllerRegistryError("runtime selection is not the locked project result")
    if runtime_selection.final_schedule_accessed:
        raise ControllerRegistryError("M5A must not consume m42_final_v0 evidence")
    _require_digest(dataset_fingerprint, "dataset_fingerprint")
    if runtime_selection.m3b_dataset_fingerprint != dataset_fingerprint:
        raise ControllerRegistryError("M4 and M4.2 evidence refer to different M3B data")
    if len(checkpoints) != 6:
        raise ControllerRegistryError("controller registry requires exactly six checkpoints")
    frozen = checkpoints
    if runtime_selection.representative_per_task_checkpoint_fingerprint not in {
        checkpoint.checkpoint_fingerprint for checkpoint in checkpoints
    }:
        raise ControllerRegistryError(
            "M4.2 runtime selection does not bind any frozen PerTask checkpoint"
        )
    runtime_keys = {checkpoint.checkpoint_fingerprint for checkpoint in checkpoints}
    if set(validation_runtimes) != runtime_keys:
        raise ControllerRegistryError("validation-runtime evidence does not cover six checkpoints")
    evaluation_configs = {checkpoint.manifest.config.evaluation for checkpoint in checkpoints}
    if len(evaluation_configs) != 1:
        raise ControllerRegistryError("PerTask controllers disagree on rollout configuration")
    rollout_config = locked_controller_rollout_config(next(iter(evaluation_configs)))
    entries = tuple(
        _entry_from_evidence(
            checkpoint,
            dataset_fingerprint=dataset_fingerprint,
            runtime_selection=runtime_selection,
            validation_runtime=validation_runtimes[checkpoint.checkpoint_fingerprint],
            rollout_config=rollout_config,
            artifact_validator=artifact_validator,
        )
        for checkpoint in frozen
    )
    registry = ControllerRegistry(entries=entries)
    locators = ControllerRegistryLocators(
        registry_fingerprint=registry.registry_fingerprint,
        locations=tuple(
            ControllerLocation(
                task_id=checkpoint.task_id or "",
                run_root=checkpoint.run_root,
                checkpoint_relative_path=checkpoint.selected_checkpoint.relative_path,
            )
            for checkpoint in frozen
        ),
    )
    return registry, locators


def load_controller_registry_metadata(
    *,
    checkpoint_root: Path,
    dataset_root: Path,
    runtime_selection_path: Path,
) -> tuple[ControllerRegistry, ControllerRegistryLocators]:
    """Load only M3B completion, M4 validation selections, and M4.2 runtime locks.

    In particular, this gate never opens M4 comparison, test, or historical
    fresh-seed artifacts.  All action bounds come from the currently active M1
    environment and are fingerprinted before any controller can be dispatched.
    """

    checkpoints = _discover_per_task_checkpoints(checkpoint_root)
    dataset_fingerprint = _dataset_fingerprint(dataset_root)
    runtime = load_runtime_selection(runtime_selection_path)
    validation_runtimes = {
        checkpoint.checkpoint_fingerprint: _validation_runtime(checkpoint)
        for checkpoint in checkpoints
    }
    action_contracts = {
        canonical_json(dict(item.identity.action_space_contract))
        for item in validation_runtimes.values()
    }
    if len(action_contracts) != 1:
        raise ControllerRegistryError("selected PerTask runtimes disagree on action bounds")
    processor_fingerprint = _m41_processor_fingerprint(
        next(iter(validation_runtimes.values())).identity.action_space_contract
    )
    _validate_runtime_authority(
        runtime_selection_path=runtime_selection_path,
        runtime_selection=runtime,
        checkpoints=checkpoints,
        m41_runtime_processor_fingerprint=processor_fingerprint,
    )
    return build_controller_registry(
        checkpoints=checkpoints,
        dataset_fingerprint=dataset_fingerprint,
        runtime_selection=runtime,
        validation_runtimes=validation_runtimes,
    )


def _fixture_digest(label: str) -> str:
    return _fingerprint({"fixture": label})


def build_fixture_controller_registry() -> ControllerRegistry:
    """Return deterministic, non-executable six-controller identity for CPU tests."""

    horizon = ExecutionHorizonConfig(LOCKED_EXECUTION_HORIZON)
    common = {
        "m3b_export_fingerprint": _fixture_digest("m3b"),
        "m3b_split_manifest_digest": _fixture_digest("split"),
        "m41_runtime_processor_fingerprint": _fixture_digest("m41-runtime"),
        "execution_horizon_fingerprint": horizon.fingerprint,
        "runtime_selection_fingerprint": _fixture_digest("runtime-selection"),
        "action_runtime_fingerprint": _fixture_digest("action-runtime"),
        "runtime_implementation_fingerprint": _fixture_digest("runtime-implementation"),
        "runtime_evaluation_git_commit": "c" * 40,
        "horizon_selection_fingerprint": _fixture_digest("horizon-selection"),
        "gripper_selection_fingerprint": _fixture_digest("gripper-selection"),
        "runtime_experiment_manifest_fingerprint": _fixture_digest("runtime-manifest"),
    }
    model = ActModelConfig.for_variant(ActVariant.PER_TASK).to_dict()
    entries = []
    for index, task_spec in enumerate(CANONICAL_TASK_SPECS):
        task_id = stable_task_id(task_spec)
        entries.append(
            ControllerRegistryEntry(
                task_spec=task_spec,
                task_id=task_id,
                run_fingerprint=_fixture_digest(f"run-{index}"),
                checkpoint_fingerprint=_fixture_digest(f"checkpoint-{index}"),
                checkpoint_step=100_000,
                selection_fingerprint=_fixture_digest(f"selection-{index}"),
                model_component_fingerprint=_fixture_digest(f"model-{index}"),
                preprocessor_fingerprint=_fixture_digest(f"preprocessor-{index}"),
                postprocessor_fingerprint=_fixture_digest(f"postprocessor-{index}"),
                train_statistics_fingerprint=_fixture_digest(f"statistics-{index}"),
                producer_git_commit="a" * 40,
                model_config=model,
                data_contract={
                    "video_backend": "pyav",
                    "download_videos": False,
                    "return_uint8": True,
                    "fps": 20,
                    "m3b_feature_contract": {
                        "image": "observation.images.base_camera",
                        "state": "observation.state",
                        "action": "action",
                    },
                    "policy_state_schema": {
                        "schema_version": "PandaPolicyStateV0",
                        "dimension": ACT_STATE_COMPONENTS,
                    },
                },
                source_data_contract_fingerprint=_fixture_digest(f"source-data-{index}"),
                controller_runtime_fingerprint=_fixture_digest(f"controller-runtime-{index}"),
                environment_id="LangMani-PickPlaceByInstruction-v0",
                action_space_contract={
                    "bounds_source": "environment_action_space",
                    "lower_bounds": [-1.0] * ACT_ACTION_COMPONENTS,
                    "lower_dtype": "float32",
                    "single_action_shape": [ACT_ACTION_COMPONENTS],
                    "upper_bounds": [1.0] * ACT_ACTION_COMPONENTS,
                    "upper_dtype": "float32",
                },
                rollout_config={"maximum_episode_steps": 200, "control_frequency_hz": 20},
                action_bound_config=ActionBoundConfig(mode=ActionBoundMode.PROJECT),
                execution_horizon=LOCKED_EXECUTION_HORIZON,
                gripper_mode=LOCKED_GRIPPER_MODE,
                **common,
            )
        )
    return ControllerRegistry(entries=tuple(entries))


__all__ = [
    "CONTROLLER_ENTRY_SCHEMA_VERSION",
    "CONTROLLER_LOCATORS_SCHEMA_VERSION",
    "CONTROLLER_REGISTRY_SCHEMA_VERSION",
    "LOCKED_EXECUTION_HORIZON",
    "LOCKED_GRIPPER_MODE",
    "LOCKED_ENVIRONMENT_ID",
    "ControllerLocation",
    "ControllerRegistry",
    "ControllerRegistryEntry",
    "ControllerRegistryError",
    "ControllerRegistryLocators",
    "build_controller_registry",
    "build_fixture_controller_registry",
    "inspect_controller_action_space_contract",
    "load_controller_registry_metadata",
    "locked_controller_rollout_config",
    "validate_active_controller_environment",
]
