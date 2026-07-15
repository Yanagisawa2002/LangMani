"""Content-bound discovery of immutable M4 evidence consumed by M4.2."""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import torch

from langmani.datasets.identity import sha256_hex
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import stable_task_id
from langmani.policies.act_action_bounds import (
    ActionBoundMode,
    EvaluationRuntimeManifest,
)
from langmani.policies.act_checkpoint import ActCheckpointError, load_act_checkpoint
from langmani.policies.act_data import load_completed_m3b_dataset
from langmani.policies.act_evaluation import load_checkpoint_selection
from langmani.policies.act_types import (
    ActExperimentManifest,
    ActVariant,
    CheckpointRecord,
)
from langmani.policies.m42_schedule import M42_M3B_EXPORT_FINGERPRINT
from langmani.policies.m42_types import M42ExperimentManifest

M42_PRIOR_EVIDENCE_SCHEMA_VERSION = "langmani-m42-prior-m4-evidence-v0"
M42_M41_RUNTIME_PROCESSOR_SCHEMA = "langmani-m42-m41-runtime-processor-v0"
REPRESENTATIVE_PER_TASK_ID = stable_task_id(CANONICAL_TASK_SPECS[2])


class M42EvidenceError(RuntimeError):
    """Raised when historical M4 evidence is absent, mutable, or inconsistent."""


def _lexical_absolute(path: Path) -> Path:
    """Return an absolute path without following filesystem links."""

    expanded = path.expanduser()
    if ".." in expanded.parts:
        raise M42EvidenceError(f"evidence path must not contain parent traversal: {path}")
    return expanded if expanded.is_absolute() else Path.cwd() / expanded


def _is_link(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def _resolved_unlinked(path: Path, *, label: str, must_exist: bool = True) -> Path:
    """Resolve a path only after rejecting every lexical linked component."""

    lexical = _lexical_absolute(path)
    for component in reversed((lexical, *lexical.parents)):
        if _is_link(component):
            raise M42EvidenceError(f"{label} traverses a symlink or junction: {component}")
    try:
        resolved = lexical.resolve(strict=must_exist)
    except OSError as error:
        raise M42EvidenceError(f"missing or inaccessible {label}: {lexical}") from error
    if resolved != lexical:
        raise M42EvidenceError(f"{label} does not resolve to its lexical path: {lexical}")
    return resolved


def _require_directory(path: Path, *, label: str) -> Path:
    result = _resolved_unlinked(path, label=label)
    if not result.is_dir():
        raise M42EvidenceError(f"{label} must be a real directory: {result}")
    return result


def _require_contained(root: Path, child: Path, *, label: str) -> None:
    try:
        child.relative_to(root)
    except ValueError as error:
        raise M42EvidenceError(f"{label} escapes its owned root: {child}") from error
    if child == root:
        raise M42EvidenceError(f"{label} must be below its owned root: {child}")


def _owned_relative_directory(root: Path, relative: str, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise M42EvidenceError(f"{label} must use a non-empty portable relative path")
    portable = PurePosixPath(relative)
    if portable.is_absolute() or any(part in {"", ".", ".."} for part in portable.parts):
        raise M42EvidenceError(f"{label} must be a canonical relative path")
    if portable.as_posix() != relative:
        raise M42EvidenceError(f"{label} must be canonically normalized")
    candidate = root.joinpath(*portable.parts)
    result = _require_directory(candidate, label=label)
    _require_contained(root, result, label=label)
    return result


def _reject_linked_tree(root: Path, *, label: str) -> None:
    """Inspect a tree without following linked directories."""

    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = tuple(directory.iterdir())
        except OSError as error:
            raise M42EvidenceError(f"could not inspect {label}: {directory}") from error
        for entry in entries:
            if _is_link(entry):
                raise M42EvidenceError(f"{label} contains a symlink or junction: {entry}")
            if entry.is_dir():
                _require_contained(root, entry, label=label)
                pending.append(entry)


def _read_object(path: Path) -> dict[str, Any]:
    artifact = _resolved_unlinked(path, label="evidence artifact")
    if not artifact.is_file():
        raise M42EvidenceError(f"missing evidence artifact: {artifact}")
    try:
        value = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise M42EvidenceError(f"could not read evidence artifact: {artifact}") from error
    if not isinstance(value, dict):
        raise M42EvidenceError(f"evidence artifact must contain one JSON object: {artifact}")
    return value


def _fingerprint(value: object) -> str:
    return f"sha256:{sha256_hex(value)}"


@dataclass(frozen=True, slots=True)
class FrozenM4Checkpoint:
    variant: ActVariant
    task_id: str | None
    run_root: Path
    manifest: ActExperimentManifest
    selected_checkpoint: CheckpointRecord
    selection_fingerprint: str

    def __post_init__(self) -> None:
        root = _require_directory(self.run_root, label="frozen M4 run root")
        object.__setattr__(self, "run_root", root)
        if self.manifest.identity.variant is not self.variant:
            raise M42EvidenceError("frozen M4 variant differs from its run manifest")
        if self.manifest.identity.task_id != self.task_id:
            raise M42EvidenceError("frozen M4 task differs from its run manifest")
        if not self.manifest.complete or not self.manifest.training_state.completed:
            raise M42EvidenceError("M4.2 requires a completed M4 training run")
        if (
            self.manifest.selected_checkpoint_fingerprint
            != self.selected_checkpoint.checkpoint_fingerprint
        ):
            raise M42EvidenceError("run manifest does not bind the selected checkpoint")

    @property
    def run_fingerprint(self) -> str:
        return self.manifest.identity.run_fingerprint

    @property
    def checkpoint_fingerprint(self) -> str:
        return self.selected_checkpoint.checkpoint_fingerprint

    @property
    def checkpoint_path(self) -> Path:
        return _owned_relative_directory(
            self.run_root,
            self.selected_checkpoint.relative_path,
            label="selected checkpoint",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "variant": self.variant.value,
            "task_id": self.task_id,
            "run_root": str(self.run_root),
            "run_fingerprint": self.run_fingerprint,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "checkpoint_relative_path": self.selected_checkpoint.relative_path,
            "selection_fingerprint": self.selection_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class PriorM4Evidence:
    dataset_root: Path
    dataset_fingerprint: str
    verification_fingerprint: str
    comparison_fingerprint: str
    m41_runtime_processor_fingerprint: str
    checkpoints: tuple[FrozenM4Checkpoint, ...]

    def __post_init__(self) -> None:
        if len(self.checkpoints) != 8:
            raise M42EvidenceError("M4.2 requires exactly eight immutable M4 checkpoints")
        if len({item.run_fingerprint for item in self.checkpoints}) != 8:
            raise M42EvidenceError("M4 run fingerprints must be unique")
        per_task = tuple(item for item in self.checkpoints if item.variant is ActVariant.PER_TASK)
        if tuple(item.task_id for item in per_task) != tuple(
            stable_task_id(spec) for spec in CANONICAL_TASK_SPECS
        ):
            raise M42EvidenceError("frozen PerTask evidence must preserve canonical task order")
        variants = tuple(item.variant for item in self.checkpoints[6:])
        if variants != (ActVariant.MIXED_UNCONDITIONED, ActVariant.MIXED_TASK_ONEHOT):
            raise M42EvidenceError("frozen mixed M4 controls are incomplete or reordered")

    @property
    def representative_per_task(self) -> FrozenM4Checkpoint:
        return next(
            item
            for item in self.checkpoints
            if item.variant is ActVariant.PER_TASK and item.task_id == REPRESENTATIVE_PER_TASK_ID
        )

    @property
    def mixed_task_onehot(self) -> FrozenM4Checkpoint:
        return next(
            item for item in self.checkpoints if item.variant is ActVariant.MIXED_TASK_ONEHOT
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": M42_PRIOR_EVIDENCE_SCHEMA_VERSION,
            "dataset_root": str(self.dataset_root),
            "dataset_fingerprint": self.dataset_fingerprint,
            "verification_fingerprint": self.verification_fingerprint,
            "comparison_fingerprint": self.comparison_fingerprint,
            "m41_runtime_processor_fingerprint": self.m41_runtime_processor_fingerprint,
            "checkpoints": [item.to_dict() for item in self.checkpoints],
        }


def _m41_runtime_processor_payload(runtime: EvaluationRuntimeManifest) -> dict[str, object]:
    """Project the exact M4.1 processor identity out of a validated runtime manifest."""
    identity = runtime.identity
    if (
        identity.action_bound_config.mode is not ActionBoundMode.PROJECT
        or not runtime.action_bound_processor_reload_validated
        or not runtime.policy_processor_reload_validated
        or not runtime.checkpoint_model_reload_validated
        or not runtime.deterministic_raw_action_matched
    ):
        raise M42EvidenceError("M4.1 runtime processor evidence is incomplete or not project mode")
    return {
        "schema_version": M42_M41_RUNTIME_PROCESSOR_SCHEMA,
        "processor_class": "BoundedActionEnvPostprocessorV0",
        "action_bound_config": identity.action_bound_config.to_dict(),
        "action_space_contract": dict(identity.action_space_contract),
    }


def m41_runtime_processor_fingerprint(
    runtimes: tuple[EvaluationRuntimeManifest, ...],
) -> str:
    """Require all eight historical controls to use one exact M4.1 processor."""
    if len(runtimes) != 8:
        raise M42EvidenceError("M4.2 requires M4.1 runtime evidence for all eight controls")
    payloads = tuple(_m41_runtime_processor_payload(runtime) for runtime in runtimes)
    reference = payloads[0]
    if any(payload != reference for payload in payloads[1:]):
        raise M42EvidenceError("historical M4 runs disagree on the M4.1 runtime processor")
    return _fingerprint(reference)


def load_m42_experiment_manifest(path: str | Path) -> M42ExperimentManifest:
    """Load a persisted immutable M4.2 experiment-provenance manifest."""
    return M42ExperimentManifest.from_dict(_read_object(Path(path)))


def validate_m42_experiment_manifest(
    manifest: M42ExperimentManifest,
    evidence: PriorM4Evidence,
) -> None:
    """Bind a shared M4.2 manifest back to the exact validated M4 evidence."""
    if (
        manifest.m3b_dataset_fingerprint != evidence.dataset_fingerprint
        or manifest.prior_m4_verification_fingerprint != evidence.verification_fingerprint
        or manifest.m41_runtime_processor_fingerprint != evidence.m41_runtime_processor_fingerprint
        or manifest.m4_checkpoint_fingerprints
        != tuple(item.checkpoint_fingerprint for item in evidence.checkpoints)
    ):
        raise M42EvidenceError("M4.2 experiment manifest differs from validated M4 provenance")


def _load_run(root: Path) -> ActExperimentManifest:
    return ActExperimentManifest.from_dict(_read_object(root / "run_manifest.json"))


def _discover_runs(checkpoint_root: Path) -> dict[str, tuple[Path, ActExperimentManifest]]:
    owned_root = _require_directory(checkpoint_root, label="checkpoint root")
    found: dict[str, tuple[Path, ActExperimentManifest]] = {}
    try:
        children = sorted(owned_root.iterdir(), key=lambda item: item.name)
    except OSError as error:
        raise M42EvidenceError(f"could not inspect checkpoint root: {owned_root}") from error
    for child in children:
        if _is_link(child):
            raise M42EvidenceError(f"checkpoint root contains a linked child run: {child}")
        if not child.is_dir():
            continue
        root = _require_directory(child, label="M4 child run")
        if root.parent != owned_root:
            raise M42EvidenceError(f"M4 child run is not directly owned: {root}")
        manifest_path = root / "run_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = _load_run(root)
        fingerprint = manifest.identity.run_fingerprint
        if fingerprint in found:
            raise M42EvidenceError(
                f"duplicate run fingerprint below checkpoint root: {fingerprint}"
            )
        found[fingerprint] = (root, manifest)
    return found


def _validate_frozen_checkpoint_on_disk(checkpoint: FrozenM4Checkpoint) -> None:
    """Strictly reload one selected checkpoint and release it before the next load."""

    checkpoint_path = checkpoint.checkpoint_path
    _reject_linked_tree(checkpoint_path, label="selected checkpoint tree")
    loaded: object | None = None
    try:
        loaded = load_act_checkpoint(
            run_root=checkpoint.run_root,
            checkpoint_relative_path=checkpoint.selected_checkpoint.relative_path,
            expected_identity=checkpoint.manifest.identity,
            for_resume=False,
            restore_rng=False,
        )
        loaded_record = getattr(loaded, "record", None)
        if loaded_record != checkpoint.selected_checkpoint:
            raise M42EvidenceError(
                "strictly reloaded checkpoint record differs from frozen M4 evidence"
            )
        if loaded_record.checkpoint_fingerprint != checkpoint.checkpoint_fingerprint:
            raise M42EvidenceError(
                "strictly reloaded checkpoint fingerprint differs from frozen M4 evidence"
            )
    except ActCheckpointError as error:
        raise M42EvidenceError(
            "selected M4 checkpoint failed atomic marker, hash, or strict reload validation: "
            f"{checkpoint.checkpoint_fingerprint}: {error}"
        ) from error
    finally:
        loaded = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _validate_all_frozen_checkpoints(
    checkpoints: tuple[FrozenM4Checkpoint, ...],
) -> None:
    if len(checkpoints) != 8:
        raise M42EvidenceError("M4.2 disk validation requires all eight M4 checkpoints")
    for checkpoint in checkpoints:
        _validate_frozen_checkpoint_on_disk(checkpoint)


def load_prior_m4_evidence(
    *,
    diagnostics_root: str | Path,
    checkpoint_root: str | Path,
    dataset_root: str | Path,
    validate_dataset_storage: bool = True,
) -> PriorM4Evidence:
    """Validate and resolve the exact eight completed M4 selected checkpoints."""
    diagnostics = _require_directory(Path(diagnostics_root), label="M4 diagnostics root")
    checkpoints = _require_directory(Path(checkpoint_root), label="M4 checkpoint root")
    dataset = _require_directory(Path(dataset_root), label="M3B dataset root")
    verification = _read_object(diagnostics / "verification.json")
    required_true = (
        "passed",
        "full_experiment_validated",
        "physical_target_validated",
        "validation_selection_validated",
        "test_lock_validated",
    )
    if any(verification.get(name) is not True for name in required_true):
        raise M42EvidenceError("M4 full verification is absent or incomplete")
    comparison = _read_object(diagnostics / "target_full" / "comparison.json")
    if (
        comparison.get("passed") is not True
        or comparison.get("full_experiment_validated") is not True
    ):
        raise M42EvidenceError("M4 comparison report is absent or incomplete")
    if comparison.get("baseline_quality_validated") is not False:
        raise M42EvidenceError("M4.2 expects the recorded M4 quality gate to remain false")
    plan = _read_object(diagnostics / "target_full" / "dry_run_plan.json")
    planned = plan.get("runs")
    compared = comparison.get("runs")
    if not isinstance(planned, list) or len(planned) != 8:
        raise M42EvidenceError("M4 dry-run plan must bind exactly eight runs")
    if not isinstance(compared, list) or len(compared) != 8:
        raise M42EvidenceError("M4 comparison must bind exactly eight runs")
    completed = load_completed_m3b_dataset(
        dataset,
        require_full=True,
        validate_storage=validate_dataset_storage,
    )
    if completed.export_fingerprint != M42_M3B_EXPORT_FINGERPRINT:
        raise M42EvidenceError("M3B dataset fingerprint differs from the locked M4.2 source")
    if comparison.get("dataset_fingerprint") != completed.export_fingerprint:
        raise M42EvidenceError("M4 comparison dataset differs from the supplied M3B dataset")

    discovered = _discover_runs(checkpoints)
    result: list[FrozenM4Checkpoint] = []
    selected_runtimes: list[EvaluationRuntimeManifest] = []
    for index, (planned_item, compared_item) in enumerate(zip(planned, compared, strict=True)):
        if not isinstance(planned_item, dict) or not isinstance(compared_item, dict):
            raise M42EvidenceError("M4 run evidence entries must be JSON objects")
        run_fingerprint = planned_item.get("run_fingerprint")
        if not isinstance(run_fingerprint, str) or run_fingerprint not in discovered:
            raise M42EvidenceError(
                f"planned M4 run is missing below checkpoint root: {run_fingerprint}"
            )
        if compared_item.get("run_fingerprint") != run_fingerprint:
            raise M42EvidenceError("M4 comparison run ordering differs from the dry-run plan")
        run_root, manifest = discovered[run_fingerprint]
        expected_variant = (
            ActVariant.PER_TASK
            if index < 6
            else ActVariant.MIXED_UNCONDITIONED
            if index == 6
            else ActVariant.MIXED_TASK_ONEHOT
        )
        if manifest.identity.variant is not expected_variant:
            raise M42EvidenceError("discovered M4 run variant differs from the frozen ordering")
        expected_task = stable_task_id(CANONICAL_TASK_SPECS[index]) if index < 6 else None
        if manifest.identity.task_id != expected_task:
            raise M42EvidenceError("discovered M4 run task differs from the frozen ordering")
        selection_path = _resolved_unlinked(
            run_root / "checkpoint_selection.json",
            label="M4 checkpoint selection",
        )
        if not selection_path.is_file():
            raise M42EvidenceError(f"M4 checkpoint selection must be a real file: {selection_path}")
        selection = load_checkpoint_selection(selection_path)
        if not selection.selection_locked:
            raise M42EvidenceError("M4 selected checkpoint is not locked")
        selected = next(
            (
                item
                for item in manifest.checkpoints
                if item.checkpoint_fingerprint == selection.selected_checkpoint_fingerprint
            ),
            None,
        )
        if selected is None:
            raise M42EvidenceError("M4 selected checkpoint is absent from its run manifest")
        if compared_item.get("selected_checkpoint_fingerprint") != selected.checkpoint_fingerprint:
            raise M42EvidenceError("M4 comparison selected checkpoint differs from its lock")
        selected_runtime = EvaluationRuntimeManifest.from_dict(
            _read_object(run_root / "test" / "evaluation_runtime_manifest.json")
        )
        if selected_runtime.identity.checkpoint_fingerprint != selected.checkpoint_fingerprint:
            raise M42EvidenceError(
                "M4.1 runtime processor evidence refers to a different selected checkpoint"
            )
        selected_runtimes.append(selected_runtime)
        result.append(
            FrozenM4Checkpoint(
                variant=expected_variant,
                task_id=expected_task,
                run_root=run_root,
                manifest=manifest,
                selected_checkpoint=selected,
                selection_fingerprint=selection.selection_fingerprint,
            )
        )
    frozen = PriorM4Evidence(
        dataset_root=dataset,
        dataset_fingerprint=completed.export_fingerprint,
        verification_fingerprint=_fingerprint(verification),
        comparison_fingerprint=_fingerprint(comparison),
        m41_runtime_processor_fingerprint=m41_runtime_processor_fingerprint(
            tuple(selected_runtimes)
        ),
        checkpoints=tuple(result),
    )
    _validate_all_frozen_checkpoints(frozen.checkpoints)
    return frozen


__all__ = [
    "M42EvidenceError",
    "M42_PRIOR_EVIDENCE_SCHEMA_VERSION",
    "M42_M41_RUNTIME_PROCESSOR_SCHEMA",
    "PriorM4Evidence",
    "REPRESENTATIVE_PER_TASK_ID",
    "FrozenM4Checkpoint",
    "load_prior_m4_evidence",
    "load_m42_experiment_manifest",
    "m41_runtime_processor_fingerprint",
    "validate_m42_experiment_manifest",
]
