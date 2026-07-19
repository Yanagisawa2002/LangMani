"""Read-only binding of the selected M5A.4.1 router to frozen PerTask control."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from langmani.datasets.identity import sha256_hex
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import stable_task_id
from langmani.language.controller_registry import ControllerRegistry
from langmani.language.corpus import GeneratedLanguageCorpus
from langmani.language.llm_router import TransformersLocalTextGenerator
from langmani.language.neuro_symbolic_router import (
    NEURO_SYMBOLIC_ROUTER_NAME,
    NEURO_SYMBOLIC_ROUTER_VERSION,
    OUTLINES_VERSION,
    DeterministicSafetyArbiterV0,
    NeuroSymbolicRouterV0,
    OutlinesQwenSemanticFrameExtractorV0,
    SymbolicLexicalParserV0,
    build_qwen4b_loader_config,
    build_semantic_frame_prompt,
    select_semantic_prompt_examples,
    semantic_schema_fingerprint,
)
from langmani.language.neuro_symbolic_safety_evidence import (
    read_object,
    validate_m5a41_evidence,
)
from langmani.language.neuro_symbolic_safety_verifier import verify_m5a41_evidence
from langmani.language.router_types import LanguageSplit

NEURO_SYMBOLIC_DISPATCH_ROUTER_LABEL = "neuro_symbolic"
NEURO_SYMBOLIC_DISPATCH_SOURCE_SCHEMA = "langmani-m5a41-dispatch-source-v0"
NEURO_SYMBOLIC_CONTROLLER_BINDING_SCHEMA = "langmani-m5a41-controller-binding-v0"


class NeuroSymbolicDispatchError(RuntimeError):
    """Raised when an offline-selected router cannot safely enter dispatch."""


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise NeuroSymbolicDispatchError(f"{label} must be one object")
    return cast(Mapping[str, object], value)


def _sha(value: object) -> str:
    return f"sha256:{sha256_hex(value)}"


@dataclass(frozen=True, slots=True)
class SelectedNeuroSymbolicDispatchSource:
    """Portable semantic identity plus the local immutable evidence locator."""

    evidence_root: Path
    source_runtime_fingerprint: str
    source_artifact_fingerprint: str
    source_implementation_git: str
    router_lock_fingerprint: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    model_fingerprint: str
    model_file_identities: Mapping[str, object]
    snapshot_size_bytes: int
    dtype: str
    prompt_fingerprint: str
    rendered_prompt_fingerprint: str
    prompt_example_ids: tuple[str, ...]
    semantic_schema_fingerprint: str
    symbolic_contract_fingerprint: str
    arbiter_fingerprint: str
    semantic_maximum_new_tokens: int
    exact_rejection_taxonomy_quality_passed: bool

    def identity_dict(self) -> dict[str, object]:
        """Return a path-independent identity suitable for run fingerprints."""

        return {
            "schema_version": NEURO_SYMBOLIC_DISPATCH_SOURCE_SCHEMA,
            "router_label": NEURO_SYMBOLIC_DISPATCH_ROUTER_LABEL,
            "router_name": NEURO_SYMBOLIC_ROUTER_NAME,
            "router_version": NEURO_SYMBOLIC_ROUTER_VERSION,
            "source_runtime_fingerprint": self.source_runtime_fingerprint,
            "source_artifact_fingerprint": self.source_artifact_fingerprint,
            "source_implementation_git": self.source_implementation_git,
            "router_lock_fingerprint": self.router_lock_fingerprint,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "model_fingerprint": self.model_fingerprint,
            "model_file_identities": dict(self.model_file_identities),
            "snapshot_size_bytes": self.snapshot_size_bytes,
            "dtype": self.dtype,
            "prompt_fingerprint": self.prompt_fingerprint,
            "rendered_prompt_fingerprint": self.rendered_prompt_fingerprint,
            "prompt_example_ids": list(self.prompt_example_ids),
            "semantic_schema_fingerprint": self.semantic_schema_fingerprint,
            "symbolic_contract_fingerprint": self.symbolic_contract_fingerprint,
            "arbiter_fingerprint": self.arbiter_fingerprint,
            "semantic_maximum_new_tokens": self.semantic_maximum_new_tokens,
            "exact_rejection_taxonomy_quality_passed": (
                self.exact_rejection_taxonomy_quality_passed
            ),
            "learned_router_selected": True,
            "one_scene_control_smoke_authorized": True,
            "language_final_accessed": False,
            "control_final_accessed": False,
            "test_split_accessed": False,
            "m42_final_accessed": False,
            "smolvla_go": False,
        }

    @property
    def dispatch_source_fingerprint(self) -> str:
        return _sha(self.identity_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            **self.identity_dict(),
            "dispatch_source_fingerprint": self.dispatch_source_fingerprint,
            "evidence_root": str(self.evidence_root),
        }


@dataclass(frozen=True, slots=True)
class NeuroSymbolicControllerBinding:
    """Read-only proof that the selected router covers the exact six-controller registry."""

    dispatch_source_fingerprint: str
    controller_registry_fingerprint: str
    ordered_task_ids: tuple[str, ...]

    def identity_dict(self) -> dict[str, object]:
        return {
            "schema_version": NEURO_SYMBOLIC_CONTROLLER_BINDING_SCHEMA,
            "dispatch_source_fingerprint": self.dispatch_source_fingerprint,
            "controller_registry_fingerprint": self.controller_registry_fingerprint,
            "ordered_task_ids": list(self.ordered_task_ids),
            "controller_count": len(self.ordered_task_ids),
            "controller_metadata_only": True,
            "controller_loaded": False,
            "environment_created": False,
            "environment_reset_called": False,
            "environment_step_count": 0,
        }

    @property
    def binding_fingerprint(self) -> str:
        return _sha(self.identity_dict())

    def to_dict(self) -> dict[str, object]:
        return {**self.identity_dict(), "binding_fingerprint": self.binding_fingerprint}


def validate_selected_neuro_symbolic_dispatch_source(
    evidence_root: str | Path,
    *,
    corpus: GeneratedLanguageCorpus,
) -> SelectedNeuroSymbolicDispatchSource:
    """Independently revalidate the selected offline evidence without loading a model."""

    verified = verify_m5a41_evidence(evidence_root)
    validated = validate_m5a41_evidence(evidence_root)
    if verified.get("passed") is not True or validated.get("passed") is not True:
        raise NeuroSymbolicDispatchError("M5A.4.1 evidence did not pass independent verification")
    root = Path(cast(str, validated["root"]))
    owner = _mapping(validated.get("owner"), label="M5A.4.1 owner")
    complete = _mapping(validated.get("complete"), label="M5A.4.1 completion")
    flags = _mapping(complete.get("flags"), label="M5A.4.1 completion flags")
    selection = read_object(root / "candidate_selection.json")
    lock = read_object(root / "router_lock.json")
    runtime = read_object(root / "runtime_identity.json")
    immutability = read_object(root / "immutability_identity.json")
    if (
        selection.get("candidate") != NEURO_SYMBOLIC_ROUTER_NAME
        or selection.get("only_promotable_candidate") is not True
        or selection.get("learned_router_selected") is not True
        or selection.get("one_scene_control_smoke_authorized") is not True
        or _mapping(selection.get("safety_gate"), label="M5A.4.1 safety gate").get("passed")
        is not True
        or flags.get("neuro_symbolic_language_safety_gate_passed") is not True
        or flags.get("learned_router_selected") is not True
        or flags.get("one_scene_control_smoke_authorized") is not True
    ):
        raise NeuroSymbolicDispatchError("M5A.4.1 did not select exactly the frozen hybrid")
    for name in (
        "language_final_accessed",
        "control_final_accessed",
        "test_split_accessed",
        "m42_final_accessed",
        "smolvla_go",
    ):
        if flags.get(name) is not False:
            raise NeuroSymbolicDispatchError(f"M5A.4.1 source accessed prohibited field {name}")
    if any(
        runtime.get(name) is not False
        for name in (
            "language_final_accessed",
            "control_final_accessed",
            "test_split_accessed",
            "m42_final_accessed",
            "smolvla_started",
        )
    ):
        raise NeuroSymbolicDispatchError("M5A.4.1 runtime accessed a prohibited source or stage")
    parser = SymbolicLexicalParserV0()
    arbiter = DeterministicSafetyArbiterV0()
    prompt_examples = select_semantic_prompt_examples(
        corpus.examples_for_split(LanguageSplit.TRAIN)
    )
    prompt_content, prompt_fingerprint, prompt_ids = build_semantic_frame_prompt(
        examples=prompt_examples,
        parser=parser,
    )
    del prompt_content
    model_identity = _mapping(
        immutability.get("source_model_identity"), label="frozen model identity"
    )
    file_identities = _mapping(model_identity.get("file_identities"), label="frozen model files")
    generation = _mapping(lock.get("generation_config"), label="router generation config")
    if (
        lock.get("runtime_fingerprint") != owner.get("runtime_fingerprint")
        or lock.get("neuro_symbolic_router_locked") is not True
        or lock.get("model_id") != runtime.get("model_id")
        or lock.get("model_revision") != runtime.get("model_revision")
        or lock.get("tokenizer_revision") != runtime.get("tokenizer_revision")
        or lock.get("model_fingerprint") != runtime.get("model_fingerprint")
        or lock.get("model_file_identities") != dict(file_identities)
        or immutability.get("loaded_model_file_identities") != dict(file_identities)
        or lock.get("dtype") != "bfloat16"
        or lock.get("outlines_version") != OUTLINES_VERSION
        or lock.get("prompt_fingerprint") != prompt_fingerprint
        or lock.get("prompt_example_ids") != list(prompt_ids)
        or lock.get("semantic_schema_fingerprint") != semantic_schema_fingerprint()
        or lock.get("symbolic_contract_fingerprint") != parser.contract_fingerprint
        or lock.get("arbiter_fingerprint") != arbiter.contract_fingerprint
        or generation
        != {
            "do_sample": False,
            "repeat_count": 2,
            "semantic_maximum_new_tokens": 256,
            "quantization": "none",
        }
    ):
        raise NeuroSymbolicDispatchError("frozen model/prompt/schema/parser/arbiter lock differs")
    snapshot_size = model_identity.get("snapshot_size_bytes")
    if isinstance(snapshot_size, bool) or not isinstance(snapshot_size, int) or snapshot_size <= 0:
        raise NeuroSymbolicDispatchError("frozen model snapshot size is malformed")
    rendered = lock.get("rendered_prompt_fingerprint")
    lock_fingerprint = lock.get("lock_fingerprint")
    source_git = owner.get("implementation_git")
    if not all(isinstance(value, str) for value in (rendered, lock_fingerprint, source_git)):
        raise NeuroSymbolicDispatchError("M5A.4.1 dispatch identity is incomplete")
    return SelectedNeuroSymbolicDispatchSource(
        evidence_root=root,
        source_runtime_fingerprint=cast(str, owner["runtime_fingerprint"]),
        source_artifact_fingerprint=cast(str, validated["artifact_fingerprint"]),
        source_implementation_git=cast(str, source_git),
        router_lock_fingerprint=cast(str, lock_fingerprint),
        model_id=cast(str, lock["model_id"]),
        model_revision=cast(str, lock["model_revision"]),
        tokenizer_revision=cast(str, lock["tokenizer_revision"]),
        model_fingerprint=cast(str, lock["model_fingerprint"]),
        model_file_identities=dict(file_identities),
        snapshot_size_bytes=snapshot_size,
        dtype="bfloat16",
        prompt_fingerprint=prompt_fingerprint,
        rendered_prompt_fingerprint=cast(str, rendered),
        prompt_example_ids=prompt_ids,
        semantic_schema_fingerprint=semantic_schema_fingerprint(),
        symbolic_contract_fingerprint=parser.contract_fingerprint,
        arbiter_fingerprint=arbiter.contract_fingerprint,
        semantic_maximum_new_tokens=256,
        exact_rejection_taxonomy_quality_passed=(
            flags.get("exact_rejection_taxonomy_quality_passed") is True
        ),
    )


def bind_selected_router_to_controller_registry(
    source: SelectedNeuroSymbolicDispatchSource,
    registry: ControllerRegistry,
) -> NeuroSymbolicControllerBinding:
    """Bind semantic routes to six metadata-only PerTask entries without loading weights."""

    expected = tuple(stable_task_id(task) for task in CANONICAL_TASK_SPECS)
    observed = tuple(entry.task_id for entry in registry.entries)
    if observed != expected or len(set(observed)) != 6:
        raise NeuroSymbolicDispatchError("controller registry is not the canonical six-task order")
    for task_id in expected:
        if registry.require(task_id).task_id != task_id:
            raise NeuroSymbolicDispatchError("controller registry lookup changed task identity")
    return NeuroSymbolicControllerBinding(
        dispatch_source_fingerprint=source.dispatch_source_fingerprint,
        controller_registry_fingerprint=registry.registry_fingerprint,
        ordered_task_ids=expected,
    )


def load_selected_neuro_symbolic_router(
    source: SelectedNeuroSymbolicDispatchSource,
    *,
    corpus: GeneratedLanguageCorpus,
    device: str,
    local_files_only: bool = True,
) -> NeuroSymbolicRouterV0:
    """Load exactly the locked router once, with no fallback or mutable selection."""

    if not local_files_only:
        raise NeuroSymbolicDispatchError("selected dispatch requires the immutable local cache")
    if device == "cuda":
        try:
            import torch
        except ImportError as error:
            raise NeuroSymbolicDispatchError("CUDA dispatch requires PyTorch") from error
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available():
            raise NeuroSymbolicDispatchError("CUDA dispatch requires CUDA_VISIBLE_DEVICES=0")
        if torch.cuda.device_count() != 1:
            raise NeuroSymbolicDispatchError("dispatch requires exactly one visible GPU")
    parser = SymbolicLexicalParserV0()
    arbiter = DeterministicSafetyArbiterV0()
    examples = select_semantic_prompt_examples(corpus.examples_for_split(LanguageSplit.TRAIN))
    prompt_content, prompt_fingerprint, prompt_ids = build_semantic_frame_prompt(
        examples=examples,
        parser=parser,
    )
    if (
        prompt_fingerprint != source.prompt_fingerprint
        or prompt_ids != source.prompt_example_ids
        or parser.contract_fingerprint != source.symbolic_contract_fingerprint
        or arbiter.contract_fingerprint != source.arbiter_fingerprint
        or semantic_schema_fingerprint() != source.semantic_schema_fingerprint
    ):
        raise NeuroSymbolicDispatchError("active router contracts differ from selected evidence")
    config = build_qwen4b_loader_config(maximum_new_tokens=128)
    if (
        config.model_id != source.model_id
        or config.model_revision != source.model_revision
        or config.tokenizer_revision != source.tokenizer_revision
        or config.dtype != source.dtype
    ):
        raise NeuroSymbolicDispatchError("active Qwen loader config differs from selected evidence")
    loader = TransformersLocalTextGenerator(
        config=config,
        device=device,
        local_files_only=True,
    )
    if (
        loader.file_identities != dict(source.model_file_identities)
        or loader.snapshot_size_bytes != source.snapshot_size_bytes
        or loader.rendered_prompt_fingerprint(prompt_content) != source.rendered_prompt_fingerprint
    ):
        raise NeuroSymbolicDispatchError("local model files or chat-template rendering differ")
    extractor = OutlinesQwenSemanticFrameExtractorV0(
        loader=loader,
        prompt_content=prompt_content,
        maximum_new_tokens=source.semantic_maximum_new_tokens,
    )
    return NeuroSymbolicRouterV0(
        parser=parser,
        semantic_extractor=extractor,
        arbiter=arbiter,
    )


__all__ = [
    "NEURO_SYMBOLIC_CONTROLLER_BINDING_SCHEMA",
    "NEURO_SYMBOLIC_DISPATCH_ROUTER_LABEL",
    "NEURO_SYMBOLIC_DISPATCH_SOURCE_SCHEMA",
    "NeuroSymbolicControllerBinding",
    "NeuroSymbolicDispatchError",
    "SelectedNeuroSymbolicDispatchSource",
    "bind_selected_router_to_controller_registry",
    "load_selected_neuro_symbolic_router",
    "validate_selected_neuro_symbolic_dispatch_source",
]
