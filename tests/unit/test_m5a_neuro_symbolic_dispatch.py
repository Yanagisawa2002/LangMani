from __future__ import annotations

from pathlib import Path

import pytest

from langmani.language import neuro_symbolic_dispatch as dispatch
from langmani.language.controller_registry import build_fixture_controller_registry
from langmani.language.corpus import build_language_corpus
from langmani.language.neuro_symbolic_router import (
    OUTLINES_VERSION,
    DeterministicSafetyArbiterV0,
    NeuroSymbolicRouterV0,
    SymbolicLexicalParserV0,
    build_qwen4b_loader_config,
    build_semantic_frame_prompt,
    select_semantic_prompt_examples,
    semantic_schema_fingerprint,
)
from langmani.language.router_types import LanguageSplit


def _frozen_objects(root: Path) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    corpus = build_language_corpus()
    parser = SymbolicLexicalParserV0()
    arbiter = DeterministicSafetyArbiterV0()
    examples = select_semantic_prompt_examples(corpus.examples_for_split(LanguageSplit.TRAIN))
    _content, prompt_fingerprint, prompt_ids = build_semantic_frame_prompt(
        examples=examples,
        parser=parser,
    )
    config = build_qwen4b_loader_config(maximum_new_tokens=128)
    runtime_fingerprint = f"sha256:{'1' * 64}"
    artifact_fingerprint = f"sha256:{'2' * 64}"
    model_files = {"model.safetensors": {"size_bytes": 8, "sha256": f"sha256:{'3' * 64}"}}
    owner = {
        "runtime_fingerprint": runtime_fingerprint,
        "implementation_git": "4" * 40,
    }
    flags = {
        "neuro_symbolic_language_safety_gate_passed": True,
        "learned_router_selected": True,
        "one_scene_control_smoke_authorized": True,
        "exact_rejection_taxonomy_quality_passed": False,
        "language_final_accessed": False,
        "control_final_accessed": False,
        "test_split_accessed": False,
        "m42_final_accessed": False,
        "smolvla_go": False,
    }
    complete = {"flags": flags}
    objects = {
        "candidate_selection.json": {
            "candidate": "NeuroSymbolicRouterV0",
            "only_promotable_candidate": True,
            "learned_router_selected": True,
            "one_scene_control_smoke_authorized": True,
            "safety_gate": {"passed": True},
        },
        "router_lock.json": {
            "runtime_fingerprint": runtime_fingerprint,
            "neuro_symbolic_router_locked": True,
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "tokenizer_revision": config.tokenizer_revision,
            "model_fingerprint": f"sha256:{'5' * 64}",
            "model_file_identities": model_files,
            "dtype": "bfloat16",
            "outlines_version": OUTLINES_VERSION,
            "prompt_fingerprint": prompt_fingerprint,
            "rendered_prompt_fingerprint": f"sha256:{'6' * 64}",
            "prompt_example_ids": list(prompt_ids),
            "semantic_schema_fingerprint": semantic_schema_fingerprint(),
            "symbolic_contract_fingerprint": parser.contract_fingerprint,
            "arbiter_fingerprint": arbiter.contract_fingerprint,
            "generation_config": {
                "do_sample": False,
                "repeat_count": 2,
                "semantic_maximum_new_tokens": 256,
                "quantization": "none",
            },
            "lock_fingerprint": f"sha256:{'7' * 64}",
        },
        "runtime_identity.json": {
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "tokenizer_revision": config.tokenizer_revision,
            "model_fingerprint": f"sha256:{'5' * 64}",
            "language_final_accessed": False,
            "control_final_accessed": False,
            "test_split_accessed": False,
            "m42_final_accessed": False,
            "smolvla_started": False,
        },
        "immutability_identity.json": {
            "source_model_identity": {
                "file_identities": model_files,
                "snapshot_size_bytes": 8,
            },
            "loaded_model_file_identities": model_files,
        },
    }
    validated = {
        "passed": True,
        "root": str(root),
        "owner": owner,
        "complete": complete,
        "artifact_fingerprint": artifact_fingerprint,
    }
    return validated, objects


def _patch_evidence(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
) -> dict[str, dict[str, object]]:
    validated, objects = _frozen_objects(root)
    monkeypatch.setattr(dispatch, "verify_m5a41_evidence", lambda _root: {"passed": True})
    monkeypatch.setattr(dispatch, "validate_m5a41_evidence", lambda _root: validated)
    monkeypatch.setattr(dispatch, "read_object", lambda path: objects[path.name])
    return objects


def test_selected_m5a41_source_is_read_only_and_path_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_evidence(monkeypatch, tmp_path / "evidence")
    corpus = build_language_corpus()
    source = dispatch.validate_selected_neuro_symbolic_dispatch_source(
        tmp_path / "evidence",
        corpus=corpus,
    )
    registry = build_fixture_controller_registry()
    binding = dispatch.bind_selected_router_to_controller_registry(source, registry)

    assert "evidence_root" not in source.identity_dict()
    assert source.to_dict()["evidence_root"] == str(tmp_path / "evidence")
    assert source.exact_rejection_taxonomy_quality_passed is False
    assert binding.ordered_task_ids == tuple(entry.task_id for entry in registry.entries)
    assert binding.identity_dict()["controller_loaded"] is False
    assert binding.identity_dict()["environment_created"] is False


def test_selected_m5a41_source_rejects_unpromoted_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects = _patch_evidence(monkeypatch, tmp_path / "evidence")
    objects["candidate_selection.json"]["one_scene_control_smoke_authorized"] = False

    with pytest.raises(dispatch.NeuroSymbolicDispatchError, match="did not select"):
        dispatch.validate_selected_neuro_symbolic_dispatch_source(
            tmp_path / "evidence",
            corpus=build_language_corpus(),
        )


def test_selected_router_loader_rehashes_files_and_has_no_download_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_evidence(monkeypatch, tmp_path / "evidence")
    corpus = build_language_corpus()
    source = dispatch.validate_selected_neuro_symbolic_dispatch_source(
        tmp_path / "evidence",
        corpus=corpus,
    )

    class FakeLoader:
        def __init__(self, *, config, device, local_files_only):  # type: ignore[no-untyped-def]
            assert local_files_only is True
            self.config = config
            self.device = device
            self.file_identities = dict(source.model_file_identities)
            self.snapshot_size_bytes = source.snapshot_size_bytes

        def rendered_prompt_fingerprint(self, _prompt: str) -> str:
            return source.rendered_prompt_fingerprint

    class FakeExtractor:
        def __init__(self, *, loader, prompt_content, maximum_new_tokens):  # type: ignore[no-untyped-def]
            del loader, prompt_content
            assert maximum_new_tokens == 256
            self.compilation_seconds = 0.0
            self.generation_metadata: list[dict[str, object]] = []

        def extract(self, command: str):  # type: ignore[no-untyped-def]
            raise AssertionError(f"fixture should not infer: {command}")

    monkeypatch.setattr(dispatch, "TransformersLocalTextGenerator", FakeLoader)
    monkeypatch.setattr(dispatch, "OutlinesQwenSemanticFrameExtractorV0", FakeExtractor)

    router = dispatch.load_selected_neuro_symbolic_router(
        source,
        corpus=corpus,
        device="cpu",
        local_files_only=True,
    )
    assert isinstance(router, NeuroSymbolicRouterV0)
    with pytest.raises(dispatch.NeuroSymbolicDispatchError, match="local cache"):
        dispatch.load_selected_neuro_symbolic_router(
            source,
            corpus=corpus,
            device="cpu",
            local_files_only=False,
        )
