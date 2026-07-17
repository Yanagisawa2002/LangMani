from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import replace

import pytest

from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import stable_task_id
from langmani.language import corpus as corpus_module
from langmani.language.corpus import (
    FinalLanguageAccessError,
    build_language_corpus,
    language_corpus_counts,
    materialize_final_language_examples,
    materialize_final_language_slot_map,
    normalize_language_text,
)
from langmani.language.router_types import (
    DEFAULT_LANGUAGE_CORPUS_CONFIG,
    LanguageSplit,
    RouterContractError,
    RouterStatus,
)


@pytest.fixture(scope="module")
def corpus():  # type: ignore[no-untyped-def]
    return build_language_corpus()


def test_corpus_exact_counts_are_predeclared_and_balanced(corpus) -> None:  # type: ignore[no-untyped-def]
    counts = language_corpus_counts(corpus)
    expected_totals = {
        LanguageSplit.TRAIN: (600, 300, 900),
        LanguageSplit.VALIDATION: (180, 120, 300),
        LanguageSplit.DEVELOPMENT: (240, 180, 420),
        LanguageSplit.FINAL: (360, 240, 600),
    }
    for split, (routeable, rejected, total) in expected_totals.items():
        assert dict(counts[split]) == {
            "routeable": routeable,
            "rejected": rejected,
            "total": total,
        }
        manifest = corpus.manifest.split_manifests[split]
        expected_per_task = DEFAULT_LANGUAGE_CORPUS_CONFIG.routeable_per_task[split]
        assert manifest.routeable_count_by_task_id == {
            stable_task_id(task_spec): expected_per_task for task_spec in CANONICAL_TASK_SPECS
        }
        assert sum(manifest.rejection_count_by_reason.values()) == rejected
        rejection_counts = tuple(manifest.rejection_count_by_reason.values())
        assert max(rejection_counts) - min(rejection_counts) <= 1


def test_all_routeable_task_specs_use_the_m1_canonical_template(corpus) -> None:  # type: ignore[no-untyped-def]
    for example in corpus.examples:
        if example.expected_status is RouterStatus.ROUTE:
            assert example.expected_task_spec is not None
            assert example.expected_task_spec.instruction_template_id == "canonical_v0"
            assert example.task_id == stable_task_id(example.expected_task_spec)
        else:
            assert example.expected_task_spec is None
            assert example.expected_rejection_reason is not None


def test_language_example_ids_are_content_bound(corpus) -> None:  # type: ignore[no-untyped-def]
    example = corpus.examples_for_split(LanguageSplit.TRAIN)[0]
    with pytest.raises(RouterContractError, match="example_id"):
        replace(example, raw_text=example.raw_text + " altered")


def test_build_is_deterministic(corpus) -> None:  # type: ignore[no-untyped-def]
    rebuilt = build_language_corpus()
    assert rebuilt.manifest.corpus_fingerprint == corpus.manifest.corpus_fingerprint
    assert rebuilt.manifest.to_dict() == corpus.manifest.to_dict()
    assert tuple(example.example_id for example in rebuilt.examples) == tuple(
        example.example_id for example in corpus.examples
    )


def test_development_and_final_are_disjoint_without_exposing_final_text(corpus) -> None:  # type: ignore[no-untyped-def]
    development = corpus.manifest.split_manifests[LanguageSplit.DEVELOPMENT]
    final = corpus.manifest.split_manifests[LanguageSplit.FINAL]
    assert not set(development.family_ids) & set(final.family_ids)
    assert not set(development.example_ids) & set(final.example_ids)
    assert LanguageSplit.FINAL not in corpus.visible_examples_by_split
    assert all(family.split is not LanguageSplit.FINAL for family in corpus.families)
    assert final.texts_sealed is True
    assert "raw_text" not in final.to_dict()

    with pytest.raises(FinalLanguageAccessError, match="sealed"):
        corpus.examples_for_split(LanguageSplit.FINAL)
    with pytest.raises(FinalLanguageAccessError, match="forbidden"):
        materialize_final_language_examples(corpus)


def test_development_corpus_build_never_calls_the_final_renderer(monkeypatch) -> None:
    requested_splits: list[tuple[LanguageSplit, ...]] = []
    original = corpus_module._build_records

    def guarded_build_records(  # type: ignore[no-untyped-def]
        config,
        *,
        splits,  # type: ignore[no-untyped-def]
    ):
        requested_splits.append(splits)
        if LanguageSplit.FINAL in splits:
            raise AssertionError("development attempted to render final language")
        return original(config, splits=splits)

    monkeypatch.setattr(corpus_module, "_build_records", guarded_build_records)
    built = corpus_module.build_language_corpus()

    assert requested_splits == [
        (
            LanguageSplit.TRAIN,
            LanguageSplit.VALIDATION,
            LanguageSplit.DEVELOPMENT,
        )
    ]
    serialized = json.dumps(built.manifest.to_dict(), sort_keys=True)
    assert "raw_text" not in serialized
    assert "normalized_text" not in serialized
    assert LanguageSplit.FINAL not in built.visible_examples_by_split


def test_development_corpus_build_does_not_import_final_text_authority() -> None:
    module_name = "langmani.language._final_language_authority"
    sys.modules.pop(module_name, None)

    build_language_corpus()

    assert module_name not in sys.modules


def test_explicit_final_materialization_is_a_separate_authorized_entry(corpus) -> None:  # type: ignore[no-untyped-def]
    final_examples = materialize_final_language_examples(corpus, authorize_final=True)
    assert len(final_examples) == 600
    assert all(example.split is LanguageSplit.FINAL for example in final_examples)

    slot_map = materialize_final_language_slot_map(corpus, authorize_final=True)
    final_manifest = corpus.manifest.split_manifests[LanguageSplit.FINAL]
    assert len(slot_map) == 600
    assert set(slot_map) == set(final_manifest.example_ids)
    assert {example.example_id for example in slot_map.values()} == {
        example.example_id for example in final_examples
    }


def test_normalization_is_audit_only_and_preserves_negation() -> None:
    normalized = normalize_language_text("  DO not\tmove the Red cube!  ")
    assert normalized == "do not move the red cube!"
    assert "not" in normalized.split()


def test_required_rejection_categories_exist(corpus) -> None:  # type: ignore[no-untyped-def]
    train = corpus.examples_for_split(LanguageSplit.TRAIN)
    reasons = Counter(
        example.expected_rejection_reason
        for example in train
        if example.expected_rejection_reason is not None
    )
    assert len(reasons) >= 14
    assert all(count >= 20 for count in reasons.values())
