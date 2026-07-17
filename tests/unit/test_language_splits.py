from __future__ import annotations

import pytest

from langmani.language import corpus as corpus_module
from langmani.language.corpus import build_language_corpus
from langmani.language.router_types import (
    DEFAULT_LANGUAGE_CORPUS_CONFIG,
    LanguageSplit,
    LanguageTemplateFamily,
)
from langmani.language.schedules import (
    M5A_CONTROL_DEV_SCHEDULE_ID,
    M5A_CONTROL_FINAL_SCHEDULE_ID,
    M5A_LANGUAGE_DEV_SCHEDULE_ID,
    M5A_LANGUAGE_FINAL_SCHEDULE_ID,
    REQUIRED_EXCLUSION_SOURCE_IDS,
    ControlScheduleConfig,
    FinalControlScheduleAccessError,
    M5AScheduleError,
    SeedExclusionSource,
    build_authoritative_seed_exclusions,
    build_m5a_schedule_bundle,
    materialize_control_schedule,
)
from langmani.language.splits import (
    LanguageSplitIsolationError,
    validate_family_split_isolation,
)
from langmani.policies.m42_schedule import validate_locked_schedules


def _family(
    family_id: str,
    split: LanguageSplit,
    *,
    structural_signature: str,
    near_duplicate_group_id: str,
) -> LanguageTemplateFamily:
    return LanguageTemplateFamily(
        family_id=family_id,
        split=split,
        structural_signature=structural_signature,
        near_duplicate_group_id=near_duplicate_group_id,
        category="test",
        generation_provenance={"source": "unit_test"},
    )


def _exclusion_sources(*, first_seed: int = 1_000) -> tuple[SeedExclusionSource, ...]:
    return tuple(
        SeedExclusionSource(
            source_id=source_id,
            source_fingerprint=f"sha256:{index + 1:064x}",
            scene_seeds=(first_seed + index,),
        )
        for index, source_id in enumerate(REQUIRED_EXCLUSION_SOURCE_IDS)
    )


def test_near_duplicate_family_leakage_is_rejected() -> None:
    families = (
        _family(
            "train-family",
            LanguageSplit.TRAIN,
            structural_signature="train structure",
            near_duplicate_group_id="shared-near-form",
        ),
        _family(
            "development-family",
            LanguageSplit.DEVELOPMENT,
            structural_signature="development structure",
            near_duplicate_group_id="shared-near-form",
        ),
    )
    with pytest.raises(LanguageSplitIsolationError, match="near-duplicate"):
        validate_family_split_isolation(families, ())


def test_structural_signature_leakage_is_rejected() -> None:
    families = (
        _family(
            "validation-family",
            LanguageSplit.VALIDATION,
            structural_signature="shared structure",
            near_duplicate_group_id="validation-near",
        ),
        _family(
            "final-family",
            LanguageSplit.FINAL,
            structural_signature="shared structure",
            near_duplicate_group_id="final-near",
        ),
    )
    with pytest.raises(LanguageSplitIsolationError, match="structural signature"):
        validate_family_split_isolation(families, ())


def test_generated_isolation_signatures_do_not_encode_the_split() -> None:
    shared = "{prefix}move the {color} cube into the {side} bin{punct}"
    train = corpus_module._language_family(
        split=LanguageSplit.TRAIN,
        kind="route",
        slug="train-copy",
        category="test",
        structure_source=shared,
        generator_version=DEFAULT_LANGUAGE_CORPUS_CONFIG.generator_version,
    )
    validation = corpus_module._language_family(
        split=LanguageSplit.VALIDATION,
        kind="route",
        slug="validation-copy",
        category="test",
        structure_source=shared,
        generator_version=DEFAULT_LANGUAGE_CORPUS_CONFIG.generator_version,
    )

    assert train.structural_signature == validation.structural_signature
    assert train.near_duplicate_group_id == validation.near_duplicate_group_id
    assert all(split.value not in train.structural_signature for split in LanguageSplit)
    with pytest.raises(LanguageSplitIsolationError, match="structural signature"):
        validate_family_split_isolation((train, validation), ())


def test_control_schedule_excludes_prior_seeds_and_locks_all_four_ids() -> None:
    corpus = build_language_corpus()
    sources = _exclusion_sources(first_seed=1_000)
    config = ControlScheduleConfig(
        exclusion_sources=sources,
        candidate_seed_start=1_000,
    )
    bundle = build_m5a_schedule_bundle(corpus=corpus, config=config)

    assert bundle.language_development.schedule_id == M5A_LANGUAGE_DEV_SCHEDULE_ID
    assert bundle.language_final.schedule_id == M5A_LANGUAGE_FINAL_SCHEDULE_ID
    assert bundle.control_development.schedule_id == M5A_CONTROL_DEV_SCHEDULE_ID
    assert bundle.control_final.schedule_id == M5A_CONTROL_FINAL_SCHEDULE_ID
    assert len(bundle.control_development.ordered_scene_seeds) == 12
    assert len(bundle.control_final.ordered_scene_seeds) == 30
    assert bundle.control_development.episode_count == 72
    assert bundle.control_final.episode_count == 180
    assert len(bundle.development_episodes) == 72

    development_seeds = set(bundle.control_development.ordered_scene_seeds)
    final_seeds = set(bundle.control_final.ordered_scene_seeds)
    assert not development_seeds & final_seeds
    assert not development_seeds & config.excluded_scene_seeds
    assert not final_seeds & config.excluded_scene_seeds
    assert min(development_seeds) >= 1_011

    assert bundle.language_final.sealed is True
    assert bundle.control_final.sealed is True
    assert "raw_text" not in bundle.language_final.to_dict()
    assert "raw_text" not in bundle.control_final.to_dict()

    with pytest.raises(FinalControlScheduleAccessError, match="sealed"):
        materialize_control_schedule(bundle.control_final)


def test_control_schedule_is_deterministic_while_final_remains_sealed() -> None:
    corpus = build_language_corpus()
    config = ControlScheduleConfig(
        exclusion_sources=_exclusion_sources(first_seed=9_000),
        candidate_seed_start=8_990,
    )
    first = build_m5a_schedule_bundle(corpus=corpus, config=config)
    second = build_m5a_schedule_bundle(corpus=corpus, config=config)
    assert first.control_development.schedule_fingerprint == (
        second.control_development.schedule_fingerprint
    )
    assert first.control_final.schedule_fingerprint == second.control_final.schedule_fingerprint
    with pytest.raises(FinalControlScheduleAccessError, match="sealed"):
        materialize_control_schedule(first.control_final)


def test_schedule_bundle_uses_only_opaque_final_language_slots(monkeypatch) -> None:
    requested_splits: list[tuple[LanguageSplit, ...]] = []
    original = corpus_module._build_records

    def guarded_build_records(  # type: ignore[no-untyped-def]
        config,
        *,
        splits,  # type: ignore[no-untyped-def]
    ):
        requested_splits.append(splits)
        if LanguageSplit.FINAL in splits:
            raise AssertionError("schedule construction attempted to render final language")
        return original(config, splits=splits)

    monkeypatch.setattr(corpus_module, "_build_records", guarded_build_records)
    corpus = corpus_module.build_language_corpus()
    bundle = build_m5a_schedule_bundle(
        corpus=corpus,
        config=ControlScheduleConfig(
            exclusion_sources=_exclusion_sources(first_seed=9_000),
            candidate_seed_start=8_990,
        ),
    )

    assert requested_splits == [
        (
            LanguageSplit.TRAIN,
            LanguageSplit.VALIDATION,
            LanguageSplit.DEVELOPMENT,
        )
    ]
    assert len(bundle.language_final.ordered_example_ids) == 600
    assert all(
        example_id.startswith("langmani-m5a-sealed-final-example-")
        for example_id in bundle.language_final.ordered_example_ids
    )
    assert set(bundle.control_final.associated_language_example_ids) <= set(
        bundle.language_final.ordered_example_ids
    )


def test_all_exclusion_sources_are_required() -> None:
    incomplete = _exclusion_sources()[:-1]
    with pytest.raises(M5AScheduleError, match="every required source"):
        ControlScheduleConfig(exclusion_sources=incomplete)


def test_authoritative_exclusions_bind_every_prior_source_without_final_access() -> None:
    m42_development, m42_final = validate_locked_schedules()
    sources = build_authoritative_seed_exclusions(
        m43_development_fingerprint=f"sha256:{99:064x}",
        m43_development_scene_seeds=m42_development.ordered_scene_seeds,
    )

    assert {source.source_id for source in sources} == set(REQUIRED_EXCLUSION_SOURCE_IDS)
    by_id = {source.source_id: source for source in sources}
    assert by_id["m42_dev_v0"].scene_seeds == tuple(sorted(m42_development.ordered_scene_seeds))
    assert by_id["m42_final_v0"].scene_seeds == tuple(sorted(m42_final.ordered_scene_seeds))
    assert by_id["m43_development"].scene_seeds == by_id["m42_dev_v0"].scene_seeds
    assert all(source.source_fingerprint.startswith("sha256:") for source in sources)


def test_authoritative_exclusions_reject_a_non_m43_development_schedule() -> None:
    m42_development, _ = validate_locked_schedules()
    with pytest.raises(M5AScheduleError, match="exactly reuse"):
        build_authoritative_seed_exclusions(
            m43_development_fingerprint=f"sha256:{99:064x}",
            m43_development_scene_seeds=(*m42_development.ordered_scene_seeds[:-1], 1),
        )
