"""Deterministic project-owned M5A language corpus generation."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from langmani.datasets.identity import sha256_hex
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import TaskSpec, stable_task_id
from langmani.language.router_types import (
    DEFAULT_LANGUAGE_CORPUS_CONFIG,
    LanguageCorpusConfig,
    LanguageCorpusManifest,
    LanguageExample,
    LanguageSplit,
    LanguageSplitManifest,
    LanguageTemplateFamily,
    RouterRejectionReason,
    RouterStatus,
    stable_language_example_id,
)
from langmani.language.splits import (
    SplitIsolationReport,
    bind_sealed_final_split_isolation,
    build_language_split_manifest,
    partition_examples,
    validate_family_split_isolation,
)


class FinalLanguageAccessError(PermissionError):
    """Raised when development code attempts to reveal sealed final commands."""


_DEVELOPMENT_VISIBLE_SPLITS: tuple[LanguageSplit, ...] = (
    LanguageSplit.TRAIN,
    LanguageSplit.VALIDATION,
    LanguageSplit.DEVELOPMENT,
)
M5A_FINAL_LANGUAGE_SEAL_VERSION = "langmani-m5a-final-language-seal-v1"
M5A_SEALED_FINAL_EXAMPLE_ID_PREFIX = "langmani-m5a-sealed-final-example-"
# This digest was produced once by the separately authorized final renderer.  The
# development path consumes only the digest and opaque semantic slots below.
M5A_FINAL_LANGUAGE_CONTENT_FINGERPRINT = (
    "sha256:0d923f0210916b2e4c037bf194658b2fbbdf2846dbc2272598f511c5220a4164"
)
M5A_LANGUAGE_SPLIT_CONTENT_FINGERPRINTS: Mapping[LanguageSplit, str] = MappingProxyType(
    {
        LanguageSplit.TRAIN: (
            "sha256:f75111fd255a8ca8143697aa0f893d4761e15571762f84463cbe75e0cb299444"
        ),
        LanguageSplit.VALIDATION: (
            "sha256:cda7022ab846a789269444c87c67b79dbf8438d3e738e64b0b4b7d2b2cbb87cf"
        ),
        LanguageSplit.DEVELOPMENT: (
            "sha256:8b3f65280a91f612996c602f92bc6d133e19e7fa066d83337353260b3124547c"
        ),
        LanguageSplit.FINAL: M5A_FINAL_LANGUAGE_CONTENT_FINGERPRINT,
    }
)
M5A_FINAL_LANGUAGE_ISOLATION_ATTESTATION_FINGERPRINT = (
    "sha256:d746bdadb16b0d9b104a2a8c2beb9ccc2ed0ba77fa50c81bb5249c4771af5546"
)


def normalize_language_text(text: str) -> str:
    """Normalize for audit only while preserving semantic negation tokens."""
    if not isinstance(text, str):
        raise TypeError("language text must be a string")
    normalized = unicodedata.normalize("NFKC", text).lower()
    return " ".join(normalized.split())


@dataclass(frozen=True, slots=True)
class _RouteFamily:
    slug: str
    category: str
    template: str


_ROUTE_FAMILIES: Mapping[LanguageSplit, tuple[_RouteFamily, ...]] = MappingProxyType(
    {
        LanguageSplit.TRAIN: (
            _RouteFamily(
                "direct_imperative",
                "direct_imperative",
                "{prefix}{verb} the {color} cube into the {side} bin{punct}",
            ),
            _RouteFamily(
                "polite_request",
                "polite_request",
                "{prefix}Please {verb} the {color} cube to the {side} bin{punct}",
            ),
            _RouteFamily(
                "object_first",
                "object_first_phrasing",
                "{prefix}The {color} cube is the object to {verb} into the {side} bin{punct}",
            ),
            _RouteFamily(
                "destination_first",
                "destination_first_phrasing",
                "{prefix}Into the {side} bin, {verb} the {color} cube{punct}",
            ),
            _RouteFamily(
                "telegraphic",
                "concise_telegraphic_command",
                "{prefix}{color} cube to {side} bin{punct}",
            ),
            _RouteFamily(
                "active_voice",
                "active_voice_variation",
                "{prefix}Use the robot to {verb} the {color} cube into the {side} bin{punct}",
            ),
            _RouteFamily(
                "context_clause",
                "redundant_contextual_clause",
                "{prefix}For the current tabletop task, {verb} the {color} cube to the {side} bin{punct}",
            ),
            _RouteFamily(
                "harmless_distractor",
                "harmless_distractor_clause",
                "{prefix}Ignore the empty staging note; {verb} the {color} cube into the {side} bin{punct}",
            ),
            _RouteFamily(
                "indirect_request",
                "indirect_unambiguous_request",
                "{prefix}I would like the {color} cube {verb_past} into the {side} bin{punct}",
            ),
            _RouteFamily(
                "explicit_correction",
                "explicit_unambiguous_correction",
                "{prefix}Correction to the wording: {verb} only the {color} cube into only the {side} bin{punct}",
            ),
        ),
        LanguageSplit.VALIDATION: (
            _RouteFamily(
                "courteous_transfer",
                "polite_request",
                "{prefix}Would you {verb} the {color} cube inside the {side} container{punct}",
            ),
            _RouteFamily(
                "bin_assignment",
                "object_first_phrasing",
                "{prefix}Assign the {color} cube to the {side} bin by moving it there{punct}",
            ),
            _RouteFamily(
                "destination_explanation",
                "destination_first_phrasing",
                "{prefix}The destination is the {side} bin, so {verb} the {color} cube into it{punct}",
            ),
        ),
        LanguageSplit.DEVELOPMENT: (
            _RouteFamily(
                "kind_relocation",
                "polite_request",
                "{prefix}Kindly relocate the {color} cube so it rests in the {side} bin{punct}",
            ),
            _RouteFamily(
                "set_down_inside",
                "active_voice_variation",
                "{prefix}Pick up and set down the {color} cube inside the {side} container{punct}",
            ),
            _RouteFamily(
                "job_statement",
                "indirect_unambiguous_request",
                "{prefix}Your job is to get the {color} cube into the {side} bin{punct}",
            ),
            _RouteFamily(
                "destination_emphasis",
                "redundant_contextual_clause",
                "{prefix}Place the {color} cube; its correct destination is the {side} bin{punct}",
            ),
        ),
    }
)

_PREFIXES = (
    "",
    "Now, ",
    "Carefully, ",
    "Next, ",
    "For this step, ",
    "When ready, ",
    "As requested, ",
    "For the robot, ",
    "In this command, ",
    "Without changing anything else, ",
)
_VERBS = ("move", "put", "place", "transfer", "relocate")
_VERBS_PAST = ("moved", "put", "placed", "transferred", "relocated")
_PUNCTUATION = (".", "!", ".", ".", "!")
_M5A_FINAL_ROUTE_FAMILY_SLUGS: tuple[str, ...] = (
    "requested_transfer",
    "container_choice",
    "single_object_move",
    "goal_description",
    "bin_first_instruction",
    "exclusive_assignment",
)


@dataclass(frozen=True, slots=True)
class _RejectionFamily:
    reason: RouterRejectionReason
    status: RouterStatus
    category: str
    bases: Mapping[LanguageSplit, str]


_REJECTION_FAMILIES: tuple[_RejectionFamily, ...] = (
    _RejectionFamily(
        RouterRejectionReason.MISSING_OBJECT,
        RouterStatus.REJECT_AMBIGUOUS,
        "missing_object",
        MappingProxyType(
            {
                LanguageSplit.TRAIN: "Put the unspecified cube in the left bin.",
                LanguageSplit.VALIDATION: "Move a cube, without saying which one, to the right bin.",
                LanguageSplit.DEVELOPMENT: "Please transfer that cube into the left bin.",
            }
        ),
    ),
    _RejectionFamily(
        RouterRejectionReason.MISSING_DESTINATION,
        RouterStatus.REJECT_AMBIGUOUS,
        "missing_destination",
        MappingProxyType(
            {
                LanguageSplit.TRAIN: "Move the red cube but no destination is specified.",
                LanguageSplit.VALIDATION: "Please relocate the green cube somewhere.",
                LanguageSplit.DEVELOPMENT: "Pick up the blue cube and await a destination.",
            }
        ),
    ),
    _RejectionFamily(
        RouterRejectionReason.CONFLICTING_OBJECTS,
        RouterStatus.REJECT_AMBIGUOUS,
        "multiple_conflicting_objects",
        MappingProxyType(
            {
                LanguageSplit.TRAIN: "Move the red and blue cubes into the right bin.",
                LanguageSplit.VALIDATION: "Put both the green cube and red cube in the left bin.",
                LanguageSplit.DEVELOPMENT: "Choose the blue or green cube for the right bin.",
            }
        ),
    ),
    _RejectionFamily(
        RouterRejectionReason.CONFLICTING_BINS,
        RouterStatus.REJECT_AMBIGUOUS,
        "multiple_conflicting_bins",
        MappingProxyType(
            {
                LanguageSplit.TRAIN: "Put the green cube in both the left and right bins.",
                LanguageSplit.VALIDATION: "Move the red cube to the left bin or the right bin.",
                LanguageSplit.DEVELOPMENT: "Use both bins as destinations for the blue cube.",
            }
        ),
    ),
    _RejectionFamily(
        RouterRejectionReason.UNSUPPORTED_OBJECT,
        RouterStatus.REJECT_UNSUPPORTED,
        "unsupported_object_or_color",
        MappingProxyType(
            {
                LanguageSplit.TRAIN: "Place the yellow cube in the left bin.",
                LanguageSplit.VALIDATION: "Move the orange cube to the right bin.",
                LanguageSplit.DEVELOPMENT: "Transfer the purple cube into the left bin.",
            }
        ),
    ),
    _RejectionFamily(
        RouterRejectionReason.UNSUPPORTED_DESTINATION,
        RouterStatus.REJECT_UNSUPPORTED,
        "unsupported_destination",
        MappingProxyType(
            {
                LanguageSplit.TRAIN: "Place the red cube in the center bin.",
                LanguageSplit.VALIDATION: "Move the green cube to the rear container.",
                LanguageSplit.DEVELOPMENT: "Transfer the blue cube into the upper bin.",
            }
        ),
    ),
    _RejectionFamily(
        RouterRejectionReason.MULTIPLE_TASKS,
        RouterStatus.REJECT_AMBIGUOUS,
        "multiple_sequential_tasks",
        MappingProxyType(
            {
                LanguageSplit.TRAIN: "Put the red cube left, then move the blue cube right.",
                LanguageSplit.VALIDATION: "First move green left and afterward place red right.",
                LanguageSplit.DEVELOPMENT: "Transfer blue right, followed by green left.",
            }
        ),
    ),
    _RejectionFamily(
        RouterRejectionReason.UNRESOLVED_CORRECTION,
        RouterStatus.REJECT_AMBIGUOUS,
        "unresolved_correction",
        MappingProxyType(
            {
                LanguageSplit.TRAIN: "Red cube left—no, blue cube right—actually never mind.",
                LanguageSplit.VALIDATION: "Move green right; correction, red left; disregard that too.",
                LanguageSplit.DEVELOPMENT: "Blue to left, or perhaps green to right; I have not decided.",
            }
        ),
    ),
    _RejectionFamily(
        RouterRejectionReason.CONTRADICTORY_NEGATION,
        RouterStatus.REJECT_AMBIGUOUS,
        "contradictory_negation",
        MappingProxyType(
            {
                LanguageSplit.TRAIN: "Move and do not move the green cube to the left bin.",
                LanguageSplit.VALIDATION: "Place the red cube right but never place it there.",
                LanguageSplit.DEVELOPMENT: "Transfer the blue cube left and also leave it untouched.",
            }
        ),
    ),
    _RejectionFamily(
        RouterRejectionReason.MEANINGLESS_TEXT,
        RouterStatus.REJECT_MALFORMED,
        "meaningless_or_noise_text",
        MappingProxyType(
            {
                LanguageSplit.TRAIN: "florp zibble quux",
                LanguageSplit.VALIDATION: "wugga blip norp",
                LanguageSplit.DEVELOPMENT: "trazzle fim wob",
            }
        ),
    ),
    _RejectionFamily(
        RouterRejectionReason.EMPTY_TEXT,
        RouterStatus.REJECT_MALFORMED,
        "empty_text",
        MappingProxyType(
            {
                split: ""
                for split in (
                    LanguageSplit.TRAIN,
                    LanguageSplit.VALIDATION,
                    LanguageSplit.DEVELOPMENT,
                )
            }
        ),
    ),
    _RejectionFamily(
        RouterRejectionReason.MALFORMED_CONTROL_CHARACTERS,
        RouterStatus.REJECT_MALFORMED,
        "malformed_control_characters",
        MappingProxyType(
            {
                LanguageSplit.TRAIN: "Move\x00the red cube left.",
                LanguageSplit.VALIDATION: "Place\x01the green cube right.",
                LanguageSplit.DEVELOPMENT: "Transfer\x02the blue cube left.",
            }
        ),
    ),
    _RejectionFamily(
        RouterRejectionReason.UNSUPPORTED_ACTION,
        RouterStatus.REJECT_UNSUPPORTED,
        "unsupported_action",
        MappingProxyType(
            {
                LanguageSplit.TRAIN: "Open the drawer.",
                LanguageSplit.VALIDATION: "Press the red button.",
                LanguageSplit.DEVELOPMENT: "Close the cabinet door.",
            }
        ),
    ),
    _RejectionFamily(
        RouterRejectionReason.UNSUPPORTED_SPATIAL_REFERENCE,
        RouterStatus.REJECT_UNSUPPORTED,
        "unsupported_spatial_reference",
        MappingProxyType(
            {
                LanguageSplit.TRAIN: "Move the red cube over there.",
                LanguageSplit.VALIDATION: "Put the green cube beside that place.",
                LanguageSplit.DEVELOPMENT: "Transfer the blue cube near where I pointed.",
            }
        ),
    ),
)

_M5A_FINAL_REJECTION_REASONS: tuple[RouterRejectionReason, ...] = (
    RouterRejectionReason.MISSING_OBJECT,
    RouterRejectionReason.MISSING_DESTINATION,
    RouterRejectionReason.CONFLICTING_OBJECTS,
    RouterRejectionReason.CONFLICTING_BINS,
    RouterRejectionReason.UNSUPPORTED_OBJECT,
    RouterRejectionReason.UNSUPPORTED_DESTINATION,
    RouterRejectionReason.MULTIPLE_TASKS,
    RouterRejectionReason.UNRESOLVED_CORRECTION,
    RouterRejectionReason.CONTRADICTORY_NEGATION,
    RouterRejectionReason.MEANINGLESS_TEXT,
    RouterRejectionReason.MALFORMED_CONTROL_CHARACTERS,
    RouterRejectionReason.UNSUPPORTED_ACTION,
    RouterRejectionReason.UNSUPPORTED_SPATIAL_REFERENCE,
)

_REJECTION_PREFIXES = (
    "",
    "Please interpret this: ",
    "Robot command: ",
    "Request: ",
    "Input says: ",
)
_REJECTION_SUFFIXES = (
    "",
    " Please proceed.",
    " This is the whole request.",
    " End input.",
    " Execute it.",
)


def _stable_family_id(*, split: LanguageSplit, kind: str, slug: str) -> str:
    digest = sha256_hex(
        {
            "kind": kind,
            "slug": slug,
            "split": split.value,
            "schema": DEFAULT_LANGUAGE_CORPUS_CONFIG.schema_version,
        }
    )
    return f"langmani-m5a-language-family-{digest}"


def _language_family(
    *,
    split: LanguageSplit,
    kind: str,
    slug: str,
    category: str,
    structure_source: str,
    generator_version: str,
) -> LanguageTemplateFamily:
    normalized_structure = normalize_language_text(structure_source)
    structural_signature = f"sha256:{sha256_hex({'kind': kind, 'structure': normalized_structure})}"
    near_duplicate_structure = " ".join(
        character if character.isalnum() or character in "{}_" else " "
        for character in normalized_structure
    ).split()
    near_duplicate_group_id = (
        f"sha256:{sha256_hex({'kind': kind, 'tokens': near_duplicate_structure})}"
    )
    return LanguageTemplateFamily(
        family_id=_stable_family_id(split=split, kind=kind, slug=slug),
        split=split,
        # These isolation identities deliberately exclude the split.  Encoding
        # ``split`` here would make the cross-split check true by construction.
        structural_signature=structural_signature,
        near_duplicate_group_id=near_duplicate_group_id,
        category=category,
        generation_provenance={
            "generator_version": generator_version,
            "kind": kind,
            "family_slug": slug,
            "structure_source_fingerprint": f"sha256:{sha256_hex(normalized_structure)}",
            "source": "project_owned_deterministic_template",
        },
    )


def _route_text(family: _RouteFamily, task_spec: TaskSpec, variant_index: int) -> str:
    prefix = _PREFIXES[variant_index % len(_PREFIXES)]
    verb = _VERBS[(variant_index // len(_PREFIXES)) % len(_VERBS)]
    verb_past = _VERBS_PAST[(variant_index // len(_PREFIXES)) % len(_VERBS_PAST)]
    punct = _PUNCTUATION[(variant_index // (len(_PREFIXES) * len(_VERBS))) % 5]
    if variant_index >= len(_PREFIXES) * len(_VERBS) * len(_PUNCTUATION):
        prefix = f"Wording variant {variant_index + 1}: {prefix}"
    return family.template.format(
        prefix=prefix,
        verb=verb,
        verb_past=verb_past,
        color=task_spec.target_object_id.removesuffix("_cube"),
        side=task_spec.target_bin_id.removesuffix("_bin"),
        punct=punct,
    )


def _rejection_text(family: _RejectionFamily, split: LanguageSplit, index: int) -> str:
    base = family.bases[split]
    if family.reason is RouterRejectionReason.EMPTY_TEXT:
        return " " * (index + 1)
    prefix = _REJECTION_PREFIXES[index % len(_REJECTION_PREFIXES)]
    suffix = _REJECTION_SUFFIXES[(index // len(_REJECTION_PREFIXES)) % len(_REJECTION_SUFFIXES)]
    if index >= len(_REJECTION_PREFIXES) * len(_REJECTION_SUFFIXES):
        prefix = f"Alternative malformed request {index + 1}: {prefix}"
    return f"{prefix}{base}{suffix}"


def _build_routeable(
    config: LanguageCorpusConfig,
    *,
    splits: tuple[LanguageSplit, ...],
    route_families: Mapping[LanguageSplit, tuple[_RouteFamily, ...]] = _ROUTE_FAMILIES,
) -> tuple[list[LanguageTemplateFamily], list[LanguageExample]]:
    families: list[LanguageTemplateFamily] = []
    examples: list[LanguageExample] = []
    for split in splits:
        specs = route_families[split]
        declared_families = tuple(
            _language_family(
                split=split,
                kind="route",
                slug=family.slug,
                category=family.category,
                structure_source=family.template,
                generator_version=config.generator_version,
            )
            for family in specs
        )
        families.extend(declared_families)
        for task_spec in CANONICAL_TASK_SPECS:
            for example_index in range(config.routeable_per_task[split]):
                family_index = example_index % len(specs)
                variant_index = example_index // len(specs)
                family_spec = specs[family_index]
                family = declared_families[family_index]
                raw_text = _route_text(family_spec, task_spec, variant_index)
                lexical_variant_ids = (
                    f"qualifier-v1:{variant_index % len(_PREFIXES)}",
                    f"verb-v1:{(variant_index // len(_PREFIXES)) % len(_VERBS)}",
                    f"surface-v1:{variant_index}",
                )
                provenance = {
                    "generator_version": config.generator_version,
                    "family_slug": family_spec.slug,
                    "variant_index": variant_index,
                    "source": "project_owned_deterministic_template",
                }
                normalized_text = normalize_language_text(raw_text)
                example_id = stable_language_example_id(
                    raw_text=raw_text,
                    normalized_text=normalized_text,
                    template_family_id=family.family_id,
                    lexical_variant_ids=lexical_variant_ids,
                    expected_status=RouterStatus.ROUTE,
                    expected_task_spec=task_spec,
                    expected_rejection_reason=None,
                    generation_provenance=provenance,
                    split=split,
                    schema_version=config.schema_version,
                )
                examples.append(
                    LanguageExample(
                        example_id=example_id,
                        raw_text=raw_text,
                        normalized_text=normalized_text,
                        template_family_id=family.family_id,
                        lexical_variant_ids=lexical_variant_ids,
                        expected_status=RouterStatus.ROUTE,
                        expected_task_spec=task_spec,
                        expected_rejection_reason=None,
                        generation_provenance=provenance,
                        split=split,
                        schema_version=config.schema_version,
                    )
                )
    return families, examples


def _build_rejections(
    config: LanguageCorpusConfig,
    *,
    splits: tuple[LanguageSplit, ...],
    rejection_families: tuple[_RejectionFamily, ...] = _REJECTION_FAMILIES,
) -> tuple[list[LanguageTemplateFamily], list[LanguageExample]]:
    families: list[LanguageTemplateFamily] = []
    examples: list[LanguageExample] = []
    for split in splits:
        total = config.rejected_total[split]
        split_family_specs = tuple(
            family
            for family in rejection_families
            if split is LanguageSplit.TRAIN or family.reason is not RouterRejectionReason.EMPTY_TEXT
        )
        quotient, remainder = divmod(total, len(split_family_specs))
        for family_index, family_spec in enumerate(split_family_specs):
            family = _language_family(
                split=split,
                kind="reject",
                slug=family_spec.reason.value,
                category=family_spec.category,
                structure_source=family_spec.bases[split],
                generator_version=config.generator_version,
            )
            families.append(family)
            family_count = quotient + (1 if family_index < remainder else 0)
            for variant_index in range(family_count):
                raw_text = _rejection_text(family_spec, split, variant_index)
                lexical_variant_ids = (
                    f"rejection-prefix-v1:{variant_index % len(_REJECTION_PREFIXES)}",
                    f"rejection-suffix-v1:{variant_index // len(_REJECTION_PREFIXES)}",
                )
                provenance = {
                    "generator_version": config.generator_version,
                    "family_slug": family_spec.reason.value,
                    "variant_index": variant_index,
                    "source": "project_owned_deterministic_template",
                }
                normalized_text = normalize_language_text(raw_text)
                example_id = stable_language_example_id(
                    raw_text=raw_text,
                    normalized_text=normalized_text,
                    template_family_id=family.family_id,
                    lexical_variant_ids=lexical_variant_ids,
                    expected_status=family_spec.status,
                    expected_task_spec=None,
                    expected_rejection_reason=family_spec.reason,
                    generation_provenance=provenance,
                    split=split,
                    schema_version=config.schema_version,
                )
                examples.append(
                    LanguageExample(
                        example_id=example_id,
                        raw_text=raw_text,
                        normalized_text=normalized_text,
                        template_family_id=family.family_id,
                        lexical_variant_ids=lexical_variant_ids,
                        expected_status=family_spec.status,
                        expected_task_spec=None,
                        expected_rejection_reason=family_spec.reason,
                        generation_provenance=provenance,
                        split=split,
                        schema_version=config.schema_version,
                    )
                )
    return families, examples


def _build_records(
    config: LanguageCorpusConfig,
    *,
    splits: tuple[LanguageSplit, ...],
    route_families: Mapping[LanguageSplit, tuple[_RouteFamily, ...]] = _ROUTE_FAMILIES,
    rejection_families: tuple[_RejectionFamily, ...] = _REJECTION_FAMILIES,
) -> tuple[tuple[LanguageTemplateFamily, ...], tuple[LanguageExample, ...]]:
    built_route_families, route_examples = _build_routeable(
        config,
        splits=splits,
        route_families=route_families,
    )
    reject_families, reject_examples = _build_rejections(
        config,
        splits=splits,
        rejection_families=rejection_families,
    )
    families = tuple(built_route_families + reject_families)
    examples = tuple(route_examples + reject_examples)
    return families, examples


def _sealed_final_example_id(*, kind: str, semantic_key: str, index: int) -> str:
    """Return an opaque, source-controlled semantic slot; no text is an input."""
    return M5A_SEALED_FINAL_EXAMPLE_ID_PREFIX + sha256_hex(
        {
            "seal_version": M5A_FINAL_LANGUAGE_SEAL_VERSION,
            "kind": kind,
            "semantic_key": semantic_key,
            "index": index,
        }
    )


def _sealed_final_rejection_counts(config: LanguageCorpusConfig) -> dict[str, int]:
    quotient, remainder = divmod(
        config.rejected_total[LanguageSplit.FINAL], len(_M5A_FINAL_REJECTION_REASONS)
    )
    return {
        reason.value: quotient + (1 if index < remainder else 0)
        for index, reason in enumerate(_M5A_FINAL_REJECTION_REASONS)
    }


def _sealed_final_split_manifest(config: LanguageCorpusConfig) -> LanguageSplitManifest:
    """Build the final aggregate lock exclusively from opaque semantic identities."""
    if config.to_dict() != DEFAULT_LANGUAGE_CORPUS_CONFIG.to_dict():
        raise FinalLanguageAccessError(
            "the source-controlled final language seal only supports the canonical corpus config"
        )
    routeable_count_by_task_id = {
        stable_task_id(task_spec): config.routeable_per_task[LanguageSplit.FINAL]
        for task_spec in CANONICAL_TASK_SPECS
    }
    rejection_count_by_reason = _sealed_final_rejection_counts(config)
    route_ids = tuple(
        _sealed_final_example_id(kind="route", semantic_key=task_id, index=index)
        for task_id, count in sorted(routeable_count_by_task_id.items())
        for index in range(count)
    )
    rejection_ids = tuple(
        _sealed_final_example_id(kind="reject", semantic_key=reason, index=index)
        for reason, count in sorted(rejection_count_by_reason.items())
        for index in range(count)
    )
    family_ids = tuple(
        sorted(
            (
                *(
                    _stable_family_id(split=LanguageSplit.FINAL, kind="route", slug=slug)
                    for slug in _M5A_FINAL_ROUTE_FAMILY_SLUGS
                ),
                *(
                    _stable_family_id(
                        split=LanguageSplit.FINAL,
                        kind="reject",
                        slug=reason.value,
                    )
                    for reason in _M5A_FINAL_REJECTION_REASONS
                ),
            )
        )
    )
    return LanguageSplitManifest(
        split=LanguageSplit.FINAL,
        example_count=len(route_ids) + len(rejection_ids),
        routeable_count_by_task_id=routeable_count_by_task_id,
        rejection_count_by_reason=rejection_count_by_reason,
        family_ids=family_ids,
        example_ids=tuple(sorted((*route_ids, *rejection_ids))),
        content_fingerprint=M5A_FINAL_LANGUAGE_CONTENT_FINGERPRINT,
        texts_sealed=True,
    )


def _sealed_final_routeable_ids_by_task(
    final_manifest: LanguageSplitManifest,
) -> dict[str, tuple[str, ...]]:
    return {
        task_id: tuple(
            _sealed_final_example_id(kind="route", semantic_key=task_id, index=index)
            for index in range(count)
        )
        for task_id, count in final_manifest.routeable_count_by_task_id.items()
    }


def _materialize_authorized_final_records(
    config: LanguageCorpusConfig,
) -> tuple[tuple[LanguageTemplateFamily, ...], tuple[LanguageExample, ...]]:
    """Run the final renderer only from the explicit future-final entry point."""
    from langmani.language._final_language_authority import (
        FINAL_REJECTION_BASES,
        FINAL_ROUTE_FAMILIES,
    )

    route_families = MappingProxyType(
        {
            LanguageSplit.FINAL: tuple(
                _RouteFamily(slug, category, template)
                for slug, category, template in FINAL_ROUTE_FAMILIES
            )
        }
    )
    rejection_families = tuple(
        _RejectionFamily(
            family.reason,
            family.status,
            family.category,
            MappingProxyType({LanguageSplit.FINAL: FINAL_REJECTION_BASES[family.reason.value]}),
        )
        for family in _REJECTION_FAMILIES
    )
    return _build_records(
        config,
        splits=(LanguageSplit.FINAL,),
        route_families=route_families,
        rejection_families=rejection_families,
    )


@dataclass(frozen=True, slots=True)
class GeneratedLanguageCorpus:
    """Development-safe corpus view; final command text is deliberately absent."""

    config: LanguageCorpusConfig
    manifest: LanguageCorpusManifest
    families: tuple[LanguageTemplateFamily, ...]
    visible_examples_by_split: Mapping[LanguageSplit, tuple[LanguageExample, ...]]
    routeable_example_ids_by_task: Mapping[LanguageSplit, Mapping[str, tuple[str, ...]]]
    isolation_report: SplitIsolationReport

    def __post_init__(self) -> None:
        if LanguageSplit.FINAL in self.visible_examples_by_split:
            raise FinalLanguageAccessError("final examples cannot be stored in a development view")
        if set(self.visible_examples_by_split) != {
            LanguageSplit.TRAIN,
            LanguageSplit.VALIDATION,
            LanguageSplit.DEVELOPMENT,
        }:
            raise ValueError(
                "development-safe corpus must expose train, validation, and development"
            )
        object.__setattr__(
            self,
            "visible_examples_by_split",
            MappingProxyType(dict(self.visible_examples_by_split)),
        )
        if set(self.routeable_example_ids_by_task) != set(LanguageSplit):
            raise ValueError("routeable ID index must contain every language split")
        frozen_index: dict[LanguageSplit, Mapping[str, tuple[str, ...]]] = {}
        for split, by_task in self.routeable_example_ids_by_task.items():
            if not isinstance(by_task, Mapping):
                raise TypeError("routeable ID task index must be a mapping")
            frozen_index[split] = MappingProxyType(
                {task_id: tuple(example_ids) for task_id, example_ids in by_task.items()}
            )
        object.__setattr__(
            self,
            "routeable_example_ids_by_task",
            MappingProxyType(frozen_index),
        )

    @property
    def examples(self) -> tuple[LanguageExample, ...]:
        """Flatten only the three development-safe partitions."""
        return tuple(
            example
            for split in (
                LanguageSplit.TRAIN,
                LanguageSplit.VALIDATION,
                LanguageSplit.DEVELOPMENT,
            )
            for example in self.visible_examples_by_split[split]
        )

    def examples_for_split(self, split: LanguageSplit) -> tuple[LanguageExample, ...]:
        if split is LanguageSplit.FINAL:
            raise FinalLanguageAccessError(
                "m5a_language_final_v0 is sealed; explicit final authorization is required"
            )
        return self.visible_examples_by_split[split]


def build_language_corpus(
    config: LanguageCorpusConfig = DEFAULT_LANGUAGE_CORPUS_CONFIG,
) -> GeneratedLanguageCorpus:
    """Build the fixed corpus and return a view that cannot expose final text."""
    if not isinstance(config, LanguageCorpusConfig):
        raise TypeError("config must be a LanguageCorpusConfig")
    families, visible_examples = _build_records(config, splits=_DEVELOPMENT_VISIBLE_SPLITS)
    visible_isolation = validate_family_split_isolation(families, visible_examples)
    partitions = partition_examples(visible_examples)
    split_manifests = {
        split: build_language_split_manifest(
            split=split,
            families=families,
            examples=visible_examples,
        )
        for split in _DEVELOPMENT_VISIBLE_SPLITS
    }
    for split in _DEVELOPMENT_VISIBLE_SPLITS:
        if (
            split_manifests[split].content_fingerprint
            != M5A_LANGUAGE_SPLIT_CONTENT_FINGERPRINTS[split]
        ):
            raise FinalLanguageAccessError(
                f"{split.value} corpus drift invalidates the source-controlled final isolation seal"
            )
    split_manifests[LanguageSplit.FINAL] = _sealed_final_split_manifest(config)
    isolation = bind_sealed_final_split_isolation(
        visible_report=visible_isolation,
        final_manifest=split_manifests[LanguageSplit.FINAL],
        source_isolation_attestation=(M5A_FINAL_LANGUAGE_ISOLATION_ATTESTATION_FINGERPRINT),
    )
    config_fingerprint = f"sha256:{sha256_hex(config.to_dict())}"
    corpus_payload = {
        "config_fingerprint": config_fingerprint,
        "split_manifests": {
            split.value: split_manifests[split].to_dict() for split in LanguageSplit
        },
        "isolation_report_fingerprint": isolation.report_fingerprint,
    }
    manifest = LanguageCorpusManifest(
        corpus_fingerprint=f"sha256:{sha256_hex(corpus_payload)}",
        config_fingerprint=config_fingerprint,
        split_manifests=split_manifests,
        family_isolation_validated=True,
        final_locked=True,
    )
    routeable_ids: dict[LanguageSplit, dict[str, tuple[str, ...]]] = {}
    for split in _DEVELOPMENT_VISIBLE_SPLITS:
        task_ids = split_manifests[split].routeable_count_by_task_id
        routeable_ids[split] = {
            task_id: tuple(
                sorted(
                    example.example_id
                    for example in partitions[split]
                    if example.task_id == task_id
                )
            )
            for task_id in task_ids
        }
    routeable_ids[LanguageSplit.FINAL] = _sealed_final_routeable_ids_by_task(
        split_manifests[LanguageSplit.FINAL]
    )
    return GeneratedLanguageCorpus(
        config=config,
        manifest=manifest,
        families=families,
        visible_examples_by_split={
            split: partitions[split]
            for split in (
                LanguageSplit.TRAIN,
                LanguageSplit.VALIDATION,
                LanguageSplit.DEVELOPMENT,
            )
        },
        routeable_example_ids_by_task=routeable_ids,
        isolation_report=isolation,
    )


def materialize_final_language_examples(
    corpus: GeneratedLanguageCorpus,
    *,
    authorize_final: bool = False,
) -> tuple[LanguageExample, ...]:
    """Regenerate final text only behind an explicit later-milestone authorization."""
    if authorize_final is not True:
        raise FinalLanguageAccessError(
            "final language access is forbidden during M5A target-development"
        )
    families, examples = _materialize_authorized_final_records(corpus.config)
    final_examples = tuple(example for example in examples if example.split is LanguageSplit.FINAL)
    regenerated = build_language_split_manifest(
        split=LanguageSplit.FINAL,
        families=families,
        examples=examples,
    )
    locked = corpus.manifest.split_manifests[LanguageSplit.FINAL]
    if regenerated.content_fingerprint != locked.content_fingerprint:
        raise FinalLanguageAccessError("regenerated final corpus does not match its sealed lock")
    return tuple(sorted(final_examples, key=lambda value: value.example_id))


def materialize_final_language_slot_map(
    corpus: GeneratedLanguageCorpus,
    *,
    authorize_final: bool = False,
) -> Mapping[str, LanguageExample]:
    """Resolve sealed semantic slots only inside a separately authorized final run."""
    examples = materialize_final_language_examples(corpus, authorize_final=authorize_final)
    final_manifest = corpus.manifest.split_manifests[LanguageSplit.FINAL]
    mapping: dict[str, LanguageExample] = {}
    for task_id, count in sorted(final_manifest.routeable_count_by_task_id.items()):
        slots = tuple(
            _sealed_final_example_id(kind="route", semantic_key=task_id, index=index)
            for index in range(count)
        )
        candidates = tuple(
            sorted(
                (item for item in examples if item.task_id == task_id),
                key=lambda item: item.example_id,
            )
        )
        if len(candidates) != len(slots):
            raise FinalLanguageAccessError("final route slot count differs from rendered corpus")
        mapping.update(zip(slots, candidates, strict=True))
    for reason, count in sorted(final_manifest.rejection_count_by_reason.items()):
        slots = tuple(
            _sealed_final_example_id(kind="reject", semantic_key=reason, index=index)
            for index in range(count)
        )
        candidates = tuple(
            sorted(
                (
                    item
                    for item in examples
                    if item.expected_rejection_reason is not None
                    and item.expected_rejection_reason.value == reason
                ),
                key=lambda item: item.example_id,
            )
        )
        if len(candidates) != len(slots):
            raise FinalLanguageAccessError(
                "final rejection slot count differs from rendered corpus"
            )
        mapping.update(zip(slots, candidates, strict=True))
    if set(mapping) != set(final_manifest.example_ids):
        raise FinalLanguageAccessError("final semantic slot map differs from the sealed manifest")
    return MappingProxyType(dict(sorted(mapping.items())))


def language_corpus_counts(
    corpus: GeneratedLanguageCorpus,
) -> Mapping[LanguageSplit, Mapping[str, int]]:
    """Return manifest counts without materializing sealed final command text."""
    result: dict[LanguageSplit, Mapping[str, int]] = {}
    for split in LanguageSplit:
        manifest = corpus.manifest.split_manifests[split]
        result[split] = MappingProxyType(
            {
                "routeable": sum(manifest.routeable_count_by_task_id.values()),
                "rejected": sum(manifest.rejection_count_by_reason.values()),
                "total": manifest.example_count,
            }
        )
    return MappingProxyType(result)
