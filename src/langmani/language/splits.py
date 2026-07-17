"""Family-level isolation and immutable split manifests for the M5A corpus."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from langmani.datasets.identity import sha256_hex
from langmani.language.router_types import (
    LanguageExample,
    LanguageSplit,
    LanguageSplitManifest,
    LanguageTemplateFamily,
    RouterStatus,
)


class LanguageSplitIsolationError(ValueError):
    """Raised when structural or lexical content leaks across language splits."""


@dataclass(frozen=True, slots=True)
class SplitIsolationReport:
    """Small immutable proof that all declared isolation checks were executed."""

    passed: bool
    family_count_by_split: Mapping[LanguageSplit, int]
    example_count_by_split: Mapping[LanguageSplit, int]
    report_fingerprint: str

    def __post_init__(self) -> None:
        if self.passed is not True:
            raise LanguageSplitIsolationError("a split isolation report may only represent success")
        object.__setattr__(
            self,
            "family_count_by_split",
            MappingProxyType(dict(self.family_count_by_split)),
        )
        object.__setattr__(
            self,
            "example_count_by_split",
            MappingProxyType(dict(self.example_count_by_split)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "family_count_by_split": {
                split.value: self.family_count_by_split.get(split, 0) for split in LanguageSplit
            },
            "example_count_by_split": {
                split.value: self.example_count_by_split.get(split, 0) for split in LanguageSplit
            },
            "report_fingerprint": self.report_fingerprint,
        }


def _assert_owned_by_one_split(
    *,
    name: str,
    values: Iterable[tuple[str, LanguageSplit]],
) -> None:
    split_by_value: dict[str, LanguageSplit] = {}
    for value, split in values:
        owner = split_by_value.setdefault(value, split)
        if owner is not split:
            raise LanguageSplitIsolationError(
                f"{name} {value!r} leaks across {owner.value} and {split.value}"
            )


def validate_family_split_isolation(
    families: Sequence[LanguageTemplateFamily],
    examples: Sequence[LanguageExample],
) -> SplitIsolationReport:
    """Reject family, structural, near-duplicate, and normalized-text leakage."""
    family_by_id: dict[str, LanguageTemplateFamily] = {}
    for family in families:
        existing = family_by_id.get(family.family_id)
        if existing is not None:
            if existing != family:
                raise LanguageSplitIsolationError(
                    f"family_id {family.family_id!r} has conflicting declarations"
                )
            raise LanguageSplitIsolationError(f"duplicate family_id {family.family_id!r}")
        family_by_id[family.family_id] = family

    _assert_owned_by_one_split(
        name="structural signature",
        values=((family.structural_signature, family.split) for family in families),
    )
    _assert_owned_by_one_split(
        name="near-duplicate group",
        values=((family.near_duplicate_group_id, family.split) for family in families),
    )
    _assert_owned_by_one_split(
        name="normalized text",
        values=((example.normalized_text, example.split) for example in examples),
    )

    seen_example_ids: set[str] = set()
    for example in examples:
        if example.example_id in seen_example_ids:
            raise LanguageSplitIsolationError(f"duplicate example_id {example.example_id!r}")
        seen_example_ids.add(example.example_id)
        family = family_by_id.get(example.template_family_id)
        if family is None:
            raise LanguageSplitIsolationError(
                f"example {example.example_id!r} references an unknown template family"
            )
        if family.split is not example.split:
            raise LanguageSplitIsolationError(
                f"example {example.example_id!r} crosses its template-family split"
            )

    family_counts = Counter(family.split for family in families)
    example_counts = Counter(example.split for example in examples)
    payload = {
        "family_isolation_claims_by_split": {
            split.value: [
                {
                    "family_id": family.family_id,
                    "structural_signature": family.structural_signature,
                    "near_duplicate_group_id": family.near_duplicate_group_id,
                }
                for family in sorted(
                    (candidate for candidate in families if candidate.split is split),
                    key=lambda candidate: candidate.family_id,
                )
            ]
            for split in LanguageSplit
        },
        "example_ids_by_split": {
            split.value: sorted(
                example.example_id for example in examples if example.split is split
            )
            for split in LanguageSplit
        },
        "checks": [
            "family_id_unique",
            "structural_signature_cross_split_disjoint",
            "near_duplicate_group_cross_split_disjoint",
            "normalized_text_cross_split_disjoint",
            "example_family_split_consistent",
        ],
    }
    return SplitIsolationReport(
        passed=True,
        family_count_by_split={split: family_counts[split] for split in LanguageSplit},
        example_count_by_split={split: example_counts[split] for split in LanguageSplit},
        report_fingerprint=f"sha256:{sha256_hex(payload)}",
    )


def bind_sealed_final_split_isolation(
    *,
    visible_report: SplitIsolationReport,
    final_manifest: LanguageSplitManifest,
    source_isolation_attestation: str,
) -> SplitIsolationReport:
    """Bind an opaque final-split lock without opening its command text.

    The development corpus validates train, validation, and development records
    directly.  Final records are intentionally absent, so their source-controlled
    manifest is bound as an opaque fourth partition instead of being regenerated.
    """
    if final_manifest.split is not LanguageSplit.FINAL or not final_manifest.texts_sealed:
        raise LanguageSplitIsolationError("final isolation binding requires a sealed final lock")
    if visible_report.example_count_by_split.get(LanguageSplit.FINAL, 0) != 0:
        raise LanguageSplitIsolationError(
            "visible isolation report unexpectedly contains final text"
        )
    if visible_report.family_count_by_split.get(LanguageSplit.FINAL, 0) != 0:
        raise LanguageSplitIsolationError(
            "visible isolation report unexpectedly contains final template families"
        )
    digest = source_isolation_attestation.removeprefix("sha256:")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise LanguageSplitIsolationError("source isolation attestation must be a SHA-256 digest")
    family_counts = dict(visible_report.family_count_by_split)
    example_counts = dict(visible_report.example_count_by_split)
    family_counts[LanguageSplit.FINAL] = len(final_manifest.family_ids)
    example_counts[LanguageSplit.FINAL] = final_manifest.example_count
    payload = {
        "visible_isolation_report_fingerprint": visible_report.report_fingerprint,
        "sealed_final": {
            "split_fingerprint": final_manifest.content_fingerprint,
            "family_ids": list(final_manifest.family_ids),
            "example_ids": list(final_manifest.example_ids),
            "texts_sealed": True,
        },
        "binding_version": "m5a-opaque-final-isolation-v1",
        "source_isolation_attestation": source_isolation_attestation,
    }
    return SplitIsolationReport(
        passed=True,
        family_count_by_split=family_counts,
        example_count_by_split=example_counts,
        report_fingerprint=f"sha256:{sha256_hex(payload)}",
    )


def build_language_split_manifest(
    *,
    split: LanguageSplit,
    families: Sequence[LanguageTemplateFamily],
    examples: Sequence[LanguageExample],
) -> LanguageSplitManifest:
    """Build a content-bound manifest without embedding command text."""
    split_families = tuple(family for family in families if family.split is split)
    split_examples = tuple(example for example in examples if example.split is split)
    task_counts = Counter(
        example.task_id
        for example in split_examples
        if example.expected_status is RouterStatus.ROUTE and example.task_id is not None
    )
    rejection_counts = Counter(
        example.expected_rejection_reason.value
        for example in split_examples
        if example.expected_rejection_reason is not None
    )
    family_ids = tuple(sorted(family.family_id for family in split_families))
    example_ids = tuple(sorted(example.example_id for example in split_examples))
    content_payload = {
        "split": split.value,
        "families": [
            family.to_dict() for family in sorted(split_families, key=lambda x: x.family_id)
        ],
        "examples": [
            example.to_dict() for example in sorted(split_examples, key=lambda x: x.example_id)
        ],
    }
    return LanguageSplitManifest(
        split=split,
        example_count=len(split_examples),
        routeable_count_by_task_id=dict(task_counts),
        rejection_count_by_reason=dict(rejection_counts),
        family_ids=family_ids,
        example_ids=example_ids,
        content_fingerprint=f"sha256:{sha256_hex(content_payload)}",
        texts_sealed=split is LanguageSplit.FINAL,
    )


def partition_examples(
    examples: Sequence[LanguageExample],
) -> Mapping[LanguageSplit, tuple[LanguageExample, ...]]:
    """Return deterministically ordered, immutable language partitions."""
    buckets: dict[LanguageSplit, list[LanguageExample]] = defaultdict(list)
    for example in examples:
        buckets[example.split].append(example)
    return MappingProxyType(
        {
            split: tuple(sorted(buckets[split], key=lambda value: value.example_id))
            for split in LanguageSplit
        }
    )
