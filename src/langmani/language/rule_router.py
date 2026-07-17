"""Deterministic, auditable RuleRouterV0 language baseline."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field

from langmani.datasets.identity import sha256_hex
from langmani.environments.specs import TaskSpec
from langmani.language.router_types import (
    RouterConfidence,
    RouterDecision,
    RouterRejectionReason,
    RouterStatus,
)

RULE_ROUTER_NAME = "RuleRouterV0"
RULE_ROUTER_VERSION = "rule-router-v0"
_WORD_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
_MULTI_TASK_PATTERN = re.compile(r"\b(?:and\s+then|then|followed\s+by|after\s+that)\b")
_UNRESOLVED_CORRECTION_PATTERN = re.compile(
    r"\b(?:actually\s+never\s+mind|never\s+mind|forget\s+(?:it|that))\b"
)
_NEGATION_TOKENS = frozenset({"ignore", "not", "never", "don't", "dont", "except"})


@dataclass(frozen=True, slots=True)
class RuleRouterConfig:
    """Frozen lexical tables and precedence used by RuleRouterV0."""

    object_synonyms: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "red_cube": ("red cube", "red block", "red object", "crimson cube"),
            "green_cube": ("green cube", "green block", "green object", "emerald cube"),
            "blue_cube": ("blue cube", "blue block", "blue object", "azure cube"),
        }
    )
    bin_synonyms: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "left_bin": (
                "left bin",
                "left container",
                "left receptacle",
                "bin on the left",
                "container on the left",
            ),
            "right_bin": (
                "right bin",
                "right container",
                "right receptacle",
                "bin on the right",
                "container on the right",
            ),
        }
    )
    unsupported_object_terms: tuple[str, ...] = (
        "yellow cube",
        "orange cube",
        "purple cube",
        "black cube",
        "white cube",
        "ball",
        "cylinder",
    )
    unsupported_destination_terms: tuple[str, ...] = (
        "drawer",
        "shelf",
        "basket",
        "table",
        "over there",
    )
    unsupported_action_terms: tuple[str, ...] = (
        "open the",
        "close the",
        "stack",
        "push",
        "pull",
        "rotate",
        "throw",
    )

    def __post_init__(self) -> None:
        if tuple(self.object_synonyms) != ("red_cube", "green_cube", "blue_cube"):
            raise ValueError("rule object table must use canonical red/green/blue order")
        if tuple(self.bin_synonyms) != ("left_bin", "right_bin"):
            raise ValueError("rule bin table must use canonical left/right order")
        for table in (self.object_synonyms, self.bin_synonyms):
            if not all(
                isinstance(values, tuple)
                and values
                and all(isinstance(value, str) and value for value in values)
                for values in table.values()
            ):
                raise ValueError("rule synonym tables require non-empty tuples of terms")

    @property
    def fingerprint(self) -> str:
        return f"sha256:{sha256_hex(self.to_dict())}"

    def to_dict(self) -> dict[str, object]:
        return {
            "router_name": RULE_ROUTER_NAME,
            "router_version": RULE_ROUTER_VERSION,
            "object_synonyms": {key: list(value) for key, value in self.object_synonyms.items()},
            "bin_synonyms": {key: list(value) for key, value in self.bin_synonyms.items()},
            "unsupported_object_terms": list(self.unsupported_object_terms),
            "unsupported_destination_terms": list(self.unsupported_destination_terms),
            "unsupported_action_terms": list(self.unsupported_action_terms),
        }


def normalize_rule_text(value: str) -> str:
    """NFKC/lowercase/punctuation separation without deleting negation words."""

    normalized = unicodedata.normalize("NFKC", value).lower().replace("\u2019", "'")
    tokens = _WORD_PATTERN.findall(normalized)
    return " ".join(tokens)


def _contains_control_character(value: str) -> bool:
    return any(
        unicodedata.category(character) == "Cc" and character not in "\t\r\n" for character in value
    )


def _term_matches(
    normalized: str,
    table: Mapping[str, tuple[str, ...]],
) -> list[dict[str, object]]:
    matches: list[dict[str, object]] = []
    for semantic_id, terms in table.items():
        for term in terms:
            pattern = re.compile(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])")
            for match in pattern.finditer(normalized):
                prefix_tokens = normalized[: match.start()].split()
                window = prefix_tokens[-4:]
                negated = bool(set(window) & _NEGATION_TOKENS) or window[-2:] == ["do", "not"]
                matches.append(
                    {
                        "semantic_id": semantic_id,
                        "term": term,
                        "start": match.start(),
                        "end": match.end(),
                        "negated": negated,
                    }
                )
    return sorted(matches, key=lambda value: (int(value["start"]), str(value["semantic_id"])))


def _contains_term(normalized: str, terms: tuple[str, ...]) -> list[str]:
    return [
        term
        for term in terms
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", normalized)
    ]


class RuleRouterV0:
    """Route exactly one positive object and bin, otherwise reject safely."""

    def __init__(self, config: RuleRouterConfig | None = None) -> None:
        self.config = RuleRouterConfig() if config is None else config
        if not isinstance(self.config, RuleRouterConfig):
            raise TypeError("config must be RuleRouterConfig")

    def route(self, command: str) -> RouterDecision:
        if not isinstance(command, str):
            raise TypeError("command must be text")
        normalized = normalize_rule_text(command)
        base_evidence: dict[str, object] = {
            "config_fingerprint": self.config.fingerprint,
            "normalized_text": normalized,
        }
        if not command.strip():
            return self._reject(
                RouterStatus.REJECT_MALFORMED,
                RouterRejectionReason.EMPTY_TEXT,
                {**base_evidence, "rejection_rule": "empty_text"},
            )
        if _contains_control_character(command):
            return self._reject(
                RouterStatus.REJECT_MALFORMED,
                RouterRejectionReason.MALFORMED_CONTROL_CHARACTERS,
                {**base_evidence, "rejection_rule": "control_character"},
            )
        if not normalized or not any(character.isalpha() for character in normalized):
            return self._reject(
                RouterStatus.REJECT_MALFORMED,
                RouterRejectionReason.MEANINGLESS_TEXT,
                {**base_evidence, "rejection_rule": "meaningless_text"},
            )

        object_matches = _term_matches(normalized, self.config.object_synonyms)
        bin_matches = _term_matches(normalized, self.config.bin_synonyms)
        evidence = {
            **base_evidence,
            "matched_object_spans": object_matches,
            "matched_bin_spans": bin_matches,
            "negation_indicators": [
                value for value in (*object_matches, *bin_matches) if bool(value["negated"])
            ],
        }
        if _UNRESOLVED_CORRECTION_PATTERN.search(normalized):
            return self._reject(
                RouterStatus.REJECT_AMBIGUOUS,
                RouterRejectionReason.UNRESOLVED_CORRECTION,
                {**evidence, "rejection_rule": "unresolved_correction"},
            )

        unsupported_objects = _contains_term(normalized, self.config.unsupported_object_terms)
        unsupported_destinations = _contains_term(
            normalized, self.config.unsupported_destination_terms
        )
        unsupported_actions = _contains_term(normalized, self.config.unsupported_action_terms)
        if unsupported_objects:
            return self._reject(
                RouterStatus.REJECT_UNSUPPORTED,
                RouterRejectionReason.UNSUPPORTED_OBJECT,
                {
                    **evidence,
                    "unsupported_terms": unsupported_objects,
                    "rejection_rule": "unsupported_object",
                },
            )
        if unsupported_destinations:
            return self._reject(
                RouterStatus.REJECT_UNSUPPORTED,
                RouterRejectionReason.UNSUPPORTED_DESTINATION,
                {
                    **evidence,
                    "unsupported_terms": unsupported_destinations,
                    "rejection_rule": "unsupported_destination",
                },
            )
        if unsupported_actions:
            return self._reject(
                RouterStatus.REJECT_UNSUPPORTED,
                RouterRejectionReason.UNSUPPORTED_ACTION,
                {
                    **evidence,
                    "unsupported_terms": unsupported_actions,
                    "rejection_rule": "unsupported_action",
                },
            )

        positive_objects = {
            str(value["semantic_id"]) for value in object_matches if not bool(value["negated"])
        }
        positive_bins = {
            str(value["semantic_id"]) for value in bin_matches if not bool(value["negated"])
        }
        negated_objects = {
            str(value["semantic_id"]) for value in object_matches if bool(value["negated"])
        }
        negated_bins = {
            str(value["semantic_id"]) for value in bin_matches if bool(value["negated"])
        }
        if _MULTI_TASK_PATTERN.search(normalized) and (
            len(positive_objects) > 1 or len(positive_bins) > 1
        ):
            return self._reject(
                RouterStatus.REJECT_AMBIGUOUS,
                RouterRejectionReason.MULTIPLE_TASKS,
                {**evidence, "rejection_rule": "multiple_sequential_tasks"},
            )
        if len(positive_objects) > 1:
            return self._reject(
                RouterStatus.REJECT_AMBIGUOUS,
                RouterRejectionReason.CONFLICTING_OBJECTS,
                {
                    **evidence,
                    "object_conflicts": sorted(positive_objects),
                    "rejection_rule": "conflicting_objects",
                },
            )
        if len(positive_bins) > 1:
            return self._reject(
                RouterStatus.REJECT_AMBIGUOUS,
                RouterRejectionReason.CONFLICTING_BINS,
                {
                    **evidence,
                    "bin_conflicts": sorted(positive_bins),
                    "rejection_rule": "conflicting_bins",
                },
            )
        if (negated_objects and not positive_objects) or (negated_bins and not positive_bins):
            return self._reject(
                RouterStatus.REJECT_AMBIGUOUS,
                RouterRejectionReason.CONTRADICTORY_NEGATION,
                {**evidence, "rejection_rule": "contradictory_negation"},
            )
        if not positive_objects:
            return self._reject(
                RouterStatus.REJECT_AMBIGUOUS,
                RouterRejectionReason.MISSING_OBJECT,
                {**evidence, "rejection_rule": "missing_object"},
            )
        if not positive_bins:
            return self._reject(
                RouterStatus.REJECT_AMBIGUOUS,
                RouterRejectionReason.MISSING_DESTINATION,
                {**evidence, "rejection_rule": "missing_destination"},
            )

        object_id = next(iter(positive_objects))
        bin_id = next(iter(positive_bins))
        task_spec = TaskSpec.from_mapping(
            {
                "target_object_id": object_id,
                "target_bin_id": bin_id,
                "instruction_template_id": "canonical_v0",
            }
        )
        return RouterDecision.route(
            task_spec=task_spec,
            confidence=RouterConfidence.unavailable(
                definition="deterministic lexical rules do not produce a probability"
            ),
            router_name=RULE_ROUTER_NAME,
            router_version=RULE_ROUTER_VERSION,
            evidence={**evidence, "rejection_rule": None},
        )

    @staticmethod
    def _reject(
        status: RouterStatus,
        reason: RouterRejectionReason,
        evidence: Mapping[str, object],
    ) -> RouterDecision:
        return RouterDecision.reject(
            status=status,
            rejection_reason=reason,
            confidence=RouterConfidence.unavailable(
                definition="deterministic lexical rules do not produce a probability"
            ),
            router_name=RULE_ROUTER_NAME,
            router_version=RULE_ROUTER_VERSION,
            evidence=evidence,
        )


__all__ = [
    "RULE_ROUTER_NAME",
    "RULE_ROUTER_VERSION",
    "RuleRouterConfig",
    "RuleRouterV0",
    "normalize_rule_text",
]
