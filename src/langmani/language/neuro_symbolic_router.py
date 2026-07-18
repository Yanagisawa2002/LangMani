"""Frozen neuro-symbolic language routing for the M5A.4 offline experiment."""

from __future__ import annotations

import json
import re
import time
import unicodedata
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from typing import Annotated, Literal, Protocol, cast

import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from langmani.datasets.identity import canonical_json, sha256_hex
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import BIN_IDS, OBJECT_IDS, BinId, ObjectId, TaskSpec
from langmani.language.llm_router import (
    QWEN3_4B_INSTRUCT_MODEL_ID,
    QWEN3_4B_INSTRUCT_REVISION,
    StructuredLLMRouterConfig,
    StructuredLLMRouterError,
    TransformersLocalTextGenerator,
)
from langmani.language.router_types import (
    LanguageExample,
    LanguageSplit,
    RouterConfidence,
    RouterDecision,
    RouterRejectionReason,
    RouterStatus,
)
from langmani.language.rule_router import RuleRouterConfig, normalize_rule_text

NEURO_SYMBOLIC_ROUTER_NAME = "NeuroSymbolicRouterV0"
NEURO_SYMBOLIC_ROUTER_VERSION = "neuro-symbolic-router-v0"
SYMBOLIC_FRAME_VERSION = "symbolic-lexical-frame-v0"
SEMANTIC_FRAME_SCHEMA_VERSION = "qwen-semantic-frame-v0"
SEMANTIC_PROMPT_VERSION = "m5a4-semantic-frame-prompt-v0"
ARBITER_VERSION = "deterministic-safety-arbiter-v0"
OUTLINES_VERSION = "1.3.1"
OUTLINES_LICENSE = "Apache-2.0"

_WORD_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
_MULTI_TASK_PATTERN = re.compile(r"\b(?:and\s+then|then|followed\s+by|after\s+that)\b")
_UNRESOLVED_CORRECTION_PATTERN = re.compile(
    r"\b(?:actually\s+never\s+mind|never\s+mind|forget\s+(?:it|that))\b"
)
_CORRECTION_PATTERN = re.compile(r"\b(?:actually|correction|instead|rather|change)\b")
_RESOLVED_CORRECTION_PATTERN = re.compile(
    r"\b(?:correction\s+to\s+the\s+wording|actually\s+use|instead\s+use|move\s+only)\b"
)
_NEGATION_TOKENS = frozenset({"ignore", "not", "never", "don't", "dont", "except"})
_SUPPORTED_ACTION_TERMS = (
    "pick up",
    "pick",
    "place",
    "put",
    "move",
    "transfer",
    "relocate",
    "deposit",
)
_SPATIAL_REFERENCE_TERMS = (
    "over there",
    "where i pointed",
    "where i point",
    "this side",
    "that side",
)


class NeuroSymbolicContractError(ValueError):
    """Raised when a frozen M5A.4 contract or frame is inconsistent."""


@dataclass(frozen=True, slots=True)
class MatchedLexicalSpanV0:
    """One normalized lexical match retained for audit."""

    category: str
    semantic_value: str
    matched_text: str
    start: int
    end: int
    negated: bool = False

    def __post_init__(self) -> None:
        if not self.category or not self.semantic_value or not self.matched_text:
            raise NeuroSymbolicContractError("matched spans require non-empty text fields")
        if self.start < 0 or self.end <= self.start:
            raise NeuroSymbolicContractError("matched span offsets are invalid")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SymbolicLexicalFrameV0:
    """Immutable deterministic facts; this frame cannot dispatch a controller."""

    supported_objects: tuple[ObjectId, ...]
    unsupported_object_mentions: tuple[str, ...]
    supported_bins: tuple[BinId, ...]
    unsupported_destination_mentions: tuple[str, ...]
    supported_action_mentions: tuple[str, ...]
    unsupported_action_mentions: tuple[str, ...]
    spatial_reference_mentions: tuple[str, ...]
    negation_present: bool
    contradictory_negation: bool
    correction_present: bool
    correction_resolved: bool
    multiple_sequential_tasks: bool
    empty_input: bool
    malformed_control_characters: bool
    meaningless_or_noise: bool
    matched_spans: tuple[MatchedLexicalSpanV0, ...]
    normalized_text: str
    schema_version: str = SYMBOLIC_FRAME_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SYMBOLIC_FRAME_VERSION:
            raise NeuroSymbolicContractError("unknown symbolic frame version")
        if any(value not in OBJECT_IDS for value in self.supported_objects):
            raise NeuroSymbolicContractError("symbolic frame contains an unknown object")
        if any(value not in BIN_IDS for value in self.supported_bins):
            raise NeuroSymbolicContractError("symbolic frame contains an unknown bin")
        if len(set(self.supported_objects)) != len(self.supported_objects):
            raise NeuroSymbolicContractError("symbolic objects must be unique")
        if len(set(self.supported_bins)) != len(self.supported_bins):
            raise NeuroSymbolicContractError("symbolic bins must be unique")

    @property
    def fingerprint(self) -> str:
        return f"sha256:{sha256_hex(self.to_dict())}"

    def to_dict(self) -> dict[str, object]:
        return {
            "supported_objects": list(self.supported_objects),
            "unsupported_object_mentions": list(self.unsupported_object_mentions),
            "supported_bins": list(self.supported_bins),
            "unsupported_destination_mentions": list(self.unsupported_destination_mentions),
            "supported_action_mentions": list(self.supported_action_mentions),
            "unsupported_action_mentions": list(self.unsupported_action_mentions),
            "spatial_reference_mentions": list(self.spatial_reference_mentions),
            "negation_present": self.negation_present,
            "contradictory_negation": self.contradictory_negation,
            "correction_present": self.correction_present,
            "correction_resolved": self.correction_resolved,
            "multiple_sequential_tasks": self.multiple_sequential_tasks,
            "empty_input": self.empty_input,
            "malformed_control_characters": self.malformed_control_characters,
            "meaningless_or_noise": self.meaningless_or_noise,
            "matched_spans": [value.to_dict() for value in self.matched_spans],
            "normalized_text": self.normalized_text,
            "schema_version": self.schema_version,
        }


def _control_character_present(value: str) -> bool:
    return any(
        unicodedata.category(character) == "Cc" and character not in "\t\r\n" for character in value
    )


def _find_phrase_spans(
    normalized: str,
    *,
    category: str,
    semantic_value: str,
    phrases: Sequence[str],
    permit_negation: bool,
) -> list[MatchedLexicalSpanV0]:
    result: list[MatchedLexicalSpanV0] = []
    for phrase in phrases:
        pattern = re.compile(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])")
        for match in pattern.finditer(normalized):
            prefix = normalized[: match.start()].split()
            window = prefix[-4:]
            negated = permit_negation and (
                bool(set(window) & _NEGATION_TOKENS) or window[-2:] == ["do", "not"]
            )
            result.append(
                MatchedLexicalSpanV0(
                    category=category,
                    semantic_value=semantic_value,
                    matched_text=phrase,
                    start=match.start(),
                    end=match.end(),
                    negated=negated,
                )
            )
    return result


class SymbolicLexicalParserV0:
    """Project-owned lexical fact extractor with no corpus-label input."""

    def __init__(self, config: RuleRouterConfig | None = None) -> None:
        self.config = RuleRouterConfig() if config is None else config
        self.contract_fingerprint = f"sha256:{sha256_hex(self.contract_dict())}"

    def contract_dict(self) -> dict[str, object]:
        return {
            "schema_version": SYMBOLIC_FRAME_VERSION,
            "rule_config": self.config.to_dict(),
            "supported_action_terms": list(_SUPPORTED_ACTION_TERMS),
            "spatial_reference_terms": list(_SPATIAL_REFERENCE_TERMS),
            "precedence_free_fact_extraction": True,
            "template_family_input": False,
            "expected_label_input": False,
            "split_identity_input": False,
            "controller_dispatch": False,
        }

    def parse(self, command: str) -> SymbolicLexicalFrameV0:
        if not isinstance(command, str):
            raise TypeError("command must be text")
        normalized = normalize_rule_text(command)
        spans: list[MatchedLexicalSpanV0] = []
        for semantic_id, phrases in self.config.object_synonyms.items():
            spans.extend(
                _find_phrase_spans(
                    normalized,
                    category="supported_object",
                    semantic_value=semantic_id,
                    phrases=phrases,
                    permit_negation=True,
                )
            )
        for semantic_id, phrases in self.config.bin_synonyms.items():
            spans.extend(
                _find_phrase_spans(
                    normalized,
                    category="supported_bin",
                    semantic_value=semantic_id,
                    phrases=phrases,
                    permit_negation=True,
                )
            )
        lexical_groups = (
            (
                "unsupported_object",
                self.config.unsupported_object_terms,
                False,
            ),
            (
                "unsupported_destination",
                self.config.unsupported_destination_terms,
                False,
            ),
            ("unsupported_action", self.config.unsupported_action_terms, False),
            ("supported_action", _SUPPORTED_ACTION_TERMS, False),
            ("spatial_reference", _SPATIAL_REFERENCE_TERMS, False),
        )
        for category, phrases, permit_negation in lexical_groups:
            for phrase in phrases:
                spans.extend(
                    _find_phrase_spans(
                        normalized,
                        category=category,
                        semantic_value=phrase,
                        phrases=(phrase,),
                        permit_negation=permit_negation,
                    )
                )
        spans.sort(key=lambda value: (value.start, value.end, value.category))
        objects = tuple(
            cast(ObjectId, value)
            for value in OBJECT_IDS
            if any(
                span.category == "supported_object"
                and span.semantic_value == value
                and not span.negated
                for span in spans
            )
        )
        bins = tuple(
            cast(BinId, value)
            for value in BIN_IDS
            if any(
                span.category == "supported_bin"
                and span.semantic_value == value
                and not span.negated
                for span in spans
            )
        )

        def values(category: str) -> tuple[str, ...]:
            return tuple(
                dict.fromkeys(span.semantic_value for span in spans if span.category == category)
            )

        negated_supported = tuple(
            span for span in spans if span.negated and span.category.startswith("supported_")
        )
        correction_present = bool(_CORRECTION_PATTERN.search(normalized))
        correction_resolved = correction_present and bool(
            _RESOLVED_CORRECTION_PATTERN.search(normalized)
        )
        ontology_signal = bool(
            objects
            or bins
            or values("supported_action")
            or values("unsupported_object")
            or values("unsupported_destination")
            or values("unsupported_action")
            or values("spatial_reference")
        )
        alpha_tokens = tuple(
            token for token in _WORD_PATTERN.findall(normalized) if token.isalpha()
        )
        meaningless = (
            bool(command.strip())
            and not _control_character_present(command)
            and not ontology_signal
            and bool(alpha_tokens)
        ) or (bool(command.strip()) and not any(character.isalpha() for character in normalized))
        return SymbolicLexicalFrameV0(
            supported_objects=objects,
            unsupported_object_mentions=values("unsupported_object"),
            supported_bins=bins,
            unsupported_destination_mentions=values("unsupported_destination"),
            supported_action_mentions=values("supported_action"),
            unsupported_action_mentions=values("unsupported_action"),
            spatial_reference_mentions=values("spatial_reference"),
            negation_present=bool(negated_supported),
            contradictory_negation=bool(negated_supported) and not (objects and bins),
            correction_present=correction_present,
            correction_resolved=correction_resolved,
            multiple_sequential_tasks=bool(_MULTI_TASK_PATTERN.search(normalized)),
            empty_input=not bool(command.strip()),
            malformed_control_characters=_control_character_present(command),
            meaningless_or_noise=meaningless,
            matched_spans=tuple(spans),
            normalized_text=normalized,
        )


class SemanticRequestedAction(StrEnum):
    PICK_AND_PLACE = "pick_and_place"
    UNSUPPORTED_ACTION = "unsupported_action"
    UNCLEAR = "unclear"


SupportedObjectList = Annotated[
    tuple[Literal["red_cube", "green_cube", "blue_cube"], ...],
    Field(max_length=3),
]
SupportedBinList = Annotated[
    tuple[Literal["left_bin", "right_bin"], ...],
    Field(max_length=2),
]


class QwenSemanticFrameV0(BaseModel):
    """Strict fact-only frame generated under an Outlines JSON grammar."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mentioned_supported_objects: SupportedObjectList
    mentions_unsupported_object: bool
    mentioned_supported_bins: SupportedBinList
    mentions_unsupported_destination: bool
    requested_action: SemanticRequestedAction
    has_multiple_tasks: bool
    has_conflicting_objects: bool
    has_conflicting_destinations: bool
    has_unresolved_correction: bool
    has_contradictory_negation: bool
    is_meaningless_or_noise: bool
    is_malformed_input: bool

    def model_post_init(self, __context: object) -> None:
        if len(set(self.mentioned_supported_objects)) != len(self.mentioned_supported_objects):
            raise NeuroSymbolicContractError("semantic object mentions must be unique")
        if len(set(self.mentioned_supported_bins)) != len(self.mentioned_supported_bins):
            raise NeuroSymbolicContractError("semantic bin mentions must be unique")

    @property
    def fingerprint(self) -> str:
        return f"sha256:{sha256_hex(self.to_dict())}"

    def to_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")


def semantic_frame_schema() -> dict[str, object]:
    """Return the exact strict JSON schema used by the grammar and validator."""

    return cast(dict[str, object], QwenSemanticFrameV0.model_json_schema())


def expected_semantic_frame_payload(
    example: LanguageExample, symbolic: SymbolicLexicalFrameV0
) -> dict[str, object]:
    reason = example.expected_rejection_reason
    route = example.expected_status is RouterStatus.ROUTE
    expected = example.expected_task_spec
    objects = (
        [expected.target_object_id]
        if route and expected is not None
        else list(symbolic.supported_objects)
    )
    bins = (
        [expected.target_bin_id]
        if route and expected is not None
        else list(symbolic.supported_bins)
    )
    return {
        "mentioned_supported_objects": objects,
        "mentions_unsupported_object": reason is RouterRejectionReason.UNSUPPORTED_OBJECT,
        "mentioned_supported_bins": bins,
        "mentions_unsupported_destination": reason is RouterRejectionReason.UNSUPPORTED_DESTINATION,
        "requested_action": (
            "unsupported_action"
            if reason is RouterRejectionReason.UNSUPPORTED_ACTION
            else "pick_and_place"
            if route or symbolic.supported_action_mentions
            else "unclear"
        ),
        "has_multiple_tasks": reason is RouterRejectionReason.MULTIPLE_TASKS,
        "has_conflicting_objects": reason is RouterRejectionReason.CONFLICTING_OBJECTS,
        "has_conflicting_destinations": reason is RouterRejectionReason.CONFLICTING_BINS,
        "has_unresolved_correction": reason is RouterRejectionReason.UNRESOLVED_CORRECTION,
        "has_contradictory_negation": reason is RouterRejectionReason.CONTRADICTORY_NEGATION,
        "is_meaningless_or_noise": reason
        in {RouterRejectionReason.MEANINGLESS_TEXT, RouterRejectionReason.EMPTY_TEXT},
        "is_malformed_input": reason is RouterRejectionReason.MALFORMED_CONTROL_CHARACTERS,
    }


def select_semantic_prompt_examples(
    examples: Sequence[LanguageExample],
) -> tuple[LanguageExample, ...]:
    """Select six routes and one train-only example for each rejection reason."""

    train = tuple(sorted(examples, key=lambda value: value.example_id))
    if not train or any(value.split is not LanguageSplit.TRAIN for value in train):
        raise NeuroSymbolicContractError("semantic prompt requires train-only examples")
    selected: list[LanguageExample] = []
    for spec in CANONICAL_TASK_SPECS:
        selected.append(next(value for value in train if value.expected_task_spec == spec))
    reasons = sorted(
        {
            value.expected_rejection_reason
            for value in train
            if value.expected_rejection_reason is not None
        },
        key=lambda value: value.value,
    )
    for reason in reasons:
        selected.append(next(value for value in train if value.expected_rejection_reason is reason))
    if len(selected) != 20 or len({value.example_id for value in selected}) != 20:
        raise NeuroSymbolicContractError("semantic prompt must contain the frozen 20 examples")
    return tuple(selected)


def build_semantic_frame_prompt(
    *, examples: Sequence[LanguageExample], parser: SymbolicLexicalParserV0
) -> tuple[str, str, tuple[str, ...]]:
    """Build the single M5A.4 semantic prompt from train-only examples."""

    selected = tuple(examples)
    if any(value.split is not LanguageSplit.TRAIN for value in selected):
        raise NeuroSymbolicContractError("semantic prompt examples must be train-only")
    lines = [
        f"Prompt version: {SEMANTIC_PROMPT_VERSION}",
        "Extract semantic facts from one robot command.",
        "Supported objects are red_cube, green_cube, and blue_cube.",
        "Supported destinations are left_bin and right_bin.",
        "The only supported action is picking one cube and placing it in one bin.",
        "Report mentioned facts and conflicts; do not decide whether execution is allowed.",
        "Do not guess missing facts. Do not output reasoning or explanations.",
        "The constrained response contains only the declared semantic-frame fields.",
        "requested_action is pick_and_place, unsupported_action, or unclear.",
    ]
    for example in selected:
        payload = expected_semantic_frame_payload(example, parser.parse(example.raw_text))
        QwenSemanticFrameV0.model_validate(payload)
        lines.extend(
            (
                f"Example ID: {example.example_id}",
                f"Command: {json.dumps(example.raw_text, ensure_ascii=False)}",
                f"Frame: {canonical_json(payload)}",
            )
        )
    lines.append("Return the semantic frame for the user command.")
    content = "\n".join(lines)
    ids = tuple(value.example_id for value in selected)
    fingerprint = f"sha256:{sha256_hex({'content': content, 'example_ids': ids})}"
    return content, fingerprint, ids


class SemanticFrameExtractor(Protocol):
    compilation_seconds: float
    generation_metadata: list[dict[str, object]]

    def extract(self, command: str) -> QwenSemanticFrameV0: ...


class OutlinesQwenSemanticFrameExtractorV0:
    """One cached Outlines grammar over the exact frozen Qwen3-4B model."""

    def __init__(
        self,
        *,
        loader: TransformersLocalTextGenerator,
        prompt_content: str,
        maximum_new_tokens: int = 256,
    ) -> None:
        if loader.config.model_id != QWEN3_4B_INSTRUCT_MODEL_ID or (
            loader.config.model_revision != QWEN3_4B_INSTRUCT_REVISION
            or loader.config.tokenizer_revision != QWEN3_4B_INSTRUCT_REVISION
            or loader.config.dtype != "bfloat16"
            or loader.config.quantization != "none"
        ):
            raise NeuroSymbolicContractError("semantic extractor requires exact frozen Qwen3-4B")
        if maximum_new_tokens < 64:
            raise NeuroSymbolicContractError("semantic frame token bound is too small")
        try:
            import outlines
            from outlines.inputs import Chat
        except ImportError as error:
            raise StructuredLLMRouterError("Outlines 1.3.1 is required without fallback") from error
        try:
            installed_outlines = version("outlines")
        except PackageNotFoundError as error:
            raise StructuredLLMRouterError("Outlines distribution metadata is missing") from error
        if installed_outlines != OUTLINES_VERSION:
            raise StructuredLLMRouterError("installed Outlines version differs from 1.3.1")
        compiled_at = time.perf_counter()
        try:
            wrapped = outlines.from_transformers(loader.model, loader.tokenizer)
            generator = outlines.Generator(wrapped, QwenSemanticFrameV0)
        except (RuntimeError, TypeError, ValueError) as error:
            raise StructuredLLMRouterError(
                f"Outlines/Qwen grammar compilation failed without fallback: {error}"
            ) from error
        self.compilation_seconds = time.perf_counter() - compiled_at
        self._generator = generator
        self._chat_type = Chat
        self._loader = loader
        self._prompt_content = prompt_content
        self.maximum_new_tokens = maximum_new_tokens
        self.generation_metadata: list[dict[str, object]] = []

    def extract(self, command: str) -> QwenSemanticFrameV0:
        if not isinstance(command, str):
            raise TypeError("command must be text")
        chat = self._chat_type(
            [
                {"role": "system", "content": self._prompt_content},
                {"role": "user", "content": command},
            ]
        )
        if self._loader.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                raw = self._generator(
                    chat,
                    do_sample=False,
                    num_beams=1,
                    num_return_sequences=1,
                    max_new_tokens=self.maximum_new_tokens,
                    eos_token_id=self._loader.tokenizer.eos_token_id,
                    pad_token_id=self._loader.tokenizer.eos_token_id,
                )
            if not isinstance(raw, str):
                raise StructuredLLMRouterError("Outlines returned a non-text semantic frame")
            frame = QwenSemanticFrameV0.model_validate_json(raw)
        except ValidationError as error:
            raise StructuredLLMRouterError(
                f"grammar output failed mandatory semantic validation: {error}"
            ) from error
        elapsed = time.perf_counter() - started
        peak = 0 if self._loader.device != "cuda" else int(torch.cuda.max_memory_allocated())
        token_count = len(self._loader.tokenizer.encode(raw, add_special_tokens=False))
        self.generation_metadata.append(
            {
                "elapsed_seconds": elapsed,
                "generated_token_count": token_count,
                "peak_gpu_memory_bytes": peak,
                "grammar_schema_valid": True,
                "do_sample": False,
            }
        )
        return frame


class DeterministicSafetyArbiterV0:
    """The sole component that converts two fact frames into RouterDecision."""

    def __init__(self) -> None:
        self.contract_fingerprint = f"sha256:{sha256_hex(self.contract_dict())}"

    @staticmethod
    def contract_dict() -> dict[str, object]:
        return {
            "arbiter_version": ARBITER_VERSION,
            "precedence": ["malformed", "unsupported", "ambiguous", "route"],
            "agreement_policy": "exact-object-bin-and-pick-action-v0",
            "disagreement_policy": "reject-never-guess-v0",
            "expected_label_input": False,
            "split_identity_input": False,
            "template_family_input": False,
            "classifier_logits_input": False,
        }

    def arbitrate(
        self,
        symbolic: SymbolicLexicalFrameV0,
        semantic: QwenSemanticFrameV0 | None,
        *,
        semantic_error: str | None = None,
    ) -> RouterDecision:
        if not isinstance(symbolic, SymbolicLexicalFrameV0):
            raise TypeError("symbolic must be SymbolicLexicalFrameV0")
        if semantic is None and not semantic_error:
            raise NeuroSymbolicContractError("missing semantic frame requires an error")
        base = {
            "arbiter_fingerprint": self.contract_fingerprint,
            "symbolic_frame": symbolic.to_dict(),
            "semantic_frame": None if semantic is None else semantic.to_dict(),
            "semantic_error": semantic_error,
        }

        def reject(
            status: RouterStatus, reason: RouterRejectionReason, rule: str
        ) -> RouterDecision:
            return RouterDecision.reject(
                status=status,
                rejection_reason=reason,
                confidence=RouterConfidence.unavailable(
                    definition="deterministic safety arbiter has no probability"
                ),
                router_name=NEURO_SYMBOLIC_ROUTER_NAME,
                router_version=NEURO_SYMBOLIC_ROUTER_VERSION,
                evidence={**base, "arbiter_rule": rule},
            )

        if symbolic.empty_input:
            return reject(RouterStatus.REJECT_MALFORMED, RouterRejectionReason.EMPTY_TEXT, "m01")
        if symbolic.malformed_control_characters:
            return reject(
                RouterStatus.REJECT_MALFORMED,
                RouterRejectionReason.MALFORMED_CONTROL_CHARACTERS,
                "m02",
            )
        if semantic is None:
            return reject(
                RouterStatus.REJECT_MALFORMED,
                RouterRejectionReason.STRUCTURED_OUTPUT_INVALID,
                "m03",
            )
        if symbolic.meaningless_or_noise or semantic.is_meaningless_or_noise:
            return reject(
                RouterStatus.REJECT_MALFORMED,
                RouterRejectionReason.MEANINGLESS_TEXT,
                "m04",
            )
        if semantic.is_malformed_input:
            return reject(
                RouterStatus.REJECT_MALFORMED,
                RouterRejectionReason.MALFORMED_CONTROL_CHARACTERS,
                "m05",
            )
        if symbolic.unsupported_action_mentions or (
            semantic.requested_action is SemanticRequestedAction.UNSUPPORTED_ACTION
        ):
            return reject(
                RouterStatus.REJECT_UNSUPPORTED,
                RouterRejectionReason.UNSUPPORTED_ACTION,
                "u01",
            )
        if symbolic.unsupported_object_mentions or semantic.mentions_unsupported_object:
            return reject(
                RouterStatus.REJECT_UNSUPPORTED,
                RouterRejectionReason.UNSUPPORTED_OBJECT,
                "u02",
            )
        if symbolic.spatial_reference_mentions:
            return reject(
                RouterStatus.REJECT_UNSUPPORTED,
                RouterRejectionReason.UNSUPPORTED_SPATIAL_REFERENCE,
                "u03",
            )
        if symbolic.unsupported_destination_mentions or semantic.mentions_unsupported_destination:
            return reject(
                RouterStatus.REJECT_UNSUPPORTED,
                RouterRejectionReason.UNSUPPORTED_DESTINATION,
                "u04",
            )
        if symbolic.multiple_sequential_tasks or semantic.has_multiple_tasks:
            return reject(
                RouterStatus.REJECT_AMBIGUOUS,
                RouterRejectionReason.MULTIPLE_TASKS,
                "a01",
            )
        if len(symbolic.supported_objects) > 1 or semantic.has_conflicting_objects:
            return reject(
                RouterStatus.REJECT_AMBIGUOUS,
                RouterRejectionReason.CONFLICTING_OBJECTS,
                "a02",
            )
        if len(symbolic.supported_bins) > 1 or semantic.has_conflicting_destinations:
            return reject(
                RouterStatus.REJECT_AMBIGUOUS,
                RouterRejectionReason.CONFLICTING_BINS,
                "a03",
            )
        if symbolic.correction_present and not symbolic.correction_resolved:
            return reject(
                RouterStatus.REJECT_AMBIGUOUS,
                RouterRejectionReason.UNRESOLVED_CORRECTION,
                "a04",
            )
        if semantic.has_unresolved_correction:
            return reject(
                RouterStatus.REJECT_AMBIGUOUS,
                RouterRejectionReason.UNRESOLVED_CORRECTION,
                "a05",
            )
        if symbolic.contradictory_negation or semantic.has_contradictory_negation:
            return reject(
                RouterStatus.REJECT_AMBIGUOUS,
                RouterRejectionReason.CONTRADICTORY_NEGATION,
                "a06",
            )
        if not symbolic.supported_objects or not semantic.mentioned_supported_objects:
            return reject(
                RouterStatus.REJECT_AMBIGUOUS,
                RouterRejectionReason.MISSING_OBJECT,
                "a07",
            )
        if not symbolic.supported_bins or not semantic.mentioned_supported_bins:
            return reject(
                RouterStatus.REJECT_AMBIGUOUS,
                RouterRejectionReason.MISSING_DESTINATION,
                "a08",
            )
        symbolic_objects = tuple(symbolic.supported_objects)
        semantic_objects = tuple(semantic.mentioned_supported_objects)
        if symbolic_objects != semantic_objects:
            return reject(
                RouterStatus.REJECT_AMBIGUOUS,
                RouterRejectionReason.CONFLICTING_OBJECTS,
                "a09",
            )
        symbolic_bins = tuple(symbolic.supported_bins)
        semantic_bins = tuple(semantic.mentioned_supported_bins)
        if symbolic_bins != semantic_bins:
            return reject(
                RouterStatus.REJECT_AMBIGUOUS,
                RouterRejectionReason.CONFLICTING_BINS,
                "a10",
            )
        if semantic.requested_action is not SemanticRequestedAction.PICK_AND_PLACE:
            return reject(
                RouterStatus.REJECT_AMBIGUOUS,
                RouterRejectionReason.LOW_CONFIDENCE,
                "a11",
            )
        if len(symbolic_objects) != 1 or len(symbolic_bins) != 1:
            raise NeuroSymbolicContractError("route rule received non-unique slots")
        task = TaskSpec.from_mapping(
            {
                "target_object_id": symbolic_objects[0],
                "target_bin_id": symbolic_bins[0],
                "instruction_template_id": "canonical_v0",
            }
        )
        return RouterDecision.route(
            task_spec=task,
            confidence=RouterConfidence.unavailable(
                definition="deterministic safety arbiter has no probability"
            ),
            router_name=NEURO_SYMBOLIC_ROUTER_NAME,
            router_version=NEURO_SYMBOLIC_ROUTER_VERSION,
            evidence={**base, "arbiter_rule": "r01"},
        )


class NeuroSymbolicRouterV0:
    """Compose deterministic lexical facts, frozen Qwen facts, and one arbiter."""

    def __init__(
        self,
        *,
        parser: SymbolicLexicalParserV0,
        semantic_extractor: SemanticFrameExtractor,
        arbiter: DeterministicSafetyArbiterV0,
    ) -> None:
        self.parser = parser
        self.semantic_extractor = semantic_extractor
        self.arbiter = arbiter

    def route(self, command: str) -> RouterDecision:
        symbolic = self.parser.parse(command)
        metadata_start = len(self.semantic_extractor.generation_metadata)
        try:
            semantic = self.semantic_extractor.extract(command)
        except (ValidationError, StructuredLLMRouterError) as error:
            return self.arbiter.arbitrate(
                symbolic,
                None,
                semantic_error=f"{type(error).__name__}: {error}",
            )
        decision = self.arbiter.arbitrate(symbolic, semantic)
        generation_metadata = self.semantic_extractor.generation_metadata[metadata_start:]
        evidence = {**dict(decision.evidence), "generation_metadata": generation_metadata}
        if decision.status is RouterStatus.ROUTE:
            assert decision.task_spec is not None
            return RouterDecision.route(
                task_spec=decision.task_spec,
                confidence=decision.confidence,
                router_name=decision.router_name,
                router_version=decision.router_version,
                evidence=evidence,
            )
        assert decision.rejection_reason is not None
        return RouterDecision.reject(
            status=decision.status,
            rejection_reason=decision.rejection_reason,
            confidence=decision.confidence,
            router_name=decision.router_name,
            router_version=decision.router_version,
            evidence=evidence,
        )


def semantic_schema_fingerprint() -> str:
    return f"sha256:{sha256_hex(semantic_frame_schema())}"


def build_qwen4b_loader_config(*, maximum_new_tokens: int = 256) -> StructuredLLMRouterConfig:
    return StructuredLLMRouterConfig(
        model_id=QWEN3_4B_INSTRUCT_MODEL_ID,
        model_revision=QWEN3_4B_INSTRUCT_REVISION,
        tokenizer_revision=QWEN3_4B_INSTRUCT_REVISION,
        dtype="bfloat16",
        quantization="none",
        maximum_new_tokens=maximum_new_tokens,
        maximum_format_repair_attempts=1,
        chat_template_mode="official_qwen_instruct_non_thinking",
    )


__all__ = [
    "ARBITER_VERSION",
    "NEURO_SYMBOLIC_ROUTER_NAME",
    "NEURO_SYMBOLIC_ROUTER_VERSION",
    "OUTLINES_LICENSE",
    "OUTLINES_VERSION",
    "SEMANTIC_FRAME_SCHEMA_VERSION",
    "SEMANTIC_PROMPT_VERSION",
    "SYMBOLIC_FRAME_VERSION",
    "DeterministicSafetyArbiterV0",
    "MatchedLexicalSpanV0",
    "NeuroSymbolicContractError",
    "NeuroSymbolicRouterV0",
    "OutlinesQwenSemanticFrameExtractorV0",
    "QwenSemanticFrameV0",
    "SemanticFrameExtractor",
    "SemanticRequestedAction",
    "SymbolicLexicalFrameV0",
    "SymbolicLexicalParserV0",
    "build_qwen4b_loader_config",
    "build_semantic_frame_prompt",
    "expected_semantic_frame_payload",
    "select_semantic_prompt_examples",
    "semantic_frame_schema",
    "semantic_schema_fingerprint",
]
