"""Structured, local-only instruct-model router with bounded format repair."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import torch

from langmani.datasets.identity import canonical_json, sha256_hex
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import TaskSpec, stable_task_id
from langmani.language.router_types import (
    LanguageExample,
    LanguageSplit,
    RouterConfidence,
    RouterDecision,
    RouterRejectionReason,
    RouterStatus,
)
from langmani.language.schema_validation import (
    RouterSchemaError,
    StrictRouterPayload,
    parse_strict_router_json,
)

STRUCTURED_LLM_ROUTER_NAME = "StructuredLocalLLMRouterV0"
STRUCTURED_LLM_ROUTER_VERSION = "structured-local-llm-router-v0"
STRUCTURED_LLM_PROMPT_VERSION = "m5a-structured-routing-prompt-v0"


class StructuredLLMRouterError(RuntimeError):
    """Raised when the explicitly configured local LLM cannot be used safely."""


class LocalTextGenerator(Protocol):
    """Small project boundary implemented by a local Transformers adapter or fixture."""

    def generate(self, prompt: str, *, max_new_tokens: int) -> str: ...


@dataclass(frozen=True, slots=True)
class StructuredLLMRouterConfig:
    model_id: str
    model_revision: str
    tokenizer_revision: str
    dtype: str = "bfloat16"
    quantization: str = "none"
    maximum_new_tokens: int = 128
    maximum_format_repair_attempts: int = 1
    prompt_version: str = STRUCTURED_LLM_PROMPT_VERSION

    def __post_init__(self) -> None:
        for name in ("model_id", "model_revision", "tokenizer_revision", "prompt_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be supplied explicitly")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be float32, float16, or bfloat16")
        if self.quantization != "none":
            raise ValueError("M5A implements no implicit quantization fallback")
        if not 1 <= self.maximum_new_tokens <= 512:
            raise ValueError("maximum_new_tokens must lie in [1,512]")
        if self.maximum_format_repair_attempts not in (0, 1):
            raise ValueError("at most one format-repair attempt is permitted")

    @property
    def fingerprint(self) -> str:
        return f"sha256:{sha256_hex(self.to_dict())}"

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "maximum_new_tokens": self.maximum_new_tokens,
            "maximum_format_repair_attempts": self.maximum_format_repair_attempts,
            "prompt_version": self.prompt_version,
            "do_sample": False,
            "num_beams": 1,
            "requested_temperature": 0.0,
            "effective_sampling_temperature": None,
        }


def _expected_payload(example: LanguageExample) -> dict[str, object]:
    if example.expected_status is RouterStatus.ROUTE:
        assert example.expected_task_spec is not None
        return {
            "status": RouterStatus.ROUTE.value,
            "target_object_id": example.expected_task_spec.target_object_id,
            "target_bin_id": example.expected_task_spec.target_bin_id,
            "reason": "route",
        }
    assert example.expected_rejection_reason is not None
    return {
        "status": example.expected_status.value,
        "target_object_id": None,
        "target_bin_id": None,
        "reason": example.expected_rejection_reason.value,
    }


def build_structured_routing_prompt(
    *,
    examples: Sequence[LanguageExample],
    prompt_version: str = STRUCTURED_LLM_PROMPT_VERSION,
) -> tuple[str, str, tuple[str, ...]]:
    """Freeze one prompt from train-family examples only."""

    selected = tuple(examples)
    if any(value.split is not LanguageSplit.TRAIN for value in selected):
        raise StructuredLLMRouterError("LLM prompt examples may come from train families only")
    if len({value.example_id for value in selected}) != len(selected):
        raise StructuredLLMRouterError("LLM prompt example IDs must be unique")
    schema = (
        '{"status":"route|reject_ambiguous|reject_unsupported|reject_malformed",'
        '"target_object_id":"red_cube|green_cube|blue_cube|null",'
        '"target_bin_id":"left_bin|right_bin|null","reason":"machine_readable_reason"}'
    )
    lines = [
        f"Prompt version: {prompt_version}",
        "Map one robot command to exactly one JSON object.",
        "Never guess missing, conflicting, unsupported, or malformed intent.",
        "A rejection must use null for both target fields.",
        "Return JSON only. Do not include markdown, prose, or reasoning.",
        f"Schema: {schema}",
    ]
    for example in selected:
        lines.extend(
            (
                f"Example ID: {example.example_id}",
                f"Command: {json.dumps(example.raw_text, ensure_ascii=False)}",
                "Output: " + canonical_json(_expected_payload(example)),
            )
        )
    lines.append("Command: {COMMAND_JSON}")
    lines.append("Output:")
    template = "\n".join(lines)
    ids = tuple(value.example_id for value in selected)
    fingerprint = f"sha256:{sha256_hex({'prompt_template': template, 'example_ids': ids})}"
    return template, fingerprint, ids


def select_structured_routing_prompt_examples(
    examples: Sequence[LanguageExample],
) -> tuple[LanguageExample, ...]:
    """Select the one canonical train-only prompt shared by every M5A stage."""

    train = tuple(sorted(examples, key=lambda item: item.example_id))
    if not train or any(example.split is not LanguageSplit.TRAIN for example in train):
        raise StructuredLLMRouterError(
            "structured-routing prompt selection requires non-empty train-only examples"
        )
    selected: list[LanguageExample] = []
    for task_spec in CANONICAL_TASK_SPECS:
        task_id = stable_task_id(task_spec)
        try:
            selected.append(next(example for example in train if example.task_id == task_id))
        except StopIteration as error:
            raise StructuredLLMRouterError(
                f"train prompt pool lacks canonical task {task_id}"
            ) from error
    rejection_reasons = sorted(
        {
            example.expected_rejection_reason
            for example in train
            if example.expected_rejection_reason is not None
        },
        key=lambda reason: reason.value,
    )
    for reason in rejection_reasons:
        selected.append(
            next(example for example in train if example.expected_rejection_reason is reason)
        )
    result = tuple(selected)
    if len({example.example_id for example in result}) != len(result):
        raise StructuredLLMRouterError("canonical prompt selection produced duplicate examples")
    return result


_REASONS_BY_STATUS: Mapping[RouterStatus, frozenset[RouterRejectionReason]] = {
    RouterStatus.REJECT_AMBIGUOUS: frozenset(
        {
            RouterRejectionReason.MISSING_OBJECT,
            RouterRejectionReason.MISSING_DESTINATION,
            RouterRejectionReason.CONFLICTING_OBJECTS,
            RouterRejectionReason.CONFLICTING_BINS,
            RouterRejectionReason.MULTIPLE_TASKS,
            RouterRejectionReason.UNRESOLVED_CORRECTION,
            RouterRejectionReason.CONTRADICTORY_NEGATION,
            RouterRejectionReason.LOW_CONFIDENCE,
        }
    ),
    RouterStatus.REJECT_UNSUPPORTED: frozenset(
        {
            RouterRejectionReason.UNSUPPORTED_OBJECT,
            RouterRejectionReason.UNSUPPORTED_DESTINATION,
            RouterRejectionReason.UNSUPPORTED_ACTION,
            RouterRejectionReason.UNSUPPORTED_SPATIAL_REFERENCE,
        }
    ),
    RouterStatus.REJECT_MALFORMED: frozenset(
        {
            RouterRejectionReason.MEANINGLESS_TEXT,
            RouterRejectionReason.EMPTY_TEXT,
            RouterRejectionReason.MALFORMED_CONTROL_CHARACTERS,
            RouterRejectionReason.STRUCTURED_OUTPUT_INVALID,
            RouterRejectionReason.FORMAT_REPAIR_EXHAUSTED,
        }
    ),
}


def _validated_rejection(
    payload: StrictRouterPayload,
) -> tuple[RouterStatus, RouterRejectionReason]:
    try:
        status = RouterStatus(payload.status)
        reason = RouterRejectionReason(payload.reason)
    except ValueError as error:
        raise RouterSchemaError("structured rejection status/reason is unsupported") from error
    if status is RouterStatus.ROUTE or reason not in _REASONS_BY_STATUS[status]:
        raise RouterSchemaError("structured rejection reason does not match its status")
    return status, reason


class StructuredLocalLLMRouterV0:
    """Use strict JSON and no more than one deterministic repair generation."""

    def __init__(
        self,
        *,
        config: StructuredLLMRouterConfig,
        generator: LocalTextGenerator,
        prompt_examples: Sequence[LanguageExample] = (),
    ) -> None:
        if not isinstance(config, StructuredLLMRouterConfig):
            raise TypeError("config must be StructuredLLMRouterConfig")
        self.config = config
        self.generator = generator
        self.prompt_template, self.prompt_fingerprint, self.prompt_example_ids = (
            build_structured_routing_prompt(
                examples=prompt_examples,
                prompt_version=config.prompt_version,
            )
        )

    def route(self, command: str) -> RouterDecision:
        if not isinstance(command, str):
            raise TypeError("command must be text")
        prompt = self.prompt_template.replace(
            "{COMMAND_JSON}", json.dumps(command, ensure_ascii=False)
        )
        outputs: list[str] = []
        parse_errors: list[str] = []
        attempts = 1 + self.config.maximum_format_repair_attempts
        for attempt_index in range(attempts):
            active_prompt = prompt
            if attempt_index:
                active_prompt = self._repair_prompt(
                    command=command,
                    invalid_output=outputs[-1],
                    error=parse_errors[-1],
                )
            raw_output = self.generator.generate(
                active_prompt,
                max_new_tokens=self.config.maximum_new_tokens,
            )
            if not isinstance(raw_output, str):
                raise StructuredLLMRouterError("local generator must return text")
            outputs.append(raw_output)
            try:
                payload = parse_strict_router_json(raw_output)
                return self._decision(
                    payload,
                    command=command,
                    outputs=outputs,
                    parse_errors=parse_errors,
                )
            except (RouterSchemaError, ValueError) as error:
                parse_errors.append(f"{type(error).__name__}: {error}")
        return RouterDecision.reject(
            status=RouterStatus.REJECT_MALFORMED,
            rejection_reason=RouterRejectionReason.FORMAT_REPAIR_EXHAUSTED,
            confidence=RouterConfidence.unavailable(
                definition="structured local LLM confidence is unavailable"
            ),
            router_name=STRUCTURED_LLM_ROUTER_NAME,
            router_version=STRUCTURED_LLM_ROUTER_VERSION,
            evidence=self._evidence(command, outputs, parse_errors),
        )

    def _decision(
        self,
        payload: StrictRouterPayload,
        *,
        command: str,
        outputs: Sequence[str],
        parse_errors: Sequence[str],
    ) -> RouterDecision:
        evidence = self._evidence(command, outputs, parse_errors)
        confidence = RouterConfidence.unavailable(
            definition="structured local LLM confidence is unavailable"
        )
        if payload.status == RouterStatus.ROUTE.value:
            task_spec = TaskSpec.from_mapping(
                {
                    "target_object_id": payload.target_object_id,
                    "target_bin_id": payload.target_bin_id,
                    "instruction_template_id": "canonical_v0",
                }
            )
            return RouterDecision.route(
                task_spec=task_spec,
                confidence=confidence,
                router_name=STRUCTURED_LLM_ROUTER_NAME,
                router_version=STRUCTURED_LLM_ROUTER_VERSION,
                evidence=evidence,
            )
        status, reason = _validated_rejection(payload)
        return RouterDecision.reject(
            status=status,
            rejection_reason=reason,
            confidence=confidence,
            router_name=STRUCTURED_LLM_ROUTER_NAME,
            router_version=STRUCTURED_LLM_ROUTER_VERSION,
            evidence=evidence,
        )

    def _evidence(
        self,
        command: str,
        outputs: Sequence[str],
        parse_errors: Sequence[str],
    ) -> dict[str, object]:
        return {
            "command_fingerprint": f"sha256:{sha256_hex(command)}",
            "config_fingerprint": self.config.fingerprint,
            "prompt_fingerprint": self.prompt_fingerprint,
            "prompt_example_ids": list(self.prompt_example_ids),
            # Persist only content identities and sizes. A model may ignore the
            # JSON-only prompt and emit free-form reasoning; M5A must never
            # preserve that text as chain-of-thought evidence.
            "generation_output_fingerprints": [
                f"sha256:{sha256_hex(output)}" for output in outputs
            ],
            "generation_output_character_counts": [len(output) for output in outputs],
            "parse_errors": list(parse_errors),
            "format_repair_attempts": max(0, len(outputs) - 1),
            "confidence_available": False,
        }

    def _repair_prompt(self, *, command: str, invalid_output: str, error: str) -> str:
        return "\n".join(
            (
                f"Prompt version: {self.config.prompt_version}-repair-v0",
                "Repair only the JSON format. Do not infer a new command or explain your work.",
                "Return exactly one object with status,target_object_id,target_bin_id,reason.",
                f"Command: {json.dumps(command, ensure_ascii=False)}",
                f"Invalid output: {json.dumps(invalid_output, ensure_ascii=False)}",
                f"Validation error: {json.dumps(error, ensure_ascii=False)}",
                "Output:",
            )
        )


class TransformersLocalTextGenerator:
    """Thin public-API adapter; construction never falls back to a hosted API."""

    def __init__(
        self,
        *,
        config: StructuredLLMRouterConfig,
        device: str,
        local_files_only: bool,
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise StructuredLLMRouterError("installed Transformers is required") from error
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.dtype]
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model_id,
                revision=config.tokenizer_revision,
                local_files_only=local_files_only,
                trust_remote_code=False,
                use_fast=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                revision=config.model_revision,
                local_files_only=local_files_only,
                trust_remote_code=False,
                dtype=dtype,
            ).to(device)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise StructuredLLMRouterError(
                f"explicit local model/tokenizer load failed without fallback: {error}"
            ) from error
        self.model.eval()
        self.device = device

    def generate(self, prompt: str, *, max_new_tokens: int) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise StructuredLLMRouterError(
                f"configured instruct tokenizer lacks a usable chat template: {error}"
            ) from error
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise StructuredLLMRouterError("chat template did not return model inputs")
        model_inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in encoded.items()
        }
        input_length = cast(torch.Tensor, model_inputs["input_ids"]).shape[-1]
        with torch.inference_mode():
            generated = self.model.generate(
                **model_inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
            )
        if (
            not isinstance(generated, torch.Tensor)
            or generated.ndim != 2
            or generated.shape[0] != 1
        ):
            raise StructuredLLMRouterError("local generation returned an unexpected tensor shape")
        return cast(
            str, self.tokenizer.decode(generated[0, input_length:], skip_special_tokens=True)
        )


__all__ = [
    "STRUCTURED_LLM_PROMPT_VERSION",
    "STRUCTURED_LLM_ROUTER_NAME",
    "STRUCTURED_LLM_ROUTER_VERSION",
    "LocalTextGenerator",
    "StructuredLLMRouterConfig",
    "StructuredLLMRouterError",
    "StructuredLocalLLMRouterV0",
    "TransformersLocalTextGenerator",
    "build_structured_routing_prompt",
    "select_structured_routing_prompt_examples",
]
