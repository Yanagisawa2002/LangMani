"""Structured, local-only instruct-model router with bounded format repair."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
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
    ROUTER_REASON_CODE_VERSION,
    RouterSchemaError,
    StrictRouterPayload,
    parse_strict_router_json,
)

STRUCTURED_LLM_ROUTER_NAME = "StructuredLocalLLMRouterV0"
STRUCTURED_LLM_ROUTER_VERSION = "structured-local-llm-router-v0"
STRUCTURED_LLM_PROMPT_VERSION = "m5a-structured-routing-prompt-v1"
QWEN3_1_7B_MODEL_ID = "Qwen/Qwen3-1.7B"
QWEN3_1_7B_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
QWEN3_1_7B_LICENSE = "apache-2.0"
QWEN3_1_7B_FILE_IDENTITIES: Mapping[str, tuple[int, str]] = {
    ".gitattributes": (
        1570,
        "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930",
    ),
    "LICENSE": (11343, "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"),
    "README.md": (
        13963,
        "257e52c419dac2258852643f18af6c974f21f8c6c1b6f371b6cca6201cf29091",
    ),
    "config.json": (
        726,
        "1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197",
    ),
    "generation_config.json": (
        239,
        "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    ),
    "merges.txt": (
        1671853,
        "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    ),
    "model-00001-of-00002.safetensors": (
        3441185608,
        "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5",
    ),
    "model-00002-of-00002.safetensors": (
        622329984,
        "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9",
    ),
    "model.safetensors.index.json": (
        25605,
        "0d660e94b165eb912669a5249dff44b83188c4777a07ddb9611fb78d91b0578d",
    ),
    "tokenizer.json": (
        11422654,
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    ),
    "tokenizer_config.json": (
        9732,
        "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    ),
    "vocab.json": (
        2776833,
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
}

QWEN3_4B_INSTRUCT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
QWEN3_4B_INSTRUCT_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
QWEN3_4B_INSTRUCT_LICENSE = "apache-2.0"
QWEN3_4B_INSTRUCT_FILE_IDENTITIES: Mapping[str, tuple[int, str]] = {
    ".gitattributes": (
        1570,
        "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930",
    ),
    "LICENSE": (11343, "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"),
    "README.md": (
        8168,
        "8e3dd0c3b5b11897cc71092ccfe517bb7a9783479baa3665aad73c8d1a2041cd",
    ),
    "config.json": (
        727,
        "5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba",
    ),
    "generation_config.json": (
        238,
        "835fffe355c9438e7a25be099b3fccaa98350b83451f9fd2d99512e74f1ade48",
    ),
    "merges.txt": (
        1671839,
        "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    ),
    "model-00001-of-00003.safetensors": (
        3957900840,
        "75311d91bb08cf0b882913da464a1e722a31fb44db35208663487efb7a3d8ed6",
    ),
    "model-00002-of-00003.safetensors": (
        3987450520,
        "0b48adbb1f60e901153d91907ba11ce63bd4b8b584482e730f48808d055dfba1",
    ),
    "model-00003-of-00003.safetensors": (
        99630640,
        "7dd39ccca5e4de123c74c14af44c9bf2eb75df33b4614382af0134528e060d5d",
    ),
    "model.safetensors.index.json": (
        32819,
        "d6c42883a895dfef5b0080ed2116a1bcd764f558406b98923d675978a1abf29c",
    ),
    "tokenizer.json": (
        11422654,
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    ),
    "tokenizer_config.json": (
        9377,
        "a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3",
    ),
    "vocab.json": (
        2776833,
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
}

_PINNED_MODEL_FILES: Mapping[tuple[str, str], Mapping[str, tuple[int, str]]] = {
    (QWEN3_1_7B_MODEL_ID, QWEN3_1_7B_REVISION): QWEN3_1_7B_FILE_IDENTITIES,
    (
        QWEN3_4B_INSTRUCT_MODEL_ID,
        QWEN3_4B_INSTRUCT_REVISION,
    ): QWEN3_4B_INSTRUCT_FILE_IDENTITIES,
}
_CHAT_TEMPLATE_MODES = {
    "official_qwen_enable_thinking_false",
    "official_qwen_instruct_non_thinking",
}


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
    reason_code_version: str = ROUTER_REASON_CODE_VERSION
    chat_template_mode: str = "official_qwen_enable_thinking_false"

    def __post_init__(self) -> None:
        for name in (
            "model_id",
            "model_revision",
            "tokenizer_revision",
            "prompt_version",
            "reason_code_version",
            "chat_template_mode",
        ):
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
        if self.chat_template_mode not in _CHAT_TEMPLATE_MODES:
            raise ValueError("chat_template_mode is not one authorized Qwen transport")

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
            "reason_code_version": self.reason_code_version,
            "chat_template_mode": self.chat_template_mode,
            "do_sample": False,
            "num_beams": 1,
            "requested_temperature": 0.0,
            "effective_sampling_temperature": None,
            "eos_behavior": "tokenizer_eos_token_id",
            "thinking_enabled": False,
            "chain_of_thought_requested": False,
        }


def _expected_payload(example: LanguageExample) -> dict[str, object]:
    if example.expected_status is RouterStatus.ROUTE:
        assert example.expected_task_spec is not None
        return {
            "status": RouterStatus.ROUTE.value,
            "target_object_id": example.expected_task_spec.target_object_id,
            "target_bin_id": example.expected_task_spec.target_bin_id,
            "reason": "explicit_object_and_destination",
        }
    assert example.expected_rejection_reason is not None
    reason = {
        RouterRejectionReason.CONFLICTING_BINS: "conflicting_destinations",
        RouterRejectionReason.MEANINGLESS_TEXT: "meaningless_or_noise",
        RouterRejectionReason.EMPTY_TEXT: "malformed_input",
        RouterRejectionReason.MALFORMED_CONTROL_CHARACTERS: "malformed_input",
    }.get(example.expected_rejection_reason, example.expected_rejection_reason.value)
    return {
        "status": example.expected_status.value,
        "target_object_id": None,
        "target_bin_id": None,
        "reason": reason,
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
        '"target_bin_id":"left_bin|right_bin|null","reason":"allowed_reason_code"}'
    )
    lines = [
        f"Prompt version: {prompt_version}",
        "Map one robot command to exactly one JSON object.",
        "Never guess missing, conflicting, unsupported, or malformed intent.",
        "Supported action: pick up exactly one supported cube and place it in exactly one bin.",
        "Supported objects: red_cube, green_cube, blue_cube.",
        "Supported bins: left_bin, right_bin.",
        "Statuses: route, reject_ambiguous, reject_unsupported, reject_malformed.",
        "A rejection must use null for both target fields.",
        "Return JSON only. Do not include markdown, prose, or reasoning.",
        "Allowed reasons: explicit_object_and_destination, missing_object, "
        "missing_destination, conflicting_objects, conflicting_destinations, "
        "unresolved_correction, unsupported_object, unsupported_destination, "
        "unsupported_action, unsupported_spatial_reference, multiple_sequential_tasks, "
        "contradictory_negation, meaningless_or_noise, malformed_input, "
        "malformed_model_output.",
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
    # Keep the prompt bounded at six routes plus twelve rejections. Empty text
    # and raw control-character examples are covered by the smoke set rather
    # than embedded into the model prompt.
    omitted = {
        RouterRejectionReason.EMPTY_TEXT,
        RouterRejectionReason.MALFORMED_CONTROL_CHARACTERS,
    }
    for reason in (value for value in rejection_reasons if value not in omitted):
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

_STRUCTURED_REASON_TO_INTERNAL: Mapping[str, RouterRejectionReason] = {
    "missing_object": RouterRejectionReason.MISSING_OBJECT,
    "missing_destination": RouterRejectionReason.MISSING_DESTINATION,
    "conflicting_objects": RouterRejectionReason.CONFLICTING_OBJECTS,
    "conflicting_destinations": RouterRejectionReason.CONFLICTING_BINS,
    "unresolved_correction": RouterRejectionReason.UNRESOLVED_CORRECTION,
    "unsupported_object": RouterRejectionReason.UNSUPPORTED_OBJECT,
    "unsupported_destination": RouterRejectionReason.UNSUPPORTED_DESTINATION,
    "unsupported_action": RouterRejectionReason.UNSUPPORTED_ACTION,
    "unsupported_spatial_reference": RouterRejectionReason.UNSUPPORTED_SPATIAL_REFERENCE,
    "multiple_sequential_tasks": RouterRejectionReason.MULTIPLE_TASKS,
    "contradictory_negation": RouterRejectionReason.CONTRADICTORY_NEGATION,
    "meaningless_or_noise": RouterRejectionReason.MEANINGLESS_TEXT,
    "malformed_input": RouterRejectionReason.MALFORMED_CONTROL_CHARACTERS,
    "malformed_model_output": RouterRejectionReason.STRUCTURED_OUTPUT_INVALID,
}


def _validated_rejection(
    payload: StrictRouterPayload,
) -> tuple[RouterStatus, RouterRejectionReason]:
    try:
        status = RouterStatus(payload.status)
        reason = _STRUCTURED_REASON_TO_INTERNAL[payload.reason]
    except (KeyError, ValueError) as error:
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
            # Persist valid schema-only JSON for the required repair audit. A
            # model may ignore the JSON-only prompt and emit free-form
            # reasoning; that untrusted text is retained only by hash/size.
            "generation_output_fingerprints": [
                f"sha256:{sha256_hex(output)}" for output in outputs
            ],
            "generation_output_character_counts": [len(output) for output in outputs],
            "generation_outputs": [self._safe_output_record(output) for output in outputs],
            "generation_metadata": [
                dict(value)
                for value in getattr(self.generator, "generation_metadata", ())[-len(outputs) :]
            ],
            "parse_errors": list(parse_errors),
            "format_repair_attempts": max(0, len(outputs) - 1),
            "confidence_available": False,
            "reason_code_version": self.config.reason_code_version,
            "chain_of_thought_persisted": False,
        }

    @staticmethod
    def _safe_output_record(output: str) -> dict[str, object]:
        try:
            parse_strict_router_json(output)
        except RouterSchemaError:
            return {
                "stored": False,
                "sha256": f"sha256:{sha256_hex(output)}",
                "character_count": len(output),
                "redaction_reason": "untrusted_non_schema_text",
            }
        return {
            "stored": True,
            "text": output,
            "sha256": f"sha256:{sha256_hex(output)}",
            "character_count": len(output),
            "redaction_reason": None,
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
        self.config = config
        try:
            from huggingface_hub import snapshot_download
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise StructuredLLMRouterError("installed Transformers is required") from error
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.dtype]
        try:
            download_started = time.perf_counter()
            snapshot = Path(
                snapshot_download(
                    config.model_id,
                    revision=config.model_revision,
                    local_files_only=local_files_only,
                )
            )
            self.model_download_seconds = time.perf_counter() - download_started
            self.file_identities = self._validate_snapshot(snapshot, config=config)
            self.tokenizer = AutoTokenizer.from_pretrained(
                snapshot,
                local_files_only=local_files_only,
                trust_remote_code=False,
                use_fast=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                snapshot,
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
        self.snapshot_path = str(snapshot)
        self.snapshot_size_bytes = sum(
            int(value["size_bytes"]) for value in self.file_identities.values()
        )
        self.generation_metadata: list[dict[str, object]] = []
        self.peak_gpu_memory_bytes = 0

    @staticmethod
    def _validate_snapshot(
        snapshot: Path, *, config: StructuredLLMRouterConfig
    ) -> dict[str, dict[str, object]]:
        expected_files = _PINNED_MODEL_FILES.get((config.model_id, config.model_revision))
        if expected_files is None:
            raise StructuredLLMRouterError("configured local model is not an authorized identity")
        if config.tokenizer_revision != config.model_revision:
            raise StructuredLLMRouterError("model and tokenizer revisions must match exactly")
        expected_mode = (
            "official_qwen_enable_thinking_false"
            if config.model_id == QWEN3_1_7B_MODEL_ID
            else "official_qwen_instruct_non_thinking"
        )
        if config.chat_template_mode != expected_mode:
            raise StructuredLLMRouterError("model identity and chat-template transport differ")
        actual: dict[str, dict[str, object]] = {}
        for name, (expected_size, expected_sha) in expected_files.items():
            path = snapshot / name
            if not path.is_file():
                raise StructuredLLMRouterError(f"pinned model snapshot omitted {name}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            observed = digest.hexdigest()
            size = path.stat().st_size
            if size != expected_size or observed != expected_sha:
                raise StructuredLLMRouterError(f"pinned model file identity differs: {name}")
            actual[name] = {"size_bytes": size, "sha256": f"sha256:{observed}"}
        return actual

    def _chat_template_kwargs(self) -> dict[str, object]:
        if self.config.chat_template_mode == "official_qwen_enable_thinking_false":
            return {"enable_thinking": False}
        if self.config.chat_template_mode == "official_qwen_instruct_non_thinking":
            return {}
        raise StructuredLLMRouterError("chat-template transport is not authorized")

    def rendered_prompt_fingerprint(self, prompt: str) -> str:
        """Hash the official tokenizer transport without changing prompt semantics."""

        try:
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
                **self._chat_template_kwargs(),
            )
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise StructuredLLMRouterError(
                f"configured instruct tokenizer lacks a usable chat template: {error}"
            ) from error
        if not isinstance(rendered, str) or not rendered:
            raise StructuredLLMRouterError("chat template did not render one non-empty prompt")
        return f"sha256:{sha256_hex(rendered)}"

    def generate(self, prompt: str, *, max_new_tokens: int) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                **self._chat_template_kwargs(),
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
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **model_inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        if (
            not isinstance(generated, torch.Tensor)
            or generated.ndim != 2
            or generated.shape[0] != 1
        ):
            raise StructuredLLMRouterError("local generation returned an unexpected tensor shape")
        output_ids = generated[0, input_length:]
        elapsed = time.perf_counter() - started
        peak = 0 if self.device != "cuda" else int(torch.cuda.max_memory_allocated())
        self.peak_gpu_memory_bytes = max(self.peak_gpu_memory_bytes, peak)
        self.generation_metadata.append(
            {
                "generated_token_count": int(output_ids.shape[0]),
                "elapsed_seconds": elapsed,
                "peak_gpu_memory_bytes": peak,
                "do_sample": False,
                "enable_thinking": False,
                "chat_template_mode": self.config.chat_template_mode,
            }
        )
        return cast(str, self.tokenizer.decode(output_ids, skip_special_tokens=True))


__all__ = [
    "STRUCTURED_LLM_PROMPT_VERSION",
    "STRUCTURED_LLM_ROUTER_NAME",
    "STRUCTURED_LLM_ROUTER_VERSION",
    "QWEN3_1_7B_FILE_IDENTITIES",
    "QWEN3_1_7B_LICENSE",
    "QWEN3_1_7B_MODEL_ID",
    "QWEN3_1_7B_REVISION",
    "QWEN3_4B_INSTRUCT_FILE_IDENTITIES",
    "QWEN3_4B_INSTRUCT_LICENSE",
    "QWEN3_4B_INSTRUCT_MODEL_ID",
    "QWEN3_4B_INSTRUCT_REVISION",
    "LocalTextGenerator",
    "StructuredLLMRouterConfig",
    "StructuredLLMRouterError",
    "StructuredLocalLLMRouterV0",
    "TransformersLocalTextGenerator",
    "build_structured_routing_prompt",
    "select_structured_routing_prompt_examples",
]
