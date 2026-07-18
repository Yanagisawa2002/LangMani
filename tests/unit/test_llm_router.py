from __future__ import annotations

import json
from collections import deque

import pytest

from langmani.environments.specs import TaskSpec
from langmani.language.corpus import build_language_corpus
from langmani.language.llm_router import (
    StructuredLLMRouterConfig,
    StructuredLLMRouterError,
    StructuredLocalLLMRouterV0,
    build_structured_routing_prompt,
    select_structured_routing_prompt_examples,
)
from langmani.language.router_types import (
    LanguageExample,
    LanguageSplit,
    RouterRejectionReason,
    RouterStatus,
    stable_language_example_id,
)
from langmani.language.schema_validation import RouterSchemaError, parse_strict_router_json


class _Generator:
    def __init__(self, *outputs: str) -> None:
        self.outputs = deque(outputs)
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, max_new_tokens: int) -> str:
        assert max_new_tokens == 64
        self.prompts.append(prompt)
        return self.outputs.popleft()


def _config(*, repairs: int = 1) -> StructuredLLMRouterConfig:
    return StructuredLLMRouterConfig(
        model_id="fixture/local-instruct",
        model_revision="a" * 40,
        tokenizer_revision="b" * 40,
        dtype="float32",
        maximum_new_tokens=64,
        maximum_format_repair_attempts=repairs,
    )


def _route_json() -> str:
    return json.dumps(
        {
            "status": "route",
            "target_object_id": "red_cube",
            "target_bin_id": "right_bin",
            "reason": "explicit_object_and_destination",
        }
    )


@pytest.mark.parametrize(
    "raw",
    (
        '```json\n{"status":"route"}\n```',
        '{"status":"route","status":"route","target_object_id":"red_cube",'
        '"target_bin_id":"left_bin","reason":"explicit_object_and_destination"}',
        '{"status":"route","target_object_id":"red_cube","target_bin_id":"left_bin",'
        '"reason":"explicit_object_and_destination","extra":1}',
        '{"status":"route","target_object_id":"red_cube","target_bin_id":null,'
        '"reason":"explicit_object_and_destination"}',
        '{"status":"route","target_object_id":"red_cube","target_bin_id":"left_bin","reason":NaN}',
    ),
)
def test_strict_llm_json_rejects_unsafe_or_non_exact_output(raw: str) -> None:
    with pytest.raises(RouterSchemaError):
        parse_strict_router_json(raw)


def test_structured_llm_routes_only_after_strict_validation() -> None:
    router = StructuredLocalLLMRouterV0(config=_config(), generator=_Generator(_route_json()))

    decision = router.route("Move the red cube to the right bin.")

    assert decision.status is RouterStatus.ROUTE
    assert decision.target_object_id == "red_cube"
    assert decision.target_bin_id == "right_bin"
    assert decision.confidence.available is False
    assert decision.evidence["format_repair_attempts"] == 0


def test_structured_llm_preserves_explicit_rejection() -> None:
    output = json.dumps(
        {
            "status": "reject_unsupported",
            "target_object_id": None,
            "target_bin_id": None,
            "reason": "unsupported_object",
        }
    )
    decision = StructuredLocalLLMRouterV0(config=_config(), generator=_Generator(output)).route(
        "Move the yellow cube."
    )

    assert decision.status is RouterStatus.REJECT_UNSUPPORTED
    assert decision.rejection_reason is RouterRejectionReason.UNSUPPORTED_OBJECT
    assert decision.task_spec is None


def test_structured_llm_allows_exactly_one_format_repair() -> None:
    invalid = "private free-form reasoning that is not JSON"
    generator = _Generator(invalid, _route_json())
    decision = StructuredLocalLLMRouterV0(config=_config(), generator=generator).route("command")

    assert decision.status is RouterStatus.ROUTE
    assert decision.evidence["format_repair_attempts"] == 1
    assert invalid not in repr(decision.evidence)
    outputs = decision.evidence["generation_outputs"]
    assert outputs[0]["stored"] is False
    assert outputs[1]["stored"] is True
    assert outputs[1]["text"] == _route_json()
    assert len(decision.evidence["generation_output_fingerprints"]) == 2
    assert len(generator.prompts) == 2
    assert "Repair only the JSON format" in generator.prompts[1]


def test_structured_llm_rejects_malformed_after_bounded_repair() -> None:
    generator = _Generator("not json", "still not json")
    decision = StructuredLocalLLMRouterV0(config=_config(), generator=generator).route("command")

    assert decision.status is RouterStatus.REJECT_MALFORMED
    assert decision.rejection_reason is RouterRejectionReason.FORMAT_REPAIR_EXHAUSTED
    assert decision.task_spec is None
    assert len(generator.prompts) == 2


def _training_example() -> LanguageExample:
    task = TaskSpec("blue_cube", "left_bin", "canonical_v0")
    values = {
        "raw_text": "Please move the blue cube to the left bin.",
        "normalized_text": "please move the blue cube to the left bin.",
        "template_family_id": "train-polite-v0",
        "lexical_variant_ids": ("please",),
        "expected_status": RouterStatus.ROUTE,
        "expected_task_spec": task,
        "expected_rejection_reason": None,
        "generation_provenance": {"generator": "fixture"},
        "split": LanguageSplit.TRAIN,
    }
    return LanguageExample(
        example_id=stable_language_example_id(**values),
        **values,
    )


def test_prompt_fingerprint_is_stable_and_uses_train_examples_only() -> None:
    example = _training_example()
    first = build_structured_routing_prompt(examples=(example,))
    second = build_structured_routing_prompt(examples=(example,))

    assert first == second
    assert first[2] == (example.example_id,)
    invalid = object.__new__(LanguageExample)
    object.__setattr__(invalid, "split", LanguageSplit.DEVELOPMENT)
    with pytest.raises(StructuredLLMRouterError, match="train families only"):
        build_structured_routing_prompt(examples=(invalid,))


def test_canonical_prompt_selection_is_stable_and_covers_tasks_and_rejections() -> None:
    corpus = build_language_corpus()
    train = corpus.examples_for_split(LanguageSplit.TRAIN)

    first = select_structured_routing_prompt_examples(train)
    second = select_structured_routing_prompt_examples(tuple(reversed(train)))

    assert tuple(example.example_id for example in first) == tuple(
        example.example_id for example in second
    )
    assert len({example.task_id for example in first if example.task_id is not None}) == 6
    expected_reasons = {
        example.expected_rejection_reason
        for example in train
        if example.expected_rejection_reason is not None
    }
    selected_reasons = {
        example.expected_rejection_reason
        for example in first
        if example.expected_rejection_reason is not None
    }
    assert selected_reasons == expected_reasons - {
        RouterRejectionReason.EMPTY_TEXT,
        RouterRejectionReason.MALFORMED_CONTROL_CHARACTERS,
    }


def test_structured_llm_config_records_greedy_temperature_semantics() -> None:
    payload = _config().to_dict()

    assert payload["do_sample"] is False
    assert payload["requested_temperature"] == 0.0
    assert payload["effective_sampling_temperature"] is None
    assert payload["quantization"] == "none"
