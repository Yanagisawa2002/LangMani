from __future__ import annotations

import json

import pytest

from langmani.language.llm_router import StructuredLLMRouterConfig, StructuredLocalLLMRouterV0
from langmani.language.router_types import RouterStatus


class _DeterministicFixtureGenerator:
    def generate(self, prompt: str, *, max_new_tokens: int) -> str:
        del prompt, max_new_tokens
        return json.dumps(
            {
                "status": "route",
                "target_object_id": "green_cube",
                "target_bin_id": "left_bin",
                "reason": "explicit_object_and_destination",
            },
            sort_keys=True,
        )


@pytest.mark.integration
@pytest.mark.fixture
def test_fixture_structured_llm_adapter_is_deterministic_and_local() -> None:
    config = StructuredLLMRouterConfig(
        model_id="fixture/never-downloaded",
        model_revision="1" * 40,
        tokenizer_revision="2" * 40,
        dtype="float32",
        maximum_new_tokens=32,
    )
    router = StructuredLocalLLMRouterV0(
        config=config,
        generator=_DeterministicFixtureGenerator(),
    )

    first = router.route("Move the green cube to the left bin.")
    second = router.route("Move the green cube to the left bin.")

    assert first.status is RouterStatus.ROUTE
    assert first.decision_fingerprint == second.decision_fingerprint
    assert first.confidence.available is False
    assert first.evidence["config_fingerprint"] == config.fingerprint
