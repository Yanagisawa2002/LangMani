from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from langmani.datasets.identity import sha256_hex
from langmani.environments.specs import TaskSpec, stable_task_id
from langmani.language.controller_registry import build_fixture_controller_registry
from langmani.language.corpus import build_language_corpus
from langmani.language.dispatcher import (
    ControllerDispatcher,
    ControllerExecutionTrace,
    FixturePerTaskControllerLoader,
)
from langmani.language.evaluation import ActionStreamEvidence, ControlExecutionResult
from langmani.language.failure_attribution import FailureAttribution
from langmani.language.llm_router import (
    StructuredLLMRouterConfig,
    StructuredLocalLLMRouterV0,
)
from langmani.language.router_types import RouterDecision, RouterStatus
from langmani.language.rule_router import RuleRouterV0
from langmani.language.schedules import (
    REQUIRED_EXCLUSION_SOURCE_IDS,
    ControlScheduleConfig,
    FinalControlScheduleAccessError,
    SeedExclusionSource,
    build_m5a_schedule_bundle,
    materialize_control_schedule,
)
from langmani.language.text_calibration import TemperatureCalibrationV0
from langmani.language.text_classifier import (
    FactorizedTextClassifierV0,
    FactorizedTextRouterConfig,
    FactorizedTextRouterV0,
)


@dataclass
class _SpyEnvironment:
    reset_count: int = 0
    step_count: int = 0
    reset_task_specs: list[dict[str, str]] = field(default_factory=list)

    def reset(self, *, seed: int, options: dict[str, object]) -> tuple[object, dict[str, object]]:
        assert seed >= 0
        self.reset_count += 1
        self.reset_task_specs.append(dict(options["task_spec"]))
        return {}, {}

    def step(self, action: np.ndarray) -> tuple[object, float, bool, bool, dict[str, object]]:
        assert action.shape == (8,)
        self.step_count += 1
        return {}, 0.0, False, False, {}


class _TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(dim=8)
        self.embedding = nn.Embedding(8, 8)

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> object:
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class _Tokenizer:
    def __call__(
        self,
        text: str,
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        del text, padding, truncation, max_length, return_tensors
        return {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }


def _classifier_router(*, status: int, target_object: int, target_bin: int):
    model = FactorizedTextClassifierV0(_TinyEncoder(), dropout=0.0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.status_head.bias[status] = 10.0
        model.object_head.bias[target_object] = 10.0
        model.bin_head.bias[target_bin] = 10.0
    return FactorizedTextRouterV0(
        model=model,
        tokenizer=_Tokenizer(),
        calibration=TemperatureCalibrationV0(
            temperature=1.0,
            validation_examples=1,
            nll_before=0.1,
            nll_after=0.1,
            bounded_iterations=1,
        ),
        config=FactorizedTextRouterConfig(
            maximum_sequence_length=16,
            routing_threshold=0.0,
        ),
    )


class _Generator:
    def __init__(self, *outputs: str) -> None:
        self.outputs = deque(outputs)

    def generate(self, prompt: str, *, max_new_tokens: int) -> str:
        assert prompt and max_new_tokens == 64
        return self.outputs.popleft()


def _llm_router(payload: dict[str, object]) -> StructuredLocalLLMRouterV0:
    return StructuredLocalLLMRouterV0(
        config=StructuredLLMRouterConfig(
            model_id="fixture/local-instruct",
            model_revision="a" * 40,
            tokenizer_revision="b" * 40,
            dtype="float32",
            maximum_new_tokens=64,
            maximum_format_repair_attempts=0,
        ),
        generator=_Generator(json.dumps(payload)),
    )


def _dispatch(decision: RouterDecision, *, oracle: TaskSpec):
    registry = build_fixture_controller_registry()
    loader = FixturePerTaskControllerLoader()
    environment = _SpyEnvironment()
    result = ControllerDispatcher(registry=registry, loader=loader).dispatch(
        decision,
        environment=environment,
        evaluation_id="integration-router-dispatch",
        oracle_task_spec=oracle,
        scene_seed=123,
    )
    return result, loader, environment


@pytest.mark.parametrize(
    ("decision", "oracle"),
    (
        (
            RuleRouterV0().route("Move the red cube into the left bin."),
            TaskSpec("red_cube", "left_bin", "canonical_v0"),
        ),
        (
            _classifier_router(status=0, target_object=1, target_bin=1).route(
                "Place the green cube in the right bin."
            ),
            TaskSpec("green_cube", "right_bin", "canonical_v0"),
        ),
        (
            _llm_router(
                {
                    "status": "route",
                    "target_object_id": "blue_cube",
                    "target_bin_id": "left_bin",
                    "reason": "route",
                }
            ).route("Put the blue cube in the left bin."),
            TaskSpec("blue_cube", "left_bin", "canonical_v0"),
        ),
    ),
    ids=("rule", "classifier", "llm"),
)
def test_three_router_fixtures_dispatch_one_frozen_controller(
    decision: RouterDecision,
    oracle: TaskSpec,
) -> None:
    result, loader, environment = _dispatch(decision, oracle=oracle)

    assert result.end_to_end_success
    assert result.dispatch.controller_task_id == stable_task_id(oracle)
    assert result.failure_attribution is FailureAttribution.ROUTING_CORRECT_CONTROL_SUCCESS
    assert loader.load_count == 1
    assert environment.reset_count == environment.step_count == 1
    assert environment.reset_task_specs == [oracle.to_dict()]


@pytest.mark.parametrize(
    "decision",
    (
        RuleRouterV0().route("Move the red and blue cubes into the left bin."),
        _classifier_router(status=1, target_object=0, target_bin=0).route(
            "Move something to the left bin."
        ),
        _llm_router(
            {
                "status": "reject_unsupported",
                "target_object_id": None,
                "target_bin_id": None,
                "reason": "unsupported_object",
            }
        ).route("Move the yellow cube."),
    ),
    ids=("rule", "classifier", "llm"),
)
def test_rejection_performs_zero_controller_and_environment_work(
    decision: RouterDecision,
) -> None:
    assert decision.status is not RouterStatus.ROUTE
    oracle = TaskSpec("red_cube", "left_bin", "canonical_v0")
    result, loader, environment = _dispatch(decision, oracle=oracle)

    assert result.failure_attribution is FailureAttribution.ROUTING_FALSE_REJECTION
    assert result.dispatch.safe_rejection
    assert loader.load_count == 0
    assert loader.executors == {}
    assert environment.reset_count == environment.step_count == 0


class _PlainControlFailureExecutor:
    def __init__(self, task_id: str, environment: _SpyEnvironment) -> None:
        self.controller_task_id = task_id
        self.environment = environment

    def run_episode(
        self,
        *,
        evaluation_id: str,
        scene_seed: int,
        oracle_task_spec: TaskSpec,
    ) -> ControllerExecutionTrace:
        del evaluation_id
        self.environment.reset(
            seed=scene_seed,
            options={"task_spec": oracle_task_spec.to_dict()},
        )
        self.environment.step(np.zeros(8, dtype=np.float32))
        return ControllerExecutionTrace(
            result=ControlExecutionResult(
                success=False,
                status="control_failure",
                environment_step_count=1,
                final_evaluation={"success": False},
                action_evidence=ActionStreamEvidence.empty(),
            ),
            policy_called=True,
            environment_reset_called=True,
        )


class _PlainControlFailureLoader:
    def load(self, entry, *, environment: object):
        assert isinstance(environment, _SpyEnvironment)
        return _PlainControlFailureExecutor(entry.task_id, environment)


def test_correct_route_keeps_controller_failure_separate_from_language() -> None:
    registry = build_fixture_controller_registry()
    environment = _SpyEnvironment()
    oracle = TaskSpec("green_cube", "left_bin", "canonical_v0")
    decision = RuleRouterV0().route("Place the green cube in the left bin.")

    result = ControllerDispatcher(
        registry=registry,
        loader=_PlainControlFailureLoader(),
    ).dispatch(
        decision,
        environment=environment,
        evaluation_id="correct-routing-control-failure",
        oracle_task_spec=oracle,
        scene_seed=7,
    )

    assert result.failure_attribution is FailureAttribution.ROUTING_CORRECT_CONTROL_FAILURE
    assert result.dispatch.controller_task_id == stable_task_id(oracle)
    assert result.control is not None and result.control.success is False
    assert environment.reset_count == environment.step_count == 1


def test_wrong_route_is_attributed_before_controller_outcome() -> None:
    oracle = TaskSpec("red_cube", "right_bin", "canonical_v0")
    decision = RuleRouterV0().route("Move the blue cube into the right bin.")

    result, _loader, environment = _dispatch(decision, oracle=oracle)

    assert result.failure_attribution is FailureAttribution.ROUTING_WRONG_OBJECT
    assert result.dispatch.predicted_task_id == stable_task_id(
        TaskSpec("blue_cube", "right_bin", "canonical_v0")
    )
    assert result.dispatch.oracle_task_id == stable_task_id(oracle)
    assert environment.reset_task_specs == [oracle.to_dict()]


def test_integration_dispatch_never_materializes_final_schedule() -> None:
    corpus = build_language_corpus()
    exclusions = tuple(
        SeedExclusionSource(
            source_id=source_id,
            source_fingerprint=f"sha256:{sha256_hex({'source': source_id})}",
            scene_seeds=(),
        )
        for source_id in REQUIRED_EXCLUSION_SOURCE_IDS
    )
    bundle = build_m5a_schedule_bundle(
        corpus=corpus,
        config=ControlScheduleConfig(exclusion_sources=exclusions),
    )

    with pytest.raises(FinalControlScheduleAccessError, match="sealed"):
        materialize_control_schedule(bundle.control_final)

    assert bundle.language_final.sealed is True
    assert bundle.control_final.sealed is True
