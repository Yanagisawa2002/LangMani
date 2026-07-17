from __future__ import annotations

from dataclasses import replace

import pytest

from langmani.environments.specs import TaskSpec, stable_task_id
from langmani.language.failure_attribution import (
    FailureAttribution,
    FailureAttributionError,
    FailureAttributionEvidence,
    attribute_end_to_end,
)
from langmani.language.router_types import (
    RouterConfidence,
    RouterDecision,
    RouterRejectionReason,
    RouterStatus,
)

RED_LEFT = TaskSpec("red_cube", "left_bin", "canonical_v0")
BLUE_LEFT = TaskSpec("blue_cube", "left_bin", "canonical_v0")
RED_RIGHT = TaskSpec("red_cube", "right_bin", "canonical_v0")
BLUE_RIGHT = TaskSpec("blue_cube", "right_bin", "canonical_v0")


def _route(task_spec: TaskSpec) -> RouterDecision:
    return RouterDecision.route(
        task_spec=task_spec,
        confidence=RouterConfidence.unavailable(),
        router_name="fixture",
        router_version="v0",
    )


def _evidence(**overrides: object) -> FailureAttributionEvidence:
    values: dict[str, object] = {
        "expected_task_spec": RED_LEFT,
        "decision": _route(RED_LEFT),
        "controller_task_id": stable_task_id(RED_LEFT),
        "controller_dispatched": True,
        "control_success": True,
    }
    values.update(overrides)
    return FailureAttributionEvidence(**values)


def test_failure_attribution_preserves_all_fourteen_declared_values() -> None:
    assert [item.value for item in FailureAttribution] == [
        "routing_correct_control_success",
        "routing_correct_control_failure",
        "routing_wrong_object",
        "routing_wrong_bin",
        "routing_wrong_task",
        "routing_false_rejection",
        "routing_missed_rejection",
        "routing_malformed_output",
        "controller_load_failure",
        "controller_inference_failure",
        "invalid_action",
        "timeout",
        "environment_failure",
        "infrastructure_failure",
    ]


@pytest.mark.parametrize(
    ("task_spec", "expected"),
    [
        (BLUE_LEFT, FailureAttribution.ROUTING_WRONG_OBJECT),
        (RED_RIGHT, FailureAttribution.ROUTING_WRONG_BIN),
        (BLUE_RIGHT, FailureAttribution.ROUTING_WRONG_TASK),
    ],
)
def test_wrong_routes_are_attributed_before_downstream_control(
    task_spec: TaskSpec,
    expected: FailureAttribution,
) -> None:
    evidence = _evidence(
        decision=_route(task_spec),
        controller_task_id=stable_task_id(task_spec),
        control_success=False,
    )

    assert attribute_end_to_end(evidence) is expected


def test_routeable_rejection_is_false_rejection_but_safe_rejection_is_not_control() -> None:
    rejection = RouterDecision.reject(
        status=RouterStatus.REJECT_AMBIGUOUS,
        rejection_reason=RouterRejectionReason.CONFLICTING_OBJECTS,
        confidence=RouterConfidence.unavailable(),
        router_name="fixture",
        router_version="v0",
    )

    assert (
        attribute_end_to_end(
            FailureAttributionEvidence(expected_task_spec=RED_LEFT, decision=rejection)
        )
        is FailureAttribution.ROUTING_FALSE_REJECTION
    )
    assert (
        attribute_end_to_end(
            FailureAttributionEvidence(expected_task_spec=None, decision=rejection)
        )
        is None
    )
    assert (
        attribute_end_to_end(
            FailureAttributionEvidence(expected_task_spec=None, decision=_route(RED_LEFT))
        )
        is FailureAttribution.ROUTING_MISSED_REJECTION
    )


def test_malformed_router_output_is_separate() -> None:
    assert (
        attribute_end_to_end(
            FailureAttributionEvidence(
                expected_task_spec=RED_LEFT,
                decision=None,
                router_output_malformed=True,
            )
        )
        is FailureAttribution.ROUTING_MALFORMED_OUTPUT
    )


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"control_success": False}, FailureAttribution.ROUTING_CORRECT_CONTROL_FAILURE),
        (
            {
                "controller_task_id": None,
                "controller_dispatched": False,
                "control_success": None,
                "controller_load_failed": True,
            },
            FailureAttribution.CONTROLLER_LOAD_FAILURE,
        ),
        (
            {"control_success": None, "controller_inference_failed": True},
            FailureAttribution.CONTROLLER_INFERENCE_FAILURE,
        ),
        ({"control_success": None, "invalid_action": True}, FailureAttribution.INVALID_ACTION),
        ({"control_success": None, "timeout": True}, FailureAttribution.TIMEOUT),
        (
            {"control_success": None, "environment_failed": True},
            FailureAttribution.ENVIRONMENT_FAILURE,
        ),
        (
            {"control_success": None, "infrastructure_failed": True},
            FailureAttribution.INFRASTRUCTURE_FAILURE,
        ),
    ],
)
def test_control_and_operational_failures_remain_separate(
    changes: dict[str, object],
    expected: FailureAttribution,
) -> None:
    assert attribute_end_to_end(_evidence(**changes)) is expected


def test_successful_correct_route_is_control_success() -> None:
    assert attribute_end_to_end(_evidence()) is FailureAttribution.ROUTING_CORRECT_CONTROL_SUCCESS


def test_evidence_rejects_contradictory_failure_signals() -> None:
    with pytest.raises(FailureAttributionError, match="mutually exclusive"):
        replace(_evidence(), control_success=None, timeout=True, environment_failed=True)
    with pytest.raises(FailureAttributionError, match="failed to load"):
        replace(_evidence(), control_success=None, controller_load_failed=True)
