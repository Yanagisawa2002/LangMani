"""Exact routing-versus-control failure attribution for M5A.

The attachment enumerates fourteen categories even though an earlier prose count
called them thirteen.  This module preserves all fourteen values.  Correctly
rejected language-only examples return ``None`` because they never become control
episodes; every routeable control episode receives exactly one category.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from langmani.environments.specs import TaskSpec, stable_task_id
from langmani.language.router_types import RouterDecision, RouterStatus


class FailureAttributionError(ValueError):
    """Raised when episode evidence is contradictory or incomplete."""


class FailureAttribution(StrEnum):
    """The fourteen failure-decomposition outcomes declared by M5A."""

    ROUTING_CORRECT_CONTROL_SUCCESS = "routing_correct_control_success"
    ROUTING_CORRECT_CONTROL_FAILURE = "routing_correct_control_failure"
    ROUTING_WRONG_OBJECT = "routing_wrong_object"
    ROUTING_WRONG_BIN = "routing_wrong_bin"
    ROUTING_WRONG_TASK = "routing_wrong_task"
    ROUTING_FALSE_REJECTION = "routing_false_rejection"
    ROUTING_MISSED_REJECTION = "routing_missed_rejection"
    ROUTING_MALFORMED_OUTPUT = "routing_malformed_output"
    CONTROLLER_LOAD_FAILURE = "controller_load_failure"
    CONTROLLER_INFERENCE_FAILURE = "controller_inference_failure"
    INVALID_ACTION = "invalid_action"
    TIMEOUT = "timeout"
    ENVIRONMENT_FAILURE = "environment_failure"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


@dataclass(frozen=True, slots=True)
class FailureAttributionEvidence:
    """Minimal mutually exclusive evidence used to attribute one outcome."""

    expected_task_spec: TaskSpec | None
    decision: RouterDecision | None
    controller_task_id: str | None = None
    controller_dispatched: bool = False
    control_success: bool | None = None
    router_output_malformed: bool = False
    controller_load_failed: bool = False
    controller_inference_failed: bool = False
    invalid_action: bool = False
    timeout: bool = False
    environment_failed: bool = False
    infrastructure_failed: bool = False

    def __post_init__(self) -> None:
        for name in (
            "controller_dispatched",
            "router_output_malformed",
            "controller_load_failed",
            "controller_inference_failed",
            "invalid_action",
            "timeout",
            "environment_failed",
            "infrastructure_failed",
        ):
            if not isinstance(getattr(self, name), bool):
                raise FailureAttributionError(f"{name} must be boolean")
        if self.control_success is not None and not isinstance(self.control_success, bool):
            raise FailureAttributionError("control_success must be boolean or None")
        failures = sum(
            int(value)
            for value in (
                self.controller_load_failed,
                self.controller_inference_failed,
                self.invalid_action,
                self.timeout,
                self.environment_failed,
                self.infrastructure_failed,
            )
        )
        if failures > 1:
            raise FailureAttributionError("operational failure flags must be mutually exclusive")
        if self.control_success is True and failures:
            raise FailureAttributionError(
                "successful control cannot contain an operational failure"
            )
        if self.controller_load_failed and self.controller_dispatched:
            raise FailureAttributionError("a controller that failed to load cannot be dispatched")
        if self.controller_dispatched and self.controller_task_id is None:
            raise FailureAttributionError("dispatched control requires controller_task_id")
        if not self.controller_dispatched and self.control_success is not None:
            raise FailureAttributionError("non-dispatched control cannot report control_success")
        if self.router_output_malformed and self.decision is not None:
            raise FailureAttributionError("malformed router output cannot also be a RouterDecision")


def _wrong_route(expected: TaskSpec, actual: TaskSpec) -> FailureAttribution | None:
    wrong_object = actual.target_object_id != expected.target_object_id
    wrong_bin = actual.target_bin_id != expected.target_bin_id
    if wrong_object and wrong_bin:
        return FailureAttribution.ROUTING_WRONG_TASK
    if wrong_object:
        return FailureAttribution.ROUTING_WRONG_OBJECT
    if wrong_bin:
        return FailureAttribution.ROUTING_WRONG_BIN
    if stable_task_id(actual) != stable_task_id(expected):
        return FailureAttribution.ROUTING_WRONG_TASK
    return None


def attribute_end_to_end(
    evidence: FailureAttributionEvidence,
) -> FailureAttribution | None:
    """Attribute one language/control result without disguising routing errors.

    ``expected_task_spec=None`` denotes a rejection-language example.  A correct
    rejection returns ``None`` and is recorded as safe rejection by the dispatcher;
    it is not a control episode.  A routeable control example always returns one of
    the fourteen declared categories.
    """

    if evidence.router_output_malformed or evidence.decision is None:
        return FailureAttribution.ROUTING_MALFORMED_OUTPUT
    decision = evidence.decision
    if evidence.expected_task_spec is None:
        if decision.status is RouterStatus.ROUTE:
            return FailureAttribution.ROUTING_MISSED_REJECTION
        if evidence.controller_dispatched:
            raise FailureAttributionError("correct rejection must not dispatch a controller")
        return None
    expected = evidence.expected_task_spec
    if decision.status is not RouterStatus.ROUTE:
        if evidence.controller_dispatched:
            raise FailureAttributionError("false rejection must not dispatch a controller")
        return FailureAttribution.ROUTING_FALSE_REJECTION
    if decision.task_spec is None or decision.task_id is None:
        return FailureAttribution.ROUTING_MALFORMED_OUTPUT
    routing_error = _wrong_route(expected, decision.task_spec)
    if routing_error is not None:
        return routing_error
    if evidence.controller_load_failed:
        return FailureAttribution.CONTROLLER_LOAD_FAILURE
    if not evidence.controller_dispatched:
        return FailureAttribution.INFRASTRUCTURE_FAILURE
    if evidence.controller_task_id != decision.task_id:
        return FailureAttribution.INFRASTRUCTURE_FAILURE
    if evidence.controller_inference_failed:
        return FailureAttribution.CONTROLLER_INFERENCE_FAILURE
    if evidence.invalid_action:
        return FailureAttribution.INVALID_ACTION
    if evidence.timeout:
        return FailureAttribution.TIMEOUT
    if evidence.environment_failed:
        return FailureAttribution.ENVIRONMENT_FAILURE
    if evidence.infrastructure_failed:
        return FailureAttribution.INFRASTRUCTURE_FAILURE
    if evidence.control_success is True:
        return FailureAttribution.ROUTING_CORRECT_CONTROL_SUCCESS
    if evidence.control_success is False:
        return FailureAttribution.ROUTING_CORRECT_CONTROL_FAILURE
    return FailureAttribution.INFRASTRUCTURE_FAILURE


__all__ = [
    "FailureAttribution",
    "FailureAttributionError",
    "FailureAttributionEvidence",
    "attribute_end_to_end",
]
