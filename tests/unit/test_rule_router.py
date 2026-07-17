from __future__ import annotations

import pytest

from langmani.language.router_types import RouterRejectionReason, RouterStatus
from langmani.language.rule_router import RuleRouterV0, normalize_rule_text


@pytest.mark.parametrize(
    ("command", "object_id", "bin_id"),
    (
        ("Pick up the red cube and put it in the left bin.", "red_cube", "left_bin"),
        ("Please move the blue block into the right container.", "blue_cube", "right_bin"),
        ("Into the left receptacle, place the emerald cube.", "green_cube", "left_bin"),
    ),
)
def test_rule_router_direct_commands_and_declared_synonyms(
    command: str, object_id: str, bin_id: str
) -> None:
    decision = RuleRouterV0().route(command)

    assert decision.status is RouterStatus.ROUTE
    assert decision.target_object_id == object_id
    assert decision.target_bin_id == bin_id
    assert decision.task_spec is not None
    assert decision.confidence.available is False


def test_rule_router_rejects_object_conflict_without_guessing() -> None:
    decision = RuleRouterV0().route("Move the red cube and the blue cube into the right bin.")

    assert decision.status is RouterStatus.REJECT_AMBIGUOUS
    assert decision.rejection_reason is RouterRejectionReason.CONFLICTING_OBJECTS
    assert decision.task_spec is None
    assert decision.task_id is None


def test_rule_router_rejects_bin_conflict_without_guessing() -> None:
    decision = RuleRouterV0().route("Put the green cube in the left bin and the right bin.")

    assert decision.status is RouterStatus.REJECT_AMBIGUOUS
    assert decision.rejection_reason is RouterRejectionReason.CONFLICTING_BINS


def test_rule_router_negation_excludes_distractor_but_preserves_positive_target() -> None:
    decision = RuleRouterV0().route("Ignore the blue cube and put the green cube in the left bin.")

    assert decision.status is RouterStatus.ROUTE
    assert decision.target_object_id == "green_cube"
    assert decision.target_bin_id == "left_bin"
    assert len(decision.evidence["negation_indicators"]) >= 1


def test_rule_router_rejects_unresolved_contradictory_negation() -> None:
    decision = RuleRouterV0().route("Do not put the red cube in the left bin.")

    assert decision.status is RouterStatus.REJECT_AMBIGUOUS
    assert decision.rejection_reason is RouterRejectionReason.CONTRADICTORY_NEGATION


@pytest.mark.parametrize(
    ("command", "reason"),
    (
        ("Place the yellow cube in the left bin.", RouterRejectionReason.UNSUPPORTED_OBJECT),
        ("Place the red cube in the drawer.", RouterRejectionReason.UNSUPPORTED_DESTINATION),
        ("Open the drawer.", RouterRejectionReason.UNSUPPORTED_DESTINATION),
        ("Move it over there.", RouterRejectionReason.UNSUPPORTED_DESTINATION),
    ),
)
def test_rule_router_rejects_unsupported_concepts(
    command: str, reason: RouterRejectionReason
) -> None:
    decision = RuleRouterV0().route(command)

    assert decision.status is RouterStatus.REJECT_UNSUPPORTED
    assert decision.rejection_reason is reason


@pytest.mark.parametrize(
    ("command", "reason"),
    (
        ("", RouterRejectionReason.EMPTY_TEXT),
        (" \t\n ", RouterRejectionReason.EMPTY_TEXT),
        ("\x00red cube left bin", RouterRejectionReason.MALFORMED_CONTROL_CHARACTERS),
        ("!? 123", RouterRejectionReason.MEANINGLESS_TEXT),
    ),
)
def test_rule_router_rejects_malformed_text(command: str, reason: RouterRejectionReason) -> None:
    decision = RuleRouterV0().route(command)

    assert decision.status is RouterStatus.REJECT_MALFORMED
    assert decision.rejection_reason is reason


def test_rule_normalization_does_not_remove_negation() -> None:
    assert normalize_rule_text("DON’T move the red cube!") == "don't move the red cube"


def test_rule_router_is_deterministically_fingerprinted() -> None:
    router = RuleRouterV0()
    first = router.route("Move the red cube into the left bin.")
    second = router.route("Move the red cube into the left bin.")

    assert first.decision_fingerprint == second.decision_fingerprint
    assert first.evidence["config_fingerprint"] == router.config.fingerprint
