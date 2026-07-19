"""Explicit target-only test for the one-reset, zero-step bridge proposal."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from langmani.integrations.latentguard_bridge import (
    INITIAL_PROPOSAL_SCHEMA,
    POLICY_BINDING_SCHEMA,
    derive_probe_seed,
    export_initial_proposal,
    load_runtime_registry,
    validate_envelope,
)

_REQUIRED = (
    "LANGMANI_LATENTGUARD_BINDING",
    "LANGMANI_CHECKPOINT_ROOT",
    "LANGMANI_DATASET_ROOT",
    "LANGMANI_RUNTIME_SELECTION",
)


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.linux
@pytest.mark.target
@pytest.mark.skipif(
    any(not os.environ.get(name) for name in _REQUIRED),
    reason="real bridge probe paths are not configured",
)
def test_real_bridge_probe_resets_once_and_executes_zero_actions() -> None:
    """Load the accepted controller and prove the bounded probe does not step."""

    binding_path = Path(os.environ["LANGMANI_LATENTGUARD_BINDING"])
    raw = json.loads(binding_path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    validate_envelope(raw, expected_schema=POLICY_BINDING_SCHEMA)
    registry, locators = load_runtime_registry(
        checkpoint_root=Path(os.environ["LANGMANI_CHECKPOINT_ROOT"]),
        dataset_root=Path(os.environ["LANGMANI_DATASET_ROOT"]),
        runtime_selection_path=Path(os.environ["LANGMANI_RUNTIME_SELECTION"]),
    )
    proposal = export_initial_proposal(
        binding_envelope=raw,
        registry=registry,
        locators=locators,
        repository_root=Path(__file__).resolve().parents[2],
        probe_seed=derive_probe_seed(),
        device="cuda",
    )
    payload = validate_envelope(proposal, expected_schema=INITIAL_PROPOSAL_SCHEMA)
    assert payload["reset_count"] == 1
    assert payload["policy_query_count"] == 1
    assert payload["environment_step_count"] == 0
    assert payload["outcomes_generated"] is False
