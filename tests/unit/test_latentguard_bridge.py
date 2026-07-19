"""CPU-safe tests for the optional initial-state LatentGuard bridge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from langmani.integrations.latentguard_bridge import (
    CANDIDATE_REQUEST_SCHEMA,
    POLICY_BINDING_SCHEMA,
    PROJECTED_CANDIDATES_SCHEMA,
    LatentGuardBridgeError,
    audit_bridge,
    build_policy_binding,
    content_digest,
    derive_probe_seed,
    export_initial_proposal,
    make_envelope,
    project_candidates,
    read_envelope,
    validate_envelope,
    write_envelope,
)
from langmani.language.controller_registry import (
    ControllerLocation,
    ControllerRegistryLocators,
    build_fixture_controller_registry,
)


def _binding() -> dict[str, object]:
    registry = build_fixture_controller_registry()
    entry = sorted(registry.entries, key=lambda item: item.task_id)[0]
    return build_policy_binding(
        registry=registry,
        entry=entry,
        bridge_git_commit="d" * 40,
        bridge_git_branch="codex/m5b-latentguard-proposal-bridge",
    )


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    import hashlib

    return content_digest(
        {
            "bytes_sha256": f"sha256:{hashlib.sha256(array.tobytes(order='C')).hexdigest()}",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
    )


def _candidate_pool_digest(
    binding: dict[str, object], candidates: list[dict[str, object]]
) -> tuple[str, str, str]:
    source_digest = content_digest({"source": "fixture"})
    proposal_digest = "sha256:" + "e" * 64
    pool_digest = content_digest(
        {
            "candidate_source_digest": source_digest,
            "policy_binding_digest": binding["content_digest"],
            "proposal_digest": proposal_digest,
            "raw_candidate_digests": [item["raw_candidate_digest"] for item in candidates],
        }
    )
    return source_digest, proposal_digest, pool_digest


def _candidates(binding: dict[str, object]) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for index, name in enumerate(("identity", "a", "b", "c")):
        raw = np.full((10, 8), index * 0.75, dtype=np.float32)
        candidates.append(
            {
                "candidate_id": "pending",
                "raw_actions": raw.tolist(),
                "raw_candidate_digest": _array_digest(raw),
                "transformation_config_digest": content_digest({"name": name}),
                "transformation_id": name,
            }
        )
    source_digest, proposal_digest, pool_digest = _candidate_pool_digest(binding, candidates)
    for index, candidate in enumerate(candidates):
        candidate["candidate_id"] = (
            "lgc-sha256-"
            + hashlib.sha256(
                json.dumps(
                    {"candidate_pool_digest": pool_digest, "ordinal": index},
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
        )
    return make_envelope(
        CANDIDATE_REQUEST_SCHEMA,
        {
            "action_dimension": 8,
            "candidate_count": 4,
            "candidate_pool_digest": pool_digest,
            "candidate_source_digest": source_digest,
            "candidates": candidates,
            "environment_actions_executed": False,
            "outcomes_available": False,
            "policy_binding_digest": binding["content_digest"],
            "proposal_digest": proposal_digest,
            "source_horizon": 10,
        },
    )


def test_binding_selects_lexically_first_task_and_excludes_paths() -> None:
    binding = _binding()
    payload = validate_envelope(binding, expected_schema=POLICY_BINDING_SCHEMA)
    assert payload["task_id"] == ("langmani-pick-place-task-v0:blue_cube:left_bin:canonical_v0")
    assert payload["instruction"] == ("Pick up the blue cube and place it in the left bin.")
    text = json.dumps(binding, sort_keys=True)
    for forbidden in ("checkpoint_root", "dataset_root", "password", "credential", "D:\\"):
        assert forbidden not in text


def test_canonical_round_trip_and_unknown_field_rejection(tmp_path: Path) -> None:
    binding = _binding()
    path = tmp_path / "binding.json"
    write_envelope(path, binding)
    assert path.read_bytes().endswith(b"\n")
    loaded, _ = read_envelope(path, expected_schema=POLICY_BINDING_SCHEMA)
    assert loaded == binding
    tampered = dict(binding)
    tampered["unknown"] = True
    with pytest.raises(LatentGuardBridgeError, match="unexpected or missing"):
        validate_envelope(tampered, expected_schema=POLICY_BINDING_SCHEMA)


def test_projection_reuses_bound_processor_and_separates_identities() -> None:
    binding = _binding()
    candidates = _candidates(binding)
    projected = project_candidates(binding_envelope=binding, candidate_envelope=candidates)
    payload = validate_envelope(projected, expected_schema=PROJECTED_CANDIDATES_SCHEMA)
    assert payload["candidate_count"] == 4
    assert payload["environment_step_count"] == 0
    assert payload["outcomes_available"] is False
    rows = payload["candidates"]
    assert isinstance(rows, list)
    assert all(row["raw_candidate_digest"] != row["projected_candidate_digest"] for row in rows[2:])
    assert all(len(row["correction_mask"]) == 10 for row in rows)
    audit = audit_bridge(
        binding_envelope=binding,
        candidate_envelope=candidates,
        projected_envelope=projected,
    )
    audit_payload = validate_envelope(audit, expected_schema="LangManiLatentGuardBridgeAuditV1")
    assert audit_payload["passed"] is True
    assert audit_payload["physical_action_execution"] is False


def test_candidate_request_rejects_nonfinite_and_outcomes() -> None:
    binding = _binding()
    candidates = _candidates(binding)
    payload = dict(validate_envelope(candidates, expected_schema=CANDIDATE_REQUEST_SCHEMA))
    rows = [dict(item) for item in payload["candidates"]]  # type: ignore[union-attr]
    rows[0]["raw_actions"] = [[float("nan")] * 8 for _ in range(10)]
    payload["candidates"] = rows
    with pytest.raises(LatentGuardBridgeError, match="canonical JSON failed"):
        make_envelope(CANDIDATE_REQUEST_SCHEMA, payload)
    payload = dict(validate_envelope(candidates, expected_schema=CANDIDATE_REQUEST_SCHEMA))
    payload["outcomes_available"] = True
    outcome_bound = make_envelope(CANDIDATE_REQUEST_SCHEMA, payload)
    with pytest.raises(LatentGuardBridgeError, match="outcomes are prohibited"):
        project_candidates(binding_envelope=binding, candidate_envelope=outcome_bound)


def test_probe_seed_is_fixed_and_positive() -> None:
    assert derive_probe_seed() == derive_probe_seed()
    assert 0 <= derive_probe_seed() < 2**31


def test_fixture_proposal_resets_once_queries_once_and_never_steps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = build_fixture_controller_registry()
    binding = _binding()
    locations = []
    for entry in registry.entries:
        run_root = tmp_path / entry.task_id.replace(":", "_")
        run_root.mkdir()
        locations.append(
            ControllerLocation(
                task_id=entry.task_id,
                run_root=run_root,
                checkpoint_relative_path="checkpoints/validation_best",
            )
        )
    locators = ControllerRegistryLocators(
        registry_fingerprint=registry.registry_fingerprint,
        locations=tuple(locations),
    )

    class FakeEnvironment:
        def __init__(self) -> None:
            import gymnasium as gym

            self.action_space = gym.spaces.Box(
                low=np.full(8, -1.0, dtype=np.float32),
                high=np.full(8, 1.0, dtype=np.float32),
                dtype=np.float32,
            )
            self.unwrapped = self

        def reset(self, **_: object) -> tuple[dict[str, object], dict[str, object]]:
            return {}, {}

        def step(self, *_: object, **__: object) -> object:
            raise AssertionError("fixture environment must never step")

        def close(self) -> None:
            return None

    class FakePolicy:
        config = SimpleNamespace(device="cpu")

        def reset(self) -> None:
            return None

        def begin_task(self, _task_id: str) -> None:
            return None

        def to(self, _device: object) -> FakePolicy:
            return self

        def eval(self) -> FakePolicy:
            return self

        def predict_action_chunk(self, _batch: dict[str, object]) -> object:
            import torch

            return torch.full((1, 50, 8), 1.5, dtype=torch.float32)

    class FakeProcessor:
        steps: list[object] = []

        def reset(self) -> None:
            return None

        def __call__(self, value: object) -> object:
            return value

    context = SimpleNamespace(
        loaded=SimpleNamespace(
            policy=FakePolicy(),
            preprocessor=FakeProcessor(),
            postprocessor=FakeProcessor(),
        )
    )
    monkeypatch.setattr(
        "langmani.integrations.latentguard_bridge._git_identity",
        lambda *_args, **_kwargs: ("d" * 40, "codex/m5b-latentguard-proposal-bridge"),
    )
    monkeypatch.setattr(
        "langmani.integrations.latentguard_bridge.validate_active_controller_environment",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "langmani.integrations.latentguard_bridge.build_policy_observation",
        lambda *_args, **_kwargs: {
            "observation.images.base_camera": __import__("torch").zeros((1, 3, 8, 8)),
            "observation.state": __import__("torch").zeros((1, 8)),
        },
    )
    proposal = export_initial_proposal(
        binding_envelope=binding,
        registry=registry,
        locators=locators,
        repository_root=tmp_path,
        probe_seed=derive_probe_seed(),
        device="cpu",
        environment_factory=FakeEnvironment,
        context_loader=lambda *_args, **_kwargs: context,
    )
    payload = validate_envelope(proposal, expected_schema="LangManiLatentGuardInitialProposalV1")
    assert payload["reset_count"] == 1
    assert payload["policy_query_count"] == 1
    assert payload["environment_step_count"] == 0
    assert payload["zero_action_execution"] is True
