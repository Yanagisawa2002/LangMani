"""Phase 2C-A.1 compact checkpoint-registry contracts."""

from __future__ import annotations

import pytest

from scripts.compact_v2_phase2c_a1_evidence import _screen_component_fingerprints


def _screen(index: int) -> dict[str, object]:
    return {
        "checkpoint_fingerprint": f"sha256:{index:064x}",
        "checkpoint_components": {
            "model": "sha256:" + "a" * 64,
            "preprocessor": "sha256:" + "b" * 64,
            "postprocessor": "sha256:" + "c" * 64,
        },
    }


def test_compact_registry_uses_screened_checkpoint_components() -> None:
    first = _screen(1)
    second = _screen(2)
    result = _screen_component_fingerprints([first, second])
    assert set(result) == {
        first["checkpoint_fingerprint"],
        second["checkpoint_fingerprint"],
    }
    assert result[str(first["checkpoint_fingerprint"])]["model"] == "sha256:" + "a" * 64


def test_compact_registry_rejects_duplicate_or_incomplete_components() -> None:
    first = _screen(1)
    with pytest.raises(RuntimeError, match="component registry"):
        _screen_component_fingerprints([first, dict(first)])
    incomplete = _screen(2)
    components = incomplete["checkpoint_components"]
    assert isinstance(components, dict)
    components.pop("postprocessor")
    with pytest.raises(RuntimeError, match="component registry"):
        _screen_component_fingerprints([incomplete])
