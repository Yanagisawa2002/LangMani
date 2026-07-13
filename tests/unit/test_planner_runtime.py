from __future__ import annotations

import sys
from pathlib import Path

import pytest

from langmani.experts.runtime import (
    EXPECTED_PLANNER_RUNTIME_VERSIONS,
    PLANNER_PYTHON_ENV,
    PLANNER_RUNTIME_MODULES,
    PlannerRuntimeError,
    planner_runtime_matches_expected,
    query_planner_runtime_versions,
    resolve_planner_python,
)


def test_planner_python_defaults_to_current_interpreter_and_honors_explicit_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(PLANNER_PYTHON_ENV, raising=False)
    assert resolve_planner_python() == str(Path(sys.executable).resolve())

    side_python = tmp_path / "side-python"
    side_python.write_text("placeholder", encoding="utf-8")
    monkeypatch.setenv(PLANNER_PYTHON_ENV, str(side_python))
    assert resolve_planner_python() == str(side_python.resolve())

    monkeypatch.setenv(PLANNER_PYTHON_ENV, str(tmp_path / "missing"))
    with pytest.raises(PlannerRuntimeError, match="does not exist"):
        resolve_planner_python()


def test_planner_python_preserves_virtual_environment_launcher_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    side_python = tmp_path / "side-python"
    side_python.write_text("placeholder", encoding="utf-8")
    expected = str(side_python.absolute())
    monkeypatch.setattr(
        Path,
        "resolve",
        lambda self: (_ for _ in ()).throw(AssertionError("must not dereference venv symlink")),
    )

    assert resolve_planner_python(side_python) == expected


def test_runtime_probe_returns_the_complete_stable_version_mapping() -> None:
    versions = query_planner_runtime_versions(sys.executable)

    assert set(versions) == set(PLANNER_RUNTIME_MODULES)
    assert all(value is None or isinstance(value, str) for value in versions.values())


def test_planner_runtime_match_requires_every_pin_and_allows_torch_local_tag() -> None:
    versions = dict(EXPECTED_PLANNER_RUNTIME_VERSIONS)
    versions["torch"] = versions["torch"] + "+cu128"
    assert planner_runtime_matches_expected(versions)

    versions["numpy"] = "2.2.6"
    assert not planner_runtime_matches_expected(versions)
