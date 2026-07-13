"""Import smoke tests for the shared robotics-learning environment."""

from importlib import import_module

import pytest


@pytest.mark.parametrize(
    "module_name",
    ["torch", "mani_skill", "lerobot", "lerobot.datasets", "av", "pyarrow"],
)
def test_runtime_dependency_imports(module_name: str) -> None:
    """Each foundational runtime dependency must be importable."""
    module = import_module(module_name)

    assert module is not None


def test_langmani_imports() -> None:
    """The project package must be discoverable through the src layout."""
    module = import_module("langmani")

    assert module.__name__ == "langmani"
