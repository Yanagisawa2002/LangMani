"""Unit tests for the intentionally lightweight LangMani root package."""

import langmani


def test_package_version() -> None:
    """The public package version remains stable through M1."""
    assert langmani.__version__ == "0.1.0"


def test_public_api_is_minimal() -> None:
    """Environment registration stays opt-in through langmani.environments."""
    assert langmani.__all__ == ["__version__"]
