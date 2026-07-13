"""Runtime selection for the Linux-only M2/M3A motion-planning path."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import cast

PLANNER_PYTHON_ENV = "LANGMANI_PLANNER_PYTHON"
PLANNER_RUNTIME_DISTRIBUTIONS: Mapping[str, str] = MappingProxyType(
    {
        "gymnasium": "gymnasium",
        "h5py": "h5py",
        "mani_skill": "mani-skill",
        "mplib": "mplib",
        "numpy": "numpy",
        "opencv_python": "opencv-python",
        "pillow": "Pillow",
        "sapien": "sapien",
        "scipy": "scipy",
        "torch": "torch",
    }
)
EXPECTED_PLANNER_RUNTIME_VERSIONS: Mapping[str, str] = MappingProxyType(
    {
        "gymnasium": "1.2.3",
        "h5py": "3.16.0",
        "mani_skill": "3.0.1",
        "mplib": "0.1.1",
        "numpy": "1.26.4",
        "opencv_python": "4.11.0.86",
        "pillow": "12.3.0",
        "sapien": "3.0.3",
        "scipy": "1.15.3",
        "torch": "2.11.0",
    }
)


class PlannerRuntimeError(RuntimeError):
    """Raised when the configured planner interpreter cannot be used safely."""


def resolve_planner_python(value: str | os.PathLike[str] | None = None) -> str:
    """Resolve the explicit side-runtime interpreter, falling back to this Python."""

    configured = value if value is not None else os.environ.get(PLANNER_PYTHON_ENV)
    candidate = (
        Path(configured if configured is not None else sys.executable).expanduser().resolve()
    )
    if not candidate.is_file():
        raise PlannerRuntimeError(
            f"planner Python does not exist: {candidate}; set {PLANNER_PYTHON_ENV}"
        )
    return str(candidate)


def query_planner_runtime_versions(
    python_executable: str | os.PathLike[str] | None = None,
) -> dict[str, str | None]:
    """Read exact distribution versions from the selected interpreter."""

    interpreter = resolve_planner_python(python_executable)
    distributions_json = json.dumps(dict(PLANNER_RUNTIME_DISTRIBUTIONS), sort_keys=True)
    probe = (
        "import json\n"
        "from importlib import metadata\n"
        f"distributions = json.loads({distributions_json!r})\n"
        "versions = {}\n"
        "for key, distribution in distributions.items():\n"
        "    try:\n"
        "        versions[key] = metadata.version(distribution)\n"
        "    except metadata.PackageNotFoundError:\n"
        "        versions[key] = None\n"
        "print(json.dumps(versions, sort_keys=True))\n"
    )
    try:
        completed = subprocess.run(
            [interpreter, "-c", probe],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PlannerRuntimeError(f"planner runtime version probe failed: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic output"
        raise PlannerRuntimeError(
            f"planner runtime version probe exited {completed.returncode}: {detail}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise PlannerRuntimeError(
            "planner runtime version probe returned malformed JSON"
        ) from error
    if not isinstance(payload, dict) or set(payload) != set(PLANNER_RUNTIME_DISTRIBUTIONS):
        raise PlannerRuntimeError("planner runtime version probe returned an unexpected schema")
    result: dict[str, str | None] = {}
    for key, value in payload.items():
        if value is not None and not isinstance(value, str):
            raise PlannerRuntimeError(f"planner runtime version {key!r} is malformed")
        result[str(key)] = cast(str | None, value)
    return dict(sorted(result.items()))


def planner_runtime_matches_expected(versions: Mapping[str, str | None]) -> bool:
    """Return whether all planner-side pins match, allowing Torch's CUDA local tag."""

    for key, expected in EXPECTED_PLANNER_RUNTIME_VERSIONS.items():
        actual = versions.get(key)
        if key == "torch":
            if actual is None or actual.split("+", maxsplit=1)[0] != expected:
                return False
        elif actual != expected:
            return False
    return True


__all__ = [
    "EXPECTED_PLANNER_RUNTIME_VERSIONS",
    "PLANNER_PYTHON_ENV",
    "PLANNER_RUNTIME_DISTRIBUTIONS",
    "PlannerRuntimeError",
    "planner_runtime_matches_expected",
    "query_planner_runtime_versions",
    "resolve_planner_python",
]
