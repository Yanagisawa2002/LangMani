"""Git, runtime-version, and immutable output helpers for M4 commands."""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import torch

from langmani.policies.act_types import ExperimentMode


class RuntimeContractError(RuntimeError):
    """Raised when provenance or output immutability cannot be guaranteed."""


@dataclass(frozen=True, slots=True)
class GitState:
    commit: str
    dirty: bool
    changed_paths: tuple[str, ...]
    baseline_tracked: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "commit": self.commit,
            "dirty": self.dirty,
            "changed_paths": list(self.changed_paths),
            "baseline_tracked": self.baseline_tracked,
        }


def _git(project_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeContractError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout.strip()


def inspect_git_state(project_root: Path) -> GitState:
    """Return the exact commit and porcelain state used by a run."""
    root = project_root.resolve()
    commit = _git(root, "rev-parse", "HEAD")
    if len(commit) < 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeContractError("Git HEAD is not a full lowercase object ID")
    tracked = tuple(line for line in _git(root, "ls-files").splitlines() if line)
    porcelain = tuple(line for line in _git(root, "status", "--porcelain=v1").splitlines() if line)
    return GitState(
        commit=commit,
        dirty=bool(porcelain),
        changed_paths=porcelain,
        baseline_tracked=bool(tracked),
    )


def validate_git_for_run(
    state: GitState,
    *,
    mode: ExperimentMode,
    allow_dirty_development: bool,
) -> None:
    if not state.baseline_tracked:
        raise RuntimeContractError("M4 requires a tracked Git baseline")
    if mode is ExperimentMode.FULL and state.dirty:
        raise RuntimeContractError("full M4 experiments require a clean Git working tree")
    if state.dirty and mode is ExperimentMode.DEVELOPMENT and not allow_dirty_development:
        raise RuntimeContractError(
            "dirty working tree requires --development and --allow-dirty-development"
        )
    if state.dirty and mode is ExperimentMode.TINY_OVERFIT:
        raise RuntimeContractError(
            "tiny-overfit evidence requires a clean tree or an explicitly labeled development run"
        )


def runtime_versions() -> dict[str, str | None]:
    return {
        "lerobot": version("lerobot"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": ".".join(map(str, __import__("sys").version_info[:3])),
    }


def safe_run_directory(output_root: Path, run_fingerprint: str) -> Path:
    """Map a fingerprint to one contained path, independent of user path text."""
    if not run_fingerprint.startswith("sha256:") or len(run_fingerprint) != 71:
        raise ValueError("run_fingerprint must be a prefixed SHA-256 digest")
    root = output_root.resolve()
    destination = (root / run_fingerprint.removeprefix("sha256:")).resolve()
    if destination.parent != root:
        raise RuntimeContractError("run directory escaped output root")
    return destination


def atomic_write_json(path: Path, payload: object, *, immutable: bool = False) -> None:
    """Write one JSON file through a same-directory atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with staging.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if immutable and path.exists():
            raise FileExistsError(f"immutable artifact already exists: {path}")
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


__all__ = [
    "GitState",
    "RuntimeContractError",
    "atomic_write_json",
    "inspect_git_state",
    "runtime_versions",
    "safe_run_directory",
    "validate_git_for_run",
]
