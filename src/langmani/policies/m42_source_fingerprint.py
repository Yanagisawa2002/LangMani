"""Canonical source closure for content-bound M4.2 TaskToken training reuse.

Both a new training run and a verifier auditing an older Git commit must hash
this exact repository-relative file set.  Keeping the closure here avoids the
dangerous case where the producer and reuse gate silently protect different
training semantics.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from langmani.datasets.identity import sha256_hex

TASK_TOKEN_TRAINING_SOURCE_SCHEMA = "langmani-m42-task-token-training-source-v1"
TASK_TOKEN_TRAINING_SOURCE_FILES = (
    "scripts/train_act_task_token.py",
    "src/langmani/datasets/identity.py",
    "src/langmani/datasets/lerobot_types.py",
    "src/langmani/datasets/lerobot_writer.py",
    "src/langmani/datasets/policy_state.py",
    "src/langmani/datasets/schedule.py",
    "src/langmani/datasets/splits.py",
    "src/langmani/environments/specs.py",
    "src/langmani/policies/act_action_bounds.py",
    "src/langmani/policies/act_checkpoint.py",
    "src/langmani/policies/act_conditioning.py",
    "src/langmani/policies/act_data.py",
    "src/langmani/policies/act_runtime.py",
    "src/langmani/policies/act_task_token.py",
    "src/langmani/policies/act_training.py",
    "src/langmani/policies/act_types.py",
    "src/langmani/policies/m42_schedule.py",
    "src/langmani/policies/m42_source_fingerprint.py",
    "src/langmani/policies/m42_training.py",
    "src/langmani/policies/m42_types.py",
)


def read_task_token_training_sources(project_root: Path) -> dict[str, bytes]:
    """Read the canonical source closure from one working tree without aliases."""

    root = project_root.resolve(strict=True)
    sources: dict[str, bytes] = {}
    for relative_name in TASK_TOKEN_TRAINING_SOURCE_FILES:
        relative = PurePosixPath(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe TaskToken training source path: {relative_name}")
        source = root.joinpath(*relative.parts)
        if not source.is_file():
            raise RuntimeError(f"missing TaskToken training source: {relative_name}")
        sources[relative_name] = source.read_bytes()
    return sources


def task_token_training_source_fingerprint(sources: Mapping[str, bytes]) -> str:
    """Hash an exact canonical source closure using stable relative path keys."""

    expected = set(TASK_TOKEN_TRAINING_SOURCE_FILES)
    provided = set(sources)
    if provided != expected:
        missing = sorted(expected - provided)
        extra = sorted(provided - expected)
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise RuntimeError("invalid TaskToken training source closure: " + "; ".join(details))
    files: dict[str, str] = {}
    for relative_name in TASK_TOKEN_TRAINING_SOURCE_FILES:
        content = sources[relative_name]
        if not isinstance(content, bytes):
            raise TypeError(f"TaskToken source content must be bytes: {relative_name}")
        files[relative_name] = f"sha256:{hashlib.sha256(content).hexdigest()}"
    payload = {"schema_version": TASK_TOKEN_TRAINING_SOURCE_SCHEMA, "files": files}
    return f"sha256:{sha256_hex(payload)}"


def current_task_token_training_source_fingerprint(project_root: Path) -> str:
    """Fingerprint the canonical source closure in the current working tree."""

    return task_token_training_source_fingerprint(read_task_token_training_sources(project_root))
