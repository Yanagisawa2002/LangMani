"""Canonical serialization and persistent identifiers for M3A archives.

Persistent identities never depend on Python's randomized ``hash()`` or on
filesystem enumeration.  Every identifier is a namespaced SHA-256 digest of a
canonical JSON payload.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum

from langmani.environments.specs import TaskSpec
from langmani.experts.types import ExpertConfig

COLLECTION_SCHEMA_VERSION = "langmani-m3a-raw-v1"
_IDENTIFIER_NAMESPACE = "langmani-m3a"
_KIND_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")


def _json_value(value: object) -> object:
    """Convert supported project values to canonical JSON-compatible values."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, Enum):
        return _json_value(value.value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_value(to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON mapping keys must be strings")
            normalized[key] = _json_value(item)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json(value: object) -> str:
    """Serialize a value with stable key order, separators, and Unicode handling."""
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_hex(value: object) -> str:
    """Return the lowercase SHA-256 digest of a canonical JSON value."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_identifier(kind: str, payload: object) -> str:
    """Build a full-length, namespaced persistent identifier."""
    if not isinstance(kind, str) or _KIND_PATTERN.fullmatch(kind) is None:
        raise ValueError("identifier kind must match ^[a-z][a-z0-9-]*$")
    digest = sha256_hex({"kind": kind, "namespace": _IDENTIFIER_NAMESPACE, "payload": payload})
    return f"{_IDENTIFIER_NAMESPACE}-{kind}-{digest}"


def environment_version(environment_id: str) -> str:
    """Extract the explicit Gym version suffix from an environment ID."""
    if not isinstance(environment_id, str) or not environment_id:
        raise TypeError("environment_id must be a non-empty string")
    name, separator, version = environment_id.rpartition("-")
    if not separator or not name or not re.fullmatch(r"v[0-9]+", version):
        raise ValueError("environment_id must end in an explicit version such as '-v0'")
    return version


def expert_config_fingerprint(config: ExpertConfig | Mapping[str, object]) -> str:
    """Fingerprint the complete deterministic M2 expert configuration."""
    payload = config.to_dict() if isinstance(config, ExpertConfig) else dict(config)
    return f"sha256:{sha256_hex(payload)}"


def stable_collection_run_id(identity_payload: Mapping[str, object]) -> str:
    """Identify a collection run from content-affecting configuration."""
    return stable_identifier("collection-run", identity_payload)


def stable_candidate_scene_id(
    *,
    environment_id: str,
    scene_seed: int,
    scene_id: str,
    expert_fingerprint: str,
    control_mode: str,
    schema_version: str = COLLECTION_SCHEMA_VERSION,
) -> str:
    """Identify a candidate physical scene independently of its six tasks."""
    return stable_identifier(
        "candidate-scene",
        {
            "control_mode": control_mode,
            "environment_id": environment_id,
            "environment_version": environment_version(environment_id),
            "expert_config_fingerprint": expert_fingerprint,
            "scene_id": scene_id,
            "scene_seed": scene_seed,
            "schema_version": schema_version,
        },
    )


def stable_scheduled_episode_id(
    *,
    environment_id: str,
    scene_seed: int,
    scene_id: str,
    task_spec: TaskSpec,
    task_id: str,
    expert_fingerprint: str,
    control_mode: str,
    schema_version: str = COLLECTION_SCHEMA_VERSION,
) -> str:
    """Identify one semantic scene/task demonstration request."""
    return stable_identifier(
        "scheduled-episode",
        {
            "control_mode": control_mode,
            "environment_id": environment_id,
            "environment_version": environment_version(environment_id),
            "expert_config_fingerprint": expert_fingerprint,
            "scene_id": scene_id,
            "scene_seed": scene_seed,
            "schema_version": schema_version,
            "task_id": task_id,
            "task_spec": task_spec.to_dict(),
        },
    )


def stable_attempt_id(*, scheduled_episode_id: str, attempt_index: int) -> str:
    """Identify a bounded expert attempt; indices are one-based."""
    if isinstance(attempt_index, bool) or not isinstance(attempt_index, int):
        raise TypeError("attempt_index must be an integer")
    if attempt_index < 1:
        raise ValueError("attempt_index must be positive")
    return stable_identifier(
        "expert-attempt",
        {"attempt_index": attempt_index, "scheduled_episode_id": scheduled_episode_id},
    )


def stable_raw_trajectory_id(*, scheduled_episode_id: str, attempt_id: str) -> str:
    """Identify a native raw trajectory independently of shard placement."""
    return stable_identifier(
        "raw-trajectory",
        {"attempt_id": attempt_id, "scheduled_episode_id": scheduled_episode_id},
    )


def stable_source_shard_id(*, collection_run_id: str, shard_index: int) -> str:
    """Identify a deterministic accepted-source shard; indices are zero-based."""
    if isinstance(shard_index, bool) or not isinstance(shard_index, int):
        raise TypeError("shard_index must be an integer")
    if shard_index < 0:
        raise ValueError("shard_index must be non-negative")
    return stable_identifier(
        "source-shard",
        {"collection_run_id": collection_run_id, "shard_index": shard_index},
    )


def stable_accepted_scene_group_id(
    *, candidate_scene_id: str, raw_trajectory_ids: Sequence[str]
) -> str:
    """Identify an accepted all-six counterfactual group."""
    if len(raw_trajectory_ids) != 6:
        raise ValueError("an accepted scene group must contain exactly six raw trajectories")
    if not all(isinstance(value, str) and value for value in raw_trajectory_ids):
        raise TypeError("raw_trajectory_ids must contain non-empty strings")
    return stable_identifier(
        "accepted-scene-group",
        {
            "candidate_scene_id": candidate_scene_id,
            "raw_trajectory_ids": list(raw_trajectory_ids),
        },
    )
