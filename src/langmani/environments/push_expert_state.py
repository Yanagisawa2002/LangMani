"""Privileged, expert-only handles for the planar-pushing environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from mani_skill.utils.structs import Actor

from langmani.environments.push_specs import PushEpisodeSpec


class PushExpertContextError(RuntimeError):
    """Raised when the environment cannot provide a complete pushing context."""


@dataclass(frozen=True, slots=True)
class PushObjectHandle:
    """Stable semantic object identity plus its live simulator actor."""

    object_id: str
    actor: Actor


@dataclass(frozen=True, slots=True)
class PushExpertTaskContext:
    """All privileged handles required by one single-environment push expert."""

    environment_id: str
    episode_spec: PushEpisodeSpec
    agent: Any
    robot: Any
    target_object: PushObjectHandle
    objects: tuple[PushObjectHandle, ...]
    target_region_center: torch.Tensor
    target_region_radius: float
    target_object_planar_radius: float
    target_object_resting_height: float
    full_containment_center_radius: float
    table_top_z: float


__all__ = [
    "PushExpertContextError",
    "PushExpertTaskContext",
    "PushObjectHandle",
]
