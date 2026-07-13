"""Explicit privileged runtime state exposed only to LangMani experts.

The objects in this module are deliberately absent from Gym observations and step
``info`` dictionaries.  They are simulator handles and geometry constants for a
single-environment, privileged demonstration generator; they are not policy inputs
and they are not serializable dataset records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from mani_skill.agents.robots import Panda
from mani_skill.utils.structs import Actor

from langmani.environments.specs import BinId, EpisodeSpec, ObjectId


class ExpertContextError(RuntimeError):
    """Expected failure while materializing the privileged expert boundary."""


@dataclass(frozen=True, slots=True)
class SemanticObjectHandle:
    """Associate one movable actor with its stable semantic object ID."""

    object_id: ObjectId
    actor: Actor


@dataclass(frozen=True, slots=True)
class SemanticBinHandle:
    """Associate one destination actor with its stable semantic bin ID."""

    bin_id: BinId
    actor: Actor
    floor_center: torch.Tensor


@dataclass(frozen=True, slots=True)
class ExpertTaskContext:
    """Privileged handles and geometry for one active expert rollout.

    ``robot`` is intentionally typed as ``Any`` because ManiSkill's concrete
    articulation type differs between CPU and GPU backends.  It is always the
    articulation owned by ``agent`` and is validated when the context is built.
    """

    environment_id: str
    episode_spec: EpisodeSpec
    agent: Panda
    robot: Any
    target_object: SemanticObjectHandle
    target_bin: SemanticBinHandle
    objects: tuple[SemanticObjectHandle, ...]
    bins: tuple[SemanticBinHandle, ...]
    cube_half_extent: float
    bin_interior_half_size: float
    bin_wall_height: float
    bin_bottom_thickness: float
    table_top_z: float


__all__ = [
    "ExpertContextError",
    "ExpertTaskContext",
    "SemanticBinHandle",
    "SemanticObjectHandle",
]
