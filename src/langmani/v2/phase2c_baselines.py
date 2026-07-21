"""Trivial non-learning policy used only to sanity-check Phase 2C benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from langmani.datasets.lerobot_types import STATE_FEATURE_KEY
from langmani.v2.policy import ActionChunk, ObservationBatch, PolicyContext, PolicyIdentity
from langmani.v2.taxonomy import PUSH_TO_REGION_SKILL_ID, canonical_push_to_region_tasks


class HoldPositionPolicyAdapter:
    """Emit the current Panda joint state as a one-step hold-position action."""

    def __init__(self) -> None:
        self._context: PolicyContext | None = None
        self._queries = 0
        self._identity = PolicyIdentity(
            policy_id="hold-position-sanity-v0",
            adapter_name="hold_position_sanity_v0",
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            checkpoint_identity="none:non-learning",
            compatible_skill_families=(PUSH_TO_REGION_SKILL_ID,),
            compatible_task_ids=tuple(
                item.canonical_task_id for item in canonical_push_to_region_tasks()
            ),
        )

    @property
    def identity(self) -> PolicyIdentity:
        return self._identity

    @property
    def runtime_manifest(self) -> Mapping[str, Any]:
        return {
            "schema_version": "langmani-v2-phase2c-hold-baseline-v0",
            "policy_query_count": self._queries,
            "learned": False,
            "privileged_inputs": False,
            "action_semantic": "hold current 7 arm joints and mean gripper finger position",
        }

    def reset(self, context: PolicyContext) -> None:
        self._context = context
        self._queries = 0

    def act(self, observation: ObservationBatch) -> ActionChunk:
        if self._context is None:
            raise RuntimeError("hold-position policy requires reset before act")
        state = observation.features.get(STATE_FEATURE_KEY)
        if state is None or state.shape != (9,) or state.dtype is not torch.float32:
            raise RuntimeError("hold-position policy requires float32 state[9]")
        action = torch.cat((state[:7], state[7:9].mean().reshape(1))).reshape(1, 8)
        self._queries += 1
        return ActionChunk(
            actions=action,
            valid_mask=torch.ones(1, dtype=torch.bool),
            horizon=1,
            metadata={"policy_query_ordinal": self._queries - 1},
        )

    def close(self) -> None:
        self._context = None


__all__ = ["HoldPositionPolicyAdapter"]
