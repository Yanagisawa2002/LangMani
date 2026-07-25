"""Phase 2C-A.1 final-evaluation identity-lock contracts."""

from __future__ import annotations

import pytest

from langmani.v2.phase2c_a import TASK_IDS
from scripts.freeze_v2_phase2c_a1_final_policies import _final_identity_prefixes


def _schedules(count: int = 50) -> dict[str, object]:
    return {
        split: {
            task_id: [
                {
                    "evaluation_id": f"{split}:{task_id}:{index:03d}",
                    "reset_identity": f"sha256:{index:064x}",
                }
                for index in range(count)
            ]
            for task_id in TASK_IDS
        }
        for split in ("test_unseen_reset", "test_visual_shift")
    }


def test_final_lock_freezes_exactly_first_thirty_identities() -> None:
    counts, prefixes = _final_identity_prefixes(_schedules())
    for split in ("test_unseen_reset", "test_visual_shift"):
        for task_id in TASK_IDS:
            assert counts[split][task_id] == 30
            assert len(prefixes[split][task_id]) == 30
            assert prefixes[split][task_id][0]["evaluation_id"].endswith(":000")
            assert prefixes[split][task_id][-1]["evaluation_id"].endswith(":029")


def test_final_lock_rejects_schedule_with_fewer_than_thirty_identities() -> None:
    with pytest.raises(RuntimeError, match="final schedule count changed"):
        _final_identity_prefixes(_schedules(count=29))
