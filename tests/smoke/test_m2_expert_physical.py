"""Native-Linux physical acceptance for all six M2 expert tasks."""

from __future__ import annotations

import os
import platform

import pytest

from langmani.environments.specs import BIN_IDS, OBJECT_IDS, TaskSpec
from langmani.experts import PickPlaceExpert
from langmani.experts.command_support import (
    create_expert_environment,
    task_reset_options,
)


def _is_native_linux() -> bool:
    release = platform.release().lower()
    is_wsl = "microsoft" in release or bool(os.environ.get("WSL_INTEROP"))
    return platform.system() == "Linux" and not is_wsl


@pytest.mark.integration
@pytest.mark.skipif(
    not _is_native_linux(),
    reason="M2 mplib rollouts require the native Linux acceptance platform",
)
def test_expert_physically_solves_all_six_semantic_tasks() -> None:
    env = create_expert_environment(
        diagnostic_rendering=False,
        sim_backend="physx_cpu",
    )
    results = []
    try:
        for object_id in OBJECT_IDS:
            for bin_id in BIN_IDS:
                task_spec = TaskSpec(
                    target_object_id=object_id,
                    target_bin_id=bin_id,
                    instruction_template_id="canonical_v0",
                )
                env.reset(seed=0, options=task_reset_options(task_spec))
                result = PickPlaceExpert(env).run()
                results.append(result)
                assert result.success, result.to_dict()
                assert result.target_object_id == object_id
                assert result.target_bin_id == bin_id
                assert result.final_environment_evaluation["success"] is True
    finally:
        env.close()

    assert len(results) == 6
    assert len({result.scene_id for result in results}) == 1
    assert len({result.task_id for result in results}) == 6
