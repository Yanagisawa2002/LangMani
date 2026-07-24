from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from langmani.v2.phase2b6 import PRIMARY_SPLITS, TASK_IDS
from langmani.v2.phase2b6_v2 import (
    EXPECTED_SOURCE_EPISODES,
    PACKAGE_ID,
    REQUIRED_SUB_GATES,
    SOURCE_COMMIT,
    TARGET_BRANCH,
    EpisodeClass,
    Phase2B6V2ContractError,
    Phase2B6V2Result,
    accepted_assignments,
    authorization_state,
    classify_attempt,
    classify_result,
    enforce_starting_point,
    first_failed_gate,
    gate,
    load_exclusion_policy,
    load_production_spec,
    split_count_report,
    threshold_audit,
)
from langmani.v2.phase2b6_v2_runtime import verify_prior_evidence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY = PROJECT_ROOT / "configs" / "langmani_v2" / "phase2b6_v2_exclusion_policy.yaml"
SPEC = PROJECT_ROOT / "configs" / "langmani_v2" / "phase2b6_v2_dataset_production.yaml"
LAUNCHER = PROJECT_ROOT / "environment" / "run_v2_phase2b6_v2_production.sh"


def _passing_gates() -> dict[str, dict[str, object]]:
    return {name: gate("passed") for name in REQUIRED_SUB_GATES}


def _terminal_rows(*, exclusions: dict[str, set[int]] | None = None) -> list[dict[str, object]]:
    excluded = exclusions or {}
    rows: list[dict[str, object]] = []
    for task_id in TASK_IDS:
        for source_episode_id in range(1_000):
            is_excluded = source_episode_id in excluded.get(task_id, set())
            rows.append(
                {
                    "task_id": task_id,
                    "source_episode_id": source_episode_id,
                    "derived_episode_identity": f"{task_id}:{source_episode_id}",
                    "classification": (
                        EpisodeClass.EXCLUDED_DETERMINISTIC_PHYSICAL.value
                        if is_excluded
                        else EpisodeClass.ACCEPTED_REPLAY.value
                    ),
                    "terminal_attempt": True,
                }
            )
    return rows


def test_v2_spec_and_policy_are_frozen() -> None:
    policy = load_exclusion_policy(POLICY)
    spec = load_production_spec(SPEC, repo_root=PROJECT_ROOT)
    assert spec["derived_dataset_version"] == PACKAGE_ID
    assert spec["source_authorization"]["source_commit"] == SOURCE_COMMIT
    assert spec["source_authorization"]["target_branch"] == TARGET_BRANCH
    assert spec["_policy"]["_sha256"] == policy["_sha256"]
    assert spec["authorization"]["optimizer_steps"] == 0


def test_episode_938_is_registered_for_review_not_automatic_exclusion() -> None:
    policy = load_exclusion_policy(POLICY)
    row = cast(list[dict[str, object]], policy["registered_exclusions"])[0]
    assert row["source_episode_id"] == 938
    assert row["normal_v2_attempt_required"] is True
    assert row["automatic_exclusion"] is False
    assert row["expected_first_failed_sub_gate"] == "canonical_terminal_success_gate"


def test_source_commit_enforcement() -> None:
    enforce_starting_point(
        source_commit=SOURCE_COMMIT,
        branch=TARGET_BRANCH,
        source_is_ancestor=True,
        no_merges_since_source=True,
    )
    with pytest.raises(Phase2B6V2ContractError):
        enforce_starting_point(
            source_commit="0" * 40,
            branch=TARGET_BRANCH,
            source_is_ancestor=True,
            no_merges_since_source=True,
        )


def test_prior_evidence_manifests_are_immutable() -> None:
    report = verify_prior_evidence(PROJECT_ROOT)
    assert report["passed"] is True
    sets = cast(dict[str, dict[str, object]], report["evidence_sets"])
    assert {name: row["file_count"] for name, row in sets.items()} == {
        "phase_2b5": 27,
        "phase_2b6": 21,
        "phase_2b6_1": 24,
        "phase_2b6_1r": 19,
        "phase_2b6_1_v2": 29,
    }


def test_validated_launcher_pins_recovered_vulkan_contract() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "VK_ICD_FILENAMES=/etc/vulkan/icd.d/my_nvidia_icd.json" in text
    assert "__EGL_VENDOR_LIBRARY_FILENAMES=" in text
    assert "CUDA_VISIBLE_DEVICES=0" in text
    assert "XDG_RUNTIME_DIR=" in text
    assert "DISPLAY=" not in text
    assert "run_v2_phase2b6_v2.py" in text


def test_explicit_sub_gate_set_and_order_are_frozen() -> None:
    gates = _passing_gates()
    assert tuple(gates) == REQUIRED_SUB_GATES
    assert first_failed_gate(gates) is None
    gates["canonical_terminal_success_gate"] = gate("failed")
    gates["source_replay_outcome_agreement_gate"] = gate("failed")
    assert first_failed_gate(gates) == "canonical_terminal_success_gate"


def test_accepted_replay_requires_every_gate() -> None:
    assert (
        classify_attempt(sub_gates=_passing_gates(), full_action_execution=True)
        is EpisodeClass.ACCEPTED_REPLAY
    )


def test_physical_failure_is_excluded_without_retry() -> None:
    gates = _passing_gates()
    gates["canonical_terminal_success_gate"] = gate("failed")
    gates["source_replay_outcome_agreement_gate"] = gate("failed")
    assert (
        classify_attempt(sub_gates=gates, full_action_execution=True)
        is EpisodeClass.EXCLUDED_DETERMINISTIC_PHYSICAL
    )


def test_source_contract_failure_is_separate() -> None:
    gates = _passing_gates()
    gates["action_contract_gate"] = gate("failed")
    assert (
        classify_attempt(sub_gates=gates, full_action_execution=False)
        is EpisodeClass.EXCLUDED_SOURCE_CONTRACT
    )


def test_only_declared_infrastructure_failure_is_retryable() -> None:
    gates = _passing_gates()
    gates["environment_construction_gate"] = gate("failed")
    assert (
        classify_attempt(
            sub_gates=gates,
            full_action_execution=False,
            retryable_reason="environment_construction",
        )
        is EpisodeClass.RETRYABLE_INFRASTRUCTURE
    )
    assert (
        classify_attempt(sub_gates=gates, full_action_execution=False)
        is EpisodeClass.UNCLASSIFIED_FAILURE
    )


def test_simulator_exception_is_unclassified_and_fail_closed() -> None:
    gates = _passing_gates()
    gates["step_execution_gate"] = gate("failed")
    gates["simulator_exception_gate"] = gate("failed")
    assert (
        classify_attempt(sub_gates=gates, full_action_execution=False)
        is EpisodeClass.UNCLASSIFIED_FAILURE
    )


def test_thresholds_accept_episode_938_only_exclusion() -> None:
    report = threshold_audit(_terminal_rows(exclusions={"StackCube-v1": {938}}))
    assert report["source_count"] == EXPECTED_SOURCE_EPISODES
    assert report["accepted_count"] == 2_999
    assert report["excluded_count"] == 1
    assert report["passed"] is True


def test_per_task_exclusion_threshold_is_zero_tolerance_above_five() -> None:
    report = threshold_audit(_terminal_rows(exclusions={"StackCube-v1": set(range(6))}))
    assert report["passed"] is False
    assert report["per_task"]["StackCube-v1"]["excluded_count"] == 6


def test_total_exclusion_threshold_is_nine() -> None:
    report = threshold_audit(
        _terminal_rows(
            exclusions={
                "PickCube-v1": {0, 1, 2, 3},
                "StackCube-v1": {0, 1, 2},
                "PushCube-v1": {0, 1, 2},
            }
        )
    )
    assert report["excluded_count"] == 10
    assert report["passed"] is False


def test_unclassified_failure_blocks_acceptance() -> None:
    rows = _terminal_rows()
    rows[0]["classification"] = EpisodeClass.UNCLASSIFIED_FAILURE.value
    report = threshold_audit(rows)
    assert report["class_counts"][EpisodeClass.UNCLASSIFIED_FAILURE.value] == 1
    assert report["passed"] is False


def test_accepted_assignments_never_replace_exclusion() -> None:
    assignments = [
        {
            "task_id": "StackCube-v1",
            "source_episode_id": index,
            "derived_episode_identity": f"stack:{index}",
            "primary_split": "train",
        }
        for index in range(3)
    ]
    attempts = [
        {
            **row,
            "classification": (
                EpisodeClass.EXCLUDED_DETERMINISTIC_PHYSICAL.value
                if row["source_episode_id"] == 1
                else EpisodeClass.ACCEPTED_REPLAY.value
            ),
        }
        for row in assignments
    ]
    accepted = accepted_assignments(assignments, attempts)
    assert [row["source_episode_id"] for row in accepted] == [0, 2]


def test_split_counts_preserve_source_count_and_report_exclusion() -> None:
    assignments = [
        {
            "task_id": task_id,
            "source_episode_id": index,
            "derived_episode_identity": f"{task_id}:{index}",
            "primary_split": PRIMARY_SPLITS[index % len(PRIMARY_SPLITS)],
        }
        for task_id in TASK_IDS
        for index in range(10)
    ]
    accepted = assignments[:-1]
    report = split_count_report(assignments, accepted)
    task = cast(dict[str, dict[str, int]], report["tasks"])["PushCube-v1"]
    assert sum(row["source_count"] for row in task.values()) == 10
    assert sum(row["accepted_count"] for row in task.values()) == 9
    assert report["replacement_sampling"] is False


def test_result_classification_separates_integrity_threshold_and_restore() -> None:
    assert (
        classify_result(
            integrity_valid=True,
            threshold_valid=True,
            archive_created=True,
            restore_valid=True,
        )
        is Phase2B6V2Result.RESULT_A
    )
    assert (
        classify_result(
            integrity_valid=True,
            threshold_valid=False,
            archive_created=False,
            restore_valid=False,
        )
        is Phase2B6V2Result.RESULT_B
    )
    assert (
        classify_result(
            integrity_valid=False,
            threshold_valid=True,
            archive_created=True,
            restore_valid=True,
        )
        is Phase2B6V2Result.RESULT_C
    )
    assert (
        classify_result(
            integrity_valid=True,
            threshold_valid=True,
            archive_created=True,
            restore_valid=False,
        )
        is Phase2B6V2Result.RESULT_D
    )


def test_training_authorization_remains_false_after_dataset_acceptance() -> None:
    state = authorization_state(accepted=True)
    assert state["accepted_multiskill_dataset_validated"] is True
    assert state["act_baseline_training_eligible"] is True
    assert state["smolvla_training_eligible"] is True
    assert state["vla_jepa_training_eligible"] is True
    assert state["act_training_authorized"] is False
    assert state["smolvla_training_authorized"] is False
    assert state["vla_jepa_training_authorized"] is False
    assert state["optimizer_created"] is False
    assert state["backward_passes"] == 0
    assert state["optimizer_steps"] == 0


def test_no_policy_or_optimizer_vocabulary_in_runtime_imports() -> None:
    source = (PROJECT_ROOT / "src" / "langmani" / "v2" / "phase2b6_v2_runtime.py").read_text(
        encoding="utf-8"
    )
    assert "langmani.policies" not in source
    assert "torch.optim" not in source
    assert "backward(" not in source


def test_policy_hash_tamper_fails(tmp_path: Path) -> None:
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    policy["thresholds"]["maximum_exclusions_total"] = 10
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(Phase2B6V2ContractError):
        load_exclusion_policy(path)
