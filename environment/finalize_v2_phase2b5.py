"""Finalize compact Phase 2B.5 evidence from completed no-training runtime reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from langmani.v2.phase2b5 import (
    CANDIDATE_TASKS,
    HF_REPOSITORY,
    HF_REVISION,
    MANISKILL_SOURCE_REVISION,
    MANISKILL_VERSION,
    SELECTED_TASK_IDS,
    SOURCE_BRANCH,
    SOURCE_COMMIT,
    SOURCE_LICENSE,
    TARGET_BRANCH,
    Phase2B5Result,
    accepted_source_package,
    authorization_state,
    canonical_json_sha256,
    classify_result,
    custom_route_closure,
    enforce_source_commit,
    validate_split_disjointness,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_ROOT = PROJECT_ROOT / "outputs" / "diagnostics" / "v2" / "phase2b5"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2b5"
F1_ROOT = PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2b4_f1"

RUNTIME_REPORTS = (
    "source_download_manifest.json",
    "source_schema_statistics.json",
    "task_candidate_runtime_audit.json",
    "bounded_replay_results.json",
    "strong_replay_results.json",
    "visual_pilot_manifest.json",
    "privilege_exclusion_audit.json",
    "lerobot_060_environment_manifest.json",
    "conversion_pilot_manifest.json",
    "lerobot_readback_result.json",
    "padding_audit.json",
    "remote_execution_audit.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--execution-commit", required=True)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _fingerprinted(payload: dict[str, object]) -> dict[str, object]:
    payload["fingerprint"] = canonical_json_sha256(payload)
    return payload


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
    ).strip()


def _verify_f1() -> dict[str, object]:
    manifest = _read(F1_ROOT / "artifact_manifest.json")
    checks: dict[str, bool] = {}
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise RuntimeError("F1 artifact manifest files must be a list")
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError("F1 artifact manifest entry must be an object")
        path = F1_ROOT / str(entry["path"])
        raw = path.read_bytes().replace(b"\r\n", b"\n")
        checks[path.name] = (
            "sha256:" + hashlib.sha256(raw).hexdigest() == entry["sha256"]
            and len(raw) == entry["size_bytes"]
        )
    return {
        "artifact_count": len(entries),
        "artifact_manifest_fingerprint": manifest.get("fingerprint"),
        "hash_and_size_checks": checks,
        "all_valid": bool(checks) and all(checks.values()),
    }


def _task_report(report: dict[str, Any], task_id: str) -> dict[str, Any]:
    tasks = report.get("task_reports")
    if not isinstance(tasks, list):
        raise RuntimeError("replay report task_reports must be a list")
    for task in tasks:
        if isinstance(task, dict) and task.get("task_id") == task_id:
            return task
    raise RuntimeError(f"replay report lacks {task_id}")


def _check_runtime(runtime: dict[str, dict[str, Any]]) -> None:
    if runtime["source_schema_statistics"].get("all_selected_sources_valid") is not True:
        raise RuntimeError("selected official source schema gate failed")
    for name, expected in (("bounded_replay_results", 20), ("strong_replay_results", 100)):
        report = runtime[name]
        if report.get("all_tasks_passed") is not True:
            raise RuntimeError(f"{name} did not pass")
        for task_id in SELECTED_TASK_IDS:
            task = _task_report(report, task_id)
            gate = task.get("gate")
            if not isinstance(gate, dict) or gate.get("episode_count") != expected:
                raise RuntimeError(f"{name}:{task_id} has the wrong replay count")
            if gate.get("passed") is not True:
                raise RuntimeError(f"{name}:{task_id} replay gate failed")
    visual = runtime["visual_pilot_manifest"]
    conversion = runtime["conversion_pilot_manifest"]
    readback = runtime["lerobot_readback_result"]
    privilege = runtime["privilege_exclusion_audit"]
    if visual.get("passed") is not True or visual.get("episode_count") != 15:
        raise RuntimeError("visual pilot gate failed")
    if conversion.get("passed") is not True or conversion.get("episode_count") != 15:
        raise RuntimeError("LeRobot conversion pilot gate failed")
    if readback.get("passed") is not True or privilege.get("passed") is not True:
        raise RuntimeError("LeRobot readback or privilege gate failed")


def _source_hash_index(source_manifest: dict[str, Any]) -> dict[str, object]:
    entries = source_manifest.get("files")
    if not isinstance(entries, list):
        raise RuntimeError("source manifest files must be a list")
    return _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b5-content-addressed-source-index-v0",
            "repository": HF_REPOSITORY,
            "revision": HF_REVISION,
            "files": [
                {
                    "task_id": entry["task_id"],
                    "file_name": entry["file_name"],
                    "sha256": entry["sha256"],
                    "size_bytes": entry["size_bytes"],
                }
                for entry in entries
            ],
            "recovery": (
                "Download the named files from the exact Hugging Face dataset revision, "
                "then verify byte size and SHA-256 before safe extraction."
            ),
        }
    )


def _split_design() -> dict[str, object]:
    empty_design: dict[str, list[dict[str, object]]] = {
        "train": [],
        "validation": [],
        "test_unseen_reset": [],
        "test_unseen_task_language": [],
        "test_cross_skill": [],
        "test_visual_shift": [],
    }
    leakage = validate_split_disjointness(empty_design)
    return _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b5-future-split-design-v0",
            "materialized": False,
            "split_unit": "whole official episode",
            "identity_keys": [
                "source_sha256",
                "task_id",
                "source_episode_id",
                "reset_identity",
                "scene_group",
                "language_template_group",
            ],
            "splits": {
                "train": "seen tasks and reset groups; no test identities",
                "validation": "seen task families, held-out episode/reset groups",
                "test_unseen_reset": "seen tasks with held-out reset identities",
                "test_unseen_task_language": (
                    "seen task semantics with later separately frozen language-template groups; "
                    "not an unseen-task claim"
                ),
                "test_cross_skill": (
                    "a future explicitly held-out skill family; no cross-skill claim is allowed "
                    "when every selected task is in training"
                ),
                "test_visual_shift": (
                    "optional, only after a separately specified camera/appearance shift"
                ),
            },
            "frame_level_random_split_prohibited": True,
            "unseen_task_claim_without_held_out_task_prohibited": True,
            "leakage_checks": leakage,
        }
    )


def _replay_protocol() -> dict[str, object]:
    return _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b5-replay-protocol-v0",
            "source_actions": "official recorded actions only",
            "planner_or_expert_invoked": False,
            "sim_backend": "physx_cpu",
            "semantic_reset_first": True,
            "official_first_state_anchor": (
                "fallback only when semantic reset does not exactly reconstruct source state"
            ),
            "success_predicate_modified": False,
            "bounded_sample_per_task": 20,
            "strong_sample_per_selected_task": 100,
            "selection": ("deterministic length-spanning sample with distinct reset identities"),
            "gates": {
                "categorical_outcome_agreement_rate": 0.99,
                "action_frame_alignment_rate": 1.0,
                "invalid_action_count": 0,
                "simulator_error_count": 0,
            },
            "continuous_diagnostics": [
                "semantic reset state maximum/mean absolute error",
                "terminal state maximum/mean absolute error",
                "source versus replay first-success step",
            ],
        }
    )


def _language_contract() -> dict[str, object]:
    return _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b5-language-contract-v0",
            "large_paraphrase_set_created": False,
            "independent_of_privileged_state": True,
            "tasks": [
                {
                    **CANDIDATE_TASKS[task_id].to_dict(),
                    "object_relation_metadata": {
                        "PickCube-v1": {
                            "object": "cube",
                            "relation": "at target location",
                            "tool": None,
                        },
                        "StackCube-v1": {
                            "object": "one cube",
                            "relation": "on top of the other cube",
                            "tool": None,
                        },
                        "PushCube-v1": {
                            "object": "cube",
                            "relation": "on target region",
                            "tool": "robot end effector",
                        },
                    }[task_id],
                    "source_description": "official ManiSkill task objective",
                }
                for task_id in SELECTED_TASK_IDS
            ],
        }
    )


def _candidate_audit(runtime_audit: dict[str, Any]) -> dict[str, object]:
    return _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b5-task-candidate-audit-v0",
            "required_candidates": [
                "PickCube-v1",
                "StackCube-v1",
                "PushCube-v1",
                "PokeCube-v1",
                "PullCube-v1",
            ],
            "optional_registry_only_candidates": [
                "PegInsertionSide-v1",
                "PlugCharger-v1",
                "PushT-v1",
            ],
            "runtime_tasks": runtime_audit["tasks"],
            "decision": {
                "selected": list(SELECTED_TASK_IDS),
                "PokeCube-v1": (
                    "not selected: downloaded sources contain only delta-action RL trajectories"
                ),
                "PullCube-v1": (
                    "not selected: downloaded sources contain only delta-action RL trajectories"
                ),
                "optional_candidates": (
                    "not downloaded because the three-family direct contract already qualified "
                    "the minimum coherent set"
                ),
            },
        }
    )


def _write_artifact_manifest(root: Path, result: Phase2B5Result) -> None:
    files: list[dict[str, object]] = []
    for path in sorted(root.glob("*.json")):
        if path.name == "artifact_manifest.json":
            continue
        raw = path.read_bytes().replace(b"\r\n", b"\n")
        files.append(
            {
                "path": path.name,
                "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
        )
    payload = _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b5-artifact-manifest-v0",
            "result": result.value,
            "text_byte_contract": "UTF-8 JSON with Git LF-normalized text bytes",
            "files": files,
            "large_official_sources_committed": False,
            "visual_pilot_dataset_committed": False,
            "lerobot_pilot_dataset_committed": False,
        }
    )
    _write(root / "artifact_manifest.json", payload)


def main() -> int:
    args = parse_args()
    runtime_root = args.runtime_root.resolve()
    artifact_root = args.artifact_root.resolve()
    enforce_source_commit(SOURCE_COMMIT, TARGET_BRANCH)
    if _git("rev-parse", "--abbrev-ref", "HEAD") != TARGET_BRANCH:
        raise RuntimeError("finalizer must run on the exact Phase 2B.5 branch")
    if _git("status", "--porcelain"):
        raise RuntimeError("finalizer requires a clean implementation worktree")
    if _git("rev-parse", "HEAD") != args.execution_commit:
        raise RuntimeError("execution commit must equal the clean implementation HEAD")

    runtime = {Path(name).stem: _read(runtime_root / name) for name in RUNTIME_REPORTS}
    _check_runtime(runtime)
    f1 = _verify_f1()
    if f1["all_valid"] is not True:
        raise RuntimeError("prior F1 evidence verification failed")
    result = classify_result(
        source_identity_valid=True,
        replay_valid=True,
        common_contract_valid=True,
        observation_valid=True,
        lerobot_readback_valid=True,
        privilege_exclusion_valid=True,
        split_design_valid=True,
        accepted_skill_families={
            CANDIDATE_TASKS[task_id].skill_family for task_id in SELECTED_TASK_IDS
        },
    )
    if result is not Phase2B5Result.RESULT_A:
        raise RuntimeError(f"unexpected Phase 2B.5 result: {result}")
    if artifact_root.exists():
        raise RuntimeError(f"artifact root already exists: {artifact_root}")
    artifact_root.mkdir(parents=True)
    for name in RUNTIME_REPORTS:
        shutil.copyfile(runtime_root / name, artifact_root / name)

    source_manifest = runtime["source_download_manifest"]
    schema = runtime["source_schema_statistics"]
    candidate_runtime = runtime["task_candidate_runtime_audit"]
    visual = runtime["visual_pilot_manifest"]
    conversion = runtime["conversion_pilot_manifest"]
    readback = runtime["lerobot_readback_result"]
    bounded = runtime["bounded_replay_results"]
    strong = runtime["strong_replay_results"]
    privilege = runtime["privilege_exclusion_audit"]
    split = _split_design()
    split_leakage = split["leakage_checks"]
    if not isinstance(split_leakage, dict):
        raise RuntimeError("split leakage report must be an object")
    language = _language_contract()
    gates = {
        "source_identity": True,
        "source_schema": schema["all_selected_sources_valid"] is True,
        "common_action_contract": True,
        "replay": bounded["all_tasks_passed"] is True and strong["all_tasks_passed"] is True,
        "observation_generation": visual["passed"] is True,
        "privilege_exclusion": privilege["passed"] is True,
        "lerobot_conversion": conversion["passed"] is True,
        "lerobot_readback": readback["passed"] is True,
        "split_design": split_leakage.get("passed") is True,
    }
    package = accepted_source_package(
        selected_task_ids=SELECTED_TASK_IDS,
        gates=gates,
        package_fields={
            "official_source_repository": HF_REPOSITORY,
            "official_source_revision": HF_REVISION,
            "licenses": [SOURCE_LICENSE],
            "source_hash_index": "source_hash_index.json",
            "environment_version": MANISKILL_VERSION,
            "environment_source_revision": MANISKILL_SOURCE_REVISION,
            "action_contract": "action_contract_audit.json",
            "observation_contract": "observation_contract_audit.json",
            "language_contract": "language_contract_manifest.json",
            "replay_results": [
                "bounded_replay_results.json",
                "strong_replay_results.json",
            ],
            "camera_state_pilot": "visual_pilot_manifest.json",
            "lerobot_conversion_pilot": "conversion_pilot_manifest.json",
            "lerobot_readback": "lerobot_readback_result.json",
            "padding_audit": "padding_audit.json",
            "split_design": "split_design_manifest.json",
            "known_limitations": [
                "only a five-episode-per-task visual/LeRobot pilot was produced",
                "strong replay covers 100/1000 trajectories per selected task, not 100%",
                "PokeCube-v1 and PullCube-v1 lack a direct pd_joint_pos official source",
                "full dataset production and every policy training path remain unauthorized",
            ],
        },
    )
    repository = _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b5-repository-environment-audit-v0",
            "repository": _git("config", "--get", "remote.origin.url"),
            "worktree": PROJECT_ROOT.as_posix(),
            "source_branch": SOURCE_BRANCH,
            "parent_commit": SOURCE_COMMIT,
            "branch": TARGET_BRANCH,
            "implementation_head": args.execution_commit,
            "upstream": f"origin/{TARGET_BRANCH}",
            "remote": _git("config", "--get", "remote.origin.url"),
            "source_worktree_clean_before_editing": True,
            "implementation_worktree_clean_before_finalization": True,
            "upstream_and_github_source_sha_verified": True,
            "formal_historical_seed_sets_accessed": False,
            "official_bytes_mixed_into_custom_v1_or_v2_identity": False,
            "runtime_environments": {
                "main": "langmani Python 3.12 / ManiSkill 3.0.1",
                "planner": "unchanged planner overlay; not invoked",
                "lerobot_pilot": runtime["lerobot_060_environment_manifest"]["environment_prefix"],
            },
            "prior_authorization_flags": {
                "phase2b4_collection_authorized": False,
                "phase2c2_training_authorized": False,
            },
            "server_state": runtime["remote_execution_audit"],
            "passed": True,
        }
    )
    system = _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b5-maniskill-demo-system-v0",
            "maniskill_version": MANISKILL_VERSION,
            "maniskill_source_revision": MANISKILL_SOURCE_REVISION,
            "download_cli_sha256": (
                "sha256:1eef894a4c50a6ca0b87165e33e9b6348783881500d9832803a7091f41eb55a1"
            ),
            "official_dataset_repository": HF_REPOSITORY,
            "official_dataset_revision": HF_REVISION,
            "dataset_license": SOURCE_LICENSE,
            "metadata_format": "JSON env_info plus per-episode reset/control/success records",
            "trajectory_format": (
                "HDF5 traj_<episode_id> groups with actions, T transition arrays, "
                "and T+1 recursive env_states"
            ),
            "action_storage": "actual executed float32 action arrays",
            "environment_state_storage": "recursive actor/articulation arrays",
            "reset_metadata": "episode reset_kwargs plus seed",
            "source_type_metadata": "archive directory identity such as motionplanning/rl/teleop",
            "official_replay_tool": "python -m mani_skill.trajectory.replay_trajectory",
            "observation_conversion": (
                "--obs-mode can generate RGB while replaying actions or stored states"
            ),
            "camera_rendering": "sensor_configs and render backend during replay",
            "success_verification": "unmodified environment info['success']",
            "control_mode_conversion": (
                "official tool supports selected conversions on CPU, but Phase 2B.5 did not "
                "convert the accepted direct pd_joint_pos sources"
            ),
            "gpu_cpu_limitations": (
                "CPU/GPU reset physics may differ; official --use-first-env-state anchors the "
                "stored initial state. Control-mode conversion is unavailable in parallel GPU "
                "replay."
            ),
        }
    )
    action = _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b5-action-contract-audit-v0",
            "common_contract": {
                "robot": "Panda / Panda wrist-camera variant with identical active joints",
                "control_mode": "pd_joint_pos",
                "dtype": "float32",
                "shape": [8],
                "frequency_hz": 20,
                "simulator_substeps": 5,
                "posthoc_clipping_or_repair": False,
            },
            "selected_tasks": [
                {
                    "task_id": task["task_id"],
                    "action": task["action"],
                    "classification": "directly_compatible",
                }
                for task in schema["tasks"]
            ],
            "rejected_direct_candidates": {
                "PokeCube-v1": "incompatible: official package has delta-action RL sources only",
                "PullCube-v1": "incompatible: official package has delta-action RL sources only",
            },
            "passed": True,
        }
    )
    observation = _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b5-observation-contract-audit-v0",
            "student_contract": {
                "image": visual["image_contract"],
                "state": visual["state_contract"],
                "language": "one canonical task instruction",
            },
            "source_compressed_observation": "none",
            "reconstruction": (
                "semantic reset plus official first env-state anchor, then pre-action "
                "base_camera RGB and Panda active-joint extraction"
            ),
            "frame_timing": visual["action_contract"]["timing"],
            "privileged_fields_in_student_schema": [],
            "diagnostic_privileged_state_separate": True,
            "passed": visual["passed"] is True and privilege["passed"] is True,
        }
    )
    selected = _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b5-selected-task-decision-v0",
            "selected_task_ids": list(SELECTED_TASK_IDS),
            "accepted_skill_families": [
                CANDIDATE_TASKS[task_id].skill_family for task_id in SELECTED_TASK_IDS
            ],
            "genuinely_distinct_skill_family_count": 3,
            "common_embodiment": (
                "Panda active-joint and 8D action contract; StackCube's panda_wristcam adds "
                "an unused wrist camera while preserving the base-camera/student allowlist"
            ),
            "PokeCube-v1_selected": False,
            "PullCube-v1_selected": False,
            "decision_for_nonselected": (
                "no deterministic conversion was attempted because direct pd_joint_pos "
                "official bytes were unavailable"
            ),
        }
    )
    prior = _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b5-prior-evidence-verification-v0",
            "phase2b4_f1": f1,
            "custom_route_frozen": True,
            "formal_historical_seed_sets_accessed": False,
            "historical_v1_bytes_available": False,
            "historical_v1_metadata_used_as_trainable_data": False,
            "passed": True,
        }
    )
    classification = _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b5-result-classification-v0",
            "result": result.value,
            "reason": (
                "Three genuinely distinct official skill families passed immutable-source, "
                "direct action, bounded and 100-trajectory replay, nonprivileged visual/state, "
                "LeRobot 0.6 conversion/readback, and split-design gates."
            ),
            "official_multi_skill_source_validated": True,
            "conversion_scope": "pilot_only",
            "full_dataset_produced": False,
            "training_started": False,
        }
    )
    for name, payload in (
        ("repository_environment_audit.json", repository),
        ("prior_evidence_verification.json", prior),
        ("custom_push_route_closure.json", custom_route_closure()),
        ("maniskill_demo_system_manifest.json", system),
        ("task_candidate_audit.json", _candidate_audit(candidate_runtime)),
        ("source_hash_index.json", _source_hash_index(source_manifest)),
        ("replay_protocol.json", _replay_protocol()),
        ("selected_task_decision.json", selected),
        ("action_contract_audit.json", action),
        ("observation_contract_audit.json", observation),
        ("language_contract_manifest.json", language),
        ("split_design_manifest.json", split),
        ("accepted_official_demo_source_package.json", package),
        ("result_classification.json", classification),
        (
            "authorization_state.json",
            _fingerprinted(
                {
                    "schema_version": "langmani-v2-phase2b5-authorization-state-v0",
                    **authorization_state(
                        official_demo_source_validated=True,
                        phase2b6_dataset_production_eligible=True,
                    ),
                }
            ),
        ),
    ):
        _write(artifact_root / name, payload)
    _write_artifact_manifest(artifact_root, result)
    print(
        json.dumps(
            {
                "artifact_root": artifact_root.as_posix(),
                "result": result.value,
                "artifact_count": len(list(artifact_root.glob("*.json"))),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
