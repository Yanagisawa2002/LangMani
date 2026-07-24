"""Native no-training production runtime for LangMani Phase 2B.6-v2."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import h5py  # type: ignore[import-untyped]
import numpy as np

from langmani.v2.phase2b5 import (
    CANDIDATE_TASKS,
    CONTROL_MODE,
    IMAGE_SHAPE,
    canonical_json_sha256,
    sha256_file,
    validate_metadata_document,
)
from langmani.v2.phase2b6 import (
    TASK_IDS,
    build_cross_skill_folds,
    build_language_template_manifest,
    build_padding_audit,
    build_primary_split_manifest,
    build_task_balance_manifest,
    verify_phase2b5_package,
)
from langmani.v2.phase2b6 import (
    load_production_spec as load_v1_spec,
)
from langmani.v2.phase2b6_1_runtime import inventory_frozen_output
from langmani.v2.phase2b6_1_v2_runtime import _verify_manifest_directory
from langmani.v2.phase2b6_1r_runtime import _minimal_sapien_probe
from langmani.v2.phase2b6_runtime import (
    _array_sha256,
    _contact_sheet,
    _contact_sheet_episode_ids,
    _environment_kwargs,
    _gpu_audit,
    _process_audit,
    _read_json,
    _read_jsonl,
    _replay_one,
    _save_episode_npz,
    _selected_paths,
    _sensor_parameter_report,
    _task_slug,
    _timestamp,
    _write_json,
    validate_source_schema,
    verify_source_bytes,
)
from langmani.v2.phase2b6_v2 import (
    EXPECTED_SOURCE_EPISODES,
    EXPECTED_SOURCE_TRANSITIONS,
    PACKAGE_ID,
    REQUIRED_SUB_GATES,
    SOURCE_COMMIT,
    TARGET_BRANCH,
    EpisodeClass,
    accepted_assignments,
    authorization_state,
    classify_attempt,
    enforce_starting_point,
    fingerprinted,
    gate,
    load_production_spec,
    portable_text_sha256,
    split_count_report,
    threshold_audit,
)

_PRIOR_EVIDENCE = {
    "phase_2b5": (
        "sha256:e65adf42d3329066f52ee440fa95b095946274dd54833c04fc7c5f6aa36838c4",
        "RESULT_A",
    ),
    "phase_2b6": (
        "sha256:a4b4ace15b3d5bd5dfb2215507abd14fcbe1a0a21dce0f875d6d91dcc676f8a0",
        "RESULT_C",
    ),
    "phase_2b6_1": (
        "sha256:17427bdf2dda9ad4dcebd92e678b695f89b7f2fe147725a1daf5d6cf818ca826",
        "RESULT_D",
    ),
    "phase_2b6_1r": (
        "sha256:23406ce54be296e47d0b8186fe1d5f10a714df50d800c429b43e6c78ebfba6c9",
        "RESULT_A",
    ),
    "phase_2b6_1_v2": (
        "sha256:c49558caa9f04d445f8e34acd6e71d17b0339a82c4a5273ac8f8827f35e15a6e",
        "RESULT_B",
    ),
}


class Phase2B6V2RuntimeError(RuntimeError):
    """Raised when production cannot preserve the frozen v2 contract."""


def _git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode:
        raise Phase2B6V2RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _materialized_contract(repo_root: Path, spec: Mapping[str, object]) -> dict[str, Any]:
    """Materialize the v1-stable language/split contract under the new v2 identity."""

    base_path = repo_root / "configs" / "langmani_v2" / "phase2b6_dataset_production.yaml"
    expected = cast(Mapping[str, object], spec["source_authorization"])["phase2b6_v1_spec_sha256"]
    if "sha256:" + portable_text_sha256(base_path) != expected:
        raise Phase2B6V2RuntimeError("frozen Phase 2B.6 language/split source changed")
    base = load_v1_spec(base_path)
    base["production_run_id"] = spec["production_run_id"]
    base["derived_dataset_version"] = PACKAGE_ID
    base["tasks"] = spec["tasks"]
    base["runtime"] = spec["runtime"]
    base["camera"] = spec["camera"]
    base["visual_shift"] = spec["visual_shift"]
    base["_spec_path"] = spec["_path"]
    base["_spec_sha256"] = spec["_sha256"]
    return base


def verify_prior_evidence(repo_root: Path) -> dict[str, object]:
    """Rehash every immutable prerequisite evidence package."""

    reports: dict[str, object] = {}
    for name, (fingerprint, result_name) in _PRIOR_EVIDENCE.items():
        root = repo_root / "artifacts" / "langmani_v2" / name
        report = _verify_manifest_directory(root, fingerprint)
        result_path = root / "result_classification.json"
        if result_path.is_file():
            result = _read_json(result_path).get("result")
        elif name == "phase_2b5":
            result = "RESULT_A"
        else:
            result = None
        report["expected_result"] = result_name
        report["observed_result"] = result
        report["result_valid"] = result == result_name
        report["passed"] = report["passed"] is True and report["result_valid"] is True
        reports[name] = report
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-v2-prior-evidence-v0",
            "evidence_sets": reports,
            "passed": all(
                cast(Mapping[str, object], report)["passed"] is True for report in reports.values()
            ),
        }
    )


def _storage_audit(paths: Sequence[Path]) -> dict[str, object]:
    reports: dict[str, object] = {}
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(path.parent)
        reports[path.as_posix()] = {
            "volume": path.parent.as_posix(),
            "free_bytes": usage.free,
            "total_bytes": usage.total,
        }
    minimum = min(
        cast(int, cast(Mapping[str, object], row)["free_bytes"]) for row in reports.values()
    )
    return {
        "paths": reports,
        "minimum_free_bytes": minimum,
        "minimum_required_bytes": 60 * 1024**3,
        "passed": minimum >= 60 * 1024**3,
    }


def repository_environment_audit(
    *,
    repo_root: Path,
    source_root: Path,
    frozen_root: Path,
    production_root: Path,
    archive_root: Path,
    restore_root: Path,
    spec: Mapping[str, object],
    network_turbo_sourced: bool,
) -> dict[str, object]:
    """Verify isolated Git, roots, processes, GPU, and capacity."""

    head = _git(repo_root, "rev-parse", "HEAD")
    branch = _git(repo_root, "branch", "--show-current")
    source_ancestor = _git(repo_root, "merge-base", SOURCE_COMMIT, head) == SOURCE_COMMIT
    no_merges = not _git(repo_root, "rev-list", "--merges", f"{SOURCE_COMMIT}..{head}")
    enforce_starting_point(
        source_commit=SOURCE_COMMIT,
        branch=branch,
        source_is_ancestor=source_ancestor,
        no_merges_since_source=no_merges,
    )
    upstream = _git(repo_root, "rev-parse", "@{upstream}")
    remote_line = _git(repo_root, "ls-remote", "origin", f"refs/heads/{TARGET_BRANCH}", check=False)
    github_sha = remote_line.split()[0] if remote_line else None
    status = _git(repo_root, "status", "--porcelain=v1")
    roots_new = all(not path.exists() for path in (production_root, archive_root, restore_root))
    storage = _storage_audit((production_root, archive_root, restore_root))
    process = _process_audit()
    checks = {
        "target_branch": branch == TARGET_BRANCH,
        "source_commit_is_ancestor": source_ancestor,
        "no_merge_commits_since_source": no_merges,
        "worktree_clean": not status,
        "upstream_matches_head": upstream == head,
        "github_matches_head": github_sha == head,
        "source_root_exists": source_root.is_dir(),
        "frozen_v1_root_exists": frozen_root.is_dir(),
        "new_roots_absent": roots_new,
        "new_root_isolation": all(
            frozen_root != path and frozen_root not in path.parents
            for path in (production_root, archive_root, restore_root)
        ),
        "process_state": process["passed"] is True,
        "storage": storage["passed"] is True,
        "network_turbo_sourced": network_turbo_sourced,
    }
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-v2-repository-environment-v0",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "head": head,
            "branch": branch,
            "upstream": upstream,
            "github_branch_sha": github_sha,
            "remote": _git(repo_root, "remote", "get-url", "origin"),
            "source_commit": SOURCE_COMMIT,
            "source_root": source_root.as_posix(),
            "frozen_v1_root": frozen_root.as_posix(),
            "production_root": production_root.as_posix(),
            "archive_root": archive_root.as_posix(),
            "restore_root": restore_root.as_posix(),
            "specification_sha256": spec["_sha256"],
            "exclusion_policy_sha256": cast(Mapping[str, object], spec["_policy"])["_sha256"],
            "gpu": _gpu_audit(),
            "process_state": process,
            "storage": storage,
            "authorization": authorization_state(accepted=False),
            "checks": checks,
            "passed": all(checks.values()),
        }
    )


def prepare_production(
    *,
    repo_root: Path,
    spec_path: Path,
    phase2b5_artifact_root: Path,
    source_root: Path,
    frozen_root: Path,
    production_root: Path,
    archive_root: Path,
    restore_root: Path,
    evidence_root: Path,
    network_turbo_sourced: bool,
) -> dict[str, object]:
    """Freeze all source, policy, split, and environment identities."""

    spec = load_production_spec(spec_path, repo_root=repo_root)
    repository = repository_environment_audit(
        repo_root=repo_root,
        source_root=source_root,
        frozen_root=frozen_root,
        production_root=production_root,
        archive_root=archive_root,
        restore_root=restore_root,
        spec=spec,
        network_turbo_sourced=network_turbo_sourced,
    )
    prior = verify_prior_evidence(repo_root)
    contract = _materialized_contract(repo_root, spec)
    phase2b5 = verify_phase2b5_package(artifact_root=phase2b5_artifact_root, spec=contract)
    source = verify_source_bytes(source_root=source_root, spec=contract)
    frozen_before = inventory_frozen_output(frozen_root)
    if not all(
        report["passed"] is True for report in (repository, prior, phase2b5, source, frozen_before)
    ):
        raise Phase2B6V2RuntimeError("Phase 2B.6-v2 immutable preflight hard stop")
    production_root.mkdir(parents=True, exist_ok=False)
    evidence_root.mkdir(parents=True, exist_ok=True)
    source_schema, inventory = validate_source_schema(
        source_root=source_root,
        production_root=production_root,
        spec=contract,
    )
    if source_schema["passed"] is not True:
        raise Phase2B6V2RuntimeError("source schema hard stop")
    language = build_language_template_manifest(contract)
    splits = build_primary_split_manifest(spec=contract, episodes=inventory)
    folds = build_cross_skill_folds(spec=contract, split_manifest=splits)
    padding = build_padding_audit(cast(Sequence[Mapping[str, object]], splits["assignments"]))
    balance = build_task_balance_manifest(
        cast(Sequence[Mapping[str, object]], splits["assignments"])
    )
    frozen_spec = {key: value for key, value in spec.items() if not key.startswith("_")}
    policy = cast(Mapping[str, object], spec["_policy"])
    policy_manifest = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-v2-exclusion-policy-manifest-v0",
            "policy_id": policy["policy_id"],
            "policy_sha256": policy["_sha256"],
            "policy_fingerprint": policy["_fingerprint"],
            "registered_exclusions": policy["registered_exclusions"],
            "frozen_before_first_replay": True,
            "passed": True,
        }
    )
    frozen_spec_report = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-v2-frozen-spec-v0",
            "specification_sha256": spec["_sha256"],
            "specification_fingerprint": canonical_json_sha256(frozen_spec),
            "production_run_id": spec["production_run_id"],
            "passed": True,
        }
    )
    observation_contract = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-v2-observation-action-contract-v0",
            "observation": spec["observation"],
            "camera": spec["camera"],
            "action": spec["action"],
            "observation_before_action": True,
            "terminal_diagnostic_frame_in_policy_data": False,
            "fabricated_terminal_action": False,
            "privileged_fields_excluded": True,
            "student_policy_training_started": False,
            "optimizer_steps": 0,
            "passed": True,
        }
    )
    source_inventory = _read_json(production_root / "work" / "source_episode_inventory.json")
    documents = {
        "repository_environment_audit.json": repository,
        "prior_evidence_verification.json": prior,
        "phase2b5_source_package_verification.json": phase2b5,
        "source_inventory.json": source_inventory,
        "source_schema_result.json": source_schema,
        "source_integrity_audit.json": source,
        "frozen_partial_output_inventory_before.json": frozen_before,
        "exclusion_policy_manifest.json": policy_manifest,
        "frozen_production_specification.json": frozen_spec_report,
        "observation_action_contract.json": observation_contract,
        "language_manifest.json": language,
        "primary_split_manifest.json": splits,
        "cross_skill_folds.json": folds,
        "padding_audit_prereplay.json": padding,
        "task_balance_prereplay.json": balance,
        "authorization_state.json": fingerprinted(authorization_state(accepted=False)),
    }
    for name, document in documents.items():
        _write_json(evidence_root / name, document)
    run_root = production_root / "run"
    _write_json(run_root / "frozen_production_specification.json", frozen_spec)
    _write_json(
        run_root / "frozen_exclusion_policy.json",
        {key: value for key, value in policy.items() if not key.startswith("_")},
    )
    _write_json(production_root / "work" / "primary_split_manifest.json", splits)
    _write_json(production_root / "work" / "language_template_manifest.json", language)
    _write_json(production_root / "work" / "cross_skill_fold_manifests.json", folds)
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-v2-preflight-v0",
            "production_run_id": spec["production_run_id"],
            "source_episode_count": EXPECTED_SOURCE_EPISODES,
            "source_transition_count": EXPECTED_SOURCE_TRANSITIONS,
            "documents": sorted(documents),
            "student_policy_training_started": False,
            "optimizer_steps": 0,
            "passed": True,
        }
    )


def run_rendering_preflight(
    *, source_root: Path, production_root: Path, evidence_root: Path
) -> dict[str, object]:
    """Run Vulkan, minimal SAPIEN, and zero-step construction for all tasks."""

    vulkan = subprocess.run(
        ["vulkaninfo", "--summary"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    vulkan_passed = vulkan.returncode == 0 and "NVIDIA GeForce RTX 5090" in vulkan.stdout
    sapien = _minimal_sapien_probe()
    tasks: list[dict[str, object]] = []
    if vulkan_passed and sapien["passed"] is True:
        try:
            import gymnasium as gym
            import mani_skill.envs  # type: ignore[import-untyped]  # noqa: F401
        except (ImportError, OSError, RuntimeError) as error:
            raise Phase2B6V2RuntimeError("rendering preflight requires ManiSkill") from error
        for task_id in TASK_IDS:
            _, metadata_path = _selected_paths(source_root, task_id)
            document = _read_json(metadata_path)
            env: Any | None = None
            started = time.perf_counter()
            try:
                env = gym.make(task_id, **cast(Any, _environment_kwargs(document)))
                base: Any = env.unwrapped
                action_shape = tuple(cast(tuple[int, ...], env.action_space.shape))
                task_report: dict[str, object] = {
                    "task_id": task_id,
                    "control_mode": str(base.control_mode),
                    "control_frequency_hz": int(base.control_freq),
                    "action_shape": list(action_shape),
                    "observation_space": str(env.observation_space),
                    "explicit_reset_count": 0,
                    "explicit_step_count": 0,
                    "action_submission_count": 0,
                    "wall_clock_seconds": time.perf_counter() - started,
                    "exception": None,
                    "passed": (
                        str(base.control_mode) == CONTROL_MODE
                        and int(base.control_freq) == 20
                        and action_shape == (8,)
                    ),
                }
            except Exception as error:  # noqa: BLE001 - external simulator boundary
                task_report = {
                    "task_id": task_id,
                    "exception": {"type": type(error).__name__, "message": str(error)},
                    "explicit_reset_count": 0,
                    "explicit_step_count": 0,
                    "action_submission_count": 0,
                    "wall_clock_seconds": time.perf_counter() - started,
                    "passed": False,
                }
            finally:
                if env is not None:
                    env.close()
            tasks.append(task_report)
            if task_report["passed"] is not True:
                break
    report = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-v2-rendering-contract-v0",
            "created_at_utc": _timestamp(),
            "vulkaninfo": {
                "returncode": vulkan.returncode,
                "intended_gpu_listed": "NVIDIA GeForce RTX 5090" in vulkan.stdout,
                "stderr": vulkan.stderr[-2000:],
                "passed": vulkan_passed,
            },
            "minimal_sapien": sapien,
            "zero_step_tasks": tasks,
            "validated_launcher_environment": {
                "VK_ICD_FILENAMES": os.environ.get("VK_ICD_FILENAMES"),
                "__EGL_VENDOR_LIBRARY_FILENAMES": os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES"),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "XDG_RUNTIME_DIR": os.environ.get("XDG_RUNTIME_DIR"),
                "DISPLAY": os.environ.get("DISPLAY"),
                "WAYLAND_DISPLAY": os.environ.get("WAYLAND_DISPLAY"),
            },
            "gpu": _gpu_audit(),
            "passed": (
                vulkan_passed
                and sapien["passed"] is True
                and len(tasks) == 3
                and all(row["passed"] is True for row in tasks)
            ),
        }
    )
    _write_json(evidence_root / "validated_rendering_contract_manifest.json", report)
    _write_json(production_root / "run" / "rendering_preflight.json", report)
    if report["passed"] is not True:
        raise Phase2B6V2RuntimeError("validated Vulkan rendering preflight hard stop")
    return report


def _initial_sub_gates(rendering_fingerprint: object) -> dict[str, dict[str, object]]:
    gates = {name: gate("not_reached") for name in REQUIRED_SUB_GATES}
    gates["rendering_preflight_gate"] = gate(
        "passed", rendering_contract_fingerprint=rendering_fingerprint
    )
    gates["environment_construction_gate"] = gate("passed")
    return gates


def _validate_serialized_npz(path: Path, *, action_count: int) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as arrays:
        shapes = {name: list(arrays[name].shape) for name in arrays.files}
        dtypes = {name: str(arrays[name].dtype) for name in arrays.files}
        passed = (
            set(arrays.files) == {"rgb", "state", "action", "timestamp"}
            and shapes["rgb"] == [action_count, *IMAGE_SHAPE]
            and shapes["state"] == [action_count, 9]
            and shapes["action"] == [action_count, 8]
            and shapes["timestamp"] == [action_count]
            and dtypes
            == {
                "rgb": "uint8",
                "state": "float32",
                "action": "float32",
                "timestamp": "float64",
            }
        )
    return {
        "path": path.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": "sha256:" + sha256_file(path),
        "shapes": shapes,
        "dtypes": dtypes,
        "passed": passed,
    }


def _attempt_record(
    *,
    task_id: str,
    source_episode_id: int,
    assignment: Mapping[str, object],
    replay: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
    sub_gates: dict[str, dict[str, object]],
    attempt_index: int,
    output_path: Path,
    production_root: Path,
    retryable_reason: str | None = None,
) -> dict[str, object]:
    action_count = cast(int, replay["source_action_count"])
    sub_gates["reset_identity_gate"] = gate(
        "passed",
        reset_identity=assignment["reset_identity"],
        semantic_reset_state_error=replay["semantic_reset_state_error"],
        first_state_anchor_used=replay["first_state_anchor_used"],
    )
    action_passed = (
        replay["invalid_action_count"] == 0
        and replay["nonfinite_value_count"] == 0
        and replay["source_action_sha256"] == assignment["action_sha256"]
    )
    sub_gates["action_contract_gate"] = gate(
        "passed" if action_passed else "failed",
        action_count=action_count,
        action_shape=[action_count, 8],
        action_dtype="float32",
        action_sha256=replay["source_action_sha256"],
        expected_action_sha256=assignment["action_sha256"],
        invalid_action_count=replay["invalid_action_count"],
        nonfinite_value_count=replay["nonfinite_value_count"],
    )
    all_actions = replay["replayed_action_count"] == action_count
    sub_gates["step_execution_gate"] = gate(
        "passed" if all_actions else "failed",
        source_action_count=action_count,
        replayed_action_count=replay["replayed_action_count"],
    )
    sub_gates["simulator_exception_gate"] = gate(
        "passed" if replay["simulator_error_count"] == 0 else "failed",
        simulator_error_count=replay["simulator_error_count"],
    )
    sub_gates["canonical_terminal_success_gate"] = gate(
        "passed" if replay["replay_success"] is True else "failed",
        final_canonical_success=replay["replay_success"],
    )
    sub_gates["source_replay_outcome_agreement_gate"] = gate(
        "passed" if replay["categorical_outcome_agreement"] is True else "failed",
        source_success=replay["source_success"],
        replay_success=replay["replay_success"],
    )
    sub_gates["action_count_gate"] = gate(
        "passed" if all_actions else "failed",
        source_action_count=action_count,
        replay_action_count=replay["replayed_action_count"],
    )
    frame_count = int(arrays["rgb"].shape[0])
    sub_gates["pre_action_frame_count_gate"] = gate(
        "passed" if frame_count == action_count else "failed",
        policy_frame_count=frame_count,
        action_count=action_count,
    )
    sub_gates["observation_action_alignment_gate"] = gate(
        "passed" if frame_count == action_count else "failed",
        convention="observation[t] immediately precedes action[t]",
        terminal_diagnostic_frame_in_policy_data=False,
    )
    sub_gates["rgb_completeness_gate"] = gate(
        "passed"
        if arrays["rgb"].shape == (action_count, *IMAGE_SHAPE) and arrays["rgb"].dtype == np.uint8
        else "failed",
        shape=list(arrays["rgb"].shape),
        dtype=str(arrays["rgb"].dtype),
    )
    sub_gates["state_completeness_gate"] = gate(
        "passed"
        if arrays["state"].shape == (action_count, 9)
        and arrays["state"].dtype == np.float32
        and bool(np.all(np.isfinite(arrays["state"])))
        else "failed",
        shape=list(arrays["state"].shape),
        dtype=str(arrays["state"].dtype),
    )
    expected_timestamps = np.arange(action_count, dtype=np.float64) / 20.0
    timestamp_passed = np.array_equal(arrays["timestamp"], expected_timestamps)
    sub_gates["timestamp_monotonicity_gate"] = gate(
        "passed" if timestamp_passed else "failed",
        frame_count=len(arrays["timestamp"]),
        strictly_monotonic=bool(np.all(np.diff(arrays["timestamp"]) > 0)),
        exact_20hz=timestamp_passed,
    )
    classification = classify_attempt(
        sub_gates=sub_gates,
        full_action_execution=all_actions,
        retryable_reason=retryable_reason,
    )
    return {
        "attempt_index": attempt_index,
        "retry": attempt_index > 0,
        "retry_reason": retryable_reason,
        "terminal_attempt": classification is not EpisodeClass.RETRYABLE_INFRASTRUCTURE,
        "task_id": task_id,
        "skill_family": assignment["skill_family"],
        "source_episode_id": source_episode_id,
        "source_trajectory_identity": assignment["source_trajectory_identity"],
        "derived_episode_identity": assignment["derived_episode_identity"],
        "reset_identity": assignment["reset_identity"],
        "instruction_template_id": assignment["instruction_template_id"],
        "instruction": assignment["instruction"],
        "primary_split": assignment["primary_split"],
        "source_action_count": action_count,
        "source_action_sha256": replay["source_action_sha256"],
        "replayed_action_count": replay["replayed_action_count"],
        "policy_frame_count": frame_count,
        "action_frame_alignment": frame_count == action_count,
        "source_success": replay["source_success"],
        "replay_success": replay["replay_success"],
        "canonical_success_action_indices": replay["canonical_success_action_indices"],
        "first_canonical_success_action_index": replay["first_canonical_success_action_index"],
        "maximum_consecutive_canonical_success_steps": replay[
            "maximum_consecutive_canonical_success_steps"
        ],
        "trailing_canonical_success_steps": replay["trailing_canonical_success_steps"],
        "categorical_outcome_agreement": replay["categorical_outcome_agreement"],
        "invalid_action_count": replay["invalid_action_count"],
        "nonfinite_value_count": replay["nonfinite_value_count"],
        "simulator_error_count": replay["simulator_error_count"],
        "terminated_signal_count": replay["terminated_signal_count"],
        "truncated_signal_count": replay["truncated_signal_count"],
        "terminal_object_pose_error": replay["terminal_object_pose_error"],
        "rgb_sha256": _array_sha256(arrays["rgb"]),
        "state_sha256": _array_sha256(arrays["state"], dtype=np.dtype(np.float32)),
        "derived_action_sha256": _array_sha256(arrays["action"], dtype=np.dtype(np.float32)),
        "timestamp_sha256": _array_sha256(arrays["timestamp"], dtype=np.dtype(np.float64)),
        "raw_relative_path": (
            output_path.relative_to(production_root).as_posix()
            if classification is EpisodeClass.ACCEPTED_REPLAY
            else None
        ),
        "raw_sha256": (
            "sha256:" + sha256_file(output_path)
            if classification is EpisodeClass.ACCEPTED_REPLAY and output_path.is_file()
            else None
        ),
        "sub_gates": sub_gates,
        "first_failed_sub_gate": next(
            (name for name in REQUIRED_SUB_GATES if sub_gates[name]["status"] == "failed"),
            None,
        ),
        "classification": classification.value,
        "passed": classification is EpisodeClass.ACCEPTED_REPLAY,
    }


def run_full_production(
    *,
    repo_root: Path,
    source_root: Path,
    frozen_root: Path,
    production_root: Path,
    evidence_root: Path,
    spec_path: Path,
) -> dict[str, object]:
    """Replay all 3,000 sources once and apply the preregistered exclusions."""

    try:
        import gymnasium as gym
        import mani_skill.envs  # type: ignore[import-untyped]  # noqa: F401
    except (ImportError, OSError, RuntimeError) as error:
        raise Phase2B6V2RuntimeError("full production requires ManiSkill") from error
    load_production_spec(spec_path, repo_root=repo_root)
    rendering = _read_json(evidence_root / "validated_rendering_contract_manifest.json")
    if rendering.get("passed") is not True:
        raise Phase2B6V2RuntimeError("rendering preflight is not accepted")
    assignments_doc = _read_json(production_root / "work" / "primary_split_manifest.json")
    assignments = cast(list[dict[str, object]], assignments_doc["assignments"])
    by_key = {
        (str(row["task_id"]), int(cast(int, row["source_episode_id"]))): row for row in assignments
    }
    progress_path = production_root / "work" / "production_attempts.jsonl"
    if progress_path.exists():
        raise Phase2B6V2RuntimeError("v2 production is non-resumable and must start from episode 0")
    marker = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-v2-production-start-v0",
            "started_at_utc": _timestamp(),
            "producer_commit": _git(repo_root, "rev-parse", "HEAD"),
            "producer_worktree_clean": not _git(repo_root, "status", "--porcelain=v1"),
            "source_episode_start": 0,
            "previous_partial_records_reused": False,
            "student_policy_training_started": False,
            "optimizer_steps": 0,
        }
    )
    if marker["producer_worktree_clean"] is not True:
        raise Phase2B6V2RuntimeError("full production requires a clean committed producer")
    _write_json(production_root / "run" / "full_production_started.json", marker)
    camera_reports: dict[str, object] = {}
    exclusions: Counter[str] = Counter()
    started = time.perf_counter()
    for task_id in TASK_IDS:
        h5_path, json_path = _selected_paths(source_root, task_id)
        document = _read_json(json_path)
        episodes = validate_metadata_document(document, expected_task_id=task_id)
        env: Any | None = None
        construction_attempts: list[dict[str, object]] = []
        for construction_index in range(2):
            try:
                env = gym.make(task_id, **cast(Any, _environment_kwargs(document)))
                construction_attempts.append(
                    {"attempt_index": construction_index, "passed": True, "exception": None}
                )
                break
            except Exception as error:  # noqa: BLE001 - simulator construction boundary
                construction_attempts.append(
                    {
                        "attempt_index": construction_index,
                        "passed": False,
                        "exception": {"type": type(error).__name__, "message": str(error)},
                    }
                )
        if env is None:
            _write_json(
                evidence_root / f"construction_failure_{_task_slug(task_id)}.json",
                fingerprinted(
                    {
                        "schema_version": "langmani-v2-phase2b6-v2-construction-failure-v0",
                        "task_id": task_id,
                        "attempts": construction_attempts,
                        "result": "RESULT_C",
                        "passed": False,
                    }
                ),
            )
            raise Phase2B6V2RuntimeError(f"environment construction hard stop: {task_id}")
        base: Any = env.unwrapped
        if str(base.control_mode) != CONTROL_MODE or int(base.control_freq) != 20:
            env.close()
            raise Phase2B6V2RuntimeError("production action timing changed")
        env.reset(**cast(Mapping[str, Any], episodes[0]["reset_kwargs"]))
        camera_reports[task_id] = {
            "construction_attempts": construction_attempts,
            "parameters": _sensor_parameter_report(base),
        }
        contact_ids = _contact_sheet_episode_ids(assignments, task_id)
        try:
            with h5py.File(h5_path, "r") as trajectories:
                for metadata in sorted(episodes, key=lambda row: int(row["episode_id"])):
                    source_episode_id = int(metadata["episode_id"])
                    assignment = by_key[(task_id, source_episode_id)]
                    output_path = (
                        production_root
                        / "work"
                        / "raw"
                        / _task_slug(task_id)
                        / f"episode_{source_episode_id:06d}.npz"
                    )
                    sub_gates = _initial_sub_gates(rendering["fingerprint"])
                    try:
                        replay, arrays = _replay_one(
                            env=env,
                            base=base,
                            task_id=task_id,
                            metadata=metadata,
                            group=cast(h5py.Group, trajectories[f"traj_{source_episode_id}"]),
                            capture_arrays=True,
                        )
                        if arrays is None:
                            raise Phase2B6V2RuntimeError("replay did not capture policy arrays")
                    except Exception as error:  # noqa: BLE001 - external execution boundary
                        sub_gates["step_execution_gate"] = gate(
                            "failed",
                            exception={"type": type(error).__name__, "message": str(error)},
                        )
                        sub_gates["simulator_exception_gate"] = gate(
                            "failed",
                            exception={"type": type(error).__name__, "message": str(error)},
                        )
                        failed_row = {
                            "attempt_index": 0,
                            "retry": False,
                            "retry_reason": None,
                            "terminal_attempt": True,
                            "task_id": task_id,
                            "source_episode_id": source_episode_id,
                            "source_trajectory_identity": assignment["source_trajectory_identity"],
                            "derived_episode_identity": assignment["derived_episode_identity"],
                            "sub_gates": sub_gates,
                            "first_failed_sub_gate": "step_execution_gate",
                            "classification": EpisodeClass.UNCLASSIFIED_FAILURE.value,
                            "error": {"type": type(error).__name__, "message": str(error)},
                            "passed": False,
                        }
                        _append_jsonl(progress_path, failed_row)
                        raise Phase2B6V2RuntimeError(
                            f"unclassified replay hard stop: {task_id}/{source_episode_id}"
                        ) from error
                    assert arrays is not None
                    physical_passed = (
                        replay["replay_success"] is True
                        and replay["categorical_outcome_agreement"] is True
                        and replay["replayed_action_count"] == replay["source_action_count"]
                        and replay["invalid_action_count"] == 0
                        and replay["nonfinite_value_count"] == 0
                    )
                    if physical_passed:
                        writer_error: Exception | None = None
                        final_attempt_index = 0
                        for writer_attempt in range(2):
                            try:
                                _save_episode_npz(output_path, arrays)
                                serialization = _validate_serialized_npz(
                                    output_path,
                                    action_count=cast(int, replay["source_action_count"]),
                                )
                                if serialization["passed"] is not True:
                                    raise Phase2B6V2RuntimeError("serialized NPZ readback changed")
                                sub_gates["temporary_writer_gate"] = gate(
                                    "passed",
                                    attempt_index=writer_attempt,
                                    atomic_rename=True,
                                    file_sha256=serialization["sha256"],
                                )
                                sub_gates["episode_serialization_gate"] = gate(
                                    "passed", **serialization
                                )
                                final_attempt_index = writer_attempt
                                writer_error = None
                                break
                            except (OSError, Phase2B6V2RuntimeError) as error:
                                writer_error = error
                                sub_gates["temporary_writer_gate"] = gate(
                                    "failed",
                                    attempt_index=writer_attempt,
                                    exception={
                                        "type": type(error).__name__,
                                        "message": str(error),
                                    },
                                )
                                sub_gates["episode_serialization_gate"] = gate("not_reached")
                                retry_row = _attempt_record(
                                    task_id=task_id,
                                    source_episode_id=source_episode_id,
                                    assignment=assignment,
                                    replay=replay,
                                    arrays=arrays,
                                    sub_gates={
                                        name: dict(value) for name, value in sub_gates.items()
                                    },
                                    attempt_index=writer_attempt,
                                    output_path=output_path,
                                    production_root=production_root,
                                    retryable_reason="temporary_writer_initialization",
                                )
                                retry_row["terminal_attempt"] = False
                                _append_jsonl(progress_path, retry_row)
                                if output_path.exists():
                                    output_path.unlink()
                        if writer_error is not None:
                            raise Phase2B6V2RuntimeError(
                                f"writer retry exhausted: {task_id}/{source_episode_id}"
                            ) from writer_error
                    else:
                        final_attempt_index = 0
                    row = _attempt_record(
                        task_id=task_id,
                        source_episode_id=source_episode_id,
                        assignment=assignment,
                        replay=replay,
                        arrays=arrays,
                        sub_gates=sub_gates,
                        attempt_index=final_attempt_index,
                        output_path=output_path,
                        production_root=production_root,
                    )
                    if (
                        task_id == "StackCube-v1"
                        and source_episode_id == 938
                        and row["classification"]
                        != EpisodeClass.EXCLUDED_DETERMINISTIC_PHYSICAL.value
                    ):
                        row["classification"] = EpisodeClass.UNCLASSIFIED_FAILURE.value
                        row["terminal_attempt"] = True
                        row["forensic_comparison"] = {
                            "expected_classification": (
                                EpisodeClass.EXCLUDED_DETERMINISTIC_PHYSICAL.value
                            ),
                            "expected_first_failed_sub_gate": ("canonical_terminal_success_gate"),
                            "consistent": False,
                        }
                    elif task_id == "StackCube-v1" and source_episode_id == 938:
                        row["forensic_comparison"] = {
                            "forensic_artifact_manifest_fingerprint": (
                                "sha256:c49558caa9f04d445f8e34acd6e71d17b0339a82c4a5273ac8f8827f35e15a6e"
                            ),
                            "forensic_result_fingerprint": (
                                "sha256:30b7b6226d2762a248edc1ec9d9f68030f38bd1b1e385f00a3534c7544af84de"
                            ),
                            "expected_first_failed_sub_gate": ("canonical_terminal_success_gate"),
                            "consistent": row["first_failed_sub_gate"]
                            == "canonical_terminal_success_gate",
                        }
                    if row["classification"] != EpisodeClass.ACCEPTED_REPLAY.value:
                        row["evidence_reference"] = {
                            "kind": "terminal_instrumented_production_attempt",
                            "relative_path": "work/production_attempts.jsonl",
                            "task_id": task_id,
                            "source_episode_id": source_episode_id,
                            "attempt_index": row["attempt_index"],
                            "forensic_comparison": row.get("forensic_comparison"),
                        }
                    _append_jsonl(progress_path, row)
                    classification = str(row["classification"])
                    if classification == EpisodeClass.UNCLASSIFIED_FAILURE.value:
                        raise Phase2B6V2RuntimeError(
                            f"unclassified production hard stop: {task_id}/{source_episode_id}"
                        )
                    if classification != EpisodeClass.ACCEPTED_REPLAY.value:
                        exclusions[task_id] += 1
                        exclusions["total"] += 1
                        if exclusions[task_id] > 5 or exclusions["total"] > 9:
                            raise Phase2B6V2RuntimeError("exclusion threshold hard stop")
                    elif source_episode_id in contact_ids:
                        rgb = arrays["rgb"]
                        _contact_sheet(
                            production_root
                            / "primary"
                            / "contact_sheets"
                            / f"{task_id}_episode_{source_episode_id:06d}.png",
                            (rgb[0], rgb[len(rgb) // 2], rgb[-1], arrays["terminal_rgb"]),
                        )
                    attempted = sum(
                        row.get("terminal_attempt") is True for row in _read_jsonl(progress_path)
                    )
                    if attempted % 25 == 0:
                        print(
                            json.dumps(
                                {
                                    "phase": "phase2b6_v2_full_production",
                                    "attempted_source_episodes": attempted,
                                    "accepted": attempted - exclusions["total"],
                                    "excluded": exclusions["total"],
                                    "latest_task": task_id,
                                    "latest_source_episode_id": source_episode_id,
                                    "elapsed_seconds": time.perf_counter() - started,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
        finally:
            env.close()
    attempts = _read_jsonl(progress_path)
    threshold = threshold_audit(attempts)
    terminal = [row for row in attempts if row.get("terminal_attempt") is True]
    accepted = accepted_assignments(assignments, terminal)
    split_counts = split_count_report(assignments, accepted)
    excluded = [
        row for row in terminal if row.get("classification") != EpisodeClass.ACCEPTED_REPLAY.value
    ]
    accepted_manifest = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-v2-accepted-episode-manifest-v0",
            "episode_count": len(accepted),
            "frame_count": sum(int(cast(int, row["transition_count"])) for row in accepted),
            "episodes": accepted,
            "passed": threshold["passed"] is True,
        }
    )
    excluded_manifest = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-v2-excluded-episode-manifest-v0",
            "episode_count": len(excluded),
            "episodes": excluded,
            "failure_training_corpus_created": False,
            "passed": all(
                row["classification"]
                in {
                    EpisodeClass.EXCLUDED_DETERMINISTIC_PHYSICAL.value,
                    EpisodeClass.EXCLUDED_SOURCE_CONTRACT.value,
                }
                for row in excluded
            ),
        }
    )
    per_task: dict[str, object] = {}
    for task_id in TASK_IDS:
        rows = [row for row in terminal if row["task_id"] == task_id]
        task_accepted = [
            row for row in rows if row["classification"] == EpisodeClass.ACCEPTED_REPLAY.value
        ]
        task_excluded = [row for row in rows if row not in task_accepted]
        per_task[task_id] = fingerprinted(
            {
                "schema_version": "langmani-v2-phase2b6-v2-task-replay-summary-v0",
                "task_id": task_id,
                "skill_family": CANDIDATE_TASKS[task_id].skill_family,
                "source_count": len(rows),
                "accepted_count": len(task_accepted),
                "excluded_count": len(task_excluded),
                "accepted_frame_count": sum(
                    int(row["source_action_count"]) for row in task_accepted
                ),
                "exclusions": [
                    {
                        "source_episode_id": row["source_episode_id"],
                        "classification": row["classification"],
                        "first_failed_sub_gate": row["first_failed_sub_gate"],
                    }
                    for row in task_excluded
                ],
                "camera": camera_reports[task_id],
                "passed": len(rows) == 1_000 and len(task_excluded) <= 5,
            }
        )
        _write_json(
            evidence_root / f"replay_summary_{_task_slug(task_id)}.json",
            cast(Mapping[str, object], per_task[task_id]),
        )
    sub_gate_summary = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-v2-sub-gate-summary-v0",
            "source_episode_count": len(terminal),
            "gate_failures": {
                name: sum(
                    cast(
                        Mapping[str, object],
                        cast(Mapping[str, object], row["sub_gates"])[name],
                    )["status"]
                    == "failed"
                    for row in terminal
                )
                for name in REQUIRED_SUB_GATES
            },
            "first_failed_sub_gates": dict(
                Counter(
                    str(row["first_failed_sub_gate"])
                    for row in terminal
                    if row["first_failed_sub_gate"] is not None
                )
            ),
            "passed": len(terminal) == EXPECTED_SOURCE_EPISODES,
        }
    )
    frozen_after = inventory_frozen_output(frozen_root)
    frozen_before = _read_json(evidence_root / "frozen_partial_output_inventory_before.json")
    frozen_unchanged = frozen_before.get("fingerprint") == frozen_after.get("fingerprint")
    manifest = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-v2-production-run-manifest-v0",
            "created_at_utc": _timestamp(),
            "producer": marker,
            "source_episode_count": len(terminal),
            "source_transition_count": sum(int(row["source_action_count"]) for row in terminal),
            "accepted_episode_count": len(accepted),
            "accepted_frame_count": accepted_manifest["frame_count"],
            "excluded_episode_count": len(excluded),
            "threshold_audit": threshold,
            "per_task": {
                task_id: {
                    "fingerprint": cast(Mapping[str, object], report)["fingerprint"],
                    "accepted_count": cast(Mapping[str, object], report)["accepted_count"],
                    "excluded_count": cast(Mapping[str, object], report)["excluded_count"],
                }
                for task_id, report in per_task.items()
            },
            "previous_partial_records_reused": False,
            "frozen_v1_root_unchanged": frozen_unchanged,
            "student_policy_training_started": False,
            "optimizer_steps": 0,
            "elapsed_seconds": time.perf_counter() - started,
            "passed": (
                threshold["passed"] is True
                and frozen_unchanged
                and len(terminal) == EXPECTED_SOURCE_EPISODES
                and sum(int(row["source_action_count"]) for row in terminal)
                == EXPECTED_SOURCE_TRANSITIONS
            ),
        }
    )
    documents = {
        "production_run_manifest.json": manifest,
        "accepted_episode_manifest.json": accepted_manifest,
        "excluded_episode_manifest.json": excluded_manifest,
        "explicit_sub_gate_summary.json": sub_gate_summary,
        "primary_split_counts.json": split_counts,
        "frozen_partial_output_inventory_after.json": frozen_after,
        "exclusion_threshold_audit.json": threshold,
    }
    for name, document in documents.items():
        _write_json(evidence_root / name, document)
    _write_json(
        production_root / "primary" / "metadata" / "accepted_episode_manifest.json",
        accepted_manifest,
    )
    _write_json(
        production_root / "primary" / "metadata" / "excluded_episode_manifest.json",
        excluded_manifest,
    )
    _write_json(
        production_root / "primary" / "metadata" / "source_inventory.json",
        _read_json(evidence_root / "source_inventory.json"),
    )
    if manifest["passed"] is not True:
        raise Phase2B6V2RuntimeError("full v2 production aggregate hard stop")
    return manifest


def materialize_accepted_metadata(
    *,
    repo_root: Path,
    production_root: Path,
    evidence_root: Path,
    spec_path: Path,
) -> dict[str, object]:
    """Recompute folds, padding, and balance over accepted episodes only."""

    spec = load_production_spec(spec_path, repo_root=repo_root)
    contract = _materialized_contract(repo_root, spec)
    accepted_document = _read_json(evidence_root / "accepted_episode_manifest.json")
    accepted = cast(list[dict[str, object]], accepted_document["episodes"])
    accepted_split = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-v2-accepted-split-view-v0",
            "source_split_manifest_fingerprint": _read_json(
                evidence_root / "primary_split_manifest.json"
            )["fingerprint"],
            "assignment_count": len(accepted),
            "assignments": accepted,
            "replacement_sampling": False,
            "passed": True,
        }
    )
    folds = build_cross_skill_folds(spec=contract, split_manifest=accepted_split)
    padding = build_padding_audit(accepted)
    balance = build_task_balance_manifest(accepted)
    documents = {
        "accepted_primary_split_manifest.json": accepted_split,
        "cross_skill_fold_manifests.json": folds,
        "padding_audit.json": padding,
        "task_balance_manifest.json": balance,
    }
    for name, document in documents.items():
        _write_json(evidence_root / name, document)
    _write_json(production_root / "work" / "cross_skill_fold_manifests.json", folds)
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-v2-accepted-metadata-stage-v0",
            "accepted_episode_count": len(accepted),
            "accepted_split_fingerprint": accepted_split["fingerprint"],
            "fold_fingerprint": folds["fingerprint"],
            "padding_fingerprint": padding["fingerprint"],
            "task_balance_fingerprint": balance["fingerprint"],
            "passed": all(document["passed"] is True for document in documents.values()),
        }
    )


def verify_exclusion_accounting(*, production_root: Path, evidence_root: Path) -> dict[str, object]:
    """Independently account for accepted and excluded identities."""

    attempts = _read_jsonl(production_root / "work" / "production_attempts.jsonl")
    terminal = [row for row in attempts if row.get("terminal_attempt") is True]
    accepted_manifest = _read_json(evidence_root / "accepted_episode_manifest.json")
    excluded_manifest = _read_json(evidence_root / "excluded_episode_manifest.json")
    accepted = cast(list[dict[str, object]], accepted_manifest["episodes"])
    excluded = cast(list[dict[str, object]], excluded_manifest["episodes"])
    accepted_ids = {str(row["derived_episode_identity"]) for row in accepted}
    excluded_ids = {str(row["derived_episode_identity"]) for row in excluded}
    terminal_ids = {str(row["derived_episode_identity"]) for row in terminal}
    excluded_checks: list[dict[str, object]] = []
    for row in excluded:
        raw_path = row.get("raw_relative_path")
        absent = raw_path is None or not (production_root / str(raw_path)).exists()
        evidence_reference = row.get("evidence_reference")
        excluded_checks.append(
            {
                "task_id": row["task_id"],
                "source_episode_id": row["source_episode_id"],
                "source_trajectory_identity": row["source_trajectory_identity"],
                "source_action_sha256": row["source_action_sha256"],
                "classification": row["classification"],
                "first_failed_sub_gate": row["first_failed_sub_gate"],
                "evidence_reference": evidence_reference,
                "absent_from_policy_raw_root": absent,
                "passed": (
                    row["classification"]
                    in {
                        EpisodeClass.EXCLUDED_DETERMINISTIC_PHYSICAL.value,
                        EpisodeClass.EXCLUDED_SOURCE_CONTRACT.value,
                    }
                    and row["first_failed_sub_gate"] is not None
                    and isinstance(evidence_reference, Mapping)
                    and absent
                ),
            }
        )
    checks = {
        "all_source_identities_accounted": len(terminal) == EXPECTED_SOURCE_EPISODES,
        "terminal_identity_unique": len(terminal_ids) == EXPECTED_SOURCE_EPISODES,
        "accepted_excluded_disjoint": not (accepted_ids & excluded_ids),
        "accepted_plus_excluded_complete": accepted_ids | excluded_ids == terminal_ids,
        "accepted_manifest_count": len(accepted) == accepted_manifest["episode_count"],
        "excluded_manifest_count": len(excluded) == excluded_manifest["episode_count"],
        "every_exclusion_verified": all(row["passed"] is True for row in excluded_checks),
    }
    report = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-v2-source-accounting-verifier-v0",
            "source_episode_count": len(terminal),
            "accepted_episode_count": len(accepted),
            "excluded_episode_count": len(excluded),
            "excluded_episode_checks": excluded_checks,
            "checks": checks,
            "passed": all(checks.values()),
        }
    )
    _write_json(evidence_root / "source_accounting_verification.json", report)
    if report["passed"] is not True:
        raise Phase2B6V2RuntimeError("source accounting hard stop")
    return report


__all__ = [
    "Phase2B6V2RuntimeError",
    "materialize_accepted_metadata",
    "prepare_production",
    "repository_environment_audit",
    "run_full_production",
    "run_rendering_preflight",
    "verify_prior_evidence",
    "verify_exclusion_accounting",
]
