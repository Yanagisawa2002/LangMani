"""Fresh-environment action replay for Phase 2B pushing trajectories."""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from langmani.environments.push_specs import PushTaskSpec
from langmani.environments.push_to_region import ENV_ID
from langmani.v2.push_archive import (
    load_native_actions,
    load_native_transition_labels,
    state_mapping_sha256,
)
from langmani.v2.push_collection import load_attempt_records
from langmani.v2.push_dataset import (
    REPLAY_SCHEMA_VERSION,
    PushCollectionConfig,
    PushDatasetContractError,
    sha256_json,
)

_CATEGORICAL_KEYS = (
    "success",
    "fail",
    "wrong_object_contact",
    "wrong_object_displaced",
    "target_outside_workspace",
    "target_lifted",
    "target_toppled",
    "invalid_action",
    "action_out_of_bounds",
)


def replay_push_attempts(
    *,
    config: PushCollectionConfig,
    output_root: str | Path,
    stages: Sequence[str],
    sim_backend: str = "physx_cuda",
) -> dict[str, object]:
    """Replay every generation-accepted trajectory without constructing an expert."""

    if sim_backend != "physx_cuda":
        raise PushDatasetContractError("Phase 2B real replay requires physx_cuda")
    root = Path(output_root).resolve()
    key = "-".join(stages)
    output = root / "audits" / f"replay_{key}.json"
    if output.exists():
        return _read_json(output)
    records = load_attempt_records(root, stages)
    candidates = [record for record in records if record.get("generation_accepted") is True]
    results: list[dict[str, object]] = []
    for index, record in enumerate(candidates):
        results.append(
            _replay_one(
                record,
                config=config,
                sim_backend=sim_backend,
                replay_index=index,
            )
        )
    by_episode = {str(item["episode_id"]): item for item in results}
    accepted_records: list[dict[str, object]] = []
    rejected_records: list[dict[str, object]] = []
    failure_counts: Counter[str] = Counter()
    for record in records:
        updated = dict(record)
        replay = by_episode.get(str(record["episode_id"]))
        updated["replay"] = replay
        updated["replay_validated"] = bool(replay and replay["passed"] is True)
        updated["accepted"] = bool(
            updated.get("generation_accepted") is True and updated["replay_validated"] is True
        )
        updated["final_record_fingerprint"] = sha256_json(updated)
        if updated["accepted"] is True:
            accepted_records.append(updated)
        else:
            rejected_records.append(updated)
            if updated.get("generation_accepted") is not True:
                failures = updated.get("generation_acceptance_failures")
                if isinstance(failures, list) and failures:
                    failure_counts.update(str(item) for item in failures)
                else:
                    failure_counts["generation_rejected_unclassified"] += 1
            elif replay is not None:
                reasons = replay.get("failure_reasons")
                if isinstance(reasons, list) and reasons:
                    failure_counts.update(str(item) for item in reasons)
                else:
                    failure_counts["replay_rejected_unclassified"] += 1
    categorical_matches = sum(item["categorical_outcome_match"] is True for item in results)
    transition_label_matches = sum(item["transition_label_match"] is True for item in results)
    report = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "collection_fingerprint": config.fingerprint,
        "stages": list(stages),
        "attempt_count": len(records),
        "generation_accepted_count": len(candidates),
        "replayed_count": len(results),
        "replay_passed_count": len(accepted_records),
        "replay_failed_count": len(results) - len(accepted_records),
        "categorical_outcome_match_count": categorical_matches,
        "categorical_outcome_match_rate": (categorical_matches / len(results) if results else 0.0),
        "transition_label_match_count": transition_label_matches,
        "transition_label_match_rate": (
            transition_label_matches / len(results) if results else 0.0
        ),
        "frame_count_match_rate": (
            sum(item["frame_count_match"] is True for item in results) / len(results)
            if results
            else 0.0
        ),
        "terminal_position_error": _distribution(
            [float(item["terminal_position_error"]) for item in results]
        ),
        "terminal_orientation_error_radians": _distribution(
            [float(item["terminal_orientation_error_radians"]) for item in results]
        ),
        "failure_categories": dict(sorted(failure_counts.items())),
        "results": results,
        "accepted_record_fingerprints": [
            item["final_record_fingerprint"] for item in accepted_records
        ],
        "passed": bool(results) and len(accepted_records) == len(candidates),
    }
    _write_new_json(output, report)
    accepted_manifest = {
        "schema_version": "langmani-v2-phase2b-accepted-v0",
        "stages": list(stages),
        "records": accepted_records,
        "count": len(accepted_records),
        "fingerprint": sha256_json(accepted_records),
    }
    rejected_manifest = {
        "schema_version": "langmani-v2-phase2b-rejected-v0",
        "stages": list(stages),
        "records": rejected_records,
        "count": len(rejected_records),
        "failure_categories": dict(sorted(failure_counts.items())),
        "fingerprint": sha256_json(rejected_records),
    }
    suffix = key.replace("-", "_")
    _write_new_json(root / "accepted" / f"{suffix}.json", accepted_manifest)
    _write_new_json(root / "rejected" / f"{suffix}.json", rejected_manifest)
    _write_new_json(
        root / "rejected" / f"failure_corpus_{suffix}.json",
        {
            "schema_version": "langmani-v2-phase2b-failure-corpus-v0",
            "retention_policy": "all failed native trajectories and compact expert diagnostics",
            "records": rejected_records,
            "count": len(rejected_records),
            "fingerprint": sha256_json(rejected_records),
        },
    )
    return report


def _replay_one(
    record: Mapping[str, object],
    *,
    config: PushCollectionConfig,
    sim_backend: str,
    replay_index: int,
) -> dict[str, object]:
    import gymnasium as gym

    import langmani.environments  # noqa: F401

    started = time.perf_counter()
    task_value = record.get("task_spec")
    if not isinstance(task_value, Mapping):
        raise PushDatasetContractError("attempt record task_spec is malformed")
    task = PushTaskSpec.from_mapping(task_value)
    seed = record.get("seed")
    native_id = record.get("native_episode_id")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise PushDatasetContractError("attempt seed is malformed")
    if isinstance(native_id, bool) or not isinstance(native_id, int):
        raise PushDatasetContractError("accepted attempt lacks a native episode ID")
    h5_path = Path(str(record["raw_h5_path"]))
    actions = load_native_actions(h5_path, native_episode_id=native_id)
    recorded_labels = load_native_transition_labels(h5_path, native_episode_id=native_id)
    environment: Any = gym.make(
        ENV_ID,
        num_envs=1,
        obs_mode="state_dict",
        reward_mode="none",
        control_mode="pd_joint_pos",
        render_mode=None,
        sim_backend=sim_backend,
    )
    final_info: Mapping[str, object] = {}
    executed = 0
    replayed_labels: dict[str, list[bool]] = {
        "terminated": [],
        "truncated": [],
        "success": [],
        "fail": [],
    }
    try:
        environment.reset(seed=seed, options={"task_spec": task.to_dict()})
        base = environment.unwrapped
        initial_hash = state_mapping_sha256(base.get_state_dict())
        for action in actions:
            _observation, _reward, terminated, truncated, info = environment.step(action)
            final_info = info
            executed += 1
            replayed_labels["terminated"].append(_single_bool(terminated))
            replayed_labels["truncated"].append(_single_bool(truncated))
            json_info = _json_tensor_mapping(info)
            replayed_labels["success"].append(bool(json_info.get("success", False)))
            replayed_labels["fail"].append(bool(json_info.get("fail", False)))
        evaluation = _json_tensor_mapping(base.get_push_expert_evaluation())
        final_pose = _target_pose(base, task)
    finally:
        environment.close()

    recorded_evaluation = record.get("final_evaluation")
    if not isinstance(recorded_evaluation, Mapping):
        raise PushDatasetContractError("attempt final evaluation is malformed")
    categorical_mismatches = [
        key
        for key in _CATEGORICAL_KEYS
        if bool(evaluation.get(key, False)) != bool(recorded_evaluation.get(key, False))
    ]
    recorded_pose = np.asarray(record.get("final_target_pose"), dtype=np.float64)
    replay_pose = np.asarray(final_pose, dtype=np.float64)
    if recorded_pose.shape != (7,) or replay_pose.shape != (7,):
        raise PushDatasetContractError("recorded or replayed target pose is malformed")
    position_error = float(np.linalg.norm(recorded_pose[:3] - replay_pose[:3]))
    orientation_error = _quaternion_angle(recorded_pose[3:], replay_pose[3:])
    replay_config = config.payload.get("replay")
    if not isinstance(replay_config, Mapping):
        raise PushDatasetContractError("replay config is malformed")
    position_limit = float(replay_config["maximum_terminal_position_error"])
    orientation_limit = float(replay_config["maximum_terminal_orientation_error_radians"])
    failure_reasons: list[str] = []
    if initial_hash != record.get("initial_state_sha256"):
        failure_reasons.append("initial_state_mismatch")
    if executed != len(actions):
        failure_reasons.append("trajectory_length_mismatch")
    label_mismatches = [
        key
        for key in recorded_labels
        if not np.array_equal(
            recorded_labels[key], np.asarray(replayed_labels[key], dtype=np.bool_)
        )
    ]
    if label_mismatches:
        failure_reasons.append("transition_label_mismatch")
    if categorical_mismatches:
        failure_reasons.append("categorical_outcome_mismatch")
    if evaluation.get("success") is not True:
        failure_reasons.append("replay_success_false")
    if position_error > position_limit:
        failure_reasons.append("terminal_position_drift")
    if orientation_error > orientation_limit:
        failure_reasons.append("terminal_orientation_drift")
    if final_info and bool(_json_tensor_mapping(final_info).get("action_out_of_bounds", False)):
        failure_reasons.append("replay_action_out_of_bounds")
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "replay_index": replay_index,
        "episode_id": record["episode_id"],
        "seed": seed,
        "task_spec": task.to_dict(),
        "expert_constructed": False,
        "expert_invoked": False,
        "action_count": len(actions),
        "executed_action_count": executed,
        "frame_count_match": executed == len(actions),
        "transition_label_match": not label_mismatches,
        "transition_label_mismatches": label_mismatches,
        "initial_state_sha256": initial_hash,
        "initial_state_match": initial_hash == record.get("initial_state_sha256"),
        "categorical_outcome_match": not categorical_mismatches,
        "categorical_mismatches": categorical_mismatches,
        "terminal_position_error": position_error,
        "terminal_orientation_error_radians": orientation_error,
        "recorded_final_evaluation": dict(recorded_evaluation),
        "replayed_final_evaluation": evaluation,
        "failure_reasons": failure_reasons,
        "elapsed_seconds": time.perf_counter() - started,
        "passed": not failure_reasons,
    }


def _target_pose(base: object, task: PushTaskSpec) -> list[float]:
    actor = base.push_objects[("blue_cube", "orange_cylinder").index(task.target_object_id)]
    array = np.asarray(actor.pose.raw_pose.detach().cpu().numpy(), dtype=np.float64)
    if array.shape != (1, 7) or not np.all(np.isfinite(array)):
        raise PushDatasetContractError("replay target pose is malformed")
    return [float(item) for item in array[0]]


def _quaternion_angle(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = left / max(float(np.linalg.norm(left)), 1e-12)
    right_norm = right / max(float(np.linalg.norm(right)), 1e-12)
    cosine = min(1.0, abs(float(np.dot(left_norm, right_norm))))
    return float(2.0 * math.acos(cosine))


def _single_bool(value: object) -> bool:
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    array = np.asarray(candidate)
    if array.size != 1:
        raise PushDatasetContractError("terminal flag must contain one value")
    return bool(array.reshape(-1)[0])


def _json_tensor_mapping(value: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, item in value.items():
        candidate = item
        detach = getattr(candidate, "detach", None)
        if callable(detach):
            candidate = detach()
        cpu = getattr(candidate, "cpu", None)
        if callable(cpu):
            candidate = cpu()
        numpy = getattr(candidate, "numpy", None)
        if callable(numpy):
            candidate = numpy()
        array = np.asarray(candidate)
        if array.size != 1:
            continue
        scalar = array.reshape(-1)[0]
        result[key] = bool(scalar) if array.dtype.kind == "b" else float(scalar)
    return result


def _distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PushDatasetContractError(f"JSON root must be an object: {path}")
    return value


def _write_new_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


__all__ = ["replay_push_attempts"]
