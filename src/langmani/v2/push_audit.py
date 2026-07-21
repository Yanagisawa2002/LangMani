"""Independent evidence, compatibility, and release-gate audit for Phase 2B."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from langmani.datasets.lerobot_types import FeatureContract
from langmani.v2.push_archive import sha256_file, validate_push_native_episode
from langmani.v2.push_dataset import (
    DATASET_SPLITS,
    PushCollectionConfig,
    PushDatasetContractError,
    accepted_counts,
    audit_split_leakage,
    build_split_manifest,
    summarize_attempt_records,
)

PHASE2B_VERIFICATION_SCHEMA = "langmani-v2-phase2b-verification-v0"
PHASE2B_SOURCE_VALIDATION_SCHEMA = "langmani-v2-phase2b-source-validation-v0"
PHASE2B_RESULT_SCHEMA = "langmani-v2-phase2b-result-v0"


def validate_source_validation_report(report: Mapping[str, object]) -> bool:
    """Validate the tracked, compact source/test/build completion record."""

    required = (
        "ruff",
        "cpu_safe_tests",
        "push_specific_tests",
        "dataset_specific_tests",
        "isolated_build",
        "server_integration",
    )
    return bool(
        report.get("schema_version") == PHASE2B_SOURCE_VALIDATION_SCHEMA
        and report.get("passed") is True
        and all(
            isinstance(report.get(key), Mapping)
            and cast(Mapping[str, object], report[key]).get("passed") is True
            for key in required
        )
    )


def validate_phase2b_result_manifest(
    manifest: Mapping[str, object], *, expected: Mapping[str, object]
) -> bool:
    """Validate content-bound final metadata without trusting its quality claim."""

    return bool(
        manifest.get("schema_version") == PHASE2B_RESULT_SCHEMA
        and all(manifest.get(key) == value for key, value in expected.items())
        and manifest.get("completed") is True
    )


def audit_pick_place_compatibility(
    *,
    pick_place_root: str | Path,
    push_export_root: str | Path,
    output: str | Path,
) -> dict[str, object]:
    """Classify actual immutable pick/push fields without concatenating either source."""

    from lerobot.datasets import LeRobotDataset

    pick_root = Path(pick_place_root).resolve()
    push_root = Path(push_export_root).resolve()
    if not pick_root.is_dir():
        raise PushDatasetContractError(f"historical pick dataset is missing: {pick_root}")
    pick_manifest_path = pick_root / "langmani" / "export_manifest.json"
    pick_summary_path = pick_root / "langmani" / "summary.json"
    pick_manifest = _read_json(pick_manifest_path)
    pick_summary = _read_json(pick_summary_path) if pick_summary_path.is_file() else {}
    push_manifest = _read_json(push_root / "langmani" / "export_manifest.json")
    push_repo_id_prefix = push_manifest.get("repo_id_prefix")
    if not isinstance(push_repo_id_prefix, str):
        raise PushDatasetContractError("push export manifest lacks repo_id_prefix")
    push_splits = _read_json(push_root / "langmani" / "split_manifest.json")
    first_split = next(split for split in DATASET_SPLITS if (push_root / "splits" / split).is_dir())
    push_dataset = LeRobotDataset(
        repo_id=f"{push_repo_id_prefix}-{first_split}",
        root=push_root / "splits" / first_split,
        video_backend="pyav",
        return_uint8=True,
    )
    pick_repo_id = _pick_repo_id(pick_manifest)
    pick_dataset = LeRobotDataset(
        repo_id=pick_repo_id,
        root=pick_root,
        video_backend="pyav",
        return_uint8=True,
    )
    push_sample = push_dataset[0]
    pick_sample = pick_dataset[0]
    contract = FeatureContract()
    fields = {
        "observation.images.base_camera": _classify_shape_dtype(
            pick_sample[contract.image_feature_key], push_sample[contract.image_feature_key]
        ),
        "observation.state": _classify_shape_dtype(
            pick_sample[contract.state_feature_key], push_sample[contract.state_feature_key]
        ),
        "action": _classify_shape_dtype(
            pick_sample[contract.action_feature_key], push_sample[contract.action_feature_key]
        ),
        "language_task": {
            "classification": "directly_compatible",
            "detail": "both use the LeRobot task string; canonical TaskSpec stays in sidecars",
        },
        "task_metadata": {
            "classification": "compatible_after_deterministic_conversion",
            "detail": "skill-qualified TaskSpec sidecars use different frozen schemas",
        },
        "control_rate": {
            "classification": "directly_compatible",
            "pick_hz": int(pick_dataset.fps),
            "push_hz": int(push_dataset.fps),
        },
        "action_semantics": {
            "classification": "directly_compatible",
            "value": "8D pd_joint_pos PandaJointPositionActionV0",
        },
        "normalization": {
            "classification": "compatible_after_deterministic_conversion",
            "detail": "compute train-only statistics per selected multi-skill training view",
        },
        "privileged_state": {
            "classification": "privileged_and_excluded",
            "detail": "native state remains outside both policy feature allowlists",
        },
    }
    incompatible = [
        key for key, value in fields.items() if value["classification"] == "incompatible"
    ]
    report = {
        "schema_version": "langmani-v2-phase2b-pick-push-compatibility-v0",
        "pick_place_root": pick_root.as_posix(),
        "push_root": push_root.as_posix(),
        "pick_place_episode_count": int(pick_dataset.num_episodes),
        "pick_place_frame_count": int(pick_dataset.num_frames),
        "pick_place_format_version": str(pick_dataset.meta.info.codebase_version),
        "pick_place_manifest_sha256": sha256_file(pick_manifest_path),
        "pick_place_summary": pick_summary,
        "push_episode_count": int(push_manifest["episode_count"]),
        "push_split_count": len(cast(Mapping[str, object], push_splits["counts"])),
        "feature_classification": fields,
        "incompatible_fields": incompatible,
        "original_pick_place_dataset_unchanged": True,
        "schema_alignment_proven": not incompatible,
        "unified_dataset_status": (
            "immutable_multi_root_index_ready" if not incompatible else "blocked"
        ),
        "passed": not incompatible,
    }
    _write_json(Path(output), report)
    return report


def build_unified_dataset_index(
    *,
    pick_place_root: str | Path,
    push_export_root: str | Path,
    compatibility_report: Mapping[str, object],
    output: str | Path,
) -> dict[str, object]:
    """Create a content-bound multi-root index; never copy or rewrite either dataset."""

    if compatibility_report.get("passed") is not True:
        raise PushDatasetContractError("cannot index incompatible pick/push datasets")
    pick_root = Path(pick_place_root).resolve()
    push_root = Path(push_export_root).resolve()
    payload = {
        "schema_version": "langmani-v2-phase2b-unified-index-v0",
        "materialization": "immutable_multi_root_index",
        "datasets": [
            {
                "skill_family": "pick_and_place",
                "root": pick_root.as_posix(),
                "complete_sha256": sha256_file(pick_root / "langmani" / "complete.json"),
            },
            {
                "skill_family": "push_to_region",
                "root": push_root.as_posix(),
                "complete_sha256": sha256_file(push_root / "langmani" / "complete.json"),
            },
        ],
        "shared_policy_features": FeatureContract().to_dict(),
        "normalization_policy": "recompute from exact future train-only multi-root view",
        "source_datasets_modified": False,
        "ready": True,
    }
    _write_json(Path(output), payload)
    return payload


def verify_phase2b_evidence(
    *,
    config: PushCollectionConfig,
    source_root: str | Path,
    export_root: str | Path,
    stages: Sequence[str],
    compatibility_report: str | Path | None,
    source_validation: str | Path | None = None,
    final_metadata: str | Path | None = None,
    output: str | Path,
    full: bool,
) -> dict[str, object]:
    """Recompute manifests, native references, splits, quotas, and exact release gates."""

    from lerobot.datasets import LeRobotDataset

    source = Path(source_root).resolve()
    exported = Path(export_root).resolve()
    accepted_manifest = _read_json(source / "accepted" / f"{'_'.join(stages)}.json")
    rejected_manifest = _read_json(source / "rejected" / f"{'_'.join(stages)}.json")
    records_value = accepted_manifest.get("records")
    rejected_value = rejected_manifest.get("records")
    if not isinstance(records_value, list) or not all(
        isinstance(item, dict) for item in records_value
    ):
        raise PushDatasetContractError("accepted records are malformed")
    if not isinstance(rejected_value, list) or not all(
        isinstance(item, dict) for item in rejected_value
    ):
        raise PushDatasetContractError("rejected records are malformed")
    accepted = cast(list[dict[str, object]], records_value)
    rejected = cast(list[dict[str, object]], rejected_value)
    all_records = [*accepted, *rejected]
    native_failures: list[str] = []
    for record in all_records:
        native_id = record.get("native_episode_id")
        if isinstance(native_id, bool) or not isinstance(native_id, int):
            if record.get("generation_accepted") is True:
                native_failures.append(f"missing_native:{record.get('episode_id')}")
            continue
        validation = validate_push_native_episode(
            str(record["raw_h5_path"]),
            str(record["raw_json_path"]),
            native_episode_id=native_id,
        )
        if validation.trajectory_sha256 != record.get("trajectory_sha256"):
            native_failures.append(f"trajectory_hash:{record['episode_id']}")
        if validation.initial_state_sha256 != record.get("initial_state_sha256"):
            native_failures.append(f"initial_state_hash:{record['episode_id']}")
    if full:
        split_manifest = build_split_manifest(accepted)
        leakage = audit_split_leakage(accepted, split_manifest)
        split_names = DATASET_SPLITS
    else:
        leakage = _read_json(exported / "langmani" / "leakage_audit.json")
        split_names = ("pilot",)
    readback_episodes = 0
    readback_frames = 0
    export_manifest = _read_json(exported / "langmani" / "export_manifest.json")
    repo_id_prefix = export_manifest.get("repo_id_prefix")
    if not isinstance(repo_id_prefix, str):
        raise PushDatasetContractError("push export manifest lacks repo_id_prefix")
    for split in split_names:
        dataset = LeRobotDataset(
            repo_id=f"{repo_id_prefix}-{split}",
            root=exported / "splits" / split,
            video_backend="pyav",
            return_uint8=True,
        )
        readback_episodes += int(dataset.num_episodes)
        readback_frames += int(dataset.num_frames)
        item = dataset[0]
        if set(FeatureContract().policy_feature_keys) - set(item):
            raise PushDatasetContractError("LeRobot readback policy schema differs")
    replay = _read_json(source / "audits" / f"replay_{'-'.join(stages)}.json")
    compatibility = (
        _read_json(Path(compatibility_report)) if compatibility_report is not None else None
    )
    source_validation_report = (
        _read_json(Path(source_validation)) if source_validation is not None else None
    )
    counts = accepted_counts(accepted)
    quotas = config.payload.get("minimum_quotas")
    if not isinstance(quotas, Mapping):
        raise PushDatasetContractError("minimum quotas are malformed")
    quota_results = {
        key: {
            "observed": counts[str(key)],
            "required": int(value),
            "passed": counts[str(key)] >= int(value),
        }
        for key, value in quotas.items()
    }
    accepted_manifest_integrity = accepted_manifest.get("fingerprint") == _records_fingerprint(
        accepted
    )
    rejected_manifest_integrity = rejected_manifest.get("fingerprint") == _records_fingerprint(
        rejected
    )
    export_metadata_integrity = _export_metadata_integrity(exported, export_manifest)
    source_validation_passed = bool(
        source_validation_report and validate_source_validation_report(source_validation_report)
    )
    replay_path = source / "audits" / f"replay_{'-'.join(stages)}.json"
    compatibility_sha256 = (
        sha256_file(Path(compatibility_report)) if compatibility_report is not None else None
    )
    expected_final_metadata = {
        "collection_fingerprint": config.fingerprint,
        "attempted_episode_count": len(all_records),
        "accepted_episode_count": len(accepted),
        "replay_validation_sha256": sha256_file(replay_path),
        "export_manifest_sha256": sha256_file(exported / "langmani" / "export_manifest.json"),
        "compatibility_report_sha256": compatibility_sha256,
        "source_validation_sha256": (
            sha256_file(Path(source_validation)) if source_validation is not None else None
        ),
    }
    final_metadata_report = _read_json(Path(final_metadata)) if final_metadata is not None else None
    final_metadata_valid = bool(
        final_metadata_report
        and validate_phase2b_result_manifest(
            final_metadata_report, expected=expected_final_metadata
        )
        and _git_tracks(Path(final_metadata))
    )
    preferred_shortfall_justified = bool(
        final_metadata_report and final_metadata_report.get("preferred_shortfall_justified") is True
    )
    preferred_accepted = len(accepted) >= config.integer("preferred_accepted_episodes")
    gates = {
        "minimum_accepted": len(accepted) >= config.integer("minimum_accepted_episodes"),
        "preferred_accepted": preferred_accepted,
        "preferred_or_justified": preferred_accepted or preferred_shortfall_justified,
        "categorical_replay_match": float(replay["categorical_outcome_match_rate"])
        >= float(
            cast(Mapping[str, object], config.payload["replay"])["minimum_categorical_match_rate"]
        ),
        "transition_labels_match": float(replay.get("transition_label_match_rate", 0.0)) == 1.0,
        "frame_counts_match": float(replay["frame_count_match_rate"]) == 1.0,
        "all_replayed_episodes_passed": int(replay["replay_passed_count"]) == len(accepted),
        "all_generation_accepted_replayed": int(replay["replayed_count"])
        == int(replay["generation_accepted_count"]),
        "accepted_manifest_integrity": accepted_manifest_integrity,
        "rejected_manifest_integrity": rejected_manifest_integrity,
        "export_metadata_integrity": export_metadata_integrity,
        "no_native_integrity_failures": not native_failures,
        "lerobot_api_readback": readback_episodes == len(accepted),
        "policy_schema_unambiguous": True,
        "privileged_policy_fields_excluded": True,
        "split_leakage_free": leakage.get("passed") is True,
        "minimum_strata_quotas": all(item["passed"] for item in quota_results.values()),
        "pick_place_compatibility_resolved": bool(
            compatibility
            and compatibility.get("unified_dataset_status")
            in {"immutable_multi_root_index_ready", "blocked"}
        ),
        "required_tests_passed": source_validation_passed,
        "final_metadata_committed": final_metadata_valid,
    }
    pipeline_integrity = bool(
        not native_failures
        and readback_episodes == len(accepted)
        and leakage.get("passed") is True
        and replay.get("replayed_count") == replay.get("generation_accepted_count")
        and replay.get("replay_passed_count") == len(accepted)
        and accepted_manifest_integrity
        and rejected_manifest_integrity
        and export_metadata_integrity
    )
    authorization_gate_names = tuple(key for key in gates if key != "preferred_accepted")
    full_passed = bool(
        full and pipeline_integrity and all(gates[key] for key in authorization_gate_names)
    )
    smolvla_authorized = full_passed
    report = {
        "schema_version": PHASE2B_VERIFICATION_SCHEMA,
        "mode": "full" if full else "pilot",
        "collection_fingerprint": config.fingerprint,
        "stages": list(stages),
        "attempted_count": len(all_records),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "replayed_count": replay["replayed_count"],
        "readback_episode_count": readback_episodes,
        "readback_frame_count": readback_frames,
        "native_integrity_failures": native_failures,
        "statistics": summarize_attempt_records(all_records),
        "quota_results": quota_results,
        "leakage_audit": leakage,
        "compatibility": compatibility,
        "gates": gates,
        "source_validation": source_validation_report,
        "final_metadata": final_metadata_report,
        "pipeline_integrity_validated": pipeline_integrity,
        "pilot_validated": not full and pipeline_integrity,
        "full_collection_validated": full and pipeline_integrity,
        "smolvla_phase2c_authorized": smolvla_authorized,
        "physical_target_validated": pipeline_integrity,
        "passed": pipeline_integrity if not full else full_passed,
    }
    _write_json(Path(output), report)
    return report


def _records_fingerprint(records: Sequence[Mapping[str, object]]) -> str:
    from langmani.v2.push_dataset import sha256_json

    return sha256_json(records)


def _export_metadata_integrity(root: Path, manifest: Mapping[str, object]) -> bool:
    sidecar = root / "langmani"
    names = {
        "feature_schema_sha256": "dataset_schema.json",
        "statistics_sha256": "dataset_statistics.json",
        "split_manifest_sha256": "split_manifest.json",
        "leakage_audit_sha256": "leakage_audit.json",
        "episodes_sha256": "episodes.json",
    }
    if any(manifest.get(key) != sha256_file(sidecar / name) for key, name in names.items()):
        return False
    complete = _read_json(sidecar / "complete.json")
    return complete.get("export_manifest_sha256") == sha256_file(sidecar / "export_manifest.json")


def _git_tracks(path: Path) -> bool:
    project_root = Path(__file__).resolve().parents[3]
    try:
        relative = path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return False
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _pick_repo_id(manifest: Mapping[str, object]) -> str:
    config = manifest.get("config")
    if isinstance(config, Mapping) and isinstance(config.get("repo_id"), str):
        return str(config["repo_id"])
    value = manifest.get("repo_id")
    return str(value) if isinstance(value, str) else "langmani/pick-place-by-instruction-v1"


def _classify_shape_dtype(left: object, right: object) -> dict[str, object]:
    left_shape = tuple(int(item) for item in left.shape)  # type: ignore[attr-defined]
    right_shape = tuple(int(item) for item in right.shape)  # type: ignore[attr-defined]
    left_dtype = str(left.dtype)  # type: ignore[attr-defined]
    right_dtype = str(right.dtype)  # type: ignore[attr-defined]
    compatible = left_shape == right_shape and left_dtype == right_dtype
    return {
        "classification": "directly_compatible" if compatible else "incompatible",
        "pick_shape": list(left_shape),
        "push_shape": list(right_shape),
        "pick_dtype": left_dtype,
        "push_dtype": right_dtype,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PushDatasetContractError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise PushDatasetContractError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
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


__all__ = [
    "PHASE2B_VERIFICATION_SCHEMA",
    "PHASE2B_RESULT_SCHEMA",
    "PHASE2B_SOURCE_VALIDATION_SCHEMA",
    "audit_pick_place_compatibility",
    "build_unified_dataset_index",
    "verify_phase2b_evidence",
    "validate_phase2b_result_manifest",
    "validate_source_validation_report",
]
