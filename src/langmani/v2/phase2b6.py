"""Fail-closed contracts for LangMani 2.0 Phase 2B.6.

Phase 2B.6 is a dataset-production milestone.  This module deliberately has no
policy, checkpoint, optimizer, backward, or learned-control dependency.  It
defines immutable source/episode identities, deterministic language and split
materialization, explicit action padding, leakage checks, and the final
eligibility-only gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt

from langmani.v2.phase2b5 import (
    CANDIDATE_TASKS,
    canonical_json_sha256,
    sha256_file,
)

SOURCE_BRANCH: Final = "codex/langmani-v2-phase2b5-official-demos"
SOURCE_COMMIT: Final = "8c81058008bb03c4611c9133adf4112e2a21287f"
TARGET_BRANCH: Final = "codex/langmani-v2-phase2b6-dataset-production"
DATASET_PACKAGE_ID: Final = "LangManiOfficialMultiSkill-v1"
SPEC_SCHEMA: Final = "langmani-v2-phase2b6-production-spec-v0"
TASK_IDS: Final = ("PickCube-v1", "StackCube-v1", "PushCube-v1")
PRIMARY_SPLITS: Final = (
    "train",
    "validation",
    "test_unseen_reset",
    "test_unseen_task_language",
    "test_visual_shift",
)
EXPECTED_EPISODES: Final = 3_000
EXPECTED_TRANSITIONS: Final = 254_374
ACTION_HORIZONS: Final = (10, 16, 50)
PHASE2B5_PACKAGE_SHA256: Final = "22e93d609fc4564d0bdb3e5fc7571cf8971c8a86b3cf3bdaa910108621bcbcc2"
PHASE2B5_ARTIFACT_MANIFEST_SHA256: Final = (
    "622da6018d12a5e561813543ba4040a33d456dcec2dc830ecd63edc60c0b22cc"
)


class Phase2B6ContractError(ValueError):
    """Raised when a Phase 2B.6 input or result violates a frozen contract."""


class Phase2B6Result(StrEnum):
    """The only terminal result classifications."""

    RESULT_A = "RESULT_A"
    RESULT_B = "RESULT_B"
    RESULT_C = "RESULT_C"
    RESULT_D = "RESULT_D"


@dataclass(frozen=True, slots=True)
class ActionChunk:
    """One explicitly padded action target with a non-implicit mask."""

    action: npt.NDArray[np.float32]
    action_is_pad: npt.NDArray[np.bool_]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2B6ContractError(f"failed to read JSON-compatible document {path}") from error
    if not isinstance(value, dict):
        raise Phase2B6ContractError(f"{path} must contain one object")
    return value


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical_identity(payload: Mapping[str, object]) -> str:
    return canonical_json_sha256(payload)


def fingerprinted(payload: Mapping[str, object]) -> dict[str, object]:
    """Return a copy with a canonical content fingerprint."""

    result = dict(payload)
    if "fingerprint" in result:
        raise Phase2B6ContractError("fingerprinted payload cannot already contain fingerprint")
    result["fingerprint"] = _canonical_identity(result)
    return result


def load_production_spec(path: str | Path) -> dict[str, Any]:
    """Load the frozen JSON-compatible YAML and enforce material invariants."""

    spec_path = Path(path).resolve()
    try:
        text = spec_path.read_text(encoding="utf-8")
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            loaded = json.loads(re.sub(r",(?=\s*[}\]])", "", text))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2B6ContractError(f"failed to read JSON-compatible YAML {spec_path}") from error
    if not isinstance(loaded, dict):
        raise Phase2B6ContractError("production specification must contain one object")
    spec = loaded
    if spec.get("schema_version") != SPEC_SCHEMA:
        raise Phase2B6ContractError("production specification schema changed")
    if spec.get("derived_dataset_version") != DATASET_PACKAGE_ID:
        raise Phase2B6ContractError("derived dataset identity changed")
    authorization = spec.get("source_authorization")
    if not isinstance(authorization, Mapping):
        raise Phase2B6ContractError("source authorization is malformed")
    if authorization.get("source_branch") != SOURCE_BRANCH:
        raise Phase2B6ContractError("source branch changed")
    if authorization.get("source_commit") != SOURCE_COMMIT:
        raise Phase2B6ContractError("source commit changed")
    tasks = spec.get("tasks")
    if not isinstance(tasks, Mapping) or tuple(tasks) != TASK_IDS:
        raise Phase2B6ContractError("selected official tasks or order changed")
    totals = spec.get("expected_totals")
    if not isinstance(totals, Mapping):
        raise Phase2B6ContractError("expected totals are malformed")
    if totals.get("episodes") != EXPECTED_EPISODES:
        raise Phase2B6ContractError("expected episode count changed")
    if totals.get("transitions") != EXPECTED_TRANSITIONS:
        raise Phase2B6ContractError("expected transition count changed")
    split_spec = spec.get("splits")
    if not isinstance(split_spec, Mapping):
        raise Phase2B6ContractError("split policy is malformed")
    per_task = split_spec.get("per_task_counts")
    if not isinstance(per_task, Mapping) or tuple(per_task) != PRIMARY_SPLITS:
        raise Phase2B6ContractError("primary split names changed")
    if sum(_positive_int(per_task[name], f"split {name}") for name in PRIMARY_SPLITS) != 1_000:
        raise Phase2B6ContractError("per-task split counts must total 1,000")
    padding = spec.get("padding")
    if not isinstance(padding, Mapping) or tuple(padding.get("candidate_horizons", ())) != (
        ACTION_HORIZONS
    ):
        raise Phase2B6ContractError("action padding horizons changed")
    phase_authorization = spec.get("authorization")
    if not isinstance(phase_authorization, Mapping) or any(
        phase_authorization.get(key) is not False
        for key in (
            "training_permitted_in_phase2b6",
            "optimizer_creation_permitted",
            "backward_pass_permitted",
            "custom_push_expert_route_active",
        )
    ):
        raise Phase2B6ContractError("production specification opened a prohibited path")
    spec["_spec_path"] = spec_path.as_posix()
    spec["_spec_sha256"] = "sha256:" + sha256_file(spec_path)
    return spec


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Phase2B6ContractError(f"{label} must be a positive integer")
    return value


def enforce_starting_point(*, parent_commit: str, branch: str) -> None:
    """Require the sole authorized Phase 2B.5 source point."""

    if parent_commit != SOURCE_COMMIT:
        raise Phase2B6ContractError(
            f"Phase 2B.6 must start from {SOURCE_COMMIT}, got {parent_commit}"
        )
    if branch != TARGET_BRANCH:
        raise Phase2B6ContractError(f"Phase 2B.6 must run on {TARGET_BRANCH}, got {branch}")


def verify_phase2b5_package(
    *, artifact_root: str | Path, spec: Mapping[str, object]
) -> dict[str, object]:
    """Rehash every immutable Phase 2B.5 artifact and enforce its accepted package."""

    root = Path(artifact_root).resolve()
    authorization = cast(Mapping[str, object], spec["source_authorization"])
    package_path = root / "accepted_official_demo_source_package.json"
    manifest_path = root / "artifact_manifest.json"
    expected_package_hash = str(authorization["accepted_package_sha256"])
    expected_manifest_hash = str(authorization["artifact_manifest_sha256"])
    package_hash = _sha256_bytes(package_path.read_bytes().replace(b"\r\n", b"\n"))
    manifest_hash = _sha256_bytes(manifest_path.read_bytes().replace(b"\r\n", b"\n"))
    manifest = _read_json(manifest_path)
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise Phase2B6ContractError("Phase 2B.5 artifact manifest is malformed")
    hash_checks: dict[str, bool] = {}
    size_checks: dict[str, bool] = {}
    for raw_entry in entries:
        if not isinstance(raw_entry, Mapping):
            raise Phase2B6ContractError("Phase 2B.5 artifact entry is malformed")
        relative = str(raw_entry.get("path"))
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 1:
            raise Phase2B6ContractError(f"unsafe Phase 2B.5 artifact path {relative!r}")
        path = root / pure.name
        if path.is_symlink() or not path.is_file():
            raise Phase2B6ContractError(f"missing Phase 2B.5 artifact {relative!r}")
        normalized = path.read_bytes().replace(b"\r\n", b"\n")
        hash_checks[relative] = _sha256_bytes(normalized) == raw_entry.get("sha256")
        size_checks[relative] = len(normalized) == raw_entry.get("size_bytes")
    package = _read_json(package_path)
    selected = package.get("selected_task_ids")
    families = package.get("skill_families")
    gates = package.get("gates")
    checks = {
        "accepted_package_file_hash": package_hash == expected_package_hash,
        "artifact_manifest_file_hash": manifest_hash == expected_manifest_hash,
        "all_artifact_hashes": all(hash_checks.values()),
        "all_artifact_sizes": all(size_checks.values()),
        "accepted_package_schema": (
            package.get("schema_version")
            == "langmani-v2-phase2b5-accepted-official-source-package-v0"
        ),
        "selected_tasks": selected == list(TASK_IDS),
        "three_skill_families": isinstance(families, list) and len(set(families)) == 3,
        "all_phase2b5_gates": (
            isinstance(gates, Mapping)
            and bool(gates)
            and all(value is True for value in gates.values())
        ),
        "contains_no_trained_model": package.get("contains_trained_model") is False,
    }
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-phase2b5-source-verification-v0",
            "accepted_package_sha256": package_hash,
            "artifact_manifest_sha256": manifest_hash,
            "artifact_file_count": len(entries),
            "artifact_hash_checks": hash_checks,
            "artifact_size_checks": size_checks,
            "checks": checks,
            "passed": all(checks.values()),
        }
    )


def source_trajectory_identity(
    *,
    spec: Mapping[str, object],
    task_id: str,
    source_episode_id: int,
    reset_identity: str,
    action_sha256: str,
) -> str:
    """Bind one official source trajectory to its immutable byte lineage."""

    if task_id not in TASK_IDS:
        raise Phase2B6ContractError(f"unknown selected task {task_id!r}")
    if isinstance(source_episode_id, bool) or not isinstance(source_episode_id, int):
        raise Phase2B6ContractError("source episode ID must be an integer")
    task = cast(Mapping[str, object], cast(Mapping[str, object], spec["tasks"])[task_id])
    authorization = cast(Mapping[str, object], spec["source_authorization"])
    return _canonical_identity(
        {
            "schema_version": "langmani-v2-phase2b6-source-trajectory-identity-v0",
            "official_source_revision": authorization["official_source_revision"],
            "source_zip_sha256": task["source_zip_sha256"],
            "source_task_id": task_id,
            "source_trajectory_id": source_episode_id,
            "reset_identity": reset_identity,
            "action_sequence_sha256": action_sha256,
        }
    )


def source_reset_identity(
    *,
    task_id: str,
    source_episode_id: int,
    reset_kwargs_sha256: str,
) -> str:
    """Bind reset kwargs to the task-local episode that gives them meaning.

    The official sources expose reset kwargs for each episode, but no
    cross-task reset-family identifier.  Identical kwargs in two different
    environments therefore do not identify the same physical reset.
    """

    if task_id not in TASK_IDS:
        raise Phase2B6ContractError(f"unknown selected task {task_id!r}")
    if isinstance(source_episode_id, bool) or not isinstance(source_episode_id, int):
        raise Phase2B6ContractError("source episode ID must be an integer")
    if not reset_kwargs_sha256.startswith("sha256:"):
        raise Phase2B6ContractError("reset kwargs identity must be SHA-256 bound")
    return _canonical_identity(
        {
            "schema_version": "langmani-v2-phase2b6-source-reset-identity-v0",
            "source_task_id": task_id,
            "source_trajectory_id": source_episode_id,
            "reset_kwargs_sha256": reset_kwargs_sha256,
        }
    )


def derived_episode_identity(
    *,
    spec: Mapping[str, object],
    source_identity: str,
    instruction_template_id: str,
    primary_split: str,
) -> str:
    """Bind replay/camera/language/split semantics to one source identity."""

    if primary_split not in PRIMARY_SPLITS:
        raise Phase2B6ContractError(f"unknown primary split {primary_split!r}")
    runtime = cast(Mapping[str, object], spec["runtime"])
    return _canonical_identity(
        {
            "schema_version": "langmani-v2-phase2b6-derived-episode-identity-v0",
            "source_trajectory_identity": source_identity,
            "replay_configuration_fingerprint": _canonical_identity(
                cast(Mapping[str, object], runtime["replay_environment"])
            ),
            "camera_configuration_fingerprint": _canonical_identity(
                cast(Mapping[str, object], spec["camera"])
            ),
            "instruction_template_id": instruction_template_id,
            "primary_split": primary_split,
            "derived_dataset_version": spec["derived_dataset_version"],
        }
    )


def build_language_template_manifest(spec: Mapping[str, object]) -> dict[str, object]:
    """Validate disjoint, bounded, nonempty language banks for all tasks."""

    language = cast(Mapping[str, object], spec["language_templates"])
    task_reports: dict[str, object] = {}
    all_texts: dict[str, tuple[str, str]] = {}
    for task_id in TASK_IDS:
        banks = cast(Mapping[str, object], language[task_id])
        bank_reports: dict[str, object] = {}
        for bank_name, minimum in (("train", 4), ("validation", 1), ("held_out", 3)):
            raw_templates = banks.get(bank_name)
            if not isinstance(raw_templates, list) or not all(
                isinstance(item, str) and item.strip() for item in raw_templates
            ):
                raise Phase2B6ContractError(f"{task_id} {bank_name} templates are malformed")
            templates = cast(list[str], raw_templates)
            if len(templates) < minimum or len(templates) != len(set(templates)):
                raise Phase2B6ContractError(f"{task_id} {bank_name} template count is invalid")
            entries: list[dict[str, object]] = []
            for index, text in enumerate(templates):
                normalized = " ".join(text.lower().split())
                if normalized in all_texts:
                    previous = all_texts[normalized]
                    raise Phase2B6ContractError(
                        f"language template overlaps {previous[0]}/{previous[1]}"
                    )
                all_texts[normalized] = (task_id, bank_name)
                entries.append(
                    {
                        "template_id": _template_id(task_id, bank_name, index, text),
                        "text": text,
                        "task_id": task_id,
                        "skill_family": CANDIDATE_TASKS[task_id].skill_family,
                        "bank": bank_name,
                    }
                )
            bank_reports[bank_name] = entries
        task_reports[task_id] = bank_reports
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-language-template-manifest-v0",
            "assignment_seed": language["assignment_seed"],
            "tasks": task_reports,
            "template_counts": {
                task_id: {
                    bank: len(
                        cast(
                            list[object],
                            cast(Mapping[str, object], task_reports[task_id])[bank],
                        )
                    )
                    for bank in ("train", "validation", "held_out")
                }
                for task_id in TASK_IDS
            },
            "ambiguous_cross_task_text_count": 0,
            "privileged_coordinate_reference_count": 0,
            "passed": True,
        }
    )


def _template_id(task_id: str, bank: str, index: int, text: str) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {"task_id": task_id, "bank": bank, "index": index, "text": text},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return f"phase2b6:{task_id}:{bank}:{index:02d}:{digest[:16]}"


def _template_for_episode(
    *,
    spec: Mapping[str, object],
    task_id: str,
    split: str,
    source_identity: str,
) -> tuple[str, str, str]:
    bank = (
        "held_out"
        if split == "test_unseen_task_language"
        else ("train" if split == "train" else "validation")
    )
    language = cast(Mapping[str, object], spec["language_templates"])
    task_banks = cast(Mapping[str, object], language[task_id])
    templates = cast(list[str], task_banks[bank])
    digest = hashlib.sha256(
        f"{language['assignment_seed']}|{source_identity}|{split}".encode()
    ).digest()
    index = int.from_bytes(digest[:8], "big") % len(templates)
    text = templates[index]
    return _template_id(task_id, bank, index, text), text, bank


def build_primary_split_manifest(
    *, spec: Mapping[str, object], episodes: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Assign each source episode once using stable episode/reset identities."""

    if len(episodes) != EXPECTED_EPISODES:
        raise Phase2B6ContractError(f"split materialization requires {EXPECTED_EPISODES} episodes")
    split_spec = cast(Mapping[str, object], spec["splits"])
    counts = cast(Mapping[str, object], split_spec["per_task_counts"])
    assignments: list[dict[str, object]] = []
    for task_id in TASK_IDS:
        task_rows = [row for row in episodes if row.get("task_id") == task_id]
        if len(task_rows) != 1_000:
            raise Phase2B6ContractError(f"{task_id} requires exactly 1,000 source episodes")
        ordered = sorted(
            task_rows,
            key=lambda row: (
                hashlib.sha256(
                    (
                        f"{split_spec['assignment_seed']}|"
                        f"{row.get('source_trajectory_identity')}|"
                        f"{row.get('reset_identity')}"
                    ).encode()
                ).hexdigest(),
                int(cast(int, row.get("source_episode_id"))),
            ),
        )
        cursor = 0
        for split in PRIMARY_SPLITS:
            count = _positive_int(counts[split], f"split {split}")
            for row in ordered[cursor : cursor + count]:
                source_identity = str(row["source_trajectory_identity"])
                template_id, instruction, bank = _template_for_episode(
                    spec=spec,
                    task_id=task_id,
                    split=split,
                    source_identity=source_identity,
                )
                assignments.append(
                    {
                        "task_id": task_id,
                        "skill_family": CANDIDATE_TASKS[task_id].skill_family,
                        "source_episode_id": int(cast(int, row["source_episode_id"])),
                        "source_trajectory_identity": source_identity,
                        "reset_identity": str(row["reset_identity"]),
                        "action_sha256": str(row["action_sha256"]),
                        "initial_state_sha256": str(row["initial_state_sha256"]),
                        "transition_count": int(cast(int, row["transition_count"])),
                        "primary_split": split,
                        "language_bank": bank,
                        "instruction_template_id": template_id,
                        "instruction": instruction,
                        "derived_episode_identity": derived_episode_identity(
                            spec=spec,
                            source_identity=source_identity,
                            instruction_template_id=template_id,
                            primary_split=split,
                        ),
                        "media_namespace": f"{task_id}/{split}",
                    }
                )
            cursor += count
        if cursor != len(ordered):
            raise Phase2B6ContractError(f"{task_id} split assignment is incomplete")
    assignments.sort(
        key=lambda row: (
            TASK_IDS.index(str(row["task_id"])),
            PRIMARY_SPLITS.index(str(row["primary_split"])),
            int(cast(int, row["source_episode_id"])),
        )
    )
    leakage = audit_primary_split_leakage(assignments)
    if leakage["passed"] is not True:
        raise Phase2B6ContractError("deterministic primary split construction leaked")
    per_task_counts = {
        task_id: dict(
            Counter(str(row["primary_split"]) for row in assignments if row["task_id"] == task_id)
        )
        for task_id in TASK_IDS
    }
    global_counts = dict(Counter(str(row["primary_split"]) for row in assignments))
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-primary-split-manifest-v0",
            "assignment_seed": split_spec["assignment_seed"],
            "assignment_unit": "immutable source episode and exposed reset identity",
            "source_reset_group_limitation": split_spec["reset_group_limitation"],
            "per_task_counts": per_task_counts,
            "global_counts": global_counts,
            "assignments": assignments,
            "assignment_digest": _canonical_identity({"assignments": assignments}),
            "leakage_summary": leakage,
            "passed": (
                per_task_counts
                == {
                    task_id: {name: int(cast(int, counts[name])) for name in PRIMARY_SPLITS}
                    for task_id in TASK_IDS
                }
                and leakage["passed"] is True
            ),
        }
    )


def audit_primary_split_leakage(
    assignments: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Audit every primary split key and return a machine-readable overlap matrix."""

    fields = (
        "derived_episode_identity",
        "source_trajectory_identity",
        "reset_identity",
        "action_sha256",
        "initial_state_sha256",
    )
    matrix: dict[str, dict[str, int]] = {}
    for left_index, left in enumerate(PRIMARY_SPLITS):
        for right in PRIMARY_SPLITS[left_index + 1 :]:
            left_rows = [row for row in assignments if row.get("primary_split") == left]
            right_rows = [row for row in assignments if row.get("primary_split") == right]
            field_counts: dict[str, int] = {}
            for field in fields:
                left_values = {str(row.get(field)) for row in left_rows}
                right_values = {str(row.get(field)) for row in right_rows}
                field_counts[field] = len(left_values & right_values)
            left_media = {str(row.get("media_namespace")) for row in left_rows}
            right_media = {str(row.get("media_namespace")) for row in right_rows}
            field_counts["media_namespace"] = len(left_media & right_media)
            matrix[f"{left}__{right}"] = field_counts
    duplicate_within_fields: dict[str, int] = {}
    for field in fields[:3]:
        values = [str(row.get(field)) for row in assignments]
        duplicate_within_fields[field] = len(values) - len(set(values))
    action_values = [str(row.get("action_sha256")) for row in assignments]
    duplicate_within_fields["action_sha256"] = len(action_values) - len(set(action_values))
    overlap_count = sum(
        count for pair in matrix.values() for count in cast(Mapping[str, int], pair).values()
    )
    return {
        "schema_version": "langmani-v2-phase2b6-primary-leakage-summary-v0",
        "overlap_matrix": matrix,
        "duplicate_within_primary_assignments": duplicate_within_fields,
        "primary_split_overlap_count": overlap_count,
        "passed": overlap_count == 0
        and all(value == 0 for value in duplicate_within_fields.values()),
    }


def build_cross_skill_folds(
    *, spec: Mapping[str, object], split_manifest: Mapping[str, object]
) -> dict[str, object]:
    """Create alternate metadata-only views without duplicating media."""

    assignments = cast(list[dict[str, object]], split_manifest["assignments"])
    fold_spec = cast(Mapping[str, object], spec["cross_skill_folds"])
    folds: dict[str, object] = {}
    for fold_name, raw in fold_spec.items():
        if fold_name == "materialization":
            continue
        definition = cast(Mapping[str, object], raw)
        train_skills = cast(list[str], definition["train_skills"])
        test_skill = str(definition["test_skill"])
        train_ids = [
            str(row["derived_episode_identity"])
            for row in assignments
            if row["primary_split"] == "train" and row["skill_family"] in train_skills
        ]
        test_ids = [
            str(row["derived_episode_identity"])
            for row in assignments
            if row["primary_split"]
            in {
                "validation",
                "test_unseen_reset",
                "test_unseen_task_language",
                "test_visual_shift",
            }
            and row["skill_family"] == test_skill
        ]
        folds[fold_name] = {
            "train_skills": train_skills,
            "test_skill": test_skill,
            "train_episode_count": len(train_ids),
            "test_episode_count": len(test_ids),
            "train_episode_identities": train_ids,
            "test_episode_identities": test_ids,
            "media_duplicated": False,
            "claim_boundary": (
                "This is an alternate experiment view. A separate future model must be "
                "trained for the fold before any unseen-skill claim."
            ),
        }
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-cross-skill-folds-v0",
            "materialization": "metadata_only",
            "primary_split_manifest_fingerprint": split_manifest["fingerprint"],
            "folds": folds,
            "media_duplicated": False,
            "passed": len(folds) == 3,
        }
    )


def apply_visual_shift(
    rgb: object, *, exposure_multiplier: float, rgb_channel_multipliers: Sequence[float]
) -> npt.NDArray[np.uint8]:
    """Apply the frozen evaluation-only post-render appearance transform."""

    image = np.asarray(rgb)
    if image.dtype != np.dtype(np.uint8) or image.shape[-3:] != (256, 256, 3):
        raise Phase2B6ContractError("visual shift requires uint8 RGB [...,256,256,3]")
    gains = np.asarray(rgb_channel_multipliers, dtype=np.float32)
    if gains.shape != (3,) or not np.all(np.isfinite(gains)):
        raise Phase2B6ContractError("visual-shift channel multipliers must be finite RGB gains")
    if not math.isfinite(exposure_multiplier) or not 0.5 <= exposure_multiplier <= 1.5:
        raise Phase2B6ContractError("visual-shift exposure must stay within the bounded contract")
    shifted = image.astype(np.float32) * np.float32(exposure_multiplier) * gains
    return np.clip(np.rint(shifted), 0, 255).astype(np.uint8)


def make_action_chunk(actions: object, *, start: int, horizon: int) -> ActionChunk:
    """Materialize a zero-padded action chunk plus an explicit boolean mask."""

    array = np.asarray(actions)
    if array.ndim != 2 or array.shape[1] != 8 or array.dtype != np.dtype(np.float32):
        raise Phase2B6ContractError("action chunks require float32[T,8]")
    if isinstance(start, bool) or not isinstance(start, int) or not 0 <= start < len(array):
        raise Phase2B6ContractError("action chunk start is outside the episode")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise Phase2B6ContractError("action chunk horizon must be positive")
    chunk = np.zeros((horizon, 8), dtype=np.float32)
    mask = np.ones(horizon, dtype=np.bool_)
    available = min(horizon, len(array) - start)
    chunk[:available] = array[start : start + available]
    mask[:available] = False
    return ActionChunk(action=chunk, action_is_pad=mask)


def masked_mean_squared_error(prediction: object, target: object, action_is_pad: object) -> float:
    """Compute MSE using only non-padding action timesteps."""

    predicted = np.asarray(prediction)
    expected = np.asarray(target)
    padding = np.asarray(action_is_pad)
    if predicted.shape != expected.shape or predicted.ndim < 2 or predicted.shape[-1] != 8:
        raise Phase2B6ContractError("masked loss requires matching [...,H,8] arrays")
    if padding.dtype != np.dtype(bool) or padding.shape != predicted.shape[:-1]:
        raise Phase2B6ContractError("action_is_pad must be bool[...] matching action timesteps")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(expected)):
        raise Phase2B6ContractError("masked loss arrays must be finite")
    valid = ~padding
    if not np.any(valid):
        raise Phase2B6ContractError("masked loss requires at least one non-padding action")
    squared = np.square(predicted.astype(np.float64) - expected.astype(np.float64))
    return float(np.mean(squared[valid]))


def build_padding_audit(
    episodes: Sequence[Mapping[str, object]], *, horizons: Sequence[int] = ACTION_HORIZONS
) -> dict[str, object]:
    """Compute exact padding statistics by task, split, and chunk position."""

    if not episodes:
        raise Phase2B6ContractError("padding audit requires episodes")
    reports: dict[str, object] = {}
    for horizon in horizons:
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise Phase2B6ContractError("padding horizons must be positive integers")
        grouped: dict[str, dict[str, int | float]] = {}
        total_chunks = 0
        padded_chunks = 0
        padded_timesteps = 0
        position_counts = [0] * horizon
        for row in episodes:
            length = _positive_int(row.get("transition_count"), "episode transition count")
            task_id = str(row.get("task_id"))
            split = str(row.get("primary_split"))
            key = f"{task_id}/{split}"
            entry = grouped.setdefault(
                key,
                {
                    "episode_count": 0,
                    "total_chunks": 0,
                    "padded_chunks": 0,
                    "total_chunk_timesteps": 0,
                    "padded_timesteps": 0,
                },
            )
            episode_padded_chunks = min(length, horizon - 1)
            episode_position_counts = [min(position, length) for position in range(horizon)]
            episode_padded_timesteps = sum(episode_position_counts)
            entry["episode_count"] += 1
            entry["total_chunks"] += length
            entry["padded_chunks"] += episode_padded_chunks
            entry["total_chunk_timesteps"] += length * horizon
            entry["padded_timesteps"] += episode_padded_timesteps
            total_chunks += length
            padded_chunks += episode_padded_chunks
            padded_timesteps += episode_padded_timesteps
            position_counts = [
                current + added
                for current, added in zip(position_counts, episode_position_counts, strict=True)
            ]
        for entry in grouped.values():
            entry["padded_chunk_fraction"] = entry["padded_chunks"] / entry["total_chunks"]
            entry["padded_timestep_fraction"] = (
                entry["padded_timesteps"] / entry["total_chunk_timesteps"]
            )
        reports[str(horizon)] = {
            "horizon": horizon,
            "total_chunks": total_chunks,
            "chunks_containing_padding": padded_chunks,
            "padded_chunk_fraction": padded_chunks / total_chunks,
            "total_chunk_timesteps": total_chunks * horizon,
            "padded_timesteps": padded_timesteps,
            "padded_timestep_fraction": padded_timesteps / (total_chunks * horizon),
            "padded_count_by_chunk_position": position_counts,
            "by_task_and_split": grouped,
        }
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-padding-audit-v0",
            "mask_feature": "action_is_pad",
            "mask_dtype": "bool",
            "padding_value": "zero_vector",
            "masked_loss_contract_verified": True,
            "audits": reports,
            "optimizer_steps": 0,
            "passed": tuple(int(key) for key in reports) == tuple(horizons),
        }
    )


def build_task_balance_manifest(
    episodes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Describe episode/frame balance and future sampling choices without training."""

    train = [row for row in episodes if row.get("primary_split") == "train"]
    episode_counts = Counter(str(row["task_id"]) for row in train)
    frame_counts = {
        task_id: sum(
            int(cast(int, row["transition_count"]))
            for row in train
            if row.get("task_id") == task_id
        )
        for task_id in TASK_IDS
    }
    total_episodes = sum(episode_counts.values())
    total_frames = sum(frame_counts.values())
    uniform_task_probabilities = {task_id: 1.0 / len(TASK_IDS) for task_id in TASK_IDS}
    uniform_episode_probabilities = {
        task_id: episode_counts[task_id] / total_episodes for task_id in TASK_IDS
    }
    natural_frame_probabilities = {
        task_id: frame_counts[task_id] / total_frames for task_id in TASK_IDS
    }
    uniform_task_frame_multipliers = {
        task_id: total_frames / (len(TASK_IDS) * frame_counts[task_id]) for task_id in TASK_IDS
    }
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-task-balance-v0",
            "scope": "primary_train_only",
            "episode_counts": dict(episode_counts),
            "transition_counts": frame_counts,
            "episode_balance": {
                task_id: episode_counts[task_id] / total_episodes for task_id in TASK_IDS
            },
            "frame_balance": natural_frame_probabilities,
            "future_sampling_recommendations": {
                "uniform_task": {
                    "task_probabilities": uniform_task_probabilities,
                    "per_frame_multipliers_relative_to_natural": uniform_task_frame_multipliers,
                    "recommended_for_primary_multiskill_baselines": True,
                },
                "uniform_episode": {
                    "task_probabilities": uniform_episode_probabilities,
                    "within_task_note": (
                        "sample episodes uniformly, then sample a valid frame within the episode"
                    ),
                },
                "natural_frame_frequency": {
                    "task_probabilities": natural_frame_probabilities,
                    "warning": "StackCube receives a larger contribution because its episodes are longer.",
                },
            },
            "training_started": False,
            "optimizer_steps": 0,
            "passed": all(episode_counts[task_id] == 700 for task_id in TASK_IDS),
        }
    )


def classify_result(
    *,
    source_integrity_valid: bool,
    replay_and_derived_integrity_valid: bool,
    accepted_skill_count: int,
    archive_created: bool,
    restore_valid: bool,
) -> Phase2B6Result:
    """Apply the fixed A/B/C/D result semantics."""

    if not source_integrity_valid or not replay_and_derived_integrity_valid:
        return Phase2B6Result.RESULT_C
    if accepted_skill_count < 3:
        return Phase2B6Result.RESULT_B
    if not archive_created or not restore_valid:
        return Phase2B6Result.RESULT_D
    return Phase2B6Result.RESULT_A


def authorization_state(*, accepted: bool) -> dict[str, object]:
    """Expose later-phase eligibility while every actual training path stays closed."""

    return {
        "schema_version": "langmani-v2-phase2b6-authorization-state-v0",
        "accepted_multiskill_dataset_validated": accepted,
        "act_baseline_training_eligible": accepted,
        "smolvla_push_multiskill_training_eligible": accepted,
        "smolvla_training_eligible": accepted,
        "vla_jepa_training_eligible": accepted,
        "eligibility_is_authorization": False,
        "act_training_authorized": False,
        "smolvla_training_authorized": False,
        "vla_jepa_training_authorized": False,
        "student_policy_training_started": False,
        "optimizer_created": False,
        "optimizer_steps": 0,
        "backward_passes": 0,
        "act_training_started": False,
        "smolvla_training_started": False,
        "vla_jepa_training_started": False,
        "custom_push_expert_route_active": False,
        "custom_push_expert_reactivation_authorized": False,
    }


def create_accepted_dataset_package(
    *,
    gates: Mapping[str, bool],
    package_fields: Mapping[str, object],
) -> dict[str, object]:
    """Create the accepted package only after every frozen gate passes."""

    required = {
        "official_source_hashes",
        "source_trajectory_accounting",
        "replay_outcome_agreement",
        "action_integrity",
        "finite_values",
        "lerobot_readback",
        "primary_split_leakage",
        "privileged_field_exclusion",
        "train_only_normalization",
        "action_padding_masks",
        "archive",
        "restore",
    }
    if set(gates) != required or not all(gates.values()):
        raise Phase2B6ContractError("accepted package requires every frozen Phase 2B.6 gate")
    prohibited = {
        "model",
        "checkpoint",
        "optimizer",
        "training_run",
        "policy_weights",
    }
    if prohibited & set(package_fields):
        raise Phase2B6ContractError("accepted dataset package cannot contain trained-model fields")
    payload: dict[str, object] = {
        "schema_version": "langmani-v2-phase2b6-accepted-multiskill-dataset-v0",
        "package_identity": DATASET_PACKAGE_ID,
        "tasks": list(TASK_IDS),
        "skill_families": [CANDIDATE_TASKS[task_id].skill_family for task_id in TASK_IDS],
        "episode_count": EXPECTED_EPISODES,
        "frame_count": EXPECTED_TRANSITIONS,
        "gates": dict(gates),
        "contains_trained_model": False,
        "student_policy_training_started": False,
        "optimizer_steps": 0,
        **dict(package_fields),
    }
    return fingerprinted(payload)


def summarize_replay_gate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Apply the Phase 2B.6 per-episode 100%/zero-error replay gate."""

    if not rows:
        raise Phase2B6ContractError("replay gate requires episode rows")
    counts = {
        "source_success": sum(row.get("source_success") is True for row in rows),
        "replay_success": sum(row.get("replay_success") is True for row in rows),
        "outcome_agreement": sum(row.get("categorical_outcome_agreement") is True for row in rows),
        "action_frame_alignment": sum(row.get("action_frame_alignment") is True for row in rows),
    }
    invalid_actions = sum(_nonnegative_int(row.get("invalid_action_count", 0)) for row in rows)
    nonfinite_values = sum(_nonnegative_int(row.get("nonfinite_value_count", 0)) for row in rows)
    simulator_errors = sum(_nonnegative_int(row.get("simulator_error_count", 0)) for row in rows)
    return {
        "episode_count": len(rows),
        **counts,
        "categorical_outcome_agreement_rate": counts["outcome_agreement"] / len(rows),
        "action_frame_alignment_rate": counts["action_frame_alignment"] / len(rows),
        "invalid_action_count": invalid_actions,
        "nonfinite_value_count": nonfinite_values,
        "simulator_error_count": simulator_errors,
        "thresholds": {
            "categorical_outcome_agreement_rate": 1.0,
            "action_frame_alignment_rate": 1.0,
            "invalid_action_count": 0,
            "nonfinite_value_count": 0,
            "simulator_error_count": 0,
        },
        "passed": (
            all(value == len(rows) for value in counts.values())
            and invalid_actions == 0
            and nonfinite_values == 0
            and simulator_errors == 0
        ),
    }


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Phase2B6ContractError("count must be a non-negative integer")
    return value


def validate_student_feature_names(feature_names: Iterable[str]) -> None:
    """Reject privileged or undeclared student-policy features."""

    actual = set(feature_names)
    required = {
        "observation.images.base_camera",
        "observation.state",
        "action",
        "task",
        "task_id",
        "skill_family",
        "source_episode_identity",
        "derived_episode_identity",
        "instruction_template_id",
        "primary_split",
        "episode_index",
        "frame_index",
        "timestamp",
    }
    prohibited = {
        "object_pose",
        "goal_pose",
        "full_simulator_state",
        "privileged_contact_state",
        "reward",
        "success",
        "success_internals",
        "expert_phase",
        "future_observation",
    }
    if not required <= actual:
        raise Phase2B6ContractError(f"student schema lacks {sorted(required - actual)}")
    if actual & prohibited:
        raise Phase2B6ContractError(
            f"student schema contains privileged fields {sorted(actual & prohibited)}"
        )


def group_assignments(
    assignments: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str], list[dict[str, object]]]:
    """Return stable task/split groups for task-root production."""

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in assignments:
        task_id = str(row.get("task_id"))
        split = str(row.get("primary_split"))
        if task_id not in TASK_IDS or split not in PRIMARY_SPLITS:
            raise Phase2B6ContractError("assignment has unknown task or split")
        grouped[(task_id, split)].append(dict(row))
    for rows in grouped.values():
        rows.sort(key=lambda row: int(cast(int, row["source_episode_id"])))
    expected_groups = {(task_id, split) for task_id in TASK_IDS for split in PRIMARY_SPLITS}
    if set(grouped) != expected_groups:
        raise Phase2B6ContractError("task/split assignment groups are incomplete")
    return dict(grouped)


__all__ = [
    "ACTION_HORIZONS",
    "DATASET_PACKAGE_ID",
    "EXPECTED_EPISODES",
    "EXPECTED_TRANSITIONS",
    "PHASE2B5_ARTIFACT_MANIFEST_SHA256",
    "PHASE2B5_PACKAGE_SHA256",
    "PRIMARY_SPLITS",
    "SOURCE_BRANCH",
    "SOURCE_COMMIT",
    "TARGET_BRANCH",
    "TASK_IDS",
    "ActionChunk",
    "Phase2B6ContractError",
    "Phase2B6Result",
    "apply_visual_shift",
    "audit_primary_split_leakage",
    "authorization_state",
    "build_cross_skill_folds",
    "build_language_template_manifest",
    "build_padding_audit",
    "build_primary_split_manifest",
    "build_task_balance_manifest",
    "classify_result",
    "create_accepted_dataset_package",
    "derived_episode_identity",
    "enforce_starting_point",
    "fingerprinted",
    "group_assignments",
    "load_production_spec",
    "make_action_chunk",
    "masked_mean_squared_error",
    "source_trajectory_identity",
    "summarize_replay_gate",
    "validate_student_feature_names",
    "verify_phase2b5_package",
]
