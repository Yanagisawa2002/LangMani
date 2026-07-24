"""Fail-closed contracts for Phase 2B.6.1-R Vulkan recovery.

This module is simulator-independent.  It freezes the only three rendering
configurations that may be inspected, validates ICD manifests and clean child
environments, and derives the terminal result without authorizing replay,
production, datasets, or model activity.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Final, cast

SOURCE_BRANCH: Final = "codex/langmani-v2-phase2b6-1-replay-failure-forensics"
SOURCE_COMMIT: Final = "4f96a0f42dbaf0adb58c4c3aad79fdd7bccfde84"
TARGET_BRANCH: Final = "codex/langmani-v2-phase2b6-1r-vulkan-recovery"
TASK_ID: Final = "StackCube-v1"
EXPECTED_GPU_NAME: Final = "NVIDIA GeForce RTX 5090"
EXPECTED_CONTROL_MODE: Final = "pd_joint_pos"
EXPECTED_ACTION_SHAPE: Final = (8,)
EXPECTED_OBS_MODE: Final = "rgb"
EXPECTED_SIM_BACKEND: Final = "physx_cpu"
EXPECTED_RENDER_BACKEND: Final = "sapien_cuda"
EXPECTED_SENSOR_SIZE: Final = (256, 256)
PRIMARY_ICD_PATH: Final = "/etc/vulkan/icd.d/my_nvidia_icd.json"
SECONDARY_ICD_PATH: Final = "/etc/vulkan/icd.d/nvidia_icd.json"
EGL_VENDOR_PATH: Final = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
RENDER_ENVIRONMENT_NAMES: Final = (
    "VK_ICD_FILENAMES",
    "VK_DRIVER_FILES",
    "__EGL_VENDOR_LIBRARY_FILENAMES",
    "DISPLAY",
    "WAYLAND_DISPLAY",
    "XDG_RUNTIME_DIR",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
)
PROHIBITED_RUNTIME_TERMS: Final = (
    "run_v2_phase2b6.py produce",
    "run_v2_phase2b6_1.py run",
    "replay",
    "writer",
    "export",
    "archive",
    "restore",
    "collect_",
    "train_act",
    "smolvla",
    "vla_jepa",
)


class Phase2B61RContractError(ValueError):
    """Raised when the bounded recovery contract would be widened."""


class CandidateKind(StrEnum):
    """The only preregistered loader configurations."""

    PRIMARY = "candidate_a_explicit_egl_icd"
    SECONDARY = "candidate_b_explicit_glx_icd"
    DEFAULT = "candidate_c_clean_loader_default"


class Phase2B61RResult(StrEnum):
    """Terminal classifications for the recovery phase."""

    RESULT_A = "RESULT_A"
    RESULT_B = "RESULT_B"
    RESULT_C = "RESULT_C"
    RESULT_D = "RESULT_D"


def canonical_json_sha256(payload: Mapping[str, object]) -> str:
    """Hash one JSON object using the repository's canonical encoding."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def fingerprinted(payload: Mapping[str, object]) -> dict[str, object]:
    """Return a JSON-ready document with a content-derived fingerprint."""

    result = dict(payload)
    result.pop("fingerprint", None)
    result["fingerprint"] = canonical_json_sha256(result)
    return result


def sha256_file(path: Path) -> str:
    """Return a prefixed SHA-256 for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def enforce_starting_point(
    *,
    branch: str,
    source_commit: str,
    source_is_ancestor: bool,
) -> None:
    """Require the new branch to descend from the exact immutable source."""

    if branch != TARGET_BRANCH:
        raise Phase2B61RContractError(f"expected branch {TARGET_BRANCH}, got {branch}")
    if source_commit != SOURCE_COMMIT:
        raise Phase2B61RContractError(f"expected source commit {SOURCE_COMMIT}")
    if not source_is_ancestor:
        raise Phase2B61RContractError("source commit is not an ancestor of the execution HEAD")


def parse_icd_manifest(path: Path) -> dict[str, str]:
    """Parse a Vulkan ICD JSON without resolving or loading its library."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2B61RContractError(f"failed to parse ICD manifest {path}") from error
    if not isinstance(document, dict):
        raise Phase2B61RContractError("ICD manifest must contain one object")
    icd = document.get("ICD")
    if not isinstance(icd, dict):
        raise Phase2B61RContractError("ICD manifest is missing the ICD object")
    file_format = document.get("file_format_version")
    library_path = icd.get("library_path")
    api_version = icd.get("api_version")
    if not isinstance(file_format, str) or not file_format:
        raise Phase2B61RContractError("ICD file_format_version is missing")
    if not isinstance(library_path, str) or not library_path:
        raise Phase2B61RContractError("ICD library_path is missing")
    if not isinstance(api_version, str) or not api_version:
        raise Phase2B61RContractError("ICD api_version is missing")
    return {
        "file_format_version": file_format,
        "library_path": library_path,
        "api_version": api_version,
    }


def candidate_protocol() -> tuple[dict[str, object], ...]:
    """Return the frozen small candidate list in execution order."""

    return (
        {
            "candidate_id": CandidateKind.PRIMARY.value,
            "purpose": "primary_recovery_contract",
            "vk_icd_filenames": PRIMARY_ICD_PATH,
            "vk_driver_files": None,
            "egl_vendor_filenames": EGL_VENDOR_PATH,
            "may_run_sapien_probe": True,
            "may_run_zero_step_stackcube": True,
        },
        {
            "candidate_id": CandidateKind.SECONDARY.value,
            "purpose": "separate_official_nvidia_icd_audit",
            "vk_icd_filenames": SECONDARY_ICD_PATH,
            "vk_driver_files": None,
            "egl_vendor_filenames": EGL_VENDOR_PATH,
            "may_run_sapien_probe": False,
            "may_run_zero_step_stackcube": False,
        },
        {
            "candidate_id": CandidateKind.DEFAULT.value,
            "purpose": "negative_control_default_loader",
            "vk_icd_filenames": None,
            "vk_driver_files": None,
            "egl_vendor_filenames": None,
            "may_run_sapien_probe": True,
            "may_run_zero_step_stackcube": False,
        },
    )


def candidate_by_id(candidate_id: str) -> dict[str, object]:
    """Resolve one preregistered candidate and reject arbitrary combinations."""

    for candidate in candidate_protocol():
        if candidate["candidate_id"] == candidate_id:
            return dict(candidate)
    raise Phase2B61RContractError(f"unregistered ICD candidate: {candidate_id}")


def validate_primary_environment(environment: Mapping[str, str]) -> None:
    """Require the exact process-scoped primary Vulkan/EGL binding."""

    expected = {
        "VK_ICD_FILENAMES": PRIMARY_ICD_PATH,
        "__EGL_VENDOR_LIBRARY_FILENAMES": EGL_VENDOR_PATH,
    }
    for name, value in expected.items():
        if environment.get(name) != value:
            raise Phase2B61RContractError(f"{name} must be exactly {value}")
    if environment.get("VK_DRIVER_FILES"):
        raise Phase2B61RContractError("VK_DRIVER_FILES must remain unset for the primary contract")


def clean_child_environment(
    *,
    conda_prefix: Path,
    home: Path,
    xdg_runtime_dir: Path,
    candidate_id: str,
    inherited: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Construct the documented allowlisted environment for one fresh process."""

    candidate = candidate_by_id(candidate_id)
    source = dict(inherited or {})
    result = {
        "HOME": home.as_posix(),
        "PATH": f"{conda_prefix.as_posix()}/bin:/usr/bin:/bin",
        "CONDA_PREFIX": conda_prefix.as_posix(),
        "CONDA_DEFAULT_ENV": conda_prefix.name,
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "LANG": source.get("LANG", "C.UTF-8"),
        "LC_ALL": source.get("LC_ALL", "C.UTF-8"),
        "XDG_RUNTIME_DIR": xdg_runtime_dir.as_posix(),
        "CUDA_VISIBLE_DEVICES": "0",
    }
    if source.get("LD_LIBRARY_PATH"):
        result["LD_LIBRARY_PATH"] = source["LD_LIBRARY_PATH"]
    if source.get("NVIDIA_VISIBLE_DEVICES"):
        result["NVIDIA_VISIBLE_DEVICES"] = source["NVIDIA_VISIBLE_DEVICES"]
    vk_icd = candidate["vk_icd_filenames"]
    egl_vendor = candidate["egl_vendor_filenames"]
    if isinstance(vk_icd, str):
        result["VK_ICD_FILENAMES"] = vk_icd
    if isinstance(egl_vendor, str):
        result["__EGL_VENDOR_LIBRARY_FILENAMES"] = egl_vendor
    if candidate_id == CandidateKind.PRIMARY.value:
        validate_primary_environment(result)
    return result


def device_identity_matches(device: Mapping[str, object]) -> bool:
    """Require a CUDA-render-capable RTX 5090 and reject software devices."""

    name = str(device.get("name", ""))
    lowered = name.lower()
    return (
        EXPECTED_GPU_NAME.lower() in lowered
        and "llvmpipe" not in lowered
        and "software" not in lowered
        and device.get("is_cuda") is True
        and device.get("can_render") is True
    )


def enforce_zero_step_counts(
    *,
    explicit_reset_count: int,
    explicit_step_count: int,
    action_submission_count: int,
) -> None:
    """Fail if the recovery phase performs any forbidden control operation."""

    values = (explicit_reset_count, explicit_step_count, action_submission_count)
    if any(isinstance(value, bool) or value != 0 for value in values):
        raise Phase2B61RContractError("zero-step gate observed reset, step, or action activity")


def repeatability_passed(runs: Sequence[Mapping[str, object]]) -> bool:
    """Require three matching successful fresh-process contracts."""

    if len(runs) != 3 or any(run.get("passed") is not True for run in runs):
        return False
    identities: set[tuple[object, ...]] = {
        (
            run.get("icd_sha256"),
            run.get("vulkan_device_name"),
            run.get("render_device_name"),
            run.get("render_device_pci"),
            tuple(cast(Sequence[object], run.get("action_shape", []))),
            run.get("control_mode"),
            run.get("obs_mode"),
        )
        for run in runs
    }
    process_ids = {run.get("pid") for run in runs}
    return len(identities) == 1 and len(process_ids) == 3 and None not in process_ids


def classify_result(
    *,
    host_evidence_sufficient: bool,
    vulkan_passed: bool,
    sapien_passed: bool,
    zero_step_passed: bool,
    repeatability_validated: bool,
    correct_device_selected: bool,
    cleanup_passed: bool,
    first_failure_layer: str | None,
) -> dict[str, object]:
    """Derive exactly one fail-closed Phase 2B.6.1-R result."""

    if not host_evidence_sufficient:
        result = Phase2B61RResult.RESULT_D
        reason = "evidence_or_host_state_insufficient"
    elif vulkan_passed and sapien_passed and not zero_step_passed:
        result = Phase2B61RResult.RESULT_B
        reason = first_failure_layer or "stackcube_environment_construction"
    elif (
        vulkan_passed
        and sapien_passed
        and zero_step_passed
        and repeatability_validated
        and correct_device_selected
        and cleanup_passed
    ):
        result = Phase2B61RResult.RESULT_A
        reason = "vulkan_environment_contract_recovered"
    else:
        result = Phase2B61RResult.RESULT_C
        reason = first_failure_layer or "explicit_rendering_contract_invalid"
    eligible = result is Phase2B61RResult.RESULT_A
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1r-result-v0",
            "result": result.value,
            "reason": reason,
            "vulkan_preflight_validated": eligible,
            "stackcube_zero_step_construction_validated": eligible,
            "phase2b6_1_forensic_restart_eligible": eligible,
            "phase2b6_1_forensic_restart_authorized": False,
            "phase2b6_production_resume_authorized": False,
            "accepted_multiskill_dataset_validated": False,
            "passed": True,
        }
    )


def authorization_state() -> dict[str, object]:
    """Return the mandatory terminal authorization state for every result."""

    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1r-authorization-v0",
            "phase2b6_1_forensic_restart_authorized": False,
            "phase2b6_production_resume_authorized": False,
            "accepted_multiskill_dataset_validated": False,
            "act_training_eligible": False,
            "smolvla_training_eligible": False,
            "vla_jepa_training_eligible": False,
            "act_training_authorized": False,
            "smolvla_training_authorized": False,
            "vla_jepa_training_authorized": False,
            "optimizer_created": False,
            "backward_passes": 0,
            "optimizer_steps": 0,
            "student_policy_training_started": False,
            "accepted_dataset_package_created": False,
            "production_or_forensic_replay_started": False,
            "passed": True,
        }
    )


__all__ = [
    "CandidateKind",
    "EGL_VENDOR_PATH",
    "EXPECTED_ACTION_SHAPE",
    "EXPECTED_CONTROL_MODE",
    "EXPECTED_GPU_NAME",
    "EXPECTED_OBS_MODE",
    "EXPECTED_RENDER_BACKEND",
    "EXPECTED_SENSOR_SIZE",
    "EXPECTED_SIM_BACKEND",
    "PRIMARY_ICD_PATH",
    "PROHIBITED_RUNTIME_TERMS",
    "Phase2B61RContractError",
    "Phase2B61RResult",
    "RENDER_ENVIRONMENT_NAMES",
    "SECONDARY_ICD_PATH",
    "SOURCE_BRANCH",
    "SOURCE_COMMIT",
    "TARGET_BRANCH",
    "TASK_ID",
    "authorization_state",
    "candidate_by_id",
    "candidate_protocol",
    "canonical_json_sha256",
    "classify_result",
    "clean_child_environment",
    "device_identity_matches",
    "enforce_starting_point",
    "enforce_zero_step_counts",
    "fingerprinted",
    "parse_icd_manifest",
    "repeatability_passed",
    "sha256_file",
    "validate_primary_environment",
]
