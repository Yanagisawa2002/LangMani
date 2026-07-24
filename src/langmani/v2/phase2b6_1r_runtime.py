"""Native runtime for the bounded Phase 2B.6.1-R zero-step gate.

The parent process inventories immutable inputs and launches fresh child
processes.  A child may run ``vulkaninfo``, construct a SAPIEN RenderSystem,
construct ``StackCube-v1``, inspect static contracts, and close it.  No code in
this module resets an environment, submits an action, or advances physics.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path, PurePosixPath
from typing import Any, cast

from langmani.v2.phase2b6_1r import (
    EGL_VENDOR_PATH,
    EXPECTED_ACTION_SHAPE,
    EXPECTED_CONTROL_MODE,
    EXPECTED_GPU_NAME,
    EXPECTED_OBS_MODE,
    EXPECTED_RENDER_BACKEND,
    EXPECTED_SENSOR_SIZE,
    EXPECTED_SIM_BACKEND,
    PRIMARY_ICD_PATH,
    PROHIBITED_RUNTIME_TERMS,
    RENDER_ENVIRONMENT_NAMES,
    SOURCE_BRANCH,
    SOURCE_COMMIT,
    TARGET_BRANCH,
    TASK_ID,
    CandidateKind,
    authorization_state,
    candidate_by_id,
    candidate_protocol,
    classify_result,
    clean_child_environment,
    device_identity_matches,
    enforce_starting_point,
    enforce_zero_step_counts,
    fingerprinted,
    parse_icd_manifest,
    repeatability_passed,
    sha256_file,
    validate_primary_environment,
)


class Phase2B61RRuntimeError(RuntimeError):
    """Raised when the native recovery contract cannot be preserved."""


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2B61RRuntimeError(f"failed to read JSON {path}") from error
    if not isinstance(value, dict):
        raise Phase2B61RRuntimeError(f"{path} must contain one object")
    return value


def _write_json(path: Path, payload: Mapping[str, object], *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise Phase2B61RRuntimeError(f"refusing to overwrite evidence: {path}")
    staging = path.parent / f".{path.name}.partial"
    if staging.exists():
        raise Phase2B61RRuntimeError(f"stale staging file exists: {staging}")
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with staging.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()


def _git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=check,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _command(
    command: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
    timeout_seconds: float = 60.0,
) -> dict[str, object]:
    started = time.perf_counter()
    try:
        result = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            env=dict(environment) if environment is not None else None,
            timeout=timeout_seconds,
        )
        return {
            "command": list(command),
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": False,
            "wall_clock_seconds": time.perf_counter() - started,
            "passed": result.returncode == 0,
        }
    except subprocess.TimeoutExpired as error:
        return {
            "command": list(command),
            "exit_code": None,
            "stdout": error.stdout or "",
            "stderr": error.stderr or "",
            "timed_out": True,
            "wall_clock_seconds": time.perf_counter() - started,
            "passed": False,
        }
    except OSError as error:
        return {
            "command": list(command),
            "exit_code": None,
            "stdout": "",
            "stderr": f"{type(error).__name__}: {error}",
            "timed_out": False,
            "wall_clock_seconds": time.perf_counter() - started,
            "passed": False,
        }


def _safe_relative(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise Phase2B61RRuntimeError(f"unsafe manifest path: {relative}")
    return root.joinpath(*pure.parts)


def _fingerprint_valid(document: Mapping[str, object]) -> bool:
    from langmani.v2.phase2b6_1r import canonical_json_sha256

    expected = document.get("fingerprint")
    payload = dict(document)
    payload.pop("fingerprint", None)
    return expected == canonical_json_sha256(payload)


def _verify_artifact_manifest(root: Path) -> dict[str, object]:
    manifest = _read_json(root / "artifact_manifest.json")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise Phase2B61RRuntimeError(f"invalid artifact manifest at {root}")
    checks = []
    for raw in entries:
        if not isinstance(raw, dict):
            raise Phase2B61RRuntimeError("artifact manifest entry must be an object")
        relative = str(raw["path"])
        path = _safe_relative(root, relative)
        checks.append(
            {
                "path": relative,
                "exists": path.is_file(),
                "sha256_matches": path.is_file() and sha256_file(path) == raw.get("sha256"),
                "size_matches": path.is_file() and path.stat().st_size == raw.get("size_bytes"),
            }
        )
    return {
        "root": root.as_posix(),
        "manifest_fingerprint": manifest.get("fingerprint"),
        "manifest_fingerprint_valid": _fingerprint_valid(manifest),
        "file_count": len(checks),
        "files": checks,
        "passed": bool(checks)
        and _fingerprint_valid(manifest)
        and all(row["sha256_matches"] and row["size_matches"] for row in checks),
    }


def _verify_frozen_relevant_files(repo_root: Path, frozen_root: Path) -> dict[str, object]:
    prior = _read_json(
        repo_root
        / "artifacts"
        / "langmani_v2"
        / "phase_2b6_1"
        / "frozen_partial_output_inventory.json"
    )
    entries = prior.get("directly_relevant_files")
    if not isinstance(entries, list):
        raise Phase2B61RRuntimeError("prior frozen partial inventory is malformed")
    checks = []
    for raw in entries:
        if not isinstance(raw, dict):
            raise Phase2B61RRuntimeError("frozen inventory entry must be an object")
        relative = str(raw["path"])
        path = _safe_relative(frozen_root, relative)
        checks.append(
            {
                "path": relative,
                "exists": path.is_file(),
                "sha256_matches": path.is_file() and sha256_file(path) == raw.get("sha256"),
                "size_matches": path.is_file() and path.stat().st_size == raw.get("size_bytes"),
            }
        )
    target_npz = frozen_root / "work" / "raw" / "stackcube" / "episode_000938.npz"
    return {
        "frozen_root": frozen_root.as_posix(),
        "directly_relevant_file_count": len(checks),
        "files": checks,
        "episode_938_npz_exists": target_npz.exists(),
        "frozen_root_write_attempted": False,
        "passed": bool(checks)
        and not target_npz.exists()
        and all(row["sha256_matches"] and row["size_matches"] for row in checks),
    }


def _gpu_audit() -> dict[str, object]:
    query = _command(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,pci.bus_id,driver_version,memory.used,"
            "memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    devices = []
    if query["passed"] is True:
        for line in str(query["stdout"]).strip().splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 7:
                continue
            try:
                devices.append(
                    {
                        "uuid": fields[0],
                        "name": fields[1],
                        "pci_bus_id": fields[2],
                        "driver_version": fields[3],
                        "memory_used_mib": int(fields[4]),
                        "memory_total_mib": int(fields[5]),
                        "utilization_percent": int(fields[6]),
                    }
                )
            except ValueError:
                continue
    compute = _command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    compute_rows = [
        line.strip()
        for line in str(compute["stdout"]).splitlines()
        if line.strip() and "No running processes" not in line
    ]
    return {
        "query": query,
        "devices": devices,
        "compute_processes": compute_rows,
        "intended_gpu_present": len(devices) == 1
        and EXPECTED_GPU_NAME.lower() in str(devices[0]["name"]).lower(),
        "passed": len(devices) == 1
        and EXPECTED_GPU_NAME.lower() in str(devices[0]["name"]).lower(),
    }


def _process_audit(*, excluded_pids: set[int] | None = None) -> dict[str, object]:
    excluded = set(excluded_pids or set()) | {os.getpid(), os.getppid()}
    query = _command(["ps", "-eo", "pid=,ppid=,args="])
    matches = []
    if query["passed"] is True:
        for line in str(query["stdout"]).splitlines():
            fields = line.strip().split(maxsplit=2)
            if len(fields) != 3:
                continue
            try:
                pid = int(fields[0])
                ppid = int(fields[1])
            except ValueError:
                continue
            command = fields[2]
            lowered = command.lower()
            if pid not in excluded and any(term in lowered for term in PROHIBITED_RUNTIME_TERMS):
                matches.append({"pid": pid, "ppid": ppid, "command": command})
    return {
        "query_succeeded": query["passed"],
        "prohibited_active_processes": matches,
        "passed": query["passed"] is True and not matches,
    }


def _disk_audit(paths: Sequence[Path]) -> dict[str, object]:
    rows = []
    for path in paths:
        usage = shutil.disk_usage(path)
        rows.append(
            {
                "path": path.resolve().as_posix(),
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
            }
        )
    return {
        "paths": rows,
        "minimum_required_free_bytes": 1024**3,
        "compact_diagnostics_only": True,
        "passed": bool(rows) and all(int(cast(Any, row["free_bytes"])) >= 1024**3 for row in rows),
    }


def _repository_audit(repo_root: Path) -> dict[str, object]:
    branch = _git(repo_root, "branch", "--show-current")
    head = _git(repo_root, "rev-parse", "HEAD")
    source_is_ancestor = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, head],
            cwd=repo_root,
            check=False,
        ).returncode
        == 0
    )
    enforce_starting_point(
        branch=branch,
        source_commit=SOURCE_COMMIT,
        source_is_ancestor=source_is_ancestor,
    )
    status = _git(repo_root, "status", "--porcelain=v1")
    upstream = _git(repo_root, "rev-parse", "@{u}", check=False)
    source_remote = _git(
        repo_root,
        "ls-remote",
        "origin",
        f"refs/heads/{SOURCE_BRANCH}",
        check=False,
    )
    target_remote = _git(
        repo_root,
        "ls-remote",
        "origin",
        f"refs/heads/{TARGET_BRANCH}",
        check=False,
    )
    source_remote_sha = source_remote.split()[0] if source_remote else None
    target_remote_sha = target_remote.split()[0] if target_remote else None
    checks = {
        "target_branch": branch == TARGET_BRANCH,
        "source_commit_is_ancestor": source_is_ancestor,
        "source_remote_sha": source_remote_sha == SOURCE_COMMIT,
        "worktree_clean": not status,
        "upstream_matches_head": upstream == head,
        "target_remote_matches_head": target_remote_sha == head,
    }
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1r-repository-audit-v0",
            "repository_path": repo_root.resolve().as_posix(),
            "worktree_path": repo_root.resolve().as_posix(),
            "source_branch": SOURCE_BRANCH,
            "source_commit": SOURCE_COMMIT,
            "branch": branch,
            "head": head,
            "upstream": upstream,
            "remote": _git(repo_root, "remote", "get-url", "origin"),
            "source_remote_sha": source_remote_sha,
            "target_remote_sha": target_remote_sha,
            "status_porcelain": status,
            "source_is_ancestor": source_is_ancestor,
            "checks": checks,
            "passed": all(checks.values()),
        }
    )


def _ldconfig_inventory() -> tuple[list[str], dict[str, str]]:
    executable = shutil.which("ldconfig") or "/sbin/ldconfig"
    query = _command([executable, "-p"])
    lines = []
    sonames: dict[str, str] = {}
    if query["passed"] is True:
        for line in str(query["stdout"]).splitlines():
            stripped = line.strip()
            if "=>" not in stripped:
                continue
            name, resolved = [part.strip() for part in stripped.split("=>", maxsplit=1)]
            soname = name.split()[0]
            sonames.setdefault(soname, resolved)
            if any(term in soname for term in ("libvulkan", "libEGL", "libGLX", "libnvidia")):
                lines.append(stripped)
    return lines, sonames


def _icd_inventory(paths: Sequence[Path], sonames: Mapping[str, str]) -> list[dict[str, object]]:
    rows = []
    for path in sorted({candidate.resolve() for candidate in paths if candidate.is_file()}):
        parsed = parse_icd_manifest(path)
        library = parsed["library_path"]
        resolved_text = library if Path(library).is_absolute() else sonames.get(library)
        resolved = Path(resolved_text) if resolved_text else None
        rows.append(
            {
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                **parsed,
                "resolved_library_path": resolved.as_posix() if resolved else None,
                "referenced_library_exists": resolved is not None and resolved.is_file(),
            }
        )
    return rows


def _egl_vendor_inventory(
    paths: Sequence[Path], sonames: Mapping[str, str]
) -> list[dict[str, object]]:
    rows = []
    for path in sorted({candidate.resolve() for candidate in paths if candidate.is_file()}):
        document = _read_json(path)
        icd = document.get("ICD")
        library = icd.get("library_path") if isinstance(icd, dict) else None
        resolved_text = (
            library
            if isinstance(library, str) and Path(library).is_absolute()
            else sonames.get(str(library))
        )
        resolved = Path(resolved_text) if resolved_text else None
        rows.append(
            {
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "library_path": library,
                "resolved_library_path": resolved.as_posix() if resolved else None,
                "referenced_library_exists": resolved is not None and resolved.is_file(),
            }
        )
    return rows


def _package_versions() -> dict[str, object]:
    packages: dict[str, str] = {}
    locations: dict[str, str] = {}
    for distribution in ("mani-skill", "sapien", "torch", "numpy", "h5py"):
        try:
            packages[distribution] = importlib_metadata.version(distribution)
            locations[distribution] = str(
                importlib_metadata.distribution(distribution).locate_file("")
            )
        except importlib_metadata.PackageNotFoundError:
            packages[distribution] = "not_installed"
            locations[distribution] = "unavailable"
    return {
        "python_version": platform.python_version(),
        "python_executable": Path(sys.executable).resolve().as_posix(),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "packages": packages,
        "package_locations": locations,
    }


def _rendering_inventory(conda_prefix: Path) -> dict[str, object]:
    system_icd_paths = [
        *Path("/etc/vulkan/icd.d").glob("*.json"),
        *Path("/usr/share/vulkan/icd.d").glob("*.json"),
    ]
    egl_paths = [
        *Path("/etc/glvnd/egl_vendor.d").glob("*.json"),
        *Path("/usr/share/glvnd/egl_vendor.d").glob("*.json"),
    ]
    packaged_icds = sorted(
        (
            conda_prefix
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        ).rglob("*icd*.json")
    )
    lines, sonames = _ldconfig_inventory()
    icds = _icd_inventory(system_icd_paths, sonames)
    egl_vendors = _egl_vendor_inventory(egl_paths, sonames)
    packaged_rows = [
        {
            "path": path.resolve().as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in packaged_icds
    ]
    selected_paths = {str(row["path"]): row for row in icds}
    checks = {
        "primary_icd_present": PRIMARY_ICD_PATH in selected_paths,
        "primary_icd_library_present": bool(
            selected_paths.get(PRIMARY_ICD_PATH, {}).get("referenced_library_exists")
        ),
        "egl_vendor_present": any(row["path"] == EGL_VENDOR_PATH for row in egl_vendors),
        "vulkan_loader_present": any("libvulkan.so.1" in line for line in lines),
        "nvidia_driver_library_present": any("libnvidia" in line for line in lines),
    }
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1r-rendering-inventory-v0",
            "created_at_utc": _timestamp(),
            "system_icd_manifests": icds,
            "egl_vendor_manifests": egl_vendors,
            "relevant_loader_and_driver_libraries": lines,
            "packaged_python_icd_files": packaged_rows,
            "runtime": _package_versions(),
            "inherited_render_environment": {
                name: os.environ.get(name) for name in RENDER_ENVIRONMENT_NAMES
            },
            "vulkaninfo_binary": shutil.which("vulkaninfo"),
            "vulkaninfo_version": _command(["vulkaninfo", "--version"]),
            "checks": checks,
            "passed": all(checks.values()),
        }
    )


def _candidate_manifest(inventory: Mapping[str, object]) -> dict[str, object]:
    rows = []
    icds = cast(Sequence[Mapping[str, object]], inventory["system_icd_manifests"])
    by_path = {str(row["path"]): row for row in icds}
    for candidate in candidate_protocol():
        row = dict(candidate)
        icd_path = candidate["vk_icd_filenames"]
        icd = by_path.get(str(icd_path)) if isinstance(icd_path, str) else None
        row.update(
            {
                "icd_sha256": icd.get("sha256") if icd else None,
                "icd_library_path": icd.get("library_path") if icd else None,
                "resolved_library_path": icd.get("resolved_library_path") if icd else None,
                "referenced_library_exists": (
                    icd.get("referenced_library_exists") if icd else None
                ),
                "execution_status": "frozen_not_run",
            }
        )
        rows.append(row)
    primary = rows[0]
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1r-candidate-protocol-v0",
            "candidate_count": len(rows),
            "candidates": rows,
            "primary_candidate_id": CandidateKind.PRIMARY.value,
            "primary_identity_unambiguous": primary["icd_sha256"] is not None
            and primary["referenced_library_exists"] is True,
            "arbitrary_combinations_forbidden": True,
            "system_files_modified": False,
            "frozen_before_execution": True,
            "passed": len(rows) == 3
            and primary["icd_sha256"] is not None
            and primary["referenced_library_exists"] is True,
        }
    )


def prepare_audit(
    *,
    repo_root: Path,
    frozen_root: Path,
    evidence_root: Path,
    conda_prefix: Path,
) -> dict[str, object]:
    """Verify immutable inputs, inventory the host, and freeze candidates."""

    repo_root = repo_root.resolve()
    frozen_root = frozen_root.resolve()
    evidence_root = evidence_root.resolve()
    conda_prefix = conda_prefix.resolve()
    if evidence_root.exists() and any(evidence_root.iterdir()):
        raise Phase2B61RRuntimeError("evidence root must be new and empty")
    evidence_root.mkdir(parents=True, exist_ok=True)
    repository = _repository_audit(repo_root)
    phase2b6 = _verify_artifact_manifest(repo_root / "artifacts" / "langmani_v2" / "phase_2b6")
    phase2b6_1 = _verify_artifact_manifest(repo_root / "artifacts" / "langmani_v2" / "phase_2b6_1")
    frozen = _verify_frozen_relevant_files(repo_root, frozen_root)
    prior = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1r-prior-evidence-v0",
            "phase2b6": phase2b6,
            "phase2b6_1": phase2b6_1,
            "frozen_partial_root": frozen,
            "prior_result": "RESULT_D",
            "prior_primary_classification": "insufficient_evidence",
            "prior_first_failure_layer": "environment_construction",
            "prior_result_reinterpreted": False,
            "passed": phase2b6["passed"] is True
            and phase2b6_1["passed"] is True
            and frozen["passed"] is True,
        }
    )
    gpu = _gpu_audit()
    processes = _process_audit()
    disk = _disk_audit((repo_root, evidence_root, frozen_root))
    repository_environment = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1r-repository-environment-v0",
            "repository": repository,
            "server_checkout": repo_root.as_posix(),
            "prior_result_identity": {
                "phase": "Phase 2B.6.1",
                "result": "RESULT_D",
                "source_commit": SOURCE_COMMIT,
            },
            "gpu": gpu,
            "process_state": processes,
            "disk": disk,
            "authorization_state": authorization_state(),
            "passed": repository["passed"] is True
            and prior["passed"] is True
            and gpu["passed"] is True
            and processes["passed"] is True
            and disk["passed"] is True,
        }
    )
    inventory = _rendering_inventory(conda_prefix)
    candidates = _candidate_manifest(inventory)
    icd_hashes = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1r-icd-hashes-v0",
            "files": [
                {
                    "path": row["path"],
                    "sha256": row["sha256"],
                    "size_bytes": row["size_bytes"],
                }
                for row in cast(Sequence[Mapping[str, object]], inventory["system_icd_manifests"])
            ],
            "system_icd_files_modified": False,
            "passed": True,
        }
    )
    protocol = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1r-preflight-protocol-v0",
            "task_id": TASK_ID,
            "allowed_operations": [
                "vulkaninfo --summary",
                "minimal SAPIEN RenderSystem construction and release",
                "gym.make StackCube-v1",
                "static environment contract inspection",
                "environment close",
            ],
            "forbidden_operations": [
                "environment reset",
                "environment step",
                "robot action submission",
                "trajectory replay",
                "dataset writer",
                "production resume",
                "policy loading or training",
            ],
            "fresh_process_run_count": 3,
            "primary_candidate_id": CandidateKind.PRIMARY.value,
            "negative_control_candidate_id": CandidateKind.DEFAULT.value,
            "hard_stop_on_primary_failure": True,
            "phase2b6_production_resume_authorized": False,
            "phase2b6_1_forensic_restart_authorized": False,
            "passed": True,
        }
    )
    documents = {
        "repository_environment_audit.json": repository_environment,
        "phase2b6_1_evidence_verification.json": prior,
        "rendering_stack_inventory.json": inventory,
        "candidate_icd_manifest.json": candidates,
        "icd_hashes.json": icd_hashes,
        "frozen_preflight_protocol.json": protocol,
        "authorization_state.json": authorization_state(),
    }
    for name, document in documents.items():
        _write_json(evidence_root / name, document)
    passed = (
        repository_environment["passed"] is True
        and inventory["passed"] is True
        and candidates["passed"] is True
    )
    if not passed:
        raise Phase2B61RRuntimeError("repository, evidence, or rendering inventory gate failed")
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1r-audit-stage-v0",
            "evidence_root": evidence_root.as_posix(),
            "files_written": sorted(documents),
            "passed": True,
        }
    )


def _parse_vulkan_summary(text: str) -> dict[str, object]:
    devices = []
    current: dict[str, str] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("GPU") and line.endswith(":"):
            current = {"index": line[:-1]}
            devices.append(current)
        elif current is not None and "=" in line:
            key, value = [part.strip() for part in line.split("=", maxsplit=1)]
            if key in {
                "deviceName",
                "deviceUUID",
                "driverName",
                "driverInfo",
                "deviceType",
                "apiVersion",
            }:
                current[key] = value
    return {
        "devices": devices,
        "selected_device_names": [row["deviceName"] for row in devices if "deviceName" in row],
        "intended_gpu_listed": any(
            EXPECTED_GPU_NAME.lower() in row.get("deviceName", "").lower() for row in devices
        ),
        "software_renderer_listed": any(
            "llvmpipe" in row.get("deviceName", "").lower()
            or "software" in row.get("deviceName", "").lower()
            for row in devices
        ),
    }


def _device_snapshot(device: object) -> dict[str, object]:
    def value(name: str) -> object:
        try:
            candidate = getattr(device, name)
            return candidate() if callable(candidate) else candidate
        except Exception as error:  # noqa: BLE001 - third-party diagnostic boundary
            return f"{type(error).__name__}: {error}"

    return {
        "name": str(value("name")),
        "pci_string": str(value("pci_string")),
        "cuda_id": value("cuda_id"),
        "is_cpu": value("is_cpu"),
        "is_cuda": value("is_cuda"),
        "can_present": value("can_present"),
        "can_render": value("can_render"),
    }


def _minimal_sapien_probe() -> dict[str, object]:
    started = time.perf_counter()
    try:
        import sapien  # type: ignore[import-untyped]

        render_system = sapien.render.RenderSystem("cuda")
        device = _device_snapshot(render_system.device)
        passed = device_identity_matches(device)
        del render_system
        gc.collect()
        return {
            "device": device,
            "correct_device_selected": passed,
            "exception": None,
            "wall_clock_seconds": time.perf_counter() - started,
            "passed": passed,
        }
    except Exception as error:  # noqa: BLE001 - third-party diagnostic boundary
        return {
            "device": None,
            "correct_device_selected": False,
            "exception": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "wall_clock_seconds": time.perf_counter() - started,
            "passed": False,
        }


def _render_system_from_environment(base: object) -> object | None:
    direct = getattr(base, "render_system", None)
    if direct is not None:
        return direct
    scene = getattr(base, "scene", None)
    sub_scenes = getattr(scene, "sub_scenes", None)
    if isinstance(sub_scenes, Sequence):
        for sub_scene in sub_scenes:
            render_system = getattr(sub_scene, "render_system", None)
            if render_system is not None:
                return render_system
    return None


def _zero_step_stackcube_probe(source_root: Path) -> dict[str, object]:
    started = time.perf_counter()
    explicit_reset_count = 0
    explicit_step_count = 0
    action_submission_count = 0
    environment: Any | None = None
    close_succeeded = False
    try:
        import gymnasium as gym
        import mani_skill.envs  # type: ignore[import-untyped]  # noqa: F401

        metadata_path = source_root / "expanded" / TASK_ID / "motionplanning" / "trajectory.json"
        metadata = _read_json(metadata_path)
        env_info = metadata.get("env_info")
        if not isinstance(env_info, dict) or not isinstance(env_info.get("env_kwargs"), dict):
            raise Phase2B61RRuntimeError("pinned StackCube metadata lacks env kwargs")
        kwargs = dict(cast(Mapping[str, object], env_info["env_kwargs"]))
        kwargs.update(
            {
                "obs_mode": EXPECTED_OBS_MODE,
                "reward_mode": "none",
                "render_mode": None,
                "sim_backend": EXPECTED_SIM_BACKEND,
                "render_backend": EXPECTED_RENDER_BACKEND,
                "sensor_configs": {
                    "width": EXPECTED_SENSOR_SIZE[0],
                    "height": EXPECTED_SENSOR_SIZE[1],
                },
                "num_envs": 1,
            }
        )
        environment = gym.make(TASK_ID, **cast(Any, kwargs))
        base: Any = environment.unwrapped
        action_space = environment.action_space
        observation_space = environment.observation_space
        raw_action_shape = action_space.shape
        if raw_action_shape is None:
            raise Phase2B61RRuntimeError("StackCube action space shape is unavailable")
        action_shape = tuple(int(value) for value in raw_action_shape)
        control_mode = str(base.control_mode)
        obs_mode = str(base.obs_mode)
        environment_spec = getattr(environment, "spec", None) or getattr(base, "spec", None)
        render_system = _render_system_from_environment(base)
        if render_system is None:
            raise Phase2B61RRuntimeError("StackCube render system identity is unavailable")
        render_device = _device_snapshot(cast(Any, render_system).device)
        action_dtype = str(action_space.dtype)
        checks = {
            "task_id": getattr(environment_spec, "id", None) == TASK_ID,
            "control_mode": control_mode == EXPECTED_CONTROL_MODE,
            "control_frequency_hz": int(base.control_freq) == 20,
            "action_shape": action_shape == EXPECTED_ACTION_SHAPE,
            "action_dtype": action_dtype == "float32",
            "obs_mode": obs_mode == EXPECTED_OBS_MODE,
            "sim_backend": kwargs["sim_backend"] == EXPECTED_SIM_BACKEND,
            "render_backend": kwargs["render_backend"] == EXPECTED_RENDER_BACKEND,
            "sensor_size": kwargs["sensor_configs"]
            == {"width": EXPECTED_SENSOR_SIZE[0], "height": EXPECTED_SENSOR_SIZE[1]},
            "render_device": device_identity_matches(render_device),
        }
        enforce_zero_step_counts(
            explicit_reset_count=explicit_reset_count,
            explicit_step_count=explicit_step_count,
            action_submission_count=action_submission_count,
        )
        environment.close()
        close_succeeded = True
        environment = None
        return {
            "constructor_kwargs": kwargs,
            "constructor_may_initialize_scene_internally": True,
            "explicit_reset_count": explicit_reset_count,
            "explicit_step_count": explicit_step_count,
            "action_submission_count": action_submission_count,
            "policy_frame_count": 0,
            "action_shape": list(action_shape),
            "action_dtype": action_dtype,
            "action_space": repr(action_space),
            "observation_space": repr(observation_space),
            "control_mode": control_mode,
            "obs_mode": obs_mode,
            "render_device": render_device,
            "checks": checks,
            "close_succeeded": close_succeeded,
            "exception": None,
            "wall_clock_seconds": time.perf_counter() - started,
            "passed": all(checks.values()) and close_succeeded,
        }
    except Exception as error:  # noqa: BLE001 - third-party diagnostic boundary
        if environment is not None:
            try:
                environment.close()
                close_succeeded = True
            except Exception:  # noqa: BLE001 - retain original construction failure
                close_succeeded = False
        return {
            "constructor_may_initialize_scene_internally": True,
            "explicit_reset_count": explicit_reset_count,
            "explicit_step_count": explicit_step_count,
            "action_submission_count": action_submission_count,
            "policy_frame_count": 0,
            "close_succeeded": close_succeeded,
            "exception": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "wall_clock_seconds": time.perf_counter() - started,
            "passed": False,
        }


def child_probe(
    *,
    candidate_id: str,
    source_root: Path,
    output: Path,
    run_index: int,
    run_sapien: bool,
    run_zero_step: bool,
) -> dict[str, object]:
    """Run one fresh child probe under a preregistered process environment."""

    candidate = candidate_by_id(candidate_id)
    if candidate_id == CandidateKind.PRIMARY.value:
        validate_primary_environment(os.environ)
    if run_zero_step and candidate["may_run_zero_step_stackcube"] is not True:
        raise Phase2B61RRuntimeError("candidate is not authorized for StackCube construction")
    if run_sapien and candidate["may_run_sapien_probe"] is not True:
        raise Phase2B61RRuntimeError("candidate is not authorized for a SAPIEN probe")
    vulkan = _command(["vulkaninfo", "--summary"], timeout_seconds=60.0)
    parsed = _parse_vulkan_summary(str(vulkan["stdout"]))
    vulkan_passed = (
        vulkan["passed"] is True
        and parsed["intended_gpu_listed"] is True
        and parsed["software_renderer_listed"] is False
    )
    first_failure: str | None = None if vulkan_passed else "vulkaninfo"
    sapien_result: dict[str, object] | None = None
    zero_step_result: dict[str, object] | None = None
    if vulkan_passed and run_sapien:
        sapien_result = _minimal_sapien_probe()
        if sapien_result["passed"] is not True:
            first_failure = "sapien_probe"
    if (
        vulkan_passed
        and run_sapien
        and sapien_result is not None
        and sapien_result["passed"] is True
        and run_zero_step
    ):
        zero_step_result = _zero_step_stackcube_probe(source_root.resolve())
        if zero_step_result["passed"] is not True:
            first_failure = "stackcube_zero_step_construction"
    icd_path = candidate["vk_icd_filenames"]
    report = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1r-child-run-v0",
            "created_at_utc": _timestamp(),
            "candidate_id": candidate_id,
            "run_index": run_index,
            "pid": os.getpid(),
            "python_executable": Path(sys.executable).resolve().as_posix(),
            "process_environment": {
                name: os.environ.get(name) for name in RENDER_ENVIRONMENT_NAMES
            },
            "icd_path": icd_path,
            "icd_sha256": (
                sha256_file(Path(str(icd_path)))
                if isinstance(icd_path, str) and Path(icd_path).is_file()
                else None
            ),
            "vulkaninfo": vulkan,
            "vulkan_summary": parsed,
            "sapien_probe": sapien_result,
            "zero_step_stackcube": zero_step_result,
            "first_failure_layer": first_failure,
            "explicit_reset_count": (
                zero_step_result.get("explicit_reset_count", 0) if zero_step_result else 0
            ),
            "explicit_step_count": (
                zero_step_result.get("explicit_step_count", 0) if zero_step_result else 0
            ),
            "action_submission_count": (
                zero_step_result.get("action_submission_count", 0) if zero_step_result else 0
            ),
            "forensic_replay_started": False,
            "production_resumed": False,
            "policy_or_optimizer_loaded": False,
            "passed": vulkan_passed
            and (not run_sapien or sapien_result is not None and sapien_result["passed"] is True)
            and (
                not run_zero_step
                or zero_step_result is not None
                and zero_step_result["passed"] is True
            ),
        }
    )
    _write_json(output.resolve(), report)
    return report


def _flatten_run(run: Mapping[str, object]) -> dict[str, object]:
    parsed = cast(Mapping[str, object], run["vulkan_summary"])
    names = cast(Sequence[str], parsed.get("selected_device_names", []))
    sapien = cast(Mapping[str, object] | None, run.get("sapien_probe"))
    sapien_device = (
        cast(Mapping[str, object], sapien["device"])
        if sapien is not None and isinstance(sapien.get("device"), dict)
        else {}
    )
    zero = cast(Mapping[str, object] | None, run.get("zero_step_stackcube"))
    zero_device = (
        cast(Mapping[str, object], zero["render_device"])
        if zero is not None and isinstance(zero.get("render_device"), dict)
        else {}
    )
    render_device = zero_device or sapien_device
    return {
        "pid": run.get("pid"),
        "passed": run.get("passed"),
        "icd_sha256": run.get("icd_sha256"),
        "vulkan_device_name": names[0] if len(names) == 1 else None,
        "render_device_name": render_device.get("name"),
        "render_device_pci": render_device.get("pci_string"),
        "action_shape": zero.get("action_shape", []) if zero else [],
        "control_mode": zero.get("control_mode") if zero else None,
        "obs_mode": zero.get("obs_mode") if zero else None,
    }


def _post_child_cleanup(child_pid: int) -> dict[str, object]:
    processes = _process_audit(excluded_pids={child_pid})
    gpu = _gpu_audit()
    child_alive = Path(f"/proc/{child_pid}").exists()
    unexpected_gpu_workloads = bool(gpu["compute_processes"])
    return {
        "child_pid": child_pid,
        "child_process_alive": child_alive,
        "process_audit": processes,
        "gpu_audit": gpu,
        "unexpected_gpu_workloads": unexpected_gpu_workloads,
        "tmux_session_created": False,
        "passed": not child_alive and processes["passed"] is True and not unexpected_gpu_workloads,
    }


def _run_child(
    *,
    repo_root: Path,
    conda_prefix: Path,
    source_root: Path,
    output_root: Path,
    candidate_id: str,
    run_index: int,
    run_sapien: bool,
    run_zero_step: bool,
) -> tuple[dict[str, Any], dict[str, object], dict[str, object]]:
    run_directory = output_root / "runs" / f"{candidate_id}_{run_index}"
    run_directory.mkdir(parents=True, exist_ok=False)
    result_path = run_directory / "result.json"
    environment = clean_child_environment(
        conda_prefix=conda_prefix,
        home=Path("/root"),
        xdg_runtime_dir=output_root / "xdg-runtime",
        candidate_id=candidate_id,
        inherited=os.environ,
    )
    Path(environment["XDG_RUNTIME_DIR"]).mkdir(mode=0o700, parents=True, exist_ok=True)
    command = [
        (conda_prefix / "bin" / "python").as_posix(),
        (repo_root / "environment" / "run_v2_phase2b6_1r.py").as_posix(),
        "child-probe",
        "--repo-root",
        repo_root.as_posix(),
        "--source-root",
        source_root.as_posix(),
        "--output",
        result_path.as_posix(),
        "--candidate-id",
        candidate_id,
        "--run-index",
        str(run_index),
    ]
    if run_sapien:
        command.append("--run-sapien")
    if run_zero_step:
        command.append("--run-zero-step")
    started = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=120.0,
    )
    launch = {
        "command": command,
        "candidate_id": candidate_id,
        "run_index": run_index,
        "pid": process.pid if hasattr(process, "pid") else None,
        "exit_code": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
        "wall_clock_seconds": time.perf_counter() - started,
    }
    if not result_path.is_file():
        raise Phase2B61RRuntimeError(
            f"child process failed before writing its result: {candidate_id}/{run_index}"
        )
    result = _read_json(result_path)
    child_pid = int(result["pid"])
    cleanup = _post_child_cleanup(child_pid)
    _write_json(run_directory / "launch.json", fingerprinted(launch))
    _write_json(run_directory / "cleanup.json", fingerprinted(cleanup))
    return result, launch, cleanup


def execute_preflight(
    *,
    repo_root: Path,
    source_root: Path,
    frozen_root: Path,
    evidence_root: Path,
    output_root: Path,
    conda_prefix: Path,
    launcher_path: Path,
) -> dict[str, object]:
    """Execute the primary 3x fresh-process gate and bounded controls."""

    repo_root = repo_root.resolve()
    source_root = source_root.resolve()
    frozen_root = frozen_root.resolve()
    evidence_root = evidence_root.resolve()
    output_root = output_root.resolve()
    conda_prefix = conda_prefix.resolve()
    launcher_path = launcher_path.resolve()
    if output_root.exists():
        raise Phase2B61RRuntimeError("runtime output root must not already exist")
    output_root.mkdir(parents=True)
    required_evidence = (
        "repository_environment_audit.json",
        "phase2b6_1_evidence_verification.json",
        "rendering_stack_inventory.json",
        "candidate_icd_manifest.json",
        "frozen_preflight_protocol.json",
        "authorization_state.json",
    )
    if any(not (evidence_root / name).is_file() for name in required_evidence):
        raise Phase2B61RRuntimeError("audit stage evidence is incomplete")
    repository = _repository_audit(repo_root)
    prior = _read_json(evidence_root / "phase2b6_1_evidence_verification.json")
    candidates = _read_json(evidence_root / "candidate_icd_manifest.json")
    if (
        repository["passed"] is not True
        or prior["passed"] is not True
        or candidates["passed"] is not True
    ):
        raise Phase2B61RRuntimeError("execution preconditions are not satisfied")
    before_frozen = _verify_frozen_relevant_files(repo_root, frozen_root)
    primary_runs: list[dict[str, Any]] = []
    cleanup_rows: list[dict[str, object]] = []
    for run_index in range(1, 4):
        result, _, cleanup = _run_child(
            repo_root=repo_root,
            conda_prefix=conda_prefix,
            source_root=source_root,
            output_root=output_root,
            candidate_id=CandidateKind.PRIMARY.value,
            run_index=run_index,
            run_sapien=True,
            run_zero_step=True,
        )
        primary_runs.append(result)
        cleanup_rows.append(cleanup)
        if result["passed"] is not True or cleanup["passed"] is not True:
            break
    repeatability = repeatability_passed([_flatten_run(run) for run in primary_runs])
    controls: list[dict[str, Any]] = []
    negative_control: dict[str, Any] | None = None
    if repeatability and all(row["passed"] is True for row in cleanup_rows):
        secondary, _, secondary_cleanup = _run_child(
            repo_root=repo_root,
            conda_prefix=conda_prefix,
            source_root=source_root,
            output_root=output_root,
            candidate_id=CandidateKind.SECONDARY.value,
            run_index=1,
            run_sapien=False,
            run_zero_step=False,
        )
        controls.append(secondary)
        cleanup_rows.append(secondary_cleanup)
        negative_control, _, default_cleanup = _run_child(
            repo_root=repo_root,
            conda_prefix=conda_prefix,
            source_root=source_root,
            output_root=output_root,
            candidate_id=CandidateKind.DEFAULT.value,
            run_index=1,
            run_sapien=True,
            run_zero_step=False,
        )
        controls.append(negative_control)
        cleanup_rows.append(default_cleanup)
    after_frozen = _verify_frozen_relevant_files(repo_root, frozen_root)
    frozen_unchanged = before_frozen == after_frozen
    first_failure = next(
        (
            str(run["first_failure_layer"])
            for run in primary_runs
            if run.get("first_failure_layer") is not None
        ),
        None,
    )
    primary_vulkan = len(primary_runs) == 3 and all(
        cast(Mapping[str, object], run["vulkaninfo"])["passed"] is True
        and cast(Mapping[str, object], run["vulkan_summary"])["intended_gpu_listed"] is True
        for run in primary_runs
    )
    primary_sapien = len(primary_runs) == 3 and all(
        isinstance(run.get("sapien_probe"), dict)
        and cast(Mapping[str, object], run["sapien_probe"])["passed"] is True
        for run in primary_runs
    )
    primary_zero = len(primary_runs) == 3 and all(
        isinstance(run.get("zero_step_stackcube"), dict)
        and cast(Mapping[str, object], run["zero_step_stackcube"])["passed"] is True
        for run in primary_runs
    )
    correct_device = len(primary_runs) == 3 and all(
        device_identity_matches(
            cast(
                Mapping[str, object],
                cast(Mapping[str, object], run["zero_step_stackcube"])["render_device"],
            )
        )
        for run in primary_runs
        if isinstance(run.get("zero_step_stackcube"), dict)
    )
    cleanup_passed = bool(cleanup_rows) and all(row["passed"] is True for row in cleanup_rows)
    classification = classify_result(
        host_evidence_sufficient=repository["passed"] is True
        and prior["passed"] is True
        and frozen_unchanged,
        vulkan_passed=primary_vulkan,
        sapien_passed=primary_sapien,
        zero_step_passed=primary_zero,
        repeatability_validated=repeatability,
        correct_device_selected=correct_device,
        cleanup_passed=cleanup_passed,
        first_failure_layer=first_failure,
    )
    flattened = [_flatten_run(run) for run in primary_runs]
    vulkan_results = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1r-vulkan-results-v0",
            "primary_runs": [
                {
                    "run_index": run["run_index"],
                    "vulkaninfo": run["vulkaninfo"],
                    "summary": run["vulkan_summary"],
                    "icd_path": run["icd_path"],
                    "icd_sha256": run["icd_sha256"],
                }
                for run in primary_runs
            ],
            "candidate_controls": [
                {
                    "candidate_id": run["candidate_id"],
                    "vulkaninfo": run["vulkaninfo"],
                    "summary": run["vulkan_summary"],
                    "icd_path": run["icd_path"],
                    "icd_sha256": run["icd_sha256"],
                }
                for run in controls
            ],
            "passed": primary_vulkan,
        }
    )
    sapien_results = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1r-sapien-results-v0",
            "primary_runs": [run.get("sapien_probe") for run in primary_runs],
            "negative_control": (
                negative_control.get("sapien_probe") if negative_control else None
            ),
            "passed": primary_sapien,
        }
    )
    zero_results = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1r-zero-step-results-v0",
            "task_id": TASK_ID,
            "runs": [run.get("zero_step_stackcube") for run in primary_runs],
            "explicit_reset_count": sum(int(run["explicit_reset_count"]) for run in primary_runs),
            "explicit_step_count": sum(int(run["explicit_step_count"]) for run in primary_runs),
            "action_submission_count": sum(
                int(run["action_submission_count"]) for run in primary_runs
            ),
            "forensic_replay_started": False,
            "passed": primary_zero,
        }
    )
    repeatability_result = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1r-repeatability-v0",
            "runs": flattened,
            "fresh_process_count": len({row["pid"] for row in flattened}),
            "required_fresh_process_count": 3,
            "passed": repeatability,
        }
    )
    negative = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1r-negative-control-v0",
            "candidate": negative_control,
            "failure_required": False,
            "system_manipulated_to_force_failure": False,
            "run_only_after_primary_passed": negative_control is not None,
            "passed": negative_control is not None,
        }
    )
    device_audit = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1r-device-binding-v0",
            "expected_device_name": EXPECTED_GPU_NAME,
            "runs": flattened,
            "nvidia_smi": _gpu_audit(),
            "software_or_wrong_device_selected": not correct_device,
            "passed": correct_device,
        }
    )
    cleanup_audit = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1r-cleanup-v0",
            "runs": cleanup_rows,
            "no_leaked_child_processes": all(
                row["child_process_alive"] is False for row in cleanup_rows
            ),
            "no_unexpected_gpu_workloads": all(
                row["unexpected_gpu_workloads"] is False for row in cleanup_rows
            ),
            "no_tmux_session_created": True,
            "passed": cleanup_passed,
        }
    )
    runtime = _package_versions()
    gpu = _gpu_audit()
    launcher_manifest = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1r-launcher-runtime-v0",
            "launcher_path": launcher_path.as_posix(),
            "launcher_sha256": sha256_file(launcher_path),
            "python_entrypoint": (repo_root / "environment" / "run_v2_phase2b6_1r.py").as_posix(),
            "python_entrypoint_sha256": sha256_file(
                repo_root / "environment" / "run_v2_phase2b6_1r.py"
            ),
            "conda_environment": conda_prefix.name,
            "runtime": runtime,
            "driver": (
                cast(Sequence[Mapping[str, object]], gpu["devices"])[0].get("driver_version")
                if gpu["devices"]
                else None
            ),
            "selected_icd": PRIMARY_ICD_PATH,
            "selected_icd_sha256": (
                sha256_file(Path(PRIMARY_ICD_PATH)) if Path(PRIMARY_ICD_PATH).is_file() else None
            ),
            "selected_gpu": (
                cast(Sequence[Mapping[str, object]], gpu["devices"])[0] if gpu["devices"] else None
            ),
            "renderer_configuration": {
                "task_id": TASK_ID,
                "obs_mode": EXPECTED_OBS_MODE,
                "control_mode": EXPECTED_CONTROL_MODE,
                "sim_backend": EXPECTED_SIM_BACKEND,
                "render_backend": EXPECTED_RENDER_BACKEND,
                "sensor_size": list(EXPECTED_SENSOR_SIZE),
                "num_envs": 1,
            },
            "manual_interactive_exports_required": False,
            "passed": True,
        }
    )
    remote = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1r-remote-execution-v0",
            "repository": repository,
            "output_root": output_root.as_posix(),
            "evidence_root": evidence_root.as_posix(),
            "source_root": source_root.as_posix(),
            "frozen_root": frozen_root.as_posix(),
            "frozen_before": before_frozen,
            "frozen_after": after_frozen,
            "frozen_root_unchanged": frozen_unchanged,
            "primary_run_count": len(primary_runs),
            "negative_control_run_count": 1 if negative_control else 0,
            "forensic_replay_started": False,
            "production_resumed": False,
            "dataset_writer_started": False,
            "training_started": False,
            "optimizer_created": False,
            "backward_passes": 0,
            "optimizer_steps": 0,
            "passed": frozen_unchanged,
        }
    )
    eligibility = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1r-eligibility-v0",
            "result": classification["result"],
            "phase2b6_1_forensic_restart_eligible": classification[
                "phase2b6_1_forensic_restart_eligible"
            ],
            "eligibility_is_authorization": False,
            "phase2b6_1_forensic_restart_authorized": False,
            "phase2b6_production_resume_authorized": False,
            "passed": True,
        }
    )
    documents = {
        "launcher_runtime_manifest.json": launcher_manifest,
        "vulkaninfo_results.json": vulkan_results,
        "sapien_probe_results.json": sapien_results,
        "zero_step_stackcube_results.json": zero_results,
        "fresh_process_repeatability_result.json": repeatability_result,
        "negative_control_result.json": negative,
        "device_binding_audit.json": device_audit,
        "cleanup_audit.json": cleanup_audit,
        "result_classification.json": classification,
        "eligibility_state.json": eligibility,
        "authorization_state_final.json": authorization_state(),
        "remote_execution_audit.json": remote,
    }
    for name, document in documents.items():
        _write_json(evidence_root / name, document)
    return classification


def write_artifact_manifest(evidence_root: Path) -> dict[str, object]:
    """Hash all compact Phase 2B.6.1-R evidence except the manifest itself."""

    evidence_root = evidence_root.resolve()
    files = [
        {
            "path": path.relative_to(evidence_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(evidence_root.rglob("*.json"))
        if path.name != "artifact_manifest.json"
    ]
    manifest = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1r-artifact-manifest-v0",
            "files": files,
            "file_count": len(files),
            "large_driver_dumps_committed": False,
            "trajectory_or_dataset_bytes_committed": False,
            "passed": bool(files),
        }
    )
    _write_json(evidence_root / "artifact_manifest.json", manifest)
    return manifest


__all__ = [
    "Phase2B61RRuntimeError",
    "child_probe",
    "execute_preflight",
    "prepare_audit",
    "write_artifact_manifest",
]
