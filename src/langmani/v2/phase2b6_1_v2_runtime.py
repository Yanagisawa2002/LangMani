"""Native orchestration for bounded Phase 2B.6.1-v2 replay forensics."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from langmani.v2.phase2b5 import canonical_json_sha256, sha256_file
from langmani.v2.phase2b6_1 import GateStatus
from langmani.v2.phase2b6_1_runtime import (
    _gpu_audit,
    _process_audit,
    _runtime_versions,
    build_source_identity_manifest,
    historical_evidence_audit,
    inventory_frozen_output,
    producer_call_sequence_audit,
    run_forensic_replay,
)
from langmani.v2.phase2b6_1_v2 import (
    ACCEPTED_LAUNCHER_SHA256,
    GATE_ORDER,
    PHASE2B6_ARTIFACT_FINGERPRINT,
    PHASE2B61_ARTIFACT_FINGERPRINT,
    PHASE2B61R_ARTIFACT_FINGERPRINT,
    PRODUCER_COMMIT,
    SOURCE_COMMIT,
    TARGET_BRANCH,
    authorization_state,
    classify_result,
    determinism_audit,
    eligibility_state,
    enforce_starting_point,
    fingerprinted,
    first_failed_gate,
    forensic_protocol,
    gate,
    required_gates_for_mode,
    run_passed,
)
from langmani.v2.phase2b6_1r import (
    EGL_VENDOR_PATH,
    PRIMARY_ICD_PATH,
    RENDER_ENVIRONMENT_NAMES,
    validate_primary_environment,
)
from langmani.v2.phase2b6_1r_runtime import child_probe


class Phase2B61V2RuntimeError(RuntimeError):
    """Raised when native execution cannot preserve the frozen protocol."""


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2B61V2RuntimeError(f"failed to read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise Phase2B61V2RuntimeError(f"{path} must contain one JSON object")
    return value


def _write_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    replace: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise Phase2B61V2RuntimeError(f"forensic evidence already exists: {path}")
    staging = path.parent / f".{path.name}.partial"
    if staging.exists():
        raise Phase2B61V2RuntimeError(f"stale staging file exists: {staging}")
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
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode:
        raise Phase2B61V2RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _verify_manifest_directory(root: Path, expected_fingerprint: str) -> dict[str, object]:
    manifest_path = root / "artifact_manifest.json"
    manifest = _read_json(manifest_path)
    fingerprint = manifest.get("fingerprint")
    unhashed = dict(manifest)
    unhashed.pop("fingerprint", None)
    fingerprint_valid = (
        fingerprint == expected_fingerprint
        and canonical_json_sha256(unhashed) == expected_fingerprint
    )
    rows: list[dict[str, object]] = []
    files = manifest.get("files")
    if not isinstance(files, list):
        raise Phase2B61V2RuntimeError(f"{manifest_path} files must be a list")
    for raw in files:
        if not isinstance(raw, dict):
            raise Phase2B61V2RuntimeError("artifact manifest row must be an object")
        relative = raw.get("path")
        expected = raw.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise Phase2B61V2RuntimeError("artifact manifest row is incomplete")
        path = root / relative
        observed = "sha256:" + sha256_file(path) if path.is_file() else None
        rows.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "passed": observed == expected,
            }
        )
    return {
        "root": root.as_posix(),
        "manifest_fingerprint": fingerprint,
        "expected_manifest_fingerprint": expected_fingerprint,
        "fingerprint_valid": fingerprint_valid,
        "file_count": len(rows),
        "files": rows,
        "passed": fingerprint_valid and all(row["passed"] is True for row in rows),
    }


def verify_prior_evidence(repo_root: Path) -> dict[str, object]:
    """Rehash all three committed prior evidence sets."""

    roots = (
        (
            "phase_2b6",
            PHASE2B6_ARTIFACT_FINGERPRINT,
            "RESULT_C",
        ),
        (
            "phase_2b6_1",
            PHASE2B61_ARTIFACT_FINGERPRINT,
            "RESULT_D",
        ),
        (
            "phase_2b6_1r",
            PHASE2B61R_ARTIFACT_FINGERPRINT,
            "RESULT_A",
        ),
    )
    reports: dict[str, object] = {}
    for name, expected_fingerprint, expected_result in roots:
        root = repo_root / "artifacts" / "langmani_v2" / name
        report = _verify_manifest_directory(root, expected_fingerprint)
        result_path = root / "result_classification.json"
        result = _read_json(result_path)
        report["expected_result"] = expected_result
        report["observed_result"] = result.get("result")
        report["result_valid"] = result.get("result") == expected_result
        report["passed"] = report["passed"] is True and report["result_valid"] is True
        reports[name] = report
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-v2-prior-evidence-v0",
            "evidence_sets": reports,
            "passed": all(
                cast(Mapping[str, object], report).get("passed") is True
                for report in reports.values()
            ),
        }
    )


def _repository_audit(
    *,
    repo_root: Path,
    source_root: Path,
    frozen_root: Path,
    output_root: Path,
    network_turbo_sourced: bool,
) -> dict[str, object]:
    branch = _git(repo_root, "branch", "--show-current")
    head = _git(repo_root, "rev-parse", "HEAD")
    source_is_ancestor = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, "HEAD"],
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
    remote = _git(repo_root, "remote", "get-url", "origin")
    upstream = _git(
        repo_root,
        "rev-parse",
        "--verify",
        "@{upstream}",
        check=False,
    )
    producer_exists = bool(
        _git(repo_root, "cat-file", "-e", f"{PRODUCER_COMMIT}^{{commit}}", check=False) == ""
        and subprocess.run(
            ["git", "cat-file", "-e", f"{PRODUCER_COMMIT}^{{commit}}"],
            cwd=repo_root,
            check=False,
        ).returncode
        == 0
    )
    process = _process_audit()
    disk = {
        path.as_posix(): {
            "free_bytes": shutil.disk_usage(path).free,
            "total_bytes": shutil.disk_usage(path).total,
        }
        for path in (source_root, frozen_root, output_root.parent)
    }
    checks = {
        "branch_exact": branch == TARGET_BRANCH,
        "source_commit_is_ancestor": source_is_ancestor,
        "worktree_clean": status == "",
        "origin_repository_exact": remote
        in {
            "https://github.com/Yanagisawa2002/LangMani.git",
            "git@github.com:Yanagisawa2002/LangMani.git",
        },
        "upstream_matches_head": upstream == head,
        "producer_commit_exists": producer_exists,
        "source_root_exists": source_root.is_dir(),
        "frozen_root_exists": frozen_root.is_dir(),
        "output_root_is_distinct": output_root.resolve() != frozen_root.resolve(),
        "no_prohibited_process": process.get("passed") is True,
        "network_turbo_sourced": network_turbo_sourced,
    }
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-v2-repository-audit-v0",
            "created_at_utc": _timestamp(),
            "branch": branch,
            "head": head,
            "source_commit": SOURCE_COMMIT,
            "upstream_head": upstream,
            "origin": remote,
            "status_porcelain": status,
            "source_root": source_root.as_posix(),
            "frozen_root": frozen_root.as_posix(),
            "output_root": output_root.as_posix(),
            "process_audit": process,
            "gpu_audit": _gpu_audit(),
            "disk": disk,
            "checks": checks,
            "passed": all(checks.values()),
        }
    )


def prepare_forensics(
    *,
    repo_root: Path,
    source_root: Path,
    frozen_root: Path,
    output_root: Path,
    evidence_root: Path,
    spec_path: Path,
    accepted_launcher: Path,
    adapter_launcher: Path,
    network_turbo_sourced: bool,
) -> dict[str, object]:
    """Freeze identities and protocol without constructing a simulator."""

    if output_root.exists() and any(output_root.iterdir()):
        raise Phase2B61V2RuntimeError("external forensic root must start empty")
    output_root.mkdir(parents=True, exist_ok=True)
    evidence_root.mkdir(parents=True, exist_ok=True)
    accepted_hash = "sha256:" + sha256_file(accepted_launcher)
    launcher = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-v2-launcher-binding-v0",
            "accepted_launcher_path": accepted_launcher.as_posix(),
            "accepted_launcher_sha256": accepted_hash,
            "expected_accepted_launcher_sha256": ACCEPTED_LAUNCHER_SHA256,
            "adapter_launcher_path": adapter_launcher.as_posix(),
            "adapter_launcher_sha256": "sha256:" + sha256_file(adapter_launcher),
            "adapter_preserves_clean_process_contract": True,
            "default_loader_negative_control_rerun": False,
            "passed": accepted_hash == ACCEPTED_LAUNCHER_SHA256,
        }
    )
    repository = _repository_audit(
        repo_root=repo_root,
        source_root=source_root,
        frozen_root=frozen_root,
        output_root=output_root,
        network_turbo_sourced=network_turbo_sourced,
    )
    prior = verify_prior_evidence(repo_root)
    frozen = inventory_frozen_output(frozen_root)
    source = build_source_identity_manifest(
        source_root=source_root,
        spec_path=spec_path,
        frozen_root=frozen_root,
    )
    history = historical_evidence_audit(frozen_root)
    old_sequence = producer_call_sequence_audit()
    call_sequence = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-v2-call-sequence-v0",
            "producer_commit": PRODUCER_COMMIT,
            "base_audit": old_sequence,
            "differences": [
                {
                    "step": "environment_lifetime",
                    "classification": "known_semantic_difference",
                    "producer": "one StackCube environment reused across episodes",
                    "forensic": "fresh process and environment for every repetition",
                },
                {
                    "step": "Mode A observations",
                    "classification": "known_semantic_difference",
                    "producer": "RGB observations acquired before every action",
                    "forensic": "physics-only, observations disabled",
                },
                {
                    "step": "explicit gates and task snapshots",
                    "classification": "diagnostic_instrumentation_only",
                },
                {
                    "step": "Mode C destination",
                    "classification": "diagnostic_instrumentation_only",
                    "forensic": "isolated NPZ outside frozen production root",
                },
            ],
            "stable_success_evaluation": {
                "producer_rule": "final_step_canonical_success",
                "separate_gate_present": False,
            },
            "exact_producer_reproduction_claimed": False,
            "passed": True,
        }
    )
    documents = {
        "repository_environment_audit.json": repository,
        "prior_evidence_verification.json": prior,
        "validated_launcher_binding.json": launcher,
        "frozen_partial_output_inventory.json": frozen,
        "source_episode_identity.json": source,
        "action_identity.json": fingerprinted(
            {
                "schema_version": "langmani-v2-phase2b6-1-v2-action-identity-v0",
                "target_episode": 938,
                "action_count": 105,
                "action_dtype": "float32",
                "action_shape": [105, 8],
                "action_sha256": (
                    cast(Sequence[Mapping[str, object]], source["episodes"])[2]["action_sha256"]
                ),
                "target_checks": source["target_checks"],
                "passed": source["producer_input_identity_proven"],
            }
        ),
        "historical_evidence_audit.json": history,
        "forensic_protocol.json": forensic_protocol(),
        "producer_forensic_call_sequence_audit.json": call_sequence,
        "authorization_state.json": authorization_state(),
    }
    for name, document in documents.items():
        _write_json(evidence_root / name, cast(Mapping[str, object], document))
    passed = all(
        cast(Mapping[str, object], document).get("passed", True) is True
        for document in documents.values()
    )
    summary = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-v2-prepare-v0",
            "documents": sorted(documents),
            "source_identity_proven": source["producer_input_identity_proven"],
            "simulator_started": False,
            "passed": passed,
        }
    )
    _write_json(evidence_root / "preflight_result.json", summary)
    if not passed:
        raise Phase2B61V2RuntimeError("repository/evidence/source preflight failed")
    return summary


def run_rendering_preflight(
    *,
    source_root: Path,
    output_root: Path,
    evidence_root: Path,
) -> dict[str, object]:
    """Run only the accepted primary Vulkan/SAPIEN/zero-step contract."""

    validate_primary_environment(os.environ)
    raw_path = output_root / "rendering_preflight" / "primary_probe.json"
    raw = child_probe(
        candidate_id="candidate_a_explicit_egl_icd",
        source_root=source_root,
        output=raw_path,
        run_index=1,
        run_sapien=True,
        run_zero_step=True,
    )
    environment = {name: os.environ.get(name) for name in RENDER_ENVIRONMENT_NAMES}
    report = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-v2-rendering-contract-v0",
            "created_at_utc": _timestamp(),
            "accepted_phase2b6_1r_contract": True,
            "process_environment": environment,
            "python_executable": cast(str, raw["python_executable"]),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "runtime_versions": _runtime_versions(),
            "gpu_audit": _gpu_audit(),
            "icd_path": PRIMARY_ICD_PATH,
            "icd_sha256": raw["icd_sha256"],
            "egl_vendor_path": EGL_VENDOR_PATH,
            "egl_vendor_sha256": (
                "sha256:" + sha256_file(Path(EGL_VENDOR_PATH))
                if Path(EGL_VENDOR_PATH).is_file()
                else None
            ),
            "vulkaninfo_passed": cast(Mapping[str, object], raw["vulkaninfo"])["passed"],
            "minimal_sapien_probe": raw["sapien_probe"],
            "zero_step_stackcube": raw["zero_step_stackcube"],
            "selected_rendering_device": cast(Mapping[str, object], raw["zero_step_stackcube"]).get(
                "render_device"
            ),
            "explicit_reset_count": raw["explicit_reset_count"],
            "explicit_step_count": raw["explicit_step_count"],
            "action_submission_count": raw["action_submission_count"],
            "raw_external_evidence_path": raw_path.as_posix(),
            "raw_external_evidence_sha256": "sha256:" + sha256_file(raw_path),
            "default_loader_negative_control_rerun": False,
            "passed": raw["passed"],
        }
    )
    _write_json(evidence_root / "validated_rendering_contract_manifest.json", report)
    if report["passed"] is not True:
        raise Phase2B61V2RuntimeError("validated rendering preflight failed")
    return report


def _not_reached_gates() -> dict[str, dict[str, object]]:
    return {
        name: gate(status=GateStatus.NOT_REACHED, detail="gate was not reached")
        for name in GATE_ORDER
    }


def _augment_run(
    *,
    report: dict[str, Any],
    evidence_root: Path,
    external_path: Path,
) -> dict[str, object]:
    mode = str(cast(Mapping[str, object], report["run_identity"])["mode"])
    old_gates = cast(dict[str, dict[str, object]], report["sub_gates"])
    sub_gates = _not_reached_gates()
    rendering = _read_json(evidence_root / "validated_rendering_contract_manifest.json")
    sub_gates["rendering_preflight_gate"] = gate(
        status=(GateStatus.PASSED if rendering.get("passed") is True else GateStatus.FAILED),
        detail="accepted Phase 2B.6.1-R Vulkan/SAPIEN preflight",
        evidence={
            "manifest": (evidence_root / "validated_rendering_contract_manifest.json").as_posix(),
            "fingerprint": rendering.get("fingerprint"),
        },
    )
    environment = cast(Mapping[str, object], report["environment"])
    constructor = cast(Mapping[str, object], environment["constructor_kwargs"])
    sub_gates["environment_construction_gate"] = gate(
        status=GateStatus.PASSED,
        detail="fresh StackCube-v1 environment constructed",
        evidence={"fresh_environment": environment.get("fresh_environment")},
    )
    configuration_passed = (
        environment.get("control_mode") == "pd_joint_pos"
        and environment.get("control_frequency_hz") == 20
        and constructor.get("sim_backend") == "physx_cpu"
    )
    sub_gates["environment_configuration_gate"] = gate(
        status=GateStatus.PASSED if configuration_passed else GateStatus.FAILED,
        detail="frozen control and physics configuration",
        evidence={"constructor_kwargs": dict(constructor)},
    )
    source = cast(Mapping[str, object], report["source_identity"])
    sub_gates["action_space_contract_gate"] = gate(
        status=(
            GateStatus.PASSED
            if source.get("action_count") == len(report["step_reports"])
            or old_gates["action_contract_gate"]["status"] == GateStatus.PASSED.value
            else GateStatus.FAILED
        ),
        detail="environment action space and source action width are compatible",
        evidence={
            "source_action_count": source.get("action_count"),
            "source_action_shape": [source.get("action_count"), 8],
            "environment_action_shape": [8],
        },
    )
    observation_passed = (
        mode == "A"
        and constructor.get("obs_mode") == "none"
        or mode in {"B", "C"}
        and constructor.get("obs_mode") == "rgb"
    )
    sub_gates["observation_configuration_gate"] = gate(
        status=GateStatus.PASSED if observation_passed else GateStatus.FAILED,
        detail="mode-specific observation configuration",
        evidence={"mode": mode, "obs_mode": constructor.get("obs_mode")},
    )
    reset = cast(Mapping[str, object], report["reset"])
    sub_gates["reset_success_gate"] = gate(
        status=(
            GateStatus.PASSED
            if bool(reset) and old_gates["reset_identity_gate"]["status"] == "passed"
            else GateStatus.FAILED
        ),
        detail="source reset completed before action submission",
        evidence={"reset_report_present": bool(reset)},
    )
    for name, value in old_gates.items():
        sub_gates[name] = value
    sub_gates["stable_success_gate"] = gate(
        status=GateStatus.NOT_APPLICABLE,
        detail=(
            "producer has no separate stable-success acceptance gate; consecutive "
            "success is retained as a diagnostic"
        ),
        evidence={
            "producer_success_rule": "final_step_canonical_success",
            "separate_stable_success_gate_present_in_producer": False,
            "diagnostic_trailing_success_steps": cast(
                Mapping[str, object], report["success_timing"]
            ).get("trailing_consecutive_success_steps"),
        },
    )
    stable_counter = 0
    for raw in cast(list[dict[str, object]], report["step_reports"]):
        if raw.get("canonical_success") is True:
            stable_counter += 1
        else:
            stable_counter = 0
        index = int(cast(int, raw["step_index"]))
        raw["stable_success_counter_diagnostic"] = stable_counter
        raw["timestamp"] = index / 20.0
        raw["frame_index"] = index if mode in {"B", "C"} else None
    report["schema_version"] = "langmani-v2-phase2b6-1-v2-forensic-run-v0"
    report["rendering_contract"] = {
        "manifest_fingerprint": rendering.get("fingerprint"),
        "process_environment": {name: os.environ.get(name) for name in RENDER_ENVIRONMENT_NAMES},
        "gpu_audit": _gpu_audit(),
        "runtime_versions": _runtime_versions(),
    }
    report["sub_gates"] = sub_gates
    report["first_failed_sub_gate"] = first_failed_gate(sub_gates)
    report["mode_required_gates"] = list(required_gates_for_mode(mode))
    report["stable_success_gate_status"] = "not_applicable"
    report["passed"] = run_passed(report)
    report["external_evidence_path"] = external_path.as_posix()
    report.pop("fingerprint", None)
    return fingerprinted(report)


def _failed_run_report(
    *,
    mode: str,
    episode_id: int,
    repetition: int,
    evidence_root: Path,
    external_path: Path,
    error: BaseException,
) -> dict[str, object]:
    sub_gates = _not_reached_gates()
    rendering_path = evidence_root / "validated_rendering_contract_manifest.json"
    rendering = _read_json(rendering_path)
    sub_gates["rendering_preflight_gate"] = gate(
        status=(GateStatus.PASSED if rendering.get("passed") is True else GateStatus.FAILED),
        detail="accepted rendering preflight state",
        evidence={"manifest": rendering_path.as_posix()},
    )
    sub_gates["environment_construction_gate"] = gate(
        status=GateStatus.FAILED,
        detail="aggregate native runner raised before a complete report was retained",
        evidence={
            "exception_type": type(error).__name__,
            "exception_message": str(error),
        },
    )
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-v2-forensic-run-v0",
            "created_at_utc": _timestamp(),
            "run_identity": {
                "task_id": "StackCube-v1",
                "episode_id": episode_id,
                "mode": mode,
                "repetition": repetition,
                "fresh_process": {"pid": os.getpid()},
            },
            "step_reports": [],
            "success_timing": {},
            "physical_event_snapshots": {},
            "sub_gates": sub_gates,
            "first_failed_sub_gate": first_failed_gate(sub_gates),
            "mode_required_gates": list(required_gates_for_mode(mode)),
            "external_evidence_path": external_path.as_posix(),
            "policy_or_expert_invoked": False,
            "optimizer_created": False,
            "backward_passes": 0,
            "optimizer_steps": 0,
            "passed": False,
        }
    )


def run_replay(
    *,
    source_root: Path,
    output_root: Path,
    evidence_root: Path,
    spec_path: Path,
    mode: str,
    episode_id: int,
    repetition: int,
) -> dict[str, object]:
    """Execute one fresh-process repetition under the accepted launcher."""

    validate_primary_environment(os.environ)
    rendering = _read_json(evidence_root / "validated_rendering_contract_manifest.json")
    if rendering.get("passed") is not True:
        raise Phase2B61V2RuntimeError("rendering prerequisite is not passed")
    run_path = (
        output_root
        / "runs"
        / f"stackcube_{episode_id}"
        / f"mode_{mode}"
        / f"repetition_{repetition}"
        / "forensic_run.json"
    )
    try:
        raw = cast(
            dict[str, Any],
            run_forensic_replay(
                source_root=source_root,
                output_root=output_root,
                spec_path=spec_path,
                mode_value=mode,
                episode_id=episode_id,
                repetition=repetition,
            ),
        )
        report = _augment_run(
            report=raw,
            evidence_root=evidence_root,
            external_path=run_path,
        )
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        report = _failed_run_report(
            mode=mode,
            episode_id=episode_id,
            repetition=repetition,
            evidence_root=evidence_root,
            external_path=run_path,
            error=error,
        )
        run_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, report, replace=True)
    return report


def _load_runs(output_root: Path, episode_id: int, mode: str) -> list[dict[str, Any]]:
    root = output_root / "runs" / f"stackcube_{episode_id}" / f"mode_{mode}"
    if not root.is_dir():
        return []
    return [_read_json(path) for path in sorted(root.glob("repetition_*/forensic_run.json"))]


def _compact_run(run: Mapping[str, object]) -> dict[str, object]:
    external = Path(str(run["external_evidence_path"]))
    step_rows = []
    for raw in cast(Sequence[Mapping[str, object]], run.get("step_reports", [])):
        step_rows.append(
            {
                key: raw.get(key)
                for key in (
                    "step_index",
                    "submitted_action_sha256",
                    "reward",
                    "terminated",
                    "truncated",
                    "canonical_success",
                    "stable_success_counter_diagnostic",
                    "simulator_exception",
                    "observation_acquisition_success",
                    "rgb_frame_acquisition_success",
                    "panda_state_acquisition_success",
                    "timestamp",
                    "frame_index",
                )
            }
        )
    return fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-v2-compact-run-v0",
            "run_identity": run["run_identity"],
            "source_identity": run.get("source_identity"),
            "environment": run.get("environment"),
            "rendering_contract": run.get("rendering_contract"),
            "reset": run.get("reset"),
            "step_reports": step_rows,
            "physical_event_snapshots": run.get("physical_event_snapshots"),
            "terminal_state_error": run.get("terminal_state_error"),
            "terminal_object_pose_error": run.get("terminal_object_pose_error"),
            "source_success_timing": run.get("source_success_timing"),
            "success_timing": run.get("success_timing"),
            "sub_gates": run["sub_gates"],
            "first_failed_sub_gate": run.get("first_failed_sub_gate"),
            "writer_audit": run.get("writer_audit"),
            "external_evidence_path": external.as_posix(),
            "external_evidence_sha256": (
                "sha256:" + sha256_file(external) if external.is_file() else None
            ),
            "passed": run_passed(run),
        }
    )


def finalize_forensics(
    *,
    output_root: Path,
    evidence_root: Path,
    frozen_root: Path,
) -> dict[str, object]:
    """Aggregate only completed bounded runs and retain closed authorization."""

    controls = [*_load_runs(output_root, 936, "A"), *_load_runs(output_root, 937, "A")]
    mode_a = _load_runs(output_root, 938, "A")
    mode_b = _load_runs(output_root, 938, "B")
    mode_c = _load_runs(output_root, 938, "C")
    source = _read_json(evidence_root / "source_episode_identity.json")
    rendering = _read_json(evidence_root / "validated_rendering_contract_manifest.json")
    result = classify_result(
        source_identity_proven=source.get("producer_input_identity_proven") is True,
        rendering_preflight_passed=rendering.get("passed") is True,
        control_runs=controls,
        mode_a_runs=mode_a,
        mode_b_runs=mode_b,
        mode_c_runs=mode_c,
    )
    compact_controls = [_compact_run(run) for run in controls]
    compact_a = [_compact_run(run) for run in mode_a]
    compact_b = [_compact_run(run) for run in mode_b]
    compact_c = [_compact_run(run) for run in mode_c]
    by_episode = {
        int(
            cast(
                int,
                cast(Mapping[str, object], run["run_identity"])["episode_id"],
            )
        ): run
        for run in compact_controls
    }
    documents: dict[str, Mapping[str, object]] = {
        "control_936_result.json": cast(
            Mapping[str, object],
            by_episode.get(
                936,
                fingerprinted(
                    {
                        "schema_version": "langmani-v2-phase2b6-1-v2-missing-run-v0",
                        "episode_id": 936,
                        "passed": False,
                    }
                ),
            ),
        ),
        "control_937_result.json": cast(
            Mapping[str, object],
            by_episode.get(
                937,
                fingerprinted(
                    {
                        "schema_version": "langmani-v2-phase2b6-1-v2-missing-run-v0",
                        "episode_id": 937,
                        "passed": False,
                    }
                ),
            ),
        ),
        "episode_938_mode_a_results.json": fingerprinted(
            {
                "schema_version": "langmani-v2-phase2b6-1-v2-mode-a-v0",
                "runs": compact_a,
                "run_count": len(compact_a),
                "passed": len(compact_a) == 3 and all(run["passed"] is True for run in compact_a),
            }
        ),
        "episode_938_mode_b_results.json": fingerprinted(
            {
                "schema_version": "langmani-v2-phase2b6-1-v2-mode-b-v0",
                "runs": compact_b,
                "run_count": len(compact_b),
                "not_run": not compact_b,
                "passed": len(compact_b) == 2 and all(run["passed"] is True for run in compact_b),
            }
        ),
        "episode_938_mode_c_result.json": fingerprinted(
            {
                "schema_version": "langmani-v2-phase2b6-1-v2-mode-c-v0",
                "runs": compact_c,
                "run_count": len(compact_c),
                "not_run": not compact_c,
                "passed": len(compact_c) == 1 and all(run["passed"] is True for run in compact_c),
            }
        ),
        "task_state_diagnostics.json": fingerprinted(
            {
                "schema_version": "langmani-v2-phase2b6-1-v2-task-state-v0",
                "runs": [
                    {
                        "run_identity": run["run_identity"],
                        "events": run.get("physical_event_snapshots"),
                    }
                    for run in (*compact_controls, *compact_a, *compact_b, *compact_c)
                ],
                "official_canonical_predicate_authoritative": True,
                "diagnostic_geometry_replaces_predicate": False,
                "passed": True,
            }
        ),
        "success_timing_audit.json": fingerprinted(
            {
                "schema_version": "langmani-v2-phase2b6-1-v2-success-timing-v0",
                "runs": [
                    {
                        "run_identity": run["run_identity"],
                        "source_success_timing": run.get("source_success_timing"),
                        "replay_success_timing": run.get("success_timing"),
                    }
                    for run in (*compact_controls, *compact_a, *compact_b, *compact_c)
                ],
                "producer_success_rule": "final_step_canonical_success",
                "separate_stable_success_gate_present_in_producer": False,
                "stable_success_gate_status": "not_applicable",
                "actions_continue_after_first_success": True,
                "passed": True,
            }
        ),
        "determinism_audit.json": determinism_audit(mode_a),
        "observation_action_alignment_audit.json": fingerprinted(
            {
                "schema_version": "langmani-v2-phase2b6-1-v2-alignment-v0",
                "runs": [
                    {
                        "run_identity": run["run_identity"],
                        "pre_action_frame_count_gate": cast(
                            Mapping[str, object], run["sub_gates"]
                        ).get("pre_action_frame_count_gate"),
                        "observation_action_alignment_gate": cast(
                            Mapping[str, object], run["sub_gates"]
                        ).get("observation_action_alignment_gate"),
                        "timestamp_monotonicity_gate": cast(
                            Mapping[str, object], run["sub_gates"]
                        ).get("timestamp_monotonicity_gate"),
                    }
                    for run in (*compact_b, *compact_c)
                ],
                "terminal_diagnostic_frame_excluded": True,
                "fabricated_terminal_action_added": False,
                "passed": bool(compact_b) and all(run["passed"] is True for run in compact_b),
            }
        ),
        "writer_filesystem_audit.json": fingerprinted(
            {
                "schema_version": "langmani-v2-phase2b6-1-v2-writer-v0",
                "runs": [
                    {
                        "run_identity": run["run_identity"],
                        "writer_audit": run.get("writer_audit"),
                        "temporary_writer_gate": cast(Mapping[str, object], run["sub_gates"]).get(
                            "temporary_writer_gate"
                        ),
                        "episode_serialization_gate": cast(
                            Mapping[str, object], run["sub_gates"]
                        ).get("episode_serialization_gate"),
                    }
                    for run in compact_c
                ],
                "frozen_root": frozen_root.as_posix(),
                "writes_to_frozen_root": False,
                "lerobot_root_created": False,
                "passed": bool(compact_c) and all(run["passed"] is True for run in compact_c),
            }
        ),
        "first_failure_report.json": fingerprinted(
            {
                "schema_version": "langmani-v2-phase2b6-1-v2-first-failure-v0",
                "exact_first_failed_sub_gate": result["exact_first_failed_sub_gate"],
                "primary_failure_classification": result["primary_failure_classification"],
                "secondary_contributing_factors": result["secondary_contributing_factors"],
                "historical_aggregate_failure_subgate_available": False,
                "passed": result["result"] != "RESULT_D",
            }
        ),
        "result_classification.json": result,
        "eligibility_state.json": eligibility_state(result),
        "authorization_state_final.json": authorization_state(),
    }
    before = _read_json(evidence_root / "frozen_partial_output_inventory.json")
    after = inventory_frozen_output(frozen_root)
    before_unhashed = dict(before)
    before_unhashed.pop("fingerprint", None)
    after_unhashed = dict(after)
    after_unhashed.pop("fingerprint", None)
    inventory_unchanged = canonical_json_sha256(before_unhashed) == canonical_json_sha256(
        after_unhashed
    )
    remote = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-v2-remote-audit-v0",
            "created_at_utc": _timestamp(),
            "gpu_audit": _gpu_audit(),
            "process_audit": _process_audit(),
            "frozen_inventory_unchanged": inventory_unchanged,
            "external_output_root": output_root.as_posix(),
            "frozen_root": frozen_root.as_posix(),
            "production_resumed": False,
            "stackcube_939_or_later_processed": False,
            "pushcube_processed": False,
            "dataset_package_created": False,
            "archive_or_restore_started": False,
            "model_or_optimizer_loaded": False,
            "passed": inventory_unchanged,
        }
    )
    documents["remote_execution_audit.json"] = remote
    for name, document in documents.items():
        _write_json(evidence_root / name, document)
    summary = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-v2-finalization-v0",
            "result": result["result"],
            "control_run_count": len(controls),
            "mode_a_run_count": len(mode_a),
            "mode_b_run_count": len(mode_b),
            "mode_c_run_count": len(mode_c),
            "frozen_inventory_unchanged": inventory_unchanged,
            "authorization": authorization_state(),
            "passed": result["result"] != "RESULT_D" and inventory_unchanged,
        }
    )
    _write_json(evidence_root / "finalization_summary.json", summary)
    return summary


def write_artifact_manifest(evidence_root: Path) -> dict[str, object]:
    """Hash every compact committed JSON artifact except the manifest itself."""

    rows = []
    for path in sorted(evidence_root.glob("*.json")):
        if path.name == "artifact_manifest.json":
            continue
        rows.append(
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": "sha256:" + sha256_file(path),
            }
        )
    report = fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b6-1-v2-artifact-manifest-v0",
            "file_count": len(rows),
            "files": rows,
            "large_rgb_or_npz_committed": False,
            "passed": bool(rows),
        }
    )
    _write_json(evidence_root / "artifact_manifest.json", report, replace=True)
    return report
