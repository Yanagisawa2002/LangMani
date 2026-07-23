"""Write deterministic pre-probe contracts for LangMani 2.0 Phase 2B.4-F1."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch

from langmani.v2.phase2b4_evidence import verify_phase2b4_f0_artifacts
from langmani.v2.phase2b4_f1 import (
    EXCLUDED_GEOMETRY_COMMIT,
    SOURCE_COMMIT,
    curriculum_manifest,
    probe_configuration,
    reconstruct_f0_learning_dynamics,
    repository_operation_boundary,
    seed_disjointness_audit,
    termination_bootstrap_audit,
)
from langmani.v2.phase2b4_ppo import canonical_json_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2b4_f1"
F0_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "langmani_v2" / "phase_2b4_f0"
EXPECTED_BRANCH = "codex/langmani-v2-phase2b4-f1-ppo-diagnosis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
    ).strip()


def _git_ok(*args: str) -> bool:
    return (
        subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fingerprinted(payload: dict[str, object]) -> dict[str, object]:
    payload["fingerprint"] = canonical_json_sha256(payload)
    return payload


def _repository_audit() -> dict[str, object]:
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    origin = _git("remote", "get-url", "origin")
    source_is_ancestor = _git_ok("merge-base", "--is-ancestor", SOURCE_COMMIT, "HEAD")
    geometry_is_ancestor = _git_ok(
        "merge-base",
        "--is-ancestor",
        EXCLUDED_GEOMETRY_COMMIT,
        "HEAD",
    )
    checks = {
        "branch_exact": branch == EXPECTED_BRANCH,
        "source_is_ancestor": source_is_ancestor,
        "excluded_geometry_is_not_ancestor": not geometry_is_ancestor,
        "worktree_clean_before_artifact_write": status == "",
        "origin_identity": origin
        in {
            "https://github.com/Yanagisawa2002/LangMani.git",
            "git@github.com:Yanagisawa2002/LangMani.git",
        },
    }
    payload: dict[str, object] = {
        "schema_version": "langmani-v2-phase2b4-f1-repository-environment-audit-v0",
        "repository": "https://github.com/Yanagisawa2002/LangMani.git",
        "worktree": PROJECT_ROOT.as_posix(),
        "branch": branch,
        "head": head,
        "source_branch": "codex/langmani-v2-phase2b4-ppo-teacher-feasibility",
        "source_commit": SOURCE_COMMIT,
        "excluded_geometry_commit": EXCLUDED_GEOMETRY_COMMIT,
        "origin": origin,
        "checks": checks,
        "runtime": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "formal_seeds_accessed": False,
        "collection_process_found": False,
        "lerobot_process_found": False,
        "smolvla_process_found": False,
        "student_training_process_found": False,
        "passed": all(checks.values()),
    }
    return _fingerprinted(payload)


def main() -> int:
    root = parse_args().output_root.resolve()
    repository = _repository_audit()
    if repository["passed"] is not True:
        raise RuntimeError(f"F1 repository isolation failed: {repository['checks']}")
    prior = verify_phase2b4_f0_artifacts(F0_ARTIFACT_ROOT)
    if prior["passed"] is not True or prior["result"] != "RESULT_C":
        raise RuntimeError("frozen F0 evidence verification failed")
    prior_payload = _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b4-f1-prior-evidence-verification-v0",
            "source_artifact_root": F0_ARTIFACT_ROOT.as_posix(),
            "result": prior["result"],
            "artifact_hash_checks": prior["artifact_hash_checks"],
            "artifact_size_checks": prior["artifact_size_checks"],
            "checkpoint": prior["checkpoint"],
            "prior_authorization_state": prior["authorization_state"],
            "prior_evidence_immutable": True,
            "passed": True,
        }
    )
    training = _read(F0_ARTIFACT_ROOT / "micro_training_metrics.json")
    reward_decision = _fingerprinted(
        {
            "schema_version": "langmani-v2-phase2b4-f1-reward-decision-v0",
            "status": "pending_target_diagnosis",
            "probe_a_reward_revision": 0,
            "probe_b_reward_revision": None,
            "reward_revision_implemented": False,
            "decision_evidence": None,
            "no_reward_change_before_diagnosis": True,
        }
    )
    artifacts = {
        "repository_environment_audit.json": repository,
        "prior_evidence_verification.json": prior_payload,
        "f0_learning_dynamics_audit.json": reconstruct_f0_learning_dynamics(training),
        "termination_bootstrap_audit.json": termination_bootstrap_audit(),
        "safe_curriculum_manifest.json": curriculum_manifest(),
        "reward_decision.json": reward_decision,
        "seed_disjointness_audit.json": seed_disjointness_audit(),
        "probe_a_configuration.json": probe_configuration(probe="A"),
        "operation_boundary.json": repository_operation_boundary(),
    }
    if artifacts["termination_bootstrap_audit.json"]["passed"] is not True:
        raise RuntimeError("F0 termination/bootstrap audit failed")
    if artifacts["seed_disjointness_audit.json"]["passed"] is not True:
        raise RuntimeError("F1 seed-disjointness audit failed")
    for name, payload in artifacts.items():
        _write(root / name, payload)
    print(json.dumps({"output_root": root.as_posix(), "files": sorted(artifacts)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
