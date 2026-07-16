"""Verify the portable M4.3a semantic-alignment audit implementation.

This entry point is deliberately non-target in M4.3a.  It exercises typed
contracts, pure retrieval math, immutable evidence, and the audit CLI dry-run.
It does not load real checkpoints, step a simulator, access a final schedule,
or require the future FactorFiLM implementation.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import BIN_IDS, OBJECT_IDS, stable_task_id
from langmani.policies.act_runtime import atomic_write_json
from langmani.policies.act_semantic_audit import (
    bin_retrieval_confusion,
    build_semantic_alignment_audit,
    compute_action_chunk_distance,
    object_retrieval_confusion,
    retrieve_bin,
    retrieve_object,
    retrieve_task,
    task_retrieval_confusion,
)
from langmani.policies.m42_types import GripperRuntimeMode
from langmani.policies.m43_evidence import (
    M43AuditConfig,
    M43AuditScope,
    M43AuditScopeArtifact,
    M43EvidenceError,
    M43ObservationSource,
    ObservationSourceIdentity,
    build_policy_semantic_summary,
    stage_and_promote_audit_evidence,
    unavailable_rollout_evidence,
    validate_completed_audit_evidence,
)
from langmani.policies.m43_types import (
    M43_ACTION_CHUNK_SIZE,
    M43_ACTION_COMPONENTS,
    M43_CANONICAL_TASK_IDS,
    PRIMARY_ACTION_CHUNK_DISTANCE_METRIC,
    ActionChunkDistanceConfig,
    SemanticAlignmentAudit,
    SemanticAuditConclusion,
    SemanticFailureClass,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_OUTPUT_ROOT = OUTPUT_ROOT / "diagnostics" / "m43"
REPORT_SCHEMA = "langmani-m43-verification-v0"

PROTECTED_SOURCE_ROOTS = (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "environment",
    PROJECT_ROOT / "tests",
    PROJECT_ROOT / "docs",
    PROJECT_ROOT / ".git",
)


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _resolved_unlinked(path: Path, *, label: str) -> Path:
    lexical = _lexical_absolute(path)
    for component in (lexical, *lexical.parents):
        if component.is_symlink() or component.is_junction():
            raise RuntimeError(f"{label} traverses a symlink or junction: {component}")
    return lexical.resolve(strict=False)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _overlaps(left: Path, right: Path) -> bool:
    return _is_within(left, right) or _is_within(right, left)


def _validate_output_root(path: Path) -> Path:
    output = _resolved_unlinked(path, label="M4.3 verifier output")
    protected = tuple(
        _resolved_unlinked(value, label="protected source path") for value in PROTECTED_SOURCE_ROOTS
    )
    if any(_overlaps(output, value) for value in protected):
        raise RuntimeError("M4.3 verifier output must not overlap source or Git content")
    if output.exists() and not output.is_dir():
        raise RuntimeError("M4.3 verifier output must be a real directory when present")
    report = _resolved_unlinked(output / "verification.json", label="M4.3 verifier report")
    if report.exists() and not report.is_file():
        raise RuntimeError("M4.3 verifier report must be a real file when present")
    return output


@dataclass(slots=True)
class Report:
    checks: list[dict[str, object]] = field(default_factory=list)
    implementation_validated: bool = False
    semantic_audit_implementation_validated: bool = False
    prior_m4_evidence_validated: bool = False
    prior_m42_evidence_validated: bool = False
    dataset_fingerprint_validated: bool = False
    checkpoint_fingerprints_validated: bool = False
    semantic_audit_completed: bool = False
    state_onehot_semantic_alignment_validated: bool = False
    tasktoken_semantic_alignment_validated: bool = False
    factor_film_training_completed: bool = False
    factor_film_checkpoints_complete: bool = False
    factor_film_checkpoint_selected: bool = False
    validation_only_selection_validated: bool = False
    factor_film_reload_validated: bool = False
    development_benchmark_completed: bool = False
    post_grasp_analysis_completed: bool = False
    correct_task_retrieval_validated: bool = False
    object_retrieval_validated: bool = False
    bin_retrieval_validated: bool = False
    raw_action_metrics_validated: bool = False
    runtime_action_metrics_validated: bool = False
    development_quality_gate_passed: bool = False
    final_benchmark_authorized: bool = False
    final_schedule_accessed: bool = False
    smolvla_go: bool = False
    physical_target_validated: bool = False

    def check(self, name: str, condition: bool, detail: str) -> None:
        status = "pass" if condition else "fail"
        print(f"[{status.upper()}] {name}: {detail}")
        self.checks.append({"name": name, "status": status, "detail": detail})

    @property
    def failed(self) -> bool:
        return any(value["status"] == "fail" for value in self.checks)

    @property
    def passed(self) -> bool:
        target_or_future_claims = (
            self.prior_m4_evidence_validated,
            self.prior_m42_evidence_validated,
            self.dataset_fingerprint_validated,
            self.checkpoint_fingerprints_validated,
            self.semantic_audit_completed,
            self.state_onehot_semantic_alignment_validated,
            self.tasktoken_semantic_alignment_validated,
            self.factor_film_training_completed,
            self.factor_film_checkpoints_complete,
            self.factor_film_checkpoint_selected,
            self.validation_only_selection_validated,
            self.factor_film_reload_validated,
            self.development_benchmark_completed,
            self.post_grasp_analysis_completed,
            self.correct_task_retrieval_validated,
            self.object_retrieval_validated,
            self.bin_retrieval_validated,
            self.raw_action_metrics_validated,
            self.runtime_action_metrics_validated,
            self.development_quality_gate_passed,
            self.final_benchmark_authorized,
            self.final_schedule_accessed,
            self.smolvla_go,
            self.physical_target_validated,
        )
        return (
            not self.failed
            and self.implementation_validated
            and self.semantic_audit_implementation_validated
            and not any(target_or_future_claims)
        )

    def write(self, output_root: Path) -> None:
        payload = {
            "schema_version": REPORT_SCHEMA,
            "verification_mode": "non_target_structural",
            **{key: value for key, value in asdict(self).items() if key != "checks"},
            "checks": self.checks,
            "passed": self.passed,
        }
        atomic_write_json(output_root / "verification.json", payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def _distance_config() -> ActionChunkDistanceConfig:
    return ActionChunkDistanceConfig(
        action_lower_bounds=(-10.0,) * M43_ACTION_COMPONENTS,
        action_upper_bounds=(10.0,) * M43_ACTION_COMPONENTS,
        train_action_std=(1.0,) * M43_ACTION_COMPONENTS,
        locked_execution_horizon=10,
    )


def _reference_chunks() -> dict[str, np.ndarray]:
    chunks: dict[str, np.ndarray] = {}
    for task_spec in CANONICAL_TASK_SPECS:
        value = np.zeros((M43_ACTION_CHUNK_SIZE, M43_ACTION_COMPONENTS), dtype=np.float64)
        value[:, 0] = float(OBJECT_IDS.index(task_spec.target_object_id) * 4)
        value[:, 1] = float(BIN_IDS.index(task_spec.target_bin_id))
        chunks[stable_task_id(task_spec)] = value
    return chunks


def _digest(value: int) -> str:
    return f"sha256:{value:064x}"


def _fixture_audit(
    *,
    policy_label: str,
    observations: tuple[ObservationSourceIdentity, ...],
    references: dict[str, np.ndarray],
    config: ActionChunkDistanceConfig,
    fixture_reload_validated: bool,
) -> SemanticAlignmentAudit:
    task_results = []
    object_results = []
    bin_results = []
    complete_distance_results: dict[str, object] = {}
    task_spec_by_id = {stable_task_id(value): value for value in CANONICAL_TASK_SPECS}
    for observation in observations:
        candidate = references[observation.queried_task_id]
        arguments = {
            "observation_id": observation.observation_id,
            "requested_task_id": observation.queried_task_id,
            "candidate_chunk": candidate,
            "per_task_chunks": references,
            "config": config,
        }
        task_results.append(retrieve_task(**arguments))
        object_results.append(retrieve_object(**arguments))
        bin_results.append(retrieve_bin(**arguments))
        object_centroids = {
            object_id: np.mean(
                np.stack(
                    [
                        references[stable_task_id(task_spec)]
                        for task_spec in CANONICAL_TASK_SPECS
                        if task_spec.target_object_id == object_id
                    ]
                ),
                axis=0,
            )
            for object_id in OBJECT_IDS
        }
        bin_centroids = {
            bin_id: np.mean(
                np.stack(
                    [
                        references[stable_task_id(task_spec)]
                        for task_spec in CANONICAL_TASK_SPECS
                        if task_spec.target_bin_id == bin_id
                    ]
                ),
                axis=0,
            )
            for bin_id in BIN_IDS
        }
        requested_object = task_spec_by_id[observation.queried_task_id].target_object_id
        conditional = {
            bin_id: next(
                references[stable_task_id(task_spec)]
                for task_spec in CANONICAL_TASK_SPECS
                if task_spec.target_object_id == requested_object
                and task_spec.target_bin_id == bin_id
            )
            for bin_id in BIN_IDS
        }
        complete_distance_results[observation.observation_id] = {
            "task_references": {
                label: compute_action_chunk_distance(candidate, reference, config).to_dict()
                for label, reference in references.items()
            },
            "object_centroids": {
                label: compute_action_chunk_distance(candidate, reference, config).to_dict()
                for label, reference in object_centroids.items()
            },
            "bin_centroids": {
                label: compute_action_chunk_distance(candidate, reference, config).to_dict()
                for label, reference in bin_centroids.items()
            },
            "conditional_bin_references": {
                label: compute_action_chunk_distance(candidate, reference, config).to_dict()
                for label, reference in conditional.items()
            },
        }
    return build_semantic_alignment_audit(
        policy_label=policy_label,
        distance_config=config,
        task_results=task_results,
        object_results=object_results,
        bin_results=bin_results,
        first_interactions=(),
        conclusions=(),
        raw_policy_action_metrics={"fixture_only": True},
        runtime_action_metrics={},
        deterministic_reload_validated=fixture_reload_validated,
        deterministic_reset_validated=True,
        complete_distance_results=complete_distance_results,
    )


def _fixture_model_reload_probe() -> bool:
    """Reload a fresh tiny model without claiming a real ACT checkpoint reload."""

    source = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        source.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    sample = torch.tensor([[0.25, -0.5]])
    expected = source(sample)
    buffer = io.BytesIO()
    torch.save(source.state_dict(), buffer)
    buffer.seek(0)
    reloaded = torch.nn.Linear(2, 2, bias=False)
    reloaded.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
    return torch.equal(expected, reloaded(sample))


def _evidence_lifecycle_probe(report: Report) -> None:
    config = _distance_config()
    references = _reference_chunks()
    observations = tuple(
        ObservationSourceIdentity(
            source=M43ObservationSource.M3B_VALIDATION,
            observation_id=f"fixture-validation-{group_index:02d}-{task_index}",
            scene_id=f"fixture-scene-{group_index:02d}",
            scene_group_id=f"fixture-group-{group_index:02d}",
            scene_seed=group_index,
            source_task_id=M43_CANONICAL_TASK_IDS[0],
            source_episode_id=f"fixture-source-episode-{group_index:02d}",
            frame_index=0,
            image_fingerprint=_digest(100 + group_index),
            policy_state_fingerprint=_digest(200 + group_index),
            queried_task_id=task_id,
        )
        for group_index in range(6)
        for task_index, task_id in enumerate(M43_CANONICAL_TASK_IDS)
    )
    audit_config = M43AuditConfig(
        scope=M43AuditScope.M3B_VALIDATION,
        git_commit="a" * 40,
        m3b_fingerprint=_digest(1),
        m3b_split_manifest_digest=_digest(2),
        per_task_checkpoint_fingerprints={
            task_id: _digest(10 + index) for index, task_id in enumerate(M43_CANONICAL_TASK_IDS)
        },
        state_onehot_checkpoint_fingerprint=_digest(20),
        task_token_checkpoint_fingerprint=_digest(21),
        selected_execution_horizon=10,
        selected_gripper_runtime=GripperRuntimeMode.PROJECT,
        distance_config=config,
        ordered_observations=observations,
    )
    fixture_reload_validated = _fixture_model_reload_probe()
    report.check(
        "synthetic model reload fixture",
        fixture_reload_validated,
        "a fresh tiny model reload reproduces output without claiming a real ACT checkpoint",
    )
    audits = {
        label: _fixture_audit(
            policy_label=label,
            observations=observations,
            references=references,
            config=config,
            fixture_reload_validated=fixture_reload_validated,
        )
        for label in ("state_onehot", "task_token")
    }
    summaries = {
        label: build_policy_semantic_summary(audit, task_sensitivity_magnitude=1.0)
        for label, audit in audits.items()
    }
    required_summary_fields = {
        "full_task_top1_retrieval",
        "full_task_top2_retrieval",
        "mean_reciprocal_rank",
        "mean_correct_reference_margin",
        "target_object_retrieval",
        "destination_bin_retrieval",
        "conditional_bin_retrieval",
        "per_task_retrieval",
        "task_confusion_matrix",
        "object_confusion_matrix",
        "bin_confusion_matrix",
        "first_interaction_confusions",
        "first_interaction_evidence_available",
        "post_grasp_failure_distribution",
        "task_sensitivity_magnitude",
        "semantic_failure_classification",
        "deterministic_reload",
        "interpretation",
    }
    summary_contract_ok = all(
        set(summary) == required_summary_fields
        and set(summary["per_task_retrieval"]) == set(M43_CANONICAL_TASK_IDS)
        and isinstance(summary["semantic_failure_classification"], dict)
        and summary["deterministic_reload"]
        == {"validated": True, "available": True, "reason": None}
        and isinstance(summary["interpretation"], dict)
        for summary in summaries.values()
    )
    report.check(
        "complete semantic summary structure",
        summary_contract_ok,
        "both candidates expose retrieval, confusion, interaction, failure, sensitivity, and interpretation fields",
    )
    report.check(
        "real checkpoint reload remains unclaimed",
        not report.factor_film_reload_validated
        and not report.state_onehot_semantic_alignment_validated
        and not report.tasktoken_semantic_alignment_validated,
        "the synthetic reload fixture does not set any real-checkpoint or semantic validation flag",
    )
    artifact = M43AuditScopeArtifact(
        scope=M43AuditScope.M3B_VALIDATION,
        ordered_observation_ids=tuple(value.observation_id for value in observations),
        policy_audits=audits,
        policy_summaries=summaries,
        first_interaction_evidence_available=False,
        rollout_evidence=unavailable_rollout_evidence(),
    )
    with tempfile.TemporaryDirectory(prefix="langmani-m43-evidence-fixture-") as temporary:
        completed = stage_and_promote_audit_evidence(
            output_root=Path(temporary),
            config=audit_config,
            scope_artifacts={M43AuditScope.M3B_VALIDATION: artifact},
        )
        validated = validate_completed_audit_evidence(completed.root, expected_config=audit_config)
        lifecycle_ok = (
            validated.config.fingerprint == audit_config.fingerprint
            and validated.root == completed.root
            and (validated.root / "complete.json").is_file()
        )
        report.check(
            "immutable evidence promotion",
            lifecycle_ok,
            "staging, independent validation, atomic promotion, and completion marker passed",
        )
        summary_path = validated.root / "summary.md"
        summary_path.write_text("tampered\n", encoding="utf-8")
        try:
            validate_completed_audit_evidence(validated.root, expected_config=audit_config)
        except M43EvidenceError:
            tamper_rejected = True
        else:
            tamper_rejected = False
        report.check(
            "immutable evidence corruption rejection",
            tamper_rejected,
            "post-promotion content mutation fails checksum and semantic validation",
        )


def _math_and_contract_probe(report: Report) -> None:
    config = _distance_config()
    zeros = np.zeros((M43_ACTION_CHUNK_SIZE, M43_ACTION_COMPONENTS), dtype=np.float64)
    gripper_only = zeros.copy()
    gripper_only[:, 7] = 9.0
    gripper_distance = compute_action_chunk_distance(gripper_only, zeros, config)
    report.check(
        "primary distance excludes gripper",
        gripper_distance.primary_distance == 0.0,
        f"metric={PRIMARY_ACTION_CHUNK_DISTANCE_METRIC}",
    )
    arm = zeros.copy()
    arm[: config.locked_execution_horizon, 0] = 2.0
    arm_distance = compute_action_chunk_distance(arm, zeros, config)
    expected = float(np.sqrt(config.locked_execution_horizon) * 0.1)
    report.check(
        "range-normalized locked-horizon distance",
        np.isclose(arm_distance.primary_distance, expected),
        f"observed={arm_distance.primary_distance}; expected={expected}; horizon=10",
    )

    references = _reference_chunks()
    requested = M43_CANONICAL_TASK_IDS[3]
    candidate = references[requested]
    task = retrieve_task(
        observation_id="fixture-observation",
        requested_task_id=requested,
        candidate_chunk=candidate,
        per_task_chunks=references,
        config=config,
    )
    object_result = retrieve_object(
        observation_id="fixture-observation",
        requested_task_id=requested,
        candidate_chunk=candidate,
        per_task_chunks=references,
        config=config,
    )
    bin_result = retrieve_bin(
        observation_id="fixture-observation",
        requested_task_id=requested,
        candidate_chunk=candidate,
        per_task_chunks=references,
        config=config,
    )
    retrieval_ok = (
        task.top1_correct
        and task.correct_rank == 1
        and object_result.correct
        and bin_result.correct
        and bin_result.conditional_correct
    )
    report.check(
        "semantic retrieval fixture",
        retrieval_ok,
        "full-task, object, global-bin, and conditional-bin retrieval agree",
    )
    confusions = (
        task_retrieval_confusion((task,)),
        object_retrieval_confusion((object_result,)),
        bin_retrieval_confusion((bin_result,)),
    )
    confusion_ok = all(value.total == 1 for value in confusions)
    report.check(
        "deterministic confusion matrices",
        confusion_ok,
        "each canonical ordered fixture confusion contains exactly one observation",
    )
    payload = {
        "config": config.to_dict(),
        "distance": arm_distance.to_dict(),
        "task": task.to_dict(),
        "object": object_result.to_dict(),
        "bin": bin_result.to_dict(),
        "confusions": [value.to_dict() for value in confusions],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
    report.check(
        "portable report serialization",
        isinstance(json.loads(encoded), dict),
        "all M4.3a contract records round-trip through finite JSON",
    )
    report.check(
        "semantic taxonomy contracts",
        len(SemanticFailureClass) == 12 and len(SemanticAuditConclusion) == 7,
        "twelve exhaustive outcomes and seven permitted conclusions",
    )


def _cli_dry_run_probe(report: Report) -> None:
    with tempfile.TemporaryDirectory(prefix="langmani-m43-verify-") as temporary:
        root = Path(temporary)
        report_path = root / "command-report.json"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "audit_act_semantics.py"),
            "--mode",
            "combined",
            "--device",
            "cpu",
            "--dataset-root",
            str(root / "dataset"),
            "--m4-checkpoint-root",
            str(root / "m4-checkpoints"),
            "--task-token-checkpoint-root",
            str(root / "task-token-checkpoints"),
            "--m42-diagnostics-root",
            str(root / "m42-diagnostics"),
            "--runtime-selection",
            str(root / "runtime-selection.json"),
            "--output-root",
            str(root / "evidence"),
            "--report",
            str(report_path),
            "--dry-run",
        ]
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        payload = (
            json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        )
        flags_ok = (
            completed.returncode == 0
            and payload.get("passed") is True
            and payload.get("requested_sources") == ["m3b_validation", "m42_dev_v0"]
            and payload.get("semantic_audit_implementation_validated") is True
            and payload.get("semantic_audit_completed") is False
            and payload.get("factor_film_training_completed") is False
            and payload.get("test_split_accessed") is False
            and payload.get("fresh_seed_accessed") is False
            and payload.get("final_schedule_accessed") is False
            and payload.get("final_benchmark_authorized") is False
            and payload.get("smolvla_go") is False
            and payload.get("physical_execution") is False
        )
        report.check(
            "semantic-audit CLI dry-run",
            flags_ok,
            f"rc={completed.returncode}; combined sources remain separated and sealed",
        )


def _structural_checks(report: Report) -> None:
    report.check(
        "M4.3a imports",
        tuple(stable_task_id(value) for value in CANONICAL_TASK_SPECS) == M43_CANONICAL_TASK_IDS,
        "portable contracts and semantic-audit functions imported without FactorFiLM",
    )
    _math_and_contract_probe(report)
    _evidence_lifecycle_probe(report)
    _cli_dry_run_probe(report)
    report.check(
        "test and final access prohibition",
        not report.final_schedule_accessed
        and not report.final_benchmark_authorized
        and not report.smolvla_go,
        "non-target verification materialized no test, fresh, or final schedule identity",
    )
    report.check(
        "truthful physical-validation state",
        not any(
            (
                report.semantic_audit_completed,
                report.state_onehot_semantic_alignment_validated,
                report.tasktoken_semantic_alignment_validated,
                report.factor_film_reload_validated,
                report.physical_target_validated,
            )
        ),
        "fixtures and dry-run claim neither real checkpoint reload, semantic completion, nor physical execution",
    )
    report.semantic_audit_implementation_validated = not report.failed
    report.implementation_validated = report.semantic_audit_implementation_validated


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output_root = _validate_output_root(args.output_root)
    except Exception:  # noqa: BLE001 - unsafe report paths must never be written
        traceback.print_exc()
        return 1
    report = Report()
    try:
        _structural_checks(report)
    except Exception as error:  # noqa: BLE001 - command boundary preserves exact diagnostics
        traceback.print_exc()
        report.check(
            "unexpected verifier exception",
            False,
            f"{type(error).__name__}: {str(error) or repr(error)}",
        )
    output_root.mkdir(parents=True, exist_ok=True)
    report.write(output_root)
    print(f"[INFO] report: {output_root / 'verification.json'}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
