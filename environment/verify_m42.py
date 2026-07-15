"""Verify M4.2 contracts, the development pipeline, or the sealed final benchmark."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, cast

import torch
from lerobot.policies.act import ACTPolicy

from langmani.datasets.identity import sha256_hex
from langmani.datasets.lerobot_types import ACTION_FEATURE_KEY, IMAGE_FEATURE_KEY, STATE_FEATURE_KEY
from langmani.policies.act_checkpoint import CHECKPOINT_COMPLETION_MARKER, load_act_checkpoint
from langmani.policies.act_runtime import atomic_write_json, inspect_git_state
from langmani.policies.act_task_token import (
    build_task_token_act_config,
    inject_task_token_training_batch,
    validate_task_token_policy,
)
from langmani.policies.act_training import build_policy_and_processors, prepare_raw_batch
from langmani.policies.act_types import ActModelConfig
from langmani.policies.m42_analysis import (
    PostGraspTrace,
    classify_post_grasp_phase,
    create_go_no_go_decision,
)
from langmani.policies.m42_evidence import load_prior_m4_evidence
from langmani.policies.m42_runtime import BinaryGripperEnvPostprocessorV0
from langmani.policies.m42_schedule import (
    M42_DEV_SCHEDULE_FINGERPRINT,
    M42_FINAL_SCHEDULE_FINGERPRINT,
    load_exclusion_sources,
    materialize_locked_schedule,
    validate_locked_schedules,
)
from langmani.policies.m42_source_fingerprint import (
    TASK_TOKEN_TRAINING_SOURCE_FILES,
    task_token_training_source_fingerprint,
)
from langmani.policies.m42_training import (
    TaskTokenTrainingManifest,
    TaskTokenValidationQueue,
    canonical_fingerprint,
    fingerprint_owned_task_token_runs,
)
from langmani.policies.m42_types import (
    ExecutionHorizonConfig,
    M42Decision,
    M42GoNoGoMetrics,
    PostGraspPhase,
    TaskTokenConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_DATASET_ROOT = OUTPUT_ROOT / "datasets" / "m3b" / "langmani-pick-place-lerobot-v1"
DEFAULT_M4_MODEL_ROOT = OUTPUT_ROOT / "models" / "act"
DEFAULT_M42_MODEL_ROOT = OUTPUT_ROOT / "models" / "act-task-token"
DEFAULT_M4_DIAGNOSTICS_ROOT = OUTPUT_ROOT / "diagnostics" / "m4"
DEFAULT_M42_OUTPUT_ROOT = OUTPUT_ROOT / "diagnostics" / "m42"
REPORT_PATH = DEFAULT_M42_OUTPUT_ROOT / "verification.json"
VerificationMode = Literal["structural", "target_development", "target_final"]

PROTECTED_SOURCE_ROOTS = (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "environment",
    PROJECT_ROOT / "tests",
    PROJECT_ROOT / "docs",
    PROJECT_ROOT / ".git",
)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _overlaps(left: Path, right: Path) -> bool:
    return _is_within(left, right) or _is_within(right, left)


def _lexical_absolute(path: Path) -> Path:
    """Return an absolute path without following a symlink or junction."""

    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _resolved_unlinked(path: Path, *, label: str) -> Path:
    """Resolve one path only after rejecting linked existing components."""

    lexical = _lexical_absolute(path)
    for component in (lexical, *lexical.parents):
        if component.is_symlink() or component.is_junction():
            raise RuntimeError(f"{label} traverses a symlink or junction: {component}")
    return lexical.resolve(strict=False)


def _validate_paths(args: argparse.Namespace) -> Path:
    immutable = tuple(
        _resolved_unlinked(path, label="protected path")
        for path in (
            args.dataset_root,
            args.m4_model_root,
            args.m4_diagnostics_root,
            *PROTECTED_SOURCE_ROOTS,
        )
    )
    task_token = _resolved_unlinked(args.task_token_model_root, label="TaskToken model output")
    if any(_overlaps(task_token, path) for path in immutable):
        raise RuntimeError(
            "TaskToken model output must not overlap dataset, M4 checkpoints, diagnostics, "
            "source, or Git content"
        )
    if task_token.exists() and not task_token.is_dir():
        raise RuntimeError("TaskToken model output root must be a real directory")
    output = _resolved_unlinked(args.output_root, label="M4.2 verifier output")
    if any(_overlaps(output, path) for path in (*immutable, task_token)):
        raise RuntimeError(
            "M4.2 verifier output must not overlap dataset, checkpoint, diagnostics, "
            "TaskToken, source, or Git content"
        )
    if output.exists() and not output.is_dir():
        raise RuntimeError("M4.2 verifier output root must be a real directory")
    # verification.json is the command report exception; no physical evidence
    # directory is created here, including for dry-run execution.
    verification = _resolved_unlinked(
        output / "verification.json", label="M4.2 verification command report"
    )
    if verification.exists() and not verification.is_file():
        raise RuntimeError("M4.2 verification command report must be a real file")
    return output


@dataclass(slots=True)
class Report:
    checks: list[dict[str, object]] = field(default_factory=list)
    implementation_validated: bool = False
    prior_m4_evidence_validated: bool = False
    development_schedule_locked: bool = False
    final_schedule_locked: bool = False
    horizon_ablation_completed: bool = False
    horizon_selection_locked: bool = False
    gripper_ablation_completed: bool = False
    gripper_selection_locked: bool = False
    task_token_training_completed: bool = False
    task_token_checkpoint_selected: bool = False
    validation_only_selection_validated: bool = False
    development_benchmark_completed: bool = False
    final_benchmark_completed: bool = False
    post_grasp_analysis_completed: bool = False
    raw_action_metrics_validated: bool = False
    runtime_action_metrics_validated: bool = False
    go_no_go_decision_completed: bool = False
    smolvla_go: bool = False
    physical_target_validated: bool = False
    dry_run_validated: bool = False
    dry_run_plan: dict[str, object] = field(default_factory=dict)

    def check(self, name: str, condition: bool, detail: str) -> None:
        status = "pass" if condition else "fail"
        print(f"[{status.upper()}] {name}: {detail}")
        self.checks.append({"name": name, "status": status, "detail": detail})

    @property
    def failed(self) -> bool:
        return any(item["status"] == "fail" for item in self.checks)

    def accepted(self, mode: VerificationMode, *, dry_run: bool = False) -> bool:
        if self.failed or not self.implementation_validated:
            return False
        if mode == "structural":
            return not any(
                (
                    self.prior_m4_evidence_validated,
                    self.horizon_ablation_completed,
                    self.task_token_training_completed,
                    self.development_benchmark_completed,
                    self.final_benchmark_completed,
                    self.physical_target_validated,
                )
            )
        if dry_run:
            return all(
                (
                    self.prior_m4_evidence_validated,
                    self.development_schedule_locked,
                    self.final_schedule_locked,
                    self.dry_run_validated,
                    not self.horizon_ablation_completed,
                    not self.gripper_ablation_completed,
                    not self.task_token_training_completed,
                    not self.development_benchmark_completed,
                    not self.final_benchmark_completed,
                    not self.go_no_go_decision_completed,
                    not self.smolvla_go,
                    not self.physical_target_validated,
                )
            )
        development = all(
            (
                self.prior_m4_evidence_validated,
                self.development_schedule_locked,
                self.final_schedule_locked,
                self.horizon_ablation_completed,
                self.horizon_selection_locked,
                self.gripper_ablation_completed,
                self.gripper_selection_locked,
                self.task_token_training_completed,
                self.task_token_checkpoint_selected,
                self.validation_only_selection_validated,
                self.development_benchmark_completed,
                self.post_grasp_analysis_completed,
                self.raw_action_metrics_validated,
                self.runtime_action_metrics_validated,
                self.physical_target_validated,
            )
        )
        if mode == "target_development":
            return development and not any(
                (
                    self.final_benchmark_completed,
                    self.go_no_go_decision_completed,
                    self.smolvla_go,
                )
            )
        return development and self.final_benchmark_completed and self.go_no_go_decision_completed

    def write(
        self,
        *,
        mode: VerificationMode,
        dataset_root: Path,
        output_root: Path,
        dry_run: bool = False,
    ) -> None:
        payload = {
            "schema_version": "langmani-m42-verification-v0",
            "verification_mode": mode,
            "dry_run": dry_run,
            "dataset_root": str(dataset_root.resolve()),
            "output_root": str(output_root.resolve()),
            **{key: value for key, value in asdict(self).items() if key != "checks"},
            "checks": self.checks,
            "passed": self.accepted(mode, dry_run=dry_run),
        }
        atomic_write_json(output_root / "verification.json", payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--target-development", action="store_true")
    modes.add_argument("--target-final", action="store_true")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--m4-model-root", type=Path, default=DEFAULT_M4_MODEL_ROOT)
    parser.add_argument("--task-token-model-root", type=Path, default=DEFAULT_M42_MODEL_ROOT)
    parser.add_argument("--m4-diagnostics-root", type=Path, default=DEFAULT_M4_DIAGNOSTICS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_M42_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run and not (args.target_development or args.target_final):
        parser.error("--dry-run requires an explicit target mode")
    return args


def _mode(args: argparse.Namespace) -> VerificationMode:
    if args.target_development:
        return "target_development"
    if args.target_final:
        return "target_final"
    return "structural"


def _statistics() -> dict[str, dict[str, torch.Tensor]]:
    return {
        IMAGE_FEATURE_KEY: {
            "mean": torch.zeros(3, 1, 1),
            "std": torch.ones(3, 1, 1),
            "min": torch.zeros(3, 1, 1),
            "max": torch.ones(3, 1, 1),
        },
        STATE_FEATURE_KEY: {
            "mean": torch.zeros(9),
            "std": torch.ones(9),
            "min": -torch.ones(9),
            "max": torch.ones(9),
        },
        ACTION_FEATURE_KEY: {
            "mean": torch.zeros(8),
            "std": torch.ones(8),
            "min": -torch.ones(8),
            "max": torch.ones(8),
        },
    }


def _fixture_model() -> ActModelConfig:
    return ActModelConfig(
        chunk_size=4,
        n_action_steps=2,
        dim_model=32,
        n_heads=2,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        latent_dim=8,
        n_vae_encoder_layers=1,
        dropout=0.0,
    )


def _task_token_fixture() -> bool:
    torch.manual_seed(42)
    token_config = TaskTokenConfig(hidden_dimension=32)
    config = build_task_token_act_config(
        _fixture_model(), device="cpu", use_amp=False, token_config=token_config
    )
    policy, preprocessor, postprocessor = build_policy_and_processors(config, _statistics())
    raw = {
        IMAGE_FEATURE_KEY: torch.randint(0, 256, (1, 3, 256, 256), dtype=torch.uint8),
        STATE_FEATURE_KEY: torch.zeros(1, 9),
        ACTION_FEATURE_KEY: torch.zeros(1, 4, 8),
        "action_is_pad": torch.tensor([[False, False, False, True]]),
        "episode_index": torch.tensor([1], dtype=torch.int64),
    }
    augmented = inject_task_token_training_batch(
        raw,
        task_id_by_episode={1: "langmani-pick-place-task-v0:red_cube:left_bin:canonical_v0"},
    )
    processed = preprocessor(prepare_raw_batch(augmented))
    loss, _ = policy.forward(processed)
    if not torch.isfinite(loss):
        return False
    loss.backward()
    validate_task_token_policy(policy, token_config=token_config)
    gradient = policy.model.encoder_env_state_input_proj.weight.grad
    if gradient is None or not torch.isfinite(gradient).all():
        return False
    with tempfile.TemporaryDirectory(prefix="langmani-m42-fixture-") as temporary:
        root = Path(temporary)
        policy.save_pretrained(root, push_to_hub=False)
        preprocessor.save_pretrained(root, push_to_hub=False)
        postprocessor.save_pretrained(root, push_to_hub=False)
        reloaded = ACTPolicy.from_pretrained(root, local_files_only=True, strict=True)
        validate_task_token_policy(reloaded, token_config=token_config)
    return True


def _structural_checks(report: Report) -> None:
    dev, final = validate_locked_schedules()
    report.development_schedule_locked = dev.schedule_fingerprint == M42_DEV_SCHEDULE_FINGERPRINT
    report.final_schedule_locked = final.schedule_fingerprint == M42_FINAL_SCHEDULE_FINGERPRINT
    report.check(
        "locked development schedule",
        report.development_schedule_locked,
        f"scenes={len(dev.ordered_scene_seeds)}; fingerprint={dev.schedule_fingerprint}",
    )
    report.check(
        "locked sealed final schedule",
        report.final_schedule_locked,
        f"scenes={len(final.ordered_scene_seeds)}; fingerprint={final.schedule_fingerprint}",
    )
    exclusion = load_exclusion_sources()
    schedule_safe = (
        len(exclusion.excluded_scene_seeds) == 125
        and not set(dev.ordered_scene_seeds) & exclusion.excluded_scene_seeds
        and not set(final.ordered_scene_seeds) & exclusion.excluded_scene_seeds
        and not set(dev.ordered_scene_seeds) & set(final.ordered_scene_seeds)
    )
    report.check("seed leakage exclusion", schedule_safe, "125 historical seeds excluded")
    episodes = materialize_locked_schedule("m42_dev_v0")
    report.check("development episode contract", len(episodes) == 72, "12 scenes x 6 tasks")
    horizons_ok = tuple(
        ExecutionHorizonConfig(value).actions_per_query for value in (10, 5, 1)
    ) == (
        10,
        5,
        1,
    )
    report.check("execution horizon contract", horizons_ok, "only H=10,5,1 are accepted")
    binary = BinaryGripperEnvPostprocessorV0(low=torch.full((8,), -1.0), high=torch.ones(8))
    transformed = binary.transform(
        torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0]]), rollout_step=1
    )
    binary_ok = (
        torch.equal(transformed[0, :7], torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))
        and transformed[0, 7].item() == 1.0
    )
    report.check("binary gripper contract", binary_ok, "only eighth component changed")
    trace = PostGraspTrace(
        success=False,
        environment_failure=False,
        wrong_object_interaction=False,
        ever_grasped_target=True,
        ever_lifted_target=True,
        ever_transported_to_destination=False,
        ever_descended_at_destination=False,
        release_command_observed=False,
        released_inside_success_region=False,
        final_target_static=False,
        success_observed_before_final_step=False,
        timed_out=True,
    )
    report.check(
        "post-grasp classifier",
        classify_post_grasp_phase(trace) is PostGraspPhase.TIMED_OUT_AFTER_TARGET_GRASP,
        "timeout after target grasp remains explicit",
    )
    fixture_ok = _task_token_fixture()
    report.check(
        "TaskToken public ACT fixture",
        fixture_ok,
        "9D Panda state plus dedicated ENV token forward/backward/reload",
    )
    gate = create_go_no_go_decision(
        metrics=M42GoNoGoMetrics(
            task_token_successes=126,
            task_token_trials=180,
            per_task_aggregate_successes=143,
            per_task_aggregate_trials=180,
            task_token_successes_by_task=(21, 21, 21, 21, 21, 21),
            timeout_count=54,
            wrong_object_grasp_count=14,
            wrong_object_in_target_bin_count=5,
            target_in_wrong_bin_count=0,
            target_off_table_count=0,
            arm_projection_count=0,
            nan_count=0,
            inf_count=0,
            malformed_action_count=0,
            task_sensitivity_ratio=0.75,
        ),
        final_benchmark_fingerprint="sha256:" + "a" * 64,
    )
    report.check(
        "sealed go/no-go thresholds",
        gate.decision is M42Decision.GO_FOR_SMOLVLA,
        "exact threshold boundary calculated without running the final schedule",
    )
    report.implementation_validated = not report.failed


def _run_json(command: list[str], report_path: Path) -> dict[str, object]:
    print("[RUN] " + " ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with rc={completed.returncode}: {' '.join(command)}")
    value = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("passed") is not True:
        raise RuntimeError(f"command report is absent or failed: {report_path}")
    return value


def _target_preflight(report: Report, args: argparse.Namespace) -> object:
    native = platform.system() == "Linux" and torch.cuda.is_available()
    report.check(
        "native Linux CUDA target",
        native,
        f"platform={platform.system()}; cuda={torch.cuda.is_available()}",
    )
    if not native:
        raise RuntimeError("M4.2 target verification requires native Linux CUDA")
    git = inspect_git_state(PROJECT_ROOT)
    report.check(
        "clean tracked Git target", not git.dirty, f"commit={git.commit}; dirty={git.dirty}"
    )
    if git.dirty:
        raise RuntimeError("M4.2 target evidence requires a clean Git worktree")
    evidence = load_prior_m4_evidence(
        diagnostics_root=args.m4_diagnostics_root,
        checkpoint_root=args.m4_model_root,
        dataset_root=args.dataset_root,
        validate_dataset_storage=True,
    )
    report.prior_m4_evidence_validated = True
    report.check(
        "completed M4 full evidence",
        True,
        f"eight selected checkpoints; evidence={evidence.fingerprint}",
    )
    return evidence


def _json_object(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or path.is_junction() or not path.is_file():
        raise RuntimeError(f"unsafe or absent {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain one JSON object: {path}")
    return cast(dict[str, object], value)


def _task_token_content_matches_plan(
    *,
    manifest: TaskTokenTrainingManifest,
    queue: TaskTokenValidationQueue,
    plan: Mapping[str, object],
    historical_source_fingerprint: str,
) -> bool:
    """Ignore only the historical Git commit while binding all training semantics."""

    identity = manifest.identity.to_dict()
    stable_pairs = (
        (identity["m3b_export_fingerprint"], plan.get("dataset_fingerprint")),
        (identity["m3b_split_manifest_digest"], plan.get("split_digest")),
        (identity["train_statistics_fingerprint"], plan.get("train_statistics_fingerprint")),
        (identity["runtime_selection_fingerprint"], plan.get("runtime_selection_fingerprint")),
        (
            identity["runtime_selection_source_fingerprint"],
            plan.get("runtime_selection_source_fingerprint"),
        ),
        (
            identity["experiment_manifest_fingerprint"],
            plan.get("experiment_manifest_fingerprint"),
        ),
        (
            identity["task_token_architecture_fingerprint"],
            plan.get("task_token_architecture_fingerprint"),
        ),
        (identity["optimization_config"], plan.get("optimization")),
        (
            canonical_fingerprint(cast(Mapping[str, object], identity["model_config"])),
            plan.get("model_contract_fingerprint"),
        ),
        (
            canonical_fingerprint(cast(Mapping[str, object], identity["data_contract"])),
            plan.get("data_contract_fingerprint"),
        ),
        (
            historical_source_fingerprint,
            plan.get("task_token_training_source_fingerprint"),
        ),
    )
    if any(left != right for left, right in stable_pairs):
        return False
    if (
        len(cast(list[object], identity["ordered_train_episode_indices"]))
        != plan.get("train_episode_count")
        or len(cast(list[object], identity["ordered_validation_episode_indices"]))
        != plan.get("validation_episode_count")
        or identity["git_dirty"] is not False
        or not manifest.training_complete
        or not queue.complete
        or queue.run_fingerprint != manifest.identity.run_fingerprint
        or queue.queue_fingerprint == ""
    ):
        return False
    model = cast(Mapping[str, object], identity["model_config"])
    effective = cast(Mapping[str, object], model["effective_act_config"])
    task_experiment = dict(manifest.task_token_experiment)
    task_experiment.pop("implementation_git_commit", None)
    if effective != plan.get("effective_act_config") or canonical_fingerprint(
        task_experiment
    ) != plan.get("task_token_experiment_content_fingerprint"):
        return False
    inputs = cast(Mapping[str, Mapping[str, object]], effective["input_features"])
    outputs = cast(Mapping[str, Mapping[str, object]], effective["output_features"])
    input_shapes = {key: value["shape"] for key, value in inputs.items()}
    output_shapes = {key: value["shape"] for key, value in outputs.items()}
    planned_outputs = plan.get("output_shape")
    return not (
        plan.get("task_token_training_source_files") != list(TASK_TOKEN_TRAINING_SOURCE_FILES)
        or input_shapes != plan.get("input_shapes")
        or output_shapes.get(ACTION_FEATURE_KEY) != planned_outputs
        or [item.global_step for item in manifest.checkpoints] != plan.get("checkpoint_schedule")
        or tuple(item.checkpoint_fingerprint for item in queue.checkpoints)
        != tuple(item.checkpoint_fingerprint for item in manifest.checkpoints)
    )


def _task_token_training_source_fingerprint_at_commit(git_commit: str) -> str:
    sources: dict[str, bytes] = {}
    for name in TASK_TOKEN_TRAINING_SOURCE_FILES:
        completed = subprocess.run(
            ["git", "show", f"{git_commit}:{name}"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"cannot audit historical TaskToken source {name}: {detail}")
        sources[name] = completed.stdout
    return task_token_training_source_fingerprint(sources)


def _find_content_bound_task_token_training(
    *, model_root: Path, plan: Mapping[str, object]
) -> dict[str, object] | None:
    """Reuse one complete semantic match even when only the verifier commit changed."""

    if not model_root.exists():
        return None
    runs = fingerprint_owned_task_token_runs(model_root)
    if len(runs) > 1:
        raise RuntimeError(
            "multiple fingerprint-owned TaskToken runs, complete or incomplete, are forbidden"
        )
    if not runs:
        return None
    run_root = runs[0]
    expected_fingerprint = plan.get("run_fingerprint")
    if (
        not isinstance(expected_fingerprint, str)
        or not expected_fingerprint.startswith("sha256:")
        or run_root.name != expected_fingerprint.removeprefix("sha256:")
    ):
        raise RuntimeError("the existing TaskToken run identity differs from the requested plan")
    completion_path = run_root / "training_complete.json"
    if not completion_path.exists():
        # The sole matching run may be resumed by the training command.  It is
        # not reusable evidence until the atomic completion marker exists.
        return None
    if (
        completion_path.is_symlink()
        or completion_path.is_junction()
        or not completion_path.is_file()
    ):
        raise RuntimeError(f"unsafe completed TaskToken marker: {completion_path}")
    manifest_path = run_root / "run_manifest.json"
    queue_path = run_root / "validation_queue.json"
    summary_path = run_root / "reports" / "training_summary.json"
    completion = _json_object(completion_path, label="TaskToken completion marker")
    summary = _json_object(summary_path, label="TaskToken training summary")
    manifest = TaskTokenTrainingManifest.from_dict(
        _json_object(manifest_path, label="TaskToken run manifest")
    )
    queue = TaskTokenValidationQueue.from_dict(
        _json_object(queue_path, label="TaskToken validation queue")
    )
    historical_source_fingerprint = _task_token_training_source_fingerprint_at_commit(
        manifest.identity.git_commit
    )
    if not _task_token_content_matches_plan(
        manifest=manifest,
        queue=queue,
        plan=plan,
        historical_source_fingerprint=historical_source_fingerprint,
    ):
        raise RuntimeError(
            "the existing complete TaskToken run differs from the requested data, runtime, "
            "model, or optimization content"
        )
    final_checkpoint = manifest.checkpoints[-1]
    if (
        len(manifest.checkpoints) != 20
        or manifest.training_state.global_step != 100_000
        or completion.get("schema_version") != "langmani-m42-task-token-training-complete-v0"
        or completion.get("run_fingerprint") != manifest.identity.run_fingerprint
        or completion.get("experiment_manifest_fingerprint")
        != manifest.identity.experiment_manifest_fingerprint
        or completion.get("training_summary_fingerprint") != f"sha256:{sha256_hex(summary)}"
        or completion.get("validation_queue_fingerprint") != queue.queue_fingerprint
        or completion.get("final_checkpoint_fingerprint") != final_checkpoint.checkpoint_fingerprint
        or summary.get("passed") is not True
        or summary.get("schema_version") != "langmani-m42-task-token-training-summary-v0"
        or summary.get("task_token_training_completed") is not True
        or summary.get("checkpoint_count") != 20
        or summary.get("run_fingerprint") != manifest.identity.run_fingerprint
        or summary.get("experiment_manifest_fingerprint")
        != manifest.identity.experiment_manifest_fingerprint
        or summary.get("validation_queue_fingerprint") != queue.queue_fingerprint
        or summary.get("final_checkpoint_fingerprint") != final_checkpoint.checkpoint_fingerprint
    ):
        raise RuntimeError("completed TaskToken training evidence is internally inconsistent")
    for checkpoint in manifest.checkpoints:
        marker = run_root / checkpoint.relative_path / CHECKPOINT_COMPLETION_MARKER
        if not marker.is_file() or marker.is_symlink() or marker.is_junction():
            raise RuntimeError(f"TaskToken checkpoint is not atomically complete: {marker}")
    model = cast(Mapping[str, object], manifest.identity.model_config)
    token = cast(Mapping[str, object], model["task_token"])
    token_config = TaskTokenConfig(hidden_dimension=cast(int, token["hidden_dimension"]))
    if canonical_fingerprint(token_config.to_dict()) != canonical_fingerprint(token):
        raise RuntimeError("completed TaskToken token configuration is not canonical")
    reloaded = load_act_checkpoint(
        run_root=run_root,
        checkpoint_relative_path=final_checkpoint.relative_path,
        expected_identity=manifest.identity,
    )
    validate_task_token_policy(reloaded.policy, token_config=token_config)
    if reloaded.record.checkpoint_fingerprint != final_checkpoint.checkpoint_fingerprint:
        raise RuntimeError("fresh TaskToken reload changed the final checkpoint fingerprint")
    return {
        "passed": True,
        "task_token_training_completed": True,
        "training_reused": True,
        "content_bound_cross_commit_reuse": (
            manifest.identity.git_commit != cast(Mapping[str, object], plan["git"])["commit"]
        ),
        "run_fingerprint": manifest.identity.run_fingerprint,
        "run_manifest": str(manifest_path),
        "validation_queue": str(queue_path),
    }


def _runtime_command(
    args: argparse.Namespace, *, output_root: Path, report_path: Path, dry_run: bool
) -> list[str]:
    return [
        sys.executable,
        "scripts/run_m42_runtime_ablation.py",
        "--dataset-root",
        str(args.dataset_root.resolve()),
        "--checkpoint-root",
        str(args.m4_model_root.resolve()),
        "--m4-diagnostics-root",
        str(args.m4_diagnostics_root.resolve()),
        "--schedule",
        "m42_dev_v0",
        "--output-root",
        str(output_root),
        "--report",
        str(report_path),
    ] + (["--dry-run"] if dry_run else [])


def _training_command(
    args: argparse.Namespace,
    *,
    runtime_selection: Path,
    report_path: Path,
    dry_run: bool,
) -> list[str]:
    return [
        sys.executable,
        "scripts/train_act_task_token.py",
        "--dataset-root",
        str(args.dataset_root.resolve()),
        "--output-root",
        str(args.task_token_model_root.resolve()),
        "--runtime-selection",
        str(runtime_selection),
        "--report",
        str(report_path),
        "--dry-run" if dry_run else "--full",
        "--device",
        "cuda",
    ]


def _evaluation_command(
    args: argparse.Namespace,
    *,
    stage: Literal["development", "final"],
    output_root: Path,
    report_path: Path,
    dry_run: bool,
    expected_implementation_fingerprint: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/evaluate_m42.py",
        "--stage",
        stage,
        "--schedule",
        "m42_dev_v0" if stage == "development" else "m42_final_v0",
        "--dataset-root",
        str(args.dataset_root.resolve()),
        "--checkpoint-root",
        str(args.m4_model_root.resolve()),
        "--task-token-root",
        str(args.task_token_model_root.resolve()),
        "--m4-diagnostics-root",
        str(args.m4_diagnostics_root.resolve()),
        "--runtime-selection",
        str(args.output_root.resolve() / "runtime_ablation" / "runtime_selection.json"),
        "--output-root",
        str(output_root),
        "--report",
        str(report_path),
    ]
    if dry_run:
        command.append("--dry-run")
    elif stage == "final":
        if expected_implementation_fingerprint is None:
            raise RuntimeError("sealed final execution requires its dry-run implementation digest")
        command.extend(
            [
                "--authorize-sealed-final",
                "--expected-implementation-fingerprint",
                expected_implementation_fingerprint,
            ]
        )
    return command


def _run_target_development(report: Report, args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    if args.dry_run:
        with tempfile.TemporaryDirectory(prefix="langmani-m42-verifier-plan-") as temporary:
            planning_root = Path(temporary)
            runtime_report = planning_root / "runtime_ablation_command.json"
            runtime_command = _runtime_command(
                args,
                output_root=planning_root / "runtime_ablation",
                report_path=runtime_report,
                dry_run=True,
            )
            runtime = _run_json(
                runtime_command,
                runtime_report,
            )
            runtime_selection = output_root / "runtime_ablation" / "runtime_selection.json"
            training_report = planning_root / "task_token_training_command.json"
            training_command = _training_command(
                args,
                runtime_selection=runtime_selection,
                report_path=training_report,
                dry_run=True,
            )
            training_deferred = not runtime_selection.is_file()
            if training_deferred:
                training = {
                    "passed": True,
                    "training_started": False,
                    "task_token_training_completed": False,
                    "deferred": True,
                    "deferred_reason": (
                        "TaskToken identity depends on the development-only runtime selection; "
                        "the dry-run does not fabricate that physical selection"
                    ),
                }
            else:
                training = _run_json(training_command, training_report)
            development_report = planning_root / "development_evaluation_command.json"
            development_command = _evaluation_command(
                args,
                stage="development",
                output_root=planning_root / "development",
                report_path=development_report,
                dry_run=True,
            )
            development = _run_json(development_command, development_report)
        report.dry_run_plan = {
            "schema_version": "langmani-m42-verifier-dry-run-plan-v0",
            "runtime_ablation": {
                "command": runtime_command,
                "executed": True,
                "physical_execution": False,
            },
            "task_token_training": {
                "command": training_command,
                "executed": not training_deferred,
                "deferred": training_deferred,
                "deferred_reason": training.get("deferred_reason"),
                "physical_execution": False,
            },
            "development_evaluation": {
                "command": development_command,
                "executed": True,
                "physical_execution": False,
            },
            "final_schedule_accessed": False,
        }
        dry_run_truthful = all(
            (
                runtime.get("physical_execution") is False,
                runtime.get("horizon_ablation_completed") is False,
                training.get("training_started") is False,
                training.get("task_token_training_completed") is False,
                development.get("physical_execution") is False,
                development.get("final_schedule_accessed") is False,
                development.get("development_benchmark_completed") is False,
            )
        )
        report.dry_run_validated = dry_run_truthful
        report.check(
            "truthful target-development dry-run",
            dry_run_truthful,
            (
                "runtime, TaskToken, and development commands planned without physical execution; "
                f"TaskToken deferred={training_deferred}"
            ),
        )
        return

    runtime_report = output_root / "runtime_ablation_command.json"
    runtime = _run_json(
        _runtime_command(
            args,
            output_root=output_root / "runtime_ablation",
            report_path=runtime_report,
            dry_run=False,
        ),
        runtime_report,
    )
    runtime_boundary_valid = all(
        (
            runtime.get("physical_execution") is True,
            runtime.get("final_schedule_accessed") is False,
        )
    )
    report.check(
        "runtime ablation stayed on development schedule",
        runtime_boundary_valid,
        "physical_execution=true and final_schedule_accessed=false are required",
    )
    if not runtime_boundary_valid:
        raise RuntimeError("runtime ablation violated the development-only execution boundary")
    report.horizon_ablation_completed = bool(runtime["horizon_ablation_completed"])
    report.horizon_selection_locked = bool(runtime["horizon_selection_locked"])
    report.gripper_ablation_completed = bool(runtime["gripper_ablation_completed"])
    report.gripper_selection_locked = bool(runtime["gripper_selection_locked"])
    report.post_grasp_analysis_completed = bool(runtime["post_grasp_analysis_completed"])
    report.raw_action_metrics_validated = bool(runtime["raw_action_metrics_validated"])
    report.runtime_action_metrics_validated = bool(runtime["runtime_action_metrics_validated"])
    runtime_selection = output_root / "runtime_ablation" / "runtime_selection.json"
    training_plan_report = output_root / "task_token_training_plan.json"
    training_plan = _run_json(
        _training_command(
            args,
            runtime_selection=runtime_selection,
            report_path=training_plan_report,
            dry_run=True,
        ),
        training_plan_report,
    )
    training_report = output_root / "task_token_training_command.json"
    training = _find_content_bound_task_token_training(
        model_root=args.task_token_model_root.resolve(), plan=training_plan
    )
    if training is None:
        training = _run_json(
            _training_command(
                args,
                runtime_selection=runtime_selection,
                report_path=training_report,
                dry_run=False,
            ),
            training_report,
        )
    else:
        report.check(
            "content-bound TaskToken training reuse",
            True,
            "one complete run matches data, runtime, architecture, optimization, and checkpoints",
        )
    report.task_token_training_completed = bool(training["task_token_training_completed"])

    development_report = output_root / "development_evaluation_command.json"
    development = _run_json(
        _evaluation_command(
            args,
            stage="development",
            output_root=output_root / "development",
            report_path=development_report,
            dry_run=False,
        ),
        development_report,
    )
    development_boundary_valid = all(
        (
            development.get("physical_execution") is True,
            development.get("final_schedule_accessed") is False,
            development.get("final_benchmark_completed") is False,
            development.get("go_no_go_decision_completed") is False,
            development.get("smolvla_go") is False,
        )
    )
    report.check(
        "policy evaluation stopped before sealed final",
        development_boundary_valid,
        "development evidence is physical but final/go-no-go flags remain false",
    )
    if not development_boundary_valid:
        raise RuntimeError("development evaluation crossed the sealed-final boundary")
    report.development_benchmark_completed = bool(development["development_benchmark_completed"])
    report.task_token_checkpoint_selected = bool(development["task_token_checkpoint_selected"])
    report.validation_only_selection_validated = bool(
        development["validation_only_selection_validated"]
    )
    report.post_grasp_analysis_completed &= bool(development["post_grasp_analysis_completed"])
    report.raw_action_metrics_validated &= bool(development["raw_action_metrics_validated"])
    report.runtime_action_metrics_validated &= bool(development["runtime_action_metrics_validated"])
    report.physical_target_validated = all(
        (
            report.horizon_ablation_completed,
            report.gripper_ablation_completed,
            report.task_token_training_completed,
            report.task_token_checkpoint_selected,
            report.development_benchmark_completed,
            runtime_boundary_valid,
            development_boundary_valid,
        )
    )


def _run_target_final(report: Report, args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    if args.dry_run:
        with tempfile.TemporaryDirectory(prefix="langmani-m42-final-plan-") as temporary:
            planning_root = Path(temporary)
            dry_report = planning_root / "final_dry_run_command.json"
            planned = _run_json(
                _evaluation_command(
                    args,
                    stage="final",
                    output_root=planning_root / "final",
                    report_path=dry_report,
                    dry_run=True,
                ),
                dry_report,
            )
            planned_command = _evaluation_command(
                args,
                stage="final",
                output_root=planning_root / "final",
                report_path=dry_report,
                dry_run=True,
            )
        report.dry_run_plan = {
            "schema_version": "langmani-m42-verifier-dry-run-plan-v0",
            "final_evaluation": {
                "command": planned_command,
                "executed": True,
                "physical_execution": False,
                "authorization_granted": False,
            },
            "final_schedule_accessed": False,
        }
        truthful = all(
            (
                isinstance(planned.get("implementation_fingerprint"), str),
                planned.get("physical_execution") is False,
                planned.get("final_schedule_accessed") is False,
                planned.get("final_benchmark_completed") is False,
                planned.get("go_no_go_decision_completed") is False,
            )
        )
        report.dry_run_validated = truthful
        report.check(
            "truthful target-final dry-run",
            truthful,
            "implementation fingerprint obtained without materializing m42_final_v0",
        )
        return

    final_root = output_root / "final"
    dry_report = output_root / "final_dry_run_command.json"
    planned = _run_json(
        _evaluation_command(
            args,
            stage="final",
            output_root=final_root,
            report_path=dry_report,
            dry_run=True,
        ),
        dry_report,
    )
    implementation = planned.get("implementation_fingerprint")
    if not isinstance(implementation, str):
        raise RuntimeError("final dry-run did not provide an implementation fingerprint")
    if planned.get("final_schedule_accessed") is not False:
        raise RuntimeError("final dry-run unexpectedly accessed the sealed schedule")
    physical_report = output_root / "final_evaluation_command.json"
    final = _run_json(
        _evaluation_command(
            args,
            stage="final",
            output_root=final_root,
            report_path=physical_report,
            dry_run=False,
            expected_implementation_fingerprint=implementation,
        ),
        physical_report,
    )
    report.horizon_ablation_completed = True
    report.horizon_selection_locked = True
    report.gripper_ablation_completed = True
    report.gripper_selection_locked = True
    report.task_token_training_completed = True
    report.task_token_checkpoint_selected = bool(final["task_token_checkpoint_selected"])
    report.validation_only_selection_validated = bool(final["validation_only_selection_validated"])
    report.development_benchmark_completed = bool(final["development_benchmark_completed"])
    report.final_benchmark_completed = bool(final["final_benchmark_completed"])
    report.post_grasp_analysis_completed = bool(final["post_grasp_analysis_completed"])
    report.raw_action_metrics_validated = bool(final["raw_action_metrics_validated"])
    report.runtime_action_metrics_validated = bool(final["runtime_action_metrics_validated"])
    report.go_no_go_decision_completed = bool(final["go_no_go_decision_completed"])
    report.smolvla_go = bool(final["smolvla_go"])
    report.physical_target_validated = all(
        (
            final.get("physical_execution") is True,
            report.final_benchmark_completed,
            report.go_no_go_decision_completed,
        )
    )


def main() -> int:
    args = parse_args()
    mode = _mode(args)
    report = Report()
    try:
        output_root = _validate_paths(args)
    except Exception:  # noqa: BLE001 - unsafe report locations must not be written
        traceback.print_exc()
        return 1
    args.output_root = output_root
    try:
        _structural_checks(report)
        if mode != "structural":
            _target_preflight(report, args)
            if mode == "target_development":
                _run_target_development(report, args)
            else:
                _run_target_final(report, args)
        else:
            report.check(
                "truthful physical target status",
                not report.physical_target_validated,
                "structural mode does not claim CUDA or simulator execution",
            )
    except Exception as error:  # noqa: BLE001 - command boundary preserves exact failure
        traceback.print_exc()
        report.check(
            "unexpected verifier exception",
            False,
            f"{type(error).__name__}: {str(error) or repr(error)}",
        )
    report.write(
        mode=mode,
        dataset_root=args.dataset_root,
        output_root=output_root,
        dry_run=bool(args.dry_run),
    )
    print(f"[INFO] report: {output_root / 'verification.json'}")
    return 0 if report.accepted(mode, dry_run=bool(args.dry_run)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
