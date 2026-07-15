"""Verify M4.2 contracts, the development pipeline, or the sealed final benchmark."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

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
    create_gripper_selection,
    create_horizon_selection,
)
from langmani.policies.m42_evidence import PriorM4Evidence, load_prior_m4_evidence
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
    RUNTIME_SELECTION_SCHEMA,
    TaskTokenFairComparisonContract,
    TaskTokenTrainingManifest,
    TaskTokenValidationQueue,
    build_task_token_fair_comparison_contract,
    canonical_fingerprint,
    fingerprint_owned_task_token_runs,
)
from langmani.policies.m42_types import (
    ExecutionHorizonConfig,
    GripperRuntimeMode,
    M42Decision,
    M42ExperimentManifest,
    M42GoNoGoMetrics,
    M42SelectionRecord,
    PostGraspPhase,
    RuntimeAblationConfig,
    RuntimeAblationResult,
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

RUNTIME_LEGACY_IMPLEMENTATION_FILES = (
    "src/langmani/policies/act_rollout.py",
    "src/langmani/policies/m42_evaluation.py",
    "src/langmani/policies/m42_runtime.py",
    "src/langmani/policies/m42_analysis.py",
    "scripts/run_m42_runtime_ablation.py",
)
RUNTIME_SEMANTIC_EXACT_PATHS = (
    "scripts/run_m42_runtime_ablation.py",
    "environment/environment.yml",
    "pyproject.toml",
    "src/langmani/__init__.py",
)
RUNTIME_SEMANTIC_PREFIXES = (
    "src/langmani/policies/",
    "src/langmani/environments/",
    "src/langmani/datasets/",
    "src/langmani/experts/",
)
# These are Git pathspec roots. The exact expanded file list is discovered independently
# from the historical tree and current index, then required to be identical.
RUNTIME_SEMANTIC_SOURCE_FILES = (*RUNTIME_SEMANTIC_EXACT_PATHS, *RUNTIME_SEMANTIC_PREFIXES)
RUNTIME_REUSE_CONSUMER_REPAIR_ID = "M42TaskTokenRuntimeSelectionFrozenTupleThawV0"
RUNTIME_IMPLEMENTATION_SCHEMA = "langmani-m42-runtime-implementation-v0"
RUNTIME_SEMANTIC_CLOSURE_SCHEMA = "langmani-m42-runtime-semantic-closure-v1"
RUNTIME_COMMAND_SCHEMA = "langmani-m42-runtime-ablation-command-v0"
RUNTIME_BENCHMARK_ARTIFACT_SCHEMA = "langmani-m42-runtime-benchmark-artifact-v0"
RUNTIME_POST_GRASP_SCHEMA = "langmani-m42-post-grasp-analysis-v0"
RUNTIME_REPAIR_LINEAGE_SCHEMA = "langmani-m42-runtime-repair-lineage-v0"

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


def _target_preflight(report: Report, args: argparse.Namespace) -> PriorM4Evidence:
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


def _owned_runtime_path(
    root: Path,
    relative: object,
    *,
    label: str,
    kind: Literal["file", "directory"],
) -> Path:
    """Resolve one canonical portable child without following filesystem links."""

    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise RuntimeError(f"{label} must be a non-empty portable relative path")
    portable = PurePosixPath(relative)
    if (
        portable.is_absolute()
        or portable.as_posix() != relative
        or any(part in {"", ".", ".."} for part in portable.parts)
    ):
        raise RuntimeError(f"{label} must be a canonical relative path")
    owned_root = _resolved_unlinked(root, label="runtime evidence root")
    candidate = _resolved_unlinked(owned_root.joinpath(*portable.parts), label=label)
    if candidate == owned_root or not _is_within(candidate, owned_root):
        raise RuntimeError(f"{label} escapes the runtime evidence root")
    valid = candidate.is_file() if kind == "file" else candidate.is_dir()
    if not valid:
        raise RuntimeError(f"{label} is not a real {kind}: {candidate}")
    return candidate


@dataclass(frozen=True, slots=True)
class _TrackedRuntimeSources:
    sources: Mapping[str, bytes]
    modes: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _RuntimeSemanticClosureAudit:
    path_count: int
    closure_fingerprint: str
    historical_raw_fingerprint: str
    current_raw_fingerprint: str
    consumer_repair_id: str


def _validate_runtime_semantic_path(name: str) -> None:
    if not isinstance(name, str) or not name or "\\" in name:
        raise RuntimeError("runtime semantic Git path is not portable")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or path.as_posix() != name
        or any(part in {"", ".", ".."} for part in path.parts)
        or not (
            name in RUNTIME_SEMANTIC_EXACT_PATHS
            or any(name.startswith(prefix) for prefix in RUNTIME_SEMANTIC_PREFIXES)
        )
    ):
        raise RuntimeError(f"unsafe or out-of-scope runtime semantic path: {name!r}")


def _git_bytes(command: list[str], *, label: str) -> bytes:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"cannot audit {label}: {detail}")
    return completed.stdout


def _parse_historical_runtime_tree(git_commit: str) -> dict[str, str]:
    raw = _git_bytes(
        ["git", "ls-tree", "-rz", git_commit, "--", *RUNTIME_SEMANTIC_SOURCE_FILES],
        label="historical runtime semantic tree",
    )
    modes: dict[str, str] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_name = record.split(b"\t", maxsplit=1)
            mode, object_type, _object_id = metadata.split(b" ", maxsplit=2)
            name = raw_name.decode("utf-8", errors="strict")
            decoded_mode = mode.decode("ascii", errors="strict")
        except (UnicodeDecodeError, ValueError) as error:
            raise RuntimeError("historical runtime semantic tree is malformed") from error
        _validate_runtime_semantic_path(name)
        if object_type != b"blob" or decoded_mode not in {"100644", "100755"}:
            raise RuntimeError(f"historical runtime semantic source has unsafe mode: {name}")
        if name in modes:
            raise RuntimeError(f"historical runtime semantic source is duplicated: {name}")
        modes[name] = decoded_mode
    return modes


def _parse_current_runtime_index() -> dict[str, str]:
    raw = _git_bytes(
        ["git", "ls-files", "--stage", "-z", "--", *RUNTIME_SEMANTIC_SOURCE_FILES],
        label="current runtime semantic index",
    )
    modes: dict[str, str] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_name = record.split(b"\t", maxsplit=1)
            mode, _object_id, stage = metadata.split(b" ", maxsplit=2)
            name = raw_name.decode("utf-8", errors="strict")
            decoded_mode = mode.decode("ascii", errors="strict")
        except (UnicodeDecodeError, ValueError) as error:
            raise RuntimeError("current runtime semantic index is malformed") from error
        _validate_runtime_semantic_path(name)
        if stage != b"0" or decoded_mode not in {"100644", "100755"}:
            raise RuntimeError(f"current runtime semantic source has unsafe mode or stage: {name}")
        if name in modes:
            raise RuntimeError(f"current runtime semantic source is duplicated: {name}")
        modes[name] = decoded_mode
    untracked = _git_bytes(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            *RUNTIME_SEMANTIC_SOURCE_FILES,
        ],
        label="untracked runtime semantic sources",
    )
    untracked_paths = tuple(item for item in untracked.split(b"\0") if item)
    if untracked_paths:
        try:
            names = ", ".join(item.decode("utf-8", errors="strict") for item in untracked_paths)
        except UnicodeDecodeError as error:
            raise RuntimeError("untracked runtime semantic path is not UTF-8") from error
        raise RuntimeError(f"untracked runtime semantic source drift is forbidden: {names}")
    return modes


def _runtime_sources_at_commit(git_commit: str) -> _TrackedRuntimeSources:
    if re.fullmatch(r"[0-9a-f]{40,64}", git_commit) is None:
        raise RuntimeError("runtime producer Git commit is malformed")
    modes = _parse_historical_runtime_tree(git_commit)
    sources = {
        name: _git_bytes(
            ["git", "show", f"{git_commit}:{name}"],
            label=f"historical runtime source {name}",
        )
        for name in sorted(modes)
    }
    return _TrackedRuntimeSources(sources=sources, modes=modes)


def _current_runtime_sources() -> _TrackedRuntimeSources:
    modes = _parse_current_runtime_index()
    sources: dict[str, bytes] = {}
    project_root = _resolved_unlinked(PROJECT_ROOT, label="project root")
    for name in sorted(modes):
        path = _resolved_unlinked(PROJECT_ROOT / name, label="current runtime semantic source")
        if not path.is_file() or not _is_within(path, project_root):
            raise RuntimeError(f"current runtime semantic source is absent or unsafe: {name}")
        sources[name] = path.read_bytes()
    return _TrackedRuntimeSources(sources=sources, modes=modes)


def _raw_runtime_source_fingerprint(value: _TrackedRuntimeSources) -> str:
    return canonical_fingerprint(
        {
            "schema_version": RUNTIME_SEMANTIC_CLOSURE_SCHEMA,
            "normalization": "none",
            "files": {
                name: {
                    "mode": value.modes[name],
                    "sha256": f"sha256:{sha256_hex(content.hex())}",
                }
                for name, content in sorted(value.sources.items())
            },
        }
    )


def _audit_runtime_semantic_sources(
    historical: _TrackedRuntimeSources, current: _TrackedRuntimeSources
) -> _RuntimeSemanticClosureAudit:
    if set(historical.sources) != set(historical.modes) or set(current.sources) != set(
        current.modes
    ):
        raise RuntimeError("runtime semantic source bundle is internally incomplete")
    if set(historical.sources) != set(current.sources):
        raise RuntimeError("historical and current runtime semantic tracked path sets differ")
    if dict(historical.modes) != dict(current.modes):
        raise RuntimeError("historical and current runtime semantic file modes differ")
    for name in sorted(historical.sources):
        if historical.sources[name] != current.sources[name]:
            raise RuntimeError(f"runtime semantic source changed: {name}")
    closure_fingerprint = canonical_fingerprint(
        {
            "schema_version": RUNTIME_SEMANTIC_CLOSURE_SCHEMA,
            "comparison": "tracked-path-set-mode-and-bytes-identical",
            "files": {
                name: {
                    "mode": historical.modes[name],
                    "sha256": f"sha256:{sha256_hex(content.hex())}",
                }
                for name, content in sorted(historical.sources.items())
            },
        }
    )
    return _RuntimeSemanticClosureAudit(
        path_count=len(historical.sources),
        closure_fingerprint=closure_fingerprint,
        historical_raw_fingerprint=_raw_runtime_source_fingerprint(historical),
        current_raw_fingerprint=_raw_runtime_source_fingerprint(current),
        consumer_repair_id=RUNTIME_REUSE_CONSUMER_REPAIR_ID,
    )


def _runtime_implementation_fingerprint(git_commit: str, sources: _TrackedRuntimeSources) -> str:
    """Reproduce the runtime producer's historical five-file digest exactly."""

    if re.fullmatch(r"[0-9a-f]{40,64}", git_commit) is None:
        raise RuntimeError("runtime producer Git commit is malformed")
    if not set(RUNTIME_LEGACY_IMPLEMENTATION_FILES).issubset(sources.sources):
        raise RuntimeError("historical producer implementation source set is incomplete")
    return canonical_fingerprint(
        {
            "schema_version": RUNTIME_IMPLEMENTATION_SCHEMA,
            "git_commit": git_commit,
            "files": {
                name: f"sha256:{sha256_hex(sources.sources[name].hex())}"
                for name in RUNTIME_LEGACY_IMPLEMENTATION_FILES
            },
        }
    )


def _runtime_result_from_mapping(value: object) -> RuntimeAblationResult:
    if not isinstance(value, Mapping):
        raise RuntimeError("runtime benchmark result must be a JSON object")
    steps = value.get("successful_episode_steps")
    if not isinstance(steps, list):
        raise RuntimeError("runtime successful_episode_steps must be a JSON array")
    action_metrics = value.get("action_metrics")
    latency_metrics = value.get("latency_metrics")
    if not isinstance(action_metrics, Mapping) or not isinstance(latency_metrics, Mapping):
        raise RuntimeError("runtime action and latency metrics must be JSON objects")
    try:
        return RuntimeAblationResult(
            config_fingerprint=cast(str, value["config_fingerprint"]),
            schedule_id=cast(str, value["schedule_id"]),
            schedule_fingerprint=cast(str, value["schedule_fingerprint"]),
            model_label=cast(str, value["model_label"]),
            task_id=cast(str | None, value.get("task_id")),
            execution_horizon=cast(int, value["execution_horizon"]),
            gripper_mode=GripperRuntimeMode(cast(str, value["gripper_mode"])),
            episode_count=cast(int, value["episode_count"]),
            successes=cast(int, value["successes"]),
            post_grasp_timeouts=cast(int, value["post_grasp_timeouts"]),
            wrong_object_interactions=cast(int, value["wrong_object_interactions"]),
            wrong_object_grasp_count=cast(int, value["wrong_object_grasp_count"]),
            wrong_object_in_target_bin_count=cast(int, value["wrong_object_in_target_bin_count"]),
            target_in_wrong_bin_count=cast(int, value["target_in_wrong_bin_count"]),
            target_off_table_count=cast(int, value["target_off_table_count"]),
            invalid_action_count=cast(int, value["invalid_action_count"]),
            successful_episode_steps=tuple(cast(list[int], steps)),
            policy_query_count=cast(int, value["policy_query_count"]),
            release_sign_transitions=cast(int, value["release_sign_transitions"]),
            grasp_sign_transitions=cast(int, value["grasp_sign_transitions"]),
            unnecessary_gripper_sign_transitions=cast(
                int, value["unnecessary_gripper_sign_transitions"]
            ),
            action_metrics=cast(Mapping[str, object], action_metrics),
            latency_metrics=cast(Mapping[str, object], latency_metrics),
            report_fingerprint=cast(str | None, value.get("report_fingerprint")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("runtime benchmark result is malformed") from error


@dataclass(frozen=True, slots=True)
class _AuditedRuntimeArtifact:
    relative_path: str
    identity_fingerprint: str
    artifact_fingerprint: str
    benchmark_fingerprint: str
    result: RuntimeAblationResult
    benchmark: Mapping[str, object]


def _expected_runtime_episodes(
    evidence: PriorM4Evidence, *, model_label: str
) -> tuple[object, ...]:
    episodes = materialize_locked_schedule("m42_dev_v0")
    if model_label == "ACT-Mixed-TaskOneHot":
        return cast(tuple[object, ...], episodes)
    if model_label == "ACT-PerTask-Representative":
        return tuple(
            item for item in episodes if item.task_id == evidence.representative_per_task.task_id
        )
    raise RuntimeError(f"unknown runtime benchmark model label: {model_label}")


def _audit_runtime_artifact(
    *,
    runtime_root: Path,
    relative_path: str,
    producer_commit: str,
    producer_implementation_fingerprint: str,
    evidence: PriorM4Evidence,
    horizon_config_fingerprint: str,
    gripper_config_fingerprint: str,
) -> _AuditedRuntimeArtifact:
    directory = _owned_runtime_path(
        runtime_root,
        relative_path,
        label="runtime benchmark evidence directory",
        kind="directory",
    )
    portable = PurePosixPath(relative_path)
    if (
        len(portable.parts) != 2
        or portable.parts[0] != "evidence"
        or re.fullmatch(r"[0-9a-f]{64}", portable.parts[1]) is None
    ):
        raise RuntimeError("runtime benchmark evidence path is not content addressed")
    entries = {item.name for item in directory.iterdir()}
    if entries != {"owner.json", "benchmark.json", "complete.json"}:
        raise RuntimeError(f"runtime evidence directory has extra or missing files: {directory}")
    owner_path = _owned_runtime_path(
        runtime_root,
        f"{relative_path}/owner.json",
        label="runtime benchmark owner",
        kind="file",
    )
    benchmark_path = _owned_runtime_path(
        runtime_root,
        f"{relative_path}/benchmark.json",
        label="runtime benchmark artifact",
        kind="file",
    )
    complete_path = _owned_runtime_path(
        runtime_root,
        f"{relative_path}/complete.json",
        label="runtime benchmark completion marker",
        kind="file",
    )
    owner = _json_object(owner_path, label="runtime benchmark owner")
    artifact = _json_object(benchmark_path, label="runtime benchmark artifact")
    complete = _json_object(complete_path, label="runtime benchmark completion marker")
    identity = artifact.get("benchmark_identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError("runtime benchmark artifact lacks a JSON identity object")
    identity_fingerprint = canonical_fingerprint(cast(Mapping[str, object], identity))
    artifact_fingerprint = canonical_fingerprint(artifact)
    if (
        identity_fingerprint != f"sha256:{portable.parts[1]}"
        or owner
        != {
            "identity_fingerprint": identity_fingerprint,
            "benchmark_identity": dict(identity),
        }
        or complete
        != {
            "schema_version": RUNTIME_BENCHMARK_ARTIFACT_SCHEMA,
            "identity_fingerprint": identity_fingerprint,
            "artifact_fingerprint": artifact_fingerprint,
            "passed": True,
        }
    ):
        raise RuntimeError("runtime benchmark identity/artifact/completion fingerprints disagree")
    required_identity = {
        "schema_version",
        "evaluation_git_commit",
        "implementation_fingerprint",
        "schedule_id",
        "schedule_fingerprint",
        "model_label",
        "task_id",
        "run_fingerprint",
        "checkpoint_fingerprint",
        "execution_horizon",
        "gripper_mode",
        "config_fingerprint",
        "maximum_episode_steps",
        "ordered_episode_indices",
        "ordered_scene_seeds",
        "ordered_task_ids",
        "final_schedule_accessed",
    }
    if set(identity) != required_identity:
        raise RuntimeError("runtime benchmark identity fields differ from the frozen contract")
    model_label = identity.get("model_label")
    task_id = identity.get("task_id")
    if model_label == "ACT-Mixed-TaskOneHot":
        checkpoint = evidence.mixed_task_onehot
        expected_task_id = None
    elif model_label == "ACT-PerTask-Representative":
        checkpoint = evidence.representative_per_task
        expected_task_id = checkpoint.task_id
    else:
        raise RuntimeError("runtime benchmark uses an unknown model")
    gripper_mode = GripperRuntimeMode(cast(str, identity.get("gripper_mode")))
    expected_config = (
        horizon_config_fingerprint
        if gripper_mode is GripperRuntimeMode.PROJECT
        else gripper_config_fingerprint
    )
    expected_episodes = _expected_runtime_episodes(evidence, model_label=cast(str, model_label))
    expected_indices = [cast(Any, item).episode_index for item in expected_episodes]
    expected_seeds = [cast(Any, item).scene_seed for item in expected_episodes]
    expected_tasks = [cast(Any, item).task_id for item in expected_episodes]
    if any(
        (
            identity.get("schema_version") != RUNTIME_BENCHMARK_ARTIFACT_SCHEMA,
            identity.get("evaluation_git_commit") != producer_commit,
            identity.get("implementation_fingerprint") != producer_implementation_fingerprint,
            identity.get("schedule_id") != "m42_dev_v0",
            identity.get("schedule_fingerprint") != M42_DEV_SCHEDULE_FINGERPRINT,
            task_id != expected_task_id,
            identity.get("run_fingerprint") != checkpoint.run_fingerprint,
            identity.get("checkpoint_fingerprint") != checkpoint.checkpoint_fingerprint,
            identity.get("config_fingerprint") != expected_config,
            identity.get("maximum_episode_steps") != 200,
            identity.get("ordered_episode_indices") != expected_indices,
            identity.get("ordered_scene_seeds") != expected_seeds,
            identity.get("ordered_task_ids") != expected_tasks,
            identity.get("final_schedule_accessed") is not False,
        )
    ):
        raise RuntimeError("runtime benchmark identity differs from current locked evidence")
    benchmark = artifact.get("benchmark")
    runtime_result = artifact.get("runtime_result")
    if not isinstance(benchmark, Mapping) or not isinstance(runtime_result, Mapping):
        raise RuntimeError("runtime benchmark payload or aggregate result is absent")
    benchmark_fingerprint = canonical_fingerprint(cast(Mapping[str, object], benchmark))
    if (
        artifact.get("schema_version") != RUNTIME_BENCHMARK_ARTIFACT_SCHEMA
        or artifact.get("passed") is not True
        or artifact.get("benchmark_fingerprint") != benchmark_fingerprint
    ):
        raise RuntimeError("runtime benchmark artifact fingerprint is invalid")
    descriptor = benchmark.get("checkpoint")
    episodes = benchmark.get("episodes")
    aggregate = benchmark.get("aggregate")
    if (
        not isinstance(descriptor, Mapping)
        or not isinstance(episodes, list)
        or not isinstance(aggregate, Mapping)
    ):
        raise RuntimeError("runtime benchmark report structure is malformed")
    if any(
        (
            benchmark.get("schedule_id") != identity.get("schedule_id"),
            benchmark.get("schedule_fingerprint") != identity.get("schedule_fingerprint"),
            benchmark.get("model_label") != model_label,
            benchmark.get("execution_horizon") != identity.get("execution_horizon"),
            benchmark.get("gripper_mode") != identity.get("gripper_mode"),
            descriptor.get("run_fingerprint") != checkpoint.run_fingerprint,
            descriptor.get("checkpoint_fingerprint") != checkpoint.checkpoint_fingerprint,
            descriptor.get("dataset_fingerprint") != evidence.dataset_fingerprint,
            descriptor.get("task_id") != expected_task_id,
            len(episodes) != len(expected_episodes),
            aggregate.get("episode_count") != len(expected_episodes),
        )
    ):
        raise RuntimeError("runtime benchmark report disagrees with its identity")
    for position, (episode, expected) in enumerate(zip(episodes, expected_episodes, strict=True)):
        if not isinstance(episode, Mapping) or not isinstance(episode.get("rollout"), Mapping):
            raise RuntimeError(f"runtime episode {position} is malformed")
        rollout = cast(Mapping[str, object], episode["rollout"])
        if any(
            (
                episode.get("schedule_id") != "m42_dev_v0",
                episode.get("episode_index") != cast(Any, expected).episode_index,
                episode.get("model_label") != model_label,
                rollout.get("scene_seed") != cast(Any, expected).scene_seed,
                rollout.get("scene_id") != cast(Any, expected).scene_id,
                rollout.get("task_id") != cast(Any, expected).task_id,
                rollout.get("run_fingerprint") != checkpoint.run_fingerprint,
                rollout.get("checkpoint_fingerprint") != checkpoint.checkpoint_fingerprint,
            )
        ):
            raise RuntimeError(f"runtime episode {position} differs from the locked schedule")
    result = _runtime_result_from_mapping(runtime_result)
    aggregate_pairs = (
        (result.episode_count, aggregate.get("episode_count")),
        (result.successes, aggregate.get("successes")),
        (result.post_grasp_timeouts, aggregate.get("post_grasp_timeout_count")),
        (result.wrong_object_interactions, aggregate.get("wrong_object_interaction_count")),
        (result.wrong_object_grasp_count, aggregate.get("wrong_object_grasp_count")),
        (
            result.wrong_object_in_target_bin_count,
            aggregate.get("wrong_object_in_target_bin_count"),
        ),
        (result.target_in_wrong_bin_count, aggregate.get("target_in_wrong_bin_count")),
        (result.target_off_table_count, aggregate.get("target_off_table_count")),
        (result.invalid_action_count, aggregate.get("invalid_action_count")),
        (result.policy_query_count, aggregate.get("policy_query_count")),
        (result.release_sign_transitions, aggregate.get("release_sign_transitions")),
        (result.grasp_sign_transitions, aggregate.get("grasp_sign_transitions")),
        (
            result.unnecessary_gripper_sign_transitions,
            aggregate.get("unnecessary_gripper_sign_transitions"),
        ),
    )
    if any(
        (
            result.report_fingerprint != benchmark_fingerprint,
            result.schedule_id != "m42_dev_v0",
            result.schedule_fingerprint != M42_DEV_SCHEDULE_FINGERPRINT,
            result.model_label != model_label,
            result.task_id != expected_task_id,
            result.execution_horizon != identity.get("execution_horizon"),
            result.gripper_mode is not gripper_mode,
            result.config_fingerprint != expected_config,
            not isinstance(aggregate.get("action_metrics"), Mapping),
            isinstance(aggregate.get("action_metrics"), Mapping)
            and canonical_fingerprint(result.action_metrics)
            != canonical_fingerprint(cast(Mapping[str, object], aggregate["action_metrics"])),
            list(result.successful_episode_steps) != aggregate.get("successful_episode_steps"),
            any(left != right for left, right in aggregate_pairs),
        )
    ):
        raise RuntimeError("runtime aggregate result disagrees with the episode report")
    return _AuditedRuntimeArtifact(
        relative_path=relative_path,
        identity_fingerprint=identity_fingerprint,
        artifact_fingerprint=artifact_fingerprint,
        benchmark_fingerprint=benchmark_fingerprint,
        result=result,
        benchmark=cast(Mapping[str, object], benchmark),
    )


def _runtime_json(root: Path, name: str, *, label: str) -> dict[str, object]:
    path = _owned_runtime_path(root, name, label=label, kind="file")
    return _json_object(path, label=label)


def _audit_runtime_evidence_children(runtime_root: Path, artifact_paths: object) -> tuple[str, ...]:
    if (
        not isinstance(artifact_paths, list)
        or len(artifact_paths) != 8
        or not all(isinstance(item, str) for item in artifact_paths)
        or len(set(cast(list[str], artifact_paths))) != 8
    ):
        raise RuntimeError("runtime command must reference exactly eight unique artifacts")
    evidence_root = _owned_runtime_path(
        runtime_root,
        "evidence",
        label="runtime benchmark evidence root",
        kind="directory",
    )
    actual_children: set[str] = set()
    for child in evidence_root.iterdir():
        resolved = _resolved_unlinked(child, label="runtime benchmark evidence child")
        if not resolved.is_dir() or resolved.parent != evidence_root:
            raise RuntimeError("runtime benchmark evidence root contains an unsafe child")
        actual_children.add(f"evidence/{resolved.name}")
    declared = tuple(cast(list[str], artifact_paths))
    if actual_children != set(declared):
        raise RuntimeError("runtime evidence contains extra or missing benchmark artifacts")
    return declared


def _audit_completed_runtime_evidence(
    *,
    output_root: Path,
    report_path: Path,
    evidence: PriorM4Evidence,
    consumer_git_commit: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Strictly audit immutable runtime evidence before permitting cross-commit reuse."""

    output_root = _resolved_unlinked(output_root, label="M4.2 verifier output")
    runtime_root = _resolved_unlinked(
        output_root / "runtime_ablation", label="runtime-ablation evidence root"
    )
    if not runtime_root.is_dir():
        raise RuntimeError("completed runtime command report has no evidence directory")
    expected_runtime_entries = {
        "evidence",
        "horizon_selection.json",
        "gripper_selection.json",
        "post_grasp_analysis.json",
        "experiment_manifest.json",
        "runtime_selection.json",
    }
    actual_runtime_entries: set[str] = set()
    for entry in runtime_root.iterdir():
        resolved_entry = _resolved_unlinked(entry, label="runtime-ablation top-level artifact")
        if resolved_entry.parent != runtime_root:
            raise RuntimeError("runtime-ablation top-level artifact escapes its owned root")
        actual_runtime_entries.add(resolved_entry.name)
    if actual_runtime_entries != expected_runtime_entries:
        raise RuntimeError("runtime-ablation root contains extra, missing, or staging artifacts")
    report_path = _resolved_unlinked(report_path, label="runtime-ablation command report")
    if report_path.parent != output_root or not report_path.is_file():
        raise RuntimeError("runtime-ablation command report is absent or outside its owned root")
    runtime = _json_object(report_path, label="runtime-ablation command report")
    git = runtime.get("git")
    if not isinstance(git, Mapping):
        raise RuntimeError("runtime command report lacks its producer Git identity")
    producer_commit = git.get("commit")
    if not isinstance(producer_commit, str):
        raise RuntimeError("runtime command report producer Git commit is malformed")
    historical_sources = _runtime_sources_at_commit(producer_commit)
    current_sources = _current_runtime_sources()
    source_audit = _audit_runtime_semantic_sources(historical_sources, current_sources)
    producer_implementation = _runtime_implementation_fingerprint(
        producer_commit, historical_sources
    )
    required_true = (
        "horizon_ablation_completed",
        "horizon_selection_locked",
        "gripper_ablation_completed",
        "gripper_selection_locked",
        "post_grasp_analysis_completed",
        "raw_action_metrics_validated",
        "runtime_action_metrics_validated",
        "physical_execution",
        "passed",
    )
    if (
        runtime.get("schema_version") != RUNTIME_COMMAND_SCHEMA
        or runtime.get("execution_mode") != "physical"
        or git.get("baseline_tracked") is not True
        or git.get("dirty") is not False
        or git.get("changed_paths") != []
        or runtime.get("evaluation_git_commit", producer_commit) != producer_commit
        or any(runtime.get(name) is not True for name in required_true)
        or runtime.get("final_schedule_accessed") is not False
        or runtime.get("schedule_id") != "m42_dev_v0"
        or runtime.get("schedule_fingerprint") != M42_DEV_SCHEDULE_FINGERPRINT
        or runtime.get("implementation_fingerprint") != producer_implementation
        or runtime.get("prior_m4_evidence_fingerprint") != evidence.fingerprint
        or runtime.get("m41_runtime_processor_fingerprint")
        != evidence.m41_runtime_processor_fingerprint
        or runtime.get("m3b_dataset_fingerprint") != evidence.dataset_fingerprint
        or runtime.get("m4_checkpoint_fingerprints")
        != [item.checkpoint_fingerprint for item in evidence.checkpoints]
        or runtime.get("mixed_task_onehot_checkpoint_fingerprint")
        != evidence.mixed_task_onehot.checkpoint_fingerprint
        or runtime.get("representative_per_task_checkpoint_fingerprint")
        != evidence.representative_per_task.checkpoint_fingerprint
        or runtime.get("unique_physical_benchmark_count") != 8
        or runtime.get("unique_physical_episode_count") != 336
    ):
        raise RuntimeError("completed runtime command report fails its frozen acceptance contract")
    expected_plan = {
        "schedule_id": "m42_dev_v0",
        "scene_count": 12,
        "task_count": 6,
        "horizon_order": [10, 5, 1],
        "horizon_mixed_episode_count": 216,
        "horizon_representative_episode_count": 36,
        "horizon_physical_episode_count": 252,
        "gripper_modes": ["project", "binary"],
        "gripper_mixed_logical_episode_count": 144,
        "gripper_representative_logical_episode_count": 24,
        "gripper_logical_episode_count": 168,
        "gripper_additional_physical_episode_count": 84,
        "total_logical_episode_count": 420,
        "total_unique_physical_episode_count": 336,
        "project_gripper_evidence_reuse": "content-bound selected-horizon evidence",
        "final_schedule_materialized": False,
    }
    if runtime.get("plan") != expected_plan:
        raise RuntimeError("runtime command plan does not bind 336 unique/420 logical episodes")
    horizon_config = runtime.get("horizon_config")
    gripper_config = runtime.get("gripper_config")
    if not isinstance(horizon_config, Mapping) or not isinstance(gripper_config, Mapping):
        raise RuntimeError("runtime command report lacks both frozen ablation configurations")
    expected_horizon_config = RuntimeAblationConfig(
        schedule_id="m42_dev_v0",
        schedule_fingerprint=M42_DEV_SCHEDULE_FINGERPRINT,
        m3b_dataset_fingerprint=evidence.dataset_fingerprint,
        mixed_task_onehot_checkpoint_fingerprint=(
            evidence.mixed_task_onehot.checkpoint_fingerprint
        ),
        representative_per_task_checkpoint_fingerprint=(
            evidence.representative_per_task.checkpoint_fingerprint
        ),
        representative_task_id=cast(str, evidence.representative_per_task.task_id),
        horizons=(10, 5, 1),
        gripper_modes=(GripperRuntimeMode.PROJECT,),
        maximum_episode_steps=200,
    ).to_dict()
    if horizon_config != expected_horizon_config:
        raise RuntimeError("runtime ablation configurations differ from current frozen evidence")
    try:
        selected_horizon_from_gripper = cast(list[object], gripper_config["horizons"])[0]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("runtime gripper configuration is malformed") from error
    expected_gripper_config = RuntimeAblationConfig(
        schedule_id="m42_dev_v0",
        schedule_fingerprint=M42_DEV_SCHEDULE_FINGERPRINT,
        m3b_dataset_fingerprint=evidence.dataset_fingerprint,
        mixed_task_onehot_checkpoint_fingerprint=(
            evidence.mixed_task_onehot.checkpoint_fingerprint
        ),
        representative_per_task_checkpoint_fingerprint=(
            evidence.representative_per_task.checkpoint_fingerprint
        ),
        representative_task_id=cast(str, evidence.representative_per_task.task_id),
        horizons=(cast(int, selected_horizon_from_gripper),),
        gripper_modes=(GripperRuntimeMode.PROJECT, GripperRuntimeMode.BINARY),
        maximum_episode_steps=200,
    ).to_dict()
    if gripper_config != expected_gripper_config:
        raise RuntimeError("runtime horizon/gripper configuration candidates changed")
    horizon_config_fingerprint = canonical_fingerprint(cast(Mapping[str, object], horizon_config))
    gripper_config_fingerprint = canonical_fingerprint(cast(Mapping[str, object], gripper_config))
    horizon_payload = _runtime_json(
        runtime_root, "horizon_selection.json", label="runtime horizon selection"
    )
    gripper_payload = _runtime_json(
        runtime_root, "gripper_selection.json", label="runtime gripper selection"
    )
    horizon_selection = M42SelectionRecord.from_dict(horizon_payload)
    gripper_selection = M42SelectionRecord.from_dict(gripper_payload)
    try:
        selected_horizon = int(horizon_selection.selected_value)
    except ValueError as error:
        raise RuntimeError("runtime horizon selection is not an integer") from error
    selected_mode = GripperRuntimeMode(gripper_selection.selected_value)
    if (
        gripper_config.get("horizons") != [selected_horizon]
        or runtime.get("horizon_selection") != horizon_payload
        or runtime.get("gripper_selection") != gripper_payload
        or runtime.get("horizon_selection_fingerprint") != horizon_selection.fingerprint
        or runtime.get("gripper_selection_fingerprint") != gripper_selection.fingerprint
    ):
        raise RuntimeError("runtime selection report and immutable selection files disagree")
    artifact_paths = _audit_runtime_evidence_children(
        runtime_root, runtime.get("evidence_artifacts")
    )
    audited = tuple(
        _audit_runtime_artifact(
            runtime_root=runtime_root,
            relative_path=cast(str, relative),
            producer_commit=producer_commit,
            producer_implementation_fingerprint=producer_implementation,
            evidence=evidence,
            horizon_config_fingerprint=horizon_config_fingerprint,
            gripper_config_fingerprint=gripper_config_fingerprint,
        )
        for relative in artifact_paths
    )
    expected_sequence = tuple(
        (label, horizon, "project")
        for horizon in (10, 5, 1)
        for label in ("ACT-Mixed-TaskOneHot", "ACT-PerTask-Representative")
    ) + (
        ("ACT-Mixed-TaskOneHot", selected_horizon, "binary"),
        ("ACT-PerTask-Representative", selected_horizon, "binary"),
    )
    actual_sequence = tuple(
        (item.result.model_label, item.result.execution_horizon, item.result.gripper_mode.value)
        for item in audited
    )
    if actual_sequence != expected_sequence:
        raise RuntimeError("runtime artifacts do not cover the exact eight declared benchmarks")
    unique_episode_count = sum(item.result.episode_count for item in audited)
    selected_project_episode_count = sum(
        item.result.episode_count
        for item in audited
        if item.result.execution_horizon == selected_horizon
        and item.result.gripper_mode is GripperRuntimeMode.PROJECT
    )
    if unique_episode_count != 336 or unique_episode_count + selected_project_episode_count != 420:
        raise RuntimeError("audited runtime artifacts do not total 336 unique/420 logical episodes")
    by_key = {
        (item.result.model_label, item.result.execution_horizon, item.result.gripper_mode): item
        for item in audited
    }
    mixed_horizon = tuple(
        by_key[("ACT-Mixed-TaskOneHot", horizon, GripperRuntimeMode.PROJECT)].result
        for horizon in (10, 5, 1)
    )
    representative_horizon = tuple(
        by_key[("ACT-PerTask-Representative", horizon, GripperRuntimeMode.PROJECT)].result
        for horizon in (10, 5, 1)
    )
    recomputed_horizon = create_horizon_selection(
        mixed_results=mixed_horizon,
        representative_per_task_results=representative_horizon,
        locked_at_utc=horizon_selection.locked_at_utc,
    )
    recomputed_gripper = create_gripper_selection(
        mixed_results=(
            by_key[("ACT-Mixed-TaskOneHot", selected_horizon, GripperRuntimeMode.PROJECT)].result,
            by_key[("ACT-Mixed-TaskOneHot", selected_horizon, GripperRuntimeMode.BINARY)].result,
        ),
        representative_per_task_results=(
            by_key[
                (
                    "ACT-PerTask-Representative",
                    selected_horizon,
                    GripperRuntimeMode.PROJECT,
                )
            ].result,
            by_key[
                (
                    "ACT-PerTask-Representative",
                    selected_horizon,
                    GripperRuntimeMode.BINARY,
                )
            ].result,
        ),
        locked_at_utc=gripper_selection.locked_at_utc,
    )
    if (
        recomputed_horizon.to_dict() != horizon_payload
        or recomputed_gripper.to_dict() != gripper_payload
        or selected_mode is not GripperRuntimeMode(gripper_selection.selected_value)
    ):
        raise RuntimeError("runtime selections cannot be reproduced from their eight artifacts")
    post_grasp = _runtime_json(
        runtime_root, "post_grasp_analysis.json", label="runtime post-grasp analysis"
    )
    summaries = post_grasp.get("benchmarks")
    if (
        post_grasp.get("schema_version") != RUNTIME_POST_GRASP_SCHEMA
        or post_grasp.get("schedule_id") != "m42_dev_v0"
        or post_grasp.get("schedule_fingerprint") != M42_DEV_SCHEDULE_FINGERPRINT
        or post_grasp.get("unique_physical_benchmark_count") != 8
        or post_grasp.get("complete") is not True
        or post_grasp.get("final_schedule_accessed") is not False
        or not isinstance(summaries, list)
        or len(summaries) != 8
    ):
        raise RuntimeError("runtime post-grasp analysis is incomplete or crossed final")
    for summary, item in zip(summaries, audited, strict=True):
        aggregate = item.benchmark.get("aggregate")
        episodes = item.benchmark.get("episodes")
        if (
            not isinstance(summary, Mapping)
            or not isinstance(aggregate, Mapping)
            or not isinstance(episodes, list)
            or summary.get("artifact") != item.relative_path
            or summary.get("benchmark_fingerprint") != item.benchmark_fingerprint
            or summary.get("episode_count") != item.result.episode_count
            or summary.get("phase_counts") != aggregate.get("phase_counts")
            or summary.get("post_grasp_timeout_count") != aggregate.get("post_grasp_timeout_count")
            or summary.get("failed_record_count")
            != sum(
                isinstance(episode, Mapping)
                and episode.get("post_grasp_failure_record") is not None
                for episode in episodes
            )
        ):
            raise RuntimeError("post-grasp analysis does not reference the audited benchmarks")
    experiment_payload = _runtime_json(
        runtime_root, "experiment_manifest.json", label="runtime experiment manifest"
    )
    experiment_manifest = M42ExperimentManifest.from_dict(experiment_payload)
    if any(
        (
            experiment_manifest.implementation_git_commit != producer_commit,
            experiment_manifest.m3b_dataset_fingerprint != evidence.dataset_fingerprint,
            experiment_manifest.prior_m4_verification_fingerprint
            != evidence.verification_fingerprint,
            dict(experiment_manifest.schedule_fingerprints)
            != {
                "m42_dev_v0": M42_DEV_SCHEDULE_FINGERPRINT,
                "m42_final_v0": M42_FINAL_SCHEDULE_FINGERPRINT,
            },
            experiment_manifest.m4_checkpoint_fingerprints
            != tuple(item.checkpoint_fingerprint for item in evidence.checkpoints),
            experiment_manifest.m41_runtime_processor_fingerprint
            != evidence.m41_runtime_processor_fingerprint,
            dict(experiment_manifest.runtime_selection_fingerprints)
            != {
                "execution_horizon": horizon_selection.fingerprint,
                "gripper_runtime": gripper_selection.fingerprint,
            },
            experiment_manifest.task_token_experiment_fingerprint is not None,
            dict(experiment_manifest.artifact_paths)
            != {
                "runtime_selection": "runtime_selection.json",
                "horizon_selection": "horizon_selection.json",
                "gripper_selection": "gripper_selection.json",
                "post_grasp_analysis": "post_grasp_analysis.json",
            },
            experiment_manifest.completed is not True,
        )
    ):
        raise RuntimeError("runtime experiment manifest differs from current M3B/M4 evidence")
    experiment_fingerprint = experiment_manifest.fingerprint
    runtime_selection_payload = _runtime_json(
        runtime_root, "runtime_selection.json", label="selected runtime lock"
    )
    selection_evidence = runtime_selection_payload.get("selection_evidence")
    if not isinstance(selection_evidence, Mapping):
        raise RuntimeError("selected runtime lock lacks selection evidence")
    fair_payload = selection_evidence.get("task_token_fair_comparison_contract")
    if not isinstance(fair_payload, Mapping):
        raise RuntimeError("selected runtime lock lacks the TaskToken fairness contract")
    fair = TaskTokenFairComparisonContract.from_dict(fair_payload)
    expected_fair = build_task_token_fair_comparison_contract(
        evidence.mixed_task_onehot.manifest,
        selected_checkpoint_fingerprint=evidence.mixed_task_onehot.checkpoint_fingerprint,
    )
    if fair.to_dict() != expected_fair.to_dict():
        raise RuntimeError("runtime TaskToken fairness contract differs from current M4 evidence")
    expected_selection_evidence = {
        "horizon_selection": "horizon_selection.json",
        "gripper_selection": "gripper_selection.json",
        "post_grasp_analysis": "post_grasp_analysis.json",
        "experiment_manifest": "experiment_manifest.json",
        "task_token_fair_comparison_contract": fair.to_dict(),
        "project_evidence_reused_from_horizon": True,
    }
    if (
        runtime_selection_payload.get("schema_version") != RUNTIME_SELECTION_SCHEMA
        or runtime_selection_payload.get("execution_horizon") != selected_horizon
        or runtime_selection_payload.get("gripper_mode") != selected_mode.value
        or runtime_selection_payload.get("horizon_selection_fingerprint")
        != horizon_selection.fingerprint
        or runtime_selection_payload.get("gripper_selection_fingerprint")
        != gripper_selection.fingerprint
        or runtime_selection_payload.get("development_schedule_fingerprint")
        != M42_DEV_SCHEDULE_FINGERPRINT
        or runtime_selection_payload.get("m3b_dataset_fingerprint") != evidence.dataset_fingerprint
        or runtime_selection_payload.get("mixed_task_onehot_checkpoint_fingerprint")
        != evidence.mixed_task_onehot.checkpoint_fingerprint
        or runtime_selection_payload.get("representative_per_task_checkpoint_fingerprint")
        != evidence.representative_per_task.checkpoint_fingerprint
        or runtime_selection_payload.get("implementation_fingerprint") != producer_implementation
        or runtime_selection_payload.get("evaluation_git_commit") != producer_commit
        or runtime_selection_payload.get("experiment_manifest_fingerprint")
        != experiment_fingerprint
        or runtime_selection_payload.get("selection_evidence") != expected_selection_evidence
        or runtime_selection_payload.get("locked") is not True
        or runtime_selection_payload.get("final_schedule_accessed") is not False
    ):
        raise RuntimeError("selected runtime lock is inconsistent with audited evidence")
    runtime_selection_fingerprint = canonical_fingerprint(runtime_selection_payload)
    if (
        runtime.get("runtime_selection_fingerprint") != runtime_selection_fingerprint
        or runtime.get("experiment_manifest") != "experiment_manifest.json"
        or runtime.get("experiment_manifest_fingerprint") != experiment_fingerprint
        or runtime.get("task_token_fair_comparison_fingerprint") != fair.contract_fingerprint
    ):
        raise RuntimeError("runtime report does not bind the selected runtime and manifest")
    report_fingerprint = canonical_fingerprint(runtime)
    lineage_identity: dict[str, object] = {
        "schema_version": RUNTIME_REPAIR_LINEAGE_SCHEMA,
        "producer_git_commit": producer_commit,
        "consumer_git_commit": consumer_git_commit,
        "runtime_semantic_source_files": sorted(historical_sources.sources),
        "runtime_semantic_source_path_count": source_audit.path_count,
        "runtime_semantic_closure_fingerprint": source_audit.closure_fingerprint,
        "historical_runtime_semantic_raw_fingerprint": (source_audit.historical_raw_fingerprint),
        "current_runtime_semantic_raw_fingerprint": source_audit.current_raw_fingerprint,
        "consumer_compatibility_repair_id": source_audit.consumer_repair_id,
        "producer_implementation_fingerprint": producer_implementation,
        "runtime_command_report_fingerprint": report_fingerprint,
        "horizon_selection_fingerprint": horizon_selection.fingerprint,
        "gripper_selection_fingerprint": gripper_selection.fingerprint,
        "runtime_selection_fingerprint": runtime_selection_fingerprint,
        "experiment_manifest_fingerprint": experiment_fingerprint,
        "post_grasp_analysis_fingerprint": canonical_fingerprint(post_grasp),
        "artifact_fingerprints": [
            {
                "relative_path": item.relative_path,
                "identity_fingerprint": item.identity_fingerprint,
                "artifact_fingerprint": item.artifact_fingerprint,
                "benchmark_fingerprint": item.benchmark_fingerprint,
            }
            for item in audited
        ],
        "unique_physical_benchmark_count": 8,
        "unique_physical_episode_count": 336,
        "logical_episode_count": 420,
        "old_evidence_rewritten": False,
        "final_schedule_accessed": False,
    }
    lineage_fingerprint = canonical_fingerprint(lineage_identity)
    lineage = {**lineage_identity, "lineage_fingerprint": lineage_fingerprint}
    return runtime, lineage


def _write_runtime_repair_lineage(output_root: Path, lineage: Mapping[str, object]) -> Path:
    fingerprint = lineage.get("lineage_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or canonical_fingerprint(
            {key: value for key, value in lineage.items() if key != "lineage_fingerprint"}
        )
        != fingerprint
    ):
        raise RuntimeError("runtime repair lineage fingerprint is invalid")
    lineage_root = _resolved_unlinked(
        output_root / "runtime_repair_lineage", label="runtime repair-lineage root"
    )
    runtime_root = _resolved_unlinked(
        output_root / "runtime_ablation", label="runtime-ablation evidence root"
    )
    if _overlaps(lineage_root, runtime_root):
        raise RuntimeError("runtime repair lineage must not overlap immutable runtime evidence")
    if lineage_root.exists() and not lineage_root.is_dir():
        raise RuntimeError("runtime repair-lineage root must be a real directory")
    lineage_root.mkdir(parents=True, exist_ok=True)
    path = lineage_root / f"{fingerprint.removeprefix('sha256:')}.json"
    if path.exists():
        if _json_object(path, label="runtime repair lineage") != dict(lineage):
            raise RuntimeError("existing runtime repair lineage differs from its content address")
    else:
        atomic_write_json(path, dict(lineage), immutable=True)
    return path


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


def _run_target_development(
    report: Report,
    args: argparse.Namespace,
    *,
    prior_evidence: PriorM4Evidence | None = None,
) -> None:
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
    runtime_root = output_root / "runtime_ablation"
    report_present = (
        runtime_report.exists() or runtime_report.is_symlink() or runtime_report.is_junction()
    )
    root_present = runtime_root.exists() or runtime_root.is_symlink() or runtime_root.is_junction()
    if report_present:
        if prior_evidence is None:
            raise RuntimeError("strict runtime evidence reuse requires validated prior M4 evidence")
        git = inspect_git_state(PROJECT_ROOT)
        runtime, lineage = _audit_completed_runtime_evidence(
            output_root=output_root,
            report_path=runtime_report,
            evidence=prior_evidence,
            consumer_git_commit=git.commit,
        )
        lineage_path = _write_runtime_repair_lineage(output_root, lineage)
        report.check(
            "strict immutable runtime evidence reuse",
            True,
            (
                "8 artifacts / 336 unique / 420 logical episodes audited; "
                f"repair lineage={lineage_path.name}"
            ),
        )
    else:
        if root_present and (
            runtime_root.is_symlink()
            or runtime_root.is_junction()
            or not runtime_root.is_dir()
            or any(runtime_root.iterdir())
        ):
            raise RuntimeError(
                "runtime evidence exists without its completed command report; refusing rerun"
            )
        runtime = _run_json(
            _runtime_command(
                args,
                output_root=runtime_root,
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
            evidence = _target_preflight(report, args)
            if mode == "target_development":
                _run_target_development(report, args, prior_evidence=evidence)
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
