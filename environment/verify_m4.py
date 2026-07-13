"""Verify M4 ACT contracts, target smoke, or the complete eight-run experiment."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import subprocess
import sys
import tempfile
import traceback
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from lerobot.policies import make_policy, make_policy_config, make_pre_post_processors
from lerobot.policies.act import ACTConfig, ACTPolicy, make_act_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline

from langmani.datasets.lerobot_types import ACTION_FEATURE_KEY, IMAGE_FEATURE_KEY, STATE_FEATURE_KEY
from langmani.policies.act_action_bounds import (
    ActionBoundConfig,
    ActionBoundMode,
    BoundedActionEnvPostprocessorV0,
)
from langmani.policies.act_checkpoint import (
    CHECKPOINT_COMPLETION_MARKER,
    CHECKPOINT_MANIFEST,
    load_act_checkpoint,
    save_act_checkpoint,
)
from langmani.policies.act_conditioning import CANONICAL_TASK_IDS, canonical_task_onehot
from langmani.policies.act_data import (
    compute_train_only_statistics,
    load_completed_m3b_dataset,
)
from langmani.policies.act_evaluation import (
    authorize_test_evaluation,
    create_checkpoint_selection,
)
from langmani.policies.act_runtime import (
    atomic_write_json,
    inspect_git_state,
    runtime_versions,
    validate_git_for_run,
)
from langmani.policies.act_training import (
    build_fixture_act_config,
    build_policy_and_processors,
    prepare_raw_batch,
)
from langmani.policies.act_types import (
    ActExperimentManifest,
    ActModelConfig,
    ActOptimizationConfig,
    ActRunIdentity,
    ActVariant,
    ExperimentMode,
    TrainingState,
    ValidationResult,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TINY_OVERFIT_MAX_FINAL_LOSS_RATIO = 0.5
TASK_SENSITIVITY_MIN_CHUNK_DISTANCE = 1e-6
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
REPORT_PATH = OUTPUT_ROOT / "diagnostics" / "m4" / "verification.json"
M3B_REPORT = OUTPUT_ROOT / "diagnostics" / "m3b" / "verification.json"
DEFAULT_DATASET_ROOT = OUTPUT_ROOT / "datasets" / "m3b" / "langmani-pick-place-lerobot-v1"
DEFAULT_MODEL_ROOT = OUTPUT_ROOT / "models" / "act"
VerificationMode = Literal["structural", "target_smoke", "target_full"]


def _is_link_like(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def _validate_owned_path(owner: Path, path: Path) -> None:
    try:
        path.absolute().relative_to(owner.absolute())
    except ValueError as error:
        raise RuntimeError("M4 evidence path is outside its declared owner") from error
    if (
        _is_link_like(path)
        or _is_link_like(path.parent)
        or not path.resolve().is_relative_to(owner.resolve())
        or not path.parent.resolve().is_relative_to(owner.resolve())
    ):
        raise RuntimeError("M4 evidence path escapes its declared owner")


@dataclass(slots=True)
class Report:
    checks: list[dict[str, object]] = field(default_factory=list)
    implementation_validated: bool = False
    git_baseline_validated: bool = False
    source_dataset_validated: bool = False
    fixture_training_validated: bool = False
    cuda_training_validated: bool = False
    tiny_overfit_validated: bool = False
    checkpoint_reload_validated: bool = False
    checkpoint_model_reload_validated: bool = False
    policy_processor_reload_validated: bool = False
    action_bound_processor_reload_validated: bool = False
    raw_action_bounds_validated: bool = False
    projected_action_bounds_validated: bool = False
    closed_loop_inference_validated: bool = False
    strict_unprojected_rollout_validated: bool = False
    tiny_overfit_task_success_validated: bool = False
    train_stats_leakage_validated: bool = False
    validation_selection_validated: bool = False
    test_lock_validated: bool = False
    per_task_experiment_completed: bool = False
    mixed_unconditioned_experiment_completed: bool = False
    mixed_task_onehot_experiment_completed: bool = False
    fresh_seed_benchmark_completed: bool = False
    full_dry_run_validated: bool = False
    planned_full_run_count: int = 0
    full_experiment_validated: bool = False
    baseline_quality_validated: bool = False
    physical_target_validated: bool = False

    def check(self, name: str, condition: bool, detail: str) -> None:
        status = "pass" if condition else "fail"
        print(f"[{status.upper()}] {name}: {detail}")
        self.checks.append({"name": name, "status": status, "detail": detail})

    @property
    def failed(self) -> bool:
        return any(item["status"] == "fail" for item in self.checks)

    def accepted(self, mode: VerificationMode, *, dry_run: bool = False) -> bool:
        if self.failed or not all(
            (
                self.implementation_validated,
                self.git_baseline_validated,
                self.fixture_training_validated,
                self.checkpoint_reload_validated,
                self.train_stats_leakage_validated,
                self.validation_selection_validated,
                self.test_lock_validated,
            )
        ):
            return False
        if mode == "structural":
            return True
        if mode == "target_smoke":
            return all(
                (
                    self.source_dataset_validated,
                    self.cuda_training_validated,
                    self.tiny_overfit_validated,
                    self.checkpoint_model_reload_validated,
                    self.policy_processor_reload_validated,
                    self.action_bound_processor_reload_validated,
                    self.projected_action_bounds_validated,
                    self.closed_loop_inference_validated,
                    self.tiny_overfit_task_success_validated,
                    self.physical_target_validated,
                )
            )
        if dry_run:
            return all(
                (
                    self.source_dataset_validated,
                    self.full_dry_run_validated,
                    self.planned_full_run_count == 8,
                    not self.physical_target_validated,
                )
            )
        return all(
            (
                self.source_dataset_validated,
                self.cuda_training_validated,
                self.closed_loop_inference_validated,
                self.per_task_experiment_completed,
                self.mixed_unconditioned_experiment_completed,
                self.mixed_task_onehot_experiment_completed,
                self.fresh_seed_benchmark_completed,
                self.full_experiment_validated,
                self.physical_target_validated,
            )
        )

    def write(self, *, mode: VerificationMode, dataset_root: Path, dry_run: bool = False) -> None:
        payload = {
            "schema_version": "langmani-m4.1-verification-v3",
            "verification_mode": mode,
            "dry_run": dry_run,
            "dataset_root": str(dataset_root.resolve()),
            **{key: value for key, value in asdict(self).items() if key != "checks"},
            "checks": self.checks,
            "passed": self.accepted(mode, dry_run=dry_run),
        }
        atomic_write_json(REPORT_PATH, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--target-smoke", action="store_true")
    modes.add_argument("--target-full", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the exact eight-run full plan without training or rollout",
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--model-root",
        "--output-root",
        dest="model_root",
        type=Path,
        default=DEFAULT_MODEL_ROOT,
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument(
        "--action-bound-mode",
        choices=tuple(mode.value for mode in ActionBoundMode),
        help="required explicit environment-action behavior for target smoke and full",
    )
    parser.add_argument("--per-task-checkpoint", type=Path)
    parser.add_argument("--onehot-checkpoint", type=Path)
    args = parser.parse_args()
    if args.dry_run and not args.target_full:
        parser.error("--dry-run requires --target-full")
    return args


def _mode(args: argparse.Namespace) -> VerificationMode:
    if args.target_smoke:
        return "target_smoke"
    if args.target_full:
        return "target_full"
    return "structural"


def _stats(state_dimension: int = 9) -> dict[str, dict[str, torch.Tensor]]:
    return {
        IMAGE_FEATURE_KEY: {
            "mean": torch.zeros(3, 1, 1),
            "std": torch.ones(3, 1, 1),
            "min": torch.zeros(3, 1, 1),
            "max": torch.ones(3, 1, 1),
        },
        STATE_FEATURE_KEY: {
            "mean": torch.zeros(state_dimension),
            "std": torch.ones(state_dimension),
            "min": -torch.ones(state_dimension),
            "max": torch.ones(state_dimension),
        },
        ACTION_FEATURE_KEY: {
            "mean": torch.zeros(8),
            "std": torch.ones(8),
            "min": -torch.ones(8),
            "max": torch.ones(8),
        },
    }


def _fixture_batch() -> dict[str, torch.Tensor]:
    return {
        IMAGE_FEATURE_KEY: torch.zeros((1, 3, 256, 256), dtype=torch.uint8),
        STATE_FEATURE_KEY: torch.zeros((1, 9), dtype=torch.float32),
        ACTION_FEATURE_KEY: torch.zeros((1, 4, 8), dtype=torch.float32),
        "action_is_pad": torch.tensor([[False, False, False, True]]),
    }


def _fixture_training_and_checkpoint(git_commit: str) -> tuple[bool, bool]:
    torch.manual_seed(0)
    config = build_fixture_act_config()
    policy, preprocessor, postprocessor = build_policy_and_processors(config, _stats())
    processed = preprocessor(prepare_raw_batch(_fixture_batch()))
    loss, loss_dict = policy.forward(processed)
    if not torch.isfinite(loss) or not {"l1_loss", "kld_loss"} <= loss_dict.keys():
        return False, False
    loss.backward()
    if not all(
        torch.isfinite(parameter.grad).all()
        for parameter in policy.parameters()
        if parameter.grad is not None
    ):
        return False, False
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-5, weight_decay=1e-4)
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
    optimizer.step()
    identity = ActRunIdentity(
        m3b_export_fingerprint="sha256:" + "1" * 64,
        m3b_split_manifest_digest="sha256:" + "2" * 64,
        ordered_train_episode_indices=(0,),
        ordered_validation_episode_indices=(1,),
        variant=ActVariant.MIXED_UNCONDITIONED,
        task_id=None,
        task_onehot_mapping_version="CanonicalTaskOneHotV0",
        model_config={
            "fixture": True,
            "chunk_size": config.chunk_size,
            "n_action_steps": config.n_action_steps,
            "dim_model": config.dim_model,
            "n_heads": config.n_heads,
            "pretrained_backbone_weights": config.pretrained_backbone_weights,
        },
        data_contract={"fixture": True},
        train_statistics_fingerprint="sha256:" + "3" * 64,
        optimization_config={"optimizer": "adamw", "fixture": True},
        training_seed=0,
        experiment_mode=ExperimentMode.DRY_RUN,
        device="cpu",
        dtype="float32",
        lerobot_version="0.6.0",
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        git_commit=git_commit,
        git_dirty=False,
        dirty_development_override=False,
    )
    with tempfile.TemporaryDirectory(prefix="langmani-m4-fixture-") as temporary:
        run_root = Path(temporary) / "run"
        run_root.mkdir()
        record = save_act_checkpoint(
            run_root=run_root,
            identity=identity,
            training_state=TrainingState(global_step=1, examples_processed=1),
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            optimizer=optimizer,
        )
        loaded = load_act_checkpoint(
            run_root=run_root,
            checkpoint_relative_path=record.relative_path,
            expected_identity=identity,
        )
        reloaded = isinstance(loaded.policy, ACTPolicy) and loaded.record == record
    return True, reloaded


def _statistics_probe() -> bool:
    rows: list[dict[str, object]] = []
    for episode in (0, 1):
        for frame in (0, 1):
            rows.append(
                {
                    "episode_index": episode,
                    "frame_index": frame,
                    IMAGE_FEATURE_KEY: torch.zeros((3, 256, 256), dtype=torch.float32),
                    STATE_FEATURE_KEY: torch.full((9,), float(episode), dtype=torch.float32),
                    ACTION_FEATURE_KEY: torch.full((8,), float(frame), dtype=torch.float32),
                }
            )
    mixed = compute_train_only_statistics(
        rows,
        train_episode_indices=(0, 1),
        validation_episode_indices=(2,),
        test_episode_indices=(3,),
        task_id_by_episode={0: CANONICAL_TASK_IDS[0], 1: CANONICAL_TASK_IDS[1]},
        variant=ActVariant.MIXED_TASK_ONEHOT,
    )
    processor = mixed.to_processor_stats()[STATE_FEATURE_KEY]
    return (
        mixed.leakage_audit.passed
        and len(mixed.state.components) == 15
        and torch.equal(processor["mean"][-6:], torch.zeros(6))
        and torch.equal(processor["std"][-6:], torch.ones(6))
    )


def _selection_probe() -> tuple[bool, bool]:
    candidates = (
        ValidationResult(
            checkpoint_fingerprint="sha256:" + "4" * 64,
            checkpoint_step=10,
            schedule_digest="sha256:" + "6" * 64,
            success_rate=0.5,
            wrong_object_interaction_rate=0.0,
            target_off_table_rate=0.0,
            offline_validation_action_loss=0.1,
        ),
        ValidationResult(
            checkpoint_fingerprint="sha256:" + "5" * 64,
            checkpoint_step=20,
            schedule_digest="sha256:" + "6" * 64,
            success_rate=0.75,
            wrong_object_interaction_rate=0.0,
            target_off_table_rate=0.0,
            offline_validation_action_loss=0.2,
        ),
    )
    run = "sha256:" + "7" * 64
    selection = create_checkpoint_selection(run_fingerprint=run, candidates=candidates)
    good = authorize_test_evaluation(
        run_fingerprint=run,
        checkpoint_fingerprint=selection.selected_checkpoint_fingerprint,
        actual_schedule_digest="sha256:" + "8" * 64,
        expected_schedule_digest="sha256:" + "8" * 64,
        mode=ExperimentMode.FULL,
        selection=selection,
    )
    bad = authorize_test_evaluation(
        run_fingerprint=run,
        checkpoint_fingerprint=candidates[0].checkpoint_fingerprint,
        actual_schedule_digest="sha256:" + "8" * 64,
        expected_schedule_digest="sha256:" + "8" * 64,
        mode=ExperimentMode.FULL,
        selection=selection,
    )
    return selection.selected_checkpoint_fingerprint == candidates[1].checkpoint_fingerprint, (
        good.authorized and good.final_eligible and not bad.authorized
    )


def _structural_checks(report: Report) -> None:
    git = inspect_git_state(PROJECT_ROOT)
    report.git_baseline_validated = git.baseline_tracked
    report.check(
        "tracked Git baseline",
        report.git_baseline_validated,
        f"commit={git.commit}; dirty={git.dirty}",
    )
    api_ok = (
        version("lerobot") == "0.6.0"
        and inspect.isclass(ACTConfig)
        and "config" in inspect.signature(ACTPolicy).parameters
        and callable(getattr(ACTPolicy, "forward", None))
        and callable(getattr(ACTPolicy, "select_action", None))
        and callable(getattr(ACTPolicy, "reset", None))
        and callable(getattr(ACTPolicy, "save_pretrained", None))
        and callable(getattr(ACTPolicy, "from_pretrained", None))
        and callable(make_pre_post_processors)
        and callable(make_act_pre_post_processors)
        and callable(make_policy)
        and callable(make_policy_config)
        and all(
            callable(getattr(PolicyProcessorPipeline, name, None))
            for name in ("__call__", "reset", "save_pretrained", "from_pretrained")
        )
        and tuple(item.value for item in ActVariant)
        == ("per_task", "mixed_unconditioned", "mixed_task_onehot")
        and ActModelConfig.for_variant(ActVariant.MIXED_TASK_ONEHOT).state_dimension == 15
        and tuple(item.value for item in ActionBoundMode) == ("reject", "project")
        and inspect.isclass(BoundedActionEnvPostprocessorV0)
        and ActionBoundConfig().mode is ActionBoundMode.REJECT
    )
    report.check(
        "installed LeRobot 0.6.0 ACT public interfaces",
        api_ok,
        "ACTConfig/ACTPolicy/factory/processors and 9D/15D variants are available",
    )
    fixture_training, checkpoint_reload = _fixture_training_and_checkpoint(git.commit)
    report.fixture_training_validated = fixture_training
    report.check(
        "real CPU ACT fixture forward/backward/optimizer",
        fixture_training,
        "fixture evidence only; not CUDA, M3B, or model-quality validation",
    )
    report.checkpoint_reload_validated = checkpoint_reload
    report.checkpoint_model_reload_validated = checkpoint_reload
    report.policy_processor_reload_validated = checkpoint_reload
    report.action_bound_processor_reload_validated = api_ok
    report.check(
        "atomic checkpoint and strict local reload",
        checkpoint_reload,
        "policy and processors reloaded without Hub access",
    )
    report.train_stats_leakage_validated = _statistics_probe()
    report.check(
        "train-only normalization and identity one-hot suffix",
        report.train_stats_leakage_validated,
        "synthetic train rows exclude declared validation/test episodes",
    )
    selection_ok, lock_ok = _selection_probe()
    report.validation_selection_validated = selection_ok
    report.test_lock_validated = lock_ok
    report.check(
        "validation-only checkpoint ranking",
        selection_ok,
        "higher validation success selected deterministically",
    )
    report.check(
        "locked test authorization",
        lock_ok,
        "unselected checkpoint is rejected in full mode",
    )
    onehots_ok = all(
        np.array_equal(canonical_task_onehot(task_id), np.eye(6, dtype=np.float32)[index])
        for index, task_id in enumerate(CANONICAL_TASK_IDS)
    )
    report.check(
        "CanonicalTaskOneHotV0 ordering",
        onehots_ok,
        "six stable TaskSpec IDs map to one immutable object-major/bin-minor basis",
    )
    fixture_suite = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/unit/test_act_contract.py",
            "tests/unit/test_act_data.py",
            "tests/unit/test_act_conditioning.py",
            "tests/unit/test_act_checkpoint.py",
            "tests/unit/test_act_evaluation.py",
            "tests/unit/test_act_action_bounds.py",
            "tests/unit/test_m4_commands.py",
            "tests/integration/test_act_training.py",
            "tests/integration/test_act_rollout.py",
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )
    fixture_suite_ok = fixture_suite.returncode == 0
    report.check(
        "M4 contract and real ACT fixture suite",
        fixture_suite_ok,
        f"exit code {fixture_suite.returncode}",
    )
    report.implementation_validated = all(
        (
            api_ok,
            onehots_ok,
            report.fixture_training_validated,
            report.checkpoint_reload_validated,
            report.train_stats_leakage_validated,
            report.validation_selection_validated,
            report.test_lock_validated,
            fixture_suite_ok,
        )
    )


def _run(arguments: list[str]) -> tuple[bool, dict[str, object] | None]:
    completed = subprocess.run(arguments, cwd=PROJECT_ROOT, check=False)
    output_flag = next((flag for flag in ("--report", "--output") if flag in arguments), None)
    report_flag = arguments.index(output_flag) + 1 if output_flag is not None else None
    if report_flag is None:
        return completed.returncode == 0, None
    path = Path(arguments[report_flag])
    if not path.is_file():
        return False, None
    value = json.loads(path.read_text(encoding="utf-8"))
    return completed.returncode == 0 and value.get("passed") is True, value


def _run_prior_target_gate(
    report: Report,
    mode: VerificationMode,
    args: argparse.Namespace,
    *,
    dry_run: bool = False,
) -> Path:
    if mode == "target_smoke":
        if not M3B_REPORT.is_file():
            report.check("completed M0-M3B target gate", False, "M3B report is missing")
            raise RuntimeError(
                "M4.1 does not recreate M3A/M3B data; run the completed target-smoke chain first"
            )
        payload = json.loads(M3B_REPORT.read_text(encoding="utf-8"))
        dataset_root = Path(str(payload.get("dataset_root", ""))).resolve()
        required_flags = (
            "implementation_validated",
            "source_archive_validated",
            "smoke_export_validated",
            "lerobot_load_validated",
            "video_decode_validated",
            "source_alignment_validated",
            "split_integrity_validated",
            "privileged_leakage_validated",
            "physical_target_validated",
        )
        completed = load_completed_m3b_dataset(
            dataset_root,
            require_full=False,
            validate_storage=True,
        )
        passed = (
            payload.get("verification_mode") == "target_smoke"
            and payload.get("passed") is True
            and all(payload.get(name) is True for name in required_flags)
            and completed.summary.total_episodes == 6
        )
        report.check(
            "completed M0-M3B target gate",
            passed,
            f"validated existing six-episode dataset at {dataset_root}",
        )
        if not passed:
            raise RuntimeError("completed M3B smoke evidence is invalid")
        return dataset_root
    if dry_run:
        if not M3B_REPORT.is_file():
            report.check("completed M3B full target gate", False, "M3B report is missing")
            raise RuntimeError("full dry-run requires a completed M3B target-full report")
        payload = json.loads(M3B_REPORT.read_text(encoding="utf-8"))
        dataset_root = Path(str(payload.get("dataset_root", ""))).resolve()
        requested_root = args.dataset_root.resolve()
        required_flags = (
            "implementation_validated",
            "source_archive_validated",
            "full_export_validated",
            "lerobot_load_validated",
            "parquet_validated",
            "video_decode_validated",
            "source_alignment_validated",
            "split_integrity_validated",
            "privileged_leakage_validated",
            "physical_target_validated",
        )
        passed = (
            payload.get("verification_mode") == "target_full"
            and payload.get("passed") is True
            and all(payload.get(name) is True for name in required_flags)
            and dataset_root == requested_root
        )
        report.check(
            "completed M3B full target gate",
            passed,
            f"validated existing report and requested dataset at {dataset_root}",
        )
        if not passed:
            raise RuntimeError("completed M3B full evidence is missing, stale, or for another root")
        return dataset_root
    M3B_REPORT.unlink(missing_ok=True)
    command = [sys.executable, "environment/verify_m3b.py"]
    if mode == "target_smoke":
        command.append("--target-smoke")
    else:
        command.extend(["--target-full", "--dataset-root", str(args.dataset_root)])
        if args.source_root is not None:
            command.extend(["--source-root", str(args.source_root)])
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0 or not M3B_REPORT.is_file():
        report.check("ordered M0-M3B target gate", False, f"exit code {completed.returncode}")
        raise RuntimeError("M3B target prerequisite failed")
    payload = json.loads(M3B_REPORT.read_text(encoding="utf-8"))
    expected_flag = "smoke_export_validated" if mode == "target_smoke" else "full_export_validated"
    passed = (
        payload.get("passed") is True
        and payload.get("physical_target_validated") is True
        and payload.get(expected_flag) is True
    )
    report.check("ordered M0-M3B target gate", passed, f"mode={mode}")
    if not passed:
        raise RuntimeError("M3B report does not provide physical target evidence")
    return Path(str(payload["dataset_root"]))


def _training_command(
    *,
    dataset_root: Path,
    model_root: Path,
    variant: ActVariant,
    task_id: str | None,
    mode: str,
    steps: int,
    report_path: Path,
    planned_mode: ExperimentMode | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/train_act.py",
        "--dataset-root",
        str(dataset_root),
        "--variant",
        variant.value,
        f"--{mode}",
        "--device",
        "cuda",
        "--steps",
        str(steps),
        "--output-root",
        str(model_root),
        "--report",
        str(report_path),
    ]
    if planned_mode is not None:
        command.extend(["--planned-mode", planned_mode.value])
    if task_id is not None:
        command.extend(["--task-id", task_id])
    return command


def _evaluation_command(
    *,
    checkpoint: Path,
    dataset_root: Path,
    split: str,
    action_bound_mode: str,
    report_path: Path,
) -> list[str]:
    """Build one rollout command with an explicit environment-action boundary."""

    return [
        sys.executable,
        "scripts/evaluate_act.py",
        "--checkpoint",
        str(checkpoint),
        "--dataset-root",
        str(dataset_root),
        "--split",
        split,
        "--action-bound-mode",
        action_bound_mode,
        "--report",
        str(report_path),
    ]


def _full_training_command_or_reuse(
    *,
    dataset_root: Path,
    dataset_fingerprint: str,
    model_root: Path,
    variant: ActVariant,
    task_id: str | None,
    report_path: Path,
) -> tuple[list[str] | None, dict[str, object] | None]:
    git = inspect_git_state(PROJECT_ROOT)
    versions = runtime_versions()
    matches: list[tuple[Path, ActExperimentManifest]] = []
    for path in model_root.glob("*/run_manifest.json"):
        candidate_root = path.parent
        if (
            _is_link_like(candidate_root)
            or not candidate_root.is_dir()
            or candidate_root.resolve().parent != model_root.resolve()
        ):
            raise RuntimeError("full-run candidate directory escapes the model root")
        _validate_owned_path(model_root, path)
        try:
            manifest = ActExperimentManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (AttributeError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if (
            manifest.config.mode is ExperimentMode.FULL
            and manifest.config.device == "cuda"
            and manifest.config.allow_dirty_development is False
            and manifest.config.model == ActModelConfig.for_variant(variant)
            and manifest.config.optimization == ActOptimizationConfig()
            and manifest.identity.m3b_export_fingerprint == dataset_fingerprint
            and manifest.identity.git_commit == git.commit
            and manifest.identity.git_dirty is False
            and manifest.identity.lerobot_version == versions["lerobot"]
            and manifest.identity.torch_version == versions["torch"]
            and manifest.identity.cuda_version == versions["cuda"]
            and manifest.identity.variant is variant
            and manifest.identity.task_id == task_id
            and manifest.identity.training_seed == 0
        ):
            matches.append((path.parent, manifest))
    if len(matches) > 1:
        raise RuntimeError("multiple current full runs match one declared M4 baseline")
    command = _training_command(
        dataset_root=dataset_root,
        model_root=model_root,
        variant=variant,
        task_id=task_id,
        mode="full",
        steps=100_000,
        report_path=report_path,
    )
    if not matches:
        return command, None
    run_root, manifest = matches[0]
    summary_path = run_root / "reports" / "training_summary.json"
    if summary_path.is_file():
        _validate_owned_path(run_root, summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            not isinstance(summary, dict)
            or summary.get("passed") is not True
            or summary.get("run_fingerprint") != manifest.identity.run_fingerprint
            or not manifest.checkpoints
        ):
            raise RuntimeError("existing full-run training summary is invalid")
        return None, {
            "passed": True,
            "training_started": False,
            "reused_completed_training": True,
            "expected_output_directory": str(run_root),
            "run_fingerprint": manifest.identity.run_fingerprint,
            "last_checkpoint": manifest.checkpoints[-1].to_dict(),
        }
    declared = {item.relative_path for item in manifest.checkpoints}
    promoted: set[str] = set()
    for marker in (run_root / "checkpoints").glob(f"*/{CHECKPOINT_COMPLETION_MARKER}"):
        _validate_owned_path(run_root, marker.parent)
        _validate_owned_path(run_root, marker)
        promoted.add(marker.parent.relative_to(run_root).as_posix())
    orphans = promoted - declared
    if len(orphans) > 1:
        raise RuntimeError("incomplete full run contains multiple unreferenced checkpoints")
    if not manifest.checkpoints and not orphans:
        if _is_link_like(run_root) or run_root.resolve().parent != model_root.resolve():
            raise RuntimeError("refusing to archive an unsafe pre-checkpoint run directory")
        abandoned_parent = model_root / "_abandoned_precheckpoint"
        if abandoned_parent.exists() and (
            _is_link_like(abandoned_parent)
            or abandoned_parent.resolve().parent != model_root.resolve()
        ):
            raise RuntimeError("refusing an unsafe pre-checkpoint diagnostic directory")
        abandoned_parent.mkdir(parents=True, exist_ok=True)
        if abandoned_parent.resolve().parent != model_root.resolve():
            raise RuntimeError("pre-checkpoint diagnostic directory escapes the model root")
        suffix = 0
        abandoned = abandoned_parent / f"{run_root.name}-{suffix:04d}"
        while abandoned.exists():
            suffix += 1
            abandoned = abandoned_parent / f"{run_root.name}-{suffix:04d}"
        os.replace(run_root, abandoned)
        atomic_write_json(
            abandoned / "abandoned.json",
            {
                "schema_version": "langmani-m4-abandoned-training-run-v1",
                "reason": "interrupted_before_first_promoted_checkpoint",
                "run_fingerprint": manifest.identity.run_fingerprint,
                "git_commit": manifest.identity.git_commit,
            },
            immutable=True,
        )
        return command, None
    resume_checkpoint = next(iter(orphans)) if orphans else manifest.checkpoints[-1].relative_path
    command.extend(["--resume-checkpoint", resume_checkpoint])
    return command, None


def _evaluate_last_checkpoint(
    *,
    train_report: dict[str, object],
    dataset_root: Path,
    split: str,
    report_path: Path,
    action_bound_mode: ActionBoundMode,
    sensitivity: bool = False,
    strict_bound_probe: bool = False,
) -> tuple[bool, dict[str, object] | None]:
    checkpoint = train_report.get("last_checkpoint")
    if not isinstance(checkpoint, dict):
        return False, None
    run_root = Path(str(train_report["expected_output_directory"]))
    command = [
        sys.executable,
        "scripts/evaluate_act.py",
        "--checkpoint",
        str(run_root / str(checkpoint["relative_path"])),
        "--dataset-root",
        str(dataset_root),
        "--split",
        split,
        "--action-bound-mode",
        action_bound_mode.value,
        "--report",
        str(report_path),
    ]
    if sensitivity:
        command.append("--counterfactual-sensitivity")
    if strict_bound_probe:
        command.append("--strict-bound-probe")
    return _run(command)


def _existing_smoke_training(
    *,
    report_path: Path,
    checkpoint_override: Path | None,
    model_root: Path,
    variant: ActVariant,
    task_id: str | None,
) -> dict[str, object]:
    if checkpoint_override is None:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        if payload.get("passed") is not True:
            raise RuntimeError(f"existing training report is not successful: {report_path}")
        checkpoint_value = payload.get("last_checkpoint")
        if not isinstance(checkpoint_value, dict):
            raise RuntimeError("existing training report lacks its last checkpoint")
        run_root = Path(str(payload.get("expected_output_directory", ""))).resolve()
        checkpoint = (run_root / str(checkpoint_value.get("relative_path", ""))).resolve()
    else:
        checkpoint = checkpoint_override.resolve()
        if checkpoint.parent.name != "checkpoints":
            raise RuntimeError("smoke checkpoint must be a direct checkpoints/ child")
        run_root = checkpoint.parent.parent
        checkpoint_manifest = json.loads(
            (checkpoint / CHECKPOINT_MANIFEST).read_text(encoding="utf-8")
        )
        record = checkpoint_manifest.get("record")
        if not isinstance(record, dict):
            raise RuntimeError("smoke checkpoint lacks a record")
        payload = {
            "passed": True,
            "expected_output_directory": str(run_root),
            "last_checkpoint": record,
        }
    _validate_owned_path(model_root, run_root)
    _validate_owned_path(run_root, checkpoint)
    manifest = ActExperimentManifest.from_dict(
        json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    )
    summary_path = run_root / "reports" / "training_summary.json"
    _validate_owned_path(run_root, summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    relative = checkpoint.relative_to(run_root).as_posix()
    matching = tuple(item for item in manifest.checkpoints if item.relative_path == relative)
    if (
        manifest.identity.variant is not variant
        or manifest.identity.task_id != task_id
        or manifest.config.mode is not ExperimentMode.TINY_OVERFIT
        or manifest.config.device != "cuda"
        or len(matching) != 1
        or not (checkpoint / CHECKPOINT_COMPLETION_MARKER).is_file()
        or summary.get("passed") is not True
        or summary.get("run_fingerprint") != manifest.identity.run_fingerprint
    ):
        raise RuntimeError("existing smoke checkpoint does not match the declared M4 run")
    return payload


def _projection_metrics(
    *evaluations: dict[str, object] | None,
) -> tuple[int, int, float, list[int], float, float]:
    task_successes = 0
    strict_successes = 0
    total_actions = 0
    projected_actions = 0
    dimension_counts = [0] * 8
    maximum_excess = 0.0
    maximum_correction = 0.0
    for evaluation in evaluations:
        if evaluation is None:
            continue
        task_successes += int(evaluation.get("task_success_count", 0))
        strict_successes += int(evaluation.get("strict_unprojected_success_count", 0))
        summary = evaluation.get("action_projection_summary")
        if not isinstance(summary, dict):
            continue
        total_actions += int(summary.get("total_policy_actions", 0))
        projected_actions += int(summary.get("projected_action_count", 0))
        raw_counts = summary.get("per_action_dimension_projection_counts", [])
        if isinstance(raw_counts, list) and len(raw_counts) == 8:
            dimension_counts = [
                left + int(right) for left, right in zip(dimension_counts, raw_counts, strict=True)
            ]
        maximum_excess = max(maximum_excess, float(summary.get("maximum_bound_excess", 0.0)))
        maximum_correction = max(
            maximum_correction, float(summary.get("maximum_linf_correction", 0.0))
        )
    rate = projected_actions / total_actions if total_actions else 0.0
    return (
        task_successes,
        strict_successes,
        rate,
        dimension_counts,
        maximum_excess,
        maximum_correction,
    )


def _run_target_smoke(
    report: Report,
    dataset_root: Path,
    model_root: Path,
    args: argparse.Namespace,
) -> None:
    report.source_dataset_validated = True
    smoke_dir = REPORT_PATH.parent / "target_smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    per_train = _existing_smoke_training(
        report_path=smoke_dir / "per_task_train.json",
        checkpoint_override=args.per_task_checkpoint,
        model_root=model_root,
        variant=ActVariant.PER_TASK,
        task_id=CANONICAL_TASK_IDS[0],
    )
    onehot_train = _existing_smoke_training(
        report_path=smoke_dir / "onehot_train.json",
        checkpoint_override=args.onehot_checkpoint,
        model_root=model_root,
        variant=ActVariant.MIXED_TASK_ONEHOT,
        task_id=None,
    )
    report.cuda_training_validated = True
    report.check(
        "reuse existing CUDA tiny-overfit checkpoints",
        True,
        "loaded completed 5000-step PerTask and 10000-step TaskOneHot runs; no training invoked",
    )
    rollout_source = (PROJECT_ROOT / "src/langmani/policies/act_rollout.py").read_text(
        encoding="utf-8"
    )
    expert_absent = all(
        token not in rollout_source
        for token in ("PickPlaceExpert", "langmani.experts", "langmani.expert")
    )
    report.check(
        "learned rollout excludes M2 expert APIs",
        expert_absent,
        "ACT rollout source has no expert controller import or call",
    )
    if not expert_absent:
        return

    strict_ok, strict_probe = _evaluate_last_checkpoint(
        train_report=onehot_train,
        dataset_root=dataset_root,
        split="train",
        report_path=smoke_dir / "strict_bound_probe.json",
        action_bound_mode=ActionBoundMode.REJECT,
        strict_bound_probe=True,
    )
    strict_reproduced = (
        strict_ok
        and strict_probe is not None
        and strict_probe.get("raw_action_bounds_validated") is False
        and strict_probe.get("reject_blocked_before_env_step") is True
        and strict_probe.get("projected_action_bounds_validated") is True
        and strict_probe.get("real_projected_env_step_executed") is True
    )
    report.check(
        "strict rejection and one-step projection probe",
        strict_reproduced,
        "known raw violation rejected before env.step; projected action executed once",
    )
    if not strict_reproduced:
        return

    per_eval_ok, per_eval = _evaluate_last_checkpoint(
        train_report=per_train,
        dataset_root=dataset_root,
        split="train",
        report_path=smoke_dir / "per_task_rollout.json",
        action_bound_mode=ActionBoundMode.PROJECT,
    )
    onehot_eval_ok, onehot_eval = _evaluate_last_checkpoint(
        train_report=onehot_train,
        dataset_root=dataset_root,
        split="train",
        report_path=smoke_dir / "onehot_rollout.json",
        action_bound_mode=ActionBoundMode.PROJECT,
        sensitivity=True,
    )
    per_checkpoint = per_train.get("last_checkpoint")
    onehot_checkpoint = onehot_train.get("last_checkpoint")
    checkpoint_fingerprints_unchanged = (
        isinstance(per_checkpoint, dict)
        and isinstance(onehot_checkpoint, dict)
        and per_eval is not None
        and onehot_eval is not None
        and per_eval.get("checkpoint_fingerprint") == per_checkpoint.get("checkpoint_fingerprint")
        and onehot_eval.get("checkpoint_fingerprint")
        == onehot_checkpoint.get("checkpoint_fingerprint")
        and strict_probe is not None
        and strict_probe.get("checkpoint_fingerprint")
        == onehot_checkpoint.get("checkpoint_fingerprint")
    )
    report.check(
        "existing checkpoint fingerprints unchanged",
        checkpoint_fingerprints_unchanged,
        "strict probe and projected rollouts retained the saved checkpoint identities",
    )
    runtime_reports = (strict_probe, per_eval, onehot_eval)
    report.checkpoint_model_reload_validated = checkpoint_fingerprints_unchanged and all(
        item is not None and item.get("checkpoint_model_reload_validated") is True
        for item in runtime_reports
    )
    report.policy_processor_reload_validated = all(
        item is not None and item.get("policy_processor_reload_validated") is True
        for item in runtime_reports
    )
    report.action_bound_processor_reload_validated = all(
        item is not None and item.get("action_bound_processor_reload_validated") is True
        for item in runtime_reports
    )
    report.checkpoint_reload_validated = all(
        (
            report.checkpoint_reload_validated,
            report.checkpoint_model_reload_validated,
            report.policy_processor_reload_validated,
            report.action_bound_processor_reload_validated,
        )
    )
    report.raw_action_bounds_validated = all(
        item is not None and item.get("raw_action_bounds_validated") is True
        for item in (per_eval, onehot_eval)
    )
    report.projected_action_bounds_validated = all(
        item is not None and item.get("projected_action_bounds_validated") is True
        for item in (per_eval, onehot_eval)
    )
    real_closed_loop = all(
        ok
        and item is not None
        and item.get("physical_execution") is True
        and item.get("infrastructure_failure_count") == 0
        for ok, item in ((per_eval_ok, per_eval), (onehot_eval_ok, onehot_eval))
    )
    onehot_six = (
        onehot_eval_ok
        and onehot_eval is not None
        and onehot_eval.get("episode_count") == 6
        and onehot_eval.get("task_success_count") == 6
    )
    report.closed_loop_inference_validated = real_closed_loop
    report.tiny_overfit_task_success_validated = bool(onehot_six)
    report.tiny_overfit_validated = bool(onehot_six)
    task_successes, strict_successes, rate, dimensions, excess, correction = _projection_metrics(
        per_eval, onehot_eval
    )
    report.strict_unprojected_rollout_validated = strict_successes == 7
    report.baseline_quality_validated = bool(onehot_six)
    report.physical_target_validated = all(
        (
            report.checkpoint_reload_validated,
            report.projected_action_bounds_validated,
            report.closed_loop_inference_validated,
            report.tiny_overfit_task_success_validated,
        )
    )
    report.check(
        "task-one-hot projected tiny 6/6 quality gate",
        report.tiny_overfit_task_success_validated,
        (
            f"task_successes={task_successes}/7; strict_successes={strict_successes}/7; "
            f"projected_rate={rate:.6f}; per_dimension={dimensions}; "
            f"max_excess={excess:.8f}; max_correction={correction:.8f}"
        ),
    )
    if not onehot_six:
        substantive = None
        for item in (onehot_eval, per_eval):
            if isinstance(item, dict):
                substantive = item.get("error_message") or item.get("failure_counts")
                if substantive:
                    break
        report.check(
            "first substantive post-projection failure",
            False,
            str(substantive or "task-success gate not reached; inspect preserved rollout evidence"),
        )


def _run_target_full_dry_run(
    report: Report,
    dataset_root: Path,
    model_root: Path,
    *,
    action_bound_mode: str,
) -> None:
    """Validate all semantic full-run inputs without creating model artifacts."""

    completed = load_completed_m3b_dataset(dataset_root, require_full=True, validate_storage=True)
    report.source_dataset_validated = completed.summary.total_episodes == 360
    report.check(
        "completed real M3B full dataset",
        report.source_dataset_validated,
        f"episodes={completed.summary.total_episodes}",
    )
    full_dir = REPORT_PATH.parent / "target_full"
    full_dir.mkdir(parents=True, exist_ok=True)
    runs: list[tuple[ActVariant, str | None]] = [
        *[(ActVariant.PER_TASK, task_id) for task_id in CANONICAL_TASK_IDS],
        (ActVariant.MIXED_UNCONDITIONED, None),
        (ActVariant.MIXED_TASK_ONEHOT, None),
    ]
    expected_optimization = ActOptimizationConfig().to_dict()
    expected_checkpoints = list(range(5_000, 100_001, 5_000))
    git = inspect_git_state(PROJECT_ROOT)
    planned: list[dict[str, object]] = []
    plan_valid = report.source_dataset_validated
    for index, (variant, task_id) in enumerate(runs):
        report_path = full_dir / f"dry_run_{index:02d}.json"
        command = _training_command(
            dataset_root=dataset_root,
            model_root=model_root,
            variant=variant,
            task_id=task_id,
            mode="dry-run",
            planned_mode=ExperimentMode.FULL,
            steps=100_000,
            report_path=report_path,
        )
        ok, train_report = _run(command)
        if not ok or train_report is None:
            report.check("eight ACT full dry-run identities", False, f"failed run index={index}")
            return
        expected_model = ActModelConfig.for_variant(variant).to_dict()
        expected_train_count = 48 if variant is ActVariant.PER_TASK else 288
        expected_validation_count = 6 if variant is ActVariant.PER_TASK else 36
        expected_state_dimension = 15 if variant is ActVariant.MIXED_TASK_ONEHOT else 9
        fingerprint = train_report.get("run_fingerprint")
        output_directory = Path(str(train_report.get("expected_output_directory", ""))).resolve()
        act_config = train_report.get("act_config")
        model_contract_matches = isinstance(act_config, dict) and all(
            act_config.get(key) == value for key, value in expected_model.items()
        )
        run_valid = all(
            (
                train_report.get("passed") is True,
                train_report.get("dry_run") is True,
                train_report.get("training_started") is False,
                train_report.get("fixture_evidence") is False,
                train_report.get("planned_experiment_mode") == ExperimentMode.FULL.value,
                train_report.get("dataset_fingerprint") == completed.export_fingerprint,
                train_report.get("split_digest") == completed.split_manifest_digest,
                train_report.get("variant") == variant.value,
                train_report.get("task_id") == task_id,
                train_report.get("train_episode_count") == expected_train_count,
                train_report.get("validation_episode_count") == expected_validation_count,
                train_report.get("offline_loss_episode_count") == expected_validation_count,
                train_report.get("offline_loss_role") == "held_out_validation",
                train_report.get("optimization") == expected_optimization,
                train_report.get("checkpoint_schedule") == expected_checkpoints,
                train_report.get("evaluation_schedule") == expected_checkpoints,
                train_report.get("input_shapes", {}).get(STATE_FEATURE_KEY)
                == [expected_state_dimension],
                train_report.get("output_shape") == [8],
                model_contract_matches,
                train_report.get("git") == git.to_dict(),
                isinstance(fingerprint, str),
                output_directory.parent == model_root.resolve(),
                isinstance(fingerprint, str)
                and output_directory.name == fingerprint.removeprefix("sha256:"),
            )
        )
        plan_valid &= run_valid
        planned.append(
            {
                "index": index,
                "variant": variant.value,
                "task_id": task_id,
                "run_fingerprint": fingerprint,
                "expected_output_directory": str(output_directory),
                "output_directory_exists": output_directory.exists(),
                "train_episode_count": expected_train_count,
                "validation_episode_count": expected_validation_count,
                "train_statistics_fingerprint": train_report.get("train_statistics_fingerprint"),
                "state_dimension": expected_state_dimension,
                "action_dimension": 8,
                "contract_validated": run_valid,
                "command_report": str(report_path.resolve()),
            }
        )
    fingerprints = [item["run_fingerprint"] for item in planned]
    directories = [item["expected_output_directory"] for item in planned]
    plan_valid &= len(planned) == 8 and len(set(fingerprints)) == 8 and len(set(directories)) == 8
    plan_payload = {
        "schema_version": "langmani-m4-full-dry-run-plan-v1",
        "passed": plan_valid,
        "training_started": False,
        "rollout_started": False,
        "action_bound_mode": action_bound_mode,
        "dataset_root": str(dataset_root.resolve()),
        "dataset_fingerprint": completed.export_fingerprint,
        "split_digest": completed.split_manifest_digest,
        "git": git.to_dict(),
        "optimization": expected_optimization,
        "checkpoint_schedule": expected_checkpoints,
        "evaluation_schedule": expected_checkpoints,
        "runs": planned,
    }
    atomic_write_json(full_dir / "dry_run_plan.json", plan_payload)
    report.planned_full_run_count = len(planned)
    report.full_dry_run_validated = plan_valid
    report.check(
        "eight ACT full dry-run identities",
        plan_valid,
        (
            f"runs={len(planned)}; unique_fingerprints={len(set(fingerprints))}; "
            f"action_bound_mode={action_bound_mode}; training_started=false"
        ),
    )


def _run_target_full(
    report: Report,
    dataset_root: Path,
    model_root: Path,
    *,
    action_bound_mode: str,
) -> None:
    completed = load_completed_m3b_dataset(dataset_root, require_full=True, validate_storage=True)
    report.source_dataset_validated = completed.summary.total_episodes == 360
    report.check(
        "completed real M3B full dataset",
        report.source_dataset_validated,
        f"episodes={completed.summary.total_episodes}",
    )
    full_dir = REPORT_PATH.parent / "target_full"
    full_dir.mkdir(parents=True, exist_ok=True)
    runs: list[tuple[ActVariant, str | None]] = [
        *[(ActVariant.PER_TASK, task_id) for task_id in CANONICAL_TASK_IDS],
        (ActVariant.MIXED_UNCONDITIONED, None),
        (ActVariant.MIXED_TASK_ONEHOT, None),
    ]
    trained: list[dict[str, object]] = []
    for index, (variant, task_id) in enumerate(runs):
        command, reused_report = _full_training_command_or_reuse(
            dataset_root=dataset_root,
            dataset_fingerprint=completed.export_fingerprint,
            model_root=model_root,
            variant=variant,
            task_id=task_id,
            report_path=full_dir / f"train_{index:02d}.json",
        )
        ok, train_report = (True, reused_report) if command is None else _run(command)
        if not ok or train_report is None:
            report.check("eight ACT full training runs", False, f"failed run index={index}")
            return
        trained.append(train_report)
    report.cuda_training_validated = True
    # Full closed-loop selection evaluates every immutable checkpoint.  The
    # orchestration is intentionally explicit and stops on the first failure.
    all_closed_loop = True
    run_completed = [False] * len(trained)
    fresh_completed = [False] * len(trained)
    for index, train_report in enumerate(trained):
        run_ok = True
        run_root = Path(str(train_report["expected_output_directory"]))
        _validate_owned_path(model_root, run_root)
        if (run_root / "complete.json").is_file():
            _validate_owned_path(run_root, run_root / "run_manifest.json")
            _validate_owned_path(run_root, run_root / "complete.json")
            manifest = ActExperimentManifest.from_dict(
                json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
            )
            completion = json.loads((run_root / "complete.json").read_text(encoding="utf-8"))
            if (
                manifest.complete
                and isinstance(completion, dict)
                and completion.get("run_fingerprint") == manifest.identity.run_fingerprint
                and completion.get("selected_checkpoint_fingerprint")
                == manifest.selected_checkpoint_fingerprint
            ):
                selected_fingerprint = manifest.selected_checkpoint_fingerprint
                selected_records = tuple(
                    checkpoint
                    for checkpoint in manifest.checkpoints
                    if checkpoint.checkpoint_fingerprint == selected_fingerprint
                )
                if len(selected_records) != 1:
                    raise RuntimeError(
                        "completed full run does not identify one selected checkpoint"
                    )
                selected_name = selected_fingerprint.removeprefix("sha256:")
                analysis_name = (
                    "per_task_reference.json"
                    if manifest.identity.variant is ActVariant.PER_TASK
                    else "counterfactual_sensitivity.json"
                )
                required_completed_artifacts = (
                    run_root / "validation" / selected_name / "benchmark.json",
                    run_root / "test" / "benchmark.json",
                    run_root / "fresh_seed" / "benchmark.json",
                    run_root / "fresh_seed" / "analysis.json",
                    run_root / "reports" / analysis_name,
                )
                for path in required_completed_artifacts:
                    _validate_owned_path(run_root, path)
                _validate_owned_path(
                    run_root,
                    run_root / selected_records[0].relative_path,
                )
                if not all(path.is_file() for path in required_completed_artifacts):
                    raise RuntimeError("completed full run is missing immutable final evidence")
                for split in ("validation", "test", "fresh_seed"):
                    command = _evaluation_command(
                        checkpoint=run_root / selected_records[0].relative_path,
                        dataset_root=dataset_root,
                        split=split,
                        action_bound_mode=action_bound_mode,
                        report_path=full_dir / f"reuse_{split}_{index:02d}.json",
                    )
                    if split == "validation":
                        command.append("--lock-selection")
                    if split == "fresh_seed":
                        command.append("--counterfactual-sensitivity")
                    ok, evaluation_report = _run(command)
                    executed = (
                        ok
                        and evaluation_report is not None
                        and evaluation_report.get("physical_execution") is True
                        and evaluation_report.get("infrastructure_failure_count") == 0
                        and evaluation_report.get("reused_existing_evaluation") is True
                    )
                    run_ok &= executed
                    if split == "fresh_seed":
                        fresh_completed[index] = executed
                    if not executed:
                        break
                run_completed[index] = run_ok
                all_closed_loop &= run_ok
                if not run_ok:
                    break
                continue
            raise RuntimeError("existing completed full run has inconsistent completion evidence")
        checkpoints = sorted((run_root / "checkpoints").glob("step-*/complete.json"))
        for checkpoint_index, marker in enumerate(checkpoints):
            _validate_owned_path(run_root, marker.parent)
            _validate_owned_path(run_root, marker)
            _validate_owned_path(run_root, marker.parent / CHECKPOINT_MANIFEST)
            checkpoint_manifest = json.loads(
                (marker.parent / CHECKPOINT_MANIFEST).read_text(encoding="utf-8")
            )
            checkpoint_record = checkpoint_manifest.get("record", {})
            checkpoint_fingerprint = checkpoint_record.get("checkpoint_fingerprint")
            if not isinstance(checkpoint_fingerprint, str):
                raise RuntimeError("checkpoint manifest lacks its fingerprint")
            command = _evaluation_command(
                checkpoint=marker.parent,
                dataset_root=dataset_root,
                split="validation",
                action_bound_mode=action_bound_mode,
                report_path=full_dir / f"validation_{index:02d}_{checkpoint_index:03d}.json",
            )
            if checkpoint_index == len(checkpoints) - 1:
                command.append("--lock-selection")
            ok, evaluation_report = _run(command)
            executed = (
                ok
                and evaluation_report is not None
                and evaluation_report.get("physical_execution") is True
                and evaluation_report.get("infrastructure_failure_count") == 0
            )
            run_ok &= executed
            if not executed:
                break
        if not run_ok:
            all_closed_loop = False
            break
        if not (run_root / "checkpoint_selection.json").is_file():
            raise RuntimeError("full validation did not produce an exact immutable selection lock")
        selection_path = run_root / "checkpoint_selection.json"
        _validate_owned_path(run_root, selection_path)
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selected = selection["selected_checkpoint_fingerprint"].removeprefix("sha256:")
        selected_dir = next((run_root / "checkpoints").glob(f"*-{selected[:12]}"))
        for split in ("test", "fresh_seed"):
            command = _evaluation_command(
                checkpoint=selected_dir,
                dataset_root=dataset_root,
                split=split,
                action_bound_mode=action_bound_mode,
                report_path=full_dir / f"{split}_{index:02d}.json",
            )
            if split == "fresh_seed":
                command.append("--counterfactual-sensitivity")
            ok, evaluation_report = _run(command)
            executed = (
                ok
                and evaluation_report is not None
                and evaluation_report.get("physical_execution") is True
                and evaluation_report.get("infrastructure_failure_count") == 0
            )
            run_ok &= executed
            if split == "fresh_seed":
                fresh_completed[index] = executed
            if not executed:
                break
        run_completed[index] = run_ok
        all_closed_loop &= run_ok
        if not run_ok:
            break
    all_fresh = all(fresh_completed)
    compare_ok = False
    if all_closed_loop and all_fresh:
        command = [sys.executable, "scripts/compare_act_baselines.py"]
        for train_report in trained[:6]:
            command.extend(
                [
                    "--per-task-manifest",
                    str(Path(str(train_report["expected_output_directory"])) / "run_manifest.json"),
                ]
            )
        command.extend(
            [
                "--mixed-unconditioned-manifest",
                str(Path(str(trained[6]["expected_output_directory"])) / "run_manifest.json"),
                "--mixed-task-onehot-manifest",
                str(Path(str(trained[7]["expected_output_directory"])) / "run_manifest.json"),
                "--output",
                str(full_dir / "comparison.json"),
            ]
        )
        compare_ok, comparison = _run(command)
        if comparison is not None:
            report.baseline_quality_validated = bool(
                comparison.get("baseline_quality_validated", False)
            )
    report.closed_loop_inference_validated = all_closed_loop
    report.fresh_seed_benchmark_completed = all_fresh
    report.per_task_experiment_completed = all(run_completed[:6])
    report.mixed_unconditioned_experiment_completed = run_completed[6]
    report.mixed_task_onehot_experiment_completed = run_completed[7]
    report.full_experiment_validated = all_closed_loop and all_fresh and compare_ok
    report.physical_target_validated = report.full_experiment_validated
    report.check(
        "full validation selection, locked test, and fresh-seed evaluation",
        report.full_experiment_validated,
        "six per-task plus two mixed runs completed the declared schedules",
    )


def main() -> int:
    REPORT_PATH.unlink(missing_ok=True)
    args = parse_args()
    mode = _mode(args)
    dataset_root = args.dataset_root.resolve()
    report = Report()
    try:
        _structural_checks(report)
        if mode != "structural":
            if args.action_bound_mode != ActionBoundMode.PROJECT.value:
                raise RuntimeError(
                    "M4 target smoke/full requires explicit --action-bound-mode project"
                )
            native_target = platform.system() == "Linux" and torch.cuda.is_available()
            report.check(
                "native Linux CUDA target",
                native_target,
                f"platform={platform.system()}; cuda={torch.cuda.is_available()}",
            )
            if native_target and report.implementation_validated:
                target_git = inspect_git_state(PROJECT_ROOT)
                report.check(
                    "clean target Git worktree",
                    not target_git.dirty,
                    f"commit={target_git.commit}; dirty={target_git.dirty}",
                )
                validate_git_for_run(
                    target_git,
                    mode=ExperimentMode.FULL,
                    allow_dirty_development=False,
                )
                dataset_root = _run_prior_target_gate(report, mode, args, dry_run=args.dry_run)
                if mode == "target_smoke":
                    _run_target_smoke(
                        report,
                        dataset_root,
                        args.model_root.resolve(),
                        args,
                    )
                else:
                    if args.dry_run:
                        _run_target_full_dry_run(
                            report,
                            dataset_root,
                            args.model_root.resolve(),
                            action_bound_mode=args.action_bound_mode,
                        )
                    else:
                        _run_target_full(
                            report,
                            dataset_root,
                            args.model_root.resolve(),
                            action_bound_mode=args.action_bound_mode,
                        )
        else:
            report.check(
                "physical target status",
                not report.physical_target_validated,
                "structural mode intentionally does not claim CUDA, M3B, or closed-loop physics",
            )
    except Exception as error:  # noqa: BLE001 - verifier command boundary
        traceback.print_exc()
        report.check(
            "unexpected verifier exception",
            False,
            f"{type(error).__name__}: {str(error) or repr(error)}",
        )
    report.write(mode=mode, dataset_root=dataset_root, dry_run=args.dry_run)
    print(f"[INFO] report: {REPORT_PATH}")
    return 0 if report.accepted(mode, dry_run=args.dry_run) else 1


if __name__ == "__main__":
    raise SystemExit(main())
