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
    closed_loop_inference_validated: bool = False
    train_stats_leakage_validated: bool = False
    validation_selection_validated: bool = False
    test_lock_validated: bool = False
    per_task_experiment_completed: bool = False
    mixed_unconditioned_experiment_completed: bool = False
    mixed_task_onehot_experiment_completed: bool = False
    fresh_seed_benchmark_completed: bool = False
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

    def accepted(self, mode: VerificationMode) -> bool:
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
                    self.closed_loop_inference_validated,
                    self.physical_target_validated,
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

    def write(self, *, mode: VerificationMode, dataset_root: Path) -> None:
        payload = {
            "schema_version": "langmani-m4-verification-v1",
            "verification_mode": mode,
            "dataset_root": str(dataset_root.resolve()),
            **{key: value for key, value in asdict(self).items() if key != "checks"},
            "checks": self.checks,
            "passed": self.accepted(mode),
        }
        atomic_write_json(REPORT_PATH, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--target-smoke", action="store_true")
    modes.add_argument("--target-full", action="store_true")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--model-root",
        "--output-root",
        dest="model_root",
        type=Path,
        default=DEFAULT_MODEL_ROOT,
    )
    parser.add_argument("--source-root", type=Path)
    return parser.parse_args()


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
    if completed.returncode != 0:
        return False, None
    output_flag = next((flag for flag in ("--report", "--output") if flag in arguments), None)
    report_flag = arguments.index(output_flag) + 1 if output_flag is not None else None
    if report_flag is None:
        return True, None
    path = Path(arguments[report_flag])
    value = json.loads(path.read_text(encoding="utf-8"))
    return value.get("passed") is True, value


def _run_prior_target_gate(
    report: Report, mode: VerificationMode, args: argparse.Namespace
) -> Path:
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
    if task_id is not None:
        command.extend(["--task-id", task_id])
    return command


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
    sensitivity: bool = False,
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
        "--report",
        str(report_path),
    ]
    if sensitivity:
        command.append("--counterfactual-sensitivity")
    return _run(command)


def _run_target_smoke(report: Report, dataset_root: Path, model_root: Path) -> None:
    report.source_dataset_validated = True
    smoke_dir = REPORT_PATH.parent / "target_smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    per_ok, per_train = _run(
        _training_command(
            dataset_root=dataset_root,
            model_root=model_root,
            variant=ActVariant.PER_TASK,
            task_id=CANONICAL_TASK_IDS[0],
            mode="tiny-overfit",
            steps=5_000,
            report_path=smoke_dir / "per_task_train.json",
        )
    )
    onehot_ok, onehot_train = _run(
        _training_command(
            dataset_root=dataset_root,
            model_root=model_root,
            variant=ActVariant.MIXED_TASK_ONEHOT,
            task_id=None,
            mode="tiny-overfit",
            steps=10_000,
            report_path=smoke_dir / "onehot_train.json",
        )
    )
    report.cuda_training_validated = per_ok and onehot_ok
    report.check(
        "real CUDA tiny training",
        report.cuda_training_validated,
        "per-task and six-task one-hot ACT training commands completed",
    )
    if not report.cuda_training_validated or per_train is None or onehot_train is None:
        return
    per_eval_ok, per_eval = _evaluate_last_checkpoint(
        train_report=per_train,
        dataset_root=dataset_root,
        split="train",
        report_path=smoke_dir / "per_task_rollout.json",
    )
    onehot_eval_ok, onehot_eval = _evaluate_last_checkpoint(
        train_report=onehot_train,
        dataset_root=dataset_root,
        split="train",
        report_path=smoke_dir / "onehot_rollout.json",
        sensitivity=True,
    )
    per_run_root = Path(str(per_train["expected_output_directory"]))
    _validate_owned_path(model_root, per_run_root)
    _validate_owned_path(per_run_root, per_run_root / "reports" / "training_summary.json")
    per_summary = json.loads(
        (per_run_root / "reports" / "training_summary.json").read_text(encoding="utf-8")
    )
    initial_loss = float(per_summary["initial_validation_loss"])
    final_loss = float(per_summary["final_validation_loss"])
    per_task_loss_reduced = (
        initial_loss > 0.0 and final_loss <= initial_loss * TINY_OVERFIT_MAX_FINAL_LOSS_RATIO
    )
    onehot_run_root = Path(str(onehot_train["expected_output_directory"]))
    _validate_owned_path(model_root, onehot_run_root)
    sensitivity_path = onehot_run_root / "reports" / "counterfactual_sensitivity.json"
    _validate_owned_path(onehot_run_root, sensitivity_path)
    sensitivity_payload = (
        json.loads(sensitivity_path.read_text(encoding="utf-8"))
        if sensitivity_path.is_file()
        else {}
    )
    pairwise = sensitivity_payload.get("pairwise_chunk_distances", {})
    onehot_task_sensitive = isinstance(pairwise, dict) and any(
        float(value) > TASK_SENSITIVITY_MIN_CHUNK_DISTANCE for value in pairwise.values()
    )
    real_closed_loop = all(
        evaluation is not None
        and evaluation.get("physical_execution") is True
        and evaluation.get("infrastructure_failure_count") == 0
        for evaluation in (per_eval, onehot_eval)
    )
    onehot_six = (
        onehot_eval_ok
        and onehot_eval is not None
        and onehot_eval.get("episode_count") == 6
        and onehot_eval.get("success_rate") == 1.0
    )
    report.tiny_overfit_validated = (
        per_eval_ok and per_task_loss_reduced and onehot_six and onehot_task_sensitive
    )
    report.closed_loop_inference_validated = per_eval_ok and onehot_eval_ok and real_closed_loop
    report.checkpoint_reload_validated = report.checkpoint_reload_validated and real_closed_loop
    report.baseline_quality_validated = onehot_six and per_task_loss_reduced
    report.physical_target_validated = (
        report.cuda_training_validated
        and report.tiny_overfit_validated
        and report.closed_loop_inference_validated
    )
    report.check(
        "task-one-hot tiny 6/6 quality gate",
        report.tiny_overfit_validated,
        "per-task loss must fall by at least 50%; one-hot predictions must differ and all six tasks succeed",
    )


def _run_target_full(report: Report, dataset_root: Path, model_root: Path) -> None:
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
                    command = [
                        sys.executable,
                        "scripts/evaluate_act.py",
                        "--checkpoint",
                        str(run_root / selected_records[0].relative_path),
                        "--dataset-root",
                        str(dataset_root),
                        "--split",
                        split,
                        "--report",
                        str(full_dir / f"reuse_{split}_{index:02d}.json"),
                    ]
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
            command = [
                sys.executable,
                "scripts/evaluate_act.py",
                "--checkpoint",
                str(marker.parent),
                "--dataset-root",
                str(dataset_root),
                "--split",
                "validation",
                "--report",
                str(full_dir / f"validation_{index:02d}_{checkpoint_index:03d}.json"),
            ]
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
            command = [
                sys.executable,
                "scripts/evaluate_act.py",
                "--checkpoint",
                str(selected_dir),
                "--dataset-root",
                str(dataset_root),
                "--split",
                split,
                "--report",
                str(full_dir / f"{split}_{index:02d}.json"),
            ]
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
                dataset_root = _run_prior_target_gate(report, mode, args)
                if mode == "target_smoke":
                    _run_target_smoke(report, dataset_root, args.model_root.resolve())
                else:
                    _run_target_full(report, dataset_root, args.model_root.resolve())
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
    report.write(mode=mode, dataset_root=dataset_root)
    print(f"[INFO] report: {REPORT_PATH}")
    return 0 if report.accepted(mode) else 1


if __name__ == "__main__":
    raise SystemExit(main())
