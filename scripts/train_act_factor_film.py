"""Build, inspect, fixture-test, or train the single M4.3b FactorFiLM policy.

The implementation-stage commands are ``--dry-run`` and ``--fixture``.  The
``--target-development`` path is an explicit future authorization boundary:
it requires the completed M3B dataset, the immutable M4.3a evidence, clean Git,
CUDA, and the locked 100,000-step configuration.  This command never opens
M3B test, historical fresh-seed, m42_dev_v0, or m42_final_v0 schedules.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch

from langmani.policies.act_checkpoint import (
    CHECKPOINT_COMPLETION_MARKER,
    load_act_checkpoint,
    load_act_checkpoint_manifest,
    save_act_checkpoint,
)
from langmani.policies.act_factor_film_adapter import FactorFiLMACTPolicy
from langmani.policies.act_factor_film_training import (
    FactorFiLMProcessedBatchAugmenter,
    build_factor_film_checkpoint_contract,
    build_factor_film_dry_run_report,
    build_factor_film_policy_and_processors,
    build_factor_film_run_identity,
    build_factor_film_training_manifest,
    build_factor_film_validation_queue,
    factor_film_dataloader,
    fixture_dry_run_report,
    internal_act_experiment,
    load_factor_film_data,
    load_factor_film_temporal_views,
    run_factor_film_fixture,
    validate_authorizing_semantic_audit,
)
from langmani.policies.act_factor_film_types import (
    FactorFiLMContractError,
    FactorFiLMRunIdentity,
    FactorFiLMTrainingConfig,
    FactorFiLMTrainingManifest,
    FactorFiLMTrainingMode,
    canonical_fingerprint,
)
from langmani.policies.act_runtime import (
    atomic_write_json,
    inspect_git_state,
    safe_run_directory,
)
from langmani.policies.act_training import (
    evaluate_offline_loss,
    make_optimizer,
    seed_everything,
    train_act,
)
from langmani.policies.act_types import (
    ActDataConfig,
    ActModelConfig,
    ActOptimizationConfig,
    CheckpointRecord,
    TrainingState,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "outputs" / "datasets" / "m3b" / "langmani-pick-place-lerobot-v1"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "models" / "act-factor-film"
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "diagnostics" / "m43" / "train_act_factor_film.json"
PROTECTED_SOURCE_ROOTS = tuple(
    PROJECT_ROOT / name for name in ("src", "scripts", "environment", "tests", "docs", ".git")
)
PROTECTED_REPOSITORY_FILES = tuple(
    PROJECT_ROOT / name
    for name in (".gitignore", "AGENTS.md", "PLAN.md", "README.md", "pyproject.toml")
) + (PROJECT_ROOT / "outputs" / ".gitkeep",)
PROTECTED_HISTORICAL_ROOTS = (
    PROJECT_ROOT / "outputs" / "models" / "act",
    PROJECT_ROOT / "outputs" / "models" / "act-task-token",
    PROJECT_ROOT / "outputs" / "diagnostics" / "m42",
)
_FORBIDDEN_PATH_IDENTITIES = ("m3b_test", "fresh_seed", "m42_final_v0")


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=_non_negative_int, default=0)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--fixture", action="store_true")
    modes.add_argument("--target-development", action="store_true")
    parser.add_argument(
        "--fixture-contract",
        action="store_true",
        help="Use generated arrays only; valid solely with --dry-run for non-target CI.",
    )
    parser.add_argument("--resume-checkpoint")
    parser.add_argument(
        "--clean-staging",
        action="store_true",
        help="Remove incomplete checkpoint staging only after run-identity validation.",
    )
    return parser.parse_args()


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected one JSON object in {path}")
    return value


def _write_or_validate(path: Path, payload: Mapping[str, object]) -> None:
    if path.is_file():
        if _read_object(path) != dict(payload):
            raise RuntimeError(f"existing immutable run contract differs: {path}")
        return
    atomic_write_json(path, dict(payload), immutable=True)


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _overlaps(left: Path, right: Path) -> bool:
    return _contains(left, right) or _contains(right, left)


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _resolved_unlinked(path: Path, *, label: str) -> Path:
    lexical = _lexical_absolute(path)
    for component in (lexical, *lexical.parents):
        if component.is_symlink() or component.is_junction():
            raise FactorFiLMContractError(f"{label} traverses a symlink or junction: {component}")
    return lexical.resolve(strict=False)


def _reject_forbidden_path_identity(path: Path, *, label: str) -> None:
    lowered = tuple(part.lower() for part in path.parts)
    if any(fragment in part for part in lowered for fragment in _FORBIDDEN_PATH_IDENTITIES):
        raise FactorFiLMContractError(
            f"{label} names a prohibited test, fresh-seed, or final source"
        )


def _protected_paths(args: argparse.Namespace) -> tuple[Path, ...]:
    values = [
        args.dataset_root,
        *PROTECTED_SOURCE_ROOTS,
        *PROTECTED_REPOSITORY_FILES,
        *PROTECTED_HISTORICAL_ROOTS,
    ]
    if args.evidence_root is not None:
        values.append(args.evidence_root)
    return tuple(_resolved_unlinked(value, label="protected FactorFiLM path") for value in values)


def _validated_report_path(args: argparse.Namespace) -> Path:
    report = _resolved_unlinked(args.report, label="FactorFiLM command report")
    output = _resolved_unlinked(args.output_root, label="FactorFiLM output root")
    _reject_forbidden_path_identity(report, label="command report")
    protected = _protected_paths(args)
    if _overlaps(report, output) or any(_overlaps(report, value) for value in protected):
        raise FactorFiLMContractError(
            "FactorFiLM report must not overlap output, dataset, evidence, source, Git, or historical artifacts"
        )
    if report.exists() and (not report.is_file() or report.is_symlink() or report.is_junction()):
        raise FactorFiLMContractError("FactorFiLM command report path is unsafe")
    return report


def _validate_paths(args: argparse.Namespace) -> None:
    dataset = _resolved_unlinked(args.dataset_root, label="M3B dataset")
    output = _resolved_unlinked(args.output_root, label="FactorFiLM output root")
    _reject_forbidden_path_identity(dataset, label="dataset root")
    _reject_forbidden_path_identity(output, label="output root")
    if args.evidence_root is not None:
        evidence = _resolved_unlinked(args.evidence_root, label="M4.3a evidence root")
        _reject_forbidden_path_identity(evidence, label="evidence root")
    _validated_report_path(args)
    if output.exists() and (not output.is_dir() or output.is_symlink() or output.is_junction()):
        raise FactorFiLMContractError("FactorFiLM output root must be a real directory")
    if any(_overlaps(output, value) for value in _protected_paths(args)):
        raise FactorFiLMContractError(
            "FactorFiLM output root must not contain or overlap protected inputs or sources"
        )


def _training_config(
    args: argparse.Namespace, evidence_fingerprint: str
) -> FactorFiLMTrainingConfig:
    if args.target_development and args.device != "cuda":
        raise FactorFiLMContractError("--target-development requires --device cuda")
    mode = (
        FactorFiLMTrainingMode.TARGET_DEVELOPMENT
        if args.target_development
        else FactorFiLMTrainingMode.DRY_RUN
    )
    return FactorFiLMTrainingConfig(
        data=ActDataConfig(dataset_root=str(args.dataset_root.resolve())),
        base_model=ActModelConfig(),
        optimization=ActOptimizationConfig(),
        semantic_audit_evidence_fingerprint=evidence_fingerprint,
        training_seed=args.seed,
        mode=mode,
        device=args.device,
    )


def _truncate_metrics(
    path: Path,
    checkpoint_step: int,
    *,
    recovery_record: Mapping[str, object] | None,
) -> None:
    retained: list[str] = []
    previous = 0
    found = False
    checkpoint_record: dict[str, object] | None = None
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else ()
    for line in lines:
        value = json.loads(line)
        if not isinstance(value, dict) or not isinstance(value.get("step"), int):
            raise RuntimeError("FactorFiLM metrics.jsonl contains a malformed record")
        step = int(value["step"])
        if step <= previous:
            raise RuntimeError("FactorFiLM metrics steps are not strictly increasing")
        previous = step
        if step <= checkpoint_step:
            retained.append(json.dumps(value, sort_keys=True))
            found |= step == checkpoint_step
            if step == checkpoint_step:
                checkpoint_record = value
    if not found:
        if recovery_record is None or recovery_record.get("step") != checkpoint_step:
            raise RuntimeError(
                "metrics.jsonl lacks the resume step and checkpoint recovery evidence"
            )
        if previous >= checkpoint_step:
            raise RuntimeError("checkpoint metric recovery would break strict step ordering")
        retained.append(json.dumps(dict(recovery_record), sort_keys=True))
    elif recovery_record is not None and checkpoint_record != dict(recovery_record):
        unbound = {**dict(recovery_record), "checkpoint_path": None}
        if checkpoint_record == unbound:
            retained[-1] = json.dumps(dict(recovery_record), sort_keys=True)
        else:
            raise RuntimeError("metrics checkpoint record differs from saved recovery evidence")
    staging = path.with_name(f".{path.name}.resume-staging")
    with staging.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(retained) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staging, path)


def _completed_checkpoint_paths(run_root: Path) -> set[str]:
    completed: set[str] = set()
    for marker in (run_root / "checkpoints").glob(f"*/{CHECKPOINT_COMPLETION_MARKER}"):
        if marker.is_symlink() or marker.is_junction() or not marker.is_file():
            raise RuntimeError("FactorFiLM checkpoint completion marker is unsafe")
        completed.add(marker.parent.relative_to(run_root).as_posix())
    return completed


def _bound_checkpoint_validation_loss(
    *,
    run_root: Path,
    identity: FactorFiLMRunIdentity,
    checkpoint: CheckpointRecord,
) -> float:
    loaded = load_act_checkpoint_manifest(
        run_root=run_root,
        checkpoint_relative_path=checkpoint.relative_path,
        expected_identity=identity,
    )
    if loaded.record != checkpoint or loaded.training_metric is None:
        raise RuntimeError("FactorFiLM checkpoint lacks its bound validation metric")
    metric = loaded.training_metric
    loss = metric.get("validation_loss")
    if (
        isinstance(loss, bool)
        or not isinstance(loss, int | float)
        or not math.isfinite(float(loss))
        or float(loss) < 0
    ):
        raise RuntimeError("FactorFiLM checkpoint validation loss is missing or nonfinite")
    return float(loss)


def _resume_is_atomic_orphan(
    run_root: Path,
    *,
    requested: str,
    manifest: FactorFiLMTrainingManifest,
) -> bool:
    normalized = requested.replace("\\", "/")
    declared = {value.relative_path for value in manifest.checkpoints}
    completed = _completed_checkpoint_paths(run_root)
    if not declared.issubset(completed):
        raise RuntimeError("a manifest-declared FactorFiLM checkpoint is not complete")
    orphans = completed - declared
    if len(orphans) > 1:
        raise RuntimeError("multiple promoted FactorFiLM checkpoint orphans are ambiguous")
    if manifest.checkpoints and normalized == manifest.checkpoints[-1].relative_path:
        if orphans:
            raise RuntimeError("the newer promoted FactorFiLM orphan must be resumed first")
        return False
    if normalized in orphans and orphans == {normalized}:
        return True
    raise RuntimeError(
        "resume requires the latest declared checkpoint or sole promoted atomic orphan"
    )


def _clean_staging(run_root: Path, identity_payload: Mapping[str, object]) -> int:
    root = _resolved_unlinked(run_root, label="FactorFiLM run root")
    manifest_path = root / "run_manifest.json"
    if not run_root.exists():
        return 0
    if not root.is_dir() or root.is_symlink() or root.is_junction():
        raise RuntimeError("FactorFiLM run root must be one real directory")
    manifest = FactorFiLMTrainingManifest.from_dict(_read_object(manifest_path))
    if manifest.identity.to_dict() != dict(identity_payload):
        raise RuntimeError("staging cleanup refused a mismatched run identity")
    staging = root / ".checkpoint-staging"
    if not staging.exists():
        return 0
    if not staging.is_dir() or staging.is_symlink() or staging.is_junction():
        raise RuntimeError("FactorFiLM checkpoint staging root is unsafe")
    removed = 0
    for candidate in staging.iterdir():
        if candidate.is_symlink() or candidate.is_junction() or not candidate.is_dir():
            raise RuntimeError("FactorFiLM staging contains a linked or special entry")
        if not candidate.name.startswith("step-"):
            raise RuntimeError("FactorFiLM staging entry has an unknown owner/name")
        for artifact in candidate.rglob("*"):
            if artifact.is_symlink() or artifact.is_junction():
                raise RuntimeError("FactorFiLM staging entry contains a linked artifact")
            if not artifact.is_file() and not artifact.is_dir():
                raise RuntimeError("FactorFiLM staging entry contains a special artifact")
        resolved = candidate.resolve(strict=True)
        if resolved.parent != staging.resolve():
            raise RuntimeError("checkpoint staging entry escaped its owned directory")
        shutil.rmtree(candidate)
        removed += 1
    return removed


def _execute_fixture(args: argparse.Namespace, git_commit: str) -> dict[str, object]:
    if args.resume_checkpoint is not None or args.clean_staging:
        raise FactorFiLMContractError("fixture mode cannot resume or clean target staging")
    evidence_verified = False
    if args.evidence_root is not None:
        validate_authorizing_semantic_audit(args.evidence_root)
        evidence_verified = True
    result = run_factor_film_fixture(git_commit)
    return {
        "schema_version": "langmani-m43-factor-film-command-v0",
        **result,
        "semantic_audit_evidence_verified": evidence_verified,
        "dataset_accessed": False,
        "training_mode": "fixture",
        "development_quality_gate_passed": False,
        "final_benchmark_authorized": False,
        "validation_only_selection_validated": False,
    }


def _execute_fixture_dry_run(args: argparse.Namespace, git_commit: str) -> dict[str, object]:
    if args.resume_checkpoint is not None or args.clean_staging:
        raise FactorFiLMContractError("fixture dry-run cannot resume or clean target staging")
    report = fixture_dry_run_report(output_root=args.output_root, git_commit=git_commit)
    return {
        "schema_version": "langmani-m43-factor-film-command-v0",
        "passed": True,
        **report.to_dict(),
        "dataset_accessed": False,
        "training_mode": "dry_run_fixture_contract",
        "factor_film_training_completed": False,
        "factor_film_checkpoints_complete": False,
        "factor_film_checkpoint_selected": False,
        "validation_only_selection_validated": False,
        "development_benchmark_completed": False,
        "development_quality_gate_passed": False,
        "final_benchmark_authorized": False,
        "physical_target_validated": False,
        "smolvla_go": False,
    }


def _execute_real(args: argparse.Namespace, git: Any) -> dict[str, object]:
    if args.evidence_root is None:
        raise FactorFiLMContractError("real dry-run/target-development requires --evidence-root")
    if args.resume_checkpoint is not None and not args.target_development:
        raise FactorFiLMContractError("--resume-checkpoint is valid only with target development")
    if args.clean_staging and not args.target_development:
        raise FactorFiLMContractError("--clean-staging is valid only with target development")
    completed_audit = validate_authorizing_semantic_audit(args.evidence_root)
    config = _training_config(args, completed_audit.evidence_fingerprint)
    if args.target_development:
        if git.dirty or not git.baseline_tracked:
            raise FactorFiLMContractError(
                "target-development FactorFiLM requires a clean tracked Git baseline"
            )
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("target-development requires CUDA with bfloat16 support")
    data = load_factor_film_data(args.dataset_root)
    seed_everything(config.training_seed)
    act_config, policy, preprocessor, postprocessor, architecture = (
        build_factor_film_policy_and_processors(
            training_config=config,
            statistics=data.statistics,
        )
    )
    identity = build_factor_film_run_identity(
        training_config=config,
        data=data,
        architecture=architecture,
        git=git,
    )
    checkpoint_contract = build_factor_film_checkpoint_contract(
        identity=identity,
        architecture=architecture,
    )
    dry_report = build_factor_film_dry_run_report(
        training_config=config,
        data=data,
        architecture=architecture,
        identity=identity,
        output_root=args.output_root,
    )
    common = {
        "schema_version": "langmani-m43-factor-film-command-v0",
        **dry_report.to_dict(),
        "architecture_identity": architecture.to_dict(),
        "checkpoint_contract": checkpoint_contract.to_dict(),
        "git": git.to_dict(),
        "semantic_audit_evidence_verified": True,
        "test_accessible": False,
        "historical_fresh_accessible": False,
        "development_schedule_accessed": False,
        "development_benchmark_completed": False,
        "development_quality_gate_passed": False,
        "final_benchmark_authorized": False,
        "final_schedule_accessed": False,
        "physical_target_validated": False,
        "smolvla_go": False,
        "staging_entries_removed": 0,
        "validation_only_selection_validated": False,
    }
    if args.dry_run:
        return {
            **common,
            "passed": True,
            "training_mode": "dry_run",
            "factor_film_training_completed": False,
            "factor_film_checkpoints_complete": False,
            "factor_film_checkpoint_selected": False,
            "validation_only_selection_validated": False,
        }

    run_root = safe_run_directory(args.output_root, identity.run_fingerprint)
    resuming = args.resume_checkpoint is not None
    if run_root.exists() and not resuming:
        raise FileExistsError(f"fingerprint-owned run is immutable: {run_root}")
    if resuming and not run_root.is_dir():
        raise FileNotFoundError("resume requires the fingerprint-owned run directory")
    if resuming and (run_root / "complete.json").exists():
        raise RuntimeError("completed FactorFiLM runs are immutable and cannot resume")
    if args.clean_staging:
        common["staging_entries_removed"] = _clean_staging(run_root, identity.to_dict())
    run_root.mkdir(parents=True, exist_ok=True)

    experiment = internal_act_experiment(config)
    optimizer = make_optimizer(policy, experiment)
    state = TrainingState(global_step=0, examples_processed=0)
    checkpoints: list[CheckpointRecord] = []
    checkpoint_validation_losses: dict[str, float] = {}
    manifest_path = run_root / "run_manifest.json"
    if resuming:
        existing = FactorFiLMTrainingManifest.from_dict(_read_object(manifest_path))
        if (
            existing.identity != identity
            or existing.training_config != config
            or existing.architecture_identity != architecture
            or existing.checkpoint_contract != checkpoint_contract
        ):
            raise RuntimeError("resume manifest differs from the requested semantic identity")
        orphan = _resume_is_atomic_orphan(
            run_root,
            requested=cast(str, args.resume_checkpoint),
            manifest=existing,
        )
        checkpoints.extend(existing.checkpoints)
        checkpoint_validation_losses.update(
            {
                checkpoint.checkpoint_fingerprint: _bound_checkpoint_validation_loss(
                    run_root=run_root,
                    identity=identity,
                    checkpoint=checkpoint,
                )
                for checkpoint in existing.checkpoints
            }
        )
        loaded = load_act_checkpoint(
            run_root=run_root,
            checkpoint_relative_path=cast(str, args.resume_checkpoint),
            expected_identity=identity,
            optimizer=optimizer,
            for_resume=True,
            existing_policy=policy,
            policy_class=FactorFiLMACTPolicy,
        )
        if not isinstance(loaded.policy, FactorFiLMACTPolicy) or loaded.optimizer is None:
            raise RuntimeError("FactorFiLM resume did not restore policy and optimizer")
        policy = loaded.policy
        preprocessor = loaded.preprocessor
        postprocessor = loaded.postprocessor
        optimizer = loaded.optimizer
        state = loaded.training_state
        if orphan:
            previous_step = checkpoints[-1].global_step if checkpoints else 0
            expected_orphan_step = min(
                previous_step + config.optimization.checkpoint_interval,
                config.optimization.training_steps,
            )
            if loaded.record.global_step != expected_orphan_step:
                raise RuntimeError(
                    "promoted FactorFiLM orphan is not the next configured checkpoint"
                )
            checkpoints.append(loaded.record)
            checkpoint_validation_losses[loaded.record.checkpoint_fingerprint] = (
                _bound_checkpoint_validation_loss(
                    run_root=run_root,
                    identity=identity,
                    checkpoint=loaded.record,
                )
            )
            state = TrainingState(
                global_step=loaded.training_state.global_step,
                examples_processed=loaded.training_state.examples_processed,
                last_checkpoint_fingerprint=loaded.record.checkpoint_fingerprint,
            )
            atomic_write_json(
                manifest_path,
                build_factor_film_training_manifest(
                    identity=identity,
                    training_config=config,
                    architecture=architecture,
                    checkpoint_contract=checkpoint_contract,
                    training_state=state,
                    checkpoints=tuple(checkpoints),
                    training_complete=False,
                ).to_dict(),
            )
        recovered_metric = (
            {**dict(loaded.training_metric), "checkpoint_path": loaded.record.relative_path}
            if loaded.training_metric is not None
            else None
        )
        _truncate_metrics(
            run_root / "metrics.jsonl",
            state.global_step,
            recovery_record=recovered_metric,
        )

    _write_or_validate(run_root / "training_config.json", config.to_dict())
    _write_or_validate(run_root / "architecture_identity.json", architecture.to_dict())
    _write_or_validate(run_root / "checkpoint_contract.json", checkpoint_contract.to_dict())
    _write_or_validate(run_root / "train_statistics.json", data.statistics.to_dict())
    if not resuming:
        atomic_write_json(
            manifest_path,
            build_factor_film_training_manifest(
                identity=identity,
                training_config=config,
                architecture=architecture,
                checkpoint_contract=checkpoint_contract,
                training_state=state,
                checkpoints=(),
                training_complete=False,
            ).to_dict(),
            immutable=True,
        )

    def finalize_run(*, final_state: TrainingState, resumed_only: bool) -> dict[str, object]:
        final_manifest = build_factor_film_training_manifest(
            identity=identity,
            training_config=config,
            architecture=architecture,
            checkpoint_contract=checkpoint_contract,
            training_state=final_state,
            checkpoints=tuple(checkpoints),
            training_complete=True,
        )
        queue = build_factor_film_validation_queue(
            identity=identity,
            validation_schedule_digest=data.validation_schedule_digest,
            checkpoints=tuple(checkpoints),
            offline_validation_action_losses=checkpoint_validation_losses,
        )
        atomic_write_json(manifest_path, final_manifest.to_dict())
        _write_or_validate(run_root / "validation_queue.json", queue.to_dict())
        completion = {
            "schema_version": "langmani-m43-factor-film-training-complete-v0",
            "passed": True,
            "run_fingerprint": identity.run_fingerprint,
            "global_step": final_state.global_step,
            "checkpoint_count": len(checkpoints),
            "manifest_fingerprint": canonical_fingerprint(final_manifest.to_dict()),
            "validation_queue_fingerprint": queue.queue_fingerprint,
            "factor_film_training_completed": True,
            "factor_film_checkpoint_selected": False,
            "development_benchmark_completed": False,
            "final_schedule_accessed": False,
            "smolvla_go": False,
        }
        atomic_write_json(run_root / "complete.json", completion, immutable=True)
        return {
            **common,
            "passed": True,
            "training_mode": "target_development",
            "training_started": not resumed_only,
            "resumed_finalization_only": resumed_only,
            "factor_film_training_completed": True,
            "factor_film_checkpoints_complete": True,
            "factor_film_checkpoint_selected": False,
            "global_step": final_state.global_step,
            "checkpoint_count": len(checkpoints),
            "run_root": str(run_root),
            "validation_queue": str(run_root / "validation_queue.json"),
        }

    if resuming and state.global_step == config.optimization.training_steps:
        if len(checkpoints) != 20:
            raise RuntimeError("finalization-only resume requires all 20 checkpoints")
        return finalize_run(
            final_state=TrainingState(
                global_step=state.global_step,
                examples_processed=state.examples_processed,
                last_checkpoint_fingerprint=checkpoints[-1].checkpoint_fingerprint,
                completed=True,
            ),
            resumed_only=True,
        )
    if state.global_step > config.optimization.training_steps:
        raise RuntimeError("resume checkpoint exceeds the locked training horizon")

    train_dataset, validation_dataset = load_factor_film_temporal_views(
        data,
        act_config,
        config,
    )
    train_loader = factor_film_dataloader(
        train_dataset,
        config,
        shuffle=True,
        start_step=state.global_step,
    )
    validation_loader = factor_film_dataloader(validation_dataset, config, shuffle=False)
    augmenter = FactorFiLMProcessedBatchAugmenter(data.completed.views.task_id_by_episode)

    def checkpoint_callback(**values: object) -> str:
        step = int(values["step"])
        checkpoint_state = TrainingState(
            global_step=step,
            examples_processed=int(values["examples_processed"]),
        )
        record = save_act_checkpoint(
            run_root=run_root,
            identity=identity,
            training_state=checkpoint_state,
            policy=values["policy"],
            preprocessor=values["preprocessor"],
            postprocessor=values["postprocessor"],
            optimizer=cast(torch.optim.Optimizer, values["optimizer"]),
            training_metric=cast(Mapping[str, object], values["metric"]),
        )
        checkpoints.append(record)
        metric = cast(Mapping[str, object], values["metric"])
        validation_loss = metric.get("validation_loss")
        if (
            isinstance(validation_loss, bool)
            or not isinstance(validation_loss, int | float)
            or not math.isfinite(float(validation_loss))
            or float(validation_loss) < 0
        ):
            raise RuntimeError("checkpoint metric lacks its finite validation loss")
        checkpoint_validation_losses[record.checkpoint_fingerprint] = float(validation_loss)
        manifest_state = TrainingState(
            global_step=step,
            examples_processed=checkpoint_state.examples_processed,
            last_checkpoint_fingerprint=record.checkpoint_fingerprint,
        )
        atomic_write_json(
            manifest_path,
            build_factor_film_training_manifest(
                identity=identity,
                training_config=config,
                architecture=architecture,
                checkpoint_contract=checkpoint_contract,
                training_state=manifest_state,
                checkpoints=tuple(checkpoints),
                training_complete=False,
            ).to_dict(),
        )
        return record.relative_path

    def validation_callback(step: int, current_policy: object) -> float:
        devices = list(range(torch.cuda.device_count()))
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config.training_seed + 1_000_003 + step)
            torch.cuda.manual_seed_all(config.training_seed + 1_000_003 + step)
            return evaluate_offline_loss(
                experiment=experiment,
                policy=cast(FactorFiLMACTPolicy, current_policy),
                preprocessor=preprocessor,
                validation_batches=validation_loader,
                processed_batch_augmenter=augmenter,
            )

    initial_validation_path = run_root / "reports" / "initial_validation.json"
    if not resuming:
        initial_validation_loss = validation_callback(0, policy)
        atomic_write_json(
            initial_validation_path,
            {
                "schema_version": "langmani-m43-factor-film-initial-validation-v0",
                "run_fingerprint": identity.run_fingerprint,
                "source": "m3b_validation",
                "loss": initial_validation_loss,
            },
            immutable=True,
        )
    elif _read_object(initial_validation_path).get("run_fingerprint") != identity.run_fingerprint:
        raise RuntimeError("resume initial-validation evidence is missing or mismatched")

    outcome = train_act(
        experiment=experiment,
        policy=policy,
        preprocessor=preprocessor,
        optimizer=optimizer,
        train_batches=train_loader,
        start_step=state.global_step,
        maximum_steps=config.optimization.training_steps,
        metrics_path=run_root / "metrics.jsonl",
        checkpoint_callback=checkpoint_callback,
        postprocessor=postprocessor,
        validation_callback=validation_callback,
        seed_at_start=not resuming,
        initial_examples_processed=state.examples_processed,
        processed_batch_augmenter=augmenter,
    )
    if len(checkpoints) != 20 or checkpoints[-1].global_step != outcome.final_step:
        raise RuntimeError("target-development did not produce the exact 20 checkpoints")
    final_state = TrainingState(
        global_step=outcome.final_step,
        examples_processed=outcome.examples_processed,
        last_checkpoint_fingerprint=checkpoints[-1].checkpoint_fingerprint,
        completed=True,
    )
    return finalize_run(final_state=final_state, resumed_only=False)


def execute(args: argparse.Namespace) -> dict[str, object]:
    if args.fixture_contract and not args.dry_run:
        raise FactorFiLMContractError("--fixture-contract is valid only with --dry-run")
    if args.seed != 0:
        raise FactorFiLMContractError("M4.3b exposes only the locked experiment seed 0")
    if (args.fixture or args.fixture_contract) and args.device != "cpu":
        raise FactorFiLMContractError("fixture paths require explicit --device cpu")
    _validate_paths(args)
    git = inspect_git_state(PROJECT_ROOT)
    if args.fixture:
        return _execute_fixture(args, git.commit)
    if args.fixture_contract:
        return _execute_fixture_dry_run(args, git.commit)
    return _execute_real(args, git)


def main() -> int:
    args = parse_args()
    try:
        report = execute(args)
    except Exception as error:  # noqa: BLE001 - command boundary preserves exact diagnostics
        traceback.print_exc()
        report = {
            "schema_version": "langmani-m43-factor-film-command-v0",
            "passed": False,
            "error_type": type(error).__name__,
            "error_message": str(error) or repr(error),
            "factor_film_training_completed": False,
            "factor_film_checkpoint_selected": False,
            "development_benchmark_completed": False,
            "development_quality_gate_passed": False,
            "final_benchmark_authorized": False,
            "final_schedule_accessed": False,
            "physical_target_validated": False,
            "smolvla_go": False,
        }
    report_written = False
    report_path: Path | None = None
    try:
        report_path = _validated_report_path(args)
        atomic_write_json(report_path, report)
        report_written = True
    except Exception:  # noqa: BLE001 - unsafe report paths must remain unwritten
        traceback.print_exc()
    print(
        json.dumps(
            {
                **report,
                "report_written": report_written,
                "report": str(report_path) if report_written and report_path is not None else None,
            },
            sort_keys=True,
        )
    )
    return 0 if report.get("passed") is True and report_written else 1


if __name__ == "__main__":
    raise SystemExit(main())
