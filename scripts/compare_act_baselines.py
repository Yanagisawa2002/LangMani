"""Build the final provenance-bound comparison of all eight M4 ACT runs."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from langmani.datasets.lerobot_types import IMAGE_FEATURE_KEY, STATE_FEATURE_KEY
from langmani.datasets.schedule import CANONICAL_TASK_SPECS
from langmani.environments.specs import stable_scene_id
from langmani.policies.act_analysis import (
    compute_counterfactual_sensitivity,
    counterfactual_sensitivity_from_dict,
    task_identity_effects,
)
from langmani.policies.act_conditioning import CANONICAL_TASK_IDS
from langmani.policies.act_evaluation import (
    TestEvaluationAuthorization,
    load_checkpoint_selection,
    rollout_schedule_digest,
    summarize_rollout_benchmark,
)
from langmani.policies.act_runtime import atomic_write_json
from langmani.policies.act_types import (
    M4_SCHEMA_VERSION,
    ActComparisonReport,
    ActEvaluationConfig,
    ActExperimentManifest,
    ActModelConfig,
    ActOptimizationConfig,
    ActVariant,
    EvaluationSplit,
    ExperimentMode,
    RolloutEpisodeResult,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "models" / "act" / "comparison.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-task-manifest", type=Path, action="append", required=True)
    parser.add_argument("--mixed-unconditioned-manifest", type=Path, required=True)
    parser.add_argument("--mixed-task-onehot-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def _finite_number(value: object, *, minimum: float = 0.0) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and np.isfinite(value)
        and float(value) >= minimum
    )


def _validated_benchmark(
    value: dict[str, Any],
    *,
    variant: ActVariant,
    task_id: str | None,
    m3b_export_fingerprint: str,
    split: EvaluationSplit,
) -> None:
    raw_episodes = value.get("episodes")
    if not isinstance(raw_episodes, list):
        raise RuntimeError("rollout benchmark lacks raw episode records")
    episodes: list[RolloutEpisodeResult] = []
    for raw in raw_episodes:
        if not isinstance(raw, dict):
            raise RuntimeError("rollout episode record is malformed")
        payload = dict(raw)
        for key in (
            "action_min",
            "action_max",
            "inference_latency_ms",
            "environment_step_latency_ms",
        ):
            if not isinstance(payload.get(key), list):
                raise RuntimeError(f"rollout episode {key} must be a JSON list")
            payload[key] = tuple(payload[key])
        episodes.append(RolloutEpisodeResult(**payload))
    expected_count = {
        EvaluationSplit.VALIDATION: 6 if variant is ActVariant.PER_TASK else 36,
        EvaluationSplit.TEST: 6 if variant is ActVariant.PER_TASK else 36,
        EvaluationSplit.FRESH_SEED: 30 if variant is ActVariant.PER_TASK else 180,
    }[split]
    if len(episodes) != expected_count or any(item.split is not split for item in episodes):
        raise RuntimeError(f"{split.value} benchmark does not use its declared episode schedule")
    rebuilt = summarize_rollout_benchmark(tuple(episodes)).to_dict()
    if any(value.get(key) != expected for key, expected in rebuilt.items()):
        raise RuntimeError(f"{split.value} benchmark aggregates disagree with raw episodes")
    task_ids = {item.task_id for item in episodes}
    expected_tasks = {task_id} if variant is ActVariant.PER_TASK else set(CANONICAL_TASK_IDS)
    if task_ids != expected_tasks:
        raise RuntimeError(f"{split.value} benchmark has the wrong TaskSpec coverage")
    by_seed: dict[int, list[RolloutEpisodeResult]] = {}
    for episode in episodes:
        if episode.scene_id != stable_scene_id(episode.scene_seed):
            raise RuntimeError(f"{split.value} benchmark scene ID disagrees with its seed")
        by_seed.setdefault(episode.scene_seed, []).append(episode)
    expected_scene_count = 30 if split is EvaluationSplit.FRESH_SEED else 6
    expected_tasks_per_scene = 1 if variant is ActVariant.PER_TASK else 6
    if len(by_seed) != expected_scene_count or any(
        len(items) != expected_tasks_per_scene or {item.task_id for item in items} != expected_tasks
        for items in by_seed.values()
    ):
        raise RuntimeError(f"{split.value} benchmark is not balanced by scene and TaskSpec")
    task_specs = dict(zip(CANONICAL_TASK_IDS, CANONICAL_TASK_SPECS, strict=True))
    recomputed_digest = rollout_schedule_digest(
        m3b_export_fingerprint=m3b_export_fingerprint,
        variant=variant.value,
        task_id=task_id,
        split=split.value,
        episodes=tuple(
            {
                "scene_seed": episode.scene_seed,
                "task_spec": task_specs[episode.task_id].to_dict(),
            }
            for episode in episodes
        ),
    )
    if value.get("schedule_digest") != recomputed_digest:
        raise RuntimeError(f"{split.value} benchmark raw schedule digest is invalid")
    if (
        value.get("passed") is not True
        or value.get("infrastructure_failure_count") != 0
        or value.get("environment_action_applied") is not True
    ):
        raise RuntimeError(f"{split.value} benchmark lacks real infrastructure-clean execution")


def _manifest_facts(path: Path, expected: ActVariant) -> dict[str, object]:
    value = _read(path)
    manifest = ActExperimentManifest.from_dict(value)
    identity = manifest.identity
    config = manifest.config
    if config.variant is not expected or identity.variant is not expected:
        raise RuntimeError(f"manifest variant mismatch: {path}")
    if (
        config.mode is not ExperimentMode.FULL
        or config.device != "cuda"
        or config.model != ActModelConfig.for_variant(expected)
        or config.optimization != ActOptimizationConfig()
        or config.evaluation != ActEvaluationConfig()
        or manifest.git_dirty
        or identity.git_dirty
        or identity.cuda_version is None
    ):
        raise RuntimeError(f"comparison accepts only clean fixed-primary CUDA full runs: {path}")
    if not manifest.complete:
        raise RuntimeError(f"comparison accepts only completed runs: {path}")
    run_fingerprint = identity.run_fingerprint
    selected_checkpoint = manifest.selected_checkpoint_fingerprint
    if not isinstance(selected_checkpoint, str):
        raise RuntimeError(f"completed run lacks a selected checkpoint: {path}")
    selection = load_checkpoint_selection(path.parent / "checkpoint_selection.json")
    if (
        selection.run_fingerprint != run_fingerprint
        or selection.selected_checkpoint_fingerprint != selected_checkpoint
    ):
        raise RuntimeError(f"checkpoint-selection evidence differs from manifest: {path}")
    declared_checkpoints = {
        (item.checkpoint_fingerprint, item.global_step) for item in manifest.checkpoints
    }
    expected_steps = set(
        range(
            config.optimization.validation_interval,
            config.optimization.training_steps + 1,
            config.optimization.validation_interval,
        )
    )
    expected_candidates = {pair for pair in declared_checkpoints if pair[1] in expected_steps}
    selection_candidates = {
        (item.checkpoint_fingerprint, item.checkpoint_step) for item in selection.candidates
    }
    if selection_candidates != expected_candidates:
        raise RuntimeError(f"selection candidates differ from manifest checkpoints: {path}")
    selected_pair = (selected_checkpoint, selection.selected_checkpoint_step)
    if selected_pair not in declared_checkpoints:
        raise RuntimeError(f"selected checkpoint is absent from manifest evidence: {path}")
    selected_name = selected_checkpoint.removeprefix("sha256:")
    result_artifacts: dict[str, object] = {}
    required_results = [
        f"validation/{selected_name}/benchmark.json",
        "test/benchmark.json",
        "fresh_seed/benchmark.json",
    ]
    required_results.append(
        "reports/per_task_reference.json"
        if expected is ActVariant.PER_TASK
        else "reports/counterfactual_sensitivity.json"
    )
    for relative in required_results:
        result = _read(path.parent / relative)
        if (
            result.get("passed") is not True
            or result.get("run_fingerprint") != run_fingerprint
            or result.get("checkpoint_fingerprint") != selected_checkpoint
            or not isinstance(result.get("evaluation_git"), dict)
            or result["evaluation_git"].get("commit") != identity.git_commit
            or result["evaluation_git"].get("dirty") is not False
        ):
            raise RuntimeError(
                f"incomplete or mismatched result artifact: {path.parent / relative}"
            )
        result_artifacts[relative] = result
    validation_relative = f"validation/{selected_name}/benchmark.json"
    validation_artifact = result_artifacts[validation_relative]
    if (
        not isinstance(validation_artifact, dict)
        or validation_artifact.get("schedule_digest") != selection.validation_schedule_digest
    ):
        raise RuntimeError(f"selected validation artifact schedule is inconsistent: {path}")
    _validated_benchmark(
        validation_artifact,
        variant=expected,
        task_id=identity.task_id,
        m3b_export_fingerprint=identity.m3b_export_fingerprint,
        split=EvaluationSplit.VALIDATION,
    )
    _validated_benchmark(
        result_artifacts["test/benchmark.json"],
        variant=expected,
        task_id=identity.task_id,
        m3b_export_fingerprint=identity.m3b_export_fingerprint,
        split=EvaluationSplit.TEST,
    )
    _validated_benchmark(
        result_artifacts["fresh_seed/benchmark.json"],
        variant=expected,
        task_id=identity.task_id,
        m3b_export_fingerprint=identity.m3b_export_fingerprint,
        split=EvaluationSplit.FRESH_SEED,
    )
    if expected is not ActVariant.PER_TASK:
        sensitivity_path = "reports/counterfactual_sensitivity.json"
        raw_sensitivity = result_artifacts[sensitivity_path]
        if not isinstance(raw_sensitivity, dict):
            raise RuntimeError(f"invalid counterfactual sensitivity artifact: {path}")
        typed_fields = {
            key: raw_sensitivity[key]
            for key in (
                "variant",
                "scene_id",
                "ordered_task_ids",
                "first_actions",
                "action_chunks",
                "pairwise_first_action_distances",
                "pairwise_chunk_distances",
                "identical_unconditioned_inputs",
            )
        }
        sensitivity = counterfactual_sensitivity_from_dict(typed_fields)
        if sensitivity.variant is not expected:
            raise RuntimeError(f"counterfactual sensitivity variant mismatch: {path}")
        expected_effects = task_identity_effects(sensitivity.pairwise_chunk_distances)
        if any(raw_sensitivity.get(key) != value for key, value in expected_effects.items()):
            raise RuntimeError(f"counterfactual object/bin effect summary mismatch: {path}")
        expert_distances = raw_sensitivity.get("expert_chunk_distances_by_task")
        if (
            not isinstance(expert_distances, dict)
            or set(expert_distances) != set(CANONICAL_TASK_IDS)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not np.isfinite(value)
                or value < 0
                for value in expert_distances.values()
            )
        ):
            raise RuntimeError(f"counterfactual expert-distance evidence is invalid: {path}")
        expert_evidence = raw_sensitivity.get("expert_reference_evidence")
        if (
            not isinstance(expert_evidence, dict)
            or expert_evidence.get("panda_state_equivalent") is not True
            or not isinstance(expert_evidence.get("expert_decoded_rgb_digests"), list)
            or len(expert_evidence["expert_decoded_rgb_digests"]) != 6
        ):
            raise RuntimeError(f"counterfactual expert-scene evidence is invalid: {path}")
    for relative in ("test/benchmark.json", "fresh_seed/benchmark.json"):
        artifact = result_artifacts[relative]
        if not isinstance(artifact, dict):
            raise RuntimeError(f"invalid result artifact: {relative}")
        authorization = TestEvaluationAuthorization.from_dict(artifact.get("test_authorization"))
        if (
            not authorization.final_eligible
            or authorization.selection_fingerprint != selection.selection_fingerprint
            or authorization.checkpoint_fingerprint != selected_checkpoint
            or authorization.run_fingerprint != run_fingerprint
            or authorization.actual_schedule_digest != authorization.expected_schedule_digest
            or artifact.get("schedule_digest") != authorization.actual_schedule_digest
        ):
            raise RuntimeError(f"result lacks final selected-checkpoint authorization: {relative}")
    training_summary = _read(path.parent / "reports" / "training_summary.json")
    if (
        training_summary.get("passed") is not True
        or training_summary.get("run_fingerprint") != run_fingerprint
        or training_summary.get("variant") != config.variant.value
        or training_summary.get("task_id") != config.task_id
        or training_summary.get("effective_model_config") != config.model.to_dict()
        or training_summary.get("effective_optimization_config") != config.optimization.to_dict()
    ):
        raise RuntimeError(f"training summary is incomplete or mismatched: {path.parent}")
    duration_complete = training_summary.get("training_duration_complete")
    duration = training_summary.get("training_duration_s")
    required_finite = (
        "initial_train_loss",
        "final_train_loss",
        "initial_validation_loss",
        "final_validation_loss",
        "measured_dataloader_and_optimizer_duration_s",
        "mean_throughput_examples_per_s",
    )
    if (
        not isinstance(training_summary.get("parameter_count"), int)
        or training_summary["parameter_count"] < 1
        or not isinstance(duration_complete, bool)
        or (duration_complete is True and not _finite_number(duration))
        or (duration_complete is False and duration is not None)
        or any(not _finite_number(training_summary.get(key)) for key in required_finite)
        or not isinstance(training_summary.get("peak_gpu_allocated_bytes"), int)
        or training_summary["peak_gpu_allocated_bytes"] < 1
        or not isinstance(training_summary.get("peak_gpu_reserved_bytes"), int)
        or training_summary["peak_gpu_reserved_bytes"] < 1
    ):
        raise RuntimeError(f"training summary metrics are missing or invalid: {path.parent}")
    completion = {
        "schema_version": "langmani-m4-run-completion-v1",
        "run_fingerprint": run_fingerprint,
        "selected_checkpoint_fingerprint": selected_checkpoint,
    }
    completion_path = path.parent / "complete.json"
    if completion_path.is_file():
        if _read(completion_path) != completion:
            raise RuntimeError(f"run completion marker is inconsistent: {completion_path}")
    else:
        # Recover only the narrow crash window after the already-validated
        # manifest/evaluation transaction but before completion publication.
        atomic_write_json(completion_path, completion, immutable=True)
    return {
        "run_fingerprint": run_fingerprint,
        "dataset_fingerprint": identity.m3b_export_fingerprint,
        "split_digest": identity.m3b_split_manifest_digest,
        "git_commit": identity.git_commit,
        "training_seed": identity.training_seed,
        "lerobot_version": identity.lerobot_version,
        "torch_version": identity.torch_version,
        "cuda_version": identity.cuda_version,
        "optimization_config": config.optimization.to_dict(),
        "evaluation_config": config.evaluation.to_dict(),
        "task_id": identity.task_id,
        "selected_checkpoint_fingerprint": selected_checkpoint,
        "checkpoint_selection": selection.to_dict(),
        "result_artifacts": result_artifacts,
        "effective_config": config.to_dict(),
        "training_summary": training_summary,
    }


def _per_task_sensitivity(per_task: tuple[dict[str, object], ...]) -> dict[str, object]:
    by_task = {str(item["task_id"]): item for item in per_task}
    if set(by_task) != set(CANONICAL_TASK_IDS):
        raise RuntimeError("per-task references do not cover the canonical six tasks")
    references = [
        by_task[task_id]["result_artifacts"]["reports/per_task_reference.json"]
        for task_id in CANONICAL_TASK_IDS
    ]
    if not all(isinstance(item, dict) for item in references):
        raise RuntimeError("per-task reference artifacts are malformed")
    if any(
        item.get("panda_state_equivalent") is not True
        or not isinstance(item.get("expert_decoded_rgb_digest"), str)
        for item in references
    ):
        raise RuntimeError("per-task references lack expert-scene equivalence evidence")
    if len({item["expert_decoded_rgb_digest"] for item in references}) != 1:
        raise RuntimeError("per-task references disagree on the decoded expert scene RGB")
    scene_evidence = {
        (
            item.get("scene_seed"),
            item.get("scene_id"),
            item.get("initial_rgb_digest"),
            json.dumps(item.get("initial_panda_state"), sort_keys=True),
        )
        for item in references
    }
    if len(scene_evidence) != 1:
        raise RuntimeError("per-task policy references did not hold the physical scene fixed")
    first_actions = np.asarray([item["first_action"] for item in references], dtype=np.float64)
    chunks = np.asarray([item["action_chunk"] for item in references], dtype=np.float64)
    state = np.asarray(references[0]["initial_panda_state"], dtype=np.float32)
    controlled_input = {
        IMAGE_FEATURE_KEY: np.zeros((1, 3, 1, 1), dtype=np.uint8),
        STATE_FEATURE_KEY: state,
    }
    sensitivity = compute_counterfactual_sensitivity(
        variant=ActVariant.PER_TASK,
        scene_id=str(references[0]["scene_id"]),
        ordered_task_ids=CANONICAL_TASK_IDS,
        first_actions=first_actions,
        action_chunks=chunks,
        policy_inputs=[controlled_input for _ in CANONICAL_TASK_IDS],
    )
    return {
        **sensitivity.to_dict(),
        **task_identity_effects(sensitivity.pairwise_chunk_distances),
        "expert_chunk_distances_by_task": {
            task_id: float(reference["expert_chunk_distance"])
            for task_id, reference in zip(CANONICAL_TASK_IDS, references, strict=True)
        },
    }


def compare(args: argparse.Namespace) -> dict[str, object]:
    if len(args.per_task_manifest) != 6:
        raise ValueError("comparison requires exactly six --per-task-manifest values")
    unordered_per_task = tuple(
        _manifest_facts(path, ActVariant.PER_TASK) for path in args.per_task_manifest
    )
    if len({item["task_id"] for item in unordered_per_task}) != 6:
        raise ValueError("per-task manifests must cover six distinct stable task IDs")
    by_task_id = {item["task_id"]: item for item in unordered_per_task}
    if set(by_task_id) != set(CANONICAL_TASK_IDS):
        raise ValueError("per-task manifests must cover the canonical six stable task IDs")
    per_task = tuple(by_task_id[task_id] for task_id in CANONICAL_TASK_IDS)
    unconditioned = _manifest_facts(
        args.mixed_unconditioned_manifest, ActVariant.MIXED_UNCONDITIONED
    )
    conditioned = _manifest_facts(args.mixed_task_onehot_manifest, ActVariant.MIXED_TASK_ONEHOT)
    all_runs = (*per_task, unconditioned, conditioned)
    datasets = {item["dataset_fingerprint"] for item in all_runs}
    commits = {item["git_commit"] for item in all_runs}
    if len(datasets) != 1 or len(commits) != 1:
        raise ValueError("all compared runs must share one dataset and Git commit")
    controls = {
        (
            item["split_digest"],
            item["training_seed"],
            item["lerobot_version"],
            item["torch_version"],
            item["cuda_version"],
            json.dumps(item["optimization_config"], sort_keys=True),
            json.dumps(item["evaluation_config"], sort_keys=True),
        )
        for item in all_runs
    }
    if len(controls) != 1:
        raise ValueError(
            "all compared runs must share split, seed, runtime, optimization, and evaluation controls"
        )
    report = ActComparisonReport(
        schema_version=M4_SCHEMA_VERSION,
        dataset_fingerprint=str(datasets.pop()),
        git_commit=str(commits.pop()),
        per_task_run_fingerprints=tuple(str(item["run_fingerprint"]) for item in per_task),
        mixed_unconditioned_run_fingerprint=str(unconditioned["run_fingerprint"]),
        mixed_task_onehot_run_fingerprint=str(conditioned["run_fingerprint"]),
        interpretation={
            "per_task": "control learning and dataset quality baseline",
            "mixed_unconditioned": "counterfactual one-to-many ambiguity without a task command",
            "mixed_task_onehot": "oracle discrete task conditioning, not language understanding",
            "language_understanding": "not tested in M4",
        },
        full_experiment_validated=True,
        baseline_quality_validated=all(
            bool(item["result_artifacts"]["test/benchmark.json"].get("quality_validated", False))
            for item in all_runs
        ),
    )
    return {
        **report.to_dict(),
        "runs": list(all_runs),
        "counterfactual_sensitivity": {
            "per_task_oracle": _per_task_sensitivity(per_task),
            "mixed_unconditioned": unconditioned["result_artifacts"][
                "reports/counterfactual_sensitivity.json"
            ],
            "mixed_task_onehot": conditioned["result_artifacts"][
                "reports/counterfactual_sensitivity.json"
            ],
        },
        "note": "M4 compares control learning, ambiguity, and oracle conditioning; it does not test language understanding.",
    }


def main() -> int:
    args = parse_args()
    try:
        report = compare(args)
        passed = True
    except Exception as error:  # noqa: BLE001 - CLI boundary preserves diagnostics
        traceback.print_exc()
        report = {
            "schema_version": M4_SCHEMA_VERSION,
            "error_type": type(error).__name__,
            "error_message": str(error) or repr(error),
        }
        passed = False
    payload = {**report, "passed": passed}
    atomic_write_json(args.output, payload)
    print(json.dumps({**payload, "output": str(args.output.resolve())}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
